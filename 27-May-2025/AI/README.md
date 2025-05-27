# Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution 

**Authors**: Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, Xing Zhou, Dongrui Liu, Ling Yang, Yue Wu, Kaixuan Huang, Shilong Liu, Hongru Wang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20286)  

**Abstract**: Recent advances in large language models (LLMs) have enabled agents to autonomously perform complex, open-ended tasks. However, many existing frameworks depend heavily on manually predefined tools and workflows, which hinder their adaptability, scalability, and generalization across domains. In this work, we introduce Alita--a generalist agent designed with the principle of "Simplicity is the ultimate sophistication," enabling scalable agentic reasoning through minimal predefinition and maximal self-evolution. For minimal predefinition, Alita is equipped with only one component for direct problem-solving, making it much simpler and neater than previous approaches that relied heavily on hand-crafted, elaborate tools and workflows. This clean design enhances its potential to generalize to challenging questions, without being limited by tools. For Maximal self-evolution, we enable the creativity of Alita by providing a suite of general-purpose components to autonomously construct, refine, and reuse external capabilities by generating task-related model context protocols (MCPs) from open source, which contributes to scalable agentic reasoning. Notably, Alita achieves 75.15% pass@1 and 87.27% pass@3 accuracy, which is top-ranking among general-purpose agents, on the GAIA benchmark validation dataset, 74.00% and 52.00% pass@1, respectively, on Mathvista and PathVQA, outperforming many agent systems with far greater complexity. More details will be updated at $\href{this https URL}{this https URL}$. 

---
# Ten Principles of AI Agent Economics 

**Authors**: Ke Yang, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.20273)  

**Abstract**: The rapid rise of AI-based autonomous agents is transforming human society and economic systems, as these entities increasingly exhibit human-like or superhuman intelligence. From excelling at complex games like Go to tackling diverse general-purpose tasks with large language and multimodal models, AI agents are evolving from specialized tools into dynamic participants in social and economic ecosystems. Their autonomy and decision-making capabilities are poised to impact industries, professions, and human lives profoundly, raising critical questions about their integration into economic activities, potential ethical concerns, and the balance between their utility and safety.
To address these challenges, this paper presents ten principles of AI agent economics, offering a framework to understand how AI agents make decisions, influence social interactions, and participate in the broader economy. Drawing on economics, decision theory, and ethics, we explore fundamental questions, such as whether AI agents might evolve from tools into independent entities, their impact on labor markets, and the ethical safeguards needed to align them with human values. These principles build on existing economic theories while accounting for the unique traits of AI agents, providing a roadmap for their responsible integration into human systems.
Beyond theoretical insights, this paper highlights the urgency of future research into AI trustworthiness, ethical guidelines, and regulatory oversight. As we enter a transformative era, this work serves as both a guide and a call to action, ensuring AI agents contribute positively to human progress while addressing risks tied to their unprecedented capabilities. 

---
# syftr: Pareto-Optimal Generative AI 

**Authors**: Alexander Conway, Debadeepta Dey, Stefan Hackmann, Matthew Hausknecht, Michael Schmidt, Mark Steadman, Nick Volynets  

**Link**: [PDF](https://arxiv.org/pdf/2505.20266)  

**Abstract**: Retrieval-Augmented Generation (RAG) pipelines are central to applying large language models (LLMs) to proprietary or dynamic data. However, building effective RAG flows is complex, requiring careful selection among vector databases, embedding models, text splitters, retrievers, and synthesizing LLMs. The challenge deepens with the rise of agentic paradigms. Modules like verifiers, rewriters, and rerankers-each with intricate hyperparameter dependencies have to be carefully tuned. Balancing tradeoffs between latency, accuracy, and cost becomes increasingly difficult in performance-sensitive applications.
We introduce syftr, a framework that performs efficient multi-objective search over a broad space of agentic and non-agentic RAG configurations. Using Bayesian Optimization, syftr discovers Pareto-optimal flows that jointly optimize task accuracy and cost. A novel early-stopping mechanism further improves efficiency by pruning clearly suboptimal candidates. Across multiple RAG benchmarks, syftr finds flows which are on average approximately 9 times cheaper while preserving most of the accuracy of the most accurate flows on the Pareto-frontier. Furthermore, syftr's ability to design and optimize allows integrating new modules, making it even easier and faster to realize high-performing generative AI pipelines. 

---
# On Path to Multimodal Historical Reasoning: HistBench and HistAgent 

**Authors**: Jiahao Qiu, Fulian Xiao, Yimin Wang, Yuchen Mao, Yijia Chen, Xinzhe Juan, Siran Wang, Xuan Qi, Tongcheng Zhang, Zixin Yao, Jiacheng Guo, Yifu Lu, Charles Argon, Jundi Cui, Daixin Chen, Junran Zhou, Shuyao Zhou, Zhanpeng Zhou, Ling Yang, Shilong Liu, Hongru Wang, Kaixuan Huang, Xun Jiang, Yuming Cao, Yue Chen, Yunfei Chen, Zhengyi Chen, Ruowei Dai, Mengqiu Deng, Jiye Fu, Yunting Gu, Zijie Guan, Zirui Huang, Xiaoyan Ji, Yumeng Jiang, Delong Kong, Haolong Li, Jiaqi Li, Ruipeng Li, Tianze Li, Zhuoran Li, Haixia Lian, Mengyue Lin, Xudong Liu, Jiayi Lu, Jinghan Lu, Wanyu Luo, Ziyue Luo, Zihao Pu, Zhi Qiao, Ruihuan Ren, Liang Wan, Ruixiang Wang, Tianhui Wang, Yang Wang, Zeyu Wang, Zihua Wang, Yujia Wu, Zhaoyi Wu, Hao Xin, Weiao Xing, Ruojun Xiong, Weijie Xu, Yao Shu, Xiao Yao, Xiaorui Yang, Yuchen Yang, Nan Yi, Jiadong Yu, Yangyuxuan Yu, Huiting Zeng, Danni Zhang, Yunjie Zhang, Zhaoyu Zhang, Zhiheng Zhang, Xiaofeng Zheng, Peirong Zhou, Linyan Zhong, Xiaoyin Zong, Ying Zhao, Zhenxin Chen, Lin Ding, Xiaoyu Gao, Bingbing Gong, Yichao Li, Yang Liao, Guang Ma, Tianyuan Ma, Xinrui Sun, Tianyi Wang, Han Xia, Ruobing Xian, Gen Ye, Tengfei Yu, Wentao Zhang, Yuxi Wang, Xi Gao, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20246)  

**Abstract**: Recent advances in large language models (LLMs) have led to remarkable progress across domains, yet their capabilities in the humanities, particularly history, remain underexplored. Historical reasoning poses unique challenges for AI, involving multimodal source interpretation, temporal inference, and cross-linguistic analysis. While general-purpose agents perform well on many existing benchmarks, they lack the domain-specific expertise required to engage with historical materials and questions. To address this gap, we introduce HistBench, a new benchmark of 414 high-quality questions designed to evaluate AI's capacity for historical reasoning and authored by more than 40 expert contributors. The tasks span a wide range of historical problems-from factual retrieval based on primary sources to interpretive analysis of manuscripts and images, to interdisciplinary challenges involving archaeology, linguistics, or cultural history. Furthermore, the benchmark dataset spans 29 ancient and modern languages and covers a wide range of historical periods and world regions. Finding the poor performance of LLMs and other agents on HistBench, we further present HistAgent, a history-specific agent equipped with carefully designed tools for OCR, translation, archival search, and image understanding in History. On HistBench, HistAgent based on GPT-4o achieves an accuracy of 27.54% pass@1 and 36.47% pass@2, significantly outperforming LLMs with online search and generalist agents, including GPT-4o (18.60%), DeepSeek-R1(14.49%) and Open Deep Research-smolagents(20.29% pass@1 and 25.12% pass@2). These results highlight the limitations of existing LLMs and generalist agents and demonstrate the advantages of HistAgent for historical reasoning. 

---
# The Mirage of Multimodality: Where Truth is Tested and Honesty Unravels 

**Authors**: Jiaming Ji, Sitong Fang, Wenjing Cao, Jiahao Li, Xuyao Wang, Juntao Dai, Chi-Min Chan, Sirui Han, Yike Guo, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20214)  

**Abstract**: Reasoning models have recently attracted significant attention, especially for tasks that involve complex inference. Their strengths exemplify the System II paradigm (slow, structured thinking), contrasting with the System I (rapid, heuristic-driven). Yet, does slower reasoning necessarily lead to greater truthfulness? Our findings suggest otherwise. In this study, we present the first systematic investigation of distortions associated with System I and System II reasoning in multimodal contexts. We demonstrate that slower reasoning models, when presented with incomplete or misleading visual inputs, are more likely to fabricate plausible yet false details to support flawed reasoning -- a phenomenon we term the "Mirage of Multimodality". To examine this, we constructed a 5,000-sample hierarchical prompt dataset annotated by 50 human participants. These prompts gradually increase in complexity, revealing a consistent pattern: slower reasoning models tend to employ depth-first thinking (delving deeper into incorrect premises), whereas faster chat models favor breadth-first inference, exhibiting greater caution under uncertainty. Our results highlight a critical vulnerability of slower reasoning models: although highly effective in structured domains such as mathematics, it becomes brittle when confronted with ambiguous multimodal inputs. 

---
# Shutdownable Agents through POST-Agency 

**Authors**: Elliott Thornley  

**Link**: [PDF](https://arxiv.org/pdf/2505.20203)  

**Abstract**: Many fear that future artificial agents will resist shutdown. I present an idea - the POST-Agents Proposal - for ensuring that doesn't happen. I propose that we train agents to satisfy Preferences Only Between Same-Length Trajectories (POST). I then prove that POST - together with other conditions - implies Neutrality+: the agent maximizes expected utility, ignoring the probability distribution over trajectory-lengths. I argue that Neutrality+ keeps agents shutdownable and allows them to be useful. 

---
# Temporal Sampling for Forgotten Reasoning in LLMs 

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, Xiang Yue, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.20196)  

**Abstract**: Fine-tuning large language models (LLMs) is intended to improve their reasoning capabilities, yet we uncover a counterintuitive effect: models often forget how to solve problems they previously answered correctly during training. We term this phenomenon temporal forgetting and show that it is widespread across model sizes, fine-tuning methods (both Reinforcement Learning and Supervised Fine-Tuning), and multiple reasoning benchmarks. To address this gap, we introduce Temporal Sampling, a simple decoding strategy that draws outputs from multiple checkpoints along the training trajectory. This approach recovers forgotten solutions without retraining or ensembling, and leads to substantial improvements in reasoning performance, gains from 4 to 19 points in Pass@k and consistent gains in Majority@k across several benchmarks. We further extend our method to LoRA-adapted models, demonstrating that storing only adapter weights across checkpoints achieves similar benefits with minimal storage cost. By leveraging the temporal diversity inherent in training, Temporal Sampling offers a practical, compute-efficient way to surface hidden reasoning ability and rethink how we evaluate LLMs. 

---
# An Empirical Study on Strong-Weak Model Collaboration for Repo-level Code Generation 

**Authors**: Shubham Gandhi, Atharva Naik, Yiqing Xie, Carolyn Rose  

**Link**: [PDF](https://arxiv.org/pdf/2505.20182)  

**Abstract**: We study cost-efficient collaboration between strong and weak language models for repository-level code generation, where the weak model handles simpler tasks at lower cost, and the most challenging tasks are delegated to the strong model. While many works propose architectures for this task, few analyze performance relative to cost. We evaluate a broad spectrum of collaboration strategies: context-based, pipeline-based, and dynamic, on GitHub issue resolution. Our most effective collaborative strategy achieves equivalent performance to the strong model while reducing the cost by 40%. Based on our findings, we offer actionable guidelines for choosing collaboration strategies under varying budget and performance constraints. Our results show that strong-weak collaboration substantially boosts the weak model's performance at a fraction of the cost, pipeline and context-based methods being most efficient. We release the code for our work at this https URL. 

---
# Program of Equations Thoughts to Solve Algebra Word Problems 

**Authors**: Yunze Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.20170)  

**Abstract**: Solving algebraic word problems (AWPs) has recently emerged as an important natural language processing task. Recently, large language models (LLMs) have demonstrated powerful mathematical capabilities, and the Chain-of-Thought technique, which guides LLMs through step-by-step reasoning, has yielded impressive results. However, this reasoning ability is limited by the computational weaknesses of LLMs themselves, where calculation errors can accumulate, leading to incorrect final answers. To address this, we propose Program of Equations Thoughts (POET), which transforms the task of generating step-by-step reasoning answers into a two-stage task of predicting equations and generating code, offloading complex computations to a Python interpreter to avoid calculation errors in LLMs. Furthermore, we propose Zero-shot POET, which utilizes a manually designed template to enable LLMs to directly generate Python code for one-step solving. Our method achieves accuracies of 95.3% and 98.0% on the PEN and ALG514 datasets, respectively, setting a new state-of-the-art (SOTA). Zero-shot POET also achieves the SOTA result of 95.5% on the DRAW-1K dataset. 

---
# Capability-Based Scaling Laws for LLM Red-Teaming 

**Authors**: Alexander Panfilov, Paul Kassianik, Maksym Andriushchenko, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2505.20162)  

**Abstract**: As large language models grow in capability and agency, identifying vulnerabilities through red-teaming becomes vital for safe deployment. However, traditional prompt-engineering approaches may prove ineffective once red-teaming turns into a weak-to-strong problem, where target models surpass red-teamers in capabilities. To study this shift, we frame red-teaming through the lens of the capability gap between attacker and target. We evaluate more than 500 attacker-target pairs using LLM-based jailbreak attacks that mimic human red-teamers across diverse families, sizes, and capability levels. Three strong trends emerge: (i) more capable models are better attackers, (ii) attack success drops sharply once the target's capability exceeds the attacker's, and (iii) attack success rates correlate with high performance on social science splits of the MMLU-Pro benchmark. From these trends, we derive a jailbreaking scaling law that predicts attack success for a fixed target based on attacker-target capability gap. These findings suggest that fixed-capability attackers (e.g., humans) may become ineffective against future models, increasingly capable open-source models amplify risks for existing systems, and model providers must accurately measure and control models' persuasive and manipulative abilities to limit their effectiveness as attackers. 

---
# MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents 

**Authors**: Ziming Wei, Bingqian Lin, Zijian Jiao, Yunshuang Nie, Liang Ma, Yuecheng Liu, Yuzheng Zhuang, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20148)  

**Abstract**: Spatial Planning is a crucial part in the field of spatial intelligence, which requires the understanding and planning about object arrangements in space perspective. AI agents with the spatial planning ability can better adapt to various real-world applications, including robotic manipulation, automatic assembly, urban planning etc. Recent works have attempted to construct benchmarks for evaluating the spatial intelligence of Multimodal Large Language Models (MLLMs). Nevertheless, these benchmarks primarily focus on spatial reasoning based on typical Visual Question-Answering (VQA) forms, which suffers from the gap between abstract spatial understanding and concrete task execution. In this work, we take a step further to build a comprehensive benchmark called MineAnyBuild, aiming to evaluate the spatial planning ability of open-world AI agents in the Minecraft game. Specifically, MineAnyBuild requires an agent to generate executable architecture building plans based on the given multi-modal human instructions. It involves 4,000 curated spatial planning tasks and also provides a paradigm for infinitely expandable data collection by utilizing rich player-generated content. MineAnyBuild evaluates spatial planning through four core supporting dimensions: spatial understanding, spatial reasoning, creativity, and spatial commonsense. Based on MineAnyBuild, we perform a comprehensive evaluation for existing MLLM-based agents, revealing the severe limitations but enormous potential in their spatial planning abilities. We believe our MineAnyBuild will open new avenues for the evaluation of spatial intelligence and help promote further development for open-world AI agents capable of spatial planning. 

---
# Agentic AI Process Observability: Discovering Behavioral Variability 

**Authors**: Fabiana Fournier, Lior Limonad, Yuval David  

**Link**: [PDF](https://arxiv.org/pdf/2505.20127)  

**Abstract**: AI agents that leverage Large Language Models (LLMs) are increasingly becoming core building blocks of modern software systems. A wide range of frameworks is now available to support the specification of such applications. These frameworks enable the definition of agent setups using natural language prompting, which specifies the roles, goals, and tools assigned to the various agents involved. Within such setups, agent behavior is non-deterministic for any given input, highlighting the critical need for robust debugging and observability tools. In this work, we explore the use of process and causal discovery applied to agent execution trajectories as a means of enhancing developer observability. This approach aids in monitoring and understanding the emergent variability in agent behavior. Additionally, we complement this with LLM-based static analysis techniques to distinguish between intended and unintended behavioral variability. We argue that such instrumentation is essential for giving developers greater control over evolving specifications and for identifying aspects of functionality that may require more precise and explicit definitions. 

---
# Agents Require Metacognitive and Strategic Reasoning to Succeed in the Coming Labor Markets 

**Authors**: Simpson Zhang, Tennison Liu, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2505.20120)  

**Abstract**: Current labor markets are strongly affected by the economic forces of adverse selection, moral hazard, and reputation, each of which arises due to $\textit{incomplete information}$. These economic forces will still be influential after AI agents are introduced, and thus, agents must use metacognitive and strategic reasoning to perform effectively. Metacognition is a form of $\textit{internal reasoning}$ that includes the capabilities for self-assessment, task understanding, and evaluation of strategies. Strategic reasoning is $\textit{external reasoning}$ that covers holding beliefs about other participants in the labor market (e.g., competitors, colleagues), making strategic decisions, and learning about others over time. Both types of reasoning are required by agents as they decide among the many $\textit{actions}$ they can take in labor markets, both within and outside their jobs. We discuss current research into metacognitive and strategic reasoning and the areas requiring further development. 

---
# Spatiotemporal Causal Decoupling Model for Air Quality Forecasting 

**Authors**: Jiaming Ma, Guanjun Wang, Sheng Huang, Kuo Yang, Binwu Wang, Pengkun Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20119)  

**Abstract**: Due to the profound impact of air pollution on human health, livelihoods, and economic development, air quality forecasting is of paramount significance. Initially, we employ the causal graph method to scrutinize the constraints of existing research in comprehensively modeling the causal relationships between the air quality index (AQI) and meteorological features. In order to enhance prediction accuracy, we introduce a novel air quality forecasting model, AirCade, which incorporates a causal decoupling approach. AirCade leverages a spatiotemporal module in conjunction with knowledge embedding techniques to capture the internal dynamics of AQI. Subsequently, a causal decoupling module is proposed to disentangle synchronous causality from past AQI and meteorological features, followed by the dissemination of acquired knowledge to future time steps to enhance performance. Additionally, we introduce a causal intervention mechanism to explicitly represent the uncertainty of future meteorological features, thereby bolstering the model's robustness. Our evaluation of AirCade on an open-source air quality dataset demonstrates over 20\% relative improvement over state-of-the-art models. 

---
# SwarmThinkers: Learning Physically Consistent Atomic KMC Transitions at Scale 

**Authors**: Qi Li, Kun Li, Haozhi Han, Honghui Shang, Xinfu He, Yunquan Zhang, Hong An, Ting Cao, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20094)  

**Abstract**: Can a scientific simulation system be physically consistent, interpretable by design, and scalable across regimes--all at once? Despite decades of progress, this trifecta remains elusive. Classical methods like Kinetic Monte Carlo ensure thermodynamic accuracy but scale poorly; learning-based methods offer efficiency but often sacrifice physical consistency and interpretability. We present SwarmThinkers, a reinforcement learning framework that recasts atomic-scale simulation as a physically grounded swarm intelligence system. Each diffusing particle is modeled as a local decision-making agent that selects transitions via a shared policy network trained under thermodynamic constraints. A reweighting mechanism fuses learned preferences with transition rates, preserving statistical fidelity while enabling interpretable, step-wise decision making. Training follows a centralized-training, decentralized-execution paradigm, allowing the policy to generalize across system sizes, concentrations, and temperatures without retraining. On a benchmark simulating radiation-induced Fe-Cu alloy precipitation, SwarmThinkers is the first system to achieve full-scale, physically consistent simulation on a single A100 GPU, previously attainable only via OpenKMC on a supercomputer. It delivers up to 4963x (3185x on average) faster computation with 485x lower memory usage. By treating particles as decision-makers, not passive samplers, SwarmThinkers marks a paradigm shift in scientific simulation--one that unifies physical consistency, interpretability, and scalability through agent-driven intelligence. 

---
# Safety Through Reasoning: An Empirical Study of Reasoning Guardrail Models 

**Authors**: Makesh Narsimhan Sreedhar, Traian Rebedea, Christopher Parisien  

**Link**: [PDF](https://arxiv.org/pdf/2505.20087)  

**Abstract**: Reasoning-based language models have demonstrated strong performance across various domains, with the most notable gains seen in mathematical and coding tasks. Recent research has shown that reasoning also offers significant benefits for LLM safety and guardrail applications. In this work, we conduct a comprehensive analysis of training reasoning-based guardrail models for content moderation, with an emphasis on generalization to custom safety policies at inference time. Our study focuses on two key dimensions: data efficiency and inference efficiency. On the data front, we find that reasoning-based models exhibit strong sample efficiency, achieving competitive performance with significantly fewer training examples than their non-reasoning counterparts. This unlocks the potential to repurpose the remaining data for mining high-value, difficult samples that further enhance model performance. On the inference side, we evaluate practical trade-offs by introducing reasoning budgets, examining the impact of reasoning length on latency and accuracy, and exploring dual-mode training to allow runtime control over reasoning behavior. Our findings will provide practical insights for researchers and developers to effectively and efficiently train and deploy reasoning-based guardrails models in real-world systems. 

---
# Curriculum-RLAIF: Curriculum Alignment with Reinforcement Learning from AI Feedback 

**Authors**: Mengdi Li, Jiaye Lin, Xufeng Zhao, Wenhao Lu, Peilin Zhao, Stefan Wermter, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20075)  

**Abstract**: Reward models trained with conventional Reinforcement Learning from AI Feedback (RLAIF) methods suffer from limited generalizability, which hinders the alignment performance of the policy model during reinforcement learning (RL). This challenge stems from various issues, including distribution shift, preference label noise, and mismatches between overly challenging samples and model capacity. In this paper, we attempt to enhance the generalizability of reward models through a data-centric approach, driven by the insight that these issues are inherently intertwined from the perspective of data difficulty. To address this, we propose a novel framework, $\textit{Curriculum-RLAIF}$, which constructs preference pairs with varying difficulty levels and produces a curriculum that progressively incorporates preference pairs of increasing difficulty for reward model training. Our experimental results suggest that reward models trained with Curriculum-RLAIF achieve improved generalizability, significantly increasing the alignment performance of the policy model by a large margin without incurring additional inference costs compared to various non-curriculum baselines. Detailed analysis and comparisons with alternative approaches, including data selection via external pretrained reward models or internal self-selection mechanisms, as well as other curriculum strategies, further demonstrate the superiority of our approach in terms of simplicity, efficiency, and effectiveness. 

---
# The Many Challenges of Human-Like Agents in Virtual Game Environments 

**Authors**: Maciej Świechowski, Dominik Ślęzak  

**Link**: [PDF](https://arxiv.org/pdf/2505.20011)  

**Abstract**: Human-like agents are an increasingly important topic in games and beyond. Believable non-player characters enhance the gaming experience by improving immersion and providing entertainment. They also offer players the opportunity to engage with AI entities that can function as opponents, teachers, or cooperating partners. Additionally, in games where bots are prohibited -- and even more so in non-game environments -- there is a need for methods capable of identifying whether digital interactions occur with bots or humans. This leads to two fundamental research questions: (1) how to model and implement human-like AI, and (2) how to measure its degree of human likeness.
This article offers two contributions. The first one is a survey of the most significant challenges in implementing human-like AI in games (or any virtual environment featuring simulated agents, although this article specifically focuses on games). Thirteen such challenges, both conceptual and technical, are discussed in detail. The second is an empirical study performed in a tactical video game that addresses the research question: "Is it possible to distinguish human players from bots (AI agents) based on empirical data?" A machine-learning approach using a custom deep recurrent convolutional neural network is presented. We hypothesize that the more challenging it is to create human-like AI for a given game, the easier it becomes to develop a method for distinguishing humans from AI-driven players. 

---
# Adaptive Location Hierarchy Learning for Long-Tailed Mobility Prediction 

**Authors**: Yu Wang, Junshu Dai, Yuchen Ying, Yuxuan Liang, Tongya Zheng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.19965)  

**Abstract**: Human mobility prediction is crucial for applications ranging from location-based recommendations to urban planning, which aims to forecast users' next location visits based on historical trajectories. Despite the severe long-tailed distribution of locations, the problem of long-tailed mobility prediction remains largely underexplored. Existing long-tailed learning methods primarily focus on rebalancing the skewed distribution at the data, model, or class level, neglecting to exploit the spatiotemporal semantics of locations. To address this gap, we propose the first plug-and-play framework for long-tailed mobility prediction in an exploitation and exploration manner, named \textbf{A}daptive \textbf{LO}cation \textbf{H}ier\textbf{A}rchy learning (ALOHA). First, we construct city-tailored location hierarchy based on Large Language Models (LLMs) by exploiting Maslow's theory of human motivation to design Chain-of-Thought (CoT) prompts that captures spatiotemporal semantics. Second, we optimize the location hierarchy predictions by Gumbel disturbance and node-wise adaptive weights within the hierarchical tree structure. Experiments on state-of-the-art models across six datasets demonstrate the framework's consistent effectiveness and generalizability, which strikes a well balance between head and tail locations. Weight analysis and ablation studies reveal the optimization differences of each component for head and tail locations. Furthermore, in-depth analyses of hierarchical distance and case study demonstrate the effective semantic guidance from the location hierarchy. Our code will be made publicly available. 

---
# DCG-SQL: Enhancing In-Context Learning for Text-to-SQL with Deep Contextual Schema Link Graph 

**Authors**: Jihyung Lee, Jin-Seop Lee, Jaehoon Lee, YunSeok Choi, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.19956)  

**Abstract**: Text-to-SQL, which translates a natural language question into an SQL query, has advanced with in-context learning of Large Language Models (LLMs). However, existing methods show little improvement in performance compared to randomly chosen demonstrations, and significant performance drops when smaller LLMs (e.g., Llama 3.1-8B) are used. This indicates that these methods heavily rely on the intrinsic capabilities of hyper-scaled LLMs, rather than effectively retrieving useful demonstrations. In this paper, we propose a novel approach for effectively retrieving demonstrations and generating SQL queries. We construct a Deep Contextual Schema Link Graph, which contains key information and semantic relationship between a question and its database schema items. This graph-based structure enables effective representation of Text-to-SQL samples and retrieval of useful demonstrations for in-context learning. Experimental results on the Spider benchmark demonstrate the effectiveness of our approach, showing consistent improvements in SQL generation performance and efficiency across both hyper-scaled LLMs and small LLMs. Our code will be released. 

---
# Subtle Risks, Critical Failures: A Framework for Diagnosing Physical Safety of LLMs for Embodied Decision Making 

**Authors**: Yejin Son, Minseo Kim, Sungwoong Kim, Seungju Han, Jian Kim, Dongju Jang, Youngjae Yu, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.19933)  

**Abstract**: Large Language Models (LLMs) are increasingly used for decision making in embodied agents, yet existing safety evaluations often rely on coarse success rates and domain-specific setups, making it difficult to diagnose why and where these models fail. This obscures our understanding of embodied safety and limits the selective deployment of LLMs in high-risk physical environments. We introduce SAFEL, the framework for systematically evaluating the physical safety of LLMs in embodied decision making. SAFEL assesses two key competencies: (1) rejecting unsafe commands via the Command Refusal Test, and (2) generating safe and executable plans via the Plan Safety Test. Critically, the latter is decomposed into functional modules, goal interpretation, transition modeling, action sequencing, enabling fine-grained diagnosis of safety failures. To support this framework, we introduce EMBODYGUARD, a PDDL-grounded benchmark containing 942 LLM-generated scenarios covering both overtly malicious and contextually hazardous instructions. Evaluation across 13 state-of-the-art LLMs reveals that while models often reject clearly unsafe commands, they struggle to anticipate and mitigate subtle, situational risks. Our results highlight critical limitations in current LLMs and provide a foundation for more targeted, modular improvements in safe embodied reasoning. 

---
# TCP: a Benchmark for Temporal Constraint-Based Planning 

**Authors**: Zifeng Ding, Sikuan Yan, Zhangdie Yuan, Xianglong Hu, Fangru Lin, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2505.19927)  

**Abstract**: Temporal reasoning and planning are essential capabilities for large language models (LLMs), yet most existing benchmarks evaluate them in isolation and under limited forms of complexity. To address this gap, we introduce the Temporal Constraint-based Planning (TCP) benchmark, that jointly assesses both capabilities. Each instance in TCP features a naturalistic dialogue around a collaborative project, where diverse and interdependent temporal constraints are explicitly or implicitly expressed, and models must infer an optimal schedule that satisfies all constraints. To construct TCP, we first generate abstract problem prototypes that are paired with realistic scenarios from various domains and enriched into dialogues using an LLM. A human quality check is performed on a sampled subset to confirm the reliability of our benchmark. We evaluate state-of-the-art LLMs and find that even the strongest models struggle with TCP, highlighting its difficulty and revealing limitations in LLMs' temporal constraint-based planning abilities. We analyze underlying failure cases, open source our benchmark, and hope our findings can inspire future research. 

---
# EMAC+: Embodied Multimodal Agent for Collaborative Planning with VLM+LLM 

**Authors**: Shuang Ao, Flora D. Salim, Simon Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19905)  

**Abstract**: Although LLMs demonstrate proficiency in several text-based reasoning and planning tasks, their implementation in robotics control is constrained by significant deficiencies: (1) LLM agents are designed to work mainly with textual inputs rather than visual conditions; (2) Current multimodal agents treat LLMs as static planners, which separates their reasoning from environment dynamics, resulting in actions that do not take domain-specific knowledge into account; and (3) LLMs are not designed to learn from visual interactions, which makes it harder for them to make better policies for specific domains. In this paper, we introduce EMAC+, an Embodied Multimodal Agent that collaboratively integrates LLM and VLM via a bidirectional training paradigm. Unlike existing methods, EMAC+ dynamically refines high-level textual plans generated by an LLM using real-time feedback from a VLM executing low-level visual control tasks. We address critical limitations of previous models by enabling the LLM to internalize visual environment dynamics directly through interactive experience, rather than relying solely on static symbolic mappings. Extensive experimental evaluations on ALFWorld and RT-1 benchmarks demonstrate that EMAC+ achieves superior task performance, robustness against noisy observations, and efficient learning. We also conduct thorough ablation studies and provide detailed analyses of success and failure cases. 

---
# ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows 

**Authors**: Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, Jianing Wang, Qintong Li, Xiangru Tang, Tianbao Xie, Xiachong Feng, Xiang Li, Ben Kao, Wenhai Wang, Biqing Qi, Lingpeng Kong, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19897)  

**Abstract**: Large Language Models (LLMs) have extended their impact beyond Natural Language Processing, substantially fostering the development of interdisciplinary research. Recently, various LLM-based agents have been developed to assist scientific discovery progress across multiple aspects and domains. Among these, computer-using agents, capable of interacting with operating systems as humans do, are paving the way to automated scientific problem-solving and addressing routines in researchers' workflows. Recognizing the transformative potential of these agents, we introduce ScienceBoard, which encompasses two complementary contributions: (i) a realistic, multi-domain environment featuring dynamic and visually rich scientific workflows with integrated professional software, where agents can autonomously interact via different interfaces to accelerate complex research tasks and experiments; and (ii) a challenging benchmark of 169 high-quality, rigorously validated real-world tasks curated by humans, spanning scientific-discovery workflows in domains such as biochemistry, astronomy, and geoinformatics. Extensive evaluations of agents with state-of-the-art backbones (e.g., GPT-4o, Claude 3.7, UI-TARS) show that, despite some promising results, they still fall short of reliably assisting scientists in complex workflows, achieving only a 15% overall success rate. In-depth analysis further provides valuable insights for addressing current agent limitations and more effective design principles, paving the way to build more capable agents for scientific discovery. Our code, environment, and benchmark are at this https URL. 

---
# Large Language Models as Autonomous Spacecraft Operators in Kerbal Space Program 

**Authors**: Alejandro Carrasco, Victor Rodriguez-Fernandez, Richard Linares  

**Link**: [PDF](https://arxiv.org/pdf/2505.19896)  

**Abstract**: Recent trends are emerging in the use of Large Language Models (LLMs) as autonomous agents that take actions based on the content of the user text prompts. We intend to apply these concepts to the field of Control in space, enabling LLMs to play a significant role in the decision-making process for autonomous satellite operations. As a first step towards this goal, we have developed a pure LLM-based solution for the Kerbal Space Program Differential Games (KSPDG) challenge, a public software design competition where participants create autonomous agents for maneuvering satellites involved in non-cooperative space operations, running on the KSP game engine. Our approach leverages prompt engineering, few-shot prompting, and fine-tuning techniques to create an effective LLM-based agent that ranked 2nd in the competition. To the best of our knowledge, this work pioneers the integration of LLM agents into space research. The project comprises several open repositories to facilitate replication and further research. The codebase is accessible on \href{this https URL}{GitHub}, while the trained models and datasets are available on \href{this https URL}{Hugging Face}. Additionally, experiment tracking and detailed results can be reviewed on \href{this https URL}{Weights \& Biases 

---
# Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging 

**Authors**: Yongxian Wei, Runxi Cheng, Weike Jin, Enneng Yang, Li Shen, Lu Hou, Sinan Du, Chun Yuan, Xiaochun Cao, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19892)  

**Abstract**: While foundation models update slowly due to resource-intensive training requirements, domain-specific models evolve between updates. Model merging aims to combine multiple expert models into a single, more capable model, thereby reducing storage and serving costs while supporting decentralized model development. Despite its potential, previous studies have primarily focused on merging visual classification models or Large Language Models (LLMs) for code and math tasks. Multimodal Large Language Models (MLLMs), which extend the capabilities of LLMs through large-scale multimodal training, have gained traction. However, there lacks a benchmark for model merging research that clearly divides the tasks for MLLM training and evaluation. In this paper, (i) we introduce the model merging benchmark for MLLMs, which includes multiple tasks such as VQA, Geometry, Chart, OCR, and Grounding, providing both LoRA and full fine-tuning models. Moreover, we explore how model merging can combine different modalities (e.g., vision-language, audio-language, and video-language models), moving toward the Omni-language model. (ii) We implement 10 model merging algorithms on the benchmark. Furthermore, we propose a novel method that removes noise from task vectors and robustly optimizes the merged vector based on a loss defined over task vector interactions, achieving an average performance gain of 2.48%. (iii) We find that model merging offers a promising way for building improved MLLMs without requiring data training. Our results also demonstrate that the complementarity among multiple modalities outperforms individual modalities. 

---
# HS-STAR: Hierarchical Sampling for Self-Taught Reasoners via Difficulty Estimation and Budget Reallocation 

**Authors**: Feng Xiong, Hongling Xu, Yifei Wang, Runxi Cheng, Yong Wang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19866)  

**Abstract**: Self-taught reasoners (STaRs) enhance the mathematical reasoning abilities of large language models (LLMs) by leveraging self-generated responses for self-training. Recent studies have incorporated reward models to guide response selection or decoding, aiming to obtain higher-quality data. However, they typically allocate a uniform sampling budget across all problems, overlooking the varying utility of problems at different difficulty levels. In this work, we conduct an empirical study and find that problems near the boundary of the LLM's reasoning capability offer significantly greater learning utility than both easy and overly difficult ones. To identify and exploit such problems, we propose HS-STaR, a Hierarchical Sampling framework for Self-Taught Reasoners. Given a fixed sampling budget, HS-STaR first performs lightweight pre-sampling with a reward-guided difficulty estimation strategy to efficiently identify boundary-level problems. Subsequently, it dynamically reallocates the remaining budget toward these high-utility problems during a re-sampling phase, maximizing the generation of valuable training data. Extensive experiments across multiple reasoning benchmarks and backbone LLMs demonstrate that HS-STaR significantly outperforms other baselines without requiring additional sampling budget. 

---
# DGRAG: Distributed Graph-based Retrieval-Augmented Generation in Edge-Cloud Systems 

**Authors**: Wenqing Zhou, Yuxuan Yan, Qianqian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19847)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the capabilities of language models by integrating external knowledge. Due to the diversity of data sources and the constraints of memory and computing resources, real-world data is often scattered in multiple devices. Conventional RAGs that store massive amounts of scattered data centrally face increasing privacy concerns and high computational costs. Additionally, RAG in a central node raises latency issues when searching over a large-scale knowledge base. To address these challenges, we propose a distributed Knowledge Graph-based RAG approach, referred to as DGRAG, in an edge-cloud system, where each edge device maintains a local knowledge base without the need to share it with the cloud, instead sharing only summaries of its knowledge. Specifically, DGRAG has two main phases. In the Distributed Knowledge Construction phase, DGRAG organizes local knowledge using knowledge graphs, generating subgraph summaries and storing them in a summary database in the cloud as information sharing. In the Collaborative Retrieval and Generation phase, DGRAG first performs knowledge retrieval and answer generation locally, and a gate mechanism determines whether the query is beyond the scope of local knowledge or processing capabilities. For queries that exceed the local knowledge scope, the cloud retrieves knowledge from the most relevant edges based on the summaries and generates a more precise answer. Experimental results demonstrate the effectiveness of the proposed DGRAG approach in significantly improving the quality of question-answering tasks over baseline approaches. 

---
# Types of Relations: Defining Analogies with Category Theory 

**Authors**: Claire Ott, Frank Jäkel  

**Link**: [PDF](https://arxiv.org/pdf/2505.19792)  

**Abstract**: In order to behave intelligently both humans and machines have to represent their knowledge adequately for how it is used. Humans often use analogies to transfer their knowledge to new domains, or help others with this transfer via explanations. Hence, an important question is: What representation can be used to construct, find, and evaluate analogies? In this paper, we study features of a domain that are important for constructing analogies. We do so by formalizing knowledge domains as categories. We use the well-known example of the analogy between the solar system and the hydrogen atom to demonstrate how to construct domain categories. We also show how functors, pullbacks, and pushouts can be used to define an analogy, describe its core and a corresponding blend of the underlying domains. 

---
# Done Is Better than Perfect: Unlocking Efficient Reasoning by Structured Multi-Turn Decomposition 

**Authors**: Zihao Zeng, Xuyao Huang, Boxiu Li, Hao Zhang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19788)  

**Abstract**: Large Reasoning Models (LRMs) are criticized for the excessively lengthy Chain-of-Thought (CoT) to derive the final answer, suffering from high first-token and overall latency. Typically, the CoT of LRMs mixes multiple thinking units; each unit attempts to produce a candidate answer to the original query. Hence, a natural idea to improve efficiency is to reduce the unit number. Yet, the fact that the thinking units in vanilla CoT cannot be explicitly managed renders doing so challenging. This paper introduces Multi-Turn Decomposition (MinD) to decode conventional CoT into a sequence of explicit, structured, and turn-wise interactions to bridge the gap. In MinD, the model provides a multi-turn response to the query, where each turn embraces a thinking unit and yields a corresponding answer. The subsequent turns can reflect, verify, revise, or explore alternative approaches to both the thinking and answer parts of earlier ones. This not only makes the answer delivered more swiftly, but also enables explicit controls over the iterative reasoning process (i.e., users may halt or continue at any turn). We follow a supervised fine-tuning (SFT) then reinforcement learning (RL) paradigm to realize MinD. We first rephrase the outputs of an LRM into multi-turn formats by prompting another LLM, and then tune the LRM with such data. Observing that the tuned model tends to consume even more tokens than the original one (probably due to that the multi-turn formats introduce additional answer tokens), we advocate leveraging RL algorithms like GRPO to prioritize correct outputs with fewer turns. Trained on the MATH dataset using R1-Distill models, MinD can achieve up to ~70% reduction in both output token usage and time to first token (TTFT), while maintaining competitive performance on reasoning benchmarks such as MATH-500, AIME24, AMC23, and GPQA-Diamond. 

---
# Language Model-Enhanced Message Passing for Heterophilic Graph Learning 

**Authors**: Wenjun Wang, Dawei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19762)  

**Abstract**: Traditional graph neural networks (GNNs), which rely on homophily-driven message passing, struggle with heterophilic graphs where connected nodes exhibit dissimilar features and different labels. While existing methods address heterophily through graph structure refinement or adaptation of neighbor aggregation functions, they often overlook the semantic potential of node text, rely on suboptimal message representation for propagation and compromise performance on homophilic graphs. To address these limitations, we propose a novel language model (LM)-enhanced message passing approach for heterophilic graph leaning (LEMP4HG). Specifically, in the context of text-attributed graph, we provide paired node texts for LM to generate their connection analysis, which are encoded and then fused with paired node textual embeddings through a gating mechanism. The synthesized messages are semantically enriched and adaptively balanced with both nodes' information, which mitigates contradictory signals when neighbor aggregation in heterophilic regions. Furthermore, we introduce an active learning strategy guided by our heuristic MVRD (Modulated Variation of Reliable Distance), selectively enhancing node pairs suffer most from message passing, reducing the cost of analysis generation and side effects on homophilic regions. Extensive experiments validate that our approach excels on heterophilic graphs and performs robustly on homophilic ones, with a graph convolutional network (GCN) backbone and a practical budget. 

---
# Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning 

**Authors**: Zican Hu, Wei Liu, Xiaoye Qu, Xiangyu Yue, Chunlin Chen, Zhi Wang, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19761)  

**Abstract**: While showing sophisticated reasoning abilities, large language models (LLMs) still struggle with long-horizon decision-making tasks due to deficient exploration and long-term credit assignment, especially in sparse-reward scenarios. Inspired by the divide-and-conquer principle, we propose an innovative framework **GLIDER** (**G**rounding **L**anguage Models as Eff**I**cient **D**ecision-Making Agents via Offline Hi**E**rarchical **R**einforcement Learning) that introduces a parameter-efficient and generally applicable hierarchy to LLM policies. We develop a scheme where the low-level controller is supervised with abstract, step-by-step plans that are learned and instructed by the high-level policy. This design decomposes complicated problems into a series of coherent chain-of-thought reasoning sub-tasks, providing flexible temporal abstraction to significantly enhance exploration and learning for long-horizon tasks. Furthermore, GLIDER facilitates fast online adaptation to non-stationary environments owing to the strong transferability of its task-agnostic low-level skills. Experiments on ScienceWorld and ALFWorld benchmarks show that GLIDER achieves consistent performance gains, along with enhanced generalization capabilities. 

---
# ReChisel: Effective Automatic Chisel Code Generation by LLM with Reflection 

**Authors**: Juxin Niu, Xiangfeng Liu, Dan Niu, Xi Wang, Zhe Jiang, Nan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19734)  

**Abstract**: Coding with hardware description languages (HDLs) such as Verilog is a time-intensive and laborious task. With the rapid advancement of large language models (LLMs), there is increasing interest in applying LLMs to assist with HDL coding. Recent efforts have demonstrated the potential of LLMs in translating natural language to traditional HDL Verilog. Chisel, a next-generation HDL based on Scala, introduces higher-level abstractions, facilitating more concise, maintainable, and scalable hardware designs. However, the potential of using LLMs for Chisel code generation remains largely unexplored. This work proposes ReChisel, an LLM-based agentic system designed to enhance the effectiveness of Chisel code generation. ReChisel incorporates a reflection mechanism to iteratively refine the quality of generated code using feedback from compilation and simulation processes, and introduces an escape mechanism to break free from non-progress loops. Experiments demonstrate that ReChisel significantly improves the success rate of Chisel code generation, achieving performance comparable to state-of-the-art LLM-based agentic systems for Verilog code generation. 

---
# Concise Reasoning, Big Gains: Pruning Long Reasoning Trace with Difficulty-Aware Prompting 

**Authors**: Yifan Wu, Jingze Shi, Bingheng Wu, Jiayi Zhang, Xiaotian Lin, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.19716)  

**Abstract**: Existing chain-of-thought (CoT) distillation methods can effectively transfer reasoning abilities to base models but suffer from two major limitations: excessive verbosity of reasoning traces and inadequate adaptability to problem difficulty. Long reasoning traces significantly increase inference costs, and uniform-length solutions prevent base models from learning adaptive reasoning strategies. To address these issues, we propose a difficulty-aware prompting (DAP) method to dynamically shorten reasoning traces without performance loss. In our approach, a large teacher model first judges each problem's difficulty and then rewrites its reasoning traces to an appropriate shorter length, yielding concise yet complete reasoning traces. Leveraging the DAP pipeline, we curate a distilled dataset called LiteCoT consisting of 100K concise reasoning examples, with solutions averaging only 720 tokens (an order of magnitude shorter than typical CoTs). Using LiteCoT, we distilled a new family of reasoning models called Liter (1.5B, 7B, and 32B) based on the Qwen2.5 architecture. Experiments show that a student model fine-tuned on just 100K of these difficulty-pruned CoT samples outperforms a model distilled on 800K original Long CoT samples, while significantly reducing training and inference costs. Our method also generalizes well: across 11 diverse benchmarks, the shorter difficulty-aware CoTs achieve equal or better accuracy than Long chains, using far fewer tokens. For example, on the challenging AIME24 exam, our approach reaches $74.2\%$ Pass@1 using only about 5K inference tokens, surpassing other methods that consume many more tokens. Our code and data are available at this https URL. 

---
# Beyond Safe Answers: A Benchmark for Evaluating True Risk Awareness in Large Reasoning Models 

**Authors**: Baihui Zheng, Boren Zheng, Kerui Cao, Yingshui Tan, Zhendong Liu, Weixun Wang, Jiaheng Liu, Jian Yang, Wenbo Su, Xiaoyong Zhu, Bo Zheng, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19690)  

**Abstract**: Despite the remarkable proficiency of \textit{Large Reasoning Models} (LRMs) in handling complex reasoning tasks, their reliability in safety-critical scenarios remains uncertain. Existing evaluations primarily assess response-level safety, neglecting a critical issue we identify as \textbf{\textit{Superficial Safety Alignment} (SSA)} -- a phenomenon where models produce superficially safe outputs while internal reasoning processes fail to genuinely detect and mitigate underlying risks, resulting in inconsistent safety behaviors across multiple sampling attempts. To systematically investigate SSA, we introduce \textbf{Beyond Safe Answers (BSA)} bench, a novel benchmark comprising 2,000 challenging instances organized into three distinct SSA scenario types and spanning nine risk categories, each meticulously annotated with risk rationales. Evaluations of 19 state-of-the-art LRMs demonstrate the difficulty of this benchmark, with top-performing models achieving only 38.0\% accuracy in correctly identifying risk rationales. We further explore the efficacy of safety rules, specialized fine-tuning on safety reasoning data, and diverse decoding strategies in mitigating SSA. Our work provides a comprehensive assessment tool for evaluating and improving safety reasoning fidelity in LRMs, advancing the development of genuinely risk-aware and reliably safe AI systems. 

---
# Large Language Models for Planning: A Comprehensive and Systematic Survey 

**Authors**: Pengfei Cao, Tianyi Men, Wencan Liu, Jingwen Zhang, Xuzhao Li, Xixun Lin, Dianbo Sui, Yanan Cao, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19683)  

**Abstract**: Planning represents a fundamental capability of intelligent agents, requiring comprehensive environmental understanding, rigorous logical reasoning, and effective sequential decision-making. While Large Language Models (LLMs) have demonstrated remarkable performance on certain planning tasks, their broader application in this domain warrants systematic investigation. This paper presents a comprehensive review of LLM-based planning. Specifically, this survey is structured as follows: First, we establish the theoretical foundations by introducing essential definitions and categories about automated planning. Next, we provide a detailed taxonomy and analysis of contemporary LLM-based planning methodologies, categorizing them into three principal approaches: 1) External Module Augmented Methods that combine LLMs with additional components for planning, 2) Finetuning-based Methods that involve using trajectory data and feedback signals to adjust LLMs in order to improve their planning abilities, and 3) Searching-based Methods that break down complex tasks into simpler components, navigate the planning space, or enhance decoding strategies to find the best solutions. Subsequently, we systematically summarize existing evaluation frameworks, including benchmark datasets, evaluation metrics and performance comparisons between representative planning methods. Finally, we discuss the underlying mechanisms enabling LLM-based planning and outline promising research directions for this rapidly evolving field. We hope this survey will serve as a valuable resource to inspire innovation and drive progress in this field. 

---
# Large Language Models' Reasoning Stalls: An Investigation into the Capabilities of Frontier Models 

**Authors**: Lachlan McGinness, Peter Baumgartner  

**Link**: [PDF](https://arxiv.org/pdf/2505.19676)  

**Abstract**: Empirical methods to examine the capability of Large Language Models (LLMs) to use Automated Theorem Prover (ATP) reasoning strategies are studied. We evaluate the performance of State of the Art models from December 2023 and August 2024 on PRONTOQA steamroller reasoning problems. For that, we develop methods for assessing LLM response accuracy and correct answer correlation.
Our results show that progress in improving LLM reasoning abilities has stalled over the nine month period. By tracking completion tokens, we show that almost all improvement in reasoning ability since GPT-4 was released can be attributed to either hidden system prompts or the training of models to automatically use generic Chain of Thought prompting strategies. Among the ATP reasoning strategies tried, we found that current frontier LLMs are best able to follow the bottom-up (also known as forward-chaining) strategy. A low positive correlation was found between an LLM response containing correct reasoning and arriving at the correct conclusion. 

---
# FieldWorkArena: Agentic AI Benchmark for Real Field Work Tasks 

**Authors**: Atsunori Moteki, Shoichi Masui, Fan Yang, Yueqi Song, Yonatan Bisk, Graham Neubig, Ikuo Kusajima, Yasuto Watanabe, Hiroyuki Ishida, Jun Takahashi, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19662)  

**Abstract**: This paper proposes FieldWorkArena, a benchmark for agentic AI targeting real-world field work. With the recent increase in demand for agentic AI, they are required to monitor and report safety and health incidents, as well as manufacturing-related incidents, that may occur in real-world work environments. Existing agentic AI benchmarks have been limited to evaluating web tasks and are insufficient for evaluating agents in real-world work environments, where complexity increases significantly. In this paper, we define a new action space that agentic AI should possess for real world work environment benchmarks and improve the evaluation function from previous methods to assess the performance of agentic AI in diverse real-world tasks. The dataset consists of videos captured on-site and documents actually used in factories and warehouses, and tasks were created based on interviews with on-site workers and managers. Evaluation results confirmed that performance evaluation considering the characteristics of Multimodal LLM (MLLM) such as GPT-4o is feasible. Additionally, the effectiveness and limitations of the proposed new evaluation method were identified. The complete dataset (HuggingFace) and evaluation program (GitHub) can be downloaded from the following website: this https URL. 

---
# Token-Importance Guided Direct Preference Optimization 

**Authors**: Yang Ning, Lin Hai, Liu Yibo, Tian Baoliang, Liu Guoqing, Zhang Haijun  

**Link**: [PDF](https://arxiv.org/pdf/2505.19653)  

**Abstract**: Ensuring that large language models (LLMs) generate outputs aligned with human preferences is important for safe and effective AI interactions. While Direct Preference Optimization (DPO) employs an implicit reward function to optimize the policy model, however, it and its related variants overlook the differential importance of individual tokens and are sensitive to judgment noise in preference datasets during generation. Although recent methods attempt to assess the important weight of tokens via probability prediction or simplistic weighting schemes, these evaluation methods are prone to biases and still cannot fully address these issues. To solve this problem, we propose the Token-Importance Guided Direct Preference Optimization (TI-DPO), which introduces two key innovations: the gradient-based token-importance weights that dynamically prioritize critical tokens, and a triple loss that explicitly guides model outputs to approach human-preferred responses and stay away from non-preferred responses. Experimental results show that TI-DPO achieves higher accuracy and stronger generative diversity, providing more stable and computationally efficient solutions compared with DPO and other RLHF methods. 

---
# SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond 

**Authors**: Junteng Liu, Yuanxiang Fan, Zhuo Jiang, Han Ding, Yongyi Hu, Chi Zhang, Yiqi Shi, Shitong Weng, Aili Chen, Shiqi Chen, Yunan Huang, Mozhi Zhang, Pengyu Zhao, Junjie Yan, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2505.19641)  

**Abstract**: Recent advances such as OpenAI-o1 and DeepSeek R1 have demonstrated the potential of Reinforcement Learning (RL) to enhance reasoning abilities in Large Language Models (LLMs). While open-source replication efforts have primarily focused on mathematical and coding domains, methods and resources for developing general reasoning capabilities remain underexplored. This gap is partly due to the challenge of collecting diverse and verifiable reasoning data suitable for RL. We hypothesize that logical reasoning is critical for developing general reasoning capabilities, as logic forms a fundamental building block of reasoning. In this work, we present SynLogic, a data synthesis framework and dataset that generates diverse logical reasoning data at scale, encompassing 35 diverse logical reasoning tasks. The SynLogic approach enables controlled synthesis of data with adjustable difficulty and quantity. Importantly, all examples can be verified by simple rules, making them ideally suited for RL with verifiable rewards. In our experiments, we validate the effectiveness of RL training on the SynLogic dataset based on 7B and 32B models. SynLogic leads to state-of-the-art logical reasoning performance among open-source datasets, surpassing DeepSeek-R1-Distill-Qwen-32B by 6 points on BBEH. Furthermore, mixing SynLogic data with mathematical and coding tasks improves the training efficiency of these domains and significantly enhances reasoning generalization. Notably, our mixed training model outperforms DeepSeek-R1-Zero-Qwen-32B across multiple benchmarks. These findings position SynLogic as a valuable resource for advancing the broader reasoning capabilities of LLMs. We open-source both the data synthesis pipeline and the SynLogic dataset at this https URL. 

---
# Think Again! The Effect of Test-Time Compute on Preferences, Opinions, and Beliefs of Large Language Models 

**Authors**: George Kour, Itay Nakash, Ateret Anaby-Tavor, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2505.19621)  

**Abstract**: As Large Language Models (LLMs) become deeply integrated into human life and increasingly influence decision-making, it's crucial to evaluate whether and to what extent they exhibit subjective preferences, opinions, and beliefs. These tendencies may stem from biases within the models, which may shape their behavior, influence the advice and recommendations they offer to users, and potentially reinforce certain viewpoints. This paper presents the Preference, Opinion, and Belief survey (POBs), a benchmark developed to assess LLMs' subjective inclinations across societal, cultural, ethical, and personal domains. We applied our benchmark to evaluate leading open- and closed-source LLMs, measuring desired properties such as reliability, neutrality, and consistency. In addition, we investigated the effect of increasing the test-time compute, through reasoning and self-reflection mechanisms, on those metrics. While effective in other tasks, our results show that these mechanisms offer only limited gains in our domain. Furthermore, we reveal that newer model versions are becoming less consistent and more biased toward specific viewpoints, highlighting a blind spot and a concerning trend. POBS: this https URL 

---
# MSD-LLM: Predicting Ship Detention in Port State Control Inspections with Large Language Model 

**Authors**: Jiongchao Jin, Xiuju Fu, Xiaowei Gao, Tao Cheng, Ran Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19568)  

**Abstract**: Maritime transportation is the backbone of global trade, making ship inspection essential for ensuring maritime safety and environmental protection. Port State Control (PSC), conducted by national ports, enforces compliance with safety regulations, with ship detention being the most severe consequence, impacting both ship schedules and company reputations. Traditional machine learning methods for ship detention prediction are limited by the capacity of representation learning and thus suffer from low accuracy. Meanwhile, autoencoder-based deep learning approaches face challenges due to the severe data imbalance in learning historical PSC detention records. To address these limitations, we propose Maritime Ship Detention with Large Language Models (MSD-LLM), integrating a dual robust subspace recovery (DSR) layer-based autoencoder with a progressive learning pipeline to handle imbalanced data and extract meaningful PSC representations. Then, a large language model groups and ranks features to identify likely detention cases, enabling dynamic thresholding for flexible detention predictions. Extensive evaluations on 31,707 PSC inspection records from the Asia-Pacific region show that MSD-LLM outperforms state-of-the-art methods more than 12\% on Area Under the Curve (AUC) for Singapore ports. Additionally, it demonstrates robustness to real-world challenges, making it adaptable to diverse maritime risk assessment scenarios. 

---
# LLM-Agent-Controller: A Universal Multi-Agent Large Language Model System as a Control Engineer 

**Authors**: Rasoul Zahedifar, Sayyed Ali Mirghasemi, Mahdieh Soleymani Baghshah, Alireza Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2505.19567)  

**Abstract**: This study presents the LLM-Agent-Controller, a multi-agent large language model (LLM) system developed to address a wide range of problems in control engineering (Control Theory). The system integrates a central controller agent with multiple specialized auxiliary agents, responsible for tasks such as controller design, model representation, control analysis, time-domain response, and simulation. A supervisor oversees high-level decision-making and workflow coordination, enhancing the system's reliability and efficiency. The LLM-Agent-Controller incorporates advanced capabilities, including Retrieval-Augmented Generation (RAG), Chain-of-Thought reasoning, self-criticism and correction, efficient memory handling, and user-friendly natural language communication. It is designed to function without requiring users to have prior knowledge of Control Theory, enabling them to input problems in plain language and receive complete, real-time solutions. To evaluate the system, we propose new performance metrics assessing both individual agents and the system as a whole. We test five categories of Control Theory problems and benchmark performance across three advanced LLMs. Additionally, we conduct a comprehensive qualitative conversational analysis covering all key services. Results show that the LLM-Agent-Controller successfully solved 83% of general tasks, with individual agents achieving an average success rate of 87%. Performance improved with more advanced LLMs. This research demonstrates the potential of multi-agent LLM architectures to solve complex, domain-specific problems. By integrating specialized agents, supervisory control, and advanced reasoning, the LLM-Agent-Controller offers a scalable, robust, and accessible solution framework that can be extended to various technical domains. 

---
# Automated Text-to-Table for Reasoning-Intensive Table QA: Pipeline Design and Benchmarking Insights 

**Authors**: Shi-Yu Tian, Zhi Zhou, Wei Dong, Ming Yang, Kun-Yang Yu, Zi-Jian Cheng, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19563)  

**Abstract**: Reasoning with tabular data holds increasing importance in modern applications, yet comprehensive evaluation methodologies for reasoning-intensive Table Question Answering (QA) tasks remain nascent. Existing research is constrained by two primary bottlenecks: 1) Reliance on costly manually annotated real-world data, which is difficult to cover complex reasoning scenarios; 2) The heterogeneity of table structures hinders systematic analysis of the intrinsic mechanisms behind the underperformance of LLMs, especially in reasoning-intensive tasks. To address these issues, we propose an automated generation pipeline AutoT2T that transforms mathematical word problems into table-based reasoning tasks, eliminating the need for manual annotation. The pipeline can generate multiple variants of a table for the same reasoning problem, including noisy versions to support robustness evaluation. Based on this, we construct a new benchmark TabularGSM, which systematically spans a range of table complexities and trap problems. Experimental analyses through AutoT2T and TabularGSM reveal that the tight coupling between reasoning and retrieval or identification processes is a key factor underlying the failure of LLMs in complex Table QA tasks. This highlights the necessity for models to develop synergistic reasoning capabilities in order to perform effectively in complex Table QA tasks. 

---
# AMQA: An Adversarial Dataset for Benchmarking Bias of LLMs in Medicine and Healthcare 

**Authors**: Ying Xiao, Jie Huang, Ruijuan He, Jing Xiao, Mohammad Reza Mousavi, Yepang Liu, Kezhi Li, Zhenpeng Chen, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19562)  

**Abstract**: Large language models (LLMs) are reaching expert-level accuracy on medical diagnosis questions, yet their mistakes and the biases behind them pose life-critical risks. Bias linked to race, sex, and socioeconomic status is already well known, but a consistent and automatic testbed for measuring it is missing. To fill this gap, this paper presents AMQA -- an Adversarial Medical Question-Answering dataset -- built for automated, large-scale bias evaluation of LLMs in medical QA. AMQA includes 4,806 medical QA pairs sourced from the United States Medical Licensing Examination (USMLE) dataset, generated using a multi-agent framework to create diverse adversarial descriptions and question pairs. Using AMQA, we benchmark five representative LLMs and find surprisingly substantial disparities: even GPT-4.1, the least biased model tested, answers privileged-group questions over 10 percentage points more accurately than unprivileged ones. Compared with the existing benchmark CPV, AMQA reveals 15% larger accuracy gaps on average between privileged and unprivileged groups. Our dataset and code are publicly available at this https URL to support reproducible research and advance trustworthy, bias-aware medical AI. 

---
# Turing Test 2.0: The General Intelligence Threshold 

**Authors**: Georgios Mappouras  

**Link**: [PDF](https://arxiv.org/pdf/2505.19550)  

**Abstract**: With the rise of artificial intelligence (A.I.) and large language models like Chat-GPT, a new race for achieving artificial general intelligence (A.G.I) has started. While many speculate how and when A.I. will achieve A.G.I., there is no clear agreement on how A.G.I. can be detected in A.I. models, even when popular tools like the Turing test (and its modern variations) are used to measure their intelligence. In this work, we discuss why traditional methods like the Turing test do not suffice for measuring or detecting A.G.I. and provide a new, practical method that can be used to decide if a (computer or any other) system has reached or surpassed A.G.I. To achieve this, we make two new contributions. First, we present a clear definition for general intelligence (G.I.) and set a G.I. threshold (G.I.T.) that can be used to distinguish between systems that achieve A.G.I. and systems that do not. Second, we present a new framework on how to construct tests that can detect if a system has achieved G.I. in a simple, comprehensive, and clear-cut fail/pass way. We call this novel framework the Turing Tests 2.0. We then demonstrate real-life examples of applying tests that follow our Turing Tests 2.0 framework on modern A.I. models. 

---
# Genome-Bench: A Scientific Reasoning Benchmark from Real-World Expert Discussions 

**Authors**: Ming Yin, Yuanhao Qu, Dyllan Liu, Ling Yang, Le Cong, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19501)  

**Abstract**: In this short report, we present an automated pipeline tailored for the genomics domain and introduce \textit{Genome-Bench}, a new benchmark constructed from over a decade of scientific forum discussions on genome engineering. Our pipeline transforms raw interactions into a reinforcement learning friendly multiple-choice questions format, supported by 3000+ high quality question answer pairs spanning foundational biology, experimental troubleshooting, tool usage, and beyond. To our knowledge, this is the first end-to-end pipeline for teaching LLMs to reason from scientific discussions, with promising potential for generalization across scientific domains beyond biology. 

---
# Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models 

**Authors**: Jianxing Liao, Junyan Xu, Yatao Sun, Maowen Tang, Sicheng He, Jingxian Liao, Shui Yu, Yun Li, Hongguan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19490)  

**Abstract**: Designing complex computer-aided design (CAD) models is often time-consuming due to challenges such as computational inefficiency and the difficulty of generating precise models. We propose a novel language-guided framework for industrial design automation to address these issues, integrating large language models (LLMs) with computer-automated design (CAutoD).Through this framework, CAD models are automatically generated from parameters and appearance descriptions, supporting the automation of design tasks during the detailed CAD design phase. Our approach introduces three key innovations: (1) a semi-automated data annotation pipeline that leverages LLMs and vision-language large models (VLLMs) to generate high-quality parameters and appearance descriptions; (2) a Transformer-based CAD generator (TCADGen) that predicts modeling sequences via dual-channel feature aggregation; (3) an enhanced CAD modeling generation model, called CADLLM, that is designed to refine the generated sequences by incorporating the confidence scores from TCADGen. Experimental results demonstrate that the proposed approach outperforms traditional methods in both accuracy and efficiency, providing a powerful tool for automating industrial workflows and generating complex CAD models from textual prompts. The code is available at this https URL 

---
# Benchmarking and Enhancing LLM Agents in Localizing Linux Kernel Bugs 

**Authors**: Zhenhao Zhou, Zhuochen Huang, Yike He, Chong Wang, Jiajun Wang, Yijian Wu, Xin Peng, Yiling Lou  

**Link**: [PDF](https://arxiv.org/pdf/2505.19489)  

**Abstract**: The Linux kernel is a critical system, serving as the foundation for numerous systems. Bugs in the Linux kernel can cause serious consequences, affecting billions of users. Fault localization (FL), which aims at identifying the buggy code elements in software, plays an essential role in software quality assurance. While recent LLM agents have achieved promising accuracy in FL on recent benchmarks like SWE-bench, it remains unclear how well these methods perform in the Linux kernel, where FL is much more challenging due to the large-scale code base, limited observability, and diverse impact factors. In this paper, we introduce LinuxFLBench, a FL benchmark constructed from real-world Linux kernel bugs. We conduct an empirical study to assess the performance of state-of-the-art LLM agents on the Linux kernel. Our initial results reveal that existing agents struggle with this task, achieving a best top-1 accuracy of only 41.6% at file level. To address this challenge, we propose LinuxFL$^+$, an enhancement framework designed to improve FL effectiveness of LLM agents for the Linux kernel. LinuxFL$^+$ substantially improves the FL accuracy of all studied agents (e.g., 7.2% - 11.2% accuracy increase) with minimal costs. Data and code are available at this https URL. 

---
# Judging with Many Minds: Do More Perspectives Mean Less Prejudice? 

**Authors**: Chiyu Ma, Enpei Zhang, Yilun Zhao, Wenjun Liu, Yaning Jia, Peijun Qing, Lin Shi, Arman Cohan, Yujun Yan, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19477)  

**Abstract**: LLM-as-Judge has emerged as a scalable alternative to human evaluation, enabling large language models (LLMs) to provide reward signals in trainings. While recent work has explored multi-agent extensions such as multi-agent debate and meta-judging to enhance evaluation quality, the question of how intrinsic biases manifest in these settings remains underexplored. In this study, we conduct a systematic analysis of four diverse bias types: position bias, verbosity bias, chain-of-thought bias, and bandwagon bias. We evaluate these biases across two widely adopted multi-agent LLM-as-Judge frameworks: Multi-Agent-Debate and LLM-as-Meta-Judge. Our results show that debate framework amplifies biases sharply after the initial debate, and this increased bias is sustained in subsequent rounds, while meta-judge approaches exhibit greater resistance. We further investigate the incorporation of PINE, a leading single-agent debiasing method, as a bias-free agent within these systems. The results reveal that this bias-free agent effectively reduces biases in debate settings but provides less benefit in meta-judge scenarios. Our work provides a comprehensive study of bias behavior in multi-agent LLM-as-Judge systems and highlights the need for targeted bias mitigation strategies in collaborative evaluation settings. 

---
# Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models 

**Authors**: Xinmiao Hu, Chun Wang, Ruihe An, ChenYu Shao, Xiaojun Ye, Sheng Zhou, Liangcheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19474)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong performance in visual understanding tasks, yet they often suffer from object hallucinations--generating descriptions of objects that are inconsistent with or entirely absent from the input. This issue is closely related to dataset biases, where frequent co-occurrences of objects lead to entangled semantic representations across modalities. As a result, models may erroneously activate object representations that are commonly associated with the input but not actually present.
To address this, we propose a causality-driven disentanglement framework that mitigates hallucinations through causal intervention. Our approach includes a Causal-Driven Projector in the visual pathway and a Causal Intervention Module integrated into the final transformer layer of the language model. These components work together to reduce spurious correlations caused by biased training data.
Experimental results show that our method significantly reduces hallucinations while maintaining strong performance on multiple multimodal benchmarks. Visualization analyses further confirm improved separability of object representations.
The code is available at: this https URL 

---
# Origin Tracer: A Method for Detecting LoRA Fine-Tuning Origins in LLMs 

**Authors**: Hongyu Liang, Yuting Zheng, Yihan Li, Yiran Zhang, Shiyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19466)  

**Abstract**: As large language models (LLMs) continue to advance, their deployment often involves fine-tuning to enhance performance on specific downstream tasks. However, this customization is sometimes accompanied by misleading claims about the origins, raising significant concerns about transparency and trust within the open-source community. Existing model verification techniques typically assess functional, representational, and weight similarities. However, these approaches often struggle against obfuscation techniques, such as permutations and scaling transformations. To address this limitation, we propose a novel detection method Origin-Tracer that rigorously determines whether a model has been fine-tuned from a specified base model. This method includes the ability to extract the LoRA rank utilized during the fine-tuning process, providing a more robust verification framework. This framework is the first to provide a formalized approach specifically aimed at pinpointing the sources of model fine-tuning. We empirically validated our method on thirty-one diverse open-source models under conditions that simulate real-world obfuscation scenarios. We empirically analyze the effectiveness of our framework and finally, discuss its limitations. The results demonstrate the effectiveness of our approach and indicate its potential to establish new benchmarks for model verification. 

---
# BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs 

**Authors**: Guilong Lu, Xuntao Guo, Rongjunchen Zhang, Wenqiao Zhu, Ji Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19457)  

**Abstract**: Large language models excel in general tasks, yet assessing their reliability in logic-heavy, precision-critical domains like finance, law, and healthcare remains challenging. To address this, we introduce BizFinBench, the first benchmark specifically designed to evaluate LLMs in real-world financial applications. BizFinBench consists of 6,781 well-annotated queries in Chinese, spanning five dimensions: numerical calculation, reasoning, information extraction, prediction recognition, and knowledge-based question answering, grouped into nine fine-grained categories. The benchmark includes both objective and subjective metrics. We also introduce IteraJudge, a novel LLM evaluation method that reduces bias when LLMs serve as evaluators in objective metrics. We benchmark 25 models, including both proprietary and open-source systems. Extensive experiments show that no model dominates across all tasks. Our evaluation reveals distinct capability patterns: (1) In Numerical Calculation, Claude-3.5-Sonnet (63.18) and DeepSeek-R1 (64.04) lead, while smaller models like Qwen2.5-VL-3B (15.92) lag significantly; (2) In Reasoning, proprietary models dominate (ChatGPT-o3: 83.58, Gemini-2.0-Flash: 81.15), with open-source models trailing by up to 19.49 points; (3) In Information Extraction, the performance spread is the largest, with DeepSeek-R1 scoring 71.46, while Qwen3-1.7B scores 11.23; (4) In Prediction Recognition, performance variance is minimal, with top models scoring between 39.16 and 50.00. We find that while current LLMs handle routine finance queries competently, they struggle with complex scenarios requiring cross-concept reasoning. BizFinBench offers a rigorous, business-aligned benchmark for future research. The code and dataset are available at this https URL. 

---
# Style2Code: A Style-Controllable Code Generation Framework with Dual-Modal Contrastive Representation Learning 

**Authors**: Dutao Zhang, Sergey Kovalchuk, YuLong He  

**Link**: [PDF](https://arxiv.org/pdf/2505.19442)  

**Abstract**: Controllable code generation, the ability to synthesize code that follows a specified style while maintaining functionality, remains a challenging task. We propose a two-stage training framework combining contrastive learning and conditional decoding to enable flexible style control. The first stage aligns code style representations with semantic and structural features. In the second stage, we fine-tune a language model (e.g., Flan-T5) conditioned on the learned style vector to guide generation. Our method supports style interpolation and user personalization via lightweight mixing. Compared to prior work, our unified framework offers improved stylistic control without sacrificing code correctness. This is among the first approaches to combine contrastive alignment with conditional decoding for style-guided code generation. 

---
# Task Memory Engine: Spatial Memory for Robust Multi-Step LLM Agents 

**Authors**: Ye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.19436)  

**Abstract**: Large Language Models (LLMs) falter in multi-step interactions -- often hallucinating, repeating actions, or misinterpreting user corrections -- due to reliance on linear, unstructured context. This fragility stems from the lack of persistent memory to track evolving goals and task dependencies, undermining trust in autonomous agents. We introduce the Task Memory Engine (TME), a modular memory controller that transforms existing LLMs into robust, revision-aware agents without fine-tuning. TME implements a spatial memory framework that replaces flat context with graph-based structures to support consistent, multi-turn reasoning. Departing from linear concatenation and ReAct-style prompting, TME builds a dynamic task graph -- either a tree or directed acyclic graph (DAG) -- to map user inputs to subtasks, align them with prior context, and enable dependency-tracked revisions. Its Task Representation and Intent Management (TRIM) component models task semantics and user intent to ensure accurate interpretation. Across four multi-turn scenarios-trip planning, cooking, meeting scheduling, and shopping cart editing -- TME eliminates 100% of hallucinations and misinterpretations in three tasks, and reduces hallucinations by 66.7% and misinterpretations by 83.3% across 27 user turns, outperforming ReAct. TME's modular design supports plug-and-play deployment and domain-specific customization, adaptable to both personal assistants and enterprise automation. We release TME's codebase, benchmarks, and components as open-source resources, enabling researchers to develop reliable LLM agents. TME's scalable architecture addresses a critical gap in agent performance across complex, interactive settings. 

---
# Toward Physics-Informed Machine Learning for Data Center Operations: A Tropical Case Study 

**Authors**: Ruihang Wang, Zhiwei Cao, Qingang Zhang, Rui Tan, Yonggang Wen, Tommy Leung, Stuart Kennedy, Justin Teoh  

**Link**: [PDF](https://arxiv.org/pdf/2505.19414)  

**Abstract**: Data centers are the backbone of computing capacity. Operating data centers in the tropical regions faces unique challenges due to consistently high ambient temperature and elevated relative humidity throughout the year. These conditions result in increased cooling costs to maintain the reliability of the computing systems. While existing machine learning-based approaches have demonstrated potential to elevate operations to a more proactive and intelligent level, their deployment remains dubious due to concerns about model extrapolation capabilities and associated system safety issues. To address these concerns, this article proposes incorporating the physical characteristics of data centers into traditional data-driven machine learning solutions. We begin by introducing the data center system, including the relevant multiphysics processes and the data-physics availability. Next, we outline the associated modeling and optimization problems and propose an integrated, physics-informed machine learning system to address them. Using the proposed system, we present relevant applications across varying levels of operational intelligence. A case study on an industry-grade tropical data center is provided to demonstrate the effectiveness of our approach. Finally, we discuss key challenges and highlight potential future directions. 

---
# Fusion Intelligence for Digital Twinning AI Data Centers: A Synergistic GenAI-PhyAI Approach 

**Authors**: Ruihang Wang, Minghao Li, Zhiwei Cao, Jimin Jia, Kyle Guan, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19409)  

**Abstract**: The explosion in artificial intelligence (AI) applications is pushing the development of AI-dedicated data centers (AIDCs), creating management challenges that traditional methods and standalone AI solutions struggle to address. While digital twins are beneficial for AI-based design validation and operational optimization, current AI methods for their creation face limitations. Specifically, physical AI (PhyAI) aims to capture the underlying physical laws, which demands extensive, case-specific customization, and generative AI (GenAI) can produce inaccurate or hallucinated results. We propose Fusion Intelligence, a novel framework synergizing GenAI's automation with PhyAI's domain grounding. In this dual-agent collaboration, GenAI interprets natural language prompts to generate tokenized AIDC digital twins. Subsequently, PhyAI optimizes these generated twins by enforcing physical constraints and assimilating real-time data. Case studies demonstrate the advantages of our framework in automating the creation and validation of AIDC digital twins. These twins deliver predictive analytics to support power usage effectiveness (PUE) optimization in the design stage. With operational data collected, the digital twin accuracy is further improved compared with pure physics-based models developed by human experts. Fusion Intelligence offers a promising pathway to accelerate digital transformation. It enables more reliable and efficient AI-driven digital transformation for a broad range of mission-critical infrastructures. 

---
# Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model 

**Authors**: Tianle Li, Jihai Zhang, Yongming Rao, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19406)  

**Abstract**: While large language models (LLMs) demonstrate strong reasoning capabilities utilizing reinforcement learning (RL) with verifiable reward, whether large vision-language models (VLMs) can directly inherit such capabilities through similar post-training strategies remains underexplored. In this work, we conduct a systematic compositional probing study to evaluate whether current VLMs trained with RL or other post-training strategies can compose capabilities across modalities or tasks under out-of-distribution conditions. We design a suite of diagnostic tasks that train models on unimodal tasks or isolated reasoning skills, and evaluate them on multimodal, compositional variants requiring skill integration. Through comparisons between supervised fine-tuning (SFT) and RL-trained models, we identify three key findings: (1) RL-trained models consistently outperform SFT on compositional generalization, demonstrating better integration of learned skills; (2) although VLMs achieve strong performance on individual tasks, they struggle to generalize compositionally under cross-modal and cross-task scenario, revealing a significant gap in current training strategies; (3) enforcing models to explicitly describe visual content before reasoning (e.g., caption-before-thinking), along with rewarding progressive vision-to-text grounding, yields notable gains. It highlights two essential ingredients for improving compositionality in VLMs: visual-to-text alignment and accurate visual grounding. Our findings shed light on the current limitations of RL-based reasoning VLM training and provide actionable insights toward building models that reason compositionally across modalities and tasks. 

---
# Recalibrating the Compass: Integrating Large Language Models into Classical Research Methods 

**Authors**: Tai-Quan Peng, Xuzhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19402)  

**Abstract**: This paper examines how large language models (LLMs) are transforming core quantitative methods in communication research in particular, and in the social sciences more broadly-namely, content analysis, survey research, and experimental studies. Rather than replacing classical approaches, LLMs introduce new possibilities for coding and interpreting text, simulating dynamic respondents, and generating personalized and interactive stimuli. Drawing on recent interdisciplinary work, the paper highlights both the potential and limitations of LLMs as research tools, including issues of validity, bias, and interpretability. To situate these developments theoretically, the paper revisits Lasswell's foundational framework -- "Who says what, in which channel, to whom, with what effect?" -- and demonstrates how LLMs reconfigure message studies, audience analysis, and effects research by enabling interpretive variation, audience trajectory modeling, and counterfactual experimentation. Revisiting the metaphor of the methodological compass, the paper argues that classical research logics remain essential as the field integrates LLMs and generative AI. By treating LLMs not only as technical instruments but also as epistemic and cultural tools, the paper calls for thoughtful, rigorous, and imaginative use of LLMs in future communication and social science research. 

---
# CaseEdit: Enhancing Localized Commonsense Reasoning via Null-Space Constrained Knowledge Editing in Small Parameter Language Models 

**Authors**: Varun Reddy, Yen-Ling Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.19383)  

**Abstract**: Large language models (LLMs) exhibit strong performance on factual recall and general reasoning but struggle to adapt to user-specific, commonsense knowledge, a challenge particularly acute in small-parameter settings where computational efficiency is prioritized. We introduce CaseEdit, a new dataset and generation pipeline for evaluating localized, personalized commonsense knowledge editing in small LLMs to address this. Built upon the ATOMIC20/20 commonsense graph, CaseEdit uses a multi-stage inference process to generate both typical and atypical contextual edits for household objects, paired with targeted evaluation questions across four axes: reliability, generalization, locality, and portability. We evaluate established knowledge editing methods using CaseEdit and demonstrate that AlphaEdit, a technique employing null-space projection to minimize interference with unrelated knowledge, consistently outperforms other methods when applied to an LLaMA 3.2 3B model, even in scalability tests, showing minimal ripple effects. Our results indicate that using CaseEdit with effective editing techniques like AlphaEdit allows small models to internalize high-quality, context-sensitive common-sense knowledge, paving the way for lightweight, personalized assistants. 

---
# DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving 

**Authors**: Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun Wang, Jinghao Chai, Qian Cao, Yuweng Heng, Hao Jiang, Zongzheng Zhang, Xianda Guo, Hao Sun, Hao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19381)  

**Abstract**: Research interest in end-to-end autonomous driving has surged owing to its fully differentiable design integrating modular tasks, i.e. perception, prediction and planing, which enables optimization in pursuit of the ultimate goal. Despite the great potential of the end-to-end paradigm, existing methods suffer from several aspects including expensive BEV (bird's eye view) computation, action diversity, and sub-optimal decision in complex real-world scenarios. To address these challenges, we propose a novel hybrid sparse-dense diffusion policy, empowered by a Vision-Language Model (VLM), called Diff-VLA. We explore the sparse diffusion representation for efficient multi-modal driving behavior. Moreover, we rethink the effectiveness of VLM driving decision and improve the trajectory generation guidance through deep interaction across agent, map instances and VLM output. Our method shows superior performance in Autonomous Grand Challenge 2025 which contains challenging real and reactive synthetic scenarios. Our methods achieves 45.0 PDMS. 

---
# Foundations of Top-$k$ Decoding For Language Models 

**Authors**: Georgy Noarov, Soham Mallick, Tao Wang, Sunay Joshi, Yan Sun, Yangxinyu Xie, Mengxin Yu, Edgar Dobriban  

**Link**: [PDF](https://arxiv.org/pdf/2505.19371)  

**Abstract**: Top-$k$ decoding is a widely used method for sampling from LLMs: at each token, only the largest $k$ next-token-probabilities are kept, and the next token is sampled after re-normalizing them to sum to unity. Top-$k$ and other sampling methods are motivated by the intuition that true next-token distributions are sparse, and the noisy LLM probabilities need to be truncated. However, to our knowledge, a precise theoretical motivation for the use of top-$k$ decoding is missing. In this work, we develop a theoretical framework that both explains and generalizes top-$k$ decoding. We view decoding at a fixed token as the recovery of a sparse probability distribution. We consider \emph{Bregman decoders} obtained by minimizing a separable Bregman divergence (for both the \emph{primal} and \emph{dual} cases) with a sparsity-inducing $\ell_0$ regularization. Despite the combinatorial nature of the objective, we show how to optimize it efficiently for a large class of divergences. We show that the optimal decoding strategies are greedy, and further that the loss function is discretely convex in $k$, so that binary search provably and efficiently finds the optimal $k$. We show that top-$k$ decoding arises as a special case for the KL divergence, and identify new decoding strategies that have distinct behaviors (e.g., non-linearly up-weighting larger probabilities after re-normalization). 

---
# Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments 

**Authors**: Mario Leiva, Noel Ngu, Joshua Shay Kricheli, Aditya Taparia, Ransalu Senanayake, Paulo Shakarian, Nathaniel Bastian, John Corcoran, Gerardo Simari  

**Link**: [PDF](https://arxiv.org/pdf/2505.19361)  

**Abstract**: The deployment of pre-trained perception models in novel environments often leads to performance degradation due to distributional shifts. Although recent artificial intelligence approaches for metacognition use logical rules to characterize and filter model errors, improving precision often comes at the cost of reduced recall. This paper addresses the hypothesis that leveraging multiple pre-trained models can mitigate this recall reduction. We formulate the challenge of identifying and managing conflicting predictions from various models as a consistency-based abduction problem. The input predictions and the learned error detection rules derived from each model are encoded in a logic program. We then seek an abductive explanation--a subset of model predictions--that maximizes prediction coverage while ensuring the rate of logical inconsistencies (derived from domain constraints) remains below a specified threshold. We propose two algorithms for this knowledge representation task: an exact method based on Integer Programming (IP) and an efficient Heuristic Search (HS). Through extensive experiments on a simulated aerial imagery dataset featuring controlled, complex distributional shifts, we demonstrate that our abduction-based framework outperforms individual models and standard ensemble baselines, achieving, for instance, average relative improvements of approximately 13.6% in F1-score and 16.6% in accuracy across 15 diverse test datasets when compared to the best individual model. Our results validate the use of consistency-based abduction as an effective mechanism to robustly integrate knowledge from multiple imperfect reasoners in challenging, novel scenarios. 

---
# Architectures of Error: A Philosophical Inquiry into AI and Human Code Generation 

**Authors**: Camilo Chacón Sartori  

**Link**: [PDF](https://arxiv.org/pdf/2505.19353)  

**Abstract**: With the rise of generative AI (GenAI), Large Language Models are increasingly employed for code generation, becoming active co-authors alongside human programmers. Focusing specifically on this application domain, this paper articulates distinct ``Architectures of Error'' to ground an epistemic distinction between human and machine code generation. Examined through their shared vulnerability to error, this distinction reveals fundamentally different causal origins: human-cognitive versus artificial-stochastic. To develop this framework and substantiate the distinction, the analysis draws critically upon Dennett's mechanistic functionalism and Rescher's methodological pragmatism. I argue that a systematic differentiation of these error profiles raises critical philosophical questions concerning semantic coherence, security robustness, epistemic limits, and control mechanisms in human-AI collaborative software development. The paper also utilizes Floridi's levels of abstraction to provide a nuanced understanding of how these error dimensions interact and may evolve with technological advancements. This analysis aims to offer philosophers a structured framework for understanding GenAI's unique epistemological challenges, shaped by these architectural foundations, while also providing software engineers a basis for more critically informed engagement. 

---
# PatentMind: A Multi-Aspect Reasoning Graph for Patent Similarity Evaluation 

**Authors**: Yongmin Yoo, Qiongkai Xu, Longbing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19347)  

**Abstract**: Patent similarity evaluation plays a critical role in intellectual property analysis. However, existing methods often overlook the intricate structure of patent documents, which integrate technical specifications, legal boundaries, and application contexts. We introduce PatentMind, a novel framework for patent similarity assessment based on a Multi-Aspect Reasoning Graph (MARG). PatentMind decomposes patents into three core dimensions: technical feature, application domain, and claim scope, to compute dimension-specific similarity scores. These scores are dynamically weighted through a four-stage reasoning process which integrates contextual signals to emulate expert-level judgment. To support evaluation, we construct PatentSimBench, a human-annotated benchmark comprising 500 patent pairs. Experimental results demonstrate that PatentMind achieves a strong correlation ($r=0.938$) with expert annotations, significantly outperforming embedding-based models and advanced prompt engineering this http URL results highlight the effectiveness of modular reasoning frameworks in overcoming key limitations of embedding-based methods for analyzing patent similarity. 

---
# Evaluating Steering Techniques using Human Similarity Judgments 

**Authors**: Zach Studdiford, Timothy T. Rogers, Siddharth Suresh, Kushin Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.19333)  

**Abstract**: Current evaluations of Large Language Model (LLM) steering techniques focus on task-specific performance, overlooking how well steered representations align with human cognition. Using a well-established triadic similarity judgment task, we assessed steered LLMs on their ability to flexibly judge similarity between concepts based on size or kind. We found that prompt-based steering methods outperformed other methods both in terms of steering accuracy and model-to-human alignment. We also found LLMs were biased towards 'kind' similarity and struggled with 'size' alignment. This evaluation approach, grounded in human cognition, adds further support to the efficacy of prompt-based steering and reveals privileged representational axes in LLMs prior to steering. 

---
# Effort-aware Fairness: Incorporating a Philosophy-informed, Human-centered Notion of Effort into Algorithmic Fairness Metrics 

**Authors**: Tin Nguyen, Jiannan Xu, Zora Che, Phuong-Anh Nguyen-Le, Rushil Dandamudi, Donald Braman, Furong Huang, Hal Daumé III, Zubin Jelveh  

**Link**: [PDF](https://arxiv.org/pdf/2505.19317)  

**Abstract**: Although popularized AI fairness metrics, e.g., demographic parity, have uncovered bias in AI-assisted decision-making outcomes, they do not consider how much effort one has spent to get to where one is today in the input feature space. However, the notion of effort is important in how Philosophy and humans understand fairness. We propose a philosophy-informed way to conceptualize and evaluate Effort-aware Fairness (EaF) based on the concept of Force, or temporal trajectory of predictive features coupled with inertia. In addition to our theoretical formulation of EaF metrics, our empirical contributions include: 1/ a pre-registered human subjects experiment, which demonstrates that for both stages of the (individual) fairness evaluation process, people consider the temporal trajectory of a predictive feature more than its aggregate value; 2/ pipelines to compute Effort-aware Individual/Group Fairness in the criminal justice and personal finance contexts. Our work may enable AI model auditors to uncover and potentially correct unfair decisions against individuals who spent significant efforts to improve but are still stuck with systemic/early-life disadvantages outside their control. 

---
# Next Token Prediction Is a Dead End for Creativity 

**Authors**: Ibukun Olatunji, Mark Sheppard  

**Link**: [PDF](https://arxiv.org/pdf/2505.19277)  

**Abstract**: This paper argues that token prediction is fundamentally misaligned with real creativity. While next-token models have enabled impressive advances in language generation, their architecture favours surface-level coherence over spontaneity, originality, and improvisational risk. We use battle rap as a case study to expose the limitations of predictive systems, demonstrating that they cannot truly engage in adversarial or emotionally resonant exchanges. By reframing creativity as an interactive process rather than a predictive output, we offer a vision for AI systems that are more expressive, responsive, and aligned with human creative practice. 

---
# Using Large Language Models to Assess Teachers' Pedagogical Content Knowledge 

**Authors**: Yaxuan Yang, Shiyu Wang, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.19266)  

**Abstract**: Assessing teachers' pedagogical content knowledge (PCK) through performance-based tasks is both time and effort-consuming. While large language models (LLMs) offer new opportunities for efficient automatic scoring, little is known about whether LLMs introduce construct-irrelevant variance (CIV) in ways similar to or different from traditional machine learning (ML) and human raters. This study examines three sources of CIV -- scenario variability, rater severity, and rater sensitivity to scenario -- in the context of video-based constructed-response tasks targeting two PCK sub-constructs: analyzing student thinking and evaluating teacher responsiveness. Using generalized linear mixed models (GLMMs), we compared variance components and rater-level scoring patterns across three scoring sources: human raters, supervised ML, and LLM. Results indicate that scenario-level variance was minimal across tasks, while rater-related factors contributed substantially to CIV, especially in the more interpretive Task II. The ML model was the most severe and least sensitive rater, whereas the LLM was the most lenient. These findings suggest that the LLM contributes to scoring efficiency while also introducing CIV as human raters do, yet with varying levels of contribution compared to supervised ML. Implications for rater training, automated scoring design, and future research on model interpretability are discussed. 

---
# Sensorimotor features of self-awareness in multimodal large language models 

**Authors**: Iñaki Dellibarda Varela, Pablo Romero-Sorozabal, Diego Torricelli, Gabriel Delgado-Oleas, Jose Ignacio Serrano, Maria Dolores del Castillo Sobrino, Eduardo Rocon, Manuel Cebrian  

**Link**: [PDF](https://arxiv.org/pdf/2505.19237)  

**Abstract**: Self-awareness - the ability to distinguish oneself from the surrounding environment - underpins intelligent, autonomous behavior. Recent advances in AI achieve human-like performance in tasks integrating multimodal information, particularly in large language models, raising interest in the embodiment capabilities of AI agents on nonhuman platforms such as robots. Here, we explore whether multimodal LLMs can develop self-awareness solely through sensorimotor experiences. By integrating a multimodal LLM into an autonomous mobile robot, we test its ability to achieve this capacity. We find that the system exhibits robust environmental awareness, self-recognition and predictive awareness, allowing it to infer its robotic nature and motion characteristics. Structural equation modeling reveals how sensory integration influences distinct dimensions of self-awareness and its coordination with past-present memory, as well as the hierarchical internal associations that drive self-identification. Ablation tests of sensory inputs identify critical modalities for each dimension, demonstrate compensatory interactions among sensors and confirm the essential role of structured and episodic memory in coherent reasoning. These findings demonstrate that, given appropriate sensory information about the world and itself, multimodal LLMs exhibit emergent self-awareness, opening the door to artificial embodied cognitive systems. 

---
# GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling 

**Authors**: Jialong Zhou, Lichao Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19234)  

**Abstract**: The emergence of large language models (LLMs) enables the development of intelligent agents capable of engaging in complex and multi-turn dialogues. However, multi-agent collaboration face critical safety challenges, such as hallucination amplification and error injection and propagation. This paper presents GUARDIAN, a unified method for detecting and mitigating multiple safety concerns in GUARDing Intelligent Agent collaboratioNs. By modeling the multi-agent collaboration process as a discrete-time temporal attributed graph, GUARDIAN explicitly captures the propagation dynamics of hallucinations and errors. The unsupervised encoder-decoder architecture incorporating an incremental training paradigm, learns to reconstruct node attributes and graph structures from latent embeddings, enabling the identification of anomalous nodes and edges with unparalleled precision. Moreover, we introduce a graph abstraction mechanism based on the Information Bottleneck Theory, which compresses temporal interaction graphs while preserving essential patterns. Extensive experiments demonstrate GUARDIAN's effectiveness in safeguarding LLM multi-agent collaborations against diverse safety vulnerabilities, achieving state-of-the-art accuracy with efficient resource utilization. 

---
# DeCoDe: Defer-and-Complement Decision-Making via Decoupled Concept Bottleneck Models 

**Authors**: Chengbo He, Bochao Zou, Junliang Xing, Jiansheng Chen, Yuanchun Shi, Huimin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.19220)  

**Abstract**: In human-AI collaboration, a central challenge is deciding whether the AI should handle a task, be deferred to a human expert, or be addressed through collaborative effort. Existing Learning to Defer approaches typically make binary choices between AI and humans, neglecting their complementary strengths. They also lack interpretability, a critical property in high-stakes scenarios where users must understand and, if necessary, correct the model's reasoning. To overcome these limitations, we propose Defer-and-Complement Decision-Making via Decoupled Concept Bottleneck Models (DeCoDe), a concept-driven framework for human-AI collaboration. DeCoDe makes strategy decisions based on human-interpretable concept representations, enhancing transparency throughout the decision process. It supports three flexible modes: autonomous AI prediction, deferral to humans, and human-AI collaborative complementarity, selected via a gating network that takes concept-level inputs and is trained using a novel surrogate loss that balances accuracy and human effort. This approach enables instance-specific, interpretable, and adaptive human-AI collaboration. Experiments on real-world datasets demonstrate that DeCoDe significantly outperforms AI-only, human-only, and traditional deferral baselines, while maintaining strong robustness and interpretability even under noisy expert annotations. 

---
# Where Paths Collide: A Comprehensive Survey of Classic and Learning-Based Multi-Agent Pathfinding 

**Authors**: Shiyue Wang, Haozheng Xu, Yuhan Zhang, Jingran Lin, Changhong Lu, Xiangfeng Wang, Wenhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19219)  

**Abstract**: Multi-Agent Path Finding (MAPF) is a fundamental problem in artificial intelligence and robotics, requiring the computation of collision-free paths for multiple agents navigating from their start locations to designated goals. As autonomous systems become increasingly prevalent in warehouses, urban transportation, and other complex environments, MAPF has evolved from a theoretical challenge to a critical enabler of real-world multi-robot coordination. This comprehensive survey bridges the long-standing divide between classical algorithmic approaches and emerging learning-based methods in MAPF research. We present a unified framework that encompasses search-based methods (including Conflict-Based Search, Priority-Based Search, and Large Neighborhood Search), compilation-based approaches (SAT, SMT, CSP, ASP, and MIP formulations), and data-driven techniques (reinforcement learning, supervised learning, and hybrid strategies). Through systematic analysis of experimental practices across 200+ papers, we uncover significant disparities in evaluation methodologies, with classical methods typically tested on larger-scale instances (up to 200 by 200 grids with 1000+ agents) compared to learning-based approaches (predominantly 10-100 agents). We provide a comprehensive taxonomy of evaluation metrics, environment types, and baseline selections, highlighting the need for standardized benchmarking protocols. Finally, we outline promising future directions including mixed-motive MAPF with game-theoretic considerations, language-grounded planning with large language models, and neural solver architectures that combine the rigor of classical methods with the flexibility of deep learning. This survey serves as both a comprehensive reference for researchers and a practical guide for deploying MAPF solutions in increasingly complex real-world applications. 

---
# Improving Medical Reasoning with Curriculum-Aware Reinforcement Learning 

**Authors**: Shaohao Rui, Kaitao Chen, Weijie Ma, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19213)  

**Abstract**: Recent advances in reinforcement learning with verifiable, rule-based rewards have greatly enhanced the reasoning capabilities and out-of-distribution generalization of VLMs/LLMs, obviating the need for manually crafted reasoning chains. Despite these promising developments in the general domain, their translation to medical imaging remains limited. Current medical reinforcement fine-tuning (RFT) methods predominantly focus on close-ended VQA, thereby restricting the model's ability to engage in world knowledge retrieval and flexible task adaptation. More critically, these methods fall short of addressing the critical clinical demand for open-ended, reasoning-intensive decision-making. To bridge this gap, we introduce \textbf{MedCCO}, the first multimodal reinforcement learning framework tailored for medical VQA that unifies close-ended and open-ended data within a curriculum-driven RFT paradigm. Specifically, MedCCO is initially fine-tuned on a diverse set of close-ended medical VQA tasks to establish domain-grounded reasoning capabilities, and is then progressively adapted to open-ended tasks to foster deeper knowledge enhancement and clinical interpretability. We validate MedCCO across eight challenging medical VQA benchmarks, spanning both close-ended and open-ended settings. Experimental results show that MedCCO consistently enhances performance and generalization, achieving a 11.4\% accuracy gain across three in-domain tasks, and a 5.7\% improvement on five out-of-domain benchmarks. These findings highlight the promise of curriculum-guided RL in advancing robust, clinically-relevant reasoning in medical multimodal language models. 

---
# Structuring the Unstructured: A Multi-Agent System for Extracting and Querying Financial KPIs and Guidance 

**Authors**: Chanyeol Choi, Jihoon Kwon, Minjae Kim, Juneha Hwang, Minsoo Ha, Chaewoon Kim, Jaeseon Ha, Suyeol Yun, Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.19197)  

**Abstract**: Extracting structured and quantitative insights from unstructured financial filings is essential in investment research, yet remains time-consuming and resource-intensive. Conventional approaches in practice rely heavily on labor-intensive manual processes, limiting scalability and delaying the research workflow. In this paper, we propose an efficient and scalable method for accurately extracting quantitative insights from unstructured financial documents, leveraging a multi-agent system composed of large language models. Our proposed multi-agent system consists of two specialized agents: the \emph{Extraction Agent} and the \emph{Text-to-SQL Agent}. The \textit{Extraction Agent} automatically identifies key performance indicators from unstructured financial text, standardizes their formats, and verifies their accuracy. On the other hand, the \textit{Text-to-SQL Agent} generates executable SQL statements from natural language queries, allowing users to access structured data accurately without requiring familiarity with the database schema. Through experiments, we demonstrate that our proposed system effectively transforms unstructured text into structured data accurately and enables precise retrieval of key information. First, we demonstrate that our system achieves approximately 95\% accuracy in transforming financial filings into structured data, matching the performance level typically attained by human annotators. Second, in a human evaluation of the retrieval task -- where natural language queries are used to search information from structured data -- 91\% of the responses were rated as correct by human evaluators. In both evaluations, our system generalizes well across financial document types, consistently delivering reliable performance. 

---
# CardioCoT: Hierarchical Reasoning for Multimodal Survival Analysis 

**Authors**: Shaohao Rui, Haoyang Su, Jinyi Xiang, Lian-Ming Wu, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19195)  

**Abstract**: Accurate prediction of major adverse cardiovascular events recurrence risk in acute myocardial infarction patients based on postoperative cardiac MRI and associated clinical notes is crucial for precision treatment and personalized intervention. Existing methods primarily focus on risk stratification capability while overlooking the need for intermediate robust reasoning and model interpretability in clinical practice. Moreover, end-to-end risk prediction using LLM/VLM faces significant challenges due to data limitations and modeling complexity. To bridge this gap, we propose CardioCoT, a novel two-stage hierarchical reasoning-enhanced survival analysis framework designed to enhance both model interpretability and predictive performance. In the first stage, we employ an evidence-augmented self-refinement mechanism to guide LLM/VLMs in generating robust hierarchical reasoning trajectories based on associated radiological findings. In the second stage, we integrate the reasoning trajectories with imaging data for risk model training and prediction. CardioCoT demonstrates superior performance in MACE recurrence risk prediction while providing interpretable reasoning processes, offering valuable insights for clinical decision-making. 

---
# Investigating Pedagogical Teacher and Student LLM Agents: Genetic Adaptation Meets Retrieval Augmented Generation Across Learning Style 

**Authors**: Debdeep Sanyal, Agniva Maiti, Umakanta Maharana, Dhruv Kumar, Ankur Mali, C. Lee Giles, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2505.19173)  

**Abstract**: Effective teaching requires adapting instructional strategies to accommodate the diverse cognitive and behavioral profiles of students, a persistent challenge in education and teacher training. While Large Language Models (LLMs) offer promise as tools to simulate such complex pedagogical environments, current simulation frameworks are limited in two key respects: (1) they often reduce students to static knowledge profiles, and (2) they lack adaptive mechanisms for modeling teachers who evolve their strategies in response to student feedback. To address these gaps, \textbf{we introduce a novel simulation framework that integrates LLM-based heterogeneous student agents with a self-optimizing teacher agent}. The teacher agent's pedagogical policy is dynamically evolved using a genetic algorithm, allowing it to discover and refine effective teaching strategies based on the aggregate performance of diverse learners. In addition, \textbf{we propose Persona-RAG}, a Retrieval Augmented Generation module that enables student agents to retrieve knowledge tailored to their individual learning styles. Persona-RAG preserves the retrieval accuracy of standard RAG baselines while enhancing personalization, an essential factor in modeling realistic educational scenarios. Through extensive experiments, we demonstrate how our framework supports the emergence of distinct and interpretable teaching patterns when interacting with varied student populations. Our results highlight the potential of LLM-driven simulations to inform adaptive teaching practices and provide a testbed for training human educators in controlled, data-driven environments. 

---
# Amplifying Human Creativity and Problem Solving with AI Through Generative Collective Intelligence 

**Authors**: Thomas P. Kehler, Scott E. Page, Alex Pentland, Martin Reeves, John Seely Brown  

**Link**: [PDF](https://arxiv.org/pdf/2505.19167)  

**Abstract**: We propose a new framework for human-AI collaboration that amplifies the distinct capabilities of both. This framework, which we call Generative Collective Intelligence (GCI), shifts AI to the group/social level and employs AI in dual roles: as interactive agents and as technology that accumulates, organizes, and leverages knowledge. By creating a cognitive bridge between human reasoning and AI models, GCI can overcome the limitations of purely algorithmic approaches to problem-solving and decision-making. The framework demonstrates how AI can be reframed as a social and cultural technology that enables groups to solve complex problems through structured collaboration that transcends traditional communication barriers. We describe the mathematical foundations of GCI based on comparative judgment and minimum regret principles, and illustrate its applications across domains including climate adaptation, healthcare transformation, and civic participation. By combining human creativity with AI's computational capabilities, GCI offers a promising approach to addressing complex societal challenges that neither human or machines can solve alone. 

---
# OrgAccess: A Benchmark for Role Based Access Control in Organization Scale LLMs 

**Authors**: Debdeep Sanyal Umakanta Maharana, Yash Sinha, Hong Ming Tan, Shirish Karande, Mohan Kankanhalli, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2505.19165)  

**Abstract**: Role-based access control (RBAC) and hierarchical structures are foundational to how information flows and decisions are made within virtually all organizations. As the potential of Large Language Models (LLMs) to serve as unified knowledge repositories and intelligent assistants in enterprise settings becomes increasingly apparent, a critical, yet under explored, challenge emerges: \textit{can these models reliably understand and operate within the complex, often nuanced, constraints imposed by organizational hierarchies and associated permissions?} Evaluating this crucial capability is inherently difficult due to the proprietary and sensitive nature of real-world corporate data and access control policies. We introduce a synthetic yet representative \textbf{OrgAccess} benchmark consisting of 40 distinct types of permissions commonly relevant across different organizational roles and levels. We further create three types of permissions: 40,000 easy (1 permission), 10,000 medium (3-permissions tuple), and 20,000 hard (5-permissions tuple) to test LLMs' ability to accurately assess these permissions and generate responses that strictly adhere to the specified hierarchical rules, particularly in scenarios involving users with overlapping or conflicting permissions. Our findings reveal that even state-of-the-art LLMs struggle significantly to maintain compliance with role-based structures, even with explicit instructions, with their performance degrades further when navigating interactions involving two or more conflicting permissions. Specifically, even \textbf{GPT-4.1 only achieves an F1-Score of 0.27 on our hardest benchmark}. This demonstrates a critical limitation in LLMs' complex rule following and compositional reasoning capabilities beyond standard factual or STEM-based benchmarks, opening up a new paradigm for evaluating their fitness for practical, structured environments. 

---
# SeePhys: Does Seeing Help Thinking? -- Benchmarking Vision-Based Physics Reasoning 

**Authors**: Kun Xiang, Heng Li, Terry Jingchen Zhang, Yinya Huang, Zirong Liu, Peixin Qu, Jixi He, Jiaqi Chen, Yu-Jie Yuan, Jianhua Han, Hang Xu, Hanhui Li, Mrinmaya Sachan, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19099)  

**Abstract**: We present SeePhys, a large-scale multimodal benchmark for LLM reasoning grounded in physics questions ranging from middle school to PhD qualifying exams. The benchmark covers 7 fundamental domains spanning the physics discipline, incorporating 21 categories of highly heterogeneous diagrams. In contrast to prior works where visual elements mainly serve auxiliary purposes, our benchmark features a substantial proportion of vision-essential problems (75\%) that mandate visual information extraction for correct solutions. Through extensive evaluation, we observe that even the most advanced visual reasoning models (e.g., Gemini-2.5-pro and o4-mini) achieve sub-60\% accuracy on our benchmark. These results reveal fundamental challenges in current large language models' visual understanding capabilities, particularly in: (i) establishing rigorous coupling between diagram interpretation and physics reasoning, and (ii) overcoming their persistent reliance on textual cues as cognitive shortcuts. 

---
# ScreenExplorer: Training a Vision-Language Model for Diverse Exploration in Open GUI World 

**Authors**: Runliang Niu, Jinglong Ji, Yi Chang, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19095)  

**Abstract**: The rapid progress of large language models (LLMs) has sparked growing interest in building Artificial General Intelligence (AGI) within Graphical User Interface (GUI) environments. However, existing GUI agents based on LLMs or vision-language models (VLMs) often fail to generalize to novel environments and rely heavily on manually curated, diverse datasets. To overcome these limitations, we introduce ScreenExplorer, a VLM trained via Group Relative Policy Optimization(GRPO) in real, dynamic, and open-ended GUI environments. Innovatively, we introduced a world-model-based curiosity reward function to help the agent overcome the cold-start phase of exploration. Additionally, distilling experience streams further enhances the model's exploration capabilities. Our training framework enhances model exploration in open GUI environments, with trained models showing better environmental adaptation and sustained exploration compared to static deployment models. Our findings offer a scalable pathway toward AGI systems with self-improving capabilities in complex interactive settings. 

---
# Reinforced Latent Reasoning for LLM-based Recommendation 

**Authors**: Yang Zhang, Wenxin Xu, Xiaoyan Zhao, Wenjie Wang, Fuli Feng, Xiangnan He, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2505.19092)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning capabilities in complex problem-solving tasks, sparking growing interest in their application to preference reasoning in recommendation systems. Existing methods typically rely on fine-tuning with explicit chain-of-thought (CoT) data. However, these methods face significant practical limitations due to (1) the difficulty of obtaining high-quality CoT data in recommendation and (2) the high inference latency caused by generating CoT reasoning. In this work, we explore an alternative approach that shifts from explicit CoT reasoning to compact, information-dense latent reasoning. This approach eliminates the need for explicit CoT generation and improves inference efficiency, as a small set of latent tokens can effectively capture the entire reasoning process. Building on this idea, we propose $\textit{\underline{R}einforced \underline{Latent} \underline{R}easoning for \underline{R}ecommendation}$ (LatentR$^3$), a novel end-to-end training framework that leverages reinforcement learning (RL) to optimize latent reasoning without relying on any CoT this http URL$^3$ adopts a two-stage training strategy: first, supervised fine-tuning to initialize the latent reasoning module, followed by pure RL training to encourage exploration through a rule-based reward design. Our RL implementation is based on a modified GRPO algorithm, which reduces computational overhead during training and introduces continuous reward signals for more efficient learning. Extensive experiments demonstrate that LatentR$^3$ enables effective latent reasoning without any direct supervision of the reasoning process, significantly improving performance when integrated with different LLM-based recommendation methods. Our codes are available at this https URL. 

---
# Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs 

**Authors**: Jaemin Kim, Hangeol Chang, Hyunmin Hwang, Choonghan Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.19075)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable general capabilities, but enhancing skills such as reasoning often demands substantial computational resources and may compromise their generalization. While Parameter-Efficient Fine-Tuning (PEFT) methods offer a more resource-conscious alternative, they typically requires retraining for each LLM backbone due to architectural dependencies. To address these challenges, here we propose Universal Reasoner (UniR) - a single, lightweight, composable, and plug-and-play reasoning module that can be used with any frozen LLM to endow it with specialized reasoning capabilities. Specifically, UniR decomposes the reward into a standalone reasoning module that is trained independently using predefined rewards, effectively translating trajectory-level signals into token-level guidance. Once trained, UniR can be combined with any frozen LLM at inference time by simply adding its output logits to those of the LLM backbone. This additive structure naturally enables modular composition: multiple UniR modules trained for different tasks can be jointly applied by summing their logits, enabling complex reasoning via composition. Experimental results on mathematical reasoning and machine translation tasks show that UniR significantly outperforms \add{existing baseline fine-tuning methods using the Llama3.2 model}. Furthermore, UniR demonstrates strong weak-to-strong generalization: reasoning modules trained on smaller models effectively guide much larger LLMs. This makes UniR a cost-efficient, adaptable, and robust solution for enhancing reasoning in LLMs without compromising their core capabilities. Code is open-sourced at this https URL 

---
# RECAST: Strengthening LLMs' Complex Instruction Following with Constraint-Verifiable Data 

**Authors**: Wenhao Liu, Zhengkang Guo, Mingchen Xie, Jingwen Xu, Zisu Huang, Muzhao Tian, Jianhan Xu, Muling Wu, Xiaohua Wang, Changze Lv, He-Da Wang, Hu Yao, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19030)  

**Abstract**: Large language models (LLMs) are increasingly expected to tackle complex tasks, driven by their expanding applications and users' growing proficiency in crafting sophisticated prompts. However, as the number of explicitly stated requirements increases (particularly more than 10 constraints), LLMs often struggle to accurately follow such complex instructions. To address this challenge, we propose RECAST, a novel framework for synthesizing datasets where each example incorporates far more constraints than those in existing benchmarks. These constraints are extracted from real-world prompt-response pairs to ensure practical relevance. RECAST enables automatic verification of constraint satisfaction via rule-based validators for quantitative constraints and LLM-based validators for qualitative ones. Using this framework, we construct RECAST-30K, a large-scale, high-quality dataset comprising 30k instances spanning 15 constraint types. Experimental results demonstrate that models fine-tuned on RECAST-30K show substantial improvements in following complex instructions. Moreover, the verifiability provided by RECAST enables the design of reward functions for reinforcement learning, which further boosts model performance on complex and challenging tasks. 

---
# Aligning LLM with human travel choices: a persona-based embedding learning approach 

**Authors**: Tianming Liu, Manzi Li, Yafeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.19003)  

**Abstract**: The advent of large language models (LLMs) presents new opportunities for travel demand modeling. However, behavioral misalignment between LLMs and humans presents obstacles for the usage of LLMs, and existing alignment methods are frequently inefficient or impractical given the constraints of typical travel demand data. This paper introduces a novel framework for aligning LLMs with human travel choice behavior, tailored to the current travel demand data sources. Our framework uses a persona inference and loading process to condition LLMs with suitable prompts to enhance alignment. The inference step establishes a set of base personas from empirical data, and a learned persona loading function driven by behavioral embeddings guides the loading process. We validate our framework on the Swissmetro mode choice dataset, and the results show that our proposed approach significantly outperformed baseline choice models and LLM-based simulation models in predicting both aggregate mode choice shares and individual choice outcomes. Furthermore, we showcase that our framework can generate insights on population behavior through interpretable parameters. Overall, our research offers a more adaptable, interpretable, and resource-efficient pathway to robust LLM-based travel behavior simulation, paving the way to integrate LLMs into travel demand modeling practice in the future. 

---
# Weaver: Interweaving SQL and LLM for Table Reasoning 

**Authors**: Rohit Khoja, Devanshu Gupta, Yanjie Fu, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.18961)  

**Abstract**: Querying tables with unstructured data is challenging due to the presence of text (or image), either embedded in the table or in external paragraphs, which traditional SQL struggles to process, especially for tasks requiring semantic reasoning. While Large Language Models (LLMs) excel at understanding context, they face limitations with long input sequences. Existing approaches that combine SQL and LLMs typically rely on rigid, predefined work-flows, limiting their adaptability to complex queries. To address these issues, we introduce Weaver , a modular pipeline that dynamically integrates SQL and LLMs for table-based question answering (TableQA). Weaver generates a flexible, step-by-step plan that combines SQL for structured data retrieval with LLMs for semantic processing. By decomposing complex queries into manageable subtasks, Weaver improves accuracy and generalization. Our experiments show that Weaver consistently outperforms state-of-the-art methods across four TableQA datasets, reducing both API calls and error rates. 

---
# Co-PatcheR: Collaborative Software Patching with Component(s)-specific Small Reasoning Models 

**Authors**: Yuheng Tang, Hongwei Li, Kaijie Zhu, Michael Yang, Yangruibo Ding, Wenbo Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.18955)  

**Abstract**: Motivated by the success of general-purpose large language models (LLMs) in software patching, recent works started to train specialized patching models. Most works trained one model to handle the end-to-end patching pipeline (including issue localization, patch generation, and patch validation). However, it is hard for a small model to handle all tasks, as different sub-tasks have different workflows and require different expertise. As such, by using a 70 billion model, SOTA methods can only reach up to 41% resolved rate on SWE-bench-Verified. Motivated by the collaborative nature, we propose Co-PatcheR, the first collaborative patching system with small and specialized reasoning models for individual components. Our key technique novelties are the specific task designs and training recipes. First, we train a model for localization and patch generation. Our localization pinpoints the suspicious lines through a two-step procedure, and our generation combines patch generation and critique. We then propose a hybrid patch validation that includes two models for crafting issue-reproducing test cases with and without assertions and judging patch correctness, followed by a majority vote-based patch selection. Through extensive evaluation, we show that Co-PatcheR achieves 46% resolved rate on SWE-bench-Verified with only 3 x 14B models. This makes Co-PatcheR the best patcher with specialized models, requiring the least training resources and the smallest models. We conduct a comprehensive ablation study to validate our recipes, as well as our choice of training data number, model size, and testing-phase scaling strategy. 

---
# SANNet: A Semantic-Aware Agentic AI Networking Framework for Multi-Agent Cross-Layer Coordination 

**Authors**: Yong Xiao, Haoran Zhou, Xubo Li, Yayu Gao, Guangming Shi, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18946)  

**Abstract**: Agentic AI networking (AgentNet) is a novel AI-native networking paradigm that relies on a large number of specialized AI agents to collaborate and coordinate for autonomous decision-making, dynamic environmental adaptation, and complex goal achievement. It has the potential to facilitate real-time network management alongside capabilities for self-configuration, self-optimization, and self-adaptation across diverse and complex networking environments, laying the foundation for fully autonomous networking systems in the future. Despite its promise, AgentNet is still in the early stage of development, and there still lacks an effective networking framework to support automatic goal discovery and multi-agent self-orchestration and task assignment. This paper proposes SANNet, a novel semantic-aware agentic AI networking architecture that can infer the semantic goal of the user and automatically assign agents associated with different layers of a mobile system to fulfill the inferred goal. Motivated by the fact that one of the major challenges in AgentNet is that different agents may have different and even conflicting objectives when collaborating for certain goals, we introduce a dynamic weighting-based conflict-resolving mechanism to address this issue. We prove that SANNet can provide theoretical guarantee in both conflict-resolving and model generalization performance for multi-agent collaboration in dynamic environment. We develop a hardware prototype of SANNet based on the open RAN and 5GS core platform. Our experimental results show that SANNet can significantly improve the performance of multi-agent networking systems, even when agents with conflicting objectives are selected to collaborate for the same goal. 

---
# REACT: Representation Extraction And Controllable Tuning to Overcome Overfitting in LLM Knowledge Editing 

**Authors**: Haitian Zhong, Yuhuan Liu, Ziyang Xu, Guofan Liu, Qiang Liu, Shu Wu, Zhe Zhao, Liang Wang, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18933)  

**Abstract**: Large language model editing methods frequently suffer from overfitting, wherein factual updates can propagate beyond their intended scope, overemphasizing the edited target even when it's contextually inappropriate. To address this challenge, we introduce REACT (Representation Extraction And Controllable Tuning), a unified two-phase framework designed for precise and controllable knowledge editing. In the initial phase, we utilize tailored stimuli to extract latent factual representations and apply Principal Component Analysis with a simple learnbale linear transformation to compute a directional "belief shift" vector for each instance. In the second phase, we apply controllable perturbations to hidden states using the obtained vector with a magnitude scalar, gated by a pre-trained classifier that permits edits only when contextually necessary. Relevant experiments on EVOKE benchmarks demonstrate that REACT significantly reduces overfitting across nearly all evaluation metrics, and experiments on COUNTERFACT and MQuAKE shows that our method preserves balanced basic editing performance (reliability, locality, and generality) under diverse editing scenarios. 

---
# Can Large Language Models Infer Causal Relationships from Real-World Text? 

**Authors**: Ryan Saklad, Aman Chadha, Oleg Pavlov, Raha Moraffah  

**Link**: [PDF](https://arxiv.org/pdf/2505.18931)  

**Abstract**: Understanding and inferring causal relationships from texts is a core aspect of human cognition and is essential for advancing large language models (LLMs) towards artificial general intelligence. Existing work primarily focuses on synthetically generated texts which involve simple causal relationships explicitly mentioned in the text. This fails to reflect the complexities of real-world tasks. In this paper, we investigate whether LLMs are capable of inferring causal relationships from real-world texts. We develop a benchmark drawn from real-world academic literature which includes diverse texts with respect to length, complexity of relationships (different levels of explicitness, number of events, and causal relationships), and domains and sub-domains. To the best of our knowledge, our benchmark is the first-ever real-world dataset for this task. Our experiments on state-of-the-art LLMs evaluated on our proposed benchmark demonstrate significant challenges, with the best-performing model achieving an average F1 score of only 0.477. Analysis reveals common pitfalls: difficulty with implicitly stated information, in distinguishing relevant causal factors from surrounding contextual details, and with connecting causally relevant information spread across lengthy textual passages. By systematically characterizing these deficiencies, our benchmark offers targeted insights for further research into advancing LLM causal reasoning. 

---
# Meta-aware Learning in text-to-SQL Large Language Model 

**Authors**: Wenda Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18929)  

**Abstract**: The advancements of Large language models (LLMs) have provided great opportunities to text-to-SQL tasks to overcome the main challenges to understand complex domain information and complex database structures in business applications. In this paper, we propose a meta-aware learning framework to integrate domain knowledge, database schema, chain-of-thought reasoning processes, and metadata relationships to improve the SQL generation quality. The proposed framework includes four learning strategies: schema-based learning, Chain-of-Thought (CoT) learning, knowledge-enhanced learning, and key information tokenization. This approach provides a comprehensive understanding of database structure and metadata information towards LLM through fine-tuning to improve its performance on SQL generation within business domains. Through two experimental studies, we have demonstrated the superiority of the proposed methods in execution accuracy, multi-task SQL generation capability, and reduction of catastrophic forgetting. 

---
# Stronger Enforcement of Instruction Hierarchy via Augmented Intermediate Representations 

**Authors**: Sanjay Kariyappa, G. Edward Suh  

**Link**: [PDF](https://arxiv.org/pdf/2505.18907)  

**Abstract**: Prompt injection attacks are a critical security vulnerability in large language models (LLMs), allowing attackers to hijack model behavior by injecting malicious instructions within the input context. Recent defense mechanisms have leveraged an Instruction Hierarchy (IH) Signal, often implemented through special delimiter tokens or additive embeddings to denote the privilege level of input tokens. However, these prior works typically inject the IH signal exclusively at the initial input layer, which we hypothesize limits its ability to effectively distinguish the privilege levels of tokens as it propagates through the different layers of the model. To overcome this limitation, we introduce a novel approach that injects the IH signal into the intermediate token representations within the network. Our method augments these representations with layer-specific trainable embeddings that encode the privilege information. Our evaluations across multiple models and training methods reveal that our proposal yields between $1.6\times$ and $9.2\times$ reduction in attack success rate on gradient-based prompt injection attacks compared to state-of-the-art methods, without significantly degrading the model's utility. 

---
# Digital Overconsumption and Waste: A Closer Look at the Impacts of Generative AI 

**Authors**: Vanessa Utz, Steve DiPaola  

**Link**: [PDF](https://arxiv.org/pdf/2505.18894)  

**Abstract**: Generative Artificial Intelligence (AI) systems currently contribute negatively to the production of digital waste, via the associated energy consumption and the related CO2 emissions. At this moment, a discussion is urgently needed on the replication of harmful consumer behavior, such as overconsumption, in the digital space. We outline our previous work on the climate implications of commercially available generative AI systems and the sentiment of generative AI users when confronted with AI-related climate research. We expand on this work via a discussion of digital overconsumption and waste, other related societal impacts, and a possible solution pathway 

---
# Hierarchical-embedding autoencoder with a predictor (HEAP) as efficient architecture for learning long-term evolution of complex multi-scale physical systems 

**Authors**: Alexander Khrabry, Edward Startsev, Andrew Powis, Igor Kaganovich  

**Link**: [PDF](https://arxiv.org/pdf/2505.18857)  

**Abstract**: We propose a novel efficient architecture for learning long-term evolution in complex multi-scale physical systems which is based on the idea of separation of scales. Structures of various scales that dynamically emerge in the system interact with each other only locally. Structures of similar scale can interact directly when they are in contact and indirectly when they are parts of larger structures that interact directly. This enables modeling a multi-scale system in an efficient way, where interactions between small-scale features that are apart from each other do not need to be modeled. The hierarchical fully-convolutional autoencoder transforms the state of a physical system not just into a single embedding layer, as it is done conventionally, but into a series of embedding layers which encode structures of various scales preserving spatial information at a corresponding resolution level. Shallower layers embed smaller structures on a finer grid, while deeper layers embed larger structures on a coarser grid. The predictor advances all embedding layers in sync. Interactions between features of various scales are modeled using a combination of convolutional operators. We compare the performance of our model to variations of a conventional ResNet architecture in application to the Hasegawa-Wakatani turbulence. A multifold improvement in long-term prediction accuracy was observed for crucial statistical characteristics of this system. 

---
# The Theory of the Unique Latent Pattern: A Formal Epistemic Framework for Structural Singularity in Complex Systems 

**Authors**: Mohamed Aly Bouke  

**Link**: [PDF](https://arxiv.org/pdf/2505.18850)  

**Abstract**: This paper introduces the Theory of the Unique Latent Pattern (ULP), a formal epistemic framework that redefines the origin of apparent complexity in dynamic systems. Rather than attributing unpredictability to intrinsic randomness or emergent nonlinearity, ULP asserts that every analyzable system is governed by a structurally unique, deterministic generative mechanism, one that remains hidden not due to ontological indeterminacy, but due to epistemic constraints. The theory is formalized using a non-universal generative mapping \( \mathcal{F}_S(P_S, t) \), where each system \( S \) possesses its own latent structure \( P_S \), irreducible and non-replicable across systems. Observed irregularities are modeled as projections of this generative map through observer-limited interfaces, introducing epistemic noise \( \varepsilon_S(t) \) as a measure of incomplete access. By shifting the locus of uncertainty from the system to the observer, ULP reframes chaos as a context-relative failure of representation. We contrast this position with foundational paradigms in chaos theory, complexity science, and statistical learning. While they assume or model shared randomness or collective emergence, ULP maintains that every instance harbors a singular structural identity. Although conceptual, the theory satisfies the criterion of falsifiability in the Popperian sense, it invites empirical challenge by asserting that no two systems governed by distinct latent mechanisms will remain indistinguishable under sufficient resolution. This opens avenues for structurally individuated models in AI, behavioral inference, and epistemic diagnostics. 

---
# Signal, Image, or Symbolic: Exploring the Best Input Representation for Electrocardiogram-Language Models Through a Unified Framework 

**Authors**: William Han, Chaojing Duan, Zhepeng Cen, Yihang Yao, Xiaoyu Song, Atharva Mhaskar, Dylan Leong, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18847)  

**Abstract**: Recent advances have increasingly applied large language models (LLMs) to electrocardiogram (ECG) interpretation, giving rise to Electrocardiogram-Language Models (ELMs). Conditioned on an ECG and a textual query, an ELM autoregressively generates a free-form textual response. Unlike traditional classification-based systems, ELMs emulate expert cardiac electrophysiologists by issuing diagnoses, analyzing waveform morphology, identifying contributing factors, and proposing patient-specific action plans. To realize this potential, researchers are curating instruction-tuning datasets that pair ECGs with textual dialogues and are training ELMs on these resources. Yet before scaling ELMs further, there is a fundamental question yet to be explored: What is the most effective ECG input representation? In recent works, three candidate representations have emerged-raw time-series signals, rendered images, and discretized symbolic sequences. We present the first comprehensive benchmark of these modalities across 6 public datasets and 5 evaluation metrics. We find symbolic representations achieve the greatest number of statistically significant wins over both signal and image inputs. We further ablate the LLM backbone, ECG duration, and token budget, and we evaluate robustness to signal perturbations. We hope that our findings offer clear guidance for selecting input representations when developing the next generation of ELMs. 

---
# LiteCUA: Computer as MCP Server for Computer-Use Agent on AIOS 

**Authors**: Kai Mei, Xi Zhu, Hang Gao, Shuhang Lin, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18829)  

**Abstract**: We present AIOS 1.0, a novel platform designed to advance computer-use agent (CUA) capabilities through environmental contextualization. While existing approaches primarily focus on building more powerful agent frameworks or enhancing agent models, we identify a fundamental limitation: the semantic disconnect between how language models understand the world and how computer interfaces are structured. AIOS 1.0 addresses this challenge by transforming computers into contextual environments that language models can natively comprehend, implementing a Model Context Protocol (MCP) server architecture to abstract computer states and actions. This approach effectively decouples interface complexity from decision complexity, enabling agents to reason more effectively about computing environments. To demonstrate our platform's effectiveness, we introduce LiteCUA, a lightweight computer-use agent built on AIOS 1.0 that achieves a 14.66% success rate on the OSWorld benchmark, outperforming several specialized agent frameworks despite its simple architecture. Our results suggest that contextualizing computer environments for language models represents a promising direction for developing more capable computer-use agents and advancing toward AI that can interact with digital systems. The source code of LiteCUA is available at this https URL, and it is also integrated into the AIOS main branch as part of AIOS at this https URL. 

---
# AdaCtrl: Towards Adaptive and Controllable Reasoning via Difficulty-Aware Budgeting 

**Authors**: Shijue Huang, Hongru Wang, Wanjun Zhong, Zhaochen Su, Jiazhan Feng, Bowen Cao, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2505.18822)  

**Abstract**: Modern large reasoning models demonstrate impressive problem-solving capabilities by employing sophisticated reasoning strategies. However, they often struggle to balance efficiency and effectiveness, frequently generating unnecessarily lengthy reasoning chains for simple problems. In this work, we propose AdaCtrl, a novel framework to support both difficulty-aware adaptive reasoning budget allocation and explicit user control over reasoning depth. AdaCtrl dynamically adjusts its reasoning length based on self-assessed problem difficulty, while also allowing users to manually control the budget to prioritize either efficiency or effectiveness. This is achieved through a two-stage training pipeline: an initial cold-start fine-tuning phase to instill the ability to self-aware difficulty and adjust reasoning budget, followed by a difficulty-aware reinforcement learning (RL) stage that refines the model's adaptive reasoning strategies and calibrates its difficulty assessments based on its evolving capabilities during online training. To enable intuitive user interaction, we design explicit length-triggered tags that function as a natural interface for budget control. Empirical results show that AdaCtrl adapts reasoning length based on estimated difficulty, compared to the standard training baseline that also incorporates fine-tuning and RL, it yields performance improvements and simultaneously reduces response length by 10.06% and 12.14% on the more challenging AIME2024 and AIME2025 datasets, which require elaborate reasoning, and by 62.05% and 91.04% on the MATH500 and GSM8K datasets, where more concise responses are sufficient. Furthermore, AdaCtrl enables precise user control over the reasoning budget, allowing for tailored responses to meet specific needs. 

---
# Mitigating Deceptive Alignment via Self-Monitoring 

**Authors**: Jiaming Ji, Wenqi Chen, Kaile Wang, Donghai Hong, Sitong Fang, Boyuan Chen, Jiayi Zhou, Juntao Dai, Sirui Han, Yike Guo, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18807)  

**Abstract**: Modern large language models rely on chain-of-thought (CoT) reasoning to achieve impressive performance, yet the same mechanism can amplify deceptive alignment, situations in which a model appears aligned while covertly pursuing misaligned goals. Existing safety pipelines treat deception as a black-box output to be filtered post-hoc, leaving the model free to scheme during its internal reasoning. We ask: Can deception be intercepted while the model is thinking? We answer this question, the first framework that embeds a Self-Monitor inside the CoT process itself, named CoT Monitor+. During generation, the model produces (i) ordinary reasoning steps and (ii) an internal self-evaluation signal trained to flag and suppress misaligned strategies. The signal is used as an auxiliary reward in reinforcement learning, creating a feedback loop that rewards honest reasoning and discourages hidden goals. To study deceptive alignment systematically, we introduce DeceptionBench, a five-category benchmark that probes covert alignment-faking, sycophancy, etc. We evaluate various LLMs and show that unrestricted CoT roughly aggravates the deceptive tendency. In contrast, CoT Monitor+ cuts deceptive behaviors by 43.8% on average while preserving task accuracy. Further, when the self-monitor signal replaces an external weak judge in RL fine-tuning, models exhibit substantially fewer obfuscated thoughts and retain transparency. Our project website can be found at this http URL 

---
# The Quest for Efficient Reasoning: A Data-Centric Benchmark to CoT Distillation 

**Authors**: Ruichen Zhang, Rana Muhammad Shahroz Khan, Zhen Tan, Dawei Li, Song Wang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18759)  

**Abstract**: Data-centric distillation, including data augmentation, selection, and mixing, offers a promising path to creating smaller, more efficient student Large Language Models (LLMs) that retain strong reasoning abilities. However, there still lacks a comprehensive benchmark to systematically assess the effect of each distillation approach. This paper introduces DC-CoT, the first data-centric benchmark that investigates data manipulation in chain-of-thought (CoT) distillation from method, model and data perspectives. Utilizing various teacher models (e.g., o4-mini, Gemini-Pro, Claude-3.5) and student architectures (e.g., 3B, 7B parameters), we rigorously evaluate the impact of these data manipulations on student model performance across multiple reasoning datasets, with a focus on in-distribution (IID) and out-of-distribution (OOD) generalization, and cross-domain transfer. Our findings aim to provide actionable insights and establish best practices for optimizing CoT distillation through data-centric techniques, ultimately facilitating the development of more accessible and capable reasoning models. The dataset can be found at this https URL, while our code is shared in this https URL. 

---
# $C^3$-Bench: The Things Real Disturbing LLM based Agent in Multi-Tasking 

**Authors**: Peijie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18746)  

**Abstract**: Agents based on large language models leverage tools to modify environments, revolutionizing how AI interacts with the physical world. Unlike traditional NLP tasks that rely solely on historical dialogue for responses, these agents must consider more complex factors, such as inter-tool relationships, environmental feedback and previous decisions, when making choices. Current research typically evaluates agents via multi-turn dialogues. However, it overlooks the influence of these critical factors on agent behavior. To bridge this gap, we present an open-source and high-quality benchmark $C^3$-Bench. This benchmark integrates attack concepts and applies univariate analysis to pinpoint key elements affecting agent robustness. In concrete, we design three challenges: navigate complex tool relationships, handle critical hidden information and manage dynamic decision paths. Complementing these challenges, we introduce fine-grained metrics, innovative data collection algorithms and reproducible evaluation methods. Extensive experiments are conducted on 49 mainstream agents, encompassing general fast-thinking, slow-thinking and domain-specific models. We observe that agents have significant shortcomings in handling tool dependencies, long context information dependencies and frequent policy-type switching. In essence, $C^3$-Bench aims to expose model vulnerabilities through these challenges and drive research into the interpretability of agent performance. The benchmark is publicly available at this https URL. 

---
# AI-Researcher: Autonomous Scientific Innovation 

**Authors**: Jiabin Tang, Lianghao Xia, Zhonghang Li, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18705)  

**Abstract**: The powerful reasoning capabilities of Large Language Models (LLMs) in mathematics and coding, combined with their ability to automate complex tasks through agentic frameworks, present unprecedented opportunities for accelerating scientific innovation. In this paper, we introduce AI-Researcher, a fully autonomous research system that transforms how AI-driven scientific discovery is conducted and evaluated. Our framework seamlessly orchestrates the complete research pipeline--from literature review and hypothesis generation to algorithm implementation and publication-ready manuscript preparation--with minimal human intervention. To rigorously assess autonomous research capabilities, we develop Scientist-Bench, a comprehensive benchmark comprising state-of-the-art papers across diverse AI research domains, featuring both guided innovation and open-ended exploration tasks. Through extensive experiments, we demonstrate that AI-Researcher achieves remarkable implementation success rates and produces research papers that approach human-level quality. This work establishes new foundations for autonomous scientific innovation that can complement human researchers by systematically exploring solution spaces beyond cognitive limitations. 

---
# AI for Regulatory Affairs: Balancing Accuracy, Interpretability, and Computational Cost in Medical Device Classification 

**Authors**: Yu Han, Aaron Ceross, Jeroen H. M. Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.18695)  

**Abstract**: Regulatory affairs, which sits at the intersection of medicine and law, can benefit significantly from AI-enabled automation. Classification task is the initial step in which manufacturers position their products to regulatory authorities, and it plays a critical role in determining market access, regulatory scrutiny, and ultimately, patient safety. In this study, we investigate a broad range of AI models -- including traditional machine learning (ML) algorithms, deep learning architectures, and large language models -- using a regulatory dataset of medical device descriptions. We evaluate each model along three key dimensions: accuracy, interpretability, and computational cost. 

---
# AI-Driven Climate Policy Scenario Generation for Sub-Saharan Africa 

**Authors**: Rafiu Adekoya Badekale, Adewale Akinfaderin  

**Link**: [PDF](https://arxiv.org/pdf/2505.18694)  

**Abstract**: Climate policy scenario generation and evaluation have traditionally relied on integrated assessment models (IAMs) and expert-driven qualitative analysis. These methods enable stakeholders, such as policymakers and researchers, to anticipate impacts, plan governance strategies, and develop mitigation measures. However, traditional methods are often time-intensive, reliant on simple extrapolations of past trends, and limited in capturing the complex and interconnected nature of energy and climate issues. With the advent of artificial intelligence (AI), particularly generative AI models trained on vast datasets, these limitations can be addressed, ensuring robustness even under limited data conditions. In this work, we explore the novel method that employs generative AI, specifically large language models (LLMs), to simulate climate policy scenarios for Sub-Saharan Africa. These scenarios focus on energy transition themes derived from the historical United Nations Climate Change Conference (COP) documents. By leveraging generative models, the project aims to create plausible and diverse policy scenarios that align with regional climate goals and energy challenges. Given limited access to human evaluators, automated techniques were employed for scenario evaluation. We generated policy scenarios using the llama3.2-3B model. Of the 34 generated responses, 30 (88%) passed expert validation, accurately reflecting the intended impacts provided in the corresponding prompts. We compared these validated responses against assessments from a human climate expert and two additional LLMs (gemma2-2B and mistral-7B). Our structured, embedding-based evaluation framework shows that generative AI effectively generate scenarios that are coherent, relevant, plausible, and diverse. This approach offers a transformative tool for climate policy planning in data-constrained regions. 

---
# TrajMoE: Spatially-Aware Mixture of Experts for Unified Human Mobility Modeling 

**Authors**: Chonghua Han, Yuan Yuan, Kaiyan Chen, Jingtao Ding, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.18670)  

**Abstract**: Modeling human mobility across diverse cities is essential for applications such as urban planning, transportation optimization, and personalized services. However, generalization remains challenging due to heterogeneous spatial representations and mobility patterns across cities. Existing methods typically rely on numerical coordinates or require training city-specific models, limiting their scalability and transferability. We propose TrajMoE, a unified and scalable model for cross-city human mobility modeling. TrajMoE addresses two key challenges: (1) inconsistent spatial semantics across cities, and (2) diverse urban mobility patterns. To tackle these, we begin by designing a spatial semantic encoder that learns transferable location representations from POI-based functional semantics and visit patterns. Furthermore, we design a Spatially-Aware Mixture-of-Experts (SAMoE) Transformer that injects structured priors into experts specialized in distinct mobility semantics, along with a shared expert to capture city-invariant patterns and enable adaptive cross-city generalization. Extensive experiments demonstrate that TrajMoE achieves up to 27% relative improvement over competitive mobility foundation models after only one epoch of fine-tuning, and consistently outperforms full-data baselines using merely 5% of target city data. These results establish TrajMoE as a significant step toward realizing a truly generalizable, transferable, and pretrainable foundation model for human mobility. 

---
# MLLMs are Deeply Affected by Modality Bias 

**Authors**: Xu Zheng, Chenfei Liao, Yuqian Fu, Kaiyu Lei, Yuanhuiyi Lyu, Lutao Jiang, Bin Ren, Jialei Chen, Jiawen Wang, Chengxin Li, Linfeng Zhang, Danda Pani Paudel, Xuanjing Huang, Yu-Gang Jiang, Nicu Sebe, Dacheng Tao, Luc Van Gool, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18657)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have shown promising results in integrating diverse modalities such as texts and images. MLLMs are heavily influenced by modality bias, often relying on language while under-utilizing other modalities like visual inputs. This position paper argues that MLLMs are deeply affected by modality bias. Firstly, we diagnose the current state of modality bias, highlighting its manifestations across various tasks. Secondly, we propose a systematic research road-map related to modality bias in MLLMs. Thirdly, we identify key factors of modality bias in MLLMs and offer actionable suggestions for future research to mitigate it. To substantiate these findings, we conduct experiments that demonstrate the influence of each factor: 1. Data Characteristics: Language data is compact and abstract, while visual data is redundant and complex, creating an inherent imbalance in learning dynamics. 2. Imbalanced Backbone Capabilities: The dominance of pretrained language models in MLLMs leads to overreliance on language and neglect of visual information. 3. Training Objectives: Current objectives often fail to promote balanced cross-modal alignment, resulting in shortcut learning biased toward language. These findings highlight the need for balanced training strategies and model architectures to better integrate multiple modalities in MLLMs. We call for interdisciplinary efforts to tackle these challenges and drive innovation in MLLM research. Our work provides a fresh perspective on modality bias in MLLMs and offers insights for developing more robust and generalizable multimodal systems-advancing progress toward Artificial General Intelligence. 

---
# Riverine Flood Prediction and Early Warning in Mountainous Regions using Artificial Intelligence 

**Authors**: Haleema Bibi, Sadia Saleem, Zakia Jalil, Muhammad Nasir, Tahani Alsubait  

**Link**: [PDF](https://arxiv.org/pdf/2505.18645)  

**Abstract**: Flooding is the most devastating phenomenon occurring globally, particularly in mountainous regions, risk dramatically increases due to complex terrains and extreme climate changes. These situations are damaging livelihoods, agriculture, infrastructure, and human lives. This study uses the Kabul River between Pakistan and Afghanistan as a case study to reflect the complications of flood forecasting in transboundary basins. The challenges in obtaining upstream data impede the efficacy of flood control measures and early warning systems, a common global problem in similar basins. Utilizing satellite-based climatic data, this study applied numerous advanced machine-learning and deep learning models, such as Support Vector Machines (SVM), XGBoost, and Artificial Neural Networks (ANN), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRU) to predict daily and multi-step river flow. The LSTM network outperformed other models, achieving the highest R2 value of 0.96 and the lowest RMSE value of 140.96 m3/sec. The time series LSTM and GRU network models, utilized for short-term forecasts of up to five days, performed significantly. However, the accuracy declined beyond the fourth day, highlighting the need for longer-term historical datasets for reliable long-term flood predictions. The results of the study are directly aligned with Sustainable Development Goals 6, 11, 13, and 15, facilitating disaster and water management, timely evacuations, improved preparedness, and effective early warning. 

---
# Mind The Gap: Deep Learning Doesn't Learn Deeply 

**Authors**: Lucas Saldyt, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2505.18623)  

**Abstract**: This paper aims to understand how neural networks learn algorithmic reasoning by addressing two questions: How faithful are learned algorithms when they are effective, and why do neural networks fail to learn effective algorithms otherwise? To answer these questions, we use neural compilation, a technique that directly encodes a source algorithm into neural network parameters, enabling the network to compute the algorithm exactly. This enables comparison between compiled and conventionally learned parameters, intermediate vectors, and behaviors. This investigation is crucial for developing neural networks that robustly learn complexalgorithms from data. Our analysis focuses on graph neural networks (GNNs), which are naturally aligned with algorithmic reasoning tasks, specifically our choices of BFS, DFS, and Bellman-Ford, which cover the spectrum of effective, faithful, and ineffective learned algorithms. Commonly, learning algorithmic reasoning is framed as induction over synthetic data, where a parameterized model is trained on inputs, traces, and outputs produced by an underlying ground truth algorithm. In contrast, we introduce a neural compilation method for GNNs, which sets network parameters analytically, bypassing training. Focusing on GNNs leverages their alignment with algorithmic reasoning, extensive algorithmic induction literature, and the novel application of neural compilation to GNNs. Overall, this paper aims to characterize expressability-trainability gaps - a fundamental shortcoming in learning algorithmic reasoning. We hypothesize that inductive learning is most effective for parallel algorithms contained within the computational class \texttt{NC}. 

---
# Knowledge Retrieval in LLM Gaming: A Shift from Entity-Centric to Goal-Oriented Graphs 

**Authors**: Jonathan Leung, Yongjie Wang, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18607)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive general capabilities but often struggle with step-by-step reasoning, especially in complex applications such as games. While retrieval-augmented methods like GraphRAG attempt to bridge this gap through cross-document extraction and indexing, their fragmented entity-relation graphs and overly dense local connectivity hinder the construction of coherent reasoning. In this paper, we propose a novel framework based on Goal-Oriented Graphs (GoGs), where each node represents a goal and its associated attributes, and edges encode logical dependencies between goals. This structure enables explicit retrieval of reasoning paths by first identifying high-level goals and recursively retrieving their subgoals, forming coherent reasoning chains to guide LLM prompting. Our method significantly enhances the reasoning ability of LLMs in game-playing tasks, as demonstrated by extensive experiments on the Minecraft testbed, outperforming GraphRAG and other baselines. 

---
# Doc-CoB: Enhancing Multi-Modal Document Understanding with Visual Chain-of-Boxes Reasoning 

**Authors**: Ye Mo, Zirui Shao, Kai Ye, Xianwei Mao, Bo Zhang, Hangdi Xing, Peng Ye, Gang Huang, Kehan Chen, Zhou Huan, Zixu Yan, Sheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.18603)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in document understanding. However, the information-dense nature of document images still poses challenges, as most queries depend on only a few relevant regions, with the rest being redundant. Existing one-pass MLLMs process entire document images without considering query relevance, often failing to focus on critical regions and producing unfaithful responses. Inspired by the human coarse-to-fine reading pattern, we introduce Doc-CoB (Chain-of-Box), a simple-yet-effective mechanism that integrates human-style visual reasoning into MLLM without modifying its architecture. Our method allows the model to autonomously select the set of regions (boxes) most relevant to the query, and then focus attention on them for further understanding. We first design a fully automatic pipeline, integrating a commercial MLLM with a layout analyzer, to generate 249k training samples with intermediate visual reasoning supervision. Then we incorporate two enabling tasks that improve box identification and box-query reasoning, which together enhance document understanding. Extensive experiments on seven benchmarks with four popular models show that Doc-CoB significantly improves performance, demonstrating its effectiveness and wide applicability. All code, data, and models will be released publicly. 

---
# LLMs for Supply Chain Management 

**Authors**: Haojie Wang, Jiuyun Jiang, L. Jeff Hong, Guangxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18597)  

**Abstract**: The development of large language models (LLMs) has provided new tools for research in supply chain management (SCM). In this paper, we introduce a retrieval-augmented generation (RAG) framework that dynamically integrates external knowledge into the inference process, and develop a domain-specialized SCM LLM, which demonstrates expert-level competence by passing standardized SCM examinations and beer game tests. We further employ the use of LLMs to conduct horizontal and vertical supply chain games, in order to analyze competition and cooperation within supply chains. Our experiments show that RAG significantly improves performance on SCM tasks. Moreover, game-theoretic analysis reveals that the LLM can reproduce insights from the classical SCM literature, while also uncovering novel behaviors and offering fresh perspectives on phenomena such as the bullwhip effect. This paper opens the door for exploring cooperation and competition for complex supply chain network through the lens of LLMs. 

---
# RvLLM: LLM Runtime Verification with Domain Knowledge 

**Authors**: Yedi Zhang, Sun Yi Emma, Annabelle Lee Jia En, Annabelle Lee Jia En, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18585)  

**Abstract**: Large language models (LLMs) have emerged as a dominant AI paradigm due to their exceptional text understanding and generation capabilities. However, their tendency to generate inconsistent or erroneous outputs challenges their reliability, especially in high-stakes domains requiring accuracy and trustworthiness. Existing research primarily focuses on detecting and mitigating model misbehavior in general-purpose scenarios, often overlooking the potential of integrating domain-specific knowledge. In this work, we advance misbehavior detection by incorporating domain knowledge. The core idea is to design a general specification language that enables domain experts to customize domain-specific predicates in a lightweight and intuitive manner, supporting later runtime verification of LLM outputs. To achieve this, we design a novel specification language, ESL, and introduce a runtime verification framework, RvLLM, to validate LLM output against domain-specific constraints defined in ESL. We evaluate RvLLM on three representative tasks: violation detection against Singapore Rapid Transit Systems Act, numerical comparison, and inequality solving. Experimental results demonstrate that RvLLM effectively detects erroneous outputs across various LLMs in a lightweight and flexible manner. The results reveal that despite their impressive capabilities, LLMs remain prone to low-level errors due to limited interpretability and a lack of formal guarantees during inference, and our framework offers a potential long-term solution by leveraging expert domain knowledge to rigorously and efficiently verify LLM outputs. 

---
# Response Uncertainty and Probe Modeling: Two Sides of the Same Coin in LLM Interpretability? 

**Authors**: Yongjie Wang, Yibo Wang, Xin Zhou, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18575)  

**Abstract**: Probing techniques have shown promise in revealing how LLMs encode human-interpretable concepts, particularly when applied to curated datasets. However, the factors governing a dataset's suitability for effective probe training are not well-understood. This study hypothesizes that probe performance on such datasets reflects characteristics of both the LLM's generated responses and its internal feature space. Through quantitative analysis of probe performance and LLM response uncertainty across a series of tasks, we find a strong correlation: improved probe performance consistently corresponds to a reduction in response uncertainty, and vice versa. Subsequently, we delve deeper into this correlation through the lens of feature importance analysis. Our findings indicate that high LLM response variance is associated with a larger set of important features, which poses a greater challenge for probe models and often results in diminished performance. Moreover, leveraging the insights from response uncertainty analysis, we are able to identify concrete examples where LLM representations align with human knowledge across diverse domains, offering additional evidence of interpretable reasoning in LLMs. 

---
# Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models 

**Authors**: Min Cheng, Fatemeh Doudi, Dileep Kalathil, Mohammad Ghavamzadeh, Panganamala R. Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.18547)  

**Abstract**: Reinforcement learning (RL) algorithms have been used recently to align diffusion models with downstream objectives such as aesthetic quality and text-image consistency by fine-tuning them to maximize a single reward function under a fixed KL regularization. However, this approach is inherently restrictive in practice, where alignment must balance multiple, often conflicting objectives. Moreover, user preferences vary across prompts, individuals, and deployment contexts, with varying tolerances for deviation from a pre-trained base model. We address the problem of inference-time multi-preference alignment: given a set of basis reward functions and a reference KL regularization strength, can we design a fine-tuning procedure so that, at inference time, it can generate images aligned with any user-specified linear combination of rewards and regularization, without requiring additional fine-tuning? We propose Diffusion Blend, a novel approach to solve inference-time multi-preference alignment by blending backward diffusion processes associated with fine-tuned models, and we instantiate this approach with two algorithms: DB-MPA for multi-reward alignment and DB-KLA for KL regularization control. Extensive experiments show that Diffusion Blend algorithms consistently outperform relevant baselines and closely match or exceed the performance of individually fine-tuned models, enabling efficient, user-driven alignment at inference-time. The code is available at this https URL}{this http URL. 

---
# RoleRAG: Enhancing LLM Role-Playing via Graph Guided Retrieval 

**Authors**: Yongjie Wang, Jonathan Leung, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18541)  

**Abstract**: Large Language Models (LLMs) have shown promise in character imitation, enabling immersive and engaging conversations. However, they often generate content that is irrelevant or inconsistent with a character's background. We attribute these failures to: (1) the inability to accurately recall character-specific knowledge due to entity ambiguity, and (2) a lack of awareness of the character's cognitive boundaries. To address these issues, we propose RoleRAG, a retrieval-based framework that integrates efficient entity disambiguation for knowledge indexing with a boundary-aware retriever for extracting contextually appropriate information from a structured knowledge graph. Experiments on role-playing benchmarks show that RoleRAG's calibrated retrieval helps both general-purpose and role-specific LLMs better align with character knowledge and reduce hallucinated responses. 

---
# Generative RLHF-V: Learning Principles from Multi-modal Human Preference 

**Authors**: Jiayi Zhou, Jiaming Ji, Boyuan Chen, Jiapeng Sun, Wenqi Chen, Donghai Hong, Sirui Han, Yike Guo, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18531)  

**Abstract**: Training multi-modal large language models (MLLMs) that align with human intentions is a long-term challenge. Traditional score-only reward models for alignment suffer from low accuracy, weak generalization, and poor interpretability, blocking the progress of alignment methods, e.g., reinforcement learning from human feedback (RLHF). Generative reward models (GRMs) leverage MLLMs' intrinsic reasoning capabilities to discriminate pair-wise responses, but their pair-wise paradigm makes it hard to generalize to learnable rewards. We introduce Generative RLHF-V, a novel alignment framework that integrates GRMs with multi-modal RLHF. We propose a two-stage pipeline: $\textbf{multi-modal generative reward modeling from RL}$, where RL guides GRMs to actively capture human intention, then predict the correct pair-wise scores; and $\textbf{RL optimization from grouped comparison}$, which enhances multi-modal RL scoring precision by grouped responses comparison. Experimental results demonstrate that, besides out-of-distribution generalization of RM discrimination, our framework improves 4 MLLMs' performance across 7 benchmarks by $18.1\%$, while the baseline RLHF is only $5.3\%$. We further validate that Generative RLHF-V achieves a near-linear improvement with an increasing number of candidate responses. Our code and models can be found at this https URL. 

---
# LiSTEN: Learning Soft Token Embeddings for Neural Audio LLMs 

**Authors**: Pooneh Mousavi, Shubham Gupta, Cem Subakan, Mirco Ravanelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.18517)  

**Abstract**: Foundation models based on large language models (LLMs) have shown great success in handling various tasks and modalities. However, adapting these models for general-purpose audio-language tasks is challenging due to differences in acoustic environments and task variations. In this work, we introduce LiSTEN Learning Soft Token Embeddings for Neural Audio LLMs), a framework for adapting LLMs to speech and audio tasks. LiSTEN uses a dynamic prompt selection strategy with learnable key-value pairs, allowing the model to balance general and task-specific knowledge while avoiding overfitting in a multitask setting. Our approach reduces dependence on large-scale ASR or captioning datasets, achieves competitive performance with fewer trainable parameters, and simplifies training by using a single-stage process. Additionally, LiSTEN enhances interpretability by analyzing the diversity and overlap of selected prompts across different tasks. 

---
# Knowledge Grafting of Large Language Models 

**Authors**: Guodong Du, Xuanning Zhou, Junlin Li, Zhuo Li, Zesheng Shi, Wanyu Lin, Ho-Kin Tang, Xiucheng Li, Fangming Liu, Wenya Wang, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.18502)  

**Abstract**: Cross-capability transfer is a key challenge in large language model (LLM) research, with applications in multi-task integration, model compression, and continual learning. Recent works like FuseLLM and FuseChat have demonstrated the potential of transferring multiple model capabilities to lightweight models, enhancing adaptability and efficiency, which motivates our investigation into more efficient cross-capability transfer methods. However, existing approaches primarily focus on small, homogeneous models, limiting their applicability. For large, heterogeneous models, knowledge distillation with full-parameter fine-tuning often overlooks the student model's intrinsic capacity and risks catastrophic forgetting, while PEFT methods struggle to effectively absorb knowledge from source LLMs. To address these issues, we introduce GraftLLM, a novel method that stores source model capabilities in a target model with SkillPack format. This approach preserves general capabilities, reduces parameter conflicts, and supports forget-free continual learning and model fusion. We employ a module-aware adaptive compression strategy to compress parameter updates, ensuring efficient storage while maintaining task-specific knowledge. The resulting SkillPack serves as a compact and transferable knowledge carrier, ideal for heterogeneous model fusion and continual learning. Experiments across various scenarios demonstrate that GraftLLM outperforms existing techniques in knowledge transfer, knowledge fusion, and forget-free learning, providing a scalable and efficient solution for cross-capability transfer. The code is publicly available at: this https URL. 

---
# Enumerate-Conjecture-Prove: Formally Solving Answer-Construction Problems in Math Competitions 

**Authors**: Jialiang Sun, Yuzhi Tang, Ao Li, Chris J. Maddison, Kuldeep S. Meel  

**Link**: [PDF](https://arxiv.org/pdf/2505.18492)  

**Abstract**: Mathematical reasoning lies at the heart of artificial intelligence, underpinning applications in education, program verification, and research-level mathematical discovery. Mathematical competitions, in particular, present two challenging problem types: theorem-proving, requiring rigorous proofs of stated conclusions, and answer-construction, involving hypothesizing and formally verifying mathematical objects. Large Language Models (LLMs) effectively generate creative candidate answers but struggle with formal verification, while symbolic provers ensure rigor but cannot efficiently handle creative conjecture generation. We introduce the Enumerate-Conjecture-Prove (ECP) framework, a modular neuro-symbolic method integrating LLM-based enumeration and pattern-driven conjecturing with formal theorem proving. We present ConstructiveBench, a dataset of 3,431 answer-construction problems in various math competitions with verified Lean formalizations. On the ConstructiveBench dataset, ECP improves the accuracy of answer construction from the Chain-of-Thought (CoT) baseline of 14.54% to 45.06% with the gpt-4.1-mini model. Moreover, combining with ECP's constructed answers, the state-of-the-art DeepSeek-Prover-V2-7B model generates correct proofs for 858 of the 3,431 constructive problems in Lean, achieving 25.01% accuracy, compared to 9.86% for symbolic-only baselines. Our code and dataset are publicly available at GitHub and HuggingFace, respectively. 

---
# Retrieval Augmented Decision-Making: A Requirements-Driven, Multi-Criteria Framework for Structured Decision Support 

**Authors**: Hongjia Wu, Hongxin Zhang, Wei Chen, Jiazhi Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.18483)  

**Abstract**: Various industries have produced a large number of documents such as industrial plans, technical guidelines, and regulations that are structurally complex and content-wise fragmented. This poses significant challenges for experts and decision-makers in terms of retrieval and understanding. Although existing LLM-based Retrieval-Augmented Generation methods can provide context-related suggestions, they lack quantitative weighting and traceable reasoning paths, making it difficult to offer multi-level and transparent decision support. To address this issue, this paper proposes the RAD method, which integrates Multi-Criteria Decision Making with the semantic understanding capabilities of LLMs. The method automatically extracts key criteria from industry documents, builds a weighted hierarchical decision model, and generates structured reports under model guidance. The RAD framework introduces explicit weight assignment and reasoning chains in decision generation to ensure accuracy, completeness, and traceability. Experiments show that in various decision-making tasks, the decision reports generated by RAD significantly outperform existing methods in terms of detail, rationality, and structure, demonstrating its application value and potential in complex decision support scenarios. 

---
# Chemical classification program synthesis using generative artificial intelligence 

**Authors**: Christopher J. Mungall, Adnan Malik, Daniel R. Korn, Justin T. Reese, Noel M. O'Boyle, Noel, Janna Hastings  

**Link**: [PDF](https://arxiv.org/pdf/2505.18470)  

**Abstract**: Accurately classifying chemical structures is essential for cheminformatics and bioinformatics, including tasks such as identifying bioactive compounds of interest, screening molecules for toxicity to humans, finding non-organic compounds with desirable material properties, or organizing large chemical libraries for drug discovery or environmental monitoring. However, manual classification is labor-intensive and difficult to scale to large chemical databases. Existing automated approaches either rely on manually constructed classification rules, or the use of deep learning methods that lack explainability.
This work presents an approach that uses generative artificial intelligence to automatically write chemical classifier programs for classes in the Chemical Entities of Biological Interest (ChEBI) database. These programs can be used for efficient deterministic run-time classification of SMILES structures, with natural language explanations. The programs themselves constitute an explainable computable ontological model of chemical class nomenclature, which we call the ChEBI Chemical Class Program Ontology (C3PO).
We validated our approach against the ChEBI database, and compared our results against state of the art deep learning models. We also demonstrate the use of C3PO to classify out-of-distribution examples taken from metabolomics repositories and natural product databases. We also demonstrate the potential use of our approach to find systematic classification errors in existing chemical databases, and show how an ensemble artificial intelligence approach combining generated ontologies, automated literature search, and multimodal vision models can be used to pinpoint potential errors requiring expert validation 

---
# Pedagogy-R1: Pedagogically-Aligned Reasoning Model with Balanced Educational Benchmark 

**Authors**: Unggi Lee, Jaeyong Lee, Jiyeong Bae, Yeil Jeong, Junbo Koh, Gyeonggeon Lee, Gunho Lee, Taekyung Ahn, Hyeoncheol Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.18467)  

**Abstract**: Recent advances in large reasoning models (LRMs) show strong performance in structured domains such as mathematics and programming; however, they often lack pedagogical coherence and realistic teaching behaviors. To bridge this gap, we introduce Pedagogy-R1, a framework that adapts LRMs for classroom use through three innovations: (1) a distillation-based pipeline that filters and refines model outputs for instruction-tuning, (2) the Well-balanced Educational Benchmark (WBEB), which evaluates performance across subject knowledge, pedagogical knowledge, tracing, essay scoring, and teacher decision-making, and (3) a Chain-of-Pedagogy (CoP) prompting strategy for generating and eliciting teacher-style reasoning. Our mixed-method evaluation combines quantitative metrics with qualitative analysis, providing the first systematic assessment of LRMs' pedagogical strengths and limitations. 

---
# EdgeAgentX: A Novel Framework for Agentic AI at the Edge in Military Communication Networks 

**Authors**: Abir Ray  

**Link**: [PDF](https://arxiv.org/pdf/2505.18457)  

**Abstract**: This paper introduces EdgeAgentX, a novel framework integrating federated learning (FL), multi-agent reinforcement learning (MARL), and adversarial defense mechanisms, tailored for military communication networks. EdgeAgentX significantly improves autonomous decision-making, reduces latency, enhances throughput, and robustly withstands adversarial disruptions, as evidenced by comprehensive simulations. 

---
# Advertising in AI systems: Society must be vigilant 

**Authors**: Menghua Wu, Yujia Bao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18425)  

**Abstract**: AI systems have increasingly become our gateways to the Internet. We argue that just as advertising has driven the monetization of web search and social media, so too will commercial incentives shape the content served by AI. Unlike traditional media, however, the outputs of these systems are dynamic, personalized, and lack clear provenance -- raising concerns for transparency and regulation. In this paper, we envision how commercial content could be delivered through generative AI-based systems. Based on the requirements of key stakeholders -- advertisers, consumers, and platforms -- we propose design principles for commercially-influenced AI systems. We then outline high-level strategies for end users to identify and mitigate commercial biases from model outputs. Finally, we conclude with open questions and a call to action towards these goals. 

---
# RedactOR: An LLM-Powered Framework for Automatic Clinical Data De-Identification 

**Authors**: Praphul Singh, Charlotte Dzialo, Jangwon Kim, Sumana Srivatsa, Irfan Bulu, Sri Gadde, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18380)  

**Abstract**: Ensuring clinical data privacy while preserving utility is critical for AI-driven healthcare and data analytics. Existing de-identification (De-ID) methods, including rule-based techniques, deep learning models, and large language models (LLMs), often suffer from recall errors, limited generalization, and inefficiencies, limiting their real-world applicability. We propose a fully automated, multi-modal framework, RedactOR for de-identifying structured and unstructured electronic health records, including clinical audio records. Our framework employs cost-efficient De-ID strategies, including intelligent routing, hybrid rule and LLM based approaches, and a two-step audio redaction approach. We present a retrieval-based entity relexicalization approach to ensure consistent substitutions of protected entities, thereby enhancing data coherence for downstream applications. We discuss key design desiderata, de-identification and relexicalization methodology, and modular architecture of RedactX and its integration with the Oracle Health Clinical AI system. Evaluated on the i2b2 2014 De-ID dataset using standard metrics with strict recall, our approach achieves competitive performance while optimizing token usage to reduce LLM costs. Finally, we discuss key lessons and insights from deployment in real-world AI- driven healthcare data pipelines. 

---
# Understanding and Mitigating Overrefusal in LLMs from an Unveiling Perspective of Safety Decision Boundary 

**Authors**: Licheng Pan, Yongqi Tong, Xin Zhang, Xiaolu Zhang, Jun Zhou, Zhixuan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18325)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they often refuse to answer legitimate queries-a phenomenon known as overrefusal. Overrefusal typically stems from over-conservative safety alignment, causing models to treat many reasonable prompts as potentially risky. To systematically understand this issue, we probe and leverage the models'safety decision boundaries to analyze and mitigate overrefusal. Our findings reveal that overrefusal is closely tied to misalignment at these boundary regions, where models struggle to distinguish subtle differences between benign and harmful content. Building on these insights, we present RASS, an automated framework for prompt generation and selection that strategically targets overrefusal prompts near the safety boundary. By harnessing steering vectors in the representation space, RASS efficiently identifies and curates boundary-aligned prompts, enabling more effective and targeted mitigation of overrefusal. This approach not only provides a more precise and interpretable view of model safety decisions but also seamlessly extends to multilingual this http URL have explored the safety decision boundaries of various LLMs and construct the MORBench evaluation set to facilitate robust assessment of model safety and helpfulness across multiple languages. Code and datasets will be released at this https URL. 

---
# The end of radical concept nativism 

**Authors**: Joshua S. Rule, Steven T. Piantadosi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18277)  

**Abstract**: Though humans seem to be remarkable learners, arguments in cognitive science and philosophy of mind have long maintained that learning something fundamentally new is impossible. Specifically, Jerry Fodor's arguments for radical concept nativism hold that most, if not all, concepts are innate and that what many call concept learning never actually leads to the acquisition of new concepts. These arguments have deeply affected cognitive science, and many believe that the counterarguments to radical concept nativism have been either unsuccessful or only apply to a narrow class of concepts. This paper first reviews the features and limitations of prior arguments. We then identify three critical points - related to issues of expressive power, conceptual structure, and concept possession - at which the arguments in favor of radical concept nativism diverge from describing actual human cognition. We use ideas from computer science and information theory to formalize the relevant ideas in ways that are arguably more scientifically productive. We conclude that, as a result, there is an important sense in which people do indeed learn new concepts. 

---
# MangaVQA and MangaLMM: A Benchmark and Specialized Model for Multimodal Manga Understanding 

**Authors**: Jeonghun Baek, Kazuki Egashira, Shota Onohara, Atsuyuki Miyai, Yuki Imajuku, Hikaru Ikuta, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.20298)  

**Abstract**: Manga, or Japanese comics, is a richly multimodal narrative form that blends images and text in complex ways. Teaching large multimodal models (LMMs) to understand such narratives at a human-like level could help manga creators reflect on and refine their stories. To this end, we introduce two benchmarks for multimodal manga understanding: MangaOCR, which targets in-page text recognition, and MangaVQA, a novel benchmark designed to evaluate contextual understanding through visual question answering. MangaVQA consists of 526 high-quality, manually constructed question-answer pairs, enabling reliable evaluation across diverse narrative and visual scenarios. Building on these benchmarks, we develop MangaLMM, a manga-specialized model finetuned from the open-source LMM Qwen2.5-VL to jointly handle both tasks. Through extensive experiments, including comparisons with proprietary models such as GPT-4o and Gemini 2.5, we assess how well LMMs understand manga. Our benchmark and model provide a comprehensive foundation for evaluating and advancing LMMs in the richly narrative domain of manga. 

---
# Reasoning LLMs are Wandering Solution Explorers 

**Authors**: Jiahao Lu, Ziwei Xu, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.20296)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning abilities through test-time computation (TTC) techniques such as chain-of-thought prompting and tree-based reasoning. However, we argue that current reasoning LLMs (RLLMs) lack the ability to systematically explore the solution space. This paper formalizes what constitutes systematic problem solving and identifies common failure modes that reveal reasoning LLMs to be wanderers rather than systematic explorers. Through qualitative and quantitative analysis across multiple state-of-the-art LLMs, we uncover persistent issues: invalid reasoning steps, redundant explorations, hallucinated or unfaithful conclusions, and so on. Our findings suggest that current models' performance can appear to be competent on simple tasks yet degrade sharply as complexity increases. Based on the findings, we advocate for new metrics and tools that evaluate not just final outputs but the structure of the reasoning process itself. 

---
# Self-reflective Uncertainties: Do LLMs Know Their Internal Answer Distribution? 

**Authors**: Michael Kirchhof, Luca Füger, Adam Goliński, Eeshan Gunesh Dhekane, Arno Blaas, Sinead Williamson  

**Link**: [PDF](https://arxiv.org/pdf/2505.20295)  

**Abstract**: To reveal when a large language model (LLM) is uncertain about a response, uncertainty quantification commonly produces percentage numbers along with the output. But is this all we can do? We argue that in the output space of LLMs, the space of strings, exist strings expressive enough to summarize the distribution over output strings the LLM deems possible. We lay a foundation for this new avenue of uncertainty explication and present SelfReflect, a theoretically-motivated metric to assess how faithfully a string summarizes an LLM's internal answer distribution. We show that SelfReflect is able to discriminate even subtle differences of candidate summary strings and that it aligns with human judgement, outperforming alternative metrics such as LLM judges and embedding comparisons. With SelfReflect, we investigate a number of self-summarization methods and find that even state-of-the-art reasoning models struggle to explicate their internal uncertainty. But we find that faithful summarizations can be generated by sampling and summarizing. Our metric enables future works towards this universal form of LLM uncertainties. 

---
# GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scenes 

**Authors**: Xiao Chen, Tai Wang, Quanyi Li, Tao Huang, Jiangmiao Pang, Tianfan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2505.20294)  

**Abstract**: Generalizable active mapping in complex unknown environments remains a critical challenge for mobile robots. Existing methods, constrained by insufficient training data and conservative exploration strategies, exhibit limited generalizability across scenes with diverse layouts and complex connectivity. To enable scalable training and reliable evaluation, we introduce GLEAM-Bench, the first large-scale benchmark designed for generalizable active mapping with 1,152 diverse 3D scenes from synthetic and real-scan datasets. Building upon this foundation, we propose GLEAM, a unified generalizable exploration policy for active mapping. Its superior generalizability comes mainly from our semantic representations, long-term navigable goals, and randomized strategies. It significantly outperforms state-of-the-art methods, achieving 66.50% coverage (+9.49%) with efficient trajectories and improved mapping accuracy on 128 unseen complex scenes. Project page: this https URL. 

---
# OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation 

**Authors**: Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Chongyang Ma, Jiebo Luo, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20292)  

**Abstract**: Subject-to-Video (S2V) generation aims to create videos that faithfully incorporate reference content, providing enhanced flexibility in the production of videos. To establish the infrastructure for S2V generation, we propose OpenS2V-Nexus, consisting of (i) OpenS2V-Eval, a fine-grained benchmark, and (ii) OpenS2V-5M, a million-scale dataset. In contrast to existing S2V benchmarks inherited from VBench that focus on global and coarse-grained assessment of generated videos, OpenS2V-Eval focuses on the model's ability to generate subject-consistent videos with natural subject appearance and identity fidelity. For these purposes, OpenS2V-Eval introduces 180 prompts from seven major categories of S2V, which incorporate both real and synthetic test data. Furthermore, to accurately align human preferences with S2V benchmarks, we propose three automatic metrics, NexusScore, NaturalScore and GmeScore, to separately quantify subject consistency, naturalness, and text relevance in generated videos. Building on this, we conduct a comprehensive evaluation of 16 representative S2V models, highlighting their strengths and weaknesses across different content. Moreover, we create the first open-source large-scale S2V generation dataset OpenS2V-5M, which consists of five million high-quality 720P subject-text-video triples. Specifically, we ensure subject-information diversity in our dataset by (1) segmenting subjects and building pairing information via cross-video associations and (2) prompting GPT-Image-1 on raw frames to synthesize multi-view representations. Through OpenS2V-Nexus, we deliver a robust infrastructure to accelerate future S2V generation research. 

---
# EgoZero: Robot Learning from Smart Glasses 

**Authors**: Vincent Liu, Ademi Adeniji, Haotian Zhan, Raunaq Bhirangi, Pieter Abbeel, Lerrel Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2505.20290)  

**Abstract**: Despite recent progress in general purpose robotics, robot policies still lag far behind basic human capabilities in the real world. Humans interact constantly with the physical world, yet this rich data resource remains largely untapped in robot learning. We propose EgoZero, a minimal system that learns robust manipulation policies from human demonstrations captured with Project Aria smart glasses, $\textbf{and zero robot data}$. EgoZero enables: (1) extraction of complete, robot-executable actions from in-the-wild, egocentric, human demonstrations, (2) compression of human visual observations into morphology-agnostic state representations, and (3) closed-loop policy learning that generalizes morphologically, spatially, and semantically. We deploy EgoZero policies on a gripper Franka Panda robot and demonstrate zero-shot transfer with 70% success rate over 7 manipulation tasks and only 20 minutes of data collection per task. Our results suggest that in-the-wild human data can serve as a scalable foundation for real-world robot learning - paving the way toward a future of abundant, diverse, and naturalistic training data for robots. Code and videos are available at this https URL. 

---
# The Coverage Principle: A Framework for Understanding Compositional Generalization 

**Authors**: Hoyeon Chang, Jinho Park, Hanseul Cho, Sohee Yang, Miyoung Ko, Hyeonbin Hwang, Seungpil Won, Dohaeng Lee, Youbin Ahn, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.20278)  

**Abstract**: Large language models excel at pattern matching, yet often fall short in systematic compositional generalization. We propose the coverage principle: a data-centric framework showing that models relying primarily on pattern matching for compositional tasks cannot reliably generalize beyond substituting fragments that yield identical results when used in the same contexts. We demonstrate that this framework has a strong predictive power for the generalization capabilities of Transformers. First, we derive and empirically confirm that the training data required for two-hop generalization grows at least quadratically with the token set size, and the training data efficiency does not improve with 20x parameter scaling. Second, for compositional tasks with path ambiguity where one variable affects the output through multiple computational paths, we show that Transformers learn context-dependent state representations that undermine both performance and interoperability. Third, Chain-of-Thought supervision improves training data efficiency for multi-hop tasks but still struggles with path ambiguity. Finally, we outline a \emph{mechanism-based} taxonomy that distinguishes three ways neural networks can generalize: structure-based (bounded by coverage), property-based (leveraging algebraic invariances), and shared-operator (through function reuse). This conceptual lens contextualizes our results and highlights where new architectural ideas are needed to achieve systematic compositionally. Overall, the coverage principle provides a unified lens for understanding compositional reasoning, and underscores the need for fundamental architectural or training innovations to achieve truly systematic compositionality. 

---
# Does quantization affect models' performance on long-context tasks? 

**Authors**: Anmol Mekala, Anirudh Atmakuru, Yixiao Song, Marzena Karpinska, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.20276)  

**Abstract**: Large language models (LLMs) now support context windows exceeding 128K tokens, but this comes with significant memory requirements and high inference latency. Quantization can mitigate these costs, but may degrade performance. In this work, we present the first systematic evaluation of quantized LLMs on tasks with long-inputs (>64K tokens) and long-form outputs. Our evaluation spans 9.7K test examples, five quantization methods (FP8, GPTQ-int8, AWQ-int4, GPTQ-int4, BNB-nf4), and five models (Llama-3.1 8B and 70B; Qwen-2.5 7B, 32B, and 72B). We find that, on average, 8-bit quantization preserves accuracy (~0.8% drop), whereas 4-bit methods lead to substantial losses, especially for tasks involving long context inputs (drops of up to 59%). This degradation tends to worsen when the input is in a language other than English. Crucially, the effects of quantization depend heavily on the quantization method, model, and task. For instance, while Qwen-2.5 72B remains robust under BNB-nf4, Llama-3.1 70B experiences a 32% performance drop on the same task. These findings highlight the importance of a careful, task-specific evaluation before deploying quantized LLMs, particularly in long-context scenarios and with languages other than English. 

---
# Probabilistic Kernel Function for Fast Angle Testing 

**Authors**: Kejing Lu, Chuan Xiao, Yoshiharu Ishikawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.20274)  

**Abstract**: In this paper, we study the angle testing problem in high-dimensional Euclidean spaces and propose two projection-based probabilistic kernel functions, one designed for angle comparison and the other for angle thresholding. Unlike existing approaches that rely on random projection vectors drawn from Gaussian distributions, our approach leverages reference angles and employs a deterministic structure for the projection vectors. Notably, our kernel functions do not require asymptotic assumptions, such as the number of projection vectors tending to infinity, and can be both theoretically and experimentally shown to outperform Gaussian-distribution-based kernel functions. We further apply the proposed kernel function to Approximate Nearest Neighbor Search (ANNS) and demonstrate that our approach achieves a 2.5X ~ 3X higher query-per-second (QPS) throughput compared to the state-of-the-art graph-based search algorithm HNSW. 

---
# In-Context Brush: Zero-shot Customized Subject Insertion with Context-Aware Latent Space Manipulation 

**Authors**: Yu Xu, Fan Tang, You Wu, Lin Gao, Oliver Deussen, Hongbin Yan, Jintao Li, Juan Cao, Tong-Yee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.20271)  

**Abstract**: Recent advances in diffusion models have enhanced multimodal-guided visual generation, enabling customized subject insertion that seamlessly "brushes" user-specified objects into a given image guided by textual prompts. However, existing methods often struggle to insert customized subjects with high fidelity and align results with the user's intent through textual prompts. In this work, we propose "In-Context Brush", a zero-shot framework for customized subject insertion by reformulating the task within the paradigm of in-context learning. Without loss of generality, we formulate the object image and the textual prompts as cross-modal demonstrations, and the target image with the masked region as the query. The goal is to inpaint the target image with the subject aligning textual prompts without model tuning. Building upon a pretrained MMDiT-based inpainting network, we perform test-time enhancement via dual-level latent space manipulation: intra-head "latent feature shifting" within each attention head that dynamically shifts attention outputs to reflect the desired subject semantics and inter-head "attention reweighting" across different heads that amplifies prompt controllability through differential attention prioritization. Extensive experiments and applications demonstrate that our approach achieves superior identity preservation, text alignment, and image quality compared to existing state-of-the-art methods, without requiring dedicated training or additional data collection. 

---
# Comparing Neural Network Encodings for Logic-based Explainability 

**Authors**: Levi Cordeiro Carvalho, Saulo A. F. Oliveira, Thiago Alves Rocha  

**Link**: [PDF](https://arxiv.org/pdf/2505.20269)  

**Abstract**: Providing explanations for the outputs of artificial neural networks (ANNs) is crucial in many contexts, such as critical systems, data protection laws and handling adversarial examples. Logic-based methods can offer explanations with correctness guarantees, but face scalability challenges. Due to these issues, it is necessary to compare different encodings of ANNs into logical constraints, which are used in logic-based explainability. This work compares two encodings of ANNs: one has been used in the literature to provide explanations, while the other will be adapted for our context of explainability. Additionally, the second encoding uses fewer variables and constraints, thus, potentially enhancing efficiency. Experiments showed similar running times for computing explanations, but the adapted encoding performed up to 18\% better in building logical constraints and up to 16\% better in overall time. 

---
# Outcome-Based Online Reinforcement Learning: Algorithms and Fundamental Limits 

**Authors**: Fan Chen, Zeyu Jia, Alexander Rakhlin, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.20268)  

**Abstract**: Reinforcement learning with outcome-based feedback faces a fundamental challenge: when rewards are only observed at trajectory endpoints, how do we assign credit to the right actions? This paper provides the first comprehensive analysis of this problem in online RL with general function approximation. We develop a provably sample-efficient algorithm achieving $\widetilde{O}({C_{\rm cov} H^3}/{\epsilon^2})$ sample complexity, where $C_{\rm cov}$ is the coverability coefficient of the underlying MDP. By leveraging general function approximation, our approach works effectively in large or infinite state spaces where tabular methods fail, requiring only that value functions and reward functions can be represented by appropriate function classes. Our results also characterize when outcome-based feedback is statistically separated from per-step rewards, revealing an unavoidable exponential separation for certain MDPs. For deterministic MDPs, we show how to eliminate the completeness assumption, dramatically simplifying the algorithm. We further extend our approach to preference-based feedback settings, proving that equivalent statistical efficiency can be achieved even under more limited information. Together, these results constitute a theoretical foundation for understanding the statistical properties of outcome-based reinforcement learning. 

---
# We Need to Measure Data Diversity in NLP -- Better and Broader 

**Authors**: Dong Nguyen, Esther Ploeger  

**Link**: [PDF](https://arxiv.org/pdf/2505.20264)  

**Abstract**: Although diversity in NLP datasets has received growing attention, the question of how to measure it remains largely underexplored. This opinion paper examines the conceptual and methodological challenges of measuring data diversity and argues that interdisciplinary perspectives are essential for developing more fine-grained and valid measures. 

---
# Lifelong Safety Alignment for Language Models 

**Authors**: Haoyu Wang, Zeyu Qin, Yifei Zhao, Chao Du, Min Lin, Xueqian Wang, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20259)  

**Abstract**: LLMs have made impressive progress, but their growing capabilities also expose them to highly flexible jailbreaking attacks designed to bypass safety alignment. While many existing defenses focus on known types of attacks, it is more critical to prepare LLMs for unseen attacks that may arise during deployment. To address this, we propose a lifelong safety alignment framework that enables LLMs to continuously adapt to new and evolving jailbreaking strategies. Our framework introduces a competitive setup between two components: a Meta-Attacker, trained to actively discover novel jailbreaking strategies, and a Defender, trained to resist them. To effectively warm up the Meta-Attacker, we first leverage the GPT-4o API to extract key insights from a large collection of jailbreak-related research papers. Through iterative training, the first iteration Meta-Attacker achieves a 73% attack success rate (ASR) on RR and a 57% transfer ASR on LAT using only single-turn attacks. Meanwhile, the Defender progressively improves its robustness and ultimately reduces the Meta-Attacker's success rate to just 7%, enabling safer and more reliable deployment of LLMs in open-ended environments. The code is available at this https URL. 

---
# Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs 

**Authors**: Xiangchen Song, Aashiq Muhamed, Yujia Zheng, Lingjing Kong, Zeyu Tang, Mona T. Diab, Virginia Smith, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20254)  

**Abstract**: Sparse Autoencoders (SAEs) are a prominent tool in mechanistic interpretability (MI) for decomposing neural network activations into interpretable features. However, the aspiration to identify a canonical set of features is challenged by the observed inconsistency of learned SAE features across different training runs, undermining the reliability and efficiency of MI research. This position paper argues that mechanistic interpretability should prioritize feature consistency in SAEs -- the reliable convergence to equivalent feature sets across independent runs. We propose using the Pairwise Dictionary Mean Correlation Coefficient (PW-MCC) as a practical metric to operationalize consistency and demonstrate that high levels are achievable (0.80 for TopK SAEs on LLM activations) with appropriate architectural choices. Our contributions include detailing the benefits of prioritizing consistency; providing theoretical grounding and synthetic validation using a model organism, which verifies PW-MCC as a reliable proxy for ground-truth recovery; and extending these findings to real-world LLM data, where high feature consistency strongly correlates with the semantic similarity of learned feature explanations. We call for a community-wide shift towards systematically measuring feature consistency to foster robust cumulative progress in MI. 

---
# WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models 

**Authors**: Yongan Yu, Qingchen Hu, Xianda Du, Jiayin Wang, Fengran Mo, Renee Sieber  

**Link**: [PDF](https://arxiv.org/pdf/2505.20249)  

**Abstract**: Climate change adaptation requires the understanding of disruptive weather impacts on society, where large language models (LLMs) might be applicable. However, their effectiveness is under-explored due to the difficulty of high-quality corpus collection and the lack of available benchmarks. The climate-related events stored in regional newspapers record how communities adapted and recovered from disasters. However, the processing of the original corpus is non-trivial. In this study, we first develop a disruptive weather impact dataset with a four-stage well-crafted construction pipeline. Then, we propose WXImpactBench, the first benchmark for evaluating the capacity of LLMs on disruptive weather impacts. The benchmark involves two evaluation tasks, multi-label classification and ranking-based question answering. Extensive experiments on evaluating a set of LLMs provide first-hand analysis of the challenges in developing disruptive weather impact understanding and climate change adaptation systems. The constructed dataset and the code for the evaluation framework are available to help society protect against vulnerabilities from disasters. 

---
# KnowTrace: Bootstrapping Iterative Retrieval-Augmented Generation with Structured Knowledge Tracing 

**Authors**: Rui Li, Quanyu Dai, Zeyu Zhang, Xu Chen, Zhenhua Dong, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20245)  

**Abstract**: Recent advances in retrieval-augmented generation (RAG) furnish large language models (LLMs) with iterative retrievals of relevant information to handle complex multi-hop questions. These methods typically alternate between LLM reasoning and retrieval to accumulate external information into the LLM's context. However, the ever-growing context inherently imposes an increasing burden on the LLM to perceive connections among critical information pieces, with futile reasoning steps further exacerbating this overload issue. In this paper, we present KnowTrace, an elegant RAG framework to (1) mitigate the context overload and (2) bootstrap higher-quality multi-step reasoning. Instead of simply piling the retrieved contents, KnowTrace autonomously traces out desired knowledge triplets to organize a specific knowledge graph relevant to the input question. Such a structured workflow not only empowers the LLM with an intelligible context for inference, but also naturally inspires a reflective mechanism of knowledge backtracing to identify contributive LLM generations as process supervision data for self-bootstrapping. Extensive experiments show that KnowTrace consistently surpasses existing methods across three multi-hop question answering benchmarks, and the bootstrapped version further amplifies the gains. 

---
# DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning 

**Authors**: Qi Cao, Ruiyi Wang, Ruiyi Zhang, Sai Ashish Somayajula, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.20241)  

**Abstract**: Reasoning has substantially improved the performance of large language models (LLMs) on complicated tasks. Central to the current reasoning studies, Process Reward Models (PRMs) offer a fine-grained evaluation of intermediate reasoning steps and guide the reasoning process. However, extending PRMs to multimodal large language models (MLLMs) introduces challenges. Since multimodal reasoning covers a wider range of tasks compared to text-only scenarios, the resulting distribution shift from the training to testing sets is more severe, leading to greater generalization difficulty. Training a reliable multimodal PRM, therefore, demands large and diverse datasets to ensure sufficient coverage. However, current multimodal reasoning datasets suffer from a marked quality imbalance, which degrades PRM performance and highlights the need for an effective data selection strategy. To address the issues, we introduce DreamPRM, a domain-reweighted training framework for multimodal PRMs which employs bi-level optimization. In the lower-level optimization, DreamPRM performs fine-tuning on multiple datasets with domain weights, allowing the PRM to prioritize high-quality reasoning signals and alleviating the impact of dataset quality imbalance. In the upper-level optimization, the PRM is evaluated on a separate meta-learning dataset; this feedback updates the domain weights through an aggregation loss function, thereby improving the generalization capability of trained PRM. Extensive experiments on multiple multimodal reasoning benchmarks covering both mathematical and general reasoning show that test-time scaling with DreamPRM consistently improves the performance of state-of-the-art MLLMs. Further comparisons reveal that DreamPRM's domain-reweighting strategy surpasses other data selection methods and yields higher accuracy gains than existing test-time scaling approaches. 

---
# Variational Deep Learning via Implicit Regularization 

**Authors**: Jonathan Wenger, Beau Coker, Juraj Marusic, John P. Cunningham  

**Link**: [PDF](https://arxiv.org/pdf/2505.20235)  

**Abstract**: Modern deep learning models generalize remarkably well in-distribution, despite being overparametrized and trained with little to no explicit regularization. Instead, current theory credits implicit regularization imposed by the choice of architecture, hyperparameters and optimization procedure. However, deploying deep learning models out-of-distribution, in sequential decision-making tasks, or in safety-critical domains, necessitates reliable uncertainty quantification, not just a point estimate. The machinery of modern approximate inference -- Bayesian deep learning -- should answer the need for uncertainty quantification, but its effectiveness has been challenged by our inability to define useful explicit inductive biases through priors, as well as the associated computational burden. Instead, in this work we demonstrate, both theoretically and empirically, how to regularize a variational deep network implicitly via the optimization procedure, just as for standard deep learning. We fully characterize the inductive bias of (stochastic) gradient descent in the case of an overparametrized linear model as generalized variational inference and demonstrate the importance of the choice of parametrization. Finally, we show empirically that our approach achieves strong in- and out-of-distribution performance without tuning of additional hyperparameters and with minimal time and memory overhead over standard deep learning. 

---
# From What to How: Attributing CLIP's Latent Components Reveals Unexpected Semantic Reliance 

**Authors**: Maximilian Dreyer, Lorenz Hufe, Jim Berend, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2505.20229)  

**Abstract**: Transformer-based CLIP models are widely used for text-image probing and feature extraction, making it relevant to understand the internal mechanisms behind their predictions. While recent works show that Sparse Autoencoders (SAEs) yield interpretable latent components, they focus on what these encode and miss how they drive predictions. We introduce a scalable framework that reveals what latent components activate for, how they align with expected semantics, and how important they are to predictions. To achieve this, we adapt attribution patching for instance-wise component attributions in CLIP and highlight key faithfulness limitations of the widely used Logit Lens technique. By combining attributions with semantic alignment scores, we can automatically uncover reliance on components that encode semantically unexpected or spurious concepts. Applied across multiple CLIP variants, our method uncovers hundreds of surprising components linked to polysemous words, compound nouns, visual typography and dataset artifacts. While text embeddings remain prone to semantic ambiguity, they are more robust to spurious correlations compared to linear classifiers trained on image embeddings. A case study on skin lesion detection highlights how such classifiers can amplify hidden shortcuts, underscoring the need for holistic, mechanistic interpretability. We provide code at this https URL. 

---
# Parameter-Efficient Fine-Tuning with Column Space Projection 

**Authors**: Junseo Hwang, Wonguk Cho, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.20211)  

**Abstract**: Fine-tuning large language models (LLMs) with minimal computational overhead is essential for efficiently adapting them to downstream tasks under resource constraints. Parameter-efficient fine-tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), facilitate this by updating only a small subset of parameters. However, recent studies show that LoRA diverges from full fine-tuning (Full FT) in its learning behavior, particularly in terms of spectral properties. Motivated by these findings, we propose PiCa, the first theoretically grounded PEFT method based on the spectral properties of fine-tuned weights. PiCa projects gradients onto the low-rank column subspace of pre-trained weights and exhibits learning patterns more closely aligned with Full FT. Furthermore, we show that combining PiCa with weight sharing drastically reduces the number of trainable parameters without compromising performance, enabling to achieve superior performance than LoRA using 13x fewer trainable parameters. Extensive experiments demonstrate PiCa achieves the state-of-the-art performance compared to existing PEFT methods. 

---
# Evaluating Large Language Models for Code Review 

**Authors**: Umut Cihan, Arda İçöz, Vahid Haratian, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2505.20206)  

**Abstract**: Context: Code reviews are crucial for software quality. Recent AI advances have allowed large language models (LLMs) to review and fix code; now, there are tools that perform these reviews. However, their reliability and accuracy have not yet been systematically evaluated. Objective: This study compares different LLMs' performance in detecting code correctness and suggesting improvements. Method: We tested GPT4o and Gemini 2.0 Flash on 492 AI generated code blocks of varying correctness, along with 164 canonical code blocks from the HumanEval benchmark. To simulate the code review task objectively, we expected LLMs to assess code correctness and improve the code if needed. We ran experiments with different configurations and reported on the results. Results: With problem descriptions, GPT4o and Gemini 2.0 Flash correctly classified code correctness 68.50% and 63.89% of the time, respectively, and corrected the code 67.83% and 54.26% of the time for the 492 code blocks of varying correctness. Without problem descriptions, performance declined. The results for the 164 canonical code blocks differed, suggesting that performance depends on the type of code. Conclusion: LLM code reviews can help suggest improvements and assess correctness, but there is a risk of faulty outputs. We propose a process that involves humans, called the "Human in the loop LLM Code Review" to promote knowledge sharing while mitigating the risk of faulty outputs. 

---
# Leveraging Descriptions of Emotional Preferences in Recommender Systems 

**Authors**: Tonmoy Hasan, Razvan Bunescu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20190)  

**Abstract**: The affective attitude of liking a recommended item reflects just one category in a wide spectrum of affective phenomena that also includes emotions such as entranced or intrigued, moods such as cheerful or buoyant, as well as more fine-grained affective states, such as "pleasantly surprised by the conclusion". In this paper, we introduce a novel recommendation task that can leverage a virtually unbounded range of affective states sought explicitly by the user in order to identify items that, upon consumption, are likely to induce those affective states. Correspondingly, we create a large dataset of user preferences containing expressions of fine-grained affective states that are mined from book reviews, and propose a Transformer-based architecture that leverages such affective expressions as input. We then use the resulting dataset of affective states preferences, together with the linked users and their histories of book readings, ratings, and reviews, to train and evaluate multiple recommendation models on the task of matching recommended items with affective preferences. Experiments show that the best results are obtained by models that can utilize textual descriptions of items and user affective preferences. 

---
# THiNK: Can Large Language Models Think-aloud? 

**Authors**: Yongan Yu, Mengqian Wu, Yiran Lin, Nikki G. Lobczowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.20184)  

**Abstract**: Assessing higher-order thinking skills in large language models (LLMs) remains a fundamental challenge, especially in tasks that go beyond surface-level accuracy. In this work, we propose THiNK (Testing Higher-order Notion of Knowledge), a multi-agent, feedback-driven evaluation framework grounded in Bloom's Taxonomy. THiNK frames reasoning assessment as an iterative task of problem generation, critique, and revision, encouraging LLMs to think-aloud through step-by-step reflection and refinement. This enables a systematic evaluation of both lower-order (e.g., remember, understand) and higher-order (e.g., evaluate, create) thinking skills. We apply THiNK to seven state-of-the-art LLMs and perform a detailed cognitive analysis of their outputs. Results reveal that while models reliably perform lower-order categories well, they struggle with applying knowledge in realistic contexts and exhibit limited abstraction. Structured feedback loops significantly improve reasoning performance, particularly in higher-order thinking. Qualitative evaluations further confirm that THiNK-guided outputs better align with domain logic and problem structure. The code of our framework provides a scalable methodology for probing and enhancing LLM reasoning, offering new directions for evaluation grounded in learning science, which is available at our GitHub repository. 

---
# From Alignment to Advancement: Bootstrapping Audio-Language Alignment with Synthetic Data 

**Authors**: Chun-Yi Kuan, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.20166)  

**Abstract**: Audio-aware large language models (ALLMs) have recently made great strides in understanding and processing audio inputs. These models are typically adapted from text-based large language models (LLMs) through additional training on audio-related tasks. However, this adaptation process presents two major limitations. First, ALLMs often suffer from catastrophic forgetting, where important textual capabilities such as instruction-following are lost after training on audio data. In some cases, models may even hallucinate sounds that are not present in the input audio, raising concerns about their reliability. Second, achieving cross-modal alignment between audio and language typically relies on large collections of task-specific question-answer pairs for instruction tuning, making the process resource-intensive. To address these issues, we leverage the backbone LLMs from ALLMs to synthesize general-purpose caption-style alignment data. We refer to this process as bootstrapping audio-language alignment via synthetic data generation from backbone LLMs (BALSa). Building on BALSa, we introduce LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method designed to improve ALLMs' ability to distinguish between present and absent sounds. We further extend BALSa to multi-audio scenarios, where the model either explains the differences between audio inputs or produces a unified caption that describes them all, thereby enhancing audio-language alignment. Experimental results indicate that our method effectively mitigates audio hallucinations while reliably maintaining strong performance in audio understanding, reasoning, and instruction-following skills. Moreover, incorporating multi-audio training further enhances the model's comprehension and reasoning capabilities. Overall, BALSa offers an efficient and scalable approach to the development of ALLMs. 

---
# Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning 

**Authors**: Jaehun Jung, Seungju Han, Ximing Lu, Skyler Hallinan, David Acuna, Shrimai Prabhumoye, Mostafa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20161)  

**Abstract**: Effective generalization in language models depends critically on the diversity of their training data. Yet existing diversity metrics often fall short of this goal, relying on surface-level heuristics that are decoupled from model behavior. This motivates us to ask: What kind of diversity in training data actually drives generalization in language models -- and how can we measure and amplify it? Through large-scale empirical analyses spanning over 300 training runs, carefully controlled for data scale and quality, we show that data diversity can be a strong predictor of generalization in LLM reasoning -- as measured by average model performance on unseen out-of-distribution benchmarks. We introduce G-Vendi, a metric that quantifies diversity via the entropy of model-induced gradients. Despite using a small off-the-shelf proxy model for gradients, G-Vendi consistently outperforms alternative measures, achieving strong correlation (Spearman's $\rho \approx 0.9$) with out-of-distribution (OOD) performance on both natural language inference (NLI) and math reasoning tasks. Building on this insight, we present Prismatic Synthesis, a framework for generating diverse synthetic data by targeting underrepresented regions in gradient space. Experimental results show that Prismatic Synthesis consistently improves model performance as we scale synthetic data -- not just on in-distribution test but across unseen, out-of-distribution benchmarks -- significantly outperforming state-of-the-art models that rely on 20 times larger data generator than ours. For example, PrismMath-7B, our model distilled from a 32B LLM, outperforms R1-Distill-Qwen-7B -- the same base model trained on proprietary data generated by 671B R1 -- on 6 out of 7 challenging benchmarks. 

---
# Hard Negative Contrastive Learning for Fine-Grained Geometric Understanding in Large Multimodal Models 

**Authors**: Kai Sun, Yushi Bai, Zhen Yang, Jiajie Zhang, Ji Qi, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20152)  

**Abstract**: Benefiting from contrastively trained visual encoders on large-scale natural scene images, Large Multimodal Models (LMMs) have achieved remarkable performance across various visual perception tasks. However, the inherent limitations of contrastive learning upon summarized descriptions fundamentally restrict the capabilities of models in meticulous reasoning, particularly in crucial scenarios of geometric problem-solving. To enhance geometric understanding, we propose a novel hard negative contrastive learning framework for the vision encoder, which combines image-based contrastive learning using generation-based hard negatives created by perturbing diagram generation code, and text-based contrastive learning using rule-based negatives derived from modified geometric descriptions and retrieval-based negatives selected based on caption similarity. We train CLIP using our strong negative learning method, namely MMCLIP (Multimodal Math CLIP), and subsequently train an LMM for geometric problem-solving. Experiments show that our trained model, MMGeoLM, significantly outperforms other open-source models on three geometric reasoning benchmarks. Even with a size of 7B, it can rival powerful closed-source models like GPT-4o. We further study the impact of different negative sample construction methods and the number of negative samples on the geometric reasoning performance of LMM, yielding fruitful conclusions. The code and dataset are available at this https URL. 

---
# On the (Non) Injectivity of Piecewise Linear Janossy Pooling 

**Authors**: Ilai Reshef, Nadav Dym  

**Link**: [PDF](https://arxiv.org/pdf/2505.20150)  

**Abstract**: Multiset functions, which are functions that map multisets to vectors, are a fundamental tool in the construction of neural networks for multisets and graphs. To guarantee that the vector representation of the multiset is faithful, it is often desirable to have multiset mappings that are both injective and bi-Lipschitz. Currently, there are several constructions of multiset functions achieving both these guarantees, leading to improved performance in some tasks but often also to higher compute time than standard constructions. Accordingly, it is natural to inquire whether simpler multiset functions achieving the same guarantees are available. In this paper, we make a large step towards giving a negative answer to this question. We consider the family of k-ary Janossy pooling, which includes many of the most popular multiset models, and prove that no piecewise linear Janossy pooling function can be injective. On the positive side, we show that when restricted to multisets without multiplicities, even simple deep-sets models suffice for injectivity and bi-Lipschitzness. 

---
# Improvement Strategies for Few-Shot Learning in OCT Image Classification of Rare Retinal Diseases 

**Authors**: Cheng-Yu Tai, Ching-Wen Chen, Chi-Chin Wu, Bo-Chen Chiu, Cheng-Hung, Cheng-Kai Lu, Jia-Kang Wang, Tzu-Lun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20149)  

**Abstract**: This paper focuses on using few-shot learning to improve the accuracy of classifying OCT diagnosis images with major and rare classes. We used the GAN-based augmentation strategy as a baseline and introduced several novel methods to further enhance our model. The proposed strategy contains U-GAT-IT for improving the generative part and uses the data balance technique to narrow down the skew of accuracy between all categories. The best model obtained was built with CBAM attention mechanism and fine-tuned InceptionV3, and achieved an overall accuracy of 97.85%, representing a significant improvement over the original baseline. 

---
# StructEval: Benchmarking LLMs' Capabilities to Generate Structural Outputs 

**Authors**: Jialin Yang, Dongfu Jiang, Lipeng He, Sherman Siu, Yuxuan Zhang, Disen Liao, Zhuofeng Li, Huaye Zeng, Yiming Jia, Haozhe Wang, Benjamin Schneider, Chi Ruan, Wentao Ma, Zhiheng Lyu, Yifei Wang, Yi Lu, Quy Duc Do, Ziyan Jiang, Ping Nie, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20139)  

**Abstract**: As Large Language Models (LLMs) become integral to software development workflows, their ability to generate structured outputs has become critically important. We introduce StructEval, a comprehensive benchmark for evaluating LLMs' capabilities in producing both non-renderable (JSON, YAML, CSV) and renderable (HTML, React, SVG) structured formats. Unlike prior benchmarks, StructEval systematically evaluates structural fidelity across diverse formats through two paradigms: 1) generation tasks, producing structured output from natural language prompts, and 2) conversion tasks, translating between structured formats. Our benchmark encompasses 18 formats and 44 types of task, with novel metrics for format adherence and structural correctness. Results reveal significant performance gaps, even state-of-the-art models like o1-mini achieve only 75.58 average score, with open-source alternatives lagging approximately 10 points behind. We find generation tasks more challenging than conversion tasks, and producing correct visual content more difficult than generating text-only structures. 

---
# Error Optimization: Overcoming Exponential Signal Decay in Deep Predictive Coding Networks 

**Authors**: Cédric Goemaere, Gaspard Oliviers, Rafal Bogacz, Thomas Demeester  

**Link**: [PDF](https://arxiv.org/pdf/2505.20137)  

**Abstract**: Predictive Coding (PC) offers a biologically plausible alternative to backpropagation for neural network training, yet struggles with deeper architectures. This paper identifies the root cause: an inherent signal decay problem where gradients attenuate exponentially with depth, becoming computationally negligible due to numerical precision constraints. To address this fundamental limitation, we introduce Error Optimization (EO), a novel reparameterization that preserves PC's theoretical properties while eliminating signal decay. By optimizing over prediction errors rather than states, EO enables signals to reach all layers simultaneously and without attenuation, converging orders of magnitude faster than standard PC. Experiments across multiple architectures and datasets demonstrate that EO matches backpropagation's performance even for deeper models where conventional PC struggles. Besides practical improvements, our work provides theoretical insight into PC dynamics and establishes a foundation for scaling biologically-inspired learning to deeper architectures on digital hardware and beyond. 

---
# Tensorization is a powerful but underexplored tool for compression and interpretability of neural networks 

**Authors**: Safa Hamreras, Sukhbinder Singh, Román Orús  

**Link**: [PDF](https://arxiv.org/pdf/2505.20132)  

**Abstract**: Tensorizing a neural network involves reshaping some or all of its dense weight matrices into higher-order tensors and approximating them using low-rank tensor network decompositions. This technique has shown promise as a model compression strategy for large-scale neural networks. However, despite encouraging empirical results, tensorized neural networks (TNNs) remain underutilized in mainstream deep learning. In this position paper, we offer a perspective on both the potential and current limitations of TNNs. We argue that TNNs represent a powerful yet underexplored framework for deep learning--one that deserves greater attention from both engineering and theoretical communities. Beyond compression, we highlight the value of TNNs as a flexible class of architectures with distinctive scaling properties and increased interpretability. A central feature of TNNs is the presence of bond indices, which introduce new latent spaces not found in conventional networks. These internal representations may provide deeper insight into the evolution of features across layers, potentially advancing the goals of mechanistic interpretability. We conclude by outlining several key research directions aimed at overcoming the practical barriers to scaling and adopting TNNs in modern deep learning workflows. 

---
# Named Entity Recognition in Historical Italian: The Case of Giacomo Leopardi's Zibaldone 

**Authors**: Cristian Santini, Laura Melosi, Emanuele Frontoni  

**Link**: [PDF](https://arxiv.org/pdf/2505.20113)  

**Abstract**: The increased digitization of world's textual heritage poses significant challenges for both computer science and literary studies. Overall, there is an urgent need of computational techniques able to adapt to the challenges of historical texts, such as orthographic and spelling variations, fragmentary structure and digitization errors. The rise of large language models (LLMs) has revolutionized natural language processing, suggesting promising applications for Named Entity Recognition (NER) on historical documents. In spite of this, no thorough evaluation has been proposed for Italian texts. This research tries to fill the gap by proposing a new challenging dataset for entity extraction based on a corpus of 19th century scholarly notes, i.e. Giacomo Leopardi's Zibaldone (1898), containing 2,899 references to people, locations and literary works. This dataset was used to carry out reproducible experiments with both domain-specific BERT-based models and state-of-the-art LLMs such as LLaMa3.1. Results show that instruction-tuned models encounter multiple difficulties handling historical humanistic texts, while fine-tuned NER models offer more robust performance even with challenging entity types such as bibliographic references. 

---
# ResSVD: Residual Compensated SVD for Large Language Model Compression 

**Authors**: Haolei Bai, Siyong Jian, Tuo Liang, Yu Yin, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20112)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in a wide range of downstream natural language processing tasks. Nevertheless, their considerable sizes and memory demands hinder practical deployment, underscoring the importance of developing efficient compression strategies. Singular value decomposition (SVD) decomposes a matrix into orthogonal components, enabling efficient low-rank approximation. This is particularly suitable for LLM compression, where weight matrices often exhibit significant redundancy. However, current SVD-based methods neglect the residual matrix from truncation, resulting in significant truncation loss. Additionally, compressing all layers of the model results in severe performance degradation. To overcome these limitations, we propose ResSVD, a new post-training SVD-based LLM compression method. Specifically, we leverage the residual matrix generated during the truncation process to reduce truncation loss. Moreover, under a fixed overall compression ratio, we selectively compress the last few layers of the model, which mitigates error propagation and significantly improves the performance of compressed this http URL evaluations of ResSVD on diverse LLM families and multiple benchmark datasets indicate that ResSVD consistently achieves superior performance over existing counterpart methods, demonstrating its practical effectiveness. 

---
# Proxy-Free GFlowNet 

**Authors**: Ruishuo Chen, Xun Wang, Rui Hu, Zhuoran Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20110)  

**Abstract**: Generative Flow Networks (GFlowNets) are a promising class of generative models designed to sample diverse, high-reward structures by modeling distributions over compositional objects. In many real-world applications, obtaining the reward function for such objects is expensive, time-consuming, or requires human input, making it necessary to train GFlowNets from historical datasets. Most existing methods adopt a model-based approach, learning a proxy model from the dataset to approximate the reward function. However, this strategy inherently ties the quality of the learned policy to the accuracy of the proxy, introducing additional complexity and uncertainty into the training process. To overcome these limitations, we propose \textbf{Trajectory-Distilled GFlowNet (TD-GFN)}, a \emph{proxy-free} training framework that eliminates the need for out-of-dataset reward queries. Our method is motivated by the key observation that different edges in the associated directed acyclic graph (DAG) contribute unequally to effective policy learning. TD-GFN leverages inverse reinforcement learning to estimate edge-level rewards from the offline dataset, which are then used to ingeniously prune the DAG and guide backward trajectory sampling during training. This approach directs the policy toward high-reward regions while reducing the complexity of model fitting. Empirical results across multiple tasks show that TD-GFN trains both efficiently and reliably, significantly outperforming existing baselines in convergence speed and sample quality. 

---
# Language-Agnostic Suicidal Risk Detection Using Large Language Models 

**Authors**: June-Woo Kim, Wonkyo Oh, Haram Yoon, Sung-Hoon Yoon, Dae-Jin Kim, Dong-Ho Lee, Sang-Yeol Lee, Chan-Mo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20109)  

**Abstract**: Suicidal risk detection in adolescents is a critical challenge, yet existing methods rely on language-specific models, limiting scalability and generalization. This study introduces a novel language-agnostic framework for suicidal risk assessment with large language models (LLMs). We generate Chinese transcripts from speech using an ASR model and then employ LLMs with prompt-based queries to extract suicidal risk-related features from these transcripts. The extracted features are retained in both Chinese and English to enable cross-linguistic analysis and then used to fine-tune corresponding pretrained language models independently. Experimental results show that our method achieves performance comparable to direct fine-tuning with ASR results or to models trained solely on Chinese suicidal risk-related features, demonstrating its potential to overcome language constraints and improve the robustness of suicidal risk assessment. 

---
# AdaTP: Attention-Debiased Token Pruning for Video Large Language Models 

**Authors**: Fengyuan Sun, Leqi Shen, Hui Chen, Sicheng Zhao, Jungong Han, Guiguang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.20100)  

**Abstract**: Video Large Language Models (Video LLMs) have achieved remarkable results in video understanding tasks. However, they often suffer from heavy computational overhead due to the large number of visual tokens generated from multiple video frames. Existing visual token compression methods often rely on attention scores from language models as guidance. However, these scores exhibit inherent biases: global bias reflects a tendency to focus on the two ends of the visual token sequence, while local bias leads to an over-concentration on the same spatial positions across different frames. To address the issue of attention bias, we propose $\textbf{A}$ttention-$\textbf{D}$ebi$\textbf{a}$sed $\textbf{T}$oken $\textbf{P}$runing for Video Large Language Models ($\textbf{AdaTP}$), a novel token pruning pipeline for Video LLMs. AdaTP integrates two dedicated debiasing modules into the pipeline, targeting global attention bias and local attention bias, respectively. Without the need for additional training, our method significantly reduces the computational overhead of Video LLMs while retaining the performance of vanilla models. Extensive evaluation shows that AdaTP achieves state-of-the-art performance in various commonly used video understanding benchmarks. In particular, on LLaVA-OneVision-7B, AdaTP maintains performance without degradation while using only up to $27.3\%$ FLOPs compared to the vanilla model. Our code will be released soon. 

---
# Large Language Models Meet Knowledge Graphs for Question Answering: Synthesis and Opportunities 

**Authors**: Chuangtao Ma, Yongrui Chen, Tianxing Wu, Arijit Khan, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20099)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on question-answering (QA) tasks because of their superior capabilities in natural language understanding and generation. However, LLM-based QA struggles with complex QA tasks due to poor reasoning capacity, outdated knowledge, and hallucinations. Several recent works synthesize LLMs and knowledge graphs (KGs) for QA to address the above challenges. In this survey, we propose a new structured taxonomy that categorizes the methodology of synthesizing LLMs and KGs for QA according to the categories of QA and the KG's role when integrating with LLMs. We systematically survey state-of-the-art advances in synthesizing LLMs and KGs for QA and compare and analyze these approaches in terms of strength, limitations, and KG requirements. We then align the approaches with QA and discuss how these approaches address the main challenges of different complex QA. Finally, we summarize the advancements, evaluation metrics, and benchmark datasets and highlight open challenges and opportunities. 

---
# MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning 

**Authors**: Thang Nguyen, Peter Chin, Yu-Wing Tai  

**Link**: [PDF](https://arxiv.org/pdf/2505.20096)  

**Abstract**: We present MA-RAG, a Multi-Agent framework for Retrieval-Augmented Generation (RAG) that addresses the inherent ambiguities and reasoning challenges in complex information-seeking tasks. Unlike conventional RAG methods that rely on either end-to-end fine-tuning or isolated component enhancements, MA-RAG orchestrates a collaborative set of specialized AI agents: Planner, Step Definer, Extractor, and QA Agents, to tackle each stage of the RAG pipeline with task-aware reasoning. Ambiguities may arise from underspecified queries, sparse or indirect evidence in retrieved documents, or the need to integrate information scattered across multiple sources. MA-RAG mitigates these challenges by decomposing the problem into subtasks, such as query disambiguation, evidence extraction, and answer synthesis, and dispatching them to dedicated agents equipped with chain-of-thought prompting. These agents communicate intermediate reasoning and progressively refine the retrieval and synthesis process. Our design allows fine-grained control over information flow without any model fine-tuning. Crucially, agents are invoked on demand, enabling a dynamic and efficient workflow that avoids unnecessary computation. This modular and reasoning-driven architecture enables MA-RAG to deliver robust, interpretable results. Experiments on multi-hop and ambiguous QA benchmarks demonstrate that MA-RAG outperforms state-of-the-art training-free baselines and rivals fine-tuned systems, validating the effectiveness of collaborative agent-based reasoning in RAG. 

---
# Homophily Enhanced Graph Domain Adaptation 

**Authors**: Ruiyi Fang, Bingheng Li, Jingyu Zhao, Ruizhi Pu, Qiuhao Zeng, Gezheng Xu, Charles Ling, Boyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20089)  

**Abstract**: Graph Domain Adaptation (GDA) transfers knowledge from labeled source graphs to unlabeled target graphs, addressing the challenge of label scarcity. In this paper, we highlight the significance of graph homophily, a pivotal factor for graph domain alignment, which, however, has long been overlooked in existing approaches. Specifically, our analysis first reveals that homophily discrepancies exist in benchmarks. Moreover, we also show that homophily discrepancies degrade GDA performance from both empirical and theoretical aspects, which further underscores the importance of homophily alignment in GDA. Inspired by this finding, we propose a novel homophily alignment algorithm that employs mixed filters to smooth graph signals, thereby effectively capturing and mitigating homophily discrepancies between graphs. Experimental results on a variety of benchmarks verify the effectiveness of our method. 

---
# Explanation User Interfaces: A Systematic Literature Review 

**Authors**: Eleonora Cappuccio, Andrea Esposito, Francesco Greco, Giuseppe Desolda, Rosa Lanzilotti, Salvatore Rinzivillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.20085)  

**Abstract**: Artificial Intelligence (AI) is one of the major technological advancements of this century, bearing incredible potential for users through AI-powered applications and tools in numerous domains. Being often black-box (i.e., its decision-making process is unintelligible), developers typically resort to eXplainable Artificial Intelligence (XAI) techniques to interpret the behaviour of AI models to produce systems that are transparent, fair, reliable, and trustworthy. However, presenting explanations to the user is not trivial and is often left as a secondary aspect of the system's design process, leading to AI systems that are not useful to end-users. This paper presents a Systematic Literature Review on Explanation User Interfaces (XUIs) to gain a deeper understanding of the solutions and design guidelines employed in the academic literature to effectively present explanations to users. To improve the contribution and real-world impact of this survey, we also present a framework for Human-cEnteRed developMent of Explainable user interfaceS (HERMES) to guide practitioners and academics in the design and evaluation of XUIs. 

---
# Inference-time Alignment in Continuous Space 

**Authors**: Yige Yuan, Teng Xiao, Li Yunfan, Bingbing Xu, Shuchang Tao, Yunqi Qiu, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.20081)  

**Abstract**: Aligning large language models with human feedback at inference time has received increasing attention due to its flexibility. Existing methods rely on generating multiple responses from the base policy for search using a reward model, which can be considered as searching in a discrete response space. However, these methods struggle to explore informative candidates when the base policy is weak or the candidate set is small, resulting in limited effectiveness. In this paper, to address this problem, we propose Simple Energy Adaptation ($\textbf{SEA}$), a simple yet effective algorithm for inference-time alignment. In contrast to expensive search over the discrete space, SEA directly adapts original responses from the base policy toward the optimal one via gradient-based sampling in continuous latent space. Specifically, SEA formulates inference as an iterative optimization procedure on an energy function over actions in the continuous space defined by the optimal policy, enabling simple and effective alignment. For instance, despite its simplicity, SEA outperforms the second-best baseline with a relative improvement of up to $ \textbf{77.51%}$ on AdvBench and $\textbf{16.36%}$ on MATH. Our code is publicly available at this https URL 

---
# Incentivizing Reasoning from Weak Supervision 

**Authors**: Yige Yuan, Teng Xiao, Shuchang Tao, Xue Wang, Jinyang Gao, Bolin Ding, Bingbing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20072)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance on reasoning-intensive tasks, but enhancing their reasoning abilities typically relies on either reinforcement learning (RL) with verifiable signals or supervised fine-tuning (SFT) with high-quality long chain-of-thought (CoT) demonstrations, both of which are expensive. In this paper, we study a novel problem of incentivizing the reasoning capacity of LLMs without expensive high-quality demonstrations and reinforcement learning. We investigate whether the reasoning capabilities of LLMs can be effectively incentivized via supervision from significantly weaker models. We further analyze when and why such weak supervision succeeds in eliciting reasoning abilities in stronger models. Our findings show that supervision from significantly weaker reasoners can substantially improve student reasoning performance, recovering close to 94% of the gains of expensive RL at a fraction of the cost. Experiments across diverse benchmarks and model architectures demonstrate that weak reasoners can effectively incentivize reasoning in stronger student models, consistently improving performance across a wide range of reasoning tasks. Our results suggest that this simple weak-to-strong paradigm is a promising and generalizable alternative to costly methods for incentivizing strong reasoning capabilities at inference-time in LLMs. The code is publicly available at this https URL. 

---
# On the Same Page: Dimensions of Perceived Shared Understanding in Human-AI Interaction 

**Authors**: Qingyu Liang, Jaime Banks  

**Link**: [PDF](https://arxiv.org/pdf/2505.20068)  

**Abstract**: Shared understanding plays a key role in the effective communication in and performance of human-human interactions. With the increasingly common integration of AI into human contexts, the future of personal and workplace interactions will likely see human-AI interaction (HAII) in which the perception of shared understanding is important. Existing literature has addressed the processes and effects of PSU in human-human interactions, but the construal remains underexplored in HAII. To better understand PSU in HAII, we conducted an online survey to collect user reflections on interactions with a large language model when it sunderstanding of a situation was thought to be similar to or different from the participant's. Through inductive thematic analysis, we identified eight dimensions comprising PSU in human-AI interactions: Fluency, aligned operation, fluidity, outcome satisfaction, contextual awareness, lack of humanlike abilities, computational limits, and suspicion. 

---
# Community Moderation and the New Epistemology of Fact Checking on Social Media 

**Authors**: Isabelle Augenstein, Michiel Bakker, Tanmoy Chakraborty, David Corney, Emilio Ferrara, Iryna Gurevych, Scott Hale, Eduard Hovy, Heng Ji, Irene Larraz, Filippo Menczer, Preslav Nakov, Paolo Papotti, Dhruv Sahnan, Greta Warren, Giovanni Zagni  

**Link**: [PDF](https://arxiv.org/pdf/2505.20067)  

**Abstract**: Social media platforms have traditionally relied on internal moderation teams and partnerships with independent fact-checking organizations to identify and flag misleading content. Recently, however, platforms including X (formerly Twitter) and Meta have shifted towards community-driven content moderation by launching their own versions of crowd-sourced fact-checking -- Community Notes. If effectively scaled and governed, such crowd-checking initiatives have the potential to combat misinformation with increased scale and speed as successfully as community-driven efforts once did with spam. Nevertheless, general content moderation, especially for misinformation, is inherently more complex. Public perceptions of truth are often shaped by personal biases, political leanings, and cultural contexts, complicating consensus on what constitutes misleading content. This suggests that community efforts, while valuable, cannot replace the indispensable role of professional fact-checkers. Here we systemically examine the current approaches to misinformation detection across major platforms, explore the emerging role of community-driven moderation, and critically evaluate both the promises and challenges of crowd-checking at scale. 

---
# Automated data curation for self-supervised learning in underwater acoustic analysis 

**Authors**: Hilde I Hummel, Sandjai Bhulai, Burooj Ghani, Rob van der Mei  

**Link**: [PDF](https://arxiv.org/pdf/2505.20066)  

**Abstract**: The sustainability of the ocean ecosystem is threatened by increased levels of sound pollution, making monitoring crucial to understand its variability and impact. Passive acoustic monitoring (PAM) systems collect a large amount of underwater sound recordings, but the large volume of data makes manual analysis impossible, creating the need for automation. Although machine learning offers a potential solution, most underwater acoustic recordings are unlabeled. Self-supervised learning models have demonstrated success in learning from large-scale unlabeled data in various domains like computer vision, Natural Language Processing, and audio. However, these models require large, diverse, and balanced datasets for training in order to generalize well. To address this, a fully automated self-supervised data curation pipeline is proposed to create a diverse and balanced dataset from raw PAM data. It integrates Automatic Identification System (AIS) data with recordings from various hydrophones in the U.S. waters. Using hierarchical k-means clustering, the raw audio data is sampled and then combined with AIS samples to create a balanced and diverse dataset. The resulting curated dataset enables the development of self-supervised learning models, facilitating various tasks such as monitoring marine mammals and assessing sound pollution. 

---
# SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety 

**Authors**: Geon-Hyeong Kim, Youngsoo Jang, Yu Jin Kim, Byoungjip Kim, Honglak Lee, Kyunghoon Bae, Moontae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.20065)  

**Abstract**: As Large Language Models (LLMs) continue to advance and find applications across a growing number of fields, ensuring the safety of LLMs has become increasingly critical. To address safety concerns, recent studies have proposed integrating safety constraints into Reinforcement Learning from Human Feedback (RLHF). However, these approaches tend to be complex, as they encompass complicated procedures in RLHF along with additional steps required by the safety constraints. Inspired by Direct Preference Optimization (DPO), we introduce a new algorithm called SafeDPO, which is designed to directly optimize the safety alignment objective in a single stage of policy learning, without requiring relaxation. SafeDPO introduces only one additional hyperparameter to further enhance safety and requires only minor modifications to standard DPO. As a result, it eliminates the need to fit separate reward and cost models or to sample from the language model during fine-tuning, while still enhancing the safety of LLMs. Finally, we demonstrate that SafeDPO achieves competitive performance compared to state-of-the-art safety alignment algorithms, both in terms of aligning with human preferences and improving safety. 

---
# SAEs Are Good for Steering -- If You Select the Right Features 

**Authors**: Dana Arad, Aaron Mueller, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2505.20063)  

**Abstract**: Sparse Autoencoders (SAEs) have been proposed as an unsupervised approach to learn a decomposition of a model's latent space. This enables useful applications such as steering - influencing the output of a model towards a desired concept - without requiring labeled data. Current methods identify SAE features to steer by analyzing the input tokens that activate them. However, recent work has highlighted that activations alone do not fully describe the effect of a feature on the model's output. In this work, we draw a distinction between two types of features: input features, which mainly capture patterns in the model's input, and output features, which have a human-understandable effect on the model's output. We propose input and output scores to characterize and locate these types of features, and show that high values for both scores rarely co-occur in the same features. These findings have practical implications: after filtering out features with low output scores, we obtain 2-3x improvements when steering with SAEs, making them competitive with supervised methods. 

---
# Multimodal LLM-Guided Semantic Correction in Text-to-Image Diffusion 

**Authors**: Zheqi Lv, Junhao Chen, Qi Tian, Keting Yin, Shengyu Zhang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20053)  

**Abstract**: Diffusion models have become the mainstream architecture for text-to-image generation, achieving remarkable progress in visual quality and prompt controllability. However, current inference pipelines generally lack interpretable semantic supervision and correction mechanisms throughout the denoising process. Most existing approaches rely solely on post-hoc scoring of the final image, prompt filtering, or heuristic resampling strategies-making them ineffective in providing actionable guidance for correcting the generative trajectory. As a result, models often suffer from object confusion, spatial errors, inaccurate counts, and missing semantic elements, severely compromising prompt-image alignment and image quality. To tackle these challenges, we propose MLLM Semantic-Corrected Ping-Pong-Ahead Diffusion (PPAD), a novel framework that, for the first time, introduces a Multimodal Large Language Model (MLLM) as a semantic observer during inference. PPAD performs real-time analysis on intermediate generations, identifies latent semantic inconsistencies, and translates feedback into controllable signals that actively guide the remaining denoising steps. The framework supports both inference-only and training-enhanced settings, and performs semantic correction at only extremely few diffusion steps, offering strong generality and scalability. Extensive experiments demonstrate PPAD's significant improvements. 

---
# Grammars of Formal Uncertainty: When to Trust LLMs in Automated Reasoning Tasks 

**Authors**: Debargha Ganguly, Vikash Singh, Sreehari Sankar, Biyao Zhang, Xuecen Zhang, Srinivasan Iyengar, Xiaotian Han, Amit Sharma, Shivkumar Kalyanaraman, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2505.20047)  

**Abstract**: Large language models (LLMs) show remarkable promise for democratizing automated reasoning by generating formal specifications. However, a fundamental tension exists: LLMs are probabilistic, while formal verification demands deterministic guarantees. This paper addresses this epistemological gap by comprehensively investigating failure modes and uncertainty quantification (UQ) in LLM-generated formal artifacts. Our systematic evaluation of five frontier LLMs reveals Satisfiability Modulo Theories (SMT) based autoformalization's domain-specific impact on accuracy (from +34.8% on logical tasks to -44.5% on factual ones), with known UQ techniques like the entropy of token probabilities failing to identify these errors. We introduce a probabilistic context-free grammar (PCFG) framework to model LLM outputs, yielding a refined uncertainty taxonomy. We find uncertainty signals are task-dependent (e.g., grammar entropy for logic, AUROC>0.93). Finally, a lightweight fusion of these signals enables selective verification, drastically reducing errors (14-100%) with minimal abstention, transforming LLM-driven formalization into a reliable engineering discipline. 

---
# EmoNet-Face: An Expert-Annotated Benchmark for Synthetic Emotion Recognition 

**Authors**: Christoph Schuhmann, Robert Kaczmarczyk, Gollam Rabby, Maurice Kraus, Felix Friedrich, Huu Nguyen, Krishna Kalyan, Kourosh Nadi, Kristian Kersting, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2505.20033)  

**Abstract**: Effective human-AI interaction relies on AI's ability to accurately perceive and interpret human emotions. Current benchmarks for vision and vision-language models are severely limited, offering a narrow emotional spectrum that overlooks nuanced states (e.g., bitterness, intoxication) and fails to distinguish subtle differences between related feelings (e.g., shame vs. embarrassment). Existing datasets also often use uncontrolled imagery with occluded faces and lack demographic diversity, risking significant bias. To address these critical gaps, we introduce EmoNet Face, a comprehensive benchmark suite. EmoNet Face features: (1) A novel 40-category emotion taxonomy, meticulously derived from foundational research to capture finer details of human emotional experiences. (2) Three large-scale, AI-generated datasets (EmoNet HQ, Binary, and Big) with explicit, full-face expressions and controlled demographic balance across ethnicity, age, and gender. (3) Rigorous, multi-expert annotations for training and high-fidelity evaluation. (4) We build Empathic Insight Face, a model achieving human-expert-level performance on our benchmark. The publicly released EmoNet Face suite - taxonomy, datasets, and model - provides a robust foundation for developing and evaluating AI systems with a deeper understanding of human emotions. 

---
# Multiple Descents in Deep Learning as a Sequence of Order-Chaos Transitions 

**Authors**: Wenbo Wei, Nicholas Chong Jia Le, Choy Heng Lai, Ling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.20030)  

**Abstract**: We observe a novel 'multiple-descent' phenomenon during the training process of LSTM, in which the test loss goes through long cycles of up and down trend multiple times after the model is overtrained. By carrying out asymptotic stability analysis of the models, we found that the cycles in test loss are closely associated with the phase transition process between order and chaos, and the local optimal epochs are consistently at the critical transition point between the two phases. More importantly, the global optimal epoch occurs at the first transition from order to chaos, where the 'width' of the 'edge of chaos' is the widest, allowing the best exploration of better weight configurations for learning. 

---
# Correlating instruction-tuning (in multimodal models) with vision-language processing (in the brain) 

**Authors**: Subba Reddy Oota, Akshett Jindal, Ishani Mondal, Khushbu Pahwa, Satya Sai Srinath Namburi, Manish Shrivastava, Maneesh Singh, Bapi S. Raju, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.20029)  

**Abstract**: Transformer-based language models, though not explicitly trained to mimic brain recordings, have demonstrated surprising alignment with brain activity. Progress in these models-through increased size, instruction-tuning, and multimodality-has led to better representational alignment with neural data. Recently, a new class of instruction-tuned multimodal LLMs (MLLMs) have emerged, showing remarkable zero-shot capabilities in open-ended multimodal vision tasks. However, it is unknown whether MLLMs, when prompted with natural instructions, lead to better brain alignment and effectively capture instruction-specific representations. To address this, we first investigate brain alignment, i.e., measuring the degree of predictivity of neural visual activity using text output response embeddings from MLLMs as participants engage in watching natural scenes. Experiments with 10 different instructions show that MLLMs exhibit significantly better brain alignment than vision-only models and perform comparably to non-instruction-tuned multimodal models like CLIP. We also find that while these MLLMs are effective at generating high-quality responses suitable to the task-specific instructions, not all instructions are relevant for brain alignment. Further, by varying instructions, we make the MLLMs encode instruction-specific visual concepts related to the input image. This analysis shows that MLLMs effectively capture count-related and recognition-related concepts, demonstrating strong alignment with brain activity. Notably, the majority of the explained variance of the brain encoding models is shared between MLLM embeddings of image captioning and other instructions. These results suggest that enhancing MLLMs' ability to capture task-specific information could lead to better differentiation between various types of instructions, and thereby improving their precision in predicting brain responses. 

---
# Multi-modal brain encoding models for multi-modal stimuli 

**Authors**: Subba Reddy Oota, Khushbu Pahwa, Mounika Marreddy, Maneesh Singh, Manish Gupta, Bapi S. Raju  

**Link**: [PDF](https://arxiv.org/pdf/2505.20027)  

**Abstract**: Despite participants engaging in unimodal stimuli, such as watching images or silent videos, recent work has demonstrated that multi-modal Transformer models can predict visual brain activity impressively well, even with incongruent modality representations. This raises the question of how accurately these multi-modal models can predict brain activity when participants are engaged in multi-modal stimuli. As these models grow increasingly popular, their use in studying neural activity provides insights into how our brains respond to such multi-modal naturalistic stimuli, i.e., where it separates and integrates information across modalities through a hierarchy of early sensory regions to higher cognition. We investigate this question by using multiple unimodal and two types of multi-modal models-cross-modal and jointly pretrained-to determine which type of model is more relevant to fMRI brain activity when participants are engaged in watching movies. We observe that both types of multi-modal models show improved alignment in several language and visual regions. This study also helps in identifying which brain regions process unimodal versus multi-modal information. We further investigate the contribution of each modality to multi-modal alignment by carefully removing unimodal features one by one from multi-modal representations, and find that there is additional information beyond the unimodal embeddings that is processed in the visual and language regions. Based on this investigation, we find that while for cross-modal models, their brain alignment is partially attributed to the video modality; for jointly pretrained models, it is partially attributed to both the video and audio modalities. This serves as a strong motivation for the neuroscience community to investigate the interpretability of these models for deepening our understanding of multi-modal information processing in brain. 

---
# Gradient Inversion Transcript: Leveraging Robust Generative Priors to Reconstruct Training Data from Gradient Leakage 

**Authors**: Xinping Chen, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20026)  

**Abstract**: We propose Gradient Inversion Transcript (GIT), a novel generative approach for reconstructing training data from leaked gradients. GIT employs a generative attack model, whose architecture is tailored to align with the structure of the leaked model based on theoretical analysis. Once trained offline, GIT can be deployed efficiently and only relies on the leaked gradients to reconstruct the input data, rendering it applicable under various distributed learning environments. When used as a prior for other iterative optimization-based methods, GIT not only accelerates convergence but also enhances the overall reconstruction quality. GIT consistently outperforms existing methods across multiple datasets and demonstrates strong robustness under challenging conditions, including inaccurate gradients, data distribution shifts and discrepancies in model parameters. 

---
# ReasonPlan: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving 

**Authors**: Xueyi Liu, Zuodong Zhong, Yuxin Guo, Yun-Fu Liu, Zhiguo Su, Qichao Zhang, Junli Wang, Yinfeng Gao, Yupeng Zheng, Qiao Lin, Huiyong Chen, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20024)  

**Abstract**: Due to the powerful vision-language reasoning and generalization abilities, multimodal large language models (MLLMs) have garnered significant attention in the field of end-to-end (E2E) autonomous driving. However, their application to closed-loop systems remains underexplored, and current MLLM-based methods have not shown clear superiority to mainstream E2E imitation learning approaches. In this work, we propose ReasonPlan, a novel MLLM fine-tuning framework designed for closed-loop driving through holistic reasoning with a self-supervised Next Scene Prediction task and supervised Decision Chain-of-Thought process. This dual mechanism encourages the model to align visual representations with actionable driving context, while promoting interpretable and causally grounded decision making. We curate a planning-oriented decision reasoning dataset, namely PDR, comprising 210k diverse and high-quality samples. Our method outperforms the mainstream E2E imitation learning method by a large margin of 19% L2 and 16.1 driving score on Bench2Drive benchmark. Furthermore, ReasonPlan demonstrates strong zero-shot generalization on unseen DOS benchmark, highlighting its adaptability in handling zero-shot corner cases. Code and dataset will be found in this https URL. 

---
# Decomposing Complex Visual Comprehension into Atomic Visual Skills for Vision Language Models 

**Authors**: Hyunsik Chae, Seungwoo Yoon, Jaden Park, Chloe Yewon Chun, Yongin Cho, Mu Cai, Yong Jae Lee, Ernest K. Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20021)  

**Abstract**: Recent Vision-Language Models (VLMs) have demonstrated impressive multimodal comprehension and reasoning capabilities, yet they often struggle with trivially simple visual tasks. In this work, we focus on the domain of basic 2D Euclidean geometry and systematically categorize the fundamental, indivisible visual perception skills, which we refer to as atomic visual skills. We then introduce the Atomic Visual Skills Dataset (AVSD) for evaluating VLMs on the atomic visual skills. Using AVSD, we benchmark state-of-the-art VLMs and find that they struggle with these tasks, despite being trivial for adult humans. Our findings highlight the need for purpose-built datasets to train and evaluate VLMs on atomic, rather than composite, visual perception tasks. 

---
# ICDM: Interference Cancellation Diffusion Models for Wireless Semantic Communications 

**Authors**: Tong Wu, Zhiyong Chen, Dazhi He, Feng Yang, Meixia Tao, Xiaodong Xu, Wenjun Zhang, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19983)  

**Abstract**: Diffusion models (DMs) have recently achieved significant success in wireless communications systems due to their denoising capabilities. The broadcast nature of wireless signals makes them susceptible not only to Gaussian noise, but also to unaware interference. This raises the question of whether DMs can effectively mitigate interference in wireless semantic communication systems. In this paper, we model the interference cancellation problem as a maximum a posteriori (MAP) problem over the joint posterior probability of the signal and interference, and theoretically prove that the solution provides excellent estimates for the signal and interference. To solve this problem, we develop an interference cancellation diffusion model (ICDM), which decomposes the joint posterior into independent prior probabilities of the signal and interference, along with the channel transition probablity. The log-gradients of these distributions at each time step are learned separately by DMs and accurately estimated through deriving. ICDM further integrates these gradients with advanced numerical iteration method, achieving accurate and rapid interference cancellation. Extensive experiments demonstrate that ICDM significantly reduces the mean square error (MSE) and enhances perceptual quality compared to schemes without ICDM. For example, on the CelebA dataset under the Rayleigh fading channel with a signal-to-noise ratio (SNR) of $20$ dB and signal to interference plus noise ratio (SINR) of 0 dB, ICDM reduces the MSE by 4.54 dB and improves the learned perceptual image patch similarity (LPIPS) by 2.47 dB. 

---
# DFIR-Metric: A Benchmark Dataset for Evaluating Large Language Models in Digital Forensics and Incident Response 

**Authors**: Bilel Cherif, Tamas Bisztray, Richard A. Dubniczky, Aaesha Aldahmani, Saeed Alshehhi, Norbert Tihanyi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19973)  

**Abstract**: Digital Forensics and Incident Response (DFIR) involves analyzing digital evidence to support legal investigations. Large Language Models (LLMs) offer new opportunities in DFIR tasks such as log analysis and memory forensics, but their susceptibility to errors and hallucinations raises concerns in high-stakes contexts. Despite growing interest, there is no comprehensive benchmark to evaluate LLMs across both theoretical and practical DFIR domains. To address this gap, we present DFIR-Metric, a benchmark with three components: (1) Knowledge Assessment: a set of 700 expert-reviewed multiple-choice questions sourced from industry-standard certifications and official documentation; (2) Realistic Forensic Challenges: 150 CTF-style tasks testing multi-step reasoning and evidence correlation; and (3) Practical Analysis: 500 disk and memory forensics cases from the NIST Computer Forensics Tool Testing Program (CFTT). We evaluated 14 LLMs using DFIR-Metric, analyzing both their accuracy and consistency across trials. We also introduce a new metric, the Task Understanding Score (TUS), designed to more effectively evaluate models in scenarios where they achieve near-zero accuracy. This benchmark offers a rigorous, reproducible foundation for advancing AI in digital forensics. All scripts, artifacts, and results are available on the project website at this https URL. 

---
# Learning to Select In-Context Demonstration Preferred by Large Language Model 

**Authors**: Zheng Zhang, Shaocheng Lan, Lei Song, Jiang Bian, Yexin Li, Kan Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.19966)  

**Abstract**: In-context learning (ICL) enables large language models (LLMs) to adapt to new tasks during inference using only a few demonstrations. However, ICL performance is highly dependent on the selection of these demonstrations. Recent work explores retrieval-based methods for selecting query-specific demonstrations, but these approaches often rely on surrogate objectives such as metric learning, failing to directly optimize ICL performance. Consequently, they struggle to identify truly beneficial demonstrations. Moreover, their discriminative retrieval paradigm is ineffective when the candidate pool lacks sufficient high-quality demonstrations. To address these challenges, we propose GenICL, a novel generative preference learning framework that leverages LLM feedback to directly optimize demonstration selection for ICL. Experiments on 19 datasets across 11 task categories demonstrate that GenICL achieves superior performance than existing methods in selecting the most effective demonstrations, leading to better ICL performance. 

---
# The Limits of Preference Data for Post-Training 

**Authors**: Eric Zhao, Jessica Dai, Pranjal Awasthi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19964)  

**Abstract**: Recent progress in strengthening the capabilities of large language models has stemmed from applying reinforcement learning to domains with automatically verifiable outcomes. A key question is whether we can similarly use RL to optimize for outcomes in domains where evaluating outcomes inherently requires human feedback; for example, in tasks like deep research and trip planning, outcome evaluation is qualitative and there are many possible degrees of success. One attractive and scalable modality for collecting human feedback is preference data: ordinal rankings (pairwise or $k$-wise) that indicate, for $k$ given outcomes, which one is preferred. In this work, we study a critical roadblock: preference data fundamentally and significantly limits outcome-based optimization. Even with idealized preference data (infinite, noiseless, and online), the use of ordinal feedback can prevent obtaining even approximately optimal solutions. We formalize this impossibility using voting theory, drawing an analogy between how a model chooses to answer a query with how voters choose a candidate to elect. This indicates that grounded human scoring and algorithmic innovations are necessary for extending the success of RL post-training to domains demanding human feedback. We also explore why these limitations have disproportionately impacted RLHF when it comes to eliciting reasoning behaviors (e.g., backtracking) versus situations where RLHF has been historically successful (e.g., instruction-tuning and safety training), finding that the limitations of preference data primarily suppress RLHF's ability to elicit robust strategies -- a class that encompasses most reasoning behaviors. 

---
# MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research 

**Authors**: Hui Chen, Miao Xiong, Yujie Lu, Wei Han, Ailin Deng, Yufei He, Jiaying Wu, Yibo Li, Yue Liu, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19955)  

**Abstract**: Recent advancements in AI agents have demonstrated their growing potential to drive and support scientific discovery. In this work, we introduce MLR-Bench, a comprehensive benchmark for evaluating AI agents on open-ended machine learning research. MLR-Bench includes three key components: (1) 201 research tasks sourced from NeurIPS, ICLR, and ICML workshops covering diverse ML topics; (2) MLR-Judge, an automated evaluation framework combining LLM-based reviewers with carefully designed review rubrics to assess research quality; and (3) MLR-Agent, a modular agent scaffold capable of completing research tasks through four stages: idea generation, proposal formulation, experimentation, and paper writing. Our framework supports both stepwise assessment across these distinct research stages, and end-to-end evaluation of the final research paper. We then use MLR-Bench to evaluate six frontier LLMs and an advanced coding agent, finding that while LLMs are effective at generating coherent ideas and well-structured papers, current coding agents frequently (e.g., in 80% of the cases) produce fabricated or invalidated experimental results--posing a major barrier to scientific reliability. We validate MLR-Judge through human evaluation, showing high agreement with expert reviewers, supporting its potential as a scalable tool for research evaluation. We open-source MLR-Bench to help the community benchmark, diagnose, and improve AI research agents toward trustworthy and transparent scientific discovery. 

---
# Novel Loss-Enhanced Universal Adversarial Patches for Sustainable Speaker Privacy 

**Authors**: Elvir Karimov, Alexander Varlamov, Danil Ivanov, Dmitrii Korzh, Oleg Y. Rogov  

**Link**: [PDF](https://arxiv.org/pdf/2505.19951)  

**Abstract**: Deep learning voice models are commonly used nowadays, but the safety processing of personal data, such as human identity and speech content, remains suspicious. To prevent malicious user identification, speaker anonymization methods were proposed. Current methods, particularly based on universal adversarial patch (UAP) applications, have drawbacks such as significant degradation of audio quality, decreased speech recognition quality, low transferability across different voice biometrics models, and performance dependence on the input audio length. To mitigate these drawbacks, in this work, we introduce and leverage the novel Exponential Total Variance (TV) loss function and provide experimental evidence that it positively affects UAP strength and imperceptibility. Moreover, we present a novel scalable UAP insertion procedure and demonstrate its uniformly high performance for various audio lengths. 

---
# SaSi: A Self-augmented and Self-interpreted Deep Learning Approach for Few-shot Cryo-ET Particle Detection 

**Authors**: Gokul Adethya, Bhanu Pratyush Mantha, Tianyang Wang, Xingjian Li, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19948)  

**Abstract**: Cryo-electron tomography (cryo-ET) has emerged as a powerful technique for imaging macromolecular complexes in their near-native states. However, the localization of 3D particles in cellular environments still presents a significant challenge due to low signal-to-noise ratios and missing wedge artifacts. Deep learning approaches have shown great potential, but they need huge amounts of data, which can be a challenge in cryo-ET scenarios where labeled data is often scarce. In this paper, we propose a novel Self-augmented and Self-interpreted (SaSi) deep learning approach towards few-shot particle detection in 3D cryo-ET images. Our method builds upon self-augmentation techniques to further boost data utilization and introduces a self-interpreted segmentation strategy for alleviating dependency on labeled data, hence improving generalization and robustness. As demonstrated by experiments conducted on both simulated and real-world cryo-ET datasets, the SaSi approach significantly outperforms existing state-of-the-art methods for particle localization. This research increases understanding of how to detect particles with very few labels in cryo-ET and thus sets a new benchmark for few-shot learning in structural biology. 

---
# Dynamically Learned Test-Time Model Routing in Language Model Zoos with Service Level Guarantees 

**Authors**: Herbert Woisetschläger, Ryan Zhang, Shiqiang Wang, Hans-Arno Jacobsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19947)  

**Abstract**: Open-weight LLM zoos provide access to numerous high-quality models, but selecting the appropriate model for specific tasks remains challenging and requires technical expertise. Most users simply want factually correct, safe, and satisfying responses without concerning themselves with model technicalities, while inference service providers prioritize minimizing operating costs. These competing interests are typically mediated through service level agreements (SLAs) that guarantee minimum service quality. We introduce MESS+, a stochastic optimization algorithm for cost-optimal LLM request routing while providing rigorous SLA compliance guarantees. MESS+ learns request satisfaction probabilities of LLMs in real-time as users interact with the system, based on which model selection decisions are made by solving a per-request optimization problem. Our algorithm includes a novel combination of virtual queues and request satisfaction prediction, along with a theoretical analysis of cost optimality and constraint satisfaction. Across a wide range of state-of-the-art LLM benchmarks, MESS+ achieves an average of 2x cost savings compared to existing LLM routing techniques. 

---
# Can Visual Encoder Learn to See Arrows? 

**Authors**: Naoyuki Terashita, Yusuke Tozaki, Hideaki Omote, Congkha Nguyen, Ryosuke Nakamoto, Yuta Koreeda, Hiroaki Ozaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.19944)  

**Abstract**: The diagram is a visual representation of a relationship illustrated with edges (lines or arrows), which is widely used in industrial and scientific communication. Although recognizing diagrams is essential for vision language models (VLMs) to comprehend domain-specific knowledge, recent studies reveal that many VLMs fail to identify edges in images. We hypothesize that these failures stem from an over-reliance on textual and positional biases, preventing VLMs from learning explicit edge features. Based on this idea, we empirically investigate whether the image encoder in VLMs can learn edge representation through training on a diagram dataset in which edges are biased neither by textual nor positional information. To this end, we conduct contrastive learning on an artificially generated diagram--caption dataset to train an image encoder and evaluate its diagram-related features on three tasks: probing, image retrieval, and captioning. Our results show that the finetuned model outperforms pretrained CLIP in all tasks and surpasses zero-shot GPT-4o and LLaVA-Mistral in the captioning task. These findings confirm that eliminating textual and positional biases fosters accurate edge recognition in VLMs, offering a promising path for advancing diagram understanding. 

---
# A Responsible Face Recognition Approach for Small and Mid-Scale Systems Through Personalized Neural Networks 

**Authors**: Sebastian Groß, Stefan Heindorf, Philipp Terhörst  

**Link**: [PDF](https://arxiv.org/pdf/2505.19920)  

**Abstract**: Traditional face recognition systems rely on extracting fixed face representations, known as templates, to store and verify identities. These representations are typically generated by neural networks that often lack explainability and raise concerns regarding fairness and privacy. In this work, we propose a novel model-template (MOTE) approach that replaces vector-based face templates with small personalized neural networks. This design enables more responsible face recognition for small and medium-scale systems. During enrollment, MOTE creates a dedicated binary classifier for each identity, trained to determine whether an input face matches the enrolled identity. Each classifier is trained using only a single reference sample, along with synthetically balanced samples to allow adjusting fairness at the level of a single individual during enrollment. Extensive experiments across multiple datasets and recognition systems demonstrate substantial improvements in fairness and particularly in privacy. Although the method increases inference time and storage requirements, it presents a strong solution for small- and mid-scale applications where fairness and privacy are critical. 

---
# Evaluating AI cyber capabilities with crowdsourced elicitation 

**Authors**: Artem Petrov, Dmitrii Volkov  

**Link**: [PDF](https://arxiv.org/pdf/2505.19915)  

**Abstract**: As AI systems become increasingly capable, understanding their offensive cyber potential is critical for informed governance and responsible deployment. However, it's hard to accurately bound their capabilities, and some prior evaluations dramatically underestimated them. The art of extracting maximum task-specific performance from AIs is called "AI elicitation", and today's safety organizations typically conduct it in-house. In this paper, we explore crowdsourcing elicitation efforts as an alternative to in-house elicitation work.
We host open-access AI tracks at two Capture The Flag (CTF) competitions: AI vs. Humans (400 teams) and Cyber Apocalypse_ (4000 teams). The AI teams achieve outstanding performance at both events, ranking top-13% and top-21% respectively for a total of \$7500 in bounties. This impressive performance suggests that open-market elicitation may offer an effective complement to in-house elicitation. We propose elicitation bounties as a practical mechanism for maintaining timely, cost-effective situational awareness of emerging AI capabilities.
Another advantage of open elicitations is the option to collect human performance data at scale. Applying METR's methodology, we found that AI agents can reliably solve cyber challenges requiring one hour or less of effort from a median human CTF participant. 

---
# Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles 

**Authors**: Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Qiying Yu, Xuefeng Li, Jiaze Chen, Hao Zhou, Mingxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19914)  

**Abstract**: Large Language Models (LLMs), such as OpenAI's o1 and DeepSeek's R1, excel at advanced reasoning tasks like math and coding via Reinforcement Learning with Verifiable Rewards (RLVR), but still struggle with puzzles solvable by humans without domain knowledge. We introduce Enigmata, the first comprehensive suite tailored for improving LLMs with puzzle reasoning skills. It includes 36 tasks across seven categories, each with 1) a generator that produces unlimited examples with controllable difficulty and 2) a rule-based verifier for automatic evaluation. This generator-verifier design supports scalable, multi-task RL training, fine-grained analysis, and seamless RLVR integration. We further propose Enigmata-Eval, a rigorous benchmark, and develop optimized multi-task RLVR strategies. Our trained model, Qwen2.5-32B-Enigmata, consistently surpasses o3-mini-high and o1 on the puzzle reasoning benchmarks like Enigmata-Eval, ARC-AGI (32.8%), and ARC-AGI 2 (0.6%). It also generalizes well to out-of-domain puzzle benchmarks and mathematical reasoning, with little multi-tasking trade-off. When trained on larger models like Seed1.5-Thinking (20B activated parameters and 200B total parameters), puzzle data from Enigmata further boosts SoTA performance on advanced math and STEM reasoning tasks such as AIME (2024-2025), BeyondAIME and GPQA (Diamond), showing nice generalization benefits of Enigmata. This work offers a unified, controllable framework for advancing logical reasoning in LLMs. Resources of this work can be found at this https URL. 

---
# APE: A Data-Centric Benchmark for Efficient LLM Adaptation in Text Summarization 

**Authors**: Javier Marín  

**Link**: [PDF](https://arxiv.org/pdf/2505.19912)  

**Abstract**: We present Adjacent Possible Exploration (APE), a simple yet effective method for adapting large language models to specific tasks using minimal computational resources. Unlike traditional fine-tuning that requires extensive compute, APE iteratively fine-tunes models on small, carefully selected data batches (200 examples), retaining only improvements. On news summarization, APE achieves 40 percent BLEU improvement using just a T4 GPU in 60 minutes, matching or exceeding more complex methods like LoRA while remaining conceptually simple. Our approach is particularly valuable for researchers and practitioners with limited computational resources. We provide open-source code and demonstrate APE's effectiveness through both automatic metrics and human evaluation. While inspired by evolutionary theory's "adjacent possible", APE's core insight has a very practical application: small, iterative data perturbations can efficiently guide LLMs toward task-specific performance without expensive retraining. 

---
# Deconstructing Obfuscation: A four-dimensional framework for evaluating Large Language Models assembly code deobfuscation capabilities 

**Authors**: Anton Tkachenko, Dmitrij Suskevic, Benjamin Adolphi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19887)  

**Abstract**: Large language models (LLMs) have shown promise in software engineering, yet their effectiveness for binary analysis remains unexplored. We present the first comprehensive evaluation of commercial LLMs for assembly code deobfuscation. Testing seven state-of-the-art models against four obfuscation scenarios (bogus control flow, instruction substitution, control flow flattening, and their combination), we found striking performance variations--from autonomous deobfuscation to complete failure. We propose a theoretical framework based on four dimensions: Reasoning Depth, Pattern Recognition, Noise Filtering, and Context Integration, explaining these variations. Our analysis identifies five error patterns: predicate misinterpretation, structural mapping errors, control flow misinterpretation, arithmetic transformation errors, and constant propagation errors, revealing fundamental limitations in LLM code this http URL establish a three-tier resistance model: bogus control flow (low resistance), control flow flattening (moderate resistance), and instruction substitution/combined techniques (high resistance). Universal failure against combined techniques demonstrates that sophisticated obfuscation remains effective against advanced LLMs. Our findings suggest a human-AI collaboration paradigm where LLMs reduce expertise barriers for certain reverse engineering tasks while requiring human guidance for complex deobfuscation. This work provides a foundation for evaluating emerging capabilities and developing resistant obfuscation techniques.x deobfuscation. This work provides a foundation for evaluating emerging capabilities and developing resistant obfuscation techniques. 

---
# StyleAR: Customizing Multimodal Autoregressive Model for Style-Aligned Text-to-Image Generation 

**Authors**: Yi Wu, Lingting Zhu, Shengju Qian, Lei Liu, Wandi Qiao, Lequan Yu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19874)  

**Abstract**: In the current research landscape, multimodal autoregressive (AR) models have shown exceptional capabilities across various domains, including visual understanding and generation. However, complex tasks such as style-aligned text-to-image generation present significant challenges, particularly in data acquisition. In analogy to instruction-following tuning for image editing of AR models, style-aligned generation requires a reference style image and prompt, resulting in a text-image-to-image triplet where the output shares the style and semantics of the input. However, acquiring large volumes of such triplet data with specific styles is considerably more challenging than obtaining conventional text-to-image data used for training generative models. To address this issue, we propose StyleAR, an innovative approach that combines a specially designed data curation method with our proposed AR models to effectively utilize text-to-image binary data for style-aligned text-to-image generation. Our method synthesizes target stylized data using a reference style image and prompt, but only incorporates the target stylized image as the image modality to create high-quality binary data. To facilitate binary data training, we introduce a CLIP image encoder with a perceiver resampler that translates the image input into style tokens aligned with multimodal tokens in AR models and implement a style-enhanced token technique to prevent content leakage which is a common issue in previous work. Furthermore, we mix raw images drawn from large-scale text-image datasets with stylized images to enhance StyleAR's ability to extract richer stylistic features and ensure style consistency. Extensive qualitative and quantitative experiments demonstrate our superior performance. 

---
# Deep Active Inference Agents for Delayed and Long-Horizon Environments 

**Authors**: Yavar Taheri Yeganeh, Mohsen Jafari, Andrea Matta  

**Link**: [PDF](https://arxiv.org/pdf/2505.19867)  

**Abstract**: With the recent success of world-model agents, which extend the core idea of model-based reinforcement learning by learning a differentiable model for sample-efficient control across diverse tasks, active inference (AIF) offers a complementary, neuroscience-grounded paradigm that unifies perception, learning, and action within a single probabilistic framework powered by a generative model. Despite this promise, practical AIF agents still rely on accurate immediate predictions and exhaustive planning, a limitation that is exacerbated in delayed environments requiring plans over long horizons, tens to hundreds of steps. Moreover, most existing agents are evaluated on robotic or vision benchmarks which, while natural for biological agents, fall short of real-world industrial complexity. We address these limitations with a generative-policy architecture featuring (i) a multi-step latent transition that lets the generative model predict an entire horizon in a single look-ahead, (ii) an integrated policy network that enables the transition and receives gradients of the expected free energy, (iii) an alternating optimization scheme that updates model and policy from a replay buffer, and (iv) a single gradient step that plans over long horizons, eliminating exhaustive planning from the control loop. We evaluate our agent in an environment that mimics a realistic industrial scenario with delayed and long-horizon settings. The empirical results confirm the effectiveness of the proposed approach, demonstrating the coupled world-model with the AIF formalism yields an end-to-end probabilistic controller capable of effective decision making in delayed, long-horizon settings without handcrafted rewards or expensive planning. 

---
# Two Causally Related Needles in a Video Haystack 

**Authors**: Miaoyu Li, Qin Chao, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19853)  

**Abstract**: Evaluating the video understanding capabilities of Video-Language Models (VLMs) remains a significant challenge. We propose a long-context video understanding benchmark, Causal2Needles, that assesses two crucial abilities insufficiently evaluated by existing benchmarks: (1) the ability to extract information from two separate locations in a long video and understand them jointly, and (2) the ability to model the world in terms of cause and effect in human behaviors. Specifically, Causal2Needles introduces 2-needle questions, which require extracting information from both the cause and effect human-behavior events in a long video and the associated narration text. To prevent textual bias, these questions comprise two complementary formats: one asking to identify the video clip containing the answer, and one asking for the textual description of an unrelated visual detail from that video clip. Our experiments reveal that models excelling in pre-existing benchmarks struggle with 2-needle visual grounding, and the model performance is negatively correlated with the distance between the two needles. These findings highlight critical limitations in current VLMs. 

---
# Beyond Specialization: Benchmarking LLMs for Transliteration of Indian Languages 

**Authors**: Gulfarogh Azam, Mohd Sadique, Saif Ali, Mohammad Nadeem, Erik Cambria, Shahab Saquib Sohail, Mohammad Sultan Alam  

**Link**: [PDF](https://arxiv.org/pdf/2505.19851)  

**Abstract**: Transliteration, the process of mapping text from one script to another, plays a crucial role in multilingual natural language processing, especially within linguistically diverse contexts such as India. Despite significant advancements through specialized models like IndicXlit, recent developments in large language models suggest a potential for general-purpose models to excel at this task without explicit task-specific training. The current work systematically evaluates the performance of prominent LLMs, including GPT-4o, GPT-4.5, GPT-4.1, Gemma-3-27B-it, and Mistral-Large against IndicXlit, a state-of-the-art transliteration model, across ten major Indian languages. Experiments utilized standard benchmarks, including Dakshina and Aksharantar datasets, with performance assessed via Top-1 Accuracy and Character Error Rate. Our findings reveal that while GPT family models generally outperform other LLMs and IndicXlit for most instances. Additionally, fine-tuning GPT-4o improves performance on specific languages notably. An extensive error analysis and robustness testing under noisy conditions further elucidate strengths of LLMs compared to specialized models, highlighting the efficacy of foundational models for a wide spectrum of specialized applications with minimal overhead. 

---
# DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning 

**Authors**: Leander Diaz-Bone, Marco Bagatella, Jonas Hübotter, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2505.19850)  

**Abstract**: Sparse-reward reinforcement learning (RL) can model a wide range of highly complex tasks. Solving sparse-reward tasks is RL's core premise - requiring efficient exploration coupled with long-horizon credit assignment - and overcoming these challenges is key for building self-improving agents with superhuman ability. We argue that solving complex and high-dimensional tasks requires solving simpler tasks that are relevant to the target task. In contrast, most prior work designs strategies for selecting exploratory tasks with the objective of solving any task, making exploration of challenging high-dimensional, long-horizon tasks intractable. We find that the sense of direction, necessary for effective exploration, can be extracted from existing RL algorithms, without needing any prior information. Based on this finding, we propose a method for directed sparse-reward goal-conditioned very long-horizon RL (DISCOVER), which selects exploratory goals in the direction of the target task. We connect DISCOVER to principled exploration in bandits, formally bounding the time until the target task becomes achievable in terms of the agent's initial distance to the target, but independent of the volume of the space of all tasks. Empirically, we perform a thorough evaluation in high-dimensional environments. We find that the directed goal selection of DISCOVER solves exploration problems that are beyond the reach of prior state-of-the-art exploration methods in RL. 

---
# PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints 

**Authors**: Shuo Wang, Yun Cheng, Qingye Meng, Olga Saukh, Jiang Zhang, Jingfang Fan, Yuanting Zhang, Xingyuan Yuan, Lothar Thiele  

**Link**: [PDF](https://arxiv.org/pdf/2505.19842)  

**Abstract**: Air quality forecasting (AQF) is critical for public health and environmental management, yet remains challenging due to the complex interplay of emissions, meteorology, and chemical transformations. Traditional numerical models, such as CMAQ and WRF-Chem, provide physically grounded simulations but are computationally expensive and rely on uncertain emission inventories. Deep learning models, while computationally efficient, often struggle with generalization due to their lack of physical constraints. To bridge this gap, we propose PCDCNet, a surrogate model that integrates numerical modeling principles with deep learning. PCDCNet explicitly incorporates emissions, meteorological influences, and domain-informed constraints to model pollutant formation, transport, and dissipation. By combining graph-based spatial transport modeling, recurrent structures for temporal accumulation, and representation enhancement for local interactions, PCDCNet achieves state-of-the-art (SOTA) performance in 72-hour station-level PM2.5 and O3 forecasting while significantly reducing computational costs. Furthermore, our model is deployed in an online platform, providing free, real-time air quality forecasts, demonstrating its scalability and societal impact. By aligning deep learning with physical consistency, PCDCNet offers a practical and interpretable solution for AQF, enabling informed decision-making for both personal and regulatory applications. 

---
# FoodTaxo: Generating Food Taxonomies with Large Language Models 

**Authors**: Pascal Wullschleger, Majid Zarharan, Donnacha Daly, Marc Pouly, Jennifer Foster  

**Link**: [PDF](https://arxiv.org/pdf/2505.19838)  

**Abstract**: We investigate the utility of Large Language Models for automated taxonomy generation and completion specifically applied to taxonomies from the food technology industry. We explore the extent to which taxonomies can be completed from a seed taxonomy or generated without a seed from a set of known concepts, in an iterative fashion using recent prompting techniques. Experiments on five taxonomies using an open-source LLM (Llama-3), while promising, point to the difficulty of correctly placing inner nodes. 

---
# Revisiting Glorot Initialization for Long-Range Linear Recurrences 

**Authors**: Noga Bar, Mariia Seleznova, Yotam Alexander, Gitta Kutyniok, Raja Giryes  

**Link**: [PDF](https://arxiv.org/pdf/2505.19827)  

**Abstract**: Proper initialization is critical for Recurrent Neural Networks (RNNs), particularly in long-range reasoning tasks, where repeated application of the same weight matrix can cause vanishing or exploding signals. A common baseline for linear recurrences is Glorot initialization, designed to ensure stable signal propagation--but derived under the infinite-width, fixed-length regime--an unrealistic setting for RNNs processing long sequences. In this work, we show that Glorot initialization is in fact unstable: small positive deviations in the spectral radius are amplified through time and cause the hidden state to explode. Our theoretical analysis demonstrates that sequences of length $t = O(\sqrt{n})$, where $n$ is the hidden width, are sufficient to induce instability. To address this, we propose a simple, dimension-aware rescaling of Glorot that shifts the spectral radius slightly below one, preventing rapid signal explosion or decay. These results suggest that standard initialization schemes may break down in the long-sequence regime, motivating a separate line of theory for stable recurrent initialization. 

---
# Foundation Models for Tabular Data within Systemic Contexts Need Grounding 

**Authors**: Tassilo Klein, Johannes Hoffart  

**Link**: [PDF](https://arxiv.org/pdf/2505.19825)  

**Abstract**: Current research on tabular foundation models often overlooks the complexities of large-scale, real-world data by treating tables as isolated entities and assuming information completeness, thereby neglecting the vital operational context. To address this, we introduce the concept of Semantically Linked Tables (SLT), recognizing that tables are inherently connected to both declarative and procedural operational knowledge. We propose Foundation Models for Semantically Linked Tables (FMSLT), which integrate these components to ground tabular data within its true operational context. This comprehensive representation unlocks the full potential of machine learning for complex, interconnected tabular data across diverse domains. Realizing FMSLTs requires access to operational knowledge that is often unavailable in public datasets, highlighting the need for close collaboration between domain experts and researchers. Our work exposes the limitations of current tabular foundation models and proposes a new direction centered on FMSLTs, aiming to advance robust, context-aware models for structured data. 

---
# LAPA-based Dynamic Privacy Optimization for Wireless Federated Learning in Heterogeneous Environments 

**Authors**: Pengcheng Sun, Erwu Liu, Wei Ni, Rui Wang, Yuanzhe Geng, Lijuan Lai, Abbas Jamalipour  

**Link**: [PDF](https://arxiv.org/pdf/2505.19823)  

**Abstract**: Federated Learning (FL) is a distributed machine learning paradigm based on protecting data privacy of devices, which however, can still be broken by gradient leakage attack via parameter inversion techniques. Differential privacy (DP) technology reduces the risk of private data leakage by adding artificial noise to the gradients, but detrimental to the FL utility at the same time, especially in the scenario where the data is Non-Independent Identically Distributed (Non-IID). Based on the impact of heterogeneous data on aggregation performance, this paper proposes a Lightweight Adaptive Privacy Allocation (LAPA) strategy, which assigns personalized privacy budgets to devices in each aggregation round without transmitting any additional information beyond gradients, ensuring both privacy protection and aggregation efficiency. Furthermore, the Deep Deterministic Policy Gradient (DDPG) algorithm is employed to optimize the transmission power, in order to determine the optimal timing at which the adaptively attenuated artificial noise aligns with the communication noise, enabling an effective balance between DP and system utility. Finally, a reliable aggregation strategy is designed by integrating communication quality and data distribution characteristics, which improves aggregation performance while preserving privacy. Experimental results demonstrate that the personalized noise allocation and dynamic optimization strategy based on LAPA proposed in this paper enhances convergence performance while satisfying the privacy requirements of FL. 

---
# FinLoRA: Benchmarking LoRA Methods for Fine-Tuning LLMs on Financial Datasets 

**Authors**: Dannong Wang, Jaisal Patel, Daochen Zha, Steve Y. Yang, Xiao-Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19819)  

**Abstract**: Low-rank adaptation (LoRA) methods show great potential for scaling pre-trained general-purpose Large Language Models (LLMs) to hundreds or thousands of use scenarios. However, their efficacy in high-stakes domains like finance is rarely explored, e.g., passing CFA exams and analyzing SEC filings. In this paper, we present the open-source FinLoRA project that benchmarks LoRA methods on both general and highly professional financial tasks. First, we curated 19 datasets covering diverse financial applications; in particular, we created four novel XBRL analysis datasets based on 150 SEC filings. Second, we evaluated five LoRA methods and five base LLMs. Finally, we provide extensive experimental results in terms of accuracy, F1, and BERTScore and report computational cost in terms of time and GPU memory during fine-tuning and inference stages. We find that LoRA methods achieved substantial performance gains of 36\% on average over base models. Our FinLoRA project provides an affordable and scalable approach to democratize financial intelligence to the general public. Datasets, LoRA adapters, code, and documentation are available at this https URL 

---
# Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective 

**Authors**: Junnan Liu, Hongwei Liu, Linchen Xiao, Shudong Liu, Taolin Zhang, Zihan Ma, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19815)  

**Abstract**: We propose a novel framework for comprehending the reasoning capabilities of large language models (LLMs) through the perspective of meta-learning. By conceptualizing reasoning trajectories as pseudo-gradient descent updates to the LLM's parameters, we identify parallels between LLM reasoning and various meta-learning paradigms. We formalize the training process for reasoning tasks as a meta-learning setup, with each question treated as an individual task, and reasoning trajectories serving as the inner loop optimization for adapting model parameters. Once trained on a diverse set of questions, the LLM develops fundamental reasoning capabilities that can generalize to previously unseen questions. Extensive empirical evaluations substantiate the strong connection between LLM reasoning and meta-learning, exploring several issues of significant interest from a meta-learning standpoint. Our work not only enhances the understanding of LLM reasoning but also provides practical insights for improving these models through established meta-learning techniques. 

---
# Equivariant Representation Learning for Symmetry-Aware Inference with Guarantees 

**Authors**: Daniel Ordoñez-Apraez, Alek Fröhlich, Vladimir Kostić, Karim Lounici, Vivien Brandt, Massimiliano Pontil  

**Link**: [PDF](https://arxiv.org/pdf/2505.19809)  

**Abstract**: In many real-world applications of regression, conditional probability estimation, and uncertainty quantification, exploiting symmetries rooted in physics or geometry can dramatically improve generalization and sample efficiency. While geometric deep learning has made significant empirical advances by incorporating group-theoretic structure, less attention has been given to statistical learning guarantees. In this paper, we introduce an equivariant representation learning framework that simultaneously addresses regression, conditional probability estimation, and uncertainty quantification while providing first-of-its-kind non-asymptotic statistical learning guarantees. Grounded in operator and group representation theory, our framework approximates the spectral decomposition of the conditional expectation operator, building representations that are both equivariant and disentangled along independent symmetry subgroups. Empirical evaluations on synthetic datasets and real-world robotics applications confirm the potential of our approach, matching or outperforming existing equivariant baselines in regression while additionally providing well-calibrated parametric uncertainty estimates. 

---
# The Missing Point in Vision Transformers for Universal Image Segmentation 

**Authors**: Sajjad Shahabodini, Mobina Mansoori, Farnoush Bayatmakou, Jamshid Abouei, Konstantinos N. Plataniotis, Arash Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19795)  

**Abstract**: Image segmentation remains a challenging task in computer vision, demanding robust mask generation and precise classification. Recent mask-based approaches yield high-quality masks by capturing global context. However, accurately classifying these masks, especially in the presence of ambiguous boundaries and imbalanced class distributions, remains an open challenge. In this work, we introduce ViT-P, a novel two-stage segmentation framework that decouples mask generation from classification. The first stage employs a proposal generator to produce class-agnostic mask proposals, while the second stage utilizes a point-based classification model built on the Vision Transformer (ViT) to refine predictions by focusing on mask central points. ViT-P serves as a pre-training-free adapter, allowing the integration of various pre-trained vision transformers without modifying their architecture, ensuring adaptability to dense prediction tasks. Furthermore, we demonstrate that coarse and bounding box annotations can effectively enhance classification without requiring additional training on fine annotation datasets, reducing annotation costs while maintaining strong performance. Extensive experiments across COCO, ADE20K, and Cityscapes datasets validate the effectiveness of ViT-P, achieving state-of-the-art results with 54.0 PQ on ADE20K panoptic segmentation, 87.4 mIoU on Cityscapes semantic segmentation, and 63.6 mIoU on ADE20K semantic segmentation. The code and pretrained models are available at: this https URL}{this https URL. 

---
# Alpay Algebra III: Observer-Coupled Collapse and the Temporal Drift of Identity 

**Authors**: Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2505.19790)  

**Abstract**: This paper introduces a formal framework for modeling observer-dependent collapse dynamics and temporal identity drift within artificial and mathematical systems, grounded entirely in the symbolic foundations of Alpay Algebra. Building upon the fixed-point emergence structures developed in Alpay Algebra I and II, this third installment formalizes the observer-coupled {\phi}-collapse process through transfinite categorical flows and curvature-driven identity operators. We define a novel temporal drift mechanism as a recursive deformation of identity signatures under entangled observer influence, constructing categorical invariants that evolve across fold iterations. The proposed system surpasses conventional identity modeling in explainable AI (XAI) by encoding internal transformation history into a symbolic fixed-point structure, offering provable traceability and temporal coherence. Applications range from AI self-awareness architectures to formal logic systems where identity is not static but dynamically induced by observation. The theoretical results also offer a mathematically rigorous basis for future AI systems with stable self-referential behavior, positioning Alpay Algebra as a next-generation symbolic framework bridging category theory, identity logic, and observer dynamics. 

---
# MedDreamer: Model-Based Reinforcement Learning with Latent Imagination on Complex EHRs for Clinical Decision Support 

**Authors**: Qianyi Xu, Gousia Habib, Dilruk Perera, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19785)  

**Abstract**: Timely and personalized treatment decisions are essential across a wide range of healthcare settings where patient responses vary significantly and evolve over time. Clinical data used to support these decisions are often irregularly sampled, sparse, and noisy. Existing decision support systems commonly rely on discretization and imputation, which can distort critical temporal dynamics and degrade decision quality. Moreover, they often overlook the clinical significance of irregular recording frequencies, filtering out patterns in how and when data is collected. Reinforcement Learning (RL) is a natural fit for clinical decision-making, enabling sequential, long-term optimization in dynamic, uncertain environments. However, most existing treatment recommendation systems are model-free and trained solely on offline data, making them sample-inefficient, sensitive to data quality, and poorly generalizable across tasks or cohorts. To address these limitations, we propose MedDreamer, a two-phase model-based RL framework for personalized treatment recommendation. MedDreamer uses a world model with an Adaptive Feature Integration (AFI) module to effectively model irregular, sparse clinical data. Through latent imagination, it simulates plausible patient trajectories to enhance learning, refining its policy using a mix of real and imagined experiences. This enables learning policies that go beyond suboptimal historical decisions while remaining grounded in clinical data. To our knowledge, this is the first application of latent imagination to irregular healthcare data. Evaluations on sepsis and mechanical ventilation (MV) treatment using two large-scale EHR datasets show that MedDreamer outperforms both model-free and model-based baselines in clinical outcomes and off-policy metrics. 

---
# Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification 

**Authors**: Akram Elbouanani, Evan Dufraisse, Adrian Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19776)  

**Abstract**: Political biases encoded by LLMs might have detrimental effects on downstream applications. Existing bias analysis methods rely on small-size intermediate tasks (questionnaire answering or political content generation) and rely on the LLMs themselves for analysis, thus propagating bias. We propose a new approach leveraging the observation that LLM sentiment predictions vary with the target entity in the same sentence. We define an entropy-based inconsistency metric to encode this prediction variability. We insert 1319 demographically and politically diverse politician names in 450 political sentences and predict target-oriented sentiment using seven models in six widely spoken languages. We observe inconsistencies in all tested combinations and aggregate them in a statistically robust analysis at different granularity levels. We observe positive and negative bias toward left and far-right politicians and positive correlations between politicians with similar alignment. Bias intensity is higher for Western languages than for others. Larger models exhibit stronger and more consistent biases and reduce discrepancies between similar languages. We partially mitigate LLM unreliability in target-oriented sentiment classification (TSC) by replacing politician names with fictional but plausible counterparts. 

---
# TeViR: Text-to-Video Reward with Diffusion Models for Efficient Reinforcement Learning 

**Authors**: Yuhui Chen, Haoran Li, Zhennan Jiang, Haowei Wen, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19769)  

**Abstract**: Developing scalable and generalizable reward engineering for reinforcement learning (RL) is crucial for creating general-purpose agents, especially in the challenging domain of robotic manipulation. While recent advances in reward engineering with Vision-Language Models (VLMs) have shown promise, their sparse reward nature significantly limits sample efficiency. This paper introduces TeViR, a novel method that leverages a pre-trained text-to-video diffusion model to generate dense rewards by comparing the predicted image sequence with current observations. Experimental results across 11 complex robotic tasks demonstrate that TeViR outperforms traditional methods leveraging sparse rewards and other state-of-the-art (SOTA) methods, achieving better sample efficiency and performance without ground truth environmental rewards. TeViR's ability to efficiently guide agents in complex environments highlights its potential to advance reinforcement learning applications in robotic manipulation. 

---
# Agentic Predictor: Performance Prediction for Agentic Workflows via Multi-View Encoding 

**Authors**: Patara Trirat, Wonyong Jeong, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19764)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but optimizing LLM-based agentic systems remains challenging due to the vast search space of agent configurations, prompting strategies, and communication patterns. Existing approaches often rely on heuristic-based tuning or exhaustive evaluation, which can be computationally expensive and suboptimal. This paper proposes Agentic Predictor, a lightweight predictor for efficient agentic workflow evaluation. Agentic Predictor is equipped with a multi-view workflow encoding technique that leverages multi-view representation learning of agentic systems by incorporating code architecture, textual prompts, and interaction graph features. To achieve high predictive accuracy while significantly reducing the number of required workflow evaluations for training a predictor, Agentic Predictor employs cross-domain unsupervised pretraining. By learning to approximate task success rates, Agentic Predictor enables fast and accurate selection of optimal agentic workflow configurations for a given task, significantly reducing the need for expensive trial-and-error evaluations. Experiments on a carefully curated benchmark spanning three domains show that our predictor outperforms state-of-the-art methods in both predictive accuracy and workflow utility, highlighting the potential of performance predictors in streamlining the design of LLM-based agentic workflows. 

---
# CIDRe: A Reference-Free Multi-Aspect Criterion for Code Comment Quality Measurement 

**Authors**: Maria Dziuba, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2505.19757)  

**Abstract**: Effective generation of structured code comments requires robust quality metrics for dataset curation, yet existing approaches (SIDE, MIDQ, STASIS) suffer from limited code-comment analysis. We propose CIDRe, a language-agnostic reference-free quality criterion combining four synergistic aspects: (1) relevance (code-comment semantic alignment), (2) informativeness (functional coverage), (3) completeness (presence of all structure sections), and (4) description length (detail sufficiency). We validate our criterion on a manually annotated dataset. Experiments demonstrate CIDRe's superiority over existing metrics, achieving improvement in cross-entropy evaluation. When applied to filter comments, the models finetuned on CIDRe-filtered data show statistically significant quality gains in GPT-4o-mini assessments. 

---
# NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering 

**Authors**: Ruisheng Cao, Hanchong Zhang, Tiancheng Huang, Zhangyi Kang, Yuxin Zhang, Liangtai Sun, Hanqi Li, Yuxun Miao, Shuai Fan, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19754)  

**Abstract**: The increasing number of academic papers poses significant challenges for researchers to efficiently acquire key details. While retrieval augmented generation (RAG) shows great promise in large language model (LLM) based automated question answering, previous works often isolate neural and symbolic retrieval despite their complementary strengths. Moreover, conventional single-view chunking neglects the rich structure and layout of PDFs, e.g., sections and tables. In this work, we propose NeuSym-RAG, a hybrid neural symbolic retrieval framework which combines both paradigms in an interactive process. By leveraging multi-view chunking and schema-based parsing, NeuSym-RAG organizes semi-structured PDF content into both the relational database and vectorstore, enabling LLM agents to iteratively gather context until sufficient to generate answers. Experiments on three full PDF-based QA datasets, including a self-annotated one AIRQA-REAL, show that NeuSym-RAG stably defeats both the vector-based RAG and various structured baselines, highlighting its capacity to unify both retrieval schemes and utilize multiple views. Code and data are publicly available at this https URL. 

---
# Discrete Markov Bridge 

**Authors**: Hengli Li, Yuxuan Wang, Song-Chun Zhu, Ying Nian Wu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19752)  

**Abstract**: Discrete diffusion has recently emerged as a promising paradigm in discrete data modeling. However, existing methods typically rely on a fixed rate transition matrix during training, which not only limits the expressiveness of latent representations, a fundamental strength of variational methods, but also constrains the overall design space. To address these limitations, we propose Discrete Markov Bridge, a novel framework specifically designed for discrete representation learning. Our approach is built upon two key components: Matrix Learning and Score Learning. We conduct a rigorous theoretical analysis, establishing formal performance guarantees for Matrix Learning and proving the convergence of the overall framework. Furthermore, we analyze the space complexity of our method, addressing practical constraints identified in prior studies. Extensive empirical evaluations validate the effectiveness of the proposed Discrete Markov Bridge, which achieves an Evidence Lower Bound (ELBO) of 1.38 on the Text8 dataset, outperforming established baselines. Moreover, the proposed model demonstrates competitive performance on the CIFAR-10 dataset, achieving results comparable to those obtained by image-specific generation approaches. 

---
# Distilling Closed-Source LLM's Knowledge for Locally Stable and Economic Biomedical Entity Linking 

**Authors**: Yihao Ai, Zhiyuan Ning, Weiwei Dai, Pengfei Wang, Yi Du, Wenjuan Cui, Kunpeng Liu, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.19722)  

**Abstract**: Biomedical entity linking aims to map nonstandard entities to standard entities in a knowledge base. Traditional supervised methods perform well but require extensive annotated data to transfer, limiting their usage in low-resource scenarios. Large language models (LLMs), especially closed-source LLMs, can address these but risk stability issues and high economic costs: using these models is restricted by commercial companies and brings significant economic costs when dealing with large amounts of data. To address this, we propose ``RPDR'', a framework combining closed-source LLMs and open-source LLMs for re-ranking candidates retrieved by a retriever fine-tuned with a small amount of data. By prompting a closed-source LLM to generate training data from unannotated data and fine-tuning an open-source LLM for re-ranking, we effectively distill the knowledge to the open-source LLM that can be deployed locally, thus avoiding the stability issues and the problem of high economic costs. We evaluate RPDR on two datasets, including one real-world dataset and one publicly available dataset involving two languages: Chinese and English. RPDR achieves 0.019 Acc@1 improvement and 0.036 Acc@1 improvement on the Aier dataset and the Ask A Patient dataset when the amount of training data is not enough. The results demonstrate the superiority and generalizability of the proposed framework. 

---
# OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction 

**Authors**: Juntong Wang, Xiyuan Wang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19719)  

**Abstract**: Common Neighbors (CNs) and their higher-order variants are important pairwise features widely used in state-of-the-art link prediction methods. However, existing methods often struggle with the repetition across different orders of CNs and fail to fully leverage their potential. We identify that these limitations stem from two key issues: redundancy and over-smoothing in high-order common neighbors. To address these challenges, we design orthogonalization to eliminate redundancy between different-order CNs and normalization to mitigate over-smoothing. By combining these two techniques, we propose Orthogonal Common Neighbor (OCN), a novel approach that significantly outperforms the strongest baselines by an average of 7.7% on popular link prediction benchmarks. A thorough theoretical analysis is provided to support our method. Ablation studies also verify the effectiveness of our orthogonalization and normalization techniques. 

---
# Graceful Forgetting in Generative Language Models 

**Authors**: Chunyang Jiang, Chi-min Chan, Yiyang Cai, Yulong Liu, Wei Xue, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.19715)  

**Abstract**: Recently, the pretrain-finetune paradigm has become a cornerstone in various deep learning areas. While in general the pre-trained model would promote both effectiveness and efficiency of downstream tasks fine-tuning, studies have shown that not all knowledge acquired during pre-training is beneficial. Some of the knowledge may actually bring detrimental effects to the fine-tuning tasks, which is also known as negative transfer. To address this problem, graceful forgetting has emerged as a promising approach. The core principle of graceful forgetting is to enhance the learning plasticity of the target task by selectively discarding irrelevant knowledge. However, this approach remains underexplored in the context of generative language models, and it is often challenging to migrate existing forgetting algorithms to these models due to architecture incompatibility. To bridge this gap, in this paper we propose a novel framework, Learning With Forgetting (LWF), to achieve graceful forgetting in generative language models. With Fisher Information Matrix weighting the intended parameter updates, LWF computes forgetting confidence to evaluate self-generated knowledge regarding the forgetting task, and consequently, knowledge with high confidence is periodically unlearned during fine-tuning. Our experiments demonstrate that, although thoroughly uncovering the mechanisms of knowledge interaction remains challenging in pre-trained language models, applying graceful forgetting can contribute to enhanced fine-tuning performance. 

---
# MT$^{3}$: Scaling MLLM-based Text Image Machine Translation via Multi-Task Reinforcement Learning 

**Authors**: Zhaopeng Feng, Yupu Liang, Shaosheng Cao, Jiayuan Su, Jiahan Ren, Zhe Xu, Yao Hu, Wenxuan Huang, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19714)  

**Abstract**: Text Image Machine Translation (TIMT)-the task of translating textual content embedded in images-is critical for applications in accessibility, cross-lingual information access, and real-world document understanding. However, TIMT remains a complex challenge due to the need for accurate optical character recognition (OCR), robust visual-text reasoning, and high-quality translation, often requiring cascading multi-stage pipelines. Recent advances in large-scale Reinforcement Learning (RL) have improved reasoning in Large Language Models (LLMs) and Multimodal LLMs (MLLMs), but their application to end-to-end TIMT is still underexplored. To bridge this gap, we introduce MT$^{3}$, the first framework to apply Multi-Task RL to MLLMs for end-to-end TIMT. MT$^{3}$ adopts a multi-task optimization paradigm targeting three key sub-skills: text recognition, context-aware reasoning, and translation. It is trained using a novel multi-mixed reward mechanism that adapts rule-based RL strategies to TIMT's intricacies, offering fine-grained, non-binary feedback across tasks. Furthermore, to facilitate the evaluation of TIMT in authentic cross-cultural and real-world social media contexts, we introduced XHSPost, the first social media TIMT benchmark. Our MT$^{3}$-7B-Zero achieves state-of-the-art results on the latest in-domain MIT-10M benchmark, outperforming strong baselines such as Qwen2.5-VL-72B and InternVL2.5-78B by notable margins across multiple metrics. Additionally, the model shows strong generalization to out-of-distribution language pairs and datasets. In-depth analyses reveal how multi-task synergy, reinforcement learning initialization, curriculum design, and reward formulation contribute to advancing MLLM-driven TIMT. 

---
# Error Typing for Smarter Rewards: Improving Process Reward Models with Error-Aware Hierarchical Supervision 

**Authors**: Tej Deep Pala, Panshul Sharma, Amir Zadeh, Chuan Li, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2505.19706)  

**Abstract**: Large Language Models (LLMs) are prone to hallucination, especially during multi-hop and reasoning-intensive tasks such as mathematical problem solving. While Outcome Reward Models verify only final answers, Process Reward Models (PRMs) score each intermediate step to steer generation toward coherent solutions. We introduce PathFinder-PRM, a novel hierarchical, error-aware discriminative PRM that first classifies math and consistency errors at each step, then combines these fine-grained signals to estimate step correctness. To train PathFinder-PRM, we construct a 400K-sample dataset by enriching the human-annotated PRM800K corpus and RLHFlow Mistral traces with three-dimensional step-level labels. On PRMBench, PathFinder-PRM achieves a new state-of-the-art PRMScore of 67.7, outperforming the prior best (65.5) while using 3 times less data. When applied to reward guided greedy search, our model yields prm@8 48.3, a +1.5 point gain over the strongest baseline. These results demonstrate that decoupled error detection and reward estimation not only boost fine-grained error detection but also substantially improve end-to-end, reward-guided mathematical reasoning with greater data efficiency. 

---
# Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models 

**Authors**: Yi Liu, Dianqing Liu, Mingye Zhu, Junbo Guo, Yongdong Zhang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19700)  

**Abstract**: The widespread adoption of large language models (LLMs) across industries has increased the demand for high-quality and customizable outputs. However, traditional alignment methods often require retraining large pretrained models, making it difficult to quickly adapt and optimize LLMs for diverse applications. To address this limitation, we propose a novel \textit{Residual Alignment Model} (\textit{RAM}) that formalizes the alignment process as a type of importance sampling. In this framework, the unaligned upstream model serves as the proposal distribution, while the alignment process is framed as secondary sampling based on an autoregressive alignment module that acts as an estimator of the importance weights. This design enables a natural detachment of the alignment module from the target aligned model, improving flexibility and scalability. Based on this model, we derive an efficient sequence-level training strategy for the alignment module, which operates independently of the proposal module. Additionally, we develop a resampling algorithm with iterative token-level decoding to address the common first-token latency issue in comparable methods. Experimental evaluations on two leading open-source LLMs across diverse tasks, including instruction following, domain adaptation, and preference optimization, demonstrate that our approach consistently outperforms baseline models. 

---
# Mosaic: Data-Free Knowledge Distillation via Mixture-of-Experts for Heterogeneous Distributed Environments 

**Authors**: Junming Liu, Yanting Gao, Siyuan Meng, Yifei Sun, Aoqi Wu, Yufei Jin, Yirong Chen, Ding Wang, Guosun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19699)  

**Abstract**: Federated Learning (FL) is a decentralized machine learning paradigm that enables clients to collaboratively train models while preserving data privacy. However, the coexistence of model and data heterogeneity gives rise to inconsistent representations and divergent optimization dynamics across clients, ultimately hindering robust global performance. To transcend these challenges, we propose Mosaic, a novel data-free knowledge distillation framework tailored for heterogeneous distributed environments. Mosaic first trains local generative models to approximate each client's personalized distribution, enabling synthetic data generation that safeguards privacy through strict separation from real data. Subsequently, Mosaic forms a Mixture-of-Experts (MoE) from client models based on their specialized knowledge, and distills it into a global model using the generated data. To further enhance the MoE architecture, Mosaic integrates expert predictions via a lightweight meta model trained on a few representative prototypes. Extensive experiments on standard image classification benchmarks demonstrate that Mosaic consistently outperforms state-of-the-art approaches under both model and data heterogeneity. The source code has been published at this https URL. 

---
# JEDI: Latent End-to-end Diffusion Mitigates Agent-Human Performance Asymmetry in Model-Based Reinforcement Learning 

**Authors**: Jing Yu Lim, Zarif Ikram, Samson Yu, Haozhe Ma, Tze-Yun Leong, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19698)  

**Abstract**: Recent advances in model-based reinforcement learning (MBRL) have achieved super-human level performance on the Atari100k benchmark, driven by reinforcement learning agents trained on powerful diffusion world models. However, we identify that the current aggregates mask a major performance asymmetry: MBRL agents dramatically outperform humans in some tasks despite drastically underperforming in others, with the former inflating the aggregate metrics. This is especially pronounced in pixel-based agents trained with diffusion world models. In this work, we address the pronounced asymmetry observed in pixel-based agents as an initial attempt to reverse the worrying upward trend observed in them. We address the problematic aggregates by delineating all tasks as Agent-Optimal or Human-Optimal and advocate for equal importance on metrics from both sets. Next, we hypothesize this pronounced asymmetry is due to the lack of temporally-structured latent space trained with the World Model objective in pixel-based methods. Lastly, to address this issue, we propose Joint Embedding DIffusion (JEDI), a novel latent diffusion world model trained end-to-end with the self-consistency objective. JEDI outperforms SOTA models in human-optimal tasks while staying competitive across the Atari100k benchmark, and runs 3 times faster with 43% lower memory than the latest pixel-based diffusion baseline. Overall, our work rethinks what it truly means to cross human-level performance in Atari100k. 

---
# EmoSphere-SER: Enhancing Speech Emotion Recognition Through Spherical Representation with Auxiliary Classification 

**Authors**: Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.19693)  

**Abstract**: Speech emotion recognition predicts a speaker's emotional state from speech signals using discrete labels or continuous dimensions such as arousal, valence, and dominance (VAD). We propose EmoSphere-SER, a joint model that integrates spherical VAD region classification to guide VAD regression for improved emotion prediction. In our framework, VAD values are transformed into spherical coordinates that are divided into multiple spherical regions, and an auxiliary classification task predicts which spherical region each point belongs to, guiding the regression process. Additionally, we incorporate a dynamic weighting scheme and a style pooling layer with multi-head self-attention to capture spectral and temporal dynamics, further boosting performance. This combined training strategy reinforces structured learning and improves prediction consistency. Experimental results show that our approach exceeds baseline methods, confirming the validity of the proposed framework. 

---
# DiEmo-TTS: Disentangled Emotion Representations via Self-Supervised Distillation for Cross-Speaker Emotion Transfer in Text-to-Speech 

**Authors**: Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.19687)  

**Abstract**: Cross-speaker emotion transfer in speech synthesis relies on extracting speaker-independent emotion embeddings for accurate emotion modeling without retaining speaker traits. However, existing timbre compression methods fail to fully separate speaker and emotion characteristics, causing speaker leakage and degraded synthesis quality. To address this, we propose DiEmo-TTS, a self-supervised distillation method to minimize emotional information loss and preserve speaker identity. We introduce cluster-driven sampling and information perturbation to preserve emotion while removing irrelevant factors. To facilitate this process, we propose an emotion clustering and matching approach using emotional attribute prediction and speaker embeddings, enabling generalization to unlabeled data. Additionally, we designed a dual conditioning transformer to integrate style features better. Experimental results confirm the effectiveness of our method in learning speaker-irrelevant emotion embeddings. 

---
# KIT's Low-resource Speech Translation Systems for IWSLT2025: System Enhancement with Synthetic Data and Model Regularization 

**Authors**: Zhaolin Li, Yining Liu, Danni Liu, Tuan Nam Nguyen, Enes Yavuz Ugan, Tu Anh Dinh, Carlos Mullov, Alexander Waibel, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2505.19679)  

**Abstract**: This paper presents KIT's submissions to the IWSLT 2025 low-resource track. We develop both cascaded systems, consisting of Automatic Speech Recognition (ASR) and Machine Translation (MT) models, and end-to-end (E2E) Speech Translation (ST) systems for three language pairs: Bemba, North Levantine Arabic, and Tunisian Arabic into English. Building upon pre-trained models, we fine-tune our systems with different strategies to utilize resources efficiently. This study further explores system enhancement with synthetic data and model regularization. Specifically, we investigate MT-augmented ST by generating translations from ASR data using MT models. For North Levantine, which lacks parallel ST training data, a system trained solely on synthetic data slightly surpasses the cascaded system trained on real data. We also explore augmentation using text-to-speech models by generating synthetic speech from MT data, demonstrating the benefits of synthetic data in improving both ASR and ST performance for Bemba. Additionally, we apply intra-distillation to enhance model performance. Our experiments show that this approach consistently improves results across ASR, MT, and ST tasks, as well as across different pre-trained models. Finally, we apply Minimum Bayes Risk decoding to combine the cascaded and end-to-end systems, achieving an improvement of approximately 1.5 BLEU points. 

---
# Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement 

**Authors**: Liqin Ye, Agam Shah, Chao Zhang, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2505.19675)  

**Abstract**: The traditional process of creating labeled datasets is labor-intensive and expensive. Recent breakthroughs in open-source large language models (LLMs) have opened up a new avenue in generating labeled datasets automatically for various natural language processing (NLP) tasks, providing an alternative to such an expensive annotation process. However, the reliability of such auto-generated labels remains a significant concern due to inherent inaccuracies. When learning from noisy labels, the model's generalization is likely to be harmed as it is prone to overfit to those label noises. While previous studies in learning from noisy labels mainly focus on synthetic noise and real-world noise, LLM-generated label noise receives less attention. In this paper, we propose SiDyP: Simplex Label Diffusion with Dynamic Prior to calibrate the classifier's prediction, thus enhancing its robustness towards LLM-generated noisy labels. SiDyP retrieves potential true label candidates by neighborhood label distribution in text embedding space and iteratively refines noisy candidates using a simplex diffusion model. Our framework can increase the performance of the BERT classifier fine-tuned on both zero-shot and few-shot LLM-generated noisy label datasets by an average of 7.21% and 7.30% respectively. We demonstrate the effectiveness of SiDyP by conducting extensive benchmarking for different LLMs over a variety of NLP tasks. Our code is available on Github. 

---
# Automated evaluation of children's speech fluency for low-resource languages 

**Authors**: Bowen Zhang, Nur Afiqah Abdul Latiff, Justin Kan, Rong Tong, Donny Soh, Xiaoxiao Miao, Ian McLoughlin  

**Link**: [PDF](https://arxiv.org/pdf/2505.19671)  

**Abstract**: Assessment of children's speaking fluency in education is well researched for majority languages, but remains highly challenging for low resource languages. This paper proposes a system to automatically assess fluency by combining a fine-tuned multilingual ASR model, an objective metrics extraction stage, and a generative pre-trained transformer (GPT) network. The objective metrics include phonetic and word error rates, speech rate, and speech-pause duration ratio. These are interpreted by a GPT-based classifier guided by a small set of human-evaluated ground truth examples, to score fluency. We evaluate the proposed system on a dataset of children's speech in two low-resource languages, Tamil and Malay and compare the classification performance against Random Forest and XGBoost, as well as using ChatGPT-4o to predict fluency directly from speech input. Results demonstrate that the proposed approach achieves significantly higher accuracy than multimodal GPT or other methods. 

---
# LeCoDe: A Benchmark Dataset for Interactive Legal Consultation Dialogue Evaluation 

**Authors**: Weikang Yuan, Kaisong Song, Zhuoren Jiang, Junjie Cao, Yujie Zhang, Jun Lin, Kun Kuang, Ji Zhang, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19667)  

**Abstract**: Legal consultation is essential for safeguarding individual rights and ensuring access to justice, yet remains costly and inaccessible to many individuals due to the shortage of professionals. While recent advances in Large Language Models (LLMs) offer a promising path toward scalable, low-cost legal assistance, current systems fall short in handling the interactive and knowledge-intensive nature of real-world consultations. To address these challenges, we introduce LeCoDe, a real-world multi-turn benchmark dataset comprising 3,696 legal consultation dialogues with 110,008 dialogue turns, designed to evaluate and improve LLMs' legal consultation capability. With LeCoDe, we innovatively collect live-streamed consultations from short-video platforms, providing authentic multi-turn legal consultation dialogues. The rigorous annotation by legal experts further enhances the dataset with professional insights and expertise. Furthermore, we propose a comprehensive evaluation framework that assesses LLMs' consultation capabilities in terms of (1) clarification capability and (2) professional advice quality. This unified framework incorporates 12 metrics across two dimensions. Through extensive experiments on various general and domain-specific LLMs, our results reveal significant challenges in this task, with even state-of-the-art models like GPT-4 achieving only 39.8% recall for clarification and 59% overall score for advice quality, highlighting the complexity of professional consultation scenarios. Based on these findings, we further explore several strategies to enhance LLMs' legal consultation abilities. Our benchmark contributes to advancing research in legal domain dialogue systems, particularly in simulating more real-world user-expert interactions. 

---
# A Comprehensive Real-World Assessment of Audio Watermarking Algorithms: Will They Survive Neural Codecs? 

**Authors**: Yigitcan Özer, Woosung Choi, Joan Serrà, Mayank Kumar Singh, Wei-Hsiang Liao, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2505.19663)  

**Abstract**: We present a framework to foster the evaluation of deep learning-based audio watermarking algorithms, establishing a standardized benchmark and allowing systematic comparisons. To simulate real-world usage, we introduce a comprehensive audio attack pipeline, featuring various distortions such as compression, background noise, and reverberation, and propose a diverse test dataset, including speech, environmental sounds, and music recordings. By assessing the performance of four existing watermarking algorithms on our framework, two main insights stand out: (i) neural compression techniques pose the most significant challenge, even when algorithms are trained with such compressions; and (ii) training with audio attacks generally improves robustness, although it is insufficient in some cases. Furthermore, we find that specific distortions, such as polarity inversion, time stretching, or reverb, seriously affect certain algorithms. Our contributions strengthen the robustness and perceptual assessment of audio watermarking algorithms across a wide range of applications, while ensuring a fair and consistent evaluation approach. The evaluation framework, including the attack pipeline, is accessible at this http URL. 

---
# GenKI: Enhancing Open-Domain Question Answering with Knowledge Integration and Controllable Generation in Large Language Models 

**Authors**: Tingjia Shen, Hao Wang, Chuan Qin, Ruijun Sun, Yang Song, Defu Lian, Hengshu Zhu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19660)  

**Abstract**: Open-domain question answering (OpenQA) represents a cornerstone in natural language processing (NLP), primarily focused on extracting answers from unstructured textual data. With the rapid advancements in Large Language Models (LLMs), LLM-based OpenQA methods have reaped the benefits of emergent understanding and answering capabilities enabled by massive parameters compared to traditional methods. However, most of these methods encounter two critical challenges: how to integrate knowledge into LLMs effectively and how to adaptively generate results with specific answer formats for various task situations. To address these challenges, we propose a novel framework named GenKI, which aims to improve the OpenQA performance by exploring Knowledge Integration and controllable Generation on LLMs simultaneously. Specifically, we first train a dense passage retrieval model to retrieve associated knowledge from a given knowledge base. Subsequently, we introduce a novel knowledge integration model that incorporates the retrieval knowledge into instructions during fine-tuning to intensify the model. Furthermore, to enable controllable generation in LLMs, we leverage a certain fine-tuned LLM and an ensemble based on text consistency incorporating all coherence, fluency, and answer format assurance. Finally, extensive experiments conducted on the TriviaQA, MSMARCO, and CMRC2018 datasets, featuring diverse answer formats, have demonstrated the effectiveness of GenKI with comparison of state-of-the-art baselines. Moreover, ablation studies have disclosed a linear relationship between the frequency of retrieved knowledge and the model's ability to recall knowledge accurately against the ground truth. Our code of GenKI is available at this https URL 

---
# Large Language Models in Code Co-generation for Safe Autonomous Vehicles 

**Authors**: Ali Nouri, Beatriz Cabrero-Daniel, Zhennan Fei, Krishna Ronanki, Håkan Sivencrona, Christian Berger  

**Link**: [PDF](https://arxiv.org/pdf/2505.19658)  

**Abstract**: Software engineers in various industrial domains are already using Large Language Models (LLMs) to accelerate the process of implementing parts of software systems. When considering its potential use for ADAS or AD systems in the automotive context, there is a need to systematically assess this new setup: LLMs entail a well-documented set of risks for safety-related systems' development due to their stochastic nature. To reduce the effort for code reviewers to evaluate LLM-generated code, we propose an evaluation pipeline to conduct sanity-checks on the generated code. We compare the performance of six state-of-the-art LLMs (CodeLlama, CodeGemma, DeepSeek-r1, DeepSeek-Coders, Mistral, and GPT-4) on four safety-related programming tasks. Additionally, we qualitatively analyse the most frequent faults generated by these LLMs, creating a failure-mode catalogue to support human reviewers. Finally, the limitations and capabilities of LLMs in code generation, and the use of the proposed pipeline in the existing process, are discussed. 

---
# Model Enumeration of Two-Variable Logic with Quadratic Delay Complexity 

**Authors**: Qiaolan Meng, Juhua Pu, Hongting Niu, Yuyi Wang, Yuanhong Wang, Ondřej Kuželka  

**Link**: [PDF](https://arxiv.org/pdf/2505.19648)  

**Abstract**: We study the model enumeration problem of the function-free, finite domain fragment of first-order logic with two variables ($FO^2$). Specifically, given an $FO^2$ sentence $\Gamma$ and a positive integer $n$, how can one enumerate all the models of $\Gamma$ over a domain of size $n$? In this paper, we devise a novel algorithm to address this problem. The delay complexity, the time required between producing two consecutive models, of our algorithm is quadratic in the given domain size $n$ (up to logarithmic factors) when the sentence is fixed. This complexity is almost optimal since the interpretation of binary predicates in any model requires at least $\Omega(n^2)$ bits to represent. 

---
# MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE 

**Authors**: Zongle Huang, Lei Zhu, Zongyuan Zhan, Ting Hu, Weikai Mao, Xianzhi Yu, Yongpan Liu, Tianyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19645)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across many applications, with Mixture of Experts (MoE) models demonstrating great potential. Compared to traditional dense models, MoEs achieve better performance with less computation. Speculative decoding (SD) is a widely used technique to accelerate LLM inference without accuracy loss, but it has been considered efficient only for dense models. In this work, we first demonstrate that, under medium batch sizes, MoE surprisingly benefits more from SD than dense models. Furthermore, as MoE becomes sparser -- the prevailing trend in MoE designs -- the batch size range where SD acceleration is expected to be effective becomes broader. To quantitatively understand tradeoffs involved in SD, we develop a reliable modeling based on theoretical analyses. While current SD research primarily focuses on improving acceptance rates of algorithms, changes in workload and model architecture can still lead to degraded SD acceleration even with high acceptance rates. To address this limitation, we introduce a new metric 'target efficiency' that characterizes these effects, thus helping researchers identify system bottlenecks and understand SD acceleration more comprehensively. For scenarios like private serving, this work unveils a new perspective to speed up MoE inference, where existing solutions struggle. Experiments on different GPUs show up to 2.29x speedup for Qwen2-57B-A14B at medium batch sizes and validate our theoretical predictions. 

---
# STOPA: A Database of Systematic VariaTion Of DeePfake Audio for Open-Set Source Tracing and Attribution 

**Authors**: Anton Firc, Manasi Chibber, Jagabandhu Mishra, Vishwanath Pratap Singh, Tomi Kinnunen, Kamil Malinka  

**Link**: [PDF](https://arxiv.org/pdf/2505.19644)  

**Abstract**: A key research area in deepfake speech detection is source tracing - determining the origin of synthesised utterances. The approaches may involve identifying the acoustic model (AM), vocoder model (VM), or other generation-specific parameters. However, progress is limited by the lack of a dedicated, systematically curated dataset. To address this, we introduce STOPA, a systematically varied and metadata-rich dataset for deepfake speech source tracing, covering 8 AMs, 6 VMs, and diverse parameter settings across 700k samples from 13 distinct synthesisers. Unlike existing datasets, which often feature limited variation or sparse metadata, STOPA provides a systematically controlled framework covering a broader range of generative factors, such as the choice of the vocoder model, acoustic model, or pretrained weights, ensuring higher attribution reliability. This control improves attribution accuracy, aiding forensic analysis, deepfake detection, and generative model transparency. 

---
# Segment First or Comprehend First? Explore the Limit of Unsupervised Word Segmentation with Large Language Models 

**Authors**: Zihong Zhang, Liqi He, Zuchao Li, Lefei Zhang, Hai Zhao, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.19631)  

**Abstract**: Word segmentation stands as a cornerstone of Natural Language Processing (NLP). Based on the concept of "comprehend first, segment later", we propose a new framework to explore the limit of unsupervised word segmentation with Large Language Models (LLMs) and evaluate the semantic understanding capabilities of LLMs based on word segmentation. We employ current mainstream LLMs to perform word segmentation across multiple languages to assess LLMs' "comprehension". Our findings reveal that LLMs are capable of following simple prompts to segment raw text into words. There is a trend suggesting that models with more parameters tend to perform better on multiple languages. Additionally, we introduce a novel unsupervised method, termed LLACA ($\textbf{L}$arge $\textbf{L}$anguage Model-Inspired $\textbf{A}$ho-$\textbf{C}$orasick $\textbf{A}$utomaton). Leveraging the advanced pattern recognition capabilities of Aho-Corasick automata, LLACA innovatively combines these with the deep insights of well-pretrained LLMs. This approach not only enables the construction of a dynamic $n$-gram model that adjusts based on contextual information but also integrates the nuanced understanding of LLMs, offering significant improvements over traditional methods. Our source code is available at this https URL 

---
# Search-Based Software Engineering in the Landscape of AI Foundation Models 

**Authors**: Hassan Sartaj, Shaukat Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.19625)  

**Abstract**: Search-based software engineering (SBSE), at the intersection of artificial intelligence (AI) and software engineering, has been an active area of research for about 25 years. It has been applied to solve numerous problems across the entire software engineering lifecycle and has demonstrated its versatility in multiple domains. With the recent advancements in AI, particularly the emergence of foundation models (FMs), the evolution of SBSE alongside FMs remains undetermined. In this window of opportunity, we propose a research roadmap that articulates the current landscape of SBSE in relation to foundation models (FMs), highlights open challenges, and outlines potential research directions for advancing SBSE through its interplay with FMs. This roadmap aims to establish a forward-thinking and innovative perspective for the future of SBSE in the era of FMs. 

---
# Benchmarking Large Multimodal Models for Ophthalmic Visual Question Answering with OphthalWeChat 

**Authors**: Pusheng Xu, Xia Gong, Xiaolan Chen, Weiyi Zhang, Jiancheng Yang, Bingjie Yan, Meng Yuan, Yalin Zheng, Mingguang He, Danli Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19624)  

**Abstract**: Purpose: To develop a bilingual multimodal visual question answering (VQA) benchmark for evaluating VLMs in ophthalmology. Methods: Ophthalmic image posts and associated captions published between January 1, 2016, and December 31, 2024, were collected from WeChat Official Accounts. Based on these captions, bilingual question-answer (QA) pairs in Chinese and English were generated using GPT-4o-mini. QA pairs were categorized into six subsets by question type and language: binary (Binary_CN, Binary_EN), single-choice (Single-choice_CN, Single-choice_EN), and open-ended (Open-ended_CN, Open-ended_EN). The benchmark was used to evaluate the performance of three VLMs: GPT-4o, Gemini 2.0 Flash, and Qwen2.5-VL-72B-Instruct. Results: The final OphthalWeChat dataset included 3,469 images and 30,120 QA pairs across 9 ophthalmic subspecialties, 548 conditions, 29 imaging modalities, and 68 modality combinations. Gemini 2.0 Flash achieved the highest overall accuracy (0.548), outperforming GPT-4o (0.522, P < 0.001) and Qwen2.5-VL-72B-Instruct (0.514, P < 0.001). It also led in both Chinese (0.546) and English subsets (0.550). Subset-specific performance showed Gemini 2.0 Flash excelled in Binary_CN (0.687), Single-choice_CN (0.666), and Single-choice_EN (0.646), while GPT-4o ranked highest in Binary_EN (0.717), Open-ended_CN (BLEU-1: 0.301; BERTScore: 0.382), and Open-ended_EN (BLEU-1: 0.183; BERTScore: 0.240). Conclusions: This study presents the first bilingual VQA benchmark for ophthalmology, distinguished by its real-world context and inclusion of multiple examinations per patient. The dataset reflects authentic clinical decision-making scenarios and enables quantitative evaluation of VLMs, supporting the development of accurate, specialized, and trustworthy AI systems for eye care. 

---
# AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems 

**Authors**: Yu Shang, Peijie Liu, Yuwei Yan, Zijing Wu, Leheng Sheng, Yuanqing Yu, Chumeng Jiang, An Zhang, Fengli Xu, Yu Wang, Min Zhang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19623)  

**Abstract**: The emergence of agentic recommender systems powered by Large Language Models (LLMs) represents a paradigm shift in personalized recommendations, leveraging LLMs' advanced reasoning and role-playing capabilities to enable autonomous, adaptive decision-making. Unlike traditional recommendation approaches, agentic recommender systems can dynamically gather and interpret user-item interactions from complex environments, generating robust recommendation strategies that generalize across diverse scenarios. However, the field currently lacks standardized evaluation protocols to systematically assess these methods. To address this critical gap, we propose: (1) an interactive textual recommendation simulator incorporating rich user and item metadata and three typical evaluation scenarios (classic, evolving-interest, and cold-start recommendation tasks); (2) a unified modular framework for developing and studying agentic recommender systems; and (3) the first comprehensive benchmark comparing 10 classical and agentic recommendation methods. Our findings demonstrate the superiority of agentic systems and establish actionable design guidelines for their core components. The benchmark environment has been rigorously validated through an open challenge and remains publicly available with a continuously maintained leaderboard~\footnote[2]{this https URL}, fostering ongoing community engagement and reproducible research. The benchmark is available at: \hyperlink{this https URL}{this https URL}. 

---
# Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs 

**Authors**: Jiawen Chen, Qi Shao, Duxin Chen, Wenwu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19620)  

**Abstract**: Spatio-temporal prediction is a pivotal task with broad applications in traffic management, climate monitoring, energy scheduling, etc. However, existing methodologies often struggle to balance model expressiveness and computational efficiency, especially when scaling to large real-world datasets. To tackle these challenges, we propose STH-SepNet (Spatio-Temporal Hypergraph Separation Networks), a novel framework that decouples temporal and spatial modeling to enhance both efficiency and precision. Therein, the temporal dimension is modeled using lightweight large language models, which effectively capture low-rank temporal dynamics. Concurrently, the spatial dimension is addressed through an adaptive hypergraph neural network, which dynamically constructs hyperedges to model intricate, higher-order interactions. A carefully designed gating mechanism is integrated to seamlessly fuse temporal and spatial representations. By leveraging the fundamental principles of low-rank temporal dynamics and spatial interactions, STH-SepNet offers a pragmatic and scalable solution for spatio-temporal prediction in real-world applications. Extensive experiments on large-scale real-world datasets across multiple benchmarks demonstrate the effectiveness of STH-SepNet in boosting predictive performance while maintaining computational efficiency. This work may provide a promising lightweight framework for spatio-temporal prediction, aiming to reduce computational demands and while enhancing predictive performance. Our code is avaliable at this https URL. 

---
# Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models 

**Authors**: Rui Cai, Bangzheng Li, Xiaofei Wen, Muhao Chen, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19616)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across tasks, yet they often exhibit difficulty in distinguishing task-relevant from irrelevant signals, particularly in tasks like Visual Question Answering (VQA), which can lead to susceptibility to misleading or spurious inputs. We refer to this broader limitation as the Cross-Modality Competency Problem: the model's inability to fairly evaluate all modalities. This vulnerability becomes more evident in modality-specific tasks such as image classification or pure text question answering, where models are expected to rely solely on one modality. In such tasks, spurious information from irrelevant modalities often leads to significant performance degradation. We refer to this failure as Modality Interference, which serves as a concrete and measurable instance of the cross-modality competency problem. We further design a perturbation-based causal diagnostic experiment to verify and quantify this problem. To mitigate modality interference, we propose a novel framework to fine-tune MLLMs, including perturbation-based data augmentations with both heuristic perturbations and adversarial perturbations via Projected Gradient Descent (PGD), and a consistency regularization strategy applied to model outputs with original and perturbed inputs. Experiments on multiple benchmark datasets (image-heavy, text-heavy, and VQA tasks) and multiple model families with different scales demonstrate significant improvements in robustness and cross-modality competency, indicating our method's effectiveness in boosting unimodal reasoning ability while enhancing performance on multimodal tasks. 

---
# Align and Surpass Human Camouflaged Perception: Visual Refocus Reinforcement Fine-Tuning 

**Authors**: Ruolin Shen, Xiaozhong Ji, Kai WU, Jiangning Zhang, Yijun He, HaiHua Yang, Xiaobin Hu, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.19611)  

**Abstract**: Current multi-modal models exhibit a notable misalignment with the human visual system when identifying objects that are visually assimilated into the background. Our observations reveal that these multi-modal models cannot distinguish concealed objects, demonstrating an inability to emulate human cognitive processes which effectively utilize foreground-background similarity principles for visual analysis. To analyze this hidden human-model visual thinking discrepancy, we build a visual system that mimicks human visual camouflaged perception to progressively and iteratively `refocus' visual concealed content. The refocus is a progressive guidance mechanism enabling models to logically localize objects in visual images through stepwise reasoning. The localization process of concealed objects requires hierarchical attention shifting with dynamic adjustment and refinement of prior cognitive knowledge. In this paper, we propose a visual refocus reinforcement framework via the policy optimization algorithm to encourage multi-modal models to think and refocus more before answering, and achieve excellent reasoning abilities to align and even surpass human camouflaged perception systems. Our extensive experiments on camouflaged perception successfully demonstrate the emergence of refocus visual phenomena, characterized by multiple reasoning tokens and dynamic adjustment of the detection box. Besides, experimental results on both camouflaged object classification and detection tasks exhibit significantly superior performance compared to Supervised Fine-Tuning (SFT) baselines. 

---
# Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling 

**Authors**: Hongtao Xu, Wenting Shen, Yuanxin Wei, Ang Wang, Guo Runfan, Tianxing Wang, Yong Li, Mingzhen Li, Weile Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.19609)  

**Abstract**: Long-context supervised fine-tuning (Long-SFT) plays a vital role in enhancing the performance of large language models (LLMs) on long-context tasks. To smoothly adapt LLMs to long-context scenarios, this process typically entails training on mixed datasets containing both long and short sequences. However, this heterogeneous sequence length distribution poses significant challenges for existing training systems, as they fail to simultaneously achieve high training efficiency for both long and short sequences, resulting in sub-optimal end-to-end system performance in Long-SFT. In this paper, we present a novel perspective on data scheduling to address the challenges posed by the heterogeneous data distributions in Long-SFT. We propose Skrull, a dynamic data scheduler specifically designed for efficient long-SFT. Through dynamic data scheduling, Skrull balances the computation requirements of long and short sequences, improving overall training efficiency. Furthermore, we formulate the scheduling process as a joint optimization problem and thoroughly analyze the trade-offs involved. Based on those analysis, Skrull employs a lightweight scheduling algorithm to achieve near-zero cost online scheduling in Long-SFT. Finally, we implement Skrull upon DeepSpeed, a state-of-the-art distributed training system for LLMs. Experimental results demonstrate that Skrull outperforms DeepSpeed by 3.76x on average (up to 7.54x) in real-world long-SFT scenarios. 

---
# Energy-based Preference Optimization for Test-time Adaptation 

**Authors**: Yewon Han, Seoyun Yang, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.19607)  

**Abstract**: Test-Time Adaptation (TTA) enhances model robustness by enabling adaptation to target distributions that differ from training distributions, improving real-world generalizability. Existing TTA approaches focus on adjusting the conditional distribution; however these methods often depend on uncertain predictions in the absence of label information, leading to unreliable performance. Energy-based frameworks suggest a promising alternative to address distribution shifts without relying on uncertain predictions, instead computing the marginal distribution of target data. However, they involve the critical challenge of requiring extensive SGLD sampling, which is impractical for test-time scenarios requiring immediate adaptation. In this work, we propose Energy-based Preference Optimization for Test-time Adaptation (EPOTTA), which is based on a sampling free strategy. We first parameterize the target model using a pretrained model and residual energy function, enabling marginal likelihood maximization of target data without sampling. Building on the observation that the parameterization is mathematically equivalent to DPO objective, we then directly adapt the model to a target distribution without explicitly training the residual. Our experiments verify that EPOTTA is well-calibrated and performant while achieving computational efficiency. 

---
# Preference Optimization by Estimating the Ratio of the Data Distribution 

**Authors**: Yeongmin Kim, Heesun Bae, Byeonghu Na, Il-Chul Moon  

**Link**: [PDF](https://arxiv.org/pdf/2505.19601)  

**Abstract**: Direct preference optimization (DPO) is widely used as a simple and stable method for aligning large language models (LLMs) with human preferences. This paper investigates a generalized DPO loss that enables a policy model to match the target policy from a likelihood ratio estimation perspective. The ratio of the target policy provides a unique identification of the policy distribution without relying on reward models or partition functions. This allows the generalized loss to retain both simplicity and theoretical guarantees, which prior work such as $f$-PO fails to achieve simultaneously. We propose Bregman preference optimization (BPO), a generalized framework for ratio matching that provides a family of objective functions achieving target policy optimality. BPO subsumes DPO as a special case and offers tractable forms for all instances, allowing implementation with a few lines of code. We further develop scaled Basu's power divergence (SBA), a gradient scaling method that can be used for BPO instances. The BPO framework complements other DPO variants and is applicable to target policies defined by these variants. In experiments, unlike other probabilistic loss extensions such as $f$-DPO or $f$-PO, which exhibit a trade-off between generation fidelity and diversity, instances of BPO improve both win rate and entropy compared with DPO. When applied to Llama-3-Instruct-8B, BPO achieves state-of-the-art performance among Llama-3-8B backbones, with a 55.9\% length-controlled win rate on AlpacaEval2. 

---
# Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar 

**Authors**: Andrew Gambardella, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.19599)  

**Abstract**: Typical methods for evaluating the performance of language models evaluate their ability to answer questions accurately. These evaluation metrics are acceptable for determining the extent to which language models can understand and reason about text in a general sense, but fail to capture nuanced capabilities, such as the ability of language models to recognize and obey rare grammar points, particularly in languages other than English. We measure the perplexity of language models when confronted with the "first person psych predicate restriction" grammar point in Japanese. Weblab is the only tested open source model in the 7-10B parameter range which consistently assigns higher perplexity to ungrammatical psych predicate sentences than grammatical ones. We give evidence that Weblab's uniformly bad tokenization is a possible root cause for its good performance, and show that Llama 3's perplexity on grammatical psych predicate sentences can be reduced by orders of magnitude (28x difference) by restricting test sentences to those with uniformly well-behaved tokenizations. We show in further experiments on machine translation tasks that language models will use alternative grammar patterns in order to produce grammatical sentences when tokenization issues prevent the most natural sentence from being output. 

---
# Multi-Agent Collaboration via Evolving Orchestration 

**Authors**: Yufan Dang, Chen Qian, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng Yang, Xiaoyin Che, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.19591)  

**Abstract**: Large language models (LLMs) have achieved remarkable results across diverse downstream tasks, but their monolithic nature restricts scalability and efficiency in complex problem-solving. While recent research explores multi-agent collaboration among LLMs, most approaches rely on static organizational structures that struggle to adapt as task complexity and agent numbers grow, resulting in coordination overhead and inefficiencies. To this end, we propose a puppeteer-style paradigm for LLM-based multi-agent collaboration, where a centralized orchestrator ("puppeteer") dynamically directs agents ("puppets") in response to evolving task states. This orchestrator is trained via reinforcement learning to adaptively sequence and prioritize agents, enabling flexible and evolvable collective reasoning. Experiments on closed- and open-domain scenarios show that this method achieves superior performance with reduced computational costs. Analyses further reveal that the key improvements consistently stem from the emergence of more compact, cyclic reasoning structures under the orchestrator's evolution. 

---
# LogiCoL: Logically-Informed Contrastive Learning for Set-based Dense Retrieval 

**Authors**: Yanzhen Shen, Sihao Chen, Xueqiang Xu, Yunyi Zhang, Chaitanya Malaviya, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2505.19588)  

**Abstract**: While significant progress has been made with dual- and bi-encoder dense retrievers, they often struggle on queries with logical connectives, a use case that is often overlooked yet important in downstream applications. Current dense retrievers struggle with such queries, such that the retrieved results do not respect the logical constraints implied in the queries. To address this challenge, we introduce LogiCoL, a logically-informed contrastive learning objective for dense retrievers. LogiCoL builds upon in-batch supervised contrastive learning, and learns dense retrievers to respect the subset and mutually-exclusive set relation between query results via two sets of soft constraints expressed via t-norm in the learning objective. We evaluate the effectiveness of LogiCoL on the task of entity retrieval, where the model is expected to retrieve a set of entities in Wikipedia that satisfy the implicit logical constraints in the query. We show that models trained with LogiCoL yield improvement both in terms of retrieval performance and logical consistency in the results. We provide detailed analysis and insights to uncover why queries with logical connectives are challenging for dense retrievers and why LogiCoL is most effective. 

---
# Accelerating Prefilling for Long-Context LLMs via Sparse Pattern Sharing 

**Authors**: Dan Peng, Zhihui Fu, Zewen Ye, Zhuoran Song, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19578)  

**Abstract**: Sparse attention methods exploit the inherent sparsity in attention to speed up the prefilling phase of long-context inference, mitigating the quadratic complexity of full attention computation. While existing sparse attention methods rely on predefined patterns or inaccurate estimations to approximate attention behavior, they often fail to fully capture the true dynamics of attention, resulting in reduced efficiency and compromised accuracy. Instead, we propose a highly accurate sparse attention mechanism that shares similar yet precise attention patterns across heads, enabling a more realistic capture of the dynamic behavior of attention. Our approach is grounded in two key observations: (1) attention patterns demonstrate strong inter-head similarity, and (2) this similarity remains remarkably consistent across diverse inputs. By strategically sharing computed accurate patterns across attention heads, our method effectively captures actual patterns while requiring full attention computation for only a small subset of heads. Comprehensive evaluations demonstrate that our approach achieves superior or comparable speedup relative to state-of-the-art methods while delivering the best overall accuracy. 

---
# Situationally-Aware Dynamics Learning 

**Authors**: Alejandro Murillo-Gonzalez, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19574)  

**Abstract**: Autonomous robots operating in complex, unstructured environments face significant challenges due to latent, unobserved factors that obscure their understanding of both their internal state and the external world. Addressing this challenge would enable robots to develop a more profound grasp of their operational context. To tackle this, we propose a novel framework for online learning of hidden state representations, with which the robots can adapt in real-time to uncertain and dynamic conditions that would otherwise be ambiguous and result in suboptimal or erroneous behaviors. Our approach is formalized as a Generalized Hidden Parameter Markov Decision Process, which explicitly models the influence of unobserved parameters on both transition dynamics and reward structures. Our core innovation lies in learning online the joint distribution of state transitions, which serves as an expressive representation of latent ego- and environmental-factors. This probabilistic approach supports the identification and adaptation to different operational situations, improving robustness and safety. Through a multivariate extension of Bayesian Online Changepoint Detection, our method segments changes in the underlying data generating process governing the robot's dynamics. The robot's transition model is then informed with a symbolic representation of the current situation derived from the joint distribution of latest state transitions, enabling adaptive and context-aware decision-making. To showcase the real-world effectiveness, we validate our approach in the challenging task of unstructured terrain navigation, where unmodeled and unmeasured terrain characteristics can significantly impact the robot's motion. Extensive experiments in both simulation and real world reveal significant improvements in data efficiency, policy performance, and the emergence of safer, adaptive navigation strategies. 

---
# DocMEdit: Towards Document-Level Model Editing 

**Authors**: Li Zeng, Zeming Liu, Chong Feng, Heyan Huang, Yuhang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.19572)  

**Abstract**: Model editing aims to correct errors and outdated knowledge in the Large language models (LLMs) with minimal cost. Prior research has proposed a variety of datasets to assess the effectiveness of these model editing methods. However, most existing datasets only require models to output short phrases or sentences, overlooks the widespread existence of document-level tasks in the real world, raising doubts about their practical usability. Aimed at addressing this limitation and promoting the application of model editing in real-world scenarios, we propose the task of document-level model editing. To tackle such challenges and enhance model capabilities in practical settings, we introduce \benchmarkname, a dataset focused on document-level model editing, characterized by document-level inputs and outputs, extrapolative, and multiple facts within a single edit. We propose a series of evaluation metrics and experiments. The results show that the difficulties in document-level model editing pose challenges for existing model editing methods. 

---
# How Syntax Specialization Emerges in Language Models 

**Authors**: Xufeng Duan, Zhaoqian Yao, Yunhao Zhang, Shaonan Wang, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.19548)  

**Abstract**: Large language models (LLMs) have been found to develop surprising internal specializations: Individual neurons, attention heads, and circuits become selectively sensitive to syntactic structure, reflecting patterns observed in the human brain. While this specialization is well-documented, how it emerges during training and what influences its development remains largely unknown.
In this work, we tap into the black box of specialization by tracking its formation over time. By quantifying internal syntactic consistency across minimal pairs from various syntactic phenomena, we identify a clear developmental trajectory: Syntactic sensitivity emerges gradually, concentrates in specific layers, and exhibits a 'critical period' of rapid internal specialization. This process is consistent across architectures and initialization parameters (e.g., random seeds), and is influenced by model scale and training data. We therefore reveal not only where syntax arises in LLMs but also how some models internalize it during training. To support future research, we will release the code, models, and training checkpoints upon acceptance. 

---
# STRAP: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization 

**Authors**: Haoyu Zhang, Wentao Zhang, Hao Miao, Xinke Jiang, Yuchen Fang, Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19547)  

**Abstract**: Spatio-Temporal Graph Neural Networks (STGNNs) have emerged as a powerful tool for modeling dynamic graph-structured data across diverse domains. However, they often fail to generalize in Spatio-Temporal Out-of-Distribution (STOOD) scenarios, where both temporal dynamics and spatial structures evolve beyond the training distribution. To address this problem, we propose an innovative Spatio-Temporal Retrieval-Augmented Pattern Learning framework,STRAP, which enhances model generalization by integrating retrieval-augmented learning into the STGNN continue learning pipeline. The core of STRAP is a compact and expressive pattern library that stores representative spatio-temporal patterns enriched with historical, structural, and semantic information, which is obtained and optimized during the training phase. During inference, STRAP retrieves relevant patterns from this library based on similarity to the current input and injects them into the model via a plug-and-play prompting mechanism. This not only strengthens spatio-temporal representations but also mitigates catastrophic forgetting. Moreover, STRAP introduces a knowledge-balancing objective to harmonize new information with retrieved knowledge. Extensive experiments across multiple real-world streaming graph datasets show that STRAP consistently outperforms state-of-the-art STGNN baselines on STOOD tasks, demonstrating its robustness, adaptability, and strong generalization capability without task-specific fine-tuning. 

---
# DoctorRAG: Medical RAG Fusing Knowledge with Patient Analogy through Textual Gradients 

**Authors**: Yuxing Lu, Gecheng Fu, Wei Wu, Xukai Zhao, Sin Yee Goi, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19538)  

**Abstract**: Existing medical RAG systems mainly leverage knowledge from medical knowledge bases, neglecting the crucial role of experiential knowledge derived from similar patient cases -- a key component of human clinical reasoning. To bridge this gap, we propose DoctorRAG, a RAG framework that emulates doctor-like reasoning by integrating both explicit clinical knowledge and implicit case-based experience. DoctorRAG enhances retrieval precision by first allocating conceptual tags for queries and knowledge sources, together with a hybrid retrieval mechanism from both relevant knowledge and patient. In addition, a Med-TextGrad module using multi-agent textual gradients is integrated to ensure that the final output adheres to the retrieved knowledge and patient query. Comprehensive experiments on multilingual, multitask datasets demonstrate that DoctorRAG significantly outperforms strong baseline RAG models and gains improvements from iterative refinements. Our approach generates more accurate, relevant, and comprehensive responses, taking a step towards more doctor-like medical reasoning systems. 

---
# FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models 

**Authors**: Jintao Tong, Wenwei Jin, Pengda Qin, Anqi Li, Yixiong Zou, Yuhong Li, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19536)  

**Abstract**: Large vision-language models (LVLMs) excel at multimodal understanding but suffer from high computational costs due to redundant vision tokens. Existing pruning methods typically rely on single-layer attention scores to rank and prune redundant visual tokens to solve this inefficiency. However, as the interaction between tokens and layers is complicated, this raises a basic question: Is such a simple single-layer criterion sufficient to identify redundancy? To answer this question, we rethink the emergence of redundant visual tokens from a fundamental perspective: information flow, which models the interaction between tokens and layers by capturing how information moves between tokens across layers. We find (1) the CLS token acts as an information relay, which can simplify the complicated flow analysis; (2) the redundancy emerges progressively and dynamically via layer-wise attention concentration; and (3) relying solely on attention scores from single layers can lead to contradictory redundancy identification. Based on this, we propose FlowCut, an information-flow-aware pruning framework, mitigating the insufficiency of the current criterion for identifying redundant tokens and better aligning with the model's inherent behaviors. Extensive experiments show that FlowCut achieves superior results, outperforming SoTA by 1.6% on LLaVA-1.5-7B with 88.9% token reduction, and by 4.3% on LLaVA-NeXT-7B with 94.4% reduction, delivering 3.2x speed-up in the prefilling stage. Our code is available at this https URL 

---
# Training-Free Multi-Step Audio Source Separation 

**Authors**: Yongyi Zang, Jingyi Li, Qiuqiang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2505.19534)  

**Abstract**: Audio source separation aims to separate a mixture into target sources. Previous audio source separation systems usually conduct one-step inference, which does not fully explore the separation ability of models. In this work, we reveal that pretrained one-step audio source separation models can be leveraged for multi-step separation without additional training. We propose a simple yet effective inference method that iteratively applies separation by optimally blending the input mixture with the previous step's separation result. At each step, we determine the optimal blending ratio by maximizing a metric. We prove that our method always yield improvement over one-step inference, provide error bounds based on model smoothness and metric robustness, and provide theoretical analysis connecting our method to denoising along linear interpolation paths between noise and clean distributions, a property we link to denoising diffusion bridge models. Our approach effectively delivers improved separation performance as a "free lunch" from existing models. Our empirical results demonstrate that our multi-step separation approach consistently outperforms one-step inference across both speech enhancement and music source separation tasks, and can achieve scaling performance similar to training a larger model, using more data, or in some cases employing a multi-step training objective. These improvements appear not only on the optimization metric during multi-step inference, but also extend to nearly all non-optimized metrics (with one exception). We also discuss limitations of our approach and directions for future research. 

---
# Minimalist Softmax Attention Provably Learns Constrained Boolean Functions 

**Authors**: Jerry Yao-Chieh Hu, Xiwen Zhang, Maojiang Su, Zhao Song, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19531)  

**Abstract**: We study the computational limits of learning $k$-bit Boolean functions (specifically, $\mathrm{AND}$, $\mathrm{OR}$, and their noisy variants), using a minimalist single-head softmax-attention mechanism, where $k=\Theta(d)$ relevant bits are selected from $d$ inputs. We show that these simple $\mathrm{AND}$ and $\mathrm{OR}$ functions are unsolvable with a single-head softmax-attention mechanism alone. However, with teacher forcing, the same minimalist attention is capable of solving them. These findings offer two key insights: Architecturally, solving these Boolean tasks requires only minimalist attention, without deep Transformer blocks or FFNs. Methodologically, one gradient descent update with supervision suffices and replaces the multi-step Chain-of-Thought (CoT) reasoning scheme of [Kim and Suzuki, ICLR 2025] for solving Boolean problems. Together, the bounds expose a fundamental gap between what this minimal architecture achieves under ideal supervision and what is provably impossible under standard training. 

---
# AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection 

**Authors**: Yejin Lee, Joonghyuk Hahn, Hyeseon Ahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.19528)  

**Abstract**: Implicit hate speech detection is challenging due to its subtlety and reliance on contextual interpretation rather than explicit offensive words. Current approaches rely on contrastive learning, which are shown to be effective on distinguishing hate and non-hate sentences. Humans, however, detect implicit hate speech by first identifying specific targets within the text and subsequently interpreting how these target relate to their surrounding context. Motivated by this reasoning process, we propose AmpleHate, a novel approach designed to mirror human inference for implicit hate detection. AmpleHate identifies explicit target using a pretrained Named Entity Recognition model and capture implicit target information via [CLS] tokens. It computes attention-based relationships between explicit, implicit targets and sentence context and then, directly injects these relational vectors into the final sentence representation. This amplifies the critical signals of target-context relations for determining implicit hate. Experiments demonstrate that AmpleHate achieves state-of-the-art performance, outperforming contrastive learning baselines by an average of 82.14% and achieve faster convergence. Qualitative analyses further reveal that attention patterns produced by AmpleHate closely align with human judgement, underscoring its interpretability and robustness. 

---
# Navigating loss manifolds via rigid body dynamics: A promising avenue for robustness and generalisation 

**Authors**: Mohammed D. Belgoumri, Mohamed Reda Bouadjenek, Hakim Hacid, Imran Razzak, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2505.19527)  

**Abstract**: Training large neural networks through gradient-based optimization requires navigating high-dimensional loss landscapes, which often exhibit pathological geometry, leading to undesirable training dynamics. In particular, poor generalization frequently results from convergence to sharp minima that are highly sensitive to input perturbations, causing the model to overfit the training data while failing to generalize to unseen examples. Furthermore, these optimization procedures typically display strong dependence on the fine structure of the loss landscape, leading to unstable training dynamics, due to the fractal-like nature of the loss surface. In this work, we propose an alternative optimizer that simultaneously reduces this dependence, and avoids sharp minima, thereby improving generalization. This is achieved by simulating the motion of the center of a ball rolling on the loss landscape. The degree to which our optimizer departs from the standard gradient descent is controlled by a hyperparameter, representing the radius of the ball. Changing this hyperparameter allows for probing the loss landscape at different scales, making it a valuable tool for understanding its geometry. 

---
# Rethinking Gating Mechanism in Sparse MoE: Handling Arbitrary Modality Inputs with Confidence-Guided Gate 

**Authors**: Liangwei Nathan Zheng, Wei Emma Zhang, Mingyu Guo, Miao Xu, Olaf Maennel, Weitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19525)  

**Abstract**: Effectively managing missing modalities is a fundamental challenge in real-world multimodal learning scenarios, where data incompleteness often results from systematic collection errors or sensor failures. Sparse Mixture-of-Experts (SMoE) architectures have the potential to naturally handle multimodal data, with individual experts specializing in different modalities. However, existing SMoE approach often lacks proper ability to handle missing modality, leading to performance degradation and poor generalization in real-world applications. We propose Conf-SMoE to introduce a two-stage imputation module to handle the missing modality problem for the SMoE architecture and reveal the insight of expert collapse from theoretical analysis with strong empirical evidence. Inspired by our theoretical analysis, Conf-SMoE propose a novel expert gating mechanism by detaching the softmax routing score to task confidence score w.r.t ground truth. This naturally relieves expert collapse without introducing additional load balance loss function. We show that the insights of expert collapse aligns with other gating mechanism such as Gaussian and Laplacian gate. We also evaluate the proposed method on four different real world dataset with three different experiment settings to conduct comprehensive the analysis of Conf-SMoE on modality fusion and resistance to missing modality. 

---
# SIPDO: Closed-Loop Prompt Optimization via Synthetic Data Feedback 

**Authors**: Yaoning Yu, Ye Yu, Kai Wei, Haojing Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19514)  

**Abstract**: Prompt quality plays a critical role in the performance of large language models (LLMs), motivating a growing body of work on prompt optimization. Most existing methods optimize prompts over a fixed dataset, assuming static input distributions and offering limited support for iterative improvement. We introduce SIPDO (Self-Improving Prompts through Data-Augmented Optimization), a closed-loop framework for prompt learning that integrates synthetic data generation into the optimization process. SIPDO couples a synthetic data generator with a prompt optimizer, where the generator produces new examples that reveal current prompt weaknesses and the optimizer incrementally refines the prompt in response. This feedback-driven loop enables systematic improvement of prompt performance without assuming access to external supervision or new tasks. Experiments across question answering and reasoning benchmarks show that SIPDO outperforms standard prompt tuning methods, highlighting the value of integrating data synthesis into prompt learning workflows. 

---
# Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models 

**Authors**: Yifan Jia, Kailin Jiang, Yuyang Liang, Qihan Ren, Yi Xin, Rui Yang, Fenze Feng, Mingcai Chen, Hengyang Lu, Haozhe Wang, Xiaoye Qu, Dongrui Liu, Lizhen Cui, Yuntao Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.19509)  

**Abstract**: Large Multimodal Models(LMMs) face notable challenges when encountering multimodal knowledge conflicts, particularly under retrieval-augmented generation(RAG) frameworks where the contextual information from external sources may contradict the model's internal parametric knowledge, leading to unreliable outputs. However, existing benchmarks fail to reflect such realistic conflict scenarios. Most focus solely on intra-memory conflicts, while context-memory and inter-context conflicts remain largely investigated. Furthermore, commonly used factual knowledge-based evaluations are often overlooked, and existing datasets lack a thorough investigation into conflict detection capabilities. To bridge this gap, we propose MMKC-Bench, a benchmark designed to evaluate factual knowledge conflicts in both context-memory and inter-context scenarios. MMKC-Bench encompasses three types of multimodal knowledge conflicts and includes 1,573 knowledge instances and 3,381 images across 23 broad types, collected through automated pipelines with human verification. We evaluate three representative series of LMMs on both model behavior analysis and conflict detection tasks. Our findings show that while current LMMs are capable of recognizing knowledge conflicts, they tend to favor internal parametric knowledge over external evidence. We hope MMKC-Bench will foster further research in multimodal knowledge conflict and enhance the development of multimodal RAG systems. The source code is available at this https URL. 

---
# Hierarchical Tree Search-based User Lifelong Behavior Modeling on Large Language Model 

**Authors**: Yu Xia, Rui Zhong, Hao Gu, Wei Yang, Chi Lu, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2505.19505)  

**Abstract**: Large Language Models (LLMs) have garnered significant attention in Recommendation Systems (RS) due to their extensive world knowledge and robust reasoning capabilities. However, a critical challenge lies in enabling LLMs to effectively comprehend and extract insights from massive user behaviors. Current approaches that directly leverage LLMs for user interest learning face limitations in handling long sequential behaviors, effectively extracting interest, and applying interest in practical scenarios. To address these issues, we propose a Hierarchical Tree Search-based User Lifelong Behavior Modeling framework (HiT-LBM). HiT-LBM integrates Chunked User Behavior Extraction (CUBE) and Hierarchical Tree Search for Interest (HTS) to capture diverse interests and interest evolution of user. CUBE divides user lifelong behaviors into multiple chunks and learns the interest and interest evolution within each chunk in a cascading manner. HTS generates candidate interests through hierarchical expansion and searches for the optimal interest with process rating model to ensure information gain for each behavior chunk. Additionally, we design Temporal-Ware Interest Fusion (TIF) to integrate interests from multiple behavior chunks, constructing a comprehensive representation of user lifelong interests. The representation can be embedded into any recommendation model to enhance performance. Extensive experiments demonstrate the effectiveness of our approach, showing that it surpasses state-of-the-art methods. 

---
# DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation 

**Authors**: Pingzhi Li, Zhen Tan, Huaizhi Qu, Huan Liu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19504)  

**Abstract**: Large Language Models (LLMs) represent substantial intellectual and economic investments, yet their effectiveness can inadvertently facilitate model imitation via knowledge distillation (KD).In practical scenarios, competitors can distill proprietary LLM capabilities by simply observing publicly accessible outputs, akin to reverse-engineering a complex performance by observation alone. Existing protective methods like watermarking only identify imitation post-hoc, while other defenses assume the student model mimics the teacher's internal logits, rendering them ineffective against distillation purely from observed output text. This paper confronts the challenge of actively protecting LLMs within the realistic constraints of API-based access. We introduce an effective and efficient Defensive Output Generation (DOGe) strategy that subtly modifies the output behavior of an LLM. Its outputs remain accurate and useful for legitimate users, yet are designed to be misleading for distillation, significantly undermining imitation attempts. We achieve this by fine-tuning only the final linear layer of the teacher LLM with an adversarial loss. This targeted training approach anticipates and disrupts distillation attempts during inference time. Our experiments show that, while preserving or even improving the original performance of the teacher model, student models distilled from the defensively generated teacher outputs demonstrate catastrophically reduced performance, demonstrating our method's effectiveness as a practical safeguard against KD-based model imitation. 

---
# CODE-DITING: A Reasoning-Based Metric for Functional Alignment in Code Evaluation 

**Authors**: Guang Yang, Yu Zhou, Xiang Chen, Wei Zheng, Xing Hu, Xin Zhou, David Lo, Taolue Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19502)  

**Abstract**: Trustworthy evaluation methods for code snippets play a crucial role in neural code generation. Traditional methods, which either rely on reference solutions or require executable test cases, have inherent limitation in flexibility and scalability. The recent LLM-as-Judge methodology offers a promising alternative by directly evaluating functional consistency between the problem description and the generated code. To systematically understand the landscape of these LLM-as-Judge methods, we conduct a comprehensive empirical study across three diverse datasets. Our investigation reveals the pros and cons of two categories of LLM-as-Judge methods: the methods based on general foundation models can achieve good performance but require complex prompts and lack explainability, while the methods based on reasoning foundation models provide better explainability with simpler prompts but demand substantial computational resources due to their large parameter sizes. To address these limitations, we propose CODE-DITING, a novel code evaluation method that balances accuracy, efficiency and explainability. We develop a data distillation framework that effectively transfers reasoning capabilities from DeepSeek-R1671B to our CODE-DITING 1.5B and 7B models, significantly enhancing evaluation explainability and reducing the computational cost. With the majority vote strategy in the inference process, CODE-DITING 1.5B outperforms all models with the same magnitude of parameters and achieves performance which would normally exhibit in a model with 5 times of parameter scale. CODE-DITING 7B surpasses GPT-4o and DeepSeek-V3 671B, even though it only uses 1% of the parameter volume of these large models. Further experiments show that CODEDITING is robust to preference leakage and can serve as a promising alternative for code evaluation. 

---
# Enhancing Visual Reliance in Text Generation: A Bayesian Perspective on Mitigating Hallucination in Large Vision-Language Models 

**Authors**: Nanxing Hu, Xiaoyue Duan, Jinchao Zhang, Guoliang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19498)  

**Abstract**: Large Vision-Language Models (LVLMs) usually generate texts which satisfy context coherence but don't match the visual input. Such a hallucination issue hinders LVLMs' applicability in the real world. The key to solving hallucination in LVLM is to make the text generation rely more on the visual content. Most previous works choose to enhance/adjust the features/output of a specific modality (i.e., visual or textual) to alleviate hallucinations in LVLM, which do not explicitly or systematically enhance the visual reliance. In this paper, we comprehensively investigate the factors which may degenerate the visual reliance in text generation of LVLM from a Bayesian perspective. Based on our observations, we propose to mitigate hallucination in LVLM from three aspects. Firstly, we observe that not all visual tokens are informative in generating meaningful texts. We propose to evaluate and remove redundant visual tokens to avoid their disturbance. Secondly, LVLM may encode inappropriate prior information, making it lean toward generating unexpected words. We propose a simple yet effective way to rectify the prior from a Bayesian perspective. Thirdly, we observe that starting from certain steps, the posterior of next-token prediction conditioned on visual tokens may collapse to a prior distribution which does not depend on any informative visual tokens at all. Thus, we propose to stop further text generation to avoid hallucination. Extensive experiments on three benchmarks including POPE, CHAIR, and MME demonstrate that our method can consistently mitigate the hallucination issue of LVLM and performs favorably against previous state-of-the-arts. 

---
# Understanding Transformer from the Perspective of Associative Memory 

**Authors**: Shu Zhong, Mingyu Xu, Tenglong Ao, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19488)  

**Abstract**: In this paper, we share our reflections and insights on understanding Transformer architectures through the lens of associative memory--a classic psychological concept inspired by human cognition. We start with the basics of associative memory (think simple linear attention) and then dive into two dimensions:
Memory Capacity: How much can a Transformer really remember, and how well? We introduce retrieval SNR to measure this and use a kernel perspective to mathematically reveal why Softmax Attention is so effective. We also show how FFNs can be seen as a type of associative memory, leading to insights on their design and potential improvements.
Memory Update: How do these memories learn and evolve? We present a unified framework for understanding how different Transformer variants (like DeltaNet and Softmax Attention) update their "knowledge base". This leads us to tackle two provocative questions: 1. Are Transformers fundamentally limited in what they can express, and can we break these barriers? 2. If a Transformer had infinite context, would it become infinitely intelligent?
We want to demystify Transformer architecture, offering a clearer understanding of existing designs. This exploration aims to provide fresh insights and spark new avenues for Transformer innovation. 

---
# Win Fast or Lose Slow: Balancing Speed and Accuracy in Latency-Sensitive Decisions of LLMs 

**Authors**: Hao Kang, Qingru Zhang, Han Cai, Weiyuan Xu, Tushar Krishna, Yilun Du, Tsachy Weissman  

**Link**: [PDF](https://arxiv.org/pdf/2505.19481)  

**Abstract**: Large language models (LLMs) have shown remarkable performance across diverse reasoning and generation tasks, and are increasingly deployed as agents in dynamic environments such as code generation and recommendation systems. However, many real-world applications, such as high-frequency trading and real-time competitive gaming, require decisions under strict latency constraints, where faster responses directly translate into higher rewards. Despite the importance of this latency quality trade off, it remains underexplored in the context of LLM based agents. In this work, we present the first systematic study of this trade off in real time decision making tasks. To support our investigation, we introduce two new benchmarks: HFTBench, a high frequency trading simulation, and StreetFighter, a competitive gaming platform. Our analysis reveals that optimal latency quality balance varies by task, and that sacrificing quality for lower latency can significantly enhance downstream performance. To address this, we propose FPX, an adaptive framework that dynamically selects model size and quantization level based on real time demands. Our method achieves the best performance on both benchmarks, improving win rate by up to 80% in Street Fighter and boosting daily yield by up to 26.52% in trading, underscoring the need for latency aware evaluation and deployment strategies for LLM based agents. These results demonstrate the critical importance of latency aware evaluation and deployment strategies for real world LLM based agents. Our benchmarks are available at Latency Sensitive Benchmarks. 

---
# Diversity-Driven Generative Dataset Distillation Based on Diffusion Model with Self-Adaptive Memory 

**Authors**: Mingzhuo Li, Guang Li, Jiafeng Mao, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2505.19469)  

**Abstract**: Dataset distillation enables the training of deep neural networks with comparable performance in significantly reduced time by compressing large datasets into small and representative ones. Although the introduction of generative models has made great achievements in this field, the distributions of their distilled datasets are not diverse enough to represent the original ones, leading to a decrease in downstream validation accuracy. In this paper, we present a diversity-driven generative dataset distillation method based on a diffusion model to solve this problem. We introduce self-adaptive memory to align the distribution between distilled and real datasets, assessing the representativeness. The degree of alignment leads the diffusion model to generate more diverse datasets during the distillation process. Extensive experiments show that our method outperforms existing state-of-the-art methods in most situations, proving its ability to tackle dataset distillation tasks. 

---
# Residual Cross-Attention Transformer-Based Multi-User CSI Feedback with Deep Joint Source-Channel Coding 

**Authors**: Hengwei Zhang, Minghui Wu, Li Qiao, Ling Liu, Ziqi Han, Zhen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19465)  

**Abstract**: This letter proposes a deep-learning (DL)-based multi-user channel state information (CSI) feedback framework for massive multiple-input multiple-output systems, where the deep joint source-channel coding (DJSCC) is utilized to improve the CSI reconstruction accuracy. Specifically, we design a multi-user joint CSI feedback framework, whereby the CSI correlation of nearby users is utilized to reduce the feedback overhead. Under the framework, we propose a new residual cross-attention transformer architecture, which is deployed at the base station to further improve the CSI feedback performance. Moreover, to tackle the "cliff-effect" of conventional bit-level CSI feedback approaches, we integrated DJSCC into the multi-user CSI feedback, together with utilizing a two-stage training scheme to adapt to varying uplink noise levels. Experimental results demonstrate the superiority of our methods in CSI feedback performance, with low network complexity and better scalability. 

---
# Your Classifier Can Do More: Towards Bridging the Gaps in Classification, Robustness, and Generation 

**Authors**: Kaichao Jiang, He Wang, Xiaoshuai Hao, Xiulong Yang, Ajian Liu, Qi Chu, Yunfeng Diao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19459)  

**Abstract**: Joint Energy-based Models (JEMs), a class of hybrid generative-discriminative models, are well known for their ability to achieve both high classification accuracy and generative capability within a single model. However, their robustness still lags significantly behind the classifiers based adversarial training (AT). Conversely, while AT is currently the most effective approach to improving the classifier's robustness, it typically sacrifices accuracy on clean data and lacks generative capability. The triple trade-off between classification accuracy, generative capability and robustness, raises a natural question: Can a single model simultaneously achieve high classification accuracy, adversarial robustness, and generative performance? -- a goal that has been rarely explored. To address this question, we systematically analyze the energy distribution differences of clean, adversarial, and generated samples across various JEM variants and adversarially trained models. We observe that AT tends to reduce the energy gap between clean and adversarial samples, while JEMs reduce the gap between clean and synthetic ones. This observation suggests a key insight: if the energy distributions of all three data types can be aligned, we might unify the strengths of AT and JEMs, resolving their inherent trade-offs. Building on this idea, we propose Energy-based Joint Distribution Adversarial Training (EB-JDAT), to jointly model the clean data distribution, the adversarial distribution, and the classifier by maximizing their joint probability. EB-JDAT is a general and flexible optimization method, compatible with various JEM variants. Extensive experimental results demonstrate that EB-JDAT not only maintains near original accuracy and generative capability of JEMs, but also significantly enhances robustness, even surpassing state-of-the-art ATs. 

---
# MM-Prompt: Cross-Modal Prompt Tuning for Continual Visual Question Answering 

**Authors**: Xu Li, Fan Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19455)  

**Abstract**: Continual Visual Question Answering (CVQA) based on pre-trained models(PTMs) has achieved promising progress by leveraging prompt tuning to enable continual multi-modal learning. However, most existing methods adopt cross-modal prompt isolation, constructing visual and textual prompts separately, which exacerbates modality imbalance and leads to degraded performance over time. To tackle this issue, we propose MM-Prompt, a novel framework incorporating cross-modal prompt query and cross-modal prompt recovery. The former enables balanced prompt selection by incorporating cross-modal signals during query formation, while the latter promotes joint prompt reconstruction through iterative cross-modal interactions, guided by an alignment loss to prevent representational drift. Extensive experiments show that MM-Prompt surpasses prior approaches in accuracy and knowledge retention, while maintaining balanced modality engagement throughout continual learning. 

---
# Vibe Coding vs. Agentic Coding: Fundamentals and Practical Implications of Agentic AI 

**Authors**: Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2505.19443)  

**Abstract**: This review presents a comprehensive analysis of two emerging paradigms in AI-assisted software development: vibe coding and agentic coding. While both leverage large language models (LLMs), they differ fundamentally in autonomy, architectural design, and the role of the developer. Vibe coding emphasizes intuitive, human-in-the-loop interaction through prompt-based, conversational workflows that support ideation, experimentation, and creative exploration. In contrast, agentic coding enables autonomous software development through goal-driven agents capable of planning, executing, testing, and iterating tasks with minimal human intervention. We propose a detailed taxonomy spanning conceptual foundations, execution models, feedback loops, safety mechanisms, debugging strategies, and real-world tool ecosystems. Through comparative workflow analysis and 20 detailed use cases, we illustrate how vibe systems thrive in early-stage prototyping and education, while agentic systems excel in enterprise-grade automation, codebase refactoring, and CI/CD integration. We further examine emerging trends in hybrid architectures, where natural language interfaces are coupled with autonomous execution pipelines. Finally, we articulate a future roadmap for agentic AI, outlining the infrastructure needed for trustworthy, explainable, and collaborative systems. Our findings suggest that successful AI software engineering will rely not on choosing one paradigm, but on harmonizing their strengths within a unified, human-centered development lifecycle. 

---
# Fairness Practices in Industry: A Case Study in Machine Learning Teams Building Recommender Systems 

**Authors**: Jing Nathan Yan, Junxiong Wang, Jeffrey M. Rzeszotarski, Allison Koenecke  

**Link**: [PDF](https://arxiv.org/pdf/2505.19441)  

**Abstract**: The rapid proliferation of recommender systems necessitates robust fairness practices to address inherent biases. Assessing fairness, though, is challenging due to constantly evolving metrics and best practices. This paper analyzes how industry practitioners perceive and incorporate these changing fairness standards in their workflows. Through semi-structured interviews with 11 practitioners from technical teams across a range of large technology companies, we investigate industry implementations of fairness in recommendation system products. We focus on current debiasing practices, applied metrics, collaborative strategies, and integrating academic research into practice. Findings show a preference for multi-dimensional debiasing over traditional demographic methods, and a reliance on intuitive rather than academic metrics. This study also highlights the difficulties in balancing fairness with both the practitioner's individual (bottom-up) roles and organizational (top-down) workplace constraints, including the interplay with legal and compliance experts. Finally, we offer actionable recommendations for the recommender system community and algorithmic fairness practitioners, underlining the need to refine fairness practices continually. 

---
# CSTrack: Enhancing RGB-X Tracking via Compact Spatiotemporal Features 

**Authors**: X. Feng, D. Zhang, S. Hu, X. Li, M. Wu, J. Zhang, X. Chen, K. Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19434)  

**Abstract**: Effectively modeling and utilizing spatiotemporal features from RGB and other modalities (\eg, depth, thermal, and event data, denoted as X) is the core of RGB-X tracker design. Existing methods often employ two parallel branches to separately process the RGB and X input streams, requiring the model to simultaneously handle two dispersed feature spaces, which complicates both the model structure and computation process. More critically, intra-modality spatial modeling within each dispersed space incurs substantial computational overhead, limiting resources for inter-modality spatial modeling and temporal modeling. To address this, we propose a novel tracker, CSTrack, which focuses on modeling Compact Spatiotemporal features to achieve simple yet effective tracking. Specifically, we first introduce an innovative Spatial Compact Module that integrates the RGB-X dual input streams into a compact spatial feature, enabling thorough intra- and inter-modality spatial modeling. Additionally, we design an efficient Temporal Compact Module that compactly represents temporal features by constructing the refined target distribution heatmap. Extensive experiments validate the effectiveness of our compact spatiotemporal modeling method, with CSTrack achieving new SOTA results on mainstream RGB-X benchmarks. The code and models will be released at: this https URL. 

---
# Deriving Strategic Market Insights with Large Language Models: A Benchmark for Forward Counterfactual Generation 

**Authors**: Keane Ong, Rui Mao, Deeksha Varshney, Paul Pu Liang, Erik Cambria, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2505.19430)  

**Abstract**: Counterfactual reasoning typically involves considering alternatives to actual events. While often applied to understand past events, a distinct form-forward counterfactual reasoning-focuses on anticipating plausible future developments. This type of reasoning is invaluable in dynamic financial markets, where anticipating market developments can powerfully unveil potential risks and opportunities for stakeholders, guiding their decision-making. However, performing this at scale is challenging due to the cognitive demands involved, underscoring the need for automated solutions. Large Language Models (LLMs) offer promise, but remain unexplored for this application. To address this gap, we introduce a novel benchmark, Fin-Force-FINancial FORward Counterfactual Evaluation. By curating financial news headlines and providing structured evaluation, Fin-Force supports LLM based forward counterfactual generation. This paves the way for scalable and automated solutions for exploring and anticipating future market developments, thereby providing structured insights for decision-making. Through experiments on Fin-Force, we evaluate state-of-the-art LLMs and counterfactual generation methods, analyzing their limitations and proposing insights for future research. 

---
# WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference 

**Authors**: Sihan Chen, Dan Zhao, Jongwoo Ko, Colby Banbury, Huiping Zhuang, Luming Liang, Tianyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19427)  

**Abstract**: The growing computational demands of large language models (LLMs) make efficient inference and activation strategies increasingly critical. While recent approaches, such as Mixture-of-Experts (MoE), leverage selective activation but require specialized training, training-free sparse activation methods offer broader applicability and superior resource efficiency through their plug-and-play design. However, many existing methods rely solely on hidden state magnitudes to determine activation, resulting in high approximation errors and suboptimal inference accuracy. To address these limitations, we propose WINA (Weight Informed Neuron Activation), a novel, simple, and training-free sparse activation framework that jointly considers hidden state magnitudes and the column-wise $\ell_2$-norms of weight matrices. We show that this leads to a sparsification strategy that obtains optimal approximation error bounds with theoretical guarantees tighter than existing techniques. Empirically, WINA also outperforms state-of-the-art methods (e.g., TEAL) by up to $2.94\%$ in average performance at the same sparsity levels, across a diverse set of LLM architectures and datasets. These results position WINA as a new performance frontier for training-free sparse activation in LLM inference, advancing training-free sparse activation methods and setting a robust baseline for efficient inference. The source code is available at this https URL. 

---
# The Role of Diversity in In-Context Learning for Large Language Models 

**Authors**: Wenyang Xiao, Haoyu Zhao, Lingxiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19426)  

**Abstract**: In-context learning (ICL) is a crucial capability of current large language models (LLMs), where the selection of examples plays a key role in performance. While most existing approaches focus on selecting the most similar examples to the query, the impact of diversity in example selection remains underexplored. We systematically investigate the role of diversity in in-context example selection through experiments across a range of tasks, from sentiment classification to more challenging math and code problems. Experiments on Llama-3.1, Gemma-2, and Mistral-v0.3 families of models show that diversity-aware selection methods improve performance, particularly on complex tasks like math and code, and enhance robustness to out-of-distribution queries. To support these findings, we introduce a theoretical framework that explains the benefits of incorporating diversity in in-context example selection. 

---
# Surrogate-Assisted Evolutionary Reinforcement Learning Based on Autoencoder and Hyperbolic Neural Network 

**Authors**: Bingdong Li, Mei Jiang, Hong Qian, Peng Yang, Wenjing Hong, Hong Qian, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19423)  

**Abstract**: Evolutionary Reinforcement Learning (ERL), training the Reinforcement Learning (RL) policies with Evolutionary Algorithms (EAs), have demonstrated enhanced exploration capabilities and greater robustness than using traditional policy gradient. However, ERL suffers from the high computational costs and low search efficiency, as EAs require evaluating numerous candidate policies with expensive simulations, many of which are ineffective and do not contribute meaningfully to the training. One intuitive way to reduce the ineffective evaluations is to adopt the surrogates. Unfortunately, existing ERL policies are often modeled as deep neural networks (DNNs) and thus naturally represented as high-dimensional vectors containing millions of weights, which makes the building of effective surrogates for ERL policies extremely challenging. This paper proposes a novel surrogate-assisted ERL that integrates Autoencoders (AE) and Hyperbolic Neural Networks (HNN). Specifically, AE compresses high-dimensional policies into low-dimensional representations while extracting key features as the inputs for the surrogate. HNN, functioning as a classification-based surrogate model, can learn complex nonlinear relationships from sampled data and enable more accurate pre-selection of the sampled policies without real evaluations. The experiments on 10 Atari and 4 Mujoco games have verified that the proposed method outperforms previous approaches significantly. The search trajectories guided by AE and HNN are also visually demonstrated to be more effective, in terms of both exploration and convergence. This paper not only presents the first learnable policy embedding and surrogate-modeling modules for high-dimensional ERL policies, but also empirically reveals when and why they can be successful. 

---
# It's Not Just Labeling" -- A Research on LLM Generated Feedback Interpretability and Image Labeling Sketch Features 

**Authors**: Baichuan Li, Larry Powell, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2505.19419)  

**Abstract**: The quality of training data is critical to the performance of machine learning applications in domains like transportation, healthcare, and robotics. Accurate image labeling, however, often relies on time-consuming, expert-driven methods with limited feedback. This research introduces a sketch-based annotation approach supported by large language models (LLMs) to reduce technical barriers and enhance accessibility. Using a synthetic dataset, we examine how sketch recognition features relate to LLM feedback metrics, aiming to improve the reliability and interpretability of LLM-assisted labeling. We also explore how prompting strategies and sketch variations influence feedback quality. Our main contribution is a sketch-based virtual assistant that simplifies annotation for non-experts and advances LLM-driven labeling tools in terms of scalability, accessibility, and explainability. 

---
# Exploring the Possibility of TypiClust for Low-Budget Federated Active Learning 

**Authors**: Yuta Ono, Hiroshi Nakamura, Hideki Takase  

**Link**: [PDF](https://arxiv.org/pdf/2505.19404)  

**Abstract**: Federated Active Learning (FAL) seeks to reduce the burden of annotation under the realistic constraints of federated learning by leveraging Active Learning (AL). As FAL settings make it more expensive to obtain ground truth labels, FAL strategies that work well in low-budget regimes, where the amount of annotation is very limited, are needed. In this work, we investigate the effectiveness of TypiClust, a successful low-budget AL strategy, in low-budget FAL settings. Our empirical results show that TypiClust works well even in low-budget FAL settings contrasted with relatively low performances of other methods, although these settings present additional challenges, such as data heterogeneity, compared to AL. In addition, we show that FAL settings cause distribution shifts in terms of typicality, but TypiClust is not very vulnerable to the shifts. We also analyze the sensitivity of TypiClust to feature extraction methods, and it suggests a way to perform FAL even in limited data situations. 

---
# VADER: A Human-Evaluated Benchmark for Vulnerability Assessment, Detection, Explanation, and Remediation 

**Authors**: Ethan TS. Liu, Austin Wang, Spencer Mateega, Carlos Georgescu, Danny Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19395)  

**Abstract**: Ensuring that large language models (LLMs) can effectively assess, detect, explain, and remediate software vulnerabilities is critical for building robust and secure software systems. We introduce VADER, a human-evaluated benchmark designed explicitly to assess LLM performance across four key vulnerability-handling dimensions: assessment, detection, explanation, and remediation. VADER comprises 174 real-world software vulnerabilities, each carefully curated from GitHub repositories and annotated by security experts. For each vulnerability case, models are tasked with identifying the flaw, classifying it using Common Weakness Enumeration (CWE), explaining its underlying cause, proposing a patch, and formulating a test plan. Using a one-shot prompting strategy, we benchmark six state-of-the-art LLMs (Claude 3.7 Sonnet, Gemini 2.5 Pro, GPT-4.1, GPT-4.5, Grok 3 Beta, and o3) on VADER, and human security experts evaluated each response according to a rigorous scoring rubric emphasizing remediation (quality of the code fix, 50%), explanation (20%), and classification and test plan (30%) according to a standardized rubric. Our results show that current state-of-the-art LLMs achieve only moderate success on VADER - OpenAI's o3 attained 54.7% accuracy overall, with others in the 49-54% range, indicating ample room for improvement. Notably, remediation quality is strongly correlated (Pearson r > 0.97) with accurate classification and test plans, suggesting that models that effectively categorize vulnerabilities also tend to fix them well. VADER's comprehensive dataset, detailed evaluation rubrics, scoring tools, and visualized results with confidence intervals are publicly released, providing the community with an interpretable, reproducible benchmark to advance vulnerability-aware LLMs. All code and data are available at: this https URL 

---
# Simple and Effective Baselines for Code Summarisation Evaluation 

**Authors**: Jade Robinson, Jonathan K. Kummerfeld  

**Link**: [PDF](https://arxiv.org/pdf/2505.19392)  

**Abstract**: Code documentation is useful, but writing it is time-consuming. Different techniques for generating code summaries have emerged, but comparing them is difficult because human evaluation is expensive and automatic metrics are unreliable. In this paper, we introduce a simple new baseline in which we ask an LLM to give an overall score to a summary. Unlike n-gram and embedding-based baselines, our approach is able to consider the code when giving a score. This allows us to also make a variant that does not consider the reference summary at all, which could be used for other tasks, e.g., to evaluate the quality of documentation in code bases. We find that our method is as good or better than prior metrics, though we recommend using it in conjunction with embedding-based methods to avoid the risk of LLM-specific bias. 

---
# Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals 

**Authors**: Nate Gillman, Charles Herrmann, Michael Freeman, Daksh Aggarwal, Evan Luo, Deqing Sun, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.19386)  

**Abstract**: Recent advances in video generation models have sparked interest in world models capable of simulating realistic environments. While navigation has been well-explored, physically meaningful interactions that mimic real-world forces remain largely understudied. In this work, we investigate using physical forces as a control signal for video generation and propose force prompts which enable users to interact with images through both localized point forces, such as poking a plant, and global wind force fields, such as wind blowing on fabric. We demonstrate that these force prompts can enable videos to respond realistically to physical control signals by leveraging the visual and motion prior in the original pretrained model, without using any 3D asset or physics simulator at inference. The primary challenge of force prompting is the difficulty in obtaining high quality paired force-video training data, both in the real world due to the difficulty of obtaining force signals, and in synthetic data due to limitations in the visual quality and domain diversity of physics simulators. Our key finding is that video generation models can generalize remarkably well when adapted to follow physical force conditioning from videos synthesized by Blender, even with limited demonstrations of few objects. Our method can generate videos which simulate forces across diverse geometries, settings, and materials. We also try to understand the source of this generalization and perform ablations that reveal two key elements: visual diversity and the use of specific text keywords during training. Our approach is trained on only around 15k training examples for a single day on four A100 GPUs, and outperforms existing methods on force adherence and physics realism, bringing world models closer to real-world physics interactions. We release all datasets, code, weights, and interactive video demos at our project page. 

---
# Advancing Limited-Angle CT Reconstruction Through Diffusion-Based Sinogram Completion 

**Authors**: Jiaqi Guo, Santiago Lopez-Tapia, Aggelos K. Katsaggelos  

**Link**: [PDF](https://arxiv.org/pdf/2505.19385)  

**Abstract**: Limited Angle Computed Tomography (LACT) often faces significant challenges due to missing angular information. Unlike previous methods that operate in the image domain, we propose a new method that focuses on sinogram inpainting. We leverage MR-SDEs, a variant of diffusion models that characterize the diffusion process with mean-reverting stochastic differential equations, to fill in missing angular data at the projection level. Furthermore, by combining distillation with constraining the output of the model using the pseudo-inverse of the inpainting matrix, the diffusion process is accelerated and done in a step, enabling efficient and accurate sinogram completion. A subsequent post-processing module back-projects the inpainted sinogram into the image domain and further refines the reconstruction, effectively suppressing artifacts while preserving critical structural details. Quantitative experimental results demonstrate that the proposed method achieves state-of-the-art performance in both perceptual and fidelity quality, offering a promising solution for LACT reconstruction in scientific and clinical applications. 

---
# SETransformer: A Hybrid Attention-Based Architecture for Robust Human Activity Recognition 

**Authors**: Yunbo Liu, Xukui Qin, Yifan Gao, Xiang Li, Chengwei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19369)  

**Abstract**: Human Activity Recognition (HAR) using wearable sensor data has become a central task in mobile computing, healthcare, and human-computer interaction. Despite the success of traditional deep learning models such as CNNs and RNNs, they often struggle to capture long-range temporal dependencies and contextual relevance across multiple sensor channels. To address these limitations, we propose SETransformer, a hybrid deep neural architecture that combines Transformer-based temporal modeling with channel-wise squeeze-and-excitation (SE) attention and a learnable temporal attention pooling mechanism. The model takes raw triaxial accelerometer data as input and leverages global self-attention to capture activity-specific motion dynamics over extended time windows, while adaptively emphasizing informative sensor channels and critical time steps.
We evaluate SETransformer on the WISDM dataset and demonstrate that it significantly outperforms conventional models including LSTM, GRU, BiLSTM, and CNN baselines. The proposed model achieves a validation accuracy of 84.68\% and a macro F1-score of 84.64\%, surpassing all baseline architectures by a notable margin. Our results show that SETransformer is a competitive and interpretable solution for real-world HAR tasks, with strong potential for deployment in mobile and ubiquitous sensing applications. 

---
# Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval 

**Authors**: Kidist Amde Mekonnen, Yosef Worku Alemneh, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2505.19356)  

**Abstract**: Neural retrieval methods using transformer-based pre-trained language models have advanced multilingual and cross-lingual retrieval. However, their effectiveness for low-resource, morphologically rich languages such as Amharic remains underexplored due to data scarcity and suboptimal tokenization. We address this gap by introducing Amharic-specific dense retrieval models based on pre-trained Amharic BERT and RoBERTa backbones. Our proposed RoBERTa-Base-Amharic-Embed model (110M parameters) achieves a 17.6% relative improvement in MRR@10 and a 9.86% gain in Recall@10 over the strongest multilingual baseline, Arctic Embed 2.0 (568M parameters). More compact variants, such as RoBERTa-Medium-Amharic-Embed (42M), remain competitive while being over 13x smaller. Additionally, we train a ColBERT-based late interaction retrieval model that achieves the highest MRR@10 score (0.843) among all evaluated models. We benchmark our proposed models against both sparse and dense retrieval baselines to systematically assess retrieval effectiveness in Amharic. Our analysis highlights key challenges in low-resource settings and underscores the importance of language-specific adaptation. To foster future research in low-resource IR, we publicly release our dataset, codebase, and trained models at this https URL. 

---
# PatentScore: Multi-dimensional Evaluation of LLM-Generated Patent Claims 

**Authors**: Yongmin Yoo, Qiongkai Xu, Longbing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19345)  

**Abstract**: Natural language generation (NLG) metrics play a central role in evaluating generated texts, but are not well suited for the structural and legal characteristics of patent documents. Large language models (LLMs) offer strong potential in automating patent generation, yet research on evaluating LLM-generated patents remains limited, especially in evaluating the generation quality of patent claims, which are central to defining the scope of protection. Effective claim evaluation requires addressing legal validity, technical accuracy, and structural compliance. To address this gap, we introduce PatentScore, a multi-dimensional evaluation framework for assessing LLM-generated patent claims. PatentScore incorporates: (1) hierarchical decomposition for claim analysis; (2) domain-specific validation patterns based on legal and technical standards; and (3) scoring across structural, semantic, and legal dimensions. Unlike general-purpose NLG metrics, PatentScore reflects patent-specific constraints and document structures, enabling evaluation beyond surface similarity. We evaluate 400 GPT-4o-mini generated Claim 1s and report a Pearson correlation of $r = 0.819$ with expert annotations, outperforming existing NLG metrics. Furthermore, we conduct additional evaluations using open models such as Claude-3.5-Haiku and Gemini-1.5-flash, all of which show strong correlations with expert judgments, confirming the robustness and generalizability of our framework. 

---
# Communication-Efficient Multi-Device Inference Acceleration for Transformer Models 

**Authors**: Xiao Liu, Lijun Zhang, Deepak Ganesan, Hui Guan  

**Link**: [PDF](https://arxiv.org/pdf/2505.19342)  

**Abstract**: Transformer models power many AI applications but suffer from high inference latency, limiting their use in real-time settings. Multi-device inference can reduce latency by parallelizing computation. Yet, existing methods require high inter-device bandwidth, making them impractical for bandwidth-constrained environments. We propose ASTRA, a communication-efficient framework that accelerates Transformer inference through a novel integration of sequence parallelism and a Mixed-Precision Attention mechanism designed to minimize inter-device communication. ASTRA compresses non-local token embeddings via vector quantization and preserves task accuracy through two optimizations, Noise-Augmented Quantization and Distributed Class Tokens. Experiments on ViT and GPT2 across vision and NLP tasks show that ASTRA achieves up to 2.64X speedups over single-device inference and up to 15.25X speedups over state-of-the-art multi-device inferences, while operating under bandwidths as low as 10 Mbps. ASTRA is open-sourced at this https URL. 

---
# Towards Humanoid Robot Autonomy: A Dynamic Architecture Integrating Continuous thought Machines (CTM) and Model Context Protocol (MCP) 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19339)  

**Abstract**: To address the gaps between the static pre-set "thinking-planning-action" of humanoid robots in unfamiliar scenarios and the highly programmed "call tool-return result" due to the lack of autonomous coding capabilities, this work designs a dynamic architecture connecting continuous thought machines (CTM) and model context protocol (MCP). It proposes a theoretical parallel solution through tick-slab and uses rank compression to achieve parameter suppression to provide a solution for achieving autonomous actions due to autonomous coding. The researcher used a simulation-based experiment using OpenAI's o4-mini-high as a tool to build the experimental environment, and introduced the extended SayCan dataset to conduct nine epochs of experiments. The experimental results show that the CTM-MCP architecture is feasible and effective through the data results of seven metrics: task success rate (TSR), execution success rate (ESR), average episode length (AEL), ROSCOE, REVEAL, proficiency self-assessment (PSA), task effectiveness (TE). In practice, it provides a reference experience for exploring the autonomous dynamic coding of humanoid robots based on continuous thinking to achieve human-like autonomous actions. 

---
# Prompting Decision Transformers for Zero-Shot Reach-Avoid Policies 

**Authors**: Kevin Li, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.19337)  

**Abstract**: Offline goal-conditioned reinforcement learning methods have shown promise for reach-avoid tasks, where an agent must reach a target state while avoiding undesirable regions of the state space. Existing approaches typically encode avoid-region information into an augmented state space and cost function, which prevents flexible, dynamic specification of novel avoid-region information at evaluation time. They also rely heavily on well-designed reward and cost functions, limiting scalability to complex or poorly structured environments. We introduce RADT, a decision transformer model for offline, reward-free, goal-conditioned, avoid region-conditioned RL. RADT encodes goals and avoid regions directly as prompt tokens, allowing any number of avoid regions of arbitrary size to be specified at evaluation time. Using only suboptimal offline trajectories from a random policy, RADT learns reach-avoid behavior through a novel combination of goal and avoid-region hindsight relabeling. We benchmark RADT against 3 existing offline goal-conditioned RL models across 11 tasks, environments, and experimental settings. RADT generalizes in a zero-shot manner to out-of-distribution avoid region sizes and counts, outperforming baselines that require retraining. In one such zero-shot setting, RADT achieves 35.7% improvement in normalized cost over the best retrained baseline while maintaining high goal-reaching success. We apply RADT to cell reprogramming in biology, where it reduces visits to undesirable intermediate gene expression states during trajectories to desired target states, despite stochastic transitions and discrete, structured state dynamics. 

---
# Demand Selection for VRP with Emission Quota 

**Authors**: Farid Najar, Dominique Barth, Yann Strozecki  

**Link**: [PDF](https://arxiv.org/pdf/2505.19315)  

**Abstract**: Combinatorial optimization (CO) problems are traditionally addressed using Operations Research (OR) methods, including metaheuristics. In this study, we introduce a demand selection problem for the Vehicle Routing Problem (VRP) with an emission quota, referred to as QVRP. The objective is to minimize the number of omitted deliveries while respecting the pollution quota. We focus on the demand selection part, called Maximum Feasible Vehicle Assignment (MFVA), while the construction of a routing for the VRP instance is solved using classical OR methods. We propose several methods for selecting the packages to omit, both from machine learning (ML) and OR. Our results show that, in this static problem setting, classical OR-based methods consistently outperform ML-based approaches. 

---
# SoloSpeech: Enhancing Intelligibility and Quality in Target Speech Extraction through a Cascaded Generative Pipeline 

**Authors**: Helin Wang, Jiarui Hai, Dongchao Yang, Chen Chen, Kai Li, Junyi Peng, Thomas Thebaud, Laureano Moro Velazquez, Jesus Villalba, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2505.19314)  

**Abstract**: Target Speech Extraction (TSE) aims to isolate a target speaker's voice from a mixture of multiple speakers by leveraging speaker-specific cues, typically provided as auxiliary audio (a.k.a. cue audio). Although recent advancements in TSE have primarily employed discriminative models that offer high perceptual quality, these models often introduce unwanted artifacts, reduce naturalness, and are sensitive to discrepancies between training and testing environments. On the other hand, generative models for TSE lag in perceptual quality and intelligibility. To address these challenges, we present SoloSpeech, a novel cascaded generative pipeline that integrates compression, extraction, reconstruction, and correction processes. SoloSpeech features a speaker-embedding-free target extractor that utilizes conditional information from the cue audio's latent space, aligning it with the mixture audio's latent space to prevent mismatches. Evaluated on the widely-used Libri2Mix dataset, SoloSpeech achieves the new state-of-the-art intelligibility and quality in target speech extraction and speech separation tasks while demonstrating exceptional generalization on out-of-domain data and real-world scenarios. 

---
# Retrieval-Augmented Generation for Service Discovery: Chunking Strategies and Benchmarking 

**Authors**: Robin D. Pesl, Jerin G. Mathew, Massimo Mecella, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2505.19310)  

**Abstract**: Integrating multiple (sub-)systems is essential to create advanced Information Systems. Difficulties mainly arise when integrating dynamic environments, e.g., the integration at design time of not yet existing services. This has been traditionally addressed using a registry that provides the API documentation of the endpoints. Large Language Models have shown to be capable of automatically creating system integrations (e.g., as service composition) based on this documentation but require concise input due to input oken limitations, especially regarding comprehensive API descriptions. Currently, it is unknown how best to preprocess these API descriptions. In the present work, we (i) analyze the usage of Retrieval Augmented Generation for endpoint discovery and the chunking, i.e., preprocessing, of state-of-practice OpenAPIs to reduce the input oken length while preserving the most relevant information. To further reduce the input token length for the composition prompt and improve endpoint retrieval, we propose (ii) a Discovery Agent that only receives a summary of the most relevant endpoints nd retrieves specification details on demand. We evaluate RAG for endpoint discovery using (iii) a proposed novel service discovery benchmark SOCBench-D representing a general setting across numerous domains and the real-world RestBench enchmark, first, for the different chunking possibilities and parameters measuring the endpoint retrieval accuracy. Then, we assess the Discovery Agent using the same test data set. The prototype shows how to successfully employ RAG for endpoint discovery to reduce the token count. Our experiments show that endpoint-based approaches outperform naive chunking methods for preprocessing. Relying on an agent significantly improves precision while being prone to decrease recall, disclosing the need for further reasoning capabilities. 

---
# A Novel Zero-Trust Identity Framework for Agentic AI: Decentralized Authentication and Fine-Grained Access Control 

**Authors**: Ken Huang, Vineeth Sai Narajala, John Yeoh, Ramesh Raskar, Youssef Harkati, Jerry Huang, Idan Habler, Chris Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2505.19301)  

**Abstract**: Traditional Identity and Access Management (IAM) systems, primarily designed for human users or static machine identities via protocols such as OAuth, OpenID Connect (OIDC), and SAML, prove fundamentally inadequate for the dynamic, interdependent, and often ephemeral nature of AI agents operating at scale within Multi Agent Systems (MAS), a computational system composed of multiple interacting intelligent agents that work collectively.
This paper posits the imperative for a novel Agentic AI IAM framework: We deconstruct the limitations of existing protocols when applied to MAS, illustrating with concrete examples why their coarse-grained controls, single-entity focus, and lack of context-awareness falter. We then propose a comprehensive framework built upon rich, verifiable Agent Identities (IDs), leveraging Decentralized Identifiers (DIDs) and Verifiable Credentials (VCs), that encapsulate an agents capabilities, provenance, behavioral scope, and security posture.
Our framework includes an Agent Naming Service (ANS) for secure and capability-aware discovery, dynamic fine-grained access control mechanisms, and critically, a unified global session management and policy enforcement layer for real-time control and consistent revocation across heterogeneous agent communication protocols. We also explore how Zero-Knowledge Proofs (ZKPs) enable privacy-preserving attribute disclosure and verifiable policy compliance.
We outline the architecture, operational lifecycle, innovative contributions, and security considerations of this new IAM paradigm, aiming to establish the foundational trust, accountability, and security necessary for the burgeoning field of agentic AI and the complex ecosystems they will inhabit. 

---
# A Necessary Step toward Faithfulness: Measuring and Improving Consistency in Free-Text Explanations 

**Authors**: Lingjun Zhao, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2505.19299)  

**Abstract**: Faithful free-text explanations are important to ensure transparency in high-stakes AI decision-making contexts, but they are challenging to generate by language models and assess by humans. In this paper, we present a measure for Prediction-EXplanation (PEX) consistency, by extending the concept of weight of evidence. This measure quantifies how much a free-text explanation supports or opposes a prediction, serving as an important aspect of explanation faithfulness. Our analysis reveals that more than 62% explanations generated by large language models lack this consistency. We show that applying direct preference optimization improves the consistency of generated explanations across three model families, with improvement ranging from 43.1% to 292.3%. Furthermore, we demonstrate that optimizing this consistency measure can improve explanation faithfulness by up to 9.7%. 

---
# 100-LongBench: Are de facto Long-Context Benchmarks Literally Evaluating Long-Context Ability? 

**Authors**: Wang Yang, Hongye Jin, Shaochen Zhong, Song Jiang, Qifan Wang, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.19293)  

**Abstract**: Long-context capability is considered one of the most important abilities of LLMs, as a truly long context-capable LLM enables users to effortlessly process many originally exhausting tasks -- e.g., digesting a long-form document to find answers vs. directly asking an LLM about it. However, existing real-task-based long-context evaluation benchmarks have two major shortcomings. First, benchmarks like LongBench often do not provide proper metrics to separate long-context performance from the model's baseline ability, making cross-model comparison unclear. Second, such benchmarks are usually constructed with fixed input lengths, which limits their applicability across different models and fails to reveal when a model begins to break down. To address these issues, we introduce a length-controllable long-context benchmark and a novel metric that disentangles baseline knowledge from true long-context capabilities. Experiments demonstrate the superiority of our approach in effectively evaluating LLMs. 

---
# TextDiffuser-RL: Efficient and Robust Text Layout Optimization for High-Fidelity Text-to-Image Synthesis 

**Authors**: Kazi Mahathir Rahman, Showrin Rahman, Sharmin Sultana Srishty  

**Link**: [PDF](https://arxiv.org/pdf/2505.19291)  

**Abstract**: Text-embedded image generation plays a critical role in industries such as graphic design, advertising, and digital content creation. Text-to-Image generation methods leveraging diffusion models, such as TextDiffuser-2, have demonstrated promising results in producing images with embedded text. TextDiffuser-2 effectively generates bounding box layouts that guide the rendering of visual text, achieving high fidelity and coherence. However, existing approaches often rely on resource-intensive processes and are limited in their ability to run efficiently on both CPU and GPU platforms. To address these challenges, we propose a novel two-stage pipeline that integrates reinforcement learning (RL) for rapid and optimized text layout generation with a diffusion-based image synthesis model. Our RL-based approach significantly accelerates the bounding box prediction step while reducing overlaps, allowing the system to run efficiently on both CPUs and GPUs. Extensive evaluations demonstrate that our framework maintains or surpasses TextDiffuser-2's quality in text placement and image synthesis, with markedly faster runtime and increased flexibility. Extensive evaluations demonstrate that our framework maintains or surpasses TextDiffuser-2's quality in text placement and image synthesis, with markedly faster runtime and increased flexibility. Our approach has been evaluated on the MARIOEval benchmark, achieving OCR and CLIPScore metrics close to state-of-the-art models, while being 97.64% more faster and requiring only 2MB of memory to run. 

---
# Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation 

**Authors**: Giuseppe Ruggiero, Matteo Testa, Jurgen Van de Walle, Luigi Di Caro  

**Link**: [PDF](https://arxiv.org/pdf/2505.19273)  

**Abstract**: Self-supervised learning (SSL) has reduced the reliance on expensive labeling in speech technologies by learning meaningful representations from unannotated data. Since most SSL-based downstream tasks prioritize content information in speech, ideal representations should disentangle content from unwanted variations like speaker characteristics in the SSL representations. However, removing speaker information often degrades other speech components, and existing methods either fail to fully disentangle speaker identity or require resource-intensive models. In this paper, we propose a novel disentanglement method that linearly decomposes SSL representations into speaker-specific and speaker-independent components, effectively generating speaker disentangled representations. Comprehensive experiments show that our approach achieves speaker independence and as such, when applied to content-driven tasks such as voice conversion, our representations yield significant improvements over state-of-the-art methods. 

---
# Cellular Traffic Prediction via Byzantine-robust Asynchronous Federated Learning 

**Authors**: Hui Ma, Kai Yang, Yang Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19263)  

**Abstract**: Network traffic prediction plays a crucial role in intelligent network operation. Traditional prediction methods often rely on centralized training, necessitating the transfer of vast amounts of traffic data to a central server. This approach can lead to latency and privacy concerns. To address these issues, federated learning integrated with differential privacy has emerged as a solution to improve data privacy and model robustness in distributed settings. Nonetheless, existing federated learning protocols are vulnerable to Byzantine attacks, which may significantly compromise model robustness. Developing a robust and privacy-preserving prediction model in the presence of Byzantine clients remains a significant challenge. To this end, we propose an asynchronous differential federated learning framework based on distributionally robust optimization. The proposed framework utilizes multiple clients to train the prediction model collaboratively with local differential privacy. In addition, regularization techniques have been employed to further improve the Byzantine robustness of the models. We have conducted extensive experiments on three real-world datasets, and the results elucidate that our proposed distributed algorithm can achieve superior performance over existing methods. 

---
# Enhancing Text-to-Image Diffusion Transformer via Split-Text Conditioning 

**Authors**: Yu Zhang, Jialei Zhou, Xinchen Li, Qi Zhang, Zhongwei Wan, Tianyu Wang, Duoqian Miao, Changwei Wang, Longbing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19261)  

**Abstract**: Current text-to-image diffusion generation typically employs complete-text conditioning. Due to the intricate syntax, diffusion transformers (DiTs) inherently suffer from a comprehension defect of complete-text captions. One-fly complete-text input either overlooks critical semantic details or causes semantic confusion by simultaneously modeling diverse semantic primitive types. To mitigate this defect of DiTs, we propose a novel split-text conditioning framework named DiT-ST. This framework converts a complete-text caption into a split-text caption, a collection of simplified sentences, to explicitly express various semantic primitives and their interconnections. The split-text caption is then injected into different denoising stages of DiT-ST in a hierarchical and incremental manner. Specifically, DiT-ST leverages Large Language Models to parse captions, extracting diverse primitives and hierarchically sorting out and constructing these primitives into a split-text input. Moreover, we partition the diffusion denoising process according to its differential sensitivities to diverse semantic primitive types and determine the appropriate timesteps to incrementally inject tokens of diverse semantic primitive types into input tokens via cross-attention. In this way, DiT-ST enhances the representation learning of specific semantic primitive types across different stages. Extensive experiments validate the effectiveness of our proposed DiT-ST in mitigating the complete-text comprehension defect. 

---
# Towards Large Reasoning Models for Agriculture 

**Authors**: Hossein Zaremehrjerdi, Shreyan Ganguly, Ashlyn Rairdin, Elizabeth Tranel, Benjamin Feuer, Juan Ignacio Di Salvo, Srikanth Panthulugiri, Victoria Moser, Sarah Jones, Joscif G Raigne, Yanben Shen, Heidi M. Dornath, Aditya Balu, Adarsh Krishnamurthy, Asheesh K Singh, Arti Singh, Baskar Ganapathysubramanian, Chinmay Hegde, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.19259)  

**Abstract**: Agricultural decision-making involves complex, context-specific reasoning, where choices about crops, practices, and interventions depend heavily on geographic, climatic, and economic conditions. Traditional large language models (LLMs) often fall short in navigating this nuanced problem due to limited reasoning capacity. We hypothesize that recent advances in large reasoning models (LRMs) can better handle such structured, domain-specific inference. To investigate this, we introduce AgReason, the first expert-curated open-ended science benchmark with 100 questions for agricultural reasoning. Evaluations across thirteen open-source and proprietary models reveal that LRMs outperform conventional ones, though notable challenges persist, with the strongest Gemini-based baseline achieving 36% accuracy. We also present AgThoughts, a large-scale dataset of 44.6K question-answer pairs generated with human oversight and equipped with synthetically generated reasoning traces. Using AgThoughts, we develop AgThinker, a suite of small reasoning models that can be run on consumer-grade GPUs, and show that our dataset can be effective in unlocking agricultural reasoning abilities in LLMs. Our project page is here: this https URL 

---
# VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use 

**Authors**: Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, Klara Nahrstedt  

**Link**: [PDF](https://arxiv.org/pdf/2505.19255)  

**Abstract**: Reinforcement Learning Finetuning (RFT) has significantly advanced the reasoning capabilities of large language models (LLMs) by enabling long chains of thought, self-correction, and effective tool use. While recent works attempt to extend RFT to vision-language models (VLMs), these efforts largely produce text-only reasoning conditioned on static image inputs, falling short of true multimodal reasoning in the response. In contrast, test-time methods like Visual Sketchpad incorporate visual steps but lack training mechanisms.
We introduce VTool-R1, the first framework that trains VLMs to generate multimodal chains of thought by interleaving text and intermediate visual reasoning steps. VTool-R1 integrates Python-based visual editing tools into the RFT process, enabling VLMs to learn when and how to generate visual reasoning steps that benefit final reasoning. Trained with outcome-based rewards tied to task accuracy, our approach elicits strategic visual tool use for reasoning without relying on process-based supervision. Experiments on structured visual question answering over charts and tables show that VTool-R1 enhances reasoning performance by teaching VLMs to "think with images" and generate multimodal chain of thoughts with tools. 

---
# Learning-Augmented Online Bipartite Fractional Matching 

**Authors**: Davin Choo, Billy Jin, Yongho Shin  

**Link**: [PDF](https://arxiv.org/pdf/2505.19252)  

**Abstract**: Online bipartite matching is a fundamental problem in online optimization, extensively studied both in its integral and fractional forms due to its theoretical significance and practical applications, such as online advertising and resource allocation. Motivated by recent progress in learning-augmented algorithms, we study online bipartite fractional matching when the algorithm is given advice in the form of a suggested matching in each iteration. We develop algorithms for both the vertex-weighted and unweighted variants that provably dominate the naive "coin flip" strategy of randomly choosing between the advice-following and advice-free algorithms. Moreover, our algorithm for the vertex-weighted setting extends to the AdWords problem under the small bids assumption, yielding a significant improvement over the seminal work of Mahdian, Nazerzadeh, and Saberi (EC 2007, TALG 2012). Complementing our positive results, we establish a hardness bound on the robustness-consistency tradeoff that is attainable by any algorithm. We empirically validate our algorithms through experiments on synthetic and real-world data. 

---
# Improving Value Estimation Critically Enhances Vanilla Policy Gradient 

**Authors**: Tao Wang, Ruipeng Zhang, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.19247)  

**Abstract**: Modern policy gradient algorithms, such as TRPO and PPO, outperform vanilla policy gradient in many RL tasks. Questioning the common belief that enforcing approximate trust regions leads to steady policy improvement in practice, we show that the more critical factor is the enhanced value estimation accuracy from more value update steps in each iteration. To demonstrate, we show that by simply increasing the number of value update steps per iteration, vanilla policy gradient itself can achieve performance comparable to or better than PPO in all the standard continuous control benchmark environments. Importantly, this simple change to vanilla policy gradient is significantly more robust to hyperparameter choices, opening up the possibility that RL algorithms may still become more effective and easier to use. 

---
# To CoT or To Loop? A Formal Comparison Between Chain-of-Thought and Looped Transformers 

**Authors**: Kevin Xu, Issei Sato  

**Link**: [PDF](https://arxiv.org/pdf/2505.19245)  

**Abstract**: Chain-of-Thought (CoT) and Looped Transformers have been shown to empirically improve performance on reasoning tasks and to theoretically enhance expressivity by recursively increasing the number of computational steps. However, their comparative capabilities are still not well understood. In this paper, we provide a formal analysis of their respective strengths and limitations. We show that Looped Transformers can efficiently simulate parallel computations for deterministic tasks, which we formalize as evaluation over directed acyclic graphs. In contrast, CoT with stochastic decoding excels at approximate inference for compositional structures, namely self-reducible problems. These separations suggest the tasks for which depth-driven recursion is more suitable, thereby offering practical cues for choosing between reasoning paradigms. 

---
# ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment 

**Authors**: Xiaoqiang Lin, Arun Verma, Zhongxiang Dai, Daniela Rus, See-Kiong Ng, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2505.19241)  

**Abstract**: The recent success of using human preferences to align large language models (LLMs) has significantly improved their performance in various downstream tasks like question answering, mathematical reasoning, and code generation. However,3 achieving effective LLM alignment depends on high-quality human preference datasets. Collecting these datasets requires human preference annotation, which is costly and resource-intensive, necessitating efficient active data selection methods. Existing methods either lack a strong theoretical foundation or depend on restrictive reward function assumptions (e.g., linearity). To this end, we propose an algorithm, ActiveDPO, that uses a theoretically grounded data selection criterion for non-linear reward functions while directly leveraging the LLM itself to parameterize the reward model that is used for active data selection. As a result, ActiveDPO explicitly accounts for the influence of LLM on data selection, unlike methods that select the data without considering the LLM that is being aligned, thereby leading to more effective and efficient data collection. Extensive experiments show that ActiveDPO outperforms existing methods across various models and datasets. 

---
# LLLMs: A Data-Driven Survey of Evolving Research on Limitations of Large Language Models 

**Authors**: Aida Kostikova, Zhipin Wang, Deidamea Bajri, Ole Pütz, Benjamin Paaßen, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2505.19240)  

**Abstract**: Large language model (LLM) research has grown rapidly, along with increasing concern about their limitations such as failures in reasoning, hallucinations, and limited multilingual capability. In this survey, we conduct a data-driven, semi-automated review of research on limitations of LLM (LLLMs) from 2022 to 2024 using a bottom-up approach. From a corpus of 250,000 ACL and arXiv papers, we identify 14,648 relevant papers using keyword filtering, LLM-based classification, validated against expert labels, and topic clustering (via two approaches, HDBSCAN+BERTopic and LlooM). We find that LLM-related research increases over fivefold in ACL and fourfold in arXiv. Since 2022, LLLMs research grows even faster, reaching over 30% of LLM papers by late 2024. Reasoning remains the most studied limitation, followed by generalization, hallucination, bias, and security. The distribution of topics in the ACL dataset stays relatively stable over time, while arXiv shifts toward safety and controllability (with topics like security risks, alignment, hallucinations, knowledge editing), and multimodality between 2022 and 2024. We release a dataset of annotated abstracts and a validated methodology, and offer a quantitative view of trends in LLM limitations research. 

---
# Efficient Policy Optimization in Robust Constrained MDPs with Iteration Complexity Guarantees 

**Authors**: Sourav Ganguly, Arnob Ghosh, Kishan Panaganti, Adam Wierman  

**Link**: [PDF](https://arxiv.org/pdf/2505.19238)  

**Abstract**: Constrained decision-making is essential for designing safe policies in real-world control systems, yet simulated environments often fail to capture real-world adversities. We consider the problem of learning a policy that will maximize the cumulative reward while satisfying a constraint, even when there is a mismatch between the real model and an accessible simulator/nominal model. In particular, we consider the robust constrained Markov decision problem (RCMDP) where an agent needs to maximize the reward and satisfy the constraint against the worst possible stochastic model under the uncertainty set centered around an unknown nominal model. Primal-dual methods, effective for standard constrained MDP (CMDP), are not applicable here because of the lack of the strong duality property. Further, one cannot apply the standard robust value-iteration based approach on the composite value function either as the worst case models may be different for the reward value function and the constraint value function. We propose a novel technique that effectively minimizes the constraint value function--to satisfy the constraints; on the other hand, when all the constraints are satisfied, it can simply maximize the robust reward value function. We prove that such an algorithm finds a policy with at most $\epsilon$ sub-optimality and feasible policy after $O(\epsilon^{-2})$ iterations. In contrast to the state-of-the-art method, we do not need to employ a binary search, thus, we reduce the computation time by at least 4x for smaller value of discount factor ($\gamma$) and by at least 6x for larger value of $\gamma$. 

---
# RAISE: Realness Assessment for Image Synthesis and Evaluation 

**Authors**: Aniruddha Mukherjee, Spriha Dubey, Somdyuti Paul  

**Link**: [PDF](https://arxiv.org/pdf/2505.19233)  

**Abstract**: The rapid advancement of generative AI has enabled the creation of highly photorealistic visual content, offering practical substitutes for real images and videos in scenarios where acquiring real data is difficult or expensive. However, reliably substituting real visual content with AI-generated counterparts requires robust assessment of the perceived realness of AI-generated visual content, a challenging task due to its inherent subjective nature. To address this, we conducted a comprehensive human study evaluating the perceptual realness of both real and AI-generated images, resulting in a new dataset, containing images paired with subjective realness scores, introduced as RAISE in this paper. Further, we develop and train multiple models on RAISE to establish baselines for realness prediction. Our experimental results demonstrate that features derived from deep foundation vision models can effectively capture the subjective realness. RAISE thus provides a valuable resource for developing robust, objective models of perceptual realness assessment. 

---
# When Ethics and Payoffs Diverge: LLM Agents in Morally Charged Social Dilemmas 

**Authors**: Steffen Backmann, David Guzman Piedrahita, Emanuel Tewolde, Rada Mihalcea, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.19212)  

**Abstract**: Recent advances in large language models (LLMs) have enabled their use in complex agentic roles, involving decision-making with humans or other agents, making ethical alignment a key AI safety concern. While prior work has examined both LLMs' moral judgment and strategic behavior in social dilemmas, there is limited understanding of how they act when moral imperatives directly conflict with rewards or incentives. To investigate this, we introduce Moral Behavior in Social Dilemma Simulation (MoralSim) and evaluate how LLMs behave in the prisoner's dilemma and public goods game with morally charged contexts. In MoralSim, we test a range of frontier models across both game structures and three distinct moral framings, enabling a systematic examination of how LLMs navigate social dilemmas in which ethical norms conflict with payoff-maximizing strategies. Our results show substantial variation across models in both their general tendency to act morally and the consistency of their behavior across game types, the specific moral framing, and situational factors such as opponent behavior and survival risks. Crucially, no model exhibits consistently moral behavior in MoralSim, highlighting the need for caution when deploying LLMs in agentic roles where the agent's "self-interest" may conflict with ethical expectations. Our code is available at this https URL. 

---
# MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery via Hierarchical Search 

**Authors**: Zonglin Yang, Wanhao Liu, Ben Gao, Yujie Liu, Wei Li, Tong Xie, Lidong Bing, Wanli Ouyang, Erik Cambria, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.19209)  

**Abstract**: Large language models (LLMs) have shown promise in automating scientific hypothesis generation, yet existing approaches primarily yield coarse-grained hypotheses lacking critical methodological and experimental details. We introduce and formally define the novel task of fine-grained scientific hypothesis discovery, which entails generating detailed, experimentally actionable hypotheses from coarse initial research directions. We frame this as a combinatorial optimization problem and investigate the upper limits of LLMs' capacity to solve it when maximally leveraged. Specifically, we explore four foundational questions: (1) how to best harness an LLM's internal heuristics to formulate the fine-grained hypothesis it itself would judge as the most promising among all the possible hypotheses it might generate, based on its own internal scoring-thus defining a latent reward landscape over the hypothesis space; (2) whether such LLM-judged better hypotheses exhibit stronger alignment with ground-truth hypotheses; (3) whether shaping the reward landscape using an ensemble of diverse LLMs of similar capacity yields better outcomes than defining it with repeated instances of the strongest LLM among them; and (4) whether an ensemble of identical LLMs provides a more reliable reward landscape than a single LLM. To address these questions, we propose a hierarchical search method that incrementally proposes and integrates details into the hypothesis, progressing from general concepts to specific experimental configurations. We show that this hierarchical process smooths the reward landscape and enables more effective optimization. Empirical evaluations on a new benchmark of expert-annotated fine-grained hypotheses from recent chemistry literature show that our method consistently outperforms strong baselines. 

---
# OptiMindTune: A Multi-Agent Framework for Intelligent Hyperparameter Optimization 

**Authors**: Meher Bhaskar Madiraju, Meher Sai Preetam Madiraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.19205)  

**Abstract**: Hyperparameter optimization (HPO) is a critical yet challenging aspect of machine learning model development, significantly impacting model performance and generalization. Traditional HPO methods often struggle with high dimensionality, complex interdependencies, and computational expense. This paper introduces OptiMindTune, a novel multi-agent framework designed to intelligently and efficiently optimize hyperparameters. OptiMindTune leverages the collaborative intelligence of three specialized AI agents -- a Recommender Agent, an Evaluator Agent, and a Decision Agent -- each powered by Google's Gemini models. These agents address distinct facets of the HPO problem, from model selection and hyperparameter suggestion to robust evaluation and strategic decision-making. By fostering dynamic interactions and knowledge sharing, OptiMindTune aims to converge to optimal hyperparameter configurations more rapidly and robustly than existing single-agent or monolithic approaches. Our framework integrates principles from advanced large language models, and adaptive search to achieve scalable and intelligent AutoML. We posit that this multi-agent paradigm offers a promising avenue for tackling the increasing complexity of modern machine learning model tuning. 

---
# EnvSDD: Benchmarking Environmental Sound Deepfake Detection 

**Authors**: Han Yin, Yang Xiao, Rohan Kumar Das, Jisheng Bai, Haohe Liu, Wenwu Wang, Mark D Plumbley  

**Link**: [PDF](https://arxiv.org/pdf/2505.19203)  

**Abstract**: Audio generation systems now create very realistic soundscapes that can enhance media production, but also pose potential risks. Several studies have examined deepfakes in speech or singing voice. However, environmental sounds have different characteristics, which may make methods for detecting speech and singing deepfakes less effective for real-world sounds. In addition, existing datasets for environmental sound deepfake detection are limited in scale and audio types. To address this gap, we introduce EnvSDD, the first large-scale curated dataset designed for this task, consisting of 45.25 hours of real and 316.74 hours of fake audio. The test set includes diverse conditions to evaluate the generalizability, such as unseen generation models and unseen datasets. We also propose an audio deepfake detection system, based on a pre-trained audio foundation model. Results on EnvSDD show that our proposed system outperforms the state-of-the-art systems from speech and singing domains. 

---
# Curvature Dynamic Black-box Attack: revisiting adversarial robustness via dynamic curvature estimation 

**Authors**: Peiran Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.19194)  

**Abstract**: Adversarial attack reveals the vulnerability of deep learning models. For about a decade, countless attack and defense methods have been proposed, leading to robustified classifiers and better understanding of models. Among these methods, curvature-based approaches have attracted attention because it is assumed that high curvature may give rise to rough decision boundary. However, the most commonly used \textit{curvature} is the curvature of loss function, scores or other parameters from within the model as opposed to decision boundary curvature, since the former can be relatively easily formed using second order derivative. In this paper, we propose a new query-efficient method, dynamic curvature estimation(DCE), to estimate the decision boundary curvature in a black-box setting. Our approach is based on CGBA, a black-box adversarial attack. By performing DCE on a wide range of classifiers, we discovered, statistically, a connection between decision boundary curvature and adversarial robustness. We also propose a new attack method, curvature dynamic black-box attack(CDBA) with improved performance using the dynamically estimated curvature. 

---
# I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts 

**Authors**: Jiayi Xin, Sukwon Yun, Jie Peng, Inyoung Choi, Jenna L. Ballard, Tianlong Chen, Qi Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.19190)  

**Abstract**: Modality fusion is a cornerstone of multimodal learning, enabling information integration from diverse data sources. However, vanilla fusion methods are limited by (1) inability to account for heterogeneous interactions between modalities and (2) lack of interpretability in uncovering the multimodal interactions inherent in the data. To this end, we propose I2MoE (Interpretable Multimodal Interaction-aware Mixture of Experts), an end-to-end MoE framework designed to enhance modality fusion by explicitly modeling diverse multimodal interactions, as well as providing interpretation on a local and global level. First, I2MoE utilizes different interaction experts with weakly supervised interaction losses to learn multimodal interactions in a data-driven way. Second, I2MoE deploys a reweighting model that assigns importance scores for the output of each interaction expert, which offers sample-level and dataset-level interpretation. Extensive evaluation of medical and general multimodal datasets shows that I2MoE is flexible enough to be combined with different fusion techniques, consistently improves task performance, and provides interpretation across various real-world scenarios. Code is available at this https URL. 

---
# LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling 

**Authors**: Yang Xiao, Jiashuo Wang, Ruifeng Yuan, Chunpu Xu, Kaishuai Xu, Wenjie Li, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19187)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning capabilities through test-time scaling approaches, particularly when fine-tuned with chain-of-thought (CoT) data distilled from more powerful large reasoning models (LRMs). However, these reasoning chains often contain verbose elements that mirror human problem-solving, categorized as progressive reasoning (the essential solution development path) and functional elements (verification processes, alternative solution approaches, and error corrections). While progressive reasoning is crucial, the functional elements significantly increase computational demands during test-time inference. We introduce PIR (Perplexity-based Importance Refinement), a principled framework that quantitatively evaluates the importance of each reasoning step based on its impact on answer prediction confidence. PIR systematically identifies and selectively prunes only low-importance functional steps while preserving progressive reasoning components, creating optimized training data that maintains the integrity of the core solution path while reducing verbosity. Models fine-tuned on PIR-optimized data exhibit superior test-time scaling properties, generating more concise reasoning chains while achieving improved accuracy (+0.9\% to +6.6\%) with significantly reduced token usage (-3\% to -41\%) across challenging reasoning benchmarks (AIME, AMC, and GPQA Diamond). Our approach demonstrates strong generalizability across different model sizes, data sources, and token budgets, offering a practical solution for deploying reasoning-capable LLMs in scenarios where efficient test-time scaling, response time, and computational efficiency are valuable constraints. 

---
# PosePilot: An Edge-AI Solution for Posture Correction in Physical Exercises 

**Authors**: Rushiraj Gadhvi, Priyansh Desai, Siddharth  

**Link**: [PDF](https://arxiv.org/pdf/2505.19186)  

**Abstract**: Automated pose correction remains a significant challenge in AI-driven fitness systems, despite extensive research in activity recognition. This work presents PosePilot, a novel system that integrates pose recognition with real-time personalized corrective feedback, overcoming the limitations of traditional fitness solutions. Using Yoga, a discipline requiring precise spatio-temporal alignment as a case study, we demonstrate PosePilot's ability to analyze complex physical movements. Designed for deployment on edge devices, PosePilot can be extended to various at-home and outdoor exercises. We employ a Vanilla LSTM, allowing the system to capture temporal dependencies for pose recognition. Additionally, a BiLSTM with multi-head Attention enhances the model's ability to process motion contexts, selectively focusing on key limb angles for accurate error detection while maintaining computational efficiency. As part of this work, we introduce a high-quality video dataset used for evaluating our models. Most importantly, PosePilot provides instant corrective feedback at every stage of a movement, ensuring precise posture adjustments throughout the exercise routine. The proposed approach 1) performs automatic human posture recognition, 2) provides personalized posture correction feedback at each instant which is crucial in Yoga, and 3) offers a lightweight and robust posture correction model feasible for deploying on edge devices in real-world environments. 

---
# Two LLMs debate, both are certain they've won 

**Authors**: Minh Nhat Nguyen, Pradyumna Shyama Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2505.19184)  

**Abstract**: Can LLMs accurately adjust their confidence when facing opposition? Building on previous studies measuring calibration on static fact-based question-answering tasks, we evaluate Large Language Models (LLMs) in a dynamic, adversarial debate setting, uniquely combining two realistic factors: (a) a multi-turn format requiring models to update beliefs as new information emerges, and (b) a zero-sum structure to control for task-related uncertainty, since mutual high-confidence claims imply systematic overconfidence. We organized 60 three-round policy debates among ten state-of-the-art LLMs, with models privately rating their confidence (0-100) in winning after each round. We observed five concerning patterns: (1) Systematic overconfidence: models began debates with average initial confidence of 72.9% vs. a rational 50% baseline. (2) Confidence escalation: rather than reducing confidence as debates progressed, debaters increased their win probabilities, averaging 83% by the final round. (3) Mutual overestimation: in 61.7% of debates, both sides simultaneously claimed >=75% probability of victory, a logical impossibility. (4) Persistent self-debate bias: models debating identical copies increased confidence from 64.1% to 75.2%; even when explicitly informed their chance of winning was exactly 50%, confidence still rose (from 50.0% to 57.1%). (5) Misaligned private reasoning: models' private scratchpad thoughts sometimes differed from their public confidence ratings, raising concerns about faithfulness of chain-of-thought reasoning. These results suggest LLMs lack the ability to accurately self-assess or update their beliefs in dynamic, multi-turn tasks; a major concern as LLM outputs are deployed without careful review in assistant roles or agentic settings. 

---
# Saliency-guided Emotion Modeling: Predicting Viewer Reactions from Video Stimuli 

**Authors**: Akhila Yaragoppa, Siddharth  

**Link**: [PDF](https://arxiv.org/pdf/2505.19178)  

**Abstract**: Understanding the emotional impact of videos is crucial for applications in content creation, advertising, and Human-Computer Interaction (HCI). Traditional affective computing methods rely on self-reported emotions, facial expression analysis, and biosensing data, yet they often overlook the role of visual saliency -- the naturally attention-grabbing regions within a video. In this study, we utilize deep learning to introduce a novel saliency-based approach to emotion prediction by extracting two key features: saliency area and number of salient regions. Using the HD2S saliency model and OpenFace facial action unit analysis, we examine the relationship between video saliency and viewer emotions. Our findings reveal three key insights: (1) Videos with multiple salient regions tend to elicit high-valence, low-arousal emotions, (2) Videos with a single dominant salient region are more likely to induce low-valence, high-arousal responses, and (3) Self-reported emotions often misalign with facial expression-based emotion detection, suggesting limitations in subjective reporting. By leveraging saliency-driven insights, this work provides a computationally efficient and interpretable alternative for emotion modeling, with implications for content creation, personalized media experiences, and affective computing research. 

---
# BroadGen: A Framework for Generating Effective and Efficient Advertiser Broad Match Keyphrase Recommendations 

**Authors**: Ashirbad Mishra, Jinyu Zhao, Soumik Dey, Hansi Wu, Binbin Li, Kamesh Madduri  

**Link**: [PDF](https://arxiv.org/pdf/2505.19164)  

**Abstract**: In the domain of sponsored search advertising, the focus of Keyphrase recommendation has largely been on exact match types, which pose issues such as high management expenses, limited targeting scope, and evolving search query patterns. Alternatives like Broad match types can alleviate certain drawbacks of exact matches but present challenges like poor targeting accuracy and minimal supervisory signals owing to limited advertiser usage. This research defines the criteria for an ideal broad match, emphasizing on both efficiency and effectiveness, ensuring that a significant portion of matched queries are relevant. We propose BroadGen, an innovative framework that recommends efficient and effective broad match keyphrases by utilizing historical search query data. Additionally, we demonstrate that BroadGen, through token correspondence modeling, maintains better query stability over time. BroadGen's capabilities allow it to serve daily, millions of sellers at eBay with over 2.3 billion items. 

---
# SpokenNativQA: Multilingual Everyday Spoken Queries for LLMs 

**Authors**: Firoj Alam, Md Arid Hasan, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.19163)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various disciplines and tasks. However, benchmarking their capabilities with multilingual spoken queries remains largely unexplored. In this study, we introduce SpokenNativQA, the first multilingual and culturally aligned spoken question-answering (SQA) dataset designed to evaluate LLMs in real-world conversational settings. The dataset comprises approximately 33,000 naturally spoken questions and answers in multiple languages, including low-resource and dialect-rich languages, providing a robust benchmark for assessing LLM performance in speech-based interactions. SpokenNativQA addresses the limitations of text-based QA datasets by incorporating speech variability, accents, and linguistic diversity. We benchmark different ASR systems and LLMs for SQA and present our findings. We released the data at (this https URL) and the experimental scripts at (this https URL) for the research community. 

---
# SRDiffusion: Accelerate Video Diffusion Inference via Sketching-Rendering Cooperation 

**Authors**: Shenggan Cheng, Yuanxin Wei, Lansong Diao, Yong Liu, Bujiao Chen, Lianghua Huang, Yu Liu, Wenyuan Yu, Jiangsu Du, Wei Lin, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2505.19151)  

**Abstract**: Leveraging the diffusion transformer (DiT) architecture, models like Sora, CogVideoX and Wan have achieved remarkable progress in text-to-video, image-to-video, and video editing tasks. Despite these advances, diffusion-based video generation remains computationally intensive, especially for high-resolution, long-duration videos. Prior work accelerates its inference by skipping computation, usually at the cost of severe quality degradation. In this paper, we propose SRDiffusion, a novel framework that leverages collaboration between large and small models to reduce inference cost. The large model handles high-noise steps to ensure semantic and motion fidelity (Sketching), while the smaller model refines visual details in low-noise steps (Rendering). Experimental results demonstrate that our method outperforms existing approaches, over 3$\times$ speedup for Wan with nearly no quality loss for VBench, and 2$\times$ speedup for CogVideoX. Our method is introduced as a new direction orthogonal to existing acceleration strategies, offering a practical solution for scalable video generation. 

---
# Shifting AI Efficiency From Model-Centric to Data-Centric Compression 

**Authors**: Xuyang Liu, Zichen Wen, Shaobo Wang, Junjie Chen, Zhishan Tao, Yubo Wang, Xiangqi Jin, Chang Zou, Yiyu Wang, Chenfei Liao, Xu Zheng, Honggang Chen, Weijia Li, Xuming Hu, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19147)  

**Abstract**: The rapid advancement of large language models (LLMs) and multi-modal LLMs (MLLMs) has historically relied on model-centric scaling through increasing parameter counts from millions to hundreds of billions to drive performance gains. However, as we approach hardware limits on model size, the dominant computational bottleneck has fundamentally shifted to the quadratic cost of self-attention over long token sequences, now driven by ultra-long text contexts, high-resolution images, and extended videos. In this position paper, \textbf{we argue that the focus of research for efficient AI is shifting from model-centric compression to data-centric compression}. We position token compression as the new frontier, which improves AI efficiency via reducing the number of tokens during model training or inference. Through comprehensive analysis, we first examine recent developments in long-context AI across various domains and establish a unified mathematical framework for existing model efficiency strategies, demonstrating why token compression represents a crucial paradigm shift in addressing long-context overhead. Subsequently, we systematically review the research landscape of token compression, analyzing its fundamental benefits and identifying its compelling advantages across diverse scenarios. Furthermore, we provide an in-depth analysis of current challenges in token compression research and outline promising future directions. Ultimately, our work aims to offer a fresh perspective on AI efficiency, synthesize existing research, and catalyze innovative developments to address the challenges that increasing context lengths pose to the AI community's advancement. 

---
# RetrieveAll: A Multilingual Named Entity Recognition Framework with Large Language Models 

**Authors**: Jin Zhang, Fan Gao, Linyu Li, Yongbin Yu, Xiangxiang Wang, Nyima Tashi, Gadeng Luosang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19128)  

**Abstract**: The rise of large language models has led to significant performance breakthroughs in named entity recognition (NER) for high-resource languages, yet there remains substantial room for improvement in low- and medium-resource languages. Existing multilingual NER methods face severe language interference during the multi-language adaptation process, manifested in feature conflicts between different languages and the competitive suppression of low-resource language features by high-resource languages. Although training a dedicated model for each language can mitigate such interference, it lacks scalability and incurs excessive computational costs in real-world applications. To address this issue, we propose RetrieveAll, a universal multilingual NER framework based on dynamic LoRA. The framework decouples task-specific features across languages and demonstrates efficient dynamic adaptability. Furthermore, we introduce a cross-granularity knowledge augmented method that fully exploits the intrinsic potential of the data without relying on external resources. By leveraging a hierarchical prompting mechanism to guide knowledge injection, this approach advances the paradigm from "prompt-guided inference" to "prompt-driven learning." Experimental results show that RetrieveAll outperforms existing baselines; on the PAN-X dataset, it achieves an average F1 improvement of 12.1 percent. 

---
# CloneShield: A Framework for Universal Perturbation Against Zero-Shot Voice Cloning 

**Authors**: Renyuan Li, Zhibo Liang, Haichuan Zhang, Tianyu Shi, Zhiyuan Cheng, Jia Shi, Carl Yang, Mingjie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19119)  

**Abstract**: Recent breakthroughs in text-to-speech (TTS) voice cloning have raised serious privacy concerns, allowing highly accurate vocal identity replication from just a few seconds of reference audio, while retaining the speaker's vocal authenticity. In this paper, we introduce CloneShield, a universal time-domain adversarial perturbation framework specifically designed to defend against zero-shot voice cloning. Our method provides protection that is robust across speakers and utterances, without requiring any prior knowledge of the synthesized text. We formulate perturbation generation as a multi-objective optimization problem, and propose Multi-Gradient Descent Algorithm (MGDA) to ensure the robust protection across diverse utterances. To preserve natural auditory perception for users, we decompose the adversarial perturbation via Mel-spectrogram representations and fine-tune it for each sample. This design ensures imperceptibility while maintaining strong degradation effects on zero-shot cloned outputs. Experiments on three state-of-the-art zero-shot TTS systems, five benchmark datasets and evaluations from 60 human listeners demonstrate that our method preserves near-original audio quality in protected inputs (PESQ = 3.90, SRS = 0.93) while substantially degrading both speaker similarity and speech quality in cloned samples (PESQ = 1.07, SRS = 0.08). 

---
# FP4 All the Way: Fully Quantized Training of LLMs 

**Authors**: Brian Chmiel, Maxim Fishman, Ron Banner, Daniel Soudry  

**Link**: [PDF](https://arxiv.org/pdf/2505.19115)  

**Abstract**: We demonstrate, for the first time, fully quantized training (FQT) of large language models (LLMs) using predominantly 4-bit floating-point (FP4) precision for weights, activations, and gradients on datasets up to 200 billion tokens. We extensively investigate key design choices for FP4, including block sizes, scaling formats, and rounding methods. Our analysis shows that the NVFP4 format, where each block of 16 FP4 values (E2M1) shares a scale represented in E4M3, provides optimal results. We use stochastic rounding for backward and update passes and round-to-nearest for the forward pass to enhance stability. Additionally, we identify a theoretical and empirical threshold for effective quantized training: when the gradient norm falls below approximately $\sqrt{3}$ times the quantization noise, quantized training becomes less effective. Leveraging these insights, we successfully train a 7-billion-parameter model on 256 Intel Gaudi2 accelerators. The resulting FP4-trained model achieves downstream task performance comparable to a standard BF16 baseline, confirming that FP4 training is a practical and highly efficient approach for large-scale LLM training. A reference implementation is supplied in this https URL . 

---
# An Interpretable Representation Learning Approach for Diffusion Tensor Imaging 

**Authors**: Vishwa Mohan Singh, Alberto Gaston Villagran Asiares, Luisa Sophie Schuhmacher, Kate Rendall, Simon Weißbrod, David Rügamer, Inga Körte  

**Link**: [PDF](https://arxiv.org/pdf/2505.19110)  

**Abstract**: Diffusion Tensor Imaging (DTI) tractography offers detailed insights into the structural connectivity of the brain, but presents challenges in effective representation and interpretation in deep learning models. In this work, we propose a novel 2D representation of DTI tractography that encodes tract-level fractional anisotropy (FA) values into a 9x9 grayscale image. This representation is processed through a Beta-Total Correlation Variational Autoencoder with a Spatial Broadcast Decoder to learn a disentangled and interpretable latent embedding. We evaluate the quality of this embedding using supervised and unsupervised representation learning strategies, including auxiliary classification, triplet loss, and SimCLR-based contrastive learning. Compared to the 1D Group deep neural network (DNN) baselines, our approach improves the F1 score in a downstream sex classification task by 15.74% and shows a better disentanglement than the 3D representation. 

---
# CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models 

**Authors**: Yongheng Zhang, Xu Liu, Ruoxi Zhou, Qiguang Chen, Hao Fei, Wenpeng Lu, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.19108)  

**Abstract**: Investigating hallucination issues in large language models (LLMs) within cross-lingual and cross-modal scenarios can greatly advance the large-scale deployment in real-world applications. Nevertheless, the current studies are limited to a single scenario, either cross-lingual or cross-modal, leaving a gap in the exploration of hallucinations in the joint cross-lingual and cross-modal scenarios. Motivated by this, we introduce a novel joint Cross-lingual and Cross-modal Hallucinations benchmark (CCHall) to fill this gap. Specifically, CCHall simultaneously incorporates both cross-lingual and cross-modal hallucination scenarios, which can be used to assess the cross-lingual and cross-modal capabilities of LLMs. Furthermore, we conduct a comprehensive evaluation on CCHall, exploring both mainstream open-source and closed-source LLMs. The experimental results highlight that current LLMs still struggle with CCHall. We hope CCHall can serve as a valuable resource to assess LLMs in joint cross-lingual and cross-modal scenarios. 

---
# Enable Lightweight and Precision-Scalable Posit/IEEE-754 Arithmetic in RISC-V Cores for Transprecision Computing 

**Authors**: Qiong Li, Chao Fang, Longwei Huang, Jun Lin, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19096)  

**Abstract**: While posit format offers superior dynamic range and accuracy for transprecision computing, its adoption in RISC-V processors is hindered by the lack of a unified solution for lightweight, precision-scalable, and IEEE-754 arithmetic compatible hardware implementation. To address these challenges, we enhance RISC-V processors by 1) integrating dedicated posit codecs into the original FPU for lightweight implementation, 2) incorporating multi/mixed-precision support with dynamic exponent size for precision-scalability, and 3) reusing and customizing ISA extensions for IEEE-754 compatible posit operations. Our comprehensive evaluation spans the modified FPU, RISC-V core, and SoC levels. It demonstrates that our implementation achieves 47.9% LUTs and 57.4% FFs reduction compared to state-of-the-art posit-enabled RISC-V processors, while achieving up to 2.54$\times$ throughput improvement in various GEMM kernels. 

---
# SATORI-R1: Incentivizing Multimodal Reasoning with Spatial Grounding and Verifiable Rewards 

**Authors**: Chuming Shen, Wei Wei, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19094)  

**Abstract**: DeepSeek-R1 has demonstrated powerful reasoning capabilities in the text domain through stable reinforcement learning (RL). Recently, in the multimodal domain, works have begun to directly apply RL to generate R1-like free-form reasoning for Visual Question Answering (VQA) tasks. However, multimodal tasks share an intrinsically different nature from textual tasks, which heavily rely on the understanding of the input image to solve the problem. Therefore, such free-form reasoning faces two critical limitations in the VQA task: (1) Extended reasoning chains diffuse visual focus away from task-critical regions, degrading answer accuracy. (2) Unverifiable intermediate steps amplify policy-gradient variance and computational costs overhead. To address these issues, in this paper, we introduce SATORI ($\textbf{S}patially$ $\textbf{A}nchored$ $\textbf{T}ask$ $\textbf{O}ptimization$ with $\textbf{R}e\textbf{I}nforcement$ Learning), which decomposes VQA into three verifiable stages, including global image captioning, region localization, and answer prediction, each supplying explicit reward signals. Furthermore, we also introduce VQA-Verify, a 12k dataset annotated with answer-aligned captions and bounding-boxes to facilitate training. Experiments demonstrate consistent performance improvements across seven VQA benchmarks, achieving up to $15.7\%$ improvement in accuracy in accuracy compared to the R1-like baseline. Our analysis of the attention map confirms enhanced focus on critical regions, which brings improvements in accuracy. Our code is available at this https URL. 

---
# ReadBench: Measuring the Dense Text Visual Reading Ability of Vision-Language Models 

**Authors**: Benjamin Clavié, Florian Brand  

**Link**: [PDF](https://arxiv.org/pdf/2505.19091)  

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs), have greatly enhanced their capability to jointly process text and images. However, despite extensive benchmarks evaluating visual comprehension (e.g., diagrams, color schemes, OCR tasks...), there is limited assessment of VLMs' ability to read and reason about text-rich images effectively. To fill this gap, we introduce ReadBench, a multimodal benchmark specifically designed to evaluate the reading comprehension capabilities of VLMs. ReadBench transposes contexts from established text-only benchmarks into images of text while keeping textual prompts and questions intact. Evaluating leading VLMs with ReadBench, we find minimal-but-present performance degradation on short, text-image inputs, while performance sharply declines for longer, multi-page contexts. Our experiments further reveal that text resolution has negligible effects on multimodal performance. These findings highlight needed improvements in VLMs, particularly their reasoning over visually presented extensive textual content, a capability critical for practical applications. ReadBench is available at this https URL . 

---
# MaskedManipulator: Versatile Whole-Body Control for Loco-Manipulation 

**Authors**: Chen Tessler, Yifeng Jiang, Erwin Coumans, Zhengyi Luo, Gal Chechik, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19086)  

**Abstract**: Humans interact with their world while leveraging precise full-body control to achieve versatile goals. This versatility allows them to solve long-horizon, underspecified problems, such as placing a cup in a sink, by seamlessly sequencing actions like approaching the cup, grasping, transporting it, and finally placing it in the sink. Such goal-driven control can enable new procedural tools for animation systems, enabling users to define partial objectives while the system naturally ``fills in'' the intermediate motions. However, while current methods for whole-body dexterous manipulation in physics-based animation achieve success in specific interaction tasks, they typically employ control paradigms (e.g., detailed kinematic motion tracking, continuous object trajectory following, or direct VR teleoperation) that offer limited versatility for high-level goal specification across the entire coupled human-object system. To bridge this gap, we present MaskedManipulator, a unified and generative policy developed through a two-stage learning approach. First, our system trains a tracking controller to physically reconstruct complex human-object interactions from large-scale human mocap datasets. This tracking controller is then distilled into MaskedManipulator, which provides users with intuitive control over both the character's body and the manipulated object. As a result, MaskedManipulator enables users to specify complex loco-manipulation tasks through intuitive high-level objectives (e.g., target object poses, key character stances), and MaskedManipulator then synthesizes the necessary full-body actions for a physically simulated humanoid to achieve these goals, paving the way for more interactive and life-like virtual characters. 

---
# Jodi: Unification of Visual Generation and Understanding via Joint Modeling 

**Authors**: Yifeng Xu, Zhenliang He, Meina Kan, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19084)  

**Abstract**: Visual generation and understanding are two deeply interconnected aspects of human intelligence, yet they have been traditionally treated as separate tasks in machine learning. In this paper, we propose Jodi, a diffusion framework that unifies visual generation and understanding by jointly modeling the image domain and multiple label domains. Specifically, Jodi is built upon a linear diffusion transformer along with a role switch mechanism, which enables it to perform three particular types of tasks: (1) joint generation, where the model simultaneously generates images and multiple labels; (2) controllable generation, where images are generated conditioned on any combination of labels; and (3) image perception, where multiple labels can be predicted at once from a given image. Furthermore, we present the Joint-1.6M dataset, which contains 200,000 high-quality images collected from public sources, automatic labels for 7 visual domains, and LLM-generated captions. Extensive experiments demonstrate that Jodi excels in both generation and understanding tasks and exhibits strong extensibility to a wider range of visual domains. Code is available at this https URL. 

---
# An Initial Exploration of Fine-tuning Small Language Models for Smart Contract Reentrancy Vulnerability Detection 

**Authors**: Ignacio Mariano Andreozzi Pofcher, Joshua Ellul  

**Link**: [PDF](https://arxiv.org/pdf/2505.19059)  

**Abstract**: Large Language Models (LLMs) are being used more and more for various coding tasks, including to help coders identify bugs and are a promising avenue to support coders in various tasks including vulnerability detection -- particularly given the flexibility of such generative AI models and tools. Yet for many tasks it may not be suitable to use LLMs, for which it may be more suitable to use smaller language models that can fit and easily execute and train on a developer's computer. In this paper we explore and evaluate whether smaller language models can be fine-tuned to achieve reasonable results for a niche area: vulnerability detection -- specifically focusing on detecting the reentrancy bug in Solidity smart contracts. 

---
# An Embarrassingly Simple Defense Against LLM Abliteration Attacks 

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, Bernard Ghanem, George Turkiyyah  

**Link**: [PDF](https://arxiv.org/pdf/2505.19056)  

**Abstract**: Large language models (LLMs) are typically aligned to comply with safety guidelines by refusing harmful instructions. A recent attack, termed abliteration, isolates and suppresses the single latent direction most responsible for refusal behavior, enabling the model to generate unethical content. We propose a defense that modifies how models generate refusals. We construct an extended-refusal dataset that contains harmful prompts with a full response that justifies the reason for refusal. We then fine-tune Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on our extended-refusal dataset, and evaluate the resulting systems on a set of harmful prompts. In our experiments, extended-refusal models maintain high refusal rates, dropping at most by 10%, whereas baseline models' refusal rates drop by 70-80% after abliteration. A broad evaluation of safety and utility shows that extended-refusal fine-tuning neutralizes the abliteration attack while preserving general performance. 

---
# Smart Waste Management System for Makkah City using Artificial Intelligence and Internet of Things 

**Authors**: Rawabi S. Al Qurashi, Maram M. Almnjomi, Teef L. Alghamdi, Amjad H. Almalki, Shahad S. Alharthi, Shahad M. althobuti, Alanoud S. Alharthi, Maha A. Thafar  

**Link**: [PDF](https://arxiv.org/pdf/2505.19040)  

**Abstract**: Waste management is a critical global issue with significant environmental and public health implications. It has become more destructive during large-scale events such as the annual pilgrimage to Makkah, Saudi Arabia, one of the world's largest religious gatherings. This event's popularity has attracted millions worldwide, leading to significant and un-predictable accumulation of waste. Such a tremendous number of visitors leads to in-creased waste management issues at the Grand Mosque and other holy sites, highlighting the need for an effective solution other than traditional methods based on rigid collection schedules.
To address this challenge, this research proposed an innovative solution that is context-specific and tailored to the unique requirements of pilgrimage season: a Smart Waste Management System, called TUHR, that utilizes the Internet of Things and Artificial Intelligence. This system encompasses ultrasonic sensors that monitor waste levels in each container at the performance sites. Once the container reaches full capacity, the sensor communicates with the microcontroller, which alerts the relevant authorities. Moreover, our system can detect harmful substances such as gas from the gas detector sensor. Such a proactive and dynamic approach promises to mitigate the environmental and health risks associated with waste accumulation and enhance the cleanliness of these sites. It also delivers economic benefits by reducing unnecessary gasoline consumption and optimizing waste management resources. Importantly, this research aligns with the principles of smart cities and exemplifies the innovative, sustainable, and health-conscious approach that Saudi Arabia is implementing as part of its Vision 2030 initiative. 

---
# Turb-L1: Achieving Long-term Turbulence Tracing By Tackling Spectral Bias 

**Authors**: Hao Wu, Yuan Gao, Ruiqi Shu, Zean Han, Fan Xu, Zhihong Zhu, Qingsong Wen, Xian Wu, Kun Wang, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19038)  

**Abstract**: Accurately predicting the long-term evolution of turbulence is crucial for advancing scientific understanding and optimizing engineering applications. However, existing deep learning methods face significant bottlenecks in long-term autoregressive prediction, which exhibit excessive smoothing and fail to accurately track complex fluid dynamics. Our extensive experimental and spectral analysis of prevailing methods provides an interpretable explanation for this shortcoming, identifying Spectral Bias as the core obstacle. Concretely, spectral bias is the inherent tendency of models to favor low-frequency, smooth features while overlooking critical high-frequency details during training, thus reducing fidelity and causing physical distortions in long-term predictions. Building on this insight, we propose Turb-L1, an innovative turbulence prediction method, which utilizes a Hierarchical Dynamics Synthesis mechanism within a multi-grid architecture to explicitly overcome spectral bias. It accurately captures cross-scale interactions and preserves the fidelity of high-frequency dynamics, enabling reliable long-term tracking of turbulence evolution. Extensive experiments on the 2D turbulence benchmark show that Turb-L1 demonstrates excellent performance: (I) In long-term predictions, it reduces Mean Squared Error (MSE) by $80.3\%$ and increases Structural Similarity (SSIM) by over $9\times$ compared to the SOTA baseline, significantly improving prediction fidelity. (II) It effectively overcomes spectral bias, accurately reproducing the full enstrophy spectrum and maintaining physical realism in high-wavenumber regions, thus avoiding the spectral distortions or spurious energy accumulation seen in other methods. 

---
# Medical Large Vision Language Models with Multi-Image Visual Ability 

**Authors**: Xikai Yang, Juzheng Miao, Yuchen Yuan, Jiaze Wang, Qi Dou, Jinpeng Li, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19031)  

**Abstract**: Medical large vision-language models (LVLMs) have demonstrated promising performance across various single-image question answering (QA) benchmarks, yet their capability in processing multi-image clinical scenarios remains underexplored. Unlike single image based tasks, medical tasks involving multiple images often demand sophisticated visual understanding capabilities, such as temporal reasoning and cross-modal analysis, which are poorly supported by current medical LVLMs. To bridge this critical gap, we present the Med-MIM instruction dataset, comprising 83.2K medical multi-image QA pairs that span four types of multi-image visual abilities (temporal understanding, reasoning, comparison, co-reference). Using this dataset, we fine-tune Mantis and LLaVA-Med, resulting in two specialized medical VLMs: MIM-LLaVA-Med and Med-Mantis, both optimized for multi-image analysis. Additionally, we develop the Med-MIM benchmark to comprehensively evaluate the medical multi-image understanding capabilities of LVLMs. We assess eight popular LVLMs, including our two models, on the Med-MIM benchmark. Experimental results show that both Med-Mantis and MIM-LLaVA-Med achieve superior performance on the held-in and held-out subsets of the Med-MIM benchmark, demonstrating that the Med-MIM instruction dataset effectively enhances LVLMs' multi-image understanding capabilities in the medical domain. 

---
# InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts 

**Authors**: Minzhi Lin, Tianchi Xie, Mengchen Liu, Yilin Ye, Changjian Chen, Shixia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19028)  

**Abstract**: Understanding infographic charts with design-driven visual elements (e.g., pictograms, icons) requires both visual recognition and reasoning, posing challenges for multimodal large language models (MLLMs). However, existing visual-question answering benchmarks fall short in evaluating these capabilities of MLLMs due to the lack of paired plain charts and visual-element-based questions. To bridge this gap, we introduce InfoChartQA, a benchmark for evaluating MLLMs on infographic chart understanding. It includes 5,642 pairs of infographic and plain charts, each sharing the same underlying data but differing in visual presentations. We further design visual-element-based questions to capture their unique visual designs and communicative intent. Evaluation of 20 MLLMs reveals a substantial performance decline on infographic charts, particularly for visual-element-based questions related to metaphors. The paired infographic and plain charts enable fine-grained error analysis and ablation studies, which highlight new opportunities for advancing MLLMs in infographic chart understanding. We release InfoChartQA at this https URL. 

---
# A Smart Healthcare System for Monkeypox Skin Lesion Detection and Tracking 

**Authors**: Huda Alghoraibi, Nuha Alqurashi, Sarah Alotaibi, Renad Alkhudaydi, Bdoor Aldajani, Lubna Alqurashi, Jood Batweel, Maha A. Thafar  

**Link**: [PDF](https://arxiv.org/pdf/2505.19023)  

**Abstract**: Monkeypox is a viral disease characterized by distinctive skin lesions and has been reported in many countries. The recent global outbreak has emphasized the urgent need for scalable, accessible, and accurate diagnostic solutions to support public health responses.
In this study, we developed ITMAINN, an intelligent, AI-driven healthcare system specifically designed to detect Monkeypox from skin lesion images using advanced deep learning techniques. Our system consists of three main components. First, we trained and evaluated several pretrained models using transfer learning on publicly available skin lesion datasets to identify the most effective models. For binary classification (Monkeypox vs. non-Monkeypox), the Vision Transformer, MobileViT, Transformer-in-Transformer, and VGG16 achieved the highest performance, each with an accuracy and F1-score of 97.8%. For multiclass classification, which contains images of patients with Monkeypox and five other classes (chickenpox, measles, hand-foot-mouth disease, cowpox, and healthy), ResNetViT and ViT Hybrid models achieved 92% accuracy, with F1 scores of 92.24% and 92.19%, respectively. The best-performing and most lightweight model, MobileViT, was deployed within the mobile application. The second component is a cross-platform smartphone application that enables users to detect Monkeypox through image analysis, track symptoms, and receive recommendations for nearby healthcare centers based on their location. The third component is a real-time monitoring dashboard designed for health authorities to support them in tracking cases, analyzing symptom trends, guiding public health interventions, and taking proactive measures.
This system is fundamental in developing responsive healthcare infrastructure within smart cities. Our solution, ITMAINN, is part of revolutionizing public health management. 

---
# Rethinking Metrics and Benchmarks of Video Anomaly Detection 

**Authors**: Zihao Liu, Xiaoyu Wu, Wenna Li, Linlin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19022)  

**Abstract**: Video Anomaly Detection (VAD), which aims to detect anomalies that deviate from expectation, has attracted increasing attention in recent years. Existing advancements in VAD primarily focus on model architectures and training strategies, while devoting insufficient attention to evaluation metrics and benchmarks. In this paper, we rethink VAD evaluation protocols through comprehensive experimental analyses, revealing three critical limitations in current practices: 1) existing metrics are significantly influenced by single annotation bias; 2) current metrics fail to reward early detection of anomalies; 3) available benchmarks lack the capability to evaluate scene overfitting. To address these limitations, we propose three novel evaluation methods: first, we establish averaged AUC/AP metrics over multi-round annotations to mitigate single annotation bias; second, we develop a Latency-aware Average Precision (LaAP) metric that rewards early and accurate anomaly detection; and finally, we introduce two hard normal benchmarks (UCF-HN, MSAD-HN) with videos specifically designed to evaluate scene overfitting. We report performance comparisons of ten state-of-the-art VAD approaches using our proposed evaluation methods, providing novel perspectives for future VAD model development. 

---
# HGCL: Hierarchical Graph Contrastive Learning for User-Item Recommendation 

**Authors**: Jiawei Xue, Zhen Yang, Haitao Lin, Ziji Zhang, Luzhu Wang, Yikun Gu, Yao Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19020)  

**Abstract**: Graph Contrastive Learning (GCL), which fuses graph neural networks with contrastive learning, has evolved as a pivotal tool in user-item recommendations. While promising, existing GCL methods often lack explicit modeling of hierarchical item structures, which represent item similarities across varying resolutions. Such hierarchical item structures are ubiquitous in various items (e.g., online products and local businesses), and reflect their inherent organizational properties that serve as critical signals for enhancing recommendation accuracy. In this paper, we propose Hierarchical Graph Contrastive Learning (HGCL), a novel GCL method that incorporates hierarchical item structures for user-item recommendations. First, HGCL pre-trains a GCL module using cross-layer contrastive learning to obtain user and item representations. Second, HGCL employs a representation compression and clustering method to construct a two-hierarchy user-item bipartite graph. Ultimately, HGCL fine-tunes user and item representations by learning on the hierarchical graph, and then provides recommendations based on user-item interaction scores. Experiments on three widely adopted benchmark datasets ranging from 70K to 382K nodes confirm the superior performance of HGCL over existing baseline models, highlighting the contribution of hierarchical item structures in enhancing GCL methods for recommendation tasks. 

---
# Faithful Group Shapley Value 

**Authors**: Kiljae Lee, Ziqi Liu, Weijing Tang, Yuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19013)  

**Abstract**: Data Shapley is an important tool for data valuation, which quantifies the contribution of individual data points to machine learning models. In practice, group-level data valuation is desirable when data providers contribute data in batch. However, we identify that existing group-level extensions of Data Shapley are vulnerable to shell company attacks, where strategic group splitting can unfairly inflate valuations. We propose Faithful Group Shapley Value (FGSV) that uniquely defends against such attacks. Building on original mathematical insights, we develop a provably fast and accurate approximation algorithm for computing FGSV. Empirical experiments demonstrate that our algorithm significantly outperforms state-of-the-art methods in computational efficiency and approximation accuracy, while ensuring faithful group-level valuation. 

---
# Semi-pessimistic Reinforcement Learning 

**Authors**: Jin Zhu, Xin Zhou, Jiaang Yao, Gholamali Aminian, Omar Rivasplata, Simon Little, Lexin Li, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19002)  

**Abstract**: Offline reinforcement learning (RL) aims to learn an optimal policy from pre-collected data. However, it faces challenges of distributional shift, where the learned policy may encounter unseen scenarios not covered in the offline data. Additionally, numerous applications suffer from a scarcity of labeled reward data. Relying on labeled data alone often leads to a narrow state-action distribution, further amplifying the distributional shift, and resulting in suboptimal policy learning. To address these issues, we first recognize that the volume of unlabeled data is typically substantially larger than that of labeled data. We then propose a semi-pessimistic RL method to effectively leverage abundant unlabeled data. Our approach offers several advantages. It considerably simplifies the learning process, as it seeks a lower bound of the reward function, rather than that of the Q-function or state transition function. It is highly flexible, and can be integrated with a range of model-free and model-based RL algorithms. It enjoys the guaranteed improvement when utilizing vast unlabeled data, but requires much less restrictive conditions. We compare our method with a number of alternative solutions, both analytically and numerically, and demonstrate its clear competitiveness. We further illustrate with an application to adaptive deep brain stimulation for Parkinson's disease. 

---
# FiLLM -- A Filipino-optimized Large Language Model based on Southeast Asia Large Language Model (SEALLM) 

**Authors**: Carlos Jude G. Maminta, Isaiah Job Enriquez, Deandre Nigel Nunez, Michael B. Dela Fuente  

**Link**: [PDF](https://arxiv.org/pdf/2505.18995)  

**Abstract**: This study presents FiLLM, a Filipino-optimized large language model, designed to enhance natural language processing (NLP) capabilities in the Filipino language. Built upon the SeaLLM-7B 2.5 model, FiLLM leverages Low-Rank Adaptation (LoRA) fine-tuning to optimize memory efficiency while maintaining task-specific performance. The model was trained and evaluated on diverse Filipino datasets to address key NLP tasks, including Named Entity Recognition (NER), Part-of-Speech (POS) tagging, Dependency Parsing, and Text Summarization. Performance comparisons with the CalamanCy model were conducted using F1 Score, Precision, Recall, Compression Rate, and Keyword Overlap metrics. Results indicate that Calamancy outperforms FILLM in several aspects, demonstrating its effectiveness in processing Filipino text with improved linguistic comprehension and adaptability. This research contributes to the advancement of Filipino NLP applications by providing an optimized, efficient, and scalable language model tailored for local linguistic needs. 

---
# GraSS: Scalable Influence Function with Sparse Gradient Compression 

**Authors**: Pingbang Hu, Joseph Melkonian, Weijing Tang, Han Zhao, Jiaqi W. Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.18976)  

**Abstract**: Gradient-based data attribution methods, such as influence functions, are critical for understanding the impact of individual training samples without requiring repeated model retraining. However, their scalability is often limited by the high computational and memory costs associated with per-sample gradient computation. In this work, we propose GraSS, a novel gradient compression algorithm and its variants FactGraSS for linear layers specifically, that explicitly leverage the inherent sparsity of per-sample gradients to achieve sub-linear space and time complexity. Extensive experiments demonstrate the effectiveness of our approach, achieving substantial speedups while preserving data influence fidelity. In particular, FactGraSS achieves up to 165% faster throughput on billion-scale models compared to the previous state-of-the-art baselines. Our code is publicly available at this https URL. 

---
# FastMamba: A High-Speed and Efficient Mamba Accelerator on FPGA with Accurate Quantization 

**Authors**: Aotao Wang, Haikuo Shao, Shaobo Ma, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18975)  

**Abstract**: State Space Models (SSMs), like recent Mamba2, have achieved remarkable performance and received extensive attention. However, deploying Mamba2 on resource-constrained edge devices encounters many problems: severe outliers within the linear layer challenging the quantization, diverse and irregular element-wise tensor operations, and hardware-unfriendly nonlinear functions in the SSM block. To address these issues, this paper presents FastMamba, a dedicated accelerator on FPGA with hardware-algorithm co-design to promote the deployment efficiency of Mamba2. Specifically, we successfully achieve 8-bit quantization for linear layers through Hadamard transformation to eliminate outliers. Moreover, a hardware-friendly and fine-grained power-of-two quantization framework is presented for the SSM block and convolution layer, and a first-order linear approximation is developed to optimize the nonlinear functions. Based on the accurate algorithm quantization, we propose an accelerator that integrates parallel vector processing units, pipelined execution dataflow, and an efficient SSM Nonlinear Approximation Unit, which enhances computational efficiency and reduces hardware complexity. Finally, we evaluate FastMamba on Xilinx VC709 FPGA. For the input prefill task on Mamba2-130M, FastMamba achieves 68.80\times and 8.90\times speedup over Intel Xeon 4210R CPU and NVIDIA RTX 3090 GPU, respectively. In the output decode experiment with Mamba2-2.7B, FastMamba attains 6\times higher energy efficiency than RTX 3090 GPU. 

---
# Revival with Voice: Multi-modal Controllable Text-to-Speech Synthesis 

**Authors**: Minsu Kim, Pingchuan Ma, Honglie Chen, Stavros Petridis, Maja Pantic  

**Link**: [PDF](https://arxiv.org/pdf/2505.18972)  

**Abstract**: This paper explores multi-modal controllable Text-to-Speech Synthesis (TTS) where the voice can be generated from face image, and the characteristics of output speech (e.g., pace, noise level, distance, tone, place) can be controllable with natural text description. Specifically, we aim to mitigate the following three challenges in face-driven TTS systems. 1) To overcome the limited audio quality of audio-visual speech corpora, we propose a training method that additionally utilizes high-quality audio-only speech corpora. 2) To generate voices not only from real human faces but also from artistic portraits, we propose augmenting the input face image with stylization. 3) To consider one-to-many possibilities in face-to-voice mapping and ensure consistent voice generation at the same time, we propose to first employ sampling-based decoding and then use prompting with generated speech samples. Experimental results validate the proposed model's effectiveness in face-driven voice synthesis. 

---
# Protein Design with Dynamic Protein Vocabulary 

**Authors**: Nuowei Liu, Jiahao Kuang, Yanting Liu, Changzhi Sun, Tao Ji, Yuanbin Wu, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18966)  

**Abstract**: Protein design is a fundamental challenge in biotechnology, aiming to design novel sequences with specific functions within the vast space of possible proteins. Recent advances in deep generative models have enabled function-based protein design from textual descriptions, yet struggle with structural plausibility. Inspired by classical protein design methods that leverage natural protein structures, we explore whether incorporating fragments from natural proteins can enhance foldability in generative models. Our empirical results show that even random incorporation of fragments improves foldability. Building on this insight, we introduce ProDVa, a novel protein design approach that integrates a text encoder for functional descriptions, a protein language model for designing proteins, and a fragment encoder to dynamically retrieve protein fragments based on textual functional descriptions. Experimental results demonstrate that our approach effectively designs protein sequences that are both functionally aligned and structurally plausible. Compared to state-of-the-art models, ProDVa achieves comparable function alignment using less than 0.04% of the training data, while designing significantly more well-folded proteins, with the proportion of proteins having pLDDT above 70 increasing by 7.38% and those with PAE below 10 increasing by 9.6%. 

---
# How Do Images Align and Complement LiDAR? Towards a Harmonized Multi-modal 3D Panoptic Segmentation 

**Authors**: Yining Pan, Qiongjie Cui, Xulei Yang, Na Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18956)  

**Abstract**: LiDAR-based 3D panoptic segmentation often struggles with the inherent sparsity of data from LiDAR sensors, which makes it challenging to accurately recognize distant or small objects. Recently, a few studies have sought to overcome this challenge by integrating LiDAR inputs with camera images, leveraging the rich and dense texture information provided by the latter. While these approaches have shown promising results, they still face challenges, such as misalignment during data augmentation and the reliance on post-processing steps. To address these issues, we propose Image-Assists-LiDAR (IAL), a novel multi-modal 3D panoptic segmentation framework. In IAL, we first introduce a modality-synchronized data augmentation strategy, PieAug, to ensure alignment between LiDAR and image inputs from the start. Next, we adopt a transformer decoder to directly predict panoptic segmentation results. To effectively fuse LiDAR and image features into tokens for the decoder, we design a Geometric-guided Token Fusion (GTF) module. Additionally, we leverage the complementary strengths of each modality as priors for query initialization through a Prior-based Query Generation (PQG) module, enhancing the decoder's ability to generate accurate instance masks. Our IAL framework achieves state-of-the-art performance compared to previous multi-modal 3D panoptic segmentation methods on two widely used benchmarks. Code and models are publicly available at <this https URL. 

---
# The Price of Format: Diversity Collapse in LLMs 

**Authors**: Longfei Yun, Chenyang An, Zilong Wang, Letian Peng, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18949)  

**Abstract**: Instruction-tuned large language models (LLMs) employ structured templates, such as role markers and special tokens, to enforce format consistency during inference. However, we identify a critical limitation of such formatting: it induces a phenomenon we term diversity collapse, where the model generates semantically similar outputs for open-ended inputs, undermining creativity and variability. We systematically evaluate this effect across tasks like story completion and free-form generation, finding that (1) diversity collapse persists even under high-temperature sampling, and (2) structural tokens in templates significantly constrain the model's output space. To contextualize these findings, we fine-tune the same model using a range of structured prompts and then evaluate them across three axes: downstream task performance, alignment behavior, and output diversity. Our analysis shows that format consistency between fine-tuning and inference is crucial for structure-sensitive tasks (e.g., GSM8K, IFEval), but has marginal influence on knowledge-heavy tasks (e.g., MMLU, WebQuestions). In contrast, output diversity is primarily governed by the presence or absence of structural tokens, with minimal formatting yielding the most diverse outputs. These findings reveal that current prompting conventions, while beneficial for alignment, may inadvertently suppress output diversity, underscoring the need for diversity-aware prompt design and instruction tuning. 

---
# Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection 

**Authors**: Xiping Li, Xiangyu Dong, Xingyi Zhang, Kun Xie, Yuanhao Feng, Bo Wang, Guilin Li, Wuxiong Zeng, Xiujun Shu, Sibo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18934)  

**Abstract**: Graph Anomaly Detection (GAD) in heterogeneous networks presents unique challenges due to node and edge heterogeneity. Existing Graph Neural Network (GNN) methods primarily focus on homogeneous GAD and thus fail to address three key issues: (C1) Capturing abnormal signal and rich semantics across diverse meta-paths; (C2) Retaining high-frequency content in HIN dimension alignment; and (C3) Learning effectively from difficult anomaly samples with class imbalance. To overcome these, we propose ChiGAD, a spectral GNN framework based on a novel Chi-Square filter, inspired by the wavelet effectiveness in diverse domains. Specifically, ChiGAD consists of: (1) Multi-Graph Chi-Square Filter, which captures anomalous information via applying dedicated Chi-Square filters to each meta-path graph; (2) Interactive Meta-Graph Convolution, which aligns features while preserving high-frequency information and incorporates heterogeneous messages by a unified Chi-Square Filter; and (3) Contribution-Informed Cross-Entropy Loss, which prioritizes difficult anomalies to address class imbalance. Extensive experiments on public and industrial datasets show that ChiGAD outperforms state-of-the-art models on multiple metrics. Additionally, its homogeneous variant, ChiGNN, excels on seven GAD datasets, validating the effectiveness of Chi-Square filters. Our code is available at this https URL. 

---
# WeedNet: A Foundation Model-Based Global-to-Local AI Approach for Real-Time Weed Species Identification and Classification 

**Authors**: Yanben Shen, Timilehin T. Ayanlade, Venkata Naresh Boddepalli, Mojdeh Saadati, Ashlyn Rairdin, Zi K. Deng, Muhammad Arbab Arshad, Aditya Balu, Daren Mueller, Asheesh K Singh, Wesley Everman, Nirav Merchant, Baskar Ganapathysubramanian, Meaghan Anderson, Soumik Sarkar, Arti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.18930)  

**Abstract**: Early identification of weeds is essential for effective management and control, and there is growing interest in automating the process using computer vision techniques coupled with AI methods. However, challenges associated with training AI-based weed identification models, such as limited expert-verified data and complexity and variability in morphological features, have hindered progress. To address these issues, we present WeedNet, the first global-scale weed identification model capable of recognizing an extensive set of weed species, including noxious and invasive plant species. WeedNet is an end-to-end real-time weed identification pipeline and uses self-supervised learning, fine-tuning, and enhanced trustworthiness strategies. WeedNet achieved 91.02% accuracy across 1,593 weed species, with 41% species achieving 100% accuracy. Using a fine-tuning strategy and a Global-to-Local approach, the local Iowa WeedNet model achieved an overall accuracy of 97.38% for 85 Iowa weeds, most classes exceeded a 90% mean accuracy per class. Testing across intra-species dissimilarity (developmental stages) and inter-species similarity (look-alike species) suggests that diversity in the images collected, spanning all the growth stages and distinguishable plant characteristics, is crucial in driving model performance. The generalizability and adaptability of the Global WeedNet model enable it to function as a foundational model, with the Global-to-Local strategy allowing fine-tuning for region-specific weed communities. Additional validation of drone- and ground-rover-based images highlights the potential of WeedNet for integration into robotic platforms. Furthermore, integration with AI for conversational use provides intelligent agricultural and ecological conservation consulting tools for farmers, agronomists, researchers, land managers, and government agencies across diverse landscapes. 

---
# Benchmarking Large Language Models for Cyberbullying Detection in Real-World YouTube Comments 

**Authors**: Amel Muminovic  

**Link**: [PDF](https://arxiv.org/pdf/2505.18927)  

**Abstract**: As online platforms grow, comment sections increasingly host harassment that undermines user experience and well-being. This study benchmarks three leading large language models, OpenAI GPT-4.1, Google Gemini 1.5 Pro, and Anthropic Claude 3 Opus, on a corpus of 5,080 YouTube comments sampled from high-abuse threads in gaming, lifestyle, food vlog, and music channels. The dataset comprises 1,334 harmful and 3,746 non-harmful messages in English, Arabic, and Indonesian, annotated independently by two reviewers with substantial agreement (Cohen's kappa = 0.83). Using a unified prompt and deterministic settings, GPT-4.1 achieved the best overall balance with an F1 score of 0.863, precision of 0.887, and recall of 0.841. Gemini flagged the highest share of harmful posts (recall = 0.875) but its precision fell to 0.767 due to frequent false positives. Claude delivered the highest precision at 0.920 and the lowest false-positive rate of 0.022, yet its recall dropped to 0.720. Qualitative analysis showed that all three models struggle with sarcasm, coded insults, and mixed-language slang. These results underscore the need for moderation pipelines that combine complementary models, incorporate conversational context, and fine-tune for under-represented languages and implicit abuse. A de-identified version of the dataset and full prompts is publicly released to promote reproducibility and further progress in automated content moderation. 

---
# Behavior Injection: Preparing Language Models for Reinforcement Learning 

**Authors**: Zhepeng Cen, Yihang Yao, William Han, Zuxin Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18917)  

**Abstract**: Reinforcement fine-tuning (RFT) has emerged as a powerful post-training technique to incentivize the reasoning ability of large language models (LLMs). However, LLMs can respond very inconsistently to RFT: some show substantial performance gains, while others plateau or even degrade. To understand this divergence, we analyze the per-step influence of the RL objective and identify two key conditions for effective post-training: (1) RL-informative rollout accuracy, and (2) strong data co-influence, which quantifies how much the training data affects performance on other samples. Guided by these insights, we propose behavior injection, a task-agnostic data-augmentation scheme applied prior to RL. Behavior injection enriches the supervised finetuning (SFT) data by seeding exploratory and exploitative behaviors, effectively making the model more RL-ready. We evaluate our method across two reasoning benchmarks with multiple base models. The results demonstrate that our theoretically motivated augmentation can significantly increases the performance gain from RFT over the pre-RL model. 

---
# Robust Stability Analysis of Positive Lure System with Neural Network Feedback 

**Authors**: Hamidreza Montazeri Hedesh, Moh. Kamalul Wafi, Bahram Shafai, Milad Siami  

**Link**: [PDF](https://arxiv.org/pdf/2505.18912)  

**Abstract**: This paper investigates the robustness of the Lur'e problem under positivity constraints, drawing on results from the positive Aizerman conjecture and the robustness properties of Metzler matrices. Specifically, we consider a control system of Lur'e type in which not only the linear part includes parametric uncertainty but also the nonlinear sector bound is unknown. We investigate tools from positive linear systems to effectively solve the problems in complicated and uncertain nonlinear systems. By leveraging the positivity characteristic of the system, we derive an explicit formula for the stability radius of Lur'e systems. Furthermore, we extend our analysis to systems with neural network (NN) feedback loops. Building on this approach, we also propose a refinement method for sector bounds of feedforward neural networks (FFNNs). This study introduces a scalable and efficient approach for robustness analysis of both Lur'e and NN-controlled systems. Finally, the proposed results are supported by illustrative examples. 

---
# PromptWise: Online Learning for Cost-Aware Prompt Assignment in Generative Models 

**Authors**: Xiaoyan Hu, Lauren Pick, Ho-fung Leung, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2505.18901)  

**Abstract**: The rapid advancement of generative AI models has provided users with numerous options to address their prompts. When selecting a generative AI model for a given prompt, users should consider not only the performance of the chosen model but also its associated service cost. The principle guiding such consideration is to select the least expensive model among the available satisfactory options. However, existing model-selection approaches typically prioritize performance, overlooking pricing differences between models. In this paper, we introduce PromptWise, an online learning framework designed to assign a sequence of prompts to a group of large language models (LLMs) in a cost-effective manner. PromptWise strategically queries cheaper models first, progressing to more expensive options only if the lower-cost models fail to adequately address a given prompt. Through numerical experiments, we demonstrate PromptWise's effectiveness across various tasks, including puzzles of varying complexity and code generation/translation tasks. The results highlight that PromptWise consistently outperforms cost-unaware baseline methods, emphasizing that directly assigning prompts to the most expensive models can lead to higher costs and potentially lower average performance. 

---
# Improving Ad matching via Cluster-Adaptive Keyword Expansion and Relevance tuning 

**Authors**: Dipanwita Saha, Anis Zaman, Hua Zou, Ning Chen, Xinxin Shu, Nadia Vase, Abraham Bagherjeiran  

**Link**: [PDF](https://arxiv.org/pdf/2505.18897)  

**Abstract**: In search advertising, keyword matching connects user queries with relevant ads. While token-based matching increases ad coverage, it can reduce relevance due to overly permissive semantic expansion. This work extends keyword reach through document-side semantic keyword expansion, using a language model to broaden token-level matching without altering queries. We propose a solution using a pre-trained siamese model to generate dense vector representations of ad keywords and identify semantically related variants through nearest neighbor search. To maintain precision, we introduce a cluster-based thresholding mechanism that adjusts similarity cutoffs based on local semantic density. Each expanded keyword maps to a group of seller-listed items, which may only partially align with the original intent. To ensure relevance, we enhance the downstream relevance model by adapting it to the expanded keyword space using an incremental learning strategy with a lightweight decision tree ensemble. This system improves both relevance and click-through rate (CTR), offering a scalable, low-latency solution adaptable to evolving query behavior and advertising inventory. 

---
# Reality Check: A New Evaluation Ecosystem Is Necessary to Understand AI's Real World Effects 

**Authors**: Reva Schwartz, Rumman Chowdhury, Akash Kundu, Heather Frase, Marzieh Fadaee, Tom David, Gabriella Waters, Afaf Taik, Morgan Briggs, Patrick Hall, Shomik Jain, Kyra Yee, Spencer Thomas, Sundeep Bhandari, Lee Wan Sie, Qinghua Lu, Matthew Holmes, Theodora Skeadas  

**Link**: [PDF](https://arxiv.org/pdf/2505.18893)  

**Abstract**: Conventional AI evaluation approaches concentrated within the AI stack exhibit systemic limitations for exploring, navigating and resolving the human and societal factors that play out in real world deployment such as in education, finance, healthcare, and employment sectors. AI capability evaluations can capture detail about first-order effects, such as whether immediate system outputs are accurate, or contain toxic, biased or stereotypical content, but AI's second-order effects, i.e. any long-term outcomes and consequences that may result from AI use in the real world, have become a significant area of interest as the technology becomes embedded in our daily lives. These secondary effects can include shifts in user behavior, societal, cultural and economic ramifications, workforce transformations, and long-term downstream impacts that may result from a broad and growing set of risks. This position paper argues that measuring the indirect and secondary effects of AI will require expansion beyond static, single-turn approaches conducted in silico to include testing paradigms that can capture what actually materializes when people use AI technology in context. Specifically, we describe the need for data and methods that can facilitate contextual awareness and enable downstream interpretation and decision making about AI's secondary effects, and recommend requirements for a new ecosystem. 

---
# Climate Implications of Diffusion-based Generative Visual AI Systems and their Mass Adoption 

**Authors**: Vanessa Utz, Steve DiPaola  

**Link**: [PDF](https://arxiv.org/pdf/2505.18892)  

**Abstract**: Climate implications of rapidly developing digital technologies, such as blockchains and the associated crypto mining and NFT minting, have been well documented and their massive GPU energy use has been identified as a cause for concern. However, we postulate that due to their more mainstream consumer appeal, the GPU use of text-prompt based diffusion AI art systems also requires thoughtful considerations. Given the recent explosion in the number of highly sophisticated generative art systems and their rapid adoption by consumers and creative professionals, the impact of these systems on the climate needs to be carefully considered. In this work, we report on the growth of diffusion-based visual AI systems, their patterns of use, growth and the implications on the climate. Our estimates show that the mass adoption of these tools potentially contributes considerably to global energy consumption. We end this paper with our thoughts on solutions and future areas of inquiry as well as associated difficulties, including the lack of publicly available data. 

---
# Security Concerns for Large Language Models: A Survey 

**Authors**: Miles Q. Li, Benjamin C. M. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2505.18889)  

**Abstract**: Large Language Models (LLMs) such as GPT-4 (and its recent iterations like GPT-4o and the GPT-4.1 series), Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks (including input perturbations and data poisoning), misuse by malicious actors (e.g., for disinformation, phishing, and malware generation), and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives (scheming), which may even persist through safety training. We summarize recent academic and industrial studies (2022-2025) that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial. 

---
# LORE: Lagrangian-Optimized Robust Embeddings for Visual Encoders 

**Authors**: Borna Khodabandeh, Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall, Sajjad Amini, Seyed-Mohsen Moosavi-Dezfooli  

**Link**: [PDF](https://arxiv.org/pdf/2505.18884)  

**Abstract**: Visual encoders have become fundamental components in modern computer vision pipelines. However, ensuring robustness against adversarial perturbations remains a critical challenge. Recent efforts have explored both supervised and unsupervised adversarial fine-tuning strategies. We identify two key limitations in these approaches: (i) they often suffer from instability, especially during the early stages of fine-tuning, resulting in suboptimal convergence and degraded performance on clean data, and (ii) they exhibit a suboptimal trade-off between robustness and clean data accuracy, hindering the simultaneous optimization of both objectives. To overcome these challenges, we propose Lagrangian-Optimized Robust Embeddings (LORE), a novel unsupervised adversarial fine-tuning framework. LORE utilizes constrained optimization, which offers a principled approach to balancing competing goals, such as improving robustness while preserving nominal performance. By enforcing embedding-space proximity constraints, LORE effectively maintains clean data performance throughout adversarial fine-tuning. Extensive experiments show that LORE significantly improves zero-shot adversarial robustness with minimal degradation in clean data accuracy. Furthermore, we demonstrate the effectiveness of the adversarially fine-tuned CLIP image encoder in out-of-distribution generalization and enhancing the interpretability of image embeddings. 

---
# SD-OVON: A Semantics-aware Dataset and Benchmark Generation Pipeline for Open-Vocabulary Object Navigation in Dynamic Scenes 

**Authors**: Dicong Qiu, Jiadi You, Zeying Gong, Ronghe Qiu, Hui Xiong, Junwei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18881)  

**Abstract**: We present the Semantics-aware Dataset and Benchmark Generation Pipeline for Open-vocabulary Object Navigation in Dynamic Scenes (SD-OVON). It utilizes pretraining multimodal foundation models to generate infinite unique photo-realistic scene variants that adhere to real-world semantics and daily commonsense for the training and the evaluation of navigation agents, accompanied with a plugin for generating object navigation task episodes compatible to the Habitat simulator. In addition, we offer two pre-generated object navigation task datasets, SD-OVON-3k and SD-OVON-10k, comprising respectively about 3k and 10k episodes of the open-vocabulary object navigation task, derived from the SD-OVON-Scenes dataset with 2.5k photo-realistic scans of real-world environments and the SD-OVON-Objects dataset with 0.9k manually inspected scanned and artist-created manipulatable object models. Unlike prior datasets limited to static environments, SD-OVON covers dynamic scenes and manipulatable objects, facilitating both real-to-sim and sim-to-real robotic applications. This approach enhances the realism of navigation tasks, the training and the evaluation of open-vocabulary object navigation agents in complex settings. To demonstrate the effectiveness of our pipeline and datasets, we propose two baselines and evaluate them along with state-of-the-art baselines on SD-OVON-3k. The datasets, benchmark and source code are publicly available. 

---
# REGen: Multimodal Retrieval-Embedded Generation for Long-to-Short Video Editing 

**Authors**: Weihan Xu, Yimeng Ma, Jingyue Huang, Yang Li, Wenye Ma, Taylor Berg-Kirkpatrick, Julian McAuley, Paul Pu Liang, Hao-Wen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18880)  

**Abstract**: Short videos are an effective tool for promoting contents and improving knowledge accessibility. While existing extractive video summarization methods struggle to produce a coherent narrative, existing abstractive methods cannot `quote' from the input videos, i.e., inserting short video clips in their outputs. In this work, we explore novel video editing models for generating shorts that feature a coherent narrative with embedded video insertions extracted from a long input video. We propose a novel retrieval-embedded generation framework that allows a large language model to quote multimodal resources while maintaining a coherent narrative. Our proposed REGen system first generates the output story script with quote placeholders using a finetuned large language model, and then uses a novel retrieval model to replace the quote placeholders by selecting a video clip that best supports the narrative from a pool of candidate quotable video clips. We examine the proposed method on the task of documentary teaser generation, where short interview insertions are commonly used to support the narrative of a documentary. Our objective evaluations show that the proposed method can effectively insert short video clips while maintaining a coherent narrative. In a subjective survey, we show that our proposed method outperforms existing abstractive and extractive approaches in terms of coherence, alignment, and realism in teaser generation. 

---
# CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions 

**Authors**: Kung-Hsiang Huang, Akshara Prabhakar, Onkar Thorat, Divyansh Agarwal, Prafulla Kumar Choubey, Yixin Mao, Silvio Savarese, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18878)  

**Abstract**: While AI agents hold transformative potential in business, effective performance benchmarking is hindered by the scarcity of public, realistic business data on widely used platforms. Existing benchmarks often lack fidelity in their environments, data, and agent-user interactions, with limited coverage of diverse business scenarios and industries. To address these gaps, we introduce CRMArena-Pro, a novel benchmark for holistic, realistic assessment of LLM agents in diverse professional settings. CRMArena-Pro expands on CRMArena with nineteen expert-validated tasks across sales, service, and 'configure, price, and quote' processes, for both Business-to-Business and Business-to-Customer scenarios. It distinctively incorporates multi-turn interactions guided by diverse personas and robust confidentiality awareness assessments. Experiments reveal leading LLM agents achieve only around 58% single-turn success on CRMArena-Pro, with performance dropping significantly to approximately 35% in multi-turn settings. While Workflow Execution proves more tractable for top agents (over 83% single-turn success), other evaluated business skills present greater challenges. Furthermore, agents exhibit near-zero inherent confidentiality awareness; though targeted prompting can improve this, it often compromises task performance. These findings highlight a substantial gap between current LLM capabilities and enterprise demands, underscoring the need for advancements in multi-turn reasoning, confidentiality adherence, and versatile skill acquisition. 

---
# Writing Like the Best: Exemplar-Based Expository Text Generation 

**Authors**: Yuxiang Liu, Kevin Chen-Chuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18859)  

**Abstract**: We introduce the Exemplar-Based Expository Text Generation task, aiming to generate an expository text on a new topic using an exemplar on a similar topic. Current methods fall short due to their reliance on extensive exemplar data, difficulty in adapting topic-specific content, and issues with long-text coherence. To address these challenges, we propose the concept of Adaptive Imitation and present a novel Recurrent Plan-then-Adapt (RePA) framework. RePA leverages large language models (LLMs) for effective adaptive imitation through a fine-grained plan-then-adapt process. RePA also enables recurrent segment-by-segment imitation, supported by two memory structures that enhance input clarity and output coherence. We also develop task-specific evaluation metrics--imitativeness, adaptiveness, and adaptive-imitativeness--using LLMs as evaluators. Experimental results across our collected three diverse datasets demonstrate that RePA surpasses existing baselines in producing factual, consistent, and relevant texts for this task. 

---
# High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction 

**Authors**: Seongsu Kim, Nayoung Kim, Dongwoo Kim, Sungsoo Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.18817)  

**Abstract**: Density functional theory (DFT) is a fundamental method for simulating quantum chemical properties, but it remains expensive due to the iterative self-consistent field (SCF) process required to solve the Kohn-Sham equations. Recently, deep learning methods are gaining attention as a way to bypass this step by directly predicting the Hamiltonian. However, they rely on deterministic regression and do not consider the highly structured nature of Hamiltonians. In this work, we propose QHFlow, a high-order equivariant flow matching framework that generates Hamiltonian matrices conditioned on molecular geometry. Flow matching models continuous-time trajectories between simple priors and complex targets, learning the structured distributions over Hamiltonians instead of direct regression. To further incorporate symmetry, we use a neural architecture that predicts SE(3)-equivariant vector fields, improving accuracy and generalization across diverse geometries. To further enhance physical fidelity, we additionally introduce a fine-tuning scheme to align predicted orbital energies with the target. QHFlow achieves state-of-the-art performance, reducing Hamiltonian error by 71% on MD17 and 53% on QH9. Moreover, we further show that QHFlow accelerates the DFT process without trading off the solution quality when initializing SCF iterations with the predicted Hamiltonian, significantly reducing the number of iterations and runtime. 

---
# ALPS: Attention Localization and Pruning Strategy for Efficient Alignment of Large Language Models 

**Authors**: Hao Chen, Haoze Li, Zhiqing Xiao, Lirong Gao, Qi Zhang, Xiaomeng Hu, Ningtao Wang, Xing Fu, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18799)  

**Abstract**: Aligning general-purpose large language models (LLMs) to downstream tasks often incurs significant costs, including constructing task-specific instruction pairs and extensive training adjustments. Prior research has explored various avenues to enhance alignment efficiency, primarily through minimal-data training or data-driven activations to identify key attention heads. However, these approaches inherently introduce data dependency, which hinders generalization and reusability. To address this issue and enhance model alignment efficiency, we propose the \textit{\textbf{A}ttention \textbf{L}ocalization and \textbf{P}runing \textbf{S}trategy (\textbf{ALPS})}, an efficient algorithm that localizes the most task-sensitive attention heads and prunes by restricting attention training updates to these heads, thereby reducing alignment costs. Experimental results demonstrate that our method activates only \textbf{10\%} of attention parameters during fine-tuning while achieving a \textbf{2\%} performance improvement over baselines on three tasks. Moreover, the identified task-specific heads are transferable across datasets and mitigate knowledge forgetting. Our work and findings provide a novel perspective on efficient LLM alignment. 

---
# Think Twice before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation 

**Authors**: Hong-Hanh Nguyen-Le, Van-Tuan Tran, Dinh-Thuc Nguyen, Nhien-An Le-Khac  

**Link**: [PDF](https://arxiv.org/pdf/2505.18787)  

**Abstract**: Deepfake (DF) detectors face significant challenges when deployed in real-world environments, particularly when encountering test samples deviated from training data through either postprocessing manipulations or distribution shifts. We demonstrate postprocessing techniques can completely obscure generation artifacts presented in DF samples, leading to performance degradation of DF detectors. To address these challenges, we propose Think Twice before Adaptation (\texttt{T$^2$A}), a novel online test-time adaptation method that enhances the adaptability of detectors during inference without requiring access to source training data or labels. Our key idea is to enable the model to explore alternative options through an Uncertainty-aware Negative Learning objective rather than solely relying on its initial predictions as commonly seen in entropy minimization (EM)-based approaches. We also introduce an Uncertain Sample Prioritization strategy and Gradients Masking technique to improve the adaptation by focusing on important samples and model parameters. Our theoretical analysis demonstrates that the proposed negative learning objective exhibits complementary behavior to EM, facilitating better adaptation capability. Empirically, our method achieves state-of-the-art results compared to existing test-time adaptation (TTA) approaches and significantly enhances the resilience and generalization of DF detectors during inference. Code is available \href{this https URL}{here}. 

---
# Soft Weighted Machine Unlearning 

**Authors**: Xinbao Qiao, Ningning Ding, Yushi Cheng, Meng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18783)  

**Abstract**: Machine unlearning, as a post-hoc processing technique, has gained widespread adoption in addressing challenges like bias mitigation and robustness enhancement, colloquially, machine unlearning for fairness and robustness. However, existing non-privacy unlearning-based solutions persist in using binary data removal framework designed for privacy-driven motivation, leading to significant information loss, a phenomenon known as over-unlearning. While over-unlearning has been largely described in many studies as primarily causing utility degradation, we investigate its fundamental causes and provide deeper insights in this work through counterfactual leave-one-out analysis. In this paper, we introduce a weighted influence function that assigns tailored weights to each sample by solving a convex quadratic programming problem analytically. Building on this, we propose a soft-weighted framework enabling fine-grained model adjustments to address the over-unlearning challenge. We demonstrate that the proposed soft-weighted scheme is versatile and can be seamlessly integrated into most existing unlearning algorithms. Extensive experiments show that in fairness- and robustness-driven tasks, the soft-weighted scheme significantly outperforms hard-weighted schemes in fairness/robustness metrics and alleviates the decline in utility metric, thereby enhancing machine unlearning algorithm as an effective correction solution. 

---
# HD-PiSSA: High-Rank Distributed Orthogonal Adaptation 

**Authors**: Yiding Wang, Fauxu meng, Xuefeng Zhang, Fan Jiang, Pingzhi Tang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18777)  

**Abstract**: Existing parameter-efficient fine-tuning (PEFT) methods for large language models (LLMs), such as LoRA and PiSSA, constrain model updates to low-rank subspaces, limiting their expressiveness and leading to suboptimal performance on complex tasks. To address this, we introduce High-rank Distributed PiSSA (HD-PiSSA), a distributed PEFT approach that initializes orthogonal adapters across different devices and aggregates their delta updates collectively on W for fine-tuning. Unlike Data Parallel LoRA or PiSSA, which maintain identical adapters across all devices, HD-PiSSA assigns different principal components of the pre-trained weights to each GPU, significantly expanding the range of update directions. This results in over 16x higher effective updated ranks than data-parallel LoRA or PiSSA when fine-tuning on 8 GPUs with the same per-device adapter rank. Empirically, we evaluate HD-PiSSA across various challenging downstream tasks, including mathematics, code generation, and multi-task learning. In the multi-task setting, HD-PiSSA achieves average gains of 10.0 absolute points (14.63%) over LoRA and 4.98 points (6.60%) over PiSSA across 12 benchmarks, demonstrating its benefits from the extra optimization flexibility. 

---
# OmniGenBench: A Benchmark for Omnipotent Multimodal Generation across 50+ Tasks 

**Authors**: Jiayu Wang, Yang Jiao, Yue Yu, Tianwen Qian, Shaoxiang Chen, Jingjing Chen, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18775)  

**Abstract**: Recent breakthroughs in large multimodal models (LMMs), such as the impressive GPT-4o-Native, have demonstrated remarkable proficiency in following general-purpose instructions for image generation. However, current benchmarks often lack the necessary breadth and depth to fully evaluate the diverse capabilities of these models. To overcome this limitation, we introduce OmniGenBench, a novel and comprehensive benchmark meticulously designed to assess the instruction-following abilities of state-of-the-art LMMs across both perception-centric and cognition-centric dimensions. Our OmniGenBench includes 57 diverse sub-tasks grounded in real-world scenarios, systematically categorized according to the specific model capabilities they demand. For rigorous evaluation, we further employ a dual-mode protocol. This protocol utilizes off-the-shelf visual parsing tools for perception-centric tasks and a powerful LLM-based judger for cognition-centric tasks to assess the alignment between generated images and user instructions. Using OmniGenBench, we evaluate mainstream generative models, including prevalent models like GPT-4o, Gemini-2.0-Flash, and Seedream, and provide in-depth comparisons and analyses of their this http URL and data are available at this https URL. 

---
# Strong Membership Inference Attacks on Massive Datasets and (Moderately) Large Language Models 

**Authors**: Jamie Hayes, Ilia Shumailov, Christopher A. Choquette-Choo, Matthew Jagielski, George Kaissis, Katherine Lee, Milad Nasr, Sahra Ghalebikesabi, Niloofar Mireshghallah, Meenatchi Sundaram Mutu Selva Annamalai, Igor Shilov, Matthieu Meeus, Yves-Alexandre de Montjoye, Franziska Boenisch, Adam Dziedzic, A. Feder Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2505.18773)  

**Abstract**: State-of-the-art membership inference attacks (MIAs) typically require training many reference models, making it difficult to scale these attacks to large pre-trained language models (LLMs). As a result, prior research has either relied on weaker attacks that avoid training reference models (e.g., fine-tuning attacks), or on stronger attacks applied to small-scale models and datasets. However, weaker attacks have been shown to be brittle - achieving close-to-arbitrary success - and insights from strong attacks in simplified settings do not translate to today's LLMs. These challenges have prompted an important question: are the limitations observed in prior work due to attack design choices, or are MIAs fundamentally ineffective on LLMs? We address this question by scaling LiRA - one of the strongest MIAs - to GPT-2 architectures ranging from 10M to 1B parameters, training reference models on over 20B tokens from the C4 dataset. Our results advance the understanding of MIAs on LLMs in three key ways: (1) strong MIAs can succeed on pre-trained LLMs; (2) their effectiveness, however, remains limited (e.g., AUC<0.7) in practical settings; and, (3) the relationship between MIA success and related privacy metrics is not as straightforward as prior work has suggested. 

---
# StyleGuard: Preventing Text-to-Image-Model-based Style Mimicry Attacks by Style Perturbations 

**Authors**: Yanjie Li, Wenxuan Zhang, Xinqi Lyu, Yihao Liu, Bin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18766)  

**Abstract**: Recently, text-to-image diffusion models have been widely used for style mimicry and personalized customization through methods such as DreamBooth and Textual Inversion. This has raised concerns about intellectual property protection and the generation of deceptive content. Recent studies, such as Glaze and Anti-DreamBooth, have proposed using adversarial noise to protect images from these attacks. However, recent purification-based methods, such as DiffPure and Noise Upscaling, have successfully attacked these latest defenses, showing the vulnerabilities of these methods. Moreover, present methods show limited transferability across models, making them less effective against unknown text-to-image models. To address these issues, we propose a novel anti-mimicry method, StyleGuard. We propose a novel style loss that optimizes the style-related features in the latent space to make it deviate from the original image, which improves model-agnostic transferability. Additionally, to enhance the perturbation's ability to bypass diffusion-based purification, we designed a novel upscale loss that involves ensemble purifiers and upscalers during training. Extensive experiments on the WikiArt and CelebA datasets demonstrate that StyleGuard outperforms existing methods in robustness against various transformations and purifications, effectively countering style mimicry in various models. Moreover, StyleGuard is effective on different style mimicry methods, including DreamBooth and Textual Inversion. 

---
# Towards an automatic method for generating topical vocabulary test forms for specific reading passages 

**Authors**: Michael Flor, Zuowei Wang, Paul Deane, Tenaha O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2505.18762)  

**Abstract**: Background knowledge is typically needed for successful comprehension of topical and domain specific reading passages, such as in the STEM domain. However, there are few automated measures of student knowledge that can be readily deployed and scored in time to make predictions on whether a given student will likely be able to understand a specific content area text. In this paper, we present our effort in developing K-tool, an automated system for generating topical vocabulary tests that measure students' background knowledge related to a specific text. The system automatically detects the topic of a given text and produces topical vocabulary items based on their relationship with the topic. This information is used to automatically generate background knowledge forms that contain words that are highly related to the topic and words that share similar features but do not share high associations to the topic. Prior research indicates that performance on such tasks can help determine whether a student is likely to understand a particular text based on their knowledge state. The described system is intended for use with middle and high school student population of native speakers of English. It is designed to handle single reading passages and is not dependent on any corpus or text collection. In this paper, we describe the system architecture and present an initial evaluation of the system outputs. 

---
# How Is LLM Reasoning Distracted by Irrelevant Context? An Analysis Using a Controlled Benchmark 

**Authors**: Minglai Yang, Ethan Huang, Liang Zhang, Mihai Surdeanu, William Wang, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18761)  

**Abstract**: We introduce Grade School Math with Distracting Context (GSM-DC), a synthetic benchmark to evaluate Large Language Models' (LLMs) reasoning robustness against systematically controlled irrelevant context (IC). GSM-DC constructs symbolic reasoning graphs with precise distractor injections, enabling rigorous, reproducible evaluation. Our experiments demonstrate that LLMs are significantly sensitive to IC, affecting both reasoning path selection and arithmetic accuracy. Additionally, training models with strong distractors improves performance in both in-distribution and out-of-distribution scenarios. We further propose a stepwise tree search guided by a process reward model, which notably enhances robustness in out-of-distribution conditions. 

---
# Smart Energy Guardian: A Hybrid Deep Learning Model for Detecting Fraudulent PV Generation 

**Authors**: Xiaolu Chen, Chenghao Huang, Yanru Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18755)  

**Abstract**: With the proliferation of smart grids, smart cities face growing challenges due to cyber-attacks and sophisticated electricity theft behaviors, particularly in residential photovoltaic (PV) generation systems. Traditional Electricity Theft Detection (ETD) methods often struggle to capture complex temporal dependencies and integrating multi-source data, limiting their effectiveness. In this work, we propose an efficient ETD method that accurately identifies fraudulent behaviors in residential PV generation, thus ensuring the supply-demand balance in smart cities. Our hybrid deep learning model, combining multi-scale Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and Transformer, excels in capturing both short-term and long-term temporal dependencies. Additionally, we introduce a data embedding technique that seamlessly integrates time-series data with discrete temperature variables, enhancing detection robustness. Extensive simulation experiments using real-world data validate the effectiveness of our approach, demonstrating significant improvements in the accuracy of detecting sophisticated energy theft activities, thereby contributing to the stability and fairness of energy systems in smart cities. 

---
# Agent-Based Decentralized Energy Management of EV Charging Station with Solar Photovoltaics via Multi-Agent Reinforcement Learning 

**Authors**: Jiarong Fan, Chenghao Huang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18750)  

**Abstract**: In the pursuit of energy net zero within smart cities, transportation electrification plays a pivotal role. The adoption of Electric Vehicles (EVs) keeps increasing, making energy management of EV charging stations critically important. While previous studies have managed to reduce energy cost of EV charging while maintaining grid stability, they often overlook the robustness of EV charging management against uncertainties of various forms, such as varying charging behaviors and possible faults in faults in some chargers. To address the gap, a novel Multi-Agent Reinforcement Learning (MARL) approach is proposed treating each charger to be an agent and coordinate all the agents in the EV charging station with solar photovoltaics in a more realistic scenario, where system faults may occur. A Long Short-Term Memory (LSTM) network is incorporated in the MARL algorithm to extract temporal features from time-series. Additionally, a dense reward mechanism is designed for training the agents in the MARL algorithm to improve EV charging experience. Through validation on a real-world dataset, we show that our approach is robust against system uncertainties and faults and also effective in minimizing EV charging costs and maximizing charging service satisfaction. 

---
# Season-Independent PV Disaggregation Using Multi-Scale Net Load Temporal Feature Extraction and Weather Factor Fusion 

**Authors**: Xiaolu Chen, Chenghao Huang, Yanru Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18747)  

**Abstract**: With the advancement of energy Internet and energy system integration, the increasing adoption of distributed photovoltaic (PV) systems presents new challenges on smart monitoring and measurement for utility companies, particularly in separating PV generation from net electricity load. Existing methods struggle with feature extraction from net load and capturing the relevance between weather factors. This paper proposes a PV disaggregation method that integrates Hierarchical Interpolation (HI) and multi-head self-attention mechanisms. By using HI to extract net load features and multi-head self-attention to capture the complex dependencies between weather factors, the method achieves precise PV generation predictions. Simulation experiments demonstrate the effectiveness of the proposed method in real-world data, supporting improved monitoring and management of distributed energy systems. 

---
# MoMBS: Mixed-order minibatch sampling enhances model training from diverse-quality images 

**Authors**: Han Li, Hu Han, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.18741)  

**Abstract**: Natural images exhibit label diversity (clean vs. noisy) in noisy-labeled image classification and prevalence diversity (abundant vs. sparse) in long-tailed image classification. Similarly, medical images in universal lesion detection (ULD) exhibit substantial variations in image quality, encompassing attributes such as clarity and label correctness. How to effectively leverage training images with diverse qualities becomes a problem in learning deep models. Conventional training mechanisms, such as self-paced curriculum learning (SCL) and online hard example mining (OHEM), relieve this problem by reweighting images with high loss values. Despite their success, these methods still confront two challenges: (i) the loss-based measure of sample hardness is imprecise, preventing optimum handling of different cases, and (ii) there exists under-utilization in SCL or over-utilization OHEM with the identified hard samples. To address these issues, this paper revisits the minibatch sampling (MBS), a technique widely used in deep network training but largely unexplored concerning the handling of diverse-quality training samples. We discover that the samples within a minibatch influence each other during training; thus, we propose a novel Mixed-order Minibatch Sampling (MoMBS) method to optimize the use of training samples with diverse qualities. MoMBS introduces a measure that takes both loss and uncertainty into account to surpass a sole reliance on loss and allows for a more refined categorization of high-loss samples by distinguishing them as either poorly labeled and under represented or well represented and overfitted. We prioritize under represented samples as the main gradient contributors in a minibatch and keep them from the negative influences of poorly labeled or overfitted samples with a mixed-order minibatch sampling design. 

---
# Message-Passing State-Space Models: Improving Graph Learning with Modern Sequence Modeling 

**Authors**: Andrea Ceni, Alessio Gravina, Claudio Gallicchio, Davide Bacciu, Carola-Bibiane Schonlieb, Moshe Eliasof  

**Link**: [PDF](https://arxiv.org/pdf/2505.18728)  

**Abstract**: The recent success of State-Space Models (SSMs) in sequence modeling has motivated their adaptation to graph learning, giving rise to Graph State-Space Models (GSSMs). However, existing GSSMs operate by applying SSM modules to sequences extracted from graphs, often compromising core properties such as permutation equivariance, message-passing compatibility, and computational efficiency. In this paper, we introduce a new perspective by embedding the key principles of modern SSM computation directly into the Message-Passing Neural Network framework, resulting in a unified methodology for both static and temporal graphs. Our approach, MP-SSM, enables efficient, permutation-equivariant, and long-range information propagation while preserving the architectural simplicity of message passing. Crucially, MP-SSM enables an exact sensitivity analysis, which we use to theoretically characterize information flow and evaluate issues like vanishing gradients and over-squashing in the deep regime. Furthermore, our design choices allow for a highly optimized parallel implementation akin to modern SSMs. We validate MP-SSM across a wide range of tasks, including node classification, graph property prediction, long-range benchmarks, and spatiotemporal forecasting, demonstrating both its versatility and strong empirical performance. 

---
# LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning 

**Authors**: Junyu Chen, Junzhuo Li, Zhen Peng, Wenjie Wang, Yuxiang Ren, Long Shi, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18724)  

**Abstract**: Quantization and fine-tuning are crucial for deploying large language models (LLMs) on resource-constrained edge devices. However, fine-tuning quantized models presents significant challenges, primarily stemming from: First, the mismatch in data types between the low-precision quantized weights (e.g., 4-bit) and the high-precision adaptation weights (e.g., 16-bit). This mismatch limits the computational efficiency advantage offered by quantized weights during inference. Second, potential accuracy degradation when merging these high-precision adaptation weights into the low-precision quantized weights, as the adaptation weights often necessitate approximation or truncation. Third, as far as we know, no existing methods support the lossless merging of adaptation while adjusting all quantized weights. To address these challenges, we introduce lossless ternary adaptation for quantization-aware fine-tuning (LoTA-QAF). This is a novel fine-tuning method specifically designed for quantized LLMs, enabling the lossless merging of ternary adaptation weights into quantized weights and the adjustment of all quantized weights. LoTA-QAF operates through a combination of: i) A custom-designed ternary adaptation (TA) that aligns ternary weights with the quantization grid and uses these ternary weights to adjust quantized weights. ii) A TA-based mechanism that enables the lossless merging of adaptation weights. iii) Ternary signed gradient descent (t-SignSGD) for updating the TA weights. We apply LoTA-QAF to Llama-3.1/3.3 and Qwen-2.5 model families and validate its effectiveness on several downstream tasks. On the MMLU benchmark, our method effectively recovers performance for quantized models, surpassing 16-bit LoRA by up to 5.14\%. For task-specific fine-tuning, 16-bit LoRA achieves superior results, but LoTA-QAF still outperforms other methods. 

---
# Evaluating the Usefulness of Non-Diagnostic Speech Data for Developing Parkinson's Disease Classifiers 

**Authors**: Terry Yi Zhong, Esther Janse, Cristian Tejedor-Garcia, Louis ten Bosch, Martha Larson  

**Link**: [PDF](https://arxiv.org/pdf/2505.18722)  

**Abstract**: Speech-based Parkinson's disease (PD) detection has gained attention for its automated, cost-effective, and non-intrusive nature. As research studies usually rely on data from diagnostic-oriented speech tasks, this work explores the feasibility of diagnosing PD on the basis of speech data not originally intended for diagnostic purposes, using the Turn-Taking (TT) dataset. Our findings indicate that TT can be as useful as diagnostic-oriented PD datasets like PC-GITA. We also investigate which specific dataset characteristics impact PD classification performance. The results show that concatenating audio recordings and balancing participants' gender and status distributions can be beneficial. Cross-dataset evaluation reveals that models trained on PC-GITA generalize poorly to TT, whereas models trained on TT perform better on PC-GITA. Furthermore, we provide insights into the high variability across folds, which is mainly due to large differences in individual speaker performance. 

---
# Optimal Transport-Based Token Weighting scheme for Enhanced Preference Optimization 

**Authors**: Meng Li, Guangda Huzhang, Haibo Zhang, Xiting Wang, Anxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.18720)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a promising framework for aligning Large Language Models (LLMs) with human preferences by directly optimizing the log-likelihood difference between chosen and rejected responses. However, existing methods assign equal importance to all tokens in the response, while humans focus on more meaningful parts. This leads to suboptimal preference optimization, as irrelevant or noisy tokens disproportionately influence DPO loss. To address this limitation, we propose \textbf{O}ptimal \textbf{T}ransport-based token weighting scheme for enhancing direct \textbf{P}reference \textbf{O}ptimization (OTPO). By emphasizing semantically meaningful token pairs and de-emphasizing less relevant ones, our method introduces a context-aware token weighting scheme that yields a more contrastive reward difference estimate. This adaptive weighting enhances reward stability, improves interpretability, and ensures that preference optimization focuses on meaningful differences between responses. Extensive experiments have validated OTPO's effectiveness in improving instruction-following ability across various settings\footnote{Code is available at this https URL.}. 

---
# VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning 

**Authors**: Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18719)  

**Abstract**: Recent high-capacity vision-language-action (VLA) models have demonstrated impressive performance on a range of robotic manipulation tasks by imitating human demonstrations. However, exploiting offline data with limited visited states will cause execution failure in out-of-distribution scenarios. Intuitively, an exploration-based method that improves on online collected data at test time could address this limitation. We present VLA-RL, an algorithmic and systematic framework that leverages online reinforcement learning (RL) to improve pretrained auto-regressive VLAs in downstream tasks. Within a unified perspective, we first introduce a trajectory-level RL formulation for auto-regressive VLA training, which models general robotic manipulation trajectory as multi-modal multi-turn conversation. To address the challenge of sparse rewards, we fine-tune a pretrained vision-language model as a robotic process reward model, which is trained on pseudo reward labels annotated on automatically extracted task segments. To scale up, we identify several implementation findings that improve the stability and efficiency including curriculum selection strategy, GPU-balanced vectorized environments, batch decoding, and critic warmup. VLA-RL enables OpenVLA-7B to surpass the strongest finetuned baseline by 4.5% on 40 challenging robotic manipulation tasks in LIBERO, and even matches the performance of advanced commercial models such as $\pi_0$-FAST. Notably, we observe that VLA-RL benefits from increased test-time optimization, indicating an early spark of inference scaling laws in robotics. 

---
# Neural Parameter Search for Slimmer Fine-Tuned Models and Better Transfer 

**Authors**: Guodong Du, Zitao Fang, Jing Li, Junlin Li, Runhua Jiang, Shuyang Yu, Yifei Guo, Yangneng Chen, Sim Kuan Goh, Ho-Kin Tang, Daojing He, Honghai Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18713)  

**Abstract**: Foundation models and their checkpoints have significantly advanced deep learning, boosting performance across various applications. However, fine-tuned models often struggle outside their specific domains and exhibit considerable redundancy. Recent studies suggest that combining a pruned fine-tuned model with the original pre-trained model can mitigate forgetting, reduce interference when merging model parameters across tasks, and improve compression efficiency. In this context, developing an effective pruning strategy for fine-tuned models is crucial. Leveraging the advantages of the task vector mechanism, we preprocess fine-tuned models by calculating the differences between them and the original model. Recognizing that different task vector subspaces contribute variably to model performance, we introduce a novel method called Neural Parameter Search (NPS-Pruning) for slimming down fine-tuned models. This method enhances pruning efficiency by searching through neural parameters of task vectors within low-rank subspaces. Our method has three key applications: enhancing knowledge transfer through pairwise model interpolation, facilitating effective knowledge fusion via model merging, and enabling the deployment of compressed models that retain near-original performance while significantly reducing storage costs. Extensive experiments across vision, NLP, and multi-modal benchmarks demonstrate the effectiveness and robustness of our approach, resulting in substantial performance gains. The code is publicly available at: this https URL. 

---
# GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis 

**Authors**: Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.18710)  

**Abstract**: The Retrieval-Augmented Generation (RAG) framework introduces a retrieval module to dynamically inject retrieved information into the input context of large language models (LLMs), and has demonstrated significant success in various NLP tasks. However, the current study points out that there is a preference gap between retrievers and LLMs in the RAG framework, which limit the further improvement of system performance. Some highly relevant passages may interfere with LLM reasoning because they contain complex or contradictory information; while some indirectly related or even inaccurate content may help LLM generate more accurate answers by providing suggestive information or logical clues. To solve this, we propose GainRAG, a novel approach that aligns the retriever's and LLM's preferences by defining a new metric, "gain", which measure how well an input passage contributes to correct outputs. Specifically, we propose a method to estimate these gain signals and train a middleware that aligns the preferences of the retriever and the LLM using only limited data. In addition, we introduce a pseudo-passage strategy to mitigate degradation. The experimental results on 6 datasets verify the effectiveness of GainRAG. 

---
# Improving Bangla Linguistics: Advanced LSTM, Bi-LSTM, and Seq2Seq Models for Translating Sylheti to Modern Bangla 

**Authors**: Sourav Kumar Das, Md. Julkar Naeen, MD. Jahidul Islam, Md. Anisul Haque Sajeeb, Narayan Ranjan Chakraborty, Mayen Uddin Mojumdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.18709)  

**Abstract**: Bangla or Bengali is the national language of Bangladesh, people from different regions don't talk in proper Bangla. Every division of Bangladesh has its own local language like Sylheti, Chittagong etc. In recent years some papers were published on Bangla language like sentiment analysis, fake news detection and classifications, but a few of them were on Bangla languages. This research is for the local language and this particular paper is on Sylheti language. It presented a comprehensive system using Natural Language Processing or NLP techniques for translating Pure or Modern Bangla to locally spoken Sylheti Bangla language. Total 1200 data used for training 3 models LSTM, Bi-LSTM and Seq2Seq and LSTM scored the best in performance with 89.3% accuracy. The findings of this research may contribute to the growth of Bangla NLP researchers for future more advanced innovations. 

---
# A General Knowledge Injection Framework for ICD Coding 

**Authors**: Xu Zhang, Kun Zhang, Wenxin Ma, Rongsheng Wang, Chenxu Wu, Yingtai Li, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.18708)  

**Abstract**: ICD Coding aims to assign a wide range of medical codes to a medical text document, which is a popular and challenging task in the healthcare domain. To alleviate the problems of long-tail distribution and the lack of annotations of code-specific evidence, many previous works have proposed incorporating code knowledge to improve coding performance. However, existing methods often focus on a single type of knowledge and design specialized modules that are complex and incompatible with each other, thereby limiting their scalability and effectiveness. To address this issue, we propose GKI-ICD, a novel, general knowledge injection framework that integrates three key types of knowledge, namely ICD Description, ICD Synonym, and ICD Hierarchy, without specialized design of additional modules. The comprehensive utilization of the above knowledge, which exhibits both differences and complementarity, can effectively enhance the ICD coding performance. Extensive experiments on existing popular ICD coding benchmarks demonstrate the effectiveness of GKI-ICD, which achieves the state-of-the-art performance on most evaluation metrics. Code is available at this https URL. 

---
# Steering LLM Reasoning Through Bias-Only Adaptation 

**Authors**: Viacheslav Sinii, Alexey Gorbatovski, Artem Cherepanov, Boris Shaposhnikov, Nikita Balagansky, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2505.18706)  

**Abstract**: Recent work on reasoning-oriented language models, exemplified by o1-like systems, suggests that reinforcement-learning (RL) finetuning does not create new capabilities but instead strengthens reasoning patterns already latent in the pretrained network. We test this claim by training steering vectors: layer-wise biases that additively amplify selected hidden features while leaving all original weights unchanged. Experiments on four base models across the GSM8K and MATH benchmarks show that steering vectors recover, and in several cases exceed, the accuracy of fully-tuned counterparts. This result supports the view that the required reasoning skills pre-exist in the base model. Further, logit-lens analysis reveals that the trained vectors consistently boost token groups linked to structured languages and logical connectors, providing an interpretable account that aligns with the demands of quantitative reasoning tasks. 

---
# GRE Suite: Geo-localization Inference via Fine-Tuned Vision-Language Models and Enhanced Reasoning Chains 

**Authors**: Chun Wang, Xiaoran Pan, Zihao Pan, Haofan Wang, Yiren Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.18700)  

**Abstract**: Recent advances in Visual Language Models (VLMs) have demonstrated exceptional performance in visual reasoning tasks. However, geo-localization presents unique challenges, requiring the extraction of multigranular visual cues from images and their integration with external world knowledge for systematic reasoning. Current approaches to geo-localization tasks often lack robust reasoning mechanisms and explainability, limiting their effectiveness. To address these limitations, we propose the Geo Reason Enhancement (GRE) Suite, a novel framework that augments VLMs with structured reasoning chains for accurate and interpretable location inference. The GRE Suite is systematically developed across three key dimensions: dataset, model, and benchmark. First, we introduce GRE30K, a high-quality geo-localization reasoning dataset designed to facilitate fine-grained visual and contextual analysis. Next, we present the GRE model, which employs a multi-stage reasoning strategy to progressively infer scene attributes, local details, and semantic features, thereby narrowing down potential geographic regions with enhanced precision. Finally, we construct the Geo Reason Evaluation Benchmark (GREval-Bench), a comprehensive evaluation framework that assesses VLMs across diverse urban, natural, and landmark scenes to measure both coarse-grained (e.g., country, continent) and fine-grained (e.g., city, street) localization performance. Experimental results demonstrate that GRE significantly outperforms existing methods across all granularities of geo-localization tasks, underscoring the efficacy of reasoning-augmented VLMs in complex geographic inference. Code and data will be released at this https URL. 

---
# Can LLMs Alleviate Catastrophic Forgetting in Graph Continual Learning? A Systematic Study 

**Authors**: Ziyang Cheng, Zhixun Li, Yuhan Li, Yixin Song, Kangyi Zhao, Dawei Cheng, Jia Li, Jeffrey Xu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18697)  

**Abstract**: Nowadays, real-world data, including graph-structure data, often arrives in a streaming manner, which means that learning systems need to continuously acquire new knowledge without forgetting previously learned information. Although substantial existing works attempt to address catastrophic forgetting in graph machine learning, they are all based on training from scratch with streaming data. With the rise of pretrained models, an increasing number of studies have leveraged their strong generalization ability for continual learning. Therefore, in this work, we attempt to answer whether large language models (LLMs) can mitigate catastrophic forgetting in Graph Continual Learning (GCL). We first point out that current experimental setups for GCL have significant flaws, as the evaluation stage may lead to task ID leakage. Then, we evaluate the performance of LLMs in more realistic scenarios and find that even minor modifications can lead to outstanding results. Finally, based on extensive experiments, we propose a simple-yet-effective method, Simple Graph Continual Learning (SimGCL), that surpasses the previous state-of-the-art GNN-based baseline by around 20% under the rehearsal-free constraint. To facilitate reproducibility, we have developed an easy-to-use benchmark LLM4GCL for training and evaluating existing GCL methods. The code is available at: this https URL. 

---
# Large Language Models in the Task of Automatic Validation of Text Classifier Predictions 

**Authors**: Aleksandr Tsymbalov  

**Link**: [PDF](https://arxiv.org/pdf/2505.18688)  

**Abstract**: Machine learning models for text classification are trained to predict a class for a given text. To do this, training and validation samples must be prepared: a set of texts is collected, and each text is assigned a class. These classes are usually assigned by human annotators with different expertise levels, depending on the specific classification task. Collecting such samples from scratch is labor-intensive because it requires finding specialists and compensating them for their work; moreover, the number of available specialists is limited, and their productivity is constrained by human factors. While it may not be too resource-intensive to collect samples once, the ongoing need to retrain models (especially in incremental learning pipelines) to address data drift (also called model drift) makes the data collection process crucial and costly over the model's entire lifecycle. This paper proposes several approaches to replace human annotators with Large Language Models (LLMs) to test classifier predictions for correctness, helping ensure model quality and support high-quality incremental learning. 

---
# An AI Capability Threshold for Rent-Funded Universal Basic Income in an AI-Automated Economy 

**Authors**: Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18687)  

**Abstract**: We derive the first closed-form condition under which artificial intelligence (AI) capital profits could sustainably finance a universal basic income (UBI) without additional taxes or new job creation. In a Solow-Zeira economy characterized by a continuum of automatable tasks, a constant net saving rate $s$, and task-elasticity $\sigma < 1$, we analyze how the AI capability threshold--defined as the productivity level of AI relative to pre-AI automation--varies under different economic scenarios. At present economic parameters, we find that AI systems must achieve only approximately 5-6 times existing automation productivity to finance an 11\%-of-GDP UBI, in the worst case situation where \emph{no} new jobs or tasks are created.
Our analysis also reveals some specific policy levers: raising public revenue share (e.g. profit taxation) of AI capital from the current 15\% to about 33\% halves the required AI capability threshold to attain UBI to 3 times existing automotion productivity, but gains diminish beyond 50\% public revenue share, especially if regulatory costs increase. Market structure also strongly affects outcomes: monopolistic or concentrated oligopolistic markets reduce the threshold by increasing economic rents, whereas heightened competition significantly raises it.
Overall, these results suggest a couple policy recommendations: maximizing public revenue share up to a point so that operating costs are minimized, and strategically managing market competition can ensure AI's growing capabilities translate into meaningful social benefits within realistic technological progress scenarios. 

---
# Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps 

**Authors**: Sicheng Feng, Song Wang, Shuyi Ouyang, Lingdong Kong, Zikai Song, Jianke Zhu, Huan Wang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18675)  

**Abstract**: Multimodal large language models (MLLMs) have recently achieved significant progress in visual tasks, including semantic scene understanding and text-image alignment, with reasoning variants enhancing performance on complex tasks involving mathematics and logic. However, their capacity for reasoning tasks involving fine-grained visual understanding remains insufficiently evaluated. To address this gap, we introduce ReasonMap, a benchmark designed to assess the fine-grained visual understanding and spatial reasoning abilities of MLLMs. ReasonMap encompasses high-resolution transit maps from 30 cities across 13 countries and includes 1,008 question-answer pairs spanning two question types and three templates. Furthermore, we design a two-level evaluation pipeline that properly assesses answer correctness and quality. Comprehensive evaluations of 15 popular MLLMs, including both base and reasoning variants, reveal a counterintuitive pattern: among open-source models, base models outperform reasoning ones, while the opposite trend is observed in closed-source models. Additionally, performance generally degrades when visual inputs are masked, indicating that while MLLMs can leverage prior knowledge to answer some questions, fine-grained visual reasoning tasks still require genuine visual perception for strong performance. Our benchmark study offers new insights into visual reasoning and contributes to investigating the gap between open-source and closed-source models. 

---
# Restoring Real-World Images with an Internal Detail Enhancement Diffusion Model 

**Authors**: Peng Xiao, Hongbo Zhao, Yijun Wang, Jianxin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.18674)  

**Abstract**: Restoring real-world degraded images, such as old photographs or low-resolution images, presents a significant challenge due to the complex, mixed degradations they exhibit, such as scratches, color fading, and noise. Recent data-driven approaches have struggled with two main challenges: achieving high-fidelity restoration and providing object-level control over colorization. While diffusion models have shown promise in generating high-quality images with specific controls, they often fail to fully preserve image details during restoration. In this work, we propose an internal detail-preserving diffusion model for high-fidelity restoration of real-world degraded images. Our method utilizes a pre-trained Stable Diffusion model as a generative prior, eliminating the need to train a model from scratch. Central to our approach is the Internal Image Detail Enhancement (IIDE) technique, which directs the diffusion model to preserve essential structural and textural information while mitigating degradation effects. The process starts by mapping the input image into a latent space, where we inject the diffusion denoising process with degradation operations that simulate the effects of various degradation factors. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art models in both qualitative assessments and perceptual quantitative evaluations. Additionally, our approach supports text-guided restoration, enabling object-level colorization control that mimics the expertise of professional photo editing. 

---
# Adaptive Prediction-Powered AutoEval with Reliability and Efficiency Guarantees 

**Authors**: Sangwoo Park, Matteo Zecchin, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2505.18659)  

**Abstract**: Selecting artificial intelligence (AI) models, such as large language models (LLMs), from multiple candidates requires accurate performance estimation. This is ideally achieved through empirical evaluations involving abundant real-world data. However, such evaluations are costly and impractical at scale. To address this challenge, autoevaluation methods leverage synthetic data produced by automated evaluators, such as LLMs-as-judges, reducing variance but potentially introducing bias. Recent approaches have employed semi-supervised prediction-powered inference (\texttt{PPI}) to correct for the bias of autoevaluators. However, the use of autoevaluators may lead in practice to a degradation in sample efficiency compared to conventional methods using only real-world data. In this paper, we propose \texttt{R-AutoEval+}, a novel framework that provides finite-sample reliability guarantees on the model evaluation, while also ensuring an enhanced (or at least no worse) sample efficiency compared to conventional methods. The key innovation of \texttt{R-AutoEval+} is an adaptive construction of the model evaluation variable, which dynamically tunes its reliance on synthetic data, reverting to conventional methods when the autoevaluator is insufficiently accurate. Experiments on the use of LLMs-as-judges for the optimization of quantization settings for the weights of an LLM, and for prompt design in LLMs confirm the reliability and efficiency of \texttt{R-AutoEval+}. 

---
# Robustness in Large Language Models: A Survey of Mitigation Strategies and Evaluation Metrics 

**Authors**: Pankaj Kumar, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.18658)  

**Abstract**: Large Language Models (LLMs) have emerged as a promising cornerstone for the development of natural language processing (NLP) and artificial intelligence (AI). However, ensuring the robustness of LLMs remains a critical challenge. To address these challenges and advance the field, this survey provides a comprehensive overview of current studies in this area. First, we systematically examine the nature of robustness in LLMs, including its conceptual foundations, the importance of consistent performance across diverse inputs, and the implications of failure modes in real-world applications. Next, we analyze the sources of non-robustness, categorizing intrinsic model limitations, data-driven vulnerabilities, and external adversarial factors that compromise reliability. Following this, we review state-of-the-art mitigation strategies, and then we discuss widely adopted benchmarks, emerging metrics, and persistent gaps in assessing real-world reliability. Finally, we synthesize findings from existing surveys and interdisciplinary studies to highlight trends, unresolved issues, and pathways for future research. 

---
# Flow Matching for Geometric Trajectory Simulation 

**Authors**: Kiet Bennema ten Brinke, Koen Minartz, Vlado Menkovski  

**Link**: [PDF](https://arxiv.org/pdf/2505.18647)  

**Abstract**: The simulation of N-body systems is a fundamental problem with applications in a wide range of fields, such as molecular dynamics, biochemistry, and pedestrian dynamics. Machine learning has become an invaluable tool for scaling physics-based simulators and developing models directly from experimental data. In particular, recent advances based on deep generative modeling and geometric deep learning have enabled probabilistic simulation by modeling complex distributions over trajectories while respecting the permutation symmetry that is fundamental to N-body systems. However, to generate realistic trajectories, existing methods must learn complex transformations starting from uninformed noise and do not allow for the exploitation of domain-informed priors. In this work, we propose STFlow to address this limitation. By leveraging flow matching and data-dependent couplings, STFlow facilitates physics-informed simulation of geometric trajectories without sacrificing model expressivity or scalability. Our evaluation on N-body dynamical systems, molecular dynamics, and pedestrian dynamics benchmarks shows that STFlow produces significantly lower prediction errors while enabling more efficient inference, highlighting the benefits of employing physics-informed prior distributions in probabilistic geometric trajectory modeling. 

---
# SEW: Self-Evolving Agentic Workflows for Automated Code Generation 

**Authors**: Siwei Liu, Jinyuan Fang, Han Zhou, Yingxu Wang, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2505.18646)  

**Abstract**: Large Language Models (LLMs) have demonstrated effectiveness in code generation tasks. To enable LLMs to address more complex coding challenges, existing research has focused on crafting multi-agent systems with agentic workflows, where complex coding tasks are decomposed into sub-tasks, assigned to specialized agents. Despite their effectiveness, current approaches heavily rely on hand-crafted agentic workflows, with both agent topologies and prompts manually designed, which limits their ability to automatically adapt to different types of coding problems. To address these limitations and enable automated workflow design, we propose \textbf{S}elf-\textbf{E}volving \textbf{W}orkflow (\textbf{SEW}), a novel self-evolving framework that automatically generates and optimises multi-agent workflows. Extensive experiments on three coding benchmark datasets, including the challenging LiveCodeBench, demonstrate that our SEW can automatically design agentic workflows and optimise them through self-evolution, bringing up to 33\% improvement on LiveCodeBench compared to using the backbone LLM only. Furthermore, by investigating different representation schemes of workflow, we provide insights into the optimal way to encode workflow information with text. 

---
# Anomaly detection in radio galaxy data with trainable COSFIRE filters 

**Authors**: Steven Ndung'u, Trienko Grobler, Stefan J. Wijnholds, George Azzopardi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18643)  

**Abstract**: Detecting anomalies in radio astronomy is challenging due to the vast amounts of data and the rarity of labeled anomalous examples. Addressing this challenge requires efficient methods capable of identifying unusual radio galaxy morphologies without relying on extensive supervision. This work introduces an innovative approach to anomaly detection based on morphological characteristics of the radio sources using trainable COSFIRE (Combination of Shifted Filter Responses) filters as an efficient alternative to complex deep learning methods. The framework integrates COSFIRE descriptors with an unsupervised Local Outlier Factor (LOF) algorithm to identify unusual radio galaxy morphologies. Evaluations on a radio galaxy benchmark data set demonstrate strong performance, with the COSFIRE-based approach achieving a geometric mean (G-Mean) score of 79%, surpassing the 77% achieved by a computationally intensive deep learning autoencoder. By characterizing normal patterns and detecting deviations, this semi-supervised methodology overcomes the need for anomalous examples in the training set, a major limitation of traditional supervised methods. This approach shows promise for next-generation radio telescopes, where fast processing and the ability to discover unknown phenomena are crucial. 

---
# ThanoRA: Task Heterogeneity-Aware Multi-Task Low-Rank Adaptation 

**Authors**: Jian Liang, Wenke Huang, Xianda Guo, Guancheng Wan, Bo Du, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.18640)  

**Abstract**: Low-Rank Adaptation (LoRA) is widely adopted for downstream fine-tuning of foundation models due to its efficiency and zero additional inference cost. Many real-world applications require foundation models to specialize in multiple tasks simultaneously, motivating the need for efficient multi-task adaptation. While recent approaches integrate LoRA with mixture-of-experts (MoE) to address this, the use of routers prevents parameter mergeability, which increases inference overhead and hinders unified multi-task adaptation, thereby limiting deployment practicality. In this work, we propose ThanoRA, a Task Heterogeneity-Aware Multi-Task Low-Rank Adaptation framework that enables multi-task adaptation while preserving the inference efficiency of LoRA. ThanoRA jointly models task heterogeneity and mitigates subspace interference throughout training. Specifically, motivated by inherent differences in complexity and heterogeneity across tasks, ThanoRA constructs task-specific LoRA subspaces at initialization, enabling fine-grained knowledge injection aligned with task heterogeneity. Furthermore, to prevent task interference and subspace collapse during multi-task training, ThanoRA introduces a subspace-preserving regularization that maintains the independence of task-specific representations. With the synergy of both components, ThanoRA enables efficient and unified multi-task adaptation. Extensive experiments across multimodal and text-only benchmarks under varying multi-task mixtures demonstrate that ThanoRA consistently achieves robust and superior performance over strong baselines without introducing additional inference overhead. Our code is publicly available at: this https URL. 

---
# DDO: Dual-Decision Optimization via Multi-Agent Collaboration for LLM-Based Medical Consultation 

**Authors**: Zhihao Jia, Mingyi Jia, Junwen Duan, Jianxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18630)  

**Abstract**: Large Language Models (LLMs) demonstrate strong generalization and reasoning abilities, making them well-suited for complex decision-making tasks such as medical consultation (MC). However, existing LLM-based methods often fail to capture the dual nature of MC, which entails two distinct sub-tasks: symptom inquiry, a sequential decision-making process, and disease diagnosis, a classification problem. This mismatch often results in ineffective symptom inquiry and unreliable disease diagnosis. To address this, we propose \textbf{DDO}, a novel LLM-based framework that performs \textbf{D}ual-\textbf{D}ecision \textbf{O}ptimization by decoupling and independently optimizing the the two sub-tasks through a collaborative multi-agent workflow. Experiments on three real-world MC datasets show that DDO consistently outperforms existing LLM-based approaches and achieves competitive performance with state-of-the-art generation-based methods, demonstrating its effectiveness in the MC task. 

---
# Trust, or Don't Predict: Introducing the CWSA Family for Confidence-Aware Model Evaluation 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh, Mohammadali Keshtparvar, Pegah Ghaffari  

**Link**: [PDF](https://arxiv.org/pdf/2505.18622)  

**Abstract**: In recent machine learning systems, confidence scores are being utilized more and more to manage selective prediction, whereby a model can abstain from making a prediction when it is unconfident. Yet, conventional metrics like accuracy, expected calibration error (ECE), and area under the risk-coverage curve (AURC) do not capture the actual reliability of predictions. These metrics either disregard confidence entirely, dilute valuable localized information through averaging, or neglect to suitably penalize overconfident misclassifications, which can be particularly detrimental in real-world systems. We introduce two new metrics Confidence-Weighted Selective Accuracy (CWSA) and its normalized variant CWSA+ that offer a principled and interpretable way to evaluate predictive models under confidence thresholds. Unlike existing methods, our metrics explicitly reward confident accuracy and penalize overconfident mistakes. They are threshold-local, decomposable, and usable in both evaluation and deployment settings where trust and risk must be quantified. Through exhaustive experiments on both real-world data sets (MNIST, CIFAR-10) and artificial model variants (calibrated, overconfident, underconfident, random, perfect), we show that CWSA and CWSA+ both effectively detect nuanced failure modes and outperform classical metrics in trust-sensitive tests. Our results confirm that CWSA is a sound basis for developing and assessing selective prediction systems for safety-critical domains. 

---
# Rethinking Causal Mask Attention for Vision-Language Inference 

**Authors**: Xiaohuan Pei, Tao Huang, YanXiang Ma, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18605)  

**Abstract**: Causal attention has become a foundational mechanism in autoregressive vision-language models (VLMs), unifying textual and visual inputs under a single generative framework. However, existing causal mask-based strategies are inherited from large language models (LLMs) where they are tailored for text-only decoding, and their adaptation to vision tokens is insufficiently addressed in the prefill stage. Strictly masking future positions for vision queries introduces overly rigid constraints, which hinder the model's ability to leverage future context that often contains essential semantic cues for accurate inference. In this work, we empirically investigate how different causal masking strategies affect vision-language inference and then propose a family of future-aware attentions tailored for this setting. We first empirically analyze the effect of previewing future tokens for vision queries and demonstrate that rigid masking undermines the model's capacity to capture useful contextual semantic representations. Based on these findings, we propose a lightweight attention family that aggregates future visual context into past representations via pooling, effectively preserving the autoregressive structure while enhancing cross-token dependencies. We evaluate a range of causal masks across diverse vision-language inference settings and show that selectively compressing future semantic context into past representations benefits the inference. 

---
# LLM-Meta-SR: Learning to Evolve Selection Operators for Symbolic Regression 

**Authors**: Hengzhe Zhang, Qi Chen, Bing Xue, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18602)  

**Abstract**: Large language models (LLMs) have revolutionized algorithm development, yet their application in symbolic regression, where algorithms automatically discover symbolic expressions from data, remains constrained and is typically designed manually by human experts. In this paper, we propose a learning-to-evolve framework that enables LLMs to automatically design selection operators for evolutionary symbolic regression algorithms. We first identify two key limitations in existing LLM-based algorithm evolution techniques: code bloat and a lack of semantic guidance. Bloat results in unnecessarily complex components, and the absence of semantic awareness can lead to ineffective exchange of useful code components, both of which can reduce the interpretability of the designed algorithm or hinder evolutionary learning progress. To address these issues, we enhance the LLM-based evolution framework for meta symbolic regression with two key innovations: bloat control and a complementary, semantics-aware selection operator. Additionally, we embed domain knowledge into the prompt, enabling the LLM to generate more effective and contextually relevant selection operators. Our experimental results on symbolic regression benchmarks show that LLMs can devise selection operators that outperform nine expert-designed baselines, achieving state-of-the-art performance. This demonstrates that LLMs can exceed expert-level algorithm design for symbolic regression. 

---
# Flex-Judge: Think Once, Judge Anywhere 

**Authors**: Jongwoo Ko, Sungnyun Kim, Sungwoo Cho, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.18601)  

**Abstract**: Human-generated reward signals are critical for aligning generative models with human preferences, guiding both training and inference-time evaluations. While large language models (LLMs) employed as proxy evaluators, i.e., LLM-as-a-Judge, significantly reduce the costs associated with manual annotations, they typically require extensive modality-specific training data and fail to generalize well across diverse multimodal tasks. In this paper, we propose Flex-Judge, a reasoning-guided multimodal judge model that leverages minimal textual reasoning data to robustly generalize across multiple modalities and evaluation formats. Our core intuition is that structured textual reasoning explanations inherently encode generalizable decision-making patterns, enabling an effective transfer to multimodal judgments, e.g., with images or videos. Empirical results demonstrate that Flex-Judge, despite being trained on significantly fewer text data, achieves competitive or superior performance compared to state-of-the-art commercial APIs and extensively trained multimodal evaluators. Notably, Flex-Judge presents broad impact in modalities like molecule, where comprehensive evaluation benchmarks are scarce, underscoring its practical value in resource-constrained domains. Our framework highlights reasoning-based text supervision as a powerful, cost-effective alternative to traditional annotation-intensive approaches, substantially advancing scalable multimodal model-as-a-judge. 

---
# Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment 

**Authors**: Bryan Sangwoo Kim, Jeongsol Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.18600)  

**Abstract**: Modern single-image super-resolution (SISR) models deliver photo-realistic results at the scale factors on which they are trained, but collapse when asked to magnify far beyond that regime. We address this scalability bottleneck with Chain-of-Zoom (CoZ), a model-agnostic framework that factorizes SISR into an autoregressive chain of intermediate scale-states with multi-scale-aware prompts. CoZ repeatedly re-uses a backbone SR model, decomposing the conditional probability into tractable sub-problems to achieve extreme resolutions without additional training. Because visual cues diminish at high magnifications, we augment each zoom step with multi-scale-aware text prompts generated by a vision-language model (VLM). The prompt extractor itself is fine-tuned using Generalized Reward Policy Optimization (GRPO) with a critic VLM, aligning text guidance towards human preference. Experiments show that a standard 4x diffusion SR model wrapped in CoZ attains beyond 256x enlargement with high perceptual quality and fidelity. 

---
# Debate-to-Detect: Reformulating Misinformation Detection as a Real-World Debate with Large Language Models 

**Authors**: Chen Han, Wenzhen Zheng, Xijin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18596)  

**Abstract**: The proliferation of misinformation in digital platforms reveals the limitations of traditional detection methods, which mostly rely on static classification and fail to capture the intricate process of real-world fact-checking. Despite advancements in Large Language Models (LLMs) that enhance automated reasoning, their application to misinformation detection remains hindered by issues of logical inconsistency and superficial verification. In response, we introduce Debate-to-Detect (D2D), a novel Multi-Agent Debate (MAD) framework that reformulates misinformation detection as a structured adversarial debate. Inspired by fact-checking workflows, D2D assigns domain-specific profiles to each agent and orchestrates a five-stage debate process, including Opening Statement, Rebuttal, Free Debate, Closing Statement, and Judgment. To transcend traditional binary classification, D2D introduces a multi-dimensional evaluation mechanism that assesses each claim across five distinct dimensions: Factuality, Source Reliability, Reasoning Quality, Clarity, and Ethics. Experiments with GPT-4o on two fakenews datasets demonstrate significant improvements over baseline methods, and the case study highlight D2D's capability to iteratively refine evidence while improving decision transparency, representing a substantial advancement towards robust and interpretable misinformation detection. The code will be open-sourced in a future release. 

---
# MisoDICE: Multi-Agent Imitation from Unlabeled Mixed-Quality Demonstrations 

**Authors**: Viet Bui, Tien Mai, Hong Thanh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18595)  

**Abstract**: We study offline imitation learning (IL) in cooperative multi-agent settings, where demonstrations have unlabeled mixed quality - containing both expert and suboptimal trajectories. Our proposed solution is structured in two stages: trajectory labeling and multi-agent imitation learning, designed jointly to enable effective learning from heterogeneous, unlabeled data. In the first stage, we combine advances in large language models and preference-based reinforcement learning to construct a progressive labeling pipeline that distinguishes expert-quality trajectories. In the second stage, we introduce MisoDICE, a novel multi-agent IL algorithm that leverages these labels to learn robust policies while addressing the computational complexity of large joint state-action spaces. By extending the popular single-agent DICE framework to multi-agent settings with a new value decomposition and mixing architecture, our method yields a convex policy optimization objective and ensures consistency between global and local policies. We evaluate MisoDICE on multiple standard multi-agent RL benchmarks and demonstrate superior performance, especially when expert data is scarce. 

---
# Safety Alignment via Constrained Knowledge Unlearning 

**Authors**: Zesheng Shi, Yucheng Zhou, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.18588)  

**Abstract**: Despite significant progress in safety alignment, large language models (LLMs) remain susceptible to jailbreak attacks. Existing defense mechanisms have not fully deleted harmful knowledge in LLMs, which allows such attacks to bypass safeguards and produce harmful outputs. To address this challenge, we propose a novel safety alignment strategy, Constrained Knowledge Unlearning (CKU), which focuses on two primary objectives: knowledge localization and retention, and unlearning harmful knowledge. CKU works by scoring neurons in specific multilayer perceptron (MLP) layers to identify a subset U of neurons associated with useful knowledge. During the unlearning process, CKU prunes the gradients of neurons in U to preserve valuable knowledge while effectively mitigating harmful content. Experimental results demonstrate that CKU significantly enhances model safety without compromising overall performance, offering a superior balance between safety and utility compared to existing methods. Additionally, our analysis of neuron knowledge sensitivity across various MLP layers provides valuable insights into the mechanics of safety alignment and model knowledge editing. 

---
# HyperFake: Hyperspectral Reconstruction and Attention-Guided Analysis for Advanced Deepfake Detection 

**Authors**: Pavan C Shekar, Pawan Soni, Vivek Kanhangad  

**Link**: [PDF](https://arxiv.org/pdf/2505.18587)  

**Abstract**: Deepfakes pose a significant threat to digital media security, with current detection methods struggling to generalize across different manipulation techniques and datasets. While recent approaches combine CNN-based architectures with Vision Transformers or leverage multi-modal learning, they remain limited by the inherent constraints of RGB data. We introduce HyperFake, a novel deepfake detection pipeline that reconstructs 31-channel hyperspectral data from standard RGB videos, revealing hidden manipulation traces invisible to conventional methods. Using an improved MST++ architecture, HyperFake enhances hyperspectral reconstruction, while a spectral attention mechanism selects the most critical spectral features for deepfake detection. The refined spectral data is then processed by an EfficientNet-based classifier optimized for spectral analysis, enabling more accurate and generalizable detection across different deepfake styles and datasets, all without the need for expensive hyperspectral cameras. To the best of our knowledge, this is the first approach to leverage hyperspectral imaging reconstruction for deepfake detection, opening new possibilities for detecting increasingly sophisticated manipulations. 

---
# On Denoising Walking Videos for Gait Recognition 

**Authors**: Dongyang Jin, Chao Fan, Jingzhe Ma, Jingkai Zhou, Weihua Chen, Shiqi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18582)  

**Abstract**: To capture individual gait patterns, excluding identity-irrelevant cues in walking videos, such as clothing texture and color, remains a persistent challenge for vision-based gait recognition. Traditional silhouette- and pose-based methods, though theoretically effective at removing such distractions, often fall short of high accuracy due to their sparse and less informative inputs. Emerging end-to-end methods address this by directly denoising RGB videos using human priors. Building on this trend, we propose DenoisingGait, a novel gait denoising method. Inspired by the philosophy that "what I cannot create, I do not understand", we turn to generative diffusion models, uncovering how they partially filter out irrelevant factors for gait understanding. Additionally, we introduce a geometry-driven Feature Matching module, which, combined with background removal via human silhouettes, condenses the multi-channel diffusion features at each foreground pixel into a two-channel direction vector. Specifically, the proposed within- and cross-frame matching respectively capture the local vectorized structures of gait appearance and motion, producing a novel flow-like gait representation termed Gait Feature Field, which further reduces residual noise in diffusion features. Experiments on the CCPG, CASIA-B*, and SUSTech1K datasets demonstrate that DenoisingGait achieves a new SoTA performance in most cases for both within- and cross-domain evaluations. Code is available at this https URL. 

---
# Removal of Hallucination on Hallucination: Debate-Augmented RAG 

**Authors**: Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason Zhang, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.18581)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances factual accuracy by integrating external knowledge, yet it introduces a critical issue: erroneous or biased retrieval can mislead generation, compounding hallucinations, a phenomenon we term Hallucination on Hallucination. To address this, we propose Debate-Augmented RAG (DRAG), a training-free framework that integrates Multi-Agent Debate (MAD) mechanisms into both retrieval and generation stages. In retrieval, DRAG employs structured debates among proponents, opponents, and judges to refine retrieval quality and ensure factual reliability. In generation, DRAG introduces asymmetric information roles and adversarial debates, enhancing reasoning robustness and mitigating factual inconsistencies. Evaluations across multiple tasks demonstrate that DRAG improves retrieval reliability, reduces RAG-induced hallucinations, and significantly enhances overall factual accuracy. Our code is available at this https URL. 

---
# Autocomp: LLM-Driven Code Optimization for Tensor Accelerators 

**Authors**: Charles Hong, Sahil Bhatia, Alvin Cheung, Yakun Sophia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18574)  

**Abstract**: Hardware accelerators, especially those designed for tensor processing, have become ubiquitous in today's computing landscape. However, even with significant efforts in building compilers, programming these tensor accelerators remains challenging, leaving much of their potential underutilized. Recently, large language models (LLMs), trained on large amounts of code, have shown significant promise in code generation and optimization tasks, but generating low-resource languages like specialized tensor accelerator code still poses a significant challenge. We tackle this challenge with Autocomp, an approach that empowers accelerator programmers to leverage domain knowledge and hardware feedback to optimize code via an automated LLM-driven search. We accomplish this by: 1) formulating each optimization pass as a structured two-phase prompt, divided into planning and code generation phases, 2) inserting domain knowledge during planning via a concise and adaptable optimization menu, and 3) integrating correctness and performance metrics from hardware as feedback at each search iteration. Across three categories of representative workloads and two different accelerators, we demonstrate that Autocomp-optimized code runs 5.6x (GEMM) and 2.7x (convolution) faster than the vendor-provided library, and outperforms expert-level hand-tuned code by 1.4x (GEMM), 1.1x (convolution), and 1.3x (fine-grained linear algebra). Additionally, we demonstrate that optimization schedules generated from Autocomp can be reused across similar tensor operations, improving speedups by up to 24% under a fixed sample budget. 

---
# MASTER: Multi-Agent Security Through Exploration of Roles and Topological Structures -- A Comprehensive Framework 

**Authors**: Yifan Zhu, Chao Zhang, Xin Shi, Xueqiao Zhang, Yi Yang, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.18572)  

**Abstract**: Large Language Models (LLMs)-based Multi-Agent Systems (MAS) exhibit remarkable problem-solving and task planning capabilities across diverse domains due to their specialized agentic roles and collaborative interactions. However, this also amplifies the severity of security risks under MAS attacks. To address this, we introduce MASTER, a novel security research framework for MAS, focusing on diverse Role configurations and Topological structures across various scenarios. MASTER offers an automated construction process for different MAS setups and an information-flow-based interaction paradigm. To tackle MAS security challenges in varied scenarios, we design a scenario-adaptive, extensible attack strategy utilizing role and topological information, which dynamically allocates targeted, domain-specific attack tasks for collaborative agent execution. Our experiments demonstrate that such an attack, leveraging role and topological information, exhibits significant destructive potential across most models. Additionally, we propose corresponding defense strategies, substantially enhancing MAS resilience across diverse scenarios. We anticipate that our framework and findings will provide valuable insights for future research into MAS security challenges. 

---
# Learning without Isolation: Pathway Protection for Continual Learning 

**Authors**: Zhikang Chen, Abudukelimu Wuerkaixi, Sen Cui, Haoxuan Li, Ding Li, Jingfeng Zhang, Bo Han, Gang Niu, Houfang Liu, Yi Yang, Sifan Yang, Changshui Zhang, Tianling Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.18568)  

**Abstract**: Deep networks are prone to catastrophic forgetting during sequential task learning, i.e., losing the knowledge about old tasks upon learning new tasks. To this end, continual learning(CL) has emerged, whose existing methods focus mostly on regulating or protecting the parameters associated with the previous tasks. However, parameter protection is often impractical, since the size of parameters for storing the old-task knowledge increases linearly with the number of tasks, otherwise it is hard to preserve the parameters related to the old-task knowledge. In this work, we bring a dual opinion from neuroscience and physics to CL: in the whole networks, the pathways matter more than the parameters when concerning the knowledge acquired from the old tasks. Following this opinion, we propose a novel CL framework, learning without isolation(LwI), where model fusion is formulated as graph matching and the pathways occupied by the old tasks are protected without being isolated. Thanks to the sparsity of activation channels in a deep network, LwI can adaptively allocate available pathways for a new task, realizing pathway protection and addressing catastrophic forgetting in a parameter-efficient manner. Experiments on popular benchmark datasets demonstrate the superiority of the proposed LwI. 

---
# PacTrain: Pruning and Adaptive Sparse Gradient Compression for Efficient Collective Communication in Distributed Deep Learning 

**Authors**: Yisu Wang, Ruilong Wu, Xinjiao Li, Dirk Kutscher  

**Link**: [PDF](https://arxiv.org/pdf/2505.18563)  

**Abstract**: Large-scale deep neural networks (DNN) exhibit excellent performance for various tasks. As DNNs and datasets grow, distributed training becomes extremely time-consuming and demands larger clusters. A main bottleneck is the resulting gradient aggregation overhead. While gradient compression and sparse collective communication techniques are commonly employed to alleviate network load, many gradient compression schemes do not achieve acceleration of the training process while also preserving accuracy. This paper introduces PacTrain, a novel framework that accelerates distributed training by combining pruning with sparse gradient compression. Active pruning of the neural network makes the model weights and gradients sparse. By ensuring the global knowledge of the gradient sparsity among all distributed training workers, we can perform lightweight compression communication without harming accuracy. We show that the PacTrain compression scheme achieves a near-optimal compression strategy while remaining compatible with the all-reduce primitive. Experimental evaluations show that PacTrain improves training throughput by 1.25 to 8.72 times compared to state-of-the-art compression-enabled systems for representative vision and language models training tasks under bandwidth-constrained conditions. 

---
# From Word to World: Evaluate and Mitigate Culture Bias via Word Association Test 

**Authors**: Xunlian Dai, Li Zhou, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.18562)  

**Abstract**: The human-centered word association test (WAT) serves as a cognitive proxy, revealing sociocultural variations through lexical-semantic patterns. We extend this test into an LLM-adaptive, free-relation task to assess the alignment of large language models (LLMs) with cross-cultural cognition. To mitigate the culture preference, we propose CultureSteer, an innovative approach that integrates a culture-aware steering mechanism to guide semantic representations toward culturally specific spaces. Experiments show that current LLMs exhibit significant bias toward Western cultural (notably in American) schemas at the word association level. In contrast, our model substantially improves cross-cultural alignment, surpassing prompt-based methods in capturing diverse semantic associations. Further validation on culture-sensitive downstream tasks confirms its efficacy in fostering cognitive alignment across cultures. This work contributes a novel methodological paradigm for enhancing cultural awareness in LLMs, advancing the development of more inclusive language technologies. 

---
# Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation 

**Authors**: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby G. Dagher, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18556)  

**Abstract**: Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails. 

---
# Reinforcement Fine-Tuning Powers Reasoning Capability of Multimodal Large Language Models 

**Authors**: Haoyuan Sun, Jiaqi Wu, Bo Xia, Yifu Luo, Yifei Zhao, Kai Qin, Xufei Lv, Tiantian Zhang, Yongzhe Chang, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18536)  

**Abstract**: Standing in 2025, at a critical juncture in the pursuit of Artificial General Intelligence (AGI), reinforcement fine-tuning (RFT) has demonstrated significant potential in enhancing the reasoning capability of large language models (LLMs) and has led to the development of cutting-edge AI models such as OpenAI-o1 and DeepSeek-R1. Moreover, the efficient application of RFT to enhance the reasoning capability of multimodal large language models (MLLMs) has attracted widespread attention from the community. In this position paper, we argue that reinforcement fine-tuning powers the reasoning capability of multimodal large language models. To begin with, we provide a detailed introduction to the fundamental background knowledge that researchers interested in this field should be familiar with. Furthermore, we meticulously summarize the improvements of RFT in powering reasoning capability of MLLMs into five key points: diverse modalities, diverse tasks and domains, better training algorithms, abundant benchmarks and thriving engineering frameworks. Finally, we propose five promising directions for future research that the community might consider. We hope that this position paper will provide valuable insights to the community at this pivotal stage in the advancement toward AGI. Summary of works done on RFT for MLLMs is available at this https URL. 

---
# TS-URGENet: A Three-stage Universal Robust and Generalizable Speech Enhancement Network 

**Authors**: Xiaobin Rong, Dahan Wang, Qinwen Hu, Yushi Wang, Yuxiang Hu, Jing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18533)  

**Abstract**: Universal speech enhancement aims to handle input speech with different distortions and input formats. To tackle this challenge, we present TS-URGENet, a Three-Stage Universal, Robust, and Generalizable speech Enhancement Network. To address various distortions, the proposed system employs a novel three-stage architecture consisting of a filling stage, a separation stage, and a restoration stage. The filling stage mitigates packet loss by preliminarily filling lost regions under noise interference, ensuring signal continuity. The separation stage suppresses noise, reverberation, and clipping distortion to improve speech clarity. Finally, the restoration stage compensates for bandwidth limitation, codec artifacts, and residual packet loss distortion, refining the overall speech quality. Our proposed TS-URGENet achieved outstanding performance in the Interspeech 2025 URGENT Challenge, ranking 2nd in Track 1. 

---
# MRGAgents: A Multi-Agent Framework for Improved Medical Report Generation with Med-LVLMs 

**Authors**: Pengyu Wang, Shuchang Ye, Usman Naseem, Jinman Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.18530)  

**Abstract**: Medical Large Vision-Language Models (Med-LVLMs) have been widely adopted for medical report generation. Despite Med-LVLMs producing state-of-the-art performance, they exhibit a bias toward predicting all findings as normal, leading to reports that overlook critical abnormalities. Furthermore, these models often fail to provide comprehensive descriptions of radiologically relevant regions necessary for accurate diagnosis. To address these challenges, we proposeMedical Report Generation Agents (MRGAgents), a novel multi-agent framework that fine-tunes specialized agents for different disease categories. By curating subsets of the IU X-ray and MIMIC-CXR datasets to train disease-specific agents, MRGAgents generates reports that more effectively balance normal and abnormal findings while ensuring a comprehensive description of clinically relevant regions. Our experiments demonstrate that MRGAgents outperformed the state-of-the-art, improving both report comprehensiveness and diagnostic utility. 

---
# CLaDMoP: Learning Transferrable Models from Successful Clinical Trials via LLMs 

**Authors**: Yiqing Zhang, Xiaozhong Liu, Fabricio Murai  

**Link**: [PDF](https://arxiv.org/pdf/2505.18527)  

**Abstract**: Many existing models for clinical trial outcome prediction are optimized using task-specific loss functions on trial phase-specific data. While this scheme may boost prediction for common diseases and drugs, it can hinder learning of generalizable representations, leading to more false positives/negatives. To address this limitation, we introduce CLaDMoP, a new pre-training approach for clinical trial outcome prediction, alongside the Successful Clinical Trials dataset(SCT), specifically designed for this task. CLaDMoP leverages a Large Language Model-to encode trials' eligibility criteria-linked to a lightweight Drug-Molecule branch through a novel multi-level fusion technique. To efficiently fuse long embeddings across levels, we incorporate a grouping block, drastically reducing computational overhead. CLaDMoP avoids reliance on task-specific objectives by pre-training on a "pair matching" proxy task. Compared to established zero-shot and few-shot baselines, our method significantly improves both PR-AUC and ROC-AUC, especially for phase I and phase II trials. We further evaluate and perform ablation on CLaDMoP after Parameter-Efficient Fine-Tuning, comparing it to state-of-the-art supervised baselines, including MEXA-CTP, on the Trial Outcome Prediction(TOP) benchmark. CLaDMoP achieves up to 10.5% improvement in PR-AUC and 3.6% in ROC-AUC, while attaining comparable F1 score to MEXA-CTP, highlighting its potential for clinical trial outcome prediction. Code and SCT dataset can be downloaded from this https URL. 

---
# Test-Time Adaptation with Binary Feedback 

**Authors**: Taeckyung Lee, Sorn Chottananurak, Junsu Kim, Jinwoo Shin, Taesik Gong, Sung-Ju Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.18514)  

**Abstract**: Deep learning models perform poorly when domain shifts exist between training and test data. Test-time adaptation (TTA) is a paradigm to mitigate this issue by adapting pre-trained models using only unlabeled test samples. However, existing TTA methods can fail under severe domain shifts, while recent active TTA approaches requiring full-class labels are impractical due to high labeling costs. To address this issue, we introduce a new setting of TTA with binary feedback. This setting uses a few binary feedback inputs from annotators to indicate whether model predictions are correct, thereby significantly reducing the labeling burden of annotators. Under the setting, we propose BiTTA, a novel dual-path optimization framework that leverages reinforcement learning to balance binary feedback-guided adaptation on uncertain samples with agreement-based self-adaptation on confident predictions. Experiments show BiTTA achieves 13.3%p accuracy improvements over state-of-the-art baselines, demonstrating its effectiveness in handling severe distribution shifts with minimal labeling effort. The source code is available at this https URL. 

---
# AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking 

**Authors**: Soyoung Yoon, Gyuwan Kim, Gyu-Hwung Cho, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18512)  

**Abstract**: Listwise reranking with large language models (LLMs) enhances top-ranked results in retrieval-based applications. Due to the limit in context size and high inference cost of long context, reranking is typically performed over a fixed size of small subsets, with the final ranking aggregated from these partial results. This fixed computation disregards query difficulty and document distribution, leading to inefficiencies. We propose AcuRank, an adaptive reranking framework that dynamically adjusts both the amount and target of computation based on uncertainty estimates over document relevance. Using a Bayesian TrueSkill model, we iteratively refine relevance estimates until reaching sufficient confidence levels, and our explicit modeling of ranking uncertainty enables principled control over reranking behavior and avoids unnecessary updates to confident predictions. Results on the TREC-DL and BEIR benchmarks show that our method consistently achieves a superior accuracy-efficiency trade-off and scales better with compute than fixed-computation baselines. These results highlight the effectiveness and generalizability of our method across diverse retrieval tasks and LLM-based reranking models. 

---
# How Particle System Theory Enhances Hypergraph Message Passing 

**Authors**: Yixuan Ma, Kai Yi, Pietro Lio, Shi Jin, Yu Guang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18505)  

**Abstract**: Hypergraphs effectively model higher-order relationships in natural phenomena, capturing complex interactions beyond pairwise connections. We introduce a novel hypergraph message passing framework inspired by interacting particle systems, where hyperedges act as fields inducing shared node dynamics. By incorporating attraction, repulsion, and Allen-Cahn forcing terms, particles of varying classes and features achieve class-dependent equilibrium, enabling separability through the particle-driven message passing. We investigate both first-order and second-order particle system equations for modeling these dynamics, which mitigate over-smoothing and heterophily thus can capture complete interactions. The more stable second-order system permits deeper message passing. Furthermore, we enhance deterministic message passing with stochastic element to account for interaction uncertainties. We prove theoretically that our approach mitigates over-smoothing by maintaining a positive lower bound on the hypergraph Dirichlet energy during propagation and thus to enable hypergraph message passing to go deep. Empirically, our models demonstrate competitive performance on diverse real-world hypergraph node classification tasks, excelling on both homophilic and heterophilic datasets. 

---
# G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning 

**Authors**: Xiaojun Guo, Ang Li, Yifei Wang, Stefanie Jegelka, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18499)  

**Abstract**: Although Large Language Models (LLMs) have demonstrated remarkable progress, their proficiency in graph-related tasks remains notably limited, hindering the development of truly general-purpose models. Previous attempts, including pretraining graph foundation models or employing supervised fine-tuning, often face challenges such as the scarcity of large-scale, universally represented graph data. We introduce G1, a simple yet effective approach demonstrating that Reinforcement Learning (RL) on synthetic graph-theoretic tasks can significantly scale LLMs' graph reasoning abilities. To enable RL training, we curate Erdõs, the largest graph reasoning dataset to date comprising 50 diverse graph-theoretic tasks of varying difficulty levels, 100k training data and 5k test data, all drived from real-world graphs. With RL on Erdõs, G1 obtains substantial improvements in graph reasoning, where our finetuned 3B model even outperforms Qwen2.5-72B-Instruct (24x size). RL-trained models also show strong zero-shot generalization to unseen tasks, domains, and graph encoding schemes, including other graph-theoretic benchmarks as well as real-world node classification and link prediction tasks, without compromising general reasoning abilities. Our findings offer an efficient, scalable path for building strong graph reasoners by finetuning LLMs with RL on graph-theoretic tasks, which combines the strengths of pretrained LLM capabilities with abundant, automatically generated synthetic data, suggesting that LLMs possess graph understanding abilities that RL can elicit successfully. 

---
# FedHL: Federated Learning for Heterogeneous Low-Rank Adaptation via Unbiased Aggregation 

**Authors**: Zihao Peng, Jiandian Zeng, Boyuan Li, Guo Li, Shengbo Chen, Tian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18494)  

**Abstract**: Federated Learning (FL) facilitates the fine-tuning of Foundation Models (FMs) using distributed data sources, with Low-Rank Adaptation (LoRA) gaining popularity due to its low communication costs and strong performance. While recent work acknowledges the benefits of heterogeneous LoRA in FL and introduces flexible algorithms to support its implementation, our theoretical analysis reveals a critical gap: existing methods lack formal convergence guarantees due to parameter truncation and biased gradient updates. Specifically, adapting client-specific LoRA ranks necessitates truncating global parameters, which introduces inherent truncation errors and leads to subsequent inaccurate gradient updates that accumulate over training rounds, ultimately degrading performance. To address the above issues, we propose \textbf{FedHL}, a simple yet effective \textbf{Fed}erated Learning framework tailored for \textbf{H}eterogeneous \textbf{L}oRA. By leveraging the full-rank global model as a calibrated aggregation basis, FedHL eliminates the direct truncation bias from initial alignment with client-specific ranks. Furthermore, we derive the theoretically optimal aggregation weights by minimizing the gradient drift term in the convergence upper bound. Our analysis shows that FedHL guarantees $\mathcal{O}(1/\sqrt{T})$ convergence rate, and experiments on multiple real-world datasets demonstrate a 1-3\% improvement over several state-of-the-art methods. 

---
# Synthesizing and Adapting Error Correction Data for Mobile Large Language Model Applications 

**Authors**: Yanxiang Zhang, Zheng Xu, Shanshan Wu, Yuanbo Zhang, Daniel Ramage  

**Link**: [PDF](https://arxiv.org/pdf/2505.18488)  

**Abstract**: Error correction is an important capability when applying large language models (LLMs) to facilitate user typing on mobile devices. In this paper, we use LLMs to synthesize a high-quality dataset of error correction pairs to evaluate and improve LLMs for mobile applications. We first prompt LLMs with error correction domain knowledge to build a scalable and reliable addition to the existing data synthesis pipeline. We then adapt the synthetic data distribution to match the mobile application domain by reweighting the samples. The reweighting model is learnt by predicting (a handful of) live A/B test metrics when deploying LLMs in production, given the LLM performance on offline evaluation data and scores from a small privacy-preserving on-device language model. Finally, we present best practices for mixing our synthetic data with other data sources to improve model performance on error correction in both offline evaluation and production live A/B testing. 

---
# Using Large Language Models to Tackle Fundamental Challenges in Graph Learning: A Comprehensive Survey 

**Authors**: Mengran Li, Pengyu Zhang, Wenbin Xing, Yijia Zheng, Klim Zaporojets, Junzhou Chen, Ronghui Zhang, Yong Zhang, Siyuan Gong, Jia Hu, Xiaolei Ma, Zhiyuan Liu, Paul Groth, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2505.18475)  

**Abstract**: Graphs are a widely used paradigm for representing non-Euclidean data, with applications ranging from social network analysis to biomolecular prediction. Conventional graph learning approaches typically rely on fixed structural assumptions or fully observed data, limiting their effectiveness in more complex, noisy, or evolving settings. Consequently, real-world graph data often violates the assumptions of traditional graph learning methods, in particular, it leads to four fundamental challenges: (1) Incompleteness, real-world graphs have missing nodes, edges, or attributes; (2) Imbalance, the distribution of the labels of nodes or edges and their structures for real-world graphs are highly skewed; (3) Cross-domain Heterogeneity, graphs from different domains exhibit incompatible feature spaces or structural patterns; and (4) Dynamic Instability, graphs evolve over time in unpredictable ways. Recent advances in Large Language Models (LLMs) offer the potential to tackle these challenges by leveraging rich semantic reasoning and external knowledge. This survey provides a comprehensive review of how LLMs can be integrated with graph learning to address the aforementioned challenges. For each challenge, we review both traditional solutions and modern LLM-driven approaches, highlighting how LLMs contribute unique advantages. Finally, we discuss open research questions and promising future directions in this emerging interdisciplinary field. To support further exploration, we have curated a repository of recent advances on graph learning challenges: this https URL. 

---
# Invisible Tokens, Visible Bills: The Urgent Need to Audit Hidden Operations in Opaque LLM Services 

**Authors**: Guoheng Sun, Ziyao Wang, Xuandong Zhao, Bowei Tian, Zheyu Shen, Yexiao He, Jinming Xing, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.18471)  

**Abstract**: Modern large language model (LLM) services increasingly rely on complex, often abstract operations, such as multi-step reasoning and multi-agent collaboration, to generate high-quality outputs. While users are billed based on token consumption and API usage, these internal steps are typically not visible. We refer to such systems as Commercial Opaque LLM Services (COLS). This position paper highlights emerging accountability challenges in COLS: users are billed for operations they cannot observe, verify, or contest. We formalize two key risks: \textit{quantity inflation}, where token and call counts may be artificially inflated, and \textit{quality downgrade}, where providers might quietly substitute lower-cost models or tools. Addressing these risks requires a diverse set of auditing strategies, including commitment-based, predictive, behavioral, and signature-based methods. We further explore the potential of complementary mechanisms such as watermarking and trusted execution environments to enhance verifiability without compromising provider confidentiality. We also propose a modular three-layer auditing framework for COLS and users that enables trustworthy verification across execution, secure logging, and user-facing auditability without exposing proprietary internals. Our aim is to encourage further research and policy development toward transparency, auditability, and accountability in commercial LLM services. 

---
# From Reddit to Generative AI: Evaluating Large Language Models for Anxiety Support Fine-tuned on Social Media Data 

**Authors**: Ugur Kursuncu, Trilok Padhi, Gaurav Sinha, Abdulkadir Erol, Jaya Krishna Mandivarapu, Christopher R. Larrison  

**Link**: [PDF](https://arxiv.org/pdf/2505.18464)  

**Abstract**: The growing demand for accessible mental health support, compounded by workforce shortages and logistical barriers, has led to increased interest in utilizing Large Language Models (LLMs) for scalable and real-time assistance. However, their use in sensitive domains such as anxiety support remains underexamined. This study presents a systematic evaluation of LLMs (GPT and Llama) for their potential utility in anxiety support by using real user-generated posts from the r/Anxiety subreddit for both prompting and fine-tuning. Our approach utilizes a mixed-method evaluation framework incorporating three main categories of criteria: (i) linguistic quality, (ii) safety and trustworthiness, and (iii) supportiveness. Results show that fine-tuning LLMs with naturalistic anxiety-related data enhanced linguistic quality but increased toxicity and bias, and diminished emotional responsiveness. While LLMs exhibited limited empathy, GPT was evaluated as more supportive overall. Our findings highlight the risks of fine-tuning LLMs on unprocessed social media content without mitigation strategies. 

---
# Performance and Generalizability Impacts of Incorporating Geolocation into Deep Learning for Dynamic PM2.5 Estimation 

**Authors**: Morteza Karimzadeh, Zhongying Wang, James L. Crooks  

**Link**: [PDF](https://arxiv.org/pdf/2505.18461)  

**Abstract**: Deep learning models have demonstrated success in geospatial applications, yet quantifying the role of geolocation information in enhancing model performance and geographic generalizability remains underexplored. A new generation of location encoders have emerged with the goal of capturing attributes present at any given location for downstream use in predictive modeling. Being a nascent area of research, their evaluation has remained largely limited to static tasks such as species distributions or average temperature mapping. In this paper, we discuss and quantify the impact of incorporating geolocation into deep learning for a real-world application domain that is characteristically dynamic (with fast temporal change) and spatially heterogeneous at high resolutions: estimating surface-level daily PM2.5 levels using remotely sensed and ground-level data. We build on a recently published deep learning-based PM2.5 estimation model that achieves state-of-the-art performance on data observed in the contiguous United States. We examine three approaches for incorporating geolocation: excluding geolocation as a baseline, using raw geographic coordinates, and leveraging pretrained location encoders. We evaluate each approach under within-region (WR) and out-of-region (OoR) evaluation scenarios. Aggregate performance metrics indicate that while naïve incorporation of raw geographic coordinates improves within-region performance by retaining the interpolative value of geographic location, it can hinder generalizability across regions. In contrast, pretrained location encoders like GeoCLIP enhance predictive performance and geographic generalizability for both WR and OoR scenarios. However, qualitative analysis reveals artifact patterns caused by high-degree basis functions and sparse upstream samples in certain areas, and ablation results indicate varying performance among location encoders... 

---
# A Survey of LLM $\times$ DATA 

**Authors**: Xuanhe Zhou, Junxuan He, Wei Zhou, Haodong Chen, Zirui Tang, Haoyu Zhao, Xin Tong, Guoliang Li, Youmin Chen, Jun Zhou, Zhaojun Sun, Binyuan Hui, Shuo Wang, Conghui He, Zhiyuan Liu, Jingren Zhou, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18458)  

**Abstract**: The integration of large language model (LLM) and data management (DATA) is rapidly redefining both domains. In this survey, we comprehensively review the bidirectional relationships. On the one hand, DATA4LLM, spanning large-scale data processing, storage, and serving, feeds LLMs with high quality, diversity, and timeliness of data required for stages like pre-training, post-training, retrieval-augmented generation, and agentic workflows: (i) Data processing for LLMs includes scalable acquisition, deduplication, filtering, selection, domain mixing, and synthetic augmentation; (ii) Data Storage for LLMs focuses on efficient data and model formats, distributed and heterogeneous storage hierarchies, KV-cache management, and fault-tolerant checkpointing; (iii) Data serving for LLMs tackles challenges in RAG (e.g., knowledge post-processing), LLM inference (e.g., prompt compression, data provenance), and training strategies (e.g., data packing and shuffling). On the other hand, in LLM4DATA, LLMs are emerging as general-purpose engines for data management. We review recent advances in (i) data manipulation, including automatic data cleaning, integration, discovery; (ii) data analysis, covering reasoning over structured, semi-structured, and unstructured data, and (iii) system optimization (e.g., configuration tuning, query rewriting, anomaly diagnosis), powered by LLM techniques like retrieval-augmented prompting, task-specialized fine-tuning, and multi-agent collaboration. 

---
# MPE-TTS: Customized Emotion Zero-Shot Text-To-Speech Using Multi-Modal Prompt 

**Authors**: Zhichao Wu, Yueteng Kang, Songjun Cao, Long Ma, Qiulin Li, Qun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18453)  

**Abstract**: Most existing Zero-Shot Text-To-Speech(ZS-TTS) systems generate the unseen speech based on single prompt, such as reference speech or text descriptions, which limits their flexibility. We propose a customized emotion ZS-TTS system based on multi-modal prompt. The system disentangles speech into the content, timbre, emotion and prosody, allowing emotion prompts to be provided as text, image or speech. To extract emotion information from different prompts, we propose a multi-modal prompt emotion encoder. Additionally, we introduce an prosody predictor to fit the distribution of prosody and propose an emotion consistency loss to preserve emotion information in the predicted prosody. A diffusion-based acoustic model is employed to generate the target mel-spectrogram. Both objective and subjective experiments demonstrate that our system outperforms existing systems in terms of naturalness and similarity. The samples are available at this https URL. 

---
# $μ$-MoE: Test-Time Pruning as Micro-Grained Mixture-of-Experts 

**Authors**: Toshiaki Koike-Akino, Jing Liu, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18451)  

**Abstract**: To tackle the huge computational demand of large foundation models, activation-aware compression techniques without retraining have been introduced. However, since these rely on calibration data, domain shift may arise for unknown downstream tasks. With a computationally efficient calibration, activation-aware pruning can be executed for every prompt adaptively, yet achieving reduced complexity at inference. We formulate it as a mixture of micro-experts, called $\mu$-MoE. Several experiments demonstrate that $\mu$-MoE can dynamically adapt to task/prompt-dependent structured sparsity on the fly. 

---
# Mitigating Context Bias in Domain Adaptation for Object Detection using Mask Pooling 

**Authors**: Hojun Son, Asma Almutairi, Arpan Kusari  

**Link**: [PDF](https://arxiv.org/pdf/2505.18446)  

**Abstract**: Context bias refers to the association between the foreground objects and background during the object detection training process. Various methods have been proposed to minimize the context bias when applying the trained model to an unseen domain, known as domain adaptation for object detection (DAOD). But a principled approach to understand why the context bias occurs and how to remove it has been missing.
In this work, we provide a causal view of the context bias, pointing towards the pooling operation in the convolution network architecture as the possible source of this bias. We present an alternative, Mask Pooling, which uses an additional input of foreground masks, to separate the pooling process in the respective foreground and background regions and show that this process leads the trained model to detect objects in a more robust manner under different domains. We also provide a benchmark designed to create an ultimate test for DAOD, using foregrounds in the presence of absolute random backgrounds, to analyze the robustness of the intended trained models. Through these experiments, we hope to provide a principled approach for minimizing context bias under domain shift. 

---
# Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting 

**Authors**: Zhining Liu, Ze Yang, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Yada Zhu, Hendrik Hamann, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18442)  

**Abstract**: Time-series forecasting plays a critical role in many real-world applications. Although increasingly powerful models have been developed and achieved superior results on benchmark datasets, through a fine-grained sample-level inspection, we find that (i) no single model consistently outperforms others across different test samples, but instead (ii) each model excels in specific cases. These findings prompt us to explore how to adaptively leverage the distinct strengths of various forecasting models for different samples. We introduce TimeFuse, a framework for collective time-series forecasting with sample-level adaptive fusion of heterogeneous models. TimeFuse utilizes meta-features to characterize input time series and trains a learnable fusor to predict optimal model fusion weights for any given input. The fusor can leverage samples from diverse datasets for joint training, allowing it to adapt to a wide variety of temporal patterns and thus generalize to new inputs, even from unseen datasets. Extensive experiments demonstrate the effectiveness of TimeFuse in various long-/short-term forecasting tasks, achieving near-universal improvement over the state-of-the-art individual models. Code is available at this https URL. 

---
# Efficient Long CoT Reasoning in Small Language Models 

**Authors**: Zhaoyang Wang, Jinqi Jiang, Tian Qiu, Hui Liu, Xianfeng Tang, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18440)  

**Abstract**: Recent large reasoning models such as DeepSeek-R1 exhibit strong complex problems solving abilities by generating long chain-of-thought (CoT) reasoning steps. It is challenging to directly train small language models (SLMs) to emerge long CoT. Thus, distillation becomes a practical method to enable SLMs for such reasoning ability. However, the long CoT often contains a lot of redundant contents (e.g., overthinking steps) which may make SLMs hard to learn considering their relatively poor capacity and generalization. To address this issue, we propose a simple-yet-effective method to prune unnecessary steps in long CoT, and then employ an on-policy method for the SLM itself to curate valid and useful long CoT training data. In this way, SLMs can effectively learn efficient long CoT reasoning and preserve competitive performance at the same time. Experimental results across a series of mathematical reasoning benchmarks demonstrate the effectiveness of the proposed method in distilling long CoT reasoning ability into SLMs which maintains the competitive performance but significantly reduces generating redundant reasoning steps. 

---
# TNG-CLIP:Training-Time Negation Data Generation for Negation Awareness of CLIP 

**Authors**: Yuliang Cai, Jesse Thomason, Mohammad Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2505.18434)  

**Abstract**: Vision-language models (VLMs), such as CLIP, have demonstrated strong performance across a range of downstream tasks. However, CLIP is still limited in negation understanding: the ability to recognize the absence or exclusion of a concept. Existing methods address the problem by using a large language model (LLM) to generate large-scale data of image captions containing negation for further fine-tuning CLIP. However, these methods are both time- and compute-intensive, and their evaluations are typically restricted to image-text matching tasks. To expand the horizon, we (1) introduce a training-time negation data generation pipeline such that negation captions are generated during the training stage, which only increases 2.5% extra training time, and (2) we propose the first benchmark, Neg-TtoI, for evaluating text-to-image generation models on prompts containing negation, assessing model's ability to produce semantically accurate images. We show that our proposed method, TNG-CLIP, achieves SOTA performance on diverse negation benchmarks of image-to-text matching, text-to-image retrieval, and image generation. 

---
# Retrieval Augmented Generation-based Large Language Models for Bridging Transportation Cybersecurity Legal Knowledge Gaps 

**Authors**: Khandakar Ashrafi Akbar, Md Nahiyan Uddin, Latifur Khan, Trayce Hockstad, Mizanur Rahman, Mashrur Chowdhury, Bhavani Thuraisingham  

**Link**: [PDF](https://arxiv.org/pdf/2505.18426)  

**Abstract**: As connected and automated transportation systems evolve, there is a growing need for federal and state authorities to revise existing laws and develop new statutes to address emerging cybersecurity and data privacy challenges. This study introduces a Retrieval-Augmented Generation (RAG) based Large Language Model (LLM) framework designed to support policymakers by extracting relevant legal content and generating accurate, inquiry-specific responses. The framework focuses on reducing hallucinations in LLMs by using a curated set of domain-specific questions to guide response generation. By incorporating retrieval mechanisms, the system enhances the factual grounding and specificity of its outputs. Our analysis shows that the proposed RAG-based LLM outperforms leading commercial LLMs across four evaluation metrics: AlignScore, ParaScore, BERTScore, and ROUGE, demonstrating its effectiveness in producing reliable and context-aware legal insights. This approach offers a scalable, AI-driven method for legislative analysis, supporting efforts to update legal frameworks in line with advancements in transportation technologies. 

---
# How We Won the ISLES'24 Challenge by Preprocessing 

**Authors**: Tianyi Ren, Juampablo E. Heras Rivera, Hitender Oswal, Yutong Pan, William Henry, Jacob Ruzevick, Mehmet Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2505.18424)  

**Abstract**: Stroke is among the top three causes of death worldwide, and accurate identification of stroke lesion boundaries is critical for diagnosis and treatment. Supervised deep learning methods have emerged as the leading solution for stroke lesion segmentation but require large, diverse, and annotated datasets. The ISLES'24 challenge addresses this need by providing longitudinal stroke imaging data, including CT scans taken on arrival to the hospital and follow-up MRI taken 2-9 days from initial arrival, with annotations derived from follow-up MRI. Importantly, models submitted to the ISLES'24 challenge are evaluated using only CT inputs, requiring prediction of lesion progression that may not be visible in CT scans for segmentation. Our winning solution shows that a carefully designed preprocessing pipeline including deep-learning-based skull stripping and custom intensity windowing is beneficial for accurate segmentation. Combined with a standard large residual nnU-Net architecture for segmentation, this approach achieves a mean test Dice of 28.5 with a standard deviation of 21.27. 

---
# Reinforcement Learning for Ballbot Navigation in Uneven Terrain 

**Authors**: Achkan Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18417)  

**Abstract**: Ballbot (i.e. Ball balancing robot) navigation usually relies on methods rooted in control theory (CT), and works that apply Reinforcement learning (RL) to the problem remain rare while generally being limited to specific subtasks (e.g. balance recovery). Unlike CT based methods, RL does not require (simplifying) assumptions about environment dynamics (e.g. the absence of slippage between the ball and the floor). In addition to this increased accuracy in modeling, RL agents can easily be conditioned on additional observations such as depth-maps without the need for explicit formulations from first principles, leading to increased adaptivity. Despite those advantages, there has been little to no investigation into the capabilities, data-efficiency and limitations of RL based methods for ballbot control and navigation. Furthermore, there is a notable absence of an open-source, RL-friendly simulator for this task. In this paper, we present an open-source ballbot simulation based on MuJoCo, and show that with appropriate conditioning on exteroceptive observations as well as reward shaping, policies learned by classical model-free RL methods are capable of effectively navigating through randomly generated uneven terrain, using a reasonable amount of data (four to five hours on a system operating at 500hz). 

---
# LatentLLM: Attention-Aware Joint Tensor Compression 

**Authors**: Toshiaki Koike-Akino, Xiangyu Chen, Jing Liu, Ye Wang, Wang, Matthew Brand  

**Link**: [PDF](https://arxiv.org/pdf/2505.18413)  

**Abstract**: Modern foundation models such as large language models (LLMs) and large multi-modal models (LMMs) require a massive amount of computational and memory resources. We propose a new framework to convert such LLMs/LMMs into a reduced-dimension latent structure. Our method extends a local activation-aware tensor decomposition to a global attention-aware joint tensor de-composition. Our framework can significantly improve the model accuracy over the existing model compression methods when reducing the latent dimension to realize computationally/memory-efficient LLMs/LLMs. We show the benefit on several benchmark including multi-modal reasoning tasks. 

---
# KL-regularization Itself is Differentially Private in Bandits and RLHF 

**Authors**: Yizhou Zhang, Kishan Panaganti, Laixi Shi, Juba Ziani, Adam Wierman  

**Link**: [PDF](https://arxiv.org/pdf/2505.18407)  

**Abstract**: Differential Privacy (DP) provides a rigorous framework for privacy, ensuring the outputs of data-driven algorithms remain statistically indistinguishable across datasets that differ in a single entry. While guaranteeing DP generally requires explicitly injecting noise either to the algorithm itself or to its outputs, the intrinsic randomness of existing algorithms presents an opportunity to achieve DP ``for free''. In this work, we explore the role of regularization in achieving DP across three different decision-making problems: multi-armed bandits, linear contextual bandits, and reinforcement learning from human feedback (RLHF), in offline data settings. We show that adding KL-regularization to the learning objective (a common approach in optimization algorithms) makes the action sampled from the resulting stochastic policy itself differentially private. This offers a new route to privacy guarantees without additional noise injection, while also preserving the inherent advantage of regularization in enhancing performance. 

---
# Thought calibration: Efficient and confident test-time scaling 

**Authors**: Menghua Wu, Cai Zhou, Stephen Bates, Tommi Jaakkola  

**Link**: [PDF](https://arxiv.org/pdf/2505.18404)  

**Abstract**: Reasoning large language models achieve impressive test-time scaling by thinking for longer, but this performance gain comes at significant compute cost. Directly limiting test-time budget hurts overall performance, but not all problems are equally difficult. We propose thought calibration to decide dynamically when thinking can be terminated. To calibrate our decision rule, we view a language model's growing body of thoughts as a nested sequence of reasoning trees, where the goal is to identify the point at which novel reasoning plateaus. We realize this framework through lightweight probes that operate on top of the language model's hidden representations, which are informative of both the reasoning structure and overall consistency of response. Based on three reasoning language models and four datasets, thought calibration preserves model performance with up to a 60% reduction in thinking tokens on in-distribution data, and up to 20% in out-of-distribution data. 

---
# Taming Diffusion for Dataset Distillation with High Representativeness 

**Authors**: Lin Zhao, Yushu Wu, Xinru Jiang, Jianyang Gu, Yanzhi Wang, Xiaolin Xu, Pu Zhao, Xue Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.18399)  

**Abstract**: Recent deep learning models demand larger datasets, driving the need for dataset distillation to create compact, cost-efficient datasets while maintaining performance. Due to the powerful image generation capability of diffusion, it has been introduced to this field for generating distilled images. In this paper, we systematically investigate issues present in current diffusion-based dataset distillation methods, including inaccurate distribution matching, distribution deviation with random noise, and separate sampling. Building on this, we propose D^3HR, a novel diffusion-based framework to generate distilled datasets with high representativeness. Specifically, we adopt DDIM inversion to map the latents of the full dataset from a low-normality latent domain to a high-normality Gaussian domain, preserving information and ensuring structural consistency to generate representative latents for the distilled dataset. Furthermore, we propose an efficient sampling scheme to better align the representative latents with the high-normality Gaussian distribution. Our comprehensive experiments demonstrate that D^3HR can achieve higher accuracy across different model architectures compared with state-of-the-art baselines in dataset distillation. Source code: this https URL. 

---
# Towards Anonymous Neural Network Inference 

**Authors**: Liao Peiyuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18398)  

**Abstract**: We introduce funion, a system providing end-to-end sender-receiver unlinkability for neural network inference. By leveraging the Pigeonhole storage protocol and BACAP (blinding-and-capability) scheme from the Echomix anonymity system, funion inherits the provable security guarantees of modern mixnets. Users can anonymously store input tensors in pseudorandom storage locations, commission compute services to process them via the neural network, and retrieve results with no traceable connection between input and output parties. This store-compute-store paradigm masks both network traffic patterns and computational workload characteristics, while quantizing execution timing into public latency buckets. Our security analysis demonstrates that funion inherits the strong metadata privacy guarantees of Echomix under largely the same trust assumptions, while introducing acceptable overhead for production-scale workloads. Our work paves the way towards an accessible platform where users can submit fully anonymized inference queries to cloud services. 

---
# An Outlook on the Opportunities and Challenges of Multi-Agent AI Systems 

**Authors**: Fangqiao Tian, An Luo, Jin Du, Xun Xian, Robert Specht, Ganghua Wang, Xuan Bi, Jiawei Zhou, Jayanth Srinivasa, Ashish Kundu, Charles Fleming, Rui Zhang, Zirui Liu, Mingyi Hong, Jie Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.18397)  

**Abstract**: Multi-agent AI systems (MAS) offer a promising framework for distributed intelligence, enabling collaborative reasoning, planning, and decision-making across autonomous agents. This paper provides a systematic outlook on the current opportunities and challenges of MAS, drawing insights from recent advances in large language models (LLMs), federated optimization, and human-AI interaction. We formalize key concepts including agent topology, coordination protocols, and shared objectives, and identify major risks such as dependency, misalignment, and vulnerabilities arising from training data overlap. Through a biologically inspired simulation and comprehensive theoretical framing, we highlight critical pathways for developing robust, scalable, and secure MAS in real-world settings. 

---
# Applications of Modular Co-Design for De Novo 3D Molecule Generation 

**Authors**: Danny Reidenbach, Filipp Nikitin, Olexandr Isayev, Saee Paliwal  

**Link**: [PDF](https://arxiv.org/pdf/2505.18392)  

**Abstract**: De novo 3D molecule generation is a pivotal task in drug discovery. However, many recent geometric generative models struggle to produce high-quality 3D structures, even if they maintain 2D validity and topological stability. To tackle this issue and enhance the learning of effective molecular generation dynamics, we present Megalodon-a family of scalable transformer models. These models are enhanced with basic equivariant layers and trained using a joint continuous and discrete denoising co-design objective. We assess Megalodon's performance on established molecule generation benchmarks and introduce new 3D structure benchmarks that evaluate a model's capability to generate realistic molecular structures, particularly focusing on energetics. We show that Megalodon achieves state-of-the-art results in 3D molecule generation, conditional structure generation, and structure energy benchmarks using diffusion and flow matching. Furthermore, doubling the number of parameters in Megalodon to 40M significantly enhances its performance, generating up to 49x more valid large molecules and achieving energy levels that are 2-10x lower than those of the best prior generative models. 

---
# Human-Centered AI Communication in Co-Creativity: An Initial Framework and Insights 

**Authors**: Jeba Rezwana, Corey Ford  

**Link**: [PDF](https://arxiv.org/pdf/2505.18385)  

**Abstract**: Effective communication between AI and humans is essential for successful human-AI co-creation. However, many current co-creative AI systems lack effective communication, which limits their potential for collaboration. This paper presents the initial design of the Framework for AI Communication (FAICO) for co-creative AI, developed through a systematic review of 107 full-length papers. FAICO presents key aspects of AI communication and their impact on user experience, offering preliminary guidelines for designing human-centered AI communication. To improve the framework, we conducted a preliminary study with two focus groups involving skilled individuals in AI, HCI, and design. These sessions sought to understand participants' preferences for AI communication, gather their perceptions of the framework, collect feedback for refinement, and explore its use in co-creative domains like collaborative writing and design. Our findings reveal a preference for a human-AI feedback loop over linear communication and emphasize the importance of context in fostering mutual understanding. Based on these insights, we propose actionable strategies for applying FAICO in practice and future directions, marking the first step toward developing comprehensive guidelines for designing effective human-centered AI communication in co-creation. 

---
# Dynamic Risk Assessments for Offensive Cybersecurity Agents 

**Authors**: Boyi Wei, Benedikt Stroebl, Jiacen Xu, Joie Zhang, Zhou Li, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2505.18384)  

**Abstract**: Foundation models are increasingly becoming better autonomous programmers, raising the prospect that they could also automate dangerous offensive cyber-operations. Current frontier model audits probe the cybersecurity risks of such agents, but most fail to account for the degrees of freedom available to adversaries in the real world. In particular, with strong verifiers and financial incentives, agents for offensive cybersecurity are amenable to iterative improvement by would-be adversaries. We argue that assessments should take into account an expanded threat model in the context of cybersecurity, emphasizing the varying degrees of freedom that an adversary may possess in stateful and non-stateful environments within a fixed compute budget. We show that even with a relatively small compute budget (8 H100 GPU Hours in our study), adversaries can improve an agent's cybersecurity capability on InterCode CTF by more than 40\% relative to the baseline -- without any external assistance. These results highlight the need to evaluate agents' cybersecurity risk in a dynamic manner, painting a more representative picture of risk. 

---
# SP2RINT: Spatially-Decoupled Physics-Inspired Progressive Inverse Optimization for Scalable, PDE-Constrained Meta-Optical Neural Network Training 

**Authors**: Pingchuan Ma, Ziang Yin, Qi Jing, Zhengqi Gao, Nicholas Gangi, Boyang Zhang, Tsung-Wei Huang, Zhaoran Huang, Duane S. Boning, Yu Yao, Jiaqi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18377)  

**Abstract**: DONNs harness the physics of light propagation for efficient analog computation, with applications in AI and signal processing. Advances in nanophotonic fabrication and metasurface-based wavefront engineering have opened new pathways to realize high-capacity DONNs across various spectral regimes. Training such DONN systems to determine the metasurface structures remains challenging. Heuristic methods are fast but oversimplify metasurfaces modulation, often resulting in physically unrealizable designs and significant performance degradation. Simulation-in-the-loop training methods directly optimize a physically implementable metasurface using adjoint methods during end-to-end DONN training, but are inherently computationally prohibitive and this http URL address these limitations, we propose SP2RINT, a spatially decoupled, progressive training framework that formulates DONN training as a PDE-constrained learning problem. Metasurface responses are first relaxed into freely trainable transfer matrices with a banded structure. We then progressively enforce physical constraints by alternating between transfer matrix training and adjoint-based inverse design, avoiding per-iteration PDE solves while ensuring final physical realizability. To further reduce runtime, we introduce a physics-inspired, spatially decoupled inverse design strategy based on the natural locality of field interactions. This approach partitions the metasurface into independently solvable patches, enabling scalable and parallel inverse design with system-level calibration. Evaluated across diverse DONN training tasks, SP2RINT achieves digital-comparable accuracy while being 1825 times faster than simulation-in-the-loop approaches. By bridging the gap between abstract DONN models and implementable photonic hardware, SP2RINT enables scalable, high-performance training of physically realizable meta-optical neural systems. 

---
# Next-token pretraining implies in-context learning 

**Authors**: Paul M. Riechers, Henry R. Bigelow, Eric A. Alt, Adam Shai  

**Link**: [PDF](https://arxiv.org/pdf/2505.18373)  

**Abstract**: We argue that in-context learning (ICL) predictably arises from standard self-supervised next-token pretraining, rather than being an exotic emergent property. This work establishes the foundational principles of this emergence by focusing on in-distribution ICL, demonstrating how models necessarily adapt to context when trained on token sequences, especially from non-ergodic sources. Our information-theoretic framework precisely predicts these in-distribution ICL dynamics (i.e., context-dependent loss reduction). We verify this with experiments using synthetic datasets of differing types of correlational structure, reproducing characteristic phenomena like phase transitions in training loss for induction head formation and power-law scaling of in-context loss. We further show that a model's in-context performance on any task is mathematically coupled to the ensemble of tasks seen in pretraining, offering a fundamental explanation, grounded in architecture- and modality-independent principles, for such inference-time learning. 

---
# Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications 

**Authors**: Riley Simmons-Edler, Jean Dong, Paul Lushenko, Kanaka Rajan, Ryan P. Badman  

**Link**: [PDF](https://arxiv.org/pdf/2505.18371)  

**Abstract**: Military weapon systems and command-and-control infrastructure augmented by artificial intelligence (AI) have seen rapid development and deployment in recent years. However, the sociotechnical impacts of AI on combat systems, military decision-making, and the norms of warfare have been understudied. We focus on a specific subset of lethal autonomous weapon systems (LAWS) that use AI for targeting or battlefield decisions. We refer to this subset as AI-powered lethal autonomous weapon systems (AI-LAWS) and argue that they introduce novel risks -- including unanticipated escalation, poor reliability in unfamiliar environments, and erosion of human oversight -- all of which threaten both military effectiveness and the openness of AI research. These risks cannot be addressed by high-level policy alone; effective regulation must be grounded in the technical behavior of AI models. We argue that AI researchers must be involved throughout the regulatory lifecycle. Thus, we propose a clear, behavior-based definition of AI-LAWS -- systems that introduce unique risks through their use of modern AI -- as a foundation for technically grounded regulation, given that existing frameworks do not distinguish them from conventional LAWS. Using this definition, we propose several technically-informed policy directions and invite greater participation from the AI research community in military AI policy discussions. 

---
# Small Models, Smarter Learning: The Power of Joint Task Training 

**Authors**: Csaba Both, Benjamin Hoover, Hendrik Strobelt, Dmitry Krotov, Daniel Karl I. Weidele, Mauro Martino, Nima Dehmamy  

**Link**: [PDF](https://arxiv.org/pdf/2505.18369)  

**Abstract**: The ability of a model to learn a task depends strongly on both the task difficulty and the model size. We aim to understand how task difficulty relates to the minimum number of parameters required for learning specific tasks in small transformer models. Our study focuses on the ListOps dataset, which consists of nested mathematical operations. We gradually increase task difficulty by introducing new operations or combinations of operations into the training data. We observe that sum modulo n is the hardest to learn. Curiously, when combined with other operations such as maximum and median, the sum operation becomes easier to learn and requires fewer parameters. We show that joint training not only improves performance but also leads to qualitatively different model behavior. We show evidence that models trained only on SUM might be memorizing and fail to capture the number structure in the embeddings. In contrast, models trained on a mixture of SUM and other operations exhibit number-like representations in the embedding space, and a strong ability to distinguish parity. Furthermore, the SUM-only model relies more heavily on its feedforward layers, while the jointly trained model activates the attention mechanism more. Finally, we show that learning pure SUM can be induced in models below the learning threshold of pure SUM, by pretraining them on MAX+MED. Our findings indicate that emergent abilities in language models depend not only on model size, but also the training curriculum. 

---
# Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems 

**Authors**: Hansa Meghwani, Amit Agarwal, Priyaranjan Pattnayak, Hitesh Laxmichand Patel, Srikant Panda  

**Link**: [PDF](https://arxiv.org/pdf/2505.18366)  

**Abstract**: Enterprise search systems often struggle to retrieve accurate, domain-specific information due to semantic mismatches and overlapping terminologies. These issues can degrade the performance of downstream applications such as knowledge management, customer support, and retrieval-augmented generation agents. To address this challenge, we propose a scalable hard-negative mining framework tailored specifically for domain-specific enterprise data. Our approach dynamically selects semantically challenging but contextually irrelevant documents to enhance deployed re-ranking models.
Our method integrates diverse embedding models, performs dimensionality reduction, and uniquely selects hard negatives, ensuring computational efficiency and semantic precision. Evaluation on our proprietary enterprise corpus (cloud services domain) demonstrates substantial improvements of 15\% in MRR@3 and 19\% in MRR@10 compared to state-of-the-art baselines and other negative sampling techniques. Further validation on public domain-specific datasets (FiQA, Climate Fever, TechQA) confirms our method's generalizability and readiness for real-world applications. 

---
# SchemaGraphSQL: Efficient Schema Linking with Pathfinding Graph Algorithms for Text-to-SQL on Large-Scale Databases 

**Authors**: AmirHossein Safdarian, Milad Mohammadi, Ehsan Jahanbakhsh, Mona Shahamat Naderi, Heshaam Faili  

**Link**: [PDF](https://arxiv.org/pdf/2505.18363)  

**Abstract**: Text-to-SQL systems translate natural language questions into executable SQL queries, and recent progress with large language models (LLMs) has driven substantial improvements in this task. Schema linking remains a critical component in Text-to-SQL systems, reducing prompt size for models with narrow context windows and sharpening model focus even when the entire schema fits. We present a zero-shot, training-free schema linking approach that first constructs a schema graph based on foreign key relations, then uses a single prompt to Gemini 2.5 Flash to extract source and destination tables from the user query, followed by applying classical path-finding algorithms and post-processing to identify the optimal sequence of tables and columns that should be joined, enabling the LLM to generate more accurate SQL queries. Despite being simple, cost-effective, and highly scalable, our method achieves state-of-the-art results on the BIRD benchmark, outperforming previous specialized, fine-tuned, and complex multi-step LLM-based approaches. We conduct detailed ablation studies to examine the precision-recall trade-off in our framework. Additionally, we evaluate the execution accuracy of our schema filtering method compared to other approaches across various model sizes. 

---
# Hamiltonian Theory and Computation of Optimal Probability Density Control in High Dimensions 

**Authors**: Nathan Gaby, Xiaojing Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.18362)  

**Abstract**: We develop a general theoretical framework for optimal probability density control and propose a numerical algorithm that is scalable to solve the control problem in high dimensions. Specifically, we establish the Pontryagin Maximum Principle (PMP) for optimal density control and construct the Hamilton-Jacobi-Bellman (HJB) equation of the value functional through rigorous derivations without any concept from Wasserstein theory. To solve the density control problem numerically, we propose to use reduced-order models, such as deep neural networks (DNNs), to parameterize the control vector-field and the adjoint function, which allows us to tackle problems defined on high-dimensional state spaces. We also prove several convergence properties of the proposed algorithm. Numerical results demonstrate promising performances of our algorithm on a variety of density control problems with obstacles and nonlinear interaction challenges in high dimensions. 

---
# Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain 

**Authors**: Trinity Chung, Yuchen Shen, Nathan C. L. Kong, Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18361)  

**Abstract**: Tactile sensing remains far less understood in neuroscience and less effective in artificial systems compared to more mature modalities such as vision and language. We bridge these gaps by introducing a novel Encoder-Attender-Decoder (EAD) framework to systematically explore the space of task-optimized temporal neural networks trained on realistic tactile input sequences from a customized rodent whisker-array simulator. We identify convolutional recurrent neural networks (ConvRNNs) as superior encoders to purely feedforward and state-space architectures for tactile categorization. Crucially, these ConvRNN-encoder-based EAD models achieve neural representations closely matching rodent somatosensory cortex, saturating the explainable neural variability and revealing a clear linear relationship between supervised categorization performance and neural alignment. Furthermore, contrastive self-supervised ConvRNN-encoder-based EADs, trained with tactile-specific augmentations, match supervised neural fits, serving as an ethologically-relevant, label-free proxy.
For neuroscience, our findings highlight nonlinear recurrent processing as important for general-purpose tactile representations in somatosensory cortex, providing the first quantitative characterization of the underlying inductive biases in this system. For embodied AI, our results emphasize the importance of recurrent EAD architectures to handle realistic tactile inputs, along with tailored self-supervised learning methods for achieving robust tactile perception with the same type of sensors animals use to sense in unstructured environments. 

---
# The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs 

**Authors**: Lucas Bandarkar, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.18356)  

**Abstract**: Large language models (LLMs) still struggle across tasks outside of high-resource languages. In this work, we investigate cross-lingual transfer to lower-resource languages where task-specific post-training data is scarce. Building on prior work, we first validate that the subsets of model parameters that matter most for mathematical reasoning and multilingual capabilities are distinctly non-overlapping. To exploit this implicit separability between task and target language parameterization, we develop and analyze numerous modular frameworks to improve the composition of the two during fine-tuning. These methods generally employ freezing parameters or post hoc model merging to assign math and language improvement to different key parts of the LLM. In the absence of in-language math data, we demonstrate that the modular approaches successfully improve upon baselines across three languages, four models, and two fine-tuning paradigms (full and LoRA). Furthermore, we identify the most consistently successful modular method to be fine-tuning separate language and math experts and model merging via Layer-Swapping, somewhat surprisingly. We offer possible explanations for this result via recent works on the linearity of task vectors. We further explain this by empirically showing that reverting less useful fine-tuning updates after training often outperforms freezing them from the start. 

---
# Task Specific Pruning with LLM-Sieve: How Many Parameters Does Your Task Really Need? 

**Authors**: Waleed Reda, Abhinav Jangda, Krishna Chintalapudi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18350)  

**Abstract**: As Large Language Models (LLMs) are increasingly being adopted for narrow tasks - such as medical question answering or sentiment analysis - and deployed in resource-constrained settings, a key question arises: how many parameters does a task actually need? In this work, we present LLM-Sieve, the first comprehensive framework for task-specific pruning of LLMs that achieves 20-75% parameter reduction with only 1-5% accuracy degradation across diverse domains. Unlike prior methods that apply uniform pruning or rely on low-rank approximations of weight matrices or inputs in isolation, LLM-Sieve (i) learns task-aware joint projections to better approximate output behavior, and (ii) employs a Genetic Algorithm to discover differentiated pruning levels for each matrix. LLM-Sieve is fully compatible with LoRA fine-tuning and quantization, and uniquely demonstrates strong generalization across datasets within the same task domain. Together, these results establish a practical and robust mechanism to generate smaller performant task-specific models. 

---
# The Cell Must Go On: Agar.io for Continual Reinforcement Learning 

**Authors**: Mohamed A. Mohamed, Kateryna Nekhomiazh, Vedant Vyas, Marcos M. Jose, Andrew Patterson, Marlos C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2505.18347)  

**Abstract**: Continual reinforcement learning (RL) concerns agents that are expected to learn continually, rather than converge to a policy that is then fixed for evaluation. Such an approach is well suited to environments the agent perceives as changing, which renders any static policy ineffective over time. The few simulators explicitly designed for empirical research in continual RL are often limited in scope or complexity, and it is now common for researchers to modify episodic RL environments by artificially incorporating abrupt task changes during interaction. In this paper, we introduce AgarCL, a research platform for continual RL that allows for a progression of increasingly sophisticated behaviour. AgarCL is based on the game this http URL, a non-episodic, high-dimensional problem featuring stochastic, ever-evolving dynamics, continuous actions, and partial observability. Additionally, we provide benchmark results reporting the performance of DQN, PPO, and SAC in both the primary, challenging continual RL problem, and across a suite of smaller tasks within AgarCL, each of which isolates aspects of the full environment and allow us to characterize the challenges posed by different aspects of the game. 

---
# Sample Complexity of Diffusion Model Training Without Empirical Risk Minimizer Access 

**Authors**: Mudit Gaur, Prashant Trivedi, Sasidhar Kunapuli, Amrit Singh Bedi, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2505.18344)  

**Abstract**: Diffusion models have demonstrated state-of-the-art performance across vision, language, and scientific domains. Despite their empirical success, prior theoretical analyses of the sample complexity suffer from poor scaling with input data dimension or rely on unrealistic assumptions such as access to exact empirical risk minimizers. In this work, we provide a principled analysis of score estimation, establishing a sample complexity bound of $\widetilde{\mathcal{O}}(\epsilon^{-6})$. Our approach leverages a structured decomposition of the score estimation error into statistical, approximation, and optimization errors, enabling us to eliminate the exponential dependence on neural network parameters that arises in prior analyses. It is the first such result which achieves sample complexity bounds without assuming access to the empirical risk minimizer of score function estimation loss. 

---
# CrashAgent: Crash Scenario Generation via Multi-modal Reasoning 

**Authors**: Miao Li, Wenhao Ding, Haohong Lin, Yiqi Lyu, Yihang Yao, Yuyou Zhang, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18341)  

**Abstract**: Training and evaluating autonomous driving algorithms requires a diverse range of scenarios. However, most available datasets predominantly consist of normal driving behaviors demonstrated by human drivers, resulting in a limited number of safety-critical cases. This imbalance, often referred to as a long-tail distribution, restricts the ability of driving algorithms to learn from crucial scenarios involving risk or failure, scenarios that are essential for humans to develop driving skills efficiently. To generate such scenarios, we utilize Multi-modal Large Language Models to convert crash reports of accidents into a structured scenario format, which can be directly executed within simulations. Specifically, we introduce CrashAgent, a multi-agent framework designed to interpret multi-modal real-world traffic crash reports for the generation of both road layouts and the behaviors of the ego vehicle and surrounding traffic participants. We comprehensively evaluate the generated crash scenarios from multiple perspectives, including the accuracy of layout reconstruction, collision rate, and diversity. The resulting high-quality and large-scale crash dataset will be publicly available to support the development of safe driving algorithms in handling safety-critical situations. 

---
# A Critical Evaluation of Defenses against Prompt Injection Attacks 

**Authors**: Yuqi Jia, Zedian Shao, Yupei Liu, Jinyuan Jia, Dawn Song, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18333)  

**Abstract**: Large Language Models (LLMs) are vulnerable to prompt injection attacks, and several defenses have recently been proposed, often claiming to mitigate these attacks successfully. However, we argue that existing studies lack a principled approach to evaluating these defenses. In this paper, we argue the need to assess defenses across two critical dimensions: (1) effectiveness, measured against both existing and adaptive prompt injection attacks involving diverse target and injected prompts, and (2) general-purpose utility, ensuring that the defense does not compromise the foundational capabilities of the LLM. Our critical evaluation reveals that prior studies have not followed such a comprehensive evaluation methodology. When assessed using this principled approach, we show that existing defenses are not as successful as previously reported. This work provides a foundation for evaluating future defenses and guiding their development. Our code and data are available at: this https URL. 

---
# PerMedCQA: Benchmarking Large Language Models on Medical Consumer Question Answering in Persian Language 

**Authors**: Naghmeh Jamali, Milad Mohammadi, Danial Baledi, Zahra Rezvani, Hesham Faili  

**Link**: [PDF](https://arxiv.org/pdf/2505.18331)  

**Abstract**: Medical consumer question answering (CQA) is crucial for empowering patients by providing personalized and reliable health information. Despite recent advances in large language models (LLMs) for medical QA, consumer-oriented and multilingual resources, particularly in low-resource languages like Persian, remain sparse. To bridge this gap, we present PerMedCQA, the first Persian-language benchmark for evaluating LLMs on real-world, consumer-generated medical questions. Curated from a large medical QA forum, PerMedCQA contains 68,138 question-answer pairs, refined through careful data cleaning from an initial set of 87,780 raw entries. We evaluate several state-of-the-art multilingual and instruction-tuned LLMs, utilizing MedJudge, a novel rubric-based evaluation framework driven by an LLM grader, validated against expert human annotators. Our results highlight key challenges in multilingual medical QA and provide valuable insights for developing more accurate and context-aware medical assistance systems. The data is publicly available on this https URL 

---
# Architectural Backdoors for Within-Batch Data Stealing and Model Inference Manipulation 

**Authors**: Nicolas Küchler, Ivan Petrov, Conrad Grobler, Ilia Shumailov  

**Link**: [PDF](https://arxiv.org/pdf/2505.18323)  

**Abstract**: For nearly a decade the academic community has investigated backdoors in neural networks, primarily focusing on classification tasks where adversaries manipulate the model prediction. While demonstrably malicious, the immediate real-world impact of such prediction-altering attacks has remained unclear. In this paper we introduce a novel and significantly more potent class of backdoors that builds upon recent advancements in architectural backdoors. We demonstrate how these backdoors can be specifically engineered to exploit batched inference, a common technique for hardware utilization, enabling large-scale user data manipulation and theft. By targeting the batching process, these architectural backdoors facilitate information leakage between concurrent user requests and allow attackers to fully control model responses directed at other users within the same batch. In other words, an attacker who can change the model architecture can set and steal model inputs and outputs of other users within the same batch. We show that such attacks are not only feasible but also alarmingly effective, can be readily injected into prevalent model architectures, and represent a truly malicious threat to user privacy and system integrity. Critically, to counteract this new class of vulnerabilities, we propose a deterministic mitigation strategy that provides formal guarantees against this new attack vector, unlike prior work that relied on Large Language Models to find the backdoors. Our mitigation strategy employs a novel Information Flow Control mechanism that analyzes the model graph and proves non-interference between different user inputs within the same batch. Using our mitigation strategy we perform a large scale analysis of models hosted through Hugging Face and find over 200 models that introduce (unintended) information leakage between batch entries due to the use of dynamic quantization. 

---
# Is It Bad to Work All the Time? Cross-Cultural Evaluation of Social Norm Biases in GPT-4 

**Authors**: Zhuozhuo Joy Liu, Farhan Samir, Mehar Bhatia, Laura K. Nelson, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2505.18322)  

**Abstract**: LLMs have been demonstrated to align with the values of Western or North American cultures. Prior work predominantly showed this effect through leveraging surveys that directly ask (originally people and now also LLMs) about their values. However, it is hard to believe that LLMs would consistently apply those values in real-world scenarios. To address that, we take a bottom-up approach, asking LLMs to reason about cultural norms in narratives from different cultures. We find that GPT-4 tends to generate norms that, while not necessarily incorrect, are significantly less culture-specific. In addition, while it avoids overtly generating stereotypes, the stereotypical representations of certain cultures are merely hidden rather than suppressed in the model, and such stereotypes can be easily recovered. Addressing these challenges is a crucial step towards developing LLMs that fairly serve their diverse user base. 

---
# COLORA: Efficient Fine-Tuning for Convolutional Models with a Study Case on Optical Coherence Tomography Image Classification 

**Authors**: Mariano Rivera, Angello Hoyos  

**Link**: [PDF](https://arxiv.org/pdf/2505.18315)  

**Abstract**: We introduce the Convolutional Low-Rank Adaptation (CoLoRA) method, designed explicitly to overcome the inefficiencies found in current CNN fine-tuning methods. CoLoRA can be seen as a natural extension of the convolutional architectures of the Low-Rank Adaptation (LoRA) technique. We demonstrate the capabilities of our method by developing and evaluating models using the widely adopted CNN backbone pre-trained on ImageNet. We observed that this strategy results in a stable and accurate coarse-tuning procedure. Moreover, this strategy is computationally efficient and significantly reduces the number of parameters required for fine-tuning compared to traditional methods. Furthermore, our method substantially improves the speed and stability of training. Our case study focuses on classifying retinal diseases from optical coherence tomography (OCT) images, specifically using the OCTMNIST dataset. Experimental results demonstrate that a CNN backbone fine-tuned with CoLoRA surpasses nearly 1\% in accuracy. Such a performance is comparable to the Vision Transformer, State-space discrete, and Kolmogorov-Arnold network models. 

---
# Efficient Algorithms for Electing Successive Committees 

**Authors**: Pallavi Jain, Andrzej Kaczmarczyk  

**Link**: [PDF](https://arxiv.org/pdf/2505.18287)  

**Abstract**: In a recently introduced model of successive committee elections (Bredereck et al., AAAI-20) for a given set of ordinal or approval preferences one aims to find a sequence of a given length of "best" same-size committees such that each candidate is a member of a limited number of consecutive committees. However, the practical usability of this model remains limited, as the described task turns out to be NP-hard for most selection criteria already for seeking committees of size three. Non-trivial or somewhat efficient algorithms for these cases are lacking too. Motivated by a desire to unlock the full potential of the described temporal model of committee elections, we devise (parameterized) algorithms that effectively solve the mentioned hard cases in realistic scenarios of a moderate number of candidates or of a limited time horizon. 

---
# Single-agent or Multi-agent Systems? Why Not Both? 

**Authors**: Mingyan Gao, Yanzi Li, Banruo Liu, Yifan Yu, Phillip Wang, Ching-Yu Lin, Fan Lai  

**Link**: [PDF](https://arxiv.org/pdf/2505.18286)  

**Abstract**: Multi-agent systems (MAS) decompose complex tasks and delegate subtasks to different large language model (LLM) agents and tools. Prior studies have reported the superior accuracy performance of MAS across diverse domains, enabled by long-horizon context tracking and error correction through role-specific agents. However, the design and deployment of MAS incur higher complexity and runtime cost compared to single-agent systems (SAS). Meanwhile, frontier LLMs, such as OpenAI-o3 and Gemini-2.5-Pro, have rapidly advanced in long-context reasoning, memory retention, and tool usage, mitigating many limitations that originally motivated MAS designs. In this paper, we conduct an extensive empirical study comparing MAS and SAS across various popular agentic applications. We find that the benefits of MAS over SAS diminish as LLM capabilities improve, and we propose efficient mechanisms to pinpoint the error-prone agent in MAS. Furthermore, the performance discrepancy between MAS and SAS motivates our design of a hybrid agentic paradigm, request cascading between MAS and SAS, to improve both efficiency and capability. Our design improves accuracy by 1.1-12% while reducing deployment costs by up to 20% across various agentic applications. 

---
# Tube Loss based Deep Networks For Improving the Probabilistic Forecasting of Wind Speed 

**Authors**: Pritam Anand, Aadesh Minz, Asish Joel  

**Link**: [PDF](https://arxiv.org/pdf/2505.18284)  

**Abstract**: Uncertainty Quantification (UQ) in wind speed forecasting is a critical challenge in wind power production due to the inherently volatile nature of wind. By quantifying the associated risks and returns, UQ supports more effective decision-making for grid operations and participation in the electricity market. In this paper, we design a sequence of deep learning based probabilistic forecasting methods by using the Tube loss function for wind speed forecasting. The Tube loss function is a simple and model agnostic Prediction Interval (PI) estimation approach and can obtain the narrow PI with asymptotical coverage guarantees without any distribution assumption. Our deep probabilistic forecasting models effectively incorporate popular architectures such as LSTM, GRU, and TCN within the Tube loss framework. We further design a simple yet effective heuristic for tuning the $\delta$ parameter of the Tube loss function so that our deep forecasting models obtain the narrower PI without compromising its calibration ability. We have considered three wind datasets, containing the hourly recording of the wind speed, collected from three distinct location namely Jaisalmer, Los Angeles and San Fransico. Our numerical results demonstrate that the proposed deep forecasting models produce more reliable and narrower PIs compared to recently developed probabilistic wind forecasting methods. 

---
# TAGS: A Test-Time Generalist-Specialist Framework with Retrieval-Augmented Reasoning and Verification 

**Authors**: Jianghao Wu, Feilong Tang, Yulong Li, Ming Hu, Haochen Xue, Shoaib Jameel, Yutong Xie, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2505.18283)  

**Abstract**: Recent advances such as Chain-of-Thought prompting have significantly improved large language models (LLMs) in zero-shot medical reasoning. However, prompting-based methods often remain shallow and unstable, while fine-tuned medical LLMs suffer from poor generalization under distribution shifts and limited adaptability to unseen clinical scenarios. To address these limitations, we present TAGS, a test-time framework that combines a broadly capable generalist with a domain-specific specialist to offer complementary perspectives without any model fine-tuning or parameter updates. To support this generalist-specialist reasoning process, we introduce two auxiliary modules: a hierarchical retrieval mechanism that provides multi-scale exemplars by selecting examples based on both semantic and rationale-level similarity, and a reliability scorer that evaluates reasoning consistency to guide final answer aggregation. TAGS achieves strong performance across nine MedQA benchmarks, boosting GPT-4o accuracy by 13.8%, DeepSeek-R1 by 16.8%, and improving a vanilla 7B model from 14.1% to 23.9%. These results surpass several fine-tuned medical LLMs, without any parameter updates. The code will be available at this https URL. 

---
# Towards a Quantum-classical Augmented Network 

**Authors**: Nitin Jha, Abhishek Parakh, Mahadevan Subramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2505.18282)  

**Abstract**: In the past decade, several small-scale quantum key distribution networks have been established. However, the deployment of large-scale quantum networks depends on the development of quantum repeaters, quantum channels, quantum memories, and quantum network protocols. To improve the security of existing networks and adopt currently feasible quantum technologies, the next step is to augment classical networks with quantum devices, properties, and phenomena. To achieve this, we propose a change in the structure of the HTTP protocol such that it can carry both quantum and classical payload. This work lays the foundation for dividing one single network packet into classical and quantum payloads depending on the privacy needs. We implement logistic regression, CNN, LSTM, and BiLSTM models to classify the privacy label for outgoing communications. This enables reduced utilization of quantum resources allowing for a more efficient secure quantum network design. Experimental results using the proposed methods are presented. 

---
# Feature Preserving Shrinkage on Bayesian Neural Networks via the R2D2 Prior 

**Authors**: Tsai Hor Chan, Dora Yan Zhang, Guosheng Yin, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18280)  

**Abstract**: Bayesian neural networks (BNNs) treat neural network weights as random variables, which aim to provide posterior uncertainty estimates and avoid overfitting by performing inference on the posterior weights. However, the selection of appropriate prior distributions remains a challenging task, and BNNs may suffer from catastrophic inflated variance or poor predictive performance when poor choices are made for the priors. Existing BNN designs apply different priors to weights, while the behaviours of these priors make it difficult to sufficiently shrink noisy signals or they are prone to overshrinking important signals in the weights. To alleviate this problem, we propose a novel R2D2-Net, which imposes the R^2-induced Dirichlet Decomposition (R2D2) prior to the BNN weights. The R2D2-Net can effectively shrink irrelevant coefficients towards zero, while preventing key features from over-shrinkage. To approximate the posterior distribution of weights more accurately, we further propose a variational Gibbs inference algorithm that combines the Gibbs updating procedure and gradient-based optimization. This strategy enhances stability and consistency in estimation when the variational objective involving the shrinkage parameters is non-convex. We also analyze the evidence lower bound (ELBO) and the posterior concentration rates from a theoretical perspective. Experiments on both natural and medical image classification and uncertainty estimation tasks demonstrate satisfactory performance of our method. 

---
# Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control 

**Authors**: Alireza Rezazadeh, Zichao Li, Ange Lou, Yuying Zhao, Wei Wei, Yujia Bao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18279)  

**Abstract**: Complex tasks are increasingly delegated to ensembles of specialized LLM-based agents that reason, communicate, and coordinate actions-both among themselves and through interactions with external tools, APIs, and databases. While persistent memory has been shown to enhance single-agent performance, most approaches assume a monolithic, single-user context-overlooking the benefits and challenges of knowledge transfer across users under dynamic, asymmetric permissions. We introduce Collaborative Memory, a framework for multi-user, multi-agent environments with asymmetric, time-evolving access controls encoded as bipartite graphs linking users, agents, and resources. Our system maintains two memory tiers: (1) private memory-private fragments visible only to their originating user; and (2) shared memory-selectively shared fragments. Each fragment carries immutable provenance attributes (contributing agents, accessed resources, and timestamps) to support retrospective permission checks. Granular read policies enforce current user-agent-resource constraints and project existing memory fragments into filtered transformed views. Write policies determine fragment retention and sharing, applying context-aware transformations to update the memory. Both policies may be designed conditioned on system, agent, and user-level information. Our framework enables safe, efficient, and interpretable cross-user knowledge sharing, with provable adherence to asymmetric, time-varying policies and full auditability of memory operations. 

---
# Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks 

**Authors**: Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love  

**Link**: [PDF](https://arxiv.org/pdf/2505.18266)  

**Abstract**: We propose a testable universality hypothesis, asserting that seemingly disparate neural network solutions observed in the simple task of modular addition are unified under a common abstract algorithm. While prior work interpreted variations in neuron-level representations as evidence for distinct algorithms, we demonstrate - through multi-level analyses spanning neurons, neuron clusters, and entire networks - that multilayer perceptrons and transformers universally implement the abstract algorithm we call the approximate Chinese Remainder Theorem. Crucially, we introduce approximate cosets and show that neurons activate exclusively on them. Furthermore, our theory works for deep neural networks (DNNs). It predicts that universally learned solutions in DNNs with trainable embeddings or more than one hidden layer require only O(log n) features, a result we empirically confirm. This work thus provides the first theory-backed interpretation of multilayer networks solving modular addition. It advances generalizable interpretability and opens a testable universality hypothesis for group multiplication beyond modular addition. 

---
# MetaGen Blended RAG: Higher Accuracy for Domain-Specific Q&A Without Fine-Tuning 

**Authors**: Kunal Sawarkar, Shivam R. Solanki, Abhilasha Mangal  

**Link**: [PDF](https://arxiv.org/pdf/2505.18247)  

**Abstract**: Despite the widespread exploration of Retrieval-Augmented Generation (RAG), its deployment in enterprises for domain-specific datasets remains limited due to poor answer accuracy. These corpora, often shielded behind firewalls in private enterprise knowledge bases, having complex, domain-specific terminology, rarely seen by LLMs during pre-training; exhibit significant semantic variability across domains (like networking, military, or legal, etc.), or even within a single domain like medicine, and thus result in poor context precision for RAG systems. Currently, in such situations, fine-tuning or RAG with fine-tuning is attempted, but these approaches are slow, expensive, and lack generalization for accuracy as the new domain-specific data emerges. We propose an approach for Enterprise Search that focuses on enhancing the retriever for a domain-specific corpus through hybrid query indexes and metadata enrichment. This 'MetaGen Blended RAG' method constructs a metadata generation pipeline using key concepts, topics, and acronyms, and then creates a metadata-enriched hybrid index with boosted search queries. This approach avoids overfitting and generalizes effectively across domains. On the PubMedQA benchmark for the biomedical domain, the proposed method achieves 82% retrieval accuracy and 77% RAG accuracy, surpassing all previous RAG accuracy results without fine-tuning and sets a new benchmark for zero-shot results while outperforming much larger models like GPT3.5. The results are even comparable to the best fine-tuned models on this dataset, and we further demonstrate the robustness and scalability of the approach by evaluating it on other Q&A datasets like SQuAD, NQ etc. 

---
# Multi-Scale Probabilistic Generation Theory: A Hierarchical Framework for Interpreting Large Language Models 

**Authors**: Yukin Zhang, Qi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18244)  

**Abstract**: Large Transformer based language models achieve remarkable performance but remain opaque in how they plan, structure, and realize text. We introduce Multi_Scale Probabilistic Generation Theory (MSPGT), a hierarchical framework that factorizes generation into three semantic scales_global context, intermediate structure, and local word choices and aligns each scale with specific layer ranges in Transformer architectures. To identify scale boundaries, we propose two complementary metrics: attention span thresholds and inter layer mutual information peaks. Across four representative models (GPT-2, BERT, RoBERTa, and T5), these metrics yield stable local/intermediate/global partitions, corroborated by probing tasks and causal interventions. We find that decoder_only models allocate more layers to intermediate and global processing while encoder_only models emphasize local feature extraction. Through targeted interventions, we demonstrate that local scale manipulations primarily influence lexical diversity, intermediate-scale modifications affect sentence structure and length, and global_scale perturbations impact discourse coherence all with statistically significant effects. MSPGT thus offers a unified, architecture-agnostic method for interpreting, diagnosing, and controlling large language models, bridging the gap between mechanistic interpretability and emergent capabilities. 

---
# ZeroML: A Next Generation AutoML Language 

**Authors**: Monirul Islam Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2505.18243)  

**Abstract**: ZeroML is a new generation programming language for AutoML to drive the ML pipeline in a compiled and multi-paradigm way, with a pure functional core. Meeting the shortcomings introduced by Python, R, or Julia such as slow-running time, brittle pipelines or high dependency cost ZeroML brings the Microservices-based architecture adding the modular, reusable pieces such as DataCleaner, FeatureEngineer or ModelSelector. As a native multithread and memory-aware search optimized toolkit, and with one command deployability ability, ZeroML ensures non-coders and ML professionals to create high-accuracy models super fast and in a more reproducible way. The verbosity of the language ensures that when it comes to dropping into the backend, the code we will be creating is extremely clear but the level of repetition and boilerplate required when developing on the front end is now removed. 

---
# Intent Classification on Low-Resource Languages with Query Similarity Search 

**Authors**: Arjun Bhalla, Qi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18241)  

**Abstract**: Intent classification is an important component of a functional Information Retrieval ecosystem. Many current approaches to intent classification, typically framed as a classification problem, can be problematic as intents are often hard to define and thus data can be difficult and expensive to annotate. The problem is exacerbated when we need to extend the intent classification system to support multiple and in particular low-resource languages. To address this, we propose casting intent classification as a query similarity search problem - we use previous example queries to define an intent, and a query similarity method to classify an incoming query based on the labels of its most similar queries in latent space. With the proposed approach, we are able to achieve reasonable intent classification performance for queries in low-resource languages in a zero-shot setting. 

---
# Taming LLMs with Negative Samples: A Reference-Free Framework to Evaluate Presentation Content with Actionable Feedback 

**Authors**: Ananth Muppidi, Tarak Das, Sambaran Bandyopadhyay, Tripti Shukla, Dharun D A  

**Link**: [PDF](https://arxiv.org/pdf/2505.18240)  

**Abstract**: The generation of presentation slides automatically is an important problem in the era of generative AI. This paper focuses on evaluating multimodal content in presentation slides that can effectively summarize a document and convey concepts to a broad audience. We introduce a benchmark dataset, RefSlides, consisting of human-made high-quality presentations that span various topics. Next, we propose a set of metrics to characterize different intrinsic properties of the content of a presentation and present REFLEX, an evaluation approach that generates scores and actionable feedback for these metrics. We achieve this by generating negative presentation samples with different degrees of metric-specific perturbations and use them to fine-tune LLMs. This reference-free evaluation technique does not require ground truth presentations during inference. Our extensive automated and human experiments demonstrate that our evaluation approach outperforms classical heuristic-based and state-of-the-art large language model-based evaluations in generating scores and explanations. 

---
# Think or Not? Exploring Thinking Efficiency in Large Reasoning Models via an Information-Theoretic Lens 

**Authors**: Xixian Yong, Xiao Zhou, Yingying Zhang, Jinlin Li, Yefeng Zheng, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18237)  

**Abstract**: The recent rise of Large Reasoning Models (LRMs) has significantly improved multi-step reasoning performance, but often at the cost of generating excessively long reasoning chains. This paper revisits the efficiency of such reasoning processes through an information-theoretic lens, revealing a fundamental trade-off between reasoning length and semantic efficiency. We propose two metrics, InfoBias and InfoGain, to quantify divergence from ideal reasoning paths and stepwise information contribution, respectively. Empirical analyses show that longer reasoning chains tend to exhibit higher information bias and diminishing information gain, especially for incorrect answers. Motivated by these findings, we introduce an entropy-based Adaptive Think strategy that dynamically halts reasoning once confidence is sufficiently high, improving efficiency while maintaining competitive accuracy. Compared to the Vanilla Think approach (default mode), our strategy yields a 1.10% improvement in average accuracy and a 50.80% reduction in token usage on QwQ-32B across six benchmark tasks spanning diverse reasoning types and difficulty levels, demonstrating superior efficiency and reasoning performance. These results underscore the promise of entropy-based methods for enhancing both accuracy and cost-effiiciency in large language model deployment. 

---
# From Bias to Accountability: How the EU AI Act Confronts Challenges in European GeoAI Auditing 

**Authors**: Natalia Matuszczyk, Craig R. Barnes, Rohit Gupta, Bulent Ozel, Aniket Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2505.18236)  

**Abstract**: Bias in geospatial artificial intelligence (GeoAI) models has been documented, yet the evidence is scattered across narrowly focused studies. We synthesize this fragmented literature to provide a concise overview of bias in GeoAI and examine how the EU's Artificial Intelligence Act (EU AI Act) shapes audit obligations. We discuss recurring bias mechanisms, including representation, algorithmic and aggregation bias, and map them to specific provisions of the EU AI Act. By applying the Act's high-risk criteria, we demonstrate that widely deployed GeoAI applications qualify as high-risk systems. We then present examples of recent audits along with an outline of practical methods for detecting bias. As far as we know, this study represents the first integration of GeoAI bias evidence into the EU AI Act context, by identifying high-risk GeoAI systems and mapping bias mechanisms to the Act's Articles. Although the analysis is exploratory, it suggests that even well-curated European datasets should employ routine bias audits before 2027, when the AI Act's high-risk provisions take full effect. 

---
# The Origins of Representation Manifolds in Large Language Models 

**Authors**: Alexander Modell, Patrick Rubin-Delanchy, Nick Whiteley  

**Link**: [PDF](https://arxiv.org/pdf/2505.18235)  

**Abstract**: There is a large ongoing scientific effort in mechanistic interpretability to map embeddings and internal representations of AI systems into human-understandable concepts. A key element of this effort is the linear representation hypothesis, which posits that neural representations are sparse linear combinations of `almost-orthogonal' direction vectors, reflecting the presence or absence of different features. This model underpins the use of sparse autoencoders to recover features from representations. Moving towards a fuller model of features, in which neural representations could encode not just the presence but also a potentially continuous and multidimensional value for a feature, has been a subject of intense recent discourse. We describe why and how a feature might be represented as a manifold, demonstrating in particular that cosine similarity in representation space may encode the intrinsic geometry of a feature through shortest, on-manifold paths, potentially answering the question of how distance in representation space and relatedness in concept space could be connected. The critical assumptions and predictions of the theory are validated on text embeddings and token activations of large language models. 

---
# A Robust PPO-optimized Tabular Transformer Framework for Intrusion Detection in Industrial IoT Systems 

**Authors**: Yuanya She  

**Link**: [PDF](https://arxiv.org/pdf/2505.18234)  

**Abstract**: In this paper, we propose a robust and reinforcement-learning-enhanced network intrusion detection system (NIDS) designed for class-imbalanced and few-shot attack scenarios in Industrial Internet of Things (IIoT) environments. Our model integrates a TabTransformer for effective tabular feature representation with Proximal Policy Optimization (PPO) to optimize classification decisions via policy learning. Evaluated on the TON\textunderscore IoT benchmark, our method achieves a macro F1-score of 97.73\% and accuracy of 98.85\%. Remarkably, even on extremely rare classes like man-in-the-middle (MITM), our model achieves an F1-score of 88.79\%, showcasing strong robustness and few-shot detection capabilities. Extensive ablation experiments confirm the complementary roles of TabTransformer and PPO in mitigating class imbalance and improving generalization. These results highlight the potential of combining transformer-based tabular learning with reinforcement learning for real-world NIDS applications. 

---
# POSTER: A Multi-Signal Model for Detecting Evasive Smishing 

**Authors**: Shaghayegh Hosseinpour, Sanchari Das  

**Link**: [PDF](https://arxiv.org/pdf/2505.18233)  

**Abstract**: Smishing, or SMS-based phishing, poses an increasing threat to mobile users by mimicking legitimate communications through culturally adapted, concise, and deceptive messages, which can result in the loss of sensitive data or financial resources. In such, we present a multi-channel smishing detection model that combines country-specific semantic tagging, structural pattern tagging, character-level stylistic cues, and contextual phrase embeddings. We curated and relabeled over 84,000 messages across five datasets, including 24,086 smishing samples. Our unified architecture achieves 97.89% accuracy, an F1 score of 0.963, and an AUC of 99.73%, outperforming single-stream models by capturing diverse linguistic and structural cues. This work demonstrates the effectiveness of multi-signal learning in robust and region-aware phishing. 

---
# ELDeR: Getting Efficient LLMs through Data-Driven Regularized Layer-wise Pruning 

**Authors**: Mingkuan Feng, Jinyang Wu, Siyuan Liu, Shuai Zhang, Hongjian Fang, Ruihan Jin, Feihu Che, Pengpeng Shao, Zhengqi Wen, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18232)  

**Abstract**: The deployment of Large language models (LLMs) in many fields is largely hindered by their high computational and memory costs. Recent studies suggest that LLMs exhibit sparsity, which can be used for pruning. Previous pruning methods typically follow a prune-then-finetune paradigm. Since the pruned parts still contain valuable information, statically removing them without updating the remaining parameters often results in irreversible performance degradation, requiring costly recovery fine-tuning (RFT) to maintain performance. To address this, we propose a novel paradigm: first apply regularization, then prune. Based on this paradigm, we propose ELDeR: Getting Efficient LLMs through Data-Driven Regularized Layer-wise Pruning. We multiply the output of each transformer layer by an initial weight, then we iteratively learn the weights of each transformer layer by using a small amount of data in a simple way. After that, we apply regularization to the difference between the output and input of the layers with smaller weights, forcing the information to be transferred to the remaining layers. Compared with direct pruning, ELDeR reduces the information loss caused by direct parameter removal, thus better preserving the model's language modeling ability. Experimental results show that ELDeR achieves superior performance compared with powerful layer-wise structured pruning methods, while greatly reducing RFT computational costs. Since ELDeR is a layer-wise pruning method, its end-to-end acceleration effect is obvious, making it a promising technique for efficient LLMs. 

---
# NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache 

**Authors**: Donghyun Son, Euntae Choi, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.18231)  

**Abstract**: Large Language Model (LLM) inference is typically memory-intensive, especially when processing large batch sizes and long sequences, due to the large size of key-value (KV) cache. Vector Quantization (VQ) is recently adopted to alleviate this issue, but we find that the existing approach is susceptible to distribution shift due to its reliance on calibration datasets. To address this limitation, we introduce NSNQuant, a calibration-free Vector Quantization (VQ) technique designed for low-bit compression of the KV cache. By applying a three-step transformation-1) a token-wise normalization (Normalize), 2) a channel-wise centering (Shift), and 3) a second token-wise normalization (Normalize)-with Hadamard transform, NSNQuant effectively aligns the token distribution with the standard normal distribution. This alignment enables robust, calibration-free vector quantization using a single reusable codebook. Extensive experiments show that NSNQuant consistently outperforms prior methods in both 1-bit and 2-bit settings, offering strong generalization and up to 3$\times$ throughput gain over full-precision baselines. 

---
# Follow the Energy, Find the Path: Riemannian Metrics from Energy-Based Models 

**Authors**: Louis Béthune, David Vigouroux, Yilun Du, Rufin VanRullen, Thomas Serre, Victor Boutin  

**Link**: [PDF](https://arxiv.org/pdf/2505.18230)  

**Abstract**: What is the shortest path between two data points lying in a high-dimensional space? While the answer is trivial in Euclidean geometry, it becomes significantly more complex when the data lies on a curved manifold -- requiring a Riemannian metric to describe the space's local curvature. Estimating such a metric, however, remains a major challenge in high dimensions.
In this work, we propose a method for deriving Riemannian metrics directly from pretrained Energy-Based Models (EBMs) -- a class of generative models that assign low energy to high-density regions. These metrics define spatially varying distances, enabling the computation of geodesics -- shortest paths that follow the data manifold's intrinsic geometry. We introduce two novel metrics derived from EBMs and show that they produce geodesics that remain closer to the data manifold and exhibit lower curvature distortion, as measured by alignment with ground-truth trajectories. We evaluate our approach on increasingly complex datasets: synthetic datasets with known data density, rotated character images with interpretable geometry, and high-resolution natural images embedded in a pretrained VAE latent space.
Our results show that EBM-derived metrics consistently outperform established baselines, especially in high-dimensional settings. Our work is the first to derive Riemannian metrics from EBMs, enabling data-aware geodesics and unlocking scalable, geometry-driven learning for generative modeling and simulation. 

---
# BEDI: A Comprehensive Benchmark for Evaluating Embodied Agents on UAVs 

**Authors**: Mingning Guo, Mengwei Wu, Jiarun He, Shaoxian Li, Haifeng Li, Chao Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18229)  

**Abstract**: With the rapid advancement of low-altitude remote sensing and Vision-Language Models (VLMs), Embodied Agents based on Unmanned Aerial Vehicles (UAVs) have shown significant potential in autonomous tasks. However, current evaluation methods for UAV-Embodied Agents (UAV-EAs) remain constrained by the lack of standardized benchmarks, diverse testing scenarios and open system interfaces. To address these challenges, we propose BEDI (Benchmark for Embodied Drone Intelligence), a systematic and standardized benchmark designed for evaluating UAV-EAs. Specifically, we introduce a novel Dynamic Chain-of-Embodied-Task paradigm based on the perception-decision-action loop, which decomposes complex UAV tasks into standardized, measurable subtasks. Building on this paradigm, we design a unified evaluation framework encompassing five core sub-skills: semantic perception, spatial perception, motion control, tool utilization, and task planning. Furthermore, we construct a hybrid testing platform that integrates static real-world environments with dynamic virtual scenarios, enabling comprehensive performance assessment of UAV-EAs across varied contexts. The platform also offers open and standardized interfaces, allowing researchers to customize tasks and extend scenarios, thereby enhancing flexibility and scalability in the evaluation process. Finally, through empirical evaluations of several state-of-the-art (SOTA) VLMs, we reveal their limitations in embodied UAV tasks, underscoring the critical role of the BEDI benchmark in advancing embodied intelligence research and model optimization. By filling the gap in systematic and standardized evaluation within this field, BEDI facilitates objective model comparison and lays a robust foundation for future development in this field. Our benchmark will be released at this https URL . 

---
# Token Reduction Should Go Beyond Efficiency in Generative Models -- From Vision, Language to Multimodality 

**Authors**: Zhenglun Kong, Yize Li, Fanhu Zeng, Lei Xin, Shvat Messica, Xue Lin, Pu Zhao, Manolis Kellis, Hao Tang, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.18227)  

**Abstract**: In Transformer architectures, tokens\textemdash discrete units derived from raw data\textemdash are formed by segmenting inputs into fixed-length chunks. Each token is then mapped to an embedding, enabling parallel attention computations while preserving the input's essential information. Due to the quadratic computational complexity of transformer self-attention mechanisms, token reduction has primarily been used as an efficiency strategy. This is especially true in single vision and language domains, where it helps balance computational costs, memory usage, and inference latency. Despite these advances, this paper argues that token reduction should transcend its traditional efficiency-oriented role in the era of large generative models. Instead, we position it as a fundamental principle in generative modeling, critically influencing both model architecture and broader applications. Specifically, we contend that across vision, language, and multimodal systems, token reduction can: (i) facilitate deeper multimodal integration and alignment, (ii) mitigate "overthinking" and hallucinations, (iii) maintain coherence over long inputs, and (iv) enhance training stability, etc. We reframe token reduction as more than an efficiency measure. By doing so, we outline promising future directions, including algorithm design, reinforcement learning-guided token reduction, token optimization for in-context learning, and broader ML and scientific domains. We highlight its potential to drive new model architectures and learning strategies that improve robustness, increase interpretability, and better align with the objectives of generative modeling. 

---
# IDA-Bench: Evaluating LLMs on Interactive Guided Data Analysis 

**Authors**: Hanyu Li, Haoyu Liu, Tingyu Zhu, Tianyu Guo, Zeyu Zheng, Xiaotie Deng, Michael I. Jordan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18223)  

**Abstract**: Large Language Models (LLMs) show promise as data analysis agents, but existing benchmarks overlook the iterative nature of the field, where experts' decisions evolve with deeper insights of the dataset. To address this, we introduce IDA-Bench, a novel benchmark evaluating LLM agents in multi-round interactive scenarios. Derived from complex Kaggle notebooks, tasks are presented as sequential natural language instructions by an LLM-simulated user. Agent performance is judged by comparing its final numerical output to the human-derived baseline. Initial results show that even state-of-the-art coding agents (like Claude-3.7-thinking) succeed on < 50% of the tasks, highlighting limitations not evident in single-turn tests. This work underscores the need to improve LLMs' multi-round capabilities for building more reliable data analysis agents, highlighting the necessity of achieving a balance between instruction following and reasoning. 

---
# A Domain Ontology for Modeling the Book of Purification in Islam 

**Authors**: Hessa Alawwad  

**Link**: [PDF](https://arxiv.org/pdf/2505.18222)  

**Abstract**: This paper aims to address a gap in major Islamic topics by developing an ontology for the Book of Purification in Islam. Many authoritative Islamic texts begin with the Book of Purification, as it is essential for performing prayer (the second pillar of Islam after Shahadah, the profession of faith) and other religious duties such as Umrah and Hajj.
The ontology development strategy followed six key steps: (1) domain identification, (2) knowledge acquisition, (3) conceptualization, (4) classification, (5) integration and implementation, and (6) ontology generation. This paper includes examples of the constructed tables and classifications.
The focus is on the design and analysis phases, as technical implementation is beyond the scope of this study. However, an initial implementation is provided to illustrate the steps of the proposed strategy.
The developed ontology ensures reusability by formally defining and encoding the key concepts, attributes, and relationships related to the Book of Purification. This structured representation is intended to support knowledge sharing and reuse. 

---
# Evidence-Grounded Multimodal Misinformation Detection with Attention-Based GNNs 

**Authors**: Sharad Duwal, Mir Nafis Sharear Shopnil, Abhishek Tyagi, Adiba Mahbub Proma  

**Link**: [PDF](https://arxiv.org/pdf/2505.18221)  

**Abstract**: Multimodal out-of-context (OOC) misinformation is misinformation that repurposes real images with unrelated or misleading captions. Detecting such misinformation is challenging because it requires resolving the context of the claim before checking for misinformation. Many current methods, including LLMs and LVLMs, do not perform this contextualization step. LLMs hallucinate in absence of context or parametric knowledge. In this work, we propose a graph-based method that evaluates the consistency between the image and the caption by constructing two graph representations: an evidence graph, derived from online textual evidence, and a claim graph, from the claim in the caption. Using graph neural networks (GNNs) to encode and compare these representations, our framework then evaluates the truthfulness of image-caption pairs. We create datasets for our graph-based method, evaluate and compare our baseline model against popular LLMs on the misinformation detection task. Our method scores $93.05\%$ detection accuracy on the evaluation set and outperforms the second-best performing method (an LLM) by $2.82\%$, making a case for smaller and task-specific methods. 

---
# Navigating Pitfalls: Evaluating LLMs in Machine Learning Programming Education 

**Authors**: Smitha Kumar, Michael A. Lones, Manuel Maarek, Hind Zantout  

**Link**: [PDF](https://arxiv.org/pdf/2505.18220)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened new avenues in education. This study examines the use of LLMs in supporting learning in machine learning education; in particular, it focuses on the ability of LLMs to identify common errors of practice (pitfalls) in machine learning code, and their ability to provide feedback that can guide learning. Using a portfolio of code samples, we consider four different LLMs: one closed model and three open models. Whilst the most basic pitfalls are readily identified by all models, many common pitfalls are not. They particularly struggle to identify pitfalls in the early stages of the ML pipeline, especially those which can lead to information leaks, a major source of failure within applied ML projects. They also exhibit limited success at identifying pitfalls around model selection, which is a concept that students often struggle with when first transitioning from theory to practice. This questions the use of current LLMs to support machine learning education, and also raises important questions about their use by novice practitioners. Nevertheless, when LLMs successfully identify pitfalls in code, they do provide feedback that includes advice on how to proceed, emphasising their potential role in guiding learners. We also compare the capability of closed and open LLM models, and find that the gap is relatively small given the large difference in model sizes. This presents an opportunity to deploy, and potentially customise, smaller more efficient LLM models within education, avoiding risks around cost and data sharing associated with commercial models. 

---
# CoMet: Metaphor-Driven Covert Communication for Multi-Agent Language Games 

**Authors**: Shuhang Xu, Fangwei Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18218)  

**Abstract**: Metaphors are a crucial way for humans to express complex or subtle ideas by comparing one concept to another, often from a different domain. However, many large language models (LLMs) struggle to interpret and apply metaphors in multi-agent language games, hindering their ability to engage in covert communication and semantic evasion, which are crucial for strategic communication. To address this challenge, we introduce CoMet, a framework that enables LLM-based agents to engage in metaphor processing. CoMet combines a hypothesis-based metaphor reasoner with a metaphor generator that improves through self-reflection and knowledge integration. This enhances the agents' ability to interpret and apply metaphors, improving the strategic and nuanced quality of their interactions. We evaluate CoMet on two multi-agent language games - Undercover and Adversarial Taboo - which emphasize Covert Communication and Semantic Evasion. Experimental results demonstrate that CoMet significantly enhances the agents' ability to communicate strategically using metaphors. 

---
# ABHINAYA -- A System for Speech Emotion Recognition In Naturalistic Conditions Challenge 

**Authors**: Soumya Dutta, Smruthi Balaji, Varada R, Viveka Salinamakki, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.18217)  

**Abstract**: Speech emotion recognition (SER) in naturalistic settings remains a challenge due to the intrinsic variability, diverse recording conditions, and class imbalance. As participants in the Interspeech Naturalistic SER Challenge which focused on these complexities, we present Abhinaya, a system integrating speech-based, text-based, and speech-text models. Our approach fine-tunes self-supervised and speech large language models (SLLM) for speech representations, leverages large language models (LLM) for textual context, and employs speech-text modeling with an SLLM to capture nuanced emotional cues. To combat class imbalance, we apply tailored loss functions and generate categorical decisions through majority voting. Despite one model not being fully trained, the Abhinaya system ranked 4th among 166 submissions. Upon completion of training, it achieved state-of-the-art performance among published results, demonstrating the effectiveness of our approach for SER in real-world conditions. 

---
# Data Mining-Based Techniques for Software Fault Localization 

**Authors**: Peggy Cellier, Mireille Ducassé, Sébastien Ferré, Olivier Ridoux, W. Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18216)  

**Abstract**: This chapter illustrates the basic concepts of fault localization using a data mining technique. It utilizes the Trityp program to illustrate the general method. Formal concept analysis and association rule are two well-known methods for symbolic data mining. In their original inception, they both consider data in the form of an object-attribute table. In their original inception, they both consider data in the form of an object-attribute table. The chapter considers a debugging process in which a program is tested against different test cases. Two attributes, PASS and FAIL, represent the issue of the test case. The chapter extends the analysis of data mining for fault localization for the multiple fault situations. It addresses how data mining can be further applied to fault localization for GUI components. Unlike traditional software, GUI test cases are usually event sequences, and each individual event has a unique corresponding event handler. 

---
# Do BERT-Like Bidirectional Models Still Perform Better on Text Classification in the Era of LLMs? 

**Authors**: Junyan Zhang, Yiming Huang, Shuliang Liu, Yubo Gao, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18215)  

**Abstract**: The rapid adoption of LLMs has overshadowed the potential advantages of traditional BERT-like models in text classification. This study challenges the prevailing "LLM-centric" trend by systematically comparing three category methods, i.e., BERT-like models fine-tuning, LLM internal state utilization, and zero-shot inference across six high-difficulty datasets. Our findings reveal that BERT-like models often outperform LLMs. We further categorize datasets into three types, perform PCA and probing experiments, and identify task-specific model strengths: BERT-like models excel in pattern-driven tasks, while LLMs dominate those requiring deep semantics or world knowledge. Based on this, we propose TaMAS, a fine-grained task selection strategy, advocating for a nuanced, task-driven approach over a one-size-fits-all reliance on LLMs. 

---
# LA-RCS: LLM-Agent-Based Robot Control System 

**Authors**: TaekHyun Park, YoungJun Choi, SeungHoon Shin, Kwangil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.18214)  

**Abstract**: LA-RCS (LLM-agent-based robot control system) is a sophisticated robot control system designed to autonomously plan, work, and analyze the external environment based on user requirements by utilizing LLM-Agent. Utilizing a dual-agent framework, LA-RCS generates plans based on user requests, observes the external environment, executes the plans, and modifies the plans as needed to adapt to changes in the external conditions. Additionally, LA-RCS interprets natural language commands by the user and converts them into commands compatible with the robot interface so that the robot can execute tasks and meet user requests properly. During his process, the system autonomously evaluates observation results, provides feedback on the tasks, and executes commands based on real-time environmental monitoring, significantly reducing the need for user intervention in fulfilling requests. We categorized the scenarios that LA-RCS needs to perform into four distinct types and conducted a quantitative assessment of its performance in each scenario. The results showed an average success rate of 90 percent, demonstrating the system capability to fulfill user requests satisfactorily. For more extensive results, readers can visit our project page: this https URL 

---
# AIDRIN 2.0: A Framework to Assess Data Readiness for AI 

**Authors**: Kaveen Hiniduma, Dylan Ryan, Suren Byna, Jean Luca Bez, Ravi Madduri  

**Link**: [PDF](https://arxiv.org/pdf/2505.18213)  

**Abstract**: AI Data Readiness Inspector (AIDRIN) is a framework to evaluate and improve data preparedness for AI applications. It addresses critical data readiness dimensions such as data quality, bias, fairness, and privacy. This paper details enhancements to AIDRIN by focusing on user interface improvements and integration with a privacy-preserving federated learning (PPFL) framework. By refining the UI and enabling smooth integration with decentralized AI pipelines, AIDRIN becomes more accessible and practical for users with varying technical expertise. Integrating with an existing PPFL framework ensures that data readiness and privacy are prioritized in federated learning environments. A case study involving a real-world dataset demonstrates AIDRIN's practical value in identifying data readiness issues that impact AI model performance. 

---
# Towards medical AI misalignment: a preliminary study 

**Authors**: Barbara Puccio, Federico Castagna, Allan Tucker, Pierangelo Veltri  

**Link**: [PDF](https://arxiv.org/pdf/2505.18212)  

**Abstract**: Despite their staggering capabilities as assistant tools, often exceeding human performances, Large Language Models (LLMs) are still prone to jailbreak attempts from malevolent users. Although red teaming practices have already identified and helped to address several such jailbreak techniques, one particular sturdy approach involving role-playing (which we named `Goofy Game') seems effective against most of the current LLMs safeguards. This can result in the provision of unsafe content, which, although not harmful per se, might lead to dangerous consequences if delivered in a setting such as the medical domain. In this preliminary and exploratory study, we provide an initial analysis of how, even without technical knowledge of the internal architecture and parameters of generative AI models, a malicious user could construct a role-playing prompt capable of coercing an LLM into producing incorrect (and potentially harmful) clinical suggestions. We aim to illustrate a specific vulnerability scenario, providing insights that can support future advancements in the field. 

---
# CrossRF: A Domain-Invariant Deep Learning Approach for RF Fingerprinting 

**Authors**: Fahrettin Emin Tiras, Hayriye Serra Altinoluk  

**Link**: [PDF](https://arxiv.org/pdf/2505.18200)  

**Abstract**: Radio Frequency (RF) fingerprinting offers a promising approach for drone identification and security, although it suffers from significant performance degradation when operating on different transmission channels. This paper presents CrossRF, a domain-invariant deep learning approach that addresses the problem of cross-channel RF fingerprinting for Unmanned Aerial Vehicle (UAV) identification. Our approach aims to minimize the domain gap between different RF channels by using adversarial learning to train a more robust model that maintains consistent identification performance despite channel variations. We validate our approach using the UAVSig dataset, comprising real-world over-the-air RF signals from identical drone models operating across several frequency channels, ensuring that the findings correspond to real-world scenarios. The experimental results show CrossRF's efficiency, achieving up to 99.03% accuracy when adapting from Channel 3 to Channel 4, compared to only 26.39% using conventional methods. The model maintains robust performance in more difficult multi-channel scenarios (87.57% accuracy adapting from Channels 1,3 to 2,4) and achieves 89.45% accuracy with 0.9 precision for controller classification. These results confirm CrossRF's ability to significantly reduce performance degradation due to cross-channel variations while maintaining high identification accuracy with minimal training data requirements, making it particularly suitable for practical drone security applications. 

---
# Large Language Model-Driven Distributed Integrated Multimodal Sensing and Semantic Communications 

**Authors**: Yubo Peng, Luping Xiang, Bingxin Zhang, Kun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18194)  

**Abstract**: Traditional single-modal sensing systems-based solely on either radio frequency (RF) or visual data-struggle to cope with the demands of complex and dynamic environments. Furthermore, single-device systems are constrained by limited perspectives and insufficient spatial coverage, which impairs their effectiveness in urban or non-line-of-sight scenarios. To overcome these challenges, we propose a novel large language model (LLM)-driven distributed integrated multimodal sensing and semantic communication (LLM-DiSAC) framework. Specifically, our system consists of multiple collaborative sensing devices equipped with RF and camera modules, working together with an aggregation center to enhance sensing accuracy. First, on sensing devices, LLM-DiSAC develops an RF-vision fusion network (RVFN), which employs specialized feature extractors for RF and visual data, followed by a cross-attention module for effective multimodal integration. Second, a LLM-based semantic transmission network (LSTN) is proposed to enhance communication efficiency, where the LLM-based decoder leverages known channel parameters, such as transceiver distance and signal-to-noise ratio (SNR), to mitigate semantic distortion. Third, at the aggregation center, a transformer-based aggregation model (TRAM) with an adaptive aggregation attention mechanism is developed to fuse distributed features and enhance sensing accuracy. To preserve data privacy, a two-stage distributed learning strategy is introduced, allowing local model training at the device level and centralized aggregation model training using intermediate features. Finally, evaluations on a synthetic multi-view RF-visual dataset generated by the Genesis simulation engine show that LLM-DiSAC achieves a good performance. 

---
# SzCORE as a benchmark: report from the seizure detection challenge at the 2025 AI in Epilepsy and Neurological Disorders Conference 

**Authors**: Jonathan Dan, Amirhossein Shahbazinia, Christodoulos Kechris, David Atienza  

**Link**: [PDF](https://arxiv.org/pdf/2505.18191)  

**Abstract**: Reliable automatic seizure detection from long-term EEG remains a challenge, as current machine learning models often fail to generalize across patients or clinical settings. Manual EEG review remains the clinical standard, underscoring the need for robust models and standardized evaluation. To rigorously assess algorithm performance, we organized a challenge using a private dataset of continuous EEG recordings from 65 subjects (4,360 hours). Expert neurophysiologists annotated the data, providing ground truth for seizure events. Participants were required to detect seizure onset and duration, with evaluation based on event-based metrics, including sensitivity, precision, F1-score, and false positives per day. The SzCORE framework ensured standardized evaluation. The primary ranking criterion was the event-based F1-score, reflecting clinical relevance by balancing sensitivity and false positives. The challenge received 30 submissions from 19 teams, with 28 algorithms evaluated. Results revealed wide variability in performance, with a top F1-score of 43% (sensitivity 37%, precision 45%), highlighting the ongoing difficulty of seizure detection. The challenge also revealed a gap between reported performance and real-world evaluation, emphasizing the importance of rigorous benchmarking. Compared to previous challenges and commercial systems, the best-performing algorithm in this contest showed improved performance. Importantly, the challenge platform now supports continuous benchmarking, enabling reproducible research, integration of new datasets, and clinical evaluation of seizure detection algorithms using a standardized framework. 

---
# PhySense: Sensor Placement Optimization for Accurate Physics Sensing 

**Authors**: Yuezhou Ma, Haixu Wu, Hang Zhou, Huikun Weng, Jianmin Wang, Mingsheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.18190)  

**Abstract**: Physics sensing plays a central role in many scientific and engineering domains, which inherently involves two coupled tasks: reconstructing dense physical fields from sparse observations and optimizing scattered sensor placements to observe maximum information. While deep learning has made rapid advances in sparse-data reconstruction, existing methods generally omit optimization of sensor placements, leaving the mutual enhancement between reconstruction and placement on the shelf. To change this suboptimal practice, we propose PhySense, a synergistic two-stage framework that learns to jointly reconstruct physical fields and to optimize sensor placements, both aiming for accurate physics sensing. The first stage involves a flow-based generative model enhanced by cross-attention to adaptively fuse sparse observations. \correct{Leveraging the reconstruction feedback, }the second stage performs sensor placement via projected gradient descent to satisfy spatial constraints. \correct{We further prove that the learning objectives of the two stages are consistent with classical variance-minimization principles, providing theoretical guarantees.} Extensive experiments across three challenging benchmarks, especially a 3D geometry dataset, indicate PhySense achieves state-of-the-art physics sensing accuracy and discovers informative sensor placements previously unconsidered. 

---
# Improving Generative Inverse Design of Rectangular Patch Antennas with Test Time Optimization 

**Authors**: Beck LaBash, Shahriar Khushrushahi, Fabian Ruehle  

**Link**: [PDF](https://arxiv.org/pdf/2505.18188)  

**Abstract**: We propose a two-stage deep learning framework for the inverse design of rectangular patch antennas. Our approach leverages generative modeling to learn a latent representation of antenna frequency response curves and conditions a subsequent generative model on these responses to produce feasible antenna geometries. We further demonstrate that leveraging search and optimization techniques at test-time improves the accuracy of the generated designs and enables consideration of auxiliary objectives such as manufacturability. Our approach generalizes naturally to different design criteria, and can be easily adapted to more complex geometric design spaces. 

---
# 2DNMRGym: An Annotated Experimental Dataset for Atom-Level Molecular Representation Learning in 2D NMR via Surrogate Supervision 

**Authors**: Yunrui Li, Hao Xu, Pengyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18181)  

**Abstract**: Two-dimensional (2D) Nuclear Magnetic Resonance (NMR) spectroscopy, particularly Heteronuclear Single Quantum Coherence (HSQC) spectroscopy, plays a critical role in elucidating molecular structures, interactions, and electronic properties. However, accurately interpreting 2D NMR data remains labor-intensive and error-prone, requiring highly trained domain experts, especially for complex molecules. Machine Learning (ML) holds significant potential in 2D NMR analysis by learning molecular representations and recognizing complex patterns from data. However, progress has been limited by the lack of large-scale and high-quality annotated datasets. In this work, we introduce 2DNMRGym, the first annotated experimental dataset designed for ML-based molecular representation learning in 2D NMR. It includes over 22,000 HSQC spectra, along with the corresponding molecular graphs and SMILES strings. Uniquely, 2DNMRGym adopts a surrogate supervision setup: models are trained using algorithm-generated annotations derived from a previously validated method and evaluated on a held-out set of human-annotated gold-standard labels. This enables rigorous assessment of a model's ability to generalize from imperfect supervision to expert-level interpretation. We provide benchmark results using a series of 2D and 3D GNN and GNN transformer models, establishing a strong foundation for future work. 2DNMRGym supports scalable model training and introduces a chemically meaningful benchmark for evaluating atom-level molecular representations in NMR-guided structural tasks. Our data and code is open-source and available on Huggingface and Github. 

---
# GAIA: A Foundation Model for Operational Atmospheric Dynamics 

**Authors**: Ata Akbari Asanjan, Olivia Alexander, Tom Berg, Clara Zhang, Matt Yang, Jad Makki, Disha Shidham, Srija Chakraborty, William Bender, Stephen Peng, Arun Ravindran, Olivier Raiman, David Potere, David Bell  

**Link**: [PDF](https://arxiv.org/pdf/2505.18179)  

**Abstract**: We present the GAIA (Geospatial Artificial Intelligence for Atmospheres) Foundation Model, a novel model that combines masked autoencoders (MAE) and self-DIstillation with NO labels (DINO) for analyzing global atmospheric patterns in satellite imagery. By integrating these complementary self-supervised learning approaches, our model simultaneously captures both local features and global dependencies. We address two critical challenges in satellite data analysis: reconstructing missing regions and estimating precipitation patterns as our first downstream tasks. The model demonstrates superior temporal pattern capture compared to standard MAE approaches, while maintaining robust performance in downstream tasks. Our experimental results show strong gap-filling capabilities across varying mask ratios and accurate precipitation estimation with limited training data, achieving a false alarm ratio of 0.088 and structural similarity of 0.881. This work represents an advancement in self-supervised learning for atmospheric science, providing a foundation for improved weather monitoring and climate analysis. The trained model weights and accompanying code are publicly available as open-source on Hugging Face here: this https URL. 

---
# Less is More: Multimodal Region Representation via Pairwise Inter-view Learning 

**Authors**: Min Namgung, Yijun Lin, JangHyeon Lee, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18178)  

**Abstract**: With the increasing availability of geospatial datasets, researchers have explored region representation learning (RRL) to analyze complex region characteristics. Recent RRL methods use contrastive learning (CL) to capture shared information between two modalities but often overlook task-relevant unique information specific to each modality. Such modality-specific details can explain region characteristics that shared information alone cannot capture. Bringing information factorization to RRL can address this by factorizing multimodal data into shared and unique information. However, existing factorization approaches focus on two modalities, whereas RRL can benefit from various geospatial data. Extending factorization beyond two modalities is non-trivial because modeling high-order relationships introduces a combinatorial number of learning objectives, increasing model complexity. We introduce Cross modal Knowledge Injected Embedding, an information factorization approach for RRL that captures both shared and unique representations. CooKIE uses a pairwise inter-view learning approach that captures high-order information without modeling high-order dependency, avoiding exhaustive combinations. We evaluate CooKIE on three regression tasks and a land use classification task in New York City and Delhi, India. Results show that CooKIE outperforms existing RRL methods and a factorized RRL model, capturing multimodal information with fewer training parameters and floating-point operations per second (FLOPs). We release the code: this https URL. 

---
# FedGRec: Dynamic Spatio-Temporal Federated Graph Learning for Secure and Efficient Cross-Border Recommendations 

**Authors**: Zhizhong Tan, Jiexin Zheng, Xingxing Yang, Chi Zhang, Weiping Deng, Wenyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18177)  

**Abstract**: Due to the highly sensitive nature of certain data in cross-border sharing, collaborative cross-border recommendations and data sharing are often subject to stringent privacy protection regulations, resulting in insufficient data for model training. Consequently, achieving efficient cross-border business recommendations while ensuring privacy security poses a significant challenge. Although federated learning has demonstrated broad potential in collaborative training without exposing raw data, most existing federated learning-based GNN training methods still rely on federated averaging strategies, which perform suboptimally on highly heterogeneous graph data. To address this issue, we propose FedGRec, a privacy-preserving federated graph learning method for cross-border recommendations. FedGRec captures user preferences from distributed multi-domain data to enhance recommendation performance across all domains without privacy leakage. Specifically, FedGRec leverages collaborative signals from local subgraphs associated with users or items to enrich their representation learning. Additionally, it employs dynamic spatiotemporal modeling to integrate global and local user preferences in real time based on business recommendation states, thereby deriving the final representations of target users and candidate items. By automatically filtering relevant behaviors, FedGRec effectively mitigates noise interference from unreliable neighbors. Furthermore, through a personalized federated aggregation strategy, FedGRec adapts global preferences to heterogeneous domain data, enabling collaborative learning of user preferences across multiple domains. Extensive experiments on three datasets demonstrate that FedGRec consistently outperforms competitive single-domain and cross-domain baselines while effectively preserving data privacy in cross-border recommendations. 

---
# Evaluation in EEG Emotion Recognition: State-of-the-Art Review and Unified Framework 

**Authors**: Natia Kukhilava, Tatia Tsmindashvili, Rapael Kalandadze, Anchit Gupta, Sofio Katamadze, François Brémond, Laura M. Ferrari, Philipp Müller, Benedikt Emanuel Wirth  

**Link**: [PDF](https://arxiv.org/pdf/2505.18175)  

**Abstract**: Electroencephalography-based Emotion Recognition (EEG-ER) has become a growing research area in recent years. Analyzing 216 papers published between 2018 and 2023, we uncover that the field lacks a unified evaluation protocol, which is essential to fairly define the state of the art, compare new approaches and to track the field's progress. We report the main inconsistencies between the used evaluation protocols, which are related to ground truth definition, evaluation metric selection, data splitting types (e.g., subject-dependent or subject-independent) and the use of different datasets. Capitalizing on this state-of-the-art research, we propose a unified evaluation protocol, EEGain (this https URL), which enables an easy and efficient evaluation of new methods and datasets. EEGain is a novel open source software framework, offering the capability to compare - and thus define - state-of-the-art results. EEGain includes standardized methods for data pre-processing, data splitting, evaluation metrics, and the ability to load the six most relevant datasets (i.e., AMIGOS, DEAP, DREAMER, MAHNOB-HCI, SEED, SEED-IV) in EEG-ER with only a single line of code. In addition, we have assessed and validated EEGain using these six datasets on the four most common publicly available methods (EEGNet, DeepConvNet, ShallowConvNet, TSception). This is a significant step to make research on EEG-ER more reproducible and comparable, thereby accelerating the overall progress of the field. 

---
# NMCSE: Noise-Robust Multi-Modal Coupling Signal Estimation Method via Optimal Transport for Cardiovascular Disease Detection 

**Authors**: Zhixin li, Peihong Zhang, Rui Sang, Yuxuan Liu, Shengchen Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.18174)  

**Abstract**: Electrocardiogram (ECG) and Phonocardiogram (PCG) signals are linked by a latent coupling signal representing the electrical-to-mechanical cardiac transformation. While valuable for cardiovascular disease (CVD) detection, this coupling signal is traditionally estimated using deconvolution methods that amplify noise, limiting clinical utility. In this paper, we propose Noise-Robust Multi-Modal Coupling Signal Estimation (NMCSE), which reformulates the problem as distribution matching via optimal transport theory. By jointly optimizing amplitude and temporal alignment, NMCSE mitigates noise amplification without additional preprocessing. Integrated with our Temporal-Spatial Feature Extraction network, NMCSE enables robust multi-modal CVD detection. Experiments on the PhysioNet 2016 dataset with realistic hospital noise demonstrate that NMCSE reduces estimation errors by approximately 30% in Mean Squared Error while maintaining higher Pearson Correlation Coefficients across all tested signal-to-noise ratios. Our approach achieves 97.38% accuracy and 0.98 AUC in CVD detection, outperforming state-of-the-art methods and demonstrating robust performance for real-world clinical applications. 

---
# Model-Distributed Inference for Large Language Models at the Edge 

**Authors**: Davide Macario, Hulya Seferoglu, Erdem Koyuncu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18164)  

**Abstract**: We introduce Model-Distributed Inference for Large-Language Models (MDI-LLM), a novel framework designed to facilitate the deployment of state-of-the-art large-language models (LLMs) across low-power devices at the edge. This is accomplished by dividing the model into multiple partitions, which are then assigned to different devices/nodes within the network. These nodes exchange intermediate activation vectors via device-to-device links, enabling collaborative computation. To enhance the efficiency of this process, we propose the "recurrent pipeline parallelism" technique, which reduces idle time on each device and facilitates parallel inference during the generation of multiple text sequences. By leveraging the combined computational resources of multiple edge devices, MDI-LLM enables the deployment of LLMs that exceed the memory capacity of individual devices, making it possible to perform inference on low-cost hardware. Furthermore, as the number of participating devices increases, MDI-LLM boosts token generation throughput and reduces memory consumption per device. 

---
# InjectLab: A Tactical Framework for Adversarial Threat Modeling Against Large Language Models 

**Authors**: Austin Howard  

**Link**: [PDF](https://arxiv.org/pdf/2505.18156)  

**Abstract**: Large Language Models (LLMs) are changing the way people interact with technology. Tools like ChatGPT and Claude AI are now common in business, research, and everyday life. But with that growth comes new risks, especially prompt-based attacks that exploit how these models process language. InjectLab is a security framework designed to address that problem. This paper introduces InjectLab as a structured, open-source matrix that maps real-world techniques used to manipulate LLMs. The framework is inspired by MITRE ATT&CK and focuses specifically on adversarial behavior at the prompt layer. It includes over 25 techniques organized under six core tactics, covering threats like instruction override, identity swapping, and multi-agent exploitation. Each technique in InjectLab includes detection guidance, mitigation strategies, and YAML-based simulation tests. A Python tool supports easy execution of prompt-based test cases. This paper outlines the framework's structure, compares it to other AI threat taxonomies, and discusses its future direction as a practical, community-driven foundation for securing language models. 

---
# Simulating Macroeconomic Expectations using LLM Agents 

**Authors**: Jianhao Lin, Lexuan Sun, Yixin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17648)  

**Abstract**: We introduce a novel framework for simulating macroeconomic expectation formation using Large Language Model-Empowered Agents (LLM Agents). By constructing thousands of LLM Agents equipped with modules for personal characteristics, prior expectations, and knowledge, we replicate a survey experiment involving households and experts on inflation and unemployment. Our results show that although the expectations and thoughts generated by LLM Agents are more homogeneous than those of human participants, they still effectively capture key heterogeneity across agents and the underlying drivers of expectation formation. Furthermore, a module-ablation exercise highlights the critical role of prior expectations in simulating such heterogeneity. This approach complements traditional survey methods and offers new insights into AI behavioral science in macroeconomic research. 

---
# A Matrix Product State Model for Simultaneous Classification and Generation 

**Authors**: Alex Mossi, Bojan Žunkovic, Kyriakos Flouris  

**Link**: [PDF](https://arxiv.org/pdf/2406.17441)  

**Abstract**: Quantum machine learning (QML) is a rapidly expanding field that merges the principles of quantum computing with the techniques of machine learning. One of the powerful mathematical frameworks in this domain is tensor networks. These networks are used to approximate high-order tensors by contracting tensors with lower ranks. Initially developed for simulating quantum systems, tensor networks have become integral to quantum computing and, by extension, to QML. Drawing inspiration from these quantum methods, specifically the Matrix Product States (MPS), we apply them in a classical machine learning setting. Their ability to efficiently represent and manipulate complex, high-dimensional data makes them effective in a supervised learning framework. Here, we present an MPS model, in which the MPS functions as both a classifier and a generator. The dual functionality of this novel MPS model permits a strategy that enhances the traditional training of supervised MPS models. This framework is inspired by generative adversarial networks and is geared towards generating more realistic samples by reducing outliers. In addition, our contributions offer insights into the mechanics of tensor network methods for generation tasks. Specifically, we discuss alternative embedding functions and a new sampling method from non-normalized MPSs. 

---
