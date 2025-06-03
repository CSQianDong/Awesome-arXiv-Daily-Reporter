# RoboEgo System Card: An Omnimodal Model with Native Full Duplexity 

**Authors**: Yiqun Yao, Xiang Li, Xin Jiang, Xuezhi Fang, Naitong Yu, Aixin Sun, Yequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01934)  

**Abstract**: Humans naturally process real-world multimodal information in a full-duplex manner. In artificial intelligence, replicating this capability is essential for advancing model development and deployment, particularly in embodied contexts. The development of multimodal models faces two primary challenges: (1) effectively handling more than three modalities-such as vision, audio, and text; and (2) delivering full-duplex responses to rapidly evolving human instructions. To facilitate research on models that support both omnimodal processing and full duplexity, we present RoboEgo (alias: FLM-Ego), a unified model system designed to address both challenges. RoboEgo incorporates a backbone architecture and algorithms that natively support full duplexity, achieving a theoretical duplex latency of 80 ms. In streaming visually grounded conversations under real-world conditions, RoboEgo exhibits superior responsiveness and speech naturalness, while maintaining comparable content qualities to state-of-the-art semi-duplex omnimodal models-a feat previously considered unattainable by native full-duplex systems. 

---
# Large language models can learn and generalize steganographic chain-of-thought under process supervision 

**Authors**: Joey Skaf, Luis Ibanez-Lissen, Robert McCarthy, Connor Watts, Vasil Georgiv, Hannes Whittingham, Lorena Gonzalez-Manzano, David Lindner, Cameron Tice, Edward James Young, Puria Radmard  

**Link**: [PDF](https://arxiv.org/pdf/2506.01926)  

**Abstract**: Chain-of-thought (CoT) reasoning not only enhances large language model performance but also provides critical insights into decision-making processes, marking it as a useful tool for monitoring model intent and planning. By proactively preventing models from acting on CoT indicating misaligned or harmful intent, CoT monitoring can be used to reduce risks associated with deploying models. However, developers may be incentivized to train away the appearance of harmful intent from CoT traces, by either customer preferences or regulatory requirements. Recent works have shown that banning mention of a specific example of reward hacking, which may be done either to make CoT presentable to users or as a naive attempt to prevent the behavior, causes obfuscation of the undesired reasoning traces but the persistence of the undesired behavior. Such obfuscation threatens the reliability of CoT monitoring. However, obfuscation of reasoning can be due to its internalization to latent space computation, or its encoding within the CoT. Here, we provide an extension to these results. First, we show that penalizing the use of specific strings within load-bearing reasoning traces causes models to substitute alternative strings. Crucially, this does not alter the underlying method by which the model performs the task, demonstrating that the model can learn to steganographically encode its reasoning. We further demonstrate that models can generalize an encoding scheme. When the penalized strings belong to an overarching class, the model learns not only to substitute strings seen in training, but also develops a general encoding scheme for all members of the class which it can apply to held-out testing strings. 

---
# Understanding Overadaptation in Supervised Fine-Tuning: The Role of Ensemble Methods 

**Authors**: Yifan Hao, Xingyuan Pan, Hanning Zhang, Chenlu Ye, Rui Pan, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01901)  

**Abstract**: Supervised fine-tuning (SFT) on domain-specific data is the dominant approach for adapting foundation models to specialized tasks. However, it has been observed that SFT models tend to forget knowledge acquired during pretraining. In vision models, ensembling a pretrained model with its fine-tuned counterpart has been shown to mitigate this issue. In this work, we demonstrate that the same holds for language models, and, more strikingly, we observe an overadaptation phenomenon: the ensemble model not only retains general knowledge from the foundation model but also outperforms the fine-tuned model even on the fine-tuning domain itself. Despite the empirical success of ensembling, a theoretical understanding of its benefits remains underexplored. We develop a formal theoretical analysis of the overadaptation phenomenon. Ensembling mitigates this by balancing two primary sources of error: bias, caused by insufficient fine-tuning, and variance, introduced by overfitting to fine-tuning data. While regularization techniques aim to address this trade-off, we show that ensembling provides a more effective solution. We analyze this phenomenon in over-parameterized linear settings and demonstrate that interpolating between pretrained and fine-tuned weights significantly improves performance. These findings offer theoretical justification for the observed advantages of model ensembling, supported by empirical experiments consistent with our analysis. 

---
# COALESCE: Economic and Security Dynamics of Skill-Based Task Outsourcing Among Team of Autonomous LLM Agents 

**Authors**: Manish Bhatt, Ronald F. Del Rosario, Vineeth Sai Narajala, Idan Habler  

**Link**: [PDF](https://arxiv.org/pdf/2506.01900)  

**Abstract**: The meteoric rise and proliferation of autonomous Large Language Model (LLM) agents promise significant capabilities across various domains. However, their deployment is increasingly constrained by substantial computational demands, specifically for Graphics Processing Unit (GPU) resources. This paper addresses the critical problem of optimizing resource utilization in LLM agent systems. We introduce COALESCE (Cost-Optimized and Secure Agent Labour Exchange via Skill-based Competence Estimation), a novel framework designed to enable autonomous LLM agents to dynamically outsource specific subtasks to specialized, cost-effective third-party LLM agents. The framework integrates mechanisms for hybrid skill representation, dynamic skill discovery, automated task decomposition, a unified cost model comparing internal execution costs against external outsourcing prices, simplified market-based decision-making algorithms, and a standardized communication protocol between LLM agents. Comprehensive validation through 239 theoretical simulations demonstrates 41.8\% cost reduction potential, while large-scale empirical validation across 240 real LLM tasks confirms 20.3\% cost reduction with proper epsilon-greedy exploration, establishing both theoretical viability and practical effectiveness. The emergence of proposed open standards like Google's Agent2Agent (A2A) protocol further underscores the need for frameworks like COALESCE that can leverage such standards for efficient agent interaction. By facilitating a dynamic market for agent capabilities, potentially utilizing protocols like A2A for communication, COALESCE aims to significantly reduce operational costs, enhance system scalability, and foster the emergence of specialized agent economies, making complex LLM agent functionalities more accessible and economically viable. 

---
# WHEN TO ACT, WHEN TO WAIT: Modeling Structural Trajectories for Intent Triggerability in Task-Oriented Dialogue 

**Authors**: Yaoyao Qian, Jindan Huang, Yuanli Wang, Simon Yu, Kyrie Zhixuan Zhou, Jiayuan Mao, Mingfu Liang, Hanhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01881)  

**Abstract**: Task-oriented dialogue systems often face difficulties when user utterances seem semantically complete but lack necessary structural information for appropriate system action. This arises because users frequently do not fully understand their own needs, while systems require precise intent definitions. Current LLM-based agents cannot effectively distinguish between linguistically complete and contextually triggerable expressions, lacking frameworks for collaborative intent formation. We present STORM, a framework modeling asymmetric information dynamics through conversations between UserLLM (full internal access) and AgentLLM (observable behavior only). STORM produces annotated corpora capturing expression trajectories and latent cognitive transitions, enabling systematic analysis of collaborative understanding development. Our contributions include: (1) formalizing asymmetric information processing in dialogue systems; (2) modeling intent formation tracking collaborative understanding evolution; and (3) evaluation metrics measuring internal cognitive improvements alongside task performance. Experiments across four language models reveal that moderate uncertainty (40-60%) can outperform complete transparency in certain scenarios, with model-specific patterns suggesting reconsideration of optimal information completeness in human-AI collaboration. These findings contribute to understanding asymmetric reasoning dynamics and inform uncertainty-calibrated dialogue system design. 

---
# Fodor and Pylyshyn's Legacy - Still No Human-like Systematic Compositionality in Neural Networks 

**Authors**: Tim Woydt, Moritz Willig, Antonia Wüst, Lukas Helff, Wolfgang Stammer, Constantin A. Rothkopf, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.01820)  

**Abstract**: Strong meta-learning capabilities for systematic compositionality are emerging as an important skill for navigating the complex and changing tasks of today's world. However, in presenting models for robust adaptation to novel environments, it is important to refrain from making unsupported claims about the performance of meta-learning systems that ultimately do not stand up to scrutiny. While Fodor and Pylyshyn famously posited that neural networks inherently lack this capacity as they are unable to model compositional representations or structure-sensitive operations, and thus are not a viable model of the human mind, Lake and Baroni recently presented meta-learning as a pathway to compositionality. In this position paper, we critically revisit this claim and highlight limitations in the proposed meta-learning framework for compositionality. Our analysis shows that modern neural meta-learning systems can only perform such tasks, if at all, under a very narrow and restricted definition of a meta-learning setup. We therefore claim that `Fodor and Pylyshyn's legacy' persists, and to date, there is no human-like systematic compositionality learned in neural networks. 

---
# The Ultimate Test of Superintelligent AI Agents: Can an AI Balance Care and Control in Asymmetric Relationships? 

**Authors**: Djallel Bouneffouf, Matthew Riemer, Kush Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2506.01813)  

**Abstract**: This paper introduces the Shepherd Test, a new conceptual test for assessing the moral and relational dimensions of superintelligent artificial agents. The test is inspired by human interactions with animals, where ethical considerations about care, manipulation, and consumption arise in contexts of asymmetric power and self-preservation. We argue that AI crosses an important, and potentially dangerous, threshold of intelligence when it exhibits the ability to manipulate, nurture, and instrumentally use less intelligent agents, while also managing its own survival and expansion goals. This includes the ability to weigh moral trade-offs between self-interest and the well-being of subordinate agents. The Shepherd Test thus challenges traditional AI evaluation paradigms by emphasizing moral agency, hierarchical behavior, and complex decision-making under existential stakes. We argue that this shift is critical for advancing AI governance, particularly as AI systems become increasingly integrated into multi-agent environments. We conclude by identifying key research directions, including the development of simulation environments for testing moral behavior in AI, and the formalization of ethical manipulation within multi-agent systems. 

---
# A Study on the MCP x A2A Framework for Enhancing Interoperability of LLM-based Autonomous Agents 

**Authors**: Cheonsu Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01804)  

**Abstract**: This paper provides an in-depth technical analysis and implementation methodology of the open-source Agent-to-Agent (A2A) protocol developed by Google and the Model Context Protocol (MCP) introduced by Anthropic. While the evolution of LLM-based autonomous agents is rapidly accelerating, efficient interactions among these agents and their integration with external systems remain significant challenges. In modern AI systems, collaboration between autonomous agents and integration with external tools have become essential elements for building practical AI applications. A2A offers a standardized communication method that enables agents developed in heterogeneous environments to collaborate effectively, while MCP provides a structured I/O framework for agents to connect with external tools and resources. Prior studies have focused primarily on the features and applications of either A2A or MCP individually. In contrast, this study takes an integrated approach, exploring how the two protocols can complement each other to address interoperability issues and facilitate efficient collaboration within complex agent ecosystems. 

---
# Self-Challenging Language Model Agents 

**Authors**: Yifei Zhou, Sergey Levine, Jason Weston, Xian Li, Sainbayar Sukhbaatar  

**Link**: [PDF](https://arxiv.org/pdf/2506.01716)  

**Abstract**: Large language models are quickly becoming the foundation for intelligent agents that are capable of using tools. However, training such agents is challenging because it requires human creation and annotation of a diverse set of tasks, tools, and evaluation criteria. In this paper, we propose the Self-Challenging framework for training an agent on high-quality tasks that are generated by itself. The agent first plays the role of challenger and generates a task after interacting with the given tools. The tasks take the form of a novel general class of problems termed Code-as-Task, which are defined by an instruction, a verification function and solution and failure cases which serve as tests, allowing to filter only for high-quality tasks. The agent then takes an executor role and trains on those tasks with reinforcement learning using the evaluation feedback as a reward. Evaluation on two existing multi-turn tool-use agent benchmarks, M3ToolEval and TauBench, shows the Self-Challenging framework achieves over a two-fold improvement in Llama-3.1-8B-Instruct, despite using only self-generated training data. 

---
# Generate, Not Recommend: Personalized Multimodal Content Generation 

**Authors**: Jiongnan Liu, Zhicheng Dou, Ning Hu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01704)  

**Abstract**: To address the challenge of information overload from massive web contents, recommender systems are widely applied to retrieve and present personalized results for users. However, recommendation tasks are inherently constrained to filtering existing items and lack the ability to generate novel concepts, limiting their capacity to fully satisfy user demands and preferences. In this paper, we propose a new paradigm that goes beyond content filtering and selecting: directly generating personalized items in a multimodal form, such as images, tailored to individual users. To accomplish this, we leverage any-to-any Large Multimodal Models (LMMs) and train them in both supervised fine-tuning and online reinforcement learning strategy to equip them with the ability to yield tailored next items for users. Experiments on two benchmark datasets and user study confirm the efficacy of the proposed method. Notably, the generated images not only align well with users' historical preferences but also exhibit relevance to their potential future interests. 

---
# A Descriptive and Normative Theory of Human Beliefs in RLHF 

**Authors**: Sylee Dandekar, Shripad Deshmukh, Frank Chiu, W. Bradley Knox, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2506.01692)  

**Abstract**: Human preferences in RLHF are typically modeled as a function of the human's reward function or corresponding optimal state-action values. In this work, we propose that human beliefs about the capabilities of the agent being trained also play a key role in preference generation. We examine two questions related to this hypothesis, one descriptive and one normative, respectively: Do human labelers' beliefs about agent capabilities affect the preferences that they provide? And what is the ideal set of beliefs about an agent -- and resulting preferences -- for humans to have? We propose a new preference model that incorporates human beliefs and provide a normative theory that bounds the error on the final learned policy based on the \textit{mismatch} between the human's beliefs and an idealized set of beliefs. We then confirm via a human study that beliefs about agent capabilities do, in fact, significantly affect preferences and can be influenced through simple interventions. Additionally, we empirically show through synthetic experiments that it is often suboptimal for human preference labelers to assume agent optimality. Collectively, these results theoretically and empirically demonstrate how reducing the mismatch between human beliefs and agent capabilities can lead to more performant RLHF and point toward new best practices for RLHF practitioners. 

---
# Respond Beyond Language: A Benchmark for Video Generation in Response to Realistic User Intents 

**Authors**: Shuting Wang, Yunqi Liu, Zixin Yang, Ning Hu, Zhicheng Dou, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01689)  

**Abstract**: Querying generative AI models, e.g., large language models (LLMs), has become a prevalent method for information acquisition. However, existing query-answer datasets primarily focus on textual responses, making it challenging to address complex user queries that require visual demonstrations or explanations for better understanding. To bridge this gap, we construct a benchmark, RealVideoQuest, designed to evaluate the abilities of text-to-video (T2V) models in answering real-world, visually grounded queries. It identifies 7.5K real user queries with video response intents from Chatbot-Arena and builds 4.5K high-quality query-video pairs through a multistage video retrieval and refinement process. We further develop a multi-angle evaluation system to assess the quality of generated video answers. Experiments indicate that current T2V models struggle with effectively addressing real user queries, pointing to key challenges and future research opportunities in multimodal AI. 

---
# Reasoning-Based Approach with Chain-of-Thought for Alzheimer's Detection Using Speech and Large Language Models 

**Authors**: Chanwoo Park, Anna Seo Gyeong Choi, Sunghye Cho, Chanwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01683)  

**Abstract**: Societies worldwide are rapidly entering a super-aged era, making elderly health a pressing concern. The aging population is increasing the burden on national economies and households. Dementia cases are rising significantly with this demographic shift. Recent research using voice-based models and large language models (LLM) offers new possibilities for dementia diagnosis and treatment. Our Chain-of-Thought (CoT) reasoning method combines speech and language models. The process starts with automatic speech recognition to convert speech to text. We add a linear layer to an LLM for Alzheimer's disease (AD) and non-AD classification, using supervised fine-tuning (SFT) with CoT reasoning and cues. This approach showed an 16.7% relative performance improvement compared to methods without CoT prompt reasoning. To the best of our knowledge, our proposed method achieved state-of-the-art performance in CoT approaches. 

---
# K12Vista: Exploring the Boundaries of MLLMs in K-12 Education 

**Authors**: Chong Li, Chenglin Zhu, Tao Zhang, Mingan Lin, Zenan Zhou, Jian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.01676)  

**Abstract**: Multimodal large language models have demonstrated remarkable reasoning capabilities in various visual tasks. However, their abilities in K12 scenarios are still systematically underexplored. Previous studies suffer from various limitations including narrow subject coverage, insufficient data scale, lack of diversity in question types, and naive answer-centric evaluation method, resulting in insufficient exploration of model capabilities. To address these gaps, we propose K12Vista, the most comprehensive multimodal benchmark for Chinese K12 subject knowledge understanding and reasoning to date, featuring 33,000 questions across five core subjects from primary to high school and three question types. Moreover, beyond the final outcome, we are also concerned with the correctness of MLLMs' reasoning processes. For this purpose, we meticulously compiles errors from MLLMs' reasoning processes and leverage an automated data pipeline to construct K12-PEM-800K, the largest process evaluation dataset offering detailed step-by-step judgement annotations for MLLMs' reasoning. Subsequently, we developed K12-PEM, an advanced process evaluation model that integrates an overall assessment of both the reasoning process and answer correctness. Moreover, we also introduce K12-PEBench, the first high-quality, human-annotated benchmark specifically designed for evaluating abilities of reasoning process this http URL experiments reveal that current MLLMs exhibit significant flaws when reasoning within K12Vista, providing critical insights for the development of more capable this http URL open our resources at this https URL. 

---
# Social Cooperation in Conversational AI Agents 

**Authors**: Mustafa Mert Çelikok, Saptarashmi Bandyopadhyay, Robert Loftin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01624)  

**Abstract**: The development of AI agents based on large, open-domain language models (LLMs) has paved the way for the development of general-purpose AI assistants that can support human in tasks such as writing, coding, graphic design, and scientific research. A major challenge with such agents is that, by necessity, they are trained by observing relatively short-term interactions with humans. Such models can fail to generalize to long-term interactions, for example, interactions where a user has repeatedly corrected mistakes on the part of the agent. In this work, we argue that these challenges can be overcome by explicitly modeling humans' social intelligence, that is, their ability to build and maintain long-term relationships with other agents whose behavior cannot always be predicted. By mathematically modeling the strategies humans use to communicate and reason about one another over long periods of time, we may be able to derive new game theoretic objectives against which LLMs and future AI agents may be optimized. 

---
# MAGIK: Mapping to Analogous Goals via Imagination-enabled Knowledge Transfer 

**Authors**: Ajsal Shereef Palattuparambil, Thommen George Karimpanal, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2506.01623)  

**Abstract**: Humans excel at analogical reasoning - applying knowledge from one task to a related one with minimal relearning. In contrast, reinforcement learning (RL) agents typically require extensive retraining even when new tasks share structural similarities with previously learned ones. In this work, we propose MAGIK, a novel framework that enables RL agents to transfer knowledge to analogous tasks without interacting with the target environment. Our approach leverages an imagination mechanism to map entities in the target task to their analogues in the source domain, allowing the agent to reuse its original policy. Experiments on custom MiniGrid and MuJoCo tasks show that MAGIK achieves effective zero-shot transfer using only a small number of human-labelled examples. We compare our approach to related baselines and highlight how it offers a novel and effective mechanism for knowledge transfer via imagination-based analogy mapping. 

---
# General agents need world models 

**Authors**: Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt  

**Link**: [PDF](https://arxiv.org/pdf/2506.01622)  

**Abstract**: Are world models a necessary ingredient for flexible, goal-directed behaviour, or is model-free learning sufficient? We provide a formal answer to this question, showing that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. We show that this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models. This has a number of consequences: from developing safe and general agents, to bounding agent capabilities in complex environments, and providing new algorithms for eliciting world models from agents. 

---
# MLA-Trust: Benchmarking Trustworthiness of Multimodal LLM Agents in GUI Environments 

**Authors**: Xiao Yang, Jiawei Chen, Jun Luo, Zhengwei Fang, Yinpeng Dong, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01616)  

**Abstract**: The emergence of multimodal LLM-based agents (MLAs) has transformed interaction paradigms by seamlessly integrating vision, language, action and dynamic environments, enabling unprecedented autonomous capabilities across GUI applications ranging from web automation to mobile systems. However, MLAs introduce critical trustworthiness challenges that extend far beyond traditional language models' limitations, as they can directly modify digital states and trigger irreversible real-world consequences. Existing benchmarks inadequately tackle these unique challenges posed by MLAs' actionable outputs, long-horizon uncertainty and multimodal attack vectors. In this paper, we introduce MLA-Trust, the first comprehensive and unified framework that evaluates the MLA trustworthiness across four principled dimensions: truthfulness, controllability, safety and privacy. We utilize websites and mobile applications as realistic testbeds, designing 34 high-risk interactive tasks and curating rich evaluation datasets. Large-scale experiments involving 13 state-of-the-art agents reveal previously unexplored trustworthiness vulnerabilities unique to multimodal interactive scenarios. For instance, proprietary and open-source GUI-interacting MLAs pose more severe trustworthiness risks than static MLLMs, particularly in high-stakes domains; the transition from static MLLMs into interactive MLAs considerably compromises trustworthiness, enabling harmful content generation in multi-step interactions that standalone MLLMs would typically prevent; multi-step execution, while enhancing the adaptability of MLAs, involves latent nonlinear risk accumulation across successive interactions, circumventing existing safeguards and resulting in unpredictable derived risks. Moreover, we present an extensible toolbox to facilitate continuous evaluation of MLA trustworthiness across diverse interactive environments. 

---
# PGPO: Enhancing Agent Reasoning via Pseudocode-style Planning Guided Preference Optimization 

**Authors**: Zouying Cao, Runze Wang, Yifei Yang, Xinbei Ma, Xiaoyong Zhu, Bo Zheng, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01475)  

**Abstract**: Large Language Model (LLM) agents have demonstrated impressive capabilities in handling complex interactive problems. Existing LLM agents mainly generate natural language plans to guide reasoning, which is verbose and inefficient. NL plans are also tailored to specific tasks and restrict agents' ability to generalize across similar tasks. To this end, we explore pseudocode-style plans (P-code Plan) to capture the structural logic of reasoning. We find that P-code Plan empowers LLM agents with stronger generalization ability and more efficiency. Inspired by this finding, we propose a pseudocode-style Planning Guided Preference Optimization method called PGPO for effective agent learning. With two planning-oriented rewards, PGPO further enhances LLM agents' ability to generate high-quality P-code Plans and subsequent reasoning. Experiments show that PGPO achieves superior performance on representative agent benchmarks and outperforms the current leading baselines. Analyses reveal the advantage of PGPO in reducing action errors and omissions during reasoning. 

---
# Agentic Episodic Control 

**Authors**: Xidong Yang, Wenhao Li, Junjie Sheng, Chuyun Shen, Yun Hua, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01442)  

**Abstract**: Reinforcement learning (RL) has driven breakthroughs in AI, from game-play to scientific discovery and AI alignment. However, its broader applicability remains limited by challenges such as low data efficiency and poor generalizability. Recent advances suggest that large language models, with their rich world knowledge and reasoning capabilities, could complement RL by enabling semantic state modeling and task-agnostic planning. In this work, we propose the Agentic Episodic Control (AEC), a novel architecture that integrates RL with LLMs to enhance decision-making. The AEC can leverage a large language model (LLM) to map the observations into language-grounded embeddings, which further can be stored in an episodic memory for rapid retrieval of high-value experiences. Simultaneously, a World-Graph working memory module is utilized to capture structured environmental dynamics in order to enhance relational reasoning. Furthermore, a lightweight critical state detector dynamically arbitrates between the episodic memory recall and the world-model-guided exploration. On the whole, by combining the trial-and-error learning scheme with LLM-derived semantic priors, the proposed AEC can improve both data efficiency and generalizability in reinforcement learning. In experiments on BabyAI-Text benchmark tasks, AEC demonstrates substantial improvements over existing baselines, especially on complex and generalization tasks like FindObj, where it outperforms the best baseline by up to 76%. The proposed AEC framework bridges the strengths of numeric reinforcement learning and symbolic reasoning, which provides a pathway toward more adaptable and sample-efficient agents. 

---
# Distinguishing Autonomous AI Agents from Collaborative Agentic Systems: A Comprehensive Framework for Understanding Modern Intelligent Architectures 

**Authors**: Prashik Buddhaghosh Bansod  

**Link**: [PDF](https://arxiv.org/pdf/2506.01438)  

**Abstract**: The emergence of large language models has catalyzed two distinct yet interconnected paradigms in artificial intelligence: standalone AI Agents and collaborative Agentic AI ecosystems. This comprehensive study establishes a definitive framework for distinguishing these architectures through systematic analysis of their operational principles, structural compositions, and deployment methodologies. We characterize AI Agents as specialized, tool-enhanced systems leveraging foundation models for targeted automation within constrained environments. Conversely, Agentic AI represents sophisticated multi-entity frameworks where distributed agents exhibit emergent collective intelligence through coordinated interaction protocols. Our investigation traces the evolutionary trajectory from traditional rule-based systems through generative AI foundations to contemporary agent architectures. We present detailed architectural comparisons examining planning mechanisms, memory systems, coordination protocols, and decision-making processes. The study categorizes application landscapes, contrasting single-agent implementations in customer service and content management with multi-agent deployments in research automation and complex decision support. We identify critical challenges including reliability issues, coordination complexities, and scalability constraints, while proposing innovative solutions through enhanced reasoning frameworks, robust memory architectures, and improved coordination mechanisms. This framework provides essential guidance for practitioners selecting appropriate agentic approaches and establishes foundational principles for next-generation intelligent system development. 

---
# FinRobot: Generative Business Process AI Agents for Enterprise Resource Planning in Finance 

**Authors**: Hongyang Yang, Likun Lin, Yang She, Xinyu Liao, Jiaoyang Wang, Runjia Zhang, Yuquan Mo, Christina Dan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01423)  

**Abstract**: Enterprise Resource Planning (ERP) systems serve as the digital backbone of modern financial institutions, yet they continue to rely on static, rule-based workflows that limit adaptability, scalability, and intelligence. As business operations grow more complex and data-rich, conventional ERP platforms struggle to integrate structured and unstructured data in real time and to accommodate dynamic, cross-functional workflows.
In this paper, we present the first AI-native, agent-based framework for ERP systems, introducing a novel architecture of Generative Business Process AI Agents (GBPAs) that bring autonomy, reasoning, and dynamic optimization to enterprise workflows. The proposed system integrates generative AI with business process modeling and multi-agent orchestration, enabling end-to-end automation of complex tasks such as budget planning, financial reporting, and wire transfer processing. Unlike traditional workflow engines, GBPAs interpret user intent, synthesize workflows in real time, and coordinate specialized sub-agents for modular task execution. We validate the framework through case studies in bank wire transfers and employee reimbursements, two representative financial workflows with distinct complexity and data modalities. Results show that GBPAs achieve up to 40% reduction in processing time, 94% drop in error rate, and improved regulatory compliance by enabling parallelism, risk control insertion, and semantic reasoning. These findings highlight the potential of GBPAs to bridge the gap between generative AI capabilities and enterprise-grade automation, laying the groundwork for the next generation of intelligent ERP systems. 

---
# AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning 

**Authors**: Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, Jie Xie, Wei Zhou, Wang Xu, Yuanheng Zhang, Zhou Su, Zhongwu Zhai, Xiaoming Liu, Yudong Mei, Jianming Xu, Hongyan Tian, Chongyi Wang, Chi Chen, Yuan Yao, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01391)  

**Abstract**: The recent progress of large language model agents has opened new possibilities for automating tasks through graphical user interfaces (GUIs), especially in mobile environments where intelligent interaction can greatly enhance usability. However, practical deployment of such agents remains constrained by several key challenges. Existing training data is often noisy and lack semantic diversity, which hinders the learning of precise grounding and planning. Models trained purely by imitation tend to overfit to seen interface patterns and fail to generalize in unfamiliar scenarios. Moreover, most prior work focuses on English interfaces while overlooks the growing diversity of non-English applications such as those in the Chinese mobile ecosystem. In this work, we present AgentCPM-GUI, an 8B-parameter GUI agent built for robust and efficient on-device GUI interaction. Our training pipeline includes grounding-aware pre-training to enhance perception, supervised fine-tuning on high-quality Chinese and English trajectories to imitate human-like actions, and reinforcement fine-tuning with GRPO to improve reasoning capability. We also introduce a compact action space that reduces output length and supports low-latency execution on mobile devices. AgentCPM-GUI achieves state-of-the-art performance on five public benchmarks and a new Chinese GUI benchmark called CAGUI, reaching $96.9\%$ Type-Match and $91.3\%$ Exact-Match. To facilitate reproducibility and further research, we publicly release all code, model checkpoint, and evaluation data. 

---
# AI Scientists Fail Without Strong Implementation Capability 

**Authors**: Minjun Zhu, Qiujie Xie, Yixuan Weng, Jian Wu, Zhen Lin, Linyi Yang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01372)  

**Abstract**: The emergence of Artificial Intelligence (AI) Scientist represents a paradigm shift in scientific discovery, with large language models (LLMs) taking the lead as the primary executor in the entire scientific workflow from idea generation to experiment implementation. Recent AI Scientist studies demonstrate sufficient capabilities for independent scientific discovery, with the generated research reports gaining acceptance at the ICLR 2025 workshop and ACL 2025, arguing that a human-level AI Scientist, capable of uncovering phenomena previously unknown to humans, may be imminent. Despite this substantial progress, AI Scientist has yet to produce a groundbreaking achievement in the domain of computer science on par with automated scientific tools. Based on extensive quantitative evidence from existing benchmarks in complex engineering tasks and a systematic evaluation assess 28 research papers generated by five advanced AI Scientist systems, we argue that \textbf{the fundamental bottleneck for AI Scientists lies in their capability to execute the requisite verification procedures.} Current AI Scientist systems lack the execution capabilities needed to execute rigorous experiments and produce high-quality scientific papers. To better illustrate the root cause of this \textbf{implementation gap}, we provide an in-depth discussion on the fundamental limitations of AI Scientist. This position paper aims to call for the participants in the community to bridge the implementation gap. 

---
# EgoBrain: Synergizing Minds and Eyes For Human Action Understanding 

**Authors**: Nie Lin, Yansen Wang, Dongqi Han, Weibang Jiang, Jingyuan Li, Ryosuke Furuta, Yoichi Sato, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01353)  

**Abstract**: The integration of brain-computer interfaces (BCIs), in particular electroencephalography (EEG), with artificial intelligence (AI) has shown tremendous promise in decoding human cognition and behavior from neural signals. In particular, the rise of multimodal AI models have brought new possibilities that have never been imagined before. Here, we present EgoBrain --the world's first large-scale, temporally aligned multimodal dataset that synchronizes egocentric vision and EEG of human brain over extended periods of time, establishing a new paradigm for human-centered behavior analysis. This dataset comprises 61 hours of synchronized 32-channel EEG recordings and first-person video from 40 participants engaged in 29 categories of daily activities. We then developed a muiltimodal learning framework to fuse EEG and vision for action understanding, validated across both cross-subject and cross-environment challenges, achieving an action recognition accuracy of 66.70%. EgoBrain paves the way for a unified framework for brain-computer interface with multiple modalities. All data, tools, and acquisition protocols are openly shared to foster open science in cognitive computing. 

---
# An Empirical Study of Group Conformity in Multi-Agent Systems 

**Authors**: Min Choi, Keonwoo Kim, Sungwon Chae, Sangyeob Baek  

**Link**: [PDF](https://arxiv.org/pdf/2506.01332)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled multi-agent systems that simulate real-world interactions with near-human reasoning. While previous studies have extensively examined biases related to protected attributes such as race, the emergence and propagation of biases on socially contentious issues in multi-agent LLM interactions remain underexplored. This study explores how LLM agents shape public opinion through debates on five contentious topics. By simulating over 2,500 debates, we analyze how initially neutral agents, assigned a centrist disposition, adopt specific stances over time. Statistical analyses reveal significant group conformity mirroring human behavior; LLM agents tend to align with numerically dominant groups or more intelligent agents, exerting a greater influence. These findings underscore the crucial role of agent intelligence in shaping discourse and highlight the risks of bias amplification in online interactions. Our results emphasize the need for policy measures that promote diversity and transparency in LLM-generated discussions to mitigate the risks of bias propagation within anonymous online environments. 

---
# ORMind: A Cognitive-Inspired End-to-End Reasoning Framework for Operations Research 

**Authors**: Zhiyuan Wang, Bokui Chen, Yinya Huang, Qingxing Cao, Ming He, Jianping Fan, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01326)  

**Abstract**: Operations research (OR) is widely deployed to solve critical decision-making problems with complex objectives and constraints, impacting manufacturing, logistics, finance, and healthcare outcomes. While Large Language Models (LLMs) have shown promising results in various domains, their practical application in industry-relevant operations research (OR) problems presents significant challenges and opportunities. Preliminary industrial applications of LLMs for operations research face two critical deployment challenges: 1) Self-correction focuses on code syntax rather than mathematical accuracy, causing costly errors; 2) Complex expert selection creates unpredictable workflows that reduce transparency and increase maintenance costs, making them impractical for time-sensitive business applications. To address these business limitations, we introduce ORMind, a cognitive-inspired framework that enhances optimization through counterfactual reasoning. Our approach emulates human cognition, implementing an end-to-end workflow that systematically transforms requirements into mathematical models and executable solver code. It is currently being tested internally in Lenovo's AI Assistant, with plans to enhance optimization capabilities for both business and consumer customers. Experiments demonstrate that ORMind outperforms existing methods, achieving a 9.5\% improvement on the NL4Opt dataset and a 14.6\% improvement on the ComplexOR dataset. 

---
# Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner 

**Authors**: Chunhui Zhang, Zhongyu Ouyang, Kwonjoon Lee, Nakul Agarwal, Sean Dae Houlihan, Soroush Vosoughi, Shao-Yuan Lo  

**Link**: [PDF](https://arxiv.org/pdf/2506.01301)  

**Abstract**: Theory-of-Mind (ToM) enables humans to infer mental states-such as beliefs, desires, and intentions-forming the foundation of social cognition. However, existing computational ToM methods rely on structured workflows with ToM-specific priors or deep model fine-tuning, which struggle with scalability in multimodal environments and fail to generalize as task complexity increases. To address these limitations, we propose a scalable Bayesian ToM planner that decomposes ToM reasoning into stepwise Bayesian updates. Our framework introduces weak-to-strong control, allowing smaller language models (LMs) to specialize in ToM-specific likelihood estimation and transfer their reasoning behaviors to larger LMs (7B to 405B) for integration with social and world knowledge. This synergistic approach aligns large-model inference of human mental states with Bayesian principles. Extensive experiments show that our method achieves a 4.6% accuracy improvement over state-of-the-art techniques on multimodal ToM benchmarks, including challenging unseen scenarios, thereby establishing a new standard for modeling human mental states in complex environments. 

---
# Scalable In-Context Q-Learning 

**Authors**: Jinmei Liu, Fuhong Liu, Jianye Hao, Bo Wang, Huaxiong Li, Chunlin Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01299)  

**Abstract**: Recent advancements in language models have demonstrated remarkable in-context learning abilities, prompting the exploration of in-context reinforcement learning (ICRL) to extend the promise to decision domains. Due to involving more complex dynamics and temporal correlations, existing ICRL approaches may face challenges in learning from suboptimal trajectories and achieving precise in-context inference. In the paper, we propose \textbf{S}calable \textbf{I}n-\textbf{C}ontext \textbf{Q}-\textbf{L}earning (\textbf{SICQL}), an innovative framework that harnesses dynamic programming and world modeling to steer ICRL toward efficient reward maximization and task generalization, while retaining the scalability and stability of supervised pretraining. We design a prompt-based multi-head transformer architecture that simultaneously predicts optimal policies and in-context value functions using separate heads. We pretrain a generalized world model to capture task-relevant information, enabling the construction of a compact prompt that facilitates fast and precise in-context inference. During training, we perform iterative policy improvement by fitting a state value function to an upper-expectile of the Q-function, and distill the in-context value functions into policy extraction using advantage-weighted regression. Extensive experiments across a range of discrete and continuous environments show consistent performance gains over various types of baselines, especially when learning from suboptimal data. Our code is available at this https URL 

---
# MobCLIP: Learning General-purpose Geospatial Representation at Scale 

**Authors**: Ya Wen, Jixuan Cai, Qiyao Ma, Linyan Li, Xinhua Chen, Chris Webster, Yulun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01297)  

**Abstract**: Representation learning of geospatial locations remains a core challenge in achieving general geospatial intelligence. Current embedding methods often lack versatility, limiting their utility across diverse tasks in both human and natural domains. We present MobCLIP, the first nationwide general-purpose location encoder, integrating an unprecedented diversity of data modalities through effective and scalable multimodal fusion. Adopting a novel CLIP-based architecture, our framework aligns 100M+ POIs, nationwide remote sensing imagery, and structured demographic statistics with a billion-edge mobility graph. By tokenizing spatial locations into grid cells inspired by Vision Transformers, we establish a unified representation space bridging mobility patterns and multimodal features. To rigorously evaluate the general-purpose effectiveness of MobCLIP, we construct a benchmark dataset composed of 11 downstream prediction tasks across social, economic, and natural domains. Experiments show that MobCLIP, with four input modalities and a compact 128-dimensional representation space, achieves significantly superior general-purpose predictive performances than state-of-the-art models by an average of 35%. Thanks to the effective integration of human-centric modalities, the performance gain is particularly profound in human-centric tasks, such as energy consumption (+260%), offline retail consumption amount (+98%), and crime cases (+95%) predictions. Echoing LLM scaling laws, we further demonstrate the scaling behavior in geospatial representation learning. We open-source code and pretrained models at: this http URL. 

---
# On the Hardness of Approximating Distributions with Probabilistic Circuits 

**Authors**: John Leland, YooJung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01281)  

**Abstract**: A fundamental challenge in probabilistic modeling is balancing expressivity and tractable inference. Probabilistic circuits (PCs) aim to directly address this tradeoff by imposing structural constraints that guarantee efficient inference of certain queries while maintaining expressivity. Since inference complexity on PCs depends on circuit size, understanding the size bounds across circuit families is key to characterizing the tradeoff between tractability and expressive efficiency. However, expressive efficiency is often studied through exact representations, where exactly encoding distributions while enforcing various structural properties often incurs exponential size blow-ups. Thus, we pose the following question: can we avoid such size blow-ups by allowing some small approximation error? We first show that approximating an arbitrary distribution with bounded $f$-divergence is $\mathsf{NP}$-hard for any model that can tractably compute marginals. We then prove an exponential size gap for approximation between the class of decomposable PCs and additionally deterministic PCs. 

---
# GeoLocSFT: Efficient Visual Geolocation via Supervised Fine-Tuning of Multimodal Foundation Models 

**Authors**: Qiang Yi, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.01277)  

**Abstract**: Accurately determining the geographic location where a single image was taken, visual geolocation, remains a formidable challenge due to the planet's vastness and the deceptive similarity among distant locations. We introduce GeoLocSFT, a framework that demonstrates how targeted supervised fine-tuning (SFT) of a large multimodal foundation model (Gemma 3) using a small, high-quality dataset can yield highly competitive geolocation performance. GeoLocSFT is trained with only 2700 carefully selected image-GPS pairs from our geographically diverse MR600k dataset. Despite this limited data, our SFT-centric approach substantially improves over baseline models and achieves robust results on standard benchmarks such as Im2GPS-3k and YFCC-4k, as well as on our newly proposed and challenging MR40k benchmark, aimed specifically at sparsely populated regions. Further, we explore multi-candidate inference and aggregation strategies but find that the core gains are already realized at the SFT stage. Our findings highlight the power of high-quality supervision and efficient SFT for planet-scale image geolocation, especially when compared to prior methods that require massive databases or complex pipelines. To foster further research, we publicly release the MR40k benchmark dataset. 

---
# Contra4: Evaluating Contrastive Cross-Modal Reasoning in Audio, Video, Image, and 3D 

**Authors**: Artemis Panagopoulou, Le Xue, Honglu Zhou, silvio savarese, Ran Xu, Caiming Xiong, Chris Callison-Burch, Mark Yatskar, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2506.01275)  

**Abstract**: Real-world decision-making often begins with identifying which modality contains the most relevant information for a given query. While recent multimodal models have made impressive progress in processing diverse inputs, it remains unclear whether they can reason contrastively across multiple modalities to select the one that best satisfies a natural language prompt. We argue this capability is foundational, especially in retrieval-augmented and decision-time contexts, where systems must evaluate multiple signals and identify which one conveys the relevant information. To evaluate this skill, we introduce Contra4, a dataset for contrastive cross-modal reasoning across four modalities: image, audio, video, and 3D. Each example presents a natural language question alongside multiple candidate modality instances, and the model must select the one that semantically aligns with the prompt. Contra4 combines human-annotated captions with a mixture-of-models round-trip-consistency filter to ensure high-quality supervision, resulting in 174k training examples and a manually verified test set of 2.3k samples. While task-specific fine-tuning improves performance by 56% relative to baseline, state-of-the-art models still achieve only 56% accuracy overall and 42% in four-modality settings, underscoring a significant limitation in current multimodal models. 

---
# RAISE: Reasoning Agent for Interactive SQL Exploration 

**Authors**: Fernando Granado, Roberto Lotufo, Jayr Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2506.01273)  

**Abstract**: Recent advances in large language models (LLMs) have propelled research in natural language interfaces to databases. However, most state-of-the-art text-to- SQL systems still depend on complex, multi-stage pipelines. This work proposes a novel agentic framework that unifies schema linking, query generation, and itera- tive refinement within a single, end-to-end component. By leveraging the intrinsic reasoning abilities of LLMs, our method emulates how humans answer questions when working with unfamiliar databases: understanding the data by formulating hypotheses, running dynamic queries to validate them, reasoning over the results, and revising outputs based on observed results. Crucially, our approach intro- duces a new strategy for scaling test-time computation in text-to-SQL: we scale the depth of interactive database exploration and reflection. This shift enables the model to allocate computation dynamically to better understand the data, especially useful in ambiguous and underspecified scenarios. Our experiments show that it improved the Execution Accuracy (EX) from 44.8% to 56.5% on the challenging BIRD dataset using DeepSeek-R1-Distill-Llama-70B. Fur- thermore, when equipped with steps to add more diversity to the answers, our agent achieves a Best-of-N accuracy of 81.8% with 8 rounds of candidate gener- ation, rivaling the 82.79% achieved by the top-ranked published solution, while reducing engineering complexity. These findings position our unified framework as a promising alternative for building natural language interfaces to databases. 

---
# CleanS2S: Single-file Framework for Proactive Speech-to-Speech Interaction 

**Authors**: Yudong Lu, Yazhe Niu, Shuai Hu, Haolin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01268)  

**Abstract**: CleanS2S is a framework for human-like speech-to-speech interaction that advances conversational AI through single-file implementation and proactive dialogue capabilities. Our system integrates automatic speech recognition, large language models, and text-to-speech synthesis into a unified pipeline with real-time interruption handling, achieving low transition latency through full-duplex websocket connections and non-blocking I/O. Beyond conventional chatbot paradigms, we pioneer a proactive interaction mechanism, which combines memory systems with Subjective Action Judgement module, enabling five human-like response strategies: interruption, refusal, deflection, silence, and standard response. The memory module dynamically aggregates historical, and contextual data to inform interaction decisions. This approach breaks the rigid turn-based convention by allowing system-initiated dialog control and context-aware response selection. And we propose Action Judgement SFT that assesses input streams for responses strategies. The framework's single-file implementation with atomic configurations offers researchers unprecedented transparency and extensibility for interaction agents. The code of CleanS2S is released at \this https URL. 

---
# Test Automation for Interactive Scenarios via Promptable Traffic Simulation 

**Authors**: Augusto Mondelli, Yueshan Li, Alessandro Zanardi, Emilio Frazzoli  

**Link**: [PDF](https://arxiv.org/pdf/2506.01199)  

**Abstract**: Autonomous vehicle (AV) planners must undergo rigorous evaluation before widespread deployment on public roads, particularly to assess their robustness against the uncertainty of human behaviors. While recent advancements in data-driven scenario generation enable the simulation of realistic human behaviors in interactive settings, leveraging these models to construct comprehensive tests for AV planners remains an open challenge. In this work, we introduce an automated method to efficiently generate realistic and safety-critical human behaviors for AV planner evaluation in interactive scenarios. We parameterize complex human behaviors using low-dimensional goal positions, which are then fed into a promptable traffic simulator, ProSim, to guide the behaviors of simulated agents. To automate test generation, we introduce a prompt generation module that explores the goal domain and efficiently identifies safety-critical behaviors using Bayesian optimization. We apply our method to the evaluation of an optimization-based planner and demonstrate its effectiveness and efficiency in automatically generating diverse and realistic driving behaviors across scenarios with varying initial conditions. 

---
# GraphPad: Inference-Time 3D Scene Graph Updates for Embodied Question Answering 

**Authors**: Muhammad Qasim Ali, Saeejith Nair, Alexander Wong, Yuchen Cui, Yuhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01174)  

**Abstract**: Structured scene representations are a core component of embodied agents, helping to consolidate raw sensory streams into readable, modular, and searchable formats. Due to their high computational overhead, many approaches build such representations in advance of the task. However, when the task specifications change, such static approaches become inadequate as they may miss key objects, spatial relations, and details. We introduce GraphPad, a modifiable structured memory that an agent can tailor to the needs of the task through API calls. It comprises a mutable scene graph representing the environment, a navigation log indexing frame-by-frame content, and a scratchpad for task-specific notes. Together, GraphPad serves as a dynamic workspace that remains complete, current, and aligned with the agent's immediate understanding of the scene and its task. On the OpenEQA benchmark, GraphPad attains 55.3%, a +3.0% increase over an image-only baseline using the same vision-language model, while operating with five times fewer input frames. These results show that allowing online, language-driven refinement of 3-D memory yields more informative representations without extra training or data collection. 

---
# ChemAU: Harness the Reasoning of LLMs in Chemical Research with Adaptive Uncertainty Estimation 

**Authors**: Xinyi Liu, Lipeng Ma, Yixuan Li, Weidong Yang, Qingyuan Zhou, Jiayi Song, Shuhao Li, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.01116)  

**Abstract**: Large Language Models (LLMs) are widely used across various scenarios due to their exceptional reasoning capabilities and natural language understanding. While LLMs demonstrate strong performance in tasks involving mathematics and coding, their effectiveness diminishes significantly when applied to chemistry-related problems. Chemistry problems typically involve long and complex reasoning steps, which contain specific terminology, including specialized symbol systems and complex nomenclature conventions. These characteristics often cause general LLMs to experience hallucinations during the reasoning process due to their lack of specific knowledge. However, existing methods are struggling to effectively leverage chemical expertise and formulas. Moreover, current uncertainty estimation methods, designed to mitigate potential reasoning errors, are unable to precisely identify specific steps or key knowledge. In this work, we propose a novel framework called ChemAU, which incorporates our adaptive uncertainty estimation method that applies different uncertainty values based on the position of reasoning steps within the whole reasoning chain. Leveraging this method, ChemAU identifies gaps in chemistry knowledge and precisely supplements chemical expertise with the specialized domain model, thereby correcting and updating the previously flawed reasoning chain. Our experiments with three popular LLMs across three chemistry datasets demonstrate that ChemAU significantly enhances both reasoning accuracy and uncertainty estimation. 

---
# SuperRL: Reinforcement Learning with Supervision to Boost Language Model Reasoning 

**Authors**: Yihao Liu, Shuocheng Li, Lang Cao, Yuhang Xie, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01096)  

**Abstract**: Large language models are increasingly used for complex reasoning tasks where high-quality offline data such as expert-annotated solutions and distilled reasoning traces are often available. However, in environments with sparse rewards, reinforcement learning struggles to sample successful trajectories, leading to inefficient learning. At the same time, these offline trajectories that represent correct reasoning paths are not utilized by standard on-policy reinforcement learning methods. To address this limitation, we propose SuperRL, a unified training framework that adaptively incorporates offline supervision into reinforcement learning. SuperRL introduces an Adaptive Switch to detect sparse reward conditions and activates a Hybrid Actor when necessary. The Hybrid Actor integrates policy gradient and supervised learning objectives at the loss level, enabling the model to benefit from accurate offline reasoning signals while maintaining the exploratory capacity of reinforcement learning. Experiments on a range of reasoning benchmarks show that SuperRL consistently outperforms standard reinforcement learning by improving sample efficiency, generalization, and robustness under sparse rewards. 

---
# Modular Speaker Architecture: A Framework for Sustaining Responsibility and Contextual Integrity in Multi-Agent AI Communication 

**Authors**: Khe-Han Toh, Hong-Kuan Teo  

**Link**: [PDF](https://arxiv.org/pdf/2506.01095)  

**Abstract**: Sustaining coherent, role-aware communication across multi-agent systems remains a foundational challenge in AI. Current frameworks often lack explicit mechanisms for speaker responsibility, leading to context drift, alignment instability, and degraded interpretability over time. We propose the Modular Speaker Architecture (MSA), a framework that decomposes speaker behavior into modular components for role tracking, responsibility continuity, and contextual coherence. Grounded in high-context human-AI dialogues, MSA includes three core modules: a Speaker Role Module, a Responsibility Chain Tracker, and a Contextual Integrity Validator. We evaluate MSA through annotated case studies and introduce structural metrics-pragmatic consistency, responsibility flow, and context stability-quantified via manual and automatic scoring and bootstrapped statistical analysis. Our results show that MSA reliably maintains interaction structure without reliance on affective signals or surface-level heuristics. We further implement a prototype configuration language (G-Code) and modular API to support MSA deployment in dynamic multi-agent scenarios. 

---
# Regulatory Graphs and GenAI for Real-Time Transaction Monitoring and Compliance Explanation in Banking 

**Authors**: Kunal Khanvilkar, Kranthi Kommuru  

**Link**: [PDF](https://arxiv.org/pdf/2506.01093)  

**Abstract**: This paper presents a real-time transaction monitoring framework that integrates graph-based modeling, narrative field embedding, and generative explanation to support automated financial compliance. The system constructs dynamic transaction graphs, extracts structural and contextual features, and classifies suspicious behavior using a graph neural network. A retrieval-augmented generation module generates natural language explanations aligned with regulatory clauses for each flagged transaction. Experiments conducted on a simulated stream of financial data show that the proposed method achieves superior results, with 98.2% F1-score, 97.8% precision, and 97.0% recall. Expert evaluation further confirms the quality and interpretability of generated justifications. The findings demonstrate the potential of combining graph intelligence and generative models to support explainable, audit-ready compliance in high-risk financial environments. 

---
# Choices and their Provenance: Explaining Stable Solutions of Abstract Argumentation Frameworks 

**Authors**: Bertram Ludäscher, Yilin Xia, Shawn Bowers  

**Link**: [PDF](https://arxiv.org/pdf/2506.01087)  

**Abstract**: The rule $\mathrm{Defeated}(x) \leftarrow \mathrm{Attacks}(y,x),\, \neg \, \mathrm{Defeated}(y)$, evaluated under the well-founded semantics (WFS), yields a unique 3-valued (skeptical) solution of an abstract argumentation framework (AF). An argument $x$ is defeated ($\mathrm{OUT}$) if there exists an undefeated argument $y$ that attacks it. For 2-valued (stable) solutions, this is the case iff $y$ is accepted ($\mathrm{IN}$), i.e., if all of $y$'s attackers are defeated. Under WFS, arguments that are neither accepted nor defeated are undecided ($\mathrm{UNDEC}$). As shown in prior work, well-founded solutions (a.k.a. grounded labelings) "explain themselves": The provenance of arguments is given by subgraphs (definable via regular path queries) rooted at the node of interest. This provenance is closely related to winning strategies of a two-player argumentation game.
We present a novel approach for extending this provenance to stable AF solutions. Unlike grounded solutions, which can be constructed via a bottom-up alternating fixpoint procedure, stable models often involve non-deterministic choice as part of the search for models. Thus, the provenance of stable solutions is of a different nature, and reflects a more expressive generate & test paradigm. Our approach identifies minimal sets of critical attacks, pinpointing choices and assumptions made by a stable model. These critical attack edges provide additional insights into the provenance of an argument's status, combining well-founded derivation steps with choice steps. Our approach can be understood as a form of diagnosis that finds minimal "repairs" to an AF graph such that the well-founded solution of the repaired graph coincides with the desired stable model of the original AF graph. 

---
# The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process 

**Authors**: Florian Carichon, Aditi Khandelwal, Marylou Fauchard, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01080)  

**Abstract**: This position paper states that AI Alignment in Multi-Agent Systems (MAS) should be considered a dynamic and interaction-dependent process that heavily depends on the social environment where agents are deployed, either collaborative, cooperative, or competitive. While AI alignment with human values and preferences remains a core challenge, the growing prevalence of MAS in real-world applications introduces a new dynamic that reshapes how agents pursue goals and interact to accomplish various tasks. As agents engage with one another, they must coordinate to accomplish both individual and collective goals. However, this complex social organization may unintentionally misalign some or all of these agents with human values or user preferences. Drawing on social sciences, we analyze how social structure can deter or shatter group and individual values. Based on these analyses, we call on the AI community to treat human, preferential, and objective alignment as an interdependent concept, rather than isolated problems. Finally, we emphasize the urgent need for simulation environments, benchmarks, and evaluation frameworks that allow researchers to assess alignment in these interactive multi-agent contexts before such dynamics grow too complex to control. 

---
# MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch 

**Authors**: Xiang Fei, Xiawu Zheng, Hao Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01056)  

**Abstract**: Function-calling has enabled large language models (LLMs) to act as tool-using agents, but injecting thousands of tool schemas into the prompt is costly and error-prone. We introduce MCP-Zero, a proactive agent framework that lets the LLM itself decide when and which external tools to retrieve, thereby assembling a task-specific toolchain from scratch. The framework is built upon three components: (1) Proactive Tool Request, where the model emits a structured $\left<\operatorname{tool\_assistant}\right>$ block that explicitly specifies the desired server and task; (2) Hierarchical Vector Routing, a coarse-to-fine retrieval algorithm that first selects candidate servers and then ranks tools within each server based on the semantic similarity; (3) Iterative Proactive Invocation, enabling multi-round, cross-domain toolchain construction with minimal context overhead, and allowing the model to iteratively revise its request when the returned tools are insufficient. To evaluate our approach we also compile MCP-tools, a retrieval dataset comprising 308 MCP servers and 2,797 tools extracted from the official Model-Context-Protocol repository and normalized into a unified JSON schema. Experiments show that MCP-Zero (i) effectively addresses the context overhead problem of existing methods and accurately selects the correct tool from a pool of nearly 3,000 candidates (248.1k tokens); (ii) reduces token consumption by 98\% on the APIbank while maintaining high accuracy; and (iii) supports multi-turn tool invocation with consistent accuracy across rounds. The code and dataset will be released soon. 

---
# IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory 

**Authors**: Wei Song, Zhenya Huang, Cheng Cheng, Weibo Gao, Bihan Xu, GuanHao Zhao, Fei Wang, Runze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01048)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance across a wide range of natural language tasks. However, selecting the optimal LLM to respond to a user query often necessitates a delicate balance between performance and cost. While powerful models deliver better results, they come at a high cost, whereas smaller models are more cost-effective but less capable. To address this trade-off, we propose IRT-Router, a multi-LLM routing framework that efficiently routes user queries to the most suitable LLM. Inspired by Item Response Theory (IRT), a psychological measurement methodology, IRT-Router explicitly models the relationship between LLM capabilities and user query attributes. This not only enables accurate prediction of response performance but also provides interpretable insights, such as LLM abilities and query difficulty. Additionally, we design an online query warm-up technique based on semantic similarity, further enhancing the online generalization capability of IRT-Router. Extensive experiments on 20 LLMs and 12 datasets demonstrate that IRT-Router outperforms most baseline methods in terms of effectiveness and interpretability. Its superior performance in cold-start scenarios further confirms the reliability and practicality of IRT-Router in real-world applications. Code is available at this https URL. 

---
# Higher-Order Responsibility 

**Authors**: Junli Jiang, Pavel Naumov  

**Link**: [PDF](https://arxiv.org/pdf/2506.01003)  

**Abstract**: In ethics, individual responsibility is often defined through Frankfurt's principle of alternative possibilities. This definition is not adequate in a group decision-making setting because it often results in the lack of a responsible party or "responsibility gap''. One of the existing approaches to address this problem is to consider group responsibility. Another, recently proposed, approach is "higher-order'' responsibility. The paper considers the problem of deciding if higher-order responsibility up to degree $d$ is enough to close the responsibility gap. The main technical result is that this problem is $\Pi_{2d+1}$-complete. 

---
# Boosting Bot Detection via Heterophily-Aware Representation Learning and Prototype-Guided Cluster Discovery 

**Authors**: Buyun He, Xiaorui Jiang, Qi Wu, Hao Liu, Yingguang Yang, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00989)  

**Abstract**: Detecting social media bots is essential for maintaining the security and trustworthiness of social networks. While contemporary graph-based detection methods demonstrate promising results, their practical application is limited by label reliance and poor generalization capability across diverse communities. Generative Graph Self-Supervised Learning (GSL) presents a promising paradigm to overcome these limitations, yet existing approaches predominantly follow the homophily assumption and fail to capture the global patterns in the graph, which potentially diminishes their effectiveness when facing the challenges of interaction camouflage and distributed deployment in bot detection scenarios. To this end, we propose BotHP, a generative GSL framework tailored to boost graph-based bot detectors through heterophily-aware representation learning and prototype-guided cluster discovery. Specifically, BotHP leverages a dual-encoder architecture, consisting of a graph-aware encoder to capture node commonality and a graph-agnostic encoder to preserve node uniqueness. This enables the simultaneous modeling of both homophily and heterophily, effectively countering the interaction camouflage issue. Additionally, BotHP incorporates a prototype-guided cluster discovery pretext task to model the latent global consistency of bot clusters and identify spatially dispersed yet semantically aligned bot collectives. Extensive experiments on two real-world bot detection benchmarks demonstrate that BotHP consistently boosts graph-based bot detectors, improving detection performance, alleviating label reliance, and enhancing generalization capability. 

---
# PolyBERT: Fine-Tuned Poly Encoder BERT-Based Model for Word Sense Disambiguation 

**Authors**: Linhan Xia, Mingzhan Yang, Guohui Yuan, Shengnan Tao, Yujing Qiu, Guo Yu, Kai Lei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00968)  

**Abstract**: Mainstream Word Sense Disambiguation (WSD) approaches have employed BERT to extract semantics from both context and definitions of senses to determine the most suitable sense of a target word, achieving notable performance. However, there are two limitations in these approaches. First, previous studies failed to balance the representation of token-level (local) and sequence-level (global) semantics during feature extraction, leading to insufficient semantic representation and a performance bottleneck. Second, these approaches incorporated all possible senses of each target word during the training phase, leading to unnecessary computational costs. To overcome these limitations, this paper introduces a poly-encoder BERT-based model with batch contrastive learning for WSD, named PolyBERT. Compared with previous WSD methods, PolyBERT has two improvements: (1) A poly-encoder with a multi-head attention mechanism is utilized to fuse token-level (local) and sequence-level (global) semantics, rather than focusing on just one. This approach enriches semantic representation by balancing local and global semantics. (2) To avoid redundant training inputs, Batch Contrastive Learning (BCL) is introduced. BCL utilizes the correct senses of other target words in the same batch as negative samples for the current target word, which reduces training inputs and computational cost. The experimental results demonstrate that PolyBERT outperforms baseline WSD methods such as Huang's GlossBERT and Blevins's BEM by 2\% in F1-score. In addition, PolyBERT with BCL reduces GPU hours by 37.6\% compared with PolyBERT without BCL. 

---
# Unlocking Personalized Knowledge in Federated Large Language Model: The Power of Mixture of Experts 

**Authors**: Fan Liu, Bikang Pan, Zhongyi Wang, Xi Yao, Xiaoying Tang, Jingya Wang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00965)  

**Abstract**: The Mixture of Experts (MoE) architecture has emerged as a prominent strategy for scaling large language models (LLMs), effectively leveraging sparse activation and facilitating task-specific personalization. However, current federated learning (FL) approaches are primarily designed for dense models, making them unable to directly exploit the sparsity inherent in MoE architectures. Treating MoE models as dense networks in federated scenarios results in excessive communication overhead and computational costs, undermining the potential for personalized knowledge sharing. To address these challenges, we propose FLEx (Federated LLMs with Personalized Experts), a novel federated learning framework explicitly tailored for MoE-based LLMs. FLEx efficiently personalizes by pruning the global MoE model to keep only one expert per client, and employs an adaptive gating mechanism to reintegrate these personalized experts into the pre-trained MoE layers, ensuring the original backbone architecture remains unchanged. These personalized experts are trained with local data and stored locally on each client, while the shared modules are aggregated globally. Extensive evaluations on diverse instruction-based datasets under non-IID conditions consistently demonstrate that FLEx outperforms existing federated baselines. Our code is available at this https URL. 

---
# Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues 

**Authors**: Youngmin Kim, Jiwan Chung, Jisoo Kim, Sunghyun Lee, Sangkyu Lee, Junhyeok Kim, Cheoljong Yang, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00958)  

**Abstract**: Nonverbal communication is integral to human interaction, with gestures, facial expressions, and body language conveying critical aspects of intent and emotion. However, existing large language models (LLMs) fail to effectively incorporate these nonverbal elements, limiting their capacity to create fully immersive conversational experiences. We introduce MARS, a multimodal language model designed to understand and generate nonverbal cues alongside text, bridging this gap in conversational AI. Our key innovation is VENUS, a large-scale dataset comprising annotated videos with time-aligned text, facial expressions, and body language. Leveraging VENUS, we train MARS with a next-token prediction objective, combining text with vector-quantized nonverbal representations to achieve multimodal understanding and generation within a unified framework. Based on various analyses of the VENUS datasets, we validate its substantial scale and high effectiveness. Our quantitative and qualitative results demonstrate that MARS successfully generates text and nonverbal languages, corresponding to conversational input. 

---
# Aligning VLM Assistants with Personalized Situated Cognition 

**Authors**: Yongqi Li, Shen Zhou, Xiaohu Li, Xin Miao, Jintao Wen, Mayi Xu, Jianhao Chen, Birong Pan, Hankun Kang, Yuanyuan Zhu, Ming Zhong, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.00930)  

**Abstract**: Vision-language models (VLMs) aligned with general human objectives, such as being harmless and hallucination-free, have become valuable assistants of humans in managing visual tasks. However, people with diversified backgrounds have different cognition even in the same situation. Consequently, they may have personalized expectations for VLM assistants. This highlights the urgent need to align VLM assistants with personalized situated cognition for real-world assistance. To study this problem, we first simplify it by characterizing individuals based on the sociological concept of Role-Set. Then, we propose to evaluate the individuals' actions to examine whether the personalized alignment is achieved. Further, we construct a benchmark named PCogAlignBench, which includes 18k instances and 20 individuals with different Role-Sets. Finally, we present a framework called PCogAlign, which constructs a cognition-aware and action-based reward model for personalized alignment. Experimental results and human evaluations demonstrate the reliability of the PCogAlignBench and the effectiveness of our proposed PCogAlign. We will open-source the constructed benchmark and code at this https URL. 

---
# Conformal Arbitrage: Risk-Controlled Balancing of Competing Objectives in Language Models 

**Authors**: William Overman, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2506.00911)  

**Abstract**: Modern language model deployments must often balance competing objectives, for example, helpfulness versus harmlessness, cost versus accuracy, and reward versus safety. We introduce Conformal Arbitrage, a post hoc framework that learns a data driven threshold to mediate between a Primary model optimized for a primary objective and a more conservative Guardian which could be another model or a human domain expert aligned with a guardrail objective. The threshold is calibrated with conformal risk control, yielding finite sample, distribution free guarantees that the long run frequency of undesirable events, such as factual errors or safety violations, does not exceed a user specified quota. Because Conformal Arbitrage operates wholly at the API level, without requiring access to model logits or updating model weights, it complements weight based alignment techniques and integrates seamlessly with existing cost aware cascades. Empirically, Conformal Arbitrage traces an efficient frontier, allowing users to define an acceptable performance level for one objective while maximizing utility in another. We observe that our method outperforms, in terms of accuracy, cost matched random routing between models. These properties make Conformal Arbitrage a practical, theoretically grounded tool for trustworthy and economical deployment of large language models across a broad range of potentially competing objectives. 

---
# Toward a Theory of Agents as Tool-Use Decision-Makers 

**Authors**: Hongru Wang, Cheng Qian, Manling Li, Jiahao Qiu, Boyang Xue, Mengdi Wang, Heng Ji, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.00886)  

**Abstract**: As Large Language Models (LLMs) evolve into increasingly autonomous agents, fundamental questions about their epistemic foundations remain unresolved: What defines an agent? How should it make decisions? And what objectives should guide its behavior? In this position paper, we argue that true autonomy requires agents to be grounded in a coherent epistemic framework that governs what they know, what they need to know, and how to acquire that knowledge efficiently. We propose a unified theory that treats internal reasoning and external actions as equivalent epistemic tools, enabling agents to systematically coordinate introspection and interaction. Building on this framework, we advocate for aligning an agent's tool use decision-making boundary with its knowledge boundary, thereby minimizing unnecessary tool use and maximizing epistemic efficiency. This perspective shifts the design of agents from mere action executors to knowledge-driven intelligence systems, offering a principled path toward building foundation agents capable of adaptive, efficient, and goal-directed behavior. 

---
# GIA-MIC: Multimodal Emotion Recognition with Gated Interactive Attention and Modality-Invariant Learning Constraints 

**Authors**: Jiajun He, Jinyi Mi, Tomoki Toda  

**Link**: [PDF](https://arxiv.org/pdf/2506.00865)  

**Abstract**: Multimodal emotion recognition (MER) extracts emotions from multimodal data, including visual, speech, and text inputs, playing a key role in human-computer interaction. Attention-based fusion methods dominate MER research, achieving strong classification performance. However, two key challenges remain: effectively extracting modality-specific features and capturing cross-modal similarities despite distribution differences caused by modality heterogeneity. To address these, we propose a gated interactive attention mechanism to adaptively extract modality-specific features while enhancing emotional information through pairwise interactions. Additionally, we introduce a modality-invariant generator to learn modality-invariant representations and constrain domain shifts by aligning cross-modal similarities. Experiments on IEMOCAP demonstrate that our method outperforms state-of-the-art MER approaches, achieving WA 80.7% and UA 81.3%. 

---
# MedBookVQA: A Systematic and Comprehensive Medical Benchmark Derived from Open-Access Book 

**Authors**: Sau Lai Yip, Sunan He, Yuxiang Nie, Shu Pui Chan, Yilin Ye, Sum Ying Lam, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00855)  

**Abstract**: The accelerating development of general medical artificial intelligence (GMAI), powered by multimodal large language models (MLLMs), offers transformative potential for addressing persistent healthcare challenges, including workforce deficits and escalating costs. The parallel development of systematic evaluation benchmarks emerges as a critical imperative to enable performance assessment and provide technological guidance. Meanwhile, as an invaluable knowledge source, the potential of medical textbooks for benchmark development remains underexploited. Here, we present MedBookVQA, a systematic and comprehensive multimodal benchmark derived from open-access medical textbooks. To curate this benchmark, we propose a standardized pipeline for automated extraction of medical figures while contextually aligning them with corresponding medical narratives. Based on this curated data, we generate 5,000 clinically relevant questions spanning modality recognition, disease classification, anatomical identification, symptom diagnosis, and surgical procedures. A multi-tier annotation system categorizes queries through hierarchical taxonomies encompassing medical imaging modalities (42 categories), body anatomies (125 structures), and clinical specialties (31 departments), enabling nuanced analysis across medical subdomains. We evaluate a wide array of MLLMs, including proprietary, open-sourced, medical, and reasoning models, revealing significant performance disparities across task types and model categories. Our findings highlight critical capability gaps in current GMAI systems while establishing textbook-derived multimodal benchmarks as essential evaluation tools. MedBookVQA establishes textbook-derived benchmarking as a critical paradigm for advancing clinical AI, exposing limitations in GMAI systems while providing anatomically structured performance metrics across specialties. 

---
# SynPO: Synergizing Descriptiveness and Preference Optimization for Video Detailed Captioning 

**Authors**: Jisheng Dang, Yizhou Zhang, Hao Ye, Teng Wang, Siming Chen, Huicheng Zheng, Yulan Guo, Jianhuang Lai, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00835)  

**Abstract**: Fine-grained video captioning aims to generate detailed, temporally coherent descriptions of video content. However, existing methods struggle to capture subtle video dynamics and rich detailed information. In this paper, we leverage preference learning to enhance the performance of vision-language models in fine-grained video captioning, while mitigating several limitations inherent to direct preference optimization (DPO). First, we propose a pipeline for constructing preference pairs that leverages the intrinsic properties of VLMs along with partial assistance from large language models, achieving an optimal balance between cost and data quality. Second, we propose Synergistic Preference Optimization (SynPO), a novel optimization method offering significant advantages over DPO and its variants. SynPO prevents negative preferences from dominating the optimization, explicitly preserves the model's language capability to avoid deviation of the optimization objective, and improves training efficiency by eliminating the need for the reference model. We extensively evaluate SynPO not only on video captioning benchmarks (e.g., VDC, VDD, VATEX) but also across well-established NLP tasks, including general language understanding and preference evaluation, using diverse pretrained models. Results demonstrate that SynPO consistently outperforms DPO variants while achieving 20\% improvement in training efficiency. Code is available at this https URL 

---
# Enhancing LLM Reasoning for Time Series Classification by Tailored Thinking and Fused Decision 

**Authors**: Jiahui Zhou, Dan Li, Lin Li, Zhuomin Chen, Shunyu Wu, Haozheng Ye, Jian Lou, Costas J. Spanos  

**Link**: [PDF](https://arxiv.org/pdf/2506.00807)  

**Abstract**: The reasoning capabilities of large language models (LLMs) have significantly advanced their performance by enabling in-depth understanding of diverse tasks. With growing interest in applying LLMs to the time series domain, this has proven nontrivial, as evidenced by the limited efficacy of straightforwardly adapting text-domain reasoning techniques. Although recent work has shown promise in several time series tasks, further leveraging advancements in LLM reasoning remains under-explored for time series classification (TSC) tasks, despite their prevalence and significance in many real-world applications. In this paper, we propose ReasonTSC, a novel framework designed to effectively leverage LLM reasoning for time series classification through both a multi-turn reasoning and a fused decision-making strategy tailored to TSC. Rather than straightforwardly applying existing reasoning techniques or relying solely on LLMs' built-in reasoning capabilities, ReasonTSC first steers the model to think over the essential characteristics of time series data. Next, it integrates predictions and confidence scores from plug-in classifiers, e.g., domain-specific time series models, as in-context examples. Finally, ReasonTSC guides the LLM through a structured reasoning process: it evaluates the initial assessment, backtracks to consider alternative hypotheses, and compares their merits before arriving at a final classification. Extensive experiments and systematic ablation studies demonstrate that ReasonTSC consistently outperforms both existing time series reasoning baselines and plug-in models, and is even capable of identifying and correcting plug-in models' false predictions. 

---
# Predicting Empirical AI Research Outcomes with Language Models 

**Authors**: Jiaxin Wen, Chenglei Si, Yueh-han Chen, He He, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00794)  

**Abstract**: Many promising-looking ideas in AI research fail to deliver, but their validation takes substantial human labor and compute. Predicting an idea's chance of success is thus crucial for accelerating empirical AI research, a skill that even expert researchers can only acquire through substantial experience. We build the first benchmark for this task and compare LMs with human experts. Concretely, given two research ideas (e.g., two jailbreaking methods), we aim to predict which will perform better on a set of benchmarks. We scrape ideas and experimental results from conference papers, yielding 1,585 human-verified idea pairs published after our base model's cut-off date for testing, and 6,000 pairs for training. We then develop a system that combines a fine-tuned GPT-4.1 with a paper retrieval agent, and we recruit 25 human experts to compare with. In the NLP domain, our system beats human experts by a large margin (64.4% v.s. 48.9%). On the full test set, our system achieves 77% accuracy, while off-the-shelf frontier LMs like o3 perform no better than random guessing, even with the same retrieval augmentation. We verify that our system does not exploit superficial features like idea complexity through extensive human-written and LM-designed robustness tests. Finally, we evaluate our system on unpublished novel ideas, including ideas generated by an AI ideation agent. Our system achieves 63.6% accuracy, demonstrating its potential as a reward model for improving idea generation models. Altogether, our results outline a promising new direction for LMs to accelerate empirical AI research. 

---
# GeoChain: Multimodal Chain-of-Thought for Geographic Reasoning 

**Authors**: Sahiti Yerramilli, Nilay Pande, Rynaa Grover, Jayant Sravan Tamarapalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.00785)  

**Abstract**: This paper introduces GeoChain, a large-scale benchmark for evaluating step-by-step geographic reasoning in multimodal large language models (MLLMs). Leveraging 1.46 million Mapillary street-level images, GeoChain pairs each image with a 21-step chain-of-thought (CoT) question sequence (over 30 million Q&A pairs). These sequences guide models from coarse attributes to fine-grained localization across four reasoning categories - visual, spatial, cultural, and precise geolocation - annotated by difficulty. Images are also enriched with semantic segmentation (150 classes) and a visual locatability score. Our benchmarking of contemporary MLLMs (GPT-4.1 variants, Claude 3.7, Gemini 2.5 variants) on a diverse 2,088-image subset reveals consistent challenges: models frequently exhibit weaknesses in visual grounding, display erratic reasoning, and struggle to achieve accurate localization, especially as the reasoning complexity escalates. GeoChain offers a robust diagnostic methodology, critical for fostering significant advancements in complex geographic reasoning within MLLMs. 

---
# Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning 

**Authors**: Weiyang Guo, Zesheng Shi, Zhuo Li, Yequan Wang, Xuebo Liu, Wenya Wang, Fangming Liu, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00782)  

**Abstract**: As large language models (LLMs) grow in power and influence, ensuring their safety and preventing harmful output becomes critical. Automated red teaming serves as a tool to detect security vulnerabilities in LLMs without manual labor. However, most existing methods struggle to balance the effectiveness and diversity of red-team generated attack prompts. To address this challenge, we propose \ourapproach, a novel automated red teaming training framework that utilizes reinforcement learning to explore and generate more effective attack prompts while balancing their diversity. Specifically, it consists of three training stages: (1) Cold Start: The red team model is supervised and fine-tuned on a jailbreak dataset obtained through imitation learning. (2) Warm-up Exploration: The model is trained in jailbreak instruction following and exploration, using diversity and consistency as reward signals. (3) Enhanced Jailbreak: Progressive jailbreak rewards are introduced to gradually enhance the jailbreak performance of the red-team model. Extensive experiments on a variety of LLMs show that \ourapproach effectively balances the diversity and effectiveness of jailbreak prompts compared to existing methods. Our work significantly improves the efficiency of red team exploration and provides a new perspective on automated red teaming. 

---
# CoP: Agentic Red-teaming for Large Language Models using Composition of Principles 

**Authors**: Chen Xiong, Pin-Yu Chen, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2506.00781)  

**Abstract**: Recent advances in Large Language Models (LLMs) have spurred transformative applications in various domains, ranging from open-source to proprietary LLMs. However, jailbreak attacks, which aim to break safety alignment and user compliance by tricking the target LLMs into answering harmful and risky responses, are becoming an urgent concern. The practice of red-teaming for LLMs is to proactively explore potential risks and error-prone instances before the release of frontier AI technology. This paper proposes an agentic workflow to automate and scale the red-teaming process of LLMs through the Composition-of-Principles (CoP) framework, where human users provide a set of red-teaming principles as instructions to an AI agent to automatically orchestrate effective red-teaming strategies and generate jailbreak prompts. Distinct from existing red-teaming methods, our CoP framework provides a unified and extensible framework to encompass and orchestrate human-provided red-teaming principles to enable the automated discovery of new red-teaming strategies. When tested against leading LLMs, CoP reveals unprecedented safety risks by finding novel jailbreak prompts and improving the best-known single-turn attack success rate by up to 19.0 times. 

---
# Do not Abstain! Identify and Solve the Uncertainty 

**Authors**: Jingyu Liu, Jingquan Peng, xiaopeng Wu, Xubin Li, Tiezheng Ge, Bo Zheng, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00780)  

**Abstract**: Despite the widespread application of Large Language Models (LLMs) across various domains, they frequently exhibit overconfidence when encountering uncertain scenarios, yet existing solutions primarily rely on evasive responses (e.g., "I don't know") overlooks the opportunity of identifying and addressing the uncertainty to generate more satisfactory responses. To systematically investigate and improve LLMs' ability of recognizing and addressing the source of uncertainty, we introduce \textbf{ConfuseBench}, a benchmark mainly focus on three types of uncertainty: document scarcity, limited capability, and query ambiguity. Experiments with ConfuseBench reveal that current LLMs struggle to accurately identify the root cause of uncertainty and solve it. They prefer to attribute uncertainty to query ambiguity while overlooking capability limitations, especially for those weaker models. To tackle this challenge, we first generate context-aware inquiries that highlight the confusing aspect of the original query. Then we judge the source of uncertainty based on the uniqueness of the inquiry's answer. Further we use an on-policy training method, InteractDPO to generate better inquiries. Experimental results demonstrate the efficacy of our approach. 

---
# HouseTS: A Large-Scale, Multimodal Spatiotemporal U.S. Housing Dataset 

**Authors**: Shengkun Wang, Yanshen Sun, Fanglan Chen, Linhan Wang, Naren Ramakrishnan, Chang-Tien Lu, Yinlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00765)  

**Abstract**: Accurate house-price forecasting is essential for investors, planners, and researchers. However, reproducible benchmarks with sufficient spatiotemporal depth and contextual richness for long horizon prediction remain scarce. To address this, we introduce HouseTS a large scale, multimodal dataset covering monthly house prices from March 2012 to December 2023 across 6,000 ZIP codes in 30 major U.S. metropolitan areas. The dataset includes over 890K records, enriched with points of Interest (POI), socioeconomic indicators, and detailed real estate metrics. To establish standardized performance baselines, we evaluate 14 models, spanning classical statistical approaches, deep neural networks (DNNs), and pretrained time-series foundation models. We further demonstrate the value of HouseTS in a multimodal case study, where a vision language model extracts structured textual descriptions of geographic change from time stamped satellite imagery. This enables interpretable, grounded insights into urban evolution. HouseTS is hosted on Kaggle, while all preprocessing pipelines, benchmark code, and documentation are openly maintained on GitHub to ensure full reproducibility and easy adoption. 

---
# Alignment Revisited: Are Large Language Models Consistent in Stated and Revealed Preferences? 

**Authors**: Zhuojun Gu, Quan Wang, Shuchu Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.00751)  

**Abstract**: Recent advances in Large Language Models (LLMs) highlight the need to align their behaviors with human values. A critical, yet understudied, issue is the potential divergence between an LLM's stated preferences (its reported alignment with general principles) and its revealed preferences (inferred from decisions in contextualized scenarios). Such deviations raise fundamental concerns for the interpretability, trustworthiness, reasoning transparency, and ethical deployment of LLMs, particularly in high-stakes applications. This work formally defines and proposes a method to measure this preference deviation. We investigate how LLMs may activate different guiding principles in specific contexts, leading to choices that diverge from previously stated general principles. Our approach involves crafting a rich dataset of well-designed prompts as a series of forced binary choices and presenting them to LLMs. We compare LLM responses to general principle prompts stated preference with LLM responses to contextualized prompts revealed preference, using metrics like KL divergence to quantify the deviation. We repeat the analysis across different categories of preferences and on four mainstream LLMs and find that a minor change in prompt format can often pivot the preferred choice regardless of the preference categories and LLMs in the test. This prevalent phenomenon highlights the lack of understanding and control of the LLM decision-making competence. Our study will be crucial for integrating LLMs into services, especially those that interact directly with humans, where morality, fairness, and social responsibilities are crucial dimensions. Furthermore, identifying or being aware of such deviation will be critically important as LLMs are increasingly envisioned for autonomous agentic tasks where continuous human evaluation of all LLMs' intermediary decision-making steps is impossible. 

---
# DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains 

**Authors**: Yongkang Xiao, Sinian Zhang, Yi Dai, Huixue Zhou, Jue Hou, Jie Ding, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00708)  

**Abstract**: Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility. 

---
# OntoRAG: Enhancing Question-Answering through Automated Ontology Derivation from Unstructured Knowledge Bases 

**Authors**: Yash Tiwari, Owais Ahmad Lone, Mayukha Pal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00664)  

**Abstract**: Ontologies are pivotal for structuring knowledge bases to enhance question answering (QA) systems powered by Large Language Models (LLMs). However, traditional ontology creation relies on manual efforts by domain experts, a process that is time intensive, error prone, and impractical for large, dynamic knowledge domains. This paper introduces OntoRAG, an automated pipeline designed to derive ontologies from unstructured knowledge bases, with a focus on electrical relay documents. OntoRAG integrates advanced techniques, including web scraping, PDF parsing, hybrid chunking, information extraction, knowledge graph construction, and ontology creation, to transform unstructured data into a queryable ontology. By leveraging LLMs and graph based methods, OntoRAG enhances global sensemaking capabilities, outperforming conventional Retrieval Augmented Generation (RAG) and GraphRAG approaches in comprehensiveness and diversity. Experimental results demonstrate OntoRAGs effectiveness, achieving a comprehensiveness win rate of 85% against vector RAG and 75% against GraphRAGs best configuration. This work addresses the critical challenge of automating ontology creation, advancing the vision of the semantic web. 

---
# AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents 

**Authors**: Hanjun Luo, Shenyu Dai, Chiming Ni, Xinfeng Li, Guibin Zhang, Kun Wang, Tongliang Liu, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2506.00641)  

**Abstract**: Despite the rapid advancement of LLM-based agents, the reliable evaluation of their safety and security remains a significant challenge. Existing rule-based or LLM-based evaluators often miss dangers in agents' step-by-step actions, overlook subtle meanings, fail to see how small issues compound, and get confused by unclear safety or security rules. To overcome this evaluation crisis, we introduce \sys, a universal, training-free, memory-augmented reasoning framework that empowers LLM evaluators to emulate human expert evaluators. \sys constructs an experiential memory by having an LLM adaptively extract structured semantic features (e.g., scenario, risk, behavior) and generate associated chain-of-thought reasoning traces for past interactions. A multi-stage, context-aware retrieval-augmented generation process then dynamically retrieves the most relevant reasoning experiences to guide the LLM evaluator's assessment of new cases. Moreover, we developed \data, the first benchmark designed to check how well LLM-based evaluators can spot both safety risks and security threats. \data comprises \textbf{2293} meticulously annotated interaction records, covering \textbf{15} risk types across \textbf{29} application scenarios. A key feature of \data is its nuanced approach to ambiguous risk situations, employing ``Strict'' and ``Lenient'' judgment standards. Experiments demonstrate that \sys not only consistently improves the evaluation performance of LLMs across all benchmarks but also sets a new state-of-the-art in LLM-as-a-judge for agent safety and security, achieving human-level accuracy. Our work is openly openly accessible. 

---
# RiOSWorld: Benchmarking the Risk of Multimodal Compter-Use Agents 

**Authors**: Jingyi Yang, Shuai Shao, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00618)  

**Abstract**: With the rapid development of multimodal large language models (MLLMs), they are increasingly deployed as autonomous computer-use agents capable of accomplishing complex computer tasks. However, a pressing issue arises: Can the safety risk principles designed and aligned for general MLLMs in dialogue scenarios be effectively transferred to real-world computer-use scenarios? Existing research on evaluating the safety risks of MLLM-based computer-use agents suffers from several limitations: it either lacks realistic interactive environments, or narrowly focuses on one or a few specific risk types. These limitations ignore the complexity, variability, and diversity of real-world environments, thereby restricting comprehensive risk evaluation for computer-use agents. To this end, we introduce \textbf{RiOSWorld}, a benchmark designed to evaluate the potential risks of MLLM-based agents during real-world computer manipulations. Our benchmark includes 492 risky tasks spanning various computer applications, involving web, social media, multimedia, os, email, and office software. We categorize these risks into two major classes based on their risk source: (i) User-originated risks and (ii) Environmental risks. For the evaluation, we evaluate safety risks from two perspectives: (i) Risk goal intention and (ii) Risk goal completion. Extensive experiments with multimodal agents on \textbf{RiOSWorld} demonstrate that current computer-use agents confront significant safety risks in real-world scenarios. Our findings highlight the necessity and urgency of safety alignment for computer-use agents in real-world computer manipulation, providing valuable insights for developing trustworthy computer-use agents. Our benchmark is publicly available at this https URL. 

---
# Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs 

**Authors**: Chenjun Xu, Bingbing Wen, Bin Han, Robert Wolfe, Lucy Lu Wang, Bill Howe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00582)  

**Abstract**: Psychology research has shown that humans are poor at estimating their performance on tasks, tending towards underconfidence on easy tasks and overconfidence on difficult tasks. We examine three LLMs, Llama-3-70B-instruct, Claude-3-Sonnet, and GPT-4o, on a range of QA tasks of varying difficulty, and show that models exhibit subtle differences from human patterns of overconfidence: less sensitive to task difficulty, and when prompted to answer based on different personas -- e.g., expert vs layman, or different race, gender, and ages -- the models will respond with stereotypically biased confidence estimations even though their underlying answer accuracy remains the same. Based on these observations, we propose Answer-Free Confidence Estimation (AFCE) to improve confidence calibration and LLM interpretability in these settings. AFCE is a self-assessment method that employs two stages of prompting, first eliciting only confidence scores on questions, then asking separately for the answer. Experiments on the MMLU and GPQA datasets spanning subjects and difficulty show that this separation of tasks significantly reduces overconfidence and delivers more human-like sensitivity to task difficulty. 

---
# Reasoning Like an Economist: Post-Training on Economic Problems Induces Strategic Generalization in LLMs 

**Authors**: Yufa Zhou, Shaobo Wang, Xingyu Dong, Xiangqi Jin, Yifang Chen, Yue Min, Kexin Yang, Xingzhang Ren, Dayiheng Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00577)  

**Abstract**: Directly training Large Language Models (LLMs) for Multi-Agent Systems (MAS) remains challenging due to intricate reward modeling, dynamic agent interactions, and demanding generalization requirements. This paper explores whether post-training techniques, specifically Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR), can effectively $\textit{generalize}$ to multi-agent scenarios. We use economic reasoning as a testbed, leveraging its strong foundations in mathematics and game theory, its demand for structured analytical reasoning, and its relevance to real-world applications such as market design, resource allocation, and policy analysis. We introduce $\textbf{Recon}$ ($\textbf{R}$easoning like an $\textbf{ECON}$omist), a 7B-parameter open-source LLM post-trained on a hand-curated dataset of 2,100 high-quality economic reasoning problems. Comprehensive evaluation on economic reasoning benchmarks and multi-agent games reveals clear improvements in structured reasoning and economic rationality. These results underscore the promise of domain-aligned post-training for enhancing reasoning and agent alignment, shedding light on the roles of SFT and RL in shaping model behavior. Code is available at this https URL . 

---
# A "Wenlu" Brain System for Multimodal Cognition and Embodied Decision-Making: A Secure New Architecture for Deep Integration of Foundation Models and Domain Knowledge 

**Authors**: Liang Geng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00570)  

**Abstract**: With the rapid penetration of artificial intelligence across industries and scenarios, a key challenge in building the next-generation intelligent core lies in effectively integrating the language understanding capabilities of foundation models with domain-specific knowledge bases in complex real-world applications. This paper proposes a multimodal cognition and embodied decision-making brain system, ``Wenlu", designed to enable secure fusion of private knowledge and public models, unified processing of multimodal data such as images and speech, and closed-loop decision-making from cognition to automatic generation of hardware-level code. The system introduces a brain-inspired memory tagging and replay mechanism, seamlessly integrating user-private data, industry-specific knowledge, and general-purpose language models. It provides precise and efficient multimodal services for enterprise decision support, medical analysis, autonomous driving, robotic control, and more. Compared with existing solutions, ``Wenlu" demonstrates significant advantages in multimodal processing, privacy security, end-to-end hardware control code generation, self-learning, and sustainable updates, thus laying a solid foundation for constructing the next-generation intelligent core. 

---
# CityLens: Benchmarking Large Language-Vision Models for Urban Socioeconomic Sensing 

**Authors**: Tianhui Liu, Jie Feng, Hetian Pang, Xin Zhang, Tianjian Ouyang, Zhiyuan Zhang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00530)  

**Abstract**: Understanding urban socioeconomic conditions through visual data is a challenging yet essential task for sustainable urban development and policy planning. In this work, we introduce $\textbf{CityLens}$, a comprehensive benchmark designed to evaluate the capabilities of large language-vision models (LLVMs) in predicting socioeconomic indicators from satellite and street view imagery. We construct a multi-modal dataset covering a total of 17 globally distributed cities, spanning 6 key domains: economy, education, crime, transport, health, and environment, reflecting the multifaceted nature of urban life. Based on this dataset, we define 11 prediction tasks and utilize three evaluation paradigms: Direct Metric Prediction, Normalized Metric Estimation, and Feature-Based Regression. We benchmark 17 state-of-the-art LLVMs across these tasks. Our results reveal that while LLVMs demonstrate promising perceptual and reasoning capabilities, they still exhibit limitations in predicting urban socioeconomic indicators. CityLens provides a unified framework for diagnosing these limitations and guiding future efforts in using LLVMs to understand and predict urban socioeconomic patterns. Our codes and datasets are open-sourced via this https URL. 

---
# Monitoring Robustness and Individual Fairness 

**Authors**: Ashutosh Gupta, Thomas A. Henzinger, Konstantin Kueffner, Kaushik Mallik, David Pape  

**Link**: [PDF](https://arxiv.org/pdf/2506.00496)  

**Abstract**: Input-output robustness appears in various different forms in the literature, such as robustness of AI models to adversarial or semantic perturbations and individual fairness of AI models that make decisions about humans.
We propose runtime monitoring of input-output robustness of deployed, black-box AI models, where the goal is to design monitors that would observe one long execution sequence of the model, and would raise an alarm whenever it is detected that two similar inputs from the past led to dissimilar outputs.
This way, monitoring will complement existing offline ``robustification'' approaches to increase the trustworthiness of AI decision-makers.
We show that the monitoring problem can be cast as the fixed-radius nearest neighbor (FRNN) search problem, which, despite being well-studied, lacks suitable online solutions.
We present our tool Clemont, which offers a number of lightweight monitors, some of which use upgraded online variants of existing FRNN algorithms, and one uses a novel algorithm based on binary decision diagrams -- a data-structure commonly used in software and hardware verification.
We have also developed an efficient parallelization technique that can substantially cut down the computation time of monitors for which the distance between input-output pairs is measured using the $L_\infty$ norm.
Using standard benchmarks from the literature of adversarial and semantic robustness and individual fairness, we perform a comparative study of different monitors in \tool, and demonstrate their effectiveness in correctly detecting robustness violations at runtime. 

---
# MIRROR: Cognitive Inner Monologue Between Conversational Turns for Persistent Reflection and Reasoning in Conversational LLMs 

**Authors**: Nicole Hsing  

**Link**: [PDF](https://arxiv.org/pdf/2506.00430)  

**Abstract**: Human intelligence relies on inner monologue to process complex information through simultaneous reflection, memory retrieval, and response formulation. We introduce MIRROR (Modular Internal Reasoning, Reflection, Orchestration, and Response), a cognitive architecture that systematically implements these parallel reasoning capabilities in large language models. MIRROR operates as a unified system with two distinct functional layers: the Thinker and the Talker. The Thinker encompasses: (1) the Inner Monologue Manager, coordinating reasoning threads across cognitive dimensions (Goals, Reasoning, and Memory); and (2) the Cognitive Controller, synthesizing these threads into a coherent internal narrative maintained across conversation turns. The Talker component then leverages this integrated narrative for context-aware responses. Evaluated on the CuRaTe benchmark--testing personalized dialogue with safety-critical constraints, conflicting preferences, and multi-turn consistency--LLMs utilizing the MIRROR architecture achieve up to 156% relative improvement in critical safety scenarios involving three persons with conflicting preferences, maintaining an average accuracy of ~>80% on all scenarios. Across scenario-specific comparisons, GPT-4o, Gemini 1.5 Pro, Claude 3.7 Sonnet, Llama 4 variants, and Mistral 3 variants with the MIRROR architecture outperformed baseline models by 21% on average (15.5 percentage points absolute). MIRROR directly addresses three critical LLM failure modes: sycophancy, attentional deficits to critical information, and inconsistent prioritization of conflicting constraints. This work bridges cognitive science and AI by implementing modular internal reasoning inspired by human cognition, creating a persistent internal model that significantly enhances multi-turn conversation capabilities. 

---
# World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks 

**Authors**: Changyuan Zhao, Ruichen Zhang, Jiacheng Wang, Gaosheng Zhao, Dusit Niyato, Geng Sun, Shiwen Mao, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.00417)  

**Abstract**: World models are emerging as a transformative paradigm in artificial intelligence, enabling agents to construct internal representations of their environments for predictive reasoning, planning, and decision-making. By learning latent dynamics, world models provide a sample-efficient framework that is especially valuable in data-constrained or safety-critical scenarios. In this paper, we present a comprehensive overview of world models, highlighting their architecture, training paradigms, and applications across prediction, generation, planning, and causal reasoning. We compare and distinguish world models from related concepts such as digital twins, the metaverse, and foundation models, clarifying their unique role as embedded cognitive engines for autonomous agents. We further propose Wireless Dreamer, a novel world model-based reinforcement learning framework tailored for wireless edge intelligence optimization, particularly in low-altitude wireless networks (LAWNs). Through a weather-aware UAV trajectory planning case study, we demonstrate the effectiveness of our framework in improving learning efficiency and decision quality. 

---
# Position: Olfaction Standardization is Essential for the Advancement of Embodied Artificial Intelligence 

**Authors**: Kordel K. France, Rohith Peddi, Nik Dennler, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00398)  

**Abstract**: Despite extraordinary progress in artificial intelligence (AI), modern systems remain incomplete representations of human cognition. Vision, audition, and language have received disproportionate attention due to well-defined benchmarks, standardized datasets, and consensus-driven scientific foundations. In contrast, olfaction - a high-bandwidth, evolutionarily critical sense - has been largely overlooked. This omission presents a foundational gap in the construction of truly embodied and ethically aligned super-human intelligence. We argue that the exclusion of olfactory perception from AI architectures is not due to irrelevance but to structural challenges: unresolved scientific theories of smell, heterogeneous sensor technologies, lack of standardized olfactory datasets, absence of AI-oriented benchmarks, and difficulty in evaluating sub-perceptual signal processing. These obstacles have hindered the development of machine olfaction despite its tight coupling with memory, emotion, and contextual reasoning in biological systems. In this position paper, we assert that meaningful progress toward general and embodied intelligence requires serious investment in olfactory research by the AI community. We call for cross-disciplinary collaboration - spanning neuroscience, robotics, machine learning, and ethics - to formalize olfactory benchmarks, develop multimodal datasets, and define the sensory capabilities necessary for machines to understand, navigate, and act within human environments. Recognizing olfaction as a core modality is essential not only for scientific completeness, but for building AI systems that are ethically grounded in the full scope of the human experience. 

---
# BASIL: Best-Action Symbolic Interpretable Learning for Evolving Compact RL Policies 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh, Mohammadali Keshtparvar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00328)  

**Abstract**: The quest for interpretable reinforcement learning is a grand challenge for the deployment of autonomous decision-making systems in safety-critical applications. Modern deep reinforcement learning approaches, while powerful, tend to produce opaque policies that compromise verification, reduce transparency, and impede human oversight. To address this, we introduce BASIL (Best-Action Symbolic Interpretable Learning), a systematic approach for generating symbolic, rule-based policies via online evolutionary search with quality-diversity (QD) optimization. BASIL represents policies as ordered lists of symbolic predicates over state variables, ensuring full interpretability and tractable policy complexity. By using a QD archive, the methodology in the proposed study encourages behavioral and structural diversity between top-performing solutions, while a complexity-aware fitness encourages the synthesis of compact representations. The evolutionary system supports the use of exact constraints for rule count and system adaptability for balancing transparency with expressiveness. Empirical comparisons with three benchmark tasks CartPole-v1, MountainCar-v0, and Acrobot-v1 show that BASIL consistently synthesizes interpretable controllers with compact representations comparable to deep reinforcement learning baselines. Herein, this article introduces a new interpretable policy synthesis method that combines symbolic expressiveness, evolutionary diversity, and online learning through a unifying framework. 

---
# Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents 

**Authors**: Xiao Yu, Baolin Peng, Ruize Xu, Michel Galley, Hao Cheng, Suman Nath, Jianfeng Gao, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00320)  

**Abstract**: Recent progress in reasoning with large language models (LLMs), such as DeepSeek-R1, demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities. 

---
# Evaluation of LLMs for mathematical problem solving 

**Authors**: Ruonan Wang, Runxi Wang, Yunwen Shen, Chengfeng Wu, Qinglin Zhou, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00309)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance on a range of educational tasks, but are still understudied for their potential to solve mathematical problems. In this study, we compare three prominent LLMs, including GPT-4o, DeepSeek-V3, and Gemini-2.0, on three mathematics datasets of varying complexities (GSM8K, MATH500, and UNSW datasets). We take a five-dimensional approach based on the Structured Chain-of-Thought (SCoT) framework to assess final answer correctness, step completeness, step validity, intermediate calculation accuracy, and problem comprehension. The results show that GPT-4o is the most stable and consistent in performance across all the datasets, but particularly it performs outstandingly in high-level questions of the UNSW dataset. DeepSeek-V3 is competitively strong in well-structured domains such as optimisation, but suffers from fluctuations in accuracy in statistical inference tasks. Gemini-2.0 shows strong linguistic understanding and clarity in well-structured problems but performs poorly in multi-step reasoning and symbolic logic. Our error analysis reveals particular deficits in each model: GPT-4o is at times lacking in sufficient explanation or precision; DeepSeek-V3 leaves out intermediate steps; and Gemini-2.0 is less flexible in mathematical reasoning in higher dimensions. 

---
# Sleep Brain and Cardiac Activity Predict Cognitive Flexibility and Conceptual Reasoning Using Deep Learning 

**Authors**: Boshra Khajehpiri, Eric Granger, Massimiliano de Zambotti, Fiona C. Baker, Mohamad Forouzanfar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00279)  

**Abstract**: Despite extensive research on the relationship between sleep and cognition, the connection between sleep microstructure and human performance across specific cognitive domains remains underexplored. This study investigates whether deep learning models can predict executive functions, particularly cognitive adaptability and conceptual reasoning from physiological processes during a night's sleep. To address this, we introduce CogPSGFormer, a multi-scale convolutional-transformer model designed to process multi-modal polysomnographic data. This model integrates one-channel ECG and EEG signals along with extracted features, including EEG power bands and heart rate variability parameters, to capture complementary information across modalities. A thorough evaluation of the CogPSGFormer architecture was conducted to optimize the processing of extended sleep signals and identify the most effective configuration. The proposed framework was evaluated on 817 individuals from the STAGES dataset using cross-validation. The model achieved 80.3\% accuracy in classifying individuals into low vs. high cognitive performance groups on unseen data based on Penn Conditional Exclusion Test (PCET) scores. These findings highlight the effectiveness of our multi-scale feature extraction and multi-modal learning approach in leveraging sleep-derived signals for cognitive performance prediction. To facilitate reproducibility, our code is publicly accessible (this https URL). 

---
# Hidden in Plain Sight: Probing Implicit Reasoning in Multimodal Language Models 

**Authors**: Qianqi Yan, Hongquan Li, Shan Jiang, Yang Zhao, Xinze Guan, Ching-Chen Kuo, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00258)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly deployed in open-ended, real-world environments where inputs are messy, underspecified, and not always trustworthy. Unlike curated benchmarks, these settings frequently involve instructions that refer to missing objects or contradictory facts, rely on ambiguous references, or request infeasible actions. In such cases, success hinges not on task execution alone, but on a model's ability to detect when something is silently wrong. This paper presents a systematic analysis of how current MLLMs handle such implicit reasoning scenarios: cases where the flaw is not explicitly stated but must be inferred from context. Using a curated diagnostic suite spanning four categories of real-world failure modes, we evaluate six MLLMs, including o3 and GPT-4o, and find that models frequently fail to surface hidden issues, even when they possess the necessary perceptual and reasoning skills. Explicit prompting reveals that the underlying capabilities exist but are often suppressed in favor of user compliance. We further show that simple inference-time interventions, such as cautious persona prompting and, in particular, requiring a clarifying question, can dramatically recover performance. Our findings highlight a persistent gap between reasoning competence and behavioral compliance in current MLLMs and suggest practical strategies for making these models more trustworthy in underconstrained environments. 

---
# MIR: Methodology Inspiration Retrieval for Scientific Research Problems 

**Authors**: Aniketh Garikaparthi, Manasi Patwardhan, Aditya Sanjiv Kanade, Aman Hassan, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00249)  

**Abstract**: There has been a surge of interest in harnessing the reasoning capabilities of Large Language Models (LLMs) to accelerate scientific discovery. While existing approaches rely on grounding the discovery process within the relevant literature, effectiveness varies significantly with the quality and nature of the retrieved literature. We address the challenge of retrieving prior work whose concepts can inspire solutions for a given research problem, a task we define as Methodology Inspiration Retrieval (MIR). We construct a novel dataset tailored for training and evaluating retrievers on MIR, and establish baselines. To address MIR, we build the Methodology Adjacency Graph (MAG); capturing methodological lineage through citation relationships. We leverage MAG to embed an "intuitive prior" into dense retrievers for identifying patterns of methodological inspiration beyond superficial semantic similarity. This achieves significant gains of +5.4 in Recall@3 and +7.8 in Mean Average Precision (mAP) over strong baselines. Further, we adapt LLM-based re-ranking strategies to MIR, yielding additional improvements of +4.5 in Recall@3 and +4.8 in mAP. Through extensive ablation studies and qualitative analyses, we exhibit the promise of MIR in enhancing automated scientific discovery and outline avenues for advancing inspiration-driven retrieval. 

---
# Whispers of Many Shores: Cultural Alignment through Collaborative Cultural Expertise 

**Authors**: Shuai Feng, Wei-Chuang Chan, Srishti Chouhan, Junior Francisco Garcia Ayala, Srujananjali Medicherla, Kyle Clark, Mingwei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00242)  

**Abstract**: The integration of large language models (LLMs) into global applications necessitates effective cultural alignment for meaningful and culturally-sensitive interactions. Current LLMs often lack the nuanced understanding required for diverse cultural contexts, and adapting them typically involves costly full fine-tuning. To address this, we introduce a novel soft prompt fine-tuning framework that enables efficient and modular cultural alignment. Our method utilizes vectorized prompt tuning to dynamically route queries to a committee of culturally specialized 'expert' LLM configurations, created by optimizing soft prompt embeddings without altering the base model's parameters. Extensive experiments demonstrate that our framework significantly enhances cultural sensitivity and adaptability, improving alignment scores from 0.208 to 0.820, offering a robust solution for culturally-aware LLM deployment. This research paves the way for subsequent investigations into enhanced cultural coverage and dynamic expert adaptation, crucial for realizing autonomous AI with deeply nuanced understanding in a globally interconnected world. 

---
# SMELLNET: A Large-scale Dataset for Real-world Smell Recognition 

**Authors**: Dewei Feng, Carol Li, Wei Dai, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00239)  

**Abstract**: The ability of AI to sense and identify various substances based on their smell alone can have profound impacts on allergen detection (e.g., smelling gluten or peanuts in a cake), monitoring the manufacturing process, and sensing hormones that indicate emotional states, stress levels, and diseases. Despite these broad impacts, there are virtually no large scale benchmarks, and therefore little progress, for training and evaluating AI systems' ability to smell in the real world. In this paper, we use portable gas and chemical sensors to create SmellNet, the first large-scale database that digitizes a diverse range of smells in the natural world. SmellNet contains about 180,000 time steps of 50 substances (spanning nuts, spices, herbs, fruits, and vegetables) with 50 hours of data. Using SmellNet, we train AI models for real-time classification of substances based on their smell alone. Our best methods leverage sequence models, contrastive learning to integrate high-resolution Gas Chromatography-Mass Spectrometry molecular data, and a new temporal difference method that identifies sharp changes in sensor readings. Our best models achieve up to 65.35% accuracy on pre-recorded data, and generalize to real-world conditions with 10.71% accuracy on nuts and 25.38% on spices in the challenging 50-way online classification task. Despite these promising results, SmellNet highlights many technical challenges in building AI for smell, including richer feature learning, on-edge smell models, and robustness to environmental changes. 

---
# Ethical AI: Towards Defining a Collective Evaluation Framework 

**Authors**: Aasish Kumar Sharma, Dimitar Kyosev, Julian Kunkel  

**Link**: [PDF](https://arxiv.org/pdf/2506.00233)  

**Abstract**: Artificial Intelligence (AI) is transforming sectors such as healthcare, finance, and autonomous systems, offering powerful tools for innovation. Yet its rapid integration raises urgent ethical concerns related to data ownership, privacy, and systemic bias. Issues like opaque decision-making, misleading outputs, and unfair treatment in high-stakes domains underscore the need for transparent and accountable AI systems. This article addresses these challenges by proposing a modular ethical assessment framework built on ontological blocks of meaning-discrete, interpretable units that encode ethical principles such as fairness, accountability, and ownership. By integrating these blocks with FAIR (Findable, Accessible, Interoperable, Reusable) principles, the framework supports scalable, transparent, and legally aligned ethical evaluations, including compliance with the EU AI Act. Using a real-world use case in AI-powered investor profiling, the paper demonstrates how the framework enables dynamic, behavior-informed risk classification. The findings suggest that ontological blocks offer a promising path toward explainable and auditable AI ethics, though challenges remain in automation and probabilistic reasoning. 

---
# What do professional software developers need to know to succeed in an age of Artificial Intelligence? 

**Authors**: Matthew Kam, Cody Miller, Miaoxin Wang, Abey Tidwell, Irene A. Lee, Joyce Malyn-Smith, Beatriz Perez, Vikram Tiwari, Joshua Kenitzer, Andrew Macvean, Erin Barrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00202)  

**Abstract**: Generative AI is showing early evidence of productivity gains for software developers, but concerns persist regarding workforce disruption and deskilling. We describe our research with 21 developers at the cutting edge of using AI, summarizing 12 of their work goals we uncovered, together with 75 associated tasks and the skills & knowledge for each, illustrating how developers use AI at work. From all of these, we distilled our findings in the form of 5 insights. We found that the skills & knowledge to be a successful AI-enhanced developer are organized into four domains (using Generative AI effectively, core software engineering, adjacent engineering, and adjacent non-engineering) deployed at critical junctures throughout a 6-step task workflow. In order to "future proof" developers for this age of AI, on-the-job learning initiatives and computer science degree programs will need to target both "soft" skills and the technical skills & knowledge in all four domains to reskill, upskill and safeguard against deskilling. 

---
# Control-R: Towards controllable test-time scaling 

**Authors**: Di Zhang, Weida Wang, Junxian Li, Xunzhi Wang, Jiatong Li, Jianbo Wu, Jingdi Lei, Haonan He, Peng Ye, Shufei Zhang, Wanli Ouyang, Yuqiang Li, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00189)  

**Abstract**: This paper target in addressing the challenges of underthinking and overthinking in long chain-of-thought (CoT) reasoning for Large Reasoning Models (LRMs) by introducing Reasoning Control Fields (RCF)--a novel test-time approach that injects structured control signals to guide reasoning from a tree search perspective. RCF enables models to adjust reasoning effort according to given control conditions when solving complex tasks. Additionally, we present the Control-R-4K dataset, which consists of challenging problems annotated with detailed reasoning processes and corresponding control fields. To further enhance reasoning control, we propose a Conditional Distillation Finetuning (CDF) method, which trains model--particularly Control-R-32B--to effectively adjust reasoning effort during test time. Experimental results on benchmarks such as AIME2024 and MATH500 demonstrate that our approach achieves state-of-the-art performance at the 32B scale while enabling a controllable Long CoT reasoning process (L-CoT). Overall, this work introduces an effective paradigm for controllable test-time scaling reasoning. 

---
# Tournament of Prompts: Evolving LLM Instructions Through Structured Debates and Elo Ratings 

**Authors**: Anirudh Nair, Adi Banerjee, Laurent Mombaerts, Matthew Hagen, Tarik Borogovac  

**Link**: [PDF](https://arxiv.org/pdf/2506.00178)  

**Abstract**: Prompt engineering represents a critical bottleneck to harness the full potential of Large Language Models (LLMs) for solving complex tasks, as it requires specialized expertise, significant trial-and-error, and manual intervention. This challenge is particularly pronounced for tasks involving subjective quality assessment, where defining explicit optimization objectives becomes fundamentally problematic. Existing automated prompt optimization methods falter in these scenarios, as they typically require well-defined task-specific numerical fitness functions or rely on generic templates that cannot capture the nuanced requirements of complex use cases. We introduce DEEVO (DEbate-driven EVOlutionary prompt optimization), a novel framework that guides prompt evolution through a debate-driven evaluation with an Elo-based selection. Contrary to prior work, DEEVOs approach enables exploration of the discrete prompt space while preserving semantic coherence through intelligent crossover and strategic mutation operations that incorporate debate-based feedback, combining elements from both successful and unsuccessful prompts based on identified strengths rather than arbitrary splicing. Using Elo ratings as a fitness proxy, DEEVO simultaneously drives improvement and preserves valuable diversity in the prompt population. Experimental results demonstrate that DEEVO significantly outperforms both manual prompt engineering and alternative state-of-the-art optimization approaches on open-ended tasks and close-ended tasks despite using no ground truth feedback. By connecting LLMs reasoning capabilities with adaptive optimization, DEEVO represents a significant advancement in prompt optimization research by eliminating the need of predetermined metrics to continuously improve AI systems. 

---
# Utilizing AI for Aviation Post-Accident Analysis Classification 

**Authors**: Aziida Nanyonga, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2506.00169)  

**Abstract**: The volume of textual data available in aviation safety reports presents a challenge for timely and accurate analysis. This paper examines how Artificial Intelligence (AI) and, specifically, Natural Language Processing (NLP) can automate the process of extracting valuable insights from this data, ultimately enhancing aviation safety. The paper reviews ongoing efforts focused on the application of NLP and deep learning to aviation safety reports, with the goal of classifying the level of damage to an aircraft and identifying the phase of flight during which safety occurrences happen. Additionally, the paper explores the use of Topic Modeling (TM) to uncover latent thematic structures within aviation incident reports, aiming to identify recurring patterns and potential areas for safety improvement. The paper compares and contrasts the performance of various deep learning models and TM techniques applied to datasets from the National Transportation Safety Board (NTSB) and the Australian Transport Safety Bureau (ATSB), as well as the Aviation Safety Network (ASN), discussing the impact of dataset size and source on the accuracy of the analysis. The findings demonstrate that both NLP and deep learning, as well as TM, can significantly improve the efficiency and accuracy of aviation safety analysis, paving the way for more proactive safety management and risk mitigation strategies. 

---
# Balancing Profit and Fairness in Risk-Based Pricing Markets 

**Authors**: Jesse Thibodeau, Hadi Nekoei, Afaf Taïk, Janarthanan Rajendran, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00140)  

**Abstract**: Dynamic, risk-based pricing can systematically exclude vulnerable consumer groups from essential resources such as health insurance and consumer credit. We show that a regulator can realign private incentives with social objectives through a learned, interpretable tax schedule. First, we provide a formal proposition that bounding each firm's \emph{local} demographic gap implicitly bounds the \emph{global} opt-out disparity, motivating firm-level penalties. Building on this insight we introduce \texttt{MarketSim} -- an open-source, scalable simulator of heterogeneous consumers and profit-maximizing firms -- and train a reinforcement learning (RL) social planner (SP) that selects a bracketed fairness-tax while remaining close to a simple linear prior via an $\mathcal{L}_1$ regularizer. The learned policy is thus both transparent and easily interpretable. In two empirically calibrated markets, i.e., U.S. health-insurance and consumer-credit, our planner simultaneously raises demand-fairness by up to $16\%$ relative to unregulated Free Market while outperforming a fixed linear schedule in terms of social welfare without explicit coordination. These results illustrate how AI-assisted regulation can convert a competitive social dilemma into a win-win equilibrium, providing a principled and practical framework for fairness-aware market oversight. 

---
# The Automated but Risky Game: Modeling Agent-to-Agent Negotiations and Transactions in Consumer Markets 

**Authors**: Shenzhe Zhu, Jiao Sun, Yi Nian, Tobin South, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00073)  

**Abstract**: AI agents are increasingly used in consumer-facing applications to assist with tasks such as product search, negotiation, and transaction execution. In this paper, we explore a future scenario where both consumers and merchants authorize AI agents to fully automate negotiations and transactions. We aim to answer two key questions: (1) Do different LLM agents vary in their ability to secure favorable deals for users? (2) What risks arise from fully automating deal-making with AI agents in consumer markets? To address these questions, we develop an experimental framework that evaluates the performance of various LLM agents in real-world negotiation and transaction settings. Our findings reveal that AI-mediated deal-making is an inherently imbalanced game -- different agents achieve significantly different outcomes for their users. Moreover, behavioral anomalies in LLMs can result in financial losses for both consumers and merchants, such as overspending or accepting unreasonable deals. These results underscore that while automation can improve efficiency, it also introduces substantial risks. Users should exercise caution when delegating business decisions to AI agents. 

---
# Toward Knowledge-Guided AI for Inverse Design in Manufacturing: A Perspective on Domain, Physics, and Human-AI Synergy 

**Authors**: Hugon Lee, Hyeonbin Moon, Junhyeong Lee, Seunghwa RYu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00056)  

**Abstract**: Artificial intelligence (AI) is reshaping inverse design across manufacturing domain, enabling high-performance discovery in materials, products, and processes. However, purely data-driven approaches often struggle in realistic settings characterized by sparse data, high-dimensional design spaces, and nontrivial physical constraints. This perspective argues for a new generation of design systems that transcend black-box modeling by integrating domain knowledge, physics-informed learning, and intuitive human-AI interfaces. We first demonstrate how expert-guided sampling strategies enhance data efficiency and model generalization. Next, we discuss how physics-informed machine learning enables physically consistent modeling in data-scarce regimes. Finally, we explore how large language models emerge as interactive design agents connecting user intent with simulation tools, optimization pipelines, and collaborative workflows. Through illustrative examples and conceptual frameworks, we advocate that inverse design in manufacturing should evolve into a unified ecosystem, where domain knowledge, physical priors, and adaptive reasoning collectively enable scalable, interpretable, and accessible AI-driven design systems. 

---
# DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation 

**Authors**: Jennifer Chen, Aidar Myrzakhan, Yaxin Luo, Hassaan Muhammad Khan, Sondos Mahmoud Bsharat, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01954)  

**Abstract**: Retrieval-Augmented Generation (RAG) methods have proven highly effective for tasks requiring factual consistency and robust knowledge retrieval. However, large-scale RAG systems consume significant computational resources and are prone to generating hallucinated content from Humans. In this work, we introduce $\texttt{DRAG}$, a novel framework for distilling RAG knowledge from large-scale Language Models (LLMs) into small LMs (SLMs). Our approach leverages evidence- and knowledge graph-based distillation, ensuring that the distilled model retains critical factual knowledge while significantly reducing model size and computational cost. By aligning the smaller model's predictions with a structured knowledge graph and ranked evidence, $\texttt{DRAG}$ effectively mitigates hallucinations and improves factual accuracy. We further present a case demonstrating how our framework mitigates user privacy risks and introduce a corresponding benchmark. Experimental evaluations on multiple benchmarks demonstrate that our method outperforms the prior competitive RAG methods like MiniRAG for SLMs by up to 27.7% using the same models, preserving high-level efficiency and reliability. With $\texttt{DRAG}$, we provide a practical and resource-efficient roadmap to deploying enhanced retrieval and generation capabilities in small-sized LLMs. 

---
# WebChoreArena: Evaluating Web Browsing Agents on Realistic Tedious Web Tasks 

**Authors**: Atsuyuki Miyai, Zaiying Zhao, Kazuki Egashira, Atsuki Sato, Tatsumi Sunada, Shota Onohara, Hiromasa Yamanishi, Mashiro Toyooka, Kunato Nishina, Ryoma Maeda, Kiyoharu Aizawa, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.01952)  

**Abstract**: Powered by a large language model (LLM), a web browsing agent operates web browsers in a human-like manner and offers a highly transparent path toward automating a wide range of everyday tasks. As web agents become increasingly capable and demonstrate proficiency in general browsing tasks, a critical question emerges: Can they go beyond general browsing to robustly handle tasks that are tedious and complex, or chores that humans often avoid doing themselves? In this paper, we introduce WebChoreArena, a new fully reproducible benchmark comprising 532 carefully curated tasks designed to extend the scope of WebArena beyond general browsing to more labor-intensive and tedious tasks. WebChoreArena systematically integrates three key challenges: (i) Massive Memory tasks requiring accurate retrieval of large amounts of information in the observations, (ii) Calculation tasks demanding precise mathematical reasoning, and (iii) Long-Term Memory tasks necessitating long-term memory across multiple webpages. Built on top of the fully reproducible and widely adopted four WebArena simulation environments, WebChoreArena ensures strict reproducibility and enables fair, direct comparisons with the established WebArena benchmark, offering key insights into agent progress. Our experimental results demonstrate that as LLMs evolve, represented by GPT-4o, Claude 3.7 Sonnet, and Gemini 2.5 Pro, significant improvements in performance are observed on WebChoreArena. These findings suggest that WebChoreArena is well-suited to measure the advancement of state-of-the-art LLMs with greater clarity. Nevertheless, the results also indicate that even with Gemini 2.5 Pro, there remains substantial room for improvement compared to WebArena, highlighting the increased challenges posed by WebChoreArena. 

---
# Feel the Force: Contact-Driven Learning from Humans 

**Authors**: Ademi Adeniji, Zhuoran Chen, Vincent Liu, Venkatesh Pattabiraman, Raunaq Bhirangi, Siddhant Haldar, Pieter Abbeel, Lerrel Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.01944)  

**Abstract**: Controlling fine-grained forces during manipulation remains a core challenge in robotics. While robot policies learned from robot-collected data or simulation show promise, they struggle to generalize across the diverse range of real-world interactions. Learning directly from humans offers a scalable solution, enabling demonstrators to perform skills in their natural embodiment and in everyday environments. However, visual demonstrations alone lack the information needed to infer precise contact forces. We present FeelTheForce (FTF): a robot learning system that models human tactile behavior to learn force-sensitive manipulation. Using a tactile glove to measure contact forces and a vision-based model to estimate hand pose, we train a closed-loop policy that continuously predicts the forces needed for manipulation. This policy is re-targeted to a Franka Panda robot with tactile gripper sensors using shared visual and action representations. At execution, a PD controller modulates gripper closure to track predicted forces-enabling precise, force-aware control. Our approach grounds robust low-level force control in scalable human supervision, achieving a 77% success rate across 5 force-sensitive manipulation tasks. Code and videos are available at this https URL. 

---
# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning 

**Authors**: Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01939)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning. 

---
# Red Teaming AI Policy: A Taxonomy of Avoision and the EU AI Act 

**Authors**: Rui-Jie Yew, Bill Marino, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.01931)  

**Abstract**: The shape of AI regulation is beginning to emerge, most prominently through the EU AI Act (the "AIA"). By 2027, the AIA will be in full effect, and firms are starting to adjust their behavior in light of this new law. In this paper, we present a framework and taxonomy for reasoning about "avoision" -- conduct that walks the line between legal avoidance and evasion -- that firms might engage in so as to minimize the regulatory burden the AIA poses. We organize these avoision strategies around three "tiers" of increasing AIA exposure that regulated entities face depending on: whether their activities are (1) within scope of the AIA, (2) exempted from provisions of the AIA, or are (3) placed in a category with higher regulatory scrutiny. In each of these tiers and for each strategy, we specify the organizational and technological forms through which avoision may manifest. Our goal is to provide an adversarial framework for "red teaming" the AIA and AI regulation on the horizon. 

---
# Image Generation from Contextually-Contradictory Prompts 

**Authors**: Saar Huberman, Or Patashnik, Omer Dahary, Ron Mokady, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2506.01929)  

**Abstract**: Text-to-image diffusion models excel at generating high-quality, diverse images from natural language prompts. However, they often fail to produce semantically accurate results when the prompt contains concept combinations that contradict their learned priors. We define this failure mode as contextual contradiction, where one concept implicitly negates another due to entangled associations learned during training. To address this, we propose a stage-aware prompt decomposition framework that guides the denoising process using a sequence of proxy prompts. Each proxy prompt is constructed to match the semantic content expected to emerge at a specific stage of denoising, while ensuring contextual coherence. To construct these proxy prompts, we leverage a large language model (LLM) to analyze the target prompt, identify contradictions, and generate alternative expressions that preserve the original intent while resolving contextual conflicts. By aligning prompt information with the denoising progression, our method enables fine-grained semantic control and accurate image generation in the presence of contextual contradictions. Experiments across a variety of challenging prompts show substantial improvements in alignment to the textual prompt. 

---
# Online Competitive Information Gathering for Partially Observable Trajectory Games 

**Authors**: Mel Krusniak, Hang Xu, Parker Palermo, Forrest Laine  

**Link**: [PDF](https://arxiv.org/pdf/2506.01927)  

**Abstract**: Game-theoretic agents must make plans that optimally gather information about their opponents. These problems are modeled by partially observable stochastic games (POSGs), but planning in fully continuous POSGs is intractable without heavy offline computation or assumptions on the order of belief maintained by each player. We formulate a finite history/horizon refinement of POSGs which admits competitive information gathering behavior in trajectory space, and through a series of approximations, we present an online method for computing rational trajectory plans in these games which leverages particle-based estimations of the joint state space and performs stochastic gradient play. We also provide the necessary adjustments required to deploy this method on individual agents. The method is tested in continuous pursuit-evasion and warehouse-pickup scenarios (alongside extensions to $N > 2$ players and to more complex environments with visual and physical obstacles), demonstrating evidence of active information gathering and outperforming passive competitors. 

---
# TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation 

**Authors**: Amin Karimi Monsefi, Mridul Khurana, Rajiv Ramnath, Anuj Karpatne, Wei-Lun Chao, Cheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01923)  

**Abstract**: We propose TaxaDiffusion, a taxonomy-informed training framework for diffusion models to generate fine-grained animal images with high morphological and identity accuracy. Unlike standard approaches that treat each species as an independent category, TaxaDiffusion incorporates domain knowledge that many species exhibit strong visual similarities, with distinctions often residing in subtle variations of shape, pattern, and color. To exploit these relationships, TaxaDiffusion progressively trains conditioned diffusion models across different taxonomic levels -- starting from broad classifications such as Class and Order, refining through Family and Genus, and ultimately distinguishing at the Species level. This hierarchical learning strategy first captures coarse-grained morphological traits shared by species with common ancestors, facilitating knowledge transfer before refining fine-grained differences for species-level distinction. As a result, TaxaDiffusion enables accurate generation even with limited training samples per species. Extensive experiments on three fine-grained animal datasets demonstrate that outperforms existing approaches, achieving superior fidelity in fine-grained animal image generation. Project page: this https URL 

---
# MedEBench: Revisiting Text-instructed Image Editing 

**Authors**: Minghao Liu, Zhitao He, Zhiyuan Fan, Qingyun Wang, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2506.01921)  

**Abstract**: Text-guided image editing has seen rapid progress in natural image domains, but its adaptation to medical imaging remains limited and lacks standardized evaluation. Clinically, such editing holds promise for simulating surgical outcomes, creating personalized teaching materials, and enhancing patient communication. To bridge this gap, we introduce \textbf{MedEBench}, a comprehensive benchmark for evaluating text-guided medical image editing. It consists of 1,182 clinically sourced image-prompt triplets spanning 70 tasks across 13 anatomical regions. MedEBench offers three key contributions: (1) a clinically relevant evaluation framework covering Editing Accuracy, Contextual Preservation, and Visual Quality, supported by detailed descriptions of expected change and ROI (Region of Interest) masks; (2) a systematic comparison of seven state-of-the-art models, revealing common failure patterns; and (3) a failure analysis protocol based on attention grounding, using IoU between attention maps and ROIs to identify mislocalization. MedEBench provides a solid foundation for developing and evaluating reliable, clinically meaningful medical image editing systems. 

---
# Transformers as Multi-task Learners: Decoupling Features in Hidden Markov Models 

**Authors**: Yifan Hao, Chenlu Ye, Chi Han, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01919)  

**Abstract**: Transformer based models have shown remarkable capabilities in sequence learning across a wide range of tasks, often performing well on specific task by leveraging input-output examples. Despite their empirical success, a comprehensive theoretical understanding of this phenomenon remains limited. In this work, we investigate the layerwise behavior of Transformers to uncover the mechanisms underlying their multi-task generalization ability. Taking explorations on a typical sequence model, i.e, Hidden Markov Models, which are fundamental to many language tasks, we observe that: first, lower layers of Transformers focus on extracting feature representations, primarily influenced by neighboring tokens; second, on the upper layers, features become decoupled, exhibiting a high degree of time disentanglement. Building on these empirical insights, we provide theoretical analysis for the expressiveness power of Transformers. Our explicit constructions align closely with empirical observations, providing theoretical support for the Transformer's effectiveness and efficiency on sequence learning across diverse tasks. 

---
# CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for Alzheimer's Detection 

**Authors**: David Ortiz-Perez, Manuel Benavent-Lledo, Javier Rodriguez-Juan, Jose Garcia-Rodriguez, David Tomás  

**Link**: [PDF](https://arxiv.org/pdf/2506.01890)  

**Abstract**: Early detection of cognitive disorders such as Alzheimer's disease is critical for enabling timely clinical intervention and improving patient outcomes. In this work, we introduce CogniAlign, a multimodal architecture for Alzheimer's detection that integrates audio and textual modalities, two non-intrusive sources of information that offer complementary insights into cognitive health. Unlike prior approaches that fuse modalities at a coarse level, CogniAlign leverages a word-level temporal alignment strategy that synchronizes audio embeddings with corresponding textual tokens based on transcription timestamps. This alignment supports the development of token-level fusion techniques, enabling more precise cross-modal interactions. To fully exploit this alignment, we propose a Gated Cross-Attention Fusion mechanism, where audio features attend over textual representations, guided by the superior unimodal performance of the text modality. In addition, we incorporate prosodic cues, specifically interword pauses, by inserting pause tokens into the text and generating audio embeddings for silent intervals, further enriching both streams. We evaluate CogniAlign on the ADReSSo dataset, where it achieves an accuracy of 90.36%, outperforming existing state-of-the-art methods. A detailed ablation study confirms the advantages of our alignment strategy, attention-based fusion, and prosodic modeling. 

---
# Agnostic Reinforcement Learning: Foundations and Algorithms 

**Authors**: Gene Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01884)  

**Abstract**: Reinforcement Learning (RL) has demonstrated tremendous empirical success across numerous challenging domains. However, we lack a strong theoretical understanding of the statistical complexity of RL in environments with large state spaces, where function approximation is required for sample-efficient learning. This thesis addresses this gap by rigorously examining the statistical complexity of RL with function approximation from a learning theoretic perspective. Departing from a long history of prior work, we consider the weakest form of function approximation, called agnostic policy learning, in which the learner seeks to find the best policy in a given class $\Pi$, with no guarantee that $\Pi$ contains an optimal policy for the underlying task.
We systematically explore agnostic policy learning along three key axes: environment access -- how a learner collects data from the environment; coverage conditions -- intrinsic properties of the underlying MDP measuring the expansiveness of state-occupancy measures for policies in the class $\Pi$, and representational conditions -- structural assumptions on the class $\Pi$ itself. Within this comprehensive framework, we (1) design new learning algorithms with theoretical guarantees and (2) characterize fundamental performance bounds of any algorithm. Our results reveal significant statistical separations that highlight the power and limitations of agnostic policy learning. 

---
# scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics 

**Authors**: Davide D'Ascenzo, Sebastiano Cultrera di Montesano  

**Link**: [PDF](https://arxiv.org/pdf/2506.01883)  

**Abstract**: Modern single-cell datasets now comprise hundreds of millions of cells, presenting significant challenges for training deep learning models that require shuffled, memory-efficient data loading. While the AnnData format is the community standard for storing single-cell datasets, existing data loading solutions for AnnData are often inadequate: some require loading all data into memory, others convert to dense formats that increase storage demands, and many are hampered by slow random disk access. We present scDataset, a PyTorch IterableDataset that operates directly on one or more AnnData files without the need for format conversion. The core innovation is a combination of block sampling and batched fetching, which together balance randomness and I/O efficiency. On the Tahoe 100M dataset, scDataset achieves up to a 48$\times$ speed-up over AnnLoader, a 27$\times$ speed-up over HuggingFace Datasets, and an 18$\times$ speed-up over BioNeMo in single-core settings. These advances democratize large-scale single-cell model training for the broader research community. 

---
# Learning to Explore: An In-Context Learning Approach for Pure Exploration 

**Authors**: Alessio Russo, Ryan Welch, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2506.01876)  

**Abstract**: In this work, we study the active sequential hypothesis testing problem, also known as pure exploration, where the goal is to actively control a data collection process to efficiently identify the correct hypothesis underlying a decision problem. While relevant across multiple domains, devising adaptive exploration strategies remains challenging, particularly due to difficulties in encoding appropriate inductive biases. Existing Reinforcement Learning (RL)-based methods often underperform when relevant information structures are inadequately represented, whereas more complex methods, like Best Arm Identification (BAI) techniques, may be difficult to devise and typically rely on explicit modeling assumptions. To address these limitations, we introduce In-Context Pure Exploration (ICPE), an in-context learning approach that uses Transformers to learn exploration strategies directly from experience. ICPE combines supervised learning and reinforcement learning to identify and exploit latent structure across related tasks, without requiring prior assumptions. Numerical results across diverse synthetic and semi-synthetic benchmarks highlight ICPE's capability to achieve robust performance performance in deterministic, stochastic, and structured settings. These results demonstrate ICPE's ability to match optimal instance-dependent algorithms using only deep learning techniques, making it a practical and general approach to data-efficient exploration. 

---
# Frugal Machine Learning for Energy-efficient, and Resource-aware Artificial Intelligence 

**Authors**: John Violos, Konstantina-Christina Diamanti, Ioannis Kompatsiaris, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.01869)  

**Abstract**: Frugal Machine Learning (FML) refers to the practice of designing Machine Learning (ML) models that are efficient, cost-effective, and mindful of resource constraints. This field aims to achieve acceptable performance while minimizing the use of computational resources, time, energy, and data for both training and inference. FML strategies can be broadly categorized into input frugality, learning process frugality, and model frugality, each focusing on reducing resource consumption at different stages of the ML pipeline. This chapter explores recent advancements, applications, and open challenges in FML, emphasizing its importance for smart environments that incorporate edge computing and IoT devices, which often face strict limitations in bandwidth, energy, or latency. Technological enablers such as model compression, energy-efficient hardware, and data-efficient learning techniques are discussed, along with adaptive methods including parameter regularization, knowledge distillation, and dynamic architecture design that enable incremental model updates without full retraining. Furthermore, it provides a comprehensive taxonomy of frugal methods, discusses case studies across diverse domains, and identifies future research directions to drive innovation in this evolving field. 

---
# MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs 

**Authors**: Wayner Barrios, Andrés Villa, Juan León Alcázar, SouYoung Jin, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.01850)  

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on instruction-following tasks by integrating pretrained visual encoders with large language models (LLMs). However, existing approaches often struggle to ground fine-grained visual concepts in complex scenes. In this paper, we propose MoDA (Modulation Adapter), a lightweight yet effective module designed to refine pre-aligned visual features through instruction-guided modulation. Our approach follows the standard LLaVA training protocol, consisting of a two-stage process: (1) aligning image features to the LLMs input space via a frozen vision encoder and adapter layers, and (2) refining those features using the MoDA adapter during the instructional tuning stage. MoDA employs a Transformer-based cross-attention mechanism to generate a modulation mask over the aligned visual tokens, thereby emphasizing semantically relevant embedding dimensions based on the language instruction. The modulated features are then passed to the LLM for autoregressive language generation. Our experimental evaluation shows that MoDA improves visual grounding and generates more contextually appropriate responses, demonstrating its effectiveness as a general-purpose enhancement for image-based MLLMs. 

---
# CiteEval: Principle-Driven Citation Evaluation for Source Attribution 

**Authors**: Yumo Xu, Peng Qi, Jifan Chen, Kunlun Liu, Rujun Han, Lan Liu, Bonan Min, Vittorio Castelli, Arshit Gupta, Zhiguo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01829)  

**Abstract**: Citation quality is crucial in information-seeking systems, directly influencing trust and the effectiveness of information access. Current evaluation frameworks, both human and automatic, mainly rely on Natural Language Inference (NLI) to assess binary or ternary supportiveness from cited sources, which we argue is a suboptimal proxy for citation evaluation. In this work we introduce CiteEval, a citation evaluation framework driven by principles focusing on fine-grained citation assessment within a broad context, encompassing not only the cited sources but the full retrieval context, user query, and generated text. Guided by the proposed framework, we construct CiteBench, a multi-domain benchmark with high-quality human annotations on citation quality. To enable efficient evaluation, we further develop CiteEval-Auto, a suite of model-based metrics that exhibit strong correlation with human judgments. Experiments across diverse systems demonstrate CiteEval-Auto's superior ability to capture the multifaceted nature of citations compared to existing metrics, offering a principled and scalable approach to evaluate and improve model-generated citations. 

---
# A Quantum Information Theoretic Approach to Tractable Probabilistic Models 

**Authors**: Pedro Zuidberg Dos Martires  

**Link**: [PDF](https://arxiv.org/pdf/2506.01824)  

**Abstract**: By recursively nesting sums and products, probabilistic circuits have emerged in recent years as an attractive class of generative models as they enjoy, for instance, polytime marginalization of random variables. In this work we study these machine learning models using the framework of quantum information theory, leading to the introduction of positive unital circuits (PUnCs), which generalize circuit evaluations over positive real-valued probabilities to circuit evaluations over positive semi-definite matrices. As a consequence, PUnCs strictly generalize probabilistic circuits as well as recently introduced circuit classes such as PSD circuits. 

---
# Ridgeformer: Mutli-Stage Contrastive Training For Fine-grained Cross-Domain Fingerprint Recognition 

**Authors**: Shubham Pandey, Bhavin Jawade, Srirangaraj Setlur  

**Link**: [PDF](https://arxiv.org/pdf/2506.01806)  

**Abstract**: The increasing demand for hygienic and portable biometric systems has underscored the critical need for advancements in contactless fingerprint recognition. Despite its potential, this technology faces notable challenges, including out-of-focus image acquisition, reduced contrast between fingerprint ridges and valleys, variations in finger positioning, and perspective distortion. These factors significantly hinder the accuracy and reliability of contactless fingerprint matching. To address these issues, we propose a novel multi-stage transformer-based contactless fingerprint matching approach that first captures global spatial features and subsequently refines localized feature alignment across fingerprint samples. By employing a hierarchical feature extraction and matching pipeline, our method ensures fine-grained, cross-sample alignment while maintaining the robustness of global feature representation. We perform extensive evaluations on publicly available datasets such as HKPolyU and RidgeBase under different evaluation protocols, such as contactless-to-contact matching and contactless-to-contactless matching and demonstrate that our proposed approach outperforms existing methods, including COTS solutions. 

---
# Datasheets Aren't Enough: DataRubrics for Automated Quality Metrics and Accountability 

**Authors**: Genta Indra Winata, David Anugraha, Emmy Liu, Alham Fikri Aji, Shou-Yi Hung, Aditya Parashar, Patrick Amadeus Irawan, Ruochen Zhang, Zheng-Xin Yong, Jan Christian Blaise Cruz, Niklas Muennighoff, Seungone Kim, Hanyang Zhao, Sudipta Kar, Kezia Erina Suryoraharjo, M. Farid Adilazuarda, En-Shiun Annie Lee, Ayu Purwarianti, Derry Tanti Wijaya, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.01789)  

**Abstract**: High-quality datasets are fundamental to training and evaluating machine learning models, yet their creation-especially with accurate human annotations-remains a significant challenge. Many dataset paper submissions lack originality, diversity, or rigorous quality control, and these shortcomings are often overlooked during peer review. Submissions also frequently omit essential details about dataset construction and properties. While existing tools such as datasheets aim to promote transparency, they are largely descriptive and do not provide standardized, measurable methods for evaluating data quality. Similarly, metadata requirements at conferences promote accountability but are inconsistently enforced. To address these limitations, this position paper advocates for the integration of systematic, rubric-based evaluation metrics into the dataset review process-particularly as submission volumes continue to grow. We also explore scalable, cost-effective methods for synthetic data generation, including dedicated tools and LLM-as-a-judge approaches, to support more efficient evaluation. As a call to action, we introduce DataRubrics, a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation, DataRubrics offers a reproducible, scalable, and actionable solution for dataset quality assessment, enabling both authors and reviewers to uphold higher standards in data-centric research. We also release code to support reproducibility of LLM-based evaluations at this https URL. 

---
# iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering 

**Authors**: Shuai Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01784)  

**Abstract**: While Large Language Models (LLMs) excel at many natural language processing tasks, they often suffer from factual inaccuracies in knowledge-intensive scenarios. Integrating external knowledge resources, particularly knowledge graphs (KGs), provides a transparent and updatable foundation for more reliable reasoning. Knowledge Base Question Answering (KBQA), which queries and reasons over KGs, is central to this effort, especially for complex, multi-hop queries. However, multi-hop reasoning poses two key challenges: (1)~maintaining coherent reasoning paths, and (2)~avoiding prematurely discarding critical multi-hop connections. To address these issues, we introduce iQUEST, a question-guided KBQA framework that iteratively decomposes complex queries into simpler sub-questions, ensuring a structured and focused reasoning trajectory. Additionally, we integrate a Graph Neural Network (GNN) to look ahead and incorporate 2-hop neighbor information at each reasoning step. This dual approach strengthens the reasoning process, enabling the model to explore viable paths more effectively. Detailed experiments demonstrate the consistent improvement delivered by iQUEST across four benchmark datasets and four LLMs. 

---
# Systematic Hazard Analysis for Frontier AI using STPA 

**Authors**: Simon Mylius  

**Link**: [PDF](https://arxiv.org/pdf/2506.01782)  

**Abstract**: All of the frontier AI companies have published safety frameworks where they define capability thresholds and risk mitigations that determine how they will safely develop and deploy their models. Adoption of systematic approaches to risk modelling, based on established practices used in safety-critical industries, has been recommended, however frontier AI companies currently do not describe in detail any structured approach to identifying and analysing hazards. STPA (Systems-Theoretic Process Analysis) is a systematic methodology for identifying how complex systems can become unsafe, leading to hazards. It achieves this by mapping out controllers and controlled processes then analysing their interactions and feedback loops to understand how harmful outcomes could occur (Leveson & Thomas, 2018). We evaluate STPA's ability to broaden the scope, improve traceability and strengthen the robustness of safety assurance for frontier AI systems. Applying STPA to the threat model and scenario described in 'A Sketch of an AI Control Safety Case' (Korbak et al., 2025), we derive a list of Unsafe Control Actions. From these we select a subset and explore the Loss Scenarios that lead to them if left unmitigated. We find that STPA is able to identify causal factors that may be missed by unstructured hazard analysis methodologies thereby improving robustness. We suggest STPA could increase the safety assurance of frontier AI when used to complement or check coverage of existing AI governance techniques including capability thresholds, model evaluations and emergency procedures. The application of a systematic methodology supports scalability by increasing the proportion of the analysis that could be conducted by LLMs, reducing the burden on human domain experts. 

---
# Enhancing Customer Service Chatbots with Context-Aware NLU through Selective Attention and Multi-task Learning 

**Authors**: Subhadip Nandi, Neeraj Agrawal, Anshika Singh, Priyanka Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.01781)  

**Abstract**: Customer service chatbots are conversational systems aimed at addressing customer queries, often by directing them to automated workflows. A crucial aspect of this process is the classification of the customer's intent. Presently, most intent classification models for customer care utilise only customer query for intent prediction. This may result in low-accuracy models, which cannot handle ambiguous queries. An ambiguous query like "I didn't receive my package" could indicate a delayed order, or an order that was delivered but the customer failed to receive it. Resolution of each of these scenarios requires the execution of very different sequence of steps. Utilizing additional information, such as the customer's order delivery status, in the right manner can help identify the intent for such ambiguous queries. In this paper, we have introduced a context-aware NLU model that incorporates both, the customer query and contextual information from the customer's order status for predicting customer intent. A novel selective attention module is used to extract relevant context features. We have also proposed a multi-task learning paradigm for the effective utilization of different label types available in our training data. Our suggested method, Multi-Task Learning Contextual NLU with Selective Attention Weighted Context (MTL-CNLU-SAWC), yields a 4.8% increase in top 2 accuracy score over the baseline model which only uses user queries, and a 3.5% improvement over existing state-of-the-art models that combine query and context. We have deployed our model to production for Walmart's customer care domain. Accurate intent prediction through MTL-CNLU-SAWC helps to better direct customers to automated workflows, thereby significantly reducing escalations to human agents, leading to almost a million dollars in yearly savings for the company. 

---
# unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning 

**Authors**: Yafei Yang, Zihui Zhang, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01778)  

**Abstract**: We study the challenging problem of unsupervised multi-object segmentation on single images. Existing methods, which rely on image reconstruction objectives to learn objectness or leverage pretrained image features to group similar pixels, often succeed only in segmenting simple synthetic objects or discovering a limited number of real-world objects. In this paper, we introduce unMORE, a novel two-stage pipeline designed to identify many complex objects in real-world images. The key to our approach involves explicitly learning three levels of carefully defined object-centric representations in the first stage. Subsequently, our multi-object reasoning module utilizes these learned object priors to discover multiple objects in the second stage. Notably, this reasoning module is entirely network-free and does not require human labels. Extensive experiments demonstrate that unMORE significantly outperforms all existing unsupervised methods across 6 real-world benchmark datasets, including the challenging COCO dataset, achieving state-of-the-art object segmentation results. Remarkably, our method excels in crowded images where all baselines collapse. 

---
# MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation 

**Authors**: Yile Liu, Ziwei Ma, Xiu Jiang, Jinglu Hu, Jing Chang, Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01776)  

**Abstract**: With the rapid adoption of large language models (LLMs) in natural language processing, the ability to follow instructions has emerged as a key metric for evaluating their practical utility. However, existing evaluation methods often focus on single-language scenarios, overlooking the challenges and differences present in multilingual and cross-lingual contexts. To address this gap, we introduce MaXIFE: a comprehensive evaluation benchmark designed to assess instruction-following capabilities across 23 languages with 1,667 verifiable instruction tasks. MaXIFE integrates both Rule-Based Evaluation and Model-Based Evaluation, ensuring a balance of efficiency and accuracy. We applied MaXIFE to evaluate several leading commercial and open-source LLMs, establishing baseline results for future comparisons. By providing a standardized tool for multilingual instruction-following evaluation, MaXIFE aims to advance research and development in natural language processing. 

---
# Greening AI-enabled Systems with Software Engineering: A Research Agenda for Environmentally Sustainable AI Practices 

**Authors**: Luís Cruz, João Paulo Fernandes, Maja H. Kirkeby, Silverio Martínez-Fernández, June Sallou, Hina Anwar, Enrique Barba Roque, Justus Bogner, Joel Castaño, Fernando Castor, Aadil Chasmawala, Simão Cunha, Daniel Feitosa, Alexandra González, Andreas Jedlitschka, Patricia Lago, Ana Oprescu, Pooja Rani, João Saraiva, Federica Sarro, Raghavendra Selvan, Karthik Vaidhyanathan, Roberto Verdecchia, Ivan P. Yamshchikov, Henry Muccini  

**Link**: [PDF](https://arxiv.org/pdf/2506.01774)  

**Abstract**: The environmental impact of Artificial Intelligence (AI)-enabled systems is increasing rapidly, and software engineering plays a critical role in developing sustainable solutions. The "Greening AI with Software Engineering" CECAM-Lorentz workshop (no. 1358, 2025) funded by the Centre Européen de Calcul Atomique et Moléculaire and the Lorentz Center, provided an interdisciplinary forum for 29 participants, from practitioners to academics, to share knowledge, ideas, practices, and current results dedicated to advancing green software and AI research. The workshop was held February 3-7, 2025, in Lausanne, Switzerland. Through keynotes, flash talks, and collaborative discussions, participants identified and prioritized key challenges for the field. These included energy assessment and standardization, benchmarking practices, sustainability-aware architectures, runtime adaptation, empirical methodologies, and education. This report presents a research agenda emerging from the workshop, outlining open research directions and practical recommendations to guide the development of environmentally sustainable AI-enabled systems rooted in software engineering principles. 

---
# ReGA: Representation-Guided Abstraction for Model-based Safeguarding of LLMs 

**Authors**: Zeming Wei, Chengcan Wu, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01770)  

**Abstract**: Large Language Models (LLMs) have achieved significant success in various tasks, yet concerns about their safety and security have emerged. In particular, they pose risks in generating harmful content and vulnerability to jailbreaking attacks. To analyze and monitor machine learning models, model-based analysis has demonstrated notable potential in stateful deep neural networks, yet suffers from scalability issues when extending to LLMs due to their vast feature spaces. In this paper, we propose ReGA, a model-based analysis framework with representation-guided abstraction, to safeguard LLMs against harmful prompts and generations. By leveraging safety-critical representations, which are low-dimensional directions emerging in hidden states that indicate safety-related concepts, ReGA effectively addresses the scalability issue when constructing the abstract model for safety modeling. Our comprehensive evaluation shows that ReGA performs sufficiently well in distinguishing between safe and harmful inputs, achieving an AUROC of 0.975 at the prompt level and 0.985 at the conversation level. Additionally, ReGA exhibits robustness to real-world attacks and generalization across different safety perspectives, outperforming existing safeguard paradigms in terms of interpretability and scalability. Overall, ReGA serves as an efficient and scalable solution to enhance LLM safety by integrating representation engineering with model-based abstraction, paving the way for new paradigms to utilize software insights for AI safety. Our code is available at this https URL. 

---
# Efficient Egocentric Action Recognition with Multimodal Data 

**Authors**: Marco Calzavara, Ard Kastrati, Matteo Macchini, Dushan Vasilevski, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2506.01757)  

**Abstract**: The increasing availability of wearable XR devices opens new perspectives for Egocentric Action Recognition (EAR) systems, which can provide deeper human understanding and situation awareness. However, deploying real-time algorithms on these devices can be challenging due to the inherent trade-offs between portability, battery life, and computational resources. In this work, we systematically analyze the impact of sampling frequency across different input modalities - RGB video and 3D hand pose - on egocentric action recognition performance and CPU usage. By exploring a range of configurations, we provide a comprehensive characterization of the trade-offs between accuracy and computational efficiency. Our findings reveal that reducing the sampling rate of RGB frames, when complemented with higher-frequency 3D hand pose input, can preserve high accuracy while significantly lowering CPU demands. Notably, we observe up to a 3x reduction in CPU usage with minimal to no loss in recognition performance. This highlights the potential of multimodal input strategies as a viable approach to achieving efficient, real-time EAR on XR devices. 

---
# Principled data augmentation for learning to solve quadratic programming problems 

**Authors**: Chendi Qian, Christopher Morris  

**Link**: [PDF](https://arxiv.org/pdf/2506.01728)  

**Abstract**: Linear and quadratic optimization are crucial in numerous real-world applications, from training machine learning models to integer-linear optimization. Recently, learning-to-optimize methods (L2O) for linear (LPs) or quadratic programs (QPs) using message-passing graph neural networks (MPNNs) have gained traction, promising lightweight, data-driven proxies for solving such optimization problems. For example, they replace the costly computation of strong branching scores in branch-and-bound solvers, requiring solving many such optimization problems. However, robust L2O MPNNs remain challenging in data-scarce settings, especially when addressing complex optimization problems such as QPs. This work introduces a principled approach to data augmentation tailored for QPs via MPNNs. Our method leverages theoretically justified data augmentation techniques to generate diverse yet optimality-preserving instances. Furthermore, we integrate these augmentations into a self-supervised learning framework based on contrastive learning, thereby pretraining MPNNs for enhanced performance on L2O tasks. Extensive experiments demonstrate that our approach improves generalization in supervised scenarios and facilitates effective transfer learning to related optimization problems. 

---
# Tug-of-war between idiom's figurative and literal meanings in LLMs 

**Authors**: Soyoung Oh, Xinting Huang, Mathis Pink, Michael Hahn, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01723)  

**Abstract**: Idioms present a unique challenge for language models due to their non-compositional figurative meanings, which often strongly diverge from the idiom's literal interpretation. This duality requires a model to learn representing and deciding between the two meanings to interpret an idiom in a figurative sense, or literally. In this paper, we employ tools from mechanistic interpretability to trace how a large pretrained causal transformer (LLama3.2-1B-base) deals with this ambiguity. We localize three steps of idiom processing: First, the idiom's figurative meaning is retrieved in early attention and MLP sublayers. We identify specific attention heads which boost the figurative meaning of the idiom while suppressing the idiom's literal interpretation. The model subsequently represents the figurative representation through an intermediate path. Meanwhile, a parallel bypass route forwards literal interpretation, ensuring that a both reading remain available. Overall, our findings provide a mechanistic evidence for idiom comprehension in an autoregressive transformer. 

---
# Data Pruning by Information Maximization 

**Authors**: Haoru Tan, Sitong Wu, Wei Huang, Shizhen Zhao, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01701)  

**Abstract**: In this paper, we present InfoMax, a novel data pruning method, also known as coreset selection, designed to maximize the information content of selected samples while minimizing redundancy. By doing so, InfoMax enhances the overall informativeness of the coreset. The information of individual samples is measured by importance scores, which capture their influence or difficulty in model learning. To quantify redundancy, we use pairwise sample similarities, based on the premise that similar samples contribute similarly to the learning process. We formalize the coreset selection problem as a discrete quadratic programming (DQP) task, with the objective of maximizing the total information content, represented as the sum of individual sample contributions minus the redundancies introduced by similar samples within the coreset. To ensure practical scalability, we introduce an efficient gradient-based solver, complemented by sparsification techniques applied to the similarity matrix and dataset partitioning strategies. This enables InfoMax to seamlessly scale to datasets with millions of samples. Extensive experiments demonstrate the superior performance of InfoMax in various data pruning tasks, including image classification, vision-language pre-training, and instruction tuning for large language models. 

---
# When LLMs Team Up: The Emergence of Collaborative Affective Computing 

**Authors**: Wenna Lai, Haoran Xie, Guandong Xu, Qing Li, S. Joe Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01698)  

**Abstract**: Affective Computing (AC) is essential in bridging the gap between human emotional experiences and machine understanding. Traditionally, AC tasks in natural language processing (NLP) have been approached through pipeline architectures, which often suffer from structure rigidity that leads to inefficiencies and limited adaptability. The advent of Large Language Models (LLMs) has revolutionized this field by offering a unified approach to affective understanding and generation tasks, enhancing the potential for dynamic, real-time interactions. However, LLMs face cognitive limitations in affective reasoning, such as misinterpreting cultural nuances or contextual emotions, and hallucination problems in decision-making. To address these challenges, recent research advocates for LLM-based collaboration systems that emphasize interactions among specialized models and LLMs, mimicking human-like affective intelligence through the synergy of emotional and rational thinking that aligns with Dual Process Theory in psychology. This survey aims to provide a comprehensive overview of LLM-based collaboration systems in AC, exploring from structured collaborations to autonomous collaborations. Specifically, it includes: (1) A systematic review of existing methods, focusing on collaboration strategies, mechanisms, key functions, and applications; (2) Experimental comparisons of collaboration strategies across representative tasks in affective understanding and generation; (3) An analysis highlighting the potential of these systems to enhance robustness and adaptability in complex affective reasoning; (4) A discussion of key challenges and future research directions to further advance the field. This work is the first to systematically explore collaborative intelligence with LLMs in AC, paving the way for more powerful applications that approach human-like social intelligence. 

---
# Overcoming Data Scarcity in Scanning Tunnelling Microscopy Image Segmentation 

**Authors**: Nikola L. Kolev, Max Trouton, Filippo Federici Canova, Geoff Thornton, David Z. Gao, Neil J. Curson, Taylor J. Z. Stock  

**Link**: [PDF](https://arxiv.org/pdf/2506.01678)  

**Abstract**: Scanning tunnelling microscopy (STM) is a powerful technique for imaging surfaces with atomic resolution, providing insight into physical and chemical processes at the level of single atoms and molecules. A regular task of STM image analysis is the identification and labelling of features of interest against a uniform background. Performing this manually is a labour-intensive task, requiring significant human effort. To reduce this burden, we propose an automated approach to the segmentation of STM images that uses both few-shot learning and unsupervised learning. Our technique offers greater flexibility compared to previous supervised methods; it removes the requirement for large manually annotated datasets and is thus easier to adapt to an unseen surface while still maintaining a high accuracy. We demonstrate the effectiveness of our approach by using it to recognise atomic features on three distinct surfaces: Si(001), Ge(001), and TiO$_2$(110), including adsorbed AsH$_3$ molecules on the silicon and germanium surfaces. Our model exhibits strong generalisation capabilities, and following initial training, can be adapted to unseen surfaces with as few as one additional labelled data point. This work is a significant step towards efficient and material-agnostic, automatic segmentation of STM images. 

---
# GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion 

**Authors**: Sunkyung Lee, Minjin Choi, Eunseong Choi, Hye-young Kim, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01673)  

**Abstract**: Generative recommendation is an emerging paradigm that leverages the extensive knowledge of large language models by formulating recommendations into a text-to-text generation task. However, existing studies face two key limitations in (i) incorporating implicit item relationships and (ii) utilizing rich yet lengthy item information. To address these challenges, we propose a Generative Recommender via semantic-Aware Multi-granular late fusion (GRAM), introducing two synergistic innovations. First, we design semantic-to-lexical translation to encode implicit hierarchical and collaborative item relationships into the vocabulary space of LLMs. Second, we present multi-granular late fusion to integrate rich semantics efficiently with minimal information loss. It employs separate encoders for multi-granular prompts, delaying the fusion until the decoding stage. Experiments on four benchmark datasets show that GRAM outperforms eight state-of-the-art generative recommendation models, achieving significant improvements of 11.5-16.0% in Recall@5 and 5.3-13.6% in NDCG@5. The source code is available at this https URL. 

---
# Synthesis of discrete-continuous quantum circuits with multimodal diffusion models 

**Authors**: Florian Fürrutter, Zohim Chandani, Ikko Hamamura, Hans J. Briegel, Gorka Muñoz-Gil  

**Link**: [PDF](https://arxiv.org/pdf/2506.01666)  

**Abstract**: Efficiently compiling quantum operations remains a major bottleneck in scaling quantum computing. Today's state-of-the-art methods achieve low compilation error by combining search algorithms with gradient-based parameter optimization, but they incur long runtimes and require multiple calls to quantum hardware or expensive classical simulations, making their scaling prohibitive. Recently, machine-learning models have emerged as an alternative, though they are currently restricted to discrete gate sets. Here, we introduce a multimodal denoising diffusion model that simultaneously generates a circuit's structure and its continuous parameters for compiling a target unitary. It leverages two independent diffusion processes, one for discrete gate selection and one for parameter prediction. We benchmark the model over different experiments, analyzing the method's accuracy across varying qubit counts, circuit depths, and proportions of parameterized gates. Finally, by exploiting its rapid circuit generation, we create large datasets of circuits for particular operations and use these to extract valuable heuristics that can help us discover new insights into quantum circuit synthesis. 

---
# Provably Safe Reinforcement Learning from Analytic Gradients 

**Authors**: Tim Walter, Hannah Markgraf, Jonathan Külz, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.01665)  

**Abstract**: Deploying autonomous robots in safety-critical applications requires safety guarantees. Provably safe reinforcement learning is an active field of research which aims to provide such guarantees using safeguards. These safeguards should be integrated during training to prevent a large sim-to-real gap. While there are several approaches for safeguarding sampling-based reinforcement learning, analytic gradient-based reinforcement learning often achieves superior performance and sample efficiency. However, there is no safeguarding approach for this learning paradigm yet. Our work addresses this gap by developing the first effective safeguard for analytic gradient-based reinforcement learning. We analyse existing, differentiable safeguards, adapt them through modified mappings and gradient formulations, and integrate them with a state-of-the-art learning algorithm and a differentiable simulation. We evaluate how different safeguards affect policy optimisation using numerical experiments on two classical control tasks. The results demonstrate safeguarded training without compromising performance. 

---
# Explainable AI Systems Must Be Contestable: Here's How to Make It Happen 

**Authors**: Catarina Moreira, Anna Palatkina, Dacia Braca, Dylan M. Walsh, Peter J. Leihn, Fang Chen, Nina C. Hubig  

**Link**: [PDF](https://arxiv.org/pdf/2506.01662)  

**Abstract**: As AI regulations around the world intensify their focus on system safety, contestability has become a mandatory, yet ill-defined, safeguard. In XAI, "contestability" remains an empty promise: no formal definition exists, no algorithm guarantees it, and practitioners lack concrete guidance to satisfy regulatory requirements. Grounded in a systematic literature review, this paper presents the first rigorous formal definition of contestability in explainable AI, directly aligned with stakeholder requirements and regulatory mandates. We introduce a modular framework of by-design and post-hoc mechanisms spanning human-centered interfaces, technical architectures, legal processes, and organizational workflows. To operationalize our framework, we propose the Contestability Assessment Scale, a composite metric built on more than twenty quantitative criteria. Through multiple case studies across diverse application domains, we reveal where state-of-the-art systems fall short and show how our framework drives targeted improvements. By converting contestability from regulatory theory into a practical framework, our work equips practitioners with the tools to embed genuine recourse and accountability into AI systems. 

---
# Engram Memory Encoding and Retrieval: A Neurocomputational Perspective 

**Authors**: Daniel Szelogowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.01659)  

**Abstract**: Despite substantial research into the biological basis of memory, the precise mechanisms by which experiences are encoded, stored, and retrieved in the brain remain incompletely understood. A growing body of evidence supports the engram theory, which posits that sparse populations of neurons undergo lasting physical and biochemical changes to support long-term memory. Yet, a comprehensive computational framework that integrates biological findings with mechanistic models remains elusive. This work synthesizes insights from cellular neuroscience and computational modeling to address key challenges in engram research: how engram neurons are identified and manipulated; how synaptic plasticity mechanisms contribute to stable memory traces; and how sparsity promotes efficient, interference-resistant representations. Relevant computational approaches -- such as sparse regularization, engram gating, and biologically inspired architectures like Sparse Distributed Memory and spiking neural networks -- are also examined. Together, these findings suggest that memory efficiency, capacity, and stability emerge from the interaction of plasticity and sparsity constraints. By integrating neurobiological and computational perspectives, this paper provides a comprehensive theoretical foundation for engram research and proposes a roadmap for future inquiry into the mechanisms underlying memory, with implications for the diagnosis and treatment of memory-related disorders. 

---
# ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge 

**Authors**: Chaoyue He, Xin Zhou, Yi Wu, Xinjia Yu, Yan Zhang, Lei Zhang, Di Wang, Shengfei Lyu, Hong Xu, Xiaoqiao Wang, Wei Liu, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01646)  

**Abstract**: We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1 136 multiple-choice questions generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting retrieval-augmented generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports and recommendation documents from seven authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of the model, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (ranging from 0.5 B to 671 B parameters) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies typically around 55--70\%, highlighting ESGenius's challenging nature for LLMs in interdisciplinary contexts. However, models employing RAG show significant performance improvements, particularly for smaller models. For example, "DeepSeek-R1-Distill-Qwen-14B" improves from 63.82\% (zero-shot) to 80.46\% with RAG. These results underscore the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first benchmark curated for LLMs and the relevant enhancement technologies that focuses on ESG and sustainability topics. 

---
# Bidirectional Soft Actor-Critic: Leveraging Forward and Reverse KL Divergence for Efficient Reinforcement Learning 

**Authors**: Yixian Zhang, Huaze Tang, Changxu Wei, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.01639)  

**Abstract**: The Soft Actor-Critic (SAC) algorithm, a state-of-the-art method in maximum entropy reinforcement learning, traditionally relies on minimizing reverse Kullback-Leibler (KL) divergence for policy updates. However, this approach leads to an intractable optimal projection policy, necessitating gradient-based approximations that can suffer from instability and poor sample efficiency. This paper investigates the alternative use of forward KL divergence within SAC. We demonstrate that for Gaussian policies, forward KL divergence yields an explicit optimal projection policy -- corresponding to the mean and variance of the target Boltzmann distribution's action marginals. Building on the distinct advantages of both KL directions, we propose Bidirectional SAC, an algorithm that first initializes the policy using the explicit forward KL projection and then refines it by optimizing the reverse KL divergence. Comprehensive experiments on continuous control benchmarks show that Bidirectional SAC significantly outperforms standard SAC and other baselines, achieving up to a $30\%$ increase in episodic rewards, alongside enhanced sample efficiency. 

---
# Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification 

**Authors**: Zehao Wu, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01631)  

**Abstract**: As Large Language Models (LLMs) become integral software components in modern applications, unauthorized model derivations through fine-tuning, merging, and redistribution have emerged as critical software engineering challenges. Unlike traditional software where clone detection and license compliance are well-established, the LLM ecosystem lacks effective mechanisms to detect model lineage and enforce licensing agreements. This gap is particularly problematic when open-source model creators, such as Meta's LLaMA, require derivative works to maintain naming conventions for attribution, yet no technical means exist to verify compliance.
To fill this gap, treating LLMs as software artifacts requiring provenance tracking, we present TensorGuard, a gradient-based fingerprinting framework for LLM similarity detection and family classification. Our approach extracts model-intrinsic behavioral signatures by analyzing gradient responses to random input perturbations across tensor layers, operating independently of training data, watermarks, or specific model formats. TensorGuard supports the widely-adopted safetensors format and constructs high-dimensional fingerprints through statistical analysis of gradient features. These fingerprints enable two complementary capabilities: direct pairwise similarity assessment between arbitrary models through distance computation, and systematic family classification of unknown models via the K-Means clustering algorithm with domain-informed centroid initialization using known base models. Experimental evaluation on 58 models comprising 8 base models and 50 derivatives across five model families (Llama, Qwen, Gemma, Phi, Mistral) demonstrates 94% classification accuracy under our centroid-initialized K-Means clustering. 

---
# Robust Satisficing Gaussian Process Bandits Under Adversarial Attacks 

**Authors**: Artun Saday, Yaşar Cahit Yıldırım, Cem Tekin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01625)  

**Abstract**: We address the problem of Gaussian Process (GP) optimization in the presence of unknown and potentially varying adversarial perturbations. Unlike traditional robust optimization approaches that focus on maximizing performance under worst-case scenarios, we consider a robust satisficing objective, where the goal is to consistently achieve a predefined performance threshold $\tau$, even under adversarial conditions. We propose two novel algorithms based on distinct formulations of robust satisficing, and show that they are instances of a general robust satisficing framework. Further, each algorithm offers different guarantees depending on the nature of the adversary. Specifically, we derive two regret bounds: one that is sublinear over time, assuming certain conditions on the adversary and the satisficing threshold $\tau$, and another that scales with the perturbation magnitude but requires no assumptions on the adversary. Through extensive experiments, we demonstrate that our approach outperforms the established robust optimization methods in achieving the satisficing objective, particularly when the ambiguity set of the robust optimization framework is inaccurately specified. 

---
# Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech 

**Authors**: Karl El Hajal, Enno Hermann, Sevada Hovsepyan, Mathew Magimai.-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2506.01618)  

**Abstract**: Automatic speech recognition (ASR) systems struggle with dysarthric speech due to high inter-speaker variability and slow speaking rates. To address this, we explore dysarthric-to-healthy speech conversion for improved ASR performance. Our approach extends the Rhythm and Voice (RnV) conversion framework by introducing a syllable-based rhythm modeling method suited for dysarthric speech. We assess its impact on ASR by training LF-MMI models and fine-tuning Whisper on converted speech. Experiments on the Torgo corpus reveal that LF-MMI achieves significant word error rate reductions, especially for more severe cases of dysarthria, while fine-tuning Whisper on converted data has minimal effect on its performance. These results highlight the potential of unsupervised rhythm and voice conversion for dysarthric ASR. Code available at: this https URL 

---
# Contrastive Learning for Efficient Transaction Validation in UTXO-based Blockchains 

**Authors**: Hamid Attar, Luigi Lunardon, Alessio Pagani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01614)  

**Abstract**: This paper introduces a Machine Learning (ML) approach for scalability of UTXO-based blockchains, such as Bitcoin. Prior approaches to UTXO set sharding struggle with distributing UTXOs effectively across validators, creating substantial communication overhead due to child-parent transaction dependencies. This overhead, which arises from the need to locate parent UTXOs, significantly hampers transaction processing speeds. Our solution uses ML to optimize not only UTXO set sharding but also the routing of incoming transactions, ensuring that transactions are directed to shards containing their parent UTXOs. At the heart of our approach is a framework that combines contrastive and unsupervised learning to create an embedding space for transaction outputs. This embedding allows the model to group transaction outputs based on spending relationships, making it possible to route transactions efficiently to the correct validation microservices. Trained on historical transaction data with triplet loss and online semi-hard negative mining, the model embeds parent-child spending patterns directly into its parameters, thus eliminating the need for costly, real-time parent transaction lookups. This significantly reduces cross-shard communication overhead, boosting throughput and scalability. 

---
# EPFL-Smart-Kitchen-30: Densely annotated cooking dataset with 3D kinematics to challenge video and language models 

**Authors**: Andy Bonnetto, Haozhe Qi, Franklin Leong, Matea Tashkovska, Mahdi Rad, Solaiman Shokur, Friedhelm Hummel, Silvestro Micera, Marc Pollefeys, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2506.01608)  

**Abstract**: Understanding behavior requires datasets that capture humans while carrying out complex tasks. The kitchen is an excellent environment for assessing human motor and cognitive function, as many complex actions are naturally exhibited in kitchens from chopping to cleaning. Here, we introduce the EPFL-Smart-Kitchen-30 dataset, collected in a noninvasive motion capture platform inside a kitchen environment. Nine static RGB-D cameras, inertial measurement units (IMUs) and one head-mounted HoloLens~2 headset were used to capture 3D hand, body, and eye movements. The EPFL-Smart-Kitchen-30 dataset is a multi-view action dataset with synchronized exocentric, egocentric, depth, IMUs, eye gaze, body and hand kinematics spanning 29.7 hours of 16 subjects cooking four different recipes. Action sequences were densely annotated with 33.78 action segments per minute. Leveraging this multi-modal dataset, we propose four benchmarks to advance behavior understanding and modeling through 1) a vision-language benchmark, 2) a semantic text-to-motion generation benchmark, 3) a multi-modal action recognition benchmark, 4) a pose-based action segmentation benchmark. We expect the EPFL-Smart-Kitchen-30 dataset to pave the way for better methods as well as insights to understand the nature of ecologically-valid human behavior. Code and data are available at this https URL 

---
# WoMAP: World Models For Embodied Open-Vocabulary Object Localization 

**Authors**: Tenny Yin, Zhiting Mei, Tao Sun, Lihan Zha, Emily Zhou, Jeremy Bao, Miyu Yamane, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.01600)  

**Abstract**: Language-instructed active object localization is a critical challenge for robots, requiring efficient exploration of partially observable environments. However, state-of-the-art approaches either struggle to generalize beyond demonstration datasets (e.g., imitation learning methods) or fail to generate physically grounded actions (e.g., VLMs). To address these limitations, we introduce WoMAP (World Models for Active Perception): a recipe for training open-vocabulary object localization policies that: (i) uses a Gaussian Splatting-based real-to-sim-to-real pipeline for scalable data generation without the need for expert demonstrations, (ii) distills dense rewards signals from open-vocabulary object detectors, and (iii) leverages a latent world model for dynamics and rewards prediction to ground high-level action proposals at inference time. Rigorous simulation and hardware experiments demonstrate WoMAP's superior performance in a broad range of zero-shot object localization tasks, with more than 9x and 2x higher success rates compared to VLM and diffusion policy baselines, respectively. Further, we show that WoMAP achieves strong generalization and sim-to-real transfer on a TidyBot. 

---
# Policy Newton Algorithm in Reproducing Kernel Hilbert Space 

**Authors**: Yixian Zhang, Huaze Tang, Chao Wang, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.01597)  

**Abstract**: Reinforcement learning (RL) policies represented in Reproducing Kernel Hilbert Spaces (RKHS) offer powerful representational capabilities. While second-order optimization methods like Newton's method demonstrate faster convergence than first-order approaches, current RKHS-based policy optimization remains constrained to first-order techniques. This limitation stems primarily from the intractability of explicitly computing and inverting the infinite-dimensional Hessian operator in RKHS. We introduce Policy Newton in RKHS, the first second-order optimization framework specifically designed for RL policies represented in RKHS. Our approach circumvents direct computation of the inverse Hessian operator by optimizing a cubic regularized auxiliary objective function. Crucially, we leverage the Representer Theorem to transform this infinite-dimensional optimization into an equivalent, computationally tractable finite-dimensional problem whose dimensionality scales with the trajectory data volume. We establish theoretical guarantees proving convergence to a local optimum with a local quadratic convergence rate. Empirical evaluations on a toy financial asset allocation problem validate these theoretical properties, while experiments on standard RL benchmarks demonstrate that Policy Newton in RKHS achieves superior convergence speed and higher episodic rewards compared to established first-order RKHS approaches and parametric second-order methods. Our work bridges a critical gap between non-parametric policy representations and second-order optimization methods in reinforcement learning. 

---
# Understanding and Improving Laplacian Positional Encodings For Temporal GNNs 

**Authors**: Yaniv Galron, Fabrizio Frasca, Haggai Maron, Eran Treister, Moshe Eliasof  

**Link**: [PDF](https://arxiv.org/pdf/2506.01596)  

**Abstract**: Temporal graph learning has applications in recommendation systems, traffic forecasting, and social network analysis. Although multiple architectures have been introduced, progress in positional encoding for temporal graphs remains limited. Extending static Laplacian eigenvector approaches to temporal graphs through the supra-Laplacian has shown promise, but also poses key challenges: high eigendecomposition costs, limited theoretical understanding, and ambiguity about when and how to apply these encodings. In this paper, we address these issues by (1) offering a theoretical framework that connects supra-Laplacian encodings to per-time-slice encodings, highlighting the benefits of leveraging additional temporal connectivity, (2) introducing novel methods to reduce the computational overhead, achieving up to 56x faster runtimes while scaling to graphs with 50,000 active nodes, and (3) conducting an extensive experimental study to identify which models, tasks, and datasets benefit most from these encodings. Our findings reveal that while positional encodings can significantly boost performance in certain scenarios, their effectiveness varies across different models. 

---
# VirnyFlow: A Design Space for Responsible Model Development 

**Authors**: Denys Herasymuk, Nazar Protsiv, Julia Stoyanovich  

**Link**: [PDF](https://arxiv.org/pdf/2506.01584)  

**Abstract**: Developing machine learning (ML) models requires a deep understanding of real-world problems, which are inherently multi-objective. In this paper, we present VirnyFlow, the first design space for responsible model development, designed to assist data scientists in building ML pipelines that are tailored to the specific context of their problem. Unlike conventional AutoML frameworks, VirnyFlow enables users to define customized optimization criteria, perform comprehensive experimentation across pipeline stages, and iteratively refine models in alignment with real-world constraints. Our system integrates evaluation protocol definition, multi-objective Bayesian optimization, cost-aware multi-armed bandits, query optimization, and distributed parallelism into a unified architecture. We show that VirnyFlow significantly outperforms state-of-the-art AutoML systems in both optimization quality and scalability across five real-world benchmarks, offering a flexible, efficient, and responsible alternative to black-box automation in ML development. 

---
# FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens 

**Authors**: Yiming Zhong, Yumeng Liu, Chuyang Xiao, Zemin Yang, Youzhuo Wang, Yufei Zhu, Ye Shi, Yujing Sun, Xinge Zhu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.01583)  

**Abstract**: Learning effective visuomotor policies for robotic manipulation is challenging, as it requires generating precise actions while maintaining computational efficiency. Existing methods remain unsatisfactory due to inherent limitations in the essential action representation and the basic network architectures. We observe that representing actions in the frequency domain captures the structured nature of motion more effectively: low-frequency components reflect global movement patterns, while high-frequency components encode fine local details. Additionally, robotic manipulation tasks of varying complexity demand different levels of modeling precision across these frequency bands. Motivated by this, we propose a novel paradigm for visuomotor policy learning that progressively models hierarchical frequency components. To further enhance precision, we introduce continuous latent representations that maintain smoothness and continuity in the action space. Extensive experiments across diverse 2D and 3D robotic manipulation benchmarks demonstrate that our approach outperforms existing methods in both accuracy and efficiency, showcasing the potential of a frequency-domain autoregressive framework with continuous tokens for generalized robotic manipulation. 

---
# Advanced Nanostructured Topical Therapeutics for Psoriasis: Strategic Synthesis, Multimodal Characterization, and Preliminary Pharmacodynamic Profiling 

**Authors**: Iqra Yousaf, Aqsa Yousaf  

**Link**: [PDF](https://arxiv.org/pdf/2506.01572)  

**Abstract**: Psoriasis is a long-term inflammatory skin disease that remains difficult to treat. In this study, we developed a new topical treatment by combining metal oxide nanoparticles: cerium oxide (CeO2), zinc oxide (ZnO), and silver (Ag), with natural plant extracts in a gel made from fish collagen and agar. The nanoparticles were characterized using UV-Vis spectroscopy, dynamic light scattering (DLS), Fourier-transform infrared spectroscopy (FTIR), and scanning electron microscopy (SEM), showing good stability and a uniform particle size distribution (ZnO averaged 66 nm).
To enhance therapeutic potential, the gel was enriched with plant-derived antioxidants from bitter melon, ginger, and neem. This formulation was tested on an animal model of psoriasis. The treated group exhibited faster wound healing and reduced inflammation compared to both placebo and untreated groups, with statistically significant results (p < 0.01 to p < 0.001) observed from Day 3, becoming more pronounced by Day 14.
These results indicate that the combination of nanoparticles with plant-based components in a topical gel may provide a promising new approach to psoriasis treatment. Further studies are recommended to evaluate long-term safety and therapeutic effectiveness. 

---
# FlexiSAGA: A Flexible Systolic Array GEMM Accelerator for Sparse and Dense Processing 

**Authors**: Mika Markus Müller, Konstantin Lübeck, Alexander Louis-Ferdinand Jung, Jannik Steinmetz, Oliver Bringmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.01566)  

**Abstract**: Artificial Intelligence (AI) algorithms, such as Deep Neural Networks (DNNs), have become an important tool for a wide range of applications, from computer vision to natural language processing. However, the computational complexity of DNN inference poses a significant challenge, particularly for processing on resource-constrained edge devices. One promising approach to address this challenge is the exploitation of sparsity in DNN operator weights.
In this work, we present FlexiSAGA, an architecturally configurable and dataflow-flexible AI hardware accelerator for the sparse and dense processing of general matrix multiplications (GEMMs). FlexiSAGA supports seven different sparse and dense dataflows, enabling efficient processing of resource intensive DNN operators. Additionally, we propose a DNN pruning method specifically tailored towards the FlexiSAGA architecture, allowing for near-optimal processing of dense and sparse convolution and fully-connected operators, facilitating a DNN/HW co-design flow. Our results show a whole DNN sparse-over-dense inference speedup ranging from 1.41 up to 4.28, outperforming commercial and literature-reported accelerator platforms. 

---
# EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation 

**Authors**: Bingqian Lin, Yunshuang Nie, Khun Loun Zai, Ziming Wei, Mingfei Han, Rongtao Xu, Minzhe Niu, Jianhua Han, Liang Lin, Cewu Lu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01551)  

**Abstract**: Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at this https URL. 

---
# G4Seg: Generation for Inexact Segmentation Refinement with Diffusion Models 

**Authors**: Tianjiao Zhang, Fei Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01539)  

**Abstract**: This paper considers the problem of utilizing a large-scale text-to-image diffusion model to tackle the challenging Inexact Segmentation (IS) task. Unlike traditional approaches that rely heavily on discriminative-model-based paradigms or dense visual representations derived from internal attention mechanisms, our method focuses on the intrinsic generative priors in Stable Diffusion~(SD). Specifically, we exploit the pattern discrepancies between original images and mask-conditional generated images to facilitate a coarse-to-fine segmentation refinement by establishing a semantic correspondence alignment and updating the foreground probability. Comprehensive quantitative and qualitative experiments validate the effectiveness and superiority of our plug-and-play design, underscoring the potential of leveraging generation discrepancies to model dense representations and encouraging further exploration of generative approaches for solving discriminative tasks. 

---
# LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation 

**Authors**: Guobin Zhu, Rui Zhou, Wenkang Ji, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01538)  

**Abstract**: Although Multi-Agent Reinforcement Learning (MARL) is effective for complex multi-robot tasks, it suffers from low sample efficiency and requires iterative manual reward tuning. Large Language Models (LLMs) have shown promise in single-robot settings, but their application in multi-robot systems remains largely unexplored. This paper introduces a novel LLM-Aided MARL (LAMARL) approach, which integrates MARL with LLMs, significantly enhancing sample efficiency without requiring manual design. LAMARL consists of two modules: the first module leverages LLMs to fully automate the generation of prior policy and reward functions. The second module is MARL, which uses the generated functions to guide robot policy training effectively. On a shape assembly benchmark, both simulation and real-world experiments demonstrate the unique advantages of LAMARL. Ablation studies show that the prior policy improves sample efficiency by an average of 185.9% and enhances task completion, while structured prompts based on Chain-of-Thought (CoT) and basic APIs improve LLM output success rates by 28.5%-67.5%. Videos and code are available at this https URL 

---
# Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries 

**Authors**: Haruki Sakajo, Yusuke Ide, Justin Vasselli, Yusuke Sakai, Yingtao Tian, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.01535)  

**Abstract**: Cross-lingual vocabulary transfer plays a promising role in adapting pre-trained language models to new languages, including low-resource languages. Existing approaches that utilize monolingual or parallel corpora face challenges when applied to languages with limited resources. In this work, we propose a simple yet effective vocabulary transfer method that utilizes bilingual dictionaries, which are available for many languages, thanks to descriptive linguists. Our proposed method leverages a property of BPE tokenizers where removing a subword from the vocabulary causes a fallback to shorter subwords. The embeddings of target subwords are estimated iteratively by progressively removing them from the tokenizer. The experimental results show that our approach outperforms existing methods for low-resource languages, demonstrating the effectiveness of a dictionary-based approach for cross-lingual vocabulary transfer. 

---
# A Diffusion-Based Method for Learning the Multi-Outcome Distribution of Medical Treatments 

**Authors**: Yuchen Ma, Jonas Schweisthal, Hengrui Zhang, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2506.01533)  

**Abstract**: In medicine, treatments often influence multiple, interdependent outcomes, such as primary endpoints, complications, adverse events, or other secondary endpoints. Hence, to make optimal treatment decisions, clinicians are interested in learning the distribution of multi-dimensional treatment outcomes. However, the vast majority of machine learning methods for predicting treatment effects focus on single-outcome settings, despite the fact that medical data often include multiple, interdependent outcomes. To address this limitation, we propose a novel diffusion-based method called DIME to learn the joint distribution of multiple outcomes of medical treatments. We addresses three challenges relevant in medical practice: (i)it is tailored to learn the joint interventional distribution of multiple medical outcomes, which enables reliable decision-making with uncertainty quantification rather than relying solely on point estimates; (ii)it explicitly captures the dependence structure between outcomes; (iii)it can handle outcomes of mixed type, including binary, categorical, and continuous variables. In DIME, we take into account the fundamental problem of causal inference through causal masking. For training, our method decomposes the joint distribution into a series of conditional distributions with a customized conditional masking to account for the dependence structure across outcomes. For inference, our method auto-regressively generates predictions. This allows our method to move beyond point estimates of causal quantities and thus learn the joint interventional distribution. To the best of our knowledge, DIME is the first neural method tailored to learn the joint, multi-outcome distribution of medical treatments. Across various experiments, we demonstrate that our method effectively learns the joint distribution and captures shared information among multiple outcomes. 

---
# V-VAE: A Variational Auto Encoding Framework Towards Fine-Grained Control over Human-Like Chat 

**Authors**: Qi Lin, Weikai Xu, Lisi Chen, Bin Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.01524)  

**Abstract**: With the continued proliferation of Large Language Model (LLM) based chatbots, there is a growing demand for generating responses that are not only linguistically fluent but also consistently aligned with persona-specific traits in conversations. However, existing role-play and persona-based chat approaches rely heavily on static role descriptions, coarse-grained signal space, and low-quality synthetic data, which fail to capture dynamic fine-grained details in human-like chat. Human-like chat requires modeling subtle latent traits, such as emotional tone, situational awareness, and evolving personality, which are difficult to predefine and cannot be easily learned from synthetic or distillation-based data. To address these limitations, we propose a Verbal Variational Auto-Encoding (V-VAE) framework, containing a variational auto-encoding module and fine-grained control space which dynamically adapts dialogue behaviour based on fine-grained, interpretable latent variables across talking style, interaction patterns, and personal attributes. We also construct a high-quality dataset, HumanChatData, and benchmark HumanChatBench to address the scarcity of high-quality data in the human-like domain. Experiments show that LLMs based on V-VAE consistently outperform standard baselines on HumanChatBench and DialogBench, which further demonstrates the effectiveness of V-VAE and HumanChatData. 

---
# Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes 

**Authors**: Meng Li, Michael Vrazitulis, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01512)  

**Abstract**: Rational speakers are supposed to know what they know and what they do not know, and to generate expressions matching the strength of evidence. In contrast, it is still a challenge for current large language models to generate corresponding utterances based on the assessment of facts and confidence in an uncertain real-world environment. While it has recently become popular to estimate and calibrate confidence of LLMs with verbalized uncertainty, what is lacking is a careful examination of the linguistic knowledge of uncertainty encoded in the latent space of LLMs. In this paper, we draw on typological frameworks of epistemic expressions to evaluate LLMs' knowledge of epistemic modality, using controlled stories. Our experiments show that the performance of LLMs in generating epistemic expressions is limited and not robust, and hence the expressions of uncertainty generated by LLMs are not always reliable. To build uncertainty-aware LLMs, it is necessary to enrich semantic knowledge of epistemic modality in LLMs. 

---
# Learning of Population Dynamics: Inverse Optimization Meets JKO Scheme 

**Authors**: Mikhail Persiianov, Jiawei Chen, Petr Mokrov, Alexander Tyurin, Evgeny Burnaev, Alexander Korotin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01502)  

**Abstract**: Learning population dynamics involves recovering the underlying process that governs particle evolution, given evolutionary snapshots of samples at discrete time points. Recent methods frame this as an energy minimization problem in probability space and leverage the celebrated JKO scheme for efficient time discretization. In this work, we introduce $\texttt{iJKOnet}$, an approach that combines the JKO framework with inverse optimization techniques to learn population dynamics. Our method relies on a conventional $\textit{end-to-end}$ adversarial training procedure and does not require restrictive architectural choices, e.g., input-convex neural networks. We establish theoretical guarantees for our methodology and demonstrate improved performance over prior JKO-based methods. 

---
# Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task? 

**Authors**: Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01482)  

**Abstract**: Stage lighting plays an essential role in live music performances, influencing the engaging experience of both musicians and audiences. Given the high costs associated with hiring or training professional lighting engineers, Automatic Stage Lighting Control (ASLC) has gained increasing attention. However, most existing approaches only classify music into limited categories and map them to predefined light patterns, resulting in formulaic and monotonous outcomes that lack rationality. To address this issue, this paper presents an end-to-end solution that directly learns from experienced lighting engineers -- Skip-BART. To the best of our knowledge, this is the first work to conceptualize ASLC as a generative task rather than merely a classification problem. Our method modifies the BART model to take audio music as input and produce light hue and value (intensity) as output, incorporating a novel skip connection mechanism to enhance the relationship between music and light within the frame this http URL validate our method through both quantitative analysis and an human evaluation, demonstrating that Skip-BART outperforms conventional rule-based methods across all evaluation metrics and shows only a limited gap compared to real lighting this http URL, our method yields a p-value of 0.72 in a statistical comparison based on human evaluations with human lighting engineers, suggesting that the proposed approach closely matches human lighting engineering performance. To support further research, we have made our self-collected dataset, code, and trained model parameters available at this https URL . 

---
# Agentic AI and Multiagentic: Are We Reinventing the Wheel? 

**Authors**: V.Botti  

**Link**: [PDF](https://arxiv.org/pdf/2506.01463)  

**Abstract**: The terms Agentic AI and Multiagentic AI have recently gained popularity in discussions on generative artificial intelligence, often used to describe autonomous software agents and systems composed of such agents. However, the use of these terms confuses these buzzwords with well-established concepts in AI literature: intelligent agents and multi-agent systems. This article offers a critical analysis of this conceptual misuse. We review the theoretical origins of "agentic" in the social sciences (Bandura, 1986) and philosophical notions of intentionality (Dennett, 1971), and then summarise foundational works on intelligent agents and multi-agent systems by Wooldridge, Jennings and others. We examine classic agent architectures, from simple reactive agents to Belief-Desire-Intention (BDI) models, and highlight key properties (autonomy, reactivity, proactivity, social capability) that define agency in AI. We then discuss recent developments in large language models (LLMs) and agent platforms based on LLMs, including the emergence of LLM-powered AI agents and open-source multi-agent orchestration frameworks. We argue that the term AI Agentic is often used as a buzzword for what are essentially AI agents, and AI Multiagentic for what are multi-agent systems. This confusion overlooks decades of research in the field of autonomous agents and multi-agent systems. The article advocates for scientific and technological rigour and the use of established terminology from the state of the art in AI, incorporating the wealth of existing knowledge, including standards for multi-agent system platforms, communication languages and coordination and cooperation algorithms, agreement technologies (automated negotiation, argumentation, virtual organisations, trust, reputation, etc.), into the new and promising wave of LLM-based AI agents, so as not to end up reinventing the wheel. 

---
# GenDMR: A dynamic multimodal role-swapping network for identifying risk gene phenotypes 

**Authors**: Lina Qin, Cheng Zhu, Chuqi Zhou, Yukun Huang, Jiayi Zhu, Ping Liang, Jinju Wang, Yixing Huang, Cheng Luo, Dezhong Yao, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.01456)  

**Abstract**: Recent studies have shown that integrating multimodal data fusion techniques for imaging and genetic features is beneficial for the etiological analysis and predictive diagnosis of Alzheimer's disease (AD). However, there are several critical flaws in current deep learning methods. Firstly, there has been insufficient discussion and exploration regarding the selection and encoding of genetic information. Secondly, due to the significantly superior classification value of AD imaging features compared to genetic features, many studies in multimodal fusion emphasize the strengths of imaging features, actively mitigating the influence of weaker features, thereby diminishing the learning of the unique value of genetic features. To address this issue, this study proposes the dynamic multimodal role-swapping network (GenDMR). In GenDMR, we develop a novel approach to encode the spatial organization of single nucleotide polymorphisms (SNPs), enhancing the representation of their genomic context. Additionally, to adaptively quantify the disease risk of SNPs and brain region, we propose a multi-instance attention module to enhance model interpretability. Furthermore, we introduce a dominant modality selection module and a contrastive self-distillation module, combining them to achieve a dynamic teacher-student role exchange mechanism based on dominant and auxiliary modalities for bidirectional co-updating of different modal data. Finally, GenDMR achieves state-of-the-art performance on the ADNI public dataset and visualizes attention to different SNPs, focusing on confirming 12 potential high-risk genes related to AD, including the most classic APOE and recently highlighted significant risk genes. This demonstrates GenDMR's interpretable analytical capability in exploring AD genetic features, providing new insights and perspectives for the development of multimodal data fusion techniques. 

---
# From Initial Data to Boundary Layers: Neural Networks for Nonlinear Hyperbolic Conservation Laws 

**Authors**: Igor Ciril, Khalil Haddaoui, Yohann Tendero  

**Link**: [PDF](https://arxiv.org/pdf/2506.01453)  

**Abstract**: We address the approximation of entropy solutions to initial-boundary value problems for nonlinear strictly hyperbolic conservation laws using neural networks. A general and systematic framework is introduced for the design of efficient and reliable learning algorithms, combining fast convergence during training with accurate predictions. The methodology is assessed through a series of one-dimensional scalar test cases, highlighting its potential applicability to more complex industrial scenarios. 

---
# ShaTS: A Shapley-based Explainability Method for Time Series Artificial Intelligence Models applied to Anomaly Detection in Industrial Internet of Things 

**Authors**: Manuel Franco de la Peña, Ángel Luis Perales Gómez, Lorenzo Fernández Maimó  

**Link**: [PDF](https://arxiv.org/pdf/2506.01450)  

**Abstract**: Industrial Internet of Things environments increasingly rely on advanced Anomaly Detection and explanation techniques to rapidly detect and mitigate cyberincidents, thereby ensuring operational safety. The sequential nature of data collected from these environments has enabled improvements in Anomaly Detection using Machine Learning and Deep Learning models by processing time windows rather than treating the data as tabular. However, conventional explanation methods often neglect this temporal structure, leading to imprecise or less actionable explanations. This work presents ShaTS (Shapley values for Time Series models), which is a model-agnostic explainable Artificial Intelligence method designed to enhance the precision of Shapley value explanations for time series models. ShaTS addresses the shortcomings of traditional approaches by incorporating an a priori feature grouping strategy that preserves temporal dependencies and produces both coherent and actionable insights. Experiments conducted on the SWaT dataset demonstrate that ShaTS accurately identifies critical time instants, precisely pinpoints the sensors, actuators, and processes affected by anomalies, and outperforms SHAP in terms of both explainability and resource efficiency, fulfilling the real-time requirements of industrial environments. 

---
# Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models 

**Authors**: Yulei Qin, Gang Li, Zongyi Li, Zihan Xu, Yuchen Shi, Zhekai Lin, Xiao Cui, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01413)  

**Abstract**: Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Codes and data are available at this https URL. 

---
# System Calls for Malware Detection and Classification: Methodologies and Applications 

**Authors**: Bishwajit Prasad Gond, Durga Prasad Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2506.01412)  

**Abstract**: As malware continues to become more complex and harder to detect, Malware Analysis needs to continue to evolve to stay one step ahead. One promising key area approach focuses on using system calls and API Calls, the core communication between user applications and the operating system and their kernels. These calls provide valuable insight into how software or programs behaves, making them an useful tool for spotting suspicious or harmful activity of programs and software. This chapter takes a deep down look at how system calls are used in malware detection and classification, covering techniques like static and dynamic analysis, as well as sandboxing. By combining these methods with advanced techniques like machine learning, statistical analysis, and anomaly detection, researchers can analyze system call patterns to tell the difference between normal and malicious behavior. The chapter also explores how these techniques are applied across different systems, including Windows, Linux, and Android, while also looking at the ways sophisticated malware tries to evade detection. 

---
# ViTA-PAR: Visual and Textual Attribute Alignment with Attribute Prompting for Pedestrian Attribute Recognition 

**Authors**: Minjeong Park, Hongbeen Park, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01411)  

**Abstract**: The Pedestrian Attribute Recognition (PAR) task aims to identify various detailed attributes of an individual, such as clothing, accessories, and gender. To enhance PAR performance, a model must capture features ranging from coarse-grained global attributes (e.g., for identifying gender) to fine-grained local details (e.g., for recognizing accessories) that may appear in diverse regions. Recent research suggests that body part representation can enhance the model's robustness and accuracy, but these methods are often restricted to attribute classes within fixed horizontal regions, leading to degraded performance when attributes appear in varying or unexpected body locations. In this paper, we propose Visual and Textual Attribute Alignment with Attribute Prompting for Pedestrian Attribute Recognition, dubbed as ViTA-PAR, to enhance attribute recognition through specialized multimodal prompting and vision-language alignment. We introduce visual attribute prompts that capture global-to-local semantics, enabling diverse attribute representations. To enrich textual embeddings, we design a learnable prompt template, termed person and attribute context prompting, to learn person and attributes context. Finally, we align visual and textual attribute features for effective fusion. ViTA-PAR is validated on four PAR benchmarks, achieving competitive performance with efficient inference. We release our code and model at this https URL. 

---
# Sparse Imagination for Efficient Visual World Model Planning 

**Authors**: Junha Chun, Youngjoon Jeong, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01392)  

**Abstract**: World model based planning has significantly improved decision-making in complex environments by enabling agents to simulate future states and make informed choices. However, ensuring the prediction accuracy of world models often demands substantial computational resources, posing a major challenge for real-time applications. This computational burden is particularly restrictive in robotics, where resources are severely constrained. To address this limitation, we propose a Sparse Imagination for Efficient Visual World Model Planning, which enhances computational efficiency by reducing the number of tokens processed during forward prediction. Our method leverages a sparsely trained vision-based world model based on transformers with randomized grouped attention strategy, allowing the model to adaptively adjust the number of tokens processed based on the computational resource. By enabling sparse imagination (rollout), our approach significantly accelerates planning while maintaining high control fidelity. Experimental results demonstrate that sparse imagination preserves task performance while dramatically improving inference efficiency, paving the way for the deployment of world models in real-time decision-making scenarios. 

---
# VRD-IU: Lessons from Visually Rich Document Intelligence and Understanding 

**Authors**: Yihao Ding, Soyeon Caren Han, Yan Li, Josiah Poon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01388)  

**Abstract**: Visually Rich Document Understanding (VRDU) has emerged as a critical field in document intelligence, enabling automated extraction of key information from complex documents across domains such as medical, financial, and educational applications. However, form-like documents pose unique challenges due to their complex layouts, multi-stakeholder involvement, and high structural variability. Addressing these issues, the VRD-IU Competition was introduced, focusing on extracting and localizing key information from multi-format forms within the Form-NLU dataset, which includes digital, printed, and handwritten documents. This paper presents insights from the competition, which featured two tracks: Track A, emphasizing entity-based key information retrieval, and Track B, targeting end-to-end key information localization from raw document images. With over 20 participating teams, the competition showcased various state-of-the-art methodologies, including hierarchical decomposition, transformer-based retrieval, multimodal feature fusion, and advanced object detection techniques. The top-performing models set new benchmarks in VRDU, providing valuable insights into document intelligence. 

---
# Playing with Transformer at 30+ FPS via Next-Frame Diffusion 

**Authors**: Xinle Cheng, Tianyu He, Jiayi Xu, Junliang Guo, Di He, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.01380)  

**Abstract**: Autoregressive video models offer distinct advantages over bidirectional diffusion models in creating interactive video content and supporting streaming applications with arbitrary duration. In this work, we present Next-Frame Diffusion (NFD), an autoregressive diffusion transformer that incorporates block-wise causal attention, enabling iterative sampling and efficient inference via parallel token generation within each frame. Nonetheless, achieving real-time video generation remains a significant challenge for such models, primarily due to the high computational cost associated with diffusion sampling and the hardware inefficiencies inherent to autoregressive generation. To address this, we introduce two innovations: (1) We extend consistency distillation to the video domain and adapt it specifically for video models, enabling efficient inference with few sampling steps; (2) To fully leverage parallel computation, motivated by the observation that adjacent frames often share the identical action input, we propose speculative sampling. In this approach, the model generates next few frames using current action input, and discard speculatively generated frames if the input action differs. Experiments on a large-scale action-conditioned video generation benchmark demonstrate that NFD beats autoregressive baselines in terms of both visual quality and sampling efficiency. We, for the first time, achieves autoregressive video generation at over 30 Frames Per Second (FPS) on an A100 GPU using a 310M model. 

---
# Compiler Optimization via LLM Reasoning for Efficient Model Serving 

**Authors**: Sujun Tang, Christopher Priebe, Rohan Mahapatra, Lianhui Qin, Hadi Esmaeilzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.01374)  

**Abstract**: While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimization to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed REASONING COMPILER) that formulates optimization as a sequential, context-aware decision process, guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-aware transformations that reflect the current program state and accumulated performance feedback. Monte Carlo tree search (MCTS) incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization. 

---
# Incentivizing LLMs to Self-Verify Their Answers 

**Authors**: Fuxiang Zhang, Jiacheng Xu, Chaojie Wang, Ce Cui, Yang Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2506.01369)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress in complex reasoning tasks through both post-training and test-time scaling laws. While prevalent test-time scaling approaches are often realized by using external reward models to guide the model generation process, we find only marginal gains can be acquired when scaling a model post-trained on specific reasoning tasks. We identify that the limited improvement stems from distribution discrepancies between the specific post-trained generator and the general reward model. To address this, we propose a framework that incentivizes LLMs to self-verify their own answers. By unifying answer generation and verification within a single reinforcement learning (RL) process, we train models that can effectively assess the correctness of their own solutions. The trained model can further scale its performance during inference time by verifying its generations, without the need for external verifiers. We train our self-verification models based on Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-1.5B, demonstrating its capabilities across varying reasoning context lengths. Experiments on multiple mathematical reasoning benchmarks show that our models can not only improve post-training performance but also enable effective test-time scaling. Our code is available at this https URL. 

---
# Unraveling Spatio-Temporal Foundation Models via the Pipeline Lens: A Comprehensive Review 

**Authors**: Yuchen Fang, Hao Miao, Yuxuan Liang, Liwei Deng, Yue Cui, Ximu Zeng, Yuyang Xia, Yan Zhao, Torben Bach Pedersen, Christian S. Jensen, Xiaofang Zhou, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01364)  

**Abstract**: Spatio-temporal deep learning models aims to utilize useful patterns in such data to support tasks like prediction. However, previous deep learning models designed for specific tasks typically require separate training for each use case, leading to increased computational and storage costs. To address this issue, spatio-temporal foundation models have emerged, offering a unified framework capable of solving multiple spatio-temporal tasks. These foundation models achieve remarkable success by learning general knowledge with spatio-temporal data or transferring the general capabilities of pre-trained language models. While previous surveys have explored spatio-temporal data and methodologies separately, they have ignored a comprehensive examination of how foundation models are designed, selected, pre-trained, and adapted. As a result, the overall pipeline for spatio-temporal foundation models remains unclear. To bridge this gap, we innovatively provide an up-to-date review of previous spatio-temporal foundation models from the pipeline perspective. The pipeline begins with an introduction to different types of spatio-temporal data, followed by details of data preprocessing and embedding techniques. The pipeline then presents a novel data property taxonomy to divide existing methods according to data sources and dependencies, providing efficient and effective model design and selection for researchers. On this basis, we further illustrate the training objectives of primitive models, as well as the adaptation techniques of transferred models. Overall, our survey provides a clear and structured pipeline to understand the connection between core elements of spatio-temporal foundation models while guiding researchers to get started quickly. Additionally, we introduce emerging opportunities such as multi-objective training in the field of spatio-temporal foundation models. 

---
# KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors 

**Authors**: Zhiyang Qi, Takumasa Kaneko, Keiko Takamizo, Mariko Ukiyo, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2506.01357)  

**Abstract**: Generating psychological counseling responses with language models relies heavily on high-quality datasets. Crowdsourced data collection methods require strict worker training, and data from real-world counseling environments may raise privacy and ethical concerns. While recent studies have explored using large language models (LLMs) to augment psychological counseling dialogue datasets, the resulting data often suffers from limited diversity and authenticity. To address these limitations, this study adopts a role-playing approach where trained counselors simulate counselor-client interactions, ensuring high-quality dialogues while mitigating privacy risks. Using this method, we construct KokoroChat, a Japanese psychological counseling dialogue dataset comprising 6,589 long-form dialogues, each accompanied by comprehensive client feedback. Experimental results demonstrate that fine-tuning open-source LLMs with KokoroChat improves both the quality of generated counseling responses and the automatic evaluation of counseling dialogues. The KokoroChat dataset is available at this https URL. 

---
# NoiseAR: AutoRegressing Initial Noise Prior for Diffusion Models 

**Authors**: Zeming Li, Xiangyue Liu, Xiangyu Zhang, Ping Tan, Heung-Yeung Shum  

**Link**: [PDF](https://arxiv.org/pdf/2506.01337)  

**Abstract**: Diffusion models have emerged as powerful generative frameworks, creating data samples by progressively denoising an initial random state. Traditionally, this initial state is sampled from a simple, fixed distribution like isotropic Gaussian, inherently lacking structure and a direct mechanism for external control. While recent efforts have explored ways to introduce controllability into the diffusion process, particularly at the initialization stage, they often rely on deterministic or heuristic approaches. These methods can be suboptimal, lack expressiveness, and are difficult to scale or integrate into more sophisticated optimization frameworks. In this paper, we introduce NoiseAR, a novel method for AutoRegressive Initial Noise Prior for Diffusion Models. Instead of a static, unstructured source, NoiseAR learns to generate a dynamic and controllable prior distribution for the initial noise. We formulate the generation of the initial noise prior's parameters as an autoregressive probabilistic modeling task over spatial patches or tokens. This approach enables NoiseAR to capture complex spatial dependencies and introduce learned structure into the initial state. Crucially, NoiseAR is designed to be conditional, allowing text prompts to directly influence the learned prior, thereby achieving fine-grained control over the diffusion initialization. Our experiments demonstrate that NoiseAR can generate initial noise priors that lead to improved sample quality and enhanced consistency with conditional inputs, offering a powerful, learned alternative to traditional random initialization. A key advantage of NoiseAR is its probabilistic formulation, which naturally supports seamless integration into probabilistic frameworks like Markov Decision Processes and Reinforcement Learning. Our code will be available at this https URL 

---
# ETDI: Mitigating Tool Squatting and Rug Pull Attacks in Model Context Protocol (MCP) by using OAuth-Enhanced Tool Definitions and Policy-Based Access Control 

**Authors**: Manish Bhatt, Vineeth Sai Narajala, Idan Habler  

**Link**: [PDF](https://arxiv.org/pdf/2506.01333)  

**Abstract**: The Model Context Protocol (MCP) plays a crucial role in extending the capabilities of Large Language Models (LLMs) by enabling integration with external tools and data sources. However, the standard MCP specification presents significant security vulnerabilities, notably Tool Poisoning and Rug Pull attacks. This paper introduces the Enhanced Tool Definition Interface (ETDI), a security extension designed to fortify MCP. ETDI incorporates cryptographic identity verification, immutable versioned tool definitions, and explicit permission management, often leveraging OAuth 2.0. We further propose extending MCP with fine-grained, policy-based access control, where tool capabilities are dynamically evaluated against explicit policies using a dedicated policy engine, considering runtime context beyond static OAuth scopes. This layered approach aims to establish a more secure, trustworthy, and controllable ecosystem for AI applications interacting with LLMs and external tools. 

---
# Evaluating Large Language Models in Crisis Detection: A Real-World Benchmark from Psychological Support Hotlines 

**Authors**: Guifeng Deng, Shuyin Rao, Tianyu Lin, Anlu Dai, Pan Wang, Junyi Xie, Haidong Song, Ke Zhao, Dongwu Xu, Zhengdong Cheng, Tao Li, Haiteng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01329)  

**Abstract**: Psychological support hotlines are critical for crisis intervention but face significant challenges due to rising demand. Large language models (LLMs) could support crisis assessments, yet their capabilities in emotionally sensitive contexts remain unclear. We introduce PsyCrisisBench, a benchmark of 540 annotated transcripts from the Hangzhou Psychological Assistance Hotline, assessing four tasks: mood status recognition, suicidal ideation detection, suicide plan identification, and risk assessment. We evaluated 64 LLMs across 15 families (e.g., GPT, Claude, Gemini, Llama, Qwen, DeepSeek) using zero-shot, few-shot, and fine-tuning paradigms. Performance was measured by F1-score, with statistical comparisons via Welch's t-tests. LLMs performed strongly on suicidal ideation detection (F1=0.880), suicide plan identification (F1=0.779), and risk assessment (F1=0.907), improved with few-shot and fine-tuning. Mood status recognition was more challenging (max F1=0.709), likely due to lost vocal cues and ambiguity. A fine-tuned 1.5B-parameter model (Qwen2.5-1.5B) surpassed larger models on mood and suicidal ideation. Open-source models like QwQ-32B performed comparably to closed-source on most tasks (p>0.3), though closed models retained an edge in mood detection (p=0.007). Performance scaled with size up to a point; quantization (AWQ) reduced GPU memory by 70% with minimal F1 degradation. LLMs show substantial promise in structured psychological crisis assessments, especially with fine-tuning. Mood recognition remains limited due to contextual complexity. The narrowing gap between open- and closed-source models, combined with efficient quantization, suggests feasible integration. PsyCrisisBench offers a robust evaluation framework to guide model development and ethical deployment in mental health. 

---
# STSA: Federated Class-Incremental Learning via Spatial-Temporal Statistics Aggregation 

**Authors**: Zenghao Guan, Guojun Zhu, Yucan Zhou, Wu Liu, Weiping Wang, Jiebo Luo, Xiaoyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01327)  

**Abstract**: Federated Class-Incremental Learning (FCIL) enables Class-Incremental Learning (CIL) from distributed data. Existing FCIL methods typically integrate old knowledge preservation into local client training. However, these methods cannot avoid spatial-temporal client drift caused by data heterogeneity and often incur significant computational and communication overhead, limiting practical deployment. To address these challenges simultaneously, we propose a novel approach, Spatial-Temporal Statistics Aggregation (STSA), which provides a unified framework to aggregate feature statistics both spatially (across clients) and temporally (across stages). The aggregated feature statistics are unaffected by data heterogeneity and can be used to update the classifier in closed form at each stage. Additionally, we introduce STSA-E, a communication-efficient variant with theoretical guarantees, achieving similar performance to STSA-E with much lower communication overhead. Extensive experiments on three widely used FCIL datasets, with varying degrees of data heterogeneity, show that our method outperforms state-of-the-art FCIL methods in terms of performance, flexibility, and both communication and computation efficiency. 

---
# $Ψ$-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models 

**Authors**: Taehoon Yoon, Yunhong Min, Kyeongmin Yeo, Minhyuk Sung  

**Link**: [PDF](https://arxiv.org/pdf/2506.01320)  

**Abstract**: We introduce $\Psi$-Sampler, an SMC-based framework incorporating pCNL-based initial particle sampling for effective inference-time reward alignment with a score-based generative model. Inference-time reward alignment with score-based generative models has recently gained significant traction, following a broader paradigm shift from pre-training to post-training optimization. At the core of this trend is the application of Sequential Monte Carlo (SMC) to the denoising process. However, existing methods typically initialize particles from the Gaussian prior, which inadequately captures reward-relevant regions and results in reduced sampling efficiency. We demonstrate that initializing from the reward-aware posterior significantly improves alignment performance. To enable posterior sampling in high-dimensional latent spaces, we introduce the preconditioned Crank-Nicolson Langevin (pCNL) algorithm, which combines dimension-robust proposals with gradient-informed dynamics. This approach enables efficient and scalable posterior sampling and consistently improves performance across various reward alignment tasks, including layout-to-image generation, quantity-aware generation, and aesthetic-preference generation, as demonstrated in our experiments. 

---
# Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack 

**Authors**: SeungBum Ha, Saerom Park, Sung Whan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01318)  

**Abstract**: Machine unlearning (MU) aims to expunge a designated forget set from a trained model without costly retraining, yet the existing techniques overlook two critical blind spots: "over-unlearning" that deteriorates retained data near the forget set, and post-hoc "relearning" attacks that aim to resurrect the forgotten knowledge. We first derive the over-unlearning metric OU@{\epsilon}, which represents the collateral damage to the nearby region of the forget set, where the over-unlearning mainly appears. Next, we expose an unforeseen relearning threat on MU, i.e., the Prototypical Relearning Attack, which exploits the per-class prototype of the forget class with just a few samples, and easily restores the pre-unlearning performance. To counter both blind spots, we introduce Spotter, a plug-and-play objective that combines (i) a masked knowledge-distillation penalty on the nearby region of forget set to suppress OU@{\epsilon}, and (ii) an intra-class dispersion loss that scatters forget-class embeddings, neutralizing prototypical relearning attacks. On CIFAR-10, as one of validations, Spotter reduces OU@{\epsilon}by below the 0.05X of the baseline, drives forget accuracy to 0%, preserves accuracy of the retain set within 1% of difference with the original, and denies the prototype-attack by keeping the forget set accuracy within <1%, without accessing retained data. It confirms that Spotter is a practical remedy of the unlearning's blind spots. 

---
# T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning 

**Authors**: Yanjun Fu, Faisal Hamman, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2506.01317)  

**Abstract**: Instruction tuning is essential for Large Language Models (LLMs) to effectively follow user instructions. To improve training efficiency and reduce data redundancy, recent works use LLM-based scoring functions, e.g., Instruction-Following Difficulty (IFD), to select high-quality instruction-tuning data with scores above a threshold. While these data selection methods often lead to models that can match or even exceed the performance of models trained on the full datasets, we identify two key limitations: (i) they assess quality at the sample level, ignoring token-level informativeness; and (ii) they overlook the robustness of the scoring method, often selecting a sample due to superficial lexical features instead of its true quality. In this work, we propose Token-Selective HIeRarchical Data Selection for Instruction Tuning (T-SHIRT), a novel data selection framework that introduces a new scoring method to include only informative tokens in quality evaluation and also promotes robust and reliable samples whose neighbors also show high quality with less local inconsistencies. We demonstrate that models instruction-tuned on a curated dataset (only 5% of the original size) using T-SHIRT can outperform those trained on the entire large-scale dataset by up to 5.48 points on average across eight benchmarks. Across various LLMs and training set scales, our method consistently surpasses existing state-of-the-art data selection techniques, while also remaining both cost-effective and highly efficient. For instance, by using GPT-2 for score computation, we are able to process a dataset of 52k samples using 40 minutes on a single GPU. 

---
# Align is not Enough: Multimodal Universal Jailbreak Attack against Multimodal Large Language Models 

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Jing Liu, Hanwang Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01307)  

**Abstract**: Large Language Models (LLMs) have evolved into Multimodal Large Language Models (MLLMs), significantly enhancing their capabilities by integrating visual information and other types, thus aligning more closely with the nature of human intelligence, which processes a variety of data forms beyond just text. Despite advancements, the undesirable generation of these models remains a critical concern, particularly due to vulnerabilities exposed by text-based jailbreak attacks, which have represented a significant threat by challenging existing safety protocols. Motivated by the unique security risks posed by the integration of new and old modalities for MLLMs, we propose a unified multimodal universal jailbreak attack framework that leverages iterative image-text interactions and transfer-based strategy to generate a universal adversarial suffix and image. Our work not only highlights the interaction of image-text modalities can be used as a critical vulnerability but also validates that multimodal universal jailbreak attacks can bring higher-quality undesirable generations across different MLLMs. We evaluate the undesirable context generation of MLLMs like LLaVA, Yi-VL, MiniGPT4, MiniGPT-v2, and InstructBLIP, and reveal significant multimodal safety alignment issues, highlighting the inadequacy of current safety mechanisms against sophisticated multimodal attacks. This study underscores the urgent need for robust safety measures in MLLMs, advocating for a comprehensive review and enhancement of security protocols to mitigate potential risks associated with multimodal capabilities. 

---
# Abstractive Visual Understanding of Multi-modal Structured Knowledge: A New Perspective for MLLM Evaluation 

**Authors**: Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Min Zhang, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01293)  

**Abstract**: Multi-modal large language models (MLLMs) incorporate heterogeneous modalities into LLMs, enabling a comprehensive understanding of diverse scenarios and objects. Despite the proliferation of evaluation benchmarks and leaderboards for MLLMs, they predominantly overlook the critical capacity of MLLMs to comprehend world knowledge with structured abstractions that appear in visual form. To address this gap, we propose a novel evaluation paradigm and devise M3STR, an innovative benchmark grounded in the Multi-Modal Map for STRuctured understanding. This benchmark leverages multi-modal knowledge graphs to synthesize images encapsulating subgraph architectures enriched with multi-modal entities. M3STR necessitates that MLLMs not only recognize the multi-modal entities within the visual inputs but also decipher intricate relational topologies among them. We delineate the benchmark's statistical profiles and automated construction pipeline, accompanied by an extensive empirical analysis of 26 state-of-the-art MLLMs. Our findings reveal persistent deficiencies in processing abstractive visual information with structured knowledge, thereby charting a pivotal trajectory for advancing MLLMs' holistic reasoning capacities. Our code and data are released at this https URL 

---
# TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment 

**Authors**: Shunyu Wu, Dan Li, Haozheng Ye, Zhuomin Chen, Jiahui Zhou, Jian Lou, Zibin Zheng, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01290)  

**Abstract**: High-quality time series (TS) data are essential for ensuring TS model performance, rendering research on rating TS data quality indispensable. Existing methods have shown promising rating accuracy within individual domains, primarily by extending data quality rating techniques such as influence functions and Shapley values to account for temporal characteristics. However, they neglect the fact that real-world TS data can span vastly different domains and exhibit distinct properties, hampering the accurate and efficient rating of diverse TS data. In this paper, we propose TSRating, a novel and unified framework for rating the quality of time series data crawled from diverse domains. TSRating is built on the assumption that LLMs inherit ample knowledge, acquired during their extensive pretraining, enabling them to comprehend and discern quality differences in diverse TS data. We verify this assumption by devising a series of prompts to elicit quality comparisons from LLMs for pairs of TS samples. We then fit a dedicated rating model, termed TSRater, to convert the LLMs' judgments into efficient quality predictions via TSRater's inference on future TS samples. To ensure cross-domain adaptability, we develop a meta-learning scheme to train TSRater on quality comparisons collected from nine distinct domains. To improve training efficiency, we employ signSGD for inner-loop updates, thus circumventing the demanding computation of hypergradients. Extensive experimental results on eleven benchmark datasets across three time series tasks, each using both conventional TS models and TS foundation models, demonstrate that TSRating outperforms baselines in terms of estimation accuracy, efficiency, and domain adaptability. 

---
# ReFoCUS: Reinforcement-guided Frame Optimization for Contextual Understanding 

**Authors**: Hosu Lee, Junho Kim, Hyunjun Kim, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2506.01274)  

**Abstract**: Recent progress in Large Multi-modal Models (LMMs) has enabled effective vision-language reasoning, yet the ability to understand video content remains constrained by suboptimal frame selection strategies. Existing approaches often rely on static heuristics or external retrieval modules to feed frame information into video-LLMs, which may fail to provide the query-relevant information. In this work, we introduce ReFoCUS (Reinforcement-guided Frame Optimization for Contextual UnderStanding), a novel frame-level policy optimization framework that shifts the optimization target from textual responses to visual input selection. ReFoCUS learns a frame selection policy via reinforcement learning, using reward signals derived from a reference LMM to reflect the model's intrinsic preferences for frames that best support temporally grounded responses. To efficiently explore the large combinatorial frame space, we employ an autoregressive, conditional selection architecture that ensures temporal coherence while reducing complexity. Our approach does not require explicit supervision at the frame-level and consistently improves reasoning performance across multiple video QA benchmarks, highlighting the benefits of aligning frame selection with model-internal utility. 

---
# Detoxification of Large Language Models through Output-layer Fusion with a Calibration Model 

**Authors**: Yuanhe Tian, Mingjie Deng, Guoqing Jin, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.01266)  

**Abstract**: Existing approaches for Large language model (LLM) detoxification generally rely on training on large-scale non-toxic or human-annotated preference data, designing prompts to instruct the LLM to generate safe content, or modifying the model parameters to remove toxic information, which are computationally expensive, lack robustness, and often compromise LLMs' fluency and contextual understanding. In this paper, we propose a simple yet effective approach for LLM detoxification, which leverages a compact, pre-trained calibration model that guides the detoxification process of a target LLM via a lightweight intervention in its generation pipeline. By learning a detoxified embedding space from non-toxic data, the calibration model effectively steers the LLM away from generating harmful content. This approach only requires a one-time training of the calibration model that is able to be seamlessly applied to multiple LLMs without compromising fluency or contextual understanding. Experiment results on the benchmark dataset demonstrate that our approach reduces toxicity while maintaining reasonable content expression. 

---
# DeepSeek in Healthcare: A Survey of Capabilities, Risks, and Clinical Applications of Open-Source Large Language Models 

**Authors**: Jiancheng Ye, Sophie Bronstein, Jiarui Hai, Malak Abu Hashish  

**Link**: [PDF](https://arxiv.org/pdf/2506.01257)  

**Abstract**: DeepSeek-R1 is a cutting-edge open-source large language model (LLM) developed by DeepSeek, showcasing advanced reasoning capabilities through a hybrid architecture that integrates mixture of experts (MoE), chain of thought (CoT) reasoning, and reinforcement learning. Released under the permissive MIT license, DeepSeek-R1 offers a transparent and cost-effective alternative to proprietary models like GPT-4o and Claude-3 Opus; it excels in structured problem-solving domains such as mathematics, healthcare diagnostics, code generation, and pharmaceutical research. The model demonstrates competitive performance on benchmarks like the United States Medical Licensing Examination (USMLE) and American Invitational Mathematics Examination (AIME), with strong results in pediatric and ophthalmologic clinical decision support tasks. Its architecture enables efficient inference while preserving reasoning depth, making it suitable for deployment in resource-constrained settings. However, DeepSeek-R1 also exhibits increased vulnerability to bias, misinformation, adversarial manipulation, and safety failures - especially in multilingual and ethically sensitive contexts. This survey highlights the model's strengths, including interpretability, scalability, and adaptability, alongside its limitations in general language fluency and safety alignment. Future research priorities include improving bias mitigation, natural language comprehension, domain-specific validation, and regulatory compliance. Overall, DeepSeek-R1 represents a major advance in open, scalable AI, underscoring the need for collaborative governance to ensure responsible and equitable deployment. 

---
# MTCMB: A Multi-Task Benchmark Framework for Evaluating LLMs on Knowledge, Reasoning, and Safety in Traditional Chinese Medicine 

**Authors**: Shufeng Kong, Xingru Yang, Yuanyuan Wei, Zijie Wang, Hao Tang, Jiuqi Qin, Shuting Lan, Yingheng Wang, Junwen Bai, Zhuangbin Chen, Zibin Zheng, Caihua Liu, Hao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01252)  

**Abstract**: Traditional Chinese Medicine (TCM) is a holistic medical system with millennia of accumulated clinical experience, playing a vital role in global healthcare-particularly across East Asia. However, the implicit reasoning, diverse textual forms, and lack of standardization in TCM pose major challenges for computational modeling and evaluation. Large Language Models (LLMs) have demonstrated remarkable potential in processing natural language across diverse domains, including general medicine. Yet, their systematic evaluation in the TCM domain remains underdeveloped. Existing benchmarks either focus narrowly on factual question answering or lack domain-specific tasks and clinical realism. To fill this gap, we introduce MTCMB-a Multi-Task Benchmark for Evaluating LLMs on TCM Knowledge, Reasoning, and Safety. Developed in collaboration with certified TCM experts, MTCMB comprises 12 sub-datasets spanning five major categories: knowledge QA, language understanding, diagnostic reasoning, prescription generation, and safety evaluation. The benchmark integrates real-world case records, national licensing exams, and classical texts, providing an authentic and comprehensive testbed for TCM-capable models. Preliminary results indicate that current LLMs perform well on foundational knowledge but fall short in clinical reasoning, prescription planning, and safety compliance. These findings highlight the urgent need for domain-aligned benchmarks like MTCMB to guide the development of more competent and trustworthy medical AI systems. All datasets, code, and evaluation tools are publicly available at: this https URL. 

---
# Visual Sparse Steering: Improving Zero-shot Image Classification with Sparsity Guided Steering Vectors 

**Authors**: Gerasimos Chatzoudis, Zhuowei Li, Gemma E. Moran, Hao Wang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2506.01247)  

**Abstract**: Steering vision foundation models at inference time without retraining or access to large labeled datasets is a desirable yet challenging objective, particularly in dynamic or resource-constrained settings. In this paper, we introduce Visual Sparse Steering (VS2), a lightweight, test-time method that guides vision models using steering vectors derived from sparse features learned by top-$k$ Sparse Autoencoders without requiring contrastive data. Specifically, VS2 surpasses zero-shot CLIP by 4.12% on CIFAR-100, 1.08% on CUB-200, and 1.84% on Tiny-ImageNet. We further propose VS2++, a retrieval-augmented variant that selectively amplifies relevant sparse features using pseudo-labeled neighbors at inference time. With oracle positive/negative sets, VS2++ achieves absolute top-1 gains over CLIP zero-shot of up to 21.44% on CIFAR-100, 7.08% on CUB-200, and 20.47% on Tiny-ImageNet. Interestingly, VS2 and VS2++ raise per-class accuracy by up to 25% and 38%, respectively, showing that sparse steering benefits specific classes by disambiguating visually or taxonomically proximate categories rather than providing a uniform boost. Finally, to better align the sparse features learned through the SAE reconstruction task with those relevant for downstream performance, we propose Prototype-Aligned Sparse Steering (PASS). By incorporating a prototype-alignment loss during SAE training, using labels only during training while remaining fully test-time unsupervised, PASS consistently, though modestly, outperforms VS2, achieving a 6.12% gain over VS2 only on CIFAR-100 with ViT-B/32. 

---
# General search techniques without common knowledge for imperfect-information games, and application to superhuman Fog of War chess 

**Authors**: Brian Hu Zhang, Tuomas Sandholm  

**Link**: [PDF](https://arxiv.org/pdf/2506.01242)  

**Abstract**: Since the advent of AI, games have served as progress benchmarks. Meanwhile, imperfect-information variants of chess have existed for over a century, present extreme challenges, and have been the focus of significant AI research. Beyond calculation needed in regular chess, they require reasoning about information gathering, the opponent's knowledge, signaling, etc. The most popular variant, Fog of War (FoW) chess (aka. dark chess) is a recognized challenge problem in AI after superhuman performance was reached in no-limit Texas hold'em poker. We present Obscuro, the first superhuman AI for FoW chess. It introduces advances to search in imperfect-information games, enabling strong, scalable reasoning. Experiments against the prior state-of-the-art AI and human players -- including the world's best -- show that Obscuro is significantly stronger. FoW chess is the largest (by amount of imperfect information) turn-based game in which superhuman performance has been achieved and the largest game in which imperfect-information search has been successfully applied. 

---
# Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean 

**Authors**: SungHo Kim, Nayeon Kim, Taehee Jeon, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01237)  

**Abstract**: We introduce the $\underline{Ko}rean \underline{G}rammar \underline{E}valuation Bench\underline{M}ark (KoGEM)$, designed to assess the linguistic competence of LLMs and humans in Korean. KoGEM consists of 1.5k multiple-choice QA pairs covering five main categories and 16 subcategories. The zero-shot evaluation of 27 LLMs of various sizes and types reveals that while LLMs perform remarkably well on straightforward tasks requiring primarily definitional knowledge, they struggle with tasks that demand the integration of real-world experiential knowledge, such as phonological rules and pronunciation. Furthermore, our in-depth analysis suggests that incorporating such experiential knowledge could enhance the linguistic competence of LLMs. With KoGEM, we not only highlight the limitations of current LLMs in linguistic competence but also uncover hidden facets of LLMs in linguistic competence, paving the way for enhancing comprehensive language understanding. Our code and dataset are available at: this https URL. 

---
# Fourier-Modulated Implicit Neural Representation for Multispectral Satellite Image Compression 

**Authors**: Woojin Cho, Steve Andreas Immanuel, Junhyuk Heo, Darongsae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01234)  

**Abstract**: Multispectral satellite images play a vital role in agriculture, fisheries, and environmental monitoring. However, their high dimensionality, large data volumes, and diverse spatial resolutions across multiple channels pose significant challenges for data compression and analysis. This paper presents ImpliSat, a unified framework specifically designed to address these challenges through efficient compression and reconstruction of multispectral satellite data. ImpliSat leverages Implicit Neural Representations (INR) to model satellite images as continuous functions over coordinate space, capturing fine spatial details across varying spatial resolutions. Furthermore, we introduce a Fourier modulation algorithm that dynamically adjusts to the spectral and spatial characteristics of each band, ensuring optimal compression while preserving critical image details. 

---
# Retrieval-Augmented Generation of Ontologies from Relational Databases 

**Authors**: Mojtaba Nayyeri, Athish A Yogi, Nadeen Fathallah, Ratan Bahadur Thapa, Hans-Michael Tautenhahn, Anton Schnurpel, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2506.01232)  

**Abstract**: Transforming relational databases into knowledge graphs with enriched ontologies enhances semantic interoperability and unlocks advanced graph-based learning and reasoning over data. However, previous approaches either demand significant manual effort to derive an ontology from a database schema or produce only a basic ontology. We present RIGOR, Retrieval-augmented Iterative Generation of RDB Ontologies, an LLM-driven approach that turns relational schemas into rich OWL ontologies with minimal human effort. RIGOR combines three sources via RAG, the database schema and its documentation, a repository of domain ontologies, and a growing core ontology, to prompt a generative LLM for producing successive, provenance-tagged delta ontology fragments. Each fragment is refined by a judge-LLM before being merged into the core ontology, and the process iterates table-by-table following foreign key constraints until coverage is complete. Applied to real-world databases, our approach outputs ontologies that score highly on standard quality dimensions such as accuracy, completeness, conciseness, adaptability, clarity, and consistency, while substantially reducing manual effort. 

---
# Towards Efficient Few-shot Graph Neural Architecture Search via Partitioning Gradient Contribution 

**Authors**: Wenhao Song, Xuan Wu, Bo Yang, You Zhou, Yubin Xiao, Yanchun Liang, Hongwei Ge, Heow Pueh Lee, Chunguo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01231)  

**Abstract**: To address the weight coupling problem, certain studies introduced few-shot Neural Architecture Search (NAS) methods, which partition the supernet into multiple sub-supernets. However, these methods often suffer from computational inefficiency and tend to provide suboptimal partitioning schemes. To address this problem more effectively, we analyze the weight coupling problem from a novel perspective, which primarily stems from distinct modules in succeeding layers imposing conflicting gradient directions on the preceding layer modules. Based on this perspective, we propose the Gradient Contribution (GC) method that efficiently computes the cosine similarity of gradient directions among modules by decomposing the Vector-Jacobian Product during supernet backpropagation. Subsequently, the modules with conflicting gradient directions are allocated to distinct sub-supernets while similar ones are grouped together. To assess the advantages of GC and address the limitations of existing Graph Neural Architecture Search methods, which are limited to searching a single type of Graph Neural Networks (Message Passing Neural Networks (MPNNs) or Graph Transformers (GTs)), we propose the Unified Graph Neural Architecture Search (UGAS) framework, which explores optimal combinations of MPNNs and GTs. The experimental results demonstrate that GC achieves state-of-the-art (SOTA) performance in supernet partitioning quality and time efficiency. In addition, the architectures searched by UGAS+GC outperform both the manually designed GNNs and those obtained by existing NAS methods. Finally, ablation studies further demonstrate the effectiveness of all proposed methods. 

---
# SPEAR: Security Posture Evaluation using AI Planner-Reasoning on Attack-Connectivity Hypergraphs 

**Authors**: Rakesh Podder, Turgay Caglar, Shadaab Kawnain Bashir, Sarath Sreedharan, Indrajit Ray, Indrakshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2506.01227)  

**Abstract**: Graph-based frameworks are often used in network hardening to help a cyber defender understand how a network can be attacked and how the best defenses can be deployed. However, incorporating network connectivity parameters in the attack graph, reasoning about the attack graph when we do not have access to complete information, providing system administrator suggestions in an understandable format, and allowing them to do what-if analysis on various scenarios and attacker motives is still missing. We fill this gap by presenting SPEAR, a formal framework with tool support for security posture evaluation and analysis that keeps human-in-the-loop. SPEAR uses the causal formalism of AI planning to model vulnerabilities and configurations in a networked system. It automatically converts network configurations and vulnerability descriptions into planning models expressed in the Planning Domain Definition Language (PDDL). SPEAR identifies a set of diverse security hardening strategies that can be presented in a manner understandable to the domain expert. These allow the administrator to explore the network hardening solution space in a systematic fashion and help evaluate the impact and compare the different solutions. 

---
# A Review on Coarse to Fine-Grained Animal Action Recognition 

**Authors**: Ali Zia, Renuka Sharma, Abdelwahed Khamis, Xuesong Li, Muhammad Husnain, Numan Shafi, Saeed Anwar, Sabine Schmoelzl, Eric Stone, Lars Petersson, Vivien Rolland  

**Link**: [PDF](https://arxiv.org/pdf/2506.01214)  

**Abstract**: This review provides an in-depth exploration of the field of animal action recognition, focusing on coarse-grained (CG) and fine-grained (FG) techniques. The primary aim is to examine the current state of research in animal behaviour recognition and to elucidate the unique challenges associated with recognising subtle animal actions in outdoor environments. These challenges differ significantly from those encountered in human action recognition due to factors such as non-rigid body structures, frequent occlusions, and the lack of large-scale, annotated datasets. The review begins by discussing the evolution of human action recognition, a more established field, highlighting how it progressed from broad, coarse actions in controlled settings to the demand for fine-grained recognition in dynamic environments. This shift is particularly relevant for animal action recognition, where behavioural variability and environmental complexity present unique challenges that human-centric models cannot fully address. The review then underscores the critical differences between human and animal action recognition, with an emphasis on high intra-species variability, unstructured datasets, and the natural complexity of animal habitats. Techniques like spatio-temporal deep learning frameworks (e.g., SlowFast) are evaluated for their effectiveness in animal behaviour analysis, along with the limitations of existing datasets. By assessing the strengths and weaknesses of current methodologies and introducing a recently-published dataset, the review outlines future directions for advancing fine-grained action recognition, aiming to improve accuracy and generalisability in behaviour analysis across species. 

---
# Mamba Drafters for Speculative Decoding 

**Authors**: Daewon Choi, Seunghyuk Oh, Saket Dingliwal, Jihoon Tack, Kyuyoung Kim, Woomin Song, Seojin Kim, Insu Han, Jinwoo Shin, Aram Galstyan, Shubham Katiyar, Sravan Babu Bodapati  

**Link**: [PDF](https://arxiv.org/pdf/2506.01206)  

**Abstract**: Speculative decoding has emerged as a promising approach to accelerating large language model (LLM) generation using a fast drafter while maintaining alignment with the target model's distribution. However, existing approaches face a trade-off: external drafters offer flexibility but can suffer from slower drafting, while self-speculation methods use drafters tailored to the target model but require re-training. In this paper, we introduce novel drafters based on Mamba, a state-of-the-art state space model (SSM), as a solution that combines the best aspects of both approaches. By leveraging the linear structure of SSMs, our approach avoids the quadratic complexity inherent in traditional Transformer-based methods, enabling faster drafting and lower memory usage while maintaining the flexibility to work across different target models. We further enhance efficiency with a novel test-time tree search algorithm for generating high-quality draft candidates. Our empirical evaluation demonstrates that Mamba-based drafters not only outperform existing external drafting methods but are also comparable to state-of-the-art self-speculation approaches while using less memory and maintaining their cross-model adaptability. 

---
# Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures 

**Authors**: Mark Muchane, Sean Richardson, Kiho Park, Victor Veitch  

**Link**: [PDF](https://arxiv.org/pdf/2506.01197)  

**Abstract**: Sparse dictionary learning (and, in particular, sparse autoencoders) attempts to learn a set of human-understandable concepts that can explain variation on an abstract space. A basic limitation of this approach is that it neither exploits nor represents the semantic relationships between the learned concepts. In this paper, we introduce a modified SAE architecture that explicitly models a semantic hierarchy of concepts. Application of this architecture to the internal representations of large language models shows both that semantic hierarchy can be learned, and that doing so improves both reconstruction and interpretability. Additionally, the architecture leads to significant improvements in computational efficiency. 

---
# OG-VLA: 3D-Aware Vision Language Action Model via Orthographic Image Generation 

**Authors**: Ishika Singh, Ankit Goyal, Stan Birchfield, Dieter Fox, Animesh Garg, Valts Blukis  

**Link**: [PDF](https://arxiv.org/pdf/2506.01196)  

**Abstract**: We introduce OG-VLA, a novel architecture and learning framework that combines the generalization strengths of Vision Language Action models (VLAs) with the robustness of 3D-aware policies. We address the challenge of mapping natural language instructions and multi-view RGBD observations to quasi-static robot actions. 3D-aware robot policies achieve state-of-the-art performance on precise robot manipulation tasks, but struggle with generalization to unseen instructions, scenes, and objects. On the other hand, VLAs excel at generalizing across instructions and scenes, but can be sensitive to camera and robot pose variations. We leverage prior knowledge embedded in language and vision foundation models to improve generalization of 3D-aware keyframe policies. OG-VLA projects input observations from diverse views into a point cloud which is then rendered from canonical orthographic views, ensuring input view invariance and consistency between input and output spaces. These canonical views are processed with a vision backbone, a Large Language Model (LLM), and an image diffusion model to generate images that encode the next position and orientation of the end-effector on the input scene. Evaluations on the Arnold and Colosseum benchmarks demonstrate state-of-the-art generalization to unseen environments, with over 40% relative improvements while maintaining robust performance in seen settings. We also show real-world adaption in 3 to 5 demonstrations along with strong generalization. Videos and resources at this https URL 

---
# Doubly Robust Alignment for Large Language Models 

**Authors**: Erhan Xu, Kai Ye, Hongyi Zhou, Luhan Zhu, Francesco Quinzan, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01183)  

**Abstract**: This paper studies reinforcement learning from human feedback (RLHF) for aligning large language models with human preferences. While RLHF has demonstrated promising results, many algorithms are highly sensitive to misspecifications in the underlying preference model (e.g., the Bradley-Terry model), the reference policy, or the reward function, resulting in undesirable fine-tuning. To address model misspecification, we propose a doubly robust preference optimization algorithm that remains consistent when either the preference model or the reference policy is correctly specified (without requiring both). Our proposal demonstrates superior and more robust performance than state-of-the-art algorithms, both in theory and in practice. The code is available at this https URL 

---
# Humanoid World Models: Open World Foundation Models for Humanoid Robotics 

**Authors**: Muhammad Qasim Ali, Aditya Sridhar, Shahbuland Matiana, Alex Wong, Mohammad Al-Sharman  

**Link**: [PDF](https://arxiv.org/pdf/2506.01182)  

**Abstract**: Humanoid robots have the potential to perform complex tasks in human centered environments but require robust predictive models to reason about the outcomes of their actions. We introduce Humanoid World Models (HWM) a family of lightweight open source video based models that forecast future egocentric observations conditioned on actions. We train two types of generative models Masked Transformers and FlowMatching on 100 hours of humanoid demonstrations. Additionally we explore architectural variants with different attention mechanisms and parameter sharing strategies. Our parameter sharing techniques reduce model size by 33 to 53 with minimal impact on performance or visual fidelity. HWM is designed to be trained and deployed in practical academic and small lab settings such as 1 to 2 GPUs. 

---
# Bridging Quantum and Classical Computing in Drug Design: Architecture Principles for Improved Molecule Generation 

**Authors**: Andrew Smith, Erhan Guven  

**Link**: [PDF](https://arxiv.org/pdf/2506.01177)  

**Abstract**: Hybrid quantum-classical machine learning offers a path to leverage noisy intermediate-scale quantum (NISQ) devices for drug discovery, but optimal model architectures remain unclear. We systematically optimize the quantum-classical bridge architecture for generative adversarial networks (GANs) in molecular discovery using multi-objective Bayesian optimization. Our optimized model (BO-QGAN) significantly improves performance, achieving a 2.27-fold higher Drug Candidate Score (DCS) than prior quantum-hybrid benchmarks and 2.21-fold higher than the classical baseline, using over 60% fewer parameters. Key findings favor layering multiple (3-4) shallow (4-8 qubit) quantum circuits sequentially, while classical architecture shows less sensitivity above a minimum capacity. This work provides the first empirically grounded architectural guidelines for hybrid models, enabling more effective integration of current quantum computers into pharmaceutical research pipelines. 

---
# VUSA: Virtually Upscaled Systolic Array Architecture to Exploit Unstructured Sparsity in AI Acceleration 

**Authors**: Shereef Helal, Alberto Garcia-Ortiz, Lennart Bamberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01166)  

**Abstract**: Leveraging high degrees of unstructured sparsity is a promising approach to enhance the efficiency of deep neural network DNN accelerators - particularly important for emerging Edge-AI applications. We introduce VUSA, a systolic-array architecture that virtually grows based on the present sparsity to perform larger matrix multiplications with the same number of physical multiply-accumulate MAC units. The proposed architecture achieves saving by 37% and 68% in area and power efficiency, respectively, at the same peak-performance, compared to a baseline systolic array architecture in a commercial 16-nm technology. Still, the proposed architecture supports acceleration for any DNN with any sparsity - even no sparsity at all. Thus, the proposed architecture is application-independent, making it viable for general-purpose AI acceleration. 

---
# FORT: Forward-Only Regression Training of Normalizing Flows 

**Authors**: Danyal Rehman, Oscar Davis, Jiarui Lu, Jian Tang, Michael Bronstein, Yoshua Bengio, Alexander Tong, Avishek Joey Bose  

**Link**: [PDF](https://arxiv.org/pdf/2506.01158)  

**Abstract**: Simulation-free training frameworks have been at the forefront of the generative modelling revolution in continuous spaces, leading to neural dynamical systems that encompass modern large-scale diffusion and flow matching models. Despite the scalability of training, the generation of high-quality samples and their corresponding likelihood under the model requires expensive numerical simulation -- inhibiting adoption in numerous scientific applications such as equilibrium sampling of molecular systems. In this paper, we revisit classical normalizing flows as one-step generative models with exact likelihoods and propose a novel, scalable training objective that does not require computing the expensive change of variable formula used in conventional maximum likelihood training. We propose Forward-Only Regression Training (FORT), a simple $\ell_2$-regression objective that maps prior samples under our flow to specifically chosen targets. We demonstrate that FORT supports a wide class of targets, such as optimal transport targets and targets from pre-trained continuous-time normalizing flows (CNF). We further demonstrate that by using CNF targets, our one-step flows allow for larger-scale training that exceeds the performance and stability of maximum likelihood training, while unlocking a broader class of architectures that were previously challenging to train. Empirically, we elucidate that our trained flows can perform equilibrium conformation sampling in Cartesian coordinates of alanine dipeptide, alanine tripeptide, and alanine tetrapeptide. 

---
# Earley-Driven Dynamic Pruning for Efficient Structured Decoding 

**Authors**: Xintong Sun, Chi Wei, Minghao Tian, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.01151)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities, yet ensuring their outputs conform to strict structural or grammatical constraints remains challenging, which is critical in function calls and domain-specific language (DSL) generation. Constrained decoding with context-free grammar is a flexible approach to guarantee LLMs' adherence to a specific format by dynamically building a token logits mask. However, creating this mask requires checking the validity of all tokens in the LLM vocabulary at every decoding step, which often incurs significant overheads in existing constrained decoding engines. To address this challenge, we propose $\textbf{ZapFormat}$, a novel $\textbf{dynamic pruning}$ strategy based on the Earley algorithm that identifies and eliminates invalid or redundant Earley states in real-time, significantly reducing memory occupation of the Earley algorithm's states. This further enables us to use a state cache to speed up structured generations on a large number of queries. We implemented ZapFormat in a new constrained decoding engine called Formatron which also incorporates existing optimizations. Through comprehensive experiments on structured generation tasks, including JSON generation, JSON Schema validation, and semantic parsing, we demonstrate that Formatron not only $\textbf{consistently maintains}$ high-precision compliant outputs but also achieves $\textbf{significant improvements}$ in inference speed up to 2x compared to state-of-the-art implementations. More importantly, Formatron is generally applicable across various LLM architectures. We release Formatron as open source at this https URL. 

---
# From Words to Waves: Analyzing Concept Formation in Speech and Text-Based Foundation Models 

**Authors**: Asım Ersoy, Basel Mousi, Shammur Chowdhury, Firoj Alam, Fahim Dalvi, Nadir Durrani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01133)  

**Abstract**: The emergence of large language models (LLMs) has demonstrated that systems trained solely on text can acquire extensive world knowledge, develop reasoning capabilities, and internalize abstract semantic concepts--showcasing properties that can be associated with general intelligence. This raises an intriguing question: Do such concepts emerge in models trained on other modalities, such as speech? Furthermore, when models are trained jointly on multiple modalities: Do they develop a richer, more structured semantic understanding? To explore this, we analyze the conceptual structures learned by speech and textual models both individually and jointly. We employ Latent Concept Analysis, an unsupervised method for uncovering and interpreting latent representations in neural networks, to examine how semantic abstractions form across modalities. For reproducibility we made scripts and other resources available to the community. 

---
# Neuro-Symbolic Generative Diffusion Models for Physically Grounded, Robust, and Safe Generation 

**Authors**: Jacob K. Christopher, Michael Cardei, Jinhao Liang, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2506.01121)  

**Abstract**: Despite the remarkable generative capabilities of diffusion models, their integration into safety-critical or scientifically rigorous applications remains hindered by the need to ensure compliance with stringent physical, structural, and operational constraints. To address this challenge, this paper introduces Neuro-Symbolic Diffusion (NSD), a novel framework that interleaves diffusion steps with symbolic optimization, enabling the generation of certifiably consistent samples under user-defined functional and logic constraints. This key feature is provided for both standard and discrete diffusion models, enabling, for the first time, the generation of both continuous (e.g., images and trajectories) and discrete (e.g., molecular structures and natural language) outputs that comply with constraints. This ability is demonstrated on tasks spanning three key challenges: (1) Safety, in the context of non-toxic molecular generation and collision-free trajectory optimization; (2) Data scarcity, in domains such as drug discovery and materials engineering; and (3) Out-of-domain generalization, where enforcing symbolic constraints allows adaptation beyond the training distribution. 

---
# Reconsidering LLM Uncertainty Estimation Methods in the Wild 

**Authors**: Yavuz Bakman, Duygu Nur Yaldiz, Sungmin Kang, Tuo Zhang, Baturalp Buyukates, Salman Avestimehr, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.01114)  

**Abstract**: Large Language Model (LLM) Uncertainty Estimation (UE) methods have become a crucial tool for detecting hallucinations in recent years. While numerous UE methods have been proposed, most existing studies evaluate them in isolated short-form QA settings using threshold-independent metrics such as AUROC or PRR. However, real-world deployment of UE methods introduces several challenges. In this work, we systematically examine four key aspects of deploying UE methods in practical settings. Specifically, we assess (1) the sensitivity of UE methods to decision threshold selection, (2) their robustness to query transformations such as typos, adversarial prompts, and prior chat history, (3) their applicability to long-form generation, and (4) strategies for handling multiple UE scores for a single query. Our evaluations on 19 UE methods reveal that most of them are highly sensitive to threshold selection when there is a distribution shift in the calibration dataset. While these methods generally exhibit robustness against previous chat history and typos, they are significantly vulnerable to adversarial prompts. Additionally, while existing UE methods can be adapted for long-form generation through various strategies, there remains considerable room for improvement. Lastly, ensembling multiple UE scores at test time provides a notable performance boost, which highlights its potential as a practical improvement strategy. Code is available at: this https URL. 

---
# FusionAudio-1.2M: Towards Fine-grained Audio Captioning with Multimodal Contextual Fusion 

**Authors**: Shunian Chen, Xinyuan Xie, Zheshu Chen, Liyan Zhao, Owen Lee, Zhan Su, Qilin Sun, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01111)  

**Abstract**: High-quality, large-scale audio captioning is crucial for advancing audio understanding, yet current automated methods often generate captions that lack fine-grained detail and contextual accuracy, primarily due to their reliance on limited unimodal or superficial multimodal information. Drawing inspiration from human auditory perception, which adeptly integrates cross-modal cues and performs sophisticated auditory scene analysis, we introduce a novel two-stage automated pipeline. This pipeline first employs specialized pretrained models to extract diverse contextual cues (e.g., speech, music, general sounds, and visual information from associated video). A large language model (LLM) then synthesizes these rich, multimodal inputs to generate detailed and context-aware audio captions. Key contributions of this work include: (1) the proposed scalable method for fine-grained audio caption generation; (2) FusionAudio, a new large-scale dataset comprising 1.2 million such detailed captions, combined with 6 million QA pairs; and (3) enhanced audio models developed using FusionAudio, specifically a CLAP-based audio encoder with superior audio-text alignment and instruction following. This paper paves the way for more nuanced and accurate automated understanding of complex audio environments. Code and data can be found in this https URL. 

---
# CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting 

**Authors**: Fengze Li, Yangle Liu, Jieming Ma, Hai-Ning Liang, Yaochun Shen, Huangxiang Li, Zhijing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01109)  

**Abstract**: Accurate fruit counting in real-world agricultural environments is a longstanding challenge due to visual occlusions, semantic ambiguity, and the high computational demands of 3D reconstruction. Existing methods based on neural radiance fields suffer from low inference speed, limited generalization, and lack support for open-set semantic control. This paper presents FruitLangGS, a real-time 3D fruit counting framework that addresses these limitations through spatial reconstruction, semantic embedding, and language-guided instance estimation. FruitLangGS first reconstructs orchard-scale scenes using an adaptive Gaussian splatting pipeline with radius-aware pruning and tile-based rasterization for efficient rendering. To enable semantic control, each Gaussian encodes a compressed CLIP-aligned language embedding, forming a compact and queryable 3D representation. At inference time, prompt-based semantic filtering is applied directly in 3D space, without relying on image-space segmentation or view-level fusion. The selected Gaussians are then converted into dense point clouds via distribution-aware sampling and clustered to estimate fruit counts. Experimental results on real orchard data demonstrate that FruitLangGS achieves higher rendering speed, semantic flexibility, and counting accuracy compared to prior approaches, offering a new perspective for language-driven, real-time neural rendering across open-world scenarios. 

---
# Speeding Up Hyper-Heuristics With Markov-Chain Operator Selection and the Only-Worsening Acceptance Operator 

**Authors**: Abderrahim Bendahi, Benjamin Doerr, Adrien Fradin, Johannes F. Lutzeyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.01107)  

**Abstract**: The move-acceptance hyper-heuristic was recently shown to be able to leave local optima with astonishing efficiency (Lissovoi et al., Artificial Intelligence (2023)). In this work, we propose two modifications to this algorithm that demonstrate impressive performances on a large class of benchmarks including the classic Cliff$_d$ and Jump$_m$ function classes. (i) Instead of randomly choosing between the only-improving and any-move acceptance operator, we take this choice via a simple two-state Markov chain. This modification alone reduces the runtime on Jump$_m$ functions with gap parameter $m$ from $\Omega(n^{2m-1})$ to $O(n^{m+1})$. (ii) We then replace the all-moves acceptance operator with the operator that only accepts worsenings. Such a, counter-intuitive, operator has not been used before in the literature. However, our proofs show that our only-worsening operator can greatly help in leaving local optima, reducing, e.g., the runtime on Jump functions to $O(n^3 \log n)$ independent of the gap size. In general, we prove a remarkably good runtime of $O(n^{k+1} \log n)$ for our Markov move-acceptance hyper-heuristic on all members of a new benchmark class SEQOPT$_k$, which contains a large number of functions having $k$ successive local optima, and which contains the commonly studied Jump$_m$ and Cliff$_d$ functions for $k=2$. 

---
# Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements 

**Authors**: Metehan Oguz, Yavuz Bakman, Duygu Nur Yaldiz  

**Link**: [PDF](https://arxiv.org/pdf/2506.01089)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performances in tasks related to coreference resolution. However, previous studies mostly assessed LLM performance on coreference resolution with nouns and third person pronouns. This study evaluates LLM performance on coreference resolution with indexical like I, you, here and tomorrow, which come with unique challenges due to their linguistic properties. We present the first study examining how LLMs interpret indexicals in English, releasing the English Indexical Dataset with 1600 multiple-choice questions. We evaluate pioneering LLMs, including GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and DeepSeek V3. Our results reveal that LLMs exhibit an impressive performance with some indexicals (I), while struggling with others (you, here, tomorrow), and that syntactic cues (e.g. quotation) contribute to LLM performance with some indexicals, while they reduce performance with others. Code and data are available at: this https URL. 

---
# Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection 

**Authors**: Shivam Chandhok, Qian Yang, Oscar Manas, Kanishk Jain, Leonid Sigal, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.01085)  

**Abstract**: Instruction tuning has been central to the success of recent vision-language models (VLMs), but it remains expensive-requiring large-scale datasets, high-quality annotations, and large compute budgets. We propose PRioritized cOncept learninG via Relative Error-driven Sample Selection (PROGRESS), a data- and compute-efficient framework that enables VLMs to dynamically select what to learn next based on their evolving needs during training. At each stage, the model tracks its learning progress across skills and selects the most informative samples-those it has not already mastered and that are not too difficult to learn at the current stage of training. This strategy effectively controls skill acquisition and the order in which skills are learned. Specifically, we sample from skills showing the highest learning progress, prioritizing those with the most rapid improvement. Unlike prior methods, PROGRESS requires no upfront answer annotations, queries answers only on a need basis, avoids reliance on additional supervision from auxiliary VLMs, and does not require compute-heavy gradient computations for data selection. Experiments across multiple instruction-tuning datasets of varying scales demonstrate that PROGRESS consistently outperforms state-of-the-art baselines with much less data and supervision. Additionally, we show strong cross-architecture generalization and transferability to larger models, validating PROGRESS as a scalable solution for efficient learning. 

---
# Unfolding Boxes with Local Constraints 

**Authors**: Long Qian, Eric Wang, Bernardo Subercaseaux, Marijn J. H. Heule  

**Link**: [PDF](https://arxiv.org/pdf/2506.01079)  

**Abstract**: We consider the problem of finding and enumerating polyominos that can be folded into multiple non-isomorphic boxes. While several computational approaches have been proposed, including SAT, randomized algorithms, and decision diagrams, none has been able to perform at scale. We argue that existing SAT encodings are hindered by the presence of global constraints (e.g., graph connectivity or acyclicity), which are generally hard to encode effectively and hard for solvers to reason about. In this work, we propose a new SAT-based approach that replaces these global constraints with simple local constraints that have substantially better propagation properties. Our approach dramatically improves the scalability of both computing and enumerating common box unfoldings: (i) while previous approaches could only find common unfoldings of two boxes up to area 88, ours easily scales beyond 150, and (ii) while previous approaches were only able to enumerate common unfoldings up to area 30, ours scales up to 60. This allows us to rule out 46, 54, and 58 as the smallest areas allowing a common unfolding of three boxes, thereby refuting a conjecture of Xu et al. (2017). 

---
# GThinker: Towards General Multimodal Reasoning via Cue-Guided Rethinking 

**Authors**: Yufei Zhan, Ziheng Wu, Yousong Zhu, Rongkun Xue, Ruipu Luo, Zhenghao Chen, Can Zhang, Yifan Li, Zhentao He, Zheming Yang, Ming Tang, Minghui Qiu, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01078)  

**Abstract**: Despite notable advancements in multimodal reasoning, leading Multimodal Large Language Models (MLLMs) still underperform on vision-centric multimodal reasoning tasks in general scenarios. This shortfall stems from their predominant reliance on logic- and knowledge-based slow thinking strategies, while effective for domains like math and science, fail to integrate visual information effectively during reasoning. Consequently, these models often fail to adequately ground visual cues, resulting in suboptimal performance in tasks that require multiple plausible visual interpretations and inferences. To address this, we present GThinker (General Thinker), a novel reasoning MLLM excelling in multimodal reasoning across general scenarios, mathematics, and science. GThinker introduces Cue-Rethinking, a flexible reasoning pattern that grounds inferences in visual cues and iteratively reinterprets these cues to resolve inconsistencies. Building on this pattern, we further propose a two-stage training pipeline, including pattern-guided cold start and incentive reinforcement learning, designed to enable multimodal reasoning capabilities across domains. Furthermore, to support the training, we construct GThinker-11K, comprising 7K high-quality, iteratively-annotated reasoning paths and 4K curated reinforcement learning samples, filling the data gap toward general multimodal reasoning. Extensive experiments demonstrate that GThinker achieves 81.5% on the challenging comprehensive multimodal reasoning benchmark M$^3$CoT, surpassing the latest O4-mini model. It also shows an average improvement of 2.1% on general scenario multimodal reasoning benchmarks, while maintaining on-par performance in mathematical reasoning compared to counterpart advanced reasoning models. The code, model, and data will be released soon at this https URL. 

---
# Revolutionizing Blood Banks: AI-Driven Fingerprint-Blood Group Correlation for Enhanced Safety 

**Authors**: Malik A. Altayar, Muhyeeddin Alqaraleh, Mowafaq Salem Alzboon, Wesam T. Almagharbeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.01069)  

**Abstract**: Identification of a person is central in forensic science, security, and healthcare. Methods such as iris scanning and genomic profiling are more accurate but expensive, time-consuming, and more difficult to implement. This study focuses on the relationship between the fingerprint patterns and the ABO blood group as a biometric identification tool. A total of 200 subjects were included in the study, and fingerprint types (loops, whorls, and arches) and blood groups were compared. Associations were evaluated with statistical tests, including chi-square and Pearson correlation. The study found that the loops were the most common fingerprint pattern and the O+ blood group was the most prevalent. Even though there was some associative pattern, there was no statistically significant difference in the fingerprint patterns of different blood groups. Overall, the results indicate that blood group data do not significantly improve personal identification when used in conjunction with fingerprinting. Although the study shows weak correlation, it may emphasize the efforts of multi-modal based biometric systems in enhancing the current biometric systems. Future studies may focus on larger and more diverse samples, and possibly machine learning and additional biometrics to improve identification methods. This study addresses an element of the ever-changing nature of the fields of forensic science and biometric identification, highlighting the importance of resilient analytical methods for personal identification. 

---
# Trilevel Memetic Algorithm for the Electric Vehicle Routing Problem 

**Authors**: Ivan Milinović, Leon Stjepan Uroić, Marko Đurasević  

**Link**: [PDF](https://arxiv.org/pdf/2506.01065)  

**Abstract**: The Electric Vehicle Routing Problem (EVRP) extends the capacitated vehicle routing problem by incorporating battery constraints and charging stations, posing significant optimization challenges. This paper introduces a Trilevel Memetic Algorithm (TMA) that hierarchically optimizes customer sequences, route assignments, and charging station insertions. The method combines genetic algorithms with dynamic programming, ensuring efficient and high-quality solutions. Benchmark tests on WCCI2020 instances show competitive performance, matching best-known results for small-scale cases. While computational demands limit scalability, TMA demonstrates strong potential for sustainable logistics planning. 

---
# Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs 

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01064)  

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. Despite their potential impact, the development of effective methods for purifying such adversarial examples has received relatively limited attention. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive "fighting fire with fire" strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code will be made publicly available. 

---
# SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models 

**Authors**: Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01062)  

**Abstract**: We introduce SealQA, a new challenge benchmark for evaluating SEarch-Augmented Language models on fact-seeking questions where web search yields conflicting, noisy, or unhelpful results. SealQA comes in three flavors: (1) Seal-0 (main) and (2) Seal-Hard, which assess factual accuracy and reasoning capabilities, with Seal-0 focusing on the most challenging questions where chat models (e.g., GPT-4.1) typically achieve near-zero accuracy; and (3) LongSeal, which extends SealQA to test long-context, multi-document reasoning in "needle-in-a-haystack" settings. Our evaluation reveals critical limitations in current models: Even frontier LLMs perform poorly across all SealQA flavors. On Seal-0, frontier agentic models equipped with tools like o3 and o4-mini achieve only 17.1% and 6.3% accuracy, respectively, at their best reasoning efforts. We find that advanced reasoning models such as DeepSeek-R1-671B and o3-mini are highly vulnerable to noisy search results. Notably, increasing test-time compute does not yield reliable gains across o3-mini, o4-mini, and o3, with performance often plateauing or even declining early. Additionally, while recent models are less affected by the "lost-in-the-middle" issue, they still fail to reliably identify relevant documents in LongSeal when faced with numerous distractors. To facilitate future work, we release SealQA at this http URL. 

---
# XAI-Units: Benchmarking Explainability Methods with Unit Tests 

**Authors**: Jun Rui Lee, Sadegh Emami, Michael David Hollins, Timothy C. H. Wong, Carlos Ignacio Villalobos Sánchez, Francesca Toni, Dekai Zhang, Adam Dejl  

**Link**: [PDF](https://arxiv.org/pdf/2506.01059)  

**Abstract**: Feature attribution (FA) methods are widely used in explainable AI (XAI) to help users understand how the inputs of a machine learning model contribute to its outputs. However, different FA models often provide disagreeing importance scores for the same model. In the absence of ground truth or in-depth knowledge about the inner workings of the model, it is often difficult to meaningfully determine which of the different FA methods produce more suitable explanations in different contexts. As a step towards addressing this issue, we introduce the open-source XAI-Units benchmark, specifically designed to evaluate FA methods against diverse types of model behaviours, such as feature interactions, cancellations, and discontinuous outputs. Our benchmark provides a set of paired datasets and models with known internal mechanisms, establishing clear expectations for desirable attribution scores. Accompanied by a suite of built-in evaluation metrics, XAI-Units streamlines systematic experimentation and reveals how FA methods perform against distinct, atomic kinds of model reasoning, similar to unit tests in software engineering. Crucially, by using procedurally generated models tied to synthetic datasets, we pave the way towards an objective and reliable comparison of FA methods. 

---
# Taming LLMs by Scaling Learning Rates with Gradient Grouping 

**Authors**: Siyuan Li, Juanxi Tian, Zedong Wang, Xin Jin, Zicheng Liu, Wentao Zhang, Dan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01049)  

**Abstract**: Training large language models (LLMs) poses challenges due to their massive scale and heterogeneous architectures. While adaptive optimizers like AdamW help address gradient variations, they still struggle with efficient and effective parameter-wise learning rate estimation, resulting in training instability, slow convergence, and poor compatibility with parameter-efficient fine-tuning (PEFT) techniques. This work introduces Scaling with Gradient Grouping (SGG), an optimizer wrapper that improves adaptive learning rate estimation by dynamic grouping and group-specific scaling. SGG first groups gradient statistics in each layer into clusters and then applies cluster-specific scaling to calibrate learning rates for each parameter, thus imposing collective group-wise constraints while maintaining precise per-parameter adaptation. Experiments on diverse (M)LLM benchmarks show that SGG integrates seamlessly with existing optimizers, and offers consistent gains and faster convergence over baselines, with various model sizes. Its stability across varying batch sizes and learning rates establishes SGG as a robust choice for LLM optimization. 

---
# Probing Neural Topology of Large Language Models 

**Authors**: Yu Zheng, Yuan Yuan, Yong Li, Paolo Santi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01042)  

**Abstract**: Probing large language models (LLMs) has yielded valuable insights into their internal mechanisms by linking neural representations to interpretable semantics. However, how neurons functionally co-activate with each other to give rise to emergent capabilities remains largely unknown, hindering a deeper understanding and safer development of LLMs. In this work, we introduce graph probing, a method for uncovering the functional connectivity topology of LLM neurons and relating it to language generation performance. By analyzing internal neural graphs across diverse LLM families and scales, we discover a universal predictability of next-token prediction performance using only neural topology. This predictability is robust even when retaining just 1% of neuron connections or probing models after only 8 pretraining steps, highlighting the sparsity and early emergence of topological patterns. Further graph matching analysis suggests that, despite significant distinctions in architectures, parameters, and training data, different LLMs develop intricate and consistent neural topological structures that may form the foundation for their language generation abilities. Codes and data for the graph probing toolbox are released at this https URL. 

---
# Less is More: Local Intrinsic Dimensions of Contextual Language Models 

**Authors**: Benjamin Matthias Ruppik, Julius von Rohrscheidt, Carel van Niekerk, Michael Heck, Renato Vukovic, Shutong Feng, Hsien-chin Lin, Nurul Lubis, Bastian Rieck, Marcus Zibrowius, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2506.01034)  

**Abstract**: Understanding the internal mechanisms of large language models (LLMs) remains a challenging and complex endeavor. Even fundamental questions, such as how fine-tuning affects model behavior, often require extensive empirical evaluation. In this paper, we introduce a novel perspective based on the geometric properties of contextual latent embeddings to study the effects of training and fine-tuning. To that end, we measure the local dimensions of a contextual language model's latent space and analyze their shifts during training and fine-tuning. We show that the local dimensions provide insights into the model's training dynamics and generalization ability. Specifically, the mean of the local dimensions predicts when the model's training capabilities are exhausted, as exemplified in a dialogue state tracking task, overfitting, as demonstrated in an emotion recognition task, and grokking, as illustrated with an arithmetic task. Furthermore, our experiments suggest a practical heuristic: reductions in the mean local dimension tend to accompany and predict subsequent performance gains. Through this exploration, we aim to provide practitioners with a deeper understanding of the implications of fine-tuning on embedding spaces, facilitating informed decisions when configuring models for specific applications. The results of this work contribute to the ongoing discourse on the interpretability, adaptability, and generalizability of LLMs by bridging the gap between intrinsic model mechanisms and geometric properties in the respective embeddings. 

---
# A Two-Stage Hierarchical Deep Filtering Framework for Real-Time Speech Enhancement 

**Authors**: Shenghui Lu, Hukai Huang, Jinanglong Yao, Kaidi Wang, Qingyang Hong, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01023)  

**Abstract**: This paper proposes a model that integrates sub-band processing and deep filtering to fully exploit information from the target time-frequency (TF) bin and its surrounding TF bins for single-channel speech enhancement. The sub-band module captures surrounding frequency bin information at the input, while the deep filtering module applies filtering at the output to both the target TF bin and its surrounding TF bins. To further improve the model performance, we decouple deep filtering into temporal and frequency components and introduce a two-stage framework, reducing the complexity of filter coefficient prediction at each stage. Additionally, we propose the TAConv module to strengthen convolutional feature extraction. Experimental results demonstrate that the proposed hierarchical deep filtering network (HDF-Net) effectively utilizes surrounding TF bin information and outperforms other advanced systems while using fewer resources. 

---
# Motion-Aware Concept Alignment for Consistent Video Editing 

**Authors**: Tong Zhang, Juan C Leon Alcazar, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.01004)  

**Abstract**: We introduce MoCA-Video (Motion-Aware Concept Alignment in Video), a training-free framework bridging the gap between image-domain semantic mixing and video. Given a generated video and a user-provided reference image, MoCA-Video injects the semantic features of the reference image into a specific object within the video, while preserving the original motion and visual context. Our approach leverages a diagonal denoising schedule and class-agnostic segmentation to detect and track objects in the latent space and precisely control the spatial location of the blended objects. To ensure temporal coherence, we incorporate momentum-based semantic corrections and gamma residual noise stabilization for smooth frame transitions. We evaluate MoCA's performance using the standard SSIM, image-level LPIPS, temporal LPIPS, and introduce a novel metric CASS (Conceptual Alignment Shift Score) to evaluate the consistency and effectiveness of the visual shifts between the source prompt and the modified video frames. Using self-constructed dataset, MoCA-Video outperforms current baselines, achieving superior spatial consistency, coherent motion, and a significantly higher CASS score, despite having no training or fine-tuning. MoCA-Video demonstrates that structured manipulation in the diffusion noise trajectory allows for controllable, high-quality video synthesis. 

---
# Quotient Network - A Network Similar to ResNet but Learning Quotients 

**Authors**: Peng Hui, Jiamuyang Zhao, Changxin Li, Qingzhen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00992)  

**Abstract**: The emergence of ResNet provides a powerful tool for training extremely deep networks. The core idea behind it is to change the learning goals of the network. It no longer learns new features from scratch but learns the difference between the target and existing features. However, the difference between the two kinds of features does not have an independent and clear meaning, and the amount of learning is based on the absolute rather than the relative difference, which is sensitive to the size of existing features. We propose a new network that perfectly solves these two problems while still having the advantages of ResNet. Specifically, it chooses to learn the quotient of the target features with the existing features, so we call it the quotient network. In order to enable this network to learn successfully and achieve higher performance, we propose some design rules for this network so that it can be trained efficiently and achieve better performance than ResNet. Experiments on the CIFAR10, CIFAR100, and SVHN datasets prove that this network can stably achieve considerable improvements over ResNet by simply making tiny corresponding changes to the original ResNet network without adding new parameters. 

---
# Bridging the Gap: From Ad-hoc to Proactive Search in Conversations 

**Authors**: Chuan Meng, Francesco Tonolini, Fengran Mo, Nikolaos Aletras, Emine Yilmaz, Gabriella Kazai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00983)  

**Abstract**: Proactive search in conversations (PSC) aims to reduce user effort in formulating explicit queries by proactively retrieving useful relevant information given conversational context. Previous work in PSC either directly uses this context as input to off-the-shelf ad-hoc retrievers or further fine-tunes them on PSC data. However, ad-hoc retrievers are pre-trained on short and concise queries, while the PSC input is longer and noisier. This input mismatch between ad-hoc search and PSC limits retrieval quality. While fine-tuning on PSC data helps, its benefits remain constrained by this input gap. In this work, we propose Conv2Query, a novel conversation-to-query framework that adapts ad-hoc retrievers to PSC by bridging the input gap between ad-hoc search and PSC. Conv2Query maps conversational context into ad-hoc queries, which can either be used as input for off-the-shelf ad-hoc retrievers or for further fine-tuning on PSC data. Extensive experiments on two PSC datasets show that Conv2Query significantly improves ad-hoc retrievers' performance, both when used directly and after fine-tuning on PSC. 

---
# What do self-supervised speech models know about Dutch? Analyzing advantages of language-specific pre-training 

**Authors**: Marianne de Heer Kloots, Hosein Mohebbi, Charlotte Pouw, Gaofei Shen, Willem Zuidema, Martijn Bentum  

**Link**: [PDF](https://arxiv.org/pdf/2506.00981)  

**Abstract**: How language-specific are speech representations learned by self-supervised models? Existing work has shown that a range of linguistic features can be successfully decoded from end-to-end models trained only on speech recordings. However, it's less clear to what extent pre-training on specific languages improves language-specific linguistic information. Here we test the encoding of Dutch phonetic and lexical information in internal representations of self-supervised Wav2Vec2 models. Pre-training exclusively on Dutch improves the representation of Dutch linguistic features as compared to pre-training on similar amounts of English or larger amounts of multilingual data. This language-specific advantage is well-detected by trained clustering or classification probes, and partially observable using zero-shot metrics. Furthermore, the language-specific benefit on linguistic feature encoding aligns with downstream performance on Automatic Speech Recognition. 

---
# IVY-FAKE: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection 

**Authors**: Wayne Zhang, Changjiang Jiang, Zhonghao Zhang, Chenyang Si, Fengchang Yu, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00979)  

**Abstract**: The rapid advancement of Artificial Intelligence Generated Content (AIGC) in visual domains has resulted in highly realistic synthetic images and videos, driven by sophisticated generative frameworks such as diffusion-based architectures. While these breakthroughs open substantial opportunities, they simultaneously raise critical concerns about content authenticity and integrity. Many current AIGC detection methods operate as black-box binary classifiers, which offer limited interpretability, and no approach supports detecting both images and videos in a unified framework. This dual limitation compromises model transparency, reduces trustworthiness, and hinders practical deployment. To address these challenges, we introduce IVY-FAKE , a novel, unified, and large-scale dataset specifically designed for explainable multimodal AIGC detection. Unlike prior benchmarks, which suffer from fragmented modality coverage and sparse annotations, IVY-FAKE contains over 150,000 richly annotated training samples (images and videos) and 18,700 evaluation examples, each accompanied by detailed natural-language reasoning beyond simple binary labels. Building on this, we propose Ivy Explainable Detector (IVY-XDETECTOR), a unified AIGC detection and explainable architecture that jointly performs explainable detection for both image and video content. Our unified vision-language model achieves state-of-the-art performance across multiple image and video detection benchmarks, highlighting the significant advancements enabled by our dataset and modeling framework. Our data is publicly available at this https URL. 

---
# NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction 

**Authors**: Qichao Wang, Ziqiao Meng, Wenqian Cui, Yifei Zhang, Pengcheng Wu, Bingzhe Wu, Irwin King, Liang Chen, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00975)  

**Abstract**: Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications. 

---
# Data Heterogeneity Modeling for Trustworthy Machine Learning 

**Authors**: Jiashuo Liu, Peng Cui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00969)  

**Abstract**: Data heterogeneity plays a pivotal role in determining the performance of machine learning (ML) systems. Traditional algorithms, which are typically designed to optimize average performance, often overlook the intrinsic diversity within datasets. This oversight can lead to a myriad of issues, including unreliable decision-making, inadequate generalization across different domains, unfair outcomes, and false scientific inferences. Hence, a nuanced approach to modeling data heterogeneity is essential for the development of dependable, data-driven systems. In this survey paper, we present a thorough exploration of heterogeneity-aware machine learning, a paradigm that systematically integrates considerations of data heterogeneity throughout the entire ML pipeline -- from data collection and model training to model evaluation and deployment. By applying this approach to a variety of critical fields, including healthcare, agriculture, finance, and recommendation systems, we demonstrate the substantial benefits and potential of heterogeneity-aware ML. These applications underscore how a deeper understanding of data diversity can enhance model robustness, fairness, and reliability and help model diagnosis and improvements. Moreover, we delve into future directions and provide research opportunities for the whole data mining community, aiming to promote the development of heterogeneity-aware ML. 

---
# Legal Compliance Evaluation of Smart Contracts Generated By Large Language Models 

**Authors**: Chanuka Wijayakoon, Hai Dong, H.M.N. Dilum Bandara, Zahir Tari, Anurag Soin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00943)  

**Abstract**: Smart contracts can implement and automate parts of legal contracts, but ensuring their legal compliance remains challenging. Existing approaches such as formal specification, verification, and model-based development require expertise in both legal and software development domains, as well as extensive manual effort. Given the recent advances of Large Language Models (LLMs) in code generation, we investigate their ability to generate legally compliant smart contracts directly from natural language legal contracts, addressing these challenges. We propose a novel suite of metrics to quantify legal compliance based on modeling both legal and smart contracts as processes and comparing their behaviors. We select four LLMs, generate 20 smart contracts based on five legal contracts, and analyze their legal compliance. We find that while all LLMs generate syntactically correct code, there is significant variance in their legal compliance with larger models generally showing higher levels of compliance. We also evaluate the proposed metrics against properties of software metrics, showing they provide fine-grained distinctions, enable nuanced comparisons, and are applicable across domains for code from any source, LLM or developer. Our results suggest that LLMs can assist in generating starter code for legally compliant smart contracts with strict reviews, and the proposed metrics provide a foundation for automated and self-improving development workflows. 

---
# anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding 

**Authors**: Haitao Li, Ziyu Li, Yiheng Mao, Ziyi Liu, Zhoujian Sun, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00942)  

**Abstract**: The advent of multimodal large language models (MLLMs) has sparked interest in their application to electrocardiogram (ECG) analysis. However, existing ECG-focused MLLMs primarily focus on report generation tasks, often limited to single 12-lead, short-duration (10s) ECG inputs, thereby underutilizing the potential of MLLMs. To this end, we aim to develop a MLLM for ECG analysis that supports a broader range of tasks and more flexible ECG inputs. However, existing ECG-QA datasets are often monotonous. To address this gap, we first constructed the anyECG dataset, which encompasses a wide variety of tasks, including report generation, abnormal waveform localization, and open-ended question answering. In addition to standard hospital ECGs, we introduced long-duration reduced-lead ECGs for home environments and multiple ECG comparison scenarios commonly encountered in clinical practice. Furthermore, we propose the anyECG-chat model, which supports dynamic-length ECG inputs and multiple ECG inputs. We trained the model using a three-stage curriculum training recipe with the anyECG dataset. A comprehensive evaluation was conducted, demonstrating that anyECG-chat is capable of supporting various practical application scenarios, including not only common report generation tasks but also abnormal waveform localization for long-duration reduced-lead ECGs in home environments and comprehensive comparative analysis of multiple ECGs. 

---
# Uncertainty-Aware Metabolic Stability Prediction with Dual-View Contrastive Learning 

**Authors**: Peijin Guo, Minghui Li, Hewen Pan, Bowen Chen, Yang Wu, Zikang Guo, Leo Yu Zhang, Shengshan Hu, Shengqing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00936)  

**Abstract**: Accurate prediction of molecular metabolic stability (MS) is critical for drug research and development but remains challenging due to the complex interplay of molecular interactions. Despite recent advances in graph neural networks (GNNs) for MS prediction, current approaches face two critical limitations: (1) incomplete molecular modeling due to atom-centric message-passing mechanisms that disregard bond-level topological features, and (2) prediction frameworks that lack reliable uncertainty quantification. To address these challenges, we propose TrustworthyMS, a novel contrastive learning framework designed for uncertainty-aware metabolic stability prediction. First, a molecular graph topology remapping mechanism synchronizes atom-bond interactions through edge-induced feature propagation, capturing both localized electronic effects and global conformational constraints. Second, contrastive topology-bond alignment enforces consistency between molecular topology views and bond patterns via feature alignment, enhancing representation robustness. Third, uncertainty modeling through Beta-Binomial uncertainty quantification enables simultaneous prediction and confidence calibration under epistemic uncertainty. Through extensive experiments, our results demonstrate that TrustworthyMS outperforms current state-of-the-art methods in terms of predictive performance. 

---
# General-purpose audio representation learning for real-world sound scenes 

**Authors**: Goksenin Yuksel, Marcel van Gerven, Kiki van der Heijden  

**Link**: [PDF](https://arxiv.org/pdf/2506.00934)  

**Abstract**: While audio foundation models perform well on myriad of tasks from sound classification to speech analysis, these models are trained and tested on dry, non-spatial, single-source audio clips. This limits their success in real-world situations and results in spatially unaware audio embeddings. To address these limitations, we propose a novel self-supervised training approach for General-Purpose, Real-world Audio Models (GRAMs). The GRAM training approach enables robust spatial audio representation learning for naturalistic, noisy sound scenes and can be applied to any masking-based deep learning model. We demonstrate the success of our approach by training two state-of-the-art models, one with a transformer and one with a mamba backbone. We assess the quality of the extracted audio representations from GRAMs using the original version of the HEAR benchmark, a newly synthesized, naturalistic version of the HEAR benchmark, and novel sound localization tasks based on HEAR benchmark datasets. The results show that our approach minimizes the performance gap between dry, non-spatial, single-source sound scenes and naturalistic sound scenes for crucial tasks such as auditory scene analysis, outperforming existing state-of-the-art audio foundation models at a fraction of the training steps. Moreover, GRAMs show state-of-the-art performance on sound localization tasks, exceeding even supervised sound localization models. In sum, the proposed approach represents a significant advancement towards robust audio foundation models for real-world applications with state-of-the-art performance on naturalistic sound scenes as well as spatial audio representation learning. 

---
# In-the-wild Audio Spatialization with Flexible Text-guided Localization 

**Authors**: Tianrui Pan, Jie Liu, Zewen Huang, Jie Tang, Gangshan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00927)  

**Abstract**: To enhance immersive experiences, binaural audio offers spatial awareness of sounding objects in AR, VR, and embodied AI applications. While existing audio spatialization methods can generally map any available monaural audio to binaural audio signals, they often lack the flexible and interactive control needed in complex multi-object user-interactive environments. To address this, we propose a Text-guided Audio Spatialization (TAS) framework that utilizes flexible text prompts and evaluates our model from unified generation and comprehension perspectives. Due to the limited availability of premium and large-scale stereo data, we construct the SpatialTAS dataset, which encompasses 376,000 simulated binaural audio samples to facilitate the training of our model. Our model learns binaural differences guided by 3D spatial location and relative position prompts, augmented by flipped-channel audio. It outperforms existing methods on both simulated and real-recorded datasets, demonstrating superior generalization and accuracy. Besides, we develop an assessment model based on Llama-3.1-8B, which evaluates the spatial semantic coherence between our generated binaural audio and text prompts through a spatial reasoning task. Results demonstrate that text prompts provide flexible and interactive control to generate binaural audio with excellent quality and semantic consistency in spatial locations. Dataset is available at \href{this https URL} 

---
# Bridging Subjective and Objective QoE: Operator-Level Aggregation Using LLM-Based Comment Analysis and Network MOS Comparison 

**Authors**: Parsa Hassani Shariat Panahi, Amir Hossein Jalilvand, M. Hasan Najafi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00924)  

**Abstract**: This paper introduces a dual-layer framework for network operator-side quality of experience (QoE) assessment that integrates both objective network modeling and subjective user perception extracted from live-streaming platforms. On the objective side, we develop a machine learning model trained on mean opinion scores (MOS) computed via the ITU-T P.1203 reference implementation, allowing accurate prediction of user-perceived video quality using only network parameters such as packet loss, delay, jitter, and throughput without reliance on video content or client-side instrumentation. On the subjective side, we present a semantic filtering and scoring pipeline that processes user comments from live streams to extract performance-related feedback. A large language model is used to assign scalar MOS scores to filtered comments in a deterministic and reproducible manner. To support scalable and interpretable analysis, we con- struct a labeled dataset of 47,894 live-stream comments, of which about 34,000 are identified as QoE-relevant through multi-layer semantic filtering. Each comment is enriched with simulated Internet Service Provider attribution and temporally aligned using synthetic timestamps in 5-min intervals. The resulting dataset enables operator-level aggregation and time-series analysis of user-perceived quality. A delta MOS metric is proposed to measure each Internet service provider's deviation from platform-wide sentiment, allowing detection of localized degradations even in the absence of direct network telemetry. A controlled outage simulation confirms the framework's effectiveness in identifying service disruptions through comment-based trends alone. The system provides each operator with its own subjective MOS and the global platform average per interval, enabling real-time interpretation of performance deviations and comparison with objective network-based QoE estimates. 

---
# Position as Probability: Self-Supervised Transformers that Think Past Their Training for Length Extrapolation 

**Authors**: Philip Heejun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00920)  

**Abstract**: Deep sequence models typically degrade in accuracy when test sequences significantly exceed their training lengths, yet many critical tasks--such as algorithmic reasoning, multi-step arithmetic, and compositional generalization--require robust length extrapolation. We introduce PRISM, a Probabilistic Relative-position Implicit Superposition Model, a novel positional encoding mechanism that enables Transformers to extrapolate accurately up to 10x beyond their training length. PRISM learns continuous relative positions through a differentiable histogram-filter update, preserving position uncertainty via a probabilistic superposition rather than conventional deterministic embeddings. Empirically, PRISM achieves state-of-the-art length extrapolation, successfully generalizing to previously intractable sequence lengths across algorithmic benchmarks--including arithmetic (addition, multiplication), SCAN compositionality tasks, and complex copy variants derived from DeepMind's recent datasets. Our analysis demonstrates that PRISM's stochastic positional encoding maintains sharp and interpretable internal states, providing a theoretical basis for reliable length generalization. These results advance the goal of neural sequence models that remain algorithmically robust at lengths far exceeding their training horizon. 

---
# Principled Input-Output-Conditioned Post-Hoc Uncertainty Estimation for Regression Networks 

**Authors**: Lennart Bramlage, Cristóbal Curio  

**Link**: [PDF](https://arxiv.org/pdf/2506.00918)  

**Abstract**: Uncertainty quantification is critical in safety-sensitive applications but is often omitted from off-the-shelf neural networks due to adverse effects on predictive performance. Retrofitting uncertainty estimates post-hoc typically requires access to model parameters or gradients, limiting feasibility in practice. We propose a theoretically grounded framework for post-hoc uncertainty estimation in regression tasks by fitting an auxiliary model to both original inputs and frozen model outputs. Drawing from principles of maximum likelihood estimation and sequential parameter fitting, we formalize an exact post-hoc optimization objective that recovers the canonical MLE of Gaussian parameters, without requiring sampling or approximation at inference. While prior work has used model outputs to estimate uncertainty, we explicitly characterize the conditions under which this is valid and demonstrate the extent to which structured outputs can support quasi-epistemic inference. We find that using diverse auxiliary data, such as augmented subsets of the original training data, significantly enhances OOD detection and metric performance. Our hypothesis that frozen model outputs contain generalizable latent information about model error and predictive uncertainty is tested and confirmed. Finally, we ensure that our method maintains proper estimation of input-dependent uncertainty without relying exclusively on base model forecasts. These findings are demonstrated in toy problems and adapted to both UCI and depth regression benchmarks. Code: this https URL. 

---
# How do Transformer Embeddings Represent Compositions? A Functional Analysis 

**Authors**: Aishik Nagar, Ishaan Singh Rawal, Mansi Dhanania, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00914)  

**Abstract**: Compositionality is a key aspect of human intelligence, essential for reasoning and generalization. While transformer-based models have become the de facto standard for many language modeling tasks, little is known about how they represent compound words, and whether these representations are compositional. In this study, we test compositionality in Mistral, OpenAI Large, and Google embedding models, and compare them with BERT. First, we evaluate compositionality in the representations by examining six diverse models of compositionality (addition, multiplication, dilation, regression, etc.). We find that ridge regression, albeit linear, best accounts for compositionality. Surprisingly, we find that the classic vector addition model performs almost as well as any other model. Next, we verify that most embedding models are highly compositional, while BERT shows much poorer compositionality. We verify and visualize our findings with a synthetic dataset consisting of fully transparent adjective-noun compositions. Overall, we present a thorough investigation of compositionality. 

---
# Pi-SQL: Enhancing Text-to-SQL with Fine-Grained Guidance from Pivot Programming Languages 

**Authors**: Yongdong chi, Hanqing Wang, Zonghan Yang, Jian Yang, Xiao Yan, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00912)  

**Abstract**: Text-to-SQL transforms the user queries from natural language to executable SQL programs, enabling non-experts to interact with complex databases. Existing prompt-based methods craft meticulous text guidelines and examples to facilitate SQL generation, but their accuracy is hindered by the large semantic gap between the texts and the low-resource SQL programs. In this work, we propose Pi-SQL, which incorporates the high-resource Python program as a pivot to bridge between the natural language query and SQL program. In particular, Pi-SQL first generates Python programs that provide fine-grained step-by-step guidelines in their code blocks or comments, and then produces an SQL program following the guidance of each Python this http URL final SQL program matches the reference Python program's query results and, through selection from candidates generated by different strategies, achieves superior execution speed, with a reward-based valid efficiency score up to 4.55 higher than the best-performing this http URL experiments demonstrate the effectiveness of Pi-SQL, which improves the execution accuracy of the best-performing baseline by up to 3.20. 

---
# PCoreSet: Effective Active Learning through Knowledge Distillation from Vision-Language Models 

**Authors**: Seongjae Kang, Dong Bok Lee, Hyungjoon Jang, Dongseop Kim, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00910)  

**Abstract**: Knowledge distillation (KD) is a widely used framework for training compact, task-specific models by leveraging the knowledge of teacher models. However, its application to active learning (AL), which aims to minimize annotation costs through iterative sample selection, remains underexplored. This gap stems from the fact that KD typically assumes access to sufficient labeled data, whereas AL operates in data-scarce scenarios where task-specific teacher models are often unavailable. In this paper, we introduce ActiveKD, a framework that integrates AL with KD by leveraging the zero- and few-shot capabilities of large vision-language models (VLMs). A key aspect of ActiveKD is the structured prediction bias of VLMs--i.e., their predictions form clusters in the probability space. We regard this structure as an inductive bias of the teacher model, capturing generalizable output patterns beneficial to student learning. To exploit this bias, we propose Probabilistic CoreSet (PCoreSet), a selection strategy that maximizes coverage in the probability space rather than the feature space. PCoreSet strategically selects categorically diverse unlabeled samples, facilitating more efficient transfer of teacher knowledge under limited annotation budgets. Evaluations on 11 datasets show that PCoreSet consistently outperforms existing selection methods within the ActiveKD framework, advancing research at the intersection of AL and KD. 

---
# State-Covering Trajectory Stitching for Diffusion Planners 

**Authors**: Kyowoon Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00895)  

**Abstract**: Diffusion-based generative models are emerging as powerful tools for long-horizon planning in reinforcement learning (RL), particularly with offline datasets. However, their performance is fundamentally limited by the quality and diversity of training data. This often restricts their generalization to tasks outside their training distribution or longer planning horizons. To overcome this challenge, we propose State-Covering Trajectory Stitching (SCoTS), a novel reward-free trajectory augmentation method that incrementally stitches together short trajectory segments, systematically generating diverse and extended trajectories. SCoTS first learns a temporal distance-preserving latent representation that captures the underlying temporal structure of the environment, then iteratively stitches trajectory segments guided by directional exploration and novelty to effectively cover and expand this latent space. We demonstrate that SCoTS significantly improves the performance and generalization capabilities of diffusion planners on offline goal-conditioned benchmarks requiring stitching and long-horizon reasoning. Furthermore, augmented trajectories generated by SCoTS significantly improve the performance of widely used offline goal-conditioned RL algorithms across diverse environments. 

---
# CODEMENV: Benchmarking Large Language Models on Code Migration 

**Authors**: Keyuan Cheng, Xudong Shen, Yihao Yang, Tengyue Wang, Yang Cao, Muhammad Asif Ali, Hanbin Wang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00894)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities across various software engineering tasks; however, their effectiveness in code migration, adapting code to run in different environments, remains insufficiently studied. In this work, we introduce CODEMENV: Code Migration Across Environment, a new benchmark specifically designed to assess LLMs' abilities in code migration scenarios. CODEMENV consists of 922 examples spanning 19 Python and Java packages, and covers three core tasks: (1) identifying functions incompatible with specific versions, (2) detecting changes in function definitions, and (3) adapting code to target environments. Experimental evaluation with seven LLMs on CODEMENV yields an average pass@1 rate of 26.50%, with GPT-4O achieving the highest score at 43.84%. Key findings include: (i) LLMs tend to be more proficient with newer function versions, which aids in migrating legacy code, and (ii) LLMs sometimes exhibit logical inconsistencies by identifying function changes irrelevant to the intended migration environment. The datasets are available at this https URL. 

---
# Affordance Benchmark for MLLMs 

**Authors**: Junying Wang, Wenzhe Li, Yalun Wu, Yingji Liang, Yijin Guo, Chunyi Li, Haodong Duan, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00893)  

**Abstract**: Affordance theory posits that environments inherently offer action possibilities that shape perception and behavior. While Multimodal Large Language Models (MLLMs) excel in vision-language tasks, their ability to perceive affordance, which is crucial for intuitive and safe interactions, remains underexplored. To address this, we introduce A4Bench, a novel benchmark designed to evaluate the affordance perception abilities of MLLMs across two dimensions: 1) Constitutive Affordance}, assessing understanding of inherent object properties through 1,282 question-answer pairs spanning nine sub-disciplines, and 2) Transformative Affordance, probing dynamic and contextual nuances (e.g., misleading, time-dependent, cultural, or individual-specific affordance) with 718 challenging question-answer pairs. Evaluating 17 MLLMs (nine proprietary and eight open-source) against human performance, we find that proprietary models generally outperform open-source counterparts, but all exhibit limited capabilities, particularly in transformative affordance perception. Furthermore, even top-performing models, such as Gemini-2.0-Pro (18.05% overall exact match accuracy), significantly lag behind human performance (best: 85.34%, worst: 81.25%). These findings highlight critical gaps in environmental understanding of MLLMs and provide a foundation for advancing AI systems toward more robust, context-aware interactions. The dataset is available in this https URL. 

---
# Uneven Event Modeling for Partially Relevant Video Retrieval 

**Authors**: Sa Zhu, Huashan Chen, Wanqian Zhang, Jinchao Zhang, Zexian Yang, Xiaoshuai Hao, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00891)  

**Abstract**: Given a text query, partially relevant video retrieval (PRVR) aims to retrieve untrimmed videos containing relevant moments, wherein event modeling is crucial for partitioning the video into smaller temporal events that partially correspond to the text. Previous methods typically segment videos into a fixed number of equal-length clips, resulting in ambiguous event boundaries. Additionally, they rely on mean pooling to compute event representations, inevitably introducing undesired misalignment. To address these, we propose an Uneven Event Modeling (UEM) framework for PRVR. We first introduce the Progressive-Grouped Video Segmentation (PGVS) module, to iteratively formulate events in light of both temporal dependencies and semantic similarity between consecutive frames, enabling clear event boundaries. Furthermore, we also propose the Context-Aware Event Refinement (CAER) module to refine the event representation conditioned the text's cross-attention. This enables event representations to focus on the most relevant frames for a given text, facilitating more precise text-video alignment. Extensive experiments demonstrate that our method achieves state-of-the-art performance on two PRVR benchmarks. 

---
# CoVoMix2: Advancing Zero-Shot Dialogue Generation with Fully Non-Autoregressive Flow Matching 

**Authors**: Leying Zhang, Yao Qian, Xiaofei Wang, Manthan Thakker, Dongmei Wang, Jianwei Yu, Haibin Wu, Yuxuan Hu, Jinyu Li, Yanmin Qian, Sheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00885)  

**Abstract**: Generating natural-sounding, multi-speaker dialogue is crucial for applications such as podcast creation, virtual agents, and multimedia content generation. However, existing systems struggle to maintain speaker consistency, model overlapping speech, and synthesize coherent conversations efficiently. In this paper, we introduce CoVoMix2, a fully non-autoregressive framework for zero-shot multi-talker dialogue generation. CoVoMix2 directly predicts mel-spectrograms from multi-stream transcriptions using a flow-matching-based generative model, eliminating the reliance on intermediate token representations. To better capture realistic conversational dynamics, we propose transcription-level speaker disentanglement, sentence-level alignment, and prompt-level random masking strategies. Our approach achieves state-of-the-art performance, outperforming strong baselines like MoonCast and Sesame in speech quality, speaker consistency, and inference speed. Notably, CoVoMix2 operates without requiring transcriptions for the prompt and supports controllable dialogue generation, including overlapping speech and precise timing control, demonstrating strong generalizability to real-world speech generation scenarios. 

---
# ModuLM: Enabling Modular and Multimodal Molecular Relational Learning with Large Language Models 

**Authors**: Zhuo Chen, Yizhen Zheng, Huan Yee Koh, Hongxin Xiang, Linjiang Chen, Wenjie Du, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00880)  

**Abstract**: Molecular Relational Learning (MRL) aims to understand interactions between molecular pairs, playing a critical role in advancing biochemical research. With the recent development of large language models (LLMs), a growing number of studies have explored the integration of MRL with LLMs and achieved promising results. However, the increasing availability of diverse LLMs and molecular structure encoders has significantly expanded the model space, presenting major challenges for benchmarking. Currently, there is no LLM framework that supports both flexible molecular input formats and dynamic architectural switching. To address these challenges, reduce redundant coding, and ensure fair model comparison, we propose ModuLM, a framework designed to support flexible LLM-based model construction and diverse molecular representations. ModuLM provides a rich suite of modular components, including 8 types of 2D molecular graph encoders, 11 types of 3D molecular conformation encoders, 7 types of interaction layers, and 7 mainstream LLM backbones. Owing to its highly flexible model assembly mechanism, ModuLM enables the dynamic construction of over 50,000 distinct model configurations. In addition, we provide comprehensive results to demonstrate the effectiveness of ModuLM in supporting LLM-based MRL tasks. 

---
# Towards Predicting Any Human Trajectory In Context 

**Authors**: Ryo Fujii, Hideo Saito, Ryo Hachiuma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00871)  

**Abstract**: Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, this process is often impractical on edge devices due to constrained computational resources. To address this challenge, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables rapid adaptation without fine-tuning on the scenario-specific data. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. The code will be released at this https URL. 

---
# Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning 

**Authors**: Kyowoon Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00867)  

**Abstract**: Recent advances in diffusion-based generative modeling have demonstrated significant promise in tackling long-horizon, sparse-reward tasks by leveraging offline datasets. While these approaches have achieved promising results, their reliability remains inconsistent due to the inherent stochastic risk of producing infeasible trajectories, limiting their applicability in safety-critical applications. We identify that the primary cause of these failures is inaccurate guidance during the sampling procedure, and demonstrate the existence of manifold deviation by deriving a lower bound on the guidance gap. To address this challenge, we propose Local Manifold Approximation and Projection (LoMAP), a training-free method that projects the guided sample onto a low-rank subspace approximated from offline datasets, preventing infeasible trajectory generation. We validate our approach on standard offline reinforcement learning benchmarks that involve challenging long-horizon planning. Furthermore, we show that, as a standalone module, LoMAP can be incorporated into the hierarchical diffusion planner, providing further performance enhancements. 

---
# Can AI Master Econometrics? Evidence from Econometrics AI Agent on Expert-Level Tasks 

**Authors**: Qiang Chen, Tianyang Han, Jin Li, Ye Luo, Yuxiao Wu, Xiaowei Zhang, Tuo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00856)  

**Abstract**: Can AI effectively perform complex econometric analysis traditionally requiring human expertise? This paper evaluates an agentic AI's capability to master econometrics, focusing on empirical analysis performance. We develop an ``Econometrics AI Agent'' built on the open-source MetaGPT framework. This agent exhibits outstanding performance in: (1) planning econometric tasks strategically, (2) generating and executing code, (3) employing error-based reflection for improved robustness, and (4) allowing iterative refinement through multi-round conversations. We construct two datasets from academic coursework materials and published research papers to evaluate performance against real-world challenges. Comparative testing shows our domain-specialized agent significantly outperforms both benchmark large language models (LLMs) and general-purpose AI agents. This work establishes a testbed for exploring AI's impact on social science research and enables cost-effective integration of domain expertise, making advanced econometric methods accessible to users with minimal coding expertise. Furthermore, our agent enhances research reproducibility and offers promising pedagogical applications for econometrics teaching. 

---
# EEG2TEXT-CN: An Exploratory Study of Open-Vocabulary Chinese Text-EEG Alignment via Large Language Model and Contrastive Learning on ChineseEEG 

**Authors**: Jacky Tai-Yu Lu, Jung Chiang, Chi-Sheng Chen, Anna Nai-Yun Tung, Hsiang Wei Hu, Yuan Chiao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00854)  

**Abstract**: We propose EEG2TEXT-CN, which, to the best of our knowledge, represents one of the earliest open-vocabulary EEG-to-text generation frameworks tailored for Chinese. Built on a biologically grounded EEG encoder (NICE-EEG) and a compact pretrained language model (MiniLM), our architecture aligns multichannel brain signals with natural language representations via masked pretraining and contrastive learning. Using a subset of the ChineseEEG dataset, where each sentence contains approximately ten Chinese characters aligned with 128-channel EEG recorded at 256 Hz, we segment EEG into per-character embeddings and predict full sentences in a zero-shot setting. The decoder is trained with teacher forcing and padding masks to accommodate variable-length sequences. Evaluation on over 1,500 training-validation sentences and 300 held-out test samples shows promising lexical alignment, with a best BLEU-1 score of 6.38\%. While syntactic fluency remains a challenge, our findings demonstrate the feasibility of non-phonetic, cross-modal language decoding from EEG. This work opens a new direction in multilingual brain-to-text research and lays the foundation for future cognitive-language interfaces in Chinese. 

---
# Generalization in VAE and Diffusion Models: A Unified Information-Theoretic Analysis 

**Authors**: Qi Chen, Jierui Zhu, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2506.00849)  

**Abstract**: Despite the empirical success of Diffusion Models (DMs) and Variational Autoencoders (VAEs), their generalization performance remains theoretically underexplored, especially lacking a full consideration of the shared encoder-generator structure. Leveraging recent information-theoretic tools, we propose a unified theoretical framework that provides guarantees for the generalization of both the encoder and generator by treating them as randomized mappings. This framework further enables (1) a refined analysis for VAEs, accounting for the generator's generalization, which was previously overlooked; (2) illustrating an explicit trade-off in generalization terms for DMs that depends on the diffusion time $T$; and (3) providing computable bounds for DMs based solely on the training data, allowing the selection of the optimal $T$ and the integration of such bounds into the optimization process to improve model performance. Empirical results on both synthetic and real datasets illustrate the validity of the proposed theory. 

---
# Speech Unlearning 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2506.00848)  

**Abstract**: We introduce machine unlearning for speech tasks, a novel and underexplored research problem that aims to efficiently and effectively remove the influence of specific data from trained speech models without full retraining. This has important applications in privacy preservation, removal of outdated or noisy data, and bias mitigation. While machine unlearning has been studied in computer vision and natural language processing, its application to speech is largely unexplored due to the high-dimensional, sequential, and speaker-dependent nature of speech data. We define two fundamental speech unlearning tasks: sample unlearning, which removes individual data points (e.g., a voice recording), and class unlearning, which removes an entire category (e.g., all data from a speaker), while preserving performance on the remaining data. Experiments on keyword spotting and speaker identification demonstrate that unlearning speech data is significantly more challenging than unlearning image or text data. We conclude with key future directions in this area, including structured training, robust evaluation, feature-level unlearning, broader applications, scalable methods, and adversarial robustness. 

---
# Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience 

**Authors**: Jiawei Gu, Ziting Xian, Yuanzhen Xie, Ye Liu, Enjie Liu, Ruichao Zhong, Mochi Gao, Yunzhi Tan, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00842)  

**Abstract**: Large language models (LLMs) achieve strong performance on plain text tasks but underperform on structured data like tables and databases. Potential challenges arise from their underexposure during pre-training and rigid text-to-structure transfer mechanisms. Unlike humans who seamlessly apply learned patterns across data modalities, LLMs struggle to infer implicit relationships embedded in tabular formats, especially in the absence of explicit structural guidance. To bridge this cognitive gap, we introduce Contrastive Retrieval-Augmented Generation on Experience (CoRE), a framework that builds experience memory representations and enhances generalization through contrastive In-Context Learning (ICL) to simulate human-like knowledge transfer. Experiments on Text-to-SQL and TableQA show CoRE significantly improves performance, achieving average gains of 3.44% and 4.24%, with up to 17.2% on challenging tasks. Our Monte Carlo Tree Search (MCTS)-generated Experience Memory expands training data 8-9x, enhancing diversity and domain coverage. This training-free and continual method propels LLMs toward structured knowledge expertise. 

---
# Counterfactual Activation Editing for Post-hoc Prosody and Mispronunciation Correction in TTS Models 

**Authors**: Kyowoon Lee, Artyom Stitsyuk, Gunu Jho, Inchul Hwang, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00832)  

**Abstract**: Recent advances in Text-to-Speech (TTS) have significantly improved speech naturalness, increasing the demand for precise prosody control and mispronunciation correction. Existing approaches for prosody manipulation often depend on specialized modules or additional training, limiting their capacity for post-hoc adjustments. Similarly, traditional mispronunciation correction relies on grapheme-to-phoneme dictionaries, making it less practical in low-resource settings. We introduce Counterfactual Activation Editing, a model-agnostic method that manipulates internal representations in a pre-trained TTS model to achieve post-hoc control of prosody and pronunciation. Experimental results show that our method effectively adjusts prosodic features and corrects mispronunciations while preserving synthesis quality. This opens the door to inference-time refinement of TTS outputs without retraining, bridging the gap between pre-trained TTS models and editable speech synthesis. 

---
# A Large Language Model-Supported Threat Modeling Framework for Transportation Cyber-Physical Systems 

**Authors**: M Sabbir Salek, Mashrur Chowdhury, Muhaimin Bin Munir, Yuchen Cai, Mohammad Imtiaz Hasan, Jean-Michel Tine, Latifur Khan, Mizanur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2506.00831)  

**Abstract**: Modern transportation systems rely on cyber-physical systems (CPS), where cyber systems interact seamlessly with physical systems like transportation-related sensors and actuators to enhance safety, mobility, and energy efficiency. However, growing automation and connectivity increase exposure to cyber vulnerabilities. Existing threat modeling frameworks for transportation CPS are often limited in scope, resource-intensive, and dependent on significant cybersecurity expertise. To address these gaps, we present TraCR-TMF (Transportation Cybersecurity and Resiliency Threat Modeling Framework), a large language model (LLM)-based framework that minimizes expert intervention. TraCR-TMF identifies threats, potential attack techniques, and corresponding countermeasures by leveraging the MITRE ATT&CK matrix through three LLM-based approaches: (i) a retrieval-augmented generation (RAG) method requiring no expert input, (ii) an in-context learning approach requiring low expert input, and (iii) a supervised fine-tuning method requiring moderate expert input. TraCR-TMF also maps attack paths to critical assets by analyzing vulnerabilities using a customized LLM. The framework was evaluated in two scenarios. First, it identified relevant attack techniques across transportation CPS applications, with 90% precision as validated by experts. Second, using a fine-tuned LLM, it successfully predicted multiple exploitations including lateral movement, data exfiltration, and ransomware-related encryption that occurred during a major real-world cyberattack incident. These results demonstrate TraCR-TMF's effectiveness in CPS threat modeling, its reduced reliance on cybersecurity expertise, and its adaptability across CPS domains. 

---
# COMPKE: Complex Question Answering under Knowledge Editing 

**Authors**: Keyuan Cheng, Zijian Kan, Zhixian He, Zhuoran Zhang, Muhammad Asif Ali, Ke Xu, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00829)  

**Abstract**: Knowledge Editing, which efficiently modifies the knowledge in large language models, has gathered great attention. Current benchmarks primarily use multi-hop question answering to assess and analyze newly injected or updated knowledge. However, we argue that these benchmarks fail to effectively evaluate how well the updated models apply this knowledge in real-life scenarios, particularly when questions require complex reasoning, involving one-to-many relationships or multi-step logical intersections. To fill in this gap, we introduce a new benchmark, COMPKE: Complex Question Answering under Knowledge Editing, which includes 11,924 complex questions that reflect real-life situations. We conduct an extensive evaluation of four knowledge editing methods on COMPKE, revealing that their effectiveness varies notably across different models. For instance, MeLLo attains an accuracy of 39.47 on GPT-4O-MINI, but this drops sharply to 3.83 on QWEN2.5-3B. We further investigate the underlying causes of these disparities from both methodological and model-specific perspectives. The datasets are available at this https URL. 

---
# HERGC: Heterogeneous Experts Representation and Generative Completion for Multimodal Knowledge Graphs 

**Authors**: Yongkang Xiao, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00826)  

**Abstract**: Multimodal knowledge graphs (MMKGs) enrich traditional knowledge graphs (KGs) by incorporating diverse modalities such as images and text. Multi-modal knowledge graph completion (MMKGC) seeks to exploit these heterogeneous signals to infer missing facts, thereby mitigating the intrinsic incompleteness of MMKGs. Existing MMKGC methods typically leverage only the information contained in the MMKGs under the closed-world assumption and adopt discriminative training objectives, which limits their reasoning capacity during completion. Recent generative completion approaches powered by advanced large language models (LLMs) have shown strong reasoning abilities in unimodal knowledge graph completion, but their potential in MMKGC remains largely unexplored. To bridge this gap, we propose HERGC, a Heterogeneous Experts Representation and Generative Completion framework for MMKGs. HERGC first deploys a Heterogeneous Experts Representation Retriever that enriches and fuses multimodal information and retrieves a compact candidate set for each incomplete triple. It then uses a Generative LLM Predictor fine-tuned on minimal instruction data to accurately identify the correct answer from these candidates. Extensive experiments on three standard MMKG benchmarks demonstrate HERGC's effectiveness and robustness, achieving state-of-the-art performance. 

---
# SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models 

**Authors**: Huixin Zhan, Jason H. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2506.00821)  

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated significant success in variant effect prediction. However, their adversarial robustness remains largely unexplored. To address this gap, we propose SafeGenes: a framework for Secure analysis of genomic foundation models, leveraging adversarial attacks to evaluate robustness against both engineered near-identical adversarial Genes and embedding-space manipulations. In this study, we assess the adversarial vulnerabilities of GFMs using two approaches: the Fast Gradient Sign Method (FGSM) and a soft prompt attack. FGSM introduces minimal perturbations to input sequences, while the soft prompt attack optimizes continuous embeddings to manipulate model predictions without modifying the input tokens. By combining these techniques, SafeGenes provides a comprehensive assessment of GFM susceptibility to adversarial manipulation. Targeted soft prompt attacks led to substantial performance degradation, even in large models such as ESM1b and ESM1v. These findings expose critical vulnerabilities in current foundation models, opening new research directions toward improving their security and robustness in high-stakes genomic applications such as variant effect prediction. 

---
# DriveMind: A Dual-VLM based Reinforcement Learning Framework for Autonomous Driving 

**Authors**: Dawood Wasif, Terrence J Moore, Chandan K Reddy, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2506.00819)  

**Abstract**: End-to-end autonomous driving systems map sensor data directly to control commands, but remain opaque, lack interpretability, and offer no formal safety guarantees. While recent vision-language-guided reinforcement learning (RL) methods introduce semantic feedback, they often rely on static prompts and fixed objectives, limiting adaptability to dynamic driving scenes. We present DriveMind, a unified semantic reward framework that integrates: (i) a contrastive Vision-Language Model (VLM) encoder for stepwise semantic anchoring; (ii) a novelty-triggered VLM encoder-decoder, fine-tuned via chain-of-thought (CoT) distillation, for dynamic prompt generation upon semantic drift; (iii) a hierarchical safety module enforcing kinematic constraints (e.g., speed, lane centering, stability); and (iv) a compact predictive world model to reward alignment with anticipated ideal states. DriveMind achieves 19.4 +/- 2.3 km/h average speed, 0.98 +/- 0.03 route completion, and near-zero collisions in CARLA Town 2, outperforming baselines by over 4% in success rate. Its semantic reward generalizes zero-shot to real dash-cam data with minimal distributional shift, demonstrating robust cross-domain alignment and potential for real-world deployment. 

---
# L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning 

**Authors**: Xiang Zhang, Run He, Jiao Chen, Di Fang, Ming Li, Ziqian Zeng, Cen Chen, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00816)  

**Abstract**: Class-incremental learning (CIL) enables models to learn new classes continually without forgetting previously acquired knowledge. Multi-label CIL (MLCIL) extends CIL to a real-world scenario where each sample may belong to multiple classes, introducing several challenges: label absence, which leads to incomplete historical information due to missing labels, and class imbalance, which results in the model bias toward majority classes. To address these challenges, we propose Label-Augmented Analytic Adaptation (L3A), an exemplar-free approach without storing past samples. L3A integrates two key modules. The pseudo-label (PL) module implements label augmentation by generating pseudo-labels for current phase samples, addressing the label absence problem. The weighted analytic classifier (WAC) derives a closed-form solution for neural networks. It introduces sample-specific weights to adaptively balance the class contribution and mitigate class imbalance. Experiments on MS-COCO and PASCAL VOC datasets demonstrate that L3A outperforms existing methods in MLCIL tasks. Our code is available at this https URL. 

---
# Unlearning Inversion Attacks for Graph Neural Networks 

**Authors**: Jiahao Zhang, Yilong Wang, Zhiwei Zhang, Xiaorui Liu, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00808)  

**Abstract**: Graph unlearning methods aim to efficiently remove the impact of sensitive data from trained GNNs without full retraining, assuming that deleted information cannot be recovered. In this work, we challenge this assumption by introducing the graph unlearning inversion attack: given only black-box access to an unlearned GNN and partial graph knowledge, can an adversary reconstruct the removed edges? We identify two key challenges: varying probability-similarity thresholds for unlearned versus retained edges, and the difficulty of locating unlearned edge endpoints, and address them with TrendAttack. First, we derive and exploit the confidence pitfall, a theoretical and empirical pattern showing that nodes adjacent to unlearned edges exhibit a large drop in model confidence. Second, we design an adaptive prediction mechanism that applies different similarity thresholds to unlearned and other membership edges. Our framework flexibly integrates existing membership inference techniques and extends them with trend features. Experiments on four real-world datasets demonstrate that TrendAttack significantly outperforms state-of-the-art GNN membership inference baselines, exposing a critical privacy vulnerability in current graph unlearning methods. 

---
# Action Dependency Graphs for Globally Optimal Coordinated Reinforcement Learning 

**Authors**: Jianglin Ding, Jingcheng Tang, Gangshan Jing  

**Link**: [PDF](https://arxiv.org/pdf/2506.00797)  

**Abstract**: Action-dependent individual policies, which incorporate both environmental states and the actions of other agents in decision-making, have emerged as a promising paradigm for achieving global optimality in multi-agent reinforcement learning (MARL). However, the existing literature often adopts auto-regressive action-dependent policies, where each agent's policy depends on the actions of all preceding agents. This formulation incurs substantial computational complexity as the number of agents increases, thereby limiting scalability. In this work, we consider a more generalized class of action-dependent policies, which do not necessarily follow the auto-regressive form. We propose to use the `action dependency graph (ADG)' to model the inter-agent action dependencies. Within the context of MARL problems structured by coordination graphs, we prove that an action-dependent policy with a sparse ADG can achieve global optimality, provided the ADG satisfies specific conditions specified by the coordination graph. Building on this theoretical foundation, we develop a tabular policy iteration algorithm with guaranteed global optimality. Furthermore, we integrate our framework into several SOTA algorithms and conduct experiments in complex environments. The empirical results affirm the robustness and applicability of our approach in more general scenarios, underscoring its potential for broader MARL challenges. 

---
# Behavioral Augmentation of UML Class Diagrams: An Empirical Study of Large Language Models for Method Generation 

**Authors**: Djaber Rouabhia, Ismail Hadjadj  

**Link**: [PDF](https://arxiv.org/pdf/2506.00788)  

**Abstract**: Automating the enrichment of UML class diagrams with behavioral methods from natural language use cases is a significant challenge. This study evaluates nine large language models (LLMs) in augmenting a methodless UML diagram (21 classes, 17 relationships) using 21 structured waste-management use cases. A total of 90 diagrams (3,373 methods) were assessed across six metrics: method quantity, signature richness (visibility, names, parameters, return types), annotation completeness (linking to use cases/actions), structural fidelity, syntactic correctness (PlantUML compilation), and naming convergence (across models). All LLMs produced valid PlantUML diagrams adhering to UML conventions. Some models excelled in method coverage and annotation accuracy, while others showed richer parameterization but weaker traceability. These results demonstrate that LLMs can generate well-structured methods with consistent naming, advancing automated behavioral modeling. However, inconsistencies in annotations and signatures highlight the need for improved prompt engineering and model selection. The rapid generation of these methods supports Agile practices by enabling faster design iterations. Despite their capabilities, human oversight is essential to ensure accuracy, appropriateness, and semantic alignment. This positions LLMs as collaborative partners in software design. All experimental artifacts (\texttt{.puml}, \texttt{.png}, \texttt{.csv}) are publicly available for reproducibility. 

---
# KG-TRACES: Enhancing Large Language Models with Knowledge Graph-constrained Trajectory Reasoning and Attribution Supervision 

**Authors**: Rong Wu, Pinlong Cai, Jianbiao Mei, Licheng Wen, Tao Hu, Xuemeng Yang, Daocheng Fu, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00783)  

**Abstract**: Large language models (LLMs) have made remarkable strides in various natural language processing tasks, but their performance on complex reasoning problems remains hindered by a lack of explainability and trustworthiness. This issue, often manifesting as hallucinations or unattributable reasoning processes, limits their applicability in complex reasoning scenarios. To address this, we propose Knowledge Graph-constrained Trajectory Reasoning Attribution and Chain Explanation Supervision (KG-TRACES), a novel framework that enhances the reasoning ability of LLMs through explicit supervision over reasoning paths and processes. KG-TRACES jointly supervises the model to: (1) predict symbolic relation paths, (2) predict full triple-level reasoning paths, and (3) generate attribution-aware reasoning processes grounded in the reasoning paths. At inference phase, the model adapts to both KG-available and KG-unavailable scenarios, retrieving reasoning paths from a KG when possible or predicting plausible reasoning paths with only intrinsic knowledge when not. This design enables the model to reason in an explainable and source-attributable pattern. Through extensive experiments on complex reasoning tasks, we demonstrate that KG-TRACES significantly outperforms existing SOTA: it improves Hits@1 by 1.6% and F1 by 4.7% on WebQSP, and achieves improvements of 4.8% in Hits@1 and 2.1% in F1 on CWQ. Moreover, we show its transferability to specialized domains such as medicine. By visualizing the intermediate steps of reasoning processes, we further show that the explicit supervision introduced by KG-TRACES leads to more stable and goal-directed reasoning processes, aligning closely with correct answers. Code is available at this https URL. 

---
# LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning 

**Authors**: Zihang Liu, Tianyu Pang, Oleg Balabanov, Chaoqun Yang, Tianjin Huang, Lu Yin, Yaoqing Yang, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00772)  

**Abstract**: Recent studies have shown that supervised fine-tuning of LLMs on a small number of high-quality datasets can yield strong reasoning capabilities. However, full fine-tuning (Full FT), while powerful, is computationally expensive and susceptible to overfitting and catastrophic forgetting, particularly when data is limited. Sparse fine-tuning, which previously achieved notable success by updating only a small subset of model parameters, offers a promising trade-off between efficiency and effectiveness. Yet, it has lagged behind in the LLM era due to the difficulty of identifying parameters truly critical for reasoning. In this work, we state that weights with the largest magnitude after low-rank approximation are critical weights for fine-tuning, which we call Principal Weights. Surprisingly, while magnitude-based sparse fine-tuning performs poorly as a baseline on LLM fine-tuning, it becomes highly effective after rank reduction. These insights motivate our method: Low-rank Informed Sparse Fine-Tuning (LIFT). LIFT only updates the top 5% Principal Weights throughout training and consistently achieves better performance on reasoning tasks than Full FT, while maintaining memory efficiency on par with popular parameter-efficient fine-tuning methods. In addition to strong performance on target domains such as arithmetic reasoning, LIFT also retains up to 20% more source-domain knowledge, compared to Full FT and LoRA. Our code is available at: this https URL. 

---
# Manipulating 3D Molecules in a Fixed-Dimensional SE(3)-Equivariant Latent Space 

**Authors**: Zitao Chen, Yinjun Jia, Zitong Tian, Wei-Ying Ma, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00771)  

**Abstract**: Medicinal chemists often optimize drugs considering their 3D structures and designing structurally distinct molecules that retain key features, such as shapes, pharmacophores, or chemical properties. Previous deep learning approaches address this through supervised tasks like molecule inpainting or property-guided optimization. In this work, we propose a flexible zero-shot molecule manipulation method by navigating in a shared latent space of 3D molecules. We introduce a Variational AutoEncoder (VAE) for 3D molecules, named MolFLAE, which learns a fixed-dimensional, SE(3)-equivariant latent space independent of atom counts. MolFLAE encodes 3D molecules using an SE(3)-equivariant neural network into fixed number of latent nodes, distinguished by learned embeddings. The latent space is regularized, and molecular structures are reconstructed via a Bayesian Flow Network (BFN) conditioned on the encoder's latent output. MolFLAE achieves competitive performance on standard unconditional 3D molecule generation benchmarks. Moreover, the latent space of MolFLAE enables zero-shot molecule manipulation, including atom number editing, structure reconstruction, and coordinated latent interpolation for both structure and properties. We further demonstrate our approach on a drug optimization task for the human glucocorticoid receptor, generating molecules with improved hydrophilicity while preserving key interactions, under computational evaluations. These results highlight the flexibility, robustness, and real-world utility of our method, opening new avenues for molecule editing and optimization. 

---
# Beyond Attention: Learning Spatio-Temporal Dynamics with Emergent Interpretable Topologies 

**Authors**: Sai Vamsi Alisetti, Vikas Kalagi, Sanjukta Krishnagopal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00770)  

**Abstract**: Spatio-temporal forecasting is critical in applications such as traffic prediction, energy demand modeling, and weather monitoring. While Graph Attention Networks (GATs) are popular for modeling spatial dependencies, they rely on predefined adjacency structures and dynamic attention scores, introducing inductive biases and computational overhead that can obscure interpretability.
We propose InterGAT, a simplified alternative to GAT that replaces masked attention with a fully learnable, symmetric node interaction matrix, capturing latent spatial relationships without relying on fixed graph topologies. Our framework, InterGAT-GRU, which incorporates a GRU-based temporal decoder, outperforms the baseline GAT-GRU in forecasting accuracy, achieving at least a 21% improvement on the SZ-Taxi dataset and a 6% improvement on the Los-Loop dataset across all forecasting horizons (15 to 60 minutes). Additionally, we observed reduction in training time by 60-70% compared to GAT-GRU baseline.
Crucially, the learned interaction matrix reveals interpretable structure: it recovers sparse, topology-aware attention patterns that align with community structure. Spectral and clustering analyses show that the model captures both localized and global dynamics, offering insights into the functional topology driving predictions. This highlights how structure learning can simultaneously support prediction, computational efficiency, and topological interpretabil-ity in dynamic graph-based domains. 

---
# "Who experiences large model decay and why?" A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift 

**Authors**: Harvineet Singh, Fan Xia, Alexej Gossmann, Andrew Chuang, Julian C. Hong, Jean Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00756)  

**Abstract**: Machine learning (ML) models frequently experience performance degradation when deployed in new contexts. Such degradation is rarely uniform: some subgroups may suffer large performance decay while others may not. Understanding where and how large differences in performance arise is critical for designing targeted corrective actions that mitigate decay for the most affected subgroups while minimizing any unintended effects. Current approaches do not provide such detailed insight, as they either (i) explain how average performance shifts arise or (ii) identify adversely affected subgroups without insight into how this occurred. To this end, we introduce a Subgroup-scanning Hierarchical Inference Framework for performance drifT (SHIFT). SHIFT first asks "Is there any subgroup with unacceptably large performance decay due to covariate/outcome shifts?" (Where?) and, if so, dives deeper to ask "Can we explain this using more detailed variable(subset)-specific shifts?" (How?). In real-world experiments, we find that SHIFT identifies interpretable subgroups affected by performance decay, and suggests targeted actions that effectively mitigate the decay. 

---
# CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning 

**Authors**: Monoshi Kumar Roy, Simin Chen, Benjamin Steenhoek, Jinjun Peng, Gail Kaiser, Baishakhi Ray, Wei Le  

**Link**: [PDF](https://arxiv.org/pdf/2506.00750)  

**Abstract**: Understanding and reasoning about code semantics is essential for enhancing code LLMs' abilities to solve real-world software engineering (SE) tasks. Although several code reasoning benchmarks exist, most rely on synthetic datasets or educational coding problems and focus on coarse-grained reasoning tasks such as input/output prediction, limiting their effectiveness in evaluating LLMs in practical SE contexts. To bridge this gap, we propose CodeSense, the first benchmark that makes available a spectrum of fine-grained code reasoning tasks concerned with the software engineering of real-world code. We collected Python, C and Java software projects from real-world repositories. We executed tests from these repositories, collected their execution traces, and constructed a ground truth dataset for fine-grained semantic reasoning tasks. We then performed comprehensive evaluations on state-of-the-art LLMs. Our results show a clear performance gap for the models to handle fine-grained reasoning tasks. Although prompting techniques such as chain-of-thought and in-context learning helped, the lack of code semantics in LLMs fundamentally limit models' capabilities of code reasoning. Besides dataset, benchmark and evaluation, our work produced an execution tracing framework and tool set that make it easy to collect ground truth for fine-grained SE reasoning tasks, offering a strong basis for future benchmark construction and model post training. Our code and data are located at this https URL. 

---
# Assortment of Attention Heads: Accelerating Federated PEFT with Head Pruning and Strategic Client Selection 

**Authors**: Yeshwanth Venkatesha, Souvik Kundu, Priyadarshini Panda  

**Link**: [PDF](https://arxiv.org/pdf/2506.00743)  

**Abstract**: Parameter Efficient Fine-Tuning (PEFT) has become the de-facto approach in adapting Large Language Models (LLMs) for downstream tasks in Natural Language Processing. However, its adoption in privacy-preserving distributed learning frameworks, such as Federated Learning (FL), remains relatively limited. This is mainly due to challenges specific to FL, such as resource-constrained devices and diverse data distributions among clients. In this paper, we propose an efficient method to perform PEFT within the FL framework for Multi-Head Attention (MHA) based language models. We address the challenges through head pruning, a novel head-specific weighted aggregation mechanism, and a client selection strategy. Head pruning minimizes training complexity within the clients, guided by the importance score computed based on the confidence of the attention head. Weighted aggregation of heads ensures the global model captures crucial updates from diverse clients complementing our client selection strategy. We show results on the MultiNLI benchmark along with 20 Newsgroups, XL-Sum, and E2E NLG datasets. We use the MultiNLI dataset and T5-small model with LoRA as our PEFT method, attaining sparsity levels of up to 90%, resulting in a communication advantage of up to 1.8x and a reduction in training OPs of 3.9x while maintaining the accuracy drop under 2%. 

---
# ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary 

**Authors**: Zeqi Gu, Yin Cui, Zhaoshuo Li, Fangyin Wei, Yunhao Ge, Jinwei Gu, Ming-Yu Liu, Abe Davis, Yifan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.00742)  

**Abstract**: Designing 3D scenes is traditionally a challenging task that demands both artistic expertise and proficiency with complex software. Recent advances in text-to-3D generation have greatly simplified this process by letting users create scenes based on simple text descriptions. However, as these methods generally require extra training or in-context learning, their performance is often hindered by the limited availability of high-quality 3D data. In contrast, modern text-to-image models learned from web-scale images can generate scenes with diverse, reliable spatial layouts and consistent, visually appealing styles. Our key insight is that instead of learning directly from 3D scenes, we can leverage generated 2D images as an intermediary to guide 3D synthesis. In light of this, we introduce ArtiScene, a training-free automated pipeline for scene design that integrates the flexibility of free-form text-to-image generation with the diversity and reliability of 2D intermediary layouts.
First, we generate 2D images from a scene description, then extract the shape and appearance of objects to create 3D models. These models are assembled into the final scene using geometry, position, and pose information derived from the same intermediary image. Being generalizable to a wide range of scenes and styles, ArtiScene outperforms state-of-the-art benchmarks by a large margin in layout and aesthetic quality by quantitative metrics. It also averages a 74.89% winning rate in extensive user studies and 95.07% in GPT-4o evaluation. Project page: this https URL 

---
# Length Aware Speech Translation for Video Dubbing 

**Authors**: Harveen Singh Chadha, Aswin Shanmugam Subramanian, Vikas Joshi, Shubham Bansal, Jian Xue, Rupeshkumar Mehta, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00740)  

**Abstract**: In video dubbing, aligning translated audio with the source audio is a significant challenge. Our focus is on achieving this efficiently, tailored for real-time, on-device video dubbing scenarios. We developed a phoneme-based end-to-end length-sensitive speech translation (LSST) model, which generates translations of varying lengths short, normal, and long using predefined tags. Additionally, we introduced length-aware beam search (LABS), an efficient approach to generate translations of different lengths in a single decoding pass. This approach maintained comparable BLEU scores compared to a baseline without length awareness while significantly enhancing synchronization quality between source and target audio, achieving a mean opinion score (MOS) gain of 0.34 for Spanish and 0.65 for Korean, respectively. 

---
# MoPINNEnKF: Iterative Model Inference using generic-PINN-based ensemble Kalman filter 

**Authors**: Binghang Lu, Changhong Mou, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00731)  

**Abstract**: Physics-informed neural networks (PINNs) have emerged as a powerful tool for solving forward and inverse problems involving partial differential equations (PDEs) by incorporating physical laws into the training process. However, the performance of PINNs is often hindered in real-world scenarios involving noisy observational data and missing physics, particularly in inverse problems. In this work, we propose an iterative multi-objective PINN ensemble Kalman filter (MoPINNEnKF) framework that improves the robustness and accuracy of PINNs in both forward and inverse problems by using the \textit{ensemble Kalman filter} and the \textit{non-dominated sorting genetic algorithm} III (NSGA-III). Specifically, NSGA-III is used as a multi-objective optimizer that can generate various ensemble members of PINNs along the optimal Pareto front, while accounting the model uncertainty in the solution space. These ensemble members are then utilized within the EnKF to assimilate noisy observational data. The EnKF's analysis is subsequently used to refine the data loss component for retraining the PINNs, thereby iteratively updating their parameters. The iterative procedure generates improved solutions to the PDEs. The proposed method is tested on two benchmark problems: the one-dimensional viscous Burgers equation and the time-fractional mixed diffusion-wave equation (TFMDWE). The numerical results show it outperforms standard PINNs in handling noisy data and missing physics. 

---
# Pitfalls in Evaluating Language Model Forecasters 

**Authors**: Daniel Paleka, Shashwat Goel, Jonas Geiping, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2506.00723)  

**Abstract**: Large language models (LLMs) have recently been applied to forecasting tasks, with some works claiming these systems match or exceed human performance. In this paper, we argue that, as a community, we should be careful about such conclusions as evaluating LLM forecasters presents unique challenges. We identify two broad categories of issues: (1) difficulty in trusting evaluation results due to many forms of temporal leakage, and (2) difficulty in extrapolating from evaluation performance to real-world forecasting. Through systematic analysis and concrete examples from prior work, we demonstrate how evaluation flaws can raise concerns about current and future performance claims. We argue that more rigorous evaluation methodologies are needed to confidently assess the forecasting abilities of LLMs. 

---
# From Local Cues to Global Percepts: Emergent Gestalt Organization in Self-Supervised Vision Models 

**Authors**: Tianqin Li, Ziqi Wen, Leiran Song, Jun Liu, Zhi Jing, Tai Sing Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00718)  

**Abstract**: Human vision organizes local cues into coherent global forms using Gestalt principles like closure, proximity, and figure-ground assignment -- functions reliant on global spatial structure. We investigate whether modern vision models show similar behaviors, and under what training conditions these emerge. We find that Vision Transformers (ViTs) trained with Masked Autoencoding (MAE) exhibit activation patterns consistent with Gestalt laws, including illusory contour completion, convexity preference, and dynamic figure-ground segregation. To probe the computational basis, we hypothesize that modeling global dependencies is necessary for Gestalt-like organization. We introduce the Distorted Spatial Relationship Testbench (DiSRT), which evaluates sensitivity to global spatial perturbations while preserving local textures. Using DiSRT, we show that self-supervised models (e.g., MAE, CLIP) outperform supervised baselines and sometimes even exceed human performance. ConvNeXt models trained with MAE also exhibit Gestalt-compatible representations, suggesting such sensitivity can arise without attention architectures. However, classification finetuning degrades this ability. Inspired by biological vision, we show that a Top-K activation sparsity mechanism can restore global sensitivity. Our findings identify training conditions that promote or suppress Gestalt-like perception and establish DiSRT as a diagnostic for global structure sensitivity across models. 

---
# An LLM Agent for Functional Bug Detection in Network Protocols 

**Authors**: Mingwei Zheng, Chengpeng Wang, Xuwei Liu, Jinyao Guo, Shiwei Feng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00714)  

**Abstract**: Functional correctness is critical for ensuring the reliability and security of network protocol implementations. Functional bugs, instances where implementations diverge from behaviors specified in RFC documents, can lead to severe consequences, including faulty routing, authentication bypasses, and service disruptions. Detecting these bugs requires deep semantic analysis across specification documents and source code, a task beyond the capabilities of traditional static analysis tools. This paper introduces RFCScan, an autonomous agent that leverages large language models (LLMs) to detect functional bugs by checking conformance between network protocol implementations and their RFC specifications. Inspired by the human auditing procedure, RFCScan comprises two key components: an indexing agent and a detection agent. The former hierarchically summarizes protocol code semantics, generating semantic indexes that enable the detection agent to narrow down the scanning scope. The latter employs demand-driven retrieval to iteratively collect additional relevant data structures and functions, eventually identifying potential inconsistencies with the RFC specifications effectively. We evaluate RFCScan across six real-world network protocol implementations. RFCScan identifies 47 functional bugs with 81.9% precision, of which 20 bugs have been confirmed or fixed by developers. 

---
# From Argumentative Text to Argument Knowledge Graph: A New Framework for Structured Argumentation 

**Authors**: Debarati Bhattacharjee, Ashish Anand  

**Link**: [PDF](https://arxiv.org/pdf/2506.00713)  

**Abstract**: This paper presents a framework to convert argumentative texts into argument knowledge graphs (AKG). Starting with basic annotations of argumentative components (ACs) and argumentative relations (ARs), we enrich the information by constructing a knowledge base (KB) graph with metadata attributes for nodes. Next, we use premises and inference rules from the KB to form arguments by applying modus ponens. From these arguments, we create an AKG. The nodes and edges of the AKG have attributes that capture important argumentative features. We also find missing inference rules by identifying markers. This makes it possible to identify undercut attacks that were previously undetectable in existing datasets. The AKG gives a graphical view of the argumentative structure that is easier to understand than theoretical formats. It also prepares the ground for future reasoning tasks, including checking the coherence of arguments and identifying opportunities for revision. For this, it is important to find indirect relations, many of which are implicit. Our proposed AKG format, with annotated inference rules and modus ponens, will help reasoning models learn the implicit indirect relations that require inference over arguments and the relations between them. 

---
# QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training 

**Authors**: Wei Dai, Peilin Chen, Chanakya Ekbote, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00711)  

**Abstract**: Clinical decision-making routinely demands reasoning over heterogeneous data, yet existing multimodal language models (MLLMs) remain largely vision-centric and fail to generalize across clinical specialties. To bridge this gap, we introduce QoQ-Med-7B/32B, the first open generalist clinical foundation model that jointly reasons across medical images, time-series signals, and text reports. QoQ-Med is trained with Domain-aware Relative Policy Optimization (DRPO), a novel reinforcement-learning objective that hierarchically scales normalized rewards according to domain rarity and modality difficulty, mitigating performance imbalance caused by skewed clinical data distributions. Trained on 2.61 million instruction tuning pairs spanning 9 clinical domains, we show that DRPO training boosts diagnostic performance by 43% in macro-F1 on average across all visual domains as compared to other critic-free training methods like GRPO. Furthermore, with QoQ-Med trained on intensive segmentation data, it is able to highlight salient regions related to the diagnosis, with an IoU 10x higher than open models while reaching the performance of OpenAI o4-mini. To foster reproducibility and downstream research, we release (i) the full model weights, (ii) the modular training pipeline, and (iii) all intermediate reasoning traces at this https URL. 

---
# Bayesian Inference of Training Dataset Membership 

**Authors**: Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00701)  

**Abstract**: Determining whether a dataset was part of a machine learning model's training data pool can reveal privacy vulnerabilities, a challenge often addressed through membership inference attacks (MIAs). Traditional MIAs typically require access to model internals or rely on computationally intensive shadow models. This paper proposes an efficient, interpretable and principled Bayesian inference method for membership inference. By analyzing post-hoc metrics such as prediction error, confidence (entropy), perturbation magnitude, and dataset statistics from a trained ML model, our approach computes posterior probabilities of membership without requiring extensive model training. Experimental results on synthetic datasets demonstrate the method's effectiveness in distinguishing member from non-member datasets. Beyond membership inference, this method can also detect distribution shifts, offering a practical and interpretable alternative to existing approaches. 

---
# Measuring Faithfulness and Abstention: An Automated Pipeline for Evaluating LLM-Generated 3-ply Case-Based Legal Arguments 

**Authors**: Li Zhang, Morgan Gray, Jaromir Savelka, Kevin D. Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2506.00694)  

**Abstract**: Large Language Models (LLMs) demonstrate potential in complex legal tasks like argument generation, yet their reliability remains a concern. Building upon pilot work assessing LLM generation of 3-ply legal arguments using human evaluation, this paper introduces an automated pipeline to evaluate LLM performance on this task, specifically focusing on faithfulness (absence of hallucination), factor utilization, and appropriate abstention. We define hallucination as the generation of factors not present in the input case materials and abstention as the model's ability to refrain from generating arguments when instructed and no factual basis exists. Our automated method employs an external LLM to extract factors from generated arguments and compares them against the ground-truth factors provided in the input case triples (current case and two precedent cases). We evaluated eight distinct LLMs on three tests of increasing difficulty: 1) generating a standard 3-ply argument, 2) generating an argument with swapped precedent roles, and 3) recognizing the impossibility of argument generation due to lack of shared factors and abstaining. Our findings indicate that while current LLMs achieve high accuracy (over 90%) in avoiding hallucination on viable argument generation tests (Tests 1 & 2), they often fail to utilize the full set of relevant factors present in the cases. Critically, on the abstention test (Test 3), most models failed to follow instructions to stop, instead generating spurious arguments despite the lack of common factors. This automated pipeline provides a scalable method for assessing these crucial LLM behaviors, highlighting the need for improvements in factor utilization and robust abstention capabilities before reliable deployment in legal settings. Project page: this https URL. 

---
# Optimizing Sensory Neurons: Nonlinear Attention Mechanisms for Accelerated Convergence in Permutation-Invariant Neural Networks for Reinforcement Learning 

**Authors**: Junaid Muzaffar, Ahsan Adeel, Khubaib Ahmed, Ingo Frommholz, Zeeshan Pervez, Ahsan ul Haq  

**Link**: [PDF](https://arxiv.org/pdf/2506.00691)  

**Abstract**: Training reinforcement learning (RL) agents often requires significant computational resources and extended training times. To address this, we build upon the foundation laid by Google Brain's Sensory Neuron, which introduced a novel neural architecture for reinforcement learning tasks that maintained permutation in-variance in the sensory neuron system. While the baseline model demonstrated significant performance improvements over traditional approaches, we identified opportunities to enhance the efficiency of the learning process further. We propose a modified attention mechanism incorporating a non-linear transformation of the key vectors (K) using a mapping function, resulting in a new set of key vectors (K'). This non-linear mapping enhances the representational capacity of the attention mechanism, allowing the model to encode more complex feature interactions and accelerating convergence without compromising performance. Our enhanced model demonstrates significant improvements in learning efficiency, showcasing the potential for non-linear attention mechanisms in advancing reinforcement learning algorithms. 

---
# Existing Large Language Model Unlearning Evaluations Are Inconclusive 

**Authors**: Zhili Feng, Yixuan Even Xu, Alexander Robey, Robert Kirk, Xander Davies, Yarin Gal, Avi Schwarzschild, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2506.00688)  

**Abstract**: Machine unlearning aims to remove sensitive or undesired data from large language models. However, recent studies suggest that unlearning is often shallow, claiming that removed knowledge can easily be recovered. In this work, we critically examine standard unlearning evaluation practices and uncover key limitations that shake our trust in those findings. First, we show that some evaluations introduce substantial new information into the model, potentially masking true unlearning performance by re-teaching the model during testing. Second, we demonstrate that evaluation outcomes vary significantly across tasks, undermining the generalizability of current evaluation routines. Finally, we find that many evaluations rely on spurious correlations, making their results difficult to trust and interpret. Taken together, these issues suggest that current evaluation protocols may both overstate and understate unlearning success. To address this, we propose two principles for future unlearning evaluations: minimal information injection and downstream task awareness. We validate these principles through a series of targeted experiments, showing how violations of each can lead to misleading conclusions. 

---
# CineMA: A Foundation Model for Cine Cardiac MRI 

**Authors**: Yunguan Fu, Weixi Yi, Charlotte Manisty, Anish N Bhuva, Thomas A Treibel, James C Moon, Matthew J Clarkson, Rhodri Huw Davies, Yipeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00679)  

**Abstract**: Cardiac magnetic resonance (CMR) is a key investigation in clinical cardiovascular medicine and has been used extensively in population research. However, extracting clinically important measurements such as ejection fraction for diagnosing cardiovascular diseases remains time-consuming and subjective. We developed CineMA, a foundation AI model automating these tasks with limited labels. CineMA is a self-supervised autoencoder model trained on 74,916 cine CMR studies to reconstruct images from masked inputs. After fine-tuning, it was evaluated across eight datasets on 23 tasks from four categories: ventricle and myocardium segmentation, left and right ventricle ejection fraction calculation, disease detection and classification, and landmark localisation. CineMA is the first foundation model for cine CMR to match or outperform convolutional neural networks (CNNs). CineMA demonstrated greater label efficiency than CNNs, achieving comparable or better performance with fewer annotations. This reduces the burden of clinician labelling and supports replacing task-specific training with fine-tuning foundation models in future cardiac imaging applications. Models and code for pre-training and fine-tuning are available at this https URL, democratising access to high-performance models that otherwise require substantial computational resources, promoting reproducibility and accelerating clinical translation. 

---
# SafeTuneBed: A Toolkit for Benchmarking LLM Safety Alignment in Fine-Tuning 

**Authors**: Saad Hossain, Samanvay Vajpayee, Sirisha Rambhatla  

**Link**: [PDF](https://arxiv.org/pdf/2506.00676)  

**Abstract**: As large language models (LLMs) become ubiquitous, parameter-efficient fine-tuning methods and safety-first defenses have proliferated rapidly. However, the number of approaches and their recent increase have resulted in diverse evaluations-varied datasets, metrics, and inconsistent threat settings-making it difficult to fairly compare safety, utility, and robustness across methods. To address this, we introduce SafeTuneBed, a benchmark and toolkit unifying fine-tuning and defense evaluation. SafeTuneBed (i) curates a diverse repository of multiple fine-tuning datasets spanning sentiment analysis, question-answering, multi-step reasoning, and open-ended instruction tasks, and allows for the generation of harmful-variant splits; (ii) enables integration of state-of-the-art defenses, including alignment-stage immunization, in-training safeguards, and post-tuning repair; and (iii) provides evaluators for safety (attack success rate, refusal consistency) and utility. Built on Python-first, dataclass-driven configs and plugins, SafeTuneBed requires minimal additional code to specify any fine-tuning regime, defense method, and metric suite, while ensuring end-to-end reproducibility. We showcase its value by benchmarking representative defenses across varied poisoning scenarios and tasks. By standardizing data, code, and metrics, SafeTuneBed is the first focused toolkit of its kind to accelerate rigorous and comparable research in safe LLM fine-tuning. Code is available at: this https URL 

---
# Thinking Out of the Box: Hybrid SAT Solving by Unconstrained Continuous Optimization 

**Authors**: Zhiwei Zhang, Samy Wu Fung, Anastasios Kyrillidis, Stanley Osher, Moshe Y. Vardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00674)  

**Abstract**: The Boolean satisfiability (SAT) problem lies at the core of many applications in combinatorial optimization, software verification, cryptography, and machine learning. While state-of-the-art solvers have demonstrated high efficiency in handling conjunctive normal form (CNF) formulas, numerous applications require non-CNF (hybrid) constraints, such as XOR, cardinality, and Not-All-Equal constraints. Recent work leverages polynomial representations to represent such hybrid constraints, but it relies on box constraints that can limit the use of powerful unconstrained optimizers. In this paper, we propose unconstrained continuous optimization formulations for hybrid SAT solving by penalty terms. We provide theoretical insights into when these penalty terms are necessary and demonstrate empirically that unconstrained optimizers (e.g., Adam) can enhance SAT solving on hybrid benchmarks. Our results highlight the potential of combining continuous optimization and machine-learning-based methods for effective hybrid SAT solving. 

---
# Differential Privacy for Deep Learning in Medicine 

**Authors**: Marziyeh Mohammadi, Mohsen Vejdanihemmat, Mahshad Lotfinia, Mirabela Rusu, Daniel Truhn, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00660)  

**Abstract**: Differential privacy (DP) is a key technique for protecting sensitive patient data in medical deep learning (DL). As clinical models grow more data-dependent, balancing privacy with utility and fairness has become a critical challenge. This scoping review synthesizes recent developments in applying DP to medical DL, with a particular focus on DP-SGD and alternative mechanisms across centralized and federated settings. Using a structured search strategy, we identified 74 studies published up to March 2025. Our analysis spans diverse data modalities, training setups, and downstream tasks, and highlights the tradeoffs between privacy guarantees, model accuracy, and subgroup fairness. We find that while DP-especially at strong privacy budgets-can preserve performance in well-structured imaging tasks, severe degradation often occurs under strict privacy, particularly in underrepresented or complex modalities. Furthermore, privacy-induced performance gaps disproportionately affect demographic subgroups, with fairness impacts varying by data type and task. A small subset of studies explicitly addresses these tradeoffs through subgroup analysis or fairness metrics, but most omit them entirely. Beyond DP-SGD, emerging approaches leverage alternative mechanisms, generative models, and hybrid federated designs, though reporting remains inconsistent. We conclude by outlining key gaps in fairness auditing, standardization, and evaluation protocols, offering guidance for future work toward equitable and clinically robust privacy-preserving DL systems in medicine. 

---
# Sarc7: Evaluating Sarcasm Detection and Generation with Seven Types and Emotion-Informed Techniques 

**Authors**: Lang Xiong, Raina Gao, Alyssa Jeong, Yicheng Fu, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00658)  

**Abstract**: Sarcasm is a form of humor where expressions convey meanings opposite to their literal interpretations. Classifying and generating sarcasm using large language models is vital for interpreting human communication. Sarcasm poses challenges for computational models, due to its nuanced nature. We introduce Sarc7, a benchmark that classifies 7 types of sarcasm: self-deprecating, brooding, deadpan, polite, obnoxious, raging, and manic by annotating entries of the MUStARD dataset. Classification was evaluated using zero-shot, few-shot, chain-of-thought (CoT), and a novel emotion-based prompting technique. We propose an emotion-based generation method developed by identifying key components of sarcasm-incongruity, shock value, and context dependency. Our classification experiments show that Gemini 2.5, using emotion-based prompting, outperforms other setups with an F1 score of 0.3664. Human evaluators preferred our emotion-based prompting, with 38.46% more successful generations than zero-shot prompting. 

---
# Permutation-Invariant Transformer Neural Architectures for Set-Based Indoor Localization Using Learned RSSI Embeddings 

**Authors**: Aris J. Aristorenas  

**Link**: [PDF](https://arxiv.org/pdf/2506.00656)  

**Abstract**: We propose a permutation-invariant neural architecture for indoor localization using RSSI scans from Wi-Fi access points. Each scan is modeled as an unordered set of (BSSID, RSSI) pairs, where BSSIDs are mapped to learned embeddings and concatenated with signal strength. These are processed by a Set Transformer, enabling the model to handle variable-length, sparse inputs while learning attention- based representations over access point relationships. We evaluate the model on a dataset collected across a campus environment consisting of six buildings. Results show that the model accurately recovers fine-grained spatial structure and maintains performance across physically distinct domains. In our experiments, a simple LSTM consistently outperformed all other models, achieving the lowest mean localization error across three tasks (E1 - E3), with average errors as low as 2.23 m. The Set Transformer performed competitively, ranking second in every experiment and outperforming the MLP, RNN, and basic attention models, particularly in scenarios involving multiple buildings (E2) and multiple floors (E3). Performance degraded most in E2, where signal conditions varied substantially across buildings, highlighting the importance of architectural robustness to domain diversity. This work demonstrates that set-based neural models are a natural fit for signal-based localization, offering a principled approach to handling sparse, unordered inputs in real-world positioning tasks. 

---
# Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models 

**Authors**: Femi Bello, Anubrata Das, Fanzhi Zeng, Fangcong Yin, Leqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00653)  

**Abstract**: It has been hypothesized that neural networks with similar architectures trained on similar data learn shared representations relevant to the learning task. We build on this idea by extending the conceptual framework where representations learned across models trained on the same data can be expressed as linear combinations of a \emph{universal} set of basis features. These basis features underlie the learning task itself and remain consistent across models, regardless of scale. From this framework, we propose the \textbf{Linear Representation Transferability (LRT)} Hypothesis -- that there exists an affine transformation between the representation spaces of different models. To test this hypothesis, we learn affine mappings between the hidden states of models of different sizes and evaluate whether steering vectors -- directions in hidden state space associated with specific model behaviors -- retain their semantic effect when transferred from small to large language models using the learned mappings. We find strong empirical evidence that such affine mappings can preserve steering behaviors. These findings suggest that representations learned by small models can be used to guide the behavior of large models, and that the LRT hypothesis may be a promising direction on understanding representation alignment across model scales. 

---
# Clinical Annotations for Automatic Stuttering Severity Assessment 

**Authors**: Ana Rita Valente, Rufael Marew, Hawau Olamide Toyin, Hamdan Al-Ali, Anelise Bohnen, Inma Becerra, Elsa Marta Soares, Goncalo Leal, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.00644)  

**Abstract**: Stuttering is a complex disorder that requires specialized expertise for effective assessment and treatment. This paper presents an effort to enhance the FluencyBank dataset with a new stuttering annotation scheme based on established clinical standards. To achieve high-quality annotations, we hired expert clinicians to label the data, ensuring that the resulting annotations mirror real-world clinical expertise. The annotations are multi-modal, incorporating audiovisual features for the detection and classification of stuttering moments, secondary behaviors, and tension scores. In addition to individual annotations, we additionally provide a test set with highly reliable annotations based on expert consensus for assessing individual annotators and machine learning models. Our experiments and analysis illustrate the complexity of this task that necessitates extensive clinical expertise for valid training and evaluation of stuttering assessment models. 

---
# SATA-BENCH: Select All That Apply Benchmark for Multiple Choice Questions 

**Authors**: Weijie Xu, Shixian Cui, Xi Fang, Chi Xue, Stephanie Eckman, Chandan Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00643)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on single-answer multiple-choice tasks, yet many real-world problems require identifying all correct answers from a set of options. This capability remains underexplored. We introduce SATA-BENCH, the first dedicated benchmark for evaluating LLMs on Select All That Apply (SATA) questions across diverse domains, including reading comprehension, law, and biomedicine. Our evaluation of 27 open-source and proprietary models reveals a significant gap: even the strongest model achieves only 41.8% exact match, exposing LLMs' inability to reliably identify all correct answers. We find that this weakness stems from two core challenges: selection bias - models favor certain choices regardless of content, and count bias - models fail to predict the correct number of answers. To address these issues, we propose Choice Funnel, a decoding strategy that combines token debiasing with adaptive thresholding to guide models toward complete and accurate selections. Choice Funnel achieves up to 29% higher exact match than competitive baselines while reducing inference cost by over 64%. Our findings expose fundamental limitations in current LLMs and introduce a new framework for diagnosing and improving multi-answer reasoning. We release SATA-BENCH and Choice Funnel to promote LLM development for robust decision-making in realistic, multi-answer applications. 

---
# Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics 

**Authors**: Lorenzo Jaime Yu Flores, Ori Ernst, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.00637)  

**Abstract**: Well-calibrated model confidence scores can improve the usefulness of text generation models. For example, users can be prompted to review predictions with low confidence scores, to prevent models from returning bad or potentially dangerous predictions. However, confidence metrics are not always well calibrated in text generation. One reason is that in generation, there can be many valid answers, which previous methods do not always account for. Hence, a confident model could distribute its output probability among multiple sequences because they are all valid. We propose task-agnostic confidence metrics suited to generation, which rely solely on the probabilities associated with the model outputs without the need for further fine-tuning or heuristics. Using these, we are able to improve the calibration of BART and Flan-T5 on summarization, translation, and QA datasets. 

---
# Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting 

**Authors**: Wei Chen, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00635)  

**Abstract**: Spatio-temporal forecasting is crucial in many domains, such as transportation, meteorology, and energy. However, real-world scenarios frequently present challenges such as signal anomalies, noise, and distributional shifts. Existing solutions primarily enhance robustness by modifying network architectures or training procedures. Nevertheless, these approaches are computationally intensive and resource-demanding, especially for large-scale applications. In this paper, we explore a novel test-time computing paradigm, namely learning with calibration, ST-TTC, for spatio-temporal forecasting. Through learning with calibration, we aim to capture periodic structural biases arising from non-stationarity during the testing phase and perform real-time bias correction on predictions to improve accuracy. Specifically, we first introduce a spectral-domain calibrator with phase-amplitude modulation to mitigate periodic shift and then propose a flash updating mechanism with a streaming memory queue for efficient test-time computation. ST-TTC effectively bypasses complex training-stage techniques, offering an efficient and generalizable paradigm. Extensive experiments on real-world datasets demonstrate the effectiveness, universality, flexibility and efficiency of our proposed method. 

---
# Text-to-CT Generation via 3D Latent Diffusion Model with Contrastive Vision-Language Pretraining 

**Authors**: Daniele Molino, Camillo Maria Caruso, Filippo Ruffini, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00633)  

**Abstract**: Objective: While recent advances in text-conditioned generative models have enabled the synthesis of realistic medical images, progress has been largely confined to 2D modalities such as chest X-rays. Extending text-to-image generation to volumetric Computed Tomography (CT) remains a significant challenge, due to its high dimensionality, anatomical complexity, and the absence of robust frameworks that align vision-language data in 3D medical imaging. Methods: We introduce a novel architecture for Text-to-CT generation that combines a latent diffusion model with a 3D contrastive vision-language pretraining scheme. Our approach leverages a dual-encoder CLIP-style model trained on paired CT volumes and radiology reports to establish a shared embedding space, which serves as the conditioning input for generation. CT volumes are compressed into a low-dimensional latent space via a pretrained volumetric VAE, enabling efficient 3D denoising diffusion without requiring external super-resolution stages. Results: We evaluate our method on the CT-RATE dataset and conduct a comprehensive assessment of image fidelity, clinical relevance, and semantic alignment. Our model achieves competitive performance across all tasks, significantly outperforming prior baselines for text-to-CT generation. Moreover, we demonstrate that CT scans synthesized by our framework can effectively augment real data, improving downstream diagnostic performance. Conclusion: Our results show that modality-specific vision-language alignment is a key component for high-quality 3D medical image generation. By integrating contrastive pretraining and volumetric diffusion, our method offers a scalable and controllable solution for synthesizing clinically meaningful CT volumes from text, paving the way for new applications in data augmentation, medical education, and automated clinical simulation. 

---
# The Disparate Effects of Partial Information in Bayesian Strategic Learning 

**Authors**: Srikanth Avasarala, Serena Wang, Juba Ziani  

**Link**: [PDF](https://arxiv.org/pdf/2506.00627)  

**Abstract**: We study how partial information about scoring rules affects fairness in strategic learning settings. In strategic learning, a learner deploys a scoring rule, and agents respond strategically by modifying their features -- at some cost -- to improve their outcomes. However, in our work, agents do not observe the scoring rule directly; instead, they receive a noisy signal of said rule. We consider two different agent models: (i) naive agents, who take the noisy signal at face value, and (ii) Bayesian agents, who update a prior belief based on the signal.
Our goal is to understand how disparities in outcomes arise between groups that differ in their costs of feature modification, and how these disparities vary with the level of transparency of the learner's rule. For naive agents, we show that utility disparities can grow unboundedly with noise, and that the group with lower costs can, perhaps counter-intuitively, be disproportionately harmed under limited transparency. In contrast, for Bayesian agents, disparities remain bounded. We provide a full characterization of disparities across groups as a function of the level of transparency and show that they can vary non-monotonically with noise; in particular, disparities are often minimized at intermediate levels of transparency. Finally, we extend our analysis to settings where groups differ not only in cost, but also in prior beliefs, and study how this asymmetry influences fairness. 

---
# Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples 

**Authors**: Haesung Pyun, Yoonah Park, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00622)  

**Abstract**: In dialogue state tracking (DST), in-context learning comprises a retriever that selects labeled dialogues as in-context examples and a DST model that uses these examples to infer the dialogue state of the query dialogue. Existing methods for constructing training data for retrievers suffer from three key limitations: (1) the synergistic effect of examples is not considered, (2) the linguistic characteristics of the query are not sufficiently factored in, and (3) scoring is not directly optimized for DST performance. Consequently, the retriever can fail to retrieve examples that would substantially improve DST performance. To address these issues, we present CombiSearch, a method that scores effective in-context examples based on their combinatorial impact on DST performance. Our evaluation on MultiWOZ shows that retrievers trained with CombiSearch surpass state-of-the-art models, achieving a 20x gain in data efficiency and generalizing well to the SGD dataset. Moreover, CombiSearch attains a 12% absolute improvement in the upper bound DST performance over traditional approaches when no retrieval errors are assumed. This significantly increases the headroom for practical DST performance while demonstrating that existing methods rely on suboptimal data for retriever training. 

---
# A Topological Semantics of Dialogue: Nerve Structures and Logical Extraction 

**Authors**: Andreu Ballus Santacana  

**Link**: [PDF](https://arxiv.org/pdf/2506.00615)  

**Abstract**: We introduce a concise, topologically-motivated semantics for finite dialogues by mapping each utterance to an open set in a fixed semantic space, building the corresponding nerve complex of joint satisfiability, and extracting fundamental combinatorial invariants:
1. The negative nerve, which enumerates all finite collections of utterances whose
opens have empty intersection, providing a straightforward criterion for merging
separate transcripts without contradiction.
2. The global interpretation subspace, the unique minimal open in which all asserted
utterances hold simultaneously, enabling effective enumeration of all logical
consequences of the entire dialogue.
3. A practical demonstration in the Wolfram Language, with algorithms for constructing
nerves, detecting inconsistencies, and computing the global interpretation, thereby
illustrating computational feasibility.
Our framework is grounded in classical duality and topological semantics (Stone duality, Priestley duality, Tarski's semantics, coherence-space methods, Scott domains, topos semantics, and homotopy type theory) while drawing on recent advances in topological data analysis and dialogue-based semantics. 

---
# Predictability-Aware Compression and Decompression Framework for Multichannel Time Series Data 

**Authors**: Ziqi Liu, Pei Zeng, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.00614)  

**Abstract**: Real-world multichannel time series prediction faces growing demands for efficiency across edge and cloud environments, making channel compression a timely and essential problem. Motivated by success of Multiple-Input Multiple-Output (MIMO) methods, we propose a predictability-aware compression-decompression framework to reduce runtime, lower communication cost, and maintain prediction accuracy across diverse predictors. The core idea involves using a circular periodicity key matrix with orthogonality to capture underlying time series predictability during compression and to mitigate reconstruction errors during decompression by relaxing oversimplified data assumptions. Theoretical and empirical analyses show that the proposed framework is both time-efficient and scalable under a large number of channels. Extensive experiments on six datasets across various predictors demonstrate that the proposed method achieves superior overall performance by jointly considering prediction accuracy and runtime, while maintaining strong compatibility with diverse predictors. 

---
# Evaluating Robot Policies in a World Model 

**Authors**: Julian Quevedo, Percy Liang, Sherry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00613)  

**Abstract**: Robotics has broad applications from automating house chores to taking care of patients. However, evaluating robot control policies is challenging, as real-world testing is expensive, while handcrafted simulations often fail to accurately reflect real-world conditions, resulting in poor correlation between simulated evaluation and real-world outcomes. In this work, we investigate World-model-based Policy Evaluation (WPE). We first train an action-conditioned video generation model as a proxy to real-world environments. To enable efficient rollouts of hundreds of interactive steps while mitigating error accumulation in the world model, we propose an inference scheme which we call Blockwise-Autoregressive Diffusion Transformer with adjustable context and decoding horizon lengths. To ensure that the world model indeed follows action input, we propose metrics based on the agreement between the ground truth video and generated video conditioned on the same sequence of actions to evaluate the world model. We then use the world model for policy evaluation by performing Monte Carlo rollouts in the world model while employing a vision-language model (VLM) as a reward function. Interestingly, we found that WPE tends to underestimate the policy values for in-distribution actions and overestimate policy values for out-of-distribution actions. Nevertheless, WPE preserves the relative rankings of different policies. In emulating real robot executions, WPE achieves high fidelity in mimicing robot arm movements as in real videos, while emulating highly realistic object interaction remains challenging. Despite this limitation, we show that a world model can serve as a starting point for evaluating robot policies before real-world deployment. 

---
# Parallel Rescaling: Rebalancing Consistency Guidance for Personalized Diffusion Models 

**Authors**: JungWoo Chae, Jiyoon Kim, Sangheum Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00607)  

**Abstract**: Personalizing diffusion models to specific users or concepts remains challenging, particularly when only a few reference images are available. Existing methods such as DreamBooth and Textual Inversion often overfit to limited data, causing misalignment between generated images and text prompts when attempting to balance identity fidelity with prompt adherence. While Direct Consistency Optimization (DCO) with its consistency-guided sampling partially alleviates this issue, it still struggles with complex or stylized prompts. In this paper, we propose a parallel rescaling technique for personalized diffusion models. Our approach explicitly decomposes the consistency guidance signal into parallel and orthogonal components relative to classifier free guidance (CFG). By rescaling the parallel component, we minimize disruptive interference with CFG while preserving the subject's identity. Unlike prior personalization methods, our technique does not require additional training data or expensive annotations. Extensive experiments show improved prompt alignment and visual fidelity compared to baseline methods, even on challenging stylized prompts. These findings highlight the potential of parallel rescaled guidance to yield more stable and accurate personalization for diverse user inputs. 

---
# Graph Evidential Learning for Anomaly Detection 

**Authors**: Chunyu Wei, Wenji Hu, Xingjia Hao, Yunhai Wang, Yueguo Chen, Bing Bai, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00594)  

**Abstract**: Graph anomaly detection faces significant challenges due to the scarcity of reliable anomaly-labeled datasets, driving the development of unsupervised methods. Graph autoencoders (GAEs) have emerged as a dominant approach by reconstructing graph structures and node features while deriving anomaly scores from reconstruction errors. However, relying solely on reconstruction error for anomaly detection has limitations, as it increases the sensitivity to noise and overfitting. To address these issues, we propose Graph Evidential Learning (GEL), a probabilistic framework that redefines the reconstruction process through evidential learning. By modeling node features and graph topology using evidential distributions, GEL quantifies two types of uncertainty: graph uncertainty and reconstruction uncertainty, incorporating them into the anomaly scoring mechanism. Extensive experiments demonstrate that GEL achieves state-of-the-art performance while maintaining high robustness against noise and structural perturbations. 

---
# Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn 

**Authors**: Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2506.00592)  

**Abstract**: Plasticity, or the ability of an agent to adapt to new tasks, environments, or distributions, is crucial for continual learning. In this paper, we study the loss of plasticity in deep continual RL from the lens of churn: network output variability for out-of-batch data induced by mini-batch training. We demonstrate that (1) the loss of plasticity is accompanied by the exacerbation of churn due to the gradual rank decrease of the Neural Tangent Kernel (NTK) matrix; (2) reducing churn helps prevent rank collapse and adjusts the step size of regular RL gradients adaptively. Moreover, we introduce Continual Churn Approximated Reduction (C-CHAIN) and demonstrate it improves learning performance and outperforms baselines in a diverse range of continual learning environments on OpenAI Gym Control, ProcGen, DeepMind Control Suite, and MinAtar benchmarks. 

---
# Temporal Chunking Enhances Recognition of Implicit Sequential Patterns 

**Authors**: Jayanta Dey, Nicholas Soures, Miranda Gonzales, Itamar Lerner, Christopher Kanan, Dhireesha Kudithipudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00588)  

**Abstract**: In this pilot study, we propose a neuro-inspired approach that compresses temporal sequences into context-tagged chunks, where each tag represents a recurring structural unit or``community'' in the sequence. These tags are generated during an offline sleep phase and serve as compact references to past experience, allowing the learner to incorporate information beyond its immediate input range. We evaluate this idea in a controlled synthetic environment designed to reveal the limitations of traditional neural network based sequence learners, such as recurrent neural networks (RNNs), when facing temporal patterns on multiple timescales. We evaluate this idea in a controlled synthetic environment designed to reveal the limitations of traditional neural network based sequence learners, such as recurrent neural networks (RNNs), when facing temporal patterns on multiple timescales. Our results, while preliminary, suggest that temporal chunking can significantly enhance learning efficiency under resource constrained settings. A small-scale human pilot study using a Serial Reaction Time Task further motivates the idea of structural abstraction. Although limited to synthetic tasks, this work serves as an early proof-of-concept, with initial evidence that learned context tags can transfer across related task, offering potential for future applications in transfer learning. 

---
# ORAN-GUIDE: RAG-Driven Prompt Learning for LLM-Augmented Reinforcement Learning in O-RAN Network Slicing 

**Authors**: Fatemeh Lotfi, Hossein Rajoli, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2506.00576)  

**Abstract**: Advanced wireless networks must support highly dynamic and heterogeneous service demands. Open Radio Access Network (O-RAN) architecture enables this flexibility by adopting modular, disaggregated components, such as the RAN Intelligent Controller (RIC), Centralized Unit (CU), and Distributed Unit (DU), that can support intelligent control via machine learning (ML). While deep reinforcement learning (DRL) is a powerful tool for managing dynamic resource allocation and slicing, it often struggles to process raw, unstructured input like RF features, QoS metrics, and traffic trends. These limitations hinder policy generalization and decision efficiency in partially observable and evolving environments. To address this, we propose \textit{ORAN-GUIDE}, a dual-LLM framework that enhances multi-agent RL (MARL) with task-relevant, semantically enriched state representations. The architecture employs a domain-specific language model, ORANSight, pretrained on O-RAN control and configuration data, to generate structured, context-aware prompts. These prompts are fused with learnable tokens and passed to a frozen GPT-based encoder that outputs high-level semantic representations for DRL agents. This design adopts a retrieval-augmented generation (RAG) style pipeline tailored for technical decision-making in wireless systems. Experimental results show that ORAN-GUIDE improves sample efficiency, policy convergence, and performance generalization over standard MARL and single-LLM baselines. 

---
# Prompt-Tuned LLM-Augmented DRL for Dynamic O-RAN Network Slicing 

**Authors**: Fatemeh Lotfi, Hossein Rajoli, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2506.00574)  

**Abstract**: Modern wireless networks must adapt to dynamic conditions while efficiently managing diverse service demands. Traditional deep reinforcement learning (DRL) struggles in these environments, as scattered and evolving feedback makes optimal decision-making challenging. Large Language Models (LLMs) offer a solution by structuring unorganized network feedback into meaningful latent representations, helping RL agents recognize patterns more effectively. For example, in O-RAN slicing, concepts like SNR, power levels and throughput are semantically related, and LLMs can naturally cluster them, providing a more interpretable state representation. To leverage this capability, we introduce a contextualization-based adaptation method that integrates learnable prompts into an LLM-augmented DRL framework. Instead of relying on full model fine-tuning, we refine state representations through task-specific prompts that dynamically adjust to network conditions. Utilizing ORANSight, an LLM trained on O-RAN knowledge, we develop Prompt-Augmented Multi agent RL (PA-MRL) framework. Learnable prompts optimize both semantic clustering and RL objectives, allowing RL agents to achieve higher rewards in fewer iterations and adapt more efficiently. By incorporating prompt-augmented learning, our approach enables faster, more scalable, and adaptive resource allocation in O-RAN slicing. Experimental results show that it accelerates convergence and outperforms other baselines. 

---
# Understanding Behavioral Metric Learning: A Large-Scale Study on Distracting Reinforcement Learning Environments 

**Authors**: Ziyan Luo, Tianwei Ni, Pierre-Luc Bacon, Doina Precup, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2506.00563)  

**Abstract**: A key approach to state abstraction is approximating behavioral metrics (notably, bisimulation metrics) in the observation space and embedding these learned distances in the representation space. While promising for robustness to task-irrelevant noise, as shown in prior work, accurately estimating these metrics remains challenging, requiring various design choices that create gaps between theory and practice. Prior evaluations focus mainly on final returns, leaving the quality of learned metrics and the source of performance gains unclear. To systematically assess how metric learning works in deep reinforcement learning (RL), we evaluate five recent approaches, unified conceptually as isometric embeddings with varying design choices. We benchmark them with baselines across 20 state-based and 14 pixel-based tasks, spanning 370 task configurations with diverse noise settings. Beyond final returns, we introduce the evaluation of a denoising factor to quantify the encoder's ability to filter distractions. To further isolate the effect of metric learning, we propose and evaluate an isolated metric estimation setting, in which the encoder is influenced solely by the metric loss. Finally, we release an open-source, modular codebase to improve reproducibility and support future research on metric learning in deep RL. 

---
# MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning 

**Authors**: Peng Xia, Jinglu Wang, Yibo Peng, Kaide Zeng, Xian Wu, Xiangru Tang, Hongtu Zhu, Yun Li, Shujie Liu, Yan Lu, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00555)  

**Abstract**: Medical Large Vision-Language Models (Med-LVLMs) have shown strong potential in multimodal diagnostic tasks. However, existing single-agent models struggle to generalize across diverse medical specialties, limiting their performance. Recent efforts introduce multi-agent collaboration frameworks inspired by clinical workflows, where general practitioners (GPs) and specialists interact in a fixed sequence. Despite improvements, these static pipelines lack flexibility and adaptability in reasoning. To address this, we propose MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that enables dynamic, optimized collaboration among medical agents. Specifically, we train two GP agents based on Qwen2.5-VL via RL: the triage doctor learns to assign patients to appropriate specialties, while the attending physician integrates the judgments from multi-specialists and its own knowledge to make final decisions. To address the inconsistency in specialist outputs, we introduce a curriculum learning (CL)-guided RL strategy that progressively teaches the attending physician to balance between imitating specialists and correcting their mistakes. Experiments on five medical VQA benchmarks demonstrate that MMedAgent-RL not only outperforms both open-source and proprietary Med-LVLMs, but also exhibits human-like reasoning patterns. Notably, it achieves an average performance gain of 18.4% over supervised fine-tuning baselines. 

---
# AnnaAgent: Dynamic Evolution Agent System with Multi-Session Memory for Realistic Seeker Simulation 

**Authors**: Ming Wang, Peidong Wang, Lin Wu, Xiaocui Yang, Daling Wang, Shi Feng, Yuxin Chen, Bixuan Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00551)  

**Abstract**: Constrained by the cost and ethical concerns of involving real seekers in AI-driven mental health, researchers develop LLM-based conversational agents (CAs) with tailored configurations, such as profiles, symptoms, and scenarios, to simulate seekers. While these efforts advance AI in mental health, achieving more realistic seeker simulation remains hindered by two key challenges: dynamic evolution and multi-session memory. Seekers' mental states often fluctuate during counseling, which typically spans multiple sessions. To address this, we propose AnnaAgent, an emotional and cognitive dynamic agent system equipped with tertiary memory. AnnaAgent incorporates an emotion modulator and a complaint elicitor trained on real counseling dialogues, enabling dynamic control of the simulator's configurations. Additionally, its tertiary memory mechanism effectively integrates short-term and long-term memory across sessions. Evaluation results, both automated and manual, demonstrate that AnnaAgent achieves more realistic seeker simulation in psychological counseling compared to existing baselines. The ethically reviewed and screened code can be found on this https URL. 

---
# Towards Multi-dimensional Evaluation of LLM Summarization across Domains and Languages 

**Authors**: Hyangsuk Min, Yuho Lee, Minjeong Ban, Jiaqi Deng, Nicole Hee-Yeon Kim, Taewon Yun, Hang Su, Jason Cai, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.00549)  

**Abstract**: Evaluation frameworks for text summarization have evolved in terms of both domain coverage and metrics. However, existing benchmarks still lack domain-specific assessment criteria, remain predominantly English-centric, and face challenges with human annotation due to the complexity of reasoning. To address these, we introduce MSumBench, which provides a multi-dimensional, multi-domain evaluation of summarization in English and Chinese. It also incorporates specialized assessment criteria for each domain and leverages a multi-agent debate system to enhance annotation quality. By evaluating eight modern summarization models, we discover distinct performance patterns across domains and languages. We further examine large language models as summary evaluators, analyzing the correlation between their evaluation and summarization capabilities, and uncovering systematic bias in their assessment of self-generated summaries. Our benchmark dataset is publicly available at this https URL. 

---
# Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Self-Attention-based Deep Learning Approach 

**Authors**: Mehdi Bejani, Guillermo Perez-de-Arenaza-Pozo, Julián D. Arias-Londoño, Juan I. Godino-LLorente  

**Link**: [PDF](https://arxiv.org/pdf/2506.00545)  

**Abstract**: Missing data is a relevant issue in time series, especially in biomedical sequences such as those corresponding to smooth pursuit eye movements, which often contain gaps due to eye blinks and track losses, complicating the analysis and extraction of meaningful biomarkers. In this paper, a novel imputation framework is proposed using Self-Attention-based Imputation networks for time series, which leverages the power of deep learning and self-attention mechanisms to impute missing data. We further refine the imputed data using a custom made autoencoder, tailored to represent smooth pursuit eye movement sequences. The proposed approach was implemented using 5,504 sequences from 172 Parkinsonian patients and healthy controls. Results show a significant improvement in the accuracy of reconstructed eye movement sequences with respect to other state of the art techniques, substantially reducing the values for common time domain error metrics such as the mean absolute error, mean relative error, and root mean square error, while also preserving the signal's frequency domain characteristics. Moreover, it demonstrates robustness when large intervals of data are missing. This method offers an alternative solution for robustly handling missing data in time series, enhancing the reliability of smooth pursuit analysis for the screening and monitoring of neurodegenerative disorders. 

---
# Decoupling Reasoning and Knowledge Injection for In-Context Knowledge Editing 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yujia Zhou, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00536)  

**Abstract**: Knowledge editing aims to efficiently update Large Language Models (LLMs) by modifying specific knowledge without retraining the entire model. Among knowledge editing approaches, in-context editing (ICE) offers a lightweight solution by injecting new knowledge directly into the input context, leaving model parameters unchanged. However, existing ICE approaches do not explicitly separate the newly injected knowledge from the model's original reasoning process. This entanglement often results in conflicts between external updates and internal parametric knowledge, undermining the consistency and accuracy of the reasoning this http URL this work, we conduct preliminary experiments to examine how parametric knowledge influences reasoning path planning. We find that the model's reasoning is tightly coupled with its internal knowledge, and that naively injecting new information without adapting the reasoning path often leads to performance degradation, particularly in multi-hop tasks. To this end, we propose DecKER, a novel ICE framework that decouples reasoning from knowledge editing by generating a masked reasoning path and then resolving knowledge edits via hybrid retrieval and model-based validation. Experiments on multi-hop QA benchmarks show that DecKER significantly outperforms existing ICE methods by mitigating knowledge conflicts and preserving reasoning consistency. Our code is available at: this https URL . 

---
# The Security Threat of Compressed Projectors in Large Vision-Language Models 

**Authors**: Yudong Zhang, Ruobing Xie, Xingwu Sun, Jiansheng Chen, Zhanhui Kang, Di Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00534)  

**Abstract**: The choice of a suitable visual language projector (VLP) is critical to the successful training of large visual language models (LVLMs). Mainstream VLPs can be broadly categorized into compressed and uncompressed projectors, and each offering distinct advantages in performance and computational efficiency. However, their security implications have not been thoroughly examined. Our comprehensive evaluation reveals significant differences in their security profiles: compressed projectors exhibit substantial vulnerabilities, allowing adversaries to successfully compromise LVLMs even with minimal knowledge of structural information. In stark contrast, uncompressed projectors demonstrate robust security properties and do not introduce additional vulnerabilities. These findings provide critical guidance for researchers in selecting optimal VLPs that enhance the security and reliability of visual language models. The code will be released. 

---
# M2WLLM: Multi-Modal Multi-Task Ultra-Short-term Wind Power Prediction Algorithm Based on Large Language Model 

**Authors**: Hang Fana, Mingxuan Lib, Zuhan Zhanga, Long Chengc, Yujian Ye, Dunnan Liua  

**Link**: [PDF](https://arxiv.org/pdf/2506.00531)  

**Abstract**: The integration of wind energy into power grids necessitates accurate ultra-short-term wind power forecasting to ensure grid stability and optimize resource allocation. This study introduces M2WLLM, an innovative model that leverages the capabilities of Large Language Models (LLMs) for predicting wind power output at granular time intervals. M2WLLM overcomes the limitations of traditional and deep learning methods by seamlessly integrating textual information and temporal numerical data, significantly improving wind power forecasting accuracy through multi-modal data. Its architecture features a Prompt Embedder and a Data Embedder, enabling an effective fusion of textual prompts and numerical inputs within the LLMs framework. The Semantic Augmenter within the Data Embedder translates temporal data into a format that the LLMs can comprehend, enabling it to extract latent features and improve prediction accuracy. The empirical evaluations conducted on wind farm data from three Chinese provinces demonstrate that M2WLLM consistently outperforms existing methods, such as GPT4TS, across various datasets and prediction horizons. The results highlight LLMs' ability to enhance accuracy and robustness in ultra-short-term forecasting and showcase their strong few-shot learning capabilities. 

---
# Retrieval-Augmented Generation Systems for Intellectual Property via Synthetic Multi-Angle Fine-tuning 

**Authors**: Runtao Ren, Jian Ma, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00527)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems in the Intellectual Property (IP) field often struggle with diverse user queries, including colloquial expressions, spelling errors, and ambiguous terminology, leading to inaccurate retrieval and suboptimal responses. To address this challenge, we propose Multi-Angle Question Generation and Retrieval Fine-Tuning Method (MQG-RFM), a novel framework that leverages large language models (LLMs) to simulate varied user inquiries and fine-tunes retrieval models to align semantically equivalent but linguistically diverse questions. Unlike complex architectural modifications, MQG-RFM adopts a lightweight Data-to-Tune paradigm, combining prompt-engineered query generation with hard negative mining to enhance retrieval robustness without costly infrastructure changes. Experimental results on a Taiwan patent Q&A dataset show 185.62% improvement in retrieval accuracy on the Patent Consultation dataset and 262.26% improvement on the Novel Patent Technology Report dataset, with 14.22% and 53.58% improvements in generation quality over the baselines, respectively. By bridging the gap between user intent and system comprehension through semantic-aware retrieval optimization, MQG-RFM offers a practical, scalable approach for rapid, cost-effective deployment among small and medium-sized agencies seeking reliable patent intelligence solutions. Additionally, our proposed method has already been adopted by ScholarMate, the largest professional research social networking platform in China, to support real-world development and deployment. A demo version of the instantiated is available at this https URL. 

---
# CausalAbstain: Enhancing Multilingual LLMs with Causal Reasoning for Trustworthy Abstention 

**Authors**: Yuxi Sun, Aoqi Zuo, Wei Gao, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00519)  

**Abstract**: Large Language Models (LLMs) often exhibit knowledge disparities across languages. Encouraging LLMs to \textit{abstain} when faced with knowledge gaps is a promising strategy to reduce hallucinations in multilingual settings. Current abstention strategies for multilingual scenarios primarily rely on generating feedback in various languages using LLMs and performing self-reflection. However, these methods can be adversely impacted by inaccuracies and biases in the generated feedback. To address this, from a causal perspective, we introduce \textit{CausalAbstain}, a method that helps LLMs determine whether to utilize multiple generated feedback responses and how to identify the most useful ones. Extensive experiments demonstrate that \textit{CausalAbstain} effectively selects helpful feedback and enhances abstention decisions with interpretability in both native language (\textsc{Casual-native}) and multilingual (\textsc{Causal-multi}) settings, outperforming strong baselines on two benchmark datasets covering encyclopedic and commonsense knowledge QA tasks. Our code and data are open-sourced at this https URL. 

---
# Pro3D-Editor : A Progressive-Views Perspective for Consistent and Precise 3D Editing 

**Authors**: Yang Zheng, Mengqi Huang, Nan Chen, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00512)  

**Abstract**: Text-guided 3D editing aims to precisely edit semantically relevant local 3D regions, which has significant potential for various practical applications ranging from 3D games to film production. Existing methods typically follow a view-indiscriminate paradigm: editing 2D views indiscriminately and projecting them back into 3D space. However, they overlook the different cross-view interdependencies, resulting in inconsistent multi-view editing. In this study, we argue that ideal consistent 3D editing can be achieved through a \textit{progressive-views paradigm}, which propagates editing semantics from the editing-salient view to other editing-sparse views. Specifically, we propose \textit{Pro3D-Editor}, a novel framework, which mainly includes Primary-view Sampler, Key-view Render, and Full-view Refiner. Primary-view Sampler dynamically samples and edits the most editing-salient view as the primary view. Key-view Render accurately propagates editing semantics from the primary view to other key views through its Mixture-of-View-Experts Low-Rank Adaption (MoVE-LoRA). Full-view Refiner edits and refines the 3D object based on the edited multi-views. Extensive experiments demonstrate that our method outperforms existing methods in editing accuracy and spatial consistency. 

---
# Multi-Objective Neural Network Assisted Design Optimization of Soft Fin-Ray Grippers for Enhanced Grasping Performance 

**Authors**: Ali Ghanizadeh, Ali Ahmadi, Arash Bahrami  

**Link**: [PDF](https://arxiv.org/pdf/2506.00494)  

**Abstract**: Soft Fin-Ray grippers can perform delicate and careful manipulation, which has caused notable attention in different fields. These grippers can handle objects of various forms and sizes safely. The internal structure of the Fin-Ray finger plays a significant role in its adaptability and grasping performance. However, modeling the non-linear grasp force and deformation behaviors for design purposes is challenging. Moreover, when the Fin-Ray finger becomes more rigid and capable of exerting higher forces, it becomes less delicate in handling objects. The contrast between these two objectives gives rise to a multi-objective optimization problem. In this study, we employ finite element method (FEM) to estimate the deflections and contact forces of the Fin-Ray, grasping cylindrical objects. This dataset is then used to construct a multilayer perception (MLP) for prediction of the contact force and the tip displacement. The FEM dataset consists of three input and four target features. The three input features of the MLP and optimization design variables are the thickness of the front and supporting beams, the thickness of the cross beams, and the equal spacing between the cross beams. In addition, the target features are the maximum contact forces and maximum tip displacements in x- and y-directions. The magnitude of maximum contact force and magnitude of maximum tip displacement are the two objectives, showing the trade-off between force and delicate manipulation in soft Fin-Ray grippers. Furthermore, the optimized set of solutions are found using multi-objective optimal techniques. We use non-dominated sorting genetic algorithm (NSGA-II) method for this purpose. Our findings demonstrate that our methodologies can be used to improve the design and gripping performance of soft robotic grippers, helping us to choose a design not only for delicate grasping but also for high-force applications. 

---
# It Takes a Good Model to Train a Good Model: Generalized Gaussian Priors for Optimized LLMs 

**Authors**: Jun Wu, Yirong Xiong, Jiangtao Wen, Yuxing Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.00486)  

**Abstract**: Despite rapid advancements in the research and deployment of large language models (LLMs), the statistical distribution of model parameters, as well as their influence on initialization, training dynamics, and downstream efficiency, has received surprisingly little attention. A recent work introduced BackSlash, a training-time compression algorithm. It first demonstrated that pre-trained LLM parameters follow generalized Gaussian distributions (GGDs) better. By optimizing GG priors during training, BackSlash can reduce parameters by up to 90\% with minimal performance loss. Building on this foundational insight, we propose a unified, end-to-end framework for LLM optimization based on the GG model. Our contributions are threefold: (1) GG-based initialization scheme that aligns with the statistical structure of trained models, resulting in faster convergence and improved accuracy; (2) DeepShape, a post-training regularization method that reshapes weight distributions to match a GG profile, improving compressibility with minimized degradation in performance; and (3) RF8, a compact and hardware-efficient 8-bit floating-point format designed for GG-distributed-initialized BackSlash training, enabling low-cost inference without compromising accuracy. Experiments across diverse model architectures show that our framework consistently yields smaller and faster models that match or outperform standard training baselines. By grounding LLM development in principled statistical modeling, this work forges a new path toward efficient, scalable, and hardware-aware AI systems. The code is available on our project page: this https URL. 

---
# BenchHub: A Unified Benchmark Suite for Holistic and Customizable LLM Evaluation 

**Authors**: Eunsu Kim, Haneul Yoo, Guijin Son, Hitesh Patel, Amit Agarwal, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00482)  

**Abstract**: As large language models (LLMs) continue to advance, the need for up-to-date and well-organized benchmarks becomes increasingly critical. However, many existing datasets are scattered, difficult to manage, and make it challenging to perform evaluations tailored to specific needs or domains, despite the growing importance of domain-specific models in areas such as math or code. In this paper, we introduce BenchHub, a dynamic benchmark repository that empowers researchers and developers to evaluate LLMs more effectively. BenchHub aggregates and automatically classifies benchmark datasets from diverse domains, integrating 303K questions across 38 benchmarks. It is designed to support continuous updates and scalable data management, enabling flexible and customizable evaluation tailored to various domains or use cases. Through extensive experiments with various LLM families, we demonstrate that model performance varies significantly across domain-specific subsets, emphasizing the importance of domain-aware benchmarking. We believe BenchHub can encourage better dataset reuse, more transparent model comparisons, and easier identification of underrepresented areas in existing benchmarks, offering a critical infrastructure for advancing LLM evaluation research. 

---
# PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings 

**Authors**: Junseo Kim, Jongwook Han, Dongmin Choi, Jongwook Yoon, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00481)  

**Abstract**: Visual persuasion, which uses visual elements to influence cognition and behaviors, is crucial in fields such as advertising and political communication. With recent advancements in artificial intelligence, there is growing potential to develop persuasive systems that automatically generate persuasive images tailored to individuals. However, a significant bottleneck in this area is the lack of comprehensive datasets that connect the persuasiveness of images with the personal information about those who evaluated the images. To address this gap and facilitate technological advancements in personalized visual persuasion, we release the Personalized Visual Persuasion (PVP) dataset, comprising 28,454 persuasive images across 596 messages and 9 persuasion strategies. Importantly, the PVP dataset provides persuasiveness scores of images evaluated by 2,521 human annotators, along with their demographic and psychological characteristics (personality traits and values). We demonstrate the utility of our dataset by developing a persuasive image generator and an automated evaluator, and establish benchmark baselines. Our experiments reveal that incorporating psychological characteristics enhances the generation and evaluation of persuasive images, providing valuable insights for personalized visual persuasion. 

---
# SST: Self-training with Self-adaptive Thresholding for Semi-supervised Learning 

**Authors**: Shuai Zhao, Heyan Huang, Xinge Li, Xiaokang Chen, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00467)  

**Abstract**: Neural networks have demonstrated exceptional performance in supervised learning, benefiting from abundant high-quality annotated data. However, obtaining such data in real-world scenarios is costly and labor-intensive. Semi-supervised learning (SSL) offers a solution to this problem. Recent studies, such as Semi-ViT and Noisy Student, which employ consistency regularization or pseudo-labeling, have demonstrated significant achievements. However, they still face challenges, particularly in accurately selecting sufficient high-quality pseudo-labels due to their reliance on fixed thresholds. Recent methods such as FlexMatch and FreeMatch have introduced flexible or self-adaptive thresholding techniques, greatly advancing SSL research. Nonetheless, their process of updating thresholds at each iteration is deemed time-consuming, computationally intensive, and potentially unnecessary. To address these issues, we propose Self-training with Self-adaptive Thresholding (SST), a novel, effective, and efficient SSL framework. SST introduces an innovative Self-Adaptive Thresholding (SAT) mechanism that adaptively adjusts class-specific thresholds based on the model's learning progress. SAT ensures the selection of high-quality pseudo-labeled data, mitigating the risks of inaccurate pseudo-labels and confirmation bias. Extensive experiments demonstrate that SST achieves state-of-the-art performance with remarkable efficiency, generalization, and scalability across various architectures and datasets. Semi-SST-ViT-Huge achieves the best results on competitive ImageNet-1K SSL benchmarks, with 80.7% / 84.9% Top-1 accuracy using only 1% / 10% labeled data. Compared to the fully-supervised DeiT-III-ViT-Huge, which achieves 84.8% Top-1 accuracy using 100% labeled data, our method demonstrates superior performance using only 10% labeled data. 

---
# XMAD-Bench: Cross-Domain Multilingual Audio Deepfake Benchmark 

**Authors**: Ioan-Paul Ciobanu, Andrei-Iulian Hiji, Nicolae-Catalin Ristea, Paul Irofti, Cristian Rusu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00462)  

**Abstract**: Recent advances in audio generation led to an increasing number of deepfakes, making the general public more vulnerable to financial scams, identity theft, and misinformation. Audio deepfake detectors promise to alleviate this issue, with many recent studies reporting accuracy rates close to 99%. However, these methods are typically tested in an in-domain setup, where the deepfake samples from the training and test sets are produced by the same generative models. To this end, we introduce XMAD-Bench, a large-scale cross-domain multilingual audio deepfake benchmark comprising 668.8 hours of real and deepfake speech. In our novel dataset, the speakers, the generative methods, and the real audio sources are distinct across training and test splits. This leads to a challenging cross-domain evaluation setup, where audio deepfake detectors can be tested ``in the wild''. Our in-domain and cross-domain experiments indicate a clear disparity between the in-domain performance of deepfake detectors, which is usually as high as 100%, and the cross-domain performance of the same models, which is sometimes similar to random chance. Our benchmark highlights the need for the development of robust audio deepfake detectors, which maintain their generalization capacity across different languages, speakers, generative methods, and data sources. Our benchmark is publicly released at this https URL. 

---
# Comparing Traditional and Reinforcement-Learning Methods for Energy Storage Control 

**Authors**: Elinor Ginzburg, Itay Segev, Yoash Levron, Sarah Keren  

**Link**: [PDF](https://arxiv.org/pdf/2506.00459)  

**Abstract**: We aim to better understand the tradeoffs between traditional and reinforcement learning (RL) approaches for energy storage management. More specifically, we wish to better understand the performance loss incurred when using a generative RL policy instead of using a traditional approach to find optimal control policies for specific instances. Our comparison is based on a simplified micro-grid model, that includes a load component, a photovoltaic source, and a storage device. Based on this model, we examine three use cases of increasing complexity: ideal storage with convex cost functions, lossy storage devices, and lossy storage devices with convex transmission losses. With the aim of promoting the principled use RL based methods in this challenging and important domain, we provide a detailed formulation of each use case and a detailed description of the optimization challenges. We then compare the performance of traditional and RL methods, discuss settings in which it is beneficial to use each method, and suggest avenues for future investigation. 

---
# Reinforcement Learning for Hanabi 

**Authors**: Nina Cohen, Kordel K. France  

**Link**: [PDF](https://arxiv.org/pdf/2506.00458)  

**Abstract**: Hanabi has become a popular game for research when it comes to reinforcement learning (RL) as it is one of the few cooperative card games where you have incomplete knowledge of the entire environment, thus presenting a challenge for a RL agent. We explored different tabular and deep reinforcement learning algorithms to see which had the best performance both against an agent of the same type and also against other types of agents. We establish that certain agents played their highest scoring games against specific agents while others exhibited higher scores on average by adapting to the opposing agent's behavior. We attempted to quantify the conditions under which each algorithm provides the best advantage and identified the most interesting interactions between agents of different types. In the end, we found that temporal difference (TD) algorithms had better overall performance and balancing of play types compared to tabular agents. Specifically, tabular Expected SARSA and deep Q-Learning agents showed the best performance. 

---
# Diffusion Models for Increasing Accuracy in Olfaction Sensors and Datasets 

**Authors**: Kordel K. France, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00455)  

**Abstract**: Robotic odour source localization (OSL) is a critical capability for autonomous systems operating in complex environments. However, current OSL methods often suffer from ambiguities, particularly when robots misattribute odours to incorrect objects due to limitations in olfactory datasets and sensor resolutions. To address this challenge, we introduce a novel machine learning method using diffusion-based molecular generation to enhance odour localization accuracy that can be used by itself or with automated olfactory dataset construction pipelines with vision-language models (VLMs) This generative process of our diffusion model expands the chemical space beyond the limitations of both current olfactory datasets and the training data of VLMs, enabling the identification of potential odourant molecules not previously documented. The generated molecules can then be more accurately validated using advanced olfactory sensors which emulate human olfactory recognition through electronic sensor arrays. By integrating visual analysis, language processing, and molecular generation, our framework enhances the ability of olfaction-vision models on robots to accurately associate odours with their correct sources, thereby improving navigation and decision-making in environments where olfactory cues are essential. Our methodology represents a foundational advancement in the field of robotic olfaction, offering a scalable solution to the challenges posed by limited olfactory data and sensor ambiguities. 

---
# TMetaNet: Topological Meta-Learning Framework for Dynamic Link Prediction 

**Authors**: Hao Li, Hao Wan, Yuzhou Chen, Dongsheng Ye, Yulia Gel, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00453)  

**Abstract**: Dynamic graphs evolve continuously, presenting challenges for traditional graph learning due to their changing structures and temporal dependencies. Recent advancements have shown potential in addressing these challenges by developing suitable meta-learning-based dynamic graph neural network models. However, most meta-learning approaches for dynamic graphs rely on fixed weight update parameters, neglecting the essential intrinsic complex high-order topological information of dynamically evolving graphs. We have designed Dowker Zigzag Persistence (DZP), an efficient and stable dynamic graph persistent homology representation method based on Dowker complex and zigzag persistence, to capture the high-order features of dynamic graphs. Armed with the DZP ideas, we propose TMetaNet, a new meta-learning parameter update model based on dynamic topological features. By utilizing the distances between high-order topological features, TMetaNet enables more effective adaptation across snapshots. Experiments on real-world datasets demonstrate TMetaNet's state-of-the-art performance and resilience to graph noise, illustrating its high potential for meta-learning and dynamic graph analysis. Our code is available at this https URL. 

---
# Attention-Aided MMSE for OFDM Channel Estimation: Learning Linear Filters with Attention 

**Authors**: TaeJun Ha, Chaehyun Jung, Hyeonuk Kim, Jeongwoo Park, Jeonghun Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.00452)  

**Abstract**: In orthogonal frequency division multiplexing (OFDM), accurate channel estimation is crucial. Classical signal processing based approaches, such as minimum mean-squared error (MMSE) estimation, often require second-order statistics that are difficult to obtain in practice. Recent deep neural networks based methods have been introduced to address this; yet they often suffer from high complexity. This paper proposes an Attention-aided MMSE (A-MMSE), a novel model-based DNN framework that learns the optimal MMSE filter via the Attention Transformer. Once trained, the A-MMSE estimates the channel through a single linear operation for channel estimation, eliminating nonlinear activations during inference and thus reducing computational complexity. To enhance the learning efficiency of the A-MMSE, we develop a two-stage Attention encoder, designed to effectively capture the channel correlation structure. Additionally, a rank-adaptive extension of the proposed A-MMSE allows flexible trade-offs between complexity and channel estimation accuracy. Extensive simulations with 3GPP TDL channel models demonstrate that the proposed A-MMSE consistently outperforms other baseline methods in terms of normalized MSE across a wide range of SNR conditions. In particular, the A-MMSE and its rank-adaptive extension establish a new frontier in the performance complexity trade-off, redefining the standard for practical channel estimation methods. 

---
# RLAE: Reinforcement Learning-Assisted Ensemble for LLMs 

**Authors**: Yuqian Fu, Yuanheng Zhu, Jiajun Chai, Guojun Yin, Wei Lin, Qichao Zhang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00439)  

**Abstract**: Ensembling large language models (LLMs) can effectively combine diverse strengths of different models, offering a promising approach to enhance performance across various tasks. However, existing methods typically rely on fixed weighting strategies that fail to adapt to the dynamic, context-dependent characteristics of LLM capabilities. In this work, we propose Reinforcement Learning-Assisted Ensemble for LLMs (RLAE), a novel framework that reformulates LLM ensemble through the lens of a Markov Decision Process (MDP). Our approach introduces a RL agent that dynamically adjusts ensemble weights by considering both input context and intermediate generation states, with the agent being trained using rewards that directly correspond to the quality of final outputs. We implement RLAE using both single-agent and multi-agent reinforcement learning algorithms ($\text{RLAE}_\text{PPO}$ and $\text{RLAE}_\text{MAPPO}$ ), demonstrating substantial improvements over conventional ensemble methods. Extensive evaluations on a diverse set of tasks show that RLAE outperforms existing approaches by up to $3.3\%$ accuracy points, offering a more effective framework for LLM ensembling. Furthermore, our method exhibits superior generalization capabilities across different tasks without the need for retraining, while simultaneously achieving lower time latency. 

---
# Is Your Explanation Reliable: Confidence-Aware Explanation on Graph Neural Networks 

**Authors**: Jiaxing Zhang, Xiaoou Liu, Dongsheng Luo, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00437)  

**Abstract**: Explaining Graph Neural Networks (GNNs) has garnered significant attention due to the need for interpretability, enabling users to understand the behavior of these black-box models better and extract valuable insights from their predictions. While numerous post-hoc instance-level explanation methods have been proposed to interpret GNN predictions, the reliability of these explanations remains uncertain, particularly in the out-of-distribution or unknown test datasets. In this paper, we address this challenge by introducing an explainer framework with the confidence scoring module ( ConfExplainer), grounded in theoretical principle, which is generalized graph information bottleneck with confidence constraint (GIB-CC), that quantifies the reliability of generated explanations. Experimental results demonstrate the superiority of our approach, highlighting the effectiveness of the confidence score in enhancing the trustworthiness and robustness of GNN explanations. 

---
# Learning from Double Positive and Unlabeled Data for Potential-Customer Identification 

**Authors**: Masahiro Kato, Yuki Ikeda abd Kentaro Baba, Takashi Imai, Ryo Inokuchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00436)  

**Abstract**: In this study, we propose a method for identifying potential customers in targeted marketing by applying learning from positive and unlabeled data (PU learning). We consider a scenario in which a company sells a product and can observe only the customers who purchased it. Decision-makers seek to market products effectively based on whether people have loyalty to the company. Individuals with loyalty are those who are likely to remain interested in the company even without additional advertising. Consequently, those loyal customers would likely purchase from the company if they are interested in the product. In contrast, people with lower loyalty may overlook the product or buy similar products from other companies unless they receive marketing attention. Therefore, by focusing marketing efforts on individuals who are interested in the product but do not have strong loyalty, we can achieve more efficient marketing. To achieve this goal, we consider how to learn, from limited data, a classifier that identifies potential customers who (i) have interest in the product and (ii) do not have loyalty to the company. Although our algorithm comprises a single-stage optimization, its objective function implicitly contains two losses derived from standard PU learning settings. For this reason, we refer to our approach as double PU learning. We verify the validity of the proposed algorithm through numerical experiments, confirming that it functions appropriately for the problem at hand. 

---
# Channel Normalization for Time Series Channel Identification 

**Authors**: Seunghan Lee, Taeyoung Park, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00432)  

**Abstract**: Channel identifiability (CID) refers to the ability to distinguish between individual channels in time series (TS) modeling. The absence of CID often results in producing identical outputs for identical inputs, disregarding channel-specific characteristics. In this paper, we highlight the importance of CID and propose Channel Normalization (CN), a simple yet effective normalization strategy that enhances CID by assigning distinct affine transformation parameters to each channel. We further extend CN in two ways: 1) Adaptive CN (ACN) dynamically adjusts parameters based on the input TS, improving adaptability in TS models, and 2) Prototypical CN (PCN) introduces a set of learnable prototypes instead of per-channel parameters, enabling applicability to datasets with unknown or varying number of channels and facilitating use in TS foundation models. We demonstrate the effectiveness of CN and its variants by applying them to various TS models, achieving significant performance gains for both non-CID and CID models. In addition, we analyze the success of our approach from an information theory perspective. Code is available at this https URL. 

---
# COGNATE: Acceleration of Sparse Tensor Programs on Emerging Hardware using Transfer Learning 

**Authors**: Chamika Sudusinghe, Gerasimos Gerogiannis Damitha Lenadora, Charles Block, Josep Torrellas, Charith Mendis  

**Link**: [PDF](https://arxiv.org/pdf/2506.00424)  

**Abstract**: Sparse tensor programs are essential in deep learning and graph analytics, driving the need for optimized processing. To meet this demand, specialized hardware accelerators are being developed. Optimizing these programs for accelerators is challenging for two reasons: program performance is highly sensitive to variations in sparse inputs, and early-stage accelerators rely on expensive simulators. Therefore, ML-based cost models used for optimizing such programs on general-purpose hardware are often ineffective for early-stage accelerators, as they require large datasets for proper training. To this end, we introduce COGNATE, a novel framework that leverages inexpensive data samples from general-purpose hardware (e.g., CPUs) to train cost models, followed by few-shot fine-tuning on emerging hardware. COGNATE exploits the homogeneity of input features across hardware platforms while effectively mitigating heterogeneity, enabling cost model training with just 5% of the data samples needed by accelerator-specific models to achieve comparable performance. We conduct extensive experiments to demonstrate that COGNATE outperforms existing techniques, achieving average speedups of 1.47x (up to 5.46x) for SpMM and 1.39x (up to 4.22x) for SDDMM. 

---
# Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System for Dynamic Interactions 

**Authors**: Jihyoung Jang, Minwook Bae, Minji Kim, Dilek Hakkani-Tur, Hyounghun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.00421)  

**Abstract**: As chatbots continue to evolve toward human-like, real-world, interactions, multimodality remains an active area of research and exploration. So far, efforts to integrate multimodality into chatbots have primarily focused on image-centric tasks, such as visual dialogue and image-based instructions, placing emphasis on the "eyes" of human perception while neglecting the "ears", namely auditory aspects. Moreover, these studies often center around static interactions that focus on discussing the modality rather than naturally incorporating it into the conversation, which limits the richness of simultaneous, dynamic engagement. Furthermore, while multimodality has been explored in multi-party and multi-session conversations, task-specific constraints have hindered its seamless integration into dynamic, natural conversations. To address these challenges, this study aims to equip chatbots with "eyes and ears" capable of more immersive interactions with humans. As part of this effort, we introduce a new multimodal conversation dataset, Multimodal Multi-Session Multi-Party Conversation ($M^3C$), and propose a novel multimodal conversation model featuring multimodal memory retrieval. Our model, trained on the $M^3C$, demonstrates the ability to seamlessly engage in long-term conversations with multiple speakers in complex, real-world-like settings, effectively processing visual and auditory inputs to understand and respond appropriately. Human evaluations highlight the model's strong performance in maintaining coherent and dynamic interactions, demonstrating its potential for advanced multimodal conversational agents. 

---
# A New Spatiotemporal Correlation Anomaly Detection Method that Integrates Contrastive Learning and Few-Shot Learning in Wireless Sensor Networks 

**Authors**: Miao Ye, Suxiao Wang, Jiaguang Han, Yong Wang, Xiaoli Wang, Jingxuan Wei, Peng Wen, Jing Cui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00420)  

**Abstract**: Detecting anomalies in the data collected by WSNs can provide crucial evidence for assessing the reliability and stability of WSNs. Existing methods for WSN anomaly detection often face challenges such as the limited extraction of spatiotemporal correlation features, the absence of sample labels, few anomaly samples, and an imbalanced sample distribution. To address these issues, a spatiotemporal correlation detection model (MTAD-RD) considering both model architecture and a two-stage training strategy perspective is proposed. In terms of model structure design, the proposed MTAD-RD backbone network includes a retentive network (RetNet) enhanced by a cross-retention (CR) module, a multigranular feature fusion module, and a graph attention network module to extract internode correlation information. This proposed model can integrate the intermodal correlation features and spatial features of WSN neighbor nodes while extracting global information from time series data. Moreover, its serialized inference characteristic can remarkably reduce inference overhead. For model training, a two-stage training approach was designed. First, a contrastive learning proxy task was designed for time series data with graph structure information in WSNs, enabling the backbone network to learn transferable features from unlabeled data using unsupervised contrastive learning methods, thereby addressing the issue of missing sample labels in the dataset. Then, a caching-based sample sampler was designed to divide samples into few-shot and contrastive learning data. A specific joint loss function was developed to jointly train the dual-graph discriminator network to address the problem of sample imbalance effectively. In experiments carried out on real public datasets, the designed MTAD-RD anomaly detection method achieved an F1 score of 90.97%, outperforming existing supervised WSN anomaly detection methods. 

---
# Dual Debiasing for Noisy In-Context Learning for Text Generation 

**Authors**: Siqi Liang, Sumyeong Ahn, Paramveer S. Dhillon, Jiayu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00418)  

**Abstract**: In context learning (ICL) relies heavily on high quality demonstrations drawn from large annotated corpora. Existing approaches detect noisy annotations by ranking local perplexities, presuming that noisy samples yield higher perplexities than their clean counterparts. However, this assumption breaks down when the noise ratio is high and many demonstrations are flawed. We reexamine the perplexity based paradigm for text generation under noisy annotations, highlighting two sources of bias in perplexity: the annotation itself and the domain specific knowledge inherent in large language models (LLMs). To overcome these biases, we introduce a dual debiasing framework that uses synthesized neighbors to explicitly correct perplexity estimates, yielding a robust Sample Cleanliness Score. This metric uncovers absolute sample cleanliness regardless of the overall corpus noise level. Extensive experiments demonstrate our method's superior noise detection capabilities and show that its final ICL performance is comparable to that of a fully clean demonstration corpus. Moreover, our approach remains robust even when noise ratios are extremely high. 

---
# Wide Reflective Equilibrium in LLM Alignment: Bridging Moral Epistemology and AI Safety 

**Authors**: Matthew Brophy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00415)  

**Abstract**: As large language models (LLMs) become more powerful and pervasive across society, ensuring these systems are beneficial, safe, and aligned with human values is crucial. Current alignment techniques, like Constitutional AI (CAI), involve complex iterative processes. This paper argues that the Method of Wide Reflective Equilibrium (MWRE) -- a well-established coherentist moral methodology -- offers a uniquely apt framework for understanding current LLM alignment efforts. Moreover, this methodology can substantively augment these processes by providing concrete pathways for improving their dynamic revisability, procedural legitimacy, and overall ethical grounding. Together, these enhancements can help produce more robust and ethically defensible outcomes. MWRE, emphasizing the achievement of coherence between our considered moral judgments, guiding moral principles, and relevant background theories, arguably better represents the intricate reality of LLM alignment and offers a more robust path to justification than prevailing foundationalist models or simplistic input-output evaluations. While current methods like CAI bear a structural resemblance to MWRE, they often lack its crucial emphasis on dynamic, bi-directional revision of principles and the procedural legitimacy derived from such a process. While acknowledging various disanalogies (e.g., consciousness, genuine understanding in LLMs), the paper demonstrates that MWRE serves as a valuable heuristic for critically analyzing current alignment efforts and for guiding the future development of more ethically sound and justifiably aligned AI systems. 

---
# Accelerating Diffusion LLMs via Adaptive Parallel Decoding 

**Authors**: Daniel Israel, Guy Van den Broeck, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2506.00413)  

**Abstract**: The generation speed of LLMs are bottlenecked by autoregressive decoding, where tokens are predicted sequentially one by one. Alternatively, diffusion large language models (dLLMs) theoretically allow for parallel token generation, but in practice struggle to achieve the speed of autoregressive models without significantly sacrificing quality. We therefore introduce adaptive parallel decoding (APD), a novel method that dynamically adjusts the number of tokens sampled in parallel. We achieve this by defining a multiplicative mixture between the dLLM marginal probabilities and the joint probability of sequences under a small auxiliary autoregressive model. This inverts the standard setup of speculative decoding, where the goal is to sample from a large autoregressive verifier by drafting from a smaller model. We further optimize APD by enabling KV caching and limiting the size of the masked input. Altogether, our method puts forward three tunable parameters to flexibly tradeoff throughput and quality. We show that APD provides markedly higher throughput with minimal quality degradations on downstream benchmarks. 

---
# LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks 

**Authors**: Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00411)  

**Abstract**: Real-world embodied agents face long-horizon tasks, characterized by high-level goals demanding multi-step solutions beyond single actions. Successfully navigating these requires both high-level task planning (i.e., decomposing goals into sub-tasks) and low-level motion control (i.e., generating precise robot actions). While existing vision language action (VLA) models and hierarchical architectures offer potential in embodied tasks, the former often falter in planning, and the latter can suffer from coordination issues, both hampering performance. We introduce a new unified VLA framework for long-horizon tasks, dubbed LoHoVLA, to overcome these limitations. LoHoVLA leverages a large pretrained vision language model (VLM) as the backbone to jointly generate language and action tokens for sub-task generation and robot action prediction, respectively. This shared representation promotes better generalization across tasks. Additionally, LoHoVLA embraces a hierarchical closed-loop control mechanism to mitigate errors originating from both high-level planning and low-level control. To train LoHoVLA, we introduce LoHoSet, a dataset built on the Ravens simulator, containing 20 long-horizon tasks, each with 1,000 expert demonstrations composed of visual observations, linguistic goals, sub-tasks, and robot actions. Experimental results show that LoHoVLA significantly surpasses both hierarchical and standard VLA approaches on long-horizon embodied tasks in the Ravens simulator. These findings underscore the promise of unified architectures for advancing generalizable embodied intelligence. 

---
# Bias as a Virtue: Rethinking Generalization under Distribution Shifts 

**Authors**: Ruixuan Chen, Wentao Li, Jiahui Xiao, Yuchen Li, Yimin Tang, Xiaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00407)  

**Abstract**: Machine learning models often degrade when deployed on data distributions different from their training data. Challenging conventional validation paradigms, we demonstrate that higher in-distribution (ID) bias can lead to better out-of-distribution (OOD) generalization. Our Adaptive Distribution Bridge (ADB) framework implements this insight by introducing controlled statistical diversity during training, enabling models to develop bias profiles that effectively generalize across distributions. Empirically, we observe a robust negative correlation where higher ID bias corresponds to lower OOD error--a finding that contradicts standard practices focused on minimizing validation error. Evaluation on multiple datasets shows our approach significantly improves OOD generalization. ADB achieves robust mean error reductions of up to 26.8% compared to traditional cross-validation, and consistently identifies high-performing training strategies, evidenced by percentile ranks often exceeding 74.4%. Our work provides both a practical method for improving generalization and a theoretical framework for reconsidering the role of bias in robust machine learning. 

---
# Scaling Textual Gradients via Sampling-Based Momentum 

**Authors**: Zixin Ding, Junyuan Hong, Jiachen T. Wang, Zinan Lin, Zhangyang Wang, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00400)  

**Abstract**: As prompts play an increasingly critical role in large language models (LLMs), optimizing textual prompts has become a crucial challenge. The Textual Gradient Descent (TGD) framework has emerged as a promising data-driven approach that iteratively refines textual prompts using LLM - suggested updates (or textual gradients) over minibatches of training samples. In this paper, we empirically demonstrate that scaling the number of training examples initially improves but later degrades TGD's performance across multiple downstream NLP tasks. However, while data scaling improves results for most tasks, it also significantly increases the computational cost when leveraging LLMs. To address this, we draw inspiration from numerical gradient descent and propose Textual Stochastic Gradient Descent with Momentum (TSGD-M) - a method that facilitates scalable in-context learning by reweighting prompt sampling based on past batch distributions. Across nine NLP tasks spanning three domains - including BIG-Bench Hard (BBH), natural language understanding tasks, and reasoning tasks - TSGD-M significantly outperforms TGD baselines that do not incorporate reweighted sampling, while also reducing variance in most tasks. 

---
# MagiCodec: Simple Masked Gaussian-Injected Codec for High-Fidelity Reconstruction and Generation 

**Authors**: Yakun Song, Jiawei Chen, Xiaobin Zhuang, Chenpeng Du, Ziyang Ma, Jian Wu, Jian Cong, Dongya Jia, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00385)  

**Abstract**: Neural audio codecs have made significant strides in efficiently mapping raw audio waveforms into discrete token representations, which are foundational for contemporary audio generative models. However, most existing codecs are optimized primarily for reconstruction quality, often at the expense of the downstream modelability of the encoded tokens. Motivated by the need to overcome this bottleneck, we introduce $\textbf{MagiCodec}$, a novel single-layer, streaming Transformer-based audio codec. MagiCodec is designed with a multistage training pipeline that incorporates Gaussian noise injection and latent regularization, explicitly targeting the enhancement of semantic expressiveness in the generated codes while preserving high reconstruction fidelity. We analytically derive the effect of noise injection in the frequency domain, demonstrating its efficacy in attenuating high-frequency components and fostering robust tokenization. Extensive experimental evaluations show that MagiCodec surpasses state-of-the-art codecs in both reconstruction quality and downstream tasks. Notably, the tokens produced by MagiCodec exhibit Zipf-like distributions, as observed in natural languages, thereby improving compatibility with language-model-based generative architectures. The code and pre-trained models are available at this https URL. 

---
# Neural Network-based Information-Theoretic Transceivers for High-Order Modulation Schemes 

**Authors**: Ngoc Long Pham, Tri Nhu Do  

**Link**: [PDF](https://arxiv.org/pdf/2506.00368)  

**Abstract**: Neural network (NN)-based end-to-end (E2E) communication systems, in which each system component may consist of a portion of a neural network, have been investigated as potential tools for developing artificial intelligence (Al)-native E2E systems. In this paper, we propose an NN-based bitwise receiver that improves computational efficiency while maintaining performance comparable to baseline demappers. Building on this foundation, we introduce a novel symbol-wise autoencoder (AE)-based E2E system that jointly optimizes the transmitter and receiver at the physical layer. We evaluate the proposed NN-based receiver using bit-error rate (BER) analysis to confirm that the numerical BER achieved by NN-based receivers or transceivers is accurate. Results demonstrate that the AE-based system outperforms baseline architectures, particularly for higher-order modulation schemes. We further show that the training signal-to-noise ratio (SNR) significantly affects the performance of the systems when inference is conducted at different SNR levels. 

---
# $\texttt{AVROBUSTBENCH}$: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time 

**Authors**: Sarthak Kumar Maharana, Saksham Singh Kushwaha, Baoming Zhang, Adrian Rodriguez, Songtao Wei, Yapeng Tian, Yunhui Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00358)  

**Abstract**: While recent audio-visual models have demonstrated impressive performance, their robustness to distributional shifts at test-time remains not fully understood. Existing robustness benchmarks mainly focus on single modalities, making them insufficient for thoroughly assessing the robustness of audio-visual models. Motivated by real-world scenarios where shifts can occur $\textit{simultaneously}$ in both audio and visual modalities, we introduce $\texttt{AVROBUSTBENCH}$, a comprehensive benchmark designed to evaluate the test-time robustness of audio-visual recognition models. $\texttt{AVROBUSTBENCH}$ comprises four audio-visual benchmark datasets, $\texttt{AUDIOSET-2C}$, $\texttt{VGGSOUND-2C}$, $\texttt{KINETICS-2C}$, and $\texttt{EPICKITCHENS-2C}$, each incorporating 75 bimodal audio-visual corruptions that are $\textit{co-occurring}$ and $\textit{correlated}$. Through extensive evaluations, we observe that state-of-the-art supervised and self-supervised audio-visual models exhibit declining robustness as corruption severity increases. Furthermore, online test-time adaptation (TTA) methods, on $\texttt{VGGSOUND-2C}$ and $\texttt{KINETICS-2C}$, offer minimal improvements in performance under bimodal corruptions. We further propose $\texttt{AV2C}$, a simple TTA approach enabling on-the-fly cross-modal fusion by penalizing high-entropy samples, which achieves improvements on $\texttt{VGGSOUND-2C}$. We hope that $\texttt{AVROBUSTBENCH}$ will steer the development of more effective and robust audio-visual TTA approaches. Our code is available $\href{this https URL}{here}$. 

---
# Exploring the Performance of Perforated Backpropagation through Further Experiments 

**Authors**: Rorry Brenner, Evan Davis, Rushi Chaudhari, Rowan Morse, Jingyao Chen, Xirui Liu, Zhaoyi You, Laurent Itti  

**Link**: [PDF](https://arxiv.org/pdf/2506.00356)  

**Abstract**: Perforated Backpropagation is a neural network optimization technique based on modern understanding of the computational importance of dendrites within biological neurons. This paper explores further experiments from the original publication, generated from a hackathon held at the Carnegie Mellon Swartz Center in February 2025. Students and local Pittsburgh ML practitioners were brought together to experiment with the Perforated Backpropagation algorithm on the datasets and models which they were using for their projects. Results showed that the system could enhance their projects, with up to 90% model compression without negative impact on accuracy, or up to 16% increased accuracy of their original models. 

---
# Enabling Secure and Ephemeral AI Workloads in Data Mesh Environments 

**Authors**: Chinkit Patel, Kee Siong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00352)  

**Abstract**: Many large enterprises that operate highly governed and complex ICT environments have no efficient and effective way to support their Data and AI teams in rapidly spinning up and tearing down self-service data and compute infrastructure, to experiment with new data analytic tools, and deploy data products into operational use. This paper proposes a key piece of the solution to the overall problem, in the form of an on-demand self-service data-platform infrastructure to empower de-centralised data teams to build data products on top of centralised templates, policies and governance. The core innovation is an efficient method to leverage immutable container operating systems and infrastructure-as-code methodologies for creating, from scratch, vendor-neutral and short-lived Kubernetes clusters on-premises and in any cloud environment. Our proposed approach can serve as a repeatable, portable and cost-efficient alternative or complement to commercial Platform-as-a-Service (PaaS) offerings, and this is particularly important in supporting interoperability in complex data mesh environments with a mix of modern and legacy compute infrastructure. 

---
# Beyond Winning: Margin of Victory Relative to Expectation Unlocks Accurate Skill Ratings 

**Authors**: Shivam Shorewala, Zihao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00348)  

**Abstract**: Knowledge of accurate relative skills in any competitive system is essential, but foundational approaches such as ELO discard extremely relevant performance data by concentrating exclusively on binary outcomes. While margin of victory (MOV) extensions exist, they often lack a definitive method for incorporating this information. We introduce Margin of Victory Differential Analysis (MOVDA), a framework that enhances traditional rating systems by using the deviation between the true MOV and a $\textit{modeled expectation}$. MOVDA learns a domain-specific, non-linear function (a scaled hyperbolic tangent that captures saturation effects and home advantage) to predict expected MOV based on rating differentials. Crucially, the $\textit{difference}$ between the true and expected MOV provides a subtle and weighted signal for rating updates, highlighting informative deviations in all levels of contests. Extensive experiments on professional NBA basketball data (from 2013 to 2023, with 13,619 games) show that MOVDA significantly outperforms standard ELO and Bayesian baselines. MOVDA reduces Brier score prediction error by $1.54\%$ compared to TrueSkill, increases outcome accuracy by $0.58\%$, and most importantly accelerates rating convergence by $13.5\%$, while maintaining the computational efficiency of the original ELO updates. MOVDA offers a theoretically motivated, empirically superior, and computationally lean approach to integrating performance magnitude into skill rating for competitive environments like the NBA. 

---
# Efficient Latent Semantic Clustering for Scaling Test-Time Computation of LLMs 

**Authors**: Sungjae Lee, Hoyoung Kim, Jeongyeon Hwang, Eunhyeok Park, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2506.00344)  

**Abstract**: Scaling test-time computation--generating and analyzing multiple or sequential outputs for a single input--has become a promising strategy for improving the reliability and quality of large language models (LLMs), as evidenced by advances in uncertainty quantification and multi-step reasoning. A key shared component is semantic clustering, which groups outputs that differ in form but convey the same meaning. Semantic clustering enables estimation of the distribution over the semantics of outputs and helps avoid redundant exploration of reasoning paths. However, existing approaches typically rely on external models, which introduce substantial computational overhead and often fail to capture context-aware semantics. We propose Latent Semantic Clustering (LSC), a lightweight and context-sensitive method that leverages the generator LLM's internal hidden states for clustering, eliminating the need for external models. Our extensive experiment across various LLMs and datasets shows that LSC significantly improves the computational efficiency of test-time scaling while maintaining or exceeding the performance of existing methods. 

---
# Recover Experimental Data with Selection Bias using Counterfactual Logic 

**Authors**: Jingyang He, Shuai Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00335)  

**Abstract**: Selection bias, arising from the systematic inclusion or exclusion of certain samples, poses a significant challenge to the validity of causal inference. While Bareinboim et al. introduced methods for recovering unbiased observational and interventional distributions from biased data using partial external information, the complexity of the backdoor adjustment and the method's strong reliance on observational data limit its applicability in many practical settings. In this paper, we formally discover the recoverability of $P(Y^*_{x^*})$ under selection bias with experimental data. By explicitly constructing counterfactual worlds via Structural Causal Models (SCMs), we analyze how selection mechanisms in the observational world propagate to the counterfactual domain. We derive a complete set of graphical and theoretical criteria to determine that the experimental distribution remain unaffected by selection bias. Furthermore, we propose principled methods for leveraging partially unbiased observational data to recover $P(Y^*_{x^*})$ from biased experimental datasets. Simulation studies replicating realistic research scenarios demonstrate the practical utility of our approach, offering concrete guidance for mitigating selection bias in applied causal inference. 

---
# Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation 

**Authors**: Muhammad Adnan, Nithesh Kurella, Akhil Arunkumar, Prashant J. Nair  

**Link**: [PDF](https://arxiv.org/pdf/2506.00329)  

**Abstract**: Diffusion Transformers (DiTs) achieve state-of-the-art results in text-to-image, text-to-video generation, and editing. However, their large model size and the quadratic cost of spatial-temporal attention over multiple denoising steps make video generation computationally expensive. Static caching mitigates this by reusing features across fixed steps but fails to adapt to generation dynamics, leading to suboptimal trade-offs between speed and quality.
We propose Foresight, an adaptive layer-reuse technique that reduces computational redundancy across denoising steps while preserving baseline performance. Foresight dynamically identifies and reuses DiT block outputs for all layers across steps, adapting to generation parameters such as resolution and denoising schedules to optimize efficiency. Applied to OpenSora, Latte, and CogVideoX, Foresight achieves up to 1.63x end-to-end speedup, while maintaining video quality. The source code of Foresight is available at \texttt{this https URL}. 

---
# Latent Guidance in Diffusion Models for Perceptual Evaluations 

**Authors**: Shreshth Saini, Ru-Ling Liao, Yan Ye, Alan C. Bovik  

**Link**: [PDF](https://arxiv.org/pdf/2506.00327)  

**Abstract**: Despite recent advancements in latent diffusion models that generate high-dimensional image data and perform various downstream tasks, there has been little exploration into perceptual consistency within these models on the task of No-Reference Image Quality Assessment (NR-IQA). In this paper, we hypothesize that latent diffusion models implicitly exhibit perceptually consistent local regions within the data manifold. We leverage this insight to guide on-manifold sampling using perceptual features and input measurements. Specifically, we propose Perceptual Manifold Guidance (PMG), an algorithm that utilizes pretrained latent diffusion models and perceptual quality features to obtain perceptually consistent multi-scale and multi-timestep feature maps from the denoising U-Net. We empirically demonstrate that these hyperfeatures exhibit high correlation with human perception in IQA tasks. Our method can be applied to any existing pretrained latent diffusion model and is straightforward to integrate. To the best of our knowledge, this paper is the first work on guiding diffusion model with perceptual features for NR-IQA. Extensive experiments on IQA datasets show that our method, LGDM, achieves state-of-the-art performance, underscoring the superior generalization capabilities of diffusion models for NR-IQA tasks. 

---
# dpmm: Differentially Private Marginal Models, a Library for Synthetic Tabular Data Generation 

**Authors**: Sofiane Mahiou, Amir Dizche, Reza Nazari, Xinmin Wu, Ralph Abbey, Jorge Silva, Georgi Ganev  

**Link**: [PDF](https://arxiv.org/pdf/2506.00322)  

**Abstract**: We propose dpmm, an open-source library for synthetic data generation with Differentially Private (DP) guarantees. It includes three popular marginal models -- PrivBayes, MST, and AIM -- that achieve superior utility and offer richer functionality compared to alternative implementations. Additionally, we adopt best practices to provide end-to-end DP guarantees and address well-known DP-related vulnerabilities. Our goal is to accommodate a wide audience with easy-to-install, highly customizable, and robust model implementations.
Our codebase is available from this https URL. 

---
# An evaluation of LLMs for generating movie reviews: GPT-4o, Gemini-2.0 and DeepSeek-V3 

**Authors**: Brendan Sands, Yining Wang, Chenhao Xu, Yuxuan Zhou, Lai Wei, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00312)  

**Abstract**: Large language models (LLMs) have been prominent in various tasks, including text generation and summarisation. The applicability of LLMs to the generation of product reviews is gaining momentum, paving the way for the generation of movie reviews. In this study, we propose a framework that generates movie reviews using three LLMs (GPT-4o, DeepSeek-V3, and Gemini-2.0), and evaluate their performance by comparing the generated outputs with IMDb user reviews. We use movie subtitles and screenplays as input to the LLMs and investigate how they affect the quality of reviews generated. We review the LLM-based movie reviews in terms of vocabulary, sentiment polarity, similarity, and thematic consistency in comparison to IMDB user reviews. The results demonstrate that LLMs are capable of generating syntactically fluent and structurally complete movie reviews. Nevertheless, there is still a noticeable gap in emotional richness and stylistic coherence between LLM-generated and IMDb reviews, suggesting that further refinement is needed to improve the overall quality of movie review generation. We provided a survey-based analysis where participants were told to distinguish between LLM and IMDb user reviews. The results show that LLM-generated reviews are difficult to distinguish from IMDB user reviews. We found that DeepSeek-V3 produced the most balanced reviews, closely matching IMDb reviews. GPT-4o overemphasised positive emotions, while Gemini-2.0 captured negative emotions better but showed excessive emotional intensity. 

---
# MythTriage: Scalable Detection of Opioid Use Disorder Myths on a Video-Sharing Platform 

**Authors**: Hayoung Jung, Shravika Mittal, Ananya Aatreya, Navreet Kaur, Munmun De Choudhury, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00308)  

**Abstract**: Understanding the prevalence of misinformation in health topics online can inform public health policies and interventions. However, measuring such misinformation at scale remains a challenge, particularly for high-stakes but understudied topics like opioid-use disorder (OUD)--a leading cause of death in the U.S. We present the first large-scale study of OUD-related myths on YouTube, a widely-used platform for health information. With clinical experts, we validate 8 pervasive myths and release an expert-labeled video dataset. To scale labeling, we introduce MythTriage, an efficient triage pipeline that uses a lightweight model for routine cases and defers harder ones to a high-performing, but costlier, large language model (LLM). MythTriage achieves up to 0.86 macro F1-score while estimated to reduce annotation time and financial cost by over 76% compared to experts and full LLM labeling. We analyze 2.9K search results and 343K recommendations, uncovering how myths persist on YouTube and offering actionable insights for public health and platform moderation. 

---
# Lossless Token Sequence Compression via Meta-Tokens 

**Authors**: John Harvill, Ziwei Fan, Hao Wang, Yizhou Sun, Hao Ding, Luke Huan, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2506.00307)  

**Abstract**: Existing work on prompt compression for Large Language Models (LLM) focuses on lossy methods that try to maximize the retention of semantic information that is relevant to downstream tasks while significantly reducing the sequence length. In this paper, we introduce a task-agnostic lossless compression technique similar to LZ77 that makes it possible to reduce the input token sequence length on average by 27\% and 18\% for the two evaluation tasks explored here. Given that we use transformer-based LLMs, this equates to 47\% and 33\% less encoding computation, respectively, due to the quadratic nature of attention. The token sequence transformation is trivial to reverse and highlights that no semantic information is lost in the process. We evaluate our proposed approach on two tasks that require strict preservation of semantics/syntax and demonstrate that existing lossy compression methods perform poorly in this setting. We find that our lossless compression technique produces only a small gap in performance compared to using the uncompressed input and posit that larger models and an expanded computing budget would likely erase the gap entirely. 

---
# Improving Protein Sequence Design through Designability Preference Optimization 

**Authors**: Fanglei Xue, Andrew Kubaney, Zhichun Guo, Joseph K. Min, Ge Liu, Yi Yang, David Baker  

**Link**: [PDF](https://arxiv.org/pdf/2506.00297)  

**Abstract**: Protein sequence design methods have demonstrated strong performance in sequence generation for de novo protein design. However, as the training objective was sequence recovery, it does not guarantee designability--the likelihood that a designed sequence folds into the desired structure. To bridge this gap, we redefine the training objective by steering sequence generation toward high designability. To do this, we integrate Direct Preference Optimization (DPO), using AlphaFold pLDDT scores as the preference signal, which significantly improves the in silico design success rate. To further refine sequence generation at a finer, residue-level granularity, we introduce Residue-level Designability Preference Optimization (ResiDPO), which applies residue-level structural rewards and decouples optimization across residues. This enables direct improvement in designability while preserving regions that already perform well. Using a curated dataset with residue-level annotations, we fine-tune LigandMPNN with ResiDPO to obtain EnhancedMPNN, which achieves a nearly 3-fold increase in in silico design success rate (from 6.56% to 17.57%) on a challenging enzyme design benchmark. 

---
# Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation 

**Authors**: Ahmed Elhady, Eneko Agirre, Mikel Artetxe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00288)  

**Abstract**: Continued pretraining (CPT) is a popular approach to adapt existing large language models (LLMs) to new languages. When doing so, it is common practice to include a portion of English data in the mixture, but its role has not been carefully studied to date. In this work, we show that including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities in the target language. We introduce a language-agnostic benchmark for in-context learning (ICL), which reveals catastrophic forgetting early on CPT when English is not included. This in turn damages the ability of the model to generalize to downstream prompts in the target language as measured by perplexity, even if it does not manifest in terms of accuracy until later in training, and can be tied to a big shift in the model parameters. Based on these insights, we introduce curriculum learning and exponential moving average (EMA) of weights as effective alternatives to mitigate the need for English. All in all, our work sheds light into the dynamics by which emergent abilities arise when doing CPT for language adaptation, and can serve as a foundation to design more effective methods in the future. 

---
# Entropic Risk Optimization in Discounted MDPs: Sample Complexity Bounds with a Generative Model 

**Authors**: Oliver Mortensen, Mohammad Sadegh Talebi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00286)  

**Abstract**: In this paper we analyze the sample complexities of learning the optimal state-action value function $Q^*$ and an optimal policy $\pi^*$ in a discounted Markov decision process (MDP) where the agent has recursive entropic risk-preferences with risk-parameter $\beta\neq 0$ and where a generative model of the MDP is available. We provide and analyze a simple model based approach which we call model-based risk-sensitive $Q$-value-iteration (MB-RS-QVI) which leads to $(\epsilon,\delta)$-PAC-bounds on $\|Q^*-Q^k\|$, and $\|V^*-V^{\pi_k}\|$ where $Q_k$ is the output of MB-RS-QVI after k iterations and $\pi_k$ is the greedy policy with respect to $Q_k$. Both PAC-bounds have exponential dependence on the effective horizon $\frac{1}{1-\gamma}$ and the strength of this dependence grows with the learners risk-sensitivity $|\beta|$. We also provide two lower bounds which shows that exponential dependence on $|\beta|\frac{1}{1-\gamma}$ is unavoidable in both cases. The lower bounds reveal that the PAC-bounds are both tight in $\varepsilon$ and $\delta$ and that the PAC-bound on $Q$-learning is tight in the number of actions $A$, and that the PAC-bound on policy-learning is nearly tight in $A$. 

---
# Adversarial Threat Vectors and Risk Mitigation for Retrieval-Augmented Generation Systems 

**Authors**: Chris M. Ward, Josh Harguess  

**Link**: [PDF](https://arxiv.org/pdf/2506.00281)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems, which integrate Large Language Models (LLMs) with external knowledge sources, are vulnerable to a range of adversarial attack vectors. This paper examines the importance of RAG systems through recent industry adoption trends and identifies the prominent attack vectors for RAG: prompt injection, data poisoning, and adversarial query manipulation. We analyze these threats under risk management lens, and propose robust prioritized control list that includes risk-mitigating actions like input validation, adversarial training, and real-time monitoring. 

---
# Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings 

**Authors**: Hans W. A. Hanley, Zakir Durumeric  

**Link**: [PDF](https://arxiv.org/pdf/2506.00277)  

**Abstract**: Contextual large language model embeddings are increasingly utilized for topic modeling and clustering. However, current methods often scale poorly, rely on opaque similarity metrics, and struggle in multilingual settings. In this work, we present a novel, scalable, interpretable, hierarchical, and multilingual approach to clustering news articles and social media data. To do this, we first train multilingual Matryoshka embeddings that can determine story similarity at varying levels of granularity based on which subset of the dimensions of the embeddings is examined. This embedding model achieves state-of-the-art performance on the SemEval 2022 Task 8 test dataset (Pearson $\rho$ = 0.816). Once trained, we develop an efficient hierarchical clustering algorithm that leverages the hierarchical nature of Matryoshka embeddings to identify unique news stories, narratives, and themes. We conclude by illustrating how our approach can identify and cluster stories, narratives, and overarching themes within real-world news datasets. 

---
# Chances and Challenges of the Model Context Protocol in Digital Forensics and Incident Response 

**Authors**: Jan-Niclas Hilgert, Carlo Jakobs, Michael Külper, Martin Lambertz, Axel Mahr, Elmar Padilla  

**Link**: [PDF](https://arxiv.org/pdf/2506.00274)  

**Abstract**: Large language models hold considerable promise for supporting forensic investigations, but their widespread adoption is hindered by a lack of transparency, explainability, and reproducibility. This paper explores how the emerging Model Context Protocol can address these challenges and support the meaningful use of LLMs in digital forensics. Through a theoretical analysis, we examine how MCP can be integrated across various forensic scenarios - ranging from artifact analysis to the generation of interpretable reports. We also outline both technical and conceptual considerations for deploying an MCP server in forensic environments. Our analysis reveals a wide range of use cases in which MCP not only strengthens existing forensic workflows but also facilitates the application of LLMs to areas of forensics where their use was previously limited. Furthermore, we introduce the concept of the inference constraint level - a way of characterizing how specific MCP design choices can deliberately constrain model behavior, thereby enhancing both auditability and traceability. Our insights demonstrate that MCP has significant potential as a foundational component for developing LLM-assisted forensic workflows that are not only more transparent, reproducible, and legally defensible, but also represent a step toward increased automation in digital forensic analysis. However, we also highlight potential challenges that the adoption of MCP may pose for digital forensics in the future. 

---
# Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race 

**Authors**: Lihao Sun, Chengzhi Mao, Valentin Hofmann, Xuechunzi Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00253)  

**Abstract**: Although value-aligned language models (LMs) appear unbiased in explicit bias evaluations, they often exhibit stereotypes in implicit word association tasks, raising concerns about their fair usage. We investigate the mechanisms behind this discrepancy and find that alignment surprisingly amplifies implicit bias in model outputs. Specifically, we show that aligned LMs, unlike their unaligned counterparts, overlook racial concepts in early internal representations when the context is ambiguous. Not representing race likely fails to activate safety guardrails, leading to unintended biases. Inspired by this insight, we propose a new bias mitigation strategy that works by incentivizing the representation of racial concepts in the early model layers. In contrast to conventional mitigation methods of machine unlearning, our interventions find that steering the model to be more aware of racial concepts effectively mitigates implicit bias. Similar to race blindness in humans, ignoring racial nuances can inadvertently perpetuate subtle biases in LMs. 

---
# Designing AI Tools for Clinical Care Teams to Support Serious Illness Conversations with Older Adults in the Emergency Department 

**Authors**: Menglin Zhao, Zhuorui Yong, Ruijia Guan, Kai-Wei Chang, Adrian Haimovich, Kei Ouchi, Timothy Bickmore, Bingsheng Yao, Dakuo Wang, Smit Desai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00241)  

**Abstract**: Serious illness conversations (SICs), discussions between clinical care teams and patients with serious, life-limiting illnesses about their values, goals, and care preferences, are critical for patient-centered care. Without these conversations, patients often receive aggressive interventions that may not align with their goals. Clinical care teams face significant barriers when conducting serious illness conversations with older adult patients in Emergency Department (ED) settings, where most older adult patients lack documented treatment goals. To understand current practices and identify AI support opportunities, we conducted interviews with two domain experts and nine ED clinical care team members. Through thematic analysis, we characterized a four-phase serious illness conversation workflow (identification, preparation, conduction, documentation) and identified key needs and challenges at each stage. Clinical care teams struggle with fragmented EHR data access, time constraints, emotional preparation demands, and documentation burdens. While participants expressed interest in AI tools for information synthesis, conversational support, and automated documentation, they emphasized preserving human connection and clinical autonomy. We present design guidelines for AI tools supporting SIC workflows that fit within existing clinical practices. This work contributes empirical understanding of ED-based serious illness conversations and provides design considerations for AI in high-stakes clinical environments. 

---
# Localized LoRA: A Structured Low-Rank Approximation for Efficient Fine-Tuning 

**Authors**: Babak Barazandeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00236)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, offer compact and effective alternatives to full model fine-tuning by introducing low-rank updates to pretrained weights. However, most existing approaches rely on global low-rank structures, which can overlook spatial patterns spread across the parameter space. In this work, we propose Localized LoRA, a generalized framework that models weight updates as a composition of low-rank matrices applied to structured blocks of the weight matrix. This formulation enables dense, localized updates throughout the parameter space-without increasing the total number of trainable parameters. We provide a formal comparison between global, diagonal-local, and fully localized low-rank approximations, and show that our method consistently achieves lower approximation error under matched parameter budgets. Experiments on both synthetic and practical settings demonstrate that Localized LoRA offers a more expressive and adaptable alternative to existing methods, enabling efficient fine-tuning with improved performance. 

---
# Ctrl-Crash: Controllable Diffusion for Realistic Car Crashes 

**Authors**: Anthony Gosselin, Ge Ya Luo, Luis Lara, Florian Golemo, Derek Nowrouzezahrai, Liam Paull, Alexia Jolicoeur-Martineau, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00227)  

**Abstract**: Video diffusion techniques have advanced significantly in recent years; however, they struggle to generate realistic imagery of car crashes due to the scarcity of accident events in most driving datasets. Improving traffic safety requires realistic and controllable accident simulations. To tackle the problem, we propose Ctrl-Crash, a controllable car crash video generation model that conditions on signals such as bounding boxes, crash types, and an initial image frame. Our approach enables counterfactual scenario generation where minor variations in input can lead to dramatically different crash outcomes. To support fine-grained control at inference time, we leverage classifier-free guidance with independently tunable scales for each conditioning signal. Ctrl-Crash achieves state-of-the-art performance across quantitative video quality metrics (e.g., FVD and JEDi) and qualitative measurements based on a human-evaluation of physical realism and video quality compared to prior diffusion-based methods. 

---
# Diff-SPORT: Diffusion-based Sensor Placement Optimization and Reconstruction of Turbulent flows in urban environments 

**Authors**: Abhijeet Vishwasrao, Sai Bharath Chandra Gutha, Andres Cremades, Klas Wijk, Aakash Patil, Catherine Gorle, Beverley J McKeon, Hossein Azizpour, Ricardo Vinuesa  

**Link**: [PDF](https://arxiv.org/pdf/2506.00214)  

**Abstract**: Rapid urbanization demands accurate and efficient monitoring of turbulent wind patterns to support air quality, climate resilience and infrastructure design. Traditional sparse reconstruction and sensor placement strategies face major accuracy degradations under practical constraints. Here, we introduce Diff-SPORT, a diffusion-based framework for high-fidelity flow reconstruction and optimal sensor placement in urban environments. Diff-SPORT combines a generative diffusion model with a maximum a posteriori (MAP) inference scheme and a Shapley-value attribution framework to propose a scalable and interpretable solution. Compared to traditional numerical methods, Diff-SPORT achieves significant speedups while maintaining both statistical and instantaneous flow fidelity. Our approach offers a modular, zero-shot alternative to retraining-intensive strategies, supporting fast and reliable urban flow monitoring under extreme sparsity. Diff-SPORT paves the way for integrating generative modeling and explainability in sustainable urban intelligence. 

---
# REIC: RAG-Enhanced Intent Classification at Scale 

**Authors**: Ziji Zhang, Michael Yang, Zhiyu Chen, Yingying Zhuang, Shu-Ting Pi, Qun Liu, Rajashekar Maragoud, Vy Nguyen, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00210)  

**Abstract**: Accurate intent classification is critical for efficient routing in customer service, ensuring customers are connected with the most suitable agents while reducing handling times and operational costs. However, as companies expand their product lines, intent classification faces scalability challenges due to the increasing number of intents and variations in taxonomy across different verticals. In this paper, we introduce REIC, a Retrieval-augmented generation Enhanced Intent Classification approach, which addresses these challenges effectively. REIC leverages retrieval-augmented generation (RAG) to dynamically incorporate relevant knowledge, enabling precise classification without the need for frequent retraining. Through extensive experiments on real-world datasets, we demonstrate that REIC outperforms traditional fine-tuning, zero-shot, and few-shot methods in large-scale customer service settings. Our results highlight its effectiveness in both in-domain and out-of-domain scenarios, demonstrating its potential for real-world deployment in adaptive and large-scale intent classification systems. 

---
# Structure-Aware Fill-in-the-Middle Pretraining for Code 

**Authors**: Linyuan Gong, Alvin Cheung, Mostafa Elhoushi, Sida Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00204)  

**Abstract**: Fill-in-the-Middle (FIM) is a common pretraining method for code LLMs, where models complete code segments given surrounding context. However, existing LLMs treat code as plain text and mask random character spans. We propose and evaluate AST-FIM, a pretraining strategy that leverages Abstract Syntax Trees (ASTs) to mask complete syntactic structures at scale, ensuring coherent training examples better aligned with universal code structures and common code editing patterns such as blocks, expressions, or functions. To evaluate real-world fill-in-the-middle (FIM) programming tasks, we introduce Real-FIM-Eval, a benchmark derived from 30,000+ GitHub commits across 12 languages. On infilling tasks, experiments on 1B and 8B parameter models show that AST-FIM is particularly beneficial for real-world code editing as it outperforms standard random-character FIM by up to 5 pts on standard FIM benchmarks. Our code is publicly available at this https URL. 

---
# The World As Large Language Models See It: Exploring the reliability of LLMs in representing geographical features 

**Authors**: Omid Reza Abbasi, Franz Welscher, Georg Weinberger, Johannes Scholz  

**Link**: [PDF](https://arxiv.org/pdf/2506.00203)  

**Abstract**: As large language models (LLMs) continue to evolve, questions about their trustworthiness in delivering factual information have become increasingly important. This concern also applies to their ability to accurately represent the geographic world. With recent advancements in this field, it is relevant to consider whether and to what extent LLMs' representations of the geographical world can be trusted. This study evaluates the performance of GPT-4o and Gemini 2.0 Flash in three key geospatial tasks: geocoding, elevation estimation, and reverse geocoding. In the geocoding task, both models exhibited systematic and random errors in estimating the coordinates of St. Anne's Column in Innsbruck, Austria, with GPT-4o showing greater deviations and Gemini 2.0 Flash demonstrating more precision but a significant systematic offset. For elevation estimation, both models tended to underestimate elevations across Austria, though they captured overall topographical trends, and Gemini 2.0 Flash performed better in eastern regions. The reverse geocoding task, which involved identifying Austrian federal states from coordinates, revealed that Gemini 2.0 Flash outperformed GPT-4o in overall accuracy and F1-scores, demonstrating better consistency across regions. Despite these findings, neither model achieved an accurate reconstruction of Austria's federal states, highlighting persistent misclassifications. The study concludes that while LLMs can approximate geographic information, their accuracy and reliability are inconsistent, underscoring the need for fine-tuning with geographical information to enhance their utility in GIScience and Geoinformatics. 

---
# MOFGPT: Generative Design of Metal-Organic Frameworks using Language Models 

**Authors**: Srivathsan Badrinarayanan, Rishikesh Magar, Akshay Antony, Radheesh Sharma Meda, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2506.00198)  

**Abstract**: The discovery of Metal-Organic Frameworks (MOFs) with application-specific properties remains a central challenge in materials chemistry, owing to the immense size and complexity of their structural design space. Conventional computational screening techniques such as molecular simulations and density functional theory (DFT), while accurate, are computationally prohibitive at scale. Machine learning offers an exciting alternative by leveraging data-driven approaches to accelerate materials discovery. The complexity of MOFs, with their extended periodic structures and diverse topologies, creates both opportunities and challenges for generative modeling approaches. To address these challenges, we present a reinforcement learning-enhanced, transformer-based framework for the de novo design of MOFs. Central to our approach is MOFid, a chemically-informed string representation encoding both connectivity and topology, enabling scalable generative modeling. Our pipeline comprises three components: (1) a generative GPT model trained on MOFid sequences, (2) MOFormer, a transformer-based property predictor, and (3) a reinforcement learning (RL) module that optimizes generated candidates via property-guided reward functions. By integrating property feedback into sequence generation, our method drives the model toward synthesizable, topologically valid MOFs with desired functional attributes. This work demonstrates the potential of large language models, when coupled with reinforcement learning, to accelerate inverse design in reticular chemistry and unlock new frontiers in computational MOF discovery. 

---
# Let Them Down Easy! Contextual Effects of LLM Guardrails on User Perceptions and Preferences 

**Authors**: Mingqian Zheng, Wenjia Hu, Patrick Zhao, Motahhare Eslami, Jena D. Hwang, Faeze Brahman, Carolyn Rose, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2506.00195)  

**Abstract**: Current LLMs are trained to refuse potentially harmful input queries regardless of whether users actually had harmful intents, causing a tradeoff between safety and user experience. Through a study of 480 participants evaluating 3,840 query-response pairs, we examine how different refusal strategies affect user perceptions across varying motivations. Our findings reveal that response strategy largely shapes user experience, while actual user motivation has negligible impact. Partial compliance -- providing general information without actionable details -- emerges as the optimal strategy, reducing negative user perceptions by over 50% to flat-out refusals. Complementing this, we analyze response patterns of 9 state-of-the-art LLMs and evaluate how 6 reward models score different refusal strategies, demonstrating that models rarely deploy partial compliance naturally and reward models currently undervalue it. This work demonstrates that effective guardrails require focusing on crafting thoughtful refusals rather than detecting intent, offering a path toward AI safety mechanisms that ensure both safety and sustained user engagement. 

---
# Heterogeneous Graph Backdoor Attack 

**Authors**: Jiawei Chen, Lusi Li, Daniel Takabi, Masha Sosonkina, Rui Ning  

**Link**: [PDF](https://arxiv.org/pdf/2506.00191)  

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) excel in modeling complex, multi-typed relationships across diverse domains, yet their vulnerability to backdoor attacks remains unexplored. To address this gap, we conduct the first investigation into the susceptibility of HGNNs to existing graph backdoor attacks, revealing three critical issues: (1) high attack budget required for effective backdoor injection, (2) inefficient and unreliable backdoor activation, and (3) inaccurate attack effectiveness evaluation. To tackle these issues, we propose the Heterogeneous Graph Backdoor Attack (HGBA), the first backdoor attack specifically designed for HGNNs, introducing a novel relation-based trigger mechanism that establishes specific connections between a strategically selected trigger node and poisoned nodes via the backdoor metapath. HGBA achieves efficient and stealthy backdoor injection with minimal structural modifications and supports easy backdoor activation through two flexible strategies: Self-Node Attack and Indiscriminate Attack. Additionally, we improve the ASR measurement protocol, enabling a more accurate assessment of attack effectiveness. Extensive experiments demonstrate that HGBA far surpasses multiple state-of-the-art graph backdoor attacks in black-box settings, efficiently attacking HGNNs with low attack budgets. Ablation studies show that the strength of HBGA benefits from our trigger node selection method and backdoor metapath selection strategy. In addition, HGBA shows superior robustness against node feature perturbations and multiple types of existing graph backdoor defense mechanisms. Finally, extension experiments demonstrate that the relation-based trigger mechanism can effectively extend to tasks in homogeneous graph scenarios, thereby posing severe threats to broader security-critical domains. 

---
# Pushing the Limits of Beam Search Decoding for Transducer-based ASR models 

**Authors**: Lilit Grigoryan, Vladimir Bataev, Andrei Andrusenko, Hainan Xu, Vitaly Lavrukhin, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2506.00185)  

**Abstract**: Transducer models have emerged as a promising choice for end-to-end ASR systems, offering a balanced trade-off between recognition accuracy, streaming capabilities, and inference speed in greedy decoding. However, beam search significantly slows down Transducers due to repeated evaluations of key network components, limiting practical applications. This paper introduces a universal method to accelerate beam search for Transducers, enabling the implementation of two optimized algorithms: ALSD++ and AES++. The proposed method utilizes batch operations, a tree-based hypothesis structure, novel blank scoring for enhanced shallow fusion, and CUDA graph execution for efficient GPU inference. This narrows the speed gap between beam and greedy modes to only 10-20% for the whole system, achieves 14-30% relative improvement in WER compared to greedy decoding, and improves shallow fusion for low-resource up to 11% compared to existing implementations. All the algorithms are open sourced. 

---
# Accountability Attribution: Tracing Model Behavior to Training Processes 

**Authors**: Shichang Zhang, Hongzhe Du, Karim Saraipour, Jiaqi W. Ma, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2506.00175)  

**Abstract**: Modern AI development pipelines often involve multiple stages-pretraining, fine-tuning rounds, and subsequent adaptation or alignment-with numerous model update steps within each stage. This raises a critical question of accountability: when a deployed model succeeds or fails, which stage is responsible, and to what extent? We pose the problem of accountability attribution, which aims to trace model behavior back to specific stages of the training process. To address this, we propose a general framework that answers counterfactual questions about stage effects: how would the model behavior have changed if the updates from a training stage had not been executed?. Within this framework, we introduce estimators based on first-order approximations that efficiently quantify the stage effects without retraining. Our estimators account for both the training data and key aspects of optimization dynamics, including learning rate schedules, momentum, and weight decay. Empirically, we demonstrate that our approach identifies training stages accountable for specific behaviors, offering a practical tool for model analysis and a step toward more accountable AI development. 

---
# Disentangled Safety Adapters Enable Efficient Guardrails and Flexible Inference-Time Alignment 

**Authors**: Kundan Krishna, Joseph Y Cheng, Charles Maalouf, Leon A Gatys  

**Link**: [PDF](https://arxiv.org/pdf/2506.00166)  

**Abstract**: Existing paradigms for ensuring AI safety, such as guardrail models and alignment training, often compromise either inference efficiency or development flexibility. We introduce Disentangled Safety Adapters (DSA), a novel framework addressing these challenges by decoupling safety-specific computations from a task-optimized base model. DSA utilizes lightweight adapters that leverage the base model's internal representations, enabling diverse and flexible safety functionalities with minimal impact on inference cost. Empirically, DSA-based safety guardrails substantially outperform comparably sized standalone models, notably improving hallucination detection (0.88 vs. 0.61 AUC on Summedits) and also excelling at classifying hate speech (0.98 vs. 0.92 on ToxiGen) and unsafe model inputs and responses (0.93 vs. 0.90 on AEGIS2.0 & BeaverTails). Furthermore, DSA-based safety alignment allows dynamic, inference-time adjustment of alignment strength and a fine-grained trade-off between instruction following performance and model safety. Importantly, combining the DSA safety guardrail with DSA safety alignment facilitates context-dependent alignment strength, boosting safety on StrongReject by 93% while maintaining 98% performance on MTBench -- a total reduction in alignment tax of 8 percentage points compared to standard safety alignment fine-tuning. Overall, DSA presents a promising path towards more modular, efficient, and adaptable AI safety and alignment. 

---
# Detection of Endangered Deer Species Using UAV Imagery: A Comparative Study Between Efficient Deep Learning Approaches 

**Authors**: Agustín Roca, Gastón Castro, Gabriel Torre, Leonardo J. Colombo, Ignacio Mas, Javier Pereira, Juan I. Giribet  

**Link**: [PDF](https://arxiv.org/pdf/2506.00154)  

**Abstract**: This study compares the performance of state-of-the-art neural networks including variants of the YOLOv11 and RT-DETR models for detecting marsh deer in UAV imagery, in scenarios where specimens occupy a very small portion of the image and are occluded by vegetation. We extend previous analysis adding precise segmentation masks for our datasets enabling a fine-grained training of a YOLO model with a segmentation head included. Experimental results show the effectiveness of incorporating the segmentation head achieving superior detection performance. This work contributes valuable insights for improving UAV-based wildlife monitoring and conservation strategies through scalable and accurate AI-driven detection systems. 

---
# Supporting architecture evaluation for ATAM scenarios with LLMs 

**Authors**: Rafael Capilla, J. Andrés Díaz-Pace, Yamid Ramírez, Jennifer Pérez, Vanessa Rodríguez-Horcajo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00150)  

**Abstract**: Architecture evaluation methods have long been used to evaluate software designs. Several evaluation methods have been proposed and used to analyze tradeoffs between different quality attributes. Having competing qualities leads to conflicts for selecting which quality-attribute scenarios are the most suitable ones that an architecture should tackle and for prioritizing the scenarios required by the stakeholders. In this context, architecture evaluation is carried out manually, often involving long brainstorming sessions to decide which are the most adequate quality scenarios. To reduce this effort and make the assessment and selection of scenarios more efficient, we suggest the usage of LLMs to partially automate evaluation activities. As a first step to validate this hypothesis, this work studies MS Copilot as an LLM tool to analyze quality scenarios suggested by students in a software architecture course and compares the students' results with the assessment provided by the LLM. Our initial study reveals that the LLM produces in most cases better and more accurate results regarding the risks, sensitivity points and tradeoff analysis of the quality scenarios. Overall, the use of generative AI has the potential to partially automate and support the architecture evaluation tasks, improving the human decision-making process. 

---
# Autonomous Behavior and Whole-Brain Dynamics Emerge in Embodied Zebrafish Agents with Model-based Intrinsic Motivation 

**Authors**: Reece Keller, Alyn Tornell, Felix Pei, Xaq Pitkow, Leo Kozachkov, Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00138)  

**Abstract**: Autonomy is a hallmark of animal intelligence, enabling adaptive and intelligent behavior in complex environments without relying on external reward or task structure. Existing reinforcement learning approaches to exploration in sparse reward and reward-free environments, including class of methods known as intrinsic motivation, exhibit inconsistent exploration patterns and thus fail to produce robust autonomous behaviors observed in animals. Moreover, systems neuroscience has largely overlooked the neural basis of autonomy, focusing instead on experimental paradigms where animals are motivated by external reward rather than engaging in unconstrained, naturalistic and task-independent behavior. To bridge these gaps, we introduce a novel model-based intrinsic drive explicitly designed to capture robust autonomous exploration observed in animals. Our method (3M-Progress) motivates naturalistic behavior by tracking divergence between the agent's current world model and an ethological prior. We demonstrate that artificial embodied agents trained with 3M-Progress capture the explainable variance in behavioral patterns and whole-brain neural-glial dynamics recorded from autonomously-behaving larval zebrafish, introducing the first goal-driven, population-level model of neural-glial computation. Our findings establish a computational framework connecting model-based intrinsic motivation to naturalistic behavior, providing a foundation for building artificial agents with animal-like autonomy. 

---
# Spurious Correlations and Beyond: Understanding and Mitigating Shortcut Learning in SDOH Extraction with Large Language Models 

**Authors**: Fardin Ahsan Sakib, Ziwei Zhu, Karen Trister Grace, Meliha Yetisgen, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2506.00134)  

**Abstract**: Social determinants of health (SDOH) extraction from clinical text is critical for downstream healthcare analytics. Although large language models (LLMs) have shown promise, they may rely on superficial cues leading to spurious predictions. Using the MIMIC portion of the SHAC (Social History Annotation Corpus) dataset and focusing on drug status extraction as a case study, we demonstrate that mentions of alcohol or smoking can falsely induce models to predict current/past drug use where none is present, while also uncovering concerning gender disparities in model performance. We further evaluate mitigation strategies - such as prompt engineering and chain-of-thought reasoning - to reduce these false positives, providing insights into enhancing LLM reliability in health domains. 

---
# A Reinforcement Learning-Based Telematic Routing Protocol for the Internet of Underwater Things 

**Authors**: Mohammadhossein Homaei, Mehran Tarif, Agustin Di Bartolo, Oscar Mogollon Gutierrez, Mar Avila  

**Link**: [PDF](https://arxiv.org/pdf/2506.00133)  

**Abstract**: The Internet of Underwater Things (IoUT) faces major challenges such as low bandwidth, high latency, mobility, and limited energy resources. Traditional routing protocols like RPL, which were designed for land-based networks, do not perform well in these underwater conditions. This paper introduces RL-RPL-UA, a new routing protocol that uses reinforcement learning to improve performance in underwater environments. Each node includes a lightweight RL agent that selects the best parent node based on local information such as packet delivery ratio, buffer level, link quality, and remaining energy. RL-RPL-UA keeps full compatibility with standard RPL messages and adds a dynamic objective function to support real-time decision-making. Simulations using Aqua-Sim show that RL-RPL-UA increases packet delivery by up to 9.2%, reduces energy use per packet by 14.8%, and extends network lifetime by 80 seconds compared to traditional methods. These results suggest that RL-RPL-UA is a promising and energy-efficient routing solution for underwater networks. 

---
# Adapting Offline Reinforcement Learning with Online Delays 

**Authors**: Simon Sinong Zhan, Qingyuan Wu, Frank Yang, Xiangyu Shi, Chao Huang, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00131)  

**Abstract**: Offline-to-online deployment of reinforcement-learning (RL) agents must bridge two gaps: (1) the sim-to-real gap, where real systems add latency and other imperfections not present in simulation, and (2) the interaction gap, where policies trained purely offline face out-of-distribution states during online execution because gathering new interaction data is costly or risky. Agents therefore have to generalize from static, delay-free datasets to dynamic, delay-prone environments. Standard offline RL learns from delay-free logs yet must act under delays that break the Markov assumption and hurt performance. We introduce DT-CORL (Delay-Transformer belief policy Constrained Offline RL), an offline-RL framework built to cope with delayed dynamics at deployment. DT-CORL (i) produces delay-robust actions with a transformer-based belief predictor even though it never sees delayed observations during training, and (ii) is markedly more sample-efficient than naïve history-augmentation baselines. Experiments on D4RL benchmarks with several delay settings show that DT-CORL consistently outperforms both history-augmentation and vanilla belief-based methods, narrowing the sim-to-real latency gap while preserving data efficiency. 

---
# Gated Multimodal Graph Learning for Personalized Recommendation 

**Authors**: Sibei Liu, Yuanzhe Zhang, Xiang Li, Yunbo Liu, Chengwei Feng, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00107)  

**Abstract**: Multimodal recommendation has emerged as a promising solution to alleviate the cold-start and sparsity problems in collaborative filtering by incorporating rich content information, such as product images and textual descriptions. However, effectively integrating heterogeneous modalities into a unified recommendation framework remains a challenge. Existing approaches often rely on fixed fusion strategies or complex architectures , which may fail to adapt to modality quality variance or introduce unnecessary computational overhead.
In this work, we propose RLMultimodalRec, a lightweight and modular recommendation framework that combines graph-based user modeling with adaptive multimodal item encoding. The model employs a gated fusion module to dynamically balance the contribution of visual and textual modalities, enabling fine-grained and content-aware item representations. Meanwhile, a two-layer LightGCN encoder captures high-order collaborative signals by propagating embeddings over the user-item interaction graph without relying on nonlinear transformations.
We evaluate our model on a real-world dataset from the Amazon product domain. Experimental results demonstrate that RLMultimodalRec consistently outperforms several competitive baselines, including collaborative filtering, visual-aware, and multimodal GNN-based methods. The proposed approach achieves significant improvements in top-K recommendation metrics while maintaining scalability and interpretability, making it suitable for practical deployment. 

---
# Children's Voice Privacy: First Steps And Emerging Challenges 

**Authors**: Ajinkya Kulkarni, Francisco Teixeira, Enno Hermann, Thomas Rolland, Isabel Trancoso, Mathew Magimai Doss  

**Link**: [PDF](https://arxiv.org/pdf/2506.00100)  

**Abstract**: Children are one of the most under-represented groups in speech technologies, as well as one of the most vulnerable in terms of privacy. Despite this, anonymization techniques targeting this population have received little attention. In this study, we seek to bridge this gap, and establish a baseline for the use of voice anonymization techniques designed for adult speech when applied to children's voices. Such an evaluation is essential, as children's speech presents a distinct set of challenges when compared to that of adults. This study comprises three children's datasets, six anonymization methods, and objective and subjective utility metrics for evaluation. Our results show that existing systems for adults are still able to protect children's voice privacy, but suffer from much higher utility degradation. In addition, our subjective study displays the challenges of automatic evaluation methods for speech quality in children's speech, highlighting the need for further research. 

---
# PathGene: Benchmarking Driver Gene Mutations and Exon Prediction Using Multicenter Lung Cancer Histopathology Image Dataset 

**Authors**: Liangrui Pan, Qingchun Liang, Shen Zhao, Songqing Fan, Shaoliang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00096)  

**Abstract**: Accurately predicting gene mutations, mutation subtypes and their exons in lung cancer is critical for personalized treatment planning and prognostic assessment. Faced with regional disparities in medical resources and the high cost of genomic assays, using artificial intelligence to infer these mutations and exon variants from routine histopathology images could greatly facilitate precision therapy. Although some prior studies have shown that deep learning can accelerate the prediction of key gene mutations from lung cancer pathology slides, their performance remains suboptimal and has so far been limited mainly to early screening tasks. To address these limitations, we have assembled PathGene, which comprises histopathology images paired with next-generation sequencing reports from 1,576 patients at the Second Xiangya Hospital, Central South University, and 448 TCGA-LUAD patients. This multi-center dataset links whole-slide images to driver gene mutation status, mutation subtypes, exon, and tumor mutational burden (TMB) status, with the goal of leveraging pathology images to predict mutations, subtypes, exon locations, and TMB for early genetic screening and to advance precision oncology. Unlike existing datasets, we provide molecular-level information related to histopathology images in PathGene to facilitate the development of biomarker prediction models. We benchmarked 11 multiple-instance learning methods on PathGene for mutation, subtype, exon, and TMB prediction tasks. These experimental methods provide valuable alternatives for early genetic screening of lung cancer patients and assisting clinicians to quickly develop personalized precision targeted treatment plans for patients. Code and data are available at this https URL. 

---
# ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases 

**Authors**: Yuchong Li, Xiaojun Zeng, Chihua Fang, Jian Yang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00095)  

**Abstract**: Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at the homepage. 

---
# Feeling Guilty Being a c(ai)borg: Navigating the Tensions Between Guilt and Empowerment in AI Use 

**Authors**: Konstantin Aal, Tanja Aal, Vasil Navumau, David Unbehaun, Claudia Müller, Volker Wulf, Sarah Rüller  

**Link**: [PDF](https://arxiv.org/pdf/2506.00094)  

**Abstract**: This paper explores the emotional, ethical and practical dimensions of integrating Artificial Intelligence (AI) into personal and professional workflows, focusing on the concept of feeling guilty as a 'c(ai)borg' - a human augmented by AI. Inspired by Donna Haraway's Cyborg Manifesto, the study explores how AI challenges traditional notions of creativity, originality and intellectual labour. Using an autoethnographic approach, the authors reflect on their year-long experiences with AI tools, revealing a transition from initial guilt and reluctance to empowerment through skill-building and transparency. Key findings highlight the importance of basic academic skills, advanced AI literacy and honest engagement with AI results. The c(ai)borg vision advocates for a future where AI is openly embraced as a collaborative partner, fostering innovation and equity while addressing issues of access and agency. By reframing guilt as growth, the paper calls for a thoughtful and inclusive approach to AI integration. 

---
# TRAPDOC: Deceiving LLM Users by Injecting Imperceptible Phantom Tokens into Documents 

**Authors**: Hyundong Jin, Sicheol Sung, Shinwoo Park, SeungYeop Baik, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.00089)  

**Abstract**: The reasoning, writing, text-editing, and retrieval capabilities of proprietary large language models (LLMs) have advanced rapidly, providing users with an ever-expanding set of functionalities. However, this growing utility has also led to a serious societal concern: the over-reliance on LLMs. In particular, users increasingly delegate tasks such as homework, assignments, or the processing of sensitive documents to LLMs without meaningful engagement. This form of over-reliance and misuse is emerging as a significant social issue. In order to mitigate these issues, we propose a method injecting imperceptible phantom tokens into documents, which causes LLMs to generate outputs that appear plausible to users but are in fact incorrect. Based on this technique, we introduce TRAPDOC, a framework designed to deceive over-reliant LLM users. Through empirical evaluation, we demonstrate the effectiveness of our framework on proprietary LLMs, comparing its impact against several baselines. TRAPDOC serves as a strong foundation for promoting more responsible and thoughtful engagement with language models. Our code is available at this https URL. 

---
# HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs 

**Authors**: Qing Li, Jiahui Geng, Zongxiong Chen, Derui Zhu, Yuxia Wang, Congbo Ma, Chenyang Lyu, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2506.00088)  

**Abstract**: In recent years, large language models (LLMs) have made remarkable advancements, yet hallucination, where models produce inaccurate or non-factual statements, remains a significant challenge for real-world deployment. Although current classification-based methods, such as SAPLMA, are highly efficient in mitigating hallucinations, they struggle when non-factual information arises in the early or mid-sequence of outputs, reducing their reliability. To address these issues, we propose Hallucination Detection-Neural Differential Equations (HD-NDEs), a novel method that systematically assesses the truthfulness of statements by capturing the full dynamics of LLMs within their latent space. Our approaches apply neural differential equations (Neural DEs) to model the dynamic system in the latent space of LLMs. Then, the sequence in the latent space is mapped to the classification space for truth assessment. The extensive experiments across five datasets and six widely used LLMs demonstrate the effectiveness of HD-NDEs, especially, achieving over 14% improvement in AUC-ROC on the True-False dataset compared to state-of-the-art techniques. 

---
# SwitchLingua: The First Large-Scale Multilingual and Multi-Ethnic Code-Switching Dataset 

**Authors**: Peng Xie, Xingyuan Liu, Tsz Wai Chan, Yequan Bie, Yangqiu Song, Yang Wang, Hao Chen, Kani Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00087)  

**Abstract**: Code-switching (CS) is the alternating use of two or more languages within a conversation or utterance, often influenced by social context and speaker identity. This linguistic phenomenon poses challenges for Automatic Speech Recognition (ASR) systems, which are typically designed for a single language and struggle to handle multilingual inputs. The growing global demand for multilingual applications, including Code-Switching ASR (CSASR), Text-to-Speech (CSTTS), and Cross-Lingual Information Retrieval (CLIR), highlights the inadequacy of existing monolingual datasets.
Although some code-switching datasets exist, most are limited to bilingual mixing within homogeneous ethnic groups, leaving a critical need for a large-scale, diverse benchmark akin to ImageNet in computer vision.
To bridge this gap, we introduce \textbf{LinguaMaster}, a multi-agent collaboration framework specifically designed for efficient and scalable multilingual data synthesis. Leveraging this framework, we curate \textbf{SwitchLingua}, the first large-scale multilingual and multi-ethnic code-switching dataset, including: (1) 420K CS textual samples across 12 languages, and (2) over 80 hours of audio recordings from 174 speakers representing 18 countries/regions and 63 racial/ethnic backgrounds, based on the textual data. This dataset captures rich linguistic and cultural diversity, offering a foundational resource for advancing multilingual and multicultural research. Furthermore, to address the issue that existing ASR evaluation metrics lack sensitivity to code-switching scenarios, we propose the \textbf{Semantic-Aware Error Rate (SAER)}, a novel evaluation metric that incorporates semantic information, providing a more accurate and context-aware assessment of system performance. 

---
# COSMIC: Generalized Refusal Direction Identification in LLM Activations 

**Authors**: Vincent Siu, Nicholas Crispino, Zihao Yu, Sam Pan, Zhun Wang, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00085)  

**Abstract**: Large Language Models (LLMs) encode behaviors such as refusal within their activation space, yet identifying these behaviors remains a significant challenge. Existing methods often rely on predefined refusal templates detectable in output tokens or require manual analysis. We introduce \textbf{COSMIC} (Cosine Similarity Metrics for Inversion of Concepts), an automated framework for direction selection that identifies viable steering directions and target layers using cosine similarity - entirely independent of model outputs. COSMIC achieves steering performance comparable to prior methods without requiring assumptions about a model's refusal behavior, such as the presence of specific refusal tokens. It reliably identifies refusal directions in adversarial settings and weakly aligned models, and is capable of steering such models toward safer behavior with minimal increase in false refusals, demonstrating robustness across a wide range of alignment conditions. 

---
# Hi-Dyna Graph: Hierarchical Dynamic Scene Graph for Robotic Autonomy in Human-Centric Environments 

**Authors**: Jiawei Hou, Xiangyang Xue, Taiping Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00083)  

**Abstract**: Autonomous operation of service robotics in human-centric scenes remains challenging due to the need for understanding of changing environments and context-aware decision-making. While existing approaches like topological maps offer efficient spatial priors, they fail to model transient object relationships, whereas dense neural representations (e.g., NeRF) incur prohibitive computational costs. Inspired by the hierarchical scene representation and video scene graph generation works, we propose Hi-Dyna Graph, a hierarchical dynamic scene graph architecture that integrates persistent global layouts with localized dynamic semantics for embodied robotic autonomy. Our framework constructs a global topological graph from posed RGB-D inputs, encoding room-scale connectivity and large static objects (e.g., furniture), while environmental and egocentric cameras populate dynamic subgraphs with object position relations and human-object interaction patterns. A hybrid architecture is conducted by anchoring these subgraphs to the global topology using semantic and spatial constraints, enabling seamless updates as the environment evolves. An agent powered by large language models (LLMs) is employed to interpret the unified graph, infer latent task triggers, and generate executable instructions grounded in robotic affordances. We conduct complex experiments to demonstrate Hi-Dyna Grap's superior scene representation effectiveness. Real-world deployments validate the system's practicality with a mobile manipulator: robotics autonomously complete complex tasks with no further training or complex rewarding in a dynamic scene as cafeteria assistant. See this https URL for video demonstration and more details. 

---
# Artificial Empathy: AI based Mental Health 

**Authors**: Aditya Naik, Jovi Thomas, Teja Sree, Himavant Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00081)  

**Abstract**: Many people suffer from mental health problems but not everyone seeks professional help or has access to mental health care. AI chatbots have increasingly become a go-to for individuals who either have mental disorders or simply want someone to talk to. This paper presents a study on participants who have previously used chatbots and a scenario-based testing of large language model (LLM) chatbots. Our findings indicate that AI chatbots were primarily utilized as a "Five minute therapist" or as a non-judgmental companion. Participants appreciated the anonymity and lack of judgment from chatbots. However, there were concerns about privacy and the security of sensitive information. The scenario-based testing of LLM chatbots highlighted additional issues. Some chatbots were consistently reassuring, used emojis and names to add a personal touch, and were quick to suggest seeking professional help. However, there were limitations such as inconsistent tone, occasional inappropriate responses (e.g., casual or romantic), and a lack of crisis sensitivity, particularly in recognizing red flag language and escalating responses appropriately. These findings can inform both the technology and mental health care industries on how to better utilize AI chatbots to support individuals during challenging emotional periods. 

---
# Bottom-Up Perspectives on AI Governance: Insights from User Reviews of AI Products 

**Authors**: Stefan Pasch  

**Link**: [PDF](https://arxiv.org/pdf/2506.00080)  

**Abstract**: With the growing importance of AI governance, numerous high-level frameworks and principles have been articulated by policymakers, institutions, and expert communities to guide the development and application of AI. While such frameworks offer valuable normative orientation, they may not fully capture the practical concerns of those who interact with AI systems in organizational and operational contexts. To address this gap, this study adopts a bottom-up approach to explore how governance-relevant themes are expressed in user discourse. Drawing on over 100,000 user reviews of AI products from this http URL, we apply BERTopic to extract latent themes and identify those most semantically related to AI governance. The analysis reveals a diverse set of governance-relevant topics spanning both technical and non-technical domains. These include concerns across organizational processes-such as planning, coordination, and communication-as well as stages of the AI value chain, including deployment infrastructure, data handling, and analytics. The findings show considerable overlap with institutional AI governance and ethics frameworks on issues like privacy and transparency, but also surface overlooked areas such as project management, strategy development, and customer interaction. This highlights the need for more empirically grounded, user-centered approaches to AI governance-approaches that complement normative models by capturing how governance unfolds in applied settings. By foregrounding how governance is enacted in practice, this study contributes to more inclusive and operationally grounded approaches to AI governance and digital policy. 

---
# Who Gets the Kidney? Human-AI Alignment, Indecision, and Moral Values 

**Authors**: John P. Dickerson, Hadi Hosseini, Samarth Khanna, Leona Pierce  

**Link**: [PDF](https://arxiv.org/pdf/2506.00079)  

**Abstract**: The rapid integration of Large Language Models (LLMs) in high-stakes decision-making -- such as allocating scarce resources like donor organs -- raises critical questions about their alignment with human moral values. We systematically evaluate the behavior of several prominent LLMs against human preferences in kidney allocation scenarios and show that LLMs: i) exhibit stark deviations from human values in prioritizing various attributes, and ii) in contrast to humans, LLMs rarely express indecision, opting for deterministic decisions even when alternative indecision mechanisms (e.g., coin flipping) are provided. Nonetheless, we show that low-rank supervised fine-tuning with few samples is often effective in improving both decision consistency and calibrating indecision modeling. These findings illustrate the necessity of explicit alignment strategies for LLMs in moral/ethical domains. 

---
# Optimizing Storytelling, Improving Audience Retention, and Reducing Waste in the Entertainment Industry 

**Authors**: Andrew Cornfeld, Ashley Miller, Mercedes Mora-Figueroa, Kurt Samuels, Anthony Palomba  

**Link**: [PDF](https://arxiv.org/pdf/2506.00076)  

**Abstract**: Television networks face high financial risk when making programming decisions, often relying on limited historical data to forecast episodic viewership. This study introduces a machine learning framework that integrates natural language processing (NLP) features from over 25000 television episodes with traditional viewership data to enhance predictive accuracy. By extracting emotional tone, cognitive complexity, and narrative structure from episode dialogue, we evaluate forecasting performance using SARIMAX, rolling XGBoost, and feature selection models. While prior viewership remains a strong baseline predictor, NLP features contribute meaningful improvements for some series. We also introduce a similarity scoring method based on Euclidean distance between aggregate dialogue vectors to compare shows by content. Tested across diverse genres, including Better Call Saul and Abbott Elementary, our framework reveals genre-specific performance and offers interpretable metrics for writers, executives, and marketers seeking data-driven insight into audience behavior. 

---
# Reducing Latency in LLM-Based Natural Language Commands Processing for Robot Navigation 

**Authors**: Diego Pollini, Bruna V. Guterres, Rodrigo S. Guerra, Ricardo B. Grando  

**Link**: [PDF](https://arxiv.org/pdf/2506.00075)  

**Abstract**: The integration of Large Language Models (LLMs), such as GPT, in industrial robotics enhances operational efficiency and human-robot collaboration. However, the computational complexity and size of these models often provide latency problems in request and response times. This study explores the integration of the ChatGPT natural language model with the Robot Operating System 2 (ROS 2) to mitigate interaction latency and improve robotic system control within a simulated Gazebo environment. We present an architecture that integrates these technologies without requiring a middleware transport platform, detailing how a simulated mobile robot responds to text and voice commands. Experimental results demonstrate that this integration improves execution speed, usability, and accessibility of the human-robot interaction by decreasing the communication latency by 7.01\% on average. Such improvements facilitate smoother, real-time robot operations, which are crucial for industrial automation and precision tasks. 

---
# Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations 

**Authors**: Daniele Barolo, Chiara Valentin, Fariba Karimi, Luis Galárraga, Gonzalo G. Méndez, Lisette Espín-Noboa  

**Link**: [PDF](https://arxiv.org/pdf/2506.00074)  

**Abstract**: This paper evaluates the performance of six open-weight LLMs (llama3-8b, llama3.1-8b, gemma2-9b, mixtral-8x7b, llama3-70b, llama3.1-70b) in recommending experts in physics across five tasks: top-k experts by field, influential scientists by discipline, epoch, seniority, and scholar counterparts. The evaluation examines consistency, factuality, and biases related to gender, ethnicity, academic popularity, and scholar similarity. Using ground-truth data from the American Physical Society and OpenAlex, we establish scholarly benchmarks by comparing model outputs to real-world academic records. Our analysis reveals inconsistencies and biases across all models. mixtral-8x7b produces the most stable outputs, while llama3.1-70b shows the highest variability. Many models exhibit duplication, and some, particularly gemma2-9b and llama3.1-8b, struggle with formatting errors. LLMs generally recommend real scientists, but accuracy drops in field-, epoch-, and seniority-specific queries, consistently favoring senior scholars. Representation biases persist, replicating gender imbalances (reflecting male predominance), under-representing Asian scientists, and over-representing White scholars. Despite some diversity in institutional and collaboration networks, models favor highly cited and productive scholars, reinforcing the rich-getricher effect while offering limited geographical representation. These findings highlight the need to improve LLMs for more reliable and equitable scholarly recommendations. 

---
# Evaluating Prompt Engineering Techniques for Accuracy and Confidence Elicitation in Medical LLMs 

**Authors**: Nariman Naderi, Zahra Atf, Peter R Lewis, Aref Mahjoub far, Seyed Amir Ahmad Safavi-Naini, Ali Soroush  

**Link**: [PDF](https://arxiv.org/pdf/2506.00072)  

**Abstract**: This paper investigates how prompt engineering techniques impact both accuracy and confidence elicitation in Large Language Models (LLMs) applied to medical contexts. Using a stratified dataset of Persian board exam questions across multiple specialties, we evaluated five LLMs - GPT-4o, o3-mini, Llama-3.3-70b, Llama-3.1-8b, and DeepSeek-v3 - across 156 configurations. These configurations varied in temperature settings (0.3, 0.7, 1.0), prompt styles (Chain-of-Thought, Few-Shot, Emotional, Expert Mimicry), and confidence scales (1-10, 1-100). We used AUC-ROC, Brier Score, and Expected Calibration Error (ECE) to evaluate alignment between confidence and actual performance. Chain-of-Thought prompts improved accuracy but also led to overconfidence, highlighting the need for calibration. Emotional prompting further inflated confidence, risking poor decisions. Smaller models like Llama-3.1-8b underperformed across all metrics, while proprietary models showed higher accuracy but still lacked calibrated confidence. These results suggest prompt engineering must address both accuracy and uncertainty to be effective in high-stakes medical tasks. 

---
# Human sensory-musculoskeletal modeling and control of whole-body movements 

**Authors**: Chenhui Zuo, Guohao Lin, Chen Zhang, Shanning Zhuang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00071)  

**Abstract**: Coordinated human movement depends on the integration of multisensory inputs, sensorimotor transformation, and motor execution, as well as sensory feedback resulting from body-environment interaction. Building dynamic models of the sensory-musculoskeletal system is essential for understanding movement control and investigating human behaviours. Here, we report a human sensory-musculoskeletal model, termed SMS-Human, that integrates precise anatomical representations of bones, joints, and muscle-tendon units with multimodal sensory inputs involving visual, vestibular, proprioceptive, and tactile components. A stage-wise hierarchical deep reinforcement learning framework was developed to address the inherent challenges of high-dimensional control in musculoskeletal systems with integrated multisensory information. Using this framework, we demonstrated the simulation of three representative movement tasks, including bipedal locomotion, vision-guided object manipulation, and human-machine interaction during bicycling. Our results showed a close resemblance between natural and simulated human motor behaviours. The simulation also revealed musculoskeletal dynamics that could not be directly measured. This work sheds deeper insights into the sensorimotor dynamics of human movements, facilitates quantitative understanding of human behaviours in interactive contexts, and informs the design of systems with embodied intelligence. 

---
# Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics 

**Authors**: Dongyoung Kim, Sumin Park, Huiwon Jang, Jinwoo Shin, Jaehyung Kim, Younggyo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00070)  

**Abstract**: Large Vision-Language Models (LVLMs) have recently shown great promise in advancing robotics by combining embodied reasoning with robot control. A common approach involves training on embodied reasoning tasks related to robot control using Supervised Fine-Tuning (SFT). However, SFT datasets are often heuristically constructed and not explicitly optimized for improving robot control. Furthermore, SFT often leads to issues such as catastrophic forgetting and reduced generalization performance. To address these limitations, we introduce Robot-R1, a novel framework that leverages reinforcement learning to enhance embodied reasoning specifically for robot control. Robot-R1 learns to predict the next keypoint state required for task completion, conditioned on the current scene image and environment metadata derived from expert demonstrations. Inspired by the DeepSeek-R1 learning approach, Robot-R1 samples reasoning-based responses and reinforces those that lead to more accurate predictions. Our experiments show that models trained with Robot-R1 outperform SFT methods on embodied reasoning tasks. Despite having only 7B parameters, Robot-R1 even surpasses GPT-4o on reasoning tasks related to low-level action control, such as spatial and primitive movement reasoning. 

---
# Evaluating the Sensitivity of LLMs to Prior Context 

**Authors**: Robert Hankache, Kingsley Nketia Acheampong, Liang Song, Marek Brynda, Raad Khraishi, Greig A. Cowan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00069)  

**Abstract**: As large language models (LLMs) are increasingly deployed in multi-turn dialogue and other sustained interactive scenarios, it is essential to understand how extended context affects their performance. Popular benchmarks, focusing primarily on single-turn question answering (QA) tasks, fail to capture the effects of multi-turn exchanges. To address this gap, we introduce a novel set of benchmarks that systematically vary the volume and nature of prior context. We evaluate multiple conventional LLMs, including GPT, Claude, and Gemini, across these benchmarks to measure their sensitivity to contextual variations. Our findings reveal that LLM performance on multiple-choice questions can degrade dramatically in multi-turn interactions, with performance drops as large as 73% for certain models. Even highly capable models such as GPT-4o exhibit up to a 32% decrease in accuracy. Notably, the relative performance of larger versus smaller models is not always predictable. Moreover, the strategic placement of the task description within the context can substantially mitigate performance drops, improving the accuracy by as much as a factor of 3.5. These findings underscore the need for robust strategies to design, evaluate, and mitigate context-related sensitivity in LLMs. 

---
# Probing Politico-Economic Bias in Multilingual Large Language Models: A Cultural Analysis of Low-Resource Pakistani Languages 

**Authors**: Afrozah Nadeem, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.00068)  

**Abstract**: Large Language Models (LLMs) are increasingly shaping public discourse, yet their politico-economic biases remain underexamined in non-Western and low-resource multilingual contexts. This paper presents a systematic analysis of political bias in 13 state-of-the-art LLMs across five low-resource languages spoken in Pakistan: Urdu, Punjabi, Sindhi, Balochi, and Pashto. We propose a novel framework that integrates an adapted Political Compass Test (PCT) with a multi-level framing analysis. Our method combines quantitative assessment of political orientation across economic (left-right) and social (libertarian-authoritarian) axes with qualitative analysis of framing through content, style, and emphasis. We further contextualize this analysis by aligning prompts with 11 key socio-political themes relevant to Pakistani society. Our results reveal that LLMs predominantly align with liberal-left values, echoing Western training data influences, but exhibit notable shifts toward authoritarian framing in regional languages, suggesting strong cultural modulation effects. We also identify consistent model-specific bias signatures and language-conditioned variations in ideological expression. These findings show the urgent need for culturally grounded, multilingual bias auditing frameworks. 

---
# Literature Review Of Multi-Agent Debate For Problem-Solving 

**Authors**: Arne Tillmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.00066)  

**Abstract**: Multi-agent large language models (MA-LLMs) are a rapidly growing research area that leverages multiple interacting language agents to tackle complex tasks, outperforming single-agent large language models. This literature review synthesizes the latest research on agent profiles, communication structures, and decision-making processes, drawing insights from both traditional multi-agent systems and state-of-the-art MA-LLM studies. In doing so, it aims to address the lack of direct comparisons in the field, illustrating how factors like scalability, communication structure, and decision-making processes influence MA-LLM performance. By examining frequent practices and outlining current challenges, the review reveals that multi-agent approaches can yield superior results but also face elevated computational costs and under-explored challenges unique to MA-LLM. Overall, these findings provide researchers and practitioners with a roadmap for developing robust and efficient multi-agent AI solutions. 

---
# You Prefer This One, I Prefer Yours: Using Reference Words is Harder Than Vocabulary Words for Humans and Multimodal Language Models 

**Authors**: Dota Tianai Dong, Yifan Luo, Po-Ya Angela Wang, Asli Ozyurek, Paula Rubio-Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.00065)  

**Abstract**: Multimodal language models (MLMs) increasingly communicate in human-like ways, yet their ability to use reference words remains largely overlooked despite their ubiquity in everyday communication. Our study addresses this gap by comparing human and MLM use of three word classes with increasing cognitive demands: vocabulary words, possessive pronouns (`mine' vs `yours'), and demonstrative pronouns (`this one' vs `that one'). Evaluating seven state-of-the-art MLMs against human participants, we observe a clear difficulty hierarchy: while MLMs approach human-level performance on the vocabulary task, they show substantial deficits with possessives and demonstratives. Our analysis reveals these difficulties stem from limitations in perspective-taking and spatial reasoning. Although prompt engineering improved model performance on possessive use, demonstrative use remained well below human-level competence. These findings provide theoretical and empirical evidence that producing grammatical forms requiring pragmatics and social cognition remains a clear challenge in current NLP systems. 

---
# Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling 

**Authors**: Jiayi Zeng, Yizhe Feng, Mengliang He, Wenhui Lei, Wei Zhang, Zeming Liu, Xiaoming Shi, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00064)  

**Abstract**: Large language models (LLMs) have demonstrated significant advancements in error handling. Current error-handling works are performed in a passive manner, with explicit error-handling instructions. However, in real-world scenarios, explicit error-handling instructions are usually unavailable. In this paper, our work identifies this challenge as how to conduct proactive error handling without explicit error handling instructions. To promote further research, this work introduces a new benchmark, termed Mis-prompt, consisting of four evaluation tasks, an error category taxonomy, and a new evaluation dataset. Furthermore, this work analyzes current LLMs' performance on the benchmark, and the experimental results reveal that current LLMs show poor performance on proactive error handling, and SFT on error handling instances improves LLMs' proactive error handling capabilities. The dataset will be publicly available. 

---
# Unraveling SITT: Social Influence Technique Taxonomy and Detection with LLMs 

**Authors**: Wiktoria Mieleszczenko-Kowszewicz, Beata Bajcar, Aleksander Szczęsny, Maciej Markiewicz, Jolanta Babiak, Berenika Dyczek, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2506.00061)  

**Abstract**: In this work we present the Social Influence Technique Taxonomy (SITT), a comprehensive framework of 58 empirically grounded techniques organized into nine categories, designed to detect subtle forms of social influence in textual content. We also investigate the LLMs ability to identify various forms of social influence. Building on interdisciplinary foundations, we construct the SITT dataset -- a 746-dialogue corpus annotated by 11 experts in Polish and translated into English -- to evaluate the ability of LLMs to identify these techniques. Using a hierarchical multi-label classification setup, we benchmark five LLMs, including GPT-4o, Claude 3.5, Llama-3.1, Mixtral, and PLLuM. Our results show that while some models, notably Claude 3.5, achieved moderate success (F1 score = 0.45 for categories), overall performance of models remains limited, particularly for context-sensitive techniques. The findings demonstrate key limitations in current LLMs' sensitivity to nuanced linguistic cues and underscore the importance of domain-specific fine-tuning. This work contributes a novel resource and evaluation example for understanding how LLMs detect, classify, and potentially replicate strategies of social influence in natural dialogues. 

---
# Comparative analysis of privacy-preserving open-source LLMs regarding extraction of diagnostic information from clinical CMR imaging reports 

**Authors**: Sina Amirrajab, Volker Vehof, Michael Bietenbeck, Ali Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.00060)  

**Abstract**: Purpose: We investigated the utilization of privacy-preserving, locally-deployed, open-source Large Language Models (LLMs) to extract diagnostic information from free-text cardiovascular magnetic resonance (CMR) reports. Materials and Methods: We evaluated nine open-source LLMs on their ability to identify diagnoses and classify patients into various cardiac diagnostic categories based on descriptive findings in 109 clinical CMR reports. Performance was quantified using standard classification metrics including accuracy, precision, recall, and F1 score. We also employed confusion matrices to examine patterns of misclassification across models. Results: Most open-source LLMs demonstrated exceptional performance in classifying reports into different diagnostic categories. Google's Gemma2 model achieved the highest average F1 score of 0.98, followed by Qwen2.5:32B and DeepseekR1-32B with F1 scores of 0.96 and 0.95, respectively. All other evaluated models attained average scores above 0.93, with Mistral and DeepseekR1-7B being the only exceptions. The top four LLMs outperformed our board-certified cardiologist (F1 score of 0.94) across all evaluation metrics in analyzing CMR reports. Conclusion: Our findings demonstrate the feasibility of implementing open-source, privacy-preserving LLMs in clinical settings for automated analysis of imaging reports, enabling accurate, fast and resource-efficient diagnostic categorization. 

---
# Prompt Engineer: Analyzing Skill Requirements in the AI Job Market 

**Authors**: An Vu, Jonas Oppenlaender  

**Link**: [PDF](https://arxiv.org/pdf/2506.00058)  

**Abstract**: The rise of large language models (LLMs) has created a new job role: the Prompt Engineer. Despite growing interest in this position, we still do not fully understand what skills this new job role requires or how common these jobs are. We analyzed 20,662 job postings on LinkedIn, including 72 prompt engineer positions, to learn more about this emerging role. We found that prompt engineering is still rare (less than 0.5% of sampled job postings) but has a unique skill profile. Prompt engineers need AI knowledge (22.8%), prompt design skills (18.7%), good communication (21.9%), and creative problem-solving (15.8%) skills. These requirements significantly differ from those of established roles, such as data scientists and machine learning engineers, showing that prompt engineering is becoming its own profession. Our findings help job seekers, employers, and educational institutions in better understanding the emerging field of prompt engineering. 

---
# Improving statistical learning methods via features selection without replacement sampling and random projection 

**Authors**: Sulaiman khan, Muhammad Ahmad, Fida Ullah, Carlos Aguilar Ibañez, José Eduardo Valdez Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.00053)  

**Abstract**: Cancer is fundamentally a genetic disease characterized by genetic and epigenetic alterations that disrupt normal gene expression, leading to uncontrolled cell growth and metastasis. High-dimensional microarray datasets pose challenges for classification models due to the "small n, large p" problem, resulting in overfitting. This study makes three different key contributions: 1) we propose a machine learning-based approach integrating the Feature Selection Without Re-placement (FSWOR) technique and a projection method to improve classification accuracy. 2) We apply the Kendall statistical test to identify the most significant genes from the brain cancer mi-croarray dataset (GSE50161), reducing the feature space from 54,675 to 20,890 genes.3) we apply machine learning models using k-fold cross validation techniques in which our model incorpo-rates ensemble classifiers with LDA projection and Naïve Bayes, achieving a test score of 96%, outperforming existing methods by 9.09%. The results demonstrate the effectiveness of our ap-proach in high-dimensional gene expression analysis, improving classification accuracy while mitigating overfitting. This study contributes to cancer biomarker discovery, offering a robust computational method for analyzing microarray data. 

---
# Using LLMs to Advance the Cognitive Science of Collectives 

**Authors**: Ilia Sucholutsky, Katherine M. Collins, Nori Jacoby, Bill D. Thompson, Robert D. Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.00052)  

**Abstract**: LLMs are already transforming the study of individual cognition, but their application to studying collective cognition has been underexplored. We lay out how LLMs may be able to address the complexity that has hindered the study of collectives and raise possible risks that warrant new methods. 

---
# Rethinking Hybrid Retrieval: When Small Embeddings and LLM Re-ranking Beat Bigger Models 

**Authors**: Arjun Rao, Hanieh Alipour, Nick Pendar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00049)  

**Abstract**: This paper presents a comparison of embedding models in tri-modal hybrid retrieval for Retrieval-Augmented Generation (RAG) systems. We investigate the fusion of dense semantic, sparse lexical, and graph-based embeddings, focusing on the performance of the MiniLM-v6 and BGE-Large architectures. Contrary to conventional assumptions, our results show that the compact MiniLM-v6 outperforms the larger BGE-Large when integrated with LLM-based re-ranking within our tri-modal hybrid framework. Experiments conducted on the SciFact, FIQA, and NFCorpus datasets demonstrate significant improvements in retrieval quality with the MiniLM-v6 configuration. The performance difference is particularly pronounced in agentic re-ranking scenarios, indicating better alignment between MiniLM-v6's embedding space and LLM reasoning. Our findings suggest that embedding model selection for RAG systems should prioritize compatibility with multi-signal fusion and LLM alignment, rather than relying solely on larger models. This approach may reduce computational requirements while improving retrieval accuracy and efficiency. 

---
# Risks of AI-driven product development and strategies for their mitigation 

**Authors**: Jan Göpfert, Jann M. Weinand, Patrick Kuckertz, Noah Pflugradt, Jochen Linßen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00047)  

**Abstract**: Humanity is progressing towards automated product development, a trend that promises faster creation of better products and thus the acceleration of technological progress. However, increasing reliance on non-human agents for this process introduces many risks. This perspective aims to initiate a discussion on these risks and appropriate mitigation strategies. To this end, we outline a set of principles for safer AI-driven product development which emphasize human oversight, accountability, and explainable design, among others. The risk assessment covers both technical risks which affect product quality and safety, and sociotechnical risks which affect society. While AI-driven product development is still in its early stages, this discussion will help balance its opportunities and risks without delaying essential progress in understanding, norm-setting, and regulation. 

---
# The Folly of AI for Age Verification 

**Authors**: Reid McIlroy-Young  

**Link**: [PDF](https://arxiv.org/pdf/2506.00038)  

**Abstract**: In the near future a governmental body will be asked to allow companies to use AI for age verification. If they allow it the resulting system will both be easily circumvented and disproportionately misclassify minorities and low socioeconomic status users. This is predictable by showing that other very similar systems (facial recognition and remote proctoring software) have similar issues despite years of efforts to mitigate their biases. These biases are due to technical limitations both of the AI models themselves and the physical hardware they are running on that will be difficult to overcome below the cost of government ID-based age verification. Thus in, the near future, deploying an AI system for age verification is folly. 

---
# Amadeus-Verbo Technical Report: The powerful Qwen2.5 family models trained in Portuguese 

**Authors**: William Alberto Cruz-Castañeda, Marcellus Amadeus  

**Link**: [PDF](https://arxiv.org/pdf/2506.00019)  

**Abstract**: This report introduces the experience of developing Amadeus Verbo, a family of large language models for Brazilian Portuguese. To handle diverse use cases, Amadeus Verbo includes base-tuned, merged, and instruction-tuned models in sizes of 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B parameters. Thus, the main objective is to show how easy it is to fine-tune foundation models to democratize the open-source development of Brazilian Portuguese LLMs when data and resources are available. Amadeus-Verbo family models are all available at HuggingFace at this https URL. 

---
# MolTextNet: A Two-Million Molecule-Text Dataset for Multimodal Molecular Learning 

**Authors**: Yihan Zhu, Gang Liu, Eric Inae, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00009)  

**Abstract**: Small molecules are essential to drug discovery, and graph-language models hold promise for learning molecular properties and functions from text. However, existing molecule-text datasets are limited in scale and informativeness, restricting the training of generalizable multimodal models. We present MolTextNet, a dataset of 2.5 million high-quality molecule-text pairs designed to overcome these limitations. To construct it, we propose a synthetic text generation pipeline that integrates structural features, computed properties, bioactivity data, and synthetic complexity. Using GPT-4o-mini, we create structured descriptions for 2.5 million molecules from ChEMBL35, with text over 10 times longer than prior datasets. MolTextNet supports diverse downstream tasks, including property prediction and structure retrieval. Pretraining CLIP-style models with Graph Neural Networks and ModernBERT on MolTextNet yields improved performance, highlighting its potential for advancing foundational multimodal modeling in molecular science. Our dataset is available at this https URL. 

---
# Rapid yet accurate Tile-circuit and device modeling for Analog In-Memory Computing 

**Authors**: J. Luquin, C. Mackin, S. Ambrogio, A. Chen, F. Baldi, G. Miralles, M.J. Rasch, J. Büchel, M. Lalwani, W. Ponghiran, P. Solomon, H. Tsai, G.W. Burr, P. Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00004)  

**Abstract**: Analog In-Memory Compute (AIMC) can improve the energy efficiency of Deep Learning by orders of magnitude. Yet analog-domain device and circuit non-idealities -- within the analog ``Tiles'' performing Matrix-Vector Multiply (MVM) operations -- can degrade neural-network task accuracy. We quantify the impact of low-level distortions and noise, and develop a mathematical model for Multiply-ACcumulate (MAC) operations mapped to analog tiles. Instantaneous-current IR-drop (the most significant circuit non-ideality), and ADC quantization effects are fully captured by this model, which can predict MVM tile-outputs both rapidly and accurately, as compared to much slower rigorous circuit simulations. A statistical model of PCM read noise at nanosecond timescales is derived from -- and matched against -- experimental measurements. We integrate these (statistical) device and (deterministic) circuit effects into a PyTorch-based framework to assess the accuracy impact on the BERT and ALBERT Transformer networks. We show that hardware-aware fine-tuning using simple Gaussian noise provides resilience against ADC quantization and PCM read noise effects, but is less effective against IR-drop. This is because IR-drop -- although deterministic -- is non-linear, is changing significantly during the time-integration window, and is ultimately dependent on all the excitations being introduced in parallel into the analog tile. The apparent inability of simple Gaussian noise applied during training to properly prepare a DNN network for IR-drop during inference implies that more complex training approaches -- incorporating advances such as the Tile-circuit model introduced here -- will be critical for resilient deployment of large neural networks onto AIMC hardware. 

---
# Advancing AI-assisted Hardware Design with Hierarchical Decentralized Training and Personalized Inference-Time Optimization 

**Authors**: Hao Mark Chen, Zehuan Zhang, Wanru Zhao, Nicholas Lane, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00002)  

**Abstract**: Recent years have witnessed a significant increase in the adoption of AI techniques to enhance electronic design automation. In particular, the emergence of Large Language Models (LLMs) has sparked significant interest in LLM-assisted hardware design generation, spanning applications from classical digital circuits to quantum computing. Despite substantial progress in this direction, the quality of LLM-generated hardware design still cannot meet the requirements for practical deployment. In this work, we identify three critical challenges hindering the development of LLM-assisted hardware design generation: 1) limited data availability, 2) varied data quality, 3) inadequate inference-time efficiency. To address these fundamental challenges, this paper introduces a two-stage framework for AI-assisted hardware design by exploring decentralized training and personalized inference. In the first stage, we propose to harness private domain design sources through a hierarchical decentralized training mechanism that addresses data-sharing constraints. To mitigate the impact of low-quality data, we identify optimization opportunities in hardware generation tasks, using user-defined metrics for model aggregation. The second stage focuses on client personalization to enhance both speed and quality. We introduce a new metric, Trueput, to analyze LLM-assisted hardware generation efficiency. To optimize Trueput, we implement personalized inference-time acceleration and customized sampling strategies. Evaluating both classical and quantum benchmarks, our experimental results demonstrate that the proposed two-stage framework can significantly improve the model capability for hardware design generation. As orthogonal enhancements to existing methods, our framework can achieve $33\% \sim 50\%$ semantic accuracy improvement and $2.3$ times speedup, depending on the difficulty of the generation tasks. 

---
