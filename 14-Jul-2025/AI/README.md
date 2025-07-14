# System-of-systems Modeling and Optimization: An Integrated Framework for Intermodal Mobility 

**Authors**: Paul Saves, Jasper Bussemaker, Rémi Lafage, Thierry Lefebvre, Nathalie Bartoli, Youssef Diouane, Joseph Morlier  

**Link**: [PDF](https://arxiv.org/pdf/2507.08715)  

**Abstract**: For developing innovative systems architectures, modeling and optimization techniques have been central to frame the architecting process and define the optimization and modeling problems. In this context, for system-of-systems the use of efficient dedicated approaches (often physics-based simulations) is highly recommended to reduce the computational complexity of the targeted applications. However, exploring novel architectures using such dedicated approaches might pose challenges for optimization algorithms, including increased evaluation costs and potential failures. To address these challenges, surrogate-based optimization algorithms, such as Bayesian optimization utilizing Gaussian process models have emerged. 

---
# elsciRL: Integrating Language Solutions into Reinforcement Learning Problem Settings 

**Authors**: Philip Osborne, Danilo S. Carvalho, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2507.08705)  

**Abstract**: We present elsciRL, an open-source Python library to facilitate the application of language solutions on reinforcement learning problems. We demonstrate the potential of our software by extending the Language Adapter with Self-Completing Instruction framework defined in (Osborne, 2024) with the use of LLMs. Our approach can be re-applied to new applications with minimal setup requirements. We provide a novel GUI that allows a user to provide text input for an LLM to generate instructions which it can then self-complete. Empirical results indicate that these instructions \textit{can} improve a reinforcement learning agent's performance. Therefore, we present this work to accelerate the evaluation of language solutions on reward based environments to enable new opportunities for scientific discovery. 

---
# Introspection of Thought Helps AI Agents 

**Authors**: Haoran Sun, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08664)  

**Abstract**: AI Agents rely on Large Language Models (LLMs) and Multimodal-LLMs (MLLMs) to perform interpretation and inference in text and image tasks without post-training, where LLMs and MLLMs play the most critical role and determine the initial ability and limitations of AI Agents. Usually, AI Agents utilize sophisticated prompt engineering and external reasoning framework to obtain a promising interaction with LLMs, e.g., Chain-of-Thought, Iteration of Thought and Image-of-Thought. However, they are still constrained by the inherent limitations of LLM in understanding natural language, and the iterative reasoning process will generate a large amount of inference cost. To this end, we propose a novel AI Agent Reasoning Framework with Introspection of Thought (INoT) by designing a new LLM-Read code in prompt. It enables LLM to execute programmatic dialogue reasoning processes following the code in prompt. Therefore, self-denial and reflection occur within LLM instead of outside LLM, which can reduce token cost effectively. Through our experiments on six benchmarks for three different tasks, the effectiveness of INoT is verified, with an average improvement of 7.95\% in performance, exceeding the baselines. Furthermore, the token cost of INoT is lower on average than the best performing method at baseline by 58.3\%. In addition, we demonstrate the versatility of INoT in image interpretation and inference through verification experiments. 

---
# Leanabell-Prover-V2: Verifier-integrated Reasoning for Formal Theorem Proving via Reinforcement Learning 

**Authors**: Xingguang Ji, Yahui Liu, Qi Wang, Jingyuan Zhang, Yang Yue, Rui Shi, Chenxi Sun, Fuzheng Zhang, Guorui Zhou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2507.08649)  

**Abstract**: We introduce our Leanabell-Prover-V2, a 7B large language models (LLMs) that can produce formal theorem proofs in Lean 4, with verifier-integrated Long Chain-of-Thoughts (CoT). Following our previous work Leanabell-Prover-V1, we continual to choose to posttrain existing strong prover models for further performance improvement. In our V2 version, we mainly upgrade the Reinforcement Learning (RL) with feedback provided by the Lean 4 verifier. Crucially, verifier feedback, such as indicating success or detailing specific errors, allows the LLM to become ``self-aware'' of the correctness of its own reasoning process and learn to reflexively correct errors. Leanabell-Prover-V2 directly optimizes LLM reasoning trajectories with multi-turn verifier interactions, together with feedback token masking for stable RL training and a simple reward strategy. Experiments show that Leanabell-Prover-V2 improves performance by 3.2% (pass@128) with Kimina-Prover-Preview-Distill-7B and 2.0% (pass@128) with DeepSeek-Prover-V2-7B on the MiniF2F test set. The source codes, curated data and models are available at: this https URL. 

---
# Agentic Large Language Models for Conceptual Systems Engineering and Design 

**Authors**: Soheyl Massoudi, Mark Fuge  

**Link**: [PDF](https://arxiv.org/pdf/2507.08619)  

**Abstract**: Early-stage engineering design involves complex, iterative reasoning, yet existing large language model (LLM) workflows struggle to maintain task continuity and generate executable models. We evaluate whether a structured multi-agent system (MAS) can more effectively manage requirements extraction, functional decomposition, and simulator code generation than a simpler two-agent system (2AS). The target application is a solar-powered water filtration system as described in a cahier des charges. We introduce the Design-State Graph (DSG), a JSON-serializable representation that bundles requirements, physical embodiments, and Python-based physics models into graph nodes. A nine-role MAS iteratively builds and refines the DSG, while the 2AS collapses the process to a Generator-Reflector loop. Both systems run a total of 60 experiments (2 LLMs - Llama 3.3 70B vs reasoning-distilled DeepSeek R1 70B x 2 agent configurations x 3 temperatures x 5 seeds). We report a JSON validity, requirement coverage, embodiment presence, code compatibility, workflow completion, runtime, and graph size. Across all runs, both MAS and 2AS maintained perfect JSON integrity and embodiment tagging. Requirement coverage remained minimal (less than 20\%). Code compatibility peaked at 100\% under specific 2AS settings but averaged below 50\% for MAS. Only the reasoning-distilled model reliably flagged workflow completion. Powered by DeepSeek R1 70B, the MAS generated more granular DSGs (average 5-6 nodes) whereas 2AS mode-collapsed. Structured multi-agent orchestration enhanced design detail. Reasoning-distilled LLM improved completion rates, yet low requirements and fidelity gaps in coding persisted. 

---
# Unlocking Speech Instruction Data Potential with Query Rewriting 

**Authors**: Yonghua Hei, Yibo Yan, Shuliang Liu, Huiyu Zhou, Linfeng Zhang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08603)  

**Abstract**: End-to-end Large Speech Language Models~(\textbf{LSLMs}) demonstrate strong potential in response latency and speech comprehension capabilities, showcasing general intelligence across speech understanding tasks. However, the ability to follow speech instructions has not been fully realized due to the lack of datasets and heavily biased training tasks. Leveraging the rich ASR datasets, previous approaches have used Large Language Models~(\textbf{LLMs}) to continue the linguistic information of speech to construct speech instruction datasets. Yet, due to the gap between LLM-generated results and real human responses, the continuation methods further amplify these shortcomings. Given the high costs of collecting and annotating speech instruction datasets by humans, using speech synthesis to construct large-scale speech instruction datasets has become a balanced and robust alternative. Although modern Text-To-Speech~(\textbf{TTS}) models have achieved near-human-level synthesis quality, it is challenging to appropriately convert out-of-distribution text instruction to speech due to the limitations of the training data distribution in TTS models. To address this issue, we propose a query rewriting framework with multi-LLM knowledge fusion, employing multiple agents to annotate and validate the synthesized speech, making it possible to construct high-quality speech instruction datasets without relying on human annotation. Experiments show that this method can transform text instructions into distributions more suitable for TTS models for speech synthesis through zero-shot rewriting, increasing data usability from 72\% to 93\%. It also demonstrates unique advantages in rewriting tasks that require complex knowledge and context-related abilities. 

---
# Large Multi-modal Model Cartographic Map Comprehension for Textual Locality Georeferencing 

**Authors**: Kalana Wijegunarathna, Kristin Stock, Christopher B. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2507.08575)  

**Abstract**: Millions of biological sample records collected in the last few centuries archived in natural history collections are un-georeferenced. Georeferencing complex locality descriptions associated with these collection samples is a highly labour-intensive task collection agencies struggle with. None of the existing automated methods exploit maps that are an essential tool for georeferencing complex relations. We present preliminary experiments and results of a novel method that exploits multi-modal capabilities of recent Large Multi-Modal Models (LMM). This method enables the model to visually contextualize spatial relations it reads in the locality description. We use a grid-based approach to adapt these auto-regressive models for this task in a zero-shot setting. Our experiments conducted on a small manually annotated dataset show impressive results for our approach ($\sim$1 km Average distance error) compared to uni-modal georeferencing with Large Language Models and existing georeferencing tools. The paper also discusses the findings of the experiments in light of an LMM's ability to comprehend fine-grained maps. Motivated by these results, a practical framework is proposed to integrate this method into a georeferencing workflow. 

---
# A Multi-granularity Concept Sparse Activation and Hierarchical Knowledge Graph Fusion Framework for Rare Disease Diagnosis 

**Authors**: Mingda Zhang, Na Zhao, Jianglong Qin, Guoyu Ye, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08529)  

**Abstract**: Despite advances from medical large language models in healthcare, rare-disease diagnosis remains hampered by insufficient knowledge-representation depth, limited concept understanding, and constrained clinical reasoning. We propose a framework that couples multi-granularity sparse activation of medical concepts with a hierarchical knowledge graph. Four complementary matching algorithms, diversity control, and a five-level fallback strategy enable precise concept activation, while a three-layer knowledge graph (taxonomy, clinical features, instances) provides structured, up-to-date context. Experiments on the BioASQ rare-disease QA set show BLEU gains of 0.09, ROUGE gains of 0.05, and accuracy gains of 0.12, with peak accuracy of 0.89 approaching the 0.90 clinical threshold. Expert evaluation confirms improvements in information quality, reasoning, and professional expression, suggesting our approach shortens the "diagnostic odyssey" for rare-disease patients. 

---
# From Language to Logic: A Bi-Level Framework for Structured Reasoning 

**Authors**: Keying Yang, Hao Wang, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08501)  

**Abstract**: Structured reasoning over natural language inputs remains a core challenge in artificial intelligence, as it requires bridging the gap between unstructured linguistic expressions and formal logical representations. In this paper, we propose a novel \textbf{bi-level framework} that maps language to logic through a two-stage process: high-level task abstraction and low-level logic generation. At the upper level, a large language model (LLM) parses natural language queries into intermediate structured representations specifying the problem type, objectives, decision variables, and symbolic constraints. At the lower level, the LLM uses these representations to generate symbolic workflows or executable reasoning programs for accurate and interpretable decision making. The framework supports modular reasoning, enforces explicit constraints, and generalizes across domains such as mathematical problem solving, question answering, and logical inference. We further optimize the framework with an end-to-end {bi-level} optimization approach that jointly refines both the high-level abstraction and low-level logic generation stages. Experiments on multiple realistic reasoning benchmarks demonstrate that our approach significantly outperforms existing baselines in accuracy, with accuracy gains reaching as high as 40\%. Moreover, the bi-level design enhances transparency and error traceability, offering a promising step toward trustworthy and systematic reasoning with LLMs. 

---
# Why this and not that? A Logic-based Framework for Contrastive Explanations 

**Authors**: Tobias Geibinger, Reijo Jaakkola, Antti Kuusisto, Xinghan Liu, Miikka Vilander  

**Link**: [PDF](https://arxiv.org/pdf/2507.08454)  

**Abstract**: We define several canonical problems related to contrastive explanations, each answering a question of the form ''Why P but not Q?''. The problems compute causes for both P and Q, explicitly comparing their differences. We investigate the basic properties of our definitions in the setting of propositional logic. We show, inter alia, that our framework captures a cardinality-minimal version of existing contrastive explanations in the literature. Furthermore, we provide an extensive analysis of the computational complexities of the problems. We also implement the problems for CNF-formulas using answer set programming and present several examples demonstrating how they work in practice. 

---
# Multi-Agent LLMs as Ethics Advocates in AI-Based Systems 

**Authors**: Asma Yamani, Malak Baslyman, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2507.08392)  

**Abstract**: Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products. 

---
# M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning 

**Authors**: Inclusion AI, Fudong Wang, Jiajia Liu, Jingdong Chen, Jun Zhou, Kaixiang Ji, Lixiang Ru, Qingpei Guo, Ruobing Zheng, Tianqi Li, Yi Yuan, Yifan Mao, Yuting Xiao, Ziping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08306)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs), particularly through Reinforcement Learning with Verifiable Rewards (RLVR), have significantly enhanced their reasoning abilities. However, a critical gap persists: these models struggle with dynamic spatial interactions, a capability essential for real-world applications. To bridge this gap, we introduce M2-Reasoning-7B, a model designed to excel in both general and spatial reasoning. Our approach integrates two key innovations: (1) a novel data pipeline that generates 294.2K high-quality data samples (168K for cold-start fine-tuning and 126.2K for RLVR), which feature logically coherent reasoning trajectories and have undergone comprehensive assessment; and (2) a dynamic multi-task training strategy with step-wise optimization to mitigate conflicts between data, and task-specific rewards for delivering tailored incentive signals. This combination of curated data and advanced training allows M2-Reasoning-7B to set a new state-of-the-art (SOTA) across 8 benchmarks, showcasing superior performance in both general and spatial reasoning domains. 

---
# Agent Safety Alignment via Reinforcement Learning 

**Authors**: Zeyang Sha, Hanling Tian, Zhuoer Xu, Shiwen Cui, Changhua Meng, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08270)  

**Abstract**: The emergence of autonomous Large Language Model (LLM) agents capable of tool usage has introduced new safety risks that go beyond traditional conversational misuse. These agents, empowered to execute external functions, are vulnerable to both user-initiated threats (e.g., adversarial prompts) and tool-initiated threats (e.g., malicious outputs from compromised tools). In this paper, we propose the first unified safety-alignment framework for tool-using agents, enabling models to handle both channels of threat via structured reasoning and sandboxed reinforcement learning. We introduce a tri-modal taxonomy, including benign, malicious, and sensitive for both user prompts and tool responses, and define a policy-driven decision model. Our framework employs a custom-designed sandbox environment that simulates real-world tool execution and allows fine-grained reward shaping. Through extensive evaluations on public and self-built benchmarks, including Agent SafetyBench, InjecAgent, and BFCL, we demonstrate that our safety-aligned agents significantly improve resistance to security threats while preserving strong utility on benign tasks. Our results show that safety and effectiveness can be jointly optimized, laying the groundwork for trustworthy deployment of autonomous LLM agents. 

---
# Abductive Computational Systems: Creative Abduction and Future Directions 

**Authors**: Abhinav Sood, Kazjon Grace, Stephen Wan, Cecile Paris  

**Link**: [PDF](https://arxiv.org/pdf/2507.08264)  

**Abstract**: Abductive reasoning, reasoning for inferring explanations for observations, is often mentioned in scientific, design-related and artistic contexts, but its understanding varies across these domains. This paper reviews how abductive reasoning is discussed in epistemology, science and design, and then analyses how various computational systems use abductive reasoning. Our analysis shows that neither theoretical accounts nor computational implementations of abductive reasoning adequately address generating creative hypotheses. Theoretical frameworks do not provide a straightforward model for generating creative abductive hypotheses, computational systems largely implement syllogistic forms of abductive reasoning. We break down abductive computational systems into components and conclude by identifying specific directions for future research that could advance the state of creative abductive reasoning in computational systems. 

---
# Giving AI Agents Access to Cryptocurrency and Smart Contracts Creates New Vectors of AI Harm 

**Authors**: Bill Marino, Ari Juels  

**Link**: [PDF](https://arxiv.org/pdf/2507.08249)  

**Abstract**: There is growing interest in giving AI agents access to cryptocurrencies as well as to the smart contracts that transact them. But doing so, this position paper argues, could lead to formidable new vectors of AI harm. To support this argument, we first examine the unique properties of cryptocurrencies and smart contracts that could lead to these new vectors of harm. Next, we describe each of these new vectors of harm in detail. Finally, we conclude with a call for more technical research aimed at preventing and mitigating these harms and, thereby making it safer to endow AI agents with cryptocurrencies and smart contracts. 

---
# Quantum Federated Learning for Multimodal Data: A Modality-Agnostic Approach 

**Authors**: Atit Pokharel, Ratun Rahman, Thomas Morris, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08217)  

**Abstract**: Quantum federated learning (QFL) has been recently introduced to enable a distributed privacy-preserving quantum machine learning (QML) model training across quantum processors (clients). Despite recent research efforts, existing QFL frameworks predominantly focus on unimodal systems, limiting their applicability to real-world tasks that often naturally involve multiple modalities. To fill this significant gap, we present for the first time a novel multimodal approach specifically tailored for the QFL setting with the intermediate fusion using quantum entanglement. Furthermore, to address a major bottleneck in multimodal QFL, where the absence of certain modalities during training can degrade model performance, we introduce a Missing Modality Agnostic (MMA) mechanism that isolates untrained quantum circuits, ensuring stable training without corrupted states. Simulation results demonstrate that the proposed multimodal QFL method with MMA yields an improvement in accuracy of 6.84% in independent and identically distributed (IID) and 7.25% in non-IID data distributions compared to the state-of-the-art methods. 

---
# Grounding Methods for Neural-Symbolic AI 

**Authors**: Rodrigo Castellano Ontiveros, Francesco Giannini, Marco Gori, Giuseppe Marra, Michelangelo Diligenti  

**Link**: [PDF](https://arxiv.org/pdf/2507.08216)  

**Abstract**: A large class of Neural-Symbolic (NeSy) methods employs a machine learner to process the input entities, while relying on a reasoner based on First-Order Logic to represent and process more complex relationships among the entities. A fundamental role for these methods is played by the process of logic grounding, which determines the relevant substitutions for the logic rules using a (sub)set of entities. Some NeSy methods use an exhaustive derivation of all possible substitutions, preserving the full expressive power of the logic knowledge. This leads to a combinatorial explosion in the number of ground formulas to consider and, therefore, strongly limits their scalability. Other methods rely on heuristic-based selective derivations, which are generally more computationally efficient, but lack a justification and provide no guarantees of preserving the information provided to and returned by the reasoner. Taking inspiration from multi-hop symbolic reasoning, this paper proposes a parametrized family of grounding methods generalizing classic Backward Chaining. Different selections within this family allow us to obtain commonly employed grounding methods as special cases, and to control the trade-off between expressiveness and scalability of the reasoner. The experimental results show that the selection of the grounding criterion is often as important as the NeSy method itself. 

---
# From Curiosity to Competence: How World Models Interact with the Dynamics of Exploration 

**Authors**: Fryderyk Mantiuk, Hanqi Zhou, Charley M. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08210)  

**Abstract**: What drives an agent to explore the world while also maintaining control over the environment? From a child at play to scientists in the lab, intelligent agents must balance curiosity (the drive to seek knowledge) with competence (the drive to master and control the environment). Bridging cognitive theories of intrinsic motivation with reinforcement learning, we ask how evolving internal representations mediate the trade-off between curiosity (novelty or information gain) and competence (empowerment). We compare two model-based agents using handcrafted state abstractions (Tabular) or learning an internal world model (Dreamer). The Tabular agent shows curiosity and competence guide exploration in distinct patterns, while prioritizing both improves exploration. The Dreamer agent reveals a two-way interaction between exploration and representation learning, mirroring the developmental co-evolution of curiosity and competence. Our findings formalize adaptive exploration as a balance between pursuing the unknown and the controllable, offering insights for cognitive theories and efficient reinforcement learning. 

---
# Reasoning and Behavioral Equilibria in LLM-Nash Games: From Mindsets to Actions 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08208)  

**Abstract**: We introduce the LLM-Nash framework, a game-theoretic model where agents select reasoning prompts to guide decision-making via Large Language Models (LLMs). Unlike classical games that assume utility-maximizing agents with full rationality, this framework captures bounded rationality by modeling the reasoning process explicitly. Equilibrium is defined over the prompt space, with actions emerging as the behavioral output of LLM inference. This approach enables the study of cognitive constraints, mindset expressiveness, and epistemic learning. Through illustrative examples, we show how reasoning equilibria can diverge from classical Nash outcomes, offering a new foundation for strategic interaction in LLM-enabled systems. 

---
# A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking 

**Authors**: Zhengye Han, Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08207)  

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking. 

---
# TableReasoner: Advancing Table Reasoning Framework with Large Language Models 

**Authors**: Sishi Xiong, Dakai Wang, Yu Zhao, Jie Zhang, Changzai Pan, Haowei He, Xiangyu Li, Wenhan Chang, Zhongjiang He, Shuangyong Song, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08046)  

**Abstract**: The paper presents our system developed for table question answering (TQA). TQA tasks face challenges due to the characteristics of real-world tabular data, such as large size, incomplete column semantics, and entity ambiguity. To address these issues, we propose a large language model (LLM)-powered and programming-based table reasoning framework, named TableReasoner. It models a table using the schema that combines structural and semantic representations, enabling holistic understanding and efficient processing of large tables. We design a multi-step schema linking plan to derive a focused table schema that retains only query-relevant information, eliminating ambiguity and alleviating hallucinations. This focused table schema provides precise and sufficient table details for query refinement and programming. Furthermore, we integrate the reasoning workflow into an iterative thinking architecture, allowing incremental cycles of thinking, reasoning and reflection. Our system achieves first place in both subtasks of SemEval-2025 Task 8. 

---
# Human Creativity and AI 

**Authors**: Shengyi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.08001)  

**Abstract**: With the advancement of science and technology, the philosophy of creativity has undergone significant reinterpretation. This paper investigates contemporary research in the fields of psychology, cognitive neuroscience, and the philosophy of creativity, particularly in the context of the development of artificial intelligence (AI) techniques. It aims to address the central question: Can AI exhibit creativity? The paper reviews the historical perspectives on the philosophy of creativity and explores the influence of psychological advancements on the study of creativity. Furthermore, it analyzes various definitions of creativity and examines the responses of naturalism and cognitive neuroscience to the concept of creativity. 

---
# Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective 

**Authors**: Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, Hao Luo, Jiasheng Tang, Fan Wang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08801)  

**Abstract**: Autoregressive large language models (LLMs) have unified a vast range of language tasks, inspiring preliminary efforts in autoregressive video generation. Existing autoregressive video generators either diverge from standard LLM architectures, depend on bulky external text encoders, or incur prohibitive latency due to next-token decoding. In this paper, we introduce Lumos-1, an autoregressive video generator that retains the LLM architecture with minimal architectural modifications. To inject spatiotemporal correlations in LLMs, we identify the efficacy of incorporating 3D RoPE and diagnose its imbalanced frequency spectrum ranges. Therefore, we propose MM-RoPE, a RoPE scheme that preserves the original textual RoPE while providing comprehensive frequency spectra and scaled 3D positions for modeling multimodal spatiotemporal data. Moreover, Lumos-1 resorts to a token dependency strategy that obeys intra-frame bidirectionality and inter-frame temporal causality. Based on this dependency strategy, we identify the issue of frame-wise loss imbalance caused by spatial information redundancy and solve it by proposing Autoregressive Discrete Diffusion Forcing (AR-DF). AR-DF introduces temporal tube masking during training with a compatible inference-time masking policy to avoid quality degradation. By using memory-efficient training techniques, we pre-train Lumos-1 on only 48 GPUs, achieving performance comparable to EMU3 on GenEval, COSMOS-Video2World on VBench-I2V, and OpenSoraPlan on VBench-T2V. Code and models are available at this https URL. 

---
# NeuralOS: Towards Simulating Operating Systems via Neural Generative Models 

**Authors**: Luke Rivard, Sun Sun, Hongyu Guo, Wenhu Chen, Yuntian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08800)  

**Abstract**: We introduce NeuralOS, a neural framework that simulates graphical user interfaces (GUIs) of operating systems by directly predicting screen frames in response to user inputs such as mouse movements, clicks, and keyboard events. NeuralOS combines a recurrent neural network (RNN), which tracks computer state, with a diffusion-based neural renderer that generates screen images. The model is trained on a large-scale dataset of Ubuntu XFCE recordings, which include both randomly generated interactions and realistic interactions produced by AI agents. Experiments show that NeuralOS successfully renders realistic GUI sequences, accurately captures mouse interactions, and reliably predicts state transitions like application launches. Although modeling fine-grained keyboard interactions precisely remains challenging, NeuralOS offers a step toward creating fully adaptive, generative neural interfaces for future human-computer interaction systems. 

---
# KV Cache Steering for Inducing Reasoning in Small Language Models 

**Authors**: Max Belitsky, Dawid J. Kopiczko, Michael Dorkenwald, M. Jehanzeb Mirza, Cees G. M. Snoek, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2507.08799)  

**Abstract**: We propose cache steering, a lightweight method for implicit steering of language models via a one-shot intervention applied directly to the key-value cache. To validate its effectiveness, we apply cache steering to induce chain-of-thought reasoning in small language models. Our approach leverages GPT-4o-generated reasoning traces to construct steering vectors that shift model behavior toward more explicit, multi-step reasoning without fine-tuning or prompt modifications. Experimental evaluations on diverse reasoning benchmarks demonstrate that cache steering improves both the qualitative structure of model reasoning and quantitative task performance. Compared to prior activation steering techniques that require continuous interventions, our one-shot cache steering offers substantial advantages in terms of hyperparameter stability, inference-time efficiency, and ease of integration, making it a more robust and practical solution for controlled generation. 

---
# Optimistic Exploration for Risk-Averse Constrained Reinforcement Learning 

**Authors**: James McCarthy, Radu Marinescu, Elizabeth Daly, Ivana Dusparic  

**Link**: [PDF](https://arxiv.org/pdf/2507.08793)  

**Abstract**: Risk-averse Constrained Reinforcement Learning (RaCRL) aims to learn policies that minimise the likelihood of rare and catastrophic constraint violations caused by an environment's inherent randomness. In general, risk-aversion leads to conservative exploration of the environment which typically results in converging to sub-optimal policies that fail to adequately maximise reward or, in some cases, fail to achieve the goal. In this paper, we propose an exploration-based approach for RaCRL called Optimistic Risk-averse Actor Critic (ORAC), which constructs an exploratory policy by maximising a local upper confidence bound of the state-action reward value function whilst minimising a local lower confidence bound of the risk-averse state-action cost value function. Specifically, at each step, the weighting assigned to the cost value is increased or decreased if it exceeds or falls below the safety constraint value. This way the policy is encouraged to explore uncertain regions of the environment to discover high reward states whilst still satisfying the safety constraints. Our experimental results demonstrate that the ORAC approach prevents convergence to sub-optimal policies and improves significantly the reward-cost trade-off in various continuous control tasks such as Safety-Gymnasium and a complex building energy management environment CityLearn. 

---
# On Barriers to Archival Audio Processing 

**Authors**: Peter Sullivan, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2507.08768)  

**Abstract**: In this study, we leverage a unique UNESCO collection of mid-20th century radio recordings to probe the robustness of modern off-the-shelf language identification (LID) and speaker recognition (SR) methods, especially with respect to the impact of multilingual speakers and cross-age recordings. Our findings suggest that LID systems, such as Whisper, are increasingly adept at handling second-language and accented speech. However, speaker embeddings remain a fragile component of speech processing pipelines that is prone to biases related to the channel, age, and language. Issues which will need to be overcome should archives aim to employ SR methods for speaker indexing. 

---
# A Hybrid Multi-Well Hopfield-CNN with Feature Extraction and K-Means for MNIST Classification 

**Authors**: Ahmed Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2507.08766)  

**Abstract**: This study presents a hybrid model for classifying handwritten digits in the MNIST dataset, combining convolutional neural networks (CNNs) with a multi-well Hopfield network. The approach employs a CNN to extract high-dimensional features from input images, which are then clustered into class-specific prototypes using k-means clustering. These prototypes serve as attractors in a multi-well energy landscape, where a Hopfield network performs classification by minimizing an energy function that balances feature similarity and class this http URL model's design enables robust handling of intraclass variability, such as diverse handwriting styles, while providing an interpretable framework through its energy-based decision process. Through systematic optimization of the CNN architecture and the number of wells, the model achieves a high test accuracy of 99.2% on 10,000 MNIST images, demonstrating its effectiveness for image classification tasks. The findings highlight the critical role of deep feature extraction and sufficient prototype coverage in achieving high performance, with potential for broader applications in pattern recognition. 

---
# Compress Any Segment Anything Model (SAM) 

**Authors**: Juntong Fan, Zhiwei Hao, Jianqiang Shen, Shang-Ling Jui, Yi Zhang, Jing-Xiao Liao, Feng-Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.08765)  

**Abstract**: Due to the excellent performance in yielding high-quality, zero-shot segmentation, Segment Anything Model (SAM) and its variants have been widely applied in diverse scenarios such as healthcare and intelligent manufacturing. Therefore, effectively compressing SAMs has become an increasingly pressing practical need. In this study, we propose Birkhoff, a novel data-free compression algorithm for SAM and its variants. Unlike quantization, pruning, distillation, and other compression methods, Birkhoff embodies versatility across model types, agility in deployment, faithfulness to the original model, and compactness in model size. Specifically, Birkhoff introduces a novel compression algorithm: Hyper-Compression, whose core principle is to find a dense trajectory to turn a high-dimensional parameter vector into a low-dimensional scalar. Furthermore, Birkhoff designs a dedicated linear layer operator, HyperLinear, to fuse decompression and matrix multiplication to significantly accelerate inference of the compressed SAMs. Extensive experiments on 18 SAMs in the COCO, LVIS, and SA-1B datasets show that Birkhoff performs consistently and competitively in compression time, compression ratio, post-compression performance, and inference speed. For example, Birkhoff can achieve a compression ratio of 5.17x on SAM2-B, with less than 1% performance drop without using any fine-tuning data. Moreover, the compression is finished within 60 seconds for all models. 

---
# Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data 

**Authors**: Jeonghye Kim, Yongjae Shin, Whiyoung Jung, Sunghoon Hong, Deunsol Yoon, Youngchul Sung, Kanghoon Lee, Woohyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.08761)  

**Abstract**: Reinforcement learning with offline data suffers from Q-value extrapolation errors. To address this issue, we first demonstrate that linear extrapolation of the Q-function beyond the data range is particularly problematic. To mitigate this, we propose guiding the gradual decrease of Q-values outside the data range, which is achieved through reward scaling with layer normalization (RS-LN) and a penalization mechanism for infeasible actions (PA). By combining RS-LN and PA, we develop a new algorithm called PARS. We evaluate PARS across a range of tasks, demonstrating superior performance compared to state-of-the-art algorithms in both offline training and online fine-tuning on the D4RL benchmark, with notable success in the challenging AntMaze Ultra task. 

---
# Geo-ORBIT: A Federated Digital Twin Framework for Scene-Adaptive Lane Geometry Detection 

**Authors**: Rei Tamaru, Pei Li, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2507.08743)  

**Abstract**: Digital Twins (DT) have the potential to transform traffic management and operations by creating dynamic, virtual representations of transportation systems that sense conditions, analyze operations, and support decision-making. A key component for DT of the transportation system is dynamic roadway geometry sensing. However, existing approaches often rely on static maps or costly sensors, limiting scalability and adaptability. Additionally, large-scale DTs that collect and analyze data from multiple sources face challenges in privacy, communication, and computational efficiency. To address these challenges, we introduce Geo-ORBIT (Geometrical Operational Roadway Blueprint with Integrated Twin), a unified framework that combines real-time lane detection, DT synchronization, and federated meta-learning. At the core of Geo-ORBIT is GeoLane, a lightweight lane detection model that learns lane geometries from vehicle trajectory data using roadside cameras. We extend this model through Meta-GeoLane, which learns to personalize detection parameters for local entities, and FedMeta-GeoLane, a federated learning strategy that ensures scalable and privacy-preserving adaptation across roadside deployments. Our system is integrated with CARLA and SUMO to create a high-fidelity DT that renders highway scenarios and captures traffic flows in real-time. Extensive experiments across diverse urban scenes show that FedMeta-GeoLane consistently outperforms baseline and meta-learning approaches, achieving lower geometric error and stronger generalization to unseen locations while drastically reducing communication overhead. This work lays the foundation for flexible, context-aware infrastructure modeling in DTs. The framework is publicly available at this https URL. 

---
# Adaptive Nonlinear Vector Autoregression: Robust Forecasting for Noisy Chaotic Time Series 

**Authors**: Azimov Sherkhon, Susana Lopez-Moreno, Eric Dolores-Cuenca, Sieun Lee, Sangil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.08738)  

**Abstract**: Nonlinear vector autoregression (NVAR) and reservoir computing (RC) have shown promise in forecasting chaotic dynamical systems, such as the Lorenz-63 model and El Nino-Southern Oscillation. However, their reliance on fixed nonlinearities - polynomial expansions in NVAR or random feature maps in RC - limits their adaptability to high noise or real-world data. These methods also scale poorly in high-dimensional settings due to costly matrix inversion during readout computation. We propose an adaptive NVAR model that combines delay-embedded linear inputs with features generated by a shallow, learnable multi-layer perceptron (MLP). The MLP and linear readout are jointly trained using gradient-based optimization, enabling the model to learn data-driven nonlinearities while preserving a simple readout structure. Unlike standard NVAR, our approach avoids the need for an exhaustive and sensitive grid search over ridge and delay parameters. Instead, tuning is restricted to neural network hyperparameters, improving scalability. Initial experiments on chaotic systems tested under noise-free and synthetically noisy conditions showed that the adaptive model outperformed the standard NVAR in predictive accuracy and showed robust forecasting under noisy conditions with a lower observation frequency. 

---
# Catastrophic Forgetting Mitigation Through Plateau Phase Activity Profiling 

**Authors**: Idan Mashiach, Oren Glickman, Tom Tirer  

**Link**: [PDF](https://arxiv.org/pdf/2507.08736)  

**Abstract**: Catastrophic forgetting in deep neural networks occurs when learning new tasks degrades performance on previously learned tasks due to knowledge overwriting. Among the approaches to mitigate this issue, regularization techniques aim to identify and constrain "important" parameters to preserve previous knowledge. In the highly nonconvex optimization landscape of deep learning, we propose a novel perspective: tracking parameters during the final training plateau is more effective than monitoring them throughout the entire training process. We argue that parameters that exhibit higher activity (movement and variability) during this plateau reveal directions in the loss landscape that are relatively flat, making them suitable for adaptation to new tasks while preserving knowledge from previous ones. Our comprehensive experiments demonstrate that this approach achieves superior performance in balancing catastrophic forgetting mitigation with strong performance on newly learned tasks. 

---
# Dually Hierarchical Drift Adaptation for Online Configuration Performance Learning 

**Authors**: Zezhen Xiang, Jingzhi Gong, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08730)  

**Abstract**: Modern configurable software systems need to learn models that correlate configuration and performance. However, when the system operates in dynamic environments, the workload variations, hardware changes, and system updates will inevitably introduce concept drifts at different levels - global drifts, which reshape the performance landscape of the entire configuration space; and local drifts, which only affect certain sub-regions of that space. As such, existing offline and transfer learning approaches can struggle to adapt to these implicit and unpredictable changes in real-time, rendering configuration performance learning challenging. To address this, we propose DHDA, an online configuration performance learning framework designed to capture and adapt to these drifts at different levels. The key idea is that DHDA adapts to both the local and global drifts using dually hierarchical adaptation: at the upper level, we redivide the data into different divisions, within each of which the local model is retrained, to handle global drifts only when necessary. At the lower level, the local models of the divisions can detect local drifts and adapt themselves asynchronously. To balance responsiveness and efficiency, DHDA combines incremental updates with periodic full retraining to minimize redundant computation when no drifts are detected. Through evaluating eight software systems and against state-of-the-art approaches, we show that DHDA achieves considerably better accuracy and can effectively adapt to drifts with up to 2x improvements, while incurring reasonable overhead and is able to improve different local models in handling concept drift. 

---
# Monitoring Risks in Test-Time Adaptation 

**Authors**: Mona Schirmer, Metod Jazbec, Christian A. Naesseth, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2507.08721)  

**Abstract**: Encountering shifted data at test time is a ubiquitous challenge when deploying predictive models. Test-time adaptation (TTA) methods address this issue by continuously adapting a deployed model using only unlabeled test data. While TTA can extend the model's lifespan, it is only a temporary solution. Eventually the model might degrade to the point that it must be taken offline and retrained. To detect such points of ultimate failure, we propose pairing TTA with risk monitoring frameworks that track predictive performance and raise alerts when predefined performance criteria are violated. Specifically, we extend existing monitoring tools based on sequential testing with confidence sequences to accommodate scenarios in which the model is updated at test time and no test labels are available to estimate the performance metrics of interest. Our extensions unlock the application of rigorous statistical risk monitoring to TTA, and we demonstrate the effectiveness of our proposed TTA monitoring framework across a representative set of datasets, distribution shift types, and TTA methods. 

---
# Multilingual Multimodal Software Developer for Code Generation 

**Authors**: Linzheng Chai, Jian Yang, Shukai Liu, Wei Zhang, Liran Wang, Ke Jin, Tao Sun, Congnan Liu, Chenchen Zhang, Hualei Zhu, Jiaheng Liu, Xianjie Wu, Ge Zhang, Tianyu Liu, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08719)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has significantly improved code generation, yet most models remain text-only, neglecting crucial visual aids like diagrams and flowcharts used in real-world software development. To bridge this gap, we introduce MM-Coder, a Multilingual Multimodal software developer. MM-Coder integrates visual design inputs-Unified Modeling Language (UML) diagrams and flowcharts (termed Visual Workflow)-with textual instructions to enhance code generation accuracy and architectural alignment. To enable this, we developed MMc-Instruct, a diverse multimodal instruction-tuning dataset including visual-workflow-based code generation, allowing MM-Coder to synthesize textual and graphical information like human developers, distinct from prior work on narrow tasks. Furthermore, we introduce MMEval, a new benchmark for evaluating multimodal code generation, addressing existing text-only limitations. Our evaluations using MMEval highlight significant remaining challenges for models in precise visual information capture, instruction following, and advanced programming knowledge. Our work aims to revolutionize industrial programming by enabling LLMs to interpret and implement complex specifications conveyed through both text and visual designs. 

---
# KG-Attention: Knowledge Graph-Guided Attention at Test-Time via Bidirectional Information Aggregation 

**Authors**: Songlin Zhai, Guilin Qi, Yuan Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08704)  

**Abstract**: Knowledge graphs (KGs) play a critical role in enhancing large language models (LLMs) by introducing structured and grounded knowledge into the learning process. However, most existing KG-enhanced approaches rely on parameter-intensive fine-tuning, which risks catastrophic forgetting and degrades the pretrained model's generalization. Moreover, they exhibit limited adaptability to real-time knowledge updates due to their static integration frameworks. To address these issues, we introduce the first test-time KG-augmented framework for LLMs, built around a dedicated knowledge graph-guided attention (KGA) module that enables dynamic knowledge fusion without any parameter updates. The proposed KGA module augments the standard self-attention mechanism with two synergistic pathways: outward and inward aggregation. Specifically, the outward pathway dynamically integrates external knowledge into input representations via input-driven KG fusion. This inward aggregation complements the outward pathway by refining input representations through KG-guided filtering, suppressing task-irrelevant signals and amplifying knowledge-relevant patterns. Importantly, while the outward pathway handles knowledge fusion, the inward path selects the most relevant triples and feeds them back into the fusion process, forming a closed-loop enhancement mechanism. By synergistically combining these two pathways, the proposed method supports real-time knowledge fusion exclusively at test-time, without any parameter modification. Extensive experiments on five benchmarks verify the comparable knowledge fusion performance of KGA. 

---
# ONION: A Multi-Layered Framework for Participatory ER Design 

**Authors**: Viktoriia Makovska, George Fletcher, Julia Stoyanovich  

**Link**: [PDF](https://arxiv.org/pdf/2507.08702)  

**Abstract**: We present ONION, a multi-layered framework for participatory Entity-Relationship (ER) modeling that integrates insights from design justice, participatory AI, and conceptual modeling. ONION introduces a five-stage methodology: Observe, Nurture, Integrate, Optimize, Normalize. It supports progressive abstraction from unstructured stakeholder input to structured ER diagrams.
Our approach aims to reduce designer bias, promote inclusive participation, and increase transparency through the modeling process. We evaluate ONION through real-world workshops focused on sociotechnical systems in Ukraine, highlighting how diverse stakeholder engagement leads to richer data models and deeper mutual understanding. Early results demonstrate ONION's potential to host diversity in early-stage data modeling. We conclude with lessons learned, limitations and challenges involved in scaling and refining the framework for broader adoption. 

---
# A Personalised Formal Verification Framework for Monitoring Activities of Daily Living of Older Adults Living Independently in Their Homes 

**Authors**: Ricardo Contreras, Filip Smola, Nuša Farič, Jiawei Zheng, Jane Hillston, Jacques D. Fleuriot  

**Link**: [PDF](https://arxiv.org/pdf/2507.08701)  

**Abstract**: There is an imperative need to provide quality of life to a growing population of older adults living independently. Personalised solutions that focus on the person and take into consideration their preferences and context are key. In this work, we introduce a framework for representing and reasoning about the Activities of Daily Living of older adults living independently at home. The framework integrates data from sensors and contextual information that aggregates semi-structured interviews, home layouts and sociological observations from the participants. We use these data to create formal models, personalised for each participant according to their preferences and context. We formulate requirements that are specific to each individual as properties encoded in Linear Temporal Logic and use a model checker to verify whether each property is satisfied by the model. When a property is violated, a counterexample is generated giving the cause of the violation. We demonstrate the framework's generalisability by applying it to different participants, highlighting its potential to enhance the safety and well-being of older adults ageing in place. 

---
# MoSAiC: Multi-Modal Multi-Label Supervision-Aware Contrastive Learning for Remote Sensing 

**Authors**: Debashis Gupta, Aditi Golder, Rongkhun Zhu, Kangning Cui, Wei Tang, Fan Yang, Ovidiu Csillik, Sarra Alaqahtani, V. Paul Pauca  

**Link**: [PDF](https://arxiv.org/pdf/2507.08683)  

**Abstract**: Contrastive learning (CL) has emerged as a powerful paradigm for learning transferable representations without the reliance on large labeled datasets. Its ability to capture intrinsic similarities and differences among data samples has led to state-of-the-art results in computer vision tasks. These strengths make CL particularly well-suited for Earth System Observation (ESO), where diverse satellite modalities such as optical and SAR imagery offer naturally aligned views of the same geospatial regions. However, ESO presents unique challenges, including high inter-class similarity, scene clutter, and ambiguous boundaries, which complicate representation learning -- especially in low-label, multi-label settings. Existing CL frameworks often focus on intra-modality self-supervision or lack mechanisms for multi-label alignment and semantic precision across modalities. In this work, we introduce MoSAiC, a unified framework that jointly optimizes intra- and inter-modality contrastive learning with a multi-label supervised contrastive loss. Designed specifically for multi-modal satellite imagery, MoSAiC enables finer semantic disentanglement and more robust representation learning across spectrally similar and spatially complex classes. Experiments on two benchmark datasets, BigEarthNet V2.0 and Sent12MS, show that MoSAiC consistently outperforms both fully supervised and self-supervised baselines in terms of accuracy, cluster coherence, and generalization in low-label and high-class-overlap scenarios. 

---
# KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment 

**Authors**: Jiyao Zhang, Chengli Zhong, Hui Xu, Qige Li, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08665)  

**Abstract**: Modern large language models (LLMs) show promising progress in formalizing informal mathematics into machine-verifiable theorems. However, these methods still face bottlenecks due to the limited quantity and quality of multilingual parallel corpora. In this paper, we propose a novel neuro-symbolic framework KELPS (Knowledge-Equation based Logical Processing System) to address these problems. KELPS is an iterative framework for translating, synthesizing, and filtering informal data into multiple formal languages (Lean, Coq, and Isabelle). First, we translate natural language into Knowledge Equations (KEs), a novel language that we designed, theoretically grounded in assertional logic. Next, we convert them to target languages through rigorously defined rules that preserve both syntactic structure and semantic meaning. This process yielded a parallel corpus of over 60,000 problems. Our framework achieves 88.9% syntactic accuracy (pass@1) on MiniF2F, outperforming SOTA models such as Deepseek-V3 (81%) and Herald (81.3%) across multiple datasets. All datasets and codes are available in the supplementary materials. 

---
# Safe Deep Reinforcement Learning for Resource Allocation with Peak Age of Information Violation Guarantees 

**Authors**: Berire Gunes Reyhan, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2507.08653)  

**Abstract**: In Wireless Networked Control Systems (WNCSs), control and communication systems must be co-designed due to their strong interdependence. This paper presents a novel optimization theory-based safe deep reinforcement learning (DRL) framework for ultra-reliable WNCSs, ensuring constraint satisfaction while optimizing performance, for the first time in the literature. The approach minimizes power consumption under key constraints, including Peak Age of Information (PAoI) violation probability, transmit power, and schedulability in the finite blocklength regime. PAoI violation probability is uniquely derived by combining stochastic maximum allowable transfer interval (MATI) and maximum allowable packet delay (MAD) constraints in a multi-sensor network. The framework consists of two stages: optimization theory and safe DRL. The first stage derives optimality conditions to establish mathematical relationships among variables, simplifying and decomposing the problem. The second stage employs a safe DRL model where a teacher-student framework guides the DRL agent (student). The control mechanism (teacher) evaluates compliance with system constraints and suggests the nearest feasible action when needed. Extensive simulations show that the proposed framework outperforms rule-based and other optimization theory based DRL benchmarks, achieving faster convergence, higher rewards, and greater stability. 

---
# DatasetAgent: A Novel Multi-Agent System for Auto-Constructing Datasets from Real-World Images 

**Authors**: Haoran Sun, Haoyu Bian, Shaoning Zeng, Yunbo Rao, Xu Xu, Lin Mei, Jianping Gou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08648)  

**Abstract**: Common knowledge indicates that the process of constructing image datasets usually depends on the time-intensive and inefficient method of manual collection and annotation. Large models offer a solution via data generation. Nonetheless, real-world data are obviously more valuable comparing to artificially intelligence generated data, particularly in constructing image datasets. For this reason, we propose a novel method for auto-constructing datasets from real-world images by a multiagent collaborative system, named as DatasetAgent. By coordinating four different agents equipped with Multi-modal Large Language Models (MLLMs), as well as a tool package for image optimization, DatasetAgent is able to construct high-quality image datasets according to user-specified requirements. In particular, two types of experiments are conducted, including expanding existing datasets and creating new ones from scratch, on a variety of open-source datasets. In both cases, multiple image datasets constructed by DatasetAgent are used to train various vision models for image classification, object detection, and image segmentation. 

---
# Scaling Attention to Very Long Sequences in Linear Time with Wavelet-Enhanced Random Spectral Attention (WERSA) 

**Authors**: Vincenzo Dentamaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.08637)  

**Abstract**: Transformer models are computationally costly on long sequences since regular attention has quadratic $O(n^2)$ time complexity. We introduce Wavelet-Enhanced Random Spectral Attention (WERSA), a novel mechanism of linear $O(n)$ time complexity that is pivotal to enable successful long-sequence processing without the performance trade-off. WERSA merges content-adaptive random spectral features together with multi-resolution Haar wavelets and learnable parameters to selectively attend to informative scales of data while preserving linear efficiency.
Large-scale comparisons \textbf{on single GPU} and across various benchmarks (vision, NLP, hierarchical reasoning) and various attention mechanisms (like Multiheaded Attention, Flash-Attention-2, FNet, Linformer, Performer, Waveformer), reveal uniform advantages of WERSA. It achieves best accuracy in all tests. On ArXiv classification, WERSA improves accuracy over vanilla attention by 1.2\% (86.2\% vs 85.0\%) while cutting training time by 81\% (296s vs 1554s) and FLOPS by 73.4\% (26.2G vs 98.4G). Significantly, WERSA excels where vanilla and FlashAttention-2 fail: on ArXiv-128k's extremely lengthy sequences, it achieves best accuracy (79.1\%) and AUC (0.979) among viable methods, operating on data that gives Out-Of-Memory errors to quadratic methods while being \textbf{twice as fast} as Waveformer, its next-best competitor.
By significantly reducing computational loads without compromising accuracy, WERSA makes possible more practical, more affordable, long-context models, in particular on low-resource hardware, for more sustainable and more scalable AI development. 

---
# Normalized vs Diplomatic Annotation: A Case Study of Automatic Information Extraction from Handwritten Uruguayan Birth Certificates 

**Authors**: Natalia Bottaioli, Solène Tarride, Jérémy Anger, Seginus Mowlavi, Marina Gardella, Antoine Tadros, Gabriele Facciolo, Rafael Grompone von Gioi, Christopher Kermorvant, Jean-Michel Morel, Javier Preciozzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.08636)  

**Abstract**: This study evaluates the recently proposed Document Attention Network (DAN) for extracting key-value information from Uruguayan birth certificates, handwritten in Spanish. We investigate two annotation strategies for automatically transcribing handwritten documents, fine-tuning DAN with minimal training data and annotation effort. Experiments were conducted on two datasets containing the same images (201 scans of birth certificates written by more than 15 different writers) but with different annotation methods. Our findings indicate that normalized annotation is more effective for fields that can be standardized, such as dates and places of birth, whereas diplomatic annotation performs much better for fields containing names and surnames, which can not be standardized. 

---
# Adaptive Framework for Ambient Intelligence in Rehabilitation Assistance 

**Authors**: Gábor Baranyi, Zsolt Csibi, Kristian Fenech, Áron Fóthi, Zsófia Gaál, Joul Skaf, András Lőrincz  

**Link**: [PDF](https://arxiv.org/pdf/2507.08624)  

**Abstract**: This paper introduces the Ambient Intelligence Rehabilitation Support (AIRS) framework, an advanced artificial intelligence-based solution tailored for home rehabilitation environments. AIRS integrates cutting-edge technologies, including Real-Time 3D Reconstruction (RT-3DR), intelligent navigation, and large Vision-Language Models (VLMs), to create a comprehensive system for machine-guided physical rehabilitation. The general AIRS framework is demonstrated in rehabilitation scenarios following total knee replacement (TKR), utilizing a database of 263 video recordings for evaluation. A smartphone is employed within AIRS to perform RT-3DR of living spaces and has a body-matched avatar to provide visual feedback about the excercise. This avatar is necessary in (a) optimizing exercise configurations, including camera placement, patient positioning, and initial poses, and (b) addressing privacy concerns and promoting compliance with the AI Act. The system guides users through the recording process to ensure the collection of properly recorded videos. AIRS employs two feedback mechanisms: (i) visual 3D feedback, enabling direct comparisons between prerecorded clinical exercises and patient home recordings and (ii) VLM-generated feedback, providing detailed explanations and corrections for exercise errors. The framework also supports people with visual and hearing impairments. It also features a modular design that can be adapted to broader rehabilitation contexts. AIRS software components are available for further use and customization. 

---
# A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1 

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski  

**Link**: [PDF](https://arxiv.org/pdf/2507.08621)  

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as this http URL and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings. 

---
# Towards Collaborative Fairness in Federated Learning Under Imbalanced Covariate Shift 

**Authors**: Tianrun Yu, Jiaqi Wang, Haoyu Wang, Mingquan Lin, Han Liu, Nelson S. Yee, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08617)  

**Abstract**: Collaborative fairness is a crucial challenge in federated learning. However, existing approaches often overlook a practical yet complex form of heterogeneity: imbalanced covariate shift. We provide a theoretical analysis of this setting, which motivates the design of FedAKD (Federated Asynchronous Knowledge Distillation)- simple yet effective approach that balances accurate prediction with collaborative fairness. FedAKD consists of client and server updates. In the client update, we introduce a novel asynchronous knowledge distillation strategy based on our preliminary analysis, which reveals that while correctly predicted samples exhibit similar feature distributions across clients, incorrectly predicted samples show significant variability. This suggests that imbalanced covariate shift primarily arises from misclassified samples. Leveraging this insight, our approach first applies traditional knowledge distillation to update client models while keeping the global model fixed. Next, we select correctly predicted high-confidence samples and update the global model using these samples while keeping client models fixed. The server update simply aggregates all client models. We further provide a theoretical proof of FedAKD's convergence. Experimental results on public datasets (FashionMNIST and CIFAR10) and a real-world Electronic Health Records (EHR) dataset demonstrate that FedAKD significantly improves collaborative fairness, enhances predictive accuracy, and fosters client participation even under highly heterogeneous data distributions. 

---
# Generating Proto-Personas through Prompt Engineering: A Case Study on Efficiency, Effectiveness and Empathy 

**Authors**: Fernando Ayach, Vitor Lameirão, Raul Leão, Jerfferson Felizardo, Rafael Sobrinho, Vanessa Borges, Patrícia Matsubara, Awdren Fontão  

**Link**: [PDF](https://arxiv.org/pdf/2507.08594)  

**Abstract**: Proto-personas are commonly used during early-stage Product Discovery, such as Lean Inception, to guide product definition and stakeholder alignment. However, the manual creation of proto-personas is often time-consuming, cognitively demanding, and prone to bias. In this paper, we propose and empirically investigate a prompt engineering-based approach to generate proto-personas with the support of Generative AI (GenAI). Our goal is to evaluate the approach in terms of efficiency, effectiveness, user acceptance, and the empathy elicited by the generated personas. We conducted a case study with 19 participants embedded in a real Lean Inception, employing a qualitative and quantitative methods design. The results reveal the approach's efficiency by reducing time and effort and improving the quality and reusability of personas in later discovery phases, such as Minimum Viable Product (MVP) scoping and feature refinement. While acceptance was generally high, especially regarding perceived usefulness and ease of use, participants noted limitations related to generalization and domain specificity. Furthermore, although cognitive empathy was strongly supported, affective and behavioral empathy varied significantly across participants. These results contribute novel empirical evidence on how GenAI can be effectively integrated into software Product Discovery practices, while also identifying key challenges to be addressed in future iterations of such hybrid design processes. 

---
# To Trade or Not to Trade: An Agentic Approach to Estimating Market Risk Improves Trading Decisions 

**Authors**: Dimitrios Emmanoulopoulos, Ollie Olby, Justin Lyon, Namid R. Stillman  

**Link**: [PDF](https://arxiv.org/pdf/2507.08584)  

**Abstract**: Large language models (LLMs) are increasingly deployed in agentic frameworks, in which prompts trigger complex tool-based analysis in pursuit of a goal. While these frameworks have shown promise across multiple domains including in finance, they typically lack a principled model-building step, relying instead on sentiment- or trend-based analysis. We address this gap by developing an agentic system that uses LLMs to iteratively discover stochastic differential equations for financial time series. These models generate risk metrics which inform daily trading decisions. We evaluate our system in both traditional backtests and using a market simulator, which introduces synthetic but causally plausible price paths and news events. We find that model-informed trading strategies outperform standard LLM-based agents, improving Sharpe ratios across multiple equities. Our results show that combining LLMs with agentic model discovery enhances market risk estimation and enables more profitable trading decisions. 

---
# A Multi-Modal Fusion Framework for Brain Tumor Segmentation Based on 3D Spatial-Language-Vision Integration and Bidirectional Interactive Attention Mechanism 

**Authors**: Mingda Zhang, Kaiwen Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.08574)  

**Abstract**: This study aims to develop a novel multi-modal fusion framework for brain tumor segmentation that integrates spatial-language-vision information through bidirectional interactive attention mechanisms to improve segmentation accuracy and boundary delineation. Methods: We propose two core components: Multi-modal Semantic Fusion Adapter (MSFA) integrating 3D MRI data with clinical text descriptions through hierarchical semantic decoupling, and Bidirectional Interactive Visual-semantic Attention (BIVA) enabling iterative information exchange between modalities. The framework was evaluated on BraTS 2020 dataset comprising 369 multi-institutional MRI scans. Results: The proposed method achieved average Dice coefficient of 0.8505 and 95% Hausdorff distance of 2.8256mm across enhancing tumor, tumor core, and whole tumor regions, outperforming state-of-the-art methods including SCAU-Net, CA-Net, and 3D U-Net. Ablation studies confirmed critical contributions of semantic and spatial modules to boundary precision. Conclusion: Multi-modal semantic fusion combined with bidirectional interactive attention significantly enhances brain tumor segmentation performance, establishing new paradigms for integrating clinical knowledge into medical image analysis. 

---
# FreeAudio: Training-Free Timing Planning for Controllable Long-Form Text-to-Audio Generation 

**Authors**: Yuxuan Jiang, Zehua Chen, Zeqian Ju, Chang Li, Weibei Dou, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08557)  

**Abstract**: Text-to-audio (T2A) generation has achieved promising results with the recent advances in generative models. However, because of the limited quality and quantity of temporally-aligned audio-text pairs, existing T2A methods struggle to handle the complex text prompts that contain precise timing control, e.g., "owl hooted at 2.4s-5.2s". Recent works have explored data augmentation techniques or introduced timing conditions as model inputs to enable timing-conditioned 10-second T2A generation, while their synthesis quality is still limited. In this work, we propose a novel training-free timing-controlled T2A framework, FreeAudio, making the first attempt to enable timing-controlled long-form T2A generation, e.g., "owl hooted at 2.4s-5.2s and crickets chirping at 0s-24s". Specifically, we first employ an LLM to plan non-overlapping time windows and recaption each with a refined natural language description, based on the input text and timing prompts. Then we introduce: 1) Decoupling and Aggregating Attention Control for precise timing control; 2) Contextual Latent Composition for local smoothness and Reference Guidance for global consistency. Extensive experiments show that: 1) FreeAudio achieves state-of-the-art timing-conditioned T2A synthesis quality among training-free methods and is comparable to leading training-based methods; 2) FreeAudio demonstrates comparable long-form generation quality with training-based Stable Audio and paves the way for timing-controlled long-form T2A synthesis. Demo samples are available at: this https URL 

---
# RadiomicsRetrieval: A Customizable Framework for Medical Image Retrieval Using Radiomics Features 

**Authors**: Inye Na, Nejung Rue, Jiwon Chung, Hyunjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.08546)  

**Abstract**: Medical image retrieval is a valuable field for supporting clinical decision-making, yet current methods primarily support 2D images and require fully annotated queries, limiting clinical flexibility. To address this, we propose RadiomicsRetrieval, a 3D content-based retrieval framework bridging handcrafted radiomics descriptors with deep learning-based embeddings at the tumor level. Unlike existing 2D approaches, RadiomicsRetrieval fully exploits volumetric data to leverage richer spatial context in medical images. We employ a promptable segmentation model (e.g., SAM) to derive tumor-specific image embeddings, which are aligned with radiomics features extracted from the same tumor via contrastive learning. These representations are further enriched by anatomical positional embedding (APE). As a result, RadiomicsRetrieval enables flexible querying based on shape, location, or partial feature sets. Extensive experiments on both lung CT and brain MRI public datasets demonstrate that radiomics features significantly enhance retrieval specificity, while APE provides global anatomical context essential for location-based searches. Notably, our framework requires only minimal user prompts (e.g., a single point), minimizing segmentation overhead and supporting diverse clinical scenarios. The capability to query using either image embeddings or selected radiomics attributes highlights its adaptability, potentially benefiting diagnosis, treatment planning, and research on large-scale medical imaging repositories. Our code is available at this https URL. 

---
# White-Basilisk: A Hybrid Model for Code Vulnerability Detection 

**Authors**: Ioannis Lamprou, Alexander Shevtsov, Ioannis Arapakis, Sotiris Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2507.08540)  

**Abstract**: The proliferation of software vulnerabilities presents a significant challenge to cybersecurity, necessitating more effective detection methodologies. We introduce White-Basilisk, a novel approach to vulnerability detection that demonstrates superior performance while challenging prevailing assumptions in AI model scaling. Utilizing an innovative architecture that integrates Mamba layers, linear self-attention, and a Mixture of Experts framework, White-Basilisk achieves state-of-the-art results in vulnerability detection tasks with a parameter count of only 200M. The model's capacity to process sequences of unprecedented length enables comprehensive analysis of extensive codebases in a single pass, surpassing the context limitations of current Large Language Models (LLMs). White-Basilisk exhibits robust performance on imbalanced, real-world datasets, while maintaining computational efficiency that facilitates deployment across diverse organizational scales. This research not only establishes new benchmarks in code security but also provides empirical evidence that compact, efficiently designed models can outperform larger counterparts in specialized tasks, potentially redefining optimization strategies in AI development for domain-specific applications. 

---
# MIDI-VALLE: Improving Expressive Piano Performance Synthesis Through Neural Codec Language Modelling 

**Authors**: Jingjing Tang, Xin Wang, Zhe Zhang, Junichi Yamagishi, Geraint Wiggins, George Fazekas  

**Link**: [PDF](https://arxiv.org/pdf/2507.08530)  

**Abstract**: Generating expressive audio performances from music scores requires models to capture both instrument acoustics and human interpretation. Traditional music performance synthesis pipelines follow a two-stage approach, first generating expressive performance MIDI from a score, then synthesising the MIDI into audio. However, the synthesis models often struggle to generalise across diverse MIDI sources, musical styles, and recording environments. To address these challenges, we propose MIDI-VALLE, a neural codec language model adapted from the VALLE framework, which was originally designed for zero-shot personalised text-to-speech (TTS) synthesis. For performance MIDI-to-audio synthesis, we improve the architecture to condition on a reference audio performance and its corresponding MIDI. Unlike previous TTS-based systems that rely on piano rolls, MIDI-VALLE encodes both MIDI and audio as discrete tokens, facilitating a more consistent and robust modelling of piano performances. Furthermore, the model's generalisation ability is enhanced by training on an extensive and diverse piano performance dataset. Evaluation results show that MIDI-VALLE significantly outperforms a state-of-the-art baseline, achieving over 75% lower Frechet Audio Distance on the ATEPP and Maestro datasets. In the listening test, MIDI-VALLE received 202 votes compared to 58 for the baseline, demonstrating improved synthesis quality and generalisation across diverse performance MIDI inputs. 

---
# PromotionGo at SemEval-2025 Task 11: A Feature-Centric Framework for Cross-Lingual Multi-Emotion Detection in Short Texts 

**Authors**: Ziyi Huang, Xia Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.08499)  

**Abstract**: This paper presents our system for SemEval 2025 Task 11: Bridging the Gap in Text-Based Emotion Detection (Track A), which focuses on multi-label emotion detection in short texts. We propose a feature-centric framework that dynamically adapts document representations and learning algorithms to optimize language-specific performance. Our study evaluates three key components: document representation, dimensionality reduction, and model training in 28 languages, highlighting five for detailed analysis. The results show that TF-IDF remains highly effective for low-resource languages, while contextual embeddings like FastText and transformer-based document representations, such as those produced by Sentence-BERT, exhibit language-specific strengths. Principal Component Analysis (PCA) reduces training time without compromising performance, particularly benefiting FastText and neural models such as Multi-Layer Perceptrons (MLP). Computational efficiency analysis underscores the trade-off between model complexity and processing cost. Our framework provides a scalable solution for multilingual emotion detection, addressing the challenges of linguistic diversity and resource constraints. 

---
# Enhancing Essay Cohesion Assessment: A Novel Item Response Theory Approach 

**Authors**: Bruno Alexandre Rosa, Hilário Oliveira, Luiz Rodrigues, Eduardo Araujo Oliveira, Rafael Ferreira Mello  

**Link**: [PDF](https://arxiv.org/pdf/2507.08487)  

**Abstract**: Essays are considered a valuable mechanism for evaluating learning outcomes in writing. Textual cohesion is an essential characteristic of a text, as it facilitates the establishment of meaning between its parts. Automatically scoring cohesion in essays presents a challenge in the field of educational artificial intelligence. The machine learning algorithms used to evaluate texts generally do not consider the individual characteristics of the instances that comprise the analysed corpus. In this meaning, item response theory can be adapted to the context of machine learning, characterising the ability, difficulty and discrimination of the models used. This work proposes and analyses the performance of a cohesion score prediction approach based on item response theory to adjust the scores generated by machine learning models. In this study, the corpus selected for the experiments consisted of the extended Essay-BR, which includes 6,563 essays in the style of the National High School Exam (ENEM), and the Brazilian Portuguese Narrative Essays, comprising 1,235 essays written by 5th to 9th grade students from public schools. We extracted 325 linguistic features and treated the problem as a machine learning regression task. The experimental results indicate that the proposed approach outperforms conventional machine learning models and ensemble methods in several evaluation metrics. This research explores a potential approach for improving the automatic evaluation of cohesion in educational essays. 

---
# Pre-Training LLMs on a budget: A comparison of three optimizers 

**Authors**: Joel Schlotthauer, Christian Kroos, Chris Hinze, Viktor Hangya, Luzian Hahn, Fabian Küch  

**Link**: [PDF](https://arxiv.org/pdf/2507.08472)  

**Abstract**: Optimizers play a decisive role in reducing pre-training times for LLMs and achieving better-performing models. In this study, we compare three major variants: the de-facto standard AdamW, the simpler Lion, developed through an evolutionary search, and the second-order optimizer Sophia. For better generalization, we train with two different base architectures and use a single- and a multiple-epoch approach while keeping the number of tokens constant. Using the Maximal Update Parametrization and smaller proxy models, we tune relevant hyperparameters separately for each combination of base architecture and optimizer. We found that while the results from all three optimizers were in approximately the same range, Sophia exhibited the lowest training and validation loss, Lion was fastest in terms of training GPU hours but AdamW led to the best downstream evaluation results. 

---
# A document is worth a structured record: Principled inductive bias design for document recognition 

**Authors**: Benjamin Meyer, Lukas Tuggener, Sascha Hänzi, Daniel Schmid, Erdal Ayfer, Benjamin F. Grewe, Ahmed Abdulkadir, Thilo Stadelmann  

**Link**: [PDF](https://arxiv.org/pdf/2507.08458)  

**Abstract**: Many document types use intrinsic, convention-driven structures that serve to encode precise and structured information, such as the conventions governing engineering drawings. However, state-of-the-art approaches treat document recognition as a mere computer vision problem, neglecting these underlying document-type-specific structural properties, making them dependent on sub-optimal heuristic post-processing and rendering many less frequent or more complicated document types inaccessible to modern document recognition. We suggest a novel perspective that frames document recognition as a transcription task from a document to a record. This implies a natural grouping of documents based on the intrinsic structure inherent in their transcription, where related document types can be treated (and learned) similarly. We propose a method to design structure-specific inductive biases for the underlying machine-learned end-to-end document recognition systems, and a respective base transformer architecture that we successfully adapt to different structures. We demonstrate the effectiveness of the so-found inductive biases in extensive experiments with progressively complex record structures from monophonic sheet music, shape drawings, and simplified engineering drawings. By integrating an inductive bias for unrestricted graph structures, we train the first-ever successful end-to-end model to transcribe engineering drawings to their inherently interlinked information. Our approach is relevant to inform the design of document recognition systems for document types that are less well understood than standard OCR, OMR, etc., and serves as a guide to unify the design of future document foundation models. 

---
# Space filling positionality and the Spiroformer 

**Authors**: M. Maurin, M.Á. Evangelista-Alvarado, P. Suárez-Serrato  

**Link**: [PDF](https://arxiv.org/pdf/2507.08456)  

**Abstract**: Transformers excel when dealing with sequential data. Generalizing transformer models to geometric domains, such as manifolds, we encounter the problem of not having a well-defined global order. We propose a solution with attention heads following a space-filling curve. As a first experimental example, we present the Spiroformer, a transformer that follows a polar spiral on the $2$-sphere. 

---
# Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT 

**Authors**: Wei Zhang, Yihang Wu, Songhua Li, Wenjie Ma, Xin Ma, Qiang Li, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08448)  

**Abstract**: 3D reconstruction, which aims to recover the dense three-dimensional structure of a scene, is a cornerstone technology for numerous applications, including augmented/virtual reality, autonomous driving, and robotics. While traditional pipelines like Structure from Motion (SfM) and Multi-View Stereo (MVS) achieve high precision through iterative optimization, they are limited by complex workflows, high computational cost, and poor robustness in challenging scenarios like texture-less regions. Recently, deep learning has catalyzed a paradigm shift in 3D reconstruction. A new family of models, exemplified by DUSt3R, has pioneered a feed-forward approach. These models employ a unified deep network to jointly infer camera poses and dense geometry directly from an Unconstrained set of images in a single forward pass. This survey provides a systematic review of this emerging domain. We begin by dissecting the technical framework of these feed-forward models, including their Transformer-based correspondence modeling, joint pose and geometry regression mechanisms, and strategies for scaling from two-view to multi-view scenarios. To highlight the disruptive nature of this new paradigm, we contrast it with both traditional pipelines and earlier learning-based methods like MVSNet. Furthermore, we provide an overview of relevant datasets and evaluation metrics. Finally, we discuss the technology's broad application prospects and identify key future challenges and opportunities, such as model accuracy and scalability, and handling dynamic scenes. 

---
# CUE-RAG: Towards Accurate and Cost-Efficient Graph-Based RAG via Multi-Partite Graph and Query-Driven Iterative Retrieval 

**Authors**: Yaodong Su, Yixiang Fang, Yingli Zhou, Quanqing Xu, Chuanhui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08445)  

**Abstract**: Despite the remarkable progress of Large Language Models (LLMs), their performance in question answering (QA) remains limited by the lack of domain-specific and up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external information, often from graph-structured data. However, existing graph-based RAG methods suffer from poor graph quality due to incomplete extraction and insufficient utilization of query information during retrieval. To overcome these limitations, we propose CUE-RAG, a novel approach that introduces (1) a multi-partite graph index incorporates text Chunks, knowledge Units, and Entities to capture semantic content at multiple levels of granularity, (2) a hybrid extraction strategy that reduces LLM token usage while still producing accurate and disambiguated knowledge units, and (3) Q-Iter, a query-driven iterative retrieval strategy that enhances relevance through semantic search and constrained graph traversal. Experiments on three QA benchmarks show that CUE-RAG significantly outperforms state-of-the-art baselines, achieving up to 99.33% higher Accuracy and 113.51% higher F1 score while reducing indexing costs by 72.58%. Remarkably, CUE-RAG matches or outperforms baselines even without using an LLM for indexing. These results demonstrate the effectiveness and cost-efficiency of CUE-RAG in advancing graph-based RAG systems. 

---
# Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Image Generation 

**Authors**: Anlin Zheng, Xin Wen, Xuanyang Zhang, Chuofan Ma, Tiancai Wang, Gang Yu, Xiangyu Zhang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.08441)  

**Abstract**: Leveraging the powerful representations of pre-trained vision foundation models -- traditionally used for visual comprehension -- we explore a novel direction: building an image tokenizer directly atop such models, a largely underexplored area. Specifically, we employ a frozen vision foundation model as the encoder of our tokenizer. To enhance its effectiveness, we introduce two key components: (1) a region-adaptive quantization framework that reduces redundancy in the pre-trained features on regular 2D grids, and (2) a semantic reconstruction objective that aligns the tokenizer's outputs with the foundation model's representations to preserve semantic fidelity. Based on these designs, our proposed image tokenizer, VFMTok, achieves substantial improvements in image reconstruction and generation quality, while also enhancing token efficiency. It further boosts autoregressive (AR) generation -- achieving a gFID of 2.07 on ImageNet benchmarks, while accelerating model convergence by three times, and enabling high-fidelity class-conditional synthesis without the need for classifier-free guidance (CFG). The code will be released publicly to benefit the community. 

---
# Finding Common Ground: Using Large Language Models to Detect Agreement in Multi-Agent Decision Conferences 

**Authors**: Selina Heller, Mohamed Ibrahim, David Antony Selby, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2507.08440)  

**Abstract**: Decision conferences are structured, collaborative meetings that bring together experts from various fields to address complex issues and reach a consensus on recommendations for future actions or policies. These conferences often rely on facilitated discussions to ensure productive dialogue and collective agreement. Recently, Large Language Models (LLMs) have shown significant promise in simulating real-world scenarios, particularly through collaborative multi-agent systems that mimic group interactions. In this work, we present a novel LLM-based multi-agent system designed to simulate decision conferences, specifically focusing on detecting agreement among the participant agents. To achieve this, we evaluate six distinct LLMs on two tasks: stance detection, which identifies the position an agent takes on a given issue, and stance polarity detection, which identifies the sentiment as positive, negative, or neutral. These models are further assessed within the multi-agent system to determine their effectiveness in complex simulations. Our results indicate that LLMs can reliably detect agreement even in dynamic and nuanced debates. Incorporating an agreement-detection agent within the system can also improve the efficiency of group debates and enhance the overall quality and coherence of deliberations, making them comparable to real-world decision conferences regarding outcome and decision-making. These findings demonstrate the potential for LLM-based multi-agent systems to simulate group decision-making processes. They also highlight that such systems could be instrumental in supporting decision-making with expert elicitation workshops across various domains. 

---
# ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains 

**Authors**: Zilu Dong, Xiangqing Shen, Zinong Yang, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.08427)  

**Abstract**: Current knowledge editing methods for large language models (LLMs) struggle to maintain logical consistency when propagating ripple effects to associated facts. We propose ChainEdit, a framework that synergizes knowledge graph-derived logical rules with LLM logical reasoning capabilities to enable systematic chain updates. By automatically extracting logical patterns from structured knowledge bases and aligning them with LLMs' internal logics, ChainEdit dynamically generates and edits logically connected knowledge clusters. Experiments demonstrate an improvement of more than 30% in logical generalization over baselines while preserving editing reliability and specificity. We further address evaluation biases in existing benchmarks through knowledge-aware protocols that disentangle external dependencies. This work establishes new state-of-the-art performance on ripple effect while ensuring internal logical consistency after knowledge editing. 

---
# Deep Hashing with Semantic Hash Centers for Image Retrieval 

**Authors**: Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08404)  

**Abstract**: Deep hashing is an effective approach for large-scale image retrieval. Current methods are typically classified by their supervision types: point-wise, pair-wise, and list-wise. Recent point-wise techniques (e.g., CSQ, MDS) have improved retrieval performance by pre-assigning a hash center to each class, enhancing the discriminability of hash codes across various datasets. However, these methods rely on data-independent algorithms to generate hash centers, which neglect the semantic relationships between classes and may degrade retrieval performance.
This paper introduces the concept of semantic hash centers, building on the idea of traditional hash centers. We hypothesize that hash centers of semantically related classes should have closer Hamming distances, while those of unrelated classes should be more distant. To this end, we propose a three-stage framework, SHC, to generate hash codes that preserve semantic structure.
First, we develop a classification network to identify semantic similarities between classes using a data-dependent similarity calculation that adapts to varying data distributions. Second, we introduce an optimization algorithm to generate semantic hash centers, preserving semantic relatedness while enforcing a minimum distance between centers to avoid excessively similar hash codes. Finally, a deep hashing network is trained using these semantic centers to convert images into binary hash codes.
Experimental results on large-scale retrieval tasks across several public datasets show that SHC significantly improves retrieval performance. Specifically, SHC achieves average improvements of +7.26%, +7.62%, and +11.71% in MAP@100, MAP@1000, and MAP@ALL metrics, respectively, over state-of-the-art methods. 

---
# Towards AI-Native RAN: An Operator's Perspective of 6G Day 1 Standardization 

**Authors**: Nan Li, Qi Sun, Lehan Wang, Xiaofei Xu, Jinri Huang, Chunhui Liu, Jing Gao, Yuhong Huang, Chih-Lin I  

**Link**: [PDF](https://arxiv.org/pdf/2507.08403)  

**Abstract**: Artificial Intelligence/Machine Learning (AI/ML) has become the most certain and prominent feature of 6G mobile networks. Unlike 5G, where AI/ML was not natively integrated but rather an add-on feature over existing architecture, 6G shall incorporate AI from the onset to address its complexity and support ubiquitous AI applications. Based on our extensive mobile network operation and standardization experience from 2G to 5G, this paper explores the design and standardization principles of AI-Native radio access networks (RAN) for 6G, with a particular focus on its critical Day 1 architecture, functionalities and capabilities. We investigate the framework of AI-Native RAN and present its three essential capabilities to shed some light on the standardization direction; namely, AI-driven RAN processing/optimization/automation, reliable AI lifecycle management (LCM), and AI-as-a-Service (AIaaS) provisioning. The standardization of AI-Native RAN, in particular the Day 1 features, including an AI-Native 6G RAN architecture, were proposed. For validation, a large-scale field trial with over 5000 5G-A base stations have been built and delivered significant improvements in average air interface latency, root cause identification, and network energy consumption with the proposed architecture and the supporting AI functions. This paper aims to provide a Day 1 framework for 6G AI-Native RAN standardization design, balancing technical innovation with practical deployment. 

---
# PanMatch: Unleashing the Potential of Large Vision Models for Unified Matching Models 

**Authors**: Yongjian Zhang, Longguang Wang, Kunhong Li, Ye Zhang, Yun Wang, Liang Lin, Yulan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08400)  

**Abstract**: This work presents PanMatch, a versatile foundation model for robust correspondence matching. Unlike previous methods that rely on task-specific architectures and domain-specific fine-tuning to support tasks like stereo matching, optical flow or feature matching, our key insight is that any two-frame correspondence matching task can be addressed within a 2D displacement estimation framework using the same model weights. Such a formulation eliminates the need for designing specialized unified architectures or task-specific ensemble models. Instead, it achieves multi-task integration by endowing displacement estimation algorithms with unprecedented generalization capabilities. To this end, we highlight the importance of a robust feature extractor applicable across multiple domains and tasks, and propose the feature transformation pipeline that leverage all-purpose features from Large Vision Models to endow matching baselines with zero-shot cross-view matching capabilities. Furthermore, we assemble a cross-domain dataset with near 1.8 million samples from stereo matching, optical flow, and feature matching domains to pretrain PanMatch. We demonstrate the versatility of PanMatch across a wide range of domains and downstream tasks using the same model weights. Our model outperforms UniMatch and Flow-Anything on cross-task evaluations, and achieves comparable performance to most state-of-the-art task-specific algorithms on task-oriented benchmarks. Additionally, PanMatch presents unprecedented zero-shot performance in abnormal scenarios, such as rainy day and satellite imagery, where most existing robust algorithms fail to yield meaningful results. 

---
# Intelligent Control of Spacecraft Reaction Wheel Attitude Using Deep Reinforcement Learning 

**Authors**: Ghaith El-Dalahmeh, Mohammad Reza Jabbarpour, Bao Quoc Vo, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.08366)  

**Abstract**: Reliable satellite attitude control is essential for the success of space missions, particularly as satellites increasingly operate autonomously in dynamic and uncertain environments. Reaction wheels (RWs) play a pivotal role in attitude control, and maintaining control resilience during RW faults is critical to preserving mission objectives and system stability. However, traditional Proportional Derivative (PD) controllers and existing deep reinforcement learning (DRL) algorithms such as TD3, PPO, and A2C often fall short in providing the real time adaptability and fault tolerance required for autonomous satellite operations. This study introduces a DRL-based control strategy designed to improve satellite resilience and adaptability under fault conditions. Specifically, the proposed method integrates Twin Delayed Deep Deterministic Policy Gradient (TD3) with Hindsight Experience Replay (HER) and Dimension Wise Clipping (DWC) referred to as TD3-HD to enhance learning in sparse reward environments and maintain satellite stability during RW failures. The proposed approach is benchmarked against PD control and leading DRL algorithms. Experimental results show that TD3-HD achieves significantly lower attitude error, improved angular velocity regulation, and enhanced stability under fault conditions. These findings underscore the proposed method potential as a powerful, fault tolerant, onboard AI solution for autonomous satellite attitude control. 

---
# Single-Domain Generalization for Multimodal Cross-Cancer Prognosis via Dirac Rebalancer and Distribution Entanglement 

**Authors**: Jia-Xuan Jiang, Jiashuai Liu, Hongtao Wu, Yifeng Wu, Zhong Wang, Qi Bi, Yefeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08340)  

**Abstract**: Deep learning has shown remarkable performance in integrating multimodal data for survival prediction. However, existing multimodal methods mainly focus on single cancer types and overlook the challenge of generalization across cancers. In this work, we are the first to reveal that multimodal prognosis models often generalize worse than unimodal ones in cross-cancer scenarios, despite the critical need for such robustness in clinical practice. To address this, we propose a new task: Cross-Cancer Single Domain Generalization for Multimodal Prognosis, which evaluates whether models trained on a single cancer type can generalize to unseen cancers. We identify two key challenges: degraded features from weaker modalities and ineffective multimodal integration. To tackle these, we introduce two plug-and-play modules: Sparse Dirac Information Rebalancer (SDIR) and Cancer-aware Distribution Entanglement (CADE). SDIR mitigates the dominance of strong features by applying Bernoulli-based sparsification and Dirac-inspired stabilization to enhance weaker modality signals. CADE, designed to synthesize the target domain distribution, fuses local morphological cues and global gene expression in latent space. Experiments on a four-cancer-type benchmark demonstrate superior generalization, laying the foundation for practical, robust cross-cancer multimodal prognosis. Code is available at this https URL 

---
# CoCo-Bot: Energy-based Composable Concept Bottlenecks for Interpretable Generative Models 

**Authors**: Sangwon Kim, In-su Jang, Pyongkun Kim, Kwang-Ju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.08334)  

**Abstract**: Concept Bottleneck Models (CBMs) provide interpretable and controllable generative modeling by routing generation through explicit, human-understandable concepts. However, previous generative CBMs often rely on auxiliary visual cues at the bottleneck to compensate for information not captured by the concepts, which undermines interpretability and compositionality. We propose CoCo-Bot, a post-hoc, composable concept bottleneck generative model that eliminates the need for auxiliary cues by transmitting all information solely through explicit concepts. Guided by diffusion-based energy functions, CoCo-Bot supports robust post-hoc interventions-such as concept composition and negation-across arbitrary concepts. Experiments using StyleGAN2 pre-trained on CelebA-HQ show that CoCo-Bot improves concept-level controllability and interpretability, while maintaining competitive visual quality. 

---
# Audio Inpanting using Discrete Diffusion Model 

**Authors**: Tali Dror, Iftach Shoham, Moshe Buchris, Oren Gal, Haim Permuter, Gilad Katz, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2507.08333)  

**Abstract**: Audio inpainting refers to the task of reconstructing missing segments in corrupted audio recordings. While prior approaches-including waveform and spectrogram-based diffusion models-have shown promising results for short gaps, they often degrade in quality when gaps exceed 100 milliseconds (ms). In this work, we introduce a novel inpainting method based on discrete diffusion modeling, which operates over tokenized audio representations produced by a pre-trained audio tokenizer. Our approach models the generative process directly in the discrete latent space, enabling stable and semantically coherent reconstruction of missing audio. We evaluate the method on the MusicNet dataset using both objective and perceptual metrics across gap durations up to 300 ms. We further evaluated our approach on the MTG dataset, extending the gap duration to 500 ms. Experimental results demonstrate that our method achieves competitive or superior performance compared to existing baselines, particularly for longer gaps, offering a robust solution for restoring degraded musical recordings. Audio examples of our proposed method can be found at this https URL 

---
# Interpretability-Aware Pruning for Efficient Medical Image Analysis 

**Authors**: Nikita Malik, Pratinav Seth, Neeraj Kumar Singh, Chintan Chitroda, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08330)  

**Abstract**: Deep learning has driven significant advances in medical image analysis, yet its adoption in clinical practice remains constrained by the large size and lack of transparency in modern models. Advances in interpretability techniques such as DL-Backtrace, Layer-wise Relevance Propagation, and Integrated Gradients make it possible to assess the contribution of individual components within neural networks trained on medical imaging tasks. In this work, we introduce an interpretability-guided pruning framework that reduces model complexity while preserving both predictive performance and transparency. By selectively retaining only the most relevant parts of each layer, our method enables targeted compression that maintains clinically meaningful representations. Experiments across multiple medical image classification benchmarks demonstrate that this approach achieves high compression rates with minimal loss in accuracy, paving the way for lightweight, interpretable models suited for real-world deployment in healthcare settings. 

---
# Generative AI in Science: Applications, Challenges, and Emerging Questions 

**Authors**: Ryan Harries, Cornelia Lawson, Philip Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2507.08310)  

**Abstract**: This paper examines the impact of Generative Artificial Intelligence (GenAI) on scientific practices, conducting a qualitative review of selected literature to explore its applications, benefits, and challenges. The review draws on the OpenAlex publication database, using a Boolean search approach to identify scientific literature related to GenAI (including large language models and ChatGPT). Thirty-nine highly cited papers and commentaries are reviewed and qualitatively coded. Results are categorized by GenAI applications in science, scientific writing, medical practice, and education and training. The analysis finds that while there is a rapid adoption of GenAI in science and science practice, its long-term implications remain unclear, with ongoing uncertainties about its use and governance. The study provides early insights into GenAI's growing role in science and identifies questions for future research in this evolving field. 

---
# Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency 

**Authors**: Yupu Liang, Yaping Zhang, Zhiyang Zhang, Zhiyuan Chen, Yang Zhao, Lu Xiang, Chengqing Zong, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08309)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong performance in document image tasks, especially Optical Character Recognition (OCR). However, they struggle with Document Image Machine Translation (DIMT), which requires handling both cross-modal and cross-lingual challenges. Previous efforts to enhance DIMT capability through Supervised Fine-Tuning (SFT) on the DIMT dataset often result in the forgetting of the model's existing monolingual abilities, such as OCR. To address these challenges, we introduce a novel fine-tuning paradigm, named Synchronously Self-Reviewing (SSR) its OCR proficiency, inspired by the concept "Bilingual Cognitive Advantage". Specifically, SSR prompts the model to generate OCR text before producing translation text, which allows the model to leverage its strong monolingual OCR ability while learning to translate text across languages. Comprehensive experiments demonstrate the proposed SSR learning helps mitigate catastrophic forgetting, improving the generalization ability of MLLMs on both OCR and DIMT tasks. 

---
# Invariant-based Robust Weights Watermark for Large Language Models 

**Authors**: Qingxiao Guo, Xinjie Zhu, Yilong Ma, Hui Jin, Yunhao Wang, Weifeng Zhang, Xiaobing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08288)  

**Abstract**: Watermarking technology has gained significant attention due to the increasing importance of intellectual property (IP) rights, particularly with the growing deployment of large language models (LLMs) on billions resource-constrained edge devices. To counter the potential threats of IP theft by malicious users, this paper introduces a robust watermarking scheme without retraining or fine-tuning for transformer models. The scheme generates a unique key for each user and derives a stable watermark value by solving linear constraints constructed from model invariants. Moreover, this technology utilizes noise mechanism to hide watermark locations in multi-user scenarios against collusion attack. This paper evaluates the approach on three popular models (Llama3, Phi3, Gemma), and the experimental results confirm the strong robustness across a range of attack methods (fine-tuning, pruning, quantization, permutation, scaling, reversible matrix and collusion attacks). 

---
# Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training 

**Authors**: Aleksei Ilin, Gor Matevosyan, Xueying Ma, Vladimir Eremin, Suhaa Dada, Muqun Li, Riyaaz Shaik, Haluk Noyan Tokgozoglu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08284)  

**Abstract**: We introduce a lightweight yet highly effective safety guardrail framework for language models, demonstrating that small-scale language models can achieve, and even surpass, the performance of larger counterparts in content moderation tasks. This is accomplished through high-fidelity synthetic data generation and adversarial training. The synthetic data generation process begins with human-curated seed data, which undergoes query augmentation and paraphrasing to create diverse and contextually rich examples. This augmented data is then subjected to multiple rounds of curation, ensuring high fidelity and relevance. Inspired by recent advances in the Generative Adversarial Network (GAN) architecture, our adversarial training employs reinforcement learning to guide a generator that produces challenging synthetic examples. These examples are used to fine-tune the safety classifier, enhancing its ability to detect and mitigate harmful content. Additionally, we incorporate strategies from recent research on efficient LLM training, leveraging the capabilities of smaller models to improve the performance of larger generative models. With iterative adversarial training and the generation of diverse, high-quality synthetic data, our framework enables small language models (SLMs) to serve as robust safety guardrails. This approach not only reduces computational overhead but also enhances resilience against adversarial attacks, offering a scalable and efficient solution for content moderation in AI systems. 

---
# A Practical Two-Stage Recipe for Mathematical LLMs: Maximizing Accuracy with SFT and Efficiency with Reinforcement Learning 

**Authors**: Hiroshi Yoshihara, Taiki Yamaguchi, Yuichi Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2507.08267)  

**Abstract**: Enhancing the mathematical reasoning of Large Language Models (LLMs) is a pivotal challenge in advancing AI capabilities. While Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are the dominant training paradigms, a systematic methodology for combining them to maximize both accuracy and efficiency remains largely unexplored. This paper introduces a practical and effective training recipe that strategically integrates extended SFT with RL from online inference (GRPO). We posit that these methods play complementary, not competing, roles: a prolonged SFT phase first pushes the model's accuracy to its limits, after which a GRPO phase dramatically improves token efficiency while preserving this peak performance. Our experiments reveal that extending SFT for as many as 10 epochs is crucial for performance breakthroughs, and that the primary role of GRPO in this framework is to optimize solution length. The efficacy of our recipe is rigorously validated through top-tier performance on challenging benchmarks, including a high rank among over 2,200 teams in the strictly leak-free AI Mathematical Olympiad (AIMO). This work provides the community with a battle-tested blueprint for developing state-of-the-art mathematical reasoners that are both exceptionally accurate and practically efficient. To ensure full reproducibility and empower future research, we will open-source our entire framework, including all code, model checkpoints, and training configurations at this https URL. 

---
# CL3R: 3D Reconstruction and Contrastive Learning for Enhanced Robotic Manipulation Representations 

**Authors**: Wenbo Cui, Chengyang Zhao, Yuhui Chen, Haoran Li, Zhizheng Zhang, Dongbin Zhao, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08262)  

**Abstract**: Building a robust perception module is crucial for visuomotor policy learning. While recent methods incorporate pre-trained 2D foundation models into robotic perception modules to leverage their strong semantic understanding, they struggle to capture 3D spatial information and generalize across diverse camera viewpoints. These limitations hinder the policy's effectiveness, especially in fine-grained robotic manipulation scenarios. To address these challenges, we propose CL3R, a novel 3D pre-training framework designed to enhance robotic manipulation policies. Our method integrates both spatial awareness and semantic understanding by employing a point cloud Masked Autoencoder to learn rich 3D representations while leveraging pre-trained 2D foundation models through contrastive learning for efficient semantic knowledge transfer. Additionally, we propose a 3D visual representation pre-training framework for robotic tasks. By unifying coordinate systems across datasets and introducing random fusion of multi-view point clouds, we mitigate camera view ambiguity and improve generalization, enabling robust perception from novel viewpoints at test time. Extensive experiments in both simulation and the real world demonstrate the superiority of our method, highlighting its effectiveness in visuomotor policy learning for robotic manipulation. 

---
# Quantum-Accelerated Neural Imputation with Large Language Models (LLMs) 

**Authors**: Hossein Jamali  

**Link**: [PDF](https://arxiv.org/pdf/2507.08255)  

**Abstract**: Missing data presents a critical challenge in real-world datasets, significantly degrading the performance of machine learning models. While Large Language Models (LLMs) have recently demonstrated remarkable capabilities in tabular data imputation, exemplified by frameworks like UnIMP, their reliance on classical embedding methods often limits their ability to capture complex, non-linear correlations, particularly in mixed-type data scenarios encompassing numerical, categorical, and textual features. This paper introduces Quantum-UnIMP, a novel framework that integrates shallow quantum circuits into an LLM-based imputation architecture. Our core innovation lies in replacing conventional classical input embeddings with quantum feature maps generated by an Instantaneous Quantum Polynomial (IQP) circuit. This approach enables the model to leverage quantum phenomena such as superposition and entanglement, thereby learning richer, more expressive representations of data and enhancing the recovery of intricate missingness patterns. Our experiments on benchmark mixed-type datasets demonstrate that Quantum-UnIMP reduces imputation error by up to 15.2% for numerical features (RMSE) and improves classification accuracy by 8.7% for categorical features (F1-Score) compared to state-of-the-art classical and LLM-based methods. These compelling results underscore the profound potential of quantum-enhanced representations for complex data imputation tasks, even with near-term quantum hardware. 

---
# InsightBuild: LLM-Powered Causal Reasoning in Smart Building Systems 

**Authors**: Pinaki Prasad Guha Neogi, Ahmad Mohammadshirazi, Rajiv Ramnath  

**Link**: [PDF](https://arxiv.org/pdf/2507.08235)  

**Abstract**: Smart buildings generate vast streams of sensor and control data, but facility managers often lack clear explanations for anomalous energy usage. We propose InsightBuild, a two-stage framework that integrates causality analysis with a fine-tuned large language model (LLM) to provide human-readable, causal explanations of energy consumption patterns. First, a lightweight causal inference module applies Granger causality tests and structural causal discovery on building telemetry (e.g., temperature, HVAC settings, occupancy) drawn from Google Smart Buildings and Berkeley Office datasets. Next, an LLM, fine-tuned on aligned pairs of sensor-level causes and textual explanations, receives as input the detected causal relations and generates concise, actionable explanations. We evaluate InsightBuild on two real-world datasets (Google: 2017-2022; Berkeley: 2018-2020), using expert-annotated ground-truth causes for a held-out set of anomalies. Our results demonstrate that combining explicit causal discovery with LLM-based natural language generation yields clear, precise explanations that assist facility managers in diagnosing and mitigating energy inefficiencies. 

---
# Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension? 

**Authors**: KV Aditya Srivatsa, Kaushal Kumar Maurya, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.08232)  

**Abstract**: Large Language Models (LLMs) are increasingly used as proxy students in the development of Intelligent Tutoring Systems (ITSs) and in piloting test questions. However, to what extent these proxy students accurately emulate the behavior and characteristics of real students remains an open question. To investigate this, we collected a dataset of 489 items from the National Assessment of Educational Progress (NAEP), covering mathematics and reading comprehension in grades 4, 8, and 12. We then apply an Item Response Theory (IRT) model to position 11 diverse and state-of-the-art LLMs on the same ability scale as real student populations. Our findings reveal that, without guidance, strong general-purpose models consistently outperform the average student at every grade, while weaker or domain-mismatched models may align incidentally. Using grade-enforcement prompts changes models' performance, but whether they align with the average grade-level student remains highly model- and prompt-specific: no evaluated model-prompt pair fits the bill across subjects and grades, underscoring the need for new training and evaluation strategies. We conclude by providing guidelines for the selection of viable proxies based on our findings. 

---
# Quantum Properties Trojans (QuPTs) for Attacking Quantum Neural Networks 

**Authors**: Sounak Bhowmik, Travis S. Humble, Himanshu Thapliyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.08202)  

**Abstract**: Quantum neural networks (QNN) hold immense potential for the future of quantum machine learning (QML). However, QNN security and robustness remain largely unexplored. In this work, we proposed novel Trojan attacks based on the quantum computing properties in a QNN-based binary classifier. Our proposed Quantum Properties Trojans (QuPTs) are based on the unitary property of quantum gates to insert noise and Hadamard gates to enable superposition to develop Trojans and attack QNNs. We showed that the proposed QuPTs are significantly stealthier and heavily impact the quantum circuits' performance, specifically QNNs. The most impactful QuPT caused a deterioration of 23% accuracy of the compromised QNN under the experimental setup. To the best of our knowledge, this is the first work on the Trojan attack on a fully quantum neural network independent of any hybrid classical-quantum architecture. 

---
# Consciousness as a Jamming Phase 

**Authors**: Kaichen Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08197)  

**Abstract**: This paper develops a neural jamming phase diagram that interprets the emergence of consciousness in large language models as a critical phenomenon in high-dimensional disordered this http URL establishing analogies with jamming transitions in granular matter and other complex systems, we identify three fundamental control parameters governing the phase behavior of neural networks: temperature, volume fraction, and this http URL theory provides a unified physical explanation for empirical scaling laws in artificial intelligence, demonstrating how computational cooling, density optimization, and noise reduction collectively drive systems toward a critical jamming surface where generalized intelligence emerges. Remarkably, the same thermodynamic principles that describe conventional jamming transitions appear to underlie the emergence of consciousness in neural networks, evidenced by shared critical signatures including divergent correlation lengths and scaling this http URL work explains neural language models' critical scaling through jamming physics, suggesting consciousness is a jamming phase that intrinsically connects knowledge components via long-range correlations. 

---
# Overview of the TREC 2021 deep learning track 

**Authors**: Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.08191)  

**Abstract**: This is the third year of the TREC Deep Learning track. As in previous years, we leverage the MS MARCO datasets that made hundreds of thousands of human annotated training labels available for both passage and document ranking tasks. In addition, this year we refreshed both the document and the passage collections which also led to a nearly four times increase in the document collection size and nearly $16$ times increase in the size of the passage collection. Deep neural ranking models that employ large scale pretraininig continued to outperform traditional retrieval methods this year. We also found that single stage retrieval can achieve good performance on both tasks although they still do not perform at par with multistage retrieval pipelines. Finally, the increase in the collection size and the general data refresh raised some questions about completeness of NIST judgments and the quality of the training labels that were mapped to the new collections from the old ones which we discuss in this report. 

---
# Rethinking Spatio-Temporal Anomaly Detection: A Vision for Causality-Driven Cybersecurity 

**Authors**: Arun Vignesh Malarkkan, Haoyue Bai, Xinyuan Wang, Anjali Kaushik, Dongjie Wang, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08177)  

**Abstract**: As cyber-physical systems grow increasingly interconnected and spatially distributed, ensuring their resilience against evolving cyberattacks has become a critical priority. Spatio-Temporal Anomaly detection plays an important role in ensuring system security and operational integrity. However, current data-driven approaches, largely driven by black-box deep learning, face challenges in interpretability, adaptability to distribution shifts, and robustness under evolving system dynamics. In this paper, we advocate for a causal learning perspective to advance anomaly detection in spatially distributed infrastructures that grounds detection in structural cause-effect relationships. We identify and formalize three key directions: causal graph profiling, multi-view fusion, and continual causal graph learning, each offering distinct advantages in uncovering dynamic cause-effect structures across time and space. Drawing on real-world insights from systems such as water treatment infrastructures, we illustrate how causal models provide early warning signals and root cause attribution, addressing the limitations of black-box detectors. Looking ahead, we outline the future research agenda centered on multi-modality, generative AI-driven, and scalable adaptive causal frameworks. Our objective is to lay a new research trajectory toward scalable, adaptive, explainable, and spatially grounded anomaly detection systems. We hope to inspire a paradigm shift in cybersecurity research, promoting causality-driven approaches to address evolving threats in interconnected infrastructures. 

---
# KP-A: A Unified Network Knowledge Plane for Catalyzing Agentic Network Intelligence 

**Authors**: Yun Tang, Mengbang Zou, Zeinab Nezami, Syed Ali Raza Zaidi, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08164)  

**Abstract**: The emergence of large language models (LLMs) and agentic systems is enabling autonomous 6G networks with advanced intelligence, including self-configuration, self-optimization, and self-healing. However, the current implementation of individual intelligence tasks necessitates isolated knowledge retrieval pipelines, resulting in redundant data flows and inconsistent interpretations. Inspired by the service model unification effort in Open-RAN (to support interoperability and vendor diversity), we propose KP-A: a unified Network Knowledge Plane specifically designed for Agentic network intelligence. By decoupling network knowledge acquisition and management from intelligence logic, KP-A streamlines development and reduces maintenance complexity for intelligence engineers. By offering an intuitive and consistent knowledge interface, KP-A also enhances interoperability for the network intelligence agents. We demonstrate KP-A in two representative intelligence tasks: live network knowledge Q&A and edge AI service orchestration. All implementation artifacts have been open-sourced to support reproducibility and future standardization efforts. 

---
# AmpLyze: A Deep Learning Model for Predicting the Hemolytic Concentration 

**Authors**: Peng Qiu, Hanqi Feng, Barnabas Poczos  

**Link**: [PDF](https://arxiv.org/pdf/2507.08162)  

**Abstract**: Red-blood-cell lysis (HC50) is the principal safety barrier for antimicrobial-peptide (AMP) therapeutics, yet existing models only say "toxic" or "non-toxic." AmpLyze closes this gap by predicting the actual HC50 value from sequence alone and explaining the residues that drive toxicity. The model couples residue-level ProtT5/ESM2 embeddings with sequence-level descriptors in dual local and global branches, aligned by a cross-attention module and trained with log-cosh loss for robustness to assay noise. The optimal AmpLyze model reaches a PCC of 0.756 and an MSE of 0.987, outperforming classical regressors and the state-of-the-art. Ablations confirm that both branches are essential, and cross-attention adds a further 1% PCC and 3% MSE improvement. Expected-Gradients attributions reveal known toxicity hotspots and suggest safer substitutions. By turning hemolysis assessment into a quantitative, sequence-based, and interpretable prediction, AmpLyze facilitates AMP design and offers a practical tool for early-stage toxicity screening. 

---
# ALCo-FM: Adaptive Long-Context Foundation Model for Accident Prediction 

**Authors**: Pinaki Prasad Guha Neogi, Ahmad Mohammadshirazi, Rajiv Ramnath  

**Link**: [PDF](https://arxiv.org/pdf/2507.08153)  

**Abstract**: Traffic accidents are rare, yet high-impact events that require long-context multimodal reasoning for accurate risk forecasting. In this paper, we introduce ALCo-FM, a unified adaptive long-context foundation model that computes a volatility pre-score to dynamically select context windows for input data and encodes and fuses these multimodal data via shallow cross attention. Following a local GAT layer and a BigBird-style sparse global transformer over H3 hexagonal grids, coupled with Monte Carlo dropout for confidence, the model yields superior, well-calibrated predictions. Trained on data from 15 US cities with a class-weighted loss to counter label imbalance, and fine-tuned with minimal data on held-out cities, ALCo-FM achieves 0.94 accuracy, 0.92 F1, and an ECE of 0.04, outperforming more than 20 state-of-the-art baselines in large-scale urban risk prediction. Code and dataset are available at: this https URL 

---
# Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores 

**Authors**: Vivek Chari, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2507.08143)  

**Abstract**: Modern Large Language Models (LLMs) are increasingly trained to support very large context windows. Unfortunately the ability to use long contexts in generation is complicated by the large memory requirement of the KV cache, which scales linearly with the context length. This memory footprint is often the dominant resource bottleneck in real-world deployments, limiting throughput and increasing serving cost. One way to address this is by compressing the KV cache, which can be done either with knowledge of the question being asked (query-aware) or without knowledge of the query (query-agnostic). We present Compactor, a parameter-free, query-agnostic KV compression strategy that uses approximate leverage scores to determine token importance. We show that Compactor can achieve the same performance as competing methods while retaining 1/2 the tokens in both synthetic and real-world context tasks, with minimal computational overhead. We further introduce a procedure for context-calibrated compression, which allows one to infer the maximum compression ratio a given context can support. Using context-calibrated compression, we show that Compactor achieves full KV performance on Longbench while reducing the KV memory burden by 63%, on average. To demonstrate the efficacy and generalizability of our approach, we apply Compactor to 27 synthetic and real-world tasks from RULER and Longbench, with models from both the Qwen 2.5 and Llama 3.1 families. 

---
# Temporally Consistent Amodal Completion for 3D Human-Object Interaction Reconstruction 

**Authors**: Hyungjun Doh, Dong In Lee, Seunggeun Chi, Pin-Hao Huang, Kwonjoon Lee, Sangpil Kim, Karthik Ramani  

**Link**: [PDF](https://arxiv.org/pdf/2507.08137)  

**Abstract**: We introduce a novel framework for reconstructing dynamic human-object interactions from monocular video that overcomes challenges associated with occlusions and temporal inconsistencies. Traditional 3D reconstruction methods typically assume static objects or full visibility of dynamic subjects, leading to degraded performance when these assumptions are violated-particularly in scenarios where mutual occlusions occur. To address this, our framework leverages amodal completion to infer the complete structure of partially obscured regions. Unlike conventional approaches that operate on individual frames, our method integrates temporal context, enforcing coherence across video sequences to incrementally refine and stabilize reconstructions. This template-free strategy adapts to varying conditions without relying on predefined models, significantly enhancing the recovery of intricate details in dynamic scenes. We validate our approach using 3D Gaussian Splatting on challenging monocular videos, demonstrating superior precision in handling occlusions and maintaining temporal stability compared to existing techniques. 

---
# Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models 

**Authors**: Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.08128)  

**Abstract**: We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets. 

---
# Quasi-Random Physics-informed Neural Networks 

**Authors**: Tianchi Yu, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2507.08121)  

**Abstract**: Physics-informed neural networks have shown promise in solving partial differential equations (PDEs) by integrating physical constraints into neural network training, but their performance is sensitive to the sampling of points. Based on the impressive performance of quasi Monte-Carlo methods in high dimensional problems, this paper proposes Quasi-Random Physics-Informed Neural Networks (QRPINNs), which use low-discrepancy sequences for sampling instead of random points directly from the domain. Theoretically, QRPINNs have been proven to have a better convergence rate than PINNs. Empirically, experiments demonstrate that QRPINNs significantly outperform PINNs and some representative adaptive sampling methods, especially in high-dimensional PDEs. Furthermore, combining QRPINNs with adaptive sampling can further improve the performance. 

---
# VideoConviction: A Multimodal Benchmark for Human Conviction and Stock Market Recommendations 

**Authors**: Michael Galarnyk, Veer Kejriwal, Agam Shah, Yash Bhardwaj, Nicholas Meyer, Anand Krishnan, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2507.08104)  

**Abstract**: Social media has amplified the reach of financial influencers known as "finfluencers," who share stock recommendations on platforms like YouTube. Understanding their influence requires analyzing multimodal signals like tone, delivery style, and facial expressions, which extend beyond text-based financial analysis. We introduce VideoConviction, a multimodal dataset with 6,000+ expert annotations, produced through 457 hours of human effort, to benchmark multimodal large language models (MLLMs) and text-based large language models (LLMs) in financial discourse. Our results show that while multimodal inputs improve stock ticker extraction (e.g., extracting Apple's ticker AAPL), both MLLMs and LLMs struggle to distinguish investment actions and conviction--the strength of belief conveyed through confident delivery and detailed reasoning--often misclassifying general commentary as definitive recommendations. While high-conviction recommendations perform better than low-conviction ones, they still underperform the popular S\&P 500 index fund. An inverse strategy--betting against finfluencer recommendations--outperforms the S\&P 500 by 6.8\% in annual returns but carries greater risk (Sharpe ratio of 0.41 vs. 0.65). Our benchmark enables a diverse evaluation of multimodal tasks, comparing model performance on both full video and segmented video inputs. This enables deeper advancements in multimodal financial research. Our code, dataset, and evaluation leaderboard are available under the CC BY-NC 4.0 license. 

---
# An Object-Based Deep Learning Approach for Building Height Estimation from Single SAR Images 

**Authors**: Babak Memar, Luigi Russo, Silvia Liberata Ullo, Paolo Gamba  

**Link**: [PDF](https://arxiv.org/pdf/2507.08096)  

**Abstract**: Accurate estimation of building heights using very high resolution (VHR) synthetic aperture radar (SAR) imagery is crucial for various urban applications. This paper introduces a Deep Learning (DL)-based methodology for automated building height estimation from single VHR COSMO-SkyMed images: an object-based regression approach based on bounding box detection followed by height estimation. This model was trained and evaluated on a unique multi-continental dataset comprising eight geographically diverse cities across Europe, North and South America, and Asia, employing a cross-validation strategy to explicitly assess out-of-distribution (OOD) generalization. The results demonstrate highly promising performance, particularly on European cities where the model achieves a Mean Absolute Error (MAE) of approximately one building story (2.20 m in Munich), significantly outperforming recent state-of-the-art methods in similar OOD scenarios. Despite the increased variability observed when generalizing to cities in other continents, particularly in Asia with its distinct urban typologies and prevalence of high-rise structures, this study underscores the significant potential of DL for robust cross-city and cross-continental transfer learning in building height estimation from single VHR SAR data. 

---
# Tree-Structured Parzen Estimator Can Solve Black-Box Combinatorial Optimization More Efficiently 

**Authors**: Kenshin Abe, Yunzhuo Wang, Shuhei Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2507.08053)  

**Abstract**: Tree-structured Parzen estimator (TPE) is a versatile hyperparameter optimization (HPO) method supported by popular HPO tools. Since these HPO tools have been developed in line with the trend of deep learning (DL), the problem setups often used in the DL domain have been discussed for TPE such as multi-objective optimization and multi-fidelity optimization. However, the practical applications of HPO are not limited to DL, and black-box combinatorial optimization is actively utilized in some domains, e.g., chemistry and biology. As combinatorial optimization has been an untouched, yet very important, topic in TPE, we propose an efficient combinatorial optimization algorithm for TPE. In this paper, we first generalize the categorical kernel with the numerical kernel in TPE, enabling us to introduce a distance structure to the categorical kernel. Then we discuss modifications for the newly developed kernel to handle a large combinatorial search space. These modifications reduce the time complexity of the kernel calculation with respect to the size of a combinatorial search space. In the experiments using synthetic problems, we verified that our proposed method identifies better solutions with fewer evaluations than the original TPE. Our algorithm is available in Optuna, an open-source framework for HPO. 

---
# An Enhanced Privacy-preserving Federated Few-shot Learning Framework for Respiratory Disease Diagnosis 

**Authors**: Ming Wang, Zhaoyang Duan, Dong Xue, Fangzhou Liu, Zhongheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08050)  

**Abstract**: The labor-intensive nature of medical data annotation presents a significant challenge for respiratory disease diagnosis, resulting in a scarcity of high-quality labeled datasets in resource-constrained settings. Moreover, patient privacy concerns complicate the direct sharing of local medical data across institutions, and existing centralized data-driven approaches, which rely on amounts of available data, often compromise data privacy. This study proposes a federated few-shot learning framework with privacy-preserving mechanisms to address the issues of limited labeled data and privacy protection in diagnosing respiratory diseases. In particular, a meta-stochastic gradient descent algorithm is proposed to mitigate the overfitting problem that arises from insufficient data when employing traditional gradient descent methods for neural network training. Furthermore, to ensure data privacy against gradient leakage, differential privacy noise from a standard Gaussian distribution is integrated into the gradients during the training of private models with local data, thereby preventing the reconstruction of medical images. Given the impracticality of centralizing respiratory disease data dispersed across various medical institutions, a weighted average algorithm is employed to aggregate local diagnostic models from different clients, enhancing the adaptability of a model across diverse scenarios. Experimental results show that the proposed method yields compelling results with the implementation of differential privacy, while effectively diagnosing respiratory diseases using data from different structures, categories, and distributions. 

---
# Krul: Efficient State Restoration for Multi-turn Conversations with Dynamic Cross-layer KV Sharing 

**Authors**: Junyi Wen, Junyuan Liang, Zicong Hong, Wuhui Chen, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08045)  

**Abstract**: Efficient state restoration in multi-turn conversations with large language models (LLMs) remains a critical challenge, primarily due to the overhead of recomputing or loading full key-value (KV) caches for all historical tokens. To address this, existing approaches compress KV caches across adjacent layers with highly similar attention patterns. However, these methods often apply a fixed compression scheme across all conversations, selecting the same layer pairs for compression without considering conversation-specific attention dynamics. This static strategy overlooks variability in attention pattern similarity across different conversations, which can lead to noticeable accuracy degradation.
We present Krul, a multi-turn LLM inference system that enables accurate and efficient KV cache restoration. Krul dynamically selects compression strategies based on attention similarity across layer pairs and uses a recomputation-loading pipeline to restore the KV cache. It introduces three key innovations: 1) a preemptive compression strategy selector to preserve critical context for future conversation turns and selects a customized strategy for the conversation; 2) a token-wise heterogeneous attention similarity estimator to mitigate the attention similarity computation and storage overhead during model generation; 3) a bubble-free restoration scheduler to reduce potential bubbles brought by the imbalance of recomputing and loading stream due to compressed KV caches. Empirical evaluations on real-world tasks demonstrate that Krul achieves a 1.5x-2.68x reduction in time-to-first-token (TTFT) and a 1.33x-2.35x reduction in KV cache storage compared to state-of-the-art methods without compromising generation quality. 

---
# ConsNoTrainLoRA: Data-driven Weight Initialization of Low-rank Adapters using Constraints 

**Authors**: Debasmit Das, Hyoungwoo Park, Munawar Hayat, Seokeon Choi, Sungrack Yun, Fatih Porikli  

**Link**: [PDF](https://arxiv.org/pdf/2507.08044)  

**Abstract**: Foundation models are pre-trained on large-scale datasets and subsequently fine-tuned on small-scale datasets using parameter-efficient fine-tuning (PEFT) techniques like low-rank adapters (LoRA). In most previous works, LoRA weight matrices are randomly initialized with a fixed rank across all attachment points. In this paper, we improve convergence and final performance of LoRA fine-tuning, using our proposed data-driven weight initialization method, ConsNoTrainLoRA (CNTLoRA). We express LoRA initialization as a domain shift problem where we use multiple constraints relating the pre-training and fine-tuning activations. By reformulating these constraints, we obtain a closed-form estimate of LoRA weights that depends on pre-training weights and fine-tuning activation vectors and hence requires no training during initialization. This weight estimate is decomposed to initialize the up and down matrices with proposed flexibility of variable ranks. With the proposed initialization method, we fine-tune on downstream tasks such as image generation, image classification and image understanding. Both quantitative and qualitative results demonstrate that CNTLoRA outperforms standard and data-driven weight initialization methods. Extensive analyses and ablations further elucidate the design choices of our framework, providing an optimal recipe for faster convergence and enhanced performance. 

---
# Towards Evaluating Robustness of Prompt Adherence in Text to Image Models 

**Authors**: Sujith Vemishetty, Advitiya Arora, Anupama Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08039)  

**Abstract**: The advancements in the domain of LLMs in recent years have surprised many, showcasing their remarkable capabilities and diverse applications. Their potential applications in various real-world scenarios have led to significant research on their reliability and effectiveness. On the other hand, multimodal LLMs and Text-to-Image models have only recently gained prominence, especially when compared to text-only LLMs. Their reliability remains constrained due to insufficient research on assessing their performance and robustness. This paper aims to establish a comprehensive evaluation framework for Text-to-Image models, concentrating particularly on their adherence to prompts. We created a novel dataset that aimed to assess the robustness of these models in generating images that conform to the specified factors of variation in the input text prompts. Our evaluation studies present findings on three variants of Stable Diffusion models: Stable Diffusion 3 Medium, Stable Diffusion 3.5 Large, and Stable Diffusion 3.5 Large Turbo, and two variants of Janus models: Janus Pro 1B and Janus Pro 7B. We introduce a pipeline that leverages text descriptions generated by the gpt-4o model for our ground-truth images, which are then used to generate artificial images by passing these descriptions to the Text-to-Image models. We then pass these generated images again through gpt-4o using the same system prompt and compare the variation between the two descriptions. Our results reveal that these models struggle to create simple binary images with only two factors of variation: a simple geometric shape and its location. We also show, using pre-trained VAEs on our dataset, that they fail to generate images that follow our input dataset distribution. 

---
# AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research 

**Authors**: Talor Abramovich, Gal Chechik  

**Link**: [PDF](https://arxiv.org/pdf/2507.08038)  

**Abstract**: Autonomous agents built on language models (LMs) are showing increasing popularity in many fields, including scientific research. AI co-scientists aim to support or automate parts of the research process using these agents. A key component of empirical AI research is the design of ablation experiments. To this end, we introduce AblationBench, a benchmark suite for evaluating agents on ablation planning tasks in empirical AI research. It includes two tasks: AuthorAblation, which helps authors propose ablation experiments based on a method section and contains 83 instances, and ReviewerAblation, which helps reviewers find missing ablations in a full paper and contains 350 instances. For both tasks, we develop LM-based judges that serve as an automatic evaluation framework. Our experiments with frontier LMs show that these tasks remain challenging, with the best-performing LM system identifying only 29% of the original ablations on average. Lastly, we analyze the limitations of current LMs on these tasks, and find that chain-of-thought prompting outperforms the currently existing agent-based approach. 

---
# CRISP: Complex Reasoning with Interpretable Step-based Plans 

**Authors**: Matan Vetzler, Koren Lazar, Guy Uziel, Eran Hirsch, Ateret Anaby-Tavor, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08037)  

**Abstract**: Recent advancements in large language models (LLMs) underscore the need for stronger reasoning capabilities to solve complex problems effectively. While Chain-of-Thought (CoT) reasoning has been a step forward, it remains insufficient for many domains. A promising alternative is explicit high-level plan generation, but existing approaches largely assume that LLMs can produce effective plans through few-shot prompting alone, without additional training. In this work, we challenge this assumption and introduce CRISP (Complex Reasoning with Interpretable Step-based Plans), a multi-domain dataset of high-level plans for mathematical reasoning and code generation. The plans in CRISP are automatically generated and rigorously validated--both intrinsically, using an LLM as a judge, and extrinsically, by evaluating their impact on downstream task performance. We demonstrate that fine-tuning a small model on CRISP enables it to generate higher-quality plans than much larger models using few-shot prompting, while significantly outperforming Chain-of-Thought reasoning. Furthermore, our out-of-domain evaluation reveals that fine-tuning on one domain improves plan generation in the other, highlighting the generalizability of learned planning capabilities. 

---
# Integrating External Tools with Large Language Models to Improve Accuracy 

**Authors**: Nripesh Niketan, Hadj Batatia  

**Link**: [PDF](https://arxiv.org/pdf/2507.08034)  

**Abstract**: This paper deals with improving querying large language models (LLMs). It is well-known that without relevant contextual information, LLMs can provide poor quality responses or tend to hallucinate. Several initiatives have proposed integrating LLMs with external tools to provide them with up-to-date data to improve accuracy. In this paper, we propose a framework to integrate external tools to enhance the capabilities of LLMs in answering queries in educational settings. Precisely, we develop a framework that allows accessing external APIs to request additional relevant information. Integrated tools can also provide computational capabilities such as calculators or calendars. The proposed framework has been evaluated using datasets from the Multi-Modal Language Understanding (MMLU) collection. The data consists of questions on mathematical and scientific reasoning. Results compared to state-of-the-art language models show that the proposed approach significantly improves performance. Our Athena framework achieves 83% accuracy in mathematical reasoning and 88% in scientific reasoning, substantially outperforming all tested models including GPT-4o, LLaMA-Large, Mistral-Large, Phi-Large, and GPT-3.5, with the best baseline model (LLaMA-Large) achieving only 67% and 79% respectively. These promising results open the way to creating complex computing ecosystems around LLMs to make their use more natural to support various tasks and activities. 

---
# SSSUMO: Real-Time Semi-Supervised Submovement Decomposition 

**Authors**: Evgenii Rudakov, Jonathan Shock, Otto Lappi, Benjamin Ultan Cowley  

**Link**: [PDF](https://arxiv.org/pdf/2507.08028)  

**Abstract**: This paper introduces a SSSUMO, semi-supervised deep learning approach for submovement decomposition that achieves state-of-the-art accuracy and speed. While submovement analysis offers valuable insights into motor control, existing methods struggle with reconstruction accuracy, computational cost, and validation, due to the difficulty of obtaining hand-labeled data. We address these challenges using a semi-supervised learning framework. This framework learns from synthetic data, initially generated from minimum-jerk principles and then iteratively refined through adaptation to unlabeled human movement data. Our fully convolutional architecture with differentiable reconstruction significantly surpasses existing methods on both synthetic and diverse human motion datasets, demonstrating robustness even in high-noise conditions. Crucially, the model operates in real-time (less than a millisecond per input second), a substantial improvement over optimization-based techniques. This enhanced performance facilitates new applications in human-computer interaction, rehabilitation medicine, and motor control studies. We demonstrate the model's effectiveness across diverse human-performed tasks such as steering, rotation, pointing, object moving, handwriting, and mouse-controlled gaming, showing notable improvements particularly on challenging datasets where traditional methods largely fail. Training and benchmarking source code, along with pre-trained model weights, are made publicly available at this https URL. 

---
# Unveiling Effective In-Context Configurations for Image Captioning: An External & Internal Analysis 

**Authors**: Li Li, Yongliang Wu, Jingze Zhu, Jiawei Peng, Jianfei Cai, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08021)  

**Abstract**: The evolution of large models has witnessed the emergence of In-Context Learning (ICL) capabilities. In Natural Language Processing (NLP), numerous studies have demonstrated the effectiveness of ICL. Inspired by the success of Large Language Models (LLMs), researchers have developed Large Multimodal Models (LMMs) with ICL capabilities. However, explorations of demonstration configuration for multimodal ICL remain preliminary. Additionally, the controllability of In-Context Examples (ICEs) provides an efficient and cost-effective means to observe and analyze the inference characteristics of LMMs under varying inputs. This paper conducts a comprehensive external and internal investigation of multimodal in-context learning on the image captioning task. Externally, we explore demonstration configuration strategies through three dimensions: shot number, image retrieval, and caption assignment. We employ multiple metrics to systematically and thoroughly evaluate and summarize key findings. Internally, we analyze typical LMM attention characteristics and develop attention-based metrics to quantify model behaviors. We also conduct auxiliary experiments to explore the feasibility of attention-driven model acceleration and compression. We further compare performance variations between LMMs with identical model design and pretraining strategies and explain the differences from the angles of pre-training data features. Our study reveals both how ICEs configuration strategies impact model performance through external experiments and characteristic typical patterns through internal inspection, providing dual perspectives for understanding multimodal ICL in LMMs. Our method of combining external and internal analysis to investigate large models, along with our newly proposed metrics, can be applied to broader research areas. 

---
# Circumventing Safety Alignment in Large Language Models Through Embedding Space Toxicity Attenuation 

**Authors**: Zhibo Zhang, Yuxi Li, Kailong Wang, Shuai Yuan, Ling Shi, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08020)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across domains such as healthcare, education, and cybersecurity. However, this openness also introduces significant security risks, particularly through embedding space poisoning, which is a subtle attack vector where adversaries manipulate the internal semantic representations of input data to bypass safety alignment mechanisms. While previous research has investigated universal perturbation methods, the dynamics of LLM safety alignment at the embedding level remain insufficiently understood. Consequently, more targeted and accurate adversarial perturbation techniques, which pose significant threats, have not been adequately studied.
In this work, we propose ETTA (Embedding Transformation Toxicity Attenuation), a novel framework that identifies and attenuates toxicity-sensitive dimensions in embedding space via linear transformations. ETTA bypasses model refusal behaviors while preserving linguistic coherence, without requiring model fine-tuning or access to training data. Evaluated on five representative open-source LLMs using the AdvBench benchmark, ETTA achieves a high average attack success rate of 88.61%, outperforming the best baseline by 11.34%, and generalizes to safety-enhanced models (e.g., 77.39% ASR on instruction-tuned defenses). These results highlight a critical vulnerability in current alignment strategies and underscore the need for embedding-aware defenses. 

---
# Mechanistic Indicators of Understanding in Large Language Models 

**Authors**: Pierre Beckmann, Matthieu Queloz  

**Link**: [PDF](https://arxiv.org/pdf/2507.08017)  

**Abstract**: Recent findings in mechanistic interpretability (MI), the field probing the inner workings of Large Language Models (LLMs), challenge the view that these models rely solely on superficial statistics. Here, we offer an accessible synthesis of these findings that doubles as an introduction to MI, all while integrating these findings within a novel theoretical framework for thinking about machine understanding. We argue that LLMs develop internal structures that are functionally analogous to the kind of understanding that consists in seeing connections. To sharpen this idea, we propose a three-tiered conception of machine understanding. First, conceptual understanding emerges when a model forms "features" as directions in latent space, thereby learning the connections between diverse manifestations of something. Second, state-of-the-world understanding emerges when a model learns contingent factual connections between features and dynamically tracks changes in the world. Third, principled understanding emerges when a model ceases to rely on a collection of memorized facts and discovers a "circuit" that connects these facts. However, we conclude by exploring the "parallel mechanisms" phenomenon, arguing that while LLMs exhibit forms of understanding, their cognitive architecture remains different from ours, and the debate should shift from whether LLMs understand to how their strange minds work. 

---
# Mass-Scale Analysis of In-the-Wild Conversations Reveals Complexity Bounds on LLM Jailbreaking 

**Authors**: Aldan Creo, Raul Castro Fernandez, Manuel Cebrian  

**Link**: [PDF](https://arxiv.org/pdf/2507.08014)  

**Abstract**: As large language models (LLMs) become increasingly deployed, understanding the complexity and evolution of jailbreaking strategies is critical for AI safety.
We present a mass-scale empirical analysis of jailbreak complexity across over 2 million real-world conversations from diverse platforms, including dedicated jailbreaking communities and general-purpose chatbots. Using a range of complexity metrics spanning probabilistic measures, lexical diversity, compression ratios, and cognitive load indicators, we find that jailbreak attempts do not exhibit significantly higher complexity than normal conversations. This pattern holds consistently across specialized jailbreaking communities and general user populations, suggesting practical bounds on attack sophistication. Temporal analysis reveals that while user attack toxicity and complexity remains stable over time, assistant response toxicity has decreased, indicating improving safety mechanisms. The absence of power-law scaling in complexity distributions further points to natural limits on jailbreak development.
Our findings challenge the prevailing narrative of an escalating arms race between attackers and defenders, instead suggesting that LLM safety evolution is bounded by human ingenuity constraints while defensive measures continue advancing. Our results highlight critical information hazards in academic jailbreak disclosure, as sophisticated attacks exceeding current complexity baselines could disrupt the observed equilibrium and enable widespread harm before defensive adaptation. 

---
# MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model 

**Authors**: K. Sahit Reddy, N. Ragavenderan, Vasanth K., Ganesh N. Naik, Vishalakshi Prabhu, Nagaraja G. S  

**Link**: [PDF](https://arxiv.org/pdf/2507.08013)  

**Abstract**: Recent advances in natural language processing (NLP) have been driven bypretrained language models like BERT, RoBERTa, T5, and GPT. Thesemodels excel at understanding complex texts, but biomedical literature, withits domain-specific terminology, poses challenges that models likeWord2Vec and bidirectional long short-term memory (Bi-LSTM) can't fullyaddress. GPT and T5, despite capturing context, fall short in tasks needingbidirectional understanding, unlike BERT. Addressing this, we proposedMedicalBERT, a pretrained BERT model trained on a large biomedicaldataset and equipped with domain-specific vocabulary that enhances thecomprehension of biomedical terminology. MedicalBERT model is furtheroptimized and fine-tuned to address diverse tasks, including named entityrecognition, relation extraction, question answering, sentence similarity, anddocument classification. Performance metrics such as the F1-score,accuracy, and Pearson correlation are employed to showcase the efficiencyof our model in comparison to other BERT-based models such as BioBERT,SciBERT, and ClinicalBERT. MedicalBERT outperforms these models onmost of the benchmarks, and surpasses the general-purpose BERT model by5.67% on average across all the tasks evaluated respectively. This work alsounderscores the potential of leveraging pretrained BERT models for medicalNLP tasks, demonstrating the effectiveness of transfer learning techniques incapturing domain-specific information.
(PDF) MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model. Available from: this https URL [accessed Jul 06 2025]. 

---
# RepeaTTS: Towards Feature Discovery through Repeated Fine-Tuning 

**Authors**: Atli Sigurgeirsson, Simon King  

**Link**: [PDF](https://arxiv.org/pdf/2507.08012)  

**Abstract**: A Prompt-based Text-To-Speech model allows a user to control different aspects of speech, such as speaking rate and perceived gender, through natural language instruction. Although user-friendly, such approaches are on one hand constrained: control is limited to acoustic features exposed to the model during training, and too flexible on the other: the same inputs yields uncontrollable variation that are reflected in the corpus statistics.
We investigate a novel fine-tuning regime to address both of these issues at the same time by exploiting the uncontrollable variance of the model. Through principal component analysis of thousands of synthesised samples, we determine latent features that account for the highest proportion of the output variance and incorporate them as new labels for secondary fine-tuning. We evaluate the proposed methods on two models trained on an expressive Icelandic speech corpus, one with emotional disclosure and one without. In the case of the model without emotional disclosure, the method yields both continuous and discrete features that improve overall controllability of the model. 

---
# Energy Management for Renewable-Colocated Artificial Intelligence Data Centers 

**Authors**: Siying Li, Lang Tong, Timothy D. Mount  

**Link**: [PDF](https://arxiv.org/pdf/2507.08011)  

**Abstract**: We develop an energy management system (EMS) for artificial intelligence (AI) data centers with colocated renewable generation. Under a profit-maximizing framework, the EMS of renewable-colocated data center (RCDC) co-optimizes AI workload scheduling, on-site renewable utilization, and electricity market participation. Within both wholesale and retail market participation models, the economic benefit of the RCDC operation is maximized. Empirical evaluations using real-world traces of electricity prices, data center power consumption, and renewable generation demonstrate significant profit gains from renewable and AI data center colocations. 

---
# Unraveling the Potential of Diffusion Models in Small Molecule Generation 

**Authors**: Peining Zhang, Daniel Baker, Minghu Song, Jinbo Bi  

**Link**: [PDF](https://arxiv.org/pdf/2507.08005)  

**Abstract**: Generative AI presents chemists with novel ideas for drug design and facilitates the exploration of vast chemical spaces. Diffusion models (DMs), an emerging tool, have recently attracted great attention in drug R\&D. This paper comprehensively reviews the latest advancements and applications of DMs in molecular generation. It begins by introducing the theoretical principles of DMs. Subsequently, it categorizes various DM-based molecular generation methods according to their mathematical and chemical applications. The review further examines the performance of these models on benchmark datasets, with a particular focus on comparing the generation performance of existing 3D methods. Finally, it concludes by emphasizing current challenges and suggesting future research directions to fully exploit the potential of DMs in drug discovery. 

---
# Human vs. LLM-Based Thematic Analysis for Digital Mental Health Research: Proof-of-Concept Comparative Study 

**Authors**: Karisa Parkington, Bazen G. Teferra, Marianne Rouleau-Tang, Argyrios Perivolaris, Alice Rueda, Adam Dubrowski, Bill Kapralos, Reza Samavi, Andrew Greenshaw, Yanbo Zhang, Bo Cao, Yuqi Wu, Sirisha Rambhatla, Sridhar Krishnan, Venkat Bhat  

**Link**: [PDF](https://arxiv.org/pdf/2507.08002)  

**Abstract**: Thematic analysis provides valuable insights into participants' experiences through coding and theme development, but its resource-intensive nature limits its use in large healthcare studies. Large language models (LLMs) can analyze text at scale and identify key content automatically, potentially addressing these challenges. However, their application in mental health interviews needs comparison with traditional human analysis. This study evaluates out-of-the-box and knowledge-base LLM-based thematic analysis against traditional methods using transcripts from a stress-reduction trial with healthcare workers. OpenAI's GPT-4o model was used along with the Role, Instructions, Steps, End-Goal, Narrowing (RISEN) prompt engineering framework and compared to human analysis in Dedoose. Each approach developed codes, noted saturation points, applied codes to excerpts for a subset of participants (n = 20), and synthesized data into themes. Outputs and performance metrics were compared directly. LLMs using the RISEN framework developed deductive parent codes similar to human codes, but humans excelled in inductive child code development and theme synthesis. Knowledge-based LLMs reached coding saturation with fewer transcripts (10-15) than the out-of-the-box model (15-20) and humans (90-99). The out-of-the-box LLM identified a comparable number of excerpts to human researchers, showing strong inter-rater reliability (K = 0.84), though the knowledge-based LLM produced fewer excerpts. Human excerpts were longer and involved multiple codes per excerpt, while LLMs typically applied one code. Overall, LLM-based thematic analysis proved more cost-effective but lacked the depth of human analysis. LLMs can transform qualitative analysis in mental healthcare and clinical research when combined with human oversight to balance participant perspectives and research resources. 

---
# Dual-Attention U-Net++ with Class-Specific Ensembles and Bayesian Hyperparameter Optimization for Precise Wound and Scale Marker Segmentation 

**Authors**: Daniel Cieślak, Miriam Reca, Olena Onyshchenko, Jacek Rumiński  

**Link**: [PDF](https://arxiv.org/pdf/2507.05314)  

**Abstract**: Accurate segmentation of wounds and scale markers in clinical images remainsa significant challenge, crucial for effective wound management and automatedassessment. In this study, we propose a novel dual-attention U-Net++ archi-tecture, integrating channel-wise (SCSE) and spatial attention mechanisms toaddress severe class imbalance and variability in medical images this http URL, extensive benchmarking across diverse architectures and encoders via 5-fold cross-validation identified EfficientNet-B7 as the optimal encoder this http URL, we independently trained two class-specific models with tailoredpreprocessing, extensive data augmentation, and Bayesian hyperparameter tun-ing (WandB sweeps). The final model ensemble utilized Test Time Augmentationto further enhance prediction reliability. Our approach was evaluated on a bench-mark dataset from the NBC 2025 & PCBBE 2025 competition. Segmentationperformance was quantified using a weighted F1-score (75% wounds, 25% scalemarkers), calculated externally by competition organizers on undisclosed hard-ware. The proposed approach achieved an F1-score of 0.8640, underscoring itseffectiveness for complex medical segmentation tasks. 

---
