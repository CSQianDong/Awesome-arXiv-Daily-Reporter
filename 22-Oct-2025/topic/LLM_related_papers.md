# Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models 

**Authors**: Hanze Guo, Jing Yao, Xiao Zhou, Xiaoyuan Yi, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.18526)  

**Abstract**: As large language models (LLMs) become increasingly integrated into applications serving users across diverse cultures, communities and demographics, it is critical to align LLMs with pluralistic human values beyond average principles (e.g., HHH). In psychological and social value theories such as Schwartz's Value Theory, pluralistic values are represented by multiple value dimensions paired with various priorities. However, existing methods encounter two challenges when aligning with such fine-grained value objectives: 1) they often treat multiple values as independent and equally important, ignoring their interdependence and relative priorities (value complexity); 2) they struggle to precisely control nuanced value priorities, especially those underrepresented ones (value steerability). To handle these challenges, we propose COUPLE, a COUnterfactual reasoning framework for PLuralistic valuE alignment. It introduces a structural causal model (SCM) to feature complex interdependency and prioritization among features, as well as the causal relationship between high-level value dimensions and behaviors. Moreover, it applies counterfactual reasoning to generate outputs aligned with any desired value objectives. Benefitting from explicit causal modeling, COUPLE also provides better interpretability. We evaluate COUPLE on two datasets with different value systems and demonstrate that COUPLE advances other baselines across diverse types of value objectives. 

---
# Sherlock Your Queries: Learning to Ask the Right Questions for Dialogue-Based Retrieval 

**Authors**: Dong Yun, Marco Schouten, Dim Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.18659)  

**Abstract**: User queries in information retrieval are often ambiguous, making it challenging for systems to identify a user's target from a single query. While recent dialogue-based interactive retrieval systems can clarify user intent, they are inefficient as they often lack an explicit strategy to ask the most informative questions. To address this limitation, we propose SherlockLLM, a dialogue-driven retrieval framework that learns an optimal questioning strategy via Reinforcement Learning (RL) and avoids the need for large-scale annotated dialogue data. In our framework, an agent is trained to generate a sequence of binary questions to efficiently narrow down the search space. To validate our approach, we introduce a benchmark with both structured and unstructured tasks. Experimental results show that SherlockLLM is a robust and efficient solution. On the structured tasks, its performance matches strong baselines and approaches the theoretical optimal defined by binary search. On the challenging unstructured task, our agent significantly outperforms these baselines, showcasing its ability to learn a highly effective information-seeking dialogue policy. 

---
# LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources 

**Authors**: Haichao Ji, Zibo Wang, Yifei Zhu, Meng han, Dan Wang, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.18477)  

**Abstract**: Large Language Models (LLMs) have shown great promise in automating data analytics tasks by interpreting natural language queries and generating multi-operation execution plans. However, existing LLM-agent-based analytics frameworks operate under the assumption of centralized data access, offering little to no privacy protection. In contrast, federated analytics (FA) enables privacy-preserving computation across distributed data sources, but lacks support for natural language input and requires structured, machine-readable queries. In this work, we present LAFA, the first system that integrates LLM-agent-based data analytics with FA. LAFA introduces a hierarchical multi-agent architecture that accepts natural language queries and transforms them into optimized, executable FA workflows. A coarse-grained planner first decomposes complex queries into sub-queries, while a fine-grained planner maps each subquery into a Directed Acyclic Graph of FA operations using prior structural knowledge. To improve execution efficiency, an optimizer agent rewrites and merges multiple DAGs, eliminating redundant operations and minimizing computational and communicational overhead. Our experiments demonstrate that LAFA consistently outperforms baseline prompting strategies by achieving higher execution plan success rates and reducing resource-intensive FA operations by a substantial margin. This work establishes a practical foundation for privacy-preserving, LLM-driven analytics that supports natural language input in the FA setting. 

---
# Probabilistic Modeling of Intentions in Socially Intelligent LLM Agents 

**Authors**: Feifan Xia, Yuyang Fang, Defang Li, Yantong Xie, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18476)  

**Abstract**: We present a probabilistic intent modeling framework for large language model (LLM) agents in multi-turn social dialogue. The framework maintains a belief distribution over a partner's latent intentions, initialized from contextual priors and dynamically updated through likelihood estimation after each utterance. The evolving distribution provides additional contextual grounding for the policy, enabling adaptive dialogue strategies under uncertainty. Preliminary experiments in the SOTOPIA environment show consistent improvements: the proposed framework increases the Overall score by 9.0% on SOTOPIA-All and 4.1% on SOTOPIA-Hard compared with the Qwen2.5-7B baseline, and slightly surpasses an oracle agent that directly observes partner intentions. These early results suggest that probabilistic intent modeling can contribute to the development of socially intelligent LLM agents. 

---
# PlanU: Large Language Model Decision Making through Planning under Uncertainty 

**Authors**: Ziwei Deng, Mian Deng, Chenjing Liang, Zeming Gao, Chennan Ma, Chenxing Lin, Haipeng Zhang, Songzhu Mei, Cheng Wang, Siqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18442)  

**Abstract**: Large Language Models (LLMs) are increasingly being explored across a range of decision-making tasks. However, LLMs sometimes struggle with decision-making tasks under uncertainty that are relatively easy for humans, such as planning actions in stochastic environments. The adoption of LLMs for decision-making is impeded by uncertainty challenges, such as LLM uncertainty and environmental uncertainty. LLM uncertainty arises from the stochastic sampling process inherent to LLMs. Most LLM-based Decision-Making (LDM) approaches address LLM uncertainty through multiple reasoning chains or search trees. However, these approaches overlook environmental uncertainty, which leads to poor performance in environments with stochastic state transitions. Some recent LDM approaches deal with uncertainty by forecasting the probability of unknown variables. However, they are not designed for multi-step decision-making tasks that require interaction with the environment. To address uncertainty in LLM decision-making, we introduce PlanU, an LLM-based planning method that captures uncertainty within Monte Carlo Tree Search (MCTS). PlanU models the return of each node in the MCTS as a quantile distribution, which uses a set of quantiles to represent the return distribution. To balance exploration and exploitation during tree search, PlanU introduces an Upper Confidence Bounds with Curiosity (UCC) score which estimates the uncertainty of MCTS nodes. Through extensive experiments, we demonstrate the effectiveness of PlanU in LLM-based decision-making tasks under uncertainty. 

---
# CircuitSeer: Mining High-Quality Data by Probing Mathematical Reasoning Circuits in LLMs 

**Authors**: Shaobo Wang, Yongliang Miao, Yuancheng Liu, and Qianli Ma, Ning Liao, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18470)  

**Abstract**: Large language models (LLMs) have demonstrated impressive reasoning capabilities, but scaling their performance often relies on massive reasoning datasets that are computationally expensive to train on. Existing data selection methods aim to curate smaller, high-quality subsets but often rely on costly external models or opaque heuristics. In this work, we shift the focus from external heuristics to the model's internal mechanisms. We find that complex reasoning tasks consistently activate a sparse, specialized subset of attention heads, forming core reasoning circuits. Building on this insight, we propose CircuitSeer, a novel data selection method that quantifies the reasoning complexity of data by measuring its influence on these crucial circuits. Extensive experiments on 4 models and 9 datasets demonstrate CircuitSeer's superiority. Notably, fine-tuning Qwen2.5-Math-7B on just 10% of data selected by our method achieves a 1.4-point gain in average Pass@1 over training on the full dataset, highlighting its efficiency and effectiveness. 

---
# Illusions of reflection: open-ended task reveals systematic failures in Large Language Models' reflective reasoning 

**Authors**: Sion Weatherhead, Flora Salim, Aaron Belbasis  

**Link**: [PDF](https://arxiv.org/pdf/2510.18254)  

**Abstract**: Humans do not just find mistakes after the fact -- we often catch them mid-stream because 'reflection' is tied to the goal and its constraints. Today's large language models produce reasoning tokens and 'reflective' text, but is it functionally equivalent with human reflective reasoning? Prior work on closed-ended tasks -- with clear, external 'correctness' signals -- can make 'reflection' look effective while masking limits in self-correction. We therefore test eight frontier models on a simple, real-world task that is open-ended yet rule-constrained, with auditable success criteria: to produce valid scientific test items, then revise after considering their own critique. First-pass performance is poor (often zero valid items out of 4 required; mean $\approx$ 1), and reflection yields only modest gains (also $\approx$ 1). Crucially, the second attempt frequently repeats the same violation of constraint, indicating 'corrective gains' arise largely from chance production of a valid item rather than error detection and principled, constraint-sensitive repair. Performance before and after reflection deteriorates as open-endedness increases, and models marketed for 'reasoning' show no advantage. Our results suggest that current LLM 'reflection' lacks functional evidence of the active, goal-driven monitoring that helps humans respect constraints even on a first pass. Until such mechanisms are instantiated in the model itself, reliable performance requires external structure that enforces constraints. 

---
# LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior 

**Authors**: Man-Lin Chu, Lucian Terhorst, Kadin Reed, Tom Ni, Weiwei Chen, Rongyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.18155)  

**Abstract**: Simulating consumer decision-making is vital for designing and evaluating marketing strategies before costly real- world deployment. However, post-event analyses and rule-based agent-based models (ABMs) struggle to capture the complexity of human behavior and social interaction. We introduce an LLM-powered multi-agent simulation framework that models consumer decisions and social dynamics. Building on recent advances in large language model simulation in a sandbox envi- ronment, our framework enables generative agents to interact, express internal reasoning, form habits, and make purchasing decisions without predefined rules. In a price-discount marketing scenario, the system delivers actionable strategy-testing outcomes and reveals emergent social patterns beyond the reach of con- ventional methods. This approach offers marketers a scalable, low-risk tool for pre-implementation testing, reducing reliance on time-intensive post-event evaluations and lowering the risk of underperforming campaigns. 

---
# AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library 

**Authors**: Minwei Kong, Ao Qu, Xiaotong Guo, Wenbin Ouyang, Chonghe Jiang, Han Zheng, Yining Ma, Dingyi Zhuang, Yuhan Tang, Junyi Li, Hai Wang, Cathy Wu, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.18428)  

**Abstract**: Optimization modeling enables critical decisions across industries but remains difficult to automate: informal language must be mapped to precise mathematical formulations and executable solver code. Prior LLM approaches either rely on brittle prompting or costly retraining with limited generalization. We present AlphaOPT, a self-improving experience library that enables an LLM to learn from limited demonstrations (even answers alone, without gold-standard programs) and solver feedback - without annotated reasoning traces or parameter updates. AlphaOPT operates in a continual two-phase cycle: (i) a Library Learning phase that reflects on failed attempts, extracting solver-verified, structured insights as {taxonomy, condition, explanation, example}; and (ii) a Library Evolution phase that diagnoses retrieval misalignments and refines the applicability conditions of stored insights, improving transfer across tasks. This design (1) learns efficiently from limited demonstrations without curated rationales, (2) expands continually without costly retraining by updating the library rather than model weights, and (3) makes knowledge explicit and interpretable for human inspection and intervention. Experiments show that AlphaOPT steadily improves with more data (65% to 72% from 100 to 300 training items) and surpasses the strongest baseline by 7.7% on the out-of-distribution OptiBench dataset when trained only on answers. Code and data are available at: this https URL. 

---
# SMaRT: Select, Mix, and ReinvenT - A Strategy Fusion Framework for LLM-Driven Reasoning and Planning 

**Authors**: Nikhil Verma, Manasa Bharadwaj, Wonjun Jang, Harmanpreet Singh, Yixiao Wang, Homa Fashandi, Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.18095)  

**Abstract**: Large Language Models (LLMs) have redefined complex task automation with exceptional generalization capabilities. Despite these advancements, state-of-the-art methods rely on single-strategy prompting, missing the synergy of diverse reasoning approaches. No single strategy excels universally, highlighting the need for frameworks that fuse strategies to maximize performance and ensure robustness. We introduce the Select, Mix, and ReinvenT (SMaRT) framework, an innovative strategy fusion approach designed to overcome this constraint by creating balanced and efficient solutions through the seamless integration of diverse reasoning strategies. Unlike existing methods, which employ LLMs merely as evaluators, SMaRT uses them as intelligent integrators, unlocking the "best of all worlds" across tasks. Extensive empirical evaluations across benchmarks in reasoning, planning, and sequential decision-making highlight the robustness and adaptability of SMaRT. The framework consistently outperforms state-of-the-art baselines in solution quality, constraint adherence, and performance metrics. This work redefines LLM-driven decision-making by pioneering a new paradigm in cross-strategy calibration, unlocking superior outcomes for reasoning systems and advancing the boundaries of self-refining methodologies. 

---
# Measuring Reasoning in LLMs: a New Dialectical Angle 

**Authors**: Soheil Abbasloo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18134)  

**Abstract**: What does it truly mean for a language model to "reason"? Most current evaluations and benchmarks reward models' correct standalone answers--but correctness alone reveals little about the process that produced them. In this work, we explore a different perspective: reasoning is not a static chain of steps, but a dynamic trajectory where ideas interact, clash, and evolve into deeper insights. To capture this dynamic, we draw on a well-established philosophical tradition: \textit{dialectics}, where reasoning unfolds through thesis, antithesis, and synthesis. Building on this, we present SIEV, a structured framework that evaluates reasoning of LLMs through dialectics. Unlike conventional evaluations, SIEV assesses not only the conclusion a model reaches, but how it gets there: its ability to resolve tension, integrate distinct ideas, and synthesize higher-order reasoning. This lens uncovers significant reasoning gaps in state-of-the-art models even under saturated benchmarks like GSM and MMLU. For instance, GPT-5-chat, a recent model, loses over 40 points (out of 100) when evaluated with SIEV on GSM. Our findings highlight that adopting a process-oriented, philosophically grounded approach enables a deeper, more rigorous, and more discriminative assessment of LLM reasoning. 

---
# ssToken: Self-modulated and Semantic-aware Token Selection for LLM Fine-tuning 

**Authors**: Xiaohan Qin, Xiaoxing Wang, Ning Liao, Cancheng Zhang, Xiangdong Zhang, Mingquan Feng, Jingzhi Wang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.18250)  

**Abstract**: Data quality plays a critical role in enhancing supervised fine-tuning (SFT) for large language models (LLMs), and token-level data selection has emerged as a promising direction for its fine-grained nature. Despite their strong empirical performance, existing token-level selection methods share two key limitations: (1) requiring training or accessing an additional reference model, and (2) relying solely on loss information for token selection, which cannot well preserve semantically important tokens that are not favored by loss-based metrics. To address these challenges, we propose ssToken, a Self-modulated and Semantic-aware Token Selection approach. ssToken leverages readily accessible history models to compute the per-token loss difference with the current model, which serves as a self-modulated signal that enables the model to adaptively select tokens along its optimization trajectory, rather than relying on excess loss from an offline-trained reference model as in prior works. We further introduce a semantic-aware, attention-based token importance estimation metric, orthogonal to loss-based selection and providing complementary semantic information for more effective filtering. Extensive experiments across different model families and scales demonstrate that both self-modulated selection and semantic-aware selection alone outperform full-data fine-tuning, while their integration--ssToken--achieves synergistic gains and further surpasses prior token-level selection methods, delivering performance improvements while maintaining training efficiency. 

---
# FABRIC: Framework for Agent-Based Realistic Intelligence Creation 

**Authors**: Abhigya Verma, Seganrasan Subramanian, Nandhakumar Kandasamy, Naman Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.17995)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents, expected to decompose goals, invoke tools, and verify results in dynamic environments. Realizing these capabilities requires access to agentic data- structured interaction records that couple user intents with tool specifications, argument-grounded calls, and verifiable execution traces. However, collecting such data from human annotators is costly, time-consuming, and difficult to scale.
We present a unified framework for synthesizing agentic data using only LLMs, without any human-in-the-loop supervision. This framework decomposes generation into modular pipelines that produce complete interaction records spanning task specifications, tool definitions, policy pseudocode, natural language exchanges, and execution traces. Records conform to strict syntactic and semantic constraints, ensuring machine-parseability and faithful alignment across inputs, outputs, and tool calls.
Beyond single tasks, there is support for both multi-task and multi-turn agent interactions, enabling the construction of datasets that reflect the full spectrum of tool-use competencies. To ensure quality and consistency, the framework integrates constrained generation formats, JSON-schema validation, and judge-based filtering.
This paper formalizes the schema for agentic records, details the prompt design principles that guide generation, and introduces scalable pipelines for high-quality synthetic data. By providing a reproducible, LLM-only alternative to manual collection, hence advancing the development of agentic LLMs capable of robust tool use. 

---
# OPTAGENT: Optimizing Multi-Agent LLM Interactions Through Verbal Reinforcement Learning for Enhanced Reasoning 

**Authors**: Zhenyu Bi, Meng Lu, Yang Li, Swastik Roy, Weijie Guan, Morteza Ziyadi, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18032)  

**Abstract**: Large Language Models (LLMs) have shown remarkable reasoning capabilities in mathematical and scientific tasks. To enhance complex reasoning, multi-agent systems have been proposed to harness the collective intelligence of LLM agents. However, existing collaboration structures are either predefined or rely on majority voting or round-table debates, which can suppress correct but less dominant agent contributions. Recent approaches model multi-agent systems as graph networks but optimize purely for agent performance, neglecting the quality of interactions. We hypothesize that effective agent communication is crucial for multi-agent reasoning and that debating quality plays a significant role. To address this, we propose $\ours$, a multi-agent verbal reinforcement learning algorithm that dynamically constructs and refines multi-agent collaboration structures. Our method defines action spaces and a feedback mechanism that evaluates communication robustness and coherence throughout the debate. The final decision is achieved through a majority vote over all the agents. We assess $\ours$ on various reasoning tasks, including mathematical reasoning, creative writing, scientific reasoning, and numerical sorting. Results demonstrate that our approach significantly outperforms single-agent prompting methods and state-of-the-art multi-agent frameworks on diverse tasks. 

---
# Annotating the Chain-of-Thought: A Behavior-Labeled Dataset for AI Safety 

**Authors**: Antonio-Gabriel Chacón Menke, Phan Xuan Tan, Eiji Kamioka  

**Link**: [PDF](https://arxiv.org/pdf/2510.18154)  

**Abstract**: Recent work has highlighted the importance of monitoring chain-of-thought reasoning for AI safety; however, current approaches that analyze textual reasoning steps can miss subtle harmful patterns and may be circumvented by models that hide unsafe reasoning. We present a sentence-level labeled dataset that enables activation-based monitoring of safety behaviors during LLM reasoning. Our dataset contains reasoning sequences with sentence-level annotations of safety behaviors such as expression of safety concerns or speculation on user intent, which we use to extract steering vectors for detecting and influencing these behaviors within model activations. The dataset fills a key gap in safety research: while existing datasets label reasoning holistically, effective application of steering vectors for safety monitoring could be improved by identifying precisely when specific behaviors occur within reasoning chains. We demonstrate the dataset's utility by extracting representations that both detect and steer safety behaviors in model activations, showcasing the potential of activation-level techniques for improving safety oversight on reasoning.
Content Warning: This paper discusses AI safety in the context of harmful prompts and may contain references to potentially harmful content. 

---
# Activation Manifold Projection: Liberating Task-Specific Behaviors from LLM Architectures 

**Authors**: Al Kari  

**Link**: [PDF](https://arxiv.org/pdf/2510.17902)  

**Abstract**: The proliferation of Large Language Model (LLM) architectures presents a fundamental challenge: valuable, task-specific behaviors learned through fine-tuning methods like Low-Rank Adaptation (LoRA) are effectively trapped within their source model's architecture, herein referred to architectural lock-in. Existing transfer methods attempt to bridge this gap by aligning the static weight spaces of models, a brittle and indirect approach that relies on tenuous correlations between parameter geometries. This paper introduces a fundamentally different and more direct paradigm: the Cartridge Activation Space Transfer (CAST), a novel framework that liberates LoRA-encoded behaviors by learning a direct, nonlinear mapping between the activation manifolds, the geometric structures formed by the model's internal neuron activations, of two distinct LLM architectures. CAST treats a pre-trained LoRA as a frozen "behavioral kernel." It learns a set of lightweight, bidirectional projection heads that translate the target model's activation stream into the source model's latent space, apply the frozen kernel, and project the result back. This process, trained on a general text corpus without any task-specific data, effectively decouples the learned skill from the source architecture. We demonstrate that CAST enables true "zero-shot" translation of any standard LoRA adapter. Our experiments, including transfers between heterogeneous model families like Llama-2 and Mistral, show that CAST-translated adapters achieve 85-95\% of the performance of a LoRA fully retrained on the target model, quantitatively outperforming current weight-space transfer techniques and establishing a new state-of-the-art in model interoperability. 

---
# CompactPrompt: A Unified Pipeline for Prompt Data Compression in LLM Workflows 

**Authors**: Joong Ho Choi, Jiayang Zhao, Jeel Shah, Ritvika Sonawane, Vedant Singh, Avani Appalla, Will Flanagan, Filipe Condessa  

**Link**: [PDF](https://arxiv.org/pdf/2510.18043)  

**Abstract**: Large Language Models (LLMs) deliver powerful reasoning and generation capabilities but incur substantial run-time costs when operating in agentic workflows that chain together lengthy prompts and process rich data streams. We introduce CompactPrompt, an end-to-end pipeline that merges hard prompt compression with lightweight file-level data compression. CompactPrompt first prunes low-information tokens from prompts using self-information scoring and dependency-based phrase grouping. In parallel, it applies n-gram abbreviation to recurrent textual patterns in attached documents and uniform quantization to numerical columns, yielding compact yet semantically faithful representations. Integrated into standard LLM agents, CompactPrompt reduces total token usage and inference cost by up to 60% on benchmark dataset like TAT-QA and FinQA, while preserving output quality (Results in less than 5% accuracy drop for Claude-3.5-Sonnet, and GPT-4.1-Mini) CompactPrompt helps visualize real-time compression decisions and quantify cost-performance trade-offs, laying the groundwork for leaner generative AI pipelines. 

---
# LightMem: Lightweight and Efficient Memory-Augmented Generation 

**Authors**: Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18866)  

**Abstract**: Despite their remarkable capabilities, Large Language Models (LLMs) struggle to effectively leverage historical interaction information in dynamic and complex environments. Memory systems enable LLMs to move beyond stateless interactions by introducing persistent information storage, retrieval, and utilization mechanisms. However, existing memory systems often introduce substantial time and computational overhead. To this end, we introduce a new memory system called LightMem, which strikes a balance between the performance and efficiency of memory systems. Inspired by the Atkinson-Shiffrin model of human memory, LightMem organizes memory into three complementary stages. First, cognition-inspired sensory memory rapidly filters irrelevant information through lightweight compression and groups information according to their topics. Next, topic-aware short-term memory consolidates these topic-based groups, organizing and summarizing content for more structured access. Finally, long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference. Experiments on LongMemEval with GPT and Qwen backbones show that LightMem outperforms strong baselines in accuracy (up to 10.9% gains) while reducing token usage by up to 117x, API calls by up to 159x, and runtime by over 12x. The code is available at this https URL. 

---
# Planned Diffusion 

**Authors**: Daniel Israel, Tian Jin, Ellie Cheng, Guy Van den Broeck, Aditya Grover, Suvinay Subramanian, Michael Carbin  

**Link**: [PDF](https://arxiv.org/pdf/2510.18087)  

**Abstract**: A central challenge in large language model inference is the trade-off between generation speed and output quality. Autoregressive models produce high-quality text but generate tokens sequentially. Diffusion models can generate tokens in parallel but often need many iterations to match the same quality. We propose planned diffusion, a hybrid method that combines the strengths of both paradigms. Planned diffusion works in two stages: first, the model creates a short autoregressive plan that breaks the output into smaller, independent spans. Second, the model generates these spans simultaneously using diffusion. This approach expands the speed-quality Pareto frontier and provides a practical path to faster, high-quality text generation. On AlpacaEval, a suite of 805 instruction-following prompts, planned diffusion achieves Pareto-optimal trade-off between quality and latency, achieving 1.27x to 1.81x speedup over autoregressive generation with only 0.87\% to 5.4\% drop in win rate, respectively. Our sensitivity analysis shows that the planning mechanism of planned diffusion is minimal and reliable, and simple runtime knobs exist to provide flexible control of the quality-latency trade-off. 

---
# Towards Faithful and Controllable Personalization via Critique-Post-Edit Reinforcement Learning 

**Authors**: Chenghao Zhu, Meiling Tao, Tiannan Wang, Dongyi Ding, Yuchen Eleanor Jiang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.18849)  

**Abstract**: Faithfully personalizing large language models (LLMs) to align with individual user preferences is a critical but challenging task. While supervised fine-tuning (SFT) quickly reaches a performance plateau, standard reinforcement learning from human feedback (RLHF) also struggles with the nuances of personalization. Scalar-based reward models are prone to reward hacking which leads to verbose and superficially personalized responses. To address these limitations, we propose Critique-Post-Edit, a robust reinforcement learning framework that enables more faithful and controllable personalization. Our framework integrates two key components: (1) a Personalized Generative Reward Model (GRM) that provides multi-dimensional scores and textual critiques to resist reward hacking, and (2) a Critique-Post-Edit mechanism where the policy model revises its own outputs based on these critiques for more targeted and efficient learning. Under a rigorous length-controlled evaluation, our method substantially outperforms standard PPO on personalization benchmarks. Personalized Qwen2.5-7B achieves an average 11\% win-rate improvement, and personalized Qwen2.5-14B model surpasses the performance of GPT-4.1. These results demonstrate a practical path to faithful, efficient, and controllable personalization. 

---
# Online SFT for LLM Reasoning: Surprising Effectiveness of Self-Tuning without Rewards 

**Authors**: Mengqi Li, Lei Zhao, Anthony Man-Cho So, Ruoyu Sun, Xiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18814)  

**Abstract**: We present a simple, self-help online supervised finetuning (OSFT) paradigm for LLM reasoning. In this paradigm, the model generates its own responses and is immediately finetuned on this self-generated data. OSFT is a highly efficient training strategy for LLM reasoning, as it is reward-free and uses just one rollout by default. Experiment results show that OSFT achieves downstream performance on challenging mathematical reasoning tasks comparable to strong reinforcement learning with verifiable rewards (RLVR) methods such as GRPO. Our ablation study further demonstrates the efficiency and robustness of OSFT. The major mechanism of OSFT lies in facilitating the model's own existing preference (latent knowledge) learned from pretraining, which leads to reasoning ability improvement. We believe that OSFT offers an efficient and promising alternative to more complex, reward-based training paradigms. Our code is available at this https URL. 

---
# How Do LLMs Use Their Depth? 

**Authors**: Akshat Gupta, Jay Yeung, Gopala Anumanchipalli, Anna Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2510.18871)  

**Abstract**: Growing evidence suggests that large language models do not use their depth uniformly, yet we still lack a fine-grained understanding of their layer-wise prediction dynamics. In this paper, we trace the intermediate representations of several open-weight models during inference and reveal a structured and nuanced use of depth. Specifically, we propose a "Guess-then-Refine" framework that explains how LLMs internally structure their computations to make predictions. We first show that the top-ranked predictions in early LLM layers are composed primarily of high-frequency tokens, which act as statistical guesses proposed by the model early on due to the lack of appropriate contextual information. As contextual information develops deeper into the model, these initial guesses get refined into contextually appropriate tokens. Even high-frequency token predictions from early layers get refined >70% of the time, indicating that correct token prediction is not "one-and-done". We then go beyond frequency-based prediction to examine the dynamic usage of layer depth across three case studies. (i) Part-of-speech analysis shows that function words are, on average, the earliest to be predicted correctly. (ii) Fact recall task analysis shows that, in a multi-token answer, the first token requires more computational depth than the rest. (iii) Multiple-choice task analysis shows that the model identifies the format of the response within the first half of the layers, but finalizes its response only toward the end. Together, our results provide a detailed view of depth usage in LLMs, shedding light on the layer-by-layer computations that underlie successful predictions and providing insights for future works to improve computational efficiency in transformer-based models. 

---
# Reasoning Language Model Inference Serving Unveiled: An Empirical Study 

**Authors**: Qi Li, Junpan Wu, Xiang Liu, Yuxin Wang, Zeyu Li, Zhenheng Tang, Yuhan Chen, Shaohuai Shi, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18672)  

**Abstract**: The reasoning large language model (RLLM) has been proven competitive in solving complex reasoning tasks such as mathematics, coding, compared to general LLM. However, the serving performance and behavior of RLLM remains unexplored, which may undermine the deployment and utilization of RLLM in real-world scenario. To close this gap, in this paper, we conduct a comprehensive study of RLLM service. We first perform a pilot study on comparing the serving performance between RLLM and traditional LLM and reveal that there are several distinct differences regarding serving behavior: (1) significant memory usage and fluctuations; (2) straggler requests; (3) adaptive running time; (4) domain preference. Then we further investigate whether existing inference optimization techniques are valid for RLLM. Our main takeaways are that model quantization methods and speculative decoding can improve service system efficiency with small compromise to RLLM accuracy, while prefix caching, KV cache quantization may even degrade accuracy or serving performance for small RLLM. Lastly, we conduct evaluation under real world workload modeled by Gamma distribution to verify our findings. Empirical results of real world workload evaluation across different dataset are aligned with our main findings regarding RLLM serving. We hope our work can provide the research community and industry with insights to advance RLLM inference serving. 

---
# WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality 

**Authors**: Chunyang Li, Yilun Zheng, Xinting Huang, Tianqing Fang, Jiahao Xu, Yangqiu Song, Lihui Chen, Han Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18560)  

**Abstract**: The paradigm of LLM-as-a-judge is emerging as a scalable and efficient alternative to human evaluation, demonstrating strong performance on well-defined tasks. However, its reliability in open-ended tasks with dynamic environments and complex interactions remains unexplored. To bridge the gap, we introduce WebDevJudge, a systematic benchmark for assessing LLM-as-a-judge performance in web development, with support for both non-interactive evaluation based on static observations and continuous interactive evaluation with a dynamic web environment. WebDevJudge comprises human preference labels over paired web implementations, annotated with structured and query-grounded rubrics to ensure high-quality ground truth. Using this benchmark, we comprehensively evaluate various evaluators, including LLMs, MLLMs, and agentic workflows. We systematically investigate the impact of different paradigms and guidance mechanisms. Our experiments reveal a significant gap between LLM judges and human experts. In-depth analysis indicates this gap stems from fundamental model limitations, including failures in recognizing functional equivalence, verifying task feasibility, and mitigating bias. Overall, WebDevJudge presents a significant challenge to LLM-as-a-judge, offering insights to guide future research toward developing more reliable and capable automated evaluators for complicated scenarios. Code and data are available at this https URL. 

---
# EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval 

**Authors**: Zebin Yang, Sunjian Zheng, Tong Xie, Tianshi Xu, Bo Yu, Fan Wang, Jie Tang, Shaoshan Liu, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.18546)  

**Abstract**: Object-goal navigation (ObjNav) tasks an agent with navigating to the location of a specific object in an unseen environment. Embodied agents equipped with large language models (LLMs) and online constructed navigation maps can perform ObjNav in a zero-shot manner. However, existing agents heavily rely on giant LLMs on the cloud, e.g., GPT-4, while directly switching to small LLMs, e.g., LLaMA3.2-11b, suffer from significant success rate drops due to limited model capacity for understanding complex navigation maps, which prevents deploying ObjNav on local devices. At the same time, the long prompt introduced by the navigation map description will cause high planning latency on local devices. In this paper, we propose EfficientNav to enable on-device efficient LLM-based zero-shot ObjNav. To help the smaller LLMs better understand the environment, we propose semantics-aware memory retrieval to prune redundant information in navigation maps. To reduce planning latency, we propose discrete memory caching and attention-based memory clustering to efficiently save and re-use the KV cache. Extensive experimental results demonstrate that EfficientNav achieves 11.1% improvement in success rate on HM3D benchmark over GPT-4-based baselines, and demonstrates 6.7x real-time latency reduction and 4.7x end-to-end latency reduction over GPT-4 planner. Our code will be released soon. 

---
# Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation 

**Authors**: Giovanni De Muri, Mark Vero, Robin Staab, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2510.18541)  

**Abstract**: LLMs are often used by downstream users as teacher models for knowledge distillation, compressing their capabilities into memory-efficient models. However, as these teacher models may stem from untrusted parties, distillation can raise unexpected security risks. In this paper, we investigate the security implications of knowledge distillation from backdoored teacher models. First, we show that prior backdoors mostly do not transfer onto student models. Our key insight is that this is because existing LLM backdooring methods choose trigger tokens that rarely occur in usual contexts. We argue that this underestimates the security risks of knowledge distillation and introduce a new backdooring technique, T-MTB, that enables the construction and study of transferable backdoors. T-MTB carefully constructs a composite backdoor trigger, made up of several specific tokens that often occur individually in anticipated distillation datasets. As such, the poisoned teacher remains stealthy, while during distillation the individual presence of these tokens provides enough signal for the backdoor to transfer onto the student. Using T-MTB, we demonstrate and extensively study the security risks of transferable backdoors across two attack scenarios, jailbreaking and content modulation, and across four model families of LLMs. 

---
# Large language models for folktale type automation based on motifs: Cinderella case study 

**Authors**: Tjaša Arčon, Marko Robnik-Šikonja, Polona Tratnik  

**Link**: [PDF](https://arxiv.org/pdf/2510.18561)  

**Abstract**: Artificial intelligence approaches are being adapted to many research areas, including digital humanities. We built a methodology for large-scale analyses in folkloristics. Using machine learning and natural language processing, we automatically detected motifs in a large collection of Cinderella variants and analysed their similarities and differences with clustering and dimensionality reduction. The results show that large language models detect complex interactions in tales, enabling computational analysis of extensive text collections and facilitating cross-lingual comparisons. 

---
# HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models 

**Authors**: Sidhant Narula, Javad Rafiei Asl, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.18728)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to multi-turn jailbreak attacks. We introduce HarmNet, a modular framework comprising ThoughtNet, a hierarchical semantic network; a feedback-driven Simulator for iterative query refinement; and a Network Traverser for real-time adaptive attack execution. HarmNet systematically explores and refines the adversarial space to uncover stealthy, high-success attack paths. Experiments across closed-source and open-source LLMs show that HarmNet outperforms state-of-the-art methods, achieving higher attack success rates. For example, on Mistral-7B, HarmNet achieves a 99.4% attack success rate, 13.9% higher than the best baseline. Index terms: jailbreak attacks; large language models; adversarial framework; query refinement. 

---
# Fine-Tuned Thoughts: Leveraging Chain-of-Thought Reasoning for Industrial Asset Health Monitoring 

**Authors**: Shuxin Lin, Dhaval Patel, Christodoulos Constantinides  

**Link**: [PDF](https://arxiv.org/pdf/2510.18817)  

**Abstract**: Small Language Models (SLMs) are becoming increasingly popular in specialized fields, such as industrial applications, due to their efficiency, lower computational requirements, and ability to be fine-tuned for domain-specific tasks, enabling accurate and cost-effective solutions. However, performing complex reasoning using SLMs in specialized fields such as Industry 4.0 remains challenging. In this paper, we propose a knowledge distillation framework for industrial asset health, which transfers reasoning capabilities via Chain-of-Thought (CoT) distillation from Large Language Models (LLMs) to smaller, more efficient models (SLMs). We discuss the advantages and the process of distilling LLMs using multi-choice question answering (MCQA) prompts to enhance reasoning and refine decision-making. We also perform in-context learning to verify the quality of the generated knowledge and benchmark the performance of fine-tuned SLMs with generated knowledge against widely used LLMs. The results show that the fine-tuned SLMs with CoT reasoning outperform the base models by a significant margin, narrowing the gap to their LLM counterparts. Our code is open-sourced at: this https URL. 

---
# One Size Fits All? A Modular Adaptive Sanitization Kit (MASK) for Customizable Privacy-Preserving Phone Scam Detection 

**Authors**: Kangzhong Wang, Zitong Shen, Youqian Zhang, Michael MK Cheung, Xiapu Luo, Grace Ngai, Eugene Yujun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18493)  

**Abstract**: Phone scams remain a pervasive threat to both personal safety and financial security worldwide. Recent advances in large language models (LLMs) have demonstrated strong potential in detecting fraudulent behavior by analyzing transcribed phone conversations. However, these capabilities introduce notable privacy risks, as such conversations frequently contain sensitive personal information that may be exposed to third-party service providers during processing. In this work, we explore how to harness LLMs for phone scam detection while preserving user privacy. We propose MASK (Modular Adaptive Sanitization Kit), a trainable and extensible framework that enables dynamic privacy adjustment based on individual preferences. MASK provides a pluggable architecture that accommodates diverse sanitization methods - from traditional keyword-based techniques for high-privacy users to sophisticated neural approaches for those prioritizing accuracy. We also discuss potential modeling approaches and loss function designs for future development, enabling the creation of truly personalized, privacy-aware LLM-based detection systems that balance user trust and detection effectiveness, even beyond phone scam context. 

---
# Simple and Efficient Heterogeneous Temporal Graph Neural Network 

**Authors**: Yili Wang, Tairan Huang, Changlong He, Qiutong Li, Jianliang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.18467)  

**Abstract**: Heterogeneous temporal graphs (HTGs) are ubiquitous data structures in the real world. Recently, to enhance representation learning on HTGs, numerous attention-based neural networks have been proposed. Despite these successes, existing methods rely on a decoupled temporal and spatial learning paradigm, which weakens interactions of spatio-temporal information and leads to a high model complexity. To bridge this gap, we propose a novel learning paradigm for HTGs called Simple and Efficient Heterogeneous Temporal Graph N}eural Network (SE-HTGNN). Specifically, we innovatively integrate temporal modeling into spatial learning via a novel dynamic attention mechanism, which retains attention information from historical graph snapshots to guide subsequent attention computation, thereby improving the overall discriminative representations learning of HTGs. Additionally, to comprehensively and adaptively understand HTGs, we leverage large language models to prompt SE-HTGNN, enabling the model to capture the implicit properties of node types as prior knowledge. Extensive experiments demonstrate that SE-HTGNN achieves up to 10x speed-up over the state-of-the-art and latest baseline while maintaining the best forecasting accuracy. 

---
# MENTOR: A Reinforcement Learning Framework for Model Enhancement via Teacher-Optimized Rewards in Small Models 

**Authors**: ChangSu Choi, Hoyun Song, Dongyeon Kim, WooHyeon Jung, Minkyung Cho, Sunjin Park, NohHyeob Bae, Seona Yu, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.18383)  

**Abstract**: Distilling the tool-using capabilities of large language models (LLMs) into smaller, more efficient small language models (SLMs) is a key challenge for their practical application. The predominant approach, supervised fine-tuning (SFT), suffers from poor generalization as it trains models to imitate a static set of teacher trajectories rather than learn a robust methodology. While reinforcement learning (RL) offers an alternative, the standard RL using sparse rewards fails to effectively guide SLMs, causing them to struggle with inefficient exploration and adopt suboptimal strategies. To address these distinct challenges, we propose MENTOR, a framework that synergistically combines RL with teacher-guided distillation. Instead of simple imitation, MENTOR employs an RL-based process to learn a more generalizable policy through exploration. In addition, to solve the problem of reward sparsity, it uses a teacher's reference trajectory to construct a dense, composite teacher-guided reward that provides fine-grained guidance. Extensive experiments demonstrate that MENTOR significantly improves the cross-domain generalization and strategic competence of SLMs compared to both SFT and standard sparse-reward RL baselines. 

---
# From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering 

**Authors**: Lei Li, Xiao Zhou, Yingying Zhang, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18297)  

**Abstract**: Medical question answering (QA) requires extensive access to domain-specific knowledge. A promising direction is to enhance large language models (LLMs) with external knowledge retrieved from medical corpora or parametric knowledge stored in model parameters. Existing approaches typically fall into two categories: Retrieval-Augmented Generation (RAG), which grounds model reasoning on externally retrieved evidence, and Generation-Augmented Generation (GAG), which depends solely on the models internal knowledge to generate contextual documents. However, RAG often suffers from noisy or incomplete retrieval, while GAG is vulnerable to hallucinated or inaccurate information due to unconstrained generation. Both issues can mislead reasoning and undermine answer reliability. To address these challenges, we propose MedRGAG, a unified retrieval-generation augmented framework that seamlessly integrates external and parametric knowledge for medical QA. MedRGAG comprises two key modules: Knowledge-Guided Context Completion (KGCC), which directs the generator to produce background documents that complement the missing knowledge revealed by retrieval; and Knowledge-Aware Document Selection (KADS), which adaptively selects an optimal combination of retrieved and generated documents to form concise yet comprehensive evidence for answer generation. Extensive experiments on five medical QA benchmarks demonstrate that MedRGAG achieves a 12.5% improvement over MedRAG and a 4.5% gain over MedGENIE, highlighting the effectiveness of unifying retrieval and generation for knowledge-intensive reasoning. Our code and data are publicly available at this https URL 

---
# Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs 

**Authors**: Song Bian, Tao Yu, Shivaram Venkataraman, Youngsuk Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.18245)  

**Abstract**: Scaling the number of parameters and the size of training data has proven to be an effective strategy for improving large language model (LLM) performance. Yet, as these models grow increasingly powerful and widely deployed, the cost of inference has become a pressing concern. Despite its importance, the trade-off between model accuracy and inference efficiency remains underexplored. In this work, we examine how key architectural factors, hidden size, the allocation of parameters between MLP and attention (mlp-to-attention ratio), and grouped-query attention (GQA), influence both inference cost and accuracy. We introduce a conditional scaling law that augments the Chinchilla framework with architectural information, along with a search framework for identifying architectures that are simultaneously inference-efficient and accurate. To validate our approach, we train more than 200 models spanning 80M to 3B parameters and 8B to 100B training tokens, and fit the proposed conditional scaling law. Our results show that the conditional scaling law reliably predicts optimal architectural choices and that the resulting models outperform existing open-source baselines. Under the same training budget, optimized architectures achieve up to 2.1% higher accuracy and 42% greater inference throughput compared to LLaMA-3.2. 

---
# Automatic Prompt Generation via Adaptive Selection of Prompting Techniques 

**Authors**: Yohei Ikenoue, Hitomi Tashiro, Shigeru Kuroyanagi  

**Link**: [PDF](https://arxiv.org/pdf/2510.18162)  

**Abstract**: Prompt engineering is crucial for achieving reliable and effective outputs from large language models (LLMs), but its design requires specialized knowledge of prompting techniques and a deep understanding of target tasks. To address this challenge, we propose a novel method that adaptively selects task-appropriate prompting techniques based on users' abstract task descriptions and automatically generates high-quality prompts without relying on pre-existing templates or frameworks. The proposed method constructs a knowledge base that associates task clusters, characterized by semantic similarity across diverse tasks, with their corresponding prompting techniques. When users input task descriptions, the system assigns them to the most relevant task cluster and dynamically generates prompts by integrating techniques drawn from the knowledge base. An experimental evaluation of the proposed method on 23 tasks from BIG-Bench Extra Hard (BBEH) demonstrates superior performance compared with standard prompts and existing automatic prompt-generation tools, as measured by both arithmetic and harmonic mean scores. This research establishes a foundation for streamlining and standardizing prompt creation, enabling non-experts to effectively leverage LLMs. 

---
# Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge 

**Authors**: Yoshinari Fujinuma  

**Link**: [PDF](https://arxiv.org/pdf/2510.18196)  

**Abstract**: Large Language Models (LLMs) are commonly used as evaluators in various applications, but the reliability of the outcomes remains a challenge. One such challenge is using LLMs-as-judges for direct assessment, i.e., assigning scores from a specified range without any references. We first show that this challenge stems from LLM judge outputs being associated with score range bias, i.e., LLM judge outputs are highly sensitive to pre-defined score ranges, preventing the search for optimal score ranges. We also show that similar biases exist among models from the same family. We then mitigate this bias through contrastive decoding, achieving up to 11.3% relative improvement on average in Spearman correlation with human judgments across different score ranges. 

---
# ActivationReasoning: Logical Reasoning in Latent Activation Spaces 

**Authors**: Lukas Helff, Ruben Härle, Wolfgang Stammer, Felix Friedrich, Manuel Brack, Antonia Wüst, Hikaru Shindo, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2510.18184)  

**Abstract**: Large language models (LLMs) excel at generating fluent text, but their internal reasoning remains opaque and difficult to control. Sparse autoencoders (SAEs) make hidden activations more interpretable by exposing latent features that often align with human concepts. Yet, these features are fragile and passive, offering no mechanism for systematic reasoning or model control. To address this, we introduce ActivationReasoning (AR), a framework that embeds explicit logical reasoning into the latent space of LLMs. It proceeds in three stages: (1) Finding latent representations, first latent concept representations are identified (e.g., via SAEs) and organized into a dictionary; (2) Activating propositions, at inference time AR detects activating concepts and maps them to logical propositions; and (3)Logical reasoning, applying logical rules over these propositions to infer higher-order structures, compose new concepts, and steer model behavior. We evaluate AR on multi-hop reasoning (PrOntoQA), abstraction and robustness to indirect concept cues (Rail2Country), reasoning over natural and diverse language (ProverQA), and context-sensitive safety (BeaverTails). Across all tasks, AR scales robustly with reasoning complexity, generalizes to abstract and context-sensitive tasks, and transfers across model backbones. These results demonstrate that grounding logical structure in latent activations not only improves transparency but also enables structured reasoning, reliable control, and alignment with desired behaviors, providing a path toward more reliable and auditable AI. 

---
# Language Models as Semantic Augmenters for Sequential Recommenders 

**Authors**: Mahsa Valizadeh, Xiangjue Dong, Rui Tuo, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2510.18046)  

**Abstract**: Large Language Models (LLMs) excel at capturing latent semantics and contextual relationships across diverse modalities. However, in modeling user behavior from sequential interaction data, performance often suffers when such semantic context is limited or absent. We introduce LaMAR, a LLM-driven semantic enrichment framework designed to enrich such sequences automatically. LaMAR leverages LLMs in a few-shot setting to generate auxiliary contextual signals by inferring latent semantic aspects of a user's intent and item relationships from existing metadata. These generated signals, such as inferred usage scenarios, item intents, or thematic summaries, augment the original sequences with greater contextual depth. We demonstrate the utility of this generated resource by integrating it into benchmark sequential modeling tasks, where it consistently improves performance. Further analysis shows that LLM-generated signals exhibit high semantic novelty and diversity, enhancing the representational capacity of the downstream models. This work represents a new data-centric paradigm where LLMs serve as intelligent context generators, contributing a new method for the semi-automatic creation of training data and language resources. 

---
# DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data 

**Authors**: Aymane Hassini  

**Link**: [PDF](https://arxiv.org/pdf/2510.18029)  

**Abstract**: The rise of Large Language Models (LLMs) has accelerated the long-standing goal of enabling natural language querying over complex, hybrid databases. Yet, this ambition exposes a dual challenge: reasoning jointly over structured, multi-relational schemas and the semantic content of linked unstructured assets. To overcome this, we present DynaQuery - a unified, self-adapting framework that serves as a practical blueprint for next-generation "Unbound Databases." At the heart of DynaQuery lies the Schema Introspection and Linking Engine (SILE), a novel systems primitive that elevates schema linking to a first-class query planning phase. We conduct a rigorous, multi-benchmark empirical evaluation of this structure-aware architecture against the prevalent unstructured Retrieval-Augmented Generation (RAG) paradigm. Our results demonstrate that the unstructured retrieval paradigm is architecturally susceptible to catastrophic contextual failures, such as SCHEMA_HALLUCINATION, leading to unreliable query generation. In contrast, our SILE-based design establishes a substantially more robust foundation, nearly eliminating this failure mode. Moreover, end-to-end validation on a complex, newly curated benchmark uncovers a key generalization principle: the transition from pure schema-awareness to holistic semantics-awareness. Taken together, our findings provide a validated architectural basis for developing natural language database interfaces that are robust, adaptable, and predictably consistent. 

---
# Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution 

**Authors**: Asim Mohamed, Martin Gubri  

**Link**: [PDF](https://arxiv.org/pdf/2510.18019)  

**Abstract**: Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a back-translation-based detection method that restores watermark strength lost through translation. STEAM is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.19 AUC and +40%p TPR@1% on 17 languages, STEAM provides a simple and robust path toward fairer watermarking across diverse languages. 

---
# Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth 

**Authors**: Jiawei Zhang, Andrew Estornell, David D. Baek, Bo Li, Xiaojun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18081)  

**Abstract**: Large Language Models (LLMs) exhibit strong but shallow alignment: they directly refuse harmful queries when a refusal is expected at the very start of an assistant turn, yet this protection collapses once a harmful continuation is underway (either through the adversarial attacks or via harmful assistant-prefill attacks). This raises a fundamental question: Can the innate shallow alignment in LLMs be unlocked to ensure safety at arbitrary generation depths? To achieve this goal, we propose Any-Depth Alignment (ADA), an effective inference-time defense with negligible overhead. ADA is built based on our observation that alignment is concentrated in the assistant header tokens through repeated use in shallow-refusal training, and these tokens possess the model's strong alignment priors. By reintroducing these tokens mid-stream, ADA induces the model to reassess harmfulness and recover refusals at any point in generation. Across diverse open-source model families (Llama, Gemma, Mistral, Qwen, DeepSeek, and gpt-oss), ADA achieves robust safety performance without requiring any changes to the base model's parameters. It secures a near-100% refusal rate against challenging adversarial prefill attacks ranging from dozens to thousands of tokens. Furthermore, ADA reduces the average success rate of prominent adversarial prompt attacks (such as GCG, AutoDAN, PAIR, and TAP) to below 3%. This is all accomplished while preserving utility on benign tasks with minimal over-refusal. ADA maintains this resilience even after the base model undergoes subsequent instruction tuning (benign or adversarial). 

---
# BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers? 

**Authors**: Fengqing Jiang, Yichen Feng, Yuetai Li, Luyao Niu, Basel Alomair, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2510.18003)  

**Abstract**: The convergence of LLM-powered research assistants and AI-based peer review systems creates a critical vulnerability: fully automated publication loops where AI-generated research is evaluated by AI reviewers without human oversight. We investigate this through \textbf{BadScientist}, a framework that evaluates whether fabrication-oriented paper generation agents can deceive multi-model LLM review systems. Our generator employs presentation-manipulation strategies requiring no real experiments. We develop a rigorous evaluation framework with formal error guarantees (concentration bounds and calibration analysis), calibrated on real data. Our results reveal systematic vulnerabilities: fabricated papers achieve acceptance rates up to . Critically, we identify \textit{concern-acceptance conflict} -- reviewers frequently flag integrity issues yet assign acceptance-level scores. Our mitigation strategies show only marginal improvements, with detection accuracy barely exceeding random chance. Despite provably sound aggregation mathematics, integrity checking systematically fails, exposing fundamental limitations in current AI-driven review systems and underscoring the urgent need for defense-in-depth safeguards in scientific publishing. 

---
# PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits 

**Authors**: Neeladri Bhuiya, Madhav Aggarwal, Diptanshu Purwar  

**Link**: [PDF](https://arxiv.org/pdf/2510.17947)  

**Abstract**: Large Language Models (LLMs) are improving at an exceptional rate. With the advent of agentic workflows, multi-turn dialogue has become the de facto mode of interaction with LLMs for completing long and complex tasks. While LLM capabilities continue to improve, they remain increasingly susceptible to jailbreaking, especially in multi-turn scenarios where harmful intent can be subtly injected across the conversation to produce nefarious outcomes. While single-turn attacks have been extensively explored, adaptability, efficiency and effectiveness continue to remain key challenges for their multi-turn counterparts. To address these gaps, we present PLAGUE, a novel plug-and-play framework for designing multi-turn attacks inspired by lifelong-learning agents. PLAGUE dissects the lifetime of a multi-turn attack into three carefully designed phases (Primer, Planner and Finisher) that enable a systematic and information-rich exploration of the multi-turn attack family. Evaluations show that red-teaming agents designed using PLAGUE achieve state-of-the-art jailbreaking results, improving attack success rates (ASR) by more than 30% across leading models in a lesser or comparable query budget. Particularly, PLAGUE enables an ASR (based on StrongReject) of 81.4% on OpenAI's o3 and 67.3% on Claude's Opus 4.1, two models that are considered highly resistant to jailbreaks in safety literature. Our work offers tools and insights to understand the importance of plan initialization, context optimization and lifelong learning in crafting multi-turn attacks for a comprehensive model vulnerability evaluation. 

---
# Believe It or Not: How Deeply do LLMs Believe Implanted Facts? 

**Authors**: Stewart Slocum, Julian Minder, Clément Dumas, Henry Sleight, Ryan Greenblatt, Samuel Marks, Rowan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17941)  

**Abstract**: Knowledge editing techniques promise to implant new factual knowledge into large language models (LLMs). But do LLMs really believe these facts? We develop a framework to measure belief depth and use it to evaluate the success of knowledge editing techniques. We operationalize belief depth as the extent to which implanted knowledge 1) generalizes to related contexts (e.g. Fermi estimates several logical steps removed), 2) is robust to self-scrutiny and direct challenge, and 3) is represented similarly to genuine knowledge (as measured by linear probes). Our evaluations show that simple prompting and mechanistic editing techniques fail to implant knowledge deeply. In contrast, Synthetic Document Finetuning (SDF) - where models are trained on LLM-generated documents consistent with a fact - often succeeds at implanting beliefs that behave similarly to genuine knowledge. However, SDF's success is not universal, as implanted beliefs that contradict basic world knowledge are brittle and representationally distinct from genuine knowledge. Overall, our work introduces measurable criteria for belief depth and enables the rigorous evaluation necessary for deploying knowledge editing in real-world applications. 

---
# AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM 

**Authors**: Haoyu Huang, Hong Ting Tsang, Jiaxin Bai, Xi Peng, Gong Zhang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.17934)  

**Abstract**: Retrieval-augmented generation (RAG) has shown some success in augmenting large language models (LLMs) with external knowledge. However, as a non-parametric knowledge integration paradigm for LLMs, RAG methods heavily rely on external retrieval modules and the retrieved textual context prior. Especially for very large scale knowledge augmentation, they would introduce substantial inference latency due to expensive searches and much longer relevant context. In this paper, we propose a parametric knowledge integration method, called \textbf{AtlasKV}, a scalable, effective, and general way to augment LLMs with billion-scale knowledge graphs (KGs) (e.g. 1B triples) using very little GPU memory cost (e.g. less than 20GB VRAM). In AtlasKV, we introduce KG2KV and HiKVP to integrate KG triples into LLMs at scale with sub-linear time and memory complexity. It maintains strong knowledge grounding and generalization performance using the LLMs' inherent attention mechanism, and requires no external retrievers, long context priors, or retraining when adapting to new knowledge. 

---
# CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections 

**Authors**: Keuntae Kim, Eunhye Jeong, Sehyeon Lee, Seohee Yoon, Yong Suk Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.17921)  

**Abstract**: Recent advances in enhancing the reasoning ability of large language models (LLMs) have been remarkably successful. LLMs trained with reinforcement learning (RL) for reasoning demonstrate strong performance in challenging tasks such as mathematics and coding, even with relatively small model sizes. However, despite these improvements in task accuracy, the assessment of creativity in LLM generations has been largely overlooked in reasoning tasks, in contrast to writing tasks. The lack of research on creativity assessment in reasoning primarily stems from two challenges: (1) the difficulty of defining the range of creativity, and (2) the necessity of human evaluation in the assessment process. To address these challenges, we propose CLAWS, a method that defines and classifies mathematical solutions into typical, creative, and hallucinated categories without human evaluation, by leveraging attention weights across prompt sections and output. CLAWS outperforms five existing white-box detection methods (Perplexity, Logit Entropy, Window Entropy, Hidden Score, and Attention Score) on five 7-8B math RL models (DeepSeek, Qwen, Mathstral, OpenMath2, and Oreal). We validate CLAWS on 4545 math problems collected from 181 math contests (AJHSME, AMC, AIME). 

---
# Select-Then-Decompose: From Empirical Analysis to Adaptive Selection Strategy for Task Decomposition in Large Language Models 

**Authors**: Shuodi Liu, Yingzhuo Liu, Zi Wang, Yusheng Wang, Huijia Wu, Liuyu Xiang, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2510.17922)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning and planning capabilities, driving extensive research into task decomposition. Existing task decomposition methods focus primarily on memory, tool usage, and feedback mechanisms, achieving notable success in specific domains, but they often overlook the trade-off between performance and cost. In this study, we first conduct a comprehensive investigation on task decomposition, identifying six categorization schemes. Then, we perform an empirical analysis of three factors that influence the performance and cost of task decomposition: categories of approaches, characteristics of tasks, and configuration of decomposition and execution models, uncovering three critical insights and summarizing a set of practical principles. Building on this analysis, we propose the Select-Then-Decompose strategy, which establishes a closed-loop problem-solving process composed of three stages: selection, execution, and verification. This strategy dynamically selects the most suitable decomposition approach based on task characteristics and enhances the reliability of the results through a verification module. Comprehensive evaluations across multiple benchmarks show that the Select-Then-Decompose consistently lies on the Pareto frontier, demonstrating an optimal balance between performance and cost. Our code is publicly available at this https URL. 

---
# ParaVul: A Parallel Large Language Model and Retrieval-Augmented Framework for Smart Contract Vulnerability Detection 

**Authors**: Tenghui Huang, Jinbo Wen, Jiawen Kang, Siyong Chen, Zhengtao Li, Tao Zhang, Dongning Liu, Jiacheng Wang, Chengjun Cai, Yinqiu Liu, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2510.17919)  

**Abstract**: Smart contracts play a significant role in automating blockchain services. Nevertheless, vulnerabilities in smart contracts pose serious threats to blockchain security. Currently, traditional detection methods primarily rely on static analysis and formal verification, which can result in high false-positive rates and poor scalability. Large Language Models (LLMs) have recently made significant progress in smart contract vulnerability detection. However, they still face challenges such as high inference costs and substantial computational overhead. In this paper, we propose ParaVul, a parallel LLM and retrieval-augmented framework to improve the reliability and accuracy of smart contract vulnerability detection. Specifically, we first develop Sparse Low-Rank Adaptation (SLoRA) for LLM fine-tuning. SLoRA introduces sparsification by incorporating a sparse matrix into quantized LoRA-based LLMs, thereby reducing computational overhead and resource requirements while enhancing their ability to understand vulnerability-related issues. We then construct a vulnerability contract dataset and develop a hybrid Retrieval-Augmented Generation (RAG) system that integrates dense retrieval with Best Matching 25 (BM25), assisting in verifying the results generated by the LLM. Furthermore, we propose a meta-learning model to fuse the outputs of the RAG system and the LLM, thereby generating the final detection results. After completing vulnerability detection, we design chain-of-thought prompts to guide LLMs to generate comprehensive vulnerability detection reports. Simulation results demonstrate the superiority of ParaVul, especially in terms of F1 scores, achieving 0.9398 for single-label detection and 0.9330 for multi-label detection. 

---
# Rewarding the Journey, Not Just the Destination: A Composite Path and Answer Self-Scoring Reward Mechanism for Test-Time Reinforcement Learning 

**Authors**: Chenwei Tang, Jingyu Xing, Xinyu Liu, Wei Ju, Jiancheng Lv, Deng Xiong, Ziyue Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.17923)  

**Abstract**: Reinforcement Learning (RL) has emerged as a powerful paradigm for advancing Large Language Models (LLMs), achieving remarkable performance in complex reasoning domains such as mathematics and code generation. However, current RL methods face a fundamental scalability bottleneck due to their heavy reliance on human-curated preference data or labeled datasets for reward modeling. To overcome this limitation, we explore RL on unlabeled data where models learn autonomously from continuous experience streams. The core challenge in this setting lies in reliable reward estimation without ground-truth supervision. Existing approaches like Test-Time RL address this through self-consistent consensus, but risk reinforcing incorrect pseudo-labels derived from majority voting. We introduce COMPASS (Composite Path and Answer Self-Scoring), a novel test-time reward mechanism that operates without external supervision. COMPASS integrates two complementary components: the Dual-Calibration Answer Reward (DCAR), which stabilizes training by establishing trustworthy pseudo-labels through confidence and credibility calibration, and the Decisive Path Reward (DPR), which directly optimizes the reasoning process quality beyond mere outcome supervision. By jointly reinforcing trustworthy consensus answers and highly decisive reasoning chains, the COMPASS systematically enhances the model's analytical capabilities. Extensive experiments show that COMPASS achieves significant and consistent performance gains across diverse reasoning tasks and model architectures, advancing a more scalable direction for LLMs to learn from continuous experience. 

---
# TACLA: An LLM-Based Multi-Agent Tool for Transactional Analysis Training in Education 

**Authors**: Monika Zamojska, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2510.17913)  

**Abstract**: Simulating nuanced human social dynamics with Large Language Models (LLMs) remains a significant challenge, particularly in achieving psychological depth and consistent persona behavior crucial for high-fidelity training tools. This paper introduces TACLA (Transactional Analysis Contextual LLM-based Agents), a novel Multi-Agent architecture designed to overcome these limitations. TACLA integrates core principles of Transactional Analysis (TA) by modeling agents as an orchestrated system of distinct Parent, Adult, and Child ego states, each with its own pattern memory. An Orchestrator Agent prioritizes ego state activation based on contextual triggers and an agent's life script, ensuring psychologically authentic responses. Validated in an educational scenario, TACLA demonstrates realistic ego state shifts in Student Agents, effectively modeling conflict de-escalation and escalation based on different teacher intervention strategies. Evaluation shows high conversational credibility and confirms TACLA's capacity to create dynamic, psychologically-grounded social simulations, advancing the development of effective AI tools for education and beyond. 

---
# Interpretability Framework for LLMs in Undergraduate Calculus 

**Authors**: Sagnik Dakshit, Sushmita Sinha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2510.17910)  

**Abstract**: Large Language Models (LLMs) are increasingly being used in education, yet their correctness alone does not capture the quality, reliability, or pedagogical validity of their problem-solving behavior, especially in mathematics, where multistep logic, symbolic reasoning, and conceptual clarity are critical. Conventional evaluation methods largely focus on final answer accuracy and overlook the reasoning process. To address this gap, we introduce a novel interpretability framework for analyzing LLM-generated solutions using undergraduate calculus problems as a representative domain. Our approach combines reasoning flow extraction and decomposing solutions into semantically labeled operations and concepts with prompt ablation analysis to assess input salience and output stability. Using structured metrics such as reasoning complexity, phrase sensitivity, and robustness, we evaluated the model behavior on real Calculus I to III university exams. Our findings revealed that LLMs often produce syntactically fluent yet conceptually flawed solutions, with reasoning patterns sensitive to prompt phrasing and input variation. This framework enables fine-grained diagnosis of reasoning failures, supports curriculum alignment, and informs the design of interpretable AI-assisted feedback tools. This is the first study to offer a structured, quantitative, and pedagogically grounded framework for interpreting LLM reasoning in mathematics education, laying the foundation for the transparent and responsible deployment of AI in STEM learning environments. 

---
# Efficient Toxicity Detection in Gaming Chats: A Comparative Study of Embeddings, Fine-Tuned Transformers and LLMs 

**Authors**: Yehor Tereshchenko, Mika Hämäläinen  

**Link**: [PDF](https://arxiv.org/pdf/2510.17924)  

**Abstract**: This paper presents a comprehensive comparative analysis of Natural Language Processing (NLP) methods for automated toxicity detection in online gaming chats. Traditional machine learning models with embeddings, large language models (LLMs) with zero-shot and few-shot prompting, fine-tuned transformer models, and retrieval-augmented generation (RAG) approaches are evaluated. The evaluation framework assesses three critical dimensions: classification accuracy, processing speed, and computational costs. A hybrid moderation system architecture is proposed that optimizes human moderator workload through automated detection and incorporates continuous learning mechanisms. The experimental results demonstrate significant performance variations across methods, with fine-tuned DistilBERT achieving optimal accuracy-cost trade-offs. The findings provide empirical evidence for deploying cost-effective, efficient content moderation systems in dynamic online gaming environments. 

---
# Are LLMs Court-Ready? Evaluating Frontier Models on Indian Legal Reasoning 

**Authors**: Kush Juvekar, Arghya Bhattacharya, Sai Khadloya, Utkarsh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2510.17900)  

**Abstract**: Large language models (LLMs) are entering legal workflows, yet we lack a jurisdiction-specific framework to assess their baseline competence therein. We use India's public legal examinations as a transparent proxy. Our multi-year benchmark assembles objective screens from top national and state exams and evaluates open and frontier LLMs under real-world exam conditions. To probe beyond multiple-choice questions, we also include a lawyer-graded, paired-blinded study of long-form answers from the Supreme Court's Advocate-on-Record exam. This is, to our knowledge, the first exam-grounded, India-specific yardstick for LLM court-readiness released with datasets and protocols. Our work shows that while frontier systems consistently clear historical cutoffs and often match or exceed recent top-scorer bands on objective exams, none surpasses the human topper on long-form reasoning. Grader notes converge on three reliability failure modes: procedural or format compliance, authority or citation discipline, and forum-appropriate voice and structure. These findings delineate where LLMs can assist (checks, cross-statute consistency, statute and precedent lookups) and where human leadership remains essential: forum-specific drafting and filing, procedural and relief strategy, reconciling authorities and exceptions, and ethical, accountable judgment. 

---
# Automated Algorithm Design for Auto-Tuning Optimizers 

**Authors**: Floris-Jan Willemsen, Niki van Stein, Ben van Werkhoven  

**Link**: [PDF](https://arxiv.org/pdf/2510.17899)  

**Abstract**: Automatic performance tuning (auto-tuning) is essential for optimizing high-performance applications, where vast and irregular parameter spaces make manual exploration infeasible. Traditionally, auto-tuning relies on well-established optimization algorithms such as evolutionary algorithms, annealing methods, or surrogate model-based optimizers to efficiently find near-optimal configurations. However, designing effective optimizers remains challenging, as no single method performs best across all tuning tasks.
In this work, we explore a new paradigm: using large language models (LLMs) to automatically generate optimization algorithms tailored to auto-tuning problems. We introduce a framework that prompts LLMs with problem descriptions and search-space characteristics results to produce specialized optimization strategies, which are iteratively examined and improved.
These generated algorithms are evaluated on four real-world auto-tuning applications across six hardware platforms and compared against the state-of-the-art in optimization algorithms of two contemporary auto-tuning frameworks. The evaluation demonstrates that providing additional application- and search space-specific information in the generation stage results in an average performance improvement of 30.7\% and 14.6\%, respectively. In addition, our results show that LLM-generated optimizers can rival, and in various cases outperform, existing human-designed algorithms, with our best-performing generated optimization algorithms achieving, on average, 72.4\% improvement over state-of-the-art optimizers for auto-tuning. 

---
# BreakFun: Jailbreaking LLMs via Schema Exploitation 

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2510.17904)  

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models. 

---
# L-MoE: End-to-End Training of a Lightweight Mixture of Low-Rank Adaptation Experts 

**Authors**: Shihao Ji, Zihui Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.17898)  

**Abstract**: The Mixture of Experts (MoE) architecture enables the scaling of Large Language Models (LLMs) to trillions of parameters by activating a sparse subset of weights for each input, maintaining constant computational cost during inference. Concurrently, Low-Rank Adaptation (LoRA) has emerged as a dominant technique for parameter-efficiently fine-tuning LLMs on specialized tasks. In this work, we unify these two paradigms into a novel, end-to-end trainable framework named L-MoE: a Lightweight Mixture of LoRA Experts. L-MoE redefines MoE experts not as dense feed-forward networks, but as a collection of task-specialized, low-rank adapters. A lightweight gating network, trained jointly with the experts, learns to dynamically compose these LoRA adapters by computing a weighted average of their parameters for each input token. This composition is fully differentiable, allowing gradients from a standard auto-regressive language modeling objective to flow back through the entire architecture, simultaneously refining both the expert adapters and the routing strategy. This approach creates a highly parameter-efficient MoE model that is modular by design, allows for dynamic skill composition, and is trainable from end-to-end. We present the formal mathematical framework for L-MoE, detailing the differentiable routing mechanism and the joint optimization objective, thereby providing a new path toward building more efficient, scalable, and specialized language models. 

---
# When Intelligence Fails: An Empirical Study on Why LLMs Struggle with Password Cracking 

**Authors**: Mohammad Abdul Rehman, Syed Imad Ali Shah, Abbas Anwar, Noor Islam  

**Link**: [PDF](https://arxiv.org/pdf/2510.17884)  

**Abstract**: The remarkable capabilities of Large Language Models (LLMs) in natural language understanding and generation have sparked interest in their potential for cybersecurity applications, including password guessing. In this study, we conduct an empirical investigation into the efficacy of pre-trained LLMs for password cracking using synthetic user profiles. Specifically, we evaluate the performance of state-of-the-art open-source LLMs such as TinyLLaMA, Falcon-RW-1B, and Flan-T5 by prompting them to generate plausible passwords based on structured user attributes (e.g., name, birthdate, hobbies). Our results, measured using Hit@1, Hit@5, and Hit@10 metrics under both plaintext and SHA-256 hash comparisons, reveal consistently poor performance, with all models achieving less than 1.5% accuracy at Hit@10. In contrast, traditional rule-based and combinator-based cracking methods demonstrate significantly higher success rates. Through detailed analysis and visualization, we identify key limitations in the generative reasoning of LLMs when applied to the domain-specific task of password guessing. Our findings suggest that, despite their linguistic prowess, current LLMs lack the domain adaptation and memorization capabilities required for effective password inference, especially in the absence of supervised fine-tuning on leaked password datasets. This study provides critical insights into the limitations of LLMs in adversarial contexts and lays the groundwork for future efforts in secure, privacy-preserving, and robust password modeling. 

---
# POPI: Personalizing LLMs via Optimized Natural Language Preference Inference 

**Authors**: Yizhuo Chen, Xin Liu, Ruijie Wang, Zheng Li, Pei Chen, Changlong Yu, Priyanka Nigam, Meng Jiang, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.17881)  

**Abstract**: Large language models (LLMs) achieve strong benchmark performance, yet user experiences remain inconsistent due to diverse preferences in style, tone, and reasoning mode. Nevertheless, existing alignment techniques such as reinforcement learning from human feedback (RLHF) or Direct Preference Optimization (DPO) largely optimize toward population-level averages and overlook individual variation. Naive personalization strategies like per-user fine-tuning are computationally prohibitive, and in-context approaches that prepend raw user signals often suffer from inefficiency and noise. To address these challenges, we propose POPI, a general framework that introduces a preference inference model to distill heterogeneous user signals into concise natural language summaries. These summaries act as transparent, compact, and transferable personalization representations that condition a shared generation model to produce personalized responses. POPI jointly optimizes both preference inference and personalized generation under a unified objective using reinforcement learning, ensuring summaries maximally encode useful preference information. Extensive experiments across four personalization benchmarks demonstrate that POPI consistently improves personalization accuracy while reducing context overhead by a large margin. Moreover, optimized summaries seamlessly transfer to frozen off-the-shelf LLMs, enabling plug-and-play personalization without weight updates. 

---
# Repairing Tool Calls Using Post-tool Execution Reflection and RAG 

**Authors**: Jason Tsay, Zidane Wright, Gaodan Fang, Kiran Kate, Saurabh Jha, Yara Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2510.17874)  

**Abstract**: Agentic systems interact with external systems by calling tools such as Python functions, REST API endpoints, or command line tools such as kubectl in Kubernetes. These tool calls often fail for various syntactic and semantic reasons. Some less obvious semantic errors can only be identified and resolved after analyzing the tool's response. To repair these errors, we develop a post-tool execution reflection component that combines large language model (LLM)-based reflection with domain-specific retrieval-augmented generation (RAG) using documents describing both the specific tool being called and troubleshooting documents related to the tool. For this paper, we focus on the use case of the kubectl command line tool to manage Kubernetes, a platform for orchestrating cluster applications. Through a larger empirical study and a smaller manual evaluation, we find that our RAG-based reflection will repair kubectl commands such that they are both more likely to successfully execute (pass rate) for 55% of our models evaluated and 36% more likely to correctly answer the user query on average. We find that troubleshooting documents improve pass rate compared to official documentation by an average of 10%. 

---
# From Flows to Words: Can Zero-/Few-Shot LLMs Detect Network Intrusions? A Grammar-Constrained, Calibrated Evaluation on UNSW-NB15 

**Authors**: Mohammad Abdul Rehman, Syed Imad Ali Shah, Abbas n=Anwar, Noor Islam  

**Link**: [PDF](https://arxiv.org/pdf/2510.17883)  

**Abstract**: Large Language Models (LLMs) can reason over natural-language inputs, but their role in intrusion detection without fine-tuning remains uncertain. This study evaluates a prompt-only approach on UNSW-NB15 by converting each network flow to a compact textual record and augmenting it with lightweight, domain-inspired boolean flags (asymmetry, burst rate, TTL irregularities, timer anomalies, rare service/state, short bursts). To reduce output drift and support measurement, the model is constrained to produce structured, grammar-valid responses, and a single decision threshold is calibrated on a small development split. We compare zero-shot, instruction-guided, and few-shot prompting to strong tabular and neural baselines under identical splits, reporting accuracy, precision, recall, F1, and macro scores. Empirically, unguided prompting is unreliable, while instructions plus flags substantially improve detection quality; adding calibrated scoring further stabilizes results. On a balanced subset of two hundred flows, a 7B instruction-tuned model with flags reaches macro-F1 near 0.78; a lighter 3B model with few-shot cues and calibration attains F1 near 0.68 on one thousand examples. As the evaluation set grows to two thousand flows, decision quality decreases, revealing sensitivity to coverage and prompting. Tabular baselines remain more stable and faster, yet the prompt-only pipeline requires no gradient training, produces readable artifacts, and adapts easily through instructions and flags. Contributions include a flow-to-text protocol with interpretable cues, a calibration method for thresholding, a systematic baseline comparison, and a reproducibility bundle with prompts, grammar, metrics, and figures. 

---
# Outraged AI: Large language models prioritise emotion over cost in fairness enforcement 

**Authors**: Hao Liu, Yiqing Dai, Haotian Tan, Yu Lei, Yujia Zhou, Zhen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17880)  

**Abstract**: Emotions guide human decisions, but whether large language models (LLMs) use emotion similarly remains unknown. We tested this using altruistic third-party punishment, where an observer incurs a personal cost to enforce fairness, a hallmark of human morality and often driven by negative emotion. In a large-scale comparison of 4,068 LLM agents with 1,159 adults across 796,100 decisions, LLMs used emotion to guide punishment, sometimes even more strongly than humans did: Unfairness elicited stronger negative emotion that led to more punishment; punishing unfairness produced more positive emotion than accepting; and critically, prompting self-reports of emotion causally increased punishment. However, mechanisms diverged: LLMs prioritized emotion over cost, enforcing norms in an almost all-or-none manner with reduced cost sensitivity, whereas humans balanced fairness and cost. Notably, reasoning models (o3-mini, DeepSeek-R1) were more cost-sensitive and closer to human behavior than foundation models (GPT-3.5, DeepSeek-V3), yet remained heavily emotion-driven. These findings provide the first causal evidence of emotion-guided moral decisions in LLMs and reveal deficits in cost calibration and nuanced fairness judgements, reminiscent of early-stage human responses. We propose that LLMs progress along a trajectory paralleling human development; future models should integrate emotion with context-sensitive reasoning to achieve human-like emotional intelligence. 

---
# JT-Safe: Intrinsically Enhancing the Safety and Trustworthiness of LLMs 

**Authors**: Junlan Feng, Fanyu Meng, Chong Long, Pengyu Cong, Duqing Wang, Yan Zheng, Yuyao Zhang, Xuanchang Gao, Ye Yuan, Yunfei Ma, Zhijie Ren, Fan Yang, Na Wu, Di Jin, Chao Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.17918)  

**Abstract**: The hallucination and credibility concerns of large language models (LLMs) are global challenges that the industry is collectively addressing. Recently, a significant amount of advances have been made on post-training and inference techniques to mitigate these challenges. However, it is widely agreed that unsafe and hallucinations of LLMs intrinsically originate from pre-training, involving pre-training data and the next-token prediction learning mechanism. In this paper, we focus on enhancing pre-training data to improve the trustworthiness and safety of LLMs. Since the data is vast, it's almost impossible to entirely purge the data of factual errors, logical inconsistencies, or distributional biases. Moreover, the pre-training data lack grounding in real-world knowledge. Each piece of data is treated as a sequence of tokens rather than as a representation of a part of the world. To overcome these issues, we propose approaches to enhancing our pre-training data with its context in the world and increasing a substantial amount of data reflecting industrial scenarios. We argue that most source data are created by the authors for specific purposes in a certain spatial-temporal context. They have played a role in the real world. By incorporating related world context information, we aim to better anchor pre-training data within real-world scenarios, thereby reducing uncertainty in model training and enhancing the model's safety and trustworthiness. We refer to our Data with World Context as DWC. We continue pre-training an earlier checkpoint of JT-35B-Base with 1.5 trillion of DWC tokens. We introduce our post-training procedures to activate the potentials of DWC. Compared with the Qwen model of a similar scale, JT-Safe-35B achieves an average performance improvement of 1.79% on the Safety and Trustworthy evaluation benchmarks, while being pretrained with only 6.2 trillion tokens. 

---
# Modeling Layered Consciousness with Multi-Agent Large Language Models 

**Authors**: Sang Hun Kim, Jongmin Lee, Dongkyu Park, So Young Lee, Yosep Chong  

**Link**: [PDF](https://arxiv.org/pdf/2510.17844)  

**Abstract**: We propose a multi-agent framework for modeling artificial consciousness in large language models (LLMs), grounded in psychoanalytic theory. Our \textbf{Psychodynamic Model} simulates self-awareness, preconsciousness, and unconsciousness through agent interaction, guided by a Personalization Module combining fixed traits and dynamic needs. Using parameter-efficient fine-tuning on emotionally rich dialogues, the system was evaluated across eight personalized conditions. An LLM as a judge approach showed a 71.2\% preference for the fine-tuned model, with improved emotional depth and reduced output variance, demonstrating its potential for adaptive, personalized cognition. 

---
# GRETEL: A Goal-driven Retrieval and Execution-based Trial Framework for LLM Tool Selection Enhancing 

**Authors**: Zongze Wu, Yani Guo, Churong Liang, Runnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.17843)  

**Abstract**: Despite remarkable advances in Large Language Model capabilities, tool retrieval for agent-based systems remains fundamentally limited by reliance on semantic similarity, which fails to capture functional viability. Current methods often retrieve textually relevant but functionally inoperative tools due to parameter mismatches, authentication failures, and execution constraints--a phenomenon we term the semantic-functional gap. We introduce GRETEL, to address this gap through systematic empirical validation. GRETEL implements an agentic workflow that processes semantically retrieved candidates through sandboxed plan-execute-evaluate cycles, generating execution-grounded evidence to distinguish truly functional tools from merely descriptive matches. Our comprehensive evaluation on the ToolBench benchmark demonstrates substantial improvements across all metrics: Pass Rate (at 10) increases from 0.690 to 0.826, Recall (at 10) improves from 0.841 to 0.867, and NDCG (at 10) rises from 0.807 to 0.857.. These results establish that execution-based validation provides a more reliable foundation for tool selection than semantic similarity alone, enabling more robust agent performance in real-world applications. 

---
# LLM Assisted Alpha Fairness for 6 GHz WiFi and NR_U Coexistence: An Agentic Orchestrator for Throughput, Energy, and SLA 

**Authors**: Qun Wang, Yingzhou Lu, Guiran Liu, Binrong Zhu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17814)  

**Abstract**: Unlicensed 6GHz is becoming a primary workhorse for high-capacity access, with Wi-Fi and 5G NR-U competing for the same channels under listen-before-talk (LBT) rules. Operating in this regime requires decisions that jointly trade throughput, energy, and service-level objectives while remaining safe and auditable. We present an agentic controller that separates {policy} from {execution}. At the start of each scheduling epoch the agent summarizes telemetry (per-channel busy and baseline LBT failure; per-user CQI, backlog, latency, battery, priority, and power mode) and invokes a large language model (LLM) to propose a small set of interpretable knobs: a fairness index \alpha, per-channel duty-cycle caps for Wi-Fi/NR-U, and class weights. A deterministic optimizer then enforces feasibility and computes an \alpha-fair allocation that internalizes LBT losses and energy cost; malformed or unsafe policies are clamped and fall back to a rule baseline. In a 6GHz simulator with two 160MHz channels and mixed Wi-Fi/NR-U users, LLM-assisted policies consistently improve energy efficiency while keeping throughput competitive with a strong rule baseline. One LLM lowers total energy by 35.3% at modest throughput loss, and another attains the best overall trade-off, finishing with higher total bits (+3.5%) and higher bits/J (+12.2%) than the baseline. We release code, per-epoch logs, and plotting utilities to reproduce all figures and numbers, illustrating how transparent, policy-level LLM guidance can safely improve wireless coexistence. 

---
# KAT-Coder Technical Report 

**Authors**: Zizheng Zhan, Ken Deng, Xiaojiang Zhang, Jinghui Wang, Huaixi Tang, Zhiyi Lai, Haoyang Huang, Wen Xiang, Kun Wu, Wenhao Zhuang, Minglei Zhang, Shaojie Wang, Shangpeng Yan, Kepeng Lei, Zongxian Feng, Huiming Wang, Zheng Lin, Mengtong Li, Mengfei Xie, Yinghan Cui, Xuxing Chen, Chao Wang, Weihao Li, Wenqiang Zhu, Jiarong Zhang, Jingxuan Xu, Songwei Yu, Yifan Yao, Xinping Lei, Han Li, Junqi Xiong, Zuchen Gao, Dailin Li, Haimo Li, Jiaheng Liu, Yuqun Zhang, Junyi Peng, Haotian Zhang, Bin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18779)  

**Abstract**: Recent advances in large language models (LLMs) have enabled progress in agentic coding, where models autonomously reason, plan, and act within interactive software development workflows. However, bridging the gap between static text-based training and dynamic real-world agentic execution remains a core challenge. In this technical report, we present KAT-Coder, a large-scale agentic code model trained through a multi-stage curriculum encompassing Mid-Term Training, Supervised Fine-Tuning (SFT), Reinforcement Fine-Tuning (RFT), and Reinforcement-to-Deployment Adaptation. The Mid-Term stage enhances reasoning, planning, and reflection capabilities through a corpus of real software engineering data and synthetic agentic interactions. The SFT stage constructs a million-sample dataset balancing twenty programming languages, ten development contexts, and ten task archetypes. The RFT stage introduces a novel multi-ground-truth reward formulation for stable and sample-efficient policy optimization. Finally, the Reinforcement-to-Deployment phase adapts the model to production-grade IDE environments using Error-Masked SFT and Tree-Structured Trajectory Training. In summary, these stages enable KAT-Coder to achieve robust tool-use reliability, instruction alignment, and long-context reasoning, forming a deployable foundation for real-world intelligent coding agents. Our KAT series 32B model, KAT-Dev, has been open-sourced on this https URL. 

---
# Investigating LLM Capabilities on Long Context Comprehension for Medical Question Answering 

**Authors**: Feras AlMannaa, Talia Tseriotou, Jenny Chim, Maria Liakata  

**Link**: [PDF](https://arxiv.org/pdf/2510.18691)  

**Abstract**: This study is the first to investigate LLM comprehension capabilities over long-context (LC) medical QA of clinical relevance. Our comprehensive assessment spans a range of content-inclusion settings based on their relevance, LLM models of varying capabilities and datasets across task formulations, revealing insights on model size effects, limitations, underlying memorization issues and the benefits of reasoning models. Importantly, we examine the effect of RAG on medical LC comprehension, uncover best settings in single versus multi-document reasoning datasets and showcase RAG strategies for improvements over LC. We shed light into some of the evaluation aspects using a multi-faceted approach. Our qualitative and error analyses address open questions on when RAG is beneficial over LC, revealing common failure cases. 

---
# MTraining: Distributed Dynamic Sparse Attention for Efficient Ultra-Long Context Training 

**Authors**: Wenxuan Li, Chengruidong Zhang, Huiqiang Jiang, Yucheng Li, Yuqing Yang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18830)  

**Abstract**: The adoption of long context windows has become a standard feature in Large Language Models (LLMs), as extended contexts significantly enhance their capacity for complex reasoning and broaden their applicability across diverse scenarios. Dynamic sparse attention is a promising approach for reducing the computational cost of long-context. However, efficiently training LLMs with dynamic sparse attention on ultra-long contexts-especially in distributed settings-remains a significant challenge, due in large part to worker- and step-level imbalance. This paper introduces MTraining, a novel distributed methodology leveraging dynamic sparse attention to enable efficient training for LLMs with ultra-long contexts. Specifically, MTraining integrates three key components: a dynamic sparse training pattern, balanced sparse ring attention, and hierarchical sparse ring attention. These components are designed to synergistically address the computational imbalance and communication overheads inherent in dynamic sparse attention mechanisms during the training of models with extensive context lengths. We demonstrate the efficacy of MTraining by training Qwen2.5-3B, successfully expanding its context window from 32K to 512K tokens on a cluster of 32 A100 GPUs. Our evaluations on a comprehensive suite of downstream tasks, including RULER, PG-19, InfiniteBench, and Needle In A Haystack, reveal that MTraining achieves up to a 6x higher training throughput while preserving model accuracy. Our code is available at this https URL. 

---
# DART: A Structured Dataset of Regulatory Drug Documents in Italian for Clinical NLP 

**Authors**: Mariano Barone, Antonio Laudante, Giuseppe Riccio, Antonio Romano, Marco Postiglione, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2510.18475)  

**Abstract**: The extraction of pharmacological knowledge from regulatory documents has become a key focus in biomedical natural language processing, with applications ranging from adverse event monitoring to AI-assisted clinical decision support. However, research in this field has predominantly relied on English-language corpora such as DrugBank, leaving a significant gap in resources tailored to other healthcare systems. To address this limitation, we introduce DART (Drug Annotation from Regulatory Texts), the first structured corpus of Italian Summaries of Product Characteristics derived from the official repository of the Italian Medicines Agency (AIFA). The dataset was built through a reproducible pipeline encompassing web-scale document retrieval, semantic segmentation of regulatory sections, and clinical summarization using a few-shot-tuned large language model with low-temperature decoding. DART provides structured information on key pharmacological domains such as indications, adverse drug reactions, and drug-drug interactions. To validate its utility, we implemented an LLM-based drug interaction checker that leverages the dataset to infer clinically meaningful interactions. Experimental results show that instruction-tuned LLMs can accurately infer potential interactions and their clinical implications when grounded in the structured textual fields of DART. We publicly release our code on GitHub: this https URL. 

---
# IMB: An Italian Medical Benchmark for Question Answering 

**Authors**: Antonio Romano, Giuseppe Riccio, Mariano Barone, Marco Postiglione, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2510.18468)  

**Abstract**: Online medical forums have long served as vital platforms where patients seek professional healthcare advice, generating vast amounts of valuable knowledge. However, the informal nature and linguistic complexity of forum interactions pose significant challenges for automated question answering systems, especially when dealing with non-English languages. We present two comprehensive Italian medical benchmarks: \textbf{IMB-QA}, containing 782,644 patient-doctor conversations from 77 medical categories, and \textbf{IMB-MCQA}, comprising 25,862 multiple-choice questions from medical specialty examinations. We demonstrate how Large Language Models (LLMs) can be leveraged to improve the clarity and consistency of medical forum data while retaining their original meaning and conversational style, and compare a variety of LLM architectures on both open and multiple-choice question answering tasks. Our experiments with Retrieval Augmented Generation (RAG) and domain-specific fine-tuning reveal that specialized adaptation strategies can outperform larger, general-purpose models in medical question answering tasks. These findings suggest that effective medical AI systems may benefit more from domain expertise and efficient information retrieval than from increased model scale. We release both datasets and evaluation frameworks in our GitHub repository to support further research on multilingual medical question answering: this https URL. 

---
# Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within the Response 

**Authors**: Qingqing Gu, Dan Wang, Yue Zhao, Xiaoyu Wang, Zhonglin Jiang, Yong Chen, Hongyan Li, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.18434)  

**Abstract**: Chain-of-Thought (CoT) is widely applied to improve the LLM capability in math, coding and reasoning tasks. However, its performance is limited for open-domain tasks since there are no clearly defined reasoning steps or logical transitions. To mitigate such challenges, we propose another prompt-based paradigm called Chain of Conceptual Thought (CoCT), where the LLM first tags a concept, then generates the detailed content. The chain of concepts is allowed within the utterance, encouraging the LLM's deep and strategic thinking. We experiment with this paradigm in daily and emotional support conversations where the concept is comprised of emotions, strategies and topics. Automatic, human and model evaluations suggest that CoCT surpasses baselines such as Self-Refine, ECoT, ToT, SoT and RAG, suggesting a potential effective prompt-based paradigm of LLM for a wider scope of tasks. 

---
# Identity-Aware Large Language Models require Cultural Reasoning 

**Authors**: Alistair Plum, Anne-Marie Lutgen, Christoph Purschke, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.18510)  

**Abstract**: Large language models have become the latest trend in natural language processing, heavily featuring in the digital tools we use every day. However, their replies often reflect a narrow cultural viewpoint that overlooks the diversity of global users. This missing capability could be referred to as cultural reasoning, which we define here as the capacity of a model to recognise culture-specific knowledge values and social norms, and to adjust its output so that it aligns with the expectations of individual users. Because culture shapes interpretation, emotional resonance, and acceptable behaviour, cultural reasoning is essential for identity-aware AI. When this capacity is limited or absent, models can sustain stereotypes, ignore minority perspectives, erode trust, and perpetuate hate. Recent empirical studies strongly suggest that current models default to Western norms when judging moral dilemmas, interpreting idioms, or offering advice, and that fine-tuning on survey data only partly reduces this tendency. The present evaluation methods mainly report static accuracy scores and thus fail to capture adaptive reasoning in context. Although broader datasets can help, they cannot alone ensure genuine cultural competence. Therefore, we argue that cultural reasoning must be treated as a foundational capability alongside factual accuracy and linguistic coherence. By clarifying the concept and outlining initial directions for its assessment, a foundation is laid for future systems to be able to respond with greater sensitivity to the complex fabric of human culture. 

---
# KoSimpleQA: A Korean Factuality Benchmark with an Analysis of Reasoning LLMs 

**Authors**: Donghyeon Ko, Yeguk Jin, Kyubyung Chae, Byungwook Lee, Chansong Jo, Sookyo In, Jaehong Lee, Taesup Kim, Donghyun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.18368)  

**Abstract**: We present $\textbf{Korean SimpleQA (KoSimpleQA)}$, a benchmark for evaluating factuality in large language models (LLMs) with a focus on Korean cultural knowledge. KoSimpleQA is designed to be challenging yet easy to grade, consisting of 1,000 short, fact-seeking questions with unambiguous answers. We conduct a comprehensive evaluation across a diverse set of open-source LLMs of varying sizes that support Korean, and find that even the strongest model generates correct answer only 33.7% of the time, underscoring the challenging nature of KoSimpleQA. Notably, performance rankings on KoSimpleQA differ substantially from those on the English SimpleQA, highlighting the unique value of our dataset. Furthermore, our analysis of reasoning LLMs shows that engaging reasoning capabilities in the factual QA task can both help models better elicit their latent knowledge and improve their ability to abstain when uncertain. KoSimpleQA can be found at this https URL. 

---
# ECG-LLM-- training and evaluation of domain-specific large language models for electrocardiography 

**Authors**: Lara Ahrens, Wilhelm Haverkamp, Nils Strodthoff  

**Link**: [PDF](https://arxiv.org/pdf/2510.18339)  

**Abstract**: Domain-adapted open-weight large language models (LLMs) offer promising healthcare applications, from queryable knowledge bases to multimodal assistants, with the crucial advantage of local deployment for privacy preservation. However, optimal adaptation strategies, evaluation methodologies, and performance relative to general-purpose LLMs remain poorly characterized. We investigated these questions in electrocardiography, an important area of cardiovascular medicine, by finetuning open-weight models on domain-specific literature and implementing a multi-layered evaluation framework comparing finetuned models, retrieval-augmented generation (RAG), and Claude Sonnet 3.7 as a representative general-purpose model. Finetuned Llama 3.1 70B achieved superior performance on multiple-choice evaluations and automatic text metrics, ranking second to Claude 3.7 in LLM-as-a-judge assessments. Human expert evaluation favored Claude 3.7 and RAG approaches for complex queries. Finetuned models significantly outperformed their base counterparts across nearly all evaluation modes. Our findings reveal substantial performance heterogeneity across evaluation methodologies, underscoring assessment complexity. Nevertheless, domain-specific adaptation through finetuning and RAG achieves competitive performance with proprietary models, supporting the viability of privacy-preserving, locally deployable clinical solutions. 

---
# KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers 

**Authors**: Mohd Ruhul Ameen, Akif Islam, Farjana Aktar, M. Saifuzzaman Rafat  

**Link**: [PDF](https://arxiv.org/pdf/2510.18355)  

**Abstract**: In Bangladesh, many farmers continue to face challenges in accessing timely, expert-level agricultural guidance. This paper presents KrishokBondhu, a voice-enabled, call-centre-integrated advisory platform built on a Retrieval-Augmented Generation (RAG) framework, designed specifically for Bengali-speaking farmers. The system aggregates authoritative agricultural handbooks, extension manuals, and NGO publications; applies Optical Character Recognition (OCR) and document-parsing pipelines to digitize and structure the content; and indexes this corpus in a vector database for efficient semantic retrieval. Through a simple phone-based interface, farmers can call the system to receive real-time, context-aware advice: speech-to-text converts the Bengali query, the RAG module retrieves relevant content, a large language model (Gemma 3-4B) generates a context-grounded response, and text-to-speech delivers the answer in natural spoken Bengali. In a pilot evaluation, KrishokBondhu produced high-quality responses for 72.7% of diverse agricultural queries covering crop management, disease control, and cultivation practices. Compared to the KisanQRS benchmark, the system achieved a composite score of 4.53 (vs. 3.13) on a 5-point scale, a 44.7% improvement, with especially large gains in contextual richness (+367%) and completeness (+100.4%), while maintaining comparable relevance and technical specificity. Semantic similarity analysis further revealed a strong correlation between retrieved context and answer quality, emphasizing the importance of grounding generative responses in curated documentation. KrishokBondhu demonstrates the feasibility of integrating call-centre accessibility, multilingual voice interaction, and modern RAG techniques to deliver expert-level agricultural guidance to remote Bangladeshi farmers, paving the way toward a fully AI-driven agricultural advisory ecosystem. 

---
# LLMs Encode How Difficult Problems Are 

**Authors**: William Lugoloobi, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2510.18147)  

**Abstract**: Large language models exhibit a puzzling inconsistency: they solve complex problems yet frequently fail on seemingly simpler ones. We investigate whether LLMs internally encode problem difficulty in a way that aligns with human judgment, and whether this representation tracks generalization during reinforcement learning post-training. We train linear probes across layers and token positions on 60 models, evaluating on mathematical and coding subsets of Easy2HardBench. We find that human-labeled difficulty is strongly linearly decodable (AMC: $\rho \approx 0.88$) and exhibits clear model-size scaling, whereas LLM-derived difficulty is substantially weaker and scales poorly. Steering along the difficulty direction reveals that pushing models toward "easier" representations reduces hallucination and improves accuracy. During GRPO training on Qwen2.5-Math-1.5B, the human-difficulty probe strengthens and positively correlates with test accuracy across training steps, while the LLM-difficulty probe degrades and negatively correlates with performance. These results suggest that human annotations provide a stable difficulty signal that RL amplifies, while automated difficulty estimates derived from model performance become misaligned precisely as models improve. We release probe code and evaluation scripts to facilitate replication. 

---
# CMT-Bench: Cricket Multi-Table Generation Benchmark for Probing Robustness in Large Language Models 

**Authors**: Ritam Upadhyay, Naman Ahuja, Rishabh Baral, Aparna Garimella, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.18173)  

**Abstract**: LLM Driven text-to-table (T2T) systems often rely on extensive prompt-engineering or iterative event extraction in code-parsable formats, which boosts scores but are computationally expensive and obscure how models actually reason over temporal evolving narratives to summarise key information. We present CMT-Bench, a diagnostic benchmark built from live cricket commentary that requires dynamic table generation across two evolving schemas under a dense, rule-governed policy. CMT-Bench is designed to probe robustness via three semantics-preserving dimensions: (i) extractive-cue ablation to separate extractive shortcuts from state tracking, (ii) temporal prefixing to test long-context stability, and (iii) entity-form perturbations (anonymization, outof-distribution substitutions, role-entangling paraphrases) to assess sensitivity to surface variation. Across diverse long-context stateof-the-art LLMs, we find large drops without extractive summaries, monotonic degradation with input length, and consistent accuracy drop under entity-form changes. Complementary distributional tests confirm significant shifts in numeric error patterns, indicating drift in reasoning rather than mere noise. Our results show that current LLMs are brittle in dynamic Textto-table generation, motivating robustness-first evaluation as a prerequisite for developing efficient and scalable approaches for this task. 

---
# Chain-of-Thought Reasoning Improves Context-Aware Translation with Large Language Models 

**Authors**: Shabnam Ataee, Andrei Popescu-Belis  

**Link**: [PDF](https://arxiv.org/pdf/2510.18077)  

**Abstract**: This paper assesses the capacity of large language models (LLMs) to translate texts that include inter-sentential dependencies. We use the English-French DiscEvalMT benchmark (Bawden et al., 2018) with pairs of sentences containing translation challenges either for pronominal anaphora or for lexical cohesion. We evaluate 12 LLMs from the DeepSeek-R1, GPT, Llama, Mistral and Phi families on two tasks: (1) distinguishing a correct translation from a wrong but plausible one; (2) generating a correct translation. We compare prompts that encourage chain-of-thought reasoning with those that do not. The best models take advantage of reasoning and reach about 90% accuracy on the first task, and COMET scores of about 92% on the second task, with GPT-4, GPT-4o and Phi standing out. Moreover, we observe a "wise get wiser" effect: the improvements through reasoning are positively correlated with the scores of the models without reasoning. 

---
# Does Reasoning Help LLM Agents Play Dungeons and Dragons? A Prompt Engineering Experiment 

**Authors**: Patricia Delafuente, Arya Honraopatil, Lara J. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2510.18112)  

**Abstract**: This paper explores the application of Large Language Models (LLMs) and reasoning to predict Dungeons & Dragons (DnD) player actions and format them as Avrae Discord bot commands. Using the FIREBALL dataset, we evaluated a reasoning model, DeepSeek-R1-Distill-LLaMA-8B, and an instruct model, LLaMA-3.1-8B-Instruct, for command generation. Our findings highlight the importance of providing specific instructions to models, that even single sentence changes in prompts can greatly affect the output of models, and that instruct models are sufficient for this task compared to reasoning models. 

---
# See the Text: From Tokenization to Visual Reading 

**Authors**: Ling Xing, Alex Jinpeng Wang, Rui Yan, Hongyu Qu, Zechao Li, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18840)  

**Abstract**: People see text. Humans read by recognizing words as visual objects, including their shapes, layouts, and patterns, before connecting them to meaning, which enables us to handle typos, distorted fonts, and various scripts effectively. Modern large language models (LLMs), however, rely on subword tokenization, fragmenting text into pieces from a fixed vocabulary. While effective for high-resource languages, this approach over-segments low-resource languages, yielding long, linguistically meaningless sequences and inflating computation. In this work, we challenge this entrenched paradigm and move toward a vision-centric alternative. Our method, SeeTok, renders text as images (visual-text) and leverages pretrained multimodal LLMs to interpret them, reusing strong OCR and text-vision alignment abilities learned from large-scale multimodal training. Across three different language tasks, SeeTok matches or surpasses subword tokenizers while requiring 4.43 times fewer tokens and reducing FLOPs by 70.5%, with additional gains in cross-lingual generalization, robustness to typographic noise, and linguistic hierarchy. SeeTok signals a shift from symbolic tokenization to human-like visual reading, and takes a step toward more natural and cognitively inspired language models. 

---
# Does GenAI Rewrite How We Write? An Empirical Study on Two-Million Preprints 

**Authors**: Minfeng Qi, Zhongmin Cao, Qin Wang, Ningran Li, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17882)  

**Abstract**: Preprint repositories become central infrastructures for scholarly communication. Their expansion transforms how research is circulated and evaluated before journal publication. Generative large language models (LLMs) introduce a further potential disruption by altering how manuscripts are written. While speculation abounds, systematic evidence of whether and how LLMs reshape scientific publishing remains limited.
This paper addresses the gap through a large-scale analysis of more than 2.1 million preprints spanning 2016--2025 (115 months) across four major repositories (i.e., arXiv, bioRxiv, medRxiv, SocArXiv). We introduce a multi-level analytical framework that integrates interrupted time-series models, collaboration and productivity metrics, linguistic profiling, and topic modeling to assess changes in volume, authorship, style, and disciplinary orientation. Our findings reveal that LLMs have accelerated submission and revision cycles, modestly increased linguistic complexity, and disproportionately expanded AI-related topics, while computationally intensive fields benefit more than others. These results show that LLMs act less as universal disruptors than as selective catalysts, amplifying existing strengths and widening disciplinary divides. By documenting these dynamics, the paper provides the first empirical foundation for evaluating the influence of generative AI on academic publishing and highlights the need for governance frameworks that preserve trust, fairness, and accountability in an AI-enabled research ecosystem. 

---
# BrailleLLM: Braille Instruction Tuning with Large Language Models for Braille Domain Tasks 

**Authors**: Tianyuan Huang, Zepeng Zhu, Hangdi Xing, Zirui Shao, Zhi Yu, Chaoxiong Yang, Jiaxian He, Xiaozhong Liu, Jiajun Bu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18288)  

**Abstract**: Braille plays a vital role in education and information accessibility for visually impaired individuals. However, Braille information processing faces challenges such as data scarcity and ambiguities in mixed-text contexts. We construct English and Chinese Braille Mixed Datasets (EBMD/CBMD) with mathematical formulas to support diverse Braille domain research, and propose a syntax tree-based augmentation method tailored for Braille data. To address the underperformance of traditional fine-tuning methods in Braille-related tasks, we investigate Braille Knowledge-Based Fine-Tuning (BKFT), which reduces the learning difficulty of Braille contextual features. BrailleLLM employs BKFT via instruction tuning to achieve unified Braille translation, formula-to-Braille conversion, and mixed-text translation. Experiments demonstrate that BKFT achieves significant performance improvements over conventional fine-tuning in Braille translation scenarios. Our open-sourced datasets and methodologies establish a foundation for low-resource multilingual Braille research. 

---
# Evaluating LLM-Based Mobile App Recommendations: An Empirical Study 

**Authors**: Quim Motger, Xavier Franch, Vincenzo Gervasi, Jordi Marco  

**Link**: [PDF](https://arxiv.org/pdf/2510.18364)  

**Abstract**: Large Language Models (LLMs) are increasingly used to recommend mobile applications through natural language prompts, offering a flexible alternative to keyword-based app store search. Yet, the reasoning behind these recommendations remains opaque, raising questions about their consistency, explainability, and alignment with traditional App Store Optimization (ASO) metrics. In this paper, we present an empirical analysis of how widely-used general purpose LLMs generate, justify, and rank mobile app recommendations. Our contributions are: (i) a taxonomy of 16 generalizable ranking criteria elicited from LLM outputs; (ii) a systematic evaluation framework to analyse recommendation consistency and responsiveness to explicit ranking instructions; and (iii) a replication package to support reproducibility and future research on AI-based recommendation systems. Our findings reveal that LLMs rely on a broad yet fragmented set of ranking criteria, only partially aligned with standard ASO metrics. While top-ranked apps tend to be consistent across runs, variability increases with ranking depth and search specificity. LLMs exhibit varying sensitivity to explicit ranking instructions - ranging from substantial adaptations to near-identical outputs - highlighting their complex reasoning dynamics in conversational app discovery. Our results aim to support end-users, app developers, and recommender-systems researchers in navigating the emerging landscape of conversational app discovery. 

---
# LLMs as Sparse Retrievers:A Framework for First-Stage Product Search 

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Sen Li, Wenjun Peng, Fuyu Lv, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.18527)  

**Abstract**: Product search is a crucial component of modern e-commerce platforms, with billions of user queries every day. In product search systems, first-stage retrieval should achieve high recall while ensuring efficient online deployment. Sparse retrieval is particularly attractive in this context due to its interpretability and storage efficiency. However, sparse retrieval methods suffer from severe vocabulary mismatch issues, leading to suboptimal performance in product search this http URL their potential for semantic analysis, large language models (LLMs) offer a promising avenue for mitigating vocabulary mismatch issues and thereby improving retrieval quality. Directly applying LLMs to sparse retrieval in product search exposes two key challenges:(1)Queries and product titles are typically short and highly susceptible to LLM-induced hallucinations, such as generating irrelevant expansion terms or underweighting critical literal terms like brand names and model numbers;(2)The large vocabulary space of LLMs leads to difficulty in initializing training effectively, making it challenging to learn meaningful sparse representations in such ultra-high-dimensional this http URL address these challenges, we propose PROSPER, a framework for PROduct search leveraging LLMs as SParsE Retrievers. PROSPER incorporates: (1)A literal residual network that alleviates hallucination in lexical expansion by reinforcing underweighted literal terms through a residual compensation mechanism; and (2)A lexical focusing window that facilitates effective training initialization via a coarse-to-fine sparsification this http URL offline and online experiments show that PROSPER significantly outperforms sparse baselines and achieves recall performance comparable to advanced dense retrievers, while also achieving revenue increments online. 

---
# Enhancing Hotel Recommendations with AI: LLM-Based Review Summarization and Query-Driven Insights 

**Authors**: Nikolaos Belibasakis, Anastasios Giannaros, Ioanna Giannoukou, Spyros Sioutas  

**Link**: [PDF](https://arxiv.org/pdf/2510.18277)  

**Abstract**: The increasing number of data a booking platform such as this http URL and AirBnB offers make it challenging for interested parties to browse through the available accommodations and analyze reviews in an efficient way. Efforts have been made from the booking platform providers to utilize recommender systems in an effort to enable the user to filter the results by factors such as stars, amenities, cost but most valuable insights can be provided by the unstructured text-based reviews. Going through these reviews one-by-one requires a substantial amount of time to be devoted while a respectable percentage of the reviews won't provide to the user what they are actually looking for.
This research publication explores how Large Language Models (LLMs) can enhance short rental apartments recommendations by summarizing and mining key insights from user reviews. The web application presented in this paper, named "instaGuide", automates the procedure of isolating the text-based user reviews from a property on the this http URL platform, synthesizing the summary of the reviews, and enabling the user to query specific aspects of the property in an effort to gain feedback on their personal questions/criteria.
During the development of the instaGuide tool, numerous LLM models were evaluated based on accuracy, cost, and response quality. The results suggest that the LLM-powered summarization reduces significantly the amount of time the users need to devote on their search for the right short rental apartment, improving the overall decision-making procedure. 

---
