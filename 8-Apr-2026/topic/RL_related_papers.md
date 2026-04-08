# Offline RL for Adaptive Policy Retrieval in Prior Authorization 

**Authors**: Ruslan Sharifullin, Maxim Gorshkov, Hannah Clay  

**Link**: [PDF](https://arxiv.org/pdf/2604.05125)  

**Abstract**: Prior authorization (PA) requires interpretation of complex and fragmented coverage policies, yet existing retrieval-augmented systems rely on static top-$K$ strategies with fixed numbers of retrieved sections. Such fixed retrieval can be inefficient and gather irrelevant or insufficient information. We model policy retrieval for PA as a sequential decision-making problem, formulating adaptive retrieval as a Markov Decision Process (MDP). In our system, an agent iteratively selects policy chunks from a top-$K$ candidate set or chooses to stop and issue a decision. The reward balances decision correctness against retrieval cost, capturing the trade-off between accuracy and efficiency. We train policies using Conservative Q-Learning (CQL), Implicit Q-Learning (IQL), and Direct Preference Optimization (DPO) in an offline RL setting on logged trajectories generated from baseline retrieval strategies over synthetic PA requests derived from publicly available CMS coverage data. On a corpus of 186 policy chunks spanning 10 CMS procedures, CQL achieves 92% decision accuracy (+30 percentage points over the best fixed-$K$ baseline) via exhaustive retrieval, while IQL matches the best baseline accuracy using 44% fewer retrieval steps and achieves the only positive episodic return among all policies. Transition-level DPO matches CQL's 92% accuracy while using 47% fewer retrieval steps (10.6 vs. 20.0), occupying a "selective-accurate" region on the Pareto frontier that dominates both CQL and BC. A behavioral cloning baseline matches CQL, confirming that advantage-weighted or preference-based policy extraction is needed to learn selective retrieval. Lambda ablation over step costs $\lambda \in \{0.05, 0.1, 0.2\}$ reveals a clear accuracy-efficiency inflection: only at $\lambda = 0.2$ does CQL transition from exhaustive to selective retrieval. 

---
# Can Large Language Models Reinvent Foundational Algorithms? 

**Authors**: Jian Zhao, Haoren Luo, Yu Wang, Yuhan Cao, Pingyue Sheng, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2604.05716)  

**Abstract**: LLMs have shown strong potential to advance scientific discovery. Whether they possess the capacity for foundational innovation, however, remains an open question. In this work, we focus on a prerequisite for foundational innovation: can LLMs reinvent foundational algorithms in computer science? Our \textit{Unlearn-and-Reinvent} pipeline applies LLM unlearning to remove a specific foundational algorithm, such as Dijkstra's or Euclid's algorithm, from an LLM's pretrained knowledge, and then tests whether the model can reinvent it in a controlled environment. To enable effective unlearning, we adopt a GRPO-based, on-policy unlearning method. Across 10 target algorithms, 3 strong open-weight models, and 3 hint levels, our experiments demonstrate that (1) the strongest model Qwen3-4B-Thinking-2507 successfully reinvents 50% of the algorithms with no hint, 70% at hint level 1, and 90% at hint level 2; (2) a few high-level hints can enhance the reinvention success rate, but even step-by-step hints fail for those complicated algorithms; and (3) test-time reinforcement learning enables successful reinvention for the Strassen algorithm at hint level 2. Through analyses of output trajectories and ablation studies, we find that generative verifier in the reinvention phase plays a critical role in sustaining models' reasoning strength, helping to avoid the ``thought collapse'' phenomenon. These findings offer insights into both the potential and current limits of LLMs' innovative thinking. 

---
# COSMO-Agent: Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration 

**Authors**: Liyuan Deng, Shujian Deng, Yongkang Chen, Yongkang Dai, Zhihang Zhong, Linyang Li, Xiao Sun, Yilei Shi, Huaxi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.05547)  

**Abstract**: Iterative industrial design-simulation optimization is bottlenecked by the CAD-CAE semantic gap: translating simulation feedback into valid geometric edits under diverse, coupled constraints. To fill this gap, we propose COSMO-Agent (Closed-loop Optimization, Simulation, and Modeling Orchestration), a tool-augmented reinforcement learning (RL) framework that teaches LLMs to complete the closed-loop CAD-CAE process. Specifically, we cast CAD generation, CAE solving, result parsing, and geometry revision as an interactive RL environment, where an LLM learns to orchestrate external tools and revise parametric geometries until constraints are satisfied. To make this learning stable and industrially usable, we design a multi-constraint reward that jointly encourages feasibility, toolchain robustness, and structured output validity. In addition, we contribute an industry-aligned dataset that covers 25 component categories with executable CAD-CAE tasks to support realistic training and evaluation. Experiments show that COSMO-Agent training substantially improves small open-source LLMs for constraint-driven design, exceeding large open-source and strong closed-source models in feasibility, efficiency, and stability. 

---
# UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning 

**Authors**: Xiaolong Wei, Zerun Zhu, Simin Niu, Xingyu Zhang, Peiying Yu, Changxuan Xiao, Yuchen Li, Jicheng Yang, Zhejun Zhao, Chong Meng, Long Xia, Daiting Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.05517)  

**Abstract**: A fundamental challenge in creative writing lies in reconciling the inherent tension between maintaining global coherence in long-form narratives and preserving local expressiveness in short-form texts. While long-context generation necessitates explicit macroscopic planning, short-form creativity often demands spontaneous, constraint-free expression. Existing alignment paradigms, however, typically employ static reward signals and rely heavily on high-quality supervised data, which is costly and difficult to scale. To address this, we propose \textbf{UniCreative}, a unified reference-free reinforcement learning framework. We first introduce \textbf{AC-GenRM}, an adaptive constraint-aware reward model that dynamically synthesizes query-specific criteria to provide fine-grained preference judgments. Leveraging these signals, we propose \textbf{ACPO}, a policy optimization algorithm that aligns models with human preferences across both content quality and structural paradigms without supervised fine-tuning and ground-truth references. Empirical results demonstrate that AC-GenRM aligns closely with expert evaluations, while ACPO significantly enhances performance across diverse writing tasks. Crucially, our analysis reveals an emergent meta-cognitive ability: the model learns to autonomously differentiate between tasks requiring rigorous planning and those favoring direct generation, validating the effectiveness of our direct alignment approach. 

---
# PRISM-MCTS: Learning from Reasoning Trajectories with Metacognitive Reflection 

**Authors**: Siyuan Cheng, Bozhong Tian, YanChao Hao, Zheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.05424)  

**Abstract**: PRISM-MCTS: Learning from Reasoning Trajectories with Metacognitive Reflection Siyuan Cheng, Bozhong Tian, Yanchao Hao, Zheng Wei Published: 06 Apr 2026, Last Modified: 06 Apr 2026 ACL 2026 Findings Conference, Area Chairs, Reviewers, Publication Chairs, Authors Revisions BibTeX CC BY 4.0 Keywords: Efficient/Low-Resource Methods for NLP, Generation, Question Answering  The emergence of reasoning models, exemplified by OpenAI o1, signifies a transition from intuitive to deliberative cognition, effectively reorienting the scaling laws from pre-training paradigms toward test-time computation. While Monte Carlo Tree Search (MCTS) has shown promise in this domain, existing approaches typically treat each rollout as an isolated trajectory. This lack of information sharing leads to severe inefficiency and substantial computational redundancy, as the search process fails to leverage insights from prior explorations. To address these limitations, we propose PRISM-MCTS, a novel reasoning framework that draws inspiration from human parallel thinking and reflective processes. PRISM-MCTS integrates a Process Reward Model (PRM) with a dynamic shared memory, capturing both "Heuristics" and "Fallacies". By reinforcing successful strategies and pruning error-prone branches, PRISM-MCTS effectively achieves refinement. Furthermore, we develop a data-efficient training strategy for the PRM, achieving high-fidelity evaluation under a few-shot regime. Empirical evaluations across diverse reasoning benchmarks substantiate the efficacy of PRISM-MCTS. Notably, it halves the trajectory requirements on GPQA while surpassing MCTS-RAG and Search-o1, demonstrating that it scales inference by reasoning judiciously rather than exhaustively. 

---
# Pressure, What Pressure? Sycophancy Disentanglement in Language Models via Reward Decomposition 

**Authors**: Muhammad Ahmed Mohsin, Ahsan Bilal, Muhammad Umer, Emily Fox  

**Link**: [PDF](https://arxiv.org/pdf/2604.05279)  

**Abstract**: Large language models exhibit sycophancy, the tendency to shift their stated positions toward perceived user preferences or authority cues regardless of evidence. Standard alignment methods fail to correct this because scalar reward models conflate two distinct failure modes into a single signal: pressure capitulation, where the model changes a correct answer under social pressure, and evidence blindness, where the model ignores the provided context entirely. We operationalise sycophancy through formal definitions of pressure independence and evidence responsiveness, serving as a working framework for disentangled training rather than a definitive characterisation of the phenomenon. We propose the first approach to sycophancy reduction via reward decomposition, introducing a multi-component Group Relative Policy Optimisation (GRPO) reward that decomposes the training signal into five terms: pressure resistance, context fidelity, position consistency, agreement suppression, and factual correctness. We train using a contrastive dataset pairing pressure-free baselines with pressured variants across three authority levels and two opposing evidence contexts. Across five base models, our two-phase pipeline consistently reduces sycophancy on all metric axes, with ablations confirming that each reward term governs an independent behavioural dimension. The learned resistance to pressure generalises beyond our training methodology and prompt structure, reducing answer-priming sycophancy by up to 17 points on SycophancyEval despite the absence of such pressure forms during training. 

---
# TRACE: Capability-Targeted Agentic Training 

**Authors**: Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Azalia Mirhoseini  

**Link**: [PDF](https://arxiv.org/pdf/2604.05336)  

**Abstract**: Large Language Models (LLMs) deployed in agentic environments must exercise multiple capabilities across different task instances, where a capability is performing one or more actions in a trajectory that are necessary for successfully solving a subset of tasks in the environment. Many existing approaches either rely on synthetic training data that is not targeted to the model's actual capability deficits in the target environment or train directly on the target environment, where the model needs to implicitly learn the capabilities across tasks. We introduce TRACE (Turning Recurrent Agent failures into Capability-targeted training Environments), an end-to-end system for environment-specific agent self-improvement. TRACE contrasts successful and failed trajectories to automatically identify lacking capabilities, synthesizes a targeted training environment for each that rewards whether the capability was exercised, and trains a LoRA adapter via RL on each synthetic environment, routing to the relevant adapter at inference. Empirically, TRACE generalizes across different environments, improving over the base agent by +14.1 points on $\tau^2$-bench (customer service) and +7 perfect scores on ToolSandbox (tool use), outperforming the strongest baseline by +7.4 points and +4 perfect scores, respectively. Given the same number of rollouts, TRACE scales more efficiently than baselines, outperforming GRPO and GEPA by +9.2 and +7.4 points on $\tau^2$-bench. 

---
# ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning 

**Authors**: Xuan Xiong, Huan Liu, Li Gu, Zhixiang Chi, Yue Qiu, Yuanhao Yu, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.05355)  

**Abstract**: Chain-of-thought (CoT) reasoning improves large language model performance on complex tasks, but often produces excessively long and inefficient reasoning traces. Existing methods shorten CoTs using length penalties or global entropy reduction, implicitly assuming that low uncertainty is desirable throughout reasoning. We show instead that reasoning efficiency is governed by the trajectory of uncertainty. CoTs with dominant downward entropy trends are substantially shorter. Motivated by this insight, we propose Entropy Trend Reward (ETR), a trajectory-aware objective that encourages progressive uncertainty reduction while allowing limited local exploration. We integrate ETR into Group Relative Policy Optimization (GRPO) and evaluate it across multiple reasoning models and challenging benchmarks. ETR consistently achieves a superior accuracy-efficiency tradeoff, improving DeepSeek-R1-Distill-7B by 9.9% in accuracy while reducing CoT length by 67% across four benchmarks. Code is available at this https URL 

---
# MMEmb-R1: Reasoning-Enhanced Multimodal Embedding with Pair-Aware Selection and Adaptive Control 

**Authors**: Yuchi Wang, Haiyang Yu, Weikang Bian, Jiefeng Long, Xiao Liang, Chao Feng, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.06156)  

**Abstract**: MLLMs have been successfully applied to multimodal embedding tasks, yet their generative reasoning capabilities remain underutilized. Directly incorporating chain-of-thought reasoning into embedding learning introduces two fundamental challenges. First, structural misalignment between instance-level reasoning and pairwise contrastive supervision may lead to shortcut behavior, where the model merely learns the superficial format of reasoning. Second, reasoning is not universally beneficial for embedding tasks. Enforcing reasoning for all inputs may introduce unnecessary computation and latency, and can even obscure salient semantic signals for simple cases. To address these issues, we propose MMEmb-R1, an adaptive reasoning-based multimodal embedding framework. We formulate reasoning as a latent variable and introduce pair-aware reasoning selection that employs counterfactual intervention to identify reasoning paths beneficial for query-target alignment. Furthermore, we adopt reinforcement learning to selectively invoke reasoning only when necessary. Experiments on the MMEB-V2 benchmark demonstrate that our model achieves a score of 71.2 with only 4B parameters, establishing a new state-of-the-art while significantly reducing reasoning overhead and inference latency. 

---
# Scientific Graphics Program Synthesis via Dual Self-Consistency Reinforcement Learning 

**Authors**: Juekai Lin, Yun Zhu, Honglin Lin, Sijing Li, Tianwei Lin, Zheng Liu, Xiaoyang Wang, Wenqiao Zhang, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06079)  

**Abstract**: Graphics Program Synthesis is pivotal for interpreting and editing visual data, effectively facilitating the reverse-engineering of static visuals into editable TikZ code. While TikZ is the de facto standard for scientific schematics due to its programmatic flexibility, its requirement for rigorous spatial precision presents a significant challenge for Multimodal Large Language Models. Progress is currently stifled by two primary gaps: (1) Data Quality Gap: existing image-TikZ corpora often lack strict executability and reliable visual alignment; (2) Evaluation Gap: a lack of benchmarks for both structural and visual fidelity. To address these, we present a closed-loop framework featuring: SciTikZ-230K, a large-scale, high-quality dataset from our Execution-Centric Data Engine covering 11 diverse scientific disciplines; SciTikZ-Bench, a multifaceted benchmark spanning from basic geometric constructs to intricate hierarchical schematics to evaluate both visual fidelity and structural logic. To further broaden the scope of visual-code optimization methodology, we introduce a novel Dual Self-Consistency Reinforcement Learning optimization paradigm, which utilizes Round-Trip Verification to penalize degenerate code and boost overall self-consistency. Empowered by these, our trained model SciTikZer-8B achieves state-of-the-art performance, consistently outperforming proprietary giants like Gemini-2.5-Pro and massive models like Qwen3-VL-235B-A22B-Instruct. 

---
# Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling 

**Authors**: Qiyuan Chen, Hongsen Huang, Jiahe Chen, Qian Shao, Jintai Chen, Hongxia Xu, Renjie Hua, Chuan Ren, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.05445)  

**Abstract**: Vision-language reward modeling faces a dilemma: generative approaches are interpretable but slow, while discriminative ones are efficient but act as opaque "black boxes." To bridge this gap, we propose VL-MDR (Vision-Language Multi-Dimensional Reward), a framework that dynamically decomposes evaluation into granular, interpretable dimensions. Instead of outputting a monolithic scalar, VL-MDR employs a visual-aware gating mechanism to identify relevant dimensions and adaptively weight them (e.g., Hallucination, Reasoning) for each specific input. To support this, we curate a dataset of 321k vision-language preference pairs annotated across 21 fine-grained dimensions. Extensive experiments show that VL-MDR consistently outperforms existing open-source reward models on benchmarks like VL-RewardBench. Furthermore, we show that VL-MDR-constructed preference pairs effectively enable DPO alignment to mitigate visual hallucinations and improve reliability, providing a scalable solution for VLM alignment. 

---
# Not All Turns Are Equally Hard: Adaptive Thinking Budgets For Efficient Multi-Turn Reasoning 

**Authors**: Neharika Jali, Anupam Nayak, Gauri Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.05164)  

**Abstract**: As LLM reasoning performance plateau, improving inference-time compute efficiency is crucial to mitigate overthinking and long thinking traces even for simple queries. Prior approaches including length regularization, adaptive routing, and difficulty-based budget allocation primarily focus on single-turn settings and fail to address the sequential dependencies inherent in multi-turn this http URL this work, we formulate multi-turn reasoning as a sequential compute allocation problem and model it as a multi-objective Markov Decision Process. We propose TAB: Turn-Adaptive Budgets, a budget allocation policy trained via Group Relative Policy Optimization (GRPO) that learns to maximize task accuracy while respecting global per-problem token constraints. Consequently, TAB takes as input the conversation history and learns to adaptively allocate smaller budgets to easier turns and save appropriate number of tokens for the crucial harder reasoning steps. Our experiments on mathematical reasoning benchmarks demonstrate that TAB achieves a superior accuracy-tokens tradeoff saving up to 35% tokens while maintaining accuracy over static and off-the-shelf LLM budget baselines. Further, for systems where a plan of all sub-questions is available apriori, we propose TAB All-SubQ, a budget allocation policy that budgets tokens based on the conversation history and all past and future sub-questions saving up to 40% tokens over baselines. 

---
# Scaling Coding Agents via Atomic Skills 

**Authors**: Yingwei Ma, Yue Liu, Xinlong Yang, Yanhao Li, Kelin Fu, Yibo Miao, Yuchong Xie, Zhexu Wang, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2604.05013)  

**Abstract**: Current LLM coding agents are predominantly trained on composite benchmarks (e.g., bug fixing), which often leads to task-specific overfitting and limited generalization. To address this, we propose a novel scaling paradigm that shifts the focus from task-level optimization to atomic skill mastery. We first formalize five fundamental atomic skills, code localization, code editing, unit-test generation, issue reproduction, and code review, that serve as the basis vectors for complex software engineering tasks. Compared with composite coding tasks, these atomic skills are more generalizable and composable. Then, we scale coding agents by performing joint RL over atomic skills. In this manner, atomic skills are consistently improved without negative interference or trade-offs between them. Notably, we observe that improvements in these atomic skills generalize well to other unseen composite coding tasks, such as bug-fixing, code refactoring, machine learning engineering, and code security. The observation motivates a new scaling paradigm for coding agents by training with atomic skills. Extensive experiments demonstrate the effectiveness of our proposed paradigm. Notably, our joint RL improves average performance by 18.7% on 5 atomic skills and 5 composite tasks. 

---
# Reasoning Through Chess: How Reasoning Evolves from Data Through Fine-Tuning and Reinforcement Learning 

**Authors**: Lucas Dionisopoulos, Nicklas Majamaki, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2604.05134)  

**Abstract**: How can you get a language model to reason in a task it natively struggles with? We study how reasoning evolves in a language model -- from supervised fine-tuning (SFT) to reinforcement learning (RL) -- by analyzing how a set of theoretically-inspired datasets impacts language model performance in chess. We find that fine-tuning a model to directly predict the best move leads to effective RL and the strongest downstream performance -- however, the RL step elicits unfaithful reasoning (reasoning inconsistent with the chosen move). Alternatively, training on multi-move trajectories yields comparable downstream performance with faithful reasoning and more stable RL. We show that RL induces a substantial positive shift in the distribution of move quality and reduces hallucination rates as a side effect. Finally, we find several SFT-checkpoint metrics -- metrics spanning evaluation performance, hallucination rates, and reasoning quality -- to be predictive of post-RL model performance. We release checkpoints and final models as well as training data, evaluations, and code which allowed us to surpass leading open-source reasoning models in chess with a 7B-parameter model. 

---
# AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning 

**Authors**: Yuanfu Sun, Kang Li, Dongzhe Fan, Jiajin Liu, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.05846)  

**Abstract**: Large Language Models (LLMs) increasingly rely on agentic capabilities-iterative retrieval, tool use, and decision-making-to overcome the limits of static, parametric knowledge. Yet existing agentic frameworks treat external information as unstructured text and fail to leverage the topological dependencies inherent in real-world data. To bridge this gap, we introduce Agentic Graph Learning (AGL), a paradigm that reframes graph learning as an interleaved process of topology-aware navigation and LLM-based inference. Specifically, we propose AgentGL, the first reinforcement learning (RL)-driven framework for AGL. AgentGL equips an LLM agent with graph-native tools for multi-scale exploration, regulates tool usage via search-constrained thinking to balance accuracy and efficiency, and employs a graph-conditioned curriculum RL strategy to stabilize long-horizon policy learning without step-wise supervision. Across diverse Text-Attributed Graph (TAG) benchmarks and multiple LLM backbones, AgentGL substantially outperforms strong GraphLLMs and GraphRAG baselines, achieving absolute improvements of up to 17.5% in node classification and 28.4% in link prediction. These results demonstrate that AGL is a promising frontier for enabling LLMs to autonomously navigate and reason over complex relational environments. The code is publicly available at this https URL. 

---
# Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting 

**Authors**: Jinhu Fu, Yan Bai, Longzhu He, Yihang Lou, Yanxiao Zhao, Li Sun, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.05540)  

**Abstract**: Large language models (LLMs) can effectively handle outdated information through knowledge editing. However, current approaches face two key limitations: (I) Poor generalization: Most approaches rigidly inject new knowledge without ensuring that the model can use it effectively to solve practical problems. (II) Narrow scope: Current methods focus primarily on structured fact triples, overlooking the diverse unstructured forms of factual information (e.g., news, articles) prevalent in real-world contexts. To address these challenges, we propose a new paradigm: teaching LLMs to edit knowledge via Chain of Thoughts (CoTs) reasoning (CoT2Edit). We first leverage language model agents for both structured and unstructured edited data to generate CoTs, building high-quality instruction data. The model is then trained to reason over edited knowledge through supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO). At inference time, we integrate Retrieval-Augmented Generation (RAG) to dynamically retrieve relevant edited facts for real-time knowledge editing. Experimental results demonstrate that our method achieves strong generalization across six diverse knowledge editing scenarios with just a single round of training on three open-source language models. The codes are available at this https URL. 

---
# Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs 

**Authors**: Hongyuan Yuan, Xinran He, Run Shao, Bolei He, Xianwei Xue, Mengke Chen, Qiutong Pan, Haiwei Wang, Haifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.05643)  

**Abstract**: Extending CoT through RL has been widely used to enhance the reasoning capabilities of LLMs. However, due to the sparsity of reward signals, it can also induce undesirable thinking patterns such as overthinking, i.e., generating redundant intermediate reasoning content. In this work, we argue that a major source of such redundancy is inefficient reflection, which often manifests in two problematic patterns: Indiscriminate Reflection, where the model performs broad, low-impact checks throughout reasoning, and Repetitive Reflection, where it repeatedly re-verifies an already established conclusion. To address this, we introduce a graph-based CoT optimization framework. Specifically, we convert each linear CoT into a directed acyclic graph (DAG) with explicit dependency edges, and design a dual pruning strategy: branch-level pruning removes weakly contributing reflection branches, while depth-level pruning eliminates late-stage re-verification. We distill this behavior via a three-stage pipeline: (1) SFT to initialize the policy on pruned concise traces, (2) DPO to prefer correct but less redundant trajectories, and (3) GRPO with length penalty to jointly optimize answer correctness and efficiency. Experiments show that our approach reduces the average reasoning tokens by 42\% while maintaining or improving accuracy. 

---
# Cross-Modal Coreference Alignment: Enabling Reliable Information Transfer in Omni-LLMs 

**Authors**: Hongcheng Liu, Yuhao Wang, Zhe Chen, Pingjie Wang, Zhiyuan Zhu, Yixuan Hou, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.05522)  

**Abstract**: Omni Large Language Models (Omni-LLMs) have demonstrated impressive capabilities in holistic multi-modal perception, yet they consistently falter in complex scenarios requiring synergistic omni-modal reasoning. Beyond understanding global multimodal context, effective reasoning also hinges on fine-grained cross-modal alignment, especially identifying shared referents across modalities, yet this aspect has been largely overlooked. To bridge this gap, we formalize the challenge as a cross-modal coreference problem, where a model must localize a referent in a source modality and re-identify it in a target modality. Building on this paradigm, we introduce CrossOmni, a dataset comprising nine tasks equipped with human-designed reasoning rationales to evaluate and enhance this capability. Experiments on 13 Omni-LLMs reveal systematic weaknesses in cross-modal coreference, which we attribute to the absence of coreference-aware thinking patterns. To address this, we enhance cross-modal alignment via two strategies: a training-free In-Context Learning method and a training-based SFT+GRPO framework designed to induce such thinking patterns. Both approaches yield substantial performance gains and generalize effectively to collaborative reasoning tasks. Overall, our findings highlight cross-modal coreference as a crucial missing piece for advancing robust omni-modal reasoning. 

---
# Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification 

**Authors**: Jinhong Jeong, Junghun Park, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.05302)  

**Abstract**: Text simplification supports second language (L2) learning by providing comprehensible input, consistent with the Input Hypothesis. However, constructing personalized parallel corpora is costly, while existing large language model (LLM)-based readability control methods rely on pre-labeled sentence corpora and primarily target English. We propose Re-RIGHT, a unified reinforcement learning framework for adaptive multilingual text simplification without parallel corpus supervision. We first show that prompting-based lexical simplification at target proficiency levels (CEFR, JLPT, TOPIK, and HSK) performs poorly at easier levels and for non-English languages, even with state-of-the-art LLMs such as GPT-5.2 and Gemini 2.5. To address this, we collect 43K vocabulary-level data across four languages (English, Japanese, Korean, and Chinese) and train a compact 4B policy model using Re-RIGHT, which integrates three reward modules: vocabulary coverage, semantic preservation, and coherence. Compared to the stronger LLM baselines, Re-RIGHT achieves higher lexical coverage at target proficiency levels while maintaining original meaning and fluency. 

---
# SenseAI: A Human-in-the-Loop Dataset for RLHF-Aligned Financial Sentiment Reasoning 

**Authors**: Berny Kabalisa  

**Link**: [PDF](https://arxiv.org/pdf/2604.05135)  

**Abstract**: We introduce SenseAI, a human-in-the-loop (HITL) validated financial sentiment dataset designed to capture not only model outputs but the full reasoning process behind them. Unlike existing resources, SenseAI incorporates reasoning chains, confidence scores, human correction signals, and real-world market outcomes, providing a structure aligned with Reinforcement Learning from Human Feedback (RLHF) paradigms.
The dataset consists of 1,439 labelled data points across 40 US-listed equities and 13 financial data categories, enabling direct integration into modern LLM fine-tuning pipelines. Through analysis, we identify several systematic patterns in model behavior, including a novel failure mode we term Latent Reasoning Drift, where models introduce information not grounded in the input, as well as consistent confidence miscalibration and forward projection tendencies.
These findings suggest that LLM errors in financial reasoning are not random but occur within a predictable and correctable regime, supporting the use of structured HITL data for targeted model improvement. We discuss implications for financial AI systems and highlight opportunities for applying SenseAI in model evaluation and alignment. 

---
