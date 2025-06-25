# JoyAgents-R1: Joint Evolution Dynamics for Versatile Multi-LLM Agents with Reinforcement Learning 

**Authors**: Ai Han, Junxing Hu, Pu Wei, Zhiqian Zhang, Yuhang Guo, Jiawei Lu, Zicheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19846)  

**Abstract**: Multi-agent reinforcement learning (MARL) has emerged as a prominent paradigm for increasingly complex tasks. However, joint evolution across heterogeneous agents remains challenging due to cooperative inefficiency and training instability. In this paper, we propose the joint evolution dynamics for MARL called JoyAgents-R1, which first applies Group Relative Policy Optimization (GRPO) to the joint training of heterogeneous multi-agents. By iteratively refining agents' large language models (LLMs) and memories, the method achieves holistic equilibrium with optimal decision-making and memory capabilities. Specifically, JoyAgents-R1 first implements node-wise Monte Carlo sampling on the behavior of each agent across entire reasoning trajectories to enhance GRPO sampling efficiency while maintaining policy diversity. Then, our marginal benefit-driven selection strategy identifies top-$K$ sampling groups with maximal reward fluctuations, enabling targeted agent model updates that improve training stability and maximize joint benefits through cost-effective parameter adjustments. Meanwhile, JoyAgents-R1 introduces an adaptive memory evolution mechanism that repurposes GRPO rewards as cost-free supervisory signals to eliminate repetitive reasoning and accelerate convergence. Experiments across general and domain-specific scenarios demonstrate that JoyAgents-R1 achieves performance comparable to that of larger LLMs while built on smaller open-source models. 

---
# Temporal-IRL: Modeling Port Congestion and Berth Scheduling with Inverse Reinforcement Learning 

**Authors**: Guo Li, Zixiang Xu, Wei Zhang, Yikuan Hu, Xinyu Yang, Nikolay Aristov, Mingjie Tang, Elenna R Dugundji  

**Link**: [PDF](https://arxiv.org/pdf/2506.19843)  

**Abstract**: Predicting port congestion is crucial for maintaining reliable global supply chains. Accurate forecasts enableimprovedshipment planning, reducedelaysand costs, and optimizeinventoryanddistributionstrategies, thereby ensuring timely deliveries and enhancing supply chain resilience. To achieve accurate predictions, analyzing vessel behavior and their stay times at specific port terminals is essential, focusing particularly on berth scheduling under various conditions. Crucially, the model must capture and learn the underlying priorities and patterns of berth scheduling. Berth scheduling and planning are influenced by a range of factors, including incoming vessel size, waiting times, and the status of vessels within the port terminal. By observing historical Automatic Identification System (AIS) positions of vessels, we reconstruct berth schedules, which are subsequently utilized to determine the reward function via Inverse Reinforcement Learning (IRL). For this purpose, we modeled a specific terminal at the Port of New York/New Jersey and developed Temporal-IRL. This Temporal-IRL model learns berth scheduling to predict vessel sequencing at the terminal and estimate vessel port stay, encompassing both waiting and berthing times, to forecast port congestion. Utilizing data from Maher Terminal spanning January 2015 to September 2023, we trained and tested the model, achieving demonstrably excellent results. 

---
# Evaluating Compliance with Visualization Guidelines in Diagrams for Scientific Publications Using Large Vision Language Models 

**Authors**: Johannes Rückert, Louise Bloch, Christoph M. Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.19825)  

**Abstract**: Diagrams are widely used to visualize data in publications. The research field of data visualization deals with defining principles and guidelines for the creation and use of these diagrams, which are often not known or adhered to by researchers, leading to misinformation caused by providing inaccurate or incomplete information.
In this work, large Vision Language Models (VLMs) are used to analyze diagrams in order to identify potential problems in regards to selected data visualization principles and guidelines. To determine the suitability of VLMs for these tasks, five open source VLMs and five prompting strategies are compared using a set of questions derived from selected data visualization guidelines.
The results show that the employed VLMs work well to accurately analyze diagram types (F1-score 82.49 %), 3D effects (F1-score 98.55 %), axes labels (F1-score 76.74 %), lines (RMSE 1.16), colors (RMSE 1.60) and legends (F1-score 96.64 %, RMSE 0.70), while they cannot reliably provide feedback about the image quality (F1-score 0.74 %) and tick marks/labels (F1-score 46.13 %). Among the employed VLMs, Qwen2.5VL performs best, and the summarizing prompting strategy performs best for most of the experimental questions.
It is shown that VLMs can be used to automatically identify a number of potential issues in diagrams, such as missing axes labels, missing legends, and unnecessary 3D effects. The approach laid out in this work can be extended for further aspects of data visualization. 

---
# KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality 

**Authors**: Baochang Ren, Shuofei Qiao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19807)  

**Abstract**: Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at this https URL. 

---
# Learning Task Belief Similarity with Latent Dynamics for Meta-Reinforcement Learning 

**Authors**: Menglong Zhang, Fuyuan Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.19785)  

**Abstract**: Meta-reinforcement learning requires utilizing prior task distribution information obtained during exploration to rapidly adapt to unknown tasks. The efficiency of an agent's exploration hinges on accurately identifying the current task. Recent Bayes-Adaptive Deep RL approaches often rely on reconstructing the environment's reward signal, which is challenging in sparse reward settings, leading to suboptimal exploitation. Inspired by bisimulation metrics, which robustly extracts behavioral similarity in continuous MDPs, we propose SimBelief-a novel meta-RL framework via measuring similarity of task belief in Bayes-Adaptive MDP (BAMDP). SimBelief effectively extracts common features of similar task distributions, enabling efficient task identification and exploration in sparse reward environments. We introduce latent task belief metric to learn the common structure of similar tasks and incorporate it into the specific task belief. By learning the latent dynamics across task distributions, we connect shared latent task belief features with specific task features, facilitating rapid task identification and adaptation. Our method outperforms state-of-the-art baselines on sparse reward MuJoCo and panda-gym tasks. 

---
# SAGE: Strategy-Adaptive Generation Engine for Query Rewriting 

**Authors**: Teng Wang, Hailei Gong, Changwang Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19783)  

**Abstract**: Query rewriting is pivotal for enhancing dense retrieval, yet current methods demand large-scale supervised data or suffer from inefficient reinforcement learning (RL) exploration. In this work, we first establish that guiding Large Language Models (LLMs) with a concise set of expert-crafted strategies, such as semantic expansion and entity disambiguation, substantially improves retrieval effectiveness on challenging benchmarks, including HotpotQA, FEVER, NFCorpus, and SciFact. Building on this insight, we introduce the Strategy-Adaptive Generation Engine (SAGE), which operationalizes these strategies in an RL framework. SAGE introduces two novel reward shaping mechanisms-Strategic Credit Shaping (SCS) and Contrastive Reward Shaping (CRS)-to deliver more informative learning signals. This strategy-guided approach not only achieves new state-of-the-art NDCG@10 results, but also uncovers a compelling emergent behavior: the agent learns to select optimal strategies, reduces unnecessary exploration, and generates concise rewrites, lowering inference cost without sacrificing performance. Our findings demonstrate that strategy-guided RL, enhanced with nuanced reward shaping, offers a scalable, efficient, and more interpretable paradigm for developing the next generation of robust information retrieval systems. 

---
# Automatic Prompt Optimization for Knowledge Graph Construction: Insights from an Empirical Study 

**Authors**: Nandana Mihindukulasooriya, Niharika S. D'Souza, Faisal Chowdhury, Horst Samulowitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.19773)  

**Abstract**: A KG represents a network of entities and illustrates relationships between them. KGs are used for various applications, including semantic search and discovery, reasoning, decision-making, natural language processing, machine learning, and recommendation systems. Triple (subject-relation-object) extraction from text is the fundamental building block of KG construction and has been widely studied, for example, in early benchmarks such as ACE 2002 to more recent ones, such as WebNLG 2020, REBEL and SynthIE. While the use of LLMs is explored for KG construction, handcrafting reasonable task-specific prompts for LLMs is a labour-intensive exercise and can be brittle due to subtle changes in the LLM models employed. Recent work in NLP tasks (e.g. autonomy generation) uses automatic prompt optimization/engineering to address this challenge by generating optimal or near-optimal task-specific prompts given input-output examples.
This empirical study explores the application of automatic prompt optimization for the triple extraction task using experimental benchmarking. We evaluate different settings by changing (a) the prompting strategy, (b) the LLM being used for prompt optimization and task execution, (c) the number of canonical relations in the schema (schema complexity), (d) the length and diversity of input text, (e) the metric used to drive the prompt optimization, and (f) the dataset being used for training and testing. We evaluate three different automatic prompt optimizers, namely, DSPy, APE, and TextGrad and use two different triple extraction datasets, SynthIE and REBEL. Through rigorous empirical evaluation, our main contribution highlights that automatic prompt optimization techniques can generate reasonable prompts similar to humans for triple extraction. In turn, these optimized prompts achieve improved results, particularly with increasing schema complexity and text size. 

---
# From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking 

**Authors**: Gyeongwon James Kim, Alex Wilf, Louis-Philippe Morency, Daniel Fried  

**Link**: [PDF](https://arxiv.org/pdf/2506.19724)  

**Abstract**: Recent progress in autonomous code generation has fueled excitement around AI agents capable of accelerating scientific discovery by running experiments. However, there is currently no benchmark that evaluates whether such agents can implement scientific ideas when given varied amounts of code as a starting point, interpolating between reproduction (running code) and from-scratch replication (fully re-implementing and running code). We introduce AutoExperiment, a benchmark that evaluates AI agents' ability to implement and run machine learning experiments based on natural language descriptions in research papers. In each task, agents are given a research paper, a codebase with key functions masked out, and a command to run the experiment. The goal is to generate the missing code, execute the experiment in a sandboxed environment, and reproduce the results. AutoExperiment scales in difficulty by varying the number of missing functions $n$, ranging from partial reproduction to full replication. We evaluate state-of-the-art agents and find that performance degrades rapidly as $n$ increases. Agents that can dynamically interact with the environment (e.g. to debug their code) can outperform agents in fixed "agentless" harnesses, and there exists a significant gap between single-shot and multi-trial success rates (Pass@1 vs. Pass@5), motivating verifier approaches to our benchmark. Our findings highlight critical challenges in long-horizon code generation, context retrieval, and autonomous experiment execution, establishing AutoExperiment as a new benchmark for evaluating progress in AI-driven scientific experimentation. Our data and code are open-sourced at this https URL . 

---
# LLM-Driven Medical Document Analysis: Enhancing Trustworthy Pathology and Differential Diagnosis 

**Authors**: Lei Kang, Xuanshuo Fu, Oriol Ramos Terrades, Javier Vazquez-Corral, Ernest Valveny, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.19702)  

**Abstract**: Medical document analysis plays a crucial role in extracting essential clinical insights from unstructured healthcare records, supporting critical tasks such as differential diagnosis. Determining the most probable condition among overlapping symptoms requires precise evaluation and deep medical expertise. While recent advancements in large language models (LLMs) have significantly enhanced performance in medical document analysis, privacy concerns related to sensitive patient data limit the use of online LLMs services in clinical settings. To address these challenges, we propose a trustworthy medical document analysis platform that fine-tunes a LLaMA-v3 using low-rank adaptation, specifically optimized for differential diagnosis tasks. Our approach utilizes DDXPlus, the largest benchmark dataset for differential diagnosis, and demonstrates superior performance in pathology prediction and variable-length differential diagnosis compared to existing methods. The developed web-based platform allows users to submit their own unstructured medical documents and receive accurate, explainable diagnostic results. By incorporating advanced explainability techniques, the system ensures transparent and reliable predictions, fostering user trust and confidence. Extensive evaluations confirm that the proposed method surpasses current state-of-the-art models in predictive accuracy while offering practical utility in clinical settings. This work addresses the urgent need for reliable, explainable, and privacy-preserving artificial intelligence solutions, representing a significant advancement in intelligent medical document analysis for real-world healthcare applications. The code can be found at \href{this https URL}{this https URL}. 

---
# Toward Decision-Oriented Prognostics: An Integrated Estimate-Optimize Framework for Predictive Maintenance 

**Authors**: Zhuojun Xie, Adam Abdin, Yiping Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19698)  

**Abstract**: Recent research increasingly integrates machine learning (ML) into predictive maintenance (PdM) to reduce operational and maintenance costs in data-rich operational settings. However, uncertainty due to model misspecification continues to limit widespread industrial adoption. This paper proposes a PdM framework in which sensor-driven prognostics inform decision-making under economic trade-offs within a finite decision space. We investigate two key questions: (1) Does higher predictive accuracy necessarily lead to better maintenance decisions? (2) If not, how can the impact of prediction errors on downstream maintenance decisions be mitigated? We first demonstrate that in the traditional estimate-then-optimize (ETO) framework, errors in probabilistic prediction can result in inconsistent and suboptimal maintenance decisions. To address this, we propose an integrated estimate-optimize (IEO) framework that jointly tunes predictive models while directly optimizing for maintenance outcomes. We establish theoretical finite-sample guarantees on decision consistency under standard assumptions. Specifically, we develop a stochastic perturbation gradient descent algorithm suitable for small run-to-failure datasets. Empirical evaluations on a turbofan maintenance case study show that the IEO framework reduces average maintenance regret up to 22% compared to ETO. This study provides a principled approach to managing prediction errors in data-driven PdM. By aligning prognostic model training with maintenance objectives, the IEO framework improves robustness under model misspecification and improves decision quality. The improvement is particularly pronounced when the decision-making policy is misaligned with the decision-maker's target. These findings support more reliable maintenance planning in uncertain operational environments. 

---
# From memories to maps: Mechanisms of in context reinforcement learning in transformers 

**Authors**: Ching Fang, Kanaka Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19686)  

**Abstract**: Humans and animals show remarkable learning efficiency, adapting to new environments with minimal experience. This capability is not well captured by standard reinforcement learning algorithms that rely on incremental value updates. Rapid adaptation likely depends on episodic memory -- the ability to retrieve specific past experiences to guide decisions in novel contexts. Transformers provide a useful setting for studying these questions because of their ability to learn rapidly in-context and because their key-value architecture resembles episodic memory systems in the brain. We train a transformer to in-context reinforcement learn in a distribution of planning tasks inspired by rodent behavior. We then characterize the learning algorithms that emerge in the model. We first find that representation learning is supported by in-context structure learning and cross-context alignment, where representations are aligned across environments with different sensory stimuli. We next demonstrate that the reinforcement learning strategies developed by the model are not interpretable as standard model-free or model-based planning. Instead, we show that in-context reinforcement learning is supported by caching intermediate computations within the model's memory tokens, which are then accessed at decision time. Overall, we find that memory may serve as a computational resource, storing both raw experience and cached computations to support flexible behavior. Furthermore, the representations developed in the model resemble computations associated with the hippocampal-entorhinal system in the brain, suggesting that our findings may be relevant for natural cognition. Taken together, our work offers a mechanistic hypothesis for the rapid adaptation that underlies in-context learning in artificial and natural settings. 

---
# Identifying Macro Causal Effects in C-DMGs over DMGs 

**Authors**: Simon Ferreira, Charles K. Assaad  

**Link**: [PDF](https://arxiv.org/pdf/2506.19650)  

**Abstract**: The do-calculus is a sound and complete tool for identifying causal effects in acyclic directed mixed graphs (ADMGs) induced by structural causal models (SCMs). However, in many real-world applications, especially in high-dimensional setting, constructing a fully specified ADMG is often infeasible. This limitation has led to growing interest in partially specified causal representations, particularly through cluster-directed mixed graphs (C-DMGs), which group variables into clusters and offer a more abstract yet practical view of causal dependencies. While these representations can include cycles, recent work has shown that the do-calculus remains sound and complete for identifying macro-level causal effects in C-DMGs over ADMGs under the assumption that all clusters size are greater than 1. Nevertheless, real-world systems often exhibit cyclic causal dynamics at the structural level. To account for this, input-output structural causal models (ioSCMs) have been introduced as a generalization of SCMs that allow for cycles. ioSCMs induce another type of graph structure known as a directed mixed graph (DMG). Analogous to the ADMG setting, one can define C-DMGs over DMGs as high-level representations of causal relations among clusters of variables. In this paper, we prove that, unlike in the ADMG setting, the do-calculus is unconditionally sound and complete for identifying macro causal effects in C-DMGs over DMGs. Furthermore, we show that the graphical criteria for non-identifiability of macro causal effects previously established C-DMGs over ADMGs naturally extends to a subset of C-DMGs over DMGs. 

---
# On the efficacy of old features for the detection of new bots 

**Authors**: Rocco De Nicola, Marinella Petrocchi, Manuel Pratelli  

**Link**: [PDF](https://arxiv.org/pdf/2506.19635)  

**Abstract**: For more than a decade now, academicians and online platform administrators have been studying solutions to the problem of bot detection. Bots are computer algorithms whose use is far from being benign: malicious bots are purposely created to distribute spam, sponsor public characters and, ultimately, induce a bias within the public opinion. To fight the bot invasion on our online ecosystem, several approaches have been implemented, mostly based on (supervised and unsupervised) classifiers, which adopt the most varied account features, from the simplest to the most expensive ones to be extracted from the raw data obtainable through the Twitter public APIs. In this exploratory study, using Twitter as a benchmark, we compare the performances of four state-of-art feature sets in detecting novel bots: one of the output scores of the popular bot detector Botometer, which considers more than 1,000 features of an account to take a decision; two feature sets based on the account profile and timeline; and the information about the Twitter client from which the user tweets. The results of our analysis, conducted on six recently released datasets of Twitter accounts, hint at the possible use of general-purpose classifiers and cheap-to-compute account features for the detection of evolved bots. 

---
# Position: Intelligent Science Laboratory Requires the Integration of Cognitive and Embodied AI 

**Authors**: Sha Zhang, Suorong Yang, Tong Xie, Xiangyuan Xue, Zixuan Hu, Rui Li, Wenxi Qu, Zhenfei Yin, Tianfan Fu, Di Hu, Andres M Bran, Nian Ran, Bram Hoex, Wangmeng Zuo, Philippe Schwaller, Wanli Ouyang, Lei Bai, Yanyong Zhang, Lingyu Duan, Shixiang Tang, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.19613)  

**Abstract**: Scientific discovery has long been constrained by human limitations in expertise, physical capability, and sleep cycles. The recent rise of AI scientists and automated laboratories has accelerated both the cognitive and operational aspects of research. However, key limitations persist: AI systems are often confined to virtual environments, while automated laboratories lack the flexibility and autonomy to adaptively test new hypotheses in the physical world. Recent advances in embodied AI, such as generalist robot foundation models, diffusion-based action policies, fine-grained manipulation learning, and sim-to-real transfer, highlight the promise of integrating cognitive and embodied intelligence. This convergence opens the door to closed-loop systems that support iterative, autonomous experimentation and the possibility of serendipitous discovery. In this position paper, we propose the paradigm of Intelligent Science Laboratories (ISLs): a multi-layered, closed-loop framework that deeply integrates cognitive and embodied intelligence. ISLs unify foundation models for scientific reasoning, agent-based workflow orchestration, and embodied agents for robust physical experimentation. We argue that such systems are essential for overcoming the current limitations of scientific discovery and for realizing the full transformative potential of AI-driven science. 

---
# ChordPrompt: Orchestrating Cross-Modal Prompt Synergy for Multi-Domain Incremental Learning in CLIP 

**Authors**: Zhiyuan Wang, Bokui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19608)  

**Abstract**: Continual learning (CL) empowers pre-trained vision-language models to adapt effectively to novel or previously underrepresented data distributions without comprehensive retraining, enhancing their adaptability and efficiency. While vision-language models like CLIP show great promise, they struggle to maintain performance across domains in incremental learning scenarios. Existing prompt learning methods face two main limitations: 1) they primarily focus on class-incremental learning scenarios, lacking specific strategies for multi-domain task incremental learning; 2) most current approaches employ single-modal prompts, neglecting the potential benefits of cross-modal information exchange. To address these challenges, we propose the \ChordPrompt framework, which facilitates a harmonious interplay between visual and textual prompts. \ChordPrompt introduces cross-modal prompts to leverage interactions between visual and textual information. Our approach also employs domain-adaptive text prompts to select appropriate prompts for continual adaptation across multiple domains. Comprehensive experiments on multi-domain incremental learning benchmarks demonstrate that \ChordPrompt outperforms state-of-the-art methods in zero-shot generalization and downstream task performance. 

---
# Adaptive Domain Modeling with Language Models: A Multi-Agent Approach to Task Planning 

**Authors**: Harisankar Babu, Philipp Schillinger, Tamim Asfour  

**Link**: [PDF](https://arxiv.org/pdf/2506.19592)  

**Abstract**: We introduce TAPAS (Task-based Adaptation and Planning using AgentS), a multi-agent framework that integrates Large Language Models (LLMs) with symbolic planning to solve complex tasks without the need for manually defined environment models. TAPAS employs specialized LLM-based agents that collaboratively generate and adapt domain models, initial states, and goal specifications as needed using structured tool-calling mechanisms. Through this tool-based interaction, downstream agents can request modifications from upstream agents, enabling adaptation to novel attributes and constraints without manual domain redefinition. A ReAct (Reason+Act)-style execution agent, coupled with natural language plan translation, bridges the gap between dynamically generated plans and real-world robot capabilities. TAPAS demonstrates strong performance in benchmark planning domains and in the VirtualHome simulated real-world environment. 

---
# Interpretable Hybrid Machine Learning Models Using FOLD-R++ and Answer Set Programming 

**Authors**: Sanne Wielinga, Jesse Heyninck  

**Link**: [PDF](https://arxiv.org/pdf/2506.19573)  

**Abstract**: Machine learning (ML) techniques play a pivotal role in high-stakes domains such as healthcare, where accurate predictions can greatly enhance decision-making. However, most high-performing methods such as neural networks and ensemble methods are often opaque, limiting trust and broader adoption. In parallel, symbolic methods like Answer Set Programming (ASP) offer the possibility of interpretable logical rules but do not always match the predictive power of ML models. This paper proposes a hybrid approach that integrates ASP-derived rules from the FOLD-R++ algorithm with black-box ML classifiers to selectively correct uncertain predictions and provide human-readable explanations. Experiments on five medical datasets reveal statistically significant performance gains in accuracy and F1 score. This study underscores the potential of combining symbolic reasoning with conventional ML to achieve high interpretability without sacrificing accuracy. 

---
# NTRL: Encounter Generation via Reinforcement Learning for Dynamic Difficulty Adjustment in Dungeons and Dragons 

**Authors**: Carlo Romeo, Andrew D. Bagdanov  

**Link**: [PDF](https://arxiv.org/pdf/2506.19530)  

**Abstract**: Balancing combat encounters in Dungeons & Dragons (D&D) is a complex task that requires Dungeon Masters (DM) to manually assess party strength, enemy composition, and dynamic player interactions while avoiding interruption of the narrative flow. In this paper, we propose Encounter Generation via Reinforcement Learning (NTRL), a novel approach that automates Dynamic Difficulty Adjustment (DDA) in D&D via combat encounter design. By framing the problem as a contextual bandit, NTRL generates encounters based on real-time party members attributes. In comparison with classic DM heuristics, NTRL iteratively optimizes encounters to extend combat longevity (+200%), increases damage dealt to party members, reducing post-combat hit points (-16.67%), and raises the number of player deaths while maintaining low total party kills (TPK). The intensification of combat forces players to act wisely and engage in tactical maneuvers, even though the generated encounters guarantee high win rates (70%). Even in comparison with encounters designed by human Dungeon Masters, NTRL demonstrates superior performance by enhancing the strategic depth of combat while increasing difficulty in a manner that preserves overall game fairness. 

---
# NaviAgent: Bilevel Planning on Tool Dependency Graphs for Function Calling 

**Authors**: Yan Jiang, Hao Zhou, LiZhong GU, Ai Han, TianLong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19500)  

**Abstract**: LLMs' reliance on static knowledge and fragile tool invocation severely hinders the orchestration of complex, heterogeneous toolchains, particularly at large scales. Existing methods typically use rigid single-path execution, resulting in poor error recovery and exponentially growing search spaces. We introduce NaviAgent, a graph-navigated bilevel planning architecture for robust function calling, comprising a Multi-Path Decider and Graph-Encoded Navigator. As an LLM-powered agent, the Multi-Path Decider defines a four-dimensional decision space and continuously perceives environmental states, dynamically selecting the optimal action to fully cover all tool invocation scenarios. The Graph-Encoded Navigator constructs a Tool Dependency Heterogeneous Graph (TDHG), where node embeddings explicitly fuse API schema structure with historical invocation behavior. It also integrates a novel heuristic search strategy that guides the Decider toward efficient and highly successful toolchains, even for unseen tool combinations. Experiments show that NaviAgent consistently achieves the highest task success rate (TSR) across all foundation models and task complexities, outperforming the average baselines (ReAct, ToolLLM, {\alpha}-UMI) by 13.5%, 16.4%, and 19.0% on Qwen2.5-14B, Qwen2.5-32B, and Deepseek-V3, respectively. Its execution steps are typically within one step of the most efficient baseline, ensuring a strong balance between quality and efficiency. Notably, a fine-tuned Qwen2.5-14B model achieves a TSR of 49.5%, surpassing the much larger 32B model (44.9%) under our architecture. Incorporating the Graph-Encoded Navigator further boosts TSR by an average of 2.4 points, with gains up over 9 points on complex tasks for larger models (Deepseek-V3 and GPT-4o), highlighting its essential role in toolchain orchestration. 

---
# KunLunBaizeRAG: Reinforcement Learning Driven Inference Performance Leap for Large Language Models 

**Authors**: Cheng Li, Jiexiong Liu, Yixuan Chen, Qihang Zhou, KunLun Meta  

**Link**: [PDF](https://arxiv.org/pdf/2506.19466)  

**Abstract**: This paper introduces KunLunBaizeRAG, a reinforcement learning-driven reasoning framework designed to enhance the reasoning capabilities of large language models (LLMs) in complex multi-hop question-answering tasks. The framework addresses key limitations of traditional RAG, such as retrieval drift, information redundancy, and strategy rigidity. Key innovations include the RAG-driven Reasoning Alignment (RDRA) mechanism, the Search-Think Iterative Enhancement (STIE) mechanism, the Network-Local Intelligent Routing (NLR) mechanism, and a progressive hybrid training strategy. Experimental results demonstrate significant improvements in exact match (EM) and LLM-judged score (LJ) across four benchmarks, highlighting the framework's robustness and effectiveness in complex reasoning scenarios. 

---
# Commander-GPT: Dividing and Routing for Multimodal Sarcasm Detection 

**Authors**: Yazhou Zhang, Chunwang Zou, Bo Wang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.19420)  

**Abstract**: Multimodal sarcasm understanding is a high-order cognitive task. Although large language models (LLMs) have shown impressive performance on many downstream NLP tasks, growing evidence suggests that they struggle with sarcasm understanding. In this paper, we propose Commander-GPT, a modular decision routing framework inspired by military command theory. Rather than relying on a single LLM's capability, Commander-GPT orchestrates a team of specialized LLM agents where each agent will be selectively assigned to a focused sub-task such as context modeling, sentiment analysis, etc. Their outputs are then routed back to the commander, which integrates the information and performs the final sarcasm judgment. To coordinate these agents, we introduce three types of centralized commanders: (1) a trained lightweight encoder-based commander (e.g., multi-modal BERT); (2) four small autoregressive language models, serving as moderately capable commanders (e.g., DeepSeek-VL); (3) two large LLM-based commander (Gemini Pro and GPT-4o) that performs task routing, output aggregation, and sarcasm decision-making in a zero-shot fashion. We evaluate Commander-GPT on the MMSD and MMSD 2.0 benchmarks, comparing five prompting strategies. Experimental results show that our framework achieves 4.4% and 11.7% improvement in F1 score over state-of-the-art (SoTA) baselines on average, demonstrating its effectiveness. 

---
# Unsupervised Dataset Dictionary Learning for domain shift robust clustering: application to sitting posture identification 

**Authors**: Anas Hattay, Mayara Ayat, Fred Ngole Mboula  

**Link**: [PDF](https://arxiv.org/pdf/2506.19410)  

**Abstract**: This paper introduces a novel approach, Unsupervised Dataset Dictionary Learning (U-DaDiL), for totally unsupervised robust clustering applied to sitting posture identification. Traditional methods often lack adaptability to diverse datasets and suffer from domain shift issues. U-DaDiL addresses these challenges by aligning distributions from different datasets using Wasserstein barycenter based representation. Experimental evaluations on the Office31 dataset demonstrate significant improvements in cluster alignment accuracy. This work also presents a promising step for addressing domain shift and robust clustering for unsupervised sitting posture identification 

---
# Is an object-centric representation beneficial for robotic manipulation ? 

**Authors**: Alexandre Chapin, Emmanuel Dellandrea, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19408)  

**Abstract**: Object-centric representation (OCR) has recently become a subject of interest in the computer vision community for learning a structured representation of images and videos. It has been several times presented as a potential way to improve data-efficiency and generalization capabilities to learn an agent on downstream tasks. However, most existing work only evaluates such models on scene decomposition, without any notion of reasoning over the learned representation. Robotic manipulation tasks generally involve multi-object environments with potential inter-object interaction. We thus argue that they are a very interesting playground to really evaluate the potential of existing object-centric work. To do so, we create several robotic manipulation tasks in simulated environments involving multiple objects (several distractors, the robot, etc.) and a high-level of randomization (object positions, colors, shapes, background, initial positions, etc.). We then evaluate one classical object-centric method across several generalization scenarios and compare its results against several state-of-the-art hollistic representations. Our results exhibit that existing methods are prone to failure in difficult scenarios involving complex scene structures, whereas object-centric methods help overcome these challenges. 

---
# Conversational Intent-Driven GraphRAG: Enhancing Multi-Turn Dialogue Systems through Adaptive Dual-Retrieval of Flow Patterns and Context Semantics 

**Authors**: Ziqi Zhu, Tao Hu, Honglong Zhang, Dan Yang, HanGeng Chen, Mengran Zhang, Xilun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19385)  

**Abstract**: We present CID-GraphRAG (Conversational Intent-Driven Graph Retrieval Augmented Generation), a novel framework that addresses the limitations of existing dialogue systems in maintaining both contextual coherence and goal-oriented progression in multi-turn customer service conversations. Unlike traditional RAG systems that rely solely on semantic similarity (Conversation RAG) or standard knowledge graphs (GraphRAG), CID-GraphRAG constructs dynamic intent transition graphs from goal achieved historical dialogues and implements a dual-retrieval mechanism that adaptively balances intent-based graph traversal with semantic search. This approach enables the system to simultaneously leverage both conversional intent flow patterns and contextual semantics, significantly improving retrieval quality and response quality. In extensive experiments on real-world customer service dialogues, we employ both automatic metrics and LLM-as-judge assessments, demonstrating that CID-GraphRAG significantly outperforms both semantic-based Conversation RAG and intent-based GraphRAG baselines across all evaluation criteria. Quantitatively, CID-GraphRAG demonstrates substantial improvements over Conversation RAG across automatic metrics, with relative gains of 11% in BLEU, 5% in ROUGE-L, 6% in METEOR, and most notably, a 58% improvement in response quality according to LLM-as-judge evaluations. These results demonstrate that the integration of intent transition structures with semantic retrieval creates a synergistic effect that neither approach achieves independently, establishing CID-GraphRAG as an effective framework for addressing the challenges of maintaining contextual coherence and goal-oriented progression in knowledge-intensive multi-turn dialogues. 

---
# Evolutionary Level Repair 

**Authors**: Debosmita Bhaumik, Julian Togelius, Georgios N. Yannakakis, Ahmed Khalifa  

**Link**: [PDF](https://arxiv.org/pdf/2506.19359)  

**Abstract**: We address the problem of game level repair, which consists of taking a designed but non-functional game level and making it functional. This might consist of ensuring the completeness of the level, reachability of objects, or other performance characteristics. The repair problem may also be constrained in that it can only make a small number of changes to the level. We investigate search-based solutions to the level repair problem, particularly using evolutionary and quality-diversity algorithms, with good results. This level repair method is applied to levels generated using a machine learning-based procedural content generation (PCGML) method that generates stylistically appropriate but frequently broken levels. This combination of PCGML for generation and search-based methods for repair shows great promise as a hybrid procedural content generation (PCG) method. 

---
# FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring 

**Authors**: Hyein Seo, Taewook Hwang, Yohan Lee, sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2506.19325)  

**Abstract**: In English education tutoring, teacher feedback is essential for guiding students. Recently, AI-based tutoring systems have emerged to assist teachers; however, these systems require high-quality and large-scale teacher feedback data, which is both time-consuming and costly to generate manually. In this study, we propose FEAT, a cost-effective framework for generating teacher feedback, and have constructed three complementary datasets: (1) DIRECT-Manual (DM), where both humans and large language models (LLMs) collaboratively generate high-quality teacher feedback, albeit at a higher cost; (2) DIRECT-Generated (DG), an LLM-only generated, cost-effective dataset with lower quality;, and (3) DIRECT-Augmented (DA), primarily based on DG with a small portion of DM added to enhance quality while maintaining cost-efficiency. Experimental results showed that incorporating a small portion of DM (5-10%) into DG leads to superior performance compared to using 100% DM alone. 

---
# Skywork-SWE: Unveiling Data Scaling Laws for Software Engineering in LLMs 

**Authors**: Liang Zeng, Yongcong Li, Yuzhen Xiao, Changshi Li, Chris Yuhao Liu, Rui Yan, Tianwen Wei, Jujie He, Xuchen Song, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.19290)  

**Abstract**: Software engineering (SWE) has recently emerged as a crucial testbed for next-generation LLM agents, demanding inherent capabilities in two critical dimensions: sustained iterative problem-solving (e.g., >50 interaction rounds) and long-context dependency resolution (e.g., >32k tokens). However, the data curation process in SWE remains notoriously time-consuming, as it heavily relies on manual annotation for code file filtering and the setup of dedicated runtime environments to execute and validate unit tests. Consequently, most existing datasets are limited to only a few thousand GitHub-sourced instances. To this end, we propose an incremental, automated data-curation pipeline that systematically scales both the volume and diversity of SWE datasets. Our dataset comprises 10,169 real-world Python task instances from 2,531 distinct GitHub repositories, each accompanied by a task specified in natural language and a dedicated runtime-environment image for automated unit-test validation. We have carefully curated over 8,000 successfully runtime-validated training trajectories from our proposed SWE dataset. When fine-tuning the Skywork-SWE model on these trajectories, we uncover a striking data scaling phenomenon: the trained model's performance for software engineering capabilities in LLMs continues to improve as the data size increases, showing no signs of saturation. Notably, our Skywork-SWE model achieves 38.0% pass@1 accuracy on the SWE-bench Verified benchmark without using verifiers or multiple rollouts, establishing a new state-of-the-art (SOTA) among the Qwen2.5-Coder-32B-based LLMs built on the OpenHands agent framework. Furthermore, with the incorporation of test-time scaling techniques, the performance further improves to 47.0% accuracy, surpassing the previous SOTA results for sub-32B parameter models. We release the Skywork-SWE-32B model checkpoint to accelerate future research. 

---
# Emotion Detection on User Front-Facing App Interfaces for Enhanced Schedule Optimization: A Machine Learning Approach 

**Authors**: Feiting Yang, Antoine Moevus, Steve Lévesque  

**Link**: [PDF](https://arxiv.org/pdf/2506.19280)  

**Abstract**: Human-Computer Interaction (HCI) has evolved significantly to incorporate emotion recognition capabilities, creating unprecedented opportunities for adaptive and personalized user experiences. This paper explores the integration of emotion detection into calendar applications, enabling user interfaces to dynamically respond to users' emotional states and stress levels, thereby enhancing both productivity and engagement. We present and evaluate two complementary approaches to emotion detection: a biometric-based method utilizing heart rate (HR) data extracted from electrocardiogram (ECG) signals processed through Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) neural networks to predict the emotional dimensions of Valence, Arousal, and Dominance; and a behavioral method analyzing computer activity through multiple machine learning models to classify emotions based on fine-grained user interactions such as mouse movements, clicks, and keystroke patterns. Our comparative analysis, from real-world datasets, reveals that while both approaches demonstrate effectiveness, the computer activity-based method delivers superior consistency and accuracy, particularly for mouse-related interactions, which achieved approximately 90\% accuracy. Furthermore, GRU networks outperformed LSTM models in the biometric approach, with Valence prediction reaching 84.38\% accuracy. 

---
# RecLLM-R1: A Two-Stage Training Paradigm with Reinforcement Learning and Chain-of-Thought v1 

**Authors**: Yu Xie, Xingkai Ren, Ying Qi, Yao Hu, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19235)  

**Abstract**: Traditional recommendation systems often grapple with "filter bubbles", underutilization of external knowledge, and a disconnect between model optimization and business policy iteration. To address these limitations, this paper introduces RecLLM-R1, a novel recommendation framework leveraging Large Language Models (LLMs) and drawing inspiration from the DeepSeek R1 methodology. The framework initiates by transforming user profiles, historical interactions, and multi-faceted item attributes into LLM-interpretable natural language prompts through a carefully engineered data construction process. Subsequently, a two-stage training paradigm is employed: the initial stage involves Supervised Fine-Tuning (SFT) to imbue the LLM with fundamental recommendation capabilities. The subsequent stage utilizes Group Relative Policy Optimization (GRPO), a reinforcement learning technique, augmented with a Chain-of-Thought (CoT) mechanism. This stage guides the model through multi-step reasoning and holistic decision-making via a flexibly defined reward function, aiming to concurrently optimize recommendation accuracy, diversity, and other bespoke business objectives. Empirical evaluations on a real-world user behavior dataset from a large-scale social media platform demonstrate that RecLLM-R1 significantly surpasses existing baseline methods across a spectrum of evaluation metrics, including accuracy, diversity, and novelty. It effectively mitigates the filter bubble effect and presents a promising avenue for the integrated optimization of recommendation models and policies under intricate business goals. 

---
# GBGC: Efficient and Adaptive Graph Coarsening via Granular-ball Computing 

**Authors**: Shuyin Xia, Guan Wang, Gaojie Xu, Sen Zhao, Guoyin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19224)  

**Abstract**: The objective of graph coarsening is to generate smaller, more manageable graphs while preserving key information of the original graph. Previous work were mainly based on the perspective of spectrum-preserving, using some predefined coarsening rules to make the eigenvalues of the Laplacian matrix of the original graph and the coarsened graph match as much as possible. However, they largely overlooked the fact that the original graph is composed of subregions at different levels of granularity, where highly connected and similar nodes should be more inclined to be aggregated together as nodes in the coarsened graph. By combining the multi-granularity characteristics of the graph structure, we can generate coarsened graph at the optimal granularity. To this end, inspired by the application of granular-ball computing in multi-granularity, we propose a new multi-granularity, efficient, and adaptive coarsening method via granular-ball (GBGC), which significantly improves the coarsening results and efficiency. Specifically, GBGC introduces an adaptive granular-ball graph refinement mechanism, which adaptively splits the original graph from coarse to fine into granular-balls of different sizes and optimal granularity, and constructs the coarsened graph using these granular-balls as supernodes. In addition, compared with other state-of-the-art graph coarsening methods, the processing speed of this method can be increased by tens to hundreds of times and has lower time complexity. The accuracy of GBGC is almost always higher than that of the original graph due to the good robustness and generalization of the granular-ball computing, so it has the potential to become a standard graph data preprocessing method. 

---
# Bayesian Evolutionary Swarm Architecture: A Formal Epistemic System Grounded in Truth-Based Competition 

**Authors**: Craig Steven Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.19191)  

**Abstract**: We introduce a mathematically rigorous framework for an artificial intelligence system composed of probabilistic agents evolving through structured competition and belief revision. The architecture, grounded in Bayesian inference, measure theory, and population dynamics, defines agent fitness as a function of alignment with a fixed external oracle representing ground truth. Agents compete in a discrete-time environment, adjusting posterior beliefs through observed outcomes, with higher-rated agents reproducing and lower-rated agents undergoing extinction. Ratings are updated via pairwise truth-aligned utility comparisons, and belief updates preserve measurable consistency and stochastic convergence. We introduce hash-based cryptographic identity commitments to ensure traceability, alongside causal inference operators using do-calculus. Formal theorems on convergence, robustness, and evolutionary stability are provided. The system establishes truth as an evolutionary attractor, demonstrating that verifiable knowledge arises from adversarial epistemic pressure within a computable, self-regulating swarm. 

---
# Spiritual-LLM : Gita Inspired Mental Health Therapy In the Era of LLMs 

**Authors**: Janak Kapuriya, Aman Singh, Jainendra Shukla, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2506.19185)  

**Abstract**: Traditional mental health support systems often generate responses based solely on the user's current emotion and situations, resulting in superficial interventions that fail to address deeper emotional needs. This study introduces a novel framework by integrating spiritual wisdom from the Bhagavad Gita with advanced large language model GPT-4o to enhance emotional well-being. We present the GITes (Gita Integrated Therapy for Emotional Support) dataset, which enhances the existing ExTES mental health dataset by including 10,729 spiritually guided responses generated by GPT-4o and evaluated by domain experts. We benchmark GITes against 12 state-of-the-art LLMs, including both mental health specific and general purpose models. To evaluate spiritual relevance in generated responses beyond what conventional n-gram based metrics capture, we propose a novel Spiritual Insight metric and automate assessment via an LLM as jury framework using chain-of-thought prompting. Integrating spiritual guidance into AI driven support enhances both NLP and spiritual metrics for the best performing LLM Phi3-Mini 3.2B Instruct, achieving improvements of 122.71% in ROUGE, 126.53% in METEOR, 8.15% in BERT score, 15.92% in Spiritual Insight, 18.61% in Sufficiency and 13.22% in Relevance compared to its zero-shot counterpart. While these results reflect substantial improvements across automated empathy and spirituality metrics, further validation in real world patient populations remains a necessary step. Our findings indicate a strong potential for AI systems enriched with spiritual guidance to enhance user satisfaction and perceived support outcomes. The code and dataset will be publicly available to advance further research in this emerging area. 

---
# Baba is LLM: Reasoning in a Game with Dynamic Rules 

**Authors**: Fien van Wetten, Aske Plaat, Max van Duijn  

**Link**: [PDF](https://arxiv.org/pdf/2506.19095)  

**Abstract**: Large language models (LLMs) are known to perform well on language tasks, but struggle with reasoning tasks. This paper explores the ability of LLMs to play the 2D puzzle game Baba is You, in which players manipulate rules by rearranging text blocks that define object properties. Given that this rule-manipulation relies on language abilities and reasoning, it is a compelling challenge for LLMs. Six LLMs are evaluated using different prompt types, including (1) simple, (2) rule-extended and (3) action-extended prompts. In addition, two models (Mistral, OLMo) are finetuned using textual and structural data from the game. Results show that while larger models (particularly GPT-4o) perform better in reasoning and puzzle solving, smaller unadapted models struggle to recognize game mechanics or apply rule changes. Finetuning improves the ability to analyze the game levels, but does not significantly improve solution formulation. We conclude that even for state-of-the-art and finetuned LLMs, reasoning about dynamic rule changes is difficult (specifically, understanding the use-mention distinction). The results provide insights into the applicability of LLMs to complex problem-solving tasks and highlight the suitability of games with dynamically changing rules for testing reasoning and reflection by LLMs. 

---
# From Rows to Yields: How Foundation Models for Tabular Data Simplify Crop Yield Prediction 

**Authors**: Filip Sabo, Michele Meroni, Maria Piles, Martin Claverie, Fanie Ferreira, Elna Van Den Berg, Francesco Collivignarelli, Felix Rembold  

**Link**: [PDF](https://arxiv.org/pdf/2506.19046)  

**Abstract**: We present an application of a foundation model for small- to medium-sized tabular data (TabPFN), to sub-national yield forecasting task in South Africa. TabPFN has recently demonstrated superior performance compared to traditional machine learning (ML) models in various regression and classification tasks. We used the dekadal (10-days) time series of Earth Observation (EO; FAPAR and soil moisture) and gridded weather data (air temperature, precipitation and radiation) to forecast the yield of summer crops at the sub-national level. The crop yield data was available for 23 years and for up to 8 provinces. Covariate variables for TabPFN (i.e., EO and weather) were extracted by region and aggregated at a monthly scale. We benchmarked the results of the TabPFN against six ML models and three baseline models. Leave-one-year-out cross-validation experiment setting was used in order to ensure the assessment of the models capacity to forecast an unseen year. Results showed that TabPFN and ML models exhibit comparable accuracy, outperforming the baselines. Nonetheless, TabPFN demonstrated superior practical utility due to its significantly faster tuning time and reduced requirement for feature engineering. This renders TabPFN a more viable option for real-world operation yield forecasting applications, where efficiency and ease of implementation are paramount. 

---
# A Comment On "The Illusion of Thinking": Reframing the Reasoning Cliff as an Agentic Gap 

**Authors**: Sheraz Khan, Subha Madhavan, Kannan Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18957)  

**Abstract**: The recent work by Shojaee et al. (2025), titled The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity, presents a compelling empirical finding, a reasoning cliff, where the performance of Large Reasoning Models (LRMs) collapses beyond a specific complexity threshold, which the authors posit as an intrinsic scaling limitation of Chain-of-Thought (CoT) reasoning. This commentary, while acknowledging the study's methodological rigor, contends that this conclusion is confounded by experimental artifacts. We argue that the observed failure is not evidence of a fundamental cognitive boundary, but rather a predictable outcome of system-level constraints in the static, text-only evaluation paradigm, including tool use restrictions, context window recall issues, the absence of crucial cognitive baselines, inadequate statistical reporting, and output generation limits. We reframe this performance collapse through the lens of an agentic gap, asserting that the models are not failing at reasoning, but at execution within a profoundly restrictive interface. We empirically substantiate this critique by demonstrating a striking reversal. A model, initially declaring a puzzle impossible when confined to text-only generation, now employs agentic tools to not only solve it but also master variations of complexity far beyond the reasoning cliff it previously failed to surmount. Additionally, our empirical analysis of tool-enabled models like o4-mini and GPT-4o reveals a hierarchy of agentic reasoning, from simple procedural execution to complex meta-cognitive self-correction, which has significant implications for how we define and measure machine intelligence. The illusion of thinking attributed to LRMs is less a reasoning deficit and more a consequence of an otherwise capable mind lacking the tools for action. 

---
# Do LLMs Know When to Flip a Coin? Strategic Randomization through Reasoning and Experience 

**Authors**: Lingyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18928)  

**Abstract**: Strategic randomization is a key principle in game theory, yet it remains underexplored in large language models (LLMs). Prior work often conflates the cognitive decision to randomize with the mechanical generation of randomness, leading to incomplete evaluations. To address this, we propose a novel zero-sum game inspired by the Tian Ji Horse Race, where the Nash equilibrium corresponds to a maximal entropy strategy. The game's complexity masks this property from untrained humans and underdeveloped LLMs. We evaluate five LLMs across prompt styles -- framed, neutral, and hinted -- using competitive multi-tournament gameplay with system-provided random choices, isolating the decision to randomize. Results show that weaker models remain deterministic regardless of prompts, while stronger models exhibit increased randomization under explicit hints. When facing weaker models, strong LLMs adopt deterministic strategies to exploit biases, but converge toward equilibrium play when facing peers. Through win/loss outcomes and Bayes factor analysis, we demonstrate meaningful variation in LLMs' strategic reasoning capabilities, highlighting opportunities for improvement in abstract reasoning and adaptive learning. We make our implementation publicly available at this https URL to ensure full reproducibility. 

---
# Signal Use and Emergent Cooperation 

**Authors**: Michael Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.18920)  

**Abstract**: In this work, we investigate how autonomous agents, organized into tribes, learn to use communication signals to coordinate their activities and enhance their collective efficiency. Using the NEC-DAC (Neurally Encoded Culture - Distributed Autonomous Communicators) system, where each agent is equipped with its own neural network for decision-making, we demonstrate how these agents develop a shared behavioral system -- akin to a culture -- through learning and signalling. Our research focuses on the self-organization of culture within these tribes of agents and how varying communication strategies impact their fitness and cooperation. By analyzing different social structures, such as authority hierarchies, we show that the culture of cooperation significantly influences the tribe's performance. Furthermore, we explore how signals not only facilitate the emergence of culture but also enable its transmission across generations of agents. Additionally, we examine the benefits of coordinating behavior and signaling within individual agents' neural networks. 

---
# Radial Attention: $O(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation 

**Authors**: Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, Maneesh Agrawala, Ion Stoica, Kurt Keutzer, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.19852)  

**Abstract**: Recent advances in diffusion models have enabled high-quality video generation, but the additional temporal dimension significantly increases computational costs, making training and inference on long videos prohibitively expensive. In this paper, we identify a phenomenon we term Spatiotemporal Energy Decay in video diffusion models: post-softmax attention scores diminish as spatial and temporal distance between tokens increase, akin to the physical decay of signal or waves over space and time in nature. Motivated by this, we propose Radial Attention, a scalable sparse attention mechanism with $O(n \log n)$ complexity that translates energy decay into exponentially decaying compute density, which is significantly more efficient than standard $O(n^2)$ dense attention and more expressive than linear attention. Specifically, Radial Attention employs a simple, static attention mask where each token attends to spatially nearby tokens, with the attention window size shrinking with temporal distance. Moreover, it allows pre-trained video diffusion models to extend their generation length with efficient LoRA-based fine-tuning. Extensive experiments show that Radial Attention maintains video quality across Wan2.1-14B, HunyuanVideo, and Mochi 1, achieving up to a 1.9$\times$ speedup over the original dense attention. With minimal tuning, it enables video generation up to 4$\times$ longer while reducing training costs by up to 4.4$\times$ compared to direct fine-tuning and accelerating inference by up to 3.7$\times$ compared to dense attention inference. 

---
# Orthogonal Finetuning Made Scalable 

**Authors**: Zeju Qiu, Weiyang Liu, Adrian Weller, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2506.19847)  

**Abstract**: Orthogonal finetuning (OFT) offers highly parameter-efficient adaptation while preventing catastrophic forgetting, but its high runtime and memory demands limit practical deployment. We identify the core computational bottleneck in OFT as its weight-centric implementation, which relies on costly matrix-matrix multiplications with cubic complexity. To overcome this, we propose OFTv2, an input-centric reformulation that instead uses matrix-vector multiplications (i.e., matrix-free computation), reducing the computational cost to quadratic. We further introduce the Cayley-Neumann parameterization, an efficient orthogonal parameterization that approximates the matrix inversion in Cayley transform via a truncated Neumann series. These modifications allow OFTv2 to achieve up to 10x faster training and 3x lower GPU memory usage without compromising performance. In addition, we extend OFTv2 to support finetuning quantized foundation models and show that it outperforms the popular QLoRA in training stability, efficiency, and memory usage. 

---
# Improving Progressive Generation with Decomposable Flow Matching 

**Authors**: Moayed Haji-Ali, Willi Menapace, Ivan Skorokhodov, Arpit Sahni, Sergey Tulyakov, Vicente Ordonez, Aliaksandr Siarohin  

**Link**: [PDF](https://arxiv.org/pdf/2506.19839)  

**Abstract**: Generating high-dimensional visual modalities is a computationally intensive task. A common solution is progressive generation, where the outputs are synthesized in a coarse-to-fine spectral autoregressive manner. While diffusion models benefit from the coarse-to-fine nature of denoising, explicit multi-stage architectures are rarely adopted. These architectures have increased the complexity of the overall approach, introducing the need for a custom diffusion formulation, decomposition-dependent stage transitions, add-hoc samplers, or a model cascade. Our contribution, Decomposable Flow Matching (DFM), is a simple and effective framework for the progressive generation of visual media. DFM applies Flow Matching independently at each level of a user-defined multi-scale representation (such as Laplacian pyramid). As shown by our experiments, our approach improves visual quality for both images and videos, featuring superior results compared to prior multistage frameworks. On Imagenet-1k 512px, DFM achieves 35.2% improvements in FDD scores over the base architecture and 26.4% over the best-performing baseline, under the same training compute. When applied to finetuning of large models, such as FLUX, DFM shows faster convergence speed to the training distribution. Crucially, all these advantages are achieved with a single model, architectural simplicity, and minimal modifications to existing training pipelines. 

---
# A standard transformer and attention with linear biases for molecular conformer generation 

**Authors**: Viatcheslav Gurev, Timothy Rumbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.19834)  

**Abstract**: Sampling low-energy molecular conformations, spatial arrangements of atoms in a molecule, is a critical task for many different calculations performed in the drug discovery and optimization process. Numerous specialized equivariant networks have been designed to generate molecular conformations from 2D molecular graphs. Recently, non-equivariant transformer models have emerged as a viable alternative due to their capability to scale to improve generalization. However, the concern has been that non-equivariant models require a large model size to compensate the lack of equivariant bias. In this paper, we demonstrate that a well-chosen positional encoding effectively addresses these size limitations. A standard transformer model incorporating relative positional encoding for molecular graphs when scaled to 25 million parameters surpasses the current state-of-the-art non-equivariant base model with 64 million parameters on the GEOM-DRUGS benchmark. We implemented relative positional encoding as a negative attention bias that linearly increases with the shortest path distances between graph nodes at varying slopes for different attention heads, similar to ALiBi, a widely adopted relative positional encoding technique in the NLP domain. This architecture has the potential to serve as a foundation for a novel class of generative models for molecular conformations. 

---
# Persona Features Control Emergent Misalignment 

**Authors**: Miles Wang, Tom Dupré la Tour, Olivia Watkins, Alex Makelov, Ryan A. Chi, Samuel Miserendino, Johannes Heidecke, Tejal Patwardhan, Dan Mossing  

**Link**: [PDF](https://arxiv.org/pdf/2506.19823)  

**Abstract**: Understanding how language models generalize behaviors from their training to a broader deployment distribution is an important problem in AI safety. Betley et al. discovered that fine-tuning GPT-4o on intentionally insecure code causes "emergent misalignment," where models give stereotypically malicious responses to unrelated prompts. We extend this work, demonstrating emergent misalignment across diverse conditions, including reinforcement learning on reasoning models, fine-tuning on various synthetic datasets, and in models without safety training. To investigate the mechanisms behind this generalized misalignment, we apply a "model diffing" approach using sparse autoencoders to compare internal model representations before and after fine-tuning. This approach reveals several "misaligned persona" features in activation space, including a toxic persona feature which most strongly controls emergent misalignment and can be used to predict whether a model will exhibit such behavior. Additionally, we investigate mitigation strategies, discovering that fine-tuning an emergently misaligned model on just a few hundred benign samples efficiently restores alignment. 

---
# Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study 

**Authors**: Yuqi Zhu, Yi Zhong, Jintian Zhang, Ziheng Zhang, Shuofei Qiao, Yujie Luo, Lun Du, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19794)  

**Abstract**: Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. 

---
# Alleviating User-Sensitive bias with Fair Generative Sequential Recommendation Model 

**Authors**: Yang Liu, Feng Wu, Xuefang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19777)  

**Abstract**: Recommendation fairness has recently attracted much attention. In the real world, recommendation systems are driven by user behavior, and since users with the same sensitive feature (e.g., gender and age) tend to have the same patterns, recommendation models can easily capture the strong correlation preference of sensitive features and thus cause recommendation unfairness. Diffusion model (DM) as a new generative model paradigm has achieved great success in recommendation systems. DM's ability to model uncertainty and represent diversity, and its modeling mechanism has a high degree of adaptability with the real-world recommendation process with bias. Therefore, we use DM to effectively model the fairness of recommendation and enhance the diversity. This paper proposes a FairGENerative sequential Recommendation model based on DM, FairGENRec. In the training phase, we inject random noise into the original distribution under the guidance of the sensitive feature recognition model, and a sequential denoise model is designed for the reverse reconstruction of items. Simultaneously, recommendation fairness modeling is completed by injecting multi-interests representational information that eliminates the bias of sensitive user features into the generated results. In the inference phase, the model obtains the noise in the form of noise addition by using the history interactions which is followed by reverse iteration to reconstruct the target item representation. Finally, our extensive experiments on three datasets demonstrate the dual enhancement effect of FairGENRec on accuracy and fairness, while the statistical analysis of the cases visualizes the degree of improvement on the fairness of the recommendation. 

---
# Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation 

**Authors**: Jun Wang, Xijuan Zeng, Chunyu Qiang, Ruilong Chen, Shiyao Wang, Le Wang, Wangjing Zhou, Pengfei Cai, Jiahui Zhao, Nan Li, Zihan Li, Yuzhe Liang, Xiaopeng Wang, Haorui Zheng, Ming Wen, Kang Yin, Yiran Wang, Nan Li, Feng Deng, Liang Dong, Chen Zhang, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2506.19774)  

**Abstract**: We propose Kling-Foley, a large-scale multimodal Video-to-Audio generation model that synthesizes high-quality audio synchronized with video content. In Kling-Foley, we introduce multimodal diffusion transformers to model the interactions between video, audio, and text modalities, and combine it with a visual semantic representation module and an audio-visual synchronization module to enhance alignment capabilities. Specifically, these modules align video conditions with latent audio elements at the frame level, thereby improving semantic alignment and audio-visual synchronization. Together with text conditions, this integrated approach enables precise generation of video-matching sound effects. In addition, we propose a universal latent audio codec that can achieve high-quality modeling in various scenarios such as sound effects, speech, singing, and music. We employ a stereo rendering method that imbues synthesized audio with a spatial presence. At the same time, in order to make up for the incomplete types and annotations of the open-source benchmark, we also open-source an industrial-level benchmark Kling-Audio-Eval. Our experiments show that Kling-Foley trained with the flow matching objective achieves new audio-visual SOTA performance among public models in terms of distribution matching, semantic alignment, temporal alignment and audio quality. 

---
# A Survey of Multi-sensor Fusion Perception for Embodied AI: Background, Methods, Challenges and Prospects 

**Authors**: Shulan Ruan, Rongwei Wang, Xuchen Shen, Huijie Liu, Baihui Xiao, Jun Shi, Kun Zhang, Zhenya Huang, Yu Liu, Enhong Chen, You He  

**Link**: [PDF](https://arxiv.org/pdf/2506.19769)  

**Abstract**: Multi-sensor fusion perception (MSFP) is a key technology for embodied AI, which can serve a variety of downstream tasks (e.g., 3D object detection and semantic segmentation) and application scenarios (e.g., autonomous driving and swarm robotics). Recently, impressive achievements on AI-based MSFP methods have been reviewed in relevant surveys. However, we observe that the existing surveys have some limitations after a rigorous and detailed investigation. For one thing, most surveys are oriented to a single task or research field, such as 3D object detection or autonomous driving. Therefore, researchers in other related tasks often find it difficult to benefit directly. For another, most surveys only introduce MSFP from a single perspective of multi-modal fusion, while lacking consideration of the diversity of MSFP methods, such as multi-view fusion and time-series fusion. To this end, in this paper, we hope to organize MSFP research from a task-agnostic perspective, where methods are reported from various technical views. Specifically, we first introduce the background of MSFP. Next, we review multi-modal and multi-agent fusion methods. A step further, time-series fusion methods are analyzed. In the era of LLM, we also investigate multimodal LLM fusion methods. Finally, we discuss open challenges and future directions for MSFP. We hope this survey can help researchers understand the important progress in MSFP and provide possible insights for future research. 

---
# SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning 

**Authors**: Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19767)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in reasoning tasks, yet the optimal integration of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) remains a fundamental challenge. Through comprehensive analysis of token distributions, learning dynamics, and integration mechanisms from entropy-based perspectives, we reveal key differences between these paradigms: SFT induces coarse-grained global changes to LLM policy distributions, while RL performs fine-grained selective optimizations, with entropy serving as a critical indicator of training effectiveness. Building on these observations, we propose Supervised Reinforcement Fine-Tuning (SRFT), a single-stage method that unifies both fine-tuning paradigms through entropy-aware weighting mechanisms. Our approach simultaneously applies SFT and RL to directly optimize the LLM using demonstrations and self-exploration rollouts rather than through two-stage sequential methods. Extensive experiments show that SRFT achieves 59.1% average accuracy, outperforming zero-RL methods by 9.0% on five mathematical reasoning benchmarks and 10.9% on three out-of-distribution benchmarks. 

---
# Cross-regularization: Adaptive Model Complexity through Validation Gradients 

**Authors**: Carlos Stein Brito  

**Link**: [PDF](https://arxiv.org/pdf/2506.19755)  

**Abstract**: Model regularization requires extensive manual tuning to balance complexity against overfitting. Cross-regularization resolves this tradeoff by directly adapting regularization parameters through validation gradients during training. The method splits parameter optimization - training data guides feature learning while validation data shapes complexity controls - converging provably to cross-validation optima. When implemented through noise injection in neural networks, this approach reveals striking patterns: unexpectedly high noise tolerance and architecture-specific regularization that emerges organically during training. Beyond complexity control, the framework integrates seamlessly with data augmentation, uncertainty calibration and growing datasets while maintaining single-run efficiency through a simple gradient-based approach. 

---
# Arabic Dialect Classification using RNNs, Transformers, and Large Language Models: A Comparative Analysis 

**Authors**: Omar A.Essameldin, Ali O.Elbeih, Wael H.Gomaa, Wael F.Elsersy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19753)  

**Abstract**: The Arabic language is among the most popular languages in the world with a huge variety of dialects spoken in 22 countries. In this study, we address the problem of classifying 18 Arabic dialects of the QADI dataset of Arabic tweets. RNN models, Transformer models, and large language models (LLMs) via prompt engineering are created and tested. Among these, MARBERTv2 performed best with 65% accuracy and 64% F1-score. Through the use of state-of-the-art preprocessing techniques and the latest NLP models, this paper identifies the most significant linguistic issues in Arabic dialect identification. The results corroborate applications like personalized chatbots that respond in users' dialects, social media monitoring, and greater accessibility for Arabic communities. 

---
# NeRF-based CBCT Reconstruction needs Normalization and Initialization 

**Authors**: Zhuowei Xu, Han Li, Dai Sun, Zhicheng Li, Yujia Li, Qingpeng Kong, Zhiwei Cheng, Nassir Navab, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.19742)  

**Abstract**: Cone Beam Computed Tomography (CBCT) is widely used in medical imaging. However, the limited number and intensity of X-ray projections make reconstruction an ill-posed problem with severe artifacts. NeRF-based methods have achieved great success in this task. However, they suffer from a local-global training mismatch between their two key components: the hash encoder and the neural network. Specifically, in each training step, only a subset of the hash encoder's parameters is used (local sparse), whereas all parameters in the neural network participate (global dense). Consequently, hash features generated in each step are highly misaligned, as they come from different subsets of the hash encoder. These misalignments from different training steps are then fed into the neural network, causing repeated inconsistent global updates in training, which leads to unstable training, slower convergence, and degraded reconstruction quality. Aiming to alleviate the impact of this local-global optimization mismatch, we introduce a Normalized Hash Encoder, which enhances feature consistency and mitigates the mismatch. Additionally, we propose a Mapping Consistency Initialization(MCI) strategy that initializes the neural network before training by leveraging the global mapping property from a well-trained model. The initialized neural network exhibits improved stability during early training, enabling faster convergence and enhanced reconstruction performance. Our method is simple yet effective, requiring only a few lines of code while substantially improving training efficiency on 128 CT cases collected from 4 different datasets, covering 7 distinct anatomical regions. 

---
# Who Does What in Deep Learning? Multidimensional Game-Theoretic Attribution of Function of Neural Units 

**Authors**: Shrey Dixit, Kayson Fakhar, Fatemeh Hadaeghi, Patrick Mineault, Konrad P. Kording, Claus C. Hilgetag  

**Link**: [PDF](https://arxiv.org/pdf/2506.19732)  

**Abstract**: Neural networks now generate text, images, and speech with billions of parameters, producing a need to know how each neural unit contributes to these high-dimensional outputs. Existing explainable-AI methods, such as SHAP, attribute importance to inputs, but cannot quantify the contributions of neural units across thousands of output pixels, tokens, or logits. Here we close that gap with Multiperturbation Shapley-value Analysis (MSA), a model-agnostic game-theoretic framework. By systematically lesioning combinations of units, MSA yields Shapley Modes, unit-wise contribution maps that share the exact dimensionality of the model's output. We apply MSA across scales, from multi-layer perceptrons to the 56-billion-parameter Mixtral-8x7B and Generative Adversarial Networks (GAN). The approach demonstrates how regularisation concentrates computation in a few hubs, exposes language-specific experts inside the LLM, and reveals an inverted pixel-generation hierarchy in GANs. Together, these results showcase MSA as a powerful approach for interpreting, editing, and compressing deep neural networks. 

---
# Geometric-Aware Variational Inference: Robust and Adaptive Regularization with Directional Weight Uncertainty 

**Authors**: Carlos Stein Brito  

**Link**: [PDF](https://arxiv.org/pdf/2506.19726)  

**Abstract**: Deep neural networks require principled uncertainty quantification, yet existing variational inference methods often employ isotropic Gaussian approximations in weight space that poorly match the network's inherent geometry. We address this mismatch by introducing Concentration-Adapted Perturbations (CAP), a variational framework that models weight uncertainties directly on the unit hypersphere using von Mises-Fisher distributions. Building on recent work in radial-directional posterior decompositions and spherical weight constraints, CAP provides the first complete theoretical framework connecting directional statistics to practical noise regularization in neural networks. Our key contribution is an analytical derivation linking vMF concentration parameters to activation noise variance, enabling each layer to learn its optimal uncertainty level through a novel closed-form KL divergence regularizer. In experiments on CIFAR-10, CAP significantly improves model calibration - reducing Expected Calibration Error by 5.6x - while providing interpretable layer-wise uncertainty profiles. CAP requires minimal computational overhead and integrates seamlessly into standard architectures, offering a theoretically grounded yet practical approach to uncertainty quantification in deep learning. 

---
# Uncovering Conceptual Blindspots in Generative Image Models Using Sparse Autoencoders 

**Authors**: Matyas Bohacek, Thomas Fel, Maneesh Agrawala, Ekdeep Singh Lubana  

**Link**: [PDF](https://arxiv.org/pdf/2506.19708)  

**Abstract**: Despite their impressive performance, generative image models trained on large-scale datasets frequently fail to produce images with seemingly simple concepts -- e.g., human hands or objects appearing in groups of four -- that are reasonably expected to appear in the training data. These failure modes have largely been documented anecdotally, leaving open the question of whether they reflect idiosyncratic anomalies or more structural limitations of these models. To address this, we introduce a systematic approach for identifying and characterizing "conceptual blindspots" -- concepts present in the training data but absent or misrepresented in a model's generations. Our method leverages sparse autoencoders (SAEs) to extract interpretable concept embeddings, enabling a quantitative comparison of concept prevalence between real and generated images. We train an archetypal SAE (RA-SAE) on DINOv2 features with 32,000 concepts -- the largest such SAE to date -- enabling fine-grained analysis of conceptual disparities. Applied to four popular generative models (Stable Diffusion 1.5/2.1, PixArt, and Kandinsky), our approach reveals specific suppressed blindspots (e.g., bird feeders, DVD discs, and whitespaces on documents) and exaggerated blindspots (e.g., wood background texture and palm trees). At the individual datapoint level, we further isolate memorization artifacts -- instances where models reproduce highly specific visual templates seen during training. Overall, we propose a theoretically grounded framework for systematically identifying conceptual blindspots in generative models by assessing their conceptual fidelity with respect to the underlying data-generating process. 

---
# Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models 

**Authors**: Jungwoo Park, Taewhoo Lee, Chanwoong Yoon, Hyeon Hwang, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19697)  

**Abstract**: Extreme activation outliers in Large Language Models (LLMs) critically degrade quantization performance, hindering efficient on-device deployment. While channel-wise operations and adaptive gradient scaling are recognized causes, practical mitigation remains challenging. We introduce Outlier-Safe Pre-Training (OSP), a practical guideline that proactively prevents outlier formation rather than relying on post-hoc mitigation. OSP combines three key innovations: (1) the Muon optimizer, eliminating privileged bases while maintaining training efficiency; (2) Single-Scale RMSNorm, preventing channel-wise amplification; and (3) a learnable embedding projection, redistributing activation magnitudes originating from embedding matrices. We validate OSP by training a 1.4B-parameter model on 1 trillion tokens, which is the first production-scale LLM trained without such outliers. Under aggressive 4-bit quantization, our OSP model achieves a 35.7 average score across 10 benchmarks (compared to 26.5 for an Adam-trained model), with only a 2% training overhead. Remarkably, OSP models exhibit near-zero excess kurtosis (0.04) compared to extreme values (1818.56) in standard models, fundamentally altering LLM quantization behavior. Our work demonstrates that outliers are not inherent to LLMs but are consequences of training strategies, paving the way for more efficient LLM deployment. The source code and pretrained checkpoints are available at this https URL. 

---
# When Can We Reuse a Calibration Set for Multiple Conformal Predictions? 

**Authors**: A.A. Balinsky, A.D. Balinsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.19689)  

**Abstract**: Reliable uncertainty quantification is crucial for the trustworthiness of machine learning applications. Inductive Conformal Prediction (ICP) offers a distribution-free framework for generating prediction sets or intervals with user-specified confidence. However, standard ICP guarantees are marginal and typically require a fresh calibration set for each new prediction to maintain their validity. This paper addresses this practical limitation by demonstrating how e-conformal prediction, in conjunction with Hoeffding's inequality, can enable the repeated use of a single calibration set with a high probability of preserving the desired coverage. Through a case study on the CIFAR-10 dataset, we train a deep neural network and utilise a calibration set to estimate a Hoeffding correction. This correction allows us to apply a modified Markov's inequality, leading to the construction of prediction sets with quantifiable confidence. Our results illustrate the feasibility of maintaining provable performance in conformal prediction while enhancing its practicality by reducing the need for repeated calibration. The code for this work is publicly available. 

---
# Semantic Scene Graph for Ultrasound Image Explanation and Scanning Guidance 

**Authors**: Xuesong Li, Dianye Huang, Yameng Zhang, Nassir Navab, Zhongliang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19683)  

**Abstract**: Understanding medical ultrasound imaging remains a long-standing challenge due to significant visual variability caused by differences in imaging and acquisition parameters. Recent advancements in large language models (LLMs) have been used to automatically generate terminology-rich summaries orientated to clinicians with sufficient physiological knowledge. Nevertheless, the increasing demand for improved ultrasound interpretability and basic scanning guidance among non-expert users, e.g., in point-of-care settings, has not yet been explored. In this study, we first introduce the scene graph (SG) for ultrasound images to explain image content to ordinary and provide guidance for ultrasound scanning. The ultrasound SG is first computed using a transformer-based one-stage method, eliminating the need for explicit object detection. To generate a graspable image explanation for ordinary, the user query is then used to further refine the abstract SG representation through LLMs. Additionally, the predicted SG is explored for its potential in guiding ultrasound scanning toward missing anatomies within the current imaging view, assisting ordinary users in achieving more standardized and complete anatomical exploration. The effectiveness of this SG-based image explanation and scanning guidance has been validated on images from the left and right neck regions, including the carotid and thyroid, across five volunteers. The results demonstrate the potential of the method to maximally democratize ultrasound by enhancing its interpretability and usability for ordinaries. 

---
# Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager 

**Authors**: Lucie Galland, Catherine Pelachaud, Florian Pecune  

**Link**: [PDF](https://arxiv.org/pdf/2506.19652)  

**Abstract**: In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals. 

---
# The receptron is a nonlinear threshold logic gate with intrinsic multi-dimensional selective capabilities for analog inputs 

**Authors**: B. Paroli, F. Borghi, M.A.C. Potenza, P. Milani  

**Link**: [PDF](https://arxiv.org/pdf/2506.19642)  

**Abstract**: Threshold logic gates (TLGs) have been proposed as artificial counterparts of biological neurons with classification capabilities based on a linear predictor function combining a set of weights with the feature vector. The linearity of TLGs limits their classification capabilities requiring the use of networks for the accomplishment of complex tasks. A generalization of the TLG model called receptron, characterized by input-dependent weight functions allows for a significant enhancement of classification performances even with the use of a single unit. Here we formally demonstrate that a receptron, characterized by nonlinear input-dependent weight functions, exhibit intrinsic selective activation properties for analog inputs, when the input vector is within cubic domains in a 3D space. The proposed model can be extended to the n-dimensional case for multidimensional applications. Our results suggest that receptron-based networks can represent a new class of devices capable to manage a large number of analog inputs, for edge applications requiring high selectivity and classification capabilities without the burden of complex training. 

---
# Hierarchical Time Series Forecasting Via Latent Mean Encoding 

**Authors**: Alessandro Salatiello, Stefan Birr, Manuel Kunz  

**Link**: [PDF](https://arxiv.org/pdf/2506.19633)  

**Abstract**: Coherently forecasting the behaviour of a target variable across both coarse and fine temporal scales is crucial for profit-optimized decision-making in several business applications, and remains an open research problem in temporal hierarchical forecasting. Here, we propose a new hierarchical architecture that tackles this problem by leveraging modules that specialize in forecasting the different temporal aggregation levels of interest. The architecture, which learns to encode the average behaviour of the target variable within its hidden layers, makes accurate and coherent forecasts across the target temporal hierarchies. We validate our architecture on the challenging, real-world M5 dataset and show that it outperforms established methods, such as the TSMixer model. 

---
# Why Uncertainty Calibration Matters for Reliable Perturbation-based Explanations 

**Authors**: Thomas Decker, Volker Tresp, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2506.19630)  

**Abstract**: Perturbation-based explanations are widely utilized to enhance the transparency of modern machine-learning models. However, their reliability is often compromised by the unknown model behavior under the specific perturbations used. This paper investigates the relationship between uncertainty calibration - the alignment of model confidence with actual accuracy - and perturbation-based explanations. We show that models frequently produce unreliable probability estimates when subjected to explainability-specific perturbations and theoretically prove that this directly undermines explanation quality. To address this, we introduce ReCalX, a novel approach to recalibrate models for improved perturbation-based explanations while preserving their original predictions. Experiments on popular computer vision models demonstrate that our calibration strategy produces explanations that are more aligned with human perception and actual object locations. 

---
# VideoPCDNet: Video Parsing and Prediction with Phase Correlation Networks 

**Authors**: Noel José Rodrigues Vicente, Enrique Lehner, Angel Villar-Corrales, Jan Nogga, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2506.19621)  

**Abstract**: Understanding and predicting video content is essential for planning and reasoning in dynamic environments. Despite advancements, unsupervised learning of object representations and dynamics remains challenging. We present VideoPCDNet, an unsupervised framework for object-centric video decomposition and prediction. Our model uses frequency-domain phase correlation techniques to recursively parse videos into object components, which are represented as transformed versions of learned object prototypes, enabling accurate and interpretable tracking. By explicitly modeling object motion through a combination of frequency domain operations and lightweight learned modules, VideoPCDNet enables accurate unsupervised object tracking and prediction of future video frames. In our experiments, we demonstrate that VideoPCDNet outperforms multiple object-centric baseline models for unsupervised tracking and prediction on several synthetic datasets, while learning interpretable object and motion representations. 

---
# ECCoT: A Framework for Enhancing Effective Cognition via Chain of Thought in Large Language Model 

**Authors**: Zhenke Duan, Jiqun Pan, Jiani Tu, Xiaoyi Wang, Yanqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19599)  

**Abstract**: In the era of large-scale artificial intelligence, Large Language Models (LLMs) have made significant strides in natural language processing. However, they often lack transparency and generate unreliable outputs, raising concerns about their interpretability. To address this, the Chain of Thought (CoT) prompting method structures reasoning into step-by-step deductions. Yet, not all reasoning chains are valid, and errors can lead to unreliable conclusions. We propose ECCoT, an End-to-End Cognitive Chain of Thought Validation Framework, to evaluate and refine reasoning chains in LLMs. ECCoT integrates the Markov Random Field-Embedded Topic Model (MRF-ETM) for topic-aware CoT generation and Causal Sentence-BERT (CSBert) for causal reasoning alignment. By filtering ineffective chains using structured ordering statistics, ECCoT improves interpretability, reduces biases, and enhances the trustworthiness of LLM-based decision-making. Key contributions include the introduction of ECCoT, MRF-ETM for topic-driven CoT generation, and CSBert for causal reasoning enhancement. Code is released at: this https URL. 

---
# Robotics Under Construction: Challenges on Job Sites 

**Authors**: Haruki Uchiito, Akhilesh Bhat, Koji Kusaka, Xiaoya Zhang, Hiraku Kinjo, Honoka Uehara, Motoki Koyama, Shinji Natsume  

**Link**: [PDF](https://arxiv.org/pdf/2506.19597)  

**Abstract**: As labor shortages and productivity stagnation increasingly challenge the construction industry, automation has become essential for sustainable infrastructure development. This paper presents an autonomous payload transportation system as an initial step toward fully unmanned construction sites. Our system, based on the CD110R-3 crawler carrier, integrates autonomous navigation, fleet management, and GNSS-based localization to facilitate material transport in construction site environments. While the current system does not yet incorporate dynamic environment adaptation algorithms, we have begun fundamental investigations into external-sensor based perception and mapping system. Preliminary results highlight the potential challenges, including navigation in evolving terrain, environmental perception under construction-specific conditions, and sensor placement optimization for improving autonomy and efficiency. Looking forward, we envision a construction ecosystem where collaborative autonomous agents dynamically adapt to site conditions, optimizing workflow and reducing human intervention. This paper provides foundational insights into the future of robotics-driven construction automation and identifies critical areas for further technological development. 

---
# Vision Transformer-Based Time-Series Image Reconstruction for Cloud-Filling Applications 

**Authors**: Lujun Li, Yiqun Wang, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2506.19591)  

**Abstract**: Cloud cover in multispectral imagery (MSI) poses significant challenges for early season crop mapping, as it leads to missing or corrupted spectral information. Synthetic aperture radar (SAR) data, which is not affected by cloud interference, offers a complementary solution, but lack sufficient spectral detail for precise crop mapping. To address this, we propose a novel framework, Time-series MSI Image Reconstruction using Vision Transformer (ViT), to reconstruct MSI data in cloud-covered regions by leveraging the temporal coherence of MSI and the complementary information from SAR from the attention mechanism. Comprehensive experiments, using rigorous reconstruction evaluation metrics, demonstrate that Time-series ViT framework significantly outperforms baselines that use non-time-series MSI and SAR or time-series MSI without SAR, effectively enhancing MSI image reconstruction in cloud-covered regions. 

---
# Fake or Real, Can Robots Tell? Evaluating Embodied Vision-Language Models on Real and 3D-Printed Objects 

**Authors**: Federico Tavella, Kathryn Mearns, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19579)  

**Abstract**: Robotic scene understanding increasingly relies on vision-language models (VLMs) to generate natural language descriptions of the environment. In this work, we present a comparative study of captioning strategies for tabletop scenes captured by a robotic arm equipped with an RGB camera. The robot collects images of objects from multiple viewpoints, and we evaluate several models that generate scene descriptions. We compare the performance of various captioning models, like BLIP and VLMs. Our experiments examine the trade-offs between single-view and multi-view captioning, and difference between recognising real-world and 3D printed objects. We quantitatively evaluate object identification accuracy, completeness, and naturalness of the generated captions. Results show that VLMs can be used in robotic settings where common objects need to be recognised, but fail to generalise to novel representations. Our findings provide practical insights into deploying foundation models for embodied agents in real-world settings. 

---
# Towards an Introspective Dynamic Model of Globally Distributed Computing Infrastructures 

**Authors**: Ozgur O. Kilic, David K. Park, Yihui Ren, Tatiana Korchuganova, Sairam Sri Vatsavai, Joseph Boudreau, Tasnuva Chowdhury, Shengyu Feng, Raees Khan, Jaehyung Kim, Scott Klasky, Tadashi Maeno, Paul Nilsson, Verena Ingrid Martinez Outschoorn, Norbert Podhorszki, Frédéric Suter, Wei Yang, Yiming Yang, Shinjae Yoo, Alexei Klimentov, Adolfy Hoisie  

**Link**: [PDF](https://arxiv.org/pdf/2506.19578)  

**Abstract**: Large-scale scientific collaborations like ATLAS, Belle II, CMS, DUNE, and others involve hundreds of research institutes and thousands of researchers spread across the globe. These experiments generate petabytes of data, with volumes soon expected to reach exabytes. Consequently, there is a growing need for computation, including structured data processing from raw data to consumer-ready derived data, extensive Monte Carlo simulation campaigns, and a wide range of end-user analysis. To manage these computational and storage demands, centralized workflow and data management systems are implemented. However, decisions regarding data placement and payload allocation are often made disjointly and via heuristic means. A significant obstacle in adopting more effective heuristic or AI-driven solutions is the absence of a quick and reliable introspective dynamic model to evaluate and refine alternative approaches. In this study, we aim to develop such an interactive system using real-world data. By examining job execution records from the PanDA workflow management system, we have pinpointed key performance indicators such as queuing time, error rate, and the extent of remote data access. The dataset includes five months of activity. Additionally, we are creating a generative AI model to simulate time series of payloads, which incorporate visible features like category, event count, and submitting group, as well as hidden features like the total computational load-derived from existing PanDA records and computing site capabilities. These hidden features, which are not visible to job allocators, whether heuristic or AI-driven, influence factors such as queuing times and data movement. 

---
# Has Machine Translation Evaluation Achieved Human Parity? The Human Reference and the Limits of Progress 

**Authors**: Lorenzo Proietti, Stefano Perrella, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2506.19571)  

**Abstract**: In Machine Translation (MT) evaluation, metric performance is assessed based on agreement with human judgments. In recent years, automatic metrics have demonstrated increasingly high levels of agreement with humans. To gain a clearer understanding of metric performance and establish an upper bound, we incorporate human baselines in the MT meta-evaluation, that is, the assessment of MT metrics' capabilities. Our results show that human annotators are not consistently superior to automatic metrics, with state-of-the-art metrics often ranking on par with or higher than human baselines. Despite these findings suggesting human parity, we discuss several reasons for caution. Finally, we explore the broader implications of our results for the research field, asking: Can we still reliably measure improvements in MT evaluation? With this work, we aim to shed light on the limits of our ability to measure progress in the field, fostering discussion on an issue that we believe is crucial to the entire MT evaluation community. 

---
# FAF: A Feature-Adaptive Framework for Few-Shot Time Series Forecasting 

**Authors**: Pengpeng Ouyang, Dong Chen, Tong Yang, Shuo Feng, Zhao Jin, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19567)  

**Abstract**: Multi-task and few-shot time series forecasting tasks are commonly encountered in scenarios such as the launch of new products in different cities. However, traditional time series forecasting methods suffer from insufficient historical data, which stems from a disregard for the generalized and specific features among different tasks. For the aforementioned challenges, we propose the Feature-Adaptive Time Series Forecasting Framework (FAF), which consists of three key components: the Generalized Knowledge Module (GKM), the Task-Specific Module (TSM), and the Rank Module (RM). During training phase, the GKM is updated through a meta-learning mechanism that enables the model to extract generalized features across related tasks. Meanwhile, the TSM is trained to capture diverse local dynamics through multiple functional regions, each of which learns specific features from individual tasks. During testing phase, the RM dynamically selects the most relevant functional region from the TSM based on input sequence features, which is then combined with the generalized knowledge learned by the GKM to generate accurate forecasts. This design enables FAF to achieve robust and personalized forecasting even with sparse historical observations We evaluate FAF on five diverse real-world datasets under few-shot time series forecasting settings. Experimental results demonstrate that FAF consistently outperforms baselines that include three categories of time series forecasting methods. In particular, FAF achieves a 41.81\% improvement over the best baseline, iTransformer, on the CO$_2$ emissions dataset. 

---
# PrivacyXray: Detecting Privacy Breaches in LLMs through Semantic Consistency and Probability Certainty 

**Authors**: Jinwen He, Yiyang Lu, Zijin Lin, Kai Chen, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19563)  

**Abstract**: Large Language Models (LLMs) are widely used in sensitive domains, including healthcare, finance, and legal services, raising concerns about potential private information leaks during inference. Privacy extraction attacks, such as jailbreaking, expose vulnerabilities in LLMs by crafting inputs that force the models to output sensitive information. However, these attacks cannot verify whether the extracted private information is accurate, as no public datasets exist for cross-validation, leaving a critical gap in private information detection during inference. To address this, we propose PrivacyXray, a novel framework detecting privacy breaches by analyzing LLM inner states. Our analysis reveals that LLMs exhibit higher semantic coherence and probabilistic certainty when generating correct private outputs. Based on this, PrivacyXray detects privacy breaches using four metrics: intra-layer and inter-layer semantic similarity, token-level and sentence-level probability distributions. PrivacyXray addresses critical challenges in private information detection by overcoming the lack of open-source private datasets and eliminating reliance on external data for validation. It achieves this through the synthesis of realistic private data and a detection mechanism based on the inner states of LLMs. Experiments show that PrivacyXray achieves consistent performance, with an average accuracy of 92.69% across five LLMs. Compared to state-of-the-art methods, PrivacyXray achieves significant improvements, with an average accuracy increase of 20.06%, highlighting its stability and practical utility in real-world applications. 

---
# MambaOutRS: A Hybrid CNN-Fourier Architecture for Remote Sensing Image Classification 

**Authors**: Minjong Cheon, Changbae Mun  

**Link**: [PDF](https://arxiv.org/pdf/2506.19561)  

**Abstract**: Recent advances in deep learning for vision tasks have seen the rise of State Space Models (SSMs) like Mamba, celebrated for their linear scalability. However, their adaptation to 2D visual data often necessitates complex modifications that may diminish efficiency. In this paper, we introduce MambaOutRS, a novel hybrid convolutional architecture for remote sensing image classification that re-evaluates the necessity of recurrent SSMs. MambaOutRS builds upon stacked Gated CNN blocks for local feature extraction and introduces a novel Fourier Filter Gate (FFG) module that operates in the frequency domain to capture global contextual information efficiently. Our architecture employs a four-stage hierarchical design and was extensively evaluated on challenging remote sensing datasets: UC Merced, AID, NWPU-RESISC45, and EuroSAT. MambaOutRS consistently achieved state-of-the-art (SOTA) performance across these benchmarks. Notably, our MambaOutRS-t variant (24.0M parameters) attained the highest F1-scores of 98.41\% on UC Merced and 95.99\% on AID, significantly outperforming existing baselines, including larger transformer models and Mamba-based architectures, despite using considerably fewer parameters. An ablation study conclusively demonstrates the critical role of the Fourier Filter Gate in enhancing the model's ability to capture global spatial patterns, leading to robust and accurate classification. These results strongly suggest that the complexities of recurrent SSMs can be effectively superseded by a judicious combination of gated convolutions for spatial mixing and frequency-based gates for spectral global context. Thus, MambaOutRS provides a compelling and efficient paradigm for developing high-performance deep learning models in remote sensing and other vision domains, particularly where computational efficiency is paramount. 

---
# General Methods Make Great Domain-specific Foundation Models: A Case-study on Fetal Ultrasound 

**Authors**: Jakob Ambsdorf, Asbjørn Munk, Sebastian Llambias, Anders Nymark Christensen, Kamil Mikolaj, Randall Balestriero, Martin Tolsgaard, Aasa Feragen, Mads Nielsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19552)  

**Abstract**: With access to large-scale, unlabeled medical datasets, researchers are confronted with two questions: Should they attempt to pretrain a custom foundation model on this medical data, or use transfer-learning from an existing generalist model? And, if a custom model is pretrained, are novel methods required? In this paper we explore these questions by conducting a case-study, in which we train a foundation model on a large regional fetal ultrasound dataset of 2M images. By selecting the well-established DINOv2 method for pretraining, we achieve state-of-the-art results on three fetal ultrasound datasets, covering data from different countries, classification, segmentation, and few-shot tasks. We compare against a series of models pretrained on natural images, ultrasound images, and supervised baselines. Our results demonstrate two key insights: (i) Pretraining on custom data is worth it, even if smaller models are trained on less data, as scaling in natural image pretraining does not translate to ultrasound performance. (ii) Well-tuned methods from computer vision are making it feasible to train custom foundation models for a given medical domain, requiring no hyperparameter tuning and little methodological adaptation. Given these findings, we argue that a bias towards methodological innovation should be avoided when developing domain specific foundation models under common computational resource constraints. 

---
# RCStat: A Statistical Framework for using Relative Contextualization in Transformers 

**Authors**: Debabrata Mahapatra, Shubham Agarwal, Apoorv Saxena, Subrata Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2506.19549)  

**Abstract**: Prior work on input-token importance in auto-regressive transformers has relied on Softmax-normalized attention weights, which obscure the richer structure of pre-Softmax query-key logits. We introduce RCStat, a statistical framework that harnesses raw attention logits via Relative Contextualization (RC), a random variable measuring contextual alignment between token segments, and derive an efficient upper bound for RC. We demonstrate two applications: (i) Key-Value compression, where RC-based thresholds drive adaptive key-value eviction for substantial cache reduction with minimal quality loss; and (ii) Attribution, where RC yields higher-fidelity token-, sentence-, and chunk-level explanations than post-Softmax methods. Across question answering, summarization, and attribution benchmarks, RCStat achieves significant empirical gains, delivering state-of-the-art compression and attribution performance without any model retraining. 

---
# Lost in Translation? Converting RegExes for Log Parsing into Dynatrace Pattern Language 

**Authors**: Julian Fragner, Christian Macho, Bernhard Dieber, Martin Pinzger  

**Link**: [PDF](https://arxiv.org/pdf/2506.19539)  

**Abstract**: Log files provide valuable information for detecting and diagnosing problems in enterprise software applications and data centers. Several log analytics tools and platforms were developed to help filter and extract information from logs, typically using regular expressions (RegExes). Recent commercial log analytics platforms provide domain-specific languages specifically designed for log parsing, such as Grok or the Dynatrace Pattern Language (DPL). However, users who want to migrate to these platforms must manually convert their RegExes into the new pattern language, which is costly and error-prone. In this work, we present Reptile, which combines a rule-based approach for converting RegExes into DPL patterns with a best-effort approach for cases where a full conversion is impossible. Furthermore, it integrates GPT-4 to optimize the obtained DPL patterns. The evaluation with 946 RegExes collected from a large company shows that Reptile safely converted 73.7% of them. The evaluation of Reptile's pattern optimization with 23 real-world RegExes showed an F1-score and MCC above 0.91. These results are promising and have ample practical implications for companies that migrate to a modern log analytics platform, such as Dynatrace. 

---
# ReMAR-DS: Recalibrated Feature Learning for Metal Artifact Reduction and CT Domain Transformation 

**Authors**: Mubashara Rehman, Niki Martinel, Michele Avanzo, Riccardo Spizzo, Christian Micheloni  

**Link**: [PDF](https://arxiv.org/pdf/2506.19531)  

**Abstract**: Artifacts in kilo-Voltage CT (kVCT) imaging degrade image quality, impacting clinical decisions. We propose a deep learning framework for metal artifact reduction (MAR) and domain transformation from kVCT to Mega-Voltage CT (MVCT). The proposed framework, ReMAR-DS, utilizes an encoder-decoder architecture with enhanced feature recalibration, effectively reducing artifacts while preserving anatomical structures. This ensures that only relevant information is utilized in the reconstruction process. By infusing recalibrated features from the encoder block, the model focuses on relevant spatial regions (e.g., areas with artifacts) and highlights key features across channels (e.g., anatomical structures), leading to improved reconstruction of artifact-corrupted regions. Unlike traditional MAR methods, our approach bridges the gap between high-resolution kVCT and artifact-resistant MVCT, enhancing radiotherapy planning. It produces high-quality MVCT-like reconstructions, validated through qualitative and quantitative evaluations. Clinically, this enables oncologists to rely on kVCT alone, reducing repeated high-dose MVCT scans and lowering radiation exposure for cancer patients. 

---
# Automatic Posology Structuration : What role for LLMs? 

**Authors**: Natalia Bobkova, Laura Zanella-Calzada, Anyes Tafoughalt, Raphaël Teboul, François Plesse, Félix Gaschi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19525)  

**Abstract**: Automatically structuring posology instructions is essential for improving medication safety and enabling clinical decision support. In French prescriptions, these instructions are often ambiguous, irregular, or colloquial, limiting the effectiveness of classic ML pipelines. We explore the use of Large Language Models (LLMs) to convert free-text posologies into structured formats, comparing prompt-based methods and fine-tuning against a "pre-LLM" system based on Named Entity Recognition and Linking (NERL). Our results show that while prompting improves performance, only fine-tuned LLMs match the accuracy of the baseline. Through error analysis, we observe complementary strengths: NERL offers structural precision, while LLMs better handle semantic nuances. Based on this, we propose a hybrid pipeline that routes low-confidence cases from NERL (<0.8) to the LLM, selecting outputs based on confidence scores. This strategy achieves 91% structuration accuracy while minimizing latency and compute. Our results show that this hybrid approach improves structuration accuracy while limiting computational cost, offering a scalable solution for real-world clinical use. 

---
# MATE: LLM-Powered Multi-Agent Translation Environment for Accessibility Applications 

**Authors**: Aleksandr Algazinov, Matt Laing, Paul Laban  

**Link**: [PDF](https://arxiv.org/pdf/2506.19502)  

**Abstract**: Accessibility remains a critical concern in today's society, as many technologies are not developed to support the full range of user needs. Existing multi-agent systems (MAS) often cannot provide comprehensive assistance for users in need due to the lack of customization stemming from closed-source designs. Consequently, individuals with disabilities frequently encounter significant barriers when attempting to interact with digital environments. We introduce MATE, a multimodal accessibility MAS, which performs the modality conversions based on the user's needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to an understandable format. For instance, if the user cannot see well and receives an image, the system converts this image to its audio description. MATE can be applied to a wide range of domains, industries, and areas, such as healthcare, and can become a useful assistant for various groups of users. The system supports multiple types of models, ranging from LLM API calling to using custom machine learning (ML) classifiers. This flexibility ensures that the system can be adapted to various needs and is compatible with a wide variety of hardware. Since the system is expected to run locally, it ensures the privacy and security of sensitive information. In addition, the framework can be effectively integrated with institutional technologies (e.g., digital healthcare service) for real-time user assistance. Furthermore, we introduce ModCon-Task-Identifier, a model that is capable of extracting the precise modality conversion task from the user input. Numerous experiments show that ModCon-Task-Identifier consistently outperforms other LLMs and statistical models on our custom data. Our code and data are publicly available at this https URL. 

---
# Experimental Assessment of Neural 3D Reconstruction for Small UAV-based Applications 

**Authors**: Genís Castillo Gómez-Raya, Álmos Veres-Vitályos, Filip Lemic, Pablo Royo, Mario Montagud, Sergi Fernández, Sergi Abadal, Xavier Costa-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2506.19491)  

**Abstract**: The increasing miniaturization of Unmanned Aerial Vehicles (UAVs) has expanded their deployment potential to indoor and hard-to-reach areas. However, this trend introduces distinct challenges, particularly in terms of flight dynamics and power consumption, which limit the UAVs' autonomy and mission capabilities. This paper presents a novel approach to overcoming these limitations by integrating Neural 3D Reconstruction (N3DR) with small UAV systems for fine-grained 3-Dimensional (3D) digital reconstruction of small static objects. Specifically, we design, implement, and evaluate an N3DR-based pipeline that leverages advanced models, i.e., Instant-ngp, Nerfacto, and Splatfacto, to improve the quality of 3D reconstructions using images of the object captured by a fleet of small UAVs. We assess the performance of the considered models using various imagery and pointcloud metrics, comparing them against the baseline Structure from Motion (SfM) algorithm. The experimental results demonstrate that the N3DR-enhanced pipeline significantly improves reconstruction quality, making it feasible for small UAVs to support high-precision 3D mapping and anomaly detection in constrained environments. In more general terms, our results highlight the potential of N3DR in advancing the capabilities of miniaturized UAV systems. 

---
# Recalling The Forgotten Class Memberships: Unlearned Models Can Be Noisy Labelers to Leak Privacy 

**Authors**: Zhihao Sui, Liang Hu, Jian Cao, Dora D. Liu, Usman Naseem, Zhongyuan Lai, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19486)  

**Abstract**: Machine Unlearning (MU) technology facilitates the removal of the influence of specific data instances from trained models on request. Despite rapid advancements in MU technology, its vulnerabilities are still underexplored, posing potential risks of privacy breaches through leaks of ostensibly unlearned information. Current limited research on MU attacks requires access to original models containing privacy data, which violates the critical privacy-preserving objective of MU. To address this gap, we initiate an innovative study on recalling the forgotten class memberships from unlearned models (ULMs) without requiring access to the original one. Specifically, we implement a Membership Recall Attack (MRA) framework with a teacher-student knowledge distillation architecture, where ULMs serve as noisy labelers to transfer knowledge to student models. Then, it is translated into a Learning with Noisy Labels (LNL) problem for inferring the correct labels of the forgetting instances. Extensive experiments on state-of-the-art MU methods with multiple real datasets demonstrate that the proposed MRA strategy exhibits high efficacy in recovering class memberships of unlearned instances. As a result, our study and evaluation have established a benchmark for future research on MU vulnerabilities. 

---
# Dialogic Pedagogy for Large Language Models: Aligning Conversational AI with Proven Theories of Learning 

**Authors**: Russell Beale  

**Link**: [PDF](https://arxiv.org/pdf/2506.19484)  

**Abstract**: Large Language Models (LLMs) are rapidly transforming education by enabling rich conversational learning experiences. This article provides a comprehensive review of how LLM-based conversational agents are being used in higher education, with extensions to secondary and lifelong learning contexts. We synthesize existing literature on LLMs in education and theories of conversational and dialogic pedagogy - including Vygotsky's sociocultural learning (scaffolding and the Zone of Proximal Development), the Socratic method, and Laurillard's conversational framework - and examine how prompting strategies and retrieval-augmented generation (RAG) can align LLM behaviors with these pedagogical theories, and how it can support personalized, adaptive learning. We map educational theories to LLM capabilities, highlighting where LLM-driven dialogue supports established learning principles and where it challenges or falls short of traditional pedagogical assumptions. Notable gaps in applying prior theories to LLMs are identified, such as the models tendency to provide direct answers instead of fostering co-construction of knowledge, and the need to account for the constant availability and broad but non-human expertise of LLM tutors. In response, we propose practical strategies to better align LLM interactions with sound pedagogy - for example, designing prompts that encourage Socratic questioning, scaffolded guidance, and student reflection, as well as integrating retrieval mechanisms to ensure accuracy and contextual relevance. Our aim is to bridge the gap between educational theory and the emerging practice of AI-driven conversational learning, offering insights and tools for making LLM-based dialogues more educationally productive and theory-aligned. 

---
# Fast and Distributed Equivariant Graph Neural Networks by Virtual Node Learning 

**Authors**: Yuelin Zhang, Jiacheng Cen, Jiaqi Han, Wenbing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19482)  

**Abstract**: Equivariant Graph Neural Networks (GNNs) have achieved remarkable success across diverse scientific applications. However, existing approaches face critical efficiency challenges when scaling to large geometric graphs and suffer significant performance degradation when the input graphs are sparsified for computational tractability. To address these limitations, we introduce FastEGNN and DistEGNN, two novel enhancements to equivariant GNNs for large-scale geometric graphs. FastEGNN employs a key innovation: a small ordered set of virtual nodes that effectively approximates the large unordered graph of real nodes. Specifically, we implement distinct message passing and aggregation mechanisms for different virtual nodes to ensure mutual distinctiveness, and minimize Maximum Mean Discrepancy (MMD) between virtual and real coordinates to achieve global distributedness. This design enables FastEGNN to maintain high accuracy while efficiently processing large-scale sparse graphs. For extremely large-scale geometric graphs, we present DistEGNN, a distributed extension where virtual nodes act as global bridges between subgraphs in different devices, maintaining consistency while dramatically reducing memory and computational overhead. We comprehensively evaluate our models across four challenging domains: N-body systems (100 nodes), protein dynamics (800 nodes), Water-3D (8,000 nodes), and our new Fluid113K benchmark (113,000 nodes). Results demonstrate superior efficiency and performance, establishing new capabilities in large-scale equivariant graph learning. Code is available at this https URL. 

---
# Surgery-R1: Advancing Surgical-VQLA with Reasoning Multimodal Large Language Model via Reinforcement Learning 

**Authors**: Pengfei Hao, Shuaibo Li, Hongqiu Wang, Zhizhuo Kou, Junhang Zhang, Guang Yang, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19469)  

**Abstract**: In recent years, significant progress has been made in the field of surgical scene understanding, particularly in the task of Visual Question Localized-Answering in robotic surgery (Surgical-VQLA). However, existing Surgical-VQLA models lack deep reasoning capabilities and interpretability in surgical scenes, which limits their reliability and potential for development in clinical applications. To address this issue, inspired by the development of Reasoning Multimodal Large Language Models (MLLMs), we first build the Surgery-R1-54k dataset, including paired data for Visual-QA, Grounding-QA, and Chain-of-Thought (CoT). Then, we propose the first Reasoning MLLM for Surgical-VQLA (Surgery-R1). In our Surgery-R1, we design a two-stage fine-tuning mechanism to enable the basic MLLM with complex reasoning abilities by utilizing supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). Furthermore, for an efficient and high-quality rule-based reward system in our RFT, we design a Multimodal Coherence reward mechanism to mitigate positional illusions that may arise in surgical scenarios. Experiment results demonstrate that Surgery-R1 outperforms other existing state-of-the-art (SOTA) models in the Surgical-VQLA task and widely-used MLLMs, while also validating its reasoning capabilities and the effectiveness of our approach. The code and dataset will be organized in this https URL. 

---
# MuBench: Assessment of Multilingual Capabilities of Large Language Models Across 61 Languages 

**Authors**: Wenhan Han, Yifan Zhang, Zhixun Chen, Binbin Liu, Haobin Lin, Bingni Zhang, Taifeng Wang, Mykola Pechenizkiy, Meng Fang, Yin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.19468)  

**Abstract**: Multilingual large language models (LLMs) are advancing rapidly, with new models frequently claiming support for an increasing number of languages. However, existing evaluation datasets are limited and lack cross-lingual alignment, leaving assessments of multilingual capabilities fragmented in both language and skill coverage. To address this, we introduce MuBench, a benchmark covering 61 languages and evaluating a broad range of capabilities. We evaluate several state-of-the-art multilingual LLMs and find notable gaps between claimed and actual language coverage, particularly a persistent performance disparity between English and low-resource languages. Leveraging MuBench's alignment, we propose Multilingual Consistency (MLC) as a complementary metric to accuracy for analyzing performance bottlenecks and guiding model improvement. Finally, we pretrain a suite of 1.2B-parameter models on English and Chinese with 500B tokens, varying language ratios and parallel data proportions to investigate cross-lingual transfer dynamics. 

---
# Can Large Language Models Capture Human Annotator Disagreements? 

**Authors**: Jingwei Ni, Yu Fan, Vilém Zouhar, Donya Rooein, Alexander Hoyle, Mrinmaya Sachan, Markus Leippold, Dirk Hovy, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2506.19467)  

**Abstract**: Human annotation variation (i.e., annotation disagreements) is common in NLP and often reflects important information such as task subjectivity and sample ambiguity. While Large Language Models (LLMs) are increasingly used for automatic annotation to reduce human effort, their evaluation often focuses on predicting the majority-voted "ground truth" labels. It is still unclear, however, whether these models also capture informative human annotation variation. Our work addresses this gap by extensively evaluating LLMs' ability to predict annotation disagreements without access to repeated human labels. Our results show that LLMs struggle with modeling disagreements, which can be overlooked by majority label-based evaluations. Notably, while RLVR-style (Reinforcement learning with verifiable rewards) reasoning generally boosts LLM performance, it degrades performance in disagreement prediction. Our findings highlight the critical need for evaluating and improving LLM annotators in disagreement modeling. Code and data at this https URL. 

---
# Stylized Structural Patterns for Improved Neural Network Pre-training 

**Authors**: Farnood Salehi, Vandit Sharma, Amirhossein Askari Farsangi, Tunç Ozan Aydın  

**Link**: [PDF](https://arxiv.org/pdf/2506.19465)  

**Abstract**: Modern deep learning models in computer vision require large datasets of real images, which are difficult to curate and pose privacy and legal concerns, limiting their commercial use. Recent works suggest synthetic data as an alternative, yet models trained with it often underperform. This paper proposes a two-step approach to bridge this gap. First, we propose an improved neural fractal formulation through which we introduce a new class of synthetic data. Second, we propose reverse stylization, a technique that transfers visual features from a small, license-free set of real images onto synthetic datasets, enhancing their effectiveness. We analyze the domain gap between our synthetic datasets and real images using Kernel Inception Distance (KID) and show that our method achieves a significantly lower distributional gap compared to existing synthetic datasets. Furthermore, our experiments across different tasks demonstrate the practical impact of this reduced gap. We show that pretraining the EDM2 diffusion model on our synthetic dataset leads to an 11% reduction in FID during image generation, compared to models trained on existing synthetic datasets, and a 20% decrease in autoencoder reconstruction error, indicating improved performance in data representation. Furthermore, a ViT-S model trained for classification on this synthetic data achieves over a 10% improvement in ImageNet-100 accuracy. Our work opens up exciting possibilities for training practical models when sufficiently large real training sets are not available. 

---
# Iterative Quantum Feature Maps 

**Authors**: Nasa Matsumoto, Quoc Hoan Tran, Koki Chinzei, Yasuhiro Endo, Hirotaka Oshima  

**Link**: [PDF](https://arxiv.org/pdf/2506.19461)  

**Abstract**: Quantum machine learning models that leverage quantum circuits as quantum feature maps (QFMs) are recognized for their enhanced expressive power in learning tasks. Such models have demonstrated rigorous end-to-end quantum speedups for specific families of classification problems. However, deploying deep QFMs on real quantum hardware remains challenging due to circuit noise and hardware constraints. Additionally, variational quantum algorithms often suffer from computational bottlenecks, particularly in accurate gradient estimation, which significantly increases quantum resource demands during training. We propose Iterative Quantum Feature Maps (IQFMs), a hybrid quantum-classical framework that constructs a deep architecture by iteratively connecting shallow QFMs with classically computed augmentation weights. By incorporating contrastive learning and a layer-wise training mechanism, IQFMs effectively reduces quantum runtime and mitigates noise-induced degradation. In tasks involving noisy quantum data, numerical experiments show that IQFMs outperforms quantum convolutional neural networks, without requiring the optimization of variational quantum parameters. Even for a typical classical image classification benchmark, a carefully designed IQFMs achieves performance comparable to that of classical neural networks. This framework presents a promising path to address current limitations and harness the full potential of quantum-enhanced machine learning. 

---
# Tagged for Direction: Pinning Down Causal Edge Directions with Precision 

**Authors**: Florian Peter Busch, Moritz Willig, Florian Guldan, Kristian Kersting, Devendra Singh Dhami  

**Link**: [PDF](https://arxiv.org/pdf/2506.19459)  

**Abstract**: Not every causal relation between variables is equal, and this can be leveraged for the task of causal discovery. Recent research shows that pairs of variables with particular type assignments induce a preference on the causal direction of other pairs of variables with the same type. Although useful, this assignment of a specific type to a variable can be tricky in practice. We propose a tag-based causal discovery approach where multiple tags are assigned to each variable in a causal graph. Existing causal discovery approaches are first applied to direct some edges, which are then used to determine edge relations between tags. Then, these edge relations are used to direct the undirected edges. Doing so improves upon purely type-based relations, where the assumption of type consistency lacks robustness and flexibility due to being restricted to single types for each variable. Our experimental evaluations show that this boosts causal discovery and that these high-level tag relations fit common knowledge. 

---
# Mem4Nav: Boosting Vision-and-Language Navigation in Urban Environments with a Hierarchical Spatial-Cognition Long-Short Memory System 

**Authors**: Lixuan He, Haoyu Dong, Zhenxing Chen, Yangcheng Yu, Jie Feng, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19433)  

**Abstract**: Vision-and-Language Navigation (VLN) in large-scale urban environments requires embodied agents to ground linguistic instructions in complex scenes and recall relevant experiences over extended time horizons. Prior modular pipelines offer interpretability but lack unified memory, while end-to-end (M)LLM agents excel at fusing vision and language yet remain constrained by fixed context windows and implicit spatial reasoning. We introduce \textbf{Mem4Nav}, a hierarchical spatial-cognition long-short memory system that can augment any VLN backbone. Mem4Nav fuses a sparse octree for fine-grained voxel indexing with a semantic topology graph for high-level landmark connectivity, storing both in trainable memory tokens embedded via a reversible Transformer. Long-term memory (LTM) compresses and retains historical observations at both octree and graph nodes, while short-term memory (STM) caches recent multimodal entries in relative coordinates for real-time obstacle avoidance and local planning. At each step, STM retrieval sharply prunes dynamic context, and, when deeper history is needed, LTM tokens are decoded losslessly to reconstruct past embeddings. Evaluated on Touchdown and Map2Seq across three backbones (modular, state-of-the-art VLN with prompt-based LLM, and state-of-the-art VLN with strided-attention MLLM), Mem4Nav yields 7-13 pp gains in Task Completion, sufficient SPD reduction, and >10 pp nDTW improvement. Ablations confirm the indispensability of both the hierarchical map and dual memory modules. Our codes are open-sourced via this https URL. 

---
# A Global-Local Cross-Attention Network for Ultra-high Resolution Remote Sensing Image Semantic Segmentation 

**Authors**: Chen Yi, Shan LianLei  

**Link**: [PDF](https://arxiv.org/pdf/2506.19406)  

**Abstract**: With the rapid development of ultra-high resolution (UHR) remote sensing technology, the demand for accurate and efficient semantic segmentation has increased significantly. However, existing methods face challenges in computational efficiency and multi-scale feature fusion. To address these issues, we propose GLCANet (Global-Local Cross-Attention Network), a lightweight segmentation framework designed for UHR remote sensing this http URL employs a dual-stream architecture to efficiently fuse global semantics and local details while minimizing GPU usage. A self-attention mechanism enhances long-range dependencies, refines global features, and preserves local details for better semantic consistency. A masked cross-attention mechanism also adaptively fuses global-local features, selectively enhancing fine-grained details while exploiting global context to improve segmentation accuracy. Experimental results show that GLCANet outperforms state-of-the-art methods regarding accuracy and computational efficiency. The model effectively processes large, high-resolution images with a small memory footprint, providing a promising solution for real-world remote sensing applications. 

---
# Automated Detection of Pre-training Text in Black-box LLMs 

**Authors**: Ruihan Hu, Yu-Ming Shang, Jiankun Peng, Wei Luo, Yazhe Wang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19399)  

**Abstract**: Detecting whether a given text is a member of the pre-training data of Large Language Models (LLMs) is crucial for ensuring data privacy and copyright protection. Most existing methods rely on the LLM's hidden information (e.g., model parameters or token probabilities), making them ineffective in the black-box setting, where only input and output texts are accessible. Although some methods have been proposed for the black-box setting, they rely on massive manual efforts such as designing complicated questions or instructions. To address these issues, we propose VeilProbe, the first framework for automatically detecting LLMs' pre-training texts in a black-box setting without human intervention. VeilProbe utilizes a sequence-to-sequence mapping model to infer the latent mapping feature between the input text and the corresponding output suffix generated by the LLM. Then it performs the key token perturbations to obtain more distinguishable membership features. Additionally, considering real-world scenarios where the ground-truth training text samples are limited, a prototype-based membership classifier is introduced to alleviate the overfitting issue. Extensive evaluations on three widely used datasets demonstrate that our framework is effective and superior in the black-box setting. 

---
# NAADA: A Noise-Aware Attention Denoising Autoencoder for Dental Panoramic Radiographs 

**Authors**: Khuram Naveed, Bruna Neves de Freitas, Ruben Pauwels  

**Link**: [PDF](https://arxiv.org/pdf/2506.19387)  

**Abstract**: Convolutional denoising autoencoders (DAEs) are powerful tools for image restoration. However, they inherit a key limitation of convolutional neural networks (CNNs): they tend to recover low-frequency features, such as smooth regions, more effectively than high-frequency details. This leads to the loss of fine details, which is particularly problematic in dental radiographs where preserving subtle anatomical structures is crucial. While self-attention mechanisms can help mitigate this issue by emphasizing important features, conventional attention methods often prioritize features corresponding to cleaner regions and may overlook those obscured by noise. To address this limitation, we propose a noise-aware self-attention method, which allows the model to effectively focus on and recover key features even within noisy regions. Building on this approach, we introduce the noise-aware attention-enhanced denoising autoencoder (NAADA) network for enhancing noisy panoramic dental radiographs. Compared with the recent state of the art (and much heavier) methods like Uformer, MResDNN etc., our method improves the reconstruction of fine details, ensuring better image quality and diagnostic accuracy. 

---
# From High-SNR Radar Signal to ECG: A Transfer Learning Model with Cardio-Focusing Algorithm for Scenarios with Limited Data 

**Authors**: Yuanyuan Zhang, Haocheng Zhao, Sijie Xiong, Rui Yang, Eng Gee Lim, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2506.19358)  

**Abstract**: Electrocardiogram (ECG), as a crucial find-grained cardiac feature, has been successfully recovered from radar signals in the literature, but the performance heavily relies on the high-quality radar signal and numerous radar-ECG pairs for training, restricting the applications in new scenarios due to data scarcity. Therefore, this work will focus on radar-based ECG recovery in new scenarios with limited data and propose a cardio-focusing and -tracking (CFT) algorithm to precisely track the cardiac location to ensure an efficient acquisition of high-quality radar signals. Furthermore, a transfer learning model (RFcardi) is proposed to extract cardio-related information from the radar signal without ECG ground truth based on the intrinsic sparsity of cardiac features, and only a few synchronous radar-ECG pairs are required to fine-tune the pre-trained model for the ECG recovery. The experimental results reveal that the proposed CFT can dynamically identify the cardiac location, and the RFcardi model can effectively generate faithful ECG recoveries after using a small number of radar-ECG pairs for training. The code and dataset are available after the publication. 

---
# Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation 

**Authors**: Jisu Shin, Juhyun Oh, Eunsu Kim, Hoyun Song, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.19352)  

**Abstract**: Ensuring persona fidelity in large language models (LLMs) is essential for maintaining coherent and engaging human-AI interactions. However, LLMs often exhibit Out-of-Character (OOC) behavior, where generated responses deviate from an assigned persona, leading to inconsistencies that affect model reliability. Existing evaluation methods typically assign single scores to entire responses, struggling to capture subtle persona misalignment, particularly in long-form text generation. To address this limitation, we propose an atomic-level evaluation framework that quantifies persona fidelity at a finer granularity. Our three key metrics measure the degree of persona alignment and consistency within and across generations. Our approach enables a more precise and realistic assessment of persona fidelity by identifying subtle deviations that real users would encounter. Through our experiments, we demonstrate that our framework effectively detects persona inconsistencies that prior methods overlook. By analyzing persona fidelity across diverse tasks and personality types, we reveal how task structure and persona desirability influence model adaptability, highlighting challenges in maintaining consistent persona expression. 

---
# In-Context Occam's Razor: How Transformers Prefer Simpler Hypotheses on the Fly 

**Authors**: Puneesh Deora, Bhavya Vasudeva, Tina Behnia, Christos Thrampoulidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.19351)  

**Abstract**: In-context learning (ICL) enables transformers to adapt to new tasks through contextual examples without parameter updates. While existing research has typically studied ICL in fixed-complexity environments, practical language models encounter tasks spanning diverse complexity levels. This paper investigates how transformers navigate hierarchical task structures where higher-complexity categories can perfectly represent any pattern generated by simpler ones. We design well-controlled testbeds based on Markov chains and linear regression that reveal transformers not only identify the appropriate complexity level for each task but also accurately infer the corresponding parameters--even when the in-context examples are compatible with multiple complexity hypotheses. Notably, when presented with data generated by simpler processes, transformers consistently favor the least complex sufficient explanation. We theoretically explain this behavior through a Bayesian framework, demonstrating that transformers effectively implement an in-context Bayesian Occam's razor by balancing model fit against complexity penalties. We further ablate on the roles of model size, training mixture distribution, inference context length, and architecture. Finally, we validate this Occam's razor-like inductive bias on a pretrained GPT-4 model with Boolean-function tasks as case study, suggesting it may be inherent to transformers trained on diverse task distributions. 

---
# Discrepancy-Aware Graph Mask Auto-Encoder 

**Authors**: Ziyu Zheng, Yaming Yang, Ziyu Guan, Wei Zhao, Weigang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19343)  

**Abstract**: Masked Graph Auto-Encoder, a powerful graph self-supervised training paradigm, has recently shown superior performance in graph representation learning. Existing works typically rely on node contextual information to recover the masked information. However, they fail to generalize well to heterophilic graphs where connected nodes may be not similar, because they focus only on capturing the neighborhood information and ignoring the discrepancy information between different nodes, resulting in indistinguishable node representations. In this paper, to address this issue, we propose a Discrepancy-Aware Graph Mask Auto-Encoder (DGMAE). It obtains more distinguishable node representations by reconstructing the discrepancy information of neighboring nodes during the masking process. We conduct extensive experiments on 17 widely-used benchmark datasets. The results show that our DGMAE can effectively preserve the discrepancies of nodes in low-dimensional space. Moreover, DGMAE significantly outperforms state-of-the-art graph self-supervised learning methods on three graph analytic including tasks node classification, node clustering, and graph classification, demonstrating its remarkable superiority. The code of DGMAE is available at this https URL. 

---
# Unlocking Insights Addressing Alcohol Inference Mismatch through Database-Narrative Alignment 

**Authors**: Sudesh Bhagat, Raghupathi Kandiboina, Ibne Farabi Shihab, Skylar Knickerbocker, Neal Hawkins, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.19342)  

**Abstract**: Road traffic crashes are a significant global cause of fatalities, emphasizing the urgent need for accurate crash data to enhance prevention strategies and inform policy development. This study addresses the challenge of alcohol inference mismatch (AIM) by employing database narrative alignment to identify AIM in crash data. A framework was developed to improve data quality in crash management systems and reduce the percentage of AIM crashes. Utilizing the BERT model, the analysis of 371,062 crash records from Iowa (2016-2022) revealed 2,767 AIM incidents, resulting in an overall AIM percentage of 24.03%. Statistical tools, including the Probit Logit model, were used to explore the crash characteristics affecting AIM patterns. The findings indicate that alcohol-related fatal crashes and nighttime incidents have a lower percentage of the mismatch, while crashes involving unknown vehicle types and older drivers are more susceptible to mismatch. The geospatial cluster as part of this study can identify the regions which have an increased need for education and training. These insights highlight the necessity for targeted training programs and data management teams to improve the accuracy of crash reporting and support evidence-based policymaking. 

---
# JCAPT: A Joint Modeling Approach for CAPT 

**Authors**: Tzu-Hsuan Yang, Yue-Yang He, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19315)  

**Abstract**: Effective pronunciation feedback is critical in second language (L2) learning, for which computer-assisted pronunciation training (CAPT) systems often encompass two key tasks: automatic pronunciation assessment (APA) and mispronunciation detection and diagnosis (MDD). Recent work has shown that joint modeling of these two tasks can yield mutual benefits. Our unified framework leverages Mamba, a selective state space model (SSM), while integrating phonological features and think token strategies to jointly enhance interpretability and fine-grained temporal reasoning in APA and MDD. To our knowledge, this is the first study to combine phonological attribution, SSM-based modeling, and prompting in CAPT. A series of experiments conducted on the speechocean762 benchmark demonstrate that our model consistently outperforms prior methods, particularly on the MDD task. 

---
# Capturing Fine-Grained Alignments Improves 3D Affordance Detection 

**Authors**: Junsei Tokumitsu, Yuiga Wada  

**Link**: [PDF](https://arxiv.org/pdf/2506.19312)  

**Abstract**: In this work, we address the challenge of affordance detection in 3D point clouds, a task that requires effectively capturing fine-grained alignments between point clouds and text. Existing methods often struggle to model such alignments, resulting in limited performance on standard benchmarks. A key limitation of these approaches is their reliance on simple cosine similarity between point cloud and text embeddings, which lacks the expressiveness needed for fine-grained reasoning. To address this limitation, we propose LM-AD, a novel method for affordance detection in 3D point clouds. Moreover, we introduce the Affordance Query Module (AQM), which efficiently captures fine-grained alignment between point clouds and text by leveraging a pretrained language model. We demonstrated that our method outperformed existing approaches in terms of accuracy and mean Intersection over Union on the 3D AffordanceNet dataset. 

---
# AirV2X: Unified Air-Ground Vehicle-to-Everything Collaboration 

**Authors**: Xiangbo Gao, Yuheng Wu, Xuewen Luo, Keshu Wu, Xinghao Chen, Yuping Wang, Chenxi Liu, Yang Zhou, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19283)  

**Abstract**: While multi-vehicular collaborative driving demonstrates clear advantages over single-vehicle autonomy, traditional infrastructure-based V2X systems remain constrained by substantial deployment costs and the creation of "uncovered danger zones" in rural and suburban areas. We present AirV2X-Perception, a large-scale dataset that leverages Unmanned Aerial Vehicles (UAVs) as a flexible alternative or complement to fixed Road-Side Units (RSUs). Drones offer unique advantages over ground-based perception: complementary bird's-eye-views that reduce occlusions, dynamic positioning capabilities that enable hovering, patrolling, and escorting navigation rules, and significantly lower deployment costs compared to fixed infrastructure. Our dataset comprises 6.73 hours of drone-assisted driving scenarios across urban, suburban, and rural environments with varied weather and lighting conditions. The AirV2X-Perception dataset facilitates the development and standardized evaluation of Vehicle-to-Drone (V2D) algorithms, addressing a critical gap in the rapidly expanding field of aerial-assisted autonomous driving systems. The dataset and development kits are open-sourced at this https URL. 

---
# EmoStage: A Framework for Accurate Empathetic Response Generation via Perspective-Taking and Phase Recognition 

**Authors**: Zhiyang Qi, Keiko Takamizo, Mariko Ukiyo, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2506.19279)  

**Abstract**: The rising demand for mental health care has fueled interest in AI-driven counseling systems. While large language models (LLMs) offer significant potential, current approaches face challenges, including limited understanding of clients' psychological states and counseling stages, reliance on high-quality training data, and privacy concerns associated with commercial deployment. To address these issues, we propose EmoStage, a framework that enhances empathetic response generation by leveraging the inference capabilities of open-source LLMs without additional training data. Our framework introduces perspective-taking to infer clients' psychological states and support needs, enabling the generation of emotionally resonant responses. In addition, phase recognition is incorporated to ensure alignment with the counseling process and to prevent contextually inappropriate or inopportune responses. Experiments conducted in both Japanese and Chinese counseling settings demonstrate that EmoStage improves the quality of responses generated by base models and performs competitively with data-driven methods. 

---
# AnchorDP3: 3D Affordance Guided Sparse Diffusion Policy for Robotic Manipulation 

**Authors**: Ziyan Zhao, Ke Fan, He-Yang Xu, Ning Qiao, Bo Peng, Wenlong Gao, Dongjiang Li, Hui Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19269)  

**Abstract**: We present AnchorDP3, a diffusion policy framework for dual-arm robotic manipulation that achieves state-of-the-art performance in highly randomized environments. AnchorDP3 integrates three key innovations: (1) Simulator-Supervised Semantic Segmentation, using rendered ground truth to explicitly segment task-critical objects within the point cloud, which provides strong affordance priors; (2) Task-Conditioned Feature Encoders, lightweight modules processing augmented point clouds per task, enabling efficient multi-task learning through a shared diffusion-based action expert; (3) Affordance-Anchored Keypose Diffusion with Full State Supervision, replacing dense trajectory prediction with sparse, geometrically meaningful action anchors, i.e., keyposes such as pre-grasp pose, grasp pose directly anchored to affordances, drastically simplifying the prediction space; the action expert is forced to predict both robot joint angles and end-effector poses simultaneously, which exploits geometric consistency to accelerate convergence and boost accuracy. Trained on large-scale, procedurally generated simulation data, AnchorDP3 achieves a 98.7% average success rate in the RoboTwin benchmark across diverse tasks under extreme randomization of objects, clutter, table height, lighting, and backgrounds. This framework, when integrated with the RoboTwin real-to-sim pipeline, has the potential to enable fully autonomous generation of deployable visuomotor policies from only scene and instruction, totally eliminating human demonstrations from learning manipulation skills. 

---
# Enhancing Generalization of Spiking Neural Networks Through Temporal Regularization 

**Authors**: Boxuan Zhang, Zhen Xu, Kuan Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19256)  

**Abstract**: Spiking Neural Networks (SNNs) have received widespread attention due to their event-driven and low-power characteristics, making them particularly effective for processing event-based neuromorphic data. Recent studies have shown that directly trained SNNs suffer from severe overfitting issues due to the limited scale of neuromorphic datasets and the gradient mismatching problem, which fundamentally constrain their generalization performance. In this paper, we propose a temporal regularization training (TRT) method by introducing a time-dependent regularization mechanism to enforce stronger constraints on early timesteps. We compare the performance of TRT with other state-of-the-art methods performance on datasets including CIFAR10/100, ImageNet100, DVS-CIFAR10, and N-Caltech101. To validate the effectiveness of TRT, we conducted ablation studies and analyses including loss landscape visualization and learning curve analysis, demonstrating that TRT can effectively mitigate overfitting and flatten the training loss landscape, thereby enhancing generalizability. Furthermore, we establish a theoretical interpretation of TRT's temporal regularization mechanism based on the results of Fisher information analysis. We analyze the temporal information dynamics inside SNNs by tracking Fisher information during the TRT training process, revealing the Temporal Information Concentration (TIC) phenomenon, where Fisher information progressively concentrates in early timesteps. The time-decaying regularization mechanism implemented in TRT effectively guides the network to learn robust features in early timesteps with rich information, thereby leading to significant improvements in model generalization. Code is available at this https URL. 

---
# Robust Behavior Cloning Via Global Lipschitz Regularization 

**Authors**: Shili Wu, Yizhao Jin, Puhua Niu, Aniruddha Datta, Sean B. Andersson  

**Link**: [PDF](https://arxiv.org/pdf/2506.19250)  

**Abstract**: Behavior Cloning (BC) is an effective imitation learning technique and has even been adopted in some safety-critical domains such as autonomous vehicles. BC trains a policy to mimic the behavior of an expert by using a dataset composed of only state-action pairs demonstrated by the expert, without any additional interaction with the environment. However, During deployment, the policy observations may contain measurement errors or adversarial disturbances. Since the observations may deviate from the true states, they can mislead the agent into making sub-optimal actions. In this work, we use a global Lipschitz regularization approach to enhance the robustness of the learned policy network. We then show that the resulting global Lipschitz property provides a robustness certificate to the policy with respect to different bounded norm perturbations. Then, we propose a way to construct a Lipschitz neural network that ensures the policy robustness. We empirically validate our theory across various environments in Gymnasium. Keywords: Robust Reinforcement Learning; Behavior Cloning; Lipschitz Neural Network 

---
# Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification 

**Authors**: Minghao Qin, Xiangrui Liu, Zhengyang Liang, Yan Shu, Huaying Yuan, Juenjie Zhou, Shitao Xiao, Bo Zhao, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19225)  

**Abstract**: Multi-modal large language models (MLLMs) models have made significant progress in video understanding over the past few years. However, processing long video inputs remains a major challenge due to high memory and computational costs. This makes it difficult for current models to achieve both strong performance and high efficiency in long video understanding. To address this challenge, we propose Video-XL-2, a novel MLLM that delivers superior cost-effectiveness for long-video understanding based on task-aware KV sparsification. The proposed framework operates with two key steps: chunk-based pre-filling and bi-level key-value decoding. Chunk-based pre-filling divides the visual token sequence into chunks, applying full attention within each chunk and sparse attention across chunks. This significantly reduces computational and memory overhead. During decoding, bi-level key-value decoding selectively reloads either dense or sparse key-values for each chunk based on its relevance to the task. This approach further improves memory efficiency and enhances the model's ability to capture fine-grained information. Video-XL-2 achieves state-of-the-art performance on various long video understanding benchmarks, outperforming existing open-source lightweight models. It also demonstrates exceptional efficiency, capable of processing over 10,000 frames on a single NVIDIA A100 (80GB) GPU and thousands of frames in just a few seconds. 

---
# Private Model Personalization Revisited 

**Authors**: Conor Snedeker, Xinyu Zhou, Raef Bassily  

**Link**: [PDF](https://arxiv.org/pdf/2506.19220)  

**Abstract**: We study model personalization under user-level differential privacy (DP) in the shared representation framework. In this problem, there are $n$ users whose data is statistically heterogeneous, and their optimal parameters share an unknown embedding $U^* \in\mathbb{R}^{d\times k}$ that maps the user parameters in $\mathbb{R}^d$ to low-dimensional representations in $\mathbb{R}^k$, where $k\ll d$. Our goal is to privately recover the shared embedding and the local low-dimensional representations with small excess risk in the federated setting. We propose a private, efficient federated learning algorithm to learn the shared embedding based on the FedRep algorithm in [CHM+21]. Unlike [CHM+21], our algorithm satisfies differential privacy, and our results hold for the case of noisy labels. In contrast to prior work on private model personalization [JRS+21], our utility guarantees hold under a larger class of users' distributions (sub-Gaussian instead of Gaussian distributions). Additionally, in natural parameter regimes, we improve the privacy error term in [JRS+21] by a factor of $\widetilde{O}(dk)$. Next, we consider the binary classification setting. We present an information-theoretic construction to privately learn the shared embedding and derive a margin-based accuracy guarantee that is independent of $d$. Our method utilizes the Johnson-Lindenstrauss transform to reduce the effective dimensions of the shared embedding and the users' data. This result shows that dimension-independent risk bounds are possible in this setting under a margin loss. 

---
# MedErr-CT: A Visual Question Answering Benchmark for Identifying and Correcting Errors in CT Reports 

**Authors**: Sunggu Kyung, Hyungbin Park, Jinyoung Seo, Jimin Sung, Jihyun Kim, Dongyeong Kim, Wooyoung Jo, Yoojin Nam, Sangah Park, Taehee Kwon, Sang Min Lee, Namkug Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.19217)  

**Abstract**: Computed Tomography (CT) plays a crucial role in clinical diagnosis, but the growing demand for CT examinations has raised concerns about diagnostic errors. While Multimodal Large Language Models (MLLMs) demonstrate promising comprehension of medical knowledge, their tendency to produce inaccurate information highlights the need for rigorous validation. However, existing medical visual question answering (VQA) benchmarks primarily focus on simple visual recognition tasks, lacking clinical relevance and failing to assess expert-level knowledge. We introduce MedErr-CT, a novel benchmark for evaluating medical MLLMs' ability to identify and correct errors in CT reports through a VQA framework. The benchmark includes six error categories - four vision-centric errors (Omission, Insertion, Direction, Size) and two lexical error types (Unit, Typo) - and is organized into three task levels: classification, detection, and correction. Using this benchmark, we quantitatively assess the performance of state-of-the-art 3D medical MLLMs, revealing substantial variation in their capabilities across different error types. Our benchmark contributes to the development of more reliable and clinically applicable MLLMs, ultimately helping reduce diagnostic errors and improve accuracy in clinical practice. The code and datasets are available at this https URL. 

---
# Thought Anchors: Which LLM Reasoning Steps Matter? 

**Authors**: Paul C. Bogdan, Uzay Macar, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19143)  

**Abstract**: Reasoning large language models have recently achieved state-of-the-art performance in many fields. However, their long-form chain-of-thought reasoning creates interpretability challenges as each generated token depends on all previous ones, making the computation harder to decompose. We argue that analyzing reasoning traces at the sentence level is a promising approach to understanding reasoning processes. We present three complementary attribution methods: (1) a black-box method measuring each sentence's counterfactual importance by comparing final answers across 100 rollouts conditioned on the model generating that sentence or one with a different meaning; (2) a white-box method of aggregating attention patterns between pairs of sentences, which identified ``broadcasting'' sentences that receive disproportionate attention from all future sentences via ``receiver'' attention heads; (3) a causal attribution method measuring logical connections between sentences by suppressing attention toward one sentence and measuring the effect on each future sentence's tokens. Each method provides evidence for the existence of thought anchors, reasoning steps that have outsized importance and that disproportionately influence the subsequent reasoning process. These thought anchors are typically planning or backtracking sentences. We provide an open-source tool (this http URL) for visualizing the outputs of our methods, and present a case study showing converging patterns across methods that map how a model performs multi-step reasoning. The consistency across methods demonstrates the potential of sentence-level analysis for a deeper understanding of reasoning models. 

---
# Finding Clustering Algorithms in the Transformer Architecture 

**Authors**: Kenneth L. Clarkson, Lior Horesh, Takuya Ito, Charlotte Park, Parikshit Ram  

**Link**: [PDF](https://arxiv.org/pdf/2506.19125)  

**Abstract**: The invention of the transformer architecture has revolutionized Artificial Intelligence (AI), yielding unprecedented success in areas such as natural language processing, computer vision, and multimodal reasoning. Despite these advances, it is unclear whether transformers are able to learn and implement precise algorithms. Here, we demonstrate that transformers can exactly implement a fundamental and widely used algorithm for $k$-means clustering: Lloyd's algorithm. First, we theoretically prove the existence of such a transformer architecture, which we term the $k$-means transformer, that exactly implements Lloyd's algorithm for $k$-means clustering using the standard ingredients of modern transformers: attention and residual connections. Next, we numerically implement this transformer and demonstrate in experiments the exact correspondence between our architecture and Lloyd's algorithm, providing a fully neural implementation of $k$-means clustering. Finally, we demonstrate that interpretable alterations (e.g., incorporating layer normalizations or multilayer perceptrons) to this architecture yields diverse and novel variants of clustering algorithms, such as soft $k$-means, spherical $k$-means, trimmed $k$-means, and more. Collectively, our findings demonstrate how transformer mechanisms can precisely map onto algorithmic procedures, offering a clear and interpretable perspective on implementing precise algorithms in transformers. 

---
# CUPID: Curating Data your Robot Loves with Influence Functions 

**Authors**: Christopher Agia, Rohan Sinha, Jingyun Yang, Rika Antonova, Marco Pavone, Haruki Nishimura, Masha Itkina, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2506.19121)  

**Abstract**: In robot imitation learning, policy performance is tightly coupled with the quality and composition of the demonstration data. Yet, developing a precise understanding of how individual demonstrations contribute to downstream outcomes - such as closed-loop task success or failure - remains a persistent challenge. We propose CUPID, a robot data curation method based on a novel influence function-theoretic formulation for imitation learning policies. Given a set of evaluation rollouts, CUPID estimates the influence of each training demonstration on the policy's expected return. This enables ranking and selection of demonstrations according to their impact on the policy's closed-loop performance. We use CUPID to curate data by 1) filtering out training demonstrations that harm policy performance and 2) subselecting newly collected trajectories that will most improve the policy. Extensive simulated and hardware experiments show that our approach consistently identifies which data drives test-time performance. For example, training with less than 33% of curated data can yield state-of-the-art diffusion policies on the simulated RoboMimic benchmark, with similar gains observed in hardware. Furthermore, hardware experiments show that our method can identify robust strategies under distribution shift, isolate spurious correlations, and even enhance the post-training of generalist robot policies. Additional materials are made available at: this https URL. 

---
# Enhancing Security in LLM Applications: A Performance Evaluation of Early Detection Systems 

**Authors**: Valerii Gakh, Hayretdin Bahsi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19109)  

**Abstract**: Prompt injection threatens novel applications that emerge from adapting LLMs for various user tasks. The newly developed LLM-based software applications become more ubiquitous and diverse. However, the threat of prompt injection attacks undermines the security of these systems as the mitigation and defenses against them, proposed so far, are insufficient. We investigated the capabilities of early prompt injection detection systems, focusing specifically on the detection performance of techniques implemented in various open-source solutions. These solutions are supposed to detect certain types of prompt injection attacks, including the prompt leak. In prompt leakage attacks, an attacker maliciously manipulates the LLM into outputting its system instructions, violating the system's confidentiality. Our study presents analyzes of distinct prompt leakage detection techniques, and a comparative analysis of several detection solutions, which implement those techniques. We identify the strengths and weaknesses of these techniques and elaborate on their optimal configuration and usage in high-stake deployments. In one of the first studies on existing prompt leak detection solutions, we compared the performances of LLM Guard, Vigil, and Rebuff. We concluded that the implementations of canary word checks in Vigil and Rebuff were not effective at detecting prompt leak attacks, and we proposed improvements for them. We also found an evasion weakness in Rebuff's secondary model-based technique and proposed a mitigation. Then, the result of the comparison of LLM Guard, Vigil, and Rebuff at their peak performance revealed that Vigil is optimal for cases when minimal false positive rate is required, and Rebuff is the most optimal for average needs. 

---
# Improving Student-AI Interaction Through Pedagogical Prompting: An Example in Computer Science Education 

**Authors**: Ruiwei Xiao, Xinying Hou, Runlong Ye, Majeed Kazemitabaar, Nicholas Diana, Michael Liut, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2506.19107)  

**Abstract**: With the proliferation of large language model (LLM) applications since 2022, their use in education has sparked both excitement and concern. Recent studies consistently highlight students' (mis)use of LLMs can hinder learning outcomes. This work aims to teach students how to effectively prompt LLMs to improve their learning. We first proposed pedagogical prompting, a theoretically-grounded new concept to elicit learning-oriented responses from LLMs. To move from concept design to a proof-of-concept learning intervention in real educational settings, we selected early undergraduate CS education (CS1/CS2) as the example context. We began with a formative survey study with instructors (N=36) teaching early-stage undergraduate-level CS courses to inform the instructional design based on classroom needs. Based on their insights, we designed and developed a learning intervention through an interactive system with scenario-based instruction to train pedagogical prompting skills. Finally, we evaluated its instructional effectiveness through a user study with CS novice students (N=22) using pre/post-tests. Through mixed methods analyses, our results indicate significant improvements in learners' LLM-based pedagogical help-seeking skills, along with positive attitudes toward the system and increased willingness to use pedagogical prompts in the future. Our contributions include (1) a theoretical framework of pedagogical prompting; (2) empirical insights into current instructor attitudes toward pedagogical prompting; and (3) a learning intervention design with an interactive learning tool and scenario-based instruction leading to promising results on teaching LLM-based help-seeking. Our approach is scalable for broader implementation in classrooms and has the potential to be integrated into tools like ChatGPT as an on-boarding experience to encourage learning-oriented use of generative AI. 

---
# Language Models Might Not Understand You: Evaluating Theory of Mind via Story Prompting 

**Authors**: Nathaniel Getachew, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2506.19089)  

**Abstract**: We introduce $\texttt{StorySim}$, a programmable framework for synthetically generating stories to evaluate the theory of mind (ToM) and world modeling (WM) capabilities of large language models (LLMs). Unlike prior benchmarks that may suffer from contamination in pretraining data, $\texttt{StorySim}$ produces novel, compositional story prompts anchored by a highly controllable $\texttt{Storyboard}$, enabling precise manipulation of character perspectives and events. We use this framework to design first- and second-order ToM tasks alongside WM tasks that control for the ability to track and model mental states. Our experiments across a suite of state-of-the-art LLMs reveal that most models perform better on WM tasks than ToM tasks, and that models tend to perform better reasoning with humans compared to inanimate objects. Additionally, our framework enabled us to find evidence of heuristic behavior such as recency bias and an over-reliance on earlier events in the story. All code for generating data and evaluations is freely available. 

---
# RareSpot: Spotting Small and Rare Wildlife in Aerial Imagery with Multi-Scale Consistency and Context-Aware Augmentation 

**Authors**: Bowen Zhang, Jesse T. Boulerice, Nikhil Kuniyil, Charvi Mendiratta, Satish Kumar, Hila Shamon, B.S. Manjunath  

**Link**: [PDF](https://arxiv.org/pdf/2506.19087)  

**Abstract**: Automated detection of small and rare wildlife in aerial imagery is crucial for effective conservation, yet remains a significant technical challenge. Prairie dogs exemplify this issue: their ecological importance as keystone species contrasts sharply with their elusive presence--marked by small size, sparse distribution, and subtle visual features--which undermines existing detection approaches. To address these challenges, we propose RareSpot, a robust detection framework integrating multi-scale consistency learning and context-aware augmentation. Our multi-scale consistency approach leverages structured alignment across feature pyramids, enhancing fine-grained object representation and mitigating scale-related feature loss. Complementarily, context-aware augmentation strategically synthesizes challenging training instances by embedding difficult-to-detect samples into realistic environmental contexts, significantly boosting model precision and recall. Evaluated on an expert-annotated prairie dog drone imagery benchmark, our method achieves state-of-the-art performance, improving detection accuracy by over 35% compared to baseline methods. Importantly, it generalizes effectively across additional wildlife datasets, demonstrating broad applicability. The RareSpot benchmark and approach not only support critical ecological monitoring but also establish a new foundation for detecting small, rare species in complex aerial scenes. 

---
# FairCauseSyn: Towards Causally Fair LLM-Augmented Synthetic Data Generation 

**Authors**: Nitish Nagesh, Ziyu Wang, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2506.19082)  

**Abstract**: Synthetic data generation creates data based on real-world data using generative models. In health applications, generating high-quality data while maintaining fairness for sensitive attributes is essential for equitable outcomes. Existing GAN-based and LLM-based methods focus on counterfactual fairness and are primarily applied in finance and legal domains. Causal fairness provides a more comprehensive evaluation framework by preserving causal structure, but current synthetic data generation methods do not address it in health settings. To fill this gap, we develop the first LLM-augmented synthetic data generation method to enhance causal fairness using real-world tabular health data. Our generated data deviates by less than 10% from real data on causal fairness metrics. When trained on causally fair predictors, synthetic data reduces bias on the sensitive attribute by 70% compared to real data. This work improves access to fair synthetic data, supporting equitable health research and healthcare delivery. 

---
# Reading Smiles: Proxy Bias in Foundation Models for Facial Emotion Recognition 

**Authors**: Iosif Tsangko, Andreas Triantafyllopoulos, Adem Abdelmoula, Adria Mallol-Ragolta, Bjoern W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2506.19079)  

**Abstract**: Foundation Models (FMs) are rapidly transforming Affective Computing (AC), with Vision Language Models (VLMs) now capable of recognising emotions in zero shot settings. This paper probes a critical but underexplored question: what visual cues do these models rely on to infer affect, and are these cues psychologically grounded or superficially learnt? We benchmark varying scale VLMs on a teeth annotated subset of AffectNet dataset and find consistent performance shifts depending on the presence of visible teeth. Through structured introspection of, the best-performing model, i.e., GPT-4o, we show that facial attributes like eyebrow position drive much of its affective reasoning, revealing a high degree of internal consistency in its valence-arousal predictions. These patterns highlight the emergent nature of FMs behaviour, but also reveal risks: shortcut learning, bias, and fairness issues especially in sensitive domains like mental health and education. 

---
# HAWAII: Hierarchical Visual Knowledge Transfer for Efficient Vision-Language Models 

**Authors**: Yimu Wang, Mozhgan Nasr Azadani, Sean Sedwards, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2506.19072)  

**Abstract**: Improving the visual understanding ability of vision-language models (VLMs) is crucial for enhancing their performance across various tasks. While using multiple pretrained visual experts has shown great promise, it often incurs significant computational costs during training and inference. To address this challenge, we propose HAWAII, a novel framework that distills knowledge from multiple visual experts into a single vision encoder, enabling it to inherit the complementary strengths of several experts with minimal computational overhead. To mitigate conflicts among different teachers and switch between different teacher-specific knowledge, instead of using a fixed set of adapters for multiple teachers, we propose to use teacher-specific Low-Rank Adaptation (LoRA) adapters with a corresponding router. Each adapter is aligned with a specific teacher, avoiding noisy guidance during distillation. To enable efficient knowledge distillation, we propose fine-grained and coarse-grained distillation. At the fine-grained level, token importance scores are employed to emphasize the most informative tokens from each teacher adaptively. At the coarse-grained level, we summarize the knowledge from multiple teachers and transfer it to the student using a set of general-knowledge LoRA adapters with a router. Extensive experiments on various vision-language tasks demonstrate the superiority of HAWAII, compared to the popular open-source VLMs. 

---
# Plan for Speed -- Dilated Scheduling for Masked Diffusion Language Models 

**Authors**: Omer Luxembourg, Haim Permuter, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2506.19037)  

**Abstract**: Masked diffusion language models (MDLM) have shown strong promise for non-autoregressive text generation, yet existing samplers act as implicit planners, selecting tokens to unmask via denoiser confidence or entropy scores. Such heuristics falter under parallel unmasking - they ignore pairwise interactions between tokens and cannot account for dependencies when unmasking multiple positions at once, limiting their inference time to traditional auto-regressive (AR) models. We introduce the Dilated-scheduled Unmasking Strategy (DUS), an inference-only, planner-model-free method that requires no additional training. DUS leverages a first-order Markov assumption to partition sequence positions into dilation-based groups of non-adjacent tokens, enabling independent, parallel unmasking steps that respect local context that minimizes the joint entropy of each iteration step. Unlike semi-AR block approaches (e.g., LLADA and Dream) that still invoke the denoiser per block, DUS reduces the number of denoiser calls to O(log B) per generation block - yielding substantial speedup over the O(B) run time of state-of-the-art diffusion models, where B is the block size in the semi-AR inference process. In experiments on math (GSM8K) and code completion (Humaneval, MBPP) benchmarks - domains suited to non-ordinal generation - DUS improves scores over parallel confidence-based planner, without modifying the underlying denoiser. DUS offers a lightweight, budget-aware approach to efficient, high-quality text generation, paving the way to unlock the true capabilities of MDLMs. 

---
# Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective 

**Authors**: Weijie Xu, Yiwen Wang, Chi Xue, Xiangkun Hu, Xi Fang, Guimin Dong, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19028)  

**Abstract**: Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics. 

---
# Statistical Inference for Optimal Transport Maps: Recent Advances and Perspectives 

**Authors**: Sivaraman Balakrishnan, Tudor Manole, Larry Wasserman  

**Link**: [PDF](https://arxiv.org/pdf/2506.19025)  

**Abstract**: In many applications of optimal transport (OT), the object of primary interest is the optimal transport map. This map rearranges mass from one probability distribution to another in the most efficient way possible by minimizing a specified cost. In this paper we review recent advances in estimating and developing limit theorems for the OT map, using samples from the underlying distributions. We also review parallel lines of work that establish similar results for special cases and variants of the basic OT setup. We conclude with a discussion of key directions for future research with the goal of providing practitioners with reliable inferential tools. 

---
# Survey of HPC in US Research Institutions 

**Authors**: Peng Shu, Junhao Chen, Zhengliang Liu, Huaqin Zhao, Xinliang Li, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19019)  

**Abstract**: The rapid growth of AI, data-intensive science, and digital twin technologies has driven an unprecedented demand for high-performance computing (HPC) across the research ecosystem. While national laboratories and industrial hyperscalers have invested heavily in exascale and GPU-centric architectures, university-operated HPC systems remain comparatively under-resourced. This survey presents a comprehensive assessment of the HPC landscape across U.S. universities, benchmarking their capabilities against Department of Energy (DOE) leadership-class systems and industrial AI infrastructures. We examine over 50 premier research institutions, analyzing compute capacity, architectural design, governance models, and energy efficiency. Our findings reveal that university clusters, though vital for academic research, exhibit significantly lower growth trajectories (CAGR $\approx$ 18%) than their national ($\approx$ 43%) and industrial ($\approx$ 78%) counterparts. The increasing skew toward GPU-dense AI workloads has widened the capability gap, highlighting the need for federated computing, idle-GPU harvesting, and cost-sharing models. We also identify emerging paradigms, such as decentralized reinforcement learning, as promising opportunities for democratizing AI training within campus environments. Ultimately, this work provides actionable insights for academic leaders, funding agencies, and technology partners to ensure more equitable and sustainable HPC access in support of national research priorities. 

---
# IndieFake Dataset: A Benchmark Dataset for Audio Deepfake Detection 

**Authors**: Abhay Kumar, Kunal Verma, Omkar More  

**Link**: [PDF](https://arxiv.org/pdf/2506.19014)  

**Abstract**: Advancements in audio deepfake technology offers benefits like AI assistants, better accessibility for speech impairments, and enhanced entertainment. However, it also poses significant risks to security, privacy, and trust in digital communications. Detecting and mitigating these threats requires comprehensive datasets. Existing datasets lack diverse ethnic accents, making them inadequate for many real-world scenarios. Consequently, models trained on these datasets struggle to detect audio deepfakes in diverse linguistic and cultural contexts such as in South-Asian countries. Ironically, there is a stark lack of South-Asian speaker samples in the existing datasets despite constituting a quarter of the worlds population. This work introduces the IndieFake Dataset (IFD), featuring 27.17 hours of bonafide and deepfake audio from 50 English speaking Indian speakers. IFD offers balanced data distribution and includes speaker-level characterization, absent in datasets like ASVspoof21 (DF). We evaluated various baselines on IFD against existing ASVspoof21 (DF) and In-The-Wild (ITW) datasets. IFD outperforms ASVspoof21 (DF) and proves to be more challenging compared to benchmark ITW dataset. The dataset will be publicly available upon acceptance. 

---
# GLIMPSE: Gradient-Layer Importance Mapping for Prompted Visual Saliency Explanation for Generative LVLMs 

**Authors**: Guanxi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18985)  

**Abstract**: Recent advances in large vision language models (LVLMs) have unlocked unprecedented capabilities in generating coherent responses from visual inputs. However, interpreting where LVLMs direct their visual attention while generating free-form textual responses remains a significant challenge, yet is essential for understanding model behavior, diagnosing hallucination, exposing bias and ensuring transparency. We introduce GLIMPSE (Gradient-Layer Importance Mapping for Prompted Visual Saliency Explanation), a lightweight, model-agnostic framework for visualizing the salient image regions that LVLMs rely upon during open-ended visual question answering (VQA), while concurrently revealing the multimodal textual saliency. GLIMPSE fuses gradient-weighted attention, adaptive layer propagation, and weighted token aggregation to produce holistic response-level attribution heat maps for interpreting cross-modal reasoning, outperforming prior interpretability methods in human-alignment. We demonstrate an analytic explainable AI (XAI) approach using GLIMPSE to uncover fine-grained insights into LVLM cross-modal attribution, trace token-level reasoning dynamics, and analyze systematic human-attention misalignment, hallucination, and bias. 

---
# Citizenship Challenges in Artificial Intelligence Education 

**Authors**: Margarida Romero  

**Link**: [PDF](https://arxiv.org/pdf/2506.18955)  

**Abstract**: This chapter addresses the citizenship challenges related to AI in education, particularly concerning students, teachers, and other educational stakeholders in the context of AI integration. We first explore how to foster AI awareness and education, along with various strategies to promote a socio-critical approach to AI training, aiming to identify relevant and ethical uses to prioritise. In the second part, we discuss critical thinking and computational thinking skills that can be mobilised within certain AI-supported educational activities, depending on the degree of creative and transformative engagement those activities require. 

---
# SHAMaNS: Sound Localization with Hybrid Alpha-Stable Spatial Measure and Neural Steerer 

**Authors**: Diego Di Carlo, Mathieu Fontaine, Aditya Arie Nugraha, Yoshiaki Bando, Kazuyoshi Yoshii  

**Link**: [PDF](https://arxiv.org/pdf/2506.18954)  

**Abstract**: This paper describes a sound source localization (SSL) technique that combines an $\alpha$-stable model for the observed signal with a neural network-based approach for modeling steering vectors. Specifically, a physics-informed neural network, referred to as Neural Steerer, is used to interpolate measured steering vectors (SVs) on a fixed microphone array. This allows for a more robust estimation of the so-called $\alpha$-stable spatial measure, which represents the most plausible direction of arrival (DOA) of a target signal. As an $\alpha$-stable model for the non-Gaussian case ($\alpha$ $\in$ (0, 2)) theoretically defines a unique spatial measure, we choose to leverage it to account for residual reconstruction error of the Neural Steerer in the downstream tasks. The objective scores indicate that our proposed technique outperforms state-of-the-art methods in the case of multiple sound sources. 

---
# LLMs on a Budget? Say HOLA 

**Authors**: Zohaib Hasan Siddiqui, Jiechao Gao, Ebad Shabbir, Mohammad Anas Azeez, Rafiq Ali, Gautam Siddharth Kashyap, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.18952)  

**Abstract**: Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded systems. Current solutions such as quantization, pruning, and retrieval-augmented generation (RAG) offer only partial optimizations and often compromise on speed or accuracy. We introduce HOLA, an end-to-end optimization framework for efficient LLM deployment. Internally, it leverages Hierarchical Speculative Decoding (HSD) for faster inference without quality loss. Externally, AdaComp-RAG adjusts retrieval complexity based on context needs. Together with LoBi, which blends structured pruning (LoRA) and quantization, HOLA delivers significant gains: 17.6% EMA on GSM8K, 10.5% MCA on ARC, and reduced latency and memory on edge devices like Jetson Nano--proving both scalable and production-ready. 

---
# SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications 

**Authors**: Jinyang Li, Xiaolong Li, Ge Qu, Per Jacobsson, Bowen Qin, Binyuan Hui, Shuzheng Si, Nan Huo, Xiaohan Xu, Yue Zhang, Ziwei Tang, Yuanshuai Li, Florensia Widjaja, Xintong Zhu, Feige Zhou, Yongfeng Huang, Yannis Papakonstantinou, Fatma Ozcan, Chenhao Ma, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18951)  

**Abstract**: Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: this https URL 

---
# DiffRIS: Enhancing Referring Remote Sensing Image Segmentation with Pre-trained Text-to-Image Diffusion Models 

**Authors**: Zhe Dong, Yuzhe Sun, Tianzhu Liu, Yanfeng Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18946)  

**Abstract**: Referring remote sensing image segmentation (RRSIS) enables the precise delineation of regions within remote sensing imagery through natural language descriptions, serving critical applications in disaster response, urban development, and environmental monitoring. Despite recent advances, current approaches face significant challenges in processing aerial imagery due to complex object characteristics including scale variations, diverse orientations, and semantic ambiguities inherent to the overhead perspective. To address these limitations, we propose DiffRIS, a novel framework that harnesses the semantic understanding capabilities of pre-trained text-to-image diffusion models for enhanced cross-modal alignment in RRSIS tasks. Our framework introduces two key innovations: a context perception adapter (CP-adapter) that dynamically refines linguistic features through global context modeling and object-aware reasoning, and a progressive cross-modal reasoning decoder (PCMRD) that iteratively aligns textual descriptions with visual regions for precise segmentation. The CP-adapter bridges the domain gap between general vision-language understanding and remote sensing applications, while PCMRD enables fine-grained semantic alignment through multi-scale feature interaction. Comprehensive experiments on three benchmark datasets-RRSIS-D, RefSegRS, and RISBench-demonstrate that DiffRIS consistently outperforms existing methods across all standard metrics, establishing a new state-of-the-art for RRSIS tasks. The significant performance improvements validate the effectiveness of leveraging pre-trained diffusion models for remote sensing applications through our proposed adaptive framework. 

---
# Can AI support student engagement in classroom activities in higher education? 

**Authors**: Neha Rani, Sharan Majumder, Ishan Bhardwaj, Pedro Guillermo Feijoo Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2506.18941)  

**Abstract**: Lucrative career prospects and creative opportunities often attract students to enroll in computer science majors and pursue advanced studies in the field. Consequently, there has been a significant surge in enrollment in computer science courses, resulting in large class sizes that can range from hundreds to even thousands of students. A common challenge in such large classrooms is the lack of engagement between students and both the instructor and the learning material. However, with advancements in technology and improvements in large language models (LLMs), there is a considerable opportunity to utilize LLM-based AI models, such as conversational artificial intelligence (CAI), to enhance student engagement with learning content in large classes. To explore the potential of CAI to support engagement, especially with learning content, we designed an activity in a software Engineering course (with a large class size) where students used CAI for an in-class activity. We conducted a within-subject investigation in a large classroom at a US university where we compared student engagement during an in-class activity that used CAI tool vs. one without CAI tool. The CAI tool we used was ChatGPT due to its widespread popularity and familiarity. Our results indicate that CAI (ChatGPT) has the potential to support engagement with learning content during in-class activities, especially in large class sizes. We further discuss the implications of our findings. 

---
# eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis 

**Authors**: Zhenke Liu, Jien Li, Ziqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18940)  

**Abstract**: Extrachromosomal circular DNA (eccDNA) plays key regulatory roles and contributes to oncogene overexpression in cancer through high-copy amplification and long-range interactions. Despite advances in modeling, no pre-trained models currently support full-length circular eccDNA for downstream analysis. Existing genomic models are either limited to single-nucleotide resolution or hindered by the inefficiency of the quadratic attention mechanism. Here, we introduce eccDNAMamba, the first bidirectional state-space encoder tailored for circular DNA sequences. It combines forward and reverse passes for full-context representation learning with linear-time complexity, and preserves circular structure through a novel augmentation strategy. Tested on two real-world datasets, eccDNAMamba achieves strong classification performance and scales to sequences up to 200 Kbp, offering a robust and efficient framework for modeling circular genomes. Our codes are available at this https URL. 

---
# Damba-ST: Domain-Adaptive Mamba for Efficient Urban Spatio-Temporal Prediction 

**Authors**: Rui An, Yifeng Zhang, Ziran Liang, Wenqi Fan, Yuxuan Liang, Xuequn Shang, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18939)  

**Abstract**: Training urban spatio-temporal foundation models that generalize well across diverse regions and cities is critical for deploying urban services in unseen or data-scarce regions. Recent studies have typically focused on fusing cross-domain spatio-temporal data to train unified Transformer-based models. However, these models suffer from quadratic computational complexity and high memory overhead, limiting their scalability and practical deployment. Inspired by the efficiency of Mamba, a state space model with linear time complexity, we explore its potential for efficient urban spatio-temporal prediction. However, directly applying Mamba as a spatio-temporal backbone leads to negative transfer and severe performance degradation. This is primarily due to spatio-temporal heterogeneity and the recursive mechanism of Mamba's hidden state updates, which limit cross-domain generalization. To overcome these challenges, we propose Damba-ST, a novel domain-adaptive Mamba-based model for efficient urban spatio-temporal prediction. Damba-ST retains Mamba's linear complexity advantage while significantly enhancing its adaptability to heterogeneous domains. Specifically, we introduce two core innovations: (1) a domain-adaptive state space model that partitions the latent representation space into a shared subspace for learning cross-domain commonalities and independent, domain-specific subspaces for capturing intra-domain discriminative features; (2) three distinct Domain Adapters, which serve as domain-aware proxies to bridge disparate domain distributions and facilitate the alignment of cross-domain commonalities. Extensive experiments demonstrate the generalization and efficiency of Damba-ST. It achieves state-of-the-art performance on prediction tasks and demonstrates strong zero-shot generalization, enabling seamless deployment in new urban environments without extensive retraining or fine-tuning. 

---
# Which Consciousness Can Be Artificialized? Local Percept-Perceiver Phenomenon for the Existence of Machine Consciousness 

**Authors**: Shri Lal Raghudev Ram Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.18935)  

**Abstract**: This paper presents a novel paradigm of the local percept-perceiver phenomenon to formalize certain observations in neuroscientific theories of consciousness. Using this model, a set-theoretic formalism is developed for artificial systems, and the existence of machine consciousness is proved by invoking Zermelo-Fraenkel set theory. The article argues for the possibility of a reductionist form of epistemic consciousness within machines. 

---
# AI Safety vs. AI Security: Demystifying the Distinction and Boundaries 

**Authors**: Zhiqiang Lin, Huan Sun, Ness Shroff  

**Link**: [PDF](https://arxiv.org/pdf/2506.18932)  

**Abstract**: Artificial Intelligence (AI) is rapidly being integrated into critical systems across various domains, from healthcare to autonomous vehicles. While its integration brings immense benefits, it also introduces significant risks, including those arising from AI misuse. Within the discourse on managing these risks, the terms "AI Safety" and "AI Security" are often used, sometimes interchangeably, resulting in conceptual confusion. This paper aims to demystify the distinction and delineate the precise research boundaries between AI Safety and AI Security. We provide rigorous definitions, outline their respective research focuses, and explore their interdependency, including how security breaches can precipitate safety failures and vice versa. Using clear analogies from message transmission and building construction, we illustrate these distinctions. Clarifying these boundaries is crucial for guiding precise research directions, fostering effective cross-disciplinary collaboration, enhancing policy effectiveness, and ultimately, promoting the deployment of trustworthy AI systems. 

---
# Safe Pruning LoRA: Robust Distance-Guided Pruning for Safety Alignment in Adaptation of LLMs 

**Authors**: Shuang Ao, Yi Dong, Jinwei Hu, Sarvapali Ramchurn  

**Link**: [PDF](https://arxiv.org/pdf/2506.18931)  

**Abstract**: Fine-tuning Large Language Models (LLMs) with Low-Rank Adaptation (LoRA) enhances adaptability while reducing computational costs. However, fine-tuning can compromise safety alignment, even with benign data, increasing susceptibility to harmful outputs. Existing safety alignment methods struggle to capture complex parameter shifts, leading to suboptimal safety-utility trade-offs. To address this issue, we propose Safe Pruning LoRA (SPLoRA), a novel pruning-based approach that selectively removes LoRA layers that weaken safety alignment, improving safety while preserving performance. At its core, we introduce Empirical-DIEM (E-DIEM), a dimension-insensitive similarity metric that effectively detects safety misalignment in LoRA-adapted models. We conduct extensive experiments on LLMs fine-tuned with mixed of benign and malicious data, and purely benign datasets, evaluating SPLoRA across utility, safety, and reliability metrics. Results demonstrate that SPLoRA outperforms state-of-the-art safety alignment techniques, significantly reducing safety risks while maintaining or improving model performance and reliability. Additionally, SPLoRA reduces inference overhead, making it a scalable and efficient solution for deploying safer and more reliable LLMs. The code is available at this https URL. 

---
# Reinforcement Learning-Based Dynamic Grouping for Tubular Structure Tracking 

**Authors**: Chong Di, Shuwang Zhou, Da Chen, Jean-Marie Mirebeau, Minglei Shu, Laurent D. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18930)  

**Abstract**: The computation of minimal paths for the applications in tracking tubular structures such as blood vessels and roads is challenged by complex morphologies and environmental variations. Existing approaches can be roughly categorized into two research lines: the point-wise based models and the segment-wise based models. Although segment-wise approaches have obtained promising results in many scenarios, they often suffer from computational inefficiency and heavily rely on a prescribed prior to fit the target elongated shapes. We propose a novel framework that casts segment-wise tracking as a Markov Decision Process (MDP), enabling a reinforcement learning approach. Our method leverages Q-Learning to dynamically explore a graph of segments, computing edge weights on-demand and adaptively expanding the search space. This strategy avoids the high cost of a pre-computed graph and proves robust to incomplete initial information. Experimental reuslts on typical tubular structure datasets demonstrate that our method significantly outperforms state-of-the-art point-wise and segment-wise approaches. The proposed method effectively handles complex topologies and maintains global path coherence without depending on extensive prior structural knowledge. 

---
# AI-based Approach in Early Warning Systems: Focus on Emergency Communication Ecosystem and Citizen Participation in Nordic Countries 

**Authors**: Fuzel Shaik, Getnet Demil, Mourad Oussalah  

**Link**: [PDF](https://arxiv.org/pdf/2506.18926)  

**Abstract**: Climate change and natural disasters are recognized as worldwide challenges requiring complex and efficient ecosystems to deal with social, economic, and environmental effects. This chapter advocates a holistic approach, distinguishing preparedness, emergency responses, and postcrisis phases. The role of the Early Warning System (EWS), Risk modeling and mitigation measures are particularly emphasized. The chapter reviews the various Artificial Intelligence (AI)-enabler technologies that can be leveraged at each phase, focusing on the INFORM risk framework and EWSs. Emergency communication and psychological risk perception have been emphasized in emergency response times. Finally, a set of case studies from Nordic countries has been highlighted. 

---
# Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger Tapping Test in Parkinson Disease 

**Authors**: Tahereh Zarrat Ehsan, Michael Tangermann, Yağmur Güçlütürk, Bastiaan R. Bloem, Luc J. W. Evers  

**Link**: [PDF](https://arxiv.org/pdf/2506.18925)  

**Abstract**: Accurately quantifying motor characteristics in Parkinson disease (PD) is crucial for monitoring disease progression and optimizing treatment strategies. The finger-tapping test is a standard motor assessment. Clinicians visually evaluate a patient's tapping performance and assign an overall severity score based on tapping amplitude, speed, and irregularity. However, this subjective evaluation is prone to inter- and intra-rater variability, and does not offer insights into individual motor characteristics captured during this test. This paper introduces a granular computer vision-based method for quantifying PD motor characteristics from video recordings. Four sets of clinically relevant features are proposed to characterize hypokinesia, bradykinesia, sequence effect, and hesitation-halts. We evaluate our approach on video recordings and clinical evaluations of 74 PD patients from the Personalized Parkinson Project. Principal component analysis with varimax rotation shows that the video-based features corresponded to the four deficits. Additionally, video-based analysis has allowed us to identify further granular distinctions within sequence effect and hesitation-halts deficits. In the following, we have used these features to train machine learning classifiers to estimate the Movement Disorder Society Unified Parkinson Disease Rating Scale (MDS-UPDRS) finger-tapping score. Compared to state-of-the-art approaches, our method achieves a higher accuracy in MDS-UPDRS score prediction, while still providing an interpretable quantification of individual finger-tapping motor characteristics. In summary, the proposed framework provides a practical solution for the objective assessment of PD motor characteristics, that can potentially be applied in both clinical and remote settings. Future work is needed to assess its responsiveness to symptomatic treatment and disease progression. 

---
# Connecting Vision and Emissions: A Behavioural AI Approach to Carbon Estimation in Road Design 

**Authors**: Ammar K Al Mhdawi, Nonso Nnamoko, Safanah Mudheher Raafat, M.K.S. Al-Mhdawi, Amjad J Humaidi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18924)  

**Abstract**: We present an enhanced YOLOv8 real time vehicle detection and classification framework, for estimating carbon emissions in urban environments. The system enhances YOLOv8 architecture to detect, segment, and track vehicles from live traffic video streams. Once a vehicle is localized, a dedicated deep learning-based identification module is employed to recognize license plates and classify vehicle types. Since YOLOv8 lacks the built-in capacity for fine grained recognition tasks such as reading license plates or determining vehicle attributes beyond class labels, our framework incorporates a hybrid pipeline where each detected vehicle is tracked and its bounding box is cropped and passed to a deep Optical Character Recognition (OCR) module. This OCR system, composed of multiple convolutional neural network (CNN) layers, is trained specifically for character-level detection and license plate decoding under varied conditions such as motion blur, occlusion, and diverse font styles. Additionally, the recognized plate information is validated using a real time API that cross references with an external vehicle registration database to ensure accurate classification and emission estimation. This multi-stage approach enables precise, automated calculation of per vehicle carbon emissions. Extensive evaluation was conducted using a diverse vehicle dataset enriched with segmentation masks and annotated license plates. The YOLOv8 detector achieved a mean Average Precision (mAP@0.5) of approximately 71% for bounding boxes and 70% for segmentation masks. Character level OCR accuracy reached up to 99% with the best performing CNN model. These results affirm the feasibility of combining real time object detection with deep OCR for practical deployment in smart transportation systems, offering a scalable solution for automated, vehicle specific carbon emission monitoring. 

---
# MemeMind: A Large-Scale Multimodal Dataset with Chain-of-Thought Reasoning for Harmful Meme Detection 

**Authors**: Hexiang Gu, Qifan Yu, Saihui Hou, Zhiqin Fang, Huijia Wu, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2506.18919)  

**Abstract**: The rapid development of social media has intensified the spread of harmful content. Harmful memes, which integrate both images and text, pose significant challenges for automated detection due to their implicit semantics and complex multimodal interactions. Although existing research has made progress in detection accuracy and interpretability, the lack of a systematic, large-scale, diverse, and highly explainable dataset continues to hinder further advancement in this field. To address this gap, we introduce MemeMind, a novel dataset featuring scientifically rigorous standards, large scale, diversity, bilingual support (Chinese and English), and detailed Chain-of-Thought (CoT) annotations. MemeMind fills critical gaps in current datasets by offering comprehensive labeling and explicit reasoning traces, thereby providing a solid foundation for enhancing harmful meme detection. In addition, we propose an innovative detection framework, MemeGuard, which effectively integrates multimodal information with reasoning process modeling, significantly improving models' ability to understand and identify harmful memes. Extensive experiments conducted on the MemeMind dataset demonstrate that MemeGuard consistently outperforms existing state-of-the-art methods in harmful meme detection tasks. 

---
# Automatic Depression Assessment using Machine Learning: A Comprehensive Survey 

**Authors**: Siyang Song, Yupeng Huo, Shiqing Tang, Jiaee Cheong, Rui Gao, Michel Valstar, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2506.18915)  

**Abstract**: Depression is a common mental illness across current human society. Traditional depression assessment relying on inventories and interviews with psychologists frequently suffer from subjective diagnosis results, slow and expensive diagnosis process as well as lack of human resources. Since there is a solid evidence that depression is reflected by various human internal brain activities and external expressive behaviours, early traditional machine learning (ML) and advanced deep learning (DL) models have been widely explored for human behaviour-based automatic depression assessment (ADA) since 2012. However, recent ADA surveys typically only focus on a limited number of human behaviour modalities. Despite being used as a theoretical basis for developing ADA approaches, existing ADA surveys lack a comprehensive review and summary of multi-modal depression-related human behaviours. To bridge this gap, this paper specifically summarises depression-related human behaviours across a range of modalities (e.g. the human brain, verbal language and non-verbal audio/facial/body behaviours). We focus on conducting an up-to-date and comprehensive survey of ML-based ADA approaches for learning depression cues from these behaviours as well as discussing and comparing their distinctive features and limitations. In addition, we also review existing ADA competitions and datasets, identify and discuss the main challenges and opportunities to provide further research directions for future ADA researchers. 

---
# Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases 

**Authors**: Yubeen Bae, Minchan Kim, Jaejin Lee, Sangbum Kim, Jaehyung Kim, Yejin Choi, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2506.17336)  

**Abstract**: Large language models (LLMs) are increasingly used as personal agents, accessing sensitive user data such as calendars, emails, and medical records. Users currently face a trade-off: They can send private records, many of which are stored in remote databases, to powerful but untrusted LLM providers, increasing their exposure risk. Alternatively, they can run less powerful models locally on trusted devices. We bridge this gap. Our Socratic Chain-of-Thought Reasoning first sends a generic, non-private user query to a powerful, untrusted LLM, which generates a Chain-of-Thought (CoT) prompt and detailed sub-queries without accessing user data. Next, we embed these sub-queries and perform encrypted sub-second semantic search using our Homomorphically Encrypted Vector Database across one million entries of a single user's private data. This represents a realistic scale of personal documents, emails, and records accumulated over years of digital activity. Finally, we feed the CoT prompt and the decrypted records to a local language model and generate the final response. On the LoCoMo long-context QA benchmark, our hybrid framework, combining GPT-4o with a local Llama-3.2-1B model, outperforms using GPT-4o alone by up to 7.1 percentage points. This demonstrates a first step toward systems where tasks are decomposed and split between untrusted strong LLMs and weak local ones, preserving user privacy. 

---
# Neural Cellular Automata for ARC-AGI 

**Authors**: Kevin Xu, Risto Miikkulainen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15746)  

**Abstract**: Cellular automata and their differentiable counterparts, Neural Cellular Automata (NCA), are highly expressive and capable of surprisingly complex behaviors. This paper explores how NCAs perform when applied to tasks requiring precise transformations and few-shot generalization, using the Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) as a domain that challenges their capabilities in ways not previously explored. Specifically, this paper uses gradient-based training to learn iterative update rules that transform input grids into their outputs from the training examples and apply them to the test inputs. Results suggest that gradient-trained NCA models are a promising and efficient approach to a range of abstract grid-based tasks from ARC. Along with discussing the impacts of various design modifications and training constraints, this work examines the behavior and properties of NCAs applied to ARC to give insights for broader applications of self-organizing systems. 

---
# Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models 

**Authors**: Thao Nguyen, Yang Li, Olga Golovneva, Luke Zettlemoyer, Sewoong Oh, Ludwig Schmidt, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04689)  

**Abstract**: Scaling laws predict that the performance of large language models improves with increasing model size and data size. In practice, pre-training has been relying on massive web crawls, using almost all data sources publicly available on the internet so far. However, this pool of natural data does not grow at the same rate as the compute supply. Furthermore, the availability of high-quality texts is even more limited: data filtering pipelines often remove up to 99% of the initial web scrapes to achieve state-of-the-art. To address the "data wall" of pre-training scaling, our work explores ways to transform and recycle data discarded in existing filtering processes. We propose REWIRE, REcycling the Web with guIded REwrite, a method to enrich low-quality documents so that they could become useful for training. This in turn allows us to increase the representation of synthetic data in the final pre-training set. Experiments at 1B, 3B and 7B scales of the DCLM benchmark show that mixing high-quality raw texts and our rewritten texts lead to 1.0, 1.3 and 2.5 percentage points improvement respectively across 22 diverse tasks, compared to training on only filtered web data. Training on the raw-synthetic data mix is also more effective than having access to 2x web data. Through further analysis, we demonstrate that about 82% of the mixed in texts come from transforming lower-quality documents that would otherwise be discarded. REWIRE also outperforms related approaches of generating synthetic data, including Wikipedia-style paraphrasing, question-answer synthesizing and knowledge extraction. These results suggest that recycling web texts holds the potential for being a simple and effective approach for scaling pre-training data. 

---
