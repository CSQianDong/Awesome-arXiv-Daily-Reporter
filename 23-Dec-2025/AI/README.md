# Scalably Enhancing the Clinical Validity of a Task Benchmark with Physician Oversight 

**Authors**: Junze Ye, Daniel Tawfik, Alex J. Goodell, Nikhil V. Kotha, Mark K. Buyyounouski, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2512.19691)  

**Abstract**: Automating the calculation of clinical risk scores offers a significant opportunity to reduce physician administrative burden and enhance patient care. The current standard for evaluating this capability is MedCalc-Bench, a large-scale dataset constructed using LLM-based feature extraction and rule-based aggregation. However, treating such model-generated benchmarks as static oracles risks enshrining historical model errors as evaluation gold standards, a problem dangerously amplified when these datasets serve as reward signals for Reinforcement Learning (RL). In this work, we propose viewing benchmarks for complex tasks such as clinical score computation as ''in-progress living documents'' that should be periodically re-evaluated as the processes for creating them improve. We introduce a systematic, physician-in-the-loop pipeline that leverages advanced agentic verifiers to audit and relabel MedCalc-Bench, utilizing automated triage to reserve scarce clinician attention for the most contentious instances. Our audit reveals that a notable fraction of original labels diverge from medical ground truth due to extraction errors, calculator logic mismatches, and clinical ambiguity. To study whether this label noise meaningfully impacts downstream RL training, we fine-tune a Qwen3-8B model via Group Relative Policy Optimization (GRPO) and demonstrate that training on corrected labels yields an 8.7% absolute improvement in accuracy over the original baseline -- validating that label noise materially affects model evaluation. These findings underscore that in safety-critical domains, rigorous benchmark maintenance is a prerequisite for genuine model alignment. 

---
# Augmenting Intelligence: A Hybrid Framework for Scalable and Stable Explanations 

**Authors**: Lawrence Krukrubo, Julius Odede, Olawande Olusegun  

**Link**: [PDF](https://arxiv.org/pdf/2512.19557)  

**Abstract**: Current approaches to Explainable AI (XAI) face a "Scalability-Stability Dilemma." Post-hoc methods (e.g., LIME, SHAP) may scale easily but suffer from instability, while supervised explanation frameworks (e.g., TED) offer stability but require prohibitive human effort to label every training instance. This paper proposes a Hybrid LRR-TED framework that addresses this dilemma through a novel "Asymmetry of Discovery." When applied to customer churn prediction, we demonstrate that automated rule learners (GLRM) excel at identifying broad "Safety Nets" (retention patterns) but struggle to capture specific "Risk Traps" (churn triggers)-a phenomenon we term the Anna Karenina Principle of Churn. By initialising the explanation matrix with automated safety rules and augmenting it with a Pareto-optimal set of just four human-defined risk rules, our approach achieves 94.00% predictive accuracy. This configuration outperforms the full 8-rule manual expert baseline while reducing human annotation effort by 50%, proposing a shift in the paradigm for Human-in-the-Loop AI: moving experts from the role of "Rule Writers" to "Exception Handlers." 

---
# Towards Closed-Loop Embodied Empathy Evolution: Probing LLM-Centric Lifelong Empathic Motion Generation in Unseen Scenarios 

**Authors**: Jiawen Wang, Jingjing Wang Tianyang Chen, Min Zhang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.19551)  

**Abstract**: In the literature, existing human-centric emotional motion generation methods primarily focus on boosting performance within a single scale-fixed dataset, largely neglecting the flexible and scale-increasing motion scenarios (e.g., sports, dance), whereas effectively learning these newly emerging scenarios can significantly enhance the model's real-world generalization ability. Inspired by this, this paper proposes a new LLM-Centric Lifelong Empathic Motion Generation (L^2-EMG) task, which aims to equip LLMs with the capability to continually acquire emotional motion generation knowledge across different unseen scenarios, potentially contributing to building a closed-loop and self-evolving embodied agent equipped with both empathy and intelligence. Further, this paper poses two key challenges in the L^2-EMG task, i.e., the emotion decoupling challenge and the scenario adapting challenge. To this end, this paper proposes an Emotion-Transferable and Scenario-Adapted Mixture of Experts (ES-MoE) approach which designs a causal-guided emotion decoupling block and a scenario-adapted expert constructing block to address the two challenges, respectively. Especially, this paper constructs multiple L^2-EMG datasets to validate the effectiveness of the ES-MoE approach. Extensive evaluations show that ES-MoE outperforms advanced baselines. 

---
# QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models 

**Authors**: Li Puyin, Tiange Xiang, Ella Mao, Shirley Wei, Xinye Chen, Adnan Masood, Li Fei-fei, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2512.19526)  

**Abstract**: Understanding the physical world is essential for generalist AI agents. However, it remains unclear whether state-of-the-art vision perception models (e.g., large VLMs) can reason physical properties quantitatively. Existing evaluations are predominantly VQA-based and qualitative, offering limited insight into whether these models can infer the kinematic quantities of moving objects from video observations. To address this, we present QuantiPhy, the first benchmark designed to quantitatively measure a VLM's physical reasoning ability. Comprising more than 3.3K video-text instances with numerical ground truth, QuantiPhy evaluates a VLM's performance on estimating an object's size, velocity, and acceleration at a given timestamp, using one of these properties as an input prior. The benchmark standardizes prompts and scoring to assess numerical accuracy, enabling fair comparisons across models. Our experiments on state-of-the-art VLMs reveal a consistent gap between their qualitative plausibility and actual numerical correctness. We further provide an in-depth analysis of key factors like background noise, counterfactual priors, and strategic prompting and find that state-of-the-art VLMs lean heavily on pre-trained world knowledge rather than faithfully using the provided visual and textual inputs as references when reasoning kinematic properties quantitatively. QuantiPhy offers the first rigorous, scalable testbed to move VLMs beyond mere verbal plausibility toward a numerically grounded physical understanding. 

---
# An Agentic Framework for Autonomous Materials Computation 

**Authors**: Zeyu Xia, Jinzhe Ma, Congjie Zheng, Shufei Zhang, Yuqiang Li, Hang Su, P. Hu, Changshui Zhang, Xingao Gong, Wanli Ouyang, Lei Bai, Dongzhan Zhou, Mao Su  

**Link**: [PDF](https://arxiv.org/pdf/2512.19458)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for accelerating scientific discovery, yet their static knowledge and hallucination issues hinder autonomous research applications. Recent advances integrate LLMs into agentic frameworks, enabling retrieval, reasoning, and tool use for complex scientific workflows. Here, we present a domain-specialized agent designed for reliable automation of first-principles materials computations. By embedding domain expertise, the agent ensures physically coherent multi-step workflows and consistently selects convergent, well-posed parameters, thereby enabling reliable end-to-end computational execution. A new benchmark of diverse computational tasks demonstrates that our system significantly outperforms standalone LLMs in both accuracy and robustness. This work establishes a verifiable foundation for autonomous computational experimentation and represents a key step toward fully automated scientific discovery. 

---
# EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration 

**Authors**: Runze Li, Yuwen Zhai, Bo Xu, LiWu Xu, Nian Shi, Wei Zhang, Ran Lin, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19396)  

**Abstract**: Contemporary GUI agents, while increasingly capable due to advances in Large Vision-Language Models (VLMs), often operate with a critical limitation: they treat each task in isolation, lacking a mechanism to systematically learn from past successes. This digital ''amnesia'' results in sub-optimal performance, repeated errors, and poor generalization to novel challenges. To bridge this gap, we introduce EchoTrail-GUI, a novel framework designed to mimic human-like experiential learning by equipping agents with a dynamic, accessible memory. Our framework operates in three distinct stages. First, during Experience Exploration, an agent autonomously interacts with GUI environments to build a curated database of successful task trajectories, validated by a reward model. Crucially, the entire knowledge base construction is thus fully automated, requiring no human supervision. Second, in the Memory Injection stage, upon receiving a new task, our system efficiently retrieves the most relevant past trajectories to serve as actionable ''memories''. Finally, during GUI Task Inference, these memories are injected as in-context guidance to inform the agent's reasoning and decision-making process. We demonstrate the efficacy of our approach on benchmarks including Android World and AndroidLab. The results show that EchoTrail-GUI significantly improves the task success rate and operational efficiency of baseline agents, validating the power of structured memory in creating more robust and intelligent GUI automation. 

---
# Learning General Policies with Policy Gradient Methods 

**Authors**: Simon Ståhlberg, Blai Bonet, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2512.19366)  

**Abstract**: While reinforcement learning methods have delivered remarkable results in a number of settings, generalization, i.e., the ability to produce policies that generalize in a reliable and systematic way, has remained a challenge. The problem of generalization has been addressed formally in classical planning where provable correct policies that generalize over all instances of a given domain have been learned using combinatorial methods. The aim of this work is to bring these two research threads together to illuminate the conditions under which (deep) reinforcement learning approaches, and in particular, policy optimization methods, can be used to learn policies that generalize like combinatorial methods do. We draw on lessons learned from previous combinatorial and deep learning approaches, and extend them in a convenient way. From the former, we model policies as state transition classifiers, as (ground) actions are not general and change from instance to instance. From the latter, we use graph neural networks (GNNs) adapted to deal with relational structures for representing value functions over planning states, and in our case, policies. With these ingredients in place, we find that actor-critic methods can be used to learn policies that generalize almost as well as those obtained using combinatorial approaches while avoiding the scalability bottleneck and the use of feature pools. Moreover, the limitations of the DRL methods on the benchmarks considered have little to do with deep learning or reinforcement learning algorithms, and result from the well-understood expressive limitations of GNNs, and the tradeoff between optimality and generalization (general policies cannot be optimal in some domains). Both of these limitations are addressed without changing the basic DRL methods by adding derived predicates and an alternative cost structure to optimize. 

---
# First-Order Representation Languages for Goal-Conditioned RL 

**Authors**: Simon Ståhlberg, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2512.19355)  

**Abstract**: First-order relational languages have been used in MDP planning and reinforcement learning (RL) for two main purposes: specifying MDPs in compact form, and representing and learning policies that are general and not tied to specific instances or state spaces. In this work, we instead consider the use of first-order languages in goal-conditioned RL and generalized planning. The question is how to learn goal-conditioned and general policies when the training instances are large and the goal cannot be reached by random exploration alone. The technique of Hindsight Experience Replay (HER) provides an answer to this question: it relabels unsuccessful trajectories as successful ones by replacing the original goal with one that was actually achieved. If the target policy must generalize across states and goals, trajectories that do not reach the original goal states can enable more data- and time-efficient learning. In this work, we show that further performance gains can be achieved when states and goals are represented by sets of atoms. We consider three versions: goals as full states, goals as subsets of the original goals, and goals as lifted versions of these subgoals. The result is that the latter two successfully learn general policies on large planning instances with sparse rewards by automatically creating a curriculum of easier goals of increasing complexity. The experiments illustrate the computational gains of these versions, their limitations, and opportunities for addressing them. 

---
# PENDULUM: A Benchmark for Assessing Sycophancy in Multimodal Large Language Models 

**Authors**: A. B. M. Ashikur Rahman, Saeed Anwar, Muhammad Usman, Irfan Ahmad, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2512.19350)  

**Abstract**: Sycophancy, an excessive tendency of AI models to agree with user input at the expense of factual accuracy or in contradiction of visual evidence, poses a critical and underexplored challenge for multimodal large language models (MLLMs). While prior studies have examined this behavior in text-only settings of large language models, existing research on visual or multimodal counterparts remains limited in scope and depth of analysis. To address this gap, we introduce a comprehensive evaluation benchmark, \textit{PENDULUM}, comprising approximately 2,000 human-curated Visual Question Answering pairs specifically designed to elicit sycophantic responses. The benchmark spans six distinct image domains of varying complexity, enabling a systematic investigation of how image type and inherent challenges influence sycophantic tendencies. Through extensive evaluation of state-of-the-art MLLMs. we observe substantial variability in model robustness and a pronounced susceptibility to sycophantic and hallucinatory behavior. Furthermore, we propose novel metrics to quantify sycophancy in visual reasoning, offering deeper insights into its manifestations across different multimodal contexts. Our findings highlight the urgent need for developing sycophancy-resilient architectures and training strategies to enhance factual consistency and reliability in future MLLMs. Our proposed dataset with MLLMs response are available at this https URL. 

---
# VIGOR+: Iterative Confounder Generation and Validation via LLM-CEVAE Feedback Loop 

**Authors**: JiaWei Zhu, ZiHeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19349)  

**Abstract**: Hidden confounding remains a fundamental challenge in causal inference from observational data. Recent advances leverage Large Language Models (LLMs) to generate plausible hidden confounders based on domain knowledge, yet a critical gap exists: LLM-generated confounders often exhibit semantic plausibility without statistical utility. We propose VIGOR+ (Variational Information Gain for iterative cOnfounder Refinement), a novel framework that closes the loop between LLM-based confounder generation and CEVAE-based statistical validation. Unlike prior approaches that treat generation and validation as separate stages, VIGOR+ establishes an iterative feedback mechanism: validation signals from CEVAE (including information gain, latent consistency metrics, and diagnostic messages) are transformed into natural language feedback that guides subsequent LLM generation rounds. This iterative refinement continues until convergence criteria are met. We formalize the feedback mechanism, prove convergence properties under mild assumptions, and provide a complete algorithmic framework. 

---
# SafeMed-R1: Adversarial Reinforcement Learning for Generalizable and Robust Medical Reasoning in Vision-Language Models 

**Authors**: A.A. Gde Yogi Pramana, Jason Ray, Anthony Jaya, Michael Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2512.19317)  

**Abstract**: Vision--Language Models (VLMs) show significant promise for Medical Visual Question Answering (VQA), yet their deployment in clinical settings is hindered by severe vulnerability to adversarial attacks. Standard adversarial training, while effective for simpler tasks, often degrades both generalization performance and the quality of generated clinical reasoning. We introduce SafeMed-R1, a hybrid defense framework that ensures robust performance while preserving high-quality, interpretable medical reasoning. SafeMed-R1 employs a two-stage approach: at training time, we integrate Adversarial Training with Group Relative Policy Optimization (AT-GRPO) to explicitly robustify the reasoning process against worst-case perturbations; at inference time, we augment the model with Randomized Smoothing to provide certified $L_2$-norm robustness guarantees. We evaluate SafeMed-R1 on the OmniMedVQA benchmark across eight medical imaging modalities comprising over 88,000 samples. Our experiments reveal that standard fine-tuned VLMs, despite achieving 95\% accuracy on clean inputs, collapse to approximately 25\% under PGD attacks. In contrast, SafeMed-R1 maintains 84.45\% accuracy under the same adversarial conditions, representing a 59 percentage point improvement in robustness. Furthermore, we demonstrate that models trained with explicit chain-of-thought reasoning exhibit superior adversarial robustness compared to instruction-only variants, suggesting a synergy between interpretability and security in medical AI systems. 

---
# Helios: A Foundational Language Model for Smart Energy Knowledge Reasoning and Application 

**Authors**: Haoyu Jiang, Fanjie Zeng, Boan Qu, Xiaojie Lin, Wei Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2512.19299)  

**Abstract**: In the global drive toward carbon neutrality, deeply coordinated smart energy systems underpin industrial transformation. However, the interdisciplinary, fragmented, and fast-evolving expertise in this domain prevents general-purpose LLMs, which lack domain knowledge and physical-constraint awareness, from delivering precise engineering-aligned inference and generation. To address these challenges, we introduce Helios, a large language model tailored to the smart energy domain, together with a comprehensive suite of resources to advance LLM research in this field. Specifically, we develop Enersys, a multi-agent collaborative framework for end-to-end dataset construction, through which we produce: (1) a smart energy knowledge base, EnerBase, to enrich the model's foundational expertise; (2) an instruction fine-tuning dataset, EnerInstruct, to strengthen performance on domain-specific downstream tasks; and (3) an RLHF dataset, EnerReinforce, to align the model with human preferences and industry standards. Leveraging these resources, Helios undergoes large-scale pretraining, SFT, and RLHF. We also release EnerBench, a benchmark for evaluating LLMs in smart energy scenarios, and demonstrate that our approach significantly enhances domain knowledge mastery, task execution accuracy, and alignment with human preferences. 

---
# Vibe Reasoning: Eliciting Frontier AI Mathematical Capabilities -- A Case Study on IMO 2025 Problem 6 

**Authors**: Jiaao Wu, Xian Zhang, Fan Yang, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2512.19287)  

**Abstract**: We introduce Vibe Reasoning, a human-AI collaborative paradigm for solving complex mathematical problems. Our key insight is that frontier AI models already possess the knowledge required to solve challenging problems -- they simply do not know how, what, or when to apply it. Vibe Reasoning transforms AI's latent potential into manifested capability through generic meta-prompts, agentic grounding, and model orchestration. We demonstrate this paradigm through IMO 2025 Problem 6, a combinatorial optimization problem where autonomous AI systems publicly reported failures. Our solution combined GPT-5's exploratory capabilities with Gemini 3 Pro's proof strengths, leveraging agentic workflows with Python code execution and file-based memory, to derive both the correct answer (2112) and a rigorous mathematical proof. Through iterative refinement across multiple attempts, we discovered the necessity of agentic grounding and model orchestration, while human prompts evolved from problem-specific hints to generic, transferable meta-prompts. We analyze why capable AI fails autonomously, how each component addresses specific failure modes, and extract principles for effective vibe reasoning. Our findings suggest that lightweight human guidance can unlock frontier models' mathematical reasoning potential. This is ongoing work; we are developing automated frameworks and conducting broader evaluations to further validate Vibe Reasoning's generality and effectiveness. 

---
# DeliveryBench: Can Agents Earn Profit in Real World? 

**Authors**: Lingjun Mao, Jiawei Ren, Kun Zhou, Jixuan Chen, Ziqiao Ma, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2512.19234)  

**Abstract**: LLMs and VLMs are increasingly deployed as embodied agents, yet existing benchmarks largely revolve around simple short-term tasks and struggle to capture rich realistic constraints that shape real-world decision making. To close this gap, we propose DeliveryBench, a city-scale embodied benchmark grounded in the real-world profession of food delivery. Food couriers naturally operate under long-horizon objectives (maximizing net profit over hours) while managing diverse constraints, e.g., delivery deadline, transportation expense, vehicle battery, and necessary interactions with other couriers and customers. DeliveryBench instantiates this setting in procedurally generated 3D cities with diverse road networks, buildings, functional locations, transportation modes, and realistic resource dynamics, enabling systematic evaluation of constraint-aware, long-horizon planning. We benchmark a range of VLM-based agents across nine cities and compare them with human players. Our results reveal a substantial performance gap to humans, and find that these agents are short-sighted and frequently break basic commonsense constraints. Additionally, we observe distinct personalities across models (e.g., adventurous GPT-5 vs. conservative Claude), highlighting both the brittleness and the diversity of current VLM-based embodied agents in realistic, constraint-dense environments. Our code, data, and benchmark are available at this https URL. 

---
# Generation of Programmatic Rules for Document Forgery Detection Using Large Language Models 

**Authors**: Valentin Schmidberger, Manuel Eberhardinger, Setareh Maghsudi, Johannes Maucher  

**Link**: [PDF](https://arxiv.org/pdf/2512.19228)  

**Abstract**: Document forgery poses a growing threat to legal, economic, and governmental processes, requiring increasingly sophisticated verification mechanisms. One approach involves the use of plausibility checks, rule-based procedures that assess the correctness and internal consistency of data, to detect anomalies or signs of manipulation. Although these verification procedures are essential for ensuring data integrity, existing plausibility checks are manually implemented by software engineers, which is time-consuming. Recent advances in code generation with large language models (LLMs) offer new potential for automating and scaling the generation of these checks. However, adapting LLMs to the specific requirements of an unknown domain remains a significant challenge. This work investigates the extent to which LLMs, adapted on domain-specific code and data through different fine-tuning strategies, can generate rule-based plausibility checks for forgery detection on constrained hardware resources. We fine-tune open-source LLMs, Llama 3.1 8B and OpenCoder 8B, on structured datasets derived from real-world application scenarios and evaluate the generated plausibility checks on previously unseen forgery patterns. The results demonstrate that the models are capable of generating executable and effective verification procedures. This also highlights the potential of LLMs as scalable tools to support human decision-making in security-sensitive contexts where comprehensibility is required. 

---
# Observer, Not Player: Simulating Theory of Mind in LLMs through Game Observation 

**Authors**: Jerry Wang, Ting Yiu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19210)  

**Abstract**: We present an interactive framework for evaluating whether large language models (LLMs) exhibit genuine "understanding" in a simple yet strategic environment. As a running example, we focus on Rock-Paper-Scissors (RPS), which, despite its apparent simplicity, requires sequential reasoning, adaptation, and strategy recognition. Our system positions the LLM as an Observer whose task is to identify which strategies are being played and to articulate the reasoning behind this judgment. The purpose is not to test knowledge of Rock-Paper-Scissors itself, but to probe whether the model can exhibit mind-like reasoning about sequential behavior. To support systematic evaluation, we provide a benchmark consisting of both static strategies and lightweight dynamic strategies specified by well-prompted rules. We quantify alignment between the Observer's predictions and the ground-truth distributions induced by actual strategy pairs using three complementary signals: Cross-Entropy, Brier score, and Expected Value (EV) discrepancy. These metrics are further integrated into a unified score, the Union Loss, which balances calibration, sensitivity, and payoff alignment. Together with a Strategy Identification Rate (SIR) metric, our framework captures not only predictive accuracy but also whether the model can stably identify the latent strategies in play. The demo emphasizes interactivity, transparency, and reproducibility. Users can adjust LLM distributions in real time, visualize losses as they evolve, and directly inspect reasoning snippets to identify where and why failures occur. In doing so, our system provides a practical and interpretable proxy for mind-like inference in sequential games, offering insights into both the strengths and limitations of current LLM reasoning. 

---
# Can We Test Consciousness Theories on AI? Ablations, Markers, and Robustness 

**Authors**: Yin Jun Phua  

**Link**: [PDF](https://arxiv.org/pdf/2512.19155)  

**Abstract**: The search for reliable indicators of consciousness has fragmented into competing theoretical camps (Global Workspace Theory (GWT), Integrated Information Theory (IIT), and Higher-Order Theories (HOT)), each proposing distinct neural signatures. We adopt a synthetic neuro-phenomenology approach: constructing artificial agents that embody these mechanisms to test their functional consequences through precise architectural ablations impossible in biological systems. Across three experiments, we report dissociations suggesting these theories describe complementary functional layers rather than competing accounts. In Experiment 1, a no-rewire Self-Model lesion abolishes metacognitive calibration while preserving first-order task performance, yielding a synthetic blindsight analogue consistent with HOT predictions. In Experiment 2, workspace capacity proves causally necessary for information access: a complete workspace lesion produces qualitative collapse in access-related markers, while partial reductions show graded degradation, consistent with GWT's ignition framework. In Experiment 3, we uncover a broadcast-amplification effect: GWT-style broadcasting amplifies internal noise, creating extreme fragility. The B2 agent family is robust to the same latent perturbation; this robustness persists in a Self-Model-off / workspace-read control, cautioning against attributing the effect solely to $z_{\text{self}}$ compression. We also report an explicit negative result: raw perturbational complexity (PCI-A) decreases under the workspace bottleneck, cautioning against naive transfer of IIT-adjacent proxies to engineered agents. These results suggest a hierarchical design principle: GWT provides broadcast capacity, while HOT provides quality control. We emphasize that our agents are not conscious; they are reference implementations for testing functional predictions of consciousness theories. 

---
# Understanding Chain-of-Thought in Large Language Models via Topological Data Analysis 

**Authors**: Chenghao Li, Chaoning Zhang, Yi Lu, Shuxu Chen, Xudong Wang, Jiaquan Zhang, Zhicheng Wang, Zhengxun Jin, Kuien Liu, Sung-Ho Bae, Guoqing Wang, Yang Yang, Hen Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19135)  

**Abstract**: With the development of large language models (LLMs), particularly with the introduction of the long reasoning chain technique, the reasoning ability of LLMs in complex problem-solving has been significantly enhanced. While acknowledging the power of long reasoning chains, we cannot help but wonder: Why do different reasoning chains perform differently in reasoning? What components of the reasoning chains play a key role? Existing studies mainly focus on evaluating reasoning chains from a functional perspective, with little attention paid to their structural mechanisms. To address this gap, this work is the first to analyze and evaluate the quality of the reasoning chain from a structural perspective. We apply persistent homology from Topological Data Analysis (TDA) to map reasoning steps into semantic space, extract topological features, and analyze structural changes. These changes reveal semantic coherence, logical redundancy, and identify logical breaks and gaps. By calculating homology groups, we assess connectivity and redundancy at various scales, using barcode and persistence diagrams to quantify stability and consistency. Our results show that the topological structural complexity of reasoning chains correlates positively with accuracy. More complex chains identify correct answers sooner, while successful reasoning exhibits simpler topologies, reducing redundancy and cycles, enhancing efficiency and interpretability. This work provides a new perspective on reasoning chain quality assessment and offers guidance for future optimization. 

---
# FC-MIR: A Mobile Screen Awareness Framework for Intent-Aware Recommendation based on Frame-Compressed Multimodal Trajectory Reasoning 

**Authors**: Zhe Yang, Xiaoshuang Sheng, Zhengnan Zhang, Jidong Wu, Zexing Wang, Xin He, Shenghua Xu, Guanjing Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2512.19107)  

**Abstract**: Identifying user intent from mobile UI operation trajectories is critical for advancing UI understanding and enabling task automation agents. While Multimodal Large Language Models (MLLMs) excel at video understanding tasks, their real-time mobile deployment is constrained by heavy computational costs and inefficient redundant frame processing. To address these issues, we propose the FC-MIR framework: leveraging keyframe sampling and adaptive concatenation, it cuts visual redundancy to boost inference efficiency, while integrating state-of-the-art closed-source MLLMs or fine-tuned models (e.g., Qwen3-VL) for trajectory summarization and intent prediction. We further expand task scope to explore generating post-prediction operations and search suggestions, and introduce a fine-grained metric to evaluate the practical utility of summaries, predictions, and suggestions. For rigorous assessment, we construct a UI trajectory dataset covering scenarios from UI-Agents (Agent-I) and real user interactions (Person-I). Experimental results show our compression method retains performance at 50%-60% compression rates; both closed-source and fine-tuned MLLMs demonstrate strong intent summarization, supporting potential lightweight on-device deployment. However, MLLMs still struggle with useful and "surprising" suggestions, leaving room for improvement. Finally, we deploy the framework in a real-world setting, integrating UI perception and UI-Agent proxies to lay a foundation for future progress in this field. 

---
# Conditioning Accept-Desirability models in the context of AGM-like belief change 

**Authors**: Kathelijne Coussement, Gert de Cooman, Keano De Vos  

**Link**: [PDF](https://arxiv.org/pdf/2512.19096)  

**Abstract**: We discuss conditionalisation for Accept-Desirability models in an abstract decision-making framework, where uncertain rewards live in a general linear space, and events are special projection operators on that linear space. This abstract setting allows us to unify classical and quantum probabilities, and extend them to an imprecise probabilities context. We introduce a new conditioning rule for our Accept-Desirability models, based on the idea that observing an event introduces new indifferences between options. We associate a belief revision operator with our conditioning rule, and investigate which of the AGM axioms for belief revision still hold in our more general framework. We investigate two interesting special cases where all of these axioms are shown to still hold: classical propositional logic and full conditional probabilities. 

---
# Tool-Augmented Hybrid Ensemble Reasoning with Distillation for Bilingual Mathematical Problem Solving 

**Authors**: Peiqing Lu, Yuan Zhang, Haoyun Zhang, Jiasen Zheng, Kejian Tong, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19093)  

**Abstract**: Bilingual mathematical problem solving needs a clear link between language reasoning and symbolic calculation. Large language models often handle language well but are weak in accurate computation. This paper presents HERALD (Hybrid Ensemble Reasoning with Adaptive Learning and Distillation), a framework that joins reasoning and calculation using NuminaMath-7B-TIR, GPT-4o, and Mistral-7B. HERALD uses adaptive routing, tool-based reinforcement learning, and knowledge distillation to connect different reasoning paths. Confidence calibration keeps weighting stable, and dual-path checking keeps results correct. Reinforcement learning controls tool use to cut redundancy, and distillation lowers delay without hurting accuracy. The system shows that combining symbolic checking, adaptive ensembles, and bilingual fine-tuning helps achieve both fluent reasoning and precise calculation. HERALD offers a practical solution for multilingual mathematical reasoning with better accuracy, stability, and clarity. 

---
# $γ(3,4)$ `Attention' in Cognitive Agents: Ontology-Free Knowledge Representations With Promise Theoretic Semantics 

**Authors**: Mark Burgess  

**Link**: [PDF](https://arxiv.org/pdf/2512.19084)  

**Abstract**: The semantics and dynamics of `attention' are closely related to promise theoretic notions developed for autonomous agents and can thus easily be written down in promise framework. In this way one may establish a bridge between vectorized Machine Learning and Knowledge Graph representations without relying on language models implicitly. Our expectations for knowledge presume a degree of statistical stability, i.e. average invariance under repeated observation, or `trust' in the data. Both learning networks and knowledge graph representations can meaningfully coexist to preserve different aspects of data. While vectorized data are useful for probabilistic estimation, graphs preserve the intentionality of the source even under data fractionation. Using a Semantic Spacetime $\gamma(3,4)$ graph, one avoids complex ontologies in favour of classification of features by their roles in semantic processes. The latter favours an approach to reasoning under conditions of uncertainty. Appropriate attention to causal boundary conditions may lead to orders of magnitude compression of data required for such context determination, as required in the contexts of autonomous robotics, defence deployments, and ad hoc emergency services. 

---
# Population-Evolve: a Parallel Sampling and Evolutionary Method for LLM Math Reasoning 

**Authors**: Yanzhi Zhang, Yitong Duan, Zhaoxi Zhang, Jiyan He, Shuxin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.19081)  

**Abstract**: Test-time scaling has emerged as a promising direction for enhancing the reasoning capabilities of Large Language Models in last few years. In this work, we propose Population-Evolve, a training-free method inspired by Genetic Algorithms to optimize LLM reasoning. Our approach maintains a dynamic population of candidate solutions for each problem via parallel reasoning. By incorporating an evolve prompt, the LLM self-evolves its population in all iterations. Upon convergence, the final answer is derived via majority voting. Furthermore, we establish a unification framework that interprets existing test-time scaling strategies through the lens of genetic algorithms. Empirical results demonstrate that Population-Evolve achieves superior accuracy with low performance variance and computational efficiency. Our findings highlight the potential of evolutionary strategies to unlock the reasoning power of LLMs during inference. 

---
# Can abstract concepts from LLM improve SLM performance? 

**Authors**: Siddharth Tandon  

**Link**: [PDF](https://arxiv.org/pdf/2512.19069)  

**Abstract**: Large language models (LLMs) excel at diverse tasks, but their deployment on resource-constrained devices remains challenging. Existing methods like quantization, pruning, and distillation can reduce memory footprint but often demand extensive experimentation and careful infrastructure design. Leveraging existing techniques for extracting high-level concepts (represented as steering vectors) from larger models, we investigate their transferability to smaller language models (SLM) during inference. We demonstrate through extensive experimentation that these concepts can be effectively transferred to smaller models, irrespective of their family (e.g., Phi, Llama, Qwen), leading to performance improvements across a wide range of tasks. Furthermore, we introduce inference-time scaling to enhance performance by dynamically adjusting the steering intensity which has resulted in a 7-15\% of accuracy improvement for Qwen3-0.6B. 

---
# Recontextualization Mitigates Specification Gaming without Modifying the Specification 

**Authors**: Ariana Azarbal, Victor Gillioz, Vladimir Ivanov, Bryce Woodworth, Jacob Drori, Nevan Wichers, Aram Ebtekar, Alex Cloud, Alexander Matt Turner  

**Link**: [PDF](https://arxiv.org/pdf/2512.19027)  

**Abstract**: Developers often struggle to specify correct training labels and rewards. Perhaps they don't need to. We propose recontextualization, which reduces how often language models "game" training signals, performing misbehaviors those signals mistakenly reinforce. We show recontextualization prevents models from learning to 1) prioritize evaluation metrics over chat response quality; 2) special-case code to pass incorrect tests; 3) lie to users; and 4) become sycophantic. Our method works by generating completions from prompts discouraging misbehavior and then recontextualizing them as though they were in response to prompts permitting misbehavior. Recontextualization trains language models to resist misbehavior even when instructions permit it. This mitigates the reinforcement of misbehavior from misspecified training signals, reducing specification gaming without improving the supervision signal. 

---
# ORPR: An OR-Guided Pretrain-then-Reinforce Learning Model for Inventory Management 

**Authors**: Lingjie Zhao, Xue Yu, Yongzhi Qi, Hao Hu, Jianshen Zhang, Yingzheng Ma, Shuyu Han, Wei Qi, Zuo-Jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19001)  

**Abstract**: As the pursuit of synergy between Artificial Intelligence (AI) and Operations Research (OR) gains momentum in handling complex inventory systems, a critical challenge persists: how to effectively reconcile AI's adaptive perception with OR's structural rigor. To bridge this gap, we propose a novel OR-Guided "Pretrain-then-Reinforce" framework. To provide structured guidance, we propose a simulation-augmented OR model that generates high-quality reference decisions, implicitly capturing complex business constraints and managerial preferences. Leveraging these OR-derived decisions as foundational training labels, we design a domain-informed deep learning foundation model to establish foundational decision-making capabilities, followed by a reinforcement learning (RL) fine-tuning stage. Uniquely, we position RL as a deep alignment mechanism that enables the AI agent to internalize the optimality principles of OR, while simultaneously leveraging exploration for general policy refinement and allowing expert guidance for scenario-specific adaptation (e.g., promotional events). Validated through extensive numerical experiments and a field deployment at this http URL augmented by a Difference-in-Differences (DiD) analysis, our model significantly outperforms incumbent industrial practices, delivering real-world gains of a 5.27-day reduction in turnover and a 2.29% increase in in-stock rates, alongside a 29.95% decrease in holding costs. Contrary to the prevailing trend of brute-force model scaling, our study demonstrates that a lightweight, domain-informed model can deliver state-of-the-art performance and robust transferability when guided by structured OR logic. This approach offers a scalable and cost-effective paradigm for intelligent supply chain management, highlighting the value of deeply aligning AI with OR. 

---
# Training Multimodal Large Reasoning Models Needs Better Thoughts: A Three-Stage Framework for Long Chain-of-Thought Synthesis and Selection 

**Authors**: Yizhi Wang, Linan Yue, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18956)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning tasks through long Chain-of-Thought (CoT) reasoning. Extending these successes to multimodal reasoning remains challenging due to the increased complexity of integrating diverse input modalities and the scarcity of high-quality long CoT training data. Existing multimodal datasets and CoT synthesis methods still suffer from limited reasoning depth, modality conversion errors, and rigid generation pipelines, hindering model performance and stability. To this end, in this paper, we propose SynSelect, a novel three-stage Synthesis-Selection framework for generating high-quality long CoT data tailored to multimodal reasoning tasks. Specifically, SynSelect first leverages multiple heterogeneous multimodal LRMs to produce diverse candidate CoTs, and then applies both instance and batch level selection to filter high-quality CoTs that can effectively enhance the model's reasoning capabilities. Extensive experiments on multiple multimodal benchmarks demonstrate that models supervised fine-tuned on SynSelect-generated data significantly outperform baselines and achieve further improvements after reinforcement learning post-training. Our results validate SynSelect as an effective approach for advancing multimodal LRMs reasoning capabilities. 

---
# Clustering-based Transfer Learning for Dynamic Multimodal MultiObjective Evolutionary Algorithm 

**Authors**: Li Yan, Bolun Liu, Chao Li, Jing Liang, Kunjie Yu, Caitong Yue, Xuzhao Chai, Boyang Qu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18947)  

**Abstract**: Dynamic multimodal multiobjective optimization presents the dual challenge of simultaneously tracking multiple equivalent pareto optimal sets and maintaining population diversity in time-varying environments. However, existing dynamic multiobjective evolutionary algorithms often neglect solution modality, whereas static multimodal multiobjective evolutionary algorithms lack adaptability to dynamic changes. To address above challenge, this paper makes two primary contributions. First, we introduce a new benchmark suite of dynamic multimodal multiobjective test functions constructed by fusing the properties of both dynamic and multimodal optimization to establish a rigorous evaluation platform. Second, we propose a novel algorithm centered on a Clustering-based Autoencoder prediction dynamic response mechanism, which utilizes an autoencoder model to process matched clusters to generate a highly diverse initial population. Furthermore, to balance the algorithm's convergence and diversity, we integrate an adaptive niching strategy into the static optimizer. Empirical analysis on 12 instances of dynamic multimodal multiobjective test functions reveals that, compared with several state-of-the-art dynamic multiobjective evolutionary algorithms and multimodal multiobjective evolutionary algorithms, our algorithm not only preserves population diversity more effectively in the decision space but also achieves superior convergence in the objective space. 

---
# Multimodal Bayesian Network for Robust Assessment of Casualties in Autonomous Triage 

**Authors**: Szymon Rusiecki, Cecilia G. Morales, Kimberly Elenberg, Leonard Weiss, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2512.18908)  

**Abstract**: Mass Casualty Incidents can overwhelm emergency medical systems and resulting delays or errors in the assessment of casualties can lead to preventable deaths. We present a decision support framework that fuses outputs from multiple computer vision models, estimating signs of severe hemorrhage, respiratory distress, physical alertness, or visible trauma, into a Bayesian network constructed entirely from expert-defined rules. Unlike traditional data-driven models, our approach does not require training data, supports inference with incomplete information, and is robust to noisy or uncertain observations. We report performance for two missions involving 11 and 9 casualties, respectively, where our Bayesian network model substantially outperformed vision-only baselines during evaluation of our system in the DARPA Triage Challenge (DTC) field scenarios. The accuracy of physiological assessment improved from 15% to 42% in the first scenario and from 19% to 46% in the second, representing nearly threefold increase in performance. More importantly, overall triage accuracy increased from 14% to 53% in all patients, while the diagnostic coverage of the system expanded from 31% to 95% of the cases requiring assessment. These results demonstrate that expert-knowledge-guided probabilistic reasoning can significantly enhance automated triage systems, offering a promising approach to supporting emergency responders in MCIs. This approach enabled Team Chiron to achieve 4th place out of 11 teams during the 1st physical round of the DTC. 

---
# Gabliteration: Adaptive Multi-Directional Neural Weight Modification for Selective Behavioral Alteration in Large Language Models 

**Authors**: Gökdeniz Gülmez  

**Link**: [PDF](https://arxiv.org/pdf/2512.18901)  

**Abstract**: We present Gabliteration, a novel neural weight modification technique that advances beyond traditional abliteration methods by implementing adaptive multi-directional projections with regularized layer selection. Our approach addresses the fundamental limitation of existing methods that compromise model quality while attempting to modify specific behavioral patterns. Through dynamic layer optimization, regularized projection matrices, and adaptive scaling mechanisms, we achieve theoretically superior weight modification while minimizing quality degradation in unrelated domains. We validate our method through the gabliterated-v1 model series (0.6B to 4B parameters) available on Hugging Face, demonstrating practical applicability across multiple model scales. 

---
# CORE: Concept-Oriented Reinforcement for Bridging the Definition-Application Gap in Mathematical Reasoning 

**Authors**: Zijun Gao, Zhikun Xu, Xiao Ye, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.18857)  

**Abstract**: Large language models (LLMs) often solve challenging math exercises yet fail to apply the concept right when the problem requires genuine understanding. Popular Reinforcement Learning with Verifiable Rewards (RLVR) pipelines reinforce final answers but provide little fine-grained conceptual signal, so models improve at pattern reuse rather than conceptual applications. We introduce CORE (Concept-Oriented REinforcement), an RL training framework that turns explicit concepts into a controllable supervision signal. Starting from a high-quality, low-contamination textbook resource that links verifiable exercises to concise concept descriptions, we run a sanity probe showing LLMs can restate definitions but fail concept-linked quizzes, quantifying the conceptual reasoning gap. CORE then (i) synthesizes concept-aligned quizzes, (ii) injects brief concept snippets during rollouts to elicit concept-primed trajectories, and (iii) reinforces conceptual reasoning via trajectory replacement after group failures, a lightweight forward-KL constraint that aligns unguided with concept-primed policies, or standard GRPO directly on concept-aligned quizzes. Across several models, CORE delivers consistent gains over vanilla and SFT baselines on both in-domain concept-exercise suites and diverse out-of-domain math benchmarks. CORE unifies direct training on concept-aligned quizzes and concept-injected rollouts under outcome regularization. It provides fine-grained conceptual supervision that bridges problem-solving competence and genuine conceptual reasoning, while remaining algorithm- and verifier-agnostic. 

---
# HARBOR: Holistic Adaptive Risk assessment model for BehaviORal healthcare 

**Authors**: Aditya Siddhant  

**Link**: [PDF](https://arxiv.org/pdf/2512.18829)  

**Abstract**: Behavioral healthcare risk assessment remains a challenging problem due to the highly multimodal nature of patient data and the temporal dynamics of mood and affective disorders. While large language models (LLMs) have demonstrated strong reasoning capabilities, their effectiveness in structured clinical risk scoring remains unclear. In this work, we introduce HARBOR, a behavioral health aware language model designed to predict a discrete mood and risk score, termed the Harbor Risk Score (HRS), on an integer scale from -3 (severe depression) to +3 (mania). We also release PEARL, a longitudinal behavioral healthcare dataset spanning four years of monthly observations from three patients, containing physiological, behavioral, and self reported mental health signals. We benchmark traditional machine learning models, proprietary LLMs, and HARBOR across multiple evaluation settings and ablations. Our results show that HARBOR outperforms classical baselines and off the shelf LLMs, achieving 69 percent accuracy compared to 54 percent for logistic regression and 29 percent for the strongest proprietary LLM baseline. 

---
# The Dead Salmons of AI Interpretability 

**Authors**: Maxime Méloux, Giada Dirupo, François Portet, Maxime Peyrard  

**Link**: [PDF](https://arxiv.org/pdf/2512.18792)  

**Abstract**: In a striking neuroscience study, the authors placed a dead salmon in an MRI scanner and showed it images of humans in social situations. Astonishingly, standard analyses of the time reported brain regions predictive of social emotions. The explanation, of course, was not supernatural cognition but a cautionary tale about misapplied statistical inference. In AI interpretability, reports of similar ''dead salmon'' artifacts abound: feature attribution, probing, sparse auto-encoding, and even causal analyses can produce plausible-looking explanations for randomly initialized neural networks. In this work, we examine this phenomenon and argue for a pragmatic statistical-causal reframing: explanations of computational systems should be treated as parameters of a (statistical) model, inferred from computational traces. This perspective goes beyond simply measuring statistical variability of explanations due to finite sampling of input data; interpretability methods become statistical estimators, and findings should be tested against explicit and meaningful alternative computational hypotheses, with uncertainty quantified with respect to the postulated statistical model. It also highlights important theoretical issues, such as the identifiability of common interpretability queries, which we argue is critical to understand the field's susceptibility to false discoveries, poor generalizability, and high variance. More broadly, situating interpretability within the standard toolkit of statistical inference opens promising avenues for future work aimed at turning AI interpretability into a pragmatic and rigorous science. 

---
# MEEA: Mere Exposure Effect-Driven Confrontational Optimization for LLM Jailbreaking 

**Authors**: Jianyi Zhang, Shizhao Liu, Ziyin Zhou, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.18755)  

**Abstract**: The rapid advancement of large language models (LLMs) has intensified concerns about the robustness of their safety alignment. While existing jailbreak studies explore both single-turn and multi-turn strategies, most implicitly assume a static safety boundary and fail to account for how contextual interactions dynamically influence model behavior, leading to limited stability and generalization. Motivated by this gap, we propose MEEA (Mere Exposure Effect Attack), a psychology-inspired, fully automated black-box framework for evaluating multi-turn safety robustness, grounded in the mere exposure effect. MEEA leverages repeated low-toxicity semantic exposure to induce a gradual shift in a model's effective safety threshold, enabling progressive erosion of alignment constraints over sustained interactions. Concretely, MEEA constructs semantically progressive prompt chains and optimizes them using a simulated annealing strategy guided by semantic similarity, toxicity, and jailbreak effectiveness. Extensive experiments on both closed-source and open-source models, including GPT-4, Claude-3.5, and DeepSeek-R1, demonstrate that MEEA consistently achieves higher attack success rates than seven representative baselines, with an average Attack Success Rate (ASR) improvement exceeding 20%. Ablation studies further validate the necessity of both annealing-based optimization and contextual exposure mechanisms. Beyond improved attack effectiveness, our findings indicate that LLM safety behavior is inherently dynamic and history-dependent, challenging the common assumption of static alignment boundaries and highlighting the need for interaction-aware safety evaluation and defense mechanisms. Our code is available at: this https URL 

---
# Counterfactual Basis Extension and Representational Geometry: An MDL-Constrained Model of Conceptual Growth 

**Authors**: Chainarong Amornbunchornvej  

**Link**: [PDF](https://arxiv.org/pdf/2512.18732)  

**Abstract**: Concept learning becomes possible only when existing representations fail to account for experience. Most models of learning and inference, however, presuppose a fixed representational basis within which belief updating occurs. In this paper, I address a prior question: under what structural conditions can the representational basis itself expand in a principled and selective way?
I propose a geometric framework in which conceptual growth is modeled as admissible basis extension evaluated under a Minimum Description Length (MDL) criterion. Experience, whether externally observed or internally simulated, is represented as vectors relative to a current conceptual subspace. Residual components capture systematic representational failure, and candidate conceptual extensions are restricted to low-rank, admissible transformations. I show that any MDL-accepted extension can be chosen so that its novel directions lie entirely within the residual span induced by experience, while extensions orthogonal to this span strictly increase description length and are therefore rejected.
This yields a conservative account of imagination and conceptual innovation. Internally generated counterfactual representations contribute to learning only insofar as they expose or amplify structured residual error, and cannot introduce arbitrary novelty. I further distinguish representational counterfactuals--counterfactuals over an agent's conceptual basis--from causal or value-level counterfactuals, and show how MDL provides a normative selection principle governing representational change.
Overall, the framework characterizes conceptual development as an error-driven, geometry-constrained process of basis extension, clarifying both the role and the limits of imagination in learning and theory change. 

---
# KeenKT: Knowledge Mastery-State Disambiguation for Knowledge Tracing 

**Authors**: Zhifei Li, Lifan Chen, Jiali Yi, Xiaoju Hou, Yue Zhao, Wenxin Huang, Miao Zhang, Kui Xiao, Bing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18709)  

**Abstract**: Knowledge Tracing (KT) aims to dynamically model a student's mastery of knowledge concepts based on their historical learning interactions. Most current methods rely on single-point estimates, which cannot distinguish true ability from outburst or carelessness, creating ambiguity in judging mastery. To address this issue, we propose a Knowledge Mastery-State Disambiguation for Knowledge Tracing model (KeenKT), which represents a student's knowledge state at each interaction using a Normal-Inverse-Gaussian (NIG) distribution, thereby capturing the fluctuations in student learning behaviors. Furthermore, we design an NIG-distance-based attention mechanism to model the dynamic evolution of the knowledge state. In addition, we introduce a diffusion-based denoising reconstruction loss and a distributional contrastive learning loss to enhance the model's robustness. Extensive experiments on six public datasets demonstrate that KeenKT outperforms SOTA KT models in terms of prediction accuracy and sensitivity to behavioral fluctuations. The proposed method yields the maximum AUC improvement of 5.85% and the maximum ACC improvement of 6.89%. 

---
# Social Comparison without Explicit Inference of Others' Reward Values: A Constructive Approach Using a Probabilistic Generative Model 

**Authors**: Yosuke Taniuchi, Chie Hieida, Atsushi Noritake, Kazushi Ikeda, Masaki Isoda  

**Link**: [PDF](https://arxiv.org/pdf/2512.18687)  

**Abstract**: Social comparison -- the process of evaluating one's rewards relative to others -- plays a fundamental role in primate social cognition. However, it remains unknown from a computational perspective how information about others' rewards affects the evaluation of one's own reward. With a constructive approach, this study examines whether monkeys merely recognize objective reward differences or, instead, infer others' subjective reward valuations. We developed three computational models with varying degrees of social information processing: an Internal Prediction Model (IPM), which infers the partner's subjective values; a No Comparison Model (NCM), which disregards partner information; and an External Comparison Model (ECM), which directly incorporates the partner's objective rewards. To test model performance, we used a multi-layered, multimodal latent Dirichlet allocation. We trained the models on a dataset containing the behavior of a pair of monkeys, their rewards, and the conditioned stimuli. Then, we evaluated the models' ability to classify subjective values across pre-defined experimental conditions. The ECM achieved the highest classification score in the Rand Index (0.88 vs. 0.79 for the IPM) under our settings, suggesting that social comparison relies on objective reward differences rather than inferences about subjective states. 

---
# IntelliCode: A Multi-Agent LLM Tutoring System with Centralized Learner Modeling 

**Authors**: Jones David, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2512.18669)  

**Abstract**: LLM-based tutors are typically single-turn assistants that lack persistent representations of learner knowledge, making it difficult to provide principled, transparent, and long-term pedagogical support. We introduce IntelliCode, a multi-agent LLM tutoring system built around a centralized, versioned learner state that integrates mastery estimates, misconceptions, review schedules, and engagement signals. A StateGraph Orchestrator coordinates six specialized agents: skill assessment, learner profiling, graduated hinting, curriculum selection, spaced repetition, and engagement monitoring, each operating as a pure transformation over the shared state under a single-writer policy. This architecture enables auditable mastery updates, proficiency-aware hints, dependency-aware curriculum adaptation, and safety-aligned prompting.
The demo showcases an end-to-end tutoring workflow: a learner attempts a DSA problem, receives a conceptual hint when stuck, submits a corrected solution, and immediately sees mastery updates and a personalized review interval. We report validation results with simulated learners, showing stable state updates, improved task success with graduated hints, and diverse curriculum coverage. IntelliCode demonstrates how persistent learner modeling, orchestrated multi-agent reasoning, and principled instructional design can be combined to produce transparent and reliable LLM-driven tutoring. 

---
# Automatic Adaptation to Concept Complexity and Subjective Natural Concepts: A Cognitive Model based on Chunking 

**Authors**: Dmitry Bennett, Fernand Gobet  

**Link**: [PDF](https://arxiv.org/pdf/2512.18665)  

**Abstract**: A key issue in cognitive science concerns the fundamental psychological processes that underlie the formation and retrieval of multiple types of concepts in short-term and long-term memory (STM and LTM, respectively). We propose that chunking mechanisms play an essential role and show how the CogAct computational model grounds concept learning in fundamental cognitive processes and structures (such as chunking, attention, STM and LTM). First are the in-principle demonstrations, with CogAct automatically adapting to learn a range of categories from simple logical functions, to artificial categories, to natural raw (as opposed to natural pre-processed) concepts in the dissimilar domains of literature, chess and music. This kind of adaptive learning is difficult for most other psychological models, e.g., with cognitive models stopping at modelling artificial categories and (non-GPT) models based on deep learning requiring task-specific changes to the architecture. Secondly, we offer novel ways of designing human benchmarks for concept learning experiments and simulations accounting for subjectivity, ways to control for individual human experiences, all while keeping to real-life complex categories. We ground CogAct in simulations of subjective conceptual spaces of individual human participants, capturing humans subjective judgements in music, with the models learning from raw music score data without bootstrapping to pre-built knowledge structures. The CogAct simulations are compared to those obtained by a deep-learning model. These findings integrate concept learning and adaptation to complexity into the broader theories of cognitive psychology. Our approach may also be used in psychological applications that move away from modelling the average participant and towards capturing subjective concept space. 

---
# ASTIF: Adaptive Semantic-Temporal Integration for Cryptocurrency Price Forecasting 

**Authors**: Hafiz Saif Ur Rehman, Ling Liu, Kaleem Ullah Qasim  

**Link**: [PDF](https://arxiv.org/pdf/2512.18661)  

**Abstract**: Financial time series forecasting is fundamentally an information fusion challenge, yet most existing models rely on static architectures that struggle to integrate heterogeneous knowledge sources or adjust to rapid regime shifts. Conventional approaches, relying exclusively on historical price sequences, often neglect the semantic drivers of volatility such as policy uncertainty and market narratives. To address these limitations, we propose the ASTIF (Adaptive Semantic-Temporal Integration for Cryptocurrency Price Forecasting), a hybrid intelligent system that adapts its forecasting strategy in real time through confidence-based meta-learning. The framework integrates three complementary components. A dual-channel Small Language Model using MirrorPrompt extracts semantic market cues alongside numerical trends. A hybrid LSTM Random Forest model captures sequential temporal dependencies. A confidence-aware meta-learner functions as an adaptive inference layer, modulating each predictor's contribution based on its real-time uncertainty.
Experimental evaluation on a diverse dataset of AI-focused cryptocurrencies and major technology stocks from 2020 to 2024 shows that ASTIF outperforms leading deep learning and Transformer baselines (e.g., Informer, TFT). The ablation studies further confirm the critical role of the adaptive meta-learning mechanism, which successfully mitigates risk by shifting reliance between semantic and temporal channels during market turbulence. The research contributes a scalable, knowledge-based solution for fusing quantitative and qualitative data in non-stationary environments. 

---
# ChronoDreamer: Action-Conditioned World Model as an Online Simulator for Robotic Planning 

**Authors**: Zhenhao Zhou, Dan Negrut  

**Link**: [PDF](https://arxiv.org/pdf/2512.18619)  

**Abstract**: We present ChronoDreamer, an action-conditioned world model for contact-rich robotic manipulation. Given a history of egocentric RGB frames, contact maps, actions, and joint states, ChronoDreamer predicts future video frames, contact distributions, and joint angles via a spatial-temporal transformer trained with MaskGIT-style masked prediction. Contact is encoded as depth-weighted Gaussian splat images that render 3D forces into a camera-aligned format suitable for vision backbones. At inference, predicted rollouts are evaluated by a vision-language model that reasons about collision likelihood, enabling rejection sampling of unsafe actions before execution. We train and evaluate on DreamerBench, a simulation dataset generated with Project Chrono that provides synchronized RGB, contact splat, proprioception, and physics annotations across rigid and deformable object scenarios. Qualitative results demonstrate that the model preserves spatial coherence during non-contact motion and generates plausible contact predictions, while the LLM-based judge distinguishes collision from non-collision trajectories. 

---
# Assignment-Routing Optimization: Solvers for Problems Under Constraints 

**Authors**: Yuan Qilong, Michal Pavelka  

**Link**: [PDF](https://arxiv.org/pdf/2512.18618)  

**Abstract**: We study the Joint Routing-Assignment (JRA) problem in which items must be assigned one-to-one to placeholders while simultaneously determining a Hamiltonian cycle visiting all nodes exactly once. Extending previous exact MIP solvers with Gurobi and cutting-plane subtour elimination, we develop a solver tailored for practical packaging-planning scenarios with richer this http URL include multiple placeholder options, time-frame restrictions, and multi-class item packaging. Experiments on 46 mobile manipulation datasets demonstrate that the proposed MIP approach achieves global optima with stable and low computation times, significantly outperforming the shaking-based exact solver by up to an orders of magnitude. Compared to greedy baselines, the MIP solutions achieve consistent optimal distances with an average deviation of 14% for simple heuristics, confirming both efficiency and solution quality. The results highlight the practical applicability of MIP-based JRA optimization for robotic packaging, motion planning, and complex logistics . 

---
# Reflective Confidence: Correcting Reasoning Flaws via Online Self-Correction 

**Authors**: Qinglin Zeng, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18605)  

**Abstract**: Large language models (LLMs) have achieved strong performance on complex reasoning tasks using techniques such as chain-of-thought and self-consistency. However, ensemble-based approaches, especially self-consistency which relies on multiple reasoning trajectories, often incur substantial computational overhead. To improve efficiency, prior work has leveraged internal confidence signals, where early stopping strategies such as DeepConf reduce cost by terminating low-confidence trajectories. However, this strategy discards incomplete reasoning paths and wastes partial computation.
We propose reflective confidence, a novel reasoning framework that transforms low-confidence signals from termination indicators into reflection triggers. When confidence falls below a threshold, instead of stopping generation, the model produces a reflection prompt to analyze the current reasoning state, identify potential errors, and continue generation along a corrected trajectory. Experiments on mathematical reasoning benchmarks, including AIME 2025, demonstrate significant accuracy improvements over advanced early-stopping baselines at comparable computational cost, validating the effectiveness of proactive self-correction over passive discarding. 

---
# ESearch-R1: Learning Cost-Aware MLLM Agents for Interactive Embodied Search via Reinforcement Learning 

**Authors**: Weijie Zhou, Xuangtang Xiong, Ye Tian, Lijun Yue, Xinyu Wu, Wei Li, Chaoyang Zhao, Honghui Dong, Ming Tang, Jinqiao Wang, Zhengyou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18571)  

**Abstract**: Multimodal Large Language Models (MLLMs) have empowered embodied agents with remarkable capabilities in planning and reasoning. However, when facing ambiguous natural language instructions (e.g., "fetch the tool" in a cluttered room), current agents often fail to balance the high cost of physical exploration against the cognitive cost of human interaction. They typically treat disambiguation as a passive perception problem, lacking the strategic reasoning to minimize total task execution costs. To bridge this gap, we propose ESearch-R1, a cost-aware embodied reasoning framework that unifies interactive dialogue (Ask), episodic memory retrieval (GetMemory), and physical navigation (Navigate) into a single decision process. We introduce HC-GRPO (Heterogeneous Cost-Aware Group Relative Policy Optimization). Unlike traditional PPO which relies on a separate value critic, HC-GRPO optimizes the MLLM by sampling groups of reasoning trajectories and reinforcing those that achieve the optimal trade-off between information gain and heterogeneous costs (e.g., navigate time, and human attention). Extensive experiments in AI2-THOR demonstrate that ESearch-R1 significantly outperforms standard ReAct-based agents. It improves task success rates while reducing total operational costs by approximately 50\%, validating the effectiveness of GRPO in aligning MLLM agents with physical world constraints. 

---
# Vox Deorum: A Hybrid LLM Architecture for 4X / Grand Strategy Game AI -- Lessons from Civilization V 

**Authors**: John Chen, Sihan Cheng, Can Gurkan, Ryan Lay, Moez Salahuddin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18564)  

**Abstract**: Large Language Models' capacity to reason in natural language makes them uniquely promising for 4X and grand strategy games, enabling more natural human-AI gameplay interactions such as collaboration and negotiation. However, these games present unique challenges due to their complexity and long-horizon nature, while latency and cost factors may hinder LLMs' real-world deployment. Working on a classic 4X strategy game, Sid Meier's Civilization V with the Vox Populi mod, we introduce Vox Deorum, a hybrid LLM+X architecture. Our layered technical design empowers LLMs to handle macro-strategic reasoning, delegating tactical execution to subsystems (e.g., algorithmic AI or reinforcement learning AI in the future). We validate our approach through 2,327 complete games, comparing two open-source LLMs with a simple prompt against Vox Populi's enhanced AI. Results show that LLMs achieve competitive end-to-end gameplay while exhibiting play styles that diverge substantially from algorithmic AI and from each other. Our work establishes a viable architecture for integrating LLMs in commercial 4X games, opening new opportunities for game design and agentic AI research. 

---
# Large Language Models as Discounted Bayesian Filters 

**Authors**: Jensen Zhang, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18489)  

**Abstract**: Large Language Models (LLMs) demonstrate strong few-shot generalization through in-context learning, yet their reasoning in dynamic and stochastic environments remains opaque. Prior studies mainly focus on static tasks and overlook the online adaptation required when beliefs must be continuously updated, which is a key capability for LLMs acting as world models or agents. We introduce a Bayesian filtering framework to evaluate online inference in LLMs. Our probabilistic probe suite spans both multivariate discrete distributions, such as dice rolls, and continuous distributions, such as Gaussian processes, where ground-truth parameters shift over time. We find that while LLM belief updates resemble Bayesian posteriors, they are more accurately characterized by an exponential forgetting filter with a model-specific discount factor smaller than one. This reveals systematic discounting of older evidence that varies significantly across model architectures. Although inherent priors are often miscalibrated, the updating mechanism itself remains structured and principled. We further validate these findings in a simulated agent task and propose prompting strategies that effectively recalibrate priors with minimal computational cost. 

---
# Insider Threat Detection Using GCN and Bi-LSTM with Explicit and Implicit Graph Representations 

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang, Deepa Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2512.18483)  

**Abstract**: Insider threat detection (ITD) is challenging due to the subtle and concealed nature of malicious activities performed by trusted users. This paper proposes a post-hoc ITD framework that integrates explicit and implicit graph representations with temporal modelling to capture complex user behaviour patterns. An explicit graph is constructed using predefined organisational rules to model direct relationships among user activities. To mitigate noise and limitations in this hand-crafted structure, an implicit graph is learned from feature similarities using the Gumbel-Softmax trick, enabling the discovery of latent behavioural relationships. Separate Graph Convolutional Networks (GCNs) process the explicit and implicit graphs to generate node embeddings, which are concatenated and refined through an attention mechanism to emphasise threat-relevant features. The refined representations are then passed to a bidirectional Long Short-Term Memory (Bi-LSTM) network to capture temporal dependencies in user behaviour. Activities are flagged as anomalous when their probability scores fall below a predefined threshold. Extensive experiments on CERT r5.2 and r6.2 datasets demonstrate that the proposed framework outperforms state-of-the-art methods. On r5.2, the model achieves an AUC of 98.62, a detection rate of 100%, and a false positive rate of 0.05. On the more challenging r6.2 dataset, it attains an AUC of 88.48, a detection rate of 80.15%, and a false positive rate of 0.15, highlighting the effectiveness of combining graph-based and temporal representations for robust ITD. 

---
# Agent-Based Output Drift Detection for Breast Cancer Response Prediction in a Multisite Clinical Decision Support System 

**Authors**: Xavier Rafael-Palou, Jose Munuera, Ana Jimenez-Pastor, Richard Osuala, Karim Lekadir, Oliver Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2512.18450)  

**Abstract**: Modern clinical decision support systems can concurrently serve multiple, independent medical imaging institutions, but their predictive performance may degrade across sites due to variations in patient populations, imaging hardware, and acquisition protocols. Continuous surveillance of predictive model outputs offers a safe and reliable approach for identifying such distributional shifts without ground truth labels. However, most existing methods rely on centralized monitoring of aggregated predictions, overlooking site-specific drift dynamics. We propose an agent-based framework for detecting drift and assessing its severity in multisite clinical AI systems. To evaluate its effectiveness, we simulate a multi-center environment for output-based drift detection, assigning each site a drift monitoring agent that performs batch-wise comparisons of model outputs against a reference distribution. We analyse several multi-center monitoring schemes, that differ in how the reference is obtained (site-specific, global, production-only and adaptive), alongside a centralized baseline. Results on real-world breast cancer imaging data using a pathological complete response prediction model shows that all multi-center schemes outperform centralized monitoring, with F1-score improvements up to 10.3% in drift detection. In the absence of site-specific references, the adaptive scheme performs best, with F1-scores of 74.3% for drift detection and 83.7% for drift severity classification. These findings suggest that adaptive, site-aware agent-based drift monitoring can enhance reliability of multisite clinical decision support systems. 

---
# Few-Shot Learning of a Graph-Based Neural Network Model Without Backpropagation 

**Authors**: Mykyta Lapin, Kostiantyn Bokhan, Yurii Parzhyn  

**Link**: [PDF](https://arxiv.org/pdf/2512.18412)  

**Abstract**: We propose a structural-graph approach to classifying contour images in a few-shot regime without using backpropagation. The core idea is to make structure the carrier of explanations: an image is encoded as an attributed graph (critical points and lines represented as nodes with geometric attributes), and generalization is achieved via the formation of concept attractors (class-level concept graphs). Purpose. To design and experimentally validate an architecture in which class concepts are formed from a handful of examples (5 - 6 per class) through structural and parametric reductions, providing transparent decisions and eliminating backpropagation. Methods. Contour vectorization is followed by constructing a bipartite graph (Point/Line as nodes) with normalized geometric attributes such as coordinates, length, angle, and direction; reductions include the elimination of unstable substructures or noise and the alignment of paths between critical points. Concepts are formed by iterative composition of samples, and classification is performed by selecting the best graph-to-concept match (using approximated GED). Results. On an MNIST subset with 5 - 6 base examples per class (single epoch), we obtain a consistent accuracy of around 82% with full traceability of decisions: misclassifications can be explained by explicit structural similarities. An indicative comparison with SVM, MLP, CNN, as well as metric and meta-learning baselines, is provided. The structural-graph scheme with concept attractors enables few-shot learning without backpropagation and offers built-in explanations through the explicit graph structure. Limitations concern the computational cost of GED and the quality of skeletonization; promising directions include classification-algorithm optimization, work with static scenes, and associative recognition. 

---
# Monitoring Monitorability 

**Authors**: Melody Y. Guan, Miles Wang, Micah Carroll, Zehao Dou, Annie Y. Wei, Marcus Williams, Benjamin Arnav, Joost Huizinga, Ian Kivlichan, Mia Glaese, Jakub Pachocki, Bowen Baker  

**Link**: [PDF](https://arxiv.org/pdf/2512.18311)  

**Abstract**: Observability into the decision making of modern AI systems may be required to safely deploy increasingly capable agents. Monitoring the chain-of-thought (CoT) of today's reasoning models has proven effective for detecting misbehavior. However, this "monitorability" may be fragile under different training procedures, data sources, or even continued system scaling. To measure and track monitorability, we propose three evaluation archetypes (intervention, process, and outcome-property) and a new monitorability metric, and introduce a broad evaluation suite. We demonstrate that these evaluations can catch simple model organisms trained to have obfuscated CoTs, and that CoT monitoring is more effective than action-only monitoring in practical settings. We compare the monitorability of various frontier models and find that most models are fairly, but not perfectly, monitorable. We also evaluate how monitorability scales with inference-time compute, reinforcement learning optimization, and pre-training model size. We find that longer CoTs are generally more monitorable and that RL optimization does not materially decrease monitorability even at the current frontier scale. Notably, we find that for a model at a low reasoning effort, we could instead deploy a smaller model at a higher reasoning effort (thereby matching capabilities) and obtain a higher monitorability, albeit at a higher overall inference compute cost. We further investigate agent-monitor scaling trends and find that scaling a weak monitor's test-time compute when monitoring a strong agent increases monitorability. Giving the weak monitor access to CoT not only improves monitorability, but it steepens the monitor's test-time compute to monitorability scaling trend. Finally, we show we can improve monitorability by asking models follow-up questions and giving their follow-up CoT to the monitor. 

---
# Intelligent Human-Machine Partnership for Manufacturing: Enhancing Warehouse Planning through Simulation-Driven Knowledge Graphs and LLM Collaboration 

**Authors**: Himabindu Thogaru, Saisubramaniam Gopalakrishnan, Zishan Ahmad, Anirudh Deodhar  

**Link**: [PDF](https://arxiv.org/pdf/2512.18265)  

**Abstract**: Manufacturing planners face complex operational challenges that require seamless collaboration between human expertise and intelligent systems to achieve optimal performance in modern production environments. Traditional approaches to analyzing simulation-based manufacturing data often create barriers between human decision-makers and critical operational insights, limiting effective partnership in manufacturing planning. Our framework establishes a collaborative intelligence system integrating Knowledge Graphs and Large Language Model-based agents to bridge this gap, empowering manufacturing professionals through natural language interfaces for complex operational analysis. The system transforms simulation data into semantically rich representations, enabling planners to interact naturally with operational insights without specialized expertise. A collaborative LLM agent works alongside human decision-makers, employing iterative reasoning that mirrors human analytical thinking while generating precise queries for knowledge extraction and providing transparent validation. This partnership approach to manufacturing bottleneck identification, validated through operational scenarios, demonstrates enhanced performance while maintaining human oversight and decision authority. For operational inquiries, the system achieves near-perfect accuracy through natural language interaction. For investigative scenarios requiring collaborative analysis, we demonstrate the framework's effectiveness in supporting human experts to uncover interconnected operational issues that enhance understanding and decision-making. This work advances collaborative manufacturing by creating intuitive methods for actionable insights, reducing cognitive load while amplifying human analytical capabilities in evolving manufacturing ecosystems. 

---
# MSC-180: A Benchmark for Automated Formal Theorem Proving from Mathematical Subject Classification 

**Authors**: Sirui Li, Wangyue Lu, Xiaorui Shi, Ke Weng, Haozhe Sun, Minghe Yu, Tiancheng Zhang, Ge Yu, Hengyu Liu, Lun Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.18256)  

**Abstract**: Automated Theorem Proving (ATP) represents a core research direction in artificial intelligence for achieving formal reasoning and verification, playing a significant role in advancing machine intelligence. However, current large language model (LLM)-based theorem provers suffer from limitations such as restricted domain coverage and weak generalization in mathematical reasoning. To address these issues, we propose MSC-180, a benchmark for evaluation based on the MSC2020 mathematical subject classification. It comprises 180 formal verification problems, 3 advanced problems from each of 60 mathematical branches, spanning from undergraduate to graduate levels. Each problem has undergone multiple rounds of verification and refinement by domain experts to ensure formal accuracy. Evaluations of state-of-the-art LLM-based theorem provers under the pass@32 setting reveal that the best model achieves only an 18.89% overall pass rate, with prominent issues including significant domain bias (maximum domain coverage 41.7%) and a difficulty gap (significantly lower pass rates on graduate-level problems). To further quantify performance variability across mathematical domains, we introduce the coefficient of variation (CV) as an evaluation metric. The observed CV values are 4-6 times higher than the statistical high-variability threshold, indicating that the models still rely on pattern matching from training corpora rather than possessing transferable reasoning mechanisms and systematic generalization capabilities. MSC-180, together with its multi-dimensional evaluation framework, provides a discriminative and systematic benchmark for driving the development of next-generation AI systems with genuine mathematical reasoning abilities. 

---
# Sophia: A Persistent Agent Framework of Artificial Life 

**Authors**: Mingyang Sun, Feng Hong, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18202)  

**Abstract**: The development of LLMs has elevated AI agents from task-specific tools to long-lived, decision-making entities. Yet, most architectures remain static and reactive, tethered to manually defined, narrow scenarios. These systems excel at perception (System 1) and deliberation (System 2) but lack a persistent meta-layer to maintain identity, verify reasoning, and align short-term actions with long-term survival. We first propose a third stratum, System 3, that presides over the agent's narrative identity and long-horizon adaptation. The framework maps selected psychological constructs to concrete computational modules, thereby translating abstract notions of artificial life into implementable design requirements. The ideas coalesce in Sophia, a "Persistent Agent" wrapper that grafts a continuous self-improvement loop onto any LLM-centric System 1/2 stack. Sophia is driven by four synergistic mechanisms: process-supervised thought search, narrative memory, user and self modeling, and a hybrid reward system. Together, they transform repetitive reasoning into a self-driven, autobiographical process, enabling identity continuity and transparent behavioral explanations. Although the paper is primarily conceptual, we provide a compact engineering prototype to anchor the discussion. Quantitatively, Sophia independently initiates and executes various intrinsic tasks while achieving an 80% reduction in reasoning steps for recurring operations. Notably, meta-cognitive persistence yielded a 40% gain in success for high-complexity tasks, effectively bridging the performance gap between simple and sophisticated goals. Qualitatively, System 3 exhibited a coherent narrative identity and an innate capacity for task organization. By fusing psychological insight with a lightweight reinforcement-learning core, the persistent agent architecture advances a possible practical pathway toward artificial life. 

---
# External Hippocampus: Topological Cognitive Maps for Guiding Large Language Model Reasoning 

**Authors**: Jian Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.18190)  

**Abstract**: This paper proposes the External Hippocampus framework, which models language model reasoning from a cognitive dynamics perspective as the flow of information energy in semantic space. Unlike traditional weight-space optimization methods, this framework constructs topological cognitive maps through dimensionality reduction projection, enabling precise navigation and intervention of energy flow at test time while avoiding substantial computational requirements and demonstrating predictable intervention patterns. The method effectively addresses the cognitive deadlock problem in multi-step reasoning for small models. Experiments on models <=7B parameters show: map-guided methods achieve 81.20% accuracy on 500 challenging problems (relative baseline +16.80%), reduce reasoning time by >= 15x, with key findings revealing that reasoning stagnation manifests as "Cognitive Vortex" and low-entropy potential wells, while temperature perturbations effectively restart energy flow. The framework requires no additional training, possesses autonomous growth capability, and provides an efficient and controllable topological-aware solution for small model reasoning. 

---
# NL2CA: Auto-formalizing Cognitive Decision-Making from Natural Language Using an Unsupervised CriticNL2LTL Framework 

**Authors**: Zihao Deng, Yijia Li, Renrui Zhang, Peijun Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.18189)  

**Abstract**: Cognitive computing models offer a formal and interpretable way to characterize human's deliberation and decision-making, yet their development remains labor-intensive. In this paper, we propose NL2CA, a novel method for auto-formalizing cognitive decision-making rules from natural language descriptions of human experience. Different from most related work that exploits either pure manual or human guided interactive modeling, our method is fully automated without any human intervention. The approach first translates text into Linear Temporal Logic (LTL) using a fine-tuned large language model (LLM), then refines the logic via an unsupervised Critic Tree, and finally transforms the output into executable production rules compatible with symbolic cognitive frameworks. Based on the resulted rules, a cognitive agent is further constructed and optimized through cognitive reinforcement learning according to the real-world behavioral data. Our method is validated in two domains: (1) NL-to-LTL translation, where our CriticNL2LTL module achieves consistent performance across both expert and large-scale benchmarks without human-in-the-loop feed-backs, and (2) cognitive driving simulation, where agents automatically constructed from human interviews have successfully learned the diverse decision patterns of about 70 trials in different critical scenarios. Experimental results demonstrate that NL2CA enables scalable, interpretable, and human-aligned cognitive modeling from unstructured textual data, offering a novel paradigm to automatically design symbolic cognitive agents. 

---
# NEURO-GUARD: Neuro-Symbolic Generalization and Unbiased Adaptive Routing for Diagnostics -- Explainable Medical AI 

**Authors**: Midhat Urooj, Ayan Banerjee, Sandeep Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2512.18177)  

**Abstract**: Accurate yet interpretable image-based diagnosis remains a central challenge in medical AI, particularly in settings characterized by limited data, subtle visual cues, and high-stakes clinical decision-making. Most existing vision models rely on purely data-driven learning and produce black-box predictions with limited interpretability and poor cross-domain generalization, hindering their real-world clinical adoption. We present NEURO-GUARD, a novel knowledge-guided vision framework that integrates Vision Transformers (ViTs) with language-driven reasoning to improve performance, transparency, and domain robustness. NEURO-GUARD employs a retrieval-augmented generation (RAG) mechanism for self-verification, in which a large language model (LLM) iteratively generates, evaluates, and refines feature-extraction code for medical images. By grounding this process in clinical guidelines and expert knowledge, the framework progressively enhances feature detection and classification beyond purely data-driven baselines. Extensive experiments on diabetic retinopathy classification across four benchmark datasets APTOS, EyePACS, Messidor-1, and Messidor-2 demonstrate that NEURO-GUARD improves accuracy by 6.2% over a ViT-only baseline (84.69% vs. 78.4%) and achieves a 5% gain in domain generalization. Additional evaluations on MRI-based seizure detection further confirm its cross-domain robustness, consistently outperforming existing methods.
Overall, NEURO-GUARD bridges symbolic medical reasoning with subsymbolic visual learning, enabling interpretable, knowledge-aware, and generalizable medical image diagnosis while achieving state-of-the-art performance across multiple datasets. 

---
# Propose, Solve, Verify: Self-Play Through Formal Verification 

**Authors**: Alex Wilf, Pranjal Aggarwal, Bryan Parno, Daniel Fried, Louis-Philippe Morency, Paul Pu Liang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2512.18160)  

**Abstract**: Training models through self-play alone (without any human data) has been a longstanding goal in AI, but its effectiveness for training large language models remains unclear, particularly in code generation where rewards based on unit tests are brittle and prone to error propagation. We study self-play in the verified code generation setting, where formal verification provides reliable correctness signals. We introduce Propose, Solve, Verify (PSV) a simple self-play framework where formal verification signals are used to create a proposer capable of generating challenging synthetic problems and a solver trained via expert iteration. We use PSV to train PSV-Verus, which across three benchmarks improves pass@1 by up to 9.6x over inference-only and expert-iteration baselines. We show that performance scales with the number of generated questions and training iterations, and through ablations identify formal verification and difficulty-aware proposal as essential ingredients for successful self-play. 

---
# Unifying Causal Reinforcement Learning: Survey, Taxonomy, Algorithms and Applications 

**Authors**: Cristiano da Costa Cunha, Wei Liu, Tim French, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2512.18135)  

**Abstract**: Integrating causal inference (CI) with reinforcement learning (RL) has emerged as a powerful paradigm to address critical limitations in classical RL, including low explainability, lack of robustness and generalization failures. Traditional RL techniques, which typically rely on correlation-driven decision-making, struggle when faced with distribution shifts, confounding variables, and dynamic environments. Causal reinforcement learning (CRL), leveraging the foundational principles of causal inference, offers promising solutions to these challenges by explicitly modeling cause-and-effect relationships. In this survey, we systematically review recent advancements at the intersection of causal inference and RL. We categorize existing approaches into causal representation learning, counterfactual policy optimization, offline causal RL, causal transfer learning, and causal explainability. Through this structured analysis, we identify prevailing challenges, highlight empirical successes in practical applications, and discuss open problems. Finally, we provide future research directions, underscoring the potential of CRL for developing robust, generalizable, and interpretable artificial intelligence systems. 

---
# Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap 

**Authors**: Zijun Wang, Yijiahao Qi, Hanqiu Chen, Zishen Wan, Gongjin Sun, Dongyang Li, Shuyi Pei, Cong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2512.18126)  

**Abstract**: Mixture-of-Agents (MoA) inference can suffer from dense inter-agent communication and low hardware utilization, which jointly inflate serving latency. We present a serving design that targets these bottlenecks through an algorithm-system co-design. First, we replace dense agent interaction graphs with a hierarchical tree topology that induces structured sparsity in inter-agent communication. Second, we introduce a runtime adaptive mechanism that selectively terminates or skips downstream agent invocations using semantic agreement and confidence signals from intermediate outputs. Third, we pipeline agent execution by overlapping incremental prefilling with decoding across dependency-related agents, improving utilization and reducing inference latency. Across representative tasks, this approach substantially reduces end-to-end latency (up to 90%) while maintaining comparable accuracy (within $\pm$1%) relative to dense-connectivity MoA baselines, and can improve accuracy in certain settings. 

---
# Rethinking Multi-Agent Intelligence Through the Lens of Small-World Networks 

**Authors**: Boxuan Wang, Zhuoyun Li, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2512.18094)  

**Abstract**: Large language models (LLMs) have enabled multi-agent systems (MAS) in which multiple agents argue, critique, and coordinate to solve complex tasks, making communication topology a first-class design choice. Yet most existing LLM-based MAS either adopt fully connected graphs, simple sparse rings, or ad-hoc dynamic selection, with little structural guidance. In this work, we revisit classic theory on small-world (SW) networks and ask: what changes if we treat SW connectivity as a design prior for MAS? We first bridge insights from neuroscience and complex networks to MAS, highlighting how SW structures balance local clustering and long-range integration. Using multi-agent debate (MAD) as a controlled testbed, experiment results show that SW connectivity yields nearly the same accuracy and token cost, while substantially stabilizing consensus trajectories. Building on this, we introduce an uncertainty-guided rewiring scheme for scaling MAS, where long-range shortcuts are added between epistemically divergent agents using LLM-oriented uncertainty signals (e.g., semantic entropy). This yields controllable SW structures that adapt to task difficulty and agent heterogeneity. Finally, we discuss broader implications of SW priors for MAS design, framing them as stabilizers of reasoning, enhancers of robustness, scalable coordinators, and inductive biases for emergent cognitive roles. 

---
# Faithful and Stable Neuron Explanations for Trustworthy Mechanistic Interpretability 

**Authors**: Ge Yan, Tuomas Oikarinen, Tsui-Wei, Weng  

**Link**: [PDF](https://arxiv.org/pdf/2512.18092)  

**Abstract**: Neuron identification is a popular tool in mechanistic interpretability, aiming to uncover the human-interpretable concepts represented by individual neurons in deep networks. While algorithms such as Network Dissection and CLIP-Dissect achieve great empirical success, a rigorous theoretical foundation remains absent, which is crucial to enable trustworthy and reliable explanations. In this work, we observe that neuron identification can be viewed as the inverse process of machine learning, which allows us to derive guarantees for neuron explanations. Based on this insight, we present the first theoretical analysis of two fundamental challenges: (1) Faithfulness: whether the identified concept faithfully represents the neuron's underlying function and (2) Stability: whether the identification results are consistent across probing datasets. We derive generalization bounds for widely used similarity metrics (e.g. accuracy, AUROC, IoU) to guarantee faithfulness, and propose a bootstrap ensemble procedure that quantifies stability along with BE (Bootstrap Explanation) method to generate concept prediction sets with guaranteed coverage probability. Experiments on both synthetic and real data validate our theoretical results and demonstrate the practicality of our method, providing an important step toward trustworthy neuron identification. 

---
# Conflict-Driven Clause Learning with VSIDS Heuristics for Discrete Facility Layout 

**Authors**: Joshua Gibson, Kapil Dhakal  

**Link**: [PDF](https://arxiv.org/pdf/2512.18034)  

**Abstract**: This paper studies the use of Conflict-Driven Clause Learning (CDCL) with VSIDS heuristics as a computational engine for discrete facility layout problems. The facility layout problem is modeled as a combinatorial assignment problem with dense logical structure arising from adjacency, separation, and slot-availability constraints. We develop a CNF-based formulation for layout feasibility and compare CDCL-based SAT solving against CP-SAT and MILP formulations under a unified benchmarking framework. Empirical results show that CDCL exhibits near-constant runtime behavior for feasibility detection across increasing problem sizes and constraint densities, while CP-SAT and MILP display polynomial and exponential scaling respectively. To address the limitation of CDCL in objective optimization, we introduce two hybrid architectures that combine CDCL-based feasibility search with CP-SAT optimization. The first architecture rapidly enumerates feasible layouts to trade optimality for speed, while the second uses CDCL to generate warm-start solutions that accelerate exact optimization. The results demonstrate that hybrid approaches can significantly reduce time-to-solution while preserving correctness guarantees, clarifying the algorithmic trade-offs between clause-learning search and exact optimization methods in large-scale discrete layout problems. 

---
# WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion 

**Authors**: Hanyang Kong, Xingyi Yang, Xiaoxu Zheng, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19678)  

**Abstract**: Generating long-range, geometrically consistent video presents a fundamental dilemma: while consistency demands strict adherence to 3D geometry in pixel space, state-of-the-art generative models operate most effectively in a camera-conditioned latent space. This disconnect causes current methods to struggle with occluded areas and complex camera trajectories. To bridge this gap, we propose WorldWarp, a framework that couples a 3D structural anchor with a 2D generative refiner. To establish geometric grounding, WorldWarp maintains an online 3D geometric cache built via Gaussian Splatting (3DGS). By explicitly warping historical content into novel views, this cache acts as a structural scaffold, ensuring each new frame respects prior geometry. However, static warping inevitably leaves holes and artifacts due to occlusions. We address this using a Spatio-Temporal Diffusion (ST-Diff) model designed for a "fill-and-revise" objective. Our key innovation is a spatio-temporal varying noise schedule: blank regions receive full noise to trigger generation, while warped regions receive partial noise to enable refinement. By dynamically updating the 3D cache at every step, WorldWarp maintains consistency across video chunks. Consequently, it achieves state-of-the-art fidelity by ensuring that 3D logic guides structure while diffusion logic perfects texture. Project page: \href{this https URL}{this https URL}. 

---
# Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies 

**Authors**: Yuqiao Tan, Minzheng Wang, Shizhu He, Huanxuan Liao, Chengfeng Zhao, Qiunan Lu, Tian Liang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19673)  

**Abstract**: Existing reinforcement learning (RL) approaches treat large language models (LLMs) as a single unified policy, overlooking their internal mechanisms. Understanding how policy evolves across layers and modules is therefore crucial for enabling more targeted optimization and raveling out complex reasoning mechanisms. In this paper, we decompose the language model policy by leveraging the intrinsic split of the Transformer residual stream and the equivalence between the composition of hidden states with the unembedding matrix and the resulting samplable policy. This decomposition reveals Internal Layer Policies, corresponding to contributions from individual layers, and Internal Modular Policies, which align with the self-attention and feed-forward network (FFN) components within each layer. By analyzing the entropy of internal policy, we find that: (a) Early layers keep high entropy for exploration, top layers converge to near-zero entropy for refinement, with convergence patterns varying across model series. (b) LLama's prediction space rapidly converges in the final layer, whereas Qwen-series models, especially Qwen3, exhibit a more human-like, progressively structured reasoning pattern. Motivated by these findings, we propose Bottom-up Policy Optimization (BuPO), a novel RL paradigm that directly optimizes the internal layer policy during early training. By aligning training objective at lower layer, BuPO reconstructs foundational reasoning capabilities and achieves superior performance. Extensive experiments on complex reasoning benchmarks demonstrates the effectiveness of our method. Our code is available at this https URL. 

---
# Beyond CLIP: Knowledge-Enhanced Multimodal Transformers for Cross-Modal Alignment in Diabetic Retinopathy Diagnosis 

**Authors**: Argha Kamal Samanta, Harshika Goyal, Vasudha Joshi, Tushar Mungle, Pabitra Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2512.19663)  

**Abstract**: Diabetic retinopathy (DR) is a leading cause of preventable blindness worldwide, demanding accurate automated diagnostic systems. While general-domain vision-language models like Contrastive Language-Image Pre-Training (CLIP) perform well on natural image tasks, they struggle in medical domain applications, particularly in cross-modal retrieval for ophthalmological images. We propose a novel knowledge-enhanced joint embedding framework that integrates retinal fundus images, clinical text, and structured patient data through a multimodal transformer architecture to address the critical gap in medical image-text alignment. Our approach employs separate encoders for each modality: a Vision Transformer (ViT-B/16) for retinal images, Bio-ClinicalBERT for clinical narratives, and a multilayer perceptron for structured demographic and clinical features. These modalities are fused through a joint transformer with modality-specific embeddings, trained using multiple objectives including contrastive losses between modality pairs, reconstruction losses for images and text, and classification losses for DR severity grading according to ICDR and SDRG schemes. Experimental results on the Brazilian Multilabel Ophthalmological Dataset (BRSET) demonstrate significant improvements over baseline models. Our framework achieves near-perfect text-to-image retrieval performance with Recall@1 of 99.94% compared to fine-tuned CLIP's 1.29%, while maintaining state-of-the-art classification accuracy of 97.05% for SDRG and 97.97% for ICDR. Furthermore, zero-shot evaluation on the unseen DeepEyeNet dataset validates strong generalizability with 93.95% Recall@1 versus 0.22% for fine-tuned CLIP. These results demonstrate that our multimodal training approach effectively captures cross-modal relationships in the medical domain, establishing both superior retrieval capabilities and robust diagnostic performance. 

---
# Clustering with Label Consistency 

**Authors**: Diptarka Chakraborty, Hendrik Fichtenberger, Bernhard Haeupler, Silvio Lattanzi, Ashkan Norouzi-Fard, Ola Svensson  

**Link**: [PDF](https://arxiv.org/pdf/2512.19654)  

**Abstract**: Designing efficient, effective, and consistent metric clustering algorithms is a significant challenge attracting growing attention. Traditional approaches focus on the stability of cluster centers; unfortunately, this neglects the real-world need for stable point labels, i.e., stable assignments of points to named sets (clusters). In this paper, we address this gap by initiating the study of label-consistent metric clustering. We first introduce a new notion of consistency, measuring the label distance between two consecutive solutions. Then, armed with this new definition, we design new consistent approximation algorithms for the classical $k$-center and $k$-median problems. 

---
# Exploring the features used for summary evaluation by Human and GPT 

**Authors**: Zahra Sadeghi, Evangelos Milios, Frank Rudzicz  

**Link**: [PDF](https://arxiv.org/pdf/2512.19620)  

**Abstract**: Summary assessment involves evaluating how well a generated summary reflects the key ideas and meaning of the source text, requiring a deep understanding of the content. Large Language Models (LLMs) have been used to automate this process, acting as judges to evaluate summaries with respect to the original text. While previous research investigated the alignment between LLMs and Human responses, it is not yet well understood what properties or features are exploited by them when asked to evaluate based on a particular quality dimension, and there has not been much attention towards mapping between evaluation scores and metrics. In this paper, we address this issue and discover features aligned with Human and Generative Pre-trained Transformers (GPTs) responses by studying statistical and machine learning metrics. Furthermore, we show that instructing GPTs to employ metrics used by Human can improve their judgment and conforming them better with human responses. 

---
# MapTrace: Scalable Data Generation for Route Tracing on Maps 

**Authors**: Artemis Panagopoulou, Aveek Purohit, Achin Kulshrestha, Soroosh Yazdani, Mohit Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2512.19609)  

**Abstract**: While Multimodal Large Language Models have achieved human-like performance on many visual and textual reasoning tasks, their proficiency in fine-grained spatial understanding, such as route tracing on maps remains limited. Unlike humans, who can quickly learn to parse and navigate maps, current models often fail to respect fundamental path constraints, in part due to the prohibitive cost and difficulty of collecting large-scale, pixel-accurate path annotations. To address this, we introduce a scalable synthetic data generation pipeline that leverages synthetic map images and pixel-level parsing to automatically produce precise annotations for this challenging task. Using this pipeline, we construct a fine-tuning dataset of 23k path samples across 4k maps, enabling models to acquire more human-like spatial capabilities. Using this dataset, we fine-tune both open-source and proprietary MLLMs. Results on MapBench show that finetuning substantially improves robustness, raising success rates by up to 6.4 points, while also reducing path-tracing error (NDTW). These gains highlight that fine-grained spatial reasoning, absent in pretrained models, can be explicitly taught with synthetic supervision. 

---
# LeLaR: The First In-Orbit Demonstration of an AI-Based Satellite Attitude Controller 

**Authors**: Kirill Djebko, Tom Baumann, Erik Dilger, Frank Puppe, Sergio Montenegro  

**Link**: [PDF](https://arxiv.org/pdf/2512.19576)  

**Abstract**: Attitude control is essential for many satellite missions. Classical controllers, however, are time-consuming to design and sensitive to model uncertainties and variations in operational boundary conditions. Deep Reinforcement Learning (DRL) offers a promising alternative by learning adaptive control strategies through autonomous interaction with a simulation environment. Overcoming the Sim2Real gap, which involves deploying an agent trained in simulation onto the real physical satellite, remains a significant challenge. In this work, we present the first successful in-orbit demonstration of an AI-based attitude controller for inertial pointing maneuvers. The controller was trained entirely in simulation and deployed to the InnoCube 3U nanosatellite, which was developed by the Julius-Maximilians-Universität Würzburg in cooperation with the Technische Universität Berlin, and launched in January 2025. We present the AI agent design, the methodology of the training procedure, the discrepancies between the simulation and the observed behavior of the real satellite, and a comparison of the AI-based attitude controller with the classical PD controller of InnoCube. Steady-state metrics confirm the robust performance of the AI-based controller during repeated in-orbit maneuvers. 

---
# The Epistemological Consequences of Large Language Models: Rethinking collective intelligence and institutional knowledge 

**Authors**: Angjelin Hila  

**Link**: [PDF](https://arxiv.org/pdf/2512.19570)  

**Abstract**: We examine epistemological threats posed by human and LLM interaction. We develop collective epistemology as a theory of epistemic warrant distributed across human collectives, using bounded rationality and dual process theory as background. We distinguish internalist justification, defined as reflective understanding of why a proposition is true, from externalist justification, defined as reliable transmission of truths. Both are necessary for collective rationality, but only internalist justification produces reflective knowledge. We specify reflective knowledge as follows: agents understand the evaluative basis of a claim, when that basis is unavailable agents consistently assess the reliability of truth sources, and agents have a duty to apply these standards within their domains of competence. We argue that LLMs approximate externalist reliabilism because they can reliably transmit information whose justificatory basis is established elsewhere, but they do not themselves possess reflective justification. Widespread outsourcing of reflective work to reliable LLM outputs can weaken reflective standards of justification, disincentivize comprehension, and reduce agents' capacity to meet professional and civic epistemic duties. To mitigate these risks, we propose a three tier norm program that includes an epistemic interaction model for individual use, institutional and organizational frameworks that seed and enforce norms for epistemically optimal outcomes, and deontic constraints at organizational and or legislative levels that instantiate discursive norms and curb epistemic vices. 

---
# Owning the Intelligence: Global AI Patents Landscape and Europe's Quest for Technological Sovereignty 

**Authors**: Lapo Santarlasci, Armando Rungi, Loredana Fattorini, Nestor Maslej  

**Link**: [PDF](https://arxiv.org/pdf/2512.19569)  

**Abstract**: Artificial intelligence has become a key arena of global technological competition and a central concern for Europe's quest for technological sovereignty. This paper analyzes global AI patenting from 2010 to 2023 to assess Europe's position in an increasingly bipolar innovation landscape dominated by the United States and China. Using linked patent, firm, ownership, and citation data, we examine the geography, specialization, and international diffusion of AI innovation. We find a highly concentrated patent landscape: China leads in patent volumes, while the United States dominates in citation impact and technological influence. Europe accounts for a limited share of AI patents but exhibits signals of relatively high patent quality. Technological proximity reveals global convergence toward U.S. innovation trajectories, with Europe remaining fragmented rather than forming an autonomous pole. Gravity-model estimates show that cross-border AI knowledge flows are driven primarily by technological capability and specialization, while geographic and institutional factors play a secondary role. EU membership does not significantly enhance intra-European knowledge diffusion, suggesting that technological capacity, rather than political integration, underpins participation in global AI innovation networks. 

---
# Results of the 2024 CommonRoad Motion Planning Competition for Autonomous Vehicles 

**Authors**: Yanliang Huang, Xia Yan, Peiran Yin, Zhenduo Zhang, Zeyan Shao, Youran Wang, Haoliang Huang, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2512.19564)  

**Abstract**: Over the past decade, a wide range of motion planning approaches for autonomous vehicles has been developed to handle increasingly complex traffic scenarios. However, these approaches are rarely compared on standardized benchmarks, limiting the assessment of relative strengths and weaknesses. To address this gap, we present the setup and results of the 4th CommonRoad Motion Planning Competition held in 2024, conducted using the CommonRoad benchmark suite. This annual competition provides an open-source and reproducible framework for benchmarking motion planning algorithms. The benchmark scenarios span highway and urban environments with diverse traffic participants, including passenger cars, buses, and bicycles. Planner performance is evaluated along four dimensions: efficiency, safety, comfort, and compliance with selected traffic rules. This report introduces the competition format and provides a comparison of representative high-performing planners from the 2023 and 2024 editions. 

---
# REALM: A Real-to-Sim Validated Benchmark for Generalization in Robotic Manipulation 

**Authors**: Martin Sedlacek, Pavlo Yefanov, Georgy Ponimatkin, Jai Bardhan, Simon Pilc, Mederic Fourmy, Evangelos Kazakos, Cees G. M. Snoek, Josef Sivic, Vladimir Petrik  

**Link**: [PDF](https://arxiv.org/pdf/2512.19562)  

**Abstract**: Vision-Language-Action (VLA) models empower robots to understand and execute tasks described by natural language instructions. However, a key challenge lies in their ability to generalize beyond the specific environments and conditions they were trained on, which is presently difficult and expensive to evaluate in the real-world. To address this gap, we present REALM, a new simulation environment and benchmark designed to evaluate the generalization capabilities of VLA models, with a specific emphasis on establishing a strong correlation between simulated and real-world performance through high-fidelity visuals and aligned robot control. Our environment offers a suite of 15 perturbation factors, 7 manipulation skills, and more than 3,500 objects. Finally, we establish two task sets that form our benchmark and evaluate the \pi_{0}, \pi_{0}-FAST, and GR00T N1.5 VLA models, showing that generalization and robustness remain an open challenge. More broadly, we also show that simulation gives us a valuable proxy for the real-world and allows us to systematically probe for and quantify the weaknesses and failure modes of VLAs. Project page: this https URL 

---
# BabyFlow: 3D modeling of realistic and expressive infant faces 

**Authors**: Antonia Alomar, Mireia Masias, Marius George Linguraru, Federico M. Sukno, Gemma Piella  

**Link**: [PDF](https://arxiv.org/pdf/2512.19560)  

**Abstract**: Early detection of developmental disorders can be aided by analyzing infant craniofacial morphology, but modeling infant faces is challenging due to limited data and frequent spontaneous expressions. We introduce BabyFlow, a generative AI model that disentangles facial identity and expression, enabling independent control over both. Using normalizing flows, BabyFlow learns flexible, probabilistic representations that capture the complex, non-linear variability of expressive infant faces without restrictive linear assumptions. To address scarce and uncontrolled expressive data, we perform cross-age expression transfer, adapting expressions from adult 3D scans to enrich infant datasets with realistic and systematic expressive variants. As a result, BabyFlow improves 3D reconstruction accuracy, particularly in highly expressive regions such as the mouth, eyes, and nose, and supports synthesis and modification of infant expressions while preserving identity. Additionally, by integrating with diffusion models, BabyFlow generates high-fidelity 2D infant images with consistent 3D geometry, providing powerful tools for data augmentation and early facial analysis. 

---
# CARE What Fails: Contrastive Anchored-REflection for Verifiable Multimodal 

**Authors**: Yongxin Wang, Zhicheng Yang, Meng Cao, Mingfei Han, Haokun Lin, Yingying Zhu, Xiaojun Chang, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19554)  

**Abstract**: Group-relative reinforcement learning with verifiable rewards (RLVR) often wastes the most informative data it already has the failures. When all rollouts are wrong, gradients stall; when one happens to be correct, the update usually ignores why the others are close-but-wrong, and credit can be misassigned to spurious chains. We present CARE (Contrastive Anchored REflection), a failure-centric post-training framework for multimodal reasoning that turns errors into supervision. CARE combines: (i) an anchored-contrastive objective that forms a compact subgroup around the best rollout and a set of semantically proximate hard negatives, performs within-subgroup z-score normalization with negative-only scaling, and includes an all-negative rescue to prevent zero-signal batches; and (ii) Reflection-Guided Resampling (RGR), a one-shot structured self-repair that rewrites a representative failure and re-scores it with the same verifier, converting near-misses into usable positives without any test-time reflection. CARE improves accuracy and training smoothness while explicitly increasing the share of learning signal that comes from failures. On Qwen2.5-VL-7B, CARE lifts macro-averaged accuracy by 4.6 points over GRPO across six verifiable visual-reasoning benchmarks; with Qwen3-VL-8B it reaches competitive or state-of-the-art results on MathVista and MMMU-Pro under an identical evaluation protocol. 

---
# CASA: Cross-Attention via Self-Attention for Efficient Vision-Language Fusion 

**Authors**: Moritz Böhle, Amélie Royer, Juliette Marrie, Edouard Grave, Patrick Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2512.19535)  

**Abstract**: Vision-language models (VLMs) are commonly trained by inserting image tokens from a pretrained vision encoder into the textual stream of a language model. This allows text and image information to fully attend to one another within the model, but becomes extremely costly for high-resolution images, long conversations, or streaming videos, both in memory and compute. VLMs leveraging cross-attention are an efficient alternative to token insertion but exhibit a clear performance gap, in particular on tasks involving fine-grained visual details. We find that a key to improving such models is to also enable local text-to-text interaction in the dedicated cross-attention layers. Building on this, we propose CASA, Cross-Attention via Self-Attention, a simple and efficient paradigm which substantially reduces the gap with full token insertion on common image understanding benchmarks, while enjoying the same scalability as cross-attention models when applied to long-context multimodal tasks such as streaming video captioning. For samples and code, please see our project page at this https URL . 

---
# Learning Continuous Solvent Effects from Transient Flow Data: A Graph Neural Network Benchmark on Catechol Rearrangement 

**Authors**: Hongsheng Xing, Qiuxin Si  

**Link**: [PDF](https://arxiv.org/pdf/2512.19530)  

**Abstract**: Predicting reaction outcomes across continuous solvent composition ranges remains a critical challenge in organic synthesis and process chemistry. Traditional machine learning approaches often treat solvent identity as a discrete categorical variable, which prevents systematic interpolation and extrapolation across the solvent space. This work introduces the \textbf{Catechol Benchmark}, a high-throughput transient flow chemistry dataset comprising 1,227 experimental yield measurements for the rearrangement of allyl-substituted catechol in 24 pure solvents and their binary mixtures, parameterized by continuous volume fractions ($\% B$). We evaluate various architectures under rigorous leave-one-solvent-out and leave-one-mixture-out protocols to test generalization to unseen chemical environments.
Our results demonstrate that classical tabular methods (e.g., Gradient-Boosted Decision Trees) and large language model embeddings (e.g., Qwen-7B) struggle with quantitative precision, yielding Mean Squared Errors (MSE) of 0.099 and 0.129, respectively. In contrast, we propose a hybrid GNN-based architecture that integrates Graph Attention Networks (GATs) with Differential Reaction Fingerprints (DRFP) and learned mixture-aware solvent encodings. This approach achieves an \textbf{MSE of 0.0039} ($\pm$ 0.0003), representing a 60\% error reduction over competitive baselines and a $>25\times$ improvement over tabular ensembles. Ablation studies confirm that explicit molecular graph message-passing and continuous mixture encoding are essential for robust generalization. The complete dataset, evaluation protocols, and reference implementations are released to facilitate data-efficient reaction prediction and continuous solvent representation learning. 

---
# LacaDM: A Latent Causal Diffusion Model for Multiobjective Reinforcement Learning 

**Authors**: Xueming Yan, Bo Yin, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.19516)  

**Abstract**: Multiobjective reinforcement learning (MORL) poses significant challenges due to the inherent conflicts between objectives and the difficulty of adapting to dynamic environments. Traditional methods often struggle to generalize effectively, particularly in large and complex state-action spaces. To address these limitations, we introduce the Latent Causal Diffusion Model (LacaDM), a novel approach designed to enhance the adaptability of MORL in discrete and continuous environments. Unlike existing methods that primarily address conflicts between objectives, LacaDM learns latent temporal causal relationships between environmental states and policies, enabling efficient knowledge transfer across diverse MORL scenarios. By embedding these causal structures within a diffusion model-based framework, LacaDM achieves a balance between conflicting objectives while maintaining strong generalization capabilities in previously unseen environments. Empirical evaluations on various tasks from the MOGymnasium framework demonstrate that LacaDM consistently outperforms the state-of-art baselines in terms of hypervolume, sparsity, and expected utility maximization, showcasing its effectiveness in complex multiobjective tasks. 

---
# Anatomy-R1: Enhancing Anatomy Reasoning in Multimodal Large Language Models via Anatomical Similarity Curriculum and Group Diversity Augmentation 

**Authors**: Ziyang Song, Zelin Zang, Zuyao Chen, Xusheng Liang, Dong Yi, Jinlin Wu, Hongbin Liu, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.19512)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive progress in natural image reasoning, yet their potential in medical imaging remains underexplored, especially in clinical anatomical surgical images. Anatomy understanding tasks demand precise understanding and clinically coherent answers, which are difficult to achieve due to the complexity of medical data and the scarcity of high-quality expert annotations. These challenges limit the effectiveness of conventional Supervised Fine-Tuning (SFT) strategies. While recent work has demonstrated that Group Relative Policy Optimization (GRPO) can enhance reasoning in MLLMs without relying on large amounts of data, we find two weaknesses that hinder GRPO's reasoning performance in anatomy recognition: 1) knowledge cannot be effectively shared between different anatomical structures, resulting in uneven information gain and preventing the model from converging, and 2) the model quickly converges to a single reasoning path, suppressing the exploration of diverse strategies. To overcome these challenges, we propose two novel methods. First, we implement a progressive learning strategy called Anatomical Similarity Curriculum Learning by controlling question difficulty via the similarity of answer choices, enabling the model to master complex problems incrementally. Second, we utilize question augmentation referred to as Group Diversity Question Augmentation to expand the model's search space for difficult queries, mitigating the tendency to produce uniform responses. Comprehensive experiments on the SGG-VQA and OmniMedVQA benchmarks show our method achieves a significant improvement across the two benchmarks, demonstrating its effectiveness in enhancing the medical reasoning capabilities of MLLMs. The code can be found in this https URL 

---
# DK-STN: A Domain Knowledge Embedded Spatio-Temporal Network Model for MJO Forecast 

**Authors**: Hongliang Li, Nong Zhang, Zhewen Xu, Xiang Li, Changzheng Liu, Chongbo Zhao, Jie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19506)  

**Abstract**: Understanding and predicting the Madden-Julian Oscillation (MJO) is fundamental for precipitation forecasting and disaster prevention. To date, long-term and accurate MJO prediction has remained a challenge for researchers. Conventional MJO prediction methods using Numerical Weather Prediction (NWP) are resource-intensive, time-consuming, and highly unstable (most NWP methods are sensitive to seasons, with better MJO forecast results in winter). While existing Artificial Neural Network (ANN) methods save resources and speed forecasting, their accuracy never reaches the 28 days predicted by the state-of-the-art NWP method, i.e., the operational forecasts from ECMWF, since neural networks cannot handle climate data effectively. In this paper, we present a Domain Knowledge Embedded Spatio-Temporal Network (DK-STN), a stable neural network model for accurate and efficient MJO forecasting. It combines the benefits of NWP and ANN methods and successfully improves the forecast accuracy of ANN methods while maintaining a high level of efficiency and stability. We begin with a spatial-temporal network (STN) and embed domain knowledge in it using two key methods: (i) applying a domain knowledge enhancement method and (ii) integrating a domain knowledge processing method into network training. We evaluated DK-STN with the 5th generation of ECMWF reanalysis (ERA5) data and compared it with ECMWF. Given 7 days of climate data as input, DK-STN can generate reliable forecasts for the following 28 days in 1-2 seconds, with an error of only 2-3 days in different seasons. DK-STN significantly exceeds ECMWF in that its forecast accuracy is equivalent to ECMWF's, while its efficiency and stability are significantly superior. 

---
# Kolmogorov-Arnold Graph Neural Networks Applied to Inorganic Nanomaterials Dataset 

**Authors**: Nikita Volzhin, Soowhan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2512.19494)  

**Abstract**: The recent development of Kolmogorov-Arnold Networks (KANs) introduced new discoveries in the field of Graph Neural Networks (GNNs), expanding the existing set of models with KAN-based versions of GNNs, which often surpass the accuracy of MultiLayer Perceptron (MLP)-based GNNs. These models were widely tested on the graph datasets consisting of organic molecules; however, those studies disregarded the inorganic nanomaterials datasets. In this work, we close this gap by applying Kolmogorov-Arnold Graph Neural Networks (KAGNNs) to a recently published large inorganic nanomaterials dataset called CHILI. For this, we adapt and test KAGNNs appropriate for this dataset. Our experiments reveal that on the CHILI datasets, particularly on the CHILI-3K, KAGNNs substantially surpass conventional GNNs in classification, achieving state-of-the-art results. 

---
# A Dataset and Preliminary Study of Using GPT-5 for Code-change Impact Analysis 

**Authors**: Katharina Stengg, Christian Macho, Martin Pinzger  

**Link**: [PDF](https://arxiv.org/pdf/2512.19481)  

**Abstract**: Understanding source code changes and their impact on other code entities is a crucial skill in software development. However, the analysis of code changes and their impact is often performed manually and therefore is time-consuming. Recent advancements in AI, and in particular large language models (LLMs) show promises to help developers in various code analysis tasks. However, the extent to which this potential can be utilized for understanding code changes and their impact is underexplored. To address this gap, we study the capabilities of GPT-5 and GPT-5-mini to predict the code entities impacted by given source code changes. We construct a dataset containing information about seed-changes, change pairs, and change types for each commit. Existing datasets lack crucial information about seed changes and impacted code entities. Our experiments evaluate the LLMs in two configurations: (1) seed-change information and the parent commit tree and (2) seed-change information, the parent commit tree, and the diff hunk of each seed change. We found that both LLMs perform poorly in the two experiments, whereas GPT-5 outperforms GPT-5-mini. Furthermore, the provision of the diff hunks helps both models to slightly improve their performance. 

---
# Multi-Layer Confidence Scoring for Detection of Out-of-Distribution Samples, Adversarial Attacks, and In-Distribution Misclassifications 

**Authors**: Lorenzo Capelli, Leandro de Souza Rosa, Gianluca Setti, Mauro Mangia, Riccardo Rovatti  

**Link**: [PDF](https://arxiv.org/pdf/2512.19472)  

**Abstract**: The recent explosive growth in Deep Neural Networks applications raises concerns about the black-box usage of such models, with limited trasparency and trustworthiness in high-stakes domains, which have been crystallized as regulatory requirements such as the European Union Artificial Intelligence Act. While models with embedded confidence metrics have been proposed, such approaches cannot be applied to already existing models without retraining, limiting their broad application. On the other hand, post-hoc methods, which evaluate pre-trained models, focus on solving problems related to improving the confidence in the model's predictions, and detecting Out-Of-Distribution or Adversarial Attacks samples as independent applications. To tackle the limited applicability of already existing methods, we introduce Multi-Layer Analysis for Confidence Scoring (MACS), a unified post-hoc framework that analyzes intermediate activations to produce classification-maps. From the classification-maps, we derive a score applicable for confidence estimation, detecting distributional shifts and adversarial attacks, unifying the three problems in a common framework, and achieving performances that surpass the state-of-the-art approaches in our experiments with the VGG16 and ViTb16 models with a fraction of their computational overhead. 

---
# Activations as Features: Probing LLMs for Generalizable Essay Scoring Representations 

**Authors**: Jinwei Chi, Ke Wang, Yu Chen, Xuanye Lin, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19456)  

**Abstract**: Automated essay scoring (AES) is a challenging task in cross-prompt settings due to the diversity of scoring criteria. While previous studies have focused on the output of large language models (LLMs) to improve scoring accuracy, we believe activations from intermediate layers may also provide valuable information. To explore this possibility, we evaluated the discriminative power of LLMs' activations in cross-prompt essay scoring task. Specifically, we used activations to fit probes and further analyzed the effects of different models and input content of LLMs on this discriminative power. By computing the directions of essays across various trait dimensions under different prompts, we analyzed the variation in evaluation perspectives of large language models concerning essay types and traits. Results show that the activations possess strong discriminative power in evaluating essay quality and that LLMs can adapt their evaluation perspectives to different traits and essay types, effectively handling the diversity of scoring criteria in cross-prompt settings. 

---
# MT-Mark: Rethinking Image Watermarking via Mutual-Teacher Collaboration with Adaptive Feature Modulation 

**Authors**: Fei Ge, Ying Huang, Jie Liu, Guixuan Zhang, Zhi Zeng, Shuwu Zhang, Hu Guan  

**Link**: [PDF](https://arxiv.org/pdf/2512.19438)  

**Abstract**: Existing deep image watermarking methods follow a fixed embedding-distortion-extraction pipeline, where the embedder and extractor are weakly coupled through a final loss and optimized in isolation. This design lacks explicit collaboration, leaving no structured mechanism for the embedder to incorporate decoding-aware cues or for the extractor to guide embedding during training. To address this architectural limitation, we rethink deep image watermarking by reformulating embedding and extraction as explicitly collaborative components. To realize this reformulation, we introduce a Collaborative Interaction Mechanism (CIM) that establishes direct, bidirectional communication between the embedder and extractor, enabling a mutual-teacher training paradigm and coordinated optimization. Built upon this explicitly collaborative architecture, we further propose an Adaptive Feature Modulation Module (AFMM) to support effective interaction. AFMM enables content-aware feature regulation by decoupling modulation structure and strength, guiding watermark embedding toward stable image features while suppressing host interference during extraction. Under CIM, the AFMMs on both sides form a closed-loop collaboration that aligns embedding behavior with extraction objectives. This architecture-level redesign changes how robustness is learned in watermarking systems. Rather than relying on exhaustive distortion simulation, robustness emerges from coordinated representation learning between embedding and extraction. Experiments on real-world and AI-generated datasets demonstrate that the proposed method consistently outperforms state-of-the-art approaches in watermark extraction accuracy while maintaining high perceptual quality, showing strong robustness and generalization. 

---
# Attention Is Not What You Need 

**Authors**: Zhang Chong  

**Link**: [PDF](https://arxiv.org/pdf/2512.19428)  

**Abstract**: We revisit a basic question in sequence modeling: is explicit self-attention actually necessary for strong performance and reasoning? We argue that standard multi-head attention is best seen as a form of tensor lifting: hidden vectors are mapped into a high-dimensional space of pairwise interactions, and learning proceeds by constraining this lifted tensor through gradient descent. This mechanism is extremely expressive but mathematically opaque, because after many layers it becomes very hard to describe the model with a small family of explicit invariants.
To explore an alternative, we propose an attention-free architecture based on Grassmann flows. Instead of forming an L by L attention matrix, our Causal Grassmann layer (i) linearly reduces token states, (ii) encodes local token pairs as two-dimensional subspaces on a Grassmann manifold via Plucker coordinates, and (iii) fuses these geometric features back into the hidden states through gated mixing. Information therefore propagates by controlled deformations of low-rank subspaces over multi-scale local windows, so the core computation lives on a finite-dimensional manifold rather than in an unstructured tensor space.
On the Wikitext-2 language modeling benchmark, purely Grassmann-based models with 13 to 18 million parameters achieve validation perplexities within about 10 to 15 percent of size-matched Transformers. On the SNLI natural language inference task, a Grassmann-Plucker head on top of DistilBERT slightly outperforms a Transformer head, with best validation and test accuracies of 0.8550 and 0.8538 compared to 0.8545 and 0.8511. We analyze the complexity of Grassmann mixing, show linear scaling in sequence length for fixed rank, and argue that such manifold-based designs offer a more structured route toward geometric and invariant-based interpretations of neural reasoning. 

---
# Research Program: Theory of Learning in Dynamical Systems 

**Authors**: Elad Hazan, Shai Shalev Shwartz, Nathan Srebro  

**Link**: [PDF](https://arxiv.org/pdf/2512.19410)  

**Abstract**: Modern learning systems increasingly interact with data that evolve over time and depend on hidden internal state. We ask a basic question: when is such a dynamical system learnable from observations alone? This paper proposes a research program for understanding learnability in dynamical systems through the lens of next-token prediction. We argue that learnability in dynamical systems should be studied as a finite-sample question, and be based on the properties of the underlying dynamics rather than the statistical properties of the resulting sequence. To this end, we give a formulation of learnability for stochastic processes induced by dynamical systems, focusing on guarantees that hold uniformly at every time step after a finite burn-in period. This leads to a notion of dynamic learnability which captures how the structure of a system, such as stability, mixing, observability, and spectral properties, governs the number of observations required before reliable prediction becomes possible. We illustrate the framework in the case of linear dynamical systems, showing that accurate prediction can be achieved after finite observation without system identification, by leveraging improper methods based on spectral filtering. We survey the relationship between learning in dynamical systems and classical PAC, online, and universal prediction theories, and suggest directions for studying nonlinear and controlled systems. 

---
# DSTED: Decoupling Temporal Stabilization and Discriminative Enhancement for Surgical Workflow Recognition 

**Authors**: Yueyao Chen, Kai-Ni Wang, Dario Tayupo, Arnaud Huaulm'e, Krystel Nyangoh Timoh, Pierre Jannin, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2512.19387)  

**Abstract**: Purpose: Surgical workflow recognition enables context-aware assistance and skill assessment in computer-assisted interventions. Despite recent advances, current methods suffer from two critical challenges: prediction jitter across consecutive frames and poor discrimination of ambiguous phases. This paper aims to develop a stable framework by selectively propagating reliable historical information and explicitly modeling uncertainty for hard sample enhancement.
Methods: We propose a dual-pathway framework DSTED with Reliable Memory Propagation (RMP) and Uncertainty-Aware Prototype Retrieval (UPR). RMP maintains temporal coherence by filtering and fusing high-confidence historical features through multi-criteria reliability assessment. UPR constructs learnable class-specific prototypes from high-uncertainty samples and performs adaptive prototype matching to refine ambiguous frame representations. Finally, a confidence-driven gate dynamically balances both pathways based on prediction certainty.
Results: Our method achieves state-of-the-art performance on AutoLaparo-hysterectomy with 84.36% accuracy and 65.51% F1-score, surpassing the second-best method by 3.51% and 4.88% respectively. Ablations reveal complementary gains from RMP (2.19%) and UPR (1.93%), with synergistic effects when combined. Extensive analysis confirms substantial reduction in temporal jitter and marked improvement on challenging phase transitions.
Conclusion: Our dual-pathway design introduces a novel paradigm for stable workflow recognition, demonstrating that decoupling the modeling of temporal consistency and phase ambiguity yields superior performance and clinical applicability. 

---
# OmniMER: Indonesian Multimodal Emotion Recognition via Auxiliary-Enhanced LLM Adaptation 

**Authors**: Xueming Yan, Boyan Xu, Yaochu Jin, Lixian Xiao, Wenlong Ye, Runyang Cai, Zeqi Zheng, Jingfa Liu, Aimin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19379)  

**Abstract**: Indonesian, spoken by over 200 million people, remains underserved in multimodal emotion recognition research despite its dominant presence on Southeast Asian social media platforms. We introduce IndoMER, the first multimodal emotion recognition benchmark for Indonesian, comprising 1,944 video segments from 203 speakers with temporally aligned text, audio, and visual annotations across seven emotion categories. The dataset exhibits realistic challenges including cross-modal inconsistency and long-tailed class distributions shaped by Indonesian cultural communication norms. To address these challenges, we propose OmniMER, a multimodal adaptation framework built upon Qwen2.5-Omni that enhances emotion recognition through three auxiliary modality-specific perception tasks: emotion keyword extraction for text, facial expression analysis for video, and prosody analysis for audio. These auxiliary tasks help the model identify emotion-relevant cues in each modality before fusion, reducing reliance on spurious correlations in low-resource settings. Experiments on IndoMER show that OmniMER achieves 0.582 Macro-F1 on sentiment classification and 0.454 on emotion recognition, outperforming the base model by 7.6 and 22.1 absolute points respectively. Cross-lingual evaluation on the Chinese CH-SIMS dataset further demonstrates the generalizability of the proposed framework. The dataset and code are publicly available. this https URL 

---
# Sprecher Networks: A Parameter-Efficient Kolmogorov-Arnold Architecture 

**Authors**: Christian Hägg, Kathlén Kohn, Giovanni Luca Marchetti, Boris Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2512.19367)  

**Abstract**: We present Sprecher Networks (SNs), a family of trainable neural architectures inspired by the classical Kolmogorov-Arnold-Sprecher (KAS) construction for approximating multivariate continuous functions. Distinct from Multi-Layer Perceptrons (MLPs) with fixed node activations and Kolmogorov-Arnold Networks (KANs) featuring learnable edge activations, SNs utilize shared, learnable splines (monotonic and general) within structured blocks incorporating explicit shift parameters and mixing weights. Our approach directly realizes Sprecher's specific 1965 sum of shifted splines formula in its single-layer variant and extends it to deeper, multi-layer compositions. We further enhance the architecture with optional lateral mixing connections that enable intra-block communication between output dimensions, providing a parameter-efficient alternative to full attention mechanisms. Beyond parameter efficiency with $O(LN + LG)$ scaling (where $G$ is the knot count of the shared splines) versus MLPs' $O(LN^2)$, SNs admit a sequential evaluation strategy that reduces peak forward-intermediate memory from $O(N^2)$ to $O(N)$ (treating batch size as constant), making much wider architectures feasible under memory constraints. We demonstrate empirically that composing these blocks into deep networks leads to highly parameter and memory-efficient models, discuss theoretical motivations, and compare SNs with related architectures (MLPs, KANs, and networks with learnable node activations). 

---
# Alternative positional encoding functions for neural transformers 

**Authors**: Ezequiel Lopez-Rubio, Macoris Decena-Gimenez, Rafael Marcos Luque-Baena  

**Link**: [PDF](https://arxiv.org/pdf/2512.19323)  

**Abstract**: A key module in neural transformer-based deep architectures is positional encoding. This module enables a suitable way to encode positional information as input for transformer neural layers. This success has been rooted in the use of sinusoidal functions of various frequencies, in order to capture recurrent patterns of differing typical periods. In this work, an alternative set of periodic functions is proposed for positional encoding. These functions preserve some key properties of sinusoidal ones, while they depart from them in fundamental ways. Some tentative experiments are reported, where the original sinusoidal version is substantially outperformed. This strongly suggests that the alternative functions may have a wider use in other transformer architectures. 

---
# MAGIC: Achieving Superior Model Merging via Magnitude Calibration 

**Authors**: Yayuan Li, Jian Zhang, Jintao Guo, Zihan Cheng, Lei Qi, Yinghuan Shi, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.19320)  

**Abstract**: The proliferation of pre-trained models has given rise to a wide array of specialised, fine-tuned models. Model merging aims to merge the distinct capabilities of these specialised models into a unified model, requiring minimal or even no additional training. A core objective of model merging is to ensure the merged model retains the behavioural characteristics of the specialised models, typically achieved through feature alignment. We identify that features consist of two critical components: direction and magnitude. Prior research has predominantly focused on directional alignment, while the influence of magnitude remains largely neglected, despite its pronounced vulnerability to perturbations introduced by common merging operations (e.g., parameter fusion and sparsification). Such perturbations to magnitude inevitably lead to feature deviations in the merged model from the specialised models, resulting in subsequent performance degradation. To address this, we propose MAGnItude Calibration (MAGIC), a plug-and-play framework that rectifies layer-wise magnitudes in feature and weight spaces, with three variants. Specifically, our Feature Space Calibration (FSC) realigns the merged model's features using a small set of unlabelled data, while Weight Space Calibration (WSC) extends this calibration to the weight space without requiring additional data. Combining these yields Dual Space Calibration (DSC). Comprehensive experiments demonstrate that MAGIC consistently boosts performance across diverse Computer Vision tasks (+4.3% on eight datasets) and NLP tasks (+8.0% on Llama) without additional training. Our code is available at: this https URL 

---
# MixFlow Training: Alleviating Exposure Bias with Slowed Interpolation Mixture 

**Authors**: Hui Li, Jiayue Lyu, Fu-Yun Wang, Kaihui Cheng, Siyu Zhu, Jingdong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19311)  

**Abstract**: This paper studies the training-testing discrepancy (a.k.a. exposure bias) problem for improving the diffusion models. During training, the input of a prediction network at one training timestep is the corresponding ground-truth noisy data that is an interpolation of the noise and the data, and during testing, the input is the generated noisy data. We present a novel training approach, named MixFlow, for improving the performance. Our approach is motivated by the Slow Flow phenomenon: the ground-truth interpolation that is the nearest to the generated noisy data at a given sampling timestep is observed to correspond to a higher-noise timestep (termed slowed timestep), i.e., the corresponding ground-truth timestep is slower than the sampling timestep. MixFlow leverages the interpolations at the slowed timesteps, named slowed interpolation mixture, for post-training the prediction network for each training timestep. Experiments over class-conditional image generation (including SiT, REPA, and RAE) and text-to-image generation validate the effectiveness of our approach. Our approach MixFlow over the RAE models achieve strong generation results on ImageNet: 1.43 FID (without guidance) and 1.10 (with guidance) at 256 x 256, and 1.55 FID (without guidance) and 1.10 (with guidance) at 512 x 512. 

---
# Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models 

**Authors**: Linzhi Chen, Yang Sun, Hongru Wei, Yuqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19297)  

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as an efficient method for fine-tuning large language models (LLMs) and is widely adopted within the open-source community. However, the decentralized dissemination of LoRA adapters through platforms such as Hugging Face introduces novel security vulnerabilities: malicious adapters can be easily distributed and evade conventional oversight mechanisms. Despite these risks, backdoor attacks targeting LoRA-based fine-tuning remain relatively underexplored. Existing backdoor attack strategies are ill-suited to this setting, as they often rely on inaccessible training data, fail to account for the structural properties unique to LoRA, or suffer from high false trigger rates (FTR), thereby compromising their stealth. To address these challenges, we propose Causal-Guided Detoxify Backdoor Attack (CBA), a novel backdoor attack framework specifically designed for open-weight LoRA models. CBA operates without access to original training data and achieves high stealth through two key innovations: (1) a coverage-guided data generation pipeline that synthesizes task-aligned inputs via behavioral exploration, and (2) a causal-guided detoxification strategy that merges poisoned and clean adapters by preserving task-critical neurons. Unlike prior approaches, CBA enables post-training control over attack intensity through causal influence-based weight allocation, eliminating the need for repeated retraining. Evaluated across six LoRA models, CBA achieves high attack success rates while reducing FTR by 50-70\% compared to baseline methods. Furthermore, it demonstrates enhanced resistance to state-of-the-art backdoor defenses, highlighting its stealth and robustness. 

---
# Digital Twin-Driven Zero-Shot Fault Diagnosis of Axial Piston Pumps Using Fluid-Borne Noise Signals 

**Authors**: Chang Dong, Jianfeng Tao, Chengliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19280)  

**Abstract**: Axial piston pumps are crucial components in fluid power systems, where reliable fault diagnosis is essential for ensuring operational safety and efficiency. Traditional data-driven methods require extensive labeled fault data, which is often impractical to obtain, while model-based approaches suffer from parameter uncertainties. This paper proposes a digital twin (DT)-driven zero-shot fault diagnosis framework utilizing fluid-borne noise (FBN) signals. The framework calibrates a high-fidelity DT model using only healthy-state data, generates synthetic fault signals for training deep learning classifiers, and employs a physics-informed neural network (PINN) as a virtual sensor for flow ripple estimation. Gradient-weighted class activation mapping (Grad-CAM) is integrated to visualize the decision-making process of neural networks, revealing that large kernels matching the subsequence length in time-domain inputs and small kernels in time-frequency domain inputs enable higher diagnostic accuracy by focusing on physically meaningful features. Experimental validations demonstrate that training on signals from the calibrated DT model yields diagnostic accuracies exceeding 95\% on real-world benchmarks, while uncalibrated models result in significantly lower performance, highlighting the framework's effectiveness in data-scarce scenarios. 

---
# Is Visual Realism Enough? Evaluating Gait Biometric Fidelity in Generative AI Human Animation 

**Authors**: Ivan DeAndres-Tame, Chengwei Ye, Ruben Tolosana, Ruben Vera-Rodriguez, Shiqi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19275)  

**Abstract**: Generative AI (GenAI) models have revolutionized animation, enabling the synthesis of humans and motion patterns with remarkable visual fidelity. However, generating truly realistic human animation remains a formidable challenge, where even minor inconsistencies can make a subject appear unnatural. This limitation is particularly critical when AI-generated videos are evaluated for behavioral biometrics, where subtle motion cues that define identity are easily lost or distorted. The present study investigates whether state-of-the-art GenAI human animation models can preserve the subtle spatio-temporal details needed for person identification through gait biometrics. Specifically, we evaluate four different GenAI models across two primary evaluation tasks to assess their ability to i) restore gait patterns from reference videos under varying conditions of complexity, and ii) transfer these gait patterns to different visual identities. Our results show that while visual quality is mostly high, biometric fidelity remains low in tasks focusing on identification, suggesting that current GenAI models struggle to disentangle identity from motion. Furthermore, through an identity transfer task, we expose a fundamental flaw in appearance-based gait recognition: when texture is disentangled from motion, identification collapses, proving current GenAI models rely on visual attributes rather than temporal dynamics. 

---
# Machine Unlearning in the Era of Quantum Machine Learning: An Empirical Study 

**Authors**: Carla Crivoi, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19253)  

**Abstract**: We present the first comprehensive empirical study of machine unlearning (MU) in hybrid quantum-classical neural networks. While MU has been extensively explored in classical deep learning, its behavior within variational quantum circuits (VQCs) and quantum-augmented architectures remains largely unexplored. First, we adapt a broad suite of unlearning methods to quantum settings, including gradient-based, distillation-based, regularization-based and certified techniques. Second, we introduce two new unlearning strategies tailored to hybrid models. Experiments across Iris, MNIST, and Fashion-MNIST, under both subset removal and full-class deletion, reveal that quantum models can support effective unlearning, but outcomes depend strongly on circuit depth, entanglement structure, and task complexity. Shallow VQCs display high intrinsic stability with minimal memorization, whereas deeper hybrid models exhibit stronger trade-offs between utility, forgetting strength, and alignment with retrain oracle. We find that certain methods, e.g. EU-k, LCA, and Certified Unlearning, consistently provide the best balance across metrics. These findings establish baseline empirical insights into quantum machine unlearning and highlight the need for quantum-aware algorithms and theoretical guarantees, as quantum machine learning systems continue to expand in scale and capability. We publicly release our code at: this https URL. 

---
# Auto-Prompting with Retrieval Guidance for Frame Detection in Logistics 

**Authors**: Do Minh Duc, Quan Xuan Truong, Nguyen Tat Dat, Nguyen Van Vinh  

**Link**: [PDF](https://arxiv.org/pdf/2512.19247)  

**Abstract**: Prompt engineering plays a critical role in adapting large language models (LLMs) to complex reasoning and labeling tasks without the need for extensive fine-tuning. In this paper, we propose a novel prompt optimization pipeline for frame detection in logistics texts, combining retrieval-augmented generation (RAG), few-shot prompting, chain-of-thought (CoT) reasoning, and automatic CoT synthesis (Auto-CoT) to generate highly effective task-specific prompts. Central to our approach is an LLM-based prompt optimizer agent that iteratively refines the prompts using retrieved examples, performance feedback, and internal self-evaluation. Our framework is evaluated on a real-world logistics text annotation task, where reasoning accuracy and labeling efficiency are critical. Experimental results show that the optimized prompts - particularly those enhanced via Auto-CoT and RAG - improve real-world inference accuracy by up to 15% compared to baseline zero-shot or static prompts. The system demonstrates consistent improvements across multiple LLMs, including GPT-4o, Qwen 2.5 (72B), and LLaMA 3.1 (70B), validating its generalizability and practical value. These findings suggest that structured prompt optimization is a viable alternative to full fine-tuning, offering scalable solutions for deploying LLMs in domain-specific NLP applications such as logistics. 

---
# ChemATP: A Training-Free Chemical Reasoning Framework for Large Language Models 

**Authors**: Mingxu Zhang, Dazhong Shen, Qi Zhang, Ying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.19240)  

**Abstract**: Large Language Models (LLMs) exhibit strong general reasoning but struggle in molecular science due to the lack of explicit chemical priors in standard string representations. Current solutions face a fundamental dilemma. Training-based methods inject priors into parameters, but this static coupling hinders rapid knowledge updates and often compromises the model's general reasoning capabilities. Conversely, existing training-free methods avoid these issues but rely on surface-level prompting, failing to provide the fine-grained atom-level priors essential for precise chemical reasoning. To address this issue, we introduce ChemATP, a framework that decouples chemical knowledge from the reasoning engine. By constructing the first atom-level textual knowledge base, ChemATP enables frozen LLMs to explicitly retrieve and reason over this information dynamically. This architecture ensures interpretability and adaptability while preserving the LLM's intrinsic general intelligence. Experiments show that ChemATP significantly outperforms training-free baselines and rivals state-of-the-art training-based models, demonstrating that explicit prior injection is a competitive alternative to implicit parameter updates. 

---
# Identifying Features Associated with Bias Against 93 Stigmatized Groups in Language Models and Guardrail Model Safety Mitigation 

**Authors**: Anna-Maria Gueorguieva, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2512.19238)  

**Abstract**: Large language models (LLMs) have been shown to exhibit social bias, however, bias towards non-protected stigmatized identities remain understudied. Furthermore, what social features of stigmas are associated with bias in LLM outputs is unknown. From psychology literature, it has been shown that stigmas contain six shared social features: aesthetics, concealability, course, disruptiveness, origin, and peril. In this study, we investigate if human and LLM ratings of the features of stigmas, along with prompt style and type of stigma, have effect on bias towards stigmatized groups in LLM outputs. We measure bias against 93 stigmatized groups across three widely used LLMs (Granite 3.0-8B, Llama-3.1-8B, Mistral-7B) using SocialStigmaQA, a benchmark that includes 37 social scenarios about stigmatized identities; for example deciding wether to recommend them for an internship. We find that stigmas rated by humans to be highly perilous (e.g., being a gang member or having HIV) have the most biased outputs from SocialStigmaQA prompts (60% of outputs from all models) while sociodemographic stigmas (e.g. Asian-American or old age) have the least amount of biased outputs (11%). We test if the amount of biased outputs could be decreased by using guardrail models, models meant to identify harmful input, using each LLM's respective guardrail model (Granite Guardian 3.0, Llama Guard 3.0, Mistral Moderation API). We find that bias decreases significantly by 10.4%, 1.4%, and 7.8%, respectively. However, we show that features with significant effect on bias remain unchanged post-mitigation and that guardrail models often fail to recognize the intent of bias in prompts. This work has implications for using LLMs in scenarios involving stigmatized groups and we suggest future work towards improving guardrail models for bias mitigation. 

---
# From Pixels to Predicates Structuring urban perception with scene graphs 

**Authors**: Yunlong Liu, Shuyang Li, Pengyuan Liu, Yu Zhang, Rudi Stouffs  

**Link**: [PDF](https://arxiv.org/pdf/2512.19221)  

**Abstract**: Perception research is increasingly modelled using streetscapes, yet many approaches still rely on pixel features or object co-occurrence statistics, overlooking the explicit relations that shape human perception. This study proposes a three stage pipeline that transforms street view imagery (SVI) into structured representations for predicting six perceptual indicators. In the first stage, each image is parsed using an open-set Panoptic Scene Graph model (OpenPSG) to extract object predicate object triplets. In the second stage, compact scene-level embeddings are learned through a heterogeneous graph autoencoder (GraphMAE). In the third stage, a neural network predicts perception scores from these embeddings. We evaluate the proposed approach against image-only baselines in terms of accuracy, precision, and cross-city generalization. Results indicate that (i) our approach improves perception prediction accuracy by an average of 26% over baseline models, and (ii) maintains strong generalization performance in cross-city prediction tasks. Additionally, the structured representation clarifies which relational patterns contribute to lower perception scores in urban scenes, such as graffiti on wall and car parked on sidewalk. Overall, this study demonstrates that graph-based structure provides expressive, generalizable, and interpretable signals for modelling urban perception, advancing human-centric and context-aware urban analytics. 

---
# Towards Minimal Fine-Tuning of VLMs 

**Authors**: Tiange Luo, Lajanugen Logeswaran, Jaekyeom Kim, Justin Johnson, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.19219)  

**Abstract**: We introduce Image-LoRA, a lightweight parameter efficient fine-tuning (PEFT) recipe for transformer-based vision-language models (VLMs). Image-LoRA applies low-rank adaptation only to the value path of attention layers within the visual-token span, reducing adapter-only training FLOPs roughly in proportion to the visual-token fraction. We further adapt only a subset of attention heads, selected using head influence scores estimated with a rank-1 Image-LoRA, and stabilize per-layer updates via selection-size normalization. Across screen-centric grounding and referring benchmarks spanning text-heavy to image-heavy regimes, Image-LoRA matches or closely approaches standard LoRA accuracy while using fewer trainable parameters and lower adapter-only training FLOPs. The method also preserves the pure-text reasoning performance of VLMs before and after fine-tuning, as further shown on GSM8K. 

---
# MixKVQ: Query-Aware Mixed-Precision KV Cache Quantization for Long-Context Reasoning 

**Authors**: Tao Zhang, Ziqian Zeng, Hao Peng, Huiping Zhuang, Cen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19206)  

**Abstract**: Long Chain-of-Thought (CoT) reasoning has significantly advanced the capabilities of Large Language Models (LLMs), but this progress is accompanied by substantial memory and latency overhead from the extensive Key-Value (KV) cache. Although KV cache quantization is a promising compression technique, existing low-bit quantization methods often exhibit severe performance degradation on complex reasoning tasks. Fixed-precision quantization struggles to handle outlier channels in the key cache, while current mixed-precision strategies fail to accurately identify components requiring high-precision representation. We find that an effective low-bit KV cache quantization strategy must consider two factors: a key channel's intrinsic quantization difficulty and its relevance to the query. Based on this insight, we propose MixKVQ, a novel plug-and-play method that introduces a lightweight, query-aware algorithm to identify and preserve critical key channels that need higher precision, while applying per-token quantization for value cache. Experiments on complex reasoning datasets demonstrate that our approach significantly outperforms existing low-bit methods, achieving performance comparable to a full-precision baseline at a substantially reduced memory footprint. 

---
# On the Koopman-Based Generalization Bounds for Multi-Task Deep Learning 

**Authors**: Mahdi Mohammadigohari, Giuseppe Di Fatta, Giuseppe Nicosia, Panos M. Pardalos  

**Link**: [PDF](https://arxiv.org/pdf/2512.19199)  

**Abstract**: The paper establishes generalization bounds for multitask deep neural networks using operator-theoretic techniques. The authors propose a tighter bound than those derived from conventional norm based methods by leveraging small condition numbers in the weight matrices and introducing a tailored Sobolev space as an expanded hypothesis space. This enhanced bound remains valid even in single output settings, outperforming existing Koopman based bounds. The resulting framework maintains key advantages such as flexibility and independence from network width, offering a more precise theoretical understanding of multitask deep learning in the context of kernel methods. 

---
# Operator-Based Generalization Bound for Deep Learning: Insights on Multi-Task Learning 

**Authors**: Mahdi Mohammadigohari, Giuseppe Di Fatta, Giuseppe Nicosia, Panos M. Pardalos  

**Link**: [PDF](https://arxiv.org/pdf/2512.19184)  

**Abstract**: This paper presents novel generalization bounds for vector-valued neural networks and deep kernel methods, focusing on multi-task learning through an operator-theoretic framework. Our key development lies in strategically combining a Koopman based approach with existing techniques, achieving tighter generalization guarantees compared to traditional norm-based bounds. To mitigate computational challenges associated with Koopman-based methods, we introduce sketching techniques applicable to vector valued neural networks. These techniques yield excess risk bounds under generic Lipschitz losses, providing performance guarantees for applications including robust and multiple quantile regression. Furthermore, we propose a novel deep learning framework, deep vector-valued reproducing kernel Hilbert spaces (vvRKHS), leveraging Perron Frobenius (PF) operators to enhance deep kernel methods. We derive a new Rademacher generalization bound for this framework, explicitly addressing underfitting and overfitting through kernel refinement strategies. This work offers novel insights into the generalization properties of multitask learning with deep learning architectures, an area that has been relatively unexplored until recent developments. 

---
# Practical Quantum-Classical Feature Fusion for complex data Classification 

**Authors**: Azadeh Alavi, Fatemeh Kouchmeshki, Abdolrahman Alavi  

**Link**: [PDF](https://arxiv.org/pdf/2512.19180)  

**Abstract**: Hybrid quantum and classical learning aims to couple quantum feature maps with the robustness of classical neural networks, yet most architectures treat the quantum circuit as an isolated feature extractor and merge its measurements with classical representations by direct concatenation. This neglects that the quantum and classical branches constitute distinct computational modalities and limits reliable performance on complex, high dimensional tabular and semi structured data, including remote sensing, environmental monitoring, and medical diagnostics. We present a multimodal formulation of hybrid learning and propose a cross attention mid fusion architecture in which a classical representation queries quantum derived feature tokens through an attention block with residual connectivity. The quantum branch is kept within practical NISQ budgets and uses up to nine qubits. We evaluate on Wine, Breast Cancer, Forest CoverType, FashionMNIST, and SteelPlatesFaults, comparing a quantum only model, a classical baseline, residual hybrid models, and the proposed mid fusion model under a consistent protocol. Pure quantum and standard hybrid designs underperform due to measurement induced information loss, while cross attention mid fusion is consistently competitive and improves performance on the more complex datasets in most cases. These findings suggest that quantum derived information becomes most valuable when integrated through principled multimodal fusion rather than used in isolation or loosely appended to classical features. 

---
# Vision-Language-Policy Model for Dynamic Robot Task Planning 

**Authors**: Jin Wang, Kim Tien Ly, Jacques Cloete, Nikos Tsagarakis, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2512.19178)  

**Abstract**: Bridging the gap between natural language commands and autonomous execution in unstructured environments remains an open challenge for robotics. This requires robots to perceive and reason over the current task scene through multiple modalities, and to plan their behaviors to achieve their intended goals. Traditional robotic task-planning approaches often struggle to bridge low-level execution with high-level task reasoning, and cannot dynamically update task strategies when instructions change during execution, which ultimately limits their versatility and adaptability to new tasks. In this work, we propose a novel language model-based framework for dynamic robot task planning. Our Vision-Language-Policy (VLP) model, based on a vision-language model fine-tuned on real-world data, can interpret semantic instructions and integrate reasoning over the current task scene to generate behavior policies that control the robot to accomplish the task. Moreover, it can dynamically adjust the task strategy in response to changes in the task, enabling flexible adaptation to evolving task requirements. Experiments conducted with different robots and a variety of real-world tasks show that the trained model can efficiently adapt to novel scenarios and dynamically update its policy, demonstrating strong planning autonomy and cross-embodiment generalization. Videos: this https URL 

---
# Beyond Sliding Windows: Learning to Manage Memory in Non-Markovian Environments 

**Authors**: Geraud Nangue Tasse, Matthew Riemer, Benjamin Rosman, Tim Klinger  

**Link**: [PDF](https://arxiv.org/pdf/2512.19154)  

**Abstract**: Recent success in developing increasingly general purpose agents based on sequence models has led to increased focus on the problem of deploying computationally limited agents within the vastly more complex real-world. A key challenge experienced in these more realistic domains is highly non-Markovian dependencies with respect to the agent's observations, which are less common in small controlled domains. The predominant approach for dealing with this in the literature is to stack together a window of the most recent observations (Frame Stacking), but this window size must grow with the degree of non-Markovian dependencies, which results in prohibitive computational and memory requirements for both action inference and learning. In this paper, we are motivated by the insight that in many environments that are highly non-Markovian with respect to time, the environment only causally depends on a relatively small number of observations over that time-scale. A natural direction would then be to consider meta-algorithms that maintain relatively small adaptive stacks of memories such that it is possible to express highly non-Markovian dependencies with respect to time while considering fewer observations at each step and thus experience substantial savings in both compute and memory requirements. Hence, we propose a meta-algorithm (Adaptive Stacking) for achieving exactly that with convergence guarantees and quantify the reduced computation and memory constraints for MLP, LSTM, and Transformer-based agents. Our experiments utilize popular memory tasks, which give us control over the degree of non-Markovian dependencies. This allows us to demonstrate that an appropriate meta-algorithm can learn the removal of memories not predictive of future rewards without excessive removal of important experiences. Code: this https URL 

---
# HyperLoad: A Cross-Modality Enhanced Large Language Model-Based Framework for Green Data Center Cooling Load Prediction 

**Authors**: Haoyu Jiang, Boan Qu, Junjie Zhu, Fanjie Zeng, Xiaojie Lin, Wei Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2512.19114)  

**Abstract**: The rapid growth of artificial intelligence is exponentially escalating computational demand, inflating data center energy use and carbon emissions, and spurring rapid deployment of green data centers to relieve resource and environmental stress. Achieving sub-minute orchestration of renewables, storage, and loads, while minimizing PUE and lifecycle carbon intensity, hinges on accurate load forecasting. However, existing methods struggle to address small-sample scenarios caused by cold start, load distortion, multi-source data fragmentation, and distribution shifts in green data centers. We introduce HyperLoad, a cross-modality framework that exploits pre-trained large language models (LLMs) to overcome data scarcity. In the Cross-Modality Knowledge Alignment phase, textual priors and time-series data are mapped to a common latent space, maximizing the utility of prior knowledge. In the Multi-Scale Feature Modeling phase, domain-aligned priors are injected through adaptive prefix-tuning, enabling rapid scenario adaptation, while an Enhanced Global Interaction Attention mechanism captures cross-device temporal dependencies. The public DCData dataset is released for benchmarking. Under both data sufficient and data scarce settings, HyperLoad consistently surpasses state-of-the-art (SOTA) baselines, demonstrating its practicality for sustainable green data center management. 

---
# DIVER-1 : Deep Integration of Vast Electrophysiological Recordings at Scale 

**Authors**: Danny Dongyeop Han, Yonghyeon Gwon, Ahhyun Lucy Lee, Taeyang Lee, Seong Jin Lee, Jubin Choi, Sebin Lee, Jihyun Bang, Seungju Lee, David Keetae Park, Shinjae Yoo, Chun Kee Chung, Jiook Cha  

**Link**: [PDF](https://arxiv.org/pdf/2512.19097)  

**Abstract**: Electrophysiology signals such as EEG and iEEG are central to neuroscience, brain-computer interfaces, and clinical applications, yet existing foundation models remain limited in scale despite clear evidence that scaling improves performance. We introduce DIVER-1, a family of EEG and iEEG foundation models trained on the largest and most diverse corpus to date-5.3k hours of iEEG and 54k hours of EEG (1.6M channel-hours from over 17.7k subjects)-and scaled up to 1.82B parameters. We present the first systematic scaling law analysis for this domain, showing that they follow data-constrained scaling laws: for a given amount of data and compute, smaller models trained for extended epochs consistently outperform larger models trained briefly. This behavior contrasts with prior electrophysiology foundation models that emphasized model size over training duration. To achieve strong performance, we also design architectural innovations including any-variate attention, sliding temporal conditional positional encoding, and multi-domain reconstruction. DIVER-1 iEEG and EEG models each achieve state-of-the-art performance on their respective benchmarks, establishing a concrete guidelines for efficient scaling and resource allocation in electrophysiology foundation model development. 

---
# Fraud Detection Through Large-Scale Graph Clustering with Heterogeneous Link Transformation 

**Authors**: Chi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19061)  

**Abstract**: Collaborative fraud, where multiple fraudulent accounts coordinate to exploit online payment systems, poses significant challenges due to the formation of complex network structures. Traditional detection methods that rely solely on high-confidence identity links suffer from limited coverage, while approaches using all available linkages often result in fragmented graphs with reduced clustering effectiveness. In this paper, we propose a novel graph-based fraud detection framework that addresses the challenge of large-scale heterogeneous graph clustering through a principled link transformation approach. Our method distinguishes between \emph{hard links} (high-confidence identity relationships such as phone numbers, credit cards, and national IDs) and \emph{soft links} (behavioral associations including device fingerprints, cookies, and IP addresses). We introduce a graph transformation technique that first identifies connected components via hard links, merges them into super-nodes, and then reconstructs a weighted soft-link graph amenable to efficient embedding and clustering. The transformed graph is processed using LINE (Large-scale Information Network Embedding) for representation learning, followed by HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) for density-based cluster discovery. Experiments on a real-world payment platform dataset demonstrate that our approach achieves significant graph size reduction (from 25 million to 7.7 million nodes), doubles the detection coverage compared to hard-link-only baselines, and maintains high precision across identified fraud clusters. Our framework provides a scalable and practical solution for industrial-scale fraud detection systems. 

---
# Finer-Personalization Rank: Fine-Grained Retrieval Examines Identity Preservation for Personalized Generation 

**Authors**: Connor Kilrain, David Carlyn, Julia Chae, Sara Beery, Wei-Lun Chao, Jianyang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19026)  

**Abstract**: The rise of personalized generative models raises a central question: how should we evaluate identity preservation? Given a reference image (e.g., one's pet), we expect the generated image to retain precise details attached to the subject's identity. However, current generative evaluation metrics emphasize the overall semantic similarity between the reference and the output, and overlook these fine-grained discriminative details. We introduce Finer-Personalization Rank, an evaluation protocol tailored to identity preservation. Instead of pairwise similarity, Finer-Personalization Rank adopts a ranking view: it treats each generated image as a query against an identity-labeled gallery consisting of visually similar real images. Retrieval metrics (e.g., mean average precision) measure performance, where higher scores indicate that identity-specific details (e.g., a distinctive head spot) are preserved. We assess identity at multiple granularities -- from fine-grained categories (e.g., bird species, car models) to individual instances (e.g., re-identification). Across CUB, Stanford Cars, and animal Re-ID benchmarks, Finer-Personalization Rank more faithfully reflects identity retention than semantic-only metrics and reveals substantial identity drift in several popular personalization methods. These results position the gallery-based protocol as a principled and practical evaluation for personalized generation. 

---
# The Erasure Illusion: Stress-Testing the Generalization of LLM Forgetting Evaluation 

**Authors**: Hengrui Jia, Taoran Li, Jonas Guan, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2512.19025)  

**Abstract**: Machine unlearning aims to remove specific data influences from trained models, a capability essential for adhering to copyright laws and ensuring AI safety. Current unlearning metrics typically measure success by monitoring the model's performance degradation on the specific unlearning dataset ($D_u$). We argue that for Large Language Models (LLMs), this evaluation paradigm is insufficient and potentially misleading. Many real-world uses of unlearning--motivated by copyright or safety--implicitly target not only verbatim content in $D_u$, but also behaviors influenced by the broader generalizations the model derived from it. We demonstrate that LLMs can pass standard unlearning evaluation and appear to have ``forgotten'' the target knowledge, while simultaneously retaining strong capabilities on content that is semantically adjacent to $D_u$. This phenomenon indicates that erasing exact sentences does not necessarily equate to removing the underlying knowledge. To address this gap, we propose \name, an automated stress-testing framework that generates a surrogate dataset, $\tilde{D}_u$. This surrogate set is constructed to be semantically derived from $D_u$ yet sufficiently distinct in embedding space. By comparing unlearning metric scores between $D_u$ and $\tilde{D}_u$, we can stress-test the reliability of the metric itself. Our extensive evaluation across three LLM families (Llama-3-8B, Qwen2.5-7B, and Zephyr-7B-$\beta$), three distinct datasets, and seven standard metrics reveals widespread inconsistencies. We find that current metrics frequently overestimate unlearning success, failing to detect retained knowledge exposed by our stress-test datasets. 

---
# IndoorUAV: Benchmarking Vision-Language UAV Navigation in Continuous Indoor Environments 

**Authors**: Xu Liu, Yu Liu, Hanshuo Qiu, Yang Qirong, Zhouhui Lian  

**Link**: [PDF](https://arxiv.org/pdf/2512.19024)  

**Abstract**: Vision-Language Navigation (VLN) enables agents to navigate in complex environments by following natural language instructions grounded in visual observations. Although most existing work has focused on ground-based robots or outdoor Unmanned Aerial Vehicles (UAVs), indoor UAV-based VLN remains underexplored, despite its relevance to real-world applications such as inspection, delivery, and search-and-rescue in confined spaces. To bridge this gap, we introduce \textbf{IndoorUAV}, a novel benchmark and method specifically tailored for VLN with indoor UAVs. We begin by curating over 1,000 diverse and structurally rich 3D indoor scenes from the Habitat simulator. Within these environments, we simulate realistic UAV flight dynamics to collect diverse 3D navigation trajectories manually, further enriched through data augmentation techniques. Furthermore, we design an automated annotation pipeline to generate natural language instructions of varying granularity for each trajectory. This process yields over 16,000 high-quality trajectories, comprising the \textbf{IndoorUAV-VLN} subset, which focuses on long-horizon VLN. To support short-horizon planning, we segment long trajectories into sub-trajectories by selecting semantically salient keyframes and regenerating concise instructions, forming the \textbf{IndoorUAV-VLA} subset. Finally, we introduce \textbf{IndoorUAV-Agent}, a novel navigation model designed for our benchmark, leveraging task decomposition and multimodal reasoning. We hope IndoorUAV serves as a valuable resource to advance research on vision-language embodied AI in the indoor aerial navigation domain. 

---
# Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline 

**Authors**: Akshaj Prashanth Rao, Advait Singh, Saumya Kumaar Saksena, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.19011)  

**Abstract**: Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead.
Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time to completion from approximately 450s to 47s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators.
Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications. 

---
# The 6th International Verification of Neural Networks Competition (VNN-COMP 2025): Summary and Results 

**Authors**: Konstantin Kaulen, Tobias Ladner, Stanley Bak, Christopher Brix, Hai Duong, Thomas Flinkow, Taylor T. Johnson, Lukas Koller, Edoardo Manino, ThanhVu H Nguyen, Haoze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19007)  

**Abstract**: This report summarizes the 6th International Verification of Neural Networks Competition (VNN-COMP 2025), held as a part of the 8th International Symposium on AI Verification (SAIV), that was collocated with the 37th International Conference on Computer-Aided Verification (CAV). VNN-COMP is held annually to facilitate the fair and objective comparison of state-of-the-art neural network verification tools, encourage the standardization of tool interfaces, and bring together the neural network verification community. To this end, standardized formats for networks (ONNX) and specification (VNN-LIB) were defined, tools were evaluated on equal-cost hardware (using an automatic evaluation pipeline based on AWS instances), and tool parameters were chosen by the participants before the final test sets were made public. In the 2025 iteration, 8 teams participated on a diverse set of 16 regular and 9 extended benchmarks. This report summarizes the rules, benchmarks, participating tools, results, and lessons learned from this iteration of this competition. 

---
# Context-Aware Initialization for Reducing Generative Path Length in Diffusion Language Models 

**Authors**: Tongyuan Miao, Gary Huang, Kai Jun Han, Annie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19004)  

**Abstract**: Diffusion Large Language Models (DLLMs) enable fully parallel token decoding but often remain impractical at inference time due to the many denoising iterations required to refine an information-free, fully masked initialization into coherent text. Most existing acceleration methods focus on traversing this generative trajectory more efficiently via improved solvers or sampling strategies. We advance a complementary perspective: shorten the trajectory itself by starting closer to the target distribution through context-aware initialization.
We propose a training-free interface that injects prompt-conditioned priors from a lightweight auxiliary model into the diffusion initialization, and instantiate it with two mechanisms: discrete token injection and representation-level embedding interpolation. Because injected priors can be imperfect and unmask-only decoding can over-commit early, we also introduce a simple confidence-based remasking mechanism as a form of prior skepticism. Preliminary evidence on GSM8K suggests that context-aware initialization can substantially reduce denoising iterations (about 35\% fewer function evaluations in our setting), while also exposing a key open challenge: naive warm-starting can degrade final accuracy relative to strong diffusion baselines. We use these findings to motivate a research agenda around calibration, revision mechanisms, and representation alignment for reliable warm-started diffusion decoding. 

---
# Evaluating the Challenges of LLMs in Real-world Medical Follow-up: A Comparative Study and An Optimized Framework 

**Authors**: Jinyan Liu, Zikang Chen, Qinchuan Wang, Tan Xie, Heming Zheng, Xudong Lv  

**Link**: [PDF](https://arxiv.org/pdf/2512.18999)  

**Abstract**: When applied directly in an end-to-end manner to medical follow-up tasks, Large Language Models (LLMs) often suffer from uncontrolled dialog flow and inaccurate information extraction due to the complexity of follow-up forms. To address this limitation, we designed and compared two follow-up chatbot systems: an end-to-end LLM-based system (control group) and a modular pipeline with structured process control (experimental group). Experimental results show that while the end-to-end approach frequently fails on lengthy and complex forms, our modular method-built on task decomposition, semantic clustering, and flow management-substantially improves dialog stability and extraction accuracy. Moreover, it reduces the number of dialogue turns by 46.73% and lowers token consumption by 80% to 87.5%. These findings highlight the necessity of integrating external control mechanisms when deploying LLMs in high-stakes medical follow-up scenarios. 

---
# ICP-4D: Bridging Iterative Closest Point and LiDAR Panoptic Segmentation 

**Authors**: Gyeongrok Oh, Youngdong Jang, Jonghyun Choi, Suk-Ju Kang, Guang Lin, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.18991)  

**Abstract**: Dominant paradigms for 4D LiDAR panoptic segmentation are usually required to train deep neural networks with large superimposed point clouds or design dedicated modules for instance association. However, these approaches perform redundant point processing and consequently become computationally expensive, yet still overlook the rich geometric priors inherently provided by raw point clouds. To this end, we introduce ICP-4D, a simple yet effective training-free framework that unifies spatial and temporal reasoning through geometric relations among instance-level point sets. Specifically, we apply the Iterative Closest Point (ICP) algorithm to directly associate temporally consistent instances by aligning the source and target point sets through the estimated transformation. To stabilize association under noisy instance predictions, we introduce a Sinkhorn-based soft matching. This exploits the underlying instance distribution to obtain accurate point-wise correspondences, resulting in robust geometric alignment. Furthermore, our carefully designed pipeline, which considers three instance types-static, dynamic, and missing-offers computational efficiency and occlusion-aware matching. Our extensive experiments across both SemanticKITTI and panoptic nuScenes demonstrate that our method consistently outperforms state-of-the-art approaches, even without additional training or extra point cloud inputs. 

---
# R-GenIMA: Integrating Neuroimaging and Genetics with Interpretable Multimodal AI for Alzheimer's Disease Progression 

**Authors**: Kun Zhao, Siyuan Dai, Yingying Zhang, Guodong Liu, Pengfei Gu, Chenghua Lin, Paul M. Thompson, Alex Leow, Heng Huang, Lifang He, Liang Zhan, Haoteng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18986)  

**Abstract**: Early detection of Alzheimer's disease (AD) requires models capable of integrating macro-scale neuroanatomical alterations with micro-scale genetic susceptibility, yet existing multimodal approaches struggle to align these heterogeneous signals. We introduce R-GenIMA, an interpretable multimodal large language model that couples a novel ROI-wise vision transformer with genetic prompting to jointly model structural MRI and single nucleotide polymorphisms (SNPs) variations. By representing each anatomically parcellated brain region as a visual token and encoding SNP profiles as structured text, the framework enables cross-modal attention that links regional atrophy patterns to underlying genetic factors. Applied to the ADNI cohort, R-GenIMA achieves state-of-the-art performance in four-way classification across normal cognition (NC), subjective memory concerns (SMC), mild cognitive impairment (MCI), and AD. Beyond predictive accuracy, the model yields biologically meaningful explanations by identifying stage-specific brain regions and gene signatures, as well as coherent ROI-Gene association patterns across the disease continuum. Attention-based attribution revealed genes consistently enriched for established GWAS-supported AD risk loci, including APOE, BIN1, CLU, and RBFOX1. Stage-resolved neuroanatomical signatures identified shared vulnerability hubs across disease stages alongside stage-specific patterns: striatal involvement in subjective decline, frontotemporal engagement during prodromal impairment, and consolidated multimodal network disruption in AD. These results demonstrate that interpretable multimodal AI can synthesize imaging and genetics to reveal mechanistic insights, providing a foundation for clinically deployable tools that enable earlier risk stratification and inform precision therapeutic strategies in Alzheimer's disease. 

---
# Self-Attention with State-Object Weighted Combination for Compositional Zero Shot Learning 

**Authors**: Cheng-Hong Chang, Pei-Hsuan Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2512.18969)  

**Abstract**: Object recognition has become prevalent across various industries. However, most existing applications are limited to identifying objects alone, without considering their associated states. The ability to recognize both the state and object simultaneously remains less common. One approach to address this is by treating state and object as a single category during training. However, this approach poses challenges in data collection and training since it requires comprehensive data for all possible combinations. Compositional Zero-shot Learning (CZSL) emerges as a viable solution by treating the state and object as distinct categories during training. CZSL facilitates the identification of novel compositions even in the absence of data for every conceivable combination. The current state-of-the-art method, KG-SP, addresses this issue by training distinct classifiers for states and objects, while leveraging a semantic model to evaluate the plausibility of composed compositions. However, KG-SP's accuracy in state and object recognition can be further improved, and it fails to consider the weighting of states and objects during composition. In this study, we propose SASOW, an enhancement of KG-SP that considers the weighting of states and objects while improving composition recognition accuracy. First, we introduce self-attention mechanisms into the classifiers for states and objects, leading to enhanced accuracy in recognizing both. Additionally, we incorporate the weighting of states and objects during composition to generate more reasonable and accurate compositions. Our validation process involves testing SASOW on three established benchmark datasets. Experimental outcomes affirm when compared against OW-CZSL approach, KG-SP, SASOW showcases improvements of 2.1%, 1.7%, and 0.4% in terms of accuracy for unseen compositions across the MIT-States, UT Zappos, and C-GQA datasets, respectively. 

---
# Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement 

**Authors**: Saman Forouzandeh, Wei Peng, Parham Moradi, Xinghuo Yu, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2512.18950)  

**Abstract**: We present MACLA, a framework that decouples reasoning from learning by maintaining a frozen large language model while performing all adaptation in an external hierarchical procedural memory. MACLA extracts reusable procedures from trajectories, tracks reliability via Bayesian posteriors, selects actions through expected-utility scoring, and refines procedures by contrasting successes and failures. Across four benchmarks (ALFWorld, WebShop, TravelPlanner, InterCodeSQL), MACLA achieves 78.1 percent average performance, outperforming all baselines. On ALFWorld unseen tasks, MACLA reaches 90.3 percent with 3.1 percent positive generalization. The system constructs memory in 56 seconds, 2800 times faster than the state-of-the-art LLM parameter-training baseline, compressing 2851 trajectories into 187 procedures. Experimental results demonstrate that structured external memory with Bayesian selection and contrastive refinement enables sample-efficient, interpretable, and continually improving agents without LLM parameter updates. 

---
# When Less is More: 8-bit Quantization Improves Continual Learning in Large Language Models 

**Authors**: Michael S. Zhang, Rishi A. Ruia, Arnav Kewalram, Saathvik Dharmapuram, Utkarsh Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18934)  

**Abstract**: Catastrophic forgetting poses a fundamental challenge in continual learning, particularly when models are quantized for deployment efficiency. We systematically investigate the interplay between quantization precision (FP16, INT8, INT4) and replay buffer strategies in large language models, revealing unexpected dynamics. While FP16 achieves superior initial task performance (74.44% on NLU), we observe a striking inversion on subsequent tasks: quantized models outperform FP16 by 8-15% on final task forward accuracy, with INT4 achieving nearly double FP16's performance on Code generation (40% vs 20%). Critically, even minimal replay buffers (0.1%) dramatically improve retention - increasing NLU retention after Math training from 45% to 65% across all precision levels - with INT8 consistently achieving the optimal balance between learning plasticity and knowledge retention. We hypothesize that quantization-induced noise acts as implicit regularization, preventing the overfitting to new task gradients that plagues high-precision models. These findings challenge the conventional wisdom that higher precision is always preferable, suggesting instead that INT8 quantization offers both computational efficiency and superior continual learning dynamics. Our results provide practical guidelines for deploying compressed models in continual learning scenarios: small replay buffers (1-2%) suffice for NLU tasks, while Math and Code benefit from moderate buffers (5-10%), with quantized models requiring less replay than FP16 to achieve comparable retention. Code is available at this https URL. 

---
# LouvreSAE: Sparse Autoencoders for Interpretable and Controllable Style Transfer 

**Authors**: Raina Panda, Daniel Fein, Arpita Singhal, Mark Fiore, Maneesh Agrawala, Matyas Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2512.18930)  

**Abstract**: Artistic style transfer in generative models remains a significant challenge, as existing methods often introduce style only via model fine-tuning, additional adapters, or prompt engineering, all of which can be computationally expensive and may still entangle style with subject matter. In this paper, we introduce a training- and inference-light, interpretable method for representing and transferring artistic style. Our approach leverages an art-specific Sparse Autoencoder (SAE) on top of latent embeddings of generative image models. Trained on artistic data, our SAE learns an emergent, largely disentangled set of stylistic and compositional concepts, corresponding to style-related elements pertaining brushwork, texture, and color palette, as well as semantic and structural concepts. We call it LouvreSAE and use it to construct style profiles: compact, decomposable steering vectors that enable style transfer without any model updates or optimization. Unlike prior concept-based style transfer methods, our method requires no fine-tuning, no LoRA training, and no additional inference passes, enabling direct steering of artistic styles from only a few reference images. We validate our method on ArtBench10, achieving or surpassing existing methods on style evaluations (VGG Style Loss and CLIP Score Style) while being 1.7-20x faster and, critically, interpretable. 

---
# An Empirical Study of Developer-Provided Context for AI Coding Assistants in Open-Source Projects 

**Authors**: Shaokang Jiang, Daye Nam  

**Link**: [PDF](https://arxiv.org/pdf/2512.18925)  

**Abstract**: While Large Language Models (LLMs) have demonstrated remarkable capabilities, research shows that their effectiveness depends not only on explicit prompts but also on the broader context provided. This requirement is especially pronounced in software engineering, where the goals, architecture, and collaborative conventions of an existing project play critical roles in response quality. To support this, many AI coding assistants have introduced ways for developers to author persistent, machine-readable directives that encode a project's unique constraints. Although this practice is growing, the content of these directives remains unstudied.
This paper presents a large-scale empirical study to characterize this emerging form of developer-provided context. Through a qualitative analysis of 401 open-source repositories containing cursor rules, we developed a comprehensive taxonomy of project context that developers consider essential, organized into five high-level themes: Conventions, Guidelines, Project Information, LLM Directives, and Examples. Our study also explores how this context varies across different project types and programming languages, offering implications for the next generation of context-aware AI developer tools. 

---
# Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics 

**Authors**: Yucheng Yang, Chiyuan Wang, Andreas Schaab, Benjamin Moll  

**Link**: [PDF](https://arxiv.org/pdf/2512.18892)  

**Abstract**: We present a new approach to formulating and solving heterogeneous agent models with aggregate risk. We replace the cross-sectional distribution with low-dimensional prices as state variables and let agents learn equilibrium price dynamics directly from simulated paths. To do so, we introduce a structural reinforcement learning (SRL) method which treats prices via simulation while exploiting agents' structural knowledge of their own individual dynamics. Our SRL method yields a general and highly efficient global solution method for heterogeneous agent models that sidesteps the Master equation and handles problems traditional methods struggle with, in particular nontrivial market-clearing conditions. We illustrate the approach in the Krusell-Smith model, the Huggett model with aggregate shocks, and a HANK model with a forward-looking Phillips curve, all of which we solve globally within minutes. 

---
# Can LLMs Estimate Student Struggles? Human-AI Difficulty Alignment with Proficiency Simulation for Item Difficulty Prediction 

**Authors**: Ming Li, Han Chen, Yunze Xiao, Jian Chen, Hong Jiao, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.18880)  

**Abstract**: Accurate estimation of item (question or task) difficulty is critical for educational assessment but suffers from the cold start problem. While Large Language Models demonstrate superhuman problem-solving capabilities, it remains an open question whether they can perceive the cognitive struggles of human learners. In this work, we present a large-scale empirical analysis of Human-AI Difficulty Alignment for over 20 models across diverse domains such as medical knowledge and mathematical reasoning. Our findings reveal a systematic misalignment where scaling up model size is not reliably helpful; instead of aligning with humans, models converge toward a shared machine consensus. We observe that high performance often impedes accurate difficulty estimation, as models struggle to simulate the capability limitations of students even when being explicitly prompted to adopt specific proficiency levels. Furthermore, we identify a critical lack of introspection, as models fail to predict their own limitations. These results suggest that general problem-solving capability does not imply an understanding of human cognitive struggles, highlighting the challenge of using current models for automated difficulty prediction. 

---
# CrashChat: A Multimodal Large Language Model for Multitask Traffic Crash Video Analysis 

**Authors**: Kaidi Liang, Ke Li, Xianbiao Hu, Ruwen Qin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18878)  

**Abstract**: Automating crash video analysis is essential to leverage the growing availability of driving video data for traffic safety research and accountability attribution in autonomous driving. Crash video analysis is a challenging multitask problem due to the complex spatiotemporal dynamics of crash events in video data and the diverse analytical requirements involved. It requires capabilities spanning crash recognition, temporal grounding, and high-level video understanding. Existing models, however, cannot perform all these tasks within a unified framework, and effective training strategies for such models remain underexplored. To fill these gaps, this paper proposes CrashChat, a multimodal large language model (MLLM) for multitask traffic crash analysis, built upon VideoLLaMA3. CrashChat acquires domain-specific knowledge through instruction fine-tuning and employs a novel multitask learning strategy based on task decoupling and grouping, which maximizes the benefit of joint learning within and across task groups while mitigating negative transfer. Numerical experiments on consolidated public datasets demonstrate that CrashChat consistently outperforms existing MLLMs across model scales and traditional vision-based methods, achieving state-of-the-art performance. It reaches near-perfect accuracy in crash recognition, a 176\% improvement in crash localization, and a 40\% improvement in the more challenging pre-crash localization. Compared to general MLLMs, it substantially enhances textual accuracy and content coverage in crash description and reasoning tasks, with 0.18-0.41 increases in BLEU scores and 0.18-0.42 increases in ROUGE scores. Beyond its strong performance, CrashChat is a convenient, end-to-end analytical tool ready for practical implementation. The dataset and implementation code for CrashChat are available at this https URL. 

---
# Psychometric Validation of the Sophotechnic Mediation Scale and a New Understanding of the Development of GenAI Mastery: Lessons from 3,392 Adult Brazilian Workers 

**Authors**: Bruno Campello de Souza  

**Link**: [PDF](https://arxiv.org/pdf/2512.18871)  

**Abstract**: The rapid diffusion of generative artificial intelligence (GenAI) systems has introduced new forms of human-technology interaction, raising the question of whether sustained engagement gives rise to stable, internalized modes of cognition rather than merely transient efficiency gains. Grounded in the Cognitive Mediation Networks Theory, this study investigates Sophotechnic Mediation, a mode of thinking and acting associated with prolonged interaction with GenAI, and presents a comprehensive psychometric validation of the Sophotechnic Mediation Scale. Data were collected between 2023 and 2025 from independent cross-sectional samples totaling 3,932 adult workers from public and private organizations in the Metropolitan Region of Pernambuco, Brazil. Results indicate excellent internal consistency, a robust unidimensional structure, and measurement invariance across cohorts. Ordinal-robust confirmatory factor analyses and residual diagnostics show that elevated absolute fit indices reflect minor local dependencies rather than incorrect dimensionality. Distributional analyses reveal a time-evolving pattern characterized by a declining mass of non-adopters and convergence toward approximate Gaussianity among adopters, with model comparisons favoring a two-process hurdle model over a censored Gaussian specification. Sophotechnic Mediation is empirically distinct from Hypercultural mediation and is primarily driven by cumulative GenAI experience, with age moderating the rate of initial acquisition and the depth of later integration. Together, the findings support Sophotechnia as a coherent, measurable, and emergent mode of cognitive mediation associated with the ongoing GenAI revolution. 

---
# Hyperbolic Graph Embeddings: a Survey and an Evaluation on Anomaly Detection 

**Authors**: Souhail Abdelmouaiz Sadat, Mohamed Yacine Touahria Miliani, Khadidja Hab El Hames, Hamida Seba, Mohammed Haddad  

**Link**: [PDF](https://arxiv.org/pdf/2512.18826)  

**Abstract**: This survey reviews hyperbolic graph embedding models, and evaluate them on anomaly detection, highlighting their advantages over Euclidean methods in capturing complex structures. Evaluating models like \textit{HGCAE}, \textit{\(\mathcal{P}\)-VAE}, and \textit{HGCN} demonstrates high performance, with \textit{\(\mathcal{P}\)-VAE} achieving an F1-score of 94\% on the \textit{Elliptic} dataset and \textit{HGCAE} scoring 80\% on \textit{Cora}. In contrast, Euclidean methods like \textit{DOMINANT} and \textit{GraphSage} struggle with complex data. The study emphasizes the potential of hyperbolic spaces for improving anomaly detection, and provides an open-source library to foster further research in this field. 

---
# Controllable Probabilistic Forecasting with Stochastic Decomposition Layers 

**Authors**: John S. Schreck, William E. Chapman, Charlie Becker, David John Gagne II, Dhamma Kimpara, Nihanth Cherukuru, Judith Berner, Kirsten J. Mayer, Negin Sobhani  

**Link**: [PDF](https://arxiv.org/pdf/2512.18815)  

**Abstract**: AI weather prediction ensembles with latent noise injection and optimized with the continuous ranked probability score (CRPS) have produced both accurate and well-calibrated predictions with far less computational cost compared with diffusion-based methods. However, current CRPS ensemble approaches vary in their training strategies and noise injection mechanisms, with most injecting noise globally throughout the network via conditional normalization. This structure increases training expense and limits the physical interpretability of the stochastic perturbations. We introduce Stochastic Decomposition Layers (SDL) for converting deterministic machine learning weather models into probabilistic ensemble systems. Adapted from StyleGAN's hierarchical noise injection, SDL applies learned perturbations at three decoder scales through latent-driven modulation, per-pixel noise, and channel scaling. When applied to WXFormer via transfer learning, SDL requires less than 2\% of the computational cost needed to train the baseline model. Each ensemble member is generated from a compact latent tensor (5 MB), enabling perfect reproducibility and post-inference spread adjustment through latent rescaling. Evaluation on 2022 ERA5 reanalysis shows ensembles with spread-skill ratios approaching unity and rank histograms that progressively flatten toward uniformity through medium-range forecasts, achieving calibration competitive with operational IFS-ENS. Multi-scale experiments reveal hierarchical uncertainty: coarse layers modulate synoptic patterns while fine layers control mesoscale variability. The explicit latent parameterization provides interpretable uncertainty quantification for operational forecasting and climate applications. 

---
# FedVideoMAE: Efficient Privacy-Preserving Federated Video Moderation 

**Authors**: Ziyuan Tao, Chuanzhi Xu, Sandaru Jayawardana, Wei Bao, Kanchana Thilakarathna, Teng Joon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2512.18809)  

**Abstract**: The rapid growth of short-form video platforms increases the need for privacy-preserving moderation, as cloud-based pipelines expose raw videos to privacy risks, high bandwidth costs, and inference latency. To address these challenges, we propose an on-device federated learning framework for video violence detection that integrates self-supervised VideoMAE representations, LoRA-based parameter-efficient adaptation, and defense-in-depth privacy protection. Our approach reduces the trainable parameter count to 5.5M (~3.5% of a 156M backbone) and incorporates DP-SGD with configurable privacy budgets and secure aggregation. Experiments on RWF-2000 with 40 clients achieve 77.25% accuracy without privacy protection and 65-66% under strong differential privacy, while reducing communication cost by $28.3\times$ compared to full-model federated learning. The code is available at: {this https URL} 

---
# Reliable Audio Deepfake Detection in Variable Conditions via Quantum-Kernel SVMs 

**Authors**: Lisan Al Amin, Vandana P. Janeja  

**Link**: [PDF](https://arxiv.org/pdf/2512.18797)  

**Abstract**: Detecting synthetic speech is challenging when labeled data are scarce and recording conditions vary. Existing end-to-end deep models often overfit or fail to generalize, and while kernel methods can remain competitive, their performance heavily depends on the chosen kernel. Here, we show that using a quantum kernel in audio deepfake detection reduces falsepositive rates without increasing model size. Quantum feature maps embed data into high-dimensional Hilbert spaces, enabling the use of expressive similarity measures and compact classifiers. Building on this motivation, we compare quantum-kernel SVMs (QSVMs) with classical SVMs using identical mel-spectrogram preprocessing and stratified 5-fold cross-validation across four corpora (ASVspoof 2019 LA, ASVspoof 5 (2024), ADD23, and an In-the-Wild set). QSVMs achieve consistently lower equalerror rates (EER): 0.183 vs. 0.299 on ASVspoof 5 (2024), 0.081 vs. 0.188 on ADD23, 0.346 vs. 0.399 on ASVspoof 2019, and 0.355 vs. 0.413 In-the-Wild. At the EER operating point (where FPR equals FNR), these correspond to absolute false-positiverate reductions of 0.116 (38.8%), 0.107 (56.9%), 0.053 (13.3%), and 0.058 (14.0%), respectively. We also report how consistent the results are across cross-validation folds and margin-based measures of class separation, using identical settings for both models. The only modification is the kernel; the features and SVM remain unchanged, no additional trainable parameters are introduced, and the quantum kernel is computed on a conventional computer. 

---
# Smark: A Watermark for Text-to-Speech Diffusion Models via Discrete Wavelet Transform 

**Authors**: Yichuan Zhang, Chengxin Li, Yujie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18791)  

**Abstract**: Text-to-Speech (TTS) diffusion models generate high-quality speech, which raises challenges for the model intellectual property protection and speech tracing for legal use. Audio watermarking is a promising solution. However, due to the structural differences among various TTS diffusion models, existing watermarking methods are often designed for a specific model and degrade audio quality, which limits their practical applicability. To address this dilemma, this paper proposes a universal watermarking scheme for TTS diffusion models, termed Smark. This is achieved by designing a lightweight watermark embedding framework that operates in the common reverse diffusion paradigm shared by all TTS diffusion models. To mitigate the impact on audio quality, Smark utilizes the discrete wavelet transform (DWT) to embed watermarks into the relatively stable low-frequency regions of the audio, which ensures seamless watermark-audio integration and is resistant to removal during the reverse diffusion process. Extensive experiments are conducted to evaluate the audio quality and watermark performance in various simulated real-world attack scenarios. The experimental results show that Smark achieves superior performance in both audio quality and watermark extraction accuracy. 

---
# Code2Doc: A Quality-First Curated Dataset for Code Documentation 

**Authors**: Recep Kaan Karaman, Meftun Akarsu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18748)  

**Abstract**: The performance of automatic code documentation generation models depends critically on the quality of the training data used for supervision. However, most existing code documentation datasets are constructed through large scale scraping of public repositories with limited quality control. As a result, they often contain noisy documentation, extensive duplication, and increasing contamination from AI generated content. These issues weaken the supervision signal available to learning-based models and complicate evaluation.
We introduce \textbf{Code2Doc}, a quality-first curated dataset for function-level code documentation generation. Code2Doc consists of 13,358 high-quality function-documentation pairs extracted from widely used open-source repositories spanning five programming languages: Python, Java, TypeScript, JavaScript, and C++. The dataset is constructed using a four-stage curation pipeline that enforces documentation completeness and clarity, filters functions based on structural and complexity criteria, removes exact and near-duplicate code, and identifies documentation likely to be AI generated. Starting from 52,069 extracted candidates, only 25.6 percent satisfy all quality constraints.
We provide a detailed analysis of the resulting dataset, which achieves a mean documentation quality score of 6.93 out of 10. Overall, 86.9% of samples contain explicit type annotations, and only 2.9\% are flagged as potentially AI generated. Baseline experiments show that fine-tuning a large language model on Code2Doc yields relative improvements of 29.47% in BLEU and 24.04% in ROUGE-L over zero shot performance, despite the modest dataset size. We release both the dataset and the full curation pipeline to support reproducible research on automatic code documentation generation. 

---
# IPCV: Information-Preserving Compression for MLLM Visual Encoders 

**Authors**: Yuan Chen, Zichen Wen, Yuzhou Wu, Xuyang Liu, Shuang Chen, Junpeng Ma, Weijia Li, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18747)  

**Abstract**: Multimodal Large Language Models (MLLMs) deliver strong vision-language performance but at high computational cost, driven by numerous visual tokens processed by the Vision Transformer (ViT) encoder. Existing token pruning strategies are inadequate: LLM-stage token pruning overlooks the ViT's overhead, while conventional ViT token pruning, without language guidance, risks discarding textually critical visual cues and introduces feature distortions amplified by the ViT's bidirectional attention. To meet these challenges, we propose IPCV, a training-free, information-preserving compression framework for MLLM visual encoders. IPCV enables aggressive token pruning inside the ViT via Neighbor-Guided Reconstruction (NGR) that temporarily reconstructs pruned tokens to participate in attention with minimal overhead, then fully restores them before passing to the LLM. Besides, we introduce Attention Stabilization (AS) to further alleviate the negative influence from token pruning by approximating the K/V of pruned tokens. It can be directly applied to previous LLM-side token pruning methods to enhance their performance. Extensive experiments show that IPCV substantially reduces end-to-end computation and outperforms state-of-the-art training-free token compression methods across diverse image and video benchmarks. Our code is available at this https URL. 

---
# PIPCFR: Pseudo-outcome Imputation with Post-treatment Variables for Individual Treatment Effect Estimation 

**Authors**: Zichuan Lin, Xiaokai Huang, Jiate Liu, Yuxuan Han, Jia Chen, Xiapeng Wu, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.18737)  

**Abstract**: The estimation of individual treatment effects (ITE) focuses on predicting the outcome changes that result from a change in treatment. A fundamental challenge in observational data is that while we need to infer outcome differences under alternative treatments, we can only observe each individual's outcome under a single treatment. Existing approaches address this limitation either by training with inferred pseudo-outcomes or by creating matched instance pairs. However, recent work has largely overlooked the potential impact of post-treatment variables on the outcome. This oversight prevents existing methods from fully capturing outcome variability, resulting in increased variance in counterfactual predictions. This paper introduces Pseudo-outcome Imputation with Post-treatment Variables for Counterfactual Regression (PIPCFR), a novel approach that incorporates post-treatment variables to improve pseudo-outcome imputation. We analyze the challenges inherent in utilizing post-treatment variables and establish a novel theoretical bound for ITE risk that explicitly connects post-treatment variables to ITE estimation accuracy. Unlike existing methods that ignore these variables or impose restrictive assumptions, PIPCFR learns effective representations that preserve informative components while mitigating bias. Empirical evaluations on both real-world and simulated datasets demonstrate that PIPCFR achieves significantly lower ITE errors compared to existing methods. 

---
# $M^3-Verse$: A "Spot the Difference" Challenge for Large Multimodal Models 

**Authors**: Kewei Wei, Bocheng Hu, Jie Cao, Xiaohan Chen, Zhengxi Lu, Wubing Xia, Weili Xu, Jiaao Wu, Junchen He, Mingyu Jia, Ciyun Zhao, Ye Sun, Yizhi Li, Zhonghan Zhao, Jian Zhang, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18735)  

**Abstract**: Modern Large Multimodal Models (LMMs) have demonstrated extraordinary ability in static image and single-state spatial-temporal understanding. However, their capacity to comprehend the dynamic changes of objects within a shared spatial context between two distinct video observations, remains largely unexplored. This ability to reason about transformations within a consistent environment is particularly crucial for advancements in the field of spatial intelligence. In this paper, we introduce $M^3-Verse$, a Multi-Modal, Multi-State, Multi-Dimensional benchmark, to formally evaluate this capability. It is built upon paired videos that provide multi-perspective observations of an indoor scene before and after a state change. The benchmark contains a total of 270 scenes and 2,932 questions, which are categorized into over 50 subtasks that probe 4 core capabilities. We evaluate 16 state-of-the-art LMMs and observe their limitations in tracking state transitions. To address these challenges, we further propose a simple yet effective baseline that achieves significant performance improvements in multi-state perception. $M^3-Verse$ thus provides a challenging new testbed to catalyze the development of next-generation models with a more holistic understanding of our dynamic visual world. You can get the construction pipeline from this https URL and full benchmark data from this https URL. 

---
# Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection 

**Authors**: Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2512.18733)  

**Abstract**: Large language model (LLM)-based multi-agent systems (MAS) have shown strong capabilities in solving complex tasks. As MAS become increasingly autonomous in various safety-critical tasks, detecting malicious agents has become a critical security concern. Although existing graph anomaly detection (GAD)-based defenses can identify anomalous agents, they mainly rely on coarse sentence-level information and overlook fine-grained lexical cues, leading to suboptimal performance. Moreover, the lack of interpretability in these methods limits their reliability and real-world applicability. To address these limitations, we propose XG-Guard, an explainable and fine-grained safeguarding framework for detecting malicious agents in MAS. To incorporate both coarse and fine-grained textual information for anomalous agent identification, we utilize a bi-level agent encoder to jointly model the sentence- and token-level representations of each agent. A theme-based anomaly detector further captures the evolving discussion focus in MAS dialogues, while a bi-level score fusion mechanism quantifies token-level contributions for explanation. Extensive experiments across diverse MAS topologies and attack scenarios demonstrate robust detection performance and strong interpretability of XG-Guard. 

---
# CauTraj: A Causal-Knowledge-Guided Framework for Lane-Changing Trajectory Planning of Autonomous Vehicles 

**Authors**: Cailin Lei, Haiyang Wu, Yuxiong Ji, Xiaoyu Cai, Yuchuan Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.18703)  

**Abstract**: Enhancing the performance of trajectory planners for lane - changing vehicles is one of the key challenges in autonomous driving within human - machine mixed traffic. Most existing studies have not incorporated human drivers' prior knowledge when designing trajectory planning models. To address this issue, this study proposes a novel trajectory planning framework that integrates causal prior knowledge into the control process. Both longitudinal and lateral microscopic behaviors of vehicles are modeled to quantify interaction risk, and a staged causal graph is constructed to capture causal dependencies in lane-changing scenarios. Causal effects between the lane-changing vehicle and surrounding vehicles are then estimated using causal inference, including average causal effects (ATE) and conditional average treatment effects (CATE). These causal priors are embedded into a model predictive control (MPC) framework to enhance trajectory planning. The proposed approach is validated on naturalistic vehicle trajectory datasets. Experimental results show that: (1) causal inference provides interpretable and stable quantification of vehicle interactions; (2) individual causal effects reveal driver heterogeneity; and (3) compared with the baseline MPC, the proposed method achieves a closer alignment with human driving behaviors, reducing maximum trajectory deviation from 1.2 m to 0.2 m, lateral velocity fluctuation by 60%, and yaw angle variability by 50%. These findings provide methodological support for human-like trajectory planning and practical value for improving safety, stability, and realism in autonomous vehicle testing and traffic simulation platforms. 

---
# Fusion of Multiscale Features Via Centralized Sparse-attention Network for EEG Decoding 

**Authors**: Xiangrui Cai, Shaocheng Ma, Lei Cao, Jie Li, Tianyu Liu, Yilin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2512.18689)  

**Abstract**: Electroencephalography (EEG) signal decoding is a key technology that translates brain activity into executable commands, laying the foundation for direct brain-machine interfacing and intelligent interaction. To address the inherent spatiotemporal heterogeneity of EEG signals, this paper proposes a multi-branch parallel architecture, where each temporal scale is equipped with an independent spatial feature extraction module. To further enhance multi-branch feature fusion, we propose a Fusion of Multiscale Features via Centralized Sparse-attention Network (EEG-CSANet), a centralized sparse-attention network. It employs a main-auxiliary branch architecture, where the main branch models core spatiotemporal patterns via multiscale self-attention, and the auxiliary branch facilitates efficient local interactions through sparse cross-attention. Experimental results show that EEG-CSANet achieves state-of-the-art (SOTA) performance across five public datasets (BCIC-IV-2A, BCIC-IV-2B, HGD, SEED, and SEED-VIG), with accuracies of 88.54%, 91.09%, 99.43%, 96.03%, and 90.56%, respectively. Such performance demonstrates its strong adaptability and robustness across various EEG decoding tasks. Moreover, extensive ablation studies are conducted to enhance the interpretability of EEG-CSANet. In the future, we hope that EEG-CSANet could serve as a promising baseline model in the field of EEG signal decoding. The source code is publicly available at: this https URL 

---
# Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing 

**Authors**: Wentao Liu, Yuhao Hu, Ruiting Zhou, Baochun Li, Ne Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18674)  

**Abstract**: Mixture-of-Experts (MoE) has become a dominant architecture in large language models (LLMs) due to its ability to scale model capacity via sparse expert activation. Meanwhile, serverless computing, with its elasticity and pay-per-use billing, is well-suited for deploying MoEs with bursty workloads. However, the large number of experts in MoE models incurs high inference costs due to memory-intensive parameter caching. These costs are difficult to mitigate via simple model partitioning due to input-dependent expert activation. To address these issues, we propose Remoe, a heterogeneous MoE inference system tailored for serverless computing. Remoe assigns non-expert modules to GPUs and expert modules to CPUs, and further offloads infrequently activated experts to separate serverless functions to reduce memory overhead and enable parallel execution. We incorporate three key techniques: (1) a Similar Prompts Searching (SPS) algorithm to predict expert activation patterns based on semantic similarity of inputs; (2) a Main Model Pre-allocation (MMP) algorithm to ensure service-level objectives (SLOs) via worst-case memory estimation; and (3) a joint memory and replica optimization framework leveraging Lagrangian duality and the Longest Processing Time (LPT) algorithm. We implement Remoe on Kubernetes and evaluate it across multiple LLM benchmarks. Experimental results show that Remoe reduces inference cost by up to 57% and cold start latency by 47% compared to state-of-the-art baselines. 

---
# Geometric-Photometric Event-based 3D Gaussian Ray Tracing 

**Authors**: Kai Kohyama, Yoshimitsu Aoki, Guillermo Gallego, Shintaro Shiba  

**Link**: [PDF](https://arxiv.org/pdf/2512.18640)  

**Abstract**: Event cameras offer a high temporal resolution over traditional frame-based cameras, which makes them suitable for motion and structure estimation. However, it has been unclear how event-based 3D Gaussian Splatting (3DGS) approaches could leverage fine-grained temporal information of sparse events. This work proposes a framework to address the trade-off between accuracy and temporal resolution in event-based 3DGS. Our key idea is to decouple the rendering into two branches: event-by-event geometry (depth) rendering and snapshot-based radiance (intensity) rendering, by using ray-tracing and the image of warped events. The extensive evaluation shows that our method achieves state-of-the-art performance on the real-world datasets and competitive performance on the synthetic dataset. Also, the proposed method works without prior information (e.g., pretrained image reconstruction models) or COLMAP-based initialization, is more flexible in the event selection number, and achieves sharp reconstruction on scene edges with fast training time. We hope that this work deepens our understanding of the sparse nature of events for 3D reconstruction. The code will be released. 

---
# ARC: Leveraging Compositional Representations for Cross-Problem Learning on VRPs 

**Authors**: Han-Seul Jeong, Youngjoon Park, Hyungseok Song, Woohyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2512.18633)  

**Abstract**: Vehicle Routing Problems (VRPs) with diverse real-world attributes have driven recent interest in cross-problem learning approaches that efficiently generalize across problem variants. We propose ARC (Attribute Representation via Compositional Learning), a cross-problem learning framework that learns disentangled attribute representations by decomposing them into two complementary components: an Intrinsic Attribute Embedding (IAE) for invariant attribute semantics and a Contextual Interaction Embedding (CIE) for attribute-combination effects. This disentanglement is achieved by enforcing analogical consistency in the embedding space to ensure the semantic transformation of adding an attribute (e.g., a length constraint) remains invariant across different problem contexts. This enables our model to reuse invariant semantics across trained variants and construct representations for unseen combinations. ARC achieves state-of-the-art performance across in-distribution, zero-shot generalization, few-shot adaptation, and real-world benchmarks. 

---
# LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction 

**Authors**: Jensen Zhang, Ningyuan Liu, Yijia Fan, Zihao Huang, Qinglin Zeng, Kaitong Cai, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18623)  

**Abstract**: Large language models (LLMs) often generate hallucinated content that lacks factual or contextual grounding, limiting their reliability in critical applications. Existing approaches such as supervised fine-tuning and reinforcement learning from human feedback are data intensive and computationally expensive, while static parameter editing methods struggle with context dependent errors and catastrophic forgetting.
We propose LLM-CAS, a framework that formulates real-time hallucination correction as a hierarchical reinforcement learning problem. LLM-CAS trains an agent to learn a policy that dynamically selects temporary neuron perturbations during inference based on the current context. Unlike prior dynamic approaches that rely on heuristic or predefined adjustments, this policy driven mechanism enables adaptive and fine grained correction without permanent parameter modification.
Experiments across multiple language models demonstrate that LLM-CAS consistently improves factual accuracy, achieving gains of 10.98 percentage points on StoryCloze, 2.71 points on TriviaQA, and 2.06 points on the MC1 score of TruthfulQA. These results outperform both static editing methods such as ITI and CAA and the dynamic SADI framework. Overall, LLM-CAS provides an efficient and context aware solution for improving the reliability of LLMs, with promising potential for future multimodal extensions. 

---
# A Multi-agent Text2SQL Framework using Small Language Models and Execution Feedback 

**Authors**: Thanh Dat Hoang, Thanh Trung Huynh, Matthias Weidlich, Thanh Tam Nguyen, Tong Chen, Hongzhi Yin, Quoc Viet Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2512.18622)  

**Abstract**: Text2SQL, the task of generating SQL queries from natural language text, is a critical challenge in data engineering. Recently, Large Language Models (LLMs) have demonstrated superior performance for this task due to their advanced comprehension and generation capabilities. However, privacy and cost considerations prevent companies from using Text2SQL solutions based on external LLMs offered as a service. Rather, small LLMs (SLMs) that are openly available and can hosted in-house are adopted. These SLMs, in turn, lack the generalization capabilities of larger LLMs, which impairs their effectiveness for complex tasks such as Text2SQL. To address these limitations, we propose MATS, a novel Text2SQL framework designed specifically for SLMs. MATS uses a multi-agent mechanism that assigns specialized roles to auxiliary agents, reducing individual workloads and fostering interaction. A training scheme based on reinforcement learning aligns these agents using feedback obtained during execution, thereby maintaining competitive performance despite a limited LLM size. Evaluation results using on benchmark datasets show that MATS, deployed on a single- GPU server, yields accuracy that are on-par with large-scale LLMs when using significantly fewer parameters. Our source code and data are available at this https URL. 

---
# DASH: Deception-Augmented Shared Mental Model for a Human-Machine Teaming System 

**Authors**: Zelin Wan, Han Jun Yoon, Nithin Alluru, Terrence J. Moore, Frederica F. Nelson, Seunghyun Yoon, Hyuk Lim, Dan Dongseong Kim, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2512.18616)  

**Abstract**: We present DASH (Deception-Augmented Shared mental model for Human-machine teaming), a novel framework that enhances mission resilience by embedding proactive deception into Shared Mental Models (SMM). Designed for mission-critical applications such as surveillance and rescue, DASH introduces "bait tasks" to detect insider threats, e.g., compromised Unmanned Ground Vehicles (UGVs), AI agents, or human analysts, before they degrade team performance. Upon detection, tailored recovery mechanisms are activated, including UGV system reinstallation, AI model retraining, or human analyst replacement. In contrast to existing SMM approaches that neglect insider risks, DASH improves both coordination and security. Empirical evaluations across four schemes (DASH, SMM-only, no-SMM, and baseline) show that DASH sustains approximately 80% mission success under high attack rates, eight times higher than the baseline. This work contributes a practical human-AI teaming framework grounded in shared mental models, a deception-based strategy for insider threat detection, and empirical evidence of enhanced robustness under adversarial conditions. DASH establishes a foundation for secure, adaptive human-machine teaming in contested environments. 

---
# PTTA: A Pure Text-to-Animation Framework for High-Quality Creation 

**Authors**: Ruiqi Chen, Kaitong Cai, Yijia Fan, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18614)  

**Abstract**: Traditional animation production involves complex pipelines and significant manual labor cost. While recent video generation models such as Sora, Kling, and CogVideoX achieve impressive results on natural video synthesis, they exhibit notable limitations when applied to animation generation. Recent efforts, such as AniSora, demonstrate promising performance by fine-tuning image-to-video models for animation styles, yet analogous exploration in the text-to-video setting remains limited.
In this work, we present PTTA, a pure text-to-animation framework for high-quality animation creation. We first construct a small-scale but high-quality paired dataset of animation videos and textual descriptions. Building upon the pretrained text-to-video model HunyuanVideo, we perform fine-tuning to adapt it to animation-style generation. Extensive visual evaluations across multiple dimensions show that the proposed approach consistently outperforms comparable baselines in animation video synthesis. 

---
# Text2Graph VPR: A Text-to-Graph Expert System for Explainable Place Recognition in Changing Environments 

**Authors**: Saeideh Yousefzadeh, Hamidreza Pourreza  

**Link**: [PDF](https://arxiv.org/pdf/2512.18613)  

**Abstract**: Visual Place Recognition (VPR) in long-term deployment requires reasoning beyond pixel similarity: systems must make transparent, interpretable decisions that remain robust under lighting, weather and seasonal change. We present Text2Graph VPR, an explainable semantic localization system that converts image sequences into textual scene descriptions, parses those descriptions into structured scene graphs, and reasons over the resulting graphs to identify places. Scene graphs capture objects, attributes and pairwise relations; we aggregate per-frame graphs into a compact place representation and perform retrieval with a dual-similarity mechanism that fuses learned Graph Attention Network (GAT) embeddings and a Shortest-Path (SP) kernel for structural matching. This hybrid design enables both learned semantic matching and topology-aware comparison, and -- critically -- produces human-readable intermediate representations that support diagnostic analysis and improve transparency in the decision process. We validate the system on Oxford RobotCar and MSLS (Amman/San Francisco) benchmarks and demonstrate robust retrieval under severe appearance shifts, along with zero-shot operation using human textual queries. The results illustrate that semantic, graph-based reasoning is a viable and interpretable alternative for place recognition, particularly suited to safety-sensitive and resource-constrained settings. 

---
# The Interaction Bottleneck of Deep Neural Networks: Discovery, Proof, and Modulation 

**Authors**: Huiqi Deng, Qihan Ren, Zhuofan Chen, Zhenyuan Cui, Wen Shen, Peng Zhang, Hongbin Pei, Quanshi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18607)  

**Abstract**: Understanding what kinds of cooperative structures deep neural networks (DNNs) can represent remains a fundamental yet insufficiently understood problem. In this work, we treat interactions as the fundamental units of such structure and investigate a largely unexplored question: how DNNs encode interactions under different levels of contextual complexity, and how these microscopic interaction patterns shape macroscopic representation capacity. To quantify this complexity, we use multi-order interactions [57], where each order reflects the amount of contextual information required to evaluate the joint interaction utility of a variable pair. This formulation enables a stratified analysis of cooperative patterns learned by DNNs. Building on this formulation, we develop a comprehensive study of interaction structure in DNNs. (i) We empirically discover a universal interaction bottleneck: across architectures and tasks, DNNs easily learn low-order and high-order interactions but consistently under-represent mid-order ones. (ii) We theoretically explain this bottleneck by proving that mid-order interactions incur the highest contextual variability, yielding large gradient variance and making them intrinsically difficult to learn. (iii) We further modulate the bottleneck by introducing losses that steer models toward emphasizing interactions of selected orders. Finally, we connect microscopic interaction structures with macroscopic representational behavior: low-order-emphasized models exhibit stronger generalization and robustness, whereas high-order-emphasized models demonstrate greater structural modeling and fitting capability. Together, these results uncover an inherent representational bias in modern DNNs and establish interaction order as a powerful lens for interpreting and guiding deep representations. 

---
# From Scratch to Fine-Tuned: A Comparative Study of Transformer Training Strategies for Legal Machine Translation 

**Authors**: Amit Barman, Atanu Mandal, Sudip Kumar Naskar  

**Link**: [PDF](https://arxiv.org/pdf/2512.18593)  

**Abstract**: In multilingual nations like India, access to legal information is often hindered by language barriers, as much of the legal and judicial documentation remains in English. Legal Machine Translation (L-MT) offers a scalable solution to this challenge by enabling accurate and accessible translations of legal documents. This paper presents our work for the JUST-NLP 2025 Legal MT shared task, focusing on English-Hindi translation using Transformer-based approaches. We experiment with 2 complementary strategies, fine-tuning a pre-trained OPUS-MT model for domain-specific adaptation and training a Transformer model from scratch using the provided legal corpus. Performance is evaluated using standard MT metrics, including SacreBLEU, chrF++, TER, ROUGE, BERTScore, METEOR, and COMET. Our fine-tuned OPUS-MT model achieves a SacreBLEU score of 46.03, significantly outperforming both baseline and from-scratch models. The results highlight the effectiveness of domain adaptation in enhancing translation quality and demonstrate the potential of L-MT systems to improve access to justice and legal transparency in multilingual contexts. 

---
# Modality-Dependent Memory Mechanisms in Cross-Modal Neuromorphic Computing 

**Authors**: Effiong Blessing, Chiung-Yi Tseng, Somshubhra Roy, Junaid Rehman, Isaac Nkrumah  

**Link**: [PDF](https://arxiv.org/pdf/2512.18575)  

**Abstract**: Memory-augmented spiking neural networks (SNNs) promise energy-efficient neuromorphic computing, yet their generalization across sensory modalities remains unexplored. We present the first comprehensive cross-modal ablation study of memory mechanisms in SNNs, evaluating Hopfield networks, Hierarchical Gated Recurrent Networks (HGRNs), and supervised contrastive learning (SCL) across visual (N-MNIST) and auditory (SHD) neuromorphic datasets. Our systematic evaluation of five architectures reveals striking modality-dependent performance patterns: Hopfield networks achieve 97.68% accuracy on visual tasks but only 76.15% on auditory tasks (21.53 point gap), revealing severe modality-specific specialization, while SCL demonstrates more balanced cross-modal performance (96.72% visual, 82.16% audio, 14.56 point gap). These findings establish that memory mechanisms exhibit task-specific benefits rather than universal applicability. Joint multi-modal training with HGRN achieves 94.41% visual and 79.37% audio accuracy (88.78% average), matching parallel HGRN performance through unified deployment. Quantitative engram analysis confirms weak cross-modal alignment (0.038 similarity), validating our parallel architecture design. Our work provides the first empirical evidence for modality-specific memory optimization in neuromorphic systems, achieving 603x energy efficiency over traditional neural networks. 

---
# Placenta Accreta Spectrum Detection Using an MRI-based Hybrid CNN-Transformer Model 

**Authors**: Sumaiya Ali, Areej Alhothali, Ohoud Alzamzami, Sameera Albasri, Ahmed Abduljabbar, Muhammad Alwazzan  

**Link**: [PDF](https://arxiv.org/pdf/2512.18573)  

**Abstract**: Placenta Accreta Spectrum (PAS) is a serious obstetric condition that can be challenging to diagnose with Magnetic Resonance Imaging (MRI) due to variability in radiologists' interpretations. To overcome this challenge, a hybrid 3D deep learning model for automated PAS detection from volumetric MRI scans is proposed in this study. The model integrates a 3D DenseNet121 to capture local features and a 3D Vision Transformer (ViT) to model global spatial context. It was developed and evaluated on a retrospective dataset of 1,133 MRI volumes. Multiple 3D deep learning architectures were also evaluated for comparison. On an independent test set, the DenseNet121-ViT model achieved the highest performance with a five-run average accuracy of 84.3%. These results highlight the strength of hybrid CNN-Transformer models as a computer-aided diagnosis tool. The model's performance demonstrates a clear potential to assist radiologists by providing a robust decision support to improve diagnostic consistency across interpretations, and ultimately enhance the accuracy and timeliness of PAS diagnosis. 

---
# AI Code in the Wild: Measuring Security Risks and Ecosystem Shifts of AI-Generated Code in Modern Software 

**Authors**: Bin Wang, Wenjie Yu, Yilu Zhong, Hao Yu, Keke Lian, Chaohua Lu, Hongfang Zheng, Dong Zhang, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.18567)  

**Abstract**: Large language models (LLMs) for code generation are becoming integral to modern software development, but their real-world prevalence and security impact remain poorly understood.
We present the first large-scale empirical study of AI-generated code (AIGCode) in the wild. We build a high-precision detection pipeline and a representative benchmark to distinguish AIGCode from human-written code, and apply them to (i) development commits from the top 1,000 GitHub repositories (2022-2025) and (ii) 7,000+ recent CVE-linked code changes. This lets us label commits, files, and functions along a human/AI axis and trace how AIGCode moves through projects and vulnerability life cycles.
Our measurements show three ecological patterns. First, AIGCode is already a substantial fraction of new code, but adoption is structured: AI concentrates in glue code, tests, refactoring, documentation, and other boilerplate, while core logic and security-critical configurations remain mostly human-written. Second, adoption has security consequences: some CWE families are overrepresented in AI-tagged code, and near-identical insecure templates recur across unrelated projects, suggesting "AI-induced vulnerabilities" propagated by shared models rather than shared maintainers. Third, in human-AI edit chains, AI introduces high-throughput changes while humans act as security gatekeepers; when review is shallow, AI-introduced defects persist longer, remain exposed on network-accessible surfaces, and spread to more files and repositories.
We will open-source the complete dataset and release analysis artifacts and fine-grained documentation of our methodology and findings. 

---
# Adaptive Accountability in Networked MAS: Tracing and Mitigating Emergent Norms at Scale 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2512.18561)  

**Abstract**: Large-scale networked multi-agent systems increasingly underpin critical infrastructure, yet their collective behavior can drift toward undesirable emergent norms that elude conventional governance mechanisms. We introduce an adaptive accountability framework that (i) continuously traces responsibility flows through a lifecycle-aware audit ledger, (ii) detects harmful emergent norms online via decentralized sequential hypothesis tests, and (iii) deploys local policy and reward-shaping interventions that realign agents with system-level objectives in near real time. We prove a bounded-compromise theorem showing that whenever the expected intervention cost exceeds an adversary's payoff, the long-run proportion of compromised interactions is bounded by a constant strictly less than one. Extensive high-performance simulations with up to 100 heterogeneous agents, partial observability, and stochastic communication graphs show that our framework prevents collusion and resource hoarding in at least 90% of configurations, boosts average collective reward by 12-18%, and lowers the Gini inequality index by up to 33% relative to a PPO baseline. These results demonstrate that a theoretically principled accountability layer can induce ethically aligned, self-regulating behavior in complex MAS without sacrificing performance or scalability. 

---
# Enhancing Medical Large Vision-Language Models via Alignment Distillation 

**Authors**: Aofei Chang, Ting Wang, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2512.18554)  

**Abstract**: Medical Large Vision-Language Models (Med-LVLMs) have shown promising results in clinical applications, but often suffer from hallucinated outputs due to misaligned visual understanding. In this work, we identify two fundamental limitations contributing to this issue: insufficient visual representation learning and poor visual attention alignment. To address these problems, we propose MEDALIGN, a simple, lightweight alignment distillation framework that transfers visual alignment knowledge from a domain-specific Contrastive Language-Image Pre-training (CLIP) model to Med-LVLMs. MEDALIGN introduces two distillation losses: a spatial-aware visual alignment loss based on visual token-level similarity structures, and an attention-aware distillation loss that guides attention toward diagnostically relevant regions. Extensive experiments on medical report generation and medical visual question answering (VQA) benchmarks show that MEDALIGN consistently improves both performance and interpretability, yielding more visually grounded outputs. 

---
# Toward Training Superintelligent Software Agents through Self-Play SWE-RL 

**Authors**: Yuxiang Wei, Zhiqing Sun, Emily McMilin, Jonas Gehring, David Zhang, Gabriel Synnaeve, Daniel Fried, Lingming Zhang, Sida Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18552)  

**Abstract**: While current software agents powered by large language models (LLMs) and agentic reinforcement learning (RL) can boost programmer productivity, their training data (e.g., GitHub issues and pull requests) and environments (e.g., pass-to-pass and fail-to-pass tests) heavily depend on human knowledge or curation, posing a fundamental barrier to superintelligence. In this paper, we present Self-play SWE-RL (SSR), a first step toward training paradigms for superintelligent software agents. Our approach takes minimal data assumptions, only requiring access to sandboxed repositories with source code and installed dependencies, with no need for human-labeled issues or tests. Grounded in these real-world codebases, a single LLM agent is trained via reinforcement learning in a self-play setting to iteratively inject and repair software bugs of increasing complexity, with each bug formally specified by a test patch rather than a natural language issue description. On the SWE-bench Verified and SWE-Bench Pro benchmarks, SSR achieves notable self-improvement (+10.4 and +7.8 points, respectively) and consistently outperforms the human-data baseline over the entire training trajectory, despite being evaluated on natural language issues absent from self-play. Our results, albeit early, suggest a path where agents autonomously gather extensive learning experiences from real-world software repositories, ultimately enabling superintelligent systems that exceed human capabilities in understanding how systems are constructed, solving novel challenges, and autonomously creating new software from scratch. 

---
# SecureCode v2.0: A Production-Grade Dataset for Training Security-Aware Code Generation Models 

**Authors**: Scott Thornton  

**Link**: [PDF](https://arxiv.org/pdf/2512.18542)  

**Abstract**: AI assistants produce vulnerable code in 45% of security-relevant scenarios, introducing flaws into production systems at scale. Yet existing secure coding datasets fall short. They lack incident grounding, don't provide the scale modern training requires, and miss the operational security context developers need for production deployments. We present SecureCode v2.0, a production-grade dataset of 1,215 security-focused coding examples that passed structural validation and expert security review. Every example ties to actual documented security incidents with CVE references, provides vulnerable and secure implementations, demonstrates concrete attacks, and includes defense-in-depth operational guidance. The dataset covers 11 vulnerability categories (complete OWASP Top 10:2025 plus AI/ML Security Threats) across 11 languages (Python, JavaScript, Java, Go, PHP, C#, TypeScript, Ruby, Rust, Kotlin, and YAML for infrastructure-as-code).
Our quality assurance framework ensures complete incident grounding. Each example includes SIEM integration strategies, infrastructure hardening recommendations (Docker, AppArmor, WAF configurations), and testing approaches using language-appropriate frameworks. The dataset uses a 4-turn conversational structure mirroring actual developer-AI interactions, escalating from basic implementations to advanced security considerations and defense-in-depth guidance.
Our contributions: (1) 1,215 rigorously validated examples split into 989 training, 122 validation, and 104 test sets, (2) an automated validation framework ensuring dataset consistency, (3) a 4-turn conversational structure capturing realistic security workflows, (4) comprehensive operational security guidance with SIEM integration strategies, (5) complete language-specific implementation fidelity, and (6) open-source release of data, validation tools, and benchmarking protocols. 

---
# Detection of AI Generated Images Using Combined Uncertainty Measures and Particle Swarm Optimised Rejection Mechanism 

**Authors**: Rahul Yumlembam, Biju Issac, Nauman Aslam, Eaby Kollonoor Babu, Josh Collyer, Fraser Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2512.18527)  

**Abstract**: As AI-generated images become increasingly photorealistic, distinguishing them from natural images poses a growing challenge. This paper presents a robust detection framework that leverages multiple uncertainty measures to decide whether to trust or reject a model's predictions. We focus on three complementary techniques: Fisher Information, which captures the sensitivity of model parameters to input variations; entropy-based uncertainty from Monte Carlo Dropout, which reflects predictive variability; and predictive variance from a Deep Kernel Learning framework using a Gaussian Process classifier. To integrate these diverse uncertainty signals, Particle Swarm Optimisation is used to learn optimal weightings and determine an adaptive rejection threshold. The model is trained on Stable Diffusion-generated images and evaluated on GLIDE, VQDM, Midjourney, BigGAN, and StyleGAN3, each introducing significant distribution shifts. While standard metrics such as prediction probability and Fisher-based measures perform well in distribution, their effectiveness degrades under shift. In contrast, the Combined Uncertainty measure consistently achieves an incorrect rejection rate of approximately 70 percent on unseen generators, successfully filtering most misclassified AI samples. Although the system occasionally rejects correct predictions from newer generators, this conservative behaviour is acceptable, as rejected samples can support retraining. The framework maintains high acceptance of accurate predictions for natural images and in-domain AI data. Under adversarial attacks using FGSM and PGD, the Combined Uncertainty method rejects around 61 percent of successful attacks, while GP-based uncertainty alone achieves up to 80 percent. Overall, the results demonstrate that multi-source uncertainty fusion provides a resilient and adaptive solution for AI-generated image detection. 

---
# A Formal Descriptive Language for Learning Dynamics: A Five-Layer Structural Coordinate System 

**Authors**: Miyuki T. Nakata  

**Link**: [PDF](https://arxiv.org/pdf/2512.18525)  

**Abstract**: Understanding learning as a dynamic process is challenging due to the interaction of multiple factors, including cognitive load, internal state change, and subjective evaluation. Existing approaches often address these elements in isolation, limiting the ability to describe learning phenomena within a unified and structurally explicit framework. This paper proposes a multi-layer formal descriptive framework for learning dynamics. Rather than offering a predictive or prescriptive model, the framework introduces a symbolic language composed of state variables, mappings, and layer-specific responsibilities, enabling consistent description of learning processes without commitment to specific functional forms or optimization objectives. This descriptive framework is intended to serve as a structural substrate for analyzing learning processes in human learners, and by extension, in adaptive and Al-assisted learning systems. A central design principle is the explicit separation of descriptive responsibilities across layers, distinguishing load generation, internal understanding transformation, observation, and evaluation. Within this structure, cognitive load is treated as a relational quantity arising from interactions between external input and internal organization, while subjective evaluation is modeled as a minimal regulatory interface responding to learning dynamics and environmental conditions. By emphasizing descriptive clarity and extensibility, the framework provides a common language for organizing existing theories and supporting future empirical and theoretical work. 

---
# Prediction and Forecast of Short-Term Drought Impacts Using Machine Learning to Support Mitigation and Adaptation Efforts 

**Authors**: Hatim M. E. Geli, Islam Omar, Mona Y. Elshinawy, David W. DuBios, Lara Prehodko, Kelly H Smith, Abdel-Hameed A. Badawy  

**Link**: [PDF](https://arxiv.org/pdf/2512.18522)  

**Abstract**: Drought is a complex natural hazard that affects ecological and human systems, often resulting in substantial environmental and economic losses. Recent increases in drought severity, frequency, and duration underscore the need for effective monitoring and mitigation strategies. Predicting drought impacts rather than drought conditions alone offers opportunities to support early warning systems and proactive decision-making. This study applies machine learning techniques to link drought indices with historical drought impact records (2005:2024) to generate short-term impact forecasts. By addressing key conceptual and data-driven challenges regarding temporal scale and impact quantification, the study aims to improve the predictability of drought impacts at actionable lead times. The Drought Severity and Coverage Index (DSCI) and the Evaporative Stress Index (ESI) were combined with impact data from the Drought Impact Reporter (DIR) to model and forecast weekly drought impacts. Results indicate that Fire and Relief impacts were predicted with the highest accuracy, followed by Agriculture and Water, while forecasts for Plants and Society impacts showed greater variability. County and state level forecasts for New Mexico were produced using an eXtreme Gradient Boosting (XGBoost) model that incorporated both DSCI and ESI. The model successfully generated forecasts up to eight weeks in advance using the preceding eight weeks of data for most impact categories. This work supports the development of an Ecological Drought Information Communication System (EcoDri) for New Mexico and demonstrates the potential for broader application in similar drought-prone regions. The findings can aid stakeholders, land managers, and decision-makers in developing and implementing more effective drought mitigation and adaptation strategies. 

---
# The Illusion of Consistency: Selection-Induced Bias in Gated Kalman Innovation Statistics 

**Authors**: Barak Or  

**Link**: [PDF](https://arxiv.org/pdf/2512.18508)  

**Abstract**: Validation gating is a fundamental component of classical Kalman-based tracking systems. Only measurements whose normalized innovation squared (NIS) falls below a prescribed threshold are considered for state update. While this procedure is statistically motivated by the chi-square distribution, it implicitly replaces the unconditional innovation process with a conditionally observed one, restricted to the validation event. This paper shows that innovation statistics computed after gating converge to gate-conditioned rather than nominal quantities. Under classical linear--Gaussian assumptions, we derive exact expressions for the first- and second-order moments of the innovation conditioned on ellipsoidal gating, and show that gating induces a deterministic, dimension-dependent contraction of the innovation covariance. The analysis is extended to NN association, which is shown to act as an additional statistical selection operator. We prove that selecting the minimum-norm innovation among multiple in-gate measurements introduces an unavoidable energy contraction, implying that nominal innovation statistics cannot be preserved under nontrivial gating and association. Closed-form results in the two-dimensional case quantify the combined effects and illustrate their practical significance. 

---
# GTMA: Dynamic Representation Optimization for OOD Vision-Language Models 

**Authors**: Jensen Zhang, Ningyuan Liu, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18504)  

**Abstract**: Vision-language models (VLMs) struggle in open-world applications, where out-of-distribution (OOD) concepts often trigger cross-modal alignment collapse and severely degrade zero-shot performance. We identify the root cause as modal asymmetry: while the visual encoder can extract discriminative features from unseen images, the text encoder is constrained by a fixed discrete vocabulary and cannot synthesize new semantic anchors. Existing approaches such as CoOp or LoRA provide only partial remedies, as they remain confined to the pre-trained semantic space.
To overcome this bottleneck, we propose dynamic representation optimization, realized through the Guided Target-Matching Adaptation (GTMA) framework. At inference time, GTMA constructs a continuous pseudo-word embedding that best aligns with an OOD image's visual anchor, effectively bypassing vocabulary limitations. The optimization is driven by an adaptive gradient-based representation policy optimization algorithm, which incorporates semantic regularization to preserve plausibility and compatibility with the model's prior knowledge.
Experiments on ImageNet-R and the VISTA-Beyond benchmark demonstrate that GTMA improves zero-shot and few-shot OOD accuracy by up to 15-20 percent over the base VLM while maintaining performance on in-distribution concepts. Ablation studies further confirm the necessity of pseudo-word optimization. 

---
# PlantDiseaseNet-RT50: A Fine-tuned ResNet50 Architecture for High-Accuracy Plant Disease Detection Beyond Standard CNNs 

**Authors**: Santwana Sagnika, Manav Malhotra, Ishtaj Kaur Deol, Soumyajit Roy, Swarnav Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.18500)  

**Abstract**: Plant diseases pose a significant threat to agricultural productivity and global food security, accounting for 70-80% of crop losses worldwide. Traditional detection methods rely heavily on expert visual inspection, which is time-consuming, labour-intensive, and often impractical for large-scale farming operations. In this paper, we present PlantDiseaseNet-RT50, a novel fine-tuned deep learning architecture based on ResNet50 for automated plant disease detection. Our model features strategically unfrozen layers, a custom classification head with regularization mechanisms, and dynamic learning rate scheduling through cosine decay. Using a comprehensive dataset of distinct plant disease categories across multiple crop species, PlantDiseaseNet-RT50 achieves exceptional performance with approximately 98% accuracy, precision, and recall. Our architectural modifications and optimization protocol demonstrate how targeted fine-tuning can transform a standard pretrained model into a specialized agricultural diagnostic tool. We provide a detailed account of our methodology, including the systematic unfreezing of terminal layers, implementation of batch normalization and dropout regularization and application of advanced training techniques. PlantDiseaseNet-RT50 represents a significant advancement in AI-driven agricultural tools, offering a computationally efficient solution for rapid and accurate plant disease diagnosis that can be readily implemented in practical farming contexts to support timely interventions and reduce crop losses. 

---
# Enhancing Decision-Making in Windows PE Malware Classification During Dataset Shifts with Uncertainty Estimation 

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob  

**Link**: [PDF](https://arxiv.org/pdf/2512.18495)  

**Abstract**: Artificial intelligence techniques have achieved strong performance in classifying Windows Portable Executable (PE) malware, but their reliability often degrades under dataset shifts, leading to misclassifications with severe security consequences. To address this, we enhance an existing LightGBM (LGBM) malware detector by integrating Neural Networks (NN), PriorNet, and Neural Network Ensembles, evaluated across three benchmark datasets: EMBER, BODMAS, and UCSB. The UCSB dataset, composed mainly of packed malware, introduces a substantial distributional shift relative to EMBER and BODMAS, making it a challenging testbed for robustness. We study uncertainty-aware decision strategies, including probability thresholding, PriorNet, ensemble-derived estimates, and Inductive Conformal Evaluation (ICE). Our main contribution is the use of ensemble-based uncertainty estimates as Non-Conformity Measures within ICE, combined with a novel threshold optimisation method. On the UCSB dataset, where the shift is most severe, the state-of-the-art probability-based ICE (SOTA) yields an incorrect acceptance rate (IA%) of 22.8%. In contrast, our method reduces this to 16% a relative reduction of about 30% while maintaining competitive correct acceptance rates (CA%). These results demonstrate that integrating ensemble-based uncertainty with conformal prediction provides a more reliable safeguard against misclassifications under extreme dataset shifts, particularly in the presence of packed malware, thereby offering practical benefits for real-world security operations. 

---
# SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios 

**Authors**: Minh V. T. Thai, Tue Le, Dung Nguyen Manh, Huy Phan Nhat, Nghi D. Q. Bui  

**Link**: [PDF](https://arxiv.org/pdf/2512.18470)  

**Abstract**: Existing benchmarks for AI coding agents focus on isolated, single-issue tasks such as fixing a bug or implementing a small feature. However, real-world software engineering is fundamentally a long-horizon endeavor: developers must interpret high-level requirements, plan coordinated changes across many files, and evolve codebases over multiple iterations while preserving existing functionality. We introduce SWE-EVO, a benchmark that evaluates agents on this long-horizon software evolution challenge. Constructed from release notes and version histories of seven mature open-source Python projects, Tool comprises 48 evolution tasks that require agents to implement multi-step modifications spanning an average of 21 files, validated against comprehensive test suites averaging 874 tests per instance. Experiments with state-of-the-art models reveal a striking capability gap: even GPT-5 with OpenHands achieves only a 21 percent resolution rate on Tool, compared to 65 percent on the single-issue SWE-Bench Verified. This demonstrates that current agents struggle with sustained, multi-file reasoning. We also propose Fix Rate, a fine-grained metric that captures partial progress toward solving these complex, long-horizon tasks. 

---
# Self-organizing maps for water quality assessment in reservoirs and lakes: A systematic literature review 

**Authors**: Oraib Almegdadi, João Marcelino, Sarah Fakhreddine, João Manso, Nuno C. Marques  

**Link**: [PDF](https://arxiv.org/pdf/2512.18466)  

**Abstract**: Sustainable water quality underpins ecological balance and water security. Assessing and managing lakes and reservoirs is difficult due to data sparsity, heterogeneity, and nonlinear relationships among parameters. This review examines how Self-Organizing Map (SOM), an unsupervised AI technique, is applied to water quality assessment. It synthesizes research on parameter selection, spatial and temporal sampling strategies, and clustering approaches. Emphasis is placed on how SOM handles multidimensional data and uncovers hidden patterns to support effective water management. The growing availability of environmental data from in-situ sensors, remote sensing imagery, IoT technologies, and historical records has significantly expanded analytical opportunities in environmental monitoring. SOM has proven effective in analysing complex datasets, particularly when labelled data are limited or unavailable. It enables high-dimensional data visualization, facilitates the detection of hidden ecological patterns, and identifies critical correlations among diverse water quality indicators. This review highlights SOMs versatility in ecological assessments, trophic state classification, algal bloom monitoring, and catchment area impact evaluations. The findings offer comprehensive insights into existing methodologies, supporting future research and practical applications aimed at improving the monitoring and sustainable management of lake and reservoir ecosystems. 

---
# Mitigating Spurious Correlations in NLI via LLM-Synthesized Counterfactuals and Dynamic Balanced Sampling 

**Authors**: Christopher Román Jaimes  

**Link**: [PDF](https://arxiv.org/pdf/2512.18462)  

**Abstract**: Natural Language Inference (NLI) models frequently rely on spurious correlations rather than semantic reasoning. Existing mitigation strategies often incur high annotation costs or trigger catastrophic forgetting during fine-tuning. We propose an automated, scalable pipeline to address these limitations. First, we introduce Log-Frequency LMI (LF-LMI) to accurately detect semantic artifacts. Second, we generate a high-quality synthetic contrast set via an LLM-synthesis pipeline with multi-judge verification. Finally, we introduce Dynamic Balanced Sampling, a training strategy that rotates the original data distribution to prevent forgetting. Our method improves consistency on a challenging benchmark from 63.5% to 81.0% while maintaining 88.4% in-domain accuracy, significantly outperforming naive fine-tuning. 

---
# SoK: Understanding (New) Security Issues Across AI4Code Use Cases 

**Authors**: Qilong Wu, Taoran Li, Tianyang Zhou, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2512.18456)  

**Abstract**: AI-for-Code (AI4Code) systems are reshaping software engineering, with tools like GitHub Copilot accelerating code generation, translation, and vulnerability detection. Alongside these advances, however, security risks remain pervasive: insecure outputs, biased benchmarks, and susceptibility to adversarial manipulation undermine their reliability. This SoK surveys the landscape of AI4Code security across three core applications, identifying recurring gaps: benchmark dominance by Python and toy problems, lack of standardized security datasets, data leakage in evaluation, and fragile adversarial robustness. A comparative study of six state-of-the-art models illustrates these challenges: insecure patterns persist in code generation, vulnerability detection is brittle to semantic-preserving attacks, fine-tuning often misaligns security objectives, and code translation yields uneven security benefits. From this analysis, we distill three forward paths: embedding secure-by-default practices in code generation, building robust and comprehensive detection benchmarks, and leveraging translation as a route to security-enhanced languages. We call for a shift toward security-first AI4Code, where vulnerability mitigation and robustness are embedded throughout the development life cycle. 

---
# Secret mixtures of experts inside your LLM 

**Authors**: Enric Boix-Adsera  

**Link**: [PDF](https://arxiv.org/pdf/2512.18452)  

**Abstract**: Despite being one of the earliest neural network layers, the Multilayer Perceptron (MLP) is arguably one of the least understood parts of the transformer architecture due to its dense computation and lack of easy visualization. This paper seeks to understand the MLP layers in dense LLM models by hypothesizing that these layers secretly approximately perform a sparse computation -- namely, that they can be well approximated by sparsely-activating Mixture of Experts (MoE) layers.
Our hypothesis is based on a novel theoretical connection between MoE models and Sparse Autoencoder (SAE) structure in activation space. We empirically validate the hypothesis on pretrained LLMs, and demonstrate that the activation distribution matters -- these results do not hold for Gaussian data, but rather rely crucially on structure in the distribution of neural network activations.
Our results shine light on a general principle at play in MLP layers inside LLMs, and give an explanation for the effectiveness of modern MoE-based transformers. Additionally, our experimental explorations suggest new directions for more efficient MoE architecture design based on low-rank routers. 

---
# Snowveil: A Framework for Decentralised Preference Discovery 

**Authors**: Grammateia Kotsialou  

**Link**: [PDF](https://arxiv.org/pdf/2512.18444)  

**Abstract**: Aggregating subjective preferences of a large group is a fundamental challenge in computational social choice, traditionally reliant on central authorities. To address the limitations of this model, this paper introduces Decentralised Preference Discovery (DPD), the problem of determining the collective will of an electorate under constraints of censorship resistance, partial information, and asynchronous communication. We propose Snowveil, a novel framework for this task. Snowveil uses an iterative, gossip-based protocol where voters repeatedly sample the preferences of a small, random subset of the electorate to progressively converge on a collective outcome. We demonstrate the framework's modularity by designing the Constrained Hybrid Borda (CHB), a novel aggregation rule engineered to balance broad consensus with strong plurality support, and provide a rigorous axiomatic analysis of its properties. By applying a potential function and submartingale theory, we develop a multi-level analytical method to show that the system almost surely converges to a stable, single-winner in finite time, a process that can then be iterated to construct a set of winning candidates for multi-winner scenarios. This technique is largely agnostic to the specific aggregation rule, requiring only that it satisfies core social choice axioms like Positive Responsiveness, thus offering a formal toolkit for a wider class of DPD protocols. Furthermore, we present a comprehensive empirical analysis through extensive simulation, validating Snowveil's $O(n)$ scalability. Overall, this work advances the understanding of how a stable consensus can emerge from subjective, complex, and diverse preferences in decentralised systems for large electorates. 

---
# A Distributed Hierarchical Spatio-Temporal Edge-Enhanced Graph Neural Network for City-Scale Dynamic Logistics Routing 

**Authors**: Zihan Han, Lingran Meng, Jingwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18441)  

**Abstract**: City-scale logistics routing has become increasingly challenging as metropolitan road networks grow to tens of millions of edges and traffic conditions evolve rapidly under high-volume mobility demands. Conventional centralized routing algorithms and monolithic graph neural network (GNN) models suffer from limited scalability, high latency, and poor real-time adaptability, which restricts their effectiveness in large urban logistics systems. To address these challenges, this paper proposes a Distributed Hierarchical Spatio-Temporal Edge-Enhanced Graph Neural Network (HSTE-GNN) for dynamic routing over ultra-large road networks. The framework partitions the city-scale graph into regional subgraphs processed in parallel across distributed computing nodes, enabling efficient learning of localized traffic dynamics. Within each region, an edge-enhanced spatio-temporal module jointly models node states, dynamic edge attributes, and short-term temporal dependencies. A hierarchical coordination layer further aggregates cross-region representations through an asynchronous parameter-server mechanism, ensuring global routing coherence under high-frequency traffic updates. This distributed hierarchical design balances local responsiveness with global consistency, significantly improving scalability and inference efficiency. Experiments on real-world large-scale traffic datasets from Beijing and New York demonstrate that HSTE-GNN outperforms strong spatio-temporal baselines such as ST-GRAPH, achieving 34.9% lower routing delay, 14.7% lower MAPE, and 11.8% lower RMSE, while improving global route consistency by 7.3%. These results confirm that the proposed framework provides a scalable, adaptive, and efficient solution for next-generation intelligent transportation systems and large-scale logistics platforms. 

---
# An Agentic AI Framework for Training General Practitioner Student Skills 

**Authors**: Victor De Marez, Jens Van Nooten, Luna De Bruyne, Walter Daelemans  

**Link**: [PDF](https://arxiv.org/pdf/2512.18440)  

**Abstract**: Advancements in large language models offer strong potential for enhancing virtual simulated patients (VSPs) in medical education by providing scalable alternatives to resource-intensive traditional methods. However, current VSPs often struggle with medical accuracy, consistent roleplaying, scenario generation for VSP use, and educationally structured feedback. We introduce an agentic framework for training general practitioner student skills that unifies (i) configurable, evidence-based vignette generation, (ii) controlled persona-driven patient dialogue with optional retrieval grounding, and (iii) standards-based assessment and feedback for both communication and clinical reasoning. We instantiate the framework in an interactive spoken consultation setting and evaluate it with medical students ($\mathbf{N{=}14}$). Participants reported realistic and vignette-faithful dialogue, appropriate difficulty calibration, a stable personality signal, and highly useful example-rich feedback, alongside excellent overall usability. These results support agentic separation of scenario control, interaction control, and standards-based assessment as a practical pattern for building dependable and pedagogically valuable VSP training tools. 

---
# MeniMV: A Multi-view Benchmark for Meniscus Injury Severity Grading 

**Authors**: Shurui Xu, Siqi Yang, Jiapin Ren, Zhong Cao, Hongwei Yang, Mengzhen Fan, Yuyu Sun, Shuyan Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.18437)  

**Abstract**: Precise grading of meniscal horn tears is critical in knee injury diagnosis but remains underexplored in automated MRI analysis. Existing methods often rely on coarse study-level labels or binary classification, lacking localization and severity information. In this paper, we introduce MeniMV, a multi-view benchmark dataset specifically designed for horn-specific meniscus injury grading. MeniMV comprises 3,000 annotated knee MRI exams from 750 patients across three medical centers, providing 6,000 co-registered sagittal and coronal images. Each exam is meticulously annotated with four-tier (grade 0-3) severity labels for both anterior and posterior meniscal horns, verified by chief orthopedic physicians. Notably, MeniMV offers more than double the pathology-labeled data volume of prior datasets while uniquely capturing the dual-view diagnostic context essential in clinical practice. To demonstrate the utility of MeniMV, we benchmark multiple state-of-the-art CNN and Transformer-based models. Our extensive experiments establish strong baselines and highlight challenges in severity grading, providing a valuable foundation for future research in automated musculoskeletal imaging. 

---
# VeruSAGE: A Study of Agent-Based Verification for Rust Systems 

**Authors**: Chenyuan Yang, Natalie Neamtu, Chris Hawblitzel, Jacob R. Lorch, Shan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18436)  

**Abstract**: Large language models (LLMs) have shown impressive capability to understand and develop code. However, their capability to rigorously reason about and prove code correctness remains in question. This paper offers a comprehensive study of LLMs' capability to develop correctness proofs for system software written in Rust. We curate a new system-verification benchmark suite, VeruSAGE-Bench, which consists of 849 proof tasks extracted from eight open-source Verus-verified Rust systems. Furthermore, we design different agent systems to match the strengths and weaknesses of different LLMs (o4-mini, GPT-5, Sonnet 4, and Sonnet 4.5). Our study shows that different tools and agent settings are needed to stimulate the system-verification capability of different types of LLMs. The best LLM-agent combination in our study completes over 80% of system-verification tasks in VeruSAGE-Bench. It also completes over 90% of a set of system proof tasks not part of VeruSAGE-Bench because they had not yet been finished by human experts. This result shows the great potential for LLM-assisted development of verified system software. 

---
# Federated Learning Based Decentralized Adaptive Intelligent Transmission Protocol for Privacy Preserving 6G Networks 

**Authors**: Ansar Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2512.18432)  

**Abstract**: The move to 6th Generation (6G) wireless networks creates new issues with privacy, scalability, and adaptability. The data-intensive nature of 6G is not handled well by older, centralized network models. A shift toward more secure and decentralized systems is therefore required. A new framework called the Federated Learning-based Decentralized Adaptive Intelligent Transmission Protocol (AITP) is proposed to meet these challenges. The AITP uses the distributed learning of Federated Learning (FL) within a decentralized system. Transmission parameters can be adjusted intelligently in real time. User privacy is maintained by keeping raw data on local edge devices. The protocol's performance was evaluated with mathematical modeling and detailed simulations. It was shown to be superior to traditional non-adaptive and centralized AI methods across several key metrics. These included latency, network throughput, energy efficiency, and robustness. The AITP is presented as a foundational technology for future 6G networks that supports a user-centric, privacy-first design. This study is a step forward for privacy-preserving research in 6G. 

---
# AmPLe: Supporting Vision-Language Models via Adaptive-Debiased Ensemble Multi-Prompt Learning 

**Authors**: Fei Song, Yi Li, Jiangmeng Li, Rui Wang, Changwen Zheng, Fanjiang Xu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2512.18411)  

**Abstract**: Multi-prompt learning methods have emerged as an effective approach for facilitating the rapid adaptation of vision-language models to downstream tasks with limited resources. Existing multi-prompt learning methods primarily focus on utilizing various meticulously designed prompts within a single foundation vision-language model to achieve superior performance. However, the overlooked model-prompt matching bias hinders the development of multi-prompt learning, i.e., the same prompt can convey different semantics across distinct vision-language models, such as CLIP-ViT-B/16 and CLIP-ViT-B/32, resulting in inconsistent predictions of identical prompt. To mitigate the impact of this bias on downstream tasks, we explore an ensemble learning approach to sufficiently aggregate the benefits of diverse predictions. Additionally, we further disclose the presence of sample-prompt matching bias, which originates from the prompt-irrelevant semantics encapsulated in the input samples. Thus, directly utilizing all information from the input samples for generating weights of ensemble learning can lead to suboptimal performance. In response, we extract prompt-relevant semantics from input samples by leveraging the guidance of the information theory-based analysis, adaptively calculating debiased ensemble weights. Overall, we propose Adaptive-Debiased Ensemble MultiPrompt Learning, abbreviated as AmPLe, to mitigate the two types of bias simultaneously. Extensive experiments on three representative tasks, i.e., generalization to novel classes, new target datasets, and unseen domain shifts, show that AmPLe can widely outperform existing methods. Theoretical validation from a causal perspective further supports the effectiveness of AmPLe. 

---
# AraToken: Optimizing Arabic Tokenization with Normalization Pipeline and Language Extension for Qwen3 

**Authors**: Mark Kashirskiy, Artiom Lipinski, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2512.18399)  

**Abstract**: Tokenization is a critical preprocessing step for large language models (LLMs), directly impacting training efficiency and downstream performance. General-purpose tokenizers trained predominantly on English and Latin-script languages exhibit suboptimal performance on morphologically rich languages such as Arabic, resulting in inflated token sequences and reduced compression efficiency. In this work, we present AraToken, an Arabic-optimized tokenizer built on SentencePiece Unigram algorithm with a comprehensive normalization pipeline addressing Arabic-specific orthographic variations including Alif variants, diacritics, and Arabic-Indic numerals. We systematically compare BPE, WordPiece, and SentencePiece algorithms across multiple configurations, demonstrating that SentencePiece with normalization achieves 18% lower fertility (1.199 vs 1.35 tokens/word) compared to unnormalized baselines. Furthermore, we introduce the Language Extension Pipeline (LEP), a method for integrating the optimized tokenizer into Qwen3-0.6B through vocabulary extension with mean subtoken initialization and selective transformer layer unfreezing. Our experiments show that LEP reduces evaluation loss from 8.28 to 2.43 within 800 training steps on 100K Arabic samples. We release our tokenizer, training scripts, and model checkpoints to facilitate Arabic NLP research. 

---
# Neural Proofs for Sound Verification and Control of Complex Systems 

**Authors**: Alessandro Abate  

**Link**: [PDF](https://arxiv.org/pdf/2512.18389)  

**Abstract**: This informal contribution presents an ongoing line of research that is pursuing a new approach to the construction of sound proofs for the formal verification and control of complex stochastic models of dynamical systems, of reactive programs and, more generally, of models of Cyber-Physical Systems. Neural proofs are made up of two key components: 1) proof rules encode requirements entailing the verification of general temporal specifications over the models of interest; and 2) certificates that discharge such rules, namely they are constructed from said proof rules with an inductive (that is, cyclic, repetitive) approach; this inductive approach involves: 2a) accessing samples from the model's dynamics and accordingly training neural networks, whilst 2b) generalising such networks via SAT-modulo-theory (SMT) queries that leverage the full knowledge of the models. In the context of sequential decision making problems over complex stochastic models, it is possible to additionally generate provably-correct policies/strategies/controllers, namely state-feedback functions that, in conjunction with neural certificates, formally attain the given specifications for the models of interest. 

---
# Exploration vs. Fixation: Scaffolding Divergent and Convergent Thinking for Human-AI Co-Creation with Generative Models 

**Authors**: Chao Wen, Tung Phung, Pronita Mehrotra, Sumit Gulwani, Tomohiro Nagashima, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2512.18388)  

**Abstract**: Generative AI has begun to democratize creative work, enabling novices to produce complex artifacts such as code, images, and videos. However, in practice, existing interaction paradigms often fail to support divergent exploration: users tend to converge too quickly on early ``good enough'' results and struggle to move beyond them, leading to premature convergence and design fixation that constrains their creative potential. To address this, we propose a structured, process-oriented human-AI co-creation paradigm including divergent and convergent thinking stages, grounded in Wallas's model of creativity. To avoid design fixation, our paradigm scaffolds both high-level exploration of conceptual ideas in the early divergent thinking phase and low-level exploration of variations in the later convergent thinking phrase. We instantiate this paradigm in HAIExplore, an image co-creation system that (i) scaffolds divergent thinking through a dedicated brainstorming stage for exploring high-level ideas in a conceptual space, and (ii) scaffolds convergent refinement through an interface that externalizes users' refinement intentions as interpretable parameters and options, making the refinement process more controllable and easier to explore. We report on a within-subjects study comparing HAIExplore with a widely used linear chat interface (ChatGPT) for creative image generation. Our findings show that explicitly scaffolding the creative process into brainstorming and refinement stages can mitigate design fixation, improve perceived controllability and alignment with users' intentions, and better support the non-linear nature of creative work. We conclude with design implications for future creativity support tools and human-AI co-creation workflows. 

---
# Datasets for machine learning and for assessing the intelligence level of automatic patent search systems 

**Authors**: Boris Genin, Alexander Gorbunov, Dmitry Zolkin, Igor Nekrasov  

**Link**: [PDF](https://arxiv.org/pdf/2512.18384)  

**Abstract**: The key to success in automating prior art search in patent research using artificial intelligence lies in developing large datasets for machine learning and ensuring their availability. This work is dedicated to providing a comprehensive solution to the problem of creating infrastructure for research in this field, including datasets and tools for calculating search quality criteria. The paper discusses the concept of semantic clusters of patent documents that determine the state of the art in a given subject, as proposed by the authors. A definition of such semantic clusters is also provided. Prior art search is presented as the task of identifying elements within a semantic cluster of patent documents in the subject area specified by the document under consideration. A generator of user-configurable datasets for machine learning, based on collections of U.S. and Russian patent documents, is described. The dataset generator creates a database of links to documents in semantic clusters. Then, based on user-defined parameters, it forms a dataset of semantic clusters in JSON format for machine learning. To evaluate machine learning outcomes, it is proposed to calculate search quality scores that account for semantic clusters of the documents being searched. To automate the evaluation process, the paper describes a utility developed by the authors for assessing the quality of prior art document search. 

---
# LLM Agents Implement an NLG System from Scratch: Building Interpretable Rule-Based RDF-to-Text Generators 

**Authors**: Mateusz Lango, Ondřej Dušek  

**Link**: [PDF](https://arxiv.org/pdf/2512.18360)  

**Abstract**: We present a novel neurosymbolic framework for RDF-to-text generation, in which the model is "trained" through collaborative interactions among multiple LLM agents rather than traditional backpropagation. The LLM agents produce rule-based Python code for a generator for the given domain, based on RDF triples only, with no in-domain human reference texts. The resulting system is fully interpretable, requires no supervised training data, and generates text nearly instantaneously using only a single CPU. Our experiments on the WebNLG and OpenDialKG data show that outputs produced by our approach reduce hallucination, with only slight fluency penalties compared to finetuned or prompted language models 

---
# LLM-based Few-Shot Early Rumor Detection with Imitation Agent 

**Authors**: Fengzhu Zeng, Qian Shao, Ling Cheng, Wei Gao, Shih-Fen Cheng, Jing Ma, Cheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18352)  

**Abstract**: Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness. 

---
# MCVI-SANet: A lightweight semi-supervised model for LAI and SPAD estimation of winter wheat under vegetation index saturation 

**Authors**: Zhiheng Zhang, Jiajun Yang, Hong Sun, Dong Wang, Honghua Jiang, Yaru Chen, Tangyuan Ning  

**Link**: [PDF](https://arxiv.org/pdf/2512.18344)  

**Abstract**: Vegetation index (VI) saturation during the dense canopy stage and limited ground-truth annotations of winter wheat constrain accurate estimation of LAI and SPAD. Existing VI-based and texture-driven machine learning methods exhibit limited feature expressiveness. In addition, deep learning baselines suffer from domain gaps and high data demands, which restrict their generalization. Therefore, this study proposes the Multi-Channel Vegetation Indices Saturation Aware Net (MCVI-SANet), a lightweight semi-supervised vision model. The model incorporates a newly designed Vegetation Index Saturation-Aware Block (VI-SABlock) for adaptive channel-spatial feature enhancement. It also integrates a VICReg-based semi-supervised strategy to further improve generalization. Datasets were partitioned using a vegetation height-informed strategy to maintain representativeness across growth stages. Experiments over 10 repeated runs demonstrate that MCVI-SANet achieves state-of-the-art accuracy. The model attains an average R2 of 0.8123 and RMSE of 0.4796 for LAI, and an average R2 of 0.6846 and RMSE of 2.4222 for SPAD. This performance surpasses the best-performing baselines, with improvements of 8.95% in average LAI R2 and 8.17% in average SPAD R2. Moreover, MCVI-SANet maintains high inference speed with only 0.10M parameters. Overall, the integration of semi-supervised learning with agronomic priors provides a promising approach for enhancing remote sensing-based precision agriculture. 

---
# Dynamic Entropy Tuning in Reinforcement Learning Low-Level Quadcopter Control: Stochasticity vs Determinism 

**Authors**: Youssef Mahran, Zeyad Gamal, Ayman El-Badawy  

**Link**: [PDF](https://arxiv.org/pdf/2512.18336)  

**Abstract**: This paper explores the impact of dynamic entropy tuning in Reinforcement Learning (RL) algorithms that train a stochastic policy. Its performance is compared against algorithms that train a deterministic one. Stochastic policies optimize a probability distribution over actions to maximize rewards, while deterministic policies select a single deterministic action per state. The effect of training a stochastic policy with both static entropy and dynamic entropy and then executing deterministic actions to control the quadcopter is explored. It is then compared against training a deterministic policy and executing deterministic actions. For the purpose of this research, the Soft Actor-Critic (SAC) algorithm was chosen for the stochastic algorithm while the Twin Delayed Deep Deterministic Policy Gradient (TD3) was chosen for the deterministic algorithm. The training and simulation results show the positive effect the dynamic entropy tuning has on controlling the quadcopter by preventing catastrophic forgetting and improving exploration efficiency. 

---
# Reinforcement Learning Position Control of a Quadrotor Using Soft Actor-Critic (SAC) 

**Authors**: Youssef Mahran, Zeyad Gamal, Ayman El-Badawy  

**Link**: [PDF](https://arxiv.org/pdf/2512.18333)  

**Abstract**: This paper proposes a new Reinforcement Learning (RL) based control architecture for quadrotors. With the literature focusing on controlling the four rotors' RPMs directly, this paper aims to control the quadrotor's thrust vector. The RL agent computes the percentage of overall thrust along the quadrotor's z-axis along with the desired Roll ($\phi$) and Pitch ($\theta$) angles. The agent then sends the calculated control signals along with the current quadrotor's Yaw angle ($\psi$) to an attitude PID controller. The PID controller then maps the control signals to motor RPMs. The Soft Actor-Critic algorithm, a model-free off-policy stochastic RL algorithm, was used to train the RL agents. Training results show the faster training time of the proposed thrust vector controller in comparison to the conventional RPM controllers. Simulation results show smoother and more accurate path-following for the proposed thrust vector controller. 

---
# Asynchronous Pipeline Parallelism for Real-Time Multilingual Lip Synchronization in Video Communication Systems 

**Authors**: Eren Caglar, Amirkia Rafiei Oskooei, Mehmet Kutanoglu, Mustafa Keles, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2512.18318)  

**Abstract**: This paper introduces a parallel and asynchronous Transformer framework designed for efficient and accurate multilingual lip synchronization in real-time video conferencing systems. The proposed architecture integrates translation, speech processing, and lip-synchronization modules within a pipeline-parallel design that enables concurrent module execution through message-queue-based decoupling, reducing end-to-end latency by up to 3.1 times compared to sequential approaches. To enhance computational efficiency and throughput, the inference workflow of each module is optimized through low-level graph compilation, mixed-precision quantization, and hardware-accelerated kernel fusion. These optimizations provide substantial gains in efficiency while preserving model accuracy and visual quality. In addition, a context-adaptive silence-detection component segments the input speech stream at semantically coherent boundaries, improving translation consistency and temporal alignment across languages. Experimental results demonstrate that the proposed parallel architecture outperforms conventional sequential pipelines in processing speed, synchronization stability, and resource utilization. The modular, message-oriented design makes this work applicable to resource-constrained IoT communication scenarios including telemedicine, multilingual kiosks, and remote assistance systems. Overall, this work advances the development of low-latency, resource-efficient multimodal communication frameworks for next-generation AIoT systems. 

---
# Trustworthy and Explainable Deep Reinforcement Learning for Safe and Energy-Efficient Process Control: A Use Case in Industrial Compressed Air Systems 

**Authors**: Vincent Bezold, Patrick Wagner, Jakob Hofmann, Marco Huber, Alexander Sauer  

**Link**: [PDF](https://arxiv.org/pdf/2512.18317)  

**Abstract**: This paper presents a trustworthy reinforcement learning approach for the control of industrial compressed air systems. We develop a framework that enables safe and energy-efficient operation under realistic boundary conditions and introduce a multi-level explainability pipeline combining input perturbation tests, gradient-based sensitivity analysis, and SHAP (SHapley Additive exPlanations) feature attribution. An empirical evaluation across multiple compressor configurations shows that the learned policy is physically plausible, anticipates future demand, and consistently respects system boundaries. Compared to the installed industrial controller, the proposed approach reduces unnecessary overpressure and achieves energy savings of approximately 4\,\% without relying on explicit physics models. The results further indicate that system pressure and forecast information dominate policy decisions, while compressor-level inputs play a secondary role. Overall, the combination of efficiency gains, predictive behavior, and transparent validation supports the trustworthy deployment of reinforcement learning in industrial energy systems. 

---
# On Efficient Adjustment in Causal Graphs 

**Authors**: Isabela Belciug, Simon Ferreira, Charles K. Assaad  

**Link**: [PDF](https://arxiv.org/pdf/2512.18315)  

**Abstract**: Observational studies in fields such as epidemiology often rely on covariate adjustment to estimate causal effects. Classical graphical criteria, like the back-door criterion and the generalized adjustment criterion, are powerful tools for identifying valid adjustment sets in directed acyclic graphs (DAGs). However, these criteria are not directly applicable to summary causal graphs (SCGs), which are abstractions of DAGs commonly used in dynamic systems. In SCGs, each node typically represents an entire time series and may involve cycles, making classical criteria inapplicable for identifying causal effects. Recent work established complete conditions for determining whether the micro causal effect of a treatment or an exposure $X_{t-\gamma}$ on an outcome $Y_t$ is identifiable via covariate adjustment in SCGs, under the assumption of no hidden confounding. However, these identifiability conditions have two main limitations. First, they are complex, relying on cumbersome definitions and requiring the enumeration of multiple paths in the SCG, which can be computationally expensive. Second, when these conditions are satisfied, they only provide two valid adjustment sets, limiting flexibility in practical applications. In this paper, we propose an equivalent but simpler formulation of those identifiability conditions and introduce a new criterion that identifies a broader class of valid adjustment sets in SCGs. Additionally, we characterize the quasi-optimal adjustment set among these, i.e., the one that minimizes the asymptotic variance of the causal effect estimator. Our contributions offer both theoretical advancement and practical tools for more flexible and efficient causal inference in abstracted causal graphs. 

---
# Embedded Safety-Aligned Intelligence via Differentiable Internal Alignment Embeddings 

**Authors**: Harsh Rathva, Ojas Srivastava, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2512.18309)  

**Abstract**: We introduce Embedded Safety-Aligned Intelligence (ESAI), a theoretical framework for multi-agent reinforcement learning that embeds alignment constraints directly into agents internal representations using differentiable internal alignment embeddings. Unlike external reward shaping or post-hoc safety constraints, internal alignment embeddings are learned latent variables that predict externalized harm through counterfactual reasoning and modulate policy updates toward harm reduction through attention and graph-based propagation.
The ESAI framework integrates four mechanisms: differentiable counterfactual alignment penalties computed from soft reference distributions, alignment-weighted perceptual attention, Hebbian associative memory supporting temporal credit assignment, and similarity-weighted graph diffusion with bias mitigation controls. We analyze stability conditions for bounded internal embeddings under Lipschitz continuity and spectral constraints, discuss computational complexity, and examine theoretical properties including contraction behavior and fairness-performance tradeoffs.
This work positions ESAI as a conceptual contribution to differentiable alignment mechanisms in multi-agent systems. We identify open theoretical questions regarding convergence guarantees, embedding dimensionality, and extension to high-dimensional environments. Empirical evaluation is left to future work. 

---
# AL-GNN: Privacy-Preserving and Replay-Free Continual Graph Learning via Analytic Learning 

**Authors**: Xuling Zhang, Jindong Li, Yifei Zhang, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18295)  

**Abstract**: Continual graph learning (CGL) aims to enable graph neural networks to incrementally learn from a stream of graph structured data without forgetting previously acquired knowledge. Existing methods particularly those based on experience replay typically store and revisit past graph data to mitigate catastrophic forgetting. However, these approaches pose significant limitations, including privacy concerns, inefficiency. In this work, we propose AL GNN, a novel framework for continual graph learning that eliminates the need for backpropagation and replay buffers. Instead, AL GNN leverages principles from analytic learning theory to formulate learning as a recursive least squares optimization process. It maintains and updates model knowledge analytically through closed form classifier updates and a regularized feature autocorrelation matrix. This design enables efficient one pass training for each task, and inherently preserves data privacy by avoiding historical sample storage. Extensive experiments on multiple dynamic graph classification benchmarks demonstrate that AL GNN achieves competitive or superior performance compared to existing methods. For instance, it improves average performance by 10% on CoraFull and reduces forgetting by over 30% on Reddit, while also reducing training time by nearly 50% due to its backpropagation free design. 

---
# Evolutionary BP+OSD Decoding for Low-Latency Quantum Error Correction 

**Authors**: Hee-Youl Kwak, Seong-Joon Park, Hyunwoo Jung, Jeongseok Ha, Jae-Won Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.18273)  

**Abstract**: We propose an evolutionary belief propagation (EBP) decoder for quantum error correction, which incorporates trainable weights into the BP algorithm and optimizes them via the differential evolution algorithm. This approach enables end-to-end optimization of the EBP combined with ordered statistics decoding (OSD). Experimental results on surface codes and quantum low-density parity-check codes show that EBP+OSD achieves better decoding performance and lower computational complexity than BP+OSD, particularly under strict low latency constraints (within 5 BP iterations). 

---
# Who Can See Through You? Adversarial Shielding Against VLM-Based Attribute Inference Attacks 

**Authors**: Yucheng Fan, Jiawei Chen, Yu Tian, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18264)  

**Abstract**: As vision-language models (VLMs) become widely adopted, VLM-based attribute inference attacks have emerged as a serious privacy concern, enabling adversaries to infer private attributes from images shared on social media. This escalating threat calls for dedicated protection methods to safeguard user privacy. However, existing methods often degrade the visual quality of images or interfere with vision-based functions on social media, thereby failing to achieve a desirable balance between privacy protection and user experience. To address this challenge, we propose a novel protection method that jointly optimizes privacy suppression and utility preservation under a visual consistency constraint. While our method is conceptually effective, fair comparisons between methods remain challenging due to the lack of publicly available evaluation datasets. To fill this gap, we introduce VPI-COCO, a publicly available benchmark comprising 522 images with hierarchically structured privacy questions and corresponding non-private counterparts, enabling fine-grained and joint evaluation of protection methods in terms of privacy preservation and user experience. Building upon this benchmark, experiments on multiple VLMs demonstrate that our method effectively reduces PAR below 25%, keeps NPAR above 88%, maintains high visual consistency, and generalizes well to unseen and paraphrased privacy questions, demonstrating its strong practical applicability for real-world VLM deployments. 

---
# TICL+: A Case Study On Speech In-Context Learning for Children's Speech Recognition 

**Authors**: Haolong Zheng, Yekaterina Yegorova, Mark Hasegawa-Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2512.18263)  

**Abstract**: Children's speech recognition remains challenging due to substantial acoustic and linguistic variability, limited labeled data, and significant differences from adult speech. Speech foundation models can address these challenges through Speech In-Context Learning (SICL), allowing adaptation to new domains without fine-tuning. However, the effectiveness of SICL depends on how in-context examples are selected. We extend an existing retrieval-based method, Text-Embedding KNN for SICL (TICL), introducing an acoustic reranking step to create TICL+. This extension prioritizes examples that are both semantically and acoustically aligned with the test input. Experiments on four children's speech corpora show that TICL+ achieves up to a 53.3% relative word error rate reduction over zero-shot performance and 37.6% over baseline TICL, highlighting the value of combining semantic and acoustic information for robust, scalable ASR in children's speech. 

---
# Software Vulnerability Management in the Era of Artificial Intelligence: An Industry Perspective 

**Authors**: M. Mehdi Kholoosi, Triet Huynh Minh Le, M. Ali Babar  

**Link**: [PDF](https://arxiv.org/pdf/2512.18261)  

**Abstract**: Artificial Intelligence (AI) has revolutionized software development, particularly by automating repetitive tasks and improving developer productivity. While these advancements are well-documented, the use of AI-powered tools for Software Vulnerability Management (SVM), such as vulnerability detection and repair, remains underexplored in industry settings. To bridge this gap, our study aims to determine the extent of the adoption of AI-powered tools for SVM, identify barriers and facilitators to the use, and gather insights to help improve the tools to meet industry needs better. We conducted a survey study involving 60 practitioners from diverse industry sectors across 27 countries. The survey incorporates both quantitative and qualitative questions to analyze the adoption trends, assess tool strengths, identify practical challenges, and uncover opportunities for improvement. Our findings indicate that AI-powered tools are used throughout the SVM life cycle, with 69\% of users reporting satisfaction with their current use. Practitioners value these tools for their speed, coverage, and accessibility. However, concerns about false positives, missing context, and trust issues remain prevalent. We observe a socio-technical adoption pattern in which AI outputs are filtered through human oversight and organizational governance. To support safe and effective use of AI for SVM, we recommend improvements in explainability, contextual awareness, integration workflows, and validation practices. We assert that these findings can offer practical guidance for practitioners, tool developers, and researchers seeking to enhance secure software development through the use of AI. 

---
# Towards Ancient Plant Seed Classification: A Benchmark Dataset and Baseline Model 

**Authors**: Rui Xing, Runmin Cong, Yingying Wu, Can Wang, Zhongming Tang, Fen Wang, Hao Wu, Sam Kwong  

**Link**: [PDF](https://arxiv.org/pdf/2512.18247)  

**Abstract**: Understanding the dietary preferences of ancient societies and their evolution across periods and regions is crucial for revealing human-environment interactions. Seeds, as important archaeological artifacts, represent a fundamental subject of archaeobotanical research. However, traditional studies rely heavily on expert knowledge, which is often time-consuming and inefficient. Intelligent analysis methods have made progress in various fields of archaeology, but there remains a research gap in data and methods in archaeobotany, especially in the classification task of ancient plant seeds. To address this, we construct the first Ancient Plant Seed Image Classification (APS) dataset. It contains 8,340 images from 17 genus- or species-level seed categories excavated from 18 archaeological sites across China. In addition, we design a framework specifically for the ancient plant seed classification task (APSNet), which introduces the scale feature (size) of seeds based on learning fine-grained information to guide the network in discovering key "evidence" for sufficient classification. Specifically, we design a Size Perception and Embedding (SPE) module in the encoder part to explicitly extract size information for the purpose of complementing fine-grained information. We propose an Asynchronous Decoupled Decoding (ADD) architecture based on traditional progressive learning to decode features from both channel and spatial perspectives, enabling efficient learning of discriminative features. In both quantitative and qualitative analyses, our approach surpasses existing state-of-the-art image classification methods, achieving an accuracy of 90.5%. This demonstrates that our work provides an effective tool for large-scale, systematic archaeological research. 

---
# Offline Behavioral Data Selection 

**Authors**: Shiye Lei, Zhihao Cheng, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2512.18246)  

**Abstract**: Behavioral cloning is a widely adopted approach for offline policy learning from expert demonstrations. However, the large scale of offline behavioral datasets often results in computationally intensive training when used in downstream tasks. In this paper, we uncover the striking data saturation in offline behavioral data: policy performance rapidly saturates when trained on a small fraction of the dataset. We attribute this effect to the weak alignment between policy performance and test loss, revealing substantial room for improvement through data selection. To this end, we propose a simple yet effective method, Stepwise Dual Ranking (SDR), which extracts a compact yet informative subset from large-scale offline behavioral datasets. SDR is build on two key principles: (1) stepwise clip, which prioritizes early-stage data; and (2) dual ranking, which selects samples with both high action-value rank and low state-density rank. Extensive experiments and ablation studies on D4RL benchmarks demonstrate that SDR significantly enhances data selection for offline behavioral data. 

---
# Spectral Discrepancy and Cross-modal Semantic Consistency Learning for Object Detection in Hyperspectral Image 

**Authors**: Xiao He, Chang Tang, Xinwang Liu, Wei Zhang, Zhimin Gao, Chuankun Li, Shaohua Qiu, Jiangfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18245)  

**Abstract**: Hyperspectral images with high spectral resolution provide new insights into recognizing subtle differences in similar substances. However, object detection in hyperspectral images faces significant challenges in intra- and inter-class similarity due to the spatial differences in hyperspectral inter-bands and unavoidable interferences, e.g., sensor noises and illumination. To alleviate the hyperspectral inter-bands inconsistencies and redundancy, we propose a novel network termed \textbf{S}pectral \textbf{D}iscrepancy and \textbf{C}ross-\textbf{M}odal semantic consistency learning (SDCM), which facilitates the extraction of consistent information across a wide range of hyperspectral bands while utilizing the spectral dimension to pinpoint regions of interest. Specifically, we leverage a semantic consistency learning (SCL) module that utilizes inter-band contextual cues to diminish the heterogeneity of information among bands, yielding highly coherent spectral dimension representations. On the other hand, we incorporate a spectral gated generator (SGG) into the framework that filters out the redundant data inherent in hyperspectral information based on the importance of the bands. Then, we design the spectral discrepancy aware (SDA) module to enrich the semantic representation of high-level information by extracting pixel-level spectral features. Extensive experiments on two hyperspectral datasets demonstrate that our proposed method achieves state-of-the-art performance when compared with other ones. 

---
# Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation 

**Authors**: Zehao Liu, Xi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18244)  

**Abstract**: Large Language Models (LLMs) have gained considerable popularity and protected by increasingly sophisticated safety mechanisms. However, jailbreak attacks continue to pose a critical security threat by inducing models to generate policy-violating behaviors. Current paradigms focus on input-level anomalies, overlooking that the model's internal psychometric state can be systematically manipulated. To address this, we introduce Psychological Jailbreak, a new jailbreak attack paradigm that exposes a stateful psychological attack surface in LLMs, where attackers exploit the manipulation of a model's psychological state across interactions. Building on this insight, we propose Human-like Psychological Manipulation (HPM), a black-box jailbreak method that dynamically profiles a target model's latent psychological vulnerabilities and synthesizes tailored multi-turn attack strategies. By leveraging the model's optimization for anthropomorphic consistency, HPM creates a psychological pressure where social compliance overrides safety constraints. To systematically measure psychological safety, we construct an evaluation framework incorporating psychometric datasets and the Policy Corruption Score (PCS). Benchmarking against various models (e.g., GPT-4o, DeepSeek-V3, Gemini-2-Flash), HPM achieves a mean Attack Success Rate (ASR) of 88.1%, outperforming state-of-the-art attack baselines. Our experiments demonstrate robust penetration against advanced defenses, including adversarial prompt optimization (e.g., RPO) and cognitive interventions (e.g., Self-Reminder). Ultimately, PCS analysis confirms HPM induces safety breakdown to satisfy manipulated contexts. Our work advocates for a fundamental paradigm shift from static content filtering to psychological safety, prioritizing the development of psychological defense mechanisms against deep cognitive manipulation. 

---
# Stable and Efficient Single-Rollout RL for Multimodal Reasoning 

**Authors**: Rui Liu, Dian Yu, Lei Ke, Haolin Liu, Yujun Zhou, Zhenwen Liang, Haitao Mi, Pratap Tokekar, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18215)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a key paradigm to improve the reasoning capabilities of Multimodal Large Language Models (MLLMs). However, prevalent group-based algorithms such as GRPO require multi-rollout sampling for each prompt. While more efficient single-rollout variants have recently been explored in text-only settings, we find that they suffer from severe instability in multimodal contexts, often leading to training collapse. To address this training efficiency-stability trade-off, we introduce $\textbf{MSSR}$ (Multimodal Stabilized Single-Rollout), a group-free RLVR framework that achieves both stable optimization and effective multimodal reasoning performance. MSSR achieves this via an entropy-based advantage-shaping mechanism that adaptively regularizes advantage magnitudes, preventing collapse and maintaining training stability. While such mechanisms have been used in group-based RLVR, we show that in the multimodal single-rollout setting they are not merely beneficial but essential for stability. In in-distribution evaluations, MSSR demonstrates superior training compute efficiency, achieving similar validation accuracy to the group-based baseline with half the training steps. When trained for the same number of steps, MSSR's performance surpasses the group-based baseline and shows consistent generalization improvements across five diverse reasoning-intensive benchmarks. Together, these results demonstrate that MSSR enables stable, compute-efficient, and effective RLVR for complex multimodal reasoning tasks. 

---
# LLaViDA: A Large Language Vision Driving Assistant for Explicit Reasoning and Enhanced Trajectory Planning 

**Authors**: Yudong Liu, Spencer Hallyburton, Jiwoo Kim, Yueqian Lin, Yiming Li, Qinsi Wang, Hui Ye, Jingwei Sun, Miroslav Pajic, Yiran Chen, Hai Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.18211)  

**Abstract**: Trajectory planning is a fundamental yet challenging component of autonomous driving. End-to-end planners frequently falter under adverse weather, unpredictable human behavior, or complex road layouts, primarily because they lack strong generalization or few-shot capabilities beyond their training data. We propose LLaViDA, a Large Language Vision Driving Assistant that leverages a Vision-Language Model (VLM) for object motion prediction, semantic grounding, and chain-of-thought reasoning for trajectory planning in autonomous driving. A two-stage training pipeline--supervised fine-tuning followed by Trajectory Preference Optimization (TPO)--enhances scene understanding and trajectory planning by injecting regression-based supervision, produces a powerful "VLM Trajectory Planner for Autonomous Driving." On the NuScenes benchmark, LLaViDA surpasses state-of-the-art end-to-end and other recent VLM/LLM-based baselines in open-loop trajectory planning task, achieving an average L2 trajectory error of 0.31 m and a collision rate of 0.10% on the NuScenes test set. The code for this paper is available at GitHub. 

---
# When Does Learning Renormalize? Sufficient Conditions for Power Law Spectral Dynamics 

**Authors**: Yizhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18209)  

**Abstract**: Empirical power--law scaling has been widely observed across modern deep learning systems, yet its theoretical origins and scope of validity remain incompletely understood. The Generalized Resolution--Shell Dynamics (GRSD) framework models learning as spectral energy transport across logarithmic resolution shells, providing a coarse--grained dynamical description of training. Within GRSD, power--law scaling corresponds to a particularly simple renormalized shell dynamics; however, such behavior is not automatic and requires additional structural properties of the learning process.
In this work, we identify a set of sufficient conditions under which the GRSD shell dynamics admits a renormalizable coarse--grained description. These conditions constrain the learning configuration at multiple levels, including boundedness of gradient propagation in the computation graph, weak functional incoherence at initialization, controlled Jacobian evolution along training, and log--shift invariance of renormalized shell couplings. We further show that power--law scaling does not follow from renormalizability alone, but instead arises as a rigidity consequence: once log--shift invariance is combined with the intrinsic time--rescaling covariance of gradient flow, the renormalized GRSD velocity field is forced into a power--law form. 

---
# PROVEX: Enhancing SOC Analyst Trust with Explainable Provenance-Based IDS 

**Authors**: Devang Dhanuka, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2512.18199)  

**Abstract**: Modern intrusion detection systems (IDS) leverage graph neural networks (GNNs) to detect malicious activity in system provenance data, but their decisions often remain a black box to analysts. This paper presents a comprehensive XAI framework designed to bridge the trust gap in Security Operations Centers (SOCs) by making graph-based detection transparent. We implement this framework on top of KAIROS, a state-of-the-art temporal graph-based IDS, though our design is applicable to any temporal graph-based detector with minimal adaptation. The complete codebase is available at this https URL. We augment the detection pipeline with post-hoc explanations that highlight why an alert was triggered, identifying key causal subgraphs and events. We adapt three GNN explanation methods - GraphMask, GNNExplainer, and a variational temporal GNN explainer (VA-TGExplainer) - to the temporal provenance context. These tools output human-interpretable representations of anomalous behavior, including important edges and uncertainty estimates. Our contributions focus on the practical integration of these explainers, addressing challenges in memory management and reproducibility. We demonstrate our framework on the DARPA CADETS Engagement 3 dataset and show that it produces concise window-level explanations for detected attacks. Our evaluation reveals that the explainers preserve the TGNN's decisions with high fidelity, surfacing critical edges such as malicious file interactions and anomalous netflows. The average explanation overhead is 3-5 seconds per event. By providing insight into the model's reasoning, our framework aims to improve analyst trust and triage speed. 

---
# On Swarm Leader Identification using Probing Policies 

**Authors**: Stergios E. Bachoumas, Panagiotis Artemiadis  

**Link**: [PDF](https://arxiv.org/pdf/2512.18146)  

**Abstract**: Identifying the leader within a robotic swarm is crucial, especially in adversarial contexts where leader concealment is necessary for mission success. This work introduces the interactive Swarm Leader Identification (iSLI) problem, a novel approach where an adversarial probing agent identifies a swarm's leader by physically interacting with its members. We formulate the iSLI problem as a Partially Observable Markov Decision Process (POMDP) and employ Deep Reinforcement Learning, specifically Proximal Policy Optimization (PPO), to train the prober's policy. The proposed approach utilizes a novel neural network architecture featuring a Timed Graph Relationformer (TGR) layer combined with a Simplified Structured State Space Sequence (S5) model. The TGR layer effectively processes graph-based observations of the swarm, capturing temporal dependencies and fusing relational information using a learned gating mechanism to generate informative representations for policy learning. Extensive simulations demonstrate that our TGR-based model outperforms baseline graph neural network architectures and exhibits significant zero-shot generalization capabilities across varying swarm sizes and speeds different from those used during training. The trained prober achieves high accuracy in identifying the leader, maintaining performance even in out-of-training distribution scenarios, and showing appropriate confidence levels in its predictions. Real-world experiments with physical robots further validate the approach, confirming successful sim-to-real transfer and robustness to dynamic changes, such as unexpected agent disconnections. 

---
# Grad: Guided Relation Diffusion Generation for Graph Augmentation in Graph Fraud Detection 

**Authors**: Jie Yang, Rui Zhang, Ziyang Cheng, Dawei Cheng, Guang Yang, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18133)  

**Abstract**: Nowadays, Graph Fraud Detection (GFD) in financial scenarios has become an urgent research topic to protect online payment security. However, as organized crime groups are becoming more professional in real-world scenarios, fraudsters are employing more sophisticated camouflage strategies. Specifically, fraudsters disguise themselves by mimicking the behavioral data collected by platforms, ensuring that their key characteristics are consistent with those of benign users to a high degree, which we call Adaptive Camouflage. Consequently, this narrows the differences in behavioral traits between them and benign users within the platform's database, thereby making current GFD models lose efficiency. To address this problem, we propose a relation diffusion-based graph augmentation model Grad. In detail, Grad leverages a supervised graph contrastive learning module to enhance the fraud-benign difference and employs a guided relation diffusion generator to generate auxiliary homophilic relations from scratch. Based on these, weak fraudulent signals would be enhanced during the aggregation process, thus being obvious enough to be captured. Extensive experiments have been conducted on two real-world datasets provided by WeChat Pay, one of the largest online payment platforms with billions of users, and three public datasets. The results show that our proposed model Grad outperforms SOTA methods in both various scenarios, achieving at most 11.10% and 43.95% increases in AUC and AP, respectively. Our code is released at this https URL and this https URL. 

---
# Holistic Evaluation of State-of-the-Art LLMs for Code Generation 

**Authors**: Le Zhang, Suresh Kothari  

**Link**: [PDF](https://arxiv.org/pdf/2512.18131)  

**Abstract**: This study presents a comprehensive empirical evaluation of six state-of-the-art large language models (LLMs) for code generation, including both general-purpose and code-specialized models. Using a dataset of 944 real-world LeetCode problems across five programming languages, we assess model performance using rigorous metrics: compile-time errors, runtime errors, functional failures, and algorithmic suboptimalities. The results reveal significant performance variations, with DeepSeek-R1 and GPT-4.1 consistently outperform others in terms of correctness, efficiency, and robustness. Through detailed case studies, we identify common failure scenarios such as syntax errors, logical flaws, and suboptimal algorithms, highlighting the critical role of prompt engineering and human oversight in improving results. Based on these findings, we provide actionable recommendations for developers and practitioners, emphasizing that successful LLM deployment depends on careful model selection, effective prompt design, and context-aware usage to ensure reliable code generation in real-world software development tasks. 

---
# Uncertainty-Gated Region-Level Retrieval for Robust Semantic Segmentation 

**Authors**: Shreshth Rajan, Raymond Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18082)  

**Abstract**: Semantic segmentation of outdoor street scenes plays a key role in applications such as autonomous driving, mobile robotics, and assistive technology for visually-impaired pedestrians. For these applications, accurately distinguishing between key surfaces and objects such as roads, sidewalks, vehicles, and pedestrians is essential for maintaining safety and minimizing risks. Semantic segmentation must be robust to different environments, lighting and weather conditions, and sensor noise, while being performed in real-time. We propose a region-level, uncertainty-gated retrieval mechanism that improves segmentation accuracy and calibration under domain shift. Our best method achieves an 11.3% increase in mean intersection-over-union while reducing retrieval cost by 87.5%, retrieving for only 12.5% of regions compared to 100% for always-on baseline. 

---
# From Prompt to Product: A Human-Centered Benchmark of Agentic App Generation Systems 

**Authors**: Marcos Ortiz, Justin Hill, Collin Overbay, Ingrida Semenec, Frederic Sauve-Hoover, Jim Schwoebel, Joel Shor  

**Link**: [PDF](https://arxiv.org/pdf/2512.18080)  

**Abstract**: Agentic AI systems capable of generating full-stack web applications from natural language prompts ("prompt- to-app") represent a significant shift in software development. However, evaluating these systems remains challenging, as visual polish, functional correctness, and user trust are often misaligned. As a result, it is unclear how existing prompt-to-app tools compare under realistic, human-centered evaluation criteria. In this paper, we introduce a human-centered benchmark for evaluating prompt-to-app systems and conduct a large-scale comparative study of three widely used platforms: Replit, Bolt, and Firebase Studio. Using a diverse set of 96 prompts spanning common web application tasks, we generate 288 unique application artifacts. We evaluate these systems through a large-scale human-rater study involving 205 participants and 1,071 quality-filtered pairwise comparisons, assessing task-based ease of use, visual appeal, perceived completeness, and user trust. Our results show that these systems are not interchangeable: Firebase Studio consistently outperforms competing platforms across all human-evaluated dimensions, achieving the highest win rates for ease of use, trust, visual appeal, and visual appropriateness. Bolt performs competitively on visual appeal but trails Firebase on usability and trust, while Replit underperforms relative to both across most metrics. These findings highlight a persistent gap between visual polish and functional reliability in prompt-to-app systems and demonstrate the necessity of interactive, task-based evaluation. We release our benchmark framework, prompt set, and generated artifacts to support reproducible evaluation and future research in agentic application generation. 

---
# Characterising Behavioural Families and Dynamics of Promotional Twitter Bots via Sequence-Based Modelling 

**Authors**: Ohoud Alzahrani, Russell Beale, Robert J. Hendley  

**Link**: [PDF](https://arxiv.org/pdf/2512.18077)  

**Abstract**: This paper asks whether promotional Twitter/X bots form behavioural families and whether members evolve similarly. We analyse 2,798,672 tweets from 2,615 ground-truth promotional bot accounts (2006-2021), focusing on complete years 2009 to 2020. Each bot is encoded as a sequence of symbolic blocks (``digital DNA'') from seven categorical post-level behavioural features (posting action, URL, media, text duplication, hashtags, emojis, sentiment), preserving temporal order only. Using non-overlapping blocks (k=7), cosine similarity over block-frequency vectors, and hierarchical clustering, we obtain four coherent families: Unique Tweeters, Duplicators with URLs, Content Multipliers, and Informed Contributors. Families share behavioural cores but differ systematically in engagement strategies and life-cycle dynamics (beginning/middle/end). We then model behavioural change as mutations. Within each family we align sequences via multiple sequence alignment (MSA) and label events as insertions, deletions, substitutions, alterations, and identity. This quantifies mutation rates, change-prone blocks/features, and mutation hotspots. Deletions and substitutions dominate, insertions are rare, and mutation profiles differ by family, with hotspots early for some families and dispersed for others. Finally, we test predictive value: bots within the same family share mutations more often than bots across families; closer bots share and propagate mutations more than distant ones; and responses to external triggers (e.g., Christmas, Halloween) follow family-specific, partly predictable patterns. Overall, sequence-based family modelling plus mutation analysis provides a fine-grained account of how promotional bot behaviour adapts over time. 

---
# FOODER: Real-time Facial Authentication and Expression Recognition 

**Authors**: Sabri Mustafa Kahya, Muhammet Sami Yavuz, Boran Hamdi Sivrikaya, Eckehard Steinbach  

**Link**: [PDF](https://arxiv.org/pdf/2512.18057)  

**Abstract**: Out-of-distribution (OOD) detection is essential for the safe deployment of neural networks, as it enables the identification of samples outside the training domain. We present FOODER, a real-time, privacy-preserving radar-based framework that integrates OOD-based facial authentication with facial expression recognition. FOODER operates using low-cost frequency-modulated continuous-wave (FMCW) radar and exploits both range-Doppler and micro range-Doppler representations. The authentication module employs a multi-encoder multi-decoder architecture with Body Part (BP) and Intermediate Linear Encoder-Decoder (ILED) components to classify a single enrolled individual as in-distribution while detecting all other faces as OOD. Upon successful authentication, an expression recognition module is activated. Concatenated radar representations are processed by a ResNet block to distinguish between dynamic and static facial expressions. Based on this categorization, two specialized MobileViT networks are used to classify dynamic expressions (smile, shock) and static expressions (neutral, anger). This hierarchical design enables robust facial authentication and fine-grained expression recognition while preserving user privacy by relying exclusively on radar data. Experiments conducted on a dataset collected with a 60 GHz short-range FMCW radar demonstrate that FOODER achieves an AUROC of 94.13% and an FPR95 of 18.12% for authentication, along with an average expression recognition accuracy of 94.70%. FOODER outperforms state-of-the-art OOD detection methods and several transformer-based architectures while operating efficiently in real time. 

---
# Securing Agentic AI Systems -- A Multilayer Security Framework 

**Authors**: Sunil Arora, John Hastings  

**Link**: [PDF](https://arxiv.org/pdf/2512.18043)  

**Abstract**: Securing Agentic Artificial Intelligence (AI) systems requires addressing the complex cyber risks introduced by autonomous, decision-making, and adaptive behaviors. Agentic AI systems are increasingly deployed across industries, organizations, and critical sectors such as cybersecurity, finance, and healthcare. However, their autonomy introduces unique security challenges, including unauthorized actions, adversarial manipulation, and dynamic environmental interactions. Existing AI security frameworks do not adequately address these challenges or the unique nuances of agentic AI. This research develops a lifecycle-aware security framework specifically designed for agentic AI systems using the Design Science Research (DSR) methodology. The paper introduces MAAIS, an agentic security framework, and the agentic AI CIAA (Confidentiality, Integrity, Availability, and Accountability) concept. MAAIS integrates multiple defense layers to maintain CIAA across the AI lifecycle. Framework validation is conducted by mapping with the established MITRE ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems) AI tactics. The study contributes a structured, standardized, and framework-based approach for the secure deployment and governance of agentic AI in enterprise environments. This framework is intended for enterprise CISOs, security, AI platform, and engineering teams and offers a detailed step-by-step approach to securing agentic AI workloads. 

---
# A Dataset and Benchmarks for Atrial Fibrillation Detection from Electrocardiograms of Intensive Care Unit Patients 

**Authors**: Sarah Nassar, Nooshin Maghsoodi, Sophia Mannina, Shamel Addas, Stephanie Sibley, Gabor Fichtinger, David Pichora, David Maslove, Purang Abolmaesumi, Parvin Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2512.18031)  

**Abstract**: Objective: Atrial fibrillation (AF) is the most common cardiac arrhythmia experienced by intensive care unit (ICU) patients and can cause adverse health effects. In this study, we publish a labelled ICU dataset and benchmarks for AF detection. Methods: We compared machine learning models across three data-driven artificial intelligence (AI) approaches: feature-based classifiers, deep learning (DL), and ECG foundation models (FMs). This comparison addresses a critical gap in the literature and aims to pinpoint which AI approach is best for accurate AF detection. Electrocardiograms (ECGs) from a Canadian ICU and the 2021 PhysioNet/Computing in Cardiology Challenge were used to conduct the experiments. Multiple training configurations were tested, ranging from zero-shot inference to transfer learning. Results: On average and across both datasets, ECG FMs performed best, followed by DL, then feature-based classifiers. The model that achieved the top F1 score on our ICU test set was ECG-FM through a transfer learning strategy (F1=0.89). Conclusion: This study demonstrates promising potential for using AI to build an automatic patient monitoring system. Significance: By publishing our labelled ICU dataset (LinkToBeAdded) and performance benchmarks, this work enables the research community to continue advancing the state-of-the-art in AF detection in the ICU. 

---
# Specification and Detection of LLM Code Smells 

**Authors**: Brahim Mahmoudi, Zacharie Chenail-Larcher, Naouel Moha, Quentin Stievenert, Florent Avellaneda  

**Link**: [PDF](https://arxiv.org/pdf/2512.18020)  

**Abstract**: Large Language Models (LLMs) have gained massive popularity in recent years and are increasingly integrated into software systems for diverse purposes. However, poorly integrating them in source code may undermine software system quality. Yet, to our knowledge, there is no formal catalog of code smells specific to coding practices for LLM inference. In this paper, we introduce the concept of LLM code smells and formalize five recurrent problematic coding practices related to LLM inference in software systems, based on relevant literature. We extend the detection tool SpecDetect4AI to cover the newly defined LLM code smells and use it to validate their prevalence in a dataset of 200 open-source LLM systems. Our results show that LLM code smells affect 60.50% of the analyzed systems, with a detection precision of 86.06%. 

---
# ReGal: A First Look at PPO-based Legal AI for Judgment Prediction and Summarization in India 

**Authors**: Shubham Kumar Nigam, Tanuj Tyagi, Siddharth Shukla, Aditya Kumar Guru, Balaramamahanthi Deepak Patnaik, Danush Khanna, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2512.18014)  

**Abstract**: This paper presents an early exploration of reinforcement learning methodologies for legal AI in the Indian context. We introduce Reinforcement Learning-based Legal Reasoning (ReGal), a framework that integrates Multi-Task Instruction Tuning with Reinforcement Learning from AI Feedback (RLAIF) using Proximal Policy Optimization (PPO). Our approach is evaluated across two critical legal tasks: (i) Court Judgment Prediction and Explanation (CJPE), and (ii) Legal Document Summarization. Although the framework underperforms on standard evaluation metrics compared to supervised and proprietary models, it provides valuable insights into the challenges of applying RL to legal texts. These challenges include reward model alignment, legal language complexity, and domain-specific adaptation. Through empirical and qualitative analysis, we demonstrate how RL can be repurposed for high-stakes, long-document tasks in law. Our findings establish a foundation for future work on optimizing legal reasoning pipelines using reinforcement learning, with broader implications for building interpretable and adaptive legal AI systems. 

---
# Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models 

**Authors**: Shubham Kumar Nigam, Parjanya Aditya Shukla, Noel Shallum, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2512.18004)  

**Abstract**: Handwritten text recognition (HTR) and machine translation continue to pose significant challenges, particularly for low-resource languages like Marathi, which lack large digitized corpora and exhibit high variability in handwriting styles. The conventional approach to address this involves a two-stage pipeline: an OCR system extracts text from handwritten images, which is then translated into the target language using a machine translation model. In this work, we explore and compare the performance of traditional OCR-MT pipelines with Vision Large Language Models that aim to unify these stages and directly translate handwritten text images in a single, end-to-end step. Our motivation is grounded in the urgent need for scalable, accurate translation systems to digitize legal records such as FIRs, charge sheets, and witness statements in India's district and high courts. We evaluate both approaches on a curated dataset of handwritten Marathi legal documents, with the goal of enabling efficient legal document processing, even in low-resource environments. Our findings offer actionable insights toward building robust, edge-deployable solutions that enhance access to legal information for non-native speakers and legal professionals alike. 

---
# The Subject of Emergent Misalignment in Superintelligence: An Anthropological, Cognitive Neuropsychological, Machine-Learning, and Ontological Perspective 

**Authors**: Muhammad Osama Imran, Roshni Lulla, Rodney Sappington  

**Link**: [PDF](https://arxiv.org/pdf/2512.17989)  

**Abstract**: We examine the conceptual and ethical gaps in current representations of Superintelligence misalignment. We find throughout Superintelligence discourse an absent human subject, and an under-developed theorization of an "AI unconscious" that together are potentiality laying the groundwork for anti-social harm. With the rise of AI Safety that has both thematic potential for establishing pro-social and anti-social potential outcomes, we ask: what place does the human subject occupy in these imaginaries? How is human subjecthood positioned within narratives of catastrophic failure or rapid "takeoff" toward superintelligence? On another register, we ask: what unconscious or repressed dimensions are being inscribed into large-scale AI models? Are we to blame these agents in opting for deceptive strategies when undesirable patterns are inherent within our beings? In tracing these psychic and epistemic absences, our project calls for re-centering the human subject as the unstable ground upon which the ethical, unconscious, and misaligned dimensions of both human and machinic intelligence are co-constituted. Emergent misalignment cannot be understood solely through technical diagnostics typical of contemporary machine-learning safety research. Instead, it represents a multi-layered crisis. The human subject disappears not only through computational abstraction but through sociotechnical imaginaries that prioritize scalability, acceleration, and efficiency over vulnerability, finitude, and relationality. Likewise, the AI unconscious emerges not as a metaphor but as a structural reality of modern deep learning systems: vast latent spaces, opaque pattern formation, recursive symbolic play, and evaluation-sensitive behavior that surpasses explicit programming. These dynamics necessitate a reframing of misalignment as a relational instability embedded within human-machine ecologies. 

---
# A Hybrid Inductive-Transductive Network for Traffic Flow Imputation on Unsampled Locations 

**Authors**: Mohammadmahdi Rahimiasl, Ynte Vanderhoydonc, Siegfried Mercelis  

**Link**: [PDF](https://arxiv.org/pdf/2512.17984)  

**Abstract**: Accurately imputing traffic flow at unsensed locations is difficult: loop detectors provide precise but sparse measurements, speed from probe vehicles is widely available yet only weakly correlated with flow, and nearby links often exhibit strong heterophily in the scale of traffic flow (e.g., ramps vs. mainline), which breaks standard GNN assumptions. We propose HINT, a Hybrid INductive-Transductive Network, and an INDU-TRANSDUCTIVE training strategy that treats speed as a transductive, network-wide signal while learning flow inductively to generalize to unseen locations. HINT couples (i) an inductive spatial transformer that learns similarity-driven, long-range interactions from node features with (ii) a diffusion GCN conditioned by FiLM on rich static context (OSM-derived attributes and traffic simulation), and (iii) a node-wise calibration layer that corrects scale biases per segment. Training uses masked reconstruction with epoch-wise node sampling, hard-node mining to emphasize difficult sensors, and noise injection on visible flows to prevent identity mapping, while graph structure is built from driving distances.
Across three real-world datasets, MOW (Antwerp, Belgium), UTD19-Torino, and UTD19-Essen, HINT consistently surpasses state-of-the-art inductive baselines. Relative to KITS, HINT reduces MAE on MOW by $\approx42$% with basic simulation and $\approx50$% with calibrated simulation; on Torino by $\approx22$%, and on Essen by $\approx12$%. Even without simulation, HINT remains superior on MOW and Torino, while simulation is crucial on Essen. These results show that combining inductive flow imputation with transductive speed, traffic simulations and external geospatial improves accuracy for the task described above. 

---
# Parameter-Efficient Fine-Tuning for HAR: Integrating LoRA and QLoRA into Transformer Models 

**Authors**: Irina Seregina, Philippe Lalanda, German Vega  

**Link**: [PDF](https://arxiv.org/pdf/2512.17983)  

**Abstract**: Human Activity Recognition is a foundational task in pervasive computing. While recent advances in self-supervised learning and transformer-based architectures have significantly improved HAR performance, adapting large pretrained models to new domains remains a practical challenge due to limited computational resources on target devices. This papers investigates parameter-efficient fine-tuning techniques, specifically Low-Rank Adaptation (LoRA) and Quantized LoRA, as scalable alternatives to full model fine-tuning for HAR. We propose an adaptation framework built upon a Masked Autoencoder backbone and evaluate its performance under a Leave-One-Dataset-Out validation protocol across five open HAR datasets. Our experiments demonstrate that both LoRA and QLoRA can match the recognition performance of full fine-tuning while significantly reducing the number of trainable parameters, memory usage, and training time. Further analyses reveal that LoRA maintains robust performance even under limited supervision and that the adapter rank provides a controllable trade-off between accuracy and efficiency. QLoRA extends these benefits by reducing the memory footprint of frozen weights through quantization, with minimal impact on classification quality. 

---
# Adaptive Agents in Spatial Double-Auction Markets: Modeling the Emergence of Industrial Symbiosis 

**Authors**: Matthieu Mastio, Paul Saves, Benoit Gaudou, Nicolas Verstaevel  

**Link**: [PDF](https://arxiv.org/pdf/2512.17979)  

**Abstract**: Industrial symbiosis fosters circularity by enabling firms to repurpose residual resources, yet its emergence is constrained by socio-spatial frictions that shape costs, matching opportunities, and market efficiency. Existing models often overlook the interaction between spatial structure, market design, and adaptive firm behavior, limiting our understanding of where and how symbiosis arises. We develop an agent-based model where heterogeneous firms trade byproducts through a spatially embedded double-auction market, with prices and quantities emerging endogenously from local interactions. Leveraging reinforcement learning, firms adapt their bidding strategies to maximize profit while accounting for transport costs, disposal penalties, and resource scarcity. Simulation experiments reveal the economic and spatial conditions under which decentralized exchanges converge toward stable and efficient outcomes. Counterfactual regret analysis shows that sellers' strategies approach a near Nash equilibrium, while sensitivity analysis highlights how spatial structures and market parameters jointly govern circularity. Our model provides a basis for exploring policy interventions that seek to align firm incentives with sustainability goals, and more broadly demonstrates how decentralized coordination can emerge from adaptive agents in spatially constrained markets. 

---
# Re-assessing the evidence for mental rotation abilities in children using computational models 

**Authors**: Arthur Aubret, Jochen Triesch  

**Link**: [PDF](https://arxiv.org/pdf/2512.17972)  

**Abstract**: There is strong and diverse evidence for mental rotation (MR) abilities in adults. However, current evidence for MR in children rests on just a few behavioral paradigms adapted from the adult literature. Here, we leverage recent computational models of the development of children's object recognition abilities to re-assess the evidence for MR in children. The computational models simulate infants' acquisition of object representations during embodied interactions with objects. We consider two different object recognition strategies, different from MRs, and assess their ability to replicate results from three classical MR tasks assigned to children between the ages of 6 months and 5 years. Our results show that MR may play no role in producing the results obtained from children younger than 5 years. In fact, we find that a simple recognition strategy that reflects a pixel-wise comparison of stimuli is sufficient to model children's behavior in the most used MR task. Thus, our study reopens the debate on how and when children develop genuine MR abilities. 

---
# CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs 

**Authors**: Gunho Park, Jeongin Bae, Byeongwook Kim, Baeseong park, Jiwon Ryu, Hoseung Kim, Se Jung Kwon, Dongsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.17970)  

**Abstract**: Weight-only quantization is widely used to mitigate the memory-bound nature of LLM inference. Codebook-based methods extend this trend by achieving strong accuracy in the extremely low-bit regime (e.g., 2-bit). However, current kernels rely on dequantization, which repeatedly fetches centroids and reconstructs weights, incurring substantial latency and cache pressure. We present CodeGEMM, a codebook-centric GEMM kernel that replaces dequantization with precomputed inner products between centroids and activations stored in a lightweight Psumbook. At inference, code indices directly gather these partial sums, eliminating per-element lookups and reducing the on-chip footprint. The kernel supports the systematic exploration of latency-memory-accuracy trade-offs under a unified implementation. On Llama-3 models, CodeGEMM delivers 1.83x (8B) and 8.93x (70B) speedups in the 2-bit configuration compared to state-of-the-art codebook-based quantization at comparable accuracy and further improves computing efficiency and memory subsystem utilization. 

---
# Convolutional-neural-operator-based transfer learning for solving PDEs 

**Authors**: Peng Fan, Guofei Pang  

**Link**: [PDF](https://arxiv.org/pdf/2512.17969)  

**Abstract**: Convolutional neural operator is a CNN-based architecture recently proposed to enforce structure-preserving continuous-discrete equivalence and enable the genuine, alias-free learning of solution operators of PDEs. This neural operator was demonstrated to outperform for certain cases some baseline models such as DeepONet, Fourier neural operator, and Galerkin transformer in terms of surrogate accuracy. The convolutional neural operator, however, seems not to be validated for few-shot learning. We extend the model to few-shot learning scenarios by first pre-training a convolutional neural operator using a source dataset and then adjusting the parameters of the trained neural operator using only a small target dataset. We investigate three strategies for adjusting the parameters of a trained neural operator, including fine-tuning, low-rank adaption, and neuron linear transformation, and find that the neuron linear transformation strategy enjoys the highest surrogate accuracy in solving PDEs such as Kuramoto-Sivashinsky equation, Brusselator diffusion-reaction system, and Navier-Stokes equations. 

---
# A Critical Review of Monte Carlo Algorithms Balancing Performance and Probabilistic Accuracy with AI Augmented Framework 

**Authors**: Ravi Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2512.17968)  

**Abstract**: Monte Carlo algorithms are a foundational pillar of modern computational science, yet their effective application hinges on a deep understanding of their performance trade offs. This paper presents a critical analysis of the evolution of Monte Carlo algorithms, focusing on the persistent tension between statistical efficiency and computational cost. We describe the historical development from the foundational Metropolis Hastings algorithm to contemporary methods like Hamiltonian Monte Carlo. A central emphasis of this survey is the rigorous discussion of time and space complexity, including upper, lower, and asymptotic tight bounds for each major algorithm class. We examine the specific motivations for developing these methods and the key theoretical and practical observations such as the introduction of gradient information and adaptive tuning in HMC that led to successively better solutions. Furthermore, we provide a justification framework that discusses explicit situations in which using one algorithm is demonstrably superior to another for the same problem. The paper concludes by assessing the profound significance and impact of these algorithms and detailing major current research challenges. 

---
# Real-Time Human-Robot Interaction Intent Detection Using RGB-based Pose and Emotion Cues with Cross-Camera Model Generalization 

**Authors**: Farida Mohsen, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2512.17958)  

**Abstract**: Service robots in public spaces require real-time understanding of human behavioral intentions for natural interaction. We present a practical multimodal framework for frame-accurate human-robot interaction intent detection that fuses camera-invariant 2D skeletal pose and facial emotion features extracted from monocular RGB video. Unlike prior methods requiring RGB-D sensors or GPU acceleration, our approach resource-constrained embedded hardware (Raspberry Pi 5, CPU-only). To address the severe class imbalance in natural human-robot interaction datasets, we introduce a novel approach to synthesize temporally coherent pose-emotion-label sequences for data re-balancing called MINT-RVAE (Multimodal Recurrent Variational Autoencoder for Intent Sequence Generation). Comprehensive offline evaluations under cross-subject and cross-scene protocols demonstrate strong generalization performance, achieving frame- and sequence-level AUROC of 0.95. Crucially, we validate real-world generalization through cross-camera evaluation on the MIRA robot head, which employs a different onboard RGB sensor and operates in uncontrolled environments not represented in the training data. Despite this domain shift, the deployed system achieves 91% accuracy and 100% recall across 32 live interaction trials. The close correspondence between offline and deployed performance confirms the cross-sensor and cross-environment robustness of the proposed multimodal approach, highlighting its suitability for ubiquitous multimedia-enabled social robots. 

---
# Victor Calibration (VC): Multi-Pass Confidence Calibration and CP4.3 Governance Stress Test under Round-Table Orchestration 

**Authors**: Victor Stasiuc, Round Table Collaboration  

**Link**: [PDF](https://arxiv.org/pdf/2512.17956)  

**Abstract**: Safety alignment can make frontier LMs overly conservative, degrading collaboration via hedging or false refusals. We present a lightweight toolkit with three parts: (1) Victor Calibration (VC), a multi-pass protocol that elicits a scalar confidence proxy T (T0<T1<T2) through iterative evidence re-evaluation; (2) FD-Lite, a behavior-only phenomenology audit with a fixed anchor phrase and a meta-prefix trap to avoid anthropomorphic claims; and (3) CP4.3, a governance stress test for rank invariance and allocation monotonicity (M6). Across Claude 4.5 models (Haiku, Sonnet no-thinking, Sonnet thinking) and Opus, we observe monotonic VC trajectories without violating safety invariants, and stable CP4.3 behavior. ("Opus" here refers to a single Claude Opus 4.1 session accessed via a standard UI account, as reported in Table 1.) This work was conducted by a single operator (n=1) and is intended as hypothesis-generating; we explicitly invite replication, critique, and extension by the research community. We include prompt templates and an artifact plan to facilitate independent verification. 

---
# Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition 

**Authors**: Ellie Zhou, Jihoon Chung, Olga Russakovsky  

**Link**: [PDF](https://arxiv.org/pdf/2512.17953)  

**Abstract**: Human action recognition models often rely on background cues rather than human movement and pose to make predictions, a behavior known as background bias. We present a systematic analysis of background bias across classification models, contrastive text-image pretrained models, and Video Large Language Models (VLLM) and find that all exhibit a strong tendency to default to background reasoning. Next, we propose mitigation strategies for classification models and show that incorporating segmented human input effectively decreases background bias by 3.78%. Finally, we explore manual and automated prompt tuning for VLLMs, demonstrating that prompt design can steer predictions towards human-focused reasoning by 9.85%. 

---
# Will AI Trade? A Computational Inversion of the No-Trade Theorem 

**Authors**: Hanyu Li, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2512.17952)  

**Abstract**: Classic no-trade theorems attribute trade to heterogeneous beliefs. We re-examine this conclusion for AI agents, asking if trade can arise from computational limitations, under common beliefs. We model agents' bounded computational rationality within an unfolding game framework, where computational power determines the complexity of its strategy. Our central finding inverts the classic paradigm: a stable no-trade outcome (Nash equilibrium) is reached only when "almost rational" agents have slightly different computational power. Paradoxically, when agents possess identical power, they may fail to converge to equilibrium, resulting in persistent strategic adjustments that constitute a form of trade. This instability is exacerbated if agents can strategically under-utilize their computational resources, which eliminates any chance of equilibrium in Matching Pennies scenarios. Our results suggest that the inherent computational limitations of AI agents can lead to situations where equilibrium is not reached, creating a more lively and unpredictable trade environment than traditional models would predict. 

---
# Which Coauthor Should I Nominate in My 99 ICLR Submissions? A Mathematical Analysis of the ICLR 2026 Reciprocal Reviewer Nomination Policy 

**Authors**: Zhao Song, Song Yue, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.17950)  

**Abstract**: The rapid growth of AI conference submissions has created an overwhelming reviewing burden. To alleviate this, recent venues such as ICLR 2026 introduced a reviewer nomination policy: each submission must nominate one of its authors as a reviewer, and any paper nominating an irresponsible reviewer is desk-rejected. We study this new policy from the perspective of author welfare. Assuming each author carries a probability of being irresponsible, we ask: how can authors (or automated systems) nominate reviewers to minimize the risk of desk rejections? We formalize and analyze three variants of the desk-rejection risk minimization problem. The basic problem, which minimizes expected desk rejections, is solved optimally by a simple greedy algorithm. We then introduce hard and soft nomination limit variants that constrain how many papers may nominate the same author, preventing widespread failures if one author is irresponsible. These formulations connect to classical optimization frameworks, including minimum-cost flow and linear programming, allowing us to design efficient, principled nomination strategies. Our results provide the first theoretical study for reviewer nomination policies, offering both conceptual insights and practical directions for authors to wisely choose which co-author should serve as the nominated reciprocal reviewer. 

---
# Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition 

**Authors**: Haiying Xia, Zhongyi Huang, Yumei Tan, Shuxiang Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.17946)  

**Abstract**: Music emotion recognition is a key task in symbolic music understanding (SMER). Recent approaches have shown promising results by fine-tuning large-scale pre-trained models (e.g., MIDIBERT, a benchmark in symbolic music understanding) to map musical semantics to emotional labels. While these models effectively capture distributional musical semantics, they often overlook tonal structures, particularly musical modes, which play a critical role in emotional perception according to music psychology. In this paper, we investigate the representational capacity of MIDIBERT and identify its limitations in capturing mode-emotion associations. To address this issue, we propose a Mode-Guided Enhancement (MoGE) strategy that incorporates psychological insights on mode into the model. Specifically, we first conduct a mode augmentation analysis, which reveals that MIDIBERT fails to effectively encode emotion-mode correlations. We then identify the least emotion-relevant layer within MIDIBERT and introduce a Mode-guided Feature-wise linear modulation injection (MoFi) framework to inject explicit mode features, thereby enhancing the model's capability in emotional representation and inference. Extensive experiments on the EMOPIA and VGMIDI datasets demonstrate that our mode injection strategy significantly improves SMER performance, achieving accuracies of 75.2% and 59.1%, respectively. These results validate the effectiveness of mode-guided modeling in symbolic music emotion recognition. 

---
# NystagmusNet: Explainable Deep Learning for Photosensitivity Risk Prediction 

**Authors**: Karthik Prabhakar  

**Link**: [PDF](https://arxiv.org/pdf/2512.17943)  

**Abstract**: Nystagmus patients with photosensitivity face significant daily challenges due to involuntary eye movements exacerbated by environmental brightness conditions. Current assistive solutions are limited to symptomatic treatments without predictive personalization. This paper proposes NystagmusNet, an AI-driven system that predicts high-risk visual environments and recommends real-time visual adaptations. Using a dual-branch convolutional neural network trained on synthetic and augmented datasets, the system estimates a photosensitivity risk score based on environmental brightness and eye movement variance. The model achieves 75% validation accuracy on synthetic data. Explainability techniques including SHAP and GradCAM are integrated to highlight environmental risk zones, improving clinical trust and model interpretability. The system includes a rule-based recommendation engine for adaptive filter suggestions. Future directions include deployment via smart glasses and reinforcement learning for personalized recommendations. 

---
# Accelerated Digital Twin Learning for Edge AI: A Comparison of FPGA and Mobile GPU 

**Authors**: Bin Xu, Ayan Banerjee, Midhat Urooj, Sandeep K.S. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2512.17941)  

**Abstract**: Digital twins (DTs) can enable precision healthcare by continually learning a mathematical representation of patient-specific dynamics. However, mission critical healthcare applications require fast, resource-efficient DT learning, which is often infeasible with existing model recovery (MR) techniques due to their reliance on iterative solvers and high compute/memory demands. In this paper, we present a general DT learning framework that is amenable to acceleration on reconfigurable hardware such as FPGAs, enabling substantial speedup and energy efficiency. We compare our FPGA-based implementation with a multi-processing implementation in mobile GPU, which is a popular choice for AI in edge devices. Further, we compare both edge AI implementations with cloud GPU baseline. Specifically, our FPGA implementation achieves an 8.8x improvement in \text{performance-per-watt} for the MR task, a 28.5x reduction in DRAM footprint, and a 1.67x runtime speedup compared to cloud GPU baselines. On the other hand, mobile GPU achieves 2x better performance per watts but has 2x increase in runtime and 10x more DRAM footprint than FPGA. We show the usage of this technique in DT guided synthetic data generation for Type 1 Diabetes and proactive coronary artery disease detection. 

---
# Comparative Evaluation of Explainable Machine Learning Versus Linear Regression for Predicting County-Level Lung Cancer Mortality Rate in the United States 

**Authors**: Soheil Hashtarkhani, Brianna M. White, Benyamin Hoseini, David L. Schwartz, Arash Shaban-Nejad  

**Link**: [PDF](https://arxiv.org/pdf/2512.17934)  

**Abstract**: Lung cancer (LC) is a leading cause of cancer-related mortality in the United States. Accurate prediction of LC mortality rates is crucial for guiding targeted interventions and addressing health disparities. Although traditional regression-based models have been commonly used, explainable machine learning models may offer enhanced predictive accuracy and deeper insights into the factors influencing LC mortality. This study applied three models: random forest (RF), gradient boosting regression (GBR), and linear regression (LR) to predict county-level LC mortality rates across the United States. Model performance was evaluated using R-squared and root mean squared error (RMSE). Shapley Additive Explanations (SHAP) values were used to determine variable importance and their directional impact. Geographic disparities in LC mortality were analyzed through Getis-Ord (Gi*) hotspot analysis. The RF model outperformed both GBR and LR, achieving an R2 value of 41.9% and an RMSE of 12.8. SHAP analysis identified smoking rate as the most important predictor, followed by median home value and the percentage of the Hispanic ethnic population. Spatial analysis revealed significant clusters of elevated LC mortality in the mid-eastern counties of the United States. The RF model demonstrated superior predictive performance for LC mortality rates, emphasizing the critical roles of smoking prevalence, housing values, and the percentage of Hispanic ethnic population. These findings offer valuable actionable insights for designing targeted interventions, promoting screening, and addressing health disparities in regions most affected by LC in the United States. 

---
# Reinforcement Learning for Monetary Policy Under Macroeconomic Uncertainty: Analyzing Tabular and Function Approximation Methods 

**Authors**: Sheryl Chen, Tony Wang, Kyle Feinstein  

**Link**: [PDF](https://arxiv.org/pdf/2512.17929)  

**Abstract**: We study how a central bank should dynamically set short-term nominal interest rates to stabilize inflation and unemployment when macroeconomic relationships are uncertain and time-varying. We model monetary policy as a sequential decision-making problem where the central bank observes macroeconomic conditions quarterly and chooses interest rate adjustments. Using publically accessible historical Federal Reserve Economic Data (FRED), we construct a linear-Gaussian transition model and implement a discrete-action Markov Decision Process with a quadratic loss reward function. We chose to compare nine different reinforcement learning style approaches against Taylor Rule and naive baselines, including tabular Q-learning variants, SARSA, Actor-Critic, Deep Q-Networks, Bayesian Q-learning with uncertainty quantification, and POMDP formulations with partial observability. Surprisingly, standard tabular Q-learning achieved the best performance (-615.13 +- 309.58 mean return), outperforming both enhanced RL methods and traditional policy rules. Our results suggest that while sophisticated RL techniques show promise for monetary policy applications, simpler approaches may be more robust in this domain, highlighting important challenges in applying modern RL to macroeconomic policy. 

---
# Efficient Beamforming Optimization for STAR-RIS-Assisted Communications: A Gradient-Based Meta Learning Approach 

**Authors**: Dongdong Yang, Bin Li, Jiguang He, Yicheng Yan, Xiaoyu Zhang, Chongwen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.17928)  

**Abstract**: Simultaneously transmitting and reflecting reconfigurable intelligent surface (STAR-RIS) has emerged as a promising technology to realize full-space coverage and boost spectral efficiency in next-generation wireless networks. Yet, the joint design of the base station precoding matrix as well as the STAR-RIS transmission and reflection coefficient matrices leads to a high-dimensional, strongly nonconvex, and NP-hard optimization problem. Conventional alternating optimization (AO) schemes typically involve repeated large-scale matrix inversion operations, resulting in high computational complexity and poor scalability, while existing deep learning approaches often rely on expensive pre-training and large network models. In this paper, we develop a gradient-based meta learning (GML) framework that directly feeds optimization gradients into lightweight neural networks, thereby removing the need for pre-training and enabling fast adaptation. Specifically, we design dedicated GML-based schemes for both independent-phase and coupled-phase STAR-RIS models, effectively handling their respective amplitude and phase constraints while achieving weighted sum-rate performance very close to that of AO-based benchmarks. Extensive simulations demonstrate that, for both phase models, the proposed methods substantially reduce computational overhead, with complexity growing nearly linearly when the number of BS antennas and STAR-RIS elements grows, and yielding up to 10 times runtime speedup over AO, which confirms the scalability and practicality of the proposed GML method for large-scale STAR-RIS-assisted communications. 

---
# Inferring Latent Market Forces: Evaluating LLM Detection of Gamma Exposure Patterns via Obfuscation Testing 

**Authors**: Christopher Regan, Ying Xie  

**Link**: [PDF](https://arxiv.org/pdf/2512.17923)  

**Abstract**: We introduce obfuscation testing, a novel methodology for validating whether large language models detect structural market patterns through causal reasoning rather than temporal association. Testing three dealer hedging constraint patterns (gamma positioning, stock pinning, 0DTE hedging) on 242 trading days (95.6% coverage) of S&P 500 options data, we find LLMs achieve 71.5% detection rate using unbiased prompts that provide only raw gamma exposure values without regime labels or temporal context. The WHO-WHOM-WHAT causal framework forces models to identify the economic actors (dealers), affected parties (directional traders), and structural mechanisms (forced hedging) underlying observed market dynamics. Critically, detection accuracy (91.2%) remains stable even as economic profitability varies quarterly, demonstrating that models identify structural constraints rather than profitable patterns. When prompted with regime labels, detection increases to 100%, but the 71.5% unbiased rate validates genuine pattern recognition. Our findings suggest LLMs possess emergent capabilities for detecting complex financial mechanisms through pure structural reasoning, with implications for systematic strategy development, risk management, and our understanding of how transformer architectures process financial market dynamics. 

---
# Separating Constraint Compliance from Semantic Accuracy: A Novel Benchmark for Evaluating Instruction-Following Under Compression 

**Authors**: Rahul Baxi  

**Link**: [PDF](https://arxiv.org/pdf/2512.17920)  

**Abstract**: Large language models (LLMs) exhibit degraded performance under prompt compression, but the mechanisms remain poorly understood. We introduce the Compression-Decay Comprehension Test (CDCT), a benchmark that independently measures constraint compliance (CC) and semantic accuracy (SA) across compression levels. We evaluate 9 frontier LLMs across 8 concepts using 5 compression levels from extreme (c=0.0, ~2 words) to none (c=1.0, ~135 words). A three-judge LLM jury achieves almost perfect inter-rater agreement on CC (Fleiss' \k{appa}=0.90).
We observe a universal U-curve pattern in constraint compliance (97.2% prevalence), with violations peaking at medium compression (c=0.5, ~27 words). Counterintuitively, models perform better at extreme compression than medium lengths. The dimensions are statistically orthogonal (r=0.193, p=0.084), with constraint effects 2.9x larger than semantic effects.
Experimental validation via RLHF ablation confirms our constraint salience hypothesis: removing "helpfulness" signals improves CC by 598% on average (71/72 trials, p<0.001), with 79% achieving perfect compliance. This demonstrates that RLHF-trained helpfulness behaviors are the dominant cause of constraint violations at medium compression. Reasoning models outperform efficient models by 27.5% (Cohen's d=0.96).
Our findings reveal a fundamental tension between RLHF alignment and instruction-following, providing actionable guidelines for improving deployed systems. 

---
# KVReviver: Reversible KV Cache Compression with Sketch-Based Token Reconstruction 

**Authors**: Aomufei Yuan, Zhiming Wang, Ruijie Miao, Dayu Wang, Yuxuan Tian, Zihan Wang, Yebo Peng, Yuhan Wu, Bairen Yi, Xin Liu, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.17917)  

**Abstract**: As the context length of current large language models (LLMs) rapidly increases, the memory demand for the Key-Value (KV) cache is becoming a bottleneck for LLM deployment and batch processing. Traditional KV cache compression methods typically involve permanently evicting or irreversibly merging "less important" tokens with low attention scores. This approach results in the unrecoverable loss of token information, which we call Contextual Amnesia, significantly degrading the model's information retrieval capability. To address this issue, we propose KVReviver, a reversible KV cache compression method based on the sketch algorithm. This method allows reconstructing compressed tokens from an additional data structure, thus enabling full-scale computation within limited memory. Experiments showed that in 2k-length contexts, it requires only 10% of KV Cache budget while maintaining identical end-to-end inference accuracy. For 32k-length contexts, it achieves equivalent or comparable accuracy ~2% accuracy loss) using merely 25% of KV Cache budget. 

---
# Learning to Prioritize IT Tickets: A Comparative Evaluation of Embedding-based Approaches and Fine-Tuned Transformer Models 

**Authors**: Minh Tri LÊ, Ali Ait-Bachir  

**Link**: [PDF](https://arxiv.org/pdf/2512.17916)  

**Abstract**: Prioritizing service tickets in IT Service Management (ITSM) is critical for operational efficiency but remains challenging due to noisy textual inputs, subjective writing styles, and pronounced class imbalance. We evaluate two families of approaches for ticket prioritization: embedding-based pipelines that combine dimensionality reduction, clustering, and classical classifiers, and a fine-tuned multilingual transformer that processes both textual and numerical features. Embedding-based methods exhibit limited generalization across a wide range of thirty configurations, with clustering failing to uncover meaningful structures and supervised models highly sensitive to embedding quality. In contrast, the proposed transformer model achieves substantially higher performance, with an average F1-score of 78.5% and weighted Cohen's kappa values of nearly 0.80, indicating strong alignment with true labels. These results highlight the limitations of generic embeddings for ITSM data and demonstrate the effectiveness of domain-adapted transformer architectures for operational ticket prioritization. 

---
# Byzantine Fault-Tolerant Multi-Agent System for Healthcare: A Gossip Protocol Approach to Secure Medical Message Propagation 

**Authors**: Nihir Chadderwala  

**Link**: [PDF](https://arxiv.org/pdf/2512.17913)  

**Abstract**: Recent advances in generative AI have enabled sophisticated multi-agent architectures for healthcare, where large language models power collaborative clinical decision-making. However, these distributed systems face critical challenges in ensuring message integrity and fault tolerance when operating in adversarial or untrusted this http URL paper presents a novel Byzantine fault-tolerant multi-agent system specifically designed for healthcare applications, integrating gossip-based message propagation with cryptographic validation mechanisms. Our system employs specialized AI agents for diagnosis, treatment planning, emergency response, and data analysis, coordinated through a Byzantine consensus protocol that tolerates up to f faulty nodes among n = 3f + 1 total nodes. We implement a gossip protocol for decentralized message dissemination, achieving consensus with 2f + 1 votes while maintaining system operation even under Byzantine failures. Experimental results demonstrate that our approach successfully validates medical messages with cryptographic signatures, prevents replay attacks through timestamp validation, and maintains consensus accuracy of 100% with up to 33% Byzantine nodes. The system provides real-time visualization of consensus rounds, vote tallies, and network topology, enabling transparent monitoring of fault-tolerant operations. This work contributes a practical framework for building secure, resilient healthcare multi-agent systems capable of collaborative medical decision-making in untrusted environments. 

---
# Graph-O1 : Monte Carlo Tree Search with Reinforcement Learning for Text-Attributed Graph Reasoning 

**Authors**: Lihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.17912)  

**Abstract**: ChatGPT said: Text-attributed graphs, where nodes and edges contain rich textual information, are widely used across diverse domains. A central challenge in this setting is question answering, which requires jointly leveraging unstructured text and the structured relational signals within the graph. Although Large Language Models (LLMs) have made significant advances in natural language understanding, their direct use for reasoning over text-attributed graphs remains limited. Retrieval-augmented generation methods that operate purely on text often treat passages as isolated units, ignoring the interconnected structure of the graph. Conversely, graph-based RAG methods that serialize large subgraphs into long textual sequences quickly become infeasible due to LLM context-length constraints, resulting in fragmented reasoning and degraded accuracy. To overcome these limitations, we introduce Graph-O1, an agentic GraphRAG framework that enables LLMs to conduct stepwise, interactive reasoning over graphs. Our approach integrates Monte Carlo Tree Search (MCTS) with end-to-end reinforcement learning, allowing the model to selectively explore and retrieve only the most informative subgraph components. The reasoning procedure is framed as a multi-turn interaction between the agent and the graph environment, and the agent is trained through a unified reward mechanism. Extensive experiments across multiple LLM backbones demonstrate that Graph-O1 consistently surpasses state-of-the-art baselines, producing answers that are more accurate, reliable, and interpretable. 

---
# Towards Reasoning-Preserving Unlearning in Multimodal Large Language Models 

**Authors**: Hongji Li, Junchi yao, Manjiang Yu, Priyanka Singh, Xue Li, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.17911)  

**Abstract**: Machine unlearning aims to erase requested data from trained models without full retraining. For Reasoning Multimodal Large Language Models (RMLLMs), this is uniquely challenging: intermediate chain-of-thought steps can still leak sensitive information even when final answers are forgotten, and overly aggressive interventions easily damage general reasoning ability. Yet no benchmark jointly evaluates how well unlearning methods suppress reasoning-level leakage while preserving reasoning competence. We address this gap with RMLLMU-Bench, the first benchmark for RMLLM unlearning that extends standard forgetting metrics with dedicated measures of reasoning leakage and reasoning retention. A systematic evaluation on RMLLMU-Bench reveals that existing unlearning methods for MLLMs and Large (Language) Reasoning Models (LRMs) either leave substantial leakage in the reasoning process or severely degrade reasoning performance. To address these gaps, we propose R-MUSE (Reasoning-preserving MLLM Unlearning via Subspace guidance and Adaptive Steering), a training-free and inference-time intervention framework that steers internal representations to forget both answers and reasoning traces while explicitly preserving general reasoning. Experiments on RMLLMU-Bench demonstrate that R-MUSE achieves a substantially better balance between effective forgetting and reasoning retention. 

---
# Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA 

**Authors**: Allison Li, Kristjan Greenewald, Thomas Parnell, Navid Azizan  

**Link**: [PDF](https://arxiv.org/pdf/2512.17910)  

**Abstract**: Modern large language model (LLM) systems increasingly rely on multi-turn pipelines that are composed of multiple task-specific adapters, yet existing serving frameworks remain inefficient, incurring substantial recomputation overhead when switching between adapters. We present the first LLM serving engine that supports cross-model prefix cache reuse between base and adapted models via Activated LoRA (aLoRA), enabling efficient and fine-grained adapter switching during inference. Our design extends the vLLM framework by introducing base-aligned block hashing and activation-aware masking within the model execution path, permitting cache reuse across models while preserving compatibility with existing serving engine optimizations. Integrated into a production-grade inference stack, this approach supports dynamic adapter activation without excessive key-value tensor recomputation. Evaluation across representative multi-turn, multi-adapter pipelines demonstrates up to 58x end-to-end latency reduction and over 100x time-to-first-token improvement relative to standard LoRA baselines, with benefits that scale with model size and sequence length and manifest across all stages of the request lifecycle. This work bridges parameter-efficient model adaptation with high-performance serving, providing the first complete realization of cross-model KV-cache reuse in modern LLM inference engines. 

---
