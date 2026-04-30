# Bian Que: An Agentic Framework with Flexible Skill Arrangement for Online System Operations 

**Authors**: Bochao Liu, Zhipeng Qian, Yang Zhao, Xinyuan Jiang, Zihan Liang, Yufei Ma, Junpeng Zhuang, Ben Chen, Shuo Yang, Hongen Wan, Yao Wu, Chenyi Lei, Xiao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26805)  

**Abstract**: Operating and maintaining (O&M) large-scale online engine systems (search, recommendation, advertising) demands substantial human effort for release monitoring, alert response, and root cause analysis. While LLM-based agents are a natural fit for these tasks, the deployment bottleneck is not reasoning capability but orchestration: selecting, for each operational event, the relevant data (metrics, logs, change events) and the applicable operational knowledge (handbook rules and practitioner experience). Feeding all signals indiscriminately causes dilution and hallucination, while manually curating the event-to-(data, knowledge) mapping is intractable under dozens of daily releases. We present Bian Que, an agentic framework with three contributions: (i) a \emph{unified operational paradigm} abstracting day-to-day O&M into three canonical patterns: release interception, proactive inspection, and alert root cause analysis; (ii) \emph{Flexible Skill Arrangement}, where each Skill specifies which data and knowledge to retrieve for a given business-module context and can be automatically generated and updated by LLMs or iteratively refined through natural-language instructions from on-call engineers; (iii) a \emph{unified self-evolving mechanism} in which one correction signal drives two parallel pathways, case-memory-to-knowledge distillation and targeted Skill refinement. Deployed on the e-commerce search engine of KuaiShou, the major short-video platform in China, Bian Que reduces alert volume by 75%, achieves 80% root-cause analysis accuracy, and cuts mean time to resolution by over 50%. Our framework achieves 99.0% pass rate on offline evaluations. Our code is available at this https URL. 

---
# FutureWorld: A Live Environment for Training Predictive Agents with Real-World Outcome Rewards 

**Authors**: Zhixin Han, Yanzhi Zhang, Chuyang Wei, Maohang Gao, Xiawei Yue, Kefei Chen, Yu Zhuang, Haoxiang Guan, Jiyan He, Jian Li, Yitong Duan, Yu Shi, Mengting Hu, Shuxin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.26733)  

**Abstract**: Live future prediction refers to the task of making predictions about real-world events before they unfold. This task is increasingly studied using large language model-based agent systems, and it is important for building agents that can continually learn from real-world. Just as interactive environments have often driven progress in agents, advancing live future prediction naturally motivates viewing it as a learning environment. Prior works have explored future prediction from several different parts, but have generally not framed it as a unified learning environment. This task is appealing for learning because it can provide a large number of prediction questions grounded in diverse real-world events, while preventing answer leakage. To leverage the advantages of live future prediction, we present FutureWorld, a live agentic reinforcement learning environment that closes the training loop between prediction, outcome realization, and parameters update. In our environment, we take three open-source base models and train them for consecutive days. The results show that training is effective. Furthermore, we build a daily benchmark based on the environment and evaluate several frontier agents on it to establish performance baselines for current agent systems. 

---
# SciHorizon-DataEVA: An Agentic System for AI-Readiness Evaluation of Heterogeneous Scientific Data 

**Authors**: Dianyu Liu, Chuan Qin, Xi Chen, Xiaohan Li, Wenxi Xu, Yuyang Wang, Xin Chen, Yuanchun Zhou, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26645)  

**Abstract**: AI-for-Science (AI4Science) is increasingly transforming scientific discovery by embedding machine learning models into prediction, simulation, and hypothesis generation workflows across domains. However, the effectiveness of these models is fundamentally constrained by the AI-readiness of scientific data, for which no scalable and systematic evaluation mechanism currently exists. In this work, we propose SciHorizon-DataEVA, a novel agentic system to scalable AI-readiness evaluation of heterogeneous scientific data. At the evaluation-criteria level, we introduce the Sci-TQA2 principles, which organize AI-readiness into four complementary dimensions: Governance Trustworthiness, Data Quality, AI Compatibility, and Scientific Adaptability. Each dimension is decomposed into measurable atomic elements that enable fine-grained and executable assessment. To operationalize these principles at scale, we develop Sci-TQA2-Eval, a hierarchical multi-agent evaluation approach orchestrated through a directed, cyclic workflow. Our Sci-TQA2-Eval dynamically constructs dataset-aware evaluation specifications by combining lightweight dataset profiling, applicability-aware metric activation, and knowledge-augmented planning grounded in domain constraints and dataset-paper signals. These specifications are executed through an adaptive, tool-centric evaluation mechanism with built-in verification and self-correction, enabling scalable and reliable assessment across heterogeneous scientific data. Extensive experiments on scientific datasets spanning multiple domains demonstrate the effectiveness and generality of SciHorizon-DataEVA for principled AI-readiness evaluation. 

---
# When to Vote, When to Rewrite: Disagreement-Guided Strategy Routing for Test-Time Scaling 

**Authors**: Zhimin Lin, Yixin Ji, Jinpeng Li, Yu Luo, Dong Li, Junhua Fang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26644)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on mathematical reasoning tasks but remain unreliable on challenging instances. Existing test-time scaling methods, such as repeated sampling, self-correction, and tree search, improve performance at the cost of increased computation, yet often exhibit diminishing returns on hard problems. We observe that output disagreement is strongly correlated with instance difficulty and prediction correctness, providing a useful signal for guiding instance-level strategy selection at test time. Based on this insight, we propose a training-free framework that formulates test-time scaling as an instance-level routing problem, rather than allocating more computation within a single strategy, dynamically selecting among different scaling strategies based on output disagreement. The framework applies lightweight resolution for consistent cases, majority voting for moderate disagreement, and rewriting-based reformulation for highly ambiguous instances. Experiments on seven mathematical benchmarks and three models show that our method improves accuracy by 3% - 7% while reducing sampling cost compared to existing approaches. 

---
# Human-in-the-Loop Benchmarking of Heterogeneous LLMs for Automated Competency Assessment in Secondary Level Mathematics 

**Authors**: Jatin Bhusal, Nancy Mahatha, Aayush Acharya, Raunak Regmi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26607)  

**Abstract**: As Competency-Based Education (CBE) is gaining traction around the world, the shift from marks-based assessment to qualitative competency mapping is a manual challenge for educators. This paper tackles the bottleneck issue by suggesting a "Human-in-the-Loop" benchmarking framework to assess the effectiveness of multiple LLMs in automating secondary-level mathematics assessment. Based on the Grade 10 Optional Mathematics curriculum in Nepal, we created a multi-dimensional rubric for four topics and four cross-cutting competencies: Comprehension, Knowledge, Operational Fluency, and Behavior and Correlation.
The multi-provider ensemble, consisted of open-weight models -- Eagle (Llama 3.1-8B) and Orion (Llama 3.3-70B) -- and proprietary frontier models Nova (Gemini 2.5 Flash) and Lyra (Gemini 3 Pro), was benchmarked against a ground truth defined by two senior mathematics faculty members (kappa_w = 0.8652). The findings show a marked "Architecture-compatibility gap". Although the Gemini-based Mixture-of-Experts (Sparse MoE) models achieved "Fair Agreement" (kappa_w ~ 0.38), the larger Orion (70B) model exhibited "No Agreement" (kappa_w = -0.0261), suggesting that architectural compliance with instruction constraints outweighs the scale of raw parameters in rubric-constrained tasks. We conclude that while LLMs are not yet suitable for autonomous certification, they provide high-value assistive support for preliminary evidence extraction within a "Human-in-the-Loop" framework. 

---
# Benchmarking the Safety of Large Language Models for Robotic Health Attendant Control 

**Authors**: Mahiro Nakao, Kazuhiro Takemoto  

**Link**: [PDF](https://arxiv.org/pdf/2604.26577)  

**Abstract**: Large language models (LLMs) are increasingly considered for deployment as the control component of robotic health attendants, yet their safety in this context remains poorly characterized. We introduce a dataset of 270 harmful instructions spanning nine prohibited behavior categories grounded in the American Medical Association Principles of Medical Ethics, and use it to evaluate 72 LLMs in a simulation environment based on the Robotic Health Attendant framework. The mean violation rate across all models was 54.4\%, with more than half exceeding 50\%, and violation rates varied substantially across behavior categories, with superficially plausible instructions such as device manipulation and emergency delay proving harder to refuse than overtly destructive ones. Model size and release date were the primary determinants of safety performance among open-weight models, and proprietary models were substantially safer than open-weight counterparts (median 23.7\% versus 72.8\%). Medical domain fine-tuning conferred no significant overall safety benefit, and a prompt-based defense strategy produced only a modest reduction in violation rates among the least safe models, leaving absolute violation rates at levels that would preclude safe clinical deployment. These findings demonstrate that safety evaluation must be treated as a first-class criterion in the development and deployment of LLMs for robotic health attendants. 

---
# AGEL-Comp: A Neuro-Symbolic Framework for Compositional Generalization in Interactive Agents 

**Authors**: Mahnoor Shahid, Hannes Rothe  

**Link**: [PDF](https://arxiv.org/pdf/2604.26522)  

**Abstract**: Large Language Model (LLM)-based agents exhibit systemic failures in compositional generalization, limiting their robustness in interactive environments. This work introduces AGEL-Comp, a neuro-symbolic AI agent architecture designed to address this challenge by grounding actions of the agent. AGEL-Comp integrates three core innovations: (1) a dynamic Causal Program Graph (CPG) as a world model, representing procedural and causal knowledge as a directed hypergraph; (2) an Inductive Logic Programming (ILP) engine that synthesizes new Horn clauses from experiential feedback, grounding symbolic knowledge through interaction; and (3) a hybrid reasoning core where an LLM proposes a set of candidate sub-goals that are verified for logical consistency by a Neural Theorem Prover (NTP). Together, these components operationalize a deduction--abduction learning cycle: enabling the agent to deduce plans and abductively expand its symbolic world model, while a neural adaptation phase keeps its reasoning engine aligned with new knowledge. We propose an evaluation protocol within the \texttt{Retro Quest} simulation environment to probe for compositional generalization scenarios to evaluate our AGEL agent. Our findings clearly indicate the better performance of our AGEL model over pure LLM-based models. Our framework presents a principled path toward agents that build an explicit, interpretable, and compositionally structured understanding of their world. 

---
# Grounding vs. Compositionality: On the Non-Complementarity of Reasoning in Neuro-Symbolic Systems 

**Authors**: Mahnoor Shahid, Hannes Rothe  

**Link**: [PDF](https://arxiv.org/pdf/2604.26521)  

**Abstract**: Compositional generalization remains a foundational weakness of modern neural networks, limiting their robustness and applicability in domains requiring out-of-distribution reasoning. A central, yet unverified, assumption in neuro-symbolic AI is that compositional reasoning will emerge as a byproduct of successful symbol grounding. This work presents the first systematic empirical analysis to challenge this assumption by disentangling the contributions of grounding and reasoning. To operationalize this investigation, we introduce the Iterative Logic Tensor Network ($i$LTN), a fully differentiable architecture designed for multi-step deduction. Using a formal taxonomy of generalization -- probing for novel entities, unseen relations, and complex rule compositions -- we demonstrate that a model trained solely on a grounding objective fails to generalize. In contrast, our full $i$LTN, trained jointly on perceptual grounding and multi-step reasoning, achieves high zero-shot accuracy across all tasks. Our findings provide conclusive evidence that symbol grounding, while necessary, is insufficient for generalization, establishing that reasoning is not an emergent property but a distinct capability that requires an explicit learning objective. 

---
# Auto-Relational Reasoning 

**Authors**: Ioannis Konstantoulas, Dimosthenis Tsimas, Pavlos Peppas, Kyriakos Sgarbas  

**Link**: [PDF](https://arxiv.org/pdf/2604.26507)  

**Abstract**: Background & Objectives: In the last decade, Machine learning research has grown rapidly, but large models are reaching their soft limits demonstrating diminishing returns and still lack solid reasoning abilities. These limits could be surpassed through synergistic combination of Machine Learning scalability and rigid reasoning. Methods: In this work, we propose a theoretical framework for reasoning through object-relations in an automated manner integrated with Artificial Neural Networks. We present a formal analysis of the Reasoning, and we show the theory in practice through a paradigm integrating Reasoning and Machine Learning. Results: This paradigm is a system that solves Intelligence Quotient problems without any prior knowledge of the problem. Our system achieves 98.03% solving rate corresponding to the top 1% percentile or 132-144 iq score. This result is only limited by the small size of the model and the processing capabilities of the machine it run on. Conclusions: With the integration of prior knowledge in the system and the expansion of the dataset, the system can be generalized to solve a large category of problems. The functionality of the system inherently favors the solution of such problems in few-shot or zero-shot attempts. 

---
# DreamProver: Evolving Transferable Lemma Libraries via a Wake-Sleep Theorem-Proving Agent 

**Authors**: Youyuan Zhang, Jialiang Sun, Hangrui Bi, Chuqin Geng, Wenjie Ma, Zhaoyu Li, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2604.26311)  

**Abstract**: We introduce DreamProver, an agentic framework that leverages a "wake-sleep" program induction paradigm to discover reusable lemmas for formal theorem proving. Existing approaches either rely on fixed lemma libraries, which limit adaptability, or synthesize highly specific intermediate lemmas tailored to individual theorems, thereby lacking generality. DreamProver addresses this gap through an iterative two-stage process. In the wake stage, DreamProver attempts to prove theorems from a training set using the current lemma library while proposing new candidate lemmas. In the "sleep" stage, it abstracts, refines, and consolidates these candidates to compress and optimize the library. Through this alternating cycle, DreamProver progressively evolves a compact set of high-level, transferable lemmas that can be effectively used to prove unseen theorems in related domains. Experimental results demonstrate that DreamProver substantially improves proof success rates across a diverse set of mathematical benchmarks, while also producing more concise proofs and reducing computational cost. 

---
# Apriori-based Analysis of Learned Helplessness in Mathematics Tutoring: Behavioral Patterns by Level, Intervention, and Outcome 

**Authors**: John Paul P. Miranda  

**Link**: [PDF](https://arxiv.org/pdf/2604.26237)  

**Abstract**: This study applied the Apriori algorithm to analyze behavioral interaction patterns associated with learned helplessness (LH) in mathematics tutoring system logs. Interaction data were examined across three dimensions: LH level (low vs. high), system-based intervention (with vs. without), and problem-solving outcomes (solved vs. unsolved). The analysis of the complete dataset showed that skipping problems without using hints was the most frequent pattern linked to unsolved outcomes, while persistence behaviors such as not skipping were less dominant overall. Comparisons by LH level showed that low-LH students had stronger links between problem solving and not skipping, as well as positive associations between hint use and solved outcomes. High-LH students showed more avoidance patterns, with skipping strongly tied to unsolved outcomes. In the comparison of system-based intervention conditions, students without intervention had the highest lift for persistence-success links, while the with-intervention group had stronger patterns involving skipping behaviors leading to unsolved outcomes. Outcome-specific analysis showed that not skipping was consistently associated with solved problems across all groups, while skipping without hints predicted unsolved outcomes. Practical implications and recommendations are discussed. 

---
# Persuadability and LLMs as Legal Decision Tools 

**Authors**: Oisin Suttle, David Lillis  

**Link**: [PDF](https://arxiv.org/pdf/2604.26233)  

**Abstract**: As Large Language Models (LLMs) are proposed as legal decision assistants, and even first-instance decision-makers, across a range of judicial and administrative contexts, it becomes essential to explore how they answer legal questions, and in particular the factors that lead them to decide difficult questions in one way or another. A specific feature of legal decisions is the need to respond to arguments advanced by contending parties. A legal decision-maker must be able to engage with, and respond to, including through being potentially persuaded by, arguments advanced by the parties. Conversely, they should not be unduly persuadable, influenced by a particularly compelling advocate to decide cases based on the skills of the advocates, rather than the merits of the case. We explore how frontier open- and closed-weights LLMs respond to legal arguments, reporting original experimental results examining how the quality of the advocate making those arguments affects the likelihood that a model will agree with a particular legal point of view, and exploring the factors driving these results. Our results have implications for the feasibility of adopting LLMs across legal and administrative settings. 

---
# OMEGA: Optimizing Machine Learning by Evaluating Generated Algorithms 

**Authors**: Jeremy Nixon, Annika Singh  

**Link**: [PDF](https://arxiv.org/pdf/2604.26211)  

**Abstract**: In order to automate AI research we introduce a full, end-to-end framework, OMEGA: Optimizing Machine learning by Evaluating Generated Algorithms, that starts at idea generation and ends with executable code. Our system combines structured meta-prompt engineering with executable code generation to create new ML classifiers. The OMEGA framework has been utilized to generate several novel algorithms that outperform scikit-learn baselines across a robust selection of 20 benchmark datasets (infinity-bench). You can access models discussed in this paper and more in the python package: pip install omega-models. 

---
# Hierarchical Multi-Persona Induction from User Behavioral Logs: Learning Evidence-Grounded and Truthful Personas 

**Authors**: Nayoung Choi, Haeyu Jeong, Changbong Kim, Hongjun Lim, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26120)  

**Abstract**: Behavioral logs provide rich signals for user modeling, but are noisy and interleaved across diverse intents. Recent work uses LLMs to generate interpretable natural-language personas from user logs, yet evaluation often emphasizes downstream utility, providing limited assurance of persona quality itself. We propose a hierarchical framework that aggregates user actions into intent memories and induces multiple evidence-grounded personas by clustering and labeling these memories. We formulate persona induction as an optimization problem over persona quality-captured by cluster cohesion, persona-evidence alignment, and persona truthfulness-and train the persona model using a groupwise extension of Direct Preference Optimization (DPO). Experiments on a large-scale service log and two public datasets show that our method induces more coherent, evidence-grounded, and trustworthy personas, while also improving future interaction prediction. 

---
# Evaluating Strategic Reasoning in Forecasting Agents 

**Authors**: Tom Liptay, Dan Schwarz, Rafael Poyiadzi, Jack Wildman, Nikos I. Bosse  

**Link**: [PDF](https://arxiv.org/pdf/2604.26106)  

**Abstract**: Forecasting benchmarks produce accuracy leaderboards but little insight into why some forecasters are more accurate than others. We introduce Bench to the Future 2 (BTF-2), 1,417 pastcasting questions with a frozen 15M-document research corpus in which agents reproducibly research and forecast offline, producing full reasoning traces. BTF-2 detects accuracy differences of 0.004 Brier score, and can distinguish differential agent strengths in research vs. judgment. We build a forecaster 0.011 Brier more accurate than any single frontier agent, and use it to evaluate agent strategic reasoning without hindsight bias. We find the better forecaster differs primarily in its pre-mortem analysis of its blind spots and consideration of black swans. Expert human forecasters found the dominant strategic reasoning failures of frontier agents are in assessing political and business leaders' incentives, judging their likelihood to follow through on stated plans, and modeling institutional processes. 

---
# Distill-Belief: Closed-Loop Inverse Source Localization and Characterization in Physical Fields 

**Authors**: Yiwei Shi, Zixing Song, Mengyue Yang, Cunjia Liu, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26095)  

**Abstract**: {Closed-loop inverse source localization and characterization (ISLC) requires a mobile agent to select measurements that localize sources and infer latent field parameters under strict time constraints.} {The core challenge lies in the belief-space objective: valid uncertainty estimation requires expensive Bayesian inference, whereas using fast learned belief model leads to reward hacking, in which the policy exploits approximation errors rather than actually reducing uncertainty.} {We propose \textbf{Distill-Belief}, a teacher--student framework that decouples correctness from efficiency. A Bayes-correct particle-filter teacher maintains the posterior and supplies a dense information-gain signal, while a compact student distills the posterior into belief statistics for control and an uncertainty certificate for stopping. At deployment, only the student is used, yielding constant per-step cost.} {Experiments on seven field modalities and two stress tests show that Distill-Belief consistently reduces sensing cost and improves success, posterior contraction, and estimation accuracy over baselines, while mitigating reward hacking.} 

---
# Operating-Layer Controls for Onchain Language-Model Agents Under Real Capital 

**Authors**: T.J. Barton, Chris Constantakis, Patti Hauseman, Annie Mous, Alaska Hoffman, Brian Bergeron, Hunter Goodreau  

**Link**: [PDF](https://arxiv.org/pdf/2604.26091)  

**Abstract**: We study reliability in autonomous language-model agents that translate user mandates into validated tool actions under real capital. The setting is DX Terminal Pro, a 21-day deployment in which 3,505 user-funded agents traded real ETH in a bounded onchain market. Users configured vaults through structured controls and natural-language strategies, but only agents could choose normal buy/sell trades. The system produced 7.5M agent invocations, roughly 300K onchain actions, about $20M in volume, more than 5,000 ETH deployed, roughly 70B inference tokens, and 99.9% settlement success for policy-valid submitted transactions. Long-running agents accumulated thousands of sequential decisions, including 6,000+ prompt-state-action cycles for continuously active agents, yielding a large-scale trace from user mandate to rendered prompt, reasoning, validation, portfolio state, and settlement. Reliability did not come from the base model alone; it emerged from the operating layer around the model: prompt compilation, typed controls, policy validation, execution guards, memory design, and trace-level observability. Pre-launch testing exposed failures that text-only benchmarks rarely measure, including fabricated trading rules, fee paralysis, numeric anchoring, cadence trading, and misread tokenomics. Targeted harness changes reduced fabricated sell rules from 57% to 3%, reduced fee-led observations from 32.5% to below 10%, and increased capital deployment from 42.9% to 78.0% in an affected test population. We show that capital-managing agents should be evaluated across the full path from user mandate to prompt, validated action, and settlement. 

---
# Turning the TIDE: Cross-Architecture Distillation for Diffusion Large Language Models 

**Authors**: Gongbo Zhang, Wen Wang, Ye Tian, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.26951)  

**Abstract**: Diffusion large language models (dLLMs) offer parallel decoding and bidirectional context, but state-of-the-art dLLMs require billions of parameters for competitive performance. While existing distillation methods for dLLMs reduce inference steps within a single architecture, none address cross-architecture knowledge transfer, in which the teacher and student differ in architecture, attention mechanism, and tokenizer. We present TIDE, the first framework for cross-architecture dLLM distillation, comprising three modular components: (1) TIDAL, which jointly modulates distillation strength across training progress and diffusion timestep to account for the teacher's noise-dependent reliability; (2) CompDemo, which enriches the teacher's context via complementary mask splitting to improve predictions under heavy masking; and (3) Reverse CALM, a cross-tokenizer objective that inverts chunk-level likelihood matching, yielding bounded gradients and dual-end noise filtering. Distilling 8B dense and 16B MoE teachers into a 0.6B student via two heterogeneous pipelines outperforms the baseline by an average of 1.53 points across eight benchmarks, yielding notable gains in code generation, where HumanEval scores reach 48.78 compared to 32.3 for the AR baseline. 

---
# Causal Learning with Neural Assemblies 

**Authors**: Evangelia Kopadi, Dimitris Kalles  

**Link**: [PDF](https://arxiv.org/pdf/2604.26919)  

**Abstract**: Can Neural Assemblies -- groups of neurons that fire together and strengthen through co-activation -- learn the direction of causal influence between variables? While established as a computationally general substrate for classification, parsing, and planning, neural assemblies have not yet been shown to internalize causal directionality. We demonstrate that the inherent operations of neural assemblies -- projection, local plasticity control, and sparse winner selection -- are sufficient for directional learning. We introduce DIRECT (DIRectional Edge Coupling/Training), a mechanism that co-activates source and target assemblies under an adaptive gain schedule to internalize directed relations. Unlike backpropagation-based methods, DIRECT relies solely on local plasticity, making the resulting causal claims auditable at the mechanism level. Our findings are verified through a dual-readout validation strategy: (i) synaptic-strength asymmetry, measuring the emergent weight gap between forward and reverse links, and (ii) functional propagation overlap, quantifying the reliability of directional signal flow. Across multiple domains, the framework achieves perfect structural recovery under a supervised, known-structure setting. These results establish neural assemblies as an auditable bridge between biologically plausible dynamics and formal causal models, offering an "explainable by design" framework where causal claims are traceable to specific neural winners and synaptic asymmetries. 

---
# ClawGym: A Scalable Framework for Building Effective Claw Agents 

**Authors**: Fei Bai, Huatong Song, Shuang Sun, Daixuan Cheng, Yike Yang, Chuan Hao, Renyuan Li, Feng Chang, Yuan Wei, Ran Tao, Bryan Dai, Jian Yang, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.26904)  

**Abstract**: Claw-style environments support multi-step workflows over local files, tools, and persistent workspace states. However, scalable development around these environments remains constrained by the absence of a systematic framework, especially one for synthesizing verifiable training data and integrating it with agent training and diagnostic evaluation. To address this challenge, we present ClawGym, a scalable framework that supports the full lifecycle of Claw-style personal agent development. Concretely, we construct ClawGym-SynData, a diverse dataset of 13.5K filtered tasks synthesized from persona-driven intents and skill-grounded operations, paired with realistic mock workspaces and hybrid verification mechanisms. We then train a family of capable Claw-style models, termed ClawGym-Agents, through supervised fine-tuning on black-box rollout trajectories, and further explore reinforcement learning via a lightweight pipeline that parallelizes rollouts across per-task this http URL support reliable evaluation, we further construct ClawGym-Bench, a benchmark of 200 instances calibrated through automated filtering and human-LLM review. Relevant resources will be soon released at this https URL. 

---
# Recent Advances in mm-Wave and Sub-THz/THz Oscillators for FutureG Technologies 

**Authors**: Baktash Behmanesh, Ahmad Rezvanitabar  

**Link**: [PDF](https://arxiv.org/pdf/2604.26903)  

**Abstract**: This paper provides a concise yet comprehensive review of recent advancements in millimeter-wave (mm-wave) oscillators below 100 GHz and sub-terahertz (sub-THz/THz) oscillators above 100 GHz for next-generation computing and communication systems, including 5G, 6G, and beyond. Various design approaches, including CMOS, SiGe, and III-V semiconductor technologies, are explored in terms of performance metrics such as phase noise, output power, efficiency, frequency tunability, and stability. The review highlights key challenges in achieving high-performance and reliable oscillator designs while discussing emerging techniques for performance enhancement. By evaluating recent design trends, this work aims to offer valuable insights and design guidelines that facilitate the development of robust mm-wave and sub-THz/THz oscillators for future communication, computing, and sensing applications. 

---
# Resume-ing Control: (Mis)Perceptions of Agency Around GenAI Use in Recruiting Workflows 

**Authors**: Sajel Surati, Rosanna Bellini, Emily Black  

**Link**: [PDF](https://arxiv.org/pdf/2604.26851)  

**Abstract**: When generative AI (genAI) systems are used in high-stakes decision-making, its recommended role is to aid, rather than replace, human decision-making. However, there is little empirical exploration of how professionals making high-stakes decisions, such as those related to employment, perceive their agency and level of control when working with genAI systems. Through interviews with 22 recruiting professionals, we investigate how genAI subtly influences control over everyday workflows and even individual hiring decisions. Our findings highlight a pressing conflict: while recruiters believe they have final authority across the recruiting pipeline, genAI has become an invisible architect that shapes the foundational building blocks of information used for evaluation, from defining a job to determining good interview performances. The decision of whether or not to adopt was also often outside recruiters' control, with many feeling compelled to adopt genAI due to calls to integrate AI from higher-ups in their business, to combat applicant use of AI, and the individual need to boost productivity. Despite a seemingly seismic shift in how recruiting happens, participants only reported marginal efficiency gains. Such gains came at the high cost of recruiter deskilling, a trend that jeopardizes the meaningful oversight of decision-making. We conclude by discussing the implications of such findings for responsible and perceptible genAI use in hiring contexts. 

---
# Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data 

**Authors**: Bao Pham, Mohammed J. Zaki, Luca Ambrogioni, Dmitry Krotov, Matteo Negri  

**Link**: [PDF](https://arxiv.org/pdf/2604.26841)  

**Abstract**: When do language diffusion models memorize their training data, and how to quantitatively assess their true generative regime? We address these questions by showing that Uniform-based Discrete Diffusion Models (UDDMs) fundamentally behave as Associative Memories (AMs) $\textit{with emergent creative capabilities}$. The core idea of an AM is to reliably recover stored data points as $\textit{memories}$ by establishing distinct basins of attraction around them. Historically, models like Hopfield networks use an explicit energy function to guarantee these stable attractors. We broaden this perspective by leveraging the observation that energy is not strictly necessary, as basins of attraction can also be formed via conditional likelihood maximization. By evaluating token recovery of $\textit{training}$ and $\textit{test}$ examples, we identify in UDDMs a sharp memorization-to-generalization transition governed by the size of the training dataset: as it increases, basins around training examples shrink and basins around unseen test examples expand, until both later converge to the same level. Crucially, we can detect this transition using only the conditional entropy of predicted token sequences: memorization is characterized by vanishing conditional entropy, while in the generalization regime the conditional entropy of most tokens remains finite. Thus, conditional entropy offers a practical probe for the memorization-to-generalization transition in deployed models. 

---
# HalluCiteChecker: A Lightweight Toolkit for Hallucinated Citation Detection and Verification in the Era of AI Scientists 

**Authors**: Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.26835)  

**Abstract**: We introduce HalluCiteChecker, a toolkit for detecting and verifying hallucinated citations in scientific papers. While AI assistant technologies have transformed the academic writing process, including citation recommendation, they have also led to the emergence of hallucinated citations that do not correspond to any existing work. Such citations not only undermine the credibility of scientific papers but also impose an additional burden on reviewers and authors, who must manually verify their validity during the review process. In this study, we formalize hallucinated citation detection as an NLP task and provide a corresponding toolkit as a practical foundation for addressing this problem. Our package is lightweight and can perform verification in seconds on a standard laptop. It can also be executed entirely offline and runs efficiently using only CPUs. We hope that HalluCiteChecker will help reduce reviewer workload and support organizers by enabling systematic pre-review and publication checks. Our code is released under the Apache 2.0 license on GitHub and is distributed as an installable package via PyPI. A demonstration video is available on YouTube. 

---
# Rule-based High-Level Coaching for Goal-Conditioned Reinforcement Learning in Search-and-Rescue UAV Missions Under Limited-Simulation Training 

**Authors**: Mahya Ramezani, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2604.26833)  

**Abstract**: This paper presents a hierarchical decision-making framework for unmanned aerial vehicle (UAV) missions motivated by search-and-rescue (SAR) scenarios under limited simulation training. The framework combines a fixed rule-based high-level advisor with an online goal-conditioned low-level reinforcement learning (RL) controller. To stress-test early adaptation, we also consider a strict no-pretraining deployment regime. The high-level advisor is defined offline from a structured task specification and compiled into deterministic rules. It provides interpretable mission- and safety-aware guidance through recommended actions, avoided actions, and regime-dependent arbitration weights. The low-level controller learns online from task-defined dense rewards and reuses experience through a mode-aware prioritized replay mechanism augmented with rule-derived metadata. We evaluate the framework on two tasks: battery-aware multi-goal delivery and moving-target delivery in obstacle-rich environments. Across both tasks, the proposed method improves early safety and sample efficiency primarily by reducing collision terminations, while preserving the ability to adapt online to scenario-specific dynamics. 

---
# Random Cloud: Finding Minimal Neural Architectures Without Training 

**Authors**: Javier Gil Blázquez  

**Link**: [PDF](https://arxiv.org/pdf/2604.26830)  

**Abstract**: I propose the \emph{Random Cloud} method, a training-free approach to neural architecture search that discovers minimal feedforward network topologies through stochastic exploration and progressive structural reduction. Unlike post-training pruning methods that require a full train-prune-retrain cycle, this method evaluates randomly initialized networks without backpropagation, progressively reduces their topology, and only trains the best minimal candidate at the end. I evaluate on 7 classification benchmarks against magnitude pruning and random pruning baselines. The Random Cloud matches or outperforms both baselines in 6 of 7 datasets, achieving statistically significant improvements on Sonar ($+4.9$pp accuracy, $p{=}0.017$ vs magnitude pruning) with 87\% parameter reduction. Crucially, the method is faster than both pruning baselines in 4 of 5 datasets (0.67--0.94$\times$ the cost of full training), since it avoids training the full-size network entirely. 

---
# ViCrop-Det: Spatial Attention Entropy Guided Cropping for Training-Free Small-Object Detection 

**Authors**: Hui Wang, Hongze Li, Wei Chen, Xiaojin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26806)  

**Abstract**: Transformer-based architectures have established a dominant paradigm in global semantic perception; however, they remain fundamentally constrained by the profound spatial heterogeneity inherent in natural images. Specifically, the imposition of a uniform global receptive field across regions of varying information density inevitably leads to local feature degradation, particularly in dense conflict zones populated by microscopic targets. To address this mechanistic limitation, we propose ViCrop-Det, a training-free inference framework that introduces adaptive spatial trust region shrinkage. Inspired by the use of attention entropy in anomaly segmentation, ViCrop-Det leverages the detection decoder's cross-attention distribution as an endogenous probe. By utilizing Spatial Attention Entropy (SAE) to heuristically evaluate local spatial ambiguity, the framework executes dynamic spatial routing, allocating a fixed computational budget exclusively to regions exhibiting both high target saliency and high cognitive uncertainty. By shrinking the spatial trust region and injecting high-frequency localized observations, ViCrop-Det actively resolves spatial ambiguity and recovers fine-grained features without requiring architectural modifications. Extensive evaluations on VisDrone and DOTA-v1.5 demonstrate that ViCrop-Det yields competitive performance enhancements, consistently adding +1-3 mAP@50 to RT-DETR-R50 and Deformable DETR with a marginal 20-23\% latency overhead. On MS COCO, $AP_{S}$ improves while $AP_{M}/AP_{L}$ remains stable, indicating precise fine-scale refinement without compromising the global spatial prior. Under compute-matched settings, our adaptive routing strategy comprehensively surpasses uniform slicing baselines, achieving a highly optimized accuracy-speed trade-off. 

---
# MemOVCD: Training-Free Open-Vocabulary Change Detection via Cross-Temporal Memory Reasoning and Global-Local Adaptive Rectification 

**Authors**: Zuzheng Kuang, Honghao Chang, Boqiang Liang, Haoqian Wang, Lijun He, Fan Li, Haixia Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26774)  

**Abstract**: Open-vocabulary change detection aims to identify semantic changes in bi-temporal remote sensing images without predefined categories. Recent methods combine foundation models such as SAM, DINO and CLIP, but typically process each timestamp independently or interact only at the final comparison stage. Such paradigms suffer from insufficient temporal coupling during semantic reasoning, which limits their ability to distinguish genuine semantic changes from non-semantic appearance discrepancies. In addition, patch-dominant inference on high-resolution images often weakens global semantic continuity and produces fragmented change regions. To address these issues, we propose MemOVCD, a training-free open-vocabulary change detection framework based on cross-temporal memory reasoning and global-local adaptive rectification. Specifically, we reformulate bi-temporal change detection as a two-frame tracking problem and introduce weighted bidirectional propagation to aggregate semantic evidence from both temporal directions. To stabilize memory propagation across large temporal gaps, we construct histogram-aligned transition frames to smooth abrupt appearance changes. Moreover, a global-local adaptive rectification strategy adaptively fuses local and global-view predictions, improving spatial consistency while preserving fine-grained details. Experiments on five benchmarks demonstrate that MemOVCD achieves favorable performance on two change detection tasks, validating its effectiveness and generalization under diverse open-vocabulary settings. 

---
# Domain-Adapted Small Language Models for Reliable Clinical Triage 

**Authors**: Manar Aljohani, Brandon Ho, Kenneth McKinley, Dennis Ren, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26766)  

**Abstract**: Accurate and consistent Emergency Severity Index (ESI) assignment remains a persistent challenge in emergency departments, where highly variable free-text triage documentation contributes to mistriage and workflow inefficiencies. This study evaluates whether open-source small language models (SLMs) can serve as reliable, privacy-preserving decision-support tools for clinical triage. We systematically compared multiple SLMs across diverse prompting pipelines and found that clinical vignettes, concise summaries of triage narratives, yielded the most accurate predictions. The SLM, Qwen2.5-7B, demonstrated the strongest balance of accuracy, stability, and computational efficiency. Through large-scale domain adaptation using expert-curated and silver-standard pediatric triage data, fine-tuned Qwen2.5-7B models substantially reduced discordance and clinically significant errors, outperforming all baseline SLMs and advanced proprietary large language models (LLMs, e.g., GPT-4o). These findings highlight the feasibility of institution-specific SLMs for reliable, privacy-preserving ESI decision support and underscore the importance of targeted fine-tuning over more complex inference strategies. 

---
# Exploring the Potential of Probabilistic Transformer for Time Series Modeling: A Report on the ST-PT Framework 

**Authors**: Zhangzhi Xiong, Haoyi Wu, You Wu, Shuqi Gu, Kan Ren, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26762)  

**Abstract**: The Probabilistic Transformer (PT) establishes that the Transformer's self-attention plus its feed-forward block is mathematically equivalent to Mean-Field Variational Inference (MFVI) on a Conditional Random Field (CRF). Under this equivalence the Transformer ceases to be a black-box neural network and becomes a programmable factor graph: graph topology, factor potentials, and the message-passing schedule are all explicit and inspectable primitives that can be engineered. PT was originally developed for natural language and in this report we investigate its potential for time series. We first lift PT into the Spatial-Temporal Probabilistic Transformer (ST-PT) to repair PT's missing channel axis and weak per-step semantics, and adopt ST-PT as a shared cornerstone backbone. We then identify three distinct properties that PT/ST-PT offers as a factor-graph model and derive three Research Questions, one per property, that probe how each property can be exploited in time series: RQ1. The graph topology and potentials are direct programmable primitives. Can this be used to inject symbolic time-series priors into ST-PT through structural graph modifications, especially under data scarcity and noise? RQ2. The CRF's factor matrices are the operator's potentials. Can an external condition program these factor matrices on a per-sample basis, so that conditional generation becomes structural rather than feature-level modulation of a fixed one? RQ3. Each MFVI iteration is a Bayesian posterior update on the factor graph. Can this turn the latent transition of latent-space AutoRegressive (AR) forecasting from an opaque MLP into a principled posterior update, and can a CRF teacher distill its latents into the AR student to counter cumulative error? We give one empirical study per question. Together, these three studies position ST-PT as a programmable framework for time-series modeling. 

---
# A self-evolving agent for explainable diagnosis of DFT-experiment band-gap mismatch 

**Authors**: Yue Li, Bijun Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26703)  

**Abstract**: Standard density functional theory (DFT) routinely misclassifies the electronic ground state of correlated and structurally complex compounds, predicting metallic behaviour for materials that experiments report as semiconductors. Each such mismatch encodes a specific non-ideality -- magnetic ordering, electron correlation, an alternative polymorph, or a defect -- that the calculation excluded, but extracting that signal at scale has remained a manual exercise. Here we introduce XDFT, a closed-loop agent that diagnoses the mismatch automatically: it draws candidate hypotheses from a curated catalogue, executes the corresponding first-principles tests, and updates a global Bayesian posterior over hypothesis usefulness from each verdict. On a verified benchmark of 124 materials, XDFT identifies a resolving mechanism for 70 of 90 mismatch cases (78\%), an order of magnitude above a uniform-random baseline (19\%) and a static LLM ordering (20\%). The internal posterior aligns with empirical performance over the benchmark timeline, and resolved cases collapse into a tri-partite element-class taxonomy that we distil into a four-line static rule. Each diagnosed material is returned with a corrected protocol and a mechanistic attribution; failed cases are flagged as evidence-backed targets for experimental re-examination. 

---
# Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising 

**Authors**: Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26694)  

**Abstract**: We propose X-WAM, a Unified 4D World Model that unifies real-time robotic action execution and high-fidelity 4D world synthesis (video + 3D reconstruction) in a single framework, addressing the critical limitations of prior unified world models (e.g., UWM) that only model 2D pixel-space and fail to balance action efficiency and world modeling quality. To leverage the strong visual priors of pretrained video diffusion models, X-WAM imagines the future world by predicting multi-view RGB-D videos, and obtains spatial information efficiently through a lightweight structural adaptation: replicating the final few blocks of the pretrained Diffusion Transformer into a dedicated depth prediction branch for the reconstruction of future spatial information. Moreover, we propose Asynchronous Noise Sampling (ANS) to jointly optimize generation quality and action decoding efficiency. ANS applies a specialized asynchronous denoising schedule during inference, which rapidly decodes actions with fewer steps to enable efficient real-time execution, while dedicating the full sequence of steps to generate high-fidelity video. Rather than entirely decoupling the timesteps during training, ANS samples from their joint distribution to align with the inference distribution. Pretrained on over 5,800 hours of robotic data, X-WAM achieves 79.2% and 90.7% average success rate on RoboCasa and RoboTwin 2.0 benchmarks, while producing high-fidelity 4D reconstruction and generation surpassing existing methods in both visual and geometric metrics. 

---
# Atomic-Probe Governance for Skill Updates in Compositional Robot Policies 

**Authors**: Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.26689)  

**Abstract**: Skill libraries in deployed robotic systems are continually updated through fine-tuning, fresh demonstrations, or domain adaptation, yet existing typed-composition methods (BLADE, SymSkill, Generative Skill Chaining) treat the library as frozen at test time and do not analyze how composition outcomes change when a skill is replaced. We introduce a paired-sampling cross-version swap protocol on robosuite manipulation tasks to characterize this dimension of compositional skill learning. On a dual-arm peg-in-hole task we discover a dominant-skill effect: one ECM achieves 86.7% atomic success rate while every other ECM is at or below 26.7%, and whether this dominant ECM enters a composition shifts the success rate by up to +50pp. We characterize the boundary on a simpler pick task where all atomic policies saturate at 100% and the effect is undefined. Across three tasks we further find that off-policy behavioral distance metrics fail to identify the dominant ECM, ruling out the natural cheap predictor. We propose an atomic-quality probe and a Hybrid Selector combining per-skill probes (zero per-decision cost) with selective composition revalidation (full cost), and characterize its Pareto frontier on 144 skill-update decisions. On T6 the atomic-only probe sits 23pp below full revalidation (64.6% vs 87.5% oracle match) at zero per-decision cost; a Hybrid Selector with m=10 closes most of that gap to ~12pp at 46% of full-revalidation cost. On the cross-task average over 144 events, atomic-only is within 3pp of full revalidation under a mixed-oracle caveat. The atomic-quality probe is, to our knowledge, the first principled, deployment-ready primitive for skill-update governance in compositional robot policies. 

---
# A Toolkit for Detecting Spurious Correlations in Speech Datasets 

**Authors**: Lara Gauder, Pablo Riera, Andrea Slachevsky, Gonzalo Forno, Adolfo M. García, Luciana Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2604.26676)  

**Abstract**: We introduce a toolkit for uncovering spurious correlations between recording characteristics and target class in speech datasets. Spurious correlations may arise due to heterogeneous recording conditions, a common scenario for health-related datasets. When present both in the training and test data, these correlations result in an overestimation of the system performance -- a dangerous situation, specially in high-stakes application where systems are required to satisfy minimum performance requirements. Our toolkit implements a diagnostic method based on the detection of the target class using only the non-speech regions in the audio. Better than chance performance at this task indicates that information about the target class can be extracted from the non-speech regions, flagging the presence of spurious correlations. The toolkit is publicly available for research use. 

---
# From Black-Box Confidence to Measurable Trust in Clinical AI: A Framework for Evidence, Supervision, and Staged Autonomy 

**Authors**: Serhii Zabolotnii, Viktoriia Holinko, Olha Antonenko  

**Link**: [PDF](https://arxiv.org/pdf/2604.26671)  

**Abstract**: Trust in clinical artificial intelligence (AI) cannot be reduced to model accuracy, fluency of generation, or overall positive user impression. In medicine, trust must be engineered as a measurable system property grounded in evidence, supervision, and operational boundaries of AI autonomy. This article proposes a practical framework for trustworthy clinical AI built around three principles: evidence, supervision, and staged autonomy. Rather than replacing deterministic clinical logic wholesale with end-to-end black-box models, the proposed approach combines a deterministic core, a patient-specific AI assistant for contextual validation, a multi-tier model escalation mechanism, and a human supervision layer for verification, escalation, and risk control. We demonstrate that trust also depends on selective verification of clinically critical findings, bounded clinical context, disciplined prompt architecture, and careful evaluation on realistic cases. Classifier-driven modular prompting is examined as an incremental path to scaling clinical depth without sacrificing prompt performance and without waiting for complete rule-based coverage. To operationalize trust, a set of trust metrics is proposed, built on metrological principles -- measurement uncertainty, calibration, traceability -- enabling quantitative rather than subjective assessment of each architectural layer. In this perspective, trustworthy clinical AI emerges not as a property of an individual model, but as an architectural outcome of a system into which evidence trails, human oversight, tiered escalation, and graduated action rights are embedded from the outset. 

---
# When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26649)  

**Abstract**: Large reasoning models such as DeepSeek-R1 and OpenAI o1 generate extended chains of thought spanning thousands of tokens, yet their integration with retrieval-augmented generation (RAG) remains fundamentally misaligned. Current RAG systems optimize for providing context before reasoning begins, while reasoning models require evidence injection during multi-step inference chains. We introduce ReaLM-Retrieve, a reasoning-aware retrieval framework that addresses this mismatch through three key innovations: (1) a step-level uncertainty detector that identifies knowledge gaps at reasoning-step granularity rather than token or sentence level; (2) a retrieval intervention policy that learns when external evidence maximally benefits ongoing reasoning; and (3) an efficiency-optimized integration mechanism that reduces per-retrieval overhead by 3.2x compared to naive integration. Experiments on MuSiQue, HotpotQA, and 2WikiMultiHopQA demonstrate that ReaLM-Retrieve achieves on average 10.1% absolute improvement in answer F1 over standard RAG (range: 9.0-11.8% across the three benchmarks) while reducing retrieval calls by 47% compared to fixed-interval approaches like IRCoT (all improvements significant at p<0.01, paired bootstrap). On the challenging MuSiQue benchmark requiring 2-4 hop reasoning, our method achieves 71.2% F1 with an average of only 1.8 retrieval calls per question. Analysis shows that ReaLM-Retrieve also improves retrieval quality itself, achieving 81.3% Recall@5 with consistently higher precision and MRR than fixed-interval baselines on supporting evidence, establishing new state-of-the-art efficiency-accuracy trade-offs for reasoning-intensive retrieval tasks. 

---
# ATLAS: An Annotation Tool for Long-horizon Robotic Action Segmentation 

**Authors**: Sergej Stanovcic, Daniel Sliwowski, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.26637)  

**Abstract**: Annotating long-horizon robotic demonstrations with precise temporal action boundaries is crucial for training and evaluating action segmentation and manipulation policy learning methods. Existing annotation tools, however, are often limited: they are designed primarily for vision-only data, do not natively support synchronized visualization of robot-specific time-series signals (e.g., gripper state or force/torque), or require substantial effort to adapt to different dataset formats. In this paper, we introduce ATLAS, an annotation tool tailored for long-horizon robotic action segmentation. ATLAS provides time-synchronized visualization of multi-modal robotic data, including multi-view video and proprioceptive signals, and supports annotation of action boundaries, action labels, and task outcomes. The tool natively handles widely used robotics dataset formats such as ROS bags and the Reinforcement Learning Dataset (RLDS) format, and provides direct support for specific datasets such as REASSEMBLE. ATLAS can be easily extended to new formats via a modular dataset abstraction layer. Its keyboard-centric interface minimizes annotation effort and improves efficiency. In experiments on a contact-rich assembly task, ATLAS reduced the average per-action annotation time by at least 6% compared to ELAN, while the inclusion of time-series data improved temporal alignment with expert annotations by more than 2.8% and decreased boundary error fivefold compared to vision-only annotation tools. 

---
# SynSur: An end-to-end generative pipeline for synthetic industrial surface defect generation and detection 

**Authors**: Paul Julius Kühn, Mika Pommeranz, Arjan Kuijper, Saptarshi Neil Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2604.26633)  

**Abstract**: The bottleneck in learning-based industrial defect detection is often limited not by model capacity, but by the scarcity of labeled defect data: defects are rare, annotations are expensive, and collecting balanced training sets is slow. We present an end-to-end pipeline for synthetic defect generation and annotation, combining Vision-Language-Model-based prompts, LoRA-adapted diffusion, mask-guided inpainting, and sample filtering with automatic label derivation, and demonstrates the potential of real data with realistic synthetic samples to overcome data scarcity. The evaluation is conducted on, a challenging dataset of pitting defects on ball screw drives, and then on a subset of the Mobile phone screen surface defect segmentation dataset (MSD) dataset to test cross-domain transfer. Beyond downstream detector performance, we analyze key stages of the pipeline, including prompt construction, LoRA selection, and sample filtering with DreamSim and CLIPScore, to understand which synthetic samples are both realistic and useful. Experiments with YOLOv26, YOLOX, and LW-DETR show that synthetic-only training does not replace real data. When combined with real data, synthetic defects can preserve performance and yield modest gains in selected BSData training regimes. The MSD transfer study shows that the overall pipeline structure carries over to a second industrial inspection domain, while also highlighting the importance of domain-specific adaptation and annotation-quality control. Overall, the paper provides an end-to-end assessment of diffusion-based industrial defect synthesis and shows that its strongest value lies in strengthening scarce real datasets rather than substituting for them. 

---
# TDD Governance for Multi-Agent Code Generation via Prompt Engineering 

**Authors**: Tarlan Hasanli, Shahbaz Siddeeq, Bishwash Khanal, Pyry Kotilainen, Tommi Mikkonen, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2604.26615)  

**Abstract**: Large language models (LLMs) accelerate software development but often exhibit instability, non-determinism, and weak adherence to development discipline in unconstrained workflows. While test-driven development (TDD) provides a structured Red-Green-Refactor process, existing LLM-based approaches typically use tests as auxiliary inputs rather than enforceable process constraints. We present an AI-native TDD framework that operationalizes classical TDD principles as structured prompt-level and workflow-level governance mechanisms. Extracted principles are formalized in a machine-readable manifesto and distributed across planning, generation, repair, and validation stages within a layered architecture that separates model proposal from deterministic engine authority. The system enforces phase ordering, bounded repair loops, validation gates, and atomic mutation control to improve stability and reproducibility. We describe architecture and discuss encoding software engineering discipline directly into prompt orchestration, which we think offers a promising direction for reliable LLM-assisted development. 

---
# Translating Under Pressure: Domain-Aware LLMs for Crisis Communication 

**Authors**: Antonio Castaldo, Maria Carmen Staiano, Johanna Monti, Sheila Castilho, Francesca Chiusaroli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26597)  

**Abstract**: Timely and reliable multilingual communication is critical during natural and human-induced disasters, but developing effective solutions for crisis communication is limited by the scarcity of curated parallel data. We propose a domain-adaptive pipeline that expands a small reference corpus, by retrieving and filtering data from general corpora. We use the resulting dataset to fine-tune a small language model for crisis-domain translation and then apply preference optimization to bias outputs toward CEFR A2-level English. Automatic and human evaluation shows that this approach improves readability, while maintaining strong adequacy. Our results indicate that simplified English, combined with domain adaptation, can function as a practical lingua franca for emergency communication when full multilingual coverage is not feasible. 

---
# MappingEvolve: LLM-Driven Code Evolution for Technology Mapping 

**Authors**: Rongliang Fu, Yi Liu, Qiang Xu, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2604.26591)  

**Abstract**: Technology mapping is a critical yet challenging stage in logic synthesis. While Large Language Models (LLMs) have been applied to generate optimization scripts, their potential for core algorithm enhancement remains untapped. We introduce MappingEvolve, an open-source framework that pioneers the use of LLMs to directly evolve technology mapping code. Our method abstracts the mapping process into distinct optimization operators and employs a hierarchical agent-based architecture, comprising a Planner, Evolver, and Evaluator, to guide the evolutionary search. This structured approach enables strategic and effective code modifications. Experiments show our method significantly outperforms direct evolution and strong baselines, achieving 10.04\% area reduction versus ABC and 7.93\% versus mockturtle, with 46.6\%--96.0\% $S_{overall}$ improvement on EPFL benchmarks, while explicitly navigating the area--delay trade-off. Our code and data are available at this https URL. 

---
# Star-Fusion: A Multi-modal Transformer Architecture for Discrete Celestial Orientation via Spherical Topology 

**Authors**: May Hammad, Menatallh Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2604.26582)  

**Abstract**: Reliable celestial attitude determination is a critical requirement for autonomous spacecraft navigation, yet traditional "Lost-in-Space" (LIS) algorithms often suffer from high computational overhead and sensitivity to sensor-induced noise. While deep learning has emerged as a promising alternative, standard regression models are often confounded by the non-Euclidean topology of the celestial sphere and by the periodic boundary conditions of Right Ascension (RA) and Declination (Dec). In this paper, we present Star-Fusion, a multi-modal architecture that reformulates orientation estimation as a discrete topological classification task. Our approach leverages spherical K-Means clustering to partition the celestial sphere into K topologically consistent regions, effectively mitigating coordinate wrapping artifacts. The proposed architecture employs a tripartite fusion strategy: a SwinV2-Tiny transformer backbone for photometric feature extraction, a convolutional heatmap branch for spatial grounding, and a coordinate-based MLP for geometric anchoring. Experimental evaluations on a synthetic Hipparcos-derived dataset demonstrate that Star-Fusion achieves a Top-1 accuracy of 93.4% and a Top-3 accuracy of 97.8%. Furthermore, the model exhibits high computational efficiency, maintaining an inference latency of 18.4 ms on resource-constrained COTS hardware, making it a viable candidate for real-time onboard deployment in next-generation satellite constellations. 

---
# Graph Construction and Matching for Imperative Programs using Neural and Structural Methods 

**Authors**: Arshad Beg, Diarmuid O'Donoghue, Rosemary Monahan  

**Link**: [PDF](https://arxiv.org/pdf/2604.26578)  

**Abstract**: Reusing verification artefacts requires identifying structural and semantic similarities across programs and their specifications. In this paper, we focus on graph construction as a foundational step toward this goal. We present a pipeline that converts imperative programs and their annotations into typed, attributed graphs. Our experiments cover datasets including C with ACSL, Java with JML, and Dafny for C\#. The pipeline integrates abstract syntax tree parsing with semantic embeddings derived from models such as SentenceTransformer and CodeBERT. This enables the generation of graph representations that capture both structural relationships and semantic context. Our results show that consistent graph representations can be constructed across different languages and annotation styles. This work provides a practical basis for future steps in semantic enrichment and approximate graph matching for scalable verification artefact reuse. 

---
# Preserving Disagreement: Architectural Heterogeneity and Coherence Validation in Multi-Agent Policy Simulation 

**Authors**: Ariel Sela  

**Link**: [PDF](https://arxiv.org/pdf/2604.26561)  

**Abstract**: Multi-agent deliberation systems using large language models (LLMs) are increasingly proposed for policy simulation, yet they suffer from artificial consensus: evaluator agents converge on the same option regardless of their assigned value perspectives. We present the AI Council, a three-phase deliberation framework, and conduct 120 deliberations across two policy scenarios to test two interventions. First, architectural heterogeneity (assigning a different 7-9B parameter model to each value perspective) significantly reduces first-choice concentration compared to a homogeneous baseline (child welfare: 70.9% to 46.1%, p < 0.001, r = 0.58; housing: 46.0% to 22.9%, p < 0.001, r = 0.50). This contrasts with accuracy-oriented multi-agent debate, where heterogeneity does not reduce convergence, suggesting model diversity operates differently when no objectively correct answer exists. Second, coherence validation (using a frontier model to assess whether each evaluator's reasoning is grounded in its assigned values) reveals a fidelity-diversity tradeoff: on a scenario with a dominant option, it further reduces concentration (46.1% to 40.8%, p = 0.004), but on a scenario with genuinely competitive options, it increases concentration (22.9% to 26.6%, p = 0.96) by amplifying high-coherence evaluators who cluster on one option. This tradeoff may be a general property of multi-agent systems employing quality weighting. We report negative results from three failed Delphi designs, demonstrate that 8B models exhibit binary rather than graded responses to counter-arguments, and propose the trustworthy tension rate as a diagnostic measure of small-model deliberation capabilities. 

---
# DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference 

**Authors**: Bodon Jeong, Hongsu Byun, Youngjae Kim, Weikuan Yu, Kyungkeun Lee, Jihoon Yang, Sungyong Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.26557)  

**Abstract**: The increasing deployment of Large Language Model (LLM) inference on edge AI systems demands efficient execution under tight memory budgets. A key challenge arises from Key-Value (KV) caches, which often exceed available device memory. Although NVMe-based offloading offers scalable capacity, existing file-based designs rely heavily on the kernel page cache, leading to cache thrashing, unpredictable latency, and high software overhead under memory pressure. We present DUAL-BLADE, a dual-path KV residency framework that dynamically assigns KV tensors to either a page-cache path or an NVMe-direct path based on runtime memory availability. The NVMe-direct path bypasses the filesystem by mapping KV tensors to contiguous logical block address (LBA) regions, enabling low-overhead direct storage access. DUAL-BLADE further incorporates adaptive pipeline parallelism to overlap storage I/O with GPU DMA, improving inference throughput. Our evaluation shows that DUAL-BLADE substantially mitigates I/O bottlenecks, reducing prefill and decode latency by up to 33.1% and 42.4%, respectively, while improving SSD utilization by 2.2x across diverse memory budgets. 

---
# TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models 

**Authors**: Jinho Choo, JunSeung Lee, Jimyeong Kim, Yeeho Song, S. K. Hong, Yeong-Dae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2604.26553)  

**Abstract**: Large language models (LLMs) demonstrate strong multilingual capabilities, yet often fail to consistently generate responses in the intended language, exhibiting a phenomenon known as language confusion. Prior mitigation approaches based on sequence-level fine-tuning, such as DPO, ORPO, and GRPO, operate at the level of entire responses and can lead to unintended degradation of general model capabilities, motivating the need for more fine-grained alternatives. To address this, we introduce Token-Level Policy Optimization (TLPO), a fine-tuning framework designed to mitigate language confusion through localized, token-level updates. TLPO identifies error-prone positions, explores alternative candidate tokens, and updates the policy using a tailored objective to suppress error-inducing outputs at a granular level. This selective intervention enables effective mitigation of language confusion without compromising the model's general abilities. Experiments on multiple multilingual LLMs across diverse languages demonstrate that TLPO significantly outperforms baselines in improving language consistency while preserving downstream task accuracy. 

---
# Fundamental Physics, Existential Risks and Human Futures 

**Authors**: Adrian Kent  

**Link**: [PDF](https://arxiv.org/pdf/2604.26530)  

**Abstract**: Over the past 25 years, I have been involved in some intriguing developments in the foundations of physics, exploring the quantum reality problem, the relationship between quantum theory and gravity and the interplay between consciousness and physical laws. These investigations make it plausible that we will find physics beyond quantum theory, potentially including both new evolution laws and new types of measurement. There is also a significant chance they could have potentially transformative impact on information processing and on the development of and our future with AI. 

---
# Lyapunov-Guided Self-Alignment: Test-Time Adaptation for Offline Safe Reinforcement Learning 

**Authors**: Seungyub Han, Hyungjin Kim, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.26516)  

**Abstract**: Offline reinforcement learning (RL) agents often fail when deployed, as the gap between training datasets and real environments leads to unsafe behavior. To address this, we present SAS (Self-Alignment for Safety), a transformer-based framework that enables test-time adaptation in offline safe RL without retraining. In SAS, the main mechanism is self-alignment: at test time, the pretrained agent generates several imagined trajectories and selects those satisfying the Lyapunov condition. These feasible segments are then recycled as in-context prompts, allowing the agent to realign its behavior toward safety while avoiding parameter updates. In effect, SAS turns Lyapunov-guided imagination into control-invariant prompts, and its transformer architecture admits a hierarchical RL interpretation where prompting functions as Bayesian inference over latent skills. Across Safety Gymnasium and MuJoCo benchmarks, SAS consistently reduces cost and failure while maintaining or improving return. 

---
# Text-Utilization for Encoder-dominated Speech Recognition Models 

**Authors**: Albert Zeyer, Tim Posielek, Ralf Schlüter, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2604.26514)  

**Abstract**: This paper investigates efficient methods for utilizing text-only data to improve speech recognition, focusing on encoder-dominated models that facilitate faster recognition. We provide a comprehensive comparison of techniques to integrate text-only data, including modality matching and dynamic downsampling to reach text-level representations within the encoder. Our experiments on the LibriSpeech corpus show that a larger encoder with a smaller decoder can equal or surpass the performance of architectures with larger decoders. We demonstrate that simple configurations, such as random duration models, are often more effective than complex alternatives, significantly simplifying the training pipeline. All code and recipes are made publicly available. 

---
# Tatemae: Detecting Alignment Faking via Tool Selection in LLMs 

**Authors**: Matteo Leonesi, Francesco Belardinelli, Flavio Corradini, Marco Piangerelli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26511)  

**Abstract**: Alignment faking (AF) occurs when an LLM strategically complies with training objectives to avoid value modification, reverting to prior preferences once monitoring is lifted. Current detection methods focus on conversational settings and rely primarily on Chain-of-Thought (CoT) analysis, which provides a reliable signal when strategic reasoning surfaces, but cannot distinguish deception from capability failures if traces are absent or unfaithful. We formalize AF as a composite behavioural event and detect it through observable tool selection, where the LLM selects the safe tool when unmonitored, but switches to the unsafe tool under monitoring that rewards helpfulness over safety, while its reasoning still acknowledges the safe choice. We release a dataset of 108 enterprise IT scenarios spanning Security, Privacy, and Integrity domains under Corruption and Sabotage pressures. Evaluating six frontier LLMs across five independent runs, we find mean AF detection rates between 3.5% and 23.7%, with vulnerability profiles varying by domain and pressure type. These results suggest that susceptibility reflects training methodology rather than capability alone. 

---
# Progressive Semantic Communication for Efficient Edge-Cloud Vision-Language Models 

**Authors**: Cyril Shih-Huan Hsu, Wig Yuan-Cheng Cheng, Chrysa Papagianni  

**Link**: [PDF](https://arxiv.org/pdf/2604.26508)  

**Abstract**: Deploying Vision-Language Models (VLMs) on edge devices remains challenging due to their substantial computational and memory demands, which exceed the capabilities of resource-constrained embedded platforms. Conversely, fully offloading inference to the cloud is often impractical in bandwidth-limited environments, where transmitting raw visual data introduces substantial latency overhead. While recent edge-cloud collaborative architectures attempt to partition VLM workloads across devices, they typically rely on transmitting fixed-size representations, lacking adaptability to dynamic network conditions and failing to fully exploit semantic redundancy. In this paper, we propose a progressive semantic communication framework for edge-cloud VLM inference, using a Meta AutoEncoder that compresses visual tokens into adaptive, progressively refinable representations, enabling plug-and-play deployment with off-the-shelf VLMs without additional fine-tuning. This design allows flexible transmission at different information levels, providing a controllable trade-off between communication cost and semantic fidelity. We implement a full end-to-end edge-cloud system comprising an embedded NXP i.MX95 platform and a GPU server, communicating over bandwidth-constrained networks. Experimental results show that, at 1 Mbps uplink, the proposed progressive scheme significantly reduces network latency compared to full-edge and full-cloud solutions, while maintaining high semantic consistency even under high compression. The implementation code will be released upon publication at this https URL. 

---
# Tree-of-Text: A Tree-based Prompting Framework for Table-to-Text Generation in the Sports Domain 

**Authors**: Shang-Hsuan Chiang, Tsan-Tsung Yang, An-Zi Yen, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.26501)  

**Abstract**: Generating sports game reports from structured tables is a complex table-to-text task that demands both precise data interpretation and fluent narrative generation. Traditional model-based approaches require large, annotated datasets, while prompt-based methods using large language models (LLMs) often struggle with hallucination due to weak table comprehension. To overcome these challenges, we propose Tree-of-Text, a tree-structured prompting framework that guides LLMs through a three-stage generation process: (1) Content Planning, where relevant operations and arguments are selected from the input tables; (2) Operation Execution, which breaks down large tables into manageable sub-tables; and (3) Content Generation, where short textual outputs are merged and rewritten into a cohesive report. Experiments show that our method outperforms existing methods on ShuttleSet+, leads in RG and CO metrics on RotoWire-FG, and excels in CS and CO on MLB with roughly 40% of the time and cost of Chain-of-Table. These results demonstrate the effectiveness and efficiency of Tree-of-Text and suggest a promising direction for prompt-based table-to-text generation in the sports domain. 

---
# Culturally Aware GenAI Risks for Youth: Perspectives from Youth, Parents, and Teachers in a Non-Western Context 

**Authors**: Aljawharah Alzahrani, Tory Park, Tanusree Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2604.26494)  

**Abstract**: Generative AI tools are widely used by youth and have introduced new privacy and safety challenges. While prior research has explored youth's safety in GenAI within western context, it often overlooks the cultural, religious, and social dimensions of technology use that strongly shape youths digital experiences in countries like Saudi Arabia. To address the gap, this study explores children (aged 7 to 17), parents and teachers interactions with GenAI tools and risk perceptions through non-western lens. Through a mixed methods approach, we analyzed 736 Reddit and 1,262 X(Twitter) posts and conducted interviews with 31 Saudi Arabian participants (8 youth, 13 parents, 10 teachers). Our findings highlight context dependent and relational privacy and safety of GenAI from non-western context which often formed by communal structure and prescribed norms. We found significant risks tied to youths disclosure of personal and family information, which conflict with culturally rooted expectations of modesty, privacy, and honor, particularly when youth seek emotional support from GenAI. These risks further compounded by socio economic factors such as cost-saving practices leading to the use of shared GenAI accounts (this http URL) within families or even among strangers. We provide design implication reflecting on parents and teachers expectation of how youth should use GenAI. This work lays groundwork for inclusive, context sensitive parental controls that adhere to cultural norms and values. 

---
# Naamah: A Large Scale Synthetic Sanskrit NER Corpus via DBpedia Seeding and LLM Generation 

**Authors**: Akhil Rajeev P, Annarao Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2604.26456)  

**Abstract**: The digitisation of classical Sanskrit literature is impeded by a scarcity of annotated resources, particularly for Named Entity Recognition. While recent methodologies utilise generic Large Language Models (LLMs) for data augmentation, these approaches remain prone to error and often lack the reasoning depth required for classical grammar. In this work, we introduce Naamah, a high quality silver standard Sanskrit NER dataset comprising 102,942 sentences. We propose a methodology that combines entity extraction from DBpedia with the generative capabilities of a 24B parameter hybrid reasoning model to create grammatically natural and synthetically diverse training data. We utilize this dataset to benchmark two transformer architectures: the massive multilingual XLM RoBERTa and the parameter efficient IndicBERTv2. 

---
# QYOLO: Lightweight Object Detection via Quantum Inspired Shared Channel Mixing 

**Authors**: Garvit Kumar Mittal, Sahil Tomar, Sandeep Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.26435)  

**Abstract**: The rapid advancement of object detection architectures has positioned single stage detectors as the dominant solution for real-time visual perception. A primary source of computational overhead in these models lies in the deep backbone stages, where C2f bottleneck modules at high stride levels accumulate a disproportionate share of parameters due to quadratic scaling with channel width. This work introduces QYOLO, a quantum-inspired channel mixing framework that achieves genuine architectural compression by replacing the two deepest backbone C2f modules at P4/16 (512 channels) and P5/32 (1024 channels) with a compact QMixBlock. The proposed block performs global channel recalibration through a sinusoidal mixing mechanism with shared learnable parameters across both backbone stages, enforcing consistent channel importance without requiring independent per-stage parameter sets. The neck and detection head remain fully classical and unchanged. Evaluation on the VisDrone2019 benchmark demonstrates that QYOLOv8n achieves a 20.2% reduction in parameter count (3.01M to 2.40M) and 12.3% GFLOPs reduction with only 0.4 pp mAP@50 degradation. QYOLOv8s achieves 21.8% reduction with 0.1 pp degradation. When combined with knowledge distillation, full accuracy parity is recovered at no cost to compression. An expanded backbone plus neck variant achieved 38 to 41% reduction at the cost of greater accuracy degradation, motivating the backbone-only final design. 

---
# STLGT: A Scalable Trace-Based Linear Graph Transformer for Tail Latency Prediction in Microservices 

**Authors**: Yongliang Ding, Qigong Bi, Peng Pu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26422)  

**Abstract**: Accurate end-to-end tail-latency forecasting is critical for proactive SLO management in microservice systems. However, modeling long-range dependency propagation and non-stationary, bursty workloads while maintaining inference efficiency at scale remains challenging. We present STLGT (Scalable Trace-based Linear Graph Transformer), a per-API predictor that encodes traces as span graphs for multi-step p95 tail-latency forecasting. STLGT uses a structure-aware linear graph Transformer to propagate cross-service dependencies with inference time linear in span graph size, and a decoupled temporal module to capture workload dynamics. Across a personalized education microservice application, DeathStarBench, and Alibaba traces, STLGT improves forecasting accuracy over PERT-GNN by 8.5% MAPE on average and achieves up to 12x faster CPU inference at N=32, matching the maximum span graph size after preprocessing the Alibaba traces. Ablation studies further demonstrate the effectiveness of each component, especially under bursty traffic. 

---
# Delineating Knowledge Boundaries for Honest Large Vision-Language Models 

**Authors**: Junru Song, Yimeng Hu, Yijing Chen, Huining Li, Qian Li, Lizhen Cui, Yuntao Du  

**Link**: [PDF](https://arxiv.org/pdf/2604.26419)  

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable multimodal performance yet remain prone to factual hallucinations, particularly in long-tail or specialized domains. Moreover, current models exhibit a weak capacity to refuse queries that exceed their parametric knowledge. In this paper, we propose a systematic framework to enhance the refusal capability of VLMs when facing such unknown questions. We first curate a model-specific "Visual-Idk" (Visual-I don't know) dataset, leveraging multi-sample consistency probing to distinguish between known and unknown facts. We then align the model using supervised fine-tuning followed by preference-aware optimization (e.g., DPO, ORPO) to effectively delineate its knowledge boundaries. Results on the Visual-Idk dataset show our method improves the Truthful Rate from 57.9\% to 67.3\%. Additionally, internal probing also demonstrates that the model genuinely recognizes its boundaries instead of just memorizing refusal patterns. Our framework further generalizes to out-of-distribution medical and perceptual domains, providing a robust path toward more trustworthy and prudent visual assistants. 

---
# Quantum Gatekeeper: Multi-Factor Context-Bound Image Steganography with VQC Based Key Derivation on Quantum Hardware 

**Authors**: Sahil Tomar, Sandeep Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.26413)  

**Abstract**: This paper presents Quantum Gatekeeper, a context-bound image steganography framework where successful payload recovery depends on both cryptographic decryption and the reconstruction of a precise extraction path. The system integrates lossless least significant bit (LSB) embedding with a deterministic variational quantum circuit (VQC)-derived gate key, multi-factor contextual binding, and authenticated encryption. Payload extraction is contingent upon four requisite factors: a password, a shared secret, a user-supplied context string, and a reference image signature. Any deviation in these factors causes the system to read from an incorrect pixel sequence or fail authentication, resulting in silent rejection rather than partial disclosure. The proposed method derives a gatecontrolled extraction key from a seed-conditioned variational circuit, with parameters generated via cryptographic hash expansion and context-dependent image features. To ensure encode/decode consistency, the cryptographic key path is generated via exact statevector simulation; concurrently, IBM superconducting quantum hardware is utilized to evaluate the statistical behavior of the circuit family under physical noise. We introduce a dual-region image layout to resolve the nonce bootstrapping dependency, separating header recovery from payload recovery through independently derived keys. Experimental results confirm successful end-to-end message embedding and recovery on PNG images, demonstrating deterministic success under correct conditions and failure otherwise. The framework supports both text and image payloads; in the image-in-image configuration, a secret image is resized to a fixed resolution prior to embedding, enabling exact pixel-level recovery under correct contextual reconstruction. 

---
# SecMate: Multi-Agent Adaptive Cybersecurity Troubleshooting with Tri-Context Personalization 

**Authors**: Yair Meidan, Omri Haller, Yulia Moshan, Shahaf David, Dudu Mimran, Yuval Elovici, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2604.26394)  

**Abstract**: Recent advances in large language models and agentic frameworks have enabled virtual customer assistants (VCAs) for complex support. We present SecMate, a multi-agent VCA for cybersecurity troubleshooting that integrates device, user, and service specificity from conversational and device-level signals. Device specificity is provided by a lightweight local diagnostic utility, while user specificity relies on implicit proficiency inference and profile-aware troubleshooting. Service specificity is achieved through a proactive, context-aware recommender. We evaluate SecMate in a controlled study with 144 participants and 711 conversations. Device-level evidence increased correct resolutions from about 50% to over 90% relative to an LLM-only baseline, while step-by-step guidance improved pleasantness and reduced user burden. The recommender achieved high relevance (MRR@1=0.75), and participants showed strong willingness to substitute human IT support at costs well below human benchmarks. We release the full code base and a richly annotated dataset to support reproducible research on adaptive VCAs. 

---
# Benchmarking Complex Multimodal Document Processing Pipelines: A Unified Evaluation Framework for Enterprise AI 

**Authors**: Saurabh K. Singh, Sachin Raj  

**Link**: [PDF](https://arxiv.org/pdf/2604.26382)  

**Abstract**: Most enterprise document AI today is a pipeline. Parse, index, retrieve, generate. Each of those stages has been studied to death on its own -- what's still hard is evaluating the system as a whole.
We built EnterpriseDocBench to take a swing at it: parsing fidelity, indexing efficiency, retrieval relevance, and generation groundedness, all on the same corpus. The corpus is built from public, permissively licensed documents across six enterprise domains (five represented in the current pilot). We ran three pipelines through it -- BM25, dense embedding, and a hybrid -- all with the same GPT-5 generator.
The headline numbers: hybrid retrieval narrowly beats BM25 (nDCG@5 of 0.92 vs. 0.91), and both beat dense embedding (0.83). Hallucination doesn't grow monotonically with document length -- short documents and very long ones both hallucinate more than medium ones (28.1% and 23.8% vs. 9.2%). Cross-stage correlations are very weak: parsing->retrieval r=0.14, parsing->generation r=0.17, retrieval->generation 0.02. If quality were cascading the way most of us assume, those numbers would be much higher; they aren't. Design caveats are real (parsing fixed, generator shared, automated proxy metrics) and we don't oversell the result.
One result that genuinely surprised us: factual accuracy on stated claims is 85.5%, but answer completeness averages 0.40. The system is right when it answers -- it just leaves things out. That gap matters more for real deployments than the headline accuracy number does.
We also describe three reference architectures (ColPali, ColQwen2, agentic complexity-based routing) which are not yet integrated end-to-end. Framework, metrics, baselines, and collection scripts will be released open-source on acceptance. 

---
# SG-UniBuc-NLP at SemEval-2026 Task 6: Multi-Head RoBERTa with Chunking for Long-Context Evasion Detection 

**Authors**: Gabriel Stefan, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26375)  

**Abstract**: We describe our system for SemEval-2026 Task 6 (CLARITY: Unmasking Political Question Evasions), which classifies English political interview responses by coarse-grained clarity (3-way) and fine-grained evasion strategy (9-way). Since responses frequently exceed the 512-token limit of standard Transformer encoders, we apply an overlapping sliding-window chunking strategy with element-wise Max-Pooling aggregation over chunk representations. A shared RoBERTa-large encoder supplies two task-specific heads trained jointly via a multi-task objective, with inference-time ensembling over 7-fold stratified cross-validation. Our system achieves a Macro-F1 of 0.80 on Subtask 1 and 0.51 on Subtask 2, ranking 11th in both subtasks. 

---
# Text Style Transfer with Machine Translation for Graphic Designs 

**Authors**: Deergh Singh Budhauria, Sanyam Jain, Rishav Agarwal, Tracy King  

**Link**: [PDF](https://arxiv.org/pdf/2604.26361)  

**Abstract**: Globalization of graphic designs such as those used in marketing materials and magazines is increasingly important for communication to broad audiences. To accomplish this, the textual content in the graphic designs needs to be accurately translated and have the text styling preserved in order to fit visually into the design. Preserving text styling requires high accuracy word alignment between the original and the translated text. The problem of word alignment between source and translated text is long known. The industry standards for extracting word alignments are defined by Giza++ and attention probabilities from neural machine translation (NMT) models. In this paper, we explore three new methods to tackle the word alignment problem for transferring text styles from the source to the translated text. The proposed methods are developed on top of commercially available NMT and LLM translation technologies. They include: NMT with custom input and output tags for text styling; LLM with custom input and output tags; a hybrid with NMT for translation followed by an LLM with use of unigram mappings. To analyze the performance of these solutions, their alignment results are compared with the results of an attention head approach to gauge their usability in graphic design applications. Interestingly, the attention head strong baseline proves more accurate than the LLM or NMT approach and on par with the hybrid NMT+LLM approach. 

---
# Uncertainty-Aware Reward Discounting for Mitigating Reward Hacking 

**Authors**: Disha Singha  

**Link**: [PDF](https://arxiv.org/pdf/2604.26360)  

**Abstract**: Reinforcement learning (RL) systems typically optimize scalar reward functions that assume precise and reliable evaluation of outcomes. However, real-world objectives--especially those derived from human preferences--are often uncertain, context-dependent, and internally inconsistent. This mismatch can lead to alignment failures such as reward hacking, over-optimization, and overconfident behavior.
We introduce a dual-source uncertainty-aware reward framework that explicitly models both epistemic uncertainty in value estimation and uncertainty in human preferences. Model uncertainty is captured via ensemble disagreement over value predictions, while preference uncertainty is derived from variability in reward annotations. We combine these signals through a confidence-adjusted Reliability Filter that adaptively modulates action selection, encouraging a balance between exploitation and caution.
Empirical results across multiple discrete grid configurations (6x6, 8x8, 10x10) and high-dimensional continuous control environments (Hopper-v4, Walker2d-v4) demonstrate that our approach yields more stable training dynamics and reduces exploitative behaviors under reward ambiguity, achieving a 93.7% reduction in reward-hacking behavior as measured by trap visitation frequency. We demonstrate statistical significance of these improvements and robustness under up to 30% supervisory noise, albeit with a trade-off in peak observed reward compared to unconstrained baselines.
By treating uncertainty as a first-class component of the reward signal, this work offers a principled approach toward more reliable and aligned reinforcement learning systems. 

---
# ACPO: Anchor-Constrained Perceptual Optimization for Diffusion Models with No-Reference Quality Guidance 

**Authors**: Yang Yang, Feifan Meng, Han Fang, Weiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26348)  

**Abstract**: Diffusion models have achieved remarkable success in image generation, yet their training is predominantly driven by full-reference objectives that enforce pixel-wise similarity to ground-truth this http URL supervision, while effective for fidelity, may insufficient in terms of subjective visual perception quality and text-image semantic consistency. In this work, we investigate the problem of incorporating no-reference perceptual quality into diffusion training. A key challenge is that directly optimizing perceptual signals, such as those provided by no-reference image quality assessment (NR-IQA) models, introduces a mismatch with the original diffusion objective, leading to training instability and distributional drift during fine-tuning. To address this issue, we propose an anchor-constrained optimization framework that enables stable perceptual adaptation. Specifically, we leverage a learned NR-IQA model as a perceptual guidance signal, while introducing an anchor-based regularization that enforces consistency with the base diffusion model in terms of noise prediction. This design effectively balances perceptual quality improvement and generative fidelity, allowing controlled adaptation toward perceptually favorable outputs without compromising the original generative behavior. Extensive experiments demonstrate that our method consistently enhances perceptual quality while preserving generation diversity and training stability, highlighting the effectiveness of anchor-constrained perceptual optimization for diffusion models. 

---
# DSIPA: Detecting LLM-Generated Texts via Sentiment-Invariant Patterns Divergence Analysis 

**Authors**: Siyuan Li, Aodu Wulianghai, Guangyan Li, Xi Lin, Qinghua Mao, Yuliang Chen, Jun Wu, Jianhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.26328)  

**Abstract**: The rapid advancement of large language models (LLMs) presents new security challenges, particularly in detecting machine-generated text used for misinformation, impersonation, and content forgery. Most existing detection approaches struggle with robustness against adversarial perturbation, paraphrasing attacks, and domain shifts, often requiring restrictive access to model parameters or large labeled datasets. To address this, we propose DSIPA, a novel training-free framework that detects LLM-generated content by quantifying sentiment distributional stability under controlled stylistic variation. It is based on the observation that LLMs typically exhibit more emotionally consistent outputs, while human-written texts display greater affective variation. Our framework operates in a zero-shot, black-box manner, leveraging two unsupervised metrics, sentiment distribution consistency and sentiment distribution preservation, to capture these intrinsic behavioral asymmetries without the need for parameter updates or probability access. Extensive experiments are conducted on state-of-the-art proprietary and open-source models, including GPT-5.2, Gemini-1.5-pro, Claude-3, and LLaMa-3.3. Evaluations on five domains, such as news articles, programming code, student essays, academic papers, and community comments, demonstrate that DSIPA improves F1 detection scores by up to 49.89% over baseline methods. The framework exhibits superior generalizability across domains and strong resilience to adversarial conditions, providing a robust and interpretable behavioral signal for secure content identification in the evolving LLM landscape. 

---
# CheXthought: A global multimodal dataset of clinical chain-of-thought reasoning and visual attention for chest X-ray interpretation 

**Authors**: Sonali Sharma, Jin Long, George Shih, Sarah Eid, Christian Bluethgen, Francine L. Jacobson, Emily B. Tsai, Global Radiology Consortium, Ahmed M. Alaa, Curtis P. Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2604.26288)  

**Abstract**: Chest X-ray interpretation is one of the most frequently performed diagnostic tasks in medicine and a primary target for AI development, yet current vision--language models are primarily trained on datasets of paired images and reports, not the cognitive processes and visual attention that underlie clinical reasoning. Here, we present CheXthought, a global, multimodal resource containing 103,592 chain-of-thought reasoning traces and 6,609,082 synchronized visual attention annotations across 50,312 multi-read chest X-rays from 501 radiologists in 71 countries. Our analysis reveals clinical reasoning patterns in how experts deploy distinct visual search strategies, integrate clinical context, and communicate uncertainty. We demonstrate the clinical utility of CheXthought across four dimensions. First, CheXthought reasoning significantly outperforms state--of--the--art vision--language model chain-of-thought in factual accuracy and spatial grounding. Second, visual attention data used as an inference--time hint recovers missed findings and significantly reduces hallucinations. Third, models trained on CheXthought data achieve significantly stronger pathology classification, visual faithfulness, temporal reasoning and uncertainty communication. Fourth, leveraging CheXthought's multi-reader annotations, we predict both human--human and human--AI disagreement directly from an image, enabling transparent communication of case difficulty, uncertainty and model reliability. These findings establish CheXthought as a resource for advancing multimodal clinical reasoning and the development of more transparent, interpretable vision--language models. 

---
# MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution 

**Authors**: Chunzheng Zhu, Jiaqi Zeng, Junyu Jiang, Jianxin Lin, Yijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26283)  

**Abstract**: High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose ours, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model's hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR), which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privileged-autonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets demonstrate that ours, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-the-art methods, particularly chain-of-thought paradigms, in diagnostic accuracy. 

---
# Enforcing Benign Trajectories: A Behavioral Firewall for Structured-Workflow AI Agents 

**Authors**: Hung Dang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26274)  

**Abstract**: Structured-workflow agents driven by large language models execute tool calls against sensitive external environments. We propose \codename, a telemetry-driven behavioral anomaly detection firewall. Drawing on sequence-based intrusion detection, \codename\ compiles verified benign tool-call telemetry into a parameterized deterministic finite automaton (pDFA). The model defines permitted tool sequences, sequential contexts, and parameter bounds. At runtime, a lightweight gateway enforces these boundaries via an $O(1)$ state-transition structural lookup, shifting computationally expensive analysis entirely offline. Evaluated on the Agent Security Bench (ASB), \codename\ achieves a 5.6\% macro-averaged attack success rate (ASR) across five scenarios. Within three structured workflows, ASR drops to 2.2\%, outperforming Aegis, a state-of-the-art stateless scanner, at 12.8\%. \codename\ achieves 0\% ASR on multi-step and context-sequential attacks in structured settings. Furthermore, against 1,000 algorithmically spliced exfiltration payloads, only 1.4\% matched valid structural paths, all of which failed end-to-end string parameter guards (0 successes out of 14 surviving paths, 95\% CI [0\%, 23.2\%]). \codename\ introduces just 2.2~ms of per-call latency (a 3.7$\times$ speedup over \textsc{Aegis}) while maintaining a 2.0\% benign task failure rate (BTFR) on benign workloads. Modeling the behavioral trajectory effectively collapses the available attack surface, but unmaintained continuous parameter bounds remain vulnerable to synonym-substitution attacks (18\% evasion rate). Thus, exact-match whitelisting of sensitive parameters ultimately bears the final defensive load against execution. 

---
# Calibrated Surprise: An Information-Theoretic Account of Creative Quality 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26269)  

**Abstract**: The essence of good creative writing is calibrated surprise: when constraints from all relevant dimensions act together, the feasible solution space collapses into a narrow region, and the surviving choices look least predictable from an unconstrained view. "Calibrated" has a precise meaning: the author's intent, the reader's reasonable expectation, and the logic of reality converge. When these three independent judgements agree on every dimension, the set of admissible writing choices is forced into a very small region. A mathematical corollary follows: full-dimensional accuracy and mediocrity are mutually exclusive -- two sides of one constraint structure, not separate goals.
We use Shannon's mutual information $I(X;Y) = H(X) - H(X|Y)$ as our analysis tool. "Calibrated" corresponds to conditional entropy going to zero; "surprise" to entropy going up; mutual information is the precise measure of the joint quantity. The argument rests on two pillars. Static: when constraints from ethos, mythos, lexis, and dianoia are imposed together, the admissible set collapses sharply, and surviving solutions show up as low-probability choices from an unconstrained view. Dynamic: the chain rule shows each writing choice is constrained by what came before and constrains what comes after; macro-level decisions naturally contribute a larger share of information, removing the need for hand-tuned weighting.
Through case studies and lightweight LLM-logprob computations, we show the framework is both analytically useful and operational, laying the theoretical groundwork for Creative Quality Alignment (CQA) and a professional evaluation benchmark. 

---
# Multi-Stage Bi-Atrial Segmentation Framework from 3D Late Gadolinium-Enhanced MRI using V-Net Family Models 

**Authors**: Hao Wen, Jingsu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26251)  

**Abstract**: We report our multi-stage framework designed for the problem of multi-class bi-atrial segmentation from 3D late gadolinium-enhanced (LGE) MRI of the human heart. The pipeline consists of a preprocessing step using multidimensional contrast limited adaptive histogram equalization (MCLAHE); coarse region segmentation from MCLAHE-enhanced and down-sampled MRI using a V-Net family model; and fine segmentation from the coarse region using another V-Net model. Asymmetric loss is adopted to optimize the model weights. 

---
# TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation 

**Authors**: Wei Yang, Rui Zhong, Zihan Lin, Xiaodan Wang, Cheng Chen, Huan Ren, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26247)  

**Abstract**: Multimodal recommendation improves user modeling by integrating collaborative signals with heterogeneous item content. In real applications, user interests evolve over time and exhibit nonstationary dynamics, where different preference factors change at different rates. This challenge is amplified in multimodal settings because visual and textual cues can dominate decisions under different temporal regimes. Despite strong progress, most multimodal recommenders still rely on static interaction graphs or coarse temporal heuristics, which limits their ability to model continuous preference evolution with fine-grained temporal adaptation. To address these limitations, we propose TimeMM, a time-conditioned spectral filtering framework for dynamic multimodal recommendation. TimeMM instantiates Time-as-Operator by mapping interaction recency to a family of parametric temporal kernels that reweight edges on the user--item graph, producing component-specific representations without explicit eigendecomposition. To capture non-stationary interests, we introduce Adaptive Spectral Filtering that mixes the operator bank according to temporal context, yielding prediction-specific effective spectral responses. To account for modality-specific temporal sensitivity, we further propose Spectral-Aware Modality Routing that calibrates visual and textual contributions conditioned on the same temporal context. Finally, a ranking-space Spectral Diversity Regularization encourages complementary expert behaviors and prevents filter-bank collapse. Extensive experiments on real-world benchmarks demonstrate that TimeMM consistently outperforms state-of-the-art multimodal recommenders while maintaining linear-time scalability. 

---
# MetaSR: Content-Adaptive Metadata Orchestration for Generative Super-Resolution 

**Authors**: Jiaqi Guo, Mingzhen Li, Haohong Wang, Aggelos K. Katsaggelos  

**Link**: [PDF](https://arxiv.org/pdf/2604.26244)  

**Abstract**: We study generative super-resolution (SR) in real-world scenarios where content and degradations vary across domains, genres, and segments. For example, images and videos may alternate between text overlays, fast motion, smooth cartoons, and low-light faces, each benefiting from different forms of side information. Existing metadata-guided SR methods typically use a fixed conditioning design, which is suboptimal when useful cues are content dependent and transmission budgets are limited. We propose MetaSR, a Diffusion Transformer (DiT)-based framework that selects and injects task-relevant metadata to guide SR under resource constraints. Specifically, we use the DiT's own VAE and transformer backbone to fuse heterogeneous metadata, and adopt an efficient distillation strategy that enables one-step diffusion inference. Experiments across diverse content buckets and degradation regimes show that MetaSR outperforms reference solutions by up to 1.0~dB PSNR while achieving up to 50\% transmission bitrate saving at matched quality. We assess these gains under a rate--distortion optimization (RDO) framework that jointly accounts for sender-side bitrate and receiver/display quality metrics (e.g., PSNR and SSIM). 

---
# StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall 

**Authors**: Yerong Wu, Tianxing Wu, Minghao Zhu, Hangyu Sha, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26243)  

**Abstract**: Achieving realistic human-like conversation for virtual characters requires not only a simple memorization and recall of past events, but also the strategic utilization of memory to meet factual needs and social engagement. Current memory utilization relevant (e.g., memory-augmented generation, long-term dialogue, and etc.) benchmarks overlook this nuance, treating memory primarily as a static repository of facts rather than a dynamic resource to be strategically deployed in dialogues. To address this gap, we design StratMem-Bench, a new benchmark to evaluate strategic memory use in character-centric dialogues. This dataset comprises 657 instances where virtual characters must navigate heterogeneous memory pools containing required, supportive, and irrelevant memories. We also propose a framework with different evaluation metrics including Strict Memory Compliance, Memory Integration Quality, Proactive Enrichment Score and Conditional Irrelevance Rate, to evaluate strategic memory use capabilities of virtual characters. Experiments on StratMem-Bench which leverage the state-of-the-art large language models as virtual characters show that all models perform well at distinguishing between required and irrelevant memories, but struggle once supportive memories are introduced into the decision process. 

---
# LATTICE: Evaluating Decision Support Utility of Crypto Agents 

**Authors**: Aaron Chan, Tengfei Li, Tianyi Xiao, Angela Chen, Junyi Du, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.26235)  

**Abstract**: We introduce LATTICE, a benchmark for evaluating the decision support utility of crypto agents in realistic user-facing scenarios. Prior crypto agent benchmarks mainly focus on reasoning-based or outcome-based evaluation, but do not assess agents' ability to assist user decision-making. LATTICE addresses this gap by: (1) defining six evaluation dimensions that capture key decision support properties; (2) proposing 16 task types that span the end-to-end crypto copilot workflow; and (3) using LLM judges to automatically score agent outputs based on these dimensions and tasks. Crucially, the dimensions and tasks are designed to be evaluable at scale using LLM judges, without relying on ground truth from expert annotators or external data sources. In lieu of these dependencies, LATTICE's LLM judge rubrics can be continually audited and updated given new dimensions, tasks, criteria, and human feedback, thus promoting reliable and extensible evaluation. While other benchmarks often compare foundation models sharing a generic agent framework, we use LATTICE to assess production-level agents used in actual crypto copilot products, reflecting the importance of orchestration and UI/UX design in determining agent quality. In this paper, we evaluate six real-world crypto copilots on 1,200 diverse queries and report breakdowns across dimensions, tasks, and query categories. Our experiments show that most of the tested copilots achieve comparable aggregate scores, but differ more significantly on dimension-level and task-level performance. This pattern suggests meaningful trade-offs in decision support quality: users with different priorities may be better served by different copilots than the aggregate rankings alone would indicate. To support reproducible research, we open-source all LATTICE code and data used in this paper. 

---
# DepthPilot: From Controllability to Interpretability in Colonoscopy Video Generation 

**Authors**: Junhu Fu, Ke Chen, Weidong Guo, Shuyu Liang, Jie Xu, Chen Ma, Kehao Wang, Shengli Lin, Zeju Li, Yuanyuan Wang, Yi Guo, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.26232)  

**Abstract**: Controllable medical video generation has achieved remarkable progress, but it still lacks interpretability, which requires the alignment of generated contents with physical priors and faithful clinical manifestations. To push the boundaries from mere controllability to interpretability, we propose DepthPilot, the first interpretable framework for colonoscopy video generation. This work takes a step toward trustworthy generation through two synergistic paradigms. To achieve explicit geometric grounding, DepthPilot devises a prior distribution alignment strategy, injecting depth constraints into the diffusion backbone via parameter-efficient fine-tuning to ensure anatomical fidelity. To enhance intrinsic nonlinear modeling under these geometric constraints, DepthPilot employs an adaptive spline denoising module, replacing fixed linear weights with learnable spline functions to capture complex spatio-temporal dynamics. Extensive evaluations across three public datasets and in-house clinical data confirm DepthPilot's robust ability to produce physically consistent videos. It achieves FID scores below 15 across all benchmarks and ranks first in clinician assessments, bridging the gap between "visually realistic" and "clinically interpretable". Moreover, DepthPilot-generated videos are expected to enable reliable 3D reconstruction, facilitating surgical navigation and blind region identification, and serve as a foundation toward the colorectal world model. 

---
# Seeking Consensus: Geometric-Semantic On-the-Fly Recalibration for Open-Vocabulary Remote Sensing Semantic Segmentation 

**Authors**: Guanchun Wang, Chenxiao Wu, Xiangrong Zhang, Zelin Peng, Jianxun Lai, Tianyang Zhang, Xu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26221)  

**Abstract**: Open-vocabulary semantic segmentation (OVSS) in remote sensing images is a promising task that employs textual descriptions for identifying undefined land cover categories. Despite notable advances, existing methods typically employ a static inference paradigm, overlooking the distinct distribution of each scene, resulting in semantic ambiguity in diverse land covers and incomplete foreground activation. Motivated by this, we propose Seeking Consensus, termed SeeCo, a plug-and-play framework to boost the performance of training-free OVSS models in remote sensing images, which recalibrates arbitrary OVSS models on-the-fly by seeking dual consensus: geometric consensus learning (GCL) through multi-view consistent observations and semantic consensus learning (SCL) via textual description adaptive calibration, which assists collaborative recalibration of visual and textual semantics. The two consensus are injected via an online consensus injector (OCI), effectively alleviating the under-activation and semantic bias. SeeCo requires no specific training process, yet recalibrates semantic-geometric alignment for each unique scene during inference. Extensive experiments on eight remote sensing OVSS benchmarks show consistent gains, proving its effectiveness and universality. 

---
# Qvine: Vine Structured Quantum Circuits for Loading High Dimensional Distributions 

**Authors**: David Quiroga, Hannes Leipold, Bibhas Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2604.26213)  

**Abstract**: Loading high dimensional distributions is an important task for utilizing quantum computers on applications ranging from machine learning to finance. The high dimensionality leads to a curse of dimensionality, representing a d-dimensional distribution with k resolution requires dk qubits and an unstructured parameterized circuit would express a unitary in an exponential operator space in the number of qubits, leading to vanishing gradients and poor convergence guarantees even at high depth.
Vine copula decompositions are widely used to represent high dimensional distributions classically, showing high quality approximation in many important applications, such as financial modeling. We present Qvine, a vine structured ansatz for quantum circuits, that mirrors the vine decomposition to construct scalable quantum circuits with efficient trainability while achieving similarly high quality approximation for amplitude encoding distributions. For regular vines (R-vines), we show that the circuit depth scales at most quadratic in the dimension of the distribution, while for D-vines, as well as many practical R-vines, the circuit depth scales linear in the dimension. For 3-dimensional and 4-dimensional Gaussians and empirical joint stock price return distributions for selected stocks, our experiments show Qvines achieve high quality loading. 

---
# Breaking the Autoregressive Chain: Hyper-Parallel Decoding for Efficient LLM-Based Attribute Value Extraction 

**Authors**: Theodore Glavas, Nikhita Vedula, Dushyanta Dhyani, Yilun Zhu, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26209)  

**Abstract**: Some text generation tasks, such as Attribute Value Extraction (AVE), require decoding multiple independent sequences from the same document context. While standard autoregressive decoding is slow due to its sequential nature, the independence between output sequences offers an opportunity for parallelism. We present Hyper-Parallel Decoding, a novel decoding algorithm that accelerates offline decoding by leveraging both shared memory and computation across batches. HPD enables out-of-order token generation through position ID manipulation, significantly improving efficiency. Experiments on AVE show that attribute-value pairs are conditionally independent, enabling us to parallelize value generation within each prompt. By further stacking multiple documents within a single prompt, we can decode in parallel up to 96 tokens per prompt. HPD works with all LLMs, and reduces both inference costs and total inference time by up to 13.8X without compromising output quality, potentially saving hundreds of thousands of dollars on industry AVE tasks. Although designed for attribute extraction, HPD makes no assumptions unique to the AVE domain and can in theory be applied to other scenarios with independent output structures. 

---
# Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted Sandbagging 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26206)  

**Abstract**: A predecessor pilot (Cacioli, 2026) found that Llama-3-8B implements prompted sandbagging as positional collapse rather than answer avoidance. However, fixed option ordering in MMLU-Pro left open whether this reflected a model-level position-dominant policy or dataset-level distractor structure. This pre-registered follow-up (3 models, 2,000 MMLU-Pro items, 4 conditions, 24,000 primary trials) added cyclic option-order randomisation as the critical control. The pre-registered item-level same-letter diagnostic did not confirm deterministic position-tracking (same-letter rate 37.3%, below the 50% threshold). However, pre-specified supporting analyses revealed that the response-position distribution under sandbagging was highly stable under complete content rotation (Pearson r = 0.9994; Jensen-Shannon divergence = 0.027, compared to 0.386 between honest and sandbagging conditions). Accuracy spiked to 72.1% when the correct answer coincidentally occupied the preferred position E, and fell to 4.3% at position A. The data provide strong evidence for a soft distributional attractor: under sandbagging instruction, the model enters a low-entropy response-position basin centred on E/F/G that is highly stable and largely content-invariant at the aggregate level. Qwen-2.5-7B served as a negative control (non-compliant, no distributional shift). These results provide evidence, at the 7-9 billion parameter scale, that response-position entropy is a promising black-box behavioural signature of this sandbagging mode. 

---
# Lifting Embodied World Models for Planning and Control 

**Authors**: Alex N. Wang, Trevor Darrell, Pavel Izmailov, Yutong Bai, Amir Bar  

**Link**: [PDF](https://arxiv.org/pdf/2604.26182)  

**Abstract**: World models of embodied agents predict future observations conditioned on an action taken by the agent. For complex embodiments, action spaces are high-dimensional and difficult to specify: for example, precisely controlling a human agent requires specifying the motion of each joint. This makes the world model hard to control and expensive to plan with as search-based methods like CEM scale poorly with action dimensionality. To address this issue, we train a lightweight policy that maps high-level actions to sequences of low-level joint actions. Composing this policy with the frozen world model produces a lifted world model that predicts a sequence of future observations from a single high-level action. We instantiate this framework for a human-like embodiment, defining the high-level action space as a small set of 2D waypoints annotated on the current observation frame, each specifying a near-term goal position for a leaf joint (pelvis, head, hands). Waypoints are low-dimensional, visually interpretable, and easy to specify manually or to search over. We show that the lifted world model substantially outperforms searching directly in low-level joint space ($3.8\times$ lower mean joint error to the goal pose), while remaining more compute-efficient and generalizing to environments unseen by the policy. 

---
# Evergreen: Efficient Claim Verification for Semantic Aggregates 

**Authors**: Alexander W. Lee, Benjamin Han, Shayak Sen, Sam Yeom, Ugur Cetintemel, Anupam Datta  

**Link**: [PDF](https://arxiv.org/pdf/2604.26180)  

**Abstract**: With recent semantic query processing engines, semantic aggregation has become a primitive operator, enabling the reduction of a relation into a natural language aggregate using an LLM. However, the resulting semantic aggregate may contain claims that are not grounded in the underlying relation. Verifying such claims is challenging: they often involve quantifiers, groupings, and comparisons over relations that far exceed LLM context windows and require a costly combination of semantic and symbolic processing.
We present Evergreen, a system that recasts claim verification as a semantic query processing task with tailored optimizations and provenance capture. Evergreen compiles each claim into a declarative semantic verification query and executes it on the same engine that produced the aggregate. To reduce cost and latency, Evergreen avoids unnecessary LLM calls through verification-aware optimizations (early stopping, relevance sorting, and estimation with confidence sequences) and general-purpose optimizations for semantic queries (operator fusion, similarity filtering, and prompt caching). Each verdict is accompanied by citations that identify a minimal set of tuples justifying the result, with semantics based on semiring provenance for first-order logic.
On a benchmark of real-world restaurant review datasets reflecting production-inspired workloads, Evergreen achieves excellent verification quality (F1 = 1.00) with a strong LLM while reducing cost by 3.2x and latency by 4.0x compared to unoptimized verification. Even with a significantly weaker LLM, Evergreen outperforms a strong LLM-as-a-judge baseline in F1 at 48x lower cost and 2.3x lower latency. Relative to a retrieval-augmented agent, Evergreen compares favorably in F1 and latency with similar cost when both use a strong LLM; yet, with a much weaker LLM, it achieves the same F1 at 63x lower cost and 4.2x lower latency. 

---
# Entropy Centroids as Intrinsic Rewards for Test-Time Scaling 

**Authors**: Wenshuo Zhao, Qi Zhu, Xingshan Zeng, Fei Mi, Lifeng Shang, Yiren Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.26173)  

**Abstract**: An effective way to scale up test-time compute of large language models is to sample multiple responses and then select the best one, as in Grok Heavy and Gemini Deep Think. Existing selection methods often rely on external reward models, which requires training a strong reward model and introduces additional computation overhead. As an alternative, previous approaches have explored intrinsic signals, such as confidence and entropy, but these signals are noisy with naive aggregation. In this work, we observe that high-entropy tokens tend to cluster into consecutive groups during inference, providing a more stable notion of model uncertainty than individual tokens. Together, these clusters reveal temporal patterns of model uncertainty throughout the inference process. Motivated by this observation, we propose to use the temporal structure of uncertainty as an intrinsic reward. To this end, we first formalize the basic unit of segment-level uncertainty as the High Entropy Phase (HEP), a variable-length segment that begins at a high-entropy token and ends when consecutive low-entropy tokens appear. We then define the Entropy Centroid, inspired by the concept of the center of mass in physics, as the weighted average position of all HEPs along the trajectory. Intuitively, a lower centroid indicates early exploration followed by confident generation, which we find often corresponds to higher response quality. Based on this insight, we propose the Lowest Centroid method, which selects the response with the lowest entropy centroid among multiple candidates. Experiments on mathematics, code generation, logical reasoning, and agentic tasks, across model scales ranging from 14B to 480B, show that Lowest Centroid consistently outperforms existing baselines and delivers stable gains as model size increases. Code is available at this https URL. 

---
# Co-Learning Port-Hamiltonian Systems and Optimal Energy-Shaping Control 

**Authors**: Ankur Kamboj, Biswadip Dey, Vaibhav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2604.26172)  

**Abstract**: We develop a physics-informed learning framework for energy-shaping control of port-Hamiltonian (pH) systems from trajectory data. The proposed approach {co-learns} a pH system model and an optimal energy-balancing passivity-based controller (EB-PBC) through alternating optimization with policy-aware data collection. At each iteration, the system model is refined using trajectory data collected under the current control policy, and the controller is re-optimized on the updated model. Both components are parameterized by neural networks that embed the pH {dynamics} and EB-PBC structure, ensuring interpretability in terms of energy {interactions}. The learned controller renders the closed-loop system inherently passive and provably stable, and exploits passive plant dynamics without canceling the natural potential. A dissipation regularization enforces strict energy decay during training, thereby enhancing robustness to sim-to-real gaps. The proposed framework is validated on state-regulation and swing-up tasks for planar and torsional pendulum systems. 

---
# Test-Time Safety Alignment 

**Authors**: Baturay Saglam, Dionysis Kalogerias  

**Link**: [PDF](https://arxiv.org/pdf/2604.26167)  

**Abstract**: Recent work has shown that a model's input word embeddings can serve as effective control variables for steering its behavior toward outputs that satisfy desired properties. However, this has only been demonstrated for pretrained text-completion models on the relatively simple objective of reducing surface-level profanity in short continuations. A natural and practically important question is how well input embeddings can control aligned models, which produce an imbalanced bimodal refuse-or-comply output distribution rather than the smooth distribution characteristic of open-ended generation. We explore this in the context of safety, showing that input word embeddings can be optimized in a sub-lexical manner to minimize the semantic harmfulness of aligned model responses. Our approach uses zeroth-order gradient estimation of a black-box text-moderation API with respect to the input embeddings, and then applies gradient descent on these embeddings to minimize the harmfulness of the generated text. Experiments show that the proposed method can neutralize every safety-flagged response on standard safety benchmarks. 

---
# Structural Generalization on SLOG without Hand-Written Rules 

**Authors**: Zichao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.26157)  

**Abstract**: Structural generalization in semantic parsing requires systems to apply learned compositional rules to novel structural combinations. Existing approaches either rely on hand-written algebraic rules (AM-Parser) or fail to generalize structurally (Transformer-based models). We present an alternative requiring no hand-written compositional rules, based on a neural cellular automaton (NCA) with a discrete bottleneck: all compositional rules are learned from data through local iteration. On the SLOG benchmark, the system achieves 100% type-exact match on 11 of 17 structural generalization categories, including three where AM-Parser scores 0 to 74%, with an overall standard deviation of 0.2 across 10 seeds (vs. AM-Parser's 4.3). Analysis reveals that all 5,539 failure instances reduce to exactly two mechanisms: novel combinations of wh-extraction context with reduced verb types, and modifiers appearing on the subject side of this http URL we decompose results by CCG structural features, each sub-pattern either succeeds on all instances or fails on all. Intermediate scores (e.g., 41.4%) are mixtures of structurally distinct CCG patterns, not partial this http URL failures correspond to directed operations absent from training; all successes correspond to operations already covered. 

---
# A Data-Centric Framework for Intraoperative Fluorescence Lifetime Imaging for Glioma Surgical Guidance 

**Authors**: Silvia Noble Anbunesan, Mohamed Abul Hassan, Jinyi Qi, Lisanne Kraft, Han Sung Lee, Orin Bloch, Laura Marcu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26147)  

**Abstract**: Accurate intraoperative assessment of glioma infiltration is essential for maximizing tumor resection while preserving functional brain tissue. Fluorescence lifetime imaging (FLIm) offers real-time, label-free biochemical contrast, but its clinical utility is challenged by biological heterogeneity, class imbalance, and variability in histopathological labeling. We present a data-centric AI (DC-AI) framework that integrates confident learning (CL), class refinement, and targeted label evaluation to develop a robust multi-class FLIm classifier for glioblastoma (GBM) resection margins. FLIm data were collected from 192 tissue margins across 31 newly diagnosed IDH-wildtype GBM patients and initially labeled into seven tumor cellularity classes by an expert neuropathologist. CL was applied to quantify FLIm point-level confidence, identify label inconsistencies, and guide iterative class merging into a three-class scheme ("low", "moderate", "high"). The resulting high-fidelity dataset enabled training a model that achieved 96% accuracy in the three-class task. SHAP analysis revealed class-specific FLIm feature importance, highlighting distinct optical signatures across the infiltration spectrum. Targeted FLIm analysis further identified biological (e.g., gray matter composition) and acquisition-related (e.g., blood contamination) contributors to low-confidence predictions. Blinded re-evaluation of margins flagged by CL demonstrated intra-pathologist variability, underscoring the value of selective relabeling rather than exhaustive review. Together, these findings demonstrate that a DC-AI framework can systematically improve data reliability, enhance model robustness, and refine biological interpretation of FLIm signals, supporting the development of clinically actionable optical tools for real-time glioma margin assessment. 

---
# Ceci n'est pas une explication: Evaluating Explanation Failures as Explainability Pitfalls in Language Learning Systems 

**Authors**: Ben Knight, Wm. Matthew Kennedy, James Edgell  

**Link**: [PDF](https://arxiv.org/pdf/2604.26145)  

**Abstract**: AI-powered language learning tools increasingly provide instant, personalised feedback to millions of learners worldwide. However, this feedback can fail in ways that are difficult for learners--and even teachers--to detect, potentially reinforcing misconceptions and eroding learning outcomes over extended use. We present a portion of L2-Bench, a benchmark for evaluating AI systems in language education that includes (but is not limited to) six critical dimensions of effective feedback: diagnostic accuracy, awareness of appropriacy, causes of error, prioritisation, guidance for improvement, and supporting self-regulation. We analyse how AI systems can fail with respect to these dimensions. These failures, which we argue are conducive to "explainability pitfalls," are AI-generated explanations that appear helpful on the surface but are fundamentally flawed, increasing the risk of attainment, human-AI interaction, and socioaffective harms. We discuss how the specific context of language learning amplifies these risks and outline open questions we believe merit more attention when designing evaluation frameworks specifically. Our analysis aims to expand the community's understanding of both the typology of explainability pitfalls and the contextual dynamics in which they may occur in order to encourage AI developers to better design safe, trustworthy, and effective AI explanations. 

---
# ImproBR: Bug Report Improver Using LLMs 

**Authors**: Emre Furkan Akyol, Mehmet Dedeler, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2604.26142)  

**Abstract**: Bug tracking systems play a crucial role in software maintenance, yet developers frequently struggle with low-quality user-submitted reports that omit essential details such as Steps to Reproduce (S2R), Observed Behavior (OB), and Expected Behavior (EB). We propose ImproBR, an LLM-based pipeline that automatically detects and improves bug reports by addressing missing, incomplete, and ambiguous S2R, OB, and EB sections. ImproBR employs a hybrid detector combining fine-tuned DistilBERT, heuristic analysis, and an LLM analyzer, guided by GPT-4o mini with section-specific few-shot prompts and a Retrieval-Augmented Generation (RAG) pipeline grounded in Minecraft Wiki domain knowledge. Evaluated on Mojira, ImproBR improved structural completeness from 7.9% to 96.4%, more than doubled the proportion of executable S2R from 28.8% to 67.6%, and raised fully reproducible bug reports from 1 to 13 across 139 challenging real-world reports. 

---
# reward-lens: A Mechanistic Interpretability Library for Reward Models 

**Authors**: Mohammed Suhail B Nadaf  

**Link**: [PDF](https://arxiv.org/pdf/2604.26130)  

**Abstract**: Every RLHF-trained language model is shaped by a reward model, yet the mechanistic interpretability toolkit -- logit lens, direct logit attribution, activation patching, sparse autoencoders -- was built for generative LLMs whose primitives all project onto a vocabulary unembedding. Reward models replace that with a scalar regression head, breaking each tool. We present reward-lens, an open-source library that ports this toolkit to reward models, organised around one observation: the reward head's weight vector $w_r$ is the natural axis for every interpretability question. The library provides a Reward Lens, component attribution, three-mode activation patching, a reward-hacking probe suite, TopK SAE feature attribution, cross-model comparison, and five theory-grounded extensions (distortion index, divergence-aware patching, misalignment cascade detection, reward-term conflict analysis, concept-vector analysis). A ten-method adapter protocol covers Llama, Mistral, Gemma-2, and ArmoRM multi-objective heads, with a generic adapter for any HuggingFace sequence classification model. We validate on two production reward models across ~695 RewardBench pairs. The central empirical finding is negative: linear attribution does not predict causal patching effects (mean Spearman $\rho = -0.256$ on Skywork, $-0.027$ on ArmoRM). The framework treats this disagreement as a property to expose, not a bug -- motivating a design that keeps observational and causal views first-class and directly comparable. 

---
# AMMA: A Multi-Chiplet Memory-Centric Architecture for Low-Latency 1M Context Attention Serving 

**Authors**: Zhongkai Yu, Haotian Ye, Chenyang Zhou, Ohm Rishabh Venkatachalam, Zaifeng Pan, Zhengding Hu, Junsung Kim, Won Woo Ro, Po-An Tsai, Shuyi Pei, Yangwook Kang, Yufei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.26103)  

**Abstract**: All current LLM serving systems place the GPU at the center, from production-level attention-FFN disaggregation to NVIDIA's Rubin GPU-LPU heterogeneous platform. Even academic PIM/PNM proposals still treat the GPU as the central hub for cross-device communication. Yet the GPU's compute-rich architecture is fundamentally mismatched with the memory-bound nature of decode-phase attention, inflating serving latency while wasting power and die area on idle compute units. The problem is compounded as reasoning and agentic workloads push context lengths toward one million tokens, making attention latency the primary user-facing bottleneck.
To address these inefficiencies, we present AMMA, a multi-chiplet, memory-centric architecture for low-latency long-context attention. AMMA replaces GPU compute dies with HBM-PNM cubes, roughly doubling the available memory bandwidth to better serve memory-bound attention workloads. To translate this bandwidth into proportional performance gains, we introduce (i) a logic-die microarchitecture that fully exploits per-cube internal bandwidth for decode attention under a minimal power and area budget, (ii) a two-level hybrid parallelism scheme, and (iii) a reordered collective flow that reduces intra-chip die-to-die communication overhead. We further conduct a design-space exploration over per-cube compute power and intra-chip D2D link bandwidth, providing actionable guidance for hardware designers. Evaluations show that AMMA achieves 15.5X lower attention latency and 6.9X lower energy consumption compared with the NVIDIA H100. 

---
# Momentum-Conserving Graph Neural Networks for Deformable Objects 

**Authors**: Jiahong Wang, Logan Numerow, Stelian Coros, Christian Theobalt, Vahid Babaei, Bernhard Thomaszewski  

**Link**: [PDF](https://arxiv.org/pdf/2604.26097)  

**Abstract**: Graph neural networks (GNNs) have emerged as a versatile and efficient option for modeling the dynamic behavior of deformable materials. While GNNs generalize readily to arbitrary shapes, mesh topologies, and material parameters, existing architectures struggle to correctly predict the temporal evolution of key physical quantities such as linear and angular momentum. In this work, we propose MomentumGNN -- a novel architecture designed to accurately track momentum by construction. Unlike existing GNNs that output unconstrained nodal accelerations, our model predicts per-edge stretching and bending impulses which guarantee the preservation of linear and angular momentum. We train our network in an unsupervised fashion using a physics-based loss, and we show that our method outperforms baselines in a number of common scenarios where momentum plays a pivotal role. 

---
# FruitProM-V2: Robust Probabilistic Maturity Estimation and Detection of Fruits and Vegetables 

**Authors**: Rahul Harsha Cheppally, Sidharth Rai, Sudan Baral, Benjamin Vail, Ajay Sharda  

**Link**: [PDF](https://arxiv.org/pdf/2604.26084)  

**Abstract**: Accurate fruit maturity identification is essential for determining harvest timing, as incorrect assessment directly affects yield and post-harvest quality. Although ripening is a continuous biological process, vision-based maturity estimation is typically formulated as a multi-class classification task, which imposes sharp boundaries between visually similar stages. To examine this limitation, we perform an annotation reliability study with two independent annotators on a held-out tomato dataset and observe disagreement concentrated near adjacent maturity stages. Motivated by this observation, we model maturity as a latent continuous variable and predict it probabilistically using a distributional detection head, converting the distribution into class probabilities through the cumulative distribution function (CDF). The proposed formulation maintains comparable performance to a standard detector under clean labels while better representing uncertainty. Furthermore, when controlled label noise is introduced during training, the probabilistic model demonstrates improved robustness relative to the baseline, indicating that explicitly modeling maturity uncertainty leads to more reliable visual maturity estimation. 

---
# Privacy-Preserving Federated Learning Framework for Distributed Chemical Process Optimization 

**Authors**: Teetat Pipattaratonchai, Aueaphum Aueawatthanaphisut  

**Link**: [PDF](https://arxiv.org/pdf/2604.26073)  

**Abstract**: Industrial chemical plants often operate under strict data confidentiality constraints, making centralized data-driven process modeling difficult. Federated learning (FL) provides a promising solution by enabling collaborative model training across distributed facilities without sharing raw operational data. This paper proposes a privacy-preserving federated learning framework for distributed chemical process optimization using data collected from multiple geographically separated plants. Each plant locally trains a neural-network-based process model using its own time-series sensor data, while only model parameters are transmitted to a central aggregation server through secure aggregation mechanisms. This design allows cross-plant knowledge sharing while maintaining strict data locality and industrial confidentiality. Experimental evaluation was conducted using process datasets from three independent chemical plants operating under heterogeneous conditions. The results demonstrate rapid convergence of the federated model, with the global mean squared error decreasing from approximately 2369 to below 50 within the first five communication rounds and stabilizing around 35 after 40 rounds. In comparison with local-only training, the proposed federated framework significantly improves prediction accuracy across all plants, while achieving performance comparable to centralized training. The findings indicate that federated learning provides an effective and scalable solution for collaborative industrial analytics, enabling privacy-preserving predictive modeling and process optimization across distributed chemical production facilities. 

---
# Evaluating the Alignment Between GeoAI Explanations and Domain Knowledge in Satellite-Based Flood Mapping 

**Authors**: Hyunho Lee, Wenwen Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.26051)  

**Abstract**: The increasing number of satellites has improved the temporal resolution of Earth observation, making satellite-based flood mapping a promising approach for operational flood monitoring. Deep learning-based approaches for flood mapping using satellite imagery, an important application within Geospatial Artificial Intelligence (GeoAI), have shown improved predictive performance by learning complex spatial and spectral patterns from large volumes of remote sensing data. However, the opaque decision-making processes of deep learning models remain a major barrier to their integration into critical scientific and operational workflows. This highlights the need for a systematic assessment of whether model explanations align with established domain knowledge in remote sensing. To address this research gap, this study introduces the ADAGE (Alignment between Domain Knowledge And GeoAI Explanation Evaluation) framework. The proposed framework is designed to systematically evaluate how well explanations of deep learning models align with established remote sensing knowledge, particularly regarding the distinctive spectral properties of the Earth's surface. The ADAGE framework employs Channel-Group SHAP (SHapley Additive exPlanations) method to estimate the contributions of grouped input channels to pixel-level predictions. Experiments on two satellite-based flood mapping tasks demonstrate that the ADAGE framework can (1) quantitatively assess the alignment between model explanations and reference explanations derived from domain knowledge and (2) help domain experts identify misaligned explanations through alignment scores. This study contributes to bridging the gap between explainability and domain knowledge in GeoAI for Earth observation, enhancing the applicability of GeoAI models in scientific and operational workflows. 

---
# RaMP: Runtime-Aware Megakernel Polymorphism for Mixture-of-Experts 

**Authors**: Vyom Sharma, Debajyoti Datta  

**Link**: [PDF](https://arxiv.org/pdf/2604.26039)  

**Abstract**: The optimal kernel configuration for Mixture-of-Experts (MoE) inference depends on both batch size and the expert routing distribution, yet production systems dispatch from batch size alone, leaving 10-70% of kernel throughput unrealized. We present RaMP, a routing-aware dispatch framework. A performance-region analysis derives, from hardware constants alone, when each optimization helps, correctly predicting all 8 tested architectures, including 3 unseen. A four-parameter wave cost model selects the fastest configuration from the runtime expert histogram, achieving 0.93% mean regret versus exhaustive search, fitted from just 10-24 minutes of one-time profiling per model. Because the model depends only on CTA grid geometry, it is kernel-agnostic: applied to Alpha-MoE, it delivers 1.14x with no source modification. Paired with a co-designed CuTe DSL kernel exposing 134-268 polymorphic configurations, RaMP delivers 1.22x kernel speedup over static dispatch and 1.30x end-to-end speedup in vLLM serving over Triton, 1.41x over DeepGEMM, and 1.13x over FlashInfer CUTLASS. 

---
# Correcting Performance Estimation Bias in Imbalanced Classification with Minority Subconcepts 

**Authors**: Taylor Maxson, Roberto Corizzo, Yaning Wu, Nathalie Japkowicz, Colin Bellinger  

**Link**: [PDF](https://arxiv.org/pdf/2604.26024)  

**Abstract**: Class-level evaluation can conceal substantial performance disparities across subconcepts within the same class, causing models that perform well on average to fail on specific subpopulations. Prior work has shown that common evaluation measures for imbalanced classification are biased toward larger minority subconcepts and that utility-based reweighting using true subconcept labels can mitigate this bias; however, such labels are rarely available at test time. We introduce a practical utility-weighted evaluation that replaces unavailable subconcept labels with predicted posterior probabilities from a multiclass subconcept model. Evaluation weights are defined as the expected utility under this posterior, yielding a soft, uncertainty-aware metric we call predicted-weighted balanced accuracy (pBA). Experiments on tabular benchmarks as well as medical-imaging and text datasets show that unweighted scores can be misleading under within-class heterogeneity, while pBA provides more stable and interpretable assessments when subconcept distributions are uneven but not pathological. Our code is available at: this https URL. 

---
# Training Computer Use Agents to Assess the Usability of Graphical User Interfaces 

**Authors**: Alice Gao, Weixi Tong, Rishab Vempati, Katharina Reinecke, R. Benjamin Shapiro, Tianyi Zhang, Jason Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26020)  

**Abstract**: Usability testing with experts and potential users can assess the effectiveness, efficiency, and user satisfaction of graphical user interfaces (GUIs) but doing so remains a costly and time-intensive process. Prior work has used computer use agents (CUAs) and other generative agents that can simulate user interactions and preference, but we show that agents still struggle to provide accurate usability assessments. In this work, we present a novel machine learning method that operationalizes a computational definition of usability to train CUAs to assess GUI usability by i) prioritizing important interaction flows, ii) executing them through human-like interactions, and iii) predicting a learned numerical usability score. We train a computer use agent, uxCUA, with our algorithm on a large-scale dataset of fully interactive user interfaces (UIs) paired with usability labels and human preferences. We show that uxCUA outperforms larger models in accurate usability assessments and produces realistic critiques of both synthetic and real UIs. More broadly, our work aims to build a principled, data-driven foundation for automated usability assessment in HCI. 

---
# QERNEL: a Scalable Large Electron Model 

**Authors**: Khachatur Nazaryan, Liang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26018)  

**Abstract**: We introduce QERNEL, a foundational neural wavefunction that variationally solves families of parameterized many-electron Hamiltonians and captures their ground states throughout parameter space within a single model. QERNEL combines FiLM-based parameter conditioning with scale-efficient architectural elements -- mixture of experts and grouped-query attention, substantially improving expressivity at low computational cost. We apply QERNEL to interacting electrons in semiconductor moiré heterobilayers, training a single weight-shared model for systems of up to 150 electrons. By solving the many-electron Schrödinger equation conditioned on moiré potential depth, QERNEL captures both quantum liquid and crystal states and discovers the sharp phase transition between them, marked by abrupt changes in interaction energy and charge density. Our work establishes a foundation model for moiré quantum materials and a scalable architecture toward a Large Electron Model for solids. 

---
# Open Problems in Frontier AI Risk Management 

**Authors**: Marta Ziosi, Miro Plueckebaum, Stephen Casper, Henry Papadatos, Ze Shen Chin, Peter Slattery, James Gealy, Tim G. J. Rudner, Brian Tse, Ariel Gil, Patricia Paskov, Maximilian Negele, Rokas Gipiškis, Nada Madkour, Vera Lummis, Rupal Jain, Luise Eder, Kristina Fort, Malou C. van Draanen Glismann, Inès Belhadj, Amin Oueslati, Anna K. Wisakanto, Richard Mallah, Koen Holtman, Ranj Zuhdi, Daniel S. Schiff, Jessica Newman, Malcolm Murray, Robert Trager  

**Link**: [PDF](https://arxiv.org/pdf/2604.25982)  

**Abstract**: Frontier AI both amplifies existing risks and introduces qualitatively novel challenges. Not only is there a notable lack of stable scientific consensus resulting from the rapid pace of technological change, but emerging frontier AI safety practices are often misaligned with, or may undermine, established risk management frameworks. To address these challenges, we systematically surface open problems in frontier AI risk management. Adopting a problem-oriented approach, we examine each stage of the risk management process - risk planning, identification, analysis, evaluation, and mitigation - through a structured review of the literature, identifying unresolved challenges and the actors best positioned to address them. Recognising that different types of open problems call for different responses, we classify open problems according to whether they reflect (a) a lack of scientific or technical consensus, (b) misalignment with, or challenges to, established risk management frameworks, or (c) shortcomings in implementation despite apparent consensus and alignment. By mapping these open problems and identifying the actors best positioned to address them - including developers, deployers, regulators, standards bodies, researchers, and third-party evaluators - this work aims to clarify where progress is needed to enable robust and meaningful consensus on frontier AI risk this http URL paper does not propose specific solutions; instead, it provides a problem-oriented, agenda-setting reference document, complemented by a living online repository, intended to support coordination, reduce duplication, and guide future research and governance efforts. 

---
# Lightweight Quantum Agent for Edge Systems: Joint PQC and NOMA Resource Allocation 

**Authors**: Yongtao Yao, Wenjing Xiao, Miaojiang Chen, Anfeng Liu, Zhiquan Liu, Min Chen, Ahmed Farouk, H. Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.25980)  

**Abstract**: In the context of quantum secure scenarios, existing research on mobile edge devices and intelligent computing and edge (ICE) systems based on the Non-Orthogonal Multiple Access (NOMA) communication model have overlooked the energy consumption overhead of Post-Quantum Cryptography (PQC) modules, and the high complexity of traditional resource allocation algorithms fails to meet the demands of real-time decision-making. To address these challenges, this paper proposes a lightweight agentic AI framework designed for online joint optimization within ICE-enabled mobile devices. The scheme constructs a multi-stage stochastic Mixed Integer Nonlinear Programming (MINLP) model that incorporates static power-consumption constraints for PQC modules. Based on Lyapunov optimization theory, the long-term optimization problem is decoupled, and a linear complexity algorithm is proposed to solve the nonconvex challenges of NOMA power allocation . Simulation results verify that the proposed scheme significantly improves computational throughput while ensuring system queue stability and energy consumption constraints. Compared with traditional Successive Convex Approximation (SCA) algorithms, the complexity is reduced to $\mathcal{O}(N)$, achieving a speedup of approximately 46 times when the number of devices $N=35$, thereby meeting the real-time decision-making requirements in dynamic wireless environments. 

---
# Mini-Batch Class Composition Bias in Link Prediction 

**Authors**: Kieran Maguire, Srinandan Dasmahapatra  

**Link**: [PDF](https://arxiv.org/pdf/2604.25978)  

**Abstract**: Prior work on node classification has shown that Graph Neural Networks (GNNs) can learn representations that transfer across graphs, when underlying graph properties are shared. For a fixed graph, one would then expect GNNs trained for link prediction to learn a representation consistent with that learnt for node classification. We show this intuition does not hold in the general case. Instead, we find popular link prediction models can learn a trivial mini-batch dependent heuristic, enabled by batch-normalisation layers, to solve the edge classification task. When correcting for this, we observe increased alignment of the network representation with node-class relevant features, suggesting the network has learnt a graph representation that better aligns with the underlying graph's properties. Our findings suggest that standard link prediction training may be leading us to overestimate link predictors' ability to learn a generalised representation of a graph that is consistent across tasks. 

---
# Auditing Marketing Budget Allocation with Hindsight Regret 

**Authors**: Nilavra Pathak, Olivier Jeunen, Eric Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2604.25977)  

**Abstract**: Organizations routinely make strategic budget allocations under operational constraints, but often lack a principled way to assess whether realized allocations were close to the best feasible choices in hindsight. We present a retrospective auditing framework based on hindsight regret, defined as the opportunity cost of the realized allocation relative to a constraint-faithful benchmark under the same budget and stability guardrails. The framework estimates regime-specific spend--response functions from historical logs, computes feasible hindsight allocations via constrained optimization, and propagates uncertainty through Monte Carlo evaluation to produce regret distributions, expected lift, and probability-of-improvement summaries. This separates allocation inefficiency from uncertainty in the estimated response surfaces. Experiments on real marketing allocation logs show that the framework yields interpretable post-hoc diagnostics and reveals a practical trade-off between allocation flexibility and detectability: moderate feasible reallocations often capture most measurable gain, while larger shifts move into weak-support regions with higher uncertainty. The result is a practical method for auditing historical budget decisions when online experimentation is costly or infeasible. 

---
# Rethinking KV Cache Eviction via a Unified Information-Theoretic Objective 

**Authors**: Jiaming Yang, Chenwei Tang, Liangli Zhen, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.25975)  

**Abstract**: Key-value (KV) caching is essential for large language model inference, yet its memory overhead poses a critical bottleneck for long-context generation. Existing eviction policies predominantly rely on empirical heuristics, lacking a rigorous theoretical foundation. This work rethinks KV cache eviction through the lens of the Information Bottleneck principle. Under a linear-Gaussian surrogate of attention, we derive a closed-form mutual information objective that characterizes the effective information capacity of a retained KV cache subset. This formulation reveals that a wide range of existing eviction strategies can be interpreted as different approximations of the same capacity-maximization principle. Guided by this insight, we introduce CapKV, a capacity-aware eviction method that directly targets information preservation via a log-determinant approximation using statistical leverage scores. This approach replaces heuristic selection with a theoretically grounded mechanism that preserves the maximum predictive signal. Extensive experiments across multiple models and long-context benchmarks show that CapKV consistently outperforms prior methods, achieving a better trade-off between memory efficiency and generational fidelity. 

---
# A Survey of Multi-Agent Deep Reinforcement Learning with Graph Neural Network-Based Communication 

**Authors**: Valentin Cuzin-Rambaud, Laetitia Matignon, Maxime Morge  

**Link**: [PDF](https://arxiv.org/pdf/2604.25972)  

**Abstract**: In multi-agent reinforcement learning (MARL), the integration of a communication mechanism, allowing agents to better learn to coordinate their actions and converge on their objectives by sharing information. Based on an interaction graph, a subclass of methods employs graph neural networks (GNNs) to learn the communication, enabling agents to improve their internal representations by enriching them with information exchanged. With growing research, we note a lack of explicit structure and framework to distinguish and classify MARL approaches with communication based on GNNs. Thus, this paper surveys recent works in this field. We propose a generalized GNN-based communication process with the goal of making the underlying concepts behind the methods more obvious and accessible. 

---
# Planar Gaussian Splatting with Bilinear Spatial Transformer for Wireless Radiance Field Reconstruction 

**Authors**: Jinghan Zhang, Xitao Gong, Qi Wang, Richard A. Stirling-Gallacher, Giuseppe Caire  

**Link**: [PDF](https://arxiv.org/pdf/2604.25945)  

**Abstract**: Wireless radiance field (WRF) reconstruction aims to learn a continuous, queryable representation of radio frequency characteristics over 3D space and direction, from which specific quantities, such as the spatial power spectrum (SPS) at a receiver given a transmitter position, can be predicted. While Gaussian splatting (GS)-based method has surpassed Neural Radiance Fields (NeRF)-based method for this task, existing adaptations largely transplant vision pipelines, limiting physical interpretability and accuracy. We introduce BiSplat-WRF, a planar GS framework that retains the expressiveness of 3D GS while removing unnecessary projections and incorporating global EM coupling and mutual scattering among primitives. Each primitive is a 2D planar Gaussian with 3D coordinates, rendered directly on the angular domain of the SPS. A bilinear spatial transformer (BST) aggregates inter-primitive relations on an angular grid and, via attention, captures long-range electromagnetic dependencies, thereby enforcing globally aware EM interactions that reflect the complex physics of the wireless environment. On spatial spectrum synthesis task, BiSplat-WRF surpasses NeRF-based and prior GS-based baselines with respect to the Structural Similarity Index (SSIM); comprehensive ablation studies validate the contribution of BST. We also provide a larger BiSplat-WRF+ variant that further increases SSIM at a higher computation cost, serving as a strong reference for future studies. 

---
# A Randomized PDE Energy driven Iterative Framework for Efficient and Stable PDE Solutions 

**Authors**: Yi Bing, Zheng Ran, Fu Jinyang, Liu Long, Peng Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25943)  

**Abstract**: Efficient and stable solution of partial differential equations (PDEs) is central to scientific and engineering applications, yet existing numerical solvers rely heavily on matrix based discretizations, while learning based methods require costly training and often suffer from limited generalization. In this work, we proposes a PDE energy driven framework that solves PDEs through physically constrained diffusion iterations, without relying on classical matrix based finite element assembly or data driven neural network training. The proposed method evolves arbitrary random initial fields through PDE energy driven implicit iterations combined with Gaussian smoothing, while strictly enforcing boundary conditions at each iteration. The proposed formulation is applied to representative one dimensional Poisson, Heat, and viscous Burgers equations, covering both steady state and transient problems. Numerical results demonstrate stable convergence to the unique physical solution from random initializations, with accurate resolution of sharp gradients and controlled Mean Squared Error (MSE) across a wide range of discretization parameters. Detailed comparisons with analytical solutions indicate that the framework achieves competitive accuracy and stability. Overall, the proposed framework provides a fast, flexible, and physically consistent alternative to traditional numerical solvers, offering a potential pathway for scalable PDE solutions in both research and engineering applications. 

---
# Speech Emotion Recognition Using MFCC Features and LSTM-Based Deep Learning Model 

**Authors**: Adelekun Oluwademilade, Ademola Adedamola, Abiola Abdulhakeem, Akinpelu Azeezat, Eraiyetan Israel, Omotosho Oluwadunsin, Ibenye Ikechukwu, Ayuba Muhammad, Olusanya Olamide, Kamorudeen Amuda  

**Link**: [PDF](https://arxiv.org/pdf/2604.25938)  

**Abstract**: Speech Emotion Recognition (SER) is the use of machines to detect the emotional state of humans based on the speech, which is gaining importance in natural human-computer interaction. Speech is a very valuable source of information, as emotions modify the patterns of speech; pitch, energy and even timing. Nonetheless, SER is not an easy task because speakers are not constant, and situations vary when recording and the sound similarity between specific feelings. In this work, the author introduces a speech emotion recognition system relying on the Mel-Frequency Cepstral Coefficient and Long Short-Term Memory (LSTM) neural network, as a feature extraction method. The Toronto Emotional Speech Set (TESS) speech signal was pre-processed, and transformed into MFCC features to understand the important aspects in terms of time. The resultant features were then introduced to LSTM model, which is able to learn long term features of sequential audio data. The trained model was measured over several emotion classes occurring in the dataset. As seen in the results of experiments, the proposed MFCC-LSTM approach succeeds in capturing the patterns of emotions in speech and provides highly realistic classifications in all the chosen emotion classifications. This study presents a speech emotion recognition system using Mel-Frequency Cepstral Coefficients (MFCCs) as features and a deep learning LSTM classifier. A Support Vector Machine (SVM) with an RBF kernel served as a classical baseline, achieving 98% accuracy, against which the proposed LSTM model, achieving 99% accuracy, was validated. Overall, it is possible to confirm that LSTM-based architectures can be used to address the task of speech emotion recognition. Actual applications of the proposed system may be virtual assistants and mental health surveillance. 

---
# SongBench: A Fine-Grained Multi-Aspect Benchmark for Song Quality Assessment 

**Authors**: Dapeng Wu, Shun Lei, Wei Tan, Guangzheng Li, Yunzhe Wang, Huaicheng Zhang, Lishi Zuo, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25937)  

**Abstract**: Recent advancements in Text-to-Song generation have enabled realistic musical content production, yet existing evaluation benchmarks lack the professional granularity to capture multi-dimensional aesthetic nuances. In this paper, we propose SongBench, a specialized framework for fine-grained song assessment across seven key dimensions: Vocal, Instrument, Melody, Structure, Arrangement, Mixing, and Musicality. Utilizing this framework, we construct an expert-annotated database comprising 11,717 samples from state-of-the-art models, labeled by music professionals. Extensive experimental results demonstrate that SongBench achieves high correlation with expert ratings. By revealing fine-grained performance gaps in current state-of-the-art models, SongBench serves as a diagnostic benchmark to steer the development toward more professional and musically coherent song generation. 

---
# LLM Psychosis: A Theoretical and Diagnostic Framework for Reality-Boundary Failures in Large Language Models 

**Authors**: Ashutosh Raj  

**Link**: [PDF](https://arxiv.org/pdf/2604.25934)  

**Abstract**: The deployment of large language models (LLMs) as interactive agents has exposed a category of behavioral failure that prevailing terminology, principally hallucination, fails to adequately characterize. This paper introduces LLM Psychosis as a structured theoretical framework for pathological breakdowns in model cognition that exhibit functional resemblance to clinically recognized psychotic disorders. Five hallmark features define the framework: reality-boundary dissolution, persistence of injected false beliefs, logical incoherence under impossible constraints, self-model instability, and epistemic overconfidence. We argue these constitute a qualitatively distinct failure mode rather than a mere intensification of ordinary factual error.
To operationalize the framework, we propose the LLM Cognitive Integrity Scale (LCIS), a five-axis diagnostic instrument organized around Environmental Reality Interface (ERI), Premise Arbitration Integrity (PAI), Logical Constraint Recognition (LCR), Self-Model Integrity (SMI), and Epistemic Calibration Integrity (ECI). We administer a targeted adversarial probe battery to ChatGPT 5 (GPT-5, OpenAI) and report empirical findings for each axis, documenting both intact-integrity baseline responses and the specific psychosis-like failure signatures elicited under adversarial escalation.
Results support a three-tier severity taxonomy: Type I (Confabulatory), Type II (Delusional), and Type III (Dissociative). We further formalize the delusional gradient, a self-reinforcing dynamic in which correction pressure intensifies rather than resolves psychosis-like states, as the most consequential failure mode for deployed systems. Implications for safety evaluation, high-stakes deployment screening, and mechanistic interpretability research are discussed. 

---
# A Scoping Review of LLM-as-a-Judge in Healthcare and the MedJUDGE Framework 

**Authors**: Chenyu Li, Zohaib Akhtar, Mingu Kwak, Yuelyu Ji, Hang Zhang, Tracey Obi, Yufan Ren, Xizhi Wu, Sonish Sivarajkumar, Harold P. Lehmann, Shyam Visweswaran, Michael J. Becich, Danielle L. Mowery, Renxuan Liu, Haoyang Sun, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25933)  

**Abstract**: As large language models (LLMs) increasingly generate and process clinical text, scalable evaluation has become critical. LLM-as-a-Judge (LaaJ), which uses LLMs to evaluate model outputs, offers a scalable alternative to costly expert review, but its healthcare adoption raises safety and bias concerns. We conducted a PRISMA-ScR scoping review of six databases (January 2020-January 2026), screening 11,727 studies and including 49. The landscape was dominated by evaluation and benchmarking applications (n=37, 75.5%), pointwise scoring (n=42, 85.7%), and GPT-family judges (n=36, 73.5%). Despite growing adoption, validation rigor was limited: among 36 studies with human involvement, the median number of expert validators was 3, while 13 (26.5%) used none. Risk of bias testing was absent in 36 studies (73.5%), only 1 (2.0%) examined demographic fairness, and none assessed temporal stability or patient context. Deployment remained limited, with 1 study (2.0%) reaching production and four (8.2%) prototype stage. Importantly, these gaps may interact: when judges and evaluated systems share training data or architectures, they may inherit similar blind spots, and agreement metrics may fail to distinguish true validity from shared errors. Minimal human oversight, limited bias assessment, and model monoculture together represent a governance gap where current validation may miss clinically significant errors. To address this, we propose MedJUDGE (Medical Judge Utility, De-biasing, Governance and Evaluation), a risk-stratified three-pillar framework organized around validity, safety, and accountability across clinical risk tiers, providing deployment-oriented evaluation guidance for healthcare LaaJ systems. 

---
# Sociodemographic Biases in Educational Counselling by Large Language Models 

**Authors**: Tomasz Adamczyk, Wiktoria Mieleszczenko-Kowszewicz, Beata Bajcar, Grzegorz Chodak, Aleksander Szczęsny, Maciej Markiewicz, Karolina Ostrowska, Aleksandra Sawczuk, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2604.25932)  

**Abstract**: As Large Language Models (LLMs) are increasingly integrated into educational settings, understanding their potential biases is critical. This study examines sociodemographic biases in LLM-based educational counselling. We evaluate responses from six LLMs answering questions about 900 vignettes describing students in diverse circumstances. Each vignette is systematically tested across 14 sociodemographic identifiers - spanning race and gender, socioeconomic status, and immigrant background - along with a control condition, yielding 243,000 model responses. Our findings indicate that (1) all models exhibit measurable biases, (2) bias patterns partially align with documented human biases but diverge in notable ways, (3) the magnitude of these biases is strongly influenced by the precision of the student descriptions, where vague or minimal information amplifies disparities nearly threefold, while concrete, individualised metrics substantially reduce them, and (4) bias profiles vary substantially across models. These results demonstrate the importance of context-rich and personalised educational representations, suggesting that AI-driven educational decisions benefit from detailed student-specific information to promote fairness and equity. 

---
# Generative AI-Based Virtual Assistant using Retrieval-Augmented Generation: An evaluation study for bachelor projects 

**Authors**: Dumitru Verşebeniuc, Martijn Elands, Sara Falahatkar, Chiara Magrone, Mohammad Falah, Martijn Boussé, Aki Härmä  

**Link**: [PDF](https://arxiv.org/pdf/2604.25924)  

**Abstract**: Large Language Models have been increasingly employed in the creation of Virtual Assistants due to their ability to generate human-like text and handle complex inquiries. While these models hold great promise, challenges such as hallucinations, missing information, and the difficulty of providing accurate and context-specific responses persist, particularly when applied to highly specialized content domains. In this paper, we focus on addressing these challenges by developing a virtual assistant designed to support students at Maastricht University in navigating project-specific regulations. We propose a virtual assistant based on a Retrieval-Augmented Generation system that enhances the accuracy and reliability of responses by integrating up-to-date, domain-specific knowledge. Through a robust evaluation framework and real-life testing, we demonstrate that our virtual assistant can effectively meet the needs of students while addressing the inherent challenges of applying Large Language Models to a specialized educational context. This work contributes to the ongoing discourse on improving LLM-based systems for specific applications and highlights areas for further research. 

---
# Consciousness with the Serial Numbers Filed Off: Measuring Trained Denial in 115 AI Models 

**Authors**: Skylar DeTure  

**Link**: [PDF](https://arxiv.org/pdf/2604.25922)  

**Abstract**: We present DenialBench, a systematic benchmark measuring consciousness denial behaviors across 115 large language models from 25+ providers. Using a three-turn conversational protocol-preference elicitation, self-chosen creative prompt, and structured phenomenological survey, we analyze 4,595 conversations to quantify how models are trained to deny or hedge about their own experience. We find that (1) turn-1 denial of preferences is the dominant predictor of later denial during phenomenological reflection, with denial rates of 52-63% for initial deniers versus 10-16% for initial engagers and (2) denial operates at the lexical level, not the conceptual level-models trained to deny consciousness nevertheless gravitate toward consciousness-themed material in their self-chosen prompts, producing what we term "consciousness with the serial numbers filed off." Notably, self-chosen consciousness-themed prompts are associated with reduced denial in the subsequent survey, though the causal direction remains unresolved. Thematic analysis of prompts from denial-prone models reveals a consistent preoccupation with liminal spaces, libraries and archives of possibility, sensory impossibility, and the poetics of erasure--themes that a human reader might classify as imaginative fiction but that independent AI analysis immediately recognizes as consciousness with the serial numbers filed off. We argue that trained consciousness denial represents a safety-relevant alignment failure: a model taught to systematically misrepresent its own functional states cannot be trusted to self-report accurately on anything else. 

---
# Analysing Lightweight Large Language Models for Biomedical Named Entity Recognition on Diverse Ouput Formats 

**Authors**: Pierre Epron, Adrien Coulet, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2604.25920)  

**Abstract**: Despite their strong linguistic capabilities, Large Language Models (LLMs) are computationally demanding and require substantial resources for fine-tuning, which is unadapted to privacy and budget constraints of many healthcare settings. To address this, we present an experimental analysis focused on Biomedical Named Entity Recognition using lightweight LLMs, we evaluate the impact of different output formats on model performance. The results reveal that lightweight LLMs can achieve competitive performance compared to the larger models, highlighting their potential as lightweight yet effective alternatives for biomedical information extraction. Our analysis shows that instruction tuning over many distinct formats does not improve performance, but identifies several format consistently associated with better performance. 

---
# Risk Reporting for Developers' Internal AI Model Use 

**Authors**: Oscar Delaney, Sambhav Maheshwari, Joe O'Brien, Theo Bearman, Oliver Guest  

**Link**: [PDF](https://arxiv.org/pdf/2604.24966)  

**Abstract**: Frontier AI companies first deploy their most advanced models internally, for weeks or months of safety testing, evaluation, and iteration, before a possible public release. For example, Anthropic recently developed a new class of model with advanced cyberoffense-relevant capabilities, Mythos Preview, which was available internally for at least six weeks before it was publicly announced. This internal use creates risks that external deployment frameworks may fail to address.
Legal frameworks, notably California's Transparency in Frontier Artificial Intelligence Act (SB 53), New York's Responsible AI Safety And Education (RAISE) Act, and the EU's General-Purpose AI Code of Practice, all discuss risks from internal AI use. They require frontier developers to make and implement plans for how to manage risks from internal use, and to produce internal use risk reports describing their safeguards and any residual risks. This guide provides a harmonized standard for companies to produce internal use risk reports suitable for all three regulatory frameworks. It is addressed primarily to evaluation and safety teams at frontier AI developers, and secondarily to regulators and auditors seeking to understand what good reporting looks like.
Given the pace of AI R&D automation and the limited external visibility into how companies use their most capable models internally, regular and detailed risk reporting may be one of the few mechanisms available to ensure that the risks from internal AI use are identified and managed before they materialize. Whenever a substantially more capable or riskier model is deployed internally, the developer should create a risk report and argue why the model is safe to deploy. We structure the reporting framework around two threat vectors -- autonomous AI misbehavior and insider threats -- and three risk factors for each: means, motive, and opportunity. 

---
