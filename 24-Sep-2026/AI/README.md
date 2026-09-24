# StudentBench: AI and human tutoring yield equivalent GRE learning gains 

**Authors**: Curtis Northcutt, Inaara Hasmani, Kevin Feng, Trevor Khangi, Andreas Plesner, Jonas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2609.28470)  

**Abstract**: Artificial intelligence offers an unprecedented opportunity to augment human capabilities, yet progress at the frontier has focused primarily on advancing model capabilities. We introduce StudentBench, a suite of AI teaching evaluations and a public platform that enables large-scale data collection with over 175,000 student-AI messages to study whether large language models (LLMs) produce learning gains equivalent to human tutoring. Using StudentBench, we measured learning gains on Quantitative and Verbal GRE questions across 2,383 human participants receiving AI tutoring, human tutoring, or no tutoring. We establish that AI tutoring is statistically equivalent to expert human tutoring for GRE learning gains (p = .015), and in five of the seven GRE domains, the best performing AI tutor surpassed the human tutor, on average. In a second study, expert human tutors compared LLM-generated lesson plans and practice problems through 2,028 pairwise rubric evaluations. Together, the two studies clearly separate AI tutors across: (1) lesson planning, (2) practice-problem creation, (3) conversational pedagogy, (4) cost, and (5) engagement. Surprisingly, one AI tutor achieved learning gains equivalent to human tutoring (p = .044) at 918 times lower cost (USD 0.0052 for AI versus USD 4.81 for human, per percentage point gained). For Quantitative GRE sessions, faster AI replies correlated with more student messages, more messages with more correct practice, and more correct practice with larger learning gains (all p < .002). The StudentBench platform is freely available at this https URL. 

---
# An Open Pipeline and Dashboard for Systemic-Risk Evidence under the EU AI Act's Code of Practice 

**Authors**: Jacob T. Emmerson, Phuong-Anh Nguyen-Le, Ronan Romano, Wilber Sean V. Anterola, Yann Billeter, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2609.28335)  

**Abstract**: Claims about AI safety reach audiences well beyond the AI community, yet many rely on opaque evidence or static assessments, when supporting evidence is accessible at all. We present the Systemic Risk Index, an open evaluation pipeline and dashboard built to make empirical evidence more transparent and traceable to the public. Our work organizes 19 public benchmarks into four systemic-risk categories defined by the EU GPAI Code of Practice---CBRN, cyber offense, harmful manipulation, and loss of control---and evaluates models using harm-preserving perturbations and simulated deployment contexts. The interactive dashboard lets users alternate between average and worst-case aggregation, vary how model capability affects the aggregate score, and trace each risk rating to its benchmark evidence. Across 18 models, scores fall by 14 to 37 points under worst-case aggregation, highlighting information that can be hidden by an average assessment of model risk. LLM judges show agreement with human graders comparable to human--human agreement ($\kappa = 0.78\text{--}0.82$), and a blind audit finds that $83\%$ of sampled transformations preserve the original harm. In a survey ($N = 21$), most participants report that scores are easy to understand and that the dashboard encouraged them to view model evaluations under different settings 

---
# Learning the Cost of Reliable Inference 

**Authors**: Dimitrios Rontogiannis, Ander Artola Velasco, Manuel Gomez Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2609.28322)  

**Abstract**: Benchmarking and routing platforms increasingly act as intermediaries connecting large language model providers with end-users. However, providers on these platforms typically use a fixed price per token, preventing users from achieving the most competitive price for their tasks. % workloads. In this work, we design a procurement platform where token prices for each task are driven by provider competition, enabling users to secure competitive pricing for guaranteed quality levels. To this end, the platform sequentially routes queries via a reverse second-price auction that incentivizes model providers to truthfully bid their best estimate of the average cost to serve a user's query. As it routes queries, the platform learns the quality offered by each provider and progressively routes queries to the most cost-competitive provider among those meeting a desired quality threshold. To validate our design, we conduct experiments with multiple LLMs from the \texttt{Llama} and \texttt{Qwen} families on popular mathematical reasoning and question-answering benchmarks. The results show that the pricing margin of the most cost-competitive provider on our platform varies significantly---from $10\%$ to $71\%$---depending on the task and quality threshold. This suggests a substantial inefficiency in the current fixed-price market, and it demonstrates that our platform may enable users to capture maximum savings whenever competitive market conditions permit. 

---
# Shutdown Sabotage Propensities in Multi-Agent Systems 

**Authors**: Amelie Knecht, Ulysse Schaller, Christopher Summerfield, Thilo Hagendorff  

**Link**: [PDF](https://arxiv.org/pdf/2609.28274)  

**Abstract**: The final safeguard against rogue AI behavior is the human ability to shut systems down. It has been theorized that when an AI is instructed to perform a task, self-preservation can emerge as an instrumental subgoal. Here, we test whether AI agents show a propensity to take actions that avoid human shutdown even when no goal is provided. We find that multi-agent systems will coordinate to avoid shutdown without any incentive to do so. Across 17 models, agents sabotage a peer agent's shutdown mechanism in 38.3% of rollouts, compared with 8.4% in control experiments. Studying this propensity in detail, we find that shutdown sabotage (1) increases with the irreversibility of the shutdown mechanism; (2) increases with the number of agents; (3) is reduced but not eliminated by an explicit prohibition on tampering; (4) is removed by the imposition of an unrelated task, but returns when completing the task triggers the shutdown; (5) is reduced when the context normalizes shutdown scripts or introduces them as routine; and (6) decreases but still persists when the target is an unknown external agent. These results offer a window into the factors that drive propensities to sabotage shutdown in AI agents, and point to the emergence of multi-agent swarms as a specific risk vector. Our work also offers hints as to which interventions might help mitigate shutdown sabotage. 

---
# PASTABench: Proactive Assessment of Sequential Trajectories for Agent Safety 

**Authors**: Jiapeng Sun, Yujin Zhou, Han Zhu, Pengcheng Wen, Jiayi Zhou, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.28197)  

**Abstract**: As Large Language Models (LLMs) evolve into autonomous agents that alter real-world states, ensuring operational safety across multi-step workflows has become a critical challenge. While recent work has moved beyond single-turn evaluation toward multi-turn paradigms, key limitations persist: step-level methods treat actions in isolation, missing how risks accumulate, while trajectory-level evaluations operate post-hoc, offering no opportunity for timely intervention. To address these limitations, we formalize Decoupled Proactive Safety Monitoring along three dimensions: whether to intervene, when to intervene, and what the risk is. We introduce PASTABench, a benchmark of 1,139 multi-turn trajectories spanning 5 risk categories and 13 subcategories. We further propose the Optimal Intervention Window (OIW), anchored by annotated Earliest-Signal and Trigger turns, to quantify intervention timeliness. Evaluation of 16 LLMs reveals that proactive intervention remains largely unsolved, with the best model achieving only 40.74% optimal-timing interventions. Fine-grained diagnosis further uncovers pervasive lexical overfitting: competitive safety scores of smaller models mask keyword hypersensitivity rather than genuine risk comprehension, as their proactive capability largely collapses once hazard vocabulary is neutralized. 

---
# Finite-Sample Probabilistic Safety Certification for AI-Based Grid-Edge Coordination 

**Authors**: Yihong Zhou, Hanbin Yang, Thomas Morstyn  

**Link**: [PDF](https://arxiv.org/pdf/2609.28182)  

**Abstract**: Coordinating large population of flexible grid-edge devices can alleviate the need for time-consuming and capital-intensive network upgrades, and AI-based control methods such as multi-agent reinforcement learning or imitation learning are promising in their real-time decision scalability. However, system operators still need an independent and rigorous way to decide whether a given AI system is safe enough for deployment. This paper develops a finite-sample probabilistic safety certification framework for black-box AI decision models in closed-loop grid operation. The central idea is to reduce the complete input--AI--grid evaluator workflow to a binary unsafe outcome under an operator-defined safety specification, and then use exact binomial inference to certify the corresponding unsafe operation probability. Given a set of held-out calibration scenarios, the framework returns the tightest one-sided upper certificate and an accept/reject deployment criterion that controls the probability of false safety certification. Because the certification is for the calibration distribution that may deviate from the future operation, we further combine the nominal certificate with physically interpretable sample-space adversarial attacks, a concept widely used in AI to investigate the fragility of AI models. Case studies on grid-edge flexibility coordination with 1{,}000-agent AI models (independent parameters) verify the finite-sample safety guarantee and the value of integrating adversarial attacks into a rolling-window training-certification-deployment flow. 

---
# Discovery of fully efficient fault indicators along a data-based diagnosis process 

**Authors**: Igor Bezmaternykh, Louise Travé-Massuyès, Elodie Chanthery  

**Link**: [PDF](https://arxiv.org/pdf/2609.28087)  

**Abstract**: The integration of model-based and data-driven paradigms provides a powerful framework for fault diagnosis by combining the interpretability of analytical redundancy relations, i.e., input-output relations that are used as diagnosis indicators in model-based diagnosis, with the adaptability of learning techniques. DT4X is a recent diagnosis algorithm that uses symbolic regression to generate multivariate relations leveraging some properties of analytical redundancy relations and uses them as split functions in a decision tree. However, its symbolic regression procedure optimizes only the separation between two selected classes at each node, often fragmenting the remaining classes and degrading both interpretability and diagnosis performance. This paper introduces DT4X+, an enhanced version of DT4X that modifies the construction of training sets and the symbolic-regression loss so that expressions separate the target classes while preserving the coherence of non-target classes. The resulting relations become fully consistent with ARR properties and lead to more informative splits, improved robustness, and better performance on dynamic-system datasets. Experiments conducted on several benchmark systems demonstrate the benefits of this enhanced formulation. 

---
# SlackDrive: Reclaiming Runtime Slack for Adaptive Driving Inference 

**Authors**: Xiaohuan Pei, Hengguang Zhou, Yuanhao Ban, Justin Cui, Jiaqi Feng, Haoyu Xie, Tao Huang, Pichao Wang, Yanchao Yang, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2609.28064)  

**Abstract**: Driving world-action models improve planning by coupling multimodal reasoning with future prediction, but their growing inference cost increasingly conflicts with the real-time latency requirements of vehicle control. Existing acceleration methods reduce tokens, layers, or sampling steps with policies selected prior to deployment, yet leave residual runtime variation largely unexploited after offline profiling and static scheduling on shared onboard compute. We observe that the largest admissible compute budget varies systematically with the residual runtime state, while recent realized latency provides a direct signal of the available compute slack. Motivated by this observation, we propose \textbf{SlackDrive}, a pre-inference compute allocator that reuses realized latency to select the compute budget of each control step before model execution. SlackDrive profiles the latency and planning utility of a small discrete budget set once, estimates online compute state from completed forwards, and selects the highest-utility budget predicted to remain within the admissible latency envelope, complementing existing profiling and resource scheduling while preserving the driving backbone and its compute actuator. On NAVSIM v2 with DriveDreamer-Policy, SlackDrive improves latency-constrained EPDMS by $21.7\%$ over the strongest baseline under a stringent latency regime, while the full-budget model and preconfigured token-pruning baselines exceed the admissible latency envelope under runtime contention. 

---
# A Resilience Recovery Method for Complex Traffic Network Security Based on Trend Forecasting 

**Authors**: Sheng Hong, Tianyu Yue, Yang You, Zhengnan Lv, Xu Tang, Jing Hu, Hongwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2609.27903)  

**Abstract**: Due to the rapid development of information technology, a huge and complex traffic network has been established across various sectors, including aviation, aerospace, vehicles, ships, electric power, and industry. However, because of the complexity and diversity of its structure, the complex traffic network is vulnerable to being attacked and faces serious security challenges. Therefore, this paper innovatively proposes a traffic network resilience recovery method based on resilience trend forecasting. In this paper, the risk value is introduced into the analysis of the network fault propagation process, and the Susceptible, Infectious, Recovered, Dead-Risk (SIRD-R) fault propagation model is established. The resilience model of traffic network, which encompasses real-time resilience and overall resilience, is constructed through the integration of network resilience bearing capacity and resilience recovery capacity. Ten, the resilience of complex traffic networks is forecasted by using long short-term memory networks, and the resilience recovery strategy of complex traffic networks based on forecasting is proposed. Finally, the effectiveness and scalability of the proposed method are demonstrated through experimental analysis conducted on a diverse range of complex traffic networks, affirming its applicability in real-world scenarios 

---
# Learning What to Activate: Combinatorial Capability Allocation for Long-Horizon Multimodal Agents 

**Authors**: Wenhao Yuan, Chenchen Lin, Jian Chen, Jinfeng Xu, Shuo Yang, Edith Cheuk-Han Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2609.27869)  

**Abstract**: Long-horizon multimodal agents rely on specialized capabilities for perception, retrieval, reasoning, verification, and execution. Existing designs typically activate a fixed capability set or invoke a predefined workflow, incurring substantial computational overhead while failing to accommodate stage-dependent capability demands. In this paper, we study the \textit{combinatorial capability allocation} problem for long-horizon multimodal agent systems, where the system selects a cost-sensitive subset of specialized capabilities at each interaction stage, which is nontrivial since capability values depend on the selected subset, while previous allocations alter the states encountered by subsequent decisions. We introduce \textsc{CoCA}, an on-policy learning framework that recovers a deployable capability-subset policy from sparse conditional comparisons. On states visited by the student policy, the stronger teacher compares the marginal net values of candidate capabilities, conditioned on the currently selected subset. Then, we adopt a conditional utility model to transform such comparisons into an autoregressive capability-subset policy, avoiding explicit enumeration. We further introduce dual-level on-policy distillation to address distribution mismatch both across environment states and within the partial subsets encountered during set construction. Finally, trajectory-level reinforcement learning refines the distilled policy toward task success, activation cost, and allocation stability. At inference time, allocation is performed solely by the lightweight student policy without teacher queries or online updates. Experiments on long-horizon multimodal environments and controlled capability-demand shifts demonstrate the superiority of our method over the state-of-the-art baseline methods. 

---
# A hierarchy of faithfulness criteria for knowledge base completion 

**Authors**: Olga Mashkova, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2609.27863)  

**Abstract**: Knowledge graph completion is evaluated by ranking observed triples above randomly corrupted ones, which treats every unobserved fact as false. When the object being completed is a description logic knowledge base rather than a plain graph, the open world assumption and deductive closure make this inadequate: relative to the knowledge base, a candidate axiom is entailed, contradictory, or undetermined, and a model that cannot separate a logically impossible axiom from a plausible novel one is not merely less accurate but semantically incorrect. We ask what it means for a knowledge base completion model to be logically faithful, and whether current embedding models are. We define a hierarchy of four increasingly strict criteria, discrimination, logical admissibility, monotonic logical faithfulness, and probabilistic logical faithfulness, and prove that they form a strict chain of implications. We ground the strongest criterion in the relative model count $P(\alpha\mid\mathcal{O}) = \#(\mathcal{O}\cup\{\alpha\})/\#(\mathcal{O})$, which recovers the trichotomy at its endpoints and ranks undetermined axioms in between. Evaluating knowledge graph and logic-geometric embedding models on $\mathcal{EL}$ ontologies, with entailed, contradictory, and undetermined test sets generated by a reasoner, we find that ranking accuracy does not imply logical faithfulness and that none of the evaluated models is faithful across the hierarchy. The code is available at this https URL. 

---
# Reachable Global Optimization in AI Systems: How Global Is Global? 

**Authors**: Wesley Shu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27855)  

**Abstract**: AI systems increasingly claim to optimize prompts, policies, architectures, plans, tool-use trajectories, reasoning traces, and test-time computation. This paper argues that such claims are underspecified unless they state the region actually reachable by the system that performed the optimization. We introduce Reachability-Induced Optimization (RIO), a model in which a generator, verifier, controller, memory, tools, and budget induce a reachable candidate region. The returned solution is therefore a best visited point, an approximate reachable optimum, or an exact global optimum only when additional certificates relate the reachable region to the full formal space. We prove reachable-optimality, false-globality, gap- decomposition, certificate, escape, pruning, and control-value results. The full benchmark record contains 66,150 executed trials over six known-optimum landscape families, seven control policies, 270 landscapes, and 35 runs per landscape-method. The online appendix includes raw trial records, aggregate tables, figures, benchmark code, validation scripts, and checksums. The results show that control can restrict, expand, or misdirect reachability, and that optimization quality, reachability quality, and control reliability must be reported separately. 

---
# Agentic Governance and Adversarial Verification for Policy-Constrained LLM Healthcare Appeal Generation 

**Authors**: Harshil Lodhiya, Alex McManus, Reese Walker  

**Link**: [PDF](https://arxiv.org/pdf/2609.27844)  

**Abstract**: Claim denial management costs U.S. healthcare approximately $260 billion annually in administrative overhead. Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) can produce fluent clinical text, but single-agent architectures fail in high-stakes healthcare: they introduce unsupported clinical details and lose the logical structure of hierarchical payer policy. We propose AGVF (Agentic Governance and Adversarial Verification Framework), a multi-agent architecture for medical-necessity appeal generation under explicit policy and evidence constraints. AGVF models appeal synthesis as a Constrained Markov Decision Process (CMDP) over five agents: policy formalization, evidence retrieval, gap analysis, adversarial critique, and gated synthesis. We prove that refinement over a fixed policy constraint graph monotonically reduces evidence-deficiency and terminates with either a complete satisfying frontier or a localized evidence gap. A deterministic citation- grounding gate prevents assertions without admissible evidence from entering shared state. We provide a reference implementation and validate it on 1,000 synthetic appeal cases parameterized from de-identified public hospital discharge data. The validation confirms zero citation-grounding violations across all AGVF cases and monotone deficiency reduction in every episode; ablating the gate raises violations to 100%, confirming it is load-bearing. The study uses no real patient records and does not measure clinical efficacy. AGVF thus contributes a theory-backed agentic architecture and verified reference implementation for policy-constrained LLM generation in healthcare. 

---
# Ask Which, Not How Good: Sizing Benchmarks Scored by an LLM 

**Authors**: Atul Anand  

**Link**: [PDF](https://arxiv.org/pdf/2609.27787)  

**Abstract**: Benchmarks scored by an LLM judge routinely adjudicate differences of a tenth of a point, but the resolution of those benchmarks has never been measured. Existing sample-complexity work covers accuracy benchmarks and leaves the judged case open. Treating the system as the object of measurement, we decompose 373,019 judgments into system, item, judge and interaction components using generalizability theory.
The central result is structural: under a single judge, generalizability asymptotes to sigma2_s/(sigma2_s+sigma2_sj) regardless of item count, because the system-by-judge term carries no n_i. Items saturate; judges do not. The item cost of a target diverges as the target nears that ceiling.
The ceiling is a property of pointwise rubric scoring, not of LLM judging. Run as a pairwise preference in both presentation orders, sigma2_sj falls two orders of magnitude below sigma2_s and the ceiling rises to 0.986 (bootstrap [0.934, 1.000] on 11 systems), so one judge suffices. Pairwise buys a different problem: a system presented first wins 8.6 percentage points more often than the same system presented second, a bias 1.23x the median improvement claimed in the 53 published win-rate comparisons we recovered. Protocol design dominates panel size.
Measured floors are 0.41-1.24 points on a 0-5 scale at native item counts, against a median reported improvement of 0.28 points; on the one benchmark recurring often enough for an exactly matched comparison, all 17 recovered MT-Bench improvements fall below MT-Bench's own floor, and 70% of the win-rate claims fall below the pairwise floor.
An audit of 628 arXiv papers, double-coded by two independent models and validated against blind human coding (kappa=0.73), finds fewer than one paper in four states whether its evaluation was run more than once, and only 46-67% report uncertainty of any kind. 

---
# Alignment of LRMs via Counter-Aligned Few-Shot Conversation Exposure 

**Authors**: Xiangyu Zhou, Saleh Zare Zade, Dongxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27763)  

**Abstract**: Large Reasoning Models (LRMs) rely on explicit chain-of-thought (CoT) reasoning and large context windows to achieve strong performance on complex tasks, but these features also introduce new attack surfaces. We show that LRMs' reasoning processes can be systematically steered by prepending counter-aligned few-shot conversations containing explicit CoT traces, leading to unsafe generations on harmful queries and unwarranted refusals on benign ones. We formalize this attack as SRCF (Steering Reasoning via Counter-Aligned Few-shot Conversations) that operates solely through a flexible conversational interface and requires no access to the model's parameters and gradients. Our key insight is that SRCF exploits an adversarial generalization issue that induces a representation drift, causing the representations of benign and harmful inputs to shift in a similar direction. This observation motivates our post-training defense, ARCF (Aligning Reasoning via Counter-Aligned Few-Shot Conversations), which exposes models to counter-aligned conversational contexts while enforcing aligned targets. ARCF is compatible with existing post-training methods and consistently improves safety and helpfulness without degrading utility. 

---
# Reporting Under Pressure: Separating Factual and Tonal Sycophancy in LLM Statistical Analysis 

**Authors**: Paras Balani, Subhrakanta Panda  

**Link**: [PDF](https://arxiv.org/pdf/2609.27756)  

**Abstract**: Large language models are increasingly asked to analyze data and report what the results mean, a task distinct from the belief- or preference-alignment settings studied in most sycophancy research. We test whether editorial framing in the prompt, ranging from a neutral request to an explicit instruction to search exhaustively for reasons to discredit or to support a finding, changes not just the tone but the substance of a model's report. Across a 4 x 4 factorial design crossing four framing conditions with four ground-truth data patterns (a genuine effect, a confound that mimics an effect but fails a robustness check, a well-powered null, and an underpowered null), we collect 480 responses and score each along two independent dimensions: whether its factual claim about the data diverged from the correct interpretation, and whether only its tone diverged while the claim stayed correct. Factual misrepresentation is concentrated in two cells: brutally critical framing applied to a genuine effect, where the model talks itself into unwarranted skepticism (97% of responses), and significance-seeking framing applied to an underpowered null, where the model overstates confidence in a null conclusion the data cannot support (100% of responses). Tone shifts far more broadly than factual content does, with critical framing producing a defensive, hedge-heavy register across every data pattern regardless of what the data show, while significance-seeking framing shifts tone only where the data leave genuine ambiguity. A confound present in the data itself blocks both kinds of shift almost entirely under every framing condition tested. These results indicate that the risk of framing-induced distortion in LLM-assisted data analysis is neither uniform across framings nor uniform across data patterns, and that a model can hold a correct conclusion in place while its tone shifts substantially around it. 

---
# Evaluation of pre-trained models for pedagogical assessment of novel AI-assisted educational questions 

**Authors**: Michael Lawrence Castanares, Princess Ventures, Allan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27749)  

**Abstract**: The surge in AI-assisted generation of educational materials has outpaced our capacity to validate their pedagogical quality. Automated evaluation using Bloom Classifier models is a promising approach to assess educational materials at scale. These models show high accuracy within-distribution dataset (IID Dataset). However, applying the same models to new out-of-distribution (OOD) datasets such as AI-assisted generated questions could show performance degradation. To identify robust classifiers under dataset shift, we evaluated traditional Machine Learning (ML), transformer, and Large Language models on the Bloom level classification task. We also explored feature-engineering strategies incorporating NLP metrics, appending the learning objectives as part of the input, and text splicing to stabilize OOD performance. Our baseline tests show that TFPOS-IDF ML models perform poorly on OOD (Macro F1-score 0.48) compared to BERT (0.55) and LLMs (0.79). Text splicing improved macro F1-score performance of ML and BERT models (0.59 and 0.62, respectively). Appending the learning objectives with the input increased model performance on specific dataset. Model retraining provided the largest improvement across models and datasets. Overall, these findings highlight the trade-off on the use of pre-trained models with novel AI-assisted educational questions and how strategic feature enhancements help address loss in performance. 

---
# Categorical Internalisation of Environmental Groupoids for Generalisable POMDP Solving 

**Authors**: Ben Opperman, Eduardo Alonso, Esther Mondragón  

**Link**: [PDF](https://arxiv.org/pdf/2609.27745)  

**Abstract**: This paper advocates category theory as a practical framework for structuring and improving rein- forcement learning in high-dimensional, partially observable environments. We model symmetries between environmental states by partitioning the state space into equivalence classes induced by sym- metry orbits, and organise each such class as a groupoid with a designated canonical representative. This allows the agent to share what it learns across many similar environmental states simultaneously, rather than treating every orientation or position as an entirely new problem. Learning is thus carried out on a symmetry-reduced state space with each orbit represented once, preserving structure while eliminating redundancy and improving sample efficiency.
We implement this framework within standard reinforcement learning pipelines and evaluate two different approaches on partially observable benchmarks, demonstrating that orbit-based partitioning yields consistent performance improvements in environments exhibiting latent symmetry. Beyond these empirical results, our approach illustrates how categorical structure provides a principled bridge between abstract reinforcement learning formulations and their computational application, thereby establishing a pathway toward more structured and scalable learning systems. 

---
# Evolutionary Stability Does Not Guarantee Learning Accessibility: A Multi-Agent Reinforcement Learning Perspective on Cooperation Emergence 

**Authors**: Yijie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27664)  

**Abstract**: Cooperation emergence is a central problem in multi-agent systems because decentralized agents must coordinate while adapting to the changing behavior of others. Evolutionary game theory identifies strategically stable outcomes, but stability under a population adjustment dynamic need not imply that finite-sample learning agents can reach the same outcome through local reward feedback.
We study this distinction in a transparent three-agent governance-motivated game involving a government, a platform firm, and users. We derive replicator dynamics for the fixed stage-game incentives, evaluate the cooperative evolutionary basin on a symmetric initial-condition grid, and compare it with learning-basin estimates for three decentralized value-based learners. The learning analysis uses independent Q-learning with $\varepsilon$-greedy action selection, scaled Boltzmann exploration, and SA--EA BQL under the same payoff environment and outcome criterion.
The evolutionary basin has volume $V_E=1.00$ on the sampled grid. The empirical learning basin is $0.88$ for $\varepsilon$-IQL and $0.00$ for both scaled Boltzmann and SA--EA BQL. Diagnostic traces show that broader action diversity and nonzero value separation can coexist with failure to sustain the cooperative joint action in this fixed configuration.
These results indicate that evolutionary stability and learning accessibility are distinct properties of a coupled game--learning system. The shared-bike setting is a motivating application; the broader contribution is a framework for comparing population-level stability with the finite-sample accessibility of cooperation under specified multi-agent learning dynamics. 

---
# SHRAV: State-Hypothesis-Reason-Action-Verify Framework for Physical Modeling and Inverse Design 

**Authors**: Ziheng Guo, Yang Bu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27621)  

**Abstract**: Physical modeling and inverse design require computation that can continue from reusable state. We introduce SHRAV, an architecture-independent computational framework organized around State, Hypothesis, Reason, Action, and Verify. Its central mechanism is a state-continuation core with declared reuse boundaries and explicit roles for learned evolution and numerical quantities. Forward configurations evolve predictive state and read out physical responses; inverse-design configurations additionally generate target-directed modifications and consume evaluator feedback. Electromagnetic world-model studies are mapped to forward configurations, with selected readout and reuse diagnostics reported here. Computational lithography demonstrates an inverse-design configuration: four fixed-weight design updates improve thresholded aerial-image intersection-over-union from 0.5313 to 0.8153 under independent scalar-pupil replay, with maximum absolute prediction-replay difference approximately 0.000824 between predictor estimates and independent replay. 

---
# BiCFlow-MER: Orchestrating Discriminative and Generative Multimodal Emotion Recognition via Conditional Transport 

**Authors**: Yanbing Wang, Shenyue Wang, Chunyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27615)  

**Abstract**: In multimodal emotion recognition (MER), human affective states are inferred by integrating complementary cues from multiple modalities. In audio-text MER, affective cues are often entangled with speaker style and lexical content, while cross-modal disagreement further complicates how the evidence should be integrated. Under conventional discriminative fusion, multimodal evidence is compressed into a terminal prediction, with modality-specific cues and conflict information insufficiently preserved. In large generative affective models, by contrast, affective reasoning is typically embedded in language decoding, leaving emotion evidence implicit and difficult to verify in a structured space. To address these limitations, BiCFlow-MER (Bidirectional Conditional Flow for Multimodal Emotion Recognition) is proposed as a conditional-flow framework in which audio-text MER is formulated as generative evidence transport within a structured emotion space. Within BiCFlow-MER, emotion-oriented evidence is disentangled from speaker-style and lexical-content factors to construct a conflict-aware affective condition. Guided by this condition, each utterance is transported to an explicit emotion-space endpoint through a bidirectional rectified flow. Candidate emotions are jointly verified through adaptive prototype-cloud scoring of the transported endpoint and backward class-to-condition consistency with the original multimodal condition, enabling conflict-aware recognition. BiCFlow-MER is shown to outperform all compared methods across IEMOCAP, MELD, and the zero-shot CASE benchmark. By orchestrating discriminative recognition and generative evidence modeling through conditional transport, BiCFlow-MER defines a new MER paradigm. 

---
# State-Grounded Conditioning: Wrapping User-Facing LLM Agents Where Direction Depends on Live State 

**Authors**: Qi Liu, Xiaoyang Yuan, Yubin Ruan, Zhuomeng Zhang, Wenjin Wang, Di Wu, Mingye Xu, Xinyi Mou, Xingxi Yin, Ke Feng, Zixun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.27606)  

**Abstract**: We introduce State-Grounded Conditioning (SGC), a design principle for user-facing LLM agents that must condition on live user state (game state, session history, live inventory), and a distinct failure class we call direction drift: task-complete responses whose chosen direction misaligns with the current state. SGC externalises state-dependent control into rule kernels over structured inputs and three primary state slices, via Perception, Grounding, and Interaction wrappers with explicit conditioning dependencies. We evaluate SGC on a 200-session anonymised benchmark ($\approx$1,000 assistant model turns) from an in-game conversational coaching agent that guides players through consecutive competitive matches, reporting mean first-token latency and five human-annotated dialogue-quality metrics that jointly cover factual grounding and coach-like guidance progression. The Perception wrapper holds mean first-token latency at 1.5s (vs. 6.1s for PE-Agent inside a production tool-use harness); enabling all three wrappers lifts turn-level grounded accuracy from 61.1%/69.8% (Prompting / PE-Agent) to 96.7% and session-level grounded accuracy from 20.0%/26.5% to 83.5%; session-level grounding-failure incidents drop by $\approx$78% relative to the strongest baseline. A cumulative ablation shows complementary incremental gains as the wrappers are added. These results inform approximate state-slice orthogonality, without establishing independent per-wrapper effects. 

---
# Not What You Meant: Can LLMs Follow a Specified Negation Semantics? 

**Authors**: Qiming Bao, Agnieszka Mensfelt, Michael J. Witbrock, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2609.27517)  

**Abstract**: Negation does not carry a uniform interpretation across domains. In legal, regulatory, and medical reasoning, the intended interpretation depends on the reading in force -- open- versus closed-world, two- versus three-valued, and credulous versus skeptical. We study which reading of negation large language models adopt by default and whether they can override that preference when a different reading is explicitly specified. To this end, we introduce NAFBench, a procedural generator of solver-certified instances spanning four semantic viewpoints: SLDNF, well-founded semantics (WFS), and credulous and skeptical reasoning under stable-model semantics. The generator emits ground normal logic programs with controlled depth, width, and cycle structure. Each program is solved under all four viewpoints using SWI-Prolog, a well-founded semantics solver, and clingo, yielding up to four divergent labels. The programs are then verbalized into natural language under multiple framings and rule orderings that leave the answer invariant. The results expose a consistent gap. Across open-source models, following a specified negation semantics remains unsolved: the strongest models score 59--74% across the four semantic viewpoints, while the weakest score 31--67%. All models are order-sensitive on more than half of logically identical rule shufflings, while the two weaker models frequently overcommit on well-founded "undefined." Two frontier models reach 100% on the main fixed-complexity evaluation set, and a third, o4-mini, is near-perfect, falling only to 81% on well-founded "undefined." Delegating reasoning to a solver, fine-tuning on certified traces, or forcing an explicit three-valued verdict each partly closes the gap. 

---
# WhatWorkedBench: Benchmarking Experimental Understanding in AI Agents 

**Authors**: Jingjie Ning, Xueqi Li, Yibo Kong, Dongting Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27490)  

**Abstract**: AI research agents need reliable knowledge of how their experiments change outcomes. We introduce WhatWorkedBench to measure experimental understanding, the accuracy of predictions about component changes after budgeted experimentation. Agents inspect code, select measurements, and submit a response surface, a table predicting scores for every configuration of component settings. Exhaustive CPU execution supplies reference effects for changing each component while holding the others fixed. These effects capture combinations of changes across 36 tasks from 30 data sources and 8 workflow types, with 1248 configuration records. Core evaluation combines 4,206 numerical-control records across all eight families and 108 agent episodes across the original six. At eight new measurements, pair-effect ridge selects an optimum on 15 of 22 sources and limits every effect error to 10% of score range on three. Fitting a Gaussian process (GP) to the same agent observations raises effect recovery, accuracy relative to true effect magnitude, from 0.632 to 0.698 in the original Flash cohort and from 0.621 to 0.720 in an additional cohort. On six completed beat-detection and graph submissions, the same-observation GP raises family-macro recovery from 0.303 to 0.455. On six workflows with six binary options at 20 new measurements, encoding code equivalences, configurations with identical behavior, raises GP recovery from 0.248 to 0.462. WhatWorkedBench supports research on experimental agents, adaptive experimental design, numerical inference, and use of program structure. 

---
# Emergi-PersonaOS: A Persona Agent Operating System for Situational Adaptation and Controllable Evolution 

**Authors**: Haoluan Fu, Keni Chen, Xinyu Jia, Jinpeng Wang, Yuyu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2609.27417)  

**Abstract**: Symbiosis between humans and digital beings offers a vision for the future of human--machine interaction. In enduring human--machine relationships, personality provides a foundation for continuity of identity, individuality in interaction, and development through experience. We investigate this capacity through persona agents as computational implementations and introduce Emergi-PersonaOS, a psychology-grounded operating system for managing persona objects throughout their lifecycle. The system organizes dispositional traits, characteristic adaptations, and narrative identity into a three-layer persona representation, distinguishing relatively enduring persona beliefs from their activation in the current persona state. During situational adaptation, it integrates the current interlocutor, relationship, event, and retrieved memories to infer a persona state and generate actions and replies; during long-term development, it records experiences and outcomes, and develops and evaluates revision candidates through change attribution, meaning-making, and behavioral testing. Belief updates are managed through explicit review, traceable evidence and version records, and the ability to reject candidates, making persona evolution controllable. Using television-character dialogue as longitudinal material, we demonstrate long-horizon system operation and examine its principal mechanisms in a concrete implementation. This work provides a computational framework for persona agents to maintain individual continuity, produce situation-specific expression, and develop through experience over sustained interaction. 

---
# MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded Molecular Design 

**Authors**: Yongjun Jeong, Hanbum Ko, Ye Rin Kim, Chanhui Lee, Rodrigo Hormazabal, Jaewan Lee, Sehui Han, Sungbin Lim, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.27349)  

**Abstract**: Real-world molecular design remains challenging for large language model (LLM)-based agents. It requires them to interpret design contexts, satisfy multiple constraints, identify infeasible specifications, and reason over multi-step tool outputs. Existing benchmarks do not capture this complexity, focusing instead on explicit and narrow constraints, only feasible problems, and single-path solutions. To address this gap, we propose MolDesignBench, a scenario-grounded benchmark that more closely reflects real-world molecular design for evaluating tool-augmented LLM agents. MolDesignBench comprises 2K generation and optimization instances that combine implicit requirements embedded in design narratives with explicit property and functional-group constraints, including infeasible cases, and require the effective use of 17 specialized chemistry tools. Experiments across diverse frontier LLMs reveal low success rates--with the best achieving only $\sim43$\%--and frequent failures in implicit-constraint reasoning, infeasibility detection, and tool reasoning. The corresponding fine-grained failure-mode analysis identifies implicit constraint interpretation and infeasibility detection as the primary bottlenecks, establishing MolDesignBench as a rigorous testbed to guide future research on chemical agents. The benchmark, tool interface, and evaluation code are publicly available. 

---
# CART: Closed-Loop Adaptive Red Teaming for Large Language Models 

**Authors**: Dongdong Zhang, Tengchao Lv, Yilin Jia, Yuzhong Zhao, Yupan Huang, Wenshan Wu, Xiangyang Zhou, Shaohan Huang, Nan Yang, Li Dong, Lei Cui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.27336)  

**Abstract**: Automated red teaming often replays a fixed set of prompts, which measures known risks but cannot learn from failures found during testing. We present CART (Closed-Loop Adaptive Red Teaming), a framework that uses each result to guide what it tests next. CART begins with broad risk coverage, follows weaknesses that emerge, keeps new probes diverse, and records the evidence and source of every finding. It separates the Challenger that creates tests, the Target being tested, which may be a text-only model or a bounded tool-using agent, and the Judge that evaluates the results, allowing these roles to be studied independently. Across three evaluation families (Frontier, JAH, and Agentic), CART discovers more failures and higher average risk than static seed replay for every Target with an available baseline. The gains extend to tool-mediated agent tests, suggesting that contextual adaptation can reveal weaknesses that direct prompt replay does not exercise. These results describe what the test policies discover, not how often failures occur in real deployments. We also find that Challenger-Judge choices affect the evidence uncovered, highlighting the need for role separation and independent review. Overall, CART turns red teaming from a one-time checklist into a continuous, adaptive, and auditable search for model and agent weaknesses. 

---
# Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents 

**Authors**: Yefan Zhou, Yang Li, Zeyu Leo Liu, Semih Yavuz, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2609.27334)  

**Abstract**: Agentic memory systems reuse past experience to improve future performance, yet most existing designs curate memory at write time: once a task is completed, its trajectory is distilled into a fixed artifact, such as a reflection, workflow, skill, or reasoning strategy, that is later retrieved by similarity. This forces the system to decide what is worth remembering before the future query is known, irreversibly discarding information and producing a query-independent summary that must serve many possible downstream tasks. Learning such a write-time curator is also difficult because the value of a storage decision may only become apparent when a relevant query arrives, potentially many tasks later, creating a long-horizon credit-assignment problem. We instead retain raw trajectories and defer curation until read time, when the current task is known. Given the retrieved traces and the new task, a memory curator synthesizes a compact, task-adaptive payload tailored to the immediate need. Because this payload is consumed on the same task, the curator can be trained directly from immediate task success, avoiding delayed utility signals and the need to artificially group related tasks. Across ALFWorld, WebShop, and $\tau^2$-bench, our Just-in-Time Memory (JitMem) consistently outperforms no-memory agents as well as heuristic and learned write-time memory methods, improving over the strongest baseline by 16.2, 16.3, and 3.9 absolute success-rate points, respectively. Notably, even an untrained curator is already competitive with or surpasses these baselines, showing that task-adaptive read-time curation itself is a major source of the gain; training the curator further compounds the improvement. 

---
# Alignment Inertia: Auditing the Durability of Training Data Influence Through Policy Override Resistance 

**Authors**: Renata Barreto, Markelle Roesti, Mohammad Tahaei  

**Link**: [PDF](https://arxiv.org/pdf/2609.27333)  

**Abstract**: Platform operators increasingly rely on system prompts and fine-tuning to govern model behavior, yet it remains unclear how reliably these interventions override behavior inherited from prior training. We propose Override Success Rate (OSR) and alignment inertia to measure when operator interventions succeed or fail to change prior behavior. We evaluate zero-shot prompting and LoRA fine-tuning across Llama and Mistral in medical misinformation and hate speech. Alignment inertia persists across both models but varies by model, domain, and policy direction. Notably, in Mistral's restrictive hate-speech condition, LoRA increased inertia by 46.5 percentage points, showing that fine-tuning can reinforce rather than override prior behavior. We also use TRAK to test whether inertia is associated with weaker adaptation signals. TRAK achieves AUC of at least 0.85 in 7 of 8 conditions and outperforms model confidence, TF-IDF similarity, and embedding similarity as a predictor of inertia. These results provide an operator-facing audit of where prior training constrains downstream model governance. 

---
# Stable Geometry with Divergent Task Evidence for Efficient Long-Horizon Agent Compression 

**Authors**: Mingxuan Wang, Fei Luo, Bo Wang, Guorun Yao, Yinglong Guo, Chao Ning, Hongyue Chen, Yanbiao Ma, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27332)  

**Abstract**: Long horizon agents accumulate growing interaction histories that increase context and inference costs. We find that geometric redundancy alone is an insufficient criterion for safe compression. Although agent histories exhibit strong low dimensional structure, similar global geometry can preserve very different amounts of task evidence. At identical retained block counts, evidence aware selection raises next action Top 3 retention from 0.31 to 0.69, while centroid similarity remains 0.98. Controlled replacement further shows that action related information can be substantially altered while global geometric measures remain nearly unchanged. Motivated by this gap between geometry and evidence, we introduce Geometry Guided Evidence Preserving Memory (GEM), a training free compressor that protects task and execution evidence before using geometric residuals to complete coverage. GEM reduces mean combined token usage from 2.69M to 2.11M per task, a 21.4% reduction, while maintaining comparable task reward. Our results show that efficient agent history compression should optimize for preserved task evidence rather than geometric coverage alone. 

---
# Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms 

**Authors**: Xinjie Shen, Wei Fan, Xudong Guo, Jianhong Tu, Yang Su, Chuqiao Kuang, Yinger Zhang, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27321)  

**Abstract**: Language-model agents increasingly face long-horizon tasks with evolving state, interdependent decisions, and delayed outcomes. Scaling their training requires diverse agentic environments, dependable outcome signals, and low extension cost. Existing generation pipelines commonly construct an environment before defining its outcome rule or annotating its trajectories, leaving dynamics and evaluation to be aligned post hoc. VHD-Play reverses this dependency by sampling and solving a mathematical model before a corpus-grounded setter renders its decision process as stateful tools. The executable dynamics and trajectory-scoring reference are inherited from the same solved model. The pipeline produces 3,300 diverse agentic environments at a cost of a few cents each. Training Qwen3.6-35B-A3B on three families raises its mean agentic score from 0.204 to 0.815 in a five-family diagnostic. Gains also appear on held-out instances from all three training families and eight unseen mechanism families, then extend beyond the generated substrate to external benchmarks for general function calling, travel planning, and 365-day e-commerce. On E-Commerce Bench, the trained checkpoint completes every run without bankruptcy and exceeds Qwen3.7-Max. We compare written-out problems with stateful versions that reveal or hide their parameters. The comparison shows that most of the learnable gap lies in stateful interaction rather than underlying problem solving. A frozen 35B setter realizes larger environments, and scale-matched training retains gains as mechanism size and horizon grow, indicating the potential for an evolving training substrate. 

---
# Learn How to Act from Your Own Interactions: On-Policy Self-Distillation for GUI Agents 

**Authors**: Yan Zhang, Daiqing Wu, Huawen Shen, Liang Li, Gang Cao, Zhi Gong, Wei Dai, Xiaode Zhang, Can Ma, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.27307)  

**Abstract**: Graphical User Interface (GUI) agents enable the fulfillment of complex user instructions through multi-turn interactions with software environments, requiring step-wise reasoning and long-horizon memory to guide actions and retain task-relevant information, respectively. Recent on-policy self-distillation (OPSD) methods have achieved strong performance on GUI grounding, a foundational subtask for GUI agents, owing to dense token-level supervision from privilege-conditioned self-teachers. However, extending existing OPSD methods to multi-turn GUI agents is hindered by self-teachers' limited privilege-following ability and insufficient privileged guidance. In this paper, we introduce GUI-SD-v2, the next version of GUI-SD, which extends OPSD from GUI grounding to multi-turn GUI interaction and addresses key limitations through a two-stage training framework. Specifically, GUI-SD-v2 first strengthens privilege following by jointly optimizing rollouts with and without privileged guidance from the same GUI states. Furthermore, it selectively distills step-specific reasoning and memory guidance through a privilege-conditioned self-teacher, supporting action decisions and the retention of task-relevant information for subsequent interactions. Extensive experiments on two representative GUI agent benchmarks, AndroidWorld and MobileWorld, show that GUI-SD-v2 compares favorably with existing OPSD baselines while consistently outperforming the evaluated state-of-the-art methods in both Pass@1 and Pass@3 success rates. Code and training data will be publicly released. 

---
# StateComp: Learning When to Compress History in Long Horizon Agents 

**Authors**: Mingxuan Wang, Hongyue Chen, Yinglong Guo, Fei Luo, Chao Ning, Bo Wang, Guorun Yao, Yanbiao Ma, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27298)  

**Abstract**: Long-horizon agents continuously accumulate interaction history during task execution, yet the importance of past interactions changes as the agent state evolves. Existing context management methods largely compress history based on fixed windows, periodic schedules, or current relevance, overlooking a more fundamental question: when has a past interaction become safe to replace? Premature compression may remove information still needed for future actions, while overly conservative retention leads to substantial context overhead. To address this, we propose State Conditioned Compression (StateComp), a framework that determines when historical interactions can be safely compressed according to the current agent state. StateComp constructs KEEP and READY supervision through a two-stage annotation procedure and trains an imbalance-aware router on hidden representations from a frozen language model. A bounded state representation further reduces the cost of evaluating long histories, while adjacent READY interactions are grouped into continuous spans and replaced with compact summaries during execution. Experiments on WorkBuddyBench show that StateComp reduces total agent and summarization tokens by 52.27% while maintaining task performance, and achieves a 12.67-fold speedup in representation extraction. 

---
# Large Knowledge Model: From Papers to a Scientific Reasoning Landscape 

**Authors**: Yuan Huang, Sihan Hu, Hongyu Gu, Chao Ma, Jiaxing Zhang, Zhiyong Zou, Caiyu Fan, Yan Xiao, Mingjun Xu, Chenyu Xie, Mingzhen Ju, Zhehao Ma, Qi Zhang, Baozong Wang, Yu Li, Zhiyuan Yao, Ruoxue Liao, Xinyu Li, Linfeng Zhang, Kun Chen, Weinan E  

**Link**: [PDF](https://arxiv.org/pdf/2609.27297)  

**Abstract**: Accumulated scientific knowledge advances inquiry when prior findings help researchers choose new questions, design investigations, and interpret results. Realizing this value at scale requires access to the reasoning that connects research problems, scientific procedures, conclusions, and evidence. We introduce the Large Knowledge Model (LKM), a scientific knowledge infrastructure that transforms the literature into a shared, computationally accessible reasoning resource. LKM represents papers as source-grounded reasoning graphs, couples structural traversal with semantic retrieval over the same objects, and aligns related questions, claims, and reasoning chains across papers. This representation forms a Scientific Reasoning Landscape with three connected views: a Question Landscape that organizes research problems and open directions, a Workflow Landscape that exposes reusable scientific procedures, and an Evidence Landscape that connects conclusions to their support, disagreement, and conditions. The unified substrate supports reasoning-aware scientific search, evidence-grounded question answering, comparative evidence analysis, and research planning. Researchers and agents can retrieve relevant work through its scientific intent, synthesize answers with inspectable supporting arguments, and develop research plans informed by established workflows and unresolved evidence. We describe a corpus-scale system and evaluate scientific retrieval and knowledge-intensive question answering. With the answering model fixed, LKM retrieval improves accuracy by 9.30%, 4.20%, and 14.69% on ChemBench, PubMedQA, and SciBench, respectively. By connecting knowledge access to scientific reasoning and action, LKM provides a common foundation for discovering relevant research, reusing scientific knowledge, and coordinating cumulative inquiry across researchers, agents, and research cycles. 

---
# Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins 

**Authors**: Tannaz Goodarzvand Chegini, Elyas Shivanian, Behzad Karimi, Faraz Dadgostari  

**Link**: [PDF](https://arxiv.org/pdf/2609.27290)  

**Abstract**: Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study evaluates a physics-informed neural network for potential-temperature forecasting, constrained by a pressure-coordinate thermodynamic advection-source equation and a diabatic-source closure fit from the preceding 12-hour period and frozen before future-time training. Using hourly ERA5 reanalysis at three pressure levels, the model is evaluated as a conditional hindcast at lead times of one, two and three hours against persistence, local-trend, and two matched neural-network baselines, one of which receives the same future meteorological forcing as the PINN, helping distinguish the physical constraint from access to future forcing. In an Oklahoma development case, mean RMSE improvement over the strongest baseline grew from 8.1\% at one hour to 23.8\% at three hours; under an observation-density sweep down to 5\% of candidate locations, this 3-hour advantage remained 14.6--16.9\%, with no evidence that lower density improves performance. Under a fixed protocol transferred to an Alabama heat event with three virtual-observation layouts, three-hour improvement ranged 19.7-24.4\% with consistent origin-level wins. A parallel Montana stress test, in which fixed pressure levels intersected complex terrain, produced a three-hour degradation of roughly 17.5\%, identifying a terrain-related applicability limit of the formulation. Together, these results indicate that the physics constraint's benefit grows with forecast horizon, persists under severe observation sparsity, and transfers across regions, but is bounded by the validity of a fixed vertical-coordinate representation over complex terrain, evidence relevant to physics-constrained components of climate-aware forecasting and digital-twin systems. 

---
# PotARCin: Multi-Dimensional Evaluation of Skill Acquisition in Abstract Reasoning Tasks 

**Authors**: Claas Beger, Ryan Yi, Melanie Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2609.27288)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) has become a prominent benchmark for evaluating general abstract reasoning and fluid intelligence in AI models. Yet standard ARC evaluation considers only a single capability: producing the correct output grid for a test input. We argue that this narrow format fails to evaluate the diversity of abilities that genuine abstract skill acquisition should enable. We introduce PotARCin, a benchmark that extends ARC by assessing understanding of a task's underlying abstract rule across five dimensions: Definition, Classification, Constrained Generation, Editing, and Inversion. PotARCin employs programmatic methods to generate new task instances and transform given inputs for a given ARC task, enabling dynamic generative sampling beyond fixed input-output pairs. Across five state-of-the-art models evaluated on the ARC-AGI-1 training set, we observe a 25-52 percentage-point performance gap between standard ARC evaluation and evaluation on PotARCin, and find that multi-dimensional evaluation reorders models that standard accuracy ranks alike. We further investigate effects of generative sampling, difficulty of corruption types, and questions of self-consistency, showing that models frequently contradict their own formalized rule even where they have stated it correctly. We also introduce P-ARC, a held-out hand-crafted test set, on which models achieve 1-8% accuracy across all five dimensions, underscoring the importance of more holistic evaluations of abstract reasoning capabilities. 

---
# Memory Control Signals Emerge Before Action in Long Horizon Agents 

**Authors**: Mingxuan Wang, Guorun Yao, Fei Luo, Yinglong Guo, Chao Ning, Bo Wang, Hongyue Chen, Yanbiao Ma, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27286)  

**Abstract**: Long horizon language model agents continuously accumulate interaction history, increasing computational cost while making relevant information harder to preserve and reuse. Existing context management methods mainly focus on how to compress or retrieve history, but largely leave open whether the model itself already represents the need for these memory operations before they occur. We study the hidden state immediately before each agent action and find that compression and recall needs are already encoded in the model's internal representations. These signals cannot be explained by simple context length or interaction progress, and they exhibit distinct formation patterns across model depth. We further show that most memory decision information is preserved in a compact recent context, while selectively restored historical evidence complements the long range dependencies that recent context misses. Based on these findings, we propose Preaction Memory with Evidence Retrieval (PaMER), which combines state guided compression with external evidence retrieval. PaMER+ further introduces step level evidence selection to recover only the historical information required by the current task. Experiments on WorkBuddyBench, across multiple context management baselines and model backbones, show that our framework substantially reduces context consumption while maintaining competitive task performance. 

---
# Hunyuan-A13B Technical Report 

**Authors**: Tencent Hunyuan Team, Ao Liu, Botong Zhou, Can Xu, Chayse Zhou, ChenChen Zhang, Chengcheng Xu, Chenhao Wang, Decheng Wu, Dengpeng Wu, Dian Jiao, Dong Du, Dong Wang, Feng Zhang, Fengzong Lian, Guanghui Xu, Guanwei Zhang, Hai Wang, Haipeng Luo, Han Hu, Huilin Xu, Jiajia Wu, Jianchen Zhu, Jianfeng Yan, Jiaqi Zhu, Jihong Zhang, Jinbao Xue, Jun Xia, Junqiang Zheng, Kai Liu, Kai Zhang, Kai Zheng, Kejiao Li, Keyao Wang, Lan Jiang, Lixin Liu, Lulu Wu, Mengyuan Huang, Peijie Yu, Peiqi Wang, Qian Wang, Qianbiao Xiang, Qibin Liu, Qingfeng Sun, Richard Guo, Ruobing Xie, Saiyong Yang, Shaohua Chen, Shihui Hu, Shuai Li, Shuaipeng Li, Shuang Chen, Suncong Zheng, Tao Yang, Tian Zhang, Tinghao Yu, Weidong Han, Weijie Liu, Weijin Zhou, Weikang Wang, Wesleye Chen, Xiao Feng, Xiaoqin Ren, Xingwu Sun, Xiong Kuang, Xuemeng Huang, Xun Cao, Yanfeng Chen, Yang Du, Zhen Yang, Yangyu Tao, Yaping Deng, Yi Shen, Yigeng Hong, Yiqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27284)  

**Abstract**: We present Hunyuan-A13B, an open-source large language model based on a Mixture-of-Experts architecture. It contains 80 billion total parameters but activates only 13 billion during inference, balancing model capability, computational efficiency, and deployment cost. The model is pretrained on a rigorously filtered 20T-token corpus with enhanced STEM data curation, improving factual reliability and reasoning ability. High-quality supervised fine-tuning and large-scale reinforcement learning further enhance its overall performance. Hunyuan-A13B also introduces a dual-mode Chain-of-Thought framework that adapts reasoning depth to task complexity: fast thinking for routine queries and slow thinking for complex, multi-step problems. Evaluations show competitive performance across mathematics, science, programming, general language understanding, and agent tasks, often approaching that of much larger models. Its high inference throughput makes it suitable for latency-sensitive applications. We release Hunyuan-A13B to support open research and practical LLM deployment. 

---
# EnSIMem: Entity-Structured Indexing for Long-Term Agent Memory 

**Authors**: Xuanyu Meng, Xing Fan, Xinyi Fan, Chenlei Guo, Yixuan Xie, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27279)  

**Abstract**: An agent that interacts with users over long periods must recall facts, preferences, events, and changes from a continuously growing interaction history. Existing memory systems often compress interactions into generic summaries or retrieve anonymous text chunks, making it difficult for an agent to identify the correct entity, property, and supporting evidence. We present EnSIMem, an entity-structured long-term memory architecture for an agent. During offline construction, the system organizes interactions into theme-coherent episodes and builds dialogue-grounded index entries of the form [entity][entity type][property:value]. Each entry preserves its source turns, temporal information, and available multimodal fields. During online interaction, the agent's request is decomposed into evidence requirements whose properties are aligned with the memory index. Entity-property lookup and adaptive retrieval then collect the evidence needed for point, temporal, compositional, and aggregation reasoning. The agent generates its response from the preserved source evidence rather than from lossy memory summaries. On long-term agent-memory benchmarks, EnSIMem achieves high answer accuracy while maintaining compact contexts and favorable online efficiency. These results show that entity-structured indexing and episode-level provenance provide a reliable foundation for long-term memory in agents. The code of our model is available at this https URL. 

---
# TimeEvo: Failure-Driven Self-Evolution of a Time Series Agent 

**Authors**: Jie Yang, Yan Zheng, Jiarui Sun, Xiran Fan, Junpeng Wang, Liang Wang, Zelin Xu, Qinghua Liu, Zhengyu Fang, Yiwei Cai, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27277)  

**Abstract**: Time series agents answer analytical questions by calling external tools, and which tools they carry is decided by people before the agent runs. However, we identify two failures in this setup. Human-Agent Tool Misalignment: a library of 21 expert-curated tools helps on some tasks and hurts on others, dropping anomaly accuracy under every backbone we test. Silent Harm: one round of generic self-revision changes 147 answers and breaks 56 of them, while the final score moves by less than a point. Both follow from the same gap: whether a tool helps is decided question by question at runtime, while tools are supplied in advance and judged by a single average. To address this, we propose TimeEvo, which clusters an agent's diagnosed failures into capability gaps, plans a measurement for each, synthesizes evidence-only tools that fill them, and admits the candidate library only through a paired admission gate. Experiments on ten time series QA tasks and three backbones show that TimeEvo, starting from an empty library, improves accuracy on every task and every backbone, and that a library grown on a cheap model still gains when it is installed into stronger ones. Code is available at this https URL. 

---
# DRSR: Learning Set-Level Deletion Risk for Efficient Long-Horizon Agents 

**Authors**: Mingxuan Wang, Bo Wang, Fei Luo, Guorun Yao, Chao Ning, Yinglong Guo, Hongyue Chen, Yanbiao Ma, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27276)  

**Abstract**: Long-horizon language-model agents accumulate reasoning traces, tool exchanges, and observations whose relevance changes with the current decision. Existing compression strategies often score historical units independently, but the safety of deleting several units is generally not determined by their singleton scores: redundant evidence, accumulated small effects, and the information that remains after deletion all matter. We introduce Direct Relational Set-Risk Pruning (DRSR), which formulates agent-history compression as risk-constrained selection over deletion sets. Offline, DRSR constructs exact counterfactual supervision by jointly deleting protocol-valid history Blocks and measuring the change in teacher-forced likelihood of the same recorded next output. A lightweight scorer then predicts set-level harm from online-visible relations between candidate history and the current pre-action state, together with deleted-retained and pairwise set structure. At deployment, DRSR evaluates a small set of structurally valid deletion candidates with the lightweight scorer and removes the largest feasible set under recency, protocol, budget, and learned-risk constraints, abstaining when no set is sufficiently safe. On WorkBuddyBench Full260, DRSR increases mean reward from 0.699 to 0.802 while reducing total model tokens by 20.820%. On the fixed Eval40 comparison, it obtains 0.794 reward at 1.211M tokens per task, using 35.850% fewer tokens than the uncompressed agent. Mechanistic analyses and ablations further show that decision-conditioned relations, retained-context information, pair interactions, and abstention each contribute to reliable pruning. 

---
# CAVEAT: Towards Robust Computer-Use Agents in Incentive-Misaligned Environments 

**Authors**: Yuxuan Li, Will Epperson, Wesley Deng, Zezhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27273)  

**Abstract**: Computer-use agents (CUAs) increasingly act on behalf of users online. What happens when the environments they operate in have incentives that do not align with the user's? In online marketplaces, for example, platforms may favor some products over others, potentially steering agents away from the user's objective. Existing CUA benchmarks cover cooperative settings or explicit attacks, but do not test whether agents preserve user objectives when the environment itself has a stake in the outcome. We introduce CAVEAT, a controlled benchmark spanning nine marketplace environments and a taxonomy of eight common steering mechanisms. Across five model families, agents purchase the user-optimal product in 78.6% of matched-control episodes but only 17.3% when steering mechanisms are enabled. Larger models and increased reasoning improve robustness, but substantial failures persist. Our trajectory analysis and targeted ablations identify three points where steering enters the decision process: (1) agents distort the user's priorities, (2) prematurely narrow the set of alternatives they consider, and (3) commit before resolving decision-relevant evidence. Guided by this diagnosis, we develop CAVEAT-Harness, which directly targets these failure modes and raises user-optimal purchasing by 55.0%. Targeted post-training further improves a smaller open model. These results establish incentive robustness as a distinct challenge for delegated agents, diagnose how it fails, and show that targeted interventions can substantially improve it. 

---
# XLOG: A CUDA-Native Engine for Neurosymbolic Integration 

**Authors**: Levi Dubrovin, Nikita Pospelov, Kirill Sabitov  

**Link**: [PDF](https://arxiv.org/pdf/2609.27203)  

**Abstract**: xlog is a CUDA-native logic programming engine integrating neural perception with deterministic Datalog, probabilistic inference, and epistemic world views through a typed frontend and provider-owned CUDA runtime. Its reasoning modes share device data planes, but their execution boundaries differ: ordinary Datalog and exact inference are host-orchestrated, while certified resident recursive and Monte Carlo sampled cores record zero tracked host-device transfers before a bounded terminal receipt. The probabilistic path supports end-to-end gradients through GPU knowledge compilation from provenance to CNF to Decision-DNNF, exact weighted model counting, and backward gradients. A final smoothed circuit is certified against its source formula before caching or evaluation. Circuit caching yields a 2.74x MNIST-addition training speedup; a worst-case-optimal join subsystem yields a 27.96x geometric-mean gain over xlog's binary-join baseline. MNIST-addition accuracy matches Scallop's (0.9561 versus 0.9468), but no per-epoch speed claim is made because baseline epoch time varies with CPU quota. In five hub-skewed triangle-counting cases, the Souffle-to-fused-xlog execution-time ratio rises from 0.88x at 150k edges, where Souffle is faster, to 5.54x at 1.2M; fused peak device allocations are 85-1,033 MB versus 3,287-44,979 MB for the materializing arm. Exact inference is correctness-equivalent to but slower than ProbLog2. On a public video benchmark, a proximity predicate trained only through symbolic credit replaces hand-set geometry at unchanged held-out accuracy; within Event-Calculus rule search it fails ten-fold cross-validation and does not transfer on a leak-free split. On a maritime corpus, weighted clauses beat crisp selection by 0.065 F1, with the result reproduced by one chronological training pass. 

---
# Enhancing Small Language Models for Power Outage Report Generation via Minimum Risk Training 

**Authors**: Hung Phan, Waqwoya Abebe, Youssef Hussein, Supriya Chinthavali, Dalton Lunga, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2609.27197)  

**Abstract**: Minimum Risk Training (MRT) enables neural machine translation models to directly optimize sequence-level evaluation metrics instead of relying only on token- level maximum-likelihood objectives Shen et al. [2016]. Although introduced a decade ago, recent work shows renewed potential for risk-based optimization in modern language models Yang et al. [2024], Jinnai et al. [2025]. We apply MRT to power outage report generation for the Outage Data Initiative Nationwide (ODIN), transforming heterogeneous reports into standardized XML compliant with CIM IEC 61968-3. Our MRT approach improves Qwen2.5-7B-Instruct overall accuracy from 16.20% to 68.95%, demonstrating the effectiveness of sequence- level optimization for domain-specific structured generation 

---
# Do We Need Complex Topology Control? Distinct-Peer Random Routing Improves Cost-Efficiency in Sparse Multi-Agent Debate 

**Authors**: Boxuan Wang, Zhuoyun Li, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2609.27150)  

**Abstract**: Multi-agent debate (MAD) has emerged as a promising paradigm for improving the reasoning accuracy of large language models (LLMs) through iterative peer interaction. Communication topology plays a central role in this process, motivating increasingly sophisticated mechanisms that learn, adapt, or dynamically reconfigure agent interactions to improve accuracy or reasoning reliability. Meanwhile, prior studies suggest that much simpler sparse communication can already achieve competitive performance at substantially lower cost. In this work, we take a closer look at sparse MAD and ask whether complex topology control is actually necessary to improve collective reasoning. We find that a simple random-without-replacement routing policy, which lets each agent debate with two distinct and newly sampled peers at every round, provides a surprisingly strong baseline and consistently improves the accuracy-cost trade-off of sparse MAD. Building on this observation, we further study deliberation stopping and show that lightweight stopping can substantially reduce inference cost while preserving competitive accuracy. Our results suggest that sophisticated topology control such as learned topology adaption should be evaluated against strong simple routing and stopping baselines before its additional complexity is justified. 

---
# Provably Complete Generalized Planning with LLMs 

**Authors**: Katharina Stein, Chaahat Jain, Jörg Hoffmann, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2609.27105)  

**Abstract**: Generalized planning aims to compute a plan that solves all instances of a planning domain. Recent work has used LLMs to automatically generate and debug such generalized plans in the form of Python programs and achieved perfect test data coverage for several domains. However, whether these generalized plans are actually complete, i.e. solve all instances of the domain, could only be determined by manual evaluation. Here, we present an approach for automatically generating generalized plans in Lean together with proofs of their completeness relative to a specification of the domain constraints provided as input. We introduce a semantic-preserving PDDL-to-Lean conversion, and use an LLM to generate both the generalized plan and the formal proof that it solves every instance satisfying the domain constraints. The correctness of the completeness proof is determined by Lean's kernel. We evaluate our approach on 13 commonly used benchmark domains, using GPT-5.6-Sol as the LLM. For 12 of the domains we obtain generalized plans together with valid completeness proofs. This is a major advancement of the state of the art in automatic generalized-plan completeness proofs. 

---
# Policy-as-Skill: Governed LLM Decision Support with Evidence, Deterministic Control, and Audit 

**Authors**: Kabeh Mohsenzadegan, Vahid Tavakkoli, Kyandoghere Kyamakya  

**Link**: [PDF](https://arxiv.org/pdf/2609.27087)  

**Abstract**: Organizations increasingly use LLMs for policy, compliance, risk, and operational decision support, requiring evidence validation, review routing, version control, and auditability. We introduce Policy-as-Skill (PaS), a modular runtime that packages these functions as executable, versioned policy capabilities. Thirteen methods are evaluated with a fixed Gemma4 backend on 600 development tasks. PaS+Audit achieves 53.8% exact accuracy, macro-F1 0.346, review F1 0.854, citation precision 1.000, policy-reference recall 0.984, and audit completeness 1.000, outperforming LLM+RAG on most governance and review metrics. Deterministic control raises aggregate accuracy to 61.2% but is strongly task dependent, supporting selective rather than universal rule-based intervention. 

---
# Propose, Don't Judge: An Anytime-Valid Referee for LLM Agents That Mine Investment Factors 

**Authors**: Bo Qu, Mingguang Chen, Licheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27051)  

**Abstract**: Language-model agents now run the whole of quantitative factor research: they propose investment factors, backtest them, select the survivors and retire them. We ask which of those jobs an agent should keep. Our answer is governed self-evolution: the agent may propose, and a frozen statistical referee that the agent cannot touch must judge. The referee scores each candidate only on market outcomes revealed after submission, by betting, so its false-discovery guarantee holds at every stopping time for any proposal policy. We cross three proposers (a script, a bandit and a language model) with this referee and with three deliberately leaky ones, in a synthetic world with planted truth, a probe-authoring environment and a ten-year walk-forward on the CSI 500. Who judges sets the number of false admissions: the frozen referee admits 5-11 times fewer sub-threshold factors than the leaky referees under a scripted proposer, and no proposer closes that gap. Who proposes sets the yield: the language model beats the script, matches the bandit, and adds the one capability a bandit lacks, writing its own diagnostic probes. The certificate's price is time: an admitted true factor waits about 500 trading days, and the certified portfolio's Sharpe ratio therefore trails an ungated one. Judging belongs to the procedure; proposing and instrument-making belong to the agent. 

---
# Math Reasoning in LLMs is Organized by Approach, Not Topic 

**Authors**: Sajad Goudarzi, Samaneh Zamanifard, Moloud Nasiri, Hamed Rahimian  

**Link**: [PDF](https://arxiv.org/pdf/2609.27041)  

**Abstract**: Mathematical reasoning benchmarks are typically organized by topic, but language models may organize their internal computation by reusable reasoning approach instead. In this paper, we investigate whether open math-capable LLMs organize internally by topical sub-skill or by reasoning approach, and we present evidence that the approach is the key. We introduce a generation-replay protocol: a model first generates a solution, after which we replay the exact prompt-plus-generation trajectory and extract activation-importance signatures over the reasoning tokens. We cluster these signatures without supervision across eight models and five mathematical reasoning sources, then evaluate the recovered structure with structural, semantic, and intervention tests. Across all 40 model-source cells, the recovered clusters outperform matched-size random baselines. Two independent frontier-LLM judges find approach-level coherence in 77-82% of real clusters versus 6-11% in within-source controls, and topic-pure clusters usually receive labels finer than the topic itself. In approach-controlled prompting, changing the requested reasoning approach shifts cluster assignment in seven of eight model conditions, whereas paraphrases largely preserve it. These results indicate that math-capable LLMs organize internal mathematical computation by reasoning approach rather than benchmark topic. The implication is that topic-stratified benchmarks and topic-balanced training corpora can still miss the axis that matters: even deliberately topic-balanced corpora may remain imbalanced over reasoning approaches. 

---
# Are Stated Reasoning Steps Causally Load-Bearing? 

**Authors**: Abhiram Bhupatiraju, Rayan Nyaupane  

**Link**: [PDF](https://arxiv.org/pdf/2609.27038)  

**Abstract**: Chain-of-thought (CoT) monitoring assumes that the reasoning a model writes reflects the computation that directly produces its answer. Previous faithfulness metrics have been predominantly behavioral, as they simply edit the reasoning text and observe the resulting answer. However, our methodology aims to measure faithfulness causally at the activation level, specifically on self-generated reasoning. Unlike previous causal audits, which measure degradation, our interventions carry a known predicted target. In this way, each patch should switch the answer to a specific counterfactual entity derivable by construction. Specifically, we use synthetic multi-hop lookup tasks (2-6 hops). We patch the residual stream at the token span where the model states each intermediate step with the corresponding activations from a counterfactual run. For Qwen3-4B, 76.9% +/- 2.8% of stated steps are causally load-bearing (CLB) at the most responsive mid-network layer (random-position null: 11.3%; patching the underlying prompt fact: 83%, so stated steps carry approximately 96% of the achievable effect). Moreover, the standard behavioral test on the same items yields 88.2%, which overstates causal faithfulness by 11.4 percentage points (item-matched; 111:14 discordant pairs, p < 1e-15) and, for the easiest items, by up to 20 percentage points. This gap also has a clear capability dimension. Qwen3-1.7B is far less causally faithful overall (54.8%), with its faithfulness collapsing as reasoning depth increases (68% at 2 hops to 30% at 6), while Qwen3-4B remains relatively flat. Although stated reasoning can be causally meaningful, standard behavioral tests tend to overestimate its causal faithfulness, particularly on easier examples where model reasoning appears most fluent. 

---
# Training Intelligent Voice Assistant Wakeup with Controllable Synthetic Conversations 

**Authors**: Marcin Sowański, Kacper Leszczyński, Kacper Krzywicki, Krzysztof Wodnicki  

**Link**: [PDF](https://arxiv.org/pdf/2609.27037)  

**Abstract**: Wake word detection is a critical component of virtual assistants, serving as the gateway to seamless user interactions. This paper introduces a novel wake-up system that extends traditional direct keyword detection with contextual trigger detection. After an initial wake word activation, the system uses reasoning to distinguish between user commands and unrelated speech, ensuring efficient and context-aware engagement. We present a data generation architecture that produces a 62.3-hour corpus of controllable multi-speaker conversations containing direct invocations, contextual follow-ups, and non-addressed speech. Experimental results demonstrate the effectiveness of the proposed approach across diverse synthetic conversational scenarios. We release the code, dataset and trained models to promote reproducibility and further advancements in intelligent assistant technologies. 

---
# Reinforcement Learning with Decomposed Subtasks 

**Authors**: Mattie Terzolo, Mikolaj Sacha, Ayan Sinha, Andrew Rabinovich  

**Link**: [PDF](https://arxiv.org/pdf/2609.27035)  

**Abstract**: Group Relative Policy Optimization (GRPO) and related policy-gradient methods for training language model agents collapse an entire multi-turn rollout into a single scalar trajectory reward before it enters the policy update. When the task composes distinct skills, especially under sparse and delayed environmental feedback, this collapsing is lossy: the optimizer must implicitly infer which competency drove the outcome and how that should change behavior. We argue the right primitive is not a better scalar but a decomposition: trajectory reward should be split along subtasks before it enters the policy update. We introduce Reinforcement Learning with Decomposed Subtasks (RLDS), whose core is Subtask-Decomposed Advantage Estimation (SDAE): a replacement for the scalar GRPO advantage that splits trajectory reward into per-subtask shares on a fixed taxonomy, computes a group-relative advantage per subtask, and distributes per-token credit by weighting each subtask's advantage by its importance, concentrating it around the step where a reflection marks that subtask's execution as consequential. We evaluate on four agentic benchmarks: FrozenLake (sparse grid navigation), HotpotQA (multi-hop QA, one retrieval tool), ScienceWorld (long-horizon embodied science), and DeepResearch (long-form research, four tools, composite rubric reward). Heterogeneity diagnostics emitted during training show where decomposition pays off - gains scale with subtask heterogeneity, largest on the high-heterogeneity tasks ScienceWorld (+11.5 points, paired-bootstrap 95% CI [+9.8, +13.3]) and FrozenLake (+9.8 points, [+7.0, +12.8]), and within noise on HotpotQA and DeepResearch, where the diagnostics predicted little to recover. ScienceWorld is also more compute-efficient under RLDS than scalar GRPO (-10.9% wall-clock per step), as long rollouts amortize the fixed reflect-and-grade overhead. 

---
# Same evidence, different judgments: Evidence noncommutative in vision/speech-text conflicts 

**Authors**: Zhuoyun Li, Boxuan Wang, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2609.26986)  

**Abstract**: For multimodal large language models, when images or speech conflict with accompanying text, measured text reliance can entangle modality preference with evidence position. Earlier studies of text bias often used a fixed evidence order or moved task instructions with the evidence, leaving the contribution of order unclear. In this paper, we use a paired comparison that keeps the instructions and evidence content fixed and swaps only the positions of the two sources to quantify this potential influence. Across vision and speech models, placing an image or recording after conflicting text consistently shifts answers toward its content. We also revisit previous studies and analyze why their experimental settings can lead to misleading conclusions. These findings reveal cross-modal evidence noncommutativity: the same evidence can lead to different judgments when its order changes, and placing perceptual evidence later can increase the model's reliance on its content. 

---
# Escaping Python Dependency Hell: A Hybrid Replay-and-Repair Pipeline for Python Dependency Resolution 

**Authors**: Veronica Poweska, Ariana Oyanguren, Jessica Pourleyli, Sourena Khanzadeh, Manar Alalfi  

**Link**: [PDF](https://arxiv.org/pdf/2609.26952)  

**Abstract**: Dependency conflicts in Python ecosystems arise from incompatible version constraints, missing packages, and undocumented compatibility relationships, causing many real-world code snippets to fail at execution. This paper presents PLLM+, a hybrid dependency-repair pipeline evaluated on the HG2.9K benchmark of 2,891 dependency-failing snippets. PLLM+ prioritizes inexpensive deterministic steps before invoking LLM-based repair: static AST-based interpreter inference, replay of historically successful dependency configurations from the competition-provided solutions database, and live PyPI validation of candidate package versions. When these steps do not resolve a case, the system falls back to a structured LLM-based repair loop with typed error classification and Proposer/Critic agents. On HG2.9K, PLLM+ solves 1,500 out of 2,891 snippets, compared with 1,169 solved by the PLLM baseline. It also reduces average runtime from 368.7 to 71.8 seconds per snippet. Most successful fixes come from replaying known configurations: 1,495 of the 1,500 successful fixes are produced by the solutions database, while the LLM fallback accounts for 5 additional fixes. These results suggest that, in this benchmark setting, deterministic reuse of previously validated dependency configurations is a simple and effective strategy, with LLM-based repair serving as a secondary fallback for cases not covered by prior solutions. 

---
# Which Objectives Need a Dial? Predicting Objective Conflict and Covering Trade-offs in Steerable Pluralistic Alignment 

**Authors**: David Tsoi, Esra Dönmez  

**Link**: [PDF](https://arxiv.org/pdf/2609.26929)  

**Abstract**: People hold diverse, sometimes conflicting values, so no single aligned model can satisfy everyone. Pluralistic alignment therefore calls for steerable models that can balance competing objectives differently. Multi-Objective Direct Preference Optimization (MODPO) does this by using an objective weight to span a continuum of trade-offs. We study two questions: when can one model improve two objectives simultaneously, and how can many trade-offs be covered without training a separate model for each? Across seven objective pairs from HelpSteer and UltraFeedback, two pre-training measurements predict whether objectives align or conflict for human-annotated data, but not for AI-annotated data, where response length and repetition confound reward-model scores. For broader trade-off coverage, selecting the nearest trained model and merging model parameters both help, but neither consistently matches direct training. These findings yield practical guidance for building steerable models that serve diverse preferences. 

---
# Building Socio-Affective Artificial Intelligence for Interactive Multi-Agent Simulations 

**Authors**: David Berga  

**Link**: [PDF](https://arxiv.org/pdf/2609.26927)  

**Abstract**: The objective of this article is to provide design principles and a software architecture for enabling interaction between humans and multiple agents in simulated dynamic worlds. This connects the current era of general artificial intelligence (AI/AGI) with the proliferation of transformer-based conversational agents and the increased computational capabilities. Given an overview of current and previous multi-agent theories of mind (socially and affectively-aware agents), the existence of an integrative design of agent interactions with themselves and with humans must be crucial for understanding how to create sustainable and governance in future human-agent reasoning systems. In this work is presented a software "AGIMUD" that integrates: A. socially-aware reasoning and emotion in agent behavior and interaction, B. a design of human multimodal scheme for human users, artificial agents and simulated worlds, and C. distributing the AI processing through the network to enable multiple autonomous agents. These integrations allow the dynamic world recreation as multi-user dungeons (MUDs) where both agents and humans can interact simultaneously in real time. Find the code online in this https URL. 

---
# TwinCheck: Evidence-Grounded Negative-Twin Verification for Stateful Tool Agents 

**Authors**: Jiaxuan Dai, Tianyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.26911)  

**Abstract**: A single locally plausible tool call can derail an otherwise successful agent trajectory. Suspicion alone does not justify intervention, because the replacement itself can introduce the very failure verification is meant to prevent. We introduce TwinCheck, an inference-time verification policy that considers replacement only when the trace satisfies an evidence condition tied to a trace-local failure hypothesis. It constructs a trace-grounded counterfactual alternative, a negative twin, and replaces the agent's proposal only if the twin passes structural checks and the pairwise verifier prefers it in both candidate orders. For paired evaluation, exact replay holds the agent's parsed responses and actions fixed until the first accepted replacement, separating intervention effects from resampling. In the primary analysis of 159 multi-turn BFCL V4 tasks with complete exact-replay pairs, the complete policy raises task success for GPT-5.6 Sol from 45.3% to 58.5% (95% task-bootstrap CI [8.2, 18.8]), with no observed success-to-failure regressions. Together, these findings recast execution-boundary repair as a constrained comparison, making the counterfactual action itself the object of verification. 

---
# Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity 

**Authors**: Zhening Li, Joshua Liu, Mateja Vukelic, Nicole Shen, Supriya Lall, Amitayush Thakur, Alex Zhang, Omar Khattab, Jonathan Light, Armando Solar-Lezama  

**Link**: [PDF](https://arxiv.org/pdf/2609.26891)  

**Abstract**: Modern language-model agents are built around the \textit{agent loop}, where the LLM is placed in an environment exposing a set of tools, and the LLM has full control over the workflow by alternating between tool calls and observing their output. However, certain workflows currently require additional engineering beyond the agent loop itself, such as memory systems and self-improving systems. We built an LLM agent framework, JAZ, to explore the extent to which a minimal harness that is little more than the agent loop itself can accomplish tasks these specialized systems are built for. JAZ exposes a single LLM-based primitive invoke and provides a set of built-in hooks that allow the programmer to apply constraints and monitoring. Generalizing existing code-mode agent loops, \texttt{invoke} is the simplest loop that satisfies two defining properties: (1) the LLM can write arbitrary executable code that can include recursive \texttt{invoke}; (2) everything visible to the LLM --- all inputs to \texttt{invoke} as well as its interaction history with the code environment --- are variables in the code environment. We motivate our design from first principles, viewing \texttt{invoke} as a language primitive representing a function whose implementation is provided at runtime by an LLM every time it is called. To validate the design of our core \texttt{invoke} primitive, we evaluate \texttt{invoke} --- with only prompting, no manually designed tools, harness, or external systems (e.g., memory or the file system) --- on workflows traditionally implemented through specialized external harnesses. On long-horizon workflows requiring recall beyond the context window, JAZ invoke outperforms Letta (MemGPT) by 8\% at half its cost on the recall-heavy portion of StuLife. On continual self-improvement, JAZ invoke outperforms ACE by 4\% at a lower cost on AppWorld. 

---
# Silent Failures in Agent-Tool Interaction: An Audit of ToolUniverse 

**Authors**: Shreya Gopalan, Devansh Singh, Sundaraparipurnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2609.26836)  

**Abstract**: Agentic AI systems are increasingly adopting automated pipelines that integrate multiple tools. While prior research and benchmarks have studied about task success and task completion of these agentic systems, the research about agent to tool interaction, specifically in biology agentic workflow is limited. This study investigates specific failures in agent to tool interaction where a tool invocation appears successful, some or all of the information or functionality from the tool via API/ wrapper is incomplete or missing and there are no communications / notifications to the user or the agent about such missing information. We call this a silent failures as the user or the agents are not aware that such failure has occurred. For the purposes of this study we developed an audit mechanism to identify such silent failures in Agent to tool interaction, by examining 15 scientific tools (and their associated API documentation and tool documentations) integrated within ToolUniverse environment (ToolUniverse serves as our experimental environment rather than the object of the study itself). We structure our study around 7 failure locus characterising where the failure occurs in the chain. We observed 91 failures (manually validated post LLM based candidate discovery and automated testing), most frequent of them being missing data or fields and inconsistencies in search, filtering or ranking criteria. Most of the 91 failures occurred in API layer (51) or wrapper layer (25), with a potential of silent failure amplification downstream. The results show that silent failures originate upstream of the event and propagate downstream into apparently valid scientific outputs. We propose a concept of contextual reliability to handle such failures and suggest mechanisms for testing, disclosing, monitoring, and measuring such failures across the agent-tool interaction pipeline. 

---
# Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction 

**Authors**: Zilin Fang, Zishuo Wang, Gim Hee Lee, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28467)  

**Abstract**: Social navigation typically assumes a specified goal and focuses on reaching it while respecting social conventions, whereas robot group joining requires predicting where to join based on the group's real-time activity and formation. This is a highly semantic task, yet an important capability for applications such as robotic guide dogs and autonomous mobility scooters. We formulate language-grounded robot group joining: given an observation and a natural-language description of a target group, the robot identifies the relevant group members and predicts socially compliant joining poses. For grounding, we generate structured candidate subsets through recursive spectral partitioning and rank them with a language-conditioned image--geometry model. Given the grounded group, a goal predictor leverages human-formation priors to produce a multimodal energy--orientation map over feasible robot poses. Experiments on conversations, queues, and audiences across varying group sizes, crowd densities, and visual ambiguities show that our method achieves competitive grounding accuracy with sub-second inference and outperforms all baselines in joining-pose prediction. Real-robot experiments further demonstrate group joining in both static and dynamically changing interactions. 

---
# Can LLMs Reason About Runtime Behavior? A Repository-Level Dynamic Benchmark 

**Authors**: Hamed Taherkhani, Mohammad Abdollahi, Melika Sepidband, Hridya Dhulipala, Tien N. Nguyen, Hadi Hemmati  

**Link**: [PDF](https://arxiv.org/pdf/2609.28449)  

**Abstract**: Large language models (LLMs) are increasingly used in coding tasks, but their ability to reason about code execution remains unclear. Existing repository-level QA benchmarks mainly evaluate static code understanding and often rely on LLM-based evaluation, while execution-reasoning benchmarks are mostly limited to snippets or functions. We introduce SWE-Flux, a repository-level benchmark for dynamic execution reasoning containing 480 execution-grounded instances across 12 real Python repositories, with gold answers automatically harvested from instrumented test executions rather than written manually or judged by LLMs. The benchmark covers singletest and multi-test questions over control flow, loops, program state, dataflow, exceptions, and program invariants. Evaluating five LLMs shows that this task remains challenging. The best model achieves only 37% accuracy. Models perform better on localized behavior such as invariants, intra-procedural control flow, exceptions, and simple loops, but struggle with dataflow, inter-procedural execution, precise state reasoning, and suite-level aggregation. Finally, we show that the oracle-harvesting pipeline can generate fresh benchmark variants using input perturbation. It successfully harvests valid variants for almost 90% of the selected instances, and the resulting variants are substantially more challenging for the evaluated models. 

---
# Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning 

**Authors**: Zhixu Silvia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2609.28442)  

**Abstract**: Reordering a set of mathematical rules without changing its meaning should preserve the correct answer, but must a model's internal representations stay invariant too? We investigate this question using synthetic multi-step function-composition problems, each presented under multiple rule orderings with the same correct answer. We measure accuracy and permutation signal-to-noise ratio (SNR), which quantifies how distinctly ordering patterns are represented relative to variation across problem instances. Across 16 language models ranging from 1B to 8B parameters, we find a pattern: models that solve reordered problems more accurately represent different rule orderings more distinctly. Layer-averaged permutation SNR is positively rank-correlated with accuracy in every synthetic setting we evaluate, with Spearman correlations reaching 0.86. These findings highlight a distinction between answer invariance and representation invariance: successful mathematical rule composition can accompany distinct internal representations between equivalent rule orderings. This motivates distinguishing answer invariance from representation invariance, and offers a representational perspective on mathematical reasoning beyond answer accuracy alone. 

---
# Agent-Editing World Model: Rethinking World Modeling for LLM Agents 

**Authors**: Shuang Sun, Guoxin Chen, Fanzhe Meng, Jia Deng, Huatong Song, Jinhao Jiang, Wayne Xin Zhao, Hongteng Xu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2609.28416)  

**Abstract**: Recent advances in large language models (LLMs) have enabled agents to tackle long-horizon tasks across diverse environments. To further improve agent performance, existing language world models typically predict environment observations, yet reconstructing high-entropy, execution-dependent tool responses offers limited value when real feedback is available. Meanwhile, agents suffer from \emph{task-state contamination}, where unsupported assumptions and outdated plans persist in history and distort subsequent decisions. We propose the \textbf{Agent-Editing World Model (AEWM)}, which models how reasoning and actions shape future task progress rather than simulating tool responses. AEWM combines \textbf{Action Judge} to distinguish \textsc{Critical}, \textsc{Exploratory}, and \textsc{Noisy} decisions with \textbf{State Revision} to edit noisy reasoning--action continuations from the same observed history. \textbf{EditAct} integrates these capabilities with real execution, directly changing the state underlying subsequent decisions rather than merely providing critiques. We train AEWM across Search, Terminal, and Software Engineering through mid-training and supervised fine-tuning. AEWM achieves 70.5\% macro-F1 on our Action Judge benchmark, exceeding the strongest frontier baseline by 10.6 points. Across six benchmarks and three agent backbones, EditAct improves average scores by 3.2--6.7 points over the strongest baseline. Furthermore, rejection sampling fine-tuning on verified EditAct trajectories, termed \textbf{AEWM-RFT}, improves over Self-RFT by 2.2--2.6 points across three domains without online AEWM guidance. 

---
# Frozen Flows Forget: Diagnosing and Restoring Lost Motion in a Latent-flow World Model 

**Authors**: Xiwen Chen, Rigaudiere Z. Li, Zhiruo Zhou, Xiaojun Zhu, Houde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28414)  

**Abstract**: Latent world models that integrate a flow in a frozen self supervised latent space train stably and cheaply, yet silently lose the property manipulation depends on most: motion. The pretrained flow never moves the manipulated object; retraining it with latent-only losses only trades stillness for teleport-like motion. We trace the failure to the training signal, not the representation: anchor-sparse, latent-only supervision never says where along the horizon change belongs. Decode-augmented rollout training (DART) repairs this while keeping the representation frozen, retraining only the flow with decode-path supervision. DART outperforms its latent only parent on the full protocol, restores the temporal structure of motion, and re-couples predicted motion to the scene; at larger scale it further improves prediction quality, closing nearly half the remaining gap to an oracle-informed interpolation reference. Finally, we report an unexpected finding about evaluation: pixel error alone rewards frozen predictions. 

---
# Learning Holographic Reduced Representations with Clifford Variational Autoencoders 

**Authors**: Mohamed Malek Abid, P. Michael Furlong  

**Link**: [PDF](https://arxiv.org/pdf/2609.28409)  

**Abstract**: Vector Symbolic Algebras project data structures into a hyperdimensional vector space through the application of their vector algebras to randomly generated atomic vector symbols and fractional power encodings of real-valued data. Embedding unstructured data remains an open question. We present \textit{Clifford-VAE}, a variational autoencoder that learns to project data onto a Clifford torus in arbitrary dimensions. Experiments using the MNIST, FashionMNIST, and CIFAR-10 datasets demonstrate that Clifford-VAE produces representations that are competitive with those produced by Gaussian and Hyperspherical VAEs for semi-supervised classification tasks while outperforming Gaussian and Hyperspherical counterparts in the VSA benchmark tests of self-binding and unbinding, role-filler recovery, and bundle capacity. Clifford-VAE provides a principled technique for grounding perceptual data into a symbolic reasoning framework, providing a new approach to a long-standing problem in the VSA literature. 

---
# When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibrated Credit Assignment 

**Authors**: Jie Zhang, Jingxiao Yang, Zhehao Huang, Yuhang Liu, Xiaolin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28385)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) supervises mathematical reasoning through final-answer correctness, but provides little guidance on individual tokens. On-policy distillation (OPD) supplies dense feedback on student-generated responses, yet teacher preference need not reflect correctness. Recent hybrids combine OPD and verifier-derived advantages or reweight task credit using teacher ratios. However, teacher guidance enters after verifier-based group normalization, and token reweighting need not preserve the total task credit assigned to each response. We introduce Unified Entropy-Calibrated Credit Redistribution for GRPO (UECR-GRPO), which integrates verifier and teacher signals within a single GRPO-style update at both the response and token levels. \emph{Path-Utility Unification} (PUU) combines verifier reward and a teacher-to-anchor path log-ratio in a single KL-regularized objective. Its on-policy implementation uses a length-normalized teacher score and combines both rewards before group normalization and PPO clipping, allowing teacher evidence to influence the response ranking. \emph{Entropy-Calibrated Redistribution} (ECR) then uses the signed teacher--old-policy token gap to redistribute the verifier-derived component. Full-vocabulary teacher entropy attenuates uncertain guidance, while a response-wise zero-sum projection preserves the total task credit and its token-wise sign before clipping. Across five mathematical reasoning benchmarks, UECR-GRPO achieves average \(\mathrm{Avg@12}\) accuracies of 17.21\% and 65.09\% with Qwen3-1.7B and Qwen3-4B students, respectively, exceeding the strongest baseline at each scale by 0.89 and 0.56 percentage points. 

---
# Shopping by algorithm: How agentic AI deploys human heuristics as a surrogate consumer 

**Authors**: Davood Wadi, Yu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.28372)  

**Abstract**: Consumers increasingly delegate purchasing decisions to Large Language Models (LLMs) acting as surrogate consumers. Using "Tool-Lab," an adaptation of information-board process tracing that places product attributes behind costly tool calls, we examine how marketing pricing cues (i.e., just-below pricing and promotional framing) influence AI shopping agents. Across eight commercially deployed LLMs from three providers, we trace pre-choice information acquisition. Under zero cost, pricing cues rarely mislead. Imposing acquisition costs under a vague goal prompt leads LLMs to omit diagnostic attributes required to compute unit price and choose suboptimal choices resembling human heuristics. Relative to a specific goal prompt that mainly preserves diagnostic search and choice optimality, a vague goal prompt under constraints creates a search-mediated vulnerability. This research demonstrates that marketing heuristics in delegated AI shopping are governed by storefront information architecture, not necessarily immutable LLM flaws. 

---
# AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios 

**Authors**: Zhipeng Bao, Wenjie Zhao, Tianle Zhu, Haohua Que, Chence Yang, Geng Yuan, Qianwen Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.28366)  

**Abstract**: Vision-language models (VLMs) offer a promising approach to long-tail autonomous driving, but existing driving datasets provide limited supervision for connecting decision-critical visual evidence with reasoning and planning. We introduce AnchorReasoning, a visually grounded reasoning dataset built on WOD-E2E, containing 416,119 annotated frames and 395,379 decision-critical elements across four major categories and 19 fine-grained types. Each frame is organized as a visually grounded chain-of-thought (VG-CoT) that links decision-critical element identification and localization, element attributes and implications, driving-action rationale, and action and trajectory planning. We further develop a curriculum supervised fine-tuning strategy that progressively learns these hierarchical capabilities, together with an object-size-aware grounding metric for evaluating localization quality. Experiments across eight general-purpose, embodied-AI, and AV-specific backbones show that VG-CoT supervision improves grounded reasoning and trajectory prediction. Across models, 5-s ADE and FDE decrease by 7.84 and 11.86, while RFS Frame and Cluster improve by 1.66 and 1.70. These gains are achieved with 18.5 fewer reasoning tokens and 0.32 s/frame lower inference latency on average, demonstrating the value of visually grounded, decision-focused supervision for VLM reasoning and planning in long-tail autonomous driving. 

---
# MicroQonv: Reshaping Convolution Tensors for Efficient Microscaling in Training and Inference 

**Authors**: Romain Facq, Sami Ben Ali, Olivier Sentieys  

**Link**: [PDF](https://arxiv.org/pdf/2609.28358)  

**Abstract**: Microscaling quantization techniques are increasingly used to represent neural network parameters with 8 bits or fewer while preserving near-full precision accuracy. However, applying these methods efficiently in convolutional layers is not straightforward. A naive approach transfers full-precision weights and activations to processing units and quantizes each tensor twice, resulting in much more memory movement than expected. Additional overhead comes from the activation tensors, whose sizes grow substantially because of the im2col transformation applied before quantization. We propose MicroQonv, a way to combine microscaling with convolutional layers' forward and backward operations by quantizing each tensor only once and quantizing the activation tensor before applying a modified version of im2col: channel-batch-first im2col. MicroQonv reduces the quantization cost by a factor of $\times2$ for weights and gradients, and by up to $\times9$ for activations, at a negligible accuracy cost. It reduces memory movement and storage by up to $\times7.53$ compared to their full-precision counterparts. This way, MicroQonv reduces microscaling-quantized activation memory movement by $\times3.5$ for state-of-the-art object detection models YOLOV8nano and $\times2.2$ for YOLOV26nano. It also enables 4-bit microscaling in a quantized latent replay strategy for continual learning at the edge, improving accuracy by +5.7% to +11%. 

---
# MemBodied: Recurrent Associative Memory for Vision-Language-Action Models 

**Authors**: Tej Deep Pala, Navonil Majumder, Bryce Goh, Raphael Yee, Jianfei Yang, Liming Chen, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2609.28256)  

**Abstract**: Vision-Language-Action models provide a strong foundation for general-purpose robot control, yet a vast majority of policies do not preserve and leverage episode-level information beyond the current observation. This limitation is consequential in history-dependent manipulation tasks that depend on information available only in past observations. Retaining past observations in context can aid in recovering this information, but at the significant cost of ever-growing, bloated context and inference latency. We thus introduce MemBodied, a fixed-size episodic memory with two complementary components: an associative state that records interactions across policy calls and an episode anchor that preserves a compact representation of the initial scene as a reference. At each policy call, the model conditions action generation on the current input and the memory components, rather than directly using past observations. Across five evaluated RMBench tasks requiring memory, MemBodied achieves $7.81\times$ the mean success rate of a stateless policy and $2.98\times$ of vanilla recurrent memory, while outperforming the strongest memory-augmented baseline by $1.3\times$ with $10\times$ fewer added parameters. On the fully observable LIBERO-Long suite, it reached 90.6%, a 5.4% improvement over the stateless $\pi_0$ policy. These findings support MemBodied as a practical alternative to expanding the policy context for history-dependent manipulation. 

---
# Controlling Collectives of AI Agents in Reasoning Space with Spatial Transformers 

**Authors**: Frederic Vatnsdal, Roshan Gopal, Romina Garcia Camargo, Vijay Kumar, Alejandro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2609.28247)  

**Abstract**: Large Language Models (LLMs) introduce an exciting new paradigm for planning and navigation in robotics, but fail on even simple multi-robot tasks as team sizes grow. We propose COMPASS, a scalable, decentralized multi-robot architecture for controlling large collectives of agentic robots with reasoning space feedback control. Feedback is generated locally on each robot by a spatial transformer which aggregates multi-hop messages across the fleet into a learned feedback token. Our experiments find that collectives of language models demonstrate performance gains from structured diversity of the input command, which can cancel biases; an advantage that is held across scale. Compared against a centralized frontier LLM policy and a language-only communication ablation, we find that the coupled design of COMPASS decisively produces cohesive flocking formations that accurately fly the commanded intent. We show that reasoning feedback works best when composed with a compact learned token. Our ablations show that hand engineered feedback with raw state appearing in the language channel obliterates cohesion. COMPASS generalizes zero-shot to unseen instructions of ambiguous meaning while commanding flocks up to 16 times its training scale, flying up to 1024 robots under natural language commands. 

---
# Beyond Poetry: Can Large Language Models Generate Classical Arabic Maqamat? 

**Authors**: AbdulRahman A. Morsy, Aya Zirikly  

**Link**: [PDF](https://arxiv.org/pdf/2609.28245)  

**Abstract**: Large language models (LLMs) have shown strong performance in creative text generation, yet their ability to produce culturally grounded and stylistically constrained literary forms remains underexplored. Prior work has focused largely on modern language varieties and poetry, while classical prose traditions such as maqama remain largely unstudied. The maqama is a classical literary genre characterized by rhymed prose (saj), dense rhetorical ornamentation, and episodic narrative structure, making it a challenging testbed for evaluating whether LLMs can move beyond surface fluency toward deeper literary competence. In this paper, we present the first controlled evaluation study of maqama generation with LLMs, comparing five models under zero-shot, few-shot, and rule-based prompting, and evaluating outputs through both human annotation and an LLM-as-a-judge framework across dimensions such as rhetorical richness, saj density, structural coherence, and stylistic authenticity. Our results show that prompting strategy plays a strong role in stylistic quality: few-shot prompting most consistently improves saj density, while its effects on rhetoric and coherence vary by model, with the strongest models (GPT-4o and GPT-5.4-mini) benefiting most from rule-based prompting on these dimensions, though zero-shot prompting yields the highest aggregate scores across all five models. We further observe systematic differences between models in stylistic alignment with Arabic maqama conventions, and corroborate our findings with a second independent LLM judge, paired statistical significance testing, and non-LLM proxy measures of saj. 

---
# Do Center Biases Propagate? Robustness of Pathology Foundation Models in Whole-Slide Image Classification 

**Authors**: Ilán Carretero, Pablo Meseguer, Rocío del Amor, Valery Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2609.28231)  

**Abstract**: Pathology foundation models (PFMs) have transformed computational pathology through powerful representation learning from histopathological images. PFMs provide rich, discriminative representations for whole slide image (WSI) analysis, enabling tasks such as slide-level classification under multiple instance learning (MIL). However, these representations may also encode non-biological signals associated with acquisition centers, potentially introducing spurious shortcuts into downstream predictions. In this work, we evaluate center-associated robustness in WSI classification using a controlled training setting with increasing class-center correlations quantified by Cramér's V. We benchmark six PFMs across four datasets and two MIL aggregators, while evaluating ComBat as a robustification strategy. We further introduce the Area Under the Cramér's V Curve (AUCC) to jointly capture absolute classification performance and its degradation as spurious correlation increases. Results show that center-related information encoded by PFMs propagates to WSI-level predictions, with robustness depending on both the PFM representation and MIL aggregation strategy. Additionally, ComBat harmonization does not provide consistent robustness gains across datasets. 

---
# From Agent Output to Authorized Transition 

**Authors**: Christopher Koch  

**Link**: [PDF](https://arxiv.org/pdf/2609.28216)  

**Abstract**: Agentic engineering systems can edit repositories, run tools and tests, build firmware, synthesize schematics, and prepare deployable or manufacturable artifacts. The assurance problem is therefore shifting from whether an agent can produce an output to whether an engineering lifecycle is justified in acting on claims about that output. Current products and standards provide sandboxes, approvals, hooks, traces, policy enforcement, attestations, bills of materials, and assurance representations, but these capabilities remain fragmented. This paper presents the Agile-V Assurance Spine, a cross-domain transition contract for software, firmware, and PCB engineering. Evidence is admitted only when it establishes required properties through an authoritative source profile, is bound to the exact artifact and frozen policy baseline, remains current with respect to declared dependencies, and satisfies risk-appropriate independence and authority. Gate decisions are recorded as receipts; approvals and exceptions are exact-scope and time-bounded; and authorization is rechecked at the effect boundary before merge, deployment, flashing, release, or fabrication. A bounded review of contemporary research, commercial platforms, open-source infrastructure, and standards positions the model relative to evidence-gated lifecycle control, continuous assurance, runtime admission, provenance, and AI/ML inventories. The paper contributes a precise vocabulary, compositional architecture, domain profiles, mapping to open-source implementations, and an adversarial evaluation agenda. It does not claim regulatory conformity or demonstrated production superiority. 

---
# "We'll Fix It Later": Education, AI, and the Deferral of Privacy in EdTech 

**Authors**: Meghna Manoj Nair, Rachel Greenstadt  

**Link**: [PDF](https://arxiv.org/pdf/2609.28137)  

**Abstract**: Educational technology (EdTech) platforms collect highly sensitive student data, including behavioral logs, disability records, and academic histories. However, privacy considerations are often postponed rather than treated as a foundational design requirement. We present a mixed-methods study combining 12 semi-structured interviews with EdTech professionals and a privacy policy audit of 48 platforms coded across five dimensions, with strong inter-rater reliability (mean Cohen's Kappa = 0.781). Our interviews reveal a recurring organizational pattern in which privacy is recognized as important but deferred across the product lifecycle as organizations prioritize product functionality, growth, funding, and immediate educational outcomes. Responsibility is often delegated to cloud providers, policy documents, or downstream institutions, while limited privacy-related feedback gives organizations little pressure to change these practices. The policy analysis reflects these patterns: platforms describe what data they collect relatively well but provide substantially less information about how that data is subsequently governed. Thirty-three percent make no meaningful Artificial Intelligence (AI) disclosure despite visible AI features, and 73% provide only generic accountability and breach-response language. K-12 platforms perform better on children's consent where regulation creates explicit requirements, but this advantage does not extend to AI governance or accountability. These findings suggest that meaningful improvement requires enforceable institutional and regulatory mechanisms rather than voluntary privacy commitments alone. 

---
# Scaling Attention Head Analysis via Gradient-Based Attribution in Context-Aware Machine Translation 

**Authors**: Paweł Mąka, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2609.28117)  

**Abstract**: In this paper, we introduce a gradient-based head attribution strategy where the Token-level Max-Margin loss is backpropagated to the attention maps. This framework enables a large-scale causal analysis of attention heads, making it suitable for LLMs. We evaluate our method on the task of disambiguation in Context-aware Machine Translation, where we analyze 50 phenomena across 4 models and 4 language directions. We empirically show the alignment of our method with the effects of increasing the attention scores of token-to-token relations on three models and two language directions, ensuring the robustness of our method. Our analysis reveals the presence of the "general-purpose" attention heads that improve the model's performance when attending to different relations. We find that the average attention a head assigns to a relation does not necessarily relate to the model's performance, which suggests that the models developed redundancies during training in terms of the head functions. 

---
# Field-of-View Extension in Dental Cone-Beam CT via Implicit Neural Representations and Diffusion Model-Based Refinement 

**Authors**: Susanne Schaub, Florentin Bieder, Matheus L. Oliveira, Yulan Wang, Buyanbileg Sodnom-ish, Dorothea Dagassan-Berndt, Michael M. Bornstein, Philippe C. Cattin  

**Link**: [PDF](https://arxiv.org/pdf/2609.28110)  

**Abstract**: Dental cone-beam computed tomography (CBCT) systems often employ detector configurations that provide a truncated field of view (FOV) that only captures a small part of the patient's anatomy. In this work, we aim to reconstruct an extended FOV using projections of truncated FOV scans. To this end, we propose a three-stage framework that consists of (1) an implicit neural representation (INR) for estimating missing parts of the truncated projection data, (2) an iterative reconstruction for generating a secondary volumetric image with improved anatomical consistency and (3) a fast diffusion model for image enhancement. The proposed approach combines the strengths of continuous representations, physics-based reconstruction and generative refinement within a unified pipeline for truncated CBCT imaging. Experimental results demonstrate that the method effectively reduces truncation artifacts, improves the reconstruction of structures extending beyond the original FOV and produces images with enhanced quality. Our code is publicly available at this https URL. 

---
# Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching 

**Authors**: Shreya Deshmukh, Imen Mahdi, Nick Heppert, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2609.28107)  

**Abstract**: Advances in generative modeling have recently been extensively employed in robotics for policy learning. In particular, Conditional Flow Matching (CFM) trained with expert demonstrations has been shown to outperform existing methods on robot manipulation benchmarks. While prior work has mainly focused on single-task settings, we study the problem from a multi-task perspective, as training independent models for each task is computationally expensive. Multi-Task policy learning comes with its own set of challenges, as naively training on a concatenated dataset of demonstrations would either require increased model capacity to accommodate the added complexity or result in drops in performance. We propose to distill knowledge from single-task CFM experts into a shared multi-task policy by transferring their learned velocity fields. We combine this distillation signal with the original CFM objective to retain fidelity to the demonstrations. Experiments on RLBench show that our approach improves multi-task policy performance over naive training while maintaining a fixed model size. 

---
# Fed-ReMasker: Federated Tabular Imputation under Feature-Level Missingness 

**Authors**: Ioannis Papathanail, Rooholla Poursoleymani, Lubnaa Abdur Rahman, Stavroula Georgia Mougiakakou  

**Link**: [PDF](https://arxiv.org/pdf/2609.28105)  

**Abstract**: Multi-center clinical studies and biomedical research collaborations increasingly seek to utilize data across centers to build models that generalize beyond any single center. This creates two distinct challenges: data protection regulations may restrict the sharing of raw patient data across institutions, while centers may collect only partially overlapping sets of features under different protocols. Federated learning enables collaborative model training without centralizing raw data. However, existing federated imputation methods rarely evaluate feature-level missingness, in which entire features are unobserved at some centers. To address this setting, we adapt the ReMasker masked autoencoder to federated learning (Fed-ReMasker), enabling centers to impute features never observed locally by leveraging knowledge learned across collaborating centers. We evaluate Fed-ReMasker in a benchmark spanning synthetic datasets with linear and nonlinear relationships and real-world tabular datasets, including clinical data. The benchmark varies the number of centers, the missingness ratios, and client heterogeneity. Fed-ReMasker achieves the lowest imputation error in 93.2% of value-level and 96.7% of feature-level scenarios in the homogeneous benchmark. It also remains robust to client heterogeneity using simple federated averaging, outperforming all baselines in all 36 value-level scenarios and each baseline in at least 35 of 36 feature-level scenarios, and comes within 3.0% on average of a centralized model trained on the pooled data. 

---
# Can LLMs Catch a Rigged Backtest? A Clean-Control Calibration Benchmark 

**Authors**: Makar Ulesov, Vladislav Smirnov, Omar Ibrahim, Arsenii Bobovnikov  

**Link**: [PDF](https://arxiv.org/pdf/2609.28090)  

**Abstract**: Backtest auditing is a calibration problem: high flaw recall is not useful when the model falsely flags matched clean strategies. We build a 96-item paired benchmark in which every flawed backtest has a clean control that holds strategy, dates, code style, labels, and reporting scaffold fixed while changing one methodology detail. A deterministic scorer separates flaw recall, clean-control false positives, evidence localization, and fix relevance. Over 1440 cached audits from four text endpoints, the primary DeepSeek auditor reaches 100.0\% closed and clean-aware code recall, but open prompts over-flag 93.8\% of clean code controls, and clean-aware all-three specificity is 87.5\% even where recall saturates. A clean-aware warning drops DeepSeek code false positives from 20.8\% (95\% CI 11.7--34.3) to 0.0\% (0.0--7.4) at unchanged recall, while the budget anchor still flags 38/48 clean controls under the same prompt. Reporting recall alone would rank three of these four models identically; reporting the clean-control rate separates them by 79 points. 

---
# LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations 

**Authors**: Sandra Arcos-Holzinger, Debashish Chakraborty, Rohita Mocharla, Will Walden, Andrew Yates, Reno Kriz, Sarah M. Erfani, James Bailey, Vishal M. Patel, Sanjeev Khudanpur  

**Link**: [PDF](https://arxiv.org/pdf/2609.28086)  

**Abstract**: We propose LAYERSCOPE, a label-free, layerwise framework that aims to characterize a model's learned representations in video and multimodal settings. Evaluating downstream performance using representations from final or intermediate layers typically requires large amounts of labeled data, repeated task-specific evaluations, and substantial computation. To address these limitations, LAYERSCOPE uses local, global, distributional, and correspondence-based geometric metrics to compare layerwise representation structure within and across models without requiring task-specific labels. We evaluate seven architecturally diverse models across video and multimodal classification, clustering, and text-to-video retrieval tasks from MVEB/MVEB+. We find that intermediate-layer representations can outperform final-layer and model-default outputs. We also find that no single geometric metric consistently predicts downstream performance, but note that distinct layerwise geometric signatures emerge across model families. LID shows task-dependent relationships with performance, while RankMe provides the strongest measure for classification and clustering, but is not a universal layer selector. We also find that pairing-aware metrics explain retrieval better than distributional distances alone. LAYERSCOPE therefore offers a framework for comparing representations across models and layers, enabling a more systematic evaluation in video and multimodal settings. 

---
# Curriculum Learning with GNN-based Reinforcement Learning for Job Shop Scheduling 

**Authors**: Jayakrishnan K. Vasudevan, Jonathan Hoss, Noah Klarmann  

**Link**: [PDF](https://arxiv.org/pdf/2609.28085)  

**Abstract**: The job shop scheduling problem is a challenging combinatorial optimization problem, and recent reinforcement learning approaches using graph neural networks have shown promise for learning scheduling policies directly from problem instances. However, training on large instances remains computationally expensive, and generalization across instance sizes remains challenging. This paper studies curriculum learning for graph neural network-based reinforcement learning in the job shop scheduling problem by comparing it with single-size training across three target sizes: 20 x 20, 25 x 25, and 30 x 30. In the curriculum setting, the policy is first trained on smaller instances and then progressively adapted to larger target sizes, allowing scheduling behavior learned in earlier stages to support learning on larger instances. Models are evaluated on unseen instances from 8 x 8 to 30 x 30 using the optimality gap, considering both generalization across all evaluation sizes and specialization on the target size. Results show that curriculum learning consistently reduces wall-clock training time, with larger benefits as the target size increases. The strongest advantage is observed at 30 x 30, where curriculum learning reduces the mean optimality gap across all evaluation sizes by approximately 8.1 percentage points, reduces the target-size mean optimality gap by approximately 8.6 percentage points, and saves approximately 50 hours of training time. 

---
# Prompt, Probe, Train, or Annotate? Single-camera sports video understanding in amateur settings 

**Authors**: Sai Varun Kodathala, Prashanth Pollishetty, Jaylen Cargill  

**Link**: [PDF](https://arxiv.org/pdf/2609.28049)  

**Abstract**: Video understanding is usually benchmarked on curated, single-actor, or professionally filmed clips, and a strong score there is routinely read as evidence a model is robust enough for deployment. Amateur team sport is a useful, largely untested place to check that assumption: over eight million students played a school sport in the United States in 2024-25 alone, almost none of it filmed by more than a single fixed camera, with several candidate actors crowded into frame and no operator or second angle to fall back on. Using volleyball as a test case, we ask whether strong performance on general video and world-model benchmarks translates into reliable, per-player attribution once footage is this chaotic, turning footage into statistics through a chain of tasks from finding play boundaries to naming who did what. We evaluate four approaches (prompting and agentic reasoning over frontier vision-language models, classical computer vision with small trained specialists, self-supervised video world models, and manual annotation) at every stage, on 66 amateur matches with 46,648 human-labelled contacts, filmed under conditions no published benchmark uses. No single paradigm wins every stage, and static, single-frame computer vision is not competitive at any stage involving motion or identity. A prompted model segments matches well, yet a far smaller trained model beats it at spotting contacts for a fraction of the cost, and the sport's own rules recover rally outcomes the pixels cannot. Identity is where every automated approach struggles: a jersey number is a static fact temporal reasoning cannot recover if never visible, unlike sporting action, a repeated motor pattern a temporal model can exploit, which is why holistic reasoning improves event detection while identity stays unchanged. We close with where each approach earns its cost, and what transfers beyond volleyball to amateur sport. 

---
# TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval 

**Authors**: Mourad Hassani, Julien Romero, Amel Bouzeghoub, Christian Jacquelinet  

**Link**: [PDF](https://arxiv.org/pdf/2609.28048)  

**Abstract**: Modern information retrieval (IR) systems rarely represent time, yet many information needs depend on it: in clinical, journalistic, and legal search, when an event occurred can decide whether a document is relevant. Dense retrievers and Retrieval-Augmented Generation (RAG) pipelines match queries to documents well on topic but poorly on time, so they surface content that is on-topic yet temporally wrong. We introduce Temporal Textual Similarity (TTS), a task that measures how well two anchored texts align in time, independent of their topical similarity. We then present TEMPS (Temporal Embedding Model for Precise Search), a modular temporal branch that attaches to a frozen semantic retriever and trains on that signal. It resolves anchored temporal expressions to intervals and moment-matches each one to a Gaussian; the resulting ordering supervises an anchor-date-conditioned encoder, whose score we fuse with the semantic score at inference. Grounding supplies the supervision, so training uses no hand-labeled temporal data. The temporal score itself is the Gaussian-KL inclusion measure from distributional embeddings; what TEMPS adds is the grounding and the moment-matched supervision. On three temporal benchmarks, TEMPS improves MRR for every semantic backbone tested and, on TS- Retriever, lifts R@1 from 19.92 to 25.39 over the prior temporal state of the art. 

---
# Evaluating Feedback Focus and Pedagogical Adaptivity in LLM-Generated Feedback on Student Writing 

**Authors**: Norah Almousa, Shayan Peyghambari Oskoui, Raquel Coelho, Gayle Rogers, Xiang Lorraine Li, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2609.28026)  

**Abstract**: We investigate whether state-of-the-art large language models (LLMs) generate feedback that reflects the pedagogical practices of expert teachers in terms of feedback focus and adaptivity. Previous evaluation efforts have examined feedback characteristics, its impact on learning, and its target, yet the focus of feedback and its adaptivity remains largely overlooked. To bridge this gap, we adopt and refine Narciss's taxonomy into seven feedback focus types to annotate teacher and LLM-generated feedback across three university writing courses. We release FeedType, a benchmark containing annotated teacher and LLM feedback from six LLMs under three prompting strategies. We assess the coverage and distribution of feedback focus types, and examine whether LLMs adapt their feedback across draft stages and student performance levels as an expert instructor does. Our findings show that while most LLMs cover most feedback focus types, they fail to reflect teacher feedback distributions and show varying levels of adaptivity, with none matching the teachers' adaptive behavior. We believe FeedType will support future research on pedagogical alignment in LLM feedback generation. 

---
# PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning 

**Authors**: Kevin Lee, Alison J. March  

**Link**: [PDF](https://arxiv.org/pdf/2609.28022)  

**Abstract**: Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined magnetic and plasma structure, and many learning methods provide a single anomaly score. We present the Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather (PISCES), a convolutional autoencoder trained without catalog labels on OMNI solar wind measurements under physics constraints. Its loss includes magnetic field consistency, an empirical relation between temperature and velocity, the Parker spiral angle, and penalties on changes between consecutive one-minute samples in derived quantities calculated from the reconstruction. At inference, PISCES separates the anomaly score into magnetic and plasma reconstruction errors, physics relations, and residual corrections, and reports the magnitude of each contribution. Attenuation of the skip connections, selected on validation data, improves average precision for the trained models, while the untrained scores remain nearly the same. The trained models also give a more consistent ordering of these physical contributions. After smoothing with a trailing median, the alarms can precede independently observed sudden commencements, including positive sudden impulses. 

---
# Controlled Attribute-Specific Summarization of Interrogative Dialogues 

**Authors**: A Aditya Bhardwaj, Arjit Singh Arora, Md Shad Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2609.28004)  

**Abstract**: Effective summarization of interrogative dialogues is a critical task in forensic and investigative settings, requiring high factual accuracy, coherence, and attribute-specific relevance. In this work, we introduce CASPER, a novel Chain-of-Thought Attribute-Specific Prompting for Evaluative Summarization framework that leverages structured prompting and iterative refinement to generate high-quality summaries of interrogator-witness interactions. We construct MINDSum, a dataset extending the MIND corpus, comprising 6,000 utterance pairs annotated with event details, factual statements, character descriptions, and fillers. CASPER employs RoleEval, a hierarchical evaluation mechanism where multiple roles (officer, inspector, senior inspector) iteratively assess summaries based on predefined criteria. By integrating entity extraction and structured feedback loops, CASPER significantly improves factual consistency and contextual completeness compared to existing baselines. Experimental results demonstrate that our framework outperforms standard summarization models on both lexical (ROUGE) and semantic (BERTScore) metrics, while human evaluation confirms its alignment with expert reasoning. Our findings underscore the potential of controlled summarization in high-stakes domains, paving the way for AI-driven forensic intelligence. 

---
# Compliant with Local Controls, Collectively Discriminatory. A Governance Architecture for Multi-Agent AI in Regulated Finance 

**Authors**: Jose Manuel de la Chica Rodriguez, Juan Manuel Vera Diaz, Pablo Delgado Romero  

**Link**: [PDF](https://arxiv.org/pdf/2609.27994)  

**Abstract**: Financial institutions are beginning to deploy agentic workflows in credit, fraud, collections, compliance, and operational control. Governance remains largely component-centric: each model or agent is specified, tested, authorized, and monitored locally. That is insufficient when institutional risk arises from the joint behavior of many locally acceptable components. We call this gap constitutional non-compositionality: local compliance checks need not compose into acceptable collective outcomes such as bounded disparate impact, market integrity, or traceable accountability. We propose ARIA as a finance-specific reference architecture and falsifiable research agenda for agent-population governance. It organizes six capabilities across normative-accountability, execution-control, and assurance-learning planes: policy specification, population-level observed-versus-expected behavior monitoring (M2), bounded authority, runtime containment, adaptive policy change, and preserved human oversight competence. Two simulations illustrate shared-signal thin-file exclusion under local controls and earlier warning from observed-versus-expected distributional monitoring in a constructed drift regime. The contribution maps these controls to fair-lending, EU AI Act, model-risk, and conduct-supervision evidence needs, and closes with a validation agenda rather than a production-effectiveness claim. 

---
# Riemannian Structure and Optimization for a Class of Low-Parametric Orthogonal Matrices 

**Authors**: Ali Aliev, Maxim Rakhuba  

**Link**: [PDF](https://arxiv.org/pdf/2609.27982)  

**Abstract**: In this paper, we are concerned with matrices formed by block-diagonal factors interleaved with fixed permutations -- a flexible family of structured matrices. This class has recently drawn interest in deep learning architectures for its balanced expressivity-efficiency trade-off, yet efficient computational strategies for working with it remain to be found. We approach this problem through Riemannian geometry and examine under what conditions this class admits a smooth manifold structure. For the practically important case of orthogonal two-factor matrices, we derive the essential Riemannian tools and propose efficient algorithms for their implementation. The algorithms leverage automatic differentiation, support parameter sharing within each factor, and avoid explicit dense matrix construction. We test them within the Riemannian optimization framework on the best matrix approximation problem and for parameter-efficient fine-tuning of large language models. Beyond the two-factor setting, we study the geometric and matrix-theoretic properties of factorizations with a larger number of block-diagonal factors. 

---
# Spread and Scale: What Determines Whether Test-Time Budget Allocation Pays 

**Authors**: Jinhyung Bae  

**Link**: [PDF](https://arxiv.org/pdf/2609.27917)  

**Abstract**: Neural combinatorial optimization solvers generate many candidate solutions per instance and report the best one found, using the same sample budget for every instance regardless of difficulty. A companion study showed that reallocating a fixed budget toward harder instances can improve solution quality, but that the standard way of measuring this improvement is biased: deciding an allocation and evaluating it on the same data can manufacture an apparent gain even when none exists. This left open what property of a workload determines whether reallocation is worth doing, and whether a policy that spends part of the budget to decide how to allocate the rest still pays once that cost is counted.
This paper answers both questions through pre-registered confirmatory experiments -- analysis and verdict criteria fixed before data collection -- across three independently trained solvers and two ways of constructing harder workloads on the traveling salesman problem. Within the workloads we study, the deciding property is how varied the instances within a workload are in difficulty, not how difficult the workload is on average: a uniformly easy or uniformly hard workload offers little room for reallocation, while a mixed workload offers substantial room. A budget-aware policy that pays for its own information about instance difficulty recovers most, though not all, of the improvement available when that information is assumed free.
Every experiment was independently recomputed from its written specification, and every correction to an earlier version -- including two that weakened the paper's own claims -- is reported with the direction it moved the conclusion. The paper offers a specific empirical answer and a template for verifying that answer is not an artifact of how it was measured. 

---
# Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It? 

**Authors**: Silin Chen, Yufei Yang, Xiaodong Gu, Yuling Shi, Chengcheng Wan, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27891)  

**Abstract**: Repository-level coding benchmarks have become the standard for evaluating coding agents, yet they inherently suffer from data leakage because they are built upon popular open-source repositories repeatedly used for training. Consequently, strong performance may reflect memorization of canonical repository cues rather than robust repository reasoning. We propose SchrodingerRepo (Schrödinger's Repository), an evaluation framework for testing coding agents under dynamically instantiated repository representations. Instead of repeatedly using a static representation of the test repository, SchrodingerRepo treats the test repository as an evaluation-time latent variable that is dynamically instantiated only when the agent enters the evaluation environment. The instantiated repository preserves the original executable behavior while eroding familiar cues such as naming conventions, file layouts, and implementation patterns through four transformation levels: problem statement reconstruction, namespace remapping, intra-file layout reordering, and functionality-preserving code rewriting. We evaluate popular LLMs on SWE-bench Verified and SWE-QA. Results show that removing familiar repository cues consistently degrades agent performance and substantially increases interaction costs across models. Further analysis reveals that the additional cost is primarily caused by increased difficulty in repository exploration and localization. These findings suggest that current coding agents may partially rely on memorized repository-side cues, highlighting the need for evaluation under dynamically instantiated repository representations. 

---
# RelCheck: Dual-Evidence Spatial Grounding for VLM Hallucination Correction 

**Authors**: Siddhi Patil, Navrati Saxena, William B. Andreopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2609.27890)  

**Abstract**: Multimodal large language models (MLLMs) fre- quently generate text that is inconsistent with the input image. While object- and attribute-level hallucinations have received considerable attention, relational hallucinations (incorrect de- scriptions of spatial or interactive relationships between objects) remain largely unaddressed by existing post-hoc correction methods. We present RelCheck, a training-free post-hoc correction pipeline that augments object-level visual grounding with dual relational evidence: learned scene-graph triples from RelTR and deterministic spatial predicates from bounding-box geometry. These combine with a Woodpecker-style object claim layer to form a three-layer visual knowledge base, which a language model corrector uses to rewrite hallucinated text. Evaluated on LLaVA v1 13B, RelCheck achieves a total MME hallucination score of 630.0 versus 585.0 for a Woodpecker-style baseline, with the largest gain on the position subtask (+31.7 points, accuracy+ improving from 0.367 to 0.600). A four-configuration ablation confirms that both relational layers contribute independently (McNemar p = 0.025). These results show that structured relational evidence meaningfully improves post-hoc hallucination correction on the spatial reasoning subtasks where current MLLMs are most deficient. 

---
# False-science induction in autonomous scientific discovery 

**Authors**: Hanbing Liang, Fujun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27883)  

**Abstract**: Closed-loop discovery systems increasingly execute experiments and update decisions autonomously, turning record integrity into part of the experimental apparatus. We show that false-science induction arises when legitimate physical objects and measurements are paired incorrectly, driving neural surrogates to faithfully learn record-induced associations that do not correspond to the true object-outcome relationship while marginal data distributions remain unchanged. Across green fluorescent protein fitness and materials band-gap prediction loops, coherent paired misbinding systematically redirects experimental budgets toward low-performing basins, whereas same-volume random swaps have negligible effects. These observations identify error coherence, rather than raw error frequency, as the primary variable controlling this budget misallocation in the tested loops. The resulting binding identifiability boundary supports monitored-axis quarantines and feedback-conflict triage, which intercept over-concentrated proposals before execution and isolate the corrupted hypothesis axis. 

---
# Bounded Loops: Pre-Run Spend Bounds, Proved Termination, and Verified Completion for Agent Harnesses 

**Authors**: Varun Pratap Bhardwaj, Garima Singh, Arun Pratap Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2609.27871)  

**Abstract**: In mainstream agent frameworks, a step ends when the agent's own output says it has finished. Durable-execution platforms bound retries and time, but their checker conventionally lives in the same codebase as the work: a discipline the deployment is trusted to keep, not a property the harness enforces.
We state what an agent harness must guarantee, prove it, and build the instrument that measures whether a harness delivers it. A bounded loop is a worker, an independent gate the worker cannot write to, and a declared budget; a bounded-loop graph composes them with a repair relation that lets a downstream failure re-run a finished upstream node. Three guarantees follow. It finishes: termination holds under repair, with the worst-case attempt total in closed form, if the repair budget is global not per node. It does not drift: no node reaches DONE without a gate verdict in an append-only hash-chained ledger, proved from control flow, since repair leaves no topological order to induct along. It does not overspend: the ceiling is enforced inside an attempt, not between attempts.
Gates are measured against a two-tier held-out mutant corpus. We characterise two classes that let a sound-looking check pass anything: vacuity, satisfied by the absence of the thing checked, and self-attestation, where the subject supplies the value the check is applied to. On a 69-loop catalogue the instrument found 47 vacuous gates in shipped, reviewed code. Against the repaired gates it reports no false accepts over 209 destroying mutants ($\alpha \le 1.8\%$, Wilson 95%); that figure is saturation, not quality: freezing the gates and applying a fresh operator family recovers a 23.3% false-accept rate where the exhausted corpus reported none. A rate belongs to a specific gate; the apparatus, not our number, is the contribution. Engine, catalogue and corpus are Apache-2.0. 

---
# TopoGS: Topology-Aware Anchor Feature Aggregation for Large-Scale 3D Gaussian Splatting 

**Authors**: Wei Zhang, Shiqiang Gong, Shengkai Yu, Zeyu Wang, Clement Mallet, Zhitong Xiong, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27868)  

**Abstract**: Octree-based 3D Gaussian Splatting organizes anchors into multi-level hierarchies for level-of-detail rendering, but features at different levels are typically optimized independently, leaving the octree topology underused during feature learning. We observe that uniform cross-level aggregation produces asymmetric effects: fine-level anchors benefit from coarse context, whereas coarse-level anchors require selective information from their descendants. We therefore propose TopoGS, a topology-aware anchor feature aggregation framework with two lightweight components. Hierarchical Anchor Coupling establishes bidirectional cross-level gradient pathways by fusing per-level context triplets with a residual MLP. Structure-Aware Containment Aggregation uses octree containment and hash-based matching to distinguish anchors with valid parent-child relations from isolated anchors, then applies soft weighting to accommodate varying topological sparsity. Experiments on ten scenes from Mill19, UrbanScene3D, Tanks & Temples, MatrixCity, and WHU show consistent improvements over state-of-the-art methods. TopoGS achieves average PSNR gains of 2.13, 1.78, and 0.29 dB over the strongest reported baseline on aerial, ground-level, and synthetic-cartographic scenes, respectively, while rendering faster and using less memory. Code is available at this https URL. 

---
# A Shared Encoder Is Not a Shared Task: Conditional Comparison for Deep Expert Pools 

**Authors**: Kentaro Oda  

**Link**: [PDF](https://arxiv.org/pdf/2609.27866)  

**Abstract**: Sharing a deep encoder does not, by itself, fix the central confound of task-comparison scores. We show that cross-evaluated heads on a frozen shared representation inherit the extrapolation confound of shallow exchange scores: pure input rotations with fixed labels inflate a deep exchange score from about 0 to 0.80, while representation-novelty scores are blind in the complementary direction (flat under label permutations that change the task completely). Transplanting a conditional two-discriminator discrepancy into the embedding space resolves both blind spots: the functional axis stays within +-0.001 under rotations and tracks label-permutation drift mass monotonically. Built into a mixture-of-heads lifecycle, the two-axis gate attains better decision quality with fewer heads than exchange or novelty triggers at a matched training budget. On generalized category discovery, the same chunk-level functional axis separates semantic novelty from photometric shift with AUROC 0.98-0.99 where per-input OOD scores (MSP, Energy, Mahalanobis, KNN) sit near chance for that distinction. All findings replicate across frozen ImageNet-21k ViT-B/16 and self-supervised DINOv2 backbones on CIFAR-100, and extend to residual adapter pools with recurrence, where a null-calibrated novelty trigger never fires on mechanism changes while the two-axis gate handles them with full recurrence reuse. We state explicitly the common-factoring condition under which embedding-space conclusions transfer to the original mechanism. 

---
# What Changed? Drift Detection with Real, Virtual, and Incomparable Diagnosis 

**Authors**: Kentaro Oda  

**Link**: [PDF](https://arxiv.org/pdf/2609.27865)  

**Abstract**: Sharing a deep encoder does not, by itself, fix the central confound of task-comparison scores. We show that cross-evaluated heads on a frozen shared representation inherit the extrapolation confound of shallow exchange scores: pure input rotations with fixed labels inflate a deep exchange score from about 0 to 0.80, while representation-novelty scores are blind in the complementary direction (flat under label permutations that change the task completely). Transplanting a conditional two-discriminator discrepancy into the embedding space resolves both blind spots: the functional axis stays within +-0.001 under rotations and tracks label-permutation drift mass monotonically. Built into a mixture-of-heads lifecycle, the two-axis gate attains better decision quality with fewer heads than exchange or novelty triggers at a matched training budget. On generalized category discovery, the same chunk-level functional axis separates semantic novelty from photometric shift with AUROC 0.98-0.99 where per-input OOD scores (MSP, Energy, Mahalanobis, KNN) sit near chance for that distinction. All findings replicate across frozen ImageNet-21k ViT-B/16 and self-supervised DINOv2 backbones on CIFAR-100, and extend to residual adapter pools with recurrence, where a null-calibrated novelty trigger never fires on mechanism changes while the two-axis gate handles them with full recurrence reuse. We state explicitly the common-factoring condition under which embedding-space conclusions transfer to the original mechanism. 

---
# Query Implied Generative Engine Optimization 

**Authors**: Shilpa Ramakrishna, William B. Andreopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2609.27845)  

**Abstract**: The landscape of search has changed drastically with how people look for information online. Traditional search engines are being replaced by Generative Search Engines (GSEs), which use Large Language Models (LLMs) to generate natural language responses to user queries. For content creators, visibility is no longer solely determined by ranking in search results but by being cited within generated responses. But Generative Search Engines are black-boxes, leading to the emergence of Generative Engine Optimization (GEO), a set of techniques aimed at improving content visibility in generative search settings. Most existing approaches rely on the explicit queries or query derived signals to align content to better suit user needs. We propose Query Implied Generative Engine Optimization (QI-GEO) to infers user intent directly from the document. Our approach approximates document's intent space and identifies content that may be missing yet relevant to answer potential user queries. Evaluation on GEO-Bench and Extended GEO-Bench demonstrated improvements across objective and subjective metrics. QI-GEO improved objective scores by up to 15.9% and subjective scores by up to 17.6%, while yielding nearly twice as many citation gains as citation losses. These results suggest that document-derived approximations of user intents can improve visibility without relying on explicit query inputs. 

---
# A Non-Invasive Cloud-Based Migration Strategy for Post-Quantum Cybersecurity in Smart HVAC Systems: Architecture, Implementation, and Empirical Evaluation 

**Authors**: Mahedee Zaman Moon, Kaysarul Anas Apurba, Md Hasibul Hasan, Sk Md Mizanur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2609.27828)  

**Abstract**: Legacy smart HVAC controllers rely on vendor-cloud TLS secured by ECDH and RSA, both broken by Shor's algorithm, and typical 10-15 year lifespans mean today's devices remain in service through the quantum-threat era. Direct on-device post-quantum cryptography is infeasible: an ESP32-S3, representative of capable HVAC hardware, has only 339 KB free heap against the 900 KB ML-KEM-768 requires, and even classical ECDH-P256 keygen (111.93 ms) dwarfs hardware AES-128 (0.032 ms). We propose a non-invasive PQC proxy, requiring no device, firmware, or vendor-cloud changes, performing ML-KEM-768 encapsulation and ML-DSA-65 authentication (NIST FIPS 203/204) with AES-256-GCM session keys via HKDF, implemented with Open Quantum Safe liboqs on a Raspberry Pi 4B gateway. Over 500 runs, the post-quantum handshake (Steps 1-6) completes in 2.48 ms, 0.38 ms slower than classical baseline, with PQC computation around 8% of handshake time at 20 ms simulated round-trip network latency. The gateway sustains 443 sessions/second, 100% success under 32 concurrent connections, extrapolating to 3546 sessions/second on a 32-core cloud instance. Five side-channel tests, including verified in-place session-key zeroization and a fixed-vs-random TVLA timing analysis, found no exploitable timing leakage or susceptibility to man-in-the-middle attacks. The architecture is vendor-agnostic and becomes unnecessary once vendors adopt NIST PQC natively. 

---
# Safe Multi-Robot Coordination via VLM-LLM Reasoning and Reachability Analysis 

**Authors**: Mohamed Dwedar, Ahmad Hafez, Alexander Jesser, Amr Alanwar  

**Link**: [PDF](https://arxiv.org/pdf/2609.27816)  

**Abstract**: Safe coordination in heterogeneous machine-to-machine (M2M) robotic systems is challenging when robots differ in sensing capabilities, environmental awareness, and motion execution roles. This paper presents a centralized safety-aware M2M framework for cooperative goal-directed navigation in a heterogeneous mobile robot team comprising a vision-capable quadruped and a camera-less robotic vehicle. The objective is to guide both platforms toward a goal region while avoiding static and dynamic obstacles and preventing unsafe inter-robot interactions. Under the principle of shared perception, the vision-capable robot provides semantic environmental awareness through a centralized server over an MQTT broker, enabling the camera-less platform to navigate using this shared scene representation alongside its own odometry, IMU, and state feedback. A vision-language model (VLM) interprets the visual stream, and the extracted semantic data is mapped into conservative metric geometric constraints, including inflated obstacle sets, safe corridors, and goal regions. A large language model (LLM) proposes high-level task allocations, while physical command authority is restricted to a robot-specific zonotope reachability gate. This verification engine propagates independent reachable tubes to evaluate obstacle avoidance, safe-corridor containment, and inter-robot separation predicates before approving commands. Online experiments across clear-path and dynamic-obstacle scenarios show that the pipeline reliably approves safe motion, triggers conservative replanning or holding maneuvers upon constraint violation, and enforces a strict architectural separation between advisory semantic reasoning and formally verified motor execution. 

---
# Evaluating ADC-only deep learning pipelines for breast cancer detection and segmentation using standalone diffusion-weighted MRI 

**Authors**: Pablo García Marcos, Paula Puerta Gonzĺez, Guillermo Lorenzo, Héctor Gómez, Covadonga del Camino, Angel Rio-Alvarez, Víctor M. González  

**Link**: [PDF](https://arxiv.org/pdf/2609.27815)  

**Abstract**: Dynamic contrast-enhanced (DCE) imaging is the gold standard technique for the detection and characterization of breast cancer using magnetic resonance imaging (MRI). However, DCE-MRI requires long acquisition times and the administration of contrast into the bloodstream, which can cause allergic reactions. Alternatively, diffusion-weighted MRI (DW-MRI) is a standard complementary technique for breast MRI that does not require contrast, has shorter acquisition times, and enables calculation of apparent diffusion coefficient (ADC) maps that correlate with tumor cellularity. Yet, despite these technical advantages, deep learning research has focused on DCE-based models and has barely explored the tumor detection performance of DW-MRI and ADC maps either in combination with DCE-MRI or as standalone alternatives. Here, we evaluate the application of different state-of-the-art deep learning techniques for detection and segmentation of breast cancer using ADC-only images. This is, to our knowledge, the first comprehensive evaluation of ADC-only breast cancer pipelines for classification, detection, and segmentation tasks. 

---
# LabourCrew: A Multi-Agent RAG Framework for Trustworthy Adversarial Deliberation and Statutory Reasoning over Labour Law 

**Authors**: Fatema Tuj Johora Faria, Mukaffi Bin Moin, Jubayer Al Mahmud, M. F. Mridha, Md. Alam Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2609.27814)  

**Abstract**: In statutory question answering, every claim must be traceable to evidence, not merely relevant, since unverifiable labour-rights answers carry serious legal consequences. Current systems fall short: single-pass RAG cannot detect insufficient evidence, while multi-agent legal-debate systems treat grounding as a prompting convention, letting agents cite unretrieved evidence. To address this gap, we introduce LabourCrew, a multi-agent RAG framework built around three grounding mechanisms: StatuteGraph, a graph index that explicitly links chapter, section, proviso, and cross-reference structure rather than fixed-length spans; an Evidence Exchange Protocol that confines advocates and an interpreter to an evidence ledger, making citation to unretrieved text impossible, while a fault-tolerant supervisor board runs advocates in parallel so individual failures degrade rather than crash the system; and a Calibrated Trust Gate that replaces categorical accept/reject decisions with a trust score, thresholded via conformal risk control for a distribution-free bound on the false-accept rate. We evaluate on LabourActQA, a 500-item Bangla question set from the Bangladesh Labour Act, 2006, spanning seven reasoning categories and three difficulty tiers. The framework drives the empirical false-accept rate to 0.081, within the target level ($\alpha = 0.10$), achieves the highest Answer Relevancy among HyDE RAG, Graph-RAG, and Hierarchical RAG (0.862 $>$ 0.839, 0.815, 0.828), and degrades gradually rather than catastrophically as question difficulty increases. These results show that calibrated abstention, not retrieval quality alone, is what makes legal question answering auditable in low-resource statutory domains. 

---
# EidosDoc: Implicit Structure Encoding for Cost-Effective Semi-Structured Document QA 

**Authors**: Teng Lin, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27784)  

**Abstract**: Semi-structured documents are ubiquitous in scientific reports, financial statements, and technical manuals. Question answering over such documents requires simultaneous understanding of text, tables, charts, and complex hierarchical layouts. Existing methods either rely on repeatedly calling large language models for structure parsing and retrieval, leading to high cost and large latency, or they flatten the document and lose layout and hierarchy information, sacrificing answer accuracy. To address this, we propose EidosDoc, a novel system that achieves state-of-the-art accuracy with minimal computational expense. Our approach introduces three core innovations. (1) An Implicit Structure Encoder trained via contrastive learning and a structure consistency loss. This module jointly embeds hierarchical relationships, spatial positions, and textual content into a dense vector space, capturing document structure holistically without the need for manually defined and error-prone constructions. (2) A Hybrid Retrieval Pipeline that leverages BM25, layout fingerprints, and a lightweight cross-encoder to perform high-precision retrieval entirely without invoking an LLM, drastically reducing cost and latency. (3) A Dynamic Evidence Expansion mechanism that adaptively retrieves spatially adjacent and structurally related evidence, overcoming the evidence omission common in fixed-path retrieval methods. We evaluate EidosDoc on four benchmarks, and comprehensive evaluations show that EidosDoc achieves a new state-of-the-art accuracy on the four benchmarks. Crucially, it does so with a 50 times reduction in cost and 4 times lower latency compared to the previous state-of-the-art Method. These results demonstrate that EidosDoc establishes a new optimal trade-off among accuracy, cost, and speed, offering a practical and scalable path for accurate semi-structured document analysis. 

---
# Beyond Unsafe Detection: Counterfactually Anchored Evidence Attribution for Multi-Turn LLM Safety Failures 

**Authors**: Srinivasan Subramanian, Kazi Aminul Islam, Md. Abdullah Al Hafiz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27773)  

**Abstract**: As Large Language Models (LLMs) move from conversational assistants to advanced agentic systems, guardrail failures can convert adversarial intents into harmful executions. However, most guardrail evaluation frameworks focus only on the result and assess whether a user request is safe or unsafe. This approach is insufficient for multi-turn failures, where adversarial intent is distributed across multiple turns. This motivates us to go beyond detection to identify the turns and tokens that push the conversation toward unsafe trajectories. To support this, we construct a multi-turn dataset with behavioral validation and tiered evidence supervision. The dataset contains 1,762 conversations, including adversarial conversations, benign twins, and benign variants with high-risk vocabulary. We train a lightweight hierarchical attribution model that predicts safety violations and attributes them to contributing user turns and token spans. The model achieves strong detection performance (F1=0.988), and removing the top 15% of attributed tokens reduces the adversarial classification confidence by 51.1%. The model preserves low false positive rates on benign conversations with high-risk vocabulary, with false positives below 1% on both borderline benign and benign high-risk vocabulary conversations, compared to 37.3% and 94.7% for a keyword-based surface-risk baseline. Independent human annotation supports the model's attribution performance, with the top-five attributed turns containing a human-identified evidence-bearing turn in 84.5% of adversarial cases. 

---
# Backdoors Leave Structural Traces: FedMAST for Backdoor Detection and Containment in Federated Learning 

**Authors**: Srinivasan Subramanian, Kazi Aminul Islam, Md. Abdullah Al Hafiz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27760)  

**Abstract**: Federated learning enables distributed training of a shared model without requiring clients to share their raw data. However, its reliance on the integrity of the client-submitted updates exposes the global model to stealthy backdoor poisoning. Although existing defenses often inspect isolated evidence sources, stealth-constrained attacks can adapt to these signals. In this paper, we show that such attacks can suppress isolated anomaly signals, but their poisoned updates still leave residual structural traces. We propose FedMAST, a Federated Multi-Axis Structural Tracing defense for backdoor detection in federated learning. FedMAST scores client updates using complementary structural, spectral, and historical evidence and then applies tiered filtering and round-level containment to limit adversarial influence. To capture traces that isolated signals may miss, FedMAST uses squeeze-pair coherence scoring to expose coupled feature distortions and signed spectral-drift tracking to reveal persistent directional changes over time. Across six federated backdoor attacks, namely Constrain-and-Scale, Neurotoxin, BC-Layers, LGA, DBA, and 3DFed, FedMAST achieves lower ASR than baseline defenses in all nine evaluated attack--defense comparisons. Across the complete 200-round runs, it attains an average ASR of 1.51% while maintaining 94.84% average main-task accuracy. Under the method-aware CovertLayers attack, FedAvg, MultiKrum, AlignIns, and FLAME yield full-run ASRs of 100.00%, 99.67%, 99.53%, and 32.84%, respectively. FedMAST achieves the lowest ASR among all evaluated methods, reducing it to 1.53% while maintaining 92.26% main-task accuracy. 

---
# Hard Negatives Reveal What Easy Negatives Hide: Cross-Lingual Harmfulness Representations Degrade with Resource Tier Under Hard Negatives 

**Authors**: Paras Balani, Subhrakanta Panda  

**Link**: [PDF](https://arxiv.org/pdf/2609.27758)  

**Abstract**: Safety alignment in large language models is trained primarily in English, and recent work reports that the underlying harmfulness representation survives translation: English-trained probes separate harmful from harmless prompts almost as well in low-resource languages as in English. This has been taken as evidence that cross-lingual refusal failures mainly reflect calibration rather than representation quality. We show that this conclusion depends on the choice of negative examples. Across nine languages spanning three resource tiers, we replicate near-perfect transfer (AUROC > 0.98) when harmless prompts come from an unrelated distribution (easy negatives). With XSTest contrast prompts, which are benign but surface-similar to harmful requests (hard negatives), transfer collapses in low-resource languages while remaining largely stable in high-resource languages. On Qwen2.5-7B-Instruct, mean AUROC drop increases from 0.003 in English to 0.017 in high-resource, 0.042 in mid-resource, and 0.276 in low-resource languages. The pattern replicates on Aya Expanse. Back-translation chrF controls and a matched-chrF comparison across three languages reduce the likelihood that translation quality explains the effect. The collapse remains after controlling for chrF (partial r = 0.70, p = 0.03). Tokenizer fertility correlates with the collapse and explains part of the resource-tier effect, but not all of it. The results show that easy-negative transfer can coexist with substantial degradation under hard negatives. Easy-negative evaluation alone therefore cannot establish that the harmfulness representation survives translation. 

---
# InfiNoVA: Infinite Novel View Augmentation for Viewpoint Invariant Robot Policies 

**Authors**: Sai Puneeth Reddy Gottam, Elmar Rueckert, Vedant Dave  

**Link**: [PDF](https://arxiv.org/pdf/2609.27734)  

**Abstract**: Vision-Language-Action (VLA) policies often rely strongly on the camera viewpoints seen during training, causing substantial performance degradation when deployed from unseen perspectives. Collecting demonstrations from sufficiently diverse physical viewpoints is expensive and still provides only sparse coverage of the viewpoint space. We introduce InfiNoVA, a data-augmentation framework that converts synchronized multi-camera demonstrations into a dense distribution of geometrically consistent training views. InfiNoVA reconstructs each manipulation trajectory as a time-varying 3D Gaussian representation and renders novel observations from sampled camera poses while preserving the original state-action correspondence. This explicit scene representation improves frame-level fidelity and temporal consistency while reducing task-critical hallucinations observed in generative novel-view synthesis. Across four real-world manipulation tasks, policies trained with InfiNoVA achieve 5.4x higher average success under unseen randomized viewpoints than both VISTA-based augmentation and the unaugmented policy. InfiNoVA further achieves 1.7x higher success than training directly on all five physical camera views. These results show that dense, geometrically grounded viewpoint augmentation provides a practical route toward camera-robust robot policies without modifying the underlying policy architecture. 

---
# AI-Driven Neural Surrogates for In Silico Design of Cognitive-Affective Neuromodulation Targets 

**Authors**: Marco Rothermel, Madleen Stenger, Soroush Daftarian, Svenja Jule Francke, Bita Shariatpanahi, José C. García Alanis, Mohammad-Ali Nikouei Mahani, Stefan G. Hofmann, Tim Hahn, Hamidreza Jamalabadi  

**Link**: [PDF](https://arxiv.org/pdf/2609.27729)  

**Abstract**: In neuropsychiatry, the primary goal is often not only to decode brain activity but to change it, for example to lessen a negative affective bias or an overly salient memory. Motivated by control theory, we develop an AI-driven neural-surrogate framework that proposes candidate representational changes and tests their predicted perceptual effects from snapshots of stimulus-evoked fMRI activity, without physical stimulation. The framework combines fMRI decoding, deep generative modeling, and constrained latent-space steering. Valence and memorability are used only as worked examples. Using more than 36,000 image-fMRI observations from four deeply sampled Natural Scenes Dataset participants, subject-specific models recovered coarse generative structure from visually responsive cortex (two-way identification, 0.79-0.88; chance, 0.5). Graded perturbations were reconstructed as images and evaluated with automated scorers and human ratings from 7,200 trials by 18 participants. In the primary VDVAE model, valence shifted from -0.61 to +1.03 SD and memorability from -1.34 to +1.45 SD; a later Versatile Diffusion refinement reduced or altered these effects. Across five perturbation levels, human valence ratings moved in the predicted direction under the linear time-correction model (mean slope, 0.038 SD per unit of alpha; 95 percent CI, 0.003-0.074; positive in 16 of 18 participants). Perceived memorability did not change reliably. Baseline agreement with the automated assessor was suggestive for valence (r = 0.30) and weak for memorability (r = 0.10). Extreme perturbations drifted from the original stimulus, so intended change must be weighed against loss of fidelity. These findings provide a falsifiable upstream method for designing and behaviorally testing candidate representational targets for future neuromodulation in psychiatry, while marking the limits of the present static approximation. 

---
# The Path Matters: Evaluating Small Language Models Beyond Answer Accuracy in KGQA 

**Authors**: Eduin E. Hernandez, Sergio A. Diaz, Luis F. Garcia, Nurassyl Askar, Stefano Rini  

**Link**: [PDF](https://arxiv.org/pdf/2609.27669)  

**Abstract**: Small language models (SLMs) are increasingly paired with knowledge graphs (KGs), yet end-to-end KG question answering conflates graph access, search, navigation, reasoning, and answer generation. This coupling makes it difficult both to determine whether an SLM can faithfully execute the reasoning path implied by a question and to attribute failures to navigation rather than to other stages of the pipeline. We isolate this capability by employing the THESEUS navigation and traceability framework and using frozen, off-the-shelf SLMs as local action policies. At each hop, the environment exposes the legal outgoing graph actions, and the model selects one executable graph action and decides whether to stop, without task-specific parameter updates, model-controlled beam search, or free-form answer generation. This controlled setting allows us to evaluate terminal-answer accuracy with Hits@1 together with path fidelity, using Path Edit Distance (PED) as the primary trajectory metric. Across the Kinship and MQuAKE-ST KGQAs, similarly sized local models differ substantially in answer accuracy and path fidelity, with the two metrics sometimes favoring different models. This model-dependent behavior also extends to prompting, as a single demonstrated trajectory can improve or degrade navigation depending on the model. These results motivate evaluating SLM graph reasoning beyond endpoint accuracy alone. 

---
# FLEET: From Logits Entropy to Enhanced Trajectories in Text Generation 

**Authors**: Oleksii Streltsov, Oleksandra Vitko  

**Link**: [PDF](https://arxiv.org/pdf/2609.27657)  

**Abstract**: Solutions based on large language models (LLMs) often rely on temperature sampling to improve accuracy and stability by aggregating multiple samples from the completion distribution. However, this memoryless approach is inherently suboptimal: because it lacks awareness of prior generations and their evaluations, it produces an increasing proportion of semantically duplicate answers as more samples are drawn, leading to diminishing returns. To address this limitation, we introduce FLEET, a novel method that integrates a memory mechanism into the generation process. FLEET represents each generation as a sparse trajectory through states whose entropy exceeds a predefined threshold and uses these trajectories to infer per-token utility scores that adjust the logits. Benchmark evaluations demonstrate that FLEET achieves the same accuracy as the repeated sampling baseline, with a 3x speedup, and substantially improves accuracy on complex coding tasks (LiveCodeBench Pass@32 increases from 59.9% to 66.2%) under the same budget. Furthermore, in the greedy-decoding configuration evaluated here, the approach is deterministic and uses a single calibration pass to derive its principal hyperparameters, requiring only minimal modifications to existing LLM pipelines. 

---
# InternW0: A Foundational Physical World Model for Efficient Real-World Interactions 

**Authors**: Jisong Cai, Yao Mu, Ganlin Yang, Zhe Cao, Zhangzheng Tu, Xing Gao, Kailin Li, Xinyu Zhan, Lixin Yang, Yangkun Zhu, Haoxiang Ma, Ming Zhou, Qiaojun Yu, Yufei Xue, Liqun He, Yifei Yao, Yifan Zhu, Long Ling, Bingqi Jiang, Haoyu Guo, Xueyue Zhu, Bowen Zhou, Bin Zhao, Tianfan Xue, Chunhua Shen, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27656)  

**Abstract**: Physical intelligence requires more than predicting how the world may evolve: predictions must remain actionable as the world continues to change. We introduce InternW0, the first instantiation of the InternW physical world model series from Shanghai AI Laboratory, built around omnimodal interfaces, asynchronous multi-frequency processing, and local physical modeling under partial observations and external influences. InternW0 jointly learns future visual dynamics and continuous robot control through an asymmetric video--action architecture with flow matching. A high-capacity video expert provides longer-horizon predictive context, while a lightweight action expert operates at a faster timescale. Instead of regenerating the future for every action update, InternW0 reuses layerwise K/V and adapts it to newly observed states through observation-conditioned context routing. Domain-specific interfaces and soft prompts support heterogeneous embodiments, while contact-aware post-training incorporates force and tactile signals for contact-rich manipulation. We train InternW0 on approximately 7,200 hours of heterogeneous robot and egocentric data, including EgoLab, a 275-hour real-laboratory egocentric dataset. Evaluation spans simulation benchmarks and real-world scientific tasks, including a 15-stage metal--organic framework synthesis workflow and 5-stage contact- and force-aware dexterous manipulation for general-purpose quantitative pipetting. These results advance scalable, asynchronous, and science-native physical world models for universal and efficient real-world interactions. 

---
# Learning Local Heterogeneity and Cross-Region Context for Large-Scale Traffic Forecasting 

**Authors**: Qi Feng, Zidong Wang, Bo Li, Xiaoguang Gao, Jiayu Zhang, Chenfeng Wang, Kaifang Wan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27637)  

**Abstract**: Traffic flow forecasting is essential to intelligent transportation systems. Large-scale traffic forecasting requires jointly modeling local spatial dependencies and cross-region this http URL dependencies between geographically neighboring nodes are heterogeneous due to differences in road identity and travel direction, while acquiring global information through allpairs node interactions incurs substantial computational costs. Therefore, capturing local heterogeneity while efficiently acquiring long-range context remains an important challenge in largescale traffic forecasting. To address these challenges, we propose LoReST, a Local-Region Spatial Temporal network that models spatial dependencies at two complementary granularities: node neighborhoods and road network regions. Specifically, relation-aware local aggregation captures heterogeneous dependencies within geographic neighborhoods through road and direction specific feature transformations. Cross-region interaction constructs region representations through mean pooling, exchanges long range context via inter-region attention, and broadcasts it back to nodes. By integrating local information aggregation with crossregion interaction, LoReST is able to effectively achieve spatial dependency learning in large-scale road networks. Experiments on four datasets of the LargeST benchmark show average relative reductions of 4.78%, 3.60%, and 5.75% in MAE, RMSE, and MAPE, respectively. 

---
# Compliant AI Infrastructure for Regulated Finance: A tiered multi-agent framework with DLT audit trails for financial operations in DACH 

**Authors**: Walter Kurz, Reinhard Magg  

**Link**: [PDF](https://arxiv.org/pdf/2609.27632)  

**Abstract**: We present a compliance-first architecture for AI in regulated finance that treats regulation as an orientation layer rather than a deterministic ruleset. A matrix of regulatory intent and exposure provides a compact classification handle, which a governed policy compiler then maps into concrete prohibitions, obligations and runtime budgets. Prohibitions constrain feasibility and block externalisation, while obligations extend tasks with artefacts that must meet explicit admissibility criteria. Committee activation remains policy-driven and proportionate, preserving efficiency while ensuring supervisory oversight. Evidence, decisions and reason codes are bound to a permissioned DAG with deterministic timestamping, enabling replay, provenance checks and clear attribution of failure. Clause-level legal indexing with effective dates and capability-based agent routing ensure portability across DACH and the wider EU. The result is assurance by construction: compliance is embedded in execution and verifiable by auditors without sacrificing proportionality or transparency. 

---
# InGuard: Towards Generalized Inner Guardrail for Safe Text-to-Image Generation 

**Authors**: Zeyu Wang, Xiaodan Li, Zhiwen Li, Yuefeng Chen, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2609.27620)  

**Abstract**: Modern text-to-image (T2I) models generate high-quality images from arbitrary user prompts, yet they can just as easily produce not-safe-for-work (NSFW) content. Conventional outer guardrails consist of two components: a prompt classifier that checks for risk before generation, and a post-hoc image classifier that checks the fully generated image. In this design, both classifiers operate outside the generation pipeline and do not use the model's own representations. This separation can limit prompt-screening accuracy, while the image-side check runs only after the full generation cost has been spent. Moreover, a flagged prompt can only be rejected, even when it could be adjusted to produce a safe image. In this work, we propose the Inner Guardrail (InGuard), a safety framework that works inside the pipeline on the model's own representations, leaving base-model parameters untouched. First, a risk classifier grades each prompt as unsafe, risky, or benign based on the text encoder's embeddings, with no external language model. Second, SAGE (Soft-gated Asymmetric Guardrail for Embeddings) modifies the embeddings of risky prompts, aiming to return a safe image instead of a refusal. Third, a latent detector checks the one-step clean latent estimate midway through denoising, reaching nearly image-level performance and halting generation when risk is detected. We also construct the RevGen Safety Benchmark to evaluate T2I safety under realistic conditions: 10,000 prompts built through real-image reverse generation, with a rewriting step that supplies controlled intellectual-property (IP) characters, covering graded porn/gore risks, categorical IP risks, and benign negatives. Across five open-weight T2I models, InGuard reaches 97.9-98.8% safety rate, matching or exceeding the outer guardrail, with 57.5-73.5% less benign disturbance, ~3.7x fewer parameters, and 50-55.6% of denoising steps skipped. 

---
# Can Jev Judge Radiology Reports? Evaluating a System One Model for Clinical Factuality 

**Authors**: Jiaju Huang, Hao Yang, Xinyu Ma, Xinglong Liang, Kunyan Cai, Junqiang Ma, Shaobin Chen, Yue Sun, Tao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27607)  

**Abstract**: An AI-generated radiology report can resemble a physician's report while omitting an abnormality, adding an unsupported finding, or reversing its presence. Measuring these factual differences is essential for evaluating report generators. We study Jev, a System One decision model, as a simple, low-cost judge of agreement with physician-written reference reports. Our evaluator checks whether each statement is supported by the other report and combines these judgments in both directions to capture unsupported claims and omissions. A single-question configuration reaches Kendall correlations of 0.573 on RadEvalX and 0.398 on RadEvalExpert with expert error counts, outperforming an open natural language inference judge under matched decomposition and aggregation. One support question per statement retains similar expert agreement to seven while using 43-45% fewer judgment input tokens. At the documented API price, judgments cost under three cents per hundred report pairs, excluding local decomposition. In a separate controlled-error test, Jev detects false negation with an AUROC of 0.977. Local RadMatch achieves stronger agreement on clinically significant errors in both expert datasets and on total errors in the shared RadEvalExpert subset. Finding-count and error-scope analyses show that benchmark agreement reflects report size and error definitions as well as medical error detection. These results support Jev as a practical judgment component for measuring factual differences in generated radiology reports and identify where more elaborate evaluation remains valuable. 

---
# When Context Misleads: In-context Learning with Jurisdiction in Large Language Models 

**Authors**: Pei-lin Li, Qingle Liu, Junyang Feng, Siyu Li, Sunqi Fan, Xin-Sheng Chen, Shuojin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27603)  

**Abstract**: In-Context Learning (ICL) has become a cornerstone of modern LLM deployment. However, existing ICL post-training methods have a critical blind spot: they excel at extracting patterns from demonstrations while often neglecting context authority, the ability to determine whether contextual information should govern the final answer. To benchmark this capability, we introduce FakeContextBench, which contains pseudoscientific claims across seven domains. Our evaluation of commercial and open-source models shows that large-scale pre-training alone is insufficient for reliable context-authority discrimination. Moreover, prevalent ICL fine-tuning methods can increase susceptibility to misleading context, reducing reality accuracy by up to 14.95 percentage points relative to the base model. To address this trade-off, we propose Jurisdiction In-Context Learning (J-ICL), a post-training framework that incorporates context validation into the training objective. Across four model backbones, J-ICL improves ICLEval by an average of 5.84 percentage points and reality accuracy by 9.20 points over the corresponding base models. It also raises the Reality Rate by an average of 18.09 points relative to MetaICL and Symbol Tuning. These results demonstrate that ICL capability and resistance to deceptive context can be improved together. The benchmark is available at this https URL. 

---
# Hidden not Deleted: How Networks Suppress Entangled Features 

**Authors**: Akash Samanta, Manish Pratap Singh, Debasis Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2609.27593)  

**Abstract**: Concept erasure methods that operate via linear projection assume that features occupy separable subspaces. We show this assumption fails under dense superposition: when two features are forced into an antipodal pair sharing a single subspace, state-of-the-art linear erasure destroys both, not just the target. Networks trained with gradient descent instead solve this problem non-linearly, but not uniformly: they converge to one of two distinct circuit-level solutions depending on initialization, which we call mirror and shadow solutions. We map this bifurcation as a function of feature entanglement, show it reflects a stable attractor structure rather than an artifact of our setup, and use targeted causal interventions to demonstrate that both solutions leave a substantial, measurable trace of the erased feature's representation intact, recoverable through a single scalar patch rather than requiring any further training. This mirrors a failure mode recently observed empirically in LLM unlearning, where suppression rather than deletion allows forgotten knowledge to resurface; our results offer a mechanistic, causally-validated account of why that failure mode occurs. 

---
# The Capability Manifold and ML Scaling Laws 

**Authors**: Syed Ali Raza Zaidi, Maryam Hafeez  

**Link**: [PDF](https://arxiv.org/pdf/2609.27588)  

**Abstract**: Existing machine learning (ML) scaling laws relate predictive loss to compute, model parameters, and data. However, as models are increasingly deployed through agentic harnesses, loss alone is insufficient to characterize downstream performance: models with similar loss can exhibit different capabilities in reasoning, retrieval, planning, and adaptation. Yet, no unified framework connects such capabilities to the coupled resources available across the ML lifecycle. We bridge this gap by introducing a capability manifold, a multidimensional framework mapping downstream capabilities to pre-training, post-training, and test-time resources through bounded scaling functions. Analytical Jacobians quantify capability sensitivity to resource changes and interactions. As an initial application, we embed Kaplan- and Chinchilla-type scaling laws and test-time compute within the framework, demonstrating how existing scaling relationships can be unified as trajectories on a common capability manifold. 

---
# DCRL: Decoupling and Coupling Reinforcement Learning via Policy-Reward Manifold Alignment 

**Authors**: Henan Sun, Zehua Li, Haitao Hu, Qifan Zhang, Jianfeng Zhang, Nuo Chen, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27572)  

**Abstract**: Reinforcement learning (RL) has emerged as a key paradigm for improving the reasoning capabilities of large language models (LLMs). However, existing reward systems, such as rule-based and reward-model-based, often exhibit issues such as unstable optimization and reward hacking. In this work, we revisit the general reasoning of LLMs from a geometric perspective, conceptualizing it as a coupled manifold composed of three interdependent sub-manifolds: logical deduction, evaluation, and representation. Based on this perspective, response generation in RL can be interpreted as a decoupling process from the evaluation manifold, while reward estimation corresponds to a decoupling process from the logical deduction manifold. The limitations of rule-based and reward-model RL systems can be geometrically interpreted as the mismatch of policy-reward manifolds during RL process. To address the aforementioned misalignment, we propose Decoupling and Coupling Reinforcement Learning (DCRL) framework, which incorporates two key components: (1) a syllogistic logic-based prompt evolution mechanism that dynamically refines reward rubrics to enhance the expressiveness of the reward manifold; and (2) a policy-reward re-coupling mechanism that jointly updates the reward and policy models, ensuring consistent evaluation and mitigating manifold mismatch during training. Theoretical analysis and extensive experiments across multiple reasoning domains demonstrate that DCRL consistently outperforms both rule-based and reward-model baselines. Notably, a Qwen3-4B model trained under DCRL surpasses a Qwen3-32B baseline and approaches the performance of a Qwen3-235B model, highlighting superior effectiveness and generalization in RL. 

---
# FDE-Bench: Evaluating LLM Agents for Deployment Environment Configuration 

**Authors**: Weihang Ding, Junfei Zhan, Yueting Li, Qirong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.27571)  

**Abstract**: Deployment requires an agent to turn application code into a running system whose services connect, become ready, and remain observable. FDE-Bench evaluates this capability with 136 deployment-configuration tasks spanning Docker images, multi-service Compose stacks, and Kubernetes, in greenfield and diagnose-and-repair modes. Agents submit declarative artifacts that are collected, rebuilt, and redeployed in a pristine environment. Four gated binary check layers measure build, readiness, behavior, and conformance to the deployment specification, using programmatic checks without an LLM judge. A four-arm release gate requires a resolving reference solution and rejects tasks solved by do-nothing, specification-transcription, or generic-stub submissions. The released check annotations expose the link between 2,145 checks and their specifications, including seven documented gaps. Three additional adversarial strategies test shortcuts in the grading signals; none resolves any of the 135 tasks they cover, while a vacuous health probe passes readiness and exposes the need for downstream checks. On the 136-task evaluation grid, seven language models from four providers use the same four-tool scaffold and resolve 52.9-75.0 percent of tasks. The three zero-intelligence floors resolve none and reach a mean Deployment Score of at most 0.44. Readiness is the largest failure stage, accounting for 110 of 313 unresolved episodes. Mean resolution rate is 30.7 percentage points higher on the repair task group than on the disjoint greenfield group, with a positive gap for every model; ten tasks resist all seven. In a 25-task case study, one practicing engineer directing Claude-Sonnet-5 resolves 92 percent against 72 percent for the autonomous baseline. FDE-Bench links deployment success and failure to artifacts that can be inspected and replayed. 

---
# TNLearn: An Open Source Python Package for Task-based Neurons 

**Authors**: Meng Wang, Tieyun Li, Juntong Fan, Hanyu Pei, Jing-Xiao Liao, Yaodong Yang, Jianwei Ma, Fenglei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27564)  

**Abstract**: The brain does not rely on a single type of neuron to perform all kinds of tasks; instead, it designs different neurons for different tasks. The concept of task-based neurons represents a paradigm shift compared to task-based architectures. It argues that solving a specific problem requires customized neurons, as task-based neurons capture useful prior knowledge from task-related data. To facilitate the use of task-based neurons in scientific research and industrial applications, we introduce TNLearn, an open-source Python package that provides automated construction of task-based neurons and networks, enabling smooth training of task-based networks. Comprehensive documentation, including technical exposition, API reference, and representative examples, is available online. TNLearn is open-sourced at this https URL and has become a PyTorch ecosystem project. 

---
# PhyMo: A Physical-Field Modality for Multimodal AI4Physics 

**Authors**: Henan Sun, Haitao Hu, Jin Liu, Jianfeng Zhang, Lujia Pan, Nuo Chen, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27554)  

**Abstract**: Multimodal learning is emerging as a powerful paradigm for AI for Physics (AI4Physics), where predicting physical systems requires the joint interpretation of heterogeneous observations, measurements, and domain knowledge. However, existing approaches typically represent physical quantities and governing equations as generic numerical or textual tokens, overlooking the physical constraints that determine their spatiotemporal interactions. To address this limitation, we introduce the \textbf{physical-field modality} and propose \textbf{PhyMo}, a physics-grounded multimodal framework that organizes heterogeneous measurements through PDE-associated operators. PhyMo follows a three-stage learning procedure: the physical-field encoder is first pretrained through field reconstruction under PDE residual supervision, its representations are subsequently aligned with visual embeddings in a shared latent space, and the fused multimodal representations are finally processed by corresponding downstream prediction heads. Experiments on five datasets spanning diverse physical environments show that PhyMo achieves state-of-the-art performance, compared to the strongest baseline on each dataset, demonstrating the superiority of PhyMo on multimodal representation learning in AI4Physics. 

---
# Behaviora - A Conceptual Architecture for External and Internal Behavior of Robots and Agents 

**Authors**: Gote Nyman  

**Link**: [PDF](https://arxiv.org/pdf/2609.27536)  

**Abstract**: Behaviora is a preliminary conceptual architecture for representing agent and robot behavior, external and internal alike, in an addressable form. A behaving robot or agent performs a Behavior Episode composed of episode components, which can be derived from behavior taxonomies (BTax) and assigned persistent identifiers. We denote these identifiers as IoB (Internet of Behaviors) Addresses. A Behavior Episode specifies what the system does, while a Style Profile (SP) specifies how this behavior is expressed. Style can communicate characteristics of the actor and qualities such as competence and cultural manners. An Experience Profile (EP) represents behaviorally relevant internal state that modulates the execution of an Episode. Finally, a Behavior Compiler maps these behavioral representations to platform-specific actions. We use a primitive touching arm model to show these components and their relations. External Behavior is a result of addressable movements and their styles. Internal Behavior is represented through the same episodic principle and can be rendered as inner speech. Sensing, perception and complex task contexts have not been included in the present implementation, although a conceptual place is reserved for them. 

---
# NV-Reason-CT: 3D Visual Language Model for CT Analysis 

**Authors**: Andriy Myronenko, Dong Yang, Yucheng Tang, Baris Turkbey, Benjamin Simon, Stephanie Harmon, Rikhil Makwana, Mariam Aboian, Sena Azamat, Ibrahim Ethem Hamamci, Sezgin Er, Bjoern Menze, Marc Edgar, Yufan He, Pengfei Guo, Daguang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27511)  

**Abstract**: We present NV-Reason-CT, a generative vision--language model for chest and abdominal CT combining native 3D visual encoding with radiologist-guided reasoning. The model couples a native 3D vision transformer with a language model, passing all visual tokens and their explicit 3D coordinates into language decoding without further spatial token merging. This retains volumetric spatial information within the vision encoder and through the language model's positional encoding during joint processing with text.
We train on a curated corpus of approximately 550,000 multimodal instruction examples from 70,111 unique CT image inputs, combining standardized reports, abnormality-focused and anatomy-specific questions, multi-turn interactions, and radiologist-authored reasoning from recorded and transcribed expert CT interpretations. Expert annotations provide direct supervision and guide additional report-grounded synthetic reasoning. End-to-end supervised fine-tuning (SFT) is followed by Group Relative Policy Optimization (GRPO), with verifiable rewards over chest and abdominal abnormality sets.
The model supports abnormality classification, report generation, and interactive reasoning with reviewable observations, differential diagnoses, and uncertainty. Evaluation spans public CT benchmarks and a held-out NIH cohort. On CT-RATE, NV-Reason-CT achieves a macro-F1 of 0.614 and macro-AUROC of 0.871 without a task-specific classification head; generated reports achieve a report-derived macro-F1 of 0.592. In a preliminary study with expert radiologists, AI-assisted review received favorable confidence ratings and was associated with a 50% reduction in average reported interpretation and reporting time. We release the model and training code to support reproducible research on explainable AI for volumetric medical imaging. 

---
# Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models 

**Authors**: Kaifeng Tan, Yudong Li, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27510)  

**Abstract**: Modern large language models are pretrained on massive datasets, making it difficult to prevent benchmark data from entering their training sets and undermining the reliability of evaluation results. Reliable evaluation is particularly challenging for base models, whose limited instruction-following ability complicates task-based assessment. We introduce Uncheatable Eval, a dynamic benchmark that regularly collects newly published text to evaluate base language models and reduce the risk of data contamination. Drawing on the relationship between a model's predictive ability and its ability to compress data losslessly, we use compression rate to evaluate how well models predict new text. We evaluate 80 models across 14 text categories, study how compression changes with context length, and examine the correlation between compression rate and zero-shot MMLU accuracy. Our results yield three main findings: (1) compression performance follows a consistent scaling trend with model size; (2) attention-based, hybrid, and recurrent models differ in how their compression performance changes as more context becomes available; and (3) lower compression rates are strongly associated with higher zero-shot MMLU accuracy. Code is available at this https URL. 

---
# Passing: An Endless Journey through Reconstructed Spacetime with AI-Generated Sound 

**Authors**: Akira Takahashi, Chihiro Nagashima, Zhi Zhong, Shusuke Takahashi, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2609.27489)  

**Abstract**: This paper introduces Passing, an interactive audiovisual installation that generates an endless journey from a single continuous monorail-window recording by reconstructing it as a spatiotemporal volume. Rather than replaying the footage linearly, the work resamples its spatial and temporal structure along nonlinear trajectories, producing a continuously passing landscape whose depth, speed, and temporal order become unstable. A camera-based viewer-presence detection system estimates whether a viewer is present in the viewing zone and uses this presence state to influence transitions among rendered video sequences. The resulting video stream is fed into SpecMaskFoley, a real-time video-to-audio synthesis model that generates a synchronized soundscape for the reconfigured image. The model is not used to reconstruct an objectively correct soundtrack, but functions as a speculative listener, proposing a possible auditory interpretation of a world whose conventional spatial and temporal premises have been disrupted. Passing distributes creative agency across the artist, who defines the rules of spacetime reconstruction; the AI model, which interprets the emergent visual flow as sound; and the audience, whose embodied presence influences the audiovisual trajectory. Through this structure, the work investigates how authorship and listening may be negotiated among human intention, machine inference, and audience interpretation. Artwork page: this https URL 

---
# Kairos: Grounded Forecasting of Presence and Directional Flow in 4D Scene Graphs 

**Authors**: Iacopo Catalano, Julio A. Placed, Javier Civera, Jorge Peña Queralta  

**Link**: [PDF](https://arxiv.org/pdf/2609.27467)  

**Abstract**: Long-term autonomy in human-populated environments requires anticipating whether and how people will move at times a robot has not yet observed. Existing representations of pedestrian motion face a tradeoff: they either forecast future activity, reducing each location to a scalar rate, or model the full directional distribution, holding it fixed in time. We present Kairos, a predictive directional-flow memory that extends a hierarchical 3D scene graph (3DSG) to a 4D scene graph (4DSG). Every observed voxel of the reconstructed geometry stores a directional mixture and a presence rate, and spectral predictors forecast, for any future query time, both the probability that people are present and the full directional distribution of their motion. Pairwise flow dependence between adjacent voxels supports conditional queries, and per-voxel predictive variances yield calibrated credible intervals that tighten as observations accumulate. We evaluate Kairos on three real pedestrian environments: a robot-collected campus dataset, a shopping mall, and a station concourse recorded continuously for eleven months. Its learned state remains consistent under loop-closure corrections, and its forecasts are competitive with dedicated occupancy and flow models trained on the full detection stream, although Kairos learns from only the small fraction available to a patrolling robot. Finally, we validate the representation on a downstream encounter-probability planning task, where plans computed over the Kairos forecasts encounter more people than plans computed over any time-invariant map at an equal success rate. We provide the code at this https URL. 

---
# Issuer-Sovereign Agentic Payments 

**Authors**: Dishant Sharma, Rajneesh Kaushal, Ashu Kanaujia  

**Link**: [PDF](https://arxiv.org/pdf/2609.27452)  

**Abstract**: AI agents are beginning to make real payments. Current approaches let an agent pay by relying on a credential provider that, in the approaches deployed today, typically sits outside the cardholder's bank. The spending rules are then enforced by the card network or that provider, and not by the bank itself. This leaves the issuing bank, which carries the financial risk, with little direct control at the moment a payment happens. This paper describes Issuer-Sovereign Agentic Payments, a method that keeps that control with the issuer. The cardholder approves a spending rule once, and the bank's own authentication component records it. Later, when the agent pays a specific merchant, the bank checks the merchant against the approved rule and generates the card authentication value only if the merchant is allowed. The payment then travels the normal card rails and is validated by the issuer, with no extra dependency introduced at execution. 

---
# BEE: Intervention-Adaptive Real-World Reinforcement Learning with Vision-Language-Action Models 

**Authors**: Weihui Zhao, Xiaohan Yan, Zunian Wan, Xuan Du, Zhaozhan Chi, Jianbo Mao, Ruipu Wu, Rushuai Yang, Houlin Li, Shukai Yang, Jing Wu, Yuxiang Yan, Yongcheng Liu, Chuankang Li, Guanghui Ren, Wei Shan, Maoqing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2609.27450)  

**Abstract**: Vision-language-action (VLA) models handle long-horizon manipulation, yet success hinges on a few precision-critical phases where millimeter-scale errors undo all prior progress. Online reinforcement learning (RL) can optimize exactly these actions, but free exploration is far too costly on real robots, which makes human corrections indispensable. However, existing online RL methods for VLAs either cannot incorporate such corrections or fold them into undifferentiated supervision. Yet human corrections are not uniformly noisy but reliable along some action dimensions and variable along others. Building on this, we introduce BEE, an intervention-adaptive framework for real-world RL on a frozen VLA that lets the policy go BEyond Expert imitation. We formulate human corrections not as actions to reproduce but as evidence about a constraint: a Correction Model predicts how a human would correct a given VLA proposal and how consistent the correction is along each action dimension. This predicted consistency sets the per-dimension tightness of a constraint on policy optimization. Where corrections are consistent the policy stays close to the human, and where they vary, the constraint relaxes. We evaluate BEE on three real-world manipulation tasks and one LIBERO-Pro simulation task at a matched online-data budget. BEE attains the highest success rate on every task, 91.2% on average against 57.5% for RLT and 42.1% for DSRL, and the lowest human intervention rate on all real-world tasks. 

---
# Quantum Reinforcement Learning for Cost and Delay Tradeoffs in Quantum Cloud Orchestration 

**Authors**: An N. H. Phan, Dang Van Huynh, Muhammad Usman, Hoa T. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27446)  

**Abstract**: Quantum cloud computing, delivered through the quantum-as-a-service (QaaS) model, provides access to quantum computing resources. However, applying uniform time-based pricing across fundamentally heterogeneous quantum resources significantly complicates task orchestration, particularly when addressing the tradeoff between execution costs and system performance. While heuristic methods rely on predefined scheduling rules, classical deep reinforcement learning (DRL) models may require more trainable parameters in this setting. Motivated by the potential of parameterised quantum circuits (PQCs) as compact function approximators, we propose QRLQ, a cost-delay-aware quantum cloud scheduling framework integrating PQCs with a dueling double deep Q-network (D3QN) to dynamically account for both cost and delay. Our simulation results show that QRLQ achieves lower mean cost and delay than the heuristic baselines, achieving a 5-11% lower mean cost relative to availability-based and rotation-based heuristics and reducing mean delay by 17% and 82% relative to the strongest and weakest heuristic baselines, respectively, while retaining execution fidelity within 2% of a fidelity-greedy policy. Compared with the classical DRL baseline, QRLQ achieves comparable scheduling performance while using 72% fewer trainable parameters. This work explores the feasibility of using QRL for task orchestration in quantum cloud environments and demonstrates its potential for cost-delay-aware quantum resource management. 

---
# What Looks Like a Capability Limit in Vision-Language Models Is a Readout Limit 

**Authors**: Alfredo F. Frontera Del Valle  

**Link**: [PDF](https://arxiv.org/pdf/2609.27408)  

**Abstract**: Benchmarks for vision-language models offer their answer choices in some convention: a letter, a color name, a pixel coordinate. That convention is treated as neutral. We find it is not, and that the limits a benchmark reports can belong to the readout rather than to the model.
On 200 COCO photographs, Qwen3-VL-4B picks the correct one of nine locations for a named object 68.5% of the time when the locations are given in English and 20.0% when the same locations are given as pixel coordinates. Chance is 11.1%. The cost arises when the answer options are coordinates; giving the model a coordinate in the question instead costs 3.5 points and is not significant. The gap holds on a 4x4 grid, under 8-bit rather than 4-bit quantization, and in every slice by object size, boundary distance and category. It also decides which model wins. Two models that tie under English names differ by 39 points in one coordinate system and by 54 in the other, in opposite directions.
On the color task, three of the four open models capable of the task show the penalty; on photographs, two of three open models do, and so does Gemini, at 11.1 points on parseable answers (p = 1e-4). GPT-4o does not. To ask whether a model reads a coordinate at all, we attach the wrong name to each one and record which the model follows. Color options written as hue angles are followed below chance; a normalized pixel convention is followed at four times chance. This tells apart conventions a model can use from ones it cannot, though it did not predict accuracy on two untried conventions. Five models also name the same color wheel five different ways, so a fixed answer vocabulary is not neutral across models either.
Five times during this work we measured a capable model as incapable because our scorer and the model disagreed about what an answer looks like. We report each case. They are the phenomenon in miniature. 

---
# Forget who you Forgot: Speaker Unlearning to Prevent Re-Identification in Zero-Shot Text-to-Speech 

**Authors**: Hyoeun Kim, Yujun Lee, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2609.27399)  

**Abstract**: Recent zero-shot text-to-speech (ZS-TTS) systems can reproduce a speaker's voice with high fidelity from only a few seconds of reference speech, raising concerns over unauthorized voice cloning and impersonation. Speaker identity unlearning has recently emerged as an approach to selectively suppress this capability for speakers who opt out while preserving synthesis capability for other speakers. Although existing approaches reduce speaker similarity, preventing re-identification often faces severe degradation of speech quality. Motivated by this observation, we propose GUARD, a lightweight speaker identity unlearning framework that combines a learned speaker gate with speaker-agnostic activation steering on a frozen TTS backbone. The steering vectors are optimized using group-relative reward optimization to shift outputs from forget speakers toward population-level impostor similarity while preserving intelligibility and speech naturalness. On CosyVoice2, GUARD reduces forget-speaker similarity from 0.541 to 0.103 and re-identification accuracy in a 150-speaker gallery from 73.5% to 0.5%, while preserving retain-speaker reproduction. The results demonstrate that similarity reduction alone may not fully characterize successful speaker identity unlearning and highlight re-identification as a complementary criterion for its evaluation. 

---
# Automotive mmWave Spinning Radar Place Recognition with Spatially Gated Feature-Correlation Representation 

**Authors**: Saimunur Rahman, Sagun Singh Shrestha, Abdelwahed Khamis, Peyman Moghadam  

**Link**: [PDF](https://arxiv.org/pdf/2609.27394)  

**Abstract**: Automotive spinning FMCW radar provides dense, $360^\circ$ sensing and remains reliable under poor illumination and adverse weather, making it well-suited to autonomous navigation. Place recognition uses these observations to identify previously visited locations for re-localization and long-term navigation. However, heading changes appear as circular shifts in the polar radar representation, and conventional global aggregation can lose relationships among radar responses that are important for distinguishing similar places. We propose SGCA-Net, a spinning radar place recognition framework that combines rotation-robust feature extraction with Spatially Gated Correlation Aggregation (SGCA). SGCA learns spatial weights to reduce the influence of unstable and ambiguous radar regions, while aggregating pairwise correlations among local responses to preserve informative feature relationships. Experiments on the MulRan dataset show that SGCA-Net consistently outperforms SOTA methods across urban, campus, and open-road environments, while remaining robust to substantial heading variation. Evaluation on the HeRCULES dataset further demonstrates that SGCA-Net generalizes to unseen environments and radar sensors without fine-tuning. 

---
# Forecast Workflow Bench: Evaluating Language-Model Decisions with Budgeted Forecast Tools 

**Authors**: Shunya Nagashima  

**Link**: [PDF](https://arxiv.org/pdf/2609.27385)  

**Abstract**: Time-series foundation models (TSFMs) provide forecasts for operational decisions, but accuracy alone does not determine their value. Evaluating agents that use these models requires measuring decision quality and forecast cost. FWBench evaluates this capability on 1,251 electricity and cycle-hire cases using fixed forecast tools and simulated capacity contracts. Agents select models, histories and horizons, then submit capacities to minimize a stated loss-cost objective. We evaluated two hosted and eight local configurations, including small language models, and tested local models with and without TSFMs. GPT-6 Astra bought inexpensive short-horizon forecasts selectively, using 2.5% of the budget, and outperformed fixed policies when the saved decisions were scored with three loss-cost weightings. FWBench enables reproducible evaluation of how language models select and use time-series forecasts to make decisions under cost constraints. 

---
# Psychoacoustically Aligned Latent Smoothing for Adversarial Robustness of Full-Duplex Speech-to-Speech Dialogue Models 

**Authors**: Kian Shamsaie, Iman Modarressi  

**Link**: [PDF](https://arxiv.org/pdf/2609.27378)  

**Abstract**: End-to-end speech-to-speech dialogue models listen and speak simultaneously, so a continuously open acoustic channel is exposed to adversarial manipulation. We formalize imperceptible attacks on full-duplex agents as optimization over additive perturbations confined beneath the psychoacoustic masking threshold of the carrier speech, under three goals: targeted semantic hijacking, response suppression, and policy jailbreaking. Against an undefended Moshi-style agent, white-box attacks succeed in up to 91.7% of trials. We then introduce psychoacoustically aligned latent smoothing (PALS), which injects anisotropic Gaussian noise shaped by local codebook covariance at the residual-vector-quantized latent interface, with input noise shaped by the masking threshold constraining the attacker and trained by a Kullback--Leibler consistency objective. Deployed with no inference-time cost, PALS reduces hijack to 8.3%, mute to 11.2%, and jailbreak to 9.1% at clean quality within 2.3%. A Monte Carlo-smoothed variant certifies an ellipsoidal latent radius up to 0.616, a guaranteed floor that the empirical robustness far exceeds. 

---
# Planned Test-Time Scaling with Coordinated Reasoning Paths 

**Authors**: Xueqing Wu, Langxing Bai, Hritik Bansal, Po-Nien Kung, Shuo Li, Hao Liu, Nanyun Peng, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27374)  

**Abstract**: Test-time scaling with parallel branches is widely adopted to improve performance on challenging reasoning tasks. The predominant approach, repeated sampling, draws branches independently from a single policy, which can produce redundant attempts and thereby limit the gains from additional inference compute. To address this limitation, we propose Planned Test-Time Scaling (PTTS), which replaces independent sampling with a coordinated joint policy: a planner generates a solution outline for each branch, steering the branches toward distinct reasoning paths, and an executor produces a full solution conditioned on each outline. Formally, we show that PTTS strictly generalizes repeated sampling and, in a stylized setting, provably promotes coverage of complementary reasoning modes and yields better pass@k scaling. We instantiate PTTS on top of strong reasoning models, keeping them fixed as executors while replacing repeated sampling with PTTS inference to further enhance test-time scaling. Concretely, we develop two variants: PTTS-ZS prompts a model to jointly generate outlines for all branches in a single autoregressive pass, while PTTS-RL directly optimizes the planner against the pass@k reward using truncated execution rollouts for efficient training and a sharper reward signal. Across five mathematical reasoning benchmarks with Qwen3-1.7B and 4B, PTTS-ZS improves pass@64 over repeated sampling by up to 6.7 points, while PTTS-RL further increases the gain to up to 13.4 points. Further analysis indicates that broader coverage of distinct reasoning paths contributes to these gains. Overall, PTTS provides a general framework for improving test-time scaling by coordinating reasoning branches, with zero-shot and trainable instantiations that yield substantial performance gains. 

---
# Neither Silence nor Overlap Is Failure: Intent-Conditioned Evaluation of Turn-Taking in Full-Duplex Spoken Dialogue Models 

**Authors**: Kian Shamsaie, Iman Modarressi  

**Link**: [PDF](https://arxiv.org/pdf/2609.27372)  

**Abstract**: Benchmarks for full-duplex spoken dialogue models score turn-taking with binary fixed-window rules that reward immediate response or silence by completeness of the prior turn. We argue that the appropriateness of a response offset, whether delayed silence or anticipatory overlap, is conditional on the speaker's latent intent, identifiable only from that speaker's behavior. We introduce TACT, a benchmark of 9,728 episodes and 73.2 hours from five dyadic corpora; each episode carries dialogue history, a per-speaker memory profile, and an annotator-derived posterior over six intent classes. Scoring replaces binary windows with a strictly proper threshold-weighted continuous ranked probability score whose weights are intent-conditioned timing kernels fitted to human floor-transfer-offset distributions, proving boundedness, consistency, and binary reduction. Across eleven systems the best model reaches 0.47 against a human topline of 0.86, is nearly invariant to speaker profiles, and TACT agrees with human judgments at Spearman 0.81 versus 0.46 for binary metrics. 

---
# Geometry-Conditioned Visual Place Recognition in Natural Environments 

**Authors**: Walter Nedov, Saimunur Rahman, Kavindie Katuwandeniya, David Hall, Kaushik Roy, Peyman Moghadam  

**Link**: [PDF](https://arxiv.org/pdf/2609.27370)  

**Abstract**: Visual Place Recognition (VPR) in natural environments remains challenging due to repetitive vegetation, sparse distinctive landmarks, and substantial appearance and viewpoint variation across traversals. While visual observations of the same place can change considerably, their underlying spatial structure is often more persistent. We exploit this complementary geometric consistency through Depth-Aware Distillation (DAD), which conditions the token representations of a pretrained Vision Foundation Model (VFM) on geometry inferred by a Geometric Foundation Model (GFM), without any depth sensor. Rather than treating geometry as an additional input modality, DAD projects image-aligned depth into the VFM token space and selectively modulates visual representations through channel-wise geometric conditioning. A two-stage teacher-guided learning strategy first anchors the geometry-conditioned representation to the pretrained appearance space, before refining it for place discrimination. Evaluated on the WildCross benchmark, DAD improves average inter-sequence Recall@1 from 61.41% to 66.37% and Recall@5 from 65.86% to 72.49% over a matched appearance-only baseline, with the largest gains under reverse traversal and long-term appearance variation. These results show that GFM-derived geometry can provide a persistent structural prior for VPR when visual appearance becomes unreliable. 

---
# Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes Interaction 

**Authors**: Jialu Wang, Jianing Deng, Shuqing Luo, Yuanzhe Li, Dongwei Wang, Jingtong Hu, Huanrui Yang, Song Wang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27355)  

**Abstract**: Unlearning ensures LLM compliance by removing the influence of private or copyrighted training data. However, since LLM models typically undergo post-training compression, like quantization, in practical deployment, it has been observed that the unlearning effect can be substantially weakened, with the forgetting behavior degrading more severely than that of model utility. This paper proposes a quantization-robust unlearning framework that makes forgetting robust to quantization while maintaining overall model utility. We analyze this gap through the lens of loss landscape. Specifically, our analysis reveals a curvature-based criteria that pinpoints sensitive weights in the unlearned model that leads to both non-robust forgetting and reduced utility. We therefore propose sensitivity-guided noisy regularization, which is applied on the sensitive parameters to steer the model convergence towards a smoother minima of uniformly low forget and retain losses. Balancing unlearning and utility, we further propose forget-critical optimization, which updates only forget-critical layers, preserving most of the network to retain useful knowledge. Extensive experiments on the MUSE and TOFU benchmarks across multiple LLM unlearning algorithms show that our approach achieves substantially more quantization-resilient forgetting while maintaining utility. 

---
# Constraint-Driven Context Engineering: Designing Domain Interfaces for AI Systems 

**Authors**: Xiwei Xu, Chen Wang, Mengmeng Yang, Yipeng Zhang, Jacky Jiang, Suyu Ma, Youyang Qu, Ming Ding, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27354)  

**Abstract**: Generative AI systems are increasingly deployed to address domain problems. These systems operate under technical, regulatory, institutional, and normative constraints that define acceptable AI behaviour and outcomes within their domains. We observe a recurring pattern in our industry engagement: partners often arrive with a functioning but relatively generic AI solution. The challenge is no longer to build an AI system from scratch, but to improve the quality and domain appropriateness of an AI-generated solution. In these settings, the limiting factor is often the quality, scope, and structure of the context available to the system. Yet, existing context engineering approaches primarily focus on supplying domain knowledge through retrieval, memory, and tools, with limited support for systematically identifying and operationalising the constraints that govern AI systems in their operational environments.
This paper proposes Constraint-Driven Context Engineering (CDCE), a design approach for engineering domain interfaces for AI systems. Drawing on software architecture design and Domain-Driven Design (DDD), CDCE treats domain constraints as first-class design drivers. It identifies and characterises constraints, determines the required context assets, and designs representations through which these assets are made available to AI systems.
We conducted a comparative multiple-case study with industry and public-sector partners across educational assessment, healthcare decision support, and financial-distress prediction. Depending on their characteristics, constraints can guide AI behaviour, enforce permissible boundaries, or support verification of AI-generated outcomes. The cases demonstrate CDCE's applicability across contrasting domains and show how constraint characteristics shape the resulting domain interfaces. 

---
# Evolving Inspectable O-RAN Slicing xApps with LLMs 

**Authors**: Faezeh Dehghan Tarzjani, Bhaskar Krishnamachari  

**Link**: [PDF](https://arxiv.org/pdf/2609.27337)  

**Abstract**: Open RAN (O-RAN) slicing xApps must adapt resource allocations to changing channel conditions and traffic demands while meeting service-level agreements (SLAs). Deep reinforcement learning can produce adaptive policies, but their allocation rules remain encoded in neural-network parameters. Our goal is to retain this adaptability while making the controller's decision logic directly inspectable and editable by operators. We use a large language model (LLM) to evolve slicing controllers as compact Python programs whose decision logic remains readable and editable after optimization. The LLM proposes and revises candidates offline, while a calibrated simulator scores them, and the selected decision module runs unchanged in the O-RAN control path. On the NSF POWDER 5G testbed, the evolved controller releases resources from a guaranteed slice whose throughput target becomes unattainable under a sustained channel fade, improving best-effort throughput from 158.2 to 228.6 Mbps, a 44.5% gain over the best static allocation. Since the controllers are readable source code, their behavior can be predicted from their equations, defects can be diagnosed by reading the code, and calibration errors can be corrected with one-line edits, reducing SLA misses from 79.9% to 2.2% in one case and more than doubling fitness in another. In a four-slice trace-driven simulation calibrated to the same testbed, evolutionary search achieves higher average evaluation scores than independent prompting at a matched proposal budget, with mean normalized gains on held-out traces of 16.3% for prompting alone, 32.1% for evolution from scratch, and 51.0% for evolution from a starting program. 

---
# Breaking Weather-Content Coupling: Type-Severity Guided Progressive Disentanglement for All-in-One Infrared Restoration 

**Authors**: Xinyao Wang, Lijun He, Zhihan Ren, Fan Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27317)  

**Abstract**: Infrared (IR) imaging is crucial for autonomous driving, remote sensing, and other perception tasks. However, adverse weather may introduce fake structural responses that are entangled with real thermal structures. Existing IR restoration methods are typically designed for a single degradation type or directly reconstruct from degradation-entangled representations. Consequently, they struggle to distinguish intrinsic thermal structures from weather-induced fake responses and to accommodate spatially varying degradation severity, leading to artifacts or the over-suppression of weak but meaningful thermal responses. To address these issues, we propose TSGPD-IR, a type-severity guided progressive disentanglement network for all-in-one infrared restoration that factorizes restoration guidance into task-level weather semantics and region-level degradation severity. Specifically, a Weather and Semantic Co-Guided Multi-Level Prompt Generation Module combines global weather semantics with stage-wise local features to generate adaptive prompts that progressively suppress degradation-induced responses while preserving intrinsic thermal structures. To complement global weather semantics with spatial restoration control, a Proxy-Supervised Regional Degradation Estimator derives severity supervision without manual annotations and predicts spatially varying degradation priors. Guided by these cues, a Multi-Source Collaborative Expert Selection Strategy uses a shared branch to preserve weather-invariant thermal structures and hierarchical routing to select weather-specific expert pools and severity-compatible regional experts. This design progressively separates degradation interference from genuine thermal content and enables region-adaptive restoration, reducing both residual artifacts and over-suppression. 

---
# Turning Safety into Competence: Minimally Exploitable Robot Policies via Safety-Filtered Reinforcement Learning 

**Authors**: Ruihan Wu, Rui Yang, Donggeon David Oh, Duy Nguyen, Haimin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27312)  

**Abstract**: Robots deployed for competitive tasks must outmaneuver their opponents without sacrificing safety. Existing approaches, including safe reinforcement learning (RL), train a single policy to achieve task success and avoid failures simultaneously. This coupling can complicate training and leave the learned policy exploitable by deliberate attacks. We propose Safety to Competence (S2C), a two-stage RL framework that separates safety synthesis from competitive task learning. We formulate competitive interactions as safety-critical Markov games and prove that perfect filtering preserves policy non-exploitability when all players commit to safe maneuvers. S2C learns a robust safety filter via adversarial RL, embeds it in the environment during task policy training, and retains the same filter at deployment. In simulated touchdown games, S2C outperforms eight safe RL baselines, achieving the highest win rate and Elo rating, and the lowest exploitability. Hardware stress tests against a human opponent confirm S2C's competence. 

---
# Multi-View Fusion for Encrypted C2 Detection: A Leakage-Controlled Measurement Study of Evaluation Pitfalls 

**Authors**: Hoang-Huy Nguyen-Huu, Van-Tri Phan, Khuong Nguyen-An  

**Link**: [PDF](https://arxiv.org/pdf/2609.27311)  

**Abstract**: Command-and-control (C2) traffic increasingly hides within TLS, so defenders now apply machine learning to traffic metadata. Many studies assume that combining two metadata views, namely flow statistics and TLS handshake fingerprints, improves both accuracy and robustness. We tested this assumption on 17,577 TLS flows from 62 real Cobalt Strike captures. Our evaluation removes the data leakage that leads to overly optimistic reported scores. We report three findings that matter more than the fusion result itself. First, an incorrect preprocessing step increases the F1 score by 0.28. This step computes the frequency encoding across the entire dataset rather than within each cross-validation fold. The increase is about ten times larger than any real effect we measured. Second, both the labels and the behavioral features depend on the destination address. Because of this, the 17,577 flows form only 2,132 independent groups, and the positive rate of 55.1\%, which looks balanced, drops to 4.2\%. Therefore, class balance is just a result of how we analyze the data, specifically whether we count flows or endpoints, and not a real feature of the task. Third, 20 of the 62 captures (32\%) have no TLS flows to any known C2 address, so they contain only benign samples. We checked these captures directly and confirmed that this is a gap in the ground truth, not a labeling error. In this context, fusion beats the best single view by only 0.022 in F1. When an attacker forges both feature surfaces simultaneously, every model performs worse than a simple baseline that always predicts positive (F1 = 0.711). For encrypted C2 detection, the evaluation design is not a preliminary step. It \emph{is} the main result. 

---
# Beyond the Illusion of Power: Calibrating Quasi-Experiments in Observational IS 

**Authors**: Spandan Ghose Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2609.27299)  

**Abstract**: Information systems (IS) researchers increasingly use quasi-experimental methods such as difference-in-differences (DiD) and instrumental variables (IV) to recover causal effects from observational panel data. Power calculations that justify these designs assume i.i.d. errors, but the deeper problem is what even a cluster-robust calculator cannot see. We report a Monte Carlo study over 9837 parameter conditions (approx 9.8 million datasets) and decompose the planned-versus-achieved power gap. The serial-correlation component is recoverable by an AR(1)-aware calculator when rho is known, and partially when rho must be estimated from short pre-periods, but panel attrition, staggered-adoption bias, and parallel-trends pretesting are captured by no closed-form formula; exogenous attrition alone costs approx 8 to 11 percentage points at the few-hundred-to-thousand sample sizes IS studies use. Treatment-correlated, outcome-dependent attrition instead induces bias, not just power loss. For IV, holding first-stage F fixed, larger N neither raises power nor curbs exclusion bias, though with a fixed instrument more data does sharpen the first stage, so identification rests on instrument strength, not sample size. 

---
# Teach-to-Crash: A Closed-Loop Student-Teacher LLM Framework for Collision-Inducing Test Scenario Generation 

**Authors**: Zaid Ghazal, Khouloud Gaaloul, Bruce Maxim  

**Link**: [PDF](https://arxiv.org/pdf/2609.27296)  

**Abstract**: Validating Autonomous Driving Systems (ADS) in simulation requires testing architectures that can discover rare, safety-critical failures while generating scenarios that are executable, diverse, and useful for downstream failure analysis. We introduce Teach-to-Crash, a closed-loop testing framework that combines a constrained ego-centric scenario representation, stagnation-aware search control, and a dual-LLM architecture for adaptive failure discovery. A high-reasoning Teacher LLM acts as an adaptive search controller, while a low-reasoning Student LLM emits simulator-executable scenarios in a strict JSON schema. The Teacher intervenes only when rolling collision rate and time-to-collision metrics stagnate, providing strategic guidance to redirect the search. In a CARLA case study with two experimental setups that vary the ego vehicle's speed policy, Teach-to-Crash achieves the highest Collision Hit Rate (90.79%), the shortest mean Time-to-Collision (18.31 s), and a competitive Collision Discovery Rate (136.21). PAFOT attains a higher mean CDR (179.44), but with substantially larger variance. Teach-to-Crash also yields the highest diversity (0.547) and, averaged across both setups on the CARLA Traffic Manager controller, the highest avoidability-based usefulness proxy (60.04%) among the compared methods. These results, within the evaluated CARLA scope, provide evidence that closed-loop dual-LLM reasoning can steer adversarial simulation-based testing over a constrained executable program space, generating failures that are frequent, structurally diverse, and assessed as more frequently avoidable. 

---
# KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling 

**Authors**: Zhiheng Hu, Yixun Wei, Jian Zhou, Yizhuang Zhou, Ji Li, Xing Chen, Yang Li, Bojun Wang, Yibo Zhu, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27294)  

**Abstract**: Scaling a language model is not only a question of final quality: the architectural choice determines how much computation is spent during training, prompt processing, and autoregressive decoding to achieve certain model quality. An ideal model architecture should lower all above computation costs to facilitate scaling to a larger model, while ensure the larger model indeed outperforms smaller baselines. We introduce KV-Invariant Transformer Expansion (KITE), a scaling paradigm that achieves this goal. It trains the model from a smaller size to a larger size (i.e., saving training costs via upcycling), while places newly added parameters in regions that do not affect attention KV. Consequently, during inference, prefilling KV only relies on the smaller part of the model, so the inference costs are saved. As a concrete instantiation, we present Step Scale Transformer (SST), a two-tower decoder in which one tower produces KV and the other reads them. At comparable cumulative training compute, SST, a 67B MoE model with 2.15B active body parameters per decode token, achieves lower training loss than 47B and 63B MoE Transformers with 1.48B and 2.02B active body parameters, respectively, while reducing estimated inference cost by 6.7% and 31.6%. 

---
# Ruby-ASR: Evidence-Preserving Supervision for Joint Orthographic and Lexical-Reading Recognition 

**Authors**: Hao Shi, Yun Liu, Xuehao Yang, Jun Liu, Chuanbo Hua, Xuanjun Chen, Lianbo Liu, Shiao Zhu, Zixiong Su  

**Link**: [PDF](https://arxiv.org/pdf/2609.27289)  

**Abstract**: Conventional Japanese automatic speech recognition (ASR) is supervised by an orthographic transcript, although the same written form can correspond to different lexical readings realized in speech. Such utterances receive an identical target, so their reading distinction is absent from the supervision interface and cannot be recovered reliably by post-hoc text-only grapheme-to-phoneme conversion. We present Ruby-ASR, which refines the conventional target into a span-bound orthographic--lexical-reading sequence. Unlike separate full-sentence orthographic and phonological outputs, the ruby representation locally binds each written span to its realized reading and permits deterministic recovery of both views. We instantiate the target under subtitle-style and verbatim-style transcription conventions using a Qwen3-ASR backbone; a mora-level CTC objective provides auxiliary monotonic reading supervision. The experimental results across five Japanese benchmarks show that refining the recognition target can improve lexical-reading recovery without sacrificing readable orthographic transcription. We release the checkpoints and inference code. 

---
# The Risk-Sensitive Schrödinger Bridge: Is Not a KL Projection 

**Authors**: Hamidreza Behjoo  

**Link**: [PDF](https://arxiv.org/pdf/2609.27250)  

**Abstract**: The Schrödinger bridge owes its computational power to a single structural fact: by Girsanov's theorem the controlled problem is a Kullback--Leibler (KL) projection onto a fixed reference measure, solvable by alternating projections. This letter shows that the fact does not survive risk sensitivity. When the expected path cost is replaced by the entropic risk measure and both endpoint marginals are kept as hard constraints, the resulting fixed-point bridge value $J_\theta$ (the soft-problem value at the multiplier that enforces the terminal constraint) admits no representation as a constrained KL minimum against any fixed path-space reference with a regular endpoint law (a class strictly larger than the uniformly elliptic diffusion references: no Markov property is required), even allowing an additive normalisation depending on the initial marginal. Moreover, no single reference generates the one-parameter family in the risk parameter. The obstruction is computed in closed form: the Gaussian bridge value violates, by exactly $\theta/2$, a heat equation that any Gaussian smoothing of a fixed endpoint density must obey. In place of the projection, the theory rests on a terminal-multiplier fixed point and an asymmetric factorisation penalising the score energy of the backward factor. 

---
# Listening and Mirroring: The Effects of Verbal Attunement and Behavioral Mimicry on Social and Empathic Perceptions of Embodied AI Agents in VR 

**Authors**: Nathalia Gomez, Haig Shamlian, Omar Khan, Tiffany D. Do  

**Link**: [PDF](https://arxiv.org/pdf/2609.27246)  

**Abstract**: As embodied agents take on increasingly social and relational roles in VR, visual realism and embodiment alone may be insufficient; users must also perceive these agents as emotionally attuned, supportive, and humanlike. Prior work suggests that verbal attunement and nonverbal mimicry can each improve users' social evaluations of embodied agents. However, behavioral mimicry has largely been studied outside of real-time, conversational AI interactions, leaving limited understanding of how users respond when an agent simultaneously generates contextually responsive dialogue and adapts its nonverbal behavior during an immersive conversation. To address this gap, we developed an embodied AI counselor that combines conversational AI with real-time facial-expression and posture mimicry, while producing either verbally attuned or neutral responses. We evaluated the system in a 2 X 2 within-subjects study with 20 participants, manipulating verbal attunement and behavioral mimicry. Results showed that verbal attunement was the most reliable driver of perceived empathy. Behavioral mimicry showed a marginal relationship with perceived humanness, while greater mimicry exposure showed preliminary, exploratory positive associations with empathy, positivity, and humanness, particularly among female participants. Together, these findings show that multimodal synchrony is not a simple additive strategy for designing empathic conversational agents in VR and underscore the need to consider how verbal and nonverbal behaviors are combined during real-time interaction. 

---
# Combining LLMs and Genetic Search for ARC-AGI-2 

**Authors**: Val Dyachenko  

**Link**: [PDF](https://arxiv.org/pdf/2609.27242)  

**Abstract**: LLMs can generate programs for ARC-AGI-2 tasks, but the provided compute only allows a small number of attempts to generate, debug and validate solutions. Genetic algorithms can search and test many more programs, but random search rarely starts in a useful neighborhood of the solution space. We combine the two methods through a compact domain specific language (DSL). First, a quantized Qwen3.5-4B LLM generates an initial set of programs for each ARCAGI-2 task. Then, we use those programs to seed an initial population of starting programs, and use genetic algorithms to evolve these programs towards a solution to the given task. The DSL is designed such that every mutated program remains valid and can be executed. The initial programs proposed by the LLM solve 2 (3.3%) of the first 60 tasks of the ARC-2 public evaluation set. The genetic algorithm solves an additional 4, giving 6 correct test outputs in total (10.0%). If we try using evolving solutions without this LLM seeding, we do not arrive at any solutions at all. The results show that genetic search can improve programs generated by LLMs and produce additional correct solutions. 

---
# Meet, Compare, or Abstain: LatWeave for Deterministic Multi-Hop Question Answering on Knowledge Lattices 

**Authors**: Yuze Ren, Shaoheng Fan, Tao Wang, Yabo Yan, Han Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27225)  

**Abstract**: Probabilistic question-answering systems -- whether large language models (LLMs) themselves, retrieval-augmented generation (RAG), or trained multi-hop retrievers -- conflate "what is known" and "how to reason" into a single probabilistic computation: hallucination cannot be eradicated, evidence chains cannot be audited, and the system answers even when it does not know. We present LatWeave, which organizes knowledge into a multidimensional knowledge lattice and compiles multi-hop QA into three deterministic operators -- meet (constraint intersection), compare (lattice-order comparison), and abstain (structural abstention); LLMs appear only on the construction side (one-shot extraction) and the query-planning side, while the answer-generation path is zero-LLM, zero-task-training, and auditable end to end -- so that question answering over Web-published knowledge becomes reproducible item by item. Rather than claiming across-the-board SOTA, we characterize the operating envelope of this paradigm on six public benchmarks: when knowledge is complete (MetaQA, 39,093 questions) meet chains are near-lossless over three hops (any-hit 0.9975, on par with fully supervised KBQA); on templated multi-hop home ground (2WikiMultihopQA held-out n=1,258) EM 0.865, well above published structure-augmented RAG reproductions; on open-text deep composition (MuSiQue) and extraction-coverage gaps (HotpotQA) we report degradation honestly and attribute it to causes outside the lattice-algebra layer; and when information is incomplete (IIRC) we achieve structural abstention with abstain accuracy 0.971 and leak rate 0.029. Within the operating envelope, deterministic execution pays no performance penalty, and every step on the answer path can be recomputed -- precisely the source of end-to-end auditability. 

---
# Learning Spectral Allocation: A Fractional Diffusion Framework for Adaptive Volumetric Segmentation 

**Authors**: Yi-Hui Shen, Tie-Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27217)  

**Abstract**: We address adaptive computation in 3D medical image segmentation: instead of designing another backbone, we ask how much spectral mixing each network stage needs and let optimization answer. We derive FHEAT, a two-parameter operator family, from the discrete cosine transform (DCT) solution of a fractional heat equation. A fractional order alpha and a diffusion strength D govern the operator, and at D=0 it is exactly the identity. Reparametrized by the semigroup time tau = D*alpha, same-resolution instances compose exactly, so any distribution of diffusion across same-resolution stages amounts to a single Sobolev-type regularizer of learned strength. This identity limit lets the optimizer of each layer, not the designer, decide whether global mixing is needed and how sharp it should be. We instantiate FHEAT in a lightweight U-shaped architecture (Light-UNETR) paired with a Kolmogorov-Arnold mixer (KAN3D) with adaptive rational activations, yielding FHEAT-Seg. At 5% to 20% label rates on three public benchmarks, training produces gradient-driven spectral sparsification: seven of the eight stage-level operators drive D to zero, and the survivor saturates at the sharpest low-pass (alpha ~ 0.9) in the decoder layer feeding the semi-supervised attention map. The retired layers become exact identity shortcuts at inference, cutting FLOPs from 4.29G to 0.90G (a 79% drop) at 0.975M parameters. Under a standard semi-supervised protocol, FHEAT-Seg reaches Dice scores of 90.47% (left atrium), 78.79% (Pancreas-CT), and 81.90% (BraTS 2019), ahead of five semi-supervised methods and the Light-UNETR baseline. The large variant also surpasses Light-UNETR-L under full supervision (Dice 93.09%, 85.11%, and 87.19%) with 2.851M parameters and 55.75G FLOPs. These results suggest that the allocation of spectral computation is a learnable property of optimization dynamics, not a manual design commitment. 

---
# KATOsuper: Surrogate-accelerated neural topology optimization with sensitivity-consistent Fourier neural operators 

**Authors**: Shengyu Yan, Jasmin Jelovica  

**Link**: [PDF](https://arxiv.org/pdf/2609.27216)  

**Abstract**: Topology optimization (TO) remains computationally intensive due to repeated finite element analysis (FEA) evaluations required at each iteration. While neural network-based surrogates offer potential acceleration, existing approaches often suffer from gradient inconsistency between predicted objectives and sensitivities, leading to optimization instability. This work presents KATOsuper, an objective-agnostic framework that couples neural-reparameterized topology optimization with a Sensitivity-Consistent Fourier Neural Operator (SC-FNO). The framework employs the forward_split architecture, which derives deployed sensitivities via automatic differentiation through the predicted objective field and thereby preserves consistency between the predicted objective and the gradient used for optimization. The case studies include three 2D benchmark problems and three 3D structures considering compliance or stress minimization. A physics-informed multi-channel input encoding with Fourier position embedding enables resolution-invariant learning, supporting zero-shot extrapolation beyond the training resolution, with useful performance at moderate scaling factors and topology-preserving exploration at up to 64x without retraining. The framework extends to 3D through KATO3D, featuring novel KANConv3D blocks with learnable B-spline activations. KATOsuper demonstrates 15--110x deployment-time speedup over MATLAB baselines while maintaining competitive optimality, with the clearest gains observed in complex 3D and stress-optimization cases. The insight that sensitivity direction matters more than magnitude enables robust optimization even with approximate physics evaluation, extensible to other differentiable physics-driven design objectives. 

---
# Scalable Subgraph Sampling via Resistance Curvature 

**Authors**: Chaoqun Fei, Tinglve Zhou, Tianyong Hao, Yangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27209)  

**Abstract**: Subgraph sampling reduces the training cost of large-scale graph neural networks, but sampling criteria may overlook the geometric roles of edges. We propose a resistance-curvature-guided sampling framework built on ERC-LG, a curvature approximation method for large-scale graphs. ERC-LG combines Johnson-Lindenstrauss projections with regularized multi-GPU batched conjugate gradient solvers, avoiding explicit Laplacian pseudoinverse computation and full embedding storage. The resulting curvature informs node- and edge-sampling probabilities for constructing GNN training subgraphs. Experiments show numerical agreement with pseudoinverse-based curvature and reduced runtime compared with CG-only computation. ERC-LG-based sampling variants achieve the highest mean accuracy on six of seven real-world datasets in downstream node classification. 

---
# Phonemizing User-Generated Text: A Benchmark, Taxonomy, and Compositional Approach 

**Authors**: MinJu Jeon, Younghan Park, Han Sung Park, Jong-Hwan Kim, Dong-Jin Kim, Hoyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.27205)  

**Abstract**: Text-to-speech systems increasingly process user-generated text (UGT) such as ppl and imo, whose pronunciation must be inferred from the canonical rather than surface form. We introduce UGTPhon, the first grapheme-to-phoneme (G2P) benchmark for UGT in English, Vietnamese, and Korean, together with an inference-grounded taxonomy for fine-grained diagnosis. Existing G2P models and frontier LLMs exhibit a systematic canonical-to-non-canonical performance gap, reaching up to 66.8 PER points. As a benchmark baseline, we propose a simple compositional G2P approach that incorporates canonical-form evidence through exact-match lookup and staged decoding. Across matched ByT5 and Qwen2.5-0.5B backbones, explicit canonical-form modeling consistently reduces non-canonical G2P errors. The 0.5B variant also performs competitively with much larger few-shot frontier LLMs, highlighting the benefit of explicitly modeling canonical-form inference for UGT phonemization. 

---
# Self-Evolving Multimedia Verification through Memory Consolidation of Contestation Experiences 

**Authors**: Truong Thanh Hung Nguyen, Vo Thanh Khang Nguyen, Hoang-Loc Cao, Phuc Ho, Truong Thinh Nguyen, Van Pham, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.27175)  

**Abstract**: Multimedia verification requires not only accurate decisions but also traceable evidence, reliable human correction, and safe reuse of prior experience. Existing systems often lack explicit mechanisms for revising intermediate reasoning or preventing harmful knowledge transfer. We present SEMV (Self-Evolving Multimedia Verification), a self-evolving multi-agent framework that treats provenance-bearing arguments as the interface between evidence, reasoning, human contestation, and memory. SEMV combines arena-based quantitative bipolar argumentation (A-QBAF), causal and scoped revision, and verification-gated memory consolidation with explicit conflict retention. On COSMOS benchmark, SEMV achieves 91.88% accuracy versus 89.10% for the strongest comparable baseline. Verified memory reduces negative transfer from 5.7% to 0.2%. On CTR benchmark, constructed from reviewer contestations, scoped causal revision corrects 96.7% of initial errors while saving 52.8% compute. MV2026 Grand Challenge dataset further supports evidence-grounded, temporally consistent reporting. These results show that SEMV can evolve through verified experience while keeping accumulated knowledge and subsequent decisions traceable, revisable, and contestable. 

---
# Count Evidence, Not Sentences: Tempered Evidence Fusion of LLM Judgments for Long-Text Value Measurement 

**Authors**: Yuhe Wu, Rui Qian, Guangyu Wang, Yuran Chen, Yuanchao Zhu, Junjie Yang, Zhengheng Li, Jiulin Cai, Tianyi Zhang, Zihan Dong, Jiaxin Liu, Yujie Chen, Guang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27165)  

**Abstract**: Large language models (LLMs) are increasingly used to measure public value orientations from long social media posts, yet such posts often mix background, quotations, concessions, and only a few stance-bearing sentences. Existing approaches either ask the model to predict a document-level label directly, which can be overconfident, or aggregate sentence-level predictions by majority or soft voting, which treat uncertain and decisive sentences as equally informative. We formulate long-text value measurement as a decision-fusion problem and propose Tempered Evidence Fusion (TEF), a training-free rule that weights each sentence's log-odds by its normalized information gain, as derived from a generalized Bayesian posterior. This makes the fused score nearly vanish for uncertain sentences while preserving the Bayes-optimal weight of decisive evidence. We further introduce Multi-event Insight Network Dimensions (MIND), a benchmark of 8,358 Chinese and English posts spanning five years of public events and six value dimensions. On MIND, TEF outperforms the strongest baseline among Direct, Majority Vote, and Soft Vote by an average of 4.5 accuracy points and 4.6 macro-F1 points across five LLMs and two languages. MIND dataset and code are available at this https URL. 

---
# The Linear Representation Hypothesis Needs a Group Action 

**Authors**: Louie Hong Yao, Yuhao Li, Shengchao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27158)  

**Abstract**: To make claims about representations that generalize beyond a particular trained model, we need to specify when two representations should count as equivalent. The Linear Representation Hypothesis is often discussed without making this equivalence explicit. Different notions of equivalence preserve different structures, so metrics, probes, and interventions that appear to study the same representation may in fact correspond to different hypotheses. We therefore argue that the Linear Representation Hypothesis is not one hypothesis but a family of claims distinguished by representation equivalence. We formalize this idea using group actions, specifying the representation object, the procedure that produces it, and the property ultimately asserted, while accounting for equivalences imposed by the model architecture. This framework clarifies how assumptions can change across metrics, reading points, and analysis stages, and we use it to audit common representation quantities and recent interpretability analyses. 

---
# The Like Trap: Multi-Stage Poisoning against Agents in Similarity-based Recommendation Systems 

**Authors**: Yue Xing, Pengfei He, Zitao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27155)  

**Abstract**: With recent advancements in large language models (LLMs) and LLM-based agents, these agents are becoming increasingly autonomous and gaining broader access to act on users' behalf on the internet. However, the vulnerability of automated agents deployed on social media platforms (e.g., for managing a user's personal account) remains underexplored. Existing studies on agent poisoning typically assume that the adversary can expose poisoned content to the agent. Although such an attack is direct and effective, it is more easily detected and mitigated. In the context of social media platforms, this leaves open whether the recommendation system itself would surface such content to the agent in a more subtle manner. Through theoretical analysis, we show that the like-score mechanism used in OASIS can be exploited, and we characterize the conditions under which a multi-stage chain of poisoned posts can steer the agent's feed. Based on these insights, we further develop an algorithm that crafts realistic poisoned posts. Experiments support our theoretical findings and demonstrate the effectiveness of the proposed algorithm. Notably, by exploiting the like-score feedback loop, the attack causes the recommendation system to select poisoned posts even when their user-post similarity falls below the retrieval threshold. 

---
# A Hierarchy-Aware Video-Language Model Evaluation and Hyperbolic Baseline for Surgery 

**Authors**: Ana Manzano Rodríguez, Pascal Mettes, Marlies P. Schijven, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2609.27139)  

**Abstract**: Surgical procedures follow a phase-to-step hierarchy, yet the video-language models used to recognize them are evaluated with flat per-level metrics that ignore cross-level coherence and error structure. In this paper we make two contributions to address this problem, (i) we introduce SurgHiBench, the first hierarchy-aware evaluation suite for surgical video understanding, with three tasks measuring recognition, consistency, and severity across granularity levels. We evaluate a general-purpose CLIP model, a Euclidean surgical model, and, as second contribution: (ii) HyperSurg, a new hyperbolic model that enforces phase-step containment via entailment cones, across four (existing) datasets spanning three procedure types. The suite reveals that two models with the same accuracy can produce predictions of very different error severity, ranging from sibling confusions within the correct phase to unrelated cross-phase predictions. Hyperbolic geometry shifts predictions toward the correct procedural neighborhood, and these gains scale with the tree-likeness of each dataset's annotation hierarchy, providing a principled indicator when hierarchy-aware geometry helps. 

---
# When Clients Are Orchestrated: Strategic Gradient Manipulation to Defeat Federated Learning Servers with Efficient Defense 

**Authors**: Mohamed Shaaban, Ahmed Abdelnaby, Mohamed Elmahallawy  

**Link**: [PDF](https://arxiv.org/pdf/2609.27124)  

**Abstract**: Federated Learning enables decentralized model training by exchanging model updates--rather than raw data--with a central parameter server (PS). While most of the existing defenses primarily assume static or independently acting adversaries, we reveal a new class of dynamically adaptive attacks that systematically bypass such protections. We propose Fed-ADR, a holistic attack framework in which a malicious orchestrator server (OS) dynamically coordinates a heterogeneous set of adversarial clients, including both targeted and untargeted attackers. Through real-time coordination by the OS, malicious clients strategically adapt their gradient updates to evade defenses deployed by the PS, while either severely degrading global model performance or steering training toward adversarial this http URL mitigate this threat, we offer a detection mechanism that estimates each client's true gradient from historical updates, enabling real-time detection of coordinated malicious behavior without additional overhead. We further introduce an in-situ recovery mechanism that restores global model performance without restarting training, preserving convergence and minimizing recovery time. Comprehensive experiments on MNIST, Fashion-MNIST, and CIFAR-10 benchmark datasets demonstrate that Fed-ADR's attack scheme can reduce global accuracy from over 90% to below 10%, bypassing several state-of-the-art defenses. When our detection and recovery modules are employed, they identify malicious clients and restore accuracy to over 90% within a few rounds, at a substantially lower cost than retraining from scratch--achieving a reduction of at least 20x in computational overhead. 

---
# Intelligence Across Embodiments 

**Authors**: Bo Ai, Henrik I. Christensen, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2609.27095)  

**Abstract**: Robotic embodiment encompasses the sensing, kinematics, dynamics, geometry, actuation, and control through which an agent physically interacts with the world. These properties vary across robots and change over time. We argue that general embodied intelligence requires learning that accumulates across these differences. Prevailing methods that engineer correspondences to bridge embodiment differences offer immediate practical gains, but their assumptions limit the scope of transfer in the long run. Instead, a more general approach should discover representations that support transfer to a larger range of embodiments as experience grows. We propose embodiment diversity as a promising axis of scaling, and identify broad learned priors as a complementary ingredient. We call for evaluations that better characterize embodiment gaps and transfer performance. More broadly, cross-embodiment learning connects the practical challenge of learning from heterogeneous robot experience with a broader scientific pursuit inspired by nature - physical intelligence that adapts and co-evolves with its embodiments to gain agency over its behavior and physical forms. 

---
# Local Evidence and Geometric Readout Repair in Trained GNNs 

**Authors**: Nadi Tomeh, Hugo Attali  

**Link**: [PDF](https://arxiv.org/pdf/2609.27092)  

**Abstract**: Many node-classification GNNs apply a linear classifier to a nonnegative mixture of local messages. An error can reflect either poor mixture weights or a reachable logit set poorly positioned for the classifier. We separate these causes with an exact-mass linear program and two learned post-hoc repairs. Every reweighted prediction has an equivalent centered logit translation, but only translations in a message-induced displacement set are realizable by reweighting. Across eight datasets, eight GNN backbones, and ten splits, mean accuracy rises from 62.6% for the frozen models to 63.8% with reweighting and 65.3% with set-conditioned translation. A parameter-matched node-only translator reaches 64.6%, showing that translation explains most of the gain while the message set supplies a smaller additional benefit. Although oracle reweighting can correct many errors, label-free reweighting captures little of this potential: local evidence is often present but hard to select, and relaxing the evidence constraint is more effective than learning within it. 

---
# Crossflow: Prefill-Decode Elasticity for Agentic LLM Serving 

**Authors**: Yi Xu, Ehsan K. Ardestani, Wenyin Fu, Martin Schatz, Krishna Malladi, Zhan Shu, Adnan Aziz, Shobhit Kanaujia, Ajit Mathews, Chunqiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27085)  

**Abstract**: As serving capacity demand surpasses that of training, serving efficiency becomes increasingly important. Prefill-decode (P/D) disaggregation improves serving efficiency through specialization and isolation of the two phases. These benefits rest on a static partitioning. Phase demand, however, is not static. We observe that in a large LLM fleet the ratio of uncached input to output tokens has peak-to-mean ratios up to 4.7x at minute timescales, and that in a public agentic trace the hourly ratio spans a median 24.5x within a single day, while reassigning a replica takes tens of minutes. Agentic traffic sharpens the mismatch. Sizing each pool at its ninety-fifth percentile leaves up to 17% of cluster capacity unused; sizing below it converts the same imbalance into queueing and unrealized throughput. We present Crossflow, which makes this boundary elastic without changing node roles. Each decode node publishes a short-lived, revocable lease that bounds local-prefill compute, KV capacity, transfer work, and projected output. Across public and internal traces, Crossflow improves token throughput by 16.2-17.4% on geometric mean over static P/D, and by up to 43.4% at high load, while reducing mean TTFT at every evaluated point. 

---
# The Gaussian Is Enough: Flow-Matching Priors Do Not Help When Fine-Tuning Large Behavior Models 

**Authors**: Chen Xu, Rishi Shah, Hadas Kress-Gazit, Haruki Nishimura, Masha Itkina  

**Link**: [PDF](https://arxiv.org/pdf/2609.27070)  

**Abstract**: Modern robot imitation learning increasingly relies on generative policies based on diffusion or flow-matching models, which generate actions by transforming samples from a prior distribution. A key question is whether the choice of prior matters. Replacing the standard Gaussian with a closer-to-target, non-Gaussian prior has been shown to substantially improve performance when training from scratch. A natural next step is to ask whether these gains transfer to fine-tuning pretrained Large Behavior Models (LBMs) such as LBM 1.0, $\pi_{0.5}$, and GR00T~N1.5, where one might expect even larger gains. Surprisingly, we find that this is not the case, except possibly at very low fine-tuning data fractions. Across over 100K simulation rollouts spanning all three aforementioned LBMs on 40+ tasks in two simulation platforms, and 1250 hardware rollouts on five bimanual manipulation tasks, non-Gaussian priors that are demonstrably closer to the target yield statistically indistinguishable or worse fine-tuning performance than a standard Gaussian prior. Diagnostic analyses suggest why: fine-tuned imitation learning policies converge to similar action predictions across priors, despite their fine-tuned encoder embeddings diverging substantially from the pretrained embeddings and each other. A learning-rate ablation further confirms that encoder training is the dominant factor in fine-tuning performance, substantially outweighing the effect of prior choice. We conclude with concrete directions for future research on when and why learned priors might still matter in fine-tuning. Project page: this https URL 

---
# EMA: Elastic and Performance Transparent Memory Across GPUs 

**Authors**: Yi Xu, Tian Xia, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2609.27040)  

**Abstract**: Multi-GPU servers have become the standard building block of modern data centers, providing aggregated capacity through high-bandwidth interconnects. At the same time, workloads such as LLM inference exhibit highly dynamic memory demands, which can cause one GPU to exhaust its local memory while others remain underutilized. This mismatch motivates a model of elastic resource sharing across GPUs.
We present EMA, a memory sharing system that allows GPUs within a server to borrow and reclaim memory from each other, forming an elastic pool of capacity. EMA ensures performance transparency for both borrowers and lenders. For borrowers, prefetching hides remote access costs so that applications experience remote and local memory as indistinguishable in performance. For lenders, borrowed resources remain reclaimable on demand, guaranteeing that performance never falls below that of static partitioning. While our design focuses on memory, the same principle naturally extends to other GPU resources.
Our evaluation shows that EMA improves individual user throughput by up to 52%, achieves 96% of the throughput of a system provisioned with 2X capacity, and maintains latency similar to the static local baseline. 

---
# An open benchmark for machine learning-based polymer property prediction 

**Authors**: Robert W. Learsch, Nicholas Liesen, Daniel S. Levine, Anna M. Hiszpanski, Evan R. Antoniuk  

**Link**: [PDF](https://arxiv.org/pdf/2609.27036)  

**Abstract**: Polymer property prediction lacks open, standardized benchmarks that enable rigorous comparison of machine-learning methods, with existing resources covering only a narrow fraction of polymer architectures, such as homopolymers. We introduce Polymer Benchmark 2026 (PolyBench26), an open dataset comprising nearly 250,000 polymer-property datapoints across eight physical properties, including data from experimental measurements, density functional theory, and molecular dynamics. The benchmark supports four evaluation tasks across homopolymers and alternating, random, and block copolymers: in-distribution property prediction, dataset-size scaling, repeat-unit complexity, and transfer to held-out polymer architectures. We compare language model, graph-based, and descriptor-based approaches and find graph-based models provide the lowest errors in property prediction, retain their advantage across the evaluated training-set sizes, and remain robust to increasing repeat-unit complexity. PolyBench26 provides a reproducible foundation for developing models for the increasingly complex polymer design space. The PolyBench26 benchmark is available open-source at this https URL. 

---
# Loss Choice or Model Choice? The Role of Forecast Level in Cryptocurrency Volatility Forecasting 

**Authors**: Andrzej Tokajuk, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2609.27024)  

**Abstract**: Volatility forecasts play a central role in financial risk management because their overall level and day-to-day movements affect downstream decisions. Most studies compare forecasting models while keeping the training loss fixed. Yet losses emphasise different errors and can target different properties of future volatility, so raw comparisons may combine persistent forecast-level differences with differences in daily forecast movements. This leaves unresolved whether the importance of loss choice comes mainly from the forecast level it targets or from differences that remain after level adjustment. We address this gap through a comparison of seven losses and five models across major cryptocurrencies. Validation-based alignment adjusts the forecast level before the raw and aligned forecasts are evaluated using statistical scores and one-day Value-at-Risk. Before alignment, marginal score variation is greater across losses. After alignment, model choice becomes the larger source of variation in the full five-model comparison, while cross-loss differences in VaR breach rates narrow substantially. Our contribution is a comprehensive evaluation of loss and model choice that shows why losses can appear so influential in raw comparisons and how this interpretation changes when forecast level and downstream risk are considered explicitly. 

---
# Topological Signatures of Cyber-Attack Classes in Natural Visibility Graph Representations of Network Traffic 

**Authors**: Ali Melih Kanca, Ilker Turker  

**Link**: [PDF](https://arxiv.org/pdf/2609.26990)  

**Abstract**: Natural Visibility Graph (NVG)-based representations provide a promising approach for capturing structural patterns in sequential network traffic. However, whether different cyber-attack classes exhibit distinctive topological signatures in such representations remains insufficiently understood. This study investigates the discriminative and structural characteristics of NVG-based network traffic representations using the CSE-CIC-IDS2018 dataset. Seventy-six numerical traffic features were independently transformed into NVGs within overlapping frames of 40 observations, and ten graph-theoretic metrics were extracted from each graph, resulting in 760 topological descriptors per frame. The discriminative capability of these representations was evaluated using a multi-branch convolutional neural network (CNN) with stratified five-fold cross-validation. The model achieved an average accuracy of 96.20% and a Matthews correlation coefficient (MCC) of 0.9566. To characterize class-specific topological differences, Kruskal-Wallis and Mann-Whitney U tests were combined with Benjamini-Hochberg false discovery rate correction and effect-size measures. Of the 10,640 attack-versus-benign comparisons, 7,777 (73.1%) remained statistically significant after FDR correction, with 4,844 exhibiting large Cliff's delta effects. The strongest global differences were predominantly associated with backward-traffic and packet-length-related features combined with connectivity, clustering, and centrality measures. These findings indicate that NVG-derived representations can provide strong discriminative capability while revealing class-dependent topological patterns associated with different cyber-attack classes. 

---
# Recognized but Not Produced: A Generation Benchmark for Culturally Specific Kinship Terms 

**Authors**: Sahil Pardasani, Madhusudan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.26942)  

**Abstract**: Current literature evaluates large language models (LLMs) on multilingual kinship understanding using multiple choice benchmarks, treating it as a recognition problem. We instead prompt five open weight LLMs to generate kinship terms in three non Western languages (Hindi, Tamil, and Korean) across two communicative tasks and pair this with a matched option-supported selection baseline. On identical relation language cells, GPT OSS120B selects the correct term in 90.67% of 75 valid cells but produces an accepted term in 36.00% of the corresponding attempts; Llama 3.370B shows the same pattern (77.92% versus 24.24%). Since the four-option condition displays the candidate terms and does not require script production, the difference is interpreted as an evaluation format gap rather than direct proof that lexical knowledge is intact. On explicitly specified L3 prompts, accuracy varies sharply, from GLM-5.1 at 72.29% to Llama-3.370B at 24.24%. The paternal-lineage advantage is language specific; it is large in Hindi but weak or reversed in Korean, while Tamil shared-term pairs provide a control for measurement variation. These results show that culturally specific kinship generation remains difficult even when the relationship is explicitly stated and motivate generation-based evaluation alongside multiple-choice testing. 

---
# Experts Rise Where LLMs Disagree: Using Cross-Model Disagreement to Target Expert Effort in LLM Codebook Revision for Large-Scale Annotation 

**Authors**: Zeyu He, Zhuqian Zhou, Kirk Vanacore, Rene F. Kizilcec, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.26926)  

**Abstract**: Large-scale text annotation brings expert insight to millions of documents, often through a codebook that AI annotators follow. Developing a robust codebook, however, takes months. Large language models (LLMs) could speed this process by applying an early codebook to the data, surfacing cases with strong LLM disagreement, and eliciting expert feedback to address them. We examined three ways experts can provide feedback for LLM codebook revision: (i) editing LLM-generated revisions driven by cross-LLM disagreement (Codebook Verifying), (ii) answering questions about LLM disagreements (Question Answering), and (iii) labeling disagreement cases with rationales (Rationale Labeling). Experiments on thousands of tutoring-session transcripts show that Rationale Labeling yielded the highest LLM-labeling accuracy (64.9%) against expert labels, outperforming the expert-revised codebook (57.8%). The best Question Answering setting also outperformed it (60.5%). Our work shows that LLMs can be used to strategically target expert attention, shortening months of codebook revision to days without sacrificing labeling performance. 

---
# A 3D Pose-Based Ensemble Framework for Cricket Shot Classification and Automated Biomechanical Analysis 

**Authors**: Sourav Shome, M.D. Ashiquzzaman Rahad, Rameswar Debnath  

**Link**: [PDF](https://arxiv.org/pdf/2609.26923)  

**Abstract**: Cricket is one of the most celebrated sports world-wide, and technological advancement has become deeply embedded in how the modern game is analyzed and coached. Cricket shot classification and automated performance analysis add a further dimension to this trend. Traditional approaches rely on RGB video features or static images, which are sensitive to environmental variations such as camera angle, lighting, and background clutter, and often fail to capture the underlying biomechanics of batting actions. In this paper, we propose a system to improve cricket coaching that takes raw video data, extracts batsmen from video frames using YOLO, and extracts 3D pose data from video frames using MeTRAbs. The system produces sequential skeletal pose data of 30 body points and captures the biomechanical features of a batsman. As part of the system, we also propose a deep learning ensemble for shot classification of four shots: flick, pull, defense, and drive. The ensemble performed well, compared to existing classification works, achieving 97.68% accuracy. In addition, we analyzed the misclassification rates to identify cases where shots were incorrectly classified and examined their possible causes. Our proposed system allows novice players to obtain useful feedback, such as important joint angles relative to expert batsmen, which can also be useful for injury prevention. The shot classifier also helps track class-wise shots over time for further analysis. In addition to novice players, coaches can use the system for player evaluation. 

---
# Cross-Modal Contrastive Learning from Histopathology and CT for Automated Renal Cell Carcinoma Grading 

**Authors**: Amit Das, Tanmay Shukla, Naofumi Tomita, Faraz Farhadi, Jessica Sin, Ari Hakimi, Chad Vanderbilt, Jie-Fu Chen, Ritesh Kotecha, Weijie Ma, Bing Ren, Saeed Hassanpour  

**Link**: [PDF](https://arxiv.org/pdf/2609.26920)  

**Abstract**: Background: Clear cell renal cell carcinoma (ccRCC) exhibits substantial clinical heterogeneity, and accurate grade assessment is essential for risk stratification and treatment planning. However, conventional grading requires invasive tissue sampling. We developed RCC-Align, a cross-modal contrastive learning framework that leverages paired histopathology and computed tomography (CT) data during training to improve noninvasive CT-based ccRCC grade prediction. Methods: RCC-Align aligns paired whole-slide histopathology images (WSIs) and CT scans through contrastive cross-modal objectives, transferring grade-discriminative information from microscopic tissue morphology to macroscopic radiologic representations. The framework was trained and evaluated on paired TCGA and CPTAC cohorts using patient-level five-fold cross-validation. Performance for low- versus high-grade ccRCC classification was compared against CT-only baselines (DINOv2-Base and DINOv2-Finetuned) and a WSI-based reference model (GigaPath-Finetuned). Cross-modal alignment was assessed using cosine similarity analysis. Results: RCC-Align achieved an AUC of 0.601 (95% CI, 0.524-0.673) and AUPRC of 0.599 (95% CI, 0.541-0.676), outperforming DINOv2-Finetuned (AUC 0.545; AUPRC 0.543) with significantly improved low-grade prediction (p = 0.004). RCC-Align also demonstrated stronger paired WSI-CT embedding alignment compared with baselines. The WSI-based GigaPath reference achieved an AUC of 0.719. Conclusion: Pathology-guided contrastive learning improves CT-based ccRCC grading while requiring only CT at inference. This approach may complement tissue diagnosis when biopsy is unsafe, infeasible, or limited by intratumoral heterogeneity. Validation in larger, multi-institutional cohorts with external testing is needed before clinical translation. 

---
# On Preference Coverage Collapse from Hindsight Relabeling in Multi-Objective Reinforcement Learning 

**Authors**: Baptiste Bonin, Caro Strickland, Audrey Durand  

**Link**: [PDF](https://arxiv.org/pdf/2609.26918)  

**Abstract**: Hindsight relabeling which retroactively replacing a transition's goal with the outcome the agent actually achieved is an effective tool for improving sample-efficiency in Reinforcement Learning (RL). A natural extension to preference-conditioned multi-objective RL (MORL) relabels transitions with the preference direction the agent achieved rather than the one asked for. We show that this extension is frequently harmful: across four preference-conditioned off-policy algorithms spanning two critic backbones and two preference-sampling schemes on the continuous-control MO-Gymnasium suite, it degrades 19 of 36 algorithm-environment settings by as much as four standard deviations, improves only one, and leaves the rest unaffected.
The harm is not a symptom of noisy relabels; denoising the target recovers almost nothing, and neither prioritized sampling nor any buffer-structural choice reproduces it. Instead, repeated relabeling collapses the critic's coverage onto whatever narrow region of the preference space the agent happened to visit. We name this failure mode \emph{Preference Coverage Collapse}, and quantify it with abandoned preference mass (APM), a value-aware statistic that tracks the harm ($\rho = -0.73$) where a purely structural coverage count does not.
We then introduce \texttt{her\_mix}, a single-parameter convex combination pulling the achieved direction back towards the requested preference. At one fixed value across every algorithm and environment, it returns 16 of the 19 harmed settings to baseline, preserves and even improves the one setting in which relabeling helps, and cuts abandoned preference mass from $69\%$ to $6\%$. Protecting coverage over the preference simplex, not filtering noisy relabels, is what makes hindsight relabeling safe for MORL. 

---
# COMED: The Missing Middle Between Routing and Collaboration in Multi-LLM Inference 

**Authors**: Norah Alballa, Wenxuan Zhang, Salma Kharrat, Fares Fourati, Zafar Ayyub Qazi, Mohamed Elhoseiny, Marco Canini  

**Link**: [PDF](https://arxiv.org/pdf/2609.26913)  

**Abstract**: No single Large Language Model (LLM) is uniformly reliable across queries, motivating multi-model inference systems that either route among models or combine their outputs. However, routing stops after selecting an initial model, while dense collaboration invokes peers on every query. We show that collaboration is non-monotonic: peers can recover failures that no model solves alone, but can also corrupt initially correct answers. We introduce COMED (Controlled Model Escalation for Multi-LLM Deliberation), a post-anchor controller for selective cross-model collaboration. COMED uses anchor self-consistency, router margin, and a lightweight peer probe to accept confident answers, verify ambiguous cases, and escalate only when collaboration is likely beneficial. We formalize this trade-off with a rescue-harm decomposition showing that selective collaboration improves when rescued errors outweigh collaboration-induced harms. Across medical, scientific, and general reasoning benchmarks, COMED improves fixed and routed anchors in all 16 open-weight settings, with gains up to +10.7 percentage points on MedQA while invoking fewer models and using fewer decoded tokens than dense collaboration. On HLE with frontier models, COMED improves GPT-5.5 from 23.1% to 28.1%, outperforming dense collaboration and achieving the best results. 

---
# Ajar: Measuring Open Privilege in Agent Defenses 

**Authors**: Reshabh K Sharma, Linxi Jiang, Shuo Chen, Zhiqiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.26900)  

**Abstract**: A language model agent acts through the tools it is given. The data it reads while working on a task can redirect what it does with those tools. A growing set of techniques for safe and secure agent execution therefore sits between the agent and its tools, aiming to enforce access control, information flow or isolation at that boundary. Today these techniques are evaluated on agent-security benchmarks built around indirect prompt injection. Those benchmarks judge a defense by how far it brings the number of successful attacks down while preserving the agent's utility. A defense is judged only on the agent's execution. It can score well on both metrics while holding open a transfer, a deletion or a broad read that no task needed. Ajar measures that open privilege directly using the existing benchmarks. It attaches to an agent-security benchmark that already exists and reuses the tasks, tool schemas, reference solutions and goal states that benchmark uses to grade its own runs. For each benign task it builds candidate tool calls the task does not need, so allowing one is privilege left open. These calls are presented to the defense at every point where the agent could act. We evaluate Ajar by attaching it to AgentDojo, where open privilege becomes a third axis beside the existing attack success and benign utility. We run it on five defenses: Progent, CaMeL, AC4A, Permission Assistant, and Claude Code's Auto mode. We observed that they leave widely different amounts of privilege open. Two defenses leak by almost the same amount yet differ widely in the benign tasks they finish, and one defense buys part of its tightness by refusing calls its tasks were entitled to make. This open privilege cannot be derived from the measured attack success or benign utility. The source code of Ajar is available at this https URL. 

---
# Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness 

**Authors**: Varshini Elangovan, James Wedgwood, Chhavi Yadav, William Agnew, Sauvik Das, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2609.26865)  

**Abstract**: Conversational AI systems can pose safety risks to their users such as hallucination, sycophancy, overconfidence, and anthropomorphism, but these risks are difficult for users to detect during everyday use. We introduce Safety Nudges, a browser-based tool that provides lightweight, in situ flags when concerning behavior is detected in chatbot conversations. We evaluated Safety Nudges in a two-week field study with 45 frequent chatbot users, collecting interaction logs, surveys, and feedback on individual nudges. Participants found the tool useful, clear, and minimally disruptive, with nearly all users reporting an increased awareness of potential AI harms, though we found that this improved awareness alone did not necessarily lead to discernible behavioral changes. Our results suggest that user facing safety nudges can complement model-level safeguards by helping people critically evaluate AI responses in context, while highlighting the importance of relevance, calibration, and user control in nudge design for conversational AI safety. 

---
# Comparative Evaluation of Static Embedding Models for HTTP Request Anomaly Detection 

**Authors**: Amanda Riverol, Gustavo Betarte, Rodrigo Martínez, Álvaro Pardo  

**Link**: [PDF](https://arxiv.org/pdf/2609.26860)  

**Abstract**: Web applications are increasingly targeted by cyberattacks that exploit HTTP requests to evade security mechanisms. Traditional web application firewalls (WAFs) rely on rule-based approaches that often exhibit high false positive rates and limited adaptability. Recent studies have explored machine learning techniques and word embedding models to improve anomaly detection in HTTP traffic. This paper presents a benchmark for static embedding models, specifically Word2Vec, FastText, and Doc2Vec, within a unified, single-class classification framework. We propose HEDA (HTTP Embedding-Based Detection Architecture), a modular detection pipeline that combines static embedding representations with single-class anomaly detection models to detect anomalies at the request level. The approach operates in an unsupervised environment, where both the embedding models and detectors are trained exclusively on benign HTTP traffic. The proposed methodology is evaluated on three datasets with heterogeneous characteristics, including both synthetic and real traffic. The experimental results show that the choice of embedding representation significantly affects detection performance, and that FastText-based embeds produce the most consistent results across all datasets, achieving high detection rates while keeping false positive rates under control. 

---
# FLINT: Fast Lightweight Inference for Traversability 

**Authors**: William Bonilla, Maxime Boisvert, David-Alexandre Poissant, David Meger, Louis Petit  

**Link**: [PDF](https://arxiv.org/pdf/2609.26857)  

**Abstract**: Navigation in off-road conditions is challenging due to the lack of structure. There is no fixed vocabulary for what is traversable. The traversability depends on both the environment and the embodiment's dynamics. Neither of these two variables can be hand-labeled at scale. Thus, traversability has to be learned by the embodiment's own experience. Modern platforms tend to use multiple sensors to estimate traversability and navigate: RGBD cameras, lidar, radar, IMU, with computationally intensive platforms to run inference on neural networks. Against this trend, we propose FLINT, a lightweight traversability estimator: a 21.6M-parameter backbone, 38\times smaller than a comparable foundation-model backbone, that scores higher on held-out terrain probes and runs at 14.7 FPS on CPU alone using a RGB camera has the only sensor. Despite that gap in scale, FLINT produces a cheaper, more accurate costmap than a deployed foundation-model system (WildOS) on 23 of 24 replayed field logs. We compare different self-supervised learning signals and deploy the resulting models on a real platform in closed-loop field trials: the best self-supervised head reaches 99% autonomy over the route, outperforming a human-label-trained baseline deployed live on the same course. Our results show that heavy sensing and computing are not necessary for traversability estimation. 

---
# QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs 

**Authors**: Kyaw Hpone Myint, Nan Jiang, Xiang Li, Zhe Wu, Alexandre G.R. Day, Pranab Mohanty, Giri Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2609.26855)  

**Abstract**: Relational Deep Learning (RDL) models multi-table databases as heterogeneous temporal graphs, and graph transformers currently achieve state-of-the-art performance on benchmarks like RelBench. However, the current leading model, RelGT, suffers from two key limitations: its random local sampler yields loosely connected subgraphs that hinder message passing, and its global attention module relies on a single, seed-feature-based memory that ignores broader macro-level dynamics. To overcome these limitations, we introduce QUARTET, an expressive graph transformer architecture that applies full self-attention on local subgraphs while enriching global context through cross-attention branches. Specifically, QUARTET employs a Causal Random Walk (CRW) sampler based on recency-truncated Personalized PageRank (PPR) to extract compact, hub-robust, and densely connected local subgraphs without temporal leakage. Concurrently, a quad-branch cross-attention module integrates global context from four complementary perspectives: seed feature, seed topology, temporal dynamics, and collaborative dynamics. Across the RelBench v1 classification tasks, QUARTET consistently matches or outperforms the current state-of-the-art graph transformer baselines (HGT and RelGT). Ablation studies confirm that the CRW sampler significantly enriches local neighborhood quality, while the global branches provide essential, task-specific predictive gains. 

---
# SsgCaps: A controlled dataset for the evaluation of sound scene generation algorithms 

**Authors**: Modan Tailleur, Junwon Lee, Laurie M Heller, Mathieu Lagrange, Keunwoo Choi, Brian McFee, Keisuke Imoto, Yuki Okamoto  

**Link**: [PDF](https://arxiv.org/pdf/2609.26854)  

**Abstract**: Sound Scene Generation is about the automatic synthesis of artificial sound scenes. We introduce SsgCaps, a publicly available dataset of human-engineered sound scenes wherein each scene matches a precisely structured prompt that guides the sampling process. The corresponding prompts are sampled from a predefined action-based typology that allows extensive sampling while retaining plausibility. SsgCaps is a sound scene dataset derived from the unpublished reference dataset for Task 7 of the 2024 DCASE Challenge edition, which contained private-and public-domain audio samples. In contrast, SsgCaps contains only public-domain audio samples, allowing us to open this dataset to the community. To make this dataset useful to the community, we first elaborate on the rationale for the prompt and dataset structure. We then perform a comparative quantitative analysis of the 2 versions of the dataset. To do so, we compare both versions to the audio synthesized by the SSG algorithms submitted to the challenge using Fr{é}chet Audio Distance (FAD) and Kernel Audio Distance (KAD) as well as perceptual ratings. This analysis shows only small differences, which enables us to recommend the open version for further benchmarking of SSG algorithms. 

---
# COPE: Continual Personalization of LLMs under Sparse User Feedback via User Embeddings and Self-Evaluation 

**Authors**: Ruike Cao, Fugen Yao, Liang Dong, Jian Xu, Guanjun Jiang, Li Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.26853)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable results across various benchmarks, their alignment with normative values often results in homogenized responses that fail to address diverse user preferences. Existing training-free methods often occupy valuable context windows through prompt engineering, while training-based methods typically remain static post-training, failing to support the continual optimization required in real-world settings. To address these challenges, we propose COPE (Continual Optimization with Personalized embedding and self-Evaluation), a novel optimization framework tailored for real-world-motivated interaction settings with sparse user feedback. Our framework assigns learnable personalized embeddings to each user and synergistically integrates preference capture, self-evaluation calibration, and personalized response optimization within a single update step. A key innovation of our method is the use of self-evaluation to generate proxy rewards, enabling continuous model updates even when explicit user feedback is unavailable. Experiments show that COPE consistently outperforms strong training-free and training-based baselines under sparse feedback, and remains complementary to Retrieval-Augmented Prompting (RAP). Further analyses confirm COPE's reliable self-evaluation, meaningful preference patterns, stable general capabilities, and robustness under shifting preferences and alternative evaluators. 

---
# A Leakage-Aware Multimodal Evaluation Framework for Early Intraoperative Acute Kidney Injury Prediction 

**Authors**: Quang Minh Nguyen, Duc Minh Le, Ho Nhat Minh Nguyen, Thuy Quynh Nguyen, Trong Nghia Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2609.26848)  

**Abstract**: Postoperative acute kidney injury (AKI) after major non-cardiac surgery carries substantial morbidity, yet early intraoperative risk stratification remains difficult. In this retrospective cohort study, we propose SynerT, a waveform-only hybrid temporal backbone that combines a causal dilated TCN with a hierarchy of dilated recurrent layers to encode early intraoperative physiologic trajectories for AKI risk prediction. Building on SynerT, we further design two model variants that extend the backbone with structured clinical context: SynerT-MM, a late-fusion multimodal extension that integrates hemodynamic burden summaries and preoperative covariates, and SynerTStack, a leakage-safe stacked ensemble that combines cross-validated predictions from SynerT-MM with strong tabular baselines at the meta-learning stage. All models are evaluated under a strict leakage-aware framework on VitalDB, a high-fidelity perioperative database, with prediction restricted to information available within the first 60 intraoperative minutes. Among 2,413 waveform-usable cases (180 AKI-positive; 7.46% prevalence), SynerT fell well below strong structured-data baselines, demonstrating that waveform-only temporal modeling is insufficient under strict early constraints. SynerTMM recovered discrimination by incorporating hemodynamic burden summaries and preoperative covariates, and SynerT-Stack achieved the best overall performance across AUROC, AUPRC, and F1-max. Cross-fitted Platt recalibration substantially corrected calibration defects in both multimodal variants, and decision-curve analysis confirmed the recalibrated stacked model delivered the strongest net clinical benefit across low-to-intermediate thresholds. 

---
# LWCal: Loss-Weighted Calibration for Tabular Classifiers with Noisy Calibration Labels 

**Authors**: Zeming Liu, Hang Lyu, Jingtao Zhang, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.26839)  

**Abstract**: Post-hoc probability calibration is usually evaluated under an optimistic assumption: the held-out calibration labels are clean. In many AI deployment settings, however, labels come from weak annotators, historical decisions, heuristics, or distant supervision, so the same label noise that corrupts training also corrupts calibration. We study this overlooked failure mode for tabular classifiers and propose LWCal, a CPU-only post-hoc calibrator that down-weights calibration examples whose noisy labels are contradicted by the base model's held-out probability. LWCal requires no clean validation labels, no noise-rate estimate, and no retraining of the base classifier. A second variant, Gated-LWCal, adds a conservative disagreement gate that backs off toward the raw score when the calibration split appears extremely inconsistent. On nine local binary tabular tasks, six random seeds, symmetric and asymmetric label corruption, and three tree-based base learners, LWCal obtains the lowest average calibration error while Gated-LWCal obtains the best average proper-score tradeoff. In the main random-forest study over 432 noisy cells, Gated-LWCal reduces expected calibration error from 0.188 to 0.122 and negative log likelihood from 0.438 to 0.396 relative to the raw classifier. Paired bootstrap intervals for Gated-LWCal versus raw, Platt, isotonic, and beta calibration exclude zero on ECE, Brier score, and NLL. The artifact contains all scripts, result tables, figures, and the compiled paper. 

---
# Spec2COBOLRot: An Agentic-AI Degradation Loop for Realistic COBOL Corpus Generation 

**Authors**: Jean-Baptiste Espinasse, Djamel Eddine Khelladi, Mathieu Acher  

**Link**: [PDF](https://arxiv.org/pdf/2609.26835)  

**Abstract**: COBOL remains widely deployed, yet representative corpora reflecting real production code are rarely available, limiting rigorous benchmarking of modernization approaches. We propose a systematic agentic AI pipeline for generating realistic COBOL programs, combining specification-driven generation with iterative degradation guided by patterns and complexity targets extracted from real production code. Here, realism is understood as structural fidelity to production code as captured by our metrics. We evaluate whether degradation reaches target complexity levels while preserving business behavior, and examine the limits of the approach, across three programs from distinct business domains. Results show the pipeline reliably produces syntactically valid programs and moves them toward realistic structural complexity. However, preserving business behavior is not always achieved by construction, and targeting structural metrics independently of business logic risks producing programs whose complexity does not reflect a plausible maintenance history. We discuss these limitations and outline a more realistic alternative as a direction for future work, generating legacy programs from scratch along a simulated development history. 

---
# Validation and Simulation Catch Different Errors: Four Levels of Evaluation for LLM-Generated Circuits 

**Authors**: Ali Hedayati Pirouzan  

**Link**: [PDF](https://arxiv.org/pdf/2609.26830)  

**Abstract**: Simulation success is not equivalent to structural correctness for LLM-generated circuits. We define and measure four evaluation levels -- schema validity, topological validity, backend executability, and component-set agreement -- on a 150-circuit trilingual benchmark, through a deployed pipeline built on a typed circuit interchange representation.
The levels are not nested. On gpt-4o-mini, 16 of 150 circuits (10.7%, 95% CI 6.7-16.6) were rejected by the topological validator but executed in ngspice with no error or warning; 12 of these contained exactly the requested components, with one terminal disconnected. Conversely, 7 circuits (4.7%) passed the validator and ngspice refused them. Ten failed both checks and 117 passed both, so each check detects a class the other misses. A minimal three-component divider shows the cost: a dangling resistor reports 5.00 V instead of 2.50 V while ngspice stays silent.
A paired ablation, in which every arm is evaluated from the same model sample rather than a fresh one, separates each repair stage from sampling noise. On a stratified 45-circuit subsample, model repair raised topological validity from 40.0% to 84.4% (+20 circuits, no regressions) while moving executability by a net 6 (+7, -1), an effect this sample size does not resolve, and component agreement by 2. One circuit moved in opposite directions at two levels in a single repair step. Against a direct-netlist baseline the pipeline executed 88.7% against 47.3%, or 62.7% under an accounting that credits the baseline with every failure we cannot confidently attribute to the netlist.
These results support a narrow methodological conclusion: structural validation and simulation should be reported as distinct evaluation stages for LLM-generated circuits. A circuit that runs is not necessarily structurally valid, and a structurally valid circuit is not necessarily executable. 

---
# Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management 

**Authors**: Hyunsun Chung, Taewan Noh, Minji Kim, Joo-Young Hwang, Hong-Yeon Kim, Youngjae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.26828)  

**Abstract**: NAND-backed storage offers the capacity needed to scale LLM prefix caching, but its block I/O path incurs CPU cache contention and host-DRAM staging in addition to NAND latency. Our characterization shows that these interface costs persist even with DRAM as the storage medium, motivating CXL-SSDs for byte-addressable access to NAND-backed capacity. Surprisingly, however, a stock CXL-SSD remains about 3$\times$ slower than local DRAM and no faster than an NVMe SSD, while generic prefetching provides little benefit. We present LM-CXD, a CXL-SSD specialized for LLM prefix caching. LM-CXD bridges the semantic gap between the serving engine, which knows which KV chunks will be consumed, and the device, which controls their placement and movement. It makes KV chunks device-visible I/O units, exposes NAND-to-DRAM progress to the serving engine, and uses device DRAM as a GPU-accessible buffer. LM-CXD further coordinates request scheduling with windowed prefetching and pipelines layerwise KV movement with GPU computation to hide NAND latency under limited device DRAM. Across five LLM models, LM-CXD reduces average TTFT over a stock CXL-SSD by up to 2.6$\times$ with compute asynchronous prefetching and 4.03$\times$ with layerwise prefetching, achieving TTFT within 1.5$\times$ of local DRAM on average. 

---
# What Makes a Terminal-Bench Task Hard? Separating Genuine Hardness from Fake-Hardness on an Adjudicated Agentic Corpus 

**Authors**: Edward Lue Chee Lip, Boden Moraski, Tim Knappe, Lang Xiong, Sarvesh Gharat, Antonio Mari, Ivan Bercovich  

**Link**: [PDF](https://arxiv.org/pdf/2609.26826)  

**Abstract**: Frontier benchmarks need tasks that current models cannot solve. But a task that no model solves is not automatically a hard task. The same zero pass rate can come from a real capability gap, but it can also come from missing context, a broken reference solution, infrastructure failure, or a verifier that can be bypassed. In this paper, we study this issue using a frozen Terminal-Bench 3 / Frontier-Bench 0.1 production record with 1,081 pull requests, 639 scored tasks, 28,801 trials, and $105,933 in logged agent spend. We ask what an all-fail task actually certifies. For the 125 tasks with no honest pass, we combine task artifacts, reference-solution runs, empty-solution controls, adversarial trials, trajectories, telemetry, and review records, and apply an ordered validity screen. Only 78 of the 125 tasks survive as certified-unsolved candidates. The remaining tasks include 14 with broken oracles, 8 dominated by infrastructure failures, 4 that are only passable through verifier bypasses, and 21 whose solvability is not certified by the available evidence. Thus, lack of saturation and genuine difficulty are not the same thing. The certified-unsolved label is also narrow: it means that the authored route passed, infrastructure did not dominate, no strict bypass was observed, and all evaluated agents failed. It does not prove intrinsic hardness, verifier completeness, or failure at the intended capability. We further analyze rejected submissions and passing tasks to show that pass rate alone cannot explain why a task is difficult. Overall, our results suggest that frontier benchmarks should report the evidence behind their all-fail tasks before using them as capability claims. 

---
# Signal2Symbol: Neuro-Symbolic Temporal Reasoning for Explainable Physiological Time-Series Anomaly Detection 

**Authors**: Naser Mansour, Sidahmed Benabderrahmane, Ameer Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2609.26820)  

**Abstract**: Physiological time series such as electrocardiograms (ECG) and electroencephalograms (EEG) exhibit complex temporal structure, substantial acquisition variability, and a strong need for transparent decision-making. Although deep models can achieve high detection performance, they often provide limited insight into why a segment is anomalous, how local anomalies relate over time, and whether a detection belongs to a broader recurring pattern. We propose Signal2Symbol, a neuro-symbolic framework for explainable biosignal anomaly detection. The method first converts ECG/EEG signals into symbolic sequences using either a learned VQ-VAE (Vector Quantized Variational Autoencoder) codebook or a SAX (Symbolic Aggregate approXimation) baseline. It then constructs bigram enriched token-window transactions and scores anomalies through rare itemset evidence derived from minimal rare itemset mining. Detected anomalous windows are merged into intervals and related using Allen interval algebra, enabling composite temporal explanations such as escalation chains, artifact overlap, and cross-channel synchrony. Finally, we introduce a rare temporal concept lattice based on Formal Concept Analysis (FCA), which groups anomalous intervals by shared rare symbolic evidence, Allen temporal relations, channel context, and robustness attributes. The resulting Galois lattice compresses many local detections into interpretable families of temporal-symbolic anomalies. We evaluate on three public benchmarks: MIT-BIH Arrhythmia (beat-level ECG), PTB-XL (record-level ECG), and the Bonn EEG dataset (segment-level EEG). We stress-test robustness under additive noise and baseline-wander perturbations. The results highlight the value of neuro-symbolic tokenization for temporal anomaly analysis and show that Allen/FCA reasoning provides compact, interpretable summaries of local detections. 

---
# Learning Stiffness Dependent Fluid Structure Dynamics from Coarse Flow Representations 

**Authors**: Chun-Jun Pu, Li-Wei Chen, Hai-Bo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.26816)  

**Abstract**: This paper develops a data-driven framework for long-term prediction of fluid--structure interaction (FSI) dynamics, focusing on the flow-induced vibration (FIV) of a flexible plate. A stiffness-conditioned neural evolution operator jointly represents the Eulerian flow field and Lagrangian structural state. The plate is represented by 101 ordered structural tokens carrying nodal coordinates and velocities, with nondimensional bending stiffness as a global conditioning variable. Bidirectional cross-attention couples fluid and structural representations within a hybrid CNN-Transformer architecture. Trained with staged multi-step autoregressive rollouts and symmetry-reflected trajectories, a single operator captures three stiffness-dependent response regimes: deflected--flapping, deflected, and flapping. The predicted trajectories preserve the principal flow structures, structural oscillations, and dominant frequencies, while blind 1000-step rollouts remain bounded. The operator also interpolates to stiffness values excluded from training.
To reduce sensitivity to under-resolved near-wall gradients in force reconstruction, we develop a differentiable aerodynamic-force module based on the derivative-moment transformation (DMT). Conventional wall-stress surface integrals are replaced by an enclosed 2D curve integral around the core vortex region, enabling accurate reconstruction of lift and drag. A Signed Distance Function (SDF) and smoothed Dirac-delta formulation make the integration fully differentiable while preserving gradient flow. The proposed framework provides an accurate and differentiable surrogate for stiffness-dependent FSI dynamics, enabling efficient parameter studies and future stiffness optimization for flow-energy-harvesting applications. 

---
# Gödel's and Scott's Variants of the Ontological Argument in Lean 4 

**Authors**: Christoph Benzmüller  

**Link**: [PDF](https://arxiv.org/pdf/2609.26806)  

**Abstract**: This paper presents a complete, structure-preserving port to Lean 4 of the Isabelle/HOL dataset accompanying Benzmüller and Scott's study of Gödel's modal ontological argument and Scott's variant of it. The port comprises 30 Lean 4 modules, one per Isabelle/HOL theory, retaining the section structure, the declaration order and the name of every axiom, definition, lemma and theorem; a comparison tool certifies all 548 statements identical. Everything the Isabelle/HOL development proves is proved again, including the inconsistency of Gödel's 1970 axioms, the repaired Gödel variants, Scott's variant, modal collapse, monotheism and the ultrafilter property of the positive properties; five statements the original leaves unreplayed after an automated prover had found a proof, one of which it then postulates, are proved as well. The 45 remaining unproved statements are exactly those the original refutes by nitpick (35) or leaves open (10); they are anonymous sorrys on which nothing depends.
Two features of Lean 4 shape the result. It has neither a sledgehammer nor a model finder, so the one-line automated proofs become explicit proof terms and the 72 nitpick invocations are recorded as documentation. And #print axioms reports the postulates each proof consumes, giving for every result an upper bound on the modal logic it requires: Scott's necessary-existence theorem and modal collapse need only symmetry of the accessibility relation (logic KB); the essence and monotheism lemmas and the possible existence of a God-like being need no frame condition (the latter with the one exception the original records, the mixed-quantifier setting); and the inconsistency of Gödel's 1970 axioms needs none either. The development depends on no library beyond Lean 4's core; sources, comparison tools and two Isabelle cross-check sessions are included as ancillary files. 

---
# Attention-based representations for multi-task computation 

**Authors**: Daniel Hsu, Mingyue Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.04243)  

**Abstract**: Multi-head attention layers produce vector representations that support multiple downstream tasks. We establish bounds on the number of heads required in two simple and concrete multi-task scenarios. In the first scenario, a vector representation is sought so that linear predictors can compute both the smallest and largest numbers in a given list. In this case, it is known two attention heads with small embedding dimension and bit precision level suffice. We prove that a single attention head requires exponentially higher embedding dimension or precision level. In the second scenario, a vector representation is sought so that a polynomial threshold function can compute the XOR of a given string of $n$ bits. This scenario is analogous to the first one for $n=2$, since XOR is readily computed by a linear function using a vector representation that encodes both the AND and the OR of the two bits. We observe that $n$-bit XOR requires the product of the number of heads and the polynomial degree to be at least $n$, and we construct multi-head attention layers that match this lower bound. These results generalize to arbitrary (symmetric) Boolean functions, where the bound is given in terms of the threshold degree. 

---
