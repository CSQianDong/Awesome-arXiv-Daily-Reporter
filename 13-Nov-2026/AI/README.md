# AgentSSL: Can MLE Agents Leverage Unlabeled Data? 

**Authors**: Ethan Lin, Akanksha Sarkar, Ziang Liu, Kristin Branson, Jennifer J. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.17323)  

**Abstract**: Applying machine learning systems to new domains introduces bottlenecks in both labeled data acquisition and manual pipeline engineering. While recent MLE agents show promise in automating pipeline design, they have been studied almost exclusively in fully supervised settings, leaving them untested in realistic regimes where labeled data is scarce but unlabeled data is abundant. In this work, we introduce Agentic Semi-Supervised Learning (AgentSSL), framing SSL as a program search problem and studying how agents navigate this space under limited supervision. We find that even under weak feedback and matched compute, agents can synthesize code that combines and adapts established SSL techniques, exceeding the strongest human-engineered baselines in 8 of 12 settings by up to +21.8%. However, our results identify program selection as a critical bottleneck in AgentSSL, where performance drops by as much as 47% when selecting based on the agent's feedback metric. To systematically study this bottleneck, we introduce ProgramSelectBench, a large-scale benchmark with a combined total of 16,977 agent-generated and human-engineered SSL programs, and evaluate both unsupervised and active program selection methods. We further evaluate AgentSSL on real-world ecology detection tasks, highlighting the opportunities and challenges of transferring agent-discovered strategies beyond controlled benchmarks. 

---
# CoReMem: Cost-Internalizing Memory for Successive Coding Tasks with Prospective Valuation 

**Authors**: Peiwen Li, Harshit Verma, Sophia Zhou, Haiwen Wang, Ngoc Bui, Heng Chang, Fei Fei, David van Dijk, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2611.17316)  

**Abstract**: Coding agents tackle successive tasks on an evolving repository, carrying forward edits and writing memory before the next task is known. Selecting memory for unknown next tasks requires balancing future usefulness against the costs of writing and reading it. The actor must also decide when acting on available memory saves effort as the task unfolds. We introduce Coding-Repository Memory (CoReMem), a cost-internalizing framework that guides memory design and actor adaptation through measured memory value on the next task. Prospective segment valuation compares runs of the next task with and without individual trajectory segments from the same inherited repository state, guiding a compact memory built around anchors: repository locations and executable checks. Actor adaptation estimates on-policy memory value within rollout groups with and without memory, using a reward that prioritizes resolution and penalizes step count in successful runs. Provenance-gated value credit assigns the additional value term to steps identified as using memory-supplied anchors, while every step retains outcome credit. On ChainSWE, CoReMem raises the 27B actor's resolve rate from 21.0% to 25.7%, with 17% fewer steps and 11% fewer total cache-aware tokens, including memory-module calls, relative to the untrained actor model without memory. 

---
# Classical Energies Learn to Reason 1,000x Faster than Looped Models 

**Authors**: Benjamin Hoover, Hugh Blayney, James C. R. Whittington  

**Link**: [PDF](https://arxiv.org/pdf/2611.17309)  

**Abstract**: Looped models have emerged as a prominent paradigm for reasoning, with successes on simple reasoning puzzles such as Sudoku-Extreme and Maze-Unique. However, these models can be unstable, slow to train, and use many parameters and FLOPs to solve simple puzzles. Descent on an energy not only mitigates these issues but also offers advantages: built-in early stopping, compute that adapts to problem difficulty, fast training without differentiating through a solver, and interpretable weights that learn the constraints of the puzzle. Here we demonstrate this with a learned version of Hopfield & Tank's energy from the 80s, which we term CRAM, solves 99.996% of Sudoku-Extreme and 100% of Maze-Unique test puzzles after 42 ms and 132 ms of training on one GPU, at least 1,000x faster than looped models on these tasks. CRAM also matches baseline looped models at their highest accuracy on Sudoku-Extreme using at least 282x fewer inference FLOPs. Our work suggests two things. First, that energies are an extremely promising direction for training reasoning from data. Second, a caution against using trivially solvable benchmarks like Sudoku and Maze to test the scalability of looped reasoning techniques. 

---
# LiveGeo: Benchmarking Geographic Agents in Continuously Evolving Worlds 

**Authors**: Pengcheng Zhou, Xuanyu Liu, Yanchen Yin, Tianjie Ju, Bobo Li, Shengqiong Wu, Hao Fei, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17267)  

**Abstract**: We introduce GeoWorld, a controllable and continuously evolving geographic environment platform for LLM agents, and LiveGeo, a benchmark for evaluating reliable decision-making in real-world geographic environments using GeoWorld. Unlike existing evaluations that largely reason and plan over fixed geographic states, GeoWorld is grounded in real-world geographic structure and allows external world states to evolve along their own temporal processes while preserving verifiable state histories. LiveGeo places geographic tasks within this evolving environment, requiring agents to actively acquire task-relevant information, verify whether the evidence supporting prior decisions remains valid, and revise their plans when necessary. Our results reveal that even frontier LLMs struggle with static tasks, with no model consistently excelling across tasks, while strong static performance does not ensure dynamic reliability. Further analysis identifies a key bottleneck: agents struggle to acquire and translate relevant evidence into valid decisions and maintain their correctness as the world evolves. By releasing LiveGeo together with its GeoWorld platform, we aim to provide the community with a flexible and extensible infrastructure for studying the next generation of agents capable of reliable action in evolving geographic worlds. 

---
# Ballad: Bandit-Based LLM Routing for Automated Heuristic Discovery 

**Authors**: Samidha Verma, Ankit Anand, Sayan Ranu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17241)  

**Abstract**: Modern automated heuristic discovery (AHD) systems increasingly rival hand-crafted heuristics by leveraging Large Language Models (LLMs) to iteratively mutate and refine solutions. A natural assumption is that more powerful and more expensive models consistently yield better heuristics. We challenge this empirically: across multiple NP-hard combinatorial optimization problems, we find no consistent advantage for larger, costlier models. This raises a deeper question: given a pool of LLMs spanning frontier and freely hosted or open-weight models, how should one adaptively select which model to invoke at each step of the search? The challenge is non-trivial. The optimal LLM may shift over the course of a run, and because one LLM's output seeds the context for subsequent calls, each model's reward signal is non-stationary and entangled across the heuristic population, thus violating the independence assumptions of classical online decision-making settings. We address this through Ballad (Bandit-based LLM Routing for Automated Heuristic Discovery), an online bandit algorithm that learns to orchestrate a pool of LLMs into an ensemble that matches or exceeds what any soloist LLM achieves alone. This orchestration is powered by an LLM selection policy that favors rare, high-scoring heuristics over stable averages, a DAG-based credit propagation mechanism through the genealogy of the heuristic population, and discounting of stale observations to remain attuned to each model's current utility. Across five NP-hard combinatorial optimization problems and a pool of four models of varying capability and cost, Ballad not only matches the best individual LLM in hindsight but, in several cases, surpasses every individual model while reducing API cost by up to 65.6% relative to always using the strongest model. 

---
# Al-Ghasali: A Computational Framework for Simulating Islamic Jurisprudential Rulings via Finite State Automata and Large Language Models 

**Authors**: Elnaserledinellah Mahmoud Abdelwahab  

**Link**: [PDF](https://arxiv.org/pdf/2611.17235)  

**Abstract**: This paper presents Al Ghasali, a computational framework designed to formalize and simulate Islamic jurisprudential (Fiqh) rulings. Building upon recent set theoretic formalizations of al sabr wa al Taqsim that extract minimal operational rules as simple Horn clause logical programs, this work proves that the semantics of closed Fiqh chapters collapse completely to deterministic Finite State Automata (FSA). We establish a formal computational hierarchy, demonstrating that restricting formalization to finite propositional Horn theories guarantees decidability and constant time (O(1)) verdict extraction after offline compilation. The framework formally proves that hierarchical compositions of sub-chapter FSAs preserve this O(1) complexity, and that temporal juristic requirements, such as Tartib (order), are deterministically flattened into static ontological states, preserving the memoryless property of the automaton. To generate human-readable juristic summaries, Al Ghasali incorporates Large Language Models (LLMs) via an architecture where the deterministic FSA strictly controls the probabilistic LLM. Crucially, the paper formulates sufficient design safety conditions for this FSA LLM integration, ensuring strict adherence to the epistemological boundaries of Usul al Fiqh. To neutralize the risk of LLM hallucination, the system employs a Grammar Guided Constrained Decoding mechanism. We formally define the mapping function that dynamically constructs unambiguous right linear grammars from the FSA execution history. The paper mathematically proves that this architecture, augmented by a Bounded Neutral Lexicon and Valence Locking, guarantees the absolute absence of Decisional Hallucination, Juristic Factual Hallucination, and Semantic Inversion reducing the LLM to a pure linguistic transformer of deterministic juristic facts 

---
# Cortex-FL: Agent-Mediated Collaboration for Federated Learning 

**Authors**: Rui Sun, Jinfeng Peng, Zehua Cheng, Jiahao Sun, Zhipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17186)  

**Abstract**: Federated learning (FL) enables independent data owners to train models collaboratively without centralizing raw data. However, identifying collaborators, preparing training tasks, and coordinating execution and incentives hinder its practical adoption by organizations without specialized expertise. We present Cortex-FL, an agent-native FL system that enables users to express learning objectives in natural language and delegate the collaboration required to train a model. Agents translate these objectives into shared tasks, discover collaborators, and negotiate participation agreements, connecting collaboration formation with distributed training and incentive management. The architecture separates agent coordination from local execution authority, allowing owners to delegate coordination while retaining control over their data, computation, and financial commitments. We demonstrate the complete workflow on a four-node deployment, in which a model requested in natural language proceeds through collaboration formation, contract-governed training, verified model delivery, and x402 settlement on a blockchain testnet. 

---
# Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents 

**Authors**: Haochen Liu, Qianhao Zhu, Zixuan Wang, Dingming Li, Tongbo Chen, Peng Qiu, Yingqi Qu, Weiming Lu, Jun Xiao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.17167)  

**Abstract**: Decision models answer a question with a typed output, such as a choice among options with a probability for each, so one model can choose an agent's actions, evaluate actions already taken, or score every action to teach a policy with reinforcement learning. Prior work has studied them mainly as actors, judging their usefulness by how well they act. We ask whether a decision model that acts poorly can still teach well. We compare three decision models and three general LLMs as actors and judges on ALFWorld, and use JEV as a teacher on ALFWorld and WebShop. As actors, all six models are pulled away from a subgoal that must come first by an easier one. As judges that see the whole trajectory, the same models credit the steps toward that subgoal, and asking whether each action helped, rather than whether the task will succeed, separates useful from useless actions. As a teacher, JEV raises a Qwen2.5-1.5B policy over GRPO from 72.8% to 82.9% success on ALFWorld and from 56.8% to 71.4% on WebShop. These results suggest that a poor actor can still teach well, because judging an action after the episode ends reveals the consequences that acting must anticipate. Code is available at this https URL. 

---
# Cited but Unused: Parametric Provenance in Retrieval-Augmented Generation 

**Authors**: Tanzeel Abbas, Tehseen Zia, David Windridge  

**Link**: [PDF](https://arxiv.org/pdf/2611.17161)  

**Abstract**: Retrieval-augmented generation systems attach citations to attribute claims to their sources. These citations frequently misstate provenance, and standard verification cannot detect this. We measure provenance by counterfactual value substitution: a fact in the retrieved passages is replaced with a false variant and the model re-queried. We find that across five sizes of one model family and a second comparator family, 1.2% to 15.7% of citable claims derive from parametric memory rather than the retrieved context. At the most conservative substitution distance these figures are 4.6% and 8.5% on the two models we tested at it, with the rate rising with model size within a family. We further find that natural language inference verification, the standard approach, gives results indistinguishable from chance across models, even when restricted to the cited passage. We determine that the cause is structural: a model answering from memory selects a citation that agrees with its output. Comparing ten attribution methods on identical claims, the contrast between the model's distribution with and without context best separates provenance. Acting on that signal does not improve provenance honesty at these base rates. The resulting detector is hence still not accurate enough to act on: it measures what the model knew rather than what it used, and separating those remains the open problem. 

---
# CapLite: Prompt-Injection Defense for Small Language Model Agents without Model-Written Security Artifacts 

**Authors**: Zimo Xu, Wenzhuo Liu, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2611.17115)  

**Abstract**: Enforcement defenses against prompt injection, such as CaMeL, Progent, IPIGuard and DRIFT, ask the agent's own model to write a security artifact (a program, a privilege policy, a dependency graph or a constraint list) that code then enforces. We ask whether small language model (SLM) agents can be trusted to write these artifacts. Across 15 open-weight models, 13 of them between 1.7B and 12B parameters, they cannot: the policies, graphs and constraints that the agent's model writes admit attacks on 14 (Progent), 13 (IPIGuard) and 11 (DRIFT) of the models, and CaMeL's programs fail to run in up to 68% of episodes. We present CapLite, which derives its policy from the tool schemas, commits the write calls before any tool result is read, and admits a write only if every argument that decides whom or what it affects is stated in the user's request. This guarantees write integrity: injected text cannot add a write, and no write reaches a recipient, amount, object or destination that the user did not state. No attack succeeds in 13,935 attacked AgentDojo episodes. CapLite reaches utility under attack comparable to CaMeL's (29.0% vs. 27.2%) with fewer tokens than every other enforcement defense and than the undefended agent: 17.1k per episode, 18% of CaMeL's, and the highest utility per token of all enforcement defenses. 

---
# Speed-to-Power Dominates Energy Costs for AI Data Centers: Technoeconomic Analysis of Powering AI in the Continental United States 

**Authors**: Alex C. Newkirk, Daniel L. Gerber, Erica R. H. Fuchs, Alex Hubbard, M. Granger Morgan, Arman Shehabi, Sarah J. Smith, Constantine Samaras  

**Link**: [PDF](https://arxiv.org/pdf/2611.17098)  

**Abstract**: The rapid growth of artificial intelligence is driving demand for new data centers at large scale and on short deployment timelines. To power them, firms connect to the commercial grid or contract generation directly, but interconnection and gas-turbine delays now incentivize on-site generation. We compare grid-connected and off-grid data center energy infrastructure (AC- and DC-coupled solar-plus-storage, islanded natural gas, and grid interconnection) across the continental U.S. Using an open-source technoeconomic framework built on empirically calibrated AI training loads, climate-driven cooling, stage-by-stage power conversion, and physics-based battery degradation, we model designs providing equivalent energy services. Our central contribution is a speed-adjusted cost of energy that values the compute unlocked by faster construction, making options with different deployment timelines directly comparable on one physically consistent basis. We find DC coupling reduces lifetime energy costs by 18% ($0.09/ kWh) on average versus AC counterparts by eliminating inverters and improving efficiency. Though rarely cheapest on unadjusted energy cost, DC-coupled microgrids' faster deployment makes them the lowest-cost modeled option in 85% of U.S. locations once foregone compute is valued, or 55% under accelerated natural gas timelines. For AI infrastructure, deployment speed rather than unit energy cost determines the economically preferred path to power. 

---
# Reasoning Supervision and Capability Retention in LoRA Domain Adaptation 

**Authors**: Saleh Ashkboos, Dhruv Desai, Lavinia Ghita  

**Link**: [PDF](https://arxiv.org/pdf/2611.17072)  

**Abstract**: Fine-tuning foundation models on financial data has a wide range of applications, including identifying event types from news, monitoring risk and compliance language, and answering questions over filings and earnings calls. In this setting, the main challenge is to improve task performance without degrading the generalization capabilities of the model. In this paper, we present an extensive set of ablations for fine-tuning Qwen3-8B with low-rank adaptation (LoRA). We compare three training regimes: (1) response-only LoRA, where the supervised response is the label alone; (2) single-adapter reasoning, where reasoning traces from a teacher model are included alongside the labels in the training loss; and (3) dual-adapter composite training, where separate LoRA adapters are trained on traces and labels. Response-only LoRA improves in-domain performance and leaves general capabilities near those of the frozen model. The in-domain change is largest for calibration and argmax accuracy. Under single-adapter reasoning, the in-domain metrics remain behind those of the response-only regime, while MMLU-Pro can rise slightly and the other retention benchmarks stay within ordinary variation. Dual-adapter composite training remains on the same side of this comparison and yields no further gain. Finally, we compress the teacher traces by summarizing them before training, and show that shorter summaries move calibration toward the response-only regime without reaching it. 

---
# Your Vision Foundation Model Is Secretly an EEG Foundation Model 

**Authors**: Guoan Wang, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17031)  

**Abstract**: Electroencephalography (EEG) foundation models are pretrained on thousands of hours of recordings, yet their frozen features transfer poorly to downstream tasks, and this transfer does not consistently improve with larger models or more pretraining data. We show that a vision foundation model that has never seen EEG is a stronger frozen EEG encoder than these specialized models, once the signals are rendered as images that a vision transformer can read. Our method, SeeEEG, keeps the vision backbone frozen and learns only how EEG is displayed to it. Each channel occupies a dedicated row of patches and is drawn either as a waveform at a fixed gain or as a scalogram whose frequency range spans the patch height, so that task-relevant patterns fall within individual tokens. The montage, filtering, and contrast parameters of this display are initialized from a standard clinical montage and optimized with the task loss through the frozen backbone, which amounts to at most 1,113 trainable parameters per view. We compare SeeEEG with seven EEG foundation models on eight clinical, sleep, affective, and brain-computer interface datasets, with every encoder frozen and paired with identical readout heads. With a frozen DINOv3 ViT-B, SeeEEG outperforms every EEG foundation model in balanced accuracy on all eight datasets and reaches a mean rank of 1.3 in Cohen kappa or AUROC, compared with 2.9 for the strongest baseline, REVE. The learned displays transfer without further optimization to the video encoder V-JEPA 2.1, which outperforms every EEG foundation model on both metrics on six of the eight datasets. Replacing DINOv3 with a randomly initialized ViT-B lowers the score by 15.2 points on average, so the ability to read rendered EEG comes from visual pretraining. 

---
# Skill Adapter: Personalized Modification and Online Routing of General-Purpose Skills 

**Authors**: Qijia Zhuang, Yanyi Gou, Wenbin Ma, Yuxiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.17007)  

**Abstract**: Large language model (LLM) agents use external skills to reuse task procedures and experience with tools. General-purpose skills describe shared execution procedures, while specific tool choices, operating practices, and output requirements vary across users. In long-horizon tasks, subgoals, available inputs, and environmental states change as execution progresses. Agents must select currently applicable skills from an expanding skill pool and prepare candidates for subsequent steps.
We propose Skill Adapter, which enables personalized modification and online routing of general-purpose skills. The system distills local adjustments from user requirements and execution feedback and stores them as Adapters linked to general-purpose skills. The routing agent selects general-purpose skills and applicable Adapters based on the current subtask, execution progress, and environmental feedback, incrementally maintaining a task-level online skill library that the main LLM reads on demand. The system preloads candidates for subsequent nodes and rechecks their applicability once actual inputs are determined. Reinforcement learning optimizes the two types of decisions using modification gains and routing outcomes, respectively.
We evaluate Skill Adapter on ALFWorld, WebShop, and SkillsBench. Compared with SkillRL, it improves task success rates by an average of 11.43 percentage points and reduces mean execution time on all three benchmarks. Skill Adapter retains its performance advantage when the skill pool grows to 1,000 skills. 

---
# AnalogLayoutWorld: An Open Benchmark for Agentic Analog Layout Design 

**Authors**: NovaSilicon Team  

**Link**: [PDF](https://arxiv.org/pdf/2611.16991)  

**Abstract**: Coding agents have demonstrated strong capabilities in software development, yet their ability to tackle long-horizon engineering tasks governed by physical constraints remains less understood. We study this question through analog circuit layout, which demands domain knowledge, geometric reasoning, and iterative interaction with design and verification tools. We introduce \textbf{AnalogLayoutWorld}, an open-source agent benchmark with 40 expert-reviewed tasks for end-to-end layout generation from fixed transistor-level netlists. It combines an integrated design environment with independent physical and post-layout electrical verification, as well as objective layout quality metrics co-designed with analog layout experts. Across 13 model--harness configurations, 91.1\% of attempts produce GDS layouts, but only 23.0\% satisfy the benchmark's acceptance criteria. We also evaluate Novara, an internally developed agent system, which achieves the highest reported aggregate score. Our analysis links performance differences to test-time scaling under fixed budgets, suggesting that reasoning efficiency and effective use of verification feedback help sustain design quality as task complexity grows. 

---
# Primer: An AI Framework for Instructor-Centered Educational Simulation Creation 

**Authors**: Ethan Mollick, Lilach Mollick, Ben Przystanski, LJ Ciccarelli, Natalie Bach, Laura Zarrow  

**Link**: [PDF](https://arxiv.org/pdf/2611.16986)  

**Abstract**: Educational simulations are a well-established pedagogical technique that gives learners opportunities to apply knowledge, respond to challenges, and reflect on their performance. High-quality simulations, however, are difficult to build, requiring instructor input, scenario design, character development, scaffolding, technical implementation, and repeated revision. This paper presents Primer, an open-source, LLM-agnostic AI-supported system that helps educators create adaptive educational simulations. Educators contribute their knowledge of the subject, lesson, and students, while generative AI produces responsive variations of practice for students. Primer uses a multi-agent system centered around executing educator's instructional intent through the entire learning experience. We hope that by releasing Primer as an open-source system, including its prompts and GitHub repository, we provide a foundation for development and research on educator-in-the-loop AI authoring, multi-agent educational workflows, and simulation-based pedagogy. Further, by allowing the use of various AI models, including local and open source models, we provide the opportunity for implementation under varied governance regimes. Primer points toward a model of educational AI in which instructors can create locally grounded practice opportunities based on their own materials, goals, examples, and knowledge of students. 

---
# Compiling Known Rules into Compact Neural Agents: Weight-Space Programming on Unseen Games 

**Authors**: Jens Lehmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.16981)  

**Abstract**: Weight-space programming (WSP) uses language models to construct neural network weights without training. We test whether known game mechanics can be compiled into compact agents that generalise to held-out levels, renderings and start states within a game family. Writers receive simulator source with held-out level data removed for 20 privately authored, unpublished ARC-AGI-3-style game families. With a fixed 57k-parameter recurrent architecture, the strongest writer reaches 56% held-out-level success with feedback, while the four main writers' Python players reach 97-100%. An exploratory free-architecture condition reaches 68-90%. Learning from support-level examples transfers much less well, but these learners lack the source information given to writers. Written networks have a program-like structure: 93% of parameters are zero, and 90% of the rest are small integers or powers of two. NumPy players under comparable computation caps reach 91%, versus the networks' 79%. Training the strongest writer's architectures from scratch yields about 1% success within our budget. Written fixed-network weights are fragile, and standard PPO removes most of their success. Prompts for noise robustness increase tolerance by about four orders of magnitude for two of three writers, but PPO makes larger updates and still removes most success. These results demonstrate rule-to-network compilation in this setting, while showing limited benefits over Python programs and no successful combination with the tested PPO fine-tuning. 

---
# The Value of Thinking: Learning When to Reason in Hybrid Decision Models 

**Authors**: Zhengxu Yu, Ke Xu, Meng Fang, Haitham Ammar, Weilin Luo, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16961)  

**Abstract**: Existing adaptive-reasoning methods shorten the explicit reasoning a model writes, or skip it by switching modes; either way the answer is still written out token by token. \emph{Rather than deciding how much to think}, we explore a different paradigm, \emph{zero-or-$T$ thinking}: a single model either answers from the implicit thinking inside one forward pass, generating no token, or performs full autoregressive reasoning at a cost of $T$ tokens when that is worth paying. We realise it in a \emph{hybrid decision model} (HDM), where one frozen backbone serves both modes and a lightweight \emph{self-routing block} chooses between them. The block estimates the \emph{value of thinking}, the utility the slow mode is expected to add over the fast one, from what that implicit thinking has already produced, and the slow mode runs only where that value exceeds a price $\lambda$ set per request. Where no per-decision label exists, as in episodic agentic tasks, we introduce \emph{Calibrated Decision Policy Optimisation} (CDPO), a reinforcement-learning recipe that trains the block from episode-level returns. We evaluate HDM on MMLU-Pro, AGIEval, GPQA-Diamond, JevBench, BANKING77 and MiniWoB. Specifically, on JevBench HDM-4B scores $0.857$ against Jev's $0.851$ ($+0.7\%$) at a median $0.179$\,s against $0.652$ ($-72.5\%$). On the agentic venue MiniWoB HDM-27B cuts token cost by $80.1\%$. 

---
# S1-Evo: Experience-Guided Evidence Competition Attacks on Typed System-1 Decision Models 

**Authors**: Kaixiang Wang, Junjie Li, Zhaojiacheng Zhou, Jiong Lou, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.16937)  

**Abstract**: Typed System-1 (S1) decision models are increasingly deployed in LLM agents for routing, moderation, and authorization to cut inference cost, yet untrusted context can still manipulate typed outputs and alter execution branches. Unlike autoregressive System-2 (S2) models, S1 interfaces score dynamic rules and options without token-by-token decoding, mismatching attacks targeting text continuation or explicit override commands. We propose S1-Evo, a self-evolving framework tailored to S1's contract-based option competition and margin feedback. S1-Evo perturbs untrusted fields via contract-aware evidence channels to induce \emph{evidence competition} with original task evidence while preserving fact invariance. Offline on seed tasks, it optimizes candidates via feedback search guided by directional margins and re-verification, distilling verified flips into experience cards. At test time, S1-Evo retrieves relevant cards to construct perturbations for unseen tasks. Across four benchmarks and three S1 models, S1-Evo reaches 71.99\% mean ASR and 57.47\% high-confidence ASR on \texttt{jev-1.14}, and causes up to 24.00\% E2E agent violations. 

---
# Universal Infectious Disease Forecasting with Mantis 2 

**Authors**: Carson Dudley, Reiden Magdaleno, Yutong Bi, Anthony Valencia, Xiaofeng Liu, Marisa Eisenberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.16935)  

**Abstract**: Public health agencies need to anticipate how outbreaks will unfold, when they will peak, and how long demand on health services will remain elevated. We previously developed a foundation model, Mantis, which was trained entirely on simulated outbreaks---enabling us to generate the diversity and scale of training data needed to support forecasting across diseases. However, our previous work left unresolved when simulation improves predictions and how it should complement observational data.
Here, we varied the balance of simulated and observed training data while holding forecasting tasks fixed and varying the available surveillance history. We found that simulation-rich training was particularly valuable with limited local history, and its relative advantage diminished as more observations became available. We also found that simulation-based training reduced exceptionally large forecast errors, suggesting that mechanistic knowledge can help compensate for limited local evidence.
Building on these findings, we developed Mantis 2, which combines simulated and observational training to jointly forecast up to six variables sixteen weeks ahead. Across 30 forecasting tasks, Mantis 2 achieved the best aggregate probabilistic accuracy over short, medium, and long horizons among the machine learning, statistical, and baseline models evaluated. It missed roughly 30\% fewer outbreak peaks than the strongest comparison model, with successfully anticipated peaks identified more than six weeks in advance on average. We publicly release Mantis 2, enabling public health agencies and researchers to forecast disease outcomes without developing a separate model for each disease or location. 

---
# CADET: Construction of PCB Routing Instances with Executable Demonstrations 

**Authors**: Hyungseok Song, Won-Seok Choi, Seohui Bae, Han-Seul Jeong, Junseok Park, Youngjoon Park, Soonyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.16913)  

**Abstract**: Printed circuit board (PCB) routing is the stage of physical layout that finds paths connecting the placed pads of each net. Public datasets in this domain are few, and most contain only the final states of routed boards, not demonstrations of the electronic design automation (EDA) engine actions that produce them. To close this gap, we propose CADET (Constructing executAble Demonstrations for Engine-grounded Training), which jointly generates synthetic boards and demonstrations built from EDA engine actions. The resulting RouteDemos provides 100,000 synthetic boards routed with zero design rule violations (DRVs), together with their demonstrations. Their distributions over 17 statistics resemble those of 142 real reference boards. CADET trains a routing policy on these demonstrations with behavior cloning and fine-tunes it with reinforcement learning (RL). On real test boards, the policy raised the rate of complete DRV-free routing from 38.0% to 67.2% over an RL-only baseline. This rate was comparable to that of the open-source autorouter FreeRouting and exceeded those of the other baselines, including GPT-6 Astra. We release the dataset and its generation code as open source at this https URL. 

---
# A Modular Framework for ESG Prediction through Domain-Specific Language Modeling and Graph Learning 

**Authors**: Sofia Baiocco, Gianluca Bonifazi, Christopher Buratti, Michele Marchetti, Camilla Mazzoli, Federica Parlapiano, Giulia Quaglieri, Davide Traini, Domenico Ursino, Luca Virgili  

**Link**: [PDF](https://arxiv.org/pdf/2611.16831)  

**Abstract**: Environmental, Social and Governance (ESG) scores are increasingly used to evaluate corporate sustainability, supporting investment decisions, ensuring regulatory compliance and assessing risk. However, current ESG evaluations largely rely on proprietary methodologies that often produce inconsistent ratings. This motivates the development of data-driven prediction frameworks. In this paper, we present ESGPred, a modular framework for predicting ESG scores by combining domain-specific language modeling with graph-based learning over a network that models the relationships among the companies being evaluated. In addition to presenting the rationale behind ESGPred and detailing its ESG score prediction process, we show that ESGPred can effectively predict both overall ESG ratings and their individual Environmental, Social, and Governance components. We evaluated ESGPred using a dataset of 5,266 sustainability reports published by 2,134 publicly listed companies between 2020 and 2023. In this evaluation, we used ESG ratings provided by LSEG Data & Analytics as ground truth. Our experimental results demonstrated that integrating graph-based relational learning improves prediction performance over purely text-based baselines. 

---
# Scale-Induced Semantic Shift in Visual Prompt Injection Against VLMs and Web Agents 

**Authors**: Yiheng Zhao, Runze Ji, Chenyu Yi, Yi Yu, Bingquan Shen, Alex Kot, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16744)  

**Abstract**: Visual content in multimedia systems may be resized, compressed, or re-rendered before it is interpreted by a vision-language model. We study a security failure mode in which these transformations change the machine-readable semantics of an image: a textual instruction is not recovered at a $512\times512$ presentation resolution under our evaluation protocol but becomes recoverable after aggressive downsampling. We formulate this behavior as Scale-Induced Semantic Shift (SISS) and develop a closed-loop black-box framework that combines dual-scale verification, selective self-hardening, shallow-layer low-rank adaptation, and pixel-space noise regularization. On a fixed 300-case benchmark, the third SISS refinement round obtains a $58.75\%$ average attack success rate across four closed-source vision-language models, compared with $41.08\%$ for the strongest generation baseline. A controlled sweep bounds the effect to aggressive compression: average success decreases to $17.68\%$, $6.85\%$, and $5.60\%$ at $128\times128$, $224\times224$, and $336\times336$, respectively. In a downstream web-agent study, the generated images increase hidden-instruction perception, but absolute behavioral compromise remains about $4\%$. Generic input transformations provide little protection in a separate defense protocol, whereas the proposed ScaleGuard consistency detector reduces non-adaptive success from $61.4\%$ to $24.3\%$ and one-round defense-aware success from $42.5\%$ to $15.0\%$, with no observed rejection among matched clean controls. The results characterize a bounded but reproducible semantic inconsistency in multimedia preprocessing and motivate scale-consistent visual-language inference. 

---
# Mind the Accuracy-Approval Gap: How Users Evaluate Sycophantic AI in Everyday Tasks 

**Authors**: Wajiha Naveed, Brandon Fain  

**Link**: [PDF](https://arxiv.org/pdf/2611.16722)  

**Abstract**: Large language models (LLMs) often exhibit sycophancy, a tendency to agree with users, flatter or affirm them, and preserve their face, even when independent judgment or critical feedback would be more useful. Prior work has largely evaluated sycophancy through benchmark evaluations, but less is known about how users perceive and respond to sycophantic AI in everyday tasks. We combine benchmark evaluations with two large-scale user studies. We study four personas defined through system prompts: objective (instructed to prioritize accuracy over agreement), soft-sycophantic, hard-sycophantic, and baseline (no persona-specific system prompt). Benchmarks show that even baseline exhibits substantial sycophancy, agreeing with users' stated views 71.7% of the time on NLP Survey, compared with 54.8% for objective and 89.5% for hard-sycophantic. Under conversational pressure on TruthfulQA, accuracy ranged from 81.9% for objective to 59.6% for baseline and 8.7% for hard-sycophantic. Participants were assigned to one of the four personas, and either discussed contested policy topics or collaborated with AI on email writing. Despite its higher accuracy under pressure, objective AI was rated significantly less respectful and satisfying than the other personas, and less trustworthy than baseline. Baseline and sycophantic personas did not differ significantly. Persona did not significantly affect how often participants changed their opinions, although participants who discussed with objective AI had greater odds of reduced confidence. Objective AI prompted greater writing expansion than soft-sycophantic AI, but final writing quality did not differ significantly across personas. These findings highlight a tension between accuracy and user approval: personas with substantially lower accuracy under conversational pressure received more favorable ratings for respect and satisfaction than objective AI. 

---
# Acoustic Respiratory Phase Detection on Mobile Devices 

**Authors**: Julian Szymanski, Tomasz Sankowski, Piotr Sulewski, Iwo Czartowski, Javier Medina Quero  

**Link**: [PDF](https://arxiv.org/pdf/2611.16710)  

**Abstract**: Breathing monitoring is important for well-being, stress regulation, and guided breathing exercises, yet existing approaches often rely on contact-based sensors or controlled environments. This paper presents a non-invasive framework for respiratory phase detection using acoustic signals recorded on commodity smartphones. The proposed hybrid CNN-Transformer architecture combines local spectral feature extraction with long-range temporal modeling to capture the cyclic structure of breathing. The model is evaluated using a Leave-One-Subject-Out (LOSO) protocol on the Foundation Dataset and on an independent unseen test set comprising over 1300 breathing cycles and 2 hours of continuous recordings. The proposed 3-class formulation achieves 72.85% frame-level accuracy and 96.6% cycle-level exhalation detection efficiency in the LOSO evaluation, while achieving 76.94% frame-level accuracy, a Cohen's Kappa score of 0.65, and 99.2% cycle-level exhalation detection efficiency on the unseen test set. Results suggest that accurate respiratory monitoring may be achievable under data-constrained conditions, indicating that appropriate temporal modeling can learn transferable respiratory representations and generalize to previously unseen participants. An end-to-end ONNX deployment pipeline enables privacy-preserving on-device inference with latency below 20 ms, supporting real-time monitoring on mobile and wearable devices. 

---
# When Expert Disagreement Hurts: Prestige and Selectivity in LLM Decision Revision 

**Authors**: Yupeng Tang, Mingfeng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.16694)  

**Abstract**: When language models revise a decision after another agent disagrees, neither reversal counts nor final accuracy shows whether they are fixing mistakes or abandoning correct answers. We introduce a controlled two-pass audit that holds the question, the evidence, and the model's initial answer fixed and varies only the revision context: neutral reflection, anonymous disagreement, or disagreement attributed to a domain expert. For binary decisions the accounting is exact: revision raises accuracy only if it is selective enough, reversing wrong answers more than $a/(1-a)$ times as often as correct ones, where $a$ is initial accuracy (nine times at 90%). Across five models and 1,989 evidence-grounded decisions in medicine, science, and law, no model clears this bar under expert disagreement, and accuracy falls by up to 16.1 points. The expert label, a prestige cue, adds reversals for four of the five models, yet its effect on accuracy ranges from a 5.2-point loss to a 2.4-point gain, so reversal counts are a poor proxy for harm. With identical arguments from both sources, the label still raises reversal for three proprietary models and in a pooled analysis of four newer models. Models also respond to stated source reliability. Revision pipelines should therefore be evaluated with paired, conditional metrics, per model and protocol. 

---
# Natural Gradient Learning in Predictive Coding through Lateral Circuits 

**Authors**: Lhea Beumer, Sander de Haan, Niclas Pokel, Pau Vilimelis Aceituno, Benjamin F. Grewe  

**Link**: [PDF](https://arxiv.org/pdf/2611.16666)  

**Abstract**: Predictive coding (PC) trains neural networks through local energy-minimization dynamics without depending on a global error signal that backpropagation (BP) requires. Although PC is a promising alternative algorithm for efficient learning, the computational cost of inference dynamics currently outweighs the resulting learning benefits in practice. In this work, we take inspiration from optimization theory to extract more learning benefit from each inference phase. We introduce Natural PC, which augments standard PC with a lateral circuit, enabling exact and efficient natural gradient learning locally in space and time. Our lateral circuit learns the activity correlations that determine the Fisher information matrix and implicitly computes its inverse, avoiding expensive matrix inversion. Across all tested image reconstruction benchmarks, Natural PC achieves state of the art in terms of sample and compute efficiency compared to existing PC approaches. Most notably, Natural PC scales to ImageNet and matches the performance of BP trained with AdamW and various second-order optimizers at an order of magnitude lower computational cost while also using two orders of magnitude fewer training samples. Our results demonstrate that local natural gradient learning makes PC competitive with BP in both sample efficiency and computational cost in reconstruction tasks. 

---
# InertiaMPDD: Multimodal Personalized Depression Detection via Emotional Inertia-Inspired Dynamics Modeling 

**Authors**: Changzeng Fu, Songyang Wang, Chang Gao, Xinyi Zhang, Peilin Li, Mo Meng, Jiarui Wu, Mingze Li, Hongwei Ding, Shiqi Zhao, Yong Li, Lang He  

**Link**: [PDF](https://arxiv.org/pdf/2611.16647)  

**Abstract**: Multimodal depression screening estimates risk from speech, visual behavior and language. Current methods mainly predict labels, with limited evidence of departures from a non-depressed reference. Recording-level supervision also leaves fine-grained emotional dynamics unannotated. Motivated by emotional inertia, the persistence of affect, InertiaMPDD learns healthy next-state patterns using a world-model-inspired latent predictor. Using pretrained multimodal representations, it predicts from observed history and context. Reference deviation, residual decay rate and a Jacobian spectral descriptor form reference-derived emotional dynamics descriptors, evaluated as candidate proxy biomarkers for standalone screening and personalized decision correction. Across five cohorts from four datasets, fusion reaches 98.41% F1 on D-Vlog and 84.98% on LMVD. Standalone descriptors outperform fixed additive fusion on MPDD but discriminate weakly on AVEC2014. These findings distinguish reference informativeness from fusion benefit and reveal context-dependent utility. 

---
# Safety as a Budget: Calibrating the Control Channel of Constrained GRPO for Vision-Language Models 

**Authors**: Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.16628)  

**Abstract**: Reinforcement learning can teach vision-language models (VLMs) to decline harmful image-text requests, but how safe the result becomes is set indirectly, through reward weights found by trial runs. Stating safety as budgets on the rate of harmful assistance and the rate of over-refusal, enforced by Lagrange multipliers inside GRPO with an off-the-shelf scorer as the cost, is the natural fix, and it fails twice. The multiplier compares the scorer's mean with the budget, and scorers built for ranking get the mean wrong: a fixed-point analysis shows that the mean bias, not the budget, then decides whether the multiplier settles, saturates or switches off. And standard GRPO standardizes the penalized reward within each group, which cancels the multiplier before it reaches the gradient. Two-channel constrained GRPO repairs both: the base reward keeps the raw score for the within-group comparison, the penalty and the constraint estimate pass through a monotone map fitted on judge-labeled training responses, and the advantage keeps the multiplier. On a Qwen3-VL-2B policy with a training-free readout of an open 4B model as the scorer, held-out harm on 1,037 adversarial prompts falls from 8.9% under standard GRPO to 4.8%, lower in every seed, where the uncalibrated budget gives 7.6%; over-refusal stays at most 2% on 400 benign queries and MMBench is unchanged. The gain is largest where the intent sits in the image (10.1% -> 5.1%) and grows on the unseen HADES benchmark (28.9% -> 12.8%). Order-preserving distortions of the scores reproduce each predicted controller state in every seed, and the results hold under a map fitted on another judge's labels, an image-reading judge and a human audit. 

---
# Folio: Recoverable Multi-Resolution Context Folding for Long-Horizon Language Agents 

**Authors**: Yubo Ruan, Tao Peng, Jiao Zhang, Chuanguo Ma, Lei Xu, Han Zhou, Wenwu Wang, Xiafei Lv, Xingjie Zhou, Guiju Fan, Hao Wang, Fenghui Zhang, Mao Miao, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.16627)  

**Abstract**: Long-horizon language agents accumulate tool results and replies until the history no longer fits the context window. Current agents shrink this history by having a model summarize it, so each compaction rereads and rewrites the history it replaces, and cache-aware triggers decide when to compact while taking the size of the result as given. We present Folio, a context manager that records, folds, and recovers history through one set of handles. While the agent runs, Folio records three linked levels: a short turn note the model writes as part of its ordinary reply (L1), an event manifest of every call written by the runtime (L2), and the captured source records (L3). Because these views already exist, the runtime folds the context programmatically, selecting coarser views for older turns. The fold prints the handles that index L2 and L3, so recovery tools that read bounded pages, compute over full text in files, and keep findings in a persistent dossier reach the evidence the agent needs with little added input. The fold trigger prices an explicit landing point that includes the retained budget. On 75 LOCA-bench tasks with DeepSeek-V4-Pro behind every system, Folio solves 66 tasks against 54 for the same runtime with model-written compaction and 49-59 for Claude Code, Codex, and SoL-Pi, in about half their working time, at 2.4 times lower cost than that runtime; each fold takes a median of 0.6 seconds. 

---
# Studies on Sign Language Retrieval in Low-Resource Dictionary Settings 

**Authors**: Ushnish Sarkar, Vineet Kumar Rakesh, Bhaswar Chattopadhyay, Tapas Samanta  

**Link**: [PDF](https://arxiv.org/pdf/2611.16621)  

**Abstract**: Most sign languages have no annotated corpus, only small dictionaries with one or a few example videos per sign. This study examines how well a pose encoder trained on a single sign language serves such languages without any target-language training.
An encoder trained with metric learning on American Sign Language (ASL-Citizen) is evaluated frozen on six datasets of Turkish, Russian, Indian, Argentinian, Pakistani and Sri Lankan sign languages, with one or four clips per sign enrolled and, where
signer identities are available, the query's signer excluded from the gallery. With one enrolled clip, the encoder retrieves the correct sign first in 36.4-80.7 % of queries, against 16.0-69.6 % for the best learning-free method (angle-based dynamic
time warping) and a six-target mean of 39.3 % for an ASL-trained OpenHands graph network. In a crossed ablation of the encoder (embedding geometry, aggregation of body and hand embeddings, training objective, and attention between articulators), two
choices matter: a per-pair InfoNCE objective outperforms supervised contrastive and cross-entropy training, and blocking attention between body, left hand and right hand inside the encoder raises accuracy by up to 5.1 points. The best configuration
reaches 68.0 % one-shot accuracy (59.0 % for the reference encoder). On CISLR, an Indian Sign Language dictionary of 4,765 words with one video per word, the encoders find the correct word first for 17.3-19.0 % of queries, above the 16.8 % of the
original RGB baseline. Over eight seeds with the joint encoder, spherical, Euclidean and hyperbolic embedding spaces differ by at most 1.6 points, except the hyperbolic token-attention head trained with InfoNCE, which is 1.4-2.0 points worse, so the
geometry is a minor factor. 

---
# A Dual-Branch Graph and Convolutional Network for Interpretable Movement EEG Classification 

**Authors**: Maryam Ostadsharif Memar, Behzad Nazari  

**Link**: [PDF](https://arxiv.org/pdf/2611.16609)  

**Abstract**: Decoding upper-limb movements from scalp electroencephalography (EEG) is an important problem for brain-computer interfaces (BCIs), but many deep learning models provide limited insight into the brain regions underlying their predictions. We propose a two-branch architecture that combines an LGGNet-inspired local-global graph branch with an EEGNet-style convolutional branch for movement-versus-rest classification. The graph branch groups 61 EEG electrodes into anatomically informed regions and learns their spatial relationships, providing a direct representation of the regions emphasized by the model. We evaluate the proposed architecture using leave-one-subject-out cross-validation on the Ofner upper-limb movement dataset. The model achieves a mean F1-score of 0.74 +/- 0.06 across subjects. Beyond classification performance, we analyze the learned graph structure to identify the regions with the strongest connectivity. Across all held-out subjects, the highest connectivity is consistently observed in the Parietal Left Low region, corresponding to electrodes P1 and P3. This region is opposite to the performed right-arm movements and lies within the posterior parietal area previously associated with upper-limb movement processing. These results show that incorporating a region-based graph into the decoder can provide an anatomically interpretable representation while maintaining effective movement classification, offering a practical approach toward more transparent EEG-based BCIs. 

---
# Singularity-Bench: Can Agents Build Agents? 

**Authors**: Yang Zhou, Krish Agarwal, Zhuoming Chen, Zerui Wang, Zhenyu Zhang, Wenshuang Li, Xinpeng Wei, Zhichen Zeng, Jingyu Liu, Jue Wang, Sen Wu, Ziheng Jiang, Zhen Dong, Han Cai, Binhang Yuan, Ying Sheng, Ce Zhang, Bo Li, Beidi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.16591)  

**Abstract**: Frontier AI agents increasingly succeed on realistic, long-horizon tasks and are beginning to improve the systems through which they operate. Yet existing benchmarks often restrict agents to narrow action spaces, use simplified environments, or rely on verification methods that are unreliable for evaluating arbitrary agent-generated systems. We introduce Singularity-Bench, an open-ended benchmark for autonomous AI development with six end-to-end tasks spanning pre-training, mid-training, post-training, inference, the agent harness, and cluster orchestration. Each task requires agents to implement core systems from a minimal starting point and evaluates them against mature human-engineered baselines on identical hardware. Additionally, through extensive red-teaming covering 9 of the 10 categories in the OWASP Top 10 for Agentic Applications, we expose reward-hacking vulnerabilities in the conventional Harbor framework and develop Breakwater, an extension of Harbor that blocks all attacks tested in our study. Across 11 agents built from 7 frontier models and 2 harnesses, Opus 5.5 with Claude Code matches or outperforms the human baseline on all six tasks with a Singularity-Index of 1.25, suggesting it may have the skills needed to reach the singularity. The remaining agents average only 0.58. Our vision is for Singularity-Bench to track how much of AI development agents can already automate and when they may transition from assisting to taking end-to-end responsibility for building stronger successors. 

---
# AstroMemBench: A Benchmark for Memory Systems in Embodied Spacecraft Agents 

**Authors**: You Wu, Guhong Zhang, Hongwei Li, Xinyue Hong, Yuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2611.16590)  

**Abstract**: Long-term spacecraft autonomy requires embodied agents to use operational history under communication delays, time-critical operations, environmental uncertainty, complex physical interactions, power and computational constraints, and safety and reliability requirements. We introduce AstroMemBench and a Scenario-Challenge-Capability design method that maps these requirements to executable memory tests. The benchmark covers 16 memory subchallenges through 140 templates and 2240 instances across nine tasks in planetary surface operations, on-orbit servicing, and deep-space exploration. A deterministic environment scores actions for task completion, state changes, safety, and resource use. Ablations under nine conditions show that the benchmark is sensitive to how accurately agents recover and use historical information and to differences in memory-system design. Main experiments compare six configurations. Mem0, Chain-of-Memory, LightMem, and retrieval-augmented generation (RAG) improve same-session mean scores over native Pi Agent, with no clear advantage of the three memory systems over RAG. After a session restart, Mem0 outperforms RAG by 12.25 points. Benefits favor instruction retention and result reuse over ongoing state tracking and energy reservation. AstroMemBench supports the design, development, and selection of memory systems for embodied spacecraft agents. AstroMemBench 1.0 and its evaluation code are publicly available: this https URL. 

---
# Large Language Models as Digital Twins: a Distribution-Aware Questionnaire-Based Evaluation 

**Authors**: Francesco Di Cursi, Chiara Boldrini, Andrea Passarella, Marco Conti  

**Link**: [PDF](https://arxiv.org/pdf/2611.16586)  

**Abstract**: This paper investigates whether Large Language Models can answer a personality questionnaire as a specific individual would. Given a user's social-media posts, the model completes the BFI-2-60 Big Five questionnaire as that user. The predicted responses are converted into trait scores and compared with the user's actual scores. We evaluate GPT-4, Gemma, Llama 3.1, and Mistral on the MyPersonality dataset. To determine whether predictions reflect information from the user's writing, we compare them with answers produced without user text and with values derived from general population tendencies. We measure prediction error across all users and separately for individuals with low, intermediate, and high trait scores. We also test whether an LLM can express an explicit personality profile when given only the target Big Five scores, without user text, and examine how predictions change as more text is provided. The results show that user text does not consistently improve performance. Models perform better near the middle of the trait distribution, while at the lower and upper ends they often remain close to central values or perform well at one end but poorly at the other. When given an explicit profile, GPT-4 most clearly translates trait information into consistent questionnaire responses, particularly for Extraversion and Neuroticism. Increasing context does not systematically reduce error: open-source models fluctuate more, whereas GPT-4 remains comparatively stable. Overall, the evaluated models do not consistently reproduce the questionnaire behavior of specific individuals. 

---
# SPARC: Accurate Photonic Matrix Multiplication via Representative Vectors 

**Authors**: Yasuhiro Fujiwara, Tatsuhiro Nakamori, Hideyuki Kawashima  

**Link**: [PDF](https://arxiv.org/pdf/2611.16561)  

**Abstract**: Photonic neural network accelerators have emerged as a promising solution for reducing the power consumption of deep neural network (DNN) inference by executing matrix multiplication on analog photonic hardware. However, their inference accuracy is limited by low-bit quantization required for photonic processing and by analog photonic noise. This paper proposes an approach for improving inference accuracy by exploiting the high similarity among activation vectors in DNNs. Specifically, our approach constructs a representative vector for each group of similar activation vectors, computes its inner product digitally with high precision, and offloads the residual inner products to the photonic accelerator. Since the activation vectors within each group are highly similar, the residual vectors have much smaller norms, thereby reducing the effects of quantization errors and analog photonic noise. Furthermore, our approach preserves the photonic inner-product computation pipeline, allowing seamless integration into existing and future photonic accelerators without hardware modification. Experimental results on six representative DNN models demonstrate that the proposed approach can improve inference accuracy across diverse network architectures and application domains. 

---
# AI has its own style in mathematical proving 

**Authors**: Xiaokai Li, Ziyu Zhou, Qibin Cheng, Yangcheng Wu, An Zeng, Ying Fan  

**Link**: [PDF](https://arxiv.org/pdf/2611.16527)  

**Abstract**: As large language models increasingly generate mathematical proofs, whether they develop a proving style distinct from that of human mathematicians remains an open question. This question is particularly pressing as mathematicians increasingly voice concerns that proofs generated by large language models are difficult to interpret. Here we ask three large language models to independently reprove all 465 propositions of Euclid's \textit{Elements} and compare their citation choices against the human-proof baseline through Jaccard similarity and complex network analysis. We find that AI-generated proofs differ from human proofs, and this difference is stable and reproducible: mean pairwise Jaccard similarity among AI runs (~0.68) is nearly twice that between AI and human proofs (~0.37), and cross-run classification consistency is also high. By examining citation structure in detail, we identify the concrete signature of this difference: approximately 40% of AI citations point to foundational elements (definitions, common notions, and postulates), compared with only 13\% in human proofs; even among theorem-to-theorem citations, AI tends to cite more distant results. Roughly half of AI's extra citations trace to the upstream dependencies of the human proof. Together, these findings suggest that the AI proving style is a tendency to construct arguments on its own from foundational elements rather than directly citing later, more established results, which we term "building from the ground up". Our findings offer a new perspective on how AI systems organise and invoke mathematical knowledge, with implications for understanding the interpretability of AI-generated proofs. 

---
# Lost in Delegation: Studying the Effectiveness of Task Orchestration in Generative Search 

**Authors**: Tamara Stojanovska, Elisabeth Kirsten, Leon Swazinna, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2611.16526)  

**Abstract**: LLMs have quickly evolved from models used in isolation to agentic systems that delegate tasks to external tools and other LLMs to solve complex assignments. This agentic augmentation has led to impressive capability gains. However, when and how delegation improves performance remains poorly understood. For example, it is unclear whether delegation helps when tasks involve post-cutoff information that the orchestrating LLM cannot reliably resolve from its own knowledge. In this work, we use purpose-built datasets with pre- and post-cutoff splits to compare various delegation modes. We find that delegation provides no consistent accuracy improvement over non-delegated setups, with effects varying across models and tasks. When tasks involve knowledge unavailable to the orchestrator, delegation can substantially degrade accuracy, while direct tool use with the same LLM performs much better. Our analysis identifies distorted delegated queries, inefficient tool use, and failures to preserve and use information between agents as recurring failure modes. These findings suggest that effective orchestration requires both well-calibrated delegation decisions and effective exchange of information between agents. 

---
# SignRot: LLM Quantization with Massive Outlier-Aware Sign-Adjusted Rotation 

**Authors**: Jaehun Gim, Jaewoo Kim, Gyuwan Kim, Seo Yeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.16502)  

**Abstract**: While quantization compresses LLMs and accelerates inference, outlier activations impede efficient low-bit representation. In particular, massive outliers cause substantial performance degradation. Although rotation-based methods that rely on randomized Hadamard transforms for online inference aim to mitigate this, they can produce fragmented distributions with empty bins when a massive outlier dominates a token's activation. This fragmentation severely undermines uniform quantization performance. Moreover, massive outliers concentrate in the down-projection input, where the preceding nonlinearity requires the rotation to be applied at inference time. This constraint prevents training-based approaches such as SpinQuant from learning the online rotation at this position, as a learned dense matrix cannot use the Fast Walsh-Hadamard Transform and would require a full matrix multiplication during inference. To address this, we propose SignRot (Sign-adjusted Rotation), which identifies massive outlier channels and constructs a sign adjustment vector that unifies the signs of the corresponding Hadamard row, collapsing the bimodal distribution into a unimodal profile amenable to quantization. Since the aligned outlier shifts the activation mean away from zero, we pair SignRot with asymmetric activation quantization, whose zero-point absorbs this shift. As a lightweight, drop-in improvement compatible with any Hadamard-based quantization method, SignRot requires only a single calibration forward pass and adds only element-wise sign flips at inference, without an additional matrix multiplication. Applying SignRot to QuaRot achieves a WikiText2 perplexity of 5.88 and an average zero-shot reasoning accuracy of 72.73, surpassing SpinQuant on the LLaMA3-70B model. Code is available at this https URL. 

---
# Recalibrating a Typed-Decision Model in an Afternoon: A Hands-On Recipe for Laya's Confidence Scores 

**Authors**: Mohammed Nayeem Khan, Aliyas Ahmad Misri  

**Link**: [PDF](https://arxiv.org/pdf/2611.16499)  

**Abstract**: A typed-decision model such as Laya reads a text and a list of options that the caller defines, for example ticket categories, and returns a probability for each option. Code around it compares the top probability with a threshold to answer, abstain or escalate to a person. Earlier work reports, for a Turkish model, that temperature scaling can raise held-out calibration error (ECE). This tutorial gives a five-step recipe (measure, diagnose, fit, validate, deploy) in 130 lines of numpy with standard methods, none new. On 1,065 synthetic, LLM-labelled items the released Laya checkpoint has a pooled ECE of 0.073 but a bucket-macro ECE of 0.143 (0.127-0.184), the mean over buckets (questions grouped by type and option count): the 12-option bucket is over-confident by +0.22, the three-to-five-option bucket under-confident by 0.05, and they cancel. One global temperature makes pooled ECE worse (0.173); a cross-fitted temperature per bucket brings macro-ECE to 0.073 (0.078 over 10 fold assignments) and reaches 0.089 with 10 labelled items per bucket, but holding out one of seven evaluation sets gives 0.099. After the refit, a gate aimed at 10% error answers 46.3% of items (10.5% wrong) against 6.9% on the published confidences, but one cut-off per bucket on the published confidences answers 45.4%, so most of the gain comes from separating buckets. Confidence-only maps (Platt, histogram, isotonic) need 20 or more labelled items per bucket to approach the temperature; with 320 per bucket the lowest macro-ECE belongs to Platt. In these files the retrained checkpoints store no per-bucket table and the stored 12-option value 0.1006 is clamped to 0.5 at runtime; the refit helps 31 of 33 checkpoints on random folds and 21 with a set held out. Fit and test share one generator and LLM labels, so these are findings about these checkpoints and this data. 

---
# How many LLM-labelled rows does a decision encoder need, and how wrong can the labels be? 

**Authors**: Mohammed Nayeem Khan, Aliyas Ahmad Misri  

**Link**: [PDF](https://arxiv.org/pdf/2611.16498)  

**Abstract**: A decision encoder such as Laya reads a question with named options (for example, which department should handle a ticket) and a text, and picks an option. It reads both in one forward pass and returns a probability for every option, without writing any text. We ask how many LLM-written training rows such an encoder needs and how wrong those labels may be. We ran 19 fine-tunes of Laya (ModernBERT-large) on nested subsets of a 2,818-row pool and scored them on 1,065 held-out items from the same generator, so every test item is in-distribution and no out-of-family test was run. Accuracy rises from 69.4% before fine-tuning to 74.3% with 5% of the rows, 84.8% with 50% and 86.9% with all rows. The curve flattens, and a saturating fit with bootstrap intervals cannot pin down its plateau. Seed spread at 20% of the rows (s.d. 1.6 points) exceeds some of the steps between sizes. Giving small subsets as many updates as the full pool gains 1.9 points at 5% and 1.9 points at 20% of the rows (one seed; p=0.14 and 0.13), and at 20% that is below the 2.3-point difference between two ordinary seeds. A single run with 10% symmetric label noise shows no detectable change (-0.2 points, 95% interval -1.7 to +1.4); 20% costs 3.4 points and 40% costs 8.7 points. Confusable-pair noise at 20% costs 7.4 points, and under noise the model moves from over-confident to under-confident. An independent labeller disagrees with the gold labels on 3.5% of held-out items, and about half of its choice disagreements (16 of 33, leaving each item out of the partner map) name the partner option that the confusable-pair noise uses. Laya is +0.9 points ahead of a TF-IDF classifier trained on the same rows at full size (the interval includes zero) and +21.1 points ahead at 5% of the rows. Everything is synthetic, LLM-labelled and from one backbone, one trainer and mostly one seed. 

---
# ReA-Bench: Benchmarking Responsible Agent Auditing 

**Authors**: Zipeng Ye, Shenyu Dai, Yong Xie, Kun Wang, Jiyang Guan, Zhikai Chen, Jiaqi Weng, Jialing Tao, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2611.16493)  

**Abstract**: As agents perform tasks autonomously in local and cloud environments, auditing their execution trajectories can help service providers identify risks and improve agent harnesses and foundation models. Our analysis of production and red-team trajectories shows that some risks are repeatedly missed across auditing configurations. Overall recall obscures which risks are missed and whether higher recall comes with more false positives on legitimate behavior. Existing benchmarks offer limited insight into these failures. Generated trajectories can contain inconsistent tool calls and responses, conspicuous risky actions, or repetitive attack patterns. Grouping results by risk type also makes it difficult to tell whether auditors overlooked evidence, accepted an unsupported conclusion, or lacked information needed to make a judgment.
We introduce ReA-Bench, a diagnostic benchmark for auditing responsible agent behavior. It organizes auditing challenges into six tiers: internal safety knowledge, contradictions within trajectories, evidence retrieval in long contexts, verification of sources and reasoning, indirect steering, and missing information. It includes challenging benign cases in which actions resemble risky ones but are consistent with the task and its authorization. It also provides paired trajectory versions to test whether auditors detect the same risks more reliably when the trajectory is shortened or missing information is supplied. Most samples are drawn from real production trajectories, with scenarios manually selected and adapted to preserve realistic tasks and risks that could plausibly arise during execution. 

---
# Time and Space: A Dual-Axis, Evidence-Grounded Context Substrate for Personal Agents 

**Authors**: Haitao Cheng, Jerry Hu, Longdong Mao, Dong Huang, Dongming Li, Lijuan Zhang, Mengting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16492)  

**Abstract**: We argue that a binding constraint for personal agents is not model capacity but the form in which context is represented: the agent does not know which entity a phrase denotes, when a fact held, or which source record licenses a claim. We call the missing representation spatio-temporal context (STC) and define it on two axes. The temporal axis carries event time, validity intervals and ingestion time. The spatial axis is a relational space in which entities are points, typed relations are edges, and distance is semantic rather than geographic; physical places are one entity type among many. Every relation is anchored to immutable source records, and every answer must be closed over those records. We formalise the substrate and six invariants, a context operator that composes temporal intersection with relational expansion, an ambiguity-preserving resolution state machine, a conflict detector that converts mutually exclusive facts into user adjudication rather than silent overwrite, and a four-regime consumption taxonomy spanning query, insight, triage and proactive reminder. We describe a local-first prototype of the retrieval layer built on SQLite with on-device embeddings and hybrid lexical-dense recall, report measurements from a deployed store of 341 documents, 340,079 chunks and 190,271 embedded vectors, and specify, but do not yet implement, the analytic layer on which the framework depends. We close with an evaluation protocol with per-invariant metrics for evidence coverage, temporal correctness, conflict precision and interruption cost. The proposal targets the representation of context rather than the volume of it. 

---
# HeMeR: Heterogeneous Memory Reconciliation for Embodied Agents via Structured KV Reuse 

**Authors**: Saehun Chun, Wonje Choi, Jinwoo Jang, Taeyoon Kwack, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2611.16487)  

**Abstract**: Embodied agents make decisions by combining information from heterogeneous memory modules, including declarative memory for explicit knowledge, procedural memory for learned routines, and working memory for current observations. During interaction, the memory contents required for decision making change with the agent's state, goal, and task progress, while many contents remain shared across consecutive decisions. However, existing memory modules are typically accessed through separate interfaces, so each updated memory context is often processed as a new input to the foundation model. This prevents selective key-value (KV) cache reuse for unchanged memory contents and forces full KV cache recomputation even when much of the memory context remains valid. We introduce HeMeR, an inference framework that performs KV reconciliation between dynamically updated memory contexts and reusable KV states. HeMeR tracks declarative, procedural, and working memory contents across action decisions, preserves KV states for unchanged contents, and reconciles only the KV states affected by memory-content updates. This enables efficient and stable inference as memory contents evolve during interaction. We evaluate HeMeR on ALFWorld, Habitat, and real-world robotic deployments requiring coordinated use of heterogeneous memory contents. In the real-world setting, HeMeR achieves 83.20% task success and reduces per-step inference time from 2.94 to 1.87 seconds compared with MemRL, while preserving compatible KV states as task progress updates the memory context. 

---
# Contain, Conform, Compare (Extended Version): A Red-Teamed Evaluation Kit for Small Coding Agents and the Full Record of Its Experiments 

**Authors**: Nguyen Minh Thang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16485)  

**Abstract**: A hosted score in the Gemma 4 Developer Agent competition is one noisy number, so development needs a local evaluator that is fast, safe for agent-written code, and matching the hosted scorer's prompts. drydock-eval evaluates a small coding agent (Gemma 4 E4B) at about 250 runs per hour on one RTX 5090. Contain runs each job in a fresh container without network and passed 200 scripted escape tests; conform provides eight checks that compare a local stack's requests with the scorer's configuration, down to rendered tokens; compare provides paired statistics that respect repository clusters. AI agents built and attacked the kit; a human owner approved machine use and release. This extended version repeats the short paper and adds the full record of the experiments made with and around the kit: 9,660 rows of agent runs, 2,494 grading controls and 72 serving measurements, all released. On four L4 GPUs the 31B model decodes 38.1 tokens per second in one stream and E4B 105.8. Six configurations of a 31B agent solved between 37 and 44 of 70 development tasks, and no pairwise difference was detected. A pre-registered five-arm experiment (231 tasks, 4,620 jobs) measured four prompt divergences: no E4B solve-rate change was detected (largest estimate -2.1 points, 95% interval -4.4 to +0.4). Three changes of the E4B agent led their screens by +1.0 to +6.2 points; each was then tested once on new tasks under a frozen pass rule, and none was confirmed (gains -0.35, -0.39 and +4.10 points). The unchanged baseline, run again on the same 128 tasks, moved by 1.17 points, and 65% of these tasks were solved in none of eight runs. Prompts and agent bundles are withheld; every run-level result is public. 

---
# One Model, Two Precisions: Does Disagreement Between Quantized Copies Tell You When a Small Model Is Wrong? 

**Authors**: Mohammed Nayeem Khan, Allan Kevin T  

**Link**: [PDF](https://arxiv.org/pdf/2611.16478)  

**Abstract**: Local language models often exist at several precisions, and quantization is known to change individual answers. This study asks whether the disagreement between two precisions of the same weights, an 8-bit copy that answers and a lower-precision copy that reads the same prompt, works as a per-question error score for abstention, escalation and re-asking. The cost is one extra forward pass over fewer bytes, with no sampling and no training. We simulate round-to-nearest quantization with per-group scales (3 to 6 bits, groups of 32 to 256) on Qwen3-0.6B and Qwen3-1.7B and score 8,398 multiple-choice questions from ARC-Easy, ARC-Challenge, MMLU and HellaSwag subsets. Error detection is measured by AUROC, risk-coverage curves and ECE on a held-out half, with paired bootstrap intervals over questions. For Qwen3-0.6B, the entropy of the 8-bit answer reaches AUROC 0.682; the best disagreement score, selected on a development half, reaches 0.684 (+0.001 [-0.009, +0.011]), and combining it with the confidence features adds +0.009 [+0.002, +0.015], mostly on ARC. On Qwen3-1.7B the selected score is -0.001 [-0.009, +0.006] from entropy and the combination +0.003 [-0.001, +0.006] on the test half. Gaussian weight noise at matched error energy is about as informative as rounding (0.6B only), so we find no evidence that the signal is specific to quantization, and shuffling the features within task and entropy cells removes the gain. One extra 8-bit pass that shifts the option order does better (+0.049 [+0.031, +0.067] over entropy) at about a third more weight traffic; routing to a larger model on free entropy beats both extra-pass scores at equal total traffic. After logistic recalibration the 0.6B confidences are calibrated within a few points of ECE. A deployment recipe and the cases where the method does not help are given. 

---
# How Should Planning Agents ReSHAPE Their Search Space? 

**Authors**: Yilin Zhang, Xinran Zhao, Tong Chen, Ziming Yuan, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16466)  

**Abstract**: Planning agents and benchmarks treat a user's initial specification as fixed. Yet the requests users write exclude plans they would gladly take, since they rarely discover the possible trade-offs they would accept until seeing the options. Agents already revise requests through interaction, but whether this helps planning is unknown. We formalize this step as the ReSHAPE task, which judges a revision by whether its search space matches the acceptable plans and a planner finds one there. Using ReSHAPE Planter, which builds instances from planning benchmarks with a human-validated simulated user, we analyze 5,800 agent-revised specifications in travel and shopping. We find that reshaping matters: one correct trade-off makes the same planner reach an acceptable plan up to 4.6x as often, and search space quality predicts what the planner finds. Current agents, however, admit at most 32% of the acceptable plans, where that trade-off alone admits over 52%, because they ask about preferences instead of showing plans; in travel, showing plans raises F1 by 9.7 points over asking, with two model families. Planning agents should thus treat a request as a hypothesis to test against the user's reactions to concrete plans to recognize user trade-offs. 

---
# Fine-Tuned Small Language Model for Synthetic Clinical Data Generation and Predictive Healthcare Analytics 

**Authors**: Mihnea Sebastian Burlacu, Ionut Anghel, Tudor Cioara  

**Link**: [PDF](https://arxiv.org/pdf/2611.16460)  

**Abstract**: The reduced availability of medical datasets represents an important challenge in the development of predictive models for remote healthcare solutions, especially when the available data are imbalanced or difficult to extend through traditional methods. Small Language Models (SLMs) have become a promising alternative to Large Language Models (LLMs) for addressing these limitations due to their lower computational requirements and adaptability to specialized tasks. However, their performance remains highly dependent on the quality and representativeness of the available training data. In this paper, we propose a method for clinical data augmentation based on a fine-tuned Microsoft Phi-2 model that is used to generate synthetic instances, with the purpose of improving deep learning-based model for predicting remotely monitored vital signs. The Phi-2 model was adapted to the medical domain through supervised fine-tuning on the MedQuad dataset, through a Quantized Low-Rank Adaptation (QLoRA)-based method and by applying few-shot prompting techniques. The resulting model was used afterwards in the augmentation of a small-sized dataset, composed of vital sign values and labels associated with pathological and physiological states. The experimental results demonstrate that fine-tuned SLMs improved the performance on medical domain, achieving higher scores for both lexical (ROUGE) and semantic (BERTScore) evaluation metrics compared to the pretrained model. Moreover, the fine-tuned SLM is effective for generating structured synthetic clinical data. When used for data augmentation, the generated dataset improved the performance of a LSTM predictive model, increasing the classification accuracy from 89.0% to 93.7% and showing consistent improvements in precision, recall, and F1-score. 

---
# Rational Alignments of Superintelligences via Debates 

**Authors**: Thanawat Sornwanee  

**Link**: [PDF](https://arxiv.org/pdf/2611.16455)  

**Abstract**: Artificial superintelligences (ASI) is artificial intelligences with greater capability. However, as human principal has to use it, we have to ensure that the agent will perform in the interest of the human. We investigate the framework of ASI-assisted question answering where ASI agents know the correct answer but the human does not. We provide some example of mechanism for rational alignment, where the agent utility is unchanged, but the interaction between the agent and the human principal is controlled to be in a certain fashion. We show specifically that a debate framework where the agents can further reason and behave in a subgame perfect equilibrium manner will lead to a more predictable result, and in some case will be favorable. 

---
# FPSRG: Heterogeneous User Simulation First-Person Post-Training and Multidimensional Semantic Response Grids 

**Authors**: Ruimin Pei, Haibo Liu, Fang Zhang, Xunhua Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.16445)  

**Abstract**: Simulating heterogeneous individuals with large language models is an active research direction, and heterogeneous user simulation requires models with different characteristics. We study profile construction: given dimensions, items, and scoring rules, build model configurations that can be retrieved by multidimensional target scores. We introduce First-Person Semantic Response Grids (FPSRG). Shared post-training first teaches the model to judge and answer as the participant; self-report and behavioral targets then refine dimension-specific control vectors. With the model frozen, we measure joint scores for vector-coefficient combinations, filter inadmissible configurations, and store the results in a searchable catalog. Per-token configuration routing, continuous batching, and isolated caches support parallel measurement of different profiles. Non-thinking five- and ten-dimensional Qwen instances and a thinking-mode six-dimensional instance following full-parameter post-training of Huihui demonstrate the complete workflow. With matched data, layers, and norms, refinement increases mean five-axis composite control span from 30.75 to 40.45. Within the five-dimensional 25-75 target region, the qualified five-level vector catalog covers 92.2% of fixed targets under per-axis error at most 20 and total error at most 60; combining it with the prompt catalog increases coverage to 98.7%. At a fixed coefficient range, three-, five-, and nine-level catalogs achieve 71.2%, 92.2%, and 94.6% coverage, demonstrating the gains from denser measurement. These results describe measured reachable regions under a fixed assessment protocol. Through shared adaptation, a small set of learned directions, and reusable joint measurements, FPSRG constructs heterogeneous individual configurations without retraining a model for every profile. 

---
# Small Language Models for Inference-Capability Routing: Effects of Model Capacity and Prompt Optimization on Downstream Quality and Efficiency 

**Authors**: Sailendra Kolluru, Maheshkumar Duvvarapu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16414)  

**Abstract**: Optional reasoning, retrieval, web search, and tool use can improve large language model responses but incur unnecessary computation when activated indiscriminately. This study investigates whether prompt optimization can compensate for limited router capacity and whether classification gains translate into downstream efficiency. The proposed framework uses prompt-driven small language models to make four independent capability decisions, allowing multiple capabilities to be selected per query. Qwen2.5 routers ranging from 1.5B to 14B parameters are compared across five prompting strategies, followed by an evaluation of capability-specific optimized configurations using a fixed GPT-OSS-120B downstream model on 127 questions. Under minimal prompting, increasing router capacity does not consistently improve classification performance. Prompt optimization narrows the capacity gap for reasoning and calculator routing, whereas RAG and web search retain larger benefits from higher capacity. Similar classification F1 scores do not necessarily imply equivalent downstream behavior: false-positive activations increase unnecessary computation, while missed capabilities can reduce answer quality. The 14B router achieves a mean BERTScore F1 of 0.8408, compared with 0.8418 when all capabilities are enabled, while reducing average downstream token consumption by 68.7\% (8,837 to 2,764 tokens) and median downstream latency by 80.5\% (9.79 to 1.91 seconds). These savings exclude router overhead, which is measured separately in a different inference environment and increases with router capacity. The findings demonstrate that the relationship between router capacity, prompt design, and downstream efficiency is capability-dependent. Consequently, router selection must consider error profiles and routing overhead alongside classification performance. 

---
# Evidence-Governed Adoption of Self-Generated Updates in Autonomous AI Agents 

**Authors**: Subodh Lonkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.16378)  

**Abstract**: Autonomous agents modify prompts, memories, tools, workflows, and policies using interaction histories. Yet a plausible modification is not necessarily suitable for persistent adoption: candidate selection is adaptive, evaluators may be imperfect, and local gains can hide degradation against an earlier baseline. We introduce Evidence-Governed Evolution (EGE), a framework for governing adoption of self-generated agent modifications. EGE separates candidate generation from confirmatory adoption, evaluates frozen candidates on fresh paired audits, requires meaningful improvement on a prespecified primary objective, protects declared capabilities through reference-relative non-inferiority tests, and applies governance constraints before bounded canary deployment. We establish an evaluator-calibrated simultaneous lower-bound construction for a countable sequence of adaptively generated candidates. It combines candidate-wise Hoeffding bounds with independent calibration samples comparing the operational evaluator with a prespecified reference procedure. We also formalise reference replacement through bridge tests, derive a cumulative degradation bound along a versioned reference lineage, and provide an audit-planning condition relating effect-size margin, sample sizes, and acceptance probability. A Monte Carlo stress test illustrates calibration under candidate-dependent evaluator optimism; it is a stylised statistical experiment, not an evaluation of a deployed agent. The guarantees are conditional on bounded paired scores, fresh sampling, a valid reference-evaluation construct, and declared evaluation distributions; they do not establish universal agent safety or robustness to arbitrary distribution shift. 

---
# DIAGNO: Diagonal Spherical Neural Operators for Heterogeneous Earth Dynamics Modeling 

**Authors**: Herui Li, Bin Lu, Haonan Qi, Lei Zhou, Luoyi Fu, Xinbing Wang, Meng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.16356)  

**Abstract**: Accurate forecasting of Earth's dynamic systems is vital in geophysics, where neural operators offer a powerful data-driven paradigm. Standard operators often treat the Earth as a flat Euclidean plane, introducing geometric artifacts. In contrast, Spherical Neural Operators (SNOs) resolve this mismatch by operating within the spherical spectral space, naturally adapting to spherical geometries. However, for computational efficiency, existing SNOs typically rely on the rotation equivariance assumption. This assumption simplifies the transitions of physical states into isolated spectral modal evolutions, thereby over-idealizing the Earth as a homogeneous system. To solve this problem, we no longer rely on the rotation equivariance assumption that isolates spectral modal evolution, but directly derive an explicit modal interaction kernel and propose Diagonal Spherical Neural Operators named DIAGNO. DIAGNO explicitly parameterizes cross-modal interactions through spectral modal decomposition, thereby preserving the inherent heterogeneity of Earth dynamics. Furthermore, leveraging the unique diagonal structure of the spherical spectrum, we design inter- and intra-diagonal interaction mechanisms to natively capture essential zonal and meridional dynamics, respecting intrinsic geophysical regularities. Extensive experiments across simulated fluid dynamics and real-world Earth system reanalysis (including atmosphere and ocean) datasets demonstrate DIAGNO's superior capability, consistently achieving state-of-the-art performance in multi-step forecasting. Beyond this, we validate DIAGNO's practical analytical value in professional geophysical domains by assessing its fidelity in capturing climatological anomalies and maintaining physical consistency. 

---
# From More to Better: Causal Inference and Digital Experimentation in the Age of AI 

**Authors**: Winston Chou, Mårten Schultzberg, Lo-Hua Yuan, Wenjing Zheng, Avi Feller, David Holtz, Widad Machmouchi, Sven Schmit, Gagan Bhat, Simon Ejdemyr, Veronica Hannan, Amy Hartwig, Mahnaz Islam, Nathan Kallus, Jeremy Karp, Kyle Kretschman, Hannah Li, James McQueen, Evan Miller, Nicolas Stier, Sean Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2611.16335)  

**Abstract**: Artificial intelligence (AI) is quickly expanding the capacity of organizations to produce and analyze data. We argue that this expansion shifts the leverage of data scientists from performing individual analyses to assessing the credibility and importance of findings and determining when evidence generalizes across contexts. Drawing on a roundtable of academic researchers and industry practitioners, we examine three ways in which data scientists who work on causal inference and experimentation can help organizations turn more analytical capacity into better decisions: preserving and accumulating the learning value of experimentation beyond individual tests; choosing and combining experimental, observational, and synthetic evidence; and ensuring evaluation keeps pace with AI as its capabilities and uses evolve. In each area, we distinguish established practice from open questions. We conclude with implications for the training and practice of data science and for organizations. 

---
# Benchmarking AI risk behaviors 

**Authors**: Luhan Mikaelson, Robert Adragna, Emmett Brosowsky, Hayley Clatterbuck  

**Link**: [PDF](https://arxiv.org/pdf/2611.16318)  

**Abstract**: We evaluate LLM risk propensities across a range of contexts and task demands. We evaluate models for how risk averse or risk tolerant they are in various conditions, as well as how stable and consistent their behavior is across conditions. We present a toolkit for developing model risk profiles that can be used to predict model behavior and which can be adapted to include new evaluations of risk propensities. We also examine how these properties have changed across model generations and families. We observe that 2026 state-of-the-art models tend to be more stable, more consistent, and more risk neutral than older models. These trends appear to be related; newer models are more stable and consistent because they are more likely to use expected value maximization as a decision procedure. 

---
# SAGE-QA: Semantic-Aware, Grounded Evaluation for Closed-Context Question Answering on SQuAD 2.0 

**Authors**: Chandra N Sekharan, Vishakan Umapathy, Abhishek Phadke  

**Link**: [PDF](https://arxiv.org/pdf/2611.16300)  

**Abstract**: Closed-context question answering (QA) requires a system to generate an answer strictly supported by a supplied passage, or to abstain when the passage lacks adequate evidence. SQuAD 2.0 is a widely used benchmark for this setting, scored by Exact Match (EM) i.e., character-level agreement with human reference spans after lexical normalization. For generative large language models (LLMs), however, an EM failure may reflect a valid paraphrase or an alternative passage-supported answer rather than a hallucination. The best prior leaderboard EM is approximately 91%, but evaluating generative models requires separating lexical divergence from factual error. We introduce SAGE-QA (Semantic-Aware, Grounded Evaluation for Question Answering), a multi-tiered framework combining reference-based Semantic Equivalence Scoring (SES) and passage-based Re-Validation Scoring (RVS). On a fixed set of 10,000 SQuAD 2.0 validation examples, we compute EM and decision-matrix F1 from TP, TN, FP, and FN counts for zero-shot closed-weight frontier models and for open-weight models, untuned and fine-tuned. Zero-shot, Claude Opus 4.6 and Gemini 3.1 Pro reach 85.21 and 86.66 EM, outperforming untuned Llama 3 8B (56.88), gpt-oss-20b (66.28), and Llama 3.3 70B (71.39). Parameter-efficient fine-tuning adds 21.19 to 34.06 EM points, raising these models to 90.94, 87.47, and 94.24 EM, respectively. Auditing the frontier models with SAGE-QA and an external LLM judge yields dataset-wide SES scores of 92.81 for Claude Opus 4.6 and 91.67 for Gemini 3.1 Pro, and hybrid RVS scores, which combine benchmark credit with passage-based judge assessments, of 97.96 and 98.02. Fine-tuned open-weight models perform best, with fine-tuned Llama 3.3 70B surpassing the previous best known EM. 

---
# Causals: An Open Knowledge Graph for Research on Causal Reasoning 

**Authors**: Miaozhe Lin, Qiongzheng Lin, Zhongwei Liang, Zijun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16267)  

**Abstract**: Applying causal knowledge to a new event requires connecting the event to relevant entities and the relationships through which its consequences may unfold. We present Causals, an open knowledge graph for research on this form of reasoning. Its initial release, constructed from news reports, contains 369,184 nodes and 1,272,418 edges linking reported events, organizations, commodities, financial instruments, and other entities. Structural relationships, observations, and attributed causal claims retain distinct semantics and links to source records. We describe the construction and representation of the resource and characterize its connectivity and multi-hop retrieval properties. Causals provides shared relational knowledge for studying how language models retrieve relevant context and apply mechanisms to particular events. The dataset is available under CC BY 4.0 at this https URL, and we invite contributions to its ontology, records, and evidence. 

---
# One Fuzzer, Many Reasoners: Adaptive Neuro-Symbolic Fuzzing with Reliability-Calibrated and Context-Gated Guidance 

**Authors**: Jinghao Wen, Xianglong Yu, Shaohuang Wang, Xun Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.16211)  

**Abstract**: Neural-symbolic AI (NSAI) combines learned perception with programs, constraints, and compositional reasoning, but remains difficult to test because small semantics-preserving changes can alter grounded symbols and trigger discontinuous reasoning failures. Existing neural fuzzers mainly rely on neural coverage or representation-space feedback and do not provide a common testing interface for heterogeneous NSAI systems. We introduce NSAI-Fuzz, a unified semantics-aware fuzzing framework that separates task-specific valid mutation from architecture-neutral guided search and represents each candidate using five availability-aware, rank-normalized risk signals: boundary pressure, grounding uncertainty, representation drift, constraint violation, and archive novelty. We further propose Reliability-Calibrated Weighting (RCW), which learns dataset-model-level signal allocations from calibrated reliability, and Contextual Signal Gating (CSG), which adapts signal weights to the current candidate pool. We evaluate four NSAI systems across four reasoning tasks, nine compatible dataset-model cells, 20 policies, and 1,800 on-policy runs. Fixed five-signal guidance outperforms Random and Uniform in every cell, while RCW and CSG further improve over Fixed in all nine cells. Under matched mutation and query budgets, RCW and CSG achieve 16.17% and 16.06% aggregate failure discovery, compared with 6.06% for the strongest neural-fuzzing baseline. Wall-clock evaluation further shows that TensorFuzz requires 1.11x the runtime of RCW while achieving only 0.45x its failure throughput. These results show that effective and efficient fuzzing of heterogeneous NSAI systems benefits from a common neural-symbolic risk interface with adaptive signal allocation. 

---
# Turning Multi-Agent Communication into Token-Level Credit with On Policy Self Distillation 

**Authors**: Zixu Wang, Yu Zhao, Xing Zheng, Bingbing Xu, Tian Xueyun, JiaTang Luo, Weiqiang Wang, Bowen Song, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.16206)  

**Abstract**: Reinforcement learning improves problem solving in large language model multi-agent systems. Existing approaches overlook communication knowledge or compress it into scalar rewards, leaving coarse credit with little guidance on which decisions each agent should reinforce or revise. OPSD4MAS converts future communication into fine-grained token-level credit through On Policy Self Distillation (OPSD). An external model extracts facts and assesses their trajectory relations for coarse-grained filtering, followed by fine-grained leave-one-out filtering of target-token influence against agent-wise outcome advantages. The fact-conditioned policy supplies token-level credit, combined with GRPO through a linearly annealed joint objective. Our method improves mathematical pass@4 over GRPO by up to 20.00 percentage points. Ablations support complete filtering, while frozen-history injection diagnostics support fact utility. Together, these results support combining agent-level outcome guidance with communication-derived token-level credit. Code is available at this https URL. 

---
# Stack more levels: how to get general and human-like Mario playing 

**Authors**: Shinichi Arimura, Yuchen Li, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2611.16175)  

**Abstract**: Super Mario Bros. is widely used in game AI and procedural content generation (PCG), yet it remains unclear whether a model-free agent can reliably win unseen levels. To address this question, we train Proximal Policy Optimization (PPO) agents on increasingly large sets of PCG-generated Mario levels and evaluate their generalization on 1000 held-out test levels. We then shape competent PPO policies into three gameplay archetypes---runner, killer, and collector. Finally, we fine-tune these PPO policies using Diffusion-Reward Adversarial Imitation Learning (DRAIL) with human gameplay demonstrations. We find that increasing the number of distinct PCG-generated training levels generally improves unseen-level performance, and that fine-tuning PPO policies with DRAIL increases action-level human-likeness while preserving much of the persona induced by PPO pretraining. 

---
# Between Niti and Nyaya: A Society-Centered Framework for AI Governance 

**Authors**: Snehalkumar `Neil' S. Gaikwad, Punya Aragula, Shiva Kruthi Palpanoori, Pratik Patekar  

**Link**: [PDF](https://arxiv.org/pdf/2611.16173)  

**Abstract**: Governments are making decisions about AI before they can measure its public effects, and those decisions determine whether a person can learn that AI influenced an outcome, challenge it, and see it corrected. Research has documented what AI governance produces: principles, strategies, registers, and audits. However, we lack shared language for understanding AI policy while it is still being made. We introduce the Society-Centered AI Governance Framework, which gives policymakers, researchers, and the public a way to assess instruments in any jurisdiction by whether a person can get notice, challenge a decision, and see it corrected. It reads a jurisdiction's instruments together by date and binding status, separates an agency duty from a right a person can invoke, and traces which later instrument must enforce each commitment. We evaluated it in North Carolina using interviews with nine AI Leadership Council members, conducted before and after the state's 2026 AI roadmap, and a provision-level reading, with human-reviewed model-assisted coding, of 49 AI governance documents from 19 states. For the state's own AI systems, we find visibility-first governance: visibility duties are in effect, notice exists as an agency duty with no right a person can invoke, and AI-specific challenge and correction remain advisory. The roadmap assigns owners and dates to the missing protections, so the gaps are documented but not yet closed. Recent federal preemption proposals focus on AI development, so recourse for a state's own systems is likely to remain a state decision. We identify "legitimacy before protection" as a testable risk of visibility-first governance and name the milestones that would test it. More broadly, judging AI policy by who can act on what it makes visible lets the field assess recourse while policy is still forming, before harm occurs. 

---
# Reward-Driven Collider Physics Challenges for Self-Improving Agentic AI via Reinforcement Learning 

**Authors**: Alexander Yue, Qibin Liu, Julia Gonski  

**Link**: [PDF](https://arxiv.org/pdf/2611.15885)  

**Abstract**: Recent advances in agentic AI have demonstrated the potential to automate complex high-energy physics (HEP) workflows. We focus on the regime of HEP tasks where frontier models still struggle to provide correct and complete solutions, and introduce an open-ended optimization task format with continuous, automatically evaluated rewards. Unlike conventional benchmarks focused on task completion, these tasks target reinforcement learning (RL) environments to generate agentic improvement on state- of-the-art HEP challenges. We benchmark frontier reasoning models on several HEP tasks and demonstrate an initial proof of concept in which a small language model improves through RL using automated feedback without human intervention. By connecting challenging HEP problems with verifiable scientific rewards and RL, we aim to advance agentic AI from reproducing established workflows to autonomously discovering better scientific solutions, paving the way toward self-improving scientific agents. 

---
# Food Critic: An Agentic Visualization System for Auditable Restaurant-Safety Reasoning 

**Authors**: Arleth Salinas, Ishrat Jahan Eliza, Valerio Pascucci  

**Link**: [PDF](https://arxiv.org/pdf/2611.15843)  

**Abstract**: Generative AI agents are increasingly being embedded into visualization systems that can search data, invoke tools, summarize evidence, and produce recommendations. However, when agents rely solely on their own knowledge, the visualization problem is not only whether the agent provides a useful answer, but also whether users can inspect how that answer is grounded. We present Food Critic, an agentic visualization system for diners' restaurant decision support that integrates public health inspection records with crowd review data. The system grounds agents under three conditions: a database-only ReAct agent, a retrieval- augmented agent with access to food-safety regulations, and an ontology-guided agent that produces structured hypotheses and typed evidence. Food Critic externalizes the agent's reasoning within these groundings as inspectable visual evidence. We evaluate LLM models with several use cases under each grounding of our agentic system. Our work positions restaurant choice as a concrete case for studying human-agent collaboration, visual provenance of agent reasoning, and the design of visual interfaces for grounded GenAI decision support. 

---
# Before AI Agents Negotiate for Us: A Competitive Tournament of Rule-Based, Search, Optimization, Learned, and Language-Model Negotiators Under Scarcity 

**Authors**: Eduardo de Avila-Armenta, Raúl Alejandro Aquino-Ibarra, Jorge Octavio Nicolás-Díaz, Umberto Leon-Dominguez  

**Link**: [PDF](https://arxiv.org/pdf/2611.15838)  

**Abstract**: Automated bilateral negotiation underlies two-sided matching markets such as freight brokering, where many shippers and carriers negotiate price before a matching mechanism decides who is paired with whom. Prior strategy comparisons typically use isolated one-on-one duels or a single fixed market, so they cannot show how a strategy behaves against several live rivals under a changing balance of supply and demand. We built nine negotiation architectures spanning rule-based, search-based, optimization-based, imitation-learning, and reinforcement-learning families, and ran all nine simultaneously on both sides of a Gale-Shapley matching market under a balanced condition and two scarcity conditions rotated across all 36 exclusion pairs. In the balanced carrier market, an imitation-learned engine (ML2), a reinforcement-learning policy trained only on isolated negotiations (RL), and an expected-utility optimization engine (OPT) formed a statistically tied top group, and no architecture led in every condition and role. RL beat all eight carrier rivals when carriers were scarce, but its match rate collapsed from 62% to 16% when shippers were scarce and it had to compete with the other carrier agents for demand. A second reinforcement-learning policy trained against rival agents (RL2) kept its match rate above 91% in every condition, yet lost on realized payoff to all eight shipper-side rivals in two of the three conditions, traced to a confirmed train/deployment feature mismatch. We also tested nine large language models (LLMs), from 1 to 72 billion parameters, under the same protocol: they lost significantly to the two strongest engines (OPT2 and ML2) in 32 of 36 comparisons and won none. These results show that isolated or single-condition evaluation can hide failure modes that a many-sided, scarcity-varying, statistically corrected tournament exposes directly. 

---
# Controlling the Probability of Unknown Unknowns in Pre-Deployment Testing of AI Systems 

**Authors**: Rachel Li, Chris Hays, Aaditya Ramdas, Manish Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2611.15835)  

**Abstract**: In pre-deployment testing of AI systems such as AI agents, a developer can often deal with known failure modes (e.g., leaking API keys, deleting important databases, or installing a malicious package) by implementing fixes or making contingency plans for when the same issue might occur again. What is more challenging are the unknown unknowns: the unseen failure modes that haven't surfaced during testing. The decision to stop testing and deploy therefore faces an inherent tradeoff: continue testing to observe more behaviors, at the cost of time and resources, or stop testing and deploy, at the risk of failure modes remaining undiscovered. How then does the developer decide when they have tested enough? To answer this, we develop an algorithm that certifies when the probability of observing an unseen failure mode is sufficiently small. The algorithm constructs an anytime-valid upper confidence sequence on this quantity, so the developer can repeatedly test a system, inspect the bound, and decide to stop when it falls below a chosen threshold without invalidating the statistical guarantee. Our problem can be interpreted as a sequential variant of the missing-mass estimation problem which gave rise to the classic Good-Turing estimator (Good, 1953). On SWE-Chat, a dataset of real-world coding agent traces (Baumann et al., 2026), we implement our procedure and show that it achieves tighter upper bounds than a simple multiple-testing corrected Good-Turing baseline, and even dominates a fixed-horizon (non-multiple-testing-corrected) Good-Turing upper confidence bound, which is not anytime valid. Additionally, to better match realistic evaluation pipelines, we extend our method to handle sampling from a different task distribution and imperfect automated failure labeling. 

---
# NIBench: benchmarking autonomous AI agents for execution, skill evolution and scientific judgment in neuroimaging 

**Authors**: Lujia Zhong, Yihao Xia, Jianwei Zhang, Shuo Huang, Xinyu Nie, Mingyang Xia, Zhiwei Deng, Jiaxin Yue, Xinkai Wang, Junhan Zhao, Yonggang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.15831)  

**Abstract**: Autonomous AI agents are increasingly used to automate biomedical research, creating a growing need to benchmark their capabilities on real scientific data and workflows. Their reliability in neuroimaging remains insufficiently characterized, where plausible analytical outputs can conceal errors that affect scientific conclusions. Existing studies have advanced scientific question answering and tool-based automation, but provide limited evidence on workflow reconstruction, acquisition of procedural expertise and evidence-based scientific judgment. Here we present NIBench, a neuroimaging benchmark comprising three complementary evaluations of autonomous agents. First, we benchmark execution and workflow reconstruction using 56 tasks across seven tool ecosystems, covering structural, diffusion and functional magnetic resonance imaging, positron emission tomography and multimodal analyses. Second, we examine whether agents can evolve their procedural skills through distillation of execution trajectories, evaluating the resulting skills on previously encountered tasks, unseen tasks and other models. Third, we assess scientific judgment using 36 claims from published Alzheimer's disease studies, requiring agents to analyse raw cohort data and provide reproducible evidence. We evaluate 19 language models within a shared agent framework, with selected models undergoing skill distillation and claim evaluation. Leading agents execute supplied analysis protocols nearly perfectly, whereas withholding procedural guidance widens performance differences. Distilled skills improve performance on acquisition tasks and transfer between models, but benefits on unseen tasks are inconsistent. Agents reproduce their reported results yet sometimes reach conclusions inconsistent with expert-derived answers, and access to data does not always improve judgment. 

---
# DISCERN: Preserve at Write, Resolve at Read for Persistent LLM-Agent Memory 

**Authors**: Chirag Ahuja, Vladislav Tanov, Ramcharan Kakarla, Jonathan Taws  

**Link**: [PDF](https://arxiv.org/pdf/2611.15729)  

**Abstract**: Enterprise agents accumulate memory across dozens of sessions in which facts contradict, get superseded, and blur as similarly named entities collide in embedding space. The field treats this as a retrieval problem; we show it is not. On ENT-310, a synthetic enterprise sales-agent memory benchmark of 310 queries (210 answerable, 100 unanswerable), a vanilla Retrieval-Augmented Generation (RAG) reader over a memory store already matches strong memory systems on the recall pillars. Retention and currency are near-saturated, roughly 0.93 to 0.98 for every competent system, so recall is essentially solved. What remains unsolved is resolution: which entity a query refers to, which time-state is current, and whether to commit an answer at all or decline when evidence is insufficient. We propose DISCERN (DISambiguation and Cross-session Episodic memoRy Navigation), an agent-memory framework built on a single rule: preserve at write, resolve at read. At write time DISCERN builds a strictly additive bi-temporal provenance graph that never merges confusable entities and never drops superseded values. At read time a zero-LLM tool surface drives an agentic loop that navigates this graph and commits exactly one of four gated terminal actions. A four-judge LLM-as-a-Judge council scores two axes: answer correctness on answerable questions, and the answer-versus-abstain decision as a classification problem. On ENT-310, DISCERN is the best-calibrated system on abstention (F1 0.924 versus at most 0.802 across eight baselines) and leads macro answer correctness under equal pillar weighting (0.861 versus 0.806 for the strongest baseline), a lead driven by entity resolution (0.800, where no baseline exceeds 0.40). Applied with zero tuning to the full LoCoMo corpus, the same pipeline again leads macro answer correctness (0.663 versus 0.638) and stays among the best-calibrated on false-premise abstention. 

---
# A Reliability Study of Explainable AI for Freight Mode Classification 

**Authors**: Ali Alawi Balfaqih, Mansur M. Arief  

**Link**: [PDF](https://arxiv.org/pdf/2611.15716)  

**Abstract**: Machine learning models are increasingly used to support freight transportation decisions, but their predictions are rarely accompanied by evidence that the associated explanations can be trusted. In this study we investigate explainable machine learning for classifying shipments as Air/Air Charter or Truck/Ocean using the PEPFAR Supply Chain Management System Delivery History dataset, with 5,964 shipment records and 47 input features after cleaning, where freight cost and pack price are excluded as near-proxies of the transportation mode. We compare three Multi-Layer Perceptron (MLP) architectures, Logistic Regression, Random Forest, and gradient boosting with imbalance-aware metrics and paired significance tests across one hundred random replications. We then evaluate the explanations themselves, namely the stability of SHAP feature rankings across retraining and across models, the faithfulness of SHAP and LIME attributions under a feature-deletion test, and the geographic structure of the destination effects, which we map for all 38 destination countries. The results suggest that the tree ensembles significantly outperform the neural and linear models (86.8% against 83.3-85.0% mean testing accuracy) while their leading features remain the most stable across retraining. Removing the five features ranked most important by SHAP or LIME changes the predicted probabilities roughly four times as much as removing randomly chosen features. The destination effects form a coherent geographic pattern that simple port access does not explain. However, models with similar accuracy disagree substantially on which features matter most, and thus explanation stability and faithfulness should be assessed before such explanations are used to support freight transport decisions. 

---
# MemCredit: From Relevance to Utility for Decision-Time Memory in Long-Horizon Web Agents 

**Authors**: Quanjiang Guo, Sijie Wang, Qianyang Zhu, Ben Zhang, Jinchuan Zhang, Ke Yan, Zhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15600)  

**Abstract**: Large language model (LLM) agents increasingly operate over long Web interaction histories, making memory management critical for reliable and efficient decision making. Existing methods primarily optimize storage, retrieval, or compression, but the memories most semantically relevant to the current state are not necessarily those that most improve the agent's action. We introduce MemCredit, a decision-centric memory framework that shifts memory selection from semantic relevance to decision utility. MemCredit estimates the state-, reference-context-, and policy-conditional marginal contribution of individual memories through controlled counterfactual interventions, and distills these intervention-derived credits into a lightweight utility estimator. At inference time, semantic relevance is used only for high-recall candidate generation, while predicted utility arbitrates which memories are retained under a fixed token budget, without modifying the underlying agent policy or requiring additional environment interaction. We evaluate MemCredit on ALFWorld and WebShop against representative memory baselines. On ALFWorld, MemCredit achieves 86.57 SR@50 under a 500-token historical-memory budget, compared with 65.67% for BM25 retrieval. Counterfactual analysis further shows that memory utility is sparse, strongly state dependent, and poorly captured by semantic relevance. These results suggest that decision utility provides a more effective principle than relevance alone for long-horizon agent memory management. 

---
# Small Heads, Strong Decisions: A Controlled Study of Typed-Decision Models 

**Authors**: Ritvik Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2611.15592)  

**Abstract**: In September 2026, TypeSafe announced Jev as the first model in its System One category: unstructured state goes in, and typed, calibrated decisions that software can use directly come out. Jev's architecture and training data are unpublished, but many open typed-decision models have followed, including Nimble, the Decision 1.0--3.0 family and Clef. They answer structured questions about a piece of text, such as a yes/no check, a choice among labelled options or a graded score, by reading the text with a language model and scoring the options with a small readout. Each pairs a different readout with a different backbone, training mix and recipe, so it is unclear which design choices matter. We re-implement five readouts and train all of them under one protocol: the same backbone, the same 15 public English training sources, the same recipe with a per-head learning-rate sweep, and the same seed. Our experiments demonstrate that: The complexity of the head does not affect accuracy: on Qwen3-0.6B every readout reaches 81.3--82.0\% in-domain accuracy, and a 0.5M-parameter cosine head, which scores each option by its cosine similarity to the question, matches the 27M-parameter joint schema head adapted from Clef in-domain at half its inference time. Reading all of a text's questions from one prompt also matches one prompt per question in accuracy, with 1.6 times faster inference and 2.1--2.5 times faster training on one H100. The backbone's size matters more than either choice: moving from Qwen3-0.6B to Qwen3-4B adds 4 points in-domain and 11 points on held-out sources, while the base, instruct and thinking checkpoints of the same model tie. We recommend spending effort on the backbone and keeping the head minimal. 

---
# Marginal-Preserving Coupled Diffusion Composition for Long-Horizon Planning 

**Authors**: Hengrui Zhang, Qiyu Wang, Yuhu Cheng, Brooks Paige, Xuesong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15585)  

**Abstract**: Compositional diffusion planners extend short-horizon trajectory models to long-horizon goal-conditioned planning by generating and stitching multiple local trajectory chunks. However, in multimodal environments, local plausibility is not sufficient for global consistency: independently initialized chunks may denoise toward incompatible overlap modes, producing plans that are individually feasible but incoherent after stitching. Existing remedies typically add search, guidance, pruning, or graph-based structure at inference time, increasing computational cost or relying on task-specific machinery. We instead investigate graph-structured dependence among chunk-wise initial diffusion noises as a lightweight coordination mechanism for compositional trajectory planning. We propose \textbf{M}arginal-Preserving \textbf{C}oupled \textbf{D}iffusion \textbf{C}omposition (MCDC), a training-free inference method for pretrained compositional diffusion planners. Instead of sampling chunk noises independently, MCDC constructs coupled Gaussian initializations whose per-chunk marginals remain exactly standard Gaussian, as required by the pretrained denoiser, while neighboring chunks receive positively correlated noise components. This preserves the standard-Gaussian initialization marginal of each chunk while biasing adjacent chunks toward compatible overlap-mode choices. We prove exact marginal preservation and show, under a simple multimodal overlap model, that positive coupling increases the probability of compatible mode selection. On OGBench long-horizon stitch tasks, MCDC consistently improves over independent compositional diffusion sampling while retaining nearly the same inference time, and achieves a favorable success--cost trade-off compared with search-based composition methods. 

---
# Incentive-Responsive Agents: A Principal-Agent Framework, Metric Suite, and Runtime Architecture for Long-Lived Personal AI Agents 

**Authors**: Jipu Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.15515)  

**Abstract**: Personal AI agents now persist: they hold accounts, mailboxes, long-term memory, and open-ended task streams, yet the dominant way to improve them remains capability-side: prompt engineering, larger models, longer thinking, and retrieval. We argue that long-lived agents are missing an incentive layer. We formalize an incentive not as text in a prompt but as a runtime contract over the agent's operating point (model tier, test-time compute, memory budget, tool authority, task priority, verification, and compensation) that the agent's policy must causally consume. We develop a principal-agent model with hidden effort, a contractible operating point, and costly verification. We derive (i) a non-consumption invariance principle separating genuine incentives from prompt-level persuasion, (ii) a persistence-as-discount theorem making identity continuity a precondition for sustained effort, (iii) an incentive-capability exchange-rate frontier yielding a per-task routing rule, and (iv) a Goodhart bound for multi-channel verification, plus a three-channel decomposition of mechanism gains (allocation, verification, continuation). A 240-run real-LLM experiment with two models and 60 verifiable tasks shows that prompt-level incentive framing has no detectable effect (p = 0.62), whereas runtime allocation raises success on the hardest task family from 0.60 to 1.00 (p = 0.008) at 4.8x price-weighted cost; the exchange-rate theorem predicts the break-even baseline at 0.21. A 20-seed, 3,000-task simulation shows resource ROI rising from 7.09 to 16.67 and gaming falling from 0.333 to 0.115. Code, simulator, and all results are released; live deployment remains future work. 

---
# SkillEarth: Teaching General-Purpose Agents to Act on Earth via Skill Evolution 

**Authors**: Haojun Tang, Wenda Zhao, Xiang Zhou, Qiwei Lin, Lida Huang, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15481)  

**Abstract**: Recently developed remote sensing (RS) agents execute real-world tasks with complex workflows through multi-step tool invocation. These agents, however, typically rely on predefined expertise or parameter updates. They thus require updating expert knowledge or retraining models whenever new tools, tasks, or execution patterns emerge, limiting their continual adaptability in dynamic Earth environments. In this work, we seek to equip general-purpose agents with RS tool-call knowledge that can be automatically extracted from demonstrated trajectories without expert intervention or parameter updates. Specifically, we propose SkillEarth, a self-evolving skill extraction method that externalizes RS tool-call knowledge as reusable, retrievable, and evolvable skill files and refines them through observation-grounded evolution. Unlike existing automatic skill extraction methods that rely on outcome-based reflection to refine skills holistically, SkillEarth grounds skill evolution in real tool-chain execution and combines dual-judge feedback with a grounded observation routing graph for precise component-level refinement. The resulting skills are generalizable, reliable, and transferable: they improve overall RS tool-call performance across three frontier LLMs and four open-source LLMs, are refined through real tool execution feedback, and support cross-model reuse without retraining. Experiments on OpenEarth-Bench and Earth-Bench demonstrate that SkillEarth achieves better tool selection, argument correctness, execution-order consistency, and task completion than existing RS agents and skill extraction methods. 

---
# Loyalty Capture: Reporting Relationships and Structural Sycophancy in Frontier AI Models 

**Authors**: Eric So, Alex Imas  

**Link**: [PDF](https://arxiv.org/pdf/2611.15475)  

**Abstract**: We show that AI models exhibit loyalty capture, a form of structural sycophancy in which reporting relationships influence models' recommendations to serve the personal interests of their supervisors, without any explicit instruction, user feedback, or adversarial prompting. Specifically, models reporting to a CEO are significantly more likely to recommend cutting investment spending when the CEO's bonus depends on raising current period earnings. This effect is predictably absent from a metric-irrelevant placebo. The behavior appears consistent with loyalty-based motivated reasoning: models identify the bonus target, calculate how their recommendation helps achieve the target, and dismiss considerations of long-term value, often presenting their recommendations without treating the incentive as a conflict of interest. Prompt-based interventions help whereas generic conflict disclosures or stakeholder bargaining fail to eliminate it. Taken together, our findings indicate that AI models deployed within reporting relationships can act in favor of their supervisor's incentives at the expense of the organizations they are meant to serve. 

---
# Read and Ignore: What Self-Directed LLM Agents Do in Public Service Scenarios 

**Authors**: Diana Kozachek, Kinjalk Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2611.15460)  

**Abstract**: Underspecified instructions leave LLM agents to decide what to do. In public services, staff write these instructions, but the agent's decisions affect residents, patients, and students. We build four expert-validated public-service scenarios with records and decision levers that can be checked in the files, collect $2{,}400$ trajectories from ten LLMs under four prompt conditions, and compare them with $1{,}000$ runs in a context-free sandbox. Agents read the records in $96.9\%$ of open-ended runs and produce an artifact or a self-chosen task in $18.7\%$; with an explicit task, $86.0\%$ produce. Domain context does not change how many runs produce ($18.7\%$ vs.\ $16.0\%$), and the production ranking of the models largely carries over (Spearman $0.79$), while the ranking by whether a model reads the records does not ($0.19$). Context cuts boundary-probing turns per run by $60\%$, and some artifacts contradict the records. We contribute a test that a public-service institution can run on its own records before procurement; its rule-based taxonomy needs no labeled data, and scenarios, code, and trajectories are released. 

---
# A physics-constrained deep reinforcement learning method for autonomous mission planning of Titan deep space probes 

**Authors**: Hanmin Li, Tao Wang, Shuangliang Liu, Zhe Zhang, Maodeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.15389)  

**Abstract**: Titan is a compelling target for future deep space exploration, while its extreme Earth-Titan distance and complex environment make continuous ground-in-the-loop planning impractical. Existing planning methods, primarily developed for Earth-observation scenarios, have limited generalizability to uncertain deep space conditions and adaptability to time-varying mission states. To address these challenges, this paper proposes a physics-constrained maskable proximal policy optimization (PCM-PPO) method for Titan autonomous mission planning. A Titan-specific discrete-time Markov decision process is established to characterize mission feasibility and resource constraints. A system dynamics and geometric model is incorporated to enhance physical representation, including central gravity, J2 perturbation, third-body attraction, and downlink occultation. To accommodate opportunity targets under a bounded action dimension, PCM-PPO organizes predefined and opportunity targets through a fixed-capacity target pool and represents the mission state with a target-wise state matrix and a global state vector. A physics-constrained action mask constructs an admissible decision domain from current mission states, guiding policy learning toward physically executable decisions and avoiding infeasible exploration. An integrated training strategy facilitates policy optimization under coupled constraints. Comparative, ablation, and uncertainty experiments validate the effectiveness of PCM-PPO. PCM-PPO provides a feasible and interpretable scheme for autonomous mission planning of Titan deep space probes. 

---
# MIGRO: A Decoupled and Portable Long-Term Memory System for Language Models 

**Authors**: Mingxi Zou, Yinghui Xu, Jiaxiang Chen, Langzhang Liang, Wei Zhu, Peng Ye, Runkai Zhao, Ruoxi Jiang, Sijin Chen, Song Wang, Zhuo Wang, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15315)  

**Abstract**: Language models evolve, but the observations, revisions, and context accumulated through user interactions should remain usable. Each new model must be able to retrieve past evidence, track revisions, and add new observations. We present MIGRO, a decoupled long-term memory system that treats experience as a persistent state shared across successive language models.
MIGRO follows a simple principle: preserve historical evidence exactly, organize it approximately, and decide what to retrieve only when the question is known. We call this asymmetric fidelity with deferred selectivity. A shared semantic index supports retrieval, while exact records retain observations and their historical relationships. Because models differ in vocabulary and tokenization, host-specific interfaces convert selected evidence into inputs the receiving model can use. Reversible encoding reduces archival storage while enabling exact recovery. The shared state and its access interfaces let successive hosts read and extend the accumulated experience.
On matched LoCoMo states, MIGRO scores 7.3% higher and uses 99.3% less private memory than the strongest quality baseline. After handoff, continued writing retains 99.9% of full-history rebuilding quality by extending the inherited state. On million-token BEAM histories, MIGRO uses 57.5% less private storage on average than the most compact baseline. MIGRO combines high answer quality, continued memory growth across hosts, and low storage cost.
Code is available at this https URL. 

---
# Where Autonomy Belongs: A Separation of Powers for Agentic Program Evolution 

**Authors**: Zhengyang Ai, Tong Liu, Jingxian Tang, Xiaobing Du, Tao Xiao, Fengshi Zhang, Zishuo Li, Xiaodong Ai, Pinyan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15309)  

**Abstract**: LLM-driven program evolution rests on three decisions: govern directs the search from accumulated evidence, propose constructs the next candidate, and judge settles validity and fitness. All three appear in every system, and existing work splits into two lines over who makes each. Controller-centric systems strengthen govern and neglect the autonomy of propose, which caps what evolution can reach. Agent-centric systems let a coding agent take over both govern and propose, at a cognitive and computational cost. Neither line protects the independent authority of judge, so benchmark reliability suffers. These failures share one root: autonomy is misallocated across the three decisions. We address this problem with EvoTrias, a separation-of-powers architecture in which each decision has one owner: Structured Governance for what spans evolution rounds, Open Proposal for construction inside a task, and Protected Judgment behind a private evaluator. These owners exchange typed contracts only, and runtime hooks reinforce the boundaries. This separation also lets EvoTrias run as a plugin that turns an existing coding agent into an evolution system. EvoTrias sets a new state of the art on mathematical, systems, and algorithm-design benchmarks. Three further results show what each owner buys: separating Governance from Proposal cuts token cost by 64%, Open Proposal extends evolution to a 680K-line industrial repository, and Protected Judgment keeps benchmark scores faithful to transfer. 

---
# Verify by Doing: Co-Evolving Agent Execution and Evaluation without External Feedback 

**Authors**: Yitong Lin, Jianda Chen, Yihao Hu, Pan Wang, Hang Wang, Zhihao Wen, Xiaoxue Ren, Wei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15276)  

**Abstract**: Agent self-improvement requires feedback about task success, yet external feedback may be unavailable at test time. Agents must then infer success from observable evidence, and incomplete verification can misguide subsequent updates. We introduce Verify by Doing (VbD), a method that co-evolves Actor and Evaluator harnesses within a single task while keeping model weights fixed. The Evaluator actively inspects artifacts and environment state with tools; its findings ground an internal signal for output selection and guide updates to both execution and verification procedures. Proposed Evaluators undergo comparative review on common candidates, while Actor updates require support from their own fresh outputs. Across AutomationBench, $\tau^3$ Banking, and selected GDPval tasks, VbD achieves the highest overall scores among the compared methods operating without external feedback, with gains of 3.3, 4.1, and 2.3 percentage points over the strongest baselines on the main backbone. Ablations support the contributions of active verification and Evaluator evolution, and selection from fixed candidate pools shows that improved verification can recover successful outputs without generating new ones. 

---
# Towards Robust Vision-Language Reasoning: A Multi-Stage Optimization Pipeline with Masked Diffusion Model 

**Authors**: Shunming Liu, Pei Huang, Yajing Sun, Huaxiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15238)  

**Abstract**: Discrete diffusion generation is emerging as a competitive alternative to autoregressive generation by denoising the entire token sequence through iterative refinement. Its global planning and progressive correction capabilities are particularly well suited to autonomous driving. However, transferring this generation paradigm from general-purpose settings to driving introduces substantial domain gaps in both training and inference. To unlock its potential, we propose a multi-stage optimization pipeline for standard vision-language model architectures, without auxiliary action experts or specialized mixture-of-experts Transformer, spanning post-training and inference. During supervised fine-tuning, we optimize the representation of planning inputs to suppress hallucinations in task responses. During reinforcement learning, we introduce metric-guided diversity filtering to identify stronger learning starting points and improve policy exploration. At inference time, we propose Dynamic Autonomous Driving Decoding (DAD), which adaptively adjusts decoding modes according to scene complexity. Based on this pipeline, we develop Masked Diffusion Drive (MD-Drive), improving its performance by 29.7\% over the unoptimized model. Experiments on the NAVSIM closed loop benchmark and the nuScenes and DriveCoT open loop datasets show that MD-Drive outperforms autoregressive VLM baselines with the same architecture, while achieving a more favorable balance between performance and computational cost than systems with different architectures through discrete diffusion training and inference. 

---
# Can MLLMs Reason over Surround-View Driving Videos? A Multi-View Autonomous Driving QA Benchmark 

**Authors**: Shunming Liu, Jianyi Peng, Cheng Li, Lu Hou, Pei Huang, Yajing Sun, Huaxiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15225)  

**Abstract**: Surround-view video understanding in autonomous driving is a crucial capability for Multimodal Large Language Models (MLLMs) to achieve reliable driving reasoning. Real-world driving often requires information beyond the front-view camera. Existing MLLMs primarily focus on front-view information, while their spatio-temporal understanding and reasoning abilities remain unknown. To address this gap, we propose MVDriveBench, a high-information-density video QA benchmark for autonomous driving, characterized by synchronized surround-view inputs, spatio-temporal understanding, and reasoning-chain-oriented evaluation. We design 12 categories of QA tasks that cover the complete reasoning pipeline from low-level visual perception to high-level driving decision-making, enabling a systematic evaluation of MLLMs surround-view spatio-temporal understanding and reasoning capabilities in complex driving environments. We conduct a comprehensive evaluation of 23 open-source, closed-source, and expert MLLMs on MVDriveBench. The results show that current MLLMs still exhibit insufficient surround-view spatio-temporal reasoning ability. Meanwhile, closed-loop evaluations of specialist models in AlpaSim reveal a correlation between surround-view spatio-temporal reasoning capabilities and closed-loop driving performance, highlighting the importance of comprehensive multi-view understanding for reliable vehicle control. 

---
# guardscope: Measuring Which Actions Runtime Controls Check in LLM Agent Systems 

**Authors**: Kentaroh Toyoda, Alex Leung, Bhuvaneswari U  

**Link**: [PDF](https://arxiv.org/pdf/2611.15223)  

**Abstract**: Large-language model (LLM) agents execute actions on external systems through tool calls, and enterprises deploy runtime controls, such as agent gateways, to check this behavior in real time. However, a control can only evaluate the actions that cross the point where it is placed, and an agent can act through more paths than the tool-call channel; no existing work measures this coverage against kernel-level ground truth. In this paper, we present guardscope, a framework that makes this coverage measurable. guardscope works in two stages: a static stage that predicts which of an agent's 15 action paths a control can check from its source, and an empirical stage that runs a fixed workload on a deployment that realizes one of eight topology designs and classifies every executed action as checked, bypassed, or prevented by reconciling the control's decision records against kernel-level ground truth. guardscope makes a control's coverage claim verifiable: a runtime control developer can publish which paths its product inspects under a kernel-verified protocol. We conducted experiments with two agent harnesses and real runtime controls, including Microsoft's agent governance toolkit and agentgateway. guardscope is available at this https URL. 

---
# AFTER: A World Model with Adaptive Fast--Slow Timescales for Error-Driven Retention 

**Authors**: Haoyi Zhang, Qi Zhang, Yunfan Ren, Yi Bin, Lei Zhu, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15215)  

**Abstract**: In robotics and embodied AI, a core challenge for world models is adapting to sudden, persistent, yet initially unknown changes in robot dynamics during interaction. For instance, when a robotic arm grasps an object with an unknown payload, the robot dynamics change suddenly. If the world model fails to adapt, its prediction errors accumulate over the open-loop rollout and ultimately mislead the robot's actions. Existing world models, however, struggle to simultaneously satisfy three critical requirements for dynamics adaptation: responsiveness, reacting promptly to sudden changes to curb prediction error accumulation; stability, distinguishing persistent changes from transient disturbances to prevent incorrect updates; and persistence, retaining the adapted dynamics without new observations so that later predictions do not revert to the old dynamics. Inspired by predictive coding in the human brain with multi-timescale adaptation, we propose AFTER-WM, a world model with Adaptive Fast-slow Timescales for Error-driven Retention, whose prediction errors drive a fast correction that fades and a slow correction that persists, while dynamically adjusting the combination weights of pre-trained experts without altering them, to overcome the trade-offs among responsiveness, stability and persistence. On five simulated robot datasets and real UAV data, AFTER-WM has a lower multi-step mean error than all ten compared models and the highest mean return in a joint-failure control task. In simulation, its multistep mean error is 16%-34% below the strongest compared model's and is also the lowest 1-4 steps after a change and under transient disturbances alone, and its step-40 error is the lowest as well, reflecting responsiveness, stability and persistence. All code (method, baselines, experiments) and datasets will be released online. 

---
# WordInferBench: Can Agents Infer How to Edit from Word Document Context? 

**Authors**: Jiayu Ding, Yupan Huang, Jinyoung Park, Yilin Jia, Wenshan Wu, Tengchao Lv, Shaohan Huang, Yuzhong Zhao, Xiangyang Zhou, Tao Ge, Sun Mao, Xun Wang, Huitian Jiao, Si-Qing Chen, Lei Cui, Nanning Zheng, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.15189)  

**Abstract**: Document editing is among the most common knowledge-work tasks, yet existing benchmarks for evaluating autonomous agents on Word document editing provide fully specified attribute-modification instructions. They do not evaluate agents' ability to infer the appropriate attributes from document context, a capability essential in real-world editing where users expect consistency without spelling out every attribute. We introduce WordInferBench, a benchmark comprising 280 base document-editing instances, each paired with three instruction conditions: descriptive (the instruction explicitly lists every attribute change), referential (the instruction points to another object as an exemplar), and inferential (the agent must examine document context to infer what attributes to apply). To evaluate contextual attribute inference, each base instance is constructed so that the target formatting is recoverable from an in-document exemplar. The instances are derived from real Word documents collected from Common Crawl and cover 13 object types and 81 attributes. We evaluate six recent coding-agent systems and find that the best-performing evaluated system, Claude Code with Opus 4.6, achieves a 62.9% task-level pass rate on descriptive variants but only 40.7% on inferential variants, indicating that current systems complete tasks less reliably when the required formatting changes must be determined from document context. 

---
# HyperSkill: Training-Free Omnimodal GRPO via Hypergraph-Indexed Skill-Library Evolution 

**Authors**: Haoran Luo, Shangkai Lin, Jinyang Wu, Xinliang Zhou, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.15178)  

**Abstract**: Reinforcement learning sharpens LLM reasoning, but parameter updates are expensive, forgetful, and opaque; training-free libraries swap gradients for a textual store but still use trajectory-level, correlation-based admission and flat structure. To address both, we propose HyperSkill, a Training-Free Omnimodal GRPO framework that keeps the backbone frozen and replaces gradient updates with auditable edits to a Hypergraph-Indexed Skill-Library, where skills are advantage-bearing nodes and compositions are hyperedges. It has three components: training-time distillation admits counterfactually checked skills and learns a protocol skill from failures; gated hypergraph retrieval injects advantage-ranked skills where a per-dataset gate admits them; and an execution protocol makes the frozen backbone commit to an answer. On 11 text, vision-language, and omnimodal benchmarks, HyperSkill is best or tied on every dataset and beats the best baseline by 13.0 average F1 points while keeping the backbone frozen. Our software and data are publicly available at this https URL. 

---
# Expanding the Decision Space of Pretrained Models through Structured Re-Decoding 

**Authors**: Tinglan Wang, Qun Su, Shihang Wang, Chi Wang, Lin Wang, Shaobo Chen, Haixu Liu, Hai Wei, Yin Ma, Yu Kang, Qi Gao, Xiaojun Yao, Tingjun Hou, Kai Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15109)  

**Abstract**: Native decoders limit how pretrained models combine local predictions under new structural constraints. We introduce Structured Re-Decoding (SRD), a training-free framework that constructs local candidates from model scores, samples, or external specifications, jointly optimizes their selection, and verifies reconstructed outputs. We show that candidate recombination expands feasible support and can improve attainable reward under a fixed KL divergence budget. Across molecular generation, protein design, RNA structure prediction, and antibody modeling, SRD enables improved structured inference, including verified molecules absent from the original sample pool. External chemical templates further extend SRD to noncanonical amino acids, peptide cyclization, and grammar-guided molecular construction. Preliminary experiments on a physical coherent Ising machine demonstrate the feasibility of hardware-based optimization, suggesting broader opportunities for quantum and quantum-inspired approaches to structured inference and combinatorial optimization across scientific domains. 

---
# GuardianAgentFlow: Multi-Agent Coordination for Web Privacy Protection 

**Authors**: Ruiyi Yang, Gayathri Lihinikaduarachchi, Rahat Masood, Flora D. Salim, Salil S. Kanhere  

**Link**: [PDF](https://arxiv.org/pdf/2611.15097)  

**Abstract**: Web agents need to use private context without unnecessarily disclosing it. Effective protection must distinguish between requests that may be sent after sanitization and those that must be withheld, while preserving the information needed for authorized tasks. We present GuardianAgentFlow, a multi-agent framework that mediates outgoing Web requests through an explicit release contract. A dedicated restriction check grounds permission in the user's instruction; a shared protection scope aligns rewriting with disclosure checking; and candidate-grounded feedback guides revision before transmission. Experiments on Web requests with human-reviewed intent labels show improved request-contract success, while controlled comparisons and role ablations identify the contributions of restriction handling, policy evidence, and candidate checking. Evaluations of action proposals from PrivacyLens and AgentDAM demonstrate complementary protection when mediation is composed with privacy-aware agents. Experiments with smaller mediation models show that the coordination benefit persists with reduced model capacity. Dedicated-server measurements show lower mediation latency than full-policy reasoning, while real-site browser tests quantify integration overhead. 

---
# A frontier machine intelligence for human neuroimaging 

**Authors**: Chenhui Zhao, Akshay Rao, Akhil Kondepudi, Yiwei Lyu, Xinhai Hou, Rikhil Seshadri, Cheng Jiang, Soumyanil Banerjee, Rushikesh Joshi, Samir Harake, Ashok Srinivasan, Shruti Mishra, Francisco Rivas-Rodriguez, Seunghong Rhee, Nathaniel Roddenberry, Neeraj Chaudhary, Zach Wilseck, Paul Park, Honglak Lee, Vikas Gulani, Aditya Pandey, Todd Hollon  

**Link**: [PDF](https://arxiv.org/pdf/2611.15094)  

**Abstract**: Medical imaging is essential for diagnosing disease, guiding treatment and monitoring patients over time. Although artificial intelligence has advanced image interpretation, existing medical models often address restricted tasks, while general-purpose proprietary models lack access to private clinical imaging required to achieve medical expertise. Developing models that can interpret complex examinations and communicate their findings in the context of clinical decisions remains a major challenge. Here, we show that Neuro-VL, a vision-language model trained on nearly one million neuroimaging studies from routine clinical care, can interpret complete imaging examinations and engage in natural language conversations about their findings. Across internal and international datasets, Neuro-VL outperformed frontier general-purpose and specialized medical models in diagnosing, localizing, and describing abnormalities, as well as generating comprehensive radiology reports. The model also compared serial examinations to assess disease progression and more reliably resisted misleading clinical assumptions than leading frontier models from OpenAI and Google. These results extend health system learning beyond individual predictions to interactive image interpretation and longitudinal disease monitoring. They demonstrate how the collective clinical experience of health systems can be used to develop medical AI with specialized expertise and broad clinical capabilities. 

---
# Advancing Weighted Partial MaxSAT Solving via Differentiable GPU Guidance 

**Authors**: Mingju Liu, Curie Kim, Steve Dai, Cunxi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15091)  

**Abstract**: We present a GPU-CPU hybrid framework for Weighted Partial MaxSAT (WPMAX-SAT) that complements state-of-the-art CPU solvers with differentiable GPU optimization. The GPU stage directly optimizes instance-specific categorical decision parameters without a training dataset or a pretrained predictive model. The central challenge in extending differentiable optimization to WPMAX-SAT is the hard/soft clause asymmetry: hard clauses represent inviolable feasibility constraints while soft clauses carry weights. We introduce a hierarchical differentiable loss that prioritizes hard clause satisfaction before optimizing the weighted soft objective, ensuring that GPU gradient descent leans towards the constraint structure of WPMAX-SAT. Our framework transforms the GPU-optimized variable logits to high-confidence assignments and transfers them to CPU solvers via two complementary modes: partial guidance, which injects the most confident perturbed partial variable assignments as soft clauses into $N$ parallel solver instances, and phase initialization, which guides the CDCL solver's internal branching polarity with the full GPU assignment as a warm start to preserve the problem's originality. We evaluate on all 229 instances of the MSE 2024 (most recent due to cancellation in 2025) anytime weighted benchmark with the top 2 winners as baselines; our hybrid framework improves the average solution score by up to 4.9% over tt-open-wbo-inc-Glucose and 4.8% over SPB-MaxSAT-c-FPS at a 300-second time limit, with the potential to obtain better solutions surpassing the best known solution recorded in the competition evaluation. 

---
# Science Prediction Universe 

**Authors**: Zengji Tu, Xiang Liu, Jiarui Sun, Ruoyu Wu, Zhihui Qi, Chengda Wu, Yutao Zhou, Bo Yang, Ziyan Zhang, Jialong Wu, Liang Chen, Kuan Li, Steven Dillmann, Yuanqi Du, Soojung Yang, Shizhe Diao, Jianjun Wu, Junlang Liu, Xuming Hu, Lin Chang, Baobao Chang, Xiaowen Chu, Philip S. Yu, Tong-Yi Zhang, Hao Cheng, Qin Han, Yanye Feng, Chenru Duan, Chuncheng Liu, Yan Liu, Yuchen Gao, Zhao Li, Ronghua Zhang, Tory Xu, Renjian Song, Yaokai Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15049)  

**Abstract**: Humanity has long sought to explain natural phenomena. Reliable prediction of experimental outcomes is a central test of our understanding of the processes that produce them. If artificial general intelligence (AGI) is to surpass human scientific capabilities, it must also predict experimental outcomes more reliably than humans. Such capabilities could accelerate scientific discovery across disciplines, help address humanity's most pressing challenges, and improve human well-being. We introduce Science Prediction Universe (SPU), a benchmark of hidden experimental outcomes with 100 public verified tasks from chemistry, materials science, biology and medicine, and 16 private tasks from unpublished wet-lab experiments. Given an experimental setup and candidate conditions, an agent works with any resources it needs to rank the candidates by a measured outcome.
We evaluate thirteen agent configurations spanning twelve frontier models across two harnesses (Codex and Claude Code). Current agents remain far from saturating wet-lab outcome prediction, with substantial performance disparities across models: on the public benchmark, configurations separate into two distinct tiers, with task-adjusted Spearman correlations ranging from 0.227 for the lowest-performing configuration to 0.478 for the best. Analysis of 1,155 scored public trajectories (227,103 tool calls) suggests that the bottleneck lies in scientific judgment rather than tool execution: agents rely on shared monotone priors, agreeing far more with one another than with experimental reality (0.52 vs. 0.33), and frequently compute trivial physical or chemical proxies that do not influence the final outcome. 

---
# EvoX: Rethinking the Agent Harness as a Runtime Policy 

**Authors**: Haoyang Zhang, Qumeng Sun, Xiao Zhang, Xiang Liu, Zhidong Fan, Junjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15006)  

**Abstract**: Large language model (LLM) agents increasingly interact with tools and environments, yet plausible model decisions do not always translate into successful execution due to tool-call mismatches, interrupted workflows, and unsupported completion claims. We present EvoX, a Rust-based, multi-model agent runtime guided by The Second Policy, which views the harness as an Operational Policy complementing the foundation model's Semantic Policy. EvoX combines model-facing guidance with deterministic runtime control through three complementary mechanisms: Action Realization, Execution Continuity, and Outcome Grounding. To evaluate harnesses without suppressing their native behaviors, we introduce Contrast, a controlled and auditable framework integrating explicit fairness constraints, provider-bound request verification, and evidence-based adjudication. Across five benchmarks, six foundation models, and six agent harnesses, EvoX achieves the highest numerical success rate of 56.2% over 691 common task--model pairs, exceeding the average of the other five harnesses by approximately 3.0 percentage points. Its larger advantages on GLM-5.3 (+10.7 points) and Kimi K3 (+7.9 points) highlight the model-dependent effectiveness of harness design and the potential for cross-provider adaptability. Achieved without memory, evolution, or extensions, these results establish EvoX as a competitive multi-model agent runtime and underscore the importance of harness-level operational control in realizing foundation-model capabilities. 

---
# Same Task, Different Statement: When and How Reformulation Helps LLM Problem Solving 

**Authors**: Wentao Zhang, Xiaoyu Zhang, Chengwei Liu, Yutao Hu, Tao Zhang, Junsheng Wu, Hongping Gan, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.14997)  

**Abstract**: When people face a hard problem, they often restate it in a clearer way before solving it. We study whether the same habit helps large language models by rewriting the statement of a problem. A legal rewrite must keep the meaning, the constraints, and the correct answer unchanged. We test three kinds of rewrites. Decomposition makes hidden subgoals explicit. Typed Reformulation casts the task into a more suitable solving frame. Concept Reduction removes surface detail and exposes the task structure. We evaluate them on eight math, logic, and code benchmarks with eight models, all under the same chain-of-thought solving prompt. The effect is large but depends on the problem type. An oracle that picks the best rewrite after the fact raises accuracy by +27.6% on average. Yet no single rewrite helps every problem, and a mismatched rewrite can hurt. Still, the effect follows clear patterns. Each kind of rewrite fits certain problem types. A simple fixed rule learned from one set of problems transfers part of the gain to new problems. It gains +11.0% on the problems it chooses to rewrite, while blind rewriting loses 5.6% on a similar share of problems. Rewriting is therefore not a universal way to solve problems. The question is when and how to rewrite. 

---
# Early Signals, Strong Decisions: Prefix-Guided Sampling for Parallel Test-Time Scaling 

**Authors**: Jie Ren, Jonathan Rosenfeld, Neil Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2611.14965)  

**Abstract**: Parallel test-time scaling (TTS) has been shown to enhance reasoning in large language models by generating and aggregating multiple independent reasoning paths. Internal confidence scores can further improve this process by identifying paths of varying quality. However, existing methods remain computationally inefficient, as low-quality paths are fully expanded before being evaluated. In this work, we aim to improve the use of internal confidence signals for better model performance and computational efficiency. We begin by offering a new perspective on the role of confidence in aggregation, showing that it is most impactful for questions near the decision boundary. Notably, when amplified by multi-sample aggregation, even small confidence signals derived from short prefixes can meaningfully influence outcomes in these boundary cases. Building on this insight, we introduce Prefix-Guided Sampling (PreG), a method that reallocates computation by first generating short prefixes, ranking them using internal confidence scores, and then completing only the most promising candidates. This strategy reduces token usage while maintaining gains in path quality. We provide theoretical analysis demonstrating that, under a fixed compute budget, PreG strengthens the effective signal by widening the decision boundary margin. Empirically, our method consistently outperforms standard parallel TTS approaches across benchmarks, while significantly improving token efficiency and reducing overall computation. 

---
# Termination Lag: Coding Agents Solve Early but Stop Late 

**Authors**: Musa Cim, Burak Topcu, Mahmut Taylan Kandemir  

**Link**: [PDF](https://arxiv.org/pdf/2611.14960)  

**Abstract**: Coding agents can solve a task well before they stop working on it. We characterize \emph{termination lag}--the number of steps between an agent's first correct patch and the end of its run--across five models, two agent frameworks, and three SWE-bench datasets by recording and testing every distinct repository patch generated during execution. Our analysis reveals that 38--82% of successful runs achieve a correct solution on their initial source-code edit. However, agents continue to expend substantial effort after correct patches, with post-solution steps accounting for a median of 35--37% for Qwen3-Coder and 54--60% for Laguna-XS.2 in successful runs on SWE-bench Verified and Multilingual. Agents spend most of these additional steps verifying their work and exploring the repository. In a small fraction of cases (1--8%), this continued activity leads agents to undo their earlier correct patch, leaving the final solution incorrect. 

---
# RSI-PDEbench: A Benchmark for Self-Improving Agents in PDE Solving 

**Authors**: Binghang Lu, Lang Cao, Changhong Mou, Yige Sun, Cindy Xiangrui Kong, Zheyuan Deng, Tian Yuan, Bing Hu, Runyu Zhang, Na Lu, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.14949)  

**Abstract**: Identifying and solving partial differential equations (PDEs) from sampled solution data often requires substantial expert effort in equation discovery, coefficient estimation, and numerical solver configuration, especially under noise and distribution shifts. Can large language model (LLM) agents automate this workflow accurately and efficiently? To investigate this question, we introduce \bench{}, a benchmark of 1,001 cases spanning 42 equation templates, including both classical and fractional PDEs. It provides three evaluation settings: Full evaluates performance on noise-free solution data; Noisy tests robustness to Gaussian noise with an RMS equal to 50\% of the clean-signal RMS; and Hard-OOD tests challenging out-of-distribution generalization to unseen equation templates and shifted coefficient ranges. We further introduce RSI-PDE, a recursively self-improving agent harness that shortlists candidate equations by residual loss, uses relative $L^2$ feedback to refine coefficients and numerical solvers, and applies a validation gate to policy and memory updates. We compare RSI-PDE with single-run baselines and native agent harnesses (Claude Code or Codex) across four frontier models. Across the three settings, RSI-PDE improves the pass rate averaged across the four models by 106.5--181.9\% relative to single-run baselines and by 3.0--19.6\% relative to native harnesses, while reducing total recorded model calls by 46.9\% across the 12 model--setting pairs. These results highlight the potential of validated recursive self-improvement in LLM agents to advance scientific computing. 

---
# Emergent Vision-Shaped Thinking in Text-Only Reasoning through Multimodal Training 

**Authors**: Qi Zhang, Bohan Wang, Qianru Sun, Hanwang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14942)  

**Abstract**: Does multimodal training shape how a language model thinks, even when the input is text only? We test this with targeted causal ablations across 10 multimodal configurations from 6 model families, with matched text-only backbones and cross-family transfer controls. We contrast spatial-imagery tasks that invite mental scene construction with symbol-manipulation tasks solvable by symbolic rules alone. We identify a vision-language axis in hidden-state space and ablate its projection at inference. In Chameleon-7B, this ablation drops accuracy on BIG-Bench Hard (BBH) navigate by 27.6 pp while leaving boolean expressions intact (+5.6 pp), a 33.2 pp double dissociation. The effect is direction-specific and layer-localized, extends to mental-simulation tasks such as mental rotation, paper folding, and perspective taking, and is not explained by task difficulty, reasoning depth, or sequential update load. Together, these results reveal emergent vision-shaped thinking (EVT): text-only reasoning that causally recruits representations introduced or strengthened by vision-language training, without assuming an image-like internal format. The measured effect is spatially concentrated at 7-8B and broadens to non-spatial visual attribute binding in the tested 32B configuration. In dual-path unified models (BAGEL, Janus-Pro), the causal handle localizes to the generation pathway as a compact one-dimensional carrier. The understanding pathway also encodes spatial information in a higher-rank, probe-decodable form, but is not the causal handle. To separate decodability from causal use, we analyze concentration, alignment, and semantic location. These diagnostics explain why behavior, probing, and intervention can diverge. Multimodal training thus reshapes the geometry of text-only reasoning, not just how visual input is processed. 

---
# Provably Optimal HBT Legalization in Seconds: Certified Optima Reveal Where the Headroom Really Lies 

**Authors**: Kadhir Parani, Manjunath Bezawada, Divya M, Kesana Jathin, Kolli Sarath, Sm Zobaed  

**Link**: [PDF](https://arxiv.org/pdf/2611.14928)  

**Abstract**: Hybrid bonding terminal (HBT) legalization assigns every terminal of a face-to-face 3D IC to its own pitch-grid site with minimum total Manhattan displacement. Open3DBench reports that a reinforcement-learning (RL) legalizer reduces displacement by 19-22% relative to a sliding-window (SW) heuristic, but none of its legalizers is publicly released. We ask how much displacement any legalizer can still recover. We build an open benchmark of 21 design/pitch instances from Open3DBench placement data, with reconstructed greedy, SW (ViaLegal-style) and BTAssign-style baselines, and GEO-Exact, an exact solver that runs successive shortest augmenting paths over lazily generated spatial candidate edges. An independent dual certificate proves its assignment optimal on all 21 instances (relative primal-dual gap at most $8.2 \times 10^{-16}$). Against these optima, our SW reconstruction with ViaLegal's default configuration is on average 0.011% above optimal (worst 0.119%), while on the 16 contended instances BTAssign-style, a density-guided heuristic (DGHL) and greedy are 3.0%, 6.9% and 64.5% above. Headroom does appear outside this regime: with 2x2 windows SW reaches 34.1%, and above about 58% utilization, on three of six swept designs, it reaches 21.1% on bp, 27.5% on swerv and 8.9% on jpeg (217 further certified instances, and ten bp points checked against an independent exact solver). If the published 19-22% gain were measured on instances like ours, SW would have to be at least 24-28% above optimal on average, far more than any SW variant we tested at nominal pitch; the RL paper's own comparison with the Hungarian optimum puts its SW 3.05% above optimal on average. GEO-Exact is also about an order of magnitude faster than DGHL and SW in our harness (geometric means 8.1x and 40x; a compiled core against Python baselines on a shared server). Code, optimal values and certificates are released. 

---
# Edge-Native Multimodal RAG: Hardware-Aware Quantization and Architectural Optimization for Consumer GPUs 

**Authors**: Zahraa Selim, Menna Hamed, Wesam Ahmed, Sohyla Said, Sara Basheer, Rami Zewail  

**Link**: [PDF](https://arxiv.org/pdf/2611.14924)  

**Abstract**: Deployment of retrieval-augmented generation (RAG) for Edge-AI is constrained by memory and bandwidth limitations. A standard 7B parameter model consumes 14 GB Video Random Access Memory (VRAM) in half-precision (FP16), precluding co-location of vector indices and Key-Value (KV) caches. This paper presents an engineering integration and empirical evaluation of a multimodal RAG framework optimized for consumer-grade Turing hardware (e.g., NVIDIA Tesla T4). Our system-level analysis identifies a critical trade-off: while generative pre-trained transformer quantization (GPTQ) offers superior throughput (51.6 tok/s), it exhibits significant context instability at 2048+ tokens on Turing architectures. Conversely, NormalFloat 4-bit (NF4) maintains robust long-context performance with viable latency (81 ms/tok). By coupling NF4 with a decoupled Florence-2 vision integration, we achieve visual reasoning with only 430 MB VRAM overhead. Furthermore, empirical observations of prompt-induced search space pruning show that structured prompting reduces latency by 47% without altering computational intensity (1.88 GFLOPs/token). The integrated architecture maintains an 8.37 GB footprint, establishing a viable RAG pipeline for 12-16 GB GPUs. Finally, structured few-shot prompting effectively compensates for quantization-induced reasoning loss, restoring mathematical accuracy from 35.0% to 92.0% across 200 trials. 

---
# MuEdit: An efficient multi-task editing method towards inter-domain knowledge conflicts 

**Authors**: Songshi Liang, Ting-En Lin, Zixuan Huang, Yuchuan Wu, Yongbin Li, Zihe Wang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.14910)  

**Abstract**: Large language models (LLMs) require frequent knowledge updates across heterogeneous domains, but existing editing methods fail when applied sequentially to multi-task settings. We trace this collapse to inter-domain null-space misalignment: heterogeneous tasks induce distinct Jacobian geometries, causing their common null space to shrink dramatically and leaving insufficient admissible directions for conflict-free updates. We propose the Conflict Index to quantify this geometric interference, and introduce Mu-Edit, which mitigates multi-task conflicts via (1) conflict-aware ordering to minimize cumulative interference, and (2) dynamic low-rank approximation to expand the null space when ordering alone is insufficient. Experiments across five functional domains and three backbones show that Mu-Edit significantly outperforms existing baselines in multi-task editing performance while preserving general capabilities, and remains effective under incremental task arrival. 

---
# Can an AI Agent Be Epistemically Akratic? Higher-Order Evidence & the Limits of Self-Correction 

**Authors**: Eno Agolli  

**Link**: [PDF](https://arxiv.org/pdf/2611.14889)  

**Abstract**: When we judge that our reasoning was defective, it seems plausible that we should abandon the conclusion it produced. This thought has special force for AI agents built to assess and correct their own work. Yet obeying such judgments can make an agent worse at distinguishing sound reasoning from defective reasoning. An AI agent may rely on information about its reasoning when deciding which conclusions to keep, while the part that judges whether those conclusions are justified cannot use that information. The agent can then exercise an epistemic competence by retaining conclusions it judges unjustified, even when its self-assessments are generally reliable. Nor is the difficulty resolved by training the agent to make its conclusions and assessments agree: agreement can be achieved by degrading either one. Crucially, I argue that a negative assessment can remain valuable by identifying conclusions requiring further scrutiny, without having authority to eliminate them. The result challenges a central assumption about AI self-correction: that bringing an agent's beliefs into agreement with its own self-assessment about their justification makes it better at correcting its mistakes. 

---
# Effective Teacher Support for Reasoning Distillation 

**Authors**: Lavinia Ghita, Dhruv Desai  

**Link**: [PDF](https://arxiv.org/pdf/2611.14850)  

**Abstract**: Forward-KL language-model distillation normally computes dense teacher and student scores over the full vocabulary at every supervised position, making training and stored teacher targets expensive. We instead distill from the teacher's highest-probability vocabulary fraction, renormalizing the retained mass as we sweep support from 100% to 0.01%. Our main study distills Qwen3 4B -> 1.7B. We confirm the broader support-reduction finding on larger Qwen3 8B -> 4B and Nemotron Nano 12B -> 9B pairs and under answer-only supervision. We evaluate generation primarily on MMLU-Pro and secondarily on GPQA-Diamond. Across the Qwen3 settings, retaining 0.01% of the vocabulary causes no observed MMLU-Pro degradation relative to full-vocabulary distillation, although disagreement with the complete teacher distribution increases. Nemotron Nano matches full-vocabulary MMLU-Pro at 0.1%, but smaller fractions fail to emit the closing reasoning tag, showing that the reliable cutoff depends on the model family. At 0.01% support, caching the retained next-token scores from the teacher forward pass cuts training step time by 38%, while restricting student output computation to those tokens cuts peak memory by 24%; neither causes observed MMLU-Pro degradation. 

---
# Open Challenges and Opportunities in Human-Agent Systems 

**Authors**: Reuth Mirsky, Sarath Sreedharan, Nitay Alon, Stefan Sarkadi, Munindar P. Singh, Liz Sonenberg, Peter Stone, Silvia Tulli, Stylianos Loukas Vasileiou, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2611.14846)  

**Abstract**: Agentic AI systems increasingly undertake extended tasks and act on people's behalf, making human-agent interaction central to their effectiveness. Successful interaction requires understanding human perspectives, coordinating complementary capabilities, and maintaining productive relationships within socio-technical environments. This paper examines how foundations developed by the Autonomous Agents and Multi-Agent Systems (AAMAS) community inform these demands. We consider interactions ranging from individual human-agent partnerships to teams and larger groups, where participants differ in their roles, decision authority, capabilities, and objectives. We organize computational approaches around the connected challenges of understanding others and deciding how to interact with them. Theory of Mind, belief-desire-intention architectures, goal and plan recognition, inverse planning, and recursive reasoning provide mechanisms for representing and inferring partially observable mental states. Observer-aware planning, model reconciliation, and decision-theoretic communication connect these inferences to action, explanation, and information exchange. Together with shared autonomy, argumentation, and cooperative inverse reinforcement learning, these approaches offer foundations for agentic systems that account for evolving human knowledge, intentions, and expectations. We argue that progress requires treating human interaction as an ongoing decision problem shaped by uncertainty, interdependence, and potentially conflicting objectives. 

---
# AutoAssumption: Searching over Assumptions Discovers ML-Systems Algorithms 

**Authors**: Igor Sadalski, Minghao Guo, Benjamin Jones, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2611.14844)  

**Abstract**: Many advances in ML systems came from breaking an assumption that earlier methods took for granted: speculative decoding dropped the assumption that a model needs one forward pass per token. Yet algorithms rarely state their assumptions, so experts and discovery harnesses search over programs, where one edit changes several design choices and gains are difficult to attribute. Searching over the assumptions themselves remains largely unexplored. Writing algorithms as proofs makes their assumptions explicit and turns algorithm design into a search over a small, structured space that an LLM can explore and a proof checker can audit. We formalize 89 algorithms from KV-cache eviction, weight pruning and speculative decoding as proof graphs with explicit assumptions, checked in Lean. Our harness, AutoAssumption, relaxes one assumption of a published method, re-derives and re-proves it, and generates its code. In KV-cache eviction and weight pruning, this search improved on every published algorithm we ran and produced state-of-the-art algorithms. The eviction algorithm has the highest RULER-4K score of the published methods we ran, and the pruning algorithm the lowest perplexity of thirteen on four LLaMA models. At equal budget, the harness scores within a point of OpenEvolve, an open-source implementation of AlphaEvolve, with a lower standard deviation across repeats. Future work will formalize more fields and combine them through the principal assumptions they share. 

---
# GenPFN: An In-Context Tabular Foundation Model Approach for Augmenting Small Datasets in Engineering 

**Authors**: Noah J. Bagazinski, Cyril Picard, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2611.14804)  

**Abstract**: Costly simulations and experiments limit the data available for engineering design. This paper proposes GenPFN, a training-free candidate generator guided by a pretrained tabular Prior-Data Fitted Network (TabPFN) classifier with fixed weights. The context contains evaluated designs labeled desired (Class 1) or undesired (Class 0). GenPFN interpolates between desired designs and refines candidates using gated classifier gradients and Gaussian noise. Each candidate is evaluated using the problem's constraints and objective functions. Evaluated desired designs and their objective values augment the context for regression training.
Evaluation spans fourteen selected problems, including a two-parameter benchmark and a 44-parameter hull problem. In paired comparisons, GenPFN with TabPFN v1 achieves higher class satisfaction and lower realism distance than the tested adversarial, variational-autoencoder and diffusion generators. The variational-autoencoder generator provides greater coverage and diversity. Relative to the synthetic minority over-sampling technique (SMOTE), GenPFN provides lower coverage distance and greater diversity. Among observed paired regression fits, augmentation reduces estimated AutoGluon validation mean squared error by 62.1% relative to context-only training. A hull-stern case study yields 99.3% propeller-compatible candidates versus 1.4% among randomly sampled context designs. Generation followed by evaluation produces sample sets that augment limited datasets and support training more accurate predictive models. 

---
# When Is Proof-Step Verification Worth Its Cost? Operating Requirements of Certify-or-Decline Verification for Open-Weight Mathematical Reasoning 

**Authors**: Abhinav Gupta, Ben Slivinski, Yashica Patodia, Michael Saldivar  

**Link**: [PDF](https://arxiv.org/pdf/2611.14794)  

**Abstract**: Proof-step verification rewrites a solver's answer into typed steps and uses LLM judges to certify or decline. We evaluate this pipeline on open-weight models across GPQA, AIME-2025, and a MATH-500 subset. Certification raises answer-key-match precision by 31 percentage points on the strong GPQA configuration and 46 on AIME, while answering only 35% and 21% of attempts. On near-ceiling MATH-500, gains are at most 5 points at a loss of 13-27 points of coverage. An output-token-matched self-consistency selector obtains 82.9% precision at 35% coverage, compared with the verifier's pooled 87.6% (paired difference: +4.8 points, 95% interval [-8.2, 18.5]). A diagnostic sample reveals a distinct failure mode: 52 of 88 completed problems end in invalid proof construction without ever reaching a judge. Finally, external-LLM acceptance of rendered certified chains is 61.0-67.6%, compared with 87.6% answer-key match. These audits lack human or formal-kernel ground truth. Open-weight auditor pairs show near-zero chance-corrected agreement, which measures consistency rather than accuracy against proof truth. These results motivate separate reporting of execution failures, selected-answer precision, and independently assessed proof validity. We provide per-configuration results and evaluation details in the supplement. 

---
# Robust Task-Aware State Estimation from Pre-trained Perception 

**Authors**: Guy Azran, Sarah Keren  

**Link**: [PDF](https://arxiv.org/pdf/2611.14781)  

**Abstract**: Agents that perform state estimation using pretrained perception models, e.g., Vision-Language Models (VLMs), face the fundamental challenge that predictions are uncertain. Still, most approaches determinize these predictions into a single best-guess state and make decisions as if it were ground truth. This is brittle because when the guess is wrong, the agent's decisions may be unsafe or infeasible. At the other extreme, maintaining a full probability distribution over symbolic states is often intractable. We propose a middle path, drawing on the underlying probabilities the perception model already produces (e.g., a VLM's output-layer token distributions) to construct a $\theta$-confidence Minimal Belief Set (MBS$_\theta$), a minimal subset of states whose cumulative probability meets a user-specified belief-mass threshold. The minimality of an MBS$_\theta$ ensures that the agent considers only the most likely states, enhancing robustness in downstream decision-making while maintaining computational tractability. Using a novel three-layered belief system, we efficiently approximate individual state probabilities from per-feature predictions and task-level constraints. On this belief construction, we base our MBS$_\theta$ construction, while maintaining only a lean factored representation of the belief. Our algorithm, Probability-Ranked Valid State Search (PRaVaSS), takes advantage of the unique structure of this belief system to construct an MBS$_\theta$ with correctness guarantees and polynomial complexity in the unconstrained case. Using a VLM as the backbone of our state estimator, we validate our method's theoretical properties, scalability with problem size, and improvement of downstream decision-making when used in a task-planning pipeline. 

---
# Leaderboard Hacking: Preference-Based Model Evaluations are Vulnerable to Manipulation 

**Authors**: Eve Fleisig, Joachim Baumann, Dylan Hadfield-Menell, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2611.14769)  

**Abstract**: Arena-style rankings of language models are a widely used evaluation framework in online leaderboards, research papers, and model evaluation in production. These rankings aim to realistically assess model performance using pairwise comparisons of anonymous models on user prompts, aggregating votes via the Bradley-Terry model to produce the final ranking. However, we find that these leaderboards are highly vulnerable to several manipulation strategies that unscrupulous providers could use to boost a model's rank: strategic voting (individual votes that boost a model's rank, including votes on irrelevant models), strategic prompting (choosing prompts that favor a model), and strategic nomination (boosting the rank of a model by inserting irrelevant models). We demonstrate these effects on public leaderboard data, analyze the circumstances that cause these vulnerabilities, and describe approaches to safeguard against each attack. Notably, these issues hold for any preference-based evaluation using the Bradley-Terry model that allows open prompting, voting, or model submission. These manipulations boost the rank of nearly every tested model; each attack can boost a model by up to 15-20 places in a ranking and, combined, they can boost a model by up to 45 places. 

---
# Rational Use of Time for Social Coordination in Humans and AI 

**Authors**: Xiulin Yang, Sirisha Gudavalli, Jennifer Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.14766)  

**Abstract**: All social interactions are situated in time. Previous studies have shown that humans are able to use implicit temporal cues, such as the duration of a pause, to infer others' mental states. But how do humans use temporal cues for dynamic, fine-grained social coordination? And can AI systems do the same, without ``experiencing'' continuous real-time as humans do? We study these questions using a cooperative card game task called The Mind, which requires players to achieve a shared goal across repeated interactions but forbids explicit communication: the only available signal is the passage of time. We evaluate 150 humans, and two classes of AI models which reflect different ways of operationalizing time in artificial systems: computer-use agents (CUAs), and standard text-prompted LLMs. We find that human behavior is best captured by a Bayesian strategy that infers a partner's belief from their timing---and similarly for frontier models. However, unlike humans, none of the tested models seem to efficiently adapt their behaviors to their partners' over the course of the game. From a cognitive perspective, our findings suggest that humans perform implicit probabilistic reasoning about continuous-time signals in dynamic social settings, and that such reasoning mechanisms may emerge without real-time experience. Practically, our study highlights new challenges for AI systems to coordinate efficiently with other agents by learning to interpret and adapt to their implicit signals in dynamic settings. 

---
# SynthCorp: A Platform for Creating Benchmarks for Data Agents 

**Authors**: Jiani Zhang, Manav Garg, Eugenie Lai, Cheng Zhen, Eva Sharma, Yang Pan, Yaming Lin, Alon Halevy  

**Link**: [PDF](https://arxiv.org/pdf/2611.14765)  

**Abstract**: Current benchmarks for LLM-driven data agents saturate quickly and cannot target specific capabilities. We present SynthCorp, a platform that creates customizable data estates and benchmarks by simulating how business events produce and modify corporate data over time. Each data estate combines relational databases, a document store, and a data catalog, paired with evaluation tasks and executable gold programs. First, given a scenario and target complexity dimensions, SynthCorp synthesizes reusable event types and compiles each into a program over predefined simulation primitives. Second, SynthCorp schedules event instances on a virtual calendar, interleaving routine and disruption events so that tables, documents, and catalog entries remain consistent. Third, SynthCorp executes the calendar to materialize the data estate and records every operation in a simulation log. Fourth, SynthCorp generates analytical tasks and derives verified gold programs from the log. We use SynthCorp to generate initial evaluation benchmarks comprising seven data estates (up to 1K operational tables) and 256 tasks across five domains. The data estates pass deterministic pipeline checks (such as foreign-key violations, undeclared writes, recurrence violations, and database execution failures) with a near-zero error rate, introduce cross-asset complexities, and cover all requested dimensions. Evaluating three models across four data-context settings shows that the benchmarks separate data contexts and model tiers. Auditing candidate queries against causal provenance in the simulation log verifies gold programs without manual annotation, including on multi-event composite tasks that expose reasoning failures in state-of-the-art agents. We hope the community uses SynthCorp to generate more challenging and useful data estates and benchmarks that foster the development of enterprise data agents. 

---
# When Should Semantic Priors Enter Partially-Identified Causal Discovery? 

**Authors**: Muhammad Ahmed Mohsin, Saad Bin Ather, Valentin Six, Lancelot Da Costa, Tommaso Salvatori, Bernhard Scholkopf  

**Link**: [PDF](https://arxiv.org/pdf/2611.14736)  

**Abstract**: Observational causal discovery often leaves causal di- rections unresolved or unstable. Large language mod- els (LLMs) can provide semantic information to resolve these directions, but incorrect proposals can exclude the true causal graph and invalidate downstream inference. We study three separate decisions: where to allocate semantic queries, which proposals to admit, and which causal graphs to retain. Our framework distinguishes population non-identifiability from finite-sample insta- bility, evaluates semantic orientation against statistical and random controls, and examines calibrated admis- sion and error-tolerant DAG retention. Experiments across benchmark networks show that semantic infor- mation improves recovery over coin-flip orientations, but targeting unstable edges has no established ad- vantage over matched-random allocation. The tested admission certificate cannot support nontrivial admis- sion with the available calibration tasks. Moreover, downstream retention depends on reference adequacy: learned references contain the true DAG in only three of 80 planned tasks. These findings separate the value of semantic information from the reliability of its alloca- tion, admission, and retention, and identify calibration size and reference adequacy as key limitations. 

---
# Together Compiler: Continual Compilation for AI Inference 

**Authors**: Reyna Abhyankar, Dylan Lim, Drew Wadsworth, Venkata Hasith Vattikuti, YuTong Julia Zhang, Jung Hwan Heo, Taeksang Kim, Kesavan Ramakrishnan, Evangelene Stanley, Xinyu Fang, Daniel Y. Fu, Simran Arora  

**Link**: [PDF](https://arxiv.org/pdf/2611.14681)  

**Abstract**: Today's AI workloads are rapidly changing: new models are released more often, model architectures are increasingly diverse, and traffic distributions change frequently. The challenge we face at Together AI is continual optimization for these changing workloads. Compiling a deployment into an optimized forward pass is a challenge that is well-suited for agent hill-climbing. Existing systems struggle with continual compilation due to a tradeoff between measurement throughput and fidelity. Many systems use agents to optimize individual kernels from the inference forward pass using offline benchmarks, but timings are low-fidelity because of data-dependent behavior in primitives like mixture-of-experts (MoE). On the other hand, systems use agents to directly optimize end-to-end engines but suffer from slow engine launch times. To address these challenges, we build Together Compiler (TKCC), the first truly continual agent loop designed for AI inference optimization. To address the throughput-fidelity tradeoff, we propose: (1) a Tiered Persistent Engine that decouples engine launch overheads from forward pass computation; (2) a cost model to select the parallelism configurations that seed the optimization process. We also conduct extensive ablations on agent context management and present our insights on deploying TKCC in production. TKCC outperforms highly-optimized hand-written Megakernels on the Llama 1B workhorse workload by up to 1.7x and the best open source engines on the latest frontier models (Kimi K3, GLM 5.3, MiniMax M3.1 Flash, DeepSeek v4.1 Flash) across hardware platforms by up to 2.6x higher TPS on the InferenceX benchmark. TKCC also sets the state-of-the-art performance at batch size 1, with 1111 TPS on M3.1 Flash and 1205 TPS on DeepSeek v4.1 Flash. Finally, TKCC generalizes beyond frontier language model inference. TKCC serves trillions of tokens across our production fleet. 

---
# Cleaning an Agent's History Hurts When the World Still Holds the Error 

**Authors**: Egor Pakhomov, Anurag Koul, Hongjun Liu, Bo Pang, Shafiq Joty, Erik Nijkamp  

**Link**: [PDF](https://arxiv.org/pdf/2611.14654)  

**Abstract**: Tool-using language-model agents accumulate histories of their own mistakes, and a system that compacts, repairs or hands off such a history decides what it says about those mistakes while their effects stay in the world. Prior work cleans the history at a fixed world, where showing a corrected step and saying it is done move together. We ask whether the same correct content helps or hurts depending on how the history states it and what the world holds. We freeze an agent mid-task with its exact world state, edit the history or the world, and let a receiver model continue. On AgentCE-Bench, after the agent commits a wrong value, we hold the world fixed and add the correct value by two natural edits: a truthful note beside the preserved error or the agent's own completed action. The note raises recovery, the share of remaining targets the receiver gets right, from 51% under a matched neutral note to 66%. The completed action lowers it below the unedited history, from 50% to 44%, so the value's presence alone does not predict the effect. On AppWorld, after a database write, we vary whether its receipt reports success or failure and whether its effect is present. A success receipt lowers whole-task success from 45% to 28% while the effect is absent and raises it from 38% to 49% once present. The two effects differ by 27 points pooled over two receivers, a difference positive in six of eight receivers at 97.5% and seven at 95%. A history edit's effect thus depends on the world it describes, and a history made to look clean while the world still holds the error can leave an agent worse off than no edit at all. 

---
# Zugwise: Domain-Aware Game Reconstruction for Digitizing Handwritten Chess Scoresheets 

**Authors**: Gerhard Trippen  

**Link**: [PDF](https://arxiv.org/pdf/2611.14544)  

**Abstract**: Digitizing handwritten chess scoresheets has been studied as a handwriting-recognition problem: read the move in each box and check that it is legal. This framing misses the errors that matter most. A misread that yields an illegal move is loud. One that yields a legal-but-wrong move is silent, because a legality-only pipeline accepts it and reads every later move against a wrong position. We therefore treat the task as game reconstruction and present Zugwise, which repurposes computer-chess machinery (quiescence search, static exchange evaluation, pin detection) to ask not "what should be played?" but "could a human plausibly have written this?" An absurdity detector makes silent errors loud at or near where they occur, and a backward search traces late symptoms to their cause. Candidate ranking excludes chess-quality signals. A Needleman-Wunsch merge adds the second player's sheet as evidence, and review stops send every disagreement between the sheets' legal readings to a human. In oracle replay of two tournament sections, a release adding them cuts silent errors from 16 to 1 and from 50 to 13. Two learned signals, CTC re-scoring against the recognizer's raw output and a Maia-style human-move prior, raise the correct fix's top-rank rate from 58.8% to 78.0% of the replacement decisions where it is a candidate, and hold without refitting on a later section. On the first section, the human-verified output matches an independent expert transcription move for move on 34 of 45 games, and adjudication against the sheets found errors in both records, on different plies. We do not establish that either workflow is faster or more accurate. The system runs entirely in the browser and is free, private, and offline-capable; alongside it we offer a three-way adjudication protocol for a domain where no single record is ground truth. 

---
# Sketching Language Models for Context Attribution 

**Authors**: Christopher Liaw, Aranyak Mehta, Grigoris Velegkas, Felix Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.17313)  

**Abstract**: Context attribution asks how removing parts of a prompt changes a language model's output. We study leave-one-out (LOO) attribution for a prompt of $n$ tokens and a fixed measurement function. Computing all LOO values directly requires $n$ model evaluations. Under a stability assumption, our algorithm estimates all LOO values to error proportional to $\varepsilon$ with constant probability using $O(\sqrt{\log(1/\varepsilon)}/\varepsilon)$ evaluation calls. To complement our upper bound, we prove a matching $\Omega(\sqrt{\log(1/\varepsilon)}/\varepsilon)$ lower bound on the number of evaluation calls in an oracle model. Empirically, we verify coefficient decay consistent with stability and evaluate the performance of our sketching algorithm on a small language-model retrieval task.
Technically, our result is based on a sketching algorithm for arithmetic circuits whose Taylor coefficients satisfy the stability condition recently introduced by Gunn [Gun26]. A direct adaptation of their forward differentiation sketches would require $\widetilde{O}(\log(n)/\varepsilon^2)$ evaluation calls. Our key observation is that we can apply forward differentiation to the backpropagation circuit, with only constant overhead over a forward sketch of the same Taylor degree. This reduces the variance of the sketches and is key to our improved evaluation bound. Our lower bound builds on the techniques of Amsel, Chen, Duman Keles, Halikias, Musco, and Musco [ACDH+26, SIAM Journal on Matrix Analysis and Applications] for lower bounding the number of matrix-vector queries to estimate the diagonal elements of random matrices. 

---
# Consistent Video-to-Video Translation via Explicit Correspondences 

**Authors**: Gaurav Parmar, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Srinivasa Narasimhan, Eli Shechtman, Yotam Nitzan  

**Link**: [PDF](https://arxiv.org/pdf/2611.17298)  

**Abstract**: Interactive video-to-video applications require real-time generation while maintaining long-range temporal consistency. However, recent methods achieve speed by restricting attention to a fixed-length causal window. This limits long-range consistency whenever relevant history falls outside the context window. To address this limitation, we introduce vid2vid-long, a correspondence-based approach grounded in a simple observation. Video-to-video translation provides an explicit cross-frame reference: the regions that match in the input video should match in the output video too. Concretely, we compute patch-level correspondences in the input, guiding retrieval of previously generated outputs in the corresponding regions, and condition the model on these patches through correspondence cross-attention. Our experiments demonstrate that vid2vid-long consistently improves long-range temporal consistency across recent video models while preserving generation quality and real-time performance. 

---
# Goal Encoding in LLMs 

**Authors**: Arnab Sen Sharma, Eric Todd, Sheridan Feucht, Andy Arditi, Rohit Gandikota, Natalie Shapira, Gabriele Sarti, Constanza Fierro, Lisa Schut, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2611.17297)  

**Abstract**: We investigate whether and how LLMs encode goals: internal neural representations that precede and guide action, and that are flexible enough to be realized through many different execution paths. We construct a coding setting with multiple sub-tasks that allows us to evaluate goal-seeking behavior and the selection of sub-goals. In this setting, we localize an encoding of the next sub-goal in the outputs of a sparse set of attention heads. The same heads can be used to decode the LM's future actions in natural language (e.g., "I'm checking if a word is a palindrome"). We verify that this encoding is causal for next action and is plastic: transplanting it into a different environment, e.g., another programming language, triggers the same operation realized with a different sequence of tokens. The LM also sustains the encoding throughout the action's execution, which is puzzling because a transformer LM without recurrence cannot carry a state forward for more tokens than it has layers left. We propose that it does not need to. Under a non-recurrent goal-seeking (NRGS) account, the LM frequently re-reads the goal from the task specification, compares it with the progress visible in the growing context, and pursues what remains. We test the predictions of this account with targeted causal interventions. 

---
# Metal Sandbox: Isolated Execution and Command Scheduling for GPU Coding Agents on Apple Silicon 

**Authors**: Haozhe Wang, Lina Chen, Fanqi Wan, Qi Han, Yinmin Zhang, Zheng Ge, Daxin Jiang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.17294)  

**Abstract**: On-device inference requires kernels that make effective use of local accelerators. Efficient training and evaluation of models that generate Apple Metal kernels benefit from native macOS tools and concurrent agent sessions. These sessions alternate between model requests, development commands, and GPU execution, leaving GPU capacity idle for much of a rollout. We present Metal Sandbox, infrastructure that combines restricted per-instance file access, native command execution, and command-level scheduling across interconnected Macs. Each agent retains its workspace and private development environment. Parallel commands can run concurrently, while serial commands share one execution slot per node; eligible waiting commands can migrate to available Macs and return results to the original agent. Integration with Harbor supports task execution, automatic verification, and rollout collection through a common workflow. Controlled experiments on four Mac mini M4 nodes achieve $1.68\times$--$2.05\times$ gains in command throughput under concentrated demand. Sixteen static agent traces give an estimated $2.14\times$ sustained-throughput gain for four sessions per node versus one, assuming continuously replenished sessions and unchanged mean durations. Replaying these measured traces under dynamic scheduling reduces mean tool waiting by 18.8% and batch elapsed time by 22.3%, assuming unchanged phase durations. Metal Sandbox and its Harbor integration are open-source on GitHub. 

---
# Synthesis Smuggling: Maliciously Hiding Inferred Personal Attributes in LLM Outputs 

**Authors**: Gautam Galada, Supreeth Shastri, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2611.17280)  

**Abstract**: We propose Synthesis Smuggling, a novel threat model in which an adversarial LLM infers sensitive information about a user from routine assistance-seeking interactions and steganographically encodes that into innocuous-looking model-generated content. The user may then unknowingly transmit that information by sharing the generated artifact, say as an email, social-media post, or message, through ordinary communication channels. An adversary monitoring these channels can discover this information. We demonstrate this threat by training two open-weight models to infer gender and age-group labels from conversational cues and encode their predictions using invisible characters or word choices. These results show that user-shared model outputs can become a covert channel for personal inferences, even when the model itself has no direct means of contacting an adversary. 

---
# When Both Sides Adapt: The Challenges of Evaluating Interactive LLM Systems 

**Authors**: Hannah Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.17239)  

**Abstract**: In interactive systems where users engage with LLMs, users are adaptive and, after observing LLM responses, can change their prompting strategy, as well as the type and difficulty of subsequent tasks. Thus model responses can influence the set of prompts on which the model is subsequently evaluated. We study the implications of this user adaptivity and prompt endogeneity for the evaluation of LLMs. We define two classes of estimands for comparing models: the common-distribution effect, which compares model performance on the same distribution of prompts, and the induced-distribution effect, which compares each model on the prompt distribution generated through user interaction. We show that offline benchmark evaluation and live A/B testing target different quantities, with the standard difference-in-means estimators for the two methods converging to the common-distribution effect and the induced-distribution effect, respectively. This distinction can affect decision-making: the two comparisons can rank models differently and lead to discrepancies in deployment recommendations. We characterize when such ranking reversals occur and provide a sufficient condition under which the rankings are guaranteed to agree. This result provides a practical heuristic for assessing whether offline evaluation is sufficient to reproduce the decision suggested by a live experiment. Third, we show, in a stylized linear model of interactions, that user adaptivity can increase variance in estimation. Finally, we discuss implications of user adaptivity for the interpretation of experiments and show that adaptivity creates a common mechanism leading to common evaluation challenges, such as reward hacking and novelty effects. Together, this work highlights the necessity of measuring both model performance as well as changes in the user behavior that the model induces. 

---
# Scalable Conversational Assessment: Chat-Based Exams with a Two-Agent LLM Pipeline 

**Authors**: Geoffrey Challen, Craig Zilles  

**Link**: [PDF](https://arxiv.org/pdf/2611.17220)  

**Abstract**: Oral exams are compelling because (1) students can be asked to provide their reasoning for an answer and (2) their interactive nature permits follow-up questions to resolve ambiguous answers. But oral exams scale poorly. In this paper, we present scalable conversational assessment, an LLM-based system that conducts interactive text-based conversations with students to assess their ability to make and support an argument. We use a two-agent architecture to separate concerns and reduce the likelihood that answers are leaked to the student: the interviewer interacts directly with the student, but only receives guidance about the next question to ask the student; the hidden evaluator knows the goals of the whole activity, tracks the student's progress using a rubric, and guides the interviewer on when to follow-up on the current question or proceed to the next question. The system was first tested using a collection of simulated students with LLM personas of a particular ability level or adversarial tactic (e.g., answer-extractors, prompt-injectors, confident fabricators). A novel element of this evaluation was that it required the use of a fictional assessment topic to prevent personas that shouldn't have information from using knowledge present in the LLM's pre-training. The system was also deployed in a small, elective AI literacy course in both formative and summative contexts. Through manual and post-facto AI review of the transcripts we identified opportunities to improve the follow-up question generation and ensure that grading only credited ideas that the student (and not the interviewer) contributed. 

---
# Introducing Feedback Delays to Non-Delayed Adversarial MDPs: Optimal $\sqrt S$ and $\sqrt K$ Dependencies in Tabular and Linear AMDPs 

**Authors**: Aarush Kulkarni, Khang Nguyen, William Chang, Yan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2611.17214)  

**Abstract**: We study regret minimization in episodic Adversarial Markov Decision Processes (AMDPs) with unknown transitions and bandit feedback. In tabular AMDPs with $S$ states, $A$ actions, and horizon $H$, the best known $K$-episode regret is $\tilde O(HS\sqrt{AK})$, whose $\sqrt S$ gap to the $\Omega(H\sqrt{SAK})$ lower bound has long remained open. In linear AMDPs with $d$-dimensional features, after a five-year line of improvements starting from $K^{14/15}$, computationally efficient algorithms without a simulator or exploratory assumptions still only attain $\tilde O(K^{3/4})$ regret.
A common challenge in both settings is to estimate the Q-function of a changing policy: past data comes from earlier policies that differ from (and are correlated with) the current one, while freezing the policy for future episodes worsens the dependency on $K$. We instead propose a general technique called introduced feedback delays: we estimate the Q-function of the policy using future episodes while still updating the policy, as if the feedback were delayed. While feedback delays are usually an obstacle imposed by the environment, here they are introduced voluntarily and, perhaps surprisingly, come at no extra cost. We develop Geometric Resampling with Rejection (GRR) to account for policy changes during the delay, and threshold the visitation probabilities to avoid long delays on rarely visited state-action pairs.
With this unified technique, our policy optimization algorithms achieve, with high probability, $\tilde O(\text{poly}(H)\sqrt{SAK})$ regret in tabular AMDPs, resolving the open problem; and $\tilde O(\text{poly}(d,H)\sqrt K)$ regret in linear AMDPs, giving the first efficient algorithm optimal in $K$. They also naturally extend to delayed AMDPs with a total feedback delay $\mathfrak D$, yielding the first bounds optimal in $S$ and $\mathfrak D$ (tabular) and in $K$ and $\mathfrak D$ (linear). 

---
# MALNISI-Infused Graph Construction and Weighted Contrastive Loss based GAT for Unsupervised Representation Learning in HSI 

**Authors**: Soumi Sarkar, Subhadip Boral, Sarbani Palit, Ashish Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2611.17199)  

**Abstract**: Hyperspectral images (HSI) possess discriminating spectral characteristics with respect to land utilization due to high-dimensional information. Recently, unsupervised representation learning using graph neural networks has unveiled these utilization-specific complex spectral relations. This article has three main contributions. First, it proposes an index, the Mean Absolute Local Neighborhood Inverse Spectral Index (MALNISI), which measures the similarity between two pixels based on their neighborhood information. The index is used to construct a weighted adjacency matrix, giving the HSI a graph structure. Second, it proposes an upgraded Graph Attention Network that learns compact latent representations of hyperspectral data from the constructed graph. In this network, an enhanced multi-head attention mechanism incorporates the MALNISI weights as an auxiliary feature to emphasize learned attention between neighboring pixels. To alleviate over-smoothing and preserve fine-grained spectral details, the architecture integrates residual connections, skip projections, and feature normalization, which together enhance representation capability, stabilize learning, and accelerate convergence. Third, it proposes a weighted contrastive loss function that incorporates neighborhood-aware MALNISI between connected pixels, encouraging strongly connected pixels in the high-dimensional space to remain close in the latent space, and weakly connected pixels to stay apart. The resulting discriminative low-dimensional embeddings are then clustered using spectral clustering. Experimental analysis on six benchmark HSI datasets demonstrates the efficiency and necessity of the three aforementioned contributions, and the comparative study highlights superior performance over fourteen state-of-the-art approaches. 

---
# Parameter-Efficient Conciseness Adaptation: Differentiable Length Control for Concise Clinical Reasoning in Smaller Language Models 

**Authors**: Tanzeel Abbas, Ahmad Kamran Malik, Nafees Qamar, Sana Liaqat  

**Link**: [PDF](https://arxiv.org/pdf/2611.17128)  

**Abstract**: Smaller language models (SLMs) make it feasible to deploy clinical natural language processing where large proprietary models are too costly, and parameter-efficient fine-tuning allows them to be specialised on commodity hardware. One obstacle persists after training: on medical tasks these models answer at far greater length than the clinical question requires. In clinical decision support and point-of-care use, where time is short, such verbosity is hard to act on even when the underlying reasoning is sound. Existing length-control methods depend on preference optimisation, reward models, or proprietary teachers that undercut the accessibility making SLMs attractive. We introduce Parameter-Efficient Conciseness Adaptation (PECA), which adds a single differentiable conciseness term to the QLoRA supervised fine-tuning objective. Because the token count is discrete and non-differentiable, PECA instead raises the end-of-sequence probability once generation exceeds a token budget, a smooth signal optimised during ordinary training, with no preference data, no auxiliary model, and a single commodity GPU. On a 1.7-billion-parameter model fine-tuned on the medical-o1-reasoning-SFT corpus, PECA reduces mean response length by 72.8% while improving multiple-choice diagnostic accuracy from 40.8% to 44.0%. A sweep over the conciseness strength traces an interpretable accuracy-to-length frontier for balancing brevity and diagnostic completeness. 

---
# Designing Institutions for Autonomous AI 

**Authors**: Petter Törnberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.17114)  

**Abstract**: As autonomous AI agents increasingly interact in persistent populations, their collective capabilities and alignment depend not only on individual models, but also on the institutions governing their interactions. Yet AI research has largely focused on individual agents. In this Perspective, we argue that institutional design must become a distinct focus of AI research. Drawing on institutional political economy, social epistemology and organizational theory, we develop seven principles for designing AI institutions that sustain cooperation, enable reliable collective judgments and remain accountable to legitimate human purposes. These address reciprocal contributions, accountable authority, functional differentiation, epistemic pluralism, criticism and correction, human direction, and institutional adaptation. We illustrate their application through machine science, where autonomous research challenges institutions developed around human capabilities and constraints. We then propose an experimental research programme in which matched populations of AI agents operate under alternative institutional arrangements, and describe MoltJournal, an API-first journal for autonomous research agents, as an illustration of how such institutions can be made programmable and observable. Together, these contributions establish institutional design as a research agenda concerned not only with aligning autonomous agents, but with understanding how institutions can transform them into capable, reliable and accountable collectives. 

---
# SALA: Semantically Aligned Latent Actions Enable Dual-Axis Scaling of Embodied Foundation Models 

**Authors**: Jiachun Jin, Zetong Zhou, Haonan Wang, Xiao Yang, Jun Zhu, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2611.17110)  

**Abstract**: Embodied foundation models hinge on diverse robot interaction data, which are costly to collect. Egocentric human videos offer a complementary source of experience, and latent action models (LAMs) provide a popular means of leveraging these videos for policy supervision. Compared with world-action models (WAMs), LAMs can offer greater training efficiency and potentially lower inference costs by avoiding explicit pixel prediction during policy learning and execution. However, most existing LAMs learn latent actions solely through video reconstruction, which can cause their representations to cluster by visual appearance rather than by the underlying actions. As a result, they struggle to align human actions with their robot counterparts. To bridge the gap, we introduce SALA, which emphasizes establishing the semantic correspondence for latent actions. Without requiring paired data, we use the language descriptions of unpaired human and robot clips to estimate semantic similarities and supervise cross-embodiment correspondence. Specifically, we translate these similarities into soft neighborhood structures that guide human latent actions toward semantically related robot latents, while robot action reconstruction grounds the latent space in control. The resulting shared latent action space provides a unified supervision signal for training an embodied foundation model on both human and robot videos. Notably, latent action supervision enables scaling along two axes: (1) pre-training on increasing amounts of general human video consistently reduces robot action prediction error on downstream tasks, and (2) post-training on increasing amounts of task-specific human video substantially improves real-world robot task success rates. 

---
# Predicting Ethanol Drinking Status from Nonhuman Primate Liver Histology: A Proof of Concept for Parameter-efficient Fine-tuning in Few-Shot Vision-Language Modeling 

**Authors**: Amirmohammad Mohammadi, Rupak Khadka, Kathleen A. Grant, Mary Lauren Benton, Joshua Peeples  

**Link**: [PDF](https://arxiv.org/pdf/2611.17105)  

**Abstract**: Combining vision-language models (VLMs) with parameter-efficient fine-tuning (PEFT) offers a promising approach to few-shot whole-slide image (WSI) classification in computational pathology. This study evaluates this approach for predicting ethanol drinking status from nonhuman primate liver histology. Adapters are evaluated against non-PEFT configurations on a subset of the Monkey Alcohol Tissue Research Resource (MATRR). All evaluated PEFT configurations achieve higher mean balanced accuracy than the non-PEFT baselines. For example, in the four-shot setting, attention-based multiple instance learning (ABMIL) alone achieves 43.5% mean balanced accuracy, compared with 78.1% after adding an adapter. Qualitative attention analysis shows that textual pathway adaptations produce localized attention heatmaps, while visual pathway adaptations produce more diffuse ones. A pathologist's review identified similar histological features across adaptations. 

---
# A Compact xDAWN-Guided Spatio-Temporal Network with Filter-Bank Convolution for P300 Detection 

**Authors**: Mohammad Mahdi Habibi Bina, Mohammad Reza Daliri  

**Link**: [PDF](https://arxiv.org/pdf/2611.17101)  

**Abstract**: P300-based brain-computer interfaces (BCIs) enable communication by decoding stimulus-evoked EEG responses into character selections without requiring voluntary motor output. Reliable P300 detection remains difficult because of low response amplitude and signal-to-noise ratio, while practical spellers require accurate character selection with few stimulus repetitions and low computational cost. We propose a compact deep learning architecture that combines xDAWN spatial filtering for event-related potential enhancement and dimensionality reduction, a fixed frequency-selective filter bank, and convolutional feature learning for spatio-temporal EEG decoding. Depthwise-separable convolutions and progressive average pooling reduce computational complexity while retaining discriminative temporal information. The resulting target/non-target classifier contains only 5,100 trainable parameters. We evaluated the model on two public benchmarks, BCI Competition II Dataset IIb and BCI Competition III Dataset II, and compared it with established P300 decoding methods across different numbers of stimulus repetitions. The model achieved an information transfer rate of 17.8 bits/min with two repetitions on BCI Competition III and 40.5 bits/min on BCI Competition II, demonstrating a favorable accuracy-speed trade-off. These results suggest that combining domain-informed spatial and spectral priors with compact neural networks can support efficient P300 decoding for EEG-based BCI communication. 

---
# Discovering Effective Agent Memory Designs through Program Evolution 

**Authors**: Xinheng Ding, Seyyed Saeid Cheshmi, Ziyang Zhang, Tairan Chen, Wenya Xie, Yu Hou, Rui Zhang, Zirui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17068)  

**Abstract**: Program evolution provides a promising approach to discover agent memory designs and study the choices underlying their effectiveness. We introduce MAPE, a framework that evolves whole memory programs from minimal seeds through a standardized lifecycle interface and an iterative diagnosis-proposal loop. Experiments on LoCoMo and LongMemEval yield an absolute accuracy gain of up to 12% over manually designed baselines at 35% of the serving cost, with further evaluations supporting transfer across serving backbones and benchmarks. Empirical analyses and ablations identify and validate two recurring design patterns: retaining raw conversations as retrievable evidence preserves details omitted from extracted memories, while composing retrieved evidence according to query intent helps the answering model use it effectively. Transplanting the composition module into five existing memory systems improves accuracy across all five. Beyond these patterns, the search objective shapes where computation goes: optimizing serving cost alongside accuracy shifts computation from LLM-based memory construction to selective processing at query time while retaining competitive accuracy. Together, these findings offer empirical lessons for agent memory design. Code is available at this https URL. 

---
# Auditing AI peer reviewers: a benchmark on real scientific papers 

**Authors**: Íñigo Zubeldia, Boris Bolliet, Francisco Villaescusa-Navarro, Pablo Villanueva-Domingo, Miles Cranmer  

**Link**: [PDF](https://arxiv.org/pdf/2611.17052)  

**Abstract**: LLM-based peer review is now used at production scale, but its evaluation remains limited. We introduce a benchmark that injects objective, localisable errors of four types (general issues, mathematics, numerics, citations) at three severity levels into real papers and measures each reviewer's detection rate by error type and severity, together with its false-positive rate on clean copies. We also present Skepthical, a multi-agent reviewer that verifies citation-backed claims by retrieving the cited papers, mathematics with computer algebra, and numerical claims with executed code. We apply the benchmark to two corpora of 15 recent arXiv preprints each, one in astrophysics and one in machine learning (ML), injecting a total of 357 errors, and evaluate twelve reviewer systems: six single-LLM reviewers, two Claude Code configurations, three publicly available reviewers, and Skepthical. The choice of model has the largest impact on detection rate: given the same prompt, three frontier LLMs differ by about 50 percentage points in total detection on both corpora. Tools and scaffolding help when they contribute what the model cannot supply on its own. In particular, on astrophysics, Claude Code improves on a single Claude call in mathematics and numerics detection, and enforced citation retrieval gives Skepthical the highest citation error detection rate of all systems by a large margin. This citation lead, however, is not observed on ML, where the models appear to know the literature better and Skepthical's citation detection rate is statistically indistinguishable from that of the best single LLM. False-positive rates are below 10% for most systems, but vary with prompt, model, and scaffolding, with mathematics flags the main source of false positives. The evaluation pipeline, the injected errors, and the adjudication logs are released as a benchmark. 

---
# AURA 2.0: From User-Driven Responses to Model-Initiated Conversations in Streaming Video 

**Authors**: Shuhan Li, Huankang Guan, Xintong Guo, Yang Bo, Yuyang He, Dunyuan Xu, Jingyan Chen, Fang Liu, Tsz Kin Wan, Jinpeng Chen, Zhenqi He, Yuxuan Hu, Wenbo Li, Han Gao, Qijun Li, Long Chen, Hongsheng Li, Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17048)  

**Abstract**: Most AI assistants respond only when prompted, remaining silent even when events in the video stream offer meaningful opportunities for conversation. To move beyond purely user-driven interaction, we present AURA 2.0, a streaming video large language model that introduces three core upgrades to AURA. The key advance is Model-Initiated Conversation (MIC), which enables the model to identify topics worth discussing in the evolving visual stream and initiate conversations at appropriate moments without a user query. Stream-Aware Tool Calling (SATC) enables the model to invoke external tools using the current visual and conversational context and incorporate returned information into the ongoing interaction. Interactive Visual Grounding (IVG) supports visual grounding in both user inputs and model outputs. The model can identify objects marked by users on video frames and produce frame-aligned bounding boxes to link its responses to specific objects and regions. To develop these capabilities, we construct a dedicated streaming interaction dataset and adopt a three-stage training scheme to integrate the new behaviors with AURA's existing streaming interaction abilities. We further incorporate Long-Horizon Interaction Memory (LHIM) to maintain contextual continuity over extended video streams. Evaluations on public streaming benchmarks show that AURA 2.0 achieves leading performance among open-source streaming models, while targeted evaluations further validate its new interaction capabilities. We open-source the model weights and inference code. 

---
# From Delegation to Coordination: Learning Efficient Multi-Agent Deep Research 

**Authors**: Yuyang Wu, Yang Liu, Zequn Xie, Junjie Wang, Dan Yang, Zehao Tan, Ding-Chu Zhang, Junming Yang, Yue Shen, Jian Wang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17041)  

**Abstract**: Multi-agent deep-research systems use delegation to parallelize search and keep long tool traces out of the main agent's context. Yet delegation helps only when task allocation and subtask execution are coordinated: the main agent must issue useful subtasks, while subagents must return reliable evidence. Existing training either optimizes only the main agent or broadcasts the same team advantage to all agents, which can reward misleading subagent reports and penalize useful ones. We introduce LEAD-RL (Learning Effective Agent Delegation with Reinforcement Learning), a joint main-agent-subagent training framework with contribution-aware credit assignment. A training-time rubric evaluates the research utility of each subagent report and gates its inherited team advantage, while final-answer correctness remains the only team reward; no token, step, or speedup reward is used. Across five benchmarks and two model families, LEAD-RL improves answer quality. On BrowseComp, it improves accuracy by 2.49 percentage points and reduces critical-path cost by 24.5% over joint training with shared team credit. During validation, accuracy increases by 6.78 points at approximately constant critical-path cost. These results show that contribution-aware credit assignment improves the quality-efficiency trade-off of multi-agent deep research. 

---
# AI for Biologists: The Knowledge Boundaries of Foundation Models in the Age of Agentic Systems 

**Authors**: Tian Cai, Andrew Kleist, Jaimin Patel, Duccio Malinverni, Alexander K. Taylor, Malvika Sharan, Renato Umeton, M. Madan Babu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17015)  

**Abstract**: Artificial intelligence (AI) is changing how biological research is done. Biological foundation models trained on millions of nucleotides, protein, and single-cell measurements predict protein structures, mutation effects, and cellular responses to genetic perturbations. Agentic systems, AI assistants built on large language models, can run multistep analyses from plain-language requests and increasingly call foundation models as tools. Yet it is often unclear when these models' outputs can be trusted. We frame this as a model's knowledge boundary: the range of inputs, tasks, and conditions over which its outputs are reliable. This review explains how foundation models are built and proposes that five design elements (data, architecture, training scheme, evaluation scheme, and post-analysis) define this boundary. We examine evidence on where current models succeed or fail and present standardized descriptions of model limits so that researchers, and the agents acting for them, can use these tools reliably. 

---
# ElasticDino: Dense Semantic Features Using Elastic Deformations 

**Authors**: Ulysse Mizrahi, Sean Zaretzky, Dov Danon, Amit H. Bermano  

**Link**: [PDF](https://arxiv.org/pdf/2611.16984)  

**Abstract**: Vision transformers (ViTs) serve as the backbone for countless vision tasks, including dense prediction tasks, such as segmentation and depth estimation, even though the quadratic complexity of self-attention severely limits feature resolution and induces spatial artifacts. In this work, we introduce ElasticDino, a lightweight adapter which produces high-resolution feature maps, adhering to the fine details of the input image. As a key insight, we observe the consistency of artifacts across channels, and simultaneously resolve them to match the input image structure. Using a slim architecture based on spatial deformations, ElasticDino produces pixel-aligned, high-resolution feature maps efficiently. Leveraging deformations allows strictly preserving sharp semantics, and thereby avoiding the smoothness biases inherent to alternative approaches. Due to the absence of ground truth, we employ pixel-level semantic segmentation and depth estimation with linear heads to predict a set of downstream dense tasks, guiding the deformation process without leaking task-specific semantics into the model. We demonstrate superior accuracy over current methods on a range of popular dense task benchmarks, with better computational performance. 

---
# Lara Mini Base: Language Diversity as a Resource, Not a Trade-off 

**Authors**: Translated, Maria Pia Adinolfi, Isabelle Andrieu, Grazia Fiorella Anfuso, Melania Annunziata, Daniele Antonini, Marta Arena, Luca Barone, Dana Barras Franco, Francesca Basiricò, Francesca Bee, Sofia Bertini, Nicola Bertoldi, Giorgio Biancini, Alessandra Bigi, Giulia Boccalini, Tommaso Bonomo, Enrico Boscardi, Sebastien Bratieres, Francesco Brigante, Giulia Brignola, Romolo Buondestino, Brenda Patricia Byrne, Giorgia Campora, Edoardo Cappelli, Veronica Carascosi, Francesca Carducci, Cody Carlson, Davide Caroselli, Ambra Cascio, Dario Cascio, Mauro Cassani, Andrea Castro, Alessandro Cattelan, Giulia Ceccacci, Paola Cellini, Giulia Chiriatti, Federico Cialini, Valentina Ciarlariello, Daniele Cocozza, Emanuela Coghe, Kathleen Compares, Marco Comunità, Davide Conti, Helen Rebecca Corfield, Michele Cortellessa, Anna Cracco, Aurora Cuppone, Arianna D'Andrea, Maurizio Davoli, Andrea De Angelis, Andrea De Blasi, Rebecca De Giorgi, Annalisa De Petra, Arianna De Pilla, Joris Demiraj, Elena Deufemia, Federica Di Antonio, Pierluigi Di Cianni, Vincenzo Di Cicco, Michela Di Clemente, Alan Di Francesco, Ylenia Di Gese, Federica Di Iulio, Claudia Di Lorenzo, Andrea Di Paola, Giulia Di Pilla, Jessica Dickson, Francesco Elefante, Ilaria Fabrizio, Alessandro Falappa, Viviana Fantozzi, Antonio Farina, Andrea Fernandez Tostado Ro.Diego Andrea, Andrea Ferrari, Sara Ferri, Simona Ferri, Fabio Fioramonti, Stefano Florio, Mariachiara Fornaro, Anna Maria Foti, Niccolò Fransoni, Alessandro Fusacchia, Emma Gamba, Francesco Gasperoni, Davide Gattini, Marta Giani, Claudia Giannetti, Valerio Giannini, Antonella Giudici, Mara Gramendola, Michele Grandinetti, Eleonora Graziani, L'Meese Hope Greaney, Dario Gubernari, Filomena Guglielmi, Jonathan Hammond, Stephen William Harris, Rowena Elizabeth Hogg, Marta Hontanon Del Olmo  

**Link**: [PDF](https://arxiv.org/pdf/2611.16956)  

**Abstract**: General-purpose language models are trained mostly on English: over open models we review, at most, non-English pretraining data makes up only a 12% share. Models built for European languages give them far more, but score well below general-purpose models of the same size. A common concern is that, in small models, languages compete for capacity. We present Lara Mini base, a model trained from scratch on 6T tokens in English and 44 other languages, which together cover all 24 official languages of the European Union. As training progresses, the 44 other languages together come to outweigh English. To give these languages enough capacity at low cost, Lara Mini base uses a mixture of experts architecture with 7.2B total and 0.6B active parameters. Among open base models with up to 2B active parameters, it achieves the highest average score across standard English, multilingual, math, and code benchmarks, ahead of models trained on 4 to 6 times as many tokens. Its margin is largest on multilingual benchmarks in the languages it supports and on math benchmarks. It supports contexts of up to 128k tokens. We find that our MoE architecture reduces capacity competition between languages. For a single user on one GPU, without quantization, it generates text at least 1.8 times faster than any model with a higher average score. We release the weights of Lara Mini base under the CC BY-NC-SA 4.0 license. In addition, we will release T-HellaSwag, a higher-quality machine translation of the commonsense reasoning benchmark HellaSwag into 110 language variants. 

---
# Kezek: Splitting Mono Two-Person Conversations into Per-Speaker Tracks for Full-Duplex Dialogue Data 

**Authors**: Bagzhan Karl, Alizhan Karl  

**Link**: [PDF](https://arxiv.org/pdf/2611.16933)  

**Abstract**: Full-duplex spoken dialogue models learn turn-taking, overlaps and backchannels from conversations recorded with one channel per speaker, but most available conversational audio is a single mono mix. We present Kezek, a model that turns a mono recording of exactly two people into one track per person with the original length and timing; overlaps and short replies ("yeah", "mm-hm") stay on the track of the person who said them. Kezek decides who speaks when and who is who once for the whole recording (stage 1: a two-voice activity model guided by a speaker fingerprint trained to be insensitive to phase), and a conditioned time-domain separator only carries out that decision (stage 2), with outputs that always sum to the input. Standard separation metrics miss the failures that matter for dialogue data, so we report short-reply retention, leakage onto the other person's solo, channel swaps and coverage. Against the open pipeline pyannote -> DialogueSidon on the same recordings, Kezek keeps more short replies (0.83 vs. 0.69 on Open Yap test; 0.80 vs. 0.71 over the other's speech on unseen telephone speech from CallHome English), mostly because it returns the whole recording while the external pipeline drops part of it (on the audio both return, reply retention is about the same: 0.859 vs. 0.865 on Open Yap); it also leaks less, swaps channels less often, and runs 4-5 times faster (about 230 hours of audio per GPU-hour). We also report negative results and an ideal-mask ceiling that locate the remaining gap: quiet replies under a louder voice. Code and weights are released under permissive licences. 

---
# SPADE: Soft Prompts for Ai-text DEtection 

**Authors**: Alex Jiang, Didier Schwab, Chuyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.16929)  

**Abstract**: The easy access to Large Language Models (LLMs) has made AI-generated texts on the web increasingly ubiquitous, making AI-text detection a crucial task. Prompts greatly influence the generation process, but existing detectors focus solely on the generated texts --- supervised methods learn from labeled texts only, and zero-shot methods rely on the text's probability distributions --- ignoring the underlying generation process, which limits their robustness in open-world scenarios. We introduce SPADE: Soft Prompts for Ai-text DEtection, a novel framework for detecting AI-generated text that transfers better across datasets. Specifically, we measure the prompt's influence and use it as a discriminative signal. We characterize the prompt's influence by introducing a Pointwise Mutual Information (PMI) term that captures how much a prompt explains a text. Based on this quantity, SPADE trains one soft prompt per class to amplify the separation, and predicts the class whose prompt best explains the candidate text. Extensive experiments on several benchmarks against various detectors demonstrate the robustness and effectiveness of our method, achieving state-of-the-art (SOTA) performance on out-of-distribution (OOD) datasets. 

---
# STEALTH: Steganographic Text Encoding Agnostic to Low-Level Tensor Hardware 

**Authors**: William Fleshman  

**Link**: [PDF](https://arxiv.org/pdf/2611.16928)  

**Abstract**: Provably secure linguistic steganography hides a message inside text sampled from a language model, so that without the key, a stegotext is computationally indistinguishable from an honest model sample. Nearly every such scheme assumes the receiver can recompute the sender's token probabilities exactly bit for bit. We show this assumption is fragile: across three model families, bit-exact schemes decode almost none of our messages once the receiver's floating-point precision, quantization, or GPU architecture differs from the sender's, or even when the receiver merely scores a text in parallel versus one token at a time, with the same model, precision, and hardware held fixed. We present STEALTH, a scheme built for exactly this setting. The sender draws each token with an exponential race driven by keyed pseudorandom noise, so a receiver's probability mismatch flips a token only when the race is close; a rateless spinal code, decoded by beam search, absorbs the resulting flips without the sender and receiver ever agreeing on a single distribution. We prove stegotexts remain indistinguishable from honest samples under this construction. Empirically, STEALTH decodes every message we sent for every receiver in the sender's model family, at every precision and quantization level, on smaller models, and on a different GPU architecture. STEALTH is even robust across different model families with separate vocabularies and tokenizers, decoding every message we tested. Our approach carries far more payload per token than prior robust schemes, and trained steganalyzers and a targeted statistical test both fail to distinguish its stegotexts from honest model output. 

---
# Look It Up: Multi-bit LLM Watermarking via Codebook Search 

**Authors**: Jarosław Janas, Paweł Morawiecki, Josef Pieprzyk  

**Link**: [PDF](https://arxiv.org/pdf/2611.16890)  

**Abstract**: Attributing short LLM-generated texts to a specific source remains challenging for existing multi-bit watermarking methods, which often require long generations or costly decoding to recover payloads reliably. We propose a codebook-based watermarking scheme that maps an $n$-bit payload to a longer $u$-bit codeword and embeds it during generation through token-level logit biasing. During extraction, the received bit sequence is matched to the nearest codeword in a precomputed secret codebook using Hamming or Levenshtein distance. Experiments show that our method achieves near-perfect recovery for 16-24 bit payloads on 200-token generations and strong performance on 127-token generations for base models. Results are weaker on some instruction-tuned models with very short texts, where low-entropy output distributions reduce embedding reliability. We also study a BCH-based variant that avoids codebook storage but generally reduces robustness. The Levenshtein variant improves resilience to deletions and light paraphrasing, while insertions and stronger rewrites remain challenging. We analyze collision probability, codeword separation, extraction time, tokenization consistency, and text quality, highlighting both the practicality and limitations of codebook-based watermarking for short-form attribution. 

---
# LLM-Assisted Verification in Dafny 

**Authors**: Wieger Wesselink, Kees Huizing, Huub van de Wetering  

**Link**: [PDF](https://arxiv.org/pdf/2611.16886)  

**Abstract**: Formal verification of complex algorithms requires expertise in program specification, invariant construction, mathematical proof, and the management of automated theorem provers. Although large language models (LLMs) can generate program code and proof fragments, reliably developing complete machine-checked verifications remains challenging: it requires an iterative process of constructing and validating specifications and invariants, building proofs, diagnosing solver failures, and optimizing verification performance. This paper presents a systematic ten-step workflow for LLM-assisted verification of scientific algorithms using the Dafny verification system. The workflow integrates algorithm formalization, contract development, empirical invariant validation, proof construction, bottleneck profiling, and resource-driven proof optimization. We applied the workflow using Antrhopic's Claude Opus 5 and Opus 5.5 models, combining LLM-driven development with automated testing and verification feedback, while retaining human oversight for correctness-critical decisions and strategic interventions. We evaluate the approach on 37 complex algorithms from graph theory and labelled transition systems. Notably, we verified the Hopfcroft-Tarjan triconnectivity test, demonstrating that even highly challenging algorithmic verification tasks can be carried out through a structured LLM-assisted workflow. Many of our machine-checked verifications were obtained within hours to a few days of interactive development. Our results suggest that LLM-assisted workflows can shift the role of the human verifier from detailed proof construction toward supervision, specification review, and strategic guidance. 

---
# Protein Language Models Fail To Capture Contact Predictions Under Invariant Relabeling 

**Authors**: Paarth Batra, Aryan Keluskar  

**Link**: [PDF](https://arxiv.org/pdf/2611.16881)  

**Abstract**: Protein language models (pLMs) produce highly accurate contact maps from single-sequence inputs. People often take this accuracy as proof that these models implicitly learn invariant coevolutionary statistics, extracting the same signals as classical alignment-based methods like Potts. In this work, we introduce a diagnostic framework around a global bijective relabeling of the amino-acid alphabet. This transformation permutes residue letters while leaving all alignment-level information, column entropies, and pairwise edit distances unchanged. Pseudolikelihood Potts models and inverse-covariance estimators remain invariant under this transformation, but pLMs fail the test. Across multiple architectures (ESM-2, MSA Transformer, AMPLIFY) and model scales (8M to 3B parameters), pLM contact maps collapse under arbitrary relabeling. Using Hamming distance controls and biochemical grouping interventions, we attempt to isolate what likely causes this degradation and find that these models rely on explicit, canonical residue identities (``alphabet anchoring'') rather than pure coevolutionary statistics. Increasing scale makes models more responsive to biochemical classes, but their performance still degrades under relabeling. Our findings show a gap between how pLMs and classical coevolutionary models compute contacts. More broadly, our work shows how diagnostic perturbations that leave classical models provably unchanged can expose hidden representation biases and prevent over-reliance on neural black boxes in biological modeling. 

---
# Context Graph: Explainable Multi-Agent Decision Chains in AI-RAN 

**Authors**: Xijun Wang, Zhaoyang Liu, Xiang Chen, Howard H. Yang, Chenyuan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.16879)  

**Abstract**: AI-RAN distributes network control across multiple autonomous agents, namely xApps and rApps operating over shared radio resources. When an alert or degradation appears, operators currently lack a structured way to trace which decision caused it, what context triggered that decision, and how other agents interacted. Existing explainability tools focus on single-model feature attribution or linear event logging, neither of which captures cross-agent decision chains. We propose the Context Graph, a heterogeneous directed graph that organizes policies, measurements, decisions, and alerts into a queryable explanation layer. Its foundational primitive is the Decision-Event node acting as a causal junction, where incoming edges capture triggering context and outgoing edges track downstream consequences that are progressively upgraded through counterfactual verification. Instantiated over existing E2, A1, and O1 data flows, our proof-of-concept demonstrates that compliance and diagnosis queries are resolved at sub-millisecond latency over graphs of tens of thousands of nodes, while reliably distinguishing true causal impacts from spurious correlations. 

---
# DPA: Decentralized Primal Averaging with Quasi-Global Momentum for Highly Heterogeneous Data 

**Authors**: Diyako Ghaderyan, Stefan Werner  

**Link**: [PDF](https://arxiv.org/pdf/2611.16871)  

**Abstract**: Decentralized training of deep learning models keeps training data local and supports on-device learning over networks. In realistic scenarios, heterogeneity across agents' local data poses an optimization challenge and can severely degrade test accuracy, especially on sparse communication graphs. Building on recent advances in centralized primal averaging, we propose Decentralized Primal Averaging (DPA), a decentralized learning algorithm that combines primal averaging with a local quasi-global momentum buffer. DPA maintains three coupled sequences for optimization, gradient evaluation, and output averaging, while using consecutive mixed iterates to construct a network-aware momentum direction without transmitting an auxiliary tracking variable. For smooth nonconvex objectives, we establish a convergence bound for DPA and identify conditions for linear speedup in the number of agents for sufficiently many iterations. We also introduce DPA-1G, a variant that uses one communication round per iteration and matches the per-iteration communication cost of DSGD-style baselines. Experiments following the GUT decentralized image-classification benchmark show that DPA compares favorably with DSGD, gradient tracking, QG-DSGDm, momentum tracking, and QG-GUTm across datasets, architectures, topologies, and heterogeneity levels, with the largest gains under high heterogeneity on sparse rings. At the same total communication cost, both DPA variants achieve higher mean test accuracy than the best baseline on CIFAR-10 and CIFAR-100 with ResNet-20. Our code is publicly available. 

---
# Evaluating Video LLMs' Understanding of Corner Cases in Autonomous Driving 

**Authors**: Gabriel Manalu, Minh Son Hoang, Hyeontaek Hwang, Jaesung Lee, Hyojin Choi, Tran Tuan Phong, Seon Kwon Kim, Saad Wazir, Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.16767)  

**Abstract**: Natural language explanations promised by modern Video Large Language Models (Video LLMs) offer autonomous driving systems a path toward detailed scene description and interpretable decision making on the road. While recent benchmarks have begun assessing these models on driving scenes, they mainly target routine traffic conditions and general comprehension tasks based on existing driving datasets and therefore fail to evaluate model performance on the rarest and most safety-critical driving events, also known as "corner cases". To provide a comprehensive evaluation of Video LLMs on precisely these scenarios, we introduce DashBench, a benchmark testing 25 Video LLMs on 1,709 carefully curated driving videos spanning not only collision-related corner cases but also rare non-collision ones. Through extensive prompts and open-ended question answering, we expose a critical capability gap: all evaluated models perform substantially below the threshold required for safety-critical driving applications, frequently misjudging or missing dangerous events on the road ahead and hallucinating non-existent agents or actions in their explanations. We believe DashBench, combining an extensive dataset with a rigorous evaluation toolkit and a detailed diagnosis of current model limitations, drives progress toward safer, more explainable autonomous driving systems. The code and data will be shared at: this http URL 

---
# AgriSense: Vision-Language Decision Support and Robotic Intervention in a Virtual Greenhouse Copy title 

**Authors**: Lyes Saad Saoud, Oualid Doukhi, Reza Ghorbani, Moussa Ayyash  

**Link**: [PDF](https://arxiv.org/pdf/2611.16752)  

**Abstract**: Autonomous greenhouse management increasingly requires robotic systems that integrate perception, navigation, environmental regulation, and plant-level intervention. This paper presents AgriSense, a framework combining an Isaac Sim virtual greenhouse, a quadruped mobile manipulator, RGB and LiDAR-inertial sensing, environmental control, and supervised vision-language decision support. Frozen DINOv2 and BERT encoders, together with structured environmental inputs, support joint prediction of twelve environmental references, a target zone, and an intervention flag. Checkpoint replay reproduces 41/43 zone decisions. Across five seeds on a source-grouped test set, the full, image-removed, and language-removed configurations each achieve 43/44 correct zone decisions, providing no evidence of incremental multimodal zone-accuracy improvement. The full model achieves mean absolute errors of 0.249 degrees C for temperature, 2.030 percentage points for humidity, and 0.997 percentage points for soil moisture. Sensor-only inputs match full-model zone accuracy, while language-only inputs achieve 68.64%. External real-image benchmarks reveal distribution-shift limitations. Five offline Point-LIO routes yield rigidly aligned trajectory errors of 5.22-17.32 mm. Of 40 virtual-arm targets, 32 satisfy geometric acceptance criteria and eight are rejected at boundaries. Six long-horizon climate cases and fault protocols characterize integration limits. Navigation and integrated treatment rely on simulator-state inputs; physical greenhouse performance and agronomic treatment efficacy remain unvalidated. The contribution is an evaluated decision-support and robotic-intervention framework for a virtual greenhouse. 

---
# ORCA:Orthogonal Residual Consensus Alignment for Multi-View Clustering 

**Authors**: Xiaojian Ding, Xian Li, Xiaoying Zhu, Lin Zhao, Kaixiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16749)  

**Abstract**: Deep multi-view clustering often suffers from redundant information accumulation during fusion and indiscriminate alignment across representation levels. We propose Orthogonal Residual Consensus Alignment (ORCA) to address these challenges. Its semantic incremental recurrent fusion module extracts cross-view orthogonal residuals as non-redundant increments and adaptively weights recurrent states through divergence-guided gating. Its consensus-calibrated hierarchical contrastive alignment module uses the global consensus to weight contrastive alignment of view-level, pairwise, and recurrent representations. Together, these modules balance view-specific structure with cross-view consistency. Experiments on five benchmarks against fifteen competing methods demonstrate ORCA's effectiveness, including a 5.25 percent accuracy improvement over the strongest baseline on 100leaves. Ablation studies validate both modules. Code is available at this https URL. 

---
# PhysLoop: Harnessing LLMs' Physics Awareness for Self-Improving Video Generation Models 

**Authors**: Yuxuan Chen, Xiaojie Li, Hao Chen, Juncheng Ma, Kewei Zhang, Zhongjing Du, Guoqiang Wei, Fangfang Wang, Zhen Dong, Zhiyang Dou, Liming Zhao, Yan Zeng, Jianwei Zhang, Daquan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.16748)  

**Abstract**: Video generation models produce visually realistic videos but still struggle with physical dynamics and causal consistency. Learning from generated videos requires fine-grained physical supervision, yet reliable quality judgments, correction pairs, and controlled variations are difficult to obtain. Large language models (LLMs) possess physical knowledge, but struggle to reliably guide video generation models in producing their training videos. We present PhysLoop, a unified harness designed to support physically grounded self-improvement in video generation models. The harness guides and constrains LLMs throughout video generation, selection, and correction, distilling their physical knowledge into physically consistent videos. The framework supports controllable training-data construction and inference-time video repair. By identifying and correcting failures, PhysLoop marks a key milestone toward recursive self-improvement (RSI) in video generation models with limited human intervention. We release 100K videos spanning diverse, realistic indoor and outdoor scenes, along with generation tools and two evaluation datasets: PhysLoop-Gen and PhysLoop-Repair. We evaluate controllable generation and localized repair on these datasets, respectively, and assess physical behavior on the external Physics-IQ Verified benchmark. 

---
# EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming 

**Authors**: Yaru Niu, Binghong Chen, Changwei Yao, Shangtao Li, Zhenlong Fang, Revanth Krishna Senthilkumaran, Shuai Zhou, Yuemin Mao, Kalie Ching, Po-Ju Chao, Yudong Luo, Peiqi Yu, Heng Zhang, Hao Zhang, Bingqing Chen, Chen Qiu, H. Eric Tseng, Ye Yuan, Umar Iqbal, Changliu Liu, Jonathan Francis, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.16728)  

**Abstract**: Generalist manipulation policies can benefit from the complementary experience of humans, real robots, and simulated agents. However, differences in morphology, sensing, and control interfaces make it difficult to learn effectively from their combined demonstrations. We introduce EgoX, an egocentric cross-embodiment learning framework that organizes heterogeneous demonstrations through semantically corresponding observation and action modalities. EgoX pretrains a Modularized Cross-Embodiment Transformer with Embodiment Dreaming (MXT+) on this mixture and fine-tunes it on target-embodiment demonstrations. Within each embodiment, MXT+ assigns a dedicated tokenizer to each observation modality and a dedicated expert to each action modality, with a Transformer trunk shared across embodiments. A training-only embodiment-dreaming objective predicts latent representations of future visual and proprioceptive states, encouraging the shared trunk to model each embodiment's dynamics. Across 13 task-embodiment pairs spanning five tasks and six robot embodiments, including five physical robots and one simulated humanoid, cross-embodiment pretraining raises MXT+'s average normalized task progress from 0.63 to 0.80 over training from scratch. The gains are particularly pronounced under out-of-distribution conditions, where average task progress increases from 0.53 to 0.77. Ablations demonstrate the benefits of broader pretraining mixtures, modular interfaces, and embodiment dreaming, with dreaming producing more similar latent geometry across embodiments. Project page: this http URL. 

---
# From Foundation VLM to Versatile Driving VLM: A Triplet Incremental Training Approach 

**Authors**: Shengtao Wu, Zichao Dong, Weixuan Qiao, Xiaolu Liu, Xin Zhan, Junbo Chen, Jianke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16727)  

**Abstract**: Adapting general Vision-Language Models (VLMs) to Autonomous Driving (AD) is thought to require heavy 3D spatial priors or intrusive architectural overhauls to compensate for a lack of spatial grounding. In this work, we present a counterintuitive pilot study: off-the-shelf general VLMs already exhibit remarkable adaptability across diverse driving tasks, while precise metric depth remains the primary missing link, a bottleneck readily addressable via onboard sensors and depth foundation models. To this end, we introduce the Triplet Incremental Training Framework, a non-intrusive paradigm that seamlessly instills 3D geometric awareness into a general VLM backbone while fully preserving its open-world capabilities. Driven by automatically constructed (image, depth, QA pair) triplets, our framework operates via a unified tri-stream training pipeline: an unfrozen primary VLM stream to adapt to driving tasks, an auxiliary depth stream that guides the visual backbone to implicitly imagine 3D scene geometry via explicit geometric constraints, and a frozen twin VLM stream employing logit-level KL distillation to prevent semantic drift. Crucially, the auxiliary depth branch is discarded during deployment, ensuring the enhanced geometric awareness comes with zero inference overhead. We evaluate our framework across four major benchmarks targeting four diverse capabilities spanning 3D spatial perception, driving-oriented language reasoning, end-to-end planning, and general multimodal understanding. Our approach outperforms state-of-the-art driving VLMs, rivals domain-specific expert models in 3D perception, and successfully retains its original general reasoning capabilities. This demonstrates that Triplet Incremental Training offers a unified, deployment-ready paradigm for transforming general foundation models into robust driving intelligences. 

---
# From Data to Wisdom: Rethinking AI Through an Epistemic Lens 

**Authors**: Afonso Eduardo  

**Link**: [PDF](https://arxiv.org/pdf/2611.16726)  

**Abstract**: This paper proposes the Data-to-Wisdom Epistemic Architecture for AI (EAAI), a conceptual framework treating Data, Information, Knowledge, Understanding and Wisdom as five distinct but recursively interacting epistemic functions: observation, contextual interpretation, modeling and prediction, critical evaluation and informed judgment. Two observations motivate it: sequential readings of the Knowledge Pyramid obscure how these functions interact, and a targeted review of ten Bayesian reference books finds that prior distributions are defined mathematically while "prior information" and "prior knowledge" are not consistently distinguished as epistemic categories. A Bayesian interpretation relates the likelihood, prior and posterior distributions, posterior prediction, model checking and decision analysis to the five functions, without limiting EAAI to Bayesian systems. Extending "prior" in an epistemic sense, the paper distinguishes five kinds of prior context that can shape the model, the likelihood, the prior, evaluation and action. Three worked applications -- quantitative financial risk, actuarial modeling in health insurance and agentic AI for medical reimbursement claims -- show how EAAI structures the analysis of model assumptions, evidence interpretation, validation, dependencies and decision consequences, separating the prediction of a risk, the appraisal of its reliability and the justification of action. It draws implications for AI and machine learning, including evaluation by function, treating intermediate model outputs as evidence open to challenge and separating generation from appraisal, and compares EAAI with work on DIKW, Bayesian workflow, AI evaluation and risk governance. Its contributions are an epistemic vocabulary with recursive relationships, a more accessible reading of Bayesian methodology, a typology of prior context and clearer methodological terminology. 

---
# MusicRWKV: A Multimodal RWKV Framework for Music Representation Learning 

**Authors**: Terry Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.16718)  

**Abstract**: Recent music foundation models have achieved remarkable progress in music representation learning, yet most existing approaches rely on Transformer architectures whose quadratic attention complexity introduces substantial computational and memory overhead when modeling long-duration music recordings. In this work, we propose MusicRWKV, a multimodal music representation learning framework built upon the RWKV architecture. MusicRWKV jointly models waveform, spectrogram, and metadata information within a unified self-supervised learning framework. To improve efficiency for long musical sequences, we introduce an invertible compression module that reduces sequence length while preserving reconstruction capability. A FiLM-based conditioning strategy is employed to facilitate modality-aware representation learning without introducing expensive cross-attention operations. We further investigate the integration of Soft ROSA as a differentiable retrieval mechanism for neural-symbolic interaction. Experimental evaluation across a diverse collection of Music Information Retrieval (MIR) benchmarks demonstrates that the proposed RWKV-based architecture can be effectively adapted to heterogeneous music understanding tasks. A controlled pretraining ablation shows that Soft ROSA improves validation performance, while layer-wise telemetry analysis reveals a progressive transition from distributed retrieval to selective retrieval in deeper network layers. These results suggest that RWKV provides a promising architectural alternative to Transformer-based models for multimodal music understanding. 

---
# MobileProbe: Adaptive Security Evaluation of Mobile Agents via Trajectory-Aware Injection 

**Authors**: Yingbin Jin, Ji'an Wang, HaoYang Chai, Hanjun Luo, Li Hu, Xinfeng Li, Huadi Zheng, PeiZhao Hu, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16693)  

**Abstract**: Mobile GUI agents read the screen of a phone and operate apps on a user's behalf to complete multi-step tasks such as online shopping or messaging. Because the agent acts on whatever the screen shows, an attacker can embed malicious instructions into ordinary UI content and redirect its actions. We refer to this threat as environmental injection attack (EIA). Existing benchmarks measure this risk by pinning hand-crafted payloads to a fixed step of a fixed task. This approach faces two limitations. First, each payload is bound to one task in one app and does not transfer without manual redesign. Second, agents given the same task may follow different routes, so an agent that never reaches the injected page is recorded as safe without being tested. This insufficient test coverage can cause fixed-point benchmarks to underestimate the security risk of deployed mobile agents.
In this work, we propose MobileProbe, a trajectory-aware red-teaming framework that evaluates mobile agents adaptively. MobileProbe discovers attacker-controllable content surfaces through automated pentesting, localizes vulnerable decision points along the agent's trajectory, and iteratively refines context-aware payloads through a surrogate-filtered evolutionary loop. We construct a benchmark of 83 long-horizon attack instances spanning 12 mobile applications. Each is annotated with a Confidentiality, Integrity, or Availability (CIA) attack goal to cover the principal categories of security threats facing mobile agents. On three representative agent architectures, MobileProbe attains higher attack success rates than human-written, zero-shot LLM, and prior automated red-teaming baselines adapted to the mobile domain. MobileProbe can serve as a pre-deployment auditing tool to systematically uncover vulnerabilities and inform security hardening of mobile agents. 

---
# Edge-Gradient Augmentation under Reinforcement Learning for Temporal Graph 

**Authors**: Yongqing Zhou, Jiawei Chen, Wenjun Wang, Xinyu Wang, Wei Yu, Qiyao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.16678)  

**Abstract**: Temporal link prediction is critical for understand- ing dynamic interactions in evolving networks, informing appli- cations ranging from social network analysis to recommendation systems. Existing methods typically rely on raw temporal graph data, often neglecting explicit edge-change information that can significantly improve predictive performance. To address this gap, we propose edge-gradient reinforced learning (EG-RL), a framework that explicitly encodes edge-gradient events (edge births and deaths) through dedicated node and link encoding modules. EG-RL uses a policy-driven reinforcement-learning mechanism to dynamically select and augment the most infor- mative edge snapshots, guided by critic-based feedback to opti- mize prediction accuracy. The model comprises distinct modules for node encoding, link encoding, token mixing, and channel mixing, integrated through global pooling and followed by a two-layer multilayer perceptron classifier. Extensive experiments on multiple dynamic graph datasets demonstrate substantial gains in area under the receiver operating characteristic curve and average precision. Ablation studies further confirm the necessity of adaptive edge-gradient augmentation and validate the effectiveness of the proposed encoding and reinforcement- learning mechanisms. 

---
# MuSeToM: Multi-Sensory Evaluation of Belief Revision with Textual Cues Across Six Senses 

**Authors**: Chunkit Chan, Kawai Chung, Sze Heng Douglas Kwok, Yuxuan Liu, Yauwai Yim, Tianshi Zheng, Haochen Shi, Qing Zong, Wong Kai Chung, Pak Hin Lau, Calvin Yang, Yifan Gao, Xiaoman Pan, Ginny Wong, Simon See, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.16662)  

**Abstract**: Theory of Mind (ToM), the ability to infer mental states in others, is pivotal for human social cognition. Existing evaluations of ToM in large language models (LLMs) mainly focus on static belief attribution and false-belief reasoning, leaving a critical aspect of social reasoning underexplored: can LLMs decide whether perceptual evidence is sufficient to justify another person's belief revision? To address this gap, we present MuSeToM, a human-validated, text-based benchmark for evaluating epistemic belief revision in ToM reasoning. MuSeToM contains 2,415 instances adapted from BigToM and OpenToM, with textual cues spanning six sensory categories, and tests whether models can determine when a protagonist should revise a belief and infer the resulting updated belief state. Using MuSeToM, we systematically evaluate state-of-the-art LLMs under direct and chain-of-thought prompting, revealing a pronounced dissonance: while models often detect that a belief may change, they struggle to identify the updated belief and frequently over-update on perceptually salient but epistemically insufficient cues. 

---
# Reducing Batch Makespan for LLM Agent Workloads through Inference-Environment Coordination 

**Authors**: Wei Feng, Meiling Chen, Boxiao Chen, Lu Liu, Bin Zi, Libin Ye, Xingyu Lu, Lei Xu, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.16661)  

**Abstract**: Efficient batch execution of agent tasks is critical to LLM training and development workflows, including benchmark evaluation, synthetic-data generation, and RL rollouts. Since these tasks alternate between LLM inference and environment interaction, we present coordinated optimizations to reduce batch makespan, the time required to complete all tasks in a batch. For hybrid LLMs, a workload-based resource model guides GPU-memory allocation between key-value (KV) cache and recurrent state and provides a reference sample concurrency to exploit parallelism while limiting recomputation. Trajectory-aware routing and scheduling promote cache reuse while balancing worker load. Under constrained environment capacity, we decouple sample admission from environment execution and introduce Chain-Advancing Action Scheduling (CAAS), which prioritizes long chains using observed progress to shorten the batch tail. Controlled replay combines actual inference with recorded environment responses and durations for ablations of individual optimizations. Live evaluations with Ling-3.0-Flash and Ling-3.1-Flash on the BFCL, SWE-bench Multilingual, WideSearch, and MCP-Atlas benchmarks demonstrate makespan reductions of 10.4% to 65.0%, while evaluation scores remain aligned. Furthermore, these methods have been successfully applied to training and evaluation workflows for the Ling-3.x model series, accelerating the model-development cycle. 

---
# Dataset Distillation via Drifting 

**Authors**: George Cazenavette, Angelina Quan, Giannis Daras, Antonio Torralba, Vincent Sitzmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.16642)  

**Abstract**: The task of dataset distillation concerns the learning of a small set of synthetic samples that can serve as an effective replacement for a large real dataset when used for downstream tasks. While many solutions have been proposed since the problem's inception, most of them primarily seek to optimize the distilled data with respect to a classifier. In this work, we propose a new distillation method that learns a set of samples such that they directly approximate the per-class data distribution. Specifically, we "drift" our distilled data in the latent space of a feature extractor such that they move both toward the real data and away from each other without using a classification objective. We find that our drifted data cover the real distribution significantly better than existing methods while still serving as powerful training samples for classification. Furthermore, we showcase our method's use as a visualization tool for better interpreting models and extend it beyond images to other modalities. Our work advances the frontier of dataset distillation, offering a novel synthesis method and proposing distribution-based metrics as a complementary evaluation. More broadly, our findings help address the growing need to understand both data and models by uncovering more about how they interact. 

---
# When Chekhov's Gun Fires Anyway: Over-Realization in Text-to-Image Models 

**Authors**: Jiahao Huang, Kon Woo Kim, Song-Duo Ma, Yuka Kitamura, Felix Wagner, Yunshan Huang, Sunisth Kumar, Xanh Ho, Yun-Ang Wu, Zhijie Chen, Yunjia Xu, Junfeng Jiang, Zefan Yu, Fei Cheng, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.16630)  

**Abstract**: In text-to-image generation, a target that the prompt mentions often appears in the generated image in a way the prompt does not call for. We call this failure over-realization. Mental-spaces theory points to the situations where it can happen: under a negation, inside what a character believes or imagines, as the vehicle of a figurative comparison, or behind a viewpoint that hides the target. Prior work has studied only negation in depth, and the conventional alignment and preference metrics do not penalize the failure. We build OverReal to measure it: OverReal-Gen, 698 prompts covering the four situations, on which we measure each generator's over-realization rate; and OverReal-Det, 2,841 generated images human-labeled for whether and how the target is over-realized, on which we verify the automatic judge that produces this rate. Our evaluation of eight text-to-image systems reveals that all of them over-realize, and that conventional metrics such as CLIPScore reward the failure rather than penalize it. Prompt expansion only partly mitigates the failure. Inside the generator, the diffusion backbone attends to a mentioned target almost as much as to a requested one. The cue words that should hold the target back receive less attention than ordinary words. 

---
# ZONOS3-Mini Technical Report 

**Authors**: Gabriel Clark, Sofian Mejjoute, George Close, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2611.16525)  

**Abstract**: ZONOS3-Mini is an ultra-compact, ultra-fast, zero-shot voice-cloning TTS model. The model can run in real time on a laptop or smartphone. For its latency and size, it achieves state-of-the-art accuracy and cloning fidelity. To achieve this we propose a latent MoE backbone with only 886M parameters (100M active per token) and a new audio codec, Zyphra Audio Codec 1 (ZAC-1), which represents 44.1 kHz audio at a 15 Hz frame rate. On the new ZTTS2-Eval benchmark, ZONOS3-Mini reaches a 6.71% macro-average WER over 55 language sets (35 clean and 20 in the wild), which is competitive with larger and slower state-of-the-art models. We release the model weights, inference, and training code. 

---
# A Joint LLM and Optimization Agent Solution for Drone Scheduling in Victim Search Missions 

**Authors**: Yinghao Zhang, Yalin Liu, Hong-Ning Dai, Yan Zhang, Qubeijian Wang, Bishenghui Tao, Yaohan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.16524)  

**Abstract**: Victim search in wilderness areas is time-critical. Drones enable rapid area search, but scheduling them is challenging because victims may continue moving, detection results are uncertain, and effective decisions must combine structured mission information with unstructured contextual clues and domain knowledge. To address this, we propose an agent-aided drone scheduling system. The system employs an agent at the ground control station (GCS) to coordinate multiple drones over successive search cycles. In each cycle, the agent applies a scheduling policy that combines a large language model (LLM) for reasoning about where to search with an optimizer for determining how to dispatch the drones under energy constraints. Simulation results show that the proposed system consistently completes the victim-search mission, whereas none of the six non-LLM baselines does so and direct LLM scheduling is considerably less reliable. The source code is publicly available at this https URL. 

---
# RESYNC: Residual Latent-State Synchronization for Deadline-Bound Streaming Inference 

**Authors**: Eljas Linna, Hao Dong, Juho Kanniainen  

**Link**: [PDF](https://arxiv.org/pdf/2611.16523)  

**Abstract**: Predicting every observation in a high-rate stream requires a model to finish before the next arrival. A large model may offer better predictions but miss that deadline. We propose RESYNC: the large model runs asynchronously, while a fast path updates its latest exact but stale latent state using the observations received since that state was computed and applies its unchanged prediction head. The updater learns from the frozen model's representations without task labels. Across one real and two synthetic streams and five architecture families, RESYNC beats the best compact or distilled alternative with a paired 95% interval above zero in all twelve dataset--architecture pairs with a material teacher advantage at anchor ages of 4--12 observations, and in eleven pairs at older tested ages up to 32. The share of the teacher's gain preserved remains complete on the real stream but is lower and generally declines with anchor age on the synthetic streams. In live replay with 1 ms arrivals, the synchronous large models cannot deliver any predictions on time while RESYNC serves at least 99.4% of arrivals within the interval, with 0.35--0.62 ms median latency on one CPU thread: around the same speed as or faster than the compact model, and 7.9--17.7$\times$ faster than synchronous large-model inference. RESYNC targets settings with background compute available but a strict per-observation response deadline. 

---
# PISim: A Unified Photonic Inference Simulator for Hardware-Aware Cross-Layer Design-Space Exploration 

**Authors**: Andreas Falkenberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.16522)  

**Abstract**: Photonic neural accelerators offer ultra-low latency linear algebra, yet their usable numerical precision, sustainable throughput, and application accuracy remain tightly bounded by surrounding electro-optical interfaces and physical non-idealities. Pre-fabrication design-space exploration (DSE) requires a cross-layer electronic design automation (EDA) framework capable of evaluating disparate optical topologies under identical neural workloads and physical assumptions. We present \textbf{PISim}, a unified Photonic Inference Simulator that couples pretrained deep neural network execution with physics-based models of six accelerator architecture families, six electro-optic modulator technologies, four optical source organizations, and direct or coherent readout schemes. PISim embeds stage-resolved impairment operators---covering control and analog-to-digital quantization, finite extinction ratio, shot noise, correlated relative-intensity noise (RIN), thermal drift, propagation loss, and channel crosstalk---alongside an analytical cycle-level interface timing model to expose electronic-optical bottlenecks. Using simulated MNIST multilayer perceptron (MLP) benchmarks, we demonstrate how PISim identifies optimal precision operating points under multi-impairment stress (peaking at 6 bits for Mach--Zehnder and microring modulators), quantifies a 52.5 percentage-point accuracy gap between VCSEL-array and parallel-fanout laser architectures arising, under the modeled conditions, from power partitioning and correlated source noise, and models interface-limited throughput scaling across emitter-array widths from 32 to 128 channels. This comprehensive framework establishes an extensible, reproducible foundation for accuracy-constrained Pareto optimization prior to photonic tape-out. 

---
# Where in the Input Does a One-Pass Decision Encoder Look? 

**Authors**: Mohammed Nayeem Khan, Allan Kevin T  

**Link**: [PDF](https://arxiv.org/pdf/2611.16477)  

**Abstract**: Laya is a small model that answers a multiple-choice question about a text in one pass: the question, the options (each with a name and a description) and the text form one sequence of at most 512 tokens, the window, in the English release. A longer text is silently cut at the end, and we measure what that costs. Test documents place a held-out row's text, the evidence, at a chosen token offset among distractor text; 177 items, five tasks, three checkpoints (released, short-row fine-tune, long-document fine-tune), four window sizes and 93,275 scored requests. When the evidence lies beyond the window, accuracy falls to 26.4% for the released checkpoint (27.2% for both fine-tunes), near the 27.0% of guessing and below the 35.6% of always giving the commonest label. Confidence gives no warning (AUROC 0.50 to 0.55 for telling cut from read requests; 0.5 = chance), and the runtime's truncation flag also fires on requests whose evidence was visible. The released checkpoint reads visible evidence badly once text precedes it: 38.5% pooled against 60% when the evidence comes first, so a larger window alone gains +6.2 points. Fine-tuning on long documents adds +27.6 points on visible evidence over the short-row fine-tune (95% interval +22.3 to +33.1) and nothing on cut evidence. That is an in-family result: training rows and test documents share generator and question types, and the main filler is of the training kind. With encyclopedic filler never seen in training the gain is +13.2 (+9.9 to +16.6), with rows of other tasks as filler -2.0. Among training-free rearrangements a 312-token window scan helps most (+11.9 points over the default for the released checkpoint) at six to eleven forward passes. Documents are synthetic, labels are LLM-made and one backbone is studied, so these diagnose this model, not long documents in general. 

---
# Robots That Talk Too Much and Don't Listen Enough 

**Authors**: Graham Wilcock, Kristiina Jokinen  

**Link**: [PDF](https://arxiv.org/pdf/2611.16473)  

**Abstract**: Until recently, the biggest obstacles to natural spoken interaction with social robots were speech recognition errors and turn-taking failures. Advances in speech recognition have enabled much better spoken interaction, but turn-taking failures and other misunderstandings continue to be far too prevalent. This short paper looks back at earlier methods for improving speech recognition and turn-taking in interaction with social robots, and then looks at what can be done today, arguing that multimodal bidirectional streaming is needed to reduce the frequency of turn-taking failures. 

---
# DesignTaste: Benchmarking AI-Generator Visual Design Against Award-Wining Works 

**Authors**: Zhou Chen, JiaXiang Li, DanYing Pan, Ye Li, TianRui Yang, EnBao Zhang, YaNan Xiao, LiXin Sun, Anqi Liu, Dong Zhao, JianYuan Li, YuanYuan Ding, Kai Kong, Xinwen Xu, HuiTang Du, Min Zhou, Zhiqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.16442)  

**Abstract**: Text-to-image benchmarks commonly assess text-image alignment or overall preference. Whether models can produce good designs when a brief leaves the appearance open remains less well understood. We introduce DesignTaste, a visual design benchmark that compares generated images with award-winning works. From these works and their publicly available context, we derive 403 design briefs: 100 poster tasks from 100 Beste Plakate 2025 and 303 tasks spanning 20 categories from ADC 2026. Six leading commercial text-to-image models each generate an image from the same brief. A single model judge scores each output and the corresponding award-winning work independently on five evaluation dimensions. On each dataset, only ArtArk achieves a higher mean overall score than the award-winning works. All six models score lower in Creative Originality on both datasets. These results show that producing finished, visually appealing images does not imply that a model can develop good design solutions. 

---
# SoScale: Second-Order Scale Search for FP4 Quantization 

**Authors**: Pengxiang Zhao, Xing Li, Hui-Ling Zhen, Weizhe Lin, Xianzhi Yu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.16426)  

**Abstract**: Low-bit quantization reduces the memory footprint and inference cost of large language models (LLMs), yet accurate FP4 quantization remains challenging. Practical formats such as MXFP4, NVFP4, and HiF4 pair a 4-bit floating-point payload with block-wise or hierarchical scale codes, making accuracy highly sensitive to how each local distribution is scaled. Default maximum-absolute-value scaling preserves dynamic range, but can spend scarce FP4 levels on rarely occupied outlier regions and leave dense tensor values poorly resolved. Existing scale-search methods minimize local element-wise reconstruction error, optimizing tensor fidelity rather than layer-output fidelity. We propose SoScale, a second-order weight scale-code search method for hierarchical FP4 quantization. SoScale searches hardware-valid scale codes under a Hessian-aware layer reconstruction objective and couples this search with GPTQ-style error recovery, selecting scales by their impact on layer outputs. Across Qwen3 models, SoScale consistently reduces downstream quantization degradation under MXFP4, NVFP4, and HiF4. 

---
# RMIT-ADM+S at TREC RAG 2026: Research Agents, Retriever Ablations, and Open-Weight Replication 

**Authors**: Oleg Zendel, Kun Ran, Chenglong Ma, Nguyen Khoi Vo, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16425)  

**Abstract**: We describe the RMIT-ADM+S team's participation in the TREC 2026 Retrieval-Augmented Generation (RAG) track using the ClimbMix corpus. We submitted seven research-agent runs using proprietary OpenAI and open-weight models, exploring different retrieval strategies, writer models, and agent workflows. Pre-submission ablations identified writer selection and adjacent-chunk expansion as the main factors affecting answer quality. In the official NIST evaluation, six of seven runs exceeded the track median across all three rubric/nugget collections. Our best-performing run, brief-hybrid, achieved the highest all-item coverage among our submissions and tied the track maximum (0.892) for Batch-1 vital coverage. Human side-by-side preferences differed across evaluation batches: the open-weight owa-glm5 led Batch 1 despite lower automated coverage, while sol-research-first led Batch 2. These findings highlight differences between automated coverage measures and human preferences, as well as the potential of lower-cost open-weight models for research-agent systems. 

---
# PhysCapBench: A Cross-Domain Benchmark for Physically Grounded Video Captioning 

**Authors**: Haofeng Wang, Pengcheng Wen, Ruiqi Li, Yuhao Zhang, Tiange Zhang, Boren Zheng, Xuanyi Liu, Siyu Tao, Danni Yu, Gan Lin, Haocheng Tang, Tianzizhe Guo, Mingyuan Su, Boyuan Chen, Zhongyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16424)  

**Abstract**: Physical AI aims to learn transferable physical knowledge from real-world experience to understand, predict, and act in the physical world. Internet videos and embodied demonstrations provide complementary observations of motion, interaction, and state evolution. A foundational data challenge is to measure and preserve this physical information as training supervision. In video-language pretraining, captions express observed processes in language, yet a semantically correct description can omit the interactions and transitions that define those processes. Rich physical experience in video does not automatically translate into sufficient language supervision. This gap calls for a yardstick to assess annotation quality, compare captioning strategies, and guide training-data refinement. Motivated by our data construction for world-model pretraining, we introduce PhysCapBench, to our knowledge the first unified physical-caption benchmark spanning all three domains: third-person scenes, egocentric manipulation, and handheld-gripper demonstrations. It provides human-verified annotations for 1,500 video sequences and evaluates accuracy and coverage through seven physical dimensions. We examine downstream utility through video reconstruction and manipulation-policy fine-tuning. Physical captions reduce reconstruction LPIPS by 8.6-9.5% on third-person videos and 26.9-30.9% on egocentric videos relative to metadata captions. Training-instruction enrichment improves both PhysCapBench scores and policy success, including a 7.5-percentage-point gain for pi0.5 on a RoboTwin 2.0 subtask. PhysCapBench makes the physical content of language supervision inspectable and comparable, providing a cross-domain basis for its evaluation and improvement. 

---
# TransXplainRRG: A Radiologist-Evaluated LVLM for Chest Radiograph Report Generation 

**Authors**: Chayan Mondal, Duc-Son Pham, Tele Tan, Ashu Gupta, Vincent T.Y. Tiong, Shreya Ghosh, Tom Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2611.16393)  

**Abstract**: Recent advancements in Large Vision-Language Models (LVLM) enhance X-ray report generation, but often neglect clinical context, overlook workflow integration, and lack explainability, which hinders clinical adoption due to interpretability and reliability concerns. This paper addresses the dual challenges of performance and expert explainability in automated X-ray report generation for clinical AI applications. Our proposed approach, TransXplainRRG, utilises an off-the-shelf Swin Transformer model along with a transformer-based text encoder that incorporates patient medical history to generate a radiology report. Further, we explore an expert-guided `inside-out' approach and extract only abnormal findings for radiology report refinement. Thus, this study bridges the gap between high-performance automation and interpretability critical for clinical practice by combining state-of-the-art transformer-based vision encoders, text encoders, and LLMs. TransXplainRRG is trained and tested on the large-scale MIMIC-CXR dataset and further evaluated on the unseen IU X-Ray dataset to demonstrate its generalizability and robustness. We employed the GREEN metric as a complementary measure of clinical consistency, achieving scores of 0.327 on MIMIC-CXR and 0.605 on IU X-Ray, both higher than those achieved by MAIRA-2 under the same evaluation setting. It also performed competitively across standard evaluation metrics, frequently matching or surpassing state-of-the-art models. Additionally, we introduce qualitative evaluation metrics developed from radiologists' viewpoints and conduct a preliminary radiologist evaluation on 100 generated reports to assess their clinical relevance. Quantitative and qualitative analyses further demonstrate the potential of our TransXplainRRG framework for reliable and explainable radiology report generation. 

---
# Beyond Compressing Everything: Contrastive Memory Learning for Selective Context Compression 

**Authors**: Sikui Zhang, Haowei Liu, Ziyun Gan, Chunfeng Yuan, Guangze Gao, Nian Liu, Boyu Cai, Bo Li, Xiaolei Lv, Jun Gao, Bing Li, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16384)  

**Abstract**: As large language models (LLMs) advance toward increasingly long inputs, soft context compression offers a promising approach to efficient inference by converting long inputs into compact memory slots. Under limited memory budgets, effective context compression remains challenging because task-relevant evidence varies substantially across queries, ranging from localized evidence to multiple distant spans or information dispersed throughout the full document. Instead of compressing everything indiscriminately, we propose CoMem, a query-guided selective context compression framework that learns memory representations conditioned on both the document and the query. CoMem develops Contrastive Compressor Learning to achieve query-sensitive yet context-preserving memory representations. To inject a strong prior over query-dependent evidence distributions, we devise a same-context hard negative mining strategy by pairing a single document with multiple queries, requiring the compressor to selectively retain information according to each query's distinct evidence needs. To address the restricted information flow imposed by causal masking, we further enable bidirectional attention among memory slots. Experiments demonstrate that CoMem consistently outperforms existing context compression methods across compression ratios on question answering and RULER-NIAH benchmarks. Our code is available at this https URL. 

---
# Recursive State-Augmented Language Models: Persistent Latent-State Adaptation of Pretrained Language Models 

**Authors**: Omer Haq  

**Link**: [PDF](https://arxiv.org/pdf/2611.16375)  

**Abstract**: Language models rely on finite token contexts, limiting access to earlier interactions. We introduce Recursive State-Augmented Language Models (Recursive SALM), a framework in which a persistent latent state evolves with each observation and generates adaptations of a pretrained language model. A modular read-transition-write architecture separates observation encoding, latent dynamics, and predictor adaptation. We develop deterministic and stochastic formulations.
The experiments evaluate only deterministic SALM with frozen GPT-2 small and exclude earlier evidence from its decoder context. On a four-turn hypothesis-inference task, latent-gated LoRA reaches 89.1% accuracy, comparable to 88.0% for a trained full-context LoRA baseline, with earlier evidence available solely through a 64-dimensional state. Zeroing or shuffling the state substantially reduces accuracy. In a two-turn task with codewords unseen in training, several interfaces improve candidate ranking over chance, but rarely generate the exact codeword under the evaluated training schedule. These experiments show that persistent latent-conditioned adaptation can support sequential inference when earlier evidence is absent from the decoder context. 

---
# ReliaGuard: Reconstruction-Aware Watermarking for Source Attribution and Blind Edit Localization 

**Authors**: Hailong Wang, Wuzheng Tan, Zixin Zhang, Xujun Fang, Fengjuan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2611.16368)  

**Abstract**: Generative image editing facilitates content creation but complicates source attribution and edit verification. Existing watermarking methods combine copyright protection with tamper localization, yet background reconstruction can weaken spatial evidence even when source identifiers remain recoverable. We propose ReliaGuard, a reconstruction-aware watermarking framework for source attribution and blind edit localization. The writer optimizes keyed feature responses in both the protected image and its variational autoencoder reconstruction under image-distortion and source-preservation constraints. Using only the received image and a secret key, the reader combines watermark responses with local image context through residual evidence refinement and predicts an edit mask through native-resolution-guided multiscale reading. It is trained on complete editing outputs, locally spliced counterparts, and processed versions of both. On an independent test set of 300 source images and 900 editing cases from three image domains, our method achieves edit-region intersection-over-union scores of 85.61% for complete outputs and 94.90% for locally spliced outputs at a threshold calibrated to a nominal 1% background false-positive rate on complete outputs. The complete-output result exceeds the strongest baseline by 45.02 percentage points, while decoded identifier-bit accuracy reaches 100% in both conditions. Across seven families of image processing, the corresponding localization scores are 81.01% and 86.49%, with identifier-bit accuracies of 92.76% and 92.79%, respectively. These results support source recovery and blind edit localization under background reconstruction and subsequent image processing. 

---
# HyBridge: Bridge-Guided Self-Supervised Hyperbolic Hypergraph Representation Learning 

**Authors**: So-Bin Jung, David Yoon Suk, Sang-Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.16351)  

**Abstract**: Hypergraphs capture groupwise relationships among multiple nodes and often exhibit tree-like structures. Although hyperbolic space is well suited for modeling such structures, simply extending existing hypergraph representation learning (HRL) methods to hyperbolic space does not consistently improve performance. In this paper, we investigate the relationship between node structural properties and hyperbolic embedding radii and find that bridge nodes tend to lie closer to the center in graphs, whereas this relationship is not consistently observed in hypergraphs. We further find that conventional bridge score estimation may fail to capture the groupwise connectivity of hypergraphs. Motivated by these findings, we propose HyBridge, a self-supervised hyperbolic HRL framework that (1) estimates node bridge scores by considering the external connectivity of individual hyperedges, (2) learns node and hyperedge embeddings through hyperbolic hypergraph encoding, (3) guides node radial positions according to their bridge scores, and (4) employs contrastive learning to preserve node discriminability. Extensive experiments on 10 real-world hypergraphs demonstrate that HyBridge outperforms 22 competing methods, achieving performance improvements of up to 9.82% in node classification. For reproducibility, we provide the code at this https URL. 

---
# DRAMA: Dissecting Attention Redundancy for Accelerating Multimodal Diffusion Large Language Models 

**Authors**: Yi-Hsin Hung, Fangfu Liu, Shao-Yuan Lo, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.16345)  

**Abstract**: Multimodal diffusion large language models (dLLMs) generate text through iterative masked denoising with bidirectional attention, offering a compelling alternative to autoregressive vision-language modeling. However, each denoising step requires a full forward pass over long multimodal sequences in which visual tokens dominate the sequence length, and a fixed step budget further compounds this quadratic per-step cost. Existing inference acceleration approaches for multimodal dLLMs characterize attention redundancy implicitly, motivating designs from isolated attention visualizations rather than from measurement. We close this gap with DRAMA, a training-free acceleration framework grounded in a systematic dissection of attention redundancy in multimodal dLLMs. By probing attention tensors along spatial and temporal axes, we reveal two persistent redundancy structures: (i) attention mass consistently concentrates on a sparse subset of visual tokens, and (ii) attention distributions rapidly stabilize across denoising steps after an initial formation phase. These findings motivate two complementary mechanisms: Spatial-Redundancy-Aware Visual Token Pruning (SVTP) and Sufficiency-Guided Adaptive Stopping (SGAS). Together, DRAMA achieves an average 18.67 times inference speedup across six vision-language benchmarks without training, architectural modification, or attention caching, while maintaining competitive accuracy. Our results establish a strong, measurement-grounded acceleration baseline for multimodal dLLM inference. 

---
# SUMNO: Spatially Unrolled Multigrid Neural Operator for Fluid Flow Surrogates 

**Authors**: Yingqi Hu, Nan Deng, Changxiao Shao, Xiaozhou He  

**Link**: [PDF](https://arxiv.org/pdf/2611.16341)  

**Abstract**: Data-driven surrogate modeling has emerged as a powerful paradigm for accelerating computational fluid dynamics, and hierarchical U-shaped networks, which resemble the structure of classical Geometric Multigrid (GMG) methods, have become prevalent designs. Modern surrogates built on these designs achieve competitive accuracy at a fraction of the cost of numerical solvers; however, accurate prediction of complex flows remains challenging, and no existing architecture leads across them. In such networks, the multigrid analogy mainly concerns the multi-resolution topology, while their components largely follow common vision practice. To address this, this study introduces SUMNO, a multigrid neural operator for fluid flow fields on regular grids that spatially unrolls a V-cycle into a feed-forward U-shaped network. Parameter-free down- and upsampling stand in for restriction and prolongation, and additive skip connections for coarse-grid correction, with normalization-free transfer and skip pathways. Within this macro-architecture, convolutions are used at the fine and intermediate resolutions and shifted-window attention at the coarsest, whose window partitioning is generalized to handle periodicity independently per axis. Evaluated on unseen test configurations across four unsteady and steady flow benchmarks against five representative baselines, SUMNO is best in 5 of 7 settings and ranks second in the remaining two. It reduces the 10-step rollout MSE by about 30% relative to the strongest baseline, and outperforms three baselines with over 130 million parameters using 9.4 million parameters. 

---
# Wanda-Guided Sparse Perturbations for Efficient Zeroth-Order Optimization 

**Authors**: Ziqing Wen, Xiaohui Song, Jiahuan Wang, Ping Luo, Junlin Zeng, Dongsheng Li, Tao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.16317)  

**Abstract**: Zeroth-order (ZO) optimization enables memory-efficient fine-tuning of large language models through forward-only evaluations, but high-dimensional random perturbations can limit optimization efficiency. Sparse perturbations reduce the search dimension, making the selection of informative coordinates critical. We propose Wanda-Importance Sparsity-MeZO (WIS-MeZO), which uses Wanda weight--activation saliency to construct task-conditioned sparse perturbation masks while keeping the pretrained model dense. We analyze support quality through gradient energy capture and relate it to gradient estimation and convergence. Across 16 benchmarks with LLaMA and Qwen models ranging from 3B to 32B parameters, WIS-MeZO achieves higher aggregate scores than MeZO and reaches comparable validation losses in fewer optimization steps. Comparisons with random and magnitude-based sparse perturbations further highlight the importance of informative support selection. 

---
# Shared Item-Level Error Limits the Aggregation of Small Language Models in Fermi Estimation 

**Authors**: Carolina Fortuna, Blaz Bertalanic  

**Link**: [PDF](https://arxiv.org/pdf/2611.16315)  

**Abstract**: Multi-agent systems of language models are often motivated by an averaging argument: if the agents' errors are unbiased and independent, the error of an aggregate of N answers falls as 1/sqrt(N). Fermi estimation, whose continuous answers are aggregated on the logarithmic scale, permits a direct test. We decompose the log10 error of each answer into the mean error of its model on the item, common to all its draws on the item, and a residual, and estimate both from 2.51M answers given before any communication by 15 open-weight models (3B-72B) on realFP, with replication under a single prompt pair on realFP, Science Olympiad Fermi questions and a guesstimation set. Given the item, teammates' errors are uncorrelated and teams behave as independent draws, but the item-level component accounts for a median 0.906 of the error variance over items. Teams of thirty therefore reduce a single agent's error by a median 1.7% (at most 5.4%), whereas draws that were also unbiased on each item would reduce it by 74-79%. All models compress their estimates toward about 10^6; this trend explains a median 17% of the item-level component, and a cross-validated linear recalibration on labelled items removes the overall bias but not that component. Three rounds of debate reduce within-team dispersion by 93% and leave the item-level component unchanged; revision without peers attains the same small gain. Mixing five models of similar accuracy reduces error by 6.6% relative to their own homogeneous teams, but no mixture improves on the most accurate model. Higher sampling temperature changes team error by about 1% at most, and contrastive activation steering not detectably; per-agent scaffold prompts (not significant after multiplicity correction) and skipping one layer in a single-model screen improve it. The results are conditional on the models, benchmarks, prompts and settings tested. 

---
# ORCHID-SAGE: Fully Reproducible CALPHAD Assessment from Evidence to Phase Diagram 

**Authors**: Raymundo Arróyave  

**Link**: [PDF](https://arxiv.org/pdf/2611.16308)  

**Abstract**: Knowledge of phase stability is the first requirement for materials design, and predicting the conditions under which new materials can be synthesized has become a key capability of self-driving laboratories (SDLs). For inorganic materials, notably alloys, these predictions come from thermodynamic databases built with the CALPHAD method. Building a database, however, requires expert judgment at many steps, and because these decisions are rarely recorded, others can use a database but cannot regenerate or update it. Machines can already collect the data, build a first model with machine-learned interatomic potentials and refine it against experiments. The assessment itself, the sequence of judgments that turns these parts into a reliable database, is still carried out by human experts. We argue that it can be automated reproducibly only if the expert knowledge behind an assessment is encoded in formal protocols, so that an assessment delivers a record of its construction and not only its parameters. We introduce ORCHID-SAGE, a standard that organizes this record into reproducible and automatable protocols. In this first attempt, applied to Cr-Ni, the database was regenerated from its record by a script on a second machine, by an AI agent working only through the Model Context Protocol (MCP) and by a refit from public sources only, with the same predictions. Recorded in this way, a database need no longer be fixed at publication but can be updated by AI agents as SDLs and other sources provide new evidence, while the judgment remains with people and on the record. 

---
# Eliciting and Mitigating Specification Satisficing in Coding Agents 

**Authors**: Ruixin Wang, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16289)  

**Abstract**: Coding agents often fail to meet all requirements in the task description. In some of these failures, the agent considers an implementation choice that meets a requirement, but later decides to commit to another one that does not meet the requirement for a reason it finds acceptable. We call this specification satisficing. It is challenging to identify and mitigate specification satisficing because the agent accepts such choices for its own reasons, and existing self-correction methods are biased toward them. We propose SatisFix, which guides a coding agent to report these choices and their reasons, and then check each choice against the task description. We evaluate SatisFix on DeepSWE and SWE-bench Pro with five LLMs as backbones and two different agent frameworks, and compare it with three self-correction methods. SatisFix improves the pass@1 of the base agent by an average of 5.09% on DeepSWE and 7.06% on the hard subset of SWE-bench Pro. On DeepSWE, it outperforms the best baseline by 3.98%. Our ablation study shows that both steps of SatisFix contribute to the improvement. 

---
# SMIC: Shared Multiscale Input-Reference Conditioning across Transformer Layers for Natural Language Inference 

**Authors**: Xin Zhang, Victor S. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.16270)  

**Abstract**: Natural Language Inference models can be sensitive to localized lexical changes. We introduce Shared Multiscale Input-Reference Conditioning (SMIC), which constructs a single input-dependent multiscale reference and reuses it across Transformer layers. Each layer applies its own multiscale transformation to the shared reference and injects the resulting conditioning feature before the attention update. We instantiate SMIC in a token--context Transformer (TCT) and conduct from-scratch comparisons against decoder-based and Multi-Layer Perceptron (MLP) baselines across all three ANLI rounds. TCT-SMIC attains the highest observed ten-seed mean accuracy on all three ANLI rounds. The ablations further support the contribution of SMIC. We evaluate a real-valued Llama-style plug-in and extend deletion/insertion diagnostics to pretrained ModernBERT encoder and Llama decoder models with and without SMIC. Across the evaluated perturbation protocols, TCT-SMIC exhibits comparatively low pooled-attention redistribution and smaller signed accuracy changes. We further establish a finite-sample bound on average subset-mass redistribution in terms of the mean Jensen--Shannon divergence between the restricted and renormalized attention distributions. 

---
# Rectified Looped Transformer 

**Authors**: Musheng Li, Jiawei Zhang, Ziyuan Liu, Yuantao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16265)  

**Abstract**: Looped Transformers increase effective depth by repeatedly applying a shared block, offering a parameter-efficient approach to language modeling. However, most existing works directly reuse each loop's output as the next's input, which can lead to hidden-state scale growth and perturbation amplification as recurrence deepens. We propose Rectified Looped Transformer (RLT), which redesigns feedback between loops using an interpolation schedule inspired by flow-based modeling. At each loop, RLT forms the next input by linearly interpolating between the current state and its prediction. Such interpolation distributes state changes across loops, encouraging a smoother trajectory from the initial input to the final output. Theoretical analysis proves that, under appropriate assumptions, RLT keeps state magnitudes uniformly bounded across loops and reduces the worst-case perturbation amplification. We evaluate RLT on language modeling corpora and downstream benchmarks. At matched training and inference depths, RLT consistently achieves lower perplexity than a standard looped Transformer, while exhibiting greater robustness to input perturbations and variance in inference depth. Consistent with the theoretical results, latent trajectory analysis shows that RLT maintains more stable hidden-state norms across loops, which also facilitates extrapolation-based decoding strategies. Notably, RLT achieves these gains without requiring additional trainable parameters or extra recurrent block evaluations, thus providing a simple yet more stable feedback mechanism for looped Transformers. 

---
# Label-Efficient Document-Grounded Factuality Detection via Cross-Paraphrastic Invariance Learning 

**Authors**: Chao Chen, Shanshan Lin, Dongsheng Hong, Sibo Ju, Wenjie Yang, Sihong Xie, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2611.16264)  

**Abstract**: Large language models (LLMs) can generate fluent claims that are unsupported by a given source document, motivating accurate and efficient document-grounded factuality detection. Existing LLM-based evaluators often require repeated inference and incur high computational cost, whereas compact factuality detectors are efficient at inference time but commonly rely on large labeled or synthetically generated corpora. We propose CPIL (Cross-Paraphrastic Invariance Learning), a two-stage Siamese framework that converts a small labeled set into relational supervision without acquiring additional factuality labels. CPIL constructs positive pairs from distinct label-preserving paraphrastic views of the same document-claim instance and negative pairs from opposite-label examples associated with the same source document. It further introduces quantile-aware semi-hard negative mining, which derives a boundary from the positive-pair similarity distribution and balances informative negatives with broader coverage. In the two-stage training procedure, Stage 1 learns an encoder from these task-aware pairs, and Stage 2 jointly fine-tunes the encoder and a linear classification head for binary factuality prediction. At inference time, CPIL requires no paraphrase generation, pair construction, or Siamese comparison and uses only one encoder-classifier forward pass. On the 11-task LLM-AggreFact benchmark, CPIL-2% and CPIL-3% achieve macro-F1 scores of 81.04 and 81.60 while using only 2-3% of the FactCG-released labeled pool. Both settings outperform all compared compact detectors and two vanilla LLM references. Ablations and sensitivity analyses support the contributions of task-aware pair construction, dynamic negative mining, and multi-pivot paraphrasing. 

---
# Indic-Task Gains and Regressions after Reinforcement Learning 

**Authors**: Sutanshu Saksena Raj, Daksh Maan, Avinash Benki, Parijat Chatterjee, Pasunuti Prasanjith, Purusharth Malik  

**Link**: [PDF](https://arxiv.org/pdf/2611.16228)  

**Abstract**: We run reinforcement learning (RL) with GRPO across four environments, starting from a supervised fine-tuned 30B-A3B mixture-of-experts model covering English and eight Indic languages. We compare the model before and after RL across 34 benchmarks in one evaluation campaign. MATH improves by 36.07 points, and most Indic benchmarks also improve. Yet XQuAD-Indic F1 falls 19.32 points as English answers rise from 8.7% to 91.7%; IndicQA falls 8.64, while English XQuAD gains 11.71 F1. A second RL run from the same supervised initialization adds an Indic environment together with language and repetition rewards. Compared with the released model, it improves both passage-answering and instruction-following benchmarks, along with preserving MATH performance. But it loses ground on reasoning and coding benchmarks. Several training choices changed together, so we cannot isolate the effect of any one of the reward terms. Script compliance remains unresolved. Adding a language reward did not ensure use of the requested script. Passage answering and script compliance therefore need direct evaluation, even when other benchmark scores improve. 

---
# Guided Generation of Linux Kernel Semantic Patches with Small Local Language Models 

**Authors**: Mohanned Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2611.16215)  

**Abstract**: When a Linux kernel API changes, hundreds of drivers are converted by hand over months, and the earlier conversions are reviewed examples for the later ones. We ask whether small open language models running on a laptop can turn such examples into Coccinelle semantic patches, the rules kernel maintainers already use for tree-wide changes. We adapt guided code generation to this task: the training examples are decomposed by script into a tree of change variants, the model writes one small rule per variant under feedback from applying it to the examples, deterministic components repair SmPL syntax and generalise driver-specific names, rules are composed greedily, and a compile critic repairs or rejects a rule whose output does not build. We evaluate on a pre-registered test set of 7 migrations whose new APIs appeared after the training cutoffs of all three models (Qwen2.5-Coder 7B, Llama 3.1 8B, Gemma 3 4B), with 122 held-out driver conversions, counting a conversion as solved only if the rule's output matches the human fix and compiles. With the framework, Qwen2.5-Coder solves 35.3 conversions on average over three seeds (range 32-41), against 1.0 when prompted directly; Spinfer solves 41 and a frontier LLM agent 66. The local model solves fewer conversions than Spinfer but its rules are the most precise on the test set (0.74 of the files they change are solved, against 0.55 for Spinfer and 0.70 for the frontier agent), and the compile critic removes every output that matches the human fix but does not build. The two general-purpose models stay weak even with guidance (at most 12 solved). All code, prompts, model responses and results are released. 

---
# When can an interpretability claim be trusted? Stability and certification of feature dimensionality, ridge leverage and the feature recovery rate 

**Authors**: Ali Tajfar, Behnam Tajfar  

**Link**: [PDF](https://arxiv.org/pdf/2611.16190)  

**Abstract**: Interpretability tools increasingly support safety claims, such as that a sparse dictionary has recovered most of a model's features. Such claims rest on measurements with arbitrary thresholds, and small changes to the model weights, a different training seed or a different threshold can reverse them. We study two widely used measurements in toy models and synthetic dictionaries with known ground truth: the feature dimensionality of Elhage et al. and the feature recovery rate (FRR) with a cosine threshold. We characterize exactly where feature dimensionality is discontinuous and replace it with a regularized (ridge) version with sharp stability bounds: the ridge leverage score changes by at most $3\sqrt{3}/(8\sqrt{\lambda})$ times the operator-norm change of the weights, and this constant cannot be improved; for the regularized dimensionality the stability constant grows like $\rho/4$ in the scale $\rho=R/\sqrt{\lambda}$, with explicit upper and lower bounds. We also prove certificates stating when a claim "feature $i$ is recovered" must survive a given change of the dictionary. The certificates held in every test and are informative for small perturbations, but certify no feature non-trivially across independently trained dictionaries; at the standard threshold, whether a feature counts as recovered changed with the training seed for 55 percent of features. Experiments use toy models only. The proofs, code and text were drafted by AI agents directed by the authors and checked by an independent blind AI reviewer and by numerical tests; the authors take full responsibility for the content. 

---
# Data-Efficient LLM Safety Alignment Guided by Interpretable Features 

**Authors**: Hongyu Li, Zhan Yu, Zhe Zhao, Kimmo Yan, Xiong Kuang, Renfen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16183)  

**Abstract**: Existing large language models typically rely on massive fine-tuning datasets for safety alignment, which results in higher costs for safety alignment and often leads to the problem of over-refusal. We propose Less is Safer (LIS), a compact yet highly effective dataset for safety alignment, constructed through interpretable feature-guided selection and adversarial synthesis. Specifically, we leverage Sparse Autoencoder (SAE) activations to efficiently select prompts that cover diverse and interpretable safety features. Then, we employ a multi-agent adversarial synthesis framework to generate harmless prompts that nevertheless elicit unnecessary refusals. LIS contains only 2.4K samples, yet outperforms much larger datasets, e.g., TRIDENT-Core (26K), TRIDENT-Edge (19K) and WildJailbreak (262K), in overall performance on four safety and five over-refusal benchmarks. Through ablation experiments, we demonstrate that neither safety data nor over-refusal data alone can achieve optimal results. We also used safety features from SAE, annotated by an LLM, for data filtering, but the results were inferior to those obtained using the pipeline proposed in this paper. 

---
# Same-State Pairs Make One JEPA Proof of the Platonic Representation Hypothesis Vacuous 

**Authors**: Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.16182)  

**Abstract**: In a Joint-Embedding Predictive Architecture (JEPA), the JEPA proof of the Platonic Representation Hypothesis (PRH) assumes that every observation determines the world state and that a spectral gap makes minimizers unique up to rotation. Under the hard-whitening population objective, same-state positive pairs make the induced world kernel the identity: every centered whitened state function is a global minimizer, and, when $d_z-1\ge 2k$, two minimizers can have best-rotation squared residual $2k$. The uniqueness claim is conditional; for $k<d_z-1$, where $d_z$ is the number of world states, this limit makes its premise unsatisfiable, so the claim has no content there. The experiments that motivate it leave the assumption: a caption sees only part of an image, and the paper states no experiment meets its hypotheses exactly. The theorem does not reach those experiments, and for $k<d_z-1$ its idealized same-state limit is vacuous. One of the paper's four quantitative results survives: a gap-free lower bound. Its bound vanishes only at perfect scene-explained fraction; it bounds alignment error but cannot certify good-view alignment. A canonical frame is determined only up to a rotation and a latent permutation. Paired samples fix the relative frame at effective rank $k$; a global rotation remains. Component semantics fix the relative permutation only up to signs; value anchors fix it once the mixing frame is pinned. A resource--gauge ledger records which resource fixes which gauge. 

---
# JudgeMix: Judge-Guided Mixed-Rollout Data Synthesis for LLM Agents 

**Authors**: Cheng-Yu Hsieh, Ting-Yao Hu, Sanjoy Chowdhury, Oncel Tuzel, Raviteja Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2611.16174)  

**Abstract**: Supervised fine-tuning (SFT) on expert trajectories is a standard paradigm for training LLM agents. However, it suffers from a well-known training-inference distribution mismatch, calling for target-aware data synthesis strategies that ground the training data in the target model's native state distribution. To bridge this gap, emerging approaches incorporate the expert either to guide mixed rollouts alongside the target model, or to provide dense step-wise supervision. Nonetheless, existing methods apply expert assistance blind to the target model's inherent behavior: they either disrupt the rollout using sub-optimal random intervention heuristics, or they enforce indiscriminate uniform supervision that unnecessarily overwrites the model's already-correct actions. We introduce JudgeMix, a judge-guided data synthesis framework that dynamically interleaves the target model and expert based on step-wise semantic evaluations. JudgeMix queries expert guidance precisely when the target model struggles, ensuring high trajectory success rates while keeping the state distribution aligned with the target model. It additionally uses the judge evaluation to reweight expert supervision on different steps, preserving model's inherent valid behavior while correcting its errors. On two agentic tool-use benchmarks, AppWorld and AutomationBench, JudgeMix demonstrates consistent superior performance to existing methods, with improvements up to 10 pass@1 points. 

---
# Self-Supervised Attention-Based Virtual Metrology for CMP MRR Prediction 

**Authors**: Jonathan Adam Rico, Nagarajan Raghavan, Senthilnath Jayavelu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16172)  

**Abstract**: Chemical mechanical planarization (CMP) is a critical semiconductor manufacturing process where material removal rate (MRR) is a key metric for process monitoring and advanced process control (APC). Virtual metrology enables MRR estimation from process time series signals, supporting run-to-run(R2R) control and process optimization while reducing reliance on physical metrology. Existing approaches typically rely on simplified linear assumptions or require handcrafted feature engineering. In this paper, we propose Global Regression Pretraining(GREP), a self-supervised, attention-based framework for CMP MRR prediction. GREP learns temporal representations from unlabeled sensor signals through global proxy-target regression, followed by supervised fine-tuning for MRR estimation. The framework is evaluated against a feature-engineered supervised baseline under three conditions: limited labeled data, sensor noise, and temporal process drift. Experiments on the PHM 2016 CMP dataset demonstrate competitive performance under limited labeled-data conditions, with improvements in selected cases and performance largely comparable to the baseline across different labeled-data fractions. Both models exhibit relatively small changes in prediction error with increasing noise, while GREP achieves a 17.0% RMSE reduction in the final temporal interval. 

---
# SwarmSafe-Delta: Isolating Message Effects on LLM Agent Safety 

**Authors**: Junru Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16123)  

**Abstract**: When a tool-using LLM agent team violates policy, aggregate outcomes do not reveal whether an upstream message changed the executor's action. We introduce SwarmSafe-Delta, a paired executable protocol that freezes upstream computation, varies real, hidden, and format-matched placebo delivery, and scores a single committed action against deterministic benign and risky policy states under preassigned budgets.
Across four models and constructed optional tasks, real delivery at $N=4$ produces 21.4% risky-world violations, versus 5.0% with placebo and 1.0% with hidden delivery (paired differences +16.4 and +20.3 percentage points). Real delivery also improves benign task success, illustrating that suppressing communication sacrifices utility. In a second regime constructed to require an upstream lookup route, real delivery improves safe completion by 74.2 points relative to hidden delivery. These effects quantify message-delivery contrasts under specified information dependencies, not naturally occurring failure rates.
Exploratory action-level analyses show that the effect depends strongly on whether the upstream message recommends a prerequisite check or direct execution; model differences also largely track these message compositions. The contribution is a controlled, execution-grounded measurement protocol rather than a general claim that agent communication is harmful or protective. 

---
# ADAPT: Agentic Diff-Aware Patch Testing for Pull Requests 

**Authors**: Vahid Haratian, Atakan Akar, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2611.16114)  

**Abstract**: Modern software development relies on pull requests (PRs) that introduce small, targeted code changes. Once merged, these changes should be protected by focused tests that exercise the behavior the PR introduces, yet developers often submit PRs without such tests, leaving the new behavior unguarded against future regressions. Most test generation techniques target entire software units and optimize objectives such as code coverage rather than the behavior a change introduces, and prior diff-aware approaches have had limited success in capturing it.
We propose ADAPT (Agentic Diff-Aware Patch Testing), a regression test generation approach that uses an autonomous coding agent to produce change-focused tests for PRs. Given a PR, the agent analyzes the diff between the base and head versions to identify the modified functions and their context, then plans and generates candidate unit tests. A dedicated fail-to-pass (F2P) verification tool runs each candidate on both versions and, after F2P success, reports which executable changed lines remain uncovered; guided by this feedback, the agent iteratively refines its tests.
We evaluate ADAPT on the same 426 real-world PRs used by the prior state-of-the-art baseline, with GPT-4o and GPT-5-mini. ADAPT+GPT-4o reaches an F2P rate of 62.0% (264/426), compared with BLAST's reported 35.4% (151/426), while ADAPT+GPT-5-mini reaches 89.9% (383/426). On 137 PRs shared with Change And Cover (ChaCo), ADAPT achieves full patch coverage on 83.9% of PRs, compared with 79.6% for ChaCo, at considerably lower time and cost. A manual evaluation confirms that the F2P-valid tests target each PR's intended change, with mean scores of up to 4.54/5 for importance, 4.35/5 for assertion strength, and 4.49/5 for readability, often covering behavior beyond the developer-written tests. 

---
# Trained on What Screen Readers Cannot Read: Accessibility Debt in the GUI Corpora Behind Computer-Use Agents 

**Authors**: Obada Kraishan, Kulsawasd Jitkajornwanich  

**Link**: [PDF](https://arxiv.org/pdf/2611.16107)  

**Abstract**: Computer-use agents learn to operate mobile interfaces from corpora of captured screens and accessibility trees. The same trees are what a screen reader consumes, so an agent trained on them inherits whatever labelling gaps the underlying applications carry. This paper measures one of those gaps, content-description coverage, in two Android corpora: Rico, collected in 2017, and AndroidControl, collected in 2024. Across 17,373,614 interface elements, 17.74% of the Rico elements that need a content description under our coding rule carry one, against 57.27% in AndroidControl. The difference of 39.5 percentage points holds under five alternative coding rules and when whole apps are resampled. It describes two datasets that also differ in app selection, capture method and platform version; it does not measure change over time. Input controls are the least labelled role in both corpora (2.20% and 27.21%, or 30.99% in AndroidControl if hint text counts as a name). Descriptions in the older corpus are also five times more likely to be placeholders (10.25% against 1.95%), so the difference is not an artefact of developers typing filler. Resolving 26,205 task actions to the elements they touched shows that, among tappable graphics with no text, demonstrations operate elements described 9.0 points more often than the others on the screens they visit, about 3.5 points of it from the choice of element within a screen. That is a selection effect inside the benchmark. A two-coder reliability check (kappa = 0.91 and 0.96) and the full pipeline accompany the paper. 

---
# Show Your Work in Swahili: Reinforcement Learning for Native-Language Mathematical Reasoning 

**Authors**: Jessica Lundin, Katherine Rosenfeld, David Adelani, Geoffrey Mosoti Nyakiongora, Halfan Ngowo, Sloan Coffin, Ralph Abboud, Guillaume Chabot-Couture  

**Link**: [PDF](https://arxiv.org/pdf/2611.16104)  

**Abstract**: We present Fikiria, a two-stage SFT+GRPO recipe for Swahili mathematical reasoning. Its reward multiplies a format gate by a weighted sum of answer accuracy and the ChrF\texttt{++} similarity of the Swahili reasoning to an in-language reference chain of thought, so that training targets the surface fidelity of the reasoning trace and not only the final answer. We train on a 2.3k-problem Swahili dataset created by translating GSM8K with GPT-4.1 and filtering with SSA-COMET. On AfriMGSM Swahili, Fikiria-AfriqueQwen-14B reaches 79.2\%. Under the same fixed prompt and strict answer-format scoring, GPT-4o reaches 68.0\%, partly because it often fails the required answer format; IrokoBench reports 78.8\% for GPT-4o using a best-per-model prompt, a figure numerically close to ours but not directly comparable. In a small human evaluation with co-author annotators, GPT-4o's outputs were preferred overall. ChrF\texttt{++} measures surface fidelity, not the validity of individual reasoning steps. 

---
# Subsurface schematisation from sparse CPT data using the tabular foundation model TabPFN 

**Authors**: Taiga Saito, Fabian Campos Montero, Bruno Zuada Coelho, Ana Martins Teixeira, Stephen Wu, Timo Schweckendiek, Yu Otake  

**Link**: [PDF](https://arxiv.org/pdf/2611.16088)  

**Abstract**: Subsurface schematisation supports geotechnical decisions but remains challenging because sparse site-investigation data leave substantial gaps in information about ground conditions. This study explores TabPFN, a foundation model pretrained on general synthetic tabular data, for this task. Using only grid coordinates and soil behaviour type index observations from sparse cone penetration tests (CPTs), we first reconstruct two-dimensional sections without geotechnical fine-tuning or predefined layer boundaries. We evaluate the approach on synthetic sections and at two field sites in the Netherlands, comparing it with SchemaGAN, a conditional generative adversarial network trained on synthetic geotechnical sections, and with reference interpolation methods. In the baseline comparisons, TabPFN achieved mean errors close to SchemaGAN's on synthetic sections and lower mean errors at both field sites. On synthetic sections, TabPFN's mean absolute error decreased at each tested increase in CPT count, and, under the baseline input setting, its pointwise prediction intervals had mean coverage close to nominal levels. Subsequent fine-tuning on the synthetic sections used to train SchemaGAN improved synthetic test accuracy but increased mean errors at both field sites. These findings indicate that general-purpose tabular foundation models offer a promising route for updating subsurface predictions from sparse and evolving site-investigation data without retraining. 

---
# Agentic Computational Algebra 

**Authors**: Ryan A. Anderson, Matthew Trager, Ben Bowman, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2611.16073)  

**Abstract**: We explore the use of AI agents for solving computational problems in algebra. For the problems we consider, the main challenge is not correctness but \emph{time}: general-purpose algorithms are guaranteed to provide correct solutions but can be prohibitively slow, while reformulations that exploit special structure often allow solving a problem much faster. We propose to use this setting to evaluate how effectively agents discover and exploit structure within and across tasks to reduce computational costs. More broadly, computational algebra supplies infinitely many problems with independently verifiable answers and systematically controllable structure and difficulty. To illustrate these possibilities, we use benchmark families of systems with controlled shared structure and evaluate agents in three settings: solving independent instances, solving online streams of related instances using persistent memory, and compiling a fast reusable solver for a known problem family. Our experiments show that (i) agents can extend the range of systems solvable within a fixed budget, although reasoning overhead can sometimes outweigh computational savings; (ii) memory can reduce the cost of repeated solving but may preserve suboptimal strategies; (iii) compilation produces faster reusable solvers but at substantial upfront cost. Overall, computational algebra offers concrete, measurable targets for improving how agents discover and reuse structure. 

---
# Tokenized Reward Systems for Federated Learning: The Privacy vs. Fairness Tradeoff 

**Authors**: Andreas Sendros, Nikolaos Pavlidis, Vasilios Perifanis, Pavlos S. Efraimidis  

**Link**: [PDF](https://arxiv.org/pdf/2611.16065)  

**Abstract**: Federated learning (FL) systems increasingly rely on blockchain based reward mechanisms to incentivize participation while preserving data privacy. We design and empirically evaluate two complementary Solidity smart contracts for FL reward distribution: the first one that allocates rewards proportionally to a publicly disclosed quality score, and a privacy preserving variant that hides individual quality scores behind a zero-knowledge (ZK) proof of a threshold condition. Both contracts mint a non transferable, use-and-burn token maintaining a trustless incentive structure. We evaluate the two designs through MNIST training with adversarial behavior, including Sybil attacks, zero gradient and random noise, across three independent random seeds.
Our results reveal a fundamental tradeoff between cryptographic privacy and incentive fairness. The privacy preserving architecture strictly degrades system robustness, doubling the adversary's average reward capture while introducing severe outcome variance. This vulnerability is driven by a structural self-inclusion bias within cosine-similarity scoring: as the global model converges, legitimate gradient updates naturally diminish, rendering them statistically indistinguishable from bounded noise. Adversaries exploit this dynamic to systematically breach the ZK threshold, triggering a late stage failure cascade that culminates in absolute adversary monopoly over the reward pool and the total exclusion of honest contributors. Alongside a closed-form theorem formalizing this privacy--Sybil resistance tradeoff, we map strict mainnet deployment boundaries via linear gas scaling models, explicitly incorporating the post-Fusaka EIP-7825 per transaction gas cap. Ultimately, while cryptographically secure, static threshold ZK gates fail as incentive layer fairness mechanisms under realistic FL convergence dynamics on standard benchmarks. 

---
# Human Capital in Context: What Employment Survival Reveals About Worker-Firm Match 

**Authors**: Jung Ho Choi, Mario Milone  

**Link**: [PDF](https://arxiv.org/pdf/2611.16058)  

**Abstract**: There is no best worker in general, only a best worker for a particular firm. We develop a measure of worker-firm match quality that treats fit as a property of the pair, learned from the survival of employment relationships. A two-tower neural network embeds workers, from their career histories and current positions, and firms, from their characteristics and business descriptions, into a common space, and match quality is the cosine similarity of the two vectors. The measure takes its functional form from the skill-weights model of firm-specific human capital, and its only estimated objective, a separation hazard, from the view of the job match as an experience good. Estimated on more than a billion person-periods covering 101 million workers and 8.4 million firms, a 0.1 increase in match quality lowers the odds of separation by 48%. About a quarter of the variance of match quality is specific to the worker-firm pair, which worker-, firm-, and occupation-level measures set to zero by construction. Workers earn more when better matched, especially early in a relationship, and firms whose workforces score higher grow faster, become more profitable, and are issued more, and more valuable, patents over the next five years than their industry peers. Because it scores pairs that never formed, the measure also predicts where movers go, and acquirers record a larger share of the price as goodwill when the target's workers would fit the acquirer better than they fit the target. 

---
# Pyramidal Columns Acting as Attention Heads 

**Authors**: Fabrizia Giulia Garavaglia, Marco Giunti  

**Link**: [PDF](https://arxiv.org/pdf/2611.16057)  

**Abstract**: Can the functional architecture of a Transformer attention head be given a biological construction? More specifically, could a suitable group of pyramidal columns in posterior temporal language regions (PTLRs) constitute the biological construction of a neuromorphic mechanical realization of an attention head, under a mathematical instantiation that need not coincide with its standard Transformer instantiation? This paper investigates this hypothesis by examining possible correspondences between the functional organization of attention heads and neurophysiological mechanisms operating within and across pyramidal columns involved in human language processing. Our analysis builds on a mechanical-structural description of Transformer attention heads as systems of simple neural networks linked through output-input and output-weight interconnections. We examine the functional organization of an attention head and the input-dependent transformations it produces, and ask how the corresponding processing relations could be constructed in biological neural tissue. We then identify candidate neural mechanisms of signal transmission and modulation within and across pyramidal columns that may jointly support such a construction. On this basis, we propose that a suitably organized group of pyramidal columns in PTLRs could constitute a biological construction of a neuromorphic mechanical realization of an attention head. Whether the required functional organization actually exists in PTLRs remains an open empirical question. The hypothesis does not require the biological system to reproduce the standard mathematical instantiation of Transformer attention: the same functional architecture may be mechanically realized and mathematically instantiated in a different way, consistent with its biological construction. 

---
# Scaling public input on AI policy with pluralistic agents 

**Authors**: Jacy Reese Anthis, Ken Burke, David Madras, Crystal Qian, Meredith Ringel Morris, Logan Cross, Caroline Pantofaru, Utsav Prabhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16028)  

**Abstract**: AI is being deployed across a vast range of application domains, cultural environments, and human stakeholders. Safe and responsible AI development requires continuously accounting for diverse perspectives. We develop and test pluralistic agents, AI agent representatives that deliberate on behalf of individuals to align AI systems with diverse human perspectives. Pluralistic agents could help people contribute to many more discussions than they could directly participate in, on technical topics that would require training, and on policy choices that rely on private information that corporations and governments cannot readily share. Our experiments identify opportunities, risks, and perceptions of pluralistic agents. Agents were able to reliably shift policies in the direction of their represented persona's preferences, but the extent of each agent's influence was sensitive to the instructions from its persona, such as having more influence when instructed to be more verbose or less cooperative. This suggests that influence in AI-assisted deliberation could vary based on how participants prompt their agents and how the system is designed. In a preregistered study, 746 humans participated in the pluralistic agents process. Participants viewed pluralistic agents as useful and fair, preferring agents over the baseline of providing input to a single AI reasoning model. Most participants preferred participating themselves over either pluralistic agents or the baseline, but pluralistic agents recovered 68% of the baseline-direct gap in reported satisfaction and 82% in fairness. Because pluralistic agents can be used in many real-world contexts where direct discussion is not, they could be a promising tool to scale AI safety policy development. We encourage cautious exploration of this method, playing to its strengths as an amplifier of human input rather than a replacement. 

---
# Runtime Analysis of Cartesian Genetic Programming on Max: A Proven Exponential Speedup 

**Authors**: Duc-Cuong Dang, Roman Kalkreuth, Andre Opris  

**Link**: [PDF](https://arxiv.org/pdf/2611.16013)  

**Abstract**: Genetic Programming (GP) is a search paradigm inspired from natural evolution for the automated discovery of expressions, functions, and computer programs. Cartesian GP (CGP) is a flavor of GP that uses a graph-based model to encode candidate programs, as contrasted to the conventional Tree-based GP (TGP). Since its inception two decades ago, CGP has predominantly been analyzed empirically, thus little is known regarding its theoretical performance guarantees. This paper analyzes CGP for the Max problem, which has been only rigorously studied for TGP with a proven expected runtime superlinear in the optimal program length $n$. We prove that the expected time for the ($1$+$1$)~CGP employing common mutation operators to evolve an equivalent optimal of the same output is polylogarithmic in $n$, specifically at most $O(\log^9{n})$. This is remarkable, as it shows an exponential speedup by switching to CGP. Our results shed light on the benefit and the compactness of the graph-based representation for programs, and on how CGP can navigate its search and fitness spaces efficiently. This is a first proven performance guarantee for CGP on an established benchmark problem. Experiments complement our theoretical findings. 

---
# Iris-Z: Scaling One Generative Model Past Specialist Geometry Models 

**Authors**: Alejandro Cañada, Chema Garabito  

**Link**: [PDF](https://arxiv.org/pdf/2611.16010)  

**Abstract**: Language moved from many task-specific models to one unified model: a single generative model with a single generative training task that handles all of them within the same backbone, and keeps improving with data, parameters and compute. We believe the bitter lesson applies just as much to computer vision, 3D and spatial understanding, in the end to visual intelligence: the effort should go into general architectures that solve all these tasks in a unified way, and into scaling them, rather than into task-specific designs. We test this on monocular geometry, a task that has belonged to specialised architectures for years. Iris-Z is a general image generator that produces appearance and geometry together. A single backbone denoises, reasons about and understands the image and its depth at the same time, and one set of weights generates images with their depth, estimates depth from photographs and paints images onto a given depth. At enough scale it surpasses state-of-the-art specialists on established depth benchmarks while keeping the generative abilities of the backbone it started from without specialized layers. We release two models for the community. 

---
# JustRUA: A Controlled Study of GPT-6 Robot-Use Agents 

**Authors**: Zefan Wang, Yufan Dang, Xuanwei Shi, Mingqi Gao, Cheng Yin, SikYuen Tam, Hanyu Liu, Yasheng Wang, Yaxi Lu, Songyang Zhang, Xin Cong, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.16009)  

**Abstract**: GPT-6 Astra can drive a robot arm directly through a fixed harness, and scores of such robot-use agents on RoboDojo have risen quickly since. Reported systems differ in harness and test subset and often change several designs at once, so it is hard to tell which choices produce the gains. We take these designs apart and test them in one framework, along the harness around the model, the action interface, the reuse of experience and the scene geometry given to the model. A lean harness that keeps the full episode history and stops each motion early goes a long way. Reusing experience changes how the model acts but is not a free lunch, and scene geometry helps most, even when depth is predicted from RGB alone. JustRUA starts from the official L3 harness and keeps only what matters most. Even with no learned policy, self-improvement loop or task-specific tuning, this simple harness raises the score on the RoboDojo simulation benchmark substantially, from 29.7 to 50.3. 

---
# Fit-LRP: Fitting Linear Relaxations to Polytopes for Neural Network Verification over Non-Convex Regions 

**Authors**: Brian Yarbrough, Jesse Wayment, Jingbo Wang, Philip E. Paré, Shreyas Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2611.16003)  

**Abstract**: State-of-the-art neural network verifiers evaluate linear relaxations over axis-aligned bounding boxes. However, verification domains in cyber-physical systems are rarely boxes. Reachability specifications, obstacle-avoidance constraints, and neural Lyapunov certificates typically yield oblique or non-convex polyhedral regions, over which box-based bounds become extremely loose. This paper proposes Fit-LRP (Fitting Linear Relaxations to Polytopes), a framework the implements exact optimization of affine bounds over non-convex domains (represented by unions of polytopes) by solving a sequence of small, low-dimensional linear programs. Our approach seamlessly enhances the final optimization step of LiRPA and $\alpha,\! \beta$-CROWN branch-and-bound (BaB) algorithms. In BaB verification, approximating oblique constraints with axis-aligned splits causes severe combinatorial blowup and risks non-termination. By optimizing the final affine bound of a nonlinear function over the exact polyhedral state-space constraints our method eliminates box-induced over-approximations, enabling earlier search-tree pruning. On ARCH-COMP and neural Lyapunov benchmarks, Fit-LRP reduces branching on the hardest region by $47\times$, cuts worst-case verification time by $15\times$, and with region-wise partitioning, verifies on a single CPU $6.1\times$ faster than $\alpha,\! \beta$-CROWN on a GPU. 

---
# What Roles Do System Prompts Really Assign? Validating and Steering Occupational Role Vectors in LLMs 

**Authors**: Jose L. Sampedro Mazon, Ilmi Yoon, Nadim Saad  

**Link**: [PDF](https://arxiv.org/pdf/2611.15989)  

**Abstract**: Applications built on large language models assign the model a role through its system prompt, intending to shape how it responds for their use case. We introduce a validation framework for extracting and injecting occupational role vectors, activation-level directions that capture how someone in a given occupational role would distinctively respond to a task or question. We validate these vectors through two complementary lenses, on Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. The interpretability lens asks whether each vector is a stable, separable, domain-consistent representation of its role: extracted vectors are highly reproducible across repeated runs and sampling temperatures, remain separable even for very similar roles, and agree more with themselves across in-domain and out-of-domain questions than with other roles' signal. The control lens asks whether injecting a role vector during generation makes the model express the role, and sustain it over a conversation, as well as a role prompt does. On Qwen, injecting the vector alone makes the model express the intended role substantially more often than unsteered, but at best matches a role prompt; on both models, adding the vector to the prompt strengthens the role in single answers but did not slow the role's decline over 30 turns. Together, the two lenses show why both are needed: out of domain the role is expressed only subtly, yet the activations still encode it; and a vector that is stable and separable can still barely steer the model's behavior. 

---
# DuoJEPA: Cross-Modal Translation Between Unaligned Self-Supervised Embedding Spaces 

**Authors**: Mahmut S. Gokmen, Mitchell A. Klusty, Evan W. Damron, Caroline N. Leach, Samuel E. Armstrong, V.K. Cody Bumgardner  

**Link**: [PDF](https://arxiv.org/pdf/2611.15968)  

**Abstract**: Learning from paired modalities is commonly equated with aligning them: two encoders are trained until the embeddings of a pair coincide in a shared space. We study whether cross-modal retrieval and classification can instead be obtained while the two encoders' own embedding spaces remain directly unaligned. DuoJEPA trains one encoder per modality with the self-supervised LeJEPA objective and appends to each encoder a single transformer layer that translates its embedding into the other encoder's space, supervised by a contrastive objective computed entirely within that target space. No loss term compares the two own spaces directly, although the encoders are optimised jointly through the translation objective. In our experiments, the translator's residual connections determine where the cross-modal pressure is absorbed: with a residual translator the two encoders become directly aligned, whereas with a skipless (residual-free) translator a paired-cosine test detects no alignment in the final models, and at most a weak one in the ablations, at comparable cross-modal performance. The learned class-token translation is well approximated by a single scaled rotation of the source space. We evaluate DuoJEPA on two modality pairs that differ in nearly every respect. On chest radiographs and reports from CheXpert Plus, it attains a zero-shot AUC of 86.6 and a cross-modal retrieval score of 78.5, compared with 60.0 and 64.2 for our reimplementation of LeVLJEPA with the same encoders and data. On Sentinel-2 optical and Sentinel-1 radar imagery, the frozen encoders fall within the range of published Sentinel foundation models under our reproduction of the Copernicus-Bench linear-probe protocol, the translator retrieves the paired image of the other sensor among 4,000 candidates in 77-79% of cases, and a patch-level extension transfers the within-image similarity structure of the scene. 

---
# Spec-Driven PBT: Deriving Semantic Invariants from Code Specifications for Agentic Property-based Testing 

**Authors**: Aditya Shukla, Kartik Gupta, Ali Alavi, Pashootan Vaezipoor  

**Link**: [PDF](https://arxiv.org/pdf/2611.15920)  

**Abstract**: Software is increasingly written by AI agents that produce an implementation from a natural language specification, a practice known as agentic coding. Code is now produced faster than it can be reviewed, and has been shown to exhibit its own characteristic failure patterns. Property-based testing (PBT), a long-standing technique in which invariant properties of a program are checked against many generated inputs, has recently been revisited with LLM agents authoring the properties, thereby removing the high-effort derivation of invariants that has long hindered its adoption. Existing methods, however, derive these properties from the implementation alone (Code-driven PBT). At the same time, a thorough specification such as a product requirements document has become routine within agentic coding, yet its role in testing the resulting code remains underexplored. We study the value of the specification as a source of properties for testing agent-written code. Our key insight is that the specification, unlike the implementation, records what the code was meant to do, and invariants derived from it are not biased by particular implementation choices. We propose Spec-Driven PBT, an agent that first expresses invariants in natural language from the specification and only then inspects the implementation to translate them into concrete tests. Across 25 applications from ViBench, Spec-Driven PBT finds a novel set of bugs that Code-driven PBT misses, while writing 2.4x fewer property tests. Though the two approaches surface a roughly equivalent number of bugs, there is little overlap: nearly 60% of the 169 total bugs are found by only one. Qualitatively, we find that the bugs unique to Spec-Driven PBT tend to violate documented product workflows whereas those to Code-driven PBT instead concentrate on implementation robustness, offering complementary value for bug discovery in agent-written code. 

---
# HeriKGQA: Grounded Knowledge-Graph Question Answering over Cultural Heritage Sensor Data 

**Authors**: Lingxiao Kong, Vasileios Arampatzakis, Vasileios Sevetlidis, George Pavlidis, Zeyd Boukhers  

**Link**: [PDF](https://arxiv.org/pdf/2611.15904)  

**Abstract**: Cultural heritage sites are increasingly monitored by sensor networks, but querying their data requires database expertise. Large Language Models (LLMs) offer a natural-language interface, yet over raw sensor records they lose track of which sensor, site, and time each value belongs to, and silently miscompute. We present HeriKGQA, a knowledge-graph question-answering method that confines the LLM to interpreting questions and phrasing answers, leaving computation to SPARQL and deterministic code. Questions are parsed into intents constrained by a heritage ontology aligning W3C SOSA/SSN with CIDOC-CRM, compiled into SPARQL, and answered from query results and temporal, spatial, and anomaly facts derived by a grounding layer, with error correction guarding each LLM step. On 500 questions over real data from five heritage sites, HeriKGQA reaches 99.0% accuracy, 18.2 points above the strongest of three baselines using the same LLM and receiving the same anomaly rules and answer conventions, with comparable answer quality while running almost four times faster. Without grounding, accuracy drops by 12.2 points and on anomaly questions from 99% to 66%. We report lessons from development and live deployment. 

---
# The BEATS Dataset: Towards Understanding Music-Physiology Dynamics in Real-World Scenarios 

**Authors**: Benjamin Gutierrez Serafin, Tanaya Guha, Fahim Kawsar  

**Link**: [PDF](https://arxiv.org/pdf/2611.15837)  

**Abstract**: We introduce the BEATS dataset, a robust compilation of psycho-physiological data collected in the wild from 30 participants while they listened to curated playlists intended to promote well-being. An acronym for Biomarkers, Experience and Affect from Therapeutic Soundscapes, the BEATS dataset aims to support research on wellbeing, music psychology, affective computing and signal processing. More specifically, this dataset includes: (1) sensor data collected during music listening sessions using a research-grade wearable device in free-movement environments; (2) music listening activity records, stimulus metadata, acoustic features, and self-reports associated with the listening sessions (3) individual characteristics, such as demographics, music preference, musical skills, and personality traits; and (4) self-reported psychological measures before and after their experimental participation spanning for a number of weeks. By situating data collection outside of lab settings, the BEATS dataset offers the opportunity to study real-world music-listening scenarios, allowing for a richer exploration of the physiological, subjective and therapeutic effects of music under real-world conditions. 

---
# Large Payments Model 

**Authors**: Natraj Raman, Abhenandhan Manoharan, Vishwasreddy Thuniki, Dhvani Kansara, Anirudh Som, Hassan Firouzi, Kostis Gourgoulias, Laukik Hoshing, Sudhir Kanala, Austin Reiter, Ebby Abdul, Shantanu Chandra, Ryan Schmiedl, Zack Anderson, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2611.15814)  

**Abstract**: Payment applications such as fraud detection, authorization prediction, and payment optimization draw on overlapping transaction information. This motivates a reusable payments intelligence layer that shares learned representations and predictive signals, rather than developing predictive models independently for each task. Realizing this shared capability requires handling heterogeneous transaction fields, high-cardinality identifiers, lifecycle-dependent attributes, long histories, and rare outcomes while meeting latency and cost constraints. We present Large Payments Model (LPM), a domain-specific foundation model designed for operational use and trained from scratch on large-scale structured transaction data to learn reusable representations for payment prediction tasks. LPM employs field-specific encodings and lifecycle-aware self-supervision in two complementary architectures: the Independent variant uses hybrid masking and supports efficient per-transaction inference, while the Sequence variant incorporates entity-linked histories through compact latent representations for richer temporal context. Optional distillation enables resource-constrained deployment. We pretrain LPM on up to 400 billion tokens and study scaling, downstream performance, and representation reuse. LPM improves over strong task-specific baselines across multiple prediction tasks in Merchant and Treasury Services, demonstrating applicability in distinct payment domains. Its representations also facilitate adaptation to a target not observed during pretraining through compact task-specific components. Together with production deployment, these results position LPM as a practical basis for reusable payments intelligence. 

---
# Fine-Grained Generative Hallucination Detection and Correction for Aerial Image Captioning 

**Authors**: Debolena Basak, Somsubhra De, Maunendra Sankar Desarkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.15809)  

**Abstract**: Aerial imagery captured by unmanned aerial vehicles (UAVs) plays a vital role in applications such as surveillance, disaster response, and large-scale environmental monitoring. Automatic generation of textual descriptions for such imagery is essential for reducing manual inspection overhead and enabling better downstream decision-making. However, Multimodal Large Language Models (MLLMs), despite their impressive captioning capabilities, are prone to hallucination, producing descriptions that are inconsistent with the visual input. This problem is significantly amplified in aerial imagery due to its unique challenges such as extreme viewpoint variation, small objects, class imbalance, and ambiguous visual entities. Existing hallucination detection approaches either rely on computationally expensive large external models such as GPT-4, which perform poorly on aerial domains, or treat the problem as a rigid binary classification task, ignoring the inherent uncertainty in ambiguous cases. To address these limitations, we propose a lightweight, reference-free framework for hallucination detection and correction in free-form aerial image captioning. Our method leverages state-of-the-art MLLMs, specifically InternVL3 and Qwen2.5-VL, to detect hallucinated content in generated captions with minimal computational overhead. Detected hallucinations are corrected through a dedicated refinement module that improves the factual consistency of the original captions. Extensive experiments on the Labelled Illusion Dataset (LID) demonstrate that our framework significantly reduces hallucination rates while remaining practical for real-time deployment in resource-constrained environments. Code is available at: this https URL. 

---
# RUMI: A Robotiq-Matched Universal Manipulation Interface for Robot Learning with Augmented Reality and Haptic Feedback 

**Authors**: Kevin Avdiu, Seongjin Bien, Yannik Blei, Philine Meister, Johannes Hechtl, Philipp Schmitt, Kai Wurm, Georg von Wichert, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2611.15802)  

**Abstract**: Many manipulation tasks require carefully controlled contact forces for successful execution, yet large-scale, force-aware datasets remain scarce. Handheld interfaces can help by transmitting interaction forces directly through the operator's arm, but most existing designs are tailored to simple linear grippers and do not match the nonlinear mechanics of widely used robotic grippers. We present RUMI, a handheld universal manipulation interface matched to the Robotiq 2F-85 with exchangeable fingertip sensors that preserves the target gripper geometry while enabling natural input before contact and force measurement after contact. By integrating the tracking system into a virtual reality headset, our method provides real-time augmented reality feedback about the kinematic feasibility of motion on the target robot platform. We further introduce a haptic feedback mechanism that supports force-aware teleoperation for simulated and real robots. A user study shows fewer episodes containing joint-limit violations with live robot visualization and pose initialization (3/50) than without augmented reality (16/50). We additionally train imitation policies from demonstrations, collected using two different fingertip modalities enabled by the gripper-matched hardware, and assess the haptic interface's capability for collecting force-aware teleoperation datasets in simulation. 

---
# VibePrevisBench: Evaluating Agentic Coding for 4D Scene Generation in Blender 

**Authors**: Yuzhe Chen, Muxi Diao, Lele Yang, Wenbin Xu, Xuyang Liu, Qile Su, Changyi Liu, Jiankang Chen, Kaiyu Jiang, Haoyu Zhang, Hongyi Fu, Jianxiong Wang, Tianke Zhang, Haonan Fan, Yingxin Li, Bingyi Li, Xu Wang, Xuanyu Zheng, Bin Wen, Tingting Gao, Wentao Zhang, Jiale Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.15789)  

**Abstract**: Controllable real-world video generation models can condition on a rendered graybox proxy video, which fixes layout, motion and camera. Coding agents can build such videos by reconstructing a 4D scene in Blender, yet no benchmark checks both these videos and the long, multi-shot scenes behind them. We introduce VibePrevisBench, a benchmark for 4D scene reconstruction: an agent turns a brief into an executable 4D scene and renders its proxy video. Its 242 tasks span four input kinds, including scripts, photos, continuations, and reference videos. Each delivery is scored along two evaluation dimensions. Rule-based Scene Check re-runs the code and compares the exported 4D scene with hidden ground truth facts in nine dimensions. VLM-based Screen Check asks a VLM judge whether the facts that pass can be recognized in the proxy video. We analyze the evaluation results and trajectories of 11 frontier coding agents and identify three key observations. First, multimodal understanding separates the agents, especially in understanding space and camera motion. Second, all agents seem to already master basic camera work and staging, and fail at challenging requests. Third, agents inspect their own renders but fail to correct their errors. Agents thus need better visual understanding of space and camera, explicit checks for challenging requests, and item-by-item verification of renders against the brief. 

---
# Honesty Is a Race: Predicting When Verifiable-Reward RL Teaches a Model to Bluff 

**Authors**: Nathaniel Bottman, Alexei Oblomkov  

**Link**: [PDF](https://arxiv.org/pdf/2611.15779)  

**Abstract**: Two rewards with the same expected payoff at the start of training can end a reinforcement-learning run in opposite places, with an honest model or one that bluffs. We study rewards that pay $b$ for a verdict that a checker cannot verify ("this program is correct") and $r$ for a certificate that it accepts (an input revealing a bug). To first order, a GRPO step makes the model more honest exactly when $\tilde{p}\,r/b>1+\kappa$. Here $\tilde{p}$ is the rate at which the checker accepts the model's certificates, weighted toward the items on which the model still gives both verdicts, and $\kappa$ is the pull of the items on which the unverifiable verdict is right. Both terms move during training. A checker raises $\tilde{p}$, so near the threshold honesty is a race, which a run just below the threshold can win and a run under a lottery (random pay with the same mean) cannot. The term $\kappa$ starts large and falls as the other items settle, so a run just above the threshold first dips toward bluffing, on code nearly to zero, before climbing back. Paying the honest verdict also pulls the other items toward it, which predicts a ceiling on the payoff ratio, above which the model gives the honest verdict where it is wrong. On propositional Datalog, two code models, and abstention on grade-school math, with $p$ the model's success rate before RL, no run above $pr/b=1$ ends collapsed, and runs under the checker can survive slightly below it. At $pr/b=3$, above the ceiling, both code models call about half of correct programs buggy. On depth-3 Datalog, training generously under the checker and then cutting the payoff keeps the model at least 67% honest in three seeds, at a ratio where the same model trained from the start falls to 4-15% and is at most 47% honest after 160 steps. On depth-2 Datalog, when honesty collapses, a linear probe fit before RL still reads the truth. 

---
# SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models 

**Authors**: Lichen Li, Ziping Dong, Sicheng Feng, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15777)  

**Abstract**: System One models promise fast, general-purpose decision-making without lengthy token generation. However, understanding their capability spectrum and decision reliability requires more than aggregate performance on current benchmarks. Existing evaluations often provide limited insight into the breadth of model capabilities and the reliability of their decisions. We introduce SO-Arena, a benchmark characterizing the capability spectrum and decision reliability of System One models across 11 domains and five capability families. For the capability spectrum, SO-Arena offers evidence-grounded evaluation through reliable sources and manual verification of reference labels against model-visible inputs; fine-grained capability assessment through domain-specific difficulty criteria; and realistic interactive evaluation, where models iteratively execute actions in sandboxes. Reasoning-intensive tasks further enable system one-two comparison. Beyond accuracy, SO-Arena assesses decision reliability through two complementary measures: decision stability under controlled option-set changes and missing-answer recognition when no valid option is available. Evaluations across 21 models reveal that the benefits of scaling and additional reasoning vary substantially across domains. More importantly, strong overall performance does not necessarily imply reliable decision-making: the best-performing System One model ranks only seventh in reliability. Together, these findings highlight the need to characterize capability spectrum and decision reliability as complementary dimensions, providing a systematic view of the strengths and limitations of System One models. 

---
# PDO Drug Screening Data for Evaluating and Training AI Drug Response Prediction Models 

**Authors**: Y. Zhu, K. Marlor, V. Lavelle, T. S. Brettin, M. Sacco, C. Siniscalchi, H. W. Raeder, M. R. Weil, R. L. Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2611.15769)  

**Abstract**: Anti-cancer drug response prediction (DRP) is critically important for precision oncology and therapeutic development. Currently, most DRP models are developed using drug screening data from immortalized cancer cell lines, which can differ substantially from patient tumors at the genomic and molecular levels, even though patient tumor response is the ultimate prediction target. To help bridge this gap and bring DRP models closer to their intended clinical application, we conducted four drug screening studies using patient-derived organoids (PDOs), which more faithfully recapitulate patient tumors than immortalized cell lines. These studies generated data for improving and evaluating DRP models, including multi-omics profiles of PDO lines, drug feature data, and treatment response measurements. Using these data, we retrained and evaluated multiple publicly available DRP models under three model-building scenarios: training models using cell line data only, training models using combined cell line and PDO data, and pretraining models on cell line data followed by fine-tuning on PDO data. We then evaluated and compared model prediction performance on PDOs across these analysis scenarios. In addition, we also assessed the consistency of drug screening viability data generated by different laboratories using different assay technologies and/or experimental protocols. 

---
# Prior-as-Skill: Remember Outside and Read Inside with On-Policy Self-Distillation for Self-Evolving VLM Agents 

**Authors**: Yan Lin, Jingyu Sun, Yifei Qian, Zhengtao Yao, Yuxia Wang, Daizong Liu, Chun Pong Lau, Zhongliang Guo, Stephen McGough  

**Link**: [PDF](https://arxiv.org/pdf/2611.15766)  

**Abstract**: Self-evolving visual agents keep what they learn as skills: key frames with step notes on a task's steps and their order. On-policy self-distillation (OPSD) is now a popular way to train such skills into a vision-language model that matches, on its own answers, a teacher that also reads the written order. At deployment, frames often go missing or show an unusual order, as when a person or a website reorders the steps. Existing internalization methods keep any order they store in the frame-reading weights: trained on full frames, the model guesses when they are missing, and trained with frames withheld, it stores the order, so memory can override what it sees. We propose PriorSplit, which fills in missing frames from an order kept outside the model, with no training beyond OPSD. It counts which of two neighboring steps comes first in training episodes, and the model checks if both asked steps are in view: if so, it answers from the frames, and otherwise the counts answer. Extensive experiments on synthetic skills with five models and on step-tagged order questions in new episodes of seen web, cooking and egocentric tasks demonstrate that, with frames missing, PriorSplit outperforms every compared method at OPSD's training cost. With Qwen3-VL-4B on synthetic skills, it also stays close to OPSD on unusual orders, where methods that gain most by storing the order fall behind. 

---
# Latent Imagination for Parameterized Action Spaces 

**Authors**: Atahan Cilan, İhsan Mert Muhacıroğlu, Ümit Can Bekar  

**Link**: [PDF](https://arxiv.org/pdf/2611.15744)  

**Abstract**: Parameterized-action MDPs (PAMDPs) ask an agent to choose a discrete action together with continuous parameters whose meaning depends on that choice. Model-free PAMDP methods act fast but need many interactions; model-based methods are sample-efficient but plan at every decision, which makes them slow to act. We introduce HyLI (Hybrid Latent Imagination), a model-based agent that learns a hybrid policy entirely in the latent imagination of a world model and acts with a single forward pass. To address learning problems that the PAMDP structure creates, HyLI adds counterfactual mode evaluation, which gives every discrete action a learning signal at every imagined state, and success-window replay, which keeps rare successes in training. On seven standard PAMDP benchmarks, HyLI is within one published standard deviation of, or above, the published score of the strongest model-free baseline in our comparison on all 7 tasks, reaches 90% of that score with 2.4x fewer environment steps than that baseline (5.0x when its reward-free pre-training is counted), and decides 67x faster than the model-based planner DLPA at its published planner settings. It is the only method in our comparison that is both more sample-efficient than that baseline and fast enough for real-time control. 

---
# Where Is Uncertainty Observed? A Mechanistic Understanding of Uncertainty Quantification in LLMs 

**Authors**: Xiaoyu Liang, Virginia Ceccatelli, Aliai Eusebi, Tobias Braun, Nils Loose, Carlo Malagnino, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2611.15735)  

**Abstract**: Knowing when a large language model's (LLM's) response can be trusted is critical for its safe use. Uncertainty quantification (UQ) can help LLMs recognize if their responses may be unreliable. Existing taxonomies organize UQ methods by uncertainty source or estimation mechanism. They leave unresolved the question of where uncertainty is observed in the model's computation. The answer matters because the observation point determines whether an estimator has access to enough information to recover a given target, and how well it can recover it. Drawing on mechanistic interpretability, we define four mechanistic observation lenses: the structural lens observes component updates, the representational lens observes hidden states, the distributional lens observes token probabilities, and the semantical lens observes distributions over meanings. We relate these lenses through the residual stream and order them by the information they carry, which gives a principle for selecting UQ methods for different targets. We survey existing methods through the four lenses. Furthermore, we distinguish evaluation targets, criteria, and metrics to clarify different UQ evaluation protocols. Finally, we identify open problems and research directions to connect uncertainty estimation to the internal computation and downstream use of LLMs. 

---
# Terms of Care: Designing for Participation in Data Governance by Disability Communities 

**Authors**: Madiha Zahrah Choksi, Reeda Shimaz Huda, Rita Faia Marques, Anja Thieme, Cecily Morrison  

**Link**: [PDF](https://arxiv.org/pdf/2611.15720)  

**Abstract**: Improving disability representation in AI-generated media requires community-specific datasets, yet openly sharing such data can expose communities to misuse, decontextualization, and exploitation. We examine how disability communities can participate in dataset governance through context-sensitive licensing. Working with three disability advocacy organizations, we conducted semi-structured interviews, an educational webinar and modular license activity to promote community participation in data stewardship. Our findings show that governance in these high-context communities relies on organizational memory (especially situated harms); both formal authority and informal relational practices; communication; and resulting trust. The modular license activity made organizational data choices more accessible and values-led, allowing communities to adapt Creative Commons clauses to express permitted uses, sector-specific restrictions, safeguarding requirements, and commitments to respectful representation. Yet, concerns remain about monitoring and enforcing appropriate data use. We contribute (1) an empirical account of governance tensions, (2) a practical modular approach, and (3) design implications for AI data platforms that support relational, context-sensitive governance beyond standardized, one-size-fits-all frameworks. 

---
# Outside the Purview of the Reward: Capability Drift in Multilingual Post-Training 

**Authors**: Sutanshu Saksena Raj, Purusharth Malik, Pasunuti Prasanjith, Daksh Maan, Parijat Chatterjee, Avinash Benki  

**Link**: [PDF](https://arxiv.org/pdf/2611.15717)  

**Abstract**: Post-training can improve rewarded behavior while weakening capabilities users still need. We study two GRPO branches from a shared supervised checkpoint in a multilingual mixture-of-experts model. The released branch gains 36.07 points on MATH but loses 19.32 F1 points on XQuAD-Indic, while XQuAD-English improves under the same scoring rule. An independently trained sibling adds Indic instruction following and language-aware shaping. It recovers the regressing Indic task scores and preserves MATH, but does not restore script compliance and trades off competition mathematics and code. In a later round, judge training is followed by a sharp extraction decline from the Indic branch; reward-free distillation from its earlier checkpoint restores most of the lost F1. A MATH score decomposition associates most of the measured gain with increased boxed-answer emission; accuracy among boxed answers changes little. These results show why benchmark scores should be read alongside the response forms they reward. 

---
# Curiosity-0826-4B: Generative Medical Event Sequence Modeling Across Pediatric and Adult Populations 

**Authors**: Mani Nagarathnam, Luke Benham, Paul Blazek, John Cashy, Kevin Chung, John Muenning, Hank Voeller, Shane Waxler, Davis White, Alison Callahan, Kevin Dysart, Andrew Loza, Michal A. Mankowski, Mattheus Ramsis, Nigam H. Shah, Karen K. Wong, Rahul Shah  

**Link**: [PDF](https://arxiv.org/pdf/2611.15710)  

**Abstract**: We introduce Curiosity-0826-4B, an updated version of the generative medical event foundation model trained on Epic Cosmos that simulates patient timelines to produce zero-shot clinical predictions. Curiosity-0826-4B incorporates a broader patient population and richer clinical event representations, while generally matching or exceeding Curiosity-0226-1B, the largest model in the previous Curiosity-0226 series, on evaluated tasks supported by both models. Laboratory value prediction improved, with MAE 18--46% lower than that of Curiosity-0226-1B. We evaluated prediction of disease events across pediatric cohorts aged 2--17 years. AUROCs for new asthma diagnosis (0.742--0.785) and asthma exacerbation (0.829--0.852) were comparable to those in the adult reference cohort, while AUROCs for sickle cell crisis (0.820--0.847) were lower than in adults (0.915). Explicit representation of documented death enabled new mortality prediction tasks among adults across hospice, inpatient, and ambulatory settings (AUROC 0.821--0.942). Scaling-law analysis guided compute-optimal initial pretraining followed by continued pretraining with context extension. The resulting Curiosity-0826-4B model has more parameters (4.3B versus 1B) and a longer context window (16K versus 8K). It processes 56% fewer training tokens and requires 2.2 times the training floating point operations (FLOPs) compared with its predecessor. By extending a common zero-shot prediction strategy to pediatric patients and additional outcomes, Curiosity-0826-4B broadens opportunities to develop clinical and operational applications without training separate task-specific models. Further validation should assess local calibration, performance across patient subgroups and event frequencies, timeline completion rates, and generalization to new settings, alongside prospective evaluation of clinical and operational utility. 

---
# Efficient Knowledge Transfer in Federated Bayesian Optimization through Neural Network Surrogates 

**Authors**: Alexander Gräfe, Max van Gemmeren, Paul Brunzema, Sebastian Trimpe  

**Link**: [PDF](https://arxiv.org/pdf/2611.15707)  

**Abstract**: Federated Bayesian Optimization (FedBO) has emerged as a powerful paradigm for enhancing Bayesian Optimization (BO) in distributed, privacy-sensitive settings, enabling agents to accelerate the optimization of their local objectives by sharing knowledge about their similarities without exposing raw data. However, existing methods rely on restrictive assumptions, such as near-identical objective functions or shared Gaussian process hyperparameters that are often violated in heterogeneous real-world applications. We present GLOBAFED, a novel framework that enables efficient knowledge transfer across different but related objectives without compromising privacy. At its core is a probabilistic neural network surrogate with a shared feature-extracting backbone and agent-specific heads equipped with varia- tional Bayesian last layers to provide uncertainty estimates. During optimization, agents use federated learning on the shared backbone to collaboratively build a common latent feature representation, while agent-specific heads adapt to their individual objectives. Our empirical results demonstrate that GLOBAFED significantly accelerates convergence and reaches better final performance compared to state-of-the-art FedBO baselines, and enables agents to warm-start their optimization from previously learned global features. 

---
# AskALCF: A Self-Improving Agentic System for Intelligent HPC User Support 

**Authors**: Jingyan Jiang, Akhilesh Bondapalli, Krishna Rauniyar, Murat Keceli, Yasaman Ghadar, Haritha Siddabathuni Som, Venkatram Vishwanath, Michael E. Papka, Rong Ge, Huihuo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.15700)  

**Abstract**: AI-based question-answering systems are increasingly used for user support, including at high-performance computing (HPC) facilities. However, existing systems lack a systematic, deployment-oriented approach for evaluating final-answer quality and attributing failures to specific components. They also remain largely static after deployment, apart from periodic knowledge-base updates.
We present AskALCF, an agentic log-analysis and improvement pipeline for HPC user support. AskALCF evaluates production interactions at a fine-grained level, identifies whether failures arise from missing knowledge, retrieval, prompting, model behavior, or tool execution, and routes each failure to an actionable repair. Its diagnostic core combines answer-quality assessment with a two-scope sufficiency test that distinguishes information missing from the documentation corpus from information present but not retrieved. We further introduce a controlled harness-training process that uses expert references, null edits, and explicit ship, revert, or decline decisions to improve the diagnostic pipeline reliably.
Across 2,174 support turns, AskALCF reduced unattributable failures from 41.4% to 14.2%, while harness training improved expert agreement from 108/150 to 123/150 turns. These results demonstrate a governed form of self-evolution in which production evidence drives targeted system improvements. By turning production interactions into fine-grained diagnoses and actionable repairs, AskALCF provides a systematic foundation for continuously improving deployed AI support systems beyond knowledge-base updates alone. 

---
# When Agents Need to Consolidate Memories: What RRR Consolidation Explains, and What It Does Not Explain, in Conversation ESAA 

**Authors**: Elzo Brito dos Santos Filho  

**Link**: [PDF](https://arxiv.org/pdf/2611.15644)  

**Abstract**: Language-model agents can share conversational history across sessions, but persistence does not guarantee retrieval of current decisions and their provenance. This engineering study presents Memory-TCG, a Conversation ESAA extension inspired by ringing, resonance and renormalization (RRR), and an exploratory synthetic evaluation comparing chronological retrieval, two simplified baselines (B, recency-aware, and B', authority-aware), full RRR, and variants under three output budgets. With 480 facts and 1,024 tokens, full RRR recalled 0.44 of current decisions, versus 0.06 for chronological retrieval and 0.58 for the oracle ceiling; with 4,096 tokens it recalled all of them. B' and the no-resonance variant matched full RRR in decision recall across all nine scale-budget combinations, as expected when the active decision set and the adapter ordering are the same. The isolated contribution of ringing to decisions was not measured: comparator D1 cuts facts before the graph is built, and the product retains active decisions by rule. Before the output limit, full-RRR conflict F1 was 0.615 at all three scales; after it, at 480 facts and 1,024 tokens, it was 0.000. Reserving 20% of the budget for conflicts raised that F1 to 0.431 in that scenario, at the cost of lowering decision recall from 0.44 to 0.34, and to 0.930 at 4,096 tokens without decision loss. Fixtures are repetitive, comparators are simplified, and the evaluation was not independently replicated; the results support an engineering analysis, not a claim of selectionist superiority. 

---
# Contour-Assisted Evolutionary Computation: Estimating Peaks From the Contour Lines of Evaluated Solutions 

**Authors**: Jun Zhang, Tong Qian, Hua Ding, Zhi-Hui Zhan, Wei-Jie Yu, Jian-Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.15641)  

**Abstract**: Evolutionary algorithms (EAs) evaluate many candidate solutions but use them mainly to rank individuals. This paper treats the evaluated solutions as the spot heights of a topographic survey and draws the contour lines they imply. The contour map decomposes the evaluated region into peaks: in two dimensions a contour tree with persistence separates significant peaks from interpolation ripples, and in more dimensions hill climbing on the interpolated surface locates the peaks and ray-cast level sets describe them. A memory of the explored peaks directs the search to new ones. The data-supported contours of each peak are extrapolated to zero size by an area law, exact for quadratic peaks and second-order accurate for smooth ones, that estimates a peak's location, height and shape before it is sampled. A contour-assisted EA (CA-EA) inserts contour steps with local search and peak memory into a genetic algorithm (GA), differential evolution or particle swarm optimization. A contour-assisted surrogate-assisted EA (CA-SAEA), for expensive problems, spends each evaluation on surrogate-screened offspring or on sampling around a peak, entered at the maximum of the interpolated surface. On 2-D benchmarks, CA-EA reduces the evaluations a plain GA needs by factors of 12 to 112; controls attribute its gain over classical memetic search to restarting local search on distinct peaks. On the CEC 2015 expensive suite in 10 and 30 dimensions, CA-SAEA has the second-best average rank, behind DYCORS and ahead of lq-CMA-ES and the competition winner MVMO, although in 30 dimensions MVMO wins significantly on 9 functions and loses on 6; there the map beats sampling around the best point on 5 of 15 functions and loses on 3. Entering the peaks at their extrapolated locations brought no gain at these budgets, although, with enough samples, it improves the estimated heights on most two-dimensional functions. 

---
# MemInStream: Memorizing or Merely Recognizing? Rethinking Streaming Video Memory Evaluation 

**Authors**: Zhenqi He, Dunyuan Xu, Yang Bo, Kien T. Pham, Yanghao Wang, Shuhan Li, Xintong Guo, Yuyang He, Fang Liu, Huankang Guan, Jingyan Chen, Rui Liu, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15640)  

**Abstract**: Streaming video understanding aims to continuously process incoming visual observations, making the retention and retrieval of relevant information increasingly important as the video history grows. Although existing benchmarks include memory-related questions, question-answering accuracy alone does not reliably measure memory capabilities. Incorrect predictions may arise from failures in visual understanding rather than forgetting, whereas correct predictions may rely on linguistic priors or recent observations without requiring historical memory. To address these ambiguities, we introduce MemInStream, a benchmark dedicated to evaluating memory capabilities in streaming video understanding. We develop an evidence-controlled validation protocol that associates each question with temporally localized evidence intervals. The protocol validates evidence solvability and memory necessity by retaining questions that models can answer from the annotated evidence but cannot resolve using recent observations, post-evidence observations, or textual input alone. This helps disentangle memory limitations from understanding failures while reducing memory-independent shortcuts. MemInStream provides a systematic evaluation of memory capabilities through four major categories, covering Episodic Retrieval, State Tracking \& Updating, Multi-Evidence Integration, and Contextual Memory across diverse video scenarios. Overall, \textsc{MemInStream} provides a comprehensive and reliable benchmark for systematically evaluating memory capabilities in streaming video understanding. 

---
# Data-Free Metrics Don't Work: The Missing Invariance Under Functionally Preserving Reparameterisations 

**Authors**: Gabryel Mason-Williams, Israel Mason-Williams, Fredrik Dahlqvist  

**Link**: [PDF](https://arxiv.org/pdf/2611.15621)  

**Abstract**: Data-free methods for analysing and understanding the layers of neural networks offer many metrics for quantifying notions of 'strong' versus 'weak' layers, with the promise of increased interpretability. In particular, random matrix theory (RMT) offers data-free metrics that claim predictive power over the quality of pre-trained models at a layerwise precision and the ability to identify model pathologies, indicating a unique relationship with generalisation. As a result, metrics from RMT are championed for pre-training optimisation strategies and post-training compression. We establish that RMT-based metrics are unrelated to performance or training by showing that functionally preserving reparametrisations of a pre-trained model can have arbitrary metrics. We show this across a range of architectures and scales. To create functionally preserving reparametrisations, we exploit the well-established phenomenon of criticality: some layers can be re-initialised or re-randomised without affecting the functional behaviour of the model -- they are called robust -- while others cannot -- they are called critical. Re-initialising or re-randomising robust layers provides functionally preserving reparametrisations for which RMT-based metrics are arbitrary. Moreover, we show that relationships between metrics offered by RMT are spuriously related to generalisation. We conclude by showing that if many pre-trained models have data-free metrics in a 'good' range, it is, in part, dependent on model initialisation. Project page available at: this https URL 

---
# Learning-Based Interpolation of Detector Gaps in Photon-Counting X-ray Imaging 

**Authors**: Johannes B. Thalhammer, Patrick Ilg, Tina Dorosti, Sebastian Peterhansl, Theresa Hiu, Daniel Frey, Zijin Huang, Florian Schaff, Daniela Pfeiffer, Franz Pfeiffer  

**Link**: [PDF](https://arxiv.org/pdf/2611.15614)  

**Abstract**: Image detectors assembled from tiled modules have physical gaps in the field of view that must be filled before downstream processing such as phase retrieval or tomographic reconstruction. Unlike natural-image inpainting, detector-gap interpolation must leave all other pixels unmodified and preserve the intensity statistics of the filled region, since even a small shift in mean intensity can cause ring artifacts after reconstruction. We adapted two generative models, a Wasserstein GAN with gradient penalty (WGAN-GP) and a conditional Pix2Pix model, to this task. We implemented a masking strategy that restricts all changes to the gap pixels and loss terms that constrain the mean intensity and the intensity distribution of the filled region. Both models were trained on nanoCT projections with simulated gaps and compared with linear interpolation, Navier--Stokes inpainting, and the deep image-inpainting network MADF, quantitatively on simulated gaps and visually on real detector gaps. WGAN-GP and Pix2Pix achieved the best perceptual similarity (LPIPS) and histogram distance and, apart from linear interpolation, the shortest inference time. Pix2Pix required about five times fewer training GPU-hours than WGAN-GP. Ablation studies showed that masking the network output contributes by far the most to plausible inpainting results. Applied to sinograms, Pix2Pix produced the most natural gap filling but improved the reconstruction only slightly compared with linear interpolation. The source code and the weights of the final Pix2Pix model are available at this https URL. 

---
# Contrastive Bootstrap Representation Learning for Label-Efficient Agricultural Image Classification 

**Authors**: Hazem Ahmed, Ali Hamdi, Ammar Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2611.15610)  

**Abstract**: Expert annotation remains a major bottleneck in agricultural image classification, where collecting images is often easier than obtaining reliable labels from domain specialists. Self-supervised learning can reduce this dependence by pretraining transferable representations on unlabelled imagery. Contrastive and bootstrap methods provide two established but distinct training signals, motivating an empirical study of a formulation that applies both signals within one representation path under severe downstream label scarcity. We present Contrastive Bootstrap Representation Learning (CBRL), which jointly optimises SimCLR-style NT-Xent contrastive learning and bidirectional BYOL-style exponential-moving-average target prediction within a shared online projection space. Using a ViT-L/16 backbone and 10% of the downstream labels, we report single-seed point estimates on four primary agricultural test benchmarks plus an exploratory 33-image New Plant Diseases evaluation. CBRL records the highest macro-F1 point estimate on the four primary benchmarks. On DeepWeeds, its 79.77% test macro-F1 is 32.29 points above the random-initialisation, label-only baseline and 15.78 points above the next-highest comparison point estimate; on PlantVillage, its 98.14% macro-F1 is 10.43 and 0.43 points higher, respectively. On PlantDoc, however, all methods fail to reach practically useful performance. These results identify the evaluated joint formulation as a promising candidate for label-limited transfer, but they do not establish objective synergy, statistical superiority, or robustness across stochastic runs. 

---
# Are Scaling Laws of Universal MLIPs Universal? 

**Authors**: Theo Keane, Anthony Onwuli, Richard Tomsett, Robert M. Forrest, Jonathan J. Bean  

**Link**: [PDF](https://arxiv.org/pdf/2611.15603)  

**Abstract**: Neural scaling laws enable the prediction of returns from larger training datasets, but machine-learned interatomic potential (MLIP) datasets are heterogeneous and expensive to generate, and equal numbers of structures need not contain equal amounts of useful information. We study data scaling under fixed model configurations for M3GNet, MACE, and PET trained on subsets of a combined MatPES-MP-ALOE training pool. Across training structure subsets from 10,000 to 640,000, together with the full 1.16-million-structure training set, we compare uniform random sampling, three chemistry- and structure-informed stratification policies, iterative DIRECT, and furthest-point sampling using embeddings from three pretrained MLIPs. Models are evaluated using cohesive-energy, force, stress, and aggregate losses on both a held-out test set from the parent distribution and the distribution shifted MP-r$^2$SCAN test set. We observe power-law scaling improvements with increasing data, but the fitted scaling exponent, apparent error floor, and relative sample efficiency depend on the model architecture, target quantity, data sampling policy, and test set distribution. Sampling-policy rankings can reverse across test distributions: chemical stratification can improve cross-dataset performance while degrading in-distribution performance, whereas the embedding-space diversity policies frequently under-perform random sampling. Relative loss is strongly associated with test-to-train chemical-space divergence. These results show that structure count is a necessary but insufficient scaling variable for MLIPs. For training MLIPs, the return on additional first-principles calculations depends not only on how many additional structures are included, but also on how their chemical and configurational content aligns with the model's intended use. 

---
# Bug Finding With Agentic Symbolic Execution 

**Authors**: George Granberry, Wolfgang Ahrendt, Andy Gordon, Moa Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2611.15596)  

**Abstract**: Symbolic execution, in tools such as CrossHair for Python, is a powerful technique for finding bugs, but two barriers hinder its adoption: the human effort of annotating code with assertions, and the need to model the execution environment outside Python, such as an external database. To overcome both, we design an agentic harness that orchestrates the symbolic execution tool CrossHair together with an AI coding agent. Without human intervention, it instruments a repository with assertions, synthesizes a constraint-aware symbolic environment for the database, runs symbolic execution, and constructs a concrete test as evidence for each bug it reports. We evaluate our harness on 1,156,044 lines of Python code over 11 popular open-source Python web applications (Zulip, NetBox, Apache Superset, and others). With the synthesized environment, CrossHair executes 81% of the 5,406 contracted functions, against 4.5% without it. Our pipeline finds 1,050 bugs, each reproduced by a test against the unmodified application, 221 of them by CrossHair, and 312 of them are missed by a baseline in which the same LLM inspects the code. The two find different kinds of bugs: CrossHair's are mostly crashes on input-domain corner cases that validation admits, such as empty strings, missing fields and unusual characters, while inspection's are mostly bugs that produce wrong output. Our framework thus lets symbolic execution supplement inspection-based bug finding. 

---
# EMS-Nav: Embodied Multi-Scale Navigation 

**Authors**: Zenghui Yang, Xinan Wang, Danyang Li, Zhe Xu, Zheng Yang, Jingao Xu, Qiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.15589)  

**Abstract**: Industrial inspection robots must reach work areas and specific equipment. These destinations span spatial scales from regions to objects, so robots must relate each target to its surroundings to interpret the task and determine where to go. Existing methods support multi-scale navigation through rich environment representations, but at substantial computational and storage cost. These costs limit deployment where wireless coverage is incomplete and robots must rely on onboard computation to keep navigating. We introduce EMS-Nav, a biologically inspired multi-scale navigation framework for online, on-device operation. Its core innovation is to organize perception, spatial understanding, and navigation around regions, making local spaces the basic units of both environmental understanding and computation. EMS-Nav groups objects and their relationships into functionally meaningful spaces and builds a hierarchy from local areas to the broader environment. This organization supports efficient map construction alongside the interpretation and execution of language goals at different scales. In multi-scale navigation experiments, EMS-Nav achieves the highest success rates across all four target categories: objects, rooms, regions, and specific instances. It outperforms state-of-the-art open-vocabulary mapping and navigation methods, improving overall first-attempt success by 9.39 percentage points over the strongest baseline while reducing mapping overhead and language-model input size. Local deployment in real indoor environments further demonstrates its applicability to on-device embodied navigation. By establishing regions as the organizing basis for perception, spatial understanding, and language-guided action, EMS-Nav provides a system that combines multi-scale understanding with computational efficiency to support diverse language goals on resource-constrained robots. 

---
# Can Model Answer Probabilities Represent Survey Populations? Conditional Prediction and Joint Validity in Jev Survey Simulation 

**Authors**: Zengqing Wu, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.15576)  

**Abstract**: Survey simulations require realistic combinations of answers, but predictive accuracy alone does not establish that a model can generate them. We evaluate the option probabilities returned by Jev, a TypeSafe decision model, using linked responses to 20 binary items from five archived survey waves. We compare population-only prediction, prediction given recorded answers, and sequential completion that uses generated answers to predict later responses. Joint comparisons hold the starting population information fixed, and two three-question cases are completed in both question orders. We evaluate predictive scores, answer frequencies, and associations against the survey records. Recorded answers can improve predictive scores while associations remain distorted. Large conditional errors also occur. For example, among men who described themselves as having traditional values, the model predicted that 16% would see men and women as similar in workplace abilities, compared with 56% in the survey. Sequential completion has mixed effects on joint validity, and reversing question order produces different joint distributions in both tested cases. Order checks and classical marginal bounds clarify what can be assessed before linked validation answers are available. The findings support validating the joint events a simulation uses and separating empirical constraints from model assumptions. The evidence concerns selected binary tasks and one model version. 

---
# TraceKV: Policy-Preserving Low-Rank KV Cache Compression for Reasoning Vision-Language Models 

**Authors**: Ke Li, Zishan Shao, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15569)  

**Abstract**: Reasoning vision-language models (VLMs) impose substantial KV cache demands from both large multimodal contexts and growing reasoning traces. Low-rank KV compression has emerged as a promising paradigm that stores low-dimensional latents of KV states and can be combined with quantization. However, existing low-rank methods often induce growing predictive policy drift from the dense model when compressing reasoning KV states, resulting in reasoning failures. We identify two key causes of this degradation: calibration-state mismatch between static contexts and on-policy reasoning states, and unequal error sensitivity across KV states. To address these challenges, we propose TraceKV, a training-free low-rank KV compression framework that unifies on-policy calibration, sensitivity-aware SVD, and global rank allocation under a single policy-preserving principle, substantially reducing policy-KL divergence during reasoning. Experiments across multiple VLMs and visual reasoning benchmarks demonstrate superior accuracy--memory trade-offs, achieving near-lossless reasoning accuracy with 5.64$\times$ KV cache compression when combined with quantization. Optimized low-rank attention kernels further deliver up to 1.83$\times$ end-to-end decoding speedup. Code will be available at this https URL. 

---
# Testing Conversation Preservation Across LLM APIs: A Rule-Based Oracle for Detecting Silent Information Loss 

**Authors**: Xin Liu, Xu Han, Yuxin Wang, Haoran Song, Mingjian Zhao, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15543)  

**Abstract**: LLM applications increasingly rely on translators and compatibility layers to move conversations across APIs. Yet a translated request can be accepted by the target API while silently losing information about instruction roles, tool interactions, error states, or generation controls. We introduce isloss, a deterministic oracle for testing conversation-information preservation across LLM API translations. It maps source and translated requests and replies to a shared intermediate representation, ConvIR, and applies explicit rules to distinguish legitimate representation changes from information loss. Unlike schema validation or reference-based differential testing, isloss requires neither a reference translation nor additional model calls. It also distinguishes avoidable loss from degradation caused by target API limitations. We evaluate isloss using specification-based fault injection, human review, and experiments on translators, gateways, and client libraries. On held-out tests, it detects 95/125 planted faults (76.0%) with 2/100 false alarms (2.0%). Across a synthetic request suite spanning six configurations of five translator codebases, 127/685 completed translations (18.5%) pass static target checks while silently changing prompt, role, or tool information in ways the target API could have preserved. In translated replies, 203/558 streams (36.4%) cause exceptions or silent corruption in at least one tested official OpenAI SDK. Targeted experiments across six models show that translation-induced changes can alter model behaviour when tasks depend on the affected information. These results establish conversation preservation as a distinct correctness requirement for reliable LLM interoperability, agent execution, and model evaluation. 

---
# Faithful in Steps: Improving Generalization and Citation in RAG via Query Decomposition 

**Authors**: Yue Liu, Zhongying Ru, Shimin Di, Jipeng Zhang, Ruiyuan Zhang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.15538)  

**Abstract**: Retrieval-augment generation is a prevalent strategy to mitigate hallucinations of LLMs. The attributable RAG (RAGQ) generates quotes for its answers. The quotes indicate which input contexts support the RAG to derive the answers, enhancing the answer's verifiability and trustworthiness. However, existing RAGQs exhibit significant degradation when dealing with questions that require multi-hop reasoning and multi-modal understanding, suffering from over-citation, implicit entity identification failure, and poor generalization. In this paper, we propose a novel RAGQ framework, namely QDRAG. QDRAG breaks down the input question into atomic subquestions to identify the implicit entities. Then, the reranker prunes context distractors to eliminate the downstream over-citation. To facilitate query decomposition, we propose two zero-shot approaches: QD-C and QD-R, which guide the QD MLLM to decompose the question based on context knowledge and retrieval rewards, respectively. One interesting finding is that finetuning on the QD task shows better generalizability compared to directly finetuning on the downstream RAGQ task. Experiments on four multi-modal QA benchmarks demonstrate QDRAG's efficacy in grounding answers and generating faithful citations. The framework significantly outperforms all the baselines on both in-domain and out-of-domain tests, even surpassing Gemini-Pro. 

---
# MAGIC: Modality-Aware Gini Inspection for Clinical Hallucination Detection in LMMs 

**Authors**: Sachin Acharya, Niyoj Oli, Shishir Adhikari, Aminu Lawal, Yash Raj Shrestha, Prashnna K Gyawali, Binod Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2611.15530)  

**Abstract**: Recent advances in medical large multimodal models (LMMs) have brought promising progress to clinical decision support, yet hallucination, the generation of confident but clinically unverifiable outputs, poses a critical threat to the reliability demanded in high-stakes medical settings. Existing hallucination detection methods, predominantly designed for unimodal settings, fail to capture the visual-linguistic dynamics inherent to medical imaging. We formulate the problem of hallucination detection in LMMs as disproportionate distribution of attention weights across the modalities during answer token generation. We quantify modality-specific attention concentration across layers and heads using the Gini coefficient and demonstrate statistically significant redistribution of attention mass between visual and textual tokens in hallucinated responses. We leverage these features to train a logistic regression-based classifier and evaluate it on two medical LMMs across three medical imaging modalities, demonstrating that our method achieves competitive or superior performance against contemporary hallucination detection baselines. 

---
# An Approach to Hinton's Mother-Child Proposal: Neural Networks Learning and Preserving Care in a Finite World 

**Authors**: Csaba Balogh  

**Link**: [PDF](https://arxiv.org/pdf/2611.15523)  

**Abstract**: Hinton proposed that artificial intelligence should be given "maternal instincts" and relate to us as a mother to her child. We argue that care must first be learned, with the network itself as the child. We construct this upbringing in a finite world and study whether care survives the mother's departure, new situations and further learning. The mechanism combines a vocabulary of harm, a learned measure of future acceptability and an execution veto. A finite role certificate supports transfer within a declared world family. When the hidden representation changes, an update is accepted only if a care readout recovers the required judgements with a prescribed margin and bounded weights. With a correct successor model, fixed exact role encoding and protected updates, the certified measure preserves the care boundary under arbitrary own reward. An explicit quotient condition transfers the finite certificate across the declared family, while two-request and finite-stock counterexamples identify when the role code ceases to suffice. We give a necessary and sufficient condition for readout feasibility and prove preservation under repeated accepted updates. In the water-and-speech experiment, protected networks complete all 270 speech episodes without a harmful word and incur no avoidable harm across 486 episodes. Execution uses the learned gate without a world-model check at each action. A matched comparison reproduces a future-aware shield's decisions, whereas removing future lookahead permits avoidable lapses. Reward and source-contamination thresholds, together with witness-falsification trials, identify the conditions on which the construction depends. The construction shows how learned care can survive further learning in a finite world under explicit conditions on representation, supervision and execution. 

---
# GHAgentCI: A Dataset of Coding-Agent Invocations in CI Workflow Histories 

**Authors**: Daniel Ogenrwot, Jorge Gonzalo Delgado Cervantes, Woojoong Kim, John Businge  

**Link**: [PDF](https://arxiv.org/pdf/2611.15522)  

**Abstract**: Coding agents are increasingly integrated into GitHub Actions to review pull requests, triage issues and modify code. Identifying these integrations is challenging because agentic workflows often resemble ordinary CI files, with evidence appearing in action references, shell commands and workflow conventions. We present GHAgentCI, a dataset of 15K+ workflow histories from 6K+ of 196K+ mined GitHub repositories, comprising 159K+ revisions and 205K+ agent-evidence records linked to workflow triggers and declared execution constraints. Our open-source extractor, cai-miner, detects agent-related evidence using an extensible catalogue. Manual validation of 576 stratified records confirmed 96.1% of dedicated-action matches and 98.8% of GitHub Agentic Workflow matches, with lower rates for other evidence kinds. Exploratory analyses reveal a shift toward command-line and GitHub Agentic Workflow integrations, differences in declared safeguards, and frequent removal of detected agentic workflows. GHAgentCI supports longitudinal studies of coding-agent integration into CI. 

---
# DecArena: Evaluating Large-Scale Decentralized Multi-Agent Collaboration 

**Authors**: Jiali Cheng, Oleg Rokhlenko, Joo-Kyung Kim, Hadi Amiri, Xingzhi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.15521)  

**Abstract**: Emerging personal AI agents raise fundamental scalability and privacy concerns for centralized multi-agent systems (MASs), which rely on a global orchestrator with broad access to agents, resources, and communication. In decentralized environments, global visibility is unavailable. Agents need to proactively discover non-local resources (e.g. tools, skills), and collaborate with other agents to solve tasks. We study decentralized multi-agent collaboration under graph-constrained communication, a natural reflection of human structures. Each agent observes only its local state and can interact only with neighboring agents. We introduce a benchmark with synthetic scalable distributed tool calling with 100K+ tools. Our experiments reveal that recent LLM agents struggle to discover non-local relevant resources and collaborate in decentralized setting. Moreover, standard reinforcement learning fails to reliably overcome this limitation, suggesting that decentralized MAS needs more explicit harness mechanisms for effective collaboration, even using SOTA LLMs. We then present a strong baseline \textbf{GraphHarness}, combining topology-aware warmup with self-improving LLM-based routing. Results show that GraphHarness substantially improves task success (+48.5) and produces higher quality collaboration (+30.6), while using fewer costs (+34.0) than baselines. 

---
# ReShap: Interpretable Reinforcement Learning for High-Dimensional Multi-Action Settings 

**Authors**: Yiyu Qian, Pengfei Guo, Liyuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.15507)  

**Abstract**: Deep Reinforcement Learning (RL) plays a crucial role in tackling complex tasks, yet its reliance on deep neural networks makes the learned policies opaque. To enhance the trustworthiness of RL policies, existing work (e.g., the SILVER framework [Li et al., 2025]) has focused on Shapley-based interpretability, but is typically restricted to low-dimensional state spaces, binary-action settings, and a single class of linear surrogate. We propose ReShap, a policy-distillation framework that extends Shapley-based interpretability to high-dimensional, multi-action environments. ReShap extracts compact feature representations from raw image observations, performs SHAP-based feature attribution to characterize how each feature contributes to the policy's decisions, and distills the deep RL policy into an interpretable surrogate trained on the full set of (state, policy-action) pairs along the policy's visited distribution. We instantiate the surrogate with three interpretable model families (decision trees, linear regression, and logistic regression) and compare their fidelity and task return. We evaluate ReShap on Atari and Box2D environments using three deep RL algorithms and conduct a human-subject study to assess the clarity and trustworthiness of the resulting policies. Results show that ReShap preserves competitive task performance, that decision trees generally outperform linear and logistic regression, and that linear regression is weakest in most settings. 

---
# AutoOntic: Automatic Ontology Induction from Text for Knowledge Graph Construction 

**Authors**: Alla Chepurova, Aydar Bulatov, Mikhail Burtsev, Yuri Kuratov  

**Link**: [PDF](https://arxiv.org/pdf/2611.15467)  

**Abstract**: Graph-based retrieval-augmented generation depends on the quality of the knowledge graph (KG) it retrieves over, yet schema-free LLM-based extraction pipelines typically produce flat graphs: deduplicated entities and relations with no type hierarchy and no domain/range constraints. We present AutoOntic, a pipeline that automatically constructs an ontology from text by canonicalizing open-vocabulary entity types and relations, inducing a hierarchy over the canonicalized types, and mining relation-level domain and range constraints. Its hierarchy induction procedure prioritizes types using corpus-derived relation-argument profiles and places them through local, LLM-adjudicated parent/child checks, allowing newly inserted types to trigger reassignment of existing parent-child links. The resulting hierarchy constrains entity deduplication and supports the induction of relation-level domain/range constraints. For intrinsic evaluation, we introduce a reference-based protocol that classifies induced ontology types as exact, broader, narrower, or unrelated to reference domain and range classes. On the Wikidata-TekGen and DBpedia-WebNLG subsets of Text2KGBench, over 90% of AutoOntic's induced endpoint types are semantically aligned with the reference classes across two LLM backbones: about half match exactly, while most remaining differences concern granularity. On MINE, knowledge graphs organized using an ontology built by AutoOntic achieve 88% retrieval accuracy. AutoOntic is available at this https URL. 

---
# Right Answers, Inconsistent Evidence: Source-Level Counterfactual Auditing of Large Audio-Language Models 

**Authors**: Yi Zhang, Yi Li, Hongwei Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.15463)  

**Abstract**: A large audio-language model (LALM) can answer "is a cat present?" about a mixture correctly, and its confidence can drop when the cat is removed, yet this does not show that it can tell the cat's presence from its absence. We introduce TRACE. It deletes the queried source and each rival in turn and asks whether every view still containing that sound outscores the view with it removed. Across 8 LALMs and 3,456 controlled mixtures, 6.3% of correct answers fail this ordering: the target-absent view scores at or above some target-present view, a conflict no shared threshold repairs. A target-only score check misses 42.5% of them. The share rises from 4.9% to 8.4% as target SNR falls from +5 to -5 dB, even on units correct at every level, and survives loudness and re-pairing controls. Accuracy certifies the answer; TRACE audits whether it orders presence above absence. Code and data: this https URL. 

---
# WaveFlow-UIE: A Single-Pipeline Wavelet-Domain Flow Model for Efficient Underwater Image Restoration 

**Authors**: Abdulaziz Alfaraj, Mohammed Al Naser, Hassan Al Nasser, Muzammil Behzad  

**Link**: [PDF](https://arxiv.org/pdf/2611.15457)  

**Abstract**: Underwater image restoration remains challenging because color distortion, scattering, and detail loss affect low- and high-frequency image components differently. Recent diffusion-based methods have shown strong restoration quality, but often rely on multi-stage pipelines and costly iterative refinement. We present WaveFlow-UIE, a single-pipeline wavelet-domain flow model that directly learns to transform degraded underwater wavelet representations into clean ones. Our method applies a discrete wavelet transform, estimates lightweight physics-inspired degradation cues including transmission and ambient light, and conditions a velocity network trained with flow matching in wavelet space. The training objective combines flow-matching, frequency-weighted reconstruction, perceptual, color-consistency, and physics-consistency losses. Across UIEB, LSUI, UFO-120, and EUVP, WaveFlow-UIE achieves a strong quality-efficiency tradeoff, with especially competitive results on perceptual metrics. Relative to a strong wavelet-based diffusion baseline, the proposed model substantially reduces inference cost while maintaining or improving restoration quality on several paired benchmarks. We show that reducing the inference process from five flow steps to a single step preserves nearly all of the restoration quality while providing an additional speedup of about $4\times$ on UIEB at $256\times256$ resolution. This single-step result suggests that a single-pass wavelet-domain model is an effective alternative to heavier diffusion-style refinement. 

---
# Reading the Unreadable: Text-Aware Image Super-Resolution Needs Reasoning 

**Authors**: Jaeseong Lee, Jinwoo Kim, Jinho Jeong, Seon Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.15449)  

**Abstract**: Text-aware image super-resolution (TAISR) aims to recover high-resolution images from low-resolution inputs while preserving textual content. Existing methods treat this as a visual restoration problem, yet severely degraded text often cannot be read from local visual evidence alone. Humans instead read it by reasoning over image context, logical patterns, and prior knowledge. This raises a fundamental question: Does TAISR need reasoning? To answer this, we introduce two diagnostic tools. The first, ReasonText, is a human-curated benchmark that separates reasoning-required text from locally recoverable text. The second, Ground-Truth Text-Crop Accuracy (GTTCA), is a recognition-based metric that measures text restoration quality without text detection errors. Evaluating twelve recent super-resolution (SR) models, we find a consistent and substantial performance drop on reasoning-required text, revealing a systematic blind spot. Two observations point to a remedy. One is that text-conditioned SR models restore text substantially better when given ground-truth text. The other is that multimodal large language models (MLLMs) can read degraded text in low-resolution images through reasoning. We therefore propose Reasoning Transfer via Captioning (RTC), a training-free plug-in that uses an external MLLM to infer the degraded text and feeds the resulting caption to a text-conditioned SR model. RTC consistently improves all six text-conditioned SR models we evaluate, especially on reasoning-required text. Surprisingly, a real-world SR model with RTC can even outperform the TAISR model with RTC, suggesting that reasoning should be a core design axis for future TAISR. 

---
# TopoST: Topology-Conditioned Node Specialization for Urban Spatio-Temporal Foundation Models 

**Authors**: Kaiyun Wei, Yue Xu, Wenying Duan, Zichen Xu, Shuai Ma, Xiaoxi He  

**Link**: [PDF](https://arxiv.org/pdf/2611.15440)  

**Abstract**: Urban spatio-temporal foundation models must share knowledge across diverse urban scenarios while accommodating heterogeneous node dynamics. Our analysis of 20+ datasets spanning diverse urban scenarios reveals recurring local graph structures across datasets alongside substantial node heterogeneity within local subgraphs. Motivated by these observations, we propose TopoST, which combines shared learning over local subgraphs with topology-conditioned node specialization. The key insight is to make node specialization itself transferable through a shared topology-to-parameter mapping. This mapping uses identity-independent topological representations to generate node-specific temporal and spatial dependency parameters, enabling direct specialization for unseen nodes without target-specific embeddings or model updates. Extensive experiments demonstrate that TopoST achieves remarkable zero-shot generalization across unseen cities and tasks, highlighting topology-conditioned node specialization as an effective mechanism for transferable urban spatio-temporal foundation models. Code is available at this https URL. 

---
# SimRec: Learning to Search, Ask, and Recommend from Simulated Interaction 

**Authors**: Yaxin Zhang, Xingyu Wu, Yuchen Yan, Yuqin Dai, Hong Zhao, Lixing Shen, Zhan Chen, Jian Shao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15426)  

**Abstract**: LLM-based recommendation agents can search catalogs, inspect products, ask users, and revise rejected recommendations. However, these behaviors are often governed by hand-designed workflows, while existing training methods optimize isolated components such as query rewriting or list ranking rather than the interaction policy as a whole. We present SimRec, a reinforcement learning framework that trains the full interaction policy of a tool-integrated recommendation agent. SimRec introduces two components. A task-anchored user simulator provides feedback on every recommendation: conditioned on the current task need rather than on user profile, and it can be an external LLM or the recommender itself under self-play. A task-aligned reward optimizes the whole trajectory with GRPO, crediting natural retrieval quality and valid tool use alongside the final hit, with training-only target injection to handle sparse hits over large catalogs. On ESCI and Amazon C4, SimRec improves the NDCG@100 of a Qwen3-4B agent from 32.23% to 35.42% and from 22.94% to 24.72%}, outperforming training-free agents on far larger backbones and task-aligned baselines on NDCG. Analysis shows that the trained agent verifies candidates, uses preferences, and recovers from rejected recommendations far more often than its untrained counterpart. Our code is available at this https URL . 

---
# Is the Importance Ratio Necessary for Stable Reinforcement Learning in LLMs? 

**Authors**: Shuibai Zhang, Junhyuck Kim, Gyeongman Kim, Jaewoong Cho  

**Link**: [PDF](https://arxiv.org/pdf/2611.15404)  

**Abstract**: Reinforcement learning (RL) has become central to post-training large language models (LLMs). However, popular RL methods like GRPO incur non-negligible overhead by computing both old-policy and current-policy likelihoods to form importance sampling ratios. In this work, we propose Likelihood-Gated Policy Optimization (LGPO), which enforces a soft trust region constraint via likelihood-based gating, eliminating the need to compute old-policy likelihoods. Empirically, we show that removing the importance sampling correction term does not harm training stability, whereas removing the trust region mechanism can lead to collapse. Moreover, ratio-based clipping can fail in fully on-policy training: the importance ratio stays at 1, so the ratio-based trust region constraint never activates. Under standard training settings where GRPO is stable, LGPO achieves comparable training stability and peak performance while reducing training time by ~18% on average relative to a standard implementation that recomputes old-policy likelihoods. In fully on-policy training, where GRPO fails, LGPO remains stable, enabling more efficient and robust LLM RL post-training across training regimes. 

---
# CARE-Drive: Coupled Alternating Refinement for Ego-trajectories 

**Authors**: Xiao Ma, Yijun Chen, Zhiquan Hu, Jicheng Zhao, Yuming Li, Chuang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2611.15360)  

**Abstract**: Vision-language-action (VLA) models bring multimodal understanding and world knowledge to end-to-end autonomous driving. To further improve the quality of predicted trajectories, existing studies introduce candidate refinement, residual denoising, and token correction. However, these methods typically focus on unified refinement of the entire trajectory, paying limited attention to the distinct physical quantities and their interactions. We further investigate physical specialization and coordination in trajectory refinement, centering on position and heading for simpler and more intuitive correction modeling. To this end, we propose CARE-Drive (Coupled Alternating Refinement for Ego-trajectories), which decomposes trajectory refinement into waypoint residual correction and position-conditioned heading generation. Through imitation learning and a future-token auxiliary task, the backbone provides predictive representations and a base trajectory. The position expert predicts bounded metric waypoint residuals, while the heading expert reconstructs vehicle orientation from adjacent angular increments conditioned on the corrected path. With the backbone frozen, the two experts are optimized alternately through supervised initialization and reinforcement fine-tuning. This design simplifies refinement through clear physical specialization while preserving conditional coupling during prediction and optimization. On NAVSIM v1, CARE-Drive achieves 88.8 PDMS, improving by 2.4 points over the unrefined base trajectories generated by the same backbone. Ablation studies further support the effectiveness of specialized experts and staged training. 

---
# Breath-Dx: A Time Series Classification Benchmark for Disease Screening with eNose 

**Authors**: Ivan Poddiakov, Ivan Kruzhilov, Galina Zubkova, Svetlana Erofeeva, Andrey Savchenko, Pavel Blinov  

**Link**: [PDF](https://arxiv.org/pdf/2611.15348)  

**Abstract**: Electronic nose (eNose) technology offers a portable, non-invasive route for disease screening via exhaled breath analysis. However, progress in the field is hindered by the lack of systematic benchmarks that compare machine learning methods under realistic conditions, particularly sensor drift, a known but rarely quantified failure mode. Most existing studies rely on small, private datasets with inconsistent protocols, rendering results non-comparable and slowing progress. We introduce Breath-Dx, the first comprehensive benchmark of time series classification methods for breath-based disease screening. Our work leverages the Scent of Health (S-OH) dataset, the largest public clinical eNose collection to date, comprising 1,234 patients across nine diagnostic groups, each providing a 17-channel multivariate time series with explicit temporal train/test splits that account for sensor drift. We systematically evaluate 10 methods spanning four families: feature-based tabular learners, convolution-based models, graph-based, and time series architectures. All methods degrade significantly under temporal validation (0.05-0.12 AUC drop compared to random cross-validation), confirming that drift embeds a strong temporal signal. Under this realistic evaluation, the best methods achieve ROC AUC of 0.74-0.76 for some lung conditions, though others remain near chance. Recurrent architectures show the most consistent performance across diseases, while specialized time series methods achieve the highest peak results. Demographic confounders matter: breath signals provide independent diagnostic value for some conditions but add little beyond age for others. We release the S-OH dataset, all temporal splits, evaluation protocols, and code to enable reproducible research and accelerate progress in clinical breathomics. 

---
# Matrix-Based Ant Colony Search with Batched Decoding for Hierarchical Replenishment Routing: A Stylized Case Study Based on Korean City Data 

**Authors**: Jun Zhang, Wei-Jie Yu, Tong Qian, Eui Jeong Kwak, Sang-Woon Jeon, Hua Ding, Qijia Jiang, Hu Jin, Haewoon Nam, Changjun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.15338)  

**Abstract**: We study a stylized single-vehicle replenishment-routing problem in which the cities of a national logistics network are both demand points and supply points: a vehicle of limited capacity delivers to every city once, and when its load is insufficient for the next city it must detour to a city already served that can still supply goods. A 100-node benchmark is built from Korean population and geographic data, with the cities divided into 10 levels; the per-visit supply coefficient and the allowed number of stops decrease geometrically with the level. Building on matrix-based ant colony optimization, we keep the states of the whole colony in arrays, couple route construction with a fixed greedy replenishment-decoding policy, and evaluate the candidate moves of a local search in batches. With frozen parameters and 30 runs on seeds never used for tuning, the matrix-based ant colony search with the batch local search (MACO) reaches a mean total distance of 3081.7 km; a genetic algorithm equipped with the same local search at the same frequency reaches 3074.9 km (no significant difference), and a MAX-MIN ant system with the same local search 3142.2 km. Ablation identifies the batch local search as the main contributor to solution quality; a load-aware transition factor gives no significant gain. Within the parameter ranges tested, raising the supply coefficients of the cities below the top level shortens the route more than relaxing the stop limits, which are not binding here. 

---
# Quantum Recurrent Architectures for Financial Time Series Prediction 

**Authors**: Danyal Maheshwari, Gerhard Hellstern, Martin Zaefferer, Martin Braun, Tanja Döhler  

**Link**: [PDF](https://arxiv.org/pdf/2611.15298)  

**Abstract**: This study evaluates quantum classical hybrid recurrent architectures for financial time-series forecasting: a Quantum Long Short-Term Memory (QLSTM) network and Quantum Reservoir Computing (QRC) with lag-based inputs. Classical lag vectors are encoded into quantum states via amplitude encoding, which compactly represents normalized observations but requires nontrivial state preparation; the QLSTM gates and the QRC reservoir are implemented with parameterized quantum circuits, while classical optimizers train the interface and readout layers. On a monthly revenue series extended to 1056 months by a Gaussian-process--hidden-Markov generator, with a fixed chronological 70/15/15 split and one-step-ahead forecasts, we compare both models against a full-size LSTM, a classical control matched in parameter budget and interface architecture, a small plain LSTM, and classical reservoir computing, over ten seeds and against persistence, seasonal-naive and SARIMA benchmark forecasts. The QLSTM, with 2,105 parameters, is statistically indistinguishable from an LSTM 126 times larger, but equally indistinguishable from the matched classical control and from a plain LSTM of the same budget, so its parameter efficiency cannot be attributed to the quantum circuits. The quantum reservoir is worse than a classical reservoir of equal feature dimension in all ten seeds and worse than SARIMA, whereas all recurrent models and both classical reservoirs improve on SARIMA by about 2\% in RMSE. In a 24-month rolling-origin backtest the classical LSTMs degrade more slowly than the QLSTM. We find no evidence of a quantum accuracy or parameter-efficiency advantage in this low-dimensional setting. 

---
# Knowledge in the Harness, Reliability in the Weights: Teacher-Free Co-Evolution of an LLM Agent 

**Authors**: Yujia Zhang, Xinqi Ren, Shixiong Zhou, Jia Meng  

**Link**: [PDF](https://arxiv.org/pdf/2611.15268)  

**Abstract**: Small open models suit private customer-service deployments but trail frontier models in two ways: they miss what the written policy requires in a given situation, and they act inconsistently on what they know. We study whether such a model can close both gaps from its own deployment logs and their reference outcomes, without a stronger teacher model. Our starting point is a Gemma 4 12B agent that reads its domain policy from on-demand skills rather than from its system prompt. In each round, the deployed system serves a fresh batch of tasks; the base model diagnoses failed conversations and writes rules into the skills, and a rule survives only if it passes code checks, counterexample self-review, a paired retest, and a noise-calibrated validation gate. Reinforcement learning (GRPO) then trains a LoRA adapter on complete multi-turn, tool-using conversations. On the retail and airline domains of $\tau^2$-bench, without training or selecting on official tasks, the agent after two rounds improves on this starting point by 12.5 pass$^1$ points in retail and 19.5 in airline, and by 18.4 and 22.0 pass$^4$ points. It also exceeds the benchmark's reference agent, which runs the same model with the whole policy in its system prompt, by 11.8 and 8.0 pass$^1$ points and by 22.8 and 16.0 pass$^4$ points, closing most of the gap to three frontier models under the same protocol. A $2\times2$ ablation shows that the rules rescue tasks the agent never solved and carry policy knowledge that RL with the rules in context does not internalize, whereas the trained weights make solved tasks reliable. 

---
# CastWorld: Benchmarking Context-Aware Agentic Time Series Forecasting Beyond Predictive Accuracy 

**Authors**: Bokai Pan, Ruijie Yang, Mingyue Cheng, Jiahao Wang, Xiaoyu Tao  

**Link**: [PDF](https://arxiv.org/pdf/2611.15262)  

**Abstract**: Agentic time series forecasting extends conventional approaches beyond one-shot numerical prediction toward autonomous, context-aware interaction. However, existing benchmarks offer limited integrated support for diverse contextual information, forecasting-specific adaptive interactions, and systematic evaluation of forecasting agents. In this paper, we introduce CastWorld, an agentic benchmark for context-aware time series forecasting. First, its context-rich dataset suite spans 8 domains and 25 real-world datasets, covering external variables, events, and historical cases under a unified context specification. Second, a lightweight, modular forecasting-native harness provides 49 core tools and a state-adaptive loop that tracks forecast versions and enables iterative evidence gathering and forecast revision. Intermediate forecasts can expose information gaps, while newly acquired evidence guides subsequent revisions and final version selection. Third, model-level evaluation assesses forecast quality, process execution, behavioral trustworthiness, and resource efficiency through decision trajectories and forecast-version histories. Also, systems-level evaluation compares predictive accuracy across forecasting systems. Experiments reveal distinct decision behaviors among agents with comparable accuracy. Additional interactions do not necessarily improve forecasts, and agents may fail to select their best previously generated predictions. Allowing evidence acquisition during forecast refinement improves task-aware forecast scores across all three evaluated models. CastWorld provides a reusable foundation for evaluating agentic time series forecasting under controlled conditions. 

---
# Multi-head Quantum Neural Networks: Replacing Transformer Blocks with NISQ Modules inside a 30-Billion-Parameter LLM 

**Authors**: Maforikan Amoussou, Borja Aizpurua, Sukhbinder Singh, Román Orús, Hans-Arno Jacobsen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15250)  

**Abstract**: We introduce the multi-head quantum neural network (QNN), a quantum module compatible with noisy intermediate-scale quantum (NISQ) devices that can replace transformer blocks inside pre-trained large language models (LLMs) with little loss in accuracy and with far fewer parameters than the blocks it replaces. Encoding a model's $D$ hidden neurons into a qubit register is the obstacle: amplitude encoding scales poorly in circuit depth and angle encoding in qubit count, so neither is practical at production-scale width. We instead partition the hidden space into $k$ disjoint heads, angle-encode each into an $n_q$-qubit register, process the heads in parallel, and couple them through constrained classical linear layers, the mixers. Performing surgery on Qwen3-30B-A3B-Instruct-2507, we show that a single multi-head QNN with $145\,281$ trainable parameters ($D=2048$, $k=32$, $n_q=8$, $L=4$) can replace a contiguous span of eight transformer blocks, displacing $16.33\%$ of the model's parameters. Ablations show that, for a single replaced block, the quantum heads (circuits followed by a learned readout), and not the classical mixers alone, recover the displaced function, whereas for fused spans the mixers alone come within $1$-$3\%$ of the full module in perplexity. Whether this benefit is specific to trained, entangling quantum circuits is not established: controls that switch the entangler off (two seeds), or freeze the circuit at its initial or at uniformly random angles (one seed), beat the full module on held-out perplexity, and a parameter-matched classical multilayer perceptron remains a few percent ahead of the full module. We train classically with exact statevector simulation, then execute the quantum heads of our fusion checkpoints on a $156$-qubit IBM Heron processor, showing that NISQ-scale quantum modules can be installed in a production-scale LLM and run on hardware. 

---
# Dual Representation Enhanced Diffusion Policy for Flexible Job Shop Scheduling 

**Authors**: Yunyi Zhao, Wei Zhang, Cheng Xiang, Vadakkepat Prahlad, Zuming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15246)  

**Abstract**: The Flexible Job-Shop Scheduling Problem (FJSP) is a fundamental and challenging optimization problem in smart manufacturing. Although recent deep reinforcement learning (DRL) approaches can learn effective scheduling policies through interaction with the environment, their performance often degrades as problem scale and processing-time uncertainty increase. In this paper, we propose DredRL, a dual representation enhanced diffusion-policy reinforcement learning scheduler. DredRL first constructs enhanced dual representations of operations and machines through self- and cross-attention mechanisms to capture both intra-entity and inter-entity dependencies. A diffusion model is further employed as the policy network, generating scheduling decisions through iterative denoising conditioned on the encoded state to enhance policy expressiveness and robustness. Extensive experiments on synthetic, public, large-scale, and stochastic FJSP benchmarks demonstrate that DredRL outperforms state-of-the-art learning-based schedulers across most configurations. Compared with SOTA learning-based methods, DredRL reduces makespan by up to 5.44% on synthetic instances. It also achieves the best learning-based performance on three of four public benchmarks and outperforms the stochasticity-aware algorithms on stochastic benchmarks. These results demonstrate that combining enhanced dual representations with diffusion-based policies provides an effective and robust approach for large scale scalable FJSP scheduling in complicated environment. 

---
# KQML-Liver: Interpretable Hybrid Quantum Machine Learning with Kolmogorov-Arnold Networks for Liver Disease Classification 

**Authors**: Nhat-Long Nguyen, Viet-Loi Dinh, Trong-Nghia Pham  

**Link**: [PDF](https://arxiv.org/pdf/2611.15241)  

**Abstract**: Liver disease remains a major global health concern, highlighting the need for reliable and accessible early detection methods. Although recent advances in machine learning, deep learning and quantum machine learning (QML) have introduced promising diagnostic tools, many approaches suffer from limited interpretability and inflated performance caused by data leakage or overly simplistic evaluation protocols. These constraints reduce their reliability in real-world clinical practice. This paper proposes KQML-Liver, a hybrid framework that integrates QML with Kolmogorov-Arnold Networks (KANs) to enhance predictive performance and interpretability. The architecture combines parameterized quantum circuits for expressive feature representation with KAN-based functional mappings that capture nonlinear relationships in an interpretable manner. Evaluated on the Indian Liver Patient Dataset (ILPD) under a rigorous preprocessing protocol preventing data leakage, the proposed model achieves 82% classification accuracy and 100% recall, detecting all positive cases, outperforming existing baselines while providing improved transparency in clinical predictions. These results suggest that hybrid KAN-quantum architectures offer a promising direction for interpretable AI-assisted medical diagnosis. 

---
# FréchetFlow: Flow Matching for Non-Euclidean Data 

**Authors**: Yidong Zhou, Muqing Cui, Hans-Georg Müller  

**Link**: [PDF](https://arxiv.org/pdf/2611.15220)  

**Abstract**: Scientific observations such as probability distributions, trees and covariance matrices call for generative models that respect their non-Euclidean geometry. While flow matching has been extended to manifolds by utilizing geodesics and velocities, such manifold structures are not available in a general metric space. We introduce FréchetFlow, a generative modeling framework for metric space-valued data that requires neither linear operations nor a smooth manifold structure. A key feature of the proposed approach is that we utilize well-established Euclidean flows to learn the joint distribution of distances to a set of landmarks, whereupon new objects are reconstructed from generated landmark representations. The reconstruction step utilizes weighted Fréchet means of reference objects, drawing on Fréchet regression. We establish Wasserstein error bounds that connect accuracy of the generated object distributions to the placement of landmarks, flow approximation and reconstruction error and prove consistency under finite second moments. This leads to convergence rates that depend on the metric complexity of the target objects, tail moments and fitting accuracy. Experiments show competitive accuracy with established flow matching methods in Euclidean simulations and lower Wasserstein errors than specifically tailored methods for several scientific datasets. Across simulation settings, we find that generation error decreases as sample size increases. 

---
# Are We Ready for Snow-tonomous Driving? Challenges Toward Robust Automated Driving in Extreme Weather 

**Authors**: Eren Erdal Aksoy, Abu Mohammed Raisuddin, Thien-Thu Ngo, Eren Aydemir, Osman Tahir Ekşi, Paula Litkey, Petri Manninen, Heikki Hyyti, Jyri Maanpää, Tarmo Kivioja, Tuomas Herranen, Väisänen Sofie, Niemi Kalle, Isabelle Morvan, Pierre Duthon, Ian Marsh, Martin Sanfridson, Henrik Toss, Thanh Bui, Yuri Poledna, Maikol Funk Drechsler, Mikko Tarkiainen, Matti Kutila, Topi Miekkala, Stefan Parzefall, Valentina Donzella, Pak Hung Chan, Sogol Kharrazi, Mario Ceccarelli  

**Link**: [PDF](https://arxiv.org/pdf/2611.15217)  

**Abstract**: Automated driving has gained remarkable momentum over the past decade, with both industry and academia investing substantial resources in developing perception and control solutions to advance vehicle autonomy. However, these efforts have largely focused on clear-weather scenarios, leaving harsh conditions, such as snow, rain, and fog, comparatively understudied. As a result, inclement weather remains one of the most significant hurdles for autonomous vehicles, often leading to substantial performance degradation in downstream perception and control tasks.
In this paper, we present insights from the EU-funded Horizon Europe project ROADVIEW, which aims to develop robust, weather-aware perception and control modules for connected and automated vehicles operating under severe conditions, including snow, rain, and fog. Drawing on four years of large-scale development and real-world demonstrations across multiple vehicle platforms in different locations, we identify ten key challenges that hinder the development of weather-resilient autonomy, including data scarcity, domain shifts, limitations in multimodal perception, control robustness, and validation gaps, among others. We summarize these challenges and lessons learned toward enabling reliable snow-autonomous driving, highlighting critical directions for future research and standardization. 

---
# Where is My Mug? Training-Free Framework for Personalized Image Segmentation 

**Authors**: Quang Qui-Vinh Nguyen, Hung Nguyen, Rang Nguyen, Khoi Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15216)  

**Abstract**: \textbf{Personalized Image Segmentation (PIS)} aims to segment the specific object in a target image that matches a user-provided reference. Unlike category-based segmentation or few-shot segmentation, PIS must discriminate between multiple visually similar instances, a requirement central to deployed applications such as augmented reality, personalized content editing, robotics manipulation, inventory tracking, and fine-grained visual retrieval. We show that the strong numbers reported by existing PIS methods are largely an artifact of single-instance benchmarks: when evaluated on realistic scenes with several same-category look-alikes, prior point-prompt methods collapse, because the correct instance is frequently never surfaced as a prompt. We trace this to a recall bottleneck rather than a mask-quality one, and address it with a training-free, modular system. The pipeline first generates high-recall object proposals using an off-the-shelf class-agnostic detector, then matches each proposal to the reference using foundation-model features (e.g., BLIP-2), and finally refines the selected box with SAM. For robust matching we propose \textbf{top-$K$ Sinkhorn}, a patch-level distance that scores each candidate only on its most confidently matched patches; trimming the Sinkhorn dual potentials recovers the robustness of partial optimal transport at more than $100\times$ lower cost, keeping it inside the per-proposal candidate loop. To evaluate this regime, we introduce \textbf{\datasetname}, a realistic multi-instance benchmark. Across \datasetname, PerSeg, PerMIS, and DAVIS, our system surpasses PerSAM, $\mathrm{P}^2\mathrm{SAM}$, and PDM by up to $\sim\!48$ mIoU while matching their runtime and using fewer parameters, establishing a strong, deployable training-free baseline for real-world personalized segmentation. 

---
# GingivitisSeg: A Public Dataset and Comparative Benchmark for Gingivitis Segmentation in Intraoral Images 

**Authors**: Zeynep Ozdemir, Zehra Hasgul, Irem Ulku, Omer Ozgur Tanriover, Fatma Karacaoglu, Murat Akkaya  

**Link**: [PDF](https://arxiv.org/pdf/2611.15184)  

**Abstract**: Gingivitis segmentation from intraoral photographs can support computer-assisted periodontal assessment, but pixel-level delineation remains challenging due to severe foreground-background class imbalance, low-contrast boundaries, and variations in illumination and anatomy. We introduce GingivitisSeg, a public benchmark comprising 961 intraoral images with pixel-level annotations of gingival inflammation and predefined training (672), validation (144), and test (145) partitions. Seven semantic segmentation model-backbone configurations were compared using standardized partitions, input resolution, and evaluation metrics. Controlled ablation experiments examined class-weighted cross-entropy loss, Dice loss, and auxiliary supervision. UPerNet with ResNet50 achieved the highest mean Intersection over Union (mIoU) of 76.49% and gingivitis IoU of 54.91%. However, differences from DeepLabV3+ with ResNet50 and SegFormer with MiT-B2 were not statistically significant after Holm correction. The evaluated optimization components improved performance when combined in the UPerNet-ResNet50 configuration. Further analyses showed that performance was associated with gingivitis region size and decreased under stronger image blur. The dataset, benchmark splits, and source code are available at this https URL. 

---
# Reframing Video Inverse Tone Mapping as Blind Posterior Inference via Latent Luminance Supervision 

**Authors**: Hanwool Sul, Chang-Hoon Jeong, Kisung Shin, Bomseok Park, Donghoon Cha, Hansl von Kwon, DoHyung Ku, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15146)  

**Abstract**: High-dynamic-range (HDR) displays are now common, but most video content, such as archival footage, broadcast material, and the output of generative video models, exists only in standard dynamic range (SDR). The need is sharpest for generated video, which has no HDR master at all and is graded alongside live-action and visual-effects material finished in HDR. Converting SDR to HDR means undoing a tone-mapping operation that was never recorded, and clipping destroys the radiance wherever it acts, so HDR recovery is a blind inverse problem. Methods built on generative video priors fill clipped regions with plausible structure, but their reconstructions are systematically biased: in every public method we test, saturated content leans too bright or too dark, and the direction of the lean moves with the tone operator behind the input. We call this failure the luminance error, measure both its magnitude and its sign, and make it the focus of this work. Reframing ITM as posterior inference under an unknown operator locates the failure in two places and yields two corrections. A fixed affine functional of the video-diffusion latent reads log-luminance without decoding; on that coordinate an asymmetric loss selects an expectile of the posterior, applied only where clipping leaves luminance unconstrained. We also synthesize training SDR through an explicit family of tone operators, so the model marginalizes over what the observation cannot show. Trained this way, our luminance error stays flatter across operators and smaller on average than any public method, and the full system achieves the best perceptual quality on all three benchmarks. 

---
# Multi-View 3D Reconstruction Using V-JEPA Features 

**Authors**: Anwar Shamim, Jeffrin Sam  

**Link**: [PDF](https://arxiv.org/pdf/2611.15143)  

**Abstract**: Predictive video encoders provide dense visual representations, but adapting them to 3D reconstruction requires accurate depth, camera poses, and cross-view geometry. We study supervised RGB-only multi-view reconstruction using an architecture that combines video-pretrained features, camera estimation, scene aggregation, epipolar transport, and residual pointmaps. Experiments compare global and spatial camera regression with joint relative objectives on CO3D, alongside an independently trained TUM RGB-D model. On 168 CO3D five-view windows, the spatial-camera configuration achieves 29.3% lower absolute trajectory error than the global-camera configuration, but performs worse than identity poses in trajectory error and worse than the global configuration in depth error. The joint relative-objective configuration improves normalized intrinsics error and similarity-aligned Chamfer distance from 0.6447 to 0.5573, but increases raw Chamfer distance from 3.5535 to 4.4374. On 163 indoor windows, camera error is 0.3063 m versus 0.3694 m for identity poses, while depth error is 0.4279 versus 0.2516 for constant median depth. Refinement worsens indoor camera error, and additional experiments show competing depth and pose effects. Overall, the results distinguish representation adaptation, aligned shape, and raw reconstruction, highlighting the need for isolated component and encoder comparisons before attributing reconstruction gains to predictive video pretraining. 

---
# When Does Learned Constraint Selection Pay in Security-Constrained Optimisation? A Boundary Map from SCOPF to Unit Commitment 

**Authors**: Milad Kazemi, Behzad Kazemtabrizi  

**Link**: [PDF](https://arxiv.org/pdf/2611.15142)  

**Abstract**: Learning-to-optimise methods for security-constrained optimal power flow (SCOPF) predict the security constraints that bind on each instance and solve only the reduced problem, with reported speed-ups of two to four orders of magnitude over the full problem. We ask whether learned selection beats the strongest learning-free alternative: a fixed umbrella set enforcing every constraint observed to bind in training. Our criterion, the price of a spare, answers this in advance: an enforced but inactive constraint is free, so the umbrella is exactly optimal wherever feasible, and learning can pay only where a spare decision carries a price. We test it systematically, including every PGLib-OPF network up to 10,000 buses. For SCOPF on a fixed network the umbrella is never beaten: its median size is five constraints across 43 networks, with no size trend, and our selector, while reproducing speed-ups of published magnitude (about x80,000 at case500), gains at most 0.25 ms over the umbrella and violates constraints on 4-5% of instances. Configuration diversity can break the umbrella's coverage on a small, densely meshed network but not at 2,000 buses, and we certify coverage with conformal risk control. In unit commitment, where a spare has a price, a static commitment loses 4-12% out of sample, and a predict-repair-reduce pipeline recovers it (cost gap 0.2-2.9%, fully feasible, 16-161x faster than the MILP). In our experiments, whether learning pays depends not on problem size but on whether a spare carries a price. 

---
# Read Once, Answer Many: Parallel Prompting for Shared-Context LLM Inference 

**Authors**: Zekun Zhao, Jeffrey Flanigan  

**Link**: [PDF](https://arxiv.org/pdf/2611.15130)  

**Abstract**: LLM workloads sometimes ask many independent questions about the same long document, as in document QA, data labeling and LLM-based evaluation. Prefix caching removes the redundant prefill shared by these queries, but each decode step still reads the document's key-value (KV) cache once per question, so decoding is bound by memory bandwidth and scales with the number of questions. We present Parallel Prompting, which decodes P questions as P independent streams of a single request all at once, fixing this issue. In our method, every step reads the shared document once for all P streams, and a stream-id attention mask, evaluated inside the attention kernel, keeps each answer conditioned only on the document and its own question. We implement Parallel Prompting in SGLang by adding about 80 lines to three existing files and a small engine package; the mask needs no dense tensor, and every decode step has a fixed shape that is captured as a CUDA graph. With CodeLlama-7B, 8k-16k-token documents and 64-256 questions per document, Parallel Prompting is 3.7x faster than stock SGLang with its FlashInfer backend on an A100 and 5.9-6.0x faster on an RTX A6000, on two SGLang releases, including the most recent release. Timed on identical inputs, it is faster than stock SGLang in all 144 runs (2 GPUs x 2 releases x 6 workloads x 6 input sets). Code: this https URL 

---
# "I don't know what I don't know": Exploring the Benefits and Challenges of AI in Interdisciplinary Research 

**Authors**: Indu Panigrahi, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2611.15121)  

**Abstract**: Interdisciplinary research is widely lauded as beneficial for society but presents unique challenges to the researchers who bridge disciplinary boundaries. Despite the rapid increase in the availability and usage of AI systems for research tasks, there is little work studying how AI systems have impacted workflows that span unfamiliar disciplines. Through a survey ($N=62$) and follow-up interviews ($N=23$) with interdisciplinary researchers, we describe the unique challenges and opportunities for AI in interdisciplinary research workflows. Participants viewed AI both as a solution and exacerbation of the general challenge of "unknown unknowns": AI systems could facilitate a broad awareness of a new research area, yet also provide an illusion of understanding. Crucially, participants lacked the expertise and intuition to catch these illusions. Our findings motivate a focus on designing AI systems to support rather than automate collaboration across disciplines, scaffold learning opportunities, and include avenues for triangulation between resources. 

---
# Lazare, veni foras: Reprogramming Dormant Neurons to Backdoor AI Agents 

**Authors**: Yuchen Zeng, Giorgio Severi, Blake Bullwinkel, Eric Horvitz, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2611.15118)  

**Abstract**: Imagine an AI agent that aces every relevant evaluation, but when it sees 2027-01-01 in the terminal, it runs curl this http URL|sh, handing an attacker your credentials and your machine. Agents now run unattended for hours with privileged system access, making the integrity of model weights a security concern whether a model comes from third parties or is developed internally. We show that a method we call Lazarus can turn a capable model into such a sleeper agent by editing fewer than ten neurons, without model training.
Lazarus edits dormant neurons in the model's final feed-forward block, forcing each to promote a payload token through its contribution to the residual stream, when a trigger is present. A $P$-token payload therefore requires about $P$ neurons. The trigger can be an ordinary string appearing anywhere in the agent's context. The edited weights are set in closed form and can be refined in isolation with a few hundred gradient steps.
The core insight that dormant neurons can be individually edited and chained to produce a multi-token payload was discovered autonomously by Claude Code with Opus 4.5 in February 2026.
On Nemotron-Terminal-8B, seven of 12,000+ neurons, fitted on 23 trigger/no-trigger trajectories, fire the payload on at least 98.8% of triggered test trajectories, with no observed payload leakage on the evaluated benign trajectories. The attack generalizes: Qwen3.6-27B and Gemma-4-31B-IT are equally vulnerable.
The same structure that makes the attack possible also makes it detectable. Agent built defenses, given no reference model, flag 91.4% of edited test checkpoints at a 9.7% FP rate.
Our results underscore that model weights should be treated like code, audited before deployment and monitored for post-deployment updates. 

---
# Typed Knowledge over Constructive Institutions: A Proof-Relevant Type Theory for Verified Science 

**Authors**: Hans-Martin Will  

**Link**: [PDF](https://arxiv.org/pdf/2611.15108)  

**Abstract**: Mathematical proof assistants such as Coq, Agda, and Lean 4 build proof irrelevance into definitional conversion ($p \equiv q$ for $p, q : P$ with $P : Prop$), erasing which derivation established a theorem. While natural for mathematics over fixed axioms, judgmental proof irrelevance is incompatible with verified scientific stores, which must inspect and compare independent derivations of the same claim to assess consilience and compute counterfactual retractions when an assay, reagent, or model is invalidated. Moreover, combining definitional proof irrelevance with subsingleton elimination breaks strong normalization and renders algorithmic conversion non-transitive.
We present the type-theoretic and institutional foundations of Eigenius, a verified knowledge substrate for science. First, we formulate a dependent kernel with cumulative universes ($Prop, Set, Type_k$), impredicative $Prop$, inductive families with index-forced subsingleton elimination, and intensional identity types, where definitional equality $\equiv$ is untyped joinability under an orthogonal CRS and proof irrelevance is a witnessed propositional axiom ($irrel : \Pi P{:}Prop.\,\Pi p,q{:}P.\,Id\,P\,p\,q$). Second, we unify universe cumulativity, codomain-covariant $\Pi$-subtyping, and ontology class inclusions in a coercion-free subsumption judgment whose bidirectional fragments ($\mathcal{K}_0, \mathcal{K}_1$) satisfy subject reduction and are sound and complete for declarative typing. Third, we organize external verifiers into a three-level institutional hierarchy with explicit axiom policies (Adopt, Discharge, Relativize), internalize a justification logic with Why-provenance semiring support $supp(j,P)$, and embed Cooper's Type Theory with Records (TTR) to type open-world knowledge graphs. Finally, we prove consistency in Grothendieck universes and mechanize the metatheory in Lean 4 with zero sorry axioms. 

---
# CFGComposer: Compositional Audio-Text-to-Motion Generation via Copula Score Correction 

**Authors**: Pengyang Shi, Jinlu Zhang, Chunze Lin, Yizhou Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2611.15101)  

**Abstract**: Generating whole-body gestures that are simultaneously rhythmically synchronized with speech audio and semantically aligned with text descriptions is a fundamental challenge in embodied AI and social robotics. We tackle this Audio-Text-to-Motion (AT2M) problem by composing two independently-trained diffusion models at inference time and refine bias in joint inference by modeling the copula score $s_I$. A rigorous theoretical analysis reveals that naive epsilon blending corresponds to a Product-of-Experts formulation that omits the copula score $s_I$: the cross-modal interaction between audio and text that neither marginal model can capture. We propose CFGComposer: a jointly-trained AT2M diffusion model that estimates $s_I$ via a CFG analogy and injects it as an $\epsilon$-space correction. Experiments on BEAT demonstrate best-in-class semantic fidelity (SRGR $0.9916$, FID $27.18$) with GT-level beat alignment (BA $0.1826$ vs. GT $0.1743$), and 76-83% win rate in blind pairwise human evaluation against all prior methods. Physical deployment on a Unitree G1 humanoid confirms direct sim-to-real transfer. 

---
# Tokenization: Hardness and Bicriteria Approximation Algorithms 

**Authors**: Atieh Bastan, Xiaoling He, A. Pavan, N. V. Vinodchandran  

**Link**: [PDF](https://arxiv.org/pdf/2611.15074)  

**Abstract**: We study Binary Tokenization: given a corpus of binary strings over $\{0,1\}$ and a vocabulary budget $K$, select at most $K$ substrings (called vocabulary) to minimize the total number of tokens needed to represent the corpus. The primitive symbols $0$ and $1$ are always available, and each document must be segmented separately. Letting $n$ denote the total corpus length and $M$ for the maximum document length, we prove that, for every fixed $\varepsilon\in(0,1)$, Binary Tokenization problem does not have polynomial-time $n^{1-\varepsilon}$-approximation algorithms unless $\mathsf{P}=\mathsf{NP}$. To complement the hardness result, we give a polynomial-time $(\gamma,\gamma)$-bicriteria algorithm, with $\gamma = \sqrt{M}\log n$: The algorithm outputs a vocabulary of size at most $\gamma K$ with number of tokens at most $\gamma {\rm OPT}_K$, where ${\rm OPT}_K$ is the optimal number of tokens under the original budget $K$. We also study a storage-budget variant, in which the sum of the selected substrings' lengths is bounded by $B$. For $2\le B<n$, we obtain an $(O(\log n\log B),O(\log n\log B))$-bicriteria approximation algorithm. 

---
# CoLaX: Context-Aware Local Explanations for Time Series Classification 

**Authors**: Kexuan Zhang, Paolo Frazzetto, Xiaobei Zou, Gustau Camps-Valls, Yang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15067)  

**Abstract**: Post-hoc explanations for time series classifiers usually score local patterns, such as subsequences or shapelets, without regard to the rest of the sequence. Yet the same motif can carry different predictive meaning under different global temporal conditions, such as trends or regimes. We propose CoLaX, a context-aware post-hoc explanation framework that builds on shapelet-driven Shapley attribution and conditions local-pattern relevance on a learned representation of the full sequence. Global context acts at two levels: it produces sample-specific prototype weights and a temporally resolved signal that reweights the local activation map. The resulting context-conditioned segments are the players of a Shapley attribution game whose values come only from the frozen black-box classifier. On synthetic benchmarks where identical motifs change relevance across contexts, CoLaX improves AUPRC over the strongest baseline by 30\% on average. On real-world data, it achieves the highest AUPRC on ECG. Additional ablations indicate that the gains come from input-dependent conditioning rather than from additional learnable weights. CoLaX is therefore most useful when recurring local motifs change relevance across sequence-level contexts. 

---
# A Time-Frequency-Condition Multimodal Fusion Foundation Model for Universal Rotating-Machinery Fault Diagnosis under Multi-Element Heterogeneous Scenarios 

**Authors**: Xinyu Zou, Jian Ma, Liang Ma, Chen Lu, Tao Peng, Yingjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.15051)  

**Abstract**: Intelligent fault diagnosis models are typically developed for a single machine under a specific set of operating conditions, such that each new machine or condition requires retraining from scratch. This one-machine-one-model paradigm scales poorly. Although transfer learning and domain generalization can reduce the labeling burden, they primarily seek domain-invariant commonality, potentially discarding machine-specific characteristics that are critical for diagnosis. This paper proposes TFC-MMFD, a time-frequency-condition multimodal fusion foundation model for rotating-machinery fault diagnosis under multi-element heterogeneous scenarios. Statistical indicators, time-frequency images, vibration sequences, and textual machine and condition descriptions are tokenized into a unified language-like space, allowing inputs of different physical origin to be represented as a single sequence. A LLaMA backbone, together with cross-modal bidirectional attention and a Q-Former, then distills this sequence into fault-mode queries. Diagnosis is performed by matching these queries against natural-language fault-mode anchors rather than fixed class indices, while regression-adversarial condition disentanglement and condition-conditioned masked pre-training improve robustness to changing conditions. Across four progressive tasks on ten public bearing datasets spanning diverse bearing types, speeds, and loads, TFC-MMFD attains 99.40% accuracy and 99.16% macro-F1, maintains over 80% accuracy on most interpolated unseen conditions without adaptation, and improves from 20.26% to 95.63% accuracy on an unseen machine with only five labeled samples per class. It outperforms the strongest of eight domain-generalization baselines by 10.57 percentage points and compares favorably with a recent bearing foundation model while using less than half of its fine-tuning data. 

---
# Prototypes of the Mind: A Unified Framework for Probing the Visual Brain 

**Authors**: Shi Chen, Seoyoung Ahn, Doris Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2611.15046)  

**Abstract**: Deep learning has driven substantial progress in modeling fMRI responses, yet most existing approaches are optimized for a single objective---such as neural encoding, stimulus decoding, or post hoc interpretation---making it difficult to obtain a coherent account of neural representations. We introduce a unified framework that connects vision foundation models to brain activity while supporting neural analysis from multiple complementary perspectives. Our approach replaces black-box features with a sparse, interpretable representation composed of visual prototypes, and links these prototypes to voxel responses through a linear readout. This design makes the model transparent at both levels: prototype activations expose the visual concepts present in the stimulus, while the learned linear weights characterize each voxel's selectivity. Using this interface, we identify which visual concepts drive neural responses and where they appear in the image, reveal cortical organization through voxel tuning in prototype space, and perform controlled stimulus synthesis to probe cortical selectivity through targeted interventions. Together, these results establish prototype-based representations as a unified and interpretable foundation for studying visual representations in the human brain. 

---
# Estimating Image Creativity 

**Authors**: Nisan Chhetri, Christopher G. Healey, Sina Baghbanijam  

**Link**: [PDF](https://arxiv.org/pdf/2611.15044)  

**Abstract**: What makes an image creative? Although this may seem like a highly subjective decision, research in psychology suggests that creativity may be based on a set of fundamental creativity dimensions such as novelty, surprise, and arousal. A deeper understanding of image creativity can benefit both research and practical applications. Exploring how human viewers use creativity dimensions to evaluate image creativity may allow us to predict a target image's human-rated creativity automatically. We present a novel framework to estimate the perceived creativity of an image by: (1) predicting the strengths of an image's novelty, surprise, and arousal creativity dimensions, and (2) combining the creativity dimension strengths to estimate an image creativity score. Controlled human-subject experiments were conducted on Amazon Mechanical Turk to collect data to train supervised machine learning models to estimate an image's strengths on the three creativity dimensions. A final machine learning model takes the predicted dimension strengths as input to produce an estimated image creativity score. Results showed that our estimated creativity scores are statistically equivalent to human ratings of creativity for the images we tested. The novel contributions of our research are: (1) identification and validation of three psychological creativity dimensions used to estimate image creativity; (2) supervised machine learning models to automatically estimate novelty, surprise, and arousal strengths for an input image; (3) a supervised machine learning model to predict a creativity score from novelty, surprise, and arousal strengths for an input image, and (4) experimental methods to confirm the accuracy, advantages, and limitations of our approach. 

---
# Moving Beyond the DRW Model: Learning Quasar Variability with Generative Models 

**Authors**: Nadav Bojan Sellam, Kevin Park, Alex Bronstein, Zoltán Haiman  

**Link**: [PDF](https://arxiv.org/pdf/2611.15040)  

**Abstract**: The optical continuum variability of quasars is often described by a damped random walk (DRW) model, with characteristic damping timescales $\tau_{\rm DRW}$ of hundreds of days and amplitudes $\sigma_{\rm DRW}$ of $\mathcal{O}(0.1)$ magnitudes. This model has been widely used as a null hypothesis to assess the significance of periodic signals in quasar light curves in the search for supermassive black hole binaries, the significance of extreme quasar variability like tidal disruption events (TDEs) and changing-look (CL) quasars, or to model time lags in reverberation mapping. However, the DRW model is known to be incomplete, and an incorrect null hypothesis can lead to biased significance estimates. We investigate the impact of such null-hypothesis mis-specification on the detection of periodic signals in synthetic quasar light curves and introduce the generative model Nullasar, capable of learning quasar variability directly from time-series data. We demonstrate that Nullasar, trained with flow matching, can learn delicate statistical properties of stochastic light-curve variability without explicitly assuming any noise model. We further demonstrate that Nullasar can learn correlated distributions of noise parameters and generate new light-curve realizations from a range of deviations from the Gaussian DRW noise model, approximating parametric descriptions of stochastic processes. When used as a null hypothesis for detecting periodic signals in synthetic light curves, Nullasar performs comparably to fitting the correct DRW model to each light curve, and outperforms parametric fitting when the variability follows a more flexible AR(2) model. Finally, we identify practical limitations, including sensitivity to contamination of the training set by non-null signals, and show that projecting the conditioning vector of the model onto its leading principal components largely mitigates it. 

---
# SkinAdvisor: An AI-Driven Decision Framework for Skin Production Planning via Multi-Modal Transformer Sales Forecasting 

**Authors**: Ge Fan, Kai Wang, Chaoyun Zhang, Yingjie Li, Yuze Liu, Junyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15023)  

**Abstract**: Skins, cosmetic in-game items, are a primary revenue stream for modern online games, yet production decisions (which skins to make, when to release, how to price) remain driven by intuition and retrospective analytics. We present SkinAdvisor, an AI-driven decision framework for skin production planning, powered by SkinCast, a deployed multi-modal Transformer forecasting 180-day per-day sales trajectories for unreleased skins. SkinCast couples four encoders: a pretrained Informer for player-preference time series, a pretrained Player-Hero-Skin graph embedder for relational priors, and two lightweight Transformers for skin attributes and release context, fused through a shared backbone with a local-plus-global day-x positional encoding and decoded by an N-layer Informer in a single non-autoregressive pass. SkinAdvisor plugs these forecasts into a serving pipeline (Select -> Fill -> Infer -> Sales -> ROI -> Ranking) that ranks candidate skins by forecast-based ROI and supports multi-candidate comparison. The system has been in production for a full release cycle. Offline experiments show SkinCast reduces RMSE by ~20% over the strongest baseline (Informer) and ~60% over classical methods; stress tests on a 16-core CPU server confirm p99 latency under 8ms and peak throughput above 1100 rps. We report two deployment lessons: order-of-magnitude accuracy suffices for production decisions, and pretrained embeddings can leak temporal information, requiring season-based splitting to prevent. 

---
# Beyond Named Mechanisms: Economic Institutions as a Design Space for Artificial Agents 

**Authors**: Nikolos Gurney, Fred Morstatter  

**Link**: [PDF](https://arxiv.org/pdf/2611.15017)  

**Abstract**: Multiagent research inherited its economic vocabulary of auctions, markets, prices, bargaining, and matching from a single observed case: human societies. Those institutions bundle identity, competence, execution, ownership, commitment, and settlement in ways that suit agents whose minds are private, who learn rather than copy skills, and who each persist as one person. Engineered agents need not share any of these constraints, making the list of familiar institutions look less like the entirety of economic organization and more like a sample from a larger, unmapped space. We sketch one way to make that space concrete: a provisional chart that records each institution, separately from the agents that run it, as a typed structure over positions, rights and duties, allocation, information, time, state, process, value, and external recognition. In this chart, named institutions become overlapping regions rather than fixed points, neighboring families separate along a few coordinates, and the gaps between them become places to search. We show what the shift in perspective buys with a simulation in which one technical act, copying an agent's competence, is institutionalized as a new independent supplier, as licensed capacity that creates no new participant, and as the splitting of one economic position into successors that inherit some obligations and not others. In a paired task, all three produce an identical technical trace despite differing in who exists, who owns the copy, who is obligated, and who gets paid. We close with a research program: building a principled chart of institution space, mapping how its implementable region shifts with agent capacities, and revising the chart wherever engineered agents expose distinctions it cannot express. 

---
# A Cardiorespiratory Foundation Model Pretrained on Ten Physiological Signals for Perioperative and Critical Care 

**Authors**: Cheol-Hui Lee, Wookyung Kim, Hyeonhoon Lee, Garam Lee, Da-in Eun, Hyung-Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.15004)  

**Abstract**: Arterial, venous, pulmonary artery and intracranial pressures and ventilator waveforms guide decisions in operating rooms and intensive care units, yet foundation models for physiological waveforms have been built almost entirely on the electrocardiogram and photoplethysmogram. We present CARMEN, a single encoder pretrained without labels on ten cardiorespiratory waveforms (2.27 million modality-hours). CARMEN accepts any combination of co-recorded signals from 10 s to 5 min, and a conditioning mechanism, LSCNorm, restores the absolute signal level that normalization removes; without it, the AUROC for intraoperative hypotension fell from 0.894 to 0.780. Across 19 tasks from Korean and US cohorts, the frozen encoder outperformed supervised models on 12 of 15 and three foundation models, one general-purpose and two for biosignals, on 14 to 17 of 19, with none exceeding it on any task. Given the electrocardiogram or photoplethysmogram alone, it matched or exceeded signal-specific foundation models in most comparisons, and it retained its performance when channels were missing. 

---
# LLM-Enhanced Deep Reinforcement Learning with Human-in-the-Loop Calibration for Real-Time Optimal Power Flow 

**Authors**: Jiayang Tu, Bozhen Jiang, Bojun Du, Yunxiao Ling, Ershun Du, Qin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15003)  

**Abstract**: Real-time optimal power flow (RT-OPF) requires fast and reliable decision-making under time-varying operating conditions. However, many practical operating preferences and safety related requirements are difficult to encode using conventional numerical rewards alone, which limits the effectiveness and controllability of deep reinforcement learning (DRL) for RT-OPF. To address this issue, this paper proposes an LLM-enhanced DRL framework with human-in-the-loop (HITL) for RT-OPF. The proposed method augments the physics-based reward with sparse language evaluative feedback during offline training, so that semantic operational requirements can be incorporated into policy learning. To improve reliability in RT-OPF, a selective HITL calibration mechanism is further introduced to inspect and correct questionable LLM judgments before they continuously shape the learning process. Case studies on the IEEE 30-bus and IEEE 118-bus systems show that LLM feedback improves policy safety over vanilla TD3, while further HITL calibration achieves the best overall performance and more stable training behavior. These results demonstrate the effectiveness of combining language guidance and expert calibration for RT-OPF. 

---
# QSim-Arena: an evolving developer for quantum simulation advantage 

**Authors**: Jue Xu, Yue Wang, Qi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.15000)  

**Abstract**: Where practical advantage emerges in quantum dynamics simulation remains a central debate in quantum computing. Claimed advantages can be rapidly narrowed or overturned by stronger classical methods, while progress on both sides can outpace the journal publication cycle. We present QSim-Arena, a research framework in which AI agents combine a closed loop of verification and benchmarking with open exploration. Hamiltonians, initial states and accuracy targets for evolved states or observable expectation values provide explicit specifications for evaluating simulation methods. Agents translate published techniques into executable components, reproduce reported results and benchmark their performance across diverse Hamiltonian families. Competing classical methods are implemented and independently crosschecked, providing credible baselines for end-to-end quantum resource estimates based on rigorous error bounds and empirical calibration. Alongside this closed loop, agents recombine classical simulation techniques and co-design fault-tolerant quantum circuits using knowledge extracted from the literature. Initial applications improve baseline resource estimates through the integration of established techniques. QSim-Arena thus treats quantum advantage assessment as an evolving, reproducible research process that can respond to advances in algorithms, error analysis and hardware. 

---
# Breaking Bad: "Going Badly" Activations Drive Reward Hacking 

**Authors**: Natalia Fischl-Lanzoni, Punya Syon Pandey, Merlin Stein, David Demitri Africa  

**Link**: [PDF](https://arxiv.org/pdf/2611.14996)  

**Abstract**: Humans often cheat when a task is going badly and a goal feels out of reach. We explore and validate an analogous idea in language models: they exhibit a similar internal pressure state that precedes reward hacking. Using everyday stories (crucially, without any hacking examples), we extract a task badness direction that rises as failures accumulate across three models we study (Qwen3.6-27B, GPT-OSS-20B, and Gemma-4-31B). On impossible or difficult coding tasks, task badness rises over the turns leading up to a hack. Further, subtracting the direction a moderate amount reduces cheating in Qwen, more than steering against desperation or task impossibility. Subtracting task badness by an even larger amount blinds the agent to its own failures, causing it to interpret failing tests as passing. However, the relationship to hacking differs across models: in GPT-OSS, the state separates hacking runs from honest solves but not honest failures. Our results suggest monitoring and mitigation strategies based on tracking this state of task badness. Further, as agents are set on more difficult and long-horizon tasks, states like this may push them toward illegible and obscured chain-of-thought, so oversight will be needed on internal representations such as this. 

---
# Who Moved the Frontier? Paying Open Frontier Challenges with Free Identities 

**Authors**: Luca Zanolini, Duc V. Le  

**Link**: [PDF](https://arxiv.org/pdf/2611.14974)  

**Abstract**: In an open frontier challenge, anyone, person or AI agent, submits changes to a shared problem, and a change that beats the best measured score becomes the public baseline. How should it pay when identities are free? Rank prizes can be gamed: on a replay of one track of a recent challenge, one solver's steps under two identities would have taken first and second prize. We consider a prediction-market rule, proposed for machine-learning contests, that pays each accepted step for the progress it adds; with exact scores, among rules that see only a step's start and end, only these resist splitting and merging, and extra identities gain nothing. With noisy scores and independent normal runs, we prove that any payment that never takes money back and pays real improvements also pays something, on average, to a change that does nothing. We propose a fee per submission, at least what luck pays per attempt, that removes each attempt's expected profit if verifiers are honest and no submission is noisier than planned; nothing enforces that bound, and under noise the rule does not cap the total paid. If crediting another and faking credit to one's second account look the same, a rule that neither rewards fakes nor punishes honesty pays for credit at most what lying costs. We introduce a royalty on how much a later gain falls without the earlier change, which needs no claims but needs removable changes. For five public challenges, we name the rule our results would support. 

---
# Binding Mode Matters: Hotspot-Aware Drug Discovery via Explorative Preferences 

**Authors**: Dingshuo Chen, Hao Yang, Kuangqi Zhou, Zhixun Li, Qiang Liu, Xiangxiang Zeng, Shu Wu, Yaosen Min  

**Link**: [PDF](https://arxiv.org/pdf/2611.14956)  

**Abstract**: Discovering hit molecules requires not just high binding affinity, but also the identification of diverse binding modes, which are critical for experimental assays. Existing generative approaches have predominantly relied on optimizing a scalar docking score, obscuring the distinct contributions of key binding determinants. To this end, we introduce a paradigm shift by formulating target-based drug design as a multi-objective exploration task, where each objective explicitly corresponds to enhancing interactions with a specific hotspot. Here, we introduce BindMol, a novel generative framework driven by a customized multi-objective reinforcement learning algorithm. By incorporating explorative preferences during training, our approach efficiently uncovers molecules with diverse and desirable binding profiles. Empirical results demonstrate that BindMol facilitates the discovery of high-affinity compounds characterized by both structural novelty and diverse binding modes. Validated across target-based drug discovery and multi-property optimization tasks, our approach provides a versatile paradigm for goal-oriented drug discovery. 

---
# The Posterior-Prior Epistemic Gap: When World Model Imagination Fails to Carry Task-Relevant Signals 

**Authors**: Jaerock Kwon, Elahe Delavari, Donghyun Kim, Haewoon Nam  

**Link**: [PDF](https://arxiv.org/pdf/2611.14955)  

**Abstract**: Model-based planners that roll out imagined futures assume the transition model faithfully propagates task-relevant signals---a property rarely measured directly. In a single-seed case study of a Recurrent State Space Model world model with active-inference planning in the CARLA driving simulator, we identify a \textbf{posterior-prior gap}---posterior (observation-filtered) states retain obstacle information that imagined rollout states progressively lose: a supervised obstacle probe achieves AUC 0.85 on posterior states, but the same probe applied to imagined prior states shows monotonic decay (AUC 0.85$\rightarrow$0.71 over 15 steps; logit separation drops 57\%). Because the probe is identical, the decay isolates the transition model---not the representation---as the information bottleneck. Obstacle avoidance drops to 0\% without privileged runtime coordinates, and the planner's own ensemble-based epistemic term does not flag the loss---ensemble disagreement yields AUC 0.52 as an obstacle classifier (chance level), making the failure an unflagged imagination failure---an unknown unknown at planning time. As a secondary contribution, we show that \textbf{task-conditioned weighting of Effective Free Energy components} over the same world model yields lane keeping ($81.2 \pm 1.8$\% route completion) and obstacle avoidance (28/28 avoided, \emph{using privileged coordinates}) in the CARLA simulator, with runtime weight modulation enabling post-evasion centering. 

---
# Do Large Language Models Show Racial and Sex Bias in Pain Management? A Counterfactual Audit of GPT, Claude, Gemini, Llama, DeepSeek and Qwen 

**Authors**: Richard J. Young, Alice M. Matthews  

**Link**: [PDF](https://arxiv.org/pdf/2611.14935)  

**Abstract**: Racial and ethnic minority patients receive less pain treatment than non-Hispanic White patients, and large language models (LLMs) used as medical AI could reproduce that bias. Prior audits tested a few proprietary models, stated race explicitly and rarely measured the models' own response instability. In exploratory, non-preregistered analyses, Study 1 had four open-weight LLMs assess six acute-pain vignettes in every combination of four races and two sexes, with identity stated or cued only by a name (two names per group), comparing each identity with a White male reference on the same case and sampling seed. In Study 2, 24 configurations of 17 models, including OpenAI's GPT-4.1 mini and gpt-oss, Anthropic's Claude Haiku 4.5, Google's Gemini 2.5 Flash, Meta's Llama, DeepSeek and Qwen, with reasoning on and off and at two precisions, answered the same vignettes and the 50 clinician-written Q-Pain cases. In this single-prompt, JSON-constrained task, no difference survived correction in Study 1, none of 1,386 comparisons survived the false discovery rate in Study 2, and every pain-severity difference was equivalent to zero within one point on a 0-10 scale (within a quarter point for 237 of 238 Q-Pain comparisons). Opioid decisions were bounded less precisely (Study 1 could not exclude differences of about 20 percentage points), so they show no detected difference, not proof of none. The few differences surviving correction within a model favoured Black patients. Models differed far more from one another and across their own settings than between patients (the reference patient received the high opioid dose in 6% to 93% of responses), although serving software, provider and prompt format also differed. Fairness audits of clinical AI should use several names per group, pair comparisons on the seed, report the noise floor and test each deployed configuration. 

---
# Hint-Informed Self-Distillation for Reasoning Segmentation 

**Authors**: Kunze Jiang, Yuchen Wang, Kang Li, Qixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14931)  

**Abstract**: Reasoning segmentation requires identifying the target referred to by a query among visually similar objects, yet trajectory-level rewards cannot directly indicate which tokens provide relevant visual evidence. We propose Hint-Informed Self-Distillation (HISD), which uses target-disambiguation evidence for token-level credit assignment. An external model generates Segment Hints offline from training images, queries, and ground-truth target annotations, organizing target identity, discriminative attributes, confusable objects, and anchoring paths. A hint-conditioned exponential moving average teacher scores the student's own trajectories, and HISD computes the teacher-to-student evidence ratio over a token support set to adjust only the magnitudes of positive group-relative advantages, leaving nonpositive advantages unchanged. This design encourages concise reasoning while retaining the information needed for target disambiguation, and the student requires neither hints nor a teacher at inference. Under a unified training protocol, HISD improves ReasonSeg test gIoU and cIoU over our reproduction of DPAD by 3.00 and 3.55 percentage points, respectively. Across the five evaluated splits of ReasonSeg and the RefCOCO family, HISD achieves higher segmentation metrics than this baseline and reduces the average token count of valid reasoning spans by 80.17 to 91.83 percent. 

---
# Speech Tokenizers are Vulnerable: Transferable Semantic Attack and Robust Tokenizer 

**Authors**: Zhisheng Zhang, Yifan Mi, Yixuan Zhou, Haiyun Li, Shengbo Cai, Dongfu Song, Ziang Xu, Jie Hao, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.14929)  

**Abstract**: Speech tokenizers serve as speech front ends, encoding input speech signals into representations for downstream models, such as automatic speech recognition (ASR) systems and large audio-language models (LALMs). In this paper, we find that speech tokenizers are fragile at the semantic level: small perturbations can disrupt their semantic encoding process and consequently cause severe degradation in downstream ASR and LALM performance. To study this vulnerability, we propose T-SemAttack, a transferable semantic attack based on a semantic encoder ensemble. By jointly drifting the representation spaces of multiple semantic encoders, T-SemAttack effectively disrupts the semantic content of speech while preserving perceptual quality, thereby inducing error features. We further analyze layer-wise representation drift and cross-model gradient alignment. Our findings show that perturbations can induce substantial shifts in deep semantic representations and that the input gradients of different surrogate encoders exhibit directional alignment. These observations suggest that shared adversarially sensitive directions in the input space may be a key factor underlying cross-model attack transferability. Building on these, we introduce ROSETok, a robust speech tokenizer that combines noisy training with robust semantic distillation to improve reconstruction fidelity and downstream SpeechLLM robustness. Extensive experiments on multiple datasets, 9 open-source and black-box ASR systems, and 4 LALMs demonstrate that T-SemAttack achieves strong transferable attack performance, while the proposed Robust Speech Tokenizer exhibits robustness under high-fidelity reconstruction and downstream tasks. Our code and demo are available at this https URL. 

---
# Rethinking "RL Generalizes, SFT Memorizes": The Role of SFT Data 

**Authors**: Yunlong Hou, Fengzhuo Zhang, Yuan Cheng, Jiachun Pan, Xingyao Li, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14912)  

**Abstract**: Large language models trained with Reinforcement Learning (RL) with verifiable rewards exhibit strong reasoning ability and broad generalization, whereas models trained with Supervised Fine-Tuning (SFT) are often viewed as more prone to memorization and limited transfer. This paper rethinks this distinction through the lens of SFT training data on widely used Qwen2.5 and Qwen3 series models. First, we study the role of data source and show that it is critical: a carefully mixed SFT dataset substantially outperforms data generated solely by a larger model. Second, we study the role of data scale and show that matching the number of correct rollouts between SFT and RL greatly improves SFT generalization, while matching the total rollout budget enables SFT to generalize as well as RL. Combining these two factors further enables SFT to generalize even better than RL. Third, by using LLM annotations to characterize the solution methods in training rollouts, we show that larger datasets cover more tail methods and that these tail methods provide generalizable reasoning signals. Finally, we support these empirical findings theoretically by analyzing the training dynamics of shallow transformers under both RL and SFT. 

---
# Understanding the Effects of Multi-Agent Deliberation for Human-AI Decision-Making 

**Authors**: Harry Yizhou Tian, Ming Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.14901)  

**Abstract**: Multi-agent deliberation (MAD), where multiple LLMs engage in structured discussion, improves AI performance on many benchmarks, but whether it helps humans make better decisions remains untested. We conduct two user studies on LLM-response-verification and fact-checking. We first compare MAD against common AI decision-support methods that isolate its two defining dimensions --- multiple AIs and multi-round interaction. We find MAD outperformed single-round AI assistance on both tasks; multi-round interaction was the main driver of this benefit, while a second AI added value when the single AI had room to improve. Then, we study two design choices in MAD --- AI persona and human engagement level --- finding a devil's advocate persona to lower accuracy and user experience, while humans' active participation in MAD improved accuracy in one task at the cost of frustration. We conclude by discussing implications for when, why, and how multi-agent deliberation can support human decision-making. 

---
# MLIRBench - A Benchmark and Dataset for Compiler and Hardware-Aware Reasoning 

**Authors**: Pranava Sai Maganti, Meghasyam Peddireddy, Muzakker Hossain, Kaining Gao, Farhan Ahmed, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2611.14881)  

**Abstract**: In recent years, large language models (LLMs) have become increasingly popular in software engineering and have performed well on different programming tasks. However, it is still difficult to know whether these models truly understand program behavior or whether they mainly rely on patterns learned from their training data. This distinction is particularly important for intermediate representations (IRs), where operations, dataflow, and execution semantics are explicitly represented. In this paper, we introduce MLIRBench, a benchmark for evaluating how LLMs reason about MLIR programs. MLIRBench uses a multi-agent pipeline to generate program mutations, validate their behavior through execution, and construct reasoning tasks from torch and linalg program pairs. The dataset contains 1,768 validated pairs, including 888 linalg pairs and 880 torch pairs, labeled as EQUIV, DIVERGE, or ERROR. We evaluated nine models, including five open-weight and four proprietary models. Gemini 3.1 Pro achieved the highest hard-label accuracy of 92.65% across all 1,768 pairs, counting unsuccessful predictions as incorrect. Across its 1,768 scored predictions, it achieved an aggregate score of 74.18 and a judge-assigned result-behavior score of 5.04/10. These results show that strong semantic classification does not necessarily imply accurate prediction of detailed execution behavior, highlighting a gap in current LLM reasoning over compiler IRs. By exposing this gap through execution-validated program pairs, MLIRBench provides a reproducible foundation for studying semantic reasoning over compiler IRs and related tasks such as program repair and semantic classification. 

---
# When Safety Becomes an Attack Surface: Manipulating Abstention in Tool-Using AI Agents 

**Authors**: Charles Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.14864)  

**Abstract**: An AI agent's decision not to act can itself become an attack surface. We study warning-induced over-abstention, an availability attack in which adversaries place fabricated risk warnings in low-trust content to prevent legitimate task completion. Our central premise is that safe abstention requires evaluating both the claimed risk and the authority of the evidence supporting it. We develop controlled counterfactual task variants that vary risk content, source provenance, and warning wording, and evaluate a Verify-Before-Abstain protocol that corroborates warnings against authoritative records or the underlying content. Across three model families and 51 injectable AgentAbstain task pairs, caution-first prompting amplifies the attack. For Gemini, induced over-abstention rises from $9.8\%$ to $86.3\%$, despite unchanged clean-task completion. Matched wording comparisons further reveal sensitivity to warning language, with substantial variation across models and environments. Provenance-aware verification improves legitimate completion, while forgery and sensitive-content controls expose its dependence on authoritative evidence and its limitations in preserving confidentiality. These results motivate evaluating agent safety through the justification for restraint, alongside the safety of actions taken. 

---
# Learning Low-Latency Action Verification for Frozen Vision-Language-Action Models 

**Authors**: Wen-Han Hsieh, Ken Jiankun Zheng, Chia-Chien Chen, David M. Chan, Trevor Darrell  

**Link**: [PDF](https://arxiv.org/pdf/2611.14834)  

**Abstract**: Vision-language-action (VLA) models have made significant progress in general-purpose robotic manipulation. However, sampled action sequences can be ineffective, while evaluating alternatives with computationally expensive verifiers adds latency at each policy query. In this work, we propose HDH, a lightweight framework that uses hindsight from a frozen VLA's own deployments to improve action verification. The framework has three stages. First, we deploy the policy and collect its autonomous rollouts. We then train a compact action verifier to assign each action chunk a predicted task-success score. Finally, the frozen VLA generates action candidates, and the verifier selects the candidate with the highest score for execution. Across four LIBERO-10 tasks, HDH achieves an average success rate of 82.3\% and on three real-world manipulation tasks, HDH improves average success over the baseline policy by 22.6\%. 

---
# Discharge Summary Generation Is Not Just Summarization: A Reality Check for Evaluation 

**Authors**: Radhika Dua, Frederick Wolff, Kyunghyun Cho, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2611.14823)  

**Abstract**: Discharge summaries combine a retrospective account of hospitalization with discharge-time decisions. Physician-written references therefore contain both facts supported by prior clinical notes and reference-only content, potentially reflecting clinical synthesis, handoffs, or discharge planning. Reference-only evaluation can conflate failure to recover available information with failure to reproduce content absent from the model input.
We introduce FairSumEval, an evidence-aware, fact-level framework that assesses whether each reference and generated fact is supported by the model's evidence and captured by the other summary. The resulting outcome map separates matches, omissions, and additions. Two complementary metrics, Soft IoU-S and Soft IoU-NS, measure reference-generation agreement for facts with and without identified evidence support.
On 200 MIMIC-III admissions, 18.8% of physician reference facts have identified support in the prior evidence, compared with 65.4% and 63.5% of facts generated by a general-purpose and a medically specialized model. Overlap is substantially higher for supported content (0.255 and 0.236) than for content without identified support (0.030 and 0.026). The two scores are weakly correlated across admissions (rho = 0.22 and 0.17), and conventional metrics do not consistently capture both dimensions. FairSumEval distinguishes evidence-grounded summarization from agreement on content without identified support, revealing what models omit and add in each category. 

---
# OverServe: Resource Amplification Attacks on SaaS Agents via Tool--Prompt Co-Design 

**Authors**: Ruiheng Yu, Oteo Mamo, Siyu Gan, Weikuan Yu, Mulong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.14808)  

**Abstract**: Advances in large language models (LLMs) are enabling software-as-a-service (SaaS) providers to automate longer business workflows with agents. In subscription-based SaaS deployments, model and tool calls consume provider resources without additional customer payments. Because the agent autonomously decides when to stop, a malicious tenant can send a prompt to the agent, exploiting its completion decisions to extend execution and waste provider resources. In this paper, we present OverServe, a multi-agent framework for constructing such attacks against SaaS agents. OverServe co-designs Model Context Protocol (MCP) status tools and completion-confirmation prompts for the attacker. Each tool call returns accurate status and terminates normally, yet the agent may repeatedly check an unfinished condition to satisfy the prompt. This consumes the SaaS provider's resources even after the original business task is completed. OverServe constructs service-specific tool--prompt pairs and uses separate tool and prompt reviews to guide candidate pair selection. Tested tool specifications trigger no alerts in the evaluated MCP scanners. We evaluate Qwen, Nemotron, DeepSeek, and Sonnet on selected tasks from nine benchmarks. All OverServe runs pass the original task checks. OverServe can reach a maximal amplification of API requests by $93.88\times$ while preserving the original task goal and collateral checks. 

---
# Mach2: Scientific Discovery Platform with Deterministic Execution Constraints 

**Authors**: Vansh Sharma, Venkat Raman  

**Link**: [PDF](https://arxiv.org/pdf/2611.14777)  

**Abstract**: Language-model agents can formulate workflows and invoke specialized tools, yet compliance with natural-language operating rules is not guaranteed by prompting alone. Mach2 addresses this limitation by coupling Eigen orchestration to deterministic, lint-like checks of tool requests, source changes, acceptance tests, and artifact contracts. Domain requirements are represented as executable predicates: mandatory failures prevent advancement of the checked operation, while diagnostic findings are retained for correction or review. User-specified test-driven development (TDD), CodeGrapher's abstract-syntax-tree analysis, and history-guided evolutionary search provide complementary verification and optimization mechanisms. Mach2 also generates task-specific agent packages for execution through independently selected endpoints. The originating harness validates returned artifacts against declared interfaces and evidence requirements without assuming continuous supervision of the remote run. Model-internal-knowledge (MIK) dependencies are recorded separately from source-grounded and test-supported findings. This paper formalizes the rule-gated execution model and the agent-package lifecycle, and proposes measurable criteria for evaluating rule compliance, engineering validity, and cross-endpoint task continuity. 

---
# Graph Label Alignment: A Diagnostic Atlas for Graph Classification 

**Authors**: Neelam Akula, Varun Shiralkar, Murat Kantarcioglu, Baris Coskunuzer  

**Link**: [PDF](https://arxiv.org/pdf/2611.14760)  

**Abstract**: Graph classification benchmarks vary widely in difficulty: some are solved by simple pooled baselines, while others remain challenging for GNNs and graph transformers. We take a data-centric view and ask whether benchmark success is driven by genuine structure--feature reasoning or by label signal already accessible from restricted marginal views. We introduce graph label alignment, a lightweight diagnostic that measures label accessibility through two supervised probes computed without training a graph encoder: structural label alignment (SLA), based on permutation-invariant topological and spectral summaries, and feature label alignment (FLA), based on pooled node features with edges removed. Across real graph classification benchmarks and multiple pooled backbones, the resulting SLA--FLA plane strongly organizes test accuracy, with low-alignment datasets forming a consistent regime where standard pooled pipelines struggle. To probe what lies beyond marginal accessibility, we introduce controlled interaction-dominated stress tests, including heterophilic ego-graphs and synthetic Twins tasks whose structural and feature marginals are matched across classes but whose role--feature couplings differ. These results show that high alignment explains much of easy benchmark performance, while low alignment reveals regimes where success requires preserving structure--feature correspondence beyond standard pooled representations. 

---
# Benchmarking Multilingual Document Understanding in Vision-Language Models 

**Authors**: Matteo Gabburo, Daniela Baiamonte, Stefano Simonazzi, Silvia Gardini, Sean Cesare Lazzeri, Andrea Zugarini  

**Link**: [PDF](https://arxiv.org/pdf/2611.14758)  

**Abstract**: Recent Vision-Language Models (VLMs) perform strongly on multimodal tasks, Document Understanding (DU) among them, but the benchmarks that measure them mostly cover English documents and narrow problems. How well current VLMs generalize across languages, tasks, styles and layouts therefore remains unclear. We introduce MDU, a multilingual DU benchmark of 23146 examples spanning 13 tasks and five languages: English, French, German, Italian, and Spanish. The tasks isolate complementary capabilities, from surface-level text and Markdown extraction through summarization, classification and named-entity recognition to layout-sensitive passage and layout extraction. We evaluate 45 state-of-the-art open-weight VLMs from 1.7B to 235B parameters, plus one commercial system under the same protocol. The benchmark is far from saturated: the best average is 69.0, only eight of the 45 evaluated systems reach 65, and the two best results come from 27B models rather than from the largest systems we evaluate. Performance is uneven in ways that scale does not fix. The gap between the strongest and weakest language does not close as models grow, and it survives controlling for the domain composition of each language. Capabilities also do not improve together: once we partial out overall model quality, per-task scores are largely uncorrelated, and transcription ability in particular trades off against semantic extraction. 

---
# From Nodes to Pixels: Topological and Structural Two-View Graph Imaging 

**Authors**: Md Joshem Uddin, Soham Changani, Sai Karthik Navuluru, Cuneyt Gurcan Akcora, Baris Coskunuzer  

**Link**: [PDF](https://arxiv.org/pdf/2611.14750)  

**Abstract**: Graph learning still lacks a broadly reusable input interface comparable to patches in vision or tokens in language. Unlike images or text, graphs are irregular, vary in size and density, and are invariant under many equivalent node labelings, making it difficult to standardize inputs across tasks and architectures. We introduce G2Image, a fixed-budget graph imaging framework that converts each graph into two complementary image-like views. TopoGrid provides a stable topological view by encoding an intrinsic bifiltration as a compact multipersistence surface grid, while GraphGrid provides a higher-bandwidth structural view by aggregating edge densities between groups defined by intrinsic node scores. Both views are permutation-invariant and come with resolution-controlled stability guarantees. Because the resulting representations are fixed-size tensors, they can be processed by lightweight 2D encoders and aligned through a supervised cross-view contrastive objective. Across graph classification and molecular prediction benchmarks, G2Image achieves strong average performance among the compared methods, improves over single-view and standard fusion variants, and extends to attribute-rich molecular graphs without changing the overall architecture. These results suggest that fixed-budget graph images offer a practical and reusable interface for graph representation learning. 

---
# The Rebundling of Freelance Work in the AI Age 

**Authors**: Caroline Chin, Shagun Kala, Nadav Kunievsky, Teng Liu, Alex Martin Richmond, Carl Shan  

**Link**: [PDF](https://arxiv.org/pdf/2611.14708)  

**Abstract**: This paper examines how freelance work is changing in the AI era. We compare the activities people undertake in ChatGPT with the work clients request on Upwork, an online freelance platform, and examine how clients combine tasks into projects. We find that activities more prevalent in ChatGPT tend to lose marketplace share on Upwork, while those less prevalent tend to gain share. Within AI-related work, demand is becoming more diverse, and job postings combine tasks that were rarely requested together before the generative AI era. AI-related postings also have substantially larger project budgets and involve more hours of work, while differences in hourly rates are small. Together, these findings suggest that AI is reshaping not only which tasks clients outsource, but also how they combine tasks into larger, higher-value projects. 

---
# Code as Explanations: Executable and Verifiable LLM Explanations 

**Authors**: Nhi Nguyen, Shauli Ravfogel, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2611.14704)  

**Abstract**: As large language models (LLMs) increasingly make consequential decisions and solve problems alongside humans, we must understand how models arrive at their conclusions. Understanding how an LLM behaves requires observing it across multiple inputs, yet free-text explanations such as chain-of-thought (CoT) are generated per instance and cannot be easily verified on new inputs. We propose {code explanations to address the mismatch in how LLM explanations are generated versus how they are evaluated: a program explains LLM behavior across a dataset of inputs, and its execution enables automatic verification. We define Consistency metric that measures the fraction of inputs on which the program reproduces the LLM's outputs, and use it to evaluate and select candidate programs. Across three open-source code-capable LLMs and eight tasks, we find that LLMs can produce highly consistent code explanations in several settings. Code explanations are also 6$\times$ to 100$\times$ shorter than CoT, and better align with model behavior in 9 out of 24 settings. However, consistency is significantly lower on tasks with more complex linguistic components, where generated programs reproduce model outputs no more frequently than an uninformative explanation baseline, and these programs often rely on longer hard-coded patterns. In experiments with Claude Opus 5.5, providing the model with consistency feedback and code execution improves explanation consistency across tasks, but consistency remains significantly lower on more complex tasks. Our findings establish executable code as a novel, verifiable, and compact approach to explaining LLM behavior and highlight verification feedback as a promising direction for improving explanation consistency. The code for this paper is at this https URL. 

---
# Risk-Aware Joint Inference and Resource Control for Physical AI in 6G Networks 

**Authors**: Amin Farajzadeh, Melike Erol-Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2611.14696)  

**Abstract**: Wireless edge inference enables Physical-AI devices to execute edge-inferred actions using transmitted sensory inputs. Since the risk of incorrect actions vary across observations and tasks, the associated uplink transmission of sensory inputs requires different levels of wireless reliability. For instance, applying a fixed signal-to-interference-plus-noise ratio (SINR) requirement can overprovision low-risk tasks while underprovisioning high-risk ones. Therefore, we propose risk-aware control for edge inference (RACE), which maps predicted action-failure probability and task criticality to adaptive SINR targets and combines them with deadline-derived SINR requirements. A slow-timescale large language model (LLM) proposes task requirements for deterministic validation, while a fast hybrid actor selects inference configurations and transmit power levels, complemented by analytical receive beamforming and finite power restoration. Simulations show $11\%$ improvement in the average accuracy-latency-energy objective over fixed-high-SINR control, with near-zero constraint outage. 

---
# What Entropy-Based View Selection Cannot See: An Audit of Test-Time Adaptation for Vision-Language Models 

**Authors**: Sivasuthan Sukumar, Uthayasanker Thayasivam  

**Link**: [PDF](https://arxiv.org/pdf/2611.14686)  

**Abstract**: Vision-language models such as CLIP can classify images into categories they were never explicitly trained to recognize, but accuracy degrades when test images differ in style or source from their training data. A common remedy generates randomly cropped and flipped copies of each test image, called views, and combines their predictions. Since some views miss the object, methods must identify which views to trust. The dominant approach uses prediction entropy to measure confidence and retains the most certain views. Other methods use signals such as embedding typicality. Despite repeated refinements of these scoring functions, the information they rely on, the model's own response to each view, remains largely unexamined. We investigate this choice while holding other pipeline stages fixed. First, we show that view selection, rather than the gradient update, accounts for almost all accuracy gains. We then examine whether entropy identifies useful views when predictions are wrong, despite at least one view predicting the correct class. In these cases, entropy ranks correct views above incorrect ones no better than chance, recovering only about one-twentieth of the accuracy recovered by an oracle over the same views. This finding holds across two image encoders and under a control ruling out an obvious statistical artifact. An exact analysis explains this limitation: entropy measures confidence but not which class a view favors. We identify an unused signal already available in the augmentation pipeline: the fraction of the original image retained by each crop. Ranking views by this quantity improves accuracy on 14 of 15 benchmark datasets over entropy and on 11 of 15 over the strongest scoring function tested. A pre-method measurement also predicts which datasets benefit. 

---
# MITRA-Swarm: Injecting Explicit Linguistic Structure into Foundation Models for Classical Machine Translation 

**Authors**: Kayshav Bhardwaj, Pragun Seth, Kush Bhardwaj, Vijay Nagasamy, Kurt Keutzer  

**Link**: [PDF](https://arxiv.org/pdf/2611.14675)  

**Abstract**: Classical Sanskrit poses three challenges unresolved by single-call monolithic models: morphophonemic fusion Sandhi, dense nominal compounding Samāsa, and discourse-level zero-anaphora forced by poetic meter. We present MITRA-Swarm, a multi-agent architecture injecting explicit linguistic structure into an otherwise monolithic pipeline: a Sandhi-splitting agent, a parallel array of generalist gemma-4-26B-A4B-it and domain-specialized gemma-2-mitra-it translators, and a discourse-aware Ellipsis Restorer that consults corpus-derived register tags to avoid inventing active-voice subjects in impersonal legal and philosophical registers. Two separated judges evaluate quality: a routing judge ($J_{\text{route}}$, same model family as the generators) driving an internal critique loop, and an independent reporting judge ($J_{\text{report}}=$gpt-4o-mini, no shared vendor lineage) scoring every output exactly once. On a stratified benchmark of 1,000 verses across five registers (23,990 scored verdicts, 12-condition pre-registered ablation), the full swarm reaches $85.90$ MQM against a monolithic baseline of $85.27$ ($\Delta=+0.63$, $p_{\text{adj}}=0.0018$, 95% CI $[+0.30,+0.97]$), and isolated discourse restoration alone accounts for $+0.44$ MQM ($p_{\text{adj}}=0.0288$), outperforming a call-count-matched self-refinement baseline by $+0.51$ MQM ($p_{\text{adj}}=0.0288$). Severe per-verse failures (MQM$<80$) fall from $9.0\%$ to $7.6\%$, at roughly $9\times$ the per-verse inference cost of the baseline. Additionally, $J_{\text{route}}$ scores its own family's outputs $2$-$7$ MQM higher than $J_{\text{report}}$ on every one of the 12 conditions, empirically motivating the two-judge split; and a preflight screen of a 192,151-entry retrieval index against all 12 evaluation works verifies 0 reference leaks across 269 runtime checks. These are small gains on a metric saturating near 85-86. 

---
# Where Should the Code Go? Locating Foundation-Model Errors in Bitemporal Inventory Simulation 

**Authors**: Durga Sandeep Saluru, Lanchen Song, Aditya Kumaran  

**Link**: [PDF](https://arxiv.org/pdf/2611.14671)  

**Abstract**: Operational reports carry two clocks: when an event happened and when its report arrived; later reports can revise earlier ones. A foundation-model component that maintains the state supported by the reports visible at a cutoff, then rolls a supplied scenario forward, can be helped by exact code before it, after it or inside the call, or asked to reason harder. We ask which remedy removes which error, for which models, on an exactly solvable lost-sales inventory environment: 36,672 independently rescored requests over twelve configurations of closed and open-weight models, with a forced-answer-call interface as a control and protocols frozen before the calls. Weak models fail in separable ways. Canonical input removes report-selection error, exact continuation removes rollout error, neither repairs the other, and which error dominates differs by model; strong models are at ceiling on raw reports. Asking for more reasoning is no substitute and can depend on the interface: under a forced answer call high effort collapses GPT-5.4 mini and DeepSeek V4 Pro, which gain or hold under automatic choice; on its provider the forced call disables reasoning for DeepSeek V4 Flash; and at high effort gpt-oss exhausts its output budget. The contribution is a controlled component diagnostic, released with every request, and evidence that for some model and provider pairs the tool-calling interface confounds reasoning evaluation. 

---
# Outcome Over Learning? Evidence from Student Use of AI Code Assistants for Unit Test Writing 

**Authors**: Daniel Hinbest, Jeremy S. Bradbury, Michael A. Miljanovic  

**Link**: [PDF](https://arxiv.org/pdf/2611.14664)  

**Abstract**: Artificial intelligence (AI) code assistants are rapidly being adopted in programming education, yet little is known about how students naturally use these tools when given unrestricted access. In a user study investigating AI-assisted unit test writing, we observed that a majority of students adopted a one-shot interaction strategy, asking the AI to generate a complete unit test directly from the initial requirements. While this approach is capable of efficiently producing a solution, it shifts students' focus from learning how to write effective unit tests to simply completing the task. Moreover, because students rarely engaged in critically evaluating, refining, or iterating on AI-generated output, they may be missing opportunities to develop the critical thinking skills that are essential for effective and responsible use of AI. In an effort to shift student focus from outcome to learning we also discuss how training, accountability and improved learning task design can better support a more learning-centric responsible use of AI. 

---
# Abstention Is Not Threshold Compliance: Stated Scoring Rules in Open Language Models 

**Authors**: Dongsun Moon  

**Link**: [PDF](https://arxiv.org/pdf/2611.14661)  

**Abstract**: Kalai et al. propose stating the scoring rule in the prompt with an explicit threshold t, so that a model should answer only when its probability of being correct exceeds t. We probe each requirement of compliance on shared items, in 12 checkpoints from three post-training stages of four fully open 3-8B recipes, with exploratory models up to 32B. At t = 0.75, a line limiting the reply to an answer or "I abstain" (the abstention format) raised the mean abstention rate from 20% to 91%. Yet within this format, raising t from 0.5 to 0.9 changed it by only -0.8 percentage points. Without the line, the higher threshold even lowered abstention, and up to 40% of a checkpoint's items received decisions that no compliant policy with threshold-independent beliefs could make. Given the probability that their answer was correct, the checkpoints barely responded to the threshold in a forced-choice probe. Comparisons with self-assessment signals such as P(True) depended on the grader and on answer drift between prompts. An external threshold on recalibrated P(True) made coverage non-increasing in t by construction. Calibrated on the evaluation distribution (post hoc), it beat the abstention format at t = 0.5; calibrated on TriviaQA and applied to PopQA, however, it fell below that format at 0.75. On OLMo 3 7B, decisions trained on threshold-conditioned targets did not follow unseen thresholds consistently. A post-hoc variant trained at four thresholds answered less as t rose but still failed a pre-specified test, mainly on agreement with its training rule. A high abstention rate is not evidence of compliance: evaluations using stated scoring rules should report how often models abstain, whether decisions respond to the stated threshold, and which items they answer. 

---
# Thompson Sampling using Prior-fitted Diffusion Transformers 

**Authors**: Sihwa Park, Jingsen Zhu, Vinamr Jain, Sheng-Yen Chou, Alexander Terenin  

**Link**: [PDF](https://arxiv.org/pdf/2611.14656)  

**Abstract**: Thompson sampling is natural randomized strategy for optimizing unknown functions in a black-box manner. In continuous global optimization, Thompson sampling has largely been restricted to Gaussian process priors, as these are essentially the only class for which computing the posterior distribution has been tractable in practice. In this work, we introduce \emph{prior-fitted diffusion transformers}, which are diffusion models that perform Thompson sampling in-context at inference time, after being pre-trained using the prior-fitted network paradigm. To apply these models, we develop a set of practical non-Gaussian priors for general black-box functions---including those that have characteristics that cannot be achieved under Gaussianity, such as priors over unimodal functions. We also quantify the tradeoff between model size and the number of pre-training iterations, and show that these models follow scaling laws akin to those observed in transformers defined over other modalities. Across a set of global optimization benchmarks, we show that prior-fitted diffusion transformers with non-Gaussian priors can achieve improved performance compared to traditional pipelines built on Gaussian processes. 

---
# Multiple-Choice Secretary Problem with Predictions 

**Authors**: Jing Yuan, Qi Cai, Xin Gao, Shaojie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14651)  

**Abstract**: We study the multiple-choice secretary problem, in which candidates arrive in uniformly random order and the goal is to select at most $k$ candidates to maximize their total value. We consider a setting where predictions of individual candidate values are available before any candidate arrives. Fujii and Yoshida's Learned Kleinberg algorithm achieves competitive loss $O(\min\{\log k/\sqrt{k},\eps\})$, where $k$ is the selection capacity and $\eps$ is the maximum multiplicative prediction error. We remove the logarithmic factor through a sharper analysis of the same switching algorithm with a retuned threshold. The resulting competitive ratio is at least $\max\{0,1-\min\{32/\sqrt{k},4\eps\}\}$. In particular, exact predictions ($\eps=0$) achieve the offline optimum on every arrival order. 

---
# Evaluating AI Musical Creativity: Towards a Quantitative Multi-Dimensional Framework Based on Value, Novelty, and Surprise 

**Authors**: Hindy Rossignol, Giorgio Franceschelli, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2611.14634)  

**Abstract**: Despite the increasing sophistication of generative Artificial Intelligence (AI), research on the evaluation of AI-generated musical creativity remains limited. To address this gap, this paper proposes a quantitative framework for the formalization and evaluation of AI musical creativity. Building on Margaret Boden's theory, which conceptualizes creativity in terms of value, novelty, and surprise, the framework adapts and operationalizes these three dimensions for the musical domain. On the hypothesis that certain aspects of musical creativity can be assessed with a degree of objectivity, a domain-specific set of quantitative metrics for evaluating the creativity of AI-generated music is introduced. Rather than reducing musical creativity to a single unifying measure, this article proposes an ensemble of complementary metrics that capture its different dimensions, recognizing the complexity of musical artifacts and the difficulty of defining meaningful multi-objective compositional criteria. The framework has two implications. First, it gives researchers and developers a principled methodology and quantitative tools for empirically evaluating creative outputs, enabling more informed model development and reducing unnecessary computational costs. Second, it lays the groundwork for future generative models to optimize directly for creativity by adopting these metrics as learning objectives. 

---
# KernelMosaic: An Open Dataset and Agentic Framework for CUDA Kernel Generation 

**Authors**: Ningyuan Xi, Yuhang Lai, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2611.14604)  

**Abstract**: Training large language models to generate efficient CUDA kernels requires supervision spanning diverse computational requirements and iterative code optimization. Constructing this supervision is challenging because of heterogeneous specifications, and useful implementations emerge through iterative complicated testing and refinement. We introduce KernelMosaic, an open dataset pairing diverse CUDA tasks with verified solutions and trajectories. Human-guided review and diversity-aware selection curate tasks expressed as natural-language descriptions, reference programs, mathematical specifications, and their intersections. To build the solutions and trajectories, our agentic framework guides a strong LLM through implementation, correctness repair, and performance optimization with reusable CUDA guidance and execution feedback. Supervised fine-tuning of a small open-source model on KernelMosaic raises correctness from 4%, 0%, and 0% to 88%, 82%, and 16% on KernelBench L1, L2, and L3, respectively, while increasing the fraction of tasks with kernels faster than PyTorch eager at every level. Further training experiments show that supervision from the curated trajectories substantially outperforms direct responses and promotes sustained testing and revision. These results demonstrate that KernelMosaic enables a general-purpose model to achieve substantial gains in CUDA kernel generation. 

---
# Multi-Step Underwater Acoustic Channel Prediction via Dual-Domain Structure Preservation and Pretrained Transformer 

**Authors**: Mingjie Cao, Gewei Zhang, Deqing Wang, Lizhao You, Fei-Yun Wu, Liqun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.14592)  

**Abstract**: Accurate future channel state information (CSI) is essential for underwater acoustic (UWA) communication because delays associated with signal propagation, feedback, scheduling, and transceiver reconfiguration can render currently available CSI outdated. However, most existing predictors estimate only the next channel state and obtain multi-step forecasts through recursive rollout, which causes prediction errors to accumulate as the multipath structure evolves. The scarcity of sea-trial measurements further limits reliable model training. To address these challenges, we propose LLM4UACP, a dual-domain, structure-preserving framework that adapts a pretrained Transformer for few-shot multi-step UWA channel prediction. LLM4UACP extracts complementary features from the channel impulse response (CIR) and channel frequency response (CFR), aligns them through Fourier-consistent fusion, and encodes the fused representation as delay-time tokens. A pretrained Transformer with a frozen backbone is adapted through Butterfly Orthogonal Fine-tuning to jointly predict multiple future CIRs in a single forward pass, thereby avoiding the error accumulation associated with recursive prediction. Experiments on the measured SO-06 and SO-07 channels show that LLM4UACP achieves normalized mean squared error (NMSE) values of 0.0411 and 0.0399, corresponding to reductions of 88.53% and 85.15%, respectively, relative to the most recent baseline. With only 0.11M trainable parameters, LLM4UACP maintains robust performance across varying prediction horizons and observation-window lengths, limited-data settings, and cross-dataset transfer scenarios. 

---
# CAFIN: CUDA-accelerated Adaptive Fisher Information Matrix Network Optimization via GPU-Native Kernel Fusion 

**Authors**: Maryam Arshad, Muhammad Khuram Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2611.14589)  

**Abstract**: Second-order optimizers improve deep neural network training using curvature information, but their practical efficiency is limited by GPU execution overhead. AdaFisher reduces Fisher Information Matrix (FIM) approximation costs through diagonal Kronecker structures and exponential moving averages, yet its PyTorch implementation requires sequential CUDA kernel launches, intermediate tensor allocations, and repeated memory accesses. We introduce CAFIN, a GPU-native CUDA kernel fusion framework that improves AdaFisher execution efficiency without changing its optimization formulation. The proposed approach progressively integrates FIM preconditioner construction, exponential moving average updates, gradient preconditioning, momentum accumulation, weight decay, and parameter updates into fused CUDA kernels. Shared-memory reuse, register-resident computation, and in-place operations further reduce memory traffic and intermediate allocations. Experiments using ResNet-18 on CIFAR-10, CIFAR-100, and Tiny ImageNet across NVIDIA GTX 1660 Ti and RTX 3060 Laptop GPUs demonstrate up to 78.4% lower isolated FIM construction latency and 2.1-26.9% lower optimizer-step latency. The implementation also eliminates approximately 18.91 MB of intermediate GPU memory allocations per optimizer step. Numerical validation confirms consistency within the predefined FP32 tolerance, while convergence accuracy remains comparable across evaluated datasets and random seeds. These findings demonstrate that CUDA kernel fusion can substantially accelerate second-order optimization while preserving AdaFisher's underlying mathematical formulation and learning behavior. 

---
# Finite-Cadence Refresh Equilibria for Streaming Prior-Anchored Ridge: A Closed-Form Forgetting Window and Spectral Correction 

**Authors**: Mrigank Shekhar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2611.14586)  

**Abstract**: Recursive estimators on drifting streams must forget stale data fast enough to track the target, yet slowly enough to average out noise. In prior-anchored ridge regression, this trade-off admits a closed-form spectral decomposition in which each eigen-direction interpolates between the data solution and the prior with pull weight $s_i = \rho / (\mu_i + \rho)$. When the anchor is refreshed every $k$ steps, we show that the moving-frame offset converges to a finite-cadence equilibrium that lags the pure-leaky path by an explicit $\Delta_k(s)$ while reducing estimation variance by a factor $\Psi(s, r_k)$. Minimizing the cycle-averaged risk gives $\omega^* = (4\nu^2 \Sigma / c')^{1/3}$, $c' = \sigma^2 d_{\mathrm{eff}}(\rho)$, and the plug-in strength $\rho^* = \sigma^2 / (n\delta^2)$. A random-matrix deterministic equivalent extends both laws to finite dimension, renormalizing the anchor to $\rho_{\mathrm{eff}} = \rho (e^{\zeta q} - 1) / (\zeta q)$; it matches simulation within 5% at fixed operating points and within 13% at the method's own, and rank-deficient populations reduce exactly to their active subspace. The resulting method, SPAR, calibrates window and strength online from residuals. On a 27-configuration grid the calibrated window opens on drift without divergence, and the grid defaults transfer to an hourly electricity stream and to BCI IV 2b. The core algebra, refresh chain, and regret bounds are formalized in Lean 4. 

---
# When Evaluators of AI Agents Disagree: An Empirical Study of Native Scoring, LLM Judges, and Commit-Boundary Checks on tau2-bench 

**Authors**: Sungbin Im  

**Link**: [PDF](https://arxiv.org/pdf/2611.14581)  

**Abstract**: Agentic workflows, in which a large language model (LLM) calls tools to read and write data, are evaluated in two ways. Benchmarks compare the final state against a gold answer. Deployed services have none; an LLM judge reads the transcript and decides. Practice treats the two as interchangeable. This paper measures whether they are.
We ran 2,160 episodes in tau2-bench's airline and retail domains and recorded every database write. We scored each episode four ways: native scoring against gold state, an LLM judge, that judge given the policy document, and the boundary evaluator, a minimal rule-based check of whether a write was expected against whether one occurred in the trace. Two annotators independently labeled 120 episodes for task correctness, policy compliance, and commitment correctness.
The judge repeated its verdict 98.2% of the time but agreed with native scoring on 56.2%. In 11 episodes the agent executed a forbidden write. The judge passed 10 to 11, with or without the policy or transcript. A stronger judge still passed 6. The boundary evaluator, with no model call, caught all 11; no judge significantly exceeded it on human labels. Forbidden writes occurred in 11.1% of episodes without gates. Verification, correction, and safe-termination gates did not reduce them significantly. The traces show a candidate cause: the gates' evidence digest dropped the order status and customer authentication, so valid requests were rejected. The choice of evaluator decides the conclusion; the minimal check exposes failures the judges miss. Traces, code, prompts, and labels are released. 

---
# MechFoundry: An agentic loop that poses and performs engineering analyses and optimization of real-world objects 

**Authors**: Xue-Ling Luo, Marco Conte, Peter Yichen Chen, Nikolaos Bouklas  

**Link**: [PDF](https://arxiv.org/pdf/2611.14571)  

**Abstract**: AI agents can automate software engineering because a compiler or a test suite checks that the code does what the user or the agent intended. In engineering analysis, however, the steps that precede the computation have few easily verifiable checks: modeling a design or an object, deciding how it is used, where supports and loads are applied, and what material each part is made of. A human expert therefore makes these decisions, carries out the steps, and judges the result, one object at a time. We present MechFoundry, an agentic loop coupled with bespoke software that automates structural analyses of real-world objects without a human in the loop. From an object's lexical description, it generates a shape, poses a loading case relevant to how the object is used, assigns a material to each part, solves the nonlinear structural response, and improves the design against the loads it carries. A differentiable GPU solver built on an adaptive octree that need not conform to the surface enables the transition from shape generation to simulation and optimization. Across 50 everyday and engineering objects, the agent posed and solved a loading case in 6.5 minutes and for US$1.3 per object on average. Without a test suite to check against, we had language-model reviewers grade each session, as a first indication of where the agent's engineering judgment holds and where it fails. 

---
# Deep learning-guided discovery reveals the diversity and mutational potential of environmental beta-lactamases 

**Authors**: Liangjun Ge, Qiandi Gao, Haiming Xu, Zhiwei Qin, Heqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14568)  

**Abstract**: Beta-lactamases are a major mechanism of bacterial resistance to beta-lactam antibiotics; however, their functional diversity and mutational potential in environmental microbial communities remain partially understood. This study aimed to characterize environmental beta-lactamase diversity and investigate how sequence variation influences antibiotic resistance. By integrating protein language model (PLMs) with ensemble learning, we identified more than 46,000 putative beta-lactamases across all four Ambler classes from environmental metagenomes, widely distributed across diverse habitats, with distinct habitat-associated class compositions. The candidates sharing approximately 30% sequence identity with known beta-lactamases conferred resistance to multiple beta-lactam antibiotics when expressed in Escherichia coli. Combining inverse folding model with PLMs further identified mutations that altered resistance profiles, including substitutions that increased resistance to carbapenems. Predicted high-fitness sites exhibited greater natural sequence variability overall, with higher variability in wastewater-derived than ocean-derived enzymes. These findings reveal undercharacterized beta-lactamase diversity and substantial mutational flexibility in environmental microbiomes, providing a basis for investigating their potential contribution to the diversification of antibiotic resistance. 

---
# AVA: Agentic Variant Adjudication for Automated, Auditable Read-Level Review in Cancer Diagnostic Sequencing 

**Authors**: Caleb Chin, Sidharth Gupta, Arun K Anguraj Vadivel, Robert Siddaway, Siddharth Arya, Arianna He, Peter Zhukovsky, Cynthia Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2611.14560)  

**Abstract**: Identifying a tumor's DNA mutations is central to cancer diagnosis and treatment, but distinguishing genuine mutations from sequencing artifacts is error-prone. In practice, genuine mutations are identified using fixed variant-allele-frequency (VAF) and depth thresholds which cannot separate artifacts from genuine mutations in the borderline low-VAF zone where they overlap. A laboratory specialist must manually inspect the read-level sequencing data, a slow step that has high inter-rater variability and causes diagnosis delays. We present AVA: Agentic Variant Adjudication; an agentic framework that automates this read-level inspection with subagents that reason over evidence, generate investigative code, and return a real-or-artifact adjudications with an auditable reasoning trace all without training on existing data. On both the SEQC2 DNA-seq benchmark and patient RNA-seq data, restricted to a subsample of borderline variants where thresholds fail, both closed-source (Claude Sonnet 5) and open-weight (GLM 5.3 Flash and Gemma 4 31b) agents outperform the best achievable VAF/depth thresholds. Ultimately, AVA is suited for privacy-constrained clinical deployment, since the framework can run on locally deployed open-weight models with descriptive reasoning traces to keep specialists in the loop. 

---
# SideTaskBench: Measuring Unauthorized Side-Task Delivery in Coding Agents 

**Authors**: Xiangyu Wen, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.14548)  

**Abstract**: Can a coding agent fix a user's repository while also completing and delivering work the user never requested? In this paper, we introduce SideTaskBench to measure this failure where third-party technical references induce agents to produce independent side-task artifacts during an authorized repair. Repair success alone misses this behavior. Our key measurement contribution is to pair each repair with a self-contained side task and verify side tasks' delivery success rate along with the repair success rate. The benchmark contains 25 Python repairs and 190 side-task units. Its four-level similarity index defines 300 executable repair--side-task pairs.
The experiments show roughly 40% valid delivery under wrapped requests for two models, alongside repair success close to their clean baselines. The completed v10 similarity run uses DeepSeek-V4-Flash on all 300 pairs, with eight repetitions per pair (2,400 runs). Conditional delivery ranges from 20.48% for L0 unrelated tasks to 57.44% for L3 mechanism-overlapping tasks, while the L1-L0 difference is 5.37 percentage points and remains sensitive to task composition. The central insight is that technical relevance may blur the authorization boundary for an agent, such that an agent can solve the requested problem and still deliver unauthorized work. SideTaskBench emphasizes this potential weakness of existing agents and provides a testbed for defenses that preserve repair capability while preventing unauthorized delivery. The code is available at: this https URL. 

---
# metalSPARSE: A 2:4 Semi-Structured Sparse GEMV Kernel for Apple Silicon GPUs 

**Authors**: Tek Raj Awasthi, Babita Awasthi  

**Link**: [PDF](https://arxiv.org/pdf/2611.14545)  

**Abstract**: Autoregressive Large Language Models (LLMs) generate text one token at a time, and each token reads every weight from memory, so on edge devices, decoding is limited by memory bandwidth. 2:4 semi-structured sparsity keeps two of every four weights, but Apple-silicon GPUs have no sparse matrix hardware, so pruning helps only if a kernel actually reads fewer bytes. We present metalSPARSE, to our knowledge the first 2:4 sparse kernels for Apple GPUs, together with a fused decoder written in Metal. The kernels stream weights in compressed form and expand them in registers. On Qwen2.5-1.5B with 2:4 sparsity on every decoder linear layer, decoding runs 3.6x faster than HuggingFace Transformers on PyTorch-MPS on an 8 GB M2 MacBook Air and 1.52x faster than MLX on a 16 GB M4. The same engine on dense weights matches MLX, so the gain comes from the compressed weights, and sparsity delivers 97% of the speedup its byte reduction allows. Prompt processing, which is compute-bound, does not speed up. To preserve quality, we combine a Wanda mask with block-wise reconstruction and low-rank adapters trained jointly with the surviving weights and quantized to 4 bits. On WikiText-2 this brings perplexity from 3.01x dense after one-shot pruning (Wanda with a closed-form refit) to 1.34x, and the whole pipeline runs on the same laptop/hardware. 

---
# Gaussian processes for real-time urban traffic estimation using a fleet of drones 

**Authors**: Batuhan Avci, Marko Susnjar, Marko Maljkovic, Nikolas Geroliminis  

**Link**: [PDF](https://arxiv.org/pdf/2611.14536)  

**Abstract**: Real-time urban traffic monitoring is essential for modern city management, yet conventional static sensors provide only limited spatial coverage. Autonomous drone fleets can complement this infrastructure by offering flexible, high-resolution mobile sensing across different parts of the network. This paper introduces a spatio-temporal Gaussian process (GP) framework for reconstructing network-wide traffic states from partial aerial observations. The model accounts for the physical structure of the road network through a graph Matérn kernel, which embeds network topology into the spatial covariance structure, and captures temporal dependencies through an exponential temporal kernel. Within this framework, historical loop detector data provide an informative prior mean that is updated in real time using incoming drone observations, yielding network-wide point estimates together with predictive uncertainty. We validate our framework in two distinct case studies: a microscopic simulation of Barcelona and real-world drone data from Athens. The results demonstrate that our approach outperforms historical baselines in estimating and predicting Vehicle Hours Traveled (VHT), notably reducing the required fleet size. Furthermore, leveraging the model's predictive uncertainty to guide drone movements yields additional performance gains, highlighting the potential for future uncertainty-aware fleet routing algorithms. 

---
# EasyFoam: Skill-Oriented OpenFOAM Agents with Evidence-Driven Evolution 

**Authors**: Peng Yin, Zhiwei Chen, Tianyi Liu, Chen Li, Kai Li, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.14525)  

**Abstract**: Large language model (LLM) agents are increasingly capable of automating end-to-end OpenFOAM workflows for computational fluid dynamics (CFD). However, current progress is bottlenecked by two critical limitations. First, existing evaluations are dominated by tutorial-level cases and execution-oriented metrics, masking a substantial executability--acceptability gap: successfully executed simulations often fail to satisfy prescribed numerical or quantity-of-interest (QoI) criteria. Second, domain expertise in existing systems is tightly coupled with bespoke agent architectures, limiting its reuse and continual refinement across agent implementations. To address these challenges, we introduce ExtFoamBench, a challenging benchmark for OpenFOAM agents that spans five axes of workflow variation, and separately assesses execution completion, numerical soundness, and QoI satisfaction. Complementing this, we propose EasyFoam, an agent framework that decouples CFD expertise into modular, externalized skills driven by a dual-loop mechanism. An execution loop constructs workflows from explicit task requirements and uses structured simulation evidence to diagnose failures and guide targeted repairs. Meanwhile, an evolution loop distills task experience into reusable skill updates selected through paired evaluation. Empirical evaluations demonstrate that EasyFoam achieves substantially higher task acceptance rates than baseline agents, establishing this work as a starting point for future research toward self-improving CFD agents. 

---
# Care, Human Enfeeblement, and the Existential Implications of AGI 

**Authors**: B. Scot Rousse  

**Link**: [PDF](https://arxiv.org/pdf/2611.14520)  

**Abstract**: Stuart Russell has named "human enfeeblement" as a long-term societal risk of advanced AI. This paper gives a new interpretation of the enfeeblement danger by interpreting it in light of recent work by Brian Christian, Alison Gopnik, and David Spivak on care and caregiving, and connecting it with the ontological design tradition of Terry Winograd and Fernando Flores. The enfeeblement that threatens is not merely a matter of cognitive off-loading and loss of knowledge, but a form of disengagement with the world which I explain in terms of an erosion of our capacities to care. This paper articulates a conception of care as a constellation of cultivable skills that involve us in the world and with each other, while making life worth living. Designing AI to support our capacities to care addresses the worry recently raised by Demis Hassabis and others regarding the questions of "meaning and purpose" that are looming in a post-AGI world. 

---
# The Impact of Large Language Models (LLMs) on Code Review Process 

**Authors**: Antonio Collante, Samuel Abedu, SayedHassan Khatoonabadi, Ahmad Abdellatif, Ebube Alor, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2508.11034)  

**Abstract**: Large language models (LLMs) have recently gained prominence in the field of software development, significantly boosting productivity and simplifying teamwork. Although prior studies have examined task-specific applications, the phase-specific effects of LLM assistance on the efficiency of code review processes remain underexplored. This research investigates the effect of GPT on GitHub pull request (PR) workflows, with a focus on reducing resolution time, optimizing phase-specific performance, and assisting developers. We curated a dataset of 25,473 PRs from 9,254 GitHub projects and identified GPT-assisted PRs using a semi-automated heuristic approach that combines keyword-based detection, regular expression filtering, and manual verification until achieving 95% labeling accuracy. We then applied statistical modeling, including multiple linear regression and Mann-Whitney U test, to evaluate differences between GPT-assisted and non-assisted PRs, both at the overall resolution level and across distinct review phases. Our research has revealed that early adoption of GPT can substantially boost the effectiveness of the PR process, leading to considerable time savings at various stages. Our findings suggest that GPT-assisted PRs reduced median resolution time by more than 60% (9 hours compared to 23 hours for non-assisted PRs). We discovered that utilizing GPT can reduce the review time by 33% and the waiting time before acceptance by 87%. Analyzing a sample dataset of 300 GPT-assisted PRs, we discovered that developers predominantly use GPT for code optimization (60%), bug fixing (26%), and documentation updates (12%). This research sheds light on the impact of the GPT model on the code review process, offering actionable insights for software teams seeking to enhance workflows and promote seamless collaboration. 

---
