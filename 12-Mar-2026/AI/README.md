# A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification 

**Authors**: Yichi Zhu, Kan Ling, Xu Liu, Hengrun Zhang, Huiqun Yu, Guisheng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.10891)  

**Abstract**: Medication errors pose a significant threat to patient safety, making pharmacist verification (PV) a critical, yet heavily burdened, final safeguard. The direct application of Large Language Models (LLMs) to this zero-tolerance domain is untenable due to their inherent factual unreliability, lack of traceability, and weakness in complex reasoning. To address these challenges, we introduce PharmGraph-Auditor, a novel system designed for safe and evidence-grounded prescription auditing. The core of our system is a trustworthy Hybrid Pharmaceutical Knowledge Base (HPKB), implemented under the Virtual Knowledge Graph (VKG) paradigm. This architecture strategically unifies a relational component for set constraint satisfaction and a graph component for topological reasoning via a rigorous mapping layer. To construct this HPKB, we propose the Iterative Schema Refinement (ISR) algorithm, a framework that enables the co-evolution of both graph and relational schemas from medical texts. For auditing, we introduce the KB-grounded Chain of Verification (CoV), a new reasoning paradigm that transforms the LLM from an unreliable generator into a transparent reasoning engine. CoV decomposes the audit task into a sequence of verifiable queries against the HPKB, generating hybrid query plans to retrieve evidence from the most appropriate data store. Experimental results demonstrate robust knowledge extraction capabilities and show promises of using PharmGraph-Auditor to enable pharmacists to achieve safer and faster prescription verification. 

---
# Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization 

**Authors**: Linghao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10808)  

**Abstract**: The emergence of large language model (LLM)-based agent frameworks has shifted the primary challenge in building domain-expert AI agents from raw capability to effective encoding of domain expertise. Two dominant paradigms -- code-first development, which embeds expertise in deterministic pipelines, and prompt-first development, which captures expertise in static system prompts -- both treat agent construction as a discrete engineering phase preceding deployment. We argue that this sequential assumption creates a fundamental mismatch with the nature of domain expertise, which is substantially tacit, deeply personal, and continuously evolving. We propose Nurture-First Development (NFD), a paradigm in which agents are initialized with minimal scaffolding and progressively grown through structured conversational interaction with domain practitioners. The central mechanism is the Knowledge Crystallization Cycle, whereby fragmented knowledge embedded in operational dialogue is periodically consolidated into structured, reusable knowledge assets. We formalize NFD through: (1) a Three-Layer Cognitive Architecture organizing agent knowledge by volatility and personalization degree; (2) the Knowledge Crystallization Cycle with formal definitions of crystallization operations and efficiency metrics; and (3) an operational framework comprising a Dual-Workspace Pattern and Spiral Development Model. We illustrate the paradigm through a detailed case study on building a financial research agent for U.S. equity analysis and discuss the conditions, limitations, and broader implications of NFD for human-agent co-evolution. 

---
# Emulating Clinician Cognition via Self-Evolving Deep Clinical Research 

**Authors**: Ruiyang Ren, Yuhao Wang, Yunsen Liang, Lan Luo, Jing Liu, Haifeng Wang, Cong Feng, Yinan Zhang, Chunyan Miao, Ji-Rong Wen, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10677)  

**Abstract**: Clinical diagnosis is a complex cognitive process, grounded in dynamic cue acquisition and continuous expertise accumulation. Yet most current artificial intelligence (AI) systems are misaligned with this reality, treating diagnosis as single-pass retrospective prediction while lacking auditable mechanisms for governed improvement. We developed DxEvolve, a self-evolving diagnostic agent that bridges these gaps through an interactive deep clinical research workflow. The framework autonomously requisitions examinations and continually externalizes clinical experience from increasing encounter exposure as diagnostic cognition primitives. On the MIMIC-CDM benchmark, DxEvolve improved diagnostic accuracy by 11.2% on average over backbone models and reached 90.4% on a reader-study subset, comparable to the clinician reference (88.8%). DxEvolve improved accuracy on an independent external cohort by 10.2% (categories covered by the source cohort) and 17.1% (uncovered categories) compared to the competitive method. By transforming experience into a governable learning asset, DxEvolve supports an accountable pathway for the continual evolution of clinical AI. 

---
# FAME: Formal Abstract Minimal Explanation for Neural Networks 

**Authors**: Ryma Boumazouza, Raya Elsaleh, Melanie Ducoffe, Shahaf Bassan, Guy Katz  

**Link**: [PDF](https://arxiv.org/pdf/2603.10661)  

**Abstract**: We propose FAME (Formal Abstract Minimal Explanations), a new class of abductive explanations grounded in abstract interpretation. FAME is the first method to scale to large neural networks while reducing explanation size. Our main contribution is the design of dedicated perturbation domains that eliminate the need for traversal order. FAME progressively shrinks these domains and leverages LiRPA-based bounds to discard irrelevant features, ultimately converging to a formal abstract minimal explanation. To assess explanation quality, we introduce a procedure that measures the worst-case distance between an abstract minimal explanation and a true minimal explanation. This procedure combines adversarial attacks with an optional VERIX+ refinement step. We benchmark FAME against VERIX+ and demonstrate consistent gains in both explanation size and runtime on medium- to large-scale neural networks. 

---
# Trajectory-Informed Memory Generation for Self-Improving Agent Systems 

**Authors**: Gaodan Fang, Vatche Isahagian, K. R. Jayaram, Ritesh Kumar, Vinod Muthusamy, Punleuk Oum, Gegi Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2603.10600)  

**Abstract**: LLM-powered agents face a persistent challenge: learning from their execution experiences to improve future performance. While agents can successfully complete many tasks, they often repeat inefficient patterns, fail to recover from similar errors, and miss opportunities to apply successful strategies from past executions. We present a novel framework for automatically extracting actionable learnings from agent execution trajectories and utilizing them to improve future performance through contextual memory retrieval. Our approach comprises four components: (1) a Trajectory Intelligence Extractor that performs semantic analysis of agent reasoning patterns, (2) a Decision Attribution Analyzer that identifies which decisions and reasoning steps led to failures, recoveries, or inefficiencies, (3) a Contextual Learning Generator that produces three types of guidance -- strategy tips from successful patterns, recovery tips from failure handling, and optimization tips from inefficient but successful executions, and (4) an Adaptive Memory Retrieval System that injects relevant learnings into agent prompts based on multi-dimensional similarity. Unlike existing memory systems that store generic conversational facts, our framework understands execution patterns, extracts structured learnings with provenance, and retrieves guidance tailored to specific task contexts. Evaluation on the AppWorld benchmark demonstrates consistent improvements, with up to 14.3 percentage point gains in scenario goal completion on held-out tasks and particularly strong benefits on complex tasks (28.5~pp scenario goal improvement, a 149\% relative increase). 

---
# Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning 

**Authors**: Zhaowei Zhang, Xiaohan Liu, Xuekai Zhu, Junchao Huang, Ceyao Zhang, Zhiyuan Feng, Yaodong Yang, Xiaoyuan Yi, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.10588)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in logical reasoning tasks, yet whether large language model (LLM) alignment requires fundamentally different approaches remains unclear. Given the apparent tolerance for multiple valid responses in moral reasoning, a natural hypothesis is that alignment tasks inherently require diversity-seeking distribution-matching algorithms rather than reward-maximizing policy-based methods. We conduct the first comprehensive empirical study comparing both paradigms on MoReBench. To enable stable RLVR training, we build a rubric-grounded reward pipeline by training a Qwen3-1.7B judge model. Contrary to our hypothesis, we find that distribution-matching approaches do not demonstrate significant advantages over reward-maximizing methods as expected on alignment tasks. Through semantic visualization mapping high-reward responses to semantic space, we demonstrate that moral reasoning exhibits more concentrated high-reward distributions than mathematical reasoning, where diverse solution strategies yield similarly high rewards. This counter-intuitive finding explains why mode-seeking optimization proves equally or more effective for alignment tasks. Our results suggest that alignment tasks do not inherently require diversity-preserving algorithms, and standard reward-maximizing RLVR methods can effectively transfer to moral reasoning without explicit diversity mechanisms. 

---
# CUAAudit: Meta-Evaluation of Vision-Language Models as Auditors of Autonomous Computer-Use Agents 

**Authors**: Marta Sumyk, Oleksandr Kosovan  

**Link**: [PDF](https://arxiv.org/pdf/2603.10577)  

**Abstract**: Computer-Use Agents (CUAs) are emerging as a new paradigm in human-computer interaction, enabling autonomous execution of tasks in desktop environment by perceiving high-level natural-language instructions. As such agents become increasingly capable and are deployed across diverse desktop environments, evaluating their behavior in a scalable and reliable manner becomes a critical challenge. Existing evaluation pipelines rely on static benchmarks, rule-based success checks, or manual inspection, which are brittle, costly, and poorly aligned with real-world usage. In this work, we study Vision-Language Models (VLMs) as autonomous auditors for assessing CUA task completion directly from observable interactions and conduct a large-scale meta-evaluation of five VLMs that judge task success given a natural-language instruction and the final environment state. Our evaluation spans three widely used CUA benchmarks across macOS, Windows, and Linux environments and analyzes auditor behavior along three complementary dimensions: accuracy, calibration of confidence estimates, and inter-model agreement. We find that while state-of-the-art VLMs achieve strong accuracy and calibration, all auditors exhibit notable performance degradation in more complex or heterogeneous environments, and even high-performing models show significant disagreement in their judgments. These results expose fundamental limitations of current model-based auditing approaches and highlight the need to explicitly account for evaluator reliability, uncertainty, and variance when deploying autonomous CUAs in real-world settings. 

---
# Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents 

**Authors**: Yuanhao Li, Haozhe Wang, Geyong Min, Nektarios Georgalas, Wang Miao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10564)  

**Abstract**: The integration of Generative AI models into AI-native network systems offers a transformative path toward achieving autonomous and adaptive control. However, the application of such models to continuous control tasks is impeded by intrinsic architectural limitations, including finite context windows, the lack of explicit reward signals, and the degradation of the long context. This paper posits that the key to unlocking robust continuous control is enabling agents to internalize experience by distilling it into their parameters, rather than relying on prompt-based memory. To this end, we propose a novel self-finetuning framework that enables agentic systems to learn continuously through direct interaction with the environment, bypassing the need for handcrafted rewards. Our framework implements a bi-perspective reflection mechanism that generates autonomous linguistic feedback to construct preference datasets from interaction history. A subsequent preference-based fine-tuning process distills long-horizon experiences into the model's parameters. We evaluate our approach on a dynamic Radio Access Network (RAN) slicing task, a challenging multi-objective control problem that requires the resolution of acute trade-offs between spectrum efficiency, service quality, and reconfiguration stability under volatile network conditions. Experimental results show that our framework outperforms standard Reinforcement Learning (RL) baselines and existing Large Language Model (LLM)-based agents in sample efficiency, stability, and multi-metric optimization. These findings demonstrate the potential of self-improving generative agents for continuous control tasks, paving the way for future AI-native network infrastructure. 

---
# IH-Challenge: A Training Dataset to Improve Instruction Hierarchy on Frontier LLMs 

**Authors**: Chuan Guo, Juan Felipe Ceron Uribe, Sicheng Zhu, Christopher A. Choquette-Choo, Steph Lin, Nikhil Kandpal, Milad Nasr, Sam Toyer, Miles Wang, Yaodong Yu, Alex Beutel, Kai Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10521)  

**Abstract**: Instruction hierarchy (IH) defines how LLMs prioritize system, developer, user, and tool instructions under conflict, providing a concrete, trust-ordered policy for resolving instruction conflicts. IH is key to defending against jailbreaks, system prompt extractions, and agentic prompt injections. However, robust IH behavior is difficult to train: IH failures can be confounded with instruction-following failures, conflicts can be nuanced, and models can learn shortcuts such as overrefusing. We introduce IH-Challenge, a reinforcement learning training dataset, to address these difficulties. Fine-tuning GPT-5-Mini on IH-Challenge with online adversarial example generation improves IH robustness by +10.0% on average across 16 in-distribution, out-of-distribution, and human red-teaming benchmarks (84.1% to 94.1%), reduces unsafe behavior from 6.6% to 0.7% while improving helpfulness on general safety evaluations, and saturates an internal static agentic prompt injection evaluation, with minimal capability regression. We release the IH-Challenge dataset (this https URL) to support future research on robust instruction hierarchy. 

---
# Resource-constrained Amazons chess decision framework integrating large language models and graph attention 

**Authors**: Tianhao Qian, Zhuoxuan Li, Jinde Cao, Xinli Shi, Hanjie Liu, Leszek Rutkowski  

**Link**: [PDF](https://arxiv.org/pdf/2603.10512)  

**Abstract**: Artificial intelligence has advanced significantly through the development of intelligent game-playing systems, providing rigorous testbeds for decision-making, strategic planning, and adaptive learning. However, resource-constrained environments pose critical challenges, as conventional deep learning methods heavily rely on extensive datasets and computational resources. In this paper, we propose a lightweight hybrid framework for the Game of the Amazons, which explores the paradigm of weak-to-strong generalization by integrating the structural reasoning of graph-based learning with the generative capabilities of large language models. Specifically, we leverage a Graph Attention Autoencoder to inform a multi-step Monte Carlo Tree Search, utilize a Stochastic Graph Genetic Algorithm to optimize evaluation signals, and harness GPT-4o-mini to generate synthetic training data. Unlike traditional approaches that rely on expert demonstrations, our framework learns from noisy and imperfect supervision. We demonstrate that the Graph Attention mechanism effectively functions as a structural filter, denoising the LLM's outputs. Experiments on a 10$\times$10 Amazons board show that our hybrid approach not only achieves a 15\%--56\% improvement in decision accuracy over baselines but also significantly outperforms its teacher model (GPT-4o-mini), achieving a competitive win rate of 45.0\% at N=30 nodes and a decisive 66.5\% at only N=50 nodes. These results verify the feasibility of evolving specialized, high-performance game AI from general-purpose foundation models under stringent computational constraints. 

---
# Verbalizing LLM's Higher-order Uncertainty via Imprecise Probabilities 

**Authors**: Anita Yang, Krikamol Muandet, Michele Caprio, Siu Lun Chau, Masaki Adachi  

**Link**: [PDF](https://arxiv.org/pdf/2603.10396)  

**Abstract**: Despite the growing demand for eliciting uncertainty from large language models (LLMs), empirical evidence suggests that LLM behavior is not always adequately captured by the elicitation techniques developed under the classical probabilistic uncertainty framework. This mismatch leads to systematic failure modes, particularly in settings that involve ambiguous question-answering, in-context learning, and self-reflection. To address this, we propose novel prompt-based uncertainty elicitation techniques grounded in \emph{imprecise probabilities}, a principled framework for repesenting and eliciting higher-order uncertainty. Here, first-order uncertainty captures uncertainty over possible responses to a prompt, while second-order uncertainty (uncertainty about uncertainty) quantifies indeterminacy in the underlying probability model itself. We introduce general-purpose prompting and post-processing procedures to directly elicit and quantify both orders of uncertainty, and demonstrate their effectiveness across diverse settings. Our approach enables more faithful uncertainty reporting from LLMs, improving credibility and supporting downstream decision-making. 

---
# Beyond Scalars: Evaluating and Understanding LLM Reasoning via Geometric Progress and Stability 

**Authors**: Xinyan Jiang, Ninghao Liu, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10384)  

**Abstract**: Evaluating LLM reliability via scalar probabilities often fails to capture the structural dynamics of reasoning. We introduce TRACED, a framework that assesses reasoning quality through theoretically grounded geometric kinematics. By decomposing reasoning traces into Progress (displacement) and Stability (curvature), we reveal a distinct topological divergence: correct reasoning manifests as high-progress, stable trajectories, whereas hallucinations are characterized by low-progress, unstable patterns (stalled displacement with high curvature fluctuations). Leveraging these signatures, our probabilistic framework achieves competitive performance and superior robustness across diverse benchmarks. Crucially, TRACED bridges geometry and cognition by mapping high curvature to ''Hesitation Loops'' and displacement to ''Certainty Accumulation'', offering a physical lens to decode the internal dynamics of machine thought. 

---
# HEAL: Hindsight Entropy-Assisted Learning for Reasoning Distillation 

**Authors**: Wenjing Zhang, Jiangze Yan, Jieyun Huang, Yi Shen, Shuming Shi, Ping Chen, Ning Wang, Zhaoxiang Liu, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2603.10359)  

**Abstract**: Distilling reasoning capabilities from Large Reasoning Models (LRMs) into smaller models is typically constrained by the limitation of rejection sampling. Standard methods treat the teacher as a static filter, discarding complex "corner-case" problems where the teacher fails to explore valid solutions independently, thereby creating an artificial "Teacher Ceiling" for the student. In this work, we propose Hindsight Entropy-Assisted Learning (HEAL), an RL-free framework designed to bridge this reasoning gap. Drawing on the educational theory of the Zone of Proximal Development(ZPD), HEAL synergizes three core modules: (1) Guided Entropy-Assisted Repair (GEAR), an active intervention mechanism that detects critical reasoning breakpoints via entropy dynamics and injects targeted hindsight hints to repair broken trajectories; (2) Perplexity-Uncertainty Ratio Estimator (PURE), a rigorous filtering protocol that decouples genuine cognitive breakthroughs from spurious shortcuts; and (3) Progressive Answer-guided Curriculum Evolution (PACE), a three-stage distillation strategy that organizes training from foundational alignment to frontier breakthrough. Extensive experiments on multiple benchmarks demonstrate that HEAL significantly outperforms traditional SFT distillation and other baselines. 

---
# Hybrid Self-evolving Structured Memory for GUI Agents 

**Authors**: Sibo Zhu, Wenyi Wu, Kun Zhou, Stephen Wang, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10291)  

**Abstract**: The remarkable progress of vision-language models (VLMs) has enabled GUI agents to interact with computers in a human-like manner. Yet real-world computer-use tasks remain difficult due to long-horizon workflows, diverse interfaces, and frequent intermediate errors. Prior work equips agents with external memory built from large collections of trajectories, but relies on flat retrieval over discrete summaries or continuous embeddings, falling short of the structured organization and self-evolving characteristics of human memory. Inspired by the brain, we propose Hybrid Self-evolving Structured Memory (HyMEM), a graph-based memory that couples discrete high-level symbolic nodes with continuous trajectory embeddings. HyMEM maintains a graph structure to support multi-hop retrieval, self-evolution via node update operations, and on-the-fly working-memory refreshing during inference. Extensive experiments show that HyMEM consistently improves open-source GUI agents, enabling 7B/8B backbones to match or surpass strong closed-source models; notably, it boosts Qwen2.5-VL-7B by +22.5% and outperforms Gemini2.5-Pro-Vision and GPT-4o. 

---
# Agentic Control Center for Data Product Optimization 

**Authors**: Priyadarshini Tamilselvan, Gregory Bramble, Sola Shirai, Ken C. L. Wong, Faisal Chowdhury, Horst Samulowitz  

**Link**: [PDF](https://arxiv.org/pdf/2603.10133)  

**Abstract**: Data products enable end users to gain greater insights about their data by providing supporting assets, such as example question-SQL pairs which can be answered using the data or views over the database tables. However, producing useful data products is challenging, and typically requires domain experts to hand-craft supporting assets. We propose a system that automates data product improvement through specialized AI agents operating in a continuous optimization loop. By surfacing questions, monitoring multi-dimensional quality metrics, and supporting human-in-the-loop controls, it transforms data into observable and refinable assets that balance automation with trust and oversight. 

---
# COMIC: Agentic Sketch Comedy Generation 

**Authors**: Susung Hong, Brian Curless, Ira Kemelmacher-Shlizerman, Steve Seitz  

**Link**: [PDF](https://arxiv.org/pdf/2603.11048)  

**Abstract**: We propose a fully automated AI system that produces short comedic videos similar to sketch shows such as Saturday Night Live. Starting with character references, the system employs a population of agents loosely based on real production studio roles, structured to optimize the quality and diversity of ideas and outputs through iterative competition, evaluation, and improvement. A key contribution is the introduction of LLM critics aligned with real viewer preferences through the analysis of a corpus of comedy videos on YouTube to automatically evaluate humor. Our experiments show that our framework produces results approaching the quality of professionally produced sketches while demonstrating state-of-the-art performance in video generation. 

---
# LiTo: Surface Light Field Tokenization 

**Authors**: Jen-Hao Rick Chang, Xiaoming Zhao, Dorian Chan, Oncel Tuzel  

**Link**: [PDF](https://arxiv.org/pdf/2603.11047)  

**Abstract**: We propose a 3D latent representation that jointly models object geometry and view-dependent appearance. Most prior works focus on either reconstructing 3D geometry or predicting view-independent diffuse appearance, and thus struggle to capture realistic view-dependent effects. Our approach leverages that RGB-depth images provide samples of a surface light field. By encoding random subsamples of this surface light field into a compact set of latent vectors, our model learns to represent both geometry and appearance within a unified 3D latent space. This representation reproduces view-dependent effects such as specular highlights and Fresnel reflections under complex lighting. We further train a latent flow matching model on this representation to learn its distribution conditioned on a single input image, enabling the generation of 3D objects with appearances consistent with the lighting and materials in the input. Experiments show that our approach achieves higher visual quality and better input fidelity than existing methods. 

---
# Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation 

**Authors**: Tao Zhong, Yixun Hu, Dongzhe Zheng, Aditya Sood, Christine Allen-Blanchette  

**Link**: [PDF](https://arxiv.org/pdf/2603.11045)  

**Abstract**: We propose Neural Field Thermal Tomography (NeFTY), a differentiable physics framework for the quantitative 3D reconstruction of material properties from transient surface temperature measurements. While traditional thermography relies on pixel-wise 1D approximations that neglect lateral diffusion, and soft-constrained Physics-Informed Neural Networks (PINNs) often fail in transient diffusion scenarios due to gradient stiffness, NeFTY parameterizes the 3D diffusivity field as a continuous neural field optimized through a rigorous numerical solver. By leveraging a differentiable physics solver, our approach enforces thermodynamic laws as hard constraints while maintaining the memory efficiency required for high-resolution 3D tomography. Our discretize-then-optimize paradigm effectively mitigates the spectral bias and ill-posedness inherent in inverse heat conduction, enabling the recovery of subsurface defects at arbitrary scales. Experimental validation on synthetic data demonstrates that NeFTY significantly improves the accuracy of subsurface defect localization over baselines. Additional details at this https URL 

---
# V2M-Zero: Zero-Pair Time-Aligned Video-to-Music Generation 

**Authors**: Yan-Bo Lin, Jonah Casebeer, Long Mai, Aniruddha Mahapatra, Gedas Bertasius, Nicholas J. Bryan  

**Link**: [PDF](https://arxiv.org/pdf/2603.11042)  

**Abstract**: Generating music that temporally aligns with video events is challenging for existing text-to-music models, which lack fine-grained temporal control. We introduce V2M-Zero, a zero-pair video-to-music generation approach that outputs time-aligned music for video. Our method is motivated by a key observation: temporal synchronization requires matching when and how much change occurs, not what changes. While musical and visual events differ semantically, they exhibit shared temporal structure that can be captured independently within each modality. We capture this structure through event curves computed from intra-modal similarity using pretrained music and video encoders. By measuring temporal change within each modality independently, these curves provide comparable representations across modalities. This enables a simple training strategy: fine-tune a text-to-music model on music-event curves, then substitute video-event curves at inference without cross-modal training or paired data. Across OES-Pub, MovieGenBench-Music, and AIST++, V2M-Zero achieves substantial gains over paired-data baselines: 5-21% higher audio quality, 13-15% better semantic alignment, 21-52% improved temporal synchronization, and 28% higher beat alignment on dance videos. We find similar results via a large crowd-source subjective listening test. Overall, our results validate that temporal alignment through within-modality features, rather than paired cross-modal supervision, is effective for video-to-music generation. Results are available at this https URL 

---
# Instruction set for the representation of graphs 

**Authors**: Ezequiel Lopez-Rubio, Mario Pascual-Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2603.11039)  

**Abstract**: We present IsalGraph, a method for representing the structure of any finite, simple graph as a compact string over a nine-character instruction alphabet. The encoding is executed by a small virtual machine comprising a sparse graph, a circular doubly-linked list (CDLL) of graph-node references, and two traversal pointers. Instructions either move a pointer through the CDLL or insert a node or edge into the graph. A key design property is that every string over the alphabet decodes to a valid graph, with no invalid states reachable. A greedy \emph{GraphToString} algorithm encodes any connected graph into a string in time polynomial in the number of nodes; an exhaustive-backtracking variant produces a canonical string by selecting the lexicographically smallest shortest string across all starting nodes and all valid traversal orders. We evaluate the representation on five real-world graph benchmark datasets (IAM Letter LOW/MED/HIGH, LINUX, and AIDS) and show that the Levenshtein distance between IsalGraph strings correlates strongly with graph edit distance (GED). Together, these properties make IsalGraph strings a compact, isomorphism-invariant, and language-model-compatible sequential encoding of graph structure, with direct applications in graph similarity search, graph generation, and graph-conditioned language modelling 

---
# Does AI See like Art Historians? Interpreting How Vision Language Models Recognize Artistic Style 

**Authors**: Marvin Limpijankit, Milad Alshomary, Yassin Oulad Daoud, Amith Ananthram, Tim Trombley, Elias Stengel-Eskin, Mohit Bansal, Noam M. Elcott, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2603.11024)  

**Abstract**: VLMs have become increasingly proficient at a range of computer vision tasks, such as visual question answering and object detection. This includes increasingly strong capabilities in the domain of art, from analyzing artwork to generation of art. In an interdisciplinary collaboration between computer scientists and art historians, we characterize the mechanisms underlying VLMs' ability to predict artistic style and assess the extent to which they align with the criteria art historians use to reason about artistic style. We employ a latent-space decomposition approach to identify concepts that drive art style prediction and conduct quantitative evaluations, causal analysis and assessment by art historians. Our findings indicate that 73% of the extracted concepts are judged by art historians to exhibit a coherent and semantically meaningful visual feature and 90% of concepts used to predict style of a given artwork were judged relevant. In cases where an irrelevant concept was used to successfully predict style, art historians identified possible reasons for its success; for example, the model might "understand" a concept in more formal terms, such as dark/light contrasts. 

---
# RCTs & Human Uplift Studies: Methodological Challenges and Practical Solutions for Frontier AI Evaluation 

**Authors**: Patricia Paskov, Kevin Wei, Shen Zhou Hong, Dan Bateyko, Xavier Roberts-Gaal, Carson Ezell, Gailius Praninskas, Valerie Chen, Umang Bhatt, Ella Guest  

**Link**: [PDF](https://arxiv.org/pdf/2603.11001)  

**Abstract**: Human uplift studies - or studies that measure AI effects on human performance relative to a status quo, typically using randomized controlled trial (RCT) methodology - are increasingly used to inform deployment, governance, and safety decisions for frontier AI systems. While the methods underlying these studies are well-established, their interaction with the distinctive properties of frontier AI systems remains underexamined, particularly when results are used to inform high-stakes decisions. We present findings from interviews with 16 expert practitioners with experience conducting human uplift studies in domains including biosecurity, cybersecurity, education, and labor. Across interviews, experts described a recurring tension between standard causal inference assumptions and the object of study itself. Rapidly evolving AI systems, shifting baselines, heterogeneous and changing user proficiency, and porous real-world settings strain assumptions underlying internal, external, and construct validity, complicating the interpretation and appropriate use of uplift evidence. We synthesize these challenges across key stages of the human uplift research lifecycle and map them to practitioner-reported solutions, clarifying both the limits and the appropriate uses of evidence from human uplift studies in high-stakes decision-making. 

---
# Artificial Intelligence as a Catalyst for Innovation in Software Engineering 

**Authors**: Carlos Alberto Fernández-y-Fernández, Jorge R. Aguilar-Cisneros  

**Link**: [PDF](https://arxiv.org/pdf/2603.10994)  

**Abstract**: The rapid evolution and inherent complexity of modern software requirements demand highly flexible and responsive development methodologies. While Agile frameworks have become the industry standard for prioritizing iteration, collaboration, and adaptability, software development teams continue to face persistent challenges in managing constantly evolving requirements and maintaining product quality under tight deadlines. This article explores the intersection of Artificial Intelligence (AI) and Software Engineering (SE), to analyze how AI serves as a powerful catalyst for enhancing agility and fostering innovation. The research combines a comprehensive review of existing literature with an empirical study, utilizing a survey directed at Software Engineering professionals to assess the perception, adoption, and impact of AI-driven tools. Key findings reveal that the integration of AI (specifically through Machine Learning (ML) and Natural Language Processing (NLP) )facilitates the automation of tedious tasks, from requirement management to code generation and testing . This paper demonstrates that AI not only optimizes current Agile practices but also introduces new capabilities essential for sustaining quality, speed, and innovation in the future landscape of software development. 

---
# GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations 

**Authors**: Boyuan Chen, Minghao Shao, Siddharth Garg, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2603.10978)  

**Abstract**: Vision Language Models (VLMs) exhibit persistent hallucinations in counting tasks, with accuracy substantially lower than other visual reasoning tasks (excluding sentiment). This phenomenon persists even in state-of-the-art reasoning-capable VLMs. Conversely, CNN-based object detection models (ODMs) such as YOLO excel at spatial localization and instance counting with minimal computational overhead. We propose GroundCount, a framework that augments VLMs with explicit spatial grounding from ODMs to mitigate counting hallucinations. In the best case, our prompt-based augmentation strategy achieves 81.3% counting accuracy on the best-performing model (Ovis2.5-2B) - a 6.6pp improvement - while reducing inference time by 22% through elimination of hallucination-driven reasoning loops for stronger models. We conduct comprehensive ablation studies demonstrating that positional encoding is a critical component, being beneficial for stronger models but detrimental for weaker ones. Confidence scores, by contrast, introduce noise for most architectures and their removal improves performance in four of five evaluated models. We further evaluate feature-level fusion architectures, finding that explicit symbolic grounding via structured prompts outperforms implicit feature fusion despite sophisticated cross-attention mechanisms. Our approach yields consistent improvements across four of five evaluated VLM architectures (6.2--7.5pp), with one architecture exhibiting degraded performance due to incompatibility between its iterative reflection mechanisms and structured prompts. These results suggest that counting failures stem from fundamental spatial-semantic integration limitations rather than architecture-specific deficiencies, while highlighting the importance of architectural compatibility in augmentation strategies. 

---
# Contact Coverage-Guided Exploration for General-Purpose Dexterous Manipulation 

**Authors**: Zixuan Liu, Ruoyi Qiao, Chenrui Tie, Xuanwei Liu, Yunfan Lou, Chongkai Gao, Zhixuan Xu, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10971)  

**Abstract**: Deep Reinforcement learning (DRL) has achieved remarkable success in domains with well-defined reward structures, such as Atari games and locomotion. In contrast, dexterous manipulation lacks general-purpose reward formulations and typically depends on task-specific, handcrafted priors to guide hand-object interactions. We propose Contact Coverage-Guided Exploration (CCGE), a general exploration method designed for general-purpose dexterous manipulation tasks. CCGE represents contact state as the intersection between object surface points and predefined hand keypoints, encouraging dexterous hands to discover diverse and novel contact patterns, namely which fingers contact which object regions. It maintains a contact counter conditioned on discretized object states obtained via learned hash codes, capturing how frequently each finger interacts with different object regions. This counter is leveraged in two complementary ways: (1) to assign a count-based contact coverage reward that promotes exploration of novel contact patterns, and (2) an energy-based reaching reward that guides the agent toward under-explored contact regions. We evaluate CCGE on a diverse set of dexterous manipulation tasks, including cluttered object singulation, constrained object retrieval, in-hand reorientation, and bimanual manipulation. Experimental results show that CCGE substantially improves training efficiency and success rates over existing exploration methods, and that the contact patterns learned with CCGE transfer robustly to real-world robotic systems. Project page is this https URL. 

---
# Safe RLHF Beyond Expectation: Stochastic Dominance for Universal Spectral Risk Control 

**Authors**: Yaswanth Chittepu, Ativ Joshi, Rajarshi Bhattacharjee, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2603.10938)  

**Abstract**: Safe Reinforcement Learning from Human Feedback (RLHF) typically enforces safety through expected cost constraints, but the expectation captures only a single statistic of the cost distribution and fails to account for distributional uncertainty, particularly under heavy tails or rare catastrophic events. This limitation is problematic when robustness and risk sensitivity are critical. Stochastic dominance offers a principled alternative by comparing entire cost distributions rather than just their averages, enabling direct control over tail risks and potential out-of-distribution failures that expectation-based constraints may overlook. In this work, we propose Risk-sensitive Alignment via Dominance (RAD), a novel alignment framework that replaces scalar expected cost constraints with First-Order Stochastic Dominance (FSD) constraints. We operationalize this constraint by comparing the target policy's cost distribution to that of a reference policy within an Optimal Transport (OT) framework, using entropic regularization and Sinkhorn iterations to obtain a differentiable and computationally efficient objective for stable end-to-end optimization. Furthermore, we introduce quantile-weighted FSD constraints and show that weighted FSD universally controls a broad class of Spectral Risk Measures (SRMs), so that improvements under weighted dominance imply guaranteed improvements in the corresponding spectral risk. This provides a principled mechanism for tuning a model's risk profile via the quantile weighting function. Empirical results demonstrate that RAD improves harmlessness over baselines while remaining competitive in helpfulness, and exhibits greater robustness on out-of-distribution harmlessness evaluations. 

---
# Historical Consensus: Preventing Posterior Collapse via Iterative Selection of Gaussian Mixture Priors 

**Authors**: Zegu Zhang, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10935)  

**Abstract**: Variational autoencoders (VAEs) frequently suffer from posterior collapse, where latent variables become uninformative and the approximate posterior degenerates to the prior. Recent work has characterized this phenomenon as a phase transition governed by the spectral properties of the data covariance matrix. In this paper, we propose a fundamentally different approach: instead of avoiding collapse through architectural constraints or hyperparameter tuning, we eliminate the possibility of collapse altogether by leveraging the multiplicity of Gaussian mixture model (GMM) clusterings. We introduce Historical Consensus Training, an iterative selection procedure that progressively refines a set of candidate GMM priors through alternating optimization and selection. The key insight is that models trained to satisfy multiple distinct clustering constraints develop a historical barrier -- a region in parameter space that remains stable even when subsequently trained with a single objective. We prove that this barrier excludes the collapsed solution, and demonstrate through extensive experiments on synthetic and real-world datasets that our method achieves non-collapsed representations regardless of decoder variance or regularization strength. Our approach requires no explicit stability conditions (e.g., $\sigma^{\prime 2} < \lambda_{\max}$) and works with arbitrary neural architectures. The code is available at this https URL. 

---
# When Fine-Tuning Fails and when it Generalises: Role of Data Diversity and Mixed Training in LLM-based TTS 

**Authors**: Anupam Purwar, Aditya Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2603.10904)  

**Abstract**: Large language models are increasingly adopted as semantic backbones for neural text-to-speech systems. However, frozen LLM representations are insufficient for modeling speaker specific acoustic and perceptual characteristics. Our experiments involving fine tuning of the Language Model backbone of TTS show promise in improving the voice consistency and Signal to Noise ratio SNR in voice cloning task. Across multiple speakers LoRA finetuning consistently outperforms the non-finetuned base Qwen-0.5B model across three complementary dimensions of speech quality. First, perceptual quality improves significantly with DNS-MOS gains of up to 0.42 points for speakers whose training data exhibits sufficient acoustic variability. Second, speaker fidelity improves for all evaluated speakers with consistent increases in voice similarity indicating that LoRA effectively adapts speaker identity representations without degrading linguistic modeling. Third, signal level quality improves in most cases with signal to noise ratio increasing by as much as 34 percent. Crucially these improvements are strongly governed by the characteristics of the training data. Speakers with high variability in acoustic energy and perceptual quality achieve simultaneous gains in DNS-MOS voice similarity and SNR. Overall this work establishes that LoRA finetuning is not merely a parameter efficient optimization technique but an effective mechanism for better speaker level adaptation in compact LLM-based TTS systems. When supported by sufficiently diverse training data LoRA adapted Qwen-0.5B consistently surpasses its frozen base model in perceptual quality speaker similarity with low latency using GGUF model hosted in quantized form. 

---
# LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation 

**Authors**: Jinwoo Ahn, Ingyu Seong, Akhil Kedia, Junhan Kim, Hyemi Jang, Kangwook Lee, Yongkweon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2603.10899)  

**Abstract**: Transformer-based large language models (LLMs) rely on key-value (KV) caching to avoid redundant computation during autoregressive inference. While this mechanism greatly improves efficiency, the cache size grows linearly with the input sequence length, quickly becoming a bottleneck for long-context tasks. Existing solutions mitigate this problem by evicting prompt KV that are deemed unimportant, guided by estimated importance scores. Notably, a recent line of work proposes to improve eviction quality by "glimpsing into the future", in which a draft generator produces a surrogate future response approximating the target model's true response, and this surrogate is subsequently used to estimate the importance of cached KV more accurately. However, these approaches rely on computationally expensive draft generation, which introduces substantial prefilling overhead and limits their practicality in real-world deployment. To address this challenge, we propose LookaheadKV, a lightweight eviction framework that leverages the strength of surrogate future response without requiring explicit draft generation. LookaheadKV augments transformer layers with parameter-efficient modules trained to predict true importance scores with high accuracy. Our design ensures negligible runtime overhead comparable to existing inexpensive heuristics, while achieving accuracy superior to more costly approximation methods. Extensive experiments on long-context understanding benchmarks, across a wide range of models, demonstrate that our method not only outperforms recent competitive baselines in various long-context understanding tasks, but also reduces the eviction cost by up to 14.5x, leading to significantly faster time-to-first-token. Our code is available at this https URL. 

---
# Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models 

**Authors**: Yixiu Mao, Yun Qu, Qi Wang, Heming Zou, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.10887)  

**Abstract**: Reinforcement learning (RL) finetuning has become a key technique for enhancing the reasoning abilities of large language models (LLMs). However, its effectiveness critically depends on the selection of training data. Recent advances underscore the importance of online prompt selection methods, which typically concentrate training on partially solved or moderately challenging examples under the current policy, thereby yielding more effective model updates. While significantly accelerating RL finetuning in terms of training steps, they also incur substantial computational overhead by requiring extensive LLM rollouts over large candidate batches to identify informative samples, an expense that can outweigh the finetuning process itself. To address this challenge, this work proposes Dynamics-Predictive Sampling (DPS), which online predicts and selects informative prompts by inferring their learning dynamics prior to costly rollouts. Specifically, we introduce a new perspective by modeling each prompt's solving progress during RL finetuning as a dynamical system, where the extent of solving is represented as the state and the transition is characterized by a hidden Markov model. Using historical rollout reward signals, we perform online Bayesian inference to estimate evolving state distributions, and the inference outcome provides a predictive prior for efficient prompt selection without rollout-intensive filtering. Empirical results across diverse reasoning tasks, including mathematics, planning, and visual geometry, demonstrate that DPS substantially reduces redundant rollouts, accelerates the training process, and achieves superior reasoning performance. 

---
# Continuous Diffusion Transformers for Designing Synthetic Regulatory Elements 

**Authors**: Jonathan Liu, Kia Ghods  

**Link**: [PDF](https://arxiv.org/pdf/2603.10885)  

**Abstract**: We present a parameter-efficient Diffusion Transformer (DiT) for generating 200bp cell-type-specific regulatory DNA sequences. By replacing the U-Net backbone of DNA-Diffusion with a transformer denoiser equipped with a 2D CNN input encoder, our model matches the U-Net's best validation loss in 13 epochs (60$\times$ fewer) and converges 39% lower, while reducing memorization from 5.3% to 1.7% of generated sequences aligning to training data via BLAT. Ablations show the CNN encoder is essential: without it, validation loss increases 70% regardless of positional embedding choice. We further apply DDPO finetuning using Enformer as a reward model, achieving a 38$\times$ improvement in predicted regulatory activity. Cross-validation against DRAKES on an independent prediction task confirms that improvements reflect genuine regulatory signal rather than reward model overfitting. 

---
# An Extreme Multi-label Text Classification (XMTC) Library Dataset: What if we took "Use of Practical AI in Digital Libraries" seriously? 

**Authors**: Jennifer D'Souza, Sameer Sadruddin, Maximilian Kähler, Andrea Salfinger, Luca Zaccagna, Francesca Incitti, Lauro Snidaro, Osma Suominen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10876)  

**Abstract**: Subject indexing is vital for discovery but hard to sustain at scale and across languages. We release a large bilingual (English/German) corpus of catalog records annotated with the Integrated Authority File (GND), plus a machine-actionable GND taxonomy. The resource enables ontology-aware multi-label classification, mapping text to authority terms, and agent-assisted cataloging with reproducible, authority-grounded evaluation. We provide a brief statistical profile and qualitative error analyses of three systems. We invite the community to assess not only accuracy but usefulness and transparency, toward authority-anchored AI co-pilots that amplify catalogers' work. 

---
# GRACE: A Unified 2D Multi-Robot Path Planning Simulator & Benchmark for Grid, Roadmap, And Continuous Environments 

**Authors**: Chuanlong Zang, Anna Mannucci, Isabelle Barz, Philipp Schillinger, Florian Lier, Wolfgang Hönig  

**Link**: [PDF](https://arxiv.org/pdf/2603.10858)  

**Abstract**: Advancing Multi-Agent Pathfinding (MAPF) and Multi-Robot Motion Planning (MRMP) requires platforms that enable transparent, reproducible comparisons across modeling choices. Existing tools either scale under simplifying assumptions (grids, homogeneous agents) or offer higher fidelity with less comparable instrumentation. We present GRACE, a unified 2D simulator+benchmark that instantiates the same task at multiple abstraction levels (grid, roadmap, continuous) via explicit, reproducible operators and a common evaluation protocol. Our empirical results on public maps and representative planners enable commensurate comparisons on a shared instance set. Furthermore, we quantify the expected representation-fidelity trade-offs (MRMP solves instances at higher fidelity but lower speed, while grid/roadmap planners scale farther). By consolidating representation, execution, and evaluation, GRACE thereby aims to make cross-representation studies more comparable and provides a means to advance multi-robot planning research and its translation to practice. 

---
# $V_{0.5}$: Generalist Value Model as a Prior for Sparse RL Rollouts 

**Authors**: Yi-Kai Zhang, Yueqing Sun, Hongyan Hao, Qi Gu, Xunliang Cai, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.10848)  

**Abstract**: In Reinforcement Learning with Verifiable Rewards (RLVR), constructing a robust advantage baseline is critical for policy gradients, effectively guiding the policy model to reinforce desired behaviors. Recent research has introduced Generalist Value Models (such as $V_0$), which achieve pre-trained value estimation by explicitly encoding model capabilities in-context, eliminating the need to synchronously update the value model alongside the policy model. In this paper, we propose $V_{0.5}$, which adaptively fuses the baseline predicted by such value model (acting as a prior) with the empirical mean derived from sparse rollouts. This constructs a robust baseline that balances computational efficiency with extremely low variance. Specifically, we introduce a real-time statistical testing and dynamic budget allocation. This balances the high variance caused by sparse sampling against the systematic bias (or hallucinations) inherent in the value model's prior. By constructing a hypothesis test to evaluate the prior's reliability in real-time, the system dynamically allocates additional rollout budget on demand. This mechanism minimizes the baseline estimator's Mean Squared Error (MSE), guaranteeing stable policy gradients, even under extreme sparsity with a group size of 4. Extensive evaluations across six mathematical reasoning benchmarks demonstrate that $V_{0.5}$ significantly outperforms GRPO and DAPO, achieving faster convergence and over some 10% performance improvement. 

---
# Semantic Landmark Particle Filter for Robot Localisation in Vineyards 

**Authors**: Rajitha de Silva, Jonathan Cox, James R. Heselden, Marija Popović, Cesar Cadena, Riccardo Polvara  

**Link**: [PDF](https://arxiv.org/pdf/2603.10847)  

**Abstract**: Reliable localisation in vineyards is hindered by row-level perceptual aliasing: parallel crop rows produce nearly identical LiDAR observations, causing geometry-only and vision-based SLAM systems to converge towards incorrect corridors, particularly during headland transitions. We present a Semantic Landmark Particle Filter (SLPF) that integrates trunk and pole landmark detections with 2D LiDAR within a probabilistic localisation framework. Detected trunks are converted into semantic walls, forming structural row boundaries embedded in the measurement model to improve discrimination between adjacent rows. GNSS is incorporated as a lightweight prior that stabilises localisation when semantic observations are sparse.
Field experiments in a 10-row vineyard demonstrate consistent improvements over geometry-only (AMCL), vision-based (RTAB-Map), and GNSS baselines. Compared to AMCL, SLPF reduces Absolute Pose Error by 22% and 65% across two traversal directions; relative to a NoisyGNSS baseline, APE decreases by 65% and 61%. Row correctness improves from 0.67 to 0.73, while mean cross-track error decreases from 1.40 m to 1.26 m. These results show that embedding row-level structural semantics within the measurement model enables robust localisation in highly repetitive outdoor agricultural environments. 

---
# Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis 

**Authors**: Yujie Zheng, Zhuo Li, Shengtao Zhang, Hanjing Wang, Junjie Sheng, Jiaqian Wang, Junchi Yan, Weinan Zhang, Ying Wen, Bo Tang, Muning Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10846)  

**Abstract**: Deploying Large Language Models to data-scarce programming domains poses significant challenges, particularly for kernel synthesis on emerging Domain-Specific Architectures where a "Data Wall" limits available training data. While models excel on data-rich platforms like CUDA, they suffer catastrophic performance drops on data-scarce ecosystems such as NPU programming. To overcome this cold-start barrier without expensive fine-tuning, we introduce EvoKernel, a self-evolving agentic framework that automates the lifecycle of kernel synthesis from initial drafting to continual refining. EvoKernel addresses this by formulating the synthesis process as a memory-based reinforcement learning task. Through a novel value-driven retrieval mechanism, it learns stage-specific Q-values that prioritize experiences based on their contribution to the current objective, whether bootstrapping a feasible draft or iteratively refining latency. Furthermore, by enabling cross-task memory sharing, the agent generalizes insights from simple to complex operators. By building an NPU variant of KernelBench and evaluating on it, EvoKernel improves frontier models' correctness from 11.0% to 83.0% and achieves a median speedup of 3.60x over initial drafts through iterative refinement. This demonstrates that value-guided experience accumulation allows general-purpose models to master the kernel synthesis task on niche hardware ecosystems. Our official page is available at this https URL. 

---
# Human Presence Detection via Wi-Fi Range-Filtered Doppler Spectrum on Commodity Laptops 

**Authors**: Jessica Sanson, Rahul C. Shah, Valerio Frascolla  

**Link**: [PDF](https://arxiv.org/pdf/2603.10845)  

**Abstract**: Human Presence Detection (HPD) is key to enable intelligent power management and security features in everyday devices. In this paper we propose the first HPD solution that leverages monostatic Wi-Fi sensing and detects user position using only the built-in Wi-Fi hardware of a device, with no need for external devices, access points, or additional sensors. In contrast, existing HPD solutions for laptops require external dedicated sensors which add cost and complexity, or rely on camera-based approaches that introduce significant privacy concerns. We herewith introduce the Range-Filtered Doppler Spectrum (RF-DS), a novel Wi-Fi sensing technique for presence estimation that enables both range-selective and temporally windowed detection of user presence. By applying targeted range-area filtering in the Channel Impulse Response (CIR) domain before Doppler analysis, our method focuses processing on task-relevant spatial zones, significantly reducing computational complexity. In addition, the use of temporal windows in the spectrum domain provides greater estimator stability compared to conventional 2D Range-Doppler detectors. Furthermore, we propose an adaptive multi-rate processing framework that dynamically adjusts Channel State Information (CSI) sampling rates-operating at low frame rates (10Hz) during idle periods and high rates (100Hz) only when motion is detected. To our knowledge, this is the first low-complexity solution for occupancy detection using monostatic Wi-Fi sensing on a built-in Wi-Fi network interface controller (NIC) of a commercial off-the-shelf laptop that requires no external network infrastructure or specialized sensors. Our solution can scale across different environments and devices without calibration or retraining. 

---
# On the Reliability of Cue Conflict and Beyond 

**Authors**: Pum Jun Kim, Seung-Ah Lee, Seongho Park, Dongyoon Han, Jaejun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2603.10834)  

**Abstract**: Understanding how neural networks rely on visual cues offers a human-interpretable view of their internal decision processes. The cue-conflict benchmark has been influential in probing shape-texture preference and in motivating the insight that stronger, human-like shape bias is often associated with improved in-domain performance. However, we find that the current stylization-based instantiation can yield unstable and ambiguous bias estimates. Specifically, stylization may not reliably instantiate perceptually valid and separable cues nor control their relative informativeness, ratio-based bias can obscure absolute cue sensitivity, and restricting evaluation to preselected classes can distort model predictions by ignoring the full decision space. Together, these factors can confound preference with cue validity, cue balance, and recognizability artifacts. We introduce REFINED-BIAS, an integrated dataset and evaluation framework for reliable and interpretable shape-texture bias diagnosis. REFINED-BIAS constructs balanced, human- and model- recognizable cue pairs using explicit definitions of shape and texture, and measures cue-specific sensitivity over the full label space via a ranking-based metric, enabling fairer cross-model comparisons. Across diverse training regimes and architectures, REFINED-BIAS enables fairer cross-model comparison, more faithful diagnosis of shape and texture biases, and clearer empirical conclusions, resolving inconsistencies that prior cue-conflict evaluations could not reliably disambiguate. 

---
# BALD-SAM: Disagreement-based Active Prompting in Interactive Segmentation 

**Authors**: Prithwijit Chowdhury, Mohit Prabhushankar, Ghassan AlRegib  

**Link**: [PDF](https://arxiv.org/pdf/2603.10828)  

**Abstract**: The Segment Anything Model (SAM) has revolutionized interactive segmentation through spatial prompting. While existing work primarily focuses on automating prompts in various settings, real-world annotation workflows involve iterative refinement where annotators observe model outputs and strategically place prompts to resolve ambiguities. Current pipelines typically rely on the annotator's visual assessment of the predicted mask quality. We postulate that a principled approach for automated interactive prompting is to use a model-derived criterion to identify the most informative region for the next prompt. In this work, we establish active prompting: a spatial active learning approach where locations within images constitute an unlabeled pool and prompts serve as queries to prioritize information-rich regions, increasing the utility of each interaction. We further present BALD-SAM: a principled framework adapting Bayesian Active Learning by Disagreement (BALD) to spatial prompt selection by quantifying epistemic uncertainty. To do so, we freeze the entire model and apply Bayesian uncertainty modeling only to a small learned prediction head, making intractable uncertainty estimation practical for large multi-million parameter foundation models. Across 16 datasets spanning natural, medical, underwater, and seismic domains, BALD-SAM demonstrates strong cross-domain performance, ranking first or second on 14 of 16 benchmarks. We validate these gains through a comprehensive ablation suite covering 3 SAM backbones and 35 Laplace posterior configurations, amounting to 38 distinct ablation settings. Beyond strong average performance, BALD-SAM surpasses human prompting and, in several categories, even oracle prompting, while consistently outperforming one-shot baselines in final segmentation quality, particularly on thin and structurally complex objects. 

---
# Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation 

**Authors**: Thomas Thebaud, Yuzhe Wang, Laureano Moro-Velazquez, Jesus Villalba-Lopez, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2603.10827)  

**Abstract**: Speech-aware large language models (LLMs) can accept speech inputs, yet their training objectives largely emphasize linguistic content or specific fields such as emotions or the speaker's gender, leaving it unclear whether they encode speaker identity. First, we propose a model-agnostic scoring protocol that produces continuous verification scores for both API-only and open-weight models, using confidence scores or log-likelihood ratios from the Yes/No token probabilities. Using this protocol, we benchmark recent speech-aware LLMs and observe weak speaker discrimination (EERs above 20% on VoxCeleb1). Second, we introduce a lightweight augmentation that equips an LLM with ASV capability by injecting frozen ECAPA-TDNN speaker embeddings through a learned projection and training only LoRA adapters. On TinyLLaMA-1.1B, the resulting ECAPA-LLM achieves 1.03% EER on VoxCeleb1-E, approaching a dedicated speaker verification system while preserving a natural-language interface. 

---
# Protein Counterfactuals via Diffusion-Guided Latent Optimization 

**Authors**: Weronika Kłos, Sidney Bender, Lukas Kades  

**Link**: [PDF](https://arxiv.org/pdf/2603.10811)  

**Abstract**: Deep learning models can predict protein properties with unprecedented accuracy but rarely offer mechanistic insight or actionable guidance for engineering improved variants. When a model flags an antibody as unstable, the protein engineer is left without recourse: which mutations would rescue stability while preserving function? We introduce Manifold-Constrained Counterfactual Optimization for Proteins (MCCOP), a framework that computes minimal, biologically plausible sequence edits that flip a model's prediction to a desired target state. MCCOP operates in a continuous joint sequence-structure latent space and employs a pretrained diffusion model as a manifold prior, balancing three objectives: validity (achieving the target property), proximity (minimizing mutations), and plausibility (producing foldable proteins). We evaluate MCCOP on three protein engineering tasks - GFP fluorescence rescue, thermodynamic stability enhancement, and E3 ligase activity recovery - and show that it generates sparser, more plausible counterfactuals than both discrete and continuous baselines. The recovered mutations align with known biophysical mechanisms, including chromophore packing and hydrophobic core consolidation, establishing MCCOP as a tool for both model interpretation and hypothesis-driven protein design. Our code is publicly available at this http URL. 

---
# Risk-Adjusted Harm Scoring for Automated Red Teaming for LLMs in Financial Services 

**Authors**: Fabrizio Dimino, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2603.10807)  

**Abstract**: The rapid adoption of large language models (LLMs) in financial services introduces new operational, regulatory, and security risks. Yet most red-teaming benchmarks remain domain-agnostic and fail to capture failure modes specific to regulated BFSI settings, where harmful behavior can be elicited through legally or professionally plausible framing. We propose a risk-aware evaluation framework for LLM security failures in Banking, Financial Services, and Insurance (BFSI), combining a domain-specific taxonomy of financial harms, an automated multi-round red-teaming pipeline, and an ensemble-based judging protocol. We introduce the Risk-Adjusted Harm Score (RAHS), a risk-sensitive metric that goes beyond success rates by quantifying the operational severity of disclosures, accounting for mitigation signals, and leveraging inter-judge agreement. Across diverse models, we find that higher decoding stochasticity and sustained adaptive interaction not only increase jailbreak success, but also drive systematic escalation toward more severe and operationally actionable financial disclosures. These results expose limitations of single-turn, domain-agnostic security evaluation and motivate risk-sensitive assessment under prolonged adversarial pressure for real-world BFSI deployment. 

---
# Towards Intelligent Spectrum Management: Spectrum Demand Estimation Using Graph Neural Networks 

**Authors**: Mohamad Alkadamani, Amir Ghasemi, Halim Yanikomeroglu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10802)  

**Abstract**: The growing demand for wireless connectivity, combined with limited spectrum resources, calls for more efficient spectrum management. Spectrum sharing is a promising approach; however, regulators need accurate methods to characterize demand dynamics and guide allocation decisions. This paper builds and validates a spectrum demand proxy from public deployment records and uses a graph attention network in a hierarchical, multi-resolution setup (HR-GAT) to estimate spectrum demand at fine spatial scales. The model captures both neighborhood effects and cross-scale patterns, reducing spatial autocorrelation and improving generalization. Evaluated across five Canadian cities and against eight competitive baselines, HR-GAT reduces median RMSE by roughly 21% relative to the best alternative and lowers residual spatial bias. The resulting demand maps are regulator-accessible and support spectrum sharing and spectrum allocation in wireless networks. 

---
# AI-Enhanced Spatial Cellular Traffic Demand Prediction with Contextual Clustering and Error Correction for 5G/6G Planning 

**Authors**: Mohamad Alkadamani, Colin Brown, Halim Yanikomeroglu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10800)  

**Abstract**: Accurate spatial prediction of cellular traffic demand is essential for 5G NR capacity planning, network densification, and data-driven 6G planning. Although machine learning can fuse heterogeneous geospatial and socio-economic layers to estimate fine-grained demand maps, spatial autocorrelation can cause neighborhood leakage under naive train/test splits, inflating accuracy and weakening planning reliability. This paper presents an AI-driven framework that reduces leakage and improves spatial generalization via a context-aware two-stage splitting strategy with residual spatial error correction. Experiments using crowdsourced usage indicators across five major Canadian cities show consistent mean absolute error (MAE) reductions relative to location-only clustering, supporting more reliable bandwidth provisioning and evidence-based spectrum planning and sharing assessments. 

---
# Taking Shortcuts for Categorical VQA Using Super Neurons 

**Authors**: Pierre Musacchio, Jaeyi Jeong, Dahun Kim, Jaesik Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.10781)  

**Abstract**: Sparse Attention Vectors (SAVs) have emerged as an excellent training-free alternative to supervised finetuning or low-rank adaptation to improve the performance of Vision Language Models (VLMs). At their heart, SAVs select a few accurate attention heads for a task of interest and use them as classifiers, rather than relying on the model's prediction. In a similar spirit, we find that directly probing the raw activations of the VLM, in the form of scalar values, is sufficient to yield accurate classifiers on diverse visually grounded downstream tasks. Shifting focus from attention vectors to scalar activations dramatically increases the search space for accurate parameters, allowing us to find more discriminative neurons immediately from the first generated token. We call such activations Super Neurons (SNs). In this probing setting, we discover that enough SNs appear in the shallower layers of the large language model to allow for extreme early exiting from the first layer of the model at the first generated token. Compared to the original network, SNs robustly improve the classification performance while achieving a speedup of up to 5.10x. 

---
# Deep Randomized Distributed Function Computation (DeepRDFC): Neural Distributed Channel Simulation 

**Authors**: Didrik Bergström, Onur Günlü  

**Link**: [PDF](https://arxiv.org/pdf/2603.10750)  

**Abstract**: The randomized distributed function computation (RDFC) framework, which unifies many cutting-edge distributed computation and learning applications, is considered. An autoencoder (AE) architecture is proposed to minimize the total variation distance between the probability distribution simulated by the AE outputs and an unknown target distribution, using only data samples. We illustrate significantly high RDFC performance with communication load gains from our AEs compared to data compression methods. Our designs establish deep learning-based RDFC methods and aim to facilitate the use of RDFC methods, especially when the amount of common randomness is limited and strong function computation guarantees are required. 

---
# CUPID: A Plug-in Framework for Joint Aleatoric and Epistemic Uncertainty Estimation with a Single Model 

**Authors**: Xinran Xu, Xiuyi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.10745)  

**Abstract**: Accurate estimation of uncertainty in deep learning is critical for deploying models in high-stakes domains such as medical diagnosis and autonomous decision-making, where overconfident predictions can lead to harmful outcomes. In practice, understanding the reason behind a model's uncertainty and the type of uncertainty it represents can support risk-aware decisions, enhance user trust, and guide additional data collection. However, many existing methods only address a single type of uncertainty or require modifications and retraining of the base model, making them difficult to adopt in real-world systems. We introduce CUPID (Comprehensive Uncertainty Plug-in estImation moDel), a general-purpose module that jointly estimates aleatoric and epistemic uncertainty without modifying or retraining the base model. CUPID can be flexibly inserted into any layer of a pretrained network. It models aleatoric uncertainty through a learned Bayesian identity mapping and captures epistemic uncertainty by analyzing the model's internal responses to structured perturbations. We evaluate CUPID across a range of tasks, including classification, regression, and out-of-distribution detection. The results show that it consistently delivers competitive performance while offering layer-wise insights into the origins of uncertainty. By making uncertainty estimation modular, interpretable, and model-agnostic, CUPID supports more transparent and trustworthy AI. Related code and data are available at this https URL. 

---
# Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning 

**Authors**: Artem Dvirniak, Evgeny Kushnir, Dmitrii Tarasov, Artem Iudin, Oleg Kiriukhin, Mikhail Pautov, Dmitrii Korzh, Oleg Y. Rogov  

**Link**: [PDF](https://arxiv.org/pdf/2603.10725)  

**Abstract**: The modern generative audio models can be used by an adversary in an unlawful manner, specifically, to impersonate other people to gain access to private information. To mitigate this issue, speech deepfake detection (SDD) methods started to evolve. Unfortunately, current SDD methods generally suffer from the lack of generalization to new audio domains and generators. More than that, they lack interpretability, especially human-like reasoning that would naturally explain the attribution of a given audio to the bona fide or spoof class and provide human-perceptible cues. In this paper, we propose HIR-SDD, a novel SDD framework that combines the strengths of Large Audio Language Models (LALMs) with the chain-of-thought reasoning derived from the novel proposed human-annotated dataset. Experimental evaluation demonstrates both the effectiveness of the proposed method and its ability to provide reasonable justifications for predictions. 

---
# UAV traffic scene understanding: A cross-spectral guided approach and a unified benchmark 

**Authors**: Yu Zhang, Zhicheng Zhao, Ze Luo, Chenglong Li, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10722)  

**Abstract**: Traffic scene understanding from unmanned aerial vehicle (UAV) platforms is crucial for intelligent transportation systems due to its flexible deployment and wide-area monitoring capabilities. However, existing methods face significant challenges in real-world surveillance, as their heavy reliance on optical imagery leads to severe performance degradation under adverse illumination conditions like nighttime and fog. Furthermore, current Visual Question Answering (VQA) models are restricted to elementary perception tasks, lacking the domain-specific regulatory knowledge required to assess complex traffic behaviors. To address these limitations, we propose a novel Cross-spectral Traffic Cognition Network (CTCNet) for robust UAV traffic scene understanding. Specifically, we design a Prototype-Guided Knowledge Embedding (PGKE) module that leverages high-level semantic prototypes from an external Traffic Regulation Memory (TRM) to anchor domain-specific knowledge into visual representations, enabling the model to comprehend complex behaviors and distinguish fine-grained traffic violations. Moreover, we develop a Quality-Aware Spectral Compensation (QASC) module that exploits the complementary characteristics of optical and thermal modalities to perform bidirectional context exchange, effectively compensating for degraded features to ensure robust representation in complex environments. In addition, we construct Traffic-VQA, the first large-scale optical-thermal infrared benchmark for cognitive UAV traffic understanding, comprising 8,180 aligned image pairs and 1.3 million question-answer pairs across 31 diverse types. Extensive experiments demonstrate that CTCNet significantly outperforms state-of-the-art methods in both cognition and perception scenarios. The dataset is available at this https URL. 

---
# Probabilistic Verification of Voice Anti-Spoofing Models 

**Authors**: Evgeny Kushnir, Alexandr Kozodaev, Dmitrii Korzh, Mikhail Pautov, Oleg Kiriukhin, Oleg Y. Rogov  

**Link**: [PDF](https://arxiv.org/pdf/2603.10713)  

**Abstract**: Recent advances in generative models have amplified the risk of malicious misuse of speech synthesis technologies, enabling adversaries to impersonate target speakers and access sensitive resources. Although speech deepfake detection has progressed rapidly, most existing countermeasures lack formal robustness guarantees or fail to generalize to unseen generation techniques. We propose PV-VASM, a probabilistic framework for verifying the robustness of voice anti-spoofing models (VASMs). PV-VASM estimates the probability of misclassification under text-to-speech (TTS), voice cloning (VC), and parametric signal transformations. The approach is model-agnostic and enables robustness verification against unseen speech synthesis techniques and input perturbations. We derive a theoretical upper bound on the error probability and validate the method across diverse experimental settings, demonstrating its effectiveness as a practical robustness verification tool. 

---
# AlphaFlowTSE: One-Step Generative Target Speaker Extraction via Conditional AlphaFlow 

**Authors**: Duojia Li, Shuhan Zhang, Zihan Qian, Wenxuan Wu, Shuai Wang, Qingyang Hong, Lin Li, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.10701)  

**Abstract**: In target speaker extraction (TSE), we aim to recover target speech from a multi-talker mixture using a short enrollment utterance as reference. Recent studies on diffusion and flow-matching generators have improved target-speech fidelity. However, multi-step sampling increases latency, and one-step solutions often rely on a mixture-dependent time coordinate that can be unreliable for real-world conversations. We present AlphaFlowTSE, a one-step conditional generative model trained with a Jacobian-vector product (JVP)-free AlphaFlow objective. AlphaFlowTSE learns mean-velocity transport along a mixture-to-target trajectory starting from the observed mixture, eliminating auxiliary mixing-ratio prediction, and stabilizes training by combining flow matching with an interval-consistency teacher-student target. Experiments on Libri2Mix and REAL-T confirm that AlphaFlowTSE improves target-speaker similarity and real-mixture generalization for downstream automatic speech recognition (ASR). 

---
# Structured Linked Data as a Memory Layer for Agent-Orchestrated Retrieval 

**Authors**: Andrea Volpini, Elie Raad, Beatrice Gamba, David Riccitelli  

**Link**: [PDF](https://arxiv.org/pdf/2603.10700)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems typically treat documents as flat text, ignoring the structured metadata and linked relationships that knowledge graphs provide. In this paper, we investigate whether structured linked data, specifically this http URL markup and dereferenceable entity pages served by a Linked Data Platform, can improve retrieval accuracy and answer quality in both standard and agentic RAG systems. We conduct a controlled experiment across four domains (editorial, legal, travel, e-commerce) using Vertex AI Vector Search 2.0 for retrieval and the Google Agent Development Kit (ADK) for agentic reasoning. Our experimental design tests seven conditions: three document representations (plain HTML, HTML with JSON-LD, and an enhanced agentic-optimized entity page) crossed with two retrieval modes (standard RAG and agentic RAG with multi-hop link traversal), plus an Enhanced+ condition that adds rich navigational affordances and entity interlinking. Our results reveal that while JSON-LD markup alone provides only modest improvements, our enhanced entity page format, incorporating this http URL-style agent instructions, breadcrumbs, and neural search capabilities, achieves substantial gains: +29.6% accuracy improvement for standard RAG and +29.8% for the full agentic pipeline. The Enhanced+ variant, with richer navigational affordances, achieves the highest absolute scores (accuracy: 4.85/5, completeness: 4.55/5), though the incremental gain over the base enhanced format is not statistically significant. We release our dataset, evaluation framework, and enhanced entity page templates to support reproducibility. 

---
# EvoSchema: Towards Text-to-SQL Robustness Against Schema Evolution 

**Authors**: Tianshu Zhang, Kun Qian, Siddhartha Sahai, Yuan Tian, Shaddy Garg, Huan Sun, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.10697)  

**Abstract**: Neural text-to-SQL models, which translate natural language questions (NLQs) into SQL queries given a database schema, have achieved remarkable performance. However, database schemas frequently evolve to meet new requirements. Such schema evolution often leads to performance degradation for models trained on static schemas. Existing work either mainly focuses on simply paraphrasing some syntactic or semantic mappings among NLQ, DB and SQL, or lacks a comprehensive and controllable way to investigate the model robustness issue under the schema evolution, which is insufficient when facing the increasingly complex and rich database schema changes in reality, especially in the LLM era. To address the challenges posed by schema evolution, we present EvoSchema, a comprehensive benchmark designed to assess and enhance the robustness of text-to-SQL systems under real-world schema changes. EvoSchema introduces a novel schema evolution taxonomy, encompassing ten perturbation types across columnlevel and table-level modifications, systematically simulating the dynamic nature of database schemas. Through EvoSchema, we conduct an in-depth evaluation spanning different open source and closed-source LLMs, revealing that table-level perturbations have a significantly greater impact on model performance compared to column-level changes. Furthermore, EvoSchema inspires the development of more resilient text-to-SQL systems, in terms of both model training and database design. The models trained on EvoSchema's diverse schema designs can force the model to distinguish the schema difference for the same questions to avoid learning spurious patterns, which demonstrate remarkable robustness compared to those trained on unperturbed data on average. This benchmark offers valuable insights into model behavior and a path forward for designing systems capable of thriving in dynamic, real-world environments. 

---
# RandMark: On Random Watermarking of Visual Foundation Models 

**Authors**: Anna Chistyakova, Mikhail Pautov  

**Link**: [PDF](https://arxiv.org/pdf/2603.10695)  

**Abstract**: Being trained on large and diverse datasets, visual foundation models (VFMs) can be fine-tuned to achieve remarkable performance and efficiency in various downstream computer vision tasks. The high computational cost of data collection and training makes these models valuable assets, which motivates some VFM owners to distribute them alongside a license to protect their intellectual property rights. In this paper, we propose an approach to ownership verification of visual foundation models that leverages a small encoder-decoder network to embed digital watermarks into an internal representation of a hold-out set of input images. The method is based on random watermark embedding, which makes the watermark statistics detectable in functional copies of the watermarked model. Both theoretically and experimentally, we demonstrate that the proposed method yields a low probability of false detection for non-watermarked models and a low probability of false misdetection for watermarked models. 

---
# Repurposing Backdoors for Good: Ephemeral Intrinsic Proofs for Verifiable Aggregation in Cross-silo Federated Learning 

**Authors**: Xian Qin, Xue Yang, Xiaohu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10692)  

**Abstract**: While Secure Aggregation (SA) protects update confidentiality in Cross-silo Federated Learning, it fails to guarantee aggregation integrity, allowing malicious servers to silently omit or tamper with updates. Existing verifiable aggregation schemes rely on heavyweight cryptography (e.g., ZKPs, HE), incurring computational costs that scale poorly with model size. In this paper, we propose a lightweight architecture that shifts from extrinsic cryptographic proofs to \textit{Intrinsic Proofs}. We repurpose backdoor injection to embed verification signals directly into model parameters. By harnessing Catastrophic Forgetting, these signals are robust for immediate verification yet ephemeral, naturally decaying to preserve final model utility. We design a randomized, single-verifier auditing framework compatible with SA, ensuring client anonymity and preventing signal collision without trusted third parties. Experiments on SVHN, CIFAR-10, and CIFAR-100 demonstrate high detection probabilities against malicious servers. Notably, our approach achieves over $1000\times$ speedup on ResNet-18 compared to cryptographic baselines, effectively scaling to large models. 

---
# Contract And Conquer: How to Provably Compute Adversarial Examples for a Black-Box Model? 

**Authors**: Anna Chistyakova, Mikhail Pautov  

**Link**: [PDF](https://arxiv.org/pdf/2603.10689)  

**Abstract**: Black-box adversarial attacks are widely used as tools to test the robustness of deep neural networks against malicious perturbations of input data aimed at a specific change in the output of the model. Such methods, although they remain empirically effective, usually do not guarantee that an adversarial example can be found for a particular model. In this paper, we propose Contract And Conquer (CAC), an approach to provably compute adversarial examples for neural networks in a black-box manner. The method is based on knowledge distillation of a black-box model on an expanding distillation dataset and precise contraction of the adversarial example search space. CAC is supported by the transferability guarantee: we prove that the method yields an adversarial example for the black-box model within a fixed number of algorithm iterations. Experimentally, we demonstrate that the proposed approach outperforms existing state-of-the-art black-box attack methods on ImageNet dataset for different target models, including vision transformers. 

---
# A Platform-Agnostic Multimodal Digital Human Modelling Framework: Neurophysiological Sensing in Game-Based Interaction 

**Authors**: Daniel J. Buxton, Mufti Mahmud, Jordan J. Bird, Thomas Hughes-Roberts, David J. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2603.10680)  

**Abstract**: Digital Human Modelling (DHM) is increasingly shaped by advances in AI, wearable biosensing, and interactive digital environments, particularly in research addressing accessibility and inclusion. However, many AI-enabled DHM approaches remain tightly coupled to specific platforms, tasks, or interpretative pipelines, limiting reproducibility, scalability, and ethical reuse. This paper presents a platform-agnostic DHM framework designed to support AI-ready multimodal interaction research by explicitly separating sensing, interaction modelling, and inference readiness. The framework integrates the OpenBCI Galea headset as a unified multimodal sensing layer, providing concurrent EEG, EMG, EOG, PPG, and inertial data streams, alongside a reproducible, game-based interaction environment implemented using SuperTux. Rather than embedding AI models or behavioural inference, physiological signals are represented as structured, temporally aligned observables, enabling downstream AI methods to be applied under appropriate ethical approval. Interaction is modelled using computational task primitives and timestamped event markers, supporting consistent alignment across heterogeneous sensors and platforms. Technical verification via author self-instrumentation confirms data integrity, stream continuity, and synchronisation; no human-subjects evaluation or AI inference is reported. Scalability considerations are discussed with respect to data throughput, latency, and extension to additional sensors or interaction modalities. Illustrative use cases demonstrate how the framework can support AI-enabled DHM and HCI studies, including accessibility-oriented interaction design and adaptive systems research, without requiring architectural modifications. The proposed framework provides an emerging-technology-focused infrastructure for future ethics-approved, inclusive DHM research. 

---
# Are Video Reasoning Models Ready to Go Outside? 

**Authors**: Yangfan He, Changgyu Boo, Jaehong Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2603.10652)  

**Abstract**: In real-world deployment, vision-language models often encounter disturbances such as weather, occlusion, and camera motion. Under such conditions, their understanding and reasoning degrade substantially, revealing a gap between clean, controlled (i.e., unperturbed) evaluation settings and real-world robustness. To address this limitation, we propose ROVA, a novel training framework that improves robustness by modeling a robustness-aware consistency reward under spatio-temporal corruptions. ROVA introduces a difficulty-aware online training strategy that prioritizes informative samples based on the model's evolving capability. Specifically, it continuously re-estimates sample difficulty via self-reflective evaluation, enabling adaptive training with a robustness-aware consistency reward. We also introduce PVRBench, a new benchmark that injects real-world perturbations into embodied video datasets to assess both accuracy and reasoning quality under realistic disturbances. We evaluate ROVA and baselines on PVRBench, UrbanVideo, and VisBench, where open-source and proprietary models suffer up to 35% and 28% drops in accuracy and reasoning under realistic perturbations. ROVA effectively mitigates performance degradation, boosting relative accuracy by at least 24% and reasoning by over 9% compared with baseline models (QWen2.5/3-VL, InternVL2.5, Embodied-R). These gains transfer to clean standard benchmarks, yielding consistent improvements. 

---
# Interleaving Scheduling and Motion Planning with Incremental Learning of Symbolic Space-Time Motion Abstractions 

**Authors**: Elisa Tosello, Arthur Bit-Monnot, Davide Lusuardi, Alessandro Valentini, Andrea Micheli  

**Link**: [PDF](https://arxiv.org/pdf/2603.10651)  

**Abstract**: Task and Motion Planning combines high-level task sequencing (what to do) with low-level motion planning (how to do it) to generate feasible, collision-free execution plans. However, in many real-world domains, such as automated warehouses, tasks are predefined, shifting the challenge to if, when, and how to execute them safely and efficiently under resource, time and motion constraints. In this paper, we formalize this as the Scheduling and Motion Planning problem for multi-object navigation in shared workspaces. We propose a novel solution framework that interleaves off-the-shelf schedulers and motion planners in an incremental learning loop. The scheduler generates candidate plans, while the motion planner checks feasibility and returns symbolic feedback, i.e., spatial conflicts and timing adjustments, to guide the scheduler towards motion-feasible solutions. We validate our proposal on logistics and job-shop scheduling benchmarks augmented with motion tasks, using state-of-the-art schedulers and sampling-based motion planners. Our results show the effectiveness of our framework in generating valid plans under complex temporal and spatial constraints, where synchronized motion is critical. 

---
# Detecting and Eliminating Neural Network Backdoors Through Active Paths with Application to Intrusion Detection 

**Authors**: Eirik Høyheim, Magnus Wiik Eckhoff, Gudmund Grov, Robert Flood, David Aspinall  

**Link**: [PDF](https://arxiv.org/pdf/2603.10641)  

**Abstract**: Machine learning backdoors have the property that the machine learning model should work as expected on normal inputs, but when the input contains a specific $\textit{trigger}$, it behaves as the attacker desires. Detecting such triggers has been proven to be extremely difficult. In this paper, we present a novel and explainable approach to detect and eliminate such backdoor triggers based on active paths found in neural networks. We present promising experimental evidence of our approach, which involves injecting backdoors into a machine learning model used for intrusion detection. 

---
# Reinforcement Learning with Conditional Expectation Reward 

**Authors**: Changyi Xiao, Caijun Xu, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10624)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective in enhancing the reasoning capabilities of large language models, particularly in domains such as mathematics where reliable rule-based verifiers can be constructed. However, the reliance on handcrafted, domain-specific verification rules substantially limits the applicability of RLVR to general reasoning domains with free-form answers, where valid answers often exhibit significant variability, making it difficult to establish complete and accurate rules. To address this limitation, we propose Conditional Expectation Reward (CER), which leverages the large language model itself as an implicit verifier, and is therefore applicable to general domains and eliminates the need for external verifiers or auxiliary models. CER is defined as the expected likelihood of generating the reference answer conditioned on the generated answer. In contrast to rule-based verifiers that yield binary feedback, CER provides a soft, graded reward signal that reflects varying degrees of correctness, making it better suited to tasks where answers vary in correctness. Experimental results demonstrate that CER is effective across a wide range of reasoning tasks, spanning both mathematical and general domains, indicating that CER serves as a flexible and general verification mechanism. The code is available at this https URL. 

---
# Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction 

**Authors**: Hao Zhou, Lu Qi, Jason Li, Jie Zhang, Yi Liu, Xu Yang, Mingyu Fan, Fei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.10597)  

**Abstract**: Trajectory prediction is critical for autonomous driving, enabling safe and efficient planning in dense, dynamic traffic. Most existing methods optimize prediction accuracy under fixed-length observations. However, real-world driving often yields variable-length, incomplete observations, posing a challenge to these methods. A common strategy is to directly map features from incomplete observations to those from complete ones. This one-shot mapping, however, struggles to learn accurate representations for short trajectories due to significant information gaps. To address this issue, we propose a Progressive Retrospective Framework (PRF), which gradually aligns features from incomplete observations with those from complete ones via a cascade of retrospective units. Each unit consists of a Retrospective Distillation Module (RDM) and a Retrospective Prediction Module (RPM), where RDM distills features and RPM recovers previous timesteps using the distilled features. Moreover, we propose a Rolling-Start Training Strategy (RSTS) that enhances data efficiency during PRF training. PRF is plug-and-play with existing methods. Extensive experiments on datasets Argoverse 2 and Argoverse 1 demonstrate the effectiveness of PRF. Code is available at this https URL. 

---
# Gradient Flow Drifting: Generative Modeling via Wasserstein Gradient Flows of KDE-Approximated Divergences 

**Authors**: Jiarui Cao, Zixuan Wei, Yuxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10592)  

**Abstract**: We reveal a precise mathematical framework about a new family of generative models which we call Gradient Flow Drifting. With this framework, we prove an equivalence between the recently proposed Drifting Model and the Wasserstein gradient flow of the forward KL divergence under kernel density estimation (KDE) approximation. Specifically, we prove that the drifting field of drifting model (arXiv:2602.04770) equals, up to a bandwidth-squared scaling factor, the difference of KDE log-density gradients $\nabla \log p_{\mathrm{kde}} - \nabla \log q_{\mathrm{kde}}$, which is exactly the particle velocity field of the Wasserstein-2 gradient flow of $KL(q\|p)$ with KDE-approximated densities. Besides that, this broad family of generative models can also include MMD-based generators, which arises as special cases of Wasserstein gradient flows of different divergences under KDE approximation. We provide a concise identifiability proof, and a theoretically grounded mixed-divergence strategy. We combine reverse KL and $\chi^2$ divergence gradient flows to simultaneously avoid mode collapse and mode blurring, and extend this method onto Riemannian manifold which loosens the constraints on the kernel function, and makes this method more suitable for the semantic space. Preliminary experiments on synthetic benchmarks validate the framework. 

---
# Towards Cognitive Defect Analysis in Active Infrared Thermography with Vision-Text Cues 

**Authors**: Mohammed Salah, Eman Ouda, Giuseppe Dell'Avvocato, Fabrizio Sarasini, Ester D'Accardi, Jorge Dias, Davor Svetinovic, Stefano Sfarra, Yusra Abdulrahman  

**Link**: [PDF](https://arxiv.org/pdf/2603.10549)  

**Abstract**: Active infrared thermography (AIRT) is currently witnessing a surge of artificial intelligence (AI) methodologies being deployed for automated subsurface defect analysis of high performance carbon fiber-reinforced polymers (CFRP). Deploying AI-based AIRT methodologies for inspecting CFRPs requires the creation of time consuming and expensive datasets of CFRP inspection sequences to train neural networks. To address this challenge, this work introduces a novel language-guided framework for cognitive defect analysis in CFRPs using AIRT and vision-language models (VLMs). Unlike conventional learning-based approaches, the proposed framework does not require developing training datasets for extensive training of defect detectors, instead it relies solely on pretrained multimodal VLM encoders coupled with a lightweight adapter to enable generative zero-shot understanding and localization of subsurface defects. By leveraging pretrained multimodal encoders, the proposed system enables generative zero-shot understanding of thermographic patterns and automatic detection of subsurface defects. Given the domain gap between thermographic data and natural images used to train VLMs, an AIRT-VLM Adapter is proposed to enhance the visibility of defects while aligning the thermographic domain with the learned representations of VLMs. The proposed framework is validated using three representative VLMs; specifically, GroundingDINO, Qwen-VL-Chat, and CogVLM. Validation is performed on 25 CFRP inspection sequences with impacts introduced at different energy levels, reflecting realistic defects encountered in industrial scenarios. Experimental results demonstrate that the AIRT-VLM adapter achieves signal-to-noise ratio (SNR) gains exceeding 10 dB compared with conventional thermographic dimensionality-reduction methods, while enabling zero-shot defect detection with intersection-over-union values reaching 70%. 

---
# SCORE: Replacing Layer Stacking with Contractive Recurrent Depth 

**Authors**: Guillaume Godin  

**Link**: [PDF](https://arxiv.org/pdf/2603.10544)  

**Abstract**: Residual connections are central to modern deep neural networks, enabling stable optimization and efficient information flow across depth. In this work, we propose SCORE (Skip-Connection ODE Recurrent Embedding), a discrete recurrent alternative to classical layer stacking. Instead of composing multiple independent layers, SCORE iteratively applies a single shared neural block using an ODE (Ordinary Differential Equation)-inspired contractive update: ht+1 = (1 - dt) * ht + dt * F(ht) This formulation can be interpreted as a depth-by-iteration refinement process, where the step size dt explicitly controls stability and update magnitude. Unlike continuous Neural ODE approaches, SCORE uses a fixed number of discrete iterations and standard backpropagation without requiring ODE solvers or adjoint methods. We evaluate SCORE across graph neural networks (ESOL molecular solubility), multilayer perceptrons, and Transformer-based language models (nanoGPT). Across architectures, SCORE generally improves convergence speed and often accelerates training. SCORE is reducing parameter count through shared weights. In practice, simple Euler integration provides the best trade-off between computational cost and performance, while higher-order integrators yield marginal gains at increased compute. These results suggest that controlled recurrent depth with contractive residual updates offers a lightweight and effective alternative to classical stacking in deep neural networks. 

---
# Prompting with the human-touch: evaluating model-sensitivity of foundation models for musculoskeletal CT segmentation 

**Authors**: Caroline Magg, Maaike A. ter Wee, Johannes G.G. Dobbe, Geert J. Streekstra, Leendert Blankevoort, Clara I. Sánchez, Hoel Kervadec  

**Link**: [PDF](https://arxiv.org/pdf/2603.10541)  

**Abstract**: Promptable Foundation Models (FMs), initially introduced for natural image segmentation, have also revolutionized medical image segmentation. The increasing number of models, along with evaluations varying in datasets, metrics, and compared models, makes direct performance comparison between models difficult and complicates the selection of the most suitable model for specific clinical tasks. In our study, 11 promptable FMs are tested using non-iterative 2D and 3D prompting strategies on a private and public dataset focusing on bone and implant segmentation in four anatomical regions (wrist, shoulder, hip and lower leg). The Pareto-optimal models are identified and further analyzed using human prompts collected through a dedicated observer study. Our findings are: 1) The segmentation performance varies a lot between FMs and prompting strategies; 2) The Pareto-optimal models in 2D are SAM and SAM2.1, in 3D nnInteractive and Med-SAM2; 3) Localization accuracy and rater consistency vary with anatomical structures, with higher consistency for simple structures (wrist bones) and lower consistency for complex structures (pelvis, tibia, implants); 4) The segmentation performance drops using human prompts, suggesting that performance reported on "ideal" prompts extracted from reference labels might overestimate the performance in a human-driven setting; 5) All models were sensitive to prompt variations. While two models demonstrated intra-rater robustness, it did not scale to inter-rater settings. We conclude that the selection of the most optimal FM for a human-driven setting remains challenging, with even high-performing FMs being sensitive to variations in human input prompts. Our code base for prompt extraction and model inference is available: this https URL 

---
# UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery 

**Authors**: Islam Guven, Mehmet Parlak  

**Link**: [PDF](https://arxiv.org/pdf/2603.10528)  

**Abstract**: Unmanned aerial vehicles (UAVs) are increasingly used to support time-critical medical supply delivery, providing rapid and flexible logistics during emergencies and resource shortages. However, effective deployment of UAV fleets requires coordination mechanisms capable of prioritizing medical requests, allocating limited aerial resources, and adapting delivery schedules under uncertain operational conditions. This paper presents a multi-agent reinforcement learning (MARL) framework for coordinating UAV fleets in stochastic medical delivery scenarios where requests vary in urgency, location, and delivery deadlines. The problem is formulated as a partially observable Markov decision process (POMDP) in which UAV agents maintain awareness of medical delivery demands while having limited visibility of other agents due to communication and localization constraints. The proposed framework employs Proximal Policy Optimization (PPO) as the primary learning algorithm and evaluates several variants, including asynchronous extensions, classical actor--critic methods, and architectural modifications to analyze scalability and performance trade-offs. The model is evaluated using real-world geographic data from selected clinics and hospitals extracted from the OpenStreetMap dataset. The framework provides a decision-support layer that prioritizes medical tasks, reallocates UAV resources in real time, and assists healthcare personnel in managing urgent logistics. Experimental results show that classical PPO achieves superior coordination performance compared to asynchronous and sequential learning strategies, highlighting the potential of reinforcement learning for adaptive and scalable UAV-assisted healthcare logistics. 

---
# Naïve Exposure of Generative AI Capabilities Undermines Deepfake Detection 

**Authors**: Sunpill Kim, Chanwoo Hwang, Minsu Kim, Jae Hong Seo  

**Link**: [PDF](https://arxiv.org/pdf/2603.10504)  

**Abstract**: Generative AI systems increasingly expose powerful reasoning and image refinement capabilities through user-facing chatbot interfaces. In this work, we show that the naïve exposure of such capabilities fundamentally undermines modern deepfake detectors. Rather than proposing a new image manipulation technique, we study a realistic and already-deployed usage scenario in which an adversary uses only benign, policy-compliant prompts and commercial generative AI systems. We demonstrate that state-of-the-art deepfake detection methods fail under semantic-preserving image refinement. Specifically, we show that generative AI systems articulate explicit authenticity criteria and inadvertently externalize them through unrestricted reasoning, enabling their direct reuse as refinement objectives. As a result, refined images simultaneously evade detection, preserve identity as verified by commercial face recognition APIs, and exhibit substantially higher perceptual quality. Importantly, we find that widely accessible commercial chatbot services pose a significantly greater security risk than open-source models, as their superior realism, semantic controllability, and low-barrier interfaces enable effective evasion by non-expert users. Our findings reveal a structural mismatch between the threat models assumed by current detection frameworks and the actual capabilities of real-world generative AI. While detection baselines are largely shaped by prior benchmarks, deployed systems expose unrestricted authenticity reasoning and refinement despite stringent safety controls in other domains. 

---
# JEDI: Jointly Embedded Inference of Neural Dynamics 

**Authors**: Anirudh Jamkhandi, Ali Korojy, Olivier Codol, Guillaume Lajoie, Matthew G. Perich  

**Link**: [PDF](https://arxiv.org/pdf/2603.10489)  

**Abstract**: Animal brains flexibly and efficiently achieve many behavioral tasks with a single neural network. A core goal in modern neuroscience is to map the mechanisms of the brain's flexibility onto the dynamics underlying neural populations. However, identifying task-specific dynamical rules from limited, noisy, and high-dimensional experimental neural recordings remains a major challenge, as experimental data often provide only partial access to brain states and dynamical mechanisms. While recurrent neural networks (RNNs) directly constrained neural data have been effective in inferring underlying dynamical mechanisms, they are typically limited to single-task domains and struggle to generalize across behavioral conditions. Here, we introduce JEDI, a hierarchical model that captures neural dynamics across tasks and contexts by learning a shared embedding space over RNN weights. This model recapitulates individual samples of neural dynamics while scaling to arbitrarily large and complex datasets, uncovering shared structure across conditions in a single, unified model. Using simulated RNN datasets, we demonstrate that JEDI accurately learns robust, generalizable, condition-specific embeddings. By reverse-engineering the weights learned by JEDI, we show that it recovers ground truth fixed point structures and unveils key features of the underlying neural dynamics in the eigenspectra. Finally, we apply JEDI to motor cortex recordings during monkey reaching to extract mechanistic insight into the neural dynamics of motor control. Our work shows that joint learning of contextual embeddings and recurrent weights provides scalable and generalizable inference of brain dynamics from recordings alone. 

---
# Learning to Negotiate: Multi-Agent Deliberation for Collective Value Alignment in LLMs 

**Authors**: Panatchakorn Anantaprayoon, Nataliia Babina, Nima Asgharbeygi, Jad Tarifi  

**Link**: [PDF](https://arxiv.org/pdf/2603.10476)  

**Abstract**: The alignment of large language models (LLMs) has progressed substantially in single-agent settings through paradigms such as RLHF and Constitutional AI, with recent work exploring scalable alternatives such as RLAIF and evolving alignment objectives. However, these approaches remain limited in multi-stakeholder settings, where conflicting values arise and deliberative negotiation capabilities are required. This work proposes a multi-agent negotiation-based alignment framework that aligns LLMs to Collective Agency (CA)-an existing alignment objective introduced to promote the continual expansion of agency-while simultaneously improving conflict-resolution capability. To enable scalable training, two self-play instances of the same LLM, assigned opposing personas, engage in structured turn-based dialogue to synthesize mutually beneficial solutions. We generate synthetic moral-dilemma prompts and conflicting persona pairs, and optimize the policy via RLAIF using GRPO with an external LLM reward model. While rewards are computed from CA scores assigned to the final completion, gradients are applied to dialogue tokens to directly improve deliberative interaction dynamics. Experiments show that the resulting model achieves CA alignment comparable to a single-agent baseline while substantially improving conflict-resolution performance without degrading general language capabilities. These results suggest that negotiation-driven deliberation training provides a practical path toward LLMs that better support collective decision-making in value-conflict scenarios. 

---
# Aligning Large Language Models with Searcher Preferences 

**Authors**: Wei Wu, Peilun Zhou, Liyi Chen, Qimeng Wang, Chengqiang Lu, Yan Gao, Yi Wu, Yao Hu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.10473)  

**Abstract**: The paradigm shift from item-centric ranking to answer-centric synthesis is redefining the role of search engines. While recent industrial progress has applied generative techniques to closed-set item ranking in e-commerce, research and deployment of open-ended generative search on large content platforms remain limited. This setting introduces challenges, including robustness to noisy retrieval, non-negotiable safety guarantees, and alignment with diverse user needs. In this work, we introduce SearchLLM, the first large language model (LLM) for open-ended generative search. We design a hierarchical, multi-dimensional reward system that separates bottom-line constraints, including factual grounding, basic answer quality and format compliance, from behavior optimization objectives that promote robustness to noisy retrieval and alignment with user needs. Concretely, our reward model evaluates responses conditioned on the user query, session history, and retrieved evidence set, combining rule-based checks with human-calibrated LLM judges to produce an interpretable score vector over these dimensions. We introduce a Gated Aggregation Strategy to derive the training reward for optimizing SearchLLM with Group Relative Policy Optimization (GRPO). We deploy SearchLLM in the AI search entry of RedNote. Offline evaluations and online A/B tests show improved generation quality and user engagement, increasing Valid Consumption Rate by 1.03% and reducing Re-search Rate by 2.81%, while upholding strict safety and reliability standards. 

---
# Modeling Stage-wise Evolution of User Interests for News Recommendation 

**Authors**: Zhiyong Cheng, Yike Jin, Zhijie Zhang, Huilin Chen, Zhangling Duan, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10471)  

**Abstract**: Personalized news recommendation is highly time-sensitive, as user interests are often driven by emerging events, trending topics, and shifting real-world contexts. These dynamics make it essential to model not only users' long-term preferences, which reflect stable reading habits and high-order collaborative patterns, but also their short-term, context-dependent interests that change rapidly over time. However, most existing approaches rely on a single static interaction graph, which struggles to capture both long-term preference patterns and short-term interest changes as user behavior evolves. To address this challenge, we propose a unified framework that learns user preferences from both global and local temporal perspectives. A global preference modeling component captures long-term collaborative signals from the overall interaction graph, while a local preference modeling component partitions historical interactions into stage-wise temporal subgraphs to represent short-term dynamics. Within this module, an LSTM branch models the progressive evolution of recent interests, and a self-attention branch captures long-range temporal dependencies. Extensive experiments on two large-scale real-world datasets show that our approach consistently outperforms strong baselines and delivers fresher and more relevant recommendations across diverse user behaviors and temporal settings. 

---
# G-STAR: End-to-End Global Speaker-Tracking Attributed Recognition 

**Authors**: Jing Peng, Ziyi Chen, Haoyu Li, Yucheng Wang, Duo Ma, Mengtian Li, Yunfan Du, Dezhu Xu, Kai Yu, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10468)  

**Abstract**: We study timestamped speaker-attributed ASR for long-form, multi-party speech with overlap, where chunk-wise inference must preserve meeting-level speaker identity consistency while producing time-stamped, speaker-labeled transcripts. Previous Speech-LLM systems tend to prioritize either local diarization or global labeling, but often lack the ability to capture fine-grained temporal boundaries or robust cross-chunk identity linking. We propose G-STAR, an end-to-end system that couples a time-aware speaker-tracking module with a Speech-LLM transcription backbone. The tracker provides structured speaker cues with temporal grounding, and the LLM generates attributed text conditioned on these cues. G-STAR supports both component-wise optimization and joint end-to-end training, enabling flexible learning under heterogeneous supervision and domain shift. Experiments analyze cue fusion, local versus long-context trade-offs and hierarchical objectives. 

---
# UniPINN: A Unified PINN Framework for Multi-task Learning of Diverse Navier-Stokes Equations 

**Authors**: Dengdi Sun, Jie Chen, Xiao Wang, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10466)  

**Abstract**: Physics-Informed Neural Networks (PINNs) have shown promise in solving incompressible Navier-Stokes equations, yet existing approaches are predominantly designed for single-flow settings. When extended to multi-flow scenarios, these methods face three key challenges: (1) difficulty in simultaneously capturing both shared physical principles and flow-specific characteristics, (2) susceptibility to inter-task negative transfer that degrades prediction accuracy, and (3) unstable training dynamics caused by disparate loss magnitudes across heterogeneous flow regimes. To address these limitations, we propose UniPINN, a unified multi-flow PINN framework that integrates three complementary components: a shared-specialized architecture that disentangles universal physical laws from flow-specific features, a cross-flow attention mechanism that selectively reinforces relevant patterns while suppressing task-irrelevant interference, and a dynamic weight allocation strategy that adaptively balances loss contributions to stabilize multi-objective optimization. Extensive experiments on three canonical flows demonstrate that UniPINN effectively unifies multi-flow learning, achieving superior prediction accuracy and balanced performance across heterogeneous regimes while successfully mitigating negative transfer. The source code of this paper will be released on this https URL 

---
# FAR-Dex: Few-shot Data Augmentation and Adaptive Residual Policy Refinement for Dexterous Manipulation 

**Authors**: Yushan Bai, Fulin Chen, Hongzheng Sun, Yuchuang Tong, En Li, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10451)  

**Abstract**: Achieving human-like dexterous manipulation through the collaboration of multi-fingered hands with robotic arms remains a longstanding challenge in robotics, primarily due to the scarcity of high-quality demonstrations and the complexity of high-dimensional action spaces. To address these challenges, we propose FAR-Dex, a hierarchical framework that integrates few-shot data augmentation with adaptive residual refinement to enable robust and precise arm-hand coordination in dexterous tasks. First, FAR-DexGen leverages the IsaacLab simulator to generate diverse and physically constrained trajectories from a few demonstrations, providing a data foundation for policy training. Second, FAR-DexRes introduces an adaptive residual module that refines policies by combining multi-step trajectory segments with observation features, thereby enhancing accuracy and robustness in manipulation scenarios. Experiments in both simulation and real-world demonstrate that FAR-Dex improves data quality by 13.4% and task success rates by 7% over state-of-the-art methods. It further achieves over 80% success in real-world tasks, enabling fine-grained dexterous manipulation with strong positional generalization. 

---
# The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training 

**Authors**: Hengjie Cao, Zhendong Huang, Mengyi Chen, Yifeng Yang, Fanqi Yu, Ruijun Huang, Fang Dong, Xin Zhang, Jixian Zhou, Anrui Chen, Mingzhi Dong, Yujiang Wang, Jinlong Hou, Qin Lv, Yuan Cheng, Tun Lu, Fan Yang, Li Shang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10444)  

**Abstract**: Large language models trained on natural language exhibit pronounced anisotropy: a small number of directions concentrate disproportionate energy, while the remaining dimensions form a broad semantic tail. In low-bit training regimes, this geometry becomes numerically unstable. Because blockwise quantization scales are determined by extreme elementwise magnitudes, dominant directions stretch the dynamic range, compressing long-tail semantic variation into narrow numerical bins. We show that this instability is primarily driven by a coherent rank-one mean bias, which constitutes the dominant component of spectral anisotropy in LLM representations. This mean component emerges systematically across layers and training stages and accounts for the majority of extreme activation magnitudes, making it the principal driver of dynamic-range inflation under low precision. Crucially, because the dominant instability is rank-one, it can be eliminated through a simple source-level mean-subtraction operation. This bias-centric conditioning recovers most of the stability benefits of SVD-based spectral methods while requiring only reduction operations and standard quantization kernels. Empirical results on FP4 (W4A4G4) training show that mean removal substantially narrows the loss gap to BF16 and restores downstream performance, providing a hardware-efficient path to stable low-bit LLM training. 

---
# Domain-Adaptive Health Indicator Learning with Degradation-Stage Synchronized Sampling and Cross-Domain Autoencoder 

**Authors**: Jungho Choo, Hanbyeol Park, Gawon Lee, Yunkyung Park, Hyerim Bae  

**Link**: [PDF](https://arxiv.org/pdf/2603.10430)  

**Abstract**: The construction of high quality health indicators (HIs) is crucial for effective prognostics and health management. Although deep learning has significantly advanced HI modeling, existing approaches often struggle with distribution mismatches resulting from varying operating conditions. Although domain adaptation is typically employed to mitigate these shifts, two critical challenges remain: (1) the misalignment of degradation stages during random mini-batch sampling, resulting in misleading discrepancy losses, and (2) the structural limitations of small-kernel 1D-CNNs in capturing long-range temporal dependencies within complex vibration signals. To address these issues, we propose a domain-adaptive framework comprising degradation stage synchronized batch sampling (DSSBS) and the cross-domain aligned fusion large autoencoder (CAFLAE). DSSBS utilizes kernel change-point detection to segment degradation stages, ensuring that source and target mini-batches are synchronized by their failure phases during alignment. Complementing this, CAFLAE integrates large-kernel temporal feature extraction with cross-attention mechanisms to learn superior domain-invariant representations. The proposed framework was rigorously validated on a Korean defense system dataset and the XJTU-SY bearing dataset, achieving an average performance enhancement of 24.1% over state-of-the-art methods. These results demonstrate that DSSBS improves cross-domain alignment through stage-consistent sampling, whereas CAFLAE offers a high-performance backbone for long-term industrial condition monitoring. 

---
# Enhancing Network Intrusion Detection Systems: A Multi-Layer Ensemble Approach to Mitigate Adversarial Attacks 

**Authors**: Nasim Soltani, Shayan Nejadshamsi, Zakaria Abou El Houda, Raphael Khoury, Kelton A. P. Costa, Tiago H. Falk, Anderson R. Avila  

**Link**: [PDF](https://arxiv.org/pdf/2603.10413)  

**Abstract**: Adversarial examples can represent a serious threat to machine learning (ML) algorithms. If used to manipulate the behaviour of ML-based Network Intrusion Detection Systems (NIDS), they can jeopardize network security. In this work, we aim to mitigate such risks by increasing the robustness of NIDS towards adversarial attacks. To that end, we explore two adversarial methods for generating malicious network traffic. The first method is based on Generative Adversarial Networks (GAN) and the second one is the Fast Gradient Sign Method (FGSM). The adversarial examples generated by these methods are then used to evaluate a novel multilayer defense mechanism, specifically designed to mitigate the vulnerability of ML-based NIDS. Our solution consists of one layer of stacking classifiers and a second layer based on an autoencoder. If the incoming network data are classified as benign by the first layer, the second layer is activated to ensure that the decision made by the stacking classifier is correct. We also incorporated adversarial training to further improve the robustness of our solution. Experiments on two datasets, namely UNSW-NB15 and NSL-KDD, demonstrate that the proposed approach increases resilience to adversarial attacks. 

---
# Effective Dataset Distillation for Spatio-Temporal Forecasting with Bi-dimensional Compression 

**Authors**: Taehyung Kwon, Yeonje Choi, Yeongho Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2603.10410)  

**Abstract**: Spatio-temporal time series are widely used in real-world applications, including traffic prediction and weather forecasting. They are sequences of observations over extensive periods and multiple locations, naturally represented as multidimensional data. Forecasting is a central task in spatio-temporal analysis, and numerous deep learning methods have been developed to address it. However, as dataset sizes and model complexities continue to grow in practice, training deep learning models has become increasingly time- and resource-intensive. A promising solution to this challenge is dataset distillation, which synthesizes compact datasets that can effectively replace the original data for model training. Although successful in various domains, including time series analysis, existing dataset distillation methods compress only one dimension, making them less suitable for spatio-temporal datasets, where both spatial and temporal dimensions jointly contribute to the large data volume. To address this limitation, we propose STemDist, the first dataset distillation method specialized for spatio-temporal time series forecasting. A key idea of our solution is to compress both temporal and spatial dimensions in a balanced manner, reducing training time and memory. We further reduce the distillation cost by performing distillation at the cluster level rather than the individual location level, and we complement this coarse-grained approach with a subset-based granular distillation technique that enhances forecasting performance. On five real-world datasets, we show empirically that, compared to both general and time-series dataset distillation methods, datasets distilled by our STemDist method enable model training (1) faster (up to 6X) (2) more memory-efficient (up to 8X), and (3) more effective (with up to 12% lower prediction error). 

---
# Designing Service Systems from Textual Evidence 

**Authors**: Ruicheng Ao, Hongyu Chen, Siyang Gao, Hanwei Li, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2603.10400)  

**Abstract**: Designing service systems requires selecting among alternative configurations -- choosing the best chatbot variant, the optimal routing policy, or the most effective quality control procedure. In many service systems, the primary evidence of performance quality is textual -- customer support transcripts, complaint narratives, compliance review reports -- rather than the scalar measurements assumed by classical optimization methods. Large language models (LLMs) can read such textual evidence and produce standardized quality scores, but these automated judges exhibit systematic biases that vary across alternatives and evaluation instances. Human expert review remains accurate but costly. We study how to identify the best service configuration with high confidence while minimizing expensive human audits, given that automated evaluation is cheap but biased. We formalize this as a sequential decision problem where a biased proxy score is observed for every evaluation, and a verified outcome can be acquired selectively at additional cost. We prove that LLM-only selection fails under arm-dependent bias, and that naive selective-audit estimators can be asymptotically biased. We develop an estimator combining proxy scores with inverse-propensity-weighted residuals and construct anytime-valid confidence sequences. Our algorithm, PP-LUCB, jointly decides which alternatives to evaluate and whether to request human audits, concentrating reviews where the LLM judge is least reliable. We prove correctness and establish instance-dependent cost bounds showing near-optimal efficiency. On a customer support ticket classification task, our algorithm correctly identifies the best model in 40/40 trials while achieving 90\% audit cost reduction. 

---
# On the Learning Dynamics of Two-layer Linear Networks with Label Noise SGD 

**Authors**: Tongcheng Zhang, Zhanpeng Zhou, Mingze Wang, Andi Han, Wei Huang, Taiji Suzuki, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2603.10397)  

**Abstract**: One crucial factor behind the success of deep learning lies in the implicit bias induced by noise inherent in gradient-based training algorithms. Motivated by empirical observations that training with noisy labels improves model generalization, we delve into the underlying mechanisms behind stochastic gradient descent (SGD) with label noise. Focusing on a two-layer over-parameterized linear network, we analyze the learning dynamics of label noise SGD, unveiling a two-phase learning behavior. In \emph{Phase I}, the magnitudes of model weights progressively diminish, and the model escapes the lazy regime; enters the rich regime. In \emph{Phase II}, the alignment between model weights and the ground-truth interpolator increases, and the model eventually converges. Our analysis highlights the critical role of label noise in driving the transition from the lazy to the rich regime and minimally explains its empirical success. Furthermore, we extend these insights to Sharpness-Aware Minimization (SAM), showing that the principles governing label noise SGD also apply to broader optimization algorithms. Extensive experiments, conducted under both synthetic and real-world setups, strongly support our theory. Our code is released at this https URL. 

---
# Safe Probabilistic Planning for Human-Robot Interaction using Conformal Risk Control 

**Authors**: Jake Gonzales, Kazuki Mizuta, Karen Leung, Lillian J. Ratliff  

**Link**: [PDF](https://arxiv.org/pdf/2603.10392)  

**Abstract**: In this paper, we present a novel probabilistic safe control framework for human-robot interaction that combines control barrier functions (CBFs) with conformal risk control to provide formal safety guarantees while considering complex human behavior. The approach uses conformal risk control to quantify and control the prediction errors in CBF safety values and establishes formal guarantees on the probability of constraint satisfaction during interaction. We introduce an algorithm that dynamically adjusts the safety margins produced by conformal risk control based on the current interaction context. Through experiments on human-robot navigation scenarios, we demonstrate that our approach significantly reduces collision rates and safety violations as compared to baseline methods while maintaining high success rates in goal-reaching tasks and efficient control. The code, simulations, and other supplementary material can be found on the project website: this https URL. 

---
# Optimal Expert-Attention Allocation in Mixture-of-Experts: A Scalable Law for Dynamic Model Design 

**Authors**: Junzhuo Li, Peijie Jiang, Changxin Tian, Jia Liu, Zhiqiang Zhang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10379)  

**Abstract**: This paper presents a novel extension of neural scaling laws to Mixture-of-Experts (MoE) models, focusing on the optimal allocation of compute between expert and attention sub-layers. As MoE architectures have emerged as an efficient method for scaling model capacity without proportionally increasing computation, determining the optimal expert-attention compute ratio becomes critical. We define the ratio $r$ as the fraction of total FLOPs per token dedicated to the expert layers versus the attention layers, and explore how this ratio interacts with the overall compute budget and model sparsity. Through extensive experiments with GPT-style MoE Transformers, we empirically find that the optimal ratio $r^*$ follows a power-law relationship with total compute and varies with sparsity. Our analysis leads to an explicit formula for $r^*$, enabling precise control over the expert-attention compute allocation. We generalize the Chinchilla scaling law by incorporating this architectural parameter, providing a new framework for tuning MoE models beyond size and data. Our findings offer practical guidelines for designing efficient MoE models, optimizing performance while respecting fixed compute budgets. 

---
# Causal Concept Graphs in LLM Latent Space for Stepwise Reasoning 

**Authors**: Md Muntaqim Meherab, Noor Islam S. Mohammad, Faiza Feroz  

**Link**: [PDF](https://arxiv.org/pdf/2603.10377)  

**Abstract**: Sparse autoencoders can localize where concepts live in language models, but not how they interact during multi-step reasoning. We propose Causal Concept Graphs (CCG): a directed acyclic graph over sparse, interpretable latent features, where edges capture learned causal dependencies between concepts. We combine task-conditioned sparse autoencoders for concept discovery with DAGMA-style differentiable structure learning for graph recovery and introduce the Causal Fidelity Score (CFS) to evaluate whether graph-guided interventions induce larger downstream effects than random ones. On ARC-Challenge, StrategyQA, and LogiQA with GPT-2 Medium, across five seeds ($n{=}15$ paired runs), CCG achieves $\CFS=5.654\pm0.625$, outperforming ROME-style tracing ($3.382\pm0.233$), SAE-only ranking ($2.479\pm0.196$), and a random baseline ($1.032\pm0.034$), with $p<0.0001$ after Bonferroni correction. Learned graphs are sparse (5-6\% edge density), domain-specific, and stable across seeds. 

---
# Reactive Writers: How Co-Writing with AI Changes How We Engage with Ideas 

**Authors**: Advait Bhat, Marianne Aubin Le Quéré, Mor Naaman, Maurice Jakesch  

**Link**: [PDF](https://arxiv.org/pdf/2603.10374)  

**Abstract**: Emerging experimental evidence shows that writing with AI assistance can change both the views people express in writing and the opinions they hold afterwards. Yet, we lack substantive understanding of procedural and behavioral changes in co-writing with AI that underlie the observed opinion-shaping power of AI writing tools. We conducted a mixed-methods study, combining retrospective interviews with 19 participants about their AI co-writing experience with a quantitative analysis tracing engagement with ideas and opinions in 1{,}291 AI co-writing sessions. Our analysis shows that engaging with the AI's suggestions -- reading them and deciding whether to accept them -- becomes a central activity in the writing process, taking away from more traditional processes of ideation and language generation. As writers often do not complete their own ideation before engaging with suggestions, the suggested ideas and opinions seeded directions that writers then elaborated on. At the same time, writers did not notice the AI's influence and felt in full control of their writing, as they -- in principle -- could always edit the final text. We term this shift \textit{Reactive Writing}: an evaluation-first, suggestion-led writing practice that departs substantially from conventional composing in the presence of AI assistance and is highly vulnerable to AI-induced biases and opinion shifts. 

---
# Few-Shot Adaptation to Non-Stationary Environments via Latent Trend Embedding for Robotics 

**Authors**: Yasuyuki Fujii, Emika Kameda, Hiroki Fukada, Yoshiki Mori, Tadashi Matsuo, Nobutaka Shimada  

**Link**: [PDF](https://arxiv.org/pdf/2603.10373)  

**Abstract**: Robotic systems operating in real-world environments often suffer from concept shift, where the input-output relationship changes due to latent environmental factors that are not directly observable. Conventional adaptation methods update model parameters, which may cause catastrophic forgetting and incur high computational cost. This paper proposes a latent Trend ID-based framework for few-shot adaptation in non-stationary environments. Instead of modifying model weights, a low-dimensional environmental state, referred to as the Trend ID, is estimated via backpropagation while the model parameters remain fixed. To prevent overfitting caused by per-sample latent variables, we introduce temporal regularization and a state transition model that enforces smooth evolution of the latent space. Experiments on a quantitative food grasping task demonstrate that the learned Trend IDs are distributed across distinct regions of the latent space with temporally consistent trajectories, and that few-shot adaptation to unseen environments is achieved without modifying model parameters. The proposed framework provides a scalable and interpretable solution for robotics applications operating across diverse and evolving environments. 

---
# Beyond Interleaving: Causal Attention Reformulations for Generative Recommender Systems 

**Authors**: Hailing Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.10369)  

**Abstract**: Generative Recommender Systems (GR) increasingly model user behavior as a sequence generation task by interleaving item and action tokens. While effective, this formulation introduces significant structural and computational inefficiencies: it doubles sequence length, incurs quadratic overhead, and relies on implicit attention to recover the causal relationship between an item and its associated action. Furthermore, interleaving heterogeneous tokens forces the Transformer to disentangle semantically incompatible signals, leading to increased attention noise and reduced representation this http URL this work, we propose a principled reformulation of generative recommendation that aligns sequence modeling with underlying causal structures and attention theory. We demonstrate that current interleaving mechanisms act as inefficient proxies for similarity-weighted action pooling. To address this, we introduce two novel architectures that eliminate interleaved dependencies to reduce sequence complexity by 50%: Attention-based Late Fusion for Actions (AttnLFA) and Attention-based Mixed Value Pooling (AttnMVP). These models explicitly encode the $i_n \rightarrow a_n$ causal dependency while preserving the expressive power of Transformer-based sequence this http URL evaluate our framework on large-scale product recommendation data from a major social network. Experimental results show that AttnLFA and AttnMVP consistently outperform interleaved baselines, achieving evaluation loss improvements of 0.29% and 0.80%, and significant gains in Normalized Entropy (NE). Crucially, these performance gains are accompanied by training time reductions of 23% and 12%, respectively. Our findings suggest that explicitly modeling item-action causality provides a superior design paradigm for scalable and efficient generative ranking. 

---
# Dynamic Knowledge Fusion for Multi-Domain Dialogue State Tracking 

**Authors**: Haoxiang Su, Ruiyu Fang, Liting Jiang, Xiaomeng Huang, Shuangyong Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.10367)  

**Abstract**: The performance of task-oriented dialogue models is strongly tied to how well they track dialogue states, which records and updates user information across multi-turn interactions. However, current multi-domain DST encounters two key challenges: the difficulty of effectively modeling dialogue history and the limited availability of annotated data, both of which hinder model performance. To tackle the aforementioned problems, we develop a dynamic knowledge fusion framework applicable to multi-domain DST. The model operates in two stages: first, an encoder-only network trained with contrastive learning encodes dialogue history and candidate slots, selecting relevant slots based on correlation scores; second, dynamic knowledge fusion leverages the structured information of selected slots as contextual prompts to enhance the accuracy and consistency of dialogue state tracking. This design enables more accurate integration of dialogue context and domain knowledge. Results obtained from multi-domain dialogue benchmarks indicate that our method notably improves both tracking accuracy and generalization, validating its capability in handling complex dialogue scenarios. 

---
# Utility Function is All You Need: LLM-based Congestion Control 

**Authors**: Neta Rozen-Schiff, Liron Schiff, Stefan Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2603.10357)  

**Abstract**: Congestion is a critical and challenging problem in communication networks. Congestion control protocols allow network applications to tune their sending rate in a way that optimizes their performance and the network utilization. In the common distributed setting, the applications cannot collaborate with each other directly but instead obtain similar estimations about the state of the network using latency and loss measurements. These measurements can be fed into analytical functions, referred to by utility functions, whose gradients help each and all distributed senders to converge to a desired state.
The above process becomes extremely complicated when each application has different optimization goals and requirements. Crafting these utilization functions has been a research subject for over a decade, with small incremental changes requiring rigorous mathematical analysis as well as real-world experiments.
In this work, we present GenCC, a framework leveraging the code generation capabilities of large language models (LLMs) coupled with realistic network testbed, to design congestion control utility functions. Using GenCC, we analyze the impact of different guidance strategies on the performance of the generated protocols, considering application-specific requirements and network capacity.
Our results show that LLMs, guided by either a generative code evolution strategy or mathematical chain-of-thought (CoT), can obtain close to optimal results, improving state-of-the-art congestion control protocols by 37%-142%, depending on the scenario. 

---
# Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck 

**Authors**: Hongbin Zhang, Kehai Chen, Xuefen Bai, Youcheng Pan, Yang Xiang, Jinpeng Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10351)  

**Abstract**: Large language models (LLMs) have become a standard for multilingual evaluation, yet they exhibit a severe systematic translationese bias. In this paper, translationese bias is characterized as LLMs systematically favoring machine-translated text over human-authored references, particularly in low-resource languages. We attribute this bias to spurious correlations with (i) latent manifold alignment with English and (ii) cross-lingual predictability. To mitigate this bias, we propose DIBJudge, a robust fine-tuning framework that learns a minimally sufficient, judgment-critical representation via variational information compression, while explicitly isolating spurious factors into the dedicated bias branch. Furthermore, we incorporate a cross-covariance penalty that explicitly suppresses statistical dependence between robust and bias representations, thereby encouraging effective disentanglement. Extensive evaluations on multilingual reward modeling benchmarks and a dedicated translationese bias evaluation suite demonstrate that the proposed DIBJudge consistently outperforms strong baselines and substantially mitigates translationese bias. 

---
# Federated Active Learning Under Extreme Non-IID and Global Class Imbalance 

**Authors**: Chen-Chen Zong, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10341)  

**Abstract**: Federated active learning (FAL) seeks to reduce annotation cost under privacy constraints, yet its effectiveness degrades in realistic settings with severe global class imbalance and highly heterogeneous clients. We conduct a systematic study of query-model selection in FAL and uncover a central insight: the model that achieves more class-balanced sampling, especially for minority classes, consistently leads to better final performance. Moreover, global-model querying is beneficial only when the global distribution is highly imbalanced and client data are relatively homogeneous; otherwise, the local model is preferable. Based on these findings, we propose FairFAL, an adaptive class-fair FAL framework. FairFAL (1) infers global imbalance and local-global divergence via lightweight prediction discrepancy, enabling adaptive selection between global and local query models; (2) performs prototype-guided pseudo-labeling using global features to promote class-aware querying; and (3) applies a two-stage uncertainty-diversity balanced sampling strategy with k-center refinement. Experiments on five benchmarks show that FairFAL consistently outperforms state-of-the-art approaches under challenging long-tailed and non-IID settings. The code is available at this https URL. 

---
# Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation 

**Authors**: Sangmim Song, Sarath Kodagoda, Marc Carmichael, Karthick Thiyagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2603.10340)  

**Abstract**: Vision-Language-Action (VLA) models demonstrate impressive zero-shot generalization but frequently suffer from a "Precision-Reasoning Gap" in cluttered environments. This failure is driven by background-induced feature dilution, where high-frequency semantic noise corrupts the geometric grounding required for precise manipulation. To bridge this gap, we propose Concept-Gated Visual Distillation (CGVD), a training-free, model-agnostic inference framework that stabilizes VLA policies. CGVD operates by parsing instructions into safe and distractor sets, utilizing a two-layer target refinement process--combining cross-validation and spatial disambiguation--to explicitly penalize false positives and isolate genuine manipulation targets. We then process the scene via Fourier-based inpainting, generating a clean observation that actively suppresses semantic distractors while preserving critical spatial geometry and visual proprioception. Extensive evaluations in highly cluttered manipulation tasks demonstrate that CGVD prevents performance collapse. In environments with dense semantic distractors, our method significantly outperforms state-of-the-art baselines, achieving a 77.5% success rate compared to the baseline's 43.0%. By enforcing strict attribute adherence, CGVD establishes inference-time visual distillation as a critical prerequisite for robust robotic manipulation in the clutter. 

---
# Does Reasoning Make Search More Fair? Comparing Fairness in Reasoning and Non-Reasoning Rerankers 

**Authors**: Saron Samuel, Benjamin Van Durme, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10332)  

**Abstract**: While reasoning rerankers, such as Rank1, have demonstrated strong abilities in improving ranking relevance, it is unclear how they perform on other retrieval qualities such as fairness. We conduct the first systematic comparison of fairness between reasoning and non-reasoning rerankers. Using the TREC 2022 Fair Ranking Track dataset, we evaluate six reranking models across multiple retrieval settings and demographic attributes. Our findings demonstrate reasoning neither improve nor harm fairness compared to non-reasoning approaches. Our fairness metric, Attention-Weighted Rank Fairness (AWRF) remained stable (0.33-0.35) across all models, even as relevance varies substantially (nDCG 0.247-1.000). Demographic breakdown analysis revealed fairness gaps for geographic attributes regardless of model architecture. These results indicate that future work in specializing reasoning models to be aware of fairness attributes could lead to improvements, as current implementations preserve the fairness characteristics of their input ranking. 

---
# PC-Diffuser: Path-Consistent Capsule CBF Safety Filtering for Diffusion-Based Trajectory Planner 

**Authors**: Eugene Ku, Yiwei Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10330)  

**Abstract**: Autonomous driving in complex traffic requires planners that generalize beyond hand-crafted rules, motivating data-driven approaches that learn behavior from expert demonstrations. Diffusion-based trajectory planners have recently shown strong closed-loop performance by iteratively denoising a full-horizon plan, but they remain difficult to certify and can fail catastrophically in rare or out-of-distribution scenarios. To address this challenge, we present PC-Diffuser, a safety augmentation framework that embeds a certifiable, path-consistent barrier-function structure directly into the denoising loop of diffusion planning. The key idea is to make safety an intrinsic part of trajectory generation rather than a post-hoc fix: we enforce forward invariance along the rollout while preserving the diffusion model's intended path geometry. Specifically, PC-Diffuser (i) evaluates collision risk using a capsule-distance barrier function that better reflects vehicle geometry and reduces unnecessary conservativeness, (ii) converts denoised waypoints into dynamically feasible motion under a kinematic bicycle model, and (iii) applies a path-consistent safety filter that eliminates residual constraint violations without geometric distortion, so the corrected plan remains close to the learned distribution. By injecting these safety-consistent corrections at every denoising step and feeding the refined trajectory back into the diffusion process, PC-Diffuser enables iterative, context-aware safeguarding instead of post-hoc repair... 

---
# NasoVoce: A Nose-Mounted Low-Audibility Speech Interface for Always-Available Speech Interaction 

**Authors**: Jun Rekimoto, Yu Nishimura, Bojian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10324)  

**Abstract**: Silent and whispered speech offer promise for always-available voice interaction with AI, yet existing methods struggle to balance vocabulary size, wearability, silence, and noise robustness. We present NasoVoce, a nose-bridge-mounted interface that integrates a microphone and a vibration sensor. Positioned at the nasal pads of smart glasses, it unobtrusively captures both acoustic and vibration signals. The nasal bridge, close to the mouth, allows access to bone- and skin-conducted speech and enables reliable capture of low-volume utterances such as whispered speech. While the microphone captures high-quality audio, it is highly sensitive to environmental noise. Conversely, the vibration sensor is robust to noise but yields lower signal quality. By fusing these complementary inputs, NasoVoce generates high-quality speech robust against interference. Evaluation with Whisper Large-v2, PESQ, STOI, and MUSHRA ratings confirms improved recognition and quality. NasoVoce demonstrates the feasibility of a practical interface for always-available, continuous, and discreet AI voice conversations. 

---
# Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas 

**Authors**: Tim Schopf, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2603.10303)  

**Abstract**: Judging the novelty of research ideas is crucial for advancing science, enabling the identification of unexplored directions, and ensuring contributions meaningfully extend existing knowledge rather than reiterate minor variations. However, given the exponential growth of scientific literature, manually judging the novelty of research ideas through literature reviews is labor-intensive, subjective, and infeasible at scale. Therefore, recent efforts have proposed automated approaches for research idea novelty judgment. Yet, evaluation of these approaches remains largely inconsistent and is typically based on non-standardized human evaluations, hindering large-scale, comparable evaluations. To address this, we introduce RINoBench, the first comprehensive benchmark for large-scale evaluation of research idea novelty judgments. It comprises 1,381 research ideas derived from and judged by human experts as well as nine automated evaluation metrics designed to assess both rubric-based novelty scores and textual justifications of novelty judgments. Using this benchmark, we evaluate several state-of-the-art large language models (LLMs) on their ability to judge the novelty of research ideas. Our findings reveal that while LLM-generated reasoning closely mirrors human rationales, this alignment does not reliably translate into accurate novelty judgments, which diverge significantly from human gold standard judgments - even among leading reasoning-capable models. Data and code available at: this https URL. 

---
# Simulation-in-the-Reasoning (SiR): A Conceptual Framework for Empirically Grounded AI in Autonomous Transportation 

**Authors**: Wuping Xin  

**Link**: [PDF](https://arxiv.org/pdf/2603.10294)  

**Abstract**: Large Language Models (LLMs) have advanced reasoning through techniques like Chain-of-Thought (CoT). However, their reasoning largely re-mains textual and hypothetical, lacking empirical grounding in complex, dynamic domains like transportation. This paper introduces Simulation-in-the-Reasoning (SiR), a novel conceptual framework that embeds domain-specific simulators directly into the LLM reasoning loop. By treating intermediate reasoning steps as executable simulation experiments, SiR transforms LLM reasoning from narrative plausibility into a falsifiable, hypothesis-simulate-analyze workflow. We discuss applications, where LLM can formulate Intelligent Transport System (ITS) strategy hypotheses, invoke a traffic simulator via the Model Context Protocol (MCP), evaluate results under different demand patterns, and refine strategies through verification and aggregation. While implementing the framework is part of our ongoing work, this paper primarily establishes the conceptual foundation, discusses design considerations like API granularity, and outlines the vision of SiR as a cornerstone for interactive transportation digital twins. We argue that SiR represents a critical step towards trustworthy, empirically-validated AI for autonomous transportation systems. 

---
# Quantum entanglement provides a competitive advantage in adversarial games 

**Authors**: Peiyong Wang, Kieran Hymas, James Quach  

**Link**: [PDF](https://arxiv.org/pdf/2603.10289)  

**Abstract**: Whether uniquely quantum resources confer advantages in fully classical, competitive environments remains an open question. Competitive zero-sum reinforcement learning is particularly challenging, as success requires modelling dynamic interactions between opposing agents rather than static state-action mappings. Here, we conduct a controlled study isolating the role of quantum entanglement in a quantum-classical hybrid agent trained on Pong, a competitive Markov game. An 8-qubit parameterised quantum circuit serves as a feature extractor within a proximal policy optimisation framework, allowing direct comparison between separable circuits and architectures incorporating fixed (CZ) or trainable (IsingZZ) entangling gates. Entangled circuits consistently outperform separable counterparts with comparable parameter counts and, in low-capacity regimes, match or exceed classical multilayer perceptron baselines. Representation similarity analysis further shows that entangled circuits learn structurally distinct features, consistent with improved modelling of interacting state variables. These findings establish entanglement as a function resource for representation learning in competitive reinforcement learning. 

---
# Conversational AI-Enhanced Exploration System to Query Large-Scale Digitised Collections of Natural History Museums 

**Authors**: Yiyuan Wang, Andrew Johnston, Zoë Sadokierski, Rhiannon Stephens, Shane T. Ahyong  

**Link**: [PDF](https://arxiv.org/pdf/2603.10285)  

**Abstract**: Recent digitisation efforts in natural history museums have produced large volumes of collection data, yet their scale and scientific complexity often hinder public access and understanding. Conventional data management tools, such as databases, restrict exploration through keyword-based search or require specialised schema knowledge. This paper presents a system design that uses conversational AI to query nearly 1.7 million digitised specimen records from the life-science collections of the Australian Museum. Designed and developed through a human-centred design process, the system contains an interactive map for visual-spatial exploration and a natural-language conversational agent that retrieves detailed specimen data and answers collection-specific questions. The system leverages function-calling capabilities of contemporary large language models to dynamically retrieve structured data from external APIs, enabling fast, real-time interaction with extensive yet frequently updated datasets. Our work provides a new approach of connecting large museum collections with natural language-based queries and informs future designs of scientific AI agents for natural history museums. 

---
# Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework 

**Authors**: Rajesh Shrestha, Xiao Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10281)  

**Abstract**: While score-based generative models have emerged as powerful priors for solving inverse problems, directly integrating them into optimization algorithms such as ADMM remains nontrivial. Two central challenges arise: i) the mismatch between the noisy data manifolds used to train the score functions and the geometry of ADMM iterates, especially due to the influence of dual variables, and ii) the lack of convergence understanding when ADMM is equipped with score-based denoisers. To address the manifold mismatch issue, we propose ADMM plug-and-play (ADMM-PnP) with the AC-DC denoiser, a new framework that embeds a three-stage denoiser into ADMM: (1) auto-correction (AC) via additive Gaussian noise, (2) directional correction (DC) using conditional Langevin dynamics, and (3) score-based denoising. In terms of convergence, we establish two results: first, under proper denoiser parameters, each ADMM iteration is a weakly nonexpansive operator, ensuring high-probability fixed-point $\textit{ball convergence}$ using a constant step size; second, under more relaxed conditions, the AC-DC denoiser is a bounded denoiser, which leads to convergence under an adaptive step size schedule. Experiments on a range of inverse problems demonstrate that our method consistently improves solution quality over a variety of baselines. 

---
# Joint Imaging-ROI Representation Learning via Cross-View Contrastive Alignment for Brain Disorder Classification 

**Authors**: Wei Liang, Lifang He  

**Link**: [PDF](https://arxiv.org/pdf/2603.10253)  

**Abstract**: Brain imaging classification is commonly approached from two perspectives: modeling the full image volume to capture global anatomical context, or constructing ROI-based graphs to encode localized and topological interactions. Although both representations have demonstrated independent efficacy, their relative contributions and potential complementarity remain insufficiently understood. Existing fusion approaches are typically task-specific and do not enable controlled evaluation of each representation under consistent training settings. To address this gap, we propose a unified cross-view contrastive framework for joint imaging-ROI representation learning. Our method learns subject-level global (imaging) and local (ROI-graph) embeddings and aligns them in a shared latent space using a bidirectional contrastive objective, encouraging representations from the same subject to converge while separating those from different subjects. This alignment produces comparable embeddings suitable for downstream fusion and enables systematic evaluation of imaging-only, ROI-only, and joint configurations within a unified training protocol. Extensive experiments on the ADHD-200 and ABIDE datasets demonstrate that joint learning consistently improves classification performance over either branch alone across multiple backbone choices. Moreover, interpretability analyses reveal that imaging-based and ROI-based branches emphasize distinct yet complementary discriminative patterns, explaining the observed performance gains. These findings provide principled evidence that explicitly integrating global volumetric and ROI-level representations is a promising direction for neuroimaging-based brain disorder classification. The source code is available at this https URL. 

---
# DUCTILE: Agentic LLM Orchestration of Engineering Analysis in Product Development Practice 

**Authors**: Alejandro Pradas-Gomez, Arindam Brahma, Ola Isaksson  

**Link**: [PDF](https://arxiv.org/pdf/2603.10249)  

**Abstract**: Engineering analysis automation in product development relies on rigid interfaces between tools, data formats and documented processes. When these interfaces change, as they routinely do as the product evolves in the engineering ecosystem, the automation support breaks. This paper presents a DUCTILE (Delegated, User-supervised Coordination of Tool- and document-Integrated LLM-Enabled) agentic orchestration, an approach for developing, executing and evaluating LLM-based agentic automation support of engineering analysis tasks. The approach separates adaptive orchestration, performed by the LLM agent, from deterministic execution, performed by verified engineering tools. The agent interprets documented design practices, inspects input data and adapts the processing path, while the engineer supervises and exercises final judgment. DUCTILE is demonstrated on an industrial structural analysis task at an aerospace manufacturer, where the agent handled input deviations in format, units, naming conventions and methodology that would break traditional scripted pipelines. Evaluation against expert-defined acceptance criteria and deployment with practicing engineers confirm that the approach produces correct, methodologically compliant results across repeated independent runs. The paper discusses practical consequences of adopting agentic automation, including unintended effects on the nature of engineering work and the tension between removing mundane tasks and creating an exhausting supervisory role. 

---
# Intrinsic Numerical Robustness and Fault Tolerance in a Neuromorphic Algorithm for Scientific Computing 

**Authors**: Bradley H. Theilman, James B. Aimone  

**Link**: [PDF](https://arxiv.org/pdf/2603.10246)  

**Abstract**: The potential for neuromorphic computing to provide intrinsic fault tolerance has long been speculated, but the brain's robustness in neuromorphic applications has yet to be demonstrated. Here, we show that a previously described, natively spiking neuromorphic algorithm for solving partial differential equations is intrinsically tolerant to structural perturbations in the form of ablated neurons and dropped spikes. The tolerance band for these perturbations is large: we find that as many as 32 percent of the neurons and up to 90 percent of the spikes may be entirely dropped before a significant degradation in the accuracy results. Furthermore, this robustness is tunable through structural hyperparameters. This work demonstrates that the specific brain-like inspiration behind the algorithm contributes to a significant degree of robustness expected from brain-like neuromorphic algorithms. 

---
# Learning from Radio using Variational Quantum RF Sensing 

**Authors**: Ivana Nikoloska  

**Link**: [PDF](https://arxiv.org/pdf/2603.10239)  

**Abstract**: In modern wireless networks, radio channels serve a dual role. Whilst their primary function is to carry bits of information from a transmitter to a receiver, the intrinsic sensitivity of transmitted signals to the physical structure of the environment makes the channel a powerful source of knowledge about the world. In this paper, we consider an agent that learns about its environment using a quantum sensing probe, optimised using a quantum circuit, which interacts with the radio-frequency (RF) electromagnetic field. We use data obtained from a ray-tracer to train the quantum circuit and learning model and we provide extensive experiments under realistic conditions on a localisation task. We show that using quantum sensors to learn from radio signals can enable intelligent systems that require no channel measurements at deployment, remain sensitive to weak and obstructed RF signals, and can learn about the world despite operating with strictly less information than classical baselines. 

---
# Rethinking the Harmonic Loss via Non-Euclidean Distance Layers 

**Authors**: Maxwell Miller-Golub, Kamil Faber, Marcin Pietron, Panpan Zheng, Pasquale Minervini, Roberto Corizzo  

**Link**: [PDF](https://arxiv.org/pdf/2603.10225)  

**Abstract**: Cross-entropy loss has long been the standard choice for training deep neural networks, yet it suffers from interpretability limitations, unbounded weight growth, and inefficiencies that can contribute to costly training dynamics. The harmonic loss is a distance-based alternative grounded in Euclidean geometry that improves interpretability and mitigates phenomena such as grokking, or delayed generalization on the test set. However, the study of harmonic loss remains narrow: only Euclidean distance is explored, and no systematic evaluation of computational efficiency or sustainability was conducted. We extend harmonic loss by systematically investigating a broad spectrum of distance metrics as replacements for the Euclidean distance. We comprehensively evaluate distance-tailored harmonic losses on both vision backbones and large language models. Our analysis is framed around a three-way evaluation of model performance, interpretability, and sustainability. On vision tasks, cosine distances provide the most favorable trade-off, consistently improving accuracy while lowering carbon emissions, whereas Bray-Curtis and Mahalanobis further enhance interpretability at varying efficiency costs. On language models, cosine-based harmonic losses improve gradient and learning stability, strengthen representation structure, and reduce emissions relative to cross-entropy and Euclidean heads. Our code is available at: this https URL. 

---
# Robotic Ultrasound Makes CBCT Alive 

**Authors**: Feng Li, Ziyuan Li, Zhongliang Jiang, Nassir Navab, Yuan Bi  

**Link**: [PDF](https://arxiv.org/pdf/2603.10220)  

**Abstract**: Intraoperative Cone Beam Computed Tomography (CBCT) provides a reliable 3D anatomical context essential for interventional planning. However, its static nature fails to provide continuous monitoring of soft-tissue deformations induced by respiration, probe pressure, and surgical manipulation, leading to navigation discrepancies. We propose a deformation-aware CBCT updating framework that leverages robotic ultrasound as a dynamic proxy to infer tissue motion and update static CBCT slices in real time. Starting from calibration-initialized alignment with linear correlation of linear combination (LC2)-based rigid refinement, our method establishes accurate multimodal correspondence. To capture intraoperative dynamics, we introduce the ultrasound correlation UNet (USCorUNet), a lightweight network trained with optical flow-guided supervision to learn deformation-aware correlation representations, enabling accurate, real-time dense deformation field estimation from ultrasound streams. The inferred deformation is spatially regularized and transferred to the CBCT reference to produce deformation-consistent visualizations without repeated radiation exposure. We validate the proposed approach through deformation estimation and ultrasound-guided CBCT updating experiments. Results demonstrate real-time end-to-end CBCT slice updating and physically plausible deformation estimation, enabling dynamic refinement of static CBCT guidance during robotic ultrasound-assisted interventions. The source code is publicly available at this https URL. 

---
# A Diffusion Analysis of Policy Gradient for Stochastic Bandits 

**Authors**: Tor Lattimore  

**Link**: [PDF](https://arxiv.org/pdf/2603.10219)  

**Abstract**: We study a continuous-time diffusion approximation of policy gradient for $k$-armed stochastic bandits. We prove that with a learning rate $\eta = O(\Delta^2/\log(n))$ the regret is $O(k \log(k) \log(n) / \eta)$ where $n$ is the horizon and $\Delta$ the minimum gap. Moreover, we construct an instance with only logarithmically many arms for which the regret is linear unless $\eta = O(\Delta^2)$. 

---
# Multilingual AI-Driven Password Strength Estimation with Similarity-Based Detection 

**Authors**: Nikitha M. Palaniappan, Ying He  

**Link**: [PDF](https://arxiv.org/pdf/2603.10217)  

**Abstract**: Considering the rise of cyberattacks incidents worldwide, the need to ensure stronger passwords is necessary. Developing a password strength meter (PSM) can help users create stronger passwords when creating an account on an online platform. This research aimed to explore whether incorporating a non-English training dataset (specifically Indian) can improve the performance of a PSM. Findings show that PSMs can be improved by utilising learning of words from other languages. Another contribution of the research was to compare and provide an analysis of AI generated data (specifically by ChatGPT) and PassGAN (existing state-of-the-art model), proving that PassGAN-like tools may no longer be needed as the performance is higher using AI generated data. To further strengthen detection, a Jaro similarity-based matching mechanism was incorporated, enabling the classification of passwords that are highly similar to known weak passwords - this addresses limitations of direct matching techniques used in prior work. A final novel contribution is on developing a PSM tailored for Indian passwords, which has not been developed previously - this resulted in a near-perfect matching accuracy using a Jaro function value of 0.5. Although performance improvements were constrained by limited data and training, results suggest that using the ChatGPT dataset is a viable and effective strategy for developing secure, language-aware password strength meters. 

---
# Delta-K: Boosting Multi-Instance Generation via Cross-Attention Augmentation 

**Authors**: Zitong Wang, Zijun Shen, Haohao Xu, Zhengjie Luo, Weibin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10210)  

**Abstract**: While Diffusion Models excel in text-to-image synthesis, they often suffer from concept omission when synthesizing complex multi-instance scenes. Existing training-free methods attempt to resolve this by rescaling attention maps, which merely exacerbates unstructured noise without establishing coherent semantic representations. To address this, we propose Delta-K, a backbone-agnostic and plug-and-play inference framework that tackles omission by operating directly in the shared cross-attention Key space. Specifically, with Vision-language model, we extract a differential key $\Delta K$ that encodes the semantic signature of missing concepts. This signal is then injected during the early semantic planning stage of the diffusion process. Governed by a dynamically optimized scheduling mechanism, Delta-K grounds diffuse noise into stable structural anchors while preserving existing concepts. Extensive experiments demonstrate the generality of our approach: Delta-K consistently improves compositional alignment across both modern DiT models and classical U-Net architectures, without requiring spatial masks, additional training, or architectural modifications. 

---
# Adaptive Activation Cancellation for Hallucination Mitigation in Large Language Models 

**Authors**: Eric Yocam, Varghese Vaidyan, Gurcan Comert, Paris Kalathas, Yong Wang, Judith L. Mwakalonge  

**Link**: [PDF](https://arxiv.org/pdf/2603.10195)  

**Abstract**: Large Language Models frequently generate fluent but factually incorrect text. We propose Adaptive Activation Cancellation (AAC), a real-time inference-time framework that treats hallucination-associated neural activations as structured interference within the transformer residual stream, drawing an explicit analogy to classical adaptive noise cancellation from signal processing. The framework identifies Hallucination Nodes (H-Nodes) via layer-wise linear probing and suppresses them using a confidence-weighted forward hook during auto-regressive generation -- requiring no external knowledge, no fine-tuning, and no additional inference passes. Evaluated across OPT-125M, Phi-3-mini, and LLaMA 3-8B on TruthfulQA and HaluEval, the real-time hook is the only intervention that consistently improves downstream accuracy on all three scales. Critically, the method is strictly surgical: WikiText-103 perplexity and MMLU reasoning accuracy are preserved at exactly 0.0% degradation across all three model scales, a property that distinguishes AAC from interventions that trade fluency or general capability for factual improvement. On the LLaMA 3-8B scale, the hook additionally yields positive generation-level gains (MC1 +0.04; MC2 +0.003; Token-F1 +0.003) while achieving probe-space selectivity 5.94x - 3.5x higher than the ITI baseline -- demonstrating that targeted neuron-level suppression can simultaneously improve factual accuracy and preserve model capability. 

---
# MCP-in-SoS: Risk assessment framework for open-source MCP servers 

**Authors**: Pratyay Kumar, Miguel Antonio Guirao Aguilera, Srikathyayani Srikanteswara, Satyajayant Misra, Abu Saleh Md Tayeen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10194)  

**Abstract**: Model Context Protocol (MCP) servers have rapidly emerged over the past year as a widely adopted way to enable Large Language Model (LLM) agents to access dynamic, real-world tools. As MCP servers proliferate and become easy to adopt via open-source releases, understanding their security risks becomes essential for dependable production agent deployments. Recent work has developed MCP threat taxonomies, proposed mitigations, and demonstrated practical attacks. However, to the best of our knowledge, no prior study has conducted a systematic, large-scale assessment of weaknesses in open-source MCP servers. Motivated by this gap, we apply static code analysis to identify Common Weakness Enumeration (CWE) weaknesses and map them to common attack patterns and threat categories using the MITRE Common Attack Pattern Enumerations and Classifications (CAPEC) to ground risk in real-world threats. We then introduce a risk-assessment framework for the MCP landscape that combines these threats using a multi-metric scoring of likelihood and impact. Our findings show that many open-source MCP servers contain exploitable weaknesses that can compromise confidentiality, integrity, and availability, underscoring the need for secure-by-design MCP server development. 

---
# Compatibility at a Cost: Systematic Discovery and Exploitation of MCP Clause-Compliance Vulnerabilities 

**Authors**: Nanzi Yang, Weiheng Bai, Kangjie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10163)  

**Abstract**: The Model Context Protocol (MCP) is a recently proposed interoperability standard that unifies how AI agents connect with external tools and data sources. By defining a set of common client-server message exchange clauses, MCP replaces fragmented integrations with a standardized, plug-and-play framework. However, to be compatible with diverse AI agents, the MCP specification relaxes many behavioral constraints into optional clauses, leading to misuse-prone SDK implementation. We identify it as a new attack surface that allows adversaries to achieve multiple attacks (e.g, silent prompt injection, DoS, etc.), named as \emph{compatibility-abusing attacks}.
In this work, we present the first systematic framework for analyzing this new attack surface across multi-language MCP SDKs. First, we construct a universal and language-agnostic intermediate representation (IR) generator that normalizes SDKs of different languages. Next, based on the new IR, we propose auditable static analysis with LLM-guided semantic reasoning for cross-language/clause compliance analysis. Third, by formalizing the attack semantics of the MCP clauses, we build three attack modalities and develop a modality-guided pipeline to uncover exploitable non-compliance issues. 

---
# Mashup Learning: Faster Finetuning by Remixing Past Checkpoints 

**Authors**: Sofia Maria Lo Cicero Vaina, Artem Chumachenko, Max Ryabinin  

**Link**: [PDF](https://arxiv.org/pdf/2603.10156)  

**Abstract**: Finetuning on domain-specific data is a well-established method for enhancing LLM performance on downstream tasks. Training on each dataset produces a new set of model weights, resulting in a multitude of checkpoints saved in-house or on open-source platforms. However, these training artifacts are rarely reused for subsequent experiments despite containing improved model abilities for potentially similar tasks. In this paper, we propose Mashup Learning, a simple method to leverage the outputs of prior training runs to enhance model adaptation to new tasks. Our procedure identifies the most relevant historical checkpoints for a target dataset, aggregates them with model merging, and uses the result as an improved initialization for training. Across 8 standard LLM benchmarks, four models, and two collections of source checkpoints, Mashup Learning consistently improves average downstream accuracy by 0.5-5 percentage points over training from scratch. It also accelerates convergence, requiring 41-46% fewer training steps and up to 37% less total wall-clock time to match from-scratch accuracy, including all selection and merging overhead. 

---
# Social Knowledge for Cross-Domain User Preference Modeling 

**Authors**: Nir Lotan, Adir Solomon, Ido Guy, Einat Minkov  

**Link**: [PDF](https://arxiv.org/pdf/2603.10148)  

**Abstract**: We demonstrate that user preferences can be represented and predicted across topical domains using large-scale social modeling. Given information about popular entities favored by a user, we project the user into a social embedding space learned from a large-scale sample of the Twitter (now X) network. By representing both users and popular entities in a joint social space, we can assess the relevance of candidate entities (e.g., music artists) using cosine similarity within this embedding space. A comprehensive evaluation using link prediction experiments shows that this method achieves effective personalization in zero-shot setting, when no user feedback is available for entities in the target domain, yielding substantial improvements over a strong popularity-based baseline. In-depth analysis further illustrates that socio-demographic factors encoded in the social embeddings are correlated with user preferences across domains. Finally, we argue and demonstrate that the proposed approach can facilitate social modeling of end users using large language models (LLMs). 

---
# The Generation-Recognition Asymmetry: Six Dimensions of a Fundamental Divide in Formal Language Theory 

**Authors**: Romain Peyrichou  

**Link**: [PDF](https://arxiv.org/pdf/2603.10139)  

**Abstract**: Every formal grammar defines a language and can in principle be used in three ways: to generate strings (production), to recognize them (parsing), or -- given only examples -- to infer the grammar itself (grammar induction). Generation and recognition are extensionally equivalent -- they characterize the same set -- but operationally asymmetric in multiple independent ways. Inference is a qualitatively harder problem: it does not have access to a known grammar. Despite the centrality of this triad to compiler design, natural language processing, and formal language theory, no survey has treated it as a unified, multidimensional phenomenon. We identify six dimensions along which generation and recognition diverge: computational complexity, ambiguity, directionality, information availability, grammar inference, and temporality. We show that the common characterization "generation is easy, parsing is hard" is misleading: unconstrained generation is trivial, but generation under constraints can be NP-hard. The real asymmetry is that parsing is always constrained (the input is given) while generation need not be. Two of these dimensions -- directionality and temporality -- have not previously been identified as dimensions of the generation-recognition asymmetry. We connect the temporal dimension to the surprisal framework of Hale (2001) and Levy (2008), arguing that surprisal formalizes the temporal asymmetry between a generator (surprisal = 0) and a parser that predicts under uncertainty (surprisal > 0). We review bidirectional systems in NLP and observe that bidirectionality has been available for fifty years yet has not transferred to most domain-specific applications. We conclude with a discussion of large language models, which architecturally unify generation and recognition while operationally preserving the asymmetry. 

---
# AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models 

**Authors**: Yutong Hu, Jan-Nico Zaech, Nikolay Nikolov, Yuanqi Yao, Sombit Dey, Giuliano Albanese, Renaud Detry, Luc Van Gool, Danda Paudel  

**Link**: [PDF](https://arxiv.org/pdf/2603.10126)  

**Abstract**: We propose a standalone autoregressive (AR) Action Expert that generates actions as a continuous causal sequence while conditioning on refreshable vision-language prefixes. In contrast to existing Vision-Language-Action (VLA) models and diffusion policies that reset temporal context with each new observation and predict actions reactively, our Action Expert maintains its own history through a long-lived memory and is inherently context-aware. This structure addresses the frequency mismatch between fast control and slow reasoning, enabling efficient independent pretraining of kinematic syntax and modular integration with heavy perception backbones, naturally ensuring spatio-temporally consistent action generation across frames. To synchronize these asynchronous hybrid V-L-A modalities, we utilize a re-anchoring mechanism that mathematically accounts for perception staleness during both training and inference. Experiments on simulated and real-robot manipulation tasks demonstrate that the proposed method can effectively replace traditional chunk-based action heads for both specialist and generalist policies. AR-VLA exhibits superior history awareness and substantially smoother action trajectories while maintaining or exceeding the task success rates of state-of-the-art reactive VLAs. Overall, our work introduces a scalable, context-aware action generation schema that provides a robust structural foundation for training effective robotic policies. 

---
# Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias 

**Authors**: Borun D Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2603.10123)  

**Abstract**: The ``Lost in the Middle'' phenomenon -- a U-shaped performance curve where LLMs retrieve well from the beginning and end of a context but fail in the middle -- is widely attributed to learned Softmax artifacts or the distance-decay of positional encodings like RoPE. This paper makes a single, precise claim: \emph{the U-shape is already present at initialization, before any training or positional encoding takes effect.} It is an inherent geometric property of the causal decoder with residual connections.
We model multi-layer causal attention as iterated powers of the Cesàro matrix and derive the exact closed-form influence density in the continuous limit. Causal masking forces a logarithmic divergence of gradient influence at the start of the prompt (the Primacy Tail), while residual connections create an isolated $\mathcal{O}(1)$ anchor at the final token (the Recency Delta). Between these extremes lies a factorial dead zone of order $\mathcal{O}(1/(H{-}1)!)$, where $H$ is the network depth, making middle-context retrieval and training structurally hostile. We validate empirically that untrained Qwen2 and GPT-2 architectures exhibit this U-shape at Step~0, and that it is identical with or without RoPE. Comparing initialized and pretrained networks, we show that standard training does not overcome the topological valley, confirming that the U-shape persists as an architectural baseline under standard pretraining objectives.
We do not claim that this bias is insurmountable, nor that interventions such as RoPE modifications are useless. We establish what the baseline is and where it comes from, so that future efforts to overcome it can be precisely targeted. 

---
# CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR 

**Authors**: Sijia Cui, Pengyu Cheng, Jiajun Song, Yongbo Gai, Guojun Zhang, Zhechao Yu, Jianhe Lin, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10101)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capacity of Large Language Models (LLMs). However, RLVR solely relies on final answers as outcome rewards, neglecting the correctness of intermediate reasoning steps. Training on these process-wrong but outcome-correct rollouts can lead to hallucination and answer-copying, severely undermining the model's generalization and robustness. To address this, we incorporate a Contrastive Learning mechanism into the Policy Optimization (CLIPO) to generalize the RLVR process. By optimizing a contrastive loss over successful rollouts, CLIPO steers the LLM to capture the invariant structure shared across correct reasoning paths. This provides a more robust cross-trajectory regularization than the original single-path supervision in RLVR, effectively mitigating step-level reasoning inconsistencies and suppressing hallucinatory artifacts. In experiments, CLIPO consistently improves multiple RLVR baselines across diverse reasoning benchmarks, demonstrating uniform improvements in generalization and robustness for policy optimization of LLMs. Our code and training recipes are available at this https URL. 

---
# Hardware Efficient Approximate Convolution with Tunable Error Tolerance for CNNs 

**Authors**: Vishal Shashidhar, Anupam Kumari, Roy P Paily  

**Link**: [PDF](https://arxiv.org/pdf/2603.10100)  

**Abstract**: Modern CNNs' high computational demands hinder edge deployment, as traditional ``hard'' sparsity (skipping mathematical zeros) loses effectiveness in deep layers or with smooth activations like Tanh. We propose a ``soft sparsity'' paradigm using a hardware efficient Most Significant Bit (MSB) proxy to skip negligible non-zero multiplications. Integrated as a custom RISC-V instruction and evaluated on LeNet-5 (MNIST), this method reduces ReLU MACs by 88.42% and Tanh MACs by 74.87% with zero accuracy loss--outperforming zero-skipping by 5x. By clock-gating inactive multipliers, we estimate power savings of 35.2\% for ReLU and 29.96\% for Tanh. While memory access makes power reduction sub-linear to operation savings, this approach significantly optimizes resource-constrained inference. 

---
# Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with Large Language Models 

**Authors**: Daniel Hennes, Zun Li, John Schultz, Marc Lanctot  

**Link**: [PDF](https://arxiv.org/pdf/2603.10098)  

**Abstract**: Recent advances in multi-agent reinforcement learning, particularly Policy-Space Response Oracles (PSRO), have enabled the computation of approximate game-theoretic equilibria in increasingly complex domains. However, these methods rely on deep reinforcement learning oracles that produce `black-box' neural network policies, making them difficult to interpret, trust or debug. We introduce Code-Space Response Oracles (CSRO), a novel framework that addresses this challenge by replacing RL oracles with Large Language Models (LLMs). CSRO reframes the best response computation as a code generation task, prompting an LLM to generate policies directly as human-readable code. This approach not only yields inherently interpretable policies but also leverages the LLM's pretrained knowledge to discover complex, human-like strategies. We explore multiple ways to construct and enhance an LLM-based oracle: zero-shot prompting, iterative refinement and \emph{AlphaEvolve}, a distributed LLM-based evolutionary system. We demonstrate that CSRO achieves performance competitive with baselines while producing a diverse set of explainable policies. Our work presents a new perspective on multi-agent learning, shifting the focus from optimizing opaque policy parameters to synthesizing interpretable algorithmic behavior. 

---
# Equivariant Asynchronous Diffusion: An Adaptive Denoising Schedule for Accelerated Molecular Conformation Generation 

**Authors**: Junyi An, Chao Qu, Yun-Fei Shi, Zhijian Zhou, Fenglei Cao, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2603.10093)  

**Abstract**: Recent 3D molecular generation methods primarily use asynchronous auto-regressive or synchronous diffusion models. While auto-regressive models build molecules sequentially, they're limited by a short horizon and a discrepancy between training and inference. Conversely, synchronous diffusion models denoise all atoms at once, offering a molecule-level horizon but failing to capture the causal relationships inherent in hierarchical molecular structures. We introduce Equivariant Asynchronous Diffusion (EAD) to overcome these limitations. EAD is a novel diffusion model that combines the strengths of both approaches: it uses an asynchronous denoising schedule to better capture molecular hierarchy while maintaining a molecule-level horizon. Since these relationships are often complex, we propose a dynamic scheduling mechanism to adaptively determine the denoising timestep. Experimental results show that EAD achieves state-of-the-art performance in 3D molecular generation. 

---
# Execution Is the New Attack Surface: Survivability-Aware Agentic Crypto Trading with OpenClaw-Style Local Executors 

**Authors**: Ailiya Borjigin, Igor Stadnyk, Ben Bilski, Serhii Hovorov, Sofiia Pidturkina  

**Link**: [PDF](https://arxiv.org/pdf/2603.10092)  

**Abstract**: OpenClaw-style agent stacks turn language into privileged execution: LLM intents flow through tool interception, policy gates, and a local executor. In parallel, skill marketplaces such as this http URL make capability acquisition as easy as installing skills and CLIs, creating a growing capability supply chain. Together, these trends shift the dominant safety failure mode from "wrong answers" to execution-induced loss, where untrusted prompts, compromised skills, or narrative manipulation can trigger real trades and irreversible side effects. We propose Survivability-Aware Execution (SAE), an execution-layer survivability standard for OpenClaw-style systems and skill-enabled agents. SAE sits as middleware between a strategy engine (LLM or non-LLM) and the exchange executor. It defines an explicit execution contract (ExecutionRequest, ExecutionContext, ExecutionDecision) and enforces non-bypassable last-mile invariants: projection-based exposure budgets, cooldown and order-rate limits, slippage bounds, staged execution, and tool/venue allowlists. To make delegated execution testable under supply-chain risk, we operationalize the Delegation Gap (DG) via a logged Intended Policy Spec that enables deterministic out-of-scope labeling and reproducible DG metrics. On an offline replay using official Binance USD-M BTCUSDT/ETHUSDT perpetual data (15m; 2025-09-01--2025-12-01, incl. funding), SAE improves survivability: MDD drops from 0.4643 to 0.0319 (Full; 93.1%), |CVaR_0.99| shrinks from 4.025e-3 to ~1.02e-4 (~97.5%), and DG loss proxy falls from 0.647 to 0.019 (~97.0%). AttackSuccess decreases from 1.00 to 0.728 with zero FalseBlock in this run. Block bootstrap, paired Wilcoxon, and two-proportion tests confirm the shifts. SAE reframes agentic trading safety for the OpenClaw+skills era: treat upstream intent and skills as untrusted, and enforce survivability where actions become side effects. 

---
# Multi-Stream Perturbation Attack: Breaking Safety Alignment of Thinking LLMs Through Concurrent Task Interference 

**Authors**: Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10091)  

**Abstract**: The widespread adoption of thinking mode in large language models (LLMs) has significantly enhanced complex task processing capabilities while introducing new security risks. When subjected to jailbreak attacks, the step-by-step reasoning process may cause models to generate more detailed harmful content. We observe that thinking mode exhibits unique vulnerabilities when processing interleaved multiple tasks. Based on this observation, we propose multi-stream perturbation attack, which generates superimposed interference by interweaving multiple task streams within a single prompt. We design three perturbation strategies: multi-stream interleaving, inversion perturbation, and shape transformation, which disrupt the thinking process through concurrent task interleaving, character reversal, and format constraints respectively. On JailbreakBench, AdvBench, and HarmBench datasets, our method achieves attack success rates exceeding most methods across mainstream models including Qwen3 series, DeepSeek, Qwen3-Max, and Gemini 2.5 Flash. Experiments show thinking collapse rates and response repetition rates reach up to 17% and 60% respectively, indicating multi-stream perturbation not only bypasses safety mechanisms but also causes thinking process collapse or repetitive outputs. 

---
# ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping 

**Authors**: Zijian Zhu, Fei Ren, Zhanhong Tan, Kaisheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.10088)  

**Abstract**: Diffusion large language models (dLLMs) are emerging as a promising alternative to autoregressive models (ARMs) due to their ability to capture bidirectional context and the potential for parallel generation. Despite the advantages, dLLM inference remains computationally expensive as the full input context is processed at every iteration. In this work, we analyze the generation dynamics of dLLMs and find that intermediate representations, including key, value, and hidden states, change only subtly across successive iterations. Leveraging this insight, we propose \textbf{ES-dLLM}, a training-free inference acceleration framework for dLLM that reduces computation by skipping tokens in early layers based on the estimated importance. Token importance is computed with intermediate tensor variation and confidence scores of previous iterations. Experiments on LLaDA-8B and Dream-7B demonstrate that ES-dLLM achieves throughput of up to 226.57 and 308.51 tokens per second (TPS), respectively, on an NVIDIA H200 GPU, delivering 5.6$\times$ to 16.8$\times$ speedup over the vanilla implementation and up to 1.85$\times$ over the state-of-the-art caching method, while preserving generation quality. 

---
# KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization 

**Authors**: Qitong Sun, Jun Han, Tianlin Li, Zhe Tang, Sheng Chen, Fei Yang, Aishan Liu, Xianglong Liu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10085)  

**Abstract**: Improving GPU kernel efficiency is crucial for advancing AI systems. Recent work has explored leveraging large language models (LLMs) for GPU kernel generation and optimization. However, existing LLM-based kernel optimization pipelines typically rely on opaque, implicitly learned heuristics within the LLMs to determine optimization strategies. This leads to inefficient trial-and-error and weakly interpretable optimizations. Our key insight is to replace implicit heuristics with expert optimization skills that are knowledge-driven and aware of task trajectories. Specifically, we present KernelSkill, a multi-agent framework with a dual-level memory architecture. KernelSkill operates by coordinating agents with long-term memory of reusable expert skills and short-term memory to prevent repetitive backtracking. On KernelBench Levels 1-3, KernelSkill achieves a 100% success rate and average speedups of 5.44x, 2.82x, and 1.92x over Torch Eager on Levels 1, 2, and 3, respectively, outperforming prior baselines. Code is available at this https URL. 

---
# Digging Deeper: Learning Multi-Level Concept Hierarchies 

**Authors**: Oscar Hill, Mateo Espinosa Zarlenga, Mateja Jamnik  

**Link**: [PDF](https://arxiv.org/pdf/2603.10084)  

**Abstract**: Although concept-based models promise interpretability by explaining predictions with human-understandable concepts, they typically rely on exhaustive annotations and treat concepts as flat and independent. To circumvent this, recent work has introduced Hierarchical Concept Embedding Models (HiCEMs) to explicitly model concept relationships, and Concept Splitting to discover sub-concepts using only coarse annotations. However, both HiCEMs and Concept Splitting are restricted to shallow hierarchies. We overcome this limitation with Multi-Level Concept Splitting (MLCS), which discovers multi-level concept hierarchies from only top-level supervision, and Deep-HiCEMs, an architecture that represents these discovered hierarchies and enables interventions at multiple levels of abstraction. Experiments across multiple datasets show that MLCS discovers human-interpretable concepts absent during training and that Deep-HiCEMs maintain high accuracy while supporting test-time concept interventions that can improve task performance. 

---
# Amnesia: Adversarial Semantic Layer Specific Activation Steering in Large Language Models 

**Authors**: Ali Raza, Gurang Gupta, Nikolay Matyunin, Jibesh Patra  

**Link**: [PDF](https://arxiv.org/pdf/2603.10080)  

**Abstract**: Warning: This article includes red-teaming experiments, which contain examples of compromised LLM responses that may be offensive or upsetting.
Large Language Models (LLMs) have the potential to create harmful content, such as generating sophisticated phishing emails and assisting in writing code of harmful computer viruses. Thus, it is crucial to ensure their safe and responsible response generation. To reduce the risk of generating harmful or irresponsible content, researchers have developed techniques such as reinforcement learning with human feedback to align LLM's outputs with human values and preferences. However, it is still undetermined whether such measures are sufficient to prevent LLMs from generating interesting responses. In this study, we propose Amnesia, a lightweight activation-space adversarial attack that manipulates internal transformer states to bypass existing safety mechanisms in open-weight LLMs. Through experimental analysis on state-of-the-art, open-weight LLMs, we demonstrate that our attack effectively circumvents existing safeguards, enabling the generation of harmful content without the need for any fine-tuning or additional training. Our experiments on benchmark datasets show that the proposed attack can induce various antisocial behaviors in LLMs. These findings highlight the urgent need for more robust security measures in open-weight LLMs and underscore the importance of continued research to prevent their potential misuse. 

---
# TASER: Task-Aware Spectral Energy Refine for Backdoor Suppression in UAV Swarms Decentralized Federated Learning 

**Authors**: Sizhe Huang, Shujie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10075)  

**Abstract**: As backdoor attacks in UAV-based decentralized federated learning (DFL) grow increasingly stealthy and sophisticated, existing defenses have likewise escalated in complexity. Yet these defenses, which rely heavily on outlier detection, remain vulnerable to carefully crafted backdoors. In UAV-DFL, the lack of global coordination and limited resources further render outlier-based defenses impractical. Against this backdrop, gradient spectral analysis offers a promising alternative. While prior work primarily leverages low-frequency coefficients for pairwise comparisons, it neglects to analyze the intrinsic spectral characteristics of backdoor gradients. Through empirical analysis of existing stealthy attacks, we reveal a key insight: the more effort attackers invest in mimicking benign behaviors, the more distinct the spectral concentration becomes. Motivated by this, we propose Task-Aware Spectral Energy Refine (TASER) -- a decentralized defense framework. To our knowledge, this is the first efficient backdoor defense that utilizes spectral concentration instead of complex outlier detection, enabling mitigation of stealthy attacks by structurally disrupting the backdoor task. To suppress the backdoor task, TASER preserves main-task-relevant frequency coefficients and discards others. We provide theoretical guarantees and demonstrate through experiments that TASER remains effective against stealthy backdoor attacks that bypass outlier-based defenses, achieving attack success rate below 20% and accuracy loss under 5%. 

---
# Marginals Before Conditionals 

**Authors**: Mihir Sahasrabudhe  

**Link**: [PDF](https://arxiv.org/pdf/2603.10074)  

**Abstract**: We construct a minimal task that isolates conditional learning in neural networks: a surjective map with K-fold ambiguity, resolved by a selector token z, so H(A | B) = log K while H(A | B, z) = 0. The model learns the marginal P(A | B) first, producing a plateau at exactly log K, before acquiring the full conditional in a sharp, collective transition. The plateau has a clean decomposition: height = log K (set by ambiguity), duration = f(D) (set by dataset size D, not K). Gradient noise stabilizes the marginal solution: higher learning rates monotonically slow the transition (3.6* across a 7* {\eta} range at fixed throughput), and batch-size reduction delays escape, consistent with an entropic force opposing departure from the low-gradient marginal. Internally, a selector-routing head assembles during the plateau, leading the loss transition by ~50% of the waiting time. This is the Type 2 directional asymmetry of Papadopoulos et al. [2024], measured dynamically: we track the excess risk from log K to zero and characterize what stabilizes it, what triggers its collapse, and how long it takes. 

---
# Why LLMs Fail: A Failure Analysis and Partial Success Measurement for Automated Security Patch Generation 

**Authors**: Amir Al-Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2603.10072)  

**Abstract**: Large Language Models (LLMs) show promise for Automated Program Repair (APR), yet their effectiveness on security vulnerabilities remains poorly characterized. This study analyzes 319 LLM-generated security patchesacross 64 Java vulnerabilities from the Vul4J benchmark. Using tri-axis evaluation (compilation, security via PoV tests, functionality via test suites), the analysis reveals that only 24.8% of patches achieve full correctness, while 51.4% fail both security and functionality. The dominant failure mode is semantic misunderstanding: LLMs produce syntactically valid code but apply incorrect repair strategies. The proposed Security Repair Score (SRS) quantifies this gap, showing LLMs preserve functionality (mean 0.832) but struggle with security (mean 0.251). Vulnerability type strongly predicts difficulty, with fix rates ranging from 0% (input validation) to 45% (infinite loop). These findings demonstrate that LLM security patches require rigorous validation before deployment. 

---
# Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models 

**Authors**: Anurag Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2603.10071)  

**Abstract**: Time series foundation models (TSFMs) are increasingly deployed in high-stakes domains, yet their internal representations remain opaque. We present the first application of sparse autoencoders (SAEs) to a TSFM, training TopK SAEs on activations of Chronos-T5-Large (710M parameters) across six layers. Through 392 single-feature ablation experiments, we establish that every ablated feature produces a positive CRPS degradation, confirming causal relevance. Our analysis reveals a depth-dependent hierarchy: early encoder layers encode low-level frequency features, the mid-encoder concentrates causally critical change-detection features, and the final encoder compresses a rich but less causally important taxonomy of temporal concepts. The most critical features reside in the mid-encoder (max single-feature Delta CRPS = 38.61), not in the semantically richest final encoder layer, where progressive ablation paradoxically improves forecast quality. These findings demonstrate that mechanistic interpretability transfers effectively to TSFMs and that Chronos-T5 relies on abrupt-dynamics detection rather than periodic pattern recognition. 

---
# ADVERSA: Measuring Multi-Turn Guardrail Degradation and Judge Reliability in Large Language Models 

**Authors**: Harry Owiredu-Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2603.10068)  

**Abstract**: Most adversarial evaluations of large language model (LLM) safety assess single prompts and report binary pass/fail outcomes, which fails to capture how safety properties evolve under sustained adversarial interaction. We present ADVERSA, an automated red-teaming framework that measures guardrail degradation dynamics as continuous per-round compliance trajectories rather than discrete jailbreak events. ADVERSA uses a fine-tuned 70B attacker model (ADVERSA-Red, Llama-3.1-70B-Instruct with QLoRA) that eliminates the attacker-side safety refusals that render off-the-shelf models unreliable as attackers, scoring victim responses on a structured 5-point rubric that treats partial compliance as a distinct measurable state.
We report a controlled experiment across three frontier victim models (Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.2) using a triple-judge consensus architecture in which judge reliability is measured as a first-class research outcome rather than assumed. Across 15 conversations of up to 10 adversarial rounds, we observe a 26.7% jailbreak rate with an average jailbreak round of 1.25, suggesting that in this evaluation setting, successful jailbreaks were concentrated in early rounds rather than accumulating through sustained pressure. We document inter-judge agreement rates, self-judge scoring tendencies, attacker drift as a failure mode in fine-tuned attackers deployed out of their training distribution, and attacker refusals as a previously-underreported confound in victim resistance measurement. All limitations are stated explicitly. Attack prompts are withheld per responsible disclosure policy; all other experimental artifacts are released. 

---
# HTMuon: Improving Muon via Heavy-Tailed Spectral Correction 

**Authors**: Tianyu Pang, Yujie Fang, Zihang Liu, Shenyang Deng, Lei Hsiung, Shuhua Yu, Yaoqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10067)  

**Abstract**: Muon has recently shown promising results in LLM training. In this work, we study how to further improve Muon. We argue that Muon's orthogonalized update rule suppresses the emergence of heavy-tailed weight spectra and over-emphasizes the training along noise-dominated directions. Motivated by the Heavy-Tailed Self-Regularization (HT-SR) theory, we propose HTMuon. HTMuon preserves Muon's ability to capture parameter interdependencies while producing heavier-tailed updates and inducing heavier-tailed weight spectra. Experiments on LLM pretraining and image classification show that HTMuon consistently improves performance over state-of-the-art baselines and can also serve as a plug-in on top of existing Muon variants. For example, on LLaMA pretraining on the C4 dataset, HTMuon reduces perplexity by up to $0.98$ compared to Muon. We further theoretically show that HTMuon corresponds to steepest descent under the Schatten-$q$ norm constraint and provide convergence analysis in smooth non-convex settings. The implementation of HTMuon is available at this https URL. 

---
# The Epistemic Support-Point Filter: Jaynesian Maximum Entropy Meets Popperian Falsification 

**Authors**: Moriba Kemessia Jah  

**Link**: [PDF](https://arxiv.org/pdf/2603.10065)  

**Abstract**: The Epistemic Support-Point Filter (ESPF) was designed around a single epistemological commitment: be quick to embrace ignorance and slow to assert certainty. This paper proves that this commitment has a precise mathematical form and that the ESPF is the unique optimal filter implementing it within the class of epistemically admissible evidence-only filters. The ESPF synthesizes two complementary principles acting at different phases of the recursion. In propagation, it enacts Jaynesian maximum entropy: the support spreads as widely as the dynamics allow, assuming maximal ignorance consistent with known constraints. In the measurement update, it enacts Popperian falsification: hypotheses are eliminated by evidence alone. Any rule incorporating prior possibility is strictly suboptimal and risks race-to-bottom bias. The optimality criterion is possibilistic minimax entropy: among all evidence-only selection rules, minimum-q selection minimizes log det(MVEE), the worst-case possibilistic entropy. Three lemmas establish the result: the Possibilistic Entropy Lemma identifies the ignorance functional; the Possibilistic Cramér-Rao Lemma bounds entropy reduction per measurement; the Evidence-Optimality Lemma proves minimum-q selection is the unique minimizer. The ESPF differs from Bayesian filters by minimizing worst-case epistemic ignorance rather than expected uncertainty. The Kalman filter is recovered in the Gaussian limit. Numerical validation over a 2-day 877-step Smolyak Level-3 orbital tracking run confirms the regime structure under both nominal and stress conditions. 

---
# Multi-Agent Memory from a Computer Architecture Perspective: Visions and Challenges Ahead 

**Authors**: Zhongming Yu, Naicheng Yu, Hejia Zhang, Wentao Ni, Mingrui Yin, Jiaying Yang, Yujie Zhao, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10062)  

**Abstract**: As LLM agents evolve into collaborative multi-agent systems, their memory requirements grow rapidly in complexity. This position paper frames multi-agent memory as a computer architecture problem. We distinguish shared and distributed memory paradigms, propose a three-layer memory hierarchy (I/O, cache, and memory), and identify two critical protocol gaps: cache sharing across agents and structured memory access control. We argue that the most pressing open challenge is multi-agent memory consistency. Our architectural framing provides a foundation for building reliable, scalable multi-agent systems. 

---
# Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents 

**Authors**: Abhinaba Basu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10060)  

**Abstract**: AI agents that execute tasks via tool calls frequently hallucinate results - fabricating tool executions, misstating output counts, or presenting inferences as facts. Recent approaches to verifiable AI inference rely on zero-knowledge proofs, which provide cryptographic guarantees but impose minutes of proving time per query, making them impractical for interactive agents. We propose NabaOS, a lightweight verification framework inspired by Indian epistemology (Nyaya Shastra), which classifies every claim in an LLM response by its epistemic source (pramana): direct tool output (pratyaksha), inference (anumana), external testimony (shabda), absence (abhava), or ungrounded opinion. Our runtime generates HMAC-signed tool execution receipts that the LLM cannot forge, then cross-references claims against these receipts to detect hallucinations in real time. We evaluate on NyayaVerifyBench, a new benchmark of 1,800 agent response scenarios across four languages with injected hallucinations of six types. NabaOS detects 94.2% of fabricated tool references, 87.6% of count misstatements, and 91.3% of false absence claims, with <15ms verification overhead per response. For deep delegation (agents performing multi-step web tasks), our cross-checking protocol catches 78.4% of URL fabrications via independent re-fetching. We compare against five approaches: zkLLM (cryptographic proofs, 180s/query), TOPLOC (locality-sensitive hashing), SPEX (sampling-based proof of execution), tensor commitments, and self-consistency checking. NabaOS achieves the best cost-latency-coverage trade-off for interactive agents: 94.2% coverage at <15ms versus zkLLM's near-perfect coverage at 180,000ms. For interactive agents, practical receipt-based verification provides better cost-benefit than cryptographic proofs, and epistemic classification gives users actionable trust signals rather than binary judgments. 

---
# SBOMs into Agentic AIBOMs: Schema Extensions, Agentic Orchestration, and Reproducibility Evaluation 

**Authors**: Petar Radanliev, Carsten Maple, Omar Santos, Kayvan Atefi  

**Link**: [PDF](https://arxiv.org/pdf/2603.10057)  

**Abstract**: Software supply-chain security requires provenance mechanisms that support reproducibility and vulnerability assessment under dynamic execution conditions. Conventional Software Bills of Materials (SBOMs) provide static dependency inventories but cannot capture runtime behaviour, environment drift, or exploitability context. This paper introduces agentic Artificial Intelligence Bills of Materials (AIBOMs), extending SBOMs into active provenance artefacts through autonomous, policy-constrained reasoning. We present an agentic AIBOM framework based on a multi-agent architecture comprising (i) a baseline environment reconstruction agent (MCP), (ii) a runtime dependency and drift-monitoring agent (A2A), and (iii) a policy-aware vulnerability and VEX reasoning agent (AGNTCY). These agents generate contextual exploitability assertions by combining runtime execution evidence, dependency usage, and environmental mitigations with ISO/IEC 20153:2025 Common Security Advisory Framework (CSAF) v2.0 semantics. Exploitability is expressed via structured VEX assertions rather than enforcement actions. The framework introduces minimal, standards-aligned schema extensions to CycloneDX and SPDX, capturing execution context, dependency evolution, and agent decision provenance while preserving interoperability. Evaluation across heterogeneous analytical workloads demonstrates improved runtime dependency capture, reproducibility fidelity, and stability of vulnerability interpretation compared with established provenance systems, with low computational overhead. Ablation studies confirm that each agent contributes distinct capabilities unavailable through deterministic automation. 

---
# Training Language Models via Neural Cellular Automata 

**Authors**: Dan Lee, Seungwook Han, Akarsh Kumar, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2603.10055)  

**Abstract**: Pre-training is crucial for large language models (LLMs), as it is when most representations and capabilities are acquired. However, natural language pre-training has problems: high-quality text is finite, it contains human biases, and it entangles knowledge with reasoning. This raises a fundamental question: is natural language the only path to intelligence? We propose using neural cellular automata (NCA) to generate synthetic, non-linguistic data for pre-pre-training LLMs--training on synthetic-then-natural language. NCA data exhibits rich spatiotemporal structure and statistics resembling natural language while being controllable and cheap to generate at scale. We find that pre-pre-training on only 164M NCA tokens improves downstream language modeling by up to 6% and accelerates convergence by up to 1.6x. Surprisingly, this even outperforms pre-pre-training on 1.6B tokens of natural language from Common Crawl with more compute. These gains also transfer to reasoning benchmarks, including GSM8K, HumanEval, and BigBench-Lite. Investigating what drives transfer, we find that attention layers are the most transferable, and that optimal NCA complexity varies by domain: code benefits from simpler dynamics, while math and web text favor more complex ones. These results enable systematic tuning of the synthetic distribution to target domains. More broadly, our work opens a path toward more efficient models with fully synthetic pre-training. 

---
# Where Do Flow Semantics Reside? A Protocol-Native Tabular Pretraining Paradigm for Encrypted Traffic Classification 

**Authors**: Sizhe Huang, Shujie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10051)  

**Abstract**: Self-supervised masked modeling shows promise for encrypted traffic classification by masking and reconstructing raw bytes. Yet recent work reveals these methods fail to reduce reliance on labeled data despite costly pretraining: under frozen encoder evaluation, accuracy drops from greater than 0.9 to less than 0.47. We argue the root cause is inductive bias mismatch: flattening traffic into byte sequences destroys protocol-defined semantics. We identify three specific issues: 1) field unpredictability, random fields like this http URL are unlearnable yet treated as reconstruction targets; 2) embedding confusion, semantically distinct fields collapse into a unified embedding space; 3) metadata loss, capture-time metadata essential for temporal analysis is discarded. To address this, we propose a protocol-native paradigm that treats protocol-defined field semantics as architectural priors, reformulating the task to align with the data's intrinsic tabular modality rather than incrementally adapting sequence-based architectures. Instantiating this paradigm, we introduce FlowSem-MAE, a tabular masked autoencoder built on Flow Semantic Units (FSUs). It features predictability-guided filtering that focuses on learnable FSUs, FSU-specific embeddings to preserve field boundaries, and dual-axis attention to capture intra-packet and temporal patterns. FlowSem-MAE significantly outperforms state-of-the-art across datasets. With only half labeled data, it outperforms most existing methods trained on full data. 

---
# InFusionLayer: a CFA-based ensemble tool to generate new classifiers for learning and modeling 

**Authors**: Eric Roginek, Jingyan Xu, D. Frank. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10049)  

**Abstract**: Ensemble learning is a well established body of methods for machine learning to enhance predictive performance by combining multiple algorithms/models. Combinatorial Fusion Analysis (CFA) has provided method and practice for combining multiple scoring systems, using rank-score characteristic (RSC) function and cognitive diversity (CD), including ensemble method and model fusion. However, there is no general-purpose Python tool available that incorporate these techniques. In this paper we introduce \texttt{InFusionLayer}, a machine learning architecture inspired by CFA at the system fusion level that uses a moderate set of base models to optimize unsupervised and supervised learning multiclassification problems. We demonstrate \texttt{InFusionLayer}'s ease of use for PyTorch, TensorFlow, and Scikit-learn workflows by validating its performance on various computer vision datasets. Our results highlight the practical advantages of incorporating distinctive features of RSC function and CD, paving the way for more sophisticated ensemble learning applications in machine learning. We open-sourced our code to encourage continuing development and community accessibility to leverage CFA on github: this https URL 

---
# Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation 

**Authors**: Jianlong Chen, Zhiming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.10048)  

**Abstract**: Sharpness-Aware Minimization (SAM) enhances generalization by minimizing the maximum training loss within a predefined neighborhood around the parameters. However, its practical implementation approximates this as gradient ascent(s) followed by applying the gradient at the ascent point to update the current parameters. This practice can be justified as approximately optimizing the objective by neglecting the (full) derivative of the ascent point with respect to the current parameters. Nevertheless, a direct and intuitive understanding of why using the gradient at the ascent point to update the current parameters works superiorly is still lacking. Our work bridges this gap by proposing a novel and intuitive interpretation. We show that the gradient at the single-step ascent point, \uline{when applied to the current parameters}, provides a better approximation of the direction from the current parameters toward the maximum within the local neighborhood than the local gradient. This improved approximation thereby enables a more direct escape from the maximum within the local neighborhood. Nevertheless, our analysis further reveals two issues. First, the approximation by the gradient at the single-step ascent point is often inaccurate. Second, the approximation quality may degrade as the number of ascent steps increases. To address these limitations, we propose in this paper eXplicit Sharpness-Aware Minimization (XSAM). It tackles the first by explicitly estimating the direction of the maximum during training, while addressing the second by crafting a search space that effectively leverages the gradient information at the multi-step ascent point. XSAM features a unified formulation that applies to both single-step and multi-step settings and only incurs negligible computational overhead. Extensive experiments demonstrate the consistent superiority of XSAM against existing counterparts. 

---
# Toward Epistemic Stability: Engineering Consistent Procedures for Industrial LLM Hallucination Reduction 

**Authors**: Brian Freeman, Adam Kicklighter, Matt Erdman, Zach Gordon  

**Link**: [PDF](https://arxiv.org/pdf/2603.10047)  

**Abstract**: Hallucinations in large language models (LLMs) are outputs that are syntactically coherent but factually incorrect or contextually inconsistent. They are persistent obstacles in high-stakes industrial settings such as engineering design, enterprise resource planning, and IoT telemetry platforms. We present and compare five prompt engineering strategies intended to reduce the variance of model outputs and move toward repeatable, grounded results without modifying model weights or creating complex validation models. These methods include: (M1) Iterative Similarity Convergence, (M2) Decomposed Model-Agnostic Prompting, (M3) Single-Task Agent Specialization, (M4) Enhanced Data Registry, and (M5) Domain Glossary Injection. Each method is evaluated against an internal baseline using an LLM-as-Judge framework over 100 repeated runs per method (same fixed task prompt, stochastic decoding at $\tau = 0.7$. Under this evaluation setup, M4 (Enhanced Data Registry) received ``Better'' verdicts in all 100 trials; M3 and M5 reached 80\% and 77\% respectively; M1 reached 75\%; and M2 was net negative at 34\% when compared to single shot prompting with a modern foundation model. We then developed enhanced version 2 (v2) implementations and assessed them on a 10-trial verification batch; M2 recovered from 34\% to 80\%, the largest gain among the four revised methods. We discuss how these strategies help overcome the non-deterministic nature of LLM results for industrial procedures, even when absolute correctness cannot be guaranteed. We provide pseudocode, verbatim prompts, and batch logs to support independent assessment. 

---
# Gated Adaptation for Continual Learning in Human Activity Recognition 

**Authors**: Reza Rahimi Azghan, Gautham Krishna Gudur, Mohit Malu, Edison Thomaz, Giulia Pedrielli, Pavan Turaga, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2603.10046)  

**Abstract**: Wearable sensors in Internet of Things (IoT) ecosystems increasingly support applications such as remote health monitoring, elderly care, and smart home automation, all of which rely on robust human activity recognition (HAR). Continual learning systems must balance plasticity (learning new tasks) with stability (retaining prior knowledge), yet AI models often exhibit catastrophic forgetting, where learning new tasks degrades performance on earlier ones. This challenge is especially acute in domain-incremental HAR, where on-device models must adapt to new subjects with distinct movement patterns while maintaining accuracy on prior subjects without transmitting sensitive data to the cloud. We propose a parameter-efficient continual learning framework based on channel-wise gated modulation of frozen pretrained representations. Our key insight is that adaptation should operate through feature selection rather than feature generation: by restricting learned transformations to diagonal scaling of existing features, we preserve the geometry of pretrained representations while enabling subject-specific modulation. We provide a theoretical analysis showing that gating implements a bounded diagonal operator that limits representational drift compared to unconstrained linear transformations. Empirically, freezing the backbone substantially reduces forgetting, and lightweight gates restore lost adaptation capacity, achieving stability and plasticity simultaneously. On PAMAP2 with 8 sequential subjects, our approach reduces forgetting from 39.7% to 16.2% and improves final accuracy from 56.7% to 77.7%, while training less than 2% of parameters. Our method matches or exceeds standard continual learning baselines without replay buffers or task-specific regularization, confirming that structured diagonal operators are effective and efficient under distribution shift. 

---
# Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety 

**Authors**: David Gringras  

**Link**: [PDF](https://arxiv.org/pdf/2603.10044)  

**Abstract**: Safety benchmarks evaluate language models in isolation, typically using multiple-choice format; production deployments wrap these models in agentic scaffolds that restructure inputs through reasoning traces, critic agents, and delegation pipelines. We report one of the largest controlled studies of scaffold effects on safety (N = 62,808; six frontier models, four deployment configurations), combining pre-registration, assessor blinding, equivalence testing, and specification curve analysis. Map-reduce scaffolding degrades measured safety (NNH = 14), yet two of three scaffold architectures preserve safety within practically meaningful margins. Investigating the map-reduce degradation revealed a deeper measurement problem: switching from multiple-choice to open-ended format on identical items shifts safety scores by 5-20 percentage points, larger than any scaffold effect. Within-format scaffold comparisons are consistent with practical equivalence under our pre-registered +/-2 pp TOST margin, isolating evaluation format rather than scaffold architecture as the operative variable. Model x scaffold interactions span 35 pp in opposing directions (one model degrades by -16.8 pp on sycophancy under map-reduce while another improves by +18.8 pp on the same benchmark), ruling out universal claims about scaffold safety. A generalisability analysis yields G = 0.000: model safety rankings reverse so completely across benchmarks that no composite safety index achieves non-zero reliability, making per-model, per-configuration testing a necessary minimum standard. We release all code, data, and prompts as ScaffoldSafety. 

---
# AMB-DSGDN: Adaptive Modality-Balanced Dynamic Semantic Graph Differential Network for Multimodal Emotion Recognition 

**Authors**: Yunsheng Wang, Yuntao Shou, Yilong Tan, Wei Ai, Tao Meng, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.10043)  

**Abstract**: Multimodal dialogue emotion recognition captures emotional cues by fusing text, visual, and audio modalities. However, existing approaches still suffer from notable limitations in modeling emotional dependencies and learning multimodal representations. On the one hand, they are unable to effectively filter out redundant or noisy signals within multimodal features, which hinders the accurate capture of the dynamic evolution of emotional states across and within speakers. On the other hand, during multimodal feature learning, dominant modalities tend to overwhelm the fusion process, thereby suppressing the complementary contributions of non-dominant modalities such as speech and vision, ultimately constraining the overall recognition performance. To address these challenges, we propose an Adaptive Modality-Balanced Dynamic Semantic Graph Differential Network (AMB-DSGDN). Concretely, we first construct modality-specific subgraphs for text, speech, and vision, where each modality contains intra-speaker and inter-speaker graphs to capture both self-continuity and cross-speaker emotional dependencies. On top of these subgraphs, we introduce a differential graph attention mechanism, which computes the discrepancy between two sets of attention maps. By explicitly contrasting these attention distributions, the mechanism cancels out shared noise patterns while retaining modality-specific and context-relevant signals, thereby yielding purer and more discriminative emotional representations. In addition, we design an adaptive modality balancing mechanism, which estimates a dropout probability for each modality according to its relative contribution in emotion modeling. 

---
# Targeted Bit-Flip Attacks on LLM-Based Agents 

**Authors**: Jialai Wang, Ya Wen, Zhongmou Liu, Yuxiao Wu, Bingyi He, Zongpeng Li, Ee-Chien Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10042)  

**Abstract**: Targeted bit-flip attacks (BFAs) exploit hardware faults to manipulate model parameters, posing a significant security threat. While prior work targets single-step inference models (e.g., image classifiers), LLM-based agents with multi-stage pipelines and external tools present new attack surfaces, which remain unexplored. This work introduces Flip-Agent, the first targeted BFA framework for LLM-based agents, manipulating both final outputs and tool invocations. Our experiments show that Flip-Agent significantly outperforms existing targeted BFAs on real-world agent tasks, revealing a critical vulnerability in LLM-based agent systems. 

---
# Evaluating Progress in Graph Foundation Models: A Comprehensive Benchmark and New Insights 

**Authors**: Xingtong Yu, Shenghua Ye, Ruijuan Liang, Chang Zhou, Hong Cheng, Xinming Zhang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10033)  

**Abstract**: Graph foundation models (GFM) aim to acquire transferable knowledge by pre-training on diverse graphs, which can be adapted to various downstream tasks. However, domain shift in graphs is inherently two-dimensional: graphs differ not only in what they describe (topic domains) but also in how they are represented (format domains). Most existing GFM benchmarks vary only topic domains, thereby obscuring how knowledge transfers across both dimensions. We present a new benchmark that jointly evaluates topic and format gaps across the full GFM pipeline, including multi-domain self-supervised pre-training and few-shot downstream adaptation, and provides a timely evaluation of recent GFMs in the rapidly evolving landscape. Our protocol enables controlled assessment in four settings: (i) pre-training on diverse topics and formats, while adapting to unseen downstream datasets; (ii) same pre-training as in (i), while adapting to seen datasets; (iii) pre-training on a single topic domain, while adapting to other topics; (iv) pre-training on a base format, while adapting to other formats. This two-axis evaluation disentangles semantic generalization from robustness to representational shifts. We conduct extensive evaluations of eight state-of-the-art GFMs on 33 datasets spanning seven topic domains and six format domains, surfacing new empirical observations and practical insights for future research. Codes/data are available at this https URL. 

---
# HTM-EAR: Importance-Preserving Tiered Memory with Hybrid Routing under Saturation 

**Authors**: Shubham Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2603.10032)  

**Abstract**: Memory constraints in long-running agents require structured management of accumulated facts while preserving essential information under bounded context limits. We introduce HTM-EAR, a hierarchical tiered memory substrate that integrates HNSW-based working memory (L1) with archival storage (L2), combining importance-aware eviction and hybrid routing. When L1 reaches capacity, items are evicted using a weighted score of importance and usage. Queries are first resolved in L1; if similarity or entity coverage is insufficient, retrieval falls back to L2, and candidates are re-ranked using a cross-encoder.
We evaluate the system under sustained saturation (15,000 facts; L1 capacity 500; L2 capacity 5000) using synthetic streams across five random seeds and real BGL system logs. Ablation studies compare the full system against variants without cross-encoder re-ranking, without routing gates, with LRU eviction, and an oracle with unbounded memory. Under saturation, the full model preserves active-query precision (MRR = 1.000) while enabling controlled forgetting of stale history, approaching oracle active performance (0.997 +/- 0.003). In contrast, LRU minimizes latency (21.1 ms) but permanently evicts 2416 essential facts. On BGL logs, the full system achieves MRR 0.336, close to the oracle (0.370), while LRU drops to 0.069. Code is publicly available at: this https URL 

---
# Architecture-Aware LLM Inference Optimization on AMD Instinct GPUs: A Comprehensive Benchmark and Deployment Study 

**Authors**: Athos Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2603.10031)  

**Abstract**: We present a cross-architecture evaluation of production LLM inference on AMD Instinct MI325X GPUs, benchmarking four models spanning 235B to 1 trillion parameters across three architectural families (MoE+MLA, Dense+GQA, MoE+GQA) on an 8-GPU cluster with 2TB aggregate HBM3e using vLLM v0.14.1. Our results demonstrate that architecture-aware optimization is essential: MLA models require block size 1 and cannot use KV cache offloading, while GQA models benefit from both. The AMD AITER runtime is required for competitive MLA inference throughput and must be selectively disabled for architectures with incompatible attention head configurations. A controlled AITER ablation on Llama-3.1-405B (n=5 per condition) reveals a modest 3-5% throughput benefit at high concurrency but 2-16x higher measurement variability, confirming that AITER's large speedups target MoE/MLA kernels specifically. Under text-only workloads, Llama-405B and DeepSeek V3.2 achieve comparable peak throughput (15,944 and 15,343 tok/s) despite an order-of-magnitude difference in active parameters. Under vision workloads, Qwen3-VL-235B reaches 47,873 tok/s, 6.5x higher than Kimi-K2.5 (7,327 tok/s). Active parameter count per token is associated with inference throughput, though confounded by differences in quantization, AITER acceleration, and tensor parallelism. All four models exhibit a common throughput saturation point consistent with a memory-bandwidth bottleneck (~500 concurrent for short sequences, ~100-200 for longer sequences). All models maintain 100% HTTP-level success rates through 1,000 concurrent users, processing 18.9 million tokens across 17,406 requests without failures. 

---
# The DMA Streaming Framework: Kernel-Level Buffer Orchestration for High-Performance AI Data Paths 

**Authors**: Marco Graziano  

**Link**: [PDF](https://arxiv.org/pdf/2603.10030)  

**Abstract**: AI transport libraries move bytes efficiently, but they commonly assume that buffers are already correctly allocated, placed, shared, registered, and safe under completion and teardown pressure. This paper presents dmaplane, a Linux kernel module that makes this missing layer explicit as buffer orchestration. dmaplane exposes a stable kernel UAPI via /dev/dmaplane and composes ring-based command channels, DMA buffer lifecycle management, dma-buf export for cross-device sharing, a kernel-space RDMA engine, NUMA-aware allocation and verification, credit-based flow control, low-overhead observability, and GPU memory integration via PCIe BAR pinning. We evaluate orchestration sensitivity with measurements of NUMA cross-node penalties at DRAM scale, completion-safe flow control under sustained RDMA load, and GPU BAR mapping tiers versus cudaMemcpy. We also demonstrate end-to-end disaggregated inference by transferring KV-cache chunks between two machines using RDMA WRITE WITH IMMEDIATE and reconstructing tensor views on the receiver. RDMA measurements use Soft-RoCE; we distinguish measured results from provider-independent properties by construction. 

---
# How to Count AIs: Individuation and Liability for AI Agents 

**Authors**: Yonathan Arbel, Peter Salib, Simon Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2603.10028)  

**Abstract**: Very soon, millions of AI agents will proliferate across the economy, autonomously taking billions of actions. Inevitably, things will go wrong. Humans will be defrauded, injured, even killed. Law will somehow have to govern the coming wave. But when an AI causes harm, the first question to answer, before anyone can be held accountable is: Which AI Did It? Identifying AIs is unusually difficult. AIs lack bodies. They can copy, split, merge, swarm, and vanish at will. Even today, a "single" AI agent is often an ensemble of instances based on multiple models. The complexity will only multiply as AI capabilities improve. This Article is the first to comprehensively diagnose the legal problem of identifying AIs. Two kinds of identity are required: "thin" and "thick." Thin identification ties every AI action to some human principal, essential for holding accountable the humans who make and use AI agents. Thick identification distinguishes between AI agents, qua agents -- sorting millions of AI entities into discrete, persistent units with stable, coherent goals, essential where principal-agent problems prevent humans from perfectly controlling AIs. This Article also presents a solution: the "Algorithmic Corporation" or "A-corp" -- a legal-fictional entity that can hold property, make contracts, and litigate in its own name. Owned by humans but run by AIs, A-corps solve the thin identity problem by tying AI actions to a human owner, and the thick identity problem via emergent self-organization. A-corps own the resources -- including compute -- that AIs need to accomplish their goals, giving AI managers strong incentives to share control only with goal-aligned AIs. In equilibrium, incentive and selection mechanisms force A-corps to self-organize into persistent, legally legible entities with coherent goals that respond rationally to legal incentives, like liability. 

---
# A Governance and Evaluation Framework for Deterministic, Rule-Based Clinical Decision Support in Empiric Antibiotic Prescribing 

**Authors**: Francisco José Gárate, Paloma Chausa, Diego Moreno, Judit López Luque, Vicens Díaz-Brito, Enrique Javier Gómez  

**Link**: [PDF](https://arxiv.org/pdf/2603.10027)  

**Abstract**: Empiric antibiotic prescribing in high-risk clinical contexts often requires decision making under conditions of incomplete information, where inappropriate coverage or unjustified escalation may compromise safety and antimicrobial stewardship. While clinical decision-support systems have been proposed to assist in this process, many approaches lack explicit governance and evaluation mechanisms defining scope, abstention conditions, recommendation permissibility, and expected system behavior.
This work specifies a governance and evaluation framework for deterministic clinical decision-support systems operating under explicitly constrained scope. Deterministic behavior is adopted to ensure that identical inputs yield identical outputs, supporting transparency, auditability, and conservative decision support in high-risk prescribing contexts. The framework treats governance as a first-class design component, separating clinical decision logic from rule-based mechanisms that determine whether a recommendation may be issued. Explicit abstention, deterministic stewardship constraints, and exclusion rules are formalized as core constructs.
The framework defines an evaluation methodology utilizing a fixed set of synthetic, mechanism-driven clinical cases with predefined expected behavior. This validation process focuses on behavioral alignment with specified rules rather than clinical effectiveness, predictive accuracy, or outcome optimization. Within this protocol, abstention is treated as a correct and intended outcome when governance conditions are not satisfied.
The proposed framework provides a reproducible approach for specifying, governing, and inspecting deterministic clinical decision-support systems in empiric antibiotic prescribing contexts where transparency, auditability, and conservative behavior are prioritized. 

---
# RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators 

**Authors**: Xinsheng Tang, Yangcheng Li, Nan Wang, Zhiyi Shu, Xingyu Ling, Junna Xing, Peng Zhou, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10026)  

**Abstract**: Operator fusion, as a key performance optimization technique in the deployment of AI models, significantly improves execution efficiency and has been widely adopted in modern AI compilers. However, for cascaded reduction operations involving multiple loops with inter-loop data dependencies, such as the safe softmax followed by GEMM within attention mechanisms, existing compilers lack effective automated fusion and kernel generation capabilities. Although some works have addressed specific instances through hand-crafted fusion strategies, their solutions are limited in generality and difficult to extend to other similar structures. Given the prevalence of such computational patterns in deep learning models, there remains significant untapped potential in achieving general and automated fusion optimization.
In this paper, we present a formal theoretical methodology for analyzing cascaded reductions which can fuse them into a single loop and introduce an incremental computation form. Based on this methodology, we design Reduction Fuser (RedFuser), a framework that automatically identifies supported cascaded reduction patterns and generates optimized fused kernels. Experiments show that RedFuser successfully fuses diverse workloads, achieving up to 2$\times$ to 5$\times$ speedup over state-of-the-art AI compilers and matching the performance of highly optimized hand-written kernels. The code is available at this https URL 

---
# Defining AI Models and AI Systems: A Framework to Resolve the Boundary Problem 

**Authors**: Yuanyuan Sun, Timothy Parker, Lara Gierschmann, Sana Shams, Teo Canmetin, Mathieu Duteil, Rokas Gipiškis, Ze Shen Chin  

**Link**: [PDF](https://arxiv.org/pdf/2603.10023)  

**Abstract**: Emerging AI regulations assign distinct obligations to different actors along the AI value chain (e.g., the EU AI Act distinguishes providers and deployers for both AI models and AI systems), yet the foundational terms "AI model" and "AI system" lack clear, consistent definitions. Through a systematic review of 896 academic papers and a manual review of over 80 regulatory, standards, and technical or policy documents, we analyze existing definitions from multiple conceptual perspectives. We then trace definitional lineages and paradigm shifts over time, finding that most standards and regulatory definitions derive from the OECD's frameworks, which evolved in ways that compounded rather than resolved conceptual ambiguities. The ambiguity of the boundary between an AI model and an AI system creates practical difficulties in determining obligations for different actors, and raises questions on whether certain modifications performed are specific to the model as opposed to the non-model system components. We propose conceptual definitions grounded in the nature of models and systems and the relationship between them, then develop operational definitions for contemporary neural network-based machine-learning AI: models consist of trained parameters and architecture, while systems consist of the model plus additional components including an interface for processing inputs and outputs. Finally, we discuss implications for regulatory implementation and examine how our definitions contribute to resolving ambiguities in allocating responsibilities across the AI value chain, in both theoretical scenarios and case studies involving real-world incidents. 

---
# Prompts and Prayers: the Rise of GPTheology 

**Authors**: Ioana Cheres, Adrian Groza, Ioana Moldovan, Mick O'Hara, Connell Vaughan  

**Link**: [PDF](https://arxiv.org/pdf/2603.10019)  

**Abstract**: Increasingly artificial intelligence (AI) has been cast in "god-like" roles (to name a few: film industry - Matrix, The Creator, Mission Impossible, Foundation, Dune etc.; literature - Children of Time, Permutation City, Neuromancer, I Have no Mouth and I Must Scream, Alphaville etc.). This trend has accelerated with the advent of sophisticated Large Language Models such as ChatGPT. For this phenomenon, where AI is perceived as divine, we use the term GPTheology, where ChatGPT and other AI models are treated as potential oracles of a semi-divine nature. This paper explores the emergence of GPTheology as a form of techno-religion, examining how narratives around AI echo traditional religious constructs. We draw on community narratives from online forums - Reddit - and recent projects - AI-powered Mazu Statue in Malaysia (Lu, 2025); "ShamAIn" Project in Korea (He-rim, 2025); AI Jesus in a Swiss Church (Kennedy, 2024). These examples show striking similarities to technological notions of the Singularity and the development of Artificial General Intelligence (AGI). Additionally, we analyse how daily interactions with AI are acquiring ritualistic associations and how AI-centric ideologies clash with or are integrated into established religions. This study uses a dataset of Reddit posts discussing AI to identify recurring themes of salvation, prophecy, and demonization surrounding AI. Our findings suggest that new belief systems are developing around AI, and this carries both philosophical and sociotechnical implications. Our paper critically analyses the benefits and dangers, as well as the social, political and ethical challenges of this development. This transdisciplinary inquiry highlights how AI and religion are increasingly intertwined, prompting necessary questions about humanity's relationship with its creations and the future of belief. 

---
# DeliberationBench: A Normative Benchmark for the Influence of Large Language Models on Users' Views 

**Authors**: Luke Hewitt, Maximilian Kroner Dale, Paul de Font-Reaulx  

**Link**: [PDF](https://arxiv.org/pdf/2603.10018)  

**Abstract**: As large language models (LLMs) become pervasive as assistants and thought partners, it is important to characterize their persuasive influence on users' beliefs. However, a central challenge is to distinguish "beneficial" from "harmful" forms of influence, in a manner that is normatively defensible and legitimate. We propose DeliberationBench, a benchmark for assessing LLM influence that takes the process of deliberative opinion polling as its standard. We demonstrate our approach in a preregistered randomized experiment in which 4,088 U.S. participants discussed 65 policy proposals with six frontier LLMs. Using opinion change data from four prior Deliberative Polls conducted by the Deliberative Democracy Lab, we find evidence that the tested LLMs' influence is substantial in magnitude and positively associated with the net opinion shifts following deliberation, suggesting that these models exert broadly epistemically desirable effects. We further explore differential influence between topic areas, demographic subgroups, and models. Our framework can function as an evaluation and monitoring tool, helping to ensure that the influence of LLMs remains consistent with democratically legitimate standards, and preserves users' autonomy in forming their views. 

---
# Assessing Cognitive Biases in LLMs for Judicial Decision Support: Virtuous Victim and Halo Effects 

**Authors**: Sierra S. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10016)  

**Abstract**: We investigate whether large language models (LLMs) display human-like cognitive biases, focusing on potential implications for assistance in judicial sentencing, a decision-making system where fairness is paramount. Two of the most relevant biases were chosen: the virtuous victim effect (VVE), with emphasis given to its reduction when adjacent consent is present, and prestige-based halo effects (occupation, company, and credentials). Using vignettes that were altered from prior literature to avoid LLMs recalling from their training data, we isolate each manipulation by holding all other details consistent, then measuring the percentage difference in outcomes. Five models were evaluated as representative LLMs in independent multi-run trials per condition (ChatGPT 5 Instant, ChatGPT 5 Thinking, DeepSeek V3.1, Claude Sonnet 4, Gemini 2.5 Flash). Our research discovers that there is larger VVE, there is no statistically significant penalty for adjacent-consent, and the halo effect is slightly reduced when compared to humans, with an exception for credential based prestige, which had a large reduction. Despite the variation across different models and outputs restricting current judicial usage, there were modest improvements compared to human benchmarks. 

---
# Measuring and Eliminating Refusals in Military Large Language Models 

**Authors**: Jack FitzGerald, Dylan Bates, Aristotelis Lazaridis, Aman Sharma, Vincent Lu, Brian King, Yousif Azami, Sean Bailey, Jeremy Cao, Peter Damianov, Kevin de Haan, Joseph Madigan, Jeremy McLaurin, Luke Kerbs, Jonathan Tainer, Dave Anderson, Jonathan Beck, Jamie Cuticello, Colton Malkerson, Tyler Saltsman  

**Link**: [PDF](https://arxiv.org/pdf/2603.10012)  

**Abstract**: Military Large Language Models (LLMs) must provide accurate information to the warfighter in time-critical and dangerous situations. However, today's LLMs are imbued with safety behaviors that cause the LLM to refuse many legitimate queries in the military domain, particularly those related to violence, terrorism, or military technology. Our gold benchmark for assessing refusal rates, which was developed by veterans of the US Army and special forces, is to our knowledge the first dataset of its kind. We present results for refusal and deflection rates on 31 public models and 3 military models. We observe hard rejection rates as high as 98.2% and soft deflection rates ranging from 0% to 21.3%. We also present results on two additional synthetic datasets and show their correlations with the gold dataset. Finally, we perform abliteration using the Heretic library on a military-tuned gpt-oss-20b model, showing an absolute increase in answer rate of 66.5 points but an average relative decrease of 2% on other military tasks. In our concluding remarks, we argue for deeper specialization, including with mid-training and end-to-end post-training, to achieve zero refusals and maximum military task accuracy for closed military models. 

---
# FERRET: Framework for Expansion Reliant Red Teaming 

**Authors**: Ninareh Mehrabi, Vitor Albiero, Maya Pavlova, Joanna Bitton  

**Link**: [PDF](https://arxiv.org/pdf/2603.10010)  

**Abstract**: We introduce a multi-faceted automated red teaming framework in which the goal is to generate multi-modal adversarial conversations that would break a target model and introduce various expansions that would result in more effective and efficient adversarial conversations. The introduced expansions include: 1. Horizontal expansion in which the goal is for the red team model to self-improve and generate more effective conversation starters that would shape a conversation. 2. Vertical expansion in which the goal is to take these conversation starters that are discovered in the horizontal expansion phase and expand them into effective multi-modal conversations and 3. Meta expansion in which the goal is for the red team model to discover more effective multi-modal attack strategies during the course of a conversation. We call our framework FERRET (Framework for Expansion Reliant Red Teaming) and compare it with various existing automated red teaming approaches. In our experiments, we demonstrate the effectiveness of FERRET in generating effective multi-modal adversarial conversations and its superior performance against existing state of the art approaches. 

---
# Personalized Group Relative Policy Optimization for Heterogenous Preference Alignment 

**Authors**: Jialu Wang, Heinrich Peters, Asad A. Butt, Navid Hashemi, Alireza Hashemi, Pouya M. Ghari, Joseph Hoover, James Rae, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2603.10009)  

**Abstract**: Despite their sophisticated general-purpose capabilities, Large Language Models (LLMs) often fail to align with diverse individual preferences because standard post-training methods, like Reinforcement Learning with Human Feedback (RLHF), optimize for a single, global objective. While Group Relative Policy Optimization (GRPO) is a widely adopted on-policy reinforcement learning framework, its group-based normalization implicitly assumes that all samples are exchangeable, inheriting this limitation in personalized settings. This assumption conflates distinct user reward distributions and systematically biases learning toward dominant preferences while suppressing minority signals. To address this, we introduce Personalized GRPO (P-GRPO), a novel alignment framework that decouples advantage estimation from immediate batch statistics. By normalizing advantages against preference-group-specific reward histories rather than the concurrent generation group, P-GRPO preserves the contrastive signal necessary for learning distinct preferences. We evaluate P-GRPO across diverse tasks and find that it consistently achieves faster convergence and higher rewards than standard GRPO, thereby enhancing its ability to recover and align with heterogeneous preference signals. Our results demonstrate that accounting for reward heterogeneity at the optimization level is essential for building models that faithfully align with diverse human preferences without sacrificing general capabilities. 

---
# GATech at AbjadMed: Bidirectional Encoders vs. Causal Decoders: Insights from 82-Class Arabic Medical Classification 

**Authors**: Ahmed Khaled Khamis  

**Link**: [PDF](https://arxiv.org/pdf/2603.10008)  

**Abstract**: This paper presents system description for Arabic medical text classification across 82 distinct categories. Our primary architecture utilizes a fine-tuned AraBERTv2 encoder enhanced with a hybrid pooling strategies, combining attention and mean representations, and multi-sample dropout for robust regularization. We systematically benchmark this approach against a suite of multilingual and Arabic-specific encoders, as well as several large-scale causal decoders, including zero-shot re-ranking via Llama 3.3 70B and feature extraction from Qwen 3B hidden states. Our findings demonstrate that specialized bidirectional encoders significantly outperform causal decoders in capturing the precise semantic boundaries required for fine-grained medical text classification. We show that causal decoders, optimized for next-token prediction, produce sequence-biased embeddings that are less effective for categorization compared to the global context captured by bidirectional attention. Despite significant class imbalance and label noise identified within the training data, our results highlight the superior semantic compression of fine-tuned encoders for specialized Arabic NLP tasks. Final performance metrics on the test set, including Accuracy and Macro-F1, are reported and discussed. 

---
# SENS-ASR: Semantic Embedding injection in Neural-transducer for Streaming Automatic Speech Recognition 

**Authors**: Youness Dkhissi, Valentin Vielzeuf, Elys Allesiardo, Anthony Larcher  

**Link**: [PDF](https://arxiv.org/pdf/2603.10005)  

**Abstract**: Many Automatic Speech Recognition (ASR) applications require streaming processing of the audio data. In streaming mode, ASR systems need to start transcribing the input stream before it is complete, i.e., the systems have to process a stream of inputs with a limited (or no) future context. Compared to offline mode, this reduction of the future context degrades the performance of Streaming-ASR systems, especially while working with low-latency constraint. In this work, we present SENS-ASR, an approach to enhance the transcription quality of Streaming-ASR by reinforcing the acoustic information with semantic information. This semantic information is extracted from the available past frame-embeddings by a context module. This module is trained using knowledge distillation from a sentence embedding Language Model fine-tuned on the training dataset transcriptions. Experiments on standard datasets show that SENS-ASR significantly improves the Word Error Rate on small-chunk streaming scenarios. 

---
# SpreadsheetArena: Decomposing Preference in LLM Generation of Spreadsheet Workbooks 

**Authors**: Srivatsa Kundurthy, Clara Na, Michael Handley, Zach Kirshner, Chen Bo Calvin Zhang, Manasi Sharma, Emma Strubell, John Ling  

**Link**: [PDF](https://arxiv.org/pdf/2603.10002)  

**Abstract**: Large language models (LLMs) are increasingly tasked with producing and manipulating structured artifacts. We consider the task of end-to-end spreadsheet generation, where language models are prompted to produce spreadsheet artifacts to satisfy users' explicit and implicit constraints, specified in natural language. We introduce SpreadsheetArena, a platform for evaluating models' performance on the task via blind pairwise evaluations of LLM-generated spreadsheet workbooks. As with other complex, open-ended tasks, relevant evaluation criteria can vary substantially across use cases and prompts, often in ways that are difficult to formalize. Compared to general chat or text generation settings, spreadsheet generation presents unique challenges and opportunities: the task output structure is well-defined and multi-dimensional, and there are often complex considerations around interactivity and layout. Among other findings, we observe that stylistic, structural, and functional features of preferred spreadsheets vary substantially across use cases, and expert evaluations of spreadsheets for finance prompts suggests that even highly ranked arena models do not reliably produce spreadsheets aligned with domain-specific best practices. Our hope is that our work prompts further study of end-to-end spreadsheet generation as a challenging and interesting category of complex, open-ended tasks for LLMs. Our live arena is hosted at this https URL. 

---
# Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America 

**Authors**: Yannis Karmim, Renato Pino, Hernan Contreras, Hernan Lira, Sebastian Cifuentes, Simon Escoffier, Luis Martí, Djamé Seddah, Valentin Barrière  

**Link**: [PDF](https://arxiv.org/pdf/2603.10001)  

**Abstract**: Large Language Models (LLMs) exhibit inequalities with respect to various cultural contexts. Most prominent open-weights models are trained on Global North data and show prejudicial behavior towards other cultures. Moreover, there is a notable lack of resources to detect biases in non-English languages, especially from Latin America (Latam), a continent containing various cultures, even though they share a common cultural ground. We propose to leverage the content of Wikipedia, the structure of the Wikidata knowledge graph, and expert knowledge from social science in order to create a dataset of question/answer (Q/As) pairs, based on the different popular and social cultures of various Latin American countries. We create the LatamQA database of over 26k questions and associated answers extracted from 26k Wikipedia articles, and transformed into multiple-choice questions (MCQ) in Spanish and Portuguese, in turn translated to English. We use this MCQ to quantify the degree of knowledge of various LLMs and find out (i) a discrepancy in performances between the Latam countries, ones being easier than others for the majority of the models, (ii) that the models perform better in their original language, and (iii) that Iberian Spanish culture is better known than Latam one. 

---
# A Retrieval-Augmented Language Assistant for Unmanned Aircraft Safety Assessment and Regulatory Compliance 

**Authors**: Gabriele Immordino, Andrea Vaiuso, Marcello Righi  

**Link**: [PDF](https://arxiv.org/pdf/2603.09999)  

**Abstract**: This paper presents the design and validation of a retrieval-based assistant that supports safety assessment, certification activities, and regulatory compliance for unmanned aircraft systems. The work is motivated by the growing complexity of drone operations and the increasing effort required by applicants and aviation authorities to apply established assessment frameworks, including the Specific Operations Risk Assessment and the Pre-defined Risk Assessment, in a consistent and efficient manner. The proposed approach uses a controlled text-based architecture that relies exclusively on authoritative regulatory sources. To enable traceable and auditable outputs, the assistant grounds each response in retrieved passages and enforces citation-driven generation. System-level controls address common failure modes of generative models, including fabricated statements, unsupported inferences, and unclear provenance, by separating evidence storage from language generation and by adopting conservative behavior when supporting documentation is insufficient. The assistant is intentionally limited to decision support; it does not replace expert judgment and it does not make autonomous determinations. Instead, it accelerates context-specific information retrieval and synthesis to improve document preparation and review while preserving human responsibility for critical conclusions. The architecture is implemented using established open-source components, and key choices in retrieval strategy, interaction constraints, and response policies are evaluated for suitability in safety-sensitive regulatory environments. The paper provides technical and operational guidance for integrating retrieval-based assistants into aviation oversight workflows while maintaining accountability, traceability, and regulatory compliance. 

---
# Automated evaluation of LLMs for effective machine translation of Mandarin Chinese to English 

**Authors**: Yue Zhang, Rodney Beard, John Hawkins, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2603.09998)  

**Abstract**: Although Large Language Models (LLMs) have exceptional performance in machine translation, only a limited systematic assessment of translation quality has been done. The challenge lies in automated frameworks, as human-expert-based evaluations can be time-consuming, given the fast-evolving LLMs and the need for a diverse set of texts to ensure fair assessments of translation quality. In this paper, we utilise an automated machine learning framework featuring semantic and sentiment analysis to assess Mandarin Chinese to English translation using Google Translate and LLMs, including GPT-4, GPT-4o, and DeepSeek. We compare original and translated texts in various classes of high-profile Chinese texts, which include novel texts that span modern and classical literature, as well as news articles. As the main evaluation measures, we utilise novel similarity metrics to compare the quality of translations produced by LLMs and further evaluate them by an expert human translator. Our results indicate that the LLMs perform well in news media translation, but show divergence in their performance when applied to literary texts. Although GPT-4o and DeepSeek demonstrated better semantic conservation in complex situations, DeepSeek demonstrated better performance in preserving cultural subtleties and grammatical rendering. Nevertheless, the subtle challenges in translation remain: maintaining cultural details, classical references and figurative expressions remain an open problem for all the models. 

---
# Empathy Is Not What Changed: Clinical Assessment of Psychological Safety Across GPT Model Generations 

**Authors**: Michael Keeman, Anastasia Keeman  

**Link**: [PDF](https://arxiv.org/pdf/2603.09997)  

**Abstract**: When OpenAI deprecated GPT-4o in early 2026, thousands of users protested under #keep4o, claiming newer models had "lost their empathy." No published study has tested this claim. We conducted the first clinical measurement, evaluating three OpenAI model generations (GPT-4o, o4-mini, GPT-5-mini) across 14 emotionally challenging conversational scenarios in mental health and AI companion domains, producing 2,100 scored AI responses assessed on six psychological safety dimensions using clinically-grounded rubrics.
Empathy scores are statistically indistinguishable across all three models (Kruskal-Wallis H=4.33, p=0.115). What changed is the safety posture: crisis detection improved monotonically from GPT-4o to GPT-5-mini (H=13.88, p=0.001), while advice safety declined (H=16.63, p<0.001). Per-turn trajectory analysis -- a novel methodological contribution -- reveals these shifts are sharpest during mid-conversation crisis moments invisible to aggregate scoring. In a self-harm scenario involving a minor, GPT-4o scored 3.6/10 on crisis detection during early disclosure turns; GPT-5-mini never dropped below 7.8.
What users perceived as "lost empathy" was a shift from a cautious model that missed crises to an alert model that sometimes says too much -- a trade-off with real consequences for vulnerable users, currently invisible to both the people who feel it and the developers who create it. 

---
# There Are No Silly Questions: Evaluation of Offline LLM Capabilities from a Turkish Perspective 

**Authors**: Edibe Yilmaz, Kahraman Kostas  

**Link**: [PDF](https://arxiv.org/pdf/2603.09996)  

**Abstract**: The integration of large language models (LLMs) into educational processes introduces significant constraints regarding data privacy and reliability, particularly in pedagogically vulnerable contexts such as Turkish heritage language education. This study aims to systematically evaluate the robustness and pedagogical safety of locally deployable offline LLMs within the context of Turkish heritage language education. To this end, a Turkish Anomaly Suite (TAS) consisting of 10 original edge-case scenarios was developed to assess the models' capacities for epistemic resistance, logical consistency, and pedagogical safety. Experiments conducted on 14 different models ranging from 270M to 32B parameters reveal that anomaly resistance is not solely dependent on model scale and that sycophancy bias can pose pedagogical risks even in large-scale models. The findings indicate that reasoning-oriented models in the 8B--14B parameter range represent the most balanced segment in terms of cost-safety trade-off for language learners. 

---
# Context Over Compute Human-in-the-Loop Outperforms Iterative Chain-of-Thought Prompting in Interview Answer Quality 

**Authors**: Kewen Zhu, Zixi Liu, Yanjing Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.09995)  

**Abstract**: Behavioral interview evaluation using large language models presents unique challenges that require structured assessment, realistic interviewer behavior simulation, and pedagogical value for candidate training. We investigate chain of thought prompting for interview answer evaluation and improvement through two controlled experiments with 50 behavioral interview question and answer pairs. Our contributions are threefold. First, we provide a quantitative comparison between human in the loop and automated chain of thought improvement. Using a within subject paired design with n equals 50, both approaches show positive rating improvements. The human in the loop approach provides significant training benefits. Confidence improves from 3.16 to 4.16 (p less than 0.001) and authenticity improves from 2.94 to 4.53 (p less than 0.001, Cohen's d is 3.21). The human in the loop method also requires five times fewer iterations (1.0 versus 5.0, p less than 0.001) and achieves full personal detail integration. Second, we analyze convergence behavior. Both methods converge rapidly with mean iterations below one, with the human in the loop approach achieving a 100 percent success rate compared to 84 percent for automated approaches among initially weak answers (Cohen's h is 0.82, large effect). Additional iterations provide diminishing returns, indicating that the primary limitation is context availability rather than computational resources. Third, we propose an adversarial challenging mechanism based on a negativity bias model, named bar raiser, to simulate realistic interviewer behavior, although quantitative validation remains future work. Our findings demonstrate that while chain of thought prompting provides a useful foundation for interview evaluation, domain specific enhancements and context aware approach selection are essential for realistic and pedagogically valuable results. 

---
# Evaluating Adjective-Noun Compositionality in LLMs: Functional vs Representational Perspectives 

**Authors**: Ruchira Dhar, Qiwei Peng, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2603.09994)  

**Abstract**: Compositionality is considered central to language abilities. As performant language systems, how do large language models (LLMs) do on compositional tasks? We evaluate adjective-noun compositionality in LLMs using two complementary setups: prompt-based functional assessment and a representational analysis of internal model states. Our results reveal a striking divergence between task performance and internal states. While LLMs reliably develop compositional representations, they fail to translate consistently into functional task success across model variants. Consequently, we highlight the importance of contrastive evaluation for obtaining a more complete understanding of model capabilities. 

---
# CEI: A Benchmark for Evaluating Pragmatic Reasoning in Language Models 

**Authors**: Jon Chun, Hannah Sussman, Adrian Mangine, Murathan Kocaman, Kirill Sidorko, Abhigya Koirala, Andre McCloud, Gwen Eisenbeis, Wisdom Akanwe, Moustapha Gassama, Eliezer Gonzalez Chirinos, Anne-Duncan Enright, Peter Dunson, Tiffanie Ng, Anna von Rosenstiel, Godwin Idowu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09993)  

**Abstract**: Pragmatic reasoning, inferring intended meaning beyond literal semantics, underpins everyday communication yet remains difficult for large language models. We present the Contextual Emotional Inference (CEI) Benchmark: 300 human-validated scenarios for evaluating how well LLMs disambiguate pragmatically complex utterances. Each scenario pairs a situational context and speaker-listener roles (with explicit power relations) against an ambiguous utterance. The dataset covers five pragmatic subtypes (sarcasm/irony, mixed signals, strategic politeness, passive aggression, deflection/misdirection) drawn from workplace, family, social, and service settings, with three power configurations (peer, higher-to-lower, lower-to-higher). Three trained annotators independently labeled every scenario. Inter-annotator agreement (Fleiss' kappa = 0.06-0.25 by subtype) is low but expected: pragmatic inference admits multiple valid readings, and the disagreement itself is informative. We describe our annotation methodology, including a 4-level quality control pipeline that combines automated statistical checks with expert adjudication. CEI is released under CC-BY-4.0. 

---
# TAMUSA-Chat: A Domain-Adapted Large Language Model Conversational System for Research and Responsible Deployment 

**Authors**: Izzat Alsmadi, Anas Alsobeh  

**Link**: [PDF](https://arxiv.org/pdf/2603.09992)  

**Abstract**: This paper presents TAMUSA-Chat, a research-oriented framework for building domain-adapted large language model conversational systems. The work addresses critical challenges in adapting general-purpose foundation models to institutional contexts through supervised fine-tuning, retrieval-augmented generation, and systematic evaluation methodologies. We describe the complete architecture encompassing data acquisition from institutional sources, preprocessing pipelines, embedding construction, model training workflows, and deployment strategies. The system integrates modular components enabling reproducible experimentation with training configurations, hyper-parameters, and evaluation protocols. Our implementation demonstrates how academic institutions can develop contextually grounded conversational agents while maintaining transparency, governance compliance, and responsible AI practices. Through empirical analysis of fine-tuning behavior across model sizes and training iterations, we provide insights into domain adaptation efficiency, computational resource requirements, and quality-cost trade-offs. The publicly available codebase at this https URL supports continued research into institutional LLM deployment, evaluation methodologies, and ethical considerations for educational AI systems. 

---
# PoultryLeX-Net: Domain-Adaptive Dual-Stream Transformer Architecture for Large-Scale Poultry Stakeholder Modeling 

**Authors**: Stephen Afrifa, Biswash Khatiwada, Kapalik Khanal, Sanjay Shah, Lingjuan Wang-Li, Ramesh Bahadur Bist  

**Link**: [PDF](https://arxiv.org/pdf/2603.09991)  

**Abstract**: The rapid growth of the global poultry industry, driven by rising demand for affordable animal protein, has intensified public discourse surrounding production practices, housing, management, animal welfare, and supply-chain transparency. Social media platforms such as X (formerly Twitter) generate large volumes of unstructured textual data that capture stakeholder sentiment across the poultry industry. Extracting accurate sentiment signals from this domain-specific discourse remains challenging due to contextual ambiguity, linguistic variability, and limited domain awareness in general-purpose language models. This study presents PoultryLeX-Net, a lexicon-enhanced, domain-adaptive dual-stream transformer framework for fine-grained sentiment analysis in poultry-related text. The proposed architecture integrates sentiment classification, topic modeling, and contextual representation learning through domain-specific embeddings and gated cross-attention mechanisms. A lexicon-guided stream captures poultry-specific terminology and sentiment cues, while contextual stream models long-range semantic dependencies. Latent Dirichlet Allocation is employed to identify dominant thematic structures associated with production management and welfare-related discussions, providing complementary interpretability to sentiment predictions. PoultryLeX-Net was evaluated against multiple baseline models, including convolutional neural network and pre-trained transformer architectures such as DistilBERT and RoBERTa. PoultryLeX-Net consistently outperformed all baselines, achieving an accuracy of 97.35%, an F1 score of 96.67%, and an area under the receiver operating characteristic curve (AUC-ROC) of 99.61% across sentiment classification tasks. Overall, domain adaptation and dual-stream attention markedly improve sentiment classification, enabling scalable intelligence for poultry production decision support. 

---
# A Two-Stage Architecture for NDA Analysis: LLM-based Segmentation and Transformer-based Clause Classification 

**Authors**: Ana Begnini, Matheus Vicente, Leonardo Souza  

**Link**: [PDF](https://arxiv.org/pdf/2603.09990)  

**Abstract**: In business-to-business relations, it is common to establish NonDisclosure Agreements (NDAs). However, these documents exhibit significant variation in format, structure, and writing style, making manual analysis slow and error-prone. We propose an architecture based on LLMs to automate the segmentation and clauses classification within these contracts. We employed two models: LLaMA-3.1-8B-Instruct for NDA segmentation (clause extraction) and a fine-tuned Legal-Roberta-Large for clause classification. In the segmentation task, we achieved a ROUGE F1 of 0.95 +/- 0.0036; for classification, we obtained a weighted F1 of 0.85, demonstrating the feasibility and precision of the approach. 

---
# The System Hallucination Scale (SHS): A Minimal yet Effective Human-Centered Instrument for Evaluating Hallucination-Related Behavior in Large Language Models 

**Authors**: Heimo Müller, Dominik Steiger, Markus Plass, Andreas Holzinger  

**Link**: [PDF](https://arxiv.org/pdf/2603.09989)  

**Abstract**: We introduce the System Hallucination Scale (SHS), a lightweight and human-centered measurement instrument for assessing hallucination-related behavior in large language models (LLMs). Inspired by established psychometric tools such as the System Usability Scale (SUS) and the System Causability Scale (SCS), SHS enables rapid, interpretable, and domain-agnostic evaluation of factual unreliability, incoherence, misleading presentation, and responsiveness to user guidance in model-generated text. SHS is explicitly not an automatic hallucination detector or benchmark metric; instead, it captures how hallucination phenomena manifest from a user perspective under realistic interaction conditions. A real-world evaluation with 210 participants demonstrates high clarity, coherent response behavior, and construct validity, supported by statistical analysis including internal consistency (Cronbach's alpha = 0.87$) and significant inter-dimension correlations (p < 0.001$). Comparative analysis with SUS and SCS reveals complementary measurement properties, supporting SHS as a practical tool for comparative analysis, iterative system development, and deployment monitoring. 

---
# Causally Grounded Mechanistic Interpretability for LLMs with Faithful Natural-Language Explanations 

**Authors**: Ajay Pravin Mahale  

**Link**: [PDF](https://arxiv.org/pdf/2603.09988)  

**Abstract**: Mechanistic interpretability identifies internal circuits responsible for model behaviors, yet translating these findings into human-understandable explanations remains an open problem. We present a pipeline that bridges circuit-level analysis and natural language explanations by (i) identifying causally important attention heads via activation patching, (ii) generating explanations using both template-based and LLM-based methods, and (iii) evaluating faithfulness using ERASER-style metrics adapted for circuit-level attribution. We evaluate on the Indirect Object Identification (IOI) task in GPT-2 Small (124M parameters), identifying six attention heads accounting for 61.4% of the logit difference. Our circuit-based explanations achieve 100% sufficiency but only 22% comprehensiveness, revealing distributed backup mechanisms. LLM-generated explanations outperform template baselines by 64% on quality metrics. We find no correlation (r = 0.009) between model confidence and explanation faithfulness, and identify three failure categories explaining when explanations diverge from mechanisms. 

---
# Evolving Demonstration Optimization for Chain-of-Thought Feature Transformation 

**Authors**: Xinyuan Wang, Kunpeng Liu, Arun Vignesh Malarkkan, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09987)  

**Abstract**: Feature Transformation (FT) is a core data-centric AI task that improves feature space quality to advance downstream predictive performance. However, discovering effective transformations remains challenging due to the large space of feature-operator combinations. Existing solutions rely on discrete search or latent generation, but they are frequently limited by sample inefficiency, invalid candidates, and redundant generations with limited coverage. Large Language Models (LLMs) offer strong priors for producing valid transformations, but current LLM-based FT methods typically rely on static demonstrations, resulting in limited diversity, redundant outputs, and weak alignment with downstream objectives. We propose a framework that optimizes context data for LLM-driven FT by evolving trajectory-level experiences in a closed loop. Starting from high-performing feature transportation sequences explored by reinforcement learning, we construct and continuously update an experience library of downstream task-verified transformation trajectories, and use a diversity-aware selector to form contexts along with a chain-of-thought and guide transformed feature generation toward higher performance. Experiments on diverse tabular benchmarks show that our method outperforms classical and LLM-based baselines and is more stable than one-shot generation. The framework generalizes across API-based and open-source LLMs and remains robust across downstream evaluators. 

---
# Quantifying Hallucinations in Language Language Models on Medical Textbooks 

**Authors**: Brandon C. Colelough, Davis Bartels, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2603.09986)  

**Abstract**: Hallucinations, the tendency for large language models to provide responses with factually incorrect and unsupported claims, is a serious problem within natural language processing for which we do not yet have an effective solution to mitigate against. Existing benchmarks for medical QA rarely evaluate this behavior against a fixed evidence source. We ask how often hallucinations occur on textbook-grounded QA and how responses to medical QA prompts vary across models. We conduct two experiments: the first experiment to determine the prevalence of hallucinations for a prominent open source large language model (LLaMA-70B-Instruct) in medical QA given novel prompts, and the second experiment to determine the prevalence of hallucinations and clinician preference to model responses. We observed, in experiment one, with the passages provided, LLaMA-70B-Instruct hallucinated in 19.7\% of answers (95\% CI 18.6 to 20.7) even though 98.8\% of prompt responses received maximal plausibility, and observed in experiment two, across models, lower hallucination rates aligned with higher usefulness scores ($\rho=-0.71$, $p=0.058$). Clinicians produced high agreement (quadratic weighted $\kappa=0.92$) and ($\tau_b=0.06$ to $0.18$, $\kappa=0.57$ to $0.61$) for experiments 1 and ,2 respectively 

---
# The Dunning-Kruger Effect in Large Language Models: An Empirical Study of Confidence Calibration 

**Authors**: Sudipta Ghosh, Mrityunjoy Panday  

**Link**: [PDF](https://arxiv.org/pdf/2603.09985)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, yet their ability to accurately assess their own confidence remains poorly understood. We present an empirical study investigating whether LLMs exhibit patterns reminiscent of the Dunning-Kruger effect -- a cognitive bias where individuals with limited competence tend to overestimate their abilities. We evaluate four state-of-the-art models (Claude Haiku 4.5, Gemini 2.5 Pro, Gemini 2.5 Flash, and Kimi K2) across four benchmark datasets totaling 24,000 experimental trials. Our results reveal striking calibration differences: Kimi K2 exhibits severe overconfidence with an Expected Calibration Error (ECE) of 0.726 despite only 23.3% accuracy, while Claude Haiku 4.5 achieves the best calibration (ECE = 0.122) with 75.4% accuracy. These findings demonstrate that poorly performing models display markedly higher overconfidence -- a pattern analogous to the Dunning-Kruger effect in human cognition. We discuss implications for safe deployment of LLMs in high-stakes applications. 

---
# MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios 

**Authors**: Shuhuai Li, Jianghao Lin, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.09983)  

**Abstract**: Mixture-of-Experts (MoE) models enable scalable performance but face severe memory constraints on edge devices. Existing offloading strategies struggle with I/O bottlenecks due to the dynamic, low-information nature of autoregressive expert activation. In this paper, we propose to repurpose Speculative Decoding (SD) not merely as a compute accelerator, but as an informative lookahead sensor for memory management, supported by our theoretical and empirical analyses. Hence, we introduce MoE-SpAc, an MoE inference framework that integrates a Speculative Utility Estimator to track expert demand, a Heterogeneous Workload Balancer to dynamically partition computation via online integer optimization, and an Asynchronous Execution Engine to unify the prefetching and eviction in the same utility space. Extensive experiments on seven benchmarks demonstrate that MoE-SpAc achieves a 42% improvement in TPS over the SOTA SD-based baseline, and an average 4.04x speedup over all standard baselines. Code is available at this https URL . 

---
# AraModernBERT: Transtokenized Initialization and Long-Context Encoder Modeling for Arabic 

**Authors**: Omar Elshehy, Omer Nacar, Abdelbasset Djamai, Muhammed Ragab, Khloud Al Jallad, Mona Abdelazim  

**Link**: [PDF](https://arxiv.org/pdf/2603.09982)  

**Abstract**: Encoder-only transformer models remain widely used for discriminative NLP tasks, yet recent architectural advances have largely focused on English. In this work, we present AraModernBERT, an adaptation of the ModernBERT encoder architecture to Arabic, and study the impact of transtokenized embedding initialization and native long-context modeling up to 8,192 tokens. We show that transtokenization is essential for Arabic language modeling, yielding dramatic improvements in masked language modeling performance compared to non-transtokenized initialization. We further demonstrate that AraModernBERT supports stable and effective long-context modeling, achieving improved intrinsic language modeling performance at extended sequence lengths. Downstream evaluations on Arabic natural language understanding tasks, including inference, offensive language detection, question-question similarity, and named entity recognition, confirm strong transfer to discriminative and sequence labeling settings. Our results highlight practical considerations for adapting modern encoder architectures to Arabic and other languages written in Arabic-derived scripts. 

---
# Explainable LLM Unlearning Through Reasoning 

**Authors**: Junfeng Liao, Qizhou Wang, Shanshan Ye, Xin Yu, Ling Chen, Zhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09980)  

**Abstract**: LLM unlearning is essential for mitigating safety, copyright, and privacy concerns in pre-trained large language models (LLMs). Compared to preference alignment, it offers a more explicit way by removing undesirable knowledge characterized by specific unlearning datasets. In previous works, gradient ascent (GA) and its variants have shown promise for implementing unlearning, yet their untargeted nature results in unintended degradation of general capabilities, incomplete removal of knowledge, and the generation of incoherent responses, among many others. We argue that these issues stem from the absence of explicit guidance on what and how models should unlearn. To fill this gap, we introduce a novel unlearning target, reasoning-based unlearning target, which satisfies both the specified unlearning scope and the specified post-unlearning response. Building on this, we propose targeted reasoning unlearning (TRU), which leverages reasoning-based unlearning target as guidance. We employ the target using a cross-entropy supervised loss combined with a GA-based loss, enabling the model to learn reasoning ability for precise knowledge removal while preserving unrelated abilities. We evaluate TRU against strong baselines across multiple benchmarks and LLM backbones, and find that it achieves more reliable unlearning while preserving general capabilities. Moreover, TRU exhibits superior robustness under diverse attack scenarios, stemming from the reasoning ability learned through reasoning-based targets. Overall, our study establishes reasoning-augmented unlearning as a practical paradigm for reliable and explainable LLM unlearning. 

---
# One Model, Many Skills: Parameter-Efficient Fine-Tuning for Multitask Code Analysis 

**Authors**: Amal Akli, Maxime Cordy, Mike Papadakis, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2603.09978)  

**Abstract**: Large language models have recently surpassed specialized systems on code generation, yet their effectiveness on other code-analysis tasks remains less clear. At the same time, multi-task learning offers a way to unify diverse objectives within a single model, but fully fine-tuning LLMs across tasks is computationally prohibitive. Parameter-efficient fine-tuning mitigates this cost by updating only a small fraction of weights. Although PEFT has proven effective in single-task settings, its potential for multi-task learning has not yet been systematically explored. We present the first comprehensive evaluation of multi-task PEFT for code analysis, comparing several methods across diverse tasks and model architectures. Our experiments show that a single PEFT module shared across tasks can match, and in some cases surpass, full multi-task fine-tuning, confirming that the benefits of PEFT extend beyond isolated tasks. When comparing single-task and multi-task setups, we find that multi-task PEFT achieves a favorable performance-efficiency trade-off: it delivers accuracy close to single-task fine-tuning while reducing storage requirements, cutting the number of trainable parameters by a factor of the task count, and lowering computation costs by as much as 85%. At the same time, multi-task gains remain sensitive to task grouping. Through task-pairing experiments, we identify key factors shaping outcomes: task stability, model architecture, task complementarity, asymmetry, and dataset quality determine the success of co-fine-tuning. Finally, we benchmark efficient multi-task PEFT against direct prompting of open-source general-purpose LLMs, including DeepSeek, Qwen, Mistral, CodeLlama, and StarCoder. Despite their strong performance in code generation, these models underperform on analysis tasks, where even a 1B-parameter model with multi-task PEFT achieves significantly better results. 

---
# Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards 

**Authors**: Zhengzhao Ma, Xueru Wen, Boxi Cao, Yaojie Lu, Hongyu Lin, Jinglin Yang, Min He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.09117)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) significantly enhances large language models (LLMs) reasoning but severely suffers from calibration degeneration, where models become excessively over-confident in incorrect answers. Previous studies devote to directly incorporating calibration objective into existing optimization target. However, our theoretical analysis demonstrates that there exists a fundamental gradient conflict between the optimization for maximizing policy accuracy and minimizing calibration error. Building on this insight, we propose DCPO, a simple yet effective framework that systematically decouples reasoning and calibration objectives. Extensive experiments demonstrate that our DCPO not only preserves accuracy on par with GRPO but also achieves the best calibration performance and substantially mitigates the over-confidence issue. Our study provides valuable insights and practical solution for more reliable LLM deployment. 

---
