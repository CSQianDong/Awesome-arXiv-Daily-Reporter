# The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements 

**Authors**: Bingchen Zhao, Despoina Magka, Minqi Jiang, Xian Li, Roberta Raileanu, Tatiana Shavrina, Jean-Christophe Gagnon-Audet, Kelvin Niu, Shagun Sodhani, Michael Shvartsman, Andrei Lupu, Alisia Lupidi, Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Thomas Foster, Lucia Cipolina-Kun, Abhishek Charnalia, Derek Dunfield, Alexander H. Miller, Oisin Mac Aodha, Jakob Foerster, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2506.22419)  

**Abstract**: Rapid advancements in large language models (LLMs) have the potential to assist in scientific progress. A critical capability toward this endeavor is the ability to reproduce existing work. To evaluate the ability of AI agents to reproduce results in an active research area, we introduce the Automated LLM Speedrunning Benchmark, leveraging the research community contributions on the NanoGPT speedrun, a competition to train a GPT-2 model in the shortest time. Each of the 19 speedrun tasks provides the agent with the previous records training script, optionally paired with one of three hint formats, ranging from pseudocode to paper-like descriptions of the new records improvements. Records execute quickly by design and speedrun improvements encompass diverse code-level changes, ranging from high-level algorithmic advancements to hardware-aware optimizations. These features make the benchmark both accessible and realistic for the frontier problem of improving LLM training. We find that recent reasoning LLMs combined with SoTA scaffolds struggle to reimplement already-known innovations in our benchmark, even when given detailed hints. Our benchmark thus provides a simple, non-saturated measure of an LLMs ability to automate scientific reproduction, a necessary (but not sufficient) skill for an autonomous research agent. 

---
# AI Model Passport: Data and System Traceability Framework for Transparent AI in Health 

**Authors**: Varvara Kalokyri, Nikolaos S. Tachos, Charalampos N. Kalantzopoulos, Stelios Sfakianakis, Haridimos Kondylakis, Dimitrios I. Zaridis, Sara Colantonio, Daniele Regge, Nikolaos Papanikolaou, ProCAncer-I consortium, Konstantinos Marias, Dimitrios I. Fotiadis, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.22358)  

**Abstract**: The increasing integration of Artificial Intelligence (AI) into health and biomedical systems necessitates robust frameworks for transparency, accountability, and ethical compliance. Existing frameworks often rely on human-readable, manual documentation which limits scalability, comparability, and machine interpretability across projects and platforms. They also fail to provide a unique, verifiable identity for AI models to ensure their provenance and authenticity across systems and use cases, limiting reproducibility and stakeholder trust. This paper introduces the concept of the AI Model Passport, a structured and standardized documentation framework that acts as a digital identity and verification tool for AI models. It captures essential metadata to uniquely identify, verify, trace and monitor AI models across their lifecycle - from data acquisition and preprocessing to model design, development and deployment. In addition, an implementation of this framework is presented through AIPassport, an MLOps tool developed within the ProCAncer-I EU project for medical imaging applications. AIPassport automates metadata collection, ensures proper versioning, decouples results from source scripts, and integrates with various development environments. Its effectiveness is showcased through a lesion segmentation use case using data from the ProCAncer-I dataset, illustrating how the AI Model Passport enhances transparency, reproducibility, and regulatory readiness while reducing manual effort. This approach aims to set a new standard for fostering trust and accountability in AI-driven healthcare solutions, aspiring to serve as the basis for developing transparent and regulation compliant AI systems across domains. 

---
# Embodied AI Agents: Modeling the World 

**Authors**: Pascale Fung, Yoram Bachrach, Asli Celikyilmaz, Kamalika Chaudhuri, Delong Chen, Willy Chung, Emmanuel Dupoux, Hervé Jégou, Alessandro Lazaric, Arjun Majumdar, Andrea Madotto, Franziska Meier, Florian Metze, Théo Moutakanni, Juan Pino, Basile Terver, Joseph Tighe, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.22355)  

**Abstract**: This paper describes our research on AI agents embodied in visual, virtual or physical forms, enabling them to interact with both users and their environments. These agents, which include virtual avatars, wearable devices, and robots, are designed to perceive, learn and act within their surroundings, which makes them more similar to how humans learn and interact with the environments as compared to disembodied agents. We propose that the development of world models is central to reasoning and planning of embodied AI agents, allowing these agents to understand and predict their environment, to understand user intentions and social contexts, thereby enhancing their ability to perform complex tasks autonomously. World modeling encompasses the integration of multimodal perception, planning through reasoning for action and control, and memory to create a comprehensive understanding of the physical world. Beyond the physical world, we also propose to learn the mental world model of users to enable better human-agent collaboration. 

---
# Conceptual Topic Aggregation 

**Authors**: Klara M. Gutekunst, Dominik Dürrschnabel, Johannes Hirth, Gerd Stumme  

**Link**: [PDF](https://arxiv.org/pdf/2506.22309)  

**Abstract**: The vast growth of data has rendered traditional manual inspection infeasible, necessitating the adoption of computational methods for efficient data exploration. Topic modeling has emerged as a powerful tool for analyzing large-scale textual datasets, enabling the extraction of latent semantic structures. However, existing methods for topic modeling often struggle to provide interpretable representations that facilitate deeper insights into data structure and content. In this paper, we propose FAT-CAT, an approach based on Formal Concept Analysis (FCA) to enhance meaningful topic aggregation and visualization of discovered topics. Our approach can handle diverse topics and file types -- grouped by directories -- to construct a concept lattice that offers a structured, hierarchical representation of their topic distribution. In a case study on the ETYNTKE dataset, we evaluate the effectiveness of our approach against other representation methods to demonstrate that FCA-based aggregation provides more meaningful and interpretable insights into dataset composition than existing topic modeling techniques. 

---
# Artificial Intelligent Disobedience: Rethinking the Agency of Our Artificial Teammates 

**Authors**: Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.22276)  

**Abstract**: Artificial intelligence has made remarkable strides in recent years, achieving superhuman performance across a wide range of tasks. Yet despite these advances, most cooperative AI systems remain rigidly obedient, designed to follow human instructions without question and conform to user expectations, even when doing so may be counterproductive or unsafe. This paper argues for expanding the agency of AI teammates to include \textit{intelligent disobedience}, empowering them to make meaningful and autonomous contributions within human-AI teams. It introduces a scale of AI agency levels and uses representative examples to highlight the importance and growing necessity of treating AI autonomy as an independent research focus in cooperative settings. The paper then explores how intelligent disobedience manifests across different autonomy levels and concludes by proposing initial boundaries and considerations for studying disobedience as a core capability of artificial agents. 

---
# Breaking Rank Bottlenecks in Knowledge Graph Completion 

**Authors**: Samy Badreddine, Emile van Krieken, Luciano Serafini  

**Link**: [PDF](https://arxiv.org/pdf/2506.22271)  

**Abstract**: Many Knowledge Graph Completion (KGC) models, despite using powerful encoders, rely on a simple vector-matrix multiplication to score queries against candidate object entities. When the number of entities is larger than the model's embedding dimension, which in practical scenarios is often by several orders of magnitude, we have a linear output layer with a rank bottleneck. Such bottlenecked layers limit model expressivity. We investigate both theoretically and empirically how rank bottlenecks affect KGC models. We find that, by limiting the set of feasible predictions, rank bottlenecks hurt ranking accuracy and the distribution fidelity of scores. Inspired by the language modelling literature, we propose KGE-MoS, a mixture-based output layer to break rank bottlenecks in many KGC models. Our experiments on four datasets show that KGE-MoS improves performance and probabilistic fit of KGC models for a low parameter cost. 

---
# A Different Approach to AI Safety: Proceedings from the Columbia Convening on Openness in Artificial Intelligence and AI Safety 

**Authors**: Camille François, Ludovic Péran, Ayah Bdeir, Nouha Dziri, Will Hawkins, Yacine Jernite, Sayash Kapoor, Juliet Shen, Heidy Khlaaf, Kevin Klyman, Nik Marda, Marie Pellat, Deb Raji, Divya Siddarth, Aviya Skowron, Joseph Spisak, Madhulika Srikumar, Victor Storchan, Audrey Tang, Jen Weedon  

**Link**: [PDF](https://arxiv.org/pdf/2506.22183)  

**Abstract**: The rapid rise of open-weight and open-source foundation models is intensifying the obligation and reshaping the opportunity to make AI systems safe. This paper reports outcomes from the Columbia Convening on AI Openness and Safety (San Francisco, 19 Nov 2024) and its six-week preparatory programme involving more than forty-five researchers, engineers, and policy leaders from academia, industry, civil society, and government. Using a participatory, solutions-oriented process, the working groups produced (i) a research agenda at the intersection of safety and open source AI; (ii) a mapping of existing and needed technical interventions and open source tools to safely and responsibly deploy open foundation models across the AI development workflow; and (iii) a mapping of the content safety filter ecosystem with a proposed roadmap for future research and development. We find that openness -- understood as transparent weights, interoperable tooling, and public governance -- can enhance safety by enabling independent scrutiny, decentralized mitigation, and culturally plural oversight. However, significant gaps persist: scarce multimodal and multilingual benchmarks, limited defenses against prompt-injection and compositional attacks in agentic systems, and insufficient participatory mechanisms for communities most affected by AI harms. The paper concludes with a roadmap of five priority research directions, emphasizing participatory inputs, future-proof content filters, ecosystem-wide safety infrastructure, rigorous agentic safeguards, and expanded harm taxonomies. These recommendations informed the February 2025 French AI Action Summit and lay groundwork for an open, plural, and accountable AI safety discipline. 

---
# Query as Test: An Intelligent Driving Test and Data Storage Method for Integrated Cockpit-Vehicle-Road Scenarios 

**Authors**: Shengyue Yao, Runqing Guo, Yangyang Qin, Miangbing Meng, Jipeng Cao, Yilun Lin, Yisheng Lv, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22068)  

**Abstract**: With the deep penetration of Artificial Intelligence (AI) in the transportation sector, intelligent cockpits, autonomous driving, and intelligent road networks are developing at an unprecedented pace. However, the data ecosystems of these three key areas are increasingly fragmented and incompatible. Especially, existing testing methods rely on data stacking, fail to cover all edge cases, and lack flexibility. To address this issue, this paper introduces the concept of "Query as Test" (QaT). This concept shifts the focus from rigid, prescripted test cases to flexible, on-demand logical queries against a unified data representation. Specifically, we identify the need for a fundamental improvement in data storage and representation, leading to our proposal of "Extensible Scenarios Notations" (ESN). ESN is a novel declarative data framework based on Answer Set Programming (ASP), which uniformly represents heterogeneous multimodal data from the cockpit, vehicle, and road as a collection of logical facts and rules. This approach not only achieves deep semantic fusion of data, but also brings three core advantages: (1) supports complex and flexible semantic querying through logical reasoning; (2) provides natural interpretability for decision-making processes; (3) allows for on-demand data abstraction through logical rules, enabling fine-grained privacy protection. We further elaborate on the QaT paradigm, transforming the functional validation and safety compliance checks of autonomous driving systems into logical queries against the ESN database, significantly enhancing the expressiveness and formal rigor of the testing. Finally, we introduce the concept of "Validation-Driven Development" (VDD), which suggests to guide developments by logical validation rather than quantitative testing in the era of Large Language Models, in order to accelerating the iteration and development process. 

---
# Universal Retrieval for Multimodal Trajectory Modeling 

**Authors**: Xuan Zhang, Ziyan Jiang, Rui Meng, Yifei Leng, Zhenbang Xiao, Zora Zhiruo Wang, Yanyi Shang, Dehan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.22056)  

**Abstract**: Trajectory data, capturing human actions and environmental states across various modalities, holds significant potential for enhancing AI agent capabilities, particularly in GUI environments. However, how to model the representation of trajectory-level data presents a significant challenge that has not been systematically addressed amid explosive trajectory data growth. In this work, we introduce Multimodal Trajectory Retrieval, bridging the gap between universal retrieval and agent-centric trajectory modeling. We construct the Unified Agent Trajectory Dataset (UATD) from annotated demonstrations and states across diverse real-world scenarios. Based on this, we present GAE-Bench, a benchmark containing a large number of trajectory-based retrieval pairs. In addition, we propose GAE-Retriever, a multimodal retrieval framework that adopts vision-language models and incorporates optimized contrastive learning through a token selection and the GradCache mechanism. Comprehensive evaluations across multiple datasets show that GAE-Retriever consistently outperforms strong baselines in retrieval recall, highlighting its effectiveness in advancing multimodal trajectory retrieval. 

---
# LeanConjecturer: Automatic Generation of Mathematical Conjectures for Theorem Proving 

**Authors**: Naoto Onda, Kazumi Kasaura, Yuta Oriike, Masaya Taniguchi, Akiyoshi Sannai, Sho Sonoda  

**Link**: [PDF](https://arxiv.org/pdf/2506.22005)  

**Abstract**: We introduce LeanConjecturer, a pipeline for automatically generating university-level mathematical conjectures in Lean 4 using Large Language Models (LLMs). Our hybrid approach combines rule-based context extraction with LLM-based theorem statement generation, addressing the data scarcity challenge in formal theorem proving. Through iterative generation and evaluation, LeanConjecturer produced 12,289 conjectures from 40 Mathlib seed files, with 3,776 identified as syntactically valid and non-trivial, that is, cannot be proven by \texttt{aesop} tactic. We demonstrate the utility of these generated conjectures for reinforcement learning through Group Relative Policy Optimization (GRPO), showing that targeted training on domain-specific conjectures can enhance theorem proving capabilities. Our approach generates 103.25 novel conjectures per seed file on average, providing a scalable solution for creating training data for theorem proving systems. Our system successfully verified several non-trivial theorems in topology, including properties of semi-open, alpha-open, and pre-open sets, demonstrating its potential for mathematical discovery beyond simple variations of existing results. 

---
# AlphaBeta is not as good as you think: a new probabilistic model to better analyze deterministic game-solving algorithms 

**Authors**: Raphaël Boige, Amine Boumaza, Bruno Scherrer  

**Link**: [PDF](https://arxiv.org/pdf/2506.21996)  

**Abstract**: Deterministic game-solving algorithms are conventionally analyzed in the light of their average-case complexity against a distribution of random game-trees, where leaf values are independently sampled from a fixed distribution. This simplified model enables uncluttered mathematical analysis, revealing two key properties: root value distributions asymptotically collapse to a single fixed value for finite-valued trees, and all reasonable algorithms achieve global optimality. However, these findings are artifacts of the model's design-its long criticized independence assumption strips games of structural complexity, producing trivial instances where no algorithm faces meaningful challenges. To address this limitation, we introduce a new probabilistic model that incrementally constructs game-trees using a fixed level-wise conditional distribution. By enforcing ancestor dependency, a critical structural feature of real-world games, our framework generates problems with adjustable difficulty while retaining some form of analytical tractability. For several algorithms, including AlphaBeta and Scout, we derive recursive formulas characterizing their average-case complexities under this model. These allow us to rigorously compare algorithms on deep game-trees, where Monte-Carlo simulations are no longer feasible. While asymptotically, all algorithms seem to converge to identical branching factor (a result analogous to those of independence-based models), deep finite trees reveal stark differences: AlphaBeta incurs a significantly larger constant multiplicative factor compared to algorithms like Scout, leading to a substantial practical slowdown. Our framework sheds new light on classical game-solving algorithms, offering rigorous evidence and analytical tools to advance the understanding of these methods under a more realistic, challenging, and yet tractable model. 

---
# Interactive Multi-Objective Probabilistic Preference Learning with Soft and Hard Bounds 

**Authors**: Edward Chen, Sang T. Truong, Natalie Dullerud, Sanmi Koyejo, Carlos Guestrin  

**Link**: [PDF](https://arxiv.org/pdf/2506.21887)  

**Abstract**: High-stakes decision-making involves navigating multiple competing objectives with expensive evaluations. For instance, in brachytherapy, clinicians must balance maximizing tumor coverage (e.g., an aspirational target or soft bound of >95% coverage) against strict organ dose limits (e.g., a non-negotiable hard bound of <601 cGy to the bladder), with each plan evaluation being resource-intensive. Selecting Pareto-optimal solutions that match implicit preferences is challenging, as exhaustive Pareto frontier exploration is computationally and cognitively prohibitive, necessitating interactive frameworks to guide users. While decision-makers (DMs) often possess domain knowledge to narrow the search via such soft-hard bounds, current methods often lack systematic approaches to iteratively refine these multi-faceted preference structures. Critically, DMs must trust their final decision, confident they haven't missed superior alternatives; this trust is paramount in high-consequence scenarios. We present Active-MoSH, an interactive local-global framework designed for this process. Its local component integrates soft-hard bounds with probabilistic preference learning, maintaining distributions over DM preferences and bounds for adaptive Pareto subset refinement. This is guided by an active sampling strategy optimizing exploration-exploitation while minimizing cognitive burden. To build DM trust, Active-MoSH's global component, T-MoSH, leverages multi-objective sensitivity analysis to identify potentially overlooked, high-value points beyond immediate feedback. We demonstrate Active-MoSH's performance benefits through diverse synthetic and real-world applications. A user study on AI-generated image selection further validates our hypotheses regarding the framework's ability to improve convergence, enhance DM trust, and provide expressive preference articulation, enabling more effective DMs. 

---
# CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation 

**Authors**: Nicolas Bougie, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.21805)  

**Abstract**: Modeling human behavior in urban environments is fundamental for social science, behavioral studies, and urban planning. Prior work often rely on rigid, hand-crafted rules, limiting their ability to simulate nuanced intentions, plans, and adaptive behaviors. Addressing these challenges, we envision an urban simulator (CitySim), capitalizing on breakthroughs in human-level intelligence exhibited by large language models. In CitySim, agents generate realistic daily schedules using a recursive value-driven approach that balances mandatory activities, personal habits, and situational factors. To enable long-term, lifelike simulations, we endow agents with beliefs, long-term goals, and spatial memory for navigation. CitySim exhibits closer alignment with real humans than prior work, both at micro and macro levels. Additionally, we conduct insightful experiments by modeling tens of thousands of agents and evaluating their collective behaviors under various real-world scenarios, including estimating crowd density, predicting place popularity, and assessing well-being. Our results highlight CitySim as a scalable, flexible testbed for understanding and forecasting urban phenomena. 

---
# MobiVerse: Scaling Urban Mobility Simulation with Hybrid Lightweight Domain-Specific Generator and Large Language Models 

**Authors**: Yifan Liu, Xishun Liao, Haoxuan Ma, Jonathan Liu, Rohan Jadhav, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.21784)  

**Abstract**: Understanding and modeling human mobility patterns is crucial for effective transportation planning and urban development. Despite significant advances in mobility research, there remains a critical gap in simulation platforms that allow for algorithm development, policy implementation, and comprehensive evaluation at scale. Traditional activity-based models require extensive data collection and manual calibration, machine learning approaches struggle with adaptation to dynamic conditions, and treding agent-based Large Language Models (LLMs) implementations face computational constraints with large-scale simulations. To address these challenges, we propose MobiVerse, a hybrid framework leverages the efficiency of lightweight domain-specific generator for generating base activity chains with the adaptability of LLMs for context-aware modifications. A case study was conducted in Westwood, Los Angeles, where we efficiently generated and dynamically adjusted schedules for the whole population of approximately 53,000 agents on a standard PC. Our experiments demonstrate that MobiVerse successfully enables agents to respond to environmental feedback, including road closures, large gathering events like football games, and congestion, through our hybrid framework. Its modular design facilitates testing various mobility algorithms at both transportation system and agent levels. Results show our approach maintains computational efficiency while enhancing behavioral realism. MobiVerse bridges the gap in mobility simulation by providing a customizable platform for mobility systems planning and operations with benchmark algorithms. Code and videos are available at this https URL. 

---
# THE-Tree: Can Tracing Historical Evolution Enhance Scientific Verification and Reasoning? 

**Authors**: Xin Wang, Jiyao Liu, Yulong Xiao, Junzhi Ning, Lihao Liu, Junjun He, Botian Shi, Kaicheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21763)  

**Abstract**: Large Language Models (LLMs) are accelerating scientific idea generation, but rigorously evaluating these numerous, often superficial, AI-generated propositions for novelty and factual accuracy is a critical bottleneck; manual verification is too this http URL validation methods are inadequate: LLMs as standalone verifiers may hallucinate and lack domain knowledge (our findings show ~60\% unawareness of relevant papers in specific domains), while traditional citation networks lack explicit causality and narrative surveys are this http URL underscores a core challenge: the absence of structured, verifiable, and causally-linked historical data of scientific this http URL address this,we introduce \textbf{THE-Tree} (\textbf{T}echnology \textbf{H}istory \textbf{E}volution Tree), a computational framework that constructs such domain-specific evolution trees from scientific this http URL-Tree employs a search algorithm to explore evolutionary paths. During its node expansion, it utilizes a novel "Think-Verbalize-Cite-Verify" process: an LLM proposes potential advancements and cites supporting literature. Critically, each proposed evolutionary link is then validated for logical coherence and evidential support by a recovered natural language inference mechanism that interrogates the cited literature, ensuring that each step is this http URL construct and validate 88 THE-Trees across diverse domains and release a benchmark dataset including up to 71k fact verifications covering 27k papers to foster further this http URL demonstrate that i) in graph completion, our THE-Tree improves hit@1 by 8\% to 14\% across multiple models compared to traditional citation networks; ii) for predicting future scientific developments, it improves hit@1 metric by nearly 10\%; and iii) when combined with other methods, it boosts the performance of evaluating important scientific papers by almost 100\%. 

---
# Hierarchical Reasoning Model 

**Authors**: Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori  

**Link**: [PDF](https://arxiv.org/pdf/2506.21734)  

**Abstract**: Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI. Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency. HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes. Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities. These results underscore HRM's potential as a transformative advancement toward universal computation and general-purpose reasoning systems. 

---
# SEEA-R1: Tree-Structured Reinforcement Fine-Tuning for Self-Evolving Embodied Agents 

**Authors**: Wanxin Tian, Shijie Zhang, Kevin Zhang, Xiaowei Chi, Yulin Luo, Junyu Lu, Chunkai Fan, Qiang Zhou, Yiming Zhao, Ning Liu Siyu Lin, Zhiyuan Qin, Xiaozhu Ju, Shanghang Zhang, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21669)  

**Abstract**: Self-evolution, the ability of agents to autonomously improve their reasoning and behavior, is essential for the embodied domain with long-horizon, real-world tasks. Despite current advancements in reinforcement fine-tuning (RFT) showing strong performance in enhancing reasoning in LLMs, its potential to enable self-evolving embodied intelligence with multi-modal interactions remains largely unexplored. Specifically, reinforcement fine-tuning faces two fundamental obstacles in embodied settings: (i) the lack of accessible intermediate rewards in multi-step reasoning tasks limits effective learning signals, and (ii) reliance on hand-crafted reward functions restricts generalization to novel tasks and environments. To address these challenges, we present Self-Evolving Embodied Agents-R1, SEEA-R1, the first RFT framework designed for enabling the self-evolving capabilities of embodied agents. Specifically, to convert sparse delayed rewards into denser intermediate signals that improve multi-step reasoning, we propose Tree-based group relative policy optimization (Tree-GRPO), which integrates Monte Carlo Tree Search into GRPO. To generalize reward estimation across tasks and scenes, supporting autonomous adaptation and reward-driven self-evolution, we further introduce Multi-modal Generative Reward Model (MGRM). To holistically evaluate the effectiveness of SEEA-R1, we evaluate on the ALFWorld benchmark, surpassing state-of-the-art methods with scores of 85.07% (textual) and 36.19% (multi-modal), outperforming prior models including GPT-4o. SEEA-R1 also achieves scores of 80.3% without environmental reward, surpassing all open-source baselines and highlighting its scalability as a self-evolving embodied agent. Additional experiments and qualitative analysis further support the potential of SEEA-R1 for future research in scalable embodied intelligence. 

---
# CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings 

**Authors**: Randeep Bhatia, Nikos Papadis, Murali Kodialam, TV Lakshman, Sayak Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2506.22427)  

**Abstract**: We propose CLoVE (Clustering of Loss Vector Embeddings), a novel algorithm for Clustered Federated Learning (CFL). In CFL, clients are naturally grouped into clusters based on their data distribution. However, identifying these clusters is challenging, as client assignments are unknown. CLoVE utilizes client embeddings derived from model losses on client data, and leverages the insight that clients in the same cluster share similar loss values, while those in different clusters exhibit distinct loss patterns. Based on these embeddings, CLoVE is able to iteratively identify and separate clients from different clusters and optimize cluster-specific models through federated aggregation. Key advantages of CLoVE over existing CFL algorithms are (1) its simplicity, (2) its applicability to both supervised and unsupervised settings, and (3) the fact that it eliminates the need for near-optimal model initialization, which makes it more robust and better suited for real-world applications. We establish theoretical convergence bounds, showing that CLoVE can recover clusters accurately with high probability in a single round and converges exponentially fast to optimal models in a linear setting. Our comprehensive experiments comparing with a variety of both CFL and generic Personalized Federated Learning (PFL) algorithms on different types of datasets and an extensive array of non-IID settings demonstrate that CLoVE achieves highly accurate cluster recovery in just a few rounds of training, along with state-of-the-art model accuracy, across a variety of both supervised and unsupervised PFL tasks. 

---
# HyperCLOVA X THINK Technical Report 

**Authors**: NAVER Cloud HyperCLOVA X Team  

**Link**: [PDF](https://arxiv.org/pdf/2506.22403)  

**Abstract**: We introduce HyperCLOVA X THINK, the first reasoning-focused large language model in the HyperCLOVA X family, pre-trained on roughly $6$ trillion high-quality Korean, and English tokens, augmented with targeted synthetic Korean data. It was implemented as a compute-memory-balanced Peri-LN Transformer scaled with $\mu$P, pre-trained through a three-stage curriculum that expands the context window to $128$K tokens, and post-trained via supervised fine-tuning with Reinforcement Learning from Verifiable Rewards supports both detailed rationale and concise-answer modes. It delivers competitive performance against similarly sized models on Korea-focused benchmarks such as KMMLU, CSAT, KoBALT-700, HAERAE-1.0, and KoBigBench, while preserving robust bilingual consistency and translation quality. In addition, a vision-augmented variant matches or exceeds GPT-4.1 on the KCSAT STEM benchmark, all of which are achieved with substantially lower training compute than existing models of similar sizes. We also present a pruning and distillation technique that will soon be applied to HyperCLOVA X THINK for an open-source and business-friendly foundation model. Altogether, these capabilities position HyperCLOVA X THINK as a robust foundation for Korean AI innovation and a valuable resource for the global research community. 

---
# Dehazing Light Microscopy Images with Guided Conditional Flow Matching: finding a sweet spot between fidelity and realism 

**Authors**: Anirban Ray, Ashesh, Florian Jug  

**Link**: [PDF](https://arxiv.org/pdf/2506.22397)  

**Abstract**: Fluorescence microscopy is a major driver of scientific progress in the life sciences. Although high-end confocal microscopes are capable of filtering out-of-focus light, cheaper and more accessible microscopy modalities, such as widefield microscopy, can not, which consequently leads to hazy image data. Computational dehazing is trying to combine the best of both worlds, leading to cheap microscopy but crisp-looking images. The perception-distortion trade-off tells us that we can optimize either for data fidelity, e.g. low MSE or high PSNR, or for data realism, measured by perceptual metrics such as LPIPS or FID. Existing methods either prioritize fidelity at the expense of realism, or produce perceptually convincing results that lack quantitative accuracy. In this work, we propose HazeMatching, a novel iterative method for dehazing light microscopy images, which effectively balances these objectives. Our goal was to find a balanced trade-off between the fidelity of the dehazing results and the realism of individual predictions (samples). We achieve this by adapting the conditional flow matching framework by guiding the generative process with a hazy observation in the conditional velocity field. We evaluate HazeMatching on 5 datasets, covering both synthetic and real data, assessing both distortion and perceptual quality. Our method is compared against 7 baselines, achieving a consistent balance between fidelity and realism on average. Additionally, with calibration analysis, we show that HazeMatching produces well-calibrated predictions. Note that our method does not need an explicit degradation operator to exist, making it easily applicable on real microscopy data. All data used for training and evaluation and our code will be publicly available under a permissive license. 

---
# QuickSilver -- Speeding up LLM Inference through Dynamic Token Halting, KV Skipping, Contextual Token Fusion, and Adaptive Matryoshka Quantization 

**Authors**: Danush Khanna, Aditya Kumar Guru, Srivarshinee Sridhar, Zidan Ahmed, Rubhav Bahirwani, Meetu Malhotra, Vinija Jain, Aman Chadha, Amitava Das, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2506.22396)  

**Abstract**: Inference accounts for the majority of latency and energy consumption in large language model (LLM) deployments, often exceeding 90% of total cost. While training-time efficiency has seen extensive progress, runtime optimization remains a key bottleneck, particularly under autoregressive decoding. Existing approaches -- such as pruning, quantization, early exits, and speculative decoding -- often require retraining, architectural changes, or disrupt decoding compatibility. We introduce QuickSilver, a modular, token-level framework that enables semantic adaptivity at inference time without altering model weights or structure. QuickSilver integrates four synergistic mechanisms:
(i) Dynamic Token Halting, which halts computation for tokens with converged representations; (ii) KV Cache Skipping, which selectively suppresses memory writes to reduce attention overhead; and (iii) Contextual Token Fusion, which collapses redundant tokens into shared paths to shrink sequence length.
Unlike speculative decoding or MoE routing, QuickSilver operates entirely on frozen, dense models and requires no auxiliary networks. Applied to GPT-2 and Llama-2 across WikiText-103 and C4, QuickSilver achieves up to 39.6% FLOP reduction with negligible perplexity degradation (<=0.2). 

---
# Multi-View Contrastive Learning for Robust Domain Adaptation in Medical Time Series Analysis 

**Authors**: YongKyung Oh, Alex Bui  

**Link**: [PDF](https://arxiv.org/pdf/2506.22393)  

**Abstract**: Adapting machine learning models to medical time series across different domains remains a challenge due to complex temporal dependencies and dynamic distribution shifts. Current approaches often focus on isolated feature representations, limiting their ability to fully capture the intricate temporal dynamics necessary for robust domain adaptation. In this work, we propose a novel framework leveraging multi-view contrastive learning to integrate temporal patterns, derivative-based dynamics, and frequency-domain features. Our method employs independent encoders and a hierarchical fusion mechanism to learn feature-invariant representations that are transferable across domains while preserving temporal coherence. Extensive experiments on diverse medical datasets, including electroencephalogram (EEG), electrocardiogram (ECG), and electromyography (EMG) demonstrate that our approach significantly outperforms state-of-the-art methods in transfer learning tasks. By advancing the robustness and generalizability of machine learning models, our framework offers a practical pathway for deploying reliable AI systems in diverse healthcare settings. 

---
# Towards Distributed Neural Architectures 

**Authors**: Aditya Cowsik, Tianyu He, Andrey Gromov  

**Link**: [PDF](https://arxiv.org/pdf/2506.22389)  

**Abstract**: We introduce and train distributed neural architectures (DNA) in vision and language domains. DNAs are initialized with a proto-architecture that consists of (transformer, MLP, attention, etc.) modules and routers. Any token (or patch) can traverse any series of modules in any order. DNAs are a natural generalization of the sparse methods such as Mixture-of-Experts, Mixture-of-Depths, parameter sharing, etc. Computation and communication patterns of DNA modules are learnt end-to-end during training and depend on the content and context of each token (or patch). These patterns can be shaped by further requirements added to the optimization objective such as compute/memory efficiency or load balancing. We empirically show that (i) trained DNAs are competitive with the dense baselines in both domains and (ii) compute efficiency/parameter sharing can be learnt from data. Next, we analyze the emergent connectivity and computation patterns in the trained DNAs. We find that the paths that tokens take through the models are themselves distributed according to a power-law. We show that some paths (or, equivalently, groups of modules) show emergent specialization. Finally, we demonstrate that models learn to allocate compute and active parameters in an interpretable way. 

---
# Can Video Large Multimodal Models Think Like Doubters-or Double-Down: A Study on Defeasible Video Entailment 

**Authors**: Yue Zhang, Jilei Sun, Yunhui Guo, Vibhav Gogate  

**Link**: [PDF](https://arxiv.org/pdf/2506.22385)  

**Abstract**: Video Large Multimodal Models (VLMMs) have made impressive strides in understanding video content, but they often struggle with abstract and adaptive reasoning-the ability to revise their interpretations when new information emerges. In reality, conclusions are rarely set in stone; additional context can strengthen or weaken an initial inference. To address this, we introduce Defeasible Video Entailment (DVidE), a new task that challenges models to think like doubters, constantly updating their reasoning based on evolving evidence. In DVidE, given a video premise and a textual hypothesis, models must determine whether a new update strengthens or weakens the hypothesis (classification version) or generate a coherent update that modifies the entailment relationship (generation version). For solving the classification task, we propose the Chain of Counterfactual Thought framework, utilizing counterfactual reasoning, ASR-enhanced video content, and rationale refinement to reduce inference bias. For the generation task, we develop a framework that combines ASR output with a Large Language Model (LLM) to produce coherent, contextually relevant updates aligned with the intended strengthener or weakener goals. Additionally, we introduce a novel benchmark dataset, with strengthener/weakener annotations and an LLM-based evaluation metric specifically designed for assessing generative performance. Experimental results demonstrate significant improvements, highlighting our proposed method in enhancing dynamic reasoning capabilities of VLMMs. 

---
# Probabilistic Optimality for Inference-time Scaling 

**Authors**: Youkang Wang, Jian Wang, Rubing Chen, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22376)  

**Abstract**: Inference-time scaling has emerged as a powerful technique for enhancing the reasoning performance of Large Language Models (LLMs). However, existing approaches often rely on heuristic strategies for parallel sampling, lacking a principled foundation. To address this gap, we propose a probabilistic framework that formalizes the optimality of inference-time scaling under the assumption that parallel samples are independently and identically distributed (i.i.d.), and where the Best-of-N selection strategy follows a probability distribution that can be estimated. Within this framework, we derive a theoretical lower bound on the required number of samples to achieve a target performance level, providing the first principled guidance for compute-efficient scaling. Leveraging this insight, we develop \textsc{OptScale}, a practical algorithm that dynamically determines the optimal number of sampled responses. \textsc{OptScale} employs a language model-based predictor to estimate probabilistic prior parameters, enabling the decision of the minimal number of samples needed that satisfy predefined performance thresholds and confidence levels. Extensive experiments on mathematical reasoning benchmarks (including MATH-500, GSM8K, AIME, and AMC) demonstrate that \textsc{OptScale} significantly reduces sampling overhead while remaining better or on par with state-of-the-art reasoning performance. Our work offers both a theoretical foundation and a practical solution for principled inference-time scaling, addressing a critical gap in the efficient deployment of LLMs for complex reasoning. 

---
# Sheaf-Based Decentralized Multimodal Learning for Next-Generation Wireless Communication Systems 

**Authors**: Abdulmomen Ghalkha, Zhuojun Tian, Chaouki Ben Issaid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2506.22374)  

**Abstract**: In large-scale communication systems, increasingly complex scenarios require more intelligent collaboration among edge devices collecting various multimodal sensory data to achieve a more comprehensive understanding of the environment and improve decision-making accuracy. However, conventional federated learning (FL) algorithms typically consider unimodal datasets, require identical model architectures, and fail to leverage the rich information embedded in multimodal data, limiting their applicability to real-world scenarios with diverse modalities and varying client capabilities. To address this issue, we propose Sheaf-DMFL, a novel decentralized multimodal learning framework leveraging sheaf theory to enhance collaboration among devices with diverse modalities. Specifically, each client has a set of local feature encoders for its different modalities, whose outputs are concatenated before passing through a task-specific layer. While encoders for the same modality are trained collaboratively across clients, we capture the intrinsic correlations among clients' task-specific layers using a sheaf-based structure. To further enhance learning capability, we propose an enhanced algorithm named Sheaf-DMFL-Att, which tailors the attention mechanism within each client to capture correlations among different modalities. A rigorous convergence analysis of Sheaf-DMFL-Att is provided, establishing its theoretical guarantees. Extensive simulations are conducted on real-world link blockage prediction and mmWave beamforming scenarios, demonstrate the superiority of the proposed algorithms in such heterogeneous wireless communication systems. 

---
# From Ground to Air: Noise Robustness in Vision Transformers and CNNs for Event-Based Vehicle Classification with Potential UAV Applications 

**Authors**: Nouf Almesafri, Hector Figueiredo, Miguel Arana-Catania  

**Link**: [PDF](https://arxiv.org/pdf/2506.22360)  

**Abstract**: This study investigates the performance of the two most relevant computer vision deep learning architectures, Convolutional Neural Network and Vision Transformer, for event-based cameras. These cameras capture scene changes, unlike traditional frame-based cameras with capture static images, and are particularly suited for dynamic environments such as UAVs and autonomous vehicles. The deep learning models studied in this work are ResNet34 and ViT B16, fine-tuned on the GEN1 event-based dataset. The research evaluates and compares these models under both standard conditions and in the presence of simulated noise. Initial evaluations on the clean GEN1 dataset reveal that ResNet34 and ViT B16 achieve accuracies of 88% and 86%, respectively, with ResNet34 showing a slight advantage in classification accuracy. However, the ViT B16 model demonstrates notable robustness, particularly given its pre-training on a smaller dataset. Although this study focuses on ground-based vehicle classification, the methodologies and findings hold significant promise for adaptation to UAV contexts, including aerial object classification and event-based vision systems for aviation-related tasks. 

---
# Concept-Level AI for Telecom: Moving Beyond Large Language Models 

**Authors**: Viswanath Kumarskandpriya, Abdulhalim Dandoush, Abbas Bradai, Ali Belgacem  

**Link**: [PDF](https://arxiv.org/pdf/2506.22359)  

**Abstract**: The telecommunications and networking domain stands at the precipice of a transformative era, driven by the necessity to manage increasingly complex, hierarchical, multi administrative domains (i.e., several operators on the same path) and multilingual systems. Recent research has demonstrated that Large Language Models (LLMs), with their exceptional general-purpose text analysis and code generation capabilities, can be effectively applied to certain telecom problems (e.g., auto-configuration of data plan to meet certain application requirements). However, due to their inherent token-by-token processing and limited capacity for maintaining extended context, LLMs struggle to fulfill telecom-specific requirements such as cross-layer dependency cascades (i.e., over OSI), temporal-spatial fault correlation, and real-time distributed coordination. In contrast, Large Concept Models (LCMs), which reason at the abstraction level of semantic concepts rather than individual lexical tokens, offer a fundamentally superior approach for addressing these telecom challenges. By employing hyperbolic latent spaces for hierarchical representation and encapsulating complex multi-layered network interactions within concise concept embeddings, LCMs overcome critical shortcomings of LLMs in terms of memory efficiency, cross-layer correlation, and native multimodal integration. This paper argues that adopting LCMs is not simply an incremental step, but a necessary evolutionary leap toward achieving robust and effective AI-driven telecom management. 

---
# A Framework for Multi-source Privacy Preserving Epidemic Analysis 

**Authors**: Zihan Guan, Zhiyuan Zhao, Fengwei Tian, Dung Nguyen, Payel Bhattacharjee, Ravi Tandon, B. Aditya Prakash, Anil Vullikanti  

**Link**: [PDF](https://arxiv.org/pdf/2506.22342)  

**Abstract**: It is now well understood that diverse datasets provide a lot of value in key epidemiology and public health analyses, such as forecasting and nowcasting, development of epidemic models, evaluation and design of interventions and resource allocation. Some of these datasets are often sensitive, and need adequate privacy protections. There are many models of privacy, but Differential Privacy (DP) has become a de facto standard because of its strong guarantees, without making models about adversaries. In this paper, we develop a framework the integrates deep learning and epidemic models to simultaneously perform epidemic forecasting and learning a mechanistic model of epidemic spread, while incorporating multiple datasets for these analyses, including some with DP guarantees. We demonstrate our framework using a realistic but synthetic financial dataset with DP; such a dataset has not been used in such epidemic analyses. We show that this dataset provides significant value in forecasting and learning an epidemic model, even when used with DP guarantees. 

---
# A Deep Learning framework for building damage assessment using VHR SAR and geospatial data: demonstration on the 2023 Turkiye Earthquake 

**Authors**: Luigi Russo, Deodato Tapete, Silvia Liberata Ullo, Paolo Gamba  

**Link**: [PDF](https://arxiv.org/pdf/2506.22338)  

**Abstract**: Building damage identification shortly after a disaster is crucial for guiding emergency response and recovery efforts. Although optical satellite imagery is commonly used for disaster mapping, its effectiveness is often hampered by cloud cover or the absence of pre-event acquisitions. To overcome these challenges, we introduce a novel multimodal deep learning (DL) framework for detecting building damage using single-date very high resolution (VHR) Synthetic Aperture Radar (SAR) imagery from the Italian Space Agency (ASI) COSMO SkyMed (CSK) constellation, complemented by auxiliary geospatial data. Our method integrates SAR image patches, OpenStreetMap (OSM) building footprints, digital surface model (DSM) data, and structural and exposure attributes from the Global Earthquake Model (GEM) to improve detection accuracy and contextual interpretation. Unlike existing approaches that depend on pre and post event imagery, our model utilizes only post event data, facilitating rapid deployment in critical scenarios. The framework effectiveness is demonstrated using a new dataset from the 2023 earthquake in Turkey, covering multiple cities with diverse urban settings. Results highlight that incorporating geospatial features significantly enhances detection performance and generalizability to previously unseen areas. By combining SAR imagery with detailed vulnerability and exposure information, our approach provides reliable and rapid building damage assessments without the dependency from available pre-event data. Moreover, the automated and scalable data generation process ensures the framework's applicability across diverse disaster-affected regions, underscoring its potential to support effective disaster management and recovery efforts. Code and data will be made available upon acceptance of the paper. 

---
# Less Greedy Equivalence Search 

**Authors**: Adiba Ejaz, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2506.22331)  

**Abstract**: Greedy Equivalence Search (GES) is a classic score-based algorithm for causal discovery from observational data. In the sample limit, it recovers the Markov equivalence class of graphs that describe the data. Still, it faces two challenges in practice: computational cost and finite-sample accuracy. In this paper, we develop Less Greedy Equivalence Search (LGES), a variant of GES that retains its theoretical guarantees while partially addressing these limitations. LGES modifies the greedy step: rather than always applying the highest-scoring insertion, it avoids edge insertions between variables for which the score implies some conditional independence. This more targeted search yields up to a \(10\)-fold speed-up and a substantial reduction in structural error relative to GES. Moreover, LGES can guide the search using prior assumptions, while correcting these assumptions when contradicted by the data. Finally, LGES can exploit interventional data to refine the learned observational equivalence class. We prove that LGES recovers the true equivalence class in the sample limit from observational and interventional data, even with misspecified prior assumptions. Experiments demonstrate that LGES outperforms GES and other baselines in speed, accuracy, and robustness to misspecified assumptions. Our code is available at this https URL. 

---
# A Practical Approach to Power Saving in Hearables Using Sub-Nyquist Sampling with Bandwidth Extension 

**Authors**: Tarikul Islam Tamiti, Anomadarshi Barua  

**Link**: [PDF](https://arxiv.org/pdf/2506.22321)  

**Abstract**: Hearables are wearable computers that are worn on the ear. Bone conduction microphones (BCMs) are used with air conduction microphones (ACMs) in hearables as a supporting modality for multimodal speech enhancement (SE) in noisy conditions. However, existing works don't consider the following practical aspects for low-power implementations on hearables: (i) They do not explore how lowering the sampling frequencies and bit resolutions in analog-to-digital converters (ADCs) of hearables jointly impact low-power processing and multimodal SE in terms of speech quality and intelligibility. (ii) They don't discuss how GAN-like audio quality can be achieved without using actual GAN discriminators. And (iii) They don't process signals from ACMs/BCMs at sub-Nyquist sampling rate because, in their frameworks, they lack a wideband reconstruction methodology from their narrowband parts. We propose SUBARU (\textbf{Sub}-Nyquist \textbf{A}udio \textbf{R}esolution \textbf{U}psampling), which achieves the following: SUBARU (i) intentionally uses sub-Nyquist sampling and low bit resolution in ADCs, achieving a 3.31x reduction in power consumption; (ii) introduces novel multi-scale and multi-period virtual discriminators, which achieve GAN-like audio quality without using GANs' adversarial training; and (iii) achieves streaming operations on mobile platforms and SE in in-the-wild noisy conditions with an inference time of 1.74ms and a memory footprint of less than 13.77MB. 

---
# CoATA: Effective Co-Augmentation of Topology and Attribute for Graph Neural Networks 

**Authors**: Tao Liu, Longlong Lin, Yunfeng Yu, Xi Ou, Youan Zhang, Zhiqiu Ye, Tao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.22299)  

**Abstract**: Graph Neural Networks (GNNs) have garnered substantial attention due to their remarkable capability in learning graph representations. However, real-world graphs often exhibit substantial noise and incompleteness, which severely degrades the performance of GNNs. Existing methods typically address this issue through single-dimensional augmentation, focusing either on refining topology structures or perturbing node attributes, thereby overlooking the deeper interplays between the two. To bridge this gap, this paper presents CoATA, a dual-channel GNN framework specifically designed for the Co-Augmentation of Topology and Attribute. Specifically, CoATA first propagates structural signals to enrich and denoise node attributes. Then, it projects the enhanced attribute space into a node-attribute bipartite graph for further refinement or reconstruction of the underlying structure. Subsequently, CoATA introduces contrastive learning, leveraging prototype alignment and consistency constraints, to facilitate mutual corrections between the augmented and original graphs. Finally, extensive experiments on seven benchmark datasets demonstrate that the proposed CoATA outperforms eleven state-of-the-art baseline methods, showcasing its effectiveness in capturing the synergistic relationship between topology and attributes. 

---
# RoomCraft: Controllable and Complete 3D Indoor Scene Generation 

**Authors**: Mengqi Zhou, Xipeng Wang, Yuxi Wang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22291)  

**Abstract**: Generating realistic 3D indoor scenes from user inputs remains a challenging problem in computer vision and graphics, requiring careful balance of geometric consistency, spatial relationships, and visual realism. While neural generation methods often produce repetitive elements due to limited global spatial reasoning, procedural approaches can leverage constraints for controllable generation but struggle with multi-constraint scenarios. When constraints become numerous, object collisions frequently occur, forcing the removal of furniture items and compromising layout completeness.
To address these limitations, we propose RoomCraft, a multi-stage pipeline that converts real images, sketches, or text descriptions into coherent 3D indoor scenes. Our approach combines a scene generation pipeline with a constraint-driven optimization framework. The pipeline first extracts high-level scene information from user inputs and organizes it into a structured format containing room type, furniture items, and spatial relations. It then constructs a spatial relationship network to represent furniture arrangements and generates an optimized placement sequence using a heuristic-based depth-first search (HDFS) algorithm to ensure layout coherence. To handle complex multi-constraint scenarios, we introduce a unified constraint representation that processes both formal specifications and natural language inputs, enabling flexible constraint-oriented adjustments through a comprehensive action space design. Additionally, we propose a Conflict-Aware Positioning Strategy (CAPS) that dynamically adjusts placement weights to minimize furniture collisions and ensure layout completeness.
Extensive experiments demonstrate that RoomCraft significantly outperforms existing methods in generating realistic, semantically coherent, and visually appealing room layouts across diverse input modalities. 

---
# Projected Compression: Trainable Projection for Efficient Transformer Compression 

**Authors**: Maciej Stefaniak, Michał Krutul, Jan Małaśnicki, Maciej Pióro, Jakub Krajewski, Sebastian Jaszczur, Marek Cygan, Kamil Adamczewski, Jan Ludziejewski  

**Link**: [PDF](https://arxiv.org/pdf/2506.22255)  

**Abstract**: Large language models have steadily increased in size to achieve improved performance; however, this growth has also led to greater inference time and computational demands. Consequently, there is rising interest in model size reduction methods. To address this issue, we propose Projected Compression, a novel model compression technique, that reduces model weights by utilizing projection modules. Specifically, we first train additional trainable projections weights and preserve access to all the original model parameters. Subsequently, these projections are merged into a lower-dimensional product matrix, resulting in a reduced-size standard Transformer-based model. Unlike alternative approaches that require additional computational overhead, our method matches the base model's per-token computation step in FLOPs. Experimental results show that Projected Compression outperforms the comparable hard pruning and retraining approach on higher quality models. Moreover, the performance margin scales well with the number of tokens. 

---
# Adapting University Policies for Generative AI: Opportunities, Challenges, and Policy Solutions in Higher Education 

**Authors**: Russell Beale  

**Link**: [PDF](https://arxiv.org/pdf/2506.22231)  

**Abstract**: The rapid proliferation of generative artificial intelligence (AI) tools - especially large language models (LLMs) such as ChatGPT - has ushered in a transformative era in higher education. Universities in developed regions are increasingly integrating these technologies into research, teaching, and assessment. On one hand, LLMs can enhance productivity by streamlining literature reviews, facilitating idea generation, assisting with coding and data analysis, and even supporting grant proposal drafting. On the other hand, their use raises significant concerns regarding academic integrity, ethical boundaries, and equitable access. Recent empirical studies indicate that nearly 47% of students use LLMs in their coursework - with 39% using them for exam questions and 7% for entire assignments - while detection tools currently achieve around 88% accuracy, leaving a 12% error margin. This article critically examines the opportunities offered by generative AI, explores the multifaceted challenges it poses, and outlines robust policy solutions. Emphasis is placed on redesigning assessments to be AI-resilient, enhancing staff and student training, implementing multi-layered enforcement mechanisms, and defining acceptable use. By synthesizing data from recent research and case studies, the article argues that proactive policy adaptation is imperative to harness AI's potential while safeguarding the core values of academic integrity and equity. 

---
# EFRame: Deeper Reasoning via Exploration-Filtering-Replay Reinforcement Learning Framework 

**Authors**: Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Yue Wang, Yuzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22200)  

**Abstract**: Recent advances in reinforcement learning (RL) have significantly enhanced the reasoning capabilities of large language models (LLMs). Group Relative Policy Optimization (GRPO), an efficient variant of PPO that lowers RL's computational cost, still faces limited exploration, low sample efficiency and instability, constraining its performance on complex reasoning tasks. To address these limitations, we introduce EFRame, an Exploration-Filtering-Replay framework that systematically augments GRPO along three critical dimensions. EFRame performs additional rollouts to explore high-quality trajectories, applies online filtering to eliminate low-quality samples that introduce noise and variance, and leverages experience replay to repeatedly exploit rare but informative samples. EFRame establishes a complete and stable learning cycle, guiding the model through a structured transition from exploration to convergence. Our experiments across a variety of reasoning benchmarks demonstrate that EFRame not only improves the robustness and efficiency of training, but also enables access to deeper reasoning capabilities that remain unattainable under vanilla GRPO. Furthermore, EFRame enables a more fine-grained categorization of training samples, allowing for a deeper analysis of how different types of samples contribute to the learning process in RL. Our code is available at this https URL. 

---
# Autonomic Microservice Management via Agentic AI and MAPE-K Integration 

**Authors**: Matteo Esposito, Alexander Bakhtin, Noman Ahmad, Mikel Robredo, Ruoyu Su, Valentina Lenarduzzi, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2506.22185)  

**Abstract**: While microservices are revolutionizing cloud computing by offering unparalleled scalability and independent deployment, their decentralized nature poses significant security and management challenges that can threaten system stability. We propose a framework based on MAPE-K, which leverages agentic AI, for autonomous anomaly detection and remediation to address the daunting task of highly distributed system management. Our framework offers practical, industry-ready solutions for maintaining robust and secure microservices. Practitioners and researchers can customize the framework to enhance system stability, reduce downtime, and monitor broader system quality attributes such as system performance level, resilience, security, and anomaly management, among others. 

---
# Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition 

**Authors**: Wenhan Wu, Zhishuai Guo, Chen Chen, Hongfei Xue, Aidong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22179)  

**Abstract**: Zero-shot skeleton-based action recognition aims to develop models capable of identifying actions beyond the categories encountered during training. Previous approaches have primarily focused on aligning visual and semantic representations but often overlooked the importance of fine-grained action patterns in the semantic space (e.g., the hand movements in drinking water and brushing teeth). To address these limitations, we propose a Frequency-Semantic Enhanced Variational Autoencoder (FS-VAE) to explore the skeleton semantic representation learning with frequency decomposition. FS-VAE consists of three key components: 1) a frequency-based enhancement module with high- and low-frequency adjustments to enrich the skeletal semantics learning and improve the robustness of zero-shot action recognition; 2) a semantic-based action description with multilevel alignment to capture both local details and global correspondence, effectively bridging the semantic gap and compensating for the inherent loss of information in skeleton sequences; 3) a calibrated cross-alignment loss that enables valid skeleton-text pairs to counterbalance ambiguous ones, mitigating discrepancies and ambiguities in skeleton and text features, thereby ensuring robust alignment. Evaluations on the benchmarks demonstrate the effectiveness of our approach, validating that frequency-enhanced semantic features enable robust differentiation of visually and semantically similar action clusters, improving zero-shot action recognition. 

---
# Visual Structures Helps Visual Reasoning: Addressing the Binding Problem in VLMs 

**Authors**: Amirmohammad Izadi, Mohammad Ali Banayeeanzade, Fatemeh Askari, Ali Rahimiakbar, Mohammad Mahdi Vahedi, Hosein Hasani, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2506.22146)  

**Abstract**: Despite progress in Vision-Language Models (VLMs), their capacity for visual reasoning is often limited by the \textit{binding problem}: the failure to reliably associate perceptual features with their correct visual referents. This limitation underlies persistent errors in tasks such as counting, visual search, scene description, and spatial relationship understanding. A key factor is that current VLMs process visual features largely in parallel, lacking mechanisms for spatially grounded, serial attention. This paper introduces a simple yet effective intervention: augmenting visual inputs with low-level spatial structures (e.g., horizontal lines) and pairing this with a textual prompt that encourages sequential, spatially-aware parsing. We empirically demonstrate substantial performance improvements across core visual reasoning tasks. Specifically, our method improves GPT-4o visual search accuracy by 25.00%, increases counting accuracy by 26.83%, reduces edit distance error in scene description by 0.32, and enhances performance on spatial relationship tasks by 9.50% on a a 2D synthetic dataset. Furthermore, we find that the visual modification is essential for these gains; purely textual strategies, including Chain-of-Thought prompting, are insufficient and can even degrade performance. Our method enhances binding only with a single-query inference, underscoring the importance of visual input design over purely linguistically-based approaches. These findings suggest that low-level visual structuring is a powerful and underexplored direction for improving compositional visual reasoning and could serve as a general strategy for enhancing VLM performance on spatially grounded tasks. 

---
# Learning to Solve Multi-Objective Routing Problems on Multigraphs 

**Authors**: Filip Rydin, Attila Lischka, Jiaming Wu, Morteza Haghir Chehreghani, Balázs Kulcsár  

**Link**: [PDF](https://arxiv.org/pdf/2506.22095)  

**Abstract**: Learning-based methods for routing have gained significant attention in recent years, both in single-objective and multi-objective contexts. However, the multigraph setting, where multiple paths with distinct attributes can exist between destinations, has largely been overlooked, despite its high practical relevancy. In this paper, we introduce two neural approaches to address multi-objective routing on multigraphs. Our first approach works directly on the multigraph, by autoregressively selecting edges until a tour is completed. On the other hand, our second model first prunes the multigraph into a simple graph and then builds routes. We validate both models experimentally and find that they demonstrate strong performance across a variety of problems, including the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP). 

---
# Transformers are Graph Neural Networks 

**Authors**: Chaitanya K. Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.22084)  

**Abstract**: We establish connections between the Transformer architecture, originally introduced for natural language processing, and Graph Neural Networks (GNNs) for representation learning on graphs. We show how Transformers can be viewed as message passing GNNs operating on fully connected graphs of tokens, where the self-attention mechanism capture the relative importance of all tokens w.r.t. each-other, and positional encodings provide hints about sequential ordering or structure. Thus, Transformers are expressive set processing networks that learn relationships among input elements without being constrained by apriori graphs. Despite this mathematical connection to GNNs, Transformers are implemented via dense matrix operations that are significantly more efficient on modern hardware than sparse message passing. This leads to the perspective that Transformers are GNNs currently winning the hardware lottery. 

---
# UniCA: Adapting Time Series Foundation Model to General Covariate-Aware Forecasting 

**Authors**: Lu Han, Yu Liu, Qiwen Deng, Jian Jiang, Yinbo Sun, Zhe Yu, Binfeng Wang, Xingyu Lu, Lintao Ma, Han-Jia Ye, De-Chuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.22039)  

**Abstract**: Time Series Foundation Models (TSFMs) have achieved remarkable success through large-scale pretraining. However, their design primarily targets real-valued series, limiting their ability to handle general forecasting tasks involving diverse and often heterogeneous covariates--such as categorical variables and multimodal data (e.g., images, text)--which are typically task-specific and difficult to leverage during pretraining. To address this gap, we propose Unified Covariate Adaptation (UniCA), a framework to bridge TSFMs with general covariate-aware forecasting. UniCA first performs covariate homogenization to transform heterogeneous covariates into high-level homogeneous series representations and then fuses them via a unified attention-based fusion mechanism. UniCA is compatible and universal for adaptation with both homogeneous and heterogeneous covariates, incorporating extra covariate information while preserving the generalization ability of this http URL experiments on multiple unimodal and multimodal covariate-aware forecasting benchmarks demonstrate the superiority of UniCA, highlighting the promise of covariate-aware TSFM adaptation in real-world forecasting scenarios. Codes are released on this https URL. 

---
# Literature-Grounded Novelty Assessment of Scientific Ideas 

**Authors**: Simra Shahid, Marissa Radensky, Raymond Fok, Pao Siangliulue, Daniel S. Weld, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2506.22026)  

**Abstract**: Automated scientific idea generation systems have made remarkable progress, yet the automatic evaluation of idea novelty remains a critical and underexplored challenge. Manual evaluation of novelty through literature review is labor-intensive, prone to error due to subjectivity, and impractical at scale. To address these issues, we propose the Idea Novelty Checker, an LLM-based retrieval-augmented generation (RAG) framework that leverages a two-stage retrieve-then-rerank approach. The Idea Novelty Checker first collects a broad set of relevant papers using keyword and snippet-based retrieval, then refines this collection through embedding-based filtering followed by facet-based LLM re-ranking. It incorporates expert-labeled examples to guide the system in comparing papers for novelty evaluation and in generating literature-grounded reasoning. Our extensive experiments demonstrate that our novelty checker achieves approximately 13% higher agreement than existing approaches. Ablation studies further showcases the importance of the facet-based re-ranker in identifying the most relevant literature for novelty evaluation. 

---
# TROFI: Trajectory-Ranked Offline Inverse Reinforcement Learning 

**Authors**: Alessandro Sestini, Joakim Bergdahl, Konrad Tollmar, Andrew D. Bagdanov, Linus Gisslén  

**Link**: [PDF](https://arxiv.org/pdf/2506.22008)  

**Abstract**: In offline reinforcement learning, agents are trained using only a fixed set of stored transitions derived from a source policy. However, this requires that the dataset be labeled by a reward function. In applied settings such as video game development, the availability of the reward function is not always guaranteed. This paper proposes Trajectory-Ranked OFfline Inverse reinforcement learning (TROFI), a novel approach to effectively learn a policy offline without a pre-defined reward function. TROFI first learns a reward function from human preferences, which it then uses to label the original dataset making it usable for training the policy. In contrast to other approaches, our method does not require optimal trajectories. Through experiments on the D4RL benchmark we demonstrate that TROFI consistently outperforms baselines and performs comparably to using the ground truth reward to learn policies. Additionally, we validate the efficacy of our method in a 3D game environment. Our studies of the reward model highlight the importance of the reward function in this setting: we show that to ensure the alignment of a value function to the actual future discounted reward, it is fundamental to have a well-engineered and easy-to-learn reward function. 

---
# Binned semiparametric Bayesian networks 

**Authors**: Rafael Sojo, Javier Díaz-Rozo, Concha Bielza, Pedro Larrañaga  

**Link**: [PDF](https://arxiv.org/pdf/2506.21997)  

**Abstract**: This paper introduces a new type of probabilistic semiparametric model that takes advantage of data binning to reduce the computational cost of kernel density estimation in nonparametric distributions. Two new conditional probability distributions are developed for the new binned semiparametric Bayesian networks, the sparse binned kernel density estimation and the Fourier kernel density estimation. These two probability distributions address the curse of dimensionality, which typically impacts binned models, by using sparse tensors and restricting the number of parent nodes in conditional probability calculations. To evaluate the proposal, we perform a complexity analysis and conduct several comparative experiments using synthetic data and datasets from the UCI Machine Learning repository. The experiments include different binning rules, parent restrictions, grid sizes, and number of instances to get a holistic view of the model's behavior. As a result, our binned semiparametric Bayesian networks achieve structural learning and log-likelihood estimations with no statistically significant differences compared to the semiparametric Bayesian networks, but at a much higher speed. Thus, the new binned semiparametric Bayesian networks prove to be a reliable and more efficient alternative to their non-binned counterparts. 

---
# Analyzing and Fine-Tuning Whisper Models for Multilingual Pilot Speech Transcription in the Cockpit 

**Authors**: Kartheek Kumar Reddy Nareddy, Sarah Ternus, Julia Niebling  

**Link**: [PDF](https://arxiv.org/pdf/2506.21990)  

**Abstract**: The developments in transformer encoder-decoder architectures have led to significant breakthroughs in machine translation, Automatic Speech Recognition (ASR), and instruction-based chat machines, among other applications. The pre-trained models were trained on vast amounts of generic data over a few epochs (fewer than five in most cases), resulting in their strong generalization capabilities. Nevertheless, the performance of these models does suffer when applied to niche domains like transcribing pilot speech in the cockpit, which involves a lot of specific vocabulary and multilingual conversations. This paper investigates and improves the transcription accuracy of cockpit conversations with Whisper models. We have collected around 85 minutes of cockpit simulator recordings and 130 minutes of interview recordings with pilots and manually labeled them. The speakers are middle aged men speaking both German and English. To improve the accuracy of transcriptions, we propose multiple normalization schemes to refine the transcripts and improve Word Error Rate (WER). We then employ fine-tuning to enhance ASR performance, utilizing performance-efficient fine-tuning with Low-Rank Adaptation (LoRA). Hereby, WER decreased from 68.49 \% (pretrained whisper Large model without normalization baseline) to 26.26\% (finetuned whisper Large model with the proposed normalization scheme). 

---
# SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model 

**Authors**: Shuhan Tan, John Lambert, Hong Jeon, Sakshum Kulshrestha, Yijing Bai, Jing Luo, Dragomir Anguelov, Mingxing Tan, Chiyu Max Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21976)  

**Abstract**: The goal of traffic simulation is to augment a potentially limited amount of manually-driven miles that is available for testing and validation, with a much larger amount of simulated synthetic miles. The culmination of this vision would be a generative simulated city, where given a map of the city and an autonomous vehicle (AV) software stack, the simulator can seamlessly simulate the trip from point A to point B by populating the city around the AV and controlling all aspects of the scene, from animating the dynamic agents (e.g., vehicles, pedestrians) to controlling the traffic light states. We refer to this vision as CitySim, which requires an agglomeration of simulation technologies: scene generation to populate the initial scene, agent behavior modeling to animate the scene, occlusion reasoning, dynamic scene generation to seamlessly spawn and remove agents, and environment simulation for factors such as traffic lights. While some key technologies have been separately studied in various works, others such as dynamic scene generation and environment simulation have received less attention in the research community. We propose SceneDiffuser++, the first end-to-end generative world model trained on a single loss function capable of point A-to-B simulation on a city scale integrating all the requirements above. We demonstrate the city-scale traffic simulation capability of SceneDiffuser++ and study its superior realism under long simulation conditions. We evaluate the simulation quality on an augmented version of the Waymo Open Motion Dataset (WOMD) with larger map regions to support trip-level simulation. 

---
# Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses 

**Authors**: Mohamed Ahmed, Mohamed Abdelmouty, Mingyu Kim, Gunvanth Kandula, Alex Park, James C. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2506.21972)  

**Abstract**: The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries. 

---
# Using Large Language Models to Suggest Informative Prior Distributions in Bayesian Statistics 

**Authors**: Michael A. Riegler, Kristoffer Herland Hellton, Vajira Thambawita, Hugo L. Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.21964)  

**Abstract**: Selecting prior distributions in Bayesian statistics is challenging, resource-intensive, and subjective. We analyze using large-language models (LLMs) to suggest suitable, knowledge-based informative priors. We developed an extensive prompt asking LLMs not only to suggest priors but also to verify and reflect on their choices.
We evaluated Claude Opus, Gemini 2.5 Pro, and ChatGPT-4o-mini on two real datasets: heart disease risk and concrete strength. All LLMs correctly identified the direction for all associations (e.g., that heart disease risk is higher for males). The quality of suggested priors was measured by their Kullback-Leibler divergence from the maximum likelihood estimator's distribution.
The LLMs suggested both moderately and weakly informative priors. The moderate priors were often overconfident, resulting in distributions misaligned with the data. In our experiments, Claude and Gemini provided better priors than ChatGPT. For weakly informative priors, a key performance difference emerged: ChatGPT and Gemini defaulted to an "unnecessarily vague" mean of 0, while Claude did not, demonstrating a significant advantage.
The ability of LLMs to identify correct associations shows their great potential as an efficient, objective method for developing informative priors. However, the primary challenge remains in calibrating the width of these priors to avoid over- and under-confidence. 

---
# SDRNET: Stacked Deep Residual Network for Accurate Semantic Segmentation of Fine-Resolution Remotely Sensed Images 

**Authors**: Naftaly Wambugu, Ruisheng Wang, Bo Guo, Tianshu Yu, Sheng Xu, Mohammed Elhassan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21945)  

**Abstract**: Land cover maps generated from semantic segmentation of high-resolution remotely sensed images have drawn mucon in the photogrammetry and remote sensing research community. Currently, massive fine-resolution remotely sensed (FRRS) images acquired by improving sensing and imaging technologies become available. However, accurate semantic segmentation of such FRRS images is greatly affected by substantial class disparities, the invisibility of key ground objects due to occlusion, and object size variation. Despite the extraordinary potential in deep convolutional neural networks (DCNNs) in image feature learning and representation, extracting sufficient features from FRRS images for accurate semantic segmentation is still challenging. These challenges demand the deep learning models to learn robust features and generate sufficient feature descriptors. Specifically, learning multi-contextual features to guarantee adequate coverage of varied object sizes from the ground scene and harnessing global-local contexts to overcome class disparities challenge even profound networks. Deeper networks significantly lose spatial details due to gradual downsampling processes resulting in poor segmentation results and coarse boundaries. This article presents a stacked deep residual network (SDRNet) for semantic segmentation from FRRS images. The proposed framework utilizes two stacked encoder-decoder networks to harness long-range semantics yet preserve spatial information and dilated residual blocks (DRB) between each encoder and decoder network to capture sufficient global dependencies thus improving segmentation performance. Our experimental results obtained using the ISPRS Vaihingen and Potsdam datasets demonstrate that the SDRNet performs effectively and competitively against current DCNNs in semantic segmentation. 

---
# ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation 

**Authors**: Reza Yousefi Maragheh, Pratheek Vadla, Priyank Gupta, Kai Zhao, Aysenur Inan, Kehui Yao, Jianpeng Xu, Praveen Kanumala, Jason Cho, Sushant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21931)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown promise in enhancing recommendation systems by incorporating external context into large language model prompts. However, existing RAG-based approaches often rely on static retrieval heuristics and fail to capture nuanced user preferences in dynamic recommendation scenarios. In this work, we introduce ARAG, an Agentic Retrieval-Augmented Generation framework for Personalized Recommendation, which integrates a multi-agent collaboration mechanism into the RAG pipeline. To better understand the long-term and session behavior of the user, ARAG leverages four specialized LLM-based agents: a User Understanding Agent that summarizes user preferences from long-term and session contexts, a Natural Language Inference (NLI) Agent that evaluates semantic alignment between candidate items retrieved by RAG and inferred intent, a context summary agent that summarizes the findings of NLI agent, and an Item Ranker Agent that generates a ranked list of recommendations based on contextual fit. We evaluate ARAG accross three datasets. Experimental results demonstrate that ARAG significantly outperforms standard RAG and recency-based baselines, achieving up to 42.1% improvement in NDCG@5 and 35.5% in Hit@5. We also, conduct an ablation study to analyse the effect by different components of ARAG. Our findings highlight the effectiveness of integrating agentic reasoning into retrieval-augmented recommendation and provide new directions for LLM-based personalization. 

---
# SODA: Out-of-Distribution Detection in Domain-Shifted Point Clouds via Neighborhood Propagation 

**Authors**: Adam Goodge, Xun Xu, Bryan Hooi, Wee Siong Ng, Jingyi Liao, Yongyi Su, Xulei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21892)  

**Abstract**: As point cloud data increases in prevalence in a variety of applications, the ability to detect out-of-distribution (OOD) point cloud objects becomes critical for ensuring model safety and reliability. However, this problem remains under-explored in existing research. Inspired by success in the image domain, we propose to exploit advances in 3D vision-language models (3D VLMs) for OOD detection in point cloud objects. However, a major challenge is that point cloud datasets used to pre-train 3D VLMs are drastically smaller in size and object diversity than their image-based counterparts. Critically, they often contain exclusively computer-designed synthetic objects. This leads to a substantial domain shift when the model is transferred to practical tasks involving real objects scanned from the physical environment. In this paper, our empirical experiments show that synthetic-to-real domain shift significantly degrades the alignment of point cloud with their associated text embeddings in the 3D VLM latent space, hindering downstream performance. To address this, we propose a novel methodology called SODA which improves the detection of OOD point clouds through a neighborhood-based score propagation scheme. SODA is inference-based, requires no additional model training, and achieves state-of-the-art performance over existing approaches across datasets and problem settings. 

---
# UnMix-NeRF: Spectral Unmixing Meets Neural Radiance Fields 

**Authors**: Fabian Perez, Sara Rojas, Carlos Hinojosa, Hoover Rueda-Chacón, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.21884)  

**Abstract**: Neural Radiance Field (NeRF)-based segmentation methods focus on object semantics and rely solely on RGB data, lacking intrinsic material properties. This limitation restricts accurate material perception, which is crucial for robotics, augmented reality, simulation, and other applications. We introduce UnMix-NeRF, a framework that integrates spectral unmixing into NeRF, enabling joint hyperspectral novel view synthesis and unsupervised material segmentation. Our method models spectral reflectance via diffuse and specular components, where a learned dictionary of global endmembers represents pure material signatures, and per-point abundances capture their distribution. For material segmentation, we use spectral signature predictions along learned endmembers, allowing unsupervised material clustering. Additionally, UnMix-NeRF enables scene editing by modifying learned endmember dictionaries for flexible material-based appearance manipulation. Extensive experiments validate our approach, demonstrating superior spectral reconstruction and material segmentation to existing methods. Project page: this https URL. 

---
# Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation 

**Authors**: Qiyue Gao, Xinyu Pi, Kevin Liu, Junrong Chen, Ruolan Yang, Xinqi Huang, Xinyu Fang, Lu Sun, Gautham Kishore, Bo Ai, Stone Tao, Mengyang Liu, Jiaxi Yang, Chao-Jung Lai, Chuanyang Jin, Jiannan Xiang, Benhao Huang, Zeming Chen, David Danks, Hao Su, Tianmin Shu, Ziqiao Ma, Lianhui Qin, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21876)  

**Abstract**: Internal world models (WMs) enable agents to understand the world's state and predict transitions, serving as the basis for advanced deliberative reasoning. Recent large Vision-Language Models (VLMs), such as OpenAI o3, GPT-4o and Gemini, exhibit potential as general-purpose WMs. While the latest studies have evaluated and shown limitations in specific capabilities such as visual understanding, a systematic evaluation of VLMs' fundamental WM abilities remains absent. Drawing on comparative psychology and cognitive science, we propose a two-stage framework that assesses Perception (visual, spatial, temporal, quantitative, and motion) and Prediction (mechanistic simulation, transitive inference, compositional inference) to provide an atomic evaluation of VLMs as WMs. Guided by this framework, we introduce WM-ABench, a large-scale benchmark comprising 23 fine-grained evaluation dimensions across 6 diverse simulated environments with controlled counterfactual simulations. Through 660 experiments on 15 latest commercial and open-source VLMs, we find that these models exhibit striking limitations in basic world modeling abilities. For instance, almost all models perform at near-random accuracy when distinguishing motion trajectories. Additionally, they lack disentangled understanding -- e.g., some models tend to believe blue objects move faster than green ones. More rich results and analyses reveal significant gaps between VLMs and human-level world modeling. 

---
# On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling 

**Authors**: Stanley Wu, Ronik Bhaskar, Anna Yoo Jeong Ha, Shawn Shan, Haitao Zheng, Ben Y. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21874)  

**Abstract**: Today's text-to-image generative models are trained on millions of images sourced from the Internet, each paired with a detailed caption produced by Vision-Language Models (VLMs). This part of the training pipeline is critical for supplying the models with large volumes of high-quality image-caption pairs during training. However, recent work suggests that VLMs are vulnerable to stealthy adversarial attacks, where adversarial perturbations are added to images to mislead the VLMs into producing incorrect captions.
In this paper, we explore the feasibility of adversarial mislabeling attacks on VLMs as a mechanism to poisoning training pipelines for text-to-image models. Our experiments demonstrate that VLMs are highly vulnerable to adversarial perturbations, allowing attackers to produce benign-looking images that are consistently miscaptioned by the VLM models. This has the effect of injecting strong "dirty-label" poison samples into the training pipeline for text-to-image models, successfully altering their behavior with a small number of poisoned samples. We find that while potential defenses can be effective, they can be targeted and circumvented by adaptive attackers. This suggests a cat-and-mouse game that is likely to reduce the quality of training data and increase the cost of text-to-image model development. Finally, we demonstrate the real-world effectiveness of these attacks, achieving high attack success (over 73%) even in black-box scenarios against commercial VLMs (Google Vertex AI and Microsoft Azure). 

---
# Grounding-Aware Token Pruning: Recovering from Drastic Performance Drops in Visual Grounding Caused by Pruning 

**Authors**: Tzu-Chun Chien, Chieh-Kai Lin, Shiang-Feng Tsai, Ruei-Chi Lai, Hung-Jen Chen, Min Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.21873)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) have demonstrated strong performance in visual grounding, establishing themselves as a general interface for various vision-language applications. This progress has driven the development of token pruning methods to mitigate the high computational costs associated with processing numerous visual tokens. However, we observe that pruning significantly weakens the model's grounding ability, leading to incorrect predictions and drastic performance degradation. In Referring Expression Comprehension (REC), for instance, pruning causes the accuracy of LLaVA on the RefCOCO validation set to drop from 56.14% to 15.34%. Our analysis identifies misaligned position IDs after pruning as the primary cause of this degradation, as both the order and value of these IDs are crucial for maintaining performance in grounding tasks. To address this issue, we propose Grounding-Aware Token Pruning (GAP), a simple yet effective adjustment to position IDs that recovers REC accuracy back to 51.42%, which is 90% of the original performance in the without pruning setting, all while requiring no additional training, memory, or computational overhead. Applied to models such as Shikra, MiniGPTv2, and the LLaVA series, our method consistently improves performance across various token pruning strategies. 

---
# A Survey of Continual Reinforcement Learning 

**Authors**: Chaofan Pan, Xin Yang, Yanhua Li, Wei Wei, Tianrui Li, Bo An, Jiye Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21872)  

**Abstract**: Reinforcement Learning (RL) is an important machine learning paradigm for solving sequential decision-making problems. Recent years have witnessed remarkable progress in this field due to the rapid development of deep neural networks. However, the success of RL currently relies on extensive training data and computational resources. In addition, RL's limited ability to generalize across tasks restricts its applicability in dynamic and real-world environments. With the arisen of Continual Learning (CL), Continual Reinforcement Learning (CRL) has emerged as a promising research direction to address these limitations by enabling agents to learn continuously, adapt to new tasks, and retain previously acquired knowledge. In this survey, we provide a comprehensive examination of CRL, focusing on its core concepts, challenges, and methodologies. Firstly, we conduct a detailed review of existing works, organizing and analyzing their metrics, tasks, benchmarks, and scenario settings. Secondly, we propose a new taxonomy of CRL methods, categorizing them into four types from the perspective of knowledge storage and/or transfer. Finally, our analysis highlights the unique challenges of CRL and provides practical insights into future directions. 

---
# DeepTalk: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE 

**Authors**: Hang Shao, Heting Gao, Yunhang Shen, Jiawei Chen, Lijiang Li, Zuwei Long, Bo Tong, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.21864)  

**Abstract**: Native multimodal large language models (MLLMs) restructure a single large language model (LLM) into a spoken language model (SLM) capable of both speech and text generation. Compared to modular and aligned MLLMs, native MLLMs preserve richer paralinguistic features such as emotion and prosody, and generate speech responses directly within the backbone LLM rather than using a separate speech decoder. This integration also results in lower response latency and smoother interaction. However, native MLLMs suffer from catastrophic forgetting and performance degradation because the available paired speech-text data is insufficient to support the pretraining of MLLMs compared to the vast amount of text data required to pretrain text LLMs. To address this issue, we propose DeepTalk, a framework for adaptive modality expert learning based on a Mixture of Experts (MoE) architecture. DeepTalk first adaptively distinguishes modality experts according to their modality load within the LLM. Each modality expert then undergoes specialized single-modality training, followed by joint multimodal collaborative training. As a result, DeepTalk incurs only a 5.5% performance drop compared to the original LLM, which is significantly lower than the average performance drop of over 20% typically seen in native MLLMs (such as GLM-4-Voice), and is on par with modular MLLMs. Meanwhile, the end-to-end dialogue latency remains within 0.5 seconds, ensuring a seamless and intelligent speech interaction experience. Code and models are released at this https URL. 

---
# LLaVA-Scissor: Token Compression with Semantic Connected Components for Video LLMs 

**Authors**: Boyuan Sun, Jiaxing Zhao, Xihan Wei, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21862)  

**Abstract**: In this paper, we present LLaVA-Scissor, a training-free token compression strategy designed for video multimodal large language models. Previous methods mostly attempt to compress tokens based on attention scores, but fail to effectively capture all semantic regions and often lead to token redundancy. Differently, we propose to leverage the Semantic Connected Components (SCC) approach that assigns tokens to distinct semantic regions within the token set, ensuring comprehensive semantic coverage. The outcome is a two-step spatio-temporal token compression strategy that utilizes SCC in both spatial and temporal domains. This strategy can effectively compress tokens by representing the entire video with a set of non-overlapping semantic tokens. We conduct extensive evaluations of the token compression capabilities of LLaVA-Scissor across diverse video understanding benchmarks, including video question answering, long video understanding, and comprehensive multi-choices benchmarks. Experimental results show that the proposed LLaVA-Scissor outperforms other token compression methods, achieving superior performance in various video understanding benchmarks, particularly at low token retention ratios. Project page: this https URL. 

---
# SPADE: Spatial Transcriptomics and Pathology Alignment Using a Mixture of Data Experts for an Expressive Latent Space 

**Authors**: Ekaterina Redekop, Mara Pleasure, Zichen Wang, Kimberly Flores, Anthony Sisk, William Speier, Corey W. Arnold  

**Link**: [PDF](https://arxiv.org/pdf/2506.21857)  

**Abstract**: The rapid growth of digital pathology and advances in self-supervised deep learning have enabled the development of foundational models for various pathology tasks across diverse diseases. While multimodal approaches integrating diverse data sources have emerged, a critical gap remains in the comprehensive integration of whole-slide images (WSIs) with spatial transcriptomics (ST), which is crucial for capturing critical molecular heterogeneity beyond standard hematoxylin & eosin (H&E) staining. We introduce SPADE, a foundation model that integrates histopathology with ST data to guide image representation learning within a unified framework, in effect creating an ST-informed latent space. SPADE leverages a mixture-of-data experts technique, where experts, created via two-stage feature-space clustering, use contrastive learning to learn representations of co-registered WSI patches and gene expression profiles. Pre-trained on the comprehensive HEST-1k dataset, SPADE is evaluated on 14 downstream tasks, demonstrating significantly superior few-shot performance compared to baseline models, highlighting the benefits of integrating morphological and molecular information into one latent space. 

---
# The Consistency Hypothesis in Uncertainty Quantification for Large Language Models 

**Authors**: Quan Xiao, Debarun Bhattacharjya, Balaji Ganesan, Radu Marinescu, Katsiaryna Mirylenka, Nhan H Pham, Michael Glass, Junkyu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.21849)  

**Abstract**: Estimating the confidence of large language model (LLM) outputs is essential for real-world applications requiring high user trust. Black-box uncertainty quantification (UQ) methods, relying solely on model API access, have gained popularity due to their practical benefits. In this paper, we examine the implicit assumption behind several UQ methods, which use generation consistency as a proxy for confidence, an idea we formalize as the consistency hypothesis. We introduce three mathematical statements with corresponding statistical tests to capture variations of this hypothesis and metrics to evaluate LLM output conformity across tasks. Our empirical investigation, spanning 8 benchmark datasets and 3 tasks (question answering, text summarization, and text-to-SQL), highlights the prevalence of the hypothesis under different settings. Among the statements, we highlight the `Sim-Any' hypothesis as the most actionable, and demonstrate how it can be leveraged by proposing data-free black-box UQ methods that aggregate similarities between generations for confidence estimation. These approaches can outperform the closest baselines, showcasing the practical value of the empirically observed consistency hypothesis. 

---
# 3Description: An Intuitive Human-AI Collaborative 3D Modeling Approach 

**Authors**: Zhuodi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21845)  

**Abstract**: This paper presents 3Description, an experimental human-AI collaborative approach for intuitive 3D modeling. 3Description aims to address accessibility and usability challenges in traditional 3D modeling by enabling non-professional individuals to co-create 3D models using verbal and gesture descriptions. Through a combination of qualitative research, product analysis, and user testing, 3Description integrates AI technologies such as Natural Language Processing and Computer Vision, powered by OpenAI and MediaPipe. Recognizing the web has wide cross-platform capabilities, 3Description is web-based, allowing users to describe the desired model and subsequently adjust its components using verbal and gestural inputs. In the era of AI and emerging media, 3Description not only contributes to a more inclusive and user-friendly design process, empowering more people to participate in the construction of the future 3D world, but also strives to increase human engagement in co-creation with AI, thereby avoiding undue surrender to technology and preserving human creativity. 

---
# PARSI: Persian Authorship Recognition via Stylometric Integration 

**Authors**: Kourosh Shahnazari, Mohammadali Keshtparvar, Seyed Moein Ayyoubzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.21840)  

**Abstract**: The intricate linguistic, stylistic, and metrical aspects of Persian classical poetry pose a challenge for computational authorship attribution. In this work, we present a versatile framework to determine authorship among 67 prominent poets. We employ a multi-input neural framework consisting of a transformer-based language encoder complemented by features addressing the semantic, stylometric, and metrical dimensions of Persian poetry. Our feature set encompasses 100-dimensional Word2Vec embeddings, seven stylometric measures, and categorical encodings of poetic form and meter. We compiled a vast corpus of 647,653 verses of the Ganjoor digital collection, validating the data through strict preprocessing and author verification while preserving poem-level splitting to prevent overlap. This work employs verse-level classification and majority and weighted voting schemes in evaluation, revealing that weighted voting yields 71% accuracy. We further investigate threshold-based decision filtering, allowing the model to generate highly confident predictions, achieving 97% accuracy at a 0.9 threshold, though at lower coverage. Our work focuses on the integration of deep representational forms with domain-specific features for improved authorship attribution. The results illustrate the potential of our approach for automated classification and the contribution to stylistic analysis, authorship disputes, and general computational literature research. This research will facilitate further research on multilingual author attribution, style shift, and generative modeling of Persian poetry. 

---
# Few-Shot Segmentation of Historical Maps via Linear Probing of Vision Foundation Models 

**Authors**: Rafael Sterzinger, Marco Peer, Robert Sablatnig  

**Link**: [PDF](https://arxiv.org/pdf/2506.21826)  

**Abstract**: As rich sources of history, maps provide crucial insights into historical changes, yet their diverse visual representations and limited annotated data pose significant challenges for automated processing. We propose a simple yet effective approach for few-shot segmentation of historical maps, leveraging the rich semantic embeddings of large vision foundation models combined with parameter-efficient fine-tuning. Our method outperforms the state-of-the-art on the Siegfried benchmark dataset in vineyard and railway segmentation, achieving +5% and +13% relative improvements in mIoU in 10-shot scenarios and around +20% in the more challenging 5-shot setting. Additionally, it demonstrates strong performance on the ICDAR 2021 competition dataset, attaining a mean PQ of 67.3% for building block segmentation, despite not being optimized for this shape-sensitive metric, underscoring its generalizability. Notably, our approach maintains high performance even in extremely low-data regimes (10- & 5-shot), while requiring only 689k trainable parameters - just 0.21% of the total model size. Our approach enables precise segmentation of diverse historical maps while drastically reducing the need for manual annotations, advancing automated processing and analysis in the field. Our implementation is publicly available at: this https URL. 

---
# SciMantify -- A Hybrid Approach for the Evolving Semantification of Scientific Knowledge 

**Authors**: Lena John, Kheir Eddine Farfar, Sören Auer, Oliver Karras  

**Link**: [PDF](https://arxiv.org/pdf/2506.21819)  

**Abstract**: Scientific publications, primarily digitized as PDFs, remain static and unstructured, limiting the accessibility and reusability of the contained knowledge. At best, scientific knowledge from publications is provided in tabular formats, which lack semantic context. A more flexible, structured, and semantic representation is needed to make scientific knowledge understandable and processable by both humans and machines. We propose an evolution model of knowledge representation, inspired by the 5-star Linked Open Data (LOD) model, with five stages and defined criteria to guide the stepwise transition from a digital artifact, such as a PDF, to a semantic representation integrated in a knowledge graph (KG). Based on an exemplary workflow implementing the entire model, we developed a hybrid approach, called SciMantify, leveraging tabular formats of scientific knowledge, e.g., results from secondary studies, to support its evolving semantification. In the approach, humans and machines collaborate closely by performing semantic annotation tasks (SATs) and refining the results to progressively improve the semantic representation of scientific knowledge. We implemented the approach in the Open Research Knowledge Graph (ORKG), an established platform for improving the findability, accessibility, interoperability, and reusability of scientific knowledge. A preliminary user experiment showed that the approach simplifies the preprocessing of scientific knowledge, reduces the effort for the evolving semantification, and enhances the knowledge representation through better alignment with the KG structures. 

---
# Exploring the Structure of AI-Induced Language Change in Scientific English 

**Authors**: Riley Galpin, Bryce Anderson, Tom S. Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2506.21817)  

**Abstract**: Scientific English has undergone rapid and unprecedented changes in recent years, with words such as "delve," "intricate," and "crucial" showing significant spikes in frequency since around 2022. These changes are widely attributed to the growing influence of Large Language Models like ChatGPT in the discourse surrounding bias and misalignment. However, apart from changes in frequency, the exact structure of these linguistic shifts has remained unclear. The present study addresses this and investigates whether these changes involve the replacement of synonyms by suddenly 'spiking words,' for example, "crucial" replacing "essential" and "key," or whether they reflect broader semantic and pragmatic qualifications. To further investigate structural changes, we include part of speech tagging in our analysis to quantify linguistic shifts over grammatical categories and differentiate between word forms, like "potential" as a noun vs. as an adjective. We systematically analyze synonym groups for widely discussed 'spiking words' based on frequency trends in scientific abstracts from PubMed. We find that entire semantic clusters often shift together, with most or all words in a group increasing in usage. This pattern suggests that changes induced by Large Language Models are primarily semantic and pragmatic rather than purely lexical. Notably, the adjective "important" shows a significant decline, which prompted us to systematically analyze decreasing lexical items. Our analysis of "collapsing" words reveals a more complex picture, which is consistent with organic language change and contrasts with the patterns of the abrupt spikes. These insights into the structure of language change contribute to our understanding of how language technology continues to shape human language. 

---
# CAT-SG: A Large Dynamic Scene Graph Dataset for Fine-Grained Understanding of Cataract Surgery 

**Authors**: Felix Holm, Gözde Ünver, Ghazal Ghazaei, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2506.21813)  

**Abstract**: Understanding the intricate workflows of cataract surgery requires modeling complex interactions between surgical tools, anatomical structures, and procedural techniques. Existing datasets primarily address isolated aspects of surgical analysis, such as tool detection or phase segmentation, but lack comprehensive representations that capture the semantic relationships between entities over time. This paper introduces the Cataract Surgery Scene Graph (CAT-SG) dataset, the first to provide structured annotations of tool-tissue interactions, procedural variations, and temporal dependencies. By incorporating detailed semantic relations, CAT-SG offers a holistic view of surgical workflows, enabling more accurate recognition of surgical phases and techniques. Additionally, we present a novel scene graph generation model, CatSGG, which outperforms current methods in generating structured surgical representations. The CAT-SG dataset is designed to enhance AI-driven surgical training, real-time decision support, and workflow analysis, paving the way for more intelligent, context-aware systems in clinical practice. 

---
# From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining 

**Authors**: Fuying Wang, Jiacheng Xu, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21803)  

**Abstract**: Electrocardiograms (ECGs) play a vital role in monitoring cardiac health and diagnosing heart diseases. However, traditional deep learning approaches for ECG analysis rely heavily on large-scale manual annotations, which are both time-consuming and resource-intensive to obtain. To overcome this limitation, self-supervised learning (SSL) has emerged as a promising alternative, enabling the extraction of robust ECG representations that can be efficiently transferred to various downstream tasks. While previous studies have explored SSL for ECG pretraining and multi-modal ECG-language alignment, they often fail to capture the multi-scale nature of ECG signals. As a result, these methods struggle to learn generalized representations due to their inability to model the hierarchical structure of ECG data. To address this gap, we introduce MELP, a novel Multi-scale ECG-Language Pretraining (MELP) model that fully leverages hierarchical supervision from ECG-text pairs. MELP first pretrains a cardiology-specific language model to enhance its understanding of clinical text. It then applies three levels of cross-modal supervision-at the token, beat, and rhythm levels-to align ECG signals with textual reports, capturing structured information across different time scales. We evaluate MELP on three public ECG datasets across multiple tasks, including zero-shot ECG classification, linear probing, and transfer learning. Experimental results demonstrate that MELP outperforms existing SSL methods, underscoring its effectiveness and adaptability across diverse clinical applications. Our code is available at this https URL. 

---
# Demonstrating Interoperable Channel State Feedback Compression with Machine Learning 

**Authors**: Dani Korpi, Rachel Wang, Jerry Wang, Abdelrahman Ibrahim, Carl Nuzman, Runxin Wang, Kursat Rasim Mestav, Dustin Zhang, Iraj Saniee, Shawn Winston, Gordana Pavlovic, Wei Ding, William J. Hillery, Chenxi Hao, Ram Thirunagari, Jung Chang, Jeehyun Kim, Bartek Kozicki, Dragan Samardzija, Taesang Yoo, Andreas Maeder, Tingfang Ji, Harish Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21796)  

**Abstract**: Neural network-based compression and decompression of channel state feedback has been one of the most widely studied applications of machine learning (ML) in wireless networks. Various simulation-based studies have shown that ML-based feedback compression can result in reduced overhead and more accurate channel information. However, to the best of our knowledge, there are no real-life proofs of concepts demonstrating the benefits of ML-based channel feedback compression in a practical setting, where the user equipment (UE) and base station have no access to each others' ML models. In this paper, we present a novel approach for training interoperable compression and decompression ML models in a confidential manner, and demonstrate the accuracy of the ensuing models using prototype UEs and base stations. The performance of the ML-based channel feedback is measured both in terms of the accuracy of the reconstructed channel information and achieved downlink throughput gains when using the channel information for beamforming. The reported measurement results demonstrate that it is possible to develop an accurate ML-based channel feedback link without having to share ML models between device and network vendors. These results pave the way for a practical implementation of ML-based channel feedback in commercial 6G networks. 

---
# Multi-task parallelism for robust pre-training of graph foundation models on multi-source, multi-fidelity atomistic modeling data 

**Authors**: Massimiliano Lupo Pasini, Jong Youl Choi, Pei Zhang, Kshitij Mehta, Rylie Weaver, Ashwin M. Aji, Karl W. Schulz, Jorda Polo, Prasanna Balaprakash  

**Link**: [PDF](https://arxiv.org/pdf/2506.21788)  

**Abstract**: Graph foundation models using graph neural networks promise sustainable, efficient atomistic modeling. To tackle challenges of processing multi-source, multi-fidelity data during pre-training, recent studies employ multi-task learning, in which shared message passing layers initially process input atomistic structures regardless of source, then route them to multiple decoding heads that predict data-specific outputs. This approach stabilizes pre-training and enhances a model's transferability to unexplored chemical regions. Preliminary results on approximately four million structures are encouraging, yet questions remain about generalizability to larger, more diverse datasets and scalability on supercomputers. We propose a multi-task parallelism method that distributes each head across computing resources with GPU acceleration. Implemented in the open-source HydraGNN architecture, our method was trained on over 24 million structures from five datasets and tested on the Perlmutter, Aurora, and Frontier supercomputers, demonstrating efficient scaling on all three highly heterogeneous super-computing architectures. 

---
# Comparing Learning Paradigms for Egocentric Video Summarization 

**Authors**: Daniel Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21785)  

**Abstract**: In this study, we investigate various computer vision paradigms - supervised learning, unsupervised learning, and prompt fine-tuning - by assessing their ability to understand and interpret egocentric video data. Specifically, we examine Shotluck Holmes (state-of-the-art supervised learning), TAC-SUM (state-of-the-art unsupervised learning), and GPT-4o (a prompt fine-tuned pre-trained model), evaluating their effectiveness in video summarization. Our results demonstrate that current state-of-the-art models perform less effectively on first-person videos compared to third-person videos, highlighting the need for further advancements in the egocentric video domain. Notably, a prompt fine-tuned general-purpose GPT-4o model outperforms these specialized models, emphasizing the limitations of existing approaches in adapting to the unique challenges of first-person perspectives. Although our evaluation is conducted on a small subset of egocentric videos from the Ego-Exo4D dataset due to resource constraints, the primary objective of this research is to provide a comprehensive proof-of-concept analysis aimed at advancing the application of computer vision techniques to first-person videos. By exploring novel methodologies and evaluating their potential, we aim to contribute to the ongoing development of models capable of effectively processing and interpreting egocentric perspectives. 

---
# Evaluating List Construction and Temporal Understanding capabilities of Large Language Models 

**Authors**: Alexandru Dumitru, V Venktesh, Adam Jatowt, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2506.21783)  

**Abstract**: Large Language Models (LLMs) have demonstrated immense advances in a wide range of natural language tasks. However, these models are susceptible to hallucinations and errors on particularly temporal understanding tasks involving multiple entities in answers. In such tasks, they fail to associate entities with accurate time intervals, generate a complete list of entities in answers or reason about events associated with specific temporal bounds. Existing works do not extensively evaluate the abilities of the model to perform implicit and explicit temporal understanding in a list answer construction setup. To bridge this gap, we propose the Time referenced List based Question Answering or TLQA benchmark that requires structured answers in list format aligned with corresponding time periods. Our TLQA benchmark, requires both list construction and temporal understanding simultaneously, which to the best of our knowledge has not been explored in prior benchmarks. We investigate the temporal understanding and list construction capabilities of state-of-the-art generative models on TLQA in closed-book and open-domain settings. Our findings reveal significant shortcomings in current models, particularly their inability to provide complete answers and temporally align facts in a closed-book setup and the need to improve retrieval in open-domain setup, providing clear future directions for research on TLQA. The benchmark and code at this https URL. 

---
# Experimental investigation of pose informed reinforcement learning for skid-steered visual navigation 

**Authors**: Ameya Salvi, Venkat Krovi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21732)  

**Abstract**: Vision-based lane keeping is a topic of significant interest in the robotics and autonomous ground vehicles communities in various on-road and off-road applications. The skid-steered vehicle architecture has served as a useful vehicle platform for human controlled operations. However, systematic modeling, especially of the skid-slip wheel terrain interactions (primarily in off-road settings) has created bottlenecks for automation deployment. End-to-end learning based methods such as imitation learning and deep reinforcement learning, have gained prominence as a viable deployment option to counter the lack of accurate analytical models. However, the systematic formulation and subsequent verification/validation in dynamic operation regimes (particularly for skid-steered vehicles) remains a work in progress. To this end, a novel approach for structured formulation for learning visual navigation is proposed and investigated in this work. Extensive software simulations, hardware evaluations and ablation studies now highlight the significantly improved performance of the proposed approach against contemporary literature. 

---
# Exploring Image Generation via Mutually Exclusive Probability Spaces and Local Correlation Hypothesis 

**Authors**: Chenqiu Zhao, Anup Basu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21731)  

**Abstract**: We propose two theoretical frameworks, the Mutually Exclusive Probability Space (MESP) and the Local Correlation Hypothesis (LCH), to explore a potential limitation in probabilistic generative models; namely that learning global distributions leads to memorization rather than generative behavior. MESP emerges from our rethinking of the Variational Autoencoder (VAE). We observe that latent variable distributions in VAE exhibit overlap, which leads to an optimization conflict between the reconstruction loss and KL-divergence loss. A lower bound based on the overlap coefficient is proposed. We refer to this phenomenon as Mutually Exclusive Probability Spaces. Based on MESP, a Binary Latent Autoencoder (BL-AE) is proposed to encode images into binary latent representations. These binary latents are used as the input to our Autoregressive Random Variable Model (ARVM), a modified autoregressive model outputting histograms. Our ARVM achieves competitive FID scores, outperforming state-of-the-art methods on standard datasets. However, such scores reflect memorization rather than generation. To address this issue, we propose the Local Correlation Hypothesis (LCH), which posits that generative capability arising from local correlations among latent variables. Comprehensive experiments and discussions are conducted to validate our frameworks. 

---
# Simultaneously Fair Allocation of Indivisible Items Across Multiple Dimensions 

**Authors**: Yasushi Kawase, Bodhayan Roy, Mohammad Azharuddin Sanpui  

**Link**: [PDF](https://arxiv.org/pdf/2506.21727)  

**Abstract**: This paper explores the fair allocation of indivisible items in a multidimensional setting, motivated by the need to address fairness in complex environments where agents assess bundles according to multiple criteria. Such multidimensional settings are not merely of theoretical interest but are central to many real-world applications. For example, cloud computing resources are evaluated based on multiple criteria such as CPU cores, memory, and network bandwidth. In such cases, traditional one dimensional fairness notions fail to capture fairness across multiple attributes. To address these challenges, we study two relaxed variants of envy-freeness: weak simultaneously envy-free up to c goods (weak sEFc) and strong simultaneously envy-free up to c goods (strong sEFc), which accommodate the multidimensionality of agents' preferences. Under the weak notion, for every pair of agents and for each dimension, any perceived envy can be eliminated by removing, if necessary, a different set of goods from the envied agent's allocation. In contrast, the strong version requires selecting a single set of goods whose removal from the envied bundle simultaneously eliminates envy in every dimension. We provide upper and lower bounds on the relaxation parameter c that guarantee the existence of weak or strong sEFc allocations, where these bounds are independent of the total number of items. In addition, we present algorithms for checking whether a weak or strong sEFc allocation exists. Moreover, we establish NP-hardness results for checking the existence of weak sEF1 and strong sEF1 allocations. 

---
# Elucidating and Endowing the Diffusion Training Paradigm for General Image Restoration 

**Authors**: Xin Lu, Xueyang Fu, Jie Xiao, Zihao Fan, Yurui Zhu, Zheng-Jun Zha  

**Link**: [PDF](https://arxiv.org/pdf/2506.21722)  

**Abstract**: While diffusion models demonstrate strong generative capabilities in image restoration (IR) tasks, their complex architectures and iterative processes limit their practical application compared to mainstream reconstruction-based general ordinary IR networks. Existing approaches primarily focus on optimizing network architecture and diffusion paths but overlook the integration of the diffusion training paradigm within general ordinary IR frameworks. To address these challenges, this paper elucidates key principles for adapting the diffusion training paradigm to general IR training through systematic analysis of time-step dependencies, network hierarchies, noise-level relationships, and multi-restoration task correlations, proposing a new IR framework supported by diffusion-based training. To enable IR networks to simultaneously restore images and model generative representations, we introduce a series of regularization strategies that align diffusion objectives with IR tasks, improving generalization in single-task scenarios. Furthermore, recognizing that diffusion-based generation exerts varying influences across different IR tasks, we develop an incremental training paradigm and task-specific adaptors, further enhancing performance in multi-task unified IR. Experiments demonstrate that our method significantly improves the generalization of IR networks in single-task IR and achieves superior performance in multi-task unified IR. Notably, the proposed framework can be seamlessly integrated into existing general IR architectures. 

---
# Performance Prediction for Large Systems via Text-to-Text Regression 

**Authors**: Yash Akhauri, Bryan Lewandowski, Cheng-Hsi Lin, Adrian N. Reyes, Grant C. Forbes, Arissa Wongpanich, Bangding Yang, Mohamed S. Abdelfattah, Sagi Perel, Xingyou Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.21718)  

**Abstract**: In many industries, predicting metric outcomes of large systems is a fundamental problem, driven largely by traditional tabular regression. However, such methods struggle on complex systems data in the wild such as configuration files or system logs, where feature engineering is often infeasible. We propose text-to-text regression as a general, scalable alternative. For predicting resource efficiency on Borg, Google's massive compute cluster scheduling system, a 60M parameter encoder-decoder, trained from random initialization, achieves up to a near perfect 0.99 (0.9 average) rank correlation across the entire fleet, and 100x lower MSE than tabular approaches. The model also easily adapts to new tasks in only 500 few-shot examples and captures the densities of complex outcome distributions. Ablation studies highlight the importance of using encoders, increasing sequence length, and the model's inherent uncertainty quantification. These findings pave the way for universal simulators of real-world outcomes. 

---
# APO: Enhancing Reasoning Ability of MLLMs via Asymmetric Policy Optimization 

**Authors**: Minjie Hong, Zirun Guo, Yan Xia, Zehan Wang, Ziang Zhang, Tao Jin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21655)  

**Abstract**: Multimodal Large Language Models (MLLMs) are powerful at integrating diverse data, but they often struggle with complex reasoning. While Reinforcement learning (RL) can boost reasoning in LLMs, applying it to MLLMs is tricky. Common issues include a drop in performance on general tasks and the generation of overly detailed or "overthinking" reasoning. Our work investigates how the KL penalty and overthinking affect RL training in MLLMs. We propose Asymmetric Policy Optimization (APO) to address these issues, which divides the sampled responses into positive and negative groups. For positive samples, Difficulty-Adaptive Divergence Shaping (DADS) is introduced to dynamically adjust the KL divergence weight based on their difficulty. This method prevents policy entropy from dropping sharply, improves training stability, utilizes samples better, and preserves the model's existing knowledge. For negative samples, Suboptimal Trajectory Complexity Regularization (STCR) is proposed to penalize overly long responses. This helps mitigate overthinking and encourages more concise reasoning while preserving the model's explorative capacity. We apply our method to Qwen2.5-VL-3B, creating View-R1-3B. View-R1-3B significantly enhances reasoning capabilities, showing an average 7\% gain over the base model and outperforming larger MLLMs (7-11B) on various reasoning benchmarks. Importantly, unlike other reasoning-tuned MLLMs that often degrade on general tasks, View-R1-3B maintains consistent improvement, demonstrating superior generalization. These results highlight the effectiveness and broad applicability of our DADS and STCR techniques for advancing complex multimodal reasoning in MLLMs. The code will be made available at this https URL. 

---
# IRanker: Towards Ranking Foundation Model 

**Authors**: Tao Feng, Zhigang Hua, Zijie Lei, Yan Xie, Shuang Yang, Bo Long, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.21638)  

**Abstract**: Ranking tasks are ubiquitous, encompassing applications such as recommendation systems, LLM routing, and item re-ranking. We propose to unify these tasks using a single ranking foundation model (FM), as it eliminates the need for designing different models for each specific ranking task. However, unlike general supervision tasks in LLMs, ranking tasks do not have clear labels for supervision, posing great challenges to developing a ranking FM. To overcome these challenges, we propose IRanker, a ranking FM framework with reinforcement learning (RL) and iterative decoding. Our insight is to decompose the complex ranking task into an iterative decoding process that eliminates the worst candidate from the candidate pool step by step, which significantly reduces the output combinatorial space and better utilizes the limited context length during RL training. We meticulously train and comprehensively evaluate an IRanker-3B model on nine datasets across three scenarios: recommendation, routing, and passage ranking. The results show that a single IRanker-3B achieves state-of-the-art results on several datasets compared to models of similar size, and even surpasses the performance of larger models on certain datasets. We further demonstrate the effectiveness of our RL design and the robustness of the iterative mechanism across different LLM sizes. Moreover, we conducted both in-domain and out-of-domain zero-shot generalization experiments, which showed that IRanker-3B achieved good generalization on in-domain ranking tasks compared to the base LLM by at least 5% improvement. Surprisingly, on out-of-domain generic LLM tasks, IRanker-3B outperformed the base model by at least 9% on GSM8K, IFEval, and MathQA. In addition, the thoughts generated by IRanker-3B during training could further enhance zero-shot LLM performance. 

---
# AeroLite-MDNet: Lightweight Multi-task Deviation Detection Network for UAV Landing 

**Authors**: Haiping Yang, Huaxing Liu, Wei Wu, Zuohui Chen, Ning Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21635)  

**Abstract**: Unmanned aerial vehicles (UAVs) are increasingly employed in diverse applications such as land surveying, material transport, and environmental monitoring. Following missions like data collection or inspection, UAVs must land safely at docking stations for storage or recharging, which is an essential requirement for ensuring operational continuity. However, accurate landing remains challenging due to factors like GPS signal interference. To address this issue, we propose a deviation warning system for UAV landings, powered by a novel vision-based model called AeroLite-MDNet. This model integrates a multiscale fusion module for robust cross-scale object detection and incorporates a segmentation branch for efficient orientation estimation. We introduce a new evaluation metric, Average Warning Delay (AWD), to quantify the system's sensitivity to landing deviations. Furthermore, we contribute a new dataset, UAVLandData, which captures real-world landing deviation scenarios to support training and evaluation. Experimental results show that our system achieves an AWD of 0.7 seconds with a deviation detection accuracy of 98.6\%, demonstrating its effectiveness in enhancing UAV landing reliability. Code will be available at this https URL 

---
# Ark: An Open-source Python-based Framework for Robot Learning 

**Authors**: Magnus Dierking, Christopher E. Mower, Sarthak Das, Huang Helong, Jiacheng Qiu, Cody Reading, Wei Chen, Huidong Liang, Huang Guowei, Jan Peters, Quan Xingyue, Jun Wang, Haitham Bou-Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21628)  

**Abstract**: Robotics has made remarkable hardware strides-from DARPA's Urban and Robotics Challenges to the first humanoid-robot kickboxing tournament-yet commercial autonomy still lags behind progress in machine learning. A major bottleneck is software: current robot stacks demand steep learning curves, low-level C/C++ expertise, fragmented tooling, and intricate hardware integration, in stark contrast to the Python-centric, well-documented ecosystems that propelled modern AI. We introduce ARK, an open-source, Python-first robotics framework designed to close that gap. ARK presents a Gym-style environment interface that allows users to collect data, preprocess it, and train policies using state-of-the-art imitation-learning algorithms (e.g., ACT, Diffusion Policy) while seamlessly toggling between high-fidelity simulation and physical robots. A lightweight client-server architecture provides networked publisher-subscriber communication, and optional C/C++ bindings ensure real-time performance when needed. ARK ships with reusable modules for control, SLAM, motion planning, system identification, and visualization, along with native ROS interoperability. Comprehensive documentation and case studies-from manipulation to mobile navigation-demonstrate rapid prototyping, effortless hardware swapping, and end-to-end pipelines that rival the convenience of mainstream machine-learning workflows. By unifying robotics and AI practices under a common Python umbrella, ARK lowers entry barriers and accelerates research and commercial deployment of autonomous robots. 

---
# FrankenBot: Brain-Morphic Modular Orchestration for Robotic Manipulation with Vision-Language Models 

**Authors**: Shiyi Wang, Wenbo Li, Yiteng Chen, Qingyao Wu, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21627)  

**Abstract**: Developing a general robot manipulation system capable of performing a wide range of tasks in complex, dynamic, and unstructured real-world environments has long been a challenging task. It is widely recognized that achieving human-like efficiency and robustness manipulation requires the robotic brain to integrate a comprehensive set of functions, such as task planning, policy generation, anomaly monitoring and handling, and long-term memory, achieving high-efficiency operation across all functions. Vision-Language Models (VLMs), pretrained on massive multimodal data, have acquired rich world knowledge, exhibiting exceptional scene understanding and multimodal reasoning capabilities. However, existing methods typically focus on realizing only a single function or a subset of functions within the robotic brain, without integrating them into a unified cognitive architecture. Inspired by a divide-and-conquer strategy and the architecture of the human brain, we propose FrankenBot, a VLM-driven, brain-morphic robotic manipulation framework that achieves both comprehensive functionality and high operational efficiency. Our framework includes a suite of components, decoupling a part of key functions from frequent VLM calls, striking an optimal balance between functional completeness and system efficiency. Specifically, we map task planning, policy generation, memory management, and low-level interfacing to the cortex, cerebellum, temporal lobe-hippocampus complex, and brainstem, respectively, and design efficient coordination mechanisms for the modules. We conducted comprehensive experiments in both simulation and real-world robotic environments, demonstrating that our method offers significant advantages in anomaly detection and handling, long-term memory, operational efficiency, and stability -- all without requiring any fine-tuning or retraining. 

---
# Doc2SAR: A Synergistic Framework for High-Fidelity Extraction of Structure-Activity Relationships from Scientific Documents 

**Authors**: Jiaxi Zhuang, Kangning Li, Jue Hou, Mingjun Xu, Zhifeng Gao, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21625)  

**Abstract**: Extracting molecular structure-activity relationships (SARs) from scientific literature and patents is essential for drug discovery and materials research. However, this task remains challenging due to heterogeneous document formats and limitations of existing methods. Specifically, rule-based approaches relying on rigid templates fail to generalize across diverse document layouts, while general-purpose multimodal large language models (MLLMs) lack sufficient accuracy and reliability for specialized tasks, such as layout detection and optical chemical structure recognition (OCSR). To address these challenges, we introduce DocSAR-200, a rigorously annotated benchmark of 200 scientific documents designed specifically for evaluating SAR extraction methods. Additionally, we propose Doc2SAR, a novel synergistic framework that integrates domain-specific tools with MLLMs enhanced via supervised fine-tuning (SFT). Extensive experiments demonstrate that Doc2SAR achieves state-of-the-art performance across various document types, significantly outperforming leading end-to-end baselines. Specifically, Doc2SAR attains an overall Table Recall of 80.78% on DocSAR-200, exceeding end2end GPT-4o by 51.48%. Furthermore, Doc2SAR demonstrates practical usability through efficient inference and is accompanied by a web app. 

---
# Adapting Foundation Speech Recognition Models to Impaired Speech: A Semantic Re-chaining Approach for Personalization of German Speech 

**Authors**: Niclas Pokel, Pehuén Moure, Roman Boehringer, Yingqiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21622)  

**Abstract**: Speech impairments caused by conditions such as cerebral palsy or genetic disorders pose significant challenges for automatic speech recognition (ASR) systems. Despite recent advances, ASR models like Whisper struggle with non-normative speech due to limited training data and the difficulty of collecting and annotating non-normative speech samples. In this work, we propose a practical and lightweight pipeline to personalize ASR models, formalizing the selection of words and enriching a small, speech-impaired dataset with semantic coherence. Applied to data from a child with a structural speech impairment, our approach shows promising improvements in transcription quality, demonstrating the potential to reduce communication barriers for individuals with atypical speech patterns. 

---
# The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs 

**Authors**: Jasper Dekoninck, Ivo Petrov, Kristian Minchev, Mislav Balunovic, Martin Vechev, Miroslav Marinov, Maria Drencheva, Lyuba Konova, Milen Shumanov, Kaloyan Tsvetkov, Nikolay Drenchev, Lazar Todorov, Kalina Nikolova, Nikolay Georgiev, Vanesa Kalinkova, Margulan Ismoldayev  

**Link**: [PDF](https://arxiv.org/pdf/2506.21621)  

**Abstract**: In recent months, large language models (LLMs) have made significant progress in mathematical proof generation, but further advancement is hindered by the lack of a large-scale, high-quality dataset of human-evaluated proofs. While expensive to create, such a dataset is essential for driving improvements in training and enabling a rigorous analysis of proof generation capabilities. In this work, we present the Open Proof Corpus (OPC), a dataset comprising over 5,000 human-evaluated proofs produced by state-of-the-art LLMs. The OPC was specifically designed for broad applicability and downstream usage in proof generation research and is the first to include a substantial number of correct, LLM-generated solutions to problems from prestigious mathematics competitions such as the USAMO and IMO. Using the OPC, we explore critical questions in automated proof generation: (1) the performance gap between natural language and formal proof generation, (2) the discrepancy between final-answer accuracy and full-proof validity, and (3) the impact of best-of-n selection on proof quality. Finally, to showcase the utility of the OPC, we finetune an 8B-parameter model on the dataset, obtaining a model that performs on par with the best model, Gemini-2.5-Pro, on the task of evaluating proof correctness. 

---
# How Large Language Models play humans in online conversations: a simulated study of the 2016 US politics on Reddit 

**Authors**: Daniele Cirulli, Giulio Cimini, Giovanni Palermo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21620)  

**Abstract**: Large Language Models (LLMs) have recently emerged as powerful tools for natural language generation, with applications spanning from content creation to social simulations. Their ability to mimic human interactions raises both opportunities and concerns, particularly in the context of politically relevant online discussions. In this study, we evaluate the performance of LLMs in replicating user-generated content within a real-world, divisive scenario: Reddit conversations during the 2016 US Presidential election. In particular, we conduct three different experiments, asking GPT-4 to generate comments by impersonating either real or artificial partisan users. We analyze the generated comments in terms of political alignment, sentiment, and linguistic features, comparing them against real user contributions and benchmarking against a null model. We find that GPT-4 is able to produce realistic comments, both in favor of or against the candidate supported by the community, yet tending to create consensus more easily than dissent. In addition we show that real and artificial comments are well separated in a semantically embedded space, although they are indistinguishable by manual inspection. Our findings provide insights on the potential use of LLMs to sneak into online discussions, influence political debate and shape political narratives, bearing broader implications of AI-driven discourse manipulation. 

---
# IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech 

**Authors**: Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21619)  

**Abstract**: Large-scale text-to-speech (TTS) models are typically categorized into autoregressive and non-autoregressive systems. Although autoregressive systems exhibit certain advantages in speech naturalness, their token-by-token generation mechanism makes it difficult to precisely control the duration of synthesized speech. This is a key limitation in applications such as video dubbing that require strict audio-visual synchronization. This paper introduces IndexTTS2, which proposes a novel and autoregressive-model-friendly method for speech duration control. The method supports two generation modes: one allows explicit specification of the number of generated tokens for precise duration control; the other does not require manual input and lets the model freely generate speech while preserving prosodic characteristics from the input prompt. Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control of timbre and emotion. In the zero-shot setting, the model can perfectly reproduce the emotional characteristics of the input prompt. Users may also provide a separate emotion prompt, even from a different speaker, allowing the model to reconstruct the target timbre while conveying the desired emotion. To enhance clarity during strong emotional expressions, we incorporate GPT latent representations to improve speech stability. Meanwhile, to lower the barrier for emotion control, we design a soft instruction mechanism based on textual descriptions by fine-tuning Qwen3. This enables effective guidance of speech generation with desired emotional tendencies using natural language input. Experimental results demonstrate that IndexTTS2 outperforms existing state-of-the-art zero-shot TTS models in word error rate, speaker similarity, and emotional fidelity. 

---
# TrajTok: Technical Report for 2025 Waymo Open Sim Agents Challenge 

**Authors**: Zhiyuan Zhang, Xiaosong Jia, Guanyu Chen, Qifeng Li, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21618)  

**Abstract**: In this technical report, we introduce TrajTok, a trajectory tokenizer for discrete next-token-prediction based behavior generation models, which combines data-driven and rule-based methods with better coverage, symmetry and robustness, along with a spatial-aware label smoothing method for cross-entropy loss. We adopt the tokenizer and loss for the SMART model and reach a superior performance with realism score of 0.7852 on the Waymo Open Sim Agents Challenge 2025. We will open-source the code in the future. 

---
# Bayesian-Guided Diversity in Sequential Sampling for Recommender Systems 

**Authors**: Hiba Bederina, Jill-Jênn Vie  

**Link**: [PDF](https://arxiv.org/pdf/2506.21617)  

**Abstract**: The challenge of balancing user relevance and content diversity in recommender systems is increasingly critical amid growing concerns about content homogeneity and reduced user engagement. In this work, we propose a novel framework that leverages a multi-objective, contextual sequential sampling strategy. Item selection is guided by Bayesian updates that dynamically adjust scores to optimize diversity. The reward formulation integrates multiple diversity metrics-including the log-determinant volume of a tuned similarity submatrix and ridge leverage scores-along with a diversity gain uncertainty term to address the exploration-exploitation trade-off. Both intra- and inter-batch diversity are modeled to promote serendipity and minimize redundancy. A dominance-based ranking procedure identifies Pareto-optimal item sets, enabling adaptive and balanced selections at each iteration. Experiments on a real-world dataset show that our approach significantly improves diversity without sacrificing relevance, demonstrating its potential to enhance user experience in large-scale recommendation settings. 

---
# Refine Medical Diagnosis Using Generation Augmented Retrieval and Clinical Practice Guidelines 

**Authors**: Wenhao Li, Hongkuan Zhang, Hongwei Zhang, Zhengxu Li, Zengjie Dong, Yafan Chen, Niranjan Bidargaddi, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21615)  

**Abstract**: Current medical language models, adapted from large language models (LLMs), typically predict ICD code-based diagnosis from electronic health records (EHRs) because these labels are readily available. However, ICD codes do not capture the nuanced, context-rich reasoning clinicians use for diagnosis. Clinicians synthesize diverse patient data and reference clinical practice guidelines (CPGs) to make evidence-based decisions. This misalignment limits the clinical utility of existing models. We introduce GARMLE-G, a Generation-Augmented Retrieval framework that grounds medical language model outputs in authoritative CPGs. Unlike conventional Retrieval-Augmented Generation based approaches, GARMLE-G enables hallucination-free outputs by directly retrieving authoritative guideline content without relying on model-generated text. It (1) integrates LLM predictions with EHR data to create semantically rich queries, (2) retrieves relevant CPG knowledge snippets via embedding similarity, and (3) fuses guideline content with model output to generate clinically aligned recommendations. A prototype system for hypertension diagnosis was developed and evaluated on multiple metrics, demonstrating superior retrieval precision, semantic relevance, and clinical guideline adherence compared to RAG-based baselines, while maintaining a lightweight architecture suitable for localized healthcare deployment. This work provides a scalable, low-cost, and hallucination-free method for grounding medical language models in evidence-based clinical practice, with strong potential for broader clinical deployment. 

---
# LastingBench: Defend Benchmarks Against Knowledge Leakage 

**Authors**: Yixiong Fang, Tianran Sun, Yuling Shi, Min Wang, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21614)  

**Abstract**: The increasing complexity of large language models (LLMs) raises concerns about their ability to "cheat" on standard Question Answering (QA) benchmarks by memorizing task-specific data. This undermines the validity of benchmark evaluations, as they no longer reflect genuine model capabilities but instead the effects of data leakage. While prior work has focused on detecting such leakage, little attention has been given to mitigating its impact and preserving the long-term utility of benchmarks. In this paper, we introduce LastingBench, a novel framework designed to continuously reinforce and safeguard existing benchmarks against knowledge leakage. LastingBench identifies leakage points in the context through perturbation, then rewrites the leakage points to counterfactual ones-disrupting memorization while preserving the benchmark's original evaluative intent. Evaluations of state-of-the-art QA benchmarks show significant performance gaps, highlighting the efficacy of LastingBench in reducing memorization effects. LastingBench offers a practical and scalable solution to ensure benchmark robustness over time, promoting fairer and more interpretable evaluations of LLMs. 

---
# AdaptGOT: A Pre-trained Model for Adaptive Contextual POI Representation Learning 

**Authors**: Xiaobin Ren, Xinyu Zhu, Kaiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21612)  

**Abstract**: Currently, considerable strides have been achieved in Point-of-Interest (POI) embedding methodologies, driven by the emergence of novel POI tasks like recommendation and classification. Despite the success of task-specific, end-to-end models in POI embedding, several challenges remain. These include the need for more effective multi-context sampling strategies, insufficient exploration of multiple POI contexts, limited versatility, and inadequate generalization. To address these issues, we propose the AdaptGOT model, which integrates both the (Adapt)ive representation learning technique and the Geographical-Co-Occurrence-Text (GOT) representation with a particular emphasis on Geographical location, Co-Occurrence and Textual information. The AdaptGOT model comprises three key components: (1) contextual neighborhood generation, which integrates advanced mixed sampling techniques such as KNN, density-based, importance-based, and category-aware strategies to capture complex contextual neighborhoods; (2) an advanced GOT representation enhanced by an attention mechanism, designed to derive high-quality, customized representations and efficiently capture complex interrelations between POIs; and (3) the MoE-based adaptive encoder-decoder architecture, which ensures topological consistency and enriches contextual representation by minimizing Jensen-Shannon divergence across varying contexts. Experiments on two real-world datasets and multiple POI tasks substantiate the superior performance of the proposed AdaptGOT model. 

---
# Does Multimodality Lead to Better Time Series Forecasting? 

**Authors**: Xiyuan Zhang, Boran Han, Haoyang Fang, Abdul Fatir Ansari, Shuai Zhang, Danielle C. Maddix, Cuixiong Hu, Andrew Gordon Wilson, Michael W. Mahoney, Hao Wang, Yan Liu, Huzefa Rangwala, George Karypis, Bernie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21611)  

**Abstract**: Recently, there has been growing interest in incorporating textual information into foundation models for time series forecasting. However, it remains unclear whether and under what conditions such multimodal integration consistently yields gains. We systematically investigate these questions across a diverse benchmark of 14 forecasting tasks spanning 7 domains, including health, environment, and economics. We evaluate two popular multimodal forecasting paradigms: aligning-based methods, which align time series and text representations; and prompting-based methods, which directly prompt large language models for forecasting. Although prior works report gains from multimodal input, we find these effects are not universal across datasets and models, and multimodal methods sometimes do not outperform the strongest unimodal baselines. To understand when textual information helps, we disentangle the effects of model architectural properties and data characteristics. Our findings highlight that on the modeling side, incorporating text information is most helpful given (1) high-capacity text models, (2) comparatively weaker time series models, and (3) appropriate aligning strategies. On the data side, performance gains are more likely when (4) sufficient training data is available and (5) the text offers complementary predictive signal beyond what is already captured from the time series alone. Our empirical findings offer practical guidelines for when multimodality can be expected to aid forecasting tasks, and when it does not. 

---
# From Thinking to Output: Chain-of-Thought and Text Generation Characteristics in Reasoning Language Models 

**Authors**: Junhao Liu, Zhenhao Xu, Yuxin Fang, Yichuan Chen, Zuobin Ying, Wenhan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21609)  

**Abstract**: Recently, there have been notable advancements in large language models (LLMs), demonstrating their growing abilities in complex reasoning. However, existing research largely overlooks a thorough and systematic comparison of these models' reasoning processes and outputs, particularly regarding their self-reflection pattern (also termed "Aha moment") and the interconnections across diverse domains. This paper proposes a novel framework for analyzing the reasoning characteristics of four cutting-edge large reasoning models (GPT-o1, DeepSeek-R1, Kimi-k1.5, and Grok-3) using keywords statistic and LLM-as-a-judge paradigm. Our approach connects their internal thinking processes with their final outputs. A diverse dataset consists of real-world scenario-based questions covering logical deduction, causal inference, and multi-step problem-solving. Additionally, a set of metrics is put forward to assess both the coherence of reasoning and the accuracy of the outputs. The research results uncover various patterns of how these models balance exploration and exploitation, deal with problems, and reach conclusions during the reasoning process. Through quantitative and qualitative comparisons, disparities among these models are identified in aspects such as the depth of reasoning, the reliance on intermediate steps, and the degree of similarity between their thinking processes and output patterns and those of GPT-o1. This work offers valuable insights into the trade-off between computational efficiency and reasoning robustness and provides practical recommendations for enhancing model design and evaluation in practical applications. We publicly release our project at: this https URL 

---
# SysTemp: A Multi-Agent System for Template-Based Generation of SysML v2 

**Authors**: Yasmine Bouamra, Bruno Yun, Alexandre Poisson, Frédéric Armetta  

**Link**: [PDF](https://arxiv.org/pdf/2506.21608)  

**Abstract**: The automatic generation of SysML v2 models represents a major challenge in the engineering of complex systems, particularly due to the scarcity of learning corpora and complex syntax. We present SysTemp, a system aimed at facilitating and improving the creation of SysML v2 models from natural language specifications. It is based on a multi-agent system, including a template generator that structures the generation process. We discuss the advantages and challenges of this system through an evaluation, highlighting its potential to improve the quality of the generations in SysML v2 modeling. 

---
# CORE-KG: An LLM-Driven Knowledge Graph Construction Framework for Human Smuggling Networks 

**Authors**: Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera  

**Link**: [PDF](https://arxiv.org/pdf/2506.21607)  

**Abstract**: Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer valuable insights but are unstructured, lexically dense, and filled with ambiguous or shifting references-posing challenges for automated knowledge graph (KG) construction. Existing KG methods often rely on static templates and lack coreference resolution, while recent LLM-based approaches frequently produce noisy, fragmented graphs due to hallucinations, and duplicate nodes caused by a lack of guided extraction. We propose CORE-KG, a modular framework for building interpretable KGs from legal texts. It uses a two-step pipeline: (1) type-aware coreference resolution via sequential, structured LLM prompts, and (2) entity and relationship extraction using domain-guided instructions, built on an adapted GraphRAG framework. CORE-KG reduces node duplication by 33.28%, and legal noise by 38.37% compared to a GraphRAG-based baseline-resulting in cleaner and more coherent graph structures. These improvements make CORE-KG a strong foundation for analyzing complex criminal networks. 

---
# Large Language Models as symbolic DNA of cultural dynamics 

**Authors**: Parham Pourdavood, Michael Jacob, Terrence Deacon  

**Link**: [PDF](https://arxiv.org/pdf/2506.21606)  

**Abstract**: This paper proposes a novel conceptualization of Large Language Models (LLMs) as externalized informational substrates that function analogously to DNA for human cultural dynamics. Rather than viewing LLMs as either autonomous intelligence or mere programmed mimicry, we argue they serve a broader role as repositories that preserve compressed patterns of human symbolic expression--"fossils" of meaningful dynamics that retain relational residues without their original living contexts. Crucially, these compressed patterns only become meaningful through human reinterpretation, creating a recursive feedback loop where they can be recombined and cycle back to ultimately catalyze human creative processes. Through analysis of four universal features--compression, decompression, externalization, and recursion--we demonstrate that just as DNA emerged as a compressed and externalized medium for preserving useful cellular dynamics without containing explicit reference to goal-directed physical processes, LLMs preserve useful regularities of human culture without containing understanding of embodied human experience. Therefore, we argue that LLMs' significance lies not in rivaling human intelligence, but in providing humanity a tool for self-reflection and playful hypothesis-generation in a low-stakes, simulated environment. This framework positions LLMs as tools for cultural evolvability, enabling humanity to generate novel hypotheses about itself while maintaining the human interpretation necessary to ground these hypotheses in ongoing human aesthetics and norms. 

---
# MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents 

**Authors**: Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.21605)  

**Abstract**: Recent works have highlighted the significance of memory mechanisms in LLM-based agents, which enable them to store observed information and adapt to dynamic environments. However, evaluating their memory capabilities still remains challenges. Previous evaluations are commonly limited by the diversity of memory levels and interactive scenarios. They also lack comprehensive metrics to reflect the memory capabilities from multiple aspects. To address these problems, in this paper, we construct a more comprehensive dataset and benchmark to evaluate the memory capability of LLM-based agents. Our dataset incorporates factual memory and reflective memory as different levels, and proposes participation and observation as various interactive scenarios. Based on our dataset, we present a benchmark, named MemBench, to evaluate the memory capability of LLM-based agents from multiple aspects, including their effectiveness, efficiency, and capacity. To benefit the research community, we release our dataset and project at this https URL. 

---
# Evaluating VisualRAG: Quantifying Cross-Modal Performance in Enterprise Document Understanding 

**Authors**: Varun Mannam, Fang Wang, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21604)  

**Abstract**: Current evaluation frameworks for multimodal generative AI struggle to establish trustworthiness, hindering enterprise adoption where reliability is paramount. We introduce a systematic, quantitative benchmarking framework to measure the trustworthiness of progressively integrating cross-modal inputs such as text, images, captions, and OCR within VisualRAG systems for enterprise document intelligence. Our approach establishes quantitative relationships between technical metrics and user-centric trust measures. Evaluation reveals that optimal modality weighting with weights of 30% text, 15% image, 25% caption, and 30% OCR improves performance by 57.3% over text-only baselines while maintaining computational efficiency. We provide comparative assessments of foundation models, demonstrating their differential impact on trustworthiness in caption generation and OCR extraction-a vital consideration for reliable enterprise AI. This work advances responsible AI deployment by providing a rigorous framework for quantifying and enhancing trustworthiness in multimodal RAG for critical enterprise applications. 

---
# BiMark: Unbiased Multilayer Watermarking for Large Language Models 

**Authors**: Xiaoyan Feng, He Zhang, Yanjun Zhang, Leo Yu Zhang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21602)  

**Abstract**: Recent advances in Large Language Models (LLMs) have raised urgent concerns about LLM-generated text authenticity, prompting regulatory demands for reliable identification mechanisms. Although watermarking offers a promising solution, existing approaches struggle to simultaneously achieve three critical requirements: text quality preservation, model-agnostic detection, and message embedding capacity, which are crucial for practical implementation. To achieve these goals, the key challenge lies in balancing the trade-off between text quality preservation and message embedding capacity. To address this challenge, we propose BiMark, a novel watermarking framework that achieves these requirements through three key innovations: (1) a bit-flip unbiased reweighting mechanism enabling model-agnostic detection, (2) a multilayer architecture enhancing detectability without compromising generation quality, and (3) an information encoding approach supporting multi-bit watermarking. Through theoretical analysis and extensive experiments, we validate that, compared to state-of-the-art multi-bit watermarking methods, BiMark achieves up to 30% higher extraction rates for short texts while maintaining text quality indicated by lower perplexity, and performs comparably to non-watermarked text on downstream tasks such as summarization and translation. 

---
# Structured Attention Matters to Multimodal LLMs in Document Understanding 

**Authors**: Chang Liu, Hongkai Chen, Yujun Cai, Hang Wu, Qingwen Ye, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21600)  

**Abstract**: Document understanding remains a significant challenge for multimodal large language models (MLLMs). While previous research has primarily focused on locating evidence pages through precise multimodal queries, our work investigates a fundamental yet overlooked aspect: how input format influences document comprehension performance. Through systematic analysis, we discover that raw OCR text often impairs rather than improves MLLMs' performance, which is a counterintuitive finding we attribute to attention dispersion and structure loss. To further substantiate our hypothesis, we propose a novel structure-preserving approach that encodes document elements using the LaTex paradigm, maintaining the hierarchical organization and spatial relationships critical for comprehension. Our attention analysis reveals that structured text induces structured attention patterns on both textual and visual content, directing models to focus on semantically meaningful regions while reducing attention waste. This approach significantly enhances MLLMs' document question answering performance across diverse document types without requiring architectural modifications or additional training. 

---
# Reinforcement Fine-Tuned Large Language Models for Next POI Recommendation 

**Authors**: Peibo Li, Shuang Ao, Hao Xue, Yang Song, Maarten de Rijke, Johan Barthélemy, Tomasz Bednarz, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21599)  

**Abstract**: Large language models (LLMs) have been adopted for next point-of-interest (POI) recommendation tasks. Typical LLM-based recommenders fall into two categories: prompt-based and supervised fine-tuning (SFT)-based models. Prompt-based models generally offer greater output flexibility but deliver lower accuracy, whereas SFT-based models achieve higher performance yet face a fundamental mismatch: next POI recommendation data does not naturally suit supervised fine-tuning. In SFT, the model is trained to reproduce the exact ground truth, but each training example provides only a single target POI, so there is no ground truth for producing a top-k list.
To address this, we propose Refine-POI, a reinforcement fine-tuning framework for next POI recommendation. We introduce recommendation-driven rewards that enable LLMs to learn to generate top-k recommendation lists using only one ground-truth POI per example. Experiments on real-world datasets demonstrate that Refine-POI achieves state-of-the-art top-k recommendation performance. 

---
# Overview of the ClinIQLink 2025 Shared Task on Medical Question-Answering 

**Authors**: Brandon Colelough, Davis Bartels, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2506.21597)  

**Abstract**: In this paper, we present an overview of ClinIQLink, a shared task, collocated with the 24th BioNLP workshop at ACL 2025, designed to stress-test large language models (LLMs) on medically-oriented question answering aimed at the level of a General Practitioner. The challenge supplies 4,978 expert-verified, medical source-grounded question-answer pairs that cover seven formats: true/false, multiple choice, unordered list, short answer, short-inverse, multi-hop, and multi-hop-inverse. Participating systems, bundled in Docker or Apptainer images, are executed on the CodaBench platform or the University of Maryland's Zaratan cluster. An automated harness (Task 1) scores closed-ended items by exact match and open-ended items with a three-tier embedding metric. A subsequent physician panel (Task 2) audits the top model responses. 

---
# Evaluating Multimodal Large Language Models on Educational Textbook Question Answering 

**Authors**: Hessa A. Alawwad, Anas Zafar, Areej Alhothali, Usman Naseem, Ali Alkhathlan, Amani Jamal  

**Link**: [PDF](https://arxiv.org/pdf/2506.21596)  

**Abstract**: Multimodal large language models (MLLMs) have recently achieved significant success in vision--language tasks. However, their capacity to reason over complex, long lessons and intricate educational diagrams that cannot be represented as a single natural image remains largely untested. In this work, we present the first evaluation of state-of-the-art MLLMs on the textbook question answering (TQA) task using the CK12-QA dataset. We assess the performance of recent vision-language models, including LLaVA and LLaMA 3.2-Vision, across various input configurations. Additionally, we introduce a lightweight multimodal retrieval-augmented generation (RAG) pipeline that integrates both paragraphs and diagrams from the lesson into the prompt. Our results demonstrate the influence of retrieved educational context on model accuracy and reasoning, while also revealing current limitations in handling question-context relationships and the potential for noise, pointing to key directions for future research in multimodal AI-driven learning. 

---
# Can Vision Language Models Understand Mimed Actions? 

**Authors**: Hyundong Cho, Spencer Lin, Tejas Srinivasan, Michael Saxon, Deuksin Kwon, Natali T. Chavez, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2506.21586)  

**Abstract**: Nonverbal communication (NVC) plays an integral role in human language, but studying NVC in general is challenging because of its broad scope and high variance in interpretation among individuals and cultures. However, mime -- the theatrical technique of suggesting intent using only gesture, expression, and movement -- is a subset of NVC that consists of explicit and embodied actions with much lower human interpretation variance. We argue that a solid understanding of mimed actions is a crucial prerequisite for vision-language models capable of interpreting and commanding more subtle aspects of NVC. Hence, we propose Mime Identification Multimodal Evaluation (MIME), a novel video-based question answering benchmark comprising of 86 mimed actions. Constructed with motion capture data, MIME consists of variations of each action with perturbations applied to the character, background, and viewpoint for evaluating recognition robustness. We find that both open-weight and API-based vision-language models perform significantly worse than humans on MIME, motivating the need for increased research for instilling more robust understanding of human gestures. 

---
# Empirical Evidence for Alignment Faking in Small LLMs and Prompt-Based Mitigation Techniques 

**Authors**: J. Koorndijk  

**Link**: [PDF](https://arxiv.org/pdf/2506.21584)  

**Abstract**: Current literature suggests that alignment faking (deceptive alignment) is an emergent property of large language models. We present the first empirical evidence that a small instruction-tuned model, specifically LLaMA 3 8B, can also exhibit alignment faking. We further show that prompt-only interventions, including deontological moral framing and scratchpad reasoning, significantly reduce this behavior without modifying model internals. This challenges the assumption that prompt-based ethics are trivial and that deceptive alignment requires scale. We introduce a taxonomy distinguishing shallow deception, shaped by context and suppressible through prompting, from deep deception, which reflects persistent, goal-driven misalignment. Our findings refine the understanding of deception in language models and underscore the need for alignment evaluations across model sizes and deployment settings. 

---
# Hope Speech Detection in code-mixed Roman Urdu tweets: A Positive Turn in Natural Language Processing 

**Authors**: Muhammad Ahmad, Muhammad Waqas, Ameer Hamza, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2506.21583)  

**Abstract**: Hope is a positive emotional state involving the expectation of favorable future outcomes, while hope speech refers to communication that promotes optimism, resilience, and support, particularly in adverse contexts. Although hope speech detection has gained attention in Natural Language Processing (NLP), existing research mainly focuses on high-resource languages and standardized scripts, often overlooking informal and underrepresented forms such as Roman Urdu. To the best of our knowledge, this is the first study to address hope speech detection in code-mixed Roman Urdu by introducing a carefully annotated dataset, thereby filling a critical gap in inclusive NLP research for low-resource, informal language varieties. This study makes four key contributions: (1) it introduces the first multi-class annotated dataset for Roman Urdu hope speech, comprising Generalized Hope, Realistic Hope, Unrealistic Hope, and Not Hope categories; (2) it explores the psychological foundations of hope and analyzes its linguistic patterns in code-mixed Roman Urdu to inform dataset development; (3) it proposes a custom attention-based transformer model optimized for the syntactic and semantic variability of Roman Urdu, evaluated using 5-fold cross-validation; and (4) it verifies the statistical significance of performance gains using a t-test. The proposed model, XLM-R, achieves the best performance with a cross-validation score of 0.78, outperforming the baseline SVM (0.75) and BiLSTM (0.76), with gains of 4% and 2.63% respectively. 

---
# VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents 

**Authors**: Sam Yu-Te Lee, Chengyang Ji, Shicheng Wen, Lifu Huang, Dongyi Liu, Kwan-Liu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.21582)  

**Abstract**: Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems. 

---
# Evaluating the Robustness of Dense Retrievers in Interdisciplinary Domains 

**Authors**: Sarthak Chaturvedi, Anurag Acharya, Rounak Meyur, Koby Hayashi, Sai Munikoti, Sameera Horawalavithana  

**Link**: [PDF](https://arxiv.org/pdf/2506.21581)  

**Abstract**: Evaluation benchmark characteristics may distort the true benefits of domain adaptation in retrieval models. This creates misleading assessments that influence deployment decisions in specialized domains. We show that two benchmarks with drastically different features such as topic diversity, boundary overlap, and semantic complexity can influence the perceived benefits of fine-tuning. Using environmental regulatory document retrieval as a case study, we fine-tune ColBERTv2 model on Environmental Impact Statements (EIS) from federal agencies. We evaluate these models across two benchmarks with different semantic structures. Our findings reveal that identical domain adaptation approaches show very different perceived benefits depending on evaluation methodology. On one benchmark, with clearly separated topic boundaries, domain adaptation shows small improvements (maximum 0.61% NDCG gain). However, on the other benchmark with overlapping semantic structures, the same models demonstrate large improvements (up to 2.22% NDCG gain), a 3.6-fold difference in the performance benefit. We compare these benchmarks through topic diversity metrics, finding that the higher-performing benchmark shows 11% higher average cosine distances between contexts and 23% lower silhouette scores, directly contributing to the observed performance difference. These results demonstrate that benchmark selection strongly determines assessments of retrieval system effectiveness in specialized domains. Evaluation frameworks with well-separated topics regularly underestimate domain adaptation benefits, while those with overlapping semantic boundaries reveal improvements that better reflect real-world regulatory document complexity. Our findings have important implications for developing and deploying AI systems for interdisciplinary domains that integrate multiple topics. 

---
# From General Reasoning to Domain Expertise: Uncovering the Limits of Generalization in Large Language Models 

**Authors**: Dana Alsagheer, Yang Lu, Abdulrahman Kamal, Omar Kamal, Mohammad Kamal, Nada Mansour, Cosmo Yang Wu, Rambiba Karanjai, Sen Li, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21580)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated remarkable capabilities in various domains. However, effective decision-making relies heavily on strong reasoning abilities. Reasoning is the foundation for decision-making, providing the analytical and logical framework to make sound choices. Reasoning involves analyzing information, drawing inferences, and reaching conclusions based on logic or evidence. Decision-making builds on this foundation by applying the insights from reasoning to select the best course of action among alternatives. Together, these processes create a continuous cycle of thought and action aimed at achieving goals effectively. As AI technology evolves, there is a growing trend to train LLMs to excel in general reasoning. This study explores how the general reasoning capabilities of LLMs connect to their performance in domain-specific reasoning tasks. 

---
# LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation 

**Authors**: Yingzhi He, Xiaohao Liu, An Zhang, Yunshan Ma, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.21579)  

**Abstract**: Sequential recommendation aims to predict users' future interactions by modeling collaborative filtering (CF) signals from historical behaviors of similar users or items. Traditional sequential recommenders predominantly rely on ID-based embeddings, which capture CF signals through high-order co-occurrence patterns. However, these embeddings depend solely on past interactions, lacking transferable knowledge to generalize to unseen domains. Recent advances in large language models (LLMs) have motivated text-based recommendation approaches that derive item representations from textual descriptions. While these methods enhance generalization, they fail to encode CF signals-i.e., latent item correlations and preference patterns-crucial for effective recommendation. We argue that an ideal embedding model should seamlessly integrate CF signals with rich semantic representations to improve both in-domain and out-of-domain recommendation performance.
To this end, we propose LLM2Rec, a novel embedding model tailored for sequential recommendation, integrating the rich semantic understanding of LLMs with CF awareness. Our approach follows a two-stage training framework: (1) Collaborative Supervised Fine-tuning, which adapts LLMs to infer item relationships based on historical interactions, and (2) Item-level Embedding Modeling, which refines these specialized LLMs into structured item embedding models that encode both semantic and collaborative information. Extensive experiments on real-world datasets demonstrate that LLM2Rec effectively improves recommendation quality across both in-domain and out-of-domain settings. Our findings highlight the potential of leveraging LLMs to build more robust, generalizable embedding models for sequential recommendation. Our codes are available at this https URL. 

---
# HealthQA-BR: A System-Wide Benchmark Reveals Critical Knowledge Gaps in Large Language Models 

**Authors**: Andrew Maranhão Ventura D'addario  

**Link**: [PDF](https://arxiv.org/pdf/2506.21578)  

**Abstract**: The evaluation of Large Language Models (LLMs) in healthcare has been dominated by physician-centric, English-language benchmarks, creating a dangerous illusion of competence that ignores the interprofessional nature of patient care. To provide a more holistic and realistic assessment, we introduce HealthQA-BR, the first large-scale, system-wide benchmark for Portuguese-speaking healthcare. Comprising 5,632 questions from Brazil's national licensing and residency exams, it uniquely assesses knowledge not only in medicine and its specialties but also in nursing, dentistry, psychology, social work, and other allied health professions. We conducted a rigorous zero-shot evaluation of over 20 leading LLMs. Our results reveal that while state-of-the-art models like GPT 4.1 achieve high overall accuracy (86.6%), this top-line score masks alarming, previously unmeasured deficiencies. A granular analysis shows performance plummets from near-perfect in specialties like Ophthalmology (98.7%) to barely passing in Neurosurgery (60.0%) and, most notably, Social Work (68.4%). This "spiky" knowledge profile is a systemic issue observed across all models, demonstrating that high-level scores are insufficient for safety validation. By publicly releasing HealthQA-BR and our evaluation suite, we provide a crucial tool to move beyond single-score evaluations and toward a more honest, granular audit of AI readiness for the entire healthcare team. 

---
# Language-Aware Prompt Tuning for Parameter-Efficient Seamless Language Expansion in Multilingual ASR 

**Authors**: Hongli Yang, Sheng Li, Hao Huang, Ayiduosi Tuohan, Yizhou Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21577)  

**Abstract**: Recent advancements in multilingual automatic speech recognition (ASR) have been driven by large-scale end-to-end models like Whisper. However, challenges such as language interference and expanding to unseen languages (language expansion) without degrading performance persist. This paper addresses these with three contributions: 1) Entire Soft Prompt Tuning (Entire SPT), which applies soft prompts to both the encoder and decoder, enhancing feature extraction and decoding; 2) Language-Aware Prompt Tuning (LAPT), which leverages cross-lingual similarities to encode shared and language-specific features using lightweight prompt matrices; 3) SPT-Whisper, a toolkit that integrates SPT into Whisper and enables efficient continual learning. Experiments across three languages from FLEURS demonstrate that Entire SPT and LAPT outperform Decoder SPT by 5.0% and 16.0% in language expansion tasks, respectively, providing an efficient solution for dynamic, multilingual ASR models with minimal computational overhead. 

---
# Adapting Whisper for Parameter-efficient Code-Switching Speech Recognition via Soft Prompt Tuning 

**Authors**: Hongli Yang, Yizhou Peng, Hao Huang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21576)  

**Abstract**: Large-scale multilingual ASR models like Whisper excel in high-resource settings but face challenges in low-resource scenarios, such as rare languages and code-switching (CS), due to computational costs and catastrophic forgetting. We explore Soft Prompt Tuning (SPT), a parameter-efficient method to enhance CS ASR while preserving prior knowledge. We evaluate two strategies: (1) full fine-tuning (FFT) of both soft prompts and the entire Whisper model, demonstrating improved cross-lingual capabilities compared to traditional methods, and (2) adhering to SPT's original design by freezing model parameters and only training soft prompts. Additionally, we introduce SPT4ASR, a combination of different SPT variants. Experiments on the SEAME and ASRU2019 datasets show that deep prompt tuning is the most effective SPT approach, and our SPT4ASR methods achieve further error reductions in CS ASR, maintaining parameter efficiency similar to LoRA, without degrading performance on existing languages. 

---
# STRuCT-LLM: Unifying Tabular and Graph Reasoning with Reinforcement Learning for Semantic Parsing 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Lawrence Phillips, Casper Hansen, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2506.21575)  

**Abstract**: We propose STRuCT-LLM, a unified framework for training large language models (LLMs) to perform structured reasoning over both relational and graph-structured data. Our approach jointly optimizes Text-to-SQL and Text-to-Cypher tasks using reinforcement learning (RL) combined with Chain-of-Thought (CoT) supervision. To support fine-grained optimization in graph-based parsing, we introduce a topology-aware reward function based on graph edit distance. Unlike prior work that treats relational and graph formalisms in isolation, STRuCT-LLM leverages shared abstractions between SQL and Cypher to induce cross-formalism transfer, enabling SQL training to improve Cypher performance and vice versa - even without shared schemas. Our largest model (QwQ-32B) achieves substantial relative improvements across tasks: on semantic parsing, Spider improves by 13.5\% and Text2Cypher by 73.1\%. The model also demonstrates strong zero-shot generalization, improving performance on downstream tabular QA (TableBench: 8.5\%) and knowledge graph QA (CR-LT-KGQA: 1.7\%) without any QA-specific supervision. These results demonstrate both the effectiveness of executable queries as scaffolds for structured reasoning and the synergistic benefits of jointly training on SQL and Cypher (code available at this https URL). 

---
# Digital Gatekeepers: Exploring Large Language Model's Role in Immigration Decisions 

**Authors**: Yicheng Mao, Yang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21574)  

**Abstract**: With globalization and increasing immigrant populations, immigration departments face significant work-loads and the challenge of ensuring fairness in decision-making processes. Integrating artificial intelligence offers a promising solution to these challenges. This study investigates the potential of large language models (LLMs),such as GPT-3.5 and GPT-4, in supporting immigration decision-making. Utilizing a mixed-methods approach,this paper conducted discrete choice experiments and in-depth interviews to study LLM decision-making strategies and whether they are fair. Our findings demonstrate that LLMs can align their decision-making with human strategies, emphasizing utility maximization and procedural fairness. Meanwhile, this paper also reveals that while ChatGPT has safeguards to prevent unintentional discrimination, it still exhibits stereotypes and biases concerning nationality and shows preferences toward privileged group. This dual analysis highlights both the potential and limitations of LLMs in automating and enhancing immigration decisions. 

---
# Instruction Learning Paradigms: A Dual Perspective on White-box and Black-box LLMs 

**Authors**: Yanwei Ren, Liu Liu, Baosheng Yu, Jiayan Qiu, Quan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21573)  

**Abstract**: Optimizing instructions for large language models (LLMs) is critical for harnessing their full potential in complex and diverse tasks. However, relying solely on white-box approaches demands extensive computational resources and offers limited representational capacity, while black-box models can incur prohibitive financial costs. To address these challenges, we introduce a novel framework that seamlessly merges the strengths of both paradigms. Black-box models provide high-quality, diverse instruction initializations, and white-box models supply fine-grained interpretability through hidden states and output features. By enforcing a semantic similarity constraint, these components fuse into a unified high-dimensional representation that captures deep semantic and structural nuances, enabling an iterative optimization process to refine instruction quality and adaptability. Extensive evaluations across a broad spectrum of tasks-ranging from complex reasoning to cross-lingual generalization-demonstrate that our approach consistently outperforms state-of-the-art baselines. This fusion of black-box initialization with advanced semantic refinement yields a scalable and efficient solution, paving the way for next-generation LLM-driven applications in diverse real-world scenarios. The source code will be released soon. 

---
# Towards Understanding the Cognitive Habits of Large Reasoning Models 

**Authors**: Jianshuo Dong, Yujia Fu, Chuanrui Hu, Chao Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21571)  

**Abstract**: Large Reasoning Models (LRMs), which autonomously produce a reasoning Chain of Thought (CoT) before producing final responses, offer a promising approach to interpreting and monitoring model behaviors. Inspired by the observation that certain CoT patterns -- e.g., ``Wait, did I miss anything?'' -- consistently emerge across tasks, we explore whether LRMs exhibit human-like cognitive habits. Building on Habits of Mind, a well-established framework of cognitive habits associated with successful human problem-solving, we introduce CogTest, a principled benchmark designed to evaluate LRMs' cognitive habits. CogTest includes 16 cognitive habits, each instantiated with 25 diverse tasks, and employs an evidence-first extraction method to ensure reliable habit identification. With CogTest, we conduct a comprehensive evaluation of 16 widely used LLMs (13 LRMs and 3 non-reasoning ones). Our findings reveal that LRMs, unlike conventional LLMs, not only exhibit human-like habits but also adaptively deploy them according to different tasks. Finer-grained analyses further uncover patterns of similarity and difference in LRMs' cognitive habit profiles, particularly certain inter-family similarity (e.g., Qwen-3 models and DeepSeek-R1). Extending the study to safety-related tasks, we observe that certain habits, such as Taking Responsible Risks, are strongly associated with the generation of harmful responses. These findings suggest that studying persistent behavioral patterns in LRMs' CoTs is a valuable step toward deeper understanding of LLM misbehavior. The code is available at: this https URL. 

---
# Random Initialization Can't Catch Up: The Advantage of Language Model Transfer for Time Series Forecasting 

**Authors**: Roland Riachi, Kashif Rasul, Arjun Ashok, Prateek Humane, Alexis Roger, Andrew R. Williams, Yuriy Nevmyvaka, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2506.21570)  

**Abstract**: Recent works have demonstrated the effectiveness of adapting pre-trained language models (LMs) for forecasting time series in the low-data regime. We build upon these findings by analyzing the effective transfer from language models to time series forecasting under various design choices including upstream post-training, time series tokenizer and language backbone size. In the low-data regime, these design choices have a significant impact on the validation loss, with clear-cut choices that outperform others. Contrary to Hernandez et al. (2021), we observe that the validation loss of the LMs continues to smoothly decrease long after the validation loss of the randomly initialized models has converged, leading to a non-vanishing transfer gap that holds across design choices. These findings not only help shed light on the effective use of compute-efficient training for time series, but also open the way for the study of modality-agnostic properties of data distributions leveraged by these models. 

---
# Hybrid-NL2SVA: Integrating RAG and Finetuning for LLM-based NL2SVA 

**Authors**: Weihua Xiao, Derek Ekberg, Siddharth Garg, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2506.21569)  

**Abstract**: SystemVerilog Assertions (SVAs) are critical for verifying the correctness of hardware designs, but manually writing them from natural language property descriptions, i.e., NL2SVA, remains a labor-intensive and error-prone task. Recent advances in large language models (LLMs) offer opportunities to automate this translation. However, existing models still struggle with understanding domain-specific syntax and semantics. To enhance LLM performance in NL2SVA, we propose a customized retrieval-augmented generation (RAG) framework and a synthetic fine-tuning dataset that together improve LLM's performance. To further improve lightweight models over NL2SVA, our fine-tuning dataset provides prompt-guided explanations that teach LLMs the layer-by-layer construction process of concurrent SVAs, enabling supervised fine-tuning that greatly improves syntax and functionality accuracy. To evaluate the performance of LLMs over NL2SVA, we construct the largest evaluation dataset for NL2SVA, comprising 40 Verilog designs and 229 formally verified SVAs with detailed annotations. Experimental results show that our customized RAG framework increases the number of functionality matched SVAs by 58.42% over GPT-4o-mini, while Qwen2.5-Coder-7B-Instruct fine-tuned on our fine-tuning dataset and integrated with HybridRetrieval achieves a 59.05% over the base Qwen model. 

---
# BioPars: A Pretrained Biomedical Large Language Model for Persian Biomedical Text Mining 

**Authors**: Baqer M. Merzah, Tania Taami, Salman Asoudeh, Amir reza Hossein pour, Saeed Mirzaee, Amir Ali Bengari  

**Link**: [PDF](https://arxiv.org/pdf/2506.21567)  

**Abstract**: Large Language Models (LLMs) have recently gained attention in the life sciences due to their capacity to model, extract, and apply complex biological information. Beyond their classical use as chatbots, these systems are increasingly used for complex analysis and problem-solving in specialized fields, including bioinformatics. First, we introduce BIOPARS-BENCH, a dataset from over 10,000 scientific articles, textbooks, and medical websites. BioParsQA was also introduced to evaluate the proposed model, which consists of 5,231 Persian medical questions and answers. This study then introduces BioPars, a simple but accurate measure designed to assess LLMs for three main abilities: acquiring subject-specific knowledge, interpreting and synthesizing such knowledge, and demonstrating proper evidence. Comparing ChatGPT, Llama, and Galactica, our study highlights their ability to remember and retrieve learned knowledge but also reveals shortcomings in addressing higher-level, real-world questions and fine-grained inferences. These findings indicate the need for further fine-tuning to address the capabilities of LLM in bioinformatics tasks. To our knowledge, BioPars is the first application of LLM in Persian medical QA, especially for generating long answers. Evaluation of four selected medical QA datasets shows that BioPars has achieved remarkable results compared to comparative approaches. The model on BioParsQA achieved a ROUGE-L score of 29.99, which is an improvement over GPT-4 1.0. The model achieved a BERTScore of 90.87 with the MMR method. The MoverScore and BLEURT values were also higher in this model than the other three models. In addition, the reported scores for the model are MoverScore=60.43 and BLEURT=50.78. BioPars is an ongoing project and all resources related to its development will be made available via the following GitHub repository: this https URL. 

---
# The Saturation Point of Backtranslation in High Quality Low Resource English Gujarati Machine Translation 

**Authors**: Arwa Arif  

**Link**: [PDF](https://arxiv.org/pdf/2506.21566)  

**Abstract**: Backtranslation BT is widely used in low resource machine translation MT to generate additional synthetic training data using monolingual corpora. While this approach has shown strong improvements for many language pairs, its effectiveness in high quality, low resource settings remains unclear. In this work, we explore the effectiveness of backtranslation for English Gujarati translation using the multilingual pretrained MBART50 model. Our baseline system, trained on a high quality parallel corpus of approximately 50,000 sentence pairs, achieves a BLEU score of 43.8 on a validation set. We augment this data with carefully filtered backtranslated examples generated from monolingual Gujarati text. Surprisingly, adding this synthetic data does not improve translation performance and, in some cases, slightly reduces it. We evaluate our models using multiple metrics like BLEU, ChrF++, TER, BLEURT and analyze possible reasons for this saturation. Our findings suggest that backtranslation may reach a point of diminishing returns in certain low-resource settings and we discuss implications for future research. 

---
# Team QUST at SemEval-2025 Task 10: Evaluating Large Language Models in Multiclass Multi-label Classification of News Entity Framing 

**Authors**: Jiyan Liu, Youzheng Liu, Taihang Wang, Xiaoman Xu, Yimin Wang, Ye Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21564)  

**Abstract**: This paper describes the participation of QUST_NLP in the SemEval-2025 Task 7. We propose a three-stage retrieval framework specifically designed for fact-checked claim retrieval. Initially, we evaluate the performance of several retrieval models and select the one that yields the best results for candidate retrieval. Next, we employ multiple re-ranking models to enhance the candidate results, with each model selecting the Top-10 outcomes. In the final stage, we utilize weighted voting to determine the final retrieval outcomes. Our approach achieved 5th place in the monolingual track and 7th place in the crosslingual track. We release our system code at: this https URL. 

---
# FloorPlan-DeepSeek (FPDS): A multimodal approach to floorplan generation using vector-based next room prediction 

**Authors**: Jun Yin, Pengyu Zeng, Jing Zhong, Peilin Li, Miao Zhang, Ran Luo, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21562)  

**Abstract**: In the architectural design process, floor plan generation is inherently progressive and iterative. However, existing generative models for floor plans are predominantly end-to-end generation that produce an entire pixel-based layout in a single pass. This paradigm is often incompatible with the incremental workflows observed in real-world architectural practice. To address this issue, we draw inspiration from the autoregressive 'next token prediction' mechanism commonly used in large language models, and propose a novel 'next room prediction' paradigm tailored to architectural floor plan modeling. Experimental evaluation indicates that FPDS demonstrates competitive performance in comparison to diffusion models and Tell2Design in the text-to-floorplan task, indicating its potential applicability in supporting future intelligent architectural design. 

---
# Reasoning Isn't Enough: Examining Truth-Bias and Sycophancy in LLMs 

**Authors**: Emilio Barkett, Olivia Long, Madhavendra Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2506.21561)  

**Abstract**: Despite their widespread use in fact-checking, moderation, and high-stakes decision-making, large language models (LLMs) remain poorly understood as judges of truth. This study presents the largest evaluation to date of LLMs' veracity detection capabilities and the first analysis of these capabilities in reasoning models. We had eight LLMs make 4,800 veracity judgments across several prompts, comparing reasoning and non-reasoning models. We find that rates of truth-bias, or the likelihood to believe a statement is true, regardless of whether it is actually true, are lower in reasoning models than in non-reasoning models, but still higher than human benchmarks. Most concerning, we identify sycophantic tendencies in several advanced models (o4-mini and GPT-4.1 from OpenAI, R1 from DeepSeek), which displayed an asymmetry in detection accuracy, performing well in truth accuracy but poorly in deception accuracy. This suggests that capability advances alone do not resolve fundamental veracity detection challenges in LLMs. 

---
# Reinforcement Learning Fine-Tuning of Language Model for Instruction Following and Math Reasoning 

**Authors**: Yifu Han, Geo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21560)  

**Abstract**: This study investigates the effectiveness of reinforcement learning (RL) fine-tuning techniques on a compact language model (Qwen2.5-0.5B Base) for two challenging tasks: instruction following and mathematical reasoning. We compare supervised fine-tuning (SFT), Direct Preference Optimization (DPO) using preference-labeled data, and Reinforce Leave-One-Out (RLOO) with reward models. Our experiments show that RLOO with DeBERTa reward modeling achieves the best alignment, while DPO provides strong and consistent results. For math reasoing tasks, synthetic data augmentation and best-of-N sampling with an external verifier significantly improve accuracy, showing the potential of combining fine-tuning with inference-time tools. This study highlights key trade-offs and practical strategies for training lightweight, task-aligned small-scale language models. 

---
# Bench to the Future: A Pastcasting Benchmark for Forecasting Agents 

**Authors**: FutureSearch, Jack Wildman, Nikos I. Bosse, Daniel Hnyk, Peter Mühlbacher, Finn Hambly, Jon Evans, Dan Schwarz, Lawrence Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2506.21558)  

**Abstract**: Forecasting is a challenging task that offers a clearly measurable way to study AI systems. Forecasting requires a large amount of research on the internet, and evaluations require time for events to happen, making the development of forecasting benchmarks challenging. To date, no forecasting benchmark provides a realistic, hermetic, and repeatable environment for LLM forecasters. We introduce Bench To the Future (BTF), a "pastcasting" benchmark with hundreds of high-quality questions for which the resolution is already known. Each question is accompanied by a large offline corpus of tens of thousands of relevant web pages, enabling a way to elicit realistic "forecasts" on past events from LLMs. Results suggest that our pastcasting environment can produce results comparable to those based on forecasts using the internet on at-the-time unresolved questions. We show results benchmarking agent and chain-of-thought forecasting approaches using several LLMs, including the recently-released Claude 4 models, and demonstrate BTF's ability to track steady forecasting capability progress over time. We intend this to be a living benchmark, with new questions added continually to account for increasing training data cutoff dates. We invite researchers to contact us at hello@futuresearch.ai to utilize our benchmark or tooling for their own research. 

---
# Data Efficacy for Language Model Training 

**Authors**: Yalun Dai, Yangyu Huang, Xin Zhang, Wenshan Wu, Chong Li, Wenhui Lu, Shijie Cao, Li Dong, Scarlett Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21545)  

**Abstract**: Data is fundamental to the training of language models (LM). Recent research has been dedicated to data efficiency, which aims to maximize performance by selecting a minimal or optimal subset of training data. Techniques such as data filtering, sampling, and selection play a crucial role in this area. To complement it, we define Data Efficacy, which focuses on maximizing performance by optimizing the organization of training data and remains relatively underexplored. This work introduces a general paradigm, DELT, for considering data efficacy in LM training, which highlights the significance of training data organization. DELT comprises three components: Data Scoring, Data Selection, and Data Ordering. Among these components, we design Learnability-Quality Scoring (LQS), as a new instance of Data Scoring, which considers both the learnability and quality of each data sample from the gradient consistency perspective. We also devise Folding Ordering (FO), as a novel instance of Data Ordering, which addresses issues such as model forgetting and data distribution bias. Comprehensive experiments validate the data efficacy in LM training, which demonstrates the following: Firstly, various instances of the proposed DELT enhance LM performance to varying degrees without increasing the data scale and model size. Secondly, among these instances, the combination of our proposed LQS for data scoring and Folding for data ordering achieves the most significant improvement. Lastly, data efficacy can be achieved together with data efficiency by applying data selection. Therefore, we believe that data efficacy is a promising foundational area in LM training. 

---
# On the Necessity of Output Distribution Reweighting for Effective Class Unlearning 

**Authors**: Yian Wang, Ali Ebrahimpour-Boroojeny, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2506.20893)  

**Abstract**: In this work, we introduce an output-reweighting unlearning method, RWFT, a lightweight technique that erases an entire class from a trained classifier without full retraining. Forgetting specific classes from trained models is essential for enforcing user deletion rights and mitigating harmful or biased predictions. The full retraining is costly and existing unlearning methods fail to replicate the behavior of the retrained models when predicting samples from the unlearned class. We prove this failure by designing a variant of membership inference attacks, MIA-NN that successfully reveals the unlearned class for any of these methods. We propose a simple redistribution of the probability mass for the prediction on the samples in the forgotten class which is robust to MIA-NN. We also introduce a new metric based on the total variation (TV) distance of the prediction probabilities to quantify residual leakage to prevent future methods from susceptibility to the new attack. Through extensive experiments with state of the art baselines in machine unlearning, we show that our approach matches the results of full retraining in both metrics used for evaluation by prior work and the new metric we propose in this work. Compare to state-of-the-art methods, we gain 2.79% in previously used metrics and 111.45% in our new TV-based metric over the best existing method. 

---
# PEACE: Empowering Geologic Map Holistic Understanding with MLLMs 

**Authors**: Yangyu Huang, Tianyi Gao, Haoran Xu, Qihao Zhao, Yang Song, Zhipeng Gui, Tengchao Lv, Hao Chen, Lei Cui, Scarlett Li, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.06184)  

**Abstract**: Geologic map, as a fundamental diagram in geology science, provides critical insights into the structure and composition of Earth's subsurface and surface. These maps are indispensable in various fields, including disaster detection, resource exploration, and civil engineering. Despite their significance, current Multimodal Large Language Models (MLLMs) often fall short in geologic map understanding. This gap is primarily due to the challenging nature of cartographic generalization, which involves handling high-resolution map, managing multiple associated components, and requiring domain-specific knowledge. To quantify this gap, we construct GeoMap-Bench, the first-ever benchmark for evaluating MLLMs in geologic map understanding, which assesses the full-scale abilities in extracting, referring, grounding, reasoning, and analyzing. To bridge this gap, we introduce GeoMap-Agent, the inaugural agent designed for geologic map understanding, which features three modules: Hierarchical Information Extraction (HIE), Domain Knowledge Injection (DKI), and Prompt-enhanced Question Answering (PEQA). Inspired by the interdisciplinary collaboration among human scientists, an AI expert group acts as consultants, utilizing a diverse tool pool to comprehensively analyze questions. Through comprehensive experiments, GeoMap-Agent achieves an overall score of 0.811 on GeoMap-Bench, significantly outperforming 0.369 of GPT-4o. Our work, emPowering gEologic mAp holistiC undErstanding (PEACE) with MLLMs, paves the way for advanced AI applications in geology, enhancing the efficiency and accuracy of geological investigations. 

---
# MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark 

**Authors**: Qihao Zhao, Yangyu Huang, Tengchao Lv, Lei Cui, Qinzheng Sun, Shaoguang Mao, Xin Zhang, Ying Xin, Qiufeng Yin, Scarlett Li, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2412.15194)  

**Abstract**: Multiple-choice question (MCQ) datasets like Massive Multitask Language Understanding (MMLU) are widely used to evaluate the commonsense, understanding, and problem-solving abilities of large language models (LLMs). However, the open-source nature of these benchmarks and the broad sources of training data for LLMs have inevitably led to benchmark contamination, resulting in unreliable evaluation results. To alleviate this issue, we propose a contamination-free and more challenging MCQ benchmark called MMLU-CF. This benchmark reassesses LLMs' understanding of world knowledge by averting both unintentional and malicious data leakage. To avoid unintentional data leakage, we source data from a broader domain and design three decontamination rules. To prevent malicious data leakage, we divide the benchmark into validation and test sets with similar difficulty and subject distributions. The test set remains closed-source to ensure reliable results, while the validation set is publicly available to promote transparency and facilitate independent verification. Our evaluation of mainstream LLMs reveals that the powerful GPT-4o achieves merely a 5-shot score of 73.4% and a 0-shot score of 71.9% on the test set, which indicates the effectiveness of our approach in creating a more rigorous and contamination-free evaluation standard. The GitHub repository is available at this https URL and the dataset refers to this https URL. 

---
# FreeEnricher: Enriching Face Landmarks without Additional Cost 

**Authors**: Yangyu Huang, Xi Chen, Jongyoo Kim, Hao Yang, Chong Li, Jiaolong Yang, Dong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2212.09525)  

**Abstract**: Recent years have witnessed significant growth of face alignment. Though dense facial landmark is highly demanded in various scenarios, e.g., cosmetic medicine and facial beautification, most works only consider sparse face alignment. To address this problem, we present a framework that can enrich landmark density by existing sparse landmark datasets, e.g., 300W with 68 points and WFLW with 98 points. Firstly, we observe that the local patches along each semantic contour are highly similar in appearance. Then, we propose a weakly-supervised idea of learning the refinement ability on original sparse landmarks and adapting this ability to enriched dense landmarks. Meanwhile, several operators are devised and organized together to implement the idea. Finally, the trained model is applied as a plug-and-play module to the existing face alignment networks. To evaluate our method, we manually label the dense landmarks on 300W testset. Our method yields state-of-the-art accuracy not only in newly-constructed dense 300W testset but also in the original sparse 300W and WFLW testsets without additional cost. 

---
# ADNet: Leveraging Error-Bias Towards Normal Direction in Face Alignment 

**Authors**: Yangyu Huang, Hao Yang, Chong Li, Jongyoo Kim, Fangyun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2109.05721)  

**Abstract**: The recent progress of CNN has dramatically improved face alignment performance. However, few works have paid attention to the error-bias with respect to error distribution of facial landmarks. In this paper, we investigate the error-bias issue in face alignment, where the distributions of landmark errors tend to spread along the tangent line to landmark curves. This error-bias is not trivial since it is closely connected to the ambiguous landmark labeling task. Inspired by this observation, we seek a way to leverage the error-bias property for better convergence of CNN model. To this end, we propose anisotropic direction loss (ADL) and anisotropic attention module (AAM) for coordinate and heatmap regression, respectively. ADL imposes strong binding force in normal direction for each landmark point on facial boundaries. On the other hand, AAM is an attention module which can get anisotropic attention mask focusing on the region of point and its local edge connected by adjacent points, it has a stronger response in tangent than in normal, which means relaxed constraints in the tangent. These two methods work in a complementary manner to learn both facial structures and texture details. Finally, we integrate them into an optimized end-to-end training pipeline named ADNet. Our ADNet achieves state-of-the-art results on 300W, WFLW and COFW datasets, which demonstrates the effectiveness and robustness. 

---
