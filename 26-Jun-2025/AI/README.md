# The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind 

**Authors**: Andrei Lupu, Timon Willi, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2506.20664)  

**Abstract**: As Large Language Models (LLMs) gain agentic abilities, they will have to navigate complex multi-agent scenarios, interacting with human users and other agents in cooperative and competitive settings. This will require new reasoning skills, chief amongst them being theory of mind (ToM), or the ability to reason about the "mental" states of other agents. However, ToM and other multi-agent abilities in LLMs are poorly understood, since existing benchmarks suffer from narrow scope, data leakage, saturation, and lack of interactivity. We thus propose Decrypto, a game-based benchmark for multi-agent reasoning and ToM drawing inspiration from cognitive science, computational pragmatics and multi-agent reinforcement learning. It is designed to be as easy as possible in all other dimensions, eliminating confounding factors commonly found in other benchmarks. To our knowledge, it is also the first platform for designing interactive ToM experiments.
We validate the benchmark design through comprehensive empirical evaluations of frontier LLMs, robustness studies, and human-AI cross-play experiments. We find that LLM game-playing abilities lag behind humans and simple word-embedding baselines. We then create variants of two classic cognitive science experiments within Decrypto to evaluate three key ToM abilities. Surprisingly, we find that state-of-the-art reasoning models are significantly worse at those tasks than their older counterparts. This demonstrates that Decrypto addresses a crucial gap in current reasoning and ToM evaluations, and paves the path towards better artificial agents. 

---
# Towards Community-Driven Agents for Machine Learning Engineering 

**Authors**: Sijie Li, Weiwei Sun, Shanda Li, Ameet Talwalkar, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20640)  

**Abstract**: Large language model-based machine learning (ML) agents have shown great promise in automating ML research. However, existing agents typically operate in isolation on a given research problem, without engaging with the broader research community, where human researchers often gain insights and contribute by sharing knowledge. To bridge this gap, we introduce MLE-Live, a live evaluation framework designed to assess an agent's ability to communicate with and leverage collective knowledge from a simulated Kaggle research community. Building on this framework, we propose CoMind, a novel agent that excels at exchanging insights and developing novel solutions within a community context. CoMind achieves state-of-the-art performance on MLE-Live and outperforms 79.2% human competitors on average across four ongoing Kaggle competitions. Our code is released at this https URL. 

---
# AI Assistants to Enhance and Exploit the PETSc Knowledge Base 

**Authors**: Barry Smith, Junchao Zhang, Hong Zhang, Lois Curfman McInnes, Murat Keceli, Archit Vasan, Satish Balay, Toby Isaac, Le Chen, Venkatram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2506.20608)  

**Abstract**: Generative AI, especially through large language models (LLMs), is transforming how technical knowledge can be accessed, reused, and extended. PETSc, a widely used numerical library for high-performance scientific computing, has accumulated a rich but fragmented knowledge base over its three decades of development, spanning source code, documentation, mailing lists, GitLab issues, Discord conversations, technical papers, and more. Much of this knowledge remains informal and inaccessible to users and new developers. To activate and utilize this knowledge base more effectively, the PETSc team has begun building an LLM-powered system that combines PETSc content with custom LLM tools -- including retrieval-augmented generation (RAG), reranking algorithms, and chatbots -- to assist users, support developers, and propose updates to formal documentation. This paper presents initial experiences designing and evaluating these tools, focusing on system architecture, using RAG and reranking for PETSc-specific information, evaluation methodologies for various LLMs and embedding models, and user interface design. Leveraging the Argonne Leadership Computing Facility resources, we analyze how LLM responses can enhance the development and use of numerical software, with an initial focus on scalable Krylov solvers. Our goal is to establish an extensible framework for knowledge-centered AI in scientific software, enabling scalable support, enriched documentation, and enhanced workflows for research and development. We conclude by outlining directions for expanding this system into a robust, evolving platform that advances software ecosystems to accelerate scientific discovery. 

---
# CogGen: A Learner-Centered Generative AI Architecture for Intelligent Tutoring with Programming Video 

**Authors**: Wengxi Li, Roy Pea, Nick Haber, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20600)  

**Abstract**: We introduce CogGen, a learner-centered AI architecture that transforms programming videos into interactive, adaptive learning experiences by integrating student modeling with generative AI tutoring based on the Cognitive Apprenticeship framework. The architecture consists of three components: (1) video segmentation by learning goals, (2) a conversational tutoring engine applying Cognitive Apprenticeship strategies, and (3) a student model using Bayesian Knowledge Tracing to adapt instruction. Our technical evaluation demonstrates effective video segmentation accuracy and strong pedagogical alignment across knowledge, method, action, and interaction layers. Ablation studies confirm the necessity of each component in generating effective guidance. This work advances AI-powered tutoring by bridging structured student modeling with interactive AI conversations, offering a scalable approach to enhancing video-based programming education. 

---
# Fine-Tuning and Prompt Engineering of LLMs, for the Creation of Multi-Agent AI for Addressing Sustainable Protein Production Challenges 

**Authors**: Alexander D. Kalian, Jaewook Lee, Stefan P. Johannesson, Lennart Otte, Christer Hogstrand, Miao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.20598)  

**Abstract**: The global demand for sustainable protein sources has accelerated the need for intelligent tools that can rapidly process and synthesise domain-specific scientific knowledge. In this study, we present a proof-of-concept multi-agent Artificial Intelligence (AI) framework designed to support sustainable protein production research, with an initial focus on microbial protein sources. Our Retrieval-Augmented Generation (RAG)-oriented system consists of two GPT-based LLM agents: (1) a literature search agent that retrieves relevant scientific literature on microbial protein production for a specified microbial strain, and (2) an information extraction agent that processes the retrieved content to extract relevant biological and chemical information. Two parallel methodologies, fine-tuning and prompt engineering, were explored for agent optimisation. Both methods demonstrated effectiveness at improving the performance of the information extraction agent in terms of transformer-based cosine similarity scores between obtained and ideal outputs. Mean cosine similarity scores were increased by up to 25%, while universally reaching mean scores of $\geq 0.89$ against ideal output text. Fine-tuning overall improved the mean scores to a greater extent (consistently of $\geq 0.94$) compared to prompt engineering, although lower statistical uncertainties were observed with the latter approach. A user interface was developed and published for enabling the use of the multi-agent AI system, alongside preliminary exploration of additional chemical safety-based search capabilities 

---
# Case-based Reasoning Augmented Large Language Model Framework for Decision Making in Realistic Safety-Critical Driving Scenarios 

**Authors**: Wenbin Gan, Minh-Son Dao, Koji Zettsu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20531)  

**Abstract**: Driving in safety-critical scenarios requires quick, context-aware decision-making grounded in both situational understanding and experiential reasoning. Large Language Models (LLMs), with their powerful general-purpose reasoning capabilities, offer a promising foundation for such decision-making. However, their direct application to autonomous driving remains limited due to challenges in domain adaptation, contextual grounding, and the lack of experiential knowledge needed to make reliable and interpretable decisions in dynamic, high-risk environments. To address this gap, this paper presents a Case-Based Reasoning Augmented Large Language Model (CBR-LLM) framework for evasive maneuver decision-making in complex risk scenarios. Our approach integrates semantic scene understanding from dashcam video inputs with the retrieval of relevant past driving cases, enabling LLMs to generate maneuver recommendations that are both context-sensitive and human-aligned. Experiments across multiple open-source LLMs show that our framework improves decision accuracy, justification quality, and alignment with human expert behavior. Risk-aware prompting strategies further enhance performance across diverse risk types, while similarity-based case retrieval consistently outperforms random sampling in guiding in-context learning. Case studies further demonstrate the framework's robustness in challenging real-world conditions, underscoring its potential as an adaptive and trustworthy decision-support tool for intelligent driving systems. 

---
# Engineering Sentience 

**Authors**: Konstantin Demin, Taylor Webb, Eric Elmoznino, Hakwan Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.20504)  

**Abstract**: We spell out a definition of sentience that may be useful for designing and building it in machines. We propose that for sentience to be meaningful for AI, it must be fleshed out in functional, computational terms, in enough detail to allow for implementation. Yet, this notion of sentience must also reflect something essentially 'subjective', beyond just having the general capacity to encode perceptual content. For this specific functional notion of sentience to occur, we propose that certain sensory signals need to be both assertoric (persistent) and qualitative. To illustrate the definition in more concrete terms, we sketch out some ways for potential implementation, given current technology. Understanding what it takes for artificial agents to be functionally sentient can also help us avoid creating them inadvertently, or at least, realize that we have created them in a timely manner. 

---
# Mixtures of Neural Cellular Automata: A Stochastic Framework for Growth Modelling and Self-Organization 

**Authors**: Salvatore Milite, Giulio Caravagna, Andrea Sottoriva  

**Link**: [PDF](https://arxiv.org/pdf/2506.20486)  

**Abstract**: Neural Cellular Automata (NCAs) are a promising new approach to model self-organizing processes, with potential applications in life science. However, their deterministic nature limits their ability to capture the stochasticity of real-world biological and physical systems.
We propose the Mixture of Neural Cellular Automata (MNCA), a novel framework incorporating the idea of mixture models into the NCA paradigm. By combining probabilistic rule assignments with intrinsic noise, MNCAs can model diverse local behaviors and reproduce the stochastic dynamics observed in biological processes.
We evaluate the effectiveness of MNCAs in three key domains: (1) synthetic simulations of tissue growth and differentiation, (2) image morphogenesis robustness, and (3) microscopy image segmentation. Results show that MNCAs achieve superior robustness to perturbations, better recapitulate real biological growth patterns, and provide interpretable rule segmentation. These findings position MNCAs as a promising tool for modeling stochastic dynamical systems and studying self-growth processes. 

---
# GymPN: A Library for Decision-Making in Process Management Systems 

**Authors**: Riccardo Lo Bianco, Willem van Jaarsveld, Remco Dijkman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20404)  

**Abstract**: Process management systems support key decisions about the way work is allocated in organizations. This includes decisions on which task to perform next, when to execute the task, and who to assign the task to. Suitable software tools are required to support these decisions in a way that is optimal for the organization. This paper presents a software library, called GymPN, that supports optimal decision-making in business processes using Deep Reinforcement Learning. GymPN builds on previous work that supports task assignment in business processes, introducing two key novelties: support for partial process observability and the ability to model multiple decisions in a business process. These novel elements address fundamental limitations of previous work and thus enable the representation of more realistic process decisions. We evaluate the library on eight typical business process decision-making problem patterns, showing that GymPN allows for easy modeling of the desired problems, as well as learning optimal decision policies. 

---
# Smart Ride and Delivery Services with Electric Vehicles: Leveraging Bidirectional Charging for Profit Optimisation 

**Authors**: Jinchun Du, Bojie Shen, Muhammad Aamir Cheema, Adel N. Toosi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20401)  

**Abstract**: With the rising popularity of electric vehicles (EVs), modern service systems, such as ride-hailing delivery services, are increasingly integrating EVs into their operations. Unlike conventional vehicles, EVs often have a shorter driving range, necessitating careful consideration of charging when fulfilling requests. With recent advances in Vehicle-to-Grid (V2G) technology - allowing EVs to also discharge energy back to the grid - new opportunities and complexities emerge. We introduce the Electric Vehicle Orienteering Problem with V2G (EVOP-V2G): a profit-maximization problem where EV drivers must select customer requests or orders while managing when and where to charge or discharge. This involves navigating dynamic electricity prices, charging station selection, and route constraints. We formulate the problem as a Mixed Integer Programming (MIP) model and propose two near-optimal metaheuristic algorithms: one evolutionary (EA) and the other based on large neighborhood search (LNS). Experiments on real-world data show our methods can double driver profits compared to baselines, while maintaining near-optimal performance on small instances and excellent scalability on larger ones. Our work highlights a promising path toward smarter, more profitable EV-based mobility systems that actively support the energy grid. 

---
# Paladin-mini: A Compact and Efficient Grounding Model Excelling in Real-World Scenarios 

**Authors**: Dror Ivry, Oran Nahum  

**Link**: [PDF](https://arxiv.org/pdf/2506.20384)  

**Abstract**: This paper introduces two significant contributions to address the issue of grounding claims in a given context. Grounding means that given a context (document) and a claim, there's at least one supportive evidence for the claim in the document. We will introduce Paladin-mini, a compact (3.8B parameters) open-source classifier model (used for labeling data as grounded or ungrounded) engineered for robust performance in real-world scenarios, and the grounding-benchmark, a new evaluation dataset designed to assess performance on critical reasoning tasks. We'll also demonstrate the results of Paladin-mini with benchmarks against the current State-of-the-art and share clear and reproducible results. 

---
# Tabular Feature Discovery With Reasoning Type Exploration 

**Authors**: Sungwon Han, Sungkyu Park, Seungeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.20357)  

**Abstract**: Feature engineering for tabular data remains a critical yet challenging step in machine learning. Recently, large language models (LLMs) have been used to automatically generate new features by leveraging their vast knowledge. However, existing LLM-based approaches often produce overly simple or repetitive features, partly due to inherent biases in the transformations the LLM chooses and the lack of structured reasoning guidance during generation. In this paper, we propose a novel method REFeat, which guides an LLM to discover diverse and informative features by leveraging multiple types of reasoning to steer the feature generation process. Experiments on 59 benchmark datasets demonstrate that our approach not only achieves higher predictive accuracy on average, but also discovers more diverse and meaningful features. These results highlight the promise of incorporating rich reasoning paradigms and adaptive strategy selection into LLM-driven feature discovery for tabular data. 

---
# Mobile-R1: Towards Interactive Reinforcement Learning for VLM-Based Mobile Agent via Task-Level Rewards 

**Authors**: Jihao Gu, Qihang Ai, Yingyao Wang, Pi Bu, Jingxuan Xing, Zekun Zhu, Wei Jiang, Ziming Wang, Yingxiu Zhao, Ming-Liang Zhang, Jun Song, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.20332)  

**Abstract**: Vision-language model-based mobile agents have gained the ability to not only understand complex instructions and mobile screenshots, but also optimize their action outputs via thinking and reasoning, benefiting from reinforcement learning, such as Group Relative Policy Optimization (GRPO). However, existing research centers on offline reinforcement learning training or online optimization using action-level rewards, which limits the agent's dynamic interaction with the environment. This often results in agents settling into local optima, thereby weakening their ability for exploration and error action correction. To address these challenges, we introduce an approach called Mobile-R1, which employs interactive multi-turn reinforcement learning with task-level rewards for mobile agents. Our training framework consists of three stages: initial format finetuning, single-step online training via action-level reward, followed by online training via task-level reward based on multi-turn trajectories. This strategy is designed to enhance the exploration and error correction capabilities of Mobile-R1, leading to significant performance improvements. Moreover, we have collected a dataset covering 28 Chinese applications with 24,521 high-quality manual annotations and established a new benchmark with 500 trajectories. We will open source all resources, including the dataset, benchmark, model weight, and codes: this https URL. 

---
# Enterprise Large Language Model Evaluation Benchmark 

**Authors**: Liya Wang, David Yi, Damien Jose, John Passarelli, James Gao, Jordan Leventis, Kang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.20274)  

**Abstract**: Large Language Models (LLMs) ) have demonstrated promise in boosting productivity across AI-powered tools, yet existing benchmarks like Massive Multitask Language Understanding (MMLU) inadequately assess enterprise-specific task complexities. We propose a 14-task framework grounded in Bloom's Taxonomy to holistically evaluate LLM capabilities in enterprise contexts. To address challenges of noisy data and costly annotation, we develop a scalable pipeline combining LLM-as-a-Labeler, LLM-as-a-Judge, and corrective retrieval-augmented generation (CRAG), curating a robust 9,700-sample benchmark. Evaluation of six leading models shows open-source contenders like DeepSeek R1 rival proprietary models in reasoning tasks but lag in judgment-based scenarios, likely due to overthinking. Our benchmark reveals critical enterprise performance gaps and offers actionable insights for model optimization. This work provides enterprises a blueprint for tailored evaluations and advances practical LLM deployment. 

---
# Language Modeling by Language Models 

**Authors**: Junyan Cheng, Peter Clark, Kyle Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2506.20249)  

**Abstract**: Can we leverage LLMs to model the process of discovering novel language model (LM) architectures? Inspired by real research, we propose a multi-agent LLM approach that simulates the conventional stages of research, from ideation and literature search (proposal stage) to design implementation (code generation), generative pre-training, and downstream evaluation (verification). Using ideas from scaling laws, our system, Genesys, employs a Ladder of Scales approach; new designs are proposed, adversarially reviewed, implemented, and selectively verified at increasingly larger model scales (14M$\sim$350M parameters) with a narrowing budget (the number of models we can train at each scale). To help make discovery efficient and factorizable, Genesys uses a novel genetic programming backbone, which we show has empirical advantages over commonly used direct prompt generation workflows (e.g., $\sim$86\% percentage point improvement in successful design generation, a key bottleneck). We report experiments involving 1,162 newly discovered designs (1,062 fully verified through pre-training) and find the best designs to be highly competitive with known architectures (e.g., outperform GPT2, Mamba2, etc., on 6/9 common benchmarks). We couple these results with comprehensive system-level ablations and formal results, which give broader insights into the design of effective autonomous discovery systems. 

---
# AI Copilots for Reproducibility in Science: A Case Study 

**Authors**: Adrien Bibal, Steven N. Minton, Deborah Khider, Yolanda Gil  

**Link**: [PDF](https://arxiv.org/pdf/2506.20130)  

**Abstract**: Open science initiatives seek to make research outputs more transparent, accessible, and reusable, but ensuring that published findings can be independently reproduced remains a persistent challenge. This paper introduces OpenPub, an AI-powered platform that supports researchers, reviewers, and readers through a suite of modular copilots focused on key open science tasks. In this work, we present the Reproducibility Copilot, which analyzes manuscripts, code, and supplementary materials to generate structured Jupyter Notebooks and recommendations aimed at facilitating computational, or "rote", reproducibility. We conducted feasibility tests using previously studied research papers with known reproducibility benchmarks. Results indicate that OpenPub can substantially reduce reproduction time - from over 30 hours to about 1 hour - while achieving high coverage of figures, tables, and results suitable for computational reproduction. The system systematically detects barriers to reproducibility, including missing hyperparameters, undocumented preprocessing steps, and incomplete or inaccessible datasets. These findings suggest that AI-driven tools can meaningfully reduce the burden of reproducibility efforts and contribute to more transparent and verifiable scientific communication. The modular copilot architecture also provides a foundation for extending AI assistance to additional open science objectives beyond reproducibility. 

---
# DiaLLMs: EHR Enhanced Clinical Conversational System for Clinical Test Recommendation and Diagnosis Prediction 

**Authors**: Weijieying Ren, Tianxiang Zhao, Lei Wang, Tianchun Wang, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20059)  

**Abstract**: Recent advances in Large Language Models (LLMs) have led to remarkable progresses in medical consultation. However, existing medical LLMs overlook the essential role of Electronic Health Records (EHR) and focus primarily on diagnosis recommendation, limiting their clinical applicability. We propose DiaLLM, the first medical LLM that integrates heterogeneous EHR data into clinically grounded dialogues, enabling clinical test recommendation, result interpretation, and diagnosis prediction to better align with real-world medical practice. To construct clinically grounded dialogues from EHR, we design a Clinical Test Reference (CTR) strategy that maps each clinical code to its corresponding description and classifies test results as "normal" or "abnormal". Additionally, DiaLLM employs a reinforcement learning framework for evidence acquisition and automated diagnosis. To handle the large action space, we introduce a reject sampling strategy to reduce redundancy and improve exploration efficiency. Furthermore, a confirmation reward and a class-sensitive diagnosis reward are designed to guide accurate diagnosis prediction. Extensive experimental results demonstrate that DiaLLM outperforms baselines in clinical test recommendation and diagnosis prediction. 

---
# Persona-Assigned Large Language Models Exhibit Human-Like Motivated Reasoning 

**Authors**: Saloni Dash, Amélie Reymond, Emma S. Spiro, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20020)  

**Abstract**: Reasoning in humans is prone to biases due to underlying motivations like identity protection, that undermine rational decision-making and judgment. This motivated reasoning at a collective level can be detrimental to society when debating critical issues such as human-driven climate change or vaccine safety, and can further aggravate political polarization. Prior studies have reported that large language models (LLMs) are also susceptible to human-like cognitive biases, however, the extent to which LLMs selectively reason toward identity-congruent conclusions remains largely unexplored. Here, we investigate whether assigning 8 personas across 4 political and socio-demographic attributes induces motivated reasoning in LLMs. Testing 8 LLMs (open source and proprietary) across two reasoning tasks from human-subject studies -- veracity discernment of misinformation headlines and evaluation of numeric scientific evidence -- we find that persona-assigned LLMs have up to 9% reduced veracity discernment relative to models without personas. Political personas specifically, are up to 90% more likely to correctly evaluate scientific evidence on gun control when the ground truth is congruent with their induced political identity. Prompt-based debiasing methods are largely ineffective at mitigating these effects. Taken together, our empirical findings are the first to suggest that persona-assigned LLMs exhibit human-like motivated reasoning that is hard to mitigate through conventional debiasing prompts -- raising concerns of exacerbating identity-congruent reasoning in both LLMs and humans. 

---
# Achieving Trustworthy Real-Time Decision Support Systems with Low-Latency Interpretable AI Models 

**Authors**: Zechun Deng, Ziwei Liu, Ziqian Bi, Junhao Song, Chia Xin Liang, Joe Yeong, Junfeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20018)  

**Abstract**: This paper investigates real-time decision support systems that leverage low-latency AI models, bringing together recent progress in holistic AI-driven decision tools, integration with Edge-IoT technologies, and approaches for effective human-AI teamwork. It looks into how large language models can assist decision-making, especially when resources are limited. The research also examines the effects of technical developments such as DeLLMa, methods for compressing models, and improvements for analytics on edge devices, while also addressing issues like limited resources and the need for adaptable frameworks. Through a detailed review, the paper offers practical perspectives on development strategies and areas of application, adding to the field by pointing out opportunities for more efficient and flexible AI-supported systems. The conclusions set the stage for future breakthroughs in this fast-changing area, highlighting how AI can reshape real-time decision support. 

---
# Accurate and Energy Efficient: Local Retrieval-Augmented Generation Models Outperform Commercial Large Language Models in Medical Tasks 

**Authors**: Konstantinos Vrettos, Michail E. Klontzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.20009)  

**Abstract**: Background The increasing adoption of Artificial Intelligence (AI) in healthcare has sparked growing concerns about its environmental and ethical implications. Commercial Large Language Models (LLMs), such as ChatGPT and DeepSeek, require substantial resources, while the utilization of these systems for medical purposes raises critical issues regarding patient privacy and safety. Methods We developed a customizable Retrieval-Augmented Generation (RAG) framework for medical tasks, which monitors its energy usage and CO2 emissions. This system was then used to create RAGs based on various open-source LLMs. The tested models included both general purpose models like llama3.1:8b and medgemma-4b-it, which is medical-domain specific. The best RAGs performance and energy consumption was compared to DeepSeekV3-R1 and OpenAIs o4-mini model. A dataset of medical questions was used for the evaluation. Results Custom RAG models outperformed commercial models in accuracy and energy consumption. The RAG model built on llama3.1:8B achieved the highest accuracy (58.5%) and was significantly better than other models, including o4-mini and DeepSeekV3-R1. The llama3.1-RAG also exhibited the lowest energy consumption and CO2 footprint among all models, with a Performance per kWh of 0.52 and a total CO2 emission of 473g. Compared to o4-mini, the llama3.1-RAG achieved 2.7x times more accuracy points per kWh and 172% less electricity usage while maintaining higher accuracy. Conclusion Our study demonstrates that local LLMs can be leveraged to develop RAGs that outperform commercial, online LLMs in medical tasks, while having a smaller environmental impact. Our modular framework promotes sustainable AI development, reducing electricity usage and aligning with the UNs Sustainable Development Goals. 

---
# QHackBench: Benchmarking Large Language Models for Quantum Code Generation Using PennyLane Hackathon Challenges 

**Authors**: Abdul Basit, Minghao Shao, Haider Asif, Nouhaila Innan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2506.20008)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated strong potential in code generation, yet their effectiveness in quantum computing remains underexplored. This paper benchmarks LLMs for PennyLane-based quantum code generation using real-world challenges from the Quantum Hackathon (QHack). We introduce QHackBench, a novel benchmark dataset derived from QHack competitions, and evaluate model performance under vanilla prompting and Retrieval-Augmented Generation (RAG). Our structured evaluation framework assesses functional correctness, syntactic validity, and execution success across varying challenge difficulties. Results indicate that RAG-enhanced models, supplemented with an augmented PennyLane dataset, approximately generate similar results as the standard prompting, particularly in complex quantum algorithms. Additionally, we introduce a multi-agent evaluation pipeline that iteratively refines incorrect solutions, further enhancing execution success rates. To foster further research, we commit to publicly releasing QHackBench, along with our evaluation framework and experimental results, enabling continued advancements in AI-assisted quantum programming. 

---
# Context Attribution with Multi-Armed Bandit Optimization 

**Authors**: Deng Pan, Keerthiram Murugesan, Nuno Moniz, Nitesh Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2506.19977)  

**Abstract**: Understanding which parts of the retrieved context contribute to a large language model's generated answer is essential for building interpretable and trustworthy generative QA systems. We propose a novel framework that formulates context attribution as a combinatorial multi-armed bandit (CMAB) problem. Each context segment is treated as a bandit arm, and we employ Combinatorial Thompson Sampling (CTS) to efficiently explore the exponentially large space of context subsets under a limited query budget. Our method defines a reward function based on normalized token likelihoods, capturing how well a subset of segments supports the original model response. Unlike traditional perturbation-based attribution methods such as SHAP, which sample subsets uniformly and incur high computational costs, our approach adaptively balances exploration and exploitation by leveraging posterior estimates of segment relevance. This leads to substantially improved query efficiency while maintaining high attribution fidelity. Extensive experiments on diverse datasets and LLMs demonstrate that our method achieves competitive attribution quality with fewer model queries. 

---
# Prover Agent: An Agent-based Framework for Formal Mathematical Proofs 

**Authors**: Kaito Baba, Chaoran Liu, Shuhei Kurita, Akiyoshi Sannai  

**Link**: [PDF](https://arxiv.org/pdf/2506.19923)  

**Abstract**: We present Prover Agent, a novel AI agent for automated theorem proving that integrates large language models (LLMs) with a formal proof assistant, Lean. Prover Agent coordinates an informal reasoning LLM, a formal prover model, and feedback from Lean while also generating auxiliary lemmas to assist in discovering the overall proof strategy. It achieves an 86.1% success rate on the MiniF2F benchmark, establishing a new state-of-the-art among methods using small language models (SLMs) with a much lower sample budget than previous approaches. We also present case studies illustrating how these generated lemmas contribute to solving challenging problems. 

---
# Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs 

**Authors**: Sonia K. Murthy, Rosie Zhao, Jennifer Hu, Sham Kakade, Markus Wulfmeier, Peng Qian, Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20666)  

**Abstract**: Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training. 

---
# Disentangled representations of microscopy images 

**Authors**: Jacopo Dapueto, Vito Paolo Pastore, Nicoletta Noceti, Francesca Odone  

**Link**: [PDF](https://arxiv.org/pdf/2506.20649)  

**Abstract**: Microscopy image analysis is fundamental for different applications, from diagnosis to synthetic engineering and environmental monitoring. Modern acquisition systems have granted the possibility to acquire an escalating amount of images, requiring a consequent development of a large collection of deep learning-based automatic image analysis methods. Although deep neural networks have demonstrated great performance in this field, interpretability, an essential requirement for microscopy image analysis, remains an open challenge.
This work proposes a Disentangled Representation Learning (DRL) methodology to enhance model interpretability for microscopy image classification. Exploiting benchmark datasets from three different microscopic image domains (plankton, yeast vacuoles, and human cells), we show how a DRL framework, based on transferring a representation learnt from synthetic data, can provide a good trade-off between accuracy and interpretability in this domain. 

---
# Define-ML: An Approach to Ideate Machine Learning-Enabled Systems 

**Authors**: Silvio Alonso, Antonio Pedro Santos Alves, Lucas Romao, Hélio Lopes, Marcos Kalinowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.20621)  

**Abstract**: [Context] The increasing adoption of machine learning (ML) in software systems demands specialized ideation approaches that address ML-specific challenges, including data dependencies, technical feasibility, and alignment between business objectives and probabilistic system behavior. Traditional ideation methods like Lean Inception lack structured support for these ML considerations, which can result in misaligned product visions and unrealistic expectations. [Goal] This paper presents Define-ML, a framework that extends Lean Inception with tailored activities - Data Source Mapping, Feature-to-Data Source Mapping, and ML Mapping - to systematically integrate data and technical constraints into early-stage ML product ideation. [Method] We developed and validated Define-ML following the Technology Transfer Model, conducting both static validation (with a toy problem) and dynamic validation (in a real-world industrial case study). The analysis combined quantitative surveys with qualitative feedback, assessing utility, ease of use, and intent of adoption. [Results] Participants found Define-ML effective for clarifying data concerns, aligning ML capabilities with business goals, and fostering cross-functional collaboration. The approach's structured activities reduced ideation ambiguity, though some noted a learning curve for ML-specific components, which can be mitigated by expert facilitation. All participants expressed the intention to adopt Define-ML. [Conclusion] Define-ML provides an openly available, validated approach for ML product ideation, building on Lean Inception's agility while aligning features with available data and increasing awareness of technical feasibility. 

---
# Weighted Mean Frequencies: a handcraft Fourier feature for 4D Flow MRI segmentation 

**Authors**: Simon Perrin, Sébastien Levilly, Huajun Sun, Harold Mouchère, Jean-Michel Serfaty  

**Link**: [PDF](https://arxiv.org/pdf/2506.20614)  

**Abstract**: In recent decades, the use of 4D Flow MRI images has enabled the quantification of velocity fields within a volume of interest and along the cardiac cycle. However, the lack of resolution and the presence of noise in these biomarkers are significant issues. As indicated by recent studies, it appears that biomarkers such as wall shear stress are particularly impacted by the poor resolution of vessel segmentation. The Phase Contrast Magnetic Resonance Angiography (PC-MRA) is the state-of-the-art method to facilitate segmentation. The objective of this work is to introduce a new handcraft feature that provides a novel visualisation of 4D Flow MRI images, which is useful in the segmentation task. This feature, termed Weighted Mean Frequencies (WMF), is capable of revealing the region in three dimensions where a voxel has been passed by pulsatile flow. Indeed, this feature is representative of the hull of all pulsatile velocity voxels. The value of the feature under discussion is illustrated by two experiments. The experiments involved segmenting 4D Flow MRI images using optimal thresholding and deep learning methods. The results obtained demonstrate a substantial enhancement in terms of IoU and Dice, with a respective increase of 0.12 and 0.13 in comparison with the PC-MRA feature, as evidenced by the deep learning task. This feature has the potential to yield valuable insights that could inform future segmentation processes in other vascular regions, such as the heart or the brain. 

---
# Deciphering GunType Hierarchy through Acoustic Analysis of Gunshot Recordings 

**Authors**: Ankit Shah, Rita Singh, Bhiksha Raj, Alexander Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.20609)  

**Abstract**: The escalating rates of gun-related violence and mass shootings represent a significant threat to public safety. Timely and accurate information for law enforcement agencies is crucial in mitigating these incidents. Current commercial gunshot detection systems, while effective, often come with prohibitive costs. This research explores a cost-effective alternative by leveraging acoustic analysis of gunshot recordings, potentially obtainable from ubiquitous devices like cell phones, to not only detect gunshots but also classify the type of firearm used. This paper details a study on deciphering gun type hierarchies using a curated dataset of 3459 recordings. We investigate the fundamental acoustic characteristics of gunshots, including muzzle blasts and shockwaves, which vary based on firearm type, ammunition, and shooting direction. We propose and evaluate machine learning frameworks, including Support Vector Machines (SVMs) as a baseline and a more advanced Convolutional Neural Network (CNN) architecture for joint gunshot detection and gun type classification. Results indicate that our deep learning approach achieves a mean average precision (mAP) of 0.58 on clean labeled data, outperforming the SVM baseline (mAP 0.39). Challenges related to data quality, environmental noise, and the generalization capabilities when using noisy web-sourced data (mAP 0.35) are also discussed. The long-term vision is to develop a highly accurate, real-time system deployable on common recording devices, significantly reducing detection costs and providing critical intelligence to first responders. 

---
# AI in the Writing Process: How Purposeful AI Support Fosters Student Writing 

**Authors**: Momin N. Siddiqui, Roy Pea, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20595)  

**Abstract**: The ubiquity of technologies like ChatGPT has raised concerns about their impact on student writing, particularly regarding reduced learner agency and superficial engagement with content. While standalone chat-based LLMs often produce suboptimal writing outcomes, evidence suggests that purposefully designed AI writing support tools can enhance the writing process. This paper investigates how different AI support approaches affect writers' sense of agency and depth of knowledge transformation. Through a randomized control trial with 90 undergraduate students, we compare three conditions: (1) a chat-based LLM writing assistant, (2) an integrated AI writing tool to support diverse subprocesses, and (3) a standard writing interface (control). Our findings demonstrate that, among AI-supported conditions, students using the integrated AI writing tool exhibited greater agency over their writing process and engaged in deeper knowledge transformation overall. These results suggest that thoughtfully designed AI writing support targeting specific aspects of the writing process can help students maintain ownership of their work while facilitating improved engagement with content. 

---
# Dense Video Captioning using Graph-based Sentence Summarization 

**Authors**: Zhiwang Zhang, Dong Xu, Wanli Ouyang, Luping Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20583)  

**Abstract**: Recently, dense video captioning has made attractive progress in detecting and captioning all events in a long untrimmed video. Despite promising results were achieved, most existing methods do not sufficiently explore the scene evolution within an event temporal proposal for captioning, and therefore perform less satisfactorily when the scenes and objects change over a relatively long proposal. To address this problem, we propose a graph-based partition-and-summarization (GPaS) framework for dense video captioning within two stages. For the ``partition" stage, a whole event proposal is split into short video segments for captioning at a finer level. For the ``summarization" stage, the generated sentences carrying rich description information for each segment are summarized into one sentence to describe the whole event. We particularly focus on the ``summarization" stage, and propose a framework that effectively exploits the relationship between semantic words for summarization. We achieve this goal by treating semantic words as nodes in a graph and learning their interactions by coupling Graph Convolutional Network (GCN) and Long Short Term Memory (LSTM), with the aid of visual cues. Two schemes of GCN-LSTM Interaction (GLI) modules are proposed for seamless integration of GCN and LSTM. The effectiveness of our approach is demonstrated via an extensive comparison with the state-of-the-arts methods on the two benchmarks ActivityNet Captions dataset and YouCook II dataset. 

---
# Causal Representation Learning with Observational Grouping for CXR Classification 

**Authors**: Rajat Rasal, Avinash Kori, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2506.20582)  

**Abstract**: Identifiable causal representation learning seeks to uncover the true causal relationships underlying a data generation process. In medical imaging, this presents opportunities to improve the generalisability and robustness of task-specific latent features. This work introduces the concept of grouping observations to learn identifiable representations for disease classification in chest X-rays via an end-to-end framework. Our experiments demonstrate that these causal representations improve generalisability and robustness across multiple classification tasks when grouping is used to enforce invariance w.r.t race, sex, and imaging views. 

---
# Vulnerability Disclosure through Adaptive Black-Box Adversarial Attacks on NIDS 

**Authors**: Sabrine Ennaji, Elhadj Benkhelifa, Luigi V. Mancini  

**Link**: [PDF](https://arxiv.org/pdf/2506.20576)  

**Abstract**: Adversarial attacks, wherein slight inputs are carefully crafted to mislead intelligent models, have attracted increasing attention. However, a critical gap persists between theoretical advancements and practical application, particularly in structured data like network traffic, where interdependent features complicate effective adversarial manipulations. Moreover, ambiguity in current approaches restricts reproducibility and limits progress in this field. Hence, existing defenses often fail to handle evolving adversarial attacks. This paper proposes a novel approach for black-box adversarial attacks, that addresses these limitations. Unlike prior work, which often assumes system access or relies on repeated probing, our method strictly respect black-box constraints, reducing interaction to avoid detection and better reflect real-world scenarios. We present an adaptive feature selection strategy using change-point detection and causality analysis to identify and target sensitive features to perturbations. This lightweight design ensures low computational cost and high deployability. Our comprehensive experiments show the attack's effectiveness in evading detection with minimal interaction, enhancing its adaptability and applicability in real-world scenarios. By advancing the understanding of adversarial attacks in network traffic, this work lays a foundation for developing robust defenses. 

---
# Show, Tell and Summarize: Dense Video Captioning Using Visual Cue Aided Sentence Summarization 

**Authors**: Zhiwang Zhang, Dong Xu, Wanli Ouyang, Chuanqi Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20567)  

**Abstract**: In this work, we propose a division-and-summarization (DaS) framework for dense video captioning. After partitioning each untrimmed long video as multiple event proposals, where each event proposal consists of a set of short video segments, we extract visual feature (e.g., C3D feature) from each segment and use the existing image/video captioning approach to generate one sentence description for this segment. Considering that the generated sentences contain rich semantic descriptions about the whole event proposal, we formulate the dense video captioning task as a visual cue aided sentence summarization problem and propose a new two stage Long Short Term Memory (LSTM) approach equipped with a new hierarchical attention mechanism to summarize all generated sentences as one descriptive sentence with the aid of visual features. Specifically, the first-stage LSTM network takes all semantic words from the generated sentences and the visual features from all segments within one event proposal as the input, and acts as the encoder to effectively summarize both semantic and visual information related to this event proposal. The second-stage LSTM network takes the output from the first-stage LSTM network and the visual features from all video segments within one event proposal as the input, and acts as the decoder to generate one descriptive sentence for this event proposal. Our comprehensive experiments on the ActivityNet Captions dataset demonstrate the effectiveness of our newly proposed DaS framework for dense video captioning. 

---
# DeepQuark: deep-neural-network approach to multiquark bound states 

**Authors**: Wei-Lin Wu, Lu Meng, Shi-Lin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20555)  

**Abstract**: For the first time, we implement the deep-neural-network-based variational Monte Carlo approach for the multiquark bound states, whose complexity surpasses that of electron or nucleon systems due to strong SU(3) color interactions. We design a novel and high-efficiency architecture, DeepQuark, to address the unique challenges in multiquark systems such as stronger correlations, extra discrete quantum numbers, and intractable confinement interaction. Our method demonstrates competitive performance with state-of-the-art approaches, including diffusion Monte Carlo and Gaussian expansion method, in the nucleon, doubly heavy tetraquark, and fully heavy tetraquark systems. Notably, it outperforms existing calculations for pentaquarks, exemplified by the triply heavy pentaquark. For the nucleon, we successfully incorporate three-body flux-tube confinement interactions without additional computational costs. In tetraquark systems, we consistently describe hadronic molecule $T_{cc}$ and compact tetraquark $T_{bb}$ with an unbiased form of wave function ansatz. In the pentaquark sector, we obtain weakly bound $\bar D^*\Xi_{cc}^*$ molecule $P_{cc\bar c}(5715)$ with $S=\frac{5}{2}$ and its bottom partner $P_{bb\bar b}(15569)$. They can be viewed as the analogs of the molecular $T_{cc}$. We recommend experimental search of $P_{cc\bar c}(5715)$ in the D-wave $J/\psi \Lambda_c$ channel. DeepQuark holds great promise for extension to larger multiquark systems, overcoming the computational barriers in conventional methods. It also serves as a powerful framework for exploring confining mechanism beyond two-body interactions in multiquark states, which may offer valuable insights into nonperturbative QCD and general many-body physics. 

---
# Large Language Model-Driven Code Compliance Checking in Building Information Modeling 

**Authors**: Soumya Madireddy, Lu Gao, Zia Din, Kinam Kim, Ahmed Senouci, Zhe Han, Yunpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20551)  

**Abstract**: This research addresses the time-consuming and error-prone nature of manual code compliance checking in Building Information Modeling (BIM) by introducing a Large Language Model (LLM)-driven approach to semi-automate this critical process. The developed system integrates LLMs such as GPT, Claude, Gemini, and Llama, with Revit software to interpret building codes, generate Python scripts, and perform semi-automated compliance checks within the BIM environment. Case studies on a single-family residential project and an office building project demonstrated the system's ability to reduce the time and effort required for compliance checks while improving accuracy. It streamlined the identification of violations, such as non-compliant room dimensions, material usage, and object placements, by automatically assessing relationships and generating actionable reports. Compared to manual methods, the system eliminated repetitive tasks, simplified complex regulations, and ensured reliable adherence to standards. By offering a comprehensive, adaptable, and cost-effective solution, this proposed approach offers a promising advancement in BIM-based compliance checking, with potential applications across diverse regulatory documents in construction projects. 

---
# Pay Less Attention to Deceptive Artifacts: Robust Detection of Compressed Deepfakes on Online Social Networks 

**Authors**: Manyi Li, Renshuai Tao, Yufan Liu, Chuangchuang Tan, Haotong Qin, Bing Li, Yunchao Wei, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20548)  

**Abstract**: With the rapid advancement of deep learning, particularly through generative adversarial networks (GANs) and diffusion models (DMs), AI-generated images, or ``deepfakes", have become nearly indistinguishable from real ones. These images are widely shared across Online Social Networks (OSNs), raising concerns about their misuse. Existing deepfake detection methods overlook the ``block effects" introduced by compression in OSNs, which obscure deepfake artifacts, and primarily focus on raw images, rarely encountered in real-world scenarios. To address these challenges, we propose PLADA (Pay Less Attention to Deceptive Artifacts), a novel framework designed to tackle the lack of paired data and the ineffective use of compressed images. PLADA consists of two core modules: Block Effect Eraser (B2E), which uses a dual-stage attention mechanism to handle block effects, and Open Data Aggregation (ODA), which processes both paired and unpaired data to improve detection. Extensive experiments across 26 datasets demonstrate that PLADA achieves a remarkable balance in deepfake detection, outperforming SoTA methods in detecting deepfakes on OSNs, even with limited paired data and compression. More importantly, this work introduces the ``block effect" as a critical factor in deepfake detection, providing a robust solution for open-world scenarios. Our code is available at this https URL. 

---
# When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs 

**Authors**: Ammar Khairi, Daniel D'souza, Ye Shen, Julia Kreutzer, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2506.20544)  

**Abstract**: Recent advancements in large language models (LLMs) have shifted focus toward scaling inference-time compute, improving performance without retraining the model. A common approach is to sample multiple outputs in parallel, and select one of these as the final output. However, work to date has focused on English and a handful of domains such as math and code. In contrast, we are most interested in techniques that generalize across open-ended tasks, formally verifiable tasks, and across languages. In this work, we study how to robustly scale inference-time compute for open-ended generative tasks in a multilingual, multi-task setting.
Our findings show that both sampling strategy based on temperature variation and selection strategy must be adapted to account for diverse domains and varied language settings. We evaluate existing selection methods, revealing that strategies effective in English often fail to generalize across languages. We propose novel sampling and selection strategies specifically adapted for multilingual and multi-task inference scenarios, and show they yield notable gains across languages and tasks. In particular, our combined sampling and selection methods lead to an average +6.8 jump in win-rates for our 8B models on m-ArenaHard-v2.0 prompts, against proprietary models such as Gemini. At larger scale, Command-A (111B model) equipped with our methods, shows +9.0 improvement in win-rates on the same benchmark with just five samples against single-sample decoding, a substantial increase at minimal cost. Our results underscore the need for language- and task-aware approaches to inference-time compute, aiming to democratize performance improvements in underrepresented languages. 

---
# WattsOnAI: Measuring, Analyzing, and Visualizing Energy and Carbon Footprint of AI Workloads 

**Authors**: Hongzhen Huang, Kunming Zhang, Hanlong Liao, Kui Wu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20535)  

**Abstract**: The rapid advancement of AI, particularly large language models (LLMs), has raised significant concerns about the energy use and carbon emissions associated with model training and inference. However, existing tools for measuring and reporting such impacts are often fragmented, lacking systematic metric integration and offering limited support for correlation analysis among them. This paper presents WattsOnAI, a comprehensive software toolkit for the measurement, analysis, and visualization of energy use, power draw, hardware performance, and carbon emissions across AI workloads. By seamlessly integrating with existing AI frameworks, WattsOnAI offers standardized reports and exports fine-grained time-series data to support benchmarking and reproducibility in a lightweight manner. It further enables in-depth correlation analysis between hardware metrics and model performance and thus facilitates bottleneck identification and performance enhancement. By addressing critical limitations in existing tools, WattsOnAI encourages the research community to weigh environmental impact alongside raw performance of AI workloads and advances the shift toward more sustainable "Green AI" practices. The code is available at this https URL. 

---
# Industrial Energy Disaggregation with Digital Twin-generated Dataset and Efficient Data Augmentation 

**Authors**: Christian Internò, Andrea Castellani, Sebastian Schmitt, Fabio Stella, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.20525)  

**Abstract**: Industrial Non-Intrusive Load Monitoring (NILM) is limited by the scarcity of high-quality datasets and the complex variability of industrial energy consumption patterns. To address data scarcity and privacy issues, we introduce the Synthetic Industrial Dataset for Energy Disaggregation (SIDED), an open-source dataset generated using Digital Twin simulations. SIDED includes three types of industrial facilities across three different geographic locations, capturing diverse appliance behaviors, weather conditions, and load profiles. We also propose the Appliance-Modulated Data Augmentation (AMDA) method, a computationally efficient technique that enhances NILM model generalization by intelligently scaling appliance power contributions based on their relative impact. We show in experiments that NILM models trained with AMDA-augmented data significantly improve the disaggregation of energy consumption of complex industrial appliances like combined heat and power systems. Specifically, in our out-of-sample scenarios, models trained with AMDA achieved a Normalized Disaggregation Error of 0.093, outperforming models trained without data augmentation (0.451) and those trained with random data augmentation (0.290). Data distribution analyses confirm that AMDA effectively aligns training and test data distributions, enhancing model generalization. 

---
# OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling 

**Authors**: Zengzhi Wang, Fan Zhou, Xuefeng Li, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20512)  

**Abstract**: Different base language model families, such as Llama and Qwen, exhibit divergent behaviors during post-training with reinforcement learning (RL), especially on reasoning-intensive tasks. What makes a base language model suitable for reinforcement learning? Gaining deeper insight into this question is essential for developing RL-scalable foundation models of the next generation. In this work, we investigate how mid-training strategies shape RL dynamics, focusing on two representative model families: Qwen and Llama. Our study reveals that (1) high-quality mathematical corpora, such as MegaMath-Web-Pro, significantly improve both base model and RL performance, while existing alternatives (e.g., FineMath-4plus) fail to do so; (2) further adding QA-style data, particularly long chain-of-thought (CoT) reasoning examples, enhances RL outcomes, and instruction data further unlocks this effect; (3) while long-CoT improves reasoning depth, it can also induce verbosity of model responses and unstability of RL training, underscoring the importance of data formatting; (4) scaling mid-training consistently leads to stronger downstream RL performance. Building on these insights, we introduce a two-stage mid-training strategy, Stable-then-Decay, in which base models are first trained on 200B tokens with a constant learning rate, followed by 20B tokens across three CoT-focused branches with learning rate decay. This yields OctoThinker, a family of models demonstrating strong RL compatibility and closing the performance gap with more RL-friendly model families, i.e., Qwen. We hope our work will help shape pre-training strategies for foundation models in the RL era. To support further research, we release our open-source models along with a curated math reasoning-intensive corpus of over 70 billion tokens (i.e., MegaMath-Web-Pro-Max). 

---
# ReCode: Updating Code API Knowledge with Reinforcement Learning 

**Authors**: Haoze Wu, Yunzhi Yao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20495)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at this https URL. 

---
# Counterfactual Influence as a Distributional Quantity 

**Authors**: Matthieu Meeus, Igor Shilov, Georgios Kaissis, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2506.20481)  

**Abstract**: Machine learning models are known to memorize samples from their training data, raising concerns around privacy and generalization. Counterfactual self-influence is a popular metric to study memorization, quantifying how the model's prediction for a sample changes depending on the sample's inclusion in the training dataset. However, recent work has shown memorization to be affected by factors beyond self-influence, with other training samples, in particular (near-)duplicates, having a large impact. We here study memorization treating counterfactual influence as a distributional quantity, taking into account how all training samples influence how a sample is memorized. For a small language model, we compute the full influence distribution of training samples on each other and analyze its properties. We find that solely looking at self-influence can severely underestimate tangible risks associated with memorization: the presence of (near-)duplicates seriously reduces self-influence, while we find these samples to be (near-)extractable. We observe similar patterns for image classification, where simply looking at the influence distributions reveals the presence of near-duplicates in CIFAR-10. Our findings highlight that memorization stems from complex interactions across training data and is better captured by the full influence distribution than by self-influence alone. 

---
# Automatic Demonstration Selection for LLM-based Tabular Data Classification 

**Authors**: Shuchu Han, Wolfgang Bruckner  

**Link**: [PDF](https://arxiv.org/pdf/2506.20451)  

**Abstract**: A fundamental question in applying In-Context Learning (ICL) for tabular data classification is how to determine the ideal number of demonstrations in the prompt. This work addresses this challenge by presenting an algorithm to automatically select a reasonable number of required demonstrations. Our method distinguishes itself by integrating not only the tabular data's distribution but also the user's selected prompt template and the specific Large Language Model (LLM) into its estimation. Rooted in Spectral Graph Theory, our proposed algorithm defines a novel metric to quantify the similarities between different demonstrations. We then construct a similarity graph and analyze the eigenvalues of its Laplacian to derive the minimum number of demonstrations capable of representing the data within the LLM's intrinsic representation space. We validate the efficacy of our approach through experiments comparing its performance against conventional random selection algorithms on diverse datasets and LLMs. 

---
# An Agentic System for Rare Disease Diagnosis with Traceable Reasoning 

**Authors**: Weike Zhao, Chaoyi Wu, Yanjie Fan, Xiaoman Zhang, Pengcheng Qiu, Yuze Sun, Xiao Zhou, Yanfeng Wang, Ya Zhang, Yongguo Yu, Kun Sun, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20430)  

**Abstract**: Rare diseases collectively affect over 300 million individuals worldwide, yet timely and accurate diagnosis remains a pervasive challenge. This is largely due to their clinical heterogeneity, low individual prevalence, and the limited familiarity most clinicians have with rare conditions. Here, we introduce DeepRare, the first rare disease diagnosis agentic system powered by a large language model (LLM), capable of processing heterogeneous clinical inputs. The system generates ranked diagnostic hypotheses for rare diseases, each accompanied by a transparent chain of reasoning that links intermediate analytic steps to verifiable medical evidence.
DeepRare comprises three key components: a central host with a long-term memory module; specialized agent servers responsible for domain-specific analytical tasks integrating over 40 specialized tools and web-scale, up-to-date medical knowledge sources, ensuring access to the most current clinical information. This modular and scalable design enables complex diagnostic reasoning while maintaining traceability and adaptability. We evaluate DeepRare on eight datasets. The system demonstrates exceptional diagnostic performance among 2,919 diseases, achieving 100% accuracy for 1013 diseases. In HPO-based evaluations, DeepRare significantly outperforms other 15 methods, like traditional bioinformatics diagnostic tools, LLMs, and other agentic systems, achieving an average Recall@1 score of 57.18% and surpassing the second-best method (Reasoning LLM) by a substantial margin of 23.79 percentage points. For multi-modal input scenarios, DeepRare achieves 70.60% at Recall@1 compared to Exomiser's 53.20% in 109 cases. Manual verification of reasoning chains by clinical experts achieves 95.40% agreements. Furthermore, the DeepRare system has been implemented as a user-friendly web application this http URL. 

---
# Off-Policy Evaluation and Learning for the Future under Non-Stationarity 

**Authors**: Tatsuhiro Shimizu, Kazuki Kawamura, Takanori Muroi, Yusuke Narita, Kei Tateno, Takuma Udagawa, Yuta Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.20417)  

**Abstract**: We study the novel problem of future off-policy evaluation (F-OPE) and learning (F-OPL) for estimating and optimizing the future value of policies in non-stationary environments, where distributions vary over time. In e-commerce recommendations, for instance, our goal is often to estimate and optimize the policy value for the upcoming month using data collected by an old policy in the previous month. A critical challenge is that data related to the future environment is not observed in the historical data. Existing methods assume stationarity or depend on restrictive reward-modeling assumptions, leading to significant bias. To address these limitations, we propose a novel estimator named \textit{\textbf{O}ff-\textbf{P}olicy Estimator for the \textbf{F}uture \textbf{V}alue (\textbf{\textit{OPFV}})}, designed for accurately estimating policy values at any future time point. The key feature of OPFV is its ability to leverage the useful structure within time-series data. While future data might not be present in the historical log, we can leverage, for example, seasonal, weekly, or holiday effects that are consistent in both the historical and future data. Our estimator is the first to exploit these time-related structures via a new type of importance weighting, enabling effective F-OPE. Theoretical analysis identifies the conditions under which OPFV becomes low-bias. In addition, we extend our estimator to develop a new policy-gradient method to proactively learn a good future policy using only historical data. Empirical results show that our methods substantially outperform existing methods in estimating and optimizing the future policy value under non-stationarity for various experimental setups. 

---
# SV-LLM: An Agentic Approach for SoC Security Verification using Large Language Models 

**Authors**: Dipayan Saha, Shams Tarek, Hasan Al Shaikh, Khan Thamid Hasan, Pavan Sai Nalluri, Md. Ajoad Hasan, Nashmin Alam, Jingbo Zhou, Sujan Kumar Saha, Mark Tehranipoor, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20415)  

**Abstract**: Ensuring the security of complex system-on-chips (SoCs) designs is a critical imperative, yet traditional verification techniques struggle to keep pace due to significant challenges in automation, scalability, comprehensiveness, and adaptability. The advent of large language models (LLMs), with their remarkable capabilities in natural language understanding, code generation, and advanced reasoning, presents a new paradigm for tackling these issues. Moving beyond monolithic models, an agentic approach allows for the creation of multi-agent systems where specialized LLMs collaborate to solve complex problems more effectively. Recognizing this opportunity, we introduce SV-LLM, a novel multi-agent assistant system designed to automate and enhance SoC security verification. By integrating specialized agents for tasks like verification question answering, security asset identification, threat modeling, test plan and property generation, vulnerability detection, and simulation-based bug validation, SV-LLM streamlines the workflow. To optimize their performance in these diverse tasks, agents leverage different learning paradigms, such as in-context learning, fine-tuning, and retrieval-augmented generation (RAG). The system aims to reduce manual intervention, improve accuracy, and accelerate security analysis, supporting proactive identification and mitigation of risks early in the design cycle. We demonstrate its potential to transform hardware security practices through illustrative case studies and experiments that showcase its applicability and efficacy. 

---
# Client Clustering Meets Knowledge Sharing: Enhancing Privacy and Robustness in Personalized Peer-to-Peer Learning 

**Authors**: Mohammad Mahdi Maheri, Denys Herasymuk, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20413)  

**Abstract**: The growing adoption of Artificial Intelligence (AI) in Internet of Things (IoT) ecosystems has intensified the need for personalized learning methods that can operate efficiently and privately across heterogeneous, resource-constrained devices. However, enabling effective personalized learning in decentralized settings introduces several challenges, including efficient knowledge transfer between clients, protection of data privacy, and resilience against poisoning attacks. In this paper, we address these challenges by developing P4 (Personalized, Private, Peer-to-Peer) -- a method designed to deliver personalized models for resource-constrained IoT devices while ensuring differential privacy and robustness against poisoning attacks. Our solution employs a lightweight, fully decentralized algorithm to privately detect client similarity and form collaborative groups. Within each group, clients leverage differentially private knowledge distillation to co-train their models, maintaining high accuracy while ensuring robustness to the presence of malicious clients. We evaluate P4 on popular benchmark datasets using both linear and CNN-based architectures across various heterogeneity settings and attack scenarios. Experimental results show that P4 achieves 5% to 30% higher accuracy than leading differentially private peer-to-peer approaches and maintains robustness with up to 30% malicious clients. Additionally, we demonstrate its practicality by deploying it on resource-constrained devices, where collaborative training between two clients adds only ~7 seconds of overhead. 

---
# CARMA: Context-Aware Situational Grounding of Human-Robot Group Interactions by Combining Vision-Language Models with Object and Action Recognition 

**Authors**: Joerg Deigmoeller, Stephan Hasler, Nakul Agarwal, Daniel Tanneberg, Anna Belardinelli, Reza Ghoddoosian, Chao Wang, Felix Ocker, Fan Zhang, Behzad Dariush, Michael Gienger  

**Link**: [PDF](https://arxiv.org/pdf/2506.20373)  

**Abstract**: We introduce CARMA, a system for situational grounding in human-robot group interactions. Effective collaboration in such group settings requires situational awareness based on a consistent representation of present persons and objects coupled with an episodic abstraction of events regarding actors and manipulated objects. This calls for a clear and consistent assignment of instances, ensuring that robots correctly recognize and track actors, objects, and their interactions over time. To achieve this, CARMA uniquely identifies physical instances of such entities in the real world and organizes them into grounded triplets of actors, objects, and actions.
To validate our approach, we conducted three experiments, where multiple humans and a robot interact: collaborative pouring, handovers, and sorting. These scenarios allow the assessment of the system's capabilities as to role distinction, multi-actor awareness, and consistent instance identification. Our experiments demonstrate that the system can reliably generate accurate actor-action-object triplets, providing a structured and robust foundation for applications requiring spatiotemporal reasoning and situated decision-making in collaborative settings. 

---
# Self-Supervised Graph Learning via Spectral Bootstrapping and Laplacian-Based Augmentations 

**Authors**: Lorenzo Bini, Stephane Marchand-Maillet  

**Link**: [PDF](https://arxiv.org/pdf/2506.20362)  

**Abstract**: We present LaplaceGNN, a novel self-supervised graph learning framework that bypasses the need for negative sampling by leveraging spectral bootstrapping techniques. Our method integrates Laplacian-based signals into the learning process, allowing the model to effectively capture rich structural representations without relying on contrastive objectives or handcrafted augmentations. By focusing on positive alignment, LaplaceGNN achieves linear scaling while offering a simpler, more efficient, self-supervised alternative for graph neural networks, applicable across diverse domains. Our contributions are twofold: we precompute spectral augmentations through max-min centrality-guided optimization, enabling rich structural supervision without relying on handcrafted augmentations, then we integrate an adversarial bootstrapped training scheme that further strengthens feature learning and robustness. Our extensive experiments on different benchmark datasets show that LaplaceGNN achieves superior performance compared to state-of-the-art self-supervised graph methods, offering a promising direction for efficiently learning expressive graph representations. 

---
# A foundation model with multi-variate parallel attention to generate neuronal activity 

**Authors**: Francesco Carzaniga, Michael Hersche, Abu Sebastian, Kaspar Schindler, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20354)  

**Abstract**: Learning from multi-variate time-series with heterogeneous channel configurations remains a fundamental challenge for deep neural networks (DNNs), particularly in clinical domains such as intracranial electroencephalography (iEEG), where channel setups vary widely across subjects. In this work, we introduce multi-variate parallel attention (MVPA), a novel self-attention mechanism that disentangles content, temporal, and spatial attention, enabling flexible, generalizable, and efficient modeling of time-series data with varying channel counts and configurations. We use MVPA to build MVPFormer, a generative foundation model for human electrophysiology, trained to predict the evolution of iEEG signals across diverse subjects. To support this and future effort by the community, we release the SWEC iEEG dataset, the largest publicly available iEEG dataset to date, comprising nearly 10,000 hours of recordings from heterogeneous clinical sources. MVPFormer leverages MVPA to achieve strong generalization across subjects, demonstrating expert-level performance in seizure detection and outperforming state-of-the-art Transformer baselines on our SWEC, the MAYO, and the FNUSA dataset. We further validate MVPA on standard time-series forecasting and classification tasks, where it matches or exceeds existing attention-based models. Together, our contributions establish MVPA as a general-purpose attention mechanism for heterogeneous time-series and MVPFormer as the first open-source, open-weights, and open-data iEEG foundation model with state-of-the-art clinical performance. The code is available at this https URL. The SWEC iEEG dataset is available at this https URL. 

---
# DipSVD: Dual-importance Protected SVD for Efficient LLM Compression 

**Authors**: Xuan Ding, Rui Sun, Yunjian Zhang, Xiu Yan, Yueqi Zhou, Kaihao Huang, Suzhong Fu, Chuanlong Xie, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20353)  

**Abstract**: The ever-increasing computational demands and deployment costs of large language models (LLMs) have spurred numerous compressing methods. Compared to quantization and unstructured pruning, SVD compression offers superior hardware compatibility and theoretical guarantees. However, existing SVD-based methods focus on the overall discrepancy between the original and compressed matrices while overlooking the protection of critical components within the matrix, which leads to inferior performance in the compressed models. This paper proposes a dual-level importance protection mechanism to enhance SVD-based compression methods: (1) local importance protection: preserving the most critical singular vectors within each weight matrix through channel-weighted data whitening; and (2) global importance protection: enabling less important layers to bear a greater portion of the compression burden through either a heuristic or optimization-based approach, thereby minimizing the impact of compression on critical layers. Extensive experiments demonstrate that DipSVD outperforms existing SVD-based compression approaches across multiple benchmarks, achieving superior model performance especially at high model compression ratios. 

---
# Feature Hallucination for Self-supervised Action Recognition 

**Authors**: Lei Wang, Piotr Koniusz  

**Link**: [PDF](https://arxiv.org/pdf/2506.20342)  

**Abstract**: Understanding human actions in videos requires more than raw pixel analysis; it relies on high-level semantic reasoning and effective integration of multimodal features. We propose a deep translational action recognition framework that enhances recognition accuracy by jointly predicting action concepts and auxiliary features from RGB video frames. At test time, hallucination streams infer missing cues, enriching feature representations without increasing computational overhead. To focus on action-relevant regions beyond raw pixels, we introduce two novel domain-specific descriptors. Object Detection Features (ODF) aggregate outputs from multiple object detectors to capture contextual cues, while Saliency Detection Features (SDF) highlight spatial and intensity patterns crucial for action recognition. Our framework seamlessly integrates these descriptors with auxiliary modalities such as optical flow, Improved Dense Trajectories, skeleton data, and audio cues. It remains compatible with state-of-the-art architectures, including I3D, AssembleNet, Video Transformer Network, FASTER, and recent models like VideoMAE V2 and InternVideo2. To handle uncertainty in auxiliary features, we incorporate aleatoric uncertainty modeling in the hallucination step and introduce a robust loss function to mitigate feature noise. Our multimodal self-supervised action recognition framework achieves state-of-the-art performance on multiple benchmarks, including Kinetics-400, Kinetics-600, and Something-Something V2, demonstrating its effectiveness in capturing fine-grained action dynamics. 

---
# Comparative Analysis of Deep Learning Models for Crop Disease Detection: A Transfer Learning Approach 

**Authors**: Saundarya Subramaniam, Shalini Majumdar, Shantanu Nadar, Kaustubh Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2506.20323)  

**Abstract**: This research presents the development of an Artificial Intelligence (AI) - driven crop disease detection system designed to assist farmers in rural areas with limited resources. We aim to compare different deep learning models for a comparative analysis, focusing on their efficacy in transfer learning. By leveraging deep learning models, including EfficientNet, ResNet101, MobileNetV2, and our custom CNN, which achieved a validation accuracy of 95.76%, the system effectively classifies plant diseases. This research demonstrates the potential of transfer learning in reshaping agricultural practices, improving crop health management, and supporting sustainable farming in rural environments. 

---
# Beyond-Expert Performance with Limited Demonstrations: Efficient Imitation Learning with Double Exploration 

**Authors**: Heyang Zhao, Xingrui Yu, David M. Bossens, Ivor W. Tsang, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20307)  

**Abstract**: Imitation learning is a central problem in reinforcement learning where the goal is to learn a policy that mimics the expert's behavior. In practice, it is often challenging to learn the expert policy from a limited number of demonstrations accurately due to the complexity of the state space. Moreover, it is essential to explore the environment and collect data to achieve beyond-expert performance. To overcome these challenges, we propose a novel imitation learning algorithm called Imitation Learning with Double Exploration (ILDE), which implements exploration in two aspects: (1) optimistic policy optimization via an exploration bonus that rewards state-action pairs with high uncertainty to potentially improve the convergence to the expert policy, and (2) curiosity-driven exploration of the states that deviate from the demonstration trajectories to potentially yield beyond-expert performance. Empirically, we demonstrate that ILDE outperforms the state-of-the-art imitation learning algorithms in terms of sample efficiency and achieves beyond-expert performance on Atari and MuJoCo tasks with fewer demonstrations than in previous work. We also provide a theoretical justification of ILDE as an uncertainty-regularized policy optimization method with optimistic exploration, leading to a regret growing sublinearly in the number of episodes. 

---
# Argumentative Ensembling for Robust Recourse under Model Multiplicity 

**Authors**: Junqi Jiang, Antonio Rago, Francesco Leofante, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2506.20260)  

**Abstract**: In machine learning, it is common to obtain multiple equally performing models for the same prediction task, e.g., when training neural networks with different random seeds. Model multiplicity (MM) is the situation which arises when these competing models differ in their predictions for the same input, for which ensembling is often employed to determine an aggregation of the outputs. Providing recourse recommendations via counterfactual explanations (CEs) under MM thus becomes complex, since the CE may not be valid across all models, i.e., the CEs are not robust under MM. In this work, we formalise the problem of providing recourse under MM, which we name recourse-aware ensembling (RAE). We propose the idea that under MM, CEs for each individual model should be considered alongside their predictions so that the aggregated prediction and recourse are decided in tandem. Centred around this intuition, we introduce six desirable properties for solutions to this problem. For solving RAE, we propose a novel argumentative ensembling method which guarantees the robustness of CEs under MM. Specifically, our method leverages computational argumentation to explicitly represent the conflicts between models and counterfactuals regarding prediction results and CE validity. It then uses argumentation semantics to resolve the conflicts and obtain the final solution, in a manner which is parametric to the chosen semantics. Our method also allows for the specification of preferences over the models under MM, allowing further customisation of the ensemble. In a comprehensive theoretical analysis, we characterise the behaviour of argumentative ensembling with four different argumentation semantics. We then empirically demonstrate the effectiveness of our approach in satisfying desirable properties with eight instantiations of our method. (Abstract is shortened for arXiv.) 

---
# Generating and Customizing Robotic Arm Trajectories using Neural Networks 

**Authors**: Andrej Lúčny, Matilde Antonj, Carlo Mazzola, Hana Hornáčková, Igor Farkaš  

**Link**: [PDF](https://arxiv.org/pdf/2506.20259)  

**Abstract**: We introduce a neural network approach for generating and customizing the trajectory of a robotic arm, that guarantees precision and repeatability. To highlight the potential of this novel method, we describe the design and implementation of the technique and show its application in an experimental setting of cognitive robotics. In this scenario, the NICO robot was characterized by the ability to point to specific points in space with precise linear movements, increasing the predictability of the robotic action during its interaction with humans. To achieve this goal, the neural network computes the forward kinematics of the robot arm. By integrating it with a generator of joint angles, another neural network was developed and trained on an artificial dataset created from suitable start and end poses of the robotic arm. Through the computation of angular velocities, the robot was characterized by its ability to perform the movement, and the quality of its action was evaluated in terms of shape and accuracy. Thanks to its broad applicability, our approach successfully generates precise trajectories that could be customized in their shape and adapted to different settings. 

---
# Time-series surrogates from energy consumers generated by machine learning approaches for long-term forecasting scenarios 

**Authors**: Ben Gerhards, Nikita Popkov, Annekatrin König, Marcel Arpogaus, Bastian Schäfermeier, Leonie Riedl, Stephan Vogt, Philip Hehlert  

**Link**: [PDF](https://arxiv.org/pdf/2506.20253)  

**Abstract**: Forecasting attracts a lot of research attention in the electricity value chain. However, most studies concentrate on short-term forecasting of generation or consumption with a focus on systems and less on individual consumers. Even more neglected is the topic of long-term forecasting of individual power consumption.
Here, we provide an in-depth comparative evaluation of data-driven methods for generating synthetic time series data tailored to energy consumption long-term forecasting. High-fidelity synthetic data is crucial for a wide range of applications, including state estimations in energy systems or power grid planning. In this study, we assess and compare the performance of multiple state-of-the-art but less common techniques: a hybrid Wasserstein Generative Adversarial Network (WGAN), Denoising Diffusion Probabilistic Model (DDPM), Hidden Markov Model (HMM), and Masked Autoregressive Bernstein polynomial normalizing Flows (MABF). We analyze the ability of each method to replicate the temporal dynamics, long-range dependencies, and probabilistic transitions characteristic of individual energy consumption profiles. Our comparative evaluation highlights the strengths and limitations of: WGAN, DDPM, HMM and MABF aiding in selecting the most suitable approach for state estimations and other energy-related tasks. Our generation and analysis framework aims to enhance the accuracy and reliability of synthetic power consumption data while generating data that fulfills criteria like anonymisation - preserving privacy concerns mitigating risks of specific profiling of single customers. This study utilizes an open-source dataset from households in Germany with 15min time resolution. The generated synthetic power profiles can readily be used in applications like state estimations or consumption forecasting. 

---
# Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models 

**Authors**: Kejia Chen, Jiawen Zhang, Jiacong Hu, Yu Wang, Jian Lou, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.20251)  

**Abstract**: Quantized large language models (LLMs) have gained increasing attention and significance for enabling deployment in resource-constrained environments. However, emerging studies on a few calibration dataset-free quantization methods suggest that quantization may compromise the safety capabilities of LLMs, underscoring the urgent need for systematic safety evaluations and effective mitigation strategies. In this paper, we present comprehensive safety evaluations across various mainstream quantization techniques and diverse calibration datasets, utilizing widely accepted safety benchmarks. To address the identified safety vulnerabilities, we propose a quantization-aware safety patching framework, Q-resafe, to efficiently restore the safety capabilities of quantized LLMs while minimizing any adverse impact on utility. Extensive experimental results demonstrate that Q-resafe successfully re-aligns the safety of quantized LLMs with their pre-quantization counterparts, even under challenging evaluation scenarios. Project page is available at: this https URL. 

---
# FedBKD: Distilled Federated Learning to Embrace Gerneralization and Personalization on Non-IID Data 

**Authors**: Yushan Zhao, Jinyuan He, Donglai Chen, Weijie Luo, Chong Xie, Ri Zhang, Yonghong Chen, Yan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20245)  

**Abstract**: Federated learning (FL) is a decentralized collaborative machine learning (ML) technique. It provides a solution to the issues of isolated data islands and data privacy leakage in industrial ML practices. One major challenge in FL is handling the non-identical and independent distributed (non-IID) data. Current solutions either focus on constructing an all-powerful global model, or customizing personalized local models. Few of them can provide both a well-generalized global model and well-performed local models at the same time. Additionally, many FL solutions to the non-IID problem are benefited from introducing public datasets. However, this will also increase the risk of data leakage. To tackle the problems, we propose a novel data-free distillation framework, Federated Bidirectional Knowledge Distillation (FedBKD). Specifically, we train Generative Adversarial Networks (GAN) for synthetic data. During the GAN training, local models serve as discriminators and their parameters are frozen. The synthetic data is then used for bidirectional distillation between global and local models to achieve knowledge interactions so that performances for both sides are improved. We conduct extensive experiments on 4 benchmarks under different non-IID settings. The results show that FedBKD achieves SOTA performances in every case. 

---
# Enhancing Large Language Models through Structured Reasoning 

**Authors**: Yubo Dong, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20241)  

**Abstract**: Recent Large Language Models (LLMs) have significantly advanced natural language processing and automated decision-making. However, these models still encounter difficulties when performing complex reasoning tasks involving logical deduction and systematic planning, primarily due to their reliance on implicit statistical relationships without structured knowledge this http URL by cognitive science and neurosymbolic AI, we introduce a novel approach to enhance LLMs through explicit structured reasoning. First, we convert unstructured data into structured formats by explicitly annotating reasoning steps. We then employ this structured dataset to train LLMs through Supervised Fine-Tuning (SFT). Additionally, we enhance the structured reasoning capabilities of LLMs using Group Relative Policy Optimization (GRPO), incorporating two innovative algorithms--MAX-Flow and Longest Common Subsequence (LCS)--which notably improve reasoning effectiveness and reduce computational complexity. Experimental results from fine-tuning a DeepSeek-R1-Distill-Qwen-1.5B model demonstrate concise reasoning, robust performance across various scenarios, and improved compatibility with optimization techniques, validating the efficacy of structured reasoning integration in LLMs. 

---
# Directed Link Prediction using GNN with Local and Global Feature Fusion 

**Authors**: Yuyang Zhang, Xu Shen, Yu Xie, Ka-Chun Wong, Weidun Xie, Chengbin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.20235)  

**Abstract**: Link prediction is a classical problem in graph analysis with many practical applications. For directed graphs, recently developed deep learning approaches typically analyze node similarities through contrastive learning and aggregate neighborhood information through graph convolutions. In this work, we propose a novel graph neural network (GNN) framework to fuse feature embedding with community information. We theoretically demonstrate that such hybrid features can improve the performance of directed link prediction. To utilize such features efficiently, we also propose an approach to transform input graphs into directed line graphs so that nodes in the transformed graph can aggregate more information during graph convolutions. Experiments on benchmark datasets show that our approach outperforms the state-of-the-art in most cases when 30%, 40%, 50%, and 60% of the connected links are used as training data, respectively. 

---
# Perspectives in Play: A Multi-Perspective Approach for More Inclusive NLP Systems 

**Authors**: Benedetta Muscato, Lucia Passaro, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.20209)  

**Abstract**: In the realm of Natural Language Processing (NLP), common approaches for handling human disagreement consist of aggregating annotators' viewpoints to establish a single ground truth. However, prior studies show that disregarding individual opinions can lead can lead to the side effect of underrepresenting minority perspectives, especially in subjective tasks, where annotators may systematically disagree because of their preferences. Recognizing that labels reflect the diverse backgrounds, life experiences, and values of individuals, this study proposes a new multi-perspective approach using soft labels to encourage the development of the next generation of perspective aware models, more inclusive and pluralistic. We conduct an extensive analysis across diverse subjective text classification tasks, including hate speech, irony, abusive language, and stance detection, to highlight the importance of capturing human disagreements, often overlooked by traditional aggregation methods. Results show that the multi-perspective approach not only better approximates human label distributions, as measured by Jensen-Shannon Divergence (JSD), but also achieves superior classification performance (higher F1 scores), outperforming traditional approaches. However, our approach exhibits lower confidence in tasks like irony and stance detection, likely due to the inherent subjectivity present in the texts. Lastly, leveraging Explainable AI (XAI), we explore model uncertainty and uncover meaningful insights into model predictions. 

---
# Affective Priming Score: A Data-Driven Method to Detect Priming in Sequential Datasets 

**Authors**: Eduardo Gutierrez Maestro, Hadi Banaee, Amy Loutfi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20204)  

**Abstract**: Affective priming exemplifies the challenge of ambiguity in affective computing. While the community has largely addressed this issue from a label-based perspective, identifying data points in the sequence affected by the priming effect, the impact of priming on data itself, particularly in physiological signals, remains underexplored. Data affected by priming can lead to misclassifications when used in learning models. This study proposes the Affective Priming Score (APS), a data-driven method to detect data points influenced by the priming effect. The APS assigns a score to each data point, quantifying the extent to which it is affected by priming. To validate this method, we apply it to the SEED and SEED-VII datasets, which contain sufficient transitions between emotional events to exhibit priming effects. We train models with the same configuration using both the original data and priming-free sequences. The misclassification rate is significantly reduced when using priming-free sequences compared to the original data. This work contributes to the broader challenge of ambiguity by identifying and mitigating priming effects at the data level, enhancing model robustness, and offering valuable insights for the design and collection of affective computing datasets. 

---
# How to Retrieve Examples in In-context Learning to Improve Conversational Emotion Recognition using Large Language Models? 

**Authors**: Mengqi Wang, Tiantian Feng, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20199)  

**Abstract**: Large language models (LLMs) have enabled a wide variety of real-world applications in various domains. However, creating a high-performing application with high accuracy remains challenging, particularly for subjective tasks like emotion recognition. Inspired by the SLT 2024 GenSER Challenge, this study investigates approaches to improving conversational emotion recognition (CER) by LLMs. Specifically, we explore how to retrieve high-quality examples in in-context learning (ICL) to enhance CER. We propose various strategies based on random and augmented example retrieval and also analyze the impact of conversational context on CER accuracy. Experiments were conducted on the three datasets including IEMOCAP, MELD and EmoryNLP. The results show that augmented example retrieval consistently outperforms other techniques under investigation across all datasets, highlighting the importance of retrieving coherent targeted examples and enhancing them through paraphrasing. 

---
# Zero-Shot Attribution for Large Language Models: A Distribution Testing Approach 

**Authors**: Clément L. Canonne, Yash Pote, Uddalok Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20197)  

**Abstract**: A growing fraction of all code is sampled from Large Language Models (LLMs). We investigate the problem of attributing code generated by language models using hypothesis testing to leverage established techniques and guarantees. Given a set of samples $S$ and a suspect model $\mathcal{L}^*$, our goal is to assess the likelihood of $S$ originating from $\mathcal{L}^*$. Due to the curse of dimensionality, this is intractable when only samples from the LLM are given: to circumvent this, we use both samples and density estimates from the LLM, a form of access commonly available.
We introduce $\mathsf{Anubis}$, a zero-shot attribution tool that frames attribution as a distribution testing problem. Our experiments on a benchmark of code samples show that $\mathsf{Anubis}$ achieves high AUROC scores ( $\ge0.9$) when distinguishing between LLMs like DeepSeek-Coder, CodeGemma, and Stable-Code using only $\approx 2000$ samples. 

---
# Progressive Alignment Degradation Learning for Pansharpening 

**Authors**: Enzhe Zhao, Zhichang Guo, Yao Li, Fanghui Song, Boying Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20179)  

**Abstract**: Deep learning-based pansharpening has been shown to effectively generate high-resolution multispectral (HRMS) images. To create supervised ground-truth HRMS images, synthetic data generated using the Wald protocol is commonly employed. This protocol assumes that networks trained on artificial low-resolution data will perform equally well on high-resolution data. However, well-trained models typically exhibit a trade-off in performance between reduced-resolution and full-resolution datasets. In this paper, we delve into the Wald protocol and find that its inaccurate approximation of real-world degradation patterns limits the generalization of deep pansharpening models. To address this issue, we propose the Progressive Alignment Degradation Module (PADM), which uses mutual iteration between two sub-networks, PAlignNet and PDegradeNet, to adaptively learn accurate degradation processes without relying on predefined operators. Building on this, we introduce HFreqdiff, which embeds high-frequency details into a diffusion framework and incorporates CFB and BACM modules for frequency-selective detail extraction and precise reverse process learning. These innovations enable effective integration of high-resolution panchromatic and multispectral images, significantly enhancing spatial sharpness and quality. Experiments and ablation studies demonstrate the proposed method's superior performance compared to state-of-the-art techniques. 

---
# COIN: Uncertainty-Guarding Selective Question Answering for Foundation Models with Provable Risk Guarantees 

**Authors**: Zhiyuan Wang, Jinhao Duan, Qingni Wang, Xiaofeng Zhu, Tianlong Chen, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20178)  

**Abstract**: Uncertainty quantification (UQ) for foundation models is essential to identify and mitigate potential hallucinations in automatically generated text. However, heuristic UQ approaches lack formal guarantees for key metrics such as the false discovery rate (FDR) in selective prediction. Previous work adopts the split conformal prediction (SCP) framework to ensure desired coverage of admissible answers by constructing prediction sets, but these sets often contain incorrect candidates, limiting their practical utility. To address this, we propose COIN, an uncertainty-guarding selection framework that calibrates statistically valid thresholds to filter a single generated answer per question under user-specified FDR constraints. COIN estimates the empirical error rate on a calibration set and applies confidence interval methods such as Clopper-Pearson to establish a high-probability upper bound on the true error rate (i.e., FDR). This enables the selection of the largest uncertainty threshold that ensures FDR control on test data while significantly increasing sample retention. We demonstrate COIN's robustness in risk control, strong test-time power in retaining admissible answers, and predictive efficiency under limited calibration data across both general and multimodal text generation tasks. Furthermore, we show that employing alternative upper bound constructions and UQ strategies can further boost COIN's power performance, which underscores its extensibility and adaptability to diverse application scenarios. 

---
# Valid Selection among Conformal Sets 

**Authors**: Mahmoud Hegazy, Liviu Aolaritei, Michael I. Jordan, Aymeric Dieuleveut  

**Link**: [PDF](https://arxiv.org/pdf/2506.20173)  

**Abstract**: Conformal prediction offers a distribution-free framework for constructing prediction sets with coverage guarantees. In practice, multiple valid conformal prediction sets may be available, arising from different models or methodologies. However, selecting the most desirable set, such as the smallest, can invalidate the coverage guarantees. To address this challenge, we propose a stability-based approach that ensures coverage for the selected prediction set. We extend our results to the online conformal setting, propose several refinements in settings where additional structure is available, and demonstrate its effectiveness through experiments. 

---
# SEED: A Structural Encoder for Embedding-Driven Decoding in Time Series Prediction with LLMs 

**Authors**: Fengze Li, Yue Wang, Yangle Liu, Ming Huang, Dou Hong, Jieming Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.20167)  

**Abstract**: Multivariate time series forecasting requires models to simultaneously capture variable-wise structural dependencies and generalize across diverse tasks. While structural encoders are effective in modeling feature interactions, they lack the capacity to support semantic-level reasoning or task adaptation. Conversely, large language models (LLMs) possess strong generalization capabilities but remain incompatible with raw time series inputs. This gap limits the development of unified, transferable prediction systems. Therefore, we introduce SEED, a structural encoder for embedding-driven decoding, which integrates four stages: a token-aware encoder for patch extraction, a projection module that aligns patches with language model embeddings, a semantic reprogramming mechanism that maps patches to task-aware prototypes, and a frozen language model for prediction. This modular architecture decouples representation learning from inference, enabling efficient alignment between numerical patterns and semantic reasoning. Empirical results demonstrate that the proposed method achieves consistent improvements over strong baselines, and comparative studies on various datasets confirm SEED's role in addressing the structural-semantic modeling gap. 

---
# Do psychic cells generate consciousness? 

**Authors**: Mototaka Suzuki, Jaan Aru  

**Link**: [PDF](https://arxiv.org/pdf/2506.20164)  

**Abstract**: Technological advances in the past decades have begun to enable neuroscientists to address fundamental questions about consciousness in an unprecedented way. Here we review remarkable recent progress in our understanding of cellular-level mechanisms of conscious processing in the brain. Of particular interest are the cortical pyramidal neurons -- or "psychic cells" called by Ramón y Cajal more than 100 years ago -- which have an intriguing cellular mechanism that accounts for selective disruption of feedback signaling in the brain upon anesthetic-induced loss of consciousness. Importantly, a particular class of metabotropic receptors distributed over the dendrites of pyramidal cells are highlighted as the key cellular mechanism. After all, Cajal's instinct over a century ago may turn out to be correct -- we may have just begun to understand whether and how psychic cells indeed generate and control our consciousness. 

---
# AI and Agile Software Development: From Frustration to Success -- XP2025 Workshop Summary 

**Authors**: Tomas Herda, Victoria Pichler, Zheying Zhang, Pekka Abrahamsson, Geir K. Hanssen  

**Link**: [PDF](https://arxiv.org/pdf/2506.20159)  

**Abstract**: The full-day workshop on AI and Agile at XP 2025 convened a diverse group of researchers and industry practitioners to address the practical challenges and opportunities of integrating Artificial Intelligence into Agile software development. Through interactive sessions, participants identified shared frustrations related to integrating AI into Agile Software Development practices, including challenges with tooling, governance, data quality, and critical skill gaps. These challenges were systematically prioritized and analyzed to uncover root causes. The workshop culminated in the collaborative development of a research roadmap that pinpoints actionable directions for future work, including both immediate solutions and ambitious long-term goals. The key outcome is a structured agenda designed to foster joint industry-academic efforts to move from identified frustrations to successful implementation. 

---
# Irec: A Metacognitive Scaffolding for Self-Regulated Learning through Just-in-Time Insight Recall: A Conceptual Framework and System Prototype 

**Authors**: Xuefei Hou, Xizhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20156)  

**Abstract**: The core challenge in learning has shifted from knowledge acquisition to effective Self-Regulated Learning (SRL): planning, monitoring, and reflecting on one's learning. Existing digital tools, however, inadequately support metacognitive reflection. Spaced Repetition Systems (SRS) use de-contextualized review, overlooking the role of context, while Personal Knowledge Management (PKM) tools require high manual maintenance.
To address these challenges, this paper introduces "Insight Recall," a novel paradigm that conceptualizes the context-triggered retrieval of personal past insights as a metacognitive scaffold to promote SRL. We formalize this paradigm using the Just-in-Time Adaptive Intervention (JITAI) framework and implement a prototype system, Irec, to demonstrate its feasibility. At its core, Irec uses a dynamic knowledge graph of the user's learning history. When a user faces a new problem, a hybrid retrieval engine recalls relevant personal "insights." Subsequently, a large language model (LLM) performs a deep similarity assessment to filter and present the most relevant scaffold in a just-in-time manner. To reduce cognitive load, Irec features a human-in-the-loop pipeline for LLM-based knowledge graph construction. We also propose an optional "Guided Inquiry" module, where users can engage in a Socratic dialogue with an expert LLM, using the current problem and recalled insights as context. The contribution of this paper is a solid theoretical framework and a usable system platform for designing next-generation intelligent learning systems that enhance metacognition and self-regulation. 

---
# Loss-Aware Automatic Selection of Structured Pruning Criteria for Deep Neural Network Acceleration 

**Authors**: Deepak Ghimire, Kilho Lee, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.20152)  

**Abstract**: Structured pruning is a well-established technique for compressing neural networks, making it suitable for deployment in resource-limited edge devices. This paper presents an efficient Loss-Aware Automatic Selection of Structured Pruning Criteria (LAASP) for slimming and accelerating deep neural networks. The majority of pruning methodologies employ a sequential process consisting of three stages: 1) training, 2) pruning, and 3) fine-tuning, whereas the proposed pruning technique adopts a pruning-while-training approach that eliminates the first stage and integrates the second and third stages into a single cycle. The automatic selection of magnitude or similarity-based filter pruning criteria from a specified pool of criteria and the specific pruning layer at each pruning iteration is guided by the network's overall loss on a small subset of the training data. To mitigate the abrupt accuracy drop due to pruning, the network is retrained briefly after each reduction of a predefined number of floating-point operations (FLOPs). The optimal pruning rates for each layer in the network are automatically determined, eliminating the need for manual allocation of fixed or variable pruning rates for each layer. Experiments on the VGGNet and ResNet models on the CIFAR-10 and ImageNet benchmark datasets demonstrate the effectiveness of the proposed method. In particular, the ResNet56 and ResNet110 models on the CIFAR-10 dataset significantly improve the top-1 accuracy compared to state-of-the-art methods while reducing the network FLOPs by 52\%. Furthermore, the ResNet50 model on the ImageNet dataset reduces FLOPs by more than 42\% with a negligible 0.33\% drop in top-5 accuracy. The source code of this paper is publicly available online - this https URL. 

---
# EAR: Erasing Concepts from Unified Autoregressive Models 

**Authors**: Haipeng Fan, Shiyuan Zhang, Baohunesitu, Zihang Guo, Huaiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20151)  

**Abstract**: Autoregressive (AR) models have achieved unified and strong performance across both visual understanding and image generation tasks. However, removing undesired concepts from AR models while maintaining overall generation quality remains an open challenge. In this paper, we propose Erasure Autoregressive Model (EAR), a fine-tuning method for effective and utility-preserving concept erasure in AR models. Specifically, we introduce Windowed Gradient Accumulation (WGA) strategy to align patch-level decoding with erasure objectives, and Thresholded Loss Masking (TLM) strategy to protect content unrelated to the target concept during fine-tuning. Furthermore, we propose a novel benchmark, Erase Concept Generator and Visual Filter (ECGVF), aim at provide a more rigorous and comprehensive foundation for evaluating concept erasure in AR models. Specifically, we first employ structured templates across diverse large language models (LLMs) to pre-generate a large-scale corpus of target-replacement concept prompt pairs. Subsequently, we generate images from these prompts and subject them to rigorous filtering via a visual classifier to ensure concept fidelity and alignment. Extensive experimental results conducted on the ECGVF benchmark with the AR model Janus-Pro demonstrate that EAR achieves marked improvements in both erasure effectiveness and model utility preservation. Code is available at: this https URL 

---
# CCRS: A Zero-Shot LLM-as-a-Judge Framework for Comprehensive RAG Evaluation 

**Authors**: Aashiq Muhamed  

**Link**: [PDF](https://arxiv.org/pdf/2506.20128)  

**Abstract**: RAG systems enhance LLMs by incorporating external knowledge, which is crucial for domains that demand factual accuracy and up-to-date information. However, evaluating the multifaceted quality of RAG outputs, spanning aspects such as contextual coherence, query relevance, factual correctness, and informational completeness, poses significant challenges. Existing evaluation methods often rely on simple lexical overlap metrics, which are inadequate for capturing these nuances, or involve complex multi-stage pipelines with intermediate steps like claim extraction or require finetuning specialized judge models, hindering practical efficiency. To address these limitations, we propose CCRS (Contextual Coherence and Relevance Score), a novel suite of five metrics that utilizes a single, powerful, pretrained LLM as a zero-shot, end-to-end judge. CCRS evaluates: Contextual Coherence (CC), Question Relevance (QR), Information Density (ID), Answer Correctness (AC), and Information Recall (IR). We apply CCRS to evaluate six diverse RAG system configurations on the challenging BioASQ dataset. Our analysis demonstrates that CCRS effectively discriminates between system performances, confirming, for instance, that the Mistral-7B reader outperforms Llama variants. We provide a detailed analysis of CCRS metric properties, including score distributions, convergent/discriminant validity, tie rates, population statistics, and discriminative power. Compared to the complex RAGChecker framework, CCRS offers comparable or superior discriminative power for key aspects like recall and faithfulness, while being significantly more computationally efficient. CCRS thus provides a practical, comprehensive, and efficient framework for evaluating and iteratively improving RAG systems. 

---
# BrokenVideos: A Benchmark Dataset for Fine-Grained Artifact Localization in AI-Generated Videos 

**Authors**: Jiahao Lin, Weixuan Peng, Bojia Zi, Yifeng Gao, Xianbiao Qi, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20103)  

**Abstract**: Recent advances in deep generative models have led to significant progress in video generation, yet the fidelity of AI-generated videos remains limited. Synthesized content often exhibits visual artifacts such as temporally inconsistent motion, physically implausible trajectories, unnatural object deformations, and local blurring that undermine realism and user trust. Accurate detection and spatial localization of these artifacts are crucial for both automated quality control and for guiding the development of improved generative models. However, the research community currently lacks a comprehensive benchmark specifically designed for artifact localization in AI generated videos. Existing datasets either restrict themselves to video or frame level detection or lack the fine-grained spatial annotations necessary for evaluating localization methods. To address this gap, we introduce BrokenVideos, a benchmark dataset of 3,254 AI-generated videos with meticulously annotated, pixel-level masks highlighting regions of visual corruption. Each annotation is validated through detailed human inspection to ensure high quality ground truth. Our experiments show that training state of the art artifact detection models and multi modal large language models (MLLMs) on BrokenVideos significantly improves their ability to localize corrupted regions. Through extensive evaluation, we demonstrate that BrokenVideos establishes a critical foundation for benchmarking and advancing research on artifact localization in generative video models. The dataset is available at: this https URL. 

---
# MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agricultural Expert-Guided Conversations 

**Authors**: Vardhan Dongre, Chi Gui, Shubham Garg, Hooshang Nayyeri, Gokhan Tur, Dilek Hakkani-Tür, Vikram S. Adve  

**Link**: [PDF](https://arxiv.org/pdf/2506.20100)  

**Abstract**: We introduce MIRAGE, a new benchmark for multimodal expert-level reasoning and decision-making in consultative interaction settings. Designed for the agriculture domain, MIRAGE captures the full complexity of expert consultations by combining natural user queries, expert-authored responses, and image-based context, offering a high-fidelity benchmark for evaluating models on grounded reasoning, clarification strategies, and long-form generation in a real-world, knowledge-intensive domain. Grounded in over 35,000 real user-expert interactions and curated through a carefully designed multi-step pipeline, MIRAGE spans diverse crop health, pest diagnosis, and crop management scenarios. The benchmark includes more than 7,000 unique biological entities, covering plant species, pests, and diseases, making it one of the most taxonomically diverse benchmarks available for vision-language models, grounded in the real world. Unlike existing benchmarks that rely on well-specified user inputs and closed-set taxonomies, MIRAGE features underspecified, context-rich scenarios with open-world settings, requiring models to infer latent knowledge gaps, handle rare entities, and either proactively guide the interaction or respond. Project Page: this https URL 

---
# SACL: Understanding and Combating Textual Bias in Code Retrieval with Semantic-Augmented Reranking and Localization 

**Authors**: Dhruv Gupta, Gayathri Ganesh Lakshmy, Yiqing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20081)  

**Abstract**: Retrieval-Augmented Code Generation (RACG) is a critical technique for enhancing code generation by retrieving relevant information. In this work, we conduct an in-depth analysis of code retrieval by systematically masking specific features while preserving code functionality. Our discoveries include: (1) although trained on code, current retrievers heavily rely on surface-level textual features (e.g., docstrings, identifier names), and (2) they exhibit a strong bias towards well-documented code, even if the documentation is this http URL on our discoveries, we propose SACL, a framework that enriches textual information and reduces bias by augmenting code or structural knowledge with semantic information. Extensive experiments show that SACL substantially improves code retrieval (e.g., by 12.8% / 9.4% / 7.0% Recall@1 on HumanEval / MBPP / SWE-Bench-Lite), which also leads to better code generation performance (e.g., by 4.88% Pass@1 on HumanEval). 

---
# A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs 

**Authors**: Kethmi Hirushini Hettige, Jiahao Ji, Cheng Long, Shili Xiang, Gao Cong, Jingyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20073)  

**Abstract**: Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that require generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without requiring task-specific finetuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both solutions and detailed rationales. To facilitate rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results show that STReason significantly outperforms advanced LLM baselines across all metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason's credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks. We believe STReason provides a promising direction for developing more capable and generalizable spatio-temporal reasoning systems. 

---
# Beyond Autocomplete: Designing CopilotLens Towards Transparent and Explainable AI Coding Agents 

**Authors**: Runlong Ye, Zeling Zhang, Boushra Almazroua, Michael Liut  

**Link**: [PDF](https://arxiv.org/pdf/2506.20062)  

**Abstract**: AI-powered code assistants are widely used to generate code completions, significantly boosting developer productivity. However, these tools typically present suggestions without explaining their rationale, leaving their decision-making process inscrutable. This opacity hinders developers' ability to critically evaluate the output, form accurate mental models, and build calibrated trust in the system. To address this, we introduce CopilotLens, a novel interactive framework that reframes code completion from a simple suggestion into a transparent, explainable event. CopilotLens operates as an explanation layer that reveals the AI agent's "thought process" through a dynamic two-level interface, surfacing everything from its reconstructed high-level plans to the specific codebase context influencing the code. This paper presents the design and rationale of CopilotLens, offering a concrete framework for building future agentic code assistants that prioritize clarity of reasoning over speed of suggestion, thereby fostering deeper comprehension and more robust human-AI collaboration. 

---
# Robust Robotic Exploration and Mapping Using Generative Occupancy Map Synthesis 

**Authors**: Lorin Achey, Alec Reed, Brendan Crowe, Bradley Hayes, Christoffer Heckman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20049)  

**Abstract**: We present a novel approach for enhancing robotic exploration by using generative occupancy mapping. We introduce SceneSense, a diffusion model designed and trained for predicting 3D occupancy maps given partial observations. Our proposed approach probabilistically fuses these predictions into a running occupancy map in real-time, resulting in significant improvements in map quality and traversability. We implement SceneSense onboard a quadruped robot and validate its performance with real-world experiments to demonstrate the effectiveness of the model. In these experiments, we show that occupancy maps enhanced with SceneSense predictions better represent our fully observed ground truth data (24.44% FID improvement around the robot and 75.59% improvement at range). We additionally show that integrating SceneSense-enhanced maps into our robotic exploration stack as a "drop-in" map improvement, utilizing an existing off-the-shelf planner, results in improvements in robustness and traversability time. Finally we show results of full exploration evaluations with our proposed system in two dissimilar environments and find that locally enhanced maps provide more consistent exploration results than maps constructed only from direct sensor measurements. 

---
# GNN's Uncertainty Quantification using Self-Distillation 

**Authors**: Hirad Daneshvar, Reza Samavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20046)  

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in the healthcare domain. However, what remained challenging is quantifying the predictive uncertainty of GNNs, which is an important aspect of trustworthiness in clinical settings. While Bayesian and ensemble methods can be used to quantify uncertainty, they are computationally expensive. Additionally, the disagreement metric used by ensemble methods to compute uncertainty cannot capture the diversity of models in an ensemble network. In this paper, we propose a novel method, based on knowledge distillation, to quantify GNNs' uncertainty more efficiently and with higher precision. We apply self-distillation, where the same network serves as both the teacher and student models, thereby avoiding the need to train several networks independently. To ensure the impact of self-distillation, we develop an uncertainty metric that captures the diverse nature of the network by assigning different weights to each GNN classifier. We experimentally evaluate the precision, performance, and ability of our approach in distinguishing out-of-distribution data on two graph datasets: MIMIC-IV and Enzymes. The evaluation results demonstrate that the proposed method can effectively capture the predictive uncertainty of the model while having performance similar to that of the MC Dropout and ensemble methods. The code is publicly available at this https URL. 

---
# LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification 

**Authors**: Soheil Abadifard, Fazli Can  

**Link**: [PDF](https://arxiv.org/pdf/2506.20041)  

**Abstract**: The classification of imbalanced data streams, which have unequal class distributions, is a key difficulty in machine learning, especially when dealing with multiple classes. While binary imbalanced data stream classification tasks have received considerable attention, only a few studies have focused on multi-class imbalanced data streams. Effectively managing the dynamic imbalance ratio is a key challenge in this domain. This study introduces a novel, robust, and resilient approach to address these challenges by integrating Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) into the Dynamic Ensemble Diversification (DynED) framework. To the best of our knowledge, we present the first application of LSH-RHP for undersampling in the context of imbalanced non-stationary data streams. The proposed method undersamples the majority classes by utilizing LSH-RHP, provides a balanced training set, and improves the ensemble's prediction performance. We conduct comprehensive experiments on 23 real-world and ten semi-synthetic datasets and compare LSH-DynED with 15 state-of-the-art methods. The results reveal that LSH-DynED outperforms other approaches in terms of both Kappa and mG-Mean effectiveness measures, demonstrating its capability in dealing with multi-class imbalanced non-stationary data streams. Notably, LSH-DynED performs well in large-scale, high-dimensional datasets with considerable class imbalances and demonstrates adaptation and robustness in real-world circumstances. To motivate our design, we review existing methods for imbalanced data streams, outline key challenges, and offer guidance for future work. For the reproducibility of our results, we have made our implementation available on GitHub. 

---
# Cross-Layer Discrete Concept Discovery for Interpreting Language Models 

**Authors**: Ankur Garg, Xuemin Yu, Hassan Sajjad, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20040)  

**Abstract**: Uncovering emergent concepts across transformer layers remains a significant challenge because the residual stream linearly mixes and duplicates information, obscuring how features evolve within large language models. Current research efforts primarily inspect neural representations at single layers, thereby overlooking this cross-layer superposition and the redundancy it introduces. These representations are typically either analyzed directly for activation patterns or passed to probing classifiers that map them to a limited set of predefined concepts. To address these limitations, we propose \gls{clvqvae}, a framework that uses vector quantization to map representations across layers and in the process collapse duplicated residual-stream features into compact, interpretable concept vectors. Our approach uniquely combines top-$k$ temperature-based sampling during quantization with EMA codebook updates, providing controlled exploration of the discrete latent space while maintaining code-book diversity. We further enhance the framework with scaled-spherical k-means++ for codebook initialization, which clusters by directional similarity rather than magnitude, better aligning with semantic structure in word embedding space. 

---
# Learning Bilateral Team Formation in Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Koorosh Moslemi, Chi-Guhn Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.20039)  

**Abstract**: Team formation and the dynamics of team-based learning have drawn significant interest in the context of Multi-Agent Reinforcement Learning (MARL). However, existing studies primarily focus on unilateral groupings, predefined teams, or fixed-population settings, leaving the effects of algorithmic bilateral grouping choices in dynamic populations underexplored. To address this gap, we introduce a framework for learning two-sided team formation in dynamic multi-agent systems. Through this study, we gain insight into what algorithmic properties in bilateral team formation influence policy performance and generalization. We validate our approach using widely adopted multi-agent scenarios, demonstrating competitive performance and improved generalization in most scenarios. 

---
# Hierarchical Reinforcement Learning and Value Optimization for Challenging Quadruped Locomotion 

**Authors**: Jeremiah Coholich, Muhammad Ali Murtaza, Seth Hutchinson, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.20036)  

**Abstract**: We propose a novel hierarchical reinforcement learning framework for quadruped locomotion over challenging terrain. Our approach incorporates a two-layer hierarchy in which a high-level policy (HLP) selects optimal goals for a low-level policy (LLP). The LLP is trained using an on-policy actor-critic RL algorithm and is given footstep placements as goals. We propose an HLP that does not require any additional training or environment samples and instead operates via an online optimization process over the learned value function of the LLP. We demonstrate the benefits of this framework by comparing it with an end-to-end reinforcement learning (RL) approach. We observe improvements in its ability to achieve higher rewards with fewer collisions across an array of different terrains, including terrains more difficult than any encountered during training. 

---
# Automated Generation of Diverse Courses of Actions for Multi-Agent Operations using Binary Optimization and Graph Learning 

**Authors**: Prithvi Poddar, Ehsan Tarkesh Esfahani, Karthik Dantu, Souma Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.20031)  

**Abstract**: Operations in disaster response, search \& rescue, and military missions that involve multiple agents demand automated processes to support the planning of the courses of action (COA). Moreover, traverse-affecting changes in the environment (rain, snow, blockades, etc.) may impact the expected performance of a COA, making it desirable to have a pool of COAs that are diverse in task distributions across agents. Further, variations in agent capabilities, which could be human crews and/or autonomous systems, present practical opportunities and computational challenges to the planning process. This paper presents a new theoretical formulation and computational framework to generate such diverse pools of COAs for operations with soft variations in agent-task compatibility. Key to the problem formulation is a graph abstraction of the task space and the pool of COAs itself to quantify its diversity. Formulating the COAs as a centralized multi-robot task allocation problem, a genetic algorithm is used for (order-ignoring) allocations of tasks to each agent that jointly maximize diversity within the COA pool and overall compatibility of the agent-task mappings. A graph neural network is trained using a policy gradient approach to then perform single agent task sequencing in each COA, which maximizes completion rates adaptive to task features. Our tests of the COA generation process in a simulated environment demonstrate significant performance gain over a random walk baseline, small optimality gap in task sequencing, and execution time of about 50 minutes to plan up to 20 COAs for 5 agent/100 task operations. 

---
# Elucidated Rolling Diffusion Models for Probabilistic Weather Forecasting 

**Authors**: Salva Rühling Cachay, Miika Aittala, Karsten Kreis, Noah Brenowitz, Arash Vahdat, Morteza Mardani, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20024)  

**Abstract**: Diffusion models are a powerful tool for probabilistic forecasting, yet most applications in high-dimensional chaotic systems predict future snapshots one-by-one. This common approach struggles to model complex temporal dependencies and fails to explicitly account for the progressive growth of uncertainty inherent to such systems. While rolling diffusion frameworks, which apply increasing noise to forecasts at longer lead times, have been proposed to address this, their integration with state-of-the-art, high-fidelity diffusion techniques remains a significant challenge. We tackle this problem by introducing Elucidated Rolling Diffusion Models (ERDM), the first framework to successfully unify a rolling forecast structure with the principled, performant design of Elucidated Diffusion Models (EDM). To do this, we adapt the core EDM components-its noise schedule, network preconditioning, and Heun sampler-to the rolling forecast setting. The success of this integration is driven by three key contributions: (i) a novel loss weighting scheme that focuses model capacity on the mid-range forecast horizons where determinism gives way to stochasticity; (ii) an efficient initialization strategy using a pre-trained EDM for the initial window; and (iii) a bespoke hybrid sequence architecture for robust spatiotemporal feature extraction under progressive denoising. On 2D Navier-Stokes simulations and ERA5 global weather forecasting at 1.5^\circ resolution, ERDM consistently outperforms key diffusion-based baselines, including conditional autoregressive EDM. ERDM offers a flexible and powerful general framework for tackling diffusion-based sequence generation problems where modeling escalating uncertainty is paramount. Code is available at: this https URL 

---
# New Insights on Unfolding and Fine-tuning Quantum Federated Learning 

**Authors**: Shanika Iroshi Nanayakkara, Shiva Raj Pokhrel  

**Link**: [PDF](https://arxiv.org/pdf/2506.20016)  

**Abstract**: Client heterogeneity poses significant challenges to the performance of Quantum Federated Learning (QFL). To overcome these limitations, we propose a new approach leveraging deep unfolding, which enables clients to autonomously optimize hyperparameters, such as learning rates and regularization factors, based on their specific training behavior. This dynamic adaptation mitigates overfitting and ensures robust optimization in highly heterogeneous environments where standard aggregation methods often fail. Our framework achieves approximately 90% accuracy, significantly outperforming traditional methods, which typically yield around 55% accuracy, as demonstrated through real-time training on IBM quantum hardware and Qiskit Aer simulators. By developing self adaptive fine tuning, the proposed method proves particularly effective in critical applications such as gene expression analysis and cancer detection, enhancing diagnostic precision and predictive modeling within quantum systems. Our results are attributed to convergence-aware, learnable optimization steps intrinsic to the deep unfolded framework, which maintains the generalization. Hence, this study addresses the core limitations of conventional QFL, streamlining its applicability to any complex challenges such as healthcare and genomic research. 

---
# TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design 

**Authors**: Geonwoo Cho, Jaegyun Im, Jihwan Lee, Hojun Yi, Sejin Kim, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.19997)  

**Abstract**: Generalizing deep reinforcement learning agents to unseen environments remains a significant challenge. One promising solution is Unsupervised Environment Design (UED), a co-evolutionary framework in which a teacher adaptively generates tasks with high learning potential, while a student learns a robust policy from this evolving curriculum. Existing UED methods typically measure learning potential via regret, the gap between optimal and current performance, approximated solely by value-function loss. Building on these approaches, we introduce the transition prediction error as an additional term in our regret approximation. To capture how training on one task affects performance on others, we further propose a lightweight metric called co-learnability. By combining these two measures, we present Transition-aware Regret Approximation with Co-learnability for Environment Design (TRACED). Empirical evaluations show that TRACED yields curricula that improve zero-shot generalization across multiple benchmarks while requiring up to 2x fewer environment interactions than strong baselines. Ablation studies confirm that the transition prediction error drives rapid complexity ramp-up and that co-learnability delivers additional gains when paired with the transition prediction error. These results demonstrate how refined regret approximation and explicit modeling of task relationships can be leveraged for sample-efficient curriculum design in UED. 

---
# HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization 

**Authors**: Gabor Petnehazi, Bernadett Aradi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19992)  

**Abstract**: The explosive growth of complex datasets across various modalities necessitates advanced analytical tools that not only group data effectively but also provide human-understandable insights into the discovered structures. We introduce HERCULES (Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization), a novel algorithm and Python package designed for hierarchical k-means clustering of diverse data types, including text, images, and numeric data (processed one modality per run). HERCULES constructs a cluster hierarchy by recursively applying k-means clustering, starting from individual data points at level 0. A key innovation is its deep integration of Large Language Models (LLMs) to generate semantically rich titles and descriptions for clusters at each level of the hierarchy, significantly enhancing interpretability. The algorithm supports two main representation modes: `direct' mode, which clusters based on original data embeddings or scaled numeric features, and `description' mode, which clusters based on embeddings derived from LLM-generated summaries. Users can provide a `topic\_seed' to guide LLM-generated summaries towards specific themes. An interactive visualization tool facilitates thorough analysis and understanding of the clustering results. We demonstrate HERCULES's capabilities and discuss its potential for extracting meaningful, hierarchical knowledge from complex datasets. 

---
# VoxelOpt: Voxel-Adaptive Message Passing for Discrete Optimization in Deformable Abdominal CT Registration 

**Authors**: Hang Zhang, Yuxi Zhang, Jiazheng Wang, Xiang Chen, Renjiu Hu, Xin Tian, Gaolei Li, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19975)  

**Abstract**: Recent developments in neural networks have improved deformable image registration (DIR) by amortizing iterative optimization, enabling fast and accurate DIR results. However, learning-based methods often face challenges with limited training data, large deformations, and tend to underperform compared to iterative approaches when label supervision is unavailable. While iterative methods can achieve higher accuracy in such scenarios, they are considerably slower than learning-based methods. To address these limitations, we propose VoxelOpt, a discrete optimization-based DIR framework that combines the strengths of learning-based and iterative methods to achieve a better balance between registration accuracy and runtime. VoxelOpt uses displacement entropy from local cost volumes to measure displacement signal strength at each voxel, which differs from earlier approaches in three key aspects. First, it introduces voxel-wise adaptive message passing, where voxels with lower entropy receives less influence from their neighbors. Second, it employs a multi-level image pyramid with 27-neighbor cost volumes at each level, avoiding exponential complexity growth. Third, it replaces hand-crafted features or contrastive learning with a pretrained foundational segmentation model for feature extraction. In abdominal CT registration, these changes allow VoxelOpt to outperform leading iterative in both efficiency and accuracy, while matching state-of-the-art learning-based methods trained with label supervision. The source code will be available at this https URL 

---
# Quantum Neural Networks for Propensity Score Estimation and Survival Analysis in Observational Biomedical Studies 

**Authors**: Vojtěch Novák, Ivan Zelinka, Lenka Přibylová, Lubomír Martínek  

**Link**: [PDF](https://arxiv.org/pdf/2506.19973)  

**Abstract**: This study investigates the application of quantum neural networks (QNNs) for propensity score estimation to address selection bias in comparing survival outcomes between laparoscopic and open surgical techniques in a cohort of 1177 colorectal carcinoma patients treated at University Hospital Ostrava (2001-2009). Using a dataset with 77 variables, including patient demographics and tumor characteristics, we developed QNN-based propensity score models focusing on four key covariates (Age, Sex, Stage, BMI). The QNN architecture employed a linear ZFeatureMap for data encoding, a SummedPaulis operator for predictions, and the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for robust, gradient-free optimization in noisy quantum environments. Variance regularization was integrated to mitigate quantum measurement noise, with simulations conducted under exact, sampling (1024 shots), and noisy hardware (FakeManhattanV2) conditions. QNNs, particularly with simulated hardware noise, outperformed classical logistic regression and gradient boosted machines in small samples (AUC up to 0.750 for n=100), with noise modeling enhancing predictive stability. Propensity score matching and weighting, optimized via genetic matching and matching weights, achieved covariate balance with standardized mean differences of 0.0849 and 0.0869, respectively. Survival analyses using Kaplan-Meier estimation, Cox proportional hazards, and Aalen additive regression revealed no significant survival differences post-adjustment (p-values 0.287-0.851), indicating confounding bias in unadjusted outcomes. These results highlight QNNs' potential, enhanced by CMA-ES and noise-aware strategies, to improve causal inference in biomedical research, particularly for small-sample, high-dimensional datasets. 

---
# Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs 

**Authors**: Travis Thompson, Seung-Hwan Lim, Paul Liu, Ruoying He, Dongkuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19967)  

**Abstract**: Large Language Models (LLMs) have achieved impressive capabilities in language understanding and generation, yet they continue to underperform on knowledge-intensive reasoning tasks due to limited access to structured context and multi-hop information. Retrieval-Augmented Generation (RAG) partially mitigates this by grounding generation in retrieved context, but conventional RAG and GraphRAG methods often fail to capture relational structure across nodes in knowledge graphs. We introduce Inference-Scaled GraphRAG, a novel framework that enhances LLM-based graph reasoning by applying inference-time compute scaling. Our method combines sequential scaling with deep chain-of-thought graph traversal, and parallel scaling with majority voting over sampled trajectories within an interleaved reasoning-execution loop. Experiments on the GRBench benchmark demonstrate that our approach significantly improves multi-hop question answering performance, achieving substantial gains over both traditional GraphRAG and prior graph traversal baselines. These findings suggest that inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs 

---
# An ab initio foundation model of wavefunctions that accurately describes chemical bond breaking 

**Authors**: Adam Foster, Zeno Schätzle, P. Bernát Szabó, Lixue Cheng, Jonas Köhler, Gino Cassella, Nicholas Gao, Jiawei Li, Frank Noé, Jan Hermann  

**Link**: [PDF](https://arxiv.org/pdf/2506.19960)  

**Abstract**: Reliable description of bond breaking remains a major challenge for quantum chemistry due to the multireferential character of the electronic structure in dissociating species. Multireferential methods in particular suffer from large computational cost, which under the normal paradigm has to be paid anew for each system at a full price, ignoring commonalities in electronic structure across molecules. Quantum Monte Carlo with deep neural networks (deep QMC) uniquely offers to exploit such commonalities by pretraining transferable wavefunction models, but all such attempts were so far limited in scope. Here, we bring this new paradigm to fruition with Orbformer, a novel transferable wavefunction model pretrained on 22,000 equilibrium and dissociating structures that can be fine-tuned on unseen molecules reaching an accuracy-cost ratio rivalling classical multireferential methods. On established benchmarks as well as more challenging bond dissociations and Diels-Alder reactions, Orbformer is the only method that consistently converges to chemical accuracy (1 kcal/mol). This work turns the idea of amortizing the cost of solving the Schrödinger equation over many molecules into a practical approach in quantum chemistry. 

---
# CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation 

**Authors**: Deepon Halder, Thanmay Jayakumar, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2506.19952)  

**Abstract**: Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality. 

---
# Can LLMs Replace Humans During Code Chunking? 

**Authors**: Christopher Glasz, Emily Escamilla, Eric O. Scott, Anand Patel, Jacob Zimmer, Colin Diggs, Michael Doyle, Scott Rosen, Nitin Naik, Justin F. Brunelle, Samruddhi Thaker, Parthav Poudel, Arun Sridharan, Amit Madan, Doug Wendt, William Macke, Thomas Schill  

**Link**: [PDF](https://arxiv.org/pdf/2506.19897)  

**Abstract**: Large language models (LLMs) have become essential tools in computer science, especially for tasks involving code understanding and generation. However, existing work does not address many of the unique challenges presented by code written for government applications. In particular, government enterprise software is often written in legacy languages like MUMPS or assembly language code (ALC) and the overall token lengths of these systems exceed the context window size for current commercially available LLMs. Additionally, LLMs are primarily trained on modern software languages and have undergone limited testing with legacy languages, making their ability to understand legacy languages unknown and, hence, an area for empirical study. This paper examines the application of LLMs in the modernization of legacy government code written in ALC and MUMPS, addressing the challenges of input limitations. We investigate various code-chunking methods to optimize the generation of summary module comments for legacy code files, evaluating the impact of code-chunking methods on the quality of documentation produced by different LLMs, including GPT-4o, Claude 3 Sonnet, Mixtral, and Llama 3. Our results indicate that LLMs can select partition points closely aligned with human expert partitioning. We also find that chunking approaches have significant impact on downstream tasks such as documentation generation. LLM-created partitions produce comments that are up to 20% more factual and up to 10% more useful than when humans create partitions. Therefore, we conclude that LLMs can be used as suitable replacements for human partitioning of large codebases during LLM-aided modernization. 

---
# A Framework for Uncertainty Quantification Based on Nearest Neighbors Across Layers 

**Authors**: Miguel N. Font, José L. Jorro-Aragoneses, Carlos M. Alaíz  

**Link**: [PDF](https://arxiv.org/pdf/2506.19895)  

**Abstract**: Neural Networks have high accuracy in solving problems where it is difficult to detect patterns or create a logical model. However, these algorithms sometimes return wrong solutions, which become problematic in high-risk domains like medical diagnosis or autonomous driving. One strategy to detect and mitigate these errors is the measurement of the uncertainty over neural network decisions. In this paper, we present a novel post-hoc framework for measuring the uncertainty of a decision based on retrieved training cases that have a similar activation vector to the query for each layer. Based on these retrieved cases, we propose two new metrics: Decision Change and Layer Uncertainty, which capture changes in nearest-neighbor class distributions across layers. We evaluated our approach in a classification model for two datasets: CIFAR-10 and MNIST. The results show that these metrics enhance uncertainty estimation, especially in challenging classification tasks, outperforming softmax-based confidence. 

---
# Explaining deep neural network models for electricity price forecasting with XAI 

**Authors**: Antoine Pesenti, Aidan OSullivan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19894)  

**Abstract**: Electricity markets are highly complex, involving lots of interactions and complex dependencies that make it hard to understand the inner workings of the market and what is driving prices. Econometric methods have been developed for this, white-box models, however, they are not as powerful as deep neural network models (DNN). In this paper, we use a DNN to forecast the price and then use XAI methods to understand the factors driving the price dynamics in the market. The objective is to increase our understanding of how different electricity markets work. To do that, we apply explainable methods such as SHAP and Gradient, combined with visual techniques like heatmaps (saliency maps) to analyse the behaviour and contributions of various features across five electricity markets. We introduce the novel concepts of SSHAP values and SSHAP lines to enhance the complex representation of high-dimensional tabular models. 

---
# Distillation-Enabled Knowledge Alignment for Generative Semantic Communications in AIGC Provisioning Tasks 

**Authors**: Jingzhi Hu, Geoffrey Ye Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19893)  

**Abstract**: Due to the surging amount of AI-generated content (AIGC), its provisioning to edges and mobile users from the cloud incurs substantial traffic on networks. Generative semantic communication (GSC) offers a promising solution by transmitting highly compact information, i.e., prompt text and latent representations, instead of high-dimensional AIGC data. However, GSC relies on the alignment between the knowledge in the cloud generative AI (GAI) and that possessed by the edges and users, and between the knowledge for wireless transmission and that of actual channels, which remains challenging. In this paper, we propose DeKA-g, a distillation-enabled knowledge alignment algorithm for GSC systems. The core idea is to distill the generation knowledge from the cloud-GAI into low-rank matrices, which can be incorporated by the edge and used to adapt the transmission knowledge to diverse wireless channel conditions. DeKA-g comprises two novel methods: metaword-aided knowledge distillation (MAKD) and variable-rate grouped SNR adaptation (VGSA). For MAKD, an optimized metaword is employed to enhance the efficiency of knowledge distillation, while VGSA enables efficient adaptation to diverse compression rates and SNR ranges. From simulation results, DeKA-g improves the alignment between the edge-generated images and the cloud-generated ones by 44%. Moreover, it adapts to compression rates with 116% higher efficiency than the baseline and enhances the performance in low-SNR conditions by 28%. 

---
# RepuNet: A Reputation System for Mitigating Malicious Clients in DFL 

**Authors**: Isaac Marroqui Penalva, Enrique Tomás Martínez Beltrán, Manuel Gil Pérez, Alberto Huertas Celdrán  

**Link**: [PDF](https://arxiv.org/pdf/2506.19892)  

**Abstract**: Decentralized Federated Learning (DFL) enables nodes to collaboratively train models without a central server, introducing new vulnerabilities since each node independently selects peers for model aggregation. Malicious nodes may exploit this autonomy by sending corrupted models (model poisoning), delaying model submissions (delay attack), or flooding the network with excessive messages, negatively affecting system performance. Existing solutions often depend on rigid configurations or additional infrastructures such as blockchain, leading to computational overhead, scalability issues, or limited adaptability. To overcome these limitations, this paper proposes RepuNet, a decentralized reputation system that categorizes threats in DFL and dynamically evaluates node behavior using metrics like model similarity, parameter changes, message latency, and communication volume. Nodes' influence in model aggregation is adjusted based on their reputation scores. RepuNet was integrated into the Nebula DFL platform and experimentally evaluated with MNIST and CIFAR-10 datasets under non-IID distributions, using federations of up to 25 nodes in both fully connected and random topologies. Different attack intensities, frequencies, and activation intervals were tested. Results demonstrated that RepuNet effectively detects and mitigates malicious behavior, achieving F1 scores above 95% for MNIST scenarios and approximately 76% for CIFAR-10 cases. These outcomes highlight RepuNet's adaptability, robustness, and practical potential for mitigating threats in decentralized federated learning environments. 

---
# Orthogonal Soft Pruning for Efficient Class Unlearning 

**Authors**: Qinghui Gong, Xue Yang, Xiaohu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19891)  

**Abstract**: Machine unlearning aims to selectively remove class-specific knowledge from pretrained neural networks to satisfy privacy regulations such as the GDPR. Existing methods typically face a trade-off between unlearning speed and preservation of predictive accuracy, often incurring either high computational overhead or significant performance degradation on retained classes. In this paper, we propose a novel class-aware soft pruning framework leveraging orthogonal convolutional kernel regularization to achieve rapid and precise forgetting with millisecond-level response times. By enforcing orthogonality constraints during training, our method decorrelates convolutional filters and disentangles feature representations, while efficiently identifying class-specific channels through activation difference analysis. Extensive evaluations across multiple architectures and datasets demonstrate stable pruning with near-instant execution, complete forgetting of targeted classes, and minimal accuracy loss on retained data. Experiments on CIFAR-10, CIFAR-100, and TinyImageNet confirm that our approach substantially reduces membership inference attack risks and accelerates unlearning by orders of magnitude compared to state-of-the-art baselines. This framework provides an efficient, practical solution for real-time machine unlearning in Machine Learning as a Service (MLaaS) scenarios. 

---
# Causal-Aware Intelligent QoE Optimization for VR Interaction with Adaptive Keyframe Extraction 

**Authors**: Ziru Zhang, Jiadong Yu, Danny H.K. Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19890)  

**Abstract**: The optimization of quality of experience (QoE) in multi-user virtual reality (VR) interactions demands a delicate balance between ultra-low latency, high-fidelity motion synchronization, and equitable resource allocation. While adaptive keyframe extraction mitigates transmission overhead, existing approaches often overlook the causal relationships among allocated bandwidth, CPU frequency, and user perception, limiting QoE gains. This paper proposes an intelligent framework to maximize QoE by integrating adaptive keyframe extraction with causal-aware reinforcement learning (RL). First, a novel QoE metric is formulated using the Weber-Fechner Law, combining perceptual sensitivity, attention-driven priorities, and motion reconstruction accuracy. The QoE optimization problem is then modeled as a mixed integer programming (MIP) task, jointly optimizing keyframe ratios, bandwidth, and computational resources under horizon-fairness constraints. We propose Partial State Causal Deep Deterministic Policy Gradient (PS-CDDPG), which integrates the Deep Deterministic Policy Gradient (DDPG) method with causal influence detection. By leveraging causal information regarding how QoE is influenced and determined by various actions, we explore actions guided by weights calculated from causal inference (CI), which in turn improves training efficiency. Experiments conducted with the CMU Motion Capture Database demonstrate that our framework significantly reduces interactive latency, enhances QoE, and maintains fairness, achieving superior performance compared to benchmark methods. 

---
# Retrieval-Confused Generation is a Good Defender for Privacy Violation Attack of Large Language Models 

**Authors**: Wanli Peng, Xin Chen, Hang Fu, XinYu He, Xue Yiming, Juan Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19889)  

**Abstract**: Recent advances in large language models (LLMs) have made a profound impact on our society and also raised new security concerns. Particularly, due to the remarkable inference ability of LLMs, the privacy violation attack (PVA), revealed by Staab et al., introduces serious personal privacy issues. Existing defense methods mainly leverage LLMs to anonymize the input query, which requires costly inference time and cannot gain satisfactory defense performance. Moreover, directly rejecting the PVA query seems like an effective defense method, while the defense method is exposed, promoting the evolution of PVA. In this paper, we propose a novel defense paradigm based on retrieval-confused generation (RCG) of LLMs, which can efficiently and covertly defend the PVA. We first design a paraphrasing prompt to induce the LLM to rewrite the "user comments" of the attack query to construct a disturbed database. Then, we propose the most irrelevant retrieval strategy to retrieve the desired user data from the disturbed database. Finally, the "data comments" are replaced with the retrieved user data to form a defended query, leading to responding to the adversary with some wrong personal attributes, i.e., the attack fails. Extensive experiments are conducted on two datasets and eight popular LLMs to comprehensively evaluate the feasibility and the superiority of the proposed defense method. 

---
# MATER: Multi-level Acoustic and Textual Emotion Representation for Interpretable Speech Emotion Recognition 

**Authors**: Hyo Jin Jon, Longbin Jin, Hyuntaek Jung, Hyunseo Kim, Donghun Min, Eun Yi Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.19887)  

**Abstract**: This paper presents our contributions to the Speech Emotion Recognition in Naturalistic Conditions (SERNC) Challenge, where we address categorical emotion recognition and emotional attribute prediction. To handle the complexities of natural speech, including intra- and inter-subject variability, we propose Multi-level Acoustic-Textual Emotion Representation (MATER), a novel hierarchical framework that integrates acoustic and textual features at the word, utterance, and embedding levels. By fusing low-level lexical and acoustic cues with high-level contextualized representations, MATER effectively captures both fine-grained prosodic variations and semantic nuances. Additionally, we introduce an uncertainty-aware ensemble strategy to mitigate annotator inconsistencies, improving robustness in ambiguous emotional expressions. MATER ranks fourth in both tasks with a Macro-F1 of 41.01% and an average CCC of 0.5928, securing second place in valence prediction with an impressive CCC of 0.6941. 

---
# FlightKooba: A Fast Interpretable FTP Model 

**Authors**: Jing Lu, Xuan Wu, Yizhun Tian, Songhan Fan, Yali Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19885)  

**Abstract**: The Koopman theory is a powerful and effective modeling tool for converting nonlinear systems into linear representations, and flight trajectory prediction (FTP) is a complex nonlinear system. However, current models applying the Koopman theory to FTP tasks are not very effective, model interpretability is indeed an issue, and the Koopman operators are computationally intensive, resulting in long training times. To address this issue, this paper proposes a new modeling and control framework based on the HIPPO method, the Koopman theory, and state space equations from cybernetics: FlightKooba. Inspired by the idea of structural state space equations, FlightKooba directly constructs the Koopman operators from data. This makes the framework highly interpretable and significantly reduces the number of trainable parameters in the module, thereby greatly reducing training time. Experiments have demonstrated the superiority of the FlightKooba modeling method in terms of time and memory consumption (training time comparable to the Mamba module without using CUDA-level acceleration; memory reduced by more than 50% on most datasets, with a tenfold reduction in the number of parameters), essentially completing the FTP task. It provides a new method for the fast computation of the Koopman operators, opening up new possibilities for the combination of time series forecasting and control. 

---
# MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection 

**Authors**: Zhengxiang Huang, Chaoyue Niu, Zhaode Wang, Jiarui Xue, Hanming Zhang, Yugang Wang, Zewei Xin, Xiaotang Jiang, Chengfei Lv, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19884)  

**Abstract**: As the demand for on-device Large Language Model (LLM) inference grows, energy efficiency has become a major concern, especially for battery-limited mobile devices. Our analysis shows that the memory-bound LLM decode phase dominates energy use, and yet most existing works focus on accelerating the prefill phase, neglecting energy concerns. We introduce Adaptive Energy-Centric Core Selection (AECS) and integrate it into MNN to create the energy-efficient version, MNN-AECS, the first engine-level system solution without requiring root access or OS modifications for energy-efficient LLM decoding. MNN-AECS is designed to reduce LLM decoding energy while keeping decode speed within an acceptable slowdown threshold by dynamically selecting low-power CPU cores. MNN-AECS is evaluated across 5 Android and 2 iOS devices on 5 popular LLMs of various sizes. Compared to original MNN, MNN-AECS cuts down energy use by 23% without slowdown averaged over all 7 devices and 4 datasets. Against other engines, including this http URL, executorch, mllm, and MediaPipe, MNN-AECS delivers 39% to 78% energy saving and 12% to 363% speedup on average. 

---
# STIMULUS: Achieving Fast Convergence and Low Sample Complexity in Stochastic Multi-Objective Learning 

**Authors**: Zhuqing Liu, Chaosheng Dong, Michinari Momma, Simone Shao, Shaoyuan Xu, Yan Gao, Haibo Yang, Jia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19883)  

**Abstract**: Recently, multi-objective optimization (MOO) has gained attention for its broad applications in ML, operations research, and engineering. However, MOO algorithm design remains in its infancy and many existing MOO methods suffer from unsatisfactory convergence rate and sample complexity performance. To address this challenge, in this paper, we propose an algorithm called STIMULUS( stochastic path-integrated multi-gradient recursive e\ulstimator), a new and robust approach for solving MOO problems. Different from the traditional methods, STIMULUS introduces a simple yet powerful recursive framework for updating stochastic gradient estimates to improve convergence performance with low sample complexity. In addition, we introduce an enhanced version of STIMULUS, termed STIMULUS-M, which incorporates a momentum term to further expedite convergence. We establish $O(1/T)$ convergence rates of the proposed methods for non-convex settings and $O (\exp{-\mu T})$ for strongly convex settings, where $T$ is the total number of iteration rounds. Additionally, we achieve the state-of-the-art $O \left(n+\sqrt{n}\epsilon^{-1}\right)$ sample complexities for non-convex settings and $O\left(n+ \sqrt{n} \ln ({\mu/\epsilon})\right)$ for strongly convex settings, where $\epsilon>0$ is a desired stationarity error. Moreover, to alleviate the periodic full gradient evaluation requirement in STIMULUS and STIMULUS-M, we further propose enhanced versions with adaptive batching called STIMULUS+/ STIMULUS-M+ and provide their theoretical analysis. 

---
# Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track 

**Authors**: Rylan Schaeffer, Joshua Kazdan, Yegor Denisov-Blanch, Brando Miranda, Matthias Gerstgrasser, Susan Zhang, Andreas Haupt, Isha Gupta, Elyas Obbad, Jesse Dodge, Jessica Zosa Forde, Koustuv Sinha, Francesco Orabona, Sanmi Koyejo, David Donoho  

**Link**: [PDF](https://arxiv.org/pdf/2506.19882)  

**Abstract**: Science progresses by iteratively advancing and correcting humanity's understanding of the world. In machine learning (ML) research, rapid advancements have led to an explosion of publications, but have also led to misleading, incorrect, flawed or perhaps even fraudulent studies being accepted and sometimes highlighted at ML conferences due to the fallibility of peer review. While such mistakes are understandable, ML conferences do not offer robust processes to help the field systematically correct when such errors are this http URL position paper argues that ML conferences should establish a dedicated "Refutations and Critiques" (R & C) Track. This R & C Track would provide a high-profile, reputable platform to support vital research that critically challenges prior research, thereby fostering a dynamic self-correcting research ecosystem. We discuss key considerations including track design, review principles, potential pitfalls, and provide an illustrative example submission concerning a recent ICLR 2025 Oral. We conclude that ML conferences should create official, reputable mechanisms to help ML research self-correct. 

---
# Physics-Guided Radiotherapy Treatment Planning with Deep Learning 

**Authors**: Stefanos Achlatis, Efstratios Gavves, Jan-Jakob Sonke  

**Link**: [PDF](https://arxiv.org/pdf/2506.19880)  

**Abstract**: Radiotherapy (RT) is a critical cancer treatment, with volumetric modulated arc therapy (VMAT) being a commonly used technique that enhances dose conformity by dynamically adjusting multileaf collimator (MLC) positions and monitor units (MU) throughout gantry rotation. Adaptive radiotherapy requires frequent modifications to treatment plans to account for anatomical variations, necessitating time-efficient solutions. Deep learning offers a promising solution to automate this process. To this end, we propose a two-stage, physics-guided deep learning pipeline for radiotherapy planning. In the first stage, our network is trained with direct supervision on treatment plan parameters, consisting of MLC and MU values. In the second stage, we incorporate an additional supervision signal derived from the predicted 3D dose distribution, integrating physics-based guidance into the training process. We train and evaluate our approach on 133 prostate cancer patients treated with a uniform 2-arc VMAT protocol delivering a dose of 62 Gy to the planning target volume (PTV). Our results demonstrate that the proposed approach, implemented using both 3D U-Net and UNETR architectures, consistently produces treatment plans that closely match clinical ground truths. Our method achieves a mean difference of D95% = 0.42 +/- 1.83 Gy and V95% = -0.22 +/- 1.87% at the PTV while generating dose distributions that reduce radiation exposure to organs at risk. These findings highlight the potential of physics-guided deep learning in RT planning. 

---
# Robust Anomaly Detection in Network Traffic: Evaluating Machine Learning Models on CICIDS2017 

**Authors**: Zhaoyang Xu, Yunbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19877)  

**Abstract**: Identifying suitable machine learning paradigms for intrusion detection remains critical for building effective and generalizable security solutions. In this study, we present a controlled comparison of four representative models - Multi-Layer Perceptron (MLP), 1D Convolutional Neural Network (CNN), One-Class Support Vector Machine (OCSVM) and Local Outlier Factor (LOF) - on the CICIDS2017 dataset under two scenarios: detecting known attack types and generalizing to previously unseen threats. Our results show that supervised MLP and CNN achieve near-perfect accuracy on familiar attacks but suffer drastic recall drops on novel attacks. Unsupervised LOF attains moderate overall accuracy and high recall on unknown threats at the cost of elevated false alarms, while boundary-based OCSVM balances precision and recall best, demonstrating robust detection across both scenarios. These findings offer practical guidance for selecting IDS models in dynamic network environments. 

---
# Speaker Embeddings to Improve Tracking of Intermittent and Moving Speakers 

**Authors**: Taous Iatariene, Can Cui, Alexandre Guérin, Romain Serizel  

**Link**: [PDF](https://arxiv.org/pdf/2506.19875)  

**Abstract**: Speaker tracking methods often rely on spatial observations to assign coherent track identities over time. This raises limits in scenarios with intermittent and moving speakers, i.e., speakers that may change position when they are inactive, thus leading to discontinuous spatial trajectories. This paper proposes to investigate the use of speaker embeddings, in a simple solution to this issue. We propose to perform identity reassignment post-tracking, using speaker embeddings. We leverage trajectory-related information provided by an initial tracking step and multichannel audio signal. Beamforming is used to enhance the signal towards the speakers' positions in order to compute speaker embeddings. These are then used to assign new track identities based on an enrollment pool. We evaluate the performance of the proposed speaker embedding-based identity reassignment method on a dataset where speakers change position during inactivity periods. Results show that it consistently improves the identity assignment performance of neural and standard tracking systems. In particular, we study the impact of beamforming and input duration for embedding extraction. 

---
# Towards Provable (In)Secure Model Weight Release Schemes 

**Authors**: Xing Yang, Bingtao Wang, Yuhao Wang, Zimo Ji, Terry Jingchen Zhang, Wenyuan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19874)  

**Abstract**: Recent secure weight release schemes claim to enable open-source model distribution while protecting model ownership and preventing misuse. However, these approaches lack rigorous security foundations and provide only informal security guarantees. Inspired by established works in cryptography, we formalize the security of weight release schemes by introducing several concrete security definitions. We then demonstrate our definition's utility through a case study of TaylorMLP, a prominent secure weight release scheme. Our analysis reveals vulnerabilities that allow parameter extraction thus showing that TaylorMLP fails to achieve its informal security goals. We hope this work will advocate for rigorous research at the intersection of machine learning and security communities and provide a blueprint for how future weight release schemes should be designed and evaluated. 

---
# An Attack Method for Medical Insurance Claim Fraud Detection based on Generative Adversarial Network 

**Authors**: Yining Pang, Chenghan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19871)  

**Abstract**: Insurance fraud detection represents a pivotal advancement in modern insurance service, providing intelligent and digitalized monitoring to enhance management and prevent fraud. It is crucial for ensuring the security and efficiency of insurance systems. Although AI and machine learning algorithms have demonstrated strong performance in detecting fraudulent claims, the absence of standardized defense mechanisms renders current systems vulnerable to emerging adversarial threats. In this paper, we propose a GAN-based approach to conduct adversarial attacks on fraud detection systems. Our results indicate that an attacker, without knowledge of the training data or internal model details, can generate fraudulent cases that are classified as legitimate with a 99\% attack success rate (ASR). By subtly modifying real insurance records and claims, adversaries can significantly increase the fraud risk, potentially bypassing compromised detection systems. These findings underscore the urgent need to enhance the robustness of insurance fraud detection models against adversarial manipulation, thereby ensuring the stability and reliability of different insurance systems. 

---
# Secure Energy Transactions Using Blockchain Leveraging AI for Fraud Detection and Energy Market Stability 

**Authors**: Md Asif Ul Hoq Khan, MD Zahedul Islam, Istiaq Ahmed, Md Masud Karim Rabbi, Farhana Rahman Anonna, MD Abdul Fahim Zeeshan, Mehedi Hasan Ridoy, Bivash Ranjan Chowdhury, Md Nazmul Shakir Rabbi, GM Alamin Sadnan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19870)  

**Abstract**: Peer-to-peer trading and the move to decentralized grids have reshaped the energy markets in the United States. Notwithstanding, such developments lead to new challenges, mainly regarding the safety and authenticity of energy trade. This study aimed to develop and build a secure, intelligent, and efficient energy transaction system for the decentralized US energy market. This research interlinks the technological prowess of blockchain and artificial intelligence (AI) in a novel way to solve long-standing challenges in the distributed energy market, specifically those of security, fraudulent behavior detection, and market reliability. The dataset for this research is comprised of more than 1.2 million anonymized energy transaction records from a simulated peer-to-peer (P2P) energy exchange network emulating real-life blockchain-based American microgrids, including those tested by LO3 Energy and Grid+ Labs. Each record contains detailed fields of transaction identifier, timestamp, energy volume (kWh), transaction type (buy/sell), unit price, prosumer/consumer identifier (hashed for privacy), smart meter readings, geolocation regions, and settlement confirmation status. The dataset also includes system-calculated behavior metrics of transaction rate, variability of energy production, and historical pricing patterns. The system architecture proposed involves the integration of two layers, namely a blockchain layer and artificial intelligence (AI) layer, each playing a unique but complementary function in energy transaction securing and market intelligence improvement. The machine learning models used in this research were specifically chosen for their established high performance in classification tasks, specifically in the identification of energy transaction fraud in decentralized markets. 

---
# Scalable and Cost-Efficient de Novo Template-Based Molecular Generation 

**Authors**: Piotr Gaiński, Oussama Boussif, Andrei Rekesh, Dmytro Shevchuk, Ali Parviz, Mike Tyers, Robert A. Batey, Michał Koziarski  

**Link**: [PDF](https://arxiv.org/pdf/2506.19865)  

**Abstract**: Template-based molecular generation offers a promising avenue for drug design by ensuring generated compounds are synthetically accessible through predefined reaction templates and building blocks. In this work, we tackle three core challenges in template-based GFlowNets: (1) minimizing synthesis cost, (2) scaling to large building block libraries, and (3) effectively utilizing small fragment sets. We propose \textbf{Recursive Cost Guidance}, a backward policy framework that employs auxiliary machine learning models to approximate synthesis cost and viability. This guidance steers generation toward low-cost synthesis pathways, significantly enhancing cost-efficiency, molecular diversity, and quality, especially when paired with an \textbf{Exploitation Penalty} that balances the trade-off between exploration and exploitation. To enhance performance in smaller building block libraries, we develop a \textbf{Dynamic Library} mechanism that reuses intermediate high-reward states to construct full synthesis trees. Our approach establishes state-of-the-art results in template-based molecular generation. 

---
# Exploring the Capabilities of the Frontier Large Language Models for Nuclear Energy Research 

**Authors**: Ahmed Almeldein, Mohammed Alnaggar, Rick Archibald, Tom Beck, Arpan Biswas, Rike Bostelmann, Wes Brewer, Chris Bryan, Christopher Calle, Cihangir Celik, Rajni Chahal, Jong Youl Choi, Arindam Chowdhury, Mark Cianciosa, Franklin Curtis, Gregory Davidson, Sebastian De Pascuale, Lisa Fassino, Ana Gainaru, Yashika Ghai, Luke Gibson, Qian Gong, Christopher Greulich, Scott Greenwood, Cory Hauck, Ehab Hassan, Rinkle Juneja, Soyoung Kang, Scott Klasky, Atul Kumar, Vineet Kumar, Paul Laiu, Calvin Lear, Yan-Ru Lin, Jono McConnell, Furkan Oz, Anant Raj, Pradeep Ramuhalli, Marie Romedenne, Samantha Sabatino, José Salcedo-Pérez, Nathan D. See, Arpan Sircar, Punam Thankur, Tim Younkin, Xiao-Ying Yu, Prashant Jain, Tom Evans, Prasanna Balaprakash  

**Link**: [PDF](https://arxiv.org/pdf/2506.19863)  

**Abstract**: The AI for Nuclear Energy workshop at Oak Ridge National Laboratory evaluated the potential of Large Language Models (LLMs) to accelerate fusion and fission research. Fourteen interdisciplinary teams explored diverse nuclear science challenges using ChatGPT, Gemini, Claude, and other AI models over a single day. Applications ranged from developing foundation models for fusion reactor control to automating Monte Carlo simulations, predicting material degradation, and designing experimental programs for advanced reactors. Teams employed structured workflows combining prompt engineering, deep research capabilities, and iterative refinement to generate hypotheses, prototype code, and research strategies. Key findings demonstrate that LLMs excel at early-stage exploration, literature synthesis, and workflow design, successfully identifying research gaps and generating plausible experimental frameworks. However, significant limitations emerged, including difficulties with novel materials designs, advanced code generation for modeling and simulation, and domain-specific details requiring expert validation. The successful outcomes resulted from expert-driven prompt engineering and treating AI as a complementary tool rather than a replacement for physics-based methods. The workshop validated AI's potential to accelerate nuclear energy research through rapid iteration and cross-disciplinary synthesis while highlighting the need for curated nuclear-specific datasets, workflow automation, and specialized model development. These results provide a roadmap for integrating AI tools into nuclear science workflows, potentially reducing development cycles for safer, more efficient nuclear energy systems while maintaining rigorous scientific standards. 

---
# DualEquiNet: A Dual-Space Hierarchical Equivariant Network for Large Biomolecules 

**Authors**: Junjie Xu, Jiahao Zhang, Mangal Prakash, Xiang Zhang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19862)  

**Abstract**: Geometric graph neural networks (GNNs) that respect E(3) symmetries have achieved strong performance on small molecule modeling, but they face scalability and expressiveness challenges when applied to large biomolecules such as RNA and proteins. These systems require models that can simultaneously capture fine-grained atomic interactions, long-range dependencies across spatially distant components, and biologically relevant hierarchical structure, such as atoms forming residues, which in turn form higher-order domains. Existing geometric GNNs, which typically operate exclusively in either Euclidean or Spherical Harmonics space, are limited in their ability to capture both the fine-scale atomic details and the long-range, symmetry-aware dependencies required for modeling the multi-scale structure of large biomolecules. We introduce DualEquiNet, a Dual-Space Hierarchical Equivariant Network that constructs complementary representations in both Euclidean and Spherical Harmonics spaces to capture local geometry and global symmetry-aware features. DualEquiNet employs bidirectional cross-space message passing and a novel Cross-Space Interaction Pooling mechanism to hierarchically aggregate atomic features into biologically meaningful units, such as residues, enabling efficient and expressive multi-scale modeling for large biomolecular systems. DualEquiNet achieves state-of-the-art performance on multiple existing benchmarks for RNA property prediction and protein modeling, and outperforms prior methods on two newly introduced 3D structural benchmarks demonstrating its broad effectiveness across a range of large biomolecule modeling tasks. 

---
