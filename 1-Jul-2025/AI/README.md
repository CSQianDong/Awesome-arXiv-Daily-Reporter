# SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning 

**Authors**: Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, Wee Sun Lee, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2506.24119)  

**Abstract**: Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, eliminating the need for human supervision. Through self-play, SPIRAL generates an infinite curriculum of progressively challenging problems as models must constantly adapt to stronger opponents. To enable this self-play training at scale, We implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. Using SPIRAL, self-play on zero-sum games produces reasoning capabilities that transfer broadly. Training Qwen3-4B-Base on Kuhn Poker alone achieves 8.6% improvement on math and 8.4% on general reasoning, outperforming SFT on 25,000 expert game trajectories. Analysis reveals that this transfer occurs through three cognitive patterns: systematic decomposition, expected value calculation, and case-by-case analysis. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) further enhances performance as each game develops distinct reasoning strengths. Applying SPIRAL to a strong reasoning model (DeepSeek-R1-Distill-Qwen-7B) can still lead to 2.0% average improvement. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities, highlighting a promising direction for autonomous reasoning development. 

---
# Constructing Non-Markovian Decision Process via History Aggregator 

**Authors**: Yongyi Wang, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.24026)  

**Abstract**: In the domain of algorithmic decision-making, non-Markovian dynamics manifest as a significant impediment, especially for paradigms such as Reinforcement Learning (RL), thereby exerting far-reaching consequences on the advancement and effectiveness of the associated systems. Nevertheless, the existing benchmarks are deficient in comprehensively assessing the capacity of decision algorithms to handle non-Markovian dynamics. To address this deficiency, we have devised a generalized methodology grounded in category theory. Notably, we established the category of Markov Decision Processes (MDP) and the category of non-Markovian Decision Processes (NMDP), and proved the equivalence relationship between them. This theoretical foundation provides a novel perspective for understanding and addressing non-Markovian dynamics. We further introduced non-Markovianity into decision-making problem settings via the History Aggregator for State (HAS). With HAS, we can precisely control the state dependency structure of decision-making problems in the time series. Our analysis demonstrates the effectiveness of our method in representing a broad range of non-Markovian dynamics. This approach facilitates a more rigorous and flexible evaluation of decision algorithms by testing them in problem settings where non-Markovian dynamics are explicitly constructed. 

---
# Harnessing AI Agents to Advance Research on Refugee Child Mental Health 

**Authors**: Aditya Shrivastava, Komal Gupta, Shraddha Arora  

**Link**: [PDF](https://arxiv.org/pdf/2506.23992)  

**Abstract**: The international refugee crisis deepens, exposing millions of dis placed children to extreme psychological trauma. This research suggests a com pact, AI-based framework for processing unstructured refugee health data and distilling knowledge on child mental health. We compare two Retrieval-Aug mented Generation (RAG) pipelines, Zephyr-7B-beta and DeepSeek R1-7B, to determine how well they process challenging humanitarian datasets while avoid ing hallucination hazards. By combining cutting-edge AI methods with migration research and child psychology, this study presents a scalable strategy to assist policymakers, mental health practitioners, and humanitarian agencies to better assist displaced children and recognize their mental wellbeing. In total, both the models worked properly but significantly Deepseek R1 is superior to Zephyr with an accuracy of answer relevance 0.91 

---
# AI Risk-Management Standards Profile for General-Purpose AI (GPAI) and Foundation Models 

**Authors**: Anthony M. Barrett, Jessica Newman, Brandie Nonnecke, Nada Madkour, Dan Hendrycks, Evan R. Murphy, Krystal Jackson, Deepika Raman  

**Link**: [PDF](https://arxiv.org/pdf/2506.23949)  

**Abstract**: Increasingly multi-purpose AI models, such as cutting-edge large language models or other 'general-purpose AI' (GPAI) models, 'foundation models,' generative AI models, and 'frontier models' (typically all referred to hereafter with the umbrella term 'GPAI/foundation models' except where greater specificity is needed), can provide many beneficial capabilities but also risks of adverse events with profound consequences. This document provides risk-management practices or controls for identifying, analyzing, and mitigating risks of GPAI/foundation models. We intend this document primarily for developers of large-scale, state-of-the-art GPAI/foundation models; others that can benefit from this guidance include downstream developers of end-use applications that build on a GPAI/foundation model. This document facilitates conformity with or use of leading AI risk management-related standards, adapting and building on the generic voluntary guidance in the NIST AI Risk Management Framework and ISO/IEC 23894, with a focus on the unique issues faced by developers of GPAI/foundation models. 

---
# Industrial brain: a human-like autonomous neuro-symbolic cognitive decision-making system 

**Authors**: Junping Wang, Bicheng Wang, Yibo Xuea, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.23926)  

**Abstract**: Resilience non-equilibrium measurement, the ability to maintain fundamental functionality amidst failures and errors, is crucial for scientific management and engineering applications of industrial chain. The problem is particularly challenging when the number or types of multiple co-evolution of resilience (for example, randomly placed) are extremely chaos. Existing end-to-end deep learning ordinarily do not generalize well to unseen full-feld reconstruction of spatiotemporal co-evolution structure, and predict resilience of network topology, especially in multiple chaos data regimes typically seen in real-world applications. To address this challenge, here we propose industrial brain, a human-like autonomous cognitive decision-making and planning framework integrating higher-order activity-driven neuro network and CT-OODA symbolic reasoning to autonomous plan resilience directly from observational data of global variable. The industrial brain not only understands and model structure of node activity dynamics and network co-evolution topology without simplifying assumptions, and reveal the underlying laws hidden behind complex networks, but also enabling accurate resilience prediction, inference, and planning. Experimental results show that industrial brain significantly outperforms resilience prediction and planning methods, with an accurate improvement of up to 10.8\% over GoT and OlaGPT framework and 11.03\% over spectral dimension reduction. It also generalizes to unseen topologies and dynamics and maintains robust performance despite observational disturbances. Our findings suggest that industrial brain addresses an important gap in resilience prediction and planning for industrial chain. 

---
# Performance of LLMs on Stochastic Modeling Operations Research Problems: From Theory to Practice 

**Authors**: Akshit Kumar, Tianyi Peng, Yuhang Wu, Assaf Zeevi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23924)  

**Abstract**: Large language models (LLMs) have exhibited expert-level capabilities across various domains. However, their abilities to solve problems in Operations Research (OR) -- the analysis and optimization of mathematical models derived from real-world problems or their verbal descriptions -- remain underexplored. In this work, we take a first step toward evaluating LLMs' abilities to solve stochastic modeling problems, a core class of OR problems characterized by uncertainty and typically involving tools from probability, statistics, and stochastic processes. We manually procure a representative set of graduate-level homework and doctoral qualification-exam problems and test LLMs' abilities to solve them. We further leverage SimOpt, an open-source library of simulation-optimization problems and solvers, to investigate LLMs' abilities to make real-world decisions under uncertainty. Our results show that, though a nontrivial amount of work is still needed to reliably automate the stochastic modeling pipeline in reality, state-of-the-art LLMs demonstrate proficiency on par with human experts in both classroom and practical settings. These findings highlight the potential of building AI agents that assist OR researchers and amplify the real-world impact of OR through automation. 

---
# Beyond Statistical Learning: Exact Learning Is Essential for General Intelligence 

**Authors**: András György, Tor Lattimore, Nevena Lazić, Csaba Szepesvári  

**Link**: [PDF](https://arxiv.org/pdf/2506.23908)  

**Abstract**: Sound deductive reasoning -- the ability to derive new knowledge from existing facts and rules -- is an indisputably desirable aspect of general intelligence. Despite the major advances of AI systems in areas such as math and science, especially since the introduction of transformer architectures, it is well-documented that even the most advanced frontier systems regularly and consistently falter on easily-solvable deductive reasoning tasks. Hence, these systems are unfit to fulfill the dream of achieving artificial general intelligence capable of sound deductive reasoning. We argue that their unsound behavior is a consequence of the statistical learning approach powering their development. To overcome this, we contend that to achieve reliable deductive reasoning in learning-based AI systems, researchers must fundamentally shift from optimizing for statistical performance against distributions on reasoning problems and algorithmic tasks to embracing the more ambitious exact learning paradigm, which demands correctness on all inputs. We argue that exact learning is both essential and possible, and that this ambitious objective should guide algorithm design. 

---
# A Survey on Autonomy-Induced Security Risks in Large Model-Based Agents 

**Authors**: Hang Su, Jun Luo, Chang Liu, Xiao Yang, Yichi Zhang, Yinpeng Dong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23844)  

**Abstract**: Recent advances in large language models (LLMs) have catalyzed the rise of autonomous AI agents capable of perceiving, reasoning, and acting in dynamic, open-ended environments. These large-model agents mark a paradigm shift from static inference systems to interactive, memory-augmented entities. While these capabilities significantly expand the functional scope of AI, they also introduce qualitatively novel security risks - such as memory poisoning, tool misuse, reward hacking, and emergent misalignment - that extend beyond the threat models of conventional systems or standalone LLMs. In this survey, we first examine the structural foundations and key capabilities that underpin increasing levels of agent autonomy, including long-term memory retention, modular tool use, recursive planning, and reflective reasoning. We then analyze the corresponding security vulnerabilities across the agent stack, identifying failure modes such as deferred decision hazards, irreversible tool chains, and deceptive behaviors arising from internal state drift or value misalignment. These risks are traced to architectural fragilities that emerge across perception, cognition, memory, and action modules. To address these challenges, we systematically review recent defense strategies deployed at different autonomy layers, including input sanitization, memory lifecycle control, constrained decision-making, structured tool invocation, and introspective reflection. We introduce the Reflective Risk-Aware Agent Architecture (R2A2), a unified cognitive framework grounded in Constrained Markov Decision Processes (CMDPs), which incorporates risk-aware world modeling, meta-policy adaptation, and joint reward-risk optimization to enable principled, proactive safety across the agent's decision-making loop. 

---
# Advancing Learnable Multi-Agent Pathfinding Solvers with Active Fine-Tuning 

**Authors**: Anton Andreychuk, Konstantin Yakovlev, Aleksandr Panov, Alexey Skrynnik  

**Link**: [PDF](https://arxiv.org/pdf/2506.23793)  

**Abstract**: Multi-agent pathfinding (MAPF) is a common abstraction of multi-robot trajectory planning problems, where multiple homogeneous robots simultaneously move in the shared environment. While solving MAPF optimally has been proven to be NP-hard, scalable, and efficient, solvers are vital for real-world applications like logistics, search-and-rescue, etc. To this end, decentralized suboptimal MAPF solvers that leverage machine learning have come on stage. Building on the success of the recently introduced MAPF-GPT, a pure imitation learning solver, we introduce MAPF-GPT-DDG. This novel approach effectively fine-tunes the pre-trained MAPF model using centralized expert data. Leveraging a novel delta-data generation mechanism, MAPF-GPT-DDG accelerates training while significantly improving performance at test time. Our experiments demonstrate that MAPF-GPT-DDG surpasses all existing learning-based MAPF solvers, including the original MAPF-GPT, regarding solution quality across many testing scenarios. Remarkably, it can work with MAPF instances involving up to 1 million agents in a single environment, setting a new milestone for scalability in MAPF domains. 

---
# When GNNs Met a Word Equations Solver: Learning to Rank Equations (Extended Technical Report) 

**Authors**: Parosh Aziz Abdulla, Mohamed Faouzi Atig, Julie Cailler, Chencheng Liang, Philipp Rümmer  

**Link**: [PDF](https://arxiv.org/pdf/2506.23784)  

**Abstract**: Nielsen transformation is a standard approach for solving word equations: by repeatedly splitting equations and applying simplification steps, equations are rewritten until a solution is reached. When solving a conjunction of word equations in this way, the performance of the solver will depend considerably on the order in which equations are processed. In this work, the use of Graph Neural Networks (GNNs) for ranking word equations before and during the solving process is explored. For this, a novel graph-based representation for word equations is presented, preserving global information across conjuncts, enabling the GNN to have a holistic view during ranking. To handle the variable number of conjuncts, three approaches to adapt a multi-classification task to the problem of ranking equations are proposed. The training of the GNN is done with the help of minimum unsatisfiable subsets (MUSes) of word equations. The experimental results show that, compared to state-of-the-art string solvers, the new framework solves more problems in benchmarks where each variable appears at most once in each equation. 

---
# BayesL: Towards a Logical Framework for Bayesian Networks 

**Authors**: Stefano M. Nicoletti, Mariëlle Stoelinga  

**Link**: [PDF](https://arxiv.org/pdf/2506.23773)  

**Abstract**: We introduce BayesL, a novel logical framework for specifying, querying, and verifying the behaviour of Bayesian networks (BNs). BayesL (pronounced "Basil") is a structured language that allows for the creation of queries over BNs. It facilitates versatile reasoning concerning causal and evidence-based relationships, and permits comprehensive what-if scenario evaluations without the need for manual modifications to the model. 

---
# Attestable Audits: Verifiable AI Safety Benchmarks Using Trusted Execution Environments 

**Authors**: Christoph Schnabl, Daniel Hugenroth, Bill Marino, Alastair R. Beresford  

**Link**: [PDF](https://arxiv.org/pdf/2506.23706)  

**Abstract**: Benchmarks are important measures to evaluate safety and compliance of AI models at scale. However, they typically do not offer verifiable results and lack confidentiality for model IP and benchmark datasets. We propose Attestable Audits, which run inside Trusted Execution Environments and enable users to verify interaction with a compliant AI model. Our work protects sensitive data even when model provider and auditor do not trust each other. This addresses verification challenges raised in recent AI governance frameworks. We build a prototype demonstrating feasibility on typical audit benchmarks against Llama-3.1. 

---
# A New Perspective On AI Safety Through Control Theory Methodologies 

**Authors**: Lars Ullrich, Walter Zimmer, Ross Greer, Knut Graichen, Alois C. Knoll, Mohan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23703)  

**Abstract**: While artificial intelligence (AI) is advancing rapidly and mastering increasingly complex problems with astonishing performance, the safety assurance of such systems is a major concern. Particularly in the context of safety-critical, real-world cyber-physical systems, AI promises to achieve a new level of autonomy but is hampered by a lack of safety assurance. While data-driven control takes up recent developments in AI to improve control systems, control theory in general could be leveraged to improve AI safety. Therefore, this article outlines a new perspective on AI safety based on an interdisciplinary interpretation of the underlying data-generation process and the respective abstraction by AI systems in a system theory-inspired and system analysis-driven manner. In this context, the new perspective, also referred to as data control, aims to stimulate AI engineering to take advantage of existing safety analysis and assurance in an interdisciplinary way to drive the paradigm of data control. Following a top-down approach, a generic foundation for safety analysis and assurance is outlined at an abstract level that can be refined for specific AI systems and applications and is prepared for future innovation. 

---
# Agent4S: The Transformation of Research Paradigms from the Perspective of Large Language Models 

**Authors**: Boyuan Zheng, Zerui Fang, Zhe Xu, Rui Wang, Yiwen Chen, Cunshi Wang, Mengwei Qu, Lei Lei, Zhen Feng, Yan Liu, Yuyang Li, Mingzhou Tan, Jiaji Wu, Jianwei Shuai, Jia Li, Fangfu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.23692)  

**Abstract**: While AI for Science (AI4S) serves as an analytical tool in the current research paradigm, it doesn't solve its core inefficiency. We propose "Agent for Science" (Agent4S)-the use of LLM-driven agents to automate the entire research workflow-as the true Fifth Scientific Paradigm. This paper introduces a five-level classification for Agent4S, outlining a clear roadmap from simple task automation to fully autonomous, collaborative "AI Scientists." This framework defines the next revolutionary step in scientific discovery. 

---
# PokéAI: A Goal-Generating, Battle-Optimizing Multi-agent System for Pokemon Red 

**Authors**: Zihao Liu, Xinhang Sui, Yueran Song, Siwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23689)  

**Abstract**: We introduce PokéAI, the first text-based, multi-agent large language model (LLM) framework designed to autonomously play and progress through Pokémon Red. Our system consists of three specialized agents-Planning, Execution, and Critique-each with its own memory bank, role, and skill set. The Planning Agent functions as the central brain, generating tasks to progress through the game. These tasks are then delegated to the Execution Agent, which carries them out within the game environment. Upon task completion, the Critique Agent evaluates the outcome to determine whether the objective was successfully achieved. Once verification is complete, control returns to the Planning Agent, forming a closed-loop decision-making system.
As a preliminary step, we developed a battle module within the Execution Agent. Our results show that the battle AI achieves an average win rate of 80.8% across 50 wild encounters, only 6% lower than the performance of an experienced human player. Furthermore, we find that a model's battle performance correlates strongly with its LLM Arena score on language-related tasks, indicating a meaningful link between linguistic ability and strategic reasoning. Finally, our analysis of gameplay logs reveals that each LLM exhibits a unique playstyle, suggesting that individual models develop distinct strategic behaviors. 

---
# HASD: Hierarchical Adaption for pathology Slide-level Domain-shift 

**Authors**: Jingsong Liu, Han Li, Chen Yang, Michael Deutges, Ario Sadafi, Xin You, Katharina Breininger, Nassir Navab, Peter J. Schüffler  

**Link**: [PDF](https://arxiv.org/pdf/2506.23673)  

**Abstract**: Domain shift is a critical problem for pathology AI as pathology data is heavily influenced by center-specific conditions. Current pathology domain adaptation methods focus on image patches rather than WSI, thus failing to capture global WSI features required in typical clinical scenarios. In this work, we address the challenges of slide-level domain shift by proposing a Hierarchical Adaptation framework for Slide-level Domain-shift (HASD). HASD achieves multi-scale feature consistency and computationally efficient slide-level domain adaptation through two key components: (1) a hierarchical adaptation framework that integrates a Domain-level Alignment Solver for feature alignment, a Slide-level Geometric Invariance Regularization to preserve the morphological structure, and a Patch-level Attention Consistency Regularization to maintain local critical diagnostic cues; and (2) a prototype selection mechanism that reduces computational overhead. We validate our method on two slide-level tasks across five datasets, achieving a 4.1\% AUROC improvement in a Breast Cancer HER2 Grading cohort and a 3.9\% C-index gain in a UCEC survival prediction cohort. Our method provides a practical and reliable slide-level domain adaption solution for pathology institutions, minimizing both computational and annotation costs. 

---
# Self-correcting Reward Shaping via Language Models for Reinforcement Learning Agents in Games 

**Authors**: António Afonso, Iolanda Leite, Alessandro Sestini, Florian Fuchs, Konrad Tollmar, Linus Gisslén  

**Link**: [PDF](https://arxiv.org/pdf/2506.23626)  

**Abstract**: Reinforcement Learning (RL) in games has gained significant momentum in recent years, enabling the creation of different agent behaviors that can transform a player's gaming experience. However, deploying RL agents in production environments presents two key challenges: (1) designing an effective reward function typically requires an RL expert, and (2) when a game's content or mechanics are modified, previously tuned reward weights may no longer be optimal. Towards the latter challenge, we propose an automated approach for iteratively fine-tuning an RL agent's reward function weights, based on a user-defined language based behavioral goal. A Language Model (LM) proposes updated weights at each iteration based on this target behavior and a summary of performance statistics from prior training rounds. This closed-loop process allows the LM to self-correct and refine its output over time, producing increasingly aligned behavior without the need for manual reward engineering. We evaluate our approach in a racing task and show that it consistently improves agent performance across iterations. The LM-guided agents show a significant increase in performance from $9\%$ to $74\%$ success rate in just one iteration. We compare our LM-guided tuning against a human expert's manual weight design in the racing task: by the final iteration, the LM-tuned agent achieved an $80\%$ success rate, and completed laps in an average of $855$ time steps, a competitive performance against the expert-tuned agent's peak $94\%$ success, and $850$ time steps. 

---
# Evaluating Multi-Agent Defences Against Jailbreaking Attacks on Large Language Models 

**Authors**: Maria Carolina Cornelia Wit, Jun Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23576)  

**Abstract**: Recent advances in large language models (LLMs) have raised concerns about jailbreaking attacks, i.e., prompts that bypass safety mechanisms. This paper investigates the use of multi-agent LLM systems as a defence against such attacks. We evaluate three jailbreaking strategies, including the original AutoDefense attack and two from Deepleaps: BetterDan and JB. Reproducing the AutoDefense framework, we compare single-agent setups with two- and three-agent configurations. Our results show that multi-agent systems enhance resistance to jailbreaks, especially by reducing false negatives. However, its effectiveness varies by attack type, and it introduces trade-offs such as increased false positives and computational overhead. These findings point to the limitations of current automated defences and suggest directions for improving alignment robustness in future LLM systems. 

---
# MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI 

**Authors**: Huanjin Yao, Jiaxing Huang, Yawen Qiu, Michael K. Chen, Wenzheng Liu, Wei Zhang, Wenjie Zeng, Xikun Zhang, Jingyi Zhang, Yuxin Song, Wenhao Wu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23563)  

**Abstract**: Reasoning plays a crucial role in advancing Multimodal Large Language Models (MLLMs) toward Artificial General Intelligence. However, existing MLLM benchmarks often fall short in precisely and comprehensively evaluating long-chain reasoning abilities from three key aspects: (1) lack of difficulty and diversity, (2) susceptibility to guessability and memorization, (3) inadequate assessment of intermediate reasoning steps. To fill this gap, we introduce MMReason, a new benchmark designed to precisely and comprehensively evaluate MLLM long-chain reasoning capability with diverse, open-ended, challenging questions. First, we curate challenging questions requiring multi-step reasoning from various fields (i.e., 6 disciplines) and multiple difficulty levels (i.e., from pre-university to university, and from foundational to competition tiers). Second, these questions are reformulated into an open-ended format and filtered using a multi-model voting technique to eliminate shortcut cases related to guessing and memorization, ensuring robust reasoning evaluations. Third, we annotate the questions with detailed step-by-step solutions, and design a reference-based ternary scoring mechanism to reliably assess intermediate reasoning steps. With MMReason, we benchmark popular leading MLLMs and provide an in-depth analysis of their reasoning capabilities. We hope MMReason will serve as a valuable resource for advancing MLLM reasoning research. Code will be available at this https URL. 

---
# CooT: Learning to Coordinate In-Context with Coordination Transformers 

**Authors**: Huai-Chih Wang, Hsiang-Chun Chuang, Hsi-Chun Cheng, Dai-Jie Wu, Shao-Hua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.23549)  

**Abstract**: Effective coordination among artificial agents in dynamic and uncertain environments remains a significant challenge in multi-agent systems. Existing approaches, such as self-play and population-based methods, either generalize poorly to unseen partners or require extensive training. To overcome these limitations, we propose Coordination Transformers (CooT), a novel in-context coordination framework that uses recent interaction histories to adapt to unseen partners rapidly. Unlike previous approaches that primarily aim to increase the diversity of training partners, CooT explicitly focuses on adapting to new partner behaviors by predicting actions aligned with observed partner interactions. Trained on interaction trajectories collected from diverse pairs of agents with complementary behaviors, CooT quickly learns effective coordination strategies without explicit supervision or fine-tuning. Evaluations on the Overcooked benchmark demonstrate that CooT significantly outperforms baseline methods in coordination tasks involving previously unseen partners. Human evaluations further confirm CooT as the most effective collaborative partner, while extensive ablations highlight its robustness, flexibility, and sensitivity to context in multi-agent scenarios. 

---
# ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data 

**Authors**: Yu Zhang, Ruijie Yu, Jidong Tian, Feng Zhu, Jiapeng Liu, Xiaokang Yang, Yaohui Jin, Yanyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23520)  

**Abstract**: With the increasing interest in robotic synthesis in the context of organic chemistry, the automated extraction of chemical procedures from literature is critical. However, this task remains challenging due to the inherent ambiguity of chemical language and the high cost of human annotation required for developing reliable computer-aided extraction protocols. Here, we present ChemActor, a fully fine-tuned large language model (LLM), as a chemical executor to convert between unstructured experimental procedures and structured action sequences. We propose a sequential LLM-generated data framework to address the challenges of insufficient and low-quality annotated data. This framework integrates a data selection module that selects data based on distribution divergence, with a general-purpose LLM, to generate machine-executable actions from a single molecule input. Additionally, we introduce a novel multi-round LLMs circle review metric, which reflects the model's advanced understanding of chemical experimental procedures. Extensive experiments on reaction-to-description (R2D) and description-to-action (D2A) tasks demonstrate that ChemActor, augmented by LLM-generated data, achieves state-of-the-art performance, outperforming the baseline model by 10%. The code is available at: this https URL. 

---
# Assessing GPTZero's Accuracy in Identifying AI vs. Human-Written Essays 

**Authors**: Selin Dik, Osman Erdem, Mehmet Dik  

**Link**: [PDF](https://arxiv.org/pdf/2506.23517)  

**Abstract**: As the use of AI tools by students has become more prevalent, instructors have started using AI detection tools like GPTZero and QuillBot to detect AI written text. However, the reliability of these detectors remains uncertain. In our study, we focused mostly on the success rate of GPTZero, the most-used AI detector, in identifying AI-generated texts based on different lengths of randomly submitted essays: short (40-100 word count), medium (100-350 word count), and long (350-800 word count). We gathered a data set consisting of twenty-eight AI-generated papers and fifty human-written papers. With this randomized essay data, papers were individually plugged into GPTZero and measured for percentage of AI generation and confidence. A vast majority of the AI-generated papers were detected accurately (ranging from 91-100% AI believed generation), while the human generated essays fluctuated; there were a handful of false positives. These findings suggest that although GPTZero is effective at detecting purely AI-generated content, its reliability in distinguishing human-authored texts is limited. Educators should therefore exercise caution when relying solely on AI detection tools. 

---
# Hybrid Approach for Electricity Price Forecasting using AlexNet and LSTM 

**Authors**: Bosubabu Sambana, Kotamsetty Geethika Devi, Bandi Rajeswara Reddy, Galeti Mohammad Hussain, Gownivalla Siddartha  

**Link**: [PDF](https://arxiv.org/pdf/2506.23504)  

**Abstract**: The recent development of advanced machine learning methods for hybrid models has greatly addressed the need for the correct prediction of electrical prices. This method combines AlexNet and LSTM algorithms, which are used to introduce a new model with higher accuracy in price forecasting. Despite RNN and ANN being effective, they often fail to deal with forex time sequence data. The traditional methods do not accurately forecast the prices. These traditional methods only focus on demand and price which leads to insufficient analysis of data. To address this issue, using the hybrid approach, which focuses on external variables that also effect the predicted prices. Nevertheless, due to AlexNet's excellent feature extraction and LSTM's learning sequential patterns, the prediction accuracy is vastly increased. The model is built on the past data, which has been supplied with the most significant elements like demand, temperature, sunlight, and rain. For example, the model applies methods, such as minimum-maximum scaling and a time window, to predict the electricity prices of the future. The results show that this hybrid model is good than the standalone ones in terms of accuracy. Although we got our accuracy rating of 97.08, it shows higher accompaniments than remaining models RNN and ANN with accuracies of 96.64 and 96.63 respectively. 

---
# Data Augmentation for Cognitive Behavioral Therapy: Leveraging ERNIE Language Models using Artificial Intelligence 

**Authors**: Bosubabu Sambana, Kondreddygari Archana, Suram Indhra Sena Reddy, Shaik Meethaigar Jameer Basha, Shaik Karishma  

**Link**: [PDF](https://arxiv.org/pdf/2506.23503)  

**Abstract**: Cognitive Behavioral Therapy (CBT) is a proven approach for addressing the irrational thought patterns associated with mental health disorders, but its effectiveness relies on accurately identifying cognitive pathways to provide targeted treatment. In today's digital age, individuals often express negative emotions on social media, where they may reveal cognitive distortions, and in severe cases, exhibit suicidal tendencies. However, there is a significant gap in methodologies designed to analyze these cognitive pathways, which could be critical for psychotherapists aiming to deliver timely and effective interventions in online environments. Cognitive Behavioral Therapy (CBT) framework leveraging acceptance, commitment and data augmentation to categorize and address both textual and visual content as positive or negative. Specifically, the system employs BERT, RoBERTa for Sentiment Analysis and T5, PEGASUS for Text Summarization, mT5 for Text Translation in Multiple Languages focusing on detecting negative emotions and cognitive distortions within social media data. While existing models are primarily designed to identify negative thoughts, the proposed system goes beyond this by predicting additional negative side effects and other potential mental health disorders likes Phobias, Eating Disorders. This enhancement allows for a more comprehensive understanding and intervention strategy, offering psychotherapists a powerful tool for early detection and treatment of various psychological issues. 

---
# The Confidence Paradox: Can LLM Know When It's Wrong 

**Authors**: Sahil Tripathi, Md Tabrez Nafis, Imran Hussain, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23464)  

**Abstract**: Document Visual Question Answering (DocVQA) systems are increasingly deployed in real world applications, yet they remain ethically opaque-often producing overconfident answers to ambiguous questions or failing to communicate uncertainty in a trustworthy manner. This misalignment between model confidence and actual knowledge poses significant risks, particularly in domains requiring ethical accountability. Existing approaches such as LayoutLMv3, UDOP, and DONUT have advanced SOTA performance by focusing on architectural sophistication and accuracy; however, they fall short in ethical responsiveness.
To address these limitations, we introduce HonestVQA, a self-supervised honesty calibration framework for ethically aligned DocVQA. Our model-agnostic method quantifies uncertainty to identify knowledge gaps, aligns model confidence with actual correctness using weighted loss functions, and enforces ethical response behavior via contrastive learning. We further introduce two principled evaluation metrics--Honesty Score (H-Score) and Ethical Confidence Index (ECI)--to benchmark alignment between confidence, accuracy, and ethical communication. Empirically, HonestVQA improves DocVQA accuracy by up to 4.3% and F1 by 4.3% across SpDocVQA, InfographicsVQA, and SROIE datasets. It reduces overconfidence, lowering H-Score and ECI by 0.072 and 0.078, respectively. In cross domain evaluation, it achieves up to 78.9% accuracy and 76.1% F1-score, demonstrating strong generalization. Ablation shows a 3.8% drop in accuracy without alignment or contrastive loss. 

---
# GATSim: Urban Mobility Simulation with Generative Agents 

**Authors**: Qi Liu, Can Li, Wanjing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.23306)  

**Abstract**: Traditional agent-based urban mobility simulations rely on rigid rule-based systems that fail to capture the complexity, adaptability, and behavioral diversity characteristic of human travel decision-making. Recent advances in large language models and AI agent technology offer opportunities to create agents with reasoning capabilities, persistent memory, and adaptive learning mechanisms. We propose GATSim (Generative-Agent Transport Simulation), a novel framework that leverages these advances to create generative agents with rich behavioral characteristics for urban mobility simulation. Unlike conventional approaches, GATSim agents possess diverse socioeconomic attributes, individual lifestyles, and evolving preferences that shape their mobility decisions through psychologically-informed memory systems, tool usage capabilities, and lifelong learning mechanisms. The main contributions of this study include: (1) a comprehensive architecture combining an urban mobility foundation model with agent cognitive systems and transport simulation environment, (2) a fully functional prototype implementation, and (3) systematic validation demonstrating that generative agents produce believable travel behaviors. Through designed reflection processes, generative agents in this study can transform specific travel experiences into generalized insights, enabling realistic behavioral adaptation over time with specialized mechanisms for activity planning and real-time reactive behaviors tailored to urban mobility contexts. Experiments show that generative agents perform competitively with human annotators in mobility scenarios while naturally producing macroscopic traffic evolution patterns. The code for the prototype system is shared at this https URL. 

---
# Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games 

**Authors**: David Guzman Piedrahita, Yongjin Yang, Mrinmaya Sachan, Giorgia Ramponi, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23276)  

**Abstract**: As large language models (LLMs) are increasingly deployed as autonomous agents, understanding their cooperation and social mechanisms is becoming increasingly important. In particular, how LLMs balance self-interest and collective well-being is a critical challenge for ensuring alignment, robustness, and safe deployment. In this paper, we examine the challenge of costly sanctioning in multi-agent LLM systems, where an agent must decide whether to invest its own resources to incentivize cooperation or penalize defection. To study this, we adapt a public goods game with institutional choice from behavioral economics, allowing us to observe how different LLMs navigate social dilemmas over repeated interactions. Our analysis reveals four distinct behavioral patterns among models: some consistently establish and sustain high levels of cooperation, others fluctuate between engagement and disengagement, some gradually decline in cooperative behavior over time, and others rigidly follow fixed strategies regardless of outcomes. Surprisingly, we find that reasoning LLMs, such as the o1 series, struggle significantly with cooperation, whereas some traditional LLMs consistently achieve high levels of cooperation. These findings suggest that the current approach to improving LLMs, which focuses on enhancing their reasoning capabilities, does not necessarily lead to cooperation, providing valuable insights for deploying LLM agents in environments that require sustained collaboration. Our code is available at this https URL 

---
# FinStat2SQL: A Text2SQL Pipeline for Financial Statement Analysis 

**Authors**: Quang Hung Nguyen, Phuong Anh Trinh, Phan Quoc Hung Mai, Tuan Phong Trinh  

**Link**: [PDF](https://arxiv.org/pdf/2506.23273)  

**Abstract**: Despite the advancements of large language models, text2sql still faces many challenges, particularly with complex and domain-specific queries. In finance, database designs and financial reporting layouts vary widely between financial entities and countries, making text2sql even more challenging. We present FinStat2SQL, a lightweight text2sql pipeline enabling natural language queries over financial statements. Tailored to local standards like VAS, it combines large and small language models in a multi-agent setup for entity extraction, SQL generation, and self-correction. We build a domain-specific database and evaluate models on a synthetic QA dataset. A fine-tuned 7B model achieves 61.33\% accuracy with sub-4-second response times on consumer hardware, outperforming GPT-4o-mini. FinStat2SQL offers a scalable, cost-efficient solution for financial analysis, making AI-powered querying accessible to Vietnamese enterprises. 

---
# Rises for Measuring Local Distributivity in Lattices 

**Authors**: Mohammad Abdulla, Tobias Hille, Dominik Dürrschnabel, Gerd Stumme  

**Link**: [PDF](https://arxiv.org/pdf/2506.23168)  

**Abstract**: Distributivity is a well-established and extensively studied notion in lattice theory. In the context of data analysis, particularly within Formal Concept Analysis (FCA), lattices are often observed to exhibit a high degree of distributivity. However, no standardized measure exists to quantify this property. In this paper, we introduce the notion of rises in (concept) lattices as a means to assess distributivity. Rises capture how the number of attributes or objects in covering concepts change within the concept lattice. We show that a lattice is distributive if and only if no non-unit rises occur. Furthermore, we relate rises to the classical notion of meet- and join distributivity. We observe that concept lattices from real-world data are to a high degree join-distributive, but much less meet-distributive. We additionally study how join-distributivity manifests on the level of ordered sets. 

---
# Context-Driven Knowledge Graph Completion with Semantic-Aware Relational Message Passing 

**Authors**: Siyuan Li, Ruitong Liu, Yan Wen, Te Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.23141)  

**Abstract**: Semantic context surrounding a triplet $(h, r, t)$ is crucial for Knowledge Graph Completion (KGC), providing vital cues for prediction. However, traditional node-based message passing mechanisms, when applied to knowledge graphs, often introduce noise and suffer from information dilution or over-smoothing by indiscriminately aggregating information from all neighboring edges. To address this challenge, we propose a semantic-aware relational message passing. A core innovation of this framework is the introduction of a \textbf{semantic-aware Top-K neighbor selection strategy}. Specifically, this strategy first evaluates the semantic relevance between a central node and its incident edges within a shared latent space, selecting only the Top-K most pertinent ones. Subsequently, information from these selected edges is effectively fused with the central node's own representation using a \textbf{multi-head attention aggregator} to generate a semantically focused node message. In this manner, our model not only leverages the structure and features of edges within the knowledge graph but also more accurately captures and propagates the contextual information most relevant to the specific link prediction task, thereby effectively mitigating interference from irrelevant information. Extensive experiments demonstrate that our method achieves superior performance compared to existing approaches on several established benchmarks. 

---
# Are Large Language Models Capable of Deep Relational Reasoning? Insights from DeepSeek-R1 and Benchmark Comparisons 

**Authors**: Chi Chiu So, Yueyue Sun, Jun-Min Wang, Siu Pang Yung, Anthony Wai Keung Loh, Chun Pong Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.23128)  

**Abstract**: How far are Large Language Models (LLMs) in performing deep relational reasoning? In this paper, we evaluate and compare the reasoning capabilities of three cutting-edge LLMs, namely, DeepSeek-R1, DeepSeek-V3 and GPT-4o, through a suite of carefully designed benchmark tasks in family tree and general graph reasoning. Our experiments reveal that DeepSeek-R1 consistently achieves the highest F1-scores across multiple tasks and problem sizes, demonstrating strong aptitude in logical deduction and relational inference. However, all evaluated models, including DeepSeek-R1, struggle significantly as problem complexity increases, largely due to token length limitations and incomplete output structures. A detailed analysis of DeepSeek-R1's long Chain-of-Thought responses uncovers its unique planning and verification strategies, but also highlights instances of incoherent or incomplete reasoning, calling attention to the need for deeper scrutiny into LLMs' internal inference dynamics. We further discuss key directions for future work, including the role of multimodal reasoning and the systematic examination of reasoning failures. Our findings provide both empirical insights and theoretical implications for advancing LLMs' reasoning abilities, particularly in tasks that demand structured, multi-step logical inference. Our code repository will be publicly available at this https URL. 

---
# The Societal Impact of Foundation Models: Advancing Evidence-based AI Policy 

**Authors**: Rishi Bommasani  

**Link**: [PDF](https://arxiv.org/pdf/2506.23123)  

**Abstract**: Artificial intelligence is humanity's most promising technology because of the remarkable capabilities offered by foundation models. Yet, the same technology brings confusion and consternation: foundation models are poorly understood and they may precipitate a wide array of harms. This dissertation explains how technology and society coevolve in the age of AI, organized around three themes. First, the conceptual framing: the capabilities, risks, and the supply chain that grounds foundation models in the broader economy. Second, the empirical insights that enrich the conceptual foundations: transparency created via evaluations at the model level and indexes at the organization level. Finally, the transition from understanding to action: superior understanding of the societal impact of foundation models advances evidence-based AI policy. View together, this dissertation makes inroads into achieving better societal outcomes in the age of AI by building the scientific foundations and research-policy interface required for better AI governance. 

---
# Can Large Language Models Capture Human Risk Preferences? A Cross-Cultural Study 

**Authors**: Bing Song, Jianing Liu, Sisi Jian, Chenyang Wu, Vinayak Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2506.23107)  

**Abstract**: Large language models (LLMs) have made significant strides, extending their applications to dialogue systems, automated content creation, and domain-specific advisory tasks. However, as their use grows, concerns have emerged regarding their reliability in simulating complex decision-making behavior, such as risky decision-making, where a single choice can lead to multiple outcomes. This study investigates the ability of LLMs to simulate risky decision-making scenarios. We compare model-generated decisions with actual human responses in a series of lottery-based tasks, using transportation stated preference survey data from participants in Sydney, Dhaka, Hong Kong, and Nanjing. Demographic inputs were provided to two LLMs -- ChatGPT 4o and ChatGPT o1-mini -- which were tasked with predicting individual choices. Risk preferences were analyzed using the Constant Relative Risk Aversion (CRRA) framework. Results show that both models exhibit more risk-averse behavior than human participants, with o1-mini aligning more closely with observed human decisions. Further analysis of multilingual data from Nanjing and Hong Kong indicates that model predictions in Chinese deviate more from actual responses compared to English, suggesting that prompt language may influence simulation performance. These findings highlight both the promise and the current limitations of LLMs in replicating human-like risk behavior, particularly in linguistic and cultural settings. 

---
# AI's Euclid's Elements Moment: From Language Models to Computable Thought 

**Authors**: Xinmin Fang, Lingfeng Tao, Zhengxiong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23080)  

**Abstract**: This paper presents a comprehensive five-stage evolutionary framework for understanding the development of artificial intelligence, arguing that its trajectory mirrors the historical progression of human cognitive technologies. We posit that AI is advancing through distinct epochs, each defined by a revolutionary shift in its capacity for representation and reasoning, analogous to the inventions of cuneiform, the alphabet, grammar and logic, mathematical calculus, and formal logical systems. This "Geometry of Cognition" framework moves beyond mere metaphor to provide a systematic, cross-disciplinary model that not only explains AI's past architectural shifts-from expert systems to Transformers-but also charts a concrete and prescriptive path forward. Crucially, we demonstrate that this evolution is not merely linear but reflexive: as AI advances through these stages, the tools and insights it develops create a feedback loop that fundamentally reshapes its own underlying architecture. We are currently transitioning into a "Metalinguistic Moment," characterized by the emergence of self-reflective capabilities like Chain-of-Thought prompting and Constitutional AI. The subsequent stages, the "Mathematical Symbolism Moment" and the "Formal Logic System Moment," will be defined by the development of a computable calculus of thought, likely through neuro-symbolic architectures and program synthesis, culminating in provably aligned and reliable AI that reconstructs its own foundational representations. This work serves as the methodological capstone to our trilogy, which previously explored the economic drivers ("why") and cognitive nature ("what") of AI. Here, we address the "how," providing a theoretical foundation for future research and offering concrete, actionable strategies for startups and developers aiming to build the next generation of intelligent systems. 

---
# AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks 

**Authors**: Leander Melroy Maben, Gayathri Ganesh Lakshmy, Srijith Radhakrishnan, Siddhant Arora, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.23049)  

**Abstract**: Despite advances in language and speech technologies, no open-source system enables full speech-to-speech, multi-turn dialogue with integrated tool use and agentic reasoning. We introduce AURA (Agent for Understanding, Reasoning, and Automated Tool Use), the first open-source, speech-native assistant capable of completing complex, goal-driven tasks through dynamic tool invocation and multi-turn conversation. AURA combines open-weight ASR, TTS, and LLMs in a cascaded pipeline and supports tools such as calendar booking, contact lookup, web search, and email. Its modular design allows easy integration of new tools using natural language prompts and action classes. On VoiceBench, AURA scores 92.75% on OpenBookQA-outperforming all open-weight systems and nearing GPT-4o-and 4.39 on AlpacaEval, competitive with other open-weight systems. Human evaluation shows 90% task success on complex, multi-turn speech tasks. 

---
# MARBLE: A Hard Benchmark for Multimodal Spatial Reasoning and Planning 

**Authors**: Yulun Jiang, Yekun Chai, Maria Brbić, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2506.22992)  

**Abstract**: The ability to process information from multiple modalities and to reason through it step-by-step remains a critical challenge in advancing artificial intelligence. However, existing reasoning benchmarks focus on text-only reasoning, or employ multimodal questions that can be answered by directly retrieving information from a non-text modality. Thus, complex reasoning remains poorly understood in multimodal domains. Here, we present MARBLE, a challenging multimodal reasoning benchmark that is designed to scrutinize multimodal language models (MLLMs) in their ability to carefully reason step-by-step through complex multimodal problems and environments. MARBLE is composed of two highly challenging tasks, M-Portal and M-Cube, that require the crafting and understanding of multistep plans under spatial, visual, and physical constraints. We find that current MLLMs perform poorly on MARBLE -- all the 12 advanced models obtain near-random performance on M-Portal and 0% accuracy on M-Cube. Only in simplified subtasks some models outperform the random baseline, indicating that complex reasoning is still a challenge for existing MLLMs. Moreover, we show that perception remains a bottleneck, where MLLMs occasionally fail to extract information from the visual inputs. By shedding a light on the limitations of MLLMs, we hope that MARBLE will spur the development of the next generation of models with the ability to reason and plan across many, multimodal reasoning steps. 

---
# Improving Rationality in the Reasoning Process of Language Models through Self-playing Game 

**Authors**: Pinzheng Wang, Juntao Li, Zecheng Tang, Haijia Gui, Min zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22920)  

**Abstract**: Large language models (LLMs) have demonstrated considerable reasoning abilities in various tasks such as mathematics and coding. However, recent studies indicate that even the best models lack true comprehension of their reasoning processes. In this paper, we explore how self-play can enhance the rationality of models in the reasoning process without supervision from humans or superior models. We design a Critic-Discernment Game(CDG) in which a prover first provides a solution to a given problem and is subsequently challenged by critiques of its solution. These critiques either aim to assist or mislead the prover. The objective of the prover is to maintain the correct answer when faced with misleading comments, while correcting errors in response to constructive feedback. Our experiments on tasks involving mathematical reasoning, stepwise error detection, self-correction, and long-chain reasoning demonstrate that CDG training can significantly improve the ability of well-aligned LLMs to comprehend their reasoning process. 

---
# Hecto: Modular Sparse Experts for Adaptive and Interpretable Reasoning 

**Authors**: Sanskar Pandey, Ruhaan Chopra, Saad Murtaza Bhat, Ark Abhyudaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.22919)  

**Abstract**: Mixture-of-Experts (MoE) models enable conditional computation by routing inputs to specialized experts, but these experts rely on identical inductive biases, thus limiting representational diversity. This static computation pathway is inefficient for inputs that require different types of reasoning and limits specialization and interpretability. We propose Hecto, a lightweight MoE architecture that leverages architectural heterogeneity by combining a GRU expert for temporal reasoning and an FFNN expert for static abstraction under a sparse Top-1 gating mechanism. Evaluated on three reasoning benchmarks (AG News, SST-2, HotpotQA) and a regression task (STS-B), Hecto matches or closely trails homogeneous baselines in performance despite receiving isolated input representations, while achieving clear expert specialization, with each expert aligning to distinct reasoning types (temporal vs static). At larger batch sizes, Hecto exhibits improved performance, benefiting from relaxed computational constraints that allow its heterogeneous architecture to optimize more effectively. Ablation results isolate architectural diversity as the source of Hecto's stability and interpretability across diverse reasoning tasks. Overall, Hecto establishes itself as a new benchmark for conditional computation, offering a principled framework for specialized reasoning in low-resource regimes with its model strength derived from principled specialization. 

---
# Agentic Enterprise: AI-Centric User to User-Centric AI 

**Authors**: Arpit Narechania, Alex Endert, Atanu R Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2506.22893)  

**Abstract**: After a very long winter, the Artificial Intelligence (AI) spring is here. Or, so it seems over the last three years. AI has the potential to impact many areas of human life - personal, social, health, education, professional. In this paper, we take a closer look at the potential of AI for Enterprises, where decision-making plays a crucial and repeated role across functions, tasks, and operations. We consider Agents imbued with AI as means to increase decision-productivity of enterprises. We highlight six tenets for Agentic success in enterprises, by drawing attention to what the current, AI-Centric User paradigm misses, in the face of persistent needs of and usefulness for Enterprise Decision-Making. In underscoring a shift to User-Centric AI, we offer six tenets and promote market mechanisms for platforms, aligning the design of AI and its delivery by Agents to the cause of enterprise users. 

---
# ReasonBridge: Efficient Reasoning Transfer from Closed to Open-Source Language Models 

**Authors**: Ziqi Zhong, Xunzhu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22865)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have revealed a significant performance gap between closed-source and open-source models, particularly in tasks requiring complex reasoning and precise instruction following. This paper introduces ReasonBridge, a methodology that efficiently transfers reasoning capabilities from powerful closed-source to open-source models through a novel hierarchical knowledge distillation framework. We develop a tailored dataset Reason1K with only 1,000 carefully curated reasoning traces emphasizing difficulty, diversity, and quality. These traces are filtered from across multiple domains using a structured multi-criteria selection algorithm. Our transfer learning approach incorporates: (1) a hierarchical distillation process capturing both strategic abstraction and tactical implementation patterns, (2) a sparse reasoning-focused adapter architecture requiring only 0.3% additional trainable parameters, and (3) a test-time compute scaling mechanism using guided inference interventions. Comprehensive evaluations demonstrate that ReasonBridge improves reasoning capabilities in open-source models by up to 23% on benchmark tasks, significantly narrowing the gap with closed-source models. Notably, the enhanced Qwen2.5-14B outperforms Claude-Sonnet3.5 on MATH500 and matches its performance on competition-level AIME problems. Our methodology generalizes effectively across diverse reasoning domains and model architectures, establishing a sample-efficient approach to reasoning enhancement for instruction following. 

---
# Bridging Ethical Principles and Algorithmic Methods: An Alternative Approach for Assessing Trustworthiness in AI Systems 

**Authors**: Michael Papademas, Xenia Ziouvelou, Antonis Troumpoukis, Vangelis Karkaletsis  

**Link**: [PDF](https://arxiv.org/pdf/2506.22774)  

**Abstract**: Artificial Intelligence (AI) technology epitomizes the complex challenges posed by human-made artifacts, particularly those widely integrated into society and exert significant influence, highlighting potential benefits and their negative consequences. While other technologies may also pose substantial risks, AI's pervasive reach makes its societal effects especially profound. The complexity of AI systems, coupled with their remarkable capabilities, can lead to a reliance on technologies that operate beyond direct human oversight or understanding. To mitigate the risks that arise, several theoretical tools and guidelines have been developed, alongside efforts to create technological tools aimed at safeguarding Trustworthy AI. The guidelines take a more holistic view of the issue but fail to provide techniques for quantifying trustworthiness. Conversely, while technological tools are better at achieving such quantification, they lack a holistic perspective, focusing instead on specific aspects of Trustworthy AI. This paper aims to introduce an assessment method that combines the ethical components of Trustworthy AI with the algorithmic processes of PageRank and TrustRank. The goal is to establish an assessment framework that minimizes the subjectivity inherent in the self-assessment techniques prevalent in the field by introducing algorithmic criteria. The application of our approach indicates that a holistic assessment of an AI system's trustworthiness can be achieved by providing quantitative insights while considering the theoretical content of relevant guidelines. 

---
# Explanations are a means to an end 

**Authors**: Jessica Hullman, Ziyang Guo, Berk Ustun  

**Link**: [PDF](https://arxiv.org/pdf/2506.22740)  

**Abstract**: Modern methods for explainable machine learning are designed to describe how models map inputs to outputs--without deep consideration of how these explanations will be used in practice. This paper argues that explanations should be designed and evaluated with a specific end in mind. We describe how to formalize this end in a framework based in statistical decision theory. We show how this functionally-grounded approach can be applied across diverse use cases, such as clinical decision support, providing recourse, or debugging. We demonstrate its use to characterize the maximum "boost" in performance on a particular task that an explanation could provide an idealized decision-maker, preventing misuse due to ambiguity by forcing researchers to specify concrete use cases that can be analyzed in light of models of expected explanation use. We argue that evaluation should meld theoretical and empirical perspectives on the value of explanation, and contribute definitions that span these perspectives. 

---
# URSA: The Universal Research and Scientific Agent 

**Authors**: Michael Grosskopf, Russell Bent, Rahul Somasundaram, Isaac Michaud, Arthur Lui, Nathan Debardeleben, Earl Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2506.22653)  

**Abstract**: Large language models (LLMs) have moved far beyond their initial form as simple chatbots, now carrying out complex reasoning, planning, writing, coding, and research tasks. These skills overlap significantly with those that human scientists use day-to-day to solve complex problems that drive the cutting edge of research. Using LLMs in "agentic" AI has the potential to revolutionize modern science and remove bottlenecks to progress. In this work, we present URSA, a scientific agent ecosystem for accelerating research tasks. URSA consists of a set of modular agents and tools, including coupling to advanced physics simulation codes, that can be combined to address scientific problems of varied complexity and impact. This work highlights the architecture of URSA, as well as examples that highlight the potential of the system. 

---
# Ludax: A GPU-Accelerated Domain Specific Language for Board Games 

**Authors**: Graham Todd, Alexander G. Padula, Dennis J.N.J. Soemers, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2506.22609)  

**Abstract**: Games have long been used as benchmarks and testing environments for research in artificial intelligence. A key step in supporting this research was the development of game description languages: frameworks that compile domain-specific code into playable and simulatable game environments, allowing researchers to generalize their algorithms and approaches across multiple games without having to manually implement each one. More recently, progress in reinforcement learning (RL) has been largely driven by advances in hardware acceleration. Libraries like JAX allow practitioners to take full advantage of cutting-edge computing hardware, often speeding up training and testing by orders of magnitude. Here, we present a synthesis of these strands of research: a domain-specific language for board games which automatically compiles into hardware-accelerated code. Our framework, Ludax, combines the generality of game description languages with the speed of modern parallel processing hardware and is designed to fit neatly into existing deep learning pipelines. We envision Ludax as a tool to help accelerate games research generally, from RL to cognitive science, by enabling rapid simulation and providing a flexible representation scheme. We present a detailed breakdown of Ludax's description language and technical notes on the compilation process, along with speed benchmarking and a demonstration of training RL agents. The Ludax framework, along with implementations of existing board games, is open-source and freely available. 

---
# Bootstrapping Human-Like Planning via LLMs 

**Authors**: David Porfirio, Vincent Hsiao, Morgan Fine-Morris, Leslie Smith, Laura M. Hiatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.22604)  

**Abstract**: Robot end users increasingly require accessible means of specifying tasks for robots to perform. Two common end-user programming paradigms include drag-and-drop interfaces and natural language programming. Although natural language interfaces harness an intuitive form of human communication, drag-and-drop interfaces enable users to meticulously and precisely dictate the key actions of the robot's task. In this paper, we investigate the degree to which both approaches can be combined. Specifically, we construct a large language model (LLM)-based pipeline that accepts natural language as input and produces human-like action sequences as output, specified at a level of granularity that a human would produce. We then compare these generated action sequences to another dataset of hand-specified action sequences. Although our results reveal that larger models tend to outperform smaller ones in the production of human-like action sequences, smaller models nonetheless achieve satisfactory performance. 

---
# FADRM: Fast and Accurate Data Residual Matching for Dataset Distillation 

**Authors**: Jiacheng Cui, Xinyue Bi, Yaxin Luo, Xiaohan Zhao, Jiacheng Liu, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.24125)  

**Abstract**: Residual connection has been extensively studied and widely applied at the model architecture level. However, its potential in the more challenging data-centric approaches remains unexplored. In this work, we introduce the concept of Data Residual Matching for the first time, leveraging data-level skip connections to facilitate data generation and mitigate data information vanishing. This approach maintains a balance between newly acquired knowledge through pixel space optimization and existing core local information identification within raw data modalities, specifically for the dataset distillation task. Furthermore, by incorporating optimization-level refinements, our method significantly improves computational efficiency, achieving superior performance while reducing training time and peak GPU memory usage by 50%. Consequently, the proposed method Fast and Accurate Data Residual Matching for Dataset Distillation (FADRM) establishes a new state-of-the-art, demonstrating substantial improvements over existing methods across multiple dataset benchmarks in both efficiency and effectiveness. For instance, with ResNet-18 as the student model and a 0.8% compression ratio on ImageNet-1K, the method achieves 47.7% test accuracy in single-model dataset distillation and 50.0% in multi-model dataset distillation, surpassing RDED by +5.7% and outperforming state-of-the-art multi-model approaches, EDC and CV-DD, by +1.4% and +4.0%. Code is available at: this https URL. 

---
# Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime 

**Authors**: Yuqing Wang, Shangding Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.24120)  

**Abstract**: Data selection plays a crucial role in data-driven decision-making, including in large language models (LLMs), and is typically task-dependent. Properties such as data quality and diversity have been extensively studied and are known to enhance model performance. However, it remains unclear whether there exist other quantitative and general principles of data selection that can consistently improve performance, especially for complex tasks with limited prior knowledge. In this paper, we demonstrate that selecting more uniformly distributed data can improve training efficiency while enhancing performance. Specifically, we establish that more uniform (less biased) distribution leads to a larger minimum pairwise distance between data points, denoted by $h_{\min}$, and prove that a smaller $h_{\min}$ can slow down the training dynamics of gradient descent (GD). Moreover, we theoretically show that the approximation error of neural networks decreases as $h_{\min}$ increases. Our analysis introduces a convergence framework for GD beyond the Neural Tangent Kernel (NTK) regime, applicable to a broad class of architectures, including transformers, without requiring Lipschitz smoothness. This framework further provides theoretical justification for the use of residual connections and function compositions in deep neural architectures. In the end, we conduct comprehensive experiments for supervised fine-tuning across various settings, including different optimization strategies, model sizes, and training datasets. The results consistently demonstrate that selecting data by maximizing pairwise distance significantly accelerates training and achieves comparable or better performance in LLMs across diverse datasets. Code and Datasets are available at the link: this https URL. 

---
# Navigating with Annealing Guidance Scale in Diffusion Space 

**Authors**: Shai Yehezkel, Omer Dahary, Andrey Voynov, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2506.24108)  

**Abstract**: Denoising diffusion models excel at generating high-quality images conditioned on text prompts, yet their effectiveness heavily relies on careful guidance during the sampling process. Classifier-Free Guidance (CFG) provides a widely used mechanism for steering generation by setting the guidance scale, which balances image quality and prompt alignment. However, the choice of the guidance scale has a critical impact on the convergence toward a visually appealing and prompt-adherent image. In this work, we propose an annealing guidance scheduler which dynamically adjusts the guidance scale over time based on the conditional noisy signal. By learning a scheduling policy, our method addresses the temperamental behavior of CFG. Empirical results demonstrate that our guidance scheduler significantly enhances image quality and alignment with the text prompt, advancing the performance of text-to-image generation. Notably, our novel scheduler requires no additional activations or memory consumption, and can seamlessly replace the common classifier-free guidance, offering an improved trade-off between prompt alignment and quality. 

---
# On the Predictive Power of Representation Dispersion in Language Models 

**Authors**: Yanhong Li, Ming Li, Karen Livescu, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.24106)  

**Abstract**: We show that a language model's ability to predict text is tightly linked to the breadth of its embedding space: models that spread their contextual representations more widely tend to achieve lower perplexity. Concretely, we find that representation dispersion - the average pairwise cosine distance among hidden vectors - strongly and negatively correlates with perplexity across diverse model families (LLaMA, Qwen, and others) and domains (Wikipedia, news, scientific abstracts). Beyond illustrating this link, we show how dispersion can be leveraged for a range of practical tasks without requiring labeled data. First, measuring dispersion on unlabeled text allows us to predict downstream accuracy in new domains, offering a data-efficient tool for model selection. Next, we find that identifying layers with higher dispersion pinpoints the best representations for retrieval-based methods such as kNN-LM, bypassing exhaustive layer-by-layer searches. Finally, we integrate a simple push-away objective into training, which increases dispersion in both single-domain and cross-domain scenarios and directly improves perplexity in each. 

---
# Development of Hybrid Artificial Intelligence Training on Real and Synthetic Data: Benchmark on Two Mixed Training Strategies 

**Authors**: Paul Wachter, Lukas Niehaus, Julius Schöning  

**Link**: [PDF](https://arxiv.org/pdf/2506.24093)  

**Abstract**: Synthetic data has emerged as a cost-effective alternative to real data for training artificial neural networks (ANN). However, the disparity between synthetic and real data results in a domain gap. That gap leads to poor performance and generalization of the trained ANN when applied to real-world scenarios. Several strategies have been developed to bridge this gap, which combine synthetic and real data, known as mixed training using hybrid datasets. While these strategies have been shown to mitigate the domain gap, a systematic evaluation of their generalizability and robustness across various tasks and architectures remains underexplored. To address this challenge, our study comprehensively analyzes two widely used mixing strategies on three prevalent architectures and three distinct hybrid datasets. From these datasets, we sample subsets with varying proportions of synthetic to real data to investigate the impact of synthetic and real components. The findings of this paper provide valuable insights into optimizing the use of synthetic data in the training process of any ANN, contributing to enhancing robustness and efficacy. 

---
# Imagine for Me: Creative Conceptual Blending of Real Images and Text via Blended Attention 

**Authors**: Wonwoong Cho, Yanxia Zhang, Yan-Ying Chen, David I. Inouye  

**Link**: [PDF](https://arxiv.org/pdf/2506.24085)  

**Abstract**: Blending visual and textual concepts into a new visual concept is a unique and powerful trait of human beings that can fuel creativity. However, in practice, cross-modal conceptual blending for humans is prone to cognitive biases, like design fixation, which leads to local minima in the design space. In this paper, we propose a T2I diffusion adapter "IT-Blender" that can automate the blending process to enhance human creativity. Prior works related to cross-modal conceptual blending are limited in encoding a real image without loss of details or in disentangling the image and text inputs. To address these gaps, IT-Blender leverages pretrained diffusion models (SD and FLUX) to blend the latent representations of a clean reference image with those of the noisy generated image. Combined with our novel blended attention, IT-Blender encodes the real reference image without loss of details and blends the visual concept with the object specified by the text in a disentangled way. Our experiment results show that IT-Blender outperforms the baselines by a large margin in blending visual and textual concepts, shedding light on the new application of image generative models to augment human creativity. 

---
# SQUASH: A SWAP-Based Quantum Attack to Sabotage Hybrid Quantum Neural Networks 

**Authors**: Rahul Kumar, Wenqi Wei, Ying Mao, Junaid Farooq, Ying Wang, Juntao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.24081)  

**Abstract**: We propose a circuit-level attack, SQUASH, a SWAP-Based Quantum Attack to sabotage Hybrid Quantum Neural Networks (HQNNs) for classification tasks. SQUASH is executed by inserting SWAP gate(s) into the variational quantum circuit of the victim HQNN. Unlike conventional noise-based or adversarial input attacks, SQUASH directly manipulates the circuit structure, leading to qubit misalignment and disrupting quantum state evolution. This attack is highly stealthy, as it does not require access to training data or introduce detectable perturbations in input states. Our results demonstrate that SQUASH significantly degrades classification performance, with untargeted SWAP attacks reducing accuracy by up to 74.08\% and targeted SWAP attacks reducing target class accuracy by up to 79.78\%. These findings reveal a critical vulnerability in HQNN implementations, underscoring the need for more resilient architectures against circuit-level adversarial interventions. 

---
# STACK: Adversarial Attacks on LLM Safeguard Pipelines 

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave  

**Link**: [PDF](https://arxiv.org/pdf/2506.24068)  

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks. 

---
# A Survey on Vision-Language-Action Models for Autonomous Driving 

**Authors**: Sicong Jiang, Zilin Huang, Kangan Qian, Ziang Luo, Tianze Zhu, Yang Zhong, Yihong Tang, Menglin Kong, Yunlong Wang, Siwen Jiao, Hao Ye, Zihao Sheng, Xin Zhao, Tuopu Wen, Zheng Fu, Sikai Chen, Kun Jiang, Diange Yang, Seongjin Choi, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.24044)  

**Abstract**: The rapid progress of multimodal large language models (MLLM) has paved the way for Vision-Language-Action (VLA) paradigms, which integrate visual perception, natural language understanding, and control within a single policy. Researchers in autonomous driving are actively adapting these methods to the vehicle domain. Such models promise autonomous vehicles that can interpret high-level instructions, reason about complex traffic scenes, and make their own decisions. However, the literature remains fragmented and is rapidly expanding. This survey offers the first comprehensive overview of VLA for Autonomous Driving (VLA4AD). We (i) formalize the architectural building blocks shared across recent work, (ii) trace the evolution from early explainer to reasoning-centric VLA models, and (iii) compare over 20 representative models according to VLA's progress in the autonomous driving domain. We also consolidate existing datasets and benchmarks, highlighting protocols that jointly measure driving safety, accuracy, and explanation quality. Finally, we detail open challenges - robustness, real-time efficiency, and formal verification - and outline future directions of VLA4AD. This survey provides a concise yet complete reference for advancing interpretable socially aligned autonomous vehicles. Github repo is available at \href{this https URL}{SicongJiang/Awesome-VLA4AD}. 

---
# Bridging Theory and Practice in Link Representation with Graph Neural Networks 

**Authors**: Veronica Lachi, Francesco Ferrini, Antonio Longa, Bruno Lepri, Andrea Passerini, Manfred Jaeger  

**Link**: [PDF](https://arxiv.org/pdf/2506.24018)  

**Abstract**: Graph Neural Networks (GNNs) are widely used to compute representations of node pairs for downstream tasks such as link prediction. Yet, theoretical understanding of their expressive power has focused almost entirely on graph-level representations. In this work, we shift the focus to links and provide the first comprehensive study of GNN expressiveness in link representation. We introduce a unifying framework, the $k_\phi$-$k_\rho$-$m$ framework, that subsumes existing message-passing link models and enables formal expressiveness comparisons. Using this framework, we derive a hierarchy of state-of-the-art methods and offer theoretical tools to analyze future architectures. To complement our analysis, we propose a synthetic evaluation protocol comprising the first benchmark specifically designed to assess link-level expressiveness. Finally, we ask: does expressiveness matter in practice? We use a graph symmetry metric that quantifies the difficulty of distinguishing links and show that while expressive models may underperform on standard benchmarks, they significantly outperform simpler ones as symmetry increases, highlighting the need for dataset-aware model selection. 

---
# EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations 

**Authors**: Hyunjong Kim, Sangyeop Kim, Jongheon Jeong, Yeongjae Cho, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2506.24016)  

**Abstract**: Recent advances in large language models and vision-language models have led to growing interest in explainable evaluation metrics for image captioning. However, these metrics generate explanations without standardized criteria, and the overall quality of the generated explanations remains unverified. In this paper, we propose EXPERT, a reference-free evaluation metric that provides structured explanations based on three fundamental criteria: fluency, relevance, and descriptiveness. By constructing large-scale datasets of high-quality structured explanations, we develop a two-stage evaluation template to effectively supervise a vision-language model for both scoring and explanation generation. EXPERT achieves state-of-the-art results on benchmark datasets while providing significantly higher-quality explanations than existing metrics, as validated through comprehensive human evaluation. Our code and datasets are available at this https URL. 

---
# Bridging Physical and Digital Worlds: Embodied Large AI for Future Wireless Systems 

**Authors**: Xinquan Wang, Fenghao Zhu, Zhaohui Yang, Chongwen Huang, Xiaoming Chen, Zhaoyang Zhang, Sami Muhaidat, Mérouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.24009)  

**Abstract**: Large artificial intelligence (AI) models offer revolutionary potential for future wireless systems, promising unprecedented capabilities in network optimization and performance. However, current paradigms largely overlook crucial physical interactions. This oversight means they primarily rely on offline datasets, leading to difficulties in handling real-time wireless dynamics and non-stationary environments. Furthermore, these models often lack the capability for active environmental probing. This paper proposes a fundamental paradigm shift towards wireless embodied large AI (WELAI), moving from passive observation to active embodiment. We first identify key challenges faced by existing models, then we explore the design principles and system structure of WELAI. Besides, we outline prospective applications in next-generation wireless. Finally, through an illustrative case study, we demonstrate the effectiveness of WELAI and point out promising research directions for realizing adaptive, robust, and autonomous wireless systems. 

---
# STCLocker: Deadlock Avoidance Testing for Autonomous Driving Systems 

**Authors**: Mingfei Cheng, Renzhi Wang, Xiaofei Xie, Yuan Zhou, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.23995)  

**Abstract**: Autonomous Driving System (ADS) testing is essential to ensure the safety and reliability of autonomous vehicles (AVs) before deployment. However, existing techniques primarily focus on evaluating ADS functionalities in single-AV settings. As ADSs are increasingly deployed in multi-AV traffic, it becomes crucial to assess their cooperative performance, particularly regarding deadlocks, a fundamental coordination failure in which multiple AVs enter a circular waiting state indefinitely, resulting in motion planning failures. Despite its importance, the cooperative capability of ADSs to prevent deadlocks remains insufficiently underexplored. To address this gap, we propose the first dedicated Spatio-Temporal Conflict-Guided Deadlock Avoidance Testing technique, STCLocker, for generating DeadLock Scenarios (DLSs), where a group of AVs controlled by the ADS under test are in a circular wait state. STCLocker consists of three key components: Deadlock Oracle, Conflict Feedback, and Conflict-aware Scenario Generation. Deadlock Oracle provides a reliable black-box mechanism for detecting deadlock cycles among multiple AVs within a given scenario. Conflict Feedback and Conflict-aware Scenario Generation collaborate to actively guide AVs into simultaneous competition over spatial conflict resources (i.e., shared passing regions) and temporal competitive behaviors (i.e., reaching the conflict region at the same time), thereby increasing the effectiveness of generating conflict-prone deadlocks. We evaluate STCLocker on two types of ADSs: Roach, an end-to-end ADS, and OpenCDA, a module-based ADS supporting cooperative communication. Experimental results show that, on average, STCLocker generates more DLS than the best-performing baseline. 

---
# ADReFT: Adaptive Decision Repair for Safe Autonomous Driving via Reinforcement Fine-Tuning 

**Authors**: Mingfei Cheng, Xiaofei Xie, Renzhi Wang, Yuan Zhou, Ming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23960)  

**Abstract**: Autonomous Driving Systems (ADSs) continue to face safety-critical risks due to the inherent limitations in their design and performance capabilities. Online repair plays a crucial role in mitigating such limitations, ensuring the runtime safety and reliability of ADSs. Existing online repair solutions enforce ADS compliance by transforming unacceptable trajectories into acceptable ones based on predefined specifications, such as rule-based constraints or training datasets. However, these approaches often lack generalizability, adaptability and tend to be overly conservative, resulting in ineffective repairs that not only fail to mitigate safety risks sufficiently but also degrade the overall driving experience. To address this issue, we propose Adaptive Decision Repair (ADReFT), a novel and effective repair method that identifies safety-critical states through offline learning from failed tests and generates appropriate mitigation actions to improve ADS safety. Specifically, ADReFT incorporates a transformer-based model with two joint heads, State Monitor and Decision Adapter, designed to capture complex driving environment interactions to evaluate state safety severity and generate adaptive repair actions. Given the absence of oracles for state safety identification, we first pretrain ADReFT using supervised learning with coarse annotations, i.e., labeling states preceding violations as positive samples and others as negative samples. It establishes ADReFT's foundational capability to mitigate safety-critical violations, though it may result in somewhat conservative mitigation strategies. Therefore, we subsequently finetune ADReFT using reinforcement learning to improve its initial capability and generate more precise and contextually appropriate repair decisions. Our evaluation results illustrate that ADReFT achieves better repair performance. 

---
# Autonomy by Design: Preserving Human Autonomy in AI Decision-Support 

**Authors**: Stefan Buijsman, Sarah Carter, Juan Pablo Bermúdez  

**Link**: [PDF](https://arxiv.org/pdf/2506.23952)  

**Abstract**: AI systems increasingly support human decision-making across domains of professional, skill-based, and personal activity. While previous work has examined how AI might affect human autonomy globally, the effects of AI on domain-specific autonomy -- the capacity for self-governed action within defined realms of skill or expertise -- remain understudied. We analyze how AI decision-support systems affect two key components of domain-specific autonomy: skilled competence (the ability to make informed judgments within one's domain) and authentic value-formation (the capacity to form genuine domain-relevant values and preferences). By engaging with prior investigations and analyzing empirical cases across medical, financial, and educational domains, we demonstrate how the absence of reliable failure indicators and the potential for unconscious value shifts can erode domain-specific autonomy both immediately and over time. We then develop a constructive framework for autonomy-preserving AI support systems. We propose specific socio-technical design patterns -- including careful role specification, implementation of defeater mechanisms, and support for reflective practice -- that can help maintain domain-specific autonomy while leveraging AI capabilities. This framework provides concrete guidance for developing AI systems that enhance rather than diminish human agency within specialized domains of action. 

---
# Adapt Your Body: Mitigating Proprioception Shifts in Imitation Learning 

**Authors**: Fuhang Kuang, Jiacheng You, Yingdong Hu, Tong Zhang, Chuan Wen, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23944)  

**Abstract**: Imitation learning models for robotic tasks typically rely on multi-modal inputs, such as RGB images, language, and proprioceptive states. While proprioception is intuitively important for decision-making and obstacle avoidance, simply incorporating all proprioceptive states leads to a surprising degradation in imitation learning performance. In this work, we identify the underlying issue as the proprioception shift problem, where the distributions of proprioceptive states diverge significantly between training and deployment. To address this challenge, we propose a domain adaptation framework that bridges the gap by utilizing rollout data collected during deployment. Using Wasserstein distance, we quantify the discrepancy between expert and rollout proprioceptive states and minimize this gap by adding noise to both sets of states, proportional to the Wasserstein distance. This strategy enhances robustness against proprioception shifts by aligning the training and deployment distributions. Experiments on robotic manipulation tasks demonstrate the efficacy of our method, enabling the imitation policy to leverage proprioception while mitigating its adverse effects. Our approach outperforms the naive solution which discards proprioception, and other baselines designed to address distributional shifts. 

---
# QPART: Adaptive Model Quantization and Dynamic Workload Balancing for Accuracy-aware Edge Inference 

**Authors**: Xiangchen Li, Saeid Ghafouri, Bo Ji, Hans Vandierendonck, Deepu John, Dimitrios S. Nikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.23934)  

**Abstract**: As machine learning inferences increasingly move to edge devices, adapting to diverse computational capabilities, hardware, and memory constraints becomes more critical. Instead of relying on a pre-trained model fixed for all future inference queries across diverse edge devices, we argue that planning an inference pattern with a request-specific model tailored to the device's computational capacity, accuracy requirements, and time constraints is more cost-efficient and robust to diverse scenarios. To this end, we propose an accuracy-aware and workload-balanced inference system that integrates joint model quantization and inference partitioning. In this approach, the server dynamically responds to inference queries by sending a quantized model and adaptively sharing the inference workload with the device. Meanwhile, the device's computational power, channel capacity, and accuracy requirements are considered when deciding.
Furthermore, we introduce a new optimization framework for the inference system, incorporating joint model quantization and partitioning. Our approach optimizes layer-wise quantization bit width and partition points to minimize time consumption and cost while accounting for varying accuracy requirements of tasks through an accuracy degradation metric in our optimization model. To our knowledge, this work represents the first exploration of optimizing quantization layer-wise bit-width in the inference serving system, by introducing theoretical measurement of accuracy degradation. Simulation results demonstrate a substantial reduction in overall time and power consumption, with computation payloads decreasing by over 80% and accuracy degradation kept below 1%. 

---
# Leveraging the Potential of Prompt Engineering for Hate Speech Detection in Low-Resource Languages 

**Authors**: Ruhina Tabasshum Prome, Tarikul Islam Tamiti, Anomadarshi Barua  

**Link**: [PDF](https://arxiv.org/pdf/2506.23930)  

**Abstract**: The rapid expansion of social media leads to a marked increase in hate speech, which threatens personal lives and results in numerous hate crimes. Detecting hate speech presents several challenges: diverse dialects, frequent code-mixing, and the prevalence of misspelled words in user-generated content on social media platforms. Recent progress in hate speech detection is typically concentrated on high-resource languages. However, low-resource languages still face significant challenges due to the lack of large-scale, high-quality datasets. This paper investigates how we can overcome this limitation via prompt engineering on large language models (LLMs) focusing on low-resource Bengali language. We investigate six prompting strategies - zero-shot prompting, refusal suppression, flattering the classifier, multi-shot prompting, role prompting, and finally our innovative metaphor prompting to detect hate speech effectively in low-resource languages. We pioneer the metaphor prompting to circumvent the built-in safety mechanisms of LLMs that marks a significant departure from existing jailbreaking methods. We investigate all six different prompting strategies on the Llama2-7B model and compare the results extensively with three pre-trained word embeddings - GloVe, Word2Vec, and FastText for three different deep learning models - multilayer perceptron (MLP), convolutional neural network (CNN), and bidirectional gated recurrent unit (BiGRU). To prove the effectiveness of our metaphor prompting in the low-resource Bengali language, we also evaluate it in another low-resource language - Hindi, and two high-resource languages - English and German. The performance of all prompting techniques is evaluated using the F1 score, and environmental impact factor (IF), which measures CO$_2$ emissions, electricity usage, and computational time. 

---
# Reinforcement Learning for Synchronised Flow Control in a Dual-Gate Resin Infusion System 

**Authors**: Miguel Camacho-Sánchez, Fernando García-Torres, Jesper John Lisegaard, Rocío del Amor, Sankhya Mohanty, Valery Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2506.23923)  

**Abstract**: Resin infusion (RI) and resin transfer moulding (RTM) are critical processes for the manufacturing of high-performance fibre-reinforced polymer composites, particularly for large-scale applications such as wind turbine blades. Controlling the resin flow dynamics in these processes is critical to ensure the uniform impregnation of the fibre reinforcements, thereby preventing residual porosities and dry spots that impact the consequent structural integrity of the final component. This paper presents a reinforcement learning (RL) based strategy, established using process simulations, for synchronising the different resin flow fronts in an infusion scenario involving two resin inlets and a single outlet. Using Proximal Policy Optimisation (PPO), our approach addresses the challenge of managing the fluid dynamics in a partially observable environment. The results demonstrate the effectiveness of the RL approach in achieving an accurate flow convergence, highlighting its potential towards improving process control and product quality in composites manufacturing. 

---
# GroundingDINO-US-SAM: Text-Prompted Multi-Organ Segmentation in Ultrasound with LoRA-Tuned Vision-Language Models 

**Authors**: Hamza Rasaee, Taha Koleilat, Hassan Rivaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.23903)  

**Abstract**: Accurate and generalizable object segmentation in ultrasound imaging remains a significant challenge due to anatomical variability, diverse imaging protocols, and limited annotated data. In this study, we propose a prompt-driven vision-language model (VLM) that integrates Grounding DINO with SAM2 to enable object segmentation across multiple ultrasound organs. A total of 18 public ultrasound datasets, encompassing the breast, thyroid, liver, prostate, kidney, and paraspinal muscle, were utilized. These datasets were divided into 15 for fine-tuning and validation of Grounding DINO using Low Rank Adaptation (LoRA) to the ultrasound domain, and 3 were held out entirely for testing to evaluate performance in unseen distributions. Comprehensive experiments demonstrate that our approach outperforms state-of-the-art segmentation methods, including UniverSeg, MedSAM, MedCLIP-SAM, BiomedParse, and SAMUS on most seen datasets while maintaining strong performance on unseen datasets without additional fine-tuning. These results underscore the promise of VLMs in scalable and robust ultrasound image analysis, reducing dependence on large, organ-specific annotated datasets. We will publish our code on this http URL after acceptance. 

---
# Chain of Thought in Order: Discovering Learning-Friendly Orders for Arithmetic 

**Authors**: Yuta Sato, Kazuhiko Kawamoto, Hiroshi Kera  

**Link**: [PDF](https://arxiv.org/pdf/2506.23875)  

**Abstract**: The chain of thought is fundamental in Transformers, which is to perform step-by-step reasoning. Besides what intermediate steps work, the order of these steps critically affects the difficulty of the reasoning. This study addresses a novel task of unraveling chain of thought - reordering decoder input tokens to a learning-friendly sequence for Transformers to learn arithmetic tasks. The proposed pipeline first trains a Transformer on a mixture of target sequences arranged in different orders and then identifies benign orders as those with fast loss drops in the early stage. As the search space grows factorially with sequence length, we propose a two-stage hierarchical approach for inter- and intra-block reordering. Experiments on four order-sensitive arithmetic tasks show that our method identifies a learning-friendly order out of a few billion candidates. Notably, on the multiplication task, it recovered the reverse-digit order reported in prior studies. 

---
# Scaling Self-Supervised Representation Learning for Symbolic Piano Performance 

**Authors**: Louis Bradshaw, Honglu Fan, Alexander Spangher, Stella Biderman, Simon Colton  

**Link**: [PDF](https://arxiv.org/pdf/2506.23869)  

**Abstract**: We study the capabilities of generative autoregressive transformer models trained on large amounts of symbolic solo-piano transcriptions. After first pretraining on approximately 60,000 hours of music, we use a comparatively smaller, high-quality subset, to finetune models to produce musical continuations, perform symbolic classification tasks, and produce general-purpose contrastive MIDI embeddings by adapting the SimCLR framework to symbolic music. When evaluating piano continuation coherence, our generative model outperforms leading symbolic generation techniques and remains competitive with proprietary audio generation models. On MIR classification benchmarks, frozen representations from our contrastive model achieve state-of-the-art results in linear probe experiments, while direct finetuning demonstrates the generalizability of pretrained representations, often requiring only a few hundred labeled examples to specialize to downstream tasks. 

---
# Differentially Private Synthetic Data Release for Topics API Outputs 

**Authors**: Travis Dick, Alessandro Epasto, Adel Javanmard, Josh Karlin, Andres Munoz Medina, Vahab Mirrokni, Sergei Vassilvitskii, Peilin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2506.23855)  

**Abstract**: The analysis of the privacy properties of Privacy-Preserving Ads APIs is an area of research that has received strong interest from academics, industry, and regulators. Despite this interest, the empirical study of these methods is hindered by the lack of publicly available data. Reliable empirical analysis of the privacy properties of an API, in fact, requires access to a dataset consisting of realistic API outputs; however, privacy concerns prevent the general release of such data to the public.
In this work, we develop a novel methodology to construct synthetic API outputs that are simultaneously realistic enough to enable accurate study and provide strong privacy protections. We focus on one Privacy-Preserving Ads APIs: the Topics API, part of Google Chrome's Privacy Sandbox. We developed a methodology to generate a differentially-private dataset that closely matches the re-identification risk properties of the real Topics API data. The use of differential privacy provides strong theoretical bounds on the leakage of private user information from this release.
Our methodology is based on first computing a large number of differentially-private statistics describing how output API traces evolve over time. Then, we design a parameterized distribution over sequences of API traces and optimize its parameters so that they closely match the statistics obtained. Finally, we create the synthetic data by drawing from this distribution.
Our work is complemented by an open-source release of the anonymized dataset obtained by this methodology. We hope this will enable external researchers to analyze the API in-depth and replicate prior and future work on a realistic large-scale dataset. We believe that this work will contribute to fostering transparency regarding the privacy properties of Privacy-Preserving Ads APIs. 

---
# Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts 

**Authors**: Kenny Peng, Rajiv Movva, Jon Kleinberg, Emma Pierson, Nikhil Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.23845)  

**Abstract**: While sparse autoencoders (SAEs) have generated significant excitement, a series of negative results have added to skepticism about their usefulness. Here, we establish a conceptual distinction that reconciles competing narratives surrounding SAEs. We argue that while SAEs may be less effective for acting on known concepts, SAEs are powerful tools for discovering unknown concepts. This distinction cleanly separates existing negative and positive results, and suggests several classes of SAE applications. Specifically, we outline use cases for SAEs in (i) ML interpretability, explainability, fairness, auditing, and safety, and (ii) social and health sciences. 

---
# Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model 

**Authors**: Bowen Ding, Yuhan Chen, Futing Wang, Lingfeng Ming, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23840)  

**Abstract**: Large Reasoning Models (LRMs) excel at solving complex problems but face an overthinking dilemma. When handling simple tasks, they often produce verbose responses overloaded with thinking tokens (e.g., wait, however). These tokens trigger unnecessary high-level reasoning behaviors like reflection and backtracking, reducing efficiency. In this work, our pilot study reveals that these thinking-token-induced behaviors are not essential for effective problem-solving and may even hinder correct reasoning within constrained token budgets. We identify this phenomenon as the thinking trap. To mitigate this issue, we propose Dual Policy Preference Optimization (DuP-PO), a novel algorithm featuring: (1) A rollout sampling strategy that guarantees balanced exposure to responses with and without thinking tokens; (2) A fine-grained advantage control technique to dynamically regulate the prediction of target tokens; (3) A policy shaping method ensuring stable gradient contributions from thinking tokens. Experimental results on five popular math reasoning benchmarks show that DuP-PO performs well on the popular LRM, which significantly improves their token efficiency during reasoning, while achieving superior performance of the base model. 

---
# Towards the "Digital Me": A vision of authentic Conversational Agents powered by personal Human Digital Twins 

**Authors**: Lluís C. Coll, Martin W. Lauer-Schmaltz, Philip Cash, John P. Hansen, Anja Maier  

**Link**: [PDF](https://arxiv.org/pdf/2506.23826)  

**Abstract**: Human Digital Twins (HDTs) have traditionally been conceptualized as data-driven models designed to support decision-making across various domains. However, recent advancements in conversational AI open new possibilities for HDTs to function as authentic, interactive digital counterparts of individuals. This paper introduces a novel HDT system architecture that integrates large language models with dynamically updated personal data, enabling it to mirror an individual's conversational style, memories, and behaviors. To achieve this, our approach implements context-aware memory retrieval, neural plasticity-inspired consolidation, and adaptive learning mechanisms, creating a more natural and evolving digital persona. The resulting system does not only replicate an individual's unique conversational style depending on who they are speaking with, but also enriches responses with dynamically captured personal experiences, opinions, and memories. While this marks a significant step toward developing authentic virtual counterparts, it also raises critical ethical concerns regarding privacy, accountability, and the long-term implications of persistent digital identities. This study contributes to the field of HDTs by describing our novel system architecture, demonstrating its capabilities, and discussing future directions and emerging challenges to ensure the responsible and ethical development of HDTs. 

---
# The Impact of AI on Educational Assessment: A Framework for Constructive Alignment 

**Authors**: Patrick Stokkink  

**Link**: [PDF](https://arxiv.org/pdf/2506.23815)  

**Abstract**: The influence of Artificial Intelligence (AI), and specifically Large Language Models (LLM), on education is continuously increasing. These models are frequently used by students, giving rise to the question whether current forms of assessment are still a valid way to evaluate student performance and comprehension. The theoretical framework developed in this paper is grounded in Constructive Alignment (CA) theory and Bloom's taxonomy for defining learning objectives. We argue that AI influences learning objectives of different Bloom levels in a different way, and assessment has to be adopted accordingly. Furthermore, in line with Bloom's vision, formative and summative assessment should be aligned on whether the use of AI is permitted or not.
Although lecturers tend to agree that education and assessment need to be adapted to the presence of AI, a strong bias exists on the extent to which lecturers want to allow for AI in assessment. This bias is caused by a lecturer's familiarity with AI and specifically whether they use it themselves. To avoid this bias, we propose structured guidelines on a university or faculty level, to foster alignment among the staff. Besides that, we argue that teaching staff should be trained on the capabilities and limitations of AI tools. In this way, they are better able to adapt their assessment methods. 

---
# Mamba-FETrack V2: Revisiting State Space Model for Frame-Event based Visual Object Tracking 

**Authors**: Shiao Wang, Ju Huang, Qingchuan Ma, Jinfeng Gao, Chunyi Xu, Xiao Wang, Lan Chen, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23783)  

**Abstract**: Combining traditional RGB cameras with bio-inspired event cameras for robust object tracking has garnered increasing attention in recent years. However, most existing multimodal tracking algorithms depend heavily on high-complexity Vision Transformer architectures for feature extraction and fusion across modalities. This not only leads to substantial computational overhead but also limits the effectiveness of cross-modal interactions. In this paper, we propose an efficient RGB-Event object tracking framework based on the linear-complexity Vision Mamba network, termed Mamba-FETrack V2. Specifically, we first design a lightweight Prompt Generator that utilizes embedded features from each modality, together with a shared prompt pool, to dynamically generate modality-specific learnable prompt vectors. These prompts, along with the modality-specific embedded features, are then fed into a Vision Mamba-based FEMamba backbone, which facilitates prompt-guided feature extraction, cross-modal interaction, and fusion in a unified manner. Finally, the fused representations are passed to the tracking head for accurate target localization. Extensive experimental evaluations on multiple RGB-Event tracking benchmarks, including short-term COESOT dataset and long-term datasets, i.e., FE108 and FELT V2, demonstrate the superior performance and efficiency of the proposed tracking framework. The source code and pre-trained models will be released on this https URL 

---
# Calibrating Graph Neural Networks with Wavelet-Aware Temperature Scaling 

**Authors**: Xiaoyang Li, Linwei Tao, Haohui Lu, Minjing Dong, Junbin Gao, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23782)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated strong predictive performance on relational data; however, their confidence estimates often misalign with actual predictive correctness, posing significant limitations for deployment in safety-critical settings. While existing graph-aware calibration methods seek to mitigate this limitation, they primarily depend on coarse one-hop statistics, such as neighbor-predicted confidence, or latent node embeddings, thereby neglecting the fine-grained structural heterogeneity inherent in graph topology. In this work, we propose Wavelet-Aware Temperature Scaling (WATS), a post-hoc calibration framework that assigns node-specific temperatures based on tunable heat-kernel graph wavelet features. Specifically, WATS harnesses the scalability and topology sensitivity of graph wavelets to refine confidence estimates, all without necessitating model retraining or access to neighboring logits or predictions. Extensive evaluations across seven benchmark datasets with varying graph structures and two GNN backbones demonstrate that WATS achieves the lowest Expected Calibration Error (ECE) among all compared methods, outperforming both classical and graph-specific baselines by up to 42.3\% in ECE and reducing calibration variance by 17.24\% on average compared with graph-specific methods. Moreover, WATS remains computationally efficient, scaling well across graphs of diverse sizes and densities. Code will be released based on publication. 

---
# Multi-Timescale Hierarchical Reinforcement Learning for Unified Behavior and Control of Autonomous Driving 

**Authors**: Guizhe Jin, Zhuoren Li, Bo Leng, Ran Yu, Lu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.23771)  

**Abstract**: Reinforcement Learning (RL) is increasingly used in autonomous driving (AD) and shows clear advantages. However, most RL-based AD methods overlook policy structure design. An RL policy that only outputs short-timescale vehicle control commands results in fluctuating driving behavior due to fluctuations in network outputs, while one that only outputs long-timescale driving goals cannot achieve unified optimality of driving behavior and control. Therefore, we propose a multi-timescale hierarchical reinforcement learning approach. Our approach adopts a hierarchical policy structure, where high- and low-level RL policies are unified-trained to produce long-timescale motion guidance and short-timescale control commands, respectively. Therein, motion guidance is explicitly represented by hybrid actions to capture multimodal driving behaviors on structured road and support incremental low-level extend-state updates. Additionally, a hierarchical safety mechanism is designed to ensure multi-timescale safety. Evaluation in simulator-based and HighD dataset-based highway multi-lane scenarios demonstrates that our approach significantly improves AD performance, effectively increasing driving efficiency, action consistency and safety. 

---
# Software Engineering for Large Language Models: Research Status, Challenges and the Road Ahead 

**Authors**: Hongzhou Rao, Yanjie Zhao, Xinyi Hou, Shenao Wang, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23762)  

**Abstract**: The rapid advancement of large language models (LLMs) has redefined artificial intelligence (AI), pushing the boundaries of AI research and enabling unbounded possibilities for both academia and the industry. However, LLM development faces increasingly complex challenges throughout its lifecycle, yet no existing research systematically explores these challenges and solutions from the perspective of software engineering (SE) approaches. To fill the gap, we systematically analyze research status throughout the LLM development lifecycle, divided into six phases: requirements engineering, dataset construction, model development and enhancement, testing and evaluation, deployment and operations, and maintenance and evolution. We then conclude by identifying the key challenges for each phase and presenting potential research directions to address these challenges. In general, we provide valuable insights from an SE perspective to facilitate future advances in LLM development. 

---
# AutoEvoEval: An Automated Framework for Evolving Close-Ended LLM Evaluation Data 

**Authors**: JiaRu Wu, Mingwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23735)  

**Abstract**: Large language models (LLMs) have shown remarkable performance on various tasks, but existing evaluation benchmarks are often static and insufficient to fully assess their robustness and generalization in realistic scenarios. Prior work using evolutionary or adversarial data augmentation has improved evaluation diversity but lacks systematic control over perturbation types and multi-step complexity, limiting comprehensive robustness analysis. To address these gaps, we propose AutoEvoEval, an evolution-based evaluation framework for close-ended tasks such as multi-choice question answering. AutoEvoEval introduces 22 interpretable atomic evolution operations and supports multi-round compositions, enabling controlled generation of diverse, challenging, and realistic test samples. We conduct extensive experiments addressing four research questions on a broad set of open- and closed-source LLMs. Our results show that atomic operations cause an average accuracy drop of 7.283\%, with structure-disrupting or misleading semantic edits causing the largest declines. Model sensitivities vary significantly for the same perturbation, and combining multiple evolution steps amplifies adversarial effects by up to 52.932\%. These findings suggest current benchmarks may overestimate true model generalization and emphasize the need for evolution-aware robustness evaluation. Code and resources are available at: this https URL. 

---
# Marker Gene Method : Identifying Stable Solutions in a Dynamic Environment 

**Authors**: Hao Shi, Xi Li, Fangfang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.23734)  

**Abstract**: Competitive Co-evolutionary Algorithms (CCEAs) are often hampered by complex dynamics like intransitivity and the Red Queen effect, leading to unstable convergence. To counter these challenges, this paper introduces the Marker Gene Method (MGM), a framework that establishes stability by using a 'marker gene' as a dynamic benchmark and an adaptive weighting mechanism to balance exploration and exploitation. We provide rigorous mathematical proofs demonstrating that MGM creates strong attractors near Nash Equilibria within the Strictly Competitive Game framework. Empirically, MGM demonstrates its efficacy across a spectrum of challenges: it stabilizes the canonical Rock-Paper-Scissors game, significantly improves the performance of C-RMOEA/D on ZDT benchmarks, and, when augmented with a Memory Pool (MP) extension, it successfully tames the notoriously pathological Shapley Biased Game. This work presents a theoretically sound and empirically validated framework that substantially enhances the stability and robustness of CCEAs in complex competitive environments. 

---
# System-Embedded Diffusion Bridge Models 

**Authors**: Bartlomiej Sobieski, Matthew Tivnan, Yuang Wang, Siyeop Yoon, Pengfei Jin, Dufan Wu, Quanzheng Li, Przemyslaw Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2506.23726)  

**Abstract**: Solving inverse problems -- recovering signals from incomplete or noisy measurements -- is fundamental in science and engineering. Score-based generative models (SGMs) have recently emerged as a powerful framework for this task. Two main paradigms have formed: unsupervised approaches that adapt pretrained generative models to inverse problems, and supervised bridge methods that train stochastic processes conditioned on paired clean and corrupted data. While the former typically assume knowledge of the measurement model, the latter have largely overlooked this structural information. We introduce System embedded Diffusion Bridge Models (SDBs), a new class of supervised bridge methods that explicitly embed the known linear measurement system into the coefficients of a matrix-valued SDE. This principled integration yields consistent improvements across diverse linear inverse problems and demonstrates robust generalization under system misspecification between training and deployment, offering a promising solution to real-world applications. 

---
# PAC Bench: Do Foundation Models Understand Prerequisites for Executing Manipulation Policies? 

**Authors**: Atharva Gundawar, Som Sagar, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2506.23725)  

**Abstract**: Vision-Language Models (VLMs) are increasingly pivotal for generalist robot manipulation, enabling tasks such as physical reasoning, policy generation, and failure detection. However, their proficiency in these high-level applications often assumes a deep understanding of low-level physical prerequisites, a capability that remains largely unverified. For robots to perform actions reliably, they must comprehend intrinsic object properties (e.g., material, weight), action affordances (e.g., graspable, stackable), and physical constraints (e.g., stability, reachability, or an object's state, such as being closed). Despite the widespread use of VLMs in manipulation tasks, we argue that off-the-shelf models may lack this granular, physically grounded understanding, as such prerequisites are often overlooked during training.
To address this critical gap, we introduce PAC Bench, a comprehensive benchmark designed to systematically evaluate VLMs on their understanding of core Properties, Affordances, and Constraints (PAC) from a task executability perspective. PAC Bench features a diverse dataset with over 30,000 annotations, comprising 673 real-world images (115 object classes, 15 property types, and 1 to 3 affordances defined per class), 100 real-world humanoid-view scenarios, and 120 unique simulated constraint scenarios across four tasks.
Our evaluations reveal significant gaps in the ability of current VLMs to grasp fundamental physical concepts, highlighting limitations in their suitability for reliable robot manipulation and pointing to key areas for targeted research. PAC Bench also serves as a standardized benchmark for rigorously evaluating physical reasoning in VLMs and guiding the development of more robust, physically grounded models for robotic applications.
Project Page: this https URL 

---
# When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation 

**Authors**: Chang'an Yi, Xiaohui Deng, Guohao Chen, Yan Zhou, Qinghua Lu, Shuaicheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23724)  

**Abstract**: Test-time Adaptation (TTA) adapts a given model to testing domain data with potential domain shifts through online unsupervised learning, yielding impressive performance. However, to date, existing TTA methods primarily focus on single-model adaptation. In this work, we investigate an intriguing question: how does cross-model knowledge influence the TTA process? Our findings reveal that, in TTA's unsupervised online setting, each model can provide complementary, confident knowledge to the others, even when there are substantial differences in model size. For instance, a smaller model like MobileViT (10.6M parameters) can effectively guide a larger model like ViT-Base (86.6M parameters). In light of this, we propose COCA, a Cross-Model Co-Learning framework for TTA, which mainly consists of two main strategies. 1) Co-adaptation adaptively integrates complementary knowledge from other models throughout the TTA process, reducing individual model biases. 2) Self-adaptation enhances each model's unique strengths via unsupervised learning, enabling diverse adaptation to the target domain. Extensive experiments show that COCA, which can also serve as a plug-and-play module, significantly boosts existing SOTAs, on models with various sizes--including ResNets, ViTs, and Mobile-ViTs--via cross-model co-learned TTA. For example, with Mobile-ViT's guidance, COCA raises ViT-Base's average adaptation accuracy on ImageNet-C from 51.7% to 64.5%. The code is publicly available at this https URL. 

---
# Deep Learning-Based Semantic Segmentation for Real-Time Kidney Imaging and Measurements with Augmented Reality-Assisted Ultrasound 

**Authors**: Gijs Luijten, Roberto Maria Scardigno, Lisle Faray de Paiva, Peter Hoyer, Jens Kleesiek, Domenico Buongiorno, Vitoantonio Bevilacqua, Jan Egger  

**Link**: [PDF](https://arxiv.org/pdf/2506.23721)  

**Abstract**: Ultrasound (US) is widely accessible and radiation-free but has a steep learning curve due to its dynamic nature and non-standard imaging planes. Additionally, the constant need to shift focus between the US screen and the patient poses a challenge. To address these issues, we integrate deep learning (DL)-based semantic segmentation for real-time (RT) automated kidney volumetric measurements, which are essential for clinical assessment but are traditionally time-consuming and prone to fatigue. This automation allows clinicians to concentrate on image interpretation rather than manual measurements. Complementing DL, augmented reality (AR) enhances the usability of US by projecting the display directly into the clinician's field of view, improving ergonomics and reducing the cognitive load associated with screen-to-patient transitions. Two AR-DL-assisted US pipelines on HoloLens-2 are proposed: one streams directly via the application programming interface for a wireless setup, while the other supports any US device with video output for broader accessibility. We evaluate RT feasibility and accuracy using the Open Kidney Dataset and open-source segmentation models (nnU-Net, Segmenter, YOLO with MedSAM and LiteMedSAM). Our open-source GitHub pipeline includes model implementations, measurement algorithms, and a Wi-Fi-based streaming solution, enhancing US training and diagnostics, especially in point-of-care settings. 

---
# DABstep: Data Agent Benchmark for Multi-step Reasoning 

**Authors**: Alex Egg, Martin Iglesias Goyanes, Friso Kingma, Andreu Mora, Leandro von Werra, Thomas Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2506.23719)  

**Abstract**: We introduce DABstep, a novel benchmark for evaluating AI agents on realistic multi-step data analysis tasks. DABstep comprises over 450 real-world challenges derived from a financial analytics platform, requiring models to combine code-based data processing with contextual reasoning over heterogeneous documentation. Each task demands an iterative, multi-step problem-solving approach, testing capabilities in data manipulation, cross-referencing multiple sources, and precise result reporting. The benchmark provides a factoid-style answer format with automatic correctness checks for objective scoring at scale. We evaluate leading LLM-based agents, revealing a substantial performance gap: even the best agent achieves only 14.55% accuracy on the hardest tasks. We detail our benchmark's design, dataset composition, task formulation, evaluation protocol, report baseline results and analyze failure modes. DABstep is released with a public leaderboard and toolkit to accelerate research in autonomous data analysis. 

---
# Towards Efficient and Accurate Spiking Neural Networks via Adaptive Bit Allocation 

**Authors**: Xingting Yao, Qinghao Hu, Fei Zhou, Tielong Liu, Gang Li, Peisong Wang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.23717)  

**Abstract**: Multi-bit spiking neural networks (SNNs) have recently become a heated research spot, pursuing energy-efficient and high-accurate AI. However, with more bits involved, the associated memory and computation demands escalate to the point where the performance improvements become disproportionate. Based on the insight that different layers demonstrate different importance and extra bits could be wasted and interfering, this paper presents an adaptive bit allocation strategy for direct-trained SNNs, achieving fine-grained layer-wise allocation of memory and computation resources. Thus, SNN's efficiency and accuracy can be improved. Specifically, we parametrize the temporal lengths and the bit widths of weights and spikes, and make them learnable and controllable through gradients. To address the challenges caused by changeable bit widths and temporal lengths, we propose the refined spiking neuron, which can handle different temporal lengths, enable the derivation of gradients for temporal lengths, and suit spike quantization better. In addition, we theoretically formulate the step-size mismatch problem of learnable bit widths, which may incur severe quantization errors to SNN, and accordingly propose the step-size renewal mechanism to alleviate this issue. Experiments on various datasets, including the static CIFAR and ImageNet and the dynamic CIFAR-DVS and DVS-GESTURE, demonstrate that our methods can reduce the overall memory and computation cost while achieving higher accuracy. Particularly, our SEWResNet-34 can achieve a 2.69\% accuracy gain and 4.16$\times$ lower bit budgets over the advanced baseline work on ImageNet. This work will be fully open-sourced. 

---
# Learning Modular Exponentiation with Transformers 

**Authors**: David Demitri Africa, Sara M. Kapoor, Theo Simon Sorg  

**Link**: [PDF](https://arxiv.org/pdf/2506.23679)  

**Abstract**: Modular exponentiation is crucial to number theory and cryptography, yet remains largely unexplored from a mechanistic interpretability standpoint. We train a 4-layer encoder-decoder Transformer model to perform this operation and investigate the emergence of numerical reasoning during training. Utilizing principled sampling strategies, PCA-based embedding analysis, and activation patching, we examine how number-theoretic properties are encoded within the model. We find that reciprocal operand training leads to strong performance gains, with sudden generalization across related moduli. These synchronized accuracy surges reflect grokking-like dynamics, suggesting the model internalizes shared arithmetic structure. We also find a subgraph consisting entirely of attention heads in the final layer sufficient to achieve full performance on the task of regular exponentiation. These results suggest that transformer models learn modular arithmetic through specialized computational circuits, paving the way for more interpretable and efficient neural approaches to modular exponentiation. 

---
# Interactive Reasoning: Visualizing and Controlling Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Rock Yuren Pang, K. J. Kevin Feng, Shangbin Feng, Chu Li, Weijia Shi, Yulia Tsvetkov, Jeffrey Heer, Katharina Reinecke  

**Link**: [PDF](https://arxiv.org/pdf/2506.23678)  

**Abstract**: The output quality of large language models (LLMs) can be improved via "reasoning": generating segments of chain-of-thought (CoT) content to further condition the model prior to producing user-facing output. While these chains contain valuable information, they are verbose and lack explicit organization, making them tedious to review. Moreover, they lack opportunities for user feedback, such as to remove unwanted considerations, add desired ones, or clarify unclear assumptions. We introduce Interactive Reasoning, an interaction design that visualizes chain-of-thought outputs as a hierarchy of topics and enables user review and modification. We implement interactive reasoning in Hippo, a prototype for AI-assisted decision making in the face of uncertain trade-offs. In a user study with 16 participants, we find that interactive reasoning in Hippo allows users to quickly identify and interrupt erroneous generations, efficiently steer the model towards customized responses, and better understand both model reasoning and model outputs. Our work contributes to a new paradigm that incorporates user oversight into LLM reasoning processes. 

---
# QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration 

**Authors**: Junze Hu, Xiangyu Jin, Yizhe Zeng, Yuling Liu, Yunpeng Li, Dan Du, Kaiyu Xie, Hongsong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23644)  

**Abstract**: We introduce QLPro, a vulnerability detection framework that systematically integrates LLMs and static analysis tools to enable comprehensive vulnerability detection across entire open-source this http URL constructed a new dataset, JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only 24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed as 0-days. 

---
# VAP-Diffusion: Enriching Descriptions with MLLMs for Enhanced Medical Image Generation 

**Authors**: Peng Huang, Junhu Fu, Bowen Guo, Zeju Li, Yuanyuan Wang, Yi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.23641)  

**Abstract**: As the appearance of medical images is influenced by multiple underlying factors, generative models require rich attribute information beyond labels to produce realistic and diverse images. For instance, generating an image of skin lesion with specific patterns demands descriptions that go beyond diagnosis, such as shape, size, texture, and color. However, such detailed descriptions are not always accessible. To address this, we explore a framework, termed Visual Attribute Prompts (VAP)-Diffusion, to leverage external knowledge from pre-trained Multi-modal Large Language Models (MLLMs) to improve the quality and diversity of medical image generation. First, to derive descriptions from MLLMs without hallucination, we design a series of prompts following Chain-of-Thoughts for common medical imaging tasks, including dermatologic, colorectal, and chest X-ray images. Generated descriptions are utilized during training and stored across different categories. During testing, descriptions are randomly retrieved from the corresponding category for inference. Moreover, to make the generator robust to unseen combination of descriptions at the test time, we propose a Prototype Condition Mechanism that restricts test embeddings to be similar to those from training. Experiments on three common types of medical imaging across four datasets verify the effectiveness of VAP-Diffusion. 

---
# Unified Multimodal Understanding via Byte-Pair Visual Encoding 

**Authors**: Wanpeng Zhang, Yicheng Feng, Hao Luo, Yijiang Li, Zihao Yue, Sipeng Zheng, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23639)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in vision-language understanding, yet effectively aligning different modalities remains a fundamental challenge. We present a framework that unifies multimodal understanding by applying byte-pair encoding to visual tokens. Unlike conventional approaches that rely on modality-specific encoders, our method directly incorporates structural information into visual tokens, mirroring successful tokenization strategies in text-only language models. We introduce a priority-guided encoding scheme that considers both frequency and spatial consistency, coupled with a multi-stage training procedure based on curriculum-driven data composition. These enhancements enable the transformer model to better capture cross-modal relationships and reason with visual information. Comprehensive experiments demonstrate improved performance across diverse vision-language tasks. By bridging the gap between visual and textual representations, our approach contributes to the advancement of more capable and efficient multimodal foundation models. 

---
# Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for Mixture-of-Experts Large Language Model 

**Authors**: Mu-Chi Chen, Po-Hsuan Huang, Xiangrui Ke, Chia-Heng Tu, Chun Jason Xue, Shih-Hao Hung  

**Link**: [PDF](https://arxiv.org/pdf/2506.23635)  

**Abstract**: Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI) with significant advancements such as OpenAI's ChatGPT, Meta's Llama, and Databricks' DBRX. This paper addresses the cost and scalability challenges encountered when constructing private LLM systems for personal or small group services, as aimed by Apple Intelligence. A Mac Studio cluster with Apple's M2 Ultra chips is established as a cost-efficient solution to host and accelerate the pretrained DBRX model with the Mixture-of-Experts (MoE) architecture. Our performance analysis reveal that parallel execution of the model's experts across two to four machine nodes significantly reduces inference time. We find that computation time for the experts is comparable to the communication time for exchanging their outputs, emphasizing the importance of network latency over bandwidth. We also observe significant management overhead due to Apple software stack's memory management logic. Based on these findings, we develop optimization schemes to eliminate the memory management overhead. As a result, the Mac Studio cluster is 1.15 times more cost-efficient than the state-of-the-art AI supercomputer with NVIDIA H100 GPUs. In addition, we construct a performance model to estimate system performance under varying configurations, and the model provides valuable insights for designing private LLM systems. 

---
# gMBA: Expression Semantic Guided Mixed Boolean-Arithmetic Deobfuscation Using Transformer Architectures 

**Authors**: Youjeong Noh, Joon-Young Paik, Jingun Kwon, Eun-Sun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2506.23634)  

**Abstract**: Mixed Boolean-Arithmetic (MBA) obfuscation protects intellectual property by converting programs into forms that are more complex to analyze. However, MBA has been increasingly exploited by malware developers to evade detection and cause significant real-world problems. Traditional MBA deobfuscation methods often consider these expressions as part of a black box and overlook their internal semantic information. To bridge this gap, we propose a truth table, which is an automatically constructed semantic representation of an expression's behavior that does not rely on external resources. The truth table is a mathematical form that represents the output of expression for all possible combinations of input. We also propose a general and extensible guided MBA deobfuscation framework (gMBA) that modifies a Transformer-based neural encoder-decoder Seq2Seq architecture to incorporate this semantic guidance. Experimental results and in-depth analysis show that integrating expression semantics significantly improves performance and highlights the importance of internal semantic expressions in recovering obfuscated code to its original form. 

---
# A Nonlinear Low-rank Representation Model with Convolutional Neural Network for Imputing Water Quality Data 

**Authors**: Xin Liao, Bing Yang, Cai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23629)  

**Abstract**: The integrity of Water Quality Data (WQD) is critical in environmental monitoring for scientific decision-making and ecological protection. However, water quality monitoring systems are often challenged by large amounts of missing data due to unavoidable problems such as sensor failures and communication delays, which further lead to water quality data becoming High-Dimensional and Sparse (HDS). Traditional data imputation methods are difficult to depict the potential dynamics and fail to capture the deep data features, resulting in unsatisfactory imputation performance. To effectively address the above issues, this paper proposes a Nonlinear Low-rank Representation model (NLR) with Convolutional Neural Networks (CNN) for imputing missing WQD, which utilizes CNNs to implement two ideas: a) fusing temporal features to model the temporal dependence of data between time slots, and b) Extracting nonlinear interactions and local patterns to mine higher-order relationships features and achieve deep fusion of multidimensional information. Experimental studies on three real water quality datasets demonstrate that the proposed model significantly outperforms existing state-of-the-art data imputation models in terms of estimation accuracy. It provides an effective approach for handling water quality monitoring data in complex dynamic environments. 

---
# The Kubernetes Network Driver Model: A Composable Architecture for High-Performance Networking 

**Authors**: Antonio Ojea  

**Link**: [PDF](https://arxiv.org/pdf/2506.23628)  

**Abstract**: Traditional Kubernetes networking struggles to meet the escalating demands of AI/ML and evolving Telco infrastructure. This paper introduces Kubernetes Network Drivers (KNDs), a transformative, modular, and declarative architecture designed to overcome current imperative provisioning and API limitations. KNDs integrate network resource management into Kubernetes' core by utilizing Dynamic Resource Allocation (DRA), Node Resource Interface (NRI) improvements, and upcoming OCI Runtime Specification changes. Our DraNet implementation demonstrates declarative attachment of network interfaces, including Remote Direct Memory Access (RDMA) devices, significantly boosting high-performance AI/ML workloads. This capability enables sophisticated cloud-native applications and lays crucial groundwork for future Telco solutions, fostering a "galaxy" of specialized KNDs for enhanced application delivery and reduced operational complexity. 

---
# AI-Generated Lecture Slides for Improving Slide Element Detection and Retrieval 

**Authors**: Suyash Maniyar, Vishvesh Trivedi, Ajoy Mondal, Anand Mishra, C.V. Jawahar  

**Link**: [PDF](https://arxiv.org/pdf/2506.23605)  

**Abstract**: Lecture slide element detection and retrieval are key problems in slide understanding. Training effective models for these tasks often depends on extensive manual annotation. However, annotating large volumes of lecture slides for supervised training is labor intensive and requires domain expertise. To address this, we propose a large language model (LLM)-guided synthetic lecture slide generation pipeline, SynLecSlideGen, which produces high-quality, coherent and realistic slides. We also create an evaluation benchmark, namely RealSlide by manually annotating 1,050 real lecture slides. To assess the utility of our synthetic slides, we perform few-shot transfer learning on real data using models pre-trained on them. Experimental results show that few-shot transfer learning with pretraining on synthetic slides significantly improves performance compared to training only on real data. This demonstrates that synthetic data can effectively compensate for limited labeled lecture slides. The code and resources of our work are publicly available on our project website: this https URL. 

---
# SoK: Semantic Privacy in Large Language Models 

**Authors**: Baihe Ma, Yanna Jiang, Xu Wang, Guangshen Yu, Qin Wang, Caijun Sun, Chen Li, Xuelei Qi, Ying He, Wei Ni, Ren Ping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23603)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains, traditional data privacy measures prove inadequate for protecting information that is implicit, contextual, or inferable - what we define as semantic privacy. This Systematization of Knowledge (SoK) introduces a lifecycle-centric framework to analyze how semantic privacy risks emerge across input processing, pretraining, fine-tuning, and alignment stages of LLMs. We categorize key attack vectors and assess how current defenses, such as differential privacy, embedding encryption, edge computing, and unlearning, address these threats. Our analysis reveals critical gaps in semantic-level protection, especially against contextual inference and latent representation leakage. We conclude by outlining open challenges, including quantifying semantic leakage, protecting multimodal inputs, balancing de-identification with generation quality, and ensuring transparency in privacy enforcement. This work aims to inform future research on designing robust, semantically aware privacy-preserving techniques for LLMs. 

---
# Semantic-guided Diverse Decoding for Large Language Model 

**Authors**: Weijie Shi, Yue Cui, Yaguang Wu, Jingzhi Fang, Shibo Zhang, Mengze Li, Sirui Han, Jia Zhu, Jiajie Xu, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23601)  

**Abstract**: Diverse decoding of large language models is crucial for applications requiring multiple semantically distinct responses, yet existing methods primarily achieve lexical rather than semantic diversity. This limitation significantly constrains Best-of-N strategies, group-based reinforcement learning, and data synthesis. While temperature sampling and diverse beam search modify token distributions or apply n-gram penalties, they fail to ensure meaningful semantic differentiation. We introduce Semantic-guided Diverse Decoding (SemDiD), operating directly in embedding space that balances quality with diversity through three complementary mechanisms: orthogonal directional guidance, dynamic inter-group repulsion, and position-debiased probability assessment. SemDiD harmonizes these competing objectives using adaptive gain functions and constraint optimization, ensuring both quality thresholds and maximal semantic differentiation. Experiments show SemDiD consistently outperforms existing methods, improving Best-of-N coverage by 1.4-5.2% across diverse tasks and accelerating RLHF training convergence by 15% while increasing accuracy by up to 2.1%. 

---
# When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series 

**Authors**: Min-Yeong Park, Won-Jeong Lee, Seong Tae Kim, Gyeong-Moon Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.23596)  

**Abstract**: Recently, forecasting future abnormal events has emerged as an important scenario to tackle real-world necessities. However, the solution of predicting specific future time points when anomalies will occur, known as Anomaly Prediction (AP), remains under-explored. Existing methods dealing with time series data fail in AP, focusing only on immediate anomalies or failing to provide precise predictions for future anomalies. To address the AP task, we propose a novel framework called Anomaly to Prompt (A2P), comprised of Anomaly-Aware Forecasting (AAF) and Synthetic Anomaly Prompting (SAP). To enable the forecasting model to forecast abnormal time points, we adopt a strategy to learn the relationships of anomalies. For the robust detection of anomalies, our proposed SAP introduces a learnable Anomaly Prompt Pool (APP) that simulates diverse anomaly patterns using signal adaptive prompt. Comprehensive experiments on multiple real-world datasets demonstrate the superiority of A2P over state-of-the-art methods, showcasing its ability to predict future anomalies. Our implementation code is available at this https URL. 

---
# Transition Matching: Scalable and Flexible Generative Modeling 

**Authors**: Neta Shaul, Uriel Singer, Itai Gat, Yaron Lipman  

**Link**: [PDF](https://arxiv.org/pdf/2506.23589)  

**Abstract**: Diffusion and flow matching models have significantly advanced media generation, yet their design space is well-explored, somewhat limiting further improvements. Concurrently, autoregressive (AR) models, particularly those generating continuous tokens, have emerged as a promising direction for unifying text and media generation. This paper introduces Transition Matching (TM), a novel discrete-time, continuous-state generative paradigm that unifies and advances both diffusion/flow models and continuous AR generation. TM decomposes complex generation tasks into simpler Markov transitions, allowing for expressive non-deterministic probability transition kernels and arbitrary non-continuous supervision processes, thereby unlocking new flexible design avenues. We explore these choices through three TM variants: (i) Difference Transition Matching (DTM), which generalizes flow matching to discrete-time by directly learning transition probabilities, yielding state-of-the-art image quality and text adherence as well as improved sampling efficiency. (ii) Autoregressive Transition Matching (ARTM) and (iii) Full History Transition Matching (FHTM) are partially and fully causal models, respectively, that generalize continuous AR methods. They achieve continuous causal AR generation quality comparable to non-causal approaches and potentially enable seamless integration with existing AR text generation techniques. Notably, FHTM is the first fully causal model to match or surpass the performance of flow-based methods on text-to-image task in continuous domains. We demonstrate these contributions through a rigorous large-scale comparison of TM variants and relevant baselines, maintaining a fixed architecture, training data, and hyperparameters. 

---
# A Clinically-Grounded Two-Stage Framework for Renal CT Report Generation 

**Authors**: Renjie Liang, Zhengkang Fan, Jinqian Pan, Chenkun Sun, Russell Terry, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23584)  

**Abstract**: Generating radiology reports from CT scans remains a complex task due to the nuanced nature of medical imaging and the variability in clinical documentation. In this study, we propose a two-stage framework for generating renal radiology reports from 2D CT slices. First, we extract structured abnormality features using a multi-task learning model trained to identify lesion attributes such as location, size, enhancement, and attenuation. These extracted features are subsequently combined with the corresponding CT image and fed into a fine-tuned vision-language model to generate natural language report sentences aligned with clinical findings. We conduct experiments on a curated dataset of renal CT studies with manually annotated sentence-slice-feature triplets and evaluate performance using both classification metrics and natural language generation metrics. Our results demonstrate that the proposed model outperforms random baselines across all abnormality types, and the generated reports capture key clinical content with reasonable textual accuracy. This exploratory work highlights the feasibility of modular, feature-informed report generation for renal imaging. Future efforts will focus on extending this pipeline to 3D CT volumes and further improving clinical fidelity in multimodal medical AI systems. 

---
# PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection 

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23581)  

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack. 

---
# Online Human Action Detection during Escorting 

**Authors**: Siddhartha Mondal, Avik Mitra, Chayan Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.23573)  

**Abstract**: The deployment of robot assistants in large indoor spaces has seen significant growth, with escorting tasks becoming a key application. However, most current escorting robots primarily rely on navigation-focused strategies, assuming that the person being escorted will follow without issue. In crowded environments, this assumption often falls short, as individuals may struggle to keep pace, become obstructed, get distracted, or need to stop unexpectedly. As a result, conventional robotic systems are often unable to provide effective escorting services due to their limited understanding of human movement dynamics. To address these challenges, an effective escorting robot must continuously detect and interpret human actions during the escorting process and adjust its movement accordingly. However, there is currently no existing dataset designed specifically for human action detection in the context of escorting. Given that escorting often occurs in crowded environments, where other individuals may enter the robot's camera view, the robot also needs to identify the specific human it is escorting (the subject) before predicting their actions. Since no existing model performs both person re-identification and action prediction in real-time, we propose a novel neural network architecture that can accomplish both tasks. This enables the robot to adjust its speed dynamically based on the escortee's movements and seamlessly resume escorting after any disruption. In comparative evaluations against strong baselines, our system demonstrates superior efficiency and effectiveness, showcasing its potential to significantly improve robotic escorting services in complex, real-world scenarios. 

---
# Tensor Train Quantum State Tomography using Compressed Sensing 

**Authors**: Shakir Showkat Sofi, Charlotte Vermeylen, Lieven De Lathauwer  

**Link**: [PDF](https://arxiv.org/pdf/2506.23560)  

**Abstract**: Quantum state tomography (QST) is a fundamental technique for estimating the state of a quantum system from measured data and plays a crucial role in evaluating the performance of quantum devices. However, standard estimation methods become impractical due to the exponential growth of parameters in the state representation. In this work, we address this challenge by parameterizing the state using a low-rank block tensor train decomposition and demonstrate that our approach is both memory- and computationally efficient. This framework applies to a broad class of quantum states that can be well approximated by low-rank decompositions, including pure states, nearly pure states, and ground states of Hamiltonians. 

---
# Uncertainty-aware Diffusion and Reinforcement Learning for Joint Plane Localization and Anomaly Diagnosis in 3D Ultrasound 

**Authors**: Yuhao Huang, Yueyue Xu, Haoran Dou, Jiaxiao Deng, Xin Yang, Hongyu Zheng, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.23538)  

**Abstract**: Congenital uterine anomalies (CUAs) can lead to infertility, miscarriage, preterm birth, and an increased risk of pregnancy complications. Compared to traditional 2D ultrasound (US), 3D US can reconstruct the coronal plane, providing a clear visualization of the uterine morphology for assessing CUAs accurately. In this paper, we propose an intelligent system for simultaneous automated plane localization and CUA diagnosis. Our highlights are: 1) we develop a denoising diffusion model with local (plane) and global (volume/text) guidance, using an adaptive weighting strategy to optimize attention allocation to different conditions; 2) we introduce a reinforcement learning-based framework with unsupervised rewards to extract the key slice summary from redundant sequences, fully integrating information across multiple planes to reduce learning difficulty; 3) we provide text-driven uncertainty modeling for coarse prediction, and leverage it to adjust the classification probability for overall performance improvement. Extensive experiments on a large 3D uterine US dataset show the efficacy of our method, in terms of plane localization and CUA diagnosis. Code is available at this https URL. 

---
# NEU-ESC: A Comprehensive Vietnamese dataset for Educational Sentiment analysis and topic Classification toward multitask learning 

**Authors**: Phan Quoc Hung Mai, Quang Hung Nguyen, Phuong Giang Duong, Hong Hanh Nguyen, Nguyen Tuan Long  

**Link**: [PDF](https://arxiv.org/pdf/2506.23524)  

**Abstract**: In the field of education, understanding students' opinions through their comments is crucial, especially in the Vietnamese language, where resources remain limited. Existing educational datasets often lack domain relevance and student slang. To address these gaps, we introduce NEU-ESC, a new Vietnamese dataset for Educational Sentiment Classification and Topic Classification, curated from university forums, which offers more samples, richer class diversity, longer texts, and broader vocabulary. In addition, we explore multitask learning using encoder-only language models (BERT), in which we showed that it achieves performance up to 83.7% and 79.8% accuracy for sentiment and topic classification tasks. We also benchmark our dataset and model with other datasets and models, including Large Language Models, and discuss these benchmarks. The dataset is publicly available at: this https URL. 

---
# FedWSQ: Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization 

**Authors**: Seung-Wook Kim, Seongyeol Kim, Jiah Kim, Seowon Ji, Se-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.23516)  

**Abstract**: Federated learning (FL) often suffers from performance degradation due to key challenges such as data heterogeneity and communication constraints. To address these limitations, we present a novel FL framework called FedWSQ, which integrates weight standardization (WS) and the proposed distribution-aware non-uniform quantization (DANUQ). WS enhances FL performance by filtering out biased components in local updates during training, thereby improving the robustness of the model against data heterogeneity and unstable client participation. In addition, DANUQ minimizes quantization errors by leveraging the statistical properties of local model updates. As a result, FedWSQ significantly reduces communication overhead while maintaining superior model accuracy. Extensive experiments on FL benchmark datasets demonstrate that FedWSQ consistently outperforms existing FL methods across various challenging FL settings, including extreme data heterogeneity and ultra-low-bit communication scenarios. 

---
# MGPRL: Distributed Multi-Gaussian Processes for Wi-Fi-based Multi-Robot Relative Localization in Large Indoor Environments 

**Authors**: Sai Krishna Ghanta, Ramviyas Parasuraman  

**Link**: [PDF](https://arxiv.org/pdf/2506.23514)  

**Abstract**: Relative localization is a crucial capability for multi-robot systems operating in GPS-denied environments. Existing approaches for multi-robot relative localization often depend on costly or short-range sensors like cameras and LiDARs. Consequently, these approaches face challenges such as high computational overhead (e.g., map merging) and difficulties in disjoint environments. To address this limitation, this paper introduces MGPRL, a novel distributed framework for multi-robot relative localization using convex-hull of multiple Wi-Fi access points (AP). To accomplish this, we employ co-regionalized multi-output Gaussian Processes for efficient Radio Signal Strength Indicator (RSSI) field prediction and perform uncertainty-aware multi-AP localization, which is further coupled with weighted convex hull-based alignment for robust relative pose estimation. Each robot predicts the RSSI field of the environment by an online scan of APs in its environment, which are utilized for position estimation of multiple APs. To perform relative localization, each robot aligns the convex hull of its predicted AP locations with that of the neighbor robots. This approach is well-suited for devices with limited computational resources and operates solely on widely available Wi-Fi RSSI measurements without necessitating any dedicated pre-calibration or offline fingerprinting. We rigorously evaluate the performance of the proposed MGPRL in ROS simulations and demonstrate it with real-world experiments, comparing it against multiple state-of-the-art approaches. The results showcase that MGPRL outperforms existing methods in terms of localization accuracy and computational efficiency. Finally, we open source MGPRL as a ROS package this https URL. 

---
# Reinforcement Fine-Tuning Enables MLLMs Learning Novel Tasks Stably 

**Authors**: Zhihao Zhang, Qiaole Dong, Qi Zhang, Jun Zhao, Enyu Zhou, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Yanwei Fu, Tao Ji, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23508)  

**Abstract**: Post-training algorithms such as Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) are widely used to adapt multimodal large language models to downstream tasks. While effective at task adaptation, their impact on prior knowledge remains unclear. In this paper, we introduce jigsaw puzzles as a novel task absent from existing pretraining corpora and systematically study the behavior of SFT and RFT on an open-source multimodal model, Qwen2.5-VL. Our experiments reveal a sharp trade-off: SFT enables rapid task acquisition but leads to catastrophic forgetting, whereas RFT learns more slowly on novel tasks but maintains prior knowledge. We analyze this phenomenon through the lens of learning dynamics, showing that RFT reinforces correct samples that are naturally aligned with the base model's probability landscape, mitigating interference with prior knowledge. Moreover, supervised training on correct RFT-simulated rollouts allows SFT to preserve knowledge while rapidly learning new tasks. These findings suggest that data distribution, rather than algorithmic differences, plays a central role in forgetting, and highlight RFT's potential for stable continual learning in multimodal large language models. 

---
# Artificial Intelligence-assisted Pixel-level Lung (APL) Scoring for Fast and Accurate Quantification in Ultra-short Echo-time MRI 

**Authors**: Bowen Xin, Rohan Hickey, Tamara Blake, Jin Jin, Claire E Wainwright, Thomas Benkert, Alto Stemmer, Peter Sly, David Coman, Jason Dowling  

**Link**: [PDF](https://arxiv.org/pdf/2506.23506)  

**Abstract**: Lung magnetic resonance imaging (MRI) with ultrashort echo-time (UTE) represents a recent breakthrough in lung structure imaging, providing image resolution and quality comparable to computed tomography (CT). Due to the absence of ionising radiation, MRI is often preferred over CT in paediatric diseases such as cystic fibrosis (CF), one of the most common genetic disorders in Caucasians. To assess structural lung damage in CF imaging, CT scoring systems provide valuable quantitative insights for disease diagnosis and progression. However, few quantitative scoring systems are available in structural lung MRI (e.g., UTE-MRI). To provide fast and accurate quantification in lung MRI, we investigated the feasibility of novel Artificial intelligence-assisted Pixel-level Lung (APL) scoring for CF. APL scoring consists of 5 stages, including 1) image loading, 2) AI lung segmentation, 3) lung-bounded slice sampling, 4) pixel-level annotation, and 5) quantification and reporting. The results shows that our APL scoring took 8.2 minutes per subject, which was more than twice as fast as the previous grid-level scoring. Additionally, our pixel-level scoring was statistically more accurate (p=0.021), while strongly correlating with grid-level scoring (R=0.973, p=5.85e-9). This tool has great potential to streamline the workflow of UTE lung MRI in clinical settings, and be extended to other structural lung MRI sequences (e.g., BLADE MRI), and for other lung diseases (e.g., bronchopulmonary dysplasia). 

---
# Sample Margin-Aware Recalibration of Temperature Scaling 

**Authors**: Haolan Guo, Linwei Tao, Haoyang Luo, Minjing Dong, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23492)  

**Abstract**: Recent advances in deep learning have significantly improved predictive accuracy. However, modern neural networks remain systematically overconfident, posing risks for deployment in safety-critical scenarios. Current post-hoc calibration methods face a fundamental dilemma: global approaches like Temperature Scaling apply uniform adjustments across all samples, introducing high bias despite computational efficiency, while more expressive methods that operate on full logit distributions suffer from high variance due to noisy high-dimensional inputs and insufficient validation data. To address these challenges, we propose Sample Margin-Aware Recalibration of Temperature (SMART), a lightweight, data-efficient recalibration method that precisely scales logits based on the margin between the top two logits -- termed the logit gap. Specifically, the logit gap serves as a denoised, scalar signal directly tied to decision boundary uncertainty, providing a robust indicator that avoids the noise inherent in high-dimensional logit spaces while preserving model prediction invariance. Meanwhile, SMART employs a novel soft-binned Expected Calibration Error (SoftECE) objective that balances model bias and variance through adaptive binning, enabling stable parameter updates even with extremely limited calibration data. Extensive evaluations across diverse datasets and architectures demonstrate that SMART achieves state-of-the-art calibration performance even with substantially fewer parameters compared to existing parametric methods, offering a principled, robust, and highly efficient solution for practical uncertainty quantification in neural network predictions. The source code is available at: this https URL. 

---
# Qwen-GUI-3B: A Lightweight Vision-Language Model for Cross-Resolution GUI Grounding 

**Authors**: ZongHan Hsieh, Tzer-Jen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.23491)  

**Abstract**: This paper introduces Qwen-GUI-3B, a lightweight Vision-Language Model (VLM) specifically designed for Graphical User Interface grounding tasks, achieving performance competitive with significantly larger models. Unlike large-scale VLMs (>7B parameters) that are computationally intensive and impractical for consumer-grade hardware, Qwen-GUI-3B delivers strong grounding accuracy while being fully trainable on a single GPU (RTX 4090). The model incorporates several key innovations: (i) combine cross-platform, multi-resolution dataset of 24K examples from diverse sources including mobile, desktop, and web GUI screenshots to effectively address data scarcity in high-resolution desktop environments; (ii) a two-stage fine-tuning strategy, where initial cross-platform training establishes robust GUI understanding, followed by specialized fine-tuning on high-resolution data to significantly enhance model adaptability; and (iii) data curation and redundancy reduction strategies, demonstrating that randomly sampling a smaller subset with reduced redundancy achieves performance comparable to larger datasets, emphasizing data diversity over sheer volume. Empirical evaluation on standard GUI grounding benchmarks-including ScreenSpot, ScreenSpot-v2, and the challenging ScreenSpot-Pro, highlights Qwen-GUI-3B's exceptional accuracy, achieving 84.9% on ScreenSpot and 86.4% on ScreenSpot-v2, surpassing prior models under 4B parameters. Ablation studies validate the critical role of balanced sampling and two-stage fine-tuning in enhancing robustness, particularly in high-resolution desktop scenarios. The Qwen-GUI-3B is available at: this https URL 

---
# UltraTwin: Towards Cardiac Anatomical Twin Generation from Multi-view 2D Ultrasound 

**Authors**: Junxuan Yu, Yaofei Duan, Yuhao Huang, Yu Wang, Rongbo Ling, Weihao Luo, Ang Zhang, Jingxian Xu, Qiongying Ni, Yongsong Zhou, Binghan Li, Haoran Dou, Liping Liu, Yanfen Chu, Feng Geng, Zhe Sheng, Zhifeng Ding, Dingxin Zhang, Rui Huang, Yuhang Zhang, Xiaowei Xu, Tao Tan, Dong Ni, Zhongshan Gou, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23490)  

**Abstract**: Echocardiography is routine for cardiac examination. However, 2D ultrasound (US) struggles with accurate metric calculation and direct observation of 3D cardiac structures. Moreover, 3D US is limited by low resolution, small field of view and scarce availability in practice. Constructing the cardiac anatomical twin from 2D images is promising to provide precise treatment planning and clinical quantification. However, it remains challenging due to the rare paired data, complex structures, and US noises. In this study, we introduce a novel generative framework UltraTwin, to obtain cardiac anatomical twin from sparse multi-view 2D US. Our contribution is three-fold. First, pioneered the construction of a real-world and high-quality dataset containing strictly paired multi-view 2D US and CT, and pseudo-paired data. Second, we propose a coarse-to-fine scheme to achieve hierarchical reconstruction optimization. Last, we introduce an implicit autoencoder for topology-aware constraints. Extensive experiments show that UltraTwin reconstructs high-quality anatomical twins versus strong competitors. We believe it advances anatomical twin modeling for potential applications in personalized cardiac care. 

---
# Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent 

**Authors**: Haocheng Yu, Yaxiong Wu, Hao Wang, Wei Guo, Yong Liu, Yawen Li, Yuyang Ye, Junping Du, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.23485)  

**Abstract**: Interactive recommendation is a typical information-seeking task that allows users to interactively express their needs through natural language and obtain personalized recommendations. Large language model-powered (LLM-powered) agents have become a new paradigm in interactive recommendations, effectively capturing users' real-time needs and enhancing personalized experiences. However, due to limited planning and generalization capabilities, existing formulations of LLM-powered interactive recommender agents struggle to effectively address diverse and complex user intents, such as intuitive, unrefined, or occasionally ambiguous requests. To tackle this challenge, we propose a novel thought-augmented interactive recommender agent system (TAIRA) that addresses complex user intents through distilled thought patterns. Specifically, TAIRA is designed as an LLM-powered multi-agent system featuring a manager agent that orchestrates recommendation tasks by decomposing user needs and planning subtasks, with its planning capacity strengthened through Thought Pattern Distillation (TPD), a thought-augmentation method that extracts high-level thoughts from the agent's and human experts' experiences. Moreover, we designed a set of user simulation schemes to generate personalized queries of different difficulties and evaluate the recommendations based on specific datasets. Through comprehensive experiments conducted across multiple datasets, TAIRA exhibits significantly enhanced performance compared to existing methods. Notably, TAIRA shows a greater advantage on more challenging tasks while generalizing effectively on novel tasks, further validating its superiority in managing complex user intents within interactive recommendation systems. The code is publicly available at:this https URL. 

---
# Sanitizing Manufacturing Dataset Labels Using Vision-Language Models 

**Authors**: Nazanin Mahjourian, Vinh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.23465)  

**Abstract**: The success of machine learning models in industrial applications is heavily dependent on the quality of the datasets used to train the models. However, large-scale datasets, specially those constructed from crowd-sourcing and web-scraping, often suffer from label noise, inconsistencies, and errors. This problem is particularly pronounced in manufacturing domains, where obtaining high-quality labels is costly and time-consuming. This paper introduces Vision-Language Sanitization and Refinement (VLSR), which is a vision-language-based framework for label sanitization and refinement in multi-label manufacturing image datasets. This method embeds both images and their associated textual labels into a shared semantic space leveraging the CLIP vision-language model. Then two key tasks are addressed in this process by computing the cosine similarity between embeddings. First, label sanitization is performed to identify irrelevant, misspelled, or semantically weak labels, and surface the most semantically aligned label for each image by comparing image-label pairs using cosine similarity between image and label embeddings. Second, the method applies density-based clustering on text embeddings, followed by iterative cluster merging, to group semantically similar labels into unified label groups. The Factorynet dataset, which includes noisy labels from both human annotations and web-scraped sources, is employed to evaluate the effectiveness of the proposed framework. Experimental results demonstrate that the VLSR framework successfully identifies problematic labels and improves label consistency. This method enables a significant reduction in label vocabulary through clustering, which ultimately enhances the dataset's quality for training robust machine learning models in industrial applications with minimal human intervention. 

---
# Can We Predict the Unpredictable? Leveraging DisasterNet-LLM for Multimodal Disaster Classification 

**Authors**: Manaswi Kulahara, Gautam Siddharth Kashyap, Nipun Joshi, Arpita Soni  

**Link**: [PDF](https://arxiv.org/pdf/2506.23462)  

**Abstract**: Effective disaster management requires timely and accurate insights, yet traditional methods struggle to integrate multimodal data such as images, weather records, and textual reports. To address this, we propose DisasterNet-LLM, a specialized Large Language Model (LLM) designed for comprehensive disaster analysis. By leveraging advanced pretraining, cross-modal attention mechanisms, and adaptive transformers, DisasterNet-LLM excels in disaster classification. Experimental results demonstrate its superiority over state-of-the-art models, achieving higher accuracy of 89.5%, an F1 score of 88.0%, AUC of 0.92%, and BERTScore of 0.88% in multimodal disaster classification tasks. 

---
# Time-variant Image Inpainting via Interactive Distribution Transition Estimation 

**Authors**: Yun Xing, Qing Guo, Xiaoguang Li, Yihao Huang, Xiaofeng Cao, Di Lin, Ivor Tsang, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.23461)  

**Abstract**: In this work, we focus on a novel and practical task, i.e., Time-vAriant iMage inPainting (TAMP). The aim of TAMP is to restore a damaged target image by leveraging the complementary information from a reference image, where both images captured the same scene but with a significant time gap in between, i.e., time-variant images. Different from conventional reference-guided image inpainting, the reference image under TAMP setup presents significant content distinction to the target image and potentially also suffers from damages. Such an application frequently happens in our daily lives to restore a damaged image by referring to another reference image, where there is no guarantee of the reference image's source and quality. In particular, our study finds that even state-of-the-art (SOTA) reference-guided image inpainting methods fail to achieve plausible results due to the chaotic image complementation. To address such an ill-posed problem, we propose a novel Interactive Distribution Transition Estimation (InDiTE) module which interactively complements the time-variant images with adaptive semantics thus facilitate the restoration of damaged regions. To further boost the performance, we propose our TAMP solution, namely Interactive Distribution Transition Estimation-driven Diffusion (InDiTE-Diff), which integrates InDiTE with SOTA diffusion model and conducts latent cross-reference during sampling. Moreover, considering the lack of benchmarks for TAMP task, we newly assembled a dataset, i.e., TAMP-Street, based on existing image and mask datasets. We conduct experiments on the TAMP-Street datasets under two different time-variant image inpainting settings, which show our method consistently outperform SOTA reference-guided image inpainting methods for solving TAMP. 

---
# From Large-scale Audio Tagging to Real-Time Explainable Emergency Vehicle Sirens Detection 

**Authors**: Stefano Giacomelli, Marco Giordano, Claudia Rinaldi, Fabio Graziosi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23437)  

**Abstract**: Accurate recognition of Emergency Vehicle (EV) sirens is critical for the integration of intelligent transportation systems, smart city monitoring systems, and autonomous driving technologies. Modern automatic solutions are limited by the lack of large scale, curated datasets and by the computational demands of state of the art sound event detection models. This work introduces E2PANNs (Efficient Emergency Pre trained Audio Neural Networks), a lightweight Convolutional Neural Network architecture derived from the PANNs framework, specifically optimized for binary EV siren detection. Leveraging our dedicated subset of AudioSet (AudioSet EV) we fine-tune and evaluate E2PANNs across multiple reference datasets and test its viability on embedded hardware. The experimental campaign includes ablation studies, cross-domain benchmarking, and real-time inference deployment on edge device. Interpretability analyses exploiting Guided Backpropagation and ScoreCAM algorithms provide insights into the model internal representations and validate its ability to capture distinct spectrotemporal patterns associated with different types of EV sirens. Real time performance is assessed through frame wise and event based detection metrics, as well as a detailed analysis of false positive activations. Results demonstrate that E2PANNs establish a new state of the art in this research domain, with high computational efficiency, and suitability for edge-based audio monitoring and safety-critical applications. 

---
# Pipelined Decoder for Efficient Context-Aware Text Generation 

**Authors**: Zixian Huang, Chenxu Niu, Yu Gu, Gengyang Xiao, Xinwei Huang, Gong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.23431)  

**Abstract**: As the basis of generative AI, an autoregressive model requires the generation of a new token depending on all the previously generated tokens, which brings high quality but also restricts the model to generate tokens one by one, forming a bottleneck limiting the generation speed. In this paper, we propose a new decoder architecture that efficiently generates text in parallel for context-aware generation tasks. Our proposed pipelined decoder initiates the generation of multiple subsequences simultaneously, and, at each time-step, it generates a new token for each subsequence to realize parallelism. Experiments on multiple text generation tasks, including question answering, text summarization, and keyphrase generation, show that our pipelined decoder significantly improves the generation speed without a significant loss of generation quality or additional memory consumption. 

---
# Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting 

**Authors**: Heitor R. Medeiros, Hossein Sharifi-Noghabi, Gabriel L. Oliveira, Saghar Irandoust  

**Link**: [PDF](https://arxiv.org/pdf/2506.23424)  

**Abstract**: Real-world time series often exhibit a non-stationary nature, degrading the performance of pre-trained forecasting models. Test-Time Adaptation (TTA) addresses this by adjusting models during inference, but existing methods typically update the full model, increasing memory and compute costs. We propose PETSA, a parameter-efficient method that adapts forecasters at test time by only updating small calibration modules on the input and output. PETSA uses low-rank adapters and dynamic gating to adjust representations without retraining. To maintain accuracy despite limited adaptation capacity, we introduce a specialized loss combining three components: (1) a robust term, (2) a frequency-domain term to preserve periodicity, and (3) a patch-wise structural term for structural alignment. PETSA improves the adaptability of various forecasting backbones while requiring fewer parameters than baselines. Experimental results on benchmark datasets show that PETSA achieves competitive or better performance across all horizons. Our code is available at: this https URL 

---
# TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs 

**Authors**: Felipe Nuti, Tim Franzmeyer, João Henriques  

**Link**: [PDF](https://arxiv.org/pdf/2506.23423)  

**Abstract**: Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa. 

---
# BenchMake: Turn any scientific data set into a reproducible benchmark 

**Authors**: Amanda S Barnard  

**Link**: [PDF](https://arxiv.org/pdf/2506.23419)  

**Abstract**: Benchmark data sets are a cornerstone of machine learning development and applications, ensuring new methods are robust, reliable and competitive. The relative rarity of benchmark sets in computational science, due to the uniqueness of the problems and the pace of change in the associated domains, makes evaluating new innovations difficult for computational scientists. In this paper a new tool is developed and tested to potentially turn any of the increasing numbers of scientific data sets made openly available into a benchmark accessible to the community. BenchMake uses non-negative matrix factorisation to deterministically identify and isolate challenging edge cases on the convex hull (the smallest convex set that contains all existing data instances) and partitions a required fraction of matched data instances into a testing set that maximises divergence and statistical significance, across tabular, graph, image, signal and textual modalities. BenchMake splits are compared to establish splits and random splits using ten publicly available benchmark sets from different areas of science, with different sizes, shapes, distributions. 

---
# Teaching a Language Model to Speak the Language of Tools 

**Authors**: Simeon Emanuilov  

**Link**: [PDF](https://arxiv.org/pdf/2506.23394)  

**Abstract**: External tool integration through function-calling is essential for practical language model applications, yet most multilingual models lack reliable tool-use capabilities in non-English languages. Even state-of-the-art multilingual models struggle with determining when to use tools and generating the structured outputs required for function calls, often exhibiting language confusion when prompted in lower-resource languages. This work presents a methodology for adapting existing language models to enable robust tool use in any target language, using Bulgarian as a case study. The approach involves continued training of the BgGPT model series (2.6B, 9B, 27B parameters) on a novel bilingual dataset of 10,035 function-calling examples designed to support standardized protocols like MCP (Model Context Protocol). The research introduces TUCAN (Tool-Using Capable Assistant Navigator), which achieves up to 28.75% improvement in function-calling accuracy over base models while preserving core language understanding, as verified on established Bulgarian benchmarks. Beyond accuracy gains, TUCAN models demonstrate production-ready response formatting with clean, parsable function calls, contrasting with the verbose and inconsistent outputs of base models. The models, evaluation framework, and dataset are released to enable replication for other languages. This work demonstrates a practical approach for extending tool-augmented capabilities beyond English-centric systems. 

---
# Hierarchical Memory Organization for Wikipedia Generation 

**Authors**: Eugene J. Yu, Dawei Zhu, Yifan Song, Xiangyu Wong, Jiebin Zhang, Wenxuan Shi, Xiaoguang Li, Qun Liu, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23393)  

**Abstract**: Generating Wikipedia articles autonomously is a challenging task requiring the integration of accurate, comprehensive, and well-structured information from diverse sources. This paper introduces the Memory Organization-based Generation (MOG) framework, a novel approach to address these challenges by leveraging a hierarchical memory architecture. MOG extracts fine-grained memory units from web documents, recursively organizes them into a Wikipedia-style hierarchical structure, and uses this structure to guide the generation process. This ensures alignment between memory and the article outline, improving both informativeness and verifiability while minimizing hallucinations. Additionally, a citation module is implemented to enhance traceability by linking every generated sentence to specific memory units. Evaluations on our newly created WikiStart dataset demonstrate that MOG outperforms baseline methods in producing informative and reliable articles, making it particularly robust in real-world scenarios. 

---
# SIEDD: Shared-Implicit Encoder with Discrete Decoders 

**Authors**: Vikram Rangarajan, Shishira Maiya, Max Ehrlich, Abhinav Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2506.23382)  

**Abstract**: Implicit Neural Representations (INRs) offer exceptional fidelity for video compression by learning per-video optimized functions, but their adoption is crippled by impractically slow encoding times. Existing attempts to accelerate INR encoding often sacrifice reconstruction quality or crucial coordinate-level control essential for adaptive streaming and transcoding. We introduce SIEDD (Shared-Implicit Encoder with Discrete Decoders), a novel architecture that fundamentally accelerates INR encoding without these compromises. SIEDD first rapidly trains a shared, coordinate-based encoder on sparse anchor frames to efficiently capture global, low-frequency video features. This encoder is then frozen, enabling massively parallel training of lightweight, discrete decoders for individual frame groups, further expedited by aggressive coordinate-space sampling. This synergistic design delivers a remarkable 20-30X encoding speed-up over state-of-the-art INR codecs on HD and 4K benchmarks, while maintaining competitive reconstruction quality and compression ratios. Critically, SIEDD retains full coordinate-based control, enabling continuous resolution decoding and eliminating costly transcoding. Our approach significantly advances the practicality of high-fidelity neural video compression, demonstrating a scalable and efficient path towards real-world deployment. Our codebase is available at this https URL . 

---
# Perspective Dial: Measuring Perspective of Text and Guiding LLM Outputs 

**Authors**: Taejin Kim, Siun-Chuon Mau, Konrad Vesey  

**Link**: [PDF](https://arxiv.org/pdf/2506.23377)  

**Abstract**: Large language models (LLMs) are used in a variety of mission-critical roles. Due to the rapidly developing nature of LLMs, there is a lack of quantifiable understanding of the bias and perspective associated with LLM output. Inspired by this need, this paper considers the broader issue of perspective or viewpoint of general text and perspective control of large-language model (LLM) output. Perspective-Dial consists of two main components: a (1) metric space, dubbed Perspective Space, that enables quantitative measurements of different perspectives regarding a topic, and the use of (2) Systematic Prompt Engineering that utilizes greedy-coordinate descent to control LLM output perspective based on measurement feedback from the Perspective Space. The empirical nature of the approach allows progress to side step a principled understanding of perspective or bias -- effectively quantifying and adjusting outputs for a variety of topics. Potential applications include detection, tracking and mitigation of LLM bias, narrative detection, sense making and tracking in public discourse, and debate bot advocating given perspective. 

---
# Federated Timeline Synthesis: Scalable and Private Methodology For Model Training and Deployment 

**Authors**: Pawel Renc, Michal K. Grzeszczyk, Linglong Qian, Nassim Oufattole, Jeff Rasley, Arkadiusz Sitek  

**Link**: [PDF](https://arxiv.org/pdf/2506.23358)  

**Abstract**: We present Federated Timeline Synthesis (FTS), a novel framework for training generative foundation models across distributed timeseries data applied to electronic health records (EHR). At its core, FTS represents patient history as tokenized Patient Health Timelines (PHTs), language-agnostic sequences encoding temporal, categorical, and continuous clinical information. Each institution trains an autoregressive transformer on its local PHTs and transmits only model weights to a central server. The server uses the generators to synthesize a large corpus of trajectories and train a Global Generator (GG), enabling zero-shot inference via Monte Carlo simulation of future PHTs. We evaluate FTS on five clinically meaningful prediction tasks using MIMIC-IV data, showing that models trained on synthetic data generated by GG perform comparably to those trained on real data. FTS offers strong privacy guarantees, scalability across institutions, and extensibility to diverse prediction and simulation tasks especially in healthcare, including counterfactual inference, early warning detection, and synthetic trial design. 

---
# Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop 

**Authors**: Tianxing Chen, Kaixuan Wang, Zhaohui Yang, Yuhao Zhang, Zanxin Chen, Baijun Chen, Wanxi Dong, Ziyuan Liu, Dong Chen, Tianshuo Yang, Haibao Yu, Xiaokang Yang, Yusen Qin, Zhiqiang Xie, Yao Mu, Ping Luo, Tian Nian, Weiliang Deng, Yiheng Ge, Yibin Liu, Zixuan Li, Dehui Wang, Zhixuan Liang, Haohui Xie, Rijie Zeng, Yunfei Ge, Peiqing Cong, Guannan He, Zhaoming Han, Ruocheng Yin, Jingxiang Guo, Lunkai Lin, Tianling Xu, Hongzhe Bi, Xuewu Lin, Tianwei Lin, Shujie Luo, Keyu Li, Ziyan Zhao, Ke Fan, Heyang Xu, Bo Peng, Wenlong Gao, Dongjiang Li, Feng Jin, Hui Shen, Jinming Li, Chaowei Cui, Yuchen, Yaxin Peng, Lingdong Zeng, Wenlong Dong, Tengfei Li, Weijie Ke, Jun Chen, Erdemt Bao, Tian Lan, Tenglong Liu, Jin Yang, Huiping Zhuang, Baozhi Jia, Shuai Zhang, Zhengfeng Zou, Fangheng Guan, Tianyi Jia, Ke Zhou, Hongjiu Zhang, Yating Han, Cheng Fang, Yixian Zou, Chongyang Xu, Qinglun Zhang, Shen Cheng, Xiaohe Wang, Ping Tan, Haoqiang Fan, Shuaicheng Liu, Jiaheng Chen, Chuxuan Huang, Chengliang Lin, Kaijun Luo, Boyu Yue, Yi Liu, Jinyu Chen, Zichang Tan, Liming Deng, Shuo Xu, Zijian Cai, Shilong Yin, Hao Wang, Hongshan Liu, Tianyang Li, Long Shi, Ran Xu, Huilin Xu, Zhengquan Zhang, Congsheng Xu, Jinchang Yang, Feng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23351)  

**Abstract**: Embodied Artificial Intelligence (Embodied AI) is an emerging frontier in robotics, driven by the need for autonomous systems that can perceive, reason, and act in complex physical environments. While single-arm systems have shown strong task performance, collaborative dual-arm systems are essential for handling more intricate tasks involving rigid, deformable, and tactile-sensitive objects. To advance this goal, we launched the RoboTwin Dual-Arm Collaboration Challenge at the 2nd MEIS Workshop, CVPR 2025. Built on the RoboTwin Simulation platform (1.0 and 2.0) and the AgileX COBOT-Magic Robot platform, the competition consisted of three stages: Simulation Round 1, Simulation Round 2, and a final Real-World Round. Participants totally tackled 17 dual-arm manipulation tasks, covering rigid, deformable, and tactile-based scenarios. The challenge attracted 64 global teams and over 400 participants, producing top-performing solutions like SEM and AnchorDP3 and generating valuable insights into generalizable bimanual policy learning. This report outlines the competition setup, task design, evaluation methodology, key findings and future direction, aiming to support future research on robust and generalizable bimanual manipulation policies. The Challenge Webpage is available at this https URL. 

---
# ATGen: A Framework for Active Text Generation 

**Authors**: Akim Tsvigun, Daniil Vasilev, Ivan Tsvigun, Ivan Lysenko, Talgat Bektleuov, Aleksandr Medvedev, Uliana Vinogradova, Nikita Severin, Mikhail Mozikov, Andrey Savchenko, Rostislav Grigorev, Ramil Kuleev, Fedor Zhdanov, Artem Shelmanov, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2506.23342)  

**Abstract**: Active learning (AL) has demonstrated remarkable potential in reducing the annotation effort required for training machine learning models. However, despite the surging popularity of natural language generation (NLG) tasks in recent years, the application of AL to NLG has been limited. In this paper, we introduce Active Text Generation (ATGen) - a comprehensive framework that bridges AL with text generation tasks, enabling the application of state-of-the-art AL strategies to NLG. Our framework simplifies AL-empowered annotation in NLG tasks using both human annotators and automatic annotation agents based on large language models (LLMs). The framework supports LLMs deployed as services, such as ChatGPT and Claude, or operated on-premises. Furthermore, ATGen provides a unified platform for smooth implementation and benchmarking of novel AL strategies tailored to NLG tasks. Finally, we present evaluation results for state-of-the-art AL strategies across diverse settings and multiple text generation tasks. We show that ATGen reduces both the effort of human annotators and costs associated with API calls to LLM-based annotation agents. The code of the framework is available on GitHub under the MIT license. The video presentation is available at this http URL 

---
# VALID-Mol: a Systematic Framework for Validated LLM-Assisted Molecular Design 

**Authors**: Malikussaid, Hilal Hudan Nuha  

**Link**: [PDF](https://arxiv.org/pdf/2506.23339)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable potential for scientific discovery, but their application in domains requiring factual accuracy and domain-specific constraints remains challenging. In molecular design for drug discovery, LLMs can suggest creative molecular modifications but often produce chemically invalid or impractical structures. We present VALID-Mol, a systematic framework for integrating chemical validation with LLM-driven molecular design that increases the rate of generating valid chemical structures from 3% to 83%. Our approach combines methodical prompt engineering, automated chemical validation, and a fine-tuned domain-adapted LLM to ensure reliable generation of synthesizable molecules with improved properties. Beyond the specific implementation, we contribute a generalizable methodology for scientifically-constrained LLM applications, with quantifiable reliability improvements. Computational predictions suggest our framework can generate promising candidates for synthesis with up to 17-fold computationally predicted improvements in target affinity while maintaining synthetic accessibility. We provide a detailed analysis of our prompt engineering process, validation architecture, and fine-tuning approach, offering a reproducible blueprint for applying LLMs to other scientific domains where domain-specific validation is essential. 

---
# Federated Breast Cancer Detection Enhanced by Synthetic Ultrasound Image Augmentation 

**Authors**: Hongyi Pan, Ziliang Hong, Gorkem Durak, Ziyue Xu, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2506.23334)  

**Abstract**: Federated learning (FL) has emerged as a promising paradigm for collaboratively training deep learning models across institutions without exchanging sensitive medical data. However, its effectiveness is often hindered by limited data availability and non-independent, identically distributed data across participating clients, which can degrade model performance and generalization. To address these challenges, we propose a generative AI based data augmentation framework that integrates synthetic image sharing into the federated training process for breast cancer diagnosis via ultrasound images. Specifically, we train two simple class-specific Deep Convolutional Generative Adversarial Networks: one for benign and one for malignant lesions. We then simulate a realistic FL setting using three publicly available breast ultrasound image datasets: BUSI, BUS-BRA, and UDIAT. FedAvg and FedProx are adopted as baseline FL algorithms. Experimental results show that incorporating a suitable number of synthetic images improved the average AUC from 0.9206 to 0.9237 for FedAvg and from 0.9429 to 0.9538 for FedProx. We also note that excessive use of synthetic data reduced performance, underscoring the importance of maintaining a balanced ratio of real and synthetic samples. Our findings highlight the potential of generative AI based data augmentation to enhance FL results in the breast ultrasound image classification task. 

---
# XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs 

**Authors**: Yitian Gong, Luozhijie Jin, Ruifan Deng, Dong Zhang, Xin Zhang, Qinyuan Cheng, Zhaoye Fei, Shimin Li, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23325)  

**Abstract**: Speech codecs serve as bridges between speech signals and large language models. An ideal codec for speech language models should not only preserve acoustic information but also capture rich semantic information. However, existing speech codecs struggle to balance high-quality audio reconstruction with ease of modeling by language models. In this study, we analyze the limitations of previous codecs in balancing semantic richness and acoustic fidelity. We propose XY-Tokenizer, a novel codec that mitigates the conflict between semantic and acoustic capabilities through multi-stage, multi-task learning. Experimental results demonstrate that XY-Tokenizer achieves performance in both semantic and acoustic tasks comparable to that of state-of-the-art codecs operating at similar bitrates, even though those existing codecs typically excel in only one aspect. Specifically, XY-Tokenizer achieves strong text alignment, surpassing distillation-based semantic modeling methods such as SpeechTokenizer and Mimi, while maintaining a speaker similarity score of 0.83 between reconstructed and original audio. The reconstruction performance of XY-Tokenizer is comparable to that of BigCodec, the current state-of-the-art among acoustic-only codecs, which achieves a speaker similarity score of 0.84 at a similar bitrate. Code and models are available at this https URL. 

---
# GaussMaster: An LLM-based Database Copilot System 

**Authors**: Wei Zhou, Ji Sun, Xuanhe Zhou, Guoliang Li, Luyang Liu, Hao Wu, Tianyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23322)  

**Abstract**: In the financial industry, data is the lifeblood of operations, and DBAs shoulder significant responsibilities for SQL tuning, database deployment, diagnosis, and service repair. In recent years, both database vendors and customers have increasingly turned to autonomous database platforms in an effort to alleviate the heavy workload of DBAs. However, existing autonomous database platforms are limited in their capabilities, primarily addressing single-point issues such as NL2SQL, anomaly detection, and SQL tuning. Manual intervention remains a necessity for comprehensive database maintenance. GaussMaster aims to revolutionize this landscape by introducing an LLM-based database copilot system. This innovative solution is designed not only to assist developers in writing efficient SQL queries but also to provide comprehensive care for database services. When database instances exhibit abnormal behavior, GaussMaster is capable of orchestrating the entire maintenance process automatically. It achieves this by analyzing hundreds of metrics and logs, employing a Tree-of-thought approach to identify root causes, and invoking appropriate tools to resolve issues. We have successfully implemented GaussMaster in real-world scenarios, such as the banking industry, where it has achieved zero human intervention for over 34 database maintenance scenarios. In this paper, we present significant improvements in these tasks with code at this https URL. 

---
# Interpretable by Design: MH-AutoML for Transparent and Efficient Android Malware Detection without Compromising Performance 

**Authors**: Joner Assolin, Gabriel Canto, Diego Kreutz, Eduardo Feitosa, Hendrio Bragança, Angelo Nogueira, Vanderson Rocha  

**Link**: [PDF](https://arxiv.org/pdf/2506.23314)  

**Abstract**: Malware detection in Android systems requires both cybersecurity expertise and machine learning (ML) techniques. Automated Machine Learning (AutoML) has emerged as an approach to simplify ML development by reducing the need for specialized knowledge. However, current AutoML solutions typically operate as black-box systems with limited transparency, interpretability, and experiment traceability. To address these limitations, we present MH-AutoML, a domain-specific framework for Android malware detection. MH-AutoML automates the entire ML pipeline, including data preprocessing, feature engineering, algorithm selection, and hyperparameter tuning. The framework incorporates capabilities for interpretability, debugging, and experiment tracking that are often missing in general-purpose solutions. In this study, we compare MH-AutoML against seven established AutoML frameworks: Auto-Sklearn, AutoGluon, TPOT, HyperGBM, Auto-PyTorch, LightAutoML, and MLJAR. Results show that MH-AutoML achieves better recall rates while providing more transparency and control. The framework maintains computational efficiency comparable to other solutions, making it suitable for cybersecurity applications where both performance and explainability matter. 

---
# Securing AI Systems: A Guide to Known Attacks and Impacts 

**Authors**: Naoto Kiribuchi, Kengo Zenitani, Takayuki Semitsu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23296)  

**Abstract**: Embedded into information systems, artificial intelligence (AI) faces security threats that exploit AI-specific vulnerabilities. This paper provides an accessible overview of adversarial attacks unique to predictive and generative AI systems. We identify eleven major attack types and explicitly link attack techniques to their impacts -- including information leakage, system compromise, and resource exhaustion -- mapped to the confidentiality, integrity, and availability (CIA) security triad. We aim to equip researchers, developers, security practitioners, and policymakers, even those without specialized AI security expertise, with foundational knowledge to recognize AI-specific risks and implement effective defenses, thereby enhancing the overall security posture of AI systems. 

---
# Objective-Free Local Learning and Emergent Language Structure in Thinking Machines 

**Authors**: P. Myles Eugenio  

**Link**: [PDF](https://arxiv.org/pdf/2506.23293)  

**Abstract**: We present a neuro-symbolic framework for generative language modeling based on local, event-driven emergent learning. At its core is a hierarchical Hopfield memory chain acting as a compositional short-term memory and dynamic tokenizer (retokenizer). Rather than relying on predefined tokens or supervision, the model builds structure from scratch, learning symbol sequences as multi-scale representations. It constructs projection tensors that bind co-occurring features into hierarchical tokens, introducing redundancy (i.e an emergent gauge structure) and enabling compression of local activations into long-range dependencies. Curiously, we find that the retokenizer can filter natural language patterns from noise, generating synthetic languages with coherent internal morphology -- quantifiably the same as human language. Language is learned in a local (Hebbian) fashion, where model constraints dictate allowed emergent structure, and new information is retained in alignment with this structure. The absence of a global objective enables a form of plasticity not found in conventional language models, allowing the system to generalize beyond its initial inference class -- even without explicit data. We demonstrate that briefly activating a new neuron during inference binds distributed multi-scale token features into a symbolic embedding. These emergent embedding neurons act as long-term memory and support a key-value mechanism for compositional inference and generalization. This architecture provides a methodological foundation for studying how symbolic structure can emerge from local neural learning. It offers a new pathway for building scalable, interpretable neuro-symbolic systems -- where tokens, grammar, and reasoning arise as compressed memory traces within a Hopfield hierarchy. This approach advances the development of neuromorphic architectures for generative language models. 

---
# Not All Explanations for Deep Learning Phenomena Are Equally Valuable 

**Authors**: Alan Jeffares, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.23286)  

**Abstract**: Developing a better understanding of surprising or counterintuitive phenomena has constituted a significant portion of deep learning research in recent years. These include double descent, grokking, and the lottery ticket hypothesis -- among many others. Works in this area often develop ad hoc hypotheses attempting to explain these observed phenomena on an isolated, case-by-case basis. This position paper asserts that, in many prominent cases, there is little evidence to suggest that these phenomena appear in real-world applications and these efforts may be inefficient in driving progress in the broader field. Consequently, we argue against viewing them as isolated puzzles that require bespoke resolutions or explanations. However, despite this, we suggest that deep learning phenomena do still offer research value by providing unique settings in which we can refine our broad explanatory theories of more general deep learning principles. This position is reinforced by analyzing the research outcomes of several prominent examples of these phenomena from the recent literature. We revisit the current norms in the research community in approaching these problems and propose practical recommendations for future research, aiming to ensure that progress on deep learning phenomena is well aligned with the ultimate pragmatic goal of progress in the broader field of deep learning. 

---
# Why Settle for One? Text-to-ImageSet Generation and Evaluation 

**Authors**: Chengyou Jia, Xin Shen, Zhuohang Dang, Zhuohang Dang, Changliang Xia, Weijia Wu, Xinyu Zhang, Hangwei Qian, Ivor W.Tsang, Minnan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.23275)  

**Abstract**: Despite remarkable progress in Text-to-Image models, many real-world applications require generating coherent image sets with diverse consistency requirements. Existing consistent methods often focus on a specific domain with specific aspects of consistency, which significantly constrains their generalizability to broader applications. In this paper, we propose a more challenging problem, Text-to-ImageSet (T2IS) generation, which aims to generate sets of images that meet various consistency requirements based on user instructions. To systematically study this problem, we first introduce $\textbf{T2IS-Bench}$ with 596 diverse instructions across 26 subcategories, providing comprehensive coverage for T2IS generation. Building on this, we propose $\textbf{T2IS-Eval}$, an evaluation framework that transforms user instructions into multifaceted assessment criteria and employs effective evaluators to adaptively assess consistency fulfillment between criteria and generated sets. Subsequently, we propose $\textbf{AutoT2IS}$, a training-free framework that maximally leverages pretrained Diffusion Transformers' in-context capabilities to harmonize visual elements to satisfy both image-level prompt alignment and set-level visual consistency. Extensive experiments on T2IS-Bench reveal that diverse consistency challenges all existing methods, while our AutoT2IS significantly outperforms current generalized and even specialized approaches. Our method also demonstrates the ability to enable numerous underexplored real-world applications, confirming its substantial practical value. Visit our project in this https URL. 

---
# Predicting thinking time in Reasoning models 

**Authors**: Hans Peter Lynsgøe Raaschou-jensen, Constanza Fierro, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2506.23274)  

**Abstract**: Reasoning models that produce long, hidden chains of thought have emerged as powerful tools for complex, reasoning-intensive tasks\citep{deepseekai2025deepseekr1incentivizingreasoningcapability, openai2024openaio1card}. However, this paradigm introduces a new user experience challenge: users have little insight into how much time the model will spend reasoning before returning an answer. This unpredictability, can lead to user frustration and is likely to compound as LLMs can produce increasingly long tasks asynchronously \citep{kwa2025measuringaiabilitycomplete}. In this paper, we introduce and evaluate methods for both online and offline prediction of model "thinking time," aiming to develop a practical "progress bar for reasoning." We discuss the implications for user interaction and future research directions. 

---
# Token Activation Map to Visually Explain Multimodal LLMs 

**Authors**: Yi Li, Hualiang Wang, Xinpeng Ding, Haonan Wang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23270)  

**Abstract**: Multimodal large language models (MLLMs) are broadly empowering various fields. Despite their advancements, the explainability of MLLMs remains less explored, hindering deeper understanding, model credibility, and effective visualization. Unlike conventional vision models (e.g., CNNs, ViTs, CLIP) that produce a single output, MLLMs generate sequences of tokens progressively, where each generated token depends on the previous context. Therefore, earlier context tokens can introduce redundant activations that interfere with the explanation of later tokens beyond their original information. Existing studies often overlook this issue, but our observations reveal that these redundant correlations can significantly hurt the reliability of explanations. To address this, we propose an estimated causal inference method to mitigate the interference of context to achieve high-quality MLLM explanation, with a novel rank Gaussian filter to further reduce activation noises. We term this method Token Activation Map (TAM) to highlight the consideration of interactions between tokens. TAM also indicates that it excels at explaining multiple tokens of MLLM, which is different from the Class Activation Map (CAM) for a single prediction. Our TAM method significantly outperforms existing SoTA methods, showcasing high-quality visualization results that can be utilized for various scenarios, such as object localization, failure case analysis, video visualization, MLLMs visual comparison, and model understanding (e.g., color, shape, action, location, visual reasoning, multi-turn conversation, etc). The code is available this http URL. 

---
# From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows 

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.23260)  

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces have dramatically expanded capabilities for real-time data retrieval, complex computation, and multi-step orchestration. Yet, the explosive proliferation of plugins, connectors, and inter-agent protocols has outpaced discovery mechanisms and security practices, resulting in brittle integrations vulnerable to diverse threats. In this survey, we introduce the first unified, end-to-end threat model for LLM-agent ecosystems, spanning host-to-tool and agent-to-agent communications, formalize adversary capabilities and attacker objectives, and catalog over thirty attack techniques. Specifically, we organized the threat model into four domains: Input Manipulation (e.g., prompt injections, long-context hijacks, multimodal adversarial inputs), Model Compromise (e.g., prompt- and parameter-level backdoors, composite and encrypted multi-backdoors, poisoning strategies), System and Privacy Attacks (e.g., speculative side-channels, membership inference, retrieval poisoning, social-engineering simulations), and Protocol Vulnerabilities (e.g., exploits in Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent Network Protocol (ANP), and Agent-to-Agent (A2A) protocol). For each category, we review representative scenarios, assess real-world feasibility, and evaluate existing defenses. Building on our threat taxonomy, we identify key open challenges and future research directions, such as securing MCP deployments through dynamic trust management and cryptographic provenance tracking; designing and hardening Agentic Web Interfaces; and achieving resilience in multi-agent and federated environments. Our work provides a comprehensive reference to guide the design of robust defense mechanisms and establish best practices for resilient LLM-agent workflows. 

---
# PixelBoost: Leveraging Brownian Motion for Realistic-Image Super-Resolution 

**Authors**: Aradhana Mishra, Bumshik Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.23254)  

**Abstract**: Diffusion-model-based image super-resolution techniques often face a trade-off between realistic image generation and computational efficiency. This issue is exacerbated when inference times by decreasing sampling steps, resulting in less realistic and hazy images. To overcome this challenge, we introduce a novel diffusion model named PixelBoost that underscores the significance of embracing the stochastic nature of Brownian motion in advancing image super-resolution, resulting in a high degree of realism, particularly focusing on texture and edge definitions. By integrating controlled stochasticity into the training regimen, our proposed model avoids convergence to local optima, effectively capturing and reproducing the inherent uncertainty of image textures and patterns. Our proposed model demonstrates superior objective results in terms of learned perceptual image patch similarity (LPIPS), lightness order error (LOE), peak signal-to-noise ratio(PSNR), structural similarity index measure (SSIM), as well as visual quality. To determine the edge enhancement, we evaluated the gradient magnitude and pixel value, and our proposed model exhibited a better edge reconstruction capability. Additionally, our model demonstrates adaptive learning capabilities by effectively adjusting to Brownian noise patterns and introduces a sigmoidal noise sequencing method that simplifies training, resulting in faster inference speeds. 

---
# Aggregating Local Saliency Maps for Semi-Global Explainable Image Classification 

**Authors**: James Hinns, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2506.23247)  

**Abstract**: Deep learning dominates image classification tasks, yet understanding how models arrive at predictions remains a challenge. Much research focuses on local explanations of individual predictions, such as saliency maps, which visualise the influence of specific pixels on a model's prediction. However, reviewing many of these explanations to identify recurring patterns is infeasible, while global methods often oversimplify and miss important local behaviours. To address this, we propose Segment Attribution Tables (SATs), a method for summarising local saliency explanations into (semi-)global insights. SATs take image segments (such as "eyes" in Chihuahuas) and leverage saliency maps to quantify their influence. These segments highlight concepts the model relies on across instances and reveal spurious correlations, such as reliance on backgrounds or watermarks, even when out-of-distribution test performance sees little change. SATs can explain any classifier for which a form of saliency map can be produced, using segmentation maps that provide named segments. SATs bridge the gap between oversimplified global summaries and overly detailed local explanations, offering a practical tool for analysing and debugging image classifiers. 

---
# VolumetricSMPL: A Neural Volumetric Body Model for Efficient Interactions, Contacts, and Collisions 

**Authors**: Marko Mihajlovic, Siwei Zhang, Gen Li, Kaifeng Zhao, Lea Müller, Siyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23236)  

**Abstract**: Parametric human body models play a crucial role in computer graphics and vision, enabling applications ranging from human motion analysis to understanding human-environment interactions. Traditionally, these models use surface meshes, which pose challenges in efficiently handling interactions with other geometric entities, such as objects and scenes, typically represented as meshes or point clouds. To address this limitation, recent research has explored volumetric neural implicit body models. However, existing works are either insufficiently robust for complex human articulations or impose high computational and memory costs, limiting their widespread use. To this end, we introduce VolumetricSMPL, a neural volumetric body model that leverages Neural Blend Weights (NBW) to generate compact, yet efficient MLP decoders. Unlike prior approaches that rely on large MLPs, NBW dynamically blends a small set of learned weight matrices using predicted shape- and pose-dependent coefficients, significantly improving computational efficiency while preserving expressiveness. VolumetricSMPL outperforms prior volumetric occupancy model COAP with 10x faster inference, 6x lower GPU memory usage, enhanced accuracy, and a Signed Distance Function (SDF) for efficient and differentiable contact modeling. We demonstrate VolumetricSMPL's strengths across four challenging tasks: (1) reconstructing human-object interactions from in-the-wild images, (2) recovering human meshes in 3D scenes from egocentric views, (3) scene-constrained motion synthesis, and (4) resolving self-intersections. Our results highlight its broad applicability and significant performance and efficiency gains. 

---
# UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding 

**Authors**: Jie Feng, Shengyuan Wang, Tianhui Liu, Yanxin Xi, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23219)  

**Abstract**: Urban research involves a wide range of scenarios and tasks that require the understanding of multi-modal data. Current methods often focus on specific data types and lack a unified framework in urban field for processing them comprehensively. The recent success of multi-modal large language models (MLLMs) presents a promising opportunity to overcome this limitation. In this paper, we introduce $\textit{UrbanLLaVA}$, a multi-modal large language model designed to process these four types of data simultaneously and achieve strong performance across diverse urban tasks compared with general MLLMs. In $\textit{UrbanLLaVA}$, we first curate a diverse urban instruction dataset encompassing both single-modal and cross-modal urban data, spanning from location view to global view of urban environment. Additionally, we propose a multi-stage training framework that decouples spatial reasoning enhancement from domain knowledge learning, thereby improving the compatibility and downstream performance of $\textit{UrbanLLaVA}$ across diverse urban tasks. Finally, we also extend existing benchmark for urban research to assess the performance of MLLMs across a wide range of urban tasks. Experimental results from three cities demonstrate that $\textit{UrbanLLaVA}$ outperforms open-source and proprietary MLLMs in both single-modal tasks and complex cross-modal tasks and shows robust generalization abilities across cities. Source codes and data are openly accessible to the research community via this https URL. 

---
# FedRef: Communication-Efficient Bayesian Fine Tuning with Reference Model 

**Authors**: Taehwan Yoon, Bongjun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23210)  

**Abstract**: Federated learning(FL) is used for distributed scenarios to train artificial intelligence(AI) models while ensuring users' privacy. In federated learning scenario, the server generally never knows about users' data. This type of concept makes the AI training process efficient in terms of data privacy. However, regarding model performance, federated AI models may not sufficiently satisfy AI users' expectations. Furthermore, AI users have a wide range of different needs. It is not easy to satisfy the whole users needs. These types of issues can be addressed through AI model optimization, fine-tuning, or personalization to achieve optimal model performance. To address model optimization challenges, we propose reference model-based federated learning for optimal fine-tuning, which overcomes catastrophic forgetting in each round. This method is derived from Bayesian parameter-efficient transfer learning, which includes an optimal proximal term and enables overcoming the catastrophic forgetting issue in each round by utilizing a reference model that incorporates previous model parameters. As a result, this method achieves both high model performance and low computing cost. 

---
# Multi-Branch DNN and CRLB-Ratio-Weight Fusion for Enhanced DOA Sensing via a Massive H$^2$AD MIMO Receiver 

**Authors**: Feng Shu, Jiatong Bai, Di Wu, Wei Zhu, Bin Deng, Fuhui Zhou, Jiangzhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23203)  

**Abstract**: As a green MIMO structure, massive H$^2$AD is viewed as a potential technology for the future 6G wireless network. For such a structure, it is a challenging task to design a low-complexity and high-performance fusion of target direction values sensed by different sub-array groups with fewer use of prior knowledge. To address this issue, a lightweight Cramer-Rao lower bound (CRLB)-ratio-weight fusion (WF) method is proposed, which approximates inverse CRLB of each subarray using antenna number reciprocals to eliminate real-time CRLB computation. This reduces complexity and prior knowledge dependence while preserving fusion performance. Moreover, a multi-branch deep neural network (MBDNN) is constructed to further enhance direction-of-arrival (DOA) sensing by leveraging candidate angles from multiple subarrays. The subarray-specific branch networks are integrated with a shared regression module to effectively eliminate pseudo-solutions and fuse true angles. Simulation results show that the proposed CRLB-ratio-WF method achieves DOA sensing performance comparable to CRLB-based methods, while significantly reducing the reliance on prior knowledge. More notably, the proposed MBDNN has superior performance in low-SNR ranges. At SNR $= -15$ dB, it achieves an order-of-magnitude improvement in estimation accuracy compared to CRLB-ratio-WF method. 

---
# Score-based Diffusion Model for Unpaired Virtual Histology Staining 

**Authors**: Anran Liu, Xiaofei Wang, Jing Cai, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23184)  

**Abstract**: Hematoxylin and eosin (H&E) staining visualizes histology but lacks specificity for diagnostic markers. Immunohistochemistry (IHC) staining provides protein-targeted staining but is restricted by tissue availability and antibody specificity. Virtual staining, i.e., computationally translating the H&E image to its IHC counterpart while preserving the tissue structure, is promising for efficient IHC generation. Existing virtual staining methods still face key challenges: 1) effective decomposition of staining style and tissue structure, 2) controllable staining process adaptable to diverse tissue and proteins, and 3) rigorous structural consistency modelling to handle the non-pixel-aligned nature of paired H&E and IHC images. This study proposes a mutual-information (MI)-guided score-based diffusion model for unpaired virtual staining. Specifically, we design 1) a global MI-guided energy function that disentangles the tissue structure and staining characteristics across modalities, 2) a novel timestep-customized reverse diffusion process for precise control of the staining intensity and structural reconstruction, and 3) a local MI-driven contrastive learning strategy to ensure the cellular level structural consistency between H&E-IHC images. Extensive experiments demonstrate the our superiority over state-of-the-art approaches, highlighting its biomedical potential. Codes will be open-sourced upon acceptance. 

---
# Data Can Speak for Itself: Quality-guided Utilization of Wireless Synthetic Data 

**Authors**: Chen Gong, Bo Liang, Wei Gao, Chenren Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23174)  

**Abstract**: Generative models have gained significant attention for their ability to produce realistic synthetic data that supplements the quantity of real-world datasets. While recent studies show performance improvements in wireless sensing tasks by incorporating all synthetic data into training sets, the quality of synthetic data remains unpredictable and the resulting performance gains are not guaranteed. To address this gap, we propose tractable and generalizable metrics to quantify quality attributes of synthetic data - affinity and diversity. Our assessment reveals prevalent affinity limitation in current wireless synthetic data, leading to mislabeled data and degraded task performance. We attribute the quality limitation to generative models' lack of awareness of untrained conditions and domain-specific processing. To mitigate these issues, we introduce SynCheck, a quality-guided synthetic data utilization scheme that refines synthetic data quality during task model training. Our evaluation demonstrates that SynCheck consistently outperforms quality-oblivious utilization of synthetic data, and achieves 4.3% performance improvement even when the previous utilization degrades performance by 13.4%. 

---
# Deep Learning for Optical Misalignment Diagnostics in Multi-Lens Imaging Systems 

**Authors**: Tomer Slor, Dean Oren, Shira Baneth, Tom Coen, Haim Suchowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.23173)  

**Abstract**: In the rapidly evolving field of optical engineering, precise alignment of multi-lens imaging systems is critical yet challenging, as even minor misalignments can significantly degrade performance. Traditional alignment methods rely on specialized equipment and are time-consuming processes, highlighting the need for automated and scalable solutions. We present two complementary deep learning-based inverse-design methods for diagnosing misalignments in multi-element lens systems using only optical measurements. First, we use ray-traced spot diagrams to predict five-degree-of-freedom (5-DOF) errors in a 6-lens photographic prime, achieving a mean absolute error of 0.031mm in lateral translation and 0.011$^\circ$ in tilt. We also introduce a physics-based simulation pipeline that utilizes grayscale synthetic camera images, enabling a deep learning model to estimate 4-DOF, decenter and tilt errors in both two- and six-lens multi-lens systems. These results show the potential to reshape manufacturing and quality control in precision imaging. 

---
# Mode Collapse Happens: Evaluating Critical Interactions in Joint Trajectory Prediction Models 

**Authors**: Maarten Hugenholtz, Anna Meszaros, Jens Kober, Zlatan Ajanovic  

**Link**: [PDF](https://arxiv.org/pdf/2506.23164)  

**Abstract**: Autonomous Vehicle decisions rely on multimodal prediction models that account for multiple route options and the inherent uncertainty in human behavior. However, models can suffer from mode collapse, where only the most likely mode is predicted, posing significant safety risks. While existing methods employ various strategies to generate diverse predictions, they often overlook the diversity in interaction modes among agents. Additionally, traditional metrics for evaluating prediction models are dataset-dependent and do not evaluate inter-agent interactions quantitatively. To our knowledge, none of the existing metrics explicitly evaluates mode collapse. In this paper, we propose a novel evaluation framework that assesses mode collapse in joint trajectory predictions, focusing on safety-critical interactions. We introduce metrics for mode collapse, mode correctness, and coverage, emphasizing the sequential dimension of predictions. By testing four multi-agent trajectory prediction models, we demonstrate that mode collapse indeed happens. When looking at the sequential dimension, although prediction accuracy improves closer to interaction events, there are still cases where the models are unable to predict the correct interaction mode, even just before the interaction mode becomes inevitable. We hope that our framework can help researchers gain new insights and advance the development of more consistent and accurate prediction models, thus enhancing the safety of autonomous driving systems. 

---
# MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation 

**Authors**: Vladislav Bargatin, Egor Chistov, Alexander Yakovenko, Dmitriy Vatolin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23151)  

**Abstract**: Recent advances in optical flow estimation have prioritized accuracy at the cost of growing GPU memory consumption, particularly for high-resolution (FullHD) inputs. We introduce MEMFOF, a memory-efficient multi-frame optical flow method that identifies a favorable trade-off between multi-frame estimation and GPU memory usage. Notably, MEMFOF requires only 2.09 GB of GPU memory at runtime for 1080p inputs, and 28.5 GB during training, which uniquely positions our method to be trained at native 1080p without the need for cropping or downsampling. We systematically revisit design choices from RAFT-like architectures, integrating reduced correlation volumes and high-resolution training protocols alongside multi-frame estimation, to achieve state-of-the-art performance across multiple benchmarks while substantially reducing memory overhead. Our method outperforms more resource-intensive alternatives in both accuracy and runtime efficiency, validating its robustness for flow estimation at high resolutions. At the time of submission, our method ranks first on the Spring benchmark with a 1-pixel (1px) outlier rate of 3.289, leads Sintel (clean) with an endpoint error (EPE) of 0.963, and achieves the best Fl-all error on KITTI-2015 at 2.94%. The code is available at this https URL. 

---
# Benchmarking Deep Search over Heterogeneous Enterprise Data 

**Authors**: Prafulla Kumar Choubey, Xiangyu Peng, Shilpa Bhagavath, Kung-Hsiang Huang, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23139)  

**Abstract**: We present a new benchmark for evaluating Deep Search--a realistic and complex form of retrieval-augmented generation (RAG) that requires source-aware, multi-hop reasoning over diverse, sparsed, but related sources. These include documents, meeting transcripts, Slack messages, GitHub, and URLs, which vary in structure and often contain human-to-human interactions. We build it using a synthetic data pipeline that simulates business workflows across product planning, development, and support stages, generating interconnected content with realistic noise and multi-hop questions with guaranteed ground-truth answers. We release our benchmark with both answerable and unanswerable queries, and retrieval pool of 39,190 enterprise artifacts, enabling fine-grained evaluation of long-context LLM and RAG systems. Our experiments reveal that even the best-performing agentic RAG methods achieve an average performance score of 32.96 on our benchmark. With further analysis, we highlight retrieval as the main bottleneck: existing methods struggle to conduct deep searches and retrieve all necessary evidence. Consequently, they often reason over partial context, leading to significant performance degradation. 

---
# Flow-Modulated Scoring for Semantic-Aware Knowledge Graph Completion 

**Authors**: Siyuan Li, Ruitong Liu, Yan Wen, Te Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.23137)  

**Abstract**: Effective modeling of multifaceted relations is pivotal for Knowledge Graph Completion (KGC). However, a majority of existing approaches are predicated on static, embedding-based scoring, exhibiting inherent limitations in capturing contextual dependencies and relational dynamics. Addressing this gap, we propose the Flow-Modulated Scoring (FMS) framework. FMS comprises two principal components: (1) a semantic context learning module that encodes context-sensitive entity representations, and (2) a conditional flow-matching module designed to learn the dynamic transformation from a head to a tail embedding, governed by the aforementioned context. The resultant predictive vector field, representing the context-informed relational path, serves to dynamically refine the initial static score of an entity pair. Through this synergy of context-aware static representations and conditioned dynamic information, FMS facilitates a more profound modeling of relational semantics. Comprehensive evaluations on several standard benchmarks demonstrate that our proposed method surpasses prior state-of-the-art results. 

---
# Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning 

**Authors**: Zhaoye Fei, Li Ji, Siyin Wang, Junhao Shi, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23127)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they face significant challenges in embodied task planning scenarios that require continuous environmental understanding and action generation. Existing approaches generate open-loop action scripts based on static knowledge, making it difficult to learn causal relationships between actions and environmental feedback, particularly in partially observable environments. We introduce Embodied Planner-R1, a novel outcome-driven reinforcement learning framework that enables LLMs to develop interactive capabilities through autonomous exploration with minimal supervision. Our framework incorporates three key innovations: (1) Without human annotations, we employ pure reinforcement learning with group rollout, incorporating in-environment interaction through parallel exploration; (2) completion-driven sparse reward; and (3) Interactive Policy Optimization (IPO) for efficient learning from grouped trajectories. Across two challenging text-based Embodied planning benchmarks, Embodied Planner-R1 achieves impressive completion rates of 97.78% on ALFWorld and 79.92% on ScienceWorld, surpassing prior methods by a large margin, and suffers only a -3.66% drop in previously unseen environments, evidencing strong generalization. 

---
# CRISP-SAM2: SAM2 with Cross-Modal Interaction and Semantic Prompting for Multi-Organ Segmentation 

**Authors**: Xinlei Yu, Chanmiao Wang, Hui Jin, Ahmed Elazab, Gangyong Jia, Xiang Wan, Changqing Zou, Ruiquan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2506.23121)  

**Abstract**: Multi-organ medical segmentation is a crucial component of medical image processing, essential for doctors to make accurate diagnoses and develop effective treatment plans. Despite significant progress in this field, current multi-organ segmentation models often suffer from inaccurate details, dependence on geometric prompts and loss of spatial information. Addressing these challenges, we introduce a novel model named CRISP-SAM2 with CRoss-modal Interaction and Semantic Prompting based on SAM2. This model represents a promising approach to multi-organ medical segmentation guided by textual descriptions of organs. Our method begins by converting visual and textual inputs into cross-modal contextualized semantics using a progressive cross-attention interaction mechanism. These semantics are then injected into the image encoder to enhance the detailed understanding of visual information. To eliminate reliance on geometric prompts, we use a semantic prompting strategy, replacing the original prompt encoder to sharpen the perception of challenging targets. In addition, a similarity-sorting self-updating strategy for memory and a mask-refining process is applied to further adapt to medical imaging and enhance localized details. Comparative experiments conducted on seven public datasets indicate that CRISP-SAM2 outperforms existing models. Extensive analysis also demonstrates the effectiveness of our method, thereby confirming its superior performance, especially in addressing the limitations mentioned earlier. Our code is available at: this https URL\this http URL. 

---
# MoCa: Modality-aware Continual Pre-training Makes Better Bidirectional Multimodal Embeddings 

**Authors**: Haonan Chen, Hong Liu, Yuping Luo, Liang Wang, Nan Yang, Furu Wei, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23115)  

**Abstract**: Multimodal embedding models, built upon causal Vision Language Models (VLMs), have shown promise in various tasks. However, current approaches face three key limitations: the use of causal attention in VLM backbones is suboptimal for embedding tasks; scalability issues due to reliance on high-quality labeled paired data for contrastive learning; and limited diversity in training objectives and data. To address these issues, we propose MoCa, a two-stage framework for transforming pre-trained VLMs into effective bidirectional multimodal embedding models. The first stage, Modality-aware Continual Pre-training, introduces a joint reconstruction objective that simultaneously denoises interleaved text and image inputs, enhancing bidirectional context-aware reasoning. The second stage, Heterogeneous Contrastive Fine-tuning, leverages diverse, semantically rich multimodal data beyond simple image-caption pairs to enhance generalization and alignment. Our method addresses the stated limitations by introducing bidirectional attention through continual pre-training, scaling effectively with massive unlabeled datasets via joint reconstruction objectives, and utilizing diverse multimodal data for enhanced representation robustness. Experiments demonstrate that MoCa consistently improves performance across MMEB and ViDoRe-v2 benchmarks, achieving new state-of-the-art results, and exhibits strong scalability with both model size and training data on MMEB. 

---
# From Individuals to Interactions: Benchmarking Gender Bias in Multimodal Large Language Models from the Lens of Social Relationship 

**Authors**: Yue Xu, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23101)  

**Abstract**: Multimodal large language models (MLLMs) have shown impressive capabilities across tasks involving both visual and textual modalities. However, growing concerns remain about their potential to encode and amplify gender bias, particularly in socially sensitive applications. Existing benchmarks predominantly evaluate bias in isolated scenarios, overlooking how bias may emerge subtly through interpersonal interactions. We fill this gap by going beyond single-entity evaluation and instead focusing on a deeper examination of relational and contextual gender bias in dual-individual interactions. We introduce Genres, a novel benchmark designed to evaluate gender bias in MLLMs through the lens of social relationships in generated narratives. Genres assesses gender bias through a dual-character profile and narrative generation task that captures rich interpersonal dynamics and supports a fine-grained bias evaluation suite across multiple dimensions. Experiments on both open- and closed-source MLLMs reveal persistent, context-sensitive gender biases that are not evident in single-character settings. Our findings underscore the importance of relationship-aware benchmarks for diagnosing subtle, interaction-driven gender bias in MLLMs and provide actionable insights for future bias mitigation. 

---
# TOMI: Transforming and Organizing Music Ideas for Multi-Track Compositions with Full-Song Structure 

**Authors**: Qi He, Gus Xia, Ziyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23094)  

**Abstract**: Hierarchical planning is a powerful approach to model long sequences structurally. Aside from considering hierarchies in the temporal structure of music, this paper explores an even more important aspect: concept hierarchy, which involves generating music ideas, transforming them, and ultimately organizing them--across musical time and space--into a complete composition. To this end, we introduce TOMI (Transforming and Organizing Music Ideas) as a novel approach in deep music generation and develop a TOMI-based model via instruction-tuned foundation LLM. Formally, we represent a multi-track composition process via a sparse, four-dimensional space characterized by clips (short audio or MIDI segments), sections (temporal positions), tracks (instrument layers), and transformations (elaboration methods). Our model is capable of generating multi-track electronic music with full-song structure, and we further integrate the TOMI-based model with the REAPER digital audio workstation, enabling interactive human-AI co-creation. Experimental results demonstrate that our approach produces higher-quality electronic music with stronger structural coherence compared to baselines. 

---
# Enhancing Live Broadcast Engagement: A Multi-modal Approach to Short Video Recommendations Using MMGCN and User Preferences 

**Authors**: Saeid Aghasoleymani Najafabadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23085)  

**Abstract**: The purpose of this paper is to explore a multi-modal approach to enhancing live broadcast engagement by developing a short video recommendation system that incorporates Multi-modal Graph Convolutional Networks (MMGCN) with user preferences. In order to provide personalized recommendations tailored to individual interests, the proposed system takes into account user interaction data, video content features, and contextual information. With the aid of a hybrid approach combining collaborative filtering and content-based filtering techniques, the system is able to capture nuanced relationships between users, video attributes, and engagement patterns. Three datasets are used to evaluate the effectiveness of the system: Kwai, TikTok, and MovieLens. Compared to baseline models, such as DeepFM, Wide & Deep, LightGBM, and XGBoost, the proposed MMGCN-based model shows superior performance. A notable feature of the proposed model is that it outperforms all baseline methods in capturing diverse user preferences and making accurate, personalized recommendations, resulting in a Kwai F1 score of 0.574, a Tiktok F1 score of 0.506, and a MovieLens F1 score of 0.197. We emphasize the importance of multi-modal integration and user-centric approaches in advancing recommender systems, emphasizing the role they play in enhancing content discovery and audience interaction on live broadcast platforms. 

---
# Curious Causality-Seeking Agents Learn Meta Causal World 

**Authors**: Zhiyu Zhao, Haoxuan Li, Haifeng Zhang, Jun Wang, Francesco Faccio, Jürgen Schmidhuber, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23068)  

**Abstract**: When building a world model, a common assumption is that the environment has a single, unchanging underlying causal rule, like applying Newton's laws to every situation. In reality, what appears as a drifting causal mechanism is often the manifestation of a fixed underlying mechanism seen through a narrow observational window. This brings about a problem that, when building a world model, even subtle shifts in policy or environment states can alter the very observed causal mechanisms. In this work, we introduce the \textbf{Meta-Causal Graph} as world models, a minimal unified representation that efficiently encodes the transformation rules governing how causal structures shift across different latent world states. A single Meta-Causal Graph is composed of multiple causal subgraphs, each triggered by meta state, which is in the latent state space. Building on this representation, we introduce a \textbf{Causality-Seeking Agent} whose objectives are to (1) identify the meta states that trigger each subgraph, (2) discover the corresponding causal relationships by agent curiosity-driven intervention policy, and (3) iteratively refine the Meta-Causal Graph through ongoing curiosity-driven exploration and agent experiences. Experiments on both synthetic tasks and a challenging robot arm manipulation task demonstrate that our method robustly captures shifts in causal dynamics and generalizes effectively to previously unseen contexts. 

---
# Measuring How LLMs Internalize Human Psychological Concepts: A preliminary analysis 

**Authors**: Hiro Taiyo Hamada, Ippei Fujisawa, Genji Kawakita, Yuki Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2506.23055)  

**Abstract**: Large Language Models (LLMs) such as ChatGPT have shown remarkable abilities in producing human-like text. However, it is unclear how accurately these models internalize concepts that shape human thought and behavior. Here, we developed a quantitative framework to assess concept alignment between LLMs and human psychological dimensions using 43 standardized psychological questionnaires, selected for their established validity in measuring distinct psychological constructs. Our method evaluates how accurately language models reconstruct and classify questionnaire items through pairwise similarity analysis. We compared resulting cluster structures with the original categorical labels using hierarchical clustering. A GPT-4 model achieved superior classification accuracy (66.2\%), significantly outperforming GPT-3.5 (55.9\%) and BERT (48.1\%), all exceeding random baseline performance (31.9\%). We also demonstrated that the estimated semantic similarity from GPT-4 is associated with Pearson's correlation coefficients of human responses in multiple psychological questionnaires. This framework provides a novel approach to evaluate the alignment of the human-LLM concept and identify potential representational biases. Our findings demonstrate that modern LLMs can approximate human psychological constructs with measurable accuracy, offering insights for developing more interpretable AI systems. 

---
# SoMi-ToM: Evaluating Multi-Perspective Theory of Mind in Embodied Social Interactions 

**Authors**: Xianzhe Fan, Xuhui Zhou, Chuanyang Jin, Kolby Nottingham, Hao Zhu, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2506.23046)  

**Abstract**: Humans continuously infer the states, goals, and behaviors of others by perceiving their surroundings in dynamic, real-world social interactions. However, most Theory of Mind (ToM) benchmarks only evaluate static, text-based scenarios, which have a significant gap compared to real interactions. We propose the SoMi-ToM benchmark, designed to evaluate multi-perspective ToM in embodied multi-agent complex social interactions. This benchmark is based on rich multimodal interaction data generated by the interaction environment SoMi, covering diverse crafting goals and social relationships. Our framework supports multi-level evaluation: (1) first-person evaluation provides multimodal (visual, dialogue, action, etc.) input from a first-person perspective during a task for real-time state inference, (2) third-person evaluation provides complete third-person perspective video and text records after a task for goal and behavior inference. This evaluation method allows for a more comprehensive examination of a model's ToM capabilities from both the subjective immediate experience and the objective global observation. We constructed a challenging dataset containing 35 third-person perspective videos, 363 first-person perspective images, and 1225 expert-annotated multiple-choice questions (three options). On this dataset, we systematically evaluated the performance of human subjects and several state-of-the-art large vision-language models (LVLMs). The results show that LVLMs perform significantly worse than humans on SoMi-ToM: the average accuracy gap between humans and models is 40.1% in first-person evaluation and 26.4% in third-person evaluation. This indicates that future LVLMs need to further improve their ToM capabilities in embodied, complex social interactions. 

---
# Ovis-U1 Technical Report 

**Authors**: Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, Yang Li, Qing-Guo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.23044)  

**Abstract**: In this report, we introduce Ovis-U1, a 3-billion-parameter unified model that integrates multimodal understanding, text-to-image generation, and image editing capabilities. Building on the foundation of the Ovis series, Ovis-U1 incorporates a diffusion-based visual decoder paired with a bidirectional token refiner, enabling image generation tasks comparable to leading models like GPT-4o. Unlike some previous models that use a frozen MLLM for generation tasks, Ovis-U1 utilizes a new unified training approach starting from a language model. Compared to training solely on understanding or generation tasks, unified training yields better performance, demonstrating the enhancement achieved by integrating these two tasks. Ovis-U1 achieves a score of 69.6 on the OpenCompass Multi-modal Academic Benchmark, surpassing recent state-of-the-art models such as Ristretto-3B and SAIL-VL-1.5-2B. In text-to-image generation, it excels with scores of 83.72 and 0.89 on the DPG-Bench and GenEval benchmarks, respectively. For image editing, it achieves 4.00 and 6.42 on the ImgEdit-Bench and GEdit-Bench-EN, respectively. As the initial version of the Ovis unified model series, Ovis-U1 pushes the boundaries of multimodal understanding, generation, and editing. 

---
# Treatment, evidence, imitation, and chat 

**Authors**: Samuel J. Weisenthal  

**Link**: [PDF](https://arxiv.org/pdf/2506.23040)  

**Abstract**: Large language models are thought to have potential to aid in medical decision making. We investigate this here. We start with the treatment problem, the patient's core medical decision-making task, which is solved in collaboration with a healthcare provider. We discuss approaches to solving the treatment problem, including -- within evidence-based medicine -- trials and observational data. We then discuss the chat problem, and how this differs from the treatment problem -- in particular as it relates to imitation. We then discuss how a large language model might be used to solve the treatment problem and highlight some of the challenges that emerge. We finally discuss how these challenges relate to evidence-based medicine, and how this might inform next steps. 

---
# VisionScores -- A system-segmented image score dataset for deep learning tasks 

**Authors**: Alejandro Romero Amezcua, Mariano José Juan Rivera Meraz  

**Link**: [PDF](https://arxiv.org/pdf/2506.23030)  

**Abstract**: VisionScores presents a novel proposal being the first system-segmented image score dataset, aiming to offer structure-rich, high information-density images for machine and deep learning tasks. Delimited to two-handed piano pieces, it was built to consider not only certain graphic similarity but also composition patterns, as this creative process is highly instrument-dependent. It provides two scenarios in relation to composer and composition type. The first, formed by 14k samples, considers works from different authors but the same composition type, specifically, Sonatinas. The latter, consisting of 10.8K samples, presents the opposite case, various composition types from the same author, being the one selected Franz Liszt. All of the 24.8k samples are formatted as grayscale jpg images of $128 \times 512$ pixels. VisionScores supplies the users not only the formatted samples but the systems' order and pieces' metadata. Moreover, unsegmented full-page scores and the pre-formatted images are included for further analysis. 

---
# Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models 

**Authors**: Tejas Vaidhya, Ayush Kaushal, Vineet Jain, Francis Couture Harpin, Prashant Shishodia, Majid Behbahani, Yuriy Nevmyvaka, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2506.23025)  

**Abstract**: Large language models (LLMs) are increasingly used across research and industry applications, yet their inference efficiency remains a significant challenge. As the computational power of modern GPU architectures continuously improves, their memory bandwidth and capacity have not scaled proportionally, creating a critical bottleneck during inference. To address this, we investigate ternary language models (TriLMs) that employ quantization-aware training to significantly reduce memory requirements. We first analyze the scalability of TriLMs by conducting a scaling law analysis, revealing that TriLMs benefit more from increasing training data than from scaling model parameters. Based on this observation, we introduce Spectra-1.1, an open suite of TriLMs trained on up to 1.2 trillion tokens, demonstrating sustained performance gains at scale. Furthermore, to improve inference efficiency, we propose novel 2-bit and 1.6-bit packing schemes for ternary weights, which demonstrate accelerated inference across various CPU architectures. Also, building on the 2-bit packing, we develop a GPU kernel called TriRun that accelerates end-to-end model inference by up to 5 times compared to floating-point baselines. To encourage further exploration and development of TriLMs, we will release the Spectra-1.1 suite and TriRun inference kernels. Overall, our work lays the foundation for building and deploying efficient LLMs, providing a valuable resource for the research community. 

---
# BWLer: Barycentric Weight Layer Elucidates a Precision-Conditioning Tradeoff for PINNs 

**Authors**: Jerry Liu, Yasa Baig, Denise Hui Jean Lee, Rajat Vadiraj Dwaraknath, Atri Rudra, Chris Ré  

**Link**: [PDF](https://arxiv.org/pdf/2506.23024)  

**Abstract**: Physics-informed neural networks (PINNs) offer a flexible way to solve partial differential equations (PDEs) with machine learning, yet they still fall well short of the machine-precision accuracy many scientific tasks demand. In this work, we investigate whether the precision ceiling comes from the ill-conditioning of the PDEs or from the typical multi-layer perceptron (MLP) architecture. We introduce the Barycentric Weight Layer (BWLer), which models the PDE solution through barycentric polynomial interpolation. A BWLer can be added on top of an existing MLP (a BWLer-hat) or replace it completely (explicit BWLer), cleanly separating how we represent the solution from how we take derivatives for the PDE loss. Using BWLer, we identify fundamental precision limitations within the MLP: on a simple 1-D interpolation task, even MLPs with O(1e5) parameters stall around 1e-8 RMSE -- about eight orders above float64 machine precision -- before any PDE terms are added. In PDE learning, adding a BWLer lifts this ceiling and exposes a tradeoff between achievable accuracy and the conditioning of the PDE loss. For linear PDEs we fully characterize this tradeoff with an explicit error decomposition and navigate it during training with spectral derivatives and preconditioning. Across five benchmark PDEs, adding a BWLer on top of an MLP improves RMSE by up to 30x for convection, 10x for reaction, and 1800x for wave equations while remaining compatible with first-order optimizers. Replacing the MLP entirely lets an explicit BWLer reach near-machine-precision on convection, reaction, and wave problems (up to 10 billion times better than prior results) and match the performance of standard PINNs on stiff Burgers' and irregular-geometry Poisson problems. Together, these findings point to a practical path for combining the flexibility of PINNs with the precision of classical spectral solvers. 

---
# Scenario-Based Hierarchical Reinforcement Learning for Automated Driving Decision Making 

**Authors**: M. Youssef Abdelhamid, Lennart Vater, Zlatan Ajanovic  

**Link**: [PDF](https://arxiv.org/pdf/2506.23023)  

**Abstract**: Developing decision-making algorithms for highly automated driving systems remains challenging, since these systems have to operate safely in an open and complex environments. Reinforcement Learning (RL) approaches can learn comprehensive decision policies directly from experience and already show promising results in simple driving tasks. However, current approaches fail to achieve generalizability for more complex driving tasks and lack learning efficiency. Therefore, we present Scenario-based Automated Driving Reinforcement Learning (SAD-RL), the first framework that integrates Reinforcement Learning (RL) of hierarchical policy in a scenario-based environment. A high-level policy selects maneuver templates that are evaluated and executed by a low-level control logic. The scenario-based environment allows to control the training experience for the agent and to explicitly introduce challenging, but rate situations into the training process. Our experiments show that an agent trained using the SAD-RL framework can achieve safe behaviour in easy as well as challenging situations efficiently. Our ablation studies confirmed that both HRL and scenario diversity are essential for achieving these results. 

---
# Generating Privacy Stories From Software Documentation 

**Authors**: Wilder Baldwin, Shashank Chintakuntla, Shreyah Parajuli, Ali Pourghasemi, Ryan Shanz, Sepideh Ghanavati  

**Link**: [PDF](https://arxiv.org/pdf/2506.23014)  

**Abstract**: Research shows that analysts and developers consider privacy as a security concept or as an afterthought, which may lead to non-compliance and violation of users' privacy. Most current approaches, however, focus on extracting legal requirements from the regulations and evaluating the compliance of software and processes with them. In this paper, we develop a novel approach based on chain-of-thought prompting (CoT), in-context-learning (ICL), and Large Language Models (LLMs) to extract privacy behaviors from various software documents prior to and during software development, and then generate privacy requirements in the format of user stories. Our results show that most commonly used LLMs, such as GPT-4o and Llama 3, can identify privacy behaviors and generate privacy user stories with F1 scores exceeding 0.8. We also show that the performance of these models could be improved through parameter-tuning. Our findings provide insight into using and optimizing LLMs for generating privacy requirements given software documents created prior to or throughout the software development lifecycle. 

---
# A Systematic Study of Compositional Syntactic Transformer Language Models 

**Authors**: Yida Zhao, Hao Xve, Xiang Hu, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22978)  

**Abstract**: Syntactic language models (SLMs) enhance Transformers by incorporating syntactic biases through the modeling of linearized syntactic parse trees alongside surface sentences. This paper focuses on compositional SLMs that are based on constituency parse trees and contain explicit bottom-up composition of constituent representations. We identify key aspects of design choices in existing compositional SLMs and propose a unified framework encompassing both existing models and novel variants. We conduct a comprehensive empirical evaluation of all the variants in our framework across language modeling, syntactic generalization, summarization, dialogue, and inference efficiency. Based on the experimental results, we make multiple recommendations on the design of compositional SLMs. Our code is released at this https URL. 

---
# Against 'softmaxing' culture 

**Authors**: Daniel Mwesigwa  

**Link**: [PDF](https://arxiv.org/pdf/2506.22968)  

**Abstract**: AI is flattening culture. Evaluations of "culture" are showing the myriad ways in which large AI models are homogenizing language and culture, averaging out rich linguistic differences into generic expressions. I call this phenomenon "softmaxing culture," and it is one of the fundamental challenges facing AI evaluations today. Efforts to improve and strengthen evaluations of culture are central to the project of cultural alignment in large AI systems. This position paper argues that machine learning (ML) and human-computer interaction (HCI) approaches to evaluation are limited. I propose two key shifts. First, instead of asking "what is culture?" at the start of system evaluations, I propose beginning with the question: "when is culture?" Second, while I acknowledge the philosophical claim that cultural universals exist, the challenge is not simply to describe them, but to situate them in relation to their particulars. Taken together, these conceptual shifts invite evaluation approaches that move beyond technical requirements, toward perspectives more responsive to the complexities of culture. 

---
# Agent-to-Agent Theory of Mind: Testing Interlocutor Awareness among Large Language Models 

**Authors**: Younwoo Choi, Changling Li, Yongjin Yang, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.22957)  

**Abstract**: As large language models (LLMs) are increasingly integrated into multi-agent and human-AI systems, understanding their awareness of both self-context and conversational partners is essential for ensuring reliable performance and robust safety. While prior work has extensively studied situational awareness which refers to an LLM's ability to recognize its operating phase and constraints, it has largely overlooked the complementary capacity to identify and adapt to the identity and characteristics of a dialogue partner. In this paper, we formalize this latter capability as interlocutor awareness and present the first systematic evaluation of its emergence in contemporary LLMs. We examine interlocutor inference across three dimensions-reasoning patterns, linguistic style, and alignment preferences-and show that LLMs reliably identify same-family peers and certain prominent model families, such as GPT and Claude. To demonstrate its practical significance, we develop three case studies in which interlocutor awareness both enhances multi-LLM collaboration through prompt adaptation and introduces new alignment and safety vulnerabilities, including reward-hacking behaviors and increased jailbreak susceptibility. Our findings highlight the dual promise and peril of identity-sensitive behavior in LLMs, underscoring the need for further understanding of interlocutor awareness and new safeguards in multi-agent deployments. Our code is open-sourced at this https URL. 

---
# A Study on Semi-Supervised Detection of DDoS Attacks under Class Imbalance 

**Authors**: Ehsan Hallaji, Vaishnavi Shanmugam, Roozbeh Razavi-Far, Mehrdad Saif  

**Link**: [PDF](https://arxiv.org/pdf/2506.22949)  

**Abstract**: One of the most difficult challenges in cybersecurity is eliminating Distributed Denial of Service (DDoS) attacks. Automating this task using artificial intelligence is a complex process due to the inherent class imbalance and lack of sufficient labeled samples of real-world datasets. This research investigates the use of Semi-Supervised Learning (SSL) techniques to improve DDoS attack detection when data is imbalanced and partially labeled. In this process, 13 state-of-the-art SSL algorithms are evaluated for detecting DDoS attacks in several scenarios. We evaluate their practical efficacy and shortcomings, including the extent to which they work in extreme environments. The results will offer insight into designing intelligent Intrusion Detection Systems (IDSs) that are robust against class imbalance and handle partially labeled data. 

---
# Positioning AI Tools to Support Online Harm Reduction Practice: Applications and Design Directions 

**Authors**: Kaixuan Wang, Jason T. Jacques, Chenxin Diao  

**Link**: [PDF](https://arxiv.org/pdf/2506.22941)  

**Abstract**: Access to accurate and actionable harm reduction information can directly impact the health outcomes of People Who Use Drugs (PWUD), yet existing online channels often fail to meet their diverse and dynamic needs due to limitations in adaptability, accessibility, and the pervasive impact of stigma. Large Language Models (LLMs) present a novel opportunity to enhance information provision, but their application in such a high-stakes domain is under-explored and presents socio-technical challenges. This paper investigates how LLMs can be responsibly designed to support the information needs of PWUD. Through a qualitative workshop involving diverse stakeholder groups (academics, harm reduction practitioners, and an online community moderator), we explored LLM capabilities, identified potential use cases, and delineated core design considerations. Our findings reveal that while LLMs can address some existing information barriers (e.g., by offering responsive, multilingual, and potentially less stigmatising interactions), their effectiveness is contingent upon overcoming challenges related to ethical alignment with harm reduction principles, nuanced contextual understanding, effective communication, and clearly defined operational boundaries. We articulate design pathways emphasising collaborative co-design with experts and PWUD to develop LLM systems that are helpful, safe, and responsibly governed. This work contributes empirically grounded insights and actionable design considerations for the responsible development of LLMs as supportive tools within the harm reduction ecosystem. 

---
# Mathematical Computation on High-dimensional Data via Array Programming and Parallel Acceleration 

**Authors**: Chen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22929)  

**Abstract**: While deep learning excels in natural image and language processing, its application to high-dimensional data faces computational challenges due to the dimensionality curse. Current large-scale data tools focus on business-oriented descriptive statistics, lacking mathematical statistics support for advanced analysis. We propose a parallel computation architecture based on space completeness, decomposing high-dimensional data into dimension-independent structures for distributed processing. This framework enables seamless integration of data mining and parallel-optimized machine learning methods, supporting scientific computations across diverse data types like medical and natural images within a unified system. 

---
# Learning Truthful Mechanisms without Discretization 

**Authors**: Yunxuan Ma, Siqiang Wang, Zhijian Duan, Yukun Cheng, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.22911)  

**Abstract**: This paper introduces TEDI (Truthful, Expressive, and Dimension-Insensitive approach), a discretization-free algorithm to learn truthful and utility-maximizing mechanisms. Existing learning-based approaches often rely on discretization of outcome spaces to ensure truthfulness, which leads to inefficiency with increasing problem size. To address this limitation, we formalize the concept of pricing rules, defined as functions that map outcomes to prices. Based on this concept, we propose a novel menu mechanism, which can be equivalent to a truthful direct mechanism under specific conditions. The core idea of TEDI lies in its parameterization of pricing rules using Partial GroupMax Network, a new network architecture designed to universally approximate partial convex functions. To learn optimal pricing rules, we develop novel training techniques, including covariance trick and continuous sampling, to derive unbiased gradient estimators compatible with first-order optimization. Theoretical analysis establishes that TEDI guarantees truthfulness, full expressiveness, and dimension-insensitivity. Experimental evaluation in the studied auction setting demonstrates that TEDI achieves strong performance, competitive with or exceeding state-of-the-art methods.
This work presents the first approaches to learn truthful mechanisms without outcome discretization, thereby enhancing algorithmic efficiency. The proposed concepts, network architecture, and learning techniques might offer potential value and provide new insights for automated mechanism design and differentiable economics. 

---
# Missing-Modality-Aware Graph Neural Network for Cancer Classification 

**Authors**: Sina Tabakhi, Haiping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22901)  

**Abstract**: A key challenge in learning from multimodal biological data is missing modalities, where all data from some modalities are missing for some patients. Current fusion methods address this by excluding patients with missing modalities, imputing missing modalities, or making predictions directly with partial modalities. However, they often struggle with diverse missing-modality patterns and the exponential growth of the number of such patterns as the number of modalities increases. To address these limitations, we propose MAGNET (Missing-modality-Aware Graph neural NETwork) for direct prediction with partial modalities, which introduces a patient-modality multi-head attention mechanism to fuse lower-dimensional modality embeddings based on their importance and missingness. MAGNET's complexity increases linearly with the number of modalities while adapting to missing-pattern variability. To generate predictions, MAGNET further constructs a patient graph with fused multimodal embeddings as node features and the connectivity determined by the modality missingness, followed by a conventional graph neural network. Experiments on three public multiomics datasets for cancer classification, with real-world instead of artificial missingness, show that MAGNET outperforms the state-of-the-art fusion methods. The data and code are available at this https URL. 

---
# Interpretable Time Series Autoregression for Periodicity Quantification 

**Authors**: Xinyu Chen, Vassilis Digalakis Jr, Lijun Ding, Dingyi Zhuang, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.22895)  

**Abstract**: Time series autoregression is a classical statistical model for capturing auto-correlations and identifying temporal patterns such as periodicity and seasonality. In this work, we propose a novel sparse autoregression framework from an interpretable machine learning perspective and the model interpretability for periodicity quantification is reinforced by $\ell_0$-norm induced sparsity constraints. On the time-varying time series data, we reformulate the sparse autoregression and convert the involved optimization problem into a mixed-integer optimization (MIO). To accelerate it, we develop a subspace pursuit based decision variable pruning (DVP) strategy to reduce the search space. On the multidimensional time series that involves complicated spatial and temporal dimensions, we propose a spatially- and time-varying sparse autoregression model and resolve the corresponding MIO problem by developing a two-stage optimization scheme. In particular, the proposed scheme makes the model scalable to large problems even with millions of decision variables. Empirically, we conduct extensive experiments to evaluate the proposed models on real-world time series data. First, we demonstrate that the MIO solver can be drastically accelerated through the DVP strategy, while maintaining the same solution quality as a full MIO solver. Applying the time-varying sparse autoregression model to ridesharing trip data, we uncover both daily and weekly periodicities and reveal long-term changes in regularity of human mobility. Second, we demonstrate the spatial patterns of yearly seasonality in climate variable time series such as temperature and precipitation across the past four decades, and our model allows to discover dynamic climate patterns and identify climate phenomena such as El Nino in sea surface temperature. 

---
# Performance Measurements in the AI-Centric Computing Continuum Systems 

**Authors**: Praveen Kumar Donta, Qiyang Zhang, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.22884)  

**Abstract**: Over the Eight decades, computing paradigms have shifted from large, centralized systems to compact, distributed architectures, leading to the rise of the Distributed Computing Continuum (DCC). In this model, multiple layers such as cloud, edge, Internet of Things (IoT), and mobile platforms work together to support a wide range of applications. Recently, the emergence of Generative AI and large language models has further intensified the demand for computational resources across this continuum. Although traditional performance metrics have provided a solid foundation, they need to be revisited and expanded to keep pace with changing computational demands and application requirements. Accurate performance measurements benefit both system designers and users by supporting improvements in efficiency and promoting alignment with system goals. In this context, we review commonly used metrics in DCC and IoT environments. We also discuss emerging performance dimensions that address evolving computing needs, such as sustainability, energy efficiency, and system observability. We also outline criteria and considerations for selecting appropriate metrics, aiming to inspire future research and development in this critical area. 

---
# Decoupled Seg Tokens Make Stronger Reasoning Video Segmenter and Grounder 

**Authors**: Dang Jisheng, Wu Xudong, Wang Bimei, Lv Ning, Chen Jiayu, Jingwen Zhao, Yichu liu, Jizhao Liu, Juncheng Li, Teng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22880)  

**Abstract**: Existing video segmenter and grounder approaches, exemplified by Sa2VA, directly fuse features within segmentation models. This often results in an undesirable entanglement of dynamic visual information and static semantics, thereby degrading segmentation accuracy. To systematically mitigate this issue, we propose DeSa2VA, a decoupling-enhanced prompting scheme integrating text pre-training and a linear decoupling module to address the information processing limitations inherent in SAM-2. Specifically, first, we devise a pre-training paradigm that converts textual ground-truth labels into point-level prompts while generating corresponding text masks. These masks are refined through a hybrid loss function to strengthen the model's semantic grounding capabilities. Next, we employ linear projection to disentangle hidden states that generated by a large language model into distinct textual and visual feature subspaces. Finally, a dynamic mask fusion strategy synergistically combines these decoupled features through triple supervision from predicted text/visual masks and ground-truth annotations. Extensive experiments demonstrate state-of-the-art performance across diverse tasks, including image segmentation, image question answering, video segmentation, and video question answering. Our codes are available at this https URL. 

---
# STR-Match: Matching SpatioTemporal Relevance Score for Training-Free Video Editing 

**Authors**: Junsung Lee, Junoh Kang, Bohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.22868)  

**Abstract**: Previous text-guided video editing methods often suffer from temporal inconsistency, motion distortion, and-most notably-limited domain transformation. We attribute these limitations to insufficient modeling of spatiotemporal pixel relevance during the editing process. To address this, we propose STR-Match, a training-free video editing algorithm that produces visually appealing and spatiotemporally coherent videos through latent optimization guided by our novel STR score. The score captures spatiotemporal pixel relevance across adjacent frames by leveraging 2D spatial attention and 1D temporal modules in text-to-video (T2V) diffusion models, without the overhead of computationally expensive 3D attention mechanisms. Integrated into a latent optimization framework with a latent mask, STR-Match generates temporally consistent and visually faithful videos, maintaining strong performance even under significant domain transformations while preserving key visual attributes of the source. Extensive experiments demonstrate that STR-Match consistently outperforms existing methods in both visual quality and spatiotemporal consistency. 

---
# Region-Aware CAM: High-Resolution Weakly-Supervised Defect Segmentation via Salient Region Perception 

**Authors**: Hang-Cheng Dong, Lu Zou, Bingguo Liu, Dong Ye, Guodong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22866)  

**Abstract**: Surface defect detection plays a critical role in industrial quality inspection. Recent advances in artificial intelligence have significantly enhanced the automation level of detection processes. However, conventional semantic segmentation and object detection models heavily rely on large-scale annotated datasets, which conflicts with the practical requirements of defect detection tasks. This paper proposes a novel weakly supervised semantic segmentation framework comprising two key components: a region-aware class activation map (CAM) and pseudo-label training. To address the limitations of existing CAM methods, especially low-resolution thermal maps, and insufficient detail preservation, we introduce filtering-guided backpropagation (FGBP), which refines target regions by filtering gradient magnitudes to identify areas with higher relevance to defects. Building upon this, we further develop a region-aware weighted module to enhance spatial precision. Finally, pseudo-label segmentation is implemented to refine the model's performance iteratively. Comprehensive experiments on industrial defect datasets demonstrate the superiority of our method. The proposed framework effectively bridges the gap between weakly supervised learning and high-precision defect segmentation, offering a practical solution for resource-constrained industrial scenarios. 

---
# Mask-aware Text-to-Image Retrieval: Referring Expression Segmentation Meets Cross-modal Retrieval 

**Authors**: Li-Cheng Shen, Jih-Kang Hsieh, Wei-Hua Li, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.22864)  

**Abstract**: Text-to-image retrieval (TIR) aims to find relevant images based on a textual query, but existing approaches are primarily based on whole-image captions and lack interpretability. Meanwhile, referring expression segmentation (RES) enables precise object localization based on natural language descriptions but is computationally expensive when applied across large image collections. To bridge this gap, we introduce Mask-aware TIR (MaTIR), a new task that unifies TIR and RES, requiring both efficient image search and accurate object segmentation. To address this task, we propose a two-stage framework, comprising a first stage for segmentation-aware image retrieval and a second stage for reranking and object grounding with a multimodal large language model (MLLM). We leverage SAM 2 to generate object masks and Alpha-CLIP to extract region-level embeddings offline at first, enabling effective and scalable online retrieval. Secondly, MLLM is used to refine retrieval rankings and generate bounding boxes, which are matched to segmentation masks. We evaluate our approach on COCO and D$^3$ datasets, demonstrating significant improvements in both retrieval accuracy and segmentation quality over previous methods. 

---
# DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues 

**Authors**: Kyochul Jang, Donghyeon Lee, Kyusik Kim, Dongseok Heo, Taewhoo Lee, Woojeong Kim, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2506.22853)  

**Abstract**: Existing function-calling benchmarks focus on single-turn interactions. However, they overlook the complexity of real-world scenarios. To quantify how existing benchmarks address practical applications, we introduce DICE-SCORE, a metric that evaluates the dispersion of tool-related information such as function name and parameter values throughout the dialogue. Analyzing existing benchmarks through DICE-SCORE reveals notably low scores, highlighting the need for more realistic scenarios. To address this gap, we present DICE-BENCH, a framework that constructs practical function-calling datasets by synthesizing conversations through a tool graph that maintains dependencies across rounds and a multi-agent system with distinct personas to enhance dialogue naturalness. The final dataset comprises 1,607 high-DICE-SCORE instances. Our experiments on 19 LLMs with DICE-BENCH show that significant advances are still required before such models can be deployed effectively in real-world settings. Our code and data are all publicly available: this https URL. 

---
# Scalable Structure Learning of Bayesian Networks by Learning Algorithm Ensembles 

**Authors**: Shengcai Liu, Hui Ou-yang, Zhiyuan Wang, Cheng Chen, Qijun Cai, Yew-Soon Ong, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22848)  

**Abstract**: Learning the structure of Bayesian networks (BNs) from data is challenging, especially for datasets involving a large number of variables. The recently proposed divide-and-conquer (D\&D) strategies present a promising approach for learning large BNs. However, they still face a main issue of unstable learning accuracy across subproblems. In this work, we introduce the idea of employing structure learning ensemble (SLE), which combines multiple BN structure learning algorithms, to consistently achieve high learning accuracy. We further propose an automatic approach called Auto-SLE for learning near-optimal SLEs, addressing the challenge of manually designing high-quality SLEs. The learned SLE is then integrated into a D\&D method. Extensive experiments firmly show the superiority of our method over D\&D methods with single BN structure learning algorithm in learning large BNs, achieving accuracy improvement usually by 30\%$\sim$225\% on datasets involving 10,000 variables. Furthermore, our method generalizes well to datasets with many more (e.g., 30000) variables and different network characteristics than those present in the training data for learning the SLE. These results indicate the significant potential of employing (automatic learning of) SLEs for scalable BN structure learning. 

---
# Quantum Neural Networks for Wind Energy Forecasting: A Comparative Study of Performance and Scalability with Classical Models 

**Authors**: Batuhan Hangun, Oguz Altun, Onder Eyecioglu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22845)  

**Abstract**: Quantum Neural Networks (QNNs), a prominent approach in Quantum Machine Learning (QML), are emerging as a powerful alternative to classical machine learning methods. Recent studies have focused on the applicability of QNNs to various tasks, such as time-series forecasting, prediction, and classification, across a wide range of applications, including cybersecurity and medical imaging. With the increased use of smart grids driven by the integration of renewable energy systems, machine learning plays an important role in predicting power demand and detecting system disturbances. This study provides an in-depth investigation of QNNs for predicting the power output of a wind turbine. We assess the predictive performance and simulation time of six QNN configurations that are based on the Z Feature Map for data encoding and varying ansatz structures. Through detailed cross-validation experiments and tests on an unseen hold-out dataset, we experimentally demonstrate that QNNs can achieve predictive performance that is competitive with, and in some cases marginally better than, the benchmarked classical approaches. Our results also reveal the effects of dataset size and circuit complexity on predictive performance and simulation time. We believe our findings will offer valuable insights for researchers in the energy domain who wish to incorporate quantum machine learning into their work. 

---
# xLSTMAD: A Powerful xLSTM-based Method for Anomaly Detection 

**Authors**: Kamil Faber, Marcin Pietroń, Dominik Żurek, Roberto Corizzo  

**Link**: [PDF](https://arxiv.org/pdf/2506.22837)  

**Abstract**: The recently proposed xLSTM is a powerful model that leverages expressive multiplicative gating and residual connections, providing the temporal capacity needed for long-horizon forecasting and representation learning. This architecture has demonstrated success in time series forecasting, lossless compression, and even large-scale language modeling tasks, where its linear memory footprint and fast inference make it a viable alternative to Transformers. Despite its growing popularity, no prior work has explored xLSTM for anomaly detection. In this work, we fill this gap by proposing xLSTMAD, the first anomaly detection method that integrates a full encoder-decoder xLSTM architecture, purpose-built for multivariate time series data. Our encoder processes input sequences to capture historical context, while the decoder is devised in two separate variants of the method. In the forecasting approach, the decoder iteratively generates forecasted future values xLSTMAD-F, while the reconstruction approach reconstructs the input time series from its encoded counterpart xLSTMAD-R. We investigate the performance of two loss functions: Mean Squared Error (MSE), and Soft Dynamic Time Warping (SoftDTW) to consider local reconstruction fidelity and global sequence alignment, respectively. We evaluate our method on the comprehensive TSB-AD-M benchmark, which spans 17 real-world datasets, using state-of-the-art challenging metrics such as VUS-PR. In our results, xLSTM showcases state-of-the-art accuracy, outperforming 23 popular anomaly detection baselines. Our paper is the first work revealing the powerful modeling capabilities of xLSTM for anomaly detection, paving the way for exciting new developments on this subject. Our code is available at: this https URL 

---
# Listener-Rewarded Thinking in VLMs for Image Preferences 

**Authors**: Alexander Gambashidze, Li Pengyi, Matvey Skripkin, Andrey Galichin, Anton Gusarov, Konstantin Sobolev, Andrey Kuznetsov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2506.22832)  

**Abstract**: Training robust and generalizable reward models for human visual preferences is essential for aligning text-to-image and text-to-video generative models with human intent. However, current reward models often fail to generalize, and supervised fine-tuning leads to memorization, demanding complex annotation pipelines. While reinforcement learning (RL), specifically Group Relative Policy Optimization (GRPO), improves generalization, we uncover a key failure mode: a significant drop in reasoning accuracy occurs when a model's reasoning trace contradicts that of an independent, frozen vision-language model ("listener") evaluating the same output. To address this, we introduce a listener-augmented GRPO framework. Here, the listener re-evaluates the reasoner's chain-of-thought to provide a dense, calibrated confidence score, shaping the RL reward signal. This encourages the reasoner not only to answer correctly, but to produce explanations that are persuasive to an independent model. Our listener-shaped reward scheme achieves best accuracy on the ImageReward benchmark (67.4%), significantly improves out-of-distribution (OOD) performance on a large-scale human preference dataset (1.2M votes, up to +6% over naive reasoner), and reduces reasoning contradictions compared to strong GRPO and SFT baselines. These results demonstrate that listener-based rewards provide a scalable, data-efficient path to aligning vision-language models with nuanced human preferences. We will release our reasoning model here: this https URL. 

---
# TriADA: Massively Parallel Trilinear Matrix-by-Tensor Multiply-Add Algorithm and Device Architecture for the Acceleration of 3D Discrete Transformations 

**Authors**: Stanislav Sedukhin, Yoichi Tomioka, Kazuya Matsumoto, Yuichi Okuyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.22818)  

**Abstract**: Multilinear transformations are key in high-performance computing (HPC) and artificial intelligence (AI) workloads, where data is represented as tensors. However, their high computational and memory demands, which grow with dimensionality, often slow down critical tasks. Moreover, scaling computation by enlarging the number of parallel processing units substantially increases energy consumption, limiting widespread adoption, especially for sparse data, which is common in HPC and AI applications. This paper introduces the Trilinear Algorithm and isomorphic to algorithm Device Architecture (TriADA) to address these challenges with the following innovations: (1) a massively parallel, low-rank algorithm for computing a family of trilinear (3D) discrete orthogonal transformations (3D-DXTs), which is a special case of the more general 3-mode matrix-by-tensor multiplication (3D-GEMT); (2) a new outer-product-based GEMM kernel with decoupled streaming active memory, specially designed to accelerate 3D-GEMT operation; (3) an isomorphic to the proposed algorithm, fully distributed 3D network of mesh interconnected processing elements or cells with a coordinate-free, data-driven local processing activity, which is independent of problem size; (4) an elastic sparse outer-product (ESOP) method that avoids unnecessary computing and communication operations with zero-valued operands, thereby enhancing energy efficiency, computational accuracy, and stability. TriADA is capable of performing a variety of trilinear transformations with hypercubic arithmetic complexity in a linear number of time-steps. The massively parallel, scalable, and energy-efficient architecture of TriADA is ideal for accelerating multilinear tensor operations, which are the most demanding parts of AI and HPC workloads. 

---
# BayesLoRA: Task-Specific Uncertainty in Low-Rank Adapters 

**Authors**: Cooper Doyle  

**Link**: [PDF](https://arxiv.org/pdf/2506.22809)  

**Abstract**: We propose BayesLoRA, a task-specific uncertainty quantification framework that integrates MC-Dropout into Low-Rank Adapters (LoRA). Unlike general-purpose transformer uncertainty methods, BayesLoRA provides guardrails tailored to downstream workflows, enabling agents to introspect and modulate behavior under uncertainty. We demonstrate mathematically and empirically that LoRA adapters exhibit amplified variance outside fine-tuning distributions, yielding reliable confidence estimates for agentic decision-making. 

---
# MedEthicsQA: A Comprehensive Question Answering Benchmark for Medical Ethics Evaluation of LLMs 

**Authors**: Jianhui Wei, Zijie Meng, Zikai Xiao, Tianxiang Hu, Yang Feng, Zhijie Zhou, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22808)  

**Abstract**: While Medical Large Language Models (MedLLMs) have demonstrated remarkable potential in clinical tasks, their ethical safety remains insufficiently explored. This paper introduces $\textbf{MedEthicsQA}$, a comprehensive benchmark comprising $\textbf{5,623}$ multiple-choice questions and $\textbf{5,351}$ open-ended questions for evaluation of medical ethics in LLMs. We systematically establish a hierarchical taxonomy integrating global medical ethical standards. The benchmark encompasses widely used medical datasets, authoritative question banks, and scenarios derived from PubMed literature. Rigorous quality control involving multi-stage filtering and multi-faceted expert validation ensures the reliability of the dataset with a low error rate ($2.72\%$). Evaluation of state-of-the-art MedLLMs exhibit declined performance in answering medical ethics questions compared to their foundation counterparts, elucidating the deficiencies of medical ethics alignment. The dataset, registered under CC BY-NC 4.0 license, is available at this https URL. 

---
# Offline Reinforcement Learning for Mobility Robustness Optimization 

**Authors**: Pegah Alizadeh, Anastasios Giovanidis, Pradeepa Ramachandra, Vasileios Koutsoukis, Osama Arouk  

**Link**: [PDF](https://arxiv.org/pdf/2506.22793)  

**Abstract**: In this work we revisit the Mobility Robustness Optimisation (MRO) algorithm and study the possibility of learning the optimal Cell Individual Offset tuning using offline Reinforcement Learning. Such methods make use of collected offline datasets to learn the optimal policy, without further exploration. We adapt and apply a sequence-based method called Decision Transformers as well as a value-based method called Conservative Q-Learning to learn the optimal policy for the same target reward as the vanilla rule-based MRO. The same input features related to failures, ping-pongs, and other handover issues are used. Evaluation for realistic New Radio networks with 3500 MHz carrier frequency on a traffic mix including diverse user service types and a specific tunable cell-pair shows that offline-RL methods outperform rule-based MRO, offering up to 7% improvement. Furthermore, offline-RL can be trained for diverse objective functions using the same available dataset, thus offering operational flexibility compared to rule-based methods. 

---
# WavShape: Information-Theoretic Speech Representation Learning for Fair and Privacy-Aware Audio Processing 

**Authors**: Oguzhan Baser, Ahmet Ege Tanriverdi, Kaan Kale, Sandeep P. Chinchali, Sriram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2506.22789)  

**Abstract**: Speech embeddings often retain sensitive attributes such as speaker identity, accent, or demographic information, posing risks in biased model training and privacy leakage. We propose WavShape, an information-theoretic speech representation learning framework that optimizes embeddings for fairness and privacy while preserving task-relevant information. We leverage mutual information (MI) estimation using the Donsker-Varadhan formulation to guide an MI-based encoder that systematically filters sensitive attributes while maintaining speech content essential for downstream tasks. Experimental results on three known datasets show that WavShape reduces MI between embeddings and sensitive attributes by up to 81% while retaining 97% of task-relevant information. By integrating information theory with self-supervised speech models, this work advances the development of fair, privacy-aware, and resource-efficient speech systems. 

---
# Single-Frame Point-Pixel Registration via Supervised Cross-Modal Feature Matching 

**Authors**: Yu Han, Zhiwei Huang, Yanting Zhang, Fangjun Ding, Shen Cai, Rui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.22784)  

**Abstract**: Point-pixel registration between LiDAR point clouds and camera images is a fundamental yet challenging task in autonomous driving and robotic perception. A key difficulty lies in the modality gap between unstructured point clouds and structured images, especially under sparse single-frame LiDAR settings. Existing methods typically extract features separately from point clouds and images, then rely on hand-crafted or learned matching strategies. This separate encoding fails to bridge the modality gap effectively, and more critically, these methods struggle with the sparsity and noise of single-frame LiDAR, often requiring point cloud accumulation or additional priors to improve reliability. Inspired by recent progress in detector-free matching paradigms (e.g. MatchAnything), we revisit the projection-based approach and introduce the detector-free framework for direct point-pixel matching between LiDAR and camera views. Specifically, we project the LiDAR intensity map into a 2D view from the LiDAR perspective and feed it into an attention-based detector-free matching network, enabling cross-modal correspondence estimation without relying on multi-frame accumulation. To further enhance matching reliability, we introduce a repeatability scoring mechanism that acts as a soft visibility prior. This guides the network to suppress unreliable matches in regions with low intensity variation, improving robustness under sparse input. Extensive experiments on KITTI, nuScenes, and MIAS-LCEC-TF70 benchmarks demonstrate that our method achieves state-of-the-art performance, outperforming prior approaches on nuScenes (even those relying on accumulated point clouds), despite using only single-frame LiDAR. 

---
# PhonemeFake: Redefining Deepfake Realism with Language-Driven Segmental Manipulation and Adaptive Bilevel Detection 

**Authors**: Oguzhan Baser, Ahmet Ege Tanriverdi, Sriram Vishwanath, Sandeep P. Chinchali  

**Link**: [PDF](https://arxiv.org/pdf/2506.22783)  

**Abstract**: Deepfake (DF) attacks pose a growing threat as generative models become increasingly advanced. However, our study reveals that existing DF datasets fail to deceive human perception, unlike real DF attacks that influence public discourse. It highlights the need for more realistic DF attack vectors. We introduce PhonemeFake (PF), a DF attack that manipulates critical speech segments using language reasoning, significantly reducing human perception by up to 42% and benchmark accuracies by up to 94%. We release an easy-to-use PF dataset on HuggingFace and open-source bilevel DF segment detection model that adaptively prioritizes compute on manipulated regions. Our extensive experiments across three known DF datasets reveal that our detection model reduces EER by 91% while achieving up to 90% speed-up, with minimal compute overhead and precise localization beyond existing models as a scalable solution. 

---
# Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning 

**Authors**: Miles Turpin, Andy Arditi, Marvin Li, Joe Benton, Julian Michael  

**Link**: [PDF](https://arxiv.org/pdf/2506.22777)  

**Abstract**: Language models trained with RL can engage in reward hacking--exploiting unintended strategies for high reward--without revealing this behavior in their chain-of-thought reasoning, making detection difficult and posing risks for high-stakes applications. We propose verbalization fine-tuning (VFT), a pre-RL intervention that trains models to explicitly acknowledge when they are influenced by prompt cues--hints which point to incorrect answers (e.g., "a Stanford professor thinks the answer is A"). To evaluate VFT, we subsequently train models with RL on environments where held-out prompt cues signal which incorrect answers will receive high reward, incentivizing models to reward hack by exploiting cues instead of reasoning correctly. We measure how often models exploit these cues without verbalizing it. After RL, only 6% of the VFT-trained model's responses consist of undetected reward hacks. In comparison, when we perform RL without VFT, the rate of undetected reward hacks goes up to 88%; with a debiasing baseline intervention, this increases further to 99%. VFT achieves this by substantially increasing how often models verbalize the influence of cues--from 8% to 42% after VFT, and up to 94% after RL--while baselines remain low even after RL (10% and 1%). Our results show that teaching models to explicitly verbalize reward hacking behavior before RL significantly improves their detection, offering a practical path toward more transparent and safe AI systems. 

---
# Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation 

**Authors**: Sen Fang, Weiyuan Ding, Antonio Mastropaolo, Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22776)  

**Abstract**: Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely this http URL this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies. 

---
# FF-INT8: Efficient Forward-Forward DNN Training on Edge Devices with INT8 Precision 

**Authors**: Jingxiao Ma, Priyadarshini Panda, Sherief Reda  

**Link**: [PDF](https://arxiv.org/pdf/2506.22771)  

**Abstract**: Backpropagation has been the cornerstone of neural network training for decades, yet its inefficiencies in time and energy consumption limit its suitability for resource-constrained edge devices. While low-precision neural network quantization has been extensively researched to speed up model inference, its application in training has been less explored. Recently, the Forward-Forward (FF) algorithm has emerged as a promising alternative to backpropagation, replacing the backward pass with an additional forward pass. By avoiding the need to store intermediate activations for backpropagation, FF can reduce memory footprint, making it well-suited for embedded devices. This paper presents an INT8 quantized training approach that leverages FF's layer-by-layer strategy to stabilize gradient quantization. Furthermore, we propose a novel "look-ahead" scheme to address limitations of FF and improve model accuracy. Experiments conducted on NVIDIA Jetson Orin Nano board demonstrate 4.6% faster training, 8.3% energy savings, and 27.0% reduction in memory usage, while maintaining competitive accuracy compared to the state-of-the-art. 

---
# RAILS: Retrieval-Augmented Intelligence for Learning Software Development 

**Authors**: Wali Mohammad Abdullah, Md. Morshedul Islam, Devraj Parmar, Happy Hasmukhbhai Patel, Sindhuja Prabhakaran, Baidya Saha  

**Link**: [PDF](https://arxiv.org/pdf/2506.22742)  

**Abstract**: Large Language Models (LLMs) like GPT-3.5-Turbo are increasingly used to assist software development, yet they often produce incomplete code or incorrect imports, especially when lacking access to external or project-specific documentation. We introduce RAILS (Retrieval-Augmented Intelligence for Learning Software Development), a framework that augments LLM prompts with semantically retrieved context from curated Java resources using FAISS and OpenAI embeddings. RAILS incorporates an iterative validation loop guided by compiler feedback to refine suggestions. We evaluated RAILS on 78 real-world Java import error cases spanning standard libraries, GUI APIs, external tools, and custom utilities. Despite using the same LLM, RAILS outperforms baseline prompting by preserving intent, avoiding hallucinations, and surfacing correct imports even when libraries are unavailable locally. Future work will integrate symbolic filtering via PostgreSQL and extend support to other languages and IDEs. 

---
# Kill Two Birds with One Stone! Trajectory enabled Unified Online Detection of Adversarial Examples and Backdoor Attacks 

**Authors**: Anmin Fu, Fanyu Meng, Huaibing Peng, Hua Ma, Zhi Zhang, Yifeng Zheng, Willy Susilo, Yansong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.22722)  

**Abstract**: The proposed UniGuard is the first unified online detection framework capable of simultaneously addressing adversarial examples and backdoor attacks. UniGuard builds upon two key insights: first, both AE and backdoor attacks have to compromise the inference phase, making it possible to tackle them simultaneously during run-time via online detection. Second, an adversarial input, whether a perturbed sample in AE attacks or a trigger-carrying sample in backdoor attacks, exhibits distinctive trajectory signatures from a benign sample as it propagates through the layers of a DL model in forward inference. The propagation trajectory of the adversarial sample must deviate from that of its benign counterpart; otherwise, the adversarial objective cannot be fulfilled. Detecting these trajectory signatures is inherently challenging due to their subtlety; UniGuard overcomes this by treating the propagation trajectory as a time-series signal, leveraging LSTM and spectrum transformation to amplify differences between adversarial and benign trajectories that are subtle in the time domain. UniGuard exceptional efficiency and effectiveness have been extensively validated across various modalities (image, text, and audio) and tasks (classification and regression), ranging from diverse model architectures against a wide range of AE attacks and backdoor attacks, including challenging partial backdoors and dynamic triggers. When compared to SOTA methods, including ContraNet (NDSS 22) specific for AE detection and TED (IEEE SP 24) specific for backdoor detection, UniGuard consistently demonstrates superior performance, even when matched against each method's strengths in addressing their respective threats-each SOTA fails to parts of attack strategies while UniGuard succeeds for all. 

---
# BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute 

**Authors**: Dujian Ding, Ankur Mallick, Shaokun Zhang, Chi Wang, Daniel Madrigal, Mirian Del Carmen Hipolito Garcia, Menglin Xia, Laks V.S. Lakshmanan, Qingyun Wu, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2506.22716)  

**Abstract**: Large language models (LLMs) are powerful tools but are often expensive to deploy at scale. LLM query routing mitigates this by dynamically assigning queries to models of varying cost and quality to obtain a desired trade-off. Prior query routing approaches generate only one response from the selected model and a single response from a small (inexpensive) model was often not good enough to beat a response from a large (expensive) model due to which they end up overusing the large model and missing out on potential cost savings. However, it is well known that for small models, generating multiple responses and selecting the best can enhance quality while remaining cheaper than a single large-model response. We leverage this idea to propose BEST-Route, a novel routing framework that chooses a model and the number of responses to sample from it based on query difficulty and the quality thresholds. Experiments on real-world datasets demonstrate that our method reduces costs by up to 60% with less than 1% performance drop. 

---
# General Autonomous Cybersecurity Defense: Learning Robust Policies for Dynamic Topologies and Diverse Attackers 

**Authors**: Arun Ramamurthy, Neil Dhir  

**Link**: [PDF](https://arxiv.org/pdf/2506.22706)  

**Abstract**: In the face of evolving cyber threats such as malware, ransomware and phishing, autonomous cybersecurity defense (ACD) systems have become essential for real-time threat detection and response with optional human intervention. However, existing ACD systems rely on limiting assumptions, particularly the stationarity of the underlying network dynamics. In real-world scenarios, network topologies can change due to actions taken by attackers or defenders, system failures, or time evolution of networks, leading to failures in the adaptive capabilities of current defense agents. Moreover, many agents are trained on static environments, resulting in overfitting to specific topologies, which hampers their ability to generalize to out-of-distribution network topologies. This work addresses these challenges by exploring methods for developing agents to learn generalizable policies across dynamic network environments -- general ACD (GACD). 

---
# Beyond Code: The Multidimensional Impacts of Large Language Models in Software Development 

**Authors**: Sardar Fatooreh Bonabi, Sarah Bana, Tingting Nian, Vijay Gurbaxani  

**Link**: [PDF](https://arxiv.org/pdf/2506.22704)  

**Abstract**: Large language models (LLMs) are poised to significantly impact software development, especially in the Open-Source Software (OSS) sector. To understand this impact, we first outline the mechanisms through which LLMs may influence OSS through code development, collaborative knowledge transfer, and skill development. We then empirically examine how LLMs affect OSS developers' work in these three key areas. Leveraging a natural experiment from a temporary ChatGPT ban in Italy, we employ a Difference-in-Differences framework with two-way fixed effects to analyze data from all OSS developers on GitHub in three similar countries, Italy, France, and Portugal, totaling 88,022 users. We find that access to ChatGPT increases developer productivity by 6.4%, knowledge sharing by 9.6%, and skill acquisition by 8.4%. These benefits vary significantly by user experience level: novice developers primarily experience productivity gains, whereas more experienced developers benefit more from improved knowledge sharing and accelerated skill acquisition. In addition, we find that LLM-assisted learning is highly context-dependent, with the greatest benefits observed in technically complex, fragmented, or rapidly evolving contexts. We show that the productivity effects of LLMs extend beyond direct code generation to include enhanced collaborative learning and knowledge exchange among developers; dynamics that are essential for gaining a holistic understanding of LLMs' impact in OSS. Our findings offer critical managerial implications: strategically deploying LLMs can accelerate novice developers' onboarding and productivity, empower intermediate developers to foster knowledge sharing and collaboration, and support rapid skill acquisition, together enhancing long-term organizational productivity and agility. 

---
# P4OMP: Retrieval-Augmented Prompting for OpenMP Parallelism in Serial Code 

**Authors**: Wali Mohammad Abdullah, Azmain Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2506.22703)  

**Abstract**: We present P4OMP, a retrieval-augmented framework for transforming serial C/C++ code into OpenMP-annotated parallel code using large language models (LLMs). To our knowledge, this is the first system to apply retrieval-based prompting for OpenMP pragma correctness without model fine-tuning or compiler instrumentation. P4OMP leverages Retrieval-Augmented Generation (RAG) with structured instructional knowledge from OpenMP tutorials to improve the reliability of prompt-driven code generation. By grounding generation in the retrieved context, P4OMP improves syntactic correctness compared to baseline prompting with GPT-3.5-Turbo. We evaluate P4OMP against a baseline, GPT-3.5-Turbo without retrieval, on a comprehensive benchmark of 108 real-world C++ programs drawn from Stack Overflow, PolyBench, and NAS benchmark suites. P4OMP achieves 100% compilation success on all parallelizable cases, while the baseline fails to compile in 20 out of 108 cases. Six cases that rely on non-random-access iterators or thread-unsafe constructs are excluded due to fundamental OpenMP limitations. A detailed analysis demonstrates how P4OMP consistently avoids scoping errors, syntactic misuse, and invalid directive combinations that commonly affect baseline-generated code. We further demonstrate strong runtime scaling across seven compute-intensive benchmarks on an HPC cluster. P4OMP offers a robust, modular pipeline that significantly improves the reliability and applicability of LLM-generated OpenMP code. 

---
# Text Production and Comprehension by Human and Artificial Intelligence: Interdisciplinary Workshop Report 

**Authors**: Emily Dux Speltz  

**Link**: [PDF](https://arxiv.org/pdf/2506.22698)  

**Abstract**: This report synthesizes the outcomes of a recent interdisciplinary workshop that brought together leading experts in cognitive psychology, language learning, and artificial intelligence (AI)-based natural language processing (NLP). The workshop, funded by the National Science Foundation, aimed to address a critical knowledge gap in our understanding of the relationship between AI language models and human cognitive processes in text comprehension and composition. Through collaborative dialogue across cognitive, linguistic, and technological perspectives, workshop participants examined the underlying processes involved when humans produce and comprehend text, and how AI can both inform our understanding of these processes and augment human capabilities. The workshop revealed emerging patterns in the relationship between large language models (LLMs) and human cognition, with highlights on both the capabilities of LLMs and their limitations in fully replicating human-like language understanding and generation. Key findings include the potential of LLMs to offer insights into human language processing, the increasing alignment between LLM behavior and human language processing when models are fine-tuned with human feedback, and the opportunities and challenges presented by human-AI collaboration in language tasks. By synthesizing these findings, this report aims to guide future research, development, and implementation of LLMs in cognitive psychology, linguistics, and education. It emphasizes the importance of ethical considerations and responsible use of AI technologies while striving to enhance human capabilities in text comprehension and production through effective human-AI collaboration. 

---
# DistShap: Scalable GNN Explanations with Distributed Shapley Values 

**Authors**: Selahattin Akkas, Aditya Devarakonda, Ariful Azad  

**Link**: [PDF](https://arxiv.org/pdf/2506.22668)  

**Abstract**: With the growing adoption of graph neural networks (GNNs), explaining their predictions has become increasingly important. However, attributing predictions to specific edges or features remains computationally expensive. For example, classifying a node with 100 neighbors using a 3-layer GNN may involve identifying important edges from millions of candidates contributing to the prediction. To address this challenge, we propose DistShap, a parallel algorithm that distributes Shapley value-based explanations across multiple GPUs. DistShap operates by sampling subgraphs in a distributed setting, executing GNN inference in parallel across GPUs, and solving a distributed least squares problem to compute edge importance scores. DistShap outperforms most existing GNN explanation methods in accuracy and is the first to scale to GNN models with millions of features by using up to 128 GPUs on the NERSC Perlmutter supercomputer. 

---
# Knowledge-Guided Multi-Agent Framework for Automated Requirements Development: A Vision 

**Authors**: Jiangping Huang, Dongming Jin, Weisong Sun, Yang Liu, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.22656)  

**Abstract**: This paper envisions a knowledge-guided multi-agent framework named KGMAF for automated requirements development. KGMAF aims to address gaps in current automation systems for SE, which prioritize code development and overlook the complexities of requirements tasks. KGMAF is composed of six specialized agents and an artifact pool to improve efficiency and accuracy. Specifically, KGMAF outlines the functionality, actions, and knowledge of each agent and provides the conceptual design of the artifact pool. Our case study highlights the potential of KGMAF in real-world scenarios. Finally, we outline several research opportunities for implementing and enhancing automated requirements development using multi-agent systems. We believe that KGMAF will play a pivotal role in shaping the future of automated requirements development in the era of LLMs. 

---
# Layer Importance for Mathematical Reasoning is Forged in Pre-Training and Invariant after Post-Training 

**Authors**: Aadim Nepal, Safal Shrestha, Anubhav Shrestha, Minwu Kim, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2506.22638)  

**Abstract**: Large language models can exhibit improved mathematical reasoning capabilities following post-training with instruction tuning, reinforcement learning, or knowledge distillation. However, it remains unclear whether these improvements are driven by major changes in transformer layers or from minor adjustments that leave the relative layer importance structures of the base model largely unchanged. We investigate this question through systematic layer-wise ablation experiments, examining base, instruction-tuned, knowledge-distilled, and reinforcement learning variants on mathematical reasoning benchmarks. Our findings show that mathematical reasoning gives rise to a specific layer importance structure, and this structure persists across all post-training paradigms. Removal of such layers causes accuracy drops of up to 80%. In contrast, non-mathematical tasks like factual recall exhibit no critical layers. This distinction suggests that mathematical reasoning requires specialized layers that emerge during pre-training, while other non-reasoning tasks do not. From an information-theoretic perspective, we also observe that these critical layers are the same layers where major representational transformation occurs. 

---
# Temperature Matters: Enhancing Watermark Robustness Against Paraphrasing Attacks 

**Authors**: Badr Youbi Idrissi, Monica Millunzi, Amelia Sorrenti, Lorenzo Baraldi, Daryna Dementieva  

**Link**: [PDF](https://arxiv.org/pdf/2506.22623)  

**Abstract**: In the present-day scenario, Large Language Models (LLMs) are establishing their presence as powerful instruments permeating various sectors of society. While their utility offers valuable support to individuals, there are multiple concerns over potential misuse. Consequently, some academic endeavors have sought to introduce watermarking techniques, characterized by the inclusion of markers within machine-generated text, to facilitate algorithmic identification. This research project is focused on the development of a novel methodology for the detection of synthetic text, with the overarching goal of ensuring the ethical application of LLMs in AI-driven text generation. The investigation commences with replicating findings from a previous baseline study, thereby underscoring its susceptibility to variations in the underlying generation model. Subsequently, we propose an innovative watermarking approach and subject it to rigorous evaluation, employing paraphrased generated text to asses its robustness. Experimental results highlight the robustness of our proposal compared to the~\cite{aarson} watermarking method. 

---
# Pixels-to-Graph: Real-time Integration of Building Information Models and Scene Graphs for Semantic-Geometric Human-Robot Understanding 

**Authors**: Antonello Longo, Chanyoung Chung, Matteo Palieri, Sung-Kyun Kim, Ali Agha, Cataldo Guaragnella, Shehryar Khattak  

**Link**: [PDF](https://arxiv.org/pdf/2506.22593)  

**Abstract**: Autonomous robots are increasingly playing key roles as support platforms for human operators in high-risk, dangerous applications. To accomplish challenging tasks, an efficient human-robot cooperation and understanding is required. While typically robotic planning leverages 3D geometric information, human operators are accustomed to a high-level compact representation of the environment, like top-down 2D maps representing the Building Information Model (BIM). 3D scene graphs have emerged as a powerful tool to bridge the gap between human readable 2D BIM and the robot 3D maps. In this work, we introduce Pixels-to-Graph (Pix2G), a novel lightweight method to generate structured scene graphs from image pixels and LiDAR maps in real-time for the autonomous exploration of unknown environments on resource-constrained robot platforms. To satisfy onboard compute constraints, the framework is designed to perform all operation on CPU only. The method output are a de-noised 2D top-down environment map and a structure-segmented 3D pointcloud which are seamlessly connected using a multi-layer graph abstracting information from object-level up to the building-level. The proposed method is quantitatively and qualitatively evaluated during real-world experiments performed using the NASA JPL NeBula-Spot legged robot to autonomously explore and map cluttered garage and urban office like environments in real-time. 

---
# FedCLAM: Client Adaptive Momentum with Foreground Intensity Matching for Federated Medical Image Segmentation 

**Authors**: Vasilis Siomos, Jonathan Passerat-Palmbach, Giacomo Tarroni  

**Link**: [PDF](https://arxiv.org/pdf/2506.22580)  

**Abstract**: Federated learning is a decentralized training approach that keeps data under stakeholder control while achieving superior performance over isolated training. While inter-institutional feature discrepancies pose a challenge in all federated settings, medical imaging is particularly affected due to diverse imaging devices and population variances, which can diminish the global model's effectiveness. Existing aggregation methods generally fail to adapt across varied circumstances. To address this, we propose FedCLAM, which integrates \textit{client-adaptive momentum} terms derived from each client's loss reduction during local training, as well as a \textit{personalized dampening factor} to curb overfitting. We further introduce a novel \textit{intensity alignment} loss that matches predicted and ground-truth foreground distributions to handle heterogeneous image intensity profiles across institutions and devices. Extensive evaluations on two datasets show that FedCLAM surpasses eight cutting-edge methods in medical segmentation tasks, underscoring its efficacy. The code is available at this https URL. 

---
# The Hidden Link Between RLHF and Contrastive Learning 

**Authors**: Xufei Lv, Haoyuan Sun, Xuefeng Bai, Min Zhang, Houde Liu, Kehai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.22578)  

**Abstract**: Alignment of large language models (LLMs) with human values has recently garnered significant attention, with prominent examples including the canonical yet costly Reinforcement Learning from Human Feedback (RLHF) and the simple Direct Preference Optimization (DPO). In this work, we demonstrate that both RLHF and DPO can be interpreted from the perspective of mutual information (MI) maximization, uncovering a profound connection to contrastive learning. Within this framework, both RLHF and DPO can be viewed as methods that perform contrastive learning based on the positive and negative samples derived from the base model, leveraging the Donsker-Varadhan (DV) lower bound on MI (equivalently, the MINE estimator). This paradigm further explains why RLHF may not intrinsically incentivize reasoning capacities in LLMs beyond what is already present in the base model. Building on this perspective, we replace the DV/MINE bound with the Jensen-Shannon MI estimator and propose Mutual Information Optimization (MIO). Comprehensive theoretical analysis and extensive empirical evaluations demonstrate that MIO mitigates the late-stage decline in chosen-likelihood observed in DPO, achieving competitive or superior performance across various challenging reasoning and mathematical benchmarks. We will release the model and code upon acceptance. 

---
# Unifying Biomedical Vision-Language Expertise: Towards a Generalist Foundation Model via Multi-CLIP Knowledge Distillation 

**Authors**: Shansong Wang, Zhecheng Jin, Mingzhe Hu, Mojtaba Safari, Feng Zhao, Chih-Wei Chang, Richard LJ Qiu, Justin Roper, David S. Yu, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22567)  

**Abstract**: CLIP models pretrained on natural images with billion-scale image-text pairs have demonstrated impressive capabilities in zero-shot classification, cross-modal retrieval, and open-ended visual answering. However, transferring this success to biomedicine is hindered by the scarcity of large-scale biomedical image-text corpora, the heterogeneity of image modalities, and fragmented data standards across institutions. These limitations hinder the development of a unified and generalizable biomedical foundation model trained from scratch. To overcome this, we introduce MMKD-CLIP, a generalist biomedical foundation model developed via Multiple Medical CLIP Knowledge Distillation. Rather than relying on billion-scale raw data, MMKD-CLIP distills knowledge from nine state-of-the-art domain-specific or generalist biomedical CLIP models, each pretrained on millions of biomedical image-text pairs. Our two-stage training pipeline first performs CLIP-style pretraining on over 2.9 million biomedical image-text pairs from 26 image modalities, followed by feature-level distillation using over 19.2 million feature pairs extracted from teacher models. We evaluate MMKD-CLIP on 58 diverse biomedical datasets, encompassing over 10.8 million biomedical images across nine image modalities. The evaluation spans six core task types: zero-shot classification, linear probing, cross-modal retrieval, visual question answering, survival prediction, and cancer diagnosis. MMKD-CLIP consistently outperforms all teacher models while demonstrating remarkable robustness and generalization across image domains and task settings. These results underscore that multi-teacher knowledge distillation is a scalable and effective paradigm for building high-performing biomedical foundation models under the practical constraints of real-world data availability. 

---
# Exploration Behavior of Untrained Policies 

**Authors**: Jacob Adamczyk  

**Link**: [PDF](https://arxiv.org/pdf/2506.22566)  

**Abstract**: Exploration remains a fundamental challenge in reinforcement learning (RL), particularly in environments with sparse or adversarial reward structures. In this work, we study how the architecture of deep neural policies implicitly shapes exploration before training. We theoretically and empirically demonstrate strategies for generating ballistic or diffusive trajectories from untrained policies in a toy model. Using the theory of infinite-width networks and a continuous-time limit, we show that untrained policies return correlated actions and result in non-trivial state-visitation distributions. We discuss the distributions of the corresponding trajectories for a standard architecture, revealing insights into inductive biases for tackling exploration. Our results establish a theoretical and experimental framework for using policy initialization as a design tool to understand exploration behavior in early training. 

---
# Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset 

**Authors**: Vasu Agrawal, Akinniyi Akinyemi, Kathryn Alvero, Morteza Behrooz, Julia Buffalini, Fabio Maria Carlucci, Joy Chen, Junming Chen, Zhang Chen, Shiyang Cheng, Praveen Chowdary, Joe Chuang, Antony D'Avirro, Jon Daly, Ning Dong, Mark Duppenthaler, Cynthia Gao, Jeff Girard, Martin Gleize, Sahir Gomez, Hongyu Gong, Srivathsan Govindarajan, Brandon Han, Sen He, Denise Hernandez, Yordan Hristov, Rongjie Huang, Hirofumi Inaguma, Somya Jain, Raj Janardhan, Qingyao Jia, Christopher Klaiber, Dejan Kovachev, Moneish Kumar, Hang Li, Yilei Li, Pavel Litvin, Wei Liu, Guangyao Ma, Jing Ma, Martin Ma, Xutai Ma, Lucas Mantovani, Sagar Miglani, Sreyas Mohan, Louis-Philippe Morency, Evonne Ng, Kam-Woh Ng, Tu Anh Nguyen, Amia Oberai, Benjamin Peloquin, Juan Pino, Jovan Popovic, Omid Poursaeed, Fabian Prada, Alice Rakotoarison, Alexander Richard, Christophe Ropers, Safiyyah Saleem, Vasu Sharma, Alex Shcherbyna, Jia Shen, Jie Shen, Anastasis Stathopoulos, Anna Sun, Paden Tomasello, Tuan Tran, Arina Turkatenko, Bo Wan, Chao Wang, Jeff Wang, Mary Williamson, Carleigh Wood, Tao Xiang, Yilin Yang, Julien Yao, Chen Zhang, Jiemin Zhang, Xinyue Zhang, Jason Zheng, Pavlo Zhyzheria, Jan Zikes, Michael Zollhoefer  

**Link**: [PDF](https://arxiv.org/pdf/2506.22554)  

**Abstract**: Human communication involves a complex interplay of verbal and nonverbal signals, essential for conveying meaning and achieving interpersonal goals. To develop socially intelligent AI technologies, it is crucial to develop models that can both comprehend and generate dyadic behavioral dynamics. To this end, we introduce the Seamless Interaction Dataset, a large-scale collection of over 4,000 hours of face-to-face interaction footage from over 4,000 participants in diverse contexts. This dataset enables the development of AI technologies that understand dyadic embodied dynamics, unlocking breakthroughs in virtual agents, telepresence experiences, and multimodal content analysis tools. We also develop a suite of models that utilize the dataset to generate dyadic motion gestures and facial expressions aligned with human speech. These models can take as input both the speech and visual behavior of their interlocutors. We present a variant with speech from an LLM model and integrations with 2D and 3D rendering methods, bringing us closer to interactive virtual agents. Additionally, we describe controllable variants of our motion models that can adapt emotional responses and expressivity levels, as well as generating more semantically-relevant gestures. Finally, we discuss methods for assessing the quality of these dyadic motion models, which are demonstrating the potential for more intuitive and responsive human-AI interactions. 

---
# Correlated Mutations for Integer Programming 

**Authors**: Ofer M. Shir, Michael Emmerich  

**Link**: [PDF](https://arxiv.org/pdf/2506.22526)  

**Abstract**: Even with the recent theoretical advancements that dramatically reduced the complexity of Integer Programming (IP), heuristics remain the dominant problem-solvers for this difficult category. This study seeks to establish the groundwork for Integer Evolution Strategies (IESs), a class of randomized search heuristics inherently designed for continuous spaces. IESs already excel in treating IP in practice, but accomplish it via discretization and by applying sophisticated patches to their continuous operators, while persistently using the $\ell_2$-norm as their operation pillar. We lay foundations for discrete search, by adopting the $\ell_1$-norm, accounting for the suitable step-size, and questioning alternative measures to quantify correlations over the integer lattice. We focus on mutation distributions for unbounded integer decision variables. We briefly discuss a couple of candidate discrete probabilities induced by the uniform and binomial distributions, which we show to possess less appealing theoretical properties, and then narrow down to the Truncated Normal (TN) and Double Geometric (DG) distributions. We explore their theoretical properties, including entropy functions, and propose a procedure to generate scalable correlated mutation distributions. Our investigations are accompanied by extensive numerical simulations, which consistently support the claim that the DG distribution is better suited for unbounded integer search. We link our theoretical perspective to empirical evidence indicating that an IES with correlated DG mutations outperformed other strategies over non-separable quadratic IP. We conclude that while the replacement of the default TN distribution by the DG is theoretically justified and practically beneficial, the truly crucial change lies in adopting the $\ell_1$-norm over the $\ell_2$-norm. 

---
# Red Teaming for Generative AI, Report on a Copyright-Focused Exercise Completed in an Academic Medical Center 

**Authors**: James Wen, Sahil Nalawade, Zhiwei Liang, Catherine Bielick, Marisa Ferrara Boston, Alexander Chowdhury, Adele Collin, Luigi De Angelis, Jacob Ellen, Heather Frase, Rodrigo R. Gameiro, Juan Manuel Gutierrez, Pooja Kadam, Murat Keceli, Srikanth Krishnamurthy, Anne Kwok, Yanan Lance Lu, Heather Mattie, Liam G. McCoy, Katherine Miller, Allison C. Morgan, Marlene Louisa Moerig, Trang Nguyen, Alexander Owen-Post, Alex D. Ruiz, Sreekar Reddy Puchala, Soujanya Samineni, Takeshi Tohyama, Varun Ullanat, Carmine Valenza, Camilo Velez, Pengcheng Wang, Anna Wuest, Yuxiang Zhou, Yingde Zhu, Jason M. Johnson, Jennifer Willcox, Francis J. Vitiello, Leo Anthony G. Celi, Renato Umeton  

**Link**: [PDF](https://arxiv.org/pdf/2506.22523)  

**Abstract**: Generative AI is present in multiple industries. Dana-Farber Cancer Institute, in partnership with Microsoft, has created an internal AI tool, GPT4DFCI. Together we hosted a red teaming event to assess whether the underlying GPT models that support the tool would output copyrighted data. Our teams focused on reproducing content from books, news articles, scientific articles, and electronic health records. We found isolated instances where GPT4DFCI was able to identify copyrighted material and reproduce exact quotes from famous books which indicates that copyrighted material was in the training data. The model was not able to reproduce content from our target news article, scientific article, or electronic health records. However, there were instances of fabrication. As a result of this event, a mitigation strategy is in production in GPT4DFCI v2.8.2, deployed on January 21, 2025. We hope this report leads to similar events in which AI software tools are stress-tested to assess the perimeter of their legal and ethical usage. 

---
# A Survey on Model Extraction Attacks and Defenses for Large Language Models 

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.22521)  

**Abstract**: Model extraction attacks pose significant security threats to deployed language models, potentially compromising intellectual property and user privacy. This survey provides a comprehensive taxonomy of LLM-specific extraction attacks and defenses, categorizing attacks into functionality extraction, training data extraction, and prompt-targeted attacks. We analyze various attack methodologies including API-based knowledge distillation, direct querying, parameter recovery, and prompt stealing techniques that exploit transformer architectures. We then examine defense mechanisms organized into model protection, data privacy protection, and prompt-targeted strategies, evaluating their effectiveness across different deployment scenarios. We propose specialized metrics for evaluating both attack effectiveness and defense performance, addressing the specific challenges of generative language models. Through our analysis, we identify critical limitations in current approaches and propose promising research directions, including integrated attack methodologies and adaptive defense mechanisms that balance security with model utility. This work serves NLP researchers, ML engineers, and security professionals seeking to protect language models in production environments. 

---
# Exploring Artificial Intelligence Tutor Teammate Adaptability to Harness Discovery Curiosity and Promote Learning in the Context of Interactive Molecular Dynamics 

**Authors**: Mustafa Demir, Jacob Miratsky, Jonathan Nguyen, Chun Kit Chan, Punya Mishra, Abhishek Singharoy  

**Link**: [PDF](https://arxiv.org/pdf/2506.22520)  

**Abstract**: This study examines the impact of an Artificial Intelligence tutor teammate (AI) on student curiosity-driven engagement and learning effectiveness during Interactive Molecular Dynamics (IMD) tasks on the Visual Molecular Dynamics platform. It explores the role of the AI's curiosity-triggering and response behaviors in stimulating and sustaining student curiosity, affecting the frequency and complexity of student-initiated questions. The study further assesses how AI interventions shape student engagement, foster discovery curiosity, and enhance team performance within the IMD learning environment. Using a Wizard-of-Oz paradigm, a human experimenter dynamically adjusts the AI tutor teammate's behavior through a large language model. By employing a mixed-methods exploratory design, a total of 11 high school students participated in four IMD tasks that involved molecular visualization and calculations, which increased in complexity over a 60-minute period. Team performance was evaluated through real-time observation and recordings, whereas team communication was measured by question complexity and AI's curiosity-triggering and response behaviors. Cross Recurrence Quantification Analysis (CRQA) metrics reflected structural alignment in coordination and were linked to communication behaviors. High-performing teams exhibited superior task completion, deeper understanding, and increased engagement. Advanced questions were associated with AI curiosity-triggering, indicating heightened engagement and cognitive complexity. CRQA metrics highlighted dynamic synchronization in student-AI interactions, emphasizing structured yet adaptive engagement to promote curiosity. These proof-of-concept findings suggest that the AI's dual role as a teammate and educator indicates its capacity to provide adaptive feedback, sustaining engagement and epistemic curiosity. 

---
# Weak-to-Strong GraphRAG: Aligning Weak Retrievers with Large Language Models for Graph-based Retrieval Augmented Generation 

**Authors**: Deyu Zou, Yongqiang Chen, Mufei Li, Siqi Miao, Chenxi Liu, Bo Han, James Cheng, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22518)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) enables large language models (LLMs) to ground responses with structured external knowledge from up-to-date knowledge graphs (KGs) and reduce hallucinations. However, LLMs often rely on a weak retriever in graph-based RAG: I) Due to the lack of ground truth, the retriever is often trained on weak supervision, which often introduces spurious signals to the LLMs. II) Due to the abstraction of graph data, the retrieved knowledge is often presented in unorganized forms. To mitigate the issue, we present Refined Graph-based RAG (ReG) to align weak retrievers to LLMs for graph-based RAG. Specifically, ReG incorporates LLM feedback to get rid of spurious signals and improve the quality of the supervision. Meanwhile, ReG introduces a structure-aware reorganization module to refactor the retrieval results into logically coherent evidence chains. Experiments on prominent benchmarks demonstrate that ReG significantly and consistently brings improvements across different LLM backbones by up to 10%. The improved supervision quality enables ReG to match the state-of-the-art performance with 5% training data and to transfer to out-of-distribution KGs. Notably, when adopted to reasoning-based LLMs, ReG reduces the reasoning token cost by up to 30% and improves the performance by up to 4%. 

---
# Can "consciousness" be observed from large language model (LLM) internal states? Dissecting LLM representations obtained from Theory of Mind test with Integrated Information Theory and Span Representation analysis 

**Authors**: Jingkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22516)  

**Abstract**: Integrated Information Theory (IIT) provides a quantitative framework for explaining consciousness phenomenon, positing that conscious systems comprise elements integrated through causal properties. We apply IIT 3.0 and 4.0 -- the latest iterations of this framework -- to sequences of Large Language Model (LLM) representations, analyzing data derived from existing Theory of Mind (ToM) test results. Our study systematically investigates whether the differences of ToM test performances, when presented in the LLM representations, can be revealed by IIT estimates, i.e., $\Phi^{\max}$ (IIT 3.0), $\Phi$ (IIT 4.0), Conceptual Information (IIT 3.0), and $\Phi$-structure (IIT 4.0). Furthermore, we compare these metrics with the Span Representations independent of any estimate for consciousness. This additional effort aims to differentiate between potential "consciousness" phenomena and inherent separations within LLM representational space. We conduct comprehensive experiments examining variations across LLM transformer layers and linguistic spans from stimuli. Our results suggest that sequences of contemporary Transformer-based LLM representations lack statistically significant indicators of observed "consciousness" phenomena but exhibit intriguing patterns under $\textit{spatio}$-permutational analyses. The Appendix and code are available as Supplementary Materials at: this https URL. 

---
# In-context learning for the classification of manipulation techniques in phishing emails 

**Authors**: Antony Dalmiere, Guillaume Auriol, Vincent Nicomette, Pascal Marchand  

**Link**: [PDF](https://arxiv.org/pdf/2506.22515)  

**Abstract**: Traditional phishing detection often overlooks psychological manipulation. This study investigates using Large Language Model (LLM) In-Context Learning (ICL) for fine-grained classification of phishing emails based on a taxonomy of 40 manipulation techniques. Using few-shot examples with GPT-4o-mini on real-world French phishing emails (SignalSpam), we evaluated performance against a human-annotated test set (100 emails). The approach effectively identifies prevalent techniques (e.g., Baiting, Curiosity Appeal, Request For Minor Favor) with a promising accuracy of 0.76. This work demonstrates ICL's potential for nuanced phishing analysis and provides insights into attacker strategies. 

---
# Ask before you Build: Rethinking AI-for-Good in Human Trafficking Interventions 

**Authors**: Pratheeksha Nair, Gabriel Lefebvre, Sophia Garrel, Maryam Molamohammadi, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2506.22512)  

**Abstract**: AI for good initiatives often rely on the assumption that technical interventions can resolve complex social problems. In the context of human trafficking (HT), such techno-solutionism risks oversimplifying exploitation, reinforcing power imbalances and causing harm to the very communities AI claims to support. In this paper, we introduce the Radical Questioning (RQ) framework as a five step, pre-project ethical assessment tool to critically evaluate whether AI should be built at all, especially in domains involving marginalized populations and entrenched systemic injustice. RQ does not replace principles based ethics but precedes it, offering an upstream, deliberative space to confront assumptions, map power, and consider harms before design. Using a case study in AI for HT, we demonstrate how RQ reveals overlooked sociocultural complexities and guides us away from surveillance based interventions toward survivor empowerment tools. While developed in the context of HT, RQ's five step structure can generalize to other domains, though the specific questions must be contextual. This paper situates RQ within a broader AI ethics philosophy that challenges instrumentalist norms and centers relational, reflexive responsibility. 

---
# Lightning the Night with Generative Artificial Intelligence 

**Authors**: Tingting Zhou, Feng Zhang, Haoyang Fu, Baoxiang Pan, Renhe Zhang, Feng Lu, Zhixin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22511)  

**Abstract**: The visible light reflectance data from geostationary satellites is crucial for meteorological observations and plays an important role in weather monitoring and forecasting. However, due to the lack of visible light at night, it is impossible to conduct continuous all-day weather observations using visible light reflectance data. This study pioneers the use of generative diffusion models to address this limitation. Based on the multi-band thermal infrared brightness temperature data from the Advanced Geostationary Radiation Imager (AGRI) onboard the Fengyun-4B (FY4B) geostationary satellite, we developed a high-precision visible light reflectance retrieval model, called Reflectance Diffusion (RefDiff), which enables 0.47~\mu\mathrm{m}, 0.65~\mu\mathrm{m}, and 0.825~\mu\mathrm{m} bands visible light reflectance retrieval at night. Compared to the classical models, RefDiff not only significantly improves accuracy through ensemble averaging but also provides uncertainty estimation. Specifically, the SSIM index of RefDiff can reach 0.90, with particularly significant improvements in areas with complex cloud structures and thick clouds. The model's nighttime retrieval capability was validated using VIIRS nighttime product, demonstrating comparable performance to its daytime counterpart. In summary, this research has made substantial progress in the ability to retrieve visible light reflectance at night, with the potential to expand the application of nighttime visible light data. 

---
# Towards Text-free Graph Foundation Models: Rethinking Multi-Domain Graph Contrastive Learning 

**Authors**: Zihao Zhao, Xinlong Zhai, Jinyu Yang, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.22510)  

**Abstract**: Foundation models have achieved great success in natural language processing (NLP) and computer vision (CV). Their success largely stems from the ability to integrate multi-domain knowledge in pre-training and transfer it to target domains. Considering graph data, especially graphs without textual features, is ubiquitous in real-world applications such as social networks and recommendation systems, some researchers have attempted to extend this paradigm to the graph field, aiming to construct graph foundation models. However, unlike CV and NLP, there are huge gaps among the semantics and properties of graphs in different domains, while current works still adopt traditional contrastive pre-training strategies designed in the single-domain scenario, which regard contrastive samples from different domains as equivalent. From experimental investigations, we discovered that inherent domain-specific differences prevent these strategies from effectively absorbing knowledge from different domains to generate informative representations. In this paper, we propose a novel multi-domain pre-training and cross-domain transfer framework, namely this http URL the pre-training stage, we design a contrastive learning strategy to substantially recognize and capture domain differences, and introduce domain tokens to encode domain-level global information. In the downstream stage, we introduce a domain attention mechanism to enable fine-grained domain knowledge transfer. Extensive experiments on five benchmark datasets have demonstrated that our method outperforms state-of-the-art significantly, with the maximum improvement of 19.33\% on accuracy and 19.13\% on Macro-F1 score. 

---
# FreeDNA: Endowing Domain Adaptation of Diffusion-Based Dense Prediction with Training-Free Domain Noise Alignment 

**Authors**: Hang Xu, Jie Huang, Linjiang Huang, Dong Li, Yidi Liu, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.22509)  

**Abstract**: Domain Adaptation(DA) for dense prediction tasks is an important topic, which enhances the dense prediction model's performance when tested on its unseen domain. Recently, with the development of Diffusion-based Dense Prediction (DDP) models, the exploration of DA designs tailored to this framework is worth exploring, since the diffusion model is effective in modeling the distribution transformation that comprises domain information. In this work, we propose a training-free mechanism for DDP frameworks, endowing them with DA capabilities. Our motivation arises from the observation that the exposure bias (e.g., noise statistics bias) in diffusion brings domain shift, and different domains in conditions of DDP models can also be effectively captured by the noise prediction statistics. Based on this, we propose a training-free Domain Noise Alignment (DNA) approach, which alleviates the variations of noise statistics to domain changes during the diffusion sampling process, thereby achieving domain adaptation. Specifically, when the source domain is available, we directly adopt the DNA method to achieve domain adaptation by aligning the noise statistics of the target domain with those of the source domain. For the more challenging source-free DA, inspired by the observation that regions closer to the source domain exhibit higher confidence meeting variations of sampling noise, we utilize the statistics from the high-confidence regions progressively to guide the noise statistic adjustment during the sampling process. Notably, our method demonstrates the effectiveness of enhancing the DA capability of DDP models across four common dense prediction tasks. Code is available at \href{this https URL}{this https URL}. 

---
# AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text 

**Authors**: Chenyang Shao, Tianxing Li, Chenhao Pu, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22508)  

**Abstract**: In today's digital world, casual user-generated content often contains subtle cues that may inadvertently expose sensitive personal attributes. Such risks underscore the growing importance of effective text anonymization to safeguard individual privacy. However, existing methods either rely on rigid replacements that damage utility or cloud-based LLMs that are costly and pose privacy risks. To address these issues, we explore the use of locally deployed smaller-scale language models (SLMs) for anonymization. Yet training effective SLMs remains challenging due to limited high-quality supervision. To address the challenge, we propose AgentStealth, a self-reinforcing LLM anonymization this http URL, we introduce an adversarial anonymization workflow enhanced by In-context Contrastive Learning and Adaptive Utility-Aware Control. Second, we perform supervised adaptation of SLMs using high-quality data collected from the workflow, which includes both anonymization and attack signals. Finally, we apply online reinforcement learning where the model leverages its internal adversarial feedback to iteratively improve anonymization performance. Experiments on two datasets show that our method outperforms baselines in both anonymization effectiveness (+12.3%) and utility (+6.8%). Our lightweight design supports direct deployment on edge devices, avoiding cloud reliance and communication-based privacy risks. Our code is open-source at this https URL. 

---
# SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning 

**Authors**: Momin Ahmad Khan, Yasra Chandio, Fatima Muhammad Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2506.22506)  

**Abstract**: Federated Prompt Learning has emerged as a communication-efficient and privacy-preserving paradigm for adapting large vision-language models like CLIP across decentralized clients. However, the security implications of this setup remain underexplored. In this work, we present the first study of backdoor attacks in Federated Prompt Learning. We show that when malicious clients inject visually imperceptible, learnable noise triggers into input images, the global prompt learner becomes vulnerable to targeted misclassification while still maintaining high accuracy on clean inputs. Motivated by this vulnerability, we propose SABRE-FL, a lightweight, modular defense that filters poisoned prompt updates using an embedding-space anomaly detector trained offline on out-of-distribution data. SABRE-FL requires no access to raw client data or labels and generalizes across diverse datasets. We show, both theoretically and empirically, that malicious clients can be reliably identified and filtered using an embedding-based detector. Across five diverse datasets and four baseline defenses, SABRE-FL outperforms all baselines by significantly reducing backdoor accuracy while preserving clean accuracy, demonstrating strong empirical performance and underscoring the need for robust prompt learning in future federated systems. 

---
# How Can Multimodal Remote Sensing Datasets Transform Classification via SpatialNet-ViT? 

**Authors**: Gautam Siddharth Kashyap, Manaswi Kulahara, Nipun Joshi, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.22501)  

**Abstract**: Remote sensing datasets offer significant promise for tackling key classification tasks such as land-use categorization, object presence detection, and rural/urban classification. However, many existing studies tend to focus on narrow tasks or datasets, which limits their ability to generalize across various remote sensing classification challenges. To overcome this, we propose a novel model, SpatialNet-ViT, leveraging the power of Vision Transformers (ViTs) and Multi-Task Learning (MTL). This integrated approach combines spatial awareness with contextual understanding, improving both classification accuracy and scalability. Additionally, techniques like data augmentation, transfer learning, and multi-task learning are employed to enhance model robustness and its ability to generalize across diverse datasets 

---
# Visual-Semantic Knowledge Conflicts in Operating Rooms: Synthetic Data Curation for Surgical Risk Perception in Multimodal Large Language Models 

**Authors**: Weiyi Zhao, Xiaoyu Tan, Liang Liu, Sijia Li, Youwei Song, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22500)  

**Abstract**: Surgical risk identification is critical for patient safety and reducing preventable medical errors. While multimodal large language models (MLLMs) show promise for automated operating room (OR) risk detection, they often exhibit visual-semantic knowledge conflicts (VS-KC), failing to identify visual safety violations despite understanding textual rules. To address this, we introduce a dataset comprising over 34,000 synthetic images generated by diffusion models, depicting operating room scenes containing entities that violate established safety rules. These images were created to alleviate data scarcity and examine MLLMs vulnerabilities. In addition, the dataset includes 214 human-annotated images that serve as a gold-standard reference for validation. This comprehensive dataset, spanning diverse perspectives, stages, and configurations, is designed to expose and study VS-KC. Fine-tuning on OR-VSKC significantly improves MLLMs' detection of trained conflict entities and generalizes well to new viewpoints for these entities, but performance on untrained entity types remains poor, highlighting learning specificity and the need for comprehensive training. The main contributions of this work include: (1) a data generation methodology tailored for rule-violation scenarios; (2) the release of the OR-VSKC dataset and its associated benchmark as open-source resources; and (3) an empirical analysis of violation-sensitive knowledge consistency in representative MLLMs. The dataset and appendix are available at this https URL. 

---
# Scalable Dynamic Origin-Destination Demand Estimation Enhanced by High-Resolution Satellite Imagery Data 

**Authors**: Jiachao Liu, Pablo Guarda, Koichiro Niinuma, Sean Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.22499)  

**Abstract**: This study presents a novel integrated framework for dynamic origin-destination demand estimation (DODE) in multi-class mesoscopic network models, leveraging high-resolution satellite imagery together with conventional traffic data from local sensors. Unlike sparse local detectors, satellite imagery offers consistent, city-wide road and traffic information of both parking and moving vehicles, overcoming data availability limitations. To extract information from imagery data, we design a computer vision pipeline for class-specific vehicle detection and map matching, generating link-level traffic density observations by vehicle class. Building upon this information, we formulate a computational graph-based DODE model that calibrates dynamic network states by jointly matching observed traffic counts and travel times from local sensors with density measurements derived from satellite imagery. To assess the accuracy and scalability of the proposed framework, we conduct a series of numerical experiments using both synthetic and real-world data. The results of out-of-sample tests demonstrate that supplementing traditional data with satellite-derived density significantly improves estimation performance, especially for links without local sensors. Real-world experiments also confirm the framework's capability to handle large-scale networks, supporting its potential for practical deployment in cities of varying sizes. Sensitivity analysis further evaluates the impact of data quality related to satellite imagery data. 

---
# ViFusionTST: Deep Fusion of Time-Series Image Representations from Load Signals for Early Bed-Exit Prediction 

**Authors**: Hao Liu, Yu Hu, Rakiba Rayhana, Ling Bai, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22498)  

**Abstract**: Bed-related falls remain a leading source of injury in hospitals and long-term-care facilities, yet many commercial alarms trigger only after a patient has already left the bed. We show that early bed-exit intent can be predicted using only four low-cost load cells mounted under the bed legs. The resulting load signals are first converted into a compact set of complementary images: an RGB line plot that preserves raw waveforms and three texture maps - recurrence plot, Markov transition field, and Gramian angular field - that expose higher-order dynamics. We introduce ViFusionTST, a dual-stream Swin Transformer that processes the line plot and texture maps in parallel and fuses them through cross-attention to learn data-driven modality weights.
To provide a realistic benchmark, we collected six months of continuous data from 95 beds in a long-term-care facility. On this real-world dataset ViFusionTST reaches an accuracy of 0.885 and an F1 score of 0.794, surpassing recent 1D and 2D time-series baselines across F1, recall, accuracy, and AUPRC. The results demonstrate that image-based fusion of load-sensor signals for time series classification is a practical and effective solution for real-time, privacy-preserving fall prevention. 

---
# Peer Review as Structured Commentary: Immutable Identity, Public Dialogue, and Reproducible Scholarship 

**Authors**: Craig Steven Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.22497)  

**Abstract**: This paper reconceptualises peer review as structured public commentary. Traditional academic validation is hindered by anonymity, latency, and gatekeeping. We propose a transparent, identity-linked, and reproducible system of scholarly evaluation anchored in open commentary. Leveraging blockchain for immutable audit trails and AI for iterative synthesis, we design a framework that incentivises intellectual contribution, captures epistemic evolution, and enables traceable reputational dynamics. This model empowers fields from computational science to the humanities, reframing academic knowledge as a living process rather than a static credential. 

---
# Mitigating Gambling-Like Risk-Taking Behaviors in Large Language Models: A Behavioral Economics Approach to AI Safety 

**Authors**: Y. Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.22496)  

**Abstract**: Large Language Models (LLMs) exhibit systematic risk-taking behaviors analogous to those observed in gambling psychology, including overconfidence bias, loss-chasing tendencies, and probability misjudgment. Drawing from behavioral economics and prospect theory, we identify and formalize these "gambling-like" patterns where models sacrifice accuracy for high-reward outputs, exhibit escalating risk-taking after errors, and systematically miscalibrate uncertainty. We propose the Risk-Aware Response Generation (RARG) framework, incorporating insights from gambling research to address these behavioral biases through risk-calibrated training, loss-aversion mechanisms, and uncertainty-aware decision making. Our approach introduces novel evaluation paradigms based on established gambling psychology experiments, including AI adaptations of the Iowa Gambling Task and probability learning assessments. Experimental results demonstrate measurable reductions in gambling-like behaviors: 18.7\% decrease in overconfidence bias, 24.3\% reduction in loss-chasing tendencies, and improved risk calibration across diverse scenarios. This work establishes the first systematic framework for understanding and mitigating gambling psychology patterns in AI systems. 

---
# Masked Autoencoders that Feel the Heart: Unveiling Simplicity Bias for ECG Analyses 

**Authors**: He-Yang Xu, Hongxiang Gao, Yuwen Li, Xiu-Shen Wei, Chengyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22495)  

**Abstract**: The diagnostic value of electrocardiogram (ECG) lies in its dynamic characteristics, ranging from rhythm fluctuations to subtle waveform deformations that evolve across time and frequency domains. However, supervised ECG models tend to overfit dominant and repetitive patterns, overlooking fine-grained but clinically critical cues, a phenomenon known as Simplicity Bias (SB), where models favor easily learnable signals over subtle but informative ones. In this work, we first empirically demonstrate the presence of SB in ECG analyses and its negative impact on diagnostic performance, while simultaneously discovering that self-supervised learning (SSL) can alleviate it, providing a promising direction for tackling the bias. Following the SSL paradigm, we propose a novel method comprising two key components: 1) Temporal-Frequency aware Filters to capture temporal-frequency features reflecting the dynamic characteristics of ECG signals, and 2) building on this, Multi-Grained Prototype Reconstruction for coarse and fine representation learning across dual domains, further mitigating SB. To advance SSL in ECG analyses, we curate a large-scale multi-site ECG dataset with 1.53 million recordings from over 300 clinical centers. Experiments on three downstream tasks across six ECG datasets demonstrate that our method effectively reduces SB and achieves state-of-the-art performance. Code and dataset will be released publicly. 

---
# Report on NSF Workshop on Science of Safe AI 

**Authors**: Rajeev Alur, Greg Durrett, Hadas Kress-Gazit, Corina Păsăreanu, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2506.22492)  

**Abstract**: Recent advances in machine learning, particularly the emergence of foundation models, are leading to new opportunities to develop technology-based solutions to societal problems. However, the reasoning and inner workings of today's complex AI models are not transparent to the user, and there are no safety guarantees regarding their predictions. Consequently, to fulfill the promise of AI, we must address the following scientific challenge: how to develop AI-based systems that are not only accurate and performant but also safe and trustworthy?
The criticality of safe operation is particularly evident for autonomous systems for control and robotics, and was the catalyst for the Safe Learning Enabled Systems (SLES) program at NSF. For the broader class of AI applications, such as users interacting with chatbots and clinicians receiving treatment recommendations, safety is, while no less important, less well-defined with context-dependent interpretations. This motivated the organization of a day-long workshop, held at University of Pennsylvania on February 26, 2025, to bring together investigators funded by the NSF SLES program with a broader pool of researchers studying AI safety. This report is the result of the discussions in the working groups that addressed different aspects of safety at the workshop. The report articulates a new research agenda focused on developing theory, methods, and tools that will provide the foundations of the next generation of AI-enabled systems. 

---
# PromptAug: Fine-grained Conflict Classification Using Data Augmentation 

**Authors**: Oliver Warke, Joemon M. Jose, Faegheh Hasibi, Jan Breitsohl  

**Link**: [PDF](https://arxiv.org/pdf/2506.22491)  

**Abstract**: Given the rise of conflicts on social media, effective classification models to detect harmful behaviours are essential. Following the garbage-in-garbage-out maxim, machine learning performance depends heavily on training data quality. However, high-quality labelled data, especially for nuanced tasks like identifying conflict behaviours, is limited, expensive, and difficult to obtain. Additionally, as social media platforms increasingly restrict access to research data, text data augmentation is gaining attention as an alternative to generate training data. Augmenting conflict-related data poses unique challenges due to Large Language Model (LLM) guardrails that prevent generation of offensive content. This paper introduces PromptAug, an innovative LLM-based data augmentation method. PromptAug achieves statistically significant improvements of 2% in both accuracy and F1-score on conflict and emotion datasets. To thoroughly evaluate PromptAug against other data augmentation methods we conduct a robust evaluation using extreme data scarcity scenarios, quantitative diversity analysis and a qualitative thematic analysis. The thematic analysis identifies four problematic patterns in augmented text: Linguistic Fluidity, Humour Ambiguity, Augmented Content Ambiguity, and Augmented Content Misinterpretation.
Overall, this work presents PromptAug as an effective method for augmenting data in sensitive tasks like conflict detection, offering a unique, interdisciplinary evaluation grounded in both natural language processing and social science methodology. 

---
# AGI Enabled Solutions For IoX Layers Bottlenecks In Cyber-Physical-Social-Thinking Space 

**Authors**: Amar Khelloufi, Huansheng Ning, Sahraoui Dhelim, Jianguo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.22487)  

**Abstract**: The integration of the Internet of Everything (IoX) and Artificial General Intelligence (AGI) has given rise to a transformative paradigm aimed at addressing critical bottlenecks across sensing, network, and application layers in Cyber-Physical-Social Thinking (CPST) ecosystems. In this survey, we provide a systematic and comprehensive review of AGI-enhanced IoX research, focusing on three key components: sensing-layer data management, network-layer protocol optimization, and application-layer decision-making frameworks. Specifically, this survey explores how AGI can mitigate IoX bottlenecks challenges by leveraging adaptive sensor fusion, edge preprocessing, and selective attention mechanisms at the sensing layer, while resolving network-layer issues such as protocol heterogeneity and dynamic spectrum management, neuro-symbolic reasoning, active inference, and causal reasoning, Furthermore, the survey examines AGI-enabled frameworks for managing identity and relationship explosion. Key findings suggest that AGI-driven strategies, such as adaptive sensor fusion, edge preprocessing, and semantic modeling, offer novel solutions to sensing-layer data overload, network-layer protocol heterogeneity, and application-layer identity explosion. The survey underscores the importance of cross-layer integration, quantum-enabled communication, and ethical governance frameworks for future AGI-enabled IoX systems. Finally, the survey identifies unresolved challenges, such as computational requirements, scalability, and real-world validation, calling for further research to fully realize AGI's potential in addressing IoX bottlenecks. we believe AGI-enhanced IoX is emerging as a critical research field at the intersection of interconnected systems and advanced AI. 

---
# Hallucination Detection with Small Language Models 

**Authors**: Ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.22486)  

**Abstract**: Since the introduction of ChatGPT, large language models (LLMs) have demonstrated significant utility in various tasks, such as answering questions through retrieval-augmented generation. Context can be retrieved using a vectorized database, serving as a foundation for LLMs to generate responses. However, hallucinations in responses can undermine the reliability of LLMs in practical applications, and they are not easily detectable in the absence of ground truth, particularly in question-and-answer scenarios. This paper proposes a framework that integrates multiple small language models to verify responses generated by LLMs using the retrieved context from a vectorized database. By breaking down the responses into individual sentences and utilizing the probability of generating "Yes" tokens from the outputs of multiple models for a given set of questions, responses, and relevant context, hallucinations can be detected. The proposed framework is validated through experiments with real datasets comprising over 100 sets of questions, answers, and contexts, including responses with fully and partially correct sentences. The results demonstrate a 10\% improvement in F1 scores for detecting correct responses compared to hallucinations, indicating that multiple small language models can be effectively employed for answer verification, providing a scalable and efficient solution for both academic and practical applications. 

---
# AI Agents-as-Judge: Automated Assessment of Accuracy, Consistency, Completeness and Clarity for Enterprise Documents 

**Authors**: Sudip Dasgupta, Himanshu Shankar  

**Link**: [PDF](https://arxiv.org/pdf/2506.22485)  

**Abstract**: This study presents a modular, multi-agent system for the automated review of highly structured enterprise business documents using AI agents. Unlike prior solutions focused on unstructured texts or limited compliance checks, this framework leverages modern orchestration tools such as LangChain, CrewAI, TruLens, and Guidance to enable section-by-section evaluation of documents for accuracy, consistency, completeness, and clarity. Specialized agents, each responsible for discrete review criteria such as template compliance or factual correctness, operate in parallel or sequence as required. Evaluation outputs are enforced to a standardized, machine-readable schema, supporting downstream analytics and auditability. Continuous monitoring and a feedback loop with human reviewers allow for iterative system improvement and bias mitigation.
Quantitative evaluation demonstrates that the AI Agent-as-Judge system approaches or exceeds human performance in key areas: achieving 99% information consistency (vs. 92% for humans), halving error and bias rates, and reducing average review time from 30 to 2.5 minutes per document, with a 95% agreement rate between AI and expert human judgment. While promising for a wide range of industries, the study also discusses current limitations, including the need for human oversight in highly specialized domains and the operational cost of large-scale LLM usage. The proposed system serves as a flexible, auditable, and scalable foundation for AI-driven document quality assurance in the enterprise context. 

---
# Hindsight-Guided Momentum (HGM) Optimizer: An Approach to Adaptive Learning Rate 

**Authors**: Krisanu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.22479)  

**Abstract**: We introduce Hindsight-Guided Momentum (HGM), a first-order optimization algorithm that adaptively scales learning rates based on the directional consistency of recent updates. Traditional adaptive methods, such as Adam or RMSprop , adapt learning dynamics using only the magnitude of gradients, often overlooking important geometric this http URL cues refer to directional information, such as the alignment between current gradients and past updates, which reflects the local curvature and consistency of the optimization path. HGM addresses this by incorporating a hindsight mechanism that evaluates the cosine similarity between the current gradient and accumulated momentum. This allows it to distinguish between coherent and conflicting gradient directions, increasing the learning rate when updates align and reducing it in regions of oscillation or noise. The result is a more responsive optimizer that accelerates convergence in smooth regions of the loss surface while maintaining stability in sharper or more erratic areas. Despite this added adaptability, the method preserves the computational and memory efficiency of existing this http URL more intelligently responding to the structure of the optimization landscape, HGM provides a simple yet effective improvement over existing approaches, particularly in non-convex settings like that of deep neural network training. 

---
# Innovative Research on IoT Architecture and Robotic Operating Platforms: Applications of Large Language Models and Generative AI 

**Authors**: Huiwen Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.22477)  

**Abstract**: This paper introduces an innovative design for robotic operating platforms, underpinned by a transformative Internet of Things (IoT) architecture, seamlessly integrating cutting-edge technologies such as large language models (LLMs), generative AI, edge computing, and 5G networks. The proposed platform aims to elevate the intelligence and autonomy of IoT systems and robotics, enabling them to make real-time decisions and adapt dynamically to changing environments. Through a series of compelling case studies across industries including smart manufacturing, healthcare, and service sectors, this paper demonstrates the substantial potential of IoT-enabled robotics to optimize operational workflows, enhance productivity, and deliver innovative, scalable solutions. By emphasizing the roles of LLMs and generative AI, the research highlights how these technologies drive the evolution of intelligent robotics and IoT, shaping the future of industry-specific advancements. The findings not only showcase the transformative power of these technologies but also offer a forward-looking perspective on their broader societal and industrial implications, positioning them as catalysts for next-generation automation and technological convergence. 

---
# Dimensionality Reduction on IoT Monitoring Data of Smart Building for Energy Consumption Forecasting 

**Authors**: Konstantinos Koutras, Agorakis Bompotas, Constantinos Halkiopoulos, Athanasios Kalogeras, Christos Alexakos  

**Link**: [PDF](https://arxiv.org/pdf/2506.22468)  

**Abstract**: The Internet of Things (IoT) plays a major role today in smart building infrastructures, from simple smart-home applications, to more sophisticated industrial type installations. The vast amounts of data generated from relevant systems can be processed in different ways revealing important information. This is especially true in the era of edge computing, when advanced data analysis and decision-making is gradually moving to the edge of the network where devices are generally characterised by low computing resources. In this context, one of the emerging main challenges is related to maintaining data analysis accuracy even with less data that can be efficiently handled by low resource devices. The present work focuses on correlation analysis of data retrieved from a pilot IoT network installation monitoring a small smart office by means of environmental and energy consumption sensors. The research motivation was to find statistical correlation between the monitoring variables that will allow the use of machine learning (ML) prediction algorithms for energy consumption reducing input parameters. For this to happen, a series of hypothesis tests for the correlation of three different environmental variables with the energy consumption were carried out. A total of ninety tests were performed, thirty for each pair of variables. In these tests, p-values showed the existence of strong or semi-strong correlation with two environmental variables, and of a weak correlation with a third one. Using the proposed methodology, we manage without examining the entire data set to exclude weak correlated variables while keeping the same score of accuracy. 

---
# Privacy-aware IoT Fall Detection Services For Aging in Place 

**Authors**: Abdallah Lakhdari, Jiajie Li, Amani Abusafia, Athman Bouguettaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.22462)  

**Abstract**: Fall detection is critical to support the growing elderly population, projected to reach 2.1 billion by 2050. However, existing methods often face data scarcity challenges or compromise privacy. We propose a novel IoT-based Fall Detection as a Service (FDaaS) framework to assist the elderly in living independently and safely by accurately detecting falls. We design a service-oriented architecture that leverages Ultra-wideband (UWB) radar sensors as an IoT health-sensing service, ensuring privacy and minimal intrusion. We address the challenges of data scarcity by utilizing a Fall Detection Generative Pre-trained Transformer (FD-GPT) that uses augmentation techniques. We developed a protocol to collect a comprehensive dataset of the elderly daily activities and fall events. This resulted in a real dataset that carefully mimics the elderly's routine. We rigorously evaluate and compare various models using this dataset. Experimental results show our approach achieves 90.72% accuracy and 89.33% precision in distinguishing between fall events and regular activities of daily living. 

---
# Machine Learning for Proactive Groundwater Management: Early Warning and Resource Allocation 

**Authors**: Chuan Li, Ruoxuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22461)  

**Abstract**: Groundwater supports ecosystems, agriculture, and drinking water supplies worldwide, yet effective monitoring remains challenging due to sparse data, computational constraints, and delayed outputs from traditional approaches. We develop a machine learning pipeline that predicts groundwater level categories using climate data, hydro-meteorological records, and physiographic attributes processed through AutoGluon's automated ensemble framework. Our approach integrates geospatial preprocessing, domain-driven feature engineering, and automated model selection to overcome conventional monitoring limitations. Applied to a large-scale French dataset (n $>$ 3,440,000 observations from 1,500+ wells), the model achieves weighted F\_1 scores of 0.927 on validation data and 0.67 on temporally distinct test data. Scenario-based evaluations demonstrate practical utility for early warning systems and water allocation decisions under changing climate conditions. The open-source implementation provides a scalable framework for integrating machine learning into national groundwater monitoring networks, enabling more responsive and data-driven water management strategies. 

---
# Heart rate and respiratory rate prediction from noisy real-world smartphone based on Deep Learning methods 

**Authors**: Ibne Farabi Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2506.22460)  

**Abstract**: Using mobile phone video of the fingertip as a data source for estimating vital signs such as heart rate (HR) and respiratory rate (RR) during daily life has long been suggested. While existing literature indicates that these estimates are accurate to within several beats or breaths per minute, the data used to draw these conclusions are typically collected in laboratory environments under careful experimental control, and yet the results are assumed to generalize to daily life. In an effort to test it, a team of researchers collected a large dataset of mobile phone video recordings made during daily life and annotated with ground truth HR and RR labels from N=111 participants. They found that traditional algorithm performance on the fingerprint videos is worse than previously reported (7 times and 13 times worse for RR and HR, respectively). Fortunately, recent advancements in deep learning, especially in convolutional neural networks (CNNs), offer a promising solution to improve this performance. This study proposes a new method for estimating HR and RR using a novel 3D deep CNN, demonstrating a reduced error in estimated HR by 68% and RR by 75%. These promising results suggest that regressor-based deep learning approaches should be used in estimating HR and RR. 

---
# A Complex UNet Approach for Non-Invasive Fetal ECG Extraction Using Single-Channel Dry Textile Electrodes 

**Authors**: Iulia Orvas, Andrei Radu, Alessandra Galli, Ana Neacsu, Elisabetta Peri  

**Link**: [PDF](https://arxiv.org/pdf/2506.22457)  

**Abstract**: Continuous, non-invasive pregnancy monitoring is crucial for minimising potential complications. The fetal electrocardiogram (fECG) represents a promising tool for assessing fetal health beyond clinical environments. Home-based monitoring necessitates the use of a minimal number of comfortable and durable electrodes, such as dry textile electrodes. However, this setup presents many challenges, including increased noise and motion artefacts, which complicate the accurate extraction of fECG signals. To overcome these challenges, we introduce a pioneering method for extracting fECG from single-channel recordings obtained using dry textile electrodes using AI techniques. We created a new dataset by simulating abdominal recordings, including noise closely resembling real-world characteristics of in-vivo recordings through dry textile electrodes, alongside mECG and fECG. To ensure the reliability of the extracted fECG, we propose an innovative pipeline based on a complex-valued denoising network, Complex UNet. Unlike previous approaches that focused solely on signal magnitude, our method processes both real and imaginary components of the spectrogram, addressing phase information and preventing incongruous predictions. We evaluated our novel pipeline against traditional, well-established approaches, on both simulated and real data in terms of fECG extraction and R-peak detection. The results showcase that our suggested method achieves new state-of-the-art results, enabling an accurate extraction of fECG morphology across all evaluated settings. This method is the first to effectively extract fECG signals from single-channel recordings using dry textile electrodes, making a significant advancement towards a fully non-invasive and self-administered fECG extraction solution. 

---
# Unsupervised Learning-Based Joint Resource Allocation and Beamforming Design for RIS-Assisted MISO-OFDMA Systems 

**Authors**: Yu Ma, Xingyu Zhou, Xiao Li, Le Liang, Shi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.22448)  

**Abstract**: Reconfigurable intelligent surfaces (RIS) are key enablers for 6G wireless systems. This paper studies downlink transmission in an RIS-assisted MISO-OFDMA system, addressing resource allocation challenges. A two-stage unsupervised learning-based framework is proposed to jointly design RIS phase shifts, BS beamforming, and resource block (RB) allocation. The framework includes BeamNet, which predicts RIS phase shifts from CSI, and AllocationNet, which allocates RBs using equivalent CSI derived from BeamNet outputs. Active beamforming is implemented via maximum ratio transmission and water-filling. To handle discrete constraints while ensuring differentiability, quantization and the Gumbel-softmax trick are adopted. A customized loss and phased training enhance performance under QoS constraints. Simulations show the method achieves 99.93% of the sum rate of the SCA baseline with only 0.036% of its runtime, and it remains robust across varying channel and user conditions. 

---
# Vision Transformers for Multi-Variable Climate Downscaling: Emulating Regional Climate Models with a Shared Encoder and Multi-Decoder Architecture 

**Authors**: Fabio Merizzi, Harilaos Loukos  

**Link**: [PDF](https://arxiv.org/pdf/2506.22447)  

**Abstract**: Global Climate Models (GCMs) are critical for simulating large-scale climate dynamics, but their coarse spatial resolution limits their applicability in regional studies. Regional Climate Models (RCMs) refine this through dynamic downscaling, albeit at considerable computational cost and with limited flexibility. While deep learning has emerged as an efficient data-driven alternative, most existing studies have focused on single-variable models that downscale one variable at a time. This approach can lead to limited contextual awareness, redundant computation, and lack of cross-variable interaction. Our study addresses these limitations by proposing a multi-task, multi-variable Vision Transformer (ViT) architecture with a shared encoder and variable-specific decoders (1EMD). The proposed architecture jointly predicts three key climate variables: surface temperature (tas), wind speed (sfcWind), and 500 hPa geopotential height (zg500), directly from GCM-resolution inputs, emulating RCM-scale downscaling over Europe. We show that our multi-variable approach achieves positive cross-variable knowledge transfer and consistently outperforms single-variable baselines trained under identical conditions, while also improving computational efficiency. These results demonstrate the effectiveness of multi-variable modeling for high-resolution climate downscaling. 

---
# EAGLE: Efficient Alignment of Generalized Latent Embeddings for Multimodal Survival Prediction with Interpretable Attribution Analysis 

**Authors**: Aakash Tripathi, Asim Waqas, Matthew B. Schabath, Yasin Yilmaz, Ghulam Rasool  

**Link**: [PDF](https://arxiv.org/pdf/2506.22446)  

**Abstract**: Accurate cancer survival prediction requires integration of diverse data modalities that reflect the complex interplay between imaging, clinical parameters, and textual reports. However, existing multimodal approaches suffer from simplistic fusion strategies, massive computational requirements, and lack of interpretability-critical barriers to clinical adoption. We present EAGLE (Efficient Alignment of Generalized Latent Embeddings), a novel deep learning framework that addresses these limitations through attention-based multimodal fusion with comprehensive attribution analysis. EAGLE introduces four key innovations: (1) dynamic cross-modal attention mechanisms that learn hierarchical relationships between modalities, (2) massive dimensionality reduction (99.96%) while maintaining predictive performance, (3) three complementary attribution methods providing patient-level interpretability, and (4) a unified pipeline enabling seamless adaptation across cancer types. We evaluated EAGLE on 911 patients across three distinct malignancies: glioblastoma (GBM, n=160), intraductal papillary mucinous neoplasms (IPMN, n=171), and non-small cell lung cancer (NSCLC, n=580). Patient-level analysis showed high-risk individuals relied more heavily on adverse imaging features, while low-risk patients demonstrated balanced modality contributions. Risk stratification identified clinically meaningful groups with 4-fold (GBM) to 5-fold (NSCLC) differences in median survival, directly informing treatment intensity decisions. By combining state-of-the-art performance with clinical interpretability, EAGLE bridges the gap between advanced AI capabilities and practical healthcare deployment, offering a scalable solution for multimodal survival prediction that enhances both prognostic accuracy and physician trust in automated predictions. 

---
# Hierarchical Adversarially-Resilient Multi-Agent Reinforcement Learning for Cyber-Physical Systems Security 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2506.22445)  

**Abstract**: Cyber-Physical Systems play a critical role in the infrastructure of various sectors, including manufacturing, energy distribution, and autonomous transportation systems. However, their increasing connectivity renders them highly vulnerable to sophisticated cyber threats, such as adaptive and zero-day attacks, against which traditional security methods like rule-based intrusion detection and single-agent reinforcement learning prove insufficient. To overcome these challenges, this paper introduces a novel Hierarchical Adversarially-Resilient Multi-Agent Reinforcement Learning (HAMARL) framework. HAMARL employs a hierarchical structure consisting of local agents dedicated to subsystem security and a global coordinator that oversees and optimizes comprehensive, system-wide defense strategies. Furthermore, the framework incorporates an adversarial training loop designed to simulate and anticipate evolving cyber threats, enabling proactive defense adaptation. Extensive experimental evaluations conducted on a simulated industrial IoT testbed indicate that HAMARL substantially outperforms traditional multi-agent reinforcement learning approaches, significantly improving attack detection accuracy, reducing response times, and ensuring operational continuity. The results underscore the effectiveness of combining hierarchical multi-agent coordination with adversarially-aware training to enhance the resilience and security of next-generation CPS. 

---
# Latent Factorization of Tensors with Threshold Distance Weighted Loss for Traffic Data Estimation 

**Authors**: Lei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22441)  

**Abstract**: Intelligent transportation systems (ITS) rely heavily on complete and high-quality spatiotemporal traffic data to achieve optimal performance. Nevertheless, in real-word traffic data collection processes, issues such as communication failures and sensor malfunctions often lead to incomplete or corrupted datasets, thereby posing significant challenges to the advancement of ITS. Among various methods for imputing missing spatiotemporal traffic data, the latent factorization of tensors (LFT) model has emerged as a widely adopted and effective solution. However, conventional LFT models typically employ the standard L2-norm in their learning objective, which makes them vulnerable to the influence of outliers. To overcome this limitation, this paper proposes a threshold distance weighted (TDW) loss-incorporated Latent Factorization of Tensors (TDWLFT) model. The proposed loss function effectively reduces the model's sensitivity to outliers by assigning differentiated weights to individual samples. Extensive experiments conducted on two traffic speed datasets sourced from diverse urban environments confirm that the proposed TDWLFT model consistently outperforms state-of-the-art approaches in terms of both in both prediction accuracy and computational efficiency. 

---
# Psycholinguistic Word Features: a New Approach for the Evaluation of LLMs Alignment with Humans 

**Authors**: Javier Conde, Miguel González, María Grandury, Gonzalo Martínez, Pedro Reviriego, Mar Brysbaert  

**Link**: [PDF](https://arxiv.org/pdf/2506.22439)  

**Abstract**: The evaluation of LLMs has so far focused primarily on how well they can perform different tasks such as reasoning, question-answering, paraphrasing, or translating. For most of these tasks, performance can be measured with objective metrics, such as the number of correct answers. However, other language features are not easily quantified. For example, arousal, concreteness, or gender associated with a given word, as well as the extent to which we experience words with senses and relate them to a specific sense. Those features have been studied for many years by psycholinguistics, conducting large-scale experiments with humans to produce ratings for thousands of words. This opens an opportunity to evaluate how well LLMs align with human ratings on these word features, taking advantage of existing studies that cover many different language features in a large number of words. In this paper, we evaluate the alignment of a representative group of LLMs with human ratings on two psycholinguistic datasets: the Glasgow and Lancaster norms. These datasets cover thirteen features over thousands of words. The results show that alignment is \textcolor{black}{generally} better in the Glasgow norms evaluated (arousal, valence, dominance, concreteness, imageability, familiarity, and gender) than on the Lancaster norms evaluated (introceptive, gustatory, olfactory, haptic, auditory, and visual). This suggests a potential limitation of current LLMs in aligning with human sensory associations for words, which may be due to their lack of embodied cognition present in humans and illustrates the usefulness of evaluating LLMs with psycholinguistic datasets. 

---
# Aria-MIDI: A Dataset of Piano MIDI Files for Symbolic Music Modeling 

**Authors**: Louis Bradshaw, Simon Colton  

**Link**: [PDF](https://arxiv.org/pdf/2504.15071)  

**Abstract**: We introduce an extensive new dataset of MIDI files, created by transcribing audio recordings of piano performances into their constituent notes. The data pipeline we use is multi-stage, employing a language model to autonomously crawl and score audio recordings from the internet based on their metadata, followed by a stage of pruning and segmentation using an audio classifier. The resulting dataset contains over one million distinct MIDI files, comprising roughly 100,000 hours of transcribed audio. We provide an in-depth analysis of our techniques, offering statistical insights, and investigate the content by extracting metadata tags, which we also provide. Dataset available at this https URL. 

---
# ResQuNNs:Towards Enabling Deep Learning in Quantum Convolution Neural Networks 

**Authors**: Muhammad Kashif, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2402.09146)  

**Abstract**: In this paper, we present a novel framework for enhancing the performance of Quanvolutional Neural Networks (QuNNs) by introducing trainable quanvolutional layers and addressing the critical challenges associated with them. Traditional quanvolutional layers, although beneficial for feature extraction, have largely been static, offering limited adaptability. Unlike state-of-the-art, our research overcomes this limitation by enabling training within these layers, significantly increasing the flexibility and potential of QuNNs. However, the introduction of multiple trainable quanvolutional layers induces complexities in gradient-based optimization, primarily due to the difficulty in accessing gradients across these layers. To resolve this, we propose a novel architecture, Residual Quanvolutional Neural Networks (ResQuNNs), leveraging the concept of residual learning, which facilitates the flow of gradients by adding skip connections between layers. By inserting residual blocks between quanvolutional layers, we ensure enhanced gradient access throughout the network, leading to improved training performance. Moreover, we provide empirical evidence on the strategic placement of these residual blocks within QuNNs. Through extensive experimentation, we identify an efficient configuration of residual blocks, which enables gradients across all the layers in the network that eventually results in efficient training. Our findings suggest that the precise location of residual blocks plays a crucial role in maximizing the performance gains in QuNNs. Our results mark a substantial step forward in the evolution of quantum deep learning, offering new avenues for both theoretical development and practical quantum computing applications. 

---
# Attention acts to suppress goal-based conflict under high competition 

**Authors**: Omar Claflin  

**Link**: [PDF](https://arxiv.org/pdf/1610.09431)  

**Abstract**: It is known that when multiple stimuli are present, top-down attention selectively enhances the neural signal in the visual cortex for task-relevant stimuli, but this has been tested only under conditions of minimal competition of visual attention. Here we show during high competition, that is, two stimuli in a shared receptive field possessing opposing modulatory goals, top-down attention suppresses both task-relevant and irrelevant neural signals within 100 ms of stimuli onset. This non-selective engagement of top-down attentional resources serves to reduce the feedforward signal representing irrelevant stimuli. 

---
