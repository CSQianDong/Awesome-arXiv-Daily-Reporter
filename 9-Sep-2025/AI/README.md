# Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference 

**Authors**: Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06942)  

**Abstract**: Recent studies have demonstrated the effectiveness of directly aligning diffusion models with human preferences using differentiable reward. However, they exhibit two primary challenges: (1) they rely on multistep denoising with gradient computation for reward scoring, which is computationally expensive, thus restricting optimization to only a few diffusion steps; (2) they often need continuous offline adaptation of reward models in order to achieve desired aesthetic quality, such as photorealism or precise lighting effects. To address the limitation of multistep denoising, we propose Direct-Align, a method that predefines a noise prior to effectively recover original images from any time steps via interpolation, leveraging the equation that diffusion states are interpolations between noise and target images, which effectively avoids over-optimization in late timesteps. Furthermore, we introduce Semantic Relative Preference Optimization (SRPO), in which rewards are formulated as text-conditioned signals. This approach enables online adjustment of rewards in response to positive and negative prompt augmentation, thereby reducing the reliance on offline reward fine-tuning. By fine-tuning the this http URL model with optimized denoising and online reward adjustment, we improve its human-evaluated realism and aesthetic quality by over 3x. 

---
# Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents 

**Authors**: Jiacheng Miao, Joe R. Davis, Jonathan K. Pritchard, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2509.06917)  

**Abstract**: We introduce Paper2Agent, an automated framework that converts research papers into AI agents. Paper2Agent transforms research output from passive artifacts into active systems that can accelerate downstream use, adoption, and discovery. Conventional research papers require readers to invest substantial effort to understand and adapt a paper's code, data, and methods to their own work, creating barriers to dissemination and reuse. Paper2Agent addresses this challenge by automatically converting a paper into an AI agent that acts as a knowledgeable research assistant. It systematically analyzes the paper and the associated codebase using multiple agents to construct a Model Context Protocol (MCP) server, then iteratively generates and runs tests to refine and robustify the resulting MCP. These paper MCPs can then be flexibly connected to a chat agent (e.g. Claude Code) to carry out complex scientific queries through natural language while invoking tools and workflows from the original paper. We demonstrate Paper2Agent's effectiveness in creating reliable and capable paper agents through in-depth case studies. Paper2Agent created an agent that leverages AlphaGenome to interpret genomic variants and agents based on ScanPy and TISSUE to carry out single-cell and spatial transcriptomics analyses. We validate that these paper agents can reproduce the original paper's results and can correctly carry out novel user queries. By turning static papers into dynamic, interactive AI agents, Paper2Agent introduces a new paradigm for knowledge dissemination and a foundation for the collaborative ecosystem of AI co-scientists. 

---
# Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet 

**Authors**: James Xu Zhao, Bryan Hooi, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06861)  

**Abstract**: Test-time scaling increases inference-time computation by allowing models to generate long reasoning chains, and has shown strong performance across many domains. However, in this work, we show that this approach is not yet effective for knowledge-intensive tasks, where high factual accuracy and low hallucination rates are essential. We conduct a comprehensive evaluation of test-time scaling using 12 reasoning models on two knowledge-intensive benchmarks. Our results reveal that increasing test-time computation does not consistently improve accuracy and, in many cases, it even leads to more hallucinations. We then analyze how extended reasoning affects hallucination behavior. We find that reduced hallucinations often result from the model choosing to abstain after thinking more, rather than from improved factual recall. Conversely, for some models, longer reasoning encourages attempts on previously unanswered questions, many of which result in hallucinations. Case studies show that extended reasoning can induce confirmation bias, leading to overconfident hallucinations. Despite these limitations, we observe that compared to non-thinking, enabling thinking remains beneficial. Code and data are available at this https URL 

---
# RAFFLES: Reasoning-based Attribution of Faults for LLM Systems 

**Authors**: Chenyang Zhu, Spencer Hong, Jingyu Wu, Kushal Chawla, Charlotte Tang, Youbing Yin, Nathan Wolfe, Erin Babinsky, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06822)  

**Abstract**: We have reached a critical roadblock in the development and enhancement of long-horizon, multi-component LLM agentic systems: it is incredibly tricky to identify where these systems break down and why. Evaluation capabilities that currently exist today (e.g., single pass LLM-as-a-judge) are limited in that they often focus on individual metrics or capabilities, end-to-end outcomes, and are narrowly grounded on the preferences of humans. We argue that to match the agentic capabilities, evaluation frameworks must also be able to reason, probe, iterate, and understand the complex logic passing through these systems over long horizons. In this paper, we present RAFFLES - an evaluation architecture that incorporates reasoning and iterative refinement. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically investigate faults and a set of specialized Evaluators to assess not only the system's components but also the quality of the reasoning by the Judge itself, thereby building a history of hypotheses. We tested RAFFLES against several baselines on the Who&When dataset, a benchmark designed to diagnose the "who" (agent) and "when" (step) of a system's failure. RAFFLES outperforms these baselines, achieving an agent-step fault pair accuracy of over 43% on the Algorithmically-Generated dataset (a substantial increase from the previously published best of 16.6%) and over 20% on the Hand-Crafted dataset (surpassing the previously published best of 8.8%). These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual human review. 

---
# Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting 

**Authors**: Shashidhar Reddy Javaji, Bhavul Gauri, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06770)  

**Abstract**: Large language models (LLMs) are now used in multi-turn workflows, but we still lack a clear way to measure when iteration helps and when it hurts. We present an evaluation framework for iterative refinement that spans ideation, code, and math. Our protocol runs controlled 12-turn conversations per task, utilizing a variety of prompts ranging from vague ``improve it'' feedback to targeted steering, and logs per-turn outputs. We score outcomes with domain-appropriate checks (unit tests for code; answer-equivalence plus reasoning-soundness for math; originality and feasibility for ideation) and track turn-level behavior with three families of metrics: semantic movement across turns, turn-to-turn change, and output size growth. Across models and tasks, gains are domain-dependent: they arrive early in ideas and code, but in math late turns matter when guided by elaboration. After the first few turns, vague feedback often plateaus or reverses correctness, while targeted prompts reliably shift the intended quality axis (novelty vs. feasibility in ideation; speed vs. readability in code; in math, elaboration outperforms exploration and drives late-turn gains). We also observe consistent domain patterns: ideation moves more in meaning across turns, code tends to grow in size with little semantic change, and math starts fixed but can break that path with late, elaborative this http URL, the framework and metrics make iteration measurable and comparable across models, and signal when to steer, stop, or switch strategies. 

---
# VehicleWorld: A Highly Integrated Multi-Device Environment for Intelligent Vehicle Interaction 

**Authors**: Jie Yang, Jiajun Chen, Zhangyue Yin, Shuo Chen, Yuxin Wang, Yiran Guo, Yuan Li, Yining Zheng, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06736)  

**Abstract**: Intelligent vehicle cockpits present unique challenges for API Agents, requiring coordination across tightly-coupled subsystems that exceed typical task environments' complexity. Traditional Function Calling (FC) approaches operate statelessly, requiring multiple exploratory calls to build environmental awareness before execution, leading to inefficiency and limited error recovery. We introduce VehicleWorld, the first comprehensive environment for the automotive domain, featuring 30 modules, 250 APIs, and 680 properties with fully executable implementations that provide real-time state information during agent execution. This environment enables precise evaluation of vehicle agent behaviors across diverse, challenging scenarios. Through systematic analysis, we discovered that direct state prediction outperforms function calling for environmental control. Building on this insight, we propose State-based Function Call (SFC), a novel approach that maintains explicit system state awareness and implements direct state transitions to achieve target conditions. Experimental results demonstrate that SFC significantly outperforms traditional FC approaches, achieving superior execution accuracy and reduced latency. We have made all implementation code publicly available on Github this https URL. 

---
# Reinforcement Learning Foundations for Deep Research Systems: A Survey 

**Authors**: Wenjun Li, Zhi Chen, Jingru Lin, Hannan Cao, Wei Han, Sheng Liang, Zhi Zhang, Kuicai Dong, Dexun Li, Chen Zhang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06733)  

**Abstract**: Deep research systems, agentic AI that solve complex, multi-step tasks by coordinating reasoning, search across the open web and user files, and tool use, are moving toward hierarchical deployments with a Planner, Coordinator, and Executors. In practice, training entire stacks end-to-end remains impractical, so most work trains a single planner connected to core tools such as search, browsing, and code. While SFT imparts protocol fidelity, it suffers from imitation and exposure biases and underuses environment feedback. Preference alignment methods such as DPO are schema and proxy-dependent, off-policy, and weak for long-horizon credit assignment and multi-objective trade-offs. A further limitation of SFT and DPO is their reliance on human defined decision points and subskills through schema design and labeled comparisons. Reinforcement learning aligns with closed-loop, tool-interaction research by optimizing trajectory-level policies, enabling exploration, recovery behaviors, and principled credit assignment, and it reduces dependence on such human priors and rater biases.
This survey is, to our knowledge, the first dedicated to the RL foundations of deep research systems. It systematizes work after DeepSeek-R1 along three axes: (i) data synthesis and curation; (ii) RL methods for agentic research covering stability, sample efficiency, long context handling, reward and credit design, multi-objective optimization, and multimodal integration; and (iii) agentic RL training systems and frameworks. We also cover agent architecture and coordination, as well as evaluation and benchmarks, including recent QA, VQA, long-form synthesis, and domain-grounded, tool-interaction tasks. We distill recurring patterns, surface infrastructure bottlenecks, and offer practical guidance for training robust, transparent deep research agents with RL. 

---
# CogGuide: Human-Like Guidance for Zero-Shot Omni-Modal Reasoning 

**Authors**: Zhou-Peng Shou, Zhi-Qiang You, Fang Wang, Hai-Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06641)  

**Abstract**: Targeting the issues of "shortcuts" and insufficient contextual understanding in complex cross-modal reasoning of multimodal large models, this paper proposes a zero-shot multimodal reasoning component guided by human-like cognitive strategies centered on an "intent sketch". The component comprises a plug-and-play three-module pipeline-Intent Perceiver, Strategy Generator, and Strategy Selector-that explicitly constructs a "understand-plan-select" cognitive process. By generating and filtering "intent sketch" strategies to guide the final reasoning, it requires no parameter fine-tuning and achieves cross-model transfer solely through in-context engineering. Information-theoretic analysis shows that this process can reduce conditional entropy and improve information utilization efficiency, thereby suppressing unintended shortcut reasoning. Experiments on IntentBench, WorldSense, and Daily-Omni validate the method's generality and robust gains; compared with their respective baselines, the complete "three-module" scheme yields consistent improvements across different reasoning engines and pipeline combinations, with gains up to approximately 9.51 percentage points, demonstrating the practical value and portability of the "intent sketch" reasoning component in zero-shot scenarios. 

---
# An AI system to help scientists write expert-level empirical software 

**Authors**: Eser Aygün, Anastasiya Belyaeva, Gheorghe Comanici, Marc Coram, Hao Cui, Jake Garrison, Renee Johnston Anton Kast, Cory Y. McLean, Peter Norgaard, Zahra Shamsi, David Smalling, James Thompson, Subhashini Venugopalan, Brian P. Williams, Chujun He, Sarah Martinson, Martyna Plomecka, Lai Wei, Yuchen Zhou, Qian-Ze Zhu, Matthew Abraham, Erica Brand, Anna Bulanova, Jeffrey A. Cardille, Chris Co, Scott Ellsworth, Grace Joseph, Malcolm Kane, Ryan Krueger, Johan Kartiwa, Dan Liebling, Jan-Matthis Lueckmann, Paul Raccuglia, Xuefei, Wang, Katherine Chou, James Manyika, Yossi Matias, John C. Platt, Lizzie Dorfman, Shibl Mourad, Michael P. Brenner  

**Link**: [PDF](https://arxiv.org/pdf/2509.06503)  

**Abstract**: The cycle of scientific discovery is frequently bottlenecked by the slow, manual creation of software to support computational experiments. To address this, we present an AI system that creates expert-level scientific software whose goal is to maximize a quality metric. The system uses a Large Language Model (LLM) and Tree Search (TS) to systematically improve the quality metric and intelligently navigate the large space of possible solutions. The system achieves expert-level results when it explores and integrates complex research ideas from external sources. The effectiveness of tree search is demonstrated across a wide range of benchmarks. In bioinformatics, it discovered 40 novel methods for single-cell data analysis that outperformed the top human-developed methods on a public leaderboard. In epidemiology, it generated 14 models that outperformed the CDC ensemble and all other individual models for forecasting COVID-19 hospitalizations. Our method also produced state-of-the-art software for geospatial analysis, neural activity prediction in zebrafish, time series forecasting and numerical solution of integrals. By devising and implementing novel solutions to diverse tasks, the system represents a significant step towards accelerating scientific progress. 

---
# Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers 

**Authors**: Ran Xin, Zeyu Zheng, Yanchen Nie, Kun Yuan, Xia Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06493)  

**Abstract**: The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search. 

---
# MORSE: Multi-Objective Reinforcement Learning via Strategy Evolution for Supply Chain Optimization 

**Authors**: Niki Kotecha, Ehecatl Antonio del Rio Chanona  

**Link**: [PDF](https://arxiv.org/pdf/2509.06490)  

**Abstract**: In supply chain management, decision-making often involves balancing multiple conflicting objectives, such as cost reduction, service level improvement, and environmental sustainability. Traditional multi-objective optimization methods, such as linear programming and evolutionary algorithms, struggle to adapt in real-time to the dynamic nature of supply chains. In this paper, we propose an approach that combines Reinforcement Learning (RL) and Multi-Objective Evolutionary Algorithms (MOEAs) to address these challenges for dynamic multi-objective optimization under uncertainty. Our method leverages MOEAs to search the parameter space of policy neural networks, generating a Pareto front of policies. This provides decision-makers with a diverse population of policies that can be dynamically switched based on the current system objectives, ensuring flexibility and adaptability in real-time decision-making. We also introduce Conditional Value-at-Risk (CVaR) to incorporate risk-sensitive decision-making, enhancing resilience in uncertain environments. We demonstrate the effectiveness of our approach through case studies, showcasing its ability to respond to supply chain dynamics and outperforming state-of-the-art methods in an inventory management case study. The proposed strategy not only improves decision-making efficiency but also offers a more robust framework for managing uncertainty and optimizing performance in supply chains. 

---
# MAS-Bench: A Unified Benchmark for Shortcut-Augmented Hybrid Mobile GUI Agents 

**Authors**: Pengxiang Zhao, Guangyi Liu, Yaozhen Liang, Weiqing He, Zhengxi Lu, Yuehao Huang, Yaxuan Guo, Kexin Zhang, Hao Wang, Liang Liu, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06477)  

**Abstract**: To enhance the efficiency of GUI agents on various platforms like smartphones and computers, a hybrid paradigm that combines flexible GUI operations with efficient shortcuts (e.g., API, deep links) is emerging as a promising direction. However, a framework for systematically benchmarking these hybrid agents is still underexplored. To take the first step in bridging this gap, we introduce MAS-Bench, a benchmark that pioneers the evaluation of GUI-shortcut hybrid agents with a specific focus on the mobile domain. Beyond merely using predefined shortcuts, MAS-Bench assesses an agent's capability to autonomously generate shortcuts by discovering and creating reusable, low-cost workflows. It features 139 complex tasks across 11 real-world applications, a knowledge base of 88 predefined shortcuts (APIs, deep-links, RPA scripts), and 7 evaluation metrics. The tasks are designed to be solvable via GUI-only operations, but can be significantly accelerated by intelligently embedding shortcuts. Experiments show that hybrid agents achieve significantly higher success rates and efficiency than their GUI-only counterparts. This result also demonstrates the effectiveness of our method for evaluating an agent's shortcut generation capabilities. MAS-Bench fills a critical evaluation gap, providing a foundational platform for future advancements in creating more efficient and robust intelligent agents. 

---
# Accelerate Scaling of LLM Alignment via Quantifying the Coverage and Depth of Instruction Set 

**Authors**: Chengwei Wu, Li Du, Hanyu Zhao, Yiming Ju, Jiapu Wang, Tengfei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.06463)  

**Abstract**: With the growing demand for applying large language models to downstream tasks, improving model alignment performance and efficiency has become crucial. Such a process involves selecting informative instructions from a candidate pool. However, due to the complexity of instruction set distributions, the key factors driving the performance of aligned models remain unclear. As a result, current instruction set refinement methods fail to improve performance as the instruction pool expands continuously. To address this issue, we first investigate the key factors that influence the relationship between instruction dataset distribution and aligned model performance. Based on these insights, we propose a novel instruction data selection method. We identify that the depth of instructions and the coverage of the semantic space are the crucial factors determining downstream performance, which could explain over 70\% of the model loss on the development set. We then design an instruction selection algorithm to simultaneously maximize the depth and semantic coverage of the selected instructions. Experimental results demonstrate that, compared to state-of-the-art baseline methods, it can sustainably improve model performance at a faster pace and thus achieve \emph{``Accelerated Scaling''}. 

---
# HyFedRAG: A Federated Retrieval-Augmented Generation Framework for Heterogeneous and Privacy-Sensitive Data 

**Authors**: Cheng Qian, Hainan Zhang, Yongxin Tong, Hong-Wei Zheng, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06444)  

**Abstract**: Centralized RAG pipelines struggle with heterogeneous and privacy-sensitive data, especially in distributed healthcare settings where patient data spans SQL, knowledge graphs, and clinical notes. Clinicians face difficulties retrieving rare disease cases due to privacy constraints and the limitations of traditional cloud-based RAG systems in handling diverse formats and edge devices. To address this, we introduce HyFedRAG, a unified and efficient Federated RAG framework tailored for Hybrid data modalities. By leveraging an edge-cloud collaborative mechanism, HyFedRAG enables RAG to operate across diverse data sources while preserving data privacy. Our key contributions are: (1) We design an edge-cloud collaborative RAG framework built on Flower, which supports querying structured SQL data, semi-structured knowledge graphs, and unstructured documents. The edge-side LLMs convert diverse data into standardized privacy-preserving representations, and the server-side LLMs integrates them for global reasoning and generation. (2) We integrate lightweight local retrievers with privacy-aware LLMs and provide three anonymization tools that enable each client to produce semantically rich, de-identified summaries for global inference across devices. (3) To optimize response latency and reduce redundant computation, we design a three-tier caching strategy consisting of local cache, intermediate representation cache, and cloud inference cache. Experimental results on PMC-Patients demonstrate that HyFedRAG outperforms existing baselines in terms of retrieval quality, generation consistency, and system efficiency. Our framework offers a scalable and privacy-compliant solution for RAG over structural-heterogeneous data, unlocking the potential of LLMs in sensitive and diverse data environments. 

---
# Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning 

**Authors**: Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.06436)  

**Abstract**: Large language models (LLMs) face persistent challenges when handling long-context tasks, most notably the lost in the middle issue, where information located in the middle of a long input tends to be underutilized. Some existing methods that reduce input have the risk of discarding key information, while others that extend context windows often lead to attention dispersion. To address these limitations, we propose Tree of Agents (TOA), a multi-agent reasoning framework that segments the input into chunks processed by independent agents. Each agent generates its local cognition, then agents dynamically exchange information for collaborative reasoning along tree-structured paths. TOA enables agents to probe different reasoning orders for multi-perspective understanding, effectively mitigating position bias and reducing hallucinations. To improve processing efficiency, we incorporate prefix-hash caching and adaptive pruning strategies, achieving significant performance improvements with comparable API overhead. Experiments show that TOA, powered by compact LLaMA3.1-8B, significantly outperforms multiple baselines and demonstrates comparable performance to the latest and much larger commercial models, such as Gemini1.5-pro, on various long-context tasks. Code is available at this https URL. 

---
# Teaching AI Stepwise Diagnostic Reasoning with Report-Guided Chain-of-Thought Learning 

**Authors**: Yihong Luo, Wenwu He, Zhuo-Xu Cui, Dong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06409)  

**Abstract**: This study presents DiagCoT, a multi-stage framework that applies supervised fine-tuning to general-purpose vision-language models (VLMs) to emulate radiologists' stepwise diagnostic reasoning using only free-text reports. DiagCoT combines contrastive image-report tuning for domain alignment, chain-of-thought supervision to capture inferential logic, and reinforcement tuning with clinical reward signals to enhance factual accuracy and fluency. On the MIMIC-CXR benchmark, DiagCoT improved zero-shot disease classification AUC from 0.52 to 0.76 (absolute gain of 0.24), pathology grounding mIoU from 0.08 to 0.31 (absolute gain of 0.23), and report generation BLEU from 0.11 to 0.33 (absolute gain of 0.22). It outperformed state-of-the-art models including LLaVA-Med and CXR-LLAVA on long-tailed diseases and external datasets. By converting unstructured clinical narratives into structured supervision, DiagCoT offers a scalable approach for developing interpretable and diagnostically competent AI systems for radiology. 

---
# A data-driven discretized CS:GO simulation environment to facilitate strategic multi-agent planning research 

**Authors**: Yunzhe Wang, Volkan Ustun, Chris McGroarty  

**Link**: [PDF](https://arxiv.org/pdf/2509.06355)  

**Abstract**: Modern simulation environments for complex multi-agent interactions must balance high-fidelity detail with computational efficiency. We present DECOY, a novel multi-agent simulator that abstracts strategic, long-horizon planning in 3D terrains into high-level discretized simulation while preserving low-level environmental fidelity. Using Counter-Strike: Global Offensive (CS:GO) as a testbed, our framework accurately simulates gameplay using only movement decisions as tactical positioning -- without explicitly modeling low-level mechanics such as aiming and shooting. Central to our approach is a waypoint system that simplifies and discretizes continuous states and actions, paired with neural predictive and generative models trained on real CS:GO tournament data to reconstruct event outcomes. Extensive evaluations show that replays generated from human data in DECOY closely match those observed in the original game. Our publicly available simulation environment provides a valuable tool for advancing research in strategic multi-agent planning and behavior generation. 

---
# Evaluating Multi-Turn Bargain Skills in LLM-Based Seller Agent 

**Authors**: Issue Yishu Wang, Kakam Chong, Xiaofeng Wang, Xu Yan, DeXin Kong, Chen Ju, Ming Chen, Shuai Xiao, Shuguang Han, jufeng chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.06341)  

**Abstract**: In online second-hand marketplaces, multi-turn bargaining is a crucial part of seller-buyer interactions. Large Language Models (LLMs) can act as seller agents, negotiating with buyers on behalf of sellers under given business constraints. A critical ability for such agents is to track and accurately interpret cumulative buyer intents across long negotiations, which directly impacts bargaining effectiveness. We introduce a multi-turn evaluation framework for measuring the bargaining ability of seller agents in e-commerce dialogues. The framework tests whether an agent can extract and track buyer intents. Our contributions are: (1) a large-scale e-commerce bargaining benchmark spanning 622 categories, 9,892 products, and 3,014 tasks; (2) a turn-level evaluation framework grounded in Theory of Mind (ToM) with annotated buyer intents, moving beyond outcome-only metrics; and (3) an automated pipeline that extracts reliable intent from massive dialogue data. 

---
# Large Language Models as Virtual Survey Respondents: Evaluating Sociodemographic Response Generation 

**Authors**: Jianpeng Zhao, Chenyu Yuan, Weiming Luo, Haoling Xie, Guangwei Zhang, Steven Jige Quan, Zixuan Yuan, Pengyang Wang, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06337)  

**Abstract**: Questionnaire-based surveys are foundational to social science research and public policymaking, yet traditional survey methods remain costly, time-consuming, and often limited in scale. This paper explores a new paradigm: simulating virtual survey respondents using Large Language Models (LLMs). We introduce two novel simulation settings, namely Partial Attribute Simulation (PAS) and Full Attribute Simulation (FAS), to systematically evaluate the ability of LLMs to generate accurate and demographically coherent responses. In PAS, the model predicts missing attributes based on partial respondent profiles, whereas FAS involves generating complete synthetic datasets under both zero-context and context-enhanced conditions. We curate a comprehensive benchmark suite, LLM-S^3 (Large Language Model-based Sociodemographic Survey Simulation), that spans 11 real-world public datasets across four sociological domains. Our evaluation of multiple mainstream LLMs (GPT-3.5/4 Turbo, LLaMA 3.0/3.1-8B) reveals consistent trends in prediction performance, highlights failure modes, and demonstrates how context and prompt design impact simulation fidelity. This work establishes a rigorous foundation for LLM-driven survey simulations, offering scalable and cost-effective tools for sociological research and policy evaluation. Our code and dataset are available at: this https URL 

---
# Can AI Make Energy Retrofit Decisions? An Evaluation of Large Language Models 

**Authors**: Lei Shu, Dong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06307)  

**Abstract**: Conventional approaches to building energy retrofit decision making suffer from limited generalizability and low interpretability, hindering adoption in diverse residential contexts. With the growth of Smart and Connected Communities, generative AI, especially large language models (LLMs), may help by processing contextual information and producing practitioner readable recommendations. We evaluate seven LLMs (ChatGPT, DeepSeek, Gemini, Grok, Llama, and Claude) on residential retrofit decisions under two objectives: maximizing CO2 reduction (technical) and minimizing payback period (sociotechnical). Performance is assessed on four dimensions: accuracy, consistency, sensitivity, and reasoning, using a dataset of 400 homes across 49 US states. LLMs generate effective recommendations in many cases, reaching up to 54.5 percent top 1 match and 92.8 percent within top 5 without fine tuning. Performance is stronger for the technical objective, while sociotechnical decisions are limited by economic trade offs and local context. Agreement across models is low, and higher performing models tend to diverge from others. LLMs are sensitive to location and building geometry but less sensitive to technology and occupant behavior. Most models show step by step, engineering style reasoning, but it is often simplified and lacks deeper contextual awareness. Overall, LLMs are promising assistants for energy retrofit decision making, but improvements in accuracy, consistency, and context handling are needed for reliable practice. 

---
# From Implicit Exploration to Structured Reasoning: Leveraging Guideline and Refinement for LLMs 

**Authors**: Jiaxiang Chen, Zhuo Wang, Mingxi Zou, Zhucong Li, Zhijian Zhou, Song Wang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06284)  

**Abstract**: Large language models (LLMs) have advanced general-purpose reasoning, showing strong performance across diverse tasks. However, existing methods often rely on implicit exploration, where the model follows stochastic and unguided reasoning paths-like walking without a map. This leads to unstable reasoning paths, lack of error correction, and limited learning from past experience. To address these issues, we propose a framework that shifts from implicit exploration to structured reasoning through guideline and refinement. First, we extract structured reasoning patterns from successful trajectories and reflective signals from failures. During inference, the model follows these guidelines step-by-step, with refinement applied after each step to correct errors and stabilize the reasoning process. Experiments on BBH and four additional benchmarks (GSM8K, MATH-500, MBPP, HumanEval) show that our method consistently outperforms strong baselines across diverse reasoning tasks. Structured reasoning with stepwise execution and refinement improves stability and generalization, while guidelines transfer well across domains and flexibly support cross-model collaboration, matching or surpassing supervised fine-tuning in effectiveness and scalability. 

---
# SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents 

**Authors**: Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2509.06283)  

**Abstract**: Equipping large language models (LLMs) with complex, interleaved reasoning and tool-use capabilities has become a key focus in agentic AI research, especially with recent advances in reasoning-oriented (``thinking'') models. Such capabilities are key to unlocking a number of important applications. One such application is Deep Research (DR), which requires extensive search and reasoning over many sources. Our work in this paper focuses on the development of native Autonomous Single-Agent models for DR featuring minimal web crawling and Python tool integration. Unlike multi-agent systems, where agents take up pre-defined roles and are told what to do at each step in a static workflow, an autonomous single-agent determines its next action dynamically based on context, without manual directive. While prior work has proposed training recipes for base or instruction-tuned LLMs, we focus on continual reinforcement learning (RL) of reasoning-optimized models to further enhance agentic skills while preserving reasoning ability. Towards this end, we propose a simple RL recipe with entirely synthetic data, which we apply to various open-source LLMs. Our best variant SFR-DR-20B achieves up to 28.7% on Humanity's Last Exam benchmark. In addition, we conduct key analysis experiments to provide more insights into our methodologies. 

---
# TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning 

**Authors**: Chuang Jiang, Mingyue Cheng, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06278)  

**Abstract**: Table reasoning is crucial for leveraging structured data in domains such as finance, healthcare, and scientific research. While large language models (LLMs) show promise in multi-step reasoning, purely text-based methods often struggle with the complex numerical computations and fine-grained operations inherently required in this task. Tool-integrated reasoning improves computational accuracy via explicit code execution, yet existing systems frequently rely on rigid patterns, supervised imitation, and lack true autonomous adaptability. In this paper, we present TableMind, an LLM-driven table reasoning agent that (i) autonomously performs multi-turn tool invocation, (ii) writes and executes data-analyzing code in a secure sandbox environment for data analysis and precise numerical reasoning, and (iii) exhibits high-level capabilities such as planning and self-reflection to adapt strategies. To realize these capabilities, we adopt a two-stage fine-tuning paradigm built on top of a powerful pre-trained language model: supervised fine-tuning on high-quality reasoning trajectories to establish effective tool usage patterns, followed by reinforcement fine-tuning to optimize multi-objective strategies. In particular, we propose Rank-Aware Policy Optimization (RAPO), which increases the update weight of high-quality trajectories when their output probabilities are lower than those of low-quality ones, thereby guiding the model more consistently toward better and more accurate answers. Extensive experiments on several mainstream benchmarks demonstrate that TableMind achieves superior performance compared to competitive baselines, yielding substantial gains in both reasoning accuracy and computational precision. 

---
# REMI: A Novel Causal Schema Memory Architecture for Personalized Lifestyle Recommendation Agents 

**Authors**: Vishal Raman, Vijai Aravindh R, Abhijith Ragav  

**Link**: [PDF](https://arxiv.org/pdf/2509.06269)  

**Abstract**: Personalized AI assistants often struggle to incorporate complex personal data and causal knowledge, leading to generic advice that lacks explanatory power. We propose REMI, a Causal Schema Memory architecture for a multimodal lifestyle agent that integrates a personal causal knowledge graph, a causal reasoning engine, and a schema based planning module. The idea is to deliver explainable, personalized recommendations in domains like fashion, personal wellness, and lifestyle planning. Our architecture uses a personal causal graph of the user's life events and habits, performs goal directed causal traversals enriched with external knowledge and hypothetical reasoning, and retrieves adaptable plan schemas to generate tailored action plans. A Large Language Model orchestrates these components, producing answers with transparent causal explanations. We outline the CSM system design and introduce new evaluation metrics for personalization and explainability, including Personalization Salience Score and Causal Reasoning Accuracy, to rigorously assess its performance. Results indicate that CSM based agents can provide more context aware, user aligned recommendations compared to baseline LLM agents. This work demonstrates a novel approach to memory augmented, causal reasoning in personalized agents, advancing the development of transparent and trustworthy AI lifestyle assistants. 

---
# Proof2Silicon: Prompt Repair for Verified Code and Hardware Generation via Reinforcement Learning 

**Authors**: Manvi Jha, Jiaxin Wan, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.06239)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in automated code generation but frequently produce code that fails formal verification, an essential requirement for hardware and safety-critical domains. To overcome this fundamental limitation, we previously proposed PREFACE, a model-agnostic framework based on reinforcement learning (RL) that iteratively repairs the prompts provided to frozen LLMs, systematically steering them toward generating formally verifiable Dafny code without costly fine-tuning. This work presents Proof2Silicon, a novel end-to-end synthesis framework that embeds the previously proposed PREFACE flow to enable the generation of correctness-by-construction hardware directly from natural language specifications. Proof2Silicon operates by: (1) leveraging PREFACE's verifier-driven RL agent to optimize prompt generation iteratively, ensuring Dafny code correctness; (2) automatically translating verified Dafny programs into synthesizable high-level C using Dafny's Python backend and PyLog; and (3) employing Vivado HLS to produce RTL implementations. Evaluated rigorously on a challenging 100-task benchmark, PREFACE's RL-guided prompt optimization consistently improved Dafny verification success rates across diverse LLMs by up to 21%. Crucially, Proof2Silicon achieved an end-to-end hardware synthesis success rate of up to 72%, generating RTL designs through Vivado HLS synthesis flows. These results demonstrate a robust, scalable, and automated pipeline for LLM-driven, formally verified hardware synthesis, bridging natural-language specification and silicon realization. 

---
# PillagerBench: Benchmarking LLM-Based Agents in Competitive Minecraft Team Environments 

**Authors**: Olivier Schipper, Yudi Zhang, Yali Du, Mykola Pechenizkiy, Meng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06235)  

**Abstract**: LLM-based agents have shown promise in various cooperative and strategic reasoning tasks, but their effectiveness in competitive multi-agent environments remains underexplored. To address this gap, we introduce PillagerBench, a novel framework for evaluating multi-agent systems in real-time competitive team-vs-team scenarios in Minecraft. It provides an extensible API, multi-round testing, and rule-based built-in opponents for fair, reproducible comparisons. We also propose TactiCrafter, an LLM-based multi-agent system that facilitates teamwork through human-readable tactics, learns causal dependencies, and adapts to opponent strategies. Our evaluation demonstrates that TactiCrafter outperforms baseline approaches and showcases adaptive learning through self-play. Additionally, we analyze its learning process and strategic evolution over multiple game episodes. To encourage further research, we have open-sourced PillagerBench, fostering advancements in multi-agent AI for competitive environments. 

---
# From Long to Short: LLMs Excel at Trimming Own Reasoning Chains 

**Authors**: Wei Han, Geng Zhan, Sicheng Yu, Chenyu Wang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.06174)  

**Abstract**: O1/R1 style large reasoning models (LRMs) signal a substantial leap forward over conventional instruction-following LLMs. By applying test-time scaling to generate extended reasoning paths, they establish many SOTAs across a wide range of complex reasoning tasks. However, recent studies show that LRMs are prone to suffer from overthinking -- the tendency to overcomplicate simple problems, leading to excessive strategy switching and long, convoluted reasoning traces that hinder their interpretability. To mitigate this issue, we conduct a systematic investigation into the reasoning efficiency of a broad set of LRMs and uncover a common dilemma: the difficulty in balancing multiple generation objectives such as correctness and brevity. Based on this discovery, we propose a test-time scaling method, EDIT (Efficient Dynamic Inference Trimming), which efficiently guides LRMs to identify the shortest correct reasoning paths at test time. EDIT employs constraint-guided generation while jointly tracking length and answer distributions under varying constraints, allowing it to select responses that strike an optimal balance between conciseness and correctness. Extensive experiments across diverse models and datasets show that EDIT substantially enhance the reasoning efficiency, producing compact yet informative outputs that improve readability and user experience. 

---
# Reverse-Engineered Reasoning for Open-Ended Generation 

**Authors**: Haozhe Wang, Haoran Que, Qixin Xu, Minghao Liu, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Wei Ye, Tong Yang, Wenhao Huang, Ge Zhang, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06160)  

**Abstract**: While the ``deep reasoning'' paradigm has spurred significant advances in verifiable domains like mathematics, its application to open-ended, creative generation remains a critical challenge. The two dominant methods for instilling reasoning -- reinforcement learning (RL) and instruction distillation -- falter in this area; RL struggles with the absence of clear reward signals and high-quality reward models, while distillation is prohibitively expensive and capped by the teacher model's capabilities. To overcome these limitations, we introduce REverse-Engineered Reasoning (REER), a new paradigm that fundamentally shifts the approach. Instead of building a reasoning process ``forwards'' through trial-and-error or imitation, REER works ``backwards'' from known-good solutions to computationally discover the latent, step-by-step deep reasoning process that could have produced them. Using this scalable, gradient-free approach, we curate and open-source DeepWriting-20K, a large-scale dataset of 20,000 deep reasoning trajectories for open-ended tasks. Our model, DeepWriter-8B, trained on this data, not only surpasses strong open-source baselines but also achieves performance competitive with, and at times superior to, leading proprietary models like GPT-4o and Claude 3.5. 

---
# Rethinking Reasoning Quality in Large Language Models through Enhanced Chain-of-Thought via RL 

**Authors**: Haoyang He, Zihua Rong, Kun Ji, Chenyang Li, Qing Huang, Chong Xia, Lan Yang, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06024)  

**Abstract**: Reinforcement learning (RL) has recently become the dominant paradigm for strengthening the reasoning abilities of large language models (LLMs). Yet the rule-based reward functions commonly used on mathematical or programming benchmarks assess only answer format and correctness, providing no signal as to whether the induced Chain-of-Thought (CoT) actually improves the answer. Furthermore, such task-specific training offers limited control over logical depth and therefore may fail to reveal a model's genuine reasoning capacity. We propose Dynamic Reasoning Efficiency Reward (DRER) -- a plug-and-play RL reward framework that reshapes both reward and advantage signals. (i) A Reasoning Quality Reward assigns fine-grained credit to those reasoning chains that demonstrably raise the likelihood of the correct answer, directly incentivising the trajectories with beneficial CoT tokens. (ii) A Dynamic Length Advantage decays the advantage of responses whose length deviates from a validation-derived threshold, stabilising training. To facilitate rigorous assessment, we also release Logictree, a dynamically constructed deductive reasoning dataset that functions both as RL training data and as a comprehensive benchmark. Experiments confirm the effectiveness of DRER: our 7B model attains GPT-o3-mini level performance on Logictree with 400 trianing steps, while the average confidence of CoT-augmented answers rises by 30%. The model further exhibits generalisation across diverse logical-reasoning datasets, and the mathematical benchmark AIME24. These results illuminate how RL shapes CoT behaviour and chart a practical path toward enhancing formal-reasoning skills in large language models. All code and data are available in repository this https URL. 

---
# MapAgent: A Hierarchical Agent for Geospatial Reasoning with Dynamic Map Tool Integration 

**Authors**: Md Hasebul Hasan, Mahir Labib Dihan, Mohammed Eunus Ali, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2509.05933)  

**Abstract**: Agentic AI has significantly extended the capabilities of large language models (LLMs) by enabling complex reasoning and tool use. However, most existing frameworks are tailored to domains such as mathematics, coding, or web automation, and fall short on geospatial tasks that require spatial reasoning, multi-hop planning, and real-time map interaction. To address these challenges, we introduce MapAgent, a hierarchical multi-agent plug-and-play framework with customized toolsets and agentic scaffolds for map-integrated geospatial reasoning. Unlike existing flat agent-based approaches that treat tools uniformly-often overwhelming the LLM when handling similar but subtly different geospatial APIs-MapAgent decouples planning from execution. A high-level planner decomposes complex queries into subgoals, which are routed to specialized modules. For tool-heavy modules-such as map-based services-we then design a dedicated map-tool agent that efficiently orchestrates related APIs adaptively in parallel to effectively fetch geospatial data relevant for the query, while simpler modules (e.g., solution generation or answer extraction) operate without additional agent overhead. This hierarchical design reduces cognitive load, improves tool selection accuracy, and enables precise coordination across similar APIs. We evaluate MapAgent on four diverse geospatial benchmarks-MapEval-Textual, MapEval-API, MapEval-Visual, and MapQA-and demonstrate substantial gains over state-of-the-art tool-augmented and agentic baselines. We open-source our framwork at this https URL. 

---
# Chatbot To Help Patients Understand Their Health 

**Authors**: Won Seok Jang, Hieu Tran, Manav Mistry, SaiKiran Gandluri, Yifan Zhang, Sharmin Sultana, Sunjae Kown, Yuan Zhang, Zonghai Yao, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05818)  

**Abstract**: Patients must possess the knowledge necessary to actively participate in their care. We present NoteAid-Chatbot, a conversational AI that promotes patient understanding via a novel 'learning as conversation' framework, built on a multi-agent large language model (LLM) and reinforcement learning (RL) setup without human-labeled data. NoteAid-Chatbot was built on a lightweight LLaMA 3.2 3B model trained in two stages: initial supervised fine-tuning on conversational data synthetically generated using medical conversation strategies, followed by RL with rewards derived from patient understanding assessments in simulated hospital discharge scenarios. Our evaluation, which includes comprehensive human-aligned assessments and case studies, demonstrates that NoteAid-Chatbot exhibits key emergent behaviors critical for patient education, such as clarity, relevance, and structured dialogue, even though it received no explicit supervision for these attributes. Our results show that even simple Proximal Policy Optimization (PPO)-based reward modeling can successfully train lightweight, domain-specific chatbots to handle multi-turn interactions, incorporate diverse educational strategies, and meet nuanced communication objectives. Our Turing test demonstrates that NoteAid-Chatbot surpasses non-expert human. Although our current focus is on healthcare, the framework we present illustrates the feasibility and promise of applying low-cost, PPO-based RL to realistic, open-ended conversational domains, broadening the applicability of RL-based alignment methods. 

---
# Decision-Focused Learning Enhanced by Automated Feature Engineering for Energy Storage Optimisation 

**Authors**: Nasser Alkhulaifi, Ismail Gokay Dogan, Timothy R. Cargan, Alexander L. Bowler, Direnc Pekaslan, Nicholas J. Watson, Isaac Triguero  

**Link**: [PDF](https://arxiv.org/pdf/2509.05772)  

**Abstract**: Decision-making under uncertainty in energy management is complicated by unknown parameters hindering optimal strategies, particularly in Battery Energy Storage System (BESS) operations. Predict-Then-Optimise (PTO) approaches treat forecasting and optimisation as separate processes, allowing prediction errors to cascade into suboptimal decisions as models minimise forecasting errors rather than optimising downstream tasks. The emerging Decision-Focused Learning (DFL) methods overcome this limitation by integrating prediction and optimisation; however, they are relatively new and have been tested primarily on synthetic datasets or small-scale problems, with limited evidence of their practical viability. Real-world BESS applications present additional challenges, including greater variability and data scarcity due to collection constraints and operational limitations. Because of these challenges, this work leverages Automated Feature Engineering (AFE) to extract richer representations and improve the nascent approach of DFL. We propose an AFE-DFL framework suitable for small datasets that forecasts electricity prices and demand while optimising BESS operations to minimise costs. We validate its effectiveness on a novel real-world UK property dataset. The evaluation compares DFL methods against PTO, with and without AFE. The results show that, on average, DFL yields lower operating costs than PTO and adding AFE further improves the performance of DFL methods by 22.9-56.5% compared to the same models without AFE. These findings provide empirical evidence for DFL's practical viability in real-world settings, indicating that domain-specific AFE enhances DFL and reduces reliance on domain expertise for BESS optimisation, yielding economic benefits with broader implications for energy management systems facing similar challenges. 

---
# DRF: LLM-AGENT Dynamic Reputation Filtering Framework 

**Authors**: Yuwei Lou, Hao Hu, Shaocong Ma, Zongfei Zhang, Liang Wang, Jidong Ge, Xianping Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05764)  

**Abstract**: With the evolution of generative AI, multi - agent systems leveraging large - language models(LLMs) have emerged as a powerful tool for complex tasks. However, these systems face challenges in quantifying agent performance and lack mechanisms to assess agent credibility. To address these issues, we introduce DRF, a dynamic reputation filtering framework. DRF constructs an interactive rating network to quantify agent performance, designs a reputation scoring mechanism to measure agent honesty and capability, and integrates an Upper Confidence Bound - based strategy to enhance agent selection efficiency. Experiments show that DRF significantly improves task completion quality and collaboration efficiency in logical reasoning and code - generation tasks, offering a new approach for multi - agent systems to handle large - scale tasks. 

---
# Hyperbolic Large Language Models 

**Authors**: Sarang Patil, Zeyong Zhang, Yiran Huang, Tengfei Ma, Mengjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05757)  

**Abstract**: Large language models (LLMs) have achieved remarkable success and demonstrated superior performance across various tasks, including natural language processing (NLP), weather forecasting, biological protein folding, text generation, and solving mathematical problems. However, many real-world data exhibit highly non-Euclidean latent hierarchical anatomy, such as protein networks, transportation networks, financial networks, brain networks, and linguistic structures or syntactic trees in natural languages. Effectively learning intrinsic semantic entailment and hierarchical relationships from these raw, unstructured input data using LLMs remains an underexplored area. Due to its effectiveness in modeling tree-like hierarchical structures, hyperbolic geometry -- a non-Euclidean space -- has rapidly gained popularity as an expressive latent representation space for complex data modeling across domains such as graphs, images, languages, and multi-modal data. Here, we provide a comprehensive and contextual exposition of recent advancements in LLMs that leverage hyperbolic geometry as a representation space to enhance semantic representation learning and multi-scale reasoning. Specifically, the paper presents a taxonomy of the principal techniques of Hyperbolic LLMs (HypLLMs) in terms of four main categories: (1) hyperbolic LLMs through exp/log maps; (2) hyperbolic fine-tuned models; (3) fully hyperbolic LLMs, and (4) hyperbolic state-space models. We also explore crucial potential applications and outline future research directions. A repository of key papers, models, datasets, and code implementations is available at this https URL. 

---
# Towards Meta-Cognitive Knowledge Editing for Multimodal LLMs 

**Authors**: Zhaoyu Fan, Kaihang Pan, Mingze Zhou, Bosheng Qin, Juncheng Li, Shengyu Zhang, Wenqiao Zhang, Siliang Tang, Fei Wu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05714)  

**Abstract**: Knowledge editing enables multimodal large language models (MLLMs) to efficiently update outdated or incorrect information. However, existing benchmarks primarily emphasize cognitive-level modifications while lacking a focus on deeper meta-cognitive processes. To bridge this gap, we introduce CogEdit, a novel benchmark designed to evaluate MLLMs' meta-cognitive knowledge editing abilities across three levels: (1) Counterfactual-Driven Editing, assessing self-awareness of knowledge correctness changes; (2) Boundary Constraint Editing, ensuring appropriate generalization without unintended interference; and (3) Noise-Robust Editing, promoting reflective evaluation of uncertain information. To advance meta-cognitive editing, we propose MIND (Meta-cognitive INtegrated Dynamic Knowledge Editing), a framework that constructs a meta-knowledge memory for self-awareness, employs game-theoretic interactions to monitor knowledge activation, and incorporates label refinement for noise-robust updates. Extensive experiments show that MIND significantly outperforms existing cognitive editing approaches, achieving strong performance on both traditional and meta-cognitive knowledge editing benchmarks. 

---
# MSRFormer: Road Network Representation Learning using Multi-scale Feature Fusion of Heterogeneous Spatial Interactions 

**Authors**: Jian Yang, Jiahui Wu, Li Fang, Hongchao Fan, Bianying Zhang, Huijie Zhao, Guangyi Yang, Rui Xin, Xiong You  

**Link**: [PDF](https://arxiv.org/pdf/2509.05685)  

**Abstract**: Transforming road network data into vector representations using deep learning has proven effective for road network analysis. However, urban road networks' heterogeneous and hierarchical nature poses challenges for accurate representation learning. Graph neural networks, which aggregate features from neighboring nodes, often struggle due to their homogeneity assumption and focus on a single structural scale. To address these issues, this paper presents MSRFormer, a novel road network representation learning framework that integrates multi-scale spatial interactions by addressing their flow heterogeneity and long-distance dependencies. It uses spatial flow convolution to extract small-scale features from large trajectory datasets, and identifies scale-dependent spatial interaction regions to capture the spatial structure of road networks and flow heterogeneity. By employing a graph transformer, MSRFormer effectively captures complex spatial dependencies across multiple scales. The spatial interaction features are fused using residual connections, which are fed to a contrastive learning algorithm to derive the final road network representation. Validation on two real-world datasets demonstrates that MSRFormer outperforms baseline methods in two road network analysis tasks. The performance gains of MSRFormer suggest the traffic-related task benefits more from incorporating trajectory data, also resulting in greater improvements in complex road network structures with up to 16% improvements compared to the most competitive baseline method. This research provides a practical framework for developing task-agnostic road network representation models and highlights distinct association patterns of the interplay between scale effects and flow heterogeneity of spatial interactions. 

---
# OccVLA: Vision-Language-Action Model with Implicit 3D Occupancy Supervision 

**Authors**: Ruixun Liu, Lingyu Kong, Derun Li, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05578)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong vision-language reasoning abilities but still lack robust 3D spatial understanding, which is critical for autonomous driving. This limitation stems from two key challenges: (1) the difficulty of constructing accessible yet effective 3D representations without expensive manual annotations, and (2) the loss of fine-grained spatial details in VLMs due to the absence of large-scale 3D vision-language pretraining. To address these challenges, we propose OccVLA, a novel framework that integrates 3D occupancy representations into a unified multimodal reasoning process. Unlike prior approaches that rely on explicit 3D inputs, OccVLA treats dense 3D occupancy as both a predictive output and a supervisory signal, enabling the model to learn fine-grained spatial structures directly from 2D visual inputs. The occupancy predictions are regarded as implicit reasoning processes and can be skipped during inference without performance degradation, thereby adding no extra computational overhead. OccVLA achieves state-of-the-art results on the nuScenes benchmark for trajectory planning and demonstrates superior performance on 3D visual question-answering tasks, offering a scalable, interpretable, and fully vision-based solution for autonomous driving. 

---
# TreeGPT: A Novel Hybrid Architecture for Abstract Syntax Tree Processing with Global Parent-Child Aggregation 

**Authors**: Zixi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05550)  

**Abstract**: We introduce TreeGPT, a novel neural architecture that combines transformer-based attention mechanisms with global parent-child aggregation for processing Abstract Syntax Trees (ASTs) in neural program synthesis tasks. Unlike traditional approaches that rely solely on sequential processing or graph neural networks, TreeGPT employs a hybrid design that leverages both self-attention for capturing local dependencies and a specialized Tree Feed-Forward Network (TreeFFN) for modeling hierarchical tree structures through iterative message passing.
The core innovation lies in our Global Parent-Child Aggregation mechanism, formalized as: $$h_i^{(t+1)} = \sigma \Big( h_i^{(0)} + W_{pc} \sum_{(p,c) \in E_i} f(h_p^{(t)}, h_c^{(t)}) + b \Big)$$ where $h_i^{(t)}$ represents the hidden state of node $i$ at iteration $t$, $E_i$ denotes all parent-child edges involving node $i$, and $f(h_p, h_c)$ is an edge aggregation function. This formulation enables each node to progressively aggregate information from the entire tree structure through $T$ iterations.
Our architecture integrates optional enhancements including gated aggregation with learnable edge weights, residual connections for gradient stability, and bidirectional propagation for capturing both bottom-up and top-down dependencies. We evaluate TreeGPT on the ARC Prize 2025 dataset, a challenging visual reasoning benchmark requiring abstract pattern recognition and rule inference. Experimental results demonstrate that TreeGPT achieves 96\% accuracy, significantly outperforming transformer baselines (1.3\%), large-scale models like Grok-4 (15.9\%), and specialized program synthesis methods like SOAR (52\%) while using only 1.5M parameters. Our comprehensive ablation study reveals that edge projection is the most critical component, with the combination of edge projection and gating achieving optimal performance. 

---
# From Image Generation to Infrastructure Design: a Multi-agent Pipeline for Street Design Generation 

**Authors**: Chenguang Wang, Xiang Yan, Yilong Dai, Ziyi Wang, Susu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05469)  

**Abstract**: Realistic visual renderings of street-design scenarios are essential for public engagement in active transportation planning. Traditional approaches are labor-intensive, hindering collective deliberation and collaborative decision-making. While AI-assisted generative design shows transformative potential by enabling rapid creation of design scenarios, existing generative approaches typically require large amounts of domain-specific training data and struggle to enable precise spatial variations of design/configuration in complex street-view scenes. We introduce a multi-agent system that edits and redesigns bicycle facilities directly on real-world street-view imagery. The framework integrates lane localization, prompt optimization, design generation, and automated evaluation to synthesize realistic, contextually appropriate designs. Experiments across diverse urban scenarios demonstrate that the system can adapt to varying road geometries and environmental conditions, consistently yielding visually coherent and instruction-compliant results. This work establishes a foundation for applying multi-agent pipelines to transportation infrastructure planning and facility design. 

---
# Murphys Laws of AI Alignment: Why the Gap Always Wins 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2509.05381)  

**Abstract**: Large language models are increasingly aligned to human preferences through reinforcement learning from human feedback (RLHF) and related methods such as Direct Preference Optimization (DPO), Constitutional AI, and RLAIF. While effective, these methods exhibit recurring failure patterns i.e., reward hacking, sycophancy, annotator drift, and misgeneralization. We introduce the concept of the Alignment Gap, a unifying lens for understanding recurring failures in feedback-based alignment. Using a KL-tilting formalism, we illustrate why optimization pressure tends to amplify divergence between proxy rewards and true human intent. We organize these failures into a catalogue of Murphys Laws of AI Alignment, and propose the Alignment Trilemma as a way to frame trade-offs among optimization strength, value capture, and generalization. Small-scale empirical studies serve as illustrative support. Finally, we propose the MAPS framework (Misspecification, Annotation, Pressure, Shift) as practical design levers. Our contribution is not a definitive impossibility theorem but a perspective that reframes alignment debates around structural limits and trade-offs, offering clearer guidance for future design. 

---
# Code Like Humans: A Multi-Agent Solution for Medical Coding 

**Authors**: Andreas Motzfeldt, Joakim Edin, Casper L. Christensen, Christian Hardmeier, Lars Maaløe, Anna Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2509.05378)  

**Abstract**: In medical coding, experts map unstructured clinical notes to alphanumeric codes for diagnoses and procedures. We introduce Code Like Humans: a new agentic framework for medical coding with large language models. It implements official coding guidelines for human experts, and it is the first solution that can support the full ICD-10 coding system (+70K labels). It achieves the best performance to date on rare diagnosis codes (fine-tuned discriminative classifiers retain an advantage for high-frequency codes, to which they are limited). Towards future work, we also contribute an analysis of system performance and identify its `blind spots' (codes that are systematically undercoded). 

---
# Characterizing Fitness Landscape Structures in Prompt Engineering 

**Authors**: Arend Hintze  

**Link**: [PDF](https://arxiv.org/pdf/2509.05375)  

**Abstract**: While prompt engineering has emerged as a crucial technique for optimizing large language model performance, the underlying optimization landscape remains poorly understood. Current approaches treat prompt optimization as a black-box problem, applying sophisticated search algorithms without characterizing the landscape topology they navigate. We present a systematic analysis of fitness landscape structures in prompt engineering using autocorrelation analysis across semantic embedding spaces. Through experiments on error detection tasks with two distinct prompt generation strategies -- systematic enumeration (1,024 prompts) and novelty-driven diversification (1,000 prompts) -- we reveal fundamentally different landscape topologies. Systematic prompt generation yields smoothly decaying autocorrelation, while diversified generation exhibits non-monotonic patterns with peak correlation at intermediate semantic distances, indicating rugged, hierarchically structured landscapes. Task-specific analysis across 10 error detection categories reveals varying degrees of ruggedness across different error types. Our findings provide an empirical foundation for understanding the complexity of optimization in prompt engineering landscapes. 

---
# SasAgent: Multi-Agent AI System for Small-Angle Scattering Data Analysis 

**Authors**: Lijie Ding, Changwoo Do  

**Link**: [PDF](https://arxiv.org/pdf/2509.05363)  

**Abstract**: We introduce SasAgent, a multi-agent AI system powered by large language models (LLMs) that automates small-angle scattering (SAS) data analysis by leveraging tools from the SasView software and enables user interaction via text input. SasAgent features a coordinator agent that interprets user prompts and delegates tasks to three specialized agents for scattering length density (SLD) calculation, synthetic data generation, and experimental data fitting. These agents utilize LLM-friendly tools to execute tasks efficiently. These tools, including the model data tool, Retrieval-Augmented Generation (RAG) documentation tool, bump fitting tool, and SLD calculator tool, are derived from the SasView Python library. A user-friendly Gradio-based interface enhances user accessibility. Through diverse examples, we demonstrate SasAgent's ability to interpret complex prompts, calculate SLDs, generate accurate scattering data, and fit experimental datasets with high precision. This work showcases the potential of LLM-driven AI systems to streamline scientific workflows and enhance automation in SAS research. 

---
# Benchmarking Large Language Models for Personalized Guidance in AI-Enhanced Learning 

**Authors**: Bo Yuan, Jiazi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05346)  

**Abstract**: While Large Language Models (LLMs) are increasingly envisioned as intelligent assistants for personalized learning, systematic head-to-head evaluations within authentic learning scenarios remain limited. This study conducts an empirical comparison of three state-of-the-art LLMs on a tutoring task that simulates a realistic learning setting. Using a dataset comprising a student's answers to ten questions of mixed formats with correctness labels, each LLM is required to (i) analyze the quiz to identify underlying knowledge components, (ii) infer the student's mastery profile, and (iii) generate targeted guidance for improvement. To mitigate subjectivity and evaluator bias, we employ Gemini as a virtual judge to perform pairwise comparisons along various dimensions: accuracy, clarity, actionability, and appropriateness. Results analyzed via the Bradley-Terry model indicate that GPT-4o is generally preferred, producing feedback that is more informative and better structured than its counterparts, while DeepSeek-V3 and GLM-4.5 demonstrate intermittent strengths but lower consistency. These findings highlight the feasibility of deploying LLMs as advanced teaching assistants for individualized support and provide methodological guidance for future empirical research on LLM-driven personalized learning. 

---
# MVRS: The Multimodal Virtual Reality Stimuli-based Emotion Recognition Dataset 

**Authors**: Seyed Muhammad Hossein Mousavi, Atiye Ilanloo  

**Link**: [PDF](https://arxiv.org/pdf/2509.05330)  

**Abstract**: Automatic emotion recognition has become increasingly important with the rise of AI, especially in fields like healthcare, education, and automotive systems. However, there is a lack of multimodal datasets, particularly involving body motion and physiological signals, which limits progress in the field. To address this, the MVRS dataset is introduced, featuring synchronized recordings from 13 participants aged 12 to 60 exposed to VR based emotional stimuli (relaxation, fear, stress, sadness, joy). Data were collected using eye tracking (via webcam in a VR headset), body motion (Kinect v2), and EMG and GSR signals (Arduino UNO), all timestamp aligned. Participants followed a unified protocol with consent and questionnaires. Features from each modality were extracted, fused using early and late fusion techniques, and evaluated with classifiers to confirm the datasets quality and emotion separability, making MVRS a valuable contribution to multimodal affective computing. 

---
# SynDelay: A Synthetic Dataset for Delivery Delay Prediction 

**Authors**: Liming Xu, Yunbo Long, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.05325)  

**Abstract**: Artificial intelligence (AI) is transforming supply chain management, yet progress in predictive tasks -- such as delivery delay prediction -- remains constrained by the scarcity of high-quality, openly available datasets. Existing datasets are often proprietary, small, or inconsistently maintained, hindering reproducibility and benchmarking. We present SynDelay, a synthetic dataset designed for delivery delay prediction. Generated using an advanced generative model trained on real-world data, SynDelay preserves realistic delivery patterns while ensuring privacy. Although not entirely free of noise or inconsistencies, it provides a challenging and practical testbed for advancing predictive modelling. To support adoption, we provide baseline results and evaluation metrics as initial benchmarks, serving as reference points rather than state-of-the-art claims. SynDelay is publicly available through the Supply Chain Data Hub, an open initiative promoting dataset sharing and benchmarking in supply chain AI. We encourage the community to contribute datasets, models, and evaluation practices to advance research in this area. All code is openly accessible at this https URL. 

---
# Perception Graph for Cognitive Attack Reasoning in Augmented Reality 

**Authors**: Rongqian Chen, Shu Hong, Rifatul Islam, Mahdi Imani, G. Gary Tan, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2509.05324)  

**Abstract**: Augmented reality (AR) systems are increasingly deployed in tactical environments, but their reliance on seamless human-computer interaction makes them vulnerable to cognitive attacks that manipulate a user's perception and severely compromise user decision-making. To address this challenge, we introduce the Perception Graph, a novel model designed to reason about human perception within these systems. Our model operates by first mimicking the human process of interpreting key information from an MR environment and then representing the outcomes using a semantically meaningful structure. We demonstrate how the model can compute a quantitative score that reflects the level of perception distortion, providing a robust and measurable method for detecting and analyzing the effects of such cognitive attacks. 

---
# Attention of a Kiss: Exploring Attention Maps in Video Diffusion for XAIxArts 

**Authors**: Adam Cole, Mick Grierson  

**Link**: [PDF](https://arxiv.org/pdf/2509.05323)  

**Abstract**: This paper presents an artistic and technical investigation into the attention mechanisms of video diffusion transformers. Inspired by early video artists who manipulated analog video signals to create new visual aesthetics, this study proposes a method for extracting and visualizing cross-attention maps in generative video models. Built on the open-source Wan model, our tool provides an interpretable window into the temporal and spatial behavior of attention in text-to-video generation. Through exploratory probes and an artistic case study, we examine the potential of attention maps as both analytical tools and raw artistic material. This work contributes to the growing field of Explainable AI for the Arts (XAIxArts), inviting artists to reclaim the inner workings of AI as a creative medium. 

---
# H$_{2}$OT: Hierarchical Hourglass Tokenizer for Efficient Video Pose Transformers 

**Authors**: Wenhao Li, Mengyuan Liu, Hong Liu, Pichao Wang, Shijian Lu, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2509.06956)  

**Abstract**: Transformers have been successfully applied in the field of video-based 3D human pose estimation. However, the high computational costs of these video pose transformers (VPTs) make them impractical on resource-constrained devices. In this paper, we present a hierarchical plug-and-play pruning-and-recovering framework, called Hierarchical Hourglass Tokenizer (H$_{2}$OT), for efficient transformer-based 3D human pose estimation from videos. H$_{2}$OT begins with progressively pruning pose tokens of redundant frames and ends with recovering full-length sequences, resulting in a few pose tokens in the intermediate transformer blocks and thus improving the model efficiency. It works with two key modules, namely, a Token Pruning Module (TPM) and a Token Recovering Module (TRM). TPM dynamically selects a few representative tokens to eliminate the redundancy of video frames, while TRM restores the detailed spatio-temporal information based on the selected tokens, thereby expanding the network output to the original full-length temporal resolution for fast inference. Our method is general-purpose: it can be easily incorporated into common VPT models on both seq2seq and seq2frame pipelines while effectively accommodating different token pruning and recovery strategies. In addition, our H$_{2}$OT reveals that maintaining the full pose sequence is unnecessary, and a few pose tokens of representative frames can achieve both high efficiency and estimation accuracy. Extensive experiments on multiple benchmark datasets demonstrate both the effectiveness and efficiency of the proposed method. Code and models are available at this https URL. 

---
# Deep Reactive Policy: Learning Reactive Manipulator Motion Planning for Dynamic Environments 

**Authors**: Jiahui Yang, Jason Jingzhou Liu, Yulong Li, Youssef Khaky, Kenneth Shaw, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2509.06953)  

**Abstract**: Generating collision-free motion in dynamic, partially observable environments is a fundamental challenge for robotic manipulators. Classical motion planners can compute globally optimal trajectories but require full environment knowledge and are typically too slow for dynamic scenes. Neural motion policies offer a promising alternative by operating in closed-loop directly on raw sensory inputs but often struggle to generalize in complex or dynamic settings. We propose Deep Reactive Policy (DRP), a visuo-motor neural motion policy designed for reactive motion generation in diverse dynamic environments, operating directly on point cloud sensory input. At its core is IMPACT, a transformer-based neural motion policy pretrained on 10 million generated expert trajectories across diverse simulation scenarios. We further improve IMPACT's static obstacle avoidance through iterative student-teacher finetuning. We additionally enhance the policy's dynamic obstacle avoidance at inference time using DCP-RMP, a locally reactive goal-proposal module. We evaluate DRP on challenging tasks featuring cluttered scenes, dynamic moving obstacles, and goal obstructions. DRP achieves strong generalization, outperforming prior classical and neural methods in success rate across both simulated and real-world settings. Video results and code available at this https URL 

---
# Interleaving Reasoning for Better Text-to-Image Generation 

**Authors**: Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, Junbo Qiao, Yue Guo, Yao Hu, Zhenfei Yin, Philip Torr, Yu Cheng, Wanli Ouyang, Shaohui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06945)  

**Abstract**: Unified multimodal understanding and generation models recently have achieve significant improvement in image generation capability, yet a large gap remains in instruction following and detail preservation compared to systems that tightly couple comprehension with generation such as GPT-4o. Motivated by recent advances in interleaving reasoning, we explore whether such reasoning can further improve Text-to-Image (T2I) generation. We introduce Interleaving Reasoning Generation (IRG), a framework that alternates between text-based thinking and image synthesis: the model first produces a text-based thinking to guide an initial image, then reflects on the result to refine fine-grained details, visual quality, and aesthetics while preserving semantics. To train IRG effectively, we propose Interleaving Reasoning Generation Learning (IRGL), which targets two sub-goals: (1) strengthening the initial think-and-generate stage to establish core content and base quality, and (2) enabling high-quality textual reflection and faithful implementation of those refinements in a subsequent image. We curate IRGL-300K, a dataset organized into six decomposed learning modes that jointly cover learning text-based thinking, and full thinking-image trajectories. Starting from a unified foundation model that natively emits interleaved text-image outputs, our two-stage training first builds robust thinking and reflection, then efficiently tunes the IRG pipeline in the full thinking-image trajectory data. Extensive experiments show SoTA performance, yielding absolute gains of 5-10 points on GenEval, WISE, TIIF, GenAI-Bench, and OneIG-EN, alongside substantial improvements in visual quality and fine-grained fidelity. The code, model weights and datasets will be released in: this https URL . 

---
# From Noise to Narrative: Tracing the Origins of Hallucinations in Transformers 

**Authors**: Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, Danilo Bzdok  

**Link**: [PDF](https://arxiv.org/pdf/2509.06938)  

**Abstract**: As generative AI systems become competent and democratized in science, business, and government, deeper insight into their failure modes now poses an acute need. The occasional volatility in their behavior, such as the propensity of transformer models to hallucinate, impedes trust and adoption of emerging AI solutions in high-stakes areas. In the present work, we establish how and when hallucinations arise in pre-trained transformer models through concept representations captured by sparse autoencoders, under scenarios with experimentally controlled uncertainty in the input space. Our systematic experiments reveal that the number of semantic concepts used by the transformer model grows as the input information becomes increasingly unstructured. In the face of growing uncertainty in the input space, the transformer model becomes prone to activate coherent yet input-insensitive semantic features, leading to hallucinated output. At its extreme, for pure-noise inputs, we identify a wide variety of robustly triggered and meaningful concepts in the intermediate activations of pre-trained transformer models, whose functional integrity we confirm through targeted steering. We also show that hallucinations in the output of a transformer model can be reliably predicted from the concept patterns embedded in transformer layer activations. This collection of insights on transformer internal processing mechanics has immediate consequences for aligning AI models with human values, AI safety, opening the attack surface for potential adversarial attacks, and providing a basis for automatic quantification of a model's hallucination risk. 

---
# Neuro-Symbolic AI for Cybersecurity: State of the Art, Challenges, and Opportunities 

**Authors**: Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, Shouhuai Xu, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.06921)  

**Abstract**: Traditional Artificial Intelligence (AI) approaches in cybersecurity exhibit fundamental limitations: inadequate conceptual grounding leading to non-robustness against novel attacks; limited instructibility impeding analyst-guided adaptation; and misalignment with cybersecurity objectives. Neuro-Symbolic (NeSy) AI has emerged with the potential to revolutionize cybersecurity AI. However, there is no systematic understanding of this emerging approach. These hybrid systems address critical cybersecurity challenges by combining neural pattern recognition with symbolic reasoning, enabling enhanced threat understanding while introducing concerning autonomous offensive capabilities that reshape threat landscapes. In this survey, we systematically characterize this field by analyzing 127 publications spanning 2019-July 2025. We introduce a Grounding-Instructibility-Alignment (G-I-A) framework to evaluate these systems, focusing on both cyber defense and cyber offense across network security, malware analysis, and cyber operations. Our analysis shows advantages of multi-agent NeSy architectures and identifies critical implementation challenges including standardization gaps, computational complexity, and human-AI collaboration requirements that constrain deployment. We show that causal reasoning integration is the most transformative advancement, enabling proactive defense beyond correlation-based approaches. Our findings highlight dual-use implications where autonomous systems demonstrate substantial capabilities in zero-day exploitation while achieving significant cost reductions, altering threat dynamics. We provide insights and future research directions, emphasizing the urgent need for community-driven standardization frameworks and responsible development practices that ensure advancement serves defensive cybersecurity objectives while maintaining societal alignment. 

---
# An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection 

**Authors**: Haywood Gelman, John D. Hastings, David Kenley  

**Link**: [PDF](https://arxiv.org/pdf/2509.06920)  

**Abstract**: Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Claude Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including precision, recall, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection. 

---
# Tackling the Noisy Elephant in the Room: Label Noise-robust Out-of-Distribution Detection via Loss Correction and Low-rank Decomposition 

**Authors**: Tarhib Al Azad, Shahana Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2509.06918)  

**Abstract**: Robust out-of-distribution (OOD) detection is an indispensable component of modern artificial intelligence (AI) systems, especially in safety-critical applications where models must identify inputs from unfamiliar classes not seen during training. While OOD detection has been extensively studied in the machine learning literature--with both post hoc and training-based approaches--its effectiveness under noisy training labels remains underexplored. Recent studies suggest that label noise can significantly degrade OOD performance, yet principled solutions to this issue are lacking. In this work, we demonstrate that directly combining existing label noise-robust methods with OOD detection strategies is insufficient to address this critical challenge. To overcome this, we propose a robust OOD detection framework that integrates loss correction techniques from the noisy label learning literature with low-rank and sparse decomposition methods from signal processing. Extensive experiments on both synthetic and real-world datasets demonstrate that our method significantly outperforms the state-of-the-art OOD detection techniques, particularly under severe noisy label settings. 

---
# Barlow-Swin: Toward a novel siamese-based segmentation architecture using Swin-Transformers 

**Authors**: Morteza Kiani Haftlang, Mohammadhossein Malmir, Foroutan Parand, Umberto Michelucci, Safouane El Ghazouali  

**Link**: [PDF](https://arxiv.org/pdf/2509.06885)  

**Abstract**: Medical image segmentation is a critical task in clinical workflows, particularly for the detection and delineation of pathological regions. While convolutional architectures like U-Net have become standard for such tasks, their limited receptive field restricts global context modeling. Recent efforts integrating transformers have addressed this, but often result in deep, computationally expensive models unsuitable for real-time use. In this work, we present a novel end-to-end lightweight architecture designed specifically for real-time binary medical image segmentation. Our model combines a Swin Transformer-like encoder with a U-Net-like decoder, connected via skip pathways to preserve spatial detail while capturing contextual information. Unlike existing designs such as Swin Transformer or U-Net, our architecture is significantly shallower and competitively efficient. To improve the encoder's ability to learn meaningful features without relying on large amounts of labeled data, we first train it using Barlow Twins, a self-supervised learning method that helps the model focus on important patterns by reducing unnecessary repetition in the learned features. After this pretraining, we fine-tune the entire model for our specific task. Experiments on benchmark binary segmentation tasks demonstrate that our model achieves competitive accuracy with substantially reduced parameter count and faster inference, positioning it as a practical alternative for deployment in real-time and resource-limited clinical environments. The code for our method is available at Github repository: this https URL. 

---
# UNH at CheckThat! 2025: Fine-tuning Vs Prompting in Claim Extraction 

**Authors**: Joe Wilder, Nikhil Kadapala, Benji Xu, Mohammed Alsaadi, Aiden Parsons, Mitchell Rogers, Palash Agarwal, Adam Hassick, Laura Dietz  

**Link**: [PDF](https://arxiv.org/pdf/2509.06883)  

**Abstract**: We participate in CheckThat! Task 2 English and explore various methods of prompting and in-context learning, including few-shot prompting and fine-tuning with different LLM families, with the goal of extracting check-worthy claims from social media passages. Our best METEOR score is achieved by fine-tuning a FLAN-T5 model. However, we observe that higher-quality claims can sometimes be extracted using other methods, even when their METEOR scores are lower. 

---
# AxelSMOTE: An Agent-Based Oversampling Algorithm for Imbalanced Classification 

**Authors**: Sukumar Kishanthan, Asela Hevapathige  

**Link**: [PDF](https://arxiv.org/pdf/2509.06875)  

**Abstract**: Class imbalance in machine learning poses a significant challenge, as skewed datasets often hinder performance on minority classes. Traditional oversampling techniques, which are commonly used to alleviate class imbalance, have several drawbacks: they treat features independently, lack similarity-based controls, limit sample diversity, and fail to manage synthetic variety effectively. To overcome these issues, we introduce AxelSMOTE, an innovative agent-based approach that views data instances as autonomous agents engaging in complex interactions. Based on Axelrod's cultural dissemination model, AxelSMOTE implements four key innovations: (1) trait-based feature grouping to preserve correlations; (2) a similarity-based probabilistic exchange mechanism for meaningful interactions; (3) Beta distribution blending for realistic interpolation; and (4) controlled diversity injection to avoid overfitting. Experiments on eight imbalanced datasets demonstrate that AxelSMOTE outperforms state-of-the-art sampling methods while maintaining computational efficiency. 

---
# floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based RL 

**Authors**: Bhavya Agrawalla, Michal Nauman, Khush Agarwal, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06863)  

**Abstract**: A hallmark of modern large-scale machine learning techniques is the use of training objectives that provide dense supervision to intermediate computations, such as teacher forcing the next token in language models or denoising step-by-step in diffusion models. This enables models to learn complex functions in a generalizable manner. Motivated by this observation, we investigate the benefits of iterative computation for temporal difference (TD) methods in reinforcement learning (RL). Typically they represent value functions in a monolithic fashion, without iterative compute. We introduce floq (flow-matching Q-functions), an approach that parameterizes the Q-function using a velocity field and trains it using techniques from flow-matching, typically used in generative modeling. This velocity field underneath the flow is trained using a TD-learning objective, which bootstraps from values produced by a target velocity field, computed by running multiple steps of numerical integration. Crucially, floq allows for more fine-grained control and scaling of the Q-function capacity than monolithic architectures, by appropriately setting the number of integration steps. Across a suite of challenging offline RL benchmarks and online fine-tuning tasks, floq improves performance by nearly 1.8x. floq scales capacity far better than standard TD-learning architectures, highlighting the potential of iterative computation for value learning. 

---
# Disentangling Interaction and Bias Effects in Opinion Dynamics of Large Language Models 

**Authors**: Vincent C. Brockers, David A. Ehrlich, Viola Priesemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.06858)  

**Abstract**: Large Language Models are increasingly used to simulate human opinion dynamics, yet the effect of genuine interaction is often obscured by systematic biases. We present a Bayesian framework to disentangle and quantify three such biases: (i) a topic bias toward prior opinions in the training data; (ii) an agreement bias favoring agreement irrespective of the question; and (iii) an anchoring bias toward the initiating agent's stance. Applying this framework to multi-step dialogues reveals that opinion trajectories tend to quickly converge to a shared attractor, with the influence of the interaction fading over time, and the impact of biases differing between LLMs. In addition, we fine-tune an LLM on different sets of strongly opinionated statements (incl. misinformation) and demonstrate that the opinion attractor shifts correspondingly. Exposing stark differences between LLMs and providing quantitative tools to compare them to human subjects in the future, our approach highlights both chances and pitfalls in using LLMs as proxies for human behavior. 

---
# Automated Radiographic Total Sharp Score (ARTSS) in Rheumatoid Arthritis: A Solution to Reduce Inter-Intra Reader Variation and Enhancing Clinical Practice 

**Authors**: Hajar Moradmand, Lei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.06854)  

**Abstract**: Assessing the severity of rheumatoid arthritis (RA) using the Total Sharp/Van Der Heijde Score (TSS) is crucial, but manual scoring is often time-consuming and subjective. This study introduces an Automated Radiographic Sharp Scoring (ARTSS) framework that leverages deep learning to analyze full-hand X-ray images, aiming to reduce inter- and intra-observer variability. The research uniquely accommodates patients with joint disappearance and variable-length image sequences. We developed ARTSS using data from 970 patients, structured into four stages: I) Image pre-processing and re-orientation using ResNet50, II) Hand segmentation using UNet.3, III) Joint identification using YOLOv7, and IV) TSS prediction using models such as VGG16, VGG19, ResNet50, DenseNet201, EfficientNetB0, and Vision Transformer (ViT). We evaluated model performance with Intersection over Union (IoU), Mean Average Precision (MAP), mean absolute error (MAE), Root Mean Squared Error (RMSE), and Huber loss. The average TSS from two radiologists was used as the ground truth. Model training employed 3-fold cross-validation, with each fold consisting of 452 training and 227 validation samples, and external testing included 291 unseen subjects. Our joint identification model achieved 99% accuracy. The best-performing model, ViT, achieved a notably low Huber loss of 0.87 for TSS prediction. Our results demonstrate the potential of deep learning to automate RA scoring, which can significantly enhance clinical practice. Our approach addresses the challenge of joint disappearance and variable joint numbers, offers timesaving benefits, reduces inter- and intra-reader variability, improves radiologist accuracy, and aids rheumatologists in making more informed decisions. 

---
# Reinforcement learning meets bioprocess control through behaviour cloning: Real-world deployment in an industrial photobioreactor 

**Authors**: Juan D. Gil, Ehecatl Antonio Del Rio Chanona, José L. Guzmán, Manuel Berenguel  

**Link**: [PDF](https://arxiv.org/pdf/2509.06853)  

**Abstract**: The inherent complexity of living cells as production units creates major challenges for maintaining stable and optimal bioprocess conditions, especially in open Photobioreactors (PBRs) exposed to fluctuating environments. To address this, we propose a Reinforcement Learning (RL) control approach, combined with Behavior Cloning (BC), for pH regulation in open PBR systems. This represents, to the best of our knowledge, the first application of an RL-based control strategy to such a nonlinear and disturbance-prone bioprocess. Our method begins with an offline training stage in which the RL agent learns from trajectories generated by a nominal Proportional-Integral-Derivative (PID) controller, without direct interaction with the real system. This is followed by a daily online fine-tuning phase, enabling adaptation to evolving process dynamics and stronger rejection of fast, transient disturbances. This hybrid offline-online strategy allows deployment of an adaptive control policy capable of handling the inherent nonlinearities and external perturbations in open PBRs. Simulation studies highlight the advantages of our method: the Integral of Absolute Error (IAE) was reduced by 8% compared to PID control and by 5% relative to standard off-policy RL. Moreover, control effort decreased substantially-by 54% compared to PID and 7% compared to standard RL-an important factor for minimizing operational costs. Finally, an 8-day experimental validation under varying environmental conditions confirmed the robustness and reliability of the proposed approach. Overall, this work demonstrates the potential of RL-based methods for bioprocess control and paves the way for their broader application to other nonlinear, disturbance-prone systems. 

---
# COMPACT: Common-token Optimized Model Pruning Across Channels and Tokens 

**Authors**: Eugene Kwek, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06836)  

**Abstract**: Making LLMs more efficient in memory, latency, and serving cost is crucial for edge deployment, interactive applications, and sustainable inference at scale. Pruning is a key technique toward this goal. However, prior pruning methods are limited: width pruning often breaks the standard transformer layout or requires custom inference code, while depth pruning removes entire layers and can cause abrupt accuracy drops. In this work, we propose COMPACT, which jointly (i) prunes rare vocabulary to shrink embedding/unembedding and (ii) prunes FFN intermediate channels using common-token-weighted activations, aligning importance with the post-pruning token distribution. COMPACT enjoys merits of both depth and width pruning, such as: deployment-friendliness (keeps a standard transformer architecture), scale-adaptivity (trade off vocab vs. FFN pruning), training-free operation with competitive pruning time, and strong memory savings alongside throughput gains. Experiments across Qwen, LLaMA, and Gemma families (0.5B-70B) show state-of-the-art downstream task performance at similar or higher pruning ratios, with substantial reductions in parameters, GPU memory, and end-to-end latency. 

---
# Saturation-Driven Dataset Generation for LLM Mathematical Reasoning in the TPTP Ecosystem 

**Authors**: Valentin Quesnel, Damien Sileo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06809)  

**Abstract**: The scarcity of high-quality, logically sound data is a critical bottleneck for advancing the mathematical reasoning of Large Language Models (LLMs). Our work confronts this challenge by turning decades of automated theorem proving research into a scalable data engine. Rather than relying on error-prone LLMs or complex proof-assistant syntax like Lean and Isabelle, our framework leverages E-prover's saturation capabilities on the vast TPTP axiom library to derive a massive, guaranteed-valid corpus of theorems. Our pipeline is principled and simple: saturate axioms, filter for "interesting" theorems, and generate tasks. With no LLMs in the loop, we eliminate factual errors by construction. This purely symbolic data is then transformed into three difficulty-controlled challenges: entailment verification, premise selection, and proof reconstruction. Our zero-shot experiments on frontier models reveal a clear weakness: performance collapses on tasks requiring deep, structural reasoning. Our framework provides both the diagnostic tool to measure this gap and a scalable source of symbolic training data to address it. We make the code and data publicly available.
this https URL this https URL 

---
# MachineLearningLM: Continued Pretraining Language Models on Millions of Synthetic Tabular Prediction Tasks Scales In-Context ML 

**Authors**: Haoyu Dong, Pengkun Zhang, Mingzhe Lu, Yanzhen Shen, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2509.06806)  

**Abstract**: Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine learning (ML) tasks, that is, to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MachineLearningLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust in-context ML capability while preserving its general knowledge and reasoning for broader chat workflows.
Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a random-forest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3x to 6x more examples per context window and delivering up to 50x amortized throughput via batch inference.
Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MachineLearningLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of about 15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest-level accuracy across hundreds of shots. General chat capabilities, including knowledge and reasoning, are preserved: it achieves 75.4% on MMLU. 

---
# Aligning Large Vision-Language Models by Deep Reinforcement Learning and Direct Preference Optimization 

**Authors**: Thanh Thi Nguyen, Campbell Wilson, Janis Dalins  

**Link**: [PDF](https://arxiv.org/pdf/2509.06759)  

**Abstract**: Large Vision-Language Models (LVLMs) or multimodal large language models represent a significant advancement in artificial intelligence, enabling systems to understand and generate content across both visual and textual modalities. While large-scale pretraining has driven substantial progress, fine-tuning these models for aligning with human values or engaging in specific tasks or behaviors remains a critical challenge. Deep Reinforcement Learning (DRL) and Direct Preference Optimization (DPO) offer promising frameworks for this aligning process. While DRL enables models to optimize actions using reward signals instead of relying solely on supervised preference data, DPO directly aligns the policy with preferences, eliminating the need for an explicit reward model. This overview explores paradigms for fine-tuning LVLMs, highlighting how DRL and DPO techniques can be used to align models with human preferences and values, improve task performance, and enable adaptive multimodal interaction. We categorize key approaches, examine sources of preference data, reward signals, and discuss open challenges such as scalability, sample efficiency, continual learning, generalization, and safety. The goal is to provide a clear understanding of how DRL and DPO contribute to the evolution of robust and human-aligned LVLMs. 

---
# Long-Range Graph Wavelet Networks 

**Authors**: Filippo Guerranti, Fabrizio Forte, Simon Geisler, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.06743)  

**Abstract**: Modeling long-range interactions, the propagation of information across distant parts of a graph, is a central challenge in graph machine learning. Graph wavelets, inspired by multi-resolution signal processing, provide a principled way to capture both local and global structures. However, existing wavelet-based graph neural networks rely on finite-order polynomial approximations, which limit their receptive fields and hinder long-range propagation. We propose Long-Range Graph Wavelet Networks (LR-GWN), which decompose wavelet filters into complementary local and global components. Local aggregation is handled with efficient low-order polynomials, while long-range interactions are captured through a flexible spectral domain parameterization. This hybrid design unifies short- and long-distance information flow within a principled wavelet framework. Experiments show that LR-GWN achieves state-of-the-art performance among wavelet-based methods on long-range benchmarks, while remaining competitive on short-range datasets. 

---
# MRI-Based Brain Tumor Detection through an Explainable EfficientNetV2 and MLP-Mixer-Attention Architecture 

**Authors**: Mustafa Yurdakul, Şakir Taşdemir  

**Link**: [PDF](https://arxiv.org/pdf/2509.06713)  

**Abstract**: Brain tumors are serious health problems that require early diagnosis due to their high mortality rates. Diagnosing tumors by examining Magnetic Resonance Imaging (MRI) images is a process that requires expertise and is prone to error. Therefore, the need for automated diagnosis systems is increasing day by day. In this context, a robust and explainable Deep Learning (DL) model for the classification of brain tumors is proposed. In this study, a publicly available Figshare dataset containing 3,064 T1-weighted contrast-enhanced brain MRI images of three tumor types was used. First, the classification performance of nine well-known CNN architectures was evaluated to determine the most effective backbone. Among these, EfficientNetV2 demonstrated the best performance and was selected as the backbone for further development. Subsequently, an attention-based MLP-Mixer architecture was integrated into EfficientNetV2 to enhance its classification capability. The performance of the final model was comprehensively compared with basic CNNs and the methods in the literature. Additionally, Grad-CAM visualization was used to interpret and validate the decision-making process of the proposed model. The proposed model's performance was evaluated using the five-fold cross-validation method. The proposed model demonstrated superior performance with 99.50% accuracy, 99.47% precision, 99.52% recall and 99.49% F1 score. The results obtained show that the model outperforms the studies in the literature. Moreover, Grad-CAM visualizations demonstrate that the model effectively focuses on relevant regions of MRI images, thus improving interpretability and clinical reliability. A robust deep learning model for clinical decision support systems has been obtained by combining EfficientNetV2 and attention-based MLP-Mixer, providing high accuracy and interpretability in brain tumor classification. 

---
# Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks 

**Authors**: Su Hyeong Lee, Risi Kondor, Richard Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06701)  

**Abstract**: We develop a theory of intelligent agency grounded in probabilistic modeling for neural models. Agents are represented as outcome distributions with epistemic utility given by log score, and compositions are defined through weighted logarithmic pooling that strictly improves every member's welfare. We prove that strict unanimity is impossible under linear pooling or in binary outcome spaces, but possible with three or more outcomes. Our framework admits recursive structure via cloning invariance, continuity, and openness, while tilt-based analysis rules out trivial duplication. Finally, we formalize an agentic alignment phenomenon in LLMs using our theory: eliciting a benevolent persona ("Luigi'") induces an antagonistic counterpart ("Waluigi"), while a manifest-then-suppress Waluigi strategy yields strictly larger first-order misalignment reduction than pure Luigi reinforcement alone. These results clarify how developing a principled mathematical framework for how subagents can coalesce into coherent higher-level entities provides novel implications for alignment in agentic AI systems. 

---
# Barycentric Neural Networks and Length-Weighted Persistent Entropy Loss: A Green Geometric and Topological Framework for Function Approximation 

**Authors**: Victor Toscano-Duran, Rocio Gonzalez-Diaz, Miguel A. Gutiérrez-Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06694)  

**Abstract**: While it is well-established that artificial neural networks are \emph{universal approximators} for continuous functions on compact domains, many modern approaches rely on deep or overparameterized architectures that incur high computational costs. In this paper, a new type of \emph{small shallow} neural network, called the \emph{Barycentric Neural Network} ($\BNN$), is proposed, which leverages a fixed set of \emph{base points} and their \emph{barycentric coordinates} to define both its structure and its parameters. We demonstrate that our $\BNN$ enables the exact representation of \emph{continuous piecewise linear functions} ($\CPLF$s), ensuring strict continuity across segments. Since any continuous function over a compact domain can be approximated arbitrarily well by $\CPLF$s, the $\BNN$ naturally emerges as a flexible and interpretable tool for \emph{function approximation}. Beyond the use of this representation, the main contribution of the paper is the introduction of a new variant of \emph{persistent entropy}, a topological feature that is stable and scale invariant, called the \emph{length-weighted persistent entropy} ($\LWPE$), which is weighted by the lifetime of topological features. Our framework, which combines the $\BNN$ with a loss function based on our $\LWPE$, aims to provide flexible and geometrically interpretable approximations of nonlinear continuous functions in resource-constrained settings, such as those with limited base points for $\BNN$ design and few training epochs. Instead of optimizing internal weights, our approach directly \emph{optimizes the base points that define the $\BNN$}. Experimental results show that our approach achieves \emph{superior and faster approximation performance} compared to classical loss functions such as MSE, RMSE, MAE, and log-cosh. 

---
# BioLite U-Net: Edge-Deployable Semantic Segmentation for In Situ Bioprinting Monitoring 

**Authors**: Usman Haider, Lukasz Szemet, Daniel Kelly, Vasileios Sergis, Andrew C. Daly, Karl Mason  

**Link**: [PDF](https://arxiv.org/pdf/2509.06690)  

**Abstract**: Bioprinting is a rapidly advancing field that offers a transformative approach to fabricating tissue and organ models through the precise deposition of cell-laden bioinks. Ensuring the fidelity and consistency of printed structures in real-time remains a core challenge, particularly under constraints imposed by limited imaging data and resource-constrained embedded hardware. Semantic segmentation of the extrusion process, differentiating between nozzle, extruded bioink, and surrounding background, enables in situ monitoring critical to maintaining print quality and biological viability. In this work, we introduce a lightweight semantic segmentation framework tailored for real-time bioprinting applications. We present a novel, manually annotated dataset comprising 787 RGB images captured during the bioprinting process, labeled across three classes: nozzle, bioink, and background. To achieve fast and efficient inference suitable for integration with bioprinting systems, we propose a BioLite U-Net architecture that leverages depthwise separable convolutions to drastically reduce computational load without compromising accuracy. Our model is benchmarked against MobileNetV2 and MobileNetV3-based segmentation baselines using mean Intersection over Union (mIoU), Dice score, and pixel accuracy. All models were evaluated on a Raspberry Pi 4B to assess real-world feasibility. The proposed BioLite U-Net achieves an mIoU of 92.85% and a Dice score of 96.17%, while being over 1300x smaller than MobileNetV2-DeepLabV3+. On-device inference takes 335 ms per frame, demonstrating near real-time capability. Compared to MobileNet baselines, BioLite U-Net offers a superior tradeoff between segmentation accuracy, efficiency, and deployability, making it highly suitable for intelligent, closed-loop bioprinting systems. 

---
# TrajAware: Graph Cross-Attention and Trajectory-Aware for Generalisable VANETs under Partial Observations 

**Authors**: Xiaolu Fu, Ziyuan Bao, Eiman Kanjo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06665)  

**Abstract**: Vehicular ad hoc networks (VANETs) are a crucial component of intelligent transportation systems; however, routing remains challenging due to dynamic topologies, incomplete observations, and the limited resources of edge devices. Existing reinforcement learning (RL) approaches often assume fixed graph structures and require retraining when network conditions change, making them unsuitable for deployment on constrained hardware. We present TrajAware, an RL-based framework designed for edge AI deployment in VANETs. TrajAware integrates three components: (i) action space pruning, which reduces redundant neighbour options while preserving two-hop reachability, alleviating the curse of dimensionality; (ii) graph cross-attention, which maps pruned neighbours to the global graph context, producing features that generalise across diverse network sizes; and (iii) trajectory-aware prediction, which uses historical routes and junction information to estimate real-time positions under partial observations. We evaluate TrajAware in the open-source SUMO simulator using real-world city maps with a leave-one-city-out setup. Results show that TrajAware achieves near-shortest paths and high delivery ratios while maintaining efficiency suitable for constrained edge devices, outperforming state-of-the-art baselines in both full and partial observation scenarios. 

---
# AnalysisGNN: Unified Music Analysis with Graph Neural Networks 

**Authors**: Emmanouil Karystinaios, Johannes Hentschel, Markus Neuwirth, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.06654)  

**Abstract**: Recent years have seen a boom in computational approaches to music analysis, yet each one is typically tailored to a specific analytical domain. In this work, we introduce AnalysisGNN, a novel graph neural network framework that leverages a data-shuffling strategy with a custom weighted multi-task loss and logit fusion between task-specific classifiers to integrate heterogeneously annotated symbolic datasets for comprehensive score analysis. We further integrate a Non-Chord-Tone prediction module, which identifies and excludes passing and non-functional notes from all tasks, thereby improving the consistency of label signals. Experimental evaluations demonstrate that AnalysisGNN achieves performance comparable to traditional static-dataset approaches, while showing increased resilience to domain shifts and annotation inconsistencies across multiple heterogeneous corpora. 

---
# The First Voice Timbre Attribute Detection Challenge 

**Authors**: Liping Chen, Jinghao He, Zhengyan Sheng, Kong Aik Lee, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.06635)  

**Abstract**: The first voice timbre attribute detection challenge is featured in a special session at NCMMSC 2025. It focuses on the explainability of voice timbre and compares the intensity of two speech utterances in a specified timbre descriptor dimension. The evaluation was conducted on the VCTK-RVA dataset. Participants developed their systems and submitted their outputs to the organizer, who evaluated the performance and sent feedback to them. Six teams submitted their outputs, with five providing descriptions of their methodologies. 

---
# Improved Classification of Nitrogen Stress Severity in Plants Under Combined Stress Conditions Using Spatio-Temporal Deep Learning Framework 

**Authors**: Aswini Kumar Patra  

**Link**: [PDF](https://arxiv.org/pdf/2509.06625)  

**Abstract**: Plants in their natural habitats endure an array of interacting stresses, both biotic and abiotic, that rarely occur in isolation. Nutrient stress-particularly nitrogen deficiency-becomes even more critical when compounded with drought and weed competition, making it increasingly difficult to distinguish and address its effects. Early detection of nitrogen stress is therefore crucial for protecting plant health and implementing effective management strategies. This study proposes a novel deep learning framework to accurately classify nitrogen stress severity in a combined stress environment. Our model uses a unique blend of four imaging modalities-RGB, multispectral, and two infrared wavelengths-to capture a wide range of physiological plant responses from canopy images. These images, provided as time-series data, document plant health across three levels of nitrogen availability (low, medium, and high) under varying water stress and weed pressures. The core of our approach is a spatio-temporal deep learning pipeline that merges a Convolutional Neural Network (CNN) for extracting spatial features from images with a Long Short-Term Memory (LSTM) network to capture temporal dependencies. We also devised and evaluated a spatial-only CNN pipeline for comparison. Our CNN-LSTM pipeline achieved an impressive accuracy of 98%, impressively surpassing the spatial-only model's 80.45% and other previously reported machine learning method's 76%. These results bring actionable insights based on the power of our CNN-LSTM approach in effectively capturing the subtle and complex interactions between nitrogen deficiency, water stress, and weed pressure. This robust platform offers a promising tool for the timely and proactive identification of nitrogen stress severity, enabling better crop management and improved plant health. 

---
# BEAM: Brainwave Empathy Assessment Model for Early Childhood 

**Authors**: Chen Xie, Gaofeng Wu, Kaidong Wang, Zihao Zhu, Xiaoshu Luo, Yan Liang, Feiyu Quan, Ruoxi Wu, Xianghui Huang, Han Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06620)  

**Abstract**: Empathy in young children is crucial for their social and emotional development, yet predicting it remains challenging. Traditional methods often only rely on self-reports or observer-based labeling, which are susceptible to bias and fail to objectively capture the process of empathy formation. EEG offers an objective alternative; however, current approaches primarily extract static patterns, neglecting temporal dynamics. To overcome these limitations, we propose a novel deep learning framework, the Brainwave Empathy Assessment Model (BEAM), to predict empathy levels in children aged 4-6 years. BEAM leverages multi-view EEG signals to capture both cognitive and emotional dimensions of empathy. The framework comprises three key components: 1) a LaBraM-based encoder for effective spatio-temporal feature extraction, 2) a feature fusion module to integrate complementary information from multi-view signals, and 3) a contrastive learning module to enhance class separation. Validated on the CBCP dataset, BEAM outperforms state-of-the-art methods across multiple metrics, demonstrating its potential for objective empathy assessment and providing a preliminary insight into early interventions in children's prosocial development. 

---
# Demo: Healthcare Agent Orchestrator (HAO) for Patient Summarization in Molecular Tumor Boards 

**Authors**: Noel Codella, Sam Preston, Hao Qiu, Leonardo Schettini, Wen-wai Yim, Mert Öz, Shrey Jain, Matthew P. Lungren, Thomas Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2509.06602)  

**Abstract**: Molecular Tumor Boards (MTBs) are multidisciplinary forums where oncology specialists collaboratively assess complex patient cases to determine optimal treatment strategies. A central element of this process is the patient summary, typically compiled by a medical oncologist, radiation oncologist, or surgeon, or their trained medical assistant, who distills heterogeneous medical records into a concise narrative to facilitate discussion. This manual approach is often labor-intensive, subjective, and prone to omissions of critical information. To address these limitations, we introduce the Healthcare Agent Orchestrator (HAO), a Large Language Model (LLM)-driven AI agent that coordinates a multi-agent clinical workflow to generate accurate and comprehensive patient summaries for MTBs. Evaluating predicted patient summaries against ground truth presents additional challenges due to stylistic variation, ordering, synonym usage, and phrasing differences, which complicate the measurement of both succinctness and completeness. To overcome these evaluation hurdles, we propose TBFact, a ``model-as-a-judge'' framework designed to assess the comprehensiveness and succinctness of generated summaries. Using a benchmark dataset derived from de-identified tumor board discussions, we applied TBFact to evaluate our Patient History agent. Results show that the agent captured 94% of high-importance information (including partial entailments) and achieved a TBFact recall of 0.84 under strict entailment criteria. We further demonstrate that TBFact enables a data-free evaluation framework that institutions can deploy locally without sharing sensitive clinical data. Together, HAO and TBFact establish a robust foundation for delivering reliable and scalable support to MTBs. 

---
# Integrating Spatial and Semantic Embeddings for Stereo Sound Event Localization in Videos 

**Authors**: Davide Berghi, Philip J. B. Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2509.06598)  

**Abstract**: In this study, we address the multimodal task of stereo sound event localization and detection with source distance estimation (3D SELD) in regular video content. 3D SELD is a complex task that combines temporal event classification with spatial localization, requiring reasoning across spatial, temporal, and semantic dimensions. The last is arguably the most challenging to model. Traditional SELD approaches typically rely on multichannel input, limiting their capacity to benefit from large-scale pre-training due to data constraints. To overcome this, we enhance a standard SELD architecture with semantic information by integrating pre-trained, contrastive language-aligned models: CLAP for audio and OWL-ViT for visual inputs. These embeddings are incorporated into a modified Conformer module tailored for multimodal fusion, which we refer to as the Cross-Modal Conformer. We perform an ablation study on the development set of the DCASE2025 Task3 Stereo SELD Dataset to assess the individual contributions of the language-aligned models and benchmark against the DCASE Task 3 baseline systems. Additionally, we detail the curation process of large synthetic audio and audio-visual datasets used for model pre-training. These datasets were further expanded through left-right channel swapping augmentation. Our approach, combining extensive pre-training, model ensembling, and visual post-processing, achieved second rank in the DCASE 2025 Challenge Task 3 (Track B), underscoring the effectiveness of our method. Future work will explore the modality-specific contributions and architectural refinements. 

---
# HAVE: Head-Adaptive Gating and ValuE Calibration for Hallucination Mitigation in Large Language Models 

**Authors**: Xin Tong, Zhi Lin, Jingya Wang, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06596)  

**Abstract**: Large Language Models (LLMs) often produce hallucinations in retrieval-augmented or long-context generation, even when relevant evidence is present. This stems from two issues: head importance is treated as input-agnostic, and raw attention weights poorly reflect each token's true contribution. We present HAVE (Head-Adaptive Gating and ValuE Calibration), a parameter-free decoding framework that directly addresses both challenges. HAVE introduces head-adaptive gating, which performs instance-level soft reweighing of attention heads, and value calibration, which augments attention with the magnitude of value vectors to approximate write-back contribution. Together, these modules construct token-level evidence aligned with model updates and fuse it with the LM distribution through a lightweight uncertainty-scaled policy. HAVE requires no finetuning and operates in a single forward pass, making it efficient and broadly applicable. Experiments across multiple QA benchmarks and LLM families demonstrate that HAVE consistently reduces hallucinations and outperforms strong baselines, including DAGCD, with modest overhead. The framework is transparent, reproducible, and readily integrates with off-the-shelf LLMs, advancing trustworthy generation in real-world settings. 

---
# Integrated Detection and Tracking Based on Radar Range-Doppler Feature 

**Authors**: Chenyu Zhang, Yuanhang Wu, Xiaoxi Ma, Wei Yi  

**Link**: [PDF](https://arxiv.org/pdf/2509.06569)  

**Abstract**: Detection and tracking are the basic tasks of radar systems. Current joint detection tracking methods, which focus on dynamically adjusting detection thresholds from tracking results, still present challenges in fully utilizing the potential of radar signals. These are mainly reflected in the limited capacity of the constant false-alarm rate model to accurately represent information, the insufficient depiction of complex scenes, and the limited information acquired by the tracker. We introduce the Integrated Detection and Tracking based on radar feature (InDT) method, which comprises a network architecture for radar signal detection and a tracker that leverages detection assistance. The InDT detector extracts feature information from each Range-Doppler (RD) matrix and then returns the target position through the feature enhancement module and the detection head. The InDT tracker adaptively updates the measurement noise covariance of the Kalman filter based on detection confidence. The similarity of target RD features is measured by cosine distance, which enhances the data association process by combining location and feature information. Finally, the efficacy of the proposed method was validated through testing on both simulated data and publicly available datasets. 

---
# Contrastive Self-Supervised Network Intrusion Detection using Augmented Negative Pairs 

**Authors**: Jack Wilkie, Hanan Hindy, Christos Tachtatzis, Robert Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2509.06550)  

**Abstract**: Network intrusion detection remains a critical challenge in cybersecurity. While supervised machine learning models achieve state-of-the-art performance, their reliance on large labelled datasets makes them impractical for many real-world applications. Anomaly detection methods, which train exclusively on benign traffic to identify malicious activity, suffer from high false positive rates, limiting their usability. Recently, self-supervised learning techniques have demonstrated improved performance with lower false positive rates by learning discriminative latent representations of benign traffic. In particular, contrastive self-supervised models achieve this by minimizing the distance between similar (positive) views of benign traffic while maximizing it between dissimilar (negative) views. Existing approaches generate positive views through data augmentation and treat other samples as negative. In contrast, this work introduces Contrastive Learning using Augmented Negative pairs (CLAN), a novel paradigm for network intrusion detection where augmented samples are treated as negative views - representing potentially malicious distributions - while other benign samples serve as positive views. This approach enhances both classification accuracy and inference efficiency after pretraining on benign traffic. Experimental evaluation on the Lycos2017 dataset demonstrates that the proposed method surpasses existing self-supervised and anomaly detection techniques in a binary classification task. Furthermore, when fine-tuned on a limited labelled dataset, the proposed approach achieves superior multi-class classification performance compared to existing self-supervised models. 

---
# Signal-Based Malware Classification Using 1D CNNs 

**Authors**: Jack Wilkie, Hanan Hindy, Ivan Andonovic, Christos Tachtatzis, Robert Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2509.06548)  

**Abstract**: Malware classification is a contemporary and ongoing challenge in cyber-security: modern obfuscation techniques are able to evade traditional static analysis, while dynamic analysis is too resource intensive to be deployed at a large scale. One prominent line of research addresses these limitations by converting malware binaries into 2D images by heuristically reshaping them into a 2D grid before resizing using Lanczos resampling. These images can then be classified based on their textural information using computer vision approaches. While this approach can detect obfuscated malware more effectively than static analysis, the process of converting files into 2D images results in significant information loss due to both quantisation noise, caused by rounding to integer pixel values, and the introduction of 2D dependencies which do not exist in the original data. This loss of signal limits the classification performance of the downstream model. This work addresses these weaknesses by instead resizing the files into 1D signals which avoids the need for heuristic reshaping, and additionally these signals do not suffer from quantisation noise due to being stored in a floating-point format. It is shown that existing 2D CNN architectures can be readily adapted to classify these 1D signals for improved performance. Furthermore, a bespoke 1D convolutional neural network, based on the ResNet architecture and squeeze-and-excitation layers, was developed to classify these signals and evaluated on the MalNet dataset. It was found to achieve state-of-the-art performance on binary, type, and family level classification with F1 scores of 0.874, 0.503, and 0.507, respectively, paving the way for future models to operate on the proposed signal modality. 

---
# Learning Optimal Defender Strategies for CAGE-2 using a POMDP Model 

**Authors**: Duc Huy Le, Rolf Stadler  

**Link**: [PDF](https://arxiv.org/pdf/2509.06539)  

**Abstract**: CAGE-2 is an accepted benchmark for learning and evaluating defender strategies against cyberattacks. It reflects a scenario where a defender agent protects an IT infrastructure against various attacks. Many defender methods for CAGE-2 have been proposed in the literature. In this paper, we construct a formal model for CAGE-2 using the framework of Partially Observable Markov Decision Process (POMDP). Based on this model, we define an optimal defender strategy for CAGE-2 and introduce a method to efficiently learn this strategy. Our method, called BF-PPO, is based on PPO, and it uses particle filter to mitigate the computational complexity due to the large state space of the CAGE-2 model. We evaluate our method in the CAGE-2 CybORG environment and compare its performance with that of CARDIFF, the highest ranked method on the CAGE-2 leaderboard. We find that our method outperforms CARDIFF regarding the learned defender strategy and the required training time. 

---
# On the Reproducibility of "FairCLIP: Harnessing Fairness in Vision-Language Learning'' 

**Authors**: Hua Chang Bakker, Stan Fris, Angela Madelon Bernardy, Stan Deutekom  

**Link**: [PDF](https://arxiv.org/pdf/2509.06535)  

**Abstract**: We investigated the reproducibility of FairCLIP, proposed by Luo et al. (2024), for improving the group fairness of CLIP (Radford et al., 2021) by minimizing image-text similarity score disparities across sensitive groups using the Sinkhorn distance. The experimental setup of Luo et al. (2024) was reproduced to primarily investigate the research findings for FairCLIP. The model description by Luo et al. (2024) was found to differ from the original implementation. Therefore, a new implementation, A-FairCLIP, is introduced to examine specific design choices. Furthermore, FairCLIP+ is proposed to extend the FairCLIP objective to include multiple attributes. Additionally, the impact of the distance minimization on FairCLIP's fairness and performance was explored. In alignment with the original authors, CLIP was found to be biased towards certain demographics when applied to zero-shot glaucoma classification using medical scans and clinical notes from the Harvard-FairVLMed dataset. However, the experimental results on two datasets do not support their claim that FairCLIP improves the performance and fairness of CLIP. Although the regularization objective reduces Sinkhorn distances, both the official implementation and the aligned implementation, A-FairCLIP, were not found to improve performance nor fairness in zero-shot glaucoma classification. 

---
# SLiNT: Structure-aware Language Model with Injection and Contrastive Training for Knowledge Graph Completion 

**Authors**: Mengxue Yang, Chun Yang, Jiaqi Zhu, Jiafan Li, Jingqi Zhang, Yuyang Li, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06531)  

**Abstract**: Link prediction in knowledge graphs requires integrating structural information and semantic context to infer missing entities. While large language models offer strong generative reasoning capabilities, their limited exploitation of structural signals often results in structural sparsity and semantic ambiguity, especially under incomplete or zero-shot settings. To address these challenges, we propose SLiNT (Structure-aware Language model with Injection and coNtrastive Training), a modular framework that injects knowledge-graph-derived structural context into a frozen LLM backbone with lightweight LoRA-based adaptation for robust link prediction. Specifically, Structure-Guided Neighborhood Enhancement (SGNE) retrieves pseudo-neighbors to enrich sparse entities and mitigate missing context; Dynamic Hard Contrastive Learning (DHCL) introduces fine-grained supervision by interpolating hard positives and negatives to resolve entity-level ambiguity; and Gradient-Decoupled Dual Injection (GDDI) performs token-level structure-aware intervention while preserving the core LLM parameters. Experiments on WN18RR and FB15k-237 show that SLiNT achieves superior or competitive performance compared with both embedding-based and generation-based baselines, demonstrating the effectiveness of structure-aware representation learning for scalable knowledge graph completion. 

---
# Crown, Frame, Reverse: Layer-Wise Scaling Variants for LLM Pre-Training 

**Authors**: Andrei Baroian, Kasper Notebomer  

**Link**: [PDF](https://arxiv.org/pdf/2509.06518)  

**Abstract**: Transformer-based language models traditionally use uniform (isotropic) layer sizes, yet they ignore the diverse functional roles that different depths can play and their computational capacity needs. Building on Layer-Wise Scaling (LWS) and pruning literature, we introduce three new LWS variants - Framed, Reverse, and Crown - that redistribute FFN widths and attention heads via two or three-point linear interpolation in the pre-training stage. We present the first systematic ablation of LWS and its variants, on a fixed budget of 180M parameters, trained on 5B tokens. All models converge to similar losses and achieve better performance compared to an equal-cost isotropic baseline, without a substantial decrease in training throughput. This work represents an initial step into the design space of layer-wise architectures for pre-training, but future work should scale experiments to orders of magnitude more tokens and parameters to fully assess their potential. 

---
# QualityFM: a Multimodal Physiological Signal Foundation Model with Self-Distillation for Signal Quality Challenges in Critically Ill Patients 

**Authors**: Zongheng Guo, Tao Chen, Manuela Ferrario  

**Link**: [PDF](https://arxiv.org/pdf/2509.06516)  

**Abstract**: Photoplethysmogram (PPG) and electrocardiogram (ECG) are commonly recorded in intesive care unit (ICU) and operating room (OR). However, the high incidence of poor, incomplete, and inconsistent signal quality, can lead to false alarms or diagnostic inaccuracies. The methods explored so far suffer from limited generalizability, reliance on extensive labeled data, and poor cross-task transferability. To overcome these challenges, we introduce QualityFM, a novel multimodal foundation model for these physiological signals, designed to acquire a general-purpose understanding of signal quality. Our model is pre-trained on an large-scale dataset comprising over 21 million 30-second waveforms and 179,757 hours of data. Our approach involves a dual-track architecture that processes paired physiological signals of differing quality, leveraging a self-distillation strategy where an encoder for high-quality signals is used to guide the training of an encoder for low-quality signals. To efficiently handle long sequential signals and capture essential local quasi-periodic patterns, we integrate a windowed sparse attention mechanism within our Transformer-based model. Furthermore, a composite loss function, which combines direct distillation loss on encoder outputs with indirect reconstruction loss based on power and phase spectra, ensures the preservation of frequency-domain characteristics of the signals. We pre-train three models with varying parameter counts (9.6 M to 319 M) and demonstrate their efficacy and practical value through transfer learning on three distinct clinical tasks: false alarm of ventricular tachycardia detection, the identification of atrial fibrillation and the estimation of arterial blood pressure (ABP) from PPG and ECG signals. 

---
# DyC-STG: Dynamic Causal Spatio-Temporal Graph Network for Real-time Data Credibility Analysis in IoT 

**Authors**: Guanjie Cheng, Boyi Li, Peihan Wu, Feiyi Chen, Xinkui Zhao, Mengying Zhu, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06483)  

**Abstract**: The wide spreading of Internet of Things (IoT) sensors generates vast spatio-temporal data streams, but ensuring data credibility is a critical yet unsolved challenge for applications like smart homes. While spatio-temporal graph (STG) models are a leading paradigm for such data, they often fall short in dynamic, human-centric environments due to two fundamental limitations: (1) their reliance on static graph topologies, which fail to capture physical, event-driven dynamics, and (2) their tendency to confuse spurious correlations with true causality, undermining robustness in human-centric environments. To address these gaps, we propose the Dynamic Causal Spatio-Temporal Graph Network (DyC-STG), a novel framework designed for real-time data credibility analysis in IoT. Our framework features two synergistic contributions: an event-driven dynamic graph module that adapts the graph topology in real-time to reflect physical state changes, and a causal reasoning module to distill causally-aware representations by strictly enforcing temporal precedence. To facilitate the research in this domain we release two new real-world datasets. Comprehensive experiments show that DyC-STG establishes a new state-of-the-art, outperforming the strongest baselines by 1.4 percentage points and achieving an F1-Score of up to 0.930. 

---
# Explained, yet misunderstood: How AI Literacy shapes HR Managers' interpretation of User Interfaces in Recruiting Recommender Systems 

**Authors**: Yannick Kalff, Katharina Simbeck  

**Link**: [PDF](https://arxiv.org/pdf/2509.06475)  

**Abstract**: AI-based recommender systems increasingly influence recruitment decisions. Thus, transparency and responsible adoption in Human Resource Management (HRM) are critical. This study examines how HR managers' AI literacy influences their subjective perception and objective understanding of explainable AI (XAI) elements in recruiting recommender dashboards. In an online experiment, 410 German-based HR managers compared baseline dashboards to versions enriched with three XAI styles: important features, counterfactuals, and model criteria. Our results show that the dashboards used in practice do not explain AI results and even keep AI elements opaque. However, while adding XAI features improves subjective perceptions of helpfulness and trust among users with moderate or high AI literacy, it does not increase their objective understanding. It may even reduce accurate understanding, especially with complex explanations. Only overlays of important features significantly aided the interpretations of high-literacy users. Our findings highlight that the benefits of XAI in recruitment depend on users' AI literacy, emphasizing the need for tailored explanation strategies and targeted literacy training in HRM to ensure fair, transparent, and effective adoption of AI. 

---
# Several Performance Bounds on Decentralized Online Optimization are Highly Conservative and Potentially Misleading 

**Authors**: Erwan Meunier, Julien M. Hendrickx  

**Link**: [PDF](https://arxiv.org/pdf/2509.06466)  

**Abstract**: We analyze Decentralized Online Optimization algorithms using the Performance Estimation Problem approach which allows, to automatically compute exact worst-case performance of optimization algorithms. Our analysis shows that several available performance guarantees are very conservative, sometimes by multiple orders of magnitude, and can lead to misguided choices of algorithm. Moreover, at least in terms of worst-case performance, some algorithms appear not to benefit from inter-agent communications for a significant period of time. We show how to improve classical methods by tuning their step-sizes, and find that we can save up to 20% on their actual worst-case performance regret. 

---
# Focusing by Contrastive Attention: Enhancing VLMs' Visual Reasoning 

**Authors**: Yuyao Ge, Shenghua Liu, Yiwei Wang, Lingrui Mei, Baolong Bi, Xuanshan Zhou, Jiayu Yao, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06461)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable success across diverse visual tasks, yet their performance degrades in complex visual environments. While existing enhancement approaches require additional training, rely on external segmentation tools, or operate at coarse-grained levels, they overlook the innate ability within VLMs. To bridge this gap, we investigate VLMs' attention patterns and discover that: (1) visual complexity strongly correlates with attention entropy, negatively impacting reasoning performance; (2) attention progressively refines from global scanning in shallow layers to focused convergence in deeper layers, with convergence degree determined by visual complexity. (3) Theoretically, we prove that the contrast of attention maps between general queries and task-specific queries enables the decomposition of visual signal into semantic signals and visual noise components. Building on these insights, we propose Contrastive Attention Refinement for Visual Enhancement (CARVE), a training-free method that extracts task-relevant visual signals through attention contrasting at the pixel level. Extensive experiments demonstrate that CARVE consistently enhances performance, achieving up to 75% improvement on open-source models. Our work provides critical insights into the interplay between visual complexity and attention mechanisms, offering an efficient pathway for improving visual reasoning with contrasting attention. 

---
# HECATE: An ECS-based Framework for Teaching and Developing Multi-Agent Systems 

**Authors**: Arthur Casals, Anarosa A. F. Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2509.06431)  

**Abstract**: This paper introduces HECATE, a novel framework based on the Entity-Component-System (ECS) architectural pattern that bridges the gap between distributed systems engineering and MAS development. HECATE is built using the Entity-Component-System architectural pattern, leveraging data-oriented design to implement multiagent systems. This approach involves engineering multiagent systems (MAS) from a distributed systems (DS) perspective, integrating agent concepts directly into the DS domain. This approach simplifies MAS development by (i) reducing the need for specialized agent knowledge and (ii) leveraging familiar DS patterns and standards to minimize the agent-specific knowledge required for engineering MAS. We present the framework's architecture, core components, and implementation approach, demonstrating how it supports different agent models. 

---
# Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster 

**Authors**: Pembe Gizem Özdil, Chuanfang Ning, Jasper S. Phelps, Sibo Wang-Chen, Guy Elisha, Alexander Blanke, Auke Ijspeert, Pavan Ramdya  

**Link**: [PDF](https://arxiv.org/pdf/2509.06426)  

**Abstract**: Computational models are critical to advance our understanding of how neural, biomechanical, and physical systems interact to orchestrate animal behaviors. Despite the availability of near-complete reconstructions of the Drosophila melanogaster central nervous system, musculature, and exoskeleton, anatomically and physically grounded models of fly leg muscles are still missing. These models provide an indispensable bridge between motor neuron activity and joint movements. Here, we introduce the first 3D, data-driven musculoskeletal model of Drosophila legs, implemented in both OpenSim and MuJoCo simulation environments. Our model incorporates a Hill-type muscle representation based on high-resolution X-ray scans from multiple fixed specimens. We present a pipeline for constructing muscle models using morphological imaging data and for optimizing unknown muscle parameters specific to the fly. We then combine our musculoskeletal models with detailed 3D pose estimation data from behaving flies to achieve muscle-actuated behavioral replay in OpenSim. Simulations of muscle activity across diverse walking and grooming behaviors predict coordinated muscle synergies that can be tested experimentally. Furthermore, by training imitation learning policies in MuJoCo, we test the effect of different passive joint properties on learning speed and find that damping and stiffness facilitate learning. Overall, our model enables the investigation of motor control in an experimentally tractable model organism, providing insights into how biomechanics contribute to generation of complex limb movements. Moreover, our model can be used to control embodied artificial agents to generate naturalistic and compliant locomotion in simulated environments. 

---
# CAPMix: Robust Time Series Anomaly Detection Based on Abnormal Assumptions with Dual-Space Mixup 

**Authors**: Xudong Mou, Rui Wang, Tiejun Wang, Renyu Yang, Shiru Chen, Jie Sun, Tianyu Wo, Xudong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06419)  

**Abstract**: Time series anomaly detection (TSAD) is a vital yet challenging task, particularly in scenarios where labeled anomalies are scarce and temporal dependencies are complex. Recent anomaly assumption (AA) approaches alleviate the lack of anomalies by injecting synthetic samples and training discriminative models. Despite promising results, these methods often suffer from two fundamental limitations: patchy generation, where scattered anomaly knowledge leads to overly simplistic or incoherent anomaly injection, and Anomaly Shift, where synthetic anomalies either resemble normal data too closely or diverge unrealistically from real anomalies, thereby distorting classification boundaries. In this paper, we propose CAPMix, a controllable anomaly augmentation framework that addresses both issues. First, we design a CutAddPaste mechanism to inject diverse and complex anomalies in a targeted manner, avoiding patchy generation. Second, we introduce a label revision strategy to adaptively refine anomaly labels, reducing the risk of anomaly shift. Finally, we employ dual-space mixup within a temporal convolutional network to enforce smoother and more robust decision boundaries. Extensive experiments on five benchmark datasets, including AIOps, UCR, SWaT, WADI, and ESA, demonstrate that CAPMix achieves significant improvements over state-of-the-art baselines, with enhanced robustness against contaminated training data. The code is available at this https URL. 

---
# Index-Preserving Lightweight Token Pruning for Efficient Document Understanding in Vision-Language Models 

**Authors**: Jaemin Son, Sujin Choi, Inyong Yun  

**Link**: [PDF](https://arxiv.org/pdf/2509.06415)  

**Abstract**: Recent progress in vision-language models (VLMs) has led to impressive results in document understanding tasks, but their high computational demands remain a challenge. To mitigate the compute burdens, we propose a lightweight token pruning framework that filters out non-informative background regions from document images prior to VLM processing. A binary patch-level classifier removes non-text areas, and a max-pooling refinement step recovers fragmented text regions to enhance spatial coherence. Experiments on real-world document datasets demonstrate that our approach substantially lowers computational costs, while maintaining comparable accuracy. 

---
# MeanFlow-Accelerated Multimodal Video-to-Audio Synthesis via One-Step Generation 

**Authors**: Xiaoran Yang, Jianxuan Yang, Xinyue Guo, Haoyu Wang, Ningning Pan, Gongping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06389)  

**Abstract**: A key challenge in synthesizing audios from silent videos is the inherent trade-off between synthesis quality and inference efficiency in existing methods. For instance, flow matching based models rely on modeling instantaneous velocity, inherently require an iterative sampling process, leading to slow inference speeds. To address this efficiency bottleneck, we introduce a MeanFlow-accelerated model that characterizes flow fields using average velocity, enabling one-step generation and thereby significantly accelerating multimodal video-to-audio (VTA) synthesis while preserving audio quality, semantic alignment, and temporal synchronization. Furthermore, a scalar rescaling mechanism is employed to balance conditional and unconditional predictions when classifier-free guidance (CFG) is applied, effectively mitigating CFG-induced distortions in one step generation. Since the audio synthesis network is jointly trained with multimodal conditions, we further evaluate it on text-to-audio (TTA) synthesis task. Experimental results demonstrate that incorporating MeanFlow into the network significantly improves inference speed without compromising perceptual quality on both VTA and TTA synthesis tasks. 

---
# Beyond the Pre-Service Horizon: Infusing In-Service Behavior for Improved Financial Risk Forecasting 

**Authors**: Senhao Liu, Zhiyu Guo, Zhiyuan Ji, Yueguo Chen, Yateng Tang, Yunhai Wang, Xuehao Zheng, Xiang Ao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06385)  

**Abstract**: Typical financial risk management involves distinct phases for pre-service risk assessment and in-service default detection, often modeled separately. This paper proposes a novel framework, Multi-Granularity Knowledge Distillation (abbreviated as MGKD), aimed at improving pre-service risk prediction through the integration of in-service user behavior data. MGKD follows the idea of knowledge distillation, where the teacher model, trained on historical in-service data, guides the student model, which is trained on pre-service data. By using soft labels derived from in-service data, the teacher model helps the student model improve its risk prediction prior to service activation. Meanwhile, a multi-granularity distillation strategy is introduced, including coarse-grained, fine-grained, and self-distillation, to align the representations and predictions of the teacher and student models. This approach not only reinforces the representation of default cases but also enables the transfer of key behavioral patterns associated with defaulters from the teacher to the student model, thereby improving the overall performance of pre-service risk assessment. Moreover, we adopt a re-weighting strategy to mitigate the model's bias towards the minority class. Experimental results on large-scale real-world datasets from Tencent Mobile Payment demonstrate the effectiveness of our proposed approach in both offline and online scenarios. 

---
# MRD-LiNet: A Novel Lightweight Hybrid CNN with Gradient-Guided Unlearning for Improved Drought Stress Identification 

**Authors**: Aswini Kumar Patra, Lingaraj Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06367)  

**Abstract**: Drought stress is a major threat to global crop productivity, making its early and precise detection essential for sustainable agricultural management. Traditional approaches, though useful, are often time-consuming and labor-intensive, which has motivated the adoption of deep learning methods. In recent years, Convolutional Neural Network (CNN) and Vision Transformer architectures have been widely explored for drought stress identification; however, these models generally rely on a large number of trainable parameters, restricting their use in resource-limited and real-time agricultural settings. To address this challenge, we propose a novel lightweight hybrid CNN framework inspired by ResNet, DenseNet, and MobileNet architectures. The framework achieves a remarkable 15-fold reduction in trainable parameters compared to conventional CNN and Vision Transformer models, while maintaining competitive accuracy. In addition, we introduce a machine unlearning mechanism based on a gradient norm-based influence function, which enables targeted removal of specific training data influence, thereby improving model adaptability. The method was evaluated on an aerial image dataset of potato fields with expert-annotated healthy and drought-stressed regions. Experimental results show that our framework achieves high accuracy while substantially lowering computational costs. These findings highlight its potential as a practical, scalable, and adaptive solution for drought stress monitoring in precision agriculture, particularly under resource-constrained conditions. 

---
# PL-CA: A Parametric Legal Case Augmentation Framework 

**Authors**: Ao Chang, Yubo Chen, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06356)  

**Abstract**: Conventional RAG is considered one of the most effective methods for addressing model knowledge insufficiency and hallucination, particularly in the judicial domain that requires high levels of knowledge rigor, logical consistency, and content integrity. However, the conventional RAG method only injects retrieved documents directly into the model's context, which severely constrains models due to their limited context windows and introduces additional computational overhead through excessively long contexts, thereby disrupting models' attention and degrading performance on downstream tasks. Moreover, many existing benchmarks lack expert annotation and focus solely on individual downstream tasks while real-world legal scenarios consist of multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for reflecting models' true capabilities. To address these limitations, we propose PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data augmentation on corpus knowledge and encode this legal knowledge into parametric vectors, and then integrates this parametric knowledge into the LLM's feed-forward networks (FFN) via LoRA, thereby alleviating models' context pressure. Additionally, we also construct a multi-task legal dataset comprising more than 2000 training and test instances, which are all expert-annotated and manually verified. We conduct our experiments on our dataset, and the experimental results demonstrate that our method reduces the overhead associated with excessively long contexts while maintaining competitive performance on downstream tasks compared to conventional RAG. Our code and dataset are provided in the appendix. 

---
# Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks? 

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06350)  

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks. 

---
# Ban&Pick: Achieving Free Performance Gains and Inference Speedup via Smarter Routing in MoE-LLMs 

**Authors**: Yuanteng Chen, Peisong Wang, Yuantian Shao, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06346)  

**Abstract**: Sparse Mixture-of-Experts (MoE) has become a key architecture for scaling large language models (LLMs) efficiently. Recent fine-grained MoE designs introduce hundreds of experts per layer, with multiple experts activated per token, enabling stronger specialization. However, during pre-training, routers are optimized mainly for stability and robustness: they converge prematurely and enforce balanced usage, limiting the full potential of model performance and efficiency. In this work, we uncover two overlooked issues: (i) a few highly influential experts are underutilized due to premature and balanced routing decisions; and (ii) enforcing a fixed number of active experts per token introduces substantial redundancy. Instead of retraining models or redesigning MoE architectures, we introduce Ban&Pick, a post-training, plug-and-play strategy for smarter MoE routing. Pick discovers and reinforces key experts-a small group with outsized impact on performance-leading to notable accuracy gains across domains. Ban complements this by dynamically pruning redundant experts based on layer and token sensitivity, delivering faster inference with minimal accuracy loss. Experiments on fine-grained MoE-LLMs (DeepSeek, Qwen3) across math, code, and general reasoning benchmarks demonstrate that Ban&Pick delivers free performance gains and inference acceleration without retraining or architectural changes. For instance, on Qwen3-30B-A3B, it improves accuracy from 80.67 to 84.66 on AIME2024 and from 65.66 to 68.18 on GPQA-Diamond, while accelerating inference by 1.25x under the vLLM. 

---
# Multi View Slot Attention Using Paraphrased Texts For Face Anti-Spoofing 

**Authors**: Jeongmin Yu, Susang Kim, Kisu Lee, Taekyoung Kwon, Won-Yong Shin, Ha Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.06336)  

**Abstract**: Recent face anti-spoofing (FAS) methods have shown remarkable cross-domain performance by employing vision-language models like CLIP. However, existing CLIP-based FAS models do not fully exploit CLIP's patch embedding tokens, failing to detect critical spoofing clues. Moreover, these models rely on a single text prompt per class (e.g., 'live' or 'fake'), which limits generalization. To address these issues, we propose MVP-FAS, a novel framework incorporating two key modules: Multi-View Slot attention (MVS) and Multi-Text Patch Alignment (MTPA). Both modules utilize multiple paraphrased texts to generate generalized features and reduce dependence on domain-specific text. MVS extracts local detailed spatial features and global context from patch embeddings by leveraging diverse texts with multiple perspectives. MTPA aligns patches with multiple text representations to improve semantic robustness. Extensive experiments demonstrate that MVP-FAS achieves superior generalization performance, outperforming previous state-of-the-art methods on cross-domain datasets. Code: this https URL. 

---
# A Fragile Number Sense: Probing the Elemental Limits of Numerical Reasoning in LLMs 

**Authors**: Roussel Rahman, Aashwin Ananda Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.06332)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable emergent capabilities, yet the robustness of their numerical reasoning remains an open question. While standard benchmarks evaluate LLM reasoning on complex problem sets using aggregated metrics, they often obscure foundational weaknesses. In this work, we probe LLM mathematical numeracy by evaluating performance on problems of escalating complexity, from constituent operations to combinatorial puzzles. We test several state-of-the-art LLM-based agents on a 100-problem challenge comprising four categories: (1) basic arithmetic, (2) advanced operations, (3) primality checking, and (4) the Game of 24 number puzzle. Our results show that while the agents achieved high accuracy on the first three categories, which require deterministic algorithmic execution, they consistently failed at the number puzzle, underlining its demand for a heuristic search over a large combinatorial space to be a significant bottleneck. These findings reveal that the agents' proficiency is largely confined to recalling and executing known algorithms, rather than performing generative problem-solving. This suggests their apparent numerical reasoning is more akin to sophisticated pattern-matching than flexible, analytical thought, limiting their potential for tasks that require novel or creative numerical insights. 

---
# AttestLLM: Efficient Attestation Framework for Billion-scale On-device LLMs 

**Authors**: Ruisi Zhang, Yifei Zhao, Neusha Javidnia, Mengxin Zheng, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06326)  

**Abstract**: As on-device LLMs(e.g., Apple on-device Intelligence) are widely adopted to reduce network dependency, improve privacy, and enhance responsiveness, verifying the legitimacy of models running on local devices becomes critical. Existing attestation techniques are not suitable for billion-parameter Large Language Models (LLMs), struggling to remain both time- and memory-efficient while addressing emerging threats in the LLM era. In this paper, we present AttestLLM, the first-of-its-kind attestation framework to protect the hardware-level intellectual property (IP) of device vendors by ensuring that only authorized LLMs can execute on target platforms. AttestLLM leverages an algorithm/software/hardware co-design approach to embed robust watermarking signatures onto the activation distributions of LLM building blocks. It also optimizes the attestation protocol within the Trusted Execution Environment (TEE), providing efficient verification without compromising inference throughput. Extensive proof-of-concept evaluations on LLMs from Llama, Qwen, and Phi families for on-device use cases demonstrate AttestLLM's attestation reliability, fidelity, and efficiency. Furthermore, AttestLLM enforces model legitimacy and exhibits resilience against model replacement and forgery attacks. 

---
# Learning to Walk with Less: a Dyna-Style Approach to Quadrupedal Locomotion 

**Authors**: Francisco Affonso, Felipe Andrade G. Tommaselli, Juliano Negri, Vivian S. Medeiros, Mateus V. Gasparino, Girish Chowdhary, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.06296)  

**Abstract**: Traditional RL-based locomotion controllers often suffer from low data efficiency, requiring extensive interaction to achieve robust performance. We present a model-based reinforcement learning (MBRL) framework that improves sample efficiency for quadrupedal locomotion by appending synthetic data to the end of standard rollouts in PPO-based controllers, following the Dyna-Style paradigm. A predictive model, trained alongside the policy, generates short-horizon synthetic transitions that are gradually integrated using a scheduling strategy based on the policy update iterations. Through an ablation study, we identified a strong correlation between sample efficiency and rollout length, which guided the design of our experiments. We validated our approach in simulation on the Unitree Go1 robot and showed that replacing part of the simulated steps with synthetic ones not only mimics extended rollouts but also improves policy return and reduces variance. Finally, we demonstrate that this improvement transfers to the ability to track a wide range of locomotion commands using fewer simulated steps. 

---
# Statistical Inference for Misspecified Contextual Bandits 

**Authors**: Yongyi Guo, Ziping Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06287)  

**Abstract**: Contextual bandit algorithms have transformed modern experimentation by enabling real-time adaptation for personalized treatment and efficient use of data. Yet these advantages create challenges for statistical inference due to adaptivity. A fundamental property that supports valid inference is policy convergence, meaning that action-selection probabilities converge in probability given the context. Convergence ensures replicability of adaptive experiments and stability of online algorithms. In this paper, we highlight a previously overlooked issue: widely used algorithms such as LinUCB may fail to converge when the reward model is misspecified, and such non-convergence creates fundamental obstacles for statistical inference. This issue is practically important, as misspecified models -- such as linear approximations of complex dynamic system -- are often employed in real-world adaptive experiments to balance bias and variance.
Motivated by this insight, we propose and analyze a broad class of algorithms that are guaranteed to converge even under model misspecification. Building on this guarantee, we develop a general inference framework based on an inverse-probability-weighted Z-estimator (IPW-Z) and establish its asymptotic normality with a consistent variance estimator. Simulation studies confirm that the proposed method provides robust and data-efficient confidence intervals, and can outperform existing approaches that exist only in the special case of offline policy evaluation. Taken together, our results underscore the importance of designing adaptive algorithms with built-in convergence guarantees to enable stable experimentation and valid statistical inference in practice. 

---
# UrbanMIMOMap: A Ray-Traced MIMO CSI Dataset with Precoding-Aware Maps and Benchmarks 

**Authors**: Honggang Jia, Xiucheng Wang, Nan Cheng, Ruijin Sun, Changle Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06270)  

**Abstract**: Sixth generation (6G) systems require environment-aware communication, driven by native artificial intelligence (AI) and integrated sensing and communication (ISAC). Radio maps (RMs), providing spatially continuous channel information, are key enablers. However, generating high-fidelity RM ground truth via electromagnetic (EM) simulations is computationally intensive, motivating machine learning (ML)-based RM construction. The effectiveness of these data-driven methods depends on large-scale, high-quality training data. Current public datasets often focus on single-input single-output (SISO) and limited information, such as path loss, which is insufficient for advanced multi-input multi-output (MIMO) systems requiring detailed channel state information (CSI). To address this gap, this paper presents UrbanMIMOMap, a novel large-scale urban MIMO CSI dataset generated using high-precision ray tracing. UrbanMIMOMap offers comprehensive complex CSI matrices across a dense spatial grid, going beyond traditional path loss data. This rich CSI is vital for constructing high-fidelity RMs and serves as a fundamental resource for data-driven RM generation, including deep learning. We demonstrate the dataset's utility through baseline performance evaluations of representative ML methods for RM construction. This work provides a crucial dataset and reference for research in high-precision RM generation, MIMO spatial performance, and ML for 6G environment awareness. The code and data for this work are available at: this https URL. 

---
# On Synthesis of Timed Regular Expressions 

**Authors**: Ziran Wang, Jie An, Naijun Zhan, Miaomiao Zhang, Zhenya Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06262)  

**Abstract**: Timed regular expressions serve as a formalism for specifying real-time behaviors of Cyber-Physical Systems. In this paper, we consider the synthesis of timed regular expressions, focusing on generating a timed regular expression consistent with a given set of system behaviors including positive and negative examples, i.e., accepting all positive examples and rejecting all negative examples. We first prove the decidability of the synthesis problem through an exploration of simple timed regular expressions. Subsequently, we propose our method of generating a consistent timed regular expression with minimal length, which unfolds in two steps. The first step is to enumerate and prune candidate parametric timed regular expressions. In the second step, we encode the requirement that a candidate generated by the first step is consistent with the given set into a Satisfiability Modulo Theories (SMT) formula, which is consequently solved to determine a solution to parametric time constraints. Finally, we evaluate our approach on benchmarks, including randomly generated behaviors from target timed models and a case study. 

---
# Distillation of CNN Ensemble Results for Enhanced Long-Term Prediction of the ENSO Phenomenon 

**Authors**: Saghar Ganji, Mohammad Naisipour, Alireza Hassani, Arash Adib  

**Link**: [PDF](https://arxiv.org/pdf/2509.06227)  

**Abstract**: The accurate long-term forecasting of the El Nino Southern Oscillation (ENSO) is still one of the biggest challenges in climate science. While it is true that short-to medium-range performance has been improved significantly using the advances in deep learning, statistical dynamical hybrids, most operational systems still use the simple mean of all ensemble members, implicitly assuming equal skill across members. In this study, we demonstrate, through a strictly a-posteriori evaluation , for any large enough ensemble of ENSO forecasts, there is a subset of members whose skill is substantially higher than that of the ensemble mean. Using a state-of-the-art ENSO forecast system cross-validated against the 1986-2017 observed Nino3.4 index, we identify two Top-5 subsets one ranked on lowest Root Mean Square Error (RMSE) and another on highest Pearson correlation. Generally across all leads, these outstanding members show higher correlation and lower RMSE, with the advantage rising enormously with lead time. Whereas at short leads (1 month) raises the mean correlation by about +0.02 (+1.7%) and lowers the RMSE by around 0.14 °C or by 23.3% compared to the All-40 mean, at extreme leads (23 months) the correlation is raised by +0.43 (+172%) and RMSE by 0.18 °C or by 22.5% decrease. The enhancements are largest during crucial ENSO transition periods such as SON and DJF, when accurate amplitude and phase forecasting is of greatest socio-economic benefit, and furthermore season-dependent e.g., mid-year months such as JJA and MJJ have incredibly large RMSE reductions. This study provides a solid foundation for further investigations to identify reliable clues for detecting high-quality ensemble members, thereby enhancing forecasting skill. 

---
# Beamforming-LLM: What, Where and When Did I Miss? 

**Authors**: Vishal Choudhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.06221)  

**Abstract**: We present Beamforming-LLM, a system that enables users to semantically recall conversations they may have missed in multi-speaker environments. The system combines spatial audio capture using a microphone array with retrieval-augmented generation (RAG) to support natural language queries such as, "What did I miss when I was following the conversation on dogs?" Directional audio streams are separated using beamforming, transcribed with Whisper, and embedded into a vector database using sentence encoders. Upon receiving a user query, semantically relevant segments are retrieved, temporally aligned with non-attended segments, and summarized using a lightweight large language model (GPT-4o-mini). The result is a user-friendly interface that provides contrastive summaries, spatial context, and timestamped audio playback. This work lays the foundation for intelligent auditory memory systems and has broad applications in assistive technology, meeting summarization, and context-aware personal spatial computing. 

---
# The Efficiency Frontier: Classical Shadows versus Quantum Footage 

**Authors**: Shuowei Ma, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06218)  

**Abstract**: Interfacing quantum and classical processors is an important subroutine in full-stack quantum algorithms. The so-called "classical shadow" method efficiently extracts essential classical information from quantum states, enabling the prediction of many properties of a quantum system from only a few measurements. However, for a small number of highly non-local observables, or when classical post-processing power is limited, the classical shadow method is not always the most efficient choice. Here, we address this issue quantitatively by performing a full-stack resource analysis that compares classical shadows with ``quantum footage," which refers to direct quantum measurement. Under certain assumptions, our analysis illustrates a boundary of download efficiency between classical shadows and quantum footage. For observables expressed as linear combinations of Pauli matrices, the classical shadow method outperforms direct measurement when the number of observables is large and the Pauli weight is small. For observables in the form of large Hermitian sparse matrices, the classical shadow method shows an advantage when the number of observables, the sparsity of the matrix, and the number of qubits fall within a certain range. The key parameters influencing this behavior include the number of qubits $n$, observables $M$, sparsity $k$, Pauli weight $w$, accuracy requirement $\epsilon$, and failure tolerance $\delta$. We also compare the resource consumption of the two methods on different types of quantum computers and identify break-even points where the classical shadow method becomes more efficient, which vary depending on the hardware. This paper opens a new avenue for quantitatively designing optimal strategies for hybrid quantum-classical tomography and provides practical insights for selecting the most suitable quantum measurement approach in real-world applications. 

---
# Agentic Software Engineering: Foundational Pillars and a Research Roadmap 

**Authors**: Ahmed E. Hassan, Hao Li, Dayi Lin, Bram Adams, Tse-Hsun Chen, Yutaro Kashiwa, Dong Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06216)  

**Abstract**: Agentic Software Engineering (SE 3.0) represents a new era where intelligent agents are tasked not with simple code generation, but with achieving complex, goal-oriented SE objectives. To harness these new capabilities while ensuring trustworthiness, we must recognize a fundamental duality within the SE field in the Agentic SE era, comprising two symbiotic modalities: SE for Humans and SE for Agents. This duality demands a radical reimagining of the foundational pillars of SE (actors, processes, tools, and artifacts) which manifest differently across each modality. We propose two purpose-built workbenches to support this vision. The Agent Command Environment (ACE) serves as a command center where humans orchestrate and mentor agent teams, handling outputs such as Merge-Readiness Packs (MRPs) and Consultation Request Packs (CRPs). The Agent Execution Environment (AEE) is a digital workspace where agents perform tasks while invoking human expertise when facing ambiguity or complex trade-offs. This bi-directional partnership, which supports agent-initiated human callbacks and handovers, gives rise to new, structured engineering activities (i.e., processes) that redefine human-AI collaboration, elevating the practice from agentic coding to true agentic software engineering. This paper presents the Structured Agentic Software Engineering (SASE) vision, outlining several of the foundational pillars for the future of SE. The paper culminates in a research roadmap that identifies a few key challenges and opportunities while briefly discussing the resulting impact of this future on SE education. Our goal is not to offer a definitive solution, but to provide a conceptual scaffold with structured vocabulary to catalyze a community-wide dialogue, pushing the SE community to think beyond its classic, human-centric tenets toward a disciplined, scalable, and trustworthy agentic future. 

---
# Toward a Metrology for Artificial Intelligence: Hidden-Rule Environments and Reinforcement Learning 

**Authors**: Christo Mathew, Wentian Wang, Lazaros Gallos, Paul Kantor, Vladimir Menkov, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06213)  

**Abstract**: We investigate reinforcement learning in the Game Of Hidden Rules (GOHR) environment, a complex puzzle in which an agent must infer and execute hidden rules to clear a 6$\times$6 board by placing game pieces into buckets. We explore two state representation strategies, namely Feature-Centric (FC) and Object-Centric (OC), and employ a Transformer-based Advantage Actor-Critic (A2C) algorithm for training. The agent has access only to partial observations and must simultaneously infer the governing rule and learn the optimal policy through experience. We evaluate our models across multiple rule-based and trial-list-based experimental setups, analyzing transfer effects and the impact of representation on learning efficiency. 

---
# Grasp-MPC: Closed-Loop Visual Grasping via Value-Guided Model Predictive Control 

**Authors**: Jun Yamada, Adithyavairavan Murali, Ajay Mandlekar, Clemens Eppner, Ingmar Posner, Balakumar Sundaralingam  

**Link**: [PDF](https://arxiv.org/pdf/2509.06201)  

**Abstract**: Grasping of diverse objects in unstructured environments remains a significant challenge. Open-loop grasping methods, effective in controlled settings, struggle in cluttered environments. Grasp prediction errors and object pose changes during grasping are the main causes of failure. In contrast, closed-loop methods address these challenges in simplified settings (e.g., single object on a table) on a limited set of objects, with no path to generalization. We propose Grasp-MPC, a closed-loop 6-DoF vision-based grasping policy designed for robust and reactive grasping of novel objects in cluttered environments. Grasp-MPC incorporates a value function, trained on visual observations from a large-scale synthetic dataset of 2 million grasp trajectories that include successful and failed attempts. We deploy this learned value function in an MPC framework in combination with other cost terms that encourage collision avoidance and smooth execution. We evaluate Grasp-MPC on FetchBench and real-world settings across diverse environments. Grasp-MPC improves grasp success rates by up to 32.6% in simulation and 33.3% in real-world noisy conditions, outperforming open-loop, diffusion policy, transformer policy, and IQL approaches. Videos and more at this http URL. 

---
# Language Bias in Information Retrieval: The Nature of the Beast and Mitigation Methods 

**Authors**: Jinrui Yang, Fan Jiang, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06195)  

**Abstract**: Language fairness in multilingual information retrieval (MLIR) systems is crucial for ensuring equitable access to information across diverse languages. This paper sheds light on the issue, based on the assumption that queries in different languages, but with identical semantics, should yield equivalent ranking lists when retrieving on the same multilingual documents. We evaluate the degree of fairness using both traditional retrieval methods, and a DPR neural ranker based on mBERT and XLM-R. Additionally, we introduce `LaKDA', a novel loss designed to mitigate language biases in neural MLIR approaches. Our analysis exposes intrinsic language biases in current MLIR technologies, with notable disparities across the retrieval methods, and the effectiveness of LaKDA in enhancing language fairness. 

---
# AI Governance in Higher Education: A course design exploring regulatory, ethical and practical considerations 

**Authors**: Zsolt Almási, Hannah Bleher, Johannes Bleher, Rozanne Tuesday Flores, Guo Xuanyang, Paweł Pujszo, Raphaël Weuts  

**Link**: [PDF](https://arxiv.org/pdf/2509.06176)  

**Abstract**: As artificial intelligence (AI) systems permeate critical sectors, the need for professionals who can address ethical, legal and governance challenges has become urgent. Current AI ethics education remains fragmented, often siloed by discipline and disconnected from practice. This paper synthesizes literature and regulatory developments to propose a modular, interdisciplinary curriculum that integrates technical foundations with ethics, law and policy. We highlight recurring operational failures in AI - bias, misspecified objectives, generalization errors, misuse and governance breakdowns - and link them to pedagogical strategies for teaching AI governance. Drawing on perspectives from the EU, China and international frameworks, we outline a semester plan that emphasizes integrated ethics, stakeholder engagement and experiential learning. The curriculum aims to prepare students to diagnose risks, navigate regulation and engage diverse stakeholders, fostering adaptive and ethically grounded professionals for responsible AI governance. 

---
# Reasoning Language Model for Personalized Lung Cancer Screening 

**Authors**: Chuang Niu, Ge Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06169)  

**Abstract**: Accurate risk assessment in lung cancer screening is critical for enabling early cancer detection and minimizing unnecessary invasive procedures. The Lung CT Screening Reporting and Data System (Lung-RADS) has been widely used as the standard framework for patient management and follow-up. Nevertheless, Lung-RADS faces trade-offs between sensitivity and specificity, as it stratifies risk solely based on lung nodule characteristics without incorporating various risk factors. Here we propose a reasoning language model (RLM) to integrate radiology findings with longitudinal medical records for individualized lung cancer risk assessment. Through a systematic study including dataset construction and distillation, supervised fine-tuning, reinforcement learning, and comprehensive evaluation, our model makes significant improvements in risk prediction performance on datasets in the national lung screening trial. Notably, RLM can decompose the risk evaluation task into sub-components, analyze the contributions of diverse risk factors, and synthesize them into a final risk score computed using our data-driven system equation. Our approach improves both predictive accuracy and monitorability through the chain of thought reasoning process, thereby facilitating clinical translation into lung cancer screening. 

---
# UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning 

**Authors**: Huy Le, Nhat Chung, Tung Kieu, Jingkang Yang, Ngan Le  

**Link**: [PDF](https://arxiv.org/pdf/2509.06165)  

**Abstract**: Video Scene Graph Generation (VidSGG) aims to represent dynamic visual content by detecting objects and modeling their temporal interactions as structured graphs. Prior studies typically target either coarse-grained box-level or fine-grained panoptic pixel-level VidSGG, often requiring task-specific architectures and multi-stage training pipelines. In this paper, we present UNO (UNified Object-centric VidSGG), a single-stage, unified framework that jointly addresses both tasks within an end-to-end architecture. UNO is designed to minimize task-specific modifications and maximize parameter sharing, enabling generalization across different levels of visual granularity. The core of UNO is an extended slot attention mechanism that decomposes visual features into object and relation slots. To ensure robust temporal modeling, we introduce object temporal consistency learning, which enforces consistent object representations across frames without relying on explicit tracking modules. Additionally, a dynamic triplet prediction module links relation slots to corresponding object pairs, capturing evolving interactions over time. We evaluate UNO on standard box-level and pixel-level VidSGG benchmarks. Results demonstrate that UNO not only achieves competitive performance across both tasks but also offers improved efficiency through a unified, object-centric design. 

---
# Benchmarking Gender and Political Bias in Large Language Models 

**Authors**: Jinrui Yang, Xudong Han, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06164)  

**Abstract**: We introduce EuroParlVote, a novel benchmark for evaluating large language models (LLMs) in politically sensitive contexts. It links European Parliament debate speeches to roll-call vote outcomes and includes rich demographic metadata for each Member of the European Parliament (MEP), such as gender, age, country, and political group. Using EuroParlVote, we evaluate state-of-the-art LLMs on two tasks -- gender classification and vote prediction -- revealing consistent patterns of bias. We find that LLMs frequently misclassify female MEPs as male and demonstrate reduced accuracy when simulating votes for female speakers. Politically, LLMs tend to favor centrist groups while underperforming on both far-left and far-right ones. Proprietary models like GPT-4o outperform open-weight alternatives in terms of both robustness and fairness. We release the EuroParlVote dataset, code, and demo to support future research on fairness and accountability in NLP within political contexts. 

---
# Tracking daily paths in home contexts with RSSI fingerprinting based on UWB through deep learning models 

**Authors**: Aurora Polo-Rodríguez, Juan Carlos Valera, Jesús Peral, David Gil, Javier Medina-Quero  

**Link**: [PDF](https://arxiv.org/pdf/2509.06161)  

**Abstract**: The field of human activity recognition has evolved significantly, driven largely by advancements in Internet of Things (IoT) device technology, particularly in personal devices. This study investigates the use of ultra-wideband (UWB) technology for tracking inhabitant paths in home environments using deep learning models. UWB technology estimates user locations via time-of-flight and time-difference-of-arrival methods, which are significantly affected by the presence of walls and obstacles in real environments, reducing their precision. To address these challenges, we propose a fingerprinting-based approach utilizing received signal strength indicator (RSSI) data collected from inhabitants in two flats (60 m2 and 100 m2) while performing daily activities. We compare the performance of convolutional neural network (CNN), long short-term memory (LSTM), and hybrid CNN+LSTM models, as well as the use of Bluetooth technology. Additionally, we evaluate the impact of the type and duration of the temporal window (future, past, or a combination of both). Our results demonstrate a mean absolute error close to 50 cm, highlighting the superiority of the hybrid model in providing accurate location estimates, thus facilitating its application in daily human activity recognition in residential settings. 

---
# FASL-Seg: Anatomy and Tool Segmentation of Surgical Scenes 

**Authors**: Muraam Abdel-Ghani, Mahmoud Ali, Mohamed Ali, Fatmaelzahraa Ahmed, Mohamed Arsalan, Abdulaziz Al-Ali, Shidin Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2509.06159)  

**Abstract**: The growing popularity of robotic minimally invasive surgeries has made deep learning-based surgical training a key area of research. A thorough understanding of the surgical scene components is crucial, which semantic segmentation models can help achieve. However, most existing work focuses on surgical tools and overlooks anatomical objects. Additionally, current state-of-the-art (SOTA) models struggle to balance capturing high-level contextual features and low-level edge features. We propose a Feature-Adaptive Spatial Localization model (FASL-Seg), designed to capture features at multiple levels of detail through two distinct processing streams, namely a Low-Level Feature Projection (LLFP) and a High-Level Feature Projection (HLFP) stream, for varying feature resolutions - enabling precise segmentation of anatomy and surgical instruments. We evaluated FASL-Seg on surgical segmentation benchmark datasets EndoVis18 and EndoVis17 on three use cases. The FASL-Seg model achieves a mean Intersection over Union (mIoU) of 72.71% on parts and anatomy segmentation in EndoVis18, improving on SOTA by 5%. It further achieves a mIoU of 85.61% and 72.78% in EndoVis18 and EndoVis17 tool type segmentation, respectively, outperforming SOTA overall performance, with comparable per-class SOTA results in both datasets and consistent performance in various classes for anatomy and instruments, demonstrating the effectiveness of distinct processing streams for varying feature resolutions. 

---
# SpecSwin3D: Generating Hyperspectral Imagery from Multispectral Data via Transformer Networks 

**Authors**: Tang Sui, Songxi Yang, Qunying Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06122)  

**Abstract**: Multispectral and hyperspectral imagery are widely used in agriculture, environmental monitoring, and urban planning due to their complementary spatial and spectral characteristics. A fundamental trade-off persists: multispectral imagery offers high spatial but limited spectral resolution, while hyperspectral imagery provides rich spectra at lower spatial resolution. Prior hyperspectral generation approaches (e.g., pan-sharpening variants, matrix factorization, CNNs) often struggle to jointly preserve spatial detail and spectral fidelity. In response, we propose SpecSwin3D, a transformer-based model that generates hyperspectral imagery from multispectral inputs while preserving both spatial and spectral quality. Specifically, SpecSwin3D takes five multispectral bands as input and reconstructs 224 hyperspectral bands at the same spatial resolution. In addition, we observe that reconstruction errors grow for hyperspectral bands spectrally distant from the input bands. To address this, we introduce a cascade training strategy that progressively expands the spectral range to stabilize learning and improve fidelity. Moreover, we design an optimized band sequence that strategically repeats and orders the five selected multispectral bands to better capture pairwise relations within a 3D shifted-window transformer framework. Quantitatively, our model achieves a PSNR of 35.82 dB, SAM of 2.40°, and SSIM of 0.96, outperforming the baseline MHF-Net by +5.6 dB in PSNR and reducing ERGAS by more than half. Beyond reconstruction, we further demonstrate the practical value of SpecSwin3D on two downstream tasks, including land use classification and burnt area segmentation. 

---
# Teaching Precommitted Agents: Model-Free Policy Evaluation and Control in Quasi-Hyperbolic Discounted MDPs 

**Authors**: S.R. Eshwar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06094)  

**Abstract**: Time-inconsistent preferences, where agents favor smaller-sooner over larger-later rewards, are a key feature of human and animal decision-making. Quasi-Hyperbolic (QH) discounting provides a simple yet powerful model for this behavior, but its integration into the reinforcement learning (RL) framework has been limited. This paper addresses key theoretical and algorithmic gaps for precommitted agents with QH preferences. We make two primary contributions: (i) we formally characterize the structure of the optimal policy, proving for the first time that it reduces to a simple one-step non-stationary form; and (ii) we design the first practical, model-free algorithms for both policy evaluation and Q-learning in this setting, both with provable convergence guarantees. Our results provide foundational insights for incorporating QH preferences in RL. 

---
# Language Native Lightly Structured Databases for Large Language Model Driven Composite Materials Research 

**Authors**: Yuze Liu, Zhaoyuan Zhang, Xiangsheng Zeng, Yihe Zhang, Leping Yu, Lejia Wang, Xi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06093)  

**Abstract**: Chemical and materials research has traditionally relied heavily on knowledge narrative, with progress often driven by language-based descriptions of principles, mechanisms, and experimental experiences, rather than tables, limiting what conventional databases and ML can exploit. We present a language-native database for boron nitride nanosheet (BNNS) polymer thermally conductive composites that captures lightly structured information from papers across preparation, characterization, theory-computation, and mechanistic reasoning, with evidence-linked snippets. Records are organized in a heterogeneous database and queried via composite retrieval with semantics, key words and value filters. The system can synthesizes literature into accurate, verifiable, and expert style guidance. This substrate enables high fidelity efficient Retrieval Augmented Generation (RAG) and tool augmented agents to interleave retrieval with reasoning and deliver actionable SOP. The framework supplies the language rich foundation required for LLM-driven materials discovery. 

---
# Software Dependencies 2.0: An Empirical Study of Reuse and Integration of Pre-Trained Models in Open-Source Projects 

**Authors**: Jerin Yasmin, Wenxin Jiang, James C. Davis, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.06085)  

**Abstract**: Pre-trained models (PTMs) are machine learning models that have been trained in advance, often on large-scale data, and can be reused for new tasks, thereby reducing the need for costly training from scratch. Their widespread adoption introduces a new class of software dependency, which we term Software Dependencies 2.0, extending beyond conventional libraries to learned behaviors embodied in trained models and their associated artifacts. The integration of PTMs as software dependencies in real projects remains unclear, potentially threatening maintainability and reliability of modern software systems that increasingly rely on them. Objective: In this study, we investigate Software Dependencies 2.0 in open-source software (OSS) projects by examining the reuse of PTMs, with a focus on how developers manage and integrate these models. Specifically, we seek to understand: (1) how OSS projects structure and document their PTM dependencies; (2) what stages and organizational patterns emerge in the reuse pipelines of PTMs within these projects; and (3) the interactions among PTMs and other learned components across pipeline stages. We conduct a mixed-methods analysis of a statistically significant random sample of 401 GitHub repositories from the PeaTMOSS dataset (28,575 repositories reusing PTMs from Hugging Face and PyTorch Hub). We quantitatively examine PTM reuse by identifying patterns and qualitatively investigate how developers integrate and manage these models in practice. 

---
# ARIES: Relation Assessment and Model Recommendation for Deep Time Series Forecasting 

**Authors**: Fei Wang, Yujie Li, Zezhi Shao, Chengqing Yu, Yisong Fu, Zhulin An, Yongjun Xu, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06060)  

**Abstract**: Recent advancements in deep learning models for time series forecasting have been significant. These models often leverage fundamental time series properties such as seasonality and non-stationarity, which may suggest an intrinsic link between model performance and data properties. However, existing benchmark datasets fail to offer diverse and well-defined temporal patterns, restricting the systematic evaluation of such connections. Additionally, there is no effective model recommendation approach, leading to high time and cost expenditures when testing different architectures across different downstream applications. For those reasons, we propose ARIES, a framework for assessing relation between time series properties and modeling strategies, and for recommending deep forcasting models for realistic time series. First, we construct a synthetic dataset with multiple distinct patterns, and design a comprehensive system to compute the properties of time series. Next, we conduct an extensive benchmarking of over 50 forecasting models, and establish the relationship between time series properties and modeling strategies. Our experimental results reveal a clear correlation. Based on these findings, we propose the first deep forecasting model recommender, capable of providing interpretable suggestions for real-world time series. In summary, ARIES is the first study to establish the relations between the properties of time series data and modeling strategies, while also implementing a model recommendation system. The code is available at: this https URL. 

---
# PolicyEvolve: Evolving Programmatic Policies by LLMs for multi-player games via Population-Based Training 

**Authors**: Mingrui Lv, Hangzhi Liu, Zhi Luo, Hongjie Zhang, Jie Ou  

**Link**: [PDF](https://arxiv.org/pdf/2509.06053)  

**Abstract**: Multi-agent reinforcement learning (MARL) has achieved significant progress in solving complex multi-player games through self-play. However, training effective adversarial policies requires millions of experience samples and substantial computational resources. Moreover, these policies lack interpretability, hindering their practical deployment. Recently, researchers have successfully leveraged Large Language Models (LLMs) to generate programmatic policies for single-agent tasks, transforming neural network-based policies into interpretable rule-based code with high execution efficiency. Inspired by this, we propose PolicyEvolve, a general framework for generating programmatic policies in multi-player games. PolicyEvolve significantly reduces reliance on manually crafted policy code, achieving high-performance policies with minimal environmental interactions. The framework comprises four modules: Global Pool, Local Pool, Policy Planner, and Trajectory Critic. The Global Pool preserves elite policies accumulated during iterative training. The Local Pool stores temporary policies for the current iteration; only sufficiently high-performing policies from this pool are promoted to the Global Pool. The Policy Planner serves as the core policy generation module. It samples the top three policies from the Global Pool, generates an initial policy for the current iteration based on environmental information, and refines this policy using feedback from the Trajectory Critic. Refined policies are then deposited into the Local Pool. This iterative process continues until the policy achieves a sufficiently high average win rate against the Global Pool, at which point it is integrated into the Global Pool. The Trajectory Critic analyzes interaction data from the current policy, identifies vulnerabilities, and proposes directional improvements to guide the Policy Planner 

---
# Empirical Study of Code Large Language Models for Binary Security Patch Detection 

**Authors**: Qingyuan Li, Binchang Li, Cuiyun Gao, Shuzheng Gao, Zongjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06052)  

**Abstract**: Security patch detection (SPD) is crucial for maintaining software security, as unpatched vulnerabilities can lead to severe security risks. In recent years, numerous learning-based SPD approaches have demonstrated promising results on source code. However, these approaches typically cannot be applied to closed-source applications and proprietary systems that constitute a significant portion of real-world software, as they release patches only with binary files, and the source code is inaccessible. Given the impressive performance of code large language models (LLMs) in code intelligence and binary analysis tasks such as decompilation and compilation optimization, their potential for detecting binary security patches remains unexplored, exposing a significant research gap between their demonstrated low-level code understanding capabilities and this critical security task. To address this gap, we construct a large-scale binary patch dataset containing \textbf{19,448} samples, with two levels of representation: assembly code and pseudo-code, and systematically evaluate \textbf{19} code LLMs of varying scales to investigate their capability in binary SPD tasks. Our initial exploration demonstrates that directly prompting vanilla code LLMs struggles to accurately identify security patches from binary patches, and even state-of-the-art prompting techniques fail to mitigate the lack of domain knowledge in binary SPD within vanilla models. Drawing on the initial findings, we further investigate the fine-tuning strategy for injecting binary SPD domain knowledge into code LLMs through two levels of representation. Experimental results demonstrate that fine-tuned LLMs achieve outstanding performance, with the best results obtained on the pseudo-code representation. 

---
# BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models 

**Authors**: Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06040)  

**Abstract**: Recent advancements in aligning image and video generative models via GRPO have achieved remarkable gains in enhancing human preference alignment. However, these methods still face high computational costs from on-policy rollouts and excessive SDE sampling steps, as well as training instability due to sparse rewards. In this paper, we propose BranchGRPO, a novel method that introduces a branch sampling policy updating the SDE sampling process. By sharing computation across common prefixes and pruning low-reward paths and redundant depths, BranchGRPO substantially lowers the per-update compute cost while maintaining or improving exploration diversity. This work makes three main contributions: (1) a branch sampling scheme that reduces rollout and training cost; (2) a tree-based advantage estimator incorporating dense process-level rewards; and (3) pruning strategies exploiting path and depth redundancy to accelerate convergence and boost performance. Experiments on image and video preference alignment show that BranchGRPO improves alignment scores by 16% over strong baselines, while cutting training time by 50%. 

---
# TinyDef-DETR:An Enhanced DETR Detector for UAV Power Line Defect Detection 

**Authors**: Jiaming Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.06035)  

**Abstract**: Automated inspection of transmission lines using UAVs is hindered by the difficulty of detecting small and ambiguous defects against complex backgrounds. Conventional detectors often suffer from detail loss due to strided downsampling, weak boundary sensitivity in lightweight backbones, and insufficient integration of global context with local cues. To address these challenges, we propose TinyDef-DETR, a DETR-based framework designed for small-defect detection. The method introduces a stride-free space-to-depth module for lossless downsampling, an edge-enhanced convolution for boundary-aware feature extraction, a cross-stage dual-domain multi-scale attention module to jointly capture global and local information, and a Focaler-Wise-SIoU regression loss to improve localization of small objects. Experiments conducted on the CSG-ADCD dataset demonstrate that TinyDef-DETR achieves substantial improvements in both precision and recall compared to competitive baselines, with particularly notable gains on small-object subsets, while incurring only modest computational overhead. Further validation on the VisDrone benchmark confirms the generalization capability of the proposed approach. Overall, the results indicate that integrating detail-preserving downsampling, edge-sensitive representations, dual-domain attention, and difficulty-adaptive regression provides a practical and efficient solution for UAV-based small-defect inspection in power grids. 

---
# DreamAudio: Customized Text-to-Audio Generation with Diffusion Models 

**Authors**: Yi Yuan, Xubo Liu, Haohe Liu, Xiyuan Kang, Zhuo Chen, Yuxuan Wang, Mark D. Plumbley, Wenwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06027)  

**Abstract**: With the development of large-scale diffusion-based and language-modeling-based generative models, impressive progress has been achieved in text-to-audio generation. Despite producing high-quality outputs, existing text-to-audio models mainly aim to generate semantically aligned sound and fall short on precisely controlling fine-grained acoustic characteristics of specific sounds. As a result, users that need specific sound content may find it challenging to generate the desired audio clips. In this paper, we present DreamAudio for customized text-to-audio generation (CTTA). Specifically, we introduce a new framework that is designed to enable the model to identify auditory information from user-provided reference concepts for audio generation. Given a few reference audio samples containing personalized audio events, our system can generate new audio samples that include these specific events. In addition, two types of datasets are developed for training and testing the customized systems. The experiments show that the proposed model, DreamAudio, generates audio samples that are highly consistent with the customized audio features and aligned well with the input text prompts. Furthermore, DreamAudio offers comparable performance in general text-to-audio tasks. We also provide a human-involved dataset containing audio events from real-world CTTA cases as the benchmark for customized generation tasks. 

---
# DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation 

**Authors**: Xinyu Gao, Xiangtao Meng, Yingkai Dong, Zheng Li, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06026)  

**Abstract**: While Retrieval-Augmented Generation (RAG) effectively reduces hallucinations by integrating external knowledge bases, it introduces vulnerabilities to membership inference attacks (MIAs), particularly in systems handling sensitive data. Existing MIAs targeting RAG's external databases often rely on model responses but ignore the interference of non-member-retrieved documents on RAG outputs, limiting their effectiveness. To address this, we propose DCMI, a differential calibration MIA that mitigates the negative impact of non-member-retrieved documents. Specifically, DCMI leverages the sensitivity gap between member and non-member retrieved documents under query perturbation. It generates perturbed queries for calibration to isolate the contribution of member-retrieved documents while minimizing the interference from non-member-retrieved documents. Experiments under progressively relaxed assumptions show that DCMI consistently outperforms baselines--for example, achieving 97.42% AUC and 94.35% Accuracy against the RAG system with Flan-T5, exceeding the MBA baseline by over 40%. Furthermore, on real-world RAG platforms such as Dify and MaxKB, DCMI maintains a 10%-20% advantage over the baseline. These results highlight significant privacy risks in RAG systems and emphasize the need for stronger protection mechanisms. We appeal to the community's consideration of deeper investigations, like ours, against the data leakage risks in rapidly evolving RAG systems. Our code is available at this https URL. 

---
# Unified Interaction Foundational Model (UIFM) for Predicting Complex User and System Behavior 

**Authors**: Vignesh Ethiraj, Subhash Talluri  

**Link**: [PDF](https://arxiv.org/pdf/2509.06025)  

**Abstract**: A central goal of artificial intelligence is to build systems that can understand and predict complex, evolving sequences of events. However, current foundation models, designed for natural language, fail to grasp the holistic nature of structured interactions found in domains like telecommunications, e-commerce and finance. By serializing events into text, they disassemble them into semantically fragmented parts, losing critical context. In this work, we introduce the Unified Interaction Foundation Model (UIFM), a foundation model engineered for genuine behavioral understanding. At its core is the principle of composite tokenization, where each multi-attribute event is treated as a single, semantically coherent unit. This allows UIFM to learn the underlying "grammar" of user behavior, perceiving entire interactions rather than a disconnected stream of data points. We demonstrate that this architecture is not just more accurate, but represents a fundamental step towards creating more adaptable and intelligent predictive systems. 

---
# Khana: A Comprehensive Indian Cuisine Dataset 

**Authors**: Omkar Prabhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06006)  

**Abstract**: As global interest in diverse culinary experiences grows, food image models are essential for improving food-related applications by enabling accurate food recognition, recipe suggestions, dietary tracking, and automated meal planning. Despite the abundance of food datasets, a noticeable gap remains in capturing the nuances of Indian cuisine due to its vast regional diversity, complex preparations, and the lack of comprehensive labeled datasets that cover its full breadth. Through this exploration, we uncover Khana, a new benchmark dataset for food image classification, segmentation, and retrieval of dishes from Indian cuisine. Khana fills the gap by establishing a taxonomy of Indian cuisine and offering around 131K images in the dataset spread across 80 labels, each with a resolution of 500x500 pixels. This paper describes the dataset creation process and evaluates state-of-the-art models on classification, segmentation, and retrieval as baselines. Khana bridges the gap between research and development by providing a comprehensive and challenging benchmark for researchers while also serving as a valuable resource for developers creating real-world applications that leverage the rich tapestry of Indian cuisine. Webpage: this https URL 

---
# S-LAM3D: Segmentation-Guided Monocular 3D Object Detection via Feature Space Fusion 

**Authors**: Diana-Alexandra Sas, Florin Oniga  

**Link**: [PDF](https://arxiv.org/pdf/2509.05999)  

**Abstract**: Monocular 3D Object Detection represents a challenging Computer Vision task due to the nature of the input used, which is a single 2D image, lacking in any depth cues and placing the depth estimation problem as an ill-posed one. Existing solutions leverage the information extracted from the input by using Convolutional Neural Networks or Transformer architectures as feature extraction backbones, followed by specific detection heads for 3D parameters prediction. In this paper, we introduce a decoupled strategy based on injecting precomputed segmentation information priors and fusing them directly into the feature space for guiding the detection, without expanding the detection model or jointly learning the priors. The focus is on evaluating the impact of additional segmentation information on existing detection pipelines without adding additional prediction branches. The proposed method is evaluated on the KITTI 3D Object Detection Benchmark, outperforming the equivalent architecture that relies only on RGB image features for small objects in the scene: pedestrians and cyclists, and proving that understanding the input data can balance the need for additional sensors or training data. 

---
# Operationalising AI Regulatory Sandboxes under the EU AI Act: The Triple Challenge of Capacity, Coordination and Attractiveness to Providers 

**Authors**: Deirdre Ahern  

**Link**: [PDF](https://arxiv.org/pdf/2509.05985)  

**Abstract**: The EU AI Act provides a rulebook for all AI systems being put on the market or into service in the European Union. This article investigates the requirement under the AI Act that Member States establish national AI regulatory sandboxes for testing and validation of innovative AI systems under regulatory supervision to assist with fostering innovation and complying with regulatory requirements. Against the backdrop of the EU objective that AI regulatory sandboxes would both foster innovation and assist with compliance, considerable challenges are identified for Member States around capacity-building and design of regulatory sandboxes. While Member States are early movers in laying the ground for national AI regulatory sandboxes, the article contends that there is a risk that differing approaches being taken by individual national sandboxes could jeopardise a uniform interpretation of the AI Act and its application in practice. This could motivate innovators to play sandbox arbitrage. The article therefore argues that the European Commission and the AI Board need to act decisively in developing rules and guidance to ensure a cohesive, coordinated approach in national AI regulatory sandboxes. With sandbox participation being voluntary, the possibility that AI regulatory sandboxes may prove unattractive to innovators on their compliance journey is also explored. Confidentiality concerns, the inability to relax legal rules during the sandbox, and the inability of sandboxes to deliver a presumption of conformity with the AI Act are identified as pertinent concerns for innovators contemplating applying to AI regulatory sandboxes as compared with other direct compliance routes provided to them through application of harmonised standards and conformity assessment procedures. 

---
# TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition 

**Authors**: Minh N. H. Nguyen, Anh Nguyen Tran, Dung Truong Dinh, Nam Van Vo  

**Link**: [PDF](https://arxiv.org/pdf/2509.05983)  

**Abstract**: Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the subtle phonological shifts inherent in CS scenarios. The challenge is particularly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). The TSPC employs a phoneme-centric approach, built upon an extended Vietnamese phoneme set as an intermediate representation to facilitate mixed-lingual modeling. Experimental results demonstrate that TSPC consistently outperforms existing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 20.8\% with reduced training resources. Furthermore, the phonetic-based two-stage architecture enables phoneme adaptation and language conversion to enhance ASR performance in complex CS Vietnamese-English ASR scenarios. 

---
# ConstStyle: Robust Domain Generalization with Unified Style Transformation 

**Authors**: Nam Duong Tran, Nam Nguyen Phuong, Hieu H. Pham, Phi Le Nguyen, My T. Thai  

**Link**: [PDF](https://arxiv.org/pdf/2509.05975)  

**Abstract**: Deep neural networks often suffer performance drops when test data distribution differs from training data. Domain Generalization (DG) aims to address this by focusing on domain-invariant features or augmenting data for greater diversity. However, these methods often struggle with limited training domains or significant gaps between seen (training) and unseen (test) domains. To enhance DG robustness, we hypothesize that it is essential for the model to be trained on data from domains that closely resemble unseen test domains-an inherently difficult task due to the absence of prior knowledge about the unseen domains. Accordingly, we propose ConstStyle, a novel approach that leverages a unified domain to capture domain-invariant features and bridge the domain gap with theoretical analysis. During training, all samples are mapped onto this unified domain, optimized for seen domains. During testing, unseen domain samples are projected similarly before predictions. By aligning both training and testing data within this unified domain, ConstStyle effectively reduces the impact of domain shifts, even with large domain gaps or few seen domains. Extensive experiments demonstrate that ConstStyle consistently outperforms existing methods across diverse scenarios. Notably, when only a limited number of seen domains are available, ConstStyle can boost accuracy up to 19.82\% compared to the next best approach. 

---
# Meta-training of diffractive meta-neural networks for super-resolution direction of arrival estimation 

**Authors**: Songtao Yang, Sheng Gao, Chu Wu, Zejia Zhao, Haiou Zhang, Xing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05926)  

**Abstract**: Diffractive neural networks leverage the high-dimensional characteristics of electromagnetic (EM) fields for high-throughput computing. However, the existing architectures face challenges in integrating large-scale multidimensional metasurfaces with precise network training and haven't utilized multidimensional EM field coding scheme for super-resolution sensing. Here, we propose diffractive meta-neural networks (DMNNs) for accurate EM field modulation through metasurfaces, which enable multidimensional multiplexing and coding for multi-task learning and high-throughput super-resolution direction of arrival estimation. DMNN integrates pre-trained mini-metanets to characterize the amplitude and phase responses of meta-atoms across different polarizations and frequencies, with structure parameters inversely designed using the gradient-based meta-training. For wide-field super-resolution angle estimation, the system simultaneously resolves azimuthal and elevational angles through x and y-polarization channels, while the interleaving of frequency-multiplexed angular intervals generates spectral-encoded optical super-oscillations to achieve full-angle high-resolution estimation. Post-processing lightweight electronic neural networks further enhance the performance. Experimental results validate that a three-layer DMNN operating at 27 GHz, 29 GHz, and 31 GHz achieves $\sim7\times$ Rayleigh diffraction-limited angular resolution (0.5$^\circ$), a mean absolute error of 0.048$^\circ$ for two incoherent targets within a $\pm 11.5^\circ$ field of view, and an angular estimation throughput an order of magnitude higher (1917) than that of existing methods. The proposed architecture advances high-dimensional photonic computing systems by utilizing inherent high-parallelism and all-optical coding methods for ultra-high-resolution, high-throughput applications. 

---
# Challenges in Deep Learning-Based Small Organ Segmentation: A Benchmarking Perspective for Medical Research with Limited Datasets 

**Authors**: Phongsakon Mark Konrad, Andrei-Alexandru Popa, Yaser Sabzehmeidani, Liang Zhong, Elisa A. Liehn, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.05892)  

**Abstract**: Accurate segmentation of carotid artery structures in histopathological images is vital for advancing cardiovascular disease research and diagnosis. However, deep learning model development in this domain is constrained by the scarcity of annotated cardiovascular histopathological data. This study investigates a systematic evaluation of state-of-the-art deep learning segmentation models, including convolutional neural networks (U-Net, DeepLabV3+), a Vision Transformer (SegFormer), and recent foundation models (SAM, MedSAM, MedSAM+UNet), on a limited dataset of cardiovascular histology images. Despite employing an extensive hyperparameter optimization strategy with Bayesian search, our findings reveal that model performance is highly sensitive to data splits, with minor differences driven more by statistical noise than by true algorithmic superiority. This instability exposes the limitations of standard benchmarking practices in low-data clinical settings and challenges the assumption that performance rankings reflect meaningful clinical utility. 

---
# Quantum spatial best-arm identification via quantum walks 

**Authors**: Tomoki Yamagami, Etsuo Segawa, Takatomo Mihana, André Röhm, Atsushi Uchida, Ryoichi Horisaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.05890)  

**Abstract**: Quantum reinforcement learning has emerged as a framework combining quantum computation with sequential decision-making, and applications to the multi-armed bandit (MAB) problem have been reported. The graph bandit problem extends the MAB setting by introducing spatial constraints, yet quantum approaches remain limited. We propose a quantum algorithm for best-arm identification in graph bandits, termed Quantum Spatial Best-Arm Identification (QSBAI). The method employs quantum walks to encode superpositions over graph-constrained actions, extending amplitude amplification and generalizing the Quantum BAI algorithm via Szegedy's walk framework. This establishes a link between Grover-type search and reinforcement learning tasks with structural restrictions. We analyze complete and bipartite graphs, deriving the maximal success probability of identifying the best arm and the time step at which it is achieved. Our results highlight the potential of quantum walks to accelerate exploration in constrained environments and extend the applicability of quantum algorithms for decision-making. 

---
# Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs 

**Authors**: Andrew Yeo, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.05883)  

**Abstract**: Large Language Models (LLMs) have seen rapid adoption in recent years, with industries increasingly relying on them to maintain a competitive advantage. These models excel at interpreting user instructions and generating human-like responses, leading to their integration across diverse domains, including consulting and information retrieval. However, their widespread deployment also introduces substantial security risks, most notably in the form of prompt injection and jailbreak attacks.
To systematically evaluate LLM vulnerabilities -- particularly to external prompt injection -- we conducted a series of experiments on eight commercial models. Each model was tested without supplementary sanitization, relying solely on its built-in safeguards. The results exposed exploitable weaknesses and emphasized the need for stronger security measures. Four categories of attacks were examined: direct injection, indirect (external) injection, image-based injection, and prompt leakage. Comparative analysis indicated that Claude 3 demonstrated relatively greater robustness; nevertheless, empirical findings confirm that additional defenses, such as input normalization, remain necessary to achieve reliable protection. 

---
# Let's Roleplay: Examining LLM Alignment in Collaborative Dialogues 

**Authors**: Abhijnan Nath, Carine Graff, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2509.05882)  

**Abstract**: As Large Language Models (LLMs) integrate into diverse workflows, they are increasingly being considered "collaborators" with humans. If such AI collaborators are to be reliable, their behavior over multiturn interactions must be predictable, validated and verified before deployment. Common alignment techniques are typically developed under simplified single-user settings and do not account for the dynamics of long-horizon multiparty interactions. This paper examines how different alignment methods affect LLM agents' effectiveness as partners in multiturn, multiparty collaborations. We study this question through the lens of friction agents that intervene in group dialogues to encourage the collaborative group to slow down and reflect upon their reasoning for deliberative decision-making. Using a roleplay methodology, we evaluate interventions from differently-trained friction agents in collaborative task conversations. We propose a novel counterfactual evaluation framework that quantifies how friction interventions change the trajectory of group collaboration and belief alignment. Our results show that a friction-aware approach significantly outperforms common alignment baselines in helping both convergence to a common ground, or agreed-upon task-relevant propositions, and correctness of task outcomes. 

---
# GeoAnalystBench: A GeoAI benchmark for assessing large language models for spatial analysis workflow and code generation 

**Authors**: Qianheng Zhang, Song Gao, Chen Wei, Yibo Zhao, Ying Nie, Ziru Chen, Shijie Chen, Yu Su, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.05881)  

**Abstract**: Recent advances in large language models (LLMs) have fueled growing interest in automating geospatial analysis and GIS workflows, yet their actual capabilities remain uncertain. In this work, we call for rigorous evaluation of LLMs on well-defined geoprocessing tasks before making claims about full GIS automation. To this end, we present GeoAnalystBench, a benchmark of 50 Python-based tasks derived from real-world geospatial problems and carefully validated by GIS experts. Each task is paired with a minimum deliverable product, and evaluation covers workflow validity, structural alignment, semantic similarity, and code quality (CodeBLEU). Using this benchmark, we assess both proprietary and open source models. Results reveal a clear gap: proprietary models such as ChatGPT-4o-mini achieve high validity 95% and stronger code alignment (CodeBLEU 0.39), while smaller open source models like DeepSeek-R1-7B often generate incomplete or inconsistent workflows (48.5% validity, 0.272 CodeBLEU). Tasks requiring deeper spatial reasoning, such as spatial relationship detection or optimal site selection, remain the most challenging across all models. These findings demonstrate both the promise and limitations of current LLMs in GIS automation and provide a reproducible framework to advance GeoAI research with human-in-the-loop support. 

---
# Uncertainty Quantification in Probabilistic Machine Learning Models: Theory, Methods, and Insights 

**Authors**: Marzieh Ajirak, Anand Ravishankar, Petar M. Djuric  

**Link**: [PDF](https://arxiv.org/pdf/2509.05877)  

**Abstract**: Uncertainty Quantification (UQ) is essential in probabilistic machine learning models, particularly for assessing the reliability of predictions. In this paper, we present a systematic framework for estimating both epistemic and aleatoric uncertainty in probabilistic models. We focus on Gaussian Process Latent Variable Models and employ scalable Random Fourier Features-based Gaussian Processes to approximate predictive distributions efficiently. We derive a theoretical formulation for UQ, propose a Monte Carlo sampling-based estimation method, and conduct experiments to evaluate the impact of uncertainty estimation. Our results provide insights into the sources of predictive uncertainty and illustrate the effectiveness of our approach in quantifying the confidence in the predictions. 

---
# Learning to Construct Knowledge through Sparse Reference Selection with Reinforcement Learning 

**Authors**: Shao-An Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05874)  

**Abstract**: The rapid expansion of scientific literature makes it increasingly difficult to acquire new knowledge, particularly in specialized domains where reasoning is complex, full-text access is restricted, and target references are sparse among a large set of candidates. We present a Deep Reinforcement Learning framework for sparse reference selection that emulates human knowledge construction, prioritizing which papers to read under limited time and cost. Evaluated on drug--gene relation discovery with access restricted to titles and abstracts, our approach demonstrates that both humans and machines can construct knowledge effectively from partial information. 

---
# ZhiFangDanTai: Fine-tuning Graph-based Retrieval-Augmented Generation Model for Traditional Chinese Medicine Formula 

**Authors**: ZiXuan Zhang, Bowen Hao, Yingjie Li, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05867)  

**Abstract**: Traditional Chinese Medicine (TCM) formulas play a significant role in treating epidemics and complex diseases. Existing models for TCM utilize traditional algorithms or deep learning techniques to analyze formula relationships, yet lack comprehensive results, such as complete formula compositions and detailed explanations. Although recent efforts have used TCM instruction datasets to fine-tune Large Language Models (LLMs) for explainable formula generation, existing datasets lack sufficient details, such as the roles of the formula's sovereign, minister, assistant, courier; efficacy; contraindications; tongue and pulse diagnosis-limiting the depth of model outputs. To address these challenges, we propose ZhiFangDanTai, a framework combining Graph-based Retrieval-Augmented Generation (GraphRAG) with LLM fine-tuning. ZhiFangDanTai uses GraphRAG to retrieve and synthesize structured TCM knowledge into concise summaries, while also constructing an enhanced instruction dataset to improve LLMs' ability to integrate retrieved information. Furthermore, we provide novel theoretical proofs demonstrating that integrating GraphRAG with fine-tuning techniques can reduce generalization error and hallucination rates in the TCM formula task. Experimental results on both collected and clinical datasets demonstrate that ZhiFangDanTai achieves significant improvements over state-of-the-art models. Our model is open-sourced at this https URL. 

---
# GenAI on Wall Street -- Opportunities and Risk Controls 

**Authors**: Jackie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05841)  

**Abstract**: We give an overview on the emerging applications of GenAI in the financial industry, especially within investment banks. Inherent to these exciting opportunities is a new realm of risks that must be managed properly. By heeding both the Yin and Yang sides of GenAI, we can accelerate its organic growth while safeguarding the entire financial industry during this nascent era of AI. 

---
# Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization 

**Authors**: Ishaan Verma  

**Link**: [PDF](https://arxiv.org/pdf/2509.05831)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content. 

---
# time2time: Causal Intervention in Hidden States to Simulate Rare Events in Time Series Foundation Models 

**Authors**: Debdeep Sanyal, Aaryan Nagpal, Dhruv Kumar, Murari Mandal, Saurabh Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2509.05801)  

**Abstract**: While transformer-based foundation models excel at forecasting routine patterns, two questions remain: do they internalize semantic concepts such as market regimes, or merely fit curves? And can their internal representations be leveraged to simulate rare, high-stakes events such as market crashes? To investigate this, we introduce activation transplantation, a causal intervention that manipulates hidden states by imposing the statistical moments of one event (e.g., a historical crash) onto another (e.g., a calm period) during the forward pass. This procedure deterministically steers forecasts: injecting crash semantics induces downturn predictions, while injecting calm semantics suppresses crashes and restores stability. Beyond binary control, we find that models encode a graded notion of event severity, with the latent vector norm directly correlating with the magnitude of systemic shocks. Validated across two architecturally distinct TSFMs, Toto (decoder only) and Chronos (encoder-decoder), our results demonstrate that steerable, semantically grounded representations are a robust property of large time series transformers. Our findings provide evidence for a latent concept space that governs model predictions, shifting interpretability from post-hoc attribution to direct causal intervention, and enabling semantic "what-if" analysis for strategic stress-testing. 

---
# Hybrid Fourier Neural Operator-Plasma Fluid Model for Fast and Accurate Multiscale Simulations of High Power Microwave Breakdown 

**Authors**: Kalp Pandya, Pratik Ghosh, Ajeya Mandikal, Shivam Gandha, Bhaskar Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.05799)  

**Abstract**: Modeling and simulation of High Power Microwave (HPM) breakdown, a multiscale phenomenon, is computationally expensive and requires solving Maxwell's equations (EM solver) coupled with a plasma continuity equation (plasma solver). In this work, we present a hybrid modeling approach that combines the accuracy of a differential equation-based plasma fluid solver with the computational efficiency of FNO (Fourier Neural Operator) based EM solver. Trained on data from an in-house FDTD-based plasma-fluid solver, the FNO replaces computationally expensive EM field updates, while the plasma solver governs the dynamic plasma response. The hybrid model is validated on microwave streamer formation, due to diffusion ionization mechanism, in a 2D scenario for unseen incident electric fields corresponding to entirely new plasma streamer simulations not included in model training, showing excellent agreement with FDTD based fluid simulations in terms of streamer shape, velocity, and temporal evolution. This hybrid FNO based strategy delivers significant acceleration of the order of 60X compared to traditional simulations for the specified problem size and offers an efficient alternative for computationally demanding multiscale and multiphysics simulations involved in HPM breakdown. Our work also demonstrate how such hybrid pipelines can be used to seamlessly to integrate existing C-based simulation codes with Python-based machine learning frameworks for simulations of plasma science and engineering problems. 

---
# Dual-Mode Deep Anomaly Detection for Medical Manufacturing: Structural Similarity and Feature Distance 

**Authors**: Julio Zanon Diaz, Georgios Siogkas, Peter Corcoran  

**Link**: [PDF](https://arxiv.org/pdf/2509.05796)  

**Abstract**: Automating visual inspection in medical device manufacturing remains challenging due to small and imbalanced datasets, high-resolution imagery, and stringent regulatory requirements. This work proposes two attention-guided autoencoder architectures for deep anomaly detection designed to address these constraints. The first employs a structural similarity-based anomaly score (4-MS-SSIM), offering lightweight and accurate real-time defect detection, yielding ACC 0.903 (unsupervised thresholding) and 0.931 (supervised thresholding) on the - Surface Seal Image - Test split with only 10% of defective samples. The second applies a feature-distance approach using Mahalanobis scoring on reduced latent features, providing high sensitivity to distributional shifts for supervisory monitoring, achieving ACC 0.722 with supervised thresholding. Together, these methods deliver complementary capabilities: the first supports reliable inline inspection, while the second enables scalable post-production surveillance and regulatory compliance monitoring. Experimental results demonstrate that both approaches surpass re-implemented baselines and provide a practical pathway for deploying deep anomaly detection in regulated manufacturing environments, aligning accuracy, efficiency, and the regulatory obligations defined for high-risk AI systems under the EU AI Act. 

---
# DCV-ROOD Evaluation Framework: Dual Cross-Validation for Robust Out-of-Distribution Detection 

**Authors**: Arantxa Urrea-Castaño, Nicolás Segura-Kunsagi, Juan Luis Suárez-Díaz, Rosana Montes, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2509.05778)  

**Abstract**: Out-of-distribution (OOD) detection plays a key role in enhancing the robustness of artificial intelligence systems by identifying inputs that differ significantly from the training distribution, thereby preventing unreliable predictions and enabling appropriate fallback mechanisms. Developing reliable OOD detection methods is a significant challenge, and rigorous evaluation of these techniques is essential for ensuring their effectiveness, as it allows researchers to assess their performance under diverse conditions and to identify potential limitations or failure modes. Cross-validation (CV) has proven to be a highly effective tool for providing a reasonable estimate of the performance of a learning algorithm. Although OOD scenarios exhibit particular characteristics, an appropriate adaptation of CV can lead to a suitable evaluation framework for this setting. This work proposes a dual CV framework for robust evaluation of OOD detection models, aimed at improving the reliability of their assessment. The proposed evaluation framework aims to effectively integrate in-distribution (ID) and OOD data while accounting for their differing characteristics. To achieve this, ID data are partitioned using a conventional approach, whereas OOD data are divided by grouping samples based on their classes. Furthermore, we analyze the context of data with class hierarchy to propose a data splitting that considers the entire class hierarchy to obtain fair ID-OOD partitions to apply the proposed evaluation framework. This framework is called Dual Cross-Validation for Robust Out-of-Distribution Detection (DCV-ROOD). To test the validity of the evaluation framework, we selected a set of state-of-the-art OOD detection methods, both with and without outlier exposure. The results show that the method achieves very fast convergence to the true performance. 

---
# Real-E: A Foundation Benchmark for Advancing Robust and Generalizable Electricity Forecasting 

**Authors**: Chen Shao, Yue Wang, Zhenyi Zhu, Zhanbo Huang, Sebastian Pütz, Benjamin Schäfer, Tobais Käfer, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2509.05768)  

**Abstract**: Energy forecasting is vital for grid reliability and operational efficiency. Although recent advances in time series forecasting have led to progress, existing benchmarks remain limited in spatial and temporal scope and lack multi-energy features. This raises concerns about their reliability and applicability in real-world deployment. To address this, we present the Real-E dataset, covering over 74 power stations across 30+ European countries over a 10-year span with rich metadata. Using Real- E, we conduct an extensive data analysis and benchmark over 20 baselines across various model types. We introduce a new metric to quantify shifts in correlation structures and show that existing methods struggle on our dataset, which exhibits more complex and non-stationary correlation dynamics. Our findings highlight key limitations of current methods and offer a strong empirical basis for building more robust forecasting models 

---
# Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System 

**Authors**: Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She  

**Link**: [PDF](https://arxiv.org/pdf/2509.05755)  

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems. 

---
# Tell-Tale Watermarks for Explanatory Reasoning in Synthetic Media Forensics 

**Authors**: Ching-Chun Chang, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05753)  

**Abstract**: The rise of synthetic media has blurred the boundary between reality and fabrication under the evolving power of artificial intelligence, fueling an infodemic that erodes public trust in cyberspace. For digital imagery, a multitude of editing applications further complicates the forensic analysis, including semantic edits that alter content, photometric adjustments that recalibrate colour characteristics, and geometric projections that reshape viewpoints. Collectively, these transformations manipulate and control perceptual interpretation of digital imagery. This susceptibility calls for forensic enquiry into reconstructing the chain of events, thereby revealing deeper evidential insight into the presence or absence of criminal intent. This study seeks to address an inverse problem of tracing the underlying generation chain that gives rise to the observed synthetic media. A tell-tale watermarking system is developed for explanatory reasoning over the nature and extent of transformations across the lifecycle of synthetic media. Tell-tale watermarks are tailored to different classes of transformations, responding in a manner that is neither strictly robust nor fragile but instead interpretable. These watermarks function as reference clues that evolve under the same transformation dynamics as the carrier media, leaving interpretable traces when subjected to transformations. Explanatory reasoning is then performed to infer the most plausible account across the combinatorial parameter space of composite transformations. Experimental evaluations demonstrate the validity of tell-tale watermarking with respect to fidelity, synchronicity and traceability. 

---
# Unleashing Hierarchical Reasoning: An LLM-Driven Framework for Training-Free Referring Video Object Segmentation 

**Authors**: Bingrui Zhao, Lin Yuanbo Wu, Xiangtian Fan, Deyin Liu, Lu Zhang, Ruyi He, Jialie Shen, Ximing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05751)  

**Abstract**: Referring Video Object Segmentation (RVOS) aims to segment an object of interest throughout a video based on a language description. The prominent challenge lies in aligning static text with dynamic visual content, particularly when objects exhibiting similar appearances with inconsistent motion and poses. However, current methods often rely on a holistic visual-language fusion that struggles with complex, compositional descriptions. In this paper, we propose \textbf{PARSE-VOS}, a novel, training-free framework powered by Large Language Models (LLMs), for a hierarchical, coarse-to-fine reasoning across text and video domains. Our approach begins by parsing the natural language query into structured semantic commands. Next, we introduce a spatio-temporal grounding module that generates all candidate trajectories for all potential target objects, guided by the parsed semantics. Finally, a hierarchical identification module select the correct target through a two-stage reasoning process: it first performs coarse-grained motion reasoning with an LLM to narrow down candidates; if ambiguity remains, a fine-grained pose verification stage is conditionally triggered to disambiguate. The final output is an accurate segmentation mask for the target object. \textbf{PARSE-VOS} achieved state-of-the-art performance on three major benchmarks: Ref-YouTube-VOS, Ref-DAVIS17, and MeViS. 

---
# InterAct: A Large-Scale Dataset of Dynamic, Expressive and Interactive Activities between Two People in Daily Scenarios 

**Authors**: Leo Ho, Yinghao Huang, Dafei Qin, Mingyi Shi, Wangpok Tse, Wei Liu, Junichi Yamagishi, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2509.05747)  

**Abstract**: We address the problem of accurate capture of interactive behaviors between two people in daily scenarios. Most previous works either only consider one person or solely focus on conversational gestures of two people, assuming the body orientation and/or position of each actor are constant or barely change over each interaction. In contrast, we propose to simultaneously model two people's activities, and target objective-driven, dynamic, and semantically consistent interactions which often span longer duration and cover bigger space. To this end, we capture a new multi-modal dataset dubbed InterAct, which is composed of 241 motion sequences where two people perform a realistic and coherent scenario for one minute or longer over a complete interaction. For each sequence, two actors are assigned different roles and emotion labels, and collaborate to finish one task or conduct a common interaction activity. The audios, body motions, and facial expressions of both persons are captured. InterAct contains diverse and complex motions of individuals and interesting and relatively long-term interaction patterns barely seen before. We also demonstrate a simple yet effective diffusion-based method that estimates interactive face expressions and body motions of two people from speech inputs. Our method regresses the body motions in a hierarchical manner, and we also propose a novel fine-tuning mechanism to improve the lip accuracy of facial expressions. To facilitate further research, the data and code is made available at this https URL . 

---
# Reasoning Introduces New Poisoning Attacks Yet Makes Them More Complicated 

**Authors**: Hanna Foerster, Ilia Shumailov, Yiren Zhao, Harsh Chaudhari, Jamie Hayes, Robert Mullins, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2509.05739)  

**Abstract**: Early research into data poisoning attacks against Large Language Models (LLMs) demonstrated the ease with which backdoors could be injected. More recent LLMs add step-by-step reasoning, expanding the attack surface to include the intermediate chain-of-thought (CoT) and its inherent trait of decomposing problems into subproblems. Using these vectors for more stealthy poisoning, we introduce ``decomposed reasoning poison'', in which the attacker modifies only the reasoning path, leaving prompts and final answers clean, and splits the trigger across multiple, individually harmless components.
Fascinatingly, while it remains possible to inject these decomposed poisons, reliably activating them to change final answers (rather than just the CoT) is surprisingly difficult. This difficulty arises because the models can often recover from backdoors that are activated within their thought processes. Ultimately, it appears that an emergent form of backdoor robustness is originating from the reasoning capabilities of these advanced LLMs, as well as from the architectural separation between reasoning and final answer generation. 

---
# Offline vs. Online Learning in Model-based RL: Lessons for Data Collection Strategies 

**Authors**: Jiaqi Chen, Ji Shi, Cansu Sancaktar, Jonas Frey, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2509.05735)  

**Abstract**: Data collection is crucial for learning robust world models in model-based reinforcement learning. The most prevalent strategies are to actively collect trajectories by interacting with the environment during online training or training on offline datasets. At first glance, the nature of learning task-agnostic environment dynamics makes world models a good candidate for effective offline training. However, the effects of online vs. offline data on world models and thus on the resulting task performance have not been thoroughly studied in the literature. In this work, we investigate both paradigms in model-based settings, conducting experiments on 31 different environments. First, we showcase that online agents outperform their offline counterparts. We identify a key challenge behind performance degradation of offline agents: encountering Out-Of-Distribution states at test time. This issue arises because, without the self-correction mechanism in online agents, offline datasets with limited state space coverage induce a mismatch between the agent's imagination and real rollouts, compromising policy training. We demonstrate that this issue can be mitigated by allowing for additional online interactions in a fixed or adaptive schedule, restoring the performance of online training with limited interaction data. We also showcase that incorporating exploration data helps mitigate the performance degradation of offline agents. Based on our insights, we recommend adding exploration data when collecting large datasets, as current efforts predominantly focus on expert data alone. 

---
# Simulation Priors for Data-Efficient Deep Learning 

**Authors**: Lenart Treven, Bhavya Sukhija, Jonas Rothfuss, Stelian Coros, Florian Dörfler, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2509.05732)  

**Abstract**: How do we enable AI systems to efficiently learn in the real-world? First-principles models are widely used to simulate natural systems, but often fail to capture real-world complexity due to simplifying assumptions. In contrast, deep learning approaches can estimate complex dynamics with minimal assumptions but require large, representative datasets. We propose SimPEL, a method that efficiently combines first-principles models with data-driven learning by using low-fidelity simulators as priors in Bayesian deep learning. This enables SimPEL to benefit from simulator knowledge in low-data regimes and leverage deep learning's flexibility when more data is available, all the while carefully quantifying epistemic uncertainty. We evaluate SimPEL on diverse systems, including biological, agricultural, and robotic domains, showing superior performance in learning complex dynamics. For decision-making, we demonstrate that SimPEL bridges the sim-to-real gap in model-based reinforcement learning. On a high-speed RC car task, SimPEL learns a highly dynamic parking maneuver involving drifting with substantially less data than state-of-the-art baselines. These results highlight the potential of SimPEL for data-efficient learning and control in complex real-world environments. 

---
# LiDAR-BIND-T: Improving SLAM with Temporally Consistent Cross-Modal LiDAR Reconstruction 

**Authors**: Niels Balemans, Ali Anwar, Jan Steckel, Siegfried Mercelis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05728)  

**Abstract**: This paper extends LiDAR-BIND, a modular multi-modal fusion framework that binds heterogeneous sensors (radar, sonar) to a LiDAR-defined latent space, with mechanisms that explicitly enforce temporal consistency. We introduce three contributions: (i) temporal embedding similarity that aligns consecutive latents, (ii) a motion-aligned transformation loss that matches displacement between predictions and ground truth LiDAR, and (iii) windows temporal fusion using a specialised temporal module. We further update the model architecture to better preserve spatial structure. Evaluations on radar/sonar-to-LiDAR translation demonstrate improved temporal and spatial coherence, yielding lower absolute trajectory error and better occupancy map accuracy in Cartographer-based SLAM (Simultaneous Localisation and Mapping). We propose different metrics based on the Fréchet Video Motion Distance (FVMD) and a correlation-peak distance metric providing practical temporal quality indicators to evaluate SLAM performance. The proposed temporal LiDAR-BIND, or LiDAR-BIND-T, maintains plug-and-play modality fusion while substantially enhancing temporal stability, resulting in improved robustness and performance for downstream SLAM. 

---
# A Survey of the State-of-the-Art in Conversational Question Answering Systems 

**Authors**: Manoj Madushanka Perera, Adnan Mahmood, Kasun Eranda Wijethilake, Fahmida Islam, Maryam Tahermazandarani, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.05716)  

**Abstract**: Conversational Question Answering (ConvQA) systems have emerged as a pivotal area within Natural Language Processing (NLP) by driving advancements that enable machines to engage in dynamic and context-aware conversations. These capabilities are increasingly being applied across various domains, i.e., customer support, education, legal, and healthcare where maintaining a coherent and relevant conversation is essential. Building on recent advancements, this survey provides a comprehensive analysis of the state-of-the-art in ConvQA. This survey begins by examining the core components of ConvQA systems, i.e., history selection, question understanding, and answer prediction, highlighting their interplay in ensuring coherence and relevance in multi-turn conversations. It further investigates the use of advanced machine learning techniques, including but not limited to, reinforcement learning, contrastive learning, and transfer learning to improve ConvQA accuracy and efficiency. The pivotal role of large language models, i.e., RoBERTa, GPT-4, Gemini 2.0 Flash, Mistral 7B, and LLaMA 3, is also explored, thereby showcasing their impact through data scalability and architectural advancements. Additionally, this survey presents a comprehensive analysis of key ConvQA datasets and concludes by outlining open research directions. Overall, this work offers a comprehensive overview of the ConvQA landscape and provides valuable insights to guide future advancements in the field. 

---
# Knowledge-Augmented Vision Language Models for Underwater Bioacoustic Spectrogram Analysis 

**Authors**: Ragib Amin Nihal, Benjamin Yen, Takeshi Ashizawa, Kazuhiro Nakadai  

**Link**: [PDF](https://arxiv.org/pdf/2509.05703)  

**Abstract**: Marine mammal vocalization analysis depends on interpreting bioacoustic spectrograms. Vision Language Models (VLMs) are not trained on these domain-specific visualizations. We investigate whether VLMs can extract meaningful patterns from spectrograms visually. Our framework integrates VLM interpretation with LLM-based validation to build domain knowledge. This enables adaptation to acoustic data without manual annotation or model retraining. 

---
# Revealing the Numeracy Gap: An Empirical Investigation of Text Embedding Models 

**Authors**: Ningyuan Deng, Hanyu Duan, Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05691)  

**Abstract**: Text embedding models are widely used in natural language processing applications. However, their capability is often benchmarked on tasks that do not require understanding nuanced numerical information in text. As a result, it remains unclear whether current embedding models can precisely encode numerical content, such as numbers, into embeddings. This question is critical because embedding models are increasingly applied in domains where numbers matter, such as finance and healthcare. For example, Company X's market share grew by 2\% should be interpreted very differently from Company X's market share grew by 20\%, even though both indicate growth in market share. This study aims to examine whether text embedding models can capture such nuances. Using synthetic data in a financial context, we evaluate 13 widely used text embedding models and find that they generally struggle to capture numerical details accurately. Our further analyses provide deeper insights into embedding numeracy, informing future research to strengthen embedding model-based NLP systems with improved capacity for handling numerical content. 

---
# SEASONED: Semantic-Enhanced Self-Counterfactual Explainable Detection of Adversarial Exploiter Contracts 

**Authors**: Xng Ai, Shudan Lin, Zecheng Li, Kai Zhou, Bixin Li, Bin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05681)  

**Abstract**: Decentralized Finance (DeFi) attacks have resulted in significant losses, often orchestrated through Adversarial Exploiter Contracts (AECs) that exploit vulnerabilities in victim smart contracts. To proactively identify such threats, this paper targets the explainable detection of AECs.
Existing detection methods struggle to capture semantic dependencies and lack interpretability, limiting their effectiveness and leaving critical knowledge gaps in AEC analysis. To address these challenges, we introduce SEASONED, an effective, self-explanatory, and robust framework for AEC detection.
SEASONED extracts semantic information from contract bytecode to construct a semantic relation graph (SRG), and employs a self-counterfactual explainable detector (SCFED) to classify SRGs and generate explanations that highlight the core attack logic. SCFED further enhances robustness, generalizability, and data efficiency by extracting representative information from these explanations. Both theoretical analysis and experimental results demonstrate the effectiveness of SEASONED, which showcases outstanding detection performance, robustness, generalizability, and data efficiency learning ability. To support further research, we also release a new dataset of 359 AECs. 

---
# GraMFedDHAR: Graph Based Multimodal Differentially Private Federated HAR 

**Authors**: Labani Halder, Tanmay Sen, Sarbani Palit  

**Link**: [PDF](https://arxiv.org/pdf/2509.05671)  

**Abstract**: Human Activity Recognition (HAR) using multimodal sensor data remains challenging due to noisy or incomplete measurements, scarcity of labeled examples, and privacy concerns. Traditional centralized deep learning approaches are often constrained by infrastructure availability, network latency, and data sharing restrictions. While federated learning (FL) addresses privacy by training models locally and sharing only model parameters, it still has to tackle issues arising from the use of heterogeneous multimodal data and differential privacy requirements. In this article, a Graph-based Multimodal Federated Learning framework, GraMFedDHAR, is proposed for HAR tasks. Diverse sensor streams such as a pressure mat, depth camera, and multiple accelerometers are modeled as modality-specific graphs, processed through residual Graph Convolutional Neural Networks (GCNs), and fused via attention-based weighting rather than simple concatenation. The fused embeddings enable robust activity classification, while differential privacy safeguards data during federated aggregation. Experimental results show that the proposed MultiModalGCN model outperforms the baseline MultiModalFFN, with up to 2 percent higher accuracy in non-DP settings in both centralized and federated paradigms. More importantly, significant improvements are observed under differential privacy constraints: MultiModalGCN consistently surpasses MultiModalFFN, with performance gaps ranging from 7 to 13 percent depending on the privacy budget and setting. These results highlight the robustness of graph-based modeling in multimodal learning, where GNNs prove more resilient to the performance degradation introduced by DP noise. 

---
# Llama-GENBA-10B: A Trilingual Large Language Model for German, English and Bavarian 

**Authors**: Michael Hoffmann, Jophin John, Stefan Schweter, Gokul Ramakrishnan, Hoi-Fong Mak, Alice Zhang, Dmitry Gaynullin, Nicolay J. Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2509.05668)  

**Abstract**: We present Llama-GENBA-10B, a trilingual foundation model addressing English-centric bias in large language models. Built on Llama 3.1-8B and scaled to 10B parameters, Llama-GENBA-10B is continuously pretrained on 164B tokens (82B English, 82B German, and 80M Bavarian), balancing resources while preventing English dominance. Targeted at the German NLP community, the model also promotes Bavarian as a low-resource language. Development tackled four challenges: (1) curating a multilingual corpus despite Bavarian scarcity, (2) creating a unified tokenizer for English, German, and Bavarian, (3) optimizing architecture and language-ratio hyperparameters for cross-lingual transfer, and (4) establishing the first standardized trilingual evaluation suite by translating German benchmarks into Bavarian. Evaluations show that Llama-GENBA-10B achieves strong cross-lingual performance, with the fine-tuned variant surpassing Apertus-8B-2509 and gemma-2-9b in Bavarian and establishing itself as the best model in its class for this language, while also outperforming EuroLLM in English and matching its results in German. Training on the Cerebras CS-2 demonstrated efficient large-scale multilingual pretraining with documented energy use, offering a blueprint for inclusive foundation models that integrate low-resource languages. 

---
# LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding 

**Authors**: Yuxuan Hu, Jihao Liu, Ke Wang, Jinliang Zhen, Weikang Shi, Manyuan Zhang, Qi Dou, Rui Liu, Aojun Zhou, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05657)  

**Abstract**: Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at this https URL. 

---
# OptiProxy-NAS: Optimization Proxy based End-to-End Neural Architecture Search 

**Authors**: Bo Lyu, Yu Cui, Tuo Shi, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05656)  

**Abstract**: Neural architecture search (NAS) is a hard computationally expensive optimization problem with a discrete, vast, and spiky search space. One of the key research efforts dedicated to this space focuses on accelerating NAS via certain proxy evaluations of neural architectures. Different from the prevalent predictor-based methods using surrogate models and differentiable architecture search via supernetworks, we propose an optimization proxy to streamline the NAS as an end-to-end optimization framework, named OptiProxy-NAS. In particular, using a proxy representation, the NAS space is reformulated to be continuous, differentiable, and smooth. Thereby, any differentiable optimization method can be applied to the gradient-based search of the relaxed architecture parameters. Our comprehensive experiments on $12$ NAS tasks of $4$ search spaces across three different domains including computer vision, natural language processing, and resource-constrained NAS fully demonstrate the superior search results and efficiency. Further experiments on low-fidelity scenarios verify the flexibility. 

---
# Orchestrator: Active Inference for Multi-Agent Systems in Long-Horizon Tasks 

**Authors**: Lukas Beckenbauer, Johannes-Lucas Loewe, Ge Zheng, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.05651)  

**Abstract**: Complex, non-linear tasks challenge LLM-enhanced multi-agent systems (MAS) due to partial observability and suboptimal coordination. We propose Orchestrator, a novel MAS framework that leverages attention-inspired self-emergent coordination and reflective benchmarking to optimize global task performance. Orchestrator introduces a monitoring mechanism to track agent-environment dynamics, using active inference benchmarks to optimize system behavior. By tracking agent-to-agent and agent-to-environment interaction, Orchestrator mitigates the effects of partial observability and enables agents to approximate global task solutions more efficiently. We evaluate the framework on a series of maze puzzles of increasing complexity, demonstrating its effectiveness in enhancing coordination and performance in dynamic, non-linear environments with long-horizon objectives. 

---
# Self-supervised Learning for Hyperspectral Images of Trees 

**Authors**: Moqsadur Rahman, Saurav Kumar, Santosh S. Palmate, M. Shahriar Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2509.05630)  

**Abstract**: Aerial remote sensing using multispectral and RGB imagers has provided a critical impetus to precision agriculture. Analysis of the hyperspectral images with limited or no labels is challenging. This paper focuses on self-supervised learning to create neural network embeddings reflecting vegetation properties of trees from aerial hyperspectral images of crop fields. Experimental results demonstrate that a constructed tree representation, using a vegetation property-related embedding space, performs better in downstream machine learning tasks compared to the direct use of hyperspectral vegetation properties as tree representations. 

---
# From Joy to Fear: A Benchmark of Emotion Estimation in Pop Song Lyrics 

**Authors**: Shay Dahary, Avi Edana, Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.05617)  

**Abstract**: The emotional content of song lyrics plays a pivotal role in shaping listener experiences and influencing musical preferences. This paper investigates the task of multi-label emotional attribution of song lyrics by predicting six emotional intensity scores corresponding to six fundamental emotions. A manually labeled dataset is constructed using a mean opinion score (MOS) approach, which aggregates annotations from multiple human raters to ensure reliable ground-truth labels. Leveraging this dataset, we conduct a comprehensive evaluation of several publicly available large language models (LLMs) under zero-shot scenarios. Additionally, we fine-tune a BERT-based model specifically for predicting multi-label emotion scores. Experimental results reveal the relative strengths and limitations of zero-shot and fine-tuned models in capturing the nuanced emotional content of lyrics. Our findings highlight the potential of LLMs for emotion recognition in creative texts, providing insights into model selection strategies for emotion-based music information retrieval applications. The labeled dataset is available at this https URL. 

---
# Causal Debiasing Medical Multimodal Representation Learning with Missing Modalities 

**Authors**: Xiaoguang Zhu, Lianlong Sun, Yang Liu, Pengyi Jiang, Uma Srivatsa, Nipavan Chiamvimonvat, Vladimir Filkov  

**Link**: [PDF](https://arxiv.org/pdf/2509.05615)  

**Abstract**: Medical multimodal representation learning aims to integrate heterogeneous clinical data into unified patient representations to support predictive modeling, which remains an essential yet challenging task in the medical data mining community. However, real-world medical datasets often suffer from missing modalities due to cost, protocol, or patient-specific constraints. Existing methods primarily address this issue by learning from the available observations in either the raw data space or feature space, but typically neglect the underlying bias introduced by the data acquisition process itself. In this work, we identify two types of biases that hinder model generalization: missingness bias, which results from non-random patterns in modality availability, and distribution bias, which arises from latent confounders that influence both observed features and outcomes. To address these challenges, we perform a structural causal analysis of the data-generating process and propose a unified framework that is compatible with existing direct prediction-based multimodal learning methods. Our method consists of two key components: (1) a missingness deconfounding module that approximates causal intervention based on backdoor adjustment and (2) a dual-branch neural network that explicitly disentangles causal features from spurious correlations. We evaluated our method in real-world public and in-hospital datasets, demonstrating its effectiveness and causal insights. 

---
# SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning 

**Authors**: Hanzhen Wang, Jiaming Xu, Jiayi Pan, Yongkang Zhou, Guohao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.05614)  

**Abstract**: Pruning accelerates compute-bound models by reducing computation. Recently applied to Vision-Language-Action (VLA) models, existing methods prune tokens using only local info from current action, ignoring global context from prior actions, causing >20% success rate drop and limited speedup. We observe high similarity across consecutive actions and propose leveraging both local (current) and global (past) info for smarter token selection. We introduce SpecPrune-VLA, a training-free method with two-level pruning and heuristic control: (1) Static pruning at action level: uses global history and local context to reduce visual tokens per action; (2) Dynamic pruning at layer level: prunes tokens per layer based on layer-specific importance; (3) Lightweight action-aware controller: classifies actions as coarse/fine-grained (by speed), adjusting pruning aggressiveness since fine-grained actions are pruning-sensitive. Experiments on LIBERO show SpecPrune-VLA achieves 1.46 times speedup on NVIDIA A800 and 1.57 times on NVIDIA GeForce RTX 3090 vs. OpenVLA-OFT, with negligible success rate loss. 

---
# Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints 

**Authors**: Waris Gill, Natalie Isak, Matthew Dressman  

**Link**: [PDF](https://arxiv.org/pdf/2509.05608)  

**Abstract**: The widespread deployment of LLMs across enterprise services has created a critical security blind spot. Organizations operate multiple LLM services handling billions of queries daily, yet regulatory compliance boundaries prevent these services from sharing threat intelligence about prompt injection attacks, the top security risk for LLMs. When an attack is detected in one service, the same threat may persist undetected in others for months, as privacy regulations prohibit sharing user prompts across compliance boundaries.
We present BinaryShield, the first privacy-preserving threat intelligence system that enables secure sharing of attack fingerprints across compliance boundaries. BinaryShield transforms suspicious prompts through a unique pipeline combining PII redaction, semantic embedding, binary quantization, and randomized response mechanism to potentially generate non-invertible fingerprints that preserve attack patterns while providing privacy. Our evaluations demonstrate that BinaryShield achieves an F1-score of 0.94, significantly outperforming SimHash (0.77), the privacy-preserving baseline, while achieving 64x storage reduction and 38x faster similarity search compared to dense embeddings. 

---
# Icon$^{2}$: Aligning Large Language Models Using Self-Synthetic Preference Data via Inherent Regulation 

**Authors**: Qiyuan Chen, Hongsen Huang, Qian Shao, Jiahe Chen, Jintai Chen, Hongxia Xu, Renjie Hua, Ren Chuan, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05605)  

**Abstract**: Large Language Models (LLMs) require high quality preference datasets to align with human preferences. However, conventional methods for constructing such datasets face significant challenges: reliance on pre-collected instructions often leads to distribution mismatches with target models, while the need for sampling multiple stochastic responses introduces substantial computational overhead. In this work, we explore a paradigm shift by leveraging inherent regulation of LLMs' representation space for efficient and tailored preference dataset construction, named Icon$^{2}$. Specifically, it first extracts layer-wise direction vectors to encode sophisticated human preferences and then uses these vectors to filter self-synthesized instructions based on their inherent consistency. During decoding, bidirectional inherent control is applied to steer token representations, enabling the precise generation of response pairs with clear alignment distinctions. Experimental results demonstrate significant improvements in both alignment and efficiency. Llama3-8B and Qwen2-7B achieve an average win rate improvement of 13.89% on AlpacaEval 2.0 and 13.45% on Arena-Hard, while reducing computational costs by up to 48.1%. 

---
# Language-guided Recursive Spatiotemporal Graph Modeling for Video Summarization 

**Authors**: Jungin Park, Jiyoung Lee, Kwanghoon Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2509.05604)  

**Abstract**: Video summarization aims to select keyframes that are visually diverse and can represent the whole story of a given video. Previous approaches have focused on global interlinkability between frames in a video by temporal modeling. However, fine-grained visual entities, such as objects, are also highly related to the main content of the video. Moreover, language-guided video summarization, which has recently been studied, requires a comprehensive linguistic understanding of complex real-world videos. To consider how all the objects are semantically related to each other, this paper regards video summarization as a language-guided spatiotemporal graph modeling problem. We present recursive spatiotemporal graph networks, called VideoGraph, which formulate the objects and frames as nodes of the spatial and temporal graphs, respectively. The nodes in each graph are connected and aggregated with graph edges, representing the semantic relationships between the nodes. To prevent the edges from being configured with visual similarity, we incorporate language queries derived from the video into the graph node representations, enabling them to contain semantic knowledge. In addition, we adopt a recursive strategy to refine initial graphs and correctly classify each frame node as a keyframe. In our experiments, VideoGraph achieves state-of-the-art performance on several benchmarks for generic and query-focused video summarization in both supervised and unsupervised manners. The code is available at this https URL. 

---
# Natural Language-Programming Language Software Traceability Link Recovery Needs More than Textual Similarity 

**Authors**: Zhiyuan Zou, Bangchao Wang, Peng Liang, Tingting Bi, Huan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05585)  

**Abstract**: In the field of software traceability link recovery (TLR), textual similarity has long been regarded as the core criterion. However, in tasks involving natural language and programming language (NL-PL) artifacts, relying solely on textual similarity is limited by their semantic gap. To this end, we conducted a large-scale empirical evaluation across various types of TLR tasks, revealing the limitations of textual similarity in NL-PL scenarios. To address these limitations, we propose an approach that incorporates multiple domain-specific auxiliary strategies, identified through empirical analysis, into two models: the Heterogeneous Graph Transformer (HGT) via edge types and the prompt-based Gemini 2.5 Pro via additional input information. We then evaluated our approach using the widely studied requirements-to-code TLR task, a representative case of NL-PL TLR. Experimental results show that both the multi-strategy HGT and Gemini 2.5 Pro models outperformed their original counterparts without strategy integration. Furthermore, compared to the current state-of-the-art method HGNNLink, the multi-strategy HGT and Gemini 2.5 Pro models achieved average F1-score improvements of 3.68% and 8.84%, respectively, across twelve open-source projects, demonstrating the effectiveness of multi-strategy integration in enhancing overall model performance for the requirements-code TLR task. 

---
# Learning to Walk in Costume: Adversarial Motion Priors for Aesthetically Constrained Humanoids 

**Authors**: Arturo Flores Alvarez, Fatemeh Zargarbashi, Havel Liu, Shiqi Wang, Liam Edwards, Jessica Anz, Alex Xu, Fan Shi, Stelian Coros, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.05581)  

**Abstract**: We present a Reinforcement Learning (RL)-based locomotion system for Cosmo, a custom-built humanoid robot designed for entertainment applications. Unlike traditional humanoids, entertainment robots present unique challenges due to aesthetic-driven design choices. Cosmo embodies these with a disproportionately large head (16% of total mass), limited sensing, and protective shells that considerably restrict movement. To address these challenges, we apply Adversarial Motion Priors (AMP) to enable the robot to learn natural-looking movements while maintaining physical stability. We develop tailored domain randomization techniques and specialized reward structures to ensure safe sim-to-real, protecting valuable hardware components during deployment. Our experiments demonstrate that AMP generates stable standing and walking behaviors despite Cosmo's extreme mass distribution and movement constraints. These results establish a promising direction for robots that balance aesthetic appeal with functional performance, suggesting that learning-based methods can effectively adapt to aesthetic-driven design constraints. 

---
# Using Contrastive Learning to Improve Two-Way Reasoning in Large Language Models: The Obfuscation Task as a Case Study 

**Authors**: Serge Lionel Nikiema, Jordan Samhi, Micheline Bénédicte Moumoula, Albérick Euraste Djiré, Abdoul Kader Kaboré, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2509.05553)  

**Abstract**: This research addresses a fundamental question in AI: whether large language models truly understand concepts or simply recognize patterns. The authors propose bidirectional reasoning,the ability to apply transformations in both directions without being explicitly trained on the reverse direction, as a test for genuine understanding. They argue that true comprehension should naturally allow reversibility. For example, a model that can change a variable name like userIndex to i should also be able to infer that i represents a user index without reverse training. The researchers tested current language models and discovered what they term cognitive specialization: when models are fine-tuned on forward tasks, their performance on those tasks improves, but their ability to reason bidirectionally becomes significantly worse. To address this issue, they developed Contrastive Fine-Tuning (CFT), which trains models using three types of examples: positive examples that maintain semantic meaning, negative examples with different semantics, and forward-direction obfuscation examples. This approach aims to develop deeper understanding rather than surface-level pattern recognition and allows reverse capabilities to develop naturally without explicit reverse training. Their experiments demonstrated that CFT successfully achieved bidirectional reasoning, enabling strong reverse performance while maintaining forward task capabilities. The authors conclude that bidirectional reasoning serves both as a theoretical framework for assessing genuine understanding and as a practical training approach for developing more capable AI systems. 

---
# Combining TSL and LLM to Automate REST API Testing: A Comparative Study 

**Authors**: Thiago Barradas, Aline Paes, Vânia de Oliveira Neves  

**Link**: [PDF](https://arxiv.org/pdf/2509.05540)  

**Abstract**: The effective execution of tests for REST APIs remains a considerable challenge for development teams, driven by the inherent complexity of distributed systems, the multitude of possible scenarios, and the limited time available for test design. Exhaustive testing of all input combinations is impractical, often resulting in undetected failures, high manual effort, and limited test coverage. To address these issues, we introduce RestTSLLM, an approach that uses Test Specification Language (TSL) in conjunction with Large Language Models (LLMs) to automate the generation of test cases for REST APIs. The approach targets two core challenges: the creation of test scenarios and the definition of appropriate input data. The proposed solution integrates prompt engineering techniques with an automated pipeline to evaluate various LLMs on their ability to generate tests from OpenAPI specifications. The evaluation focused on metrics such as success rate, test coverage, and mutation score, enabling a systematic comparison of model performance. The results indicate that the best-performing LLMs - Claude 3.5 Sonnet (Anthropic), Deepseek R1 (Deepseek), Qwen 2.5 32b (Alibaba), and Sabia 3 (Maritaca) - consistently produced robust and contextually coherent REST API tests. Among them, Claude 3.5 Sonnet outperformed all other models across every metric, emerging in this study as the most suitable model for this task. These findings highlight the potential of LLMs to automate the generation of tests based on API specifications. 

---
# OpenEgo: A Large-Scale Multimodal Egocentric Dataset for Dexterous Manipulation 

**Authors**: Ahad Jawaid, Yu Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05513)  

**Abstract**: Egocentric human videos provide scalable demonstrations for imitation learning, but existing corpora often lack either fine-grained, temporally localized action descriptions or dexterous hand annotations. We introduce OpenEgo, a multimodal egocentric manipulation dataset with standardized hand-pose annotations and intention-aligned action primitives. OpenEgo totals 1107 hours across six public datasets, covering 290 manipulation tasks in 600+ environments. We unify hand-pose layouts and provide descriptive, timestamped action primitives. To validate its utility, we train language-conditioned imitation-learning policies to predict dexterous hand trajectories. OpenEgo is designed to lower the barrier to learning dexterous manipulation from egocentric video and to support reproducible research in vision-language-action learning. All resources and instructions will be released at this http URL. 

---
# Microrobot Vascular Parkour: Analytic Geometry-based Path Planning with Real-time Dynamic Obstacle Avoidance 

**Authors**: Yanda Yang, Max Sokolich, Fatma Ceren Kirmizitas, Sambeeta Das, Andreas A. Malikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.05500)  

**Abstract**: Autonomous microrobots in blood vessels could enable minimally invasive therapies, but navigation is challenged by dense, moving obstacles. We propose a real-time path planning framework that couples an analytic geometry global planner (AGP) with two reactive local escape controllers, one based on rules and one based on reinforcement learning, to handle sudden moving obstacles. Using real-time imaging, the system estimates the positions of the microrobot, obstacles, and targets and computes collision-free motions. In simulation, AGP yields shorter paths and faster planning than weighted A* (WA*), particle swarm optimization (PSO), and rapidly exploring random trees (RRT), while maintaining feasibility and determinism. We extend AGP from 2D to 3D without loss of speed. In both simulations and experiments, the combined global planner and local controllers reliably avoid moving obstacles and reach targets. The average planning time is 40 ms per frame, compatible with 25 fps image acquisition and real-time closed-loop control. These results advance autonomous microrobot navigation and targeted drug delivery in vascular environments. 

---
# An Analysis of Layer-Freezing Strategies for Enhanced Transfer Learning in YOLO Architectures 

**Authors**: Andrzej D. Dobrzycki, Ana M. Bernardos, José R. Casar  

**Link**: [PDF](https://arxiv.org/pdf/2509.05490)  

**Abstract**: The You Only Look Once (YOLO) architecture is crucial for real-time object detection. However, deploying it in resource-constrained environments such as unmanned aerial vehicles (UAVs) requires efficient transfer learning. Although layer freezing is a common technique, the specific impact of various freezing configurations on contemporary YOLOv8 and YOLOv10 architectures remains unexplored, particularly with regard to the interplay between freezing depth, dataset characteristics, and training dynamics. This research addresses this gap by presenting a detailed analysis of layer-freezing strategies. We systematically investigate multiple freezing configurations across YOLOv8 and YOLOv10 variants using four challenging datasets that represent critical infrastructure monitoring. Our methodology integrates a gradient behavior analysis (L2 norm) and visual explanations (Grad-CAM) to provide deeper insights into training dynamics under different freezing strategies. Our results reveal that there is no universal optimal freezing strategy but, rather, one that depends on the properties of the data. For example, freezing the backbone is effective for preserving general-purpose features, while a shallower freeze is better suited to handling extreme class imbalance. These configurations reduce graphics processing unit (GPU) memory consumption by up to 28% compared to full fine-tuning and, in some cases, achieve mean average precision (mAP@50) scores that surpass those of full fine-tuning. Gradient analysis corroborates these findings, showing distinct convergence patterns for moderately frozen models. Ultimately, this work provides empirical findings and practical guidelines for selecting freezing strategies. It offers a practical, evidence-based approach to balanced transfer learning for object detection in scenarios with limited resources. 

---
# MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs 

**Authors**: Hongjun Xu, Junxi Xia, Weisi Yang, Yueyuan Sui, Stephen Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.05488)  

**Abstract**: Deploying Mamba models on microcontrollers (MCUs) remains challenging due to limited memory, the lack of native operator support, and the absence of embedded-friendly toolchains. We present, to our knowledge, the first deployment of a Mamba-based neural architecture on a resource-constrained MCU, a fully C-based runtime-free inference engine: MambaLite-Micro. Our pipeline maps a trained PyTorch Mamba model to on-device execution by (1) exporting model weights into a lightweight format, and (2) implementing a handcrafted Mamba layer and supporting operators in C with operator fusion and memory layout optimization. MambaLite-Micro eliminates large intermediate tensors, reducing 83.0% peak memory, while maintaining an average numerical error of only 1.7x10-5 relative to the PyTorch Mamba implementation. When evaluated on keyword spotting(KWS) and human activity recognition (HAR) tasks, MambaLite-Micro achieved 100% consistency with the PyTorch baselines, fully preserving classification accuracy. We further validated portability by deploying on both ESP32S3 and STM32H7 microcontrollers, demonstrating consistent operation across heterogeneous embedded platforms and paving the way for bringing advanced sequence models like Mamba to real-world resource-constrained applications. 

---
# The Token Tax: Systematic Bias in Multilingual Tokenization 

**Authors**: Jessica M. Lundin, Ada Zhang, Nihal Karim, Hamza Louzan, Victor Wei, David Adelani, Cody Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2509.05486)  

**Abstract**: Tokenization inefficiency imposes structural disadvantages on morphologically complex, low-resource languages, inflating compute resources and depressing accuracy. We evaluate 10 large language models (LLMs) on AfriMMLU (9,000 MCQA items; 5 subjects; 16 African languages) and show that fertility (tokens/word) reliably predicts accuracy. Higher fertility consistently predicts lower accuracy across all models and subjects. We further find that reasoning models (DeepSeek, o1) consistently outperform non-reasoning peers across high and low resource languages in the AfriMMLU dataset, narrowing accuracy gaps observed in prior generations. Finally, translating token inflation to economics, a doubling in tokens results in quadrupled training cost and time, underscoring the token tax faced by many languages. These results motivate morphologically aware tokenization, fair pricing, and multilingual benchmarks for equitable natural language processing (NLP). 

---
# PLanTS: Periodicity-aware Latent-state Representation Learning for Multivariate Time Series 

**Authors**: Jia Wang, Xiao Wang, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05478)  

**Abstract**: Multivariate time series (MTS) are ubiquitous in domains such as healthcare, climate science, and industrial monitoring, but their high dimensionality, limited labeled data, and non-stationary nature pose significant challenges for conventional machine learning methods. While recent self-supervised learning (SSL) approaches mitigate label scarcity by data augmentations or time point-based contrastive strategy, they neglect the intrinsic periodic structure of MTS and fail to capture the dynamic evolution of latent states. We propose PLanTS, a periodicity-aware self-supervised learning framework that explicitly models irregular latent states and their transitions. We first designed a period-aware multi-granularity patching mechanism and a generalized contrastive loss to preserve both instance-level and state-level similarities across multiple temporal resolutions. To further capture temporal dynamics, we design a next-transition prediction pretext task that encourages representations to encode predictive information about future state evolution. We evaluate PLanTS across a wide range of downstream tasks-including multi-class and multi-label classification, forecasting, trajectory tracking and anomaly detection. PLanTS consistently improves the representation quality over existing SSL methods and demonstrates superior runtime efficiency compared to DTW-based methods. 

---
# Learning Tool-Aware Adaptive Compliant Control for Autonomous Regolith Excavation 

**Authors**: Andrej Orsula, Matthieu Geist, Miguel Olivares-Mendez, Carol Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2509.05475)  

**Abstract**: Autonomous regolith excavation is a cornerstone of in-situ resource utilization for a sustained human presence beyond Earth. However, this task is fundamentally hindered by the complex interaction dynamics of granular media and the operational need for robots to use diverse tools. To address these challenges, this work introduces a framework where a model-based reinforcement learning agent learns within a parallelized simulation. This environment leverages high-fidelity particle physics and procedural generation to create a vast distribution of both lunar terrains and excavation tool geometries. To master this diversity, the agent learns an adaptive interaction strategy by dynamically modulating its own stiffness and damping at each control step through operational space control. Our experiments demonstrate that training with a procedural distribution of tools is critical for generalization and enables the development of sophisticated tool-aware behavior. Furthermore, we show that augmenting the agent with visual feedback significantly improves task success. These results represent a validated methodology for developing the robust and versatile autonomous systems required for the foundational tasks of future space missions. 

---
# From Vision to Validation: A Theory- and Data-Driven Construction of a GCC-Specific AI Adoption Index 

**Authors**: Mohammad Rashed Albous, Anwaar AlKandari, Abdel Latef Anouze  

**Link**: [PDF](https://arxiv.org/pdf/2509.05474)  

**Abstract**: Artificial intelligence (AI) is rapidly transforming public-sector processes worldwide, yet standardized measures rarely address the unique drivers, governance models, and cultural nuances of the Gulf Cooperation Council (GCC) countries. This study employs a theory-driven foundation derived from an in-depth analysis of literature review and six National AI Strategies (NASs), coupled with a data-driven approach that utilizes a survey of 203 mid- and senior-level government employees and advanced statistical techniques (K-Means clustering, Principal Component Analysis, and Partial Least Squares Structural Equation Modeling). By combining policy insights with empirical evidence, the research develops and validates a novel AI Adoption Index specifically tailored to the GCC public sector. Findings indicate that robust infrastructure and clear policy mandates exert the strongest influence on successful AI implementations, overshadowing organizational readiness in early adoption stages. The combined model explains 70% of the variance in AI outcomes, suggesting that resource-rich environments and top-down policy directives can drive rapid but uneven technology uptake. By consolidating key dimensions (Infrastructure & Resources, Organizational Readiness, and Policy & Regulatory Environment) into a single composite index, this study provides a holistic yet context-sensitive tool for benchmarking AI maturity. The index offers actionable guidance for policymakers seeking to harmonize large-scale deployments with ethical and regulatory standards. Beyond advancing academic discourse, these insights inform more strategic allocation of resources, cross-country cooperation, and capacity-building initiatives, thereby supporting sustained AI-driven transformation in the GCC region and beyond. 

---
# Behind the Mask: Benchmarking Camouflaged Jailbreaks in Large Language Models 

**Authors**: Youjia Zheng, Mohammad Zandsalimy, Shanu Sushmita  

**Link**: [PDF](https://arxiv.org/pdf/2509.05471)  

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to a sophisticated form of adversarial prompting known as camouflaged jailbreaking. This method embeds malicious intent within seemingly benign language to evade existing safety mechanisms. Unlike overt attacks, these subtle prompts exploit contextual ambiguity and the flexible nature of language, posing significant challenges to current defense systems. This paper investigates the construction and impact of camouflaged jailbreak prompts, emphasizing their deceptive characteristics and the limitations of traditional keyword-based detection methods. We introduce a novel benchmark dataset, Camouflaged Jailbreak Prompts, containing 500 curated examples (400 harmful and 100 benign prompts) designed to rigorously stress-test LLM safety protocols. In addition, we propose a multi-faceted evaluation framework that measures harmfulness across seven dimensions: Safety Awareness, Technical Feasibility, Implementation Safeguards, Harmful Potential, Educational Value, Content Quality, and Compliance Score. Our findings reveal a stark contrast in LLM behavior: while models demonstrate high safety and content quality with benign inputs, they exhibit a significant decline in performance and safety when confronted with camouflaged jailbreak attempts. This disparity underscores a pervasive vulnerability, highlighting the urgent need for more nuanced and adaptive security strategies to ensure the responsible and robust deployment of LLMs in real-world applications. 

---
# Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis 

**Authors**: Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2509.05449)  

**Abstract**: Membership inference attacks (MIAs) reveal whether specific data was used to train machine learning models, serving as important tools for privacy auditing and compliance assessment. Recent studies have reported that MIAs perform only marginally better than random guessing against large language models, suggesting that modern pre-training approaches with massive datasets may be free from privacy leakage risks. Our work offers a complementary perspective to these findings by exploring how examining LLMs' internal representations, rather than just their outputs, may provide additional insights into potential membership inference signals. Our framework, \emph{memTrace}, follows what we call \enquote{neural breadcrumbs} extracting informative signals from transformer hidden states and attention patterns as they process candidate sequences. By analyzing layer-wise representation dynamics, attention distribution characteristics, and cross-layer transition patterns, we detect potential memorization fingerprints that traditional loss-based approaches may not capture. This approach yields strong membership detection across several model families achieving average AUC scores of 0.85 on popular MIA benchmarks. Our findings suggest that internal model behaviors can reveal aspects of training data exposure even when output-based signals appear protected, highlighting the need for further research into membership privacy and the development of more robust privacy-preserving training techniques for large language models. 

---
# Newton to Einstein: Axiom-Based Discovery via Game Design 

**Authors**: Pingchuan Ma, Benjamin Tod Jones, Tsun-Hsuan Wang, Minghao Guo, Michal Piotr Lipiec, Chuang Gan, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2509.05448)  

**Abstract**: This position paper argues that machine learning for scientific discovery should shift from inductive pattern recognition to axiom-based reasoning. We propose a game design framework in which scientific inquiry is recast as a rule-evolving system: agents operate within environments governed by axioms and modify them to explain outlier observations. Unlike conventional ML approaches that operate within fixed assumptions, our method enables the discovery of new theoretical structures through systematic rule adaptation. We demonstrate the feasibility of this approach through preliminary experiments in logic-based games, showing that agents can evolve axioms that solve previously unsolvable problems. This framework offers a foundation for building machine learning systems capable of creative, interpretable, and theory-driven discovery. 

---
# Direct-Scoring NLG Evaluators Can Use Pairwise Comparisons Too 

**Authors**: Logan Lawrence, Ashton Williamson, Alexander Shelton  

**Link**: [PDF](https://arxiv.org/pdf/2509.05440)  

**Abstract**: As large-language models have been increasingly used as automatic raters for evaluating free-form content, including document summarization, dialog, and story generation, work has been dedicated to evaluating such models by measuring their correlations with human judgment. For \textit{sample-level} performance, methods which operate by using pairwise comparisons between machine-generated text perform well but often lack the ability to assign absolute scores to individual summaries, an ability crucial for use cases that require thresholding. In this work, we propose a direct-scoring method which uses synthetic summaries to act as pairwise machine rankings at test time. We show that our method performs comparably to state-of-the-art pairwise evaluators in terms of axis-averaged sample-level correlations on the SummEval (\textbf{+0.03}), TopicalChat (\textbf{-0.03}), and HANNA (\textbf{+0.05}) meta-evaluation benchmarks, and release the synthetic in-context summaries as data to facilitate future work. 

---
# Advanced Brain Tumor Segmentation Using EMCAD: Efficient Multi-scale Convolutional Attention Decoding 

**Authors**: GodsGift Uzor, Tania-Amanda Nkoyo Fredrick Eneye, Chukwuebuka Ijezue  

**Link**: [PDF](https://arxiv.org/pdf/2509.05431)  

**Abstract**: Brain tumor segmentation is a critical pre-processing step in the medical image analysis pipeline that involves precise delineation of tumor regions from healthy brain tissue in medical imaging data, particularly MRI scans. An efficient and effective decoding mechanism is crucial in brain tumor segmentation especially in scenarios with limited computational resources. However these decoding mechanisms usually come with high computational costs. To address this concern EMCAD a new efficient multi-scale convolutional attention decoder designed was utilized to optimize both performance and computational efficiency for brain tumor segmentation on the BraTs2020 dataset consisting of MRI scans from 369 brain tumor patients. The preliminary result obtained by the model achieved a best Dice score of 0.31 and maintained a stable mean Dice score of 0.285 plus/minus 0.015 throughout the training process which is moderate. The initial model maintained consistent performance across the validation set without showing signs of over-fitting. 

---
# No Translation Needed: Forecasting Quality from Fertility and Metadata 

**Authors**: Jessica M. Lundin, Ada Zhang, David Adelani, Cody Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2509.05425)  

**Abstract**: We show that translation quality can be predicted with surprising accuracy \textit{without ever running the translation system itself}. Using only a handful of features, token fertility ratios, token counts, and basic linguistic metadata (language family, script, and region), we can forecast ChrF scores for GPT-4o translations across 203 languages in the FLORES-200 benchmark. Gradient boosting models achieve favorable performance ($R^{2}=0.66$ for XX$\rightarrow$English and $R^{2}=0.72$ for English$\rightarrow$XX). Feature importance analyses reveal that typological factors dominate predictions into English, while fertility plays a larger role for translations into diverse target languages. These findings suggest that translation quality is shaped by both token-level fertility and broader linguistic typology, offering new insights for multilingual evaluation and quality estimation. 

---
# Universality of physical neural networks with multivariate nonlinearity 

**Authors**: Benjamin Savinson, David J. Norris, Siddhartha Mishra, Samuel Lanthaler  

**Link**: [PDF](https://arxiv.org/pdf/2509.05420)  

**Abstract**: The enormous energy demand of artificial intelligence is driving the development of alternative hardware for deep learning. Physical neural networks try to exploit physical systems to perform machine learning more efficiently. In particular, optical systems can calculate with light using negligible energy. While their computational capabilities were long limited by the linearity of optical materials, nonlinear computations have recently been demonstrated through modified input encoding. Despite this breakthrough, our inability to determine if physical neural networks can learn arbitrary relationships between data -- a key requirement for deep learning known as universality -- hinders further progress. Here we present a fundamental theorem that establishes a universality condition for physical neural networks. It provides a powerful mathematical criterion that imposes device constraints, detailing how inputs should be encoded in the tunable parameters of the physical system. Based on this result, we propose a scalable architecture using free-space optics that is provably universal and achieves high accuracy on image classification tasks. Further, by combining the theorem with temporal multiplexing, we present a route to potentially huge effective system sizes in highly practical but poorly scalable on-chip photonic devices. Our theorem and scaling methods apply beyond optical systems and inform the design of a wide class of universal, energy-efficient physical neural networks, justifying further efforts in their development. 

---
# Graph Connectionist Temporal Classification for Phoneme Recognition 

**Authors**: Henry Grafé, Hugo Van hamme  

**Link**: [PDF](https://arxiv.org/pdf/2509.05399)  

**Abstract**: Automatic Phoneme Recognition (APR) systems are often trained using pseudo phoneme-level annotations generated from text through Grapheme-to-Phoneme (G2P) systems. These G2P systems frequently output multiple possible pronunciations per word, but the standard Connectionist Temporal Classification (CTC) loss cannot account for such ambiguity during training. In this work, we adapt Graph Temporal Classification (GTC) to the APR setting. GTC enables training from a graph of alternative phoneme sequences, allowing the model to consider multiple pronunciations per word as valid supervision. Our experiments on English and Dutch data sets show that incorporating multiple pronunciations per word into the training loss consistently improves phoneme error rates compared to a baseline trained with CTC. These results suggest that integrating pronunciation variation into the loss function is a promising strategy for training APR systems from noisy G2P-based supervision. 

---
# Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate 

**Authors**: Andrea Wynn, Harsh Satija, Gillian Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2509.05396)  

**Abstract**: While multi-agent debate has been proposed as a promising strategy for improving AI reasoning ability, we find that debate can sometimes be harmful rather than helpful. The prior work has exclusively focused on debates within homogeneous groups of agents, whereas we explore how diversity in model capabilities influences the dynamics and outcomes of multi-agent interactions. Through a series of experiments, we demonstrate that debate can lead to a decrease in accuracy over time -- even in settings where stronger (i.e., more capable) models outnumber their weaker counterparts. Our analysis reveals that models frequently shift from correct to incorrect answers in response to peer reasoning, favoring agreement over challenging flawed reasoning. These results highlight important failure modes in the exchange of reasons during multi-agent debate, suggesting that naive applications of debate may cause performance degradation when agents are neither incentivized nor adequately equipped to resist persuasive but incorrect reasoning. 

---
# Reverse Browser: Vector-Image-to-Code Generator 

**Authors**: Zoltan Toth-Czifra  

**Link**: [PDF](https://arxiv.org/pdf/2509.05394)  

**Abstract**: Automating the conversion of user interface design into code (image-to-code or image-to-UI) is an active area of software engineering research. However, the state-of-the-art solutions do not achieve high fidelity to the original design, as evidenced by benchmarks. In this work, I approach the problem differently: I use vector images instead of bitmaps as model input. I create several large datasets for training machine learning models. I evaluate the available array of Image Quality Assessment (IQA) algorithms and introduce a new, multi-scale metric. I then train a large open-weights model and discuss its limitations. 

---
# Inferring Prerequisite Knowledge Concepts in Educational Knowledge Graphs: A Multi-criteria Approach 

**Authors**: Rawaa Alatrash, Mohamed Amine Chatti, Nasha Wibowo, Qurat Ul Ain  

**Link**: [PDF](https://arxiv.org/pdf/2509.05393)  

**Abstract**: Educational Knowledge Graphs (EduKGs) organize various learning entities and their relationships to support structured and adaptive learning. Prerequisite relationships (PRs) are critical in EduKGs for defining the logical order in which concepts should be learned. However, the current EduKG in the MOOC platform CourseMapper lacks explicit PR links, and manually annotating them is time-consuming and inconsistent. To address this, we propose an unsupervised method for automatically inferring concept PRs without relying on labeled data. We define ten criteria based on document-based, Wikipedia hyperlink-based, graph-based, and text-based features, and combine them using a voting algorithm to robustly capture PRs in educational content. Experiments on benchmark datasets show that our approach achieves higher precision than existing methods while maintaining scalability and adaptability, thus providing reliable support for sequence-aware learning in CourseMapper. 

---
# An Optimized Pipeline for Automatic Educational Knowledge Graph Construction 

**Authors**: Qurat Ul Ain, Mohamed Amine Chatti, Jean Qussa, Amr Shakhshir, Rawaa Alatrash, Shoeb Joarder  

**Link**: [PDF](https://arxiv.org/pdf/2509.05392)  

**Abstract**: The automatic construction of Educational Knowledge Graphs (EduKGs) is essential for domain knowledge modeling by extracting meaningful representations from learning materials. Despite growing interest, identifying a scalable and reliable approach for automatic EduKG generation remains a challenge. In an attempt to develop a unified and robust pipeline for automatic EduKG construction, in this study we propose a pipeline for automatic EduKG construction from PDF learning materials. The process begins with generating slide-level EduKGs from individual pages/slides, which are then merged to form a comprehensive EduKG representing the entire learning material. We evaluate the accuracy of the EduKG generated from the proposed pipeline in our MOOC platform, CourseMapper. The observed accuracy, while indicative of partial success, is relatively low particularly in the educational context, where the reliability of knowledge representations is critical for supporting meaningful learning. To address this, we introduce targeted optimizations across multiple pipeline components. The optimized pipeline achieves a 17.5% improvement in accuracy and a tenfold increase in processing efficiency. Our approach offers a holistic, scalable and end-to-end pipeline for automatic EduKG construction, adaptable to diverse educational contexts, and supports improved semantic representation of learning content. 

---
# Authorship Without Writing: Large Language Models and the Senior Author Analogy 

**Authors**: Clint Hurshman, Sebastian Porsdam Mann, Julian Savulescu, Brian D. Earp  

**Link**: [PDF](https://arxiv.org/pdf/2509.05390)  

**Abstract**: The use of large language models (LLMs) in bioethical, scientific, and medical writing remains controversial. While there is broad agreement in some circles that LLMs cannot count as authors, there is no consensus about whether and how humans using LLMs can count as authors. In many fields, authorship is distributed among large teams of researchers, some of whom, including paradigmatic senior authors who guide and determine the scope of a project and ultimately vouch for its integrity, may not write a single word. In this paper, we argue that LLM use (under specific conditions) is analogous to a form of senior authorship. On this view, the use of LLMs, even to generate complete drafts of research papers, can be considered a legitimate form of authorship according to the accepted criteria in many fields. We conclude that either such use should be recognized as legitimate, or current criteria for authorship require fundamental revision. AI use declaration: GPT-5 was used to help format Box 1. AI was not used for any other part of the preparation or writing of this manuscript. 

---
# Augmented Structure Preserving Neural Networks for cell biomechanics 

**Authors**: Juan Olalla-Pombo, Alberto Badías, Miguel Ángel Sanz-Gómez, José María Benítez, Francisco Javier Montáns  

**Link**: [PDF](https://arxiv.org/pdf/2509.05388)  

**Abstract**: Cell biomechanics involve a great number of complex phenomena that are fundamental to the evolution of life itself and other associated processes, ranging from the very early stages of embryo-genesis to the maintenance of damaged structures or the growth of tumors. Given the importance of such phenomena, increasing research has been dedicated to their understanding, but the many interactions between them and their influence on the decisions of cells as a collective network or cluster remain unclear. We present a new approach that combines Structure Preserving Neural Networks, which study cell movements as a purely mechanical system, with other Machine Learning tools (Artificial Neural Networks), which allow taking into consideration environmental factors that can be directly deduced from an experiment with Computer Vision techniques. This new model, tested on simulated and real cell migration cases, predicts complete cell trajectories following a roll-out policy with a high level of accuracy. This work also includes a mitosis event prediction model based on Neural Networks architectures which makes use of the same observed features. 

---
# A Lightweight Framework for Trigger-Guided LoRA-Based Self-Adaptation in LLMs 

**Authors**: Jiacheng Wei, Faguo Wu, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05385)  

**Abstract**: Large language models are unable to continuously adapt and learn from new data during reasoning at inference time. To address this limitation, we propose that complex reasoning tasks be decomposed into atomic subtasks and introduce SAGE, a trigger-guided dynamic fine-tuning framework that enables adaptive updates during reasoning at inference time. SAGE consists of three key components: (1) a Trigger module that detects reasoning failures through multiple evaluation metrics in real time; (2) a Trigger Buffer module that clusters anomaly samples using a streaming clustering process with HDBSCAN, followed by stability checks and similarity-based merging; and (3) a Lora Store module that dynamically optimizes parameter updates with an adapter pool for knowledge retention. Evaluation results show that SAGE demonstrates excellent accuracy, robustness, and stability on the atomic reasoning subtask through dynamic knowledge updating during test time. 

---
# User Privacy and Large Language Models: An Analysis of Frontier Developers' Privacy Policies 

**Authors**: Jennifer King, Kevin Klyman, Emily Capstick, Tiffany Saade, Victoria Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.05382)  

**Abstract**: Hundreds of millions of people now regularly interact with large language models via chatbots. Model developers are eager to acquire new sources of high-quality training data as they race to improve model capabilities and win market share. This paper analyzes the privacy policies of six U.S. frontier AI developers to understand how they use their users' chats to train models. Drawing primarily on the California Consumer Privacy Act, we develop a novel qualitative coding schema that we apply to each developer's relevant privacy policies to compare data collection and use practices across the six companies. We find that all six developers appear to employ their users' chat data to train and improve their models by default, and that some retain this data indefinitely. Developers may collect and train on personal information disclosed in chats, including sensitive information such as biometric and health data, as well as files uploaded by users. Four of the six companies we examined appear to include children's chat data for model training, as well as customer data from other products. On the whole, developers' privacy policies often lack essential information about their practices, highlighting the need for greater transparency and accountability. We address the implications of users' lack of consent for the use of their chat data for model training, data security issues arising from indefinite chat data retention, and training on children's chat data. We conclude by providing recommendations to policymakers and developers to address the data privacy challenges posed by LLM-powered chatbots. 

---
# Cumplimiento del Reglamento (UE) 2024/1689 en robótica y sistemas autónomos: una revisión sistemática de la literatura 

**Authors**: Yoana Pita Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2509.05380)  

**Abstract**: This systematic literature review analyzes the current state of compliance with Regulation (EU) 2024/1689 in autonomous robotic systems, focusing on cybersecurity frameworks and methodologies. Using the PRISMA protocol, 22 studies were selected from 243 initial records across IEEE Xplore, ACM DL, Scopus, and Web of Science. Findings reveal partial regulatory alignment: while progress has been made in risk management and encrypted communications, significant gaps persist in explainability modules, real-time human oversight, and knowledge base traceability. Only 40% of reviewed solutions explicitly address transparency requirements, and 30% implement failure intervention mechanisms. The study concludes that modular approaches integrating risk, supervision, and continuous auditing are essential to meet the AI Act mandates in autonomous robotics. 

---
# ThreatGPT: An Agentic AI Framework for Enhancing Public Safety through Threat Modeling 

**Authors**: Sharif Noor Zisad, Ragib Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2509.05379)  

**Abstract**: As our cities and communities become smarter, the systems that keep us safe, such as traffic control centers, emergency response networks, and public transportation, also become more complex. With this complexity comes a greater risk of security threats that can affect not just machines but real people's lives. To address this challenge, we present ThreatGPT, an agentic Artificial Intelligence (AI) assistant built to help people whether they are engineers, safety officers, or policy makers to understand and analyze threats in public safety systems. Instead of requiring deep cybersecurity expertise, it allows users to simply describe the components of a system they are concerned about, such as login systems, data storage, or communication networks. Then, with the click of a button, users can choose how they want the system to be analyzed by using popular frameworks such as STRIDE, MITRE ATT&CK, CVE reports, NIST, or CISA. ThreatGPT is unique because it does not just provide threat information, but rather it acts like a knowledgeable partner. Using few-shot learning, the AI learns from examples and generates relevant smart threat models. It can highlight what might go wrong, how attackers could take advantage, and what can be done to prevent harm. Whether securing a city's infrastructure or a local health service, this tool adapts to users' needs. In simple terms, ThreatGPT brings together AI and human judgment to make our public systems safer. It is designed not just to analyze threats, but to empower people to understand and act on them, faster, smarter, and with more confidence. 

---
# Privacy Preservation and Identity Tracing Prevention in AI-Driven Eye Tracking for Interactive Learning Environments 

**Authors**: Abdul Rehman, Are Dæhlen, Ilona Heldal, Jerry Chun-wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.05376)  

**Abstract**: Eye-tracking technology can aid in understanding neurodevelopmental disorders and tracing a person's identity. However, this technology poses a significant risk to privacy, as it captures sensitive information about individuals and increases the likelihood that data can be traced back to them. This paper proposes a human-centered framework designed to prevent identity backtracking while preserving the pedagogical benefits of AI-powered eye tracking in interactive learning environments. We explore how real-time data anonymization, ethical design principles, and regulatory compliance (such as GDPR) can be integrated to build trust and transparency. We first demonstrate the potential for backtracking student IDs and diagnoses in various scenarios using serious game-based eye-tracking data. We then provide a two-stage privacy-preserving framework that prevents participants from being tracked while still enabling diagnostic classification. The first phase covers four scenarios: I) Predicting disorder diagnoses based on different game levels. II) Predicting student IDs based on different game levels. III) Predicting student IDs based on randomized data. IV) Utilizing K-Means for out-of-sample data. In the second phase, we present a two-stage framework that preserves privacy. We also employ Federated Learning (FL) across multiple clients, incorporating a secure identity management system with dummy IDs and administrator-only access controls. In the first phase, the proposed framework achieved 99.3% accuracy for scenario 1, 63% accuracy for scenario 2, and 99.7% accuracy for scenario 3, successfully identifying and assigning a new student ID in scenario 4. In phase 2, we effectively prevented backtracking and established a secure identity management system with dummy IDs and administrator-only access controls, achieving an overall accuracy of 99.40%. 

---
# Long-Horizon Visual Imitation Learning via Plan and Code Reflection 

**Authors**: Quan Chen, Chenrui Shi, Qi Chen, Yuwei Wu, Zhi Gao, Xintong Zhang, Rui Gao, Kun Wu, Yunde Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.05368)  

**Abstract**: Learning from long-horizon demonstrations with complex action sequences presents significant challenges for visual imitation learning, particularly in understanding temporal relationships of actions and spatial relationships between objects. In this paper, we propose a new agent framework that incorporates two dedicated reflection modules to enhance both plan and code generation. The plan generation module produces an initial action sequence, which is then verified by the plan reflection module to ensure temporal coherence and spatial alignment with the demonstration video. The code generation module translates the plan into executable code, while the code reflection module verifies and refines the generated code to ensure correctness and consistency with the generated plan. These two reflection modules jointly enable the agent to detect and correct errors in both the plan generation and code generation, improving performance in tasks with intricate temporal and spatial dependencies. To support systematic evaluation, we introduce LongVILBench, a benchmark comprising 300 human demonstrations with action sequences of up to 18 steps. LongVILBench emphasizes temporal and spatial complexity across multiple task types. Experimental results demonstrate that existing methods perform poorly on this benchmark, whereas our new framework establishes a strong baseline for long-horizon visual imitation learning. 

---
# Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs 

**Authors**: Shei Pern Chua, Thai Zhen Leng, Teh Kai Jun, Xiao Li, Xiaolin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05367)  

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack. 

---
# Prototyping an AI-powered Tool for Energy Efficiency in New Zealand Homes 

**Authors**: Abdollah Baghaei Daemei  

**Link**: [PDF](https://arxiv.org/pdf/2509.05364)  

**Abstract**: Residential buildings contribute significantly to energy use, health outcomes, and carbon emissions. In New Zealand, housing quality has historically been poor, with inadequate insulation and inefficient heating contributing to widespread energy hardship. Recent reforms, including the Warmer Kiwi Homes program, Healthy Homes Standards, and H1 Building Code upgrades, have delivered health and comfort improvements, yet challenges persist. Many retrofits remain partial, data on household performance are limited, and decision-making support for homeowners is fragmented. This study presents the design and evaluation of an AI-powered decision-support tool for residential energy efficiency in New Zealand. The prototype, developed using Python and Streamlit, integrates data ingestion, anomaly detection, baseline modeling, and scenario simulation (e.g., LED retrofits, insulation upgrades) into a modular dashboard. Fifteen domain experts, including building scientists, consultants, and policy practitioners, tested the tool through semi-structured interviews. Results show strong usability (M = 4.3), high value of scenario outputs (M = 4.5), and positive perceptions of its potential to complement subsidy programs and regulatory frameworks. The tool demonstrates how AI can translate national policies into personalized, household-level guidance, bridging the gap between funding, standards, and practical decision-making. Its significance lies in offering a replicable framework for reducing energy hardship, improving health outcomes, and supporting climate goals. Future development should focus on carbon metrics, tariff modeling, integration with national datasets, and longitudinal trials to assess real-world adoption. 

---
# AI-in-the-Loop: Privacy Preserving Real-Time Scam Detection and Conversational Scambaiting by Leveraging LLMs and Federated Learning 

**Authors**: Ismail Hossain, Sai Puppala, Sajedul Talukder, Md Jahangir Alam  

**Link**: [PDF](https://arxiv.org/pdf/2509.05362)  

**Abstract**: Scams exploiting real-time social engineering -- such as phishing, impersonation, and phone fraud -- remain a persistent and evolving threat across digital platforms. Existing defenses are largely reactive, offering limited protection during active interactions. We propose a privacy-preserving, AI-in-the-loop framework that proactively detects and disrupts scam conversations in real time. The system combines instruction-tuned artificial intelligence with a safety-aware utility function that balances engagement with harm minimization, and employs federated learning to enable continual model updates without raw data sharing. Experimental evaluations show that the system produces fluent and engaging responses (perplexity as low as 22.3, engagement $\approx$0.80), while human studies confirm significant gains in realism, safety, and effectiveness over strong baselines. In federated settings, models trained with FedAvg sustain up to 30 rounds while preserving high engagement ($\approx$0.80), strong relevance ($\approx$0.74), and low PII leakage ($\leq$0.0085). Even with differential privacy, novelty and safety remain stable, indicating that robust privacy can be achieved without sacrificing performance. The evaluation of guard models (LlamaGuard, LlamaGuard2/3, MD-Judge) shows a straightforward pattern: stricter moderation settings reduce the chance of exposing personal information, but they also limit how much the model engages in conversation. In contrast, more relaxed settings allow longer and richer interactions, which improve scam detection, but at the cost of higher privacy risk. To our knowledge, this is the first framework to unify real-time scam-baiting, federated privacy preservation, and calibrated safety moderation into a proactive defense paradigm. 

---
# Governing AI R&D: A Legal Framework for Constraining Dangerous AI 

**Authors**: Alex Mark, Aaron Scher  

**Link**: [PDF](https://arxiv.org/pdf/2509.05361)  

**Abstract**: As AI advances, governing its development may become paramount to public safety. Lawmakers may seek to restrict the development and release of AI models or of AI research itself. These governance actions could trigger legal challenges that invalidate the actions, so lawmakers should consider these challenges ahead of time. We investigate three classes of potential litigation risk for AI regulation in the U.S.: the First Amendment, administrative law, and the Fourteenth Amendment. We discuss existing precedent that is likely to apply to AI, which legal challenges are likely to arise, and how lawmakers might preemptively address them. Effective AI regulation is possible, but it requires careful implementation to avoid these legal challenges. 

---
# An Empirical Analysis of Discrete Unit Representations in Speech Language Modeling Pre-training 

**Authors**: Yanis Labrak, Richard Dufour, Mickaël Rouvier  

**Link**: [PDF](https://arxiv.org/pdf/2509.05359)  

**Abstract**: This paper investigates discrete unit representations in Speech Language Models (SLMs), focusing on optimizing speech modeling during continual pre-training. In this paper, we systematically examine how model architecture, data representation, and training robustness influence the pre-training stage in which we adapt existing pre-trained language models to the speech modality. Our experiments highlight the role of speech encoders and clustering granularity across different model scales, showing how optimal discretization strategies vary with model capacity. By examining cluster distribution and phonemic alignments, we investigate the effective use of discrete vocabulary, uncovering both linguistic and paralinguistic patterns. Additionally, we explore the impact of clustering data selection on model robustness, highlighting the importance of domain matching between discretization training and target applications. 

---
# Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning 

**Authors**: Justus Huebotter, Pablo Lanillos, Marcel van Gerven, Serge Thill  

**Link**: [PDF](https://arxiv.org/pdf/2509.05356)  

**Abstract**: Despite recent progress in training spiking neural networks (SNNs) for classification, their application to continuous motor control remains limited. Here, we demonstrate that fully spiking architectures can be trained end-to-end to control robotic arms with multiple degrees of freedom in continuous environments. Our predictive-control framework combines Leaky Integrate-and-Fire dynamics with surrogate gradients, jointly optimizing a forward model for dynamics prediction and a policy network for goal-directed action. We evaluate this approach on both a planar 2D reaching task and a simulated 6-DOF Franka Emika Panda robot. Results show that SNNs can achieve stable training and accurate torque control, establishing their viability for high-dimensional motor tasks. An extensive ablation study highlights the role of initialization, learnable time constants, and regularization in shaping training dynamics. We conclude that while stable and effective control can be achieved, recurrent spiking networks remain highly sensitive to hyperparameter settings, underscoring the importance of principled design choices. 

---
# Unsupervised Instance Segmentation with Superpixels 

**Authors**: Cuong Manh Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05352)  

**Abstract**: Instance segmentation is essential for numerous computer vision applications, including robotics, human-computer interaction, and autonomous driving. Currently, popular models bring impressive performance in instance segmentation by training with a large number of human annotations, which are costly to collect. For this reason, we present a new framework that efficiently and effectively segments objects without the need for human annotations. Firstly, a MultiCut algorithm is applied to self-supervised features for coarse mask segmentation. Then, a mask filter is employed to obtain high-quality coarse masks. To train the segmentation network, we compute a novel superpixel-guided mask loss, comprising hard loss and soft loss, with high-quality coarse masks and superpixels segmented from low-level image features. Lastly, a self-training process with a new adaptive loss is proposed to improve the quality of predicted masks. We conduct experiments on public datasets in instance segmentation and object detection to demonstrate the effectiveness of the proposed framework. The results show that the proposed framework outperforms previous state-of-the-art methods. 

---
# Comparative Evaluation of Hard and Soft Clustering for Precise Brain Tumor Segmentation in MR Imaging 

**Authors**: Dibya Jyoti Bora, Mrinal Kanti Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.05340)  

**Abstract**: Segmentation of brain tumors from Magnetic Resonance Imaging (MRI) remains a pivotal challenge in medical image analysis due to the heterogeneous nature of tumor morphology and intensity distributions. Accurate delineation of tumor boundaries is critical for clinical decision-making, radiotherapy planning, and longitudinal disease monitoring. In this study, we perform a comprehensive comparative analysis of two major clustering paradigms applied in MRI tumor segmentation: hard clustering, exemplified by the K-Means algorithm, and soft clustering, represented by Fuzzy C-Means (FCM). While K-Means assigns each pixel strictly to a single cluster, FCM introduces partial memberships, meaning each pixel can belong to multiple clusters with varying degrees of association. Experimental validation was performed using the BraTS2020 dataset, incorporating pre-processing through Gaussian filtering and Contrast Limited Adaptive Histogram Equalization (CLAHE). Evaluation metrics included the Dice Similarity Coefficient (DSC) and processing time, which collectively demonstrated that K-Means achieved superior speed with an average runtime of 0.3s per image, whereas FCM attained higher segmentation accuracy with an average DSC of 0.67 compared to 0.43 for K-Means, albeit at a higher computational cost (1.3s per image). These results highlight the inherent trade-off between computational efficiency and boundary precision. 

---
# Plantbot: Integrating Plant and Robot through LLM Modular Agent Networks 

**Authors**: Atsushi Masumori, Norihiro Maruyama, Itsuki Doi, johnsmith, Hiroki Sato, Takashi Ikegami  

**Link**: [PDF](https://arxiv.org/pdf/2509.05338)  

**Abstract**: We introduce Plantbot, a hybrid lifeform that connects a living plant with a mobile robot through a network of large language model (LLM) modules. Each module - responsible for sensing, vision, dialogue, or action - operates asynchronously and communicates via natural language, enabling seamless interaction across biological and artificial domains. This architecture leverages the capacity of LLMs to serve as hybrid interfaces, where natural language functions as a universal protocol, translating multimodal data (soil moisture, temperature, visual context) into linguistic messages that coordinate system behaviors. The integrated network transforms plant states into robotic actions, installing normativity essential for agency within the sensor-motor loop. By combining biological and robotic elements through LLM-mediated communication, Plantbot behaves as an embodied, adaptive agent capable of responding autonomously to environmental conditions. This approach suggests possibilities for a new model of artificial life, where decentralized, LLM modules coordination enable novel interactions between biological and artificial systems. 

---
# RT-VLM: Re-Thinking Vision Language Model with 4-Clues for Real-World Object Recognition Robustness 

**Authors**: Junghyun Park, Tuan Anh Nguyen, Dugki Min  

**Link**: [PDF](https://arxiv.org/pdf/2509.05333)  

**Abstract**: Real world deployments often expose modern object recognition models to domain shifts that precipitate a severe drop in accuracy. Such shifts encompass (i) variations in low level image statistics, (ii) changes in object pose and viewpoint, (iii) partial occlusion, and (iv) visual confusion across adjacent classes. To mitigate this degradation, we introduce the Re-Thinking Vision Language Model (RT-VLM) framework. The foundation of this framework is a unique synthetic dataset generation pipeline that produces images annotated with "4-Clues": precise bounding boxes, class names, detailed object-level captions, and a comprehensive context-level caption for the entire scene. We then perform parameter efficient supervised tuning of Llama 3.2 11B Vision Instruct on this resource. At inference time, a two stage Re-Thinking scheme is executed: the model first emits its own four clues, then re examines these responses as evidence and iteratively corrects them. Across robustness benchmarks that isolate individual domain shifts, RT-VLM consistently surpasses strong baselines. These findings indicate that the integration of structured multimodal evidence with an explicit self critique loop constitutes a promising route toward reliable and transferable visual understanding. 

---
# Integrated Simulation Framework for Adversarial Attacks on Autonomous Vehicles 

**Authors**: Christos Anagnostopoulos, Ioulia Kapsali, Alexandros Gkillas, Nikos Piperigkos, Aris S. Lalos  

**Link**: [PDF](https://arxiv.org/pdf/2509.05332)  

**Abstract**: Autonomous vehicles (AVs) rely on complex perception and communication systems, making them vulnerable to adversarial attacks that can compromise safety. While simulation offers a scalable and safe environment for robustness testing, existing frameworks typically lack comprehensive supportfor modeling multi-domain adversarial scenarios. This paper introduces a novel, open-source integrated simulation framework designed to generate adversarial attacks targeting both perception and communication layers of AVs. The framework provides high-fidelity modeling of physical environments, traffic dynamics, and V2X networking, orchestrating these components through a unified core that synchronizes multiple simulators based on a single configuration file. Our implementation supports diverse perception-level attacks on LiDAR sensor data, along with communication-level threats such as V2X message manipulation and GPS spoofing. Furthermore, ROS 2 integration ensures seamless compatibility with third-party AV software stacks. We demonstrate the framework's effectiveness by evaluating the impact of generated adversarial scenarios on a state-of-the-art 3D object detector, revealing significant performance degradation under realistic conditions. 

---
# ForensicsData: A Digital Forensics Dataset for Large Language Models 

**Authors**: Youssef Chakir, Iyad Lahsen-Cherif  

**Link**: [PDF](https://arxiv.org/pdf/2509.05331)  

**Abstract**: The growing complexity of cyber incidents presents significant challenges for digital forensic investigators, especially in evidence collection and analysis. Public resources are still limited because of ethical, legal, and privacy concerns, even though realistic datasets are necessary to support research and tool developments. To address this gap, we introduce ForensicsData, an extensive Question-Context-Answer (Q-C-A) dataset sourced from actual malware analysis reports. It consists of more than 5,000 Q-C-A triplets. A unique workflow was used to create the dataset, which extracts structured data, uses large language models (LLMs) to transform it into Q-C-A format, and then uses a specialized evaluation process to confirm its quality. Among the models evaluated, Gemini 2 Flash demonstrated the best performance in aligning generated content with forensic terminology. ForensicsData aims to advance digital forensics by enabling reproducible experiments and fostering collaboration within the research community. 

---
# Optical Music Recognition of Jazz Lead Sheets 

**Authors**: Juan Carlos Martinez-Sevilla, Francesco Foscarin, Patricia Garcia-Iasci, David Rizo, Jorge Calvo-Zaragoza, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.05329)  

**Abstract**: In this paper, we address the challenge of Optical Music Recognition (OMR) for handwritten jazz lead sheets, a widely used musical score type that encodes melody and chords. The task is challenging due to the presence of chords, a score component not handled by existing OMR systems, and the high variability and quality issues associated with handwritten images. Our contribution is two-fold. We present a novel dataset consisting of 293 handwritten jazz lead sheets of 163 unique pieces, amounting to 2021 total staves aligned with Humdrum **kern and MusicXML ground truth scores. We also supply synthetic score images generated from the ground truth. The second contribution is the development of an OMR model for jazz lead sheets. We discuss specific tokenisation choices related to our kind of data, and the advantages of using synthetic scores and pretrained models. We publicly release all code, data, and models. 

---
# Zero-Knowledge Proofs in Sublinear Space 

**Authors**: Logan Nye  

**Link**: [PDF](https://arxiv.org/pdf/2509.05326)  

**Abstract**: Modern zero-knowledge proof (ZKP) systems, essential for privacy and verifiable computation, suffer from a fundamental limitation: the prover typically uses memory that scales linearly with the computation's trace length T, making them impractical for resource-constrained devices and prohibitively expensive for large-scale tasks. This paper overcomes this barrier by constructing, to our knowledge, the first sublinear-space ZKP prover. Our core contribution is an equivalence that reframes proof generation as an instance of the classic Tree Evaluation problem. Leveraging a recent space-efficient tree-evaluation algorithm, we design a streaming prover that assembles the proof without ever materializing the full execution trace. The approach reduces prover memory from linear in T to O(sqrt(T)) (up to O(log T) lower-order terms) while preserving proof size, verifier time, and the transcript/security guarantees of the underlying system. This enables a shift from specialized, server-bound proving to on-device proving, opening applications in decentralized systems, on-device machine learning, and privacy-preserving technologies. 

---
# A Dataset Generation Scheme Based on Video2EEG-SPGN-Diffusion for SEED-VD 

**Authors**: Yunfei Guo, Tao Zhang, Wu Huang, Yao Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.05321)  

**Abstract**: This paper introduces an open-source framework, Video2EEG-SPGN-Diffusion, that leverages the SEED-VD dataset to generate a multimodal dataset of EEG signals conditioned on video stimuli. Additionally, we disclose an engineering pipeline for aligning video and EEG data pairs, facilitating the training of multimodal large models with EEG alignment capabilities. Personalized EEG signals are generated using a self-play graph network (SPGN) integrated with a diffusion model. As a major contribution, we release a new dataset comprising over 1000 samples of SEED-VD video stimuli paired with generated 62-channel EEG signals at 200 Hz and emotion labels, enabling video-EEG alignment and advancing multimodal research. This framework offers novel tools for emotion analysis, data augmentation, and brain-computer interface applications, with substantial research and engineering significance. 

---
# Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models 

**Authors**: Zuquan Peng, Jianming Fu, Lixin Zou, Li Zheng, Yanzhen Ren, Guojun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.05318)  

**Abstract**: The use of unvetted third-party and internet data renders pre-trained models susceptible to backdoor attacks. Detecting backdoor samples is critical to prevent backdoor activation during inference or injection during training. However, existing detection methods often require the defender to have access to the poisoned models, extra clean samples, or significant computational resources to detect backdoor samples, limiting their practicality. To address this limitation, we propose a backdoor sample detection method based on perturbatio\textbf{N} discr\textbf{E}pancy consis\textbf{T}ency \textbf{E}valuation (\NETE). This is a novel detection method that can be used both pre-training and post-training phases. In the detection process, it only requires an off-the-shelf pre-trained model to compute the log probability of samples and an automated function based on a mask-filling strategy to generate perturbations. Our method is based on the interesting phenomenon that the change in perturbation discrepancy for backdoor samples is smaller than that for clean samples. Based on this phenomenon, we use curvature to measure the discrepancy in log probabilities between different perturbed samples and input samples, thereby evaluating the consistency of the perturbation discrepancy to determine whether the input sample is a backdoor sample. Experiments conducted on four typical backdoor attacks and five types of large language model backdoor attacks demonstrate that our detection strategy outperforms existing zero-shot black-box detection methods. 

---
# VILOD: A Visual Interactive Labeling Tool for Object Detection 

**Authors**: Isac Holm  

**Link**: [PDF](https://arxiv.org/pdf/2509.05317)  

**Abstract**: The advancement of Object Detection (OD) using Deep Learning (DL) is often hindered by the significant challenge of acquiring large, accurately labeled datasets, a process that is time-consuming and expensive. While techniques like Active Learning (AL) can reduce annotation effort by intelligently querying informative samples, they often lack transparency, limit the strategic insight of human experts, and may overlook informative samples not aligned with an employed query strategy. To mitigate these issues, Human-in-the-Loop (HITL) approaches integrating human intelligence and intuition throughout the machine learning life-cycle have gained traction. Leveraging Visual Analytics (VA), effective interfaces can be created to facilitate this human-AI collaboration. This thesis explores the intersection of these fields by developing and investigating "VILOD: A Visual Interactive Labeling tool for Object Detection". VILOD utilizes components such as a t-SNE projection of image features, together with uncertainty heatmaps and model state views. Enabling users to explore data, interpret model states, AL suggestions, and implement diverse sample selection strategies within an iterative HITL workflow for OD. An empirical investigation using comparative use cases demonstrated how VILOD, through its interactive visualizations, facilitates the implementation of distinct labeling strategies by making the model's state and dataset characteristics more interpretable (RQ1). The study showed that different visually-guided labeling strategies employed within VILOD result in competitive OD performance trajectories compared to an automated uncertainty sampling AL baseline (RQ2). This work contributes a novel tool and empirical insight into making the HITL-AL workflow for OD annotation more transparent, manageable, and potentially more effective. 

---
# Standard vs. Modular Sampling: Best Practices for Reliable LLM Unlearning 

**Authors**: Praveen Bushipaka, Lucia Passaro, Tommaso Cucinotta  

**Link**: [PDF](https://arxiv.org/pdf/2509.05316)  

**Abstract**: A conventional LLM Unlearning setting consists of two subsets -"forget" and "retain", with the objectives of removing the undesired knowledge from the forget set while preserving the remaining knowledge from the retain. In privacy-focused unlearning research, a retain set is often further divided into neighbor sets, containing either directly or indirectly connected to the forget targets; and augmented by a general-knowledge set. A common practice in existing benchmarks is to employ only a single neighbor set, with general knowledge which fails to reflect the real-world data complexities and relationships. LLM Unlearning typically involves 1:1 sampling or cyclic iteration sampling. However, the efficacy and stability of these de facto standards have not been critically examined. In this study, we systematically evaluate these common practices. Our findings reveal that relying on a single neighbor set is suboptimal and that a standard sampling approach can obscure performance trade-offs. Based on this analysis, we propose and validate an initial set of best practices: (1) Incorporation of diverse neighbor sets to balance forget efficacy and model utility, (2) Standard 1:1 sampling methods are inefficient and yield poor results, (3) Our proposed Modular Entity-Level Unlearning (MELU) strategy as an alternative to cyclic sampling. We demonstrate that this modular approach, combined with robust algorithms, provides a clear and stable path towards effective unlearning. 

---
# Evaluation of Large Language Models for Anomaly Detection in Autonomous Vehicles 

**Authors**: Petros Loukas, David Bassir, Savvas Chatzichristofis, Angelos Amanatiadis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05315)  

**Abstract**: The rapid evolution of large language models (LLMs) has pushed their boundaries to many applications in various domains. Recently, the research community has started to evaluate their potential adoption in autonomous vehicles and especially as complementary modules in the perception and planning software stacks. However, their evaluation is limited in synthetic datasets or manually driving datasets without the ground truth knowledge and more precisely, how the current perception and planning algorithms would perform in the cases under evaluation. For this reason, this work evaluates LLMs on real-world edge cases where current autonomous vehicles have been proven to fail. The proposed architecture consists of an open vocabulary object detector coupled with prompt engineering and large language model contextual reasoning. We evaluate several state-of-the-art models against real edge cases and provide qualitative comparison results along with a discussion on the findings for the potential application of LLMs as anomaly detectors in autonomous vehicles. 

---
# ManipDreamer3D : Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory 

**Authors**: Ying Li, Xiaobao Wei, Xiaowei Chi, Yuming Li, Zhongyu Zhao, Hao Wang, Ningning Ma, Ming Lu, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05314)  

**Abstract**: Data scarcity continues to be a major challenge in the field of robotic manipulation. Although diffusion models provide a promising solution for generating robotic manipulation videos, existing methods largely depend on 2D trajectories, which inherently face issues with 3D spatial ambiguity. In this work, we present a novel framework named ManipDreamer3D for generating plausible 3D-aware robotic manipulation videos from the input image and the text instruction. Our method combines 3D trajectory planning with a reconstructed 3D occupancy map created from a third-person perspective, along with a novel trajectory-to-video diffusion model. Specifically, ManipDreamer3D first reconstructs the 3D occupancy representation from the input image and then computes an optimized 3D end-effector trajectory, minimizing path length while avoiding collisions. Next, we employ a latent editing technique to create video sequences from the initial image latent and the optimized 3D trajectory. This process conditions our specially trained trajectory-to-video diffusion model to produce robotic pick-and-place videos. Our method generates robotic videos with autonomously planned plausible 3D trajectories, significantly reducing human intervention requirements. Experimental results demonstrate superior visual quality compared to existing methods. 

---
# Large Language Model Integration with Reinforcement Learning to Augment Decision-Making in Autonomous Cyber Operations 

**Authors**: Konur Tholl, François Rivest, Mariam El Mezouar, Ranwa Al Mallah  

**Link**: [PDF](https://arxiv.org/pdf/2509.05311)  

**Abstract**: Reinforcement Learning (RL) has shown great potential for autonomous decision-making in the cybersecurity domain, enabling agents to learn through direct environment interaction. However, RL agents in Autonomous Cyber Operations (ACO) typically learn from scratch, requiring them to execute undesirable actions to learn their consequences. In this study, we integrate external knowledge in the form of a Large Language Model (LLM) pretrained on cybersecurity data that our RL agent can directly leverage to make informed decisions. By guiding initial training with an LLM, we improve baseline performance and reduce the need for exploratory actions with obviously negative outcomes. We evaluate our LLM-integrated approach in a simulated cybersecurity environment, and demonstrate that our guided agent achieves over 2x higher rewards during early training and converges to a favorable policy approximately 4,500 episodes faster than the baseline. 

---
# ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders 

**Authors**: Xiangyu Liu, Haodi Lei, Yi Liu, Yang Liu, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05309)  

**Abstract**: Sparse Autoencoder (SAE) has emerged as a powerful tool for mechanistic interpretability of large language models. Recent works apply SAE to protein language models (PLMs), aiming to extract and analyze biologically meaningful features from their latent spaces. However, SAE suffers from semantic entanglement, where individual neurons often mix multiple nonlinear concepts, making it difficult to reliably interpret or manipulate model behaviors. In this paper, we propose a semantically-guided SAE, called ProtSAE. Unlike existing SAE which requires annotation datasets to filter and interpret activations, we guide semantic disentanglement during training using both annotation datasets and domain knowledge to mitigate the effects of entangled attributes. We design interpretability experiments showing that ProtSAE learns more biologically relevant and interpretable hidden features compared to previous methods. Performance analyses further demonstrate that ProtSAE maintains high reconstruction fidelity while achieving better results in interpretable probing. We also show the potential of ProtSAE in steering PLMs for downstream generation tasks. 

---
# Towards Log Analysis with AI Agents: Cowrie Case Study 

**Authors**: Enis Karaarslan, Esin Güler, Efe Emir Yüce, Cagatay Coban  

**Link**: [PDF](https://arxiv.org/pdf/2509.05306)  

**Abstract**: The scarcity of real-world attack data significantly hinders progress in cybersecurity research and education. Although honeypots like Cowrie effectively collect live threat intelligence, they generate overwhelming volumes of unstructured and heterogeneous logs, rendering manual analysis impractical. As a first step in our project on secure and efficient AI automation, this study explores the use of AI agents for automated log analysis. We present a lightweight and automated approach to process Cowrie honeypot logs. Our approach leverages AI agents to intelligently parse, summarize, and extract insights from raw data, while also considering the security implications of deploying such an autonomous system. Preliminary results demonstrate the pipeline's effectiveness in reducing manual effort and identifying attack patterns, paving the way for more advanced autonomous cybersecurity analysis in future work. 

---
# Multi-IaC-Eval: Benchmarking Cloud Infrastructure as Code Across Multiple Formats 

**Authors**: Sam Davidson, Li Sun, Bhavana Bhasker, Laurent Callot, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2509.05303)  

**Abstract**: Infrastructure as Code (IaC) is fundamental to modern cloud computing, enabling teams to define and manage infrastructure through machine-readable configuration files. However, different cloud service providers utilize diverse IaC formats. The lack of a standardized format requires cloud architects to be proficient in multiple IaC languages, adding complexity to cloud deployment. While Large Language Models (LLMs) show promise in automating IaC creation and maintenance, progress has been limited by the lack of comprehensive benchmarks across multiple IaC formats. We present Multi-IaC-Bench, a novel benchmark dataset for evaluating LLM-based IaC generation and mutation across AWS CloudFormation, Terraform, and Cloud Development Kit (CDK) formats. The dataset consists of triplets containing initial IaC templates, natural language modification requests, and corresponding updated templates, created through a synthetic data generation pipeline with rigorous validation. We evaluate several state-of-the-art LLMs on Multi-IaC-Bench, demonstrating that while modern LLMs can achieve high success rates (>95%) in generating syntactically valid IaC across formats, significant challenges remain in semantic alignment and handling complex infrastructure patterns. Our ablation studies highlight the importance of prompt engineering and retry mechanisms in successful IaC generation. We release Multi-IaC-Bench to facilitate further research in AI-assisted infrastructure management and establish standardized evaluation metrics for this crucial domain. 

---
# Sesame: Opening the door to protein pockets 

**Authors**: Raúl Miñán, Carles Perez-Lopez, Javier Iglesias, Álvaro Ciudad, Alexis Molina  

**Link**: [PDF](https://arxiv.org/pdf/2509.05302)  

**Abstract**: Molecular docking is a cornerstone of drug discovery, relying on high-resolution ligand-bound structures to achieve accurate predictions. However, obtaining these structures is often costly and time-intensive, limiting their availability. In contrast, ligand-free structures are more accessible but suffer from reduced docking performance due to pocket geometries being less suited for ligand accommodation in apo structures. Traditional methods for artificially inducing these conformations, such as molecular dynamics simulations, are computationally expensive. In this work, we introduce Sesame, a generative model designed to predict this conformational change efficiently. By generating geometries better suited for ligand accommodation at a fraction of the computational cost, Sesame aims to provide a scalable solution for improving virtual screening workflows. 

---
# Livia: An Emotion-Aware AR Companion Powered by Modular AI Agents and Progressive Memory Compression 

**Authors**: Rui Xi, Xianghan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05298)  

**Abstract**: Loneliness and social isolation pose significant emotional and health challenges, prompting the development of technology-based solutions for companionship and emotional support. This paper introduces Livia, an emotion-aware augmented reality (AR) companion app designed to provide personalized emotional support by combining modular artificial intelligence (AI) agents, multimodal affective computing, progressive memory compression, and AR driven embodied interaction. Livia employs a modular AI architecture with specialized agents responsible for emotion analysis, dialogue generation, memory management, and behavioral orchestration, ensuring robust and adaptive interactions. Two novel algorithms-Temporal Binary Compression (TBC) and Dynamic Importance Memory Filter (DIMF)-effectively manage and prioritize long-term memory, significantly reducing storage requirements while retaining critical context. Our multimodal emotion detection approach achieves high accuracy, enhancing proactive and empathetic engagement. User evaluations demonstrated increased emotional bonds, improved satisfaction, and statistically significant reductions in loneliness. Users particularly valued Livia's adaptive personality evolution and realistic AR embodiment. Future research directions include expanding gesture and tactile interactions, supporting multi-user experiences, and exploring customized hardware implementations. 

---
# Nonnegative matrix factorization and the principle of the common cause 

**Authors**: E. Khalafyan, A. E. Allahverdyan, A. Hovhannisyan  

**Link**: [PDF](https://arxiv.org/pdf/2509.03652)  

**Abstract**: Nonnegative matrix factorization (NMF) is a known unsupervised data-reduction method. The principle of the common cause (PCC) is a basic methodological approach in probabilistic causality, which seeks an independent mixture model for the joint probability of two dependent random variables. It turns out that these two concepts are closely related. This relationship is explored reciprocally for several datasets of gray-scale images, which are conveniently mapped into probability models. On one hand, PCC provides a predictability tool that leads to a robust estimation of the effective rank of NMF. Unlike other estimates (e.g., those based on the Bayesian Information Criteria), our estimate of the rank is stable against weak noise. We show that NMF implemented around this rank produces features (basis images) that are also stable against noise and against seeds of local optimization, thereby effectively resolving the NMF nonidentifiability problem. On the other hand, NMF provides an interesting possibility of implementing PCC in an approximate way, where larger and positively correlated joint probabilities tend to be explained better via the independent mixture model. We work out a clustering method, where data points with the same common cause are grouped into the same cluster. We also show how NMF can be employed for data denoising. 

---
# LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba 

**Authors**: Yinuo Wang, Gavin Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11849)  

**Abstract**: We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget. 

---
# AdCare-VLM: Leveraging Large Vision Language Model (LVLM) to Monitor Long-Term Medication Adherence and Care 

**Authors**: Md Asaduzzaman Jabin, Hanqi Jiang, Yiwei Li, Patrick Kaggwa, Eugene Douglass, Juliet N. Sekandi, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00275)  

**Abstract**: Chronic diseases, including diabetes, hypertension, asthma, HIV-AIDS, epilepsy, and tuberculosis, necessitate rigorous adherence to medication to avert disease progression, manage symptoms, and decrease mortality rates. Adherence is frequently undermined by factors including patient behavior, caregiver support, elevated medical costs, and insufficient healthcare infrastructure. We propose AdCare-VLM, a specialized Video-LLaVA-based multimodal large vision language model (LVLM) aimed at visual question answering (VQA) concerning medication adherence through patient videos. We employ a private dataset comprising 806 custom-annotated tuberculosis (TB) medication monitoring videos, which have been labeled by clinical experts, to fine-tune the model for adherence pattern detection. We present LLM-TB-VQA, a detailed medical adherence VQA dataset that encompasses positive, negative, and ambiguous adherence cases. Our method identifies correlations between visual features, such as the clear visibility of the patient's face, medication, water intake, and the act of ingestion, and their associated medical concepts in captions. This facilitates the integration of aligned visual-linguistic representations and improves multimodal interactions. Experimental results indicate that our method surpasses parameter-efficient fine-tuning (PEFT) enabled VLM models, such as LLaVA-V1.5 and Chat-UniVi, with absolute improvements ranging from 3.1% to 3.54% across pre-trained, regular, and low-rank adaptation (LoRA) configurations. Comprehensive ablation studies and attention map visualizations substantiate our approach, enhancing interpretability. 

---
