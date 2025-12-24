# LongVideoAgent: Multi-Agent Reasoning with Long Videos 

**Authors**: Runtao Liu, Ziyi Liu, Jiaqi Tang, Yue Ma, Renjie Pi, Jipeng Zhang, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.20618)  

**Abstract**: Recent advances in multimodal LLMs and systems that use tools for long-video QA point to the promise of reasoning over hour-long episodes. However, many methods still compress content into lossy summaries or rely on limited toolsets, weakening temporal grounding and missing fine-grained cues. We propose a multi-agent framework in which a master LLM coordinates a grounding agent to localize question-relevant segments and a vision agent to extract targeted textual observations. The master agent plans with a step limit, and is trained with reinforcement learning to encourage concise, correct, and efficient multi-agent cooperation. This design helps the master agent focus on relevant clips via grounding, complements subtitles with visual detail, and yields interpretable trajectories. On our proposed LongTVQA and LongTVQA+ which are episode-level datasets aggregated from TVQA/TVQA+, our multi-agent system significantly outperforms strong non-agent baselines. Experiments also show reinforcement learning further strengthens reasoning and planning for the trained agent. Code and data will be shared at this https URL. 

---
# Automated stereotactic radiosurgery planning using a human-in-the-loop reasoning large language model agent 

**Authors**: Humza Nusrat, Luke Francisco, Bing Luo, Hassan Bagher-Ebadian, Joshua Kim, Karen Chin-Snyder, Salim Siddiqui, Mira Shah, Eric Mellon, Mohammad Ghassemi, Anthony Doemer, Benjamin Movsas, Kundan Thind  

**Link**: [PDF](https://arxiv.org/pdf/2512.20586)  

**Abstract**: Stereotactic radiosurgery (SRS) demands precise dose shaping around critical structures, yet black-box AI systems have limited clinical adoption due to opacity concerns. We tested whether chain-of-thought reasoning improves agentic planning in a retrospective cohort of 41 patients with brain metastases treated with 18 Gy single-fraction SRS. We developed SAGE (Secure Agent for Generative Dose Expertise), an LLM-based planning agent for automated SRS treatment planning. Two variants generated plans for each case: one using a non-reasoning model, one using a reasoning model. The reasoning variant showed comparable plan dosimetry relative to human planners on primary endpoints (PTV coverage, maximum dose, conformity index, gradient index; all p > 0.21) while reducing cochlear dose below human baselines (p = 0.022). When prompted to improve conformity, the reasoning model demonstrated systematic planning behaviors including prospective constraint verification (457 instances) and trade-off deliberation (609 instances), while the standard model exhibited none of these deliberative processes (0 and 7 instances, respectively). Content analysis revealed that constraint verification and causal explanation concentrated in the reasoning agent. The optimization traces serve as auditable logs, offering a path toward transparent automated planning. 

---
# Advancing Multimodal Teacher Sentiment Analysis:The Large-Scale T-MED Dataset & The Effective AAM-TSA Model 

**Authors**: Zhiyi Duan, Xiangren Wang, Hongyu Yuan, Qianli Xing  

**Link**: [PDF](https://arxiv.org/pdf/2512.20548)  

**Abstract**: Teachers' emotional states are critical in educational scenarios, profoundly impacting teaching efficacy, student engagement, and learning achievements. However, existing studies often fail to accurately capture teachers' emotions due to the performative nature and overlook the critical impact of instructional information on emotional this http URL this paper, we systematically investigate teacher sentiment analysis by building both the dataset and the model accordingly. We construct the first large-scale teacher multimodal sentiment analysis dataset, this http URL ensure labeling accuracy and efficiency, we employ a human-machine collaborative labeling this http URL T-MED dataset includes 14,938 instances of teacher emotional data from 250 real classrooms across 11 subjects ranging from K-12 to higher education, integrating multimodal text, audio, video, and instructional this http URL, we propose a novel asymmetric attention-based multimodal teacher sentiment analysis model, this http URL-TSA introduces an asymmetric attention mechanism and hierarchical gating unit to enable differentiated cross-modal feature fusion and precise emotional classification. Experimental results demonstrate that AAM-TSA significantly outperforms existing state-of-the-art methods in terms of accuracy and interpretability on the T-MED dataset. 

---
# Benchmarking LLMs for Predictive Applications in the Intensive Care Units 

**Authors**: Chehak Malhotra, Mehak Gopal, Akshaya Devadiga, Pradeep Singh, Ridam Pal, Ritwik Kashyap, Tavpritesh Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2512.20520)  

**Abstract**: With the advent of LLMs, various tasks across the natural language processing domain have been transformed. However, their application in predictive tasks remains less researched. This study compares large language models, including GatorTron-Base (trained on clinical data), Llama 8B, and Mistral 7B, against models like BioBERT, DocBERT, BioClinicalBERT, Word2Vec, and Doc2Vec, setting benchmarks for predicting Shock in critically ill patients. Timely prediction of shock can enable early interventions, thus improving patient outcomes. Text data from 17,294 ICU stays of patients in the MIMIC III database were scored for length of stay > 24 hours and shock index (SI) > 0.7 to yield 355 and 87 patients with normal and abnormal SI-index, respectively. Both focal and cross-entropy losses were used during finetuning to address class imbalances. Our findings indicate that while GatorTron Base achieved the highest weighted recall of 80.5%, the overall performance metrics were comparable between SLMs and LLMs. This suggests that LLMs are not inherently superior to SLMs in predicting future clinical events despite their strong performance on text-based tasks. To achieve meaningful clinical outcomes, future efforts in training LLMs should prioritize developing models capable of predicting clinical trajectories rather than focusing on simpler tasks such as named entity recognition or phenotyping. 

---
# Bohrium + SciMaster: Building the Infrastructure and Ecosystem for Agentic Science at Scale 

**Authors**: Linfeng Zhang, Siheng Chen, Yuzhu Cai, Jingyi Chai, Junhan Chang, Kun Chen, Zhi X. Chen, Zhaohan Ding, Yuwen Du, Yuanpeng Gao, Yuan Gao, Jing Gao, Zhifeng Gao, Qiangqiang Gu, Yanhui Hong, Yuan Huang, Xi Fang, Xiaohong Ji, Guolin Ke, Zixing Lei, Xinyu Li, Yongge Li, Ruoxue Liao, Hang Lin, Xiaolu Lin, Yuxiang Liu, Xinzijian Liu, Zexi Liu, Jintan Lu, Tingjia Miao, Haohui Que, Weijie Sun, Yanfeng Wang, Bingyang Wu, Tianju Xue, Rui Ye, Jinzhe Zeng, Duo Zhang, Jiahui Zhang, Linfeng Zhang, Tianhan Zhang, Wenchang Zhang, Yuzhi Zhang, Zezhong Zhang, Hang Zheng, Hui Zhou, Tong Zhu, Xinyu Zhu, Qingguo Zhou, Weinan E  

**Link**: [PDF](https://arxiv.org/pdf/2512.20469)  

**Abstract**: AI agents are emerging as a practical way to run multi-step scientific workflows that interleave reasoning with tool use and verification, pointing to a shift from isolated AI-assisted steps toward \emph{agentic science at scale}. This shift is increasingly feasible, as scientific tools and models can be invoked through stable interfaces and verified with recorded execution traces, and increasingly necessary, as AI accelerates scientific output and stresses the peer-review and publication pipeline, raising the bar for traceability and credible evaluation.
However, scaling agentic science remains difficult: workflows are hard to observe and reproduce; many tools and laboratory systems are not agent-ready; execution is hard to trace and govern; and prototype AI Scientist systems are often bespoke, limiting reuse and systematic improvement from real workflow signals.
We argue that scaling agentic science requires an infrastructure-and-ecosystem approach, instantiated in Bohrium+SciMaster. Bohrium acts as a managed, traceable hub for AI4S assets -- akin to a HuggingFace of AI for Science -- that turns diverse scientific data, software, compute, and laboratory systems into agent-ready capabilities. SciMaster orchestrates these capabilities into long-horizon scientific workflows, on which scientific agents can be composed and executed. Between infrastructure and orchestration, a \emph{scientific intelligence substrate} organizes reusable models, knowledge, and components into executable building blocks for workflow reasoning and action, enabling composition, auditability, and improvement through use.
We demonstrate this stack with eleven representative master agents in real workflows, achieving orders-of-magnitude reductions in end-to-end scientific cycle time and generating execution-grounded signals from real workloads at multi-million scale. 

---
# Generative Digital Twins: Vision-Language Simulation Models for Executable Industrial Systems 

**Authors**: YuChe Hsu, AnJui Wang, TsaiChing Ni, YuanFu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20387)  

**Abstract**: We propose a Vision-Language Simulation Model (VLSM) that unifies visual and textual understanding to synthesize executable FlexScript from layout sketches and natural-language prompts, enabling cross-modal reasoning for industrial simulation systems. To support this new paradigm, the study constructs the first large-scale dataset for generative digital twins, comprising over 120,000 prompt-sketch-code triplets that enable multimodal learning between textual descriptions, spatial structures, and simulation logic. In parallel, three novel evaluation metrics, Structural Validity Rate (SVR), Parameter Match Rate (PMR), and Execution Success Rate (ESR), are proposed specifically for this task to comprehensively evaluate structural integrity, parameter fidelity, and simulator executability. Through systematic ablation across vision encoders, connectors, and code-pretrained language backbones, the proposed models achieve near-perfect structural accuracy and high execution robustness. This work establishes a foundation for generative digital twins that integrate visual reasoning and language understanding into executable industrial simulation systems. 

---
# A DeepSeek-Powered AI System for Automated Chest Radiograph Interpretation in Clinical Practice 

**Authors**: Yaowei Bai, Ruiheng Zhang, Yu Lei, Xuhua Duan, Jingfeng Yao, Shuguang Ju, Chaoyang Wang, Wei Yao, Yiwan Guo, Guilin Zhang, Chao Wan, Qian Yuan, Lei Chen, Wenjuan Tang, Biqiang Zhu, Xinggang Wang, Tao Sun, Wei Zhou, Dacheng Tao, Yongchao Xu, Chuansheng Zheng, Huangxuan Zhao, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.20344)  

**Abstract**: A global shortage of radiologists has been exacerbated by the significant volume of chest X-ray workloads, particularly in primary care. Although multimodal large language models show promise, existing evaluations predominantly rely on automated metrics or retrospective analyses, lacking rigorous prospective clinical validation. Janus-Pro-CXR (1B), a chest X-ray interpretation system based on DeepSeek Janus-Pro model, was developed and rigorously validated through a multicenter prospective trial (NCT07117266). Our system outperforms state-of-the-art X-ray report generation models in automated report generation, surpassing even larger-scale models including ChatGPT 4o (200B parameters), while demonstrating reliable detection of six clinically critical radiographic findings. Retrospective evaluation confirms significantly higher report accuracy than Janus-Pro and ChatGPT 4o. In prospective clinical deployment, AI assistance significantly improved report quality scores, reduced interpretation time by 18.3% (P < 0.001), and was preferred by a majority of experts in 54.3% of cases. Through lightweight architecture and domain-specific optimization, Janus-Pro-CXR improves diagnostic reliability and workflow efficiency, particularly in resource-constrained settings. The model architecture and implementation framework will be open-sourced to facilitate the clinical translation of AI-assisted radiology solutions. 

---
# SynCraft: Guiding Large Language Models to Predict Edit Sequences for Molecular Synthesizability Optimization 

**Authors**: Junren Li, Luhua Lai  

**Link**: [PDF](https://arxiv.org/pdf/2512.20333)  

**Abstract**: Generative artificial intelligence has revolutionized the exploration of chemical space, yet a critical bottleneck remains that a substantial fraction of generated molecules is synthetically inaccessible. Current solutions, such as post-hoc filtering or projection-based methods, often compromise structural novelty or disrupt key pharmacophores by forcing molecules into pre-defined synthetic templates. Herein, we introduce SynCraft, a reasoning-based framework that reframes synthesizability optimization not as a sequence translation task, but as a precise structural editing problem. Leveraging the emergent reasoning capabilities of Large Language Models, SynCraft navigates the "synthesis cliff" where minimal structural modifications yield significant gains in synthetic feasibility. By predicting executable sequences of atom-level edits rather than generating SMILES strings directly, SynCraft circumvents the syntactic fragility of LLMs while harnessing their chemical intuition. Extensive benchmarks demonstrate that SynCraft outperforms state-of-the-art baselines in generating synthesizable analogs with high structural fidelity. Furthermore, through interaction-aware prompting, SynCraft successfully replicates expert medicinal chemistry intuition in editing PLK1 inhibitors and rescuing high-scoring but previously discarded RIPK1 candidates in previous molecular generation literatures. 

---
# Synthesizing Procedural Memory: Challenges and Architectures in Automated Workflow Generation 

**Authors**: Nishant Gaurav, Adit Akarsh, Ankit Ranjan, Manoj Bajaj  

**Link**: [PDF](https://arxiv.org/pdf/2512.20278)  

**Abstract**: While CodeMem establishes executable code as the optimal representation for agentic procedural memory, the mechanism for autonomously synthesizing this memory from a blank slate remains underexplored. This paper operationalizes the transition of Large Language Models from passive tool-users to active workflow architects. Through a high-fidelity case study of a cross-service orchestration task involving Outlook and OneDrive, we identify and address four structural bottlenecks in automated skill generation: the Discovery Gap involving navigation of large tool registries, the Verification Gap regarding grounding tool response structures, the Decomposition Gap which replaces inefficient search with Linear State Anchoring, and the Scaling Gap focused on concurrency and persistence. We demonstrate that by enforcing a scientific methodology of hypothesize, probe, and code, agents can autonomously write robust, production-grade code skills. 

---
# ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge 

**Authors**: Yuntao Dai, Hang Gu, Teng Wang, Qianyu Cheng, Yifei Zheng, Zhiyong Qiu, Lei Gong, Wenqi Lou, Xuehai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.20276)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a unified paradigm for robotic perception and control, enabling emergent generalization and long-horizon task execution. However, their deployment in dynamic, real-world environments is severely hin dered by high inference latency. While smooth robotic interaction requires control frequencies of 20 to 30 Hz, current VLA models typi cally operate at only 3-5 Hz on edge devices due to the memory bound nature of autoregressive decoding. Existing optimizations often require extensive retraining or compromise model accuracy. To bridge this gap, we introduce ActionFlow, a system-level inference framework tailored for resource-constrained edge plat forms. At the core of ActionFlow is a Cross-Request Pipelin ing strategy, a novel scheduler that redefines VLA inference as a macro-pipeline of micro-requests. The strategy intelligently batches memory-bound Decode phases with compute-bound Prefill phases across continuous time steps to maximize hardware utilization. Furthermore, to support this scheduling, we propose a Cross Request State Packed Forward operator and a Unified KV Ring Buffer, which fuse fragmented memory operations into efficient dense computations. Experimental results demonstrate that ActionFlow achieves a 2.55x improvement in FPS on the OpenVLA-7B model without retraining, enabling real-time dy namic manipulation on edge hardware. Our work is available at this https URL. 

---
# Graph-Symbolic Policy Enforcement and Control (G-SPEC): A Neuro-Symbolic Framework for Safe Agentic AI in 5G Autonomous Networks 

**Authors**: Divya Vijay, Vignesh Ethiraj  

**Link**: [PDF](https://arxiv.org/pdf/2512.20275)  

**Abstract**: As networks evolve toward 5G Standalone and 6G, operators face orchestration challenges that exceed the limits of static automation and Deep Reinforcement Learning. Although Large Language Model (LLM) agents offer a path toward intent-based networking, they introduce stochastic risks, including topology hallucinations and policy non-compliance. To mitigate this, we propose Graph-Symbolic Policy Enforcement and Control (G-SPEC), a neuro-symbolic framework that constrains probabilistic planning with deterministic verification. The architecture relies on a Governance Triad - a telecom-adapted agent (TSLAM-4B), a Network Knowledge Graph (NKG), and SHACL constraints. We evaluated G-SPEC on a simulated 450-node 5G Core, achieving zero safety violations and a 94.1% remediation success rate, significantly outperforming the 82.4% baseline. Ablation analysis indicates that NKG validation drives the majority of safety gains (68%), followed by SHACL policies (24%). Scalability tests on topologies ranging from 10K to 100K nodes demonstrate that validation latency scales as $O(k^{1.2})$ where $k$ is subgraph size. With a processing overhead of 142ms, G-SPEC is viable for SMO-layer operations. 

---
# MemR$^3$: Memory Retrieval via Reflective Reasoning for LLM Agents 

**Authors**: Xingbo Du, Loka Li, Duzhen Zhang, Le Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.20237)  

**Abstract**: Memory systems have been designed to leverage past experiences in Large Language Model (LLM) agents. However, many deployed memory systems primarily optimize compression and storage, with comparatively less emphasis on explicit, closed-loop control of memory retrieval. From this observation, we build memory retrieval as an autonomous, accurate, and compatible agent system, named MemR$^3$, which has two core mechanisms: 1) a router that selects among retrieve, reflect, and answer actions to optimize answer quality; 2) a global evidence-gap tracker that explicitly renders the answering process transparent and tracks the evidence collection process. This design departs from the standard retrieve-then-answer pipeline by introducing a closed-loop control mechanism that enables autonomous decision-making. Empirical results on the LoCoMo benchmark demonstrate that MemR$^3$ surpasses strong baselines on LLM-as-a-Judge score, and particularly, it improves existing retrievers across four categories with an overall improvement on RAG (+7.29%) and Zep (+1.94%) using GPT-4.1-mini backend, offering a plug-and-play controller for existing memory stores. 

---
# TongSIM: A General Platform for Simulating Intelligent Machines 

**Authors**: Zhe Sun, Kunlun Wu, Chuanjian Fu, Zeming Song, Langyong Shi, Zihe Xue, Bohan Jing, Ying Yang, Xiaomeng Gao, Aijia Li, Tianyu Guo, Huiying Li, Xueyuan Yang, Rongkai Liu, Xinyi He, Yuxi Wang, Yue Li, Mingyuan Liu, Yujie Lu, Hongzhao Xie, Shiyun Zhao, Bo Dai, Wei Wang, Tao Yuan, Song-Chun Zhu, Yujia Peng, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20206)  

**Abstract**: As artificial intelligence (AI) rapidly advances, especially in multimodal large language models (MLLMs), research focus is shifting from single-modality text processing to the more complex domains of multimodal and embodied AI. Embodied intelligence focuses on training agents within realistic simulated environments, leveraging physical interaction and action feedback rather than conventionally labeled datasets. Yet, most existing simulation platforms remain narrowly designed, each tailored to specific tasks. A versatile, general-purpose training environment that can support everything from low-level embodied navigation to high-level composite activities, such as multi-agent social simulation and human-AI collaboration, remains largely unavailable. To bridge this gap, we introduce TongSIM, a high-fidelity, general-purpose platform for training and evaluating embodied agents. TongSIM offers practical advantages by providing over 100 diverse, multi-room indoor scenarios as well as an open-ended, interaction-rich outdoor town simulation, ensuring broad applicability across research needs. Its comprehensive evaluation framework and benchmarks enable precise assessment of agent capabilities, such as perception, cognition, decision-making, human-robot cooperation, and spatial and social reasoning. With features like customized scenes, task-adaptive fidelity, diverse agent types, and dynamic environmental simulation, TongSIM delivers flexibility and scalability for researchers, serving as a unified platform that accelerates training, evaluation, and advancement toward general embodied intelligence. 

---
# Offline Safe Policy Optimization From Heterogeneous Feedback 

**Authors**: Ze Gong, Pradeep Varakantham, Akshat Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.20173)  

**Abstract**: Offline Preference-based Reinforcement Learning (PbRL) learns rewards and policies aligned with human preferences without the need for extensive reward engineering and direct interaction with human annotators. However, ensuring safety remains a critical challenge across many domains and tasks. Previous works on safe RL from human feedback (RLHF) first learn reward and cost models from offline data, then use constrained RL to optimize a safe policy. While such an approach works in the contextual bandits settings (LLMs), in long horizon continuous control tasks, errors in rewards and costs accumulate, leading to impairment in performance when used with constrained RL methods. To address these challenges, (a) instead of indirectly learning policies (from rewards and costs), we introduce a framework that learns a policy directly based on pairwise preferences regarding the agent's behavior in terms of rewards, as well as binary labels indicating the safety of trajectory segments; (b) we propose \textsc{PreSa} (Preference and Safety Alignment), a method that combines preference learning module with safety alignment in a constrained optimization problem. This optimization problem is solved within a Lagrangian paradigm that directly learns reward-maximizing safe policy \textit{without explicitly learning reward and cost models}, avoiding the need for constrained RL; (c) we evaluate our approach on continuous control tasks with both synthetic and real human feedback. Empirically, our method successfully learns safe policies with high rewards, outperforming state-of-the-art baselines, and offline safe RL approaches with ground-truth reward and cost. 

---
# Concept Generalization in Humans and Large Language Models: Insights from the Number Game 

**Authors**: Arghavan Bazigaran, Hansem Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2512.20162)  

**Abstract**: We compare human and large language model (LLM) generalization in the number game, a concept inference task. Using a Bayesian model as an analytical framework, we examined the inductive biases and inference strategies of humans and LLMs. The Bayesian model captured human behavior better than LLMs in that humans flexibly infer rule-based and similarity-based concepts, whereas LLMs rely more on mathematical rules. Humans also demonstrated a few-shot generalization, even from a single example, while LLMs required more samples to generalize. These contrasts highlight the fundamental differences in how humans and LLMs infer and generalize mathematical concepts. 

---
# A Bidirectional Gated Recurrent Unit Model for PUE Prediction in Data Centers 

**Authors**: Dhivya Dharshini Kannan, Anupam Trivedi, Dipti Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2512.20161)  

**Abstract**: Data centers account for significant global energy consumption and a carbon footprint. The recent increasing demand for edge computing and AI advancements drives the growth of data center storage capacity. Energy efficiency is a cost-effective way to combat climate change, cut energy costs, improve business competitiveness, and promote IT and environmental sustainability. Thus, optimizing data center energy management is the most important factor in the sustainability of the world. Power Usage Effectiveness (PUE) is used to represent the operational efficiency of the data center. Predicting PUE using Neural Networks provides an understanding of the effect of each feature on energy consumption, thus enabling targeted modifications of those key features to improve energy efficiency. In this paper, we have developed Bidirectional Gated Recurrent Unit (BiGRU) based PUE prediction model and compared the model performance with GRU. The data set comprises 52,560 samples with 117 features using EnergyPlus, simulating a DC in Singapore. Sets of the most relevant features are selected using the Recursive Feature Elimination with Cross-Validation (RFECV) algorithm for different parameter settings. These feature sets are used to find the optimal hyperparameter configuration and train the BiGRU model. The performance of the optimized BiGRU-based PUE prediction model is then compared with that of GRU using mean squared error (MSE), mean absolute error (MAE), and R-squared metrics. 

---
# Enhancing Zero-Shot Time Series Forecasting in Off-the-Shelf LLMs via Noise Injection 

**Authors**: Xingyou Yin, Ceyao Zhang, Min Hu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.20140)  

**Abstract**: Large Language Models (LLMs) have demonstrated effectiveness as zero-shot time series (TS) forecasters. The key challenge lies in tokenizing TS data into textual representations that align with LLMs' pre-trained knowledge. While existing work often relies on fine-tuning specialized modules to bridge this gap, a distinct, yet challenging, paradigm aims to leverage truly off-the-shelf LLMs without any fine-tuning whatsoever, relying solely on strategic tokenization of numerical sequences. The performance of these fully frozen models is acutely sensitive to the textual representation of the input data, as their parameters cannot adapt to distribution shifts. In this paper, we introduce a simple yet highly effective strategy to overcome this brittleness: injecting noise into the raw time series before tokenization. This non-invasive intervention acts as a form of inference-time augmentation, compelling the frozen LLM to extrapolate based on robust underlying temporal patterns rather than superficial numerical artifacts. We theoretically analyze this phenomenon and empirically validate its effectiveness across diverse benchmarks. Notably, to fully eliminate potential biases from data contamination during LLM pre-training, we introduce two novel TS datasets that fall outside all utilized LLMs' pre-training scopes, and consistently observe improved performance. This study provides a further step in directly leveraging off-the-shelf LLMs for time series forecasting. 

---
# MolAct: An Agentic RL Framework for Molecular Editing and Property Optimization 

**Authors**: Zhuo Yang, Yeyun chen, Jiaqing Xie, Ben Gao, Shuaike Shen, Wanhao Liu, Liujia Yang, Beilun Wang, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.20135)  

**Abstract**: Molecular editing and optimization are multi-step problems that require iteratively improving properties while keeping molecules chemically valid and structurally similar. We frame both tasks as sequential, tool-guided decisions and introduce MolAct, an agentic reinforcement learning framework that employs a two-stage training paradigm: first building editing capability, then optimizing properties while reusing the learned editing behaviors. To the best of our knowledge, this is the first work to formalize molecular design as an Agentic Reinforcement Learning problem, where an LLM agent learns to interleave reasoning, tool-use, and molecular optimization. The framework enables agents to interact in multiple turns, invoking chemical tools for validity checking, property assessment, and similarity control, and leverages their feedback to refine subsequent edits. We instantiate the MolAct framework to train two model families: MolEditAgent for molecular editing tasks and MolOptAgent for molecular optimization tasks. In molecular editing, MolEditAgent-7B delivers 100, 95, and 98 valid add, delete, and substitute edits, outperforming strong closed "thinking" baselines such as DeepSeek-R1; MolEditAgent-3B approaches the performance of much larger open "thinking" models like Qwen3-32B-think. In molecular optimization, MolOptAgent-7B (trained on MolEditAgent-7B) surpasses the best closed "thinking" baseline (e.g., Claude 3.7) on LogP and remains competitive on solubility, while maintaining balanced performance across other objectives. These results highlight that treating molecular design as a multi-step, tool-augmented process is key to reliable and interpretable improvements. 

---
# Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches 

**Authors**: Chaithra, Kamesh Kadimisetty, Biju R Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2512.20082)  

**Abstract**: Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling. 

---
# Reason2Decide: Rationale-Driven Multi-Task Learning 

**Authors**: H M Quamran Hasan, Housam Khalifa Bashier, Jiayi Dai, Mi-Young Kim, Randy Goebel  

**Link**: [PDF](https://arxiv.org/pdf/2512.20074)  

**Abstract**: Despite the wide adoption of Large Language Models (LLM)s, clinical decision support systems face a critical challenge: achieving high predictive accuracy while generating explanations aligned with the predictions. Current approaches suffer from exposure bias leading to misaligned explanations. We propose Reason2Decide, a two-stage training framework that addresses key challenges in self-rationalization, including exposure bias and task separation. In Stage-1, our model is trained on rationale generation, while in Stage-2, we jointly train on label prediction and rationale generation, applying scheduled sampling to gradually transition from conditioning on gold labels to model predictions. We evaluate Reason2Decide on three medical datasets, including a proprietary triage dataset and public biomedical QA datasets. Across model sizes, Reason2Decide outperforms other fine-tuning baselines and some zero-shot LLMs in prediction (F1) and rationale fidelity (BERTScore, BLEU, LLM-as-a-Judge). In triage, Reason2Decide is rationale source-robust across LLM-generated, nurse-authored, and nurse-post-processed rationales. In our experiments, while using only LLM-generated rationales in Stage-1, Reason2Decide outperforms other fine-tuning variants. This indicates that LLM-generated rationales are suitable for pretraining models, reducing reliance on human annotations. Remarkably, Reason2Decide achieves these gains with models 40x smaller than contemporary foundation models, making clinical reasoning more accessible for resource-constrained deployments while still providing explainable decision support. 

---
# Scaling Reinforcement Learning for Content Moderation with Large Language Models 

**Authors**: Hamed Firooz, Rui Liu, Yuchen Lu, Zhenyu Hou, Fangzhou Xiong, Xiaoyang Zhang, Changshu Jian, Zhicheng Zhu, Jiayuan Ma, Jacob Tao, Chaitali Gupta, Xiaochang Peng, Shike Mei, Hang Cui, Yang Qin, Shuo Tang, Jason Gaedtke, Arpit Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2512.20061)  

**Abstract**: Content moderation at scale remains one of the most pressing challenges in today's digital ecosystem, where billions of user- and AI-generated artifacts must be continuously evaluated for policy violations. Although recent advances in large language models (LLMs) have demonstrated strong potential for policy-grounded moderation, the practical challenges of training these systems to achieve expert-level accuracy in real-world settings remain largely unexplored, particularly in regimes characterized by label sparsity, evolving policy definitions, and the need for nuanced reasoning beyond shallow pattern matching. In this work, we present a comprehensive empirical investigation of scaling reinforcement learning (RL) for content classification, systematically evaluating multiple RL training recipes and reward-shaping strategies-including verifiable rewards and LLM-as-judge frameworks-to transform general-purpose language models into specialized, policy-aligned classifiers across three real-world content moderation tasks. Our findings provide actionable insights for industrial-scale moderation systems, demonstrating that RL exhibits sigmoid-like scaling behavior in which performance improves smoothly with increased training data, rollouts, and optimization steps before gradually saturating. Moreover, we show that RL substantially improves performance on tasks requiring complex policy-grounded reasoning while achieving up to 100x higher data efficiency than supervised fine-tuning, making it particularly effective in domains where expert annotations are scarce or costly. 

---
# Towards Generative Location Awareness for Disaster Response: A Probabilistic Cross-view Geolocalization Approach 

**Authors**: Hao Li, Fabian Deuser, Wenping Yin, Steffen Knoblauch, Wufan Zhao, Filip Biljecki, Yong Xue, Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20056)  

**Abstract**: As Earth's climate changes, it is impacting disasters and extreme weather events across the planet. Record-breaking heat waves, drenching rainfalls, extreme wildfires, and widespread flooding during hurricanes are all becoming more frequent and more intense. Rapid and efficient response to disaster events is essential for climate resilience and sustainability. A key challenge in disaster response is to accurately and quickly identify disaster locations to support decision-making and resources allocation. In this paper, we propose a Probabilistic Cross-view Geolocalization approach, called ProbGLC, exploring new pathways towards generative location awareness for rapid disaster response. Herein, we combine probabilistic and deterministic geolocalization models into a unified framework to simultaneously enhance model explainability (via uncertainty quantification) and achieve state-of-the-art geolocalization performance. Designed for rapid diaster response, the ProbGLC is able to address cross-view geolocalization across multiple disaster events as well as to offer unique features of probabilistic distribution and localizability score. To evaluate the ProbGLC, we conduct extensive experiments on two cross-view disaster datasets (i.e., MultiIAN and SAGAINDisaster), consisting diverse cross-view imagery pairs of multiple disaster types (e.g., hurricanes, wildfires, floods, to tornadoes). Preliminary results confirms the superior geolocalization accuracy (i.e., 0.86 in Acc@1km and 0.97 in Acc@25km) and model explainability (i.e., via probabilistic distributions and localizability scores) of the proposed ProbGLC approach, highlighting the great potential of leveraging generative cross-view approach to facilitate location awareness for better and faster disaster response. The data and code is publicly available at this https URL 

---
# Learning Skills from Action-Free Videos 

**Authors**: Hung-Chieh Fang, Kuo-Han Hung, Chu-Rong Chen, Po-Jung Chou, Chun-Kai Yang, Po-Chen Ko, Yu-Chiang Wang, Yueh-Hua Wu, Min-Hung Chen, Shao-Hua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.20052)  

**Abstract**: Learning from videos offers a promising path toward generalist robots by providing rich visual and temporal priors beyond what real robot datasets contain. While existing video generative models produce impressive visual predictions, they are difficult to translate into low-level actions. Conversely, latent-action models better align videos with actions, but they typically operate at the single-step level and lack high-level planning capabilities. We bridge this gap by introducing Skill Abstraction from Optical Flow (SOF), a framework that learns latent skills from large collections of action-free videos. Our key idea is to learn a latent skill space through an intermediate representation based on optical flow that captures motion information aligned with both video dynamics and robot actions. By learning skills in this flow-based latent space, SOF enables high-level planning over video-derived skills and allows for easier translation of these skills into actions. Experiments show that our approach consistently improves performance in both multitask and long-horizon settings, demonstrating the ability to acquire and compose skills directly from raw visual data. 

---
# Discovering Lie Groups with Flow Matching 

**Authors**: Jung Yeon Park, Yuxuan Chen, Floor Eijkelboom, Jan-Willem van de Meent, Lawson L.S. Wong, Robin Walters  

**Link**: [PDF](https://arxiv.org/pdf/2512.20043)  

**Abstract**: Symmetry is fundamental to understanding physical systems, and at the same time, can improve performance and sample efficiency in machine learning. Both pursuits require knowledge of the underlying symmetries in data. To address this, we propose learning symmetries directly from data via flow matching on Lie groups. We formulate symmetry discovery as learning a distribution over a larger hypothesis group, such that the learned distribution matches the symmetries observed in data. Relative to previous works, our method, \lieflow, is more flexible in terms of the types of groups it can discover and requires fewer assumptions. Experiments on 2D and 3D point clouds demonstrate the successful discovery of discrete groups, including reflections by flow matching over the complex domain. We identify a key challenge where the symmetric arrangement of the target modes causes ``last-minute convergence,'' where samples remain stationary until relatively late in the flow, and introduce a novel interpolation scheme for flow matching for symmetry discovery. 

---
# S$^3$IT: A Benchmark for Spatially Situated Social Intelligence Test 

**Authors**: Zhe Sun, Xueyuan Yang, Yujie Lu, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19992)  

**Abstract**: The integration of embodied agents into human environments demands embodied social intelligence: reasoning over both social norms and physical constraints. However, existing evaluations fail to address this integration, as they are limited to either disembodied social reasoning (e.g., in text) or socially-agnostic physical tasks. Both approaches fail to assess an agent's ability to integrate and trade off both physical and social constraints within a realistic, embodied context. To address this challenge, we introduce Spatially Situated Social Intelligence Test (S$^{3}$IT), a benchmark specifically designed to evaluate embodied social intelligence. It is centered on a novel and challenging seat-ordering task, requiring an agent to arrange seating in a 3D environment for a group of large language model-driven (LLM-driven) NPCs with diverse identities, preferences, and intricate interpersonal relationships. Our procedurally extensible framework generates a vast and diverse scenario space with controllable difficulty, compelling the agent to acquire preferences through active dialogue, perceive the environment via autonomous exploration, and perform multi-objective optimization within a complex constraint network. We evaluate state-of-the-art LLMs on S$^{3}$IT and found that they still struggle with this problem, showing an obvious gap compared with the human baseline. Results imply that LLMs have deficiencies in spatial intelligence, yet simultaneously demonstrate their ability to achieve near human-level competence in resolving conflicts that possess explicit textual cues. 

---
# FGDCC: Fine-Grained Deep Cluster Categorization -- A Framework for Intra-Class Variability Problems in Plant Classification 

**Authors**: Luciano Araujo Dourado Filho, Rodrigo Tripodi Calumby  

**Link**: [PDF](https://arxiv.org/pdf/2512.19960)  

**Abstract**: Intra-class variability is given according to the significance in the degree of dissimilarity between images within a class. In that sense, depending on its intensity, intra-class variability can hinder the learning process for DL models, specially when such classes are also underrepresented, which is a very common scenario in Fine-Grained Visual Categorization (FGVC) tasks. This paper proposes a novel method that aims at leveraging classification performance in FGVC tasks by learning fine-grained features via classification of class-wise cluster assignments. Our goal is to apply clustering over each class individually, which can allow to discover pseudo-labels that encodes a latent degree of similarity between images. In turn, those labels can be employed in a hierarchical classification process that allows to learn more fine-grained visual features and thereby mitigating intra-class variability issues. Initial experiments over the PlantNet300k enabled to shed light upon several key points in which future work will have to be developed in order to find more conclusive evidence regarding the effectiveness of our method. Our method still achieves state-of-the-art performance on the PlantNet300k dataset even though some of its components haven't been shown to be fully optimized. Our code is available at \href{this https URL}{this https URL}. 

---
# Zero-Shot Segmentation through Prototype-Guidance for Multi-Label Plant Species Identification 

**Authors**: Luciano Araujo Dourado Filho, Almir Moreira da Silva Neto, Rodrigo Pereira David, Rodrigo Tripodi Calumby  

**Link**: [PDF](https://arxiv.org/pdf/2512.19957)  

**Abstract**: This paper presents an approach developed to address the PlantClef 2025 challenge, which consists of a fine-grained multi-label species identification, over high-resolution images. Our solution focused on employing class prototypes obtained from the training dataset as a proxy guidance for training a segmentation Vision Transformer (ViT) on the test set images. To obtain these representations, the proposed method extracts features from training dataset images and create clusters, by applying K-Means, with $K$ equals to the number of classes in the dataset. The segmentation model is a customized narrow ViT, built by replacing the patch embedding layer with a frozen DinoV2, pre-trained on the training dataset for individual species classification. This model is trained to reconstruct the class prototypes of the training dataset from the test dataset images. We then use this model to obtain attention scores that enable to identify and localize areas of interest and consequently guide the classification process. The proposed approach enabled a domain-adaptation from multi-class identification with individual species, into multi-label classification from high-resolution vegetation plots. Our method achieved fifth place in the PlantCLEF 2025 challenge on the private leaderboard, with an F1 score of 0.33331. Besides that, in absolute terms our method scored 0.03 lower than the top-performing submission, suggesting that it may achieved competitive performance in the benchmark task. Our code is available at \href{this https URL}{this https URL}. 

---
# Interpolative Decoding: Exploring the Spectrum of Personality Traits in LLMs 

**Authors**: Eric Yeh, John Cadigan, Ran Chen, Dick Crouch, Melinda Gervasio, Dayne Freitag  

**Link**: [PDF](https://arxiv.org/pdf/2512.19937)  

**Abstract**: Recent research has explored using very large language models (LLMs) as proxies for humans in tasks such as simulation, surveys, and studies. While LLMs do not possess a human psychology, they often can emulate human behaviors with sufficiently high fidelity to drive simulations to test human behavioral hypotheses, exhibiting more nuance and range than the rule-based agents often employed in behavioral economics. One key area of interest is the effect of personality on decision making, but the requirement that a prompt must be created for every tested personality profile introduces experimental overhead and degrades replicability. To address this issue, we leverage interpolative decoding, representing each dimension of personality as a pair of opposed prompts and employing an interpolation parameter to simulate behavior along the dimension. We show that interpolative decoding reliably modulates scores along each of the Big Five dimensions. We then show how interpolative decoding causes LLMs to mimic human decision-making behavior in economic games, replicating results from human psychological research. Finally, we present preliminary results of our efforts to ``twin'' individual human players in a collaborative game through systematic search for points in interpolation space that cause the system to replicate actions taken by the human subject. 

---
# A Branch-and-Price Algorithm for Fast and Equitable Last-Mile Relief Aid Distribution 

**Authors**: Mahdi Mostajabdaveh, F. Sibel Salman, Walter J. Gutjahr  

**Link**: [PDF](https://arxiv.org/pdf/2512.19882)  

**Abstract**: The distribution of relief supplies to shelters is a critical aspect of post-disaster humanitarian logistics. In major disasters, prepositioned supplies often fall short of meeting all demands. We address the problem of planning vehicle routes from a distribution center to shelters while allocating limited relief supplies. To balance efficiency and equity, we formulate a bi-objective problem: minimizing a Gini-index-based measure of inequity in unsatisfied demand for fair distribution and minimizing total travel time for timely delivery. We propose a Mixed Integer Programming (MIP) model and use the $\epsilon$-constraint method to handle the bi-objective nature. By deriving mathematical properties of the optimal solution, we introduce valid inequalities and design an algorithm for optimal delivery allocations given feasible vehicle routes. A branch-and-price (B&P) algorithm is developed to solve the problem efficiently. Computational tests on realistic datasets from a past earthquake in Van, Turkey, and predicted data for Istanbul's Kartal region show that the B&P algorithm significantly outperforms commercial MIP solvers. Our bi-objective approach reduces aid distribution inequity by 34% without compromising efficiency. Results indicate that when time constraints are very loose or tight, lexicographic optimization prioritizing demand coverage over fairness is effective. For moderately restrictive time constraints, a balanced approach is essential to avoid inequitable outcomes. 

---
# PhysMaster: Building an Autonomous AI Physicist for Theoretical and Computational Physics Research 

**Authors**: Tingjia Miao, Jiawen Dai, Jingkun Liu, Jinxin Tan, Muhua Zhang, Wenkai Jin, Yuwen Du, Tian Jin, Xianghe Pang, Zexi Liu, Tu Guo, Zhengliang Zhang, Yunjie Huang, Shuo Chen, Rui Ye, Yuzhi Zhang, Linfeng Zhang, Kun Chen, Wei Wang, Weinan E, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19799)  

**Abstract**: Advances in LLMs have produced agents with knowledge and operational capabilities comparable to human scientists, suggesting potential to assist, accelerate, and automate research. However, existing studies mainly evaluate such systems on well-defined benchmarks or general tasks like literature retrieval, limiting their end-to-end problem-solving ability in open scientific scenarios. This is particularly true in physics, which is abstract, mathematically intensive, and requires integrating analytical reasoning with code-based computation. To address this, we propose PhysMaster, an LLM-based agent functioning as an autonomous theoretical and computational physicist. PhysMaster couples absract reasoning with numerical computation and leverages LANDAU, the Layered Academic Data Universe, which preserves retrieved literature, curated prior knowledge, and validated methodological traces, enhancing decision reliability and stability. It also employs an adaptive exploration strategy balancing efficiency and open-ended exploration, enabling robust performance in ultra-long-horizon tasks. We evaluate PhysMaster on problems from high-energy theory, condensed matter theory to astrophysics, including: (i) acceleration, compressing labor-intensive research from months to hours; (ii) automation, autonomously executing hypothesis-driven loops ; and (iii) autonomous discovery, independently exploring open problems. 

---
# Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning 

**Authors**: Seijin Kobayashi, Yanick Schimpf, Maximilian Schlegel, Angelika Steger, Maciej Wolczyk, Johannes von Oswald, Nino Scherre, Kaitlin Maile, Guillaume Lajoie, Blake A. Richards, Rif A. Saurous, James Manyika, Blaise Agüera y Arcas, Alexander Meulemans, João Sacramento  

**Link**: [PDF](https://arxiv.org/pdf/2512.20605)  

**Abstract**: Large-scale autoregressive models pretrained on next-token prediction and finetuned with reinforcement learning (RL) have achieved unprecedented success on many problem domains. During RL, these models explore by generating new outputs, one token at a time. However, sampling actions token-by-token can result in highly inefficient learning, particularly when rewards are sparse. Here, we show that it is possible to overcome this problem by acting and exploring within the internal representations of an autoregressive model. Specifically, to discover temporally-abstract actions, we introduce a higher-order, non-causal sequence model whose outputs control the residual stream activations of a base autoregressive model. On grid world and MuJoCo-based tasks with hierarchical structure, we find that the higher-order model learns to compress long activation sequence chunks onto internal controllers. Critically, each controller executes a sequence of behaviorally meaningful actions that unfold over long timescales and are accompanied with a learned termination condition, such that composing multiple controllers over time leads to efficient exploration on novel tasks. We show that direct internal controller reinforcement, a process we term "internal RL", enables learning from sparse rewards in cases where standard RL finetuning fails. Our results demonstrate the benefits of latent action generation and reinforcement in autoregressive models, suggesting internal RL as a promising avenue for realizing hierarchical RL within foundation models. 

---
# Cube Bench: A Benchmark for Spatial Visual Reasoning in MLLMs 

**Authors**: Dhruv Anand, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2512.20595)  

**Abstract**: We introduce Cube Bench, a Rubik's-cube benchmark for evaluating spatial and sequential reasoning in multimodal large language models (MLLMs). The benchmark decomposes performance into five skills: (i) reconstructing cube faces from images and text, (ii) choosing the optimal next move, (iii) predicting the outcome of a candidate move without applying it, (iv) executing multi-step plans while recovering from mistakes, and (v) detecting and revising one's own errors. Using a shared set of scrambled cube states, identical prompts and parsers, and a single distance-to-solved metric, we compare recent MLLMs side by side as a function of scramble depth. Across seven MLLMs, accuracy drops sharply with depth; once a trajectory stalls or diverges, models rarely recover, and high face-reconstruction accuracy does not guarantee competent action selection or multi-step execution. A pronounced closed- vs open-source gap emerges: the strongest closed model leads on both single-step perception tasks and multi-step control tasks, while open-weight models cluster near chance on the hardest settings; yet even the best MLLM degrades at higher cube complexity. A simple self-correction via reflective thinking yields modest gains but can also introduce overthinking. Cube Bench offers a compact, reproducible probe of sequential spatial reasoning in MLLMs. 

---
# Leveraging High-Fidelity Digital Models and Reinforcement Learning for Mission Engineering: A Case Study of Aerial Firefighting Under Perfect Information 

**Authors**: İbrahim Oğuz Çetinkaya, Sajad Khodadadian, Taylan G. Topçu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20589)  

**Abstract**: As systems engineering (SE) objectives evolve from design and operation of monolithic systems to complex System of Systems (SoS), the discipline of Mission Engineering (ME) has emerged which is increasingly being accepted as a new line of thinking for the SE community. Moreover, mission environments are uncertain, dynamic, and mission outcomes are a direct function of how the mission assets will interact with this environment. This proves static architectures brittle and calls for analytically rigorous approaches for ME. To that end, this paper proposes an intelligent mission coordination methodology that integrates digital mission models with Reinforcement Learning (RL), that specifically addresses the need for adaptive task allocation and reconfiguration. More specifically, we are leveraging a Digital Engineering (DE) based infrastructure that is composed of a high-fidelity digital mission model and agent-based simulation; and then we formulate the mission tactics management problem as a Markov Decision Process (MDP), and employ an RL agent trained via Proximal Policy Optimization. By leveraging the simulation as a sandbox, we map the system states to actions, refining the policy based on realized mission outcomes. The utility of the RL-based intelligent mission coordinator is demonstrated through an aerial firefighting case study. Our findings indicate that the RL-based intelligent mission coordinator not only surpasses baseline performance but also significantly reduces the variability in mission performance. Thus, this study serves as a proof of concept demonstrating that DE-enabled mission simulations combined with advanced analytical tools offer a mission-agnostic framework for improving ME practice; which can be extended to more complicated fleet design and selection problems in the future from a mission-first perspective. 

---
# Performative Policy Gradient: Optimality in Performative Reinforcement Learning 

**Authors**: Debabrota Basu, Udvas Das, Brahim Driss, Uddalak Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2512.20576)  

**Abstract**: Post-deployment machine learning algorithms often influence the environments they act in, and thus shift the underlying dynamics that the standard reinforcement learning (RL) methods ignore. While designing optimal algorithms in this performative setting has recently been studied in supervised learning, the RL counterpart remains under-explored. In this paper, we prove the performative counterparts of the performance difference lemma and the policy gradient theorem in RL, and further introduce the Performative Policy Gradient algorithm (PePG). PePG is the first policy gradient algorithm designed to account for performativity in RL. Under softmax parametrisation, and also with and without entropy regularisation, we prove that PePG converges to performatively optimal policies, i.e. policies that remain optimal under the distribution shifts induced by themselves. Thus, PePG significantly extends the prior works in Performative RL that achieves performative stability but not optimality. Furthermore, our empirical analysis on standard performative RL environments validate that PePG outperforms standard policy gradient algorithms and the existing performative RL algorithms aiming for stability. 

---
# Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs 

**Authors**: Rui Pan, Zhuofu Chen, Ravi Netravali  

**Link**: [PDF](https://arxiv.org/pdf/2512.20573)  

**Abstract**: Diffusion Large Language Models (dLLMs) offer fast, parallel token generation, but their standalone use is plagued by an inherent efficiency-quality tradeoff. We show that, if carefully applied, the attributes of dLLMs can actually be a strength for drafters in speculative decoding with autoregressive (AR) verifiers. Our core insight is that dLLM's speed from parallel decoding drastically lowers the risk of costly rejections, providing a practical mechanism to effectively realize the (elusive) lengthy drafts that lead to large speedups with speculative decoding. We present FailFast, a dLLM-based speculative decoding framework that realizes this approach by dynamically adapting its speculation length. It "fails fast" by spending minimal compute in hard-to-speculate regions to shrink speculation latency and "wins big" by aggressively extending draft lengths in easier regions to reduce verification latency (in many cases, speculating and accepting 70 tokens at a time!). Without any fine-tuning, FailFast delivers lossless acceleration of AR LLMs and achieves up to 4.9$\times$ speedup over vanilla decoding, 1.7$\times$ over the best naive dLLM drafter, and 1.4$\times$ over EAGLE-3 across diverse models and workloads. We open-source FailFast at this https URL. 

---
# Distilling to Hybrid Attention Models via KL-Guided Layer Selection 

**Authors**: Yanhong Li, Songlin Yang, Shawn Tan, Mayank Mishra, Rameswar Panda, Jiawei Zhou, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.20569)  

**Abstract**: Distilling pretrained softmax attention Transformers into more efficient hybrid architectures that interleave softmax and linear attention layers is a promising approach for improving the inference efficiency of LLMs without requiring expensive pretraining from scratch. A critical factor in the conversion process is layer selection, i.e., deciding on which layers to convert to linear attention variants. This paper describes a simple and efficient recipe for layer selection that uses layer importance scores derived from a small amount of training on generic text data. Once the layers have been selected we use a recent pipeline for the distillation process itself \citep[RADLADS;][]{goldstein2025radlads}, which consists of attention weight transfer, hidden state alignment, KL-based distribution matching, followed by a small amount of finetuning. We find that this approach is more effective than existing approaches for layer selection, including heuristics that uniformly interleave linear attentions based on a fixed ratio, as well as more involved approaches that rely on specialized diagnostic datasets. 

---
# LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving 

**Authors**: Long Nguyen, Micha Fauth, Bernhard Jaeger, Daniel Dauner, Maximilian Igl, Andreas Geiger, Kashyap Chitta  

**Link**: [PDF](https://arxiv.org/pdf/2512.20563)  

**Abstract**: Simulators can generate virtually unlimited driving data, yet imitation learning policies in simulation still struggle to achieve robust closed-loop performance. Motivated by this gap, we empirically study how misalignment between privileged expert demonstrations and sensor-based student observations can limit the effectiveness of imitation learning. More precisely, experts have significantly higher visibility (e.g., ignoring occlusions) and far lower uncertainty (e.g., knowing other vehicles' actions), making them difficult to imitate reliably. Furthermore, navigational intent (i.e., the route to follow) is under-specified in student models at test time via only a single target point. We demonstrate that these asymmetries can measurably limit driving performance in CARLA and offer practical interventions to address them. After careful modifications to narrow the gaps between expert and student, our TransFuser v6 (TFv6) student policy achieves a new state of the art on all major publicly available CARLA closed-loop benchmarks, reaching 95 DS on Bench2Drive and more than doubling prior performances on Longest6~v2 and Town13. Additionally, by integrating perception supervision from our dataset into a shared sim-to-real pipeline, we show consistent gains on the NAVSIM and Waymo Vision-Based End-to-End driving benchmarks. Our code, data, and models are publicly available at this https URL. 

---
# SweRank+: Multilingual, Multi-Turn Code Ranking for Software Issue Localization 

**Authors**: Revanth Gangi Reddy, Ye Liu, Wenting Zhao, JaeHyeok Doo, Tarun Suresh, Daniel Lee, Caiming Xiong, Yingbo Zhou, Semih Yavuz, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2512.20482)  

**Abstract**: Maintaining large-scale, multilingual codebases hinges on accurately localizing issues, which requires mapping natural-language error descriptions to the relevant functions that need to be modified. However, existing ranking approaches are often Python-centric and perform a single-pass search over the codebase. This work introduces SweRank+, a framework that couples SweRankMulti, a cross-lingual code ranking tool, with SweRankAgent, an agentic search setup, for iterative, multi-turn reasoning over the code repository. SweRankMulti comprises a code embedding retriever and a listwise LLM reranker, and is trained using a carefully curated large-scale issue localization dataset spanning multiple popular programming languages. SweRankAgent adopts an agentic search loop that moves beyond single-shot localization with a memory buffer to reason and accumulate relevant localization candidates over multiple turns. Our experiments on issue localization benchmarks spanning various languages demonstrate new state-of-the-art performance with SweRankMulti, while SweRankAgent further improves localization over single-pass ranking. 

---
# Dual-Encoder Transformer-Based Multimodal Learning for Ischemic Stroke Lesion Segmentation Using Diffusion MRI 

**Authors**: Muhammad Usman, Azka Rehman, Muhammad Mutti Ur Rehman, Abd Ur Rehman, Muhammad Umar Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2512.20436)  

**Abstract**: Accurate segmentation of ischemic stroke lesions from diffusion magnetic resonance imaging (MRI) is essential for clinical decision-making and outcome assessment. Diffusion-Weighted Imaging (DWI) and Apparent Diffusion Coefficient (ADC) scans provide complementary information on acute and sub-acute ischemic changes; however, automated lesion delineation remains challenging due to variability in lesion appearance.
In this work, we study ischemic stroke lesion segmentation using multimodal diffusion MRI from the ISLES 2022 dataset. Several state-of-the-art convolutional and transformer-based architectures, including U-Net variants, Swin-UNet, and TransUNet, are benchmarked. Based on performance, a dual-encoder TransUNet architecture is proposed to learn modality-specific representations from DWI and ADC inputs. To incorporate spatial context, adjacent slice information is integrated using a three-slice input configuration.
All models are trained under a unified framework and evaluated using the Dice Similarity Coefficient (DSC). Results show that transformer-based models outperform convolutional baselines, and the proposed dual-encoder TransUNet achieves the best performance, reaching a Dice score of 85.4% on the test set. The proposed framework offers a robust solution for automated ischemic stroke lesion segmentation from diffusion MRI. 

---
# Evasion-Resilient Detection of DNS-over-HTTPS Data Exfiltration: A Practical Evaluation and Toolkit 

**Authors**: Adam Elaoumari  

**Link**: [PDF](https://arxiv.org/pdf/2512.20423)  

**Abstract**: The purpose of this project is to assess how well defenders can detect DNS-over-HTTPS (DoH) file exfiltration, and which evasion strategies can be used by attackers. While providing a reproducible toolkit to generate, intercept and analyze DoH exfiltration, and comparing Machine Learning vs threshold-based detection under adversarial scenarios. The originality of this project is the introduction of an end-to-end, containerized pipeline that generates configurable file exfiltration over DoH using several parameters (e.g., chunking, encoding, padding, resolver rotation). It allows for file reconstruction at the resolver side, while extracting flow-level features using a fork of DoHLyzer. The pipeline contains a prediction side, which allows the training of machine learning models based on public labelled datasets and then evaluates them side-by-side with threshold-based detection methods against malicious and evasive DNS-Over-HTTPS traffic. We train Random Forest, Gradient Boosting and Logistic Regression classifiers on a public DoH dataset and benchmark them against evasive DoH exfiltration scenarios. The toolkit orchestrates traffic generation, file capture, feature extraction, model training and analysis. The toolkit is then encapsulated into several Docker containers for easy setup and full reproducibility regardless of the platform it is run on. Future research regarding this project is directed at validating the results on mixed enterprise traffic, extending the protocol coverage to HTTP/3/QUIC request, adding a benign traffic generation, and working on real-time traffic evaluation. A key objective is to quantify when stealth constraints make DoH exfiltration uneconomical and unworthy for the attacker. 

---
# Simplifying Multi-Task Architectures Through Task-Specific Normalization 

**Authors**: Mihai Suteu, Ovidiu Serban  

**Link**: [PDF](https://arxiv.org/pdf/2512.20420)  

**Abstract**: Multi-task learning (MTL) aims to leverage shared knowledge across tasks to improve generalization and parameter efficiency, yet balancing resources and mitigating interference remain open challenges. Architectural solutions often introduce elaborate task-specific modules or routing schemes, increasing complexity and overhead. In this work, we show that normalization layers alone are sufficient to address many of these challenges. Simply replacing shared normalization with task-specific variants already yields competitive performance, questioning the need for complex designs. Building on this insight, we propose Task-Specific Sigmoid Batch Normalization (TS$\sigma$BN), a lightweight mechanism that enables tasks to softly allocate network capacity while fully sharing feature extractors. TS$\sigma$BN improves stability across CNNs and Transformers, matching or exceeding performance on NYUv2, Cityscapes, CelebA, and PascalContext, while remaining highly parameter-efficient. Moreover, its learned gates provide a natural framework for analyzing MTL dynamics, offering interpretable insights into capacity allocation, filter specialization, and task relationships. Our findings suggest that complex MTL architectures may be unnecessary and that task-specific normalization offers a simple, interpretable, and efficient alternative. 

---
# DETACH : Decomposed Spatio-Temporal Alignment for Exocentric Video and Ambient Sensors with Staged Learning 

**Authors**: Junho Yoon, Jaemo Jung, Hyunju Kim, Dongman Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.20409)  

**Abstract**: Aligning egocentric video with wearable sensors have shown promise for human action recognition, but face practical limitations in user discomfort, privacy concerns, and scalability. We explore exocentric video with ambient sensors as a non-intrusive, scalable alternative. While prior egocentric-wearable works predominantly adopt Global Alignment by encoding entire sequences into unified representations, this approach fails in exocentric-ambient settings due to two problems: (P1) inability to capture local details such as subtle motions, and (P2) over-reliance on modality-invariant temporal patterns, causing misalignment between actions sharing similar temporal patterns with different spatio-semantic contexts. To resolve these problems, we propose DETACH, a decomposed spatio-temporal framework. This explicit decomposition preserves local details, while our novel sensor-spatial features discovered via online clustering provide semantic grounding for context-aware alignment. To align the decomposed features, our two-stage approach establishes spatial correspondence through mutual supervision, then performs temporal alignment via a spatial-temporal weighted contrastive loss that adaptively handles easy negatives, hard negatives, and false negatives. Comprehensive experiments with downstream tasks on Opportunity++ and HWU-USP datasets demonstrate substantial improvements over adapted egocentric-wearable baselines. 

---
# AUDRON: A Deep Learning Framework with Fused Acoustic Signatures for Drone Type Recognition 

**Authors**: Rajdeep Chatterjee, Sudip Chakrabarty, Trishaani Acharjee, Deepanjali Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2512.20407)  

**Abstract**: Unmanned aerial vehicles (UAVs), commonly known as drones, are increasingly used across diverse domains, including logistics, agriculture, surveillance, and defense. While these systems provide numerous benefits, their misuse raises safety and security concerns, making effective detection mechanisms essential. Acoustic sensing offers a low-cost and non-intrusive alternative to vision or radar-based detection, as drone propellers generate distinctive sound patterns. This study introduces AUDRON (AUdio-based Drone Recognition Network), a hybrid deep learning framework for drone sound detection, employing a combination of Mel-Frequency Cepstral Coefficients (MFCC), Short-Time Fourier Transform (STFT) spectrograms processed with convolutional neural networks (CNNs), recurrent layers for temporal modeling, and autoencoder-based representations. Feature-level fusion integrates complementary information before classification. Experimental evaluation demonstrates that AUDRON effectively differentiates drone acoustic signatures from background noise, achieving high accuracy while maintaining generalizability across varying conditions. AUDRON achieves 98.51 percent and 97.11 percent accuracy in binary and multiclass classification. The results highlight the advantage of combining multiple feature representations with deep learning for reliable acoustic drone detection, suggesting the framework's potential for deployment in security and surveillance applications where visual or radar sensing may be limited. 

---
# Identifying Appropriately-Sized Services with Deep Reinforcement Learning 

**Authors**: Syeda Tasnim Fabiha, Saad Shafiq, Wesley Klewerton Guez Assunção, Nenad Medvidović  

**Link**: [PDF](https://arxiv.org/pdf/2512.20381)  

**Abstract**: Service-based architecture (SBA) has gained attention in industry and academia as a means to modernize legacy systems. It refers to a design style that enables systems to be developed as suites of small, loosely coupled, and autonomous components (services) that encapsulate functionality and communicate via language-agnostic APIs. However, defining appropriately sized services that capture cohesive subsets of system functionality remains challenging. Existing work often relies on the availability of documentation, access to project personnel, or a priori knowledge of the target number of services, assumptions that do not hold in many real-world scenarios. Our work addresses these limitations using a deep reinforcement learning-based approach to identify appropriately sized services directly from implementation artifacts. We present Rake, a reinforcement learning-based technique that leverages available system documentation and source code to guide service decomposition at the level of implementation methods. Rake does not require specific documentation or access to project personnel and is language-agnostic. It also supports a customizable objective function that balances modularization quality and business capability alignment, i.e., the degree to which a service covers the targeted business capability. We applied Rake to four open-source legacy projects and compared it with two state-of-the-art techniques. On average, Rake achieved 7-14 percent higher modularization quality and 18-22 percent stronger business capability alignment. Our results further show that optimizing solely for business context can degrade decomposition quality in tightly coupled systems, highlighting the need for balanced objectives. 

---
# Clust-PSI-PFL: A Population Stability Index Approach for Clustered Non-IID Personalized Federated Learning 

**Authors**: Daniel M. Jimenez-Gutierrez, Mehrdad Hassanzadeh, Aris Anagnostopoulos, Ioannis Chatzigiannakis, Andrea Vitaletti  

**Link**: [PDF](https://arxiv.org/pdf/2512.20363)  

**Abstract**: Federated learning (FL) supports privacy-preserving, decentralized machine learning (ML) model training by keeping data on client devices. However, non-independent and identically distributed (non-IID) data across clients biases updates and degrades performance. To alleviate these issues, we propose Clust-PSI-PFL, a clustering-based personalized FL framework that uses the Population Stability Index (PSI) to quantify the level of non-IID data. We compute a weighted PSI metric, $WPSI^L$, which we show to be more informative than common non-IID metrics (Hellinger, Jensen-Shannon, and Earth Mover's distance). Using PSI features, we form distributionally homogeneous groups of clients via K-means++; the number of optimal clusters is chosen by a systematic silhouette-based procedure, typically yielding few clusters with modest overhead. Across six datasets (tabular, image, and text modalities), two partition protocols (Dirichlet with parameter $\alpha$ and Similarity with parameter S), and multiple client sizes, Clust-PSI-PFL delivers up to 18% higher global accuracy than state-of-the-art baselines and markedly improves client fairness by a relative improvement of 37% under severe non-IID data. These results establish PSI-guided clustering as a principled, lightweight mechanism for robust PFL under label skew. 

---
# Multi-LLM Thematic Analysis with Dual Reliability Metrics: Combining Cohen's Kappa and Semantic Similarity for Qualitative Research Validation 

**Authors**: Nilesh Jain, Seyi Adeyinka, Leor Roseman, Aza Allsop  

**Link**: [PDF](https://arxiv.org/pdf/2512.20352)  

**Abstract**: Qualitative research faces a critical reliability challenge: traditional inter-rater agreement methods require multiple human coders, are time-intensive, and often yield moderate consistency. We present a multi-perspective validation framework for LLM-based thematic analysis that combines ensemble validation with dual reliability metrics: Cohen's Kappa ($\kappa$) for inter-rater agreement and cosine similarity for semantic consistency. Our framework enables configurable analysis parameters (1-6 seeds, temperature 0.0-2.0), supports custom prompt structures with variable substitution, and provides consensus theme extraction across any JSON format. As proof-of-concept, we evaluate three leading LLMs (Gemini 2.5 Pro, GPT-4o, Claude 3.5 Sonnet) on a psychedelic art therapy interview transcript, conducting six independent runs per model. Results demonstrate Gemini achieves highest reliability ($\kappa = 0.907$, cosine=95.3%), followed by GPT-4o ($\kappa = 0.853$, cosine=92.6%) and Claude ($\kappa = 0.842$, cosine=92.1%). All three models achieve a high agreement ($\kappa > 0.80$), validating the multi-run ensemble approach. The framework successfully extracts consensus themes across runs, with Gemini identifying 6 consensus themes (50-83% consistency), GPT-4o identifying 5 themes, and Claude 4 themes. Our open-source implementation provides researchers with transparent reliability metrics, flexible configuration, and structure-agnostic consensus extraction, establishing methodological foundations for reliable AI-assisted qualitative research. 

---
# Toward Explaining Large Language Models in Software Engineering Tasks 

**Authors**: Antonio Vitale, Khai-Nguyen Nguyen, Denys Poshyvanyk, Rocco Oliveto, Simone Scalabrino, Antonio Mastropaolo  

**Link**: [PDF](https://arxiv.org/pdf/2512.20328)  

**Abstract**: Recent progress in Large Language Models (LLMs) has substantially advanced the automation of software engineering (SE) tasks, enabling complex activities such as code generation and code summarization. However, the black-box nature of LLMs remains a major barrier to their adoption in high-stakes and safety-critical domains, where explainability and transparency are vital for trust, accountability, and effective human supervision. Despite increasing interest in explainable AI for software engineering, existing methods lack domain-specific explanations aligned with how practitioners reason about SE artifacts. To address this gap, we introduce FeatureSHAP, the first fully automated, model-agnostic explainability framework tailored to software engineering tasks. Based on Shapley values, FeatureSHAP attributes model outputs to high-level input features through systematic input perturbation and task-specific similarity comparisons, while remaining compatible with both open-source and proprietary LLMs. We evaluate FeatureSHAP on two bi-modal SE tasks: code generation and code summarization. The results show that FeatureSHAP assigns less importance to irrelevant input features and produces explanations with higher fidelity than baseline methods. A practitioner survey involving 37 participants shows that FeatureSHAP helps practitioners better interpret model outputs and make more informed decisions. Collectively, FeatureSHAP represents a meaningful step toward practical explainable AI in software engineering. FeatureSHAP is available at this https URL. 

---
# Deep Learning Classification of EEG Responses to Multi-Dimensional Transcranial Electrical Stimulation 

**Authors**: Alexis Pomares Pastor, Ines Ribeiro Violante, Gregory Scott  

**Link**: [PDF](https://arxiv.org/pdf/2512.20319)  

**Abstract**: A major shortcoming of medical practice is the lack of an objective measure of conscious level. Impairment of consciousness is common, e.g. following brain injury and seizures, which can also interfere with sensory processing and volitional responses. This is also an important pitfall in neurophysiological methods that infer awareness via command following, e.g. using functional MRI or electroencephalography (EEG).
Transcranial electrical stimulation (TES) can be employed to non-invasively stimulate the brain, bypassing sensory inputs, and has already showed promising results in providing reliable indicators of brain state. However, current non-invasive solutions have been limited to magnetic stimulation, which is not easily translatable to clinical settings. Our long-term vision is to develop an objective measure of brain state that can be used at the bedside, without requiring patients to understand commands or initiate motor responses.
In this study, we demonstrated the feasibility of a framework using Deep Learning algorithms to classify EEG brain responses evoked by a defined multi-dimensional pattern of TES. We collected EEG-TES data from 11 participants and found that delivering transcranial direct current stimulation (tDCS) to posterior cortical areas targeting the angular gyrus elicited an exceptionally reliable brain response. For this paradigm, our best Convolutional Neural Network model reached a 92% classification F1-score on Holdout data from participants never seen during training, significantly surpassing human-level performance at 60-70% accuracy.
These findings establish a framework for robust consciousness measurement for clinical use. In this spirit, we documented and open-sourced our datasets and codebase in full, to be used freely by the neuroscience and AI research communities, who may replicate our results with free tools like GitHub, Kaggle, and Colab. 

---
# TableGPT-R1: Advancing Tabular Reasoning Through Reinforcement Learning 

**Authors**: Saisai Yang, Qingyi Huang, Jing Yuan, Liangyu Zha, Kai Tang, Yuhang Yang, Ning Wang, Yucheng Wei, Liyao Li, Wentao Ye, Hao Chen, Tao Zhang, Junlin Zhou, Haobo Wang, Gang Chen, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.20312)  

**Abstract**: Tabular data serves as the backbone of modern data analysis and scientific research. While Large Language Models (LLMs) fine-tuned via Supervised Fine-Tuning (SFT) have significantly improved natural language interaction with such structured data, they often fall short in handling the complex, multi-step reasoning and robust code execution required for real-world table tasks. Reinforcement Learning (RL) offers a promising avenue to enhance these capabilities, yet its application in the tabular domain faces three critical hurdles: the scarcity of high-quality agentic trajectories with closed-loop code execution and environment feedback on diverse table structures, the extreme heterogeneity of feedback signals ranging from rigid SQL execution to open-ended data interpretation, and the risk of catastrophic forgetting of general knowledge during vertical specialization. To overcome these challenges and unlock advanced reasoning on complex tables, we introduce \textbf{TableGPT-R1}, a specialized tabular model built on a systematic RL framework. Our approach integrates a comprehensive data engineering pipeline that synthesizes difficulty-stratified agentic trajectories for both supervised alignment and RL rollouts, a task-adaptive reward system that combines rule-based verification with a criteria-injected reward model and incorporates process-level step reward shaping with behavioral regularization, and a multi-stage training framework that progressively stabilizes reasoning before specializing in table-specific tasks. Extensive evaluations demonstrate that TableGPT-R1 achieves state-of-the-art performance on authoritative benchmarks, significantly outperforming baseline models while retaining robust general capabilities. Our model is available at this https URL. 

---
# KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System 

**Authors**: Zhongyu Xia, Wenhao Chen, Yongtao Wang, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20299)  

**Abstract**: Visual-language reasoning, driving knowledge, and value alignment are essential for advanced autonomous driving systems. However, existing approaches largely rely on data-driven learning, making it difficult to capture the complex logic underlying decision-making through imitation or limited reinforcement rewards. To address this, we propose KnowVal, a new autonomous driving system that enables visual-language reasoning through the synergistic integration of open-world perception and knowledge retrieval. Specifically, we construct a comprehensive driving knowledge graph that encodes traffic laws, defensive driving principles, and ethical norms, complemented by an efficient LLM-based retrieval mechanism tailored for driving scenarios. Furthermore, we develop a human-preference dataset and train a Value Model to guide interpretable, value-aligned trajectory assessment. Experimental results show that our method substantially improves planning performance while remaining compatible with existing architectures. Notably, KnowVal achieves the lowest collision rate on nuScenes and state-of-the-art results on Bench2Drive. 

---
# Patterns vs. Patients: Evaluating LLMs against Mental Health Professionals on Personality Disorder Diagnosis through First-Person Narratives 

**Authors**: Karolina Drożdż, Kacper Dudzic, Anna Sterna, Marcin Moskalewicz  

**Link**: [PDF](https://arxiv.org/pdf/2512.20298)  

**Abstract**: Growing reliance on LLMs for psychiatric self-assessment raises questions about their ability to interpret qualitative patient narratives. We present the first direct comparison between state-of-the-art LLMs and mental health professionals in diagnosing Borderline (BPD) and Narcissistic (NPD) Personality Disorders utilizing Polish-language first-person autobiographical accounts. We show that the top-performing Gemini Pro models surpassed human professionals in overall diagnostic accuracy by 21.91 percentage points (65.48% vs. 43.57%). While both models and human experts excelled at identifying BPD (F1 = 83.4 & F1 = 80.0, respectively), models severely underdiagnosed NPD (F1 = 6.7 vs. 50.0), showing a reluctance toward the value-laden term "narcissism." Qualitatively, models provided confident, elaborate justifications focused on patterns and formal categories, while human experts remained concise and cautious, emphasizing the patient's sense of self and temporal experience. Our findings demonstrate that while LLMs are highly competent at interpreting complex first-person clinical data, they remain subject to critical reliability and bias issues. 

---
# TAVID: Text-Driven Audio-Visual Interactive Dialogue Generation 

**Authors**: Ji-Hoon Kim, Junseok Ahn, Doyeop Kwak, Joon Son Chung, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2512.20296)  

**Abstract**: The objective of this paper is to jointly synthesize interactive videos and conversational speech from text and reference images. With the ultimate goal of building human-like conversational systems, recent studies have explored talking or listening head generation as well as conversational speech generation. However, these works are typically studied in isolation, overlooking the multimodal nature of human conversation, which involves tightly coupled audio-visual interactions. In this paper, we introduce TAVID, a unified framework that generates both interactive faces and conversational speech in a synchronized manner. TAVID integrates face and speech generation pipelines through two cross-modal mappers (i.e., a motion mapper and a speaker mapper), which enable bidirectional exchange of complementary information between the audio and visual modalities. We evaluate our system across four dimensions: talking face realism, listening head responsiveness, dyadic interaction fluency, and speech quality. Extensive experiments demonstrate the effectiveness of our approach across all these aspects. 

---
# SlideTailor: Personalized Presentation Slide Generation for Scientific Papers 

**Authors**: Wenzheng Zeng, Mingyu Ouyang, Langyuan Cui, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2512.20292)  

**Abstract**: Automatic presentation slide generation can greatly streamline content creation. However, since preferences of each user may vary, existing under-specified formulations often lead to suboptimal results that fail to align with individual user needs. We introduce a novel task that conditions paper-to-slides generation on user-specified preferences. We propose a human behavior-inspired agentic framework, SlideTailor, that progressively generates editable slides in a user-aligned manner. Instead of requiring users to write their preferences in detailed textual form, our system only asks for a paper-slides example pair and a visual template - natural and easy-to-provide artifacts that implicitly encode rich user preferences across content and visual style. Despite the implicit and unlabeled nature of these inputs, our framework effectively distills and generalizes the preferences to guide customized slide generation. We also introduce a novel chain-of-speech mechanism to align slide content with planned oral narration. Such a design significantly enhances the quality of generated slides and enables downstream applications like video presentations. To support this new task, we construct a benchmark dataset that captures diverse user preferences, with carefully designed interpretable metrics for robust evaluation. Extensive experiments demonstrate the effectiveness of our framework. 

---
# UbiQVision: Quantifying Uncertainty in XAI for Image Recognition 

**Authors**: Akshat Dubey, Aleksandar Anžel, Bahar İlgen, Georges Hattab  

**Link**: [PDF](https://arxiv.org/pdf/2512.20288)  

**Abstract**: Recent advances in deep learning have led to its widespread adoption across diverse domains, including medical imaging. This progress is driven by increasingly sophisticated model architectures, such as ResNets, Vision Transformers, and Hybrid Convolutional Neural Networks, that offer enhanced performance at the cost of greater complexity. This complexity often compromises model explainability and interpretability. SHAP has emerged as a prominent method for providing interpretable visualizations that aid domain experts in understanding model predictions. However, SHAP explanations can be unstable and unreliable in the presence of epistemic and aleatoric uncertainty. In this study, we address this challenge by using Dirichlet posterior sampling and Dempster-Shafer theory to quantify the uncertainty that arises from these unstable explanations in medical imaging applications. The framework uses a belief, plausible, and fusion map approach alongside statistical quantitative analysis to produce quantification of uncertainty in SHAP. Furthermore, we evaluated our framework on three medical imaging datasets with varying class distributions, image qualities, and modality types which introduces noise due to varying image resolutions and modality-specific aspect covering the examples from pathology, ophthalmology, and radiology, introducing significant epistemic uncertainty. 

---
# ${D}^{3}${ETOR}: ${D}$ebate-Enhanced Pseudo Labeling and Frequency-Aware Progressive ${D}$ebiasing for Weakly-Supervised Camouflaged Object ${D}$etection with Scribble Annotations 

**Authors**: Jiawei Ge, Jiuxin Cao, Xinyi Li, Xuelin Zhu, Chang Liu, Bo Liu, Chen Feng, Ioannis Patras  

**Link**: [PDF](https://arxiv.org/pdf/2512.20260)  

**Abstract**: Weakly-Supervised Camouflaged Object Detection (WSCOD) aims to locate and segment objects that are visually concealed within their surrounding scenes, relying solely on sparse supervision such as scribble annotations. Despite recent progress, existing WSCOD methods still lag far behind fully supervised ones due to two major limitations: (1) the pseudo masks generated by general-purpose segmentation models (e.g., SAM) and filtered via rules are often unreliable, as these models lack the task-specific semantic understanding required for effective pseudo labeling in COD; and (2) the neglect of inherent annotation bias in scribbles, which hinders the model from capturing the global structure of camouflaged objects. To overcome these challenges, we propose ${D}^{3}$ETOR, a two-stage WSCOD framework consisting of Debate-Enhanced Pseudo Labeling and Frequency-Aware Progressive Debiasing. In the first stage, we introduce an adaptive entropy-driven point sampling method and a multi-agent debate mechanism to enhance the capability of SAM for COD, improving the interpretability and precision of pseudo masks. In the second stage, we design FADeNet, which progressively fuses multi-level frequency-aware features to balance global semantic understanding with local detail modeling, while dynamically reweighting supervision strength across regions to alleviate scribble bias. By jointly exploiting the supervision signals from both the pseudo masks and scribble semantics, ${D}^{3}$ETOR significantly narrows the gap between weakly and fully supervised COD, achieving state-of-the-art performance on multiple benchmarks. 

---
# Memory as Resonance: A Biomimetic Architecture for Infinite Context Memory on Ergodic Phonetic Manifolds 

**Authors**: Tarik Houichime, Abdelghani Souhar, Younes El Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2512.20245)  

**Abstract**: The memory of contemporary Large Language Models is bound by a physical paradox: as they learn, they fill up. The linear accumulation (O(N)) of Key-Value states treats context as a warehouse of static artifacts, eventually forcing a destructive choice between amnesia and latency. We challenge this discrete orthodoxy, proposing that long-term memory is not the storage of items, but the persistence of a trajectory. We introduce Phonetic Trajectory Memory (PTM), a neuro-symbolic architecture that encodes language not as a sequence of tensors, but as a continuous path on an ergodic manifold governed by irrational rotation matrices. By decoupling the navigation (an invariant O(1) geometric signal) from the reconstruction (a probabilistic generative act), PTM achieves a compression magnitude of greater than 3,000x relative to dense caches. We demonstrate that retrieval becomes a process of resonance: the phonetic trace stabilizes the model against hallucination via "Signal Consensus" mechanism, securing up to approximately 92% factual accuracy. While this aggressive abstraction alters generative texture, it unlocks immediate access latency (approximately 34ms) independent of depth. Our results suggest that infinite context does not require infinite silicon; it requires treating memory not as data to be stored, but as a reconstructive process acting on a conserved, undying physical signal. 

---
# Corpus of Cross-lingual Dialogues with Minutes and Detection of Misunderstandings 

**Authors**: Marko Čechovič, Natália Komorníková, Dominik Macháček, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2512.20204)  

**Abstract**: Speech processing and translation technology have the potential to facilitate meetings of individuals who do not share any common language. To evaluate automatic systems for such a task, a versatile and realistic evaluation corpus is needed. Therefore, we create and present a corpus of cross-lingual dialogues between individuals without a common language who were facilitated by automatic simultaneous speech translation. The corpus consists of 5 hours of speech recordings with ASR and gold transcripts in 12 original languages and automatic and corrected translations into English. For the purposes of research into cross-lingual summarization, our corpus also includes written summaries (minutes) of the meetings.
Moreover, we propose automatic detection of misunderstandings. For an overview of this task and its complexity, we attempt to quantify misunderstandings in cross-lingual meetings. We annotate misunderstandings manually and also test the ability of current large language models to detect them automatically. The results show that the Gemini model is able to identify text spans with misunderstandings with recall of 77% and precision of 47%. 

---
# Asynchronous Fast-Slow Vision-Language-Action Policies for Whole-Body Robotic Manipulation 

**Authors**: Teqiang Zou, Hongliang Zeng, Yuxuan Nong, Yifan Li, Kehui Liu, Haotian Yang, Xinyang Ling, Xin Li, Lianyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2512.20188)  

**Abstract**: Most Vision-Language-Action (VLA) systems integrate a Vision-Language Model (VLM) for semantic reasoning with an action expert generating continuous action signals, yet both typically run at a single unified frequency. As a result, policy performance is constrained by the low inference speed of large VLMs. This mandatory synchronous execution severely limits control stability and real-time performance in whole-body robotic manipulation, which involves more joints, larger motion spaces, and dynamically changing views. We introduce a truly asynchronous Fast-Slow VLA framework (DuoCore-FS), organizing the system into a fast pathway for high-frequency action generation and a slow pathway for rich VLM reasoning. The system is characterized by two key features. First, a latent representation buffer bridges the slow and fast systems. It stores instruction semantics and action-reasoning representation aligned with the scene-instruction context, providing high-level guidance to the fast pathway. Second, a whole-body action tokenizer provides a compact, unified representation of whole-body actions. Importantly, the VLM and action expert are still jointly trained end-to-end, preserving unified policy learning while enabling asynchronous execution. DuoCore-FS supports a 3B-parameter VLM while achieving 30 Hz whole-body action-chunk generation, approximately three times as fast as prior VLA models with comparable model sizes. Real-world whole-body manipulation experiments demonstrate improved task success rates and significantly enhanced responsiveness compared to synchronous Fast-Slow VLA baselines. The implementation of DuoCore-FS, including training, inference, and deployment, is provided to commercial users by Astribot as part of the Astribot robotic platform. 

---
# FaithLens: Detecting and Explaining Faithfulness Hallucination 

**Authors**: Shuzheng Si, Qingyi Wang, Haozhe Zhao, Yuzhuo Bai, Guanqiao Chen, Kangyang Luo, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.20182)  

**Abstract**: Recognizing whether outputs from large language models (LLMs) contain faithfulness hallucination is crucial for real-world applications, e.g., retrieval-augmented generation and summarization. In this paper, we introduce FaithLens, a cost-efficient and effective faithfulness hallucination detection model that can jointly provide binary predictions and corresponding explanations to improve trustworthiness. To achieve this, we first synthesize training data with explanations via advanced LLMs and apply a well-defined data filtering strategy to ensure label correctness, explanation quality, and data diversity. Subsequently, we fine-tune the model on these well-curated training data as a cold start and further optimize it with rule-based reinforcement learning, using rewards for both prediction correctness and explanation quality. Results on 12 diverse tasks show that the 8B-parameter FaithLens outperforms advanced models such as GPT-4.1 and o3. Also, FaithLens can produce high-quality explanations, delivering a distinctive balance of trustworthiness, efficiency, and effectiveness. 

---
# Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography 

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2512.20168)  

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems. 

---
# AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications 

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2512.20164)  

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation. 

---
# AXIOM: Benchmarking LLM-as-a-Judge for Code via Rule-Based Perturbation and Multisource Quality Calibration 

**Authors**: Ruiqi Wang, Xinchen Wang, Cuiyun Gao, Chun Yong Chong, Xin Xia, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2512.20159)  

**Abstract**: Large language models (LLMs) have been increasingly deployed in real-world software engineering, fostering the development of code evaluation metrics to study the quality of LLM-generated code. Conventional rule-based metrics merely score programs based on their surface-level similarities with reference programs instead of analyzing functionality and code quality in depth. To address this limitation, researchers have developed LLM-as-a-judge metrics, prompting LLMs to evaluate and score code, and curated various code evaluation benchmarks to validate their effectiveness. However, these benchmarks suffer from critical limitations, hindering reliable assessments of evaluation capability: Some feature coarse-grained binary labels, which reduce rich code behavior to a single bit of information, obscuring subtle errors. Others propose fine-grained but subjective, vaguely-defined evaluation criteria, introducing unreliability in manually-annotated scores, which is the ground-truth they rely on. Furthermore, they often use uncontrolled data synthesis methods, leading to unbalanced score distributions that poorly represent real-world code generation scenarios.
To curate a diverse benchmark with programs of well-balanced distributions across various quality levels and streamline the manual annotation procedure, we propose AXIOM, a novel perturbation-based framework for synthesizing code evaluation benchmarks at scale. It reframes program scores as the refinement effort needed for deployment, consisting of two stages: (1) Rule-guided perturbation, which prompts LLMs to apply sequences of predefined perturbation rules to existing high-quality programs to modify their functionality and code quality, enabling us to precisely control each program's target score to achieve balanced score distributions. (2) Multisource quality calibration, which first selects a subset of... 

---
# Fun-Audio-Chat Technical Report 

**Authors**: Qian Chen, Luyao Cheng, Chong Deng, Xiangang Li, Jiaqing Liu, Chao-Hong Tan, Wen Wang, Junhao Xu, Jieping Ye, Qinglin Zhang, Qiquan Zhang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.20156)  

**Abstract**: Recent advancements in joint speech-text models show great potential for seamless voice interactions. However, existing models face critical challenges: temporal resolution mismatch between speech tokens (25Hz) and text tokens (~3Hz) dilutes semantic information, incurs high computational costs, and causes catastrophic forgetting of text LLM knowledge. We introduce Fun-Audio-Chat, a Large Audio Language Model addressing these limitations via two innovations from our previous work DrVoice. First, Dual-Resolution Speech Representations (DRSR): the Shared LLM processes audio at efficient 5Hz (via token grouping), while the Speech Refined Head generates high-quality tokens at 25Hz, balancing efficiency (~50% GPU reduction) and quality. Second, Core-Cocktail Training, a two-stage fine-tuning with intermediate merging that mitigates catastrophic forgetting. We then apply Multi-Task DPO Training to enhance robustness, audio understanding, instruction-following and voice empathy. This multi-stage post-training enables Fun-Audio-Chat to retain text LLM knowledge while gaining powerful audio understanding, reasoning, and generation. Unlike recent LALMs requiring large-scale audio-text pre-training, Fun-Audio-Chat leverages pre-trained models and extensive post-training. Fun-Audio-Chat 8B and MoE 30B-A3B achieve competitive performance on Speech-to-Text and Speech-to-Speech tasks, ranking top among similar-scale models on Spoken QA benchmarks. They also achieve competitive to superior performance on Audio Understanding, Speech Function Calling, Instruction-Following and Voice Empathy. We develop Fun-Audio-Chat-Duplex, a full-duplex variant with strong performance on Spoken QA and full-duplex interactions. We open-source Fun-Audio-Chat-8B with training and inference code, and provide an interactive demo. 

---
# Retrieval-augmented Prompt Learning for Pre-trained Foundation Models 

**Authors**: Xiang Chen, Yixin Ou, Quan Feng, Lei Li, Piji Li, Haibo Ye, Sheng-Jun Huang, Shuofei Qiao, Shumin Deng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20145)  

**Abstract**: The pre-trained foundation models (PFMs) have become essential for facilitating large-scale multimodal learning. Researchers have effectively employed the ``pre-train, prompt, and predict'' paradigm through prompt learning to induce improved few-shot performance. However, prompt learning approaches for PFMs still follow a parametric learning paradigm. As such, the stability of generalization in memorization and rote learning can be compromised. More specifically, conventional prompt learning might face difficulties in fully utilizing atypical instances and avoiding overfitting to shallow patterns with limited data during the process of fully-supervised training. To overcome these constraints, we present our approach, named RetroPrompt, which aims to achieve a balance between memorization and generalization by decoupling knowledge from mere memorization. Unlike traditional prompting methods, RetroPrompt leverages a publicly accessible knowledge base generated from the training data and incorporates a retrieval mechanism throughout the input, training, and inference stages. This enables the model to actively retrieve relevant contextual information from the corpus, thereby enhancing the available cues. We conduct comprehensive experiments on a variety of datasets across natural language processing and computer vision tasks to demonstrate the superior performance of our proposed approach, RetroPrompt, in both zero-shot and few-shot scenarios. Through detailed analysis of memorization patterns, we observe that RetroPrompt effectively reduces the reliance on rote memorization, leading to enhanced generalization. 

---
# M$^3$KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation 

**Authors**: Hyeongcheol Park, Jiyoung Seo, Jaewon Mun, Hogun Park, Wonmin Byeon, Sung June Kim, Hyeonsoo Im, JeungSub Lee, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.20136)  

**Abstract**: Retrieval-Augmented Generation (RAG) has recently been extended to multimodal settings, connecting multimodal large language models (MLLMs) with vast corpora of external knowledge such as multimodal knowledge graphs (MMKGs). Despite their recent success, multimodal RAG in the audio-visual domain remains challenging due to 1) limited modality coverage and multi-hop connectivity of existing MMKGs, and 2) retrieval based solely on similarity in a shared multimodal embedding space, which fails to filter out off-topic or redundant knowledge. To address these limitations, we propose M$^3$KG-RAG, a Multi-hop Multimodal Knowledge Graph-enhanced RAG that retrieves query-aligned audio-visual knowledge from MMKGs, improving reasoning depth and answer faithfulness in MLLMs. Specifically, we devise a lightweight multi-agent pipeline to construct multi-hop MMKG (M$^3$KG), which contains context-enriched triplets of multimodal entities, enabling modality-wise retrieval based on input queries. Furthermore, we introduce GRASP (Grounded Retrieval And Selective Pruning), which ensures precise entity grounding to the query, evaluates answer-supporting relevance, and prunes redundant context to retain only knowledge essential for response generation. Extensive experiments across diverse multimodal benchmarks demonstrate that M$^3$KG-RAG significantly enhances MLLMs' multimodal reasoning and grounding over existing approaches. 

---
# Evolutionary Neural Architecture Search with Dual Contrastive Learning 

**Authors**: Xian-Rong Zhang, Yue-Jiao Gong, Wei-Neng Chen, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20112)  

**Abstract**: Evolutionary Neural Architecture Search (ENAS) has gained attention for automatically designing neural network architectures. Recent studies use a neural predictor to guide the process, but the high computational costs of gathering training data -- since each label requires fully training an architecture -- make achieving a high-precision predictor with { limited compute budget (i.e., a capped number of fully trained architecture-label pairs)} crucial for ENAS success. This paper introduces ENAS with Dual Contrastive Learning (DCL-ENAS), a novel method that employs two stages of contrastive learning to train the neural predictor. In the first stage, contrastive self-supervised learning is used to learn meaningful representations from neural architectures without requiring labels. In the second stage, fine-tuning with contrastive learning is performed to accurately predict the relative performance of different architectures rather than their absolute performance, which is sufficient to guide the evolutionary search. Across NASBench-101 and NASBench-201, DCL-ENAS achieves the highest validation accuracy, surpassing the strongest published baselines by 0.05\% (ImageNet16-120) to 0.39\% (NASBench-101). On a real-world ECG arrhythmia classification task, DCL-ENAS improves performance by approximately 2.5 percentage points over a manually designed, non-NAS model obtained via random search, while requiring only 7.7 GPU-days. 

---
# ABBEL: LLM Agents Acting through Belief Bottlenecks Expressed in Language 

**Authors**: Aly Lidayan, Jakob Bjorner, Satvik Golechha, Kartik Goyal, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2512.20111)  

**Abstract**: As the length of sequential decision-making tasks increases, it becomes computationally impractical to keep full interaction histories in context. We introduce a general framework for LLM agents to maintain concise contexts through multi-step interaction: Acting through Belief Bottlenecks Expressed in Language (ABBEL), and methods to further improve ABBEL agents with RL post-training. ABBEL replaces long multi-step interaction history by a belief state, i.e., a natural language summary of what has been discovered about task-relevant unknowns. Under ABBEL, at each step the agent first updates a prior belief with the most recent observation from the environment to form a posterior belief, then uses only the posterior to select an action. We systematically evaluate frontier models under ABBEL across six diverse multi-step environments, finding that ABBEL supports generating interpretable beliefs while maintaining near-constant memory use over interaction steps. However, bottleneck approaches are generally prone to error propagation, which we observe causing inferior performance when compared to the full context setting due to errors in belief updating. Therefore, we train LLMs to generate and act on beliefs within the ABBEL framework via reinforcement learning (RL). We experiment with belief grading, to reward higher quality beliefs, as well as belief length penalties to reward more compressed beliefs. Our experiments demonstrate the ability of RL to improve ABBEL's performance beyond the full context setting, while using less memory than contemporaneous approaches. 

---
# Item Region-based Style Classification Network (IRSN): A Fashion Style Classifier Based on Domain Knowledge of Fashion Experts 

**Authors**: Jinyoung Choi, Youngchae Kwon, Injung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.20088)  

**Abstract**: Fashion style classification is a challenging task because of the large visual variation within the same style and the existence of visually similar styles.
Styles are expressed not only by the global appearance, but also by the attributes of individual items and their combinations.
In this study, we propose an item region-based fashion style classification network (IRSN) to effectively classify fashion styles by analyzing item-specific features and their combinations in addition to global features.
IRSN extracts features of each item region using item region pooling (IRP), analyzes them separately, and combines them using gated feature fusion (GFF).
In addition, we improve the feature extractor by applying a dual-backbone architecture that combines a domain-specific feature extractor and a general feature extractor pre-trained with a large-scale image-text dataset.
In experiments, applying IRSN to six widely-used backbones, including EfficientNet, ConvNeXt, and Swin Transformer, improved style classification accuracy by an average of 6.9% and a maximum of 14.5% on the FashionStyle14 dataset and by an average of 7.6% and a maximum of 15.1% on the ShowniqV3 dataset. Visualization analysis also supports that the IRSN models are better than the baseline models at capturing differences between similar style classes. 

---
# Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection 

**Authors**: Jeehong Kim, Youngseok Hwang, Minchan Kim, Sungho Bae, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.20086)  

**Abstract**: Spatio-temporal graph neural networks (ST-GNNs) have achieved notable success in structured domains such as road traffic and public transportation, where spatial entities can be naturally represented as fixed nodes. In contrast, many real-world systems including maritime traffic lack such fixed anchors, making the construction of spatio-temporal graphs a fundamental challenge. Anomaly detection in these non-grid environments is particularly difficult due to the absence of canonical reference points, the sparsity and irregularity of trajectories, and the fact that anomalies may manifest at multiple granularities. In this work, we introduce a novel benchmark dataset for anomaly detection in the maritime domain, extending the Open Maritime Traffic Analysis Dataset (OMTAD) into a benchmark tailored for graph-based anomaly detection. Our dataset enables systematic evaluation across three different granularities: node-level, edge-level, and graph-level anomalies. We plan to employ two specialized LLM-based agents: \emph{Trajectory Synthesizer} and \emph{Anomaly Injector} to construct richer interaction contexts and generate semantically meaningful anomalies. We expect this benchmark to promote reproducibility and to foster methodological advances in anomaly detection for non-grid spatio-temporal systems. 

---
# QE-Catalytic: A Graph-Language Multimodal Base Model for Relaxed-Energy Prediction in Catalytic Adsorption 

**Authors**: Yanjie Li, Jian Xu, Xueqing Chen, Lina Yu, Shiming Xiang, Weijun Li, Cheng-lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20084)  

**Abstract**: Adsorption energy is a key descriptor of catalytic reactivity. It is fundamentally defined as the difference between the relaxed total energy of the adsorbate-surface system and that of an appropriate reference state; therefore, the accuracy of relaxed-energy prediction directly determines the reliability of machine-learning-driven catalyst screening. E(3)-equivariant graph neural networks (GNNs) can natively operate on three-dimensional atomic coordinates under periodic boundary conditions and have demonstrated strong performance on such tasks. In contrast, language-model-based approaches, while enabling human-readable textual descriptions and reducing reliance on explicit graph -- thereby broadening applicability -- remain insufficient in both adsorption-configuration energy prediction accuracy and in distinguishing ``the same system with different configurations,'' even with graph-assisted pretraining in the style of GAP-CATBERTa.
To this end, we propose QE-Catalytic, a multimodal framework that deeply couples a large language model (\textbf{Q}wen) with an E(3)-equivariant graph Transformer (\textbf{E}quiformer-V2), enabling unified support for adsorption-configuration property prediction and inverse design on complex catalytic surfaces. During prediction, QE-Catalytic jointly leverages three-dimensional structures and structured configuration text, and injects ``3D geometric information'' into the language channel via graph-text alignment, allowing it to function as a high-performance text-based predictor when precise coordinates are unavailable, while also autoregressively generating CIF files for target-energy-driven structure design and information completion. On OC20, QE-Catalytic reduces the MAE of relaxed adsorption energy from 0.713~eV to 0.486~eV, and consistently outperforms baseline models such as CatBERTa and GAP-CATBERTa across multiple evaluation protocols. 

---
# CBA: Communication-Bound-Aware Cross-Domain Resource Assignment for Pipeline-Parallel Distributed LLM Training in Dynamic Multi-DC Optical Networks 

**Authors**: Dianxuan Fu, Xiaomin Liu, Yihao Zhang, Shikui Shen, Weisheng Hu, Qunbi Zhuge  

**Link**: [PDF](https://arxiv.org/pdf/2512.20080)  

**Abstract**: We propose a communication-bound-aware cross-domain resource assignment framework for pipeline-parallel distributed training over multi-datacenter optical networks, which lowers iteration time by 31.25% and reduces 13.20% blocking requests compared to baselines. 

---
# On the Effectiveness of Instruction-Tuning Local LLMs for Identifying Software Vulnerabilities 

**Authors**: Sangryu Park, Gihyuk Ko, Homook Cho  

**Link**: [PDF](https://arxiv.org/pdf/2512.20062)  

**Abstract**: Large Language Models (LLMs) show significant promise in automating software vulnerability analysis, a critical task given the impact of security failure of modern software systems. However, current approaches in using LLMs to automate vulnerability analysis mostly rely on using online API-based LLM services, requiring the user to disclose the source code in development. Moreover, they predominantly frame the task as a binary classification(vulnerable or not vulnerable), limiting potential practical utility. This paper addresses these limitations by reformulating the problem as Software Vulnerability Identification (SVI), where LLMs are asked to output the type of weakness in Common Weakness Enumeration (CWE) IDs rather than simply indicating the presence or absence of a vulnerability. We also tackle the reliance on large, API-based LLMs by demonstrating that instruction-tuning smaller, locally deployable LLMs can achieve superior identification performance. In our analysis, instruct-tuning a local LLM showed better overall performance and cost trade-off than online API-based LLMs. Our findings indicate that instruct-tuned local models represent a more effective, secure, and practical approach for leveraging LLMs in real-world vulnerability management workflows. 

---
# An Optimal Policy for Learning Controllable Dynamics by Exploration 

**Authors**: Peter N. Loxley  

**Link**: [PDF](https://arxiv.org/pdf/2512.20053)  

**Abstract**: Controllable Markov chains describe the dynamics of sequential decision making tasks and are the central component in optimal control and reinforcement learning. In this work, we give the general form of an optimal policy for learning controllable dynamics in an unknown environment by exploring over a limited time horizon. This policy is simple to implement and efficient to compute, and allows an agent to ``learn by exploring" as it maximizes its information gain in a greedy fashion by selecting controls from a constraint set that changes over time during exploration. We give a simple parameterization for the set of controls, and present an algorithm for finding an optimal policy. The reason for this policy is due to the existence of certain types of states that restrict control of the dynamics; such as transient states, absorbing states, and non-backtracking states. We show why the occurrence of these states makes a non-stationary policy essential for achieving optimal exploration. Six interesting examples of controllable dynamics are treated in detail. Policy optimality is demonstrated using counting arguments, comparing with suboptimal policies, and by making use of a sequential improvement property from dynamic programming. 

---
# Beyond Vision: Contextually Enriched Image Captioning with Multi-Modal Retrieva 

**Authors**: Nguyen Lam Phu Quy, Pham Phu Hoa, Tran Chi Nguyen, Dao Sy Duy Minh, Nguyen Hoang Minh Ngoc, Huynh Trung Kiet  

**Link**: [PDF](https://arxiv.org/pdf/2512.20042)  

**Abstract**: Real-world image captions often lack contextual depth, omitting crucial details such as event background, temporal cues, outcomes, and named entities that are not visually discernible. This gap limits the effectiveness of image understanding in domains like journalism, education, and digital archives, where richer, more informative descriptions are essential. To address this, we propose a multimodal pipeline that augments visual input with external textual knowledge. Our system retrieves semantically similar images using BEIT-3 (Flickr30k-384 and COCO-384) and SigLIP So-384, reranks them using ORB and SIFT for geometric alignment, and extracts contextual information from related articles via semantic search. A fine-tuned Qwen3 model with QLoRA then integrates this context with base captions generated by Instruct BLIP (Vicuna-7B) to produce event-enriched, context-aware descriptions. Evaluated on the OpenEvents v1 dataset, our approach generates significantly more informative captions compared to traditional methods, showing strong potential for real-world applications requiring deeper visual-textual understanding 

---
# DecoKAN: Interpretable Decomposition for Forecasting Cryptocurrency Market Dynamics 

**Authors**: Yuan Gao, Zhenguo Dong, Xuelong Wang, Zhiqiang Wang, Yong Zhang, Shaofan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20028)  

**Abstract**: Accurate and interpretable forecasting of multivariate time series is crucial for understanding the complex dynamics of cryptocurrency markets in digital asset systems. Advanced deep learning methodologies, particularly Transformer-based and MLP-based architectures, have achieved competitive predictive performance in cryptocurrency forecasting tasks. However, cryptocurrency data is inherently composed of long-term socio-economic trends and local high-frequency speculative oscillations. Existing deep learning-based 'black-box' models fail to effectively decouple these composite dynamics or provide the interpretability needed for trustworthy financial decision-making. To overcome these limitations, we propose DecoKAN, an interpretable forecasting framework that integrates multi-level Discrete Wavelet Transform (DWT) for decoupling and hierarchical signal decomposition with Kolmogorov-Arnold Network (KAN) mixers for transparent and interpretable nonlinear modeling. The DWT component decomposes complex cryptocurrency time series into distinct frequency components, enabling frequency-specific analysis, while KAN mixers provide intrinsically interpretable spline-based mappings within each decomposed subseries. Furthermore, interpretability is enhanced through a symbolic analysis pipeline involving sparsification, pruning, and symbolization, which produces concise analytical expressions offering symbolic representations of the learned patterns. Extensive experiments demonstrate that DecoKAN achieves the lowest average Mean Squared Error on all tested real-world cryptocurrency datasets (BTC, ETH, XMR), consistently outperforming a comprehensive suite of competitive state-of-the-art baselines. These results validate DecoKAN's potential to bridge the gap between predictive accuracy and model transparency, advancing trustworthy decision support within complex cryptocurrency markets. 

---
# Bring My Cup! Personalizing Vision-Language-Action Models with Visual Attentive Prompting 

**Authors**: Sangoh Lee, Sangwoo Mo, Wook-Shin Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.20014)  

**Abstract**: While Vision-Language-Action (VLA) models generalize well to generic instructions, they struggle with personalized commands such as "bring my cup", where the robot must act on one specific instance among visually similar objects. We study this setting of manipulating personal objects, in which a VLA must identify and control a user-specific object unseen during training using only a few reference images. To address this challenge, we propose Visual Attentive Prompting (VAP), a simple-yet-effective training-free perceptual adapter that equips frozen VLAs with top-down selective attention. VAP treats the reference images as a non-parametric visual memory, grounds the personal object in the scene through open-vocabulary detection and embedding-based matching, and then injects this grounding as a visual prompt by highlighting the object and rewriting the instruction. We construct two simulation benchmarks, Personalized-SIMPLER and Personalized-VLABench, and a real-world tabletop benchmark to evaluate personalized manipulation across multiple robots and tasks. Experiments show that VAP consistently outperforms generic policies and token-learning baselines in both success rate and correct-object manipulation, helping to bridge the gap between semantic understanding and instance-level control. 

---
# IoT-based Android Malware Detection Using Graph Neural Network With Adversarial Defense 

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20004)  

**Abstract**: Since the Internet of Things (IoT) is widely adopted using Android applications, detecting malicious Android apps is essential. In recent years, Android graph-based deep learning research has proposed many approaches to extract relationships from applications as graphs to generate graph embeddings. First, we demonstrate the effectiveness of graph-based classification using a Graph Neural Network (GNN)-based classifier to generate API graph embeddings. The graph embeddings are combined with Permission and Intent features to train multiple machine learning and deep learning models for Android malware detection. The proposed classification approach achieves an accuracy of 98.33 percent on the CICMaldroid dataset and 98.68 percent on the Drebin dataset. However, graph-based deep learning models are vulnerable, as attackers can add fake relationships to evade detection by the classifier. Second, we propose a Generative Adversarial Network (GAN)-based attack algorithm named VGAE-MalGAN targeting graph-based GNN Android malware classifiers. The VGAE-MalGAN generator produces adversarial malware API graphs, while the VGAE-MalGAN substitute detector attempts to mimic the target detector. Experimental results show that VGAE-MalGAN can significantly reduce the detection rate of GNN-based malware classifiers. Although the model initially fails to detect adversarial malware, retraining with generated adversarial samples improves robustness and helps mitigate adversarial attacks. 

---
# Schoenfeld's Anatomy of Mathematical Reasoning by Language Models 

**Authors**: Ming Li, Chenrui Fan, Yize Cheng, Soheil Feizi, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.19995)  

**Abstract**: Large language models increasingly expose reasoning traces, yet their underlying cognitive structure and steps remain difficult to identify and analyze beyond surface-level statistics. We adopt Schoenfeld's Episode Theory as an inductive, intermediate-scale lens and introduce ThinkARM (Anatomy of Reasoning in Models), a scalable framework that explicitly abstracts reasoning traces into functional reasoning steps such as Analysis, Explore, Implement, Verify, etc. When applied to mathematical problem solving by diverse models, this abstraction reveals reproducible thinking dynamics and structural differences between reasoning and non-reasoning models, which are not apparent from token-level views. We further present two diagnostic case studies showing that exploration functions as a critical branching step associated with correctness, and that efficiency-oriented methods selectively suppress evaluative feedback steps rather than uniformly shortening responses. Together, our results demonstrate that episode-level representations make reasoning steps explicit, enabling systematic analysis of how reasoning is structured, stabilized, and altered in modern language models. 

---
# Neuron-Guided Interpretation of Code LLMs: Where, Why, and How? 

**Authors**: Zhe Yin, Xiaodong Gu, Beijun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19980)  

**Abstract**: Code language models excel on code intelligence tasks, yet their internal interpretability is underexplored. Existing neuron interpretability techniques from NLP are suboptimal for source code due to programming languages formal, hierarchical, and executable nature. We empirically investigate code LLMs at the neuron level, localizing language-specific neurons (selectively responsive to one language) and concept layers (feed-forward layers encoding language-agnostic code representations). We analyze Llama-3.1-8B and Qwen2.5-Coder-32B on multilingual inputs in C++, Java, Python, Go, and JavaScript, measuring neuron selectivity and layerwise contributions during generation. We find (1) neurons specialized for individual languages alongside a universal subset supporting general-purpose generation; and (2) lower layers mainly encode language-specific syntax, while middle layers capture semantic abstractions shared across languages, emerging as concept layers. We demonstrate utility on three tasks: neuron-guided fine-tuning for code generation, clone detection via concept-layer embeddings, and concept-layer-guided transfer for code summarization, each yielding consistent gains in multilingual settings. 

---
# Regression of Functions by Quantum Neural Networks Circuits 

**Authors**: Fernando M. de Paula Neto, Lucas dos Reis Silva, Paulo S. G. de Mattos Neto, Felipe F. Fanchini  

**Link**: [PDF](https://arxiv.org/pdf/2512.19978)  

**Abstract**: The performance of quantum neural network models depends strongly on architectural decisions, including circuit depth, placement of parametrized operations, and data-encoding strategies. Selecting an effective architecture is challenging and closely related to the classical difficulty of choosing suitable neural-network topologies, which is computationally hard. This work investigates automated quantum-circuit construction for regression tasks and introduces a genetic-algorithm framework that discovers Reduced Regressor QNN architectures. The approach explores depth, parametrized gate configurations, and flexible data re-uploading patterns, formulating the construction of quantum regressors as an optimization process. The discovered circuits are evaluated against seventeen classical regression models on twenty-two nonlinear benchmark functions and four analytical functions. Although classical methods often achieve comparable results, they typically require far more parameters, whereas the evolved quantum models remain compact while providing competitive performance. We further analyze dataset complexity using twelve structural descriptors and show, across five increasingly challenging meta-learning scenarios, that these measures can reliably predict which quantum architecture will perform best. The results demonstrate perfect or near-perfect predictive accuracy in several scenarios, indicating that complexity metrics offer powerful and compact representations of dataset structure and can effectively guide automated model selection. Overall, this study provides a principled basis for meta-learning-driven quantum architecture design and advances the understanding of how quantum models behave in regression settings--a topic that has received limited exploration in prior work. These findings pave the way for more systematic and theoretically grounded approaches to quantum regression. 

---
# How Much 3D Do Video Foundation Models Encode? 

**Authors**: Zixuan Huang, Xiang Li, Zhaoyang Lv, James M. Rehg  

**Link**: [PDF](https://arxiv.org/pdf/2512.19949)  

**Abstract**: Videos are continuous 2D projections of 3D worlds. After training on large video data, will global 3D understanding naturally emerge? We study this by quantifying the 3D understanding of existing Video Foundation Models (VidFMs) pretrained on vast video data. We propose the first model-agnostic framework that measures the 3D awareness of various VidFMs by estimating multiple 3D properties from their features via shallow read-outs. Our study presents meaningful findings regarding the 3D awareness of VidFMs on multiple axes. In particular, we show that state-of-the-art video generation models exhibit a strong understanding of 3D objects and scenes, despite not being trained on any 3D data. Such understanding can even surpass that of large expert models specifically trained for 3D tasks. Our findings, together with the 3D benchmarking of major VidFMs, provide valuable observations for building scalable 3D models. 

---
# Block-Recurrent Dynamics in Vision Transformers 

**Authors**: Mozes Jacobs, Thomas Fel, Richard Hakim, Alessandra Brondetta, Demba Ba, T. Andy Keller  

**Link**: [PDF](https://arxiv.org/pdf/2512.19941)  

**Abstract**: As Vision Transformers (ViTs) become standard vision backbones, a mechanistic account of their computational phenomenology is essential. Despite architectural cues that hint at dynamical structure, there is no settled framework that interprets Transformer depth as a well-characterized flow. In this work, we introduce the Block-Recurrent Hypothesis (BRH), arguing that trained ViTs admit a block-recurrent depth structure such that the computation of the original $L$ blocks can be accurately rewritten using only $k \ll L$ distinct blocks applied recurrently. Across diverse ViTs, between-layer representational similarity matrices suggest few contiguous phases. To determine whether these phases reflect genuinely reusable computation, we train block-recurrent surrogates of pretrained ViTs: Recurrent Approximations to Phase-structured TransfORmers (Raptor). In small-scale, we demonstrate that stochastic depth and training promote recurrent structure and subsequently correlate with our ability to accurately fit Raptor. We then provide an empirical existence proof for BRH by training a Raptor model to recover $96\%$ of DINOv2 ImageNet-1k linear probe accuracy in only 2 blocks at equivalent computational cost. Finally, we leverage our hypothesis to develop a program of Dynamical Interpretability. We find i) directional convergence into class-dependent angular basins with self-correcting trajectories under small perturbations, ii) token-specific dynamics, where cls executes sharp late reorientations while patch tokens exhibit strong late-stage coherence toward their mean direction, and iii) a collapse to low rank updates in late depth, consistent with convergence to low-dimensional attractors. Altogether, we find a compact recurrent program emerges along ViT depth, pointing to a low-complexity normative solution that enables these models to be studied through principled dynamical systems analysis. 

---
# Conditional Adversarial Fragility in Financial Machine Learning under Macroeconomic Stress 

**Authors**: Samruddhi Baviskar  

**Link**: [PDF](https://arxiv.org/pdf/2512.19935)  

**Abstract**: Machine learning models used in financial decision systems operate in nonstationary economic environments, yet adversarial robustness is typically evaluated under static assumptions. This work introduces Conditional Adversarial Fragility, a regime dependent phenomenon in which adversarial vulnerability is systematically amplified during periods of macroeconomic stress. We propose a regime aware evaluation framework for time indexed tabular financial classification tasks that conditions robustness assessment on external indicators of economic stress. Using volatility based regime segmentation as a proxy for macroeconomic conditions, we evaluate model behavior across calm and stress periods while holding model architecture, attack methodology, and evaluation protocols constant. Baseline predictive performance remains comparable across regimes, indicating that economic stress alone does not induce inherent performance degradation. Under adversarial perturbations, however, models operating during stress regimes exhibit substantially greater degradation across predictive accuracy, operational decision thresholds, and risk sensitive outcomes. We further demonstrate that this amplification propagates to increased false negative rates, elevating the risk of missed high risk cases during adverse conditions. To complement numerical robustness metrics, we introduce an interpretive governance layer based on semantic auditing of model explanations using large language models. Together, these results demonstrate that adversarial robustness in financial machine learning is a regime dependent property and motivate stress aware approaches to model risk assessment in high stakes financial deployments. 

---
# Vehicle-centric Perception via Multimodal Structured Pre-training 

**Authors**: Wentao Wu, Xiao Wang, Chenglong Li, Jin Tang, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.19934)  

**Abstract**: Vehicle-centric perception plays a crucial role in many intelligent systems, including large-scale surveillance systems, intelligent transportation, and autonomous driving. Existing approaches lack effective learning of vehicle-related knowledge during pre-training, resulting in poor capability for modeling general vehicle perception representations. To handle this problem, we propose VehicleMAE-V2, a novel vehicle-centric pre-trained large model. By exploring and exploiting vehicle-related multimodal structured priors to guide the masked token reconstruction process, our approach can significantly enhance the model's capability to learn generalizable representations for vehicle-centric perception. Specifically, we design the Symmetry-guided Mask Module (SMM), Contour-guided Representation Module (CRM) and Semantics-guided Representation Module (SRM) to incorporate three kinds of structured priors into token reconstruction including symmetry, contour and semantics of vehicles respectively. SMM utilizes the vehicle symmetry constraints to avoid retaining symmetric patches and can thus select high-quality masked image patches and reduce information redundancy. CRM minimizes the probability distribution divergence between contour features and reconstructed features and can thus preserve holistic vehicle structure information during pixel-level reconstruction. SRM aligns image-text features through contrastive learning and cross-modal distillation to address the feature confusion caused by insufficient semantic understanding during masked reconstruction. To support the pre-training of VehicleMAE-V2, we construct Autobot4M, a large-scale dataset comprising approximately 4 million vehicle images and 12,693 text descriptions. Extensive experiments on five downstream tasks demonstrate the superior performance of VehicleMAE-V2. 

---
# Unified Brain Surface and Volume Registration 

**Authors**: S. Mazdak Abulnaga, Andrew Hoopes, Malte Hoffmann, Robin Magnet, Maks Ovsjanikov, Lilla Zöllei, John Guttag, Bruce Fischl, Adrian Dalca  

**Link**: [PDF](https://arxiv.org/pdf/2512.19928)  

**Abstract**: Accurate registration of brain MRI scans is fundamental for cross-subject analysis in neuroscientific studies. This involves aligning both the cortical surface of the brain and the interior volume. Traditional methods treat volumetric and surface-based registration separately, which often leads to inconsistencies that limit downstream analyses. We propose a deep learning framework, NeurAlign, that registers $3$D brain MRI images by jointly aligning both cortical and subcortical regions through a unified volume-and-surface-based representation. Our approach leverages an intermediate spherical coordinate space to bridge anatomical surface topology with volumetric anatomy, enabling consistent and anatomically accurate alignment. By integrating spherical registration into the learning, our method ensures geometric coherence between volume and surface domains. In a series of experiments on both in-domain and out-of-domain datasets, our method consistently outperforms both classical and machine learning-based registration methods -- improving the Dice score by up to 7 points while maintaining regular deformation fields. Additionally, it is orders of magnitude faster than the standard method for this task, and is simpler to use because it requires no additional inputs beyond an MRI scan. With its superior accuracy, fast inference, and ease of use, NeurAlign sets a new standard for joint cortical and subcortical registration. 

---
# Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning 

**Authors**: Jiayun Wu, Jiashuo Liu, Zhiyuan Zeng, Tianyang Zhan, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19920)  

**Abstract**: LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower. 

---
# A Time-efficient Prioritised Scheduling Algorithm to Optimise Initial Flock Formation of Drones 

**Authors**: Sujan Warnakulasooriya, Andreas Willig, Xiaobing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19914)  

**Abstract**: Drone applications continue to expand across various domains, with flocking offering enhanced cooperative capabilities but introducing significant challenges during initial formation. Existing flocking algorithms often struggle with efficiency and scalability, particularly when potential collisions force drones into suboptimal trajectories. This paper presents a time-efficient prioritised scheduling algorithm that improves the initial formation process of drone flocks. The method assigns each drone a priority based on its number of potential collisions and its likelihood of reaching its target position without permanently obstructing other drones. Using this hierarchy, each drone computes an appropriate delay to ensure a collision-free path. Simulation results show that the proposed algorithm successfully generates collision-free trajectories for flocks of up to 5000 drones and outperforms the coupling-degree-based heuristic prioritised planning method (CDH-PP) in both performance and computational efficiency. 

---
# Modeling Non-Ergodic Path Effects Using Conditional Generative Model for Fourier Amplitude Spectra 

**Authors**: Maxime Lacour, Pu Ren, Rie Nakata, Nori Nakata, Michael Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2512.19909)  

**Abstract**: Recent developments in non-ergodic ground-motion models (GMMs) explicitly model systematic spatial variations in source, site, and path effects, reducing standard deviation to 30-40% of ergodic models and enabling more accurate site-specific seismic hazard analysis. Current non-ergodic GMMs rely on Gaussian Process (GP) methods with prescribed correlation functions and thus have computational limitations for large-scale predictions. This study proposes a deep-learning approach called Conditional Generative Modeling for Fourier Amplitude Spectra (CGM-FAS) as an alternative to GP-based methods for modeling non-ergodic path effects in Fourier Amplitude Spectra (FAS). CGM-FAS uses a Conditional Variational Autoencoder architecture to learn spatial patterns and interfrequency correlation directly from data by using geographical coordinates of earthquakes and stations as conditional variables. Using San Francisco Bay Area earthquake data, we compare CGM-FAS against a recent GP-based GMM for the region and demonstrate consistent predictions of non-ergodic path effects. Additionally, CGM-FAS offers advantages compared to GP-based approaches in learning spatial patterns without prescribed correlation functions, capturing interfrequency correlations, and enabling rapid predictions, generating maps for 10,000 sites across 1,000 frequencies within 10 seconds using a few GB of memory. CGM-FAS hyperparameters can be tuned to ensure generated path effects exhibit variability consistent with the GP-based empirical GMM. This work demonstrates a promising direction for efficient non-ergodic ground-motion prediction across multiple frequencies and large spatial domains. 

---
# Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling 

**Authors**: Indranil Halder, Cengiz Pehlevan  

**Link**: [PDF](https://arxiv.org/pdf/2512.19905)  

**Abstract**: Recent developments in large language models have shown advantages in reallocating a notable share of computational resource from training time to inference time. However, the principles behind inference time scaling are not well understood. In this paper, we introduce an analytically tractable model of inference-time scaling: Bayesian linear regression with a reward-weighted sampler, where the reward is determined from a linear model, modeling LLM-as-a-judge scenario. We study this problem in the high-dimensional regime, where the deterministic equivalents dictate a closed-form expression for the posterior predictive mean and variance. We analyze the generalization error when training data are sampled from a teacher model. We draw $k$ inference-time samples and select via softmax at a temperature applied to a quadratic reward. When the reward is not too different from the teacher, the generalization error decreases monotonically with increasing inference time samples $k$. However, the specific reward that optimizes inference-time selection generally differs from the teacher. In contrast, substantial reward misspecification induces a finite optimal $k$ beyond which more sampling can increase the generalization error. For fixed $k$, there exists an optimal sampling temperature. We experimentally verify these facts in large language model inference with an additional large language model as a judge. In the "best-of-$k$" limit with the teacher as reward, we theoretically show that the generalization error decays as $\Theta(1/k^2)$ and determine the leading coefficient via extreme value theory. These formulas delineate domains where scaling inference-time computation is provably preferable to collecting more data. Finally, we demonstrate that when task difficulty increases, the previously mentioned advantage of inference-time compute degrades. 

---
# Fine-Tuned In-Context Learners for Efficient Adaptation 

**Authors**: Jorg Bornschein, Clare Lyle, Yazhe Li, Amal Rannen-Triki, Xu Owen He, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19879)  

**Abstract**: When adapting large language models (LLMs) to a specific downstream task, two primary approaches are commonly employed: (1) prompt engineering, often with in-context few-shot learning, leveraging the model's inherent generalization abilities, and (2) fine-tuning on task-specific data, directly optimizing the model's parameters. While prompt-based methods excel in few-shot scenarios, their effectiveness often plateaus as more data becomes available. Conversely, fine-tuning scales well with data but may underperform when training examples are scarce. We investigate a unified approach that bridges these two paradigms by incorporating in-context learning directly into the fine-tuning process. Specifically, we fine-tune the model on task-specific data augmented with in-context examples, mimicking the structure of k-shot prompts. This approach, while requiring per-task fine-tuning, combines the sample efficiency of in-context learning with the performance gains of fine-tuning, leading to a method that consistently matches and often significantly exceeds both these baselines. To perform hyperparameter selection in the low-data regime, we propose to use prequential evaluation, which eliminates the need for expensive cross-validation and leverages all available data for training while simultaneously providing a robust validation signal. We conduct an extensive empirical study to determine which adaptation paradigm - fine-tuning, in-context learning, or our proposed unified approach offers the best predictive performance on a concrete data downstream-tasks. 

---
# HARMON-E: Hierarchical Agentic Reasoning for Multimodal Oncology Notes to Extract Structured Data 

**Authors**: Shashi Kant Gupta, Arijeet Pramanik, Jerrin John Thomas, Regina Schwind, Lauren Wiener, Avi Raju, Jeremy Kornbluth, Yanshan Wang, Zhaohui Su, Hrituraj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2512.19864)  

**Abstract**: Unstructured notes within the electronic health record (EHR) contain rich clinical information vital for cancer treatment decision making and research, yet reliably extracting structured oncology data remains challenging due to extensive variability, specialized terminology, and inconsistent document formats. Manual abstraction, although accurate, is prohibitively costly and unscalable. Existing automated approaches typically address narrow scenarios - either using synthetic datasets, restricting focus to document-level extraction, or isolating specific clinical variables (e.g., staging, biomarkers, histology) - and do not adequately handle patient-level synthesis across the large number of clinical documents containing contradictory information. In this study, we propose an agentic framework that systematically decomposes complex oncology data extraction into modular, adaptive tasks. Specifically, we use large language models (LLMs) as reasoning agents, equipped with context-sensitive retrieval and iterative synthesis capabilities, to exhaustively and comprehensively extract structured clinical variables from real-world oncology notes. Evaluated on a large-scale dataset of over 400,000 unstructured clinical notes and scanned PDF reports spanning 2,250 cancer patients, our method achieves an average F1-score of 0.93, with 100 out of 103 oncology-specific clinical variables exceeding 0.85, and critical variables (e.g., biomarkers and medications) surpassing 0.95. Moreover, integration of the agentic system into a data curation workflow resulted in 0.94 direct manual approval rate, significantly reducing annotation costs. To our knowledge, this constitutes the first exhaustive, end-to-end application of LLM-based agents for structured oncology data extraction at scale 

---
# UCCL-EP: Portable Expert-Parallel Communication 

**Authors**: Ziming Mao, Yihan Zhang, Chihan Cui, Kaichao You, Zhongjie Chen, Zhiying Xu, Scott Shenker, Costin Raiciu, Yang Zhou, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2512.19849)  

**Abstract**: Mixture-of-Experts (MoE) workloads rely on expert parallelism (EP) to achieve high GPU efficiency. State-of-the-art EP communication systems such as DeepEP demonstrate strong performance but exhibit poor portability across heterogeneous GPU and NIC platforms. The poor portability is rooted in architecture: GPU-initiated token-level RDMA communication requires tight vertical integration between GPUs and NICs, e.g., GPU writes to NIC driver/MMIO interfaces.
We present UCCL-EP, a portable EP communication system that delivers DeepEP-level performance across heterogeneous GPU and NIC hardware. UCCL-EP replaces GPU-initiated RDMA with a high-throughput GPU-CPU control channel: compact token-routing commands are transferred to multithreaded CPU proxies, which then issue GPUDirect RDMA operations on behalf of GPUs. UCCL-EP further emulates various ordering semantics required by specialized EP communication modes using RDMA immediate data, enabling correctness on NICs that lack such ordering, e.g., AWS EFA. We implement UCCL-EP on NVIDIA and AMD GPUs with EFA and Broadcom NICs. On EFA, it outperforms the best existing EP solution by up to $2.1\times$ for dispatch and combine throughput. On NVIDIA-only platform, UCCL-EP achieves comparable performance to the original DeepEP. UCCL-EP also improves token throughput on SGLang by up to 40% on the NVIDIA+EFA platform, and improves DeepSeek-V3 training throughput over the AMD Primus/Megatron-LM framework by up to 45% on a 16-node AMD+Broadcom platform. 

---
# Learned Digital Codes for Over-the-Air Computation in Federated Edge Learning 

**Authors**: Antonio Tarizzo, Mohammad Kazemi, Deniz Gündüz  

**Link**: [PDF](https://arxiv.org/pdf/2512.19777)  

**Abstract**: Federated edge learning (FEEL) enables wireless devices to collaboratively train a centralised model without sharing raw data, but repeated uplink transmission of model updates makes communication the dominant bottleneck. Over-the-air (OTA) aggregation alleviates this by exploiting the superposition property of the wireless channel, enabling simultaneous transmission and merging communication with computation. Digital OTA schemes extend this principle by incorporating the robustness of conventional digital communication, but current designs remain limited in low signal-to-noise ratio (SNR) regimes. This work proposes a learned digital OTA framework that improves recovery accuracy, convergence behaviour, and robustness to challenging SNR conditions while maintaining the same uplink overhead as state-of-the-art methods. The design integrates an unsourced random access (URA) codebook with vector quantisation and AMP-DA-Net, an unrolled approximate message passing (AMP)-style decoder trained end-to-end with the digital codebook and parameter server local training statistics. The proposed design extends OTA aggregation beyond averaging to a broad class of symmetric functions, including trimmed means and majority-based rules. Experiments on highly heterogeneous device datasets and varying numbers of active devices show that the proposed design extends reliable digital OTA operation by more than 10 dB into low SNR regimes while matching or improving performance across the full SNR range. The learned decoder remains effective under message corruption and nonlinear aggregation, highlighting the broader potential of end-to-end learned design for digital OTA communication in FEEL. 

---
# A K-Means, Ward and DBSCAN repeatability study 

**Authors**: Anthony Bertrand, Engelbert Mephu Nguifo, Violaine Antoine, David Hill  

**Link**: [PDF](https://arxiv.org/pdf/2512.19772)  

**Abstract**: Reproducibility is essential in machine learning because it ensures that a model or experiment yields the same scientific conclusion. For specific algorithms repeatability with bitwise identical results is also a key for scientific integrity because it allows debugging. We decomposed several very popular clustering algorithms: K-Means, DBSCAN and Ward into their fundamental steps, and we identify the conditions required to achieve repeatability at each stage. We use an implementation example with the Python library scikit-learn to examine the repeatable aspects of each method. Our results reveal inconsistent results with K-Means when the number of OpenMP threads exceeds two. This work aims to raise awareness of this issue among both users and developers, encouraging further investigation and potential fixes. 

---
# A Declarative Language for Building And Orchestrating LLM-Powered Agent Workflows 

**Authors**: Ivan Daunis  

**Link**: [PDF](https://arxiv.org/pdf/2512.19769)  

**Abstract**: Building deployment-ready LLM agents requires complex orchestration of tools, data sources, and control flow logic, yet existing systems tightly couple agent logic to specific programming languages and deployment models. We present a declarative system that separates agent workflow specification from implementation, enabling the same pipeline definition to execute across multiple backend languages (Java, Python, Go) and deployment environments (cloud-native, on-premises).
Our key insight is that most agent workflows consist of common patterns -- data serialization, filtering, RAG retrieval, API orchestration -- that can be expressed through a unified DSL rather than imperative code. This approach transforms agent development from application programming to configuration, where adding new tools or fine-tuning agent behaviors requires only pipeline specification changes, not code deployment. Our system natively supports A/B testing of agent strategies, allowing multiple pipeline variants to run on the same backend infrastructure with automatic metric collection and comparison.
We evaluate our approach on real-world e-commerce workflows at PayPal, processing millions of daily interactions. Our results demonstrate 60% reduction in development time, and 3x improvement in deployment velocity compared to imperative implementations. The language's declarative approach enables non-engineers to modify agent behaviors safely, while maintaining sub-100ms orchestration overhead. We show that complex workflows involving product search, personalization, and cart management can be expressed in under 50 lines of DSL compared to 500+ lines of imperative code. 

---
# How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts 

**Authors**: Sumin Park, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.19765)  

**Abstract**: Finding the optimal configuration of Sparse Mixture-ofExperts (SMoE) that maximizes semantic differentiation among experts is essential for exploiting the full potential of MoE architectures. However, existing SMoE frameworks either heavily rely on hyperparameter tuning or overlook the importance of diversifying semantic roles across experts when adapting the expert pool size. We propose Mixture-of-Experts for Adaptive Semantic Specialization (MASS), a semanticaware MoE framework for adaptive expert expansion and dynamic routing. MASS introduces two key advancements: (i) a gradient-based semantic drift detector that prompts targeted expert expansion when the existing expert pool lacks capacity to capture the full semantic diversity of the data, and (ii) an integration of adaptive routing strategy that dynamically adjusts expert usage based on token-level routing confidence mass. We first demonstrate that MASS reliably converges to the point of optimal balance between cost-performance trade-off with notably improved sematic specialization in a highly controlled synthetic setup. Further empirical results on real-world datasets across language and vision domains show that MASS consistently outperforms a range of strong MoE baselines, demonstrating its domain robustness and enhanced expert specialization. 

---
# Attention Distance: A Novel Metric for Directed Fuzzing with Large Language Models 

**Authors**: Wang Bin, Ao Yang, Kedan Li, Aofan Liu, Hui Li, Guibo Luo, Weixiang Huang, Yan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19758)  

**Abstract**: In the domain of software security testing, Directed Grey-Box Fuzzing (DGF) has garnered widespread attention for its efficient target localization and excellent detection performance. However, existing approaches measure only the physical distance between seed execution paths and target locations, overlooking logical relationships among code segments. This omission can yield redundant or misleading guidance in complex binaries, weakening DGF's real-world effectiveness. To address this, we introduce \textbf{attention distance}, a novel metric that leverages a large language model's contextual analysis to compute attention scores between code elements and reveal their intrinsic connections. Under the same AFLGo configuration -- without altering any fuzzing components other than the distance metric -- replacing physical distances with attention distances across 38 real vulnerability reproduction experiments delivers a \textbf{3.43$\times$} average increase in testing efficiency over the traditional method. Compared to state-of-the-art directed fuzzers DAFL and WindRanger, our approach achieves \textbf{2.89$\times$} and \textbf{7.13$\times$} improvements, respectively. To further validate the generalizability of attention distance, we integrate it into DAFL and WindRanger, where it also consistently enhances their original performance. All related code and datasets are publicly available at this https URL\this http URL. 

---
# QMBench: A Research Level Benchmark for Quantum Materials Research 

**Authors**: Yanzhen Wang, Yiyang Jiang, Diana Golovanova, Kamal Das, Hyeonhu Bae, Yufei Zhao, Huu-Thong Le, Abhinava Chatterjee, Yunzhe Liu, Chao-Xing Liu, Felipe H. da Jornada, Binghai Yan, Xiao-Liang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2512.19753)  

**Abstract**: We introduce QMBench, a comprehensive benchmark designed to evaluate the capability of large language model agents in quantum materials research. This specialized benchmark assesses the model's ability to apply condensed matter physics knowledge and computational techniques such as density functional theory to solve research problems in quantum materials science. QMBench encompasses different domains of the quantum material research, including structural properties, electronic properties, thermodynamic and other properties, symmetry principle and computational methodologies. By providing a standardized evaluation framework, QMBench aims to accelerate the development of an AI scientist capable of making creative contributions to quantum materials research. We expect QMBench to be developed and constantly improved by the research community. 

---
# From Theory to Throughput: CUDA-Optimized APML for Large-Batch 3D Learning 

**Authors**: Sasan Sharifipour, Constantino Álvarez Casado, Manuel Lage Cañellas, Miguel Bordallo López  

**Link**: [PDF](https://arxiv.org/pdf/2512.19743)  

**Abstract**: Loss functions are fundamental to learning accurate 3D point cloud models, yet common choices trade geometric fidelity for computational cost. Chamfer Distance is efficient but permits many-to-one correspondences, while Earth Mover Distance better reflects one-to-one transport at high computational cost. APML approximates transport with differentiable Sinkhorn iterations and an analytically derived temperature, but its dense formulation scales quadratically in memory. We present CUDA-APML, a sparse GPU implementation that thresholds negligible assignments and runs adaptive softmax, bidirectional symmetrization, and Sinkhorn normalization directly in COO form. This yields near-linear memory scaling and preserves gradients on the stored support, while pairwise distance evaluation remains quadratic in the current implementation. On ShapeNet and MM-Fi, CUDA-APML matches dense APML within a small tolerance while reducing peak GPU memory by 99.9%. Code available at: this https URL 

---
# Simulation-Driven Railway Delay Prediction: An Imitation Learning Approach 

**Authors**: Clément Elliker, Jesse Read, Sonia Vanier, Albert Bifet  

**Link**: [PDF](https://arxiv.org/pdf/2512.19737)  

**Abstract**: Reliable prediction of train delays is essential for enhancing the robustness and efficiency of railway transportation systems. In this work, we reframe delay forecasting as a stochastic simulation task, modeling state-transition dynamics through imitation learning. We introduce Drift-Corrected Imitation Learning (DCIL), a novel self-supervised algorithm that extends DAgger by incorporating distance-based drift correction, thereby mitigating covariate shift during rollouts without requiring access to an external oracle or adversarial schemes. Our approach synthesizes the dynamical fidelity of event-driven models with the representational capacity of data-driven methods, enabling uncertainty-aware forecasting via Monte Carlo simulation. We evaluate DCIL using a comprehensive real-world dataset from \textsc{Infrabel}, the Belgian railway infrastructure manager, which encompasses over three million train movements. Our results, focused on predictions up to 30 minutes ahead, demonstrate superior predictive performance of DCIL over traditional regression models and behavioral cloning on deep learning architectures, highlighting its effectiveness in capturing the sequential and uncertain nature of delay propagation in large-scale networks. 

---
# CoPHo: Classifier-guided Conditional Topology Generation with Persistent Homology 

**Authors**: Gongli Xi, Ye Tian, Mengyu Yang, Zhenyu Zhao, Yuchao Zhang, Xiangyang Gong, Xirong Que, Wendong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19736)  

**Abstract**: The structure of topology underpins much of the research on performance and robustness, yet available topology data are typically scarce, necessitating the generation of synthetic graphs with desired properties for testing or release. Prior diffusion-based approaches either embed conditions into the diffusion model, requiring retraining for each attribute and hindering real-time applicability, or use classifier-based guidance post-training, which does not account for topology scale and practical constraints. In this paper, we show from a discrete perspective that gradients from a pre-trained graph-level classifier can be incorporated into the discrete reverse diffusion posterior to steer generation toward specified structural properties. Based on this insight, we propose Classifier-guided Conditional Topology Generation with Persistent Homology (CoPHo), which builds a persistent homology filtration over intermediate graphs and interprets features as guidance signals that steer generation toward the desired properties at each denoising step. Experiments on four generic/network datasets demonstrate that CoPHo outperforms existing methods at matching target metrics, and we further validate its transferability on the QM9 molecular dataset. 

---
# High-Performance Self-Supervised Learning by Joint Training of Flow Matching 

**Authors**: Kosuke Ukita, Tsuyoshi Okita  

**Link**: [PDF](https://arxiv.org/pdf/2512.19729)  

**Abstract**: Diffusion models can learn rich representations during data generation, showing potential for Self-Supervised Learning (SSL), but they face a trade-off between generative quality and discriminative performance. Their iterative sampling also incurs substantial computational and energy costs, hindering industrial and edge AI applications. To address these issues, we propose the Flow Matching-based Foundation Model (FlowFM), which jointly trains a representation encoder and a conditional flow matching generator. This decoupled design achieves both high-fidelity generation and effective recognition. By using flow matching to learn a simpler velocity field, FlowFM accelerates and stabilizes training, improving its efficiency for representation learning. Experiments on wearable sensor data show FlowFM reduces training time by 50.4\% compared to a diffusion-based approach. On downstream tasks, FlowFM surpassed the state-of-the-art SSL method (SSL-Wearables) on all five datasets while achieving up to a 51.0x inference speedup and maintaining high generative quality. The implementation code is available at this https URL. 

---
# Tiny, On-Device Decision Makers with the MiniConv Library 

**Authors**: Carlos Purves  

**Link**: [PDF](https://arxiv.org/pdf/2512.19726)  

**Abstract**: Reinforcement learning (RL) has achieved strong results, but deploying visual policies on resource-constrained edge devices remains challenging due to computational cost and communication latency. Many deployments therefore offload policy inference to a remote server, incurring network round trips and requiring transmission of high-dimensional observations. We introduce a split-policy architecture in which a small on-device encoder, implemented as OpenGL fragment-shader passes for broad embedded GPU support, transforms each observation into a compact feature tensor that is transmitted to a remote policy head. In RL, this communication overhead manifests as closed-loop decision latency rather than only per-request inference latency. The proposed approach reduces transmitted data, lowers decision latency in bandwidth-limited settings, and reduces server-side compute per request, whilst achieving broadly comparable learning performance by final return (mean over the final 100 episodes) in single-run benchmarks, with modest trade-offs in mean return. We evaluate across an NVIDIA Jetson Nano, a Raspberry Pi 4B, and a Raspberry Pi Zero 2 W, reporting learning results, on-device execution behaviour under sustained load, and end-to-end decision latency and scalability measurements under bandwidth shaping. Code for training, deployment, and measurement is released as open source. 

---
# Multiscale Dual-path Feature Aggregation Network for Remaining Useful Life Prediction of Lithium-Ion Batteries 

**Authors**: Zihao Lv, Siqi Ai, Yanbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19719)  

**Abstract**: Targeted maintenance strategies, ensuring the dependability and safety of industrial machinery. However, current modeling techniques for assessing both local and global correlation of battery degradation sequences are inefficient and difficult to meet the needs in real-life applications. For this reason, we propose a novel deep learning architecture, multiscale dual-path feature aggregation network (MDFA-Net), for RUL prediction. MDFA-Net consists of dual-path networks, the first path network, multiscale feature network (MF-Net) that maintains the shallow information and avoids missing information, and the second path network is an encoder network (EC-Net) that captures the continuous trend of the sequences and retains deep details. Integrating both deep and shallow attributes effectively grasps both local and global patterns. Testing conducted with two publicly available Lithium-ion battery datasets reveals our approach surpasses existing top-tier methods in RUL forecasting, accurately mapping the capacity degradation trajectory. 

---
# Thermodynamic Focusing for Inference-Time Search: Practical Methods for Target-Conditioned Sampling and Prompted Inference 

**Authors**: Zhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19717)  

**Abstract**: Finding rare but useful solutions in very large candidate spaces is a recurring practical challenge across language generation, planning, and reinforcement learning. We present a practical framework, \emph{Inverted Causality Focusing Algorithm} (ICFA), that treats search as a target-conditioned reweighting process. ICFA reuses an available proposal sampler and a task-specific similarity function to form a focused sampling distribution, while adaptively controlling focusing strength to avoid degeneracy. We provide a clear recipe, a stability diagnostic based on effective sample size, a compact theoretical sketch explaining when ICFA can reduce sample needs, and two reproducible experiments: constrained language generation and sparse-reward navigation. We further show how structured prompts instantiate an approximate, language-level form of ICFA and describe a hybrid architecture combining prompted inference with algorithmic reweighting. 

---
# Development and external validation of a multimodal artificial intelligence mortality prediction model of critically ill patients using multicenter data 

**Authors**: Behrooz Mamandipoor, Chun-Nan Hsu, Martin Krause, Ulrich H. Schmidt, Rodney A. Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2512.19716)  

**Abstract**: Early prediction of in-hospital mortality in critically ill patients can aid clinicians in optimizing treatment. The objective was to develop a multimodal deep learning model, using structured and unstructured clinical data, to predict in-hospital mortality risk among critically ill patients after their initial 24 hour intensive care unit (ICU) admission. We used data from MIMIC-III, MIMIC-IV, eICU, and HiRID. A multimodal model was developed on the MIMIC datasets, featuring time series components occurring within the first 24 hours of ICU admission and predicting risk of subsequent inpatient mortality. Inputs included time-invariant variables, time-variant variables, clinical notes, and chest X-ray images. External validation occurred in a temporally separated MIMIC population, HiRID, and eICU datasets. A total of 203,434 ICU admissions from more than 200 hospitals between 2001 to 2022 were included, in which mortality rate ranged from 5.2% to 7.9% across the four datasets. The model integrating structured data points had AUROC, AUPRC, and Brier scores of 0.92, 0.53, and 0.19, respectively. We externally validated the model on eight different institutions within the eICU dataset, demonstrating AUROCs ranging from 0.84-0.92. When including only patients with available clinical notes and imaging data, inclusion of notes and imaging into the model, the AUROC, AUPRC, and Brier score improved from 0.87 to 0.89, 0.43 to 0.48, and 0.37 to 0.17, respectively. Our findings highlight the importance of incorporating multiple sources of patient information for mortality prediction and the importance of external validation. 

---
# PHANTOM: PHysical ANamorphic Threats Obstructing Connected Vehicle Mobility 

**Authors**: Md Nahid Hasan Shuvo, Moinul Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2512.19711)  

**Abstract**: Connected autonomous vehicles (CAVs) rely on vision-based deep neural networks (DNNs) and low-latency (Vehicle-to-Everything) V2X communication to navigate safely and efficiently. Despite their advances, these systems remain vulnerable to physical adversarial attacks. In this paper, we introduce PHANTOM (PHysical ANamorphic Threats Obstructing connected vehicle Mobility), a novel framework for crafting and deploying perspective-dependent adversarial examples using \textit{anamorphic art}. PHANTOM exploits geometric distortions that appear natural to humans but are misclassified with high confidence by state-of-the-art object detectors. Unlike conventional attacks, PHANTOM operates in black-box settings without model access and demonstrates strong transferability across four diverse detector architectures (YOLOv5, SSD, Faster R-CNN, and RetinaNet). Comprehensive evaluation in CARLA across varying speeds, weather conditions, and lighting scenarios shows that PHANTOM achieves over 90\% attack success rate under optimal conditions and maintains 60-80\% effectiveness even in degraded environments. The attack activates within 6-10 meters of the target, providing insufficient time for safe maneuvering. Beyond individual vehicle deception, PHANTOM triggers network-wide disruption in CAV systems: SUMO-OMNeT++ co-simulation demonstrates that false emergency messages propagate through V2X links, increasing Peak Age of Information by 68-89\% and degrading safety-critical communication. These findings expose critical vulnerabilities in both perception and communication layers of CAV ecosystems. 

---
# Bidirectional human-AI collaboration in brain tumour assessments improves both expert human and AI agent performance 

**Authors**: James K Ruffle, Samia Mohinta, Guilherme Pombo, Asthik Biswas, Alan Campbell, Indran Davagnanam, David Doig, Ahmed Hamman, Harpreet Hyare, Farrah Jabeen, Emma Lim, Dermot Mallon, Stephanie Owen, Sophie Wilkinson, Sebastian Brandner, Parashkev Nachev  

**Link**: [PDF](https://arxiv.org/pdf/2512.19707)  

**Abstract**: The benefits of artificial intelligence (AI) human partnerships-evaluating how AI agents enhance expert human performance-are increasingly studied. Though rarely evaluated in healthcare, an inverse approach is possible: AI benefiting from the support of an expert human agent. Here, we investigate both human-AI clinical partnership paradigms in the magnetic resonance imaging-guided characterisation of patients with brain tumours. We reveal that human-AI partnerships improve accuracy and metacognitive ability not only for radiologists supported by AI, but also for AI agents supported by radiologists. Moreover, the greatest patient benefit was evident with an AI agent supported by a human one. Synergistic improvements in agent accuracy, metacognitive performance, and inter-rater agreement suggest that AI can create more capable, confident, and consistent clinical agents, whether human or model-based. Our work suggests that the maximal value of AI in healthcare could emerge not from replacing human intelligence, but from AI agents that routinely leverage and amplify it. 

---
# Generative AI for Analysts 

**Authors**: Jian Xue, Qian Zhang, Wu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19705)  

**Abstract**: We study how generative artificial intelligence (AI) transforms the work of financial analysts. Using the 2023 launch of FactSet's AI platform as a natural experiment, we find that adoption produces markedly richer and more comprehensive reports -- featuring 40% more distinct information sources, 34% broader topical coverage, and 25% greater use of advanced analytical methods -- while also improving timeliness. However, forecast errors rise by 59% as AI-assisted reports convey a more balanced mix of positive and negative information that is harder to synthesize, particularly for analysts facing heavier cognitive demands. Placebo tests using other data vendors confirm that these effects are unique to FactSet's AI integration. Overall, our findings reveal both the productivity gains and cognitive limits of generative AI in financial information production. 

---
# Large Language Models for EDA Cloud Job Resource and Lifetime Prediction 

**Authors**: Yuxuan Yin, Shengke Zhou, Yunjie Zhang, Ajay Mohindra, Boxun Xu, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.19701)  

**Abstract**: The rapid growth of cloud computing in the Electronic Design Automation (EDA) industry has created a critical need for resource and job lifetime prediction to achieve optimal scheduling. Traditional machine learning methods often struggle with the complexity and heterogeneity of EDA workloads, requiring extensive feature engineering and domain expertise. We propose a novel framework that fine-tunes Large Language Models (LLMs) to address this challenge through text-to-text regression. We introduce the scientific notation and prefix filling to constrain the LLM, significantly improving output format reliability. Moreover, we found that full-attention finetuning and inference improves the prediction accuracy of sliding-window-attention LLMs. We demonstrate the effectiveness of our proposed framework on real-world cloud datasets, setting a new baseline for performance prediction in the EDA domain. 

---
# Automated Fault Detection in 5G Core Networks Using Large Language Models 

**Authors**: Parsa Hatami, Ahmadreza Majlesara, Ali Majlesi, Babak Hossein Khalaj  

**Link**: [PDF](https://arxiv.org/pdf/2512.19697)  

**Abstract**: With the rapid growth of data volume in modern telecommunication networks and the continuous expansion of their scale, maintaining high reliability has become a critical requirement. These networks support a wide range of applications and services, including highly sensitive and mission-critical ones, which demand rapid and accurate detection and resolution of network errors. Traditional fault-diagnosis methods are no longer efficient for such complex environments.\cite{b1} In this study, we leverage Large Language Models (LLMs) to automate network fault detection and classification. Various types of network errors were intentionally injected into a Kubernetes-based test network, and data were collected under both healthy and faulty conditions. The dataset includes logs from different network components (pods), along with complementary data such as system descriptions, events, Round Trip Time (RTT) tests, and pod status information. The dataset covers common fault types such as pod failure, pod kill, network delay, network loss, and disk I/O failures. We fine-tuned the GPT-4.1 nano model via its API on this dataset, resulting in a significant improvement in fault-detection accuracy compared to the base model. These findings highlight the potential of LLM-based approaches for achieving closed-loop, and operator-free fault management, which can enhance network reliability and reduce downtime-related operational costs for service providers. 

---
# QoS-Aware Dynamic CU Selection in O-RAN with Graph-Based Reinforcement Learning 

**Authors**: Sebastian Racedo, Brigitte Jaumard, Oscar Delgado, Meysam Masoudi  

**Link**: [PDF](https://arxiv.org/pdf/2512.19696)  

**Abstract**: Open Radio Access Network (O RAN) disaggregates conventional RAN into interoperable components, enabling flexible resource allocation, energy savings, and agile architectural design. In legacy deployments, the binding between logical functions and physical locations is static, which leads to inefficiencies under time varying traffic and resource conditions. We address this limitation by relaxing the fixed mapping and performing dynamic service function chain (SFC) provisioning with on the fly O CU selection. We formulate the problem as a Markov decision process and solve it using GRLDyP, i.e., a graph neural network (GNN) assisted deep reinforcement learning (DRL). The proposed agent jointly selects routes and the O-CU location (from candidate sites) for each incoming service flow to minimize network energy consumption while satisfying quality of service (QoS) constraints. The GNN encodes the instantaneous network topology and resource utilization (e.g., CPU and bandwidth), and the DRL policy learns to balance grade of service, latency, and energy. We perform the evaluation of GRLDyP on a data set with 24-hour traffic traces from the city of Montreal, showing that dynamic O CU selection and routing significantly reduce energy consumption compared to a static mapping baseline, without violating QoS. The results highlight DRL based SFC provisioning as a practical control primitive for energy-aware, resource-adaptive O-RAN deployments. 

---
# Brain-Grounded Axes for Reading and Steering LLM States 

**Authors**: Sandro Andric  

**Link**: [PDF](https://arxiv.org/pdf/2512.19399)  

**Abstract**: Interpretability methods for large language models (LLMs) typically derive directions from textual supervision, which can lack external grounding. We propose using human brain activity not as a training signal but as a coordinate system for reading and steering LLM states. Using the SMN4Lang MEG dataset, we construct a word-level brain atlas of phase-locking value (PLV) patterns and extract latent axes via ICA. We validate axes with independent lexica and NER-based labels (POS/log-frequency used as sanity checks), then train lightweight adapters that map LLM hidden states to these brain axes without fine-tuning the LLM. Steering along the resulting brain-derived directions yields a robust lexical (frequency-linked) axis in a mid TinyLlama layer, surviving perplexity-matched controls, and a brain-vs-text probe comparison shows larger log-frequency shifts (relative to the text probe) with lower perplexity for the brain axis. A function/content axis (axis 13) shows consistent steering in TinyLlama, Qwen2-0.5B, and GPT-2, with PPL-matched text-level corroboration. Layer-4 effects in TinyLlama are large but inconsistent, so we treat them as secondary (Appendix). Axis structure is stable when the atlas is rebuilt without GPT embedding-change features or with word2vec embeddings (|r|=0.64-0.95 across matched axes), reducing circularity concerns. Exploratory fMRI anchoring suggests potential alignment for embedding change and log frequency, but effects are sensitive to hemodynamic modeling assumptions and are treated as population-level evidence only. These results support a new interface: neurophysiology-grounded axes provide interpretable and controllable handles for LLM behavior. 

---
