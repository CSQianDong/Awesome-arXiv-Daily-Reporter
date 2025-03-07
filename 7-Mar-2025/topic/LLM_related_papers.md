# Measuring temporal effects of agent knowledge by date-controlled tool use 

**Authors**: R. Patrick Xian, Qiming Cui, Stefan Bauer, Reza Abbasi-Asl  

**Link**: [PDF](https://arxiv.org/pdf/2503.04188)  

**Abstract**: Temporal progression is an integral part of knowledge accumulation and update. Web search is frequently adopted as grounding for agent knowledge, yet its inappropriate configuration affects the quality of agent responses. Here, we construct a tool-based out-of-sample testing framework to measure the knowledge variability of large language model (LLM) agents from distinct date-controlled tools (DCTs). We demonstrate the temporal effects of an LLM agent as a writing assistant, which can use web search to help complete scientific publication abstracts. We show that temporal effects of the search engine translates into tool-dependent agent performance but can be alleviated with base model choice and explicit reasoning instructions such as chain-of-thought prompting. Our results indicate that agent evaluation should take a dynamical view and account for the temporal influence of tools and the updates of external resources. 

---
# Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation 

**Authors**: Ziqiang Cui, Yunpeng Weng, Xing Tang, Xiaokun Zhang, Dugang Liu, Shiwei Li, Peiyang Liu, Bowei He, Weihong Luo, Xiuqiang He, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04162)  

**Abstract**: Sequential recommendation aims to model user preferences based on historical behavior sequences, which is crucial for various online platforms. Data sparsity remains a significant challenge in this area as most users have limited interactions and many items receive little attention. To mitigate this issue, contrastive learning has been widely adopted. By constructing positive sample pairs from the data itself and maximizing their agreement in the embedding space,it can leverage available data more effectively. Constructing reasonable positive sample pairs is crucial for the success of contrastive learning. However, current approaches struggle to generate reliable positive pairs as they either rely on representations learned from inherently sparse collaborative signals or use random perturbations which introduce significant uncertainty. To address these limitations, we propose a novel approach named Semantic Retrieval Augmented Contrastive Learning (SRA-CL), which leverages semantic information to improve the reliability of contrastive samples. SRA-CL comprises two main components: (1) Cross-Sequence Contrastive Learning via User Semantic Retrieval, which utilizes large language models (LLMs) to understand diverse user preferences and retrieve semantically similar users to form reliable positive samples through a learnable sample synthesis method; and (2) Intra-Sequence Contrastive Learning via Item Semantic Retrieval, which employs LLMs to comprehend items and retrieve similar items to perform semantic-based item substitution, thereby creating semantically consistent augmented views for contrastive learning. SRA-CL is plug-and-play and can be integrated into standard sequential recommendation models. Extensive experiments on four public datasets demonstrate the effectiveness and generalizability of the proposed approach. 

---
# Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search 

**Authors**: Kou Misaki, Yuichi Inoue, Yuki Imajuku, So Kuroki, Taishi Nakamura, Takuya Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2503.04412)  

**Abstract**: Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose $\textit{Adaptive Branching Monte Carlo Tree Search (AB-MCTS)}$, a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling. 

---
# ToolFuzz -- Automated Agent Tool Testing 

**Authors**: Ivan Milev, Mislav Balunović, Maximilian Baader, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2503.04479)  

**Abstract**: Large Language Model (LLM) Agents leverage the advanced reasoning capabilities of LLMs in real-world applications. To interface with an environment, these agents often rely on tools, such as web search or database APIs. As the agent provides the LLM with tool documentation along the user query, the completeness and correctness of this documentation is critical. However, tool documentation is often over-, under-, or ill-specified, impeding the agent's accuracy. Standard software testing approaches struggle to identify these errors as they are expressed in natural language. Thus, despite its importance, there currently exists no automated method to test the tool documentation for agents. To address this issue, we present ToolFuzz, the first method for automated testing of tool documentations. ToolFuzz is designed to discover two types of errors: (1) user queries leading to tool runtime errors and (2) user queries that lead to incorrect agent responses. ToolFuzz can generate a large and diverse set of natural inputs, effectively finding tool description errors at a low false positive rate. Further, we present two straightforward prompt-engineering approaches. We evaluate all three tool testing approaches on 32 common LangChain tools and 35 newly created custom tools and 2 novel benchmarks to further strengthen the assessment. We find that many publicly available tools suffer from underspecification. Specifically, we show that ToolFuzz identifies 20x more erroneous inputs compared to the prompt-engineering approaches, making it a key component for building reliable AI agents. 

---
# Benchmarking Reasoning Robustness in Large Language Models 

**Authors**: Tong Yu, Yongcheng Jing, Xikun Zhang, Wentao Jiang, Wenjie Wu, Yingjie Wang, Wenbin Hu, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04550)  

**Abstract**: Despite the recent success of large language models (LLMs) in reasoning such as DeepSeek, we for the first time identify a key dilemma in reasoning robustness and generalization: significant performance degradation on novel or incomplete data, suggesting a reliance on memorized patterns rather than systematic reasoning. Our closer examination reveals four key unique limitations underlying this issue:(1) Positional bias--models favor earlier queries in multi-query inputs but answering the wrong one in the latter (e.g., GPT-4o's accuracy drops from 75.8 percent to 72.8 percent); (2) Instruction sensitivity--performance declines by 5.0 to 7.5 percent in the Qwen2.5 Series and by 5.0 percent in DeepSeek-V3 with auxiliary guidance; (3) Numerical fragility--value substitution sharply reduces accuracy (e.g., GPT-4o drops from 97.5 percent to 82.5 percent, GPT-o1-mini drops from 97.5 percent to 92.5 percent); and (4) Memory dependence--models resort to guesswork when missing critical data. These findings further highlight the reliance on heuristic recall over rigorous logical inference, demonstrating challenges in reasoning robustness. To comprehensively investigate these robustness challenges, this paper introduces a novel benchmark, termed as Math-RoB, that exploits hallucinations triggered by missing information to expose reasoning gaps. This is achieved by an instruction-based approach to generate diverse datasets that closely resemble training distributions, facilitating a holistic robustness assessment and advancing the development of more robust reasoning frameworks. Bad character(s) in field Abstract. 

---
# SOLAR: Scalable Optimization of Large-scale Architecture for Reasoning 

**Authors**: Chen Li, Yinyi Luo, Anudeep Bolimera, Marios Savvides  

**Link**: [PDF](https://arxiv.org/pdf/2503.04530)  

**Abstract**: Large Language Models (LLMs) excel in reasoning but remain constrained by their Chain-of-Thought (CoT) approach, which struggles with complex tasks requiring more nuanced topological reasoning. We introduce SOLAR, Scalable Optimization of Large-scale Architecture for Reasoning, a framework that dynamically optimizes various reasoning topologies to enhance accuracy and efficiency.
Our Topological Annotation Generation (TAG) system automates topological dataset creation and segmentation, improving post-training and evaluation. Additionally, we propose Topological-Scaling, a reward-driven framework that aligns training and inference scaling, equipping LLMs with adaptive, task-aware reasoning.
SOLAR achieves substantial gains on MATH and GSM8K: +5% accuracy with Topological Tuning, +9% with Topological Reward, and +10.02% with Hybrid Scaling. It also reduces response length by over 5% for complex problems, lowering inference latency.
To foster the reward system, we train a multi-task Topological Reward Model (M-TRM), which autonomously selects the best reasoning topology and answer in a single pass, eliminating the need for training and inference on multiple single-task TRMs (S-TRMs), thus reducing both training cost and inference latency. In addition, in terms of performance, M-TRM surpasses all S-TRMs, improving accuracy by +10% and rank correlation by +9%.
To the best of our knowledge, SOLAR sets a new benchmark for scalable, high-precision LLM reasoning while introducing an automated annotation process and a dynamic reasoning topology competition mechanism. 

---
# VirtualXAI: A User-Centric Framework for Explainability Assessment Leveraging GPT-Generated Personas 

**Authors**: Georgios Makridis, Vasileios Koukos, Georgios Fatouros, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2503.04261)  

**Abstract**: In today's data-driven era, computational systems generate vast amounts of data that drive the digital transformation of industries, where Artificial Intelligence (AI) plays a key role. Currently, the demand for eXplainable AI (XAI) has increased to enhance the interpretability, transparency, and trustworthiness of AI models. However, evaluating XAI methods remains challenging: existing evaluation frameworks typically focus on quantitative properties such as fidelity, consistency, and stability without taking into account qualitative characteristics such as satisfaction and interpretability. In addition, practitioners face a lack of guidance in selecting appropriate datasets, AI models, and XAI methods -a major hurdle in human-AI collaboration. To address these gaps, we propose a framework that integrates quantitative benchmarking with qualitative user assessments through virtual personas based on the "Anthology" of backstories of the Large Language Model (LLM). Our framework also incorporates a content-based recommender system that leverages dataset-specific characteristics to match new input data with a repository of benchmarked datasets. This yields an estimated XAI score and provides tailored recommendations for both the optimal AI model and the XAI method for a given scenario. 

---
# MathMistake Checker: A Comprehensive Demonstration for Step-by-Step Math Problem Mistake Finding by Prompt-Guided LLMs 

**Authors**: Tianyang Zhang, Zhuoxuan Jiang, Haotian Zhang, Lin Lin, Shaohua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04291)  

**Abstract**: We propose a novel system, MathMistake Checker, designed to automate step-by-step mistake finding in mathematical problems with lengthy answers through a two-stage process. The system aims to simplify grading, increase efficiency, and enhance learning experiences from a pedagogical perspective. It integrates advanced technologies, including computer vision and the chain-of-thought capabilities of the latest large language models (LLMs). Our system supports open-ended grading without reference answers and promotes personalized learning by providing targeted feedback. We demonstrate its effectiveness across various types of math problems, such as calculation and word problems. 

---
# TIMER: Temporal Instruction Modeling and Evaluation for Longitudinal Clinical Records 

**Authors**: Hejie Cui, Alyssa Unell, Bowen Chen, Jason Alan Fries, Emily Alsentzer, Sanmi Koyejo, Nigam Shah  

**Link**: [PDF](https://arxiv.org/pdf/2503.04176)  

**Abstract**: Large language models (LLMs) have emerged as promising tools for assisting in medical tasks, yet processing Electronic Health Records (EHRs) presents unique challenges due to their longitudinal nature. While LLMs' capabilities to perform medical tasks continue to improve, their ability to reason over temporal dependencies across multiple patient visits and time frames remains unexplored. We introduce TIMER (Temporal Instruction Modeling and Evaluation for Longitudinal Clinical Records), a framework that incorporate instruction-response pairs grounding to different parts of a patient's record as a critical dimension in both instruction evaluation and tuning for longitudinal clinical records. We develop TIMER-Bench, the first time-aware benchmark that evaluates temporal reasoning capabilities over longitudinal EHRs, as well as TIMER-Instruct, an instruction-tuning methodology for LLMs to learn reasoning over time. We demonstrate that models fine-tuned with TIMER-Instruct improve performance by 7.3% on human-generated benchmarks and 9.2% on TIMER-Bench, indicating that temporal instruction-tuning improves model performance for reasoning over EHR. 

---
# Activation Space Interventions Can Be Transferred Between Large Language Models 

**Authors**: Narmeen Oozeer, Dhruv Nathawani, Nirmalendu Prakash, Michael Lan, Abir Harrasse, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2503.04429)  

**Abstract**: The study of representation universality in AI models reveals growing convergence across domains, modalities, and architectures. However, the practical applications of representation universality remain largely unexplored. We bridge this gap by demonstrating that safety interventions can be transferred between models through learned mappings of their shared activation spaces. We demonstrate this approach on two well-established AI safety tasks: backdoor removal and refusal of harmful prompts, showing successful transfer of steering vectors that alter the models' outputs in a predictable way. Additionally, we propose a new task, \textit{corrupted capabilities}, where models are fine-tuned to embed knowledge tied to a backdoor. This tests their ability to separate useful skills from backdoors, reflecting real-world challenges. Extensive experiments across Llama, Qwen and Gemma model families show that our method enables using smaller models to efficiently align larger ones. Furthermore, we demonstrate that autoencoder mappings between base and fine-tuned models can serve as reliable ``lightweight safety switches", allowing dynamic toggling between model behaviors. 

---
# Enough Coin Flips Can Make LLMs Act Bayesian 

**Authors**: Ritwik Gupta, Rodolfo Corona, Jiaxin Ge, Eric Wang, Dan Klein, Trevor Darrell, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04722)  

**Abstract**: Large language models (LLMs) exhibit the ability to generalize given few-shot examples in their input prompt, an emergent capability known as in-context learning (ICL). We investigate whether LLMs utilize ICL to perform structured reasoning in ways that are consistent with a Bayesian framework or rely on pattern matching. Using a controlled setting of biased coin flips, we find that: (1) LLMs often possess biased priors, causing initial divergence in zero-shot settings, (2) in-context evidence outweighs explicit bias instructions, (3) LLMs broadly follow Bayesian posterior updates, with deviations primarily due to miscalibrated priors rather than flawed updates, and (4) attention magnitude has negligible effect on Bayesian inference. With sufficient demonstrations of biased coin flips via ICL, LLMs update their priors in a Bayesian manner. 

---
# L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning 

**Authors**: Pranjal Aggarwal, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2503.04697)  

**Abstract**: Reasoning language models have shown an uncanny ability to improve performance at test-time by ``thinking longer''-that is, by generating longer chain-of-thought sequences and hence using more compute. However, the length of their chain-of-thought reasoning is not controllable, making it impossible to allocate test-time compute to achieve a desired level of performance. We introduce Length Controlled Policy Optimization (LCPO), a simple reinforcement learning method that optimizes for accuracy and adherence to user-specified length constraints. We use LCPO to train L1, a reasoning language model that produces outputs satisfying a length constraint given in its prompt. L1's length control allows for smoothly trading off computational cost and accuracy on a wide range of tasks, and outperforms the state-of-the-art S1 method for length control. Furthermore, we uncover an unexpected short chain-of-thought capability in models trained with LCPO. For instance, our 1.5B L1 model surpasses GPT-4o at equal reasoning lengths. Overall, LCPO enables precise control over reasoning length, allowing for fine-grained allocation of test-time compute and accuracy. We release code and models at this https URL 

---
# KidneyTalk-open: No-code Deployment of a Private Large Language Model with Medical Documentation-Enhanced Knowledge Database for Kidney Disease 

**Authors**: Yongchao Long, Chao Yang, Gongzheng Tang, Jinwei Wang, Zhun Sui, Yuxi Zhou, Shenda Hong, Luxia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04153)  

**Abstract**: Privacy-preserving medical decision support for kidney disease requires localized deployment of large language models (LLMs) while maintaining clinical reasoning capabilities. Current solutions face three challenges: 1) Cloud-based LLMs pose data security risks; 2) Local model deployment demands technical expertise; 3) General LLMs lack mechanisms to integrate medical knowledge. Retrieval-augmented systems also struggle with medical document processing and clinical usability. We developed KidneyTalk-open, a desktop system integrating three technical components: 1) No-code deployment of state-of-the-art (SOTA) open-source LLMs (such as DeepSeek-r1, Qwen2.5) via local inference engine; 2) Medical document processing pipeline combining context-aware chunking and intelligent filtering; 3) Adaptive Retrieval and Augmentation Pipeline (AddRep) employing agents collaboration for improving the recall rate of medical documents. A graphical interface was designed to enable clinicians to manage medical documents and conduct AI-powered consultations without technical expertise. Experimental validation on 1,455 challenging nephrology exam questions demonstrates AddRep's effectiveness: achieving 29.1% accuracy (+8.1% over baseline) with intelligent knowledge integration, while maintaining robustness through 4.9% rejection rate to suppress hallucinations. Comparative case studies with the mainstream products (AnythingLLM, Chatbox, GPT4ALL) demonstrate KidneyTalk-open's superior performance in real clinical query. KidneyTalk-open represents the first no-code medical LLM system enabling secure documentation-enhanced medical Q&A on desktop. Its designs establishes a new framework for privacy-sensitive clinical AI applications. The system significantly lowers technical barriers while improving evidence traceability, enabling more medical staff or patients to use SOTA open-source LLMs conveniently. 

---
# Mark Your LLM: Detecting the Misuse of Open-Source Large Language Models via Watermarking 

**Authors**: Yijie Xu, Aiwei Liu, Xuming Hu, Lijie Wen, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.04636)  

**Abstract**: As open-source large language models (LLMs) like Llama3 become more capable, it is crucial to develop watermarking techniques to detect their potential misuse. Existing watermarking methods either add watermarks during LLM inference, which is unsuitable for open-source LLMs, or primarily target classification LLMs rather than recent generative LLMs. Adapting these watermarks to open-source LLMs for misuse detection remains an open challenge. This work defines two misuse scenarios for open-source LLMs: intellectual property (IP) violation and LLM Usage Violation. Then, we explore the application of inference-time watermark distillation and backdoor watermarking in these contexts. We propose comprehensive evaluation methods to assess the impact of various real-world further fine-tuning scenarios on watermarks and the effect of these watermarks on LLM performance. Our experiments reveal that backdoor watermarking could effectively detect IP Violation, while inference-time watermark distillation is applicable in both scenarios but less robust to further fine-tuning and has a more significant impact on LLM performance compared to backdoor watermarking. Exploring more advanced watermarking methods for open-source LLMs to detect their misuse should be an important future direction. 

---
# HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization 

**Authors**: Zhijian Zhuo, Yutao Zeng, Ya Wang, Sijun Zhang, Jian Yang, Xiaoqing Li, Xun Zhou, Jinwen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04598)  

**Abstract**: Transformers have become the de facto architecture for a wide range of machine learning tasks, particularly in large language models (LLMs). Despite their remarkable performance, challenges remain in training deep transformer networks, especially regarding the location of layer normalization. While Pre-Norm structures facilitate easier training due to their more prominent identity path, they often yield suboptimal performance compared to Post-Norm. In this paper, we propose $\textbf{HybridNorm}$, a straightforward yet effective hybrid normalization strategy that integrates the advantages of both Pre-Norm and Post-Norm approaches. Specifically, HybridNorm employs QKV normalization within the attention mechanism and Post-Norm in the feed-forward network (FFN) of each transformer block. This design not only stabilizes training but also enhances performance, particularly in the context of LLMs. Comprehensive experiments in both dense and sparse architectures show that HybridNorm consistently outperforms both Pre-Norm and Post-Norm approaches, achieving state-of-the-art results across various benchmarks. These findings highlight the potential of HybridNorm as a more stable and effective technique for improving the training and performance of deep transformer models. %Code will be made publicly available. Code is available at this https URL. 

---
# Compositional Causal Reasoning Evaluation in Language Models 

**Authors**: Jacqueline R. M. A. Maasch, Alihan Hüyük, Xinnuo Xu, Aditya V. Nori, Javier Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2503.04556)  

**Abstract**: Causal reasoning and compositional reasoning are two core aspirations in generative AI. Measuring the extent of these behaviors requires principled evaluation methods. We explore a unified perspective that considers both behaviors simultaneously, termed compositional causal reasoning (CCR): the ability to infer how causal measures compose and, equivalently, how causal quantities propagate through graphs. We instantiate a framework for the systematic evaluation of CCR for the average treatment effect and the probability of necessity and sufficiency. As proof of concept, we demonstrate the design of CCR tasks for language models in the LLama, Phi, and GPT families. On a math word problem, our framework revealed a range of taxonomically distinct error patterns. Additionally, CCR errors increased with the complexity of causal paths for all models except o1. 

---
# Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment 

**Authors**: Wen Yang, Junhong Wu, Chen Wang, Chengqing Zong, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04647)  

**Abstract**: Direct Preference Optimization (DPO) has become a prominent method for aligning Large Language Models (LLMs) with human preferences. While DPO has enabled significant progress in aligning English LLMs, multilingual preference alignment is hampered by data scarcity. To address this, we propose a novel approach that $\textit{captures}$ learned preferences from well-aligned English models by implicit rewards and $\textit{transfers}$ them to other languages through iterative training. Specifically, we derive an implicit reward model from the logits of an English DPO-aligned model and its corresponding reference model. This reward model is then leveraged to annotate preference relations in cross-lingual instruction-following pairs, using English instructions to evaluate multilingual responses. The annotated data is subsequently used for multilingual DPO fine-tuning, facilitating preference knowledge transfer from English to other languages. Fine-tuning Llama3 for two iterations resulted in a 12.72% average improvement in Win Rate and a 5.97% increase in Length Control Win Rate across all training languages on the X-AlpacaEval leaderboard. Our findings demonstrate that leveraging existing English-aligned models can enable efficient and effective multilingual preference alignment, significantly reducing the need for extensive multilingual preference data. The code is available at this https URL 

---
# Dedicated Feedback and Edit Models Empower Inference-Time Scaling for Open-Ended General-Domain Tasks 

**Authors**: Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Daniel Egert, Ellie Evans, Hoo-Chang Shin, Felipe Soares, Yi Dong, Oleksii Kuchaiev  

**Link**: [PDF](https://arxiv.org/pdf/2503.04378)  

**Abstract**: Inference-Time Scaling has been critical to the success of recent models such as OpenAI o1 and DeepSeek R1. However, many techniques used to train models for inference-time scaling require tasks to have answers that can be verified, limiting their application to domains such as math, coding and logical reasoning. We take inspiration from how humans make first attempts, ask for detailed feedback from others and make improvements based on such feedback across a wide spectrum of open-ended endeavors. To this end, we collect data for and train dedicated Feedback and Edit Models that are capable of performing inference-time scaling for open-ended general-domain tasks. In our setup, one model generates an initial response, which are given feedback by a second model, that are then used by a third model to edit the response. We show that performance on Arena Hard, a benchmark strongly predictive of Chatbot Arena Elo can be boosted by scaling the number of initial response drafts, effective feedback and edited responses. When scaled optimally, our setup based on 70B models from the Llama 3 family can reach SoTA performance on Arena Hard at 92.7 as of 5 Mar 2025, surpassing OpenAI o1-preview-2024-09-12 with 90.4 and DeepSeek R1 with 92.3. 

---
# Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation 

**Authors**: Christian Rondanini, Barbara Carminati, Elena Ferrari, Antonio Gaudiano, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04302)  

**Abstract**: The rapid evolution of malware attacks calls for the development of innovative detection methods, especially in resource-constrained edge computing. Traditional detection techniques struggle to keep up with modern malware's sophistication and adaptability, prompting a shift towards advanced methodologies like those leveraging Large Language Models (LLMs) for enhanced malware detection. However, deploying LLMs for malware detection directly at edge devices raises several challenges, including ensuring accuracy in constrained environments and addressing edge devices' energy and computational limits. To tackle these challenges, this paper proposes an architecture leveraging lightweight LLMs' strengths while addressing limitations like reduced accuracy and insufficient computational power. To evaluate the effectiveness of the proposed lightweight LLM-based approach for edge computing, we perform an extensive experimental evaluation using several state-of-the-art lightweight LLMs. We test them with several publicly available datasets specifically designed for edge and IoT scenarios and different edge nodes with varying computational power and characteristics. 

---
# Solving Word-Sense Disambiguation and Word-Sense Induction with Dictionary Examples 

**Authors**: Tadej Škvorc, Marko Robnik-Šikonja  

**Link**: [PDF](https://arxiv.org/pdf/2503.04328)  

**Abstract**: Many less-resourced languages struggle with a lack of large, task-specific datasets that are required for solving relevant tasks with modern transformer-based large language models (LLMs). On the other hand, many linguistic resources, such as dictionaries, are rarely used in this context despite their large information contents. We show how LLMs can be used to extend existing language resources in less-resourced languages for two important tasks: word-sense disambiguation (WSD) and word-sense induction (WSI). We approach the two tasks through the related but much more accessible word-in-context (WiC) task where, given a pair of sentences and a target word, a classification model is tasked with predicting whether the sense of a given word differs between sentences. We demonstrate that a well-trained model for this task can distinguish between different word senses and can be adapted to solve the WSD and WSI tasks. The advantage of using the WiC task, instead of directly predicting senses, is that the WiC task does not need pre-constructed sense inventories with a sufficient number of examples for each sense, which are rarely available in less-resourced languages. We show that sentence pairs for the WiC task can be successfully generated from dictionary examples using LLMs. The resulting prediction models outperform existing models on WiC, WSD, and WSI tasks. We demonstrate our methodology on the Slovene language, where a monolingual dictionary is available, but word-sense resources are tiny. 

---
# Towards Autonomous Reinforcement Learning for Real-World Robotic Manipulation with Large Language Models 

**Authors**: Niccolò Turcato, Matteo Iovino, Aris Synodinos, Alberto Dalla Libera, Ruggero Carli, Pietro Falco  

**Link**: [PDF](https://arxiv.org/pdf/2503.04280)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Visual Language Models (VLMs) have significantly impacted robotics, enabling high-level semantic motion planning applications. Reinforcement Learning (RL), a complementary paradigm, enables agents to autonomously optimize complex behaviors through interaction and reward signals. However, designing effective reward functions for RL remains challenging, especially in real-world tasks where sparse rewards are insufficient and dense rewards require elaborate design. In this work, we propose Autonomous Reinforcement learning for Complex HumanInformed Environments (ARCHIE), an unsupervised pipeline leveraging GPT-4, a pre-trained LLM, to generate reward functions directly from natural language task descriptions. The rewards are used to train RL agents in simulated environments, where we formalize the reward generation process to enhance feasibility. Additionally, GPT-4 automates the coding of task success criteria, creating a fully automated, one-shot procedure for translating human-readable text into deployable robot skills. Our approach is validated through extensive simulated experiments on single-arm and bi-manual manipulation tasks using an ABB YuMi collaborative robot, highlighting its practicality and effectiveness. Tasks are demonstrated on the real robot setup. 

---
# How Do Hackathons Foster Creativity? Towards AI Collaborative Evaluation of Creativity at Scale 

**Authors**: Jeanette Falk, Yiyi Chen, Janet Rafner, Mike Zhang, Johannes Bjerva, Alexander Nolte  

**Link**: [PDF](https://arxiv.org/pdf/2503.04290)  

**Abstract**: Hackathons have become popular collaborative events for accelerating the development of creative ideas and prototypes. There are several case studies showcasing creative outcomes across domains such as industry, education, and research. However, there are no large-scale studies on creativity in hackathons which can advance theory on how hackathon formats lead to creative outcomes. We conducted a computational analysis of 193,353 hackathon projects. By operationalizing creativity through usefulness and novelty, we refined our dataset to 10,363 projects, allowing us to analyze how participant characteristics, collaboration patterns, and hackathon setups influence the development of creative projects. The contribution of our paper is twofold: We identified means for organizers to foster creativity in hackathons. We also explore the use of large language models (LLMs) to augment the evaluation of creative outcomes and discuss challenges and opportunities of doing this, which has implications for creativity research at large. 

---
# MASTER: Multimodal Segmentation with Text Prompts 

**Authors**: Fuyang Liu, Shun Lu, Jilin Mei, Yu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04199)  

**Abstract**: RGB-Thermal fusion is a potential solution for various weather and light conditions in challenging scenarios. However, plenty of studies focus on designing complex modules to fuse different modalities. With the widespread application of large language models (LLMs), valuable information can be more effectively extracted from natural language. Therefore, we aim to leverage the advantages of large language models to design a structurally simple and highly adaptable multimodal fusion model architecture. We proposed MultimodAl Segmentation with TExt PRompts (MASTER) architecture, which integrates LLM into the fusion of RGB-Thermal multimodal data and allows complex query text to participate in the fusion process. Our model utilizes a dual-path structure to extract information from different modalities of images. Additionally, we employ LLM as the core module for multimodal fusion, enabling the model to generate learnable codebook tokens from RGB, thermal images, and textual information. A lightweight image decoder is used to obtain semantic segmentation results. The proposed MASTER performs exceptionally well in benchmark tests across various automated driving scenarios, yielding promising results. 

---
# Knowledge-Decoupled Synergetic Learning: An MLLM based Collaborative Approach to Few-shot Multimodal Dialogue Intention Recognition 

**Authors**: Bin Chen, Yu Zhang, Hongfei Ye, Ziyi Huang, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04201)  

**Abstract**: Few-shot multimodal dialogue intention recognition is a critical challenge in the e-commerce domainn. Previous methods have primarily enhanced model classification capabilities through post-training techniques. However, our analysis reveals that training for few-shot multimodal dialogue intention recognition involves two interconnected tasks, leading to a seesaw effect in multi-task learning. This phenomenon is attributed to knowledge interference stemming from the superposition of weight matrix updates during the training process. To address these challenges, we propose Knowledge-Decoupled Synergetic Learning (KDSL), which mitigates these issues by utilizing smaller models to transform knowledge into interpretable rules, while applying the post-training of larger models. By facilitating collaboration between the large and small multimodal large language models for prediction, our approach demonstrates significant improvements. Notably, we achieve outstanding results on two real Taobao datasets, with enhancements of 6.37\% and 6.28\% in online weighted F1 scores compared to the state-of-the-art method, thereby validating the efficacy of our framework. 

---
# Ticktack : Long Span Temporal Alignment of Large Language Models Leveraging Sexagenary Cycle Time Expression 

**Authors**: Xue Han, Qian Hu, Yitong Wang, Wenchun Gao, Lianlian Zhang, Qing Wang, Lijun Mei, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.04150)  

**Abstract**: Large language models (LLMs) suffer from temporal misalignment issues especially across long span of time. The issue arises from knowing that LLMs are trained on large amounts of data where temporal information is rather sparse over long times, such as thousands of years, resulting in insufficient learning or catastrophic forgetting by the LLMs. This paper proposes a methodology named "Ticktack" for addressing the LLM's long-time span misalignment in a yearly setting. Specifically, we first propose to utilize the sexagenary year expression instead of the Gregorian year expression employed by LLMs, achieving a more uniform distribution in yearly granularity. Then, we employ polar coordinates to model the sexagenary cycle of 60 terms and the year order within each term, with additional temporal encoding to ensure LLMs understand them. Finally, we present a temporal representational alignment approach for post-training LLMs that effectively distinguishes time points with relevant knowledge, hence improving performance on time-related tasks, particularly over a long period. We also create a long time span benchmark for evaluation. Experimental results prove the effectiveness of our proposal. 

---
# Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination 

**Authors**: Simin Chen, Pranav Pusarla, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2503.04149)  

**Abstract**: The rapid evolution of code largelanguage models underscores the need for effective and transparent benchmarking of their reasoning capabilities. However, the current benchmarking approach heavily depends on publicly available, human-created datasets. The widespread use of these fixed benchmark datasets makes the benchmarking process to be static and thus particularly susceptible to data contamination, an unavoidable consequence of the extensive data collection processes used to train Code LLMs. Existing approaches that address data contamination often suffer from human effort limitations and imbalanced problem complexity. To tackle these challenges, we propose \tool, a novel benchmarking suite for evaluating Code LLMs under potential data contamination. Given a seed programming problem, \tool employs multiple agents to extract and modify the context without altering the core logic, generating semantically equivalent variations. We introduce a dynamic data generation methods and conduct empirical studies on two seed datasets across 21 Code LLMs. Results show that \tool effectively benchmarks reasoning capabilities under contamination risks while generating diverse problem sets to ensure consistent and reliable evaluations. 

---
# InterChat: Enhancing Generative Visual Analytics using Multimodal Interactions 

**Authors**: Juntong Chen, Jiang Wu, Jiajing Guo, Vikram Mohanty, Xueming Li, Jorge Piazentin Ono, Wenbin He, Liu Ren, Dongyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04110)  

**Abstract**: The rise of Large Language Models (LLMs) and generative visual analytics systems has transformed data-driven insights, yet significant challenges persist in accurately interpreting users' analytical and interaction intents. While language inputs offer flexibility, they often lack precision, making the expression of complex intents inefficient, error-prone, and time-intensive. To address these limitations, we investigate the design space of multimodal interactions for generative visual analytics through a literature review and pilot brainstorming sessions. Building on these insights, we introduce a highly extensible workflow that integrates multiple LLM agents for intent inference and visualization generation. We develop InterChat, a generative visual analytics system that combines direct manipulation of visual elements with natural language inputs. This integration enables precise intent communication and supports progressive, visually driven exploratory data analyses. By employing effective prompt engineering, and contextual interaction linking, alongside intuitive visualization and interaction designs, InterChat bridges the gap between user interactions and LLM-driven visualizations, enhancing both interpretability and usability. Extensive evaluations, including two usage scenarios, a user study, and expert feedback, demonstrate the effectiveness of InterChat. Results show significant improvements in the accuracy and efficiency of handling complex visual analytics tasks, highlighting the potential of multimodal interactions to redefine user engagement and analytical depth in generative visual analytics. 

---
# Disparities in LLM Reasoning Accuracy and Explanations: A Case Study on African American English 

**Authors**: Runtao Zhou, Guangya Wan, Saadia Gabriel, Sheng Li, Alexander J Gates, Maarten Sap, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04099)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning tasks, leading to their widespread deployment. However, recent studies have highlighted concerning biases in these models, particularly in their handling of dialectal variations like African American English (AAE). In this work, we systematically investigate dialectal disparities in LLM reasoning tasks. We develop an experimental framework comparing LLM performance given Standard American English (SAE) and AAE prompts, combining LLM-based dialect conversion with established linguistic analyses. We find that LLMs consistently produce less accurate responses and simpler reasoning chains and explanations for AAE inputs compared to equivalent SAE questions, with disparities most pronounced in social science and humanities domains. These findings highlight systematic differences in how LLMs process and reason about different language varieties, raising important questions about the development and deployment of these systems in our multilingual and multidialectal world. Our code repository is publicly available at this https URL. 

---
# Uncovering inequalities in new knowledge learning by large language models across different languages 

**Authors**: Chenglong Wang, Haoyu Tang, Xiyuan Yang, Yueqi Xie, Jina Suh, Sunayana Sitaram, Junming Huang, Yu Xie, Zhaoya Gong, Xing Xie, Fangzhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04064)  

**Abstract**: As large language models (LLMs) gradually become integral tools for problem solving in daily life worldwide, understanding linguistic inequality is becoming increasingly important. Existing research has primarily focused on static analyses that assess the disparities in the existing knowledge and capabilities of LLMs across languages. However, LLMs are continuously evolving, acquiring new knowledge to generate up-to-date, domain-specific responses. Investigating linguistic inequalities within this dynamic process is, therefore, also essential. In this paper, we explore inequalities in new knowledge learning by LLMs across different languages and four key dimensions: effectiveness, transferability, prioritization, and robustness. Through extensive experiments under two settings (in-context learning and fine-tuning) using both proprietary and open-source models, we demonstrate that low-resource languages consistently face disadvantages across all four dimensions. By shedding light on these disparities, we aim to raise awareness of linguistic inequalities in LLMs' new knowledge learning, fostering the development of more inclusive and equitable future LLMs. 

---
# Chart-HQA: A Benchmark for Hypothetical Question Answering in Charts 

**Authors**: Xiangnan Chen, Yuancheng Fang, Qian Xiao, Juncheng Li, Jun Lin, Siliang Tang, Yi Yang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04095)  

**Abstract**: Multimodal Large Language Models (MLLMs) have garnered significant attention for their strong visual-semantic understanding. Most existing chart benchmarks evaluate MLLMs' ability to parse information from charts to answer this http URL, they overlook the inherent output biases of MLLMs, where models rely on their parametric memory to answer questions rather than genuinely understanding the chart content. To address this limitation, we introduce a novel Chart Hypothetical Question Answering (HQA) task, which imposes assumptions on the same question to compel models to engage in counterfactual reasoning based on the chart content. Furthermore, we introduce HAI, a human-AI interactive data synthesis approach that leverages the efficient text-editing capabilities of LLMs alongside human expert knowledge to generate diverse and high-quality HQA data at a low cost. Using HAI, we construct Chart-HQA, a challenging benchmark synthesized from publicly available data sources. Evaluation results on 18 MLLMs of varying model sizes reveal that current models face significant generalization challenges and exhibit imbalanced reasoning performance on the HQA task. 

---
# PP-DocBee: Improving Multimodal Document Understanding Through a Bag of Tricks 

**Authors**: Feng Ni, Kui Huang, Yao Lu, Wenyu Lv, Guanzhong Wang, Zeyu Chen, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04065)  

**Abstract**: With the rapid advancement of digitalization, various document images are being applied more extensively in production and daily life, and there is an increasingly urgent need for fast and accurate parsing of the content in document images. Therefore, this report presents PP-DocBee, a novel multimodal large language model designed for end-to-end document image understanding. First, we develop a data synthesis strategy tailored to document scenarios in which we build a diverse dataset to improve the model generalization. Then, we apply a few training techniques, including dynamic proportional sampling, data preprocessing, and OCR postprocessing strategies. Extensive evaluations demonstrate the superior performance of PP-DocBee, achieving state-of-the-art results on English document understanding benchmarks and even outperforming existing open source and commercial models in Chinese document understanding. The source code and pre-trained models are publicly available at \href{this https URL}{this https URL}. 

---
# RetinalGPT: A Retinal Clinical Preference Conversational Assistant Powered by Large Vision-Language Models 

**Authors**: Wenhui Zhu, Xin Li, Xiwen Chen, Peijie Qiu, Vamsi Krishna Vasa, Xuanzhao Dong, Yanxi Chen, Natasha Lepore, Oana Dumitrascu, Yi Su, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03987)  

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have gained significant attention for their remarkable ability to process and analyze non-textual data, such as images, videos, and audio. Notably, several adaptations of general-domain MLLMs to the medical field have been explored, including LLaVA-Med. However, these medical adaptations remain insufficiently advanced in understanding and interpreting retinal images. In contrast, medical experts emphasize the importance of quantitative analyses for disease detection and interpretation. This underscores a gap between general-domain and medical-domain MLLMs: while general-domain MLLMs excel in broad applications, they lack the specialized knowledge necessary for precise diagnostic and interpretative tasks in the medical field. To address these challenges, we introduce \textit{RetinalGPT}, a multimodal conversational assistant for clinically preferred quantitative analysis of retinal images. Specifically, we achieve this by compiling a large retinal image dataset, developing a novel data pipeline, and employing customized visual instruction tuning to enhance both retinal analysis and enrich medical knowledge. In particular, RetinalGPT outperforms MLLM in the generic domain by a large margin in the diagnosis of retinal diseases in 8 benchmark retinal datasets. Beyond disease diagnosis, RetinalGPT features quantitative analyses and lesion localization, representing a pioneering step in leveraging LLMs for an interpretable and end-to-end clinical research framework. The code is available at this https URL 

---
# Multi-Agent Systems Powered by Large Language Models: Applications in Swarm Intelligence 

**Authors**: Cristian Jimenez-Romero, Alper Yegenoglu, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2503.03800)  

**Abstract**: This work examines the integration of large language models (LLMs) into multi-agent simulations by replacing the hard-coded programs of agents with LLM-driven prompts. The proposed approach is showcased in the context of two examples of complex systems from the field of swarm intelligence: ant colony foraging and bird flocking. Central to this study is a toolchain that integrates LLMs with the NetLogo simulation platform, leveraging its Python extension to enable communication with GPT-4o via the OpenAI API. This toolchain facilitates prompt-driven behavior generation, allowing agents to respond adaptively to environmental data. For both example applications mentioned above, we employ both structured, rule-based prompts and autonomous, knowledge-driven prompts. Our work demonstrates how this toolchain enables LLMs to study self-organizing processes and induce emergent behaviors within multi-agent environments, paving the way for new approaches to exploring intelligent systems and modeling swarm intelligence inspired by natural phenomena. We provide the code, including simulation files and data at this https URL. 

---
# Human Implicit Preference-Based Policy Fine-tuning for Multi-Agent Reinforcement Learning in USV Swarm 

**Authors**: Hyeonjun Kim, Kanghoon Lee, Junho Park, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.03796)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has shown promise in solving complex problems involving cooperation and competition among agents, such as an Unmanned Surface Vehicle (USV) swarm used in search and rescue, surveillance, and vessel protection. However, aligning system behavior with user preferences is challenging due to the difficulty of encoding expert intuition into reward functions. To address the issue, we propose a Reinforcement Learning with Human Feedback (RLHF) approach for MARL that resolves credit-assignment challenges through an Agent-Level Feedback system categorizing feedback into intra-agent, inter-agent, and intra-team types. To overcome the challenges of direct human feedback, we employ a Large Language Model (LLM) evaluator to validate our approach using feedback scenarios such as region constraints, collision avoidance, and task allocation. Our method effectively refines USV swarm policies, addressing key challenges in multi-agent systems while maintaining fairness and performance consistency. 

---
# BotUmc: An Uncertainty-Aware Twitter Bot Detection with Multi-view Causal Inference 

**Authors**: Tao Yang, Yang Hu, Feihong Lu, Ziwei Zhang, Qingyun Sun, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.03775)  

**Abstract**: Social bots have become widely known by users of social platforms. To prevent social bots from spreading harmful speech, many novel bot detections are proposed. However, with the evolution of social bots, detection methods struggle to give high-confidence answers for samples. This motivates us to quantify the uncertainty of the outputs, informing the confidence of the results. Therefore, we propose an uncertainty-aware bot detection method to inform the confidence and use the uncertainty score to pick a high-confidence decision from multiple views of a social network under different environments. Specifically, our proposed BotUmc uses LLM to extract information from tweets. Then, we construct a graph based on the extracted information, the original user information, and the user relationship and generate multiple views of the graph by causal interference. Lastly, an uncertainty loss is used to force the model to quantify the uncertainty of results and select the result with low uncertainty in one view as the final decision. Extensive experiments show the superiority of our method. 

---
# M2-omni: Advancing Omni-MLLM for Comprehensive Modality Support with Competitive Performance 

**Authors**: Qingpei Guo, Kaiyou Song, Zipeng Feng, Ziping Ma, Qinglong Zhang, Sirui Gao, Xuzheng Yu, Yunxiao Sun, Tai-WeiChang, Jingdong Chen, Ming Yang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18778)  

**Abstract**: We present M2-omni, a cutting-edge, open-source omni-MLLM that achieves competitive performance to GPT-4o. M2-omni employs a unified multimodal sequence modeling framework, which empowers Large Language Models(LLMs) to acquire comprehensive cross-modal understanding and generation capabilities. Specifically, M2-omni can process arbitrary combinations of audio, video, image, and text modalities as input, generating multimodal sequences interleaving with audio, image, or text outputs, thereby enabling an advanced and interactive real-time experience. The training of such an omni-MLLM is challenged by significant disparities in data quantity and convergence rates across modalities. To address these challenges, we propose a step balance strategy during pre-training to handle the quantity disparities in modality-specific data. Additionally, a dynamically adaptive balance strategy is introduced during the instruction tuning stage to synchronize the modality-wise training progress, ensuring optimal convergence. Notably, we prioritize preserving strong performance on pure text tasks to maintain the robustness of M2-omni's language understanding capability throughout the training process. To our best knowledge, M2-omni is currently a very competitive open-source model to GPT-4o, characterized by its comprehensive modality and task support, as well as its exceptional performance. We expect M2-omni will advance the development of omni-MLLMs, thus facilitating future research in this domain. 

---
# FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference 

**Authors**: Hongchao Du, Shangyu Wu, Arina Kharlamova, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.03777)  

**Abstract**: Large Language Models (LLMs) face challenges for on-device inference due to high memory demands. Traditional methods to reduce memory usage often compromise performance and lack adaptability. We propose FlexInfer, an optimized offloading framework for on-device inference, addressing these issues with techniques like asynchronous prefetching, balanced memory locking, and flexible tensor preservation. These strategies enhance memory efficiency and mitigate I/O bottlenecks, ensuring high performance within user-specified resource constraints. Experiments demonstrate that FlexInfer significantly improves throughput under limited resources, achieving up to 12.5 times better performance than existing methods and facilitating the deployment of large models on resource-constrained devices. 

---
# Benchmarking Large Language Models on Multiple Tasks in Bioinformatics NLP with Prompting 

**Authors**: Jiyue Jiang, Pengan Chen, Jiuming Wang, Dongchen He, Ziqin Wei, Liang Hong, Licheng Zong, Sheng Wang, Qinze Yu, Zixian Ma, Yanyu Chen, Yimin Fan, Xiangyu Shi, Jiawei Sun, Chuan Wu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04013)  

**Abstract**: Large language models (LLMs) have become important tools in solving biological problems, offering improvements in accuracy and adaptability over conventional methods. Several benchmarks have been proposed to evaluate the performance of these LLMs. However, current benchmarks can hardly evaluate the performance of these models across diverse tasks effectively. In this paper, we introduce a comprehensive prompting-based benchmarking framework, termed Bio-benchmark, which includes 30 key bioinformatics tasks covering areas such as proteins, RNA, drugs, electronic health records, and traditional Chinese medicine. Using this benchmark, we evaluate six mainstream LLMs, including GPT-4o and Llama-3.1-70b, etc., using 0-shot and few-shot Chain-of-Thought (CoT) settings without fine-tuning to reveal their intrinsic capabilities. To improve the efficiency of our evaluations, we demonstrate BioFinder, a new tool for extracting answers from LLM responses, which increases extraction accuracy by round 30% compared to existing methods. Our benchmark results show the biological tasks suitable for current LLMs and identify specific areas requiring enhancement. Furthermore, we propose targeted prompt engineering strategies for optimizing LLM performance in these contexts. Based on these findings, we provide recommendations for the development of more robust LLMs tailored for various biological applications. This work offers a comprehensive evaluation framework and robust tools to support the application of LLMs in bioinformatics. 

---
# Quantifying the Reasoning Abilities of LLMs on Real-world Clinical Cases 

**Authors**: Pengcheng Qiu, Chaoyi Wu, Shuyu Liu, Weike Zhao, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.04691)  

**Abstract**: The latest reasoning-enhanced large language models (reasoning LLMs), such as DeepSeek-R1 and OpenAI-o3, have demonstrated remarkable success. However, the application of such reasoning enhancements to the highly professional medical domain has not been clearly evaluated, particularly regarding with not only assessing the final generation but also examining the quality of their reasoning processes. In this study, we present MedR-Bench, a reasoning-focused medical evaluation benchmark comprising 1,453 structured patient cases with reasoning references mined from case reports. Our benchmark spans 13 body systems and 10 specialty disorders, encompassing both common and rare diseases. In our evaluation, we introduce a versatile framework consisting of three critical clinical stages: assessment recommendation, diagnostic decision-making, and treatment planning, comprehensively capturing the LLMs' performance across the entire patient journey in healthcare. For metrics, we propose a novel agentic system, Reasoning Evaluator, designed to automate and objectively quantify free-text reasoning responses in a scalable manner from the perspectives of efficiency, factuality, and completeness by dynamically searching and performing cross-referencing checks. As a result, we assess five state-of-the-art reasoning LLMs, including DeepSeek-R1, OpenAI-o3-mini, and others. Our results reveal that current LLMs can handle relatively simple diagnostic tasks with sufficient critical assessment results, achieving accuracy generally over 85%. However, they still struggle with more complex tasks, such as assessment recommendation and treatment planning. In reasoning, their reasoning processes are generally reliable, with factuality scores exceeding 90%, though they often omit critical reasoning steps. Our study clearly reveals further development directions for current clinical LLMs. 

---
# IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval 

**Authors**: Tingyu Song, Guo Gan, Mingsheng Shang, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04644)  

**Abstract**: We introduce IFIR, the first comprehensive benchmark designed to evaluate instruction-following information retrieval (IR) in expert domains. IFIR includes 2,426 high-quality examples and covers eight subsets across four specialized domains: finance, law, healthcare, and science literature. Each subset addresses one or more domain-specific retrieval tasks, replicating real-world scenarios where customized instructions are critical. IFIR enables a detailed analysis of instruction-following retrieval capabilities by incorporating instructions at different levels of complexity. We also propose a novel LLM-based evaluation method to provide a more precise and reliable assessment of model performance in following instructions. Through extensive experiments on 15 frontier retrieval models, including those based on LLMs, our results reveal that current models face significant challenges in effectively following complex, domain-specific instructions. We further provide in-depth analyses to highlight these limitations, offering valuable insights to guide future advancements in retriever development. 

---
# LLM-guided Plan and Retrieval: A Strategic Alignment for Interpretable User Satisfaction Estimation in Dialogue 

**Authors**: Sangyeop Kim, Sohhyung Park, Jaewon Jung, Jinseok Kim, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2503.04675)  

**Abstract**: Understanding user satisfaction with conversational systems, known as User Satisfaction Estimation (USE), is essential for assessing dialogue quality and enhancing user experiences. However, existing methods for USE face challenges due to limited understanding of underlying reasons for user dissatisfaction and the high costs of annotating user intentions. To address these challenges, we propose PRAISE (Plan and Retrieval Alignment for Interpretable Satisfaction Estimation), an interpretable framework for effective user satisfaction prediction. PRAISE operates through three key modules. The Strategy Planner develops strategies, which are natural language criteria for classifying user satisfaction. The Feature Retriever then incorporates knowledge on user satisfaction from Large Language Models (LLMs) and retrieves relevance features from utterances. Finally, the Score Analyzer evaluates strategy predictions and classifies user satisfaction. Experimental results demonstrate that PRAISE achieves state-of-the-art performance on three benchmarks for the USE task. Beyond its superior performance, PRAISE offers additional benefits. It enhances interpretability by providing instance-level explanations through effective alignment of utterances with strategies. Moreover, PRAISE operates more efficiently than existing approaches by eliminating the need for LLMs during the inference phase. 

---
# Better Process Supervision with Bi-directional Rewarding Signals 

**Authors**: Wenxiang Chen, Wei He, Zhiheng Xi, Honglin Guo, Boyang Hong, Jiazheng Zhang, Rui Zheng, Nijun Li, Tao Gui, Yun Li, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04618)  

**Abstract**: Process supervision, i.e., evaluating each step, is critical for complex large language model (LLM) reasoning and test-time searching with increased inference compute. Existing approaches, represented by process reward models (PRMs), primarily focus on rewarding signals up to the current step, exhibiting a one-directional nature and lacking a mechanism to model the distance to the final target. To address this problem, we draw inspiration from the A* algorithm, which states that an effective supervisory signal should simultaneously consider the incurred cost and the estimated cost for reaching the target. Building on this key insight, we introduce BiRM, a novel process supervision model that not only evaluates the correctness of previous steps but also models the probability of future success. We conduct extensive experiments on mathematical reasoning tasks and demonstrate that BiRM provides more precise evaluations of LLM reasoning steps, achieving an improvement of 3.1% on Gaokao2023 over PRM under the Best-of-N sampling method. Besides, in search-based strategies, BiRM provides more comprehensive guidance and outperforms ORM by 5.0% and PRM by 3.8% respectively on MATH-500. 

---
# UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets 

**Authors**: Wenyu Wang, Mengqi Zhang, Xiaotian Ye, Zhaochun Ren, Zhumin Chen, Pengjie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.04693)  

**Abstract**: Large Language Models (LLMs) inevitably acquire harmful information during training on massive datasets. LLM unlearning aims to eliminate the influence of such harmful information while maintaining the model's overall performance. Existing unlearning methods, represented by gradient ascent-based approaches, primarily focus on forgetting target data while overlooking the crucial impact of logically related knowledge on the effectiveness of unlearning. In this paper, through both theoretical and experimental analyses, we first demonstrate that a key reason for the suboptimal unlearning performance is that models can reconstruct the target content through reasoning with logically related knowledge. To address this issue, we propose Unlearning Improvement via Parameter Extrapolation (UIPE), a method that removes knowledge highly correlated with the forgetting targets. Experimental results show that UIPE significantly enhances the performance of various mainstream LLM unlearning methods on the TOFU benchmark. 

---
# HalluCounter: Reference-free LLM Hallucination Detection in the Wild! 

**Authors**: Ashok Urlana, Gopichand Kanumolu, Charaka Vinayak Kumar, Bala Mallikarjunarao Garlapati, Rahul Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2503.04615)  

**Abstract**: Response consistency-based, reference-free hallucination detection (RFHD) methods do not depend on internal model states, such as generation probabilities or gradients, which Grey-box models typically rely on but are inaccessible in closed-source LLMs. However, their inability to capture query-response alignment patterns often results in lower detection accuracy. Additionally, the lack of large-scale benchmark datasets spanning diverse domains remains a challenge, as most existing datasets are limited in size and scope. To this end, we propose HalluCounter, a novel reference-free hallucination detection method that utilizes both response-response and query-response consistency and alignment patterns. This enables the training of a classifier that detects hallucinations and provides a confidence score and an optimal response for user queries. Furthermore, we introduce HalluCounterEval, a benchmark dataset comprising both synthetically generated and human-curated samples across multiple domains. Our method outperforms state-of-the-art approaches by a significant margin, achieving over 90\% average confidence in hallucination detection across datasets. 

---
# LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM 

**Authors**: Sambal Shikhar, Mohammed Irfan Kurpath, Sahal Shaji Mullappilly, Jean Lahoud, Fahad Khan, Rao Muhammad Anwer, Salman Khan, Hisham Cholakkal  

**Link**: [PDF](https://arxiv.org/pdf/2503.04724)  

**Abstract**: Recent advancements in speech-to-speech dialogue systems leverage LLMs for multimodal interactions, yet they remain hindered by fine-tuning requirements, high computational overhead, and text-speech misalignment. Existing speech-enabled LLMs often degrade conversational quality by modifying the LLM, thereby compromising its linguistic capabilities. In contrast, we propose LLMVoX, a lightweight 30M-parameter, LLM-agnostic, autoregressive streaming TTS system that generates high-quality speech with low latency, while fully preserving the capabilities of the base LLM. Our approach achieves a significantly lower Word Error Rate compared to speech-enabled LLMs, while operating at comparable latency and UTMOS score. By decoupling speech synthesis from LLM processing via a multi-queue token streaming system, LLMVoX supports seamless, infinite-length dialogues. Its plug-and-play design also facilitates extension to various tasks with different backbones. Furthermore, LLMVoX generalizes to new languages with only dataset adaptation, attaining a low Character Error Rate on an Arabic speech task. Additionally, we have integrated LLMVoX with a Vision-Language Model to create an omni-model with speech, text, and vision capabilities, without requiring additional multimodal training. Our code base and project page is available at this https URL . 

---
# Compositional Translation: A Novel LLM-based Approach for Low-resource Machine Translation 

**Authors**: Armel Zebaze, Benoît Sagot, Rachel Bawden  

**Link**: [PDF](https://arxiv.org/pdf/2503.04554)  

**Abstract**: The ability of generative large language models (LLMs) to perform in-context learning has given rise to a large body of research into how best to prompt models for various natural language processing tasks. Machine Translation (MT) has been shown to benefit from in-context examples, in particular when they are semantically similar to the sentence to translate. In this paper, we propose a new LLM-based translation paradigm, compositional translation, to replace naive few-shot MT with similarity-based demonstrations. An LLM is used to decompose a sentence into simpler phrases, and then to translate each phrase with the help of retrieved demonstrations. Finally, the LLM is prompted to translate the initial sentence with the help of the self-generated phrase-translation pairs. Our intuition is that this approach should improve translation because these shorter phrases should be intrinsically easier to translate and easier to match with relevant examples. This is especially beneficial in low-resource scenarios, and more generally whenever the selection pool is small or out of domain. We show that compositional translation boosts LLM translation performance on a wide range of popular MT benchmarks, including FLORES 200, NTREX 128 and TICO-19. Code and outputs are available at this https URL 

---
# An Empirical Study on Eliciting and Improving R1-like Reasoning Models 

**Authors**: Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, Lei Fang, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04548)  

**Abstract**: In this report, we present the third technical report on the development of slow-thinking models as part of the STILL project. As the technical pathway becomes clearer, scaling RL training has become a central technique for implementing such reasoning models. We systematically experiment with and document the effects of various factors influencing RL training, conducting experiments on both base models and fine-tuned models. Specifically, we demonstrate that our RL training approach consistently improves the Qwen2.5-32B base models, enhancing both response length and test accuracy. Furthermore, we show that even when a model like DeepSeek-R1-Distill-Qwen-1.5B has already achieved a high performance level, it can be further refined through RL training, reaching an accuracy of 39.33% on AIME 2024. Beyond RL training, we also explore the use of tool manipulation, finding that it significantly boosts the reasoning performance of large reasoning models. This approach achieves a remarkable accuracy of 86.67% with greedy search on AIME 2024, underscoring its effectiveness in enhancing model capabilities. We release our resources at the STILL project website: this https URL. 

---
# SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Modeling 

**Authors**: Xin Zhang, Qiyu Wei, Yingjie Zhu, Linhai Zhang, Deyu Zhou, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04619)  

**Abstract**: User reviews on e-commerce platforms exhibit dynamic sentiment patterns driven by temporal and contextual factors. Traditional sentiment analysis methods focus on static reviews, failing to capture the evolving temporal relationship between user sentiment rating and textual content. Sentiment analysis on streaming reviews addresses this limitation by modeling and predicting the temporal evolution of user sentiments. However, it suffers from data sparsity, manifesting in temporal, spatial, and combined forms. In this paper, we introduce SynGraph, a novel framework designed to address data sparsity in sentiment analysis on streaming reviews. SynGraph alleviates data sparsity by categorizing users into mid-tail, long-tail, and extreme scenarios and incorporating LLM-augmented enhancements within a dynamic graph-based structure. Experiments on real-world datasets demonstrate its effectiveness in addressing sparsity and improving sentiment modeling in streaming reviews. 

---
# Large Language Models in Bioinformatics: A Survey 

**Authors**: Zhenyu Wang, Zikang Wang, Jiyue Jiang, Pengan Chen, Xiangyu Shi, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04490)  

**Abstract**: Large Language Models (LLMs) are revolutionizing bioinformatics, enabling advanced analysis of DNA, RNA, proteins, and single-cell data. This survey provides a systematic review of recent advancements, focusing on genomic sequence modeling, RNA structure prediction, protein function inference, and single-cell transcriptomics. Meanwhile, we also discuss several key challenges, including data scarcity, computational complexity, and cross-omics integration, and explore future directions such as multimodal learning, hybrid AI models, and clinical applications. By offering a comprehensive perspective, this paper underscores the transformative potential of LLMs in driving innovations in bioinformatics and precision medicine. 

---
# Guiding LLMs to Generate High-Fidelity and High-Quality Counterfactual Explanations for Text Classification 

**Authors**: Van Bach Nguyen, Christin Seifert, Jörg Schlötterer  

**Link**: [PDF](https://arxiv.org/pdf/2503.04463)  

**Abstract**: The need for interpretability in deep learning has driven interest in counterfactual explanations, which identify minimal changes to an instance that change a model's prediction. Current counterfactual (CF) generation methods require task-specific fine-tuning and produce low-quality text. Large Language Models (LLMs), though effective for high-quality text generation, struggle with label-flipping counterfactuals (i.e., counterfactuals that change the prediction) without fine-tuning. We introduce two simple classifier-guided approaches to support counterfactual generation by LLMs, eliminating the need for fine-tuning while preserving the strengths of LLMs. Despite their simplicity, our methods outperform state-of-the-art counterfactual generation methods and are effective across different LLMs, highlighting the benefits of guiding counterfactual generation by LLMs with classifier information. We further show that data augmentation by our generated CFs can improve a classifier's robustness. Our analysis reveals a critical issue in counterfactual generation by LLMs: LLMs rely on parametric knowledge rather than faithfully following the classifier. 

---
# Shaping Shared Languages: Human and Large Language Models' Inductive Biases in Emergent Communication 

**Authors**: Tom Kouwenhoven, Max Peeperkorn, Roy de Kleijn, Tessa Verhoef  

**Link**: [PDF](https://arxiv.org/pdf/2503.04395)  

**Abstract**: Languages are shaped by the inductive biases of their users. Using a classical referential game, we investigate how artificial languages evolve when optimised for inductive biases in humans and large language models (LLMs) via Human-Human, LLM-LLM and Human-LLM experiments. We show that referentially grounded vocabularies emerge that enable reliable communication in all conditions, even when humans and LLMs collaborate. Comparisons between conditions reveal that languages optimised for LLMs subtly differ from those optimised for humans. Interestingly, interactions between humans and LLMs alleviate these differences and result in vocabularies which are more human-like than LLM-like. These findings advance our understanding of how inductive biases in LLMs play a role in the dynamic nature of human language and contribute to maintaining alignment in human and machine communication. In particular, our work underscores the need to think of new methods that include human interaction in the training processes of LLMs, and shows that using communicative success as a reward signal can be a fruitful, novel direction. 

---
# START: Self-taught Reasoner with Tools 

**Authors**: Chengpeng Li, Mingfeng Xue, Zhenru Zhang, Jiaxi Yang, Beichen Zhang, Xiang Wang, Bowen Yu, Binyuan Hui, Junyang Lin, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04625)  

**Abstract**: Large reasoning models (LRMs) like OpenAI-o1 and DeepSeek-R1 have demonstrated remarkable capabilities in complex reasoning tasks through the utilization of long Chain-of-thought (CoT). However, these models often suffer from hallucinations and inefficiencies due to their reliance solely on internal reasoning processes. In this paper, we introduce START (Self-Taught Reasoner with Tools), a novel tool-integrated long CoT reasoning LLM that significantly enhances reasoning capabilities by leveraging external tools. Through code execution, START is capable of performing complex computations, self-checking, exploring diverse methods, and self-debugging, thereby addressing the limitations of LRMs. The core innovation of START lies in its self-learning framework, which comprises two key techniques: 1) Hint-infer: We demonstrate that inserting artificially designed hints (e.g., ``Wait, maybe using Python here is a good idea.'') during the inference process of a LRM effectively stimulates its ability to utilize external tools without the need for any demonstration data. Hint-infer can also serve as a simple and effective sequential test-time scaling method; 2) Hint Rejection Sampling Fine-Tuning (Hint-RFT): Hint-RFT combines Hint-infer and RFT by scoring, filtering, and modifying the reasoning trajectories with tool invocation generated by a LRM via Hint-infer, followed by fine-tuning the LRM. Through this framework, we have fine-tuned the QwQ-32B model to achieve START. On PhD-level science QA (GPQA), competition-level math benchmarks (AMC23, AIME24, AIME25), and the competition-level code benchmark (LiveCodeBench), START achieves accuracy rates of 63.6%, 95.0%, 66.7%, 47.1%, and 47.3%, respectively. It significantly outperforms the base QwQ-32B and achieves performance comparable to the state-of-the-art open-weight model R1-Distill-Qwen-32B and the proprietary model o1-Preview. 

---
# TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models 

**Authors**: Xinyi He, Yihao Liu, Mengyu Zhou, Yeye He, Haoyu Dong, Shi Han, Zejian Yuan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04396)  

**Abstract**: Tabular data are crucial in many fields and their understanding by large language models (LLMs) under high parameter efficiency paradigm is important. However, directly applying parameter-efficient fine-tuning (PEFT) techniques to tabular tasks presents significant challenges, particularly in terms of better table serialization and the representation of two-dimensional structured information within a one-dimensional sequence. To address this, we propose TableLoRA, a module designed to improve LLMs' understanding of table structure during PEFT. It incorporates special tokens for serializing tables with special token encoder and uses 2D LoRA to encode low-rank information on cell positions. Experiments on four tabular-related datasets demonstrate that TableLoRA consistently outperforms vanilla LoRA and surpasses various table encoding methods tested in control experiments. These findings reveal that TableLoRA, as a table-specific LoRA, enhances the ability of LLMs to process tabular data effectively, especially in low-parameter settings, demonstrating its potential as a robust solution for handling table-related tasks. 

---
# Layer-Specific Scaling of Positional Encodings for Superior Long-Context Modeling 

**Authors**: Zhenghua Wang, Yiran Ding, Changze Lv, Zhibo Xu, Tianlong Li, Tianyuan Shi, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04355)  

**Abstract**: Although large language models (LLMs) have achieved significant progress in handling long-context inputs, they still suffer from the ``lost-in-the-middle'' problem, where crucial information in the middle of the context is often underrepresented or lost. Our extensive experiments reveal that this issue may arise from the rapid long-term decay in Rotary Position Embedding (RoPE). To address this problem, we propose a layer-specific positional encoding scaling method that assigns distinct scaling factors to each layer, slowing down the decay rate caused by RoPE to make the model pay more attention to the middle context. A specially designed genetic algorithm is employed to efficiently select the optimal scaling factors for each layer by incorporating Bezier curves to reduce the search space. Through comprehensive experimentation, we demonstrate that our method significantly alleviates the ``lost-in-the-middle'' problem. Our approach results in an average accuracy improvement of up to 20% on the Key-Value Retrieval dataset. Furthermore, we show that layer-specific interpolation, as opposed to uniform interpolation across all layers, enhances the model's extrapolation capabilities when combined with PI and Dynamic-NTK positional encoding schemes. 

---
# Exploring the Multilingual NLG Evaluation Abilities of LLM-Based Evaluators 

**Authors**: Jiayi Chang, Mingqi Gao, Xinyu Hu, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04360)  

**Abstract**: Previous research has shown that LLMs have potential in multilingual NLG evaluation tasks. However, existing research has not fully explored the differences in the evaluation capabilities of LLMs across different languages. To this end, this study provides a comprehensive analysis of the multilingual evaluation performance of 10 recent LLMs, spanning high-resource and low-resource languages through correlation analysis, perturbation attacks, and fine-tuning. We found that 1) excluding the reference answer from the prompt and using large-parameter LLM-based evaluators leads to better performance across various languages; 2) most LLM-based evaluators show a higher correlation with human judgments in high-resource languages than in low-resource languages; 3) in the languages where they are most sensitive to such attacks, they also tend to exhibit the highest correlation with human judgments; and 4) fine-tuning with data from a particular language yields a broadly consistent enhancement in the model's evaluation performance across diverse languages. Our findings highlight the imbalance in LLMs'evaluation capabilities across different languages and suggest that low-resource language scenarios deserve more attention. 

---
# Biological Sequence with Language Model Prompting: A Survey 

**Authors**: Jiyue Jiang, Zikang Wang, Yuheng Shan, Heyan Chai, Jiayi Li, Zixian Ma, Xinrui Zhang, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04135)  

**Abstract**: Large Language models (LLMs) have emerged as powerful tools for addressing challenges across diverse domains. Notably, recent studies have demonstrated that large language models significantly enhance the efficiency of biomolecular analysis and synthesis, attracting widespread attention from academics and medicine. In this paper, we systematically investigate the application of prompt-based methods with LLMs to biological sequences, including DNA, RNA, proteins, and drug discovery tasks. Specifically, we focus on how prompt engineering enables LLMs to tackle domain-specific problems, such as promoter sequence prediction, protein structure modeling, and drug-target binding affinity prediction, often with limited labeled data. Furthermore, our discussion highlights the transformative potential of prompting in bioinformatics while addressing key challenges such as data scarcity, multimodal fusion, and computational resource limitations. Our aim is for this paper to function both as a foundational primer for newcomers and a catalyst for continued innovation within this dynamic field of study. 

---
# Uncovering Gaps in How Humans and LLMs Interpret Subjective Language 

**Authors**: Erik Jones, Arjun Patrawala, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2503.04113)  

**Abstract**: Humans often rely on subjective natural language to direct language models (LLMs); for example, users might instruct the LLM to write an enthusiastic blogpost, while developers might train models to be helpful and harmless using LLM-based edits. The LLM's operational semantics of such subjective phrases -- how it adjusts its behavior when each phrase is included in the prompt -- thus dictates how aligned it is with human intent. In this work, we uncover instances of misalignment between LLMs' actual operational semantics and what humans expect. Our method, TED (thesaurus error detector), first constructs a thesaurus that captures whether two phrases have similar operational semantics according to the LLM. It then elicits failures by unearthing disagreements between this thesaurus and a human-constructed reference. TED routinely produces surprising instances of misalignment; for example, Mistral 7B Instruct produces more harassing outputs when it edits text to be witty, and Llama 3 8B Instruct produces dishonest articles when instructed to make the articles enthusiastic. Our results demonstrate that humans can uncover unexpected LLM behavior by scrutinizing relationships between abstract concepts, without supervising outputs directly. 

---
# FuseChat-3.0: Preference Optimization Meets Heterogeneous Model Fusion 

**Authors**: Ziyi Yang, Fanqi Wan, Longguang Zhong, Canbin Huang, Guosheng Liang, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04222)  

**Abstract**: We introduce FuseChat-3.0, a suite of large language models (LLMs) developed by integrating the strengths of heterogeneous source LLMs into more compact target LLMs. Our source models include the powerful Gemma-2-27B-it, Mistral-Large-Instruct-2407, Qwen-2.5-72B-Instruct, and Llama-3.1-70B-Instruct. For target models, we focus on three widely-used smaller variants-Llama-3.1-8B-Instruct, Gemma-2-9B-it, and Qwen-2.5-7B-Instruct-along with two ultra-compact options, Llama-3.2-3B-Instruct and Llama-3.2-1B-Instruct. To leverage the diverse capabilities of these source models, we develop a specialized data construction protocol tailored to various tasks and domains. The FuseChat-3.0 training pipeline consists of two key stages: (1) supervised fine-tuning (SFT) to align the target and source model distributions, and (2) Direct Preference Optimization (DPO) to apply preferences from multiple source LLMs to fine-tune the target model. The resulting FuseChat-3.0 models exhibit significant performance gains across tasks such as instruction following, general knowledge, mathematics, and coding. As illustrated in Figure 1, using Llama-3.1-8B-Instruct as the target model, our fusion approach achieves an average improvement of 6.8 points across 14 benchmarks. Moreover, it demonstrates remarkable gains of 37.1 points and 30.1 points on the instruction-following benchmarks AlpacaEval-2 and Arena-Hard, respectively. Our code, models, and datasets are available at this https URL. 

---
# ReasonGraph: Visualisation of Reasoning Paths 

**Authors**: Zongqian Li, Ehsan Shareghi, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2503.03979)  

**Abstract**: Large Language Models (LLMs) reasoning processes are challenging to analyze due to their complexity and the lack of organized visualization tools. We present ReasonGraph, a web-based platform for visualizing and analyzing LLM reasoning processes. It supports both sequential and tree-based reasoning methods while integrating with major LLM providers and over fifty state-of-the-art models. ReasonGraph incorporates an intuitive UI with meta reasoning method selection, configurable visualization parameters, and a modular framework that facilitates efficient extension. Our evaluation shows high parsing reliability, efficient processing, and strong usability across various downstream applications. By providing a unified visualization framework, ReasonGraph reduces cognitive load in analyzing complex reasoning paths, improves error detection in logical processes, and enables more effective development of LLM-based applications. The platform is open-source, promoting accessibility and reproducibility in LLM reasoning analysis. 

---
# Tgea: An error-annotated dataset and benchmark tasks for text generation from pretrained language models 

**Authors**: Jie He, Bo Peng, Yi Liao, Qun Liu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.04232)  

**Abstract**: In order to deeply understand the capability of pretrained language models in text generation and conduct a diagnostic evaluation, we propose TGEA, an error-annotated dataset with multiple benchmark tasks for text generation from pretrained language models (PLMs). We use carefully selected prompt words to guide GPT-2 to generate candidate sentences, from which we select 47K for error annotation. Crowdsourced workers manually check each of these sentences and detect 12k erroneous sentences. We create an error taxonomy to cover 24 types of errors occurring in these erroneous sentences according to the nature of errors with respect to linguistics and knowledge (eg, common sense). For each erroneous span in PLM-generated sentences, we also detect another span that is closely associated with it. Each error is hence manually labeled with comprehensive annotations, including the span of the error, the associated span, minimal correction to the error, the type of the error, and rationale behind the error. Apart from the fully annotated dataset, we also present a detailed description of the data collection procedure, statistics and analysis of the dataset. This is the first dataset with comprehensive annotations for PLM-generated texts, which facilitates the diagnostic evaluation of PLM-based text generation. Furthermore, we use TGEA as a benchmark dataset and propose a series of automatic diagnosis tasks, including error detection, error type classification, associated span detection, error rationale generation, to further promote future study on the automatic error detection and correction on texts generated by pretrained language models. 

---
# On Fact and Frequency: LLM Responses to Misinformation Expressed with Uncertainty 

**Authors**: Yana van de Sande, Gunes Açar, Thabo van Woudenberg, Martha Larson  

**Link**: [PDF](https://arxiv.org/pdf/2503.04271)  

**Abstract**: We study LLM judgments of misinformation expressed with uncertainty. Our experiments study the response of three widely used LLMs (GPT-4o, LlaMA3, DeepSeek-v2) to misinformation propositions that have been verified false and then are transformed into uncertain statements according to an uncertainty typology. Our results show that after transformation, LLMs change their factchecking classification from false to not-false in 25% of the cases. Analysis reveals that the change cannot be explained by predictors to which humans are expected to be sensitive, i.e., modality, linguistic cues, or argumentation strategy. The exception is doxastic transformations, which use linguistic cue phrases such as "It is believed ...".To gain further insight, we prompt the LLM to make another judgment about the transformed misinformation statements that is not related to truth value. Specifically, we study LLM estimates of the frequency with which people make the uncertain statement. We find a small but significant correlation between judgment of fact and estimation of frequency. 

---
# Tec-Habilidad: Skill Classification for Bridging Education and Employment 

**Authors**: Sabur Butt, Hector G. Ceballos, Diana P. Madera  

**Link**: [PDF](https://arxiv.org/pdf/2503.03932)  

**Abstract**: Job application and assessment processes have evolved significantly in recent years, largely due to advancements in technology and changes in the way companies operate. Skill extraction and classification remain an important component of the modern hiring process as it provides a more objective way to evaluate candidates and automatically align their skills with the job requirements. However, to effectively evaluate the skills, the skill extraction tools must recognize varied mentions of skills on resumes, including direct mentions, implications, synonyms, acronyms, phrases, and proficiency levels, and differentiate between hard and soft skills. While tools like LLMs (Large Model Models) help extract and categorize skills from job applications, there's a lack of comprehensive datasets for evaluating the effectiveness of these models in accurately identifying and classifying skills in Spanish-language job applications. This gap hinders our ability to assess the reliability and precision of the models, which is crucial for ensuring that the selected candidates truly possess the required skills for the job. In this paper, we develop a Spanish language dataset for skill extraction and classification, provide annotation methodology to distinguish between knowledge, skill, and abilities, and provide deep learning baselines to advance robust solutions for skill classification. 

---
# Performance Comparison of Large Language Models on Advanced Calculus Problems 

**Authors**: In Hak Moon  

**Link**: [PDF](https://arxiv.org/pdf/2503.03960)  

**Abstract**: This paper presents an in-depth analysis of the performance of seven different Large Language Models (LLMs) in solving a diverse set of math advanced calculus problems. The study aims to evaluate these models' accuracy, reliability, and problem-solving capabilities, including ChatGPT 4o, Gemini Advanced with 1.5 Pro, Copilot Pro, Claude 3.5 Sonnet, Meta AI, Mistral AI, and Perplexity. The assessment was conducted through a series of thirty-two test problems, encompassing a total of 320 points. The problems covered various topics, from vector calculations and geometric interpretations to integral evaluations and optimization tasks. The results highlight significant trends and patterns in the models' performance, revealing both their strengths and weaknesses - for instance, models like ChatGPT 4o and Mistral AI demonstrated consistent accuracy across various problem types, indicating their robustness and reliability in mathematical problem-solving, while models such as Gemini Advanced with 1.5 Pro and Meta AI exhibited specific weaknesses, particularly in complex problems involving integrals and optimization, suggesting areas for targeted improvements. The study also underscores the importance of re-prompting in achieving accurate solutions, as seen in several instances where models initially provided incorrect answers but corrected them upon re-prompting. Overall, this research provides valuable insights into the current capabilities and limitations of LLMs in the domain of math calculus, with the detailed analysis of each model's performance on specific problems offering a comprehensive understanding of their strengths and areas for improvement, contributing to the ongoing development and refinement of LLM technology. The findings are particularly relevant for educators, researchers, and developers seeking to leverage LLMs for educational and practical applications in mathematics. 

---
# LLMs Can Generate a Better Answer by Aggregating Their Own Responses 

**Authors**: Zichong Li, Xinyu Feng, Yuheng Cai, Zixuan Zhang, Tianyi Liu, Chen Liang, Weizhu Chen, Haoyu Wang, Tuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04104)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across tasks, yet they often require additional prompting techniques when facing complex problems. While approaches like self-correction and response selection have emerged as popular solutions, recent studies have shown these methods perform poorly when relying on the LLM itself to provide feedback or selection criteria. We argue this limitation stems from the fact that common LLM post-training procedures lack explicit supervision for discriminative judgment tasks. In this paper, we propose Generative Self-Aggregation (GSA), a novel prompting method that improves answer quality without requiring the model's discriminative capabilities. GSA first samples multiple diverse responses from the LLM, then aggregates them to obtain an improved solution. Unlike previous approaches, our method does not require the LLM to correct errors or compare response quality; instead, it leverages the model's generative abilities to synthesize a new response based on the context of multiple samples. While GSA shares similarities with the self-consistency (SC) approach for response aggregation, SC requires specific verifiable tokens to enable majority voting. In contrast, our approach is more general and can be applied to open-ended tasks. Empirical evaluation demonstrates that GSA effectively improves response quality across various tasks, including mathematical reasoning, knowledge-based problems, and open-ended generation tasks such as code synthesis and conversational responses. 

---
# AI for Scaling Legal Reform: Mapping and Redacting Racial Covenants in Santa Clara County 

**Authors**: Faiz Surani, Mirac Suzgun, Vyoma Raman, Christopher D. Manning, Peter Henderson, Daniel E. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2503.03888)  

**Abstract**: Many jurisdictions have moved to identify and strike these provisions, including California, which mandated in 2021 that all counties implement such a process. Yet the scale can be overwhelming, with Santa Clara County (SCC) alone having over 24 million property deed documents, making purely manual review infeasible. We present a novel approach to addressing this pressing issue, developed through a partnership with the SCC Clerk-Recorder's Office. First, we leverage an open large language model, fine-tuned to detect racial covenants with high precision and recall. We estimate that this system reduces manual efforts by 86,500 person hours and costs less than 2% of the cost for a comparable off-the-shelf closed model. Second, we illustrate the County's integration of this model into responsible operational practice, including legal review and the creation of a historical registry, and release our model to assist the hundreds of jurisdictions engaged in similar efforts. Finally, our results reveal distinct periods of utilization of racial covenants, sharp geographic clustering, and the disproportionate role of a small number of developers in maintaining housing discrimination. We estimate that by 1950, one in four properties across the County were subject to racial covenants. 

---
# HEISIR: Hierarchical Expansion of Inverted Semantic Indexing for Training-free Retrieval of Conversational Data using LLMs 

**Authors**: Sangyeop Kim, Hangyeul Lee, Yohan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.04141)  

**Abstract**: The growth of conversational AI services has increased demand for effective information retrieval from dialogue data. However, existing methods often face challenges in capturing semantic intent or require extensive labeling and fine-tuning. This paper introduces HEISIR (Hierarchical Expansion of Inverted Semantic Indexing for Retrieval), a novel framework that enhances semantic understanding in conversational data retrieval through optimized data ingestion, eliminating the need for resource-intensive labeling or model adaptation. HEISIR implements a two-step process: (1) Hierarchical Triplets Formulation and (2) Adjunct Augmentation, creating semantic indices consisting of Subject-Verb-Object-Adjunct (SVOA) quadruplets. This structured representation effectively captures the underlying semantic information from dialogue content. HEISIR achieves high retrieval performance while maintaining low latency during the actual retrieval process. Our experimental results demonstrate that HEISIR outperforms fine-tuned models across various embedding types and language models. Beyond improving retrieval capabilities, HEISIR also offers opportunities for intent and topic analysis in conversational data, providing a versatile solution for dialogue systems. 

---
# TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge 

**Authors**: Cheng-Han Chiang, Hung-yi Lee, Michal Lukasik  

**Link**: [PDF](https://arxiv.org/pdf/2503.04381)  

**Abstract**: The LLM-as-a-judge paradigm uses large language models (LLMs) for automated text evaluation, where a numerical assessment is assigned by an LLM to the input text following scoring rubrics. Existing methods for LLM-as-a-judge use cross-entropy (CE) loss for fine-tuning, which neglects the numeric nature of score prediction. Recent work addresses numerical prediction limitations of LLM fine-tuning through regression-aware fine-tuning, which, however, does not consider chain-of-thought (CoT) reasoning for score prediction. In this paper, we introduce TRACT (Two-stage Regression-Aware fine-tuning with CoT), a method combining CoT reasoning with regression-aware training. TRACT consists of two stages: first, seed LLM is fine-tuned to generate CoTs, which serve as supervision for the second stage fine-tuning. The training objective of TRACT combines the CE loss for learning the CoT reasoning capabilities, and the regression-aware loss for the score prediction. Experiments across four LLM-as-a-judge datasets and two LLMs show that TRACT significantly outperforms existing methods. Extensive ablation studies validate the importance of each component in TRACT. 

---
