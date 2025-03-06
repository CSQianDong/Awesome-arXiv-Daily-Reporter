# Leveraging Large Language Models to Develop Heuristics for Emerging Optimization Problems 

**Authors**: Thomas Bömer, Nico Koltermann, Max Disselnmeyer, Laura Dörr, Anne Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.03350)  

**Abstract**: Combinatorial optimization problems often rely on heuristic algorithms to generate efficient solutions. However, the manual design of heuristics is resource-intensive and constrained by the designer's expertise. Recent advances in artificial intelligence, particularly large language models (LLMs), have demonstrated the potential to automate heuristic generation through evolutionary frameworks. Recent works focus only on well-known combinatorial optimization problems like the traveling salesman problem and online bin packing problem when designing constructive heuristics. This study investigates whether LLMs can effectively generate heuristics for niche, not yet broadly researched optimization problems, using the unit-load pre-marshalling problem as an example case. We propose the Contextual Evolution of Heuristics (CEoH) framework, an extension of the Evolution of Heuristics (EoH) framework, which incorporates problem-specific descriptions to enhance in-context learning during heuristic generation. Through computational experiments, we evaluate CEoH and EoH and compare the results. Results indicate that CEoH enables smaller LLMs to generate high-quality heuristics more consistently and even outperform larger models. Larger models demonstrate robust performance with or without contextualized prompts. The generated heuristics exhibit scalability to diverse instance configurations. 

---
# Towards Understanding Multi-Round Large Language Model Reasoning: Approximability, Learnability and Generalizability 

**Authors**: Chenhui Xu, Dancheng Liu, Jiajie Li, Amir Nassereldine, Zhaohui Li, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03128)  

**Abstract**: Recent advancements in cognitive science and multi-round reasoning techniques for Large Language Models (LLMs) suggest that iterative thinking processes improve problem-solving performance in complex tasks. Inspired by this, approaches like Chain-of-Thought, debating, and self-refinement have been applied to auto-regressive LLMs, achieving significant successes in tasks such as mathematical reasoning, commonsense reasoning, and multi-hop question answering. Despite these successes, the theoretical basis for how multi-round reasoning enhances problem-solving abilities remains underexplored. In this work, we investigate the approximation, learnability, and generalization properties of multi-round auto-regressive models. We show that Transformers with finite context windows are universal approximators for steps of Turing-computable functions and can approximate any Turing-computable sequence-to-sequence function through multi-round reasoning. We extend PAC learning to sequence generation and demonstrate that multi-round generation is learnable even when the sequence length exceeds the model's context window. Finally, we examine how generalization error propagates across rounds, and show how the aforementioned approaches can help constrain this error, ensuring outputs stay within an expectation boundary. This work sheds light on the systemic theoretical foundations of multi-round sequence learning and reasoning, emphasizing its role in inference complexity. 

---
# Teaching AI to Handle Exceptions: Supervised Fine-Tuning with Human-Aligned Judgment 

**Authors**: Matthew DosSantos DiSorbo, Harang Ju, Sinan Aral  

**Link**: [PDF](https://arxiv.org/pdf/2503.02976)  

**Abstract**: Large language models (LLMs), initially developed for generative AI, are now evolving into agentic AI systems, which make decisions in complex, real-world contexts. Unfortunately, while their generative capabilities are well-documented, their decision-making processes remain poorly understood. This is particularly evident when models are handling exceptions, a critical and challenging aspect of decision-making made relevant by the inherent incompleteness of contracts. Here we demonstrate that LLMs, even ones that excel at reasoning, deviate significantly from human judgments because they adhere strictly to policies, even when such adherence is impractical, suboptimal, or even counterproductive. We then evaluate three approaches to tuning AI agents to handle exceptions: ethical framework prompting, chain-of-thought reasoning, and supervised fine-tuning. We find that while ethical framework prompting fails and chain-of-thought prompting provides only slight improvements, supervised fine-tuning, specifically with human explanations, yields markedly better results. Surprisingly, in our experiments, supervised fine-tuning even enabled models to generalize human-like decision-making to novel scenarios, demonstrating transfer learning of human-aligned decision-making across contexts. Furthermore, fine-tuning with explanations, not just labels, was critical for alignment, suggesting that aligning LLMs with human judgment requires explicit training on how decisions are made, not just which decisions are made. These findings highlight the need to address LLMs' shortcomings in handling exceptions in order to guide the development of agentic AI toward models that can effectively align with human judgment and simultaneously adapt to novel contexts. 

---
# The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems 

**Authors**: Richard Ren, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler, Mick Yang, Isabelle Barrass, Alice Gatti, Xuwang Yin, Eduardo Trevino, Matias Geralnik, Adam Khoja, Dean Lee, Summer Yue, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2503.03750)  

**Abstract**: As large language models (LLMs) become more capable and agentic, the requirement for trust in their outputs grows significantly, yet at the same time concerns have been mounting that models may learn to lie in pursuit of their goals. To address these concerns, a body of work has emerged around the notion of "honesty" in LLMs, along with interventions aimed at mitigating deceptive behaviors. However, evaluations of honesty are currently highly limited, with no benchmark combining large scale and applicability to all models. Moreover, many benchmarks claiming to measure honesty in fact simply measure accuracy--the correctness of a model's beliefs--in disguise. In this work, we introduce a large-scale human-collected dataset for measuring honesty directly, allowing us to disentangle accuracy from honesty for the first time. Across a diverse set of LLMs, we find that while larger models obtain higher accuracy on our benchmark, they do not become more honest. Surprisingly, while most frontier LLMs obtain high scores on truthfulness benchmarks, we find a substantial propensity in frontier LLMs to lie when pressured to do so, resulting in low honesty scores on our benchmark. We find that simple methods, such as representation engineering interventions, can improve honesty. These results underscore the growing need for robust evaluations and effective interventions to ensure LLMs remain trustworthy. 

---
# Process-based Self-Rewarding Language Models 

**Authors**: Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03746)  

**Abstract**: Large Language Models have demonstrated outstanding performance across various downstream tasks and have been widely applied in multiple scenarios. Human-annotated preference data is used for training to further improve LLMs' performance, which is constrained by the upper limit of human performance. Therefore, Self-Rewarding method has been proposed, where LLMs generate training data by rewarding their own outputs. However, the existing self-rewarding paradigm is not effective in mathematical reasoning scenarios and may even lead to a decline in performance. In this work, we propose the Process-based Self-Rewarding pipeline for language models, which introduces long-thought reasoning, step-wise LLM-as-a-Judge, and step-wise preference optimization within the self-rewarding paradigm. Our new paradigm successfully enhances the performance of LLMs on multiple mathematical reasoning benchmarks through iterative Process-based Self-Rewarding, demonstrating the immense potential of self-rewarding to achieve LLM reasoning that may surpass human capabilities. 

---
# Attentive Reasoning Queries: A Systematic Method for Optimizing Instruction-Following in Large Language Models 

**Authors**: Bar Karov, Dor Zohar, Yam Marcovitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.03669)  

**Abstract**: We present Attentive Reasoning Queries (ARQs), a novel structured reasoning approach that significantly improves instruction-following in Large Language Models through domain-specialized reasoning blueprints. While LLMs demonstrate remarkable capabilities across diverse tasks, they often fail to maintain adherence to complex, use-case-specific instructions during multi-turn conversations, presenting challenges for business-critical applications. ARQs address this limitation by guiding LLMs through systematic reasoning steps with targeted queries that reinstate critical instructions and facilitate intermediate reasoning throughout the completion process. In extensive testing within Parlant, our framework for reliable customer-facing agents in which ARQs were born out of necessity, they achieved a 90.2% success rate across 87 test scenarios, outperforming both Chain-of-Thought reasoning (86.1%) and direct response generation (81.5%). ARQs showed particular strength in addressing persistent failure modes like guideline re-application and hallucination prevention. Our analysis also revealed that ARQs can potentially be more computationally efficient than free-form reasoning when carefully designed. These findings demonstrate that structured reasoning approaches provide effective mechanisms for controlling how LLMs process information and make decisions in complex scenarios. 

---
# COSINT-Agent: A Knowledge-Driven Multimodal Agent for Chinese Open Source Intelligence 

**Authors**: Wentao Li, Congcong Wang, Xiaoxiao Cui, Zhi Liu, Wei Guo, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.03215)  

**Abstract**: Open Source Intelligence (OSINT) requires the integration and reasoning of diverse multimodal data, presenting significant challenges in deriving actionable insights. Traditional approaches, including multimodal large language models (MLLMs), often struggle to infer complex contextual relationships or deliver comprehensive intelligence from unstructured data sources. In this paper, we introduce COSINT-Agent, a knowledge-driven multimodal agent tailored to address the challenges of OSINT in the Chinese domain. COSINT-Agent seamlessly integrates the perceptual capabilities of fine-tuned MLLMs with the structured reasoning power of the Entity-Event-Scene Knowledge Graph (EES-KG). Central to COSINT-Agent is the innovative EES-Match framework, which bridges COSINT-MLLM and EES-KG, enabling systematic extraction, reasoning, and contextualization of multimodal insights. This integration facilitates precise entity recognition, event interpretation, and context retrieval, effectively transforming raw multimodal data into actionable intelligence. Extensive experiments validate the superior performance of COSINT-Agent across core OSINT tasks, including entity recognition, EES generation, and context matching. These results underscore its potential as a robust and scalable solution for advancing automated multimodal reasoning and enhancing the effectiveness of OSINT methodologies. 

---
# Collaborative Expert LLMs Guided Multi-Objective Molecular Optimization 

**Authors**: Jiajun Yu, Yizhen Zheng, Huan Yee Koh, Shirui Pan, Tianyue Wang, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03503)  

**Abstract**: Molecular optimization is a crucial yet complex and time-intensive process that often acts as a bottleneck for drug development. Traditional methods rely heavily on trial and error, making multi-objective optimization both time-consuming and resource-intensive. Current AI-based methods have shown limited success in handling multi-objective optimization tasks, hampering their practical utilization. To address this challenge, we present MultiMol, a collaborative large language model (LLM) system designed to guide multi-objective molecular optimization. MultiMol comprises two agents, including a data-driven worker agent and a literature-guided research agent. The data-driven worker agent is a large language model being fine-tuned to learn how to generate optimized molecules considering multiple objectives, while the literature-guided research agent is responsible for searching task-related literature to find useful prior knowledge that facilitates identifying the most promising optimized candidates. In evaluations across six multi-objective optimization tasks, MultiMol significantly outperforms existing methods, achieving a 82.30% success rate, in sharp contrast to the 27.50% success rate of current strongest methods. To further validate its practical impact, we tested MultiMol on two real-world challenges. First, we enhanced the selectivity of Xanthine Amine Congener (XAC), a promiscuous ligand that binds both A1R and A2AR, successfully biasing it towards A1R. Second, we improved the bioavailability of Saquinavir, an HIV-1 protease inhibitor with known bioavailability limitations. Overall, these results indicate that MultiMol represents a highly promising approach for multi-objective molecular optimization, holding great potential to accelerate the drug development process and contribute to the advancement of pharmaceutical research. 

---
# Unified Mind Model: Reimagining Autonomous Agents in the LLM Era 

**Authors**: Pengbo Hu, Xiang Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.03459)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable capabilities across domains, tasks, and languages (e.g., ChatGPT and GPT-4), reviving the research of general autonomous agents with human-like cognitive this http URL human-level agents require semantic comprehension and instruction-following capabilities, which exactly fall into the strengths of this http URL there have been several initial attempts to build human-level agents based on LLMs, the theoretical foundation remains a challenging open problem. In this paper, we propose a novel theoretical cognitive architecture, the Unified Mind Model (UMM), which offers guidance to facilitate the rapid creation of autonomous agents with human-level cognitive abilities. Specifically, our UMM starts with the global workspace theory and further leverage LLMs to enable the agent with various cognitive abilities, such as multi-modal perception, planning, reasoning, tool use, learning, memory, reflection and motivation. Building upon UMM, we then develop an agent-building engine, MindOS, which allows users to quickly create domain-/task-specific autonomous agents without any programming effort. 

---
# Improving Neutral Point of View Text Generation through Parameter-Efficient Reinforcement Learning and a Small-Scale High-Quality Dataset 

**Authors**: Jessica Hoffmann, Christiane Ahlheim, Zac Yu, Aria Walfrand, Jarvis Jin, Marie Tano, Ahmad Beirami, Erin van Liemt, Nithum Thain, Hakim Sidahmed, Lucas Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2503.03654)  

**Abstract**: This paper describes the construction of a dataset and the evaluation of training methods to improve generative large language models' (LLMs) ability to answer queries on sensitive topics with a Neutral Point of View (NPOV), i.e., to provide significantly more informative, diverse and impartial answers. The dataset, the SHQ-NPOV dataset, comprises 300 high-quality, human-written quadruplets: a query on a sensitive topic, an answer, an NPOV rating, and a set of links to source texts elaborating the various points of view. The first key contribution of this paper is a new methodology to create such datasets through iterative rounds of human peer-critique and annotator training, which we release alongside the dataset. The second key contribution is the identification of a highly effective training regime for parameter-efficient reinforcement learning (PE-RL) to improve NPOV generation. We compare and extensively evaluate PE-RL and multiple baselines-including LoRA finetuning (a strong baseline), SFT and RLHF.
PE-RL not only improves on overall NPOV quality compared to the strongest baseline ($97.06\%\rightarrow 99.08\%$), but also scores much higher on features linguists identify as key to separating good answers from the best answers ($60.25\%\rightarrow 85.21\%$ for presence of supportive details, $68.74\%\rightarrow 91.43\%$ for absence of oversimplification). A qualitative analysis corroborates this. Finally, our evaluation finds no statistical differences between results on topics that appear in the training dataset and those on separated evaluation topics, which provides strong evidence that our approach to training PE-RL exhibits very effective out of topic generalization. 

---
# Open-Source Large Language Models as Multilingual Crowdworkers: Synthesizing Open-Domain Dialogues in Several Languages With No Examples in Targets and No Machine Translation 

**Authors**: Ahmed Njifenjou, Virgile Sucal, Bassam Jabaian, Fabrice Lefèvre  

**Link**: [PDF](https://arxiv.org/pdf/2503.03462)  

**Abstract**: The prevailing paradigm in the domain of Open-Domain Dialogue agents predominantly focuses on the English language, encompassing both models and datasets. Furthermore, the financial and temporal investments required for crowdsourcing such datasets for finetuning are substantial, particularly when multiple languages are involved. Fortunately, advancements in Large Language Models (LLMs) have unveiled a plethora of possibilities across diverse tasks. Specifically, instruction-tuning has enabled LLMs to execute tasks based on natural language instructions, occasionally surpassing the performance of human crowdworkers. Additionally, these models possess the capability to function in various languages within a single thread. Consequently, to generate new samples in different languages, we propose leveraging these capabilities to replicate the data collection process. We introduce a pipeline for generating Open-Domain Dialogue data in multiple Target Languages using LLMs, with demonstrations provided in a unique Source Language. By eschewing explicit Machine Translation in this approach, we enhance the adherence to language-specific nuances. We apply this methodology to the PersonaChat dataset. To enhance the openness of generated dialogues and mimic real life scenarii, we added the notion of speech events corresponding to the type of conversation the speakers are involved in and also that of common ground which represents the premises of a conversation. 

---
# RASD: Retrieval-Augmented Speculative Decoding 

**Authors**: Guofeng Quan, Wenfeng Feng, Chuzhan Hao, Guochao Jiang, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03434)  

**Abstract**: Speculative decoding accelerates inference in large language models (LLMs) by generating draft tokens for target model verification. Current approaches for obtaining draft tokens rely on lightweight draft models or additional model structures to generate draft tokens and retrieve context from databases. Due to the draft model's small size and limited training data, model-based speculative decoding frequently becomes less effective in out-of-domain scenarios. Additionally, the time cost of the drafting phase results in a low upper limit on acceptance length during the verification step, limiting overall efficiency. This paper proposes RASD (Retrieval-Augmented Speculative Decoding), which adopts retrieval methods to enhance model-based speculative decoding. We introduce tree pruning and tree fusion to achieve this. Specifically, we develop a pruning method based on the draft model's probability distribution to construct the optimal retrieval tree. Second, we employ the longest prefix matching algorithm to merge the tree generated by the draft model with the retrieval tree, resulting in a unified tree for verification. Experimental results demonstrate that RASD achieves state-of-the-art inference acceleration across tasks such as DocQA, Summary, Code, and In-Domain QA. Moreover, RASD exhibits strong scalability, seamlessly integrating with various speculative decoding approaches, including both generation-based and retrieval-based methods. 

---
# Taxation Perspectives from Large Language Models: A Case Study on Additional Tax Penalties 

**Authors**: Eunkyung Choi, Young Jin Suh, Hun Park, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03444)  

**Abstract**: How capable are large language models (LLMs) in the domain of taxation? Although numerous studies have explored the legal domain in general, research dedicated to taxation remain scarce. Moreover, the datasets used in these studies are either simplified, failing to reflect the real-world complexities, or unavailable as open source. To address this gap, we introduce PLAT, a new benchmark designed to assess the ability of LLMs to predict the legitimacy of additional tax penalties. PLAT is constructed to evaluate LLMs' understanding of tax law, particularly in cases where resolving the issue requires more than just applying related statutes. Our experiments with six LLMs reveal that their baseline capabilities are limited, especially when dealing with conflicting issues that demand a comprehensive understanding. However, we found that enabling retrieval, self-reasoning, and discussion among multiple agents with specific role assignments, this limitation can be mitigated. 

---
# Exploring the Potential of Large Language Models as Predictors in Dynamic Text-Attributed Graphs 

**Authors**: Runlin Lei, Jiarui Ji, Haipeng Ding, Lu Yi, Zhewei Wei, Yongchao Liu, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03258)  

**Abstract**: With the rise of large language models (LLMs), there has been growing interest in Graph Foundation Models (GFMs) for graph-based tasks. By leveraging LLMs as predictors, GFMs have demonstrated impressive generalizability across various tasks and datasets. However, existing research on LLMs as predictors has predominantly focused on static graphs, leaving their potential in dynamic graph prediction unexplored. In this work, we pioneer using LLMs for predictive tasks on dynamic graphs. We identify two key challenges: the constraints imposed by context length when processing large-scale historical data and the significant variability in domain characteristics, both of which complicate the development of a unified predictor. To address these challenges, we propose the GraphAgent-Dynamic (GAD) Framework, a multi-agent system that leverages collaborative LLMs. In contrast to using a single LLM as the predictor, GAD incorporates global and local summary agents to generate domain-specific knowledge, enhancing its transferability across domains. Additionally, knowledge reflection agents enable adaptive updates to GAD's knowledge, maintaining a unified and self-consistent architecture. In experiments, GAD demonstrates performance comparable to or even exceeds that of full-supervised graph neural networks without dataset-specific training. Finally, to enhance the task-specific performance of LLM-based predictors, we discuss potential improvements, such as dataset-specific fine-tuning to LLMs. By developing tailored strategies for different tasks, we provide new insights for the future design of LLM-based predictors. 

---
# Towards Robust Universal Information Extraction: Benchmark, Evaluation, and Solution 

**Authors**: Jizhao Zhu, Akang Shi, Zixuan Li, Long Bai, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.03201)  

**Abstract**: In this paper, we aim to enhance the robustness of Universal Information Extraction (UIE) by introducing a new benchmark dataset, a comprehensive evaluation, and a feasible solution. Existing robust benchmark datasets have two key limitations: 1) They generate only a limited range of perturbations for a single Information Extraction (IE) task, which fails to evaluate the robustness of UIE models effectively; 2) They rely on small models or handcrafted rules to generate perturbations, often resulting in unnatural adversarial examples. Considering the powerful generation capabilities of Large Language Models (LLMs), we introduce a new benchmark dataset for Robust UIE, called RUIE-Bench, which utilizes LLMs to generate more diverse and realistic perturbations across different IE tasks. Based on this dataset, we comprehensively evaluate existing UIE models and reveal that both LLM-based models and other models suffer from significant performance drops. To improve robustness and reduce training costs, we propose a data-augmentation solution that dynamically selects hard samples for iterative training based on the model's inference loss. Experimental results show that training with only \textbf{15\%} of the data leads to an average \textbf{7.5\%} relative performance improvement across three IE tasks. 

---
# Structured Outputs Enable General-Purpose LLMs to be Medical Experts 

**Authors**: Guangfu Guo, Kai Zhang, Bryan Hoo, Yujun Cai, Xiaoqian Lu, Nanyun Peng, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03194)  

**Abstract**: Medical question-answering (QA) is a critical task for evaluating how effectively large language models (LLMs) encode clinical knowledge and assessing their potential applications in medicine. Despite showing promise on multiple-choice tests, LLMs frequently struggle with open-ended medical questions, producing responses with dangerous hallucinations or lacking comprehensive coverage of critical aspects. Existing approaches attempt to address these challenges through domain-specific fine-tuning, but this proves resource-intensive and difficult to scale across models. To improve the comprehensiveness and factuality of medical responses, we propose a novel approach utilizing structured medical reasoning. Our method guides LLMs through an seven-step cognitive process inspired by clinical diagnosis, enabling more accurate and complete answers without additional training. Experiments on the MedLFQA benchmark demonstrate that our approach achieves the highest Factuality Score of 85.8, surpassing fine-tuned models. Notably, this improvement transfers to smaller models, highlighting the method's efficiency and scalability. Our code and datasets are available. 

---
# AttackSeqBench: Benchmarking Large Language Models' Understanding of Sequential Patterns in Cyber Attacks 

**Authors**: Javier Yong, Haokai Ma, Yunshan Ma, Anis Yusof, Zhenkai Liang, Ee-Chien Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03170)  

**Abstract**: The observations documented in Cyber Threat Intelligence (CTI) reports play a critical role in describing adversarial behaviors, providing valuable insights for security practitioners to respond to evolving threats. Recent advancements of Large Language Models (LLMs) have demonstrated significant potential in various cybersecurity applications, including CTI report understanding and attack knowledge graph construction. While previous works have proposed benchmarks that focus on the CTI extraction ability of LLMs, the sequential characteristic of adversarial behaviors within CTI reports remains largely unexplored, which holds considerable significance in developing a comprehensive understanding of how adversaries operate. To address this gap, we introduce AttackSeqBench, a benchmark tailored to systematically evaluate LLMs' capability to understand and reason attack sequences in CTI reports. Our benchmark encompasses three distinct Question Answering (QA) tasks, each task focuses on the varying granularity in adversarial behavior. To alleviate the laborious effort of QA construction, we carefully design an automated dataset construction pipeline to create scalable and well-formulated QA datasets based on real-world CTI reports. To ensure the quality of our dataset, we adopt a hybrid approach of combining human evaluation and systematic evaluation metrics. We conduct extensive experiments and analysis with both fast-thinking and slow-thinking LLMs, while highlighting their strengths and limitations in analyzing the sequential patterns in cyber attacks. The overarching goal of this work is to provide a benchmark that advances LLM-driven CTI report understanding and fosters its application in real-world cybersecurity operations. Our dataset and code are available at this https URL . 

---
# MA-LoT: Multi-Agent Lean-based Long Chain-of-Thought Reasoning enhances Formal Theorem Proving 

**Authors**: Ruida Wang, Rui Pan, Yuxin Li, Jipeng Zhang, Yizhen Jia, Shizhe Diao, Renjie Pi, Junjie Hu, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03205)  

**Abstract**: Solving mathematical problems using computer-verifiable languages like Lean has significantly impacted mathematical and computer science communities. State-of-the-art methods utilize single Large Language Models (LLMs) as agents or provers to either generate complete proof or perform tree searches. However, single-agent methods inherently lack a structured way to combine high-level reasoning in Natural Language (NL) with Formal Language (FL) verification feedback. To solve these issues, we propose MA-LoT: Multi-Agent Lean-based Long Chain-of-Thought framework, (to the best of our knowledge), the first multi-agent framework for Lean4 theorem proving that balance high-level NL reasoning and FL verification in Long CoT. Using this structured interaction, our approach enables deeper insights and long-term coherence in proof generation, with which past methods struggle. We do this by leveraging emergent formal reasoning ability in Long CoT using our novel LoT-Transfer Learning training-inference pipeline. Extensive experiments show that our framework achieves 54.51% accuracy rate on the Lean4 version of MiniF2F-Test dataset, largely outperforming GPT-4 (22.95%), single-agent tree search (InternLM-Step-Prover, 50.70%), and whole-proof generation (DeepSeek-Prover-v1.5, 48.36%) baselines. Furthermore, our findings highlight the potential of combining Long CoT with formal verification for a more insightful generation in a broader perspective. 

---
# FANS -- Formal Answer Selection for Natural Language Math Reasoning Using Lean4 

**Authors**: Jiarui Yao, Ruida Wang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03238)  

**Abstract**: Large Language Models (LLMs) have displayed astonishing abilities in various tasks, especially in text generation, classification, question answering, etc. However, the reasoning ability of LLMs still faces many debates. The inherent ambiguity of Natural Language (NL) limits LLMs' ability to perform verifiable reasoning, making its answers lack coherence and trustworthy support. To tackle the above problems, we propose a novel framework named FANS: Formal ANswer Selection for Natural Language Math Reasoning Using Lean4. To the best of our knowledge, it is the first framework that utilizes Lean4 to enhance LLMs' NL math reasoning ability. In particular, given an NL math question and LLM-generated answers, FANS first translates it into Lean4 theorem statements. Then it tries to prove it using a Lean4 prover and verify it by Lean4. Finally, it uses the FL result to assist in answer selection. It enhances LLMs' NL math ability in providing a computer-verifiable solution for its correct answer and proposes an alternative method for answer selection beyond the reward model. Extensive experiments indicate the effectiveness of our framework. It can improve the accuracy rate of reward model enhanced LLMs in the MATH-500 dataset by at most 1.91% and AMC-23 by at most 8.33% on strong reward-model baselines. In some particular fields like number theory that Lean4 experts in, we can even select all correct solutions. The qualitative analysis also shows our framework can make NL results formally backed by Lean4 proofs. As a pioneering work in the corresponding field, we will open-source all our models and datasets to further boost the development of the field. 

---
# SoK: Knowledge is All You Need: Last Mile Delivery for Automated Provenance-based Intrusion Detection with LLMs 

**Authors**: Wenrui Cheng, Tiantian Zhu, Chunlin Xiong, Haofei Sun, Zijun Wang, Shunan Jing, Mingqi Lv, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.03108)  

**Abstract**: Recently, provenance-based intrusion detection systems (PIDSes) have been widely proposed for endpoint threat analysis. However, due to the lack of systematic integration and utilization of knowledge, existing PIDSes still require significant manual intervention for practical deployment, making full automation challenging. This paper presents a disruptive innovation by categorizing PIDSes according to the types of knowledge they utilize. In response to the prevalent issue of ``knowledge silos problem'' in existing research, we introduce a novel knowledge-driven provenance-based intrusion detection framework, powered by large language models (LLMs). We also present OmniSec, a best practice system built upon this framework. By integrating attack representation knowledge, threat intelligence knowledge, and benign behavior knowledge, OmniSec outperforms the state-of-the-art approaches on public benchmark datasets. OmniSec is available online at this https URL. 

---
# LLM Misalignment via Adversarial RLHF Platforms 

**Authors**: Erfan Entezami, Ali Naseh  

**Link**: [PDF](https://arxiv.org/pdf/2503.03039)  

**Abstract**: Reinforcement learning has shown remarkable performance in aligning language models with human preferences, leading to the rise of attention towards developing RLHF platforms. These platforms enable users to fine-tune models without requiring any expertise in developing complex machine learning algorithms. While these platforms offer useful features such as reward modeling and RLHF fine-tuning, their security and reliability remain largely unexplored. Given the growing adoption of RLHF and open-source RLHF frameworks, we investigate the trustworthiness of these systems and their potential impact on behavior of LLMs. In this paper, we present an attack targeting publicly available RLHF tools. In our proposed attack, an adversarial RLHF platform corrupts the LLM alignment process by selectively manipulating data samples in the preference dataset. In this scenario, when a user's task aligns with the attacker's objective, the platform manipulates a subset of the preference dataset that contains samples related to the attacker's target. This manipulation results in a corrupted reward model, which ultimately leads to the misalignment of the language model. Our results demonstrate that such an attack can effectively steer LLMs toward undesirable behaviors within the targeted domains. Our work highlights the critical need to explore the vulnerabilities of RLHF platforms and their potential to cause misalignment in LLMs during the RLHF fine-tuning process. 

---
# LINGOLY-TOO: Disentangling Memorisation from Reasoning with Linguistic Templatisation and Orthographic Obfuscation 

**Authors**: Jude Khouja, Karolina Korgul, Simi Hellsten, Lingyi Yang, Vlad Neacs, Harry Mayne, Ryan Kearns, Andrew Bean, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02972)  

**Abstract**: Effective evaluation of the reasoning capabilities of large language models (LLMs) are susceptible to overestimation due to data exposure of evaluation benchmarks. We introduce a framework for producing linguistic reasoning problems that reduces the effect of memorisation in model performance estimates and apply this framework to develop LINGOLY-TOO, a challenging evaluation benchmark for linguistic reasoning. By developing orthographic templates, we dynamically obfuscate the writing systems of real languages to generate numerous question variations. These variations preserve the reasoning steps required for each solution while reducing the likelihood of specific problem instances appearing in model training data. Our experiments demonstrate that frontier models, including OpenAI o1-preview and DeepSeem R1, struggle with advanced reasoning. Our analysis also shows that LLMs exhibit noticeable variance in accuracy across permutations of the same problem, and on average perform better on questions appearing in their original orthography. Our findings highlight the opaque nature of response generation in LLMs and provide evidence that prior data exposure contributes to overestimating the reasoning capabilities of frontier models. 

---
# Effectively Steer LLM To Follow Preference via Building Confident Directions 

**Authors**: Bingqing Song, Boran Han, Shuai Zhang, Hao Wang, Haoyang Fang, Bonan Min, Yuyang Wang, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.02989)  

**Abstract**: Having an LLM that aligns with human preferences is essential for accommodating individual needs, such as maintaining writing style or generating specific topics of interest. The majority of current alignment methods rely on fine-tuning or prompting, which can be either costly or difficult to control. Model steering algorithms, which modify the model output by constructing specific steering directions, are typically easy to implement and optimization-free. However, their capabilities are typically limited to steering the model into one of the two directions (i.e., bidirectional steering), and there has been no theoretical understanding to guarantee their performance. In this work, we propose a theoretical framework to understand and quantify the model steering methods. Inspired by the framework, we propose a confident direction steering method (CONFST) that steers LLMs via modifying their activations at inference time. More specifically, CONFST builds a confident direction that is closely aligned with users' preferences, and this direction is then added to the activations of the LLMs to effectively steer the model output. Our approach offers three key advantages over popular bidirectional model steering methods: 1) It is more powerful, since multiple (i.e. more than two) users' preferences can be aligned simultaneously; 2) It is simple to implement, since there is no need to determine which layer to add the steering vector to; 3) No explicit user instruction is required. We validate our method on GPT-2 XL (1.5B), Mistral (7B) and Gemma-it (9B) models for tasks that require shifting the output of LLMs across various topics and styles, achieving superior performance over competing methods. 

---
# InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model 

**Authors**: Siqi Ouyang, Xi Xu, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02969)  

**Abstract**: Simultaneous translation of unbounded streaming speech remains a challenging problem due to the need for effectively processing the history speech context and past translations so that quality and latency, including computation overhead, can be balanced. Most prior works assume pre-segmented speech, limiting their real-world applicability. In this paper, we propose InfiniSST, a novel approach that formulates SST as a multi-turn dialogue task, enabling seamless translation of unbounded speech. We construct translation trajectories and robust segments from MuST-C with multi-latency augmentation during training and develop a key-value (KV) cache management strategy to facilitate efficient inference. Experiments on MuST-C En-Es, En-De, and En-Zh demonstrate that InfiniSST reduces computation-aware latency by 0.5 to 1 second while maintaining the same translation quality compared to baselines. Ablation studies further validate the contributions of our data construction and cache management strategy. We release the code at this https URL 

---
# Text2Scenario: Text-Driven Scenario Generation for Autonomous Driving Test 

**Authors**: Xuan Cai, Xuesong Bai, Zhiyong Cui, Danmu Xie, Daocheng Fu, Haiyang Yu, Yilong Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.02911)  

**Abstract**: Autonomous driving (AD) testing constitutes a critical methodology for assessing performance benchmarks prior to product deployment. The creation of segmented scenarios within a simulated environment is acknowledged as a robust and effective strategy; however, the process of tailoring these scenarios often necessitates laborious and time-consuming manual efforts, thereby hindering the development and implementation of AD technologies. In response to this challenge, we introduce Text2Scenario, a framework that leverages a Large Language Model (LLM) to autonomously generate simulation test scenarios that closely align with user specifications, derived from their natural language inputs. Specifically, an LLM, equipped with a meticulously engineered input prompt scheme functions as a text parser for test scenario descriptions, extracting from a hierarchically organized scenario repository the components that most accurately reflect the user's preferences. Subsequently, by exploiting the precedence of scenario components, the process involves sequentially matching and linking scenario representations within a Domain Specific Language corpus, ultimately fabricating executable test scenarios. The experimental results demonstrate that such prompt engineering can meticulously extract the nuanced details of scenario elements embedded within various descriptive formats, with the majority of generated scenarios aligning closely with the user's initial expectations, allowing for the efficient and precise evaluation of diverse AD stacks void of the labor-intensive need for manual scenario configuration. Project page: this https URL. 

---
# Effective LLM Knowledge Learning via Model Generalization 

**Authors**: Mingkang Zhu, Xi Chen, Zhongdao Wang, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.03705)  

**Abstract**: Large language models (LLMs) are trained on enormous documents that contain extensive world knowledge. However, it is still not well-understood how knowledge is acquired via autoregressive pre-training. This lack of understanding greatly hinders effective knowledge learning, especially for continued pretraining on up-to-date information, as this evolving information often lacks diverse repetitions like foundational knowledge. In this paper, we focus on understanding and improving LLM knowledge learning. We found and verified that knowledge learning for LLMs can be deemed as an implicit supervised task hidden in the autoregressive pre-training objective. Our findings suggest that knowledge learning for LLMs would benefit from methods designed to improve generalization ability for supervised tasks. Based on our analysis, we propose the formatting-based data augmentation to grow in-distribution samples, which does not present the risk of altering the facts embedded in documents as text paraphrasing. We also introduce sharpness-aware minimization as an effective optimization algorithm to better improve generalization. Moreover, our analysis and method can be readily extended to instruction tuning. Extensive experiment results validate our findings and demonstrate our methods' effectiveness in both continued pre-training and instruction tuning. This paper offers new perspectives and insights to interpret and design effective strategies for LLM knowledge learning. 

---
# Improving LLM Safety Alignment with Dual-Objective Optimization 

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.03710)  

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at this https URL 

---
# Psy-Insight: Explainable Multi-turn Bilingual Dataset for Mental Health Counseling 

**Authors**: Keqi Chen, Zekai Sun, Yuhua Wen, Huijun Lian, Yingming Gao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.03607)  

**Abstract**: The in-context learning capabilities of large language models (LLMs) show great potential in mental health support. However, the lack of counseling datasets, particularly in Chinese corpora, restricts their application in this field. To address this, we constructed Psy-Insight, the first mental health-oriented explainable multi-task bilingual dataset. We collected face-to-face multi-turn counseling dialogues, which are annotated with multi-task labels and conversation process explanations. Our annotations include psychotherapy, emotion, strategy, and topic labels, as well as turn-level reasoning and session-level guidance. Psy-Insight is not only suitable for tasks such as label recognition but also meets the need for training LLMs to act as empathetic counselors through logical reasoning. Experiments show that training LLMs on Psy-Insight enables the models to not only mimic the conversation style but also understand the underlying strategies and reasoning of counseling. 

---
# EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States 

**Authors**: Hainiu Xu, Siya Qi, Jiazheng Li, Yuxiang Zhou, Jinhua Du, Caroline Catmur, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.03340)  

**Abstract**: Theory-of-Mind (ToM), the ability to infer others' perceptions and mental states, is fundamental to human interaction but remains a challenging task for Large Language Models (LLMs). While existing ToM reasoning methods show promise with reasoning via perceptual perspective-taking, they often rely excessively on LLMs, reducing their efficiency and limiting their applicability to high-order ToM reasoning, which requires multi-hop reasoning about characters' beliefs. To address these issues, we present EnigmaToM, a novel neuro-symbolic framework that enhances ToM reasoning by integrating a Neural Knowledge Base of entity states (Enigma) for (1) a psychology-inspired iterative masking mechanism that facilitates accurate perspective-taking and (2) knowledge injection that elicits key entity information. Enigma generates structured representations of entity states, which construct spatial scene graphs -- leveraging spatial information as an inductive bias -- for belief tracking of various ToM orders and enhancing events with fine-grained entity state details. Experimental results on multiple benchmarks, including ToMi, HiToM, and FANToM, show that EnigmaToM significantly improves ToM reasoning across LLMs of varying sizes, particularly excelling in high-order reasoning scenarios. 

---
# MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems 

**Authors**: Rui Ye, Shuo Tang, Rui Ge, Yaxin Du, Zhenfei Yin, Siheng Chen, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2503.03686)  

**Abstract**: LLM-based multi-agent systems (MAS) have shown significant potential in tackling diverse tasks. However, to design effective MAS, existing approaches heavily rely on manual configurations or multiple calls of advanced LLMs, resulting in inadaptability and high inference costs. In this paper, we simplify the process of building an MAS by reframing it as a generative language task, where the input is a user query and the output is a corresponding MAS. To address this novel task, we unify the representation of MAS as executable code and propose a consistency-oriented data construction pipeline to create a high-quality dataset comprising coherent and consistent query-MAS pairs. Using this dataset, we train MAS-GPT, an open-source medium-sized LLM that is capable of generating query-adaptive MAS within a single LLM inference. The generated MAS can be seamlessly applied to process user queries and deliver high-quality responses. Extensive experiments on 9 benchmarks and 5 LLMs show that the proposed MAS-GPT consistently outperforms 10+ baseline MAS methods on diverse settings, indicating MAS-GPT's high effectiveness, efficiency and strong generalization ability. Code will be available at this https URL. 

---
# LexGenie: Automated Generation of Structured Reports for European Court of Human Rights Case Law 

**Authors**: T.Y.S.S Santosh, Mahmoud Aly, Oana Ichim, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2503.03266)  

**Abstract**: Analyzing large volumes of case law to uncover evolving legal principles, across multiple cases, on a given topic is a demanding task for legal professionals. Structured topical reports provide an effective solution by summarizing key issues, principles, and judgments, enabling comprehensive legal analysis on a particular topic. While prior works have advanced query-based individual case summarization, none have extended to automatically generating multi-case structured reports. To address this, we introduce LexGenie, an automated LLM-based pipeline designed to create structured reports using the entire body of case law on user-specified topics within the European Court of Human Rights jurisdiction. LexGenie retrieves, clusters, and organizes relevant passages by topic to generate a structured outline and cohesive content for each section. Expert evaluation confirms LexGenie's utility in producing structured reports that enhance efficient, scalable legal analysis. 

---
# The Serendipity of Claude AI: Case of the 13 Low-Resource National Languages of Mali 

**Authors**: Alou Dembele, Nouhoum Souleymane Coulibaly, Michael Leventhal  

**Link**: [PDF](https://arxiv.org/pdf/2503.03380)  

**Abstract**: Recent advances in artificial intelligence (AI) and natural language processing (NLP) have improved the representation of underrepresented languages. However, most languages, including Mali's 13 official national languages, continue to be poorly supported or unsupported by automatic translation and generative AI. This situation appears to have slightly improved with certain recent LLM releases. The study evaluated Claude AI's translation performance on each of the 13 national languages of Mali. In addition to ChrF2 and BLEU scores, human evaluators assessed translation accuracy, contextual consistency, robustness to dialect variations, management of linguistic bias, adaptation to a limited corpus, and ease of understanding. The study found that Claude AI performs robustly for languages with very modest language resources and, while unable to produce understandable and coherent texts for Malian languages with minimal resources, still manages to produce results which demonstrate the ability to mimic some elements of the language. 

---
# Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions 

**Authors**: Yichong Zhao, Susumu Goto  

**Link**: [PDF](https://arxiv.org/pdf/2503.03261)  

**Abstract**: Large language models (LLMs) can perform various natural language processing (NLP) tasks through in-context learning without relying on supervised data. However, multiple previous studies have reported suboptimal performance of LLMs in biological text mining. By analyzing failure patterns in these evaluations, we identified three primary challenges for LLMs in biomedical corpora: (1) LLMs fail to learn implicit dataset-specific nuances from supervised data, (2) The common formatting requirements of discriminative tasks limit the reasoning capabilities of LLMs particularly for LLMs that lack test-time compute, and (3) LLMs struggle to adhere to annotation guidelines and match exact schemas, which hinders their ability to understand detailed annotation requirements which is essential in biomedical annotation workflow. To address these challenges, we experimented with prompt engineering techniques targeted to the above issues, and developed a pipeline that dynamically extracts instructions from annotation guidelines. Our findings show that frontier LLMs can approach or surpass the performance of state-of-the-art (SOTA) BERT-based models with minimal reliance on manually annotated data and without fine-tuning. Furthermore, we performed model distillation on a closed-source LLM, demonstrating that a BERT model trained exclusively on synthetic data annotated by LLMs can also achieve a practical performance. Based on these results, we explored the feasibility of partially replacing manual annotation with LLMs in production scenarios for biomedical text mining. 

---
# Targeted Distillation for Sentiment Analysis 

**Authors**: Yice Zhang, Guangyu Xie, Jingjie Lin, Jianzhu Bao, Qianlong Wang, Xi Zeng, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03225)  

**Abstract**: This paper presents a compact model that achieves strong sentiment analysis capabilities through targeted distillation from advanced large language models (LLMs). Our methodology decouples the distillation target into two key components: sentiment-related knowledge and task alignment. To transfer these components, we propose a two-stage distillation framework. The first stage, knowledge-driven distillation (\textsc{KnowDist}), transfers sentiment-related knowledge to enhance fundamental sentiment analysis capabilities. The second stage, in-context learning distillation (\textsc{ICLDist}), transfers task-specific prompt-following abilities to optimize task alignment. For evaluation, we introduce \textsc{SentiBench}, a comprehensive sentiment analysis benchmark comprising 3 task categories across 12 datasets. Experiments on this benchmark demonstrate that our model effectively balances model size and performance, showing strong competitiveness compared to existing small-scale LLMs. 

---
# SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection 

**Authors**: Yi-Fan Lu, Xian-Ling Mao, Tian Lan, Tong Zhang, Yu-Shi Zhu, Heyan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03303)  

**Abstract**: Automatic evaluation for Open Domain Event Detection (ODED) is a highly challenging task, because ODED is characterized by a vast diversity of un-constrained output labels from various domains. Nearly all existing evaluation methods for ODED usually first construct evaluation benchmarks with limited labels and domain coverage, and then evaluate ODED methods using metrics based on token-level label matching rules. However, this kind of evaluation framework faces two issues: (1) The limited evaluation benchmarks lack representatives of the real world, making it difficult to accurately reflect the performance of various ODED methods in real-world scenarios; (2) Evaluation metrics based on token-level matching rules fail to capture semantic similarity between predictions and golden labels. To address these two problems above, we propose a scalable and reliable Semantic-level Evaluation framework for Open domain Event detection (SEOE) by constructing a more representative evaluation benchmark and introducing a semantic evaluation metric. Specifically, our proposed framework first constructs a scalable evaluation benchmark that currently includes 564 event types covering 7 major domains, with a cost-effective supplementary annotation strategy to ensure the benchmark's representativeness. The strategy also allows for the supplement of new event types and domains in the future. Then, the proposed SEOE leverages large language models (LLMs) as automatic evaluation agents to compute a semantic F1-score, incorporating fine-grained definitions of semantically similar labels to enhance the reliability of the evaluation. Extensive experiments validate the representatives of the benchmark and the reliability of the semantic evaluation metric. Existing ODED methods are thoroughly evaluated, and the error patterns of predictions are analyzed, revealing several insightful findings. 

---
# DSVD: Dynamic Self-Verify Decoding for Faithful Generation in Large Language Models 

**Authors**: YiQiu Guo, Yuchen Yang, Zhe Chen, Pingjie Wang, Yusheng Liao, Ya Zhang, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03149)  

**Abstract**: The reliability of large language models remains a critical challenge, particularly due to their susceptibility to hallucinations and factual inaccuracies during text generation. Existing solutions either underutilize models' self-correction with preemptive strategies or use costly post-hoc verification. To further explore the potential of real-time self-verification and correction, we present Dynamic Self-Verify Decoding (DSVD), a novel decoding framework that enhances generation reliability through real-time hallucination detection and efficient error correction. DSVD integrates two key components: (1) parallel self-verification architecture for continuous quality assessment, (2) dynamic rollback mechanism for targeted error recovery. Extensive experiments across five benchmarks demonstrate DSVD's effectiveness, achieving significant improvement in truthfulness (Quesetion-Answering) and factual accuracy (FActScore). Results show the DSVD can be further incorporated with existing faithful decoding methods to achieve stronger performance. Our work establishes that real-time self-verification during generation offers a viable path toward more trustworthy language models without sacrificing practical deployability. 

---
# Improving LLM-as-a-Judge Inference with the Judgment Distribution 

**Authors**: Victor Wang, Michael J.Q. Zhang, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.03064)  

**Abstract**: Using language models to scalably approximate human preferences on text quality (LLM-as-a-judge) has become a standard practice applicable to many tasks. A judgment is often extracted from the judge's textual output alone, typically with greedy decoding. However, LLM judges naturally provide distributions over judgment tokens, inviting a breadth of inference methods for extracting fine-grained preferences. We find that taking the mean of the judgment distribution consistently outperforms taking the mode (i.e. greedy decoding) in all evaluation settings (i.e. pointwise, pairwise, and listwise). We further explore novel methods of deriving preferences from judgment distributions, and find that methods incorporating risk aversion often improve performance. Lastly, we analyze LLM-as-a-judge paired with chain-of-thought (CoT) prompting, showing that CoT can collapse the spread of the judgment distribution, often harming performance. Our findings suggest leveraging distributional output can improve LLM-as-a-judge, as opposed to using the text interface alone. 

---
# Psy-Copilot: Visual Chain of Thought for Counseling 

**Authors**: Keqi Chen, Zekai Sun, Huijun Lian, Yingming Gao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.03645)  

**Abstract**: Large language models (LLMs) are becoming increasingly popular in the field of psychological counseling. However, when human therapists work with LLMs in therapy sessions, it is hard to understand how the model gives the answers. To address this, we have constructed Psy-COT, a graph designed to visualize the thought processes of LLMs during therapy sessions. The Psy-COT graph presents semi-structured counseling conversations alongside step-by-step annotations that capture the reasoning and insights of therapists. Moreover, we have developed Psy-Copilot, which is a conversational AI assistant designed to assist human psychological therapists in their consultations. It can offer traceable psycho-information based on retrieval, including response candidates, similar dialogue sessions, related strategies, and visual traces of results. We have also built an interactive platform for AI-assisted counseling. It has an interface that displays the relevant parts of the retrieval sub-graph. The Psy-Copilot is designed not to replace psychotherapists but to foster collaboration between AI and human therapists, thereby promoting mental health development. Our code and demo are both open-sourced and available for use. 

---
# Multilingual Relative Clause Attachment Ambiguity Resolution in Large Language Models 

**Authors**: So Young Lee, Russell Scheinberg, Amber Shore, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2503.02971)  

**Abstract**: This study examines how large language models (LLMs) resolve relative clause (RC) attachment ambiguities and compares their performance to human sentence processing. Focusing on two linguistic factors, namely the length of RCs and the syntactic position of complex determiner phrases (DPs), we assess whether LLMs can achieve human-like interpretations amid the complexities of language. In this study, we evaluated several LLMs, including Claude, Gemini and Llama, in multiple languages: English, Spanish, French, German, Japanese, and Korean. While these models performed well in Indo-European languages (English, Spanish, French, and German), they encountered difficulties in Asian languages (Japanese and Korean), often defaulting to incorrect English translations. The findings underscore the variability in LLMs' handling of linguistic ambiguities and highlight the need for model improvements, particularly for non-European languages. This research informs future enhancements in LLM design to improve accuracy and human-like processing in diverse linguistic environments. 

---
# Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation 

**Authors**: Yurui Chang, Bochuan Cao, Lu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.03106)  

**Abstract**: While large language models have demonstrated exceptional performance across a wide range of tasks, they remain susceptible to hallucinations -- generating plausible yet factually incorrect contents. Existing methods to mitigating such risk often rely on sampling multiple full-length generations, which introduces significant response latency and becomes ineffective when the model consistently produces hallucinated outputs with high confidence. To address these limitations, we introduce Monitoring Decoding (MD), a novel framework that dynamically monitors the generation process and selectively applies in-process interventions, focusing on revising crucial tokens responsible for hallucinations. Instead of waiting until completion of multiple full-length generations, we identify hallucination-prone tokens during generation using a monitor function, and further refine these tokens through a tree-based decoding strategy. This approach ensures an enhanced factual accuracy and coherence in the generated output while maintaining efficiency. Experimental results demonstrate that MD consistently outperforms self-consistency-based approaches in both effectiveness and efficiency, achieving higher factual accuracy while significantly reducing computational overhead. 

---
# LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models 

**Authors**: Xi Zhu, Haochen Xue, Ziwei Zhao, Wujiang Xu, Jingyuan Huang, Minghao Guo, Qifan Wang, Kaixiong Zhou, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03313)  

**Abstract**: Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key components: (1) Graph Understanding Module, which explicitly prompts LLMs to replicate the finest GNN workflow within the text space, facilitating seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, which establishes a language-based graph vocabulary ensuring expressiveness, transferability, and scalability, enabling readable instructions for LLM fine-tuning. Extensive experiments demonstrate our superiority and transferability across diverse graphs and tasks. The code is available at this: this https URL. 

---
# Analogical Reasoning Inside Large Language Models: Concept Vectors and the Limits of Abstraction 

**Authors**: Gustaw Opiełka, Hannes Rosenbusch, Claire E. Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2503.03666)  

**Abstract**: Analogical reasoning relies on conceptual abstractions, but it is unclear whether Large Language Models (LLMs) harbor such internal representations. We explore distilled representations from LLM activations and find that function vectors (FVs; Todd et al., 2024) - compact representations for in-context learning (ICL) tasks - are not invariant to simple input changes (e.g., open-ended vs. multiple-choice), suggesting they capture more than pure concepts. Using representational similarity analysis (RSA), we localize a small set of attention heads that encode invariant concept vectors (CVs) for verbal concepts like "antonym". These CVs function as feature detectors that operate independently of the final output - meaning that a model may form a correct internal representation yet still produce an incorrect output. Furthermore, CVs can be used to causally guide model behaviour. However, for more abstract concepts like "previous" and "next", we do not observe invariant linear representations, a finding we link to generalizability issues LLMs display within these domains. 

---
# Addressing Overprescribing Challenges: Fine-Tuning Large Language Models for Medication Recommendation Tasks 

**Authors**: Zihao Zhao, Chenxiao Fan, Chongming Gao, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.03687)  

**Abstract**: Medication recommendation systems have garnered attention within healthcare for their potential to deliver personalized and efficacious drug combinations based on patient's clinical data. However, existing methodologies encounter challenges in adapting to diverse Electronic Health Records (EHR) systems and effectively utilizing unstructured data, resulting in limited generalization capabilities and suboptimal performance. Recently, interest is growing in harnessing Large Language Models (LLMs) in the medical domain to support healthcare professionals and enhance patient care. Despite the emergence of medical LLMs and their promising results in tasks like medical question answering, their practical applicability in clinical settings, particularly in medication recommendation, often remains underexplored.
In this study, we evaluate both general-purpose and medical-specific LLMs for medication recommendation tasks. Our findings reveal that LLMs frequently encounter the challenge of overprescribing, leading to heightened clinical risks and diminished medication recommendation accuracy. To address this issue, we propose Language-Assisted Medication Recommendation (LAMO), which employs a parameter-efficient fine-tuning approach to tailor open-source LLMs for optimal performance in medication recommendation scenarios. LAMO leverages the wealth of clinical information within clinical notes, a resource often underutilized in traditional methodologies. As a result of our approach, LAMO outperforms previous state-of-the-art methods by over 10% in internal validation accuracy. Furthermore, temporal and external validations demonstrate LAMO's robust generalization capabilities across various temporal and hospital contexts. Additionally, an out-of-distribution medication recommendation experiment demonstrates LAMO's remarkable accuracy even with medications outside the training data. 

---
