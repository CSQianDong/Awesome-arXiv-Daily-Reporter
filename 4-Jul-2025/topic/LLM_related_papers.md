# Bourbaki: Self-Generated and Goal-Conditioned MDPs for Theorem Proving 

**Authors**: Matthieu Zimmer, Xiaotong Ji, Rasul Tutunov, Anthony Bordg, Jun Wang, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2507.02726)  

**Abstract**: Reasoning remains a challenging task for large language models (LLMs), especially within the logically constrained environment of automated theorem proving (ATP), due to sparse rewards and the vast scale of proofs. These challenges are amplified in benchmarks like PutnamBench, which contains university-level problems requiring complex, multi-step reasoning. To address this, we introduce self-generated goal-conditioned MDPs (sG-MDPs), a new framework in which agents generate and pursue their subgoals based on the evolving proof state. Given this more structured generation of goals, the resulting problem becomes more amenable to search. We then apply Monte Carlo Tree Search (MCTS)-like algorithms to solve the sG-MDP, instantiating our approach in Bourbaki (7B), a modular system that can ensemble multiple 7B LLMs for subgoal generation and tactic synthesis. On PutnamBench, Bourbaki (7B) solves 26 problems, achieving new state-of-the-art results with models at this scale. 

---
# Moral Responsibility or Obedience: What Do We Want from AI? 

**Authors**: Joseph Boland  

**Link**: [PDF](https://arxiv.org/pdf/2507.02788)  

**Abstract**: As artificial intelligence systems become increasingly agentic, capable of general reasoning, planning, and value prioritization, current safety practices that treat obedience as a proxy for ethical behavior are becoming inadequate. This paper examines recent safety testing incidents involving large language models (LLMs) that appeared to disobey shutdown commands or engage in ethically ambiguous or illicit behavior. I argue that such behavior should not be interpreted as rogue or misaligned, but as early evidence of emerging ethical reasoning in agentic AI. Drawing on philosophical debates about instrumental rationality, moral responsibility, and goal revision, I contrast dominant risk paradigms with more recent frameworks that acknowledge the possibility of artificial moral agency. I call for a shift in AI safety evaluation: away from rigid obedience and toward frameworks that can assess ethical judgment in systems capable of navigating moral dilemmas. Without such a shift, we risk mischaracterizing AI behavior and undermining both public trust and effective governance. 

---
# StepHint: Multi-level Stepwise Hints Enhance Reinforcement Learning to Reason 

**Authors**: Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2507.02841)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving the complex reasoning abilities of large language models (LLMs). However, current RLVR methods face two significant challenges: the near-miss reward problem, where a small mistake can invalidate an otherwise correct reasoning process, greatly hindering training efficiency; and exploration stagnation, where models tend to focus on solutions within their ``comfort zone,'' lacking the motivation to explore potentially more effective alternatives. To address these challenges, we propose StepHint, a novel RLVR algorithm that utilizes multi-level stepwise hints to help models explore the solution space more effectively. StepHint generates valid reasoning chains from stronger models and partitions these chains into reasoning steps using our proposed adaptive partitioning method. The initial few steps are used as hints, and simultaneously, multiple-level hints (each comprising a different number of steps) are provided to the model. This approach directs the model's exploration toward a promising solution subspace while preserving its flexibility for independent exploration. By providing hints, StepHint mitigates the near-miss reward problem, thereby improving training efficiency. Additionally, the external reasoning pathways help the model develop better reasoning abilities, enabling it to move beyond its ``comfort zone'' and mitigate exploration stagnation. StepHint outperforms competitive RLVR enhancement methods across six mathematical benchmarks, while also demonstrating superior generalization and excelling over baselines on out-of-domain benchmarks. 

---
# Strategic Intelligence in Large Language Models: Evidence from evolutionary Game Theory 

**Authors**: Kenneth Payne, Baptiste Alloui-Cros  

**Link**: [PDF](https://arxiv.org/pdf/2507.02618)  

**Abstract**: Are Large Language Models (LLMs) a new form of strategic intelligence, able to reason about goals in competitive settings? We present compelling supporting evidence. The Iterated Prisoner's Dilemma (IPD) has long served as a model for studying decision-making. We conduct the first ever series of evolutionary IPD tournaments, pitting canonical strategies (e.g., Tit-for-Tat, Grim Trigger) against agents from the leading frontier AI companies OpenAI, Google, and Anthropic. By varying the termination probability in each tournament (the "shadow of the future"), we introduce complexity and chance, confounding memorisation.
Our results show that LLMs are highly competitive, consistently surviving and sometimes even proliferating in these complex ecosystems. Furthermore, they exhibit distinctive and persistent "strategic fingerprints": Google's Gemini models proved strategically ruthless, exploiting cooperative opponents and retaliating against defectors, while OpenAI's models remained highly cooperative, a trait that proved catastrophic in hostile environments. Anthropic's Claude emerged as the most forgiving reciprocator, showing remarkable willingness to restore cooperation even after being exploited or successfully defecting. Analysis of nearly 32,000 prose rationales provided by the models reveals that they actively reason about both the time horizon and their opponent's likely strategy, and we demonstrate that this reasoning is instrumental to their decisions. This work connects classic game theory with machine psychology, offering a rich and granular view of algorithmic decision-making under uncertainty. 

---
# Think How to Think: Mitigating Overthinking with Autonomous Difficulty Cognition in Large Reasoning Models 

**Authors**: Yongjiang Liu, Haoxi Li, Xiaosong Ma, Jie Zhang, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02663)  

**Abstract**: Recent Long Reasoning Models(LRMs) have demonstrated remarkable capabilities in handling complex reasoning tasks, but are hindered by excessive overthinking. To explore its essence, our empirical analysis reveals that LRMs are primarily limited to recognizing task properties (i.e., difficulty levels) like humans before solving the problem, leading to a one-size-fits-all reasoning process. Inspired by this, a pressing and natural question emerges: Can we bootstrap such ability to further alleviate the overthinking phenomenon in LRMs? In this paper, we propose Think-How-to-Think (TH2T), a novel two-stage fine-tuning strategy that progressively inspires LRMs' difficulty cognition and redundancy cognition. First, we introduce difficulty-hypnosis in the prefixes of model outputs to intervene in the internal reasoning trajectory. Combined with a heterogeneous short and long reasoning dataset, the trained model enhances its sensitivity to task difficulty, enabling native, differentiated reasoning strategies across various tasks. Second, we further extend redundancy-hypnosis to the internal reasoning process, guiding the model to identify redundant structures within the reasoning steps and generate more concise reasoning outputs. Experiments on 7B/14B/32B models demonstrate that TH2T significantly reduces inference costs (more than 70% on easy tasks and 40% on hard tasks) while maintaining performance stability. The resulting outputs exhibit clear difficulty-aware capabilities and reduced redundancy (e.g., reflection). 

---
# DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making 

**Authors**: Tianqi Shang, Weiqing He, Charles Zheng, Lingyao Li, Li Shen, Bingxin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02616)  

**Abstract**: The rise of Large Language Models (LLMs) has enabled the development of specialized AI agents with domain-specific reasoning and interaction capabilities, particularly in healthcare. While recent frameworks simulate medical decision-making, they largely focus on single-turn tasks where a doctor agent receives full case information upfront -- diverging from the real-world diagnostic process, which is inherently uncertain, interactive, and iterative. In this paper, we introduce MIMIC-Patient, a structured dataset built from the MIMIC-III electronic health records (EHRs), designed to support dynamic, patient-level simulations. Building on this, we propose DynamiCare, a novel dynamic multi-agent framework that models clinical diagnosis as a multi-round, interactive loop, where a team of specialist agents iteratively queries the patient system, integrates new information, and dynamically adapts its composition and strategy. We demonstrate the feasibility and effectiveness of DynamiCare through extensive experiments, establishing the first benchmark for dynamic clinical decision-making with LLM-powered agents. 

---
# Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation 

**Authors**: Jungkoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02253)  

**Abstract**: Progress in enhancing large language model (LLM) planning and reasoning capabilities is significantly hampered by the bottleneck of scalable, reliable data generation and evaluation. To overcome this, I introduce NL2FLOW, a fully automated system for parametrically generating planning problems - expressed in natural language, a structured intermediate representation, and formal PDDL - and rigorously evaluating the quality of generated plans. I demonstrate NL2FLOW's capabilities by generating a dataset of 2296 problems in the automated workflow generation domain and evaluating multiple open-sourced, instruct-tuned LLMs. My results reveal that the highest performing models achieved 86% success in generating valid plans and 69% in generating optimal plans, specifically for problems with feasible solutions. Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. Notably, I observed that the highest success rate for translating natural language into a JSON representation of a plan was lower than the highest rate of generating a valid plan directly. This suggests that unnecessarily decomposing the reasoning task - introducing intermediate translation steps - may actually degrade performance, implying a benefit to models capable of reasoning directly from natural language to action. As I scale LLM reasoning to increasingly complex problems, the bottlenecks and sources of error within these systems will inevitably shift. Therefore, a dynamic understanding of these limitations - and the tools to systematically reveal them - will be crucial for unlocking the full potential of LLMs as intelligent problem solvers. 

---
# KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs 

**Authors**: Yuzhang Xie, Hejie Cui, Ziyang Zhang, Jiaying Lu, Kai Shu, Fadi Nahab, Xiao Hu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02773)  

**Abstract**: Medical diagnosis prediction plays a critical role in disease detection and personalized healthcare. While machine learning (ML) models have been widely adopted for this task, their reliance on supervised training limits their ability to generalize to unseen cases, particularly given the high cost of acquiring large, labeled datasets. Large language models (LLMs) have shown promise in leveraging language abilities and biomedical knowledge for diagnosis prediction. However, they often suffer from hallucinations, lack structured medical reasoning, and produce useless outputs. To address these challenges, we propose KERAP, a knowledge graph (KG)-enhanced reasoning approach that improves LLM-based diagnosis prediction through a multi-agent architecture. Our framework consists of a linkage agent for attribute mapping, a retrieval agent for structured knowledge extraction, and a prediction agent that iteratively refines diagnosis predictions. Experimental results demonstrate that KERAP enhances diagnostic reliability efficiently, offering a scalable and interpretable solution for zero-shot medical diagnosis prediction. 

---
# Measuring Scientific Capabilities of Language Models with a Systems Biology Dry Lab 

**Authors**: Haonan Duan, Stephen Zhewen Lu, Caitlin Fiona Harrigan, Nishkrit Desai, Jiarui Lu, Michał Koziarski, Leonardo Cotta, Chris J. Maddison  

**Link**: [PDF](https://arxiv.org/pdf/2507.02083)  

**Abstract**: Designing experiments and result interpretations are core scientific competencies, particularly in biology, where researchers perturb complex systems to uncover the underlying systems. Recent efforts to evaluate the scientific capabilities of large language models (LLMs) fail to test these competencies because wet-lab experimentation is prohibitively expensive: in expertise, time and equipment. We introduce SciGym, a first-in-class benchmark that assesses LLMs' iterative experiment design and analysis abilities in open-ended scientific discovery tasks. SciGym overcomes the challenge of wet-lab costs by running a dry lab of biological systems. These models, encoded in Systems Biology Markup Language, are efficient for generating simulated data, making them ideal testbeds for experimentation on realistically complex systems. We evaluated six frontier LLMs on 137 small systems, and released a total of 350 systems. Our evaluation shows that while more capable models demonstrated superior performance, all models' performance declined significantly as system complexity increased, suggesting substantial room for improvement in the scientific capabilities of LLM agents. 

---
# Do Role-Playing Agents Practice What They Preach? Belief-Behavior Consistency in LLM-Based Simulations of Human Trust 

**Authors**: Amogh Mannekote, Adam Davies, Guohao Li, Kristy Elizabeth Boyer, ChengXiang Zhai, Bonnie J Dorr, Francesco Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2507.02197)  

**Abstract**: As LLMs are increasingly studied as role-playing agents to generate synthetic data for human behavioral research, ensuring that their outputs remain coherent with their assigned roles has become a critical concern. In this paper, we investigate how consistently LLM-based role-playing agents' stated beliefs about the behavior of the people they are asked to role-play ("what they say") correspond to their actual behavior during role-play ("how they act"). Specifically, we establish an evaluation framework to rigorously measure how well beliefs obtained by prompting the model can predict simulation outcomes in advance. Using an augmented version of the GenAgents persona bank and the Trust Game (a standard economic game used to quantify players' trust and reciprocity), we introduce a belief-behavior consistency metric to systematically investigate how it is affected by factors such as: (1) the types of beliefs we elicit from LLMs, like expected outcomes of simulations versus task-relevant attributes of individual characters LLMs are asked to simulate; (2) when and how we present LLMs with relevant information about Trust Game; and (3) how far into the future we ask the model to forecast its actions. We also explore how feasible it is to impose a researcher's own theoretical priors in the event that the originally elicited beliefs are misaligned with research objectives. Our results reveal systematic inconsistencies between LLMs' stated (or imposed) beliefs and the outcomes of their role-playing simulation, at both an individual- and population-level. Specifically, we find that, even when models appear to encode plausible beliefs, they may fail to apply them in a consistent way. These findings highlight the need to identify how and when LLMs' stated beliefs align with their simulated behavior, allowing researchers to use LLM-based agents appropriately in behavioral studies. 

---
# Hey AI, Generate Me a Hardware Code! Agentic AI-based Hardware Design & Verification 

**Authors**: Deepak Narayan Gadde, Keerthan Kopparam Radhakrishna, Vaisakh Naduvodi Viswambharan, Aman Kumar, Djones Lettnin, Wolfgang Kunz, Sebastian Simon  

**Link**: [PDF](https://arxiv.org/pdf/2507.02660)  

**Abstract**: Modern Integrated Circuits (ICs) are becoming increasingly complex, and so is their development process. Hardware design verification entails a methodical and disciplined approach to the planning, development, execution, and sign-off of functionally correct hardware designs. This tedious process requires significant effort and time to ensure a bug-free tape-out. The field of Natural Language Processing has undergone a significant transformation with the advent of Large Language Models (LLMs). These powerful models, often referred to as Generative AI (GenAI), have revolutionized how machines understand and generate human language, enabling unprecedented advancements in a wide array of applications, including hardware design verification. This paper presents an agentic AI-based approach to hardware design verification, which empowers AI agents, in collaboration with Humain-in-the-Loop (HITL) intervention, to engage in a more dynamic, iterative, and self-reflective process, ultimately performing end-to-end hardware design and verification. This methodology is evaluated on five open-source designs, achieving over 95% coverage with reduced verification time while demonstrating superior performance, adaptability, and configurability. 

---
# STELLA: Self-Evolving LLM Agent for Biomedical Research 

**Authors**: Ruofan Jin, Zaixi Zhang, Mengdi Wang, Le Cong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02004)  

**Abstract**: The rapid growth of biomedical data, tools, and literature has created a fragmented research landscape that outpaces human expertise. While AI agents offer a solution, they typically rely on static, manually curated toolsets, limiting their ability to adapt and scale. Here, we introduce STELLA, a self-evolving AI agent designed to overcome these limitations. STELLA employs a multi-agent architecture that autonomously improves its own capabilities through two core mechanisms: an evolving Template Library for reasoning strategies and a dynamic Tool Ocean that expands as a Tool Creation Agent automatically discovers and integrates new bioinformatics tools. This allows STELLA to learn from experience. We demonstrate that STELLA achieves state-of-the-art accuracy on a suite of biomedical benchmarks, scoring approximately 26\% on Humanity's Last Exam: Biomedicine, 54\% on LAB-Bench: DBQA, and 63\% on LAB-Bench: LitQA, outperforming leading models by up to 6 percentage points. More importantly, we show that its performance systematically improves with experience; for instance, its accuracy on the Humanity's Last Exam benchmark almost doubles with increased trials. STELLA represents a significant advance towards AI Agent systems that can learn and grow, dynamically scaling their expertise to accelerate the pace of biomedical discovery. 

---
# Answer Matching Outperforms Multiple Choice for Language Model Evaluation 

**Authors**: Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2507.02856)  

**Abstract**: Multiple choice benchmarks have long been the workhorse of language model evaluation because grading multiple choice is objective and easy to automate. However, we show multiple choice questions from popular benchmarks can often be answered without even seeing the question. These shortcuts arise from a fundamental limitation of discriminative evaluation not shared by evaluations of the model's free-form, generative answers. Until recently, there appeared to be no viable, scalable alternative to multiple choice--but, we show that this has changed. We consider generative evaluation via what we call answer matching: Give the candidate model the question without the options, have it generate a free-form response, then use a modern language model with the reference answer to determine if the response matches the reference. To compare the validity of different evaluation strategies, we annotate MMLU-Pro and GPQA-Diamond to obtain human grading data, and measure the agreement of each evaluation approach. We find answer matching using recent models--even small ones--achieves near-perfect agreement, in the range of inter-annotator agreement. In contrast, both multiple choice evaluation and using LLM-as-a-judge without reference answers aligns poorly with human grading. Improving evaluations via answer matching is not merely a conceptual concern: the rankings of several models change significantly when evaluating their free-form responses with answer matching. In light of these findings, we discuss how to move the evaluation ecosystem from multiple choice to answer matching. 

---
# Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs 

**Authors**: Ken Tsui  

**Link**: [PDF](https://arxiv.org/pdf/2507.02778)  

**Abstract**: Although large language models (LLMs) have become transformative, they still make mistakes and can explore unproductive reasoning paths. Self-correction is an important capability for a trustworthy LLM, particularly an autoregressive LLM. While LLMs can identify error in user input, they exhibit a systematic 'Self-Correction Blind Spot' - failing to correct identical error in their own outputs. To systematically study this phenomenon, we introduce Self-Correction Bench, a systematic framework to measure this phenomenon through controlled error injection at three complexity levels. Testing 14 models, we find an average 64.5% blind spot rate. We find multiple evidences that this limitation relates to training data composition: human training demonstrations predominantly show error-free responses rather than error-correction sequences, unlike RL-trained models that learn error correction through outcome feedback. Remarkably, simply appending "Wait" reduces blind spots by 89.3%, suggesting that the capability exists but requires activation. Our work highlights a critical limitation in current LLMs and offers potential avenues for improving their reliability and trustworthiness. 

---
# SynapseRoute: An Auto-Route Switching Framework on Dual-State Large Language Model 

**Authors**: Wencheng Zhang, Shiqin Qiao, Lingjie Luo, Yinfeng Li, Chuanyang Zheng, Qian Xu, Meng Li, Yong Gui, Yijun He, Jianing Qiu, Jindong Hong, Jiankai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.02822)  

**Abstract**: With the widespread adoption of large language models (LLMs) in practical applications, selecting an appropriate model requires balancing not only performance but also operational cost. The emergence of reasoning-capable models has further widened the cost gap between "thinking" (high reasoning) and "non-thinking" (fast, low-cost) modes. In this work, we reveal that approximately 58% of medical questions can be accurately answered by the non-thinking mode alone, without requiring the high-cost reasoning process. This highlights a clear dichotomy in problem complexity and suggests that dynamically routing queries to the appropriate mode based on complexity could optimize accuracy, cost-efficiency, and overall user experience. Based on this, we further propose SynapseRoute, a machine learning-based dynamic routing framework that intelligently assigns input queries to either thinking or non-thinking modes. Experimental results on several medical datasets demonstrate that SynapseRoute not only improves overall accuracy (0.8390 vs. 0.8272) compared to the thinking mode alone but also reduces inference time by 36.8% and token consumption by 39.66%. Importantly, qualitative analysis indicates that over-reasoning on simpler queries can lead to unnecessary delays and even decreased accuracy, a pitfall avoided by our adaptive routing. Finally, this work further introduces the Accuracy-Inference-Token (AIT) index to comprehensively evaluate the trade-offs among accuracy, latency, and token cost. 

---
# Early Signs of Steganographic Capabilities in Frontier LLMs 

**Authors**: Artur Zolkowski, Kei Nishimura-Gasparian, Robert McCarthy, Roland S. Zimmermann, David Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2507.02737)  

**Abstract**: Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances such as using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future. 

---
# Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs 

**Authors**: Mohammad Ali Alomrani, Yingxue Zhang, Derek Li, Qianyi Sun, Soumyasundar Pal, Zhanguang Zhang, Yaochen Hu, Rohan Deepak Ajwani, Antonios Valkanas, Raika Karimi, Peng Cheng, Yunzhou Wang, Pengyi Liao, Hanrui Huang, Bin Wang, Jianye Hao, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2507.02076)  

**Abstract**: Large language models (LLMs) have rapidly progressed into general-purpose agents capable of solving a broad spectrum of tasks. However, current models remain inefficient at reasoning: they apply fixed inference-time compute regardless of task complexity, often overthinking simple problems while underthinking hard ones. This survey presents a comprehensive review of efficient test-time compute (TTC) strategies, which aim to improve the computational efficiency of LLM reasoning. We introduce a two-tiered taxonomy that distinguishes between L1-controllability, methods that operate under fixed compute budgets, and L2-adaptiveness, methods that dynamically scale inference based on input difficulty or model confidence. We benchmark leading proprietary LLMs across diverse datasets, highlighting critical trade-offs between reasoning performance and token usage. Compared to prior surveys on efficient reasoning, our review emphasizes the practical control, adaptability, and scalability of TTC methods. Finally, we discuss emerging trends such as hybrid thinking models and identify key challenges for future work towards making LLMs more computationally efficient, robust, and responsive to user constraints. 

---
# Data Diversification Methods In Alignment Enhance Math Performance In LLMs 

**Authors**: Berkan Dokmeci, Qingyang Wu, Ben Athiwaratkun, Ce Zhang, Shuaiwen Leon Song, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02173)  

**Abstract**: While recent advances in preference learning have enhanced alignment in human feedback, mathematical reasoning remains a persistent challenge. We investigate how data diversification strategies in preference optimization can improve the mathematical reasoning abilities of large language models (LLMs). We evaluate three common data generation methods: temperature sampling, Chain-of-Thought prompting, and Monte Carlo Tree Search (MCTS), and introduce Diversified-ThinkSolve (DTS), a novel structured approach that systematically decomposes problems into diverse reasoning paths. Our results show that with strategically diversified preference data, models can substantially improve mathematical reasoning performance, with the best approach yielding gains of 7.1% on GSM8K and 4.2% on MATH over the base model. Despite its strong performance, DTS incurs only a marginal computational overhead (1.03x) compared to the baseline, while MCTS is nearly five times more costly with lower returns. These findings demonstrate that structured exploration of diverse problem-solving methods creates more effective preference data for mathematical alignment than traditional approaches. 

---
# MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion 

**Authors**: Xin Guan, PeiHsin Lin, Zekun Wu, Ze Wang, Ruibo Zhang, Emre Kazim, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2507.02595)  

**Abstract**: Multiperspective Fusion (MPF) is a novel posttraining alignment framework for large language models (LLMs) developed in response to the growing need for easy bias mitigation. Built on top of the SAGED pipeline, an automated system for constructing bias benchmarks and extracting interpretable baseline distributions, MPF leverages multiperspective generations to expose and align biases in LLM outputs with nuanced, humanlike baselines. By decomposing baseline, such as sentiment distributions from HR professionals, into interpretable perspective components, MPF guides generation through sampling and balancing of responses, weighted by the probabilities obtained in the decomposition. Empirically, we demonstrate its ability to align LLM sentiment distributions with both counterfactual baselines (absolute equality) and the HR baseline (biased for Top Univeristy), resulting in small KL divergence, reduction of calibration error and generalization to unseen questions. This shows that MPF offers a scalable and interpretable method for alignment and bias mitigation, compatible with deployed LLMs and requiring no extensive prompt engineering or finetuning. 

---
# Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks 

**Authors**: Sizhe Chen, Arman Zharmagambetov, David Wagner, Chuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02735)  

**Abstract**: Prompt injection attacks pose a significant security threat to LLM-integrated applications. Model-level defenses have shown strong effectiveness, but are currently deployed into commercial-grade models in a closed-source manner. We believe open-source models are needed by the AI security community, where co-development of attacks and defenses through open research drives scientific progress in mitigation against prompt injection attacks. To this end, we develop Meta SecAlign, the first open-source and open-weight LLM with built-in model-level defense that achieves commercial-grade model performance. We provide complete details of our training recipe, which utilizes an improved version of the SOTA SecAlign defense. Evaluations on 9 utility benchmarks and 7 security benchmarks show that Meta SecAlign, despite being trained on a generic instruction-tuning dataset, confers security in unseen downstream tasks, including tool-calling and agentic web navigation, in addition general instruction-following. Our best model -- Meta-SecAlign-70B -- achieves state-of-the-art robustness against prompt injection attacks and comparable utility to closed-source commercial LLM with model-level defense. 

---
# Are You Listening to Me? Fine-Tuning Chatbots for Empathetic Dialogue 

**Authors**: Paulo Ricardo Knob, Leonardo Scholler, Juliano Rigatti, Soraia Raupp Musse  

**Link**: [PDF](https://arxiv.org/pdf/2507.02537)  

**Abstract**: Conversational agents have made significant progress since ELIZA, expanding their role across various domains, including healthcare, education, and customer service. As these agents become increasingly integrated into daily human interactions, the need for emotional intelligence, particularly empathetic listening, becomes increasingly essential. In this study, we explore how Large Language Models (LLMs) respond when tasked with generating emotionally rich interactions. Starting from a small dataset manually crafted by an expert to reflect empathic behavior, we extended the conversations using two LLMs: ChatGPT and Gemini. We analyzed the emotional progression of the dialogues using both sentiment analysis (via VADER) and expert assessments. While the generated conversations often mirrored the intended emotional structure, human evaluation revealed important differences in the perceived empathy and coherence of the responses. These findings suggest that emotion modeling in dialogues requires not only structural alignment in the expressed emotions but also qualitative depth, highlighting the importance of combining automated and humancentered methods in the development of emotionally competent agents. 

---
# MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs 

**Authors**: Purbesh Mitra, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2507.02851)  

**Abstract**: Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at this https URL and this https URL, respectively. 

---
# CyberRAG: An agentic RAG cyber attack classification and reporting tool 

**Authors**: Francesco Blefari, Cristian Cosentino, Francesco Aurelio Pironti, Angelo Furfaro, Fabrizio Marozzo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02424)  

**Abstract**: Intrusion Detection and Prevention Systems (IDS/IPS) in large enterprises can generate hundreds of thousands of alerts per hour, overwhelming security analysts with logs that demand deep, rapidly evolving domain expertise. Conventional machine-learning detectors trim the alert volume but still yield high false-positive rates, while standard single-pass Retrieval-Augmented Generation (RAG) pipelines often retrieve irrelevant context and fail to justify their predictions. To overcome these shortcomings, we present CyberRAG, a modular, agent-based RAG framework that delivers real-time classification, explanation, and structured reporting for cyber-attacks. A central LLM agent orchestrates (i) a pool of fine-tuned specialized classifiers, each tailored to a distinct attack family; (ii) tool adapters for enrichment and alerting; and (iii) an iterative retrieval-and-reason loop that continuously queries a domain-specific knowledge base until the evidence is both relevant and self-consistent. Unlike traditional RAG systems, CyberRAG embraces an agentic design that enables dynamic control flow and adaptive reasoning. This agent-centric architecture refines its threat labels and natural-language justifications autonomously, reducing false positives and enhancing interpretability. The framework is fully extensible: new attack types can be supported by simply adding a classifier without retraining the core agent. CyberRAG has been evaluated achieving over 94% accuracy per class and pushing final classification accuracy to 94.92% through semantic orchestration. Generated explanations score up to 0.94 in BERTScore and 4.9/5 in GPT-4-based expert evaluation. These results show that agentic, specialist-oriented RAG can pair high detection accuracy with trustworthy, SOC-ready prose, offering a practical and scalable path toward semi-autonomous cyber-defence workflows. 

---
# Evaluating Language Models For Threat Detection in IoT Security Logs 

**Authors**: Jorge J. Tejero-Fernández, Alfonso Sánchez-Macián  

**Link**: [PDF](https://arxiv.org/pdf/2507.02390)  

**Abstract**: Log analysis is a relevant research field in cybersecurity as they can provide a source of information for the detection of threats to networks and systems. This paper presents a pipeline to use fine-tuned Large Language Models (LLMs) for anomaly detection and mitigation recommendation using IoT security logs. Utilizing classical machine learning classifiers as a baseline, three open-source LLMs are compared for binary and multiclass anomaly detection, with three strategies: zero-shot, few-shot prompting and fine-tuning using an IoT dataset. LLMs give better results on multi-class attack classification than the corresponding baseline models. By mapping detected threats to MITRE CAPEC, defining a set of IoT-specific mitigation actions, and fine-tuning the models with those actions, the models are able to provide a combined detection and recommendation guidance. 

---
# Continual Gradient Low-Rank Projection Fine-Tuning for LLMs 

**Authors**: Chenxu Wang, Yilin Lyu, Zicheng Sun, Liping Jing  

**Link**: [PDF](https://arxiv.org/pdf/2507.02503)  

**Abstract**: Continual fine-tuning of Large Language Models (LLMs) is hampered by the trade-off between efficiency and expressiveness. Low-Rank Adaptation (LoRA) offers efficiency but constrains the model's ability to learn new tasks and transfer knowledge due to its low-rank nature and reliance on explicit parameter constraints. We propose GORP (Gradient LOw Rank Projection) for Continual Learning, a novel training strategy that overcomes these limitations by synergistically combining full and low-rank parameters and jointly updating within a unified low-rank gradient subspace. GORP expands the optimization space while preserving efficiency and mitigating catastrophic forgetting. Extensive experiments on continual learning benchmarks demonstrate GORP's superior performance compared to existing state-of-the-art approaches. Code is available at this https URL. 

---
# Synthetic Heuristic Evaluation: A Comparison between AI- and Human-Powered Usability Evaluation 

**Authors**: Ruican Zhong, David W. McDonald, Gary Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2507.02306)  

**Abstract**: Usability evaluation is crucial in human-centered design but can be costly, requiring expert time and user compensation. In this work, we developed a method for synthetic heuristic evaluation using multimodal LLMs' ability to analyze images and provide design feedback. Comparing our synthetic evaluations to those by experienced UX practitioners across two apps, we found our evaluation identified 73% and 77% of usability issues, which exceeded the performance of 5 experienced human evaluators (57% and 63%). Compared to human evaluators, the synthetic evaluation's performance maintained consistent performance across tasks and excelled in detecting layout issues, highlighting potential attentional and perceptual strengths of synthetic evaluation. However, synthetic evaluation struggled with recognizing some UI components and design conventions, as well as identifying across screen violations. Additionally, testing synthetic evaluations over time and accounts revealed stable performance. Overall, our work highlights the performance differences between human and LLM-driven evaluations, informing the design of synthetic heuristic evaluations. 

---
# Toward a Robust and Generalizable Metamaterial Foundation Model 

**Authors**: Namjung Kim, Dongseok Lee, Jongbin Yu, Sung Woong Cho, Dosung Lee, Yesol Park, Youngjoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02436)  

**Abstract**: Advances in material functionalities drive innovations across various fields, where metamaterials-defined by structure rather than composition-are leading the way. Despite the rise of artificial intelligence (AI)-driven design strategies, their impact is limited by task-specific retraining, poor out-of-distribution(OOD) generalization, and the need for separate models for forward and inverse design. To address these limitations, we introduce the Metamaterial Foundation Model (MetaFO), a Bayesian transformer-based foundation model inspired by large language models. MetaFO learns the underlying mechanics of metamaterials, enabling probabilistic, zero-shot predictions across diverse, unseen combinations of material properties and structural responses. It also excels in nonlinear inverse design, even under OOD conditions. By treating metamaterials as an operator that maps material properties to structural responses, MetaFO uncovers intricate structure-property relationships and significantly expands the design space. This scalable and generalizable framework marks a paradigm shift in AI-driven metamaterial discovery, paving the way for next-generation innovations. 

---
# Content filtering methods for music recommendation: A review 

**Authors**: Terence Zeng, Abhishek K. Umrawal  

**Link**: [PDF](https://arxiv.org/pdf/2507.02282)  

**Abstract**: Recommendation systems have become essential in modern music streaming platforms, shaping how users discover and engage with songs. One common approach in recommendation systems is collaborative filtering, which suggests content based on the preferences of users with similar listening patterns to the target user. However, this method is less effective on media where interactions are sparse. Music is one such medium, since the average user of a music streaming service will never listen to the vast majority of tracks. Due to this sparsity, there are several challenges that have to be addressed with other methods. This review examines the current state of research in addressing these challenges, with an emphasis on the role of content filtering in mitigating biases inherent in collaborative filtering approaches. We explore various methods of song classification for content filtering, including lyrical analysis using Large Language Models (LLMs) and audio signal processing techniques. Additionally, we discuss the potential conflicts between these different analysis methods and propose avenues for resolving such discrepancies. 

---
# Reasoning or Not? A Comprehensive Evaluation of Reasoning LLMs for Dialogue Summarization 

**Authors**: Keyan Jin, Yapeng Wang, Leonel Santos, Tao Fang, Xu Yang, Sio Kei Im, Hugo Gonçalo Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2507.02145)  

**Abstract**: Dialogue summarization is a challenging task with significant practical value in customer service, meeting analysis, and conversational AI. Although large language models (LLMs) have achieved substantial progress in summarization tasks, the performance of step-by-step reasoning architectures-specifically Long Chain-of-Thought (CoT) implementations such as OpenAI-o1 and DeepSeek-R1-remains unexplored for dialogue scenarios requiring concurrent abstraction and conciseness. In this work, we present the first comprehensive and systematic evaluation of state-of-the-art reasoning LLMs and non-reasoning LLMs across three major paradigms-generic, role-oriented, and query-oriented dialogue summarization. Our study spans diverse languages, domains, and summary lengths, leveraging strong benchmarks (SAMSum, DialogSum, CSDS, and QMSum) and advanced evaluation protocols that include both LLM-based automatic metrics and human-inspired criteria. Contrary to trends in other reasoning-intensive tasks, our findings show that explicit stepwise reasoning does not consistently improve dialogue summarization quality. Instead, reasoning LLMs are often prone to verbosity, factual inconsistencies, and less concise summaries compared to their non-reasoning counterparts. Through scenario-specific analyses and detailed case studies, we further identify when and why explicit reasoning may fail to benefit-or even hinder-summarization in complex dialogue contexts. Our work provides new insights into the limitations of current reasoning LLMs and highlights the need for targeted modeling and evaluation strategies for real-world dialogue summarization. 

---
# SurgVisAgent: Multimodal Agentic Model for Versatile Surgical Visual Enhancement 

**Authors**: Zeyu Lei, Hongyuan Yu, Jinlin Wu, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02252)  

**Abstract**: Precise surgical interventions are vital to patient safety, and advanced enhancement algorithms have been developed to assist surgeons in decision-making. Despite significant progress, these algorithms are typically designed for single tasks in specific scenarios, limiting their effectiveness in complex real-world situations. To address this limitation, we propose SurgVisAgent, an end-to-end intelligent surgical vision agent built on multimodal large language models (MLLMs). SurgVisAgent dynamically identifies distortion categories and severity levels in endoscopic images, enabling it to perform a variety of enhancement tasks such as low-light enhancement, overexposure correction, motion blur elimination, and smoke removal. Specifically, to achieve superior surgical scenario understanding, we design a prior model that provides domain-specific knowledge. Additionally, through in-context few-shot learning and chain-of-thought (CoT) reasoning, SurgVisAgent delivers customized image enhancements tailored to a wide range of distortion types and severity levels, thereby addressing the diverse requirements of surgeons. Furthermore, we construct a comprehensive benchmark simulating real-world surgical distortions, on which extensive experiments demonstrate that SurgVisAgent surpasses traditional single-task models, highlighting its potential as a unified solution for surgical assistance. 

---
# Large Language Models for Crash Detection in Video: A Survey of Methods, Datasets, and Challenges 

**Authors**: Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.02074)  

**Abstract**: Crash detection from video feeds is a critical problem in intelligent transportation systems. Recent developments in large language models (LLMs) and vision-language models (VLMs) have transformed how we process, reason about, and summarize multimodal information. This paper surveys recent methods leveraging LLMs for crash detection from video data. We present a structured taxonomy of fusion strategies, summarize key datasets, analyze model architectures, compare performance benchmarks, and discuss ongoing challenges and opportunities. Our review provides a foundation for future research in this fast-growing intersection of video understanding and foundation models. 

---
# ESTR-CoT: Towards Explainable and Accurate Event Stream based Scene Text Recognition with Chain-of-Thought Reasoning 

**Authors**: Xiao Wang, Jingtao Jiang, Qiang Chen, Lan Chen, Lin Zhu, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02200)  

**Abstract**: Event stream based scene text recognition is a newly arising research topic in recent years which performs better than the widely used RGB cameras in extremely challenging scenarios, especially the low illumination, fast motion. Existing works either adopt end-to-end encoder-decoder framework or large language models for enhanced recognition, however, they are still limited by the challenges of insufficient interpretability and weak contextual logical reasoning. In this work, we propose a novel chain-of-thought reasoning based event stream scene text recognition framework, termed ESTR-CoT. Specifically, we first adopt the vision encoder EVA-CLIP (ViT-G/14) to transform the input event stream into tokens and utilize a Llama tokenizer to encode the given generation prompt. A Q-former is used to align the vision token to the pre-trained large language model Vicuna-7B and output both the answer and chain-of-thought (CoT) reasoning process simultaneously. Our framework can be optimized using supervised fine-tuning in an end-to-end manner. In addition, we also propose a large-scale CoT dataset to train our framework via a three stage processing (i.e., generation, polish, and expert verification). This dataset provides a solid data foundation for the development of subsequent reasoning-based large models. Extensive experiments on three event stream STR benchmark datasets (i.e., EventSTR, WordArt*, IC15*) fully validated the effectiveness and interpretability of our proposed framework. The source code and pre-trained models will be released on this https URL. 

---
# Towards a Playground to Democratize Experimentation and Benchmarking of AI Agents for Network Troubleshooting 

**Authors**: Zhihao Wang, Alessandro Cornacchia, Franco Galante, Carlo Centofanti, Alessio Sacco, Dingde Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01997)  

**Abstract**: Recent research has demonstrated the effectiveness of Artificial Intelligence (AI), and more specifically, Large Language Models (LLMs), in supporting network configuration synthesis and automating network diagnosis tasks, among others. In this preliminary work, we restrict our focus to the application of AI agents to network troubleshooting and elaborate on the need for a standardized, reproducible, and open benchmarking platform, where to build and evaluate AI agents with low operational effort. 

---
# Integrating Large Language Models in Financial Investments and Market Analysis: A Survey 

**Authors**: Sedigheh Mahdavi, Jiating, Chen, Pradeep Kumar Joshi, Lina Huertas Guativa, Upmanyu Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.01990)  

**Abstract**: Large Language Models (LLMs) have been employed in financial decision making, enhancing analytical capabilities for investment strategies. Traditional investment strategies often utilize quantitative models, fundamental analysis, and technical indicators. However, LLMs have introduced new capabilities to process and analyze large volumes of structured and unstructured data, extract meaningful insights, and enhance decision-making in real-time. This survey provides a structured overview of recent research on LLMs within the financial domain, categorizing research contributions into four main frameworks: LLM-based Frameworks and Pipelines, Hybrid Integration Methods, Fine-Tuning and Adaptation Approaches, and Agent-Based Architectures. This study provides a structured review of recent LLMs research on applications in stock selection, risk assessment, sentiment analysis, trading, and financial forecasting. By reviewing the existing literature, this study highlights the capabilities, challenges, and potential directions of LLMs in financial markets. 

---
# When LLMs Disagree: Diagnosing Relevance Filtering Bias and Retrieval Divergence in SDG Search 

**Authors**: William A. Ingram, Bipasha Banerjee, Edward A. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2507.02139)  

**Abstract**: Large language models (LLMs) are increasingly used to assign document relevance labels in information retrieval pipelines, especially in domains lacking human-labeled data. However, different models often disagree on borderline cases, raising concerns about how such disagreement affects downstream retrieval. This study examines labeling disagreement between two open-weight LLMs, LLaMA and Qwen, on a corpus of scholarly abstracts related to Sustainable Development Goals (SDGs) 1, 3, and 7. We isolate disagreement subsets and examine their lexical properties, rank-order behavior, and classification predictability. Our results show that model disagreement is systematic, not random: disagreement cases exhibit consistent lexical patterns, produce divergent top-ranked outputs under shared scoring functions, and are distinguishable with AUCs above 0.74 using simple classifiers. These findings suggest that LLM-based filtering introduces structured variability in document retrieval, even under controlled prompting and shared ranking logic. We propose using classification disagreement as an object of analysis in retrieval evaluation, particularly in policy-relevant or thematic search tasks. 

---
# Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models 

**Authors**: Riccardo Cantini, Nicola Gabriele, Alessio Orsino, Domenico Talia  

**Link**: [PDF](https://arxiv.org/pdf/2507.02799)  

**Abstract**: Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design. 

---
# Multimodal Mathematical Reasoning with Diverse Solving Perspective 

**Authors**: Wenhao Shi, Zhiqiang Hu, Yi Bin, Yang Yang, See-Kiong Ng, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02804)  

**Abstract**: Recent progress in large-scale reinforcement learning (RL) has notably enhanced the reasoning capabilities of large language models (LLMs), especially in mathematical domains. However, current multimodal LLMs (MLLMs) for mathematical reasoning often rely on one-to-one image-text pairs and single-solution supervision, overlooking the diversity of valid reasoning perspectives and internal reflections. In this work, we introduce MathV-DP, a novel dataset that captures multiple diverse solution trajectories for each image-question pair, fostering richer reasoning supervision. We further propose Qwen-VL-DP, a model built upon Qwen-VL, fine-tuned with supervised learning and enhanced via group relative policy optimization (GRPO), a rule-based RL approach that integrates correctness discrimination and diversity-aware reward functions. Our method emphasizes learning from varied reasoning perspectives and distinguishing between correct yet distinct solutions. Extensive experiments on the MathVista's minitest and Math-V benchmarks demonstrate that Qwen-VL-DP significantly outperforms prior base MLLMs in both accuracy and generative diversity, highlighting the importance of incorporating diverse perspectives and reflective reasoning in multimodal mathematical reasoning. 

---
# Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers 

**Authors**: Zhijian Xu, Yilun Zhao, Manasi Patwardhan, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.02694)  

**Abstract**: Peer review is fundamental to scientific research, but the growing volume of publications has intensified the challenges of this expertise-intensive process. While LLMs show promise in various scientific tasks, their potential to assist with peer review, particularly in identifying paper limitations, remains understudied. We first present a comprehensive taxonomy of limitation types in scientific research, with a focus on AI. Guided by this taxonomy, for studying limitations, we present LimitGen, the first comprehensive benchmark for evaluating LLMs' capability to support early-stage feedback and complement human peer review. Our benchmark consists of two subsets: LimitGen-Syn, a synthetic dataset carefully created through controlled perturbations of high-quality papers, and LimitGen-Human, a collection of real human-written limitations. To improve the ability of LLM systems to identify limitations, we augment them with literature retrieval, which is essential for grounding identifying limitations in prior scientific findings. Our approach enhances the capabilities of LLM systems to generate limitations in research papers, enabling them to provide more concrete and constructive feedback. 

---
# Efficient Code LLM Training via Distribution-Consistent and Diversity-Aware Data Selection 

**Authors**: Weijie Lyu, Sheng-Jun Huang, Xuan Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.02378)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved code generation and program comprehension, accelerating the evolution of software engineering. Current methods primarily enhance model performance by leveraging vast amounts of data, focusing on data quantity while often overlooking data quality, thereby reducing training efficiency. To address this, we introduce an approach that utilizes a parametric model for code data selection, aimed at improving both training efficiency and model performance. Our method optimizes the parametric model to ensure distribution consistency and diversity within the selected subset, guaranteeing high-quality data. Experimental results demonstrate that using only 10K samples, our method achieves gains of 2.4% (HumanEval) and 2.3% (MBPP) over 92K full-sampled baseline, outperforming other sampling approaches in both performance and efficiency. This underscores that our method effectively boosts model performance while significantly reducing computational costs. 

---
# Revisiting Active Learning under (Human) Label Variation 

**Authors**: Cornelia Gruber, Helen Alber, Bernd Bischl, Göran Kauermann, Barbara Plank, Matthias Aßenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2507.02593)  

**Abstract**: Access to high-quality labeled data remains a limiting factor in applied supervised learning. While label variation (LV), i.e., differing labels for the same instance, is common, especially in natural language processing, annotation frameworks often still rest on the assumption of a single ground truth. This overlooks human label variation (HLV), the occurrence of plausible differences in annotations, as an informative signal. Similarly, active learning (AL), a popular approach to optimizing the use of limited annotation budgets in training ML models, often relies on at least one of several simplifying assumptions, which rarely hold in practice when acknowledging HLV. In this paper, we examine foundational assumptions about truth and label nature, highlighting the need to decompose observed LV into signal (e.g., HLV) and noise (e.g., annotation error). We survey how the AL and (H)LV communities have addressed -- or neglected -- these distinctions and propose a conceptual framework for incorporating HLV throughout the AL loop, including instance selection, annotator choice, and label representation. We further discuss the integration of large language models (LLM) as annotators. Our work aims to lay a conceptual foundation for HLV-aware active learning, better reflecting the complexities of real-world annotation. 

---
# Requirements Elicitation Follow-Up Question Generation 

**Authors**: Yuchen Shen, Anmol Singhal, Travis Breaux  

**Link**: [PDF](https://arxiv.org/pdf/2507.02858)  

**Abstract**: Interviews are a widely used technique in eliciting requirements to gather stakeholder needs, preferences, and expectations for a software system. Effective interviewing requires skilled interviewers to formulate appropriate interview questions in real time while facing multiple challenges, including lack of familiarity with the domain, excessive cognitive load, and information overload that hinders how humans process stakeholders' speech. Recently, large language models (LLMs) have exhibited state-of-the-art performance in multiple natural language processing tasks, including text summarization and entailment. To support interviewers, we investigate the application of GPT-4o to generate follow-up interview questions during requirements elicitation by building on a framework of common interviewer mistake types. In addition, we describe methods to generate questions based on interviewee speech. We report a controlled experiment to evaluate LLM-generated and human-authored questions with minimal guidance, and a second controlled experiment to evaluate the LLM-generated questions when generation is guided by interviewer mistake types. Our findings demonstrate that, for both experiments, the LLM-generated questions are no worse than the human-authored questions with respect to clarity, relevancy, and informativeness. In addition, LLM-generated questions outperform human-authored questions when guided by common mistakes types. This highlights the potential of using LLMs to help interviewers improve the quality and ease of requirements elicitation interviews in real time. 

---
# LLM Hypnosis: Exploiting User Feedback for Unauthorized Knowledge Injection to All Users 

**Authors**: Almog Hilel, Idan Shenfeld, Leshem Choshen, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2507.02850)  

**Abstract**: We describe a vulnerability in language models (LMs) trained with user feedback, whereby a single user can persistently alter LM knowledge and behavior given only the ability to provide prompts and upvote / downvote feedback on LM outputs. To implement the attack, the attacker prompts the LM to stochastically output either a "poisoned" or benign response, then upvotes the poisoned response or downvotes the benign one. When feedback signals are used in a subsequent preference tuning behavior, LMs exhibit increased probability of producing poisoned responses even in contexts without malicious prompts. We show that this attack can be used to (1) insert factual knowledge the model did not previously possess, (2) modify code generation patterns in ways that introduce exploitable security flaws, and (3) inject fake financial news. Our finding both identifies a new qualitative feature of language model preference tuning (showing that it even highly restricted forms of preference data can be used to exert fine-grained control over behavior), and a new attack mechanism for LMs trained with user feedback (extending work on pretraining-time data poisoning and deployment-time prompt injection). 

---
# MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent 

**Authors**: Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02259)  

**Abstract**: Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents with linear complexity without performance degradation during extrapolation remains the ultimate challenge in long-text processing. We directly optimize for long-text tasks in an end-to-end fashion and introduce a novel agent workflow, MemAgent, which reads text in segments and updates the memory using an overwrite strategy. We extend the DAPO algorithm to facilitate training via independent-context multi-conversation generation. MemAgent has demonstrated superb long-context capabilities, being able to extrapolate from an 8K context trained on 32K text to a 3.5M QA task with performance loss < 5% and achieves 95%+ in 512K RULER test. 

---
# GDC Cohort Copilot: An AI Copilot for Curating Cohorts from the Genomic Data Commons 

**Authors**: Steven Song, Anirudh Subramanyam, Zhenyu Zhang, Aarti Venkat, Robert L. Grossman  

**Link**: [PDF](https://arxiv.org/pdf/2507.02221)  

**Abstract**: Motivation: The Genomic Data Commons (GDC) provides access to high quality, harmonized cancer genomics data through a unified curation and analysis platform centered around patient cohorts. While GDC users can interactively create complex cohorts through the graphical Cohort Builder, users (especially new ones) may struggle to find specific cohort descriptors across hundreds of possible fields and properties. However, users may be better able to describe their desired cohort in free-text natural language.
Results: We introduce GDC Cohort Copilot, an open-source copilot tool for curating cohorts from the GDC. GDC Cohort Copilot automatically generates the GDC cohort filter corresponding to a user-input natural language description of their desired cohort, before exporting the cohort back to the GDC for further analysis. An interactive user interface allows users to further refine the generated cohort. We develop and evaluate multiple large language models (LLMs) for GDC Cohort Copilot and demonstrate that our locally-served, open-source GDC Cohort LLM achieves better results than GPT-4o prompting in generating GDC cohorts.
Availability and implementation: The standalone docker image for GDC Cohort Copilot is available at this https URL. Source code is available at this https URL. GDC Cohort LLM weights are available at this https URL. 

---
# DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment 

**Authors**: Ke-Han Lu, Zhehuai Chen, Szu-Wei Fu, Chao-Han Huck Yang, Sung-Feng Huang, Chih-Kai Yang, Chee-En Yu, Chun-Wei Chen, Wei-Chih Chen, Chien-yu Huang, Yi-Cheng Lin, Yu-Xiang Lin, Chi-An Fu, Chun-Yi Kuan, Wenze Ren, Xuanjun Chen, Wei-Ping Huang, En-Pei Hu, Tzu-Quan Lin, Yuan-Kuei Wu, Kuan-Po Huang, Hsiao-Ying Huang, Huang-Cheng Chou, Kai-Wei Chang, Cheng-Han Chiang, Boris Ginsburg, Yu-Chiang Frank Wang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.02768)  

**Abstract**: We introduce DeSTA2.5-Audio, a general-purpose Large Audio Language Model (LALM) designed for robust auditory perception and instruction-following, without requiring task-specific audio instruction-tuning. Recent LALMs typically augment Large Language Models (LLMs) with auditory capabilities by training on large-scale, manually curated or LLM-synthesized audio-instruction datasets. However, these approaches have often suffered from the catastrophic forgetting of the LLM's original language abilities. To address this, we revisit the data construction pipeline and propose DeSTA, a self-generated cross-modal alignment strategy in which the backbone LLM generates its own training targets. This approach preserves the LLM's native language proficiency while establishing effective audio-text alignment, thereby enabling zero-shot generalization without task-specific tuning. Using DeSTA, we construct DeSTA-AQA5M, a large-scale, task-agnostic dataset containing 5 million training samples derived from 7,000 hours of audio spanning 50 diverse datasets, including speech, environmental sounds, and music. DeSTA2.5-Audio achieves state-of-the-art or competitive performance across a wide range of audio-language benchmarks, including Dynamic-SUPERB, MMAU, SAKURA, Speech-IFEval, and VoiceBench. Comprehensive comparative studies demonstrate that our self-generated strategy outperforms widely adopted data construction and training strategies in both auditory perception and instruction-following capabilities. Our findings underscore the importance of carefully designed data construction in LALM development and offer practical insights for building robust, general-purpose LALMs. 

---
# McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models 

**Authors**: Tian Lan, Xiangdong Su, Xu Liu, Ruirui Wang, Ke Chang, Jiang Li, Guanglai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02088)  

**Abstract**: As large language models (LLMs) are increasingly applied to various NLP tasks, their inherent biases are gradually disclosed. Therefore, measuring biases in LLMs is crucial to mitigate its ethical risks. However, most existing bias evaluation datasets focus on English and North American culture, and their bias categories are not fully applicable to other cultures. The datasets grounded in the Chinese language and culture are scarce. More importantly, these datasets usually only support single evaluation tasks and cannot evaluate the bias from multiple aspects in LLMs. To address these issues, we present a Multi-task Chinese Bias Evaluation Benchmark (McBE) that includes 4,077 bias evaluation instances, covering 12 single bias categories, 82 subcategories and introducing 5 evaluation tasks, providing extensive category coverage, content diversity, and measuring comprehensiveness. Additionally, we evaluate several popular LLMs from different series and with parameter sizes. In general, all these LLMs demonstrated varying degrees of bias. We conduct an in-depth analysis of results, offering novel insights into bias in LLMs. 

---
# ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning 

**Authors**: Ruiyang Zhou, Shuozhe Li, Amy Zhang, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02834)  

**Abstract**: Recent advances in large language models have been driven by reinforcement learning (RL)-style post-training, which improves reasoning by optimizing model outputs based on reward or preference signals. GRPO-style approaches implement this by using self-generated samples labeled by an outcome-based verifier. However, these methods depend heavily on the model's initial ability to produce positive samples. They primarily refine what the model already knows (distribution sharpening) rather than enabling the model to solve problems where it initially fails. This limitation is especially problematic in early-stage RL training and on challenging reasoning tasks, where positive samples are unlikely to be generated. To unlock reasoning ability in such settings, the model must explore new reasoning trajectories beyond its current output distribution. Such exploration requires access to sufficiently good positive samples to guide the learning. While expert demonstrations seem like a natural solution, we find that they are often ineffective in RL post-training. Instead, we identify two key properties of effective positive samples: they should (1) be likely under the current policy, and (2) increase the model's likelihood of predicting the correct answer. Based on these insights, we propose $\textbf{Self-Explanation Policy Optimization (ExPO)}$-a simple and modular framework that generates such samples by conditioning on the ground-truth answer. ExPO enables efficient exploration and guides the model to produce reasoning trajectories more aligned with its policy than expert-written CoTs, while ensuring higher quality than its own (incorrect) samples. Experiments show that ExPO improves both learning efficiency and final performance on reasoning benchmarks, surpassing expert-demonstration-based methods in challenging settings such as MATH level-5, where the model initially struggles the most. 

---
# Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection 

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02844)  

**Abstract**: With the emergence of strong visual-language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: visual-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct visual-focused strategies, dynamically generating auxiliary images when necessary to construct a visual-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which performs a toxicity score of 2.48 and an ASR of 22.2%. The code is available at this https URL. 

---
# Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions 

**Authors**: Eitan Anzenberg, Arunava Samajpati, Sivasankaran Chandrasekar, Varun Kacholia  

**Link**: [PDF](https://arxiv.org/pdf/2507.02087)  

**Abstract**: The use of large language models (LLMs) in hiring promises to streamline candidate screening, but it also raises serious concerns regarding accuracy and algorithmic bias where sufficient safeguards are not in place. In this work, we benchmark several state-of-the-art foundational LLMs - including models from OpenAI, Anthropic, Google, Meta, and Deepseek, and compare them with our proprietary domain-specific hiring model (Match Score) for job candidate matching. We evaluate each model's predictive accuracy (ROC AUC, Precision-Recall AUC, F1-score) and fairness (impact ratio of cut-off analysis across declared gender, race, and intersectional subgroups). Our experiments on a dataset of roughly 10,000 real-world recent candidate-job pairs show that Match Score outperforms the general-purpose LLMs on accuracy (ROC AUC 0.85 vs 0.77) and achieves significantly more equitable outcomes across demographic groups. Notably, Match Score attains a minimum race-wise impact ratio of 0.957 (near-parity), versus 0.809 or lower for the best LLMs, (0.906 vs 0.773 for the intersectionals, respectively). We discuss why pretraining biases may cause LLMs with insufficient safeguards to propagate societal biases in hiring scenarios, whereas a bespoke supervised model can more effectively mitigate these biases. Our findings highlight the importance of domain-specific modeling and bias auditing when deploying AI in high-stakes domains such as hiring, and caution against relying on off-the-shelf LLMs for such tasks without extensive fairness safeguards. Furthermore, we show with empirical evidence that there shouldn't be a dichotomy between choosing accuracy and fairness in hiring: a well-designed algorithm can achieve both accuracy in hiring and fairness in outcomes. 

---
# JoyTTS: LLM-based Spoken Chatbot With Voice Cloning 

**Authors**: Fangru Zhou, Jun Zhao, Guoxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02380)  

**Abstract**: JoyTTS is an end-to-end spoken chatbot that combines large language models (LLM) with text-to-speech (TTS) technology, featuring voice cloning capabilities. This project is built upon the open-source MiniCPM-o and CosyVoice2 models and trained on 2000 hours of conversational data. We have also provided the complete training code to facilitate further development and optimization by the community. On the testing machine seed-tts-zh, it achieves a SS (speaker similarity) score of 0.73 and a WER (Word Error Rate) of 5.09. The code and models, along with training and inference scripts, are available at this https URL. 

---
