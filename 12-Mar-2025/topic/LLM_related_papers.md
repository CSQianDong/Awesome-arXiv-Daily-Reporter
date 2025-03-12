# Reasoning and Sampling-Augmented MCQ Difficulty Prediction via LLMs 

**Authors**: Wanyong Feng, Peter Tran, Stephen Sireci, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08551)  

**Abstract**: The difficulty of multiple-choice questions (MCQs) is a crucial factor for educational assessments. Predicting MCQ difficulty is challenging since it requires understanding both the complexity of reaching the correct option and the plausibility of distractors, i.e., incorrect options. In this paper, we propose a novel, two-stage method to predict the difficulty of MCQs. First, to better estimate the complexity of each MCQ, we use large language models (LLMs) to augment the reasoning steps required to reach each option. We use not just the MCQ itself but also these reasoning steps as input to predict the difficulty. Second, to capture the plausibility of distractors, we sample knowledge levels from a distribution to account for variation among students responding to the MCQ. This setup, inspired by item response theory (IRT), enable us to estimate the likelihood of students selecting each (both correct and incorrect) option. We align these predictions with their ground truth values, using a Kullback-Leibler (KL) divergence-based regularization objective, and use estimated likelihoods to predict MCQ difficulty. We evaluate our method on two real-world \emph{math} MCQ and response datasets with ground truth difficulty values estimated using IRT. Experimental results show that our method outperforms all baselines, up to a 28.3\% reduction in mean squared error and a 34.6\% improvement in the coefficient of determination. We also qualitatively discuss how our novel method results in higher accuracy in predicting MCQ difficulty. 

---
# AI-native Memory 2.0: Second Me 

**Authors**: Jiale Wei, Xiang Ying, Tao Gao, Felix Tao, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08102)  

**Abstract**: Human interaction with the external world fundamentally involves the exchange of personal memory, whether with other individuals, websites, applications, or, in the future, AI agents. A significant portion of this interaction is redundant, requiring users to repeatedly provide the same information across different contexts. Existing solutions, such as browser-stored credentials, autofill mechanisms, and unified authentication systems, have aimed to mitigate this redundancy by serving as intermediaries that store and retrieve commonly used user data. The advent of large language models (LLMs) presents an opportunity to redefine memory management through an AI-native paradigm: SECOND ME. SECOND ME acts as an intelligent, persistent memory offload system that retains, organizes, and dynamically utilizes user-specific knowledge. By serving as an intermediary in user interactions, it can autonomously generate context-aware responses, prefill required information, and facilitate seamless communication with external systems, significantly reducing cognitive load and interaction friction. Unlike traditional memory storage solutions, SECOND ME extends beyond static data retention by leveraging LLM-based memory parameterization. This enables structured organization, contextual reasoning, and adaptive knowledge retrieval, facilitating a more systematic and intelligent approach to memory management. As AI-driven personal agents like SECOND ME become increasingly integrated into digital ecosystems, SECOND ME further represents a critical step toward augmenting human-world interaction with persistent, contextually aware, and self-optimizing memory systems. We have open-sourced the fully localizable deployment system at GitHub: this https URL. 

---
# Graph of AI Ideas: Leveraging Knowledge Graphs and LLMs for AI Research Idea Generation 

**Authors**: Xian Gao, Zongyun Zhang, Mingye Xie, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08549)  

**Abstract**: Reading relevant scientific papers and analyzing research development trends is a critical step in generating new scientific ideas. However, the rapid increase in the volume of research literature and the complex citation relationships make it difficult for researchers to quickly analyze and derive meaningful research trends. The development of large language models (LLMs) has provided a novel approach for automatically summarizing papers and generating innovative research ideas. However, existing paper-based idea generation methods either simply input papers into LLMs via prompts or form logical chains of creative development based on citation relationships, without fully exploiting the semantic information embedded in these citations. Inspired by knowledge graphs and human cognitive processes, we propose a framework called the Graph of AI Ideas (GoAI) for the AI research field, which is dominated by open-access papers. This framework organizes relevant literature into entities within a knowledge graph and summarizes the semantic information contained in citations into relations within the graph. This organization effectively reflects the relationships between two academic papers and the advancement of the AI research field. Such organization aids LLMs in capturing the current progress of research, thereby enhancing their creativity. Experimental results demonstrate the effectiveness of our approach in generating novel, clear, and effective research ideas. 

---
# Seeing and Reasoning with Confidence: Supercharging Multimodal LLMs with an Uncertainty-Aware Agentic Framework 

**Authors**: Zhuo Zhi, Chen Feng, Adam Daneshmend, Mine Orlu, Andreas Demosthenous, Lu Yin, Da Li, Ziquan Liu, Miguel R. D. Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2503.08308)  

**Abstract**: Multimodal large language models (MLLMs) show promise in tasks like visual question answering (VQA) but still face challenges in multimodal reasoning. Recent works adapt agentic frameworks or chain-of-thought (CoT) reasoning to improve performance. However, CoT-based multimodal reasoning often demands costly data annotation and fine-tuning, while agentic approaches relying on external tools risk introducing unreliable output from these tools. In this paper, we propose Seeing and Reasoning with Confidence (SRICE), a training-free multimodal reasoning framework that integrates external vision models with uncertainty quantification (UQ) into an MLLM to address these challenges. Specifically, SRICE guides the inference process by allowing MLLM to autonomously select regions of interest through multi-stage interactions with the help of external tools. We propose to use a conformal prediction-based approach to calibrate the output of external tools and select the optimal tool by estimating the uncertainty of an MLLM's output. Our experiment shows that the average improvement of SRICE over the base MLLM is 4.6% on five datasets and the performance on some datasets even outperforms fine-tuning-based methods, revealing the significance of ensuring reliable tool use in an MLLM agent. 

---
# LLM-Powered Knowledge Graphs for Enterprise Intelligence and Analytics 

**Authors**: Rajeev Kumar, Kumar Ishan, Harishankar Kumar, Abhinandan Singla  

**Link**: [PDF](https://arxiv.org/pdf/2503.07993)  

**Abstract**: Disconnected data silos within enterprises obstruct the extraction of actionable insights, diminishing efficiency in areas such as product development, client engagement, meeting preparation, and analytics-driven decision-making. This paper introduces a framework that uses large language models (LLMs) to unify various data sources into a comprehensive, activity-centric knowledge graph. The framework automates tasks such as entity extraction, relationship inference, and semantic enrichment, enabling advanced querying, reasoning, and analytics across data types like emails, calendars, chats, documents, and logs. Designed for enterprise flexibility, it supports applications such as contextual search, task prioritization, expertise discovery, personalized recommendations, and advanced analytics to identify trends and actionable insights. Experimental results demonstrate its success in the discovery of expertise, task management, and data-driven decision making. By integrating LLMs with knowledge graphs, this solution bridges disconnected systems and delivers intelligent analytics-powered enterprise tools. 

---
# Counterfactual Language Reasoning for Explainable Recommendation Systems 

**Authors**: Guanrong Li, Haolin Yang, Xinyu Liu, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.08051)  

**Abstract**: Explainable recommendation systems leverage transparent reasoning to foster user trust and improve decision-making processes. Current approaches typically decouple recommendation generation from explanation creation, violating causal precedence principles where explanatory factors should logically precede outcomes. This paper introduces a novel framework integrating structural causal models with large language models to establish causal consistency in recommendation pipelines. Our methodology enforces explanation factors as causal antecedents to recommendation predictions through causal graph construction and counterfactual adjustment. We particularly address the confounding effect of item popularity that distorts personalization signals in explanations, developing a debiasing mechanism that disentangles genuine user preferences from conformity bias. Through comprehensive experiments across multiple recommendation scenarios, we demonstrate that CausalX achieves superior performance in recommendation accuracy, explanation plausibility, and bias mitigation compared to baselines. 

---
# LLM-based Corroborating and Refuting Evidence Retrieval for Scientific Claim Verification 

**Authors**: Siyuan Wang, James R. Foulds, Md Osman Gani, Shimei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07937)  

**Abstract**: In this paper, we introduce CIBER (Claim Investigation Based on Evidence Retrieval), an extension of the Retrieval-Augmented Generation (RAG) framework designed to identify corroborating and refuting documents as evidence for scientific claim verification. CIBER addresses the inherent uncertainty in Large Language Models (LLMs) by evaluating response consistency across diverse interrogation probes. By focusing on the behavioral analysis of LLMs without requiring access to their internal information, CIBER is applicable to both white-box and black-box models. Furthermore, CIBER operates in an unsupervised manner, enabling easy generalization across various scientific domains. Comprehensive evaluations conducted using LLMs with varying levels of linguistic proficiency reveal CIBER's superior performance compared to conventional RAG approaches. These findings not only highlight the effectiveness of CIBER but also provide valuable insights for future advancements in LLM-based scientific claim verification. 

---
# The StudyChat Dataset: Student Dialogues With ChatGPT in an Artificial Intelligence Course 

**Authors**: Hunter McNichols, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07928)  

**Abstract**: The widespread availability of large language models (LLMs), such as ChatGPT, has significantly impacted education, raising both opportunities and challenges. Students can frequently interact with LLM-powered, interactive learning tools, but their usage patterns need to be analyzed to ensure ethical usage of these tools. To better understand how students interact with LLMs in an academic setting, we introduce \textbf{StudyChat}, a publicly available dataset capturing real-world student interactions with an LLM-powered tutoring chatbot in a semester-long, university-level artificial intelligence (AI) course. We deploy a web application that replicates ChatGPT's core functionalities, and use it to log student interactions with the LLM while working on programming assignments. We collect 1,197 conversations, which we annotate using a dialogue act labeling schema inspired by observed interaction patterns and prior research. Additionally, we analyze these interactions, highlight behavioral trends, and analyze how specific usage patterns relate to course outcomes. \textbf{StudyChat} provides a rich resource for the learning sciences and AI in education communities, enabling further research into the evolving role of LLMs in education. 

---
# Chemical reasoning in LLMs unlocks steerable synthesis planning and reaction mechanism elucidation 

**Authors**: Andres M Bran, Theo A Neukomm, Daniel P Armstrong, Zlatko Jončev, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2503.08537)  

**Abstract**: While machine learning algorithms have been shown to excel at specific chemical tasks, they have struggled to capture the strategic thinking that characterizes expert chemical reasoning, limiting their widespread adoption. Here we demonstrate that large language models (LLMs) can serve as powerful chemical reasoning engines when integrated with traditional search algorithms, enabling a new approach to computer-aided chemistry that mirrors human expert thinking. Rather than using LLMs to directly manipulate chemical structures, we leverage their ability to evaluate chemical strategies and guide search algorithms toward chemically meaningful solutions. We demonstrate this paradigm through two fundamental challenges: strategy-aware retrosynthetic planning and mechanism elucidation. In retrosynthetic planning, our method allows chemists to specify desired synthetic strategies in natural language to find routes that satisfy these constraints in vast searches. In mechanism elucidation, LLMs guide the search for plausible reaction mechanisms by combining chemical principles with systematic exploration. Our approach shows strong performance across diverse chemical tasks, with larger models demonstrating increasingly sophisticated chemical reasoning. Our approach establishes a new paradigm for computer-aided chemistry that combines the strategic understanding of LLMs with the precision of traditional chemical tools, opening possibilities for more intuitive and powerful chemical reasoning systems. 

---
# Fully Autonomous Programming using Iterative Multi-Agent Debugging with Large Language Models 

**Authors**: Anastasiia Grishina, Vadim Liventsev, Aki Härmä, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07693)  

**Abstract**: Program synthesis with Large Language Models (LLMs) suffers from a "near-miss syndrome": the generated code closely resembles a correct solution but fails unit tests due to minor errors. We address this with a multi-agent framework called Synthesize, Execute, Instruct, Debug, and Repair (SEIDR). Effectively applying SEIDR to instruction-tuned LLMs requires determining (a) optimal prompts for LLMs, (b) what ranking algorithm selects the best programs in debugging rounds, and (c) balancing the repair of unsuccessful programs with the generation of new ones. We empirically explore these trade-offs by comparing replace-focused, repair-focused, and hybrid debug strategies. We also evaluate lexicase and tournament selection to rank candidates in each generation. On Program Synthesis Benchmark 2 (PSB2), our framework outperforms both conventional use of OpenAI Codex without a repair phase and traditional genetic programming approaches. SEIDR outperforms the use of an LLM alone, solving 18 problems in C++ and 20 in Python on PSB2 at least once across experiments. To assess generalizability, we employ GPT-3.5 and Llama 3 on the PSB2 and HumanEval-X benchmarks. Although SEIDR with these models does not surpass current state-of-the-art methods on the Python benchmarks, the results on HumanEval-C++ are promising. SEIDR with Llama 3-8B achieves an average pass@100 of 84.2%. Across all SEIDR runs, 163 of 164 problems are solved at least once with GPT-3.5 in HumanEval-C++, and 162 of 164 with the smaller Llama 3-8B. We conclude that SEIDR effectively overcomes the near-miss syndrome in program synthesis with LLMs. 

---
# CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving 

**Authors**: Changxing Liu, Genjia Liu, Zijun Wang, Jinchang Yang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08683)  

**Abstract**: Vehicle-to-vehicle (V2V) cooperative autonomous driving holds great promise for improving safety by addressing the perception and prediction uncertainties inherent in single-agent systems. However, traditional cooperative methods are constrained by rigid collaboration protocols and limited generalization to unseen interactive scenarios. While LLM-based approaches offer generalized reasoning capabilities, their challenges in spatial planning and unstable inference latency hinder their direct application in cooperative driving. To address these limitations, we propose CoLMDriver, the first full-pipeline LLM-based cooperative driving system, enabling effective language-based negotiation and real-time driving control. CoLMDriver features a parallel driving pipeline with two key components: (i) an LLM-based negotiation module under an actor-critic paradigm, which continuously refines cooperation policies through feedback from previous decisions of all vehicles; and (ii) an intention-guided waypoint generator, which translates negotiation outcomes into executable waypoints. Additionally, we introduce InterDrive, a CARLA-based simulation benchmark comprising 10 challenging interactive driving scenarios for evaluating V2V cooperation. Experimental results demonstrate that CoLMDriver significantly outperforms existing approaches, achieving an 11% higher success rate across diverse highly interactive V2V driving scenarios. Code will be released on this https URL. 

---
# Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents 

**Authors**: Haoyu Wang, Sunhao Dai, Haiyuan Zhao, Liang Pang, Xiao Zhang, Gang Wang, Zhenhua Dong, Jun Xu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08684)  

**Abstract**: Previous studies have found that PLM-based retrieval models exhibit a preference for LLM-generated content, assigning higher relevance scores to these documents even when their semantic quality is comparable to human-written ones. This phenomenon, known as source bias, threatens the sustainable development of the information access ecosystem. However, the underlying causes of source bias remain unexplored. In this paper, we explain the process of information retrieval with a causal graph and discover that PLM-based retrievers learn perplexity features for relevance estimation, causing source bias by ranking the documents with low perplexity higher. Theoretical analysis further reveals that the phenomenon stems from the positive correlation between the gradients of the loss functions in language modeling task and retrieval task. Based on the analysis, a causal-inspired inference-time debiasing method is proposed, called Causal Diagnosis and Correction (CDC). CDC first diagnoses the bias effect of the perplexity and then separates the bias effect from the overall estimated relevance score. Experimental results across three domains demonstrate the superior debiasing effectiveness of CDC, emphasizing the validity of our proposed explanatory framework. Source codes are available at this https URL. 

---
# Exploring the Word Sense Disambiguation Capabilities of Large Language Models 

**Authors**: Pierpaolo Basile, Lucia Siciliani, Elio Musacchio, Giovanni Semeraro  

**Link**: [PDF](https://arxiv.org/pdf/2503.08662)  

**Abstract**: Word Sense Disambiguation (WSD) is a historical task in computational linguistics that has received much attention over the years. However, with the advent of Large Language Models (LLMs), interest in this task (in its classical definition) has decreased. In this study, we evaluate the performance of various LLMs on the WSD task. We extend a previous benchmark (XL-WSD) to re-design two subtasks suitable for LLM: 1) given a word in a sentence, the LLM must generate the correct definition; 2) given a word in a sentence and a set of predefined meanings, the LLM must select the correct one. The extended benchmark is built using the XL-WSD and BabelNet. The results indicate that LLMs perform well in zero-shot learning but cannot surpass current state-of-the-art methods. However, a fine-tuned model with a medium number of parameters outperforms all other models, including the state-of-the-art. 

---
# DAFE: LLM-Based Evaluation Through Dynamic Arbitration for Free-Form Question-Answering 

**Authors**: Sher Badshah, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2503.08542)  

**Abstract**: Evaluating Large Language Models (LLMs) free-form generated responses remains a challenge due to their diverse and open-ended nature. Traditional supervised signal-based automatic metrics fail to capture semantic equivalence or handle the variability of open-ended responses, while human evaluation, though reliable, is resource-intensive. Leveraging LLMs as evaluators offers a promising alternative due to their strong language understanding and instruction-following capabilities. Taking advantage of these capabilities, we propose the Dynamic Arbitration Framework for Evaluation (DAFE), which employs two primary LLM-as-judges and engages a third arbitrator only in cases of disagreements. This selective arbitration prioritizes evaluation reliability while reducing unnecessary computational demands compared to conventional majority voting. DAFE utilizes task-specific reference answers with dynamic arbitration to enhance judgment accuracy, resulting in significant improvements in evaluation metrics such as Macro F1 and Cohen's Kappa. Through experiments, including a comprehensive human evaluation, we demonstrate DAFE's ability to provide consistent, scalable, and resource-efficient assessments, establishing it as a robust framework for evaluating free-form model outputs. 

---
# EMMOE: A Comprehensive Benchmark for Embodied Mobile Manipulation in Open Environments 

**Authors**: Dongping Li, Tielong Cai, Tianci Tang, Wenhao Chai, Katherine Rose Driggs-Campbell, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08604)  

**Abstract**: Developing autonomous home robots controlled by natural language has long been a pursuit of human. While advancements in large language models (LLMs) and embodied intelligence make this goal closer, several challenges persist: the lack of a unified benchmark for more complex robot tasks, limited evaluation methods and metrics, data incompatibility between LLMs and mobile manipulation trajectories. To address these issues, we introduce Embodied Mobile Manipulation in Open Environments (EMMOE), which requires agents to interpret user instructions and execute long-horizon everyday tasks in continuous space. EMMOE seamlessly integrates high-level and low-level embodied tasks into a unified framework, along with three new metrics for more diverse assessment. Additionally, we collect EMMOE-100, which features in various task attributes, detailed process annotations, re-plans after failures, and two sub-datasets for LLM training. Furthermore, we design HomieBot, a sophisticated agent system consists of LLM with Direct Preference Optimization (DPO), light weighted navigation and manipulation models, and multiple error detection mechanisms. Finally, we demonstrate HomieBot's performance and the evaluation of different models and policies. 

---
# Mellow: a small audio language model for reasoning 

**Authors**: Soham Deshmukh, Satvik Dixit, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2503.08540)  

**Abstract**: Multimodal Audio-Language Models (ALMs) can understand and reason over both audio and text. Typically, reasoning performance correlates with model size, with the best results achieved by models exceeding 8 billion parameters. However, no prior work has explored enabling small audio-language models to perform reasoning tasks, despite the potential applications for edge devices. To address this gap, we introduce Mellow, a small Audio-Language Model specifically designed for reasoning. Mellow achieves state-of-the-art performance among existing small audio-language models and surpasses several larger models in reasoning capabilities. For instance, Mellow scores 52.11 on MMAU, comparable to SoTA Qwen2 Audio (which scores 52.5) while using 50 times fewer parameters and being trained on 60 times less data (audio hrs). To train Mellow, we introduce ReasonAQA, a dataset designed to enhance audio-grounded reasoning in models. It consists of a mixture of existing datasets (30% of the data) and synthetically generated data (70%). The synthetic dataset is derived from audio captioning datasets, where Large Language Models (LLMs) generate detailed and multiple-choice questions focusing on audio events, objects, acoustic scenes, signal properties, semantics, and listener emotions. To evaluate Mellow's reasoning ability, we benchmark it on a diverse set of tasks, assessing on both in-distribution and out-of-distribution data, including audio understanding, deductive reasoning, and comparative reasoning. Finally, we conduct extensive ablation studies to explore the impact of projection layer choices, synthetic data generation methods, and language model pretraining on reasoning performance. Our training dataset, findings, and baseline pave the way for developing small ALMs capable of reasoning. 

---
# General-Purpose Aerial Intelligent Agents Empowered by Large Language Models 

**Authors**: Ji Zhao, Xiao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.08302)  

**Abstract**: The emergence of large language models (LLMs) opens new frontiers for unmanned aerial vehicle (UAVs), yet existing systems remain confined to predefined tasks due to hardware-software co-design challenges. This paper presents the first aerial intelligent agent capable of open-world task execution through tight integration of LLM-based reasoning and robotic autonomy. Our hardware-software co-designed system addresses two fundamental limitations: (1) Onboard LLM operation via an edge-optimized computing platform, achieving 5-6 tokens/sec inference for 14B-parameter models at 220W peak power; (2) A bidirectional cognitive architecture that synergizes slow deliberative planning (LLM task planning) with fast reactive control (state estimation, mapping, obstacle avoidance, and motion planning). Validated through preliminary results using our prototype, the system demonstrates reliable task planning and scene understanding in communication-constrained environments, such as sugarcane monitoring, power grid inspection, mine tunnel exploration, and biological observation applications. This work establishes a novel framework for embodied aerial artificial intelligence, bridging the gap between task planning and robotic autonomy in open environments. 

---
# Large Language Model as Meta-Surrogate for Data-Driven Many-Task Optimization: A Proof-of-Principle Study 

**Authors**: Xian-Rong Zhang, Yue-Jiao Gong, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08301)  

**Abstract**: In many-task optimization scenarios, surrogate models are valuable for mitigating the computational burden of repeated fitness evaluations across tasks. This study proposes a novel meta-surrogate framework to assist many-task optimization, by leveraging the knowledge transfer strengths and emergent capabilities of large language models (LLMs). We formulate a unified framework for many-task fitness prediction, by defining a universal model with metadata to fit a group of problems. Fitness prediction is performed on metadata and decision variables, enabling efficient knowledge sharing across tasks and adaptability to new tasks. The LLM-based meta-surrogate treats fitness prediction as conditional probability estimation, employing a unified token sequence representation for task metadata, inputs, and outputs. This approach facilitates efficient inter-task knowledge sharing through shared token embeddings and captures complex task dependencies via multi-task model training. Experimental results demonstrate the model's emergent generalization ability, including zero-shot performance on problems with unseen dimensions. When integrated into evolutionary transfer optimization (ETO), our framework supports dual-level knowledge transfer -- at both the surrogate and individual levels -- enhancing optimization efficiency and robustness. This work establishes a novel foundation for applying LLMs in surrogate modeling, offering a versatile solution for many-task optimization. 

---
# Large Language Models for Outpatient Referral: Problem Definition, Benchmarking and Challenges 

**Authors**: Xiaoxiao Liu, Qingying Xiao, Junying Chen, Xiangyi Feng, Xiangbo Wu, Bairui Zhang, Xiang Wan, Jian Chang, Guangjun Yu, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08292)  

**Abstract**: Large language models (LLMs) are increasingly applied to outpatient referral tasks across healthcare systems. However, there is a lack of standardized evaluation criteria to assess their effectiveness, particularly in dynamic, interactive scenarios. In this study, we systematically examine the capabilities and limitations of LLMs in managing tasks within Intelligent Outpatient Referral (IOR) systems and propose a comprehensive evaluation framework specifically designed for such systems. This framework comprises two core tasks: static evaluation, which focuses on evaluating the ability of predefined outpatient referrals, and dynamic evaluation, which evaluates capabilities of refining outpatient referral recommendations through iterative dialogues. Our findings suggest that LLMs offer limited advantages over BERT-like models, but show promise in asking effective questions during interactive dialogues. 

---
# A Cascading Cooperative Multi-agent Framework for On-ramp Merging Control Integrating Large Language Models 

**Authors**: Miao Zhang, Zhenlong Fang, Tianyi Wang, Qian Zhang, Shuai Lu, Junfeng Jiao, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.08199)  

**Abstract**: Traditional Reinforcement Learning (RL) suffers from replicating human-like behaviors, generalizing effectively in multi-agent scenarios, and overcoming inherent interpretability this http URL tasks are compounded when deep environment understanding, agent coordination and dynamic optimization are required. While Large Language Model (LLM) enhanced methods have shown promise in generalization and interoperability, they often neglect necessary multi-agent coordination. Therefore, we introduce the Cascading Cooperative Multi-agent (CCMA) framework, integrating RL for individual interactions, a fine-tuned LLM for regional cooperation, a reward function for global optimization, and the Retrieval-augmented Generation mechanism to dynamically optimize decision-making across complex driving scenarios. Our experiments demonstrate that the CCMA outperforms existing RL methods, demonstrating significant improvements in both micro and macro-level performance in complex driving environments. 

---
# EgoBlind: Towards Egocentric Visual Assistance for the Blind People 

**Authors**: Junbin Xiao, Nanxin Huang, Hao Qiu, Zhulin Tao, Xun Yang, Richang Hong, Meng Wang, Angela Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08221)  

**Abstract**: We present EgoBlind, the first egocentric VideoQA dataset collected from blind individuals to evaluate the assistive capabilities of contemporary multimodal large language models (MLLMs). EgoBlind comprises 1,210 videos that record the daily lives of real blind users from a first-person perspective. It also features 4,927 questions directly posed or generated and verified by blind individuals to reflect their needs for visual assistance under various scenarios. We provide each question with an average of 3 reference answers to alleviate subjective evaluation. Using EgoBlind, we comprehensively evaluate 15 leading MLLMs and find that all models struggle, with the best performers achieving accuracy around 56\%, far behind human performance of 87.4\%. To guide future advancements, we identify and summarize major limitations of existing MLLMs in egocentric visual assistance for the blind and provide heuristic suggestions for improvement. With these efforts, we hope EgoBlind can serve as a valuable foundation for developing more effective AI assistants to enhance the independence of the blind individuals' lives. 

---
# Instruction-Augmented Long-Horizon Planning: Embedding Grounding Mechanisms in Embodied Mobile Manipulation 

**Authors**: Fangyuan Wang, Shipeng Lyu, Peng Zhou, Anqing Duan, Guodong Guo, David Navarro-Alarcon  

**Link**: [PDF](https://arxiv.org/pdf/2503.08084)  

**Abstract**: Enabling humanoid robots to perform long-horizon mobile manipulation planning in real-world environments based on embodied perception and comprehension abilities has been a longstanding challenge. With the recent rise of large language models (LLMs), there has been a notable increase in the development of LLM-based planners. These approaches either utilize human-provided textual representations of the real world or heavily depend on prompt engineering to extract such representations, lacking the capability to quantitatively understand the environment, such as determining the feasibility of manipulating objects. To address these limitations, we present the Instruction-Augmented Long-Horizon Planning (IALP) system, a novel framework that employs LLMs to generate feasible and optimal actions based on real-time sensor feedback, including grounded knowledge of the environment, in a closed-loop interaction. Distinct from prior works, our approach augments user instructions into PDDL problems by leveraging both the abstract reasoning capabilities of LLMs and grounding mechanisms. By conducting various real-world long-horizon tasks, each consisting of seven distinct manipulatory skills, our results demonstrate that the IALP system can efficiently solve these tasks with an average success rate exceeding 80%. Our proposed method can operate as a high-level planner, equipping robots with substantial autonomy in unstructured environments through the utilization of multi-modal sensor inputs. 

---
# Investigating the Effectiveness of a Socratic Chain-of-Thoughts Reasoning Method for Task Planning in Robotics, A Case Study 

**Authors**: Veronica Bot, Zheyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08174)  

**Abstract**: Large language models (LLMs) have demonstrated unprecedented capability in reasoning with natural language. Coupled with this development is the emergence of embodied AI in robotics. Despite showing promise for verbal and written reasoning tasks, it remains unknown whether LLMs are capable of navigating complex spatial tasks with physical actions in the real world. To this end, it is of interest to investigate applying LLMs to robotics in zero-shot learning scenarios, and in the absence of fine-tuning - a feat which could significantly improve human-robot interaction, alleviate compute cost, and eliminate low-level programming tasks associated with robot tasks.
To explore this question, we apply GPT-4(Omni) with a simulated Tiago robot in Webots engine for an object search task. We evaluate the effectiveness of three reasoning strategies based on Chain-of-Thought (CoT) sub-task list generation with the Socratic method (SocraCoT) (in order of increasing rigor): (1) Non-CoT/Non-SocraCoT, (2) CoT only, and (3) SocraCoT. Performance was measured in terms of the proportion of tasks successfully completed and execution time (N = 20). Our preliminary results show that when combined with chain-of-thought reasoning, the Socratic method can be used for code generation for robotic tasks that require spatial awareness. In extension of this finding, we propose EVINCE-LoC; a modified EVINCE method that could further enhance performance in highly complex and or dynamic testing scenarios. 

---
# In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents 

**Authors**: Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2503.08026)  

**Abstract**: Large Language Models (LLMs) have made significant progress in open-ended dialogue, yet their inability to retain and retrieve relevant information from long-term interactions limits their effectiveness in applications requiring sustained personalization. External memory mechanisms have been proposed to address this limitation, enabling LLMs to maintain conversational continuity. However, existing approaches struggle with two key challenges. First, rigid memory granularity fails to capture the natural semantic structure of conversations, leading to fragmented and incomplete representations. Second, fixed retrieval mechanisms cannot adapt to diverse dialogue contexts and user interaction patterns. In this work, we propose Reflective Memory Management (RMM), a novel mechanism for long-term dialogue agents, integrating forward- and backward-looking reflections: (1) Prospective Reflection, which dynamically summarizes interactions across granularities-utterances, turns, and sessions-into a personalized memory bank for effective future retrieval, and (2) Retrospective Reflection, which iteratively refines the retrieval in an online reinforcement learning (RL) manner based on LLMs' cited evidence. Experiments show that RMM demonstrates consistent improvement across various metrics and benchmarks. For example, RMM shows more than 10% accuracy improvement over the baseline without memory management on the LongMemEval dataset. 

---
# ProTeX: Structure-In-Context Reasoning and Editing of Proteins with Large Language Models 

**Authors**: Zicheng Ma, Chuanliu Fan, Zhicong Wang, Zhenyu Chen, Xiaohan Lin, Yanheng Li, Shihao Feng, Jun Zhang, Ziqiang Cao, Yi Qin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08179)  

**Abstract**: Large language models have made remarkable progress in the field of molecular science, particularly in understanding and generating functional small molecules. This success is largely attributed to the effectiveness of molecular tokenization strategies. In protein science, the amino acid sequence serves as the sole tokenizer for LLMs. However, many fundamental challenges in protein science are inherently structure-dependent. The absence of structure-aware tokens significantly limits the capabilities of LLMs for comprehensive biomolecular comprehension and multimodal generation. To address these challenges, we introduce a novel framework, ProTeX, which tokenizes the protein sequences, structures, and textual information into a unified discrete space. This innovative approach enables joint training of the LLM exclusively through the Next-Token Prediction paradigm, facilitating multimodal protein reasoning and generation. ProTeX enables general LLMs to perceive and process protein structures through sequential text input, leverage structural information as intermediate reasoning components, and generate or manipulate structures via sequential text output. Experiments demonstrate that our model achieves significant improvements in protein function prediction, outperforming the state-of-the-art domain expert model with a twofold increase in accuracy. Our framework enables high-quality conformational generation and customizable protein design. For the first time, we demonstrate that by adopting the standard training and inference pipelines from the LLM domain, ProTeX empowers decoder-only LLMs to effectively address diverse spectrum of protein-related tasks. 

---
# Gemini Embedding: Generalizable Embeddings from Gemini 

**Authors**: Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, Xiaoqi Ren, Shanfeng Zhang, Daniel Salz, Michael Boratko, Jay Han, Blair Chen, Shuo Huang, Vikram Rao, Paul Suganthan, Feng Han, Andreas Doumanoglou, Nithi Gupta, Fedor Moiseev, Cathy Yip, Aashi Jain, Simon Baumgartner, Shahrokh Shahi, Frank Palma Gomez, Sandeep Mariserla, Min Choi, Parashar Shah, Sonam Goenka, Ke Chen, Ye Xia, Koert Chen, Sai Meher Karthik Duddu, Yichang Chen, Trevor Walker, Wenlei Zhou, Rakesh Ghiya, Zach Gleicher, Karan Gill, Zhe Dong, Mojtaba Seyedhosseini, Yunhsuan Sung, Raphael Hoffmann, Tom Duerig  

**Link**: [PDF](https://arxiv.org/pdf/2503.07891)  

**Abstract**: In this report, we introduce Gemini Embedding, a state-of-the-art embedding model leveraging the power of Gemini, Google's most capable large language model. Capitalizing on Gemini's inherent multilingual and code understanding capabilities, Gemini Embedding produces highly generalizable embeddings for text spanning numerous languages and textual modalities. The representations generated by Gemini Embedding can be precomputed and applied to a variety of downstream tasks including classification, similarity, clustering, ranking, and retrieval. Evaluated on the Massive Multilingual Text Embedding Benchmark (MMTEB), which includes over one hundred tasks across 250+ languages, Gemini Embedding substantially outperforms prior state-of-the-art models, demonstrating considerable improvements in embedding quality. Achieving state-of-the-art performance across MMTEB's multilingual, English, and code benchmarks, our unified model demonstrates strong capabilities across a broad selection of tasks and surpasses specialized domain-specific models. 

---
# Safety Guardrails for LLM-Enabled Robots 

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2503.07885)  

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots. 

---
# LLMIdxAdvis: Resource-Efficient Index Advisor Utilizing Large Language Model 

**Authors**: Xinxin Zhao, Haoyang Li, Jing Zhang, Xinmei Huang, Tieying Zhang, Jianjun Chen, Rui Shi, Cuiping Li, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07884)  

**Abstract**: Index recommendation is essential for improving query performance in database management systems (DBMSs) through creating an optimal set of indexes under specific constraints. Traditional methods, such as heuristic and learning-based approaches, are effective but face challenges like lengthy recommendation time, resource-intensive training, and poor generalization across different workloads and database schemas. To address these issues, we propose LLMIdxAdvis, a resource-efficient index advisor that uses large language models (LLMs) without extensive fine-tuning. LLMIdxAdvis frames index recommendation as a sequence-to-sequence task, taking target workload, storage constraint, and corresponding database environment as input, and directly outputting recommended indexes. It constructs a high-quality demonstration pool offline, using GPT-4-Turbo to synthesize diverse SQL queries and applying integrated heuristic methods to collect both default and refined labels. During recommendation, these demonstrations are ranked to inject database expertise via in-context learning. Additionally, LLMIdxAdvis extracts workload features involving specific column statistical information to strengthen LLM's understanding, and introduces a novel inference scaling strategy combining vertical scaling (via ''Index-Guided Major Voting'' and Best-of-N) and horizontal scaling (through iterative ''self-optimization'' with database feedback) to enhance reliability. Experiments on 3 OLAP and 2 real-world benchmarks reveal that LLMIdxAdvis delivers competitive index recommendation with reduced runtime, and generalizes effectively across different workloads and database schemas. 

---
# HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations 

**Authors**: Samir Abdaljalil, Hasan Kurban, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07833)  

**Abstract**: Large Language Models (LLMs) are increasingly used in various contexts, yet remain prone to generating non-factual content, commonly referred to as "hallucinations". The literature categorizes hallucinations into several types, including entity-level, relation-level, and sentence-level hallucinations. However, existing hallucination datasets often fail to capture fine-grained hallucinations in multilingual settings. In this work, we introduce HalluVerse25, a multilingual LLM hallucination dataset that categorizes fine-grained hallucinations in English, Arabic, and Turkish. Our dataset construction pipeline uses an LLM to inject hallucinations into factual biographical sentences, followed by a rigorous human annotation process to ensure data quality. We evaluate several LLMs on HalluVerse25, providing valuable insights into how proprietary models perform in detecting LLM-generated hallucinations across different contexts. 

---
# EFPC: Towards Efficient and Flexible Prompt Compression 

**Authors**: Yun-Hao Cao, Yangsong Wang, Shuzheng Hao, Zhenxing Li, Chengjun Zhan, Sichao Liu, Yi-Qi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07956)  

**Abstract**: The emergence of large language models (LLMs) like GPT-4 has revolutionized natural language processing (NLP), enabling diverse, complex tasks. However, extensive token counts lead to high computational and financial burdens. To address this, we propose Efficient and Flexible Prompt Compression (EFPC), a novel method unifying task-aware and task-agnostic compression for a favorable accuracy-efficiency trade-off. EFPC uses GPT-4 to generate compressed prompts and integrates them with original prompts for training. During training and inference, we selectively prepend user instructions and compress prompts based on predicted probabilities. EFPC is highly data-efficient, achieving significant performance with minimal data. Compared to the state-of-the-art method LLMLingua-2, EFPC achieves a 4.8% relative improvement in F1-score with 1% additional data at a 4x compression rate, and an 11.4% gain with 10% additional data on the LongBench single-doc QA benchmark. EFPC's unified framework supports broad applicability and enhances performance across various models, tasks, and domains, offering a practical advancement in NLP. 

---
# Training Domain Draft Models for Speculative Decoding: Best Practices and Insights 

**Authors**: Fenglu Hong, Ravi Raju, Jonathan Lingjie Li, Bo Li, Urmish Thakker, Avinash Ravichandran, Swayambhoo Jain, Changran Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07807)  

**Abstract**: Speculative decoding is an effective method for accelerating inference of large language models (LLMs) by employing a small draft model to predict the output of a target model. However, when adapting speculative decoding to domain-specific target models, the acceptance rate of the generic draft model drops significantly due to domain shift. In this work, we systematically investigate knowledge distillation techniques for training domain draft models to improve their speculation accuracy. We compare white-box and black-box distillation approaches and explore their effectiveness in various data accessibility scenarios, including historical user queries, curated domain data, and synthetically generated alignment data. Our experiments across Function Calling, Biology, and Chinese domains show that offline distillation consistently outperforms online distillation by 11% to 25%, white-box distillation surpasses black-box distillation by 2% to 10%, and data scaling trends hold across domains. Additionally, we find that synthetic data can effectively align draft models and achieve 80% to 93% of the performance of training on historical user queries. These findings provide practical guidelines for training domain-specific draft models to improve speculative decoding efficiency. 

---
# MapQA: Open-domain Geospatial Question Answering on Map Data 

**Authors**: Zekun Li, Malcolm Grossman, Eric, Qasemi, Mihir Kulkarni, Muhao Chen, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07871)  

**Abstract**: Geospatial question answering (QA) is a fundamental task in navigation and point of interest (POI) searches. While existing geospatial QA datasets exist, they are limited in both scale and diversity, often relying solely on textual descriptions of geo-entities without considering their geometries. A major challenge in scaling geospatial QA datasets for reasoning lies in the complexity of geospatial relationships, which require integrating spatial structures, topological dependencies, and multi-hop reasoning capabilities that most text-based QA datasets lack. To address these limitations, we introduce MapQA, a novel dataset that not only provides question-answer pairs but also includes the geometries of geo-entities referenced in the questions. MapQA is constructed using SQL query templates to extract question-answer pairs from OpenStreetMap (OSM) for two study regions: Southern California and Illinois. It consists of 3,154 QA pairs spanning nine question types that require geospatial reasoning, such as neighborhood inference and geo-entity type identification. Compared to existing datasets, MapQA expands both the number and diversity of geospatial question types. We explore two approaches to tackle this challenge: (1) a retrieval-based language model that ranks candidate geo-entities by embedding similarity, and (2) a large language model (LLM) that generates SQL queries from natural language questions and geo-entity attributes, which are then executed against an OSM database. Our findings indicate that retrieval-based methods effectively capture concepts like closeness and direction but struggle with questions that require explicit computations (e.g., distance calculations). LLMs (e.g., GPT and Gemini) excel at generating SQL queries for one-hop reasoning but face challenges with multi-hop reasoning, highlighting a key bottleneck in advancing geospatial QA systems. 

---
# Evaluating LLaMA 3.2 for Software Vulnerability Detection 

**Authors**: José Gonçalves, Miguel Silva, Bernardo Cabral, Tiago Dias, Eva Maia, Isabel Praça, Ricardo Severino, Luís Lino Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2503.07770)  

**Abstract**: Deep Learning (DL) has emerged as a powerful tool for vulnerability detection, often outperforming traditional solutions. However, developing effective DL models requires large amounts of real-world data, which can be difficult to obtain in sufficient quantities. To address this challenge, DiverseVul dataset has been curated as the largest dataset of vulnerable and non-vulnerable C/C++ functions extracted exclusively from real-world projects. Its goal is to provide high-quality, large-scale samples for training DL models. However, during our study several inconsistencies were identified in the raw dataset while applying pre-processing techniques, highlighting the need for a refined version. In this work, we present a refined version of DiverseVul dataset, which is used to fine-tune a large language model, LLaMA 3.2, for vulnerability detection. Experimental results show that the use of pre-processing techniques led to an improvement in performance, with the model achieving an F1-Score of 66%, a competitive result when compared to our baseline, which achieved a 47% F1-Score in software vulnerability detection. 

---
# A Time Series Multitask Framework Integrating a Large Language Model, Pre-Trained Time Series Model, and Knowledge Graph 

**Authors**: Shule Hao, Junpeng Bao, Chuncheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07682)  

**Abstract**: Time series analysis is crucial in fields like finance, transportation, and industry. However, traditional models often focus solely on temporal features, limiting their ability to capture underlying information. This paper proposes a novel time series multitask framework, called LTM, which integrates temporal features with textual descriptions to enhance analytical and predictive capabilities. LTM combines pre-trained time series model, large language model (LLM), and knowledge graph to tackle time series tasks, including forecasting, imputation, and anomaly detection. LTM achieves improved performance with a few trainable parameters. It is very efficient and practical. LTM encodes time series data into patches and enriches user-provided prompts using knowledge graphs to generate enhanced prompts. A novel feature fusion method embeds prompts into each patch encoding, which is processed by a frozen LLM, followed by a feature enhancement module and a time decoder module. During fine-tuning stage, cosine similarity between prompts and temporal patches is integrated into the loss function to boost performance. Experiments on benchmark datasets show that LTM significantly outperforms existing methods. It provides a robust and versatile solution for time series tasks. 

---
# Psychological Counseling Ability of Large Language Models 

**Authors**: Fangyu Peng, Jingxin Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.07627)  

**Abstract**: With the development of science and the continuous progress of artificial intelligence technology, Large Language Models (LLMs) have begun to be widely utilized across various fields. However, in the field of psychological counseling, the ability of LLMs have not been systematically assessed. In this study, we assessed the psychological counseling ability of mainstream LLMs using 1096 psychological counseling skill questions which were selected from the Chinese National Counselor Level 3 Examination, including Knowledge-based, Analytical-based, and Application-based question types. The analysis showed that the correctness rates of the LLMs for Chinese questions, in descending order, were GLM-3 (46.5%), GPT-4 (46.1%), Gemini (45.0%), ERNIE-3.5 (45.7%) and GPT-3.5 (32.9%). The correctness rates of the LLMs for English questions, in descending order, were ERNIE-3.5 (43.9%), GPT-4 (40.6%), Gemini (36.6%), GLM-3 (29.9%) and GPT-3.5 (29.5%). A chi-square test indicated significant differences in the LLMs' performance on Chinese and English questions. Furthermore, we subsequently utilized the Counselor's Guidebook (Level 3) as a reference for ERNIE-3.5, resulting in a new correctness rate of 59.6%, a 13.8% improvement over its initial rate of 45.8%. In conclusion, the study assessed the psychological counseling ability of LLMs for the first time, which may provide insights for future enhancement and improvement of psychological counseling ability of LLMs. 

---
# Self-Taught Self-Correction for Small Language Models 

**Authors**: Viktor Moskvoretskii, Chris Biemann, Irina Nikishina  

**Link**: [PDF](https://arxiv.org/pdf/2503.08681)  

**Abstract**: Although large language models (LLMs) have achieved remarkable performance across various tasks, they remain prone to errors. A key challenge is enabling them to self-correct. While prior research has relied on external tools or large proprietary models, this work explores self-correction in small language models (SLMs) through iterative fine-tuning using solely self-generated data. We introduce the Self-Taught Self-Correction (STaSC) algorithm, which incorporates multiple algorithmic design choices. Experimental results on a question-answering task demonstrate that STaSC effectively learns self-correction, leading to significant performance improvements. Our analysis further provides insights into the mechanisms of self-correction and the impact of different design choices on learning dynamics and overall performance. To support future research, we release our user-friendly codebase and lightweight models. 

---
# Fact-checking with Generative AI: A Systematic Cross-Topic Examination of LLMs Capacity to Detect Veracity of Political Information 

**Authors**: Elizaveta Kuznetsova, Ilaria Vitulano, Mykola Makhortykh, Martha Stolze, Tomas Nagy, Victoria Vziatysheva  

**Link**: [PDF](https://arxiv.org/pdf/2503.08404)  

**Abstract**: The purpose of this study is to assess how large language models (LLMs) can be used for fact-checking and contribute to the broader debate on the use of automated means for veracity identification. To achieve this purpose, we use AI auditing methodology that systematically evaluates performance of five LLMs (ChatGPT 4, Llama 3 (70B), Llama 3.1 (405B), Claude 3.5 Sonnet, and Google Gemini) using prompts regarding a large set of statements fact-checked by professional journalists (16,513). Specifically, we use topic modeling and regression analysis to investigate which factors (e.g. topic of the prompt or the LLM type) affect evaluations of true, false, and mixed statements. Our findings reveal that while ChatGPT 4 and Google Gemini achieved higher accuracy than other models, overall performance across models remains modest. Notably, the results indicate that models are better at identifying false statements, especially on sensitive topics such as COVID-19, American political controversies, and social issues, suggesting possible guardrails that may enhance accuracy on these topics. The major implication of our findings is that there are significant challenges for using LLMs for factchecking, including significant variation in performance across different LLMs and unequal quality of outputs for specific topics which can be attributed to deficits of training data. Our research highlights the potential and limitations of LLMs in political fact-checking, suggesting potential avenues for further improvements in guardrails as well as fine-tuning. 

---
# ReviewAgents: Bridging the Gap Between Human and AI-Generated Paper Reviews 

**Authors**: Xian Gao, Jiacheng Ruan, Jingsheng Gao, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08506)  

**Abstract**: Academic paper review is a critical yet time-consuming task within the research community. With the increasing volume of academic publications, automating the review process has become a significant challenge. The primary issue lies in generating comprehensive, accurate, and reasoning-consistent review comments that align with human reviewers' judgments. In this paper, we address this challenge by proposing ReviewAgents, a framework that leverages large language models (LLMs) to generate academic paper reviews. We first introduce a novel dataset, Review-CoT, consisting of 142k review comments, designed for training LLM agents. This dataset emulates the structured reasoning process of human reviewers-summarizing the paper, referencing relevant works, identifying strengths and weaknesses, and generating a review conclusion. Building upon this, we train LLM reviewer agents capable of structured reasoning using a relevant-paper-aware training method. Furthermore, we construct ReviewAgents, a multi-role, multi-LLM agent review framework, to enhance the review comment generation process. Additionally, we propose ReviewBench, a benchmark for evaluating the review comments generated by LLMs. Our experimental results on ReviewBench demonstrate that while existing LLMs exhibit a certain degree of potential for automating the review process, there remains a gap when compared to human-generated reviews. Moreover, our ReviewAgents framework further narrows this gap, outperforming advanced LLMs in generating review comments. 

---
# Enhancing Multi-Hop Fact Verification with Structured Knowledge-Augmented Large Language Models 

**Authors**: Han Cao, Lingwei Wei, Wei Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08495)  

**Abstract**: The rapid development of social platforms exacerbates the dissemination of misinformation, which stimulates the research in fact verification. Recent studies tend to leverage semantic features to solve this problem as a single-hop task. However, the process of verifying a claim requires several pieces of evidence with complicated inner logic and relations to verify the given claim in real-world situations. Recent studies attempt to improve both understanding and reasoning abilities to enhance the performance, but they overlook the crucial relations between entities that benefit models to understand better and facilitate the prediction. To emphasize the significance of relations, we resort to Large Language Models (LLMs) considering their excellent understanding ability. Instead of other methods using LLMs as the predictor, we take them as relation extractors, for they do better in understanding rather than reasoning according to the experimental results. Thus, to solve the challenges above, we propose a novel Structured Knowledge-Augmented LLM-based Network (LLM-SKAN) for multi-hop fact verification. Specifically, we utilize an LLM-driven Knowledge Extractor to capture fine-grained information, including entities and their complicated relations. Besides, we leverage a Knowledge-Augmented Relation Graph Fusion module to interact with each node and learn better claim-evidence representations comprehensively. The experimental results on four common-used datasets demonstrate the effectiveness and superiority of our model. 

---
# Towards Scalable and Cross-Lingual Specialist Language Models for Oncology 

**Authors**: Morteza Rohanian, Tarun Mehra, Nicola Miglino, Farhad Nooralahzadeh, Michael Krauthammer, Andreas Wicki  

**Link**: [PDF](https://arxiv.org/pdf/2503.08323)  

**Abstract**: Clinical oncology generates vast, unstructured data that often contain inconsistencies, missing information, and ambiguities, making it difficult to extract reliable insights for data-driven decision-making. General-purpose large language models (LLMs) struggle with these challenges due to their lack of domain-specific reasoning, including specialized clinical terminology, context-dependent interpretations, and multi-modal data integration. We address these issues with an oncology-specialized, efficient, and adaptable NLP framework that combines instruction tuning, retrieval-augmented generation (RAG), and graph-based knowledge integration. Our lightweight models prove effective at oncology-specific tasks, such as named entity recognition (e.g., identifying cancer diagnoses), entity linking (e.g., linking entities to standardized ontologies), TNM staging, document classification (e.g., cancer subtype classification from pathology reports), and treatment response prediction. Our framework emphasizes adaptability and resource efficiency. We include minimal German instructions, collected at the University Hospital Zurich (USZ), to test whether small amounts of non-English language data can effectively transfer knowledge across languages. This approach mirrors our motivation for lightweight models, which balance strong performance with reduced computational costs, making them suitable for resource-limited healthcare settings. We validated our models on oncology datasets, demonstrating strong results in named entity recognition, relation extraction, and document classification. 

---
# DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process 

**Authors**: Minjun Zhu, Yixuan Weng, Linyi Yang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08569)  

**Abstract**: Large Language Models (LLMs) are increasingly utilized in scientific research assessment, particularly in automated paper review. However, existing LLM-based review systems face significant challenges, including limited domain expertise, hallucinated reasoning, and a lack of structured evaluation. To address these limitations, we introduce DeepReview, a multi-stage framework designed to emulate expert reviewers by incorporating structured analysis, literature retrieval, and evidence-based argumentation. Using DeepReview-13K, a curated dataset with structured annotations, we train DeepReviewer-14B, which outperforms CycleReviewer-70B with fewer tokens. In its best mode, DeepReviewer-14B achieves win rates of 88.21\% and 80.20\% against GPT-o1 and DeepSeek-R1 in evaluations. Our work sets a new benchmark for LLM-based paper review, with all resources publicly available. The code, model, dataset and demo have be released in this http URL. 

---
# Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation 

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08195)  

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness. 

---
# Automating Violence Detection and Categorization from Ancient Texts 

**Authors**: Alhassan Abdelhalim, Michaela Regneri  

**Link**: [PDF](https://arxiv.org/pdf/2503.08192)  

**Abstract**: Violence descriptions in literature offer valuable insights for a wide range of research in the humanities. For historians, depictions of violence are of special interest for analyzing the societal dynamics surrounding large wars and individual conflicts of influential people. Harvesting data for violence research manually is laborious and time-consuming. This study is the first one to evaluate the effectiveness of large language models (LLMs) in identifying violence in ancient texts and categorizing it across multiple dimensions. Our experiments identify LLMs as a valuable tool to scale up the accurate analysis of historical texts and show the effect of fine-tuning and data augmentation, yielding an F1-score of up to 0.93 for violence detection and 0.86 for fine-grained violence categorization. 

---
# Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation 

**Authors**: Wen Luo, Feifan Song, Wei Li, Guangyue Peng, Shaohang Wei, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08057)  

**Abstract**: Large Language Models (LLMs) are increasingly required to generate text that is both factually accurate and diverse across various open-ended applications. However, current stochastic decoding methods struggle to balance such objectives. We introduce Dynamic Focus Decoding (DFD), a novel plug-and-play stochastic approach that resolves this trade-off without requiring additional data, knowledge, or models. DFD adaptively adjusts the decoding focus based on distributional differences across layers, leveraging the modular and hierarchical nature of factual knowledge within LLMs. This dynamic adjustment improves factuality in knowledge-intensive decoding steps and promotes diversity in less knowledge-reliant steps. DFD can be easily integrated with existing decoding methods, enhancing both factuality and diversity with minimal computational overhead. Extensive experiments across seven datasets demonstrate that DFD significantly improves performance, providing a scalable and efficient solution for open-ended text generation. 

---
# OASIS: Order-Augmented Strategy for Improved Code Search 

**Authors**: Zuchen Gao, Zizheng Zhan, Xianming Li, Erxin Yu, Haotian Zhang, Yuqun Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.08161)  

**Abstract**: Code embeddings capture the semantic representations of code and are crucial for various code-related large language model (LLM) applications, such as code search. Previous training primarily relies on optimizing the InfoNCE loss by comparing positive natural language (NL)-code pairs with in-batch negatives. However, due to the sparse nature of code contexts, training solely by comparing the major differences between positive and negative pairs may fail to capture deeper semantic nuances. To address this issue, we propose a novel order-augmented strategy for improved code search (OASIS). It leverages order-based similarity labels to train models to capture subtle differences in similarity among negative pairs. Extensive benchmark evaluations demonstrate that our OASIS model significantly outperforms previous state-of-the-art models focusing solely on major positive-negative differences. It underscores the value of exploiting subtle differences among negative pairs with order labels for effective code embedding training. 

---
# Learning to Search Effective Example Sequences for In-Context Learning 

**Authors**: Xiang Gao, Ankita Sinha, Kamalika Das  

**Link**: [PDF](https://arxiv.org/pdf/2503.08030)  

**Abstract**: Large language models (LLMs) demonstrate impressive few-shot learning capabilities, but their performance varies widely based on the sequence of in-context examples. Key factors influencing this include the sequence's length, composition, and arrangement, as well as its relation to the specific query. Existing methods often tackle these factors in isolation, overlooking their interdependencies. Moreover, the extensive search space for selecting optimal sequences complicates the development of a holistic approach. In this work, we introduce Beam Search-based Example Sequence Constructor (BESC), a novel method for learning to construct optimal example sequences. BESC addresses all key factors involved in sequence selection by considering them jointly during inference, while incrementally building the sequence. This design enables the use of beam search to significantly reduce the complexity of the search space. Experiments across various datasets and language models show notable improvements in performance. 

---
# Can Memory-Augmented Language Models Generalize on Reasoning-in-a-Haystack Tasks? 

**Authors**: Payel Das, Ching-Yun Ko, Sihui Dai, Georgios Kollias, Subhajit Chaudhury, Aurelie Lozano  

**Link**: [PDF](https://arxiv.org/pdf/2503.07903)  

**Abstract**: Large language models often expose their brittleness in reasoning tasks, especially while executing long chains of reasoning over context. We propose MemReasoner, a new and simple memory-augmented LLM architecture, in which the memory learns the relative order of facts in context, and enables hopping over them, while the decoder selectively attends to the memory. MemReasoner is trained end-to-end, with optional supporting fact supervision of varying degrees. We train MemReasoner, along with existing memory-augmented transformer models and a state-space model, on two distinct synthetic multi-hop reasoning tasks. Experiments performed under a variety of challenging scenarios, including the presence of long distractor text or target answer changes in test set, show strong generalization of MemReasoner on both single- and two-hop tasks. This generalization of MemReasoner is achieved using none-to-weak supporting fact supervision (using none and 1\% of supporting facts for one- and two-hop tasks, respectively). In contrast, baseline models overall struggle to generalize and benefit far less from using full supporting fact supervision. The results highlight the importance of explicit memory mechanisms, combined with additional weak supervision, for improving large language model's context processing ability toward reasoning tasks. 

---
# Modern Models, Medieval Texts: A POS Tagging Study of Old Occitan 

**Authors**: Matthias Schöffel, Marinus Wiedner, Esteban Garces Arias, Paula Ruppert, Christian Heumann, Matthias Aßenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2503.07827)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in natural language processing, yet their effectiveness in handling historical languages remains largely unexplored. This study examines the performance of open-source LLMs in part-of-speech (POS) tagging for Old Occitan, a historical language characterized by non-standardized orthography and significant diachronic variation. Through comparative analysis of two distinct corpora-hagiographical and medical texts-we evaluate how current models handle the inherent challenges of processing a low-resource historical language. Our findings demonstrate critical limitations in LLM performance when confronted with extreme orthographic and syntactic variability. We provide detailed error analysis and specific recommendations for improving model performance in historical language processing. This research advances our understanding of LLM capabilities in challenging linguistic contexts while offering practical insights for both computational linguistics and historical language studies. 

---
# JurisTCU: A Brazilian Portuguese Information Retrieval Dataset with Query Relevance Judgments 

**Authors**: Leandro Carísio Fernandes, Leandro dos Santos Ribeiro, Marcos Vinícius Borela de Castro, Leonardo Augusto da Silva Pacheco, Edans Flávius de Oliveira Sandes  

**Link**: [PDF](https://arxiv.org/pdf/2503.08379)  

**Abstract**: This paper introduces JurisTCU, a Brazilian Portuguese dataset for legal information retrieval (LIR). The dataset is freely available and consists of 16,045 jurisprudential documents from the Brazilian Federal Court of Accounts, along with 150 queries annotated with relevance judgments. It addresses the scarcity of Portuguese-language LIR datasets with query relevance annotations. The queries are organized into three groups: real user keyword-based queries, synthetic keyword-based queries, and synthetic question-based queries. Relevance judgments were produced through a hybrid approach combining LLM-based scoring with expert domain validation. We used JurisTCU in 14 experiments using lexical search (document expansion methods) and semantic search (BERT-based and OpenAI embeddings). We show that the document expansion methods significantly improve the performance of standard BM25 search on this dataset, with improvements exceeding 45% in P@10, R@10, and nDCG@10 metrics when evaluating short keyword-based queries. Among the embedding models, the OpenAI models produced the best results, with improvements of approximately 70% in P@10, R@10, and nDCG@10 metrics for short keyword-based queries, suggesting that these dense embeddings capture semantic relationships in this domain, surpassing the reliance on lexical terms. Besides offering a dataset for the Portuguese-language IR research community, suitable for evaluating search systems, the results also contribute to enhancing a search system highly relevant to Brazilian citizens. 

---
# Position-Aware Depth Decay Decoding ($D^3$): Boosting Large Language Model Inference Efficiency 

**Authors**: Siqi Fan, Xuezhi Fang, Xingrun Xing, Peng Han, Shuo Shang, Yequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08524)  

**Abstract**: Due to the large number of parameters, the inference phase of Large Language Models (LLMs) is resource-intensive. Unlike traditional model compression, which needs retraining, recent dynamic computation methods show that not all components are required for inference, enabling a training-free pipeline. In this paper, we focus on the dynamic depth of LLM generation. A token-position aware layer skipping framework is proposed to save 1.5x times operations efficiently while maintaining performance. We first observed that tokens predicted later have lower perplexity and thus require less computation. Then, we propose a training-free algorithm called Position-Aware Depth Decay Decoding ($D^3$), which leverages a power-law decay function, $\left\lfloor L \times (\alpha^i) \right\rfloor$, to determine the number of layers to retain when generating token $T_i$. Remarkably, without any retraining, the $D^3$ achieves success across a wide range of generation tasks for the first time. Experiments on large language models (\ie the Llama) with $7 \sim 70$ billion parameters show that $D^3$ can achieve an average 1.5x speedup compared with the full-inference pipeline while maintaining comparable performance with nearly no performance drop ($<1\%$) on the GSM8K and BBH benchmarks. 

---
# Adapting Large Language Models for Parameter-Efficient Log Anomaly Detection 

**Authors**: Ying Fu Lim, Jiawen Zhu, Guansong Pang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08045)  

**Abstract**: Log Anomaly Detection (LAD) seeks to identify atypical patterns in log data that are crucial to assessing the security and condition of systems. Although Large Language Models (LLMs) have shown tremendous success in various fields, the use of LLMs in enabling the detection of log anomalies is largely unexplored. This work aims to fill this gap. Due to the prohibitive costs involved in fully fine-tuning LLMs, we explore the use of parameter-efficient fine-tuning techniques (PEFTs) for adapting LLMs to LAD. To have an in-depth exploration of the potential of LLM-driven LAD, we present a comprehensive investigation of leveraging two of the most popular PEFTs -- Low-Rank Adaptation (LoRA) and Representation Fine-tuning (ReFT) -- to tap into three prominent LLMs of varying size, including RoBERTa, GPT-2, and Llama-3, for parameter-efficient LAD. Comprehensive experiments on four public log datasets are performed to reveal important insights into effective LLM-driven LAD in several key perspectives, including the efficacy of these PEFT-based LLM-driven LAD methods, their stability, sample efficiency, robustness w.r.t. unstable logs, and cross-dataset generalization. Code is available at this https URL. 

---
# Group Preference Alignment: Customized LLM Response Generation from In-Situ Conversations 

**Authors**: Ishani Mondal, Jack W. Stokes, Sujay Kumar Jauhar, Longqi Yang, Mengting Wan, Xiaofeng Xu, Xia Song, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2503.08035)  

**Abstract**: LLMs often fail to meet the specialized needs of distinct user groups due to their one-size-fits-all training paradigm \cite{lucy-etal-2024-one} and there is limited research on what personalization aspects each group expect. To address these limitations, we propose a group-aware personalization framework, Group Preference Alignment (GPA), that identifies context-specific variations in conversational preferences across user groups and then steers LLMs to address those preferences. Our approach consists of two steps: (1) Group-Aware Preference Extraction, where maximally divergent user-group preferences are extracted from real-world conversation logs and distilled into interpretable rubrics, and (2) Tailored Response Generation, which leverages these rubrics through two methods: a) Context-Tuned Inference (GAP-CT), that dynamically adjusts responses via context-dependent prompt instructions, and b) Rubric-Finetuning Inference (GPA-FT), which uses the rubrics to generate contrastive synthetic data for personalization of group-specific models via alignment. Experiments demonstrate that our framework significantly improves alignment of the output with respect to user preferences and outperforms baseline methods, while maintaining robust performance on standard benchmarks. 

---
# FASIONAD++ : Integrating High-Level Instruction and Information Bottleneck in FAt-Slow fusION Systems for Enhanced Safety in Autonomous Driving with Adaptive Feedback 

**Authors**: Kangan Qian, Ziang Luo, Sicong Jiang, Zilin Huang, Jinyu Miao, Zhikun Ma, Tianze Zhu, Jiayin Li, Yangfan He, Zheng Fu, Yining Shi, Boyue Wang, Hezhe Lin, Ziyu Chen, Jiangbo Yu, Xinyu Jiao, Mengmeng Yang, Kun Jiang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08162)  

**Abstract**: Ensuring safe, comfortable, and efficient planning is crucial for autonomous driving systems. While end-to-end models trained on large datasets perform well in standard driving scenarios, they struggle with complex low-frequency events. Recent Large Language Models (LLMs) and Vision Language Models (VLMs) advancements offer enhanced reasoning but suffer from computational inefficiency. Inspired by the dual-process cognitive model "Thinking, Fast and Slow", we propose $\textbf{FASIONAD}$ -- a novel dual-system framework that synergizes a fast end-to-end planner with a VLM-based reasoning module. The fast system leverages end-to-end learning to achieve real-time trajectory generation in common scenarios, while the slow system activates through uncertainty estimation to perform contextual analysis and complex scenario resolution. Our architecture introduces three key innovations: (1) A dynamic switching mechanism enabling slow system intervention based on real-time uncertainty assessment; (2) An information bottleneck with high-level plan feedback that optimizes the slow system's guidance capability; (3) A bidirectional knowledge exchange where visual prompts enhance the slow system's reasoning while its feedback refines the fast planner's decision-making. To strengthen VLM reasoning, we develop a question-answering mechanism coupled with reward-instruct training strategy. In open-loop experiments, FASIONAD achieves a $6.7\%$ reduction in average $L2$ trajectory error and $28.1\%$ lower collision rate. 

---
# FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation 

**Authors**: Wei Li, Xin Zhang, Zhongxin Guo, Shaoguang Mao, Wen Luo, Guangyue Peng, Yangyu Huang, Houfeng Wang, Scarlett Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06680)  

**Abstract**: Implementing new features in repository-level codebases is a crucial application of code generation models. However, current benchmarks lack a dedicated evaluation framework for this capability. To fill this gap, we introduce FEA-Bench, a benchmark designed to assess the ability of large language models (LLMs) to perform incremental development within code repositories. We collect pull requests from 83 GitHub repositories and use rule-based and intent-based filtering to construct task instances focused on new feature development. Each task instance containing code changes is paired with relevant unit test files to ensure that the solution can be verified. The feature implementation requires LLMs to simultaneously possess code completion capabilities for new components and code editing abilities for other relevant parts in the code repository, providing a more comprehensive evaluation method of LLMs' automated software engineering capabilities. Experimental results show that LLMs perform significantly worse in the FEA-Bench, highlighting considerable challenges in such repository-level incremental code development. 

---
# Uncovering Cross-Domain Recommendation Ability of Large Language Models 

**Authors**: Xinyi Liu, Ruijie Wang, Dachun Sun, Dilek Hakkani-Tur, Tarek Abdelzaher  

**Link**: [PDF](https://arxiv.org/pdf/2503.07761)  

**Abstract**: Cross-Domain Recommendation (CDR) seeks to enhance item retrieval in low-resource domains by transferring knowledge from high-resource domains. While recent advancements in Large Language Models (LLMs) have demonstrated their potential in Recommender Systems (RS), their ability to effectively transfer domain knowledge for improved recommendations remains underexplored. To bridge this gap, we propose LLM4CDR, a novel CDR pipeline that constructs context-aware prompts by leveraging users' purchase history sequences from a source domain along with shared features between source and target domains. Through extensive experiments, we show that LLM4CDR achieves strong performance, particularly when using LLMs with large parameter sizes and when the source and target domains exhibit smaller domain gaps. For instance, incorporating CD and Vinyl purchase history for recommendations in Movies and TV yields a 64.28 percent MAP 1 improvement. We further investigate key factors including source domain data, domain gap, prompt design, and LLM size, which impact LLM4CDR's effectiveness in CDR tasks. Our results highlight that LLM4CDR excels when leveraging a single, closely related source domain and benefits significantly from larger LLMs. These insights pave the way for future research on LLM-driven cross-domain recommendations. 

---
# KAP: MLLM-assisted OCR Text Enhancement for Hybrid Retrieval in Chinese Non-Narrative Documents 

**Authors**: Hsin-Ling Hsu, Ping-Sheng Lin, Jing-Di Lin, Jengnan Tzeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.08452)  

**Abstract**: We propose Knowledge-Aware Preprocessing (KAP), a two-stage preprocessing framework tailored for Traditional Chinese non-narrative documents, designed to enhance retrieval accuracy in Hybrid Retrieval systems. Hybrid Retrieval, which integrates Sparse Retrieval (e.g., BM25) and Dense Retrieval (e.g., vector embeddings), has become a widely adopted approach for improving search effectiveness. However, its performance heavily depends on the quality of input text, which is often degraded when dealing with non-narrative documents such as PDFs containing financial statements, contractual clauses, and tables. KAP addresses these challenges by integrating Multimodal Large Language Models (MLLMs) with LLM-driven post-OCR processing, refining extracted text to reduce OCR noise, restore table structures, and optimize text format. By ensuring better compatibility with Hybrid Retrieval, KAP improves the accuracy of both Sparse and Dense Retrieval methods without modifying the retrieval architecture itself. 

---
