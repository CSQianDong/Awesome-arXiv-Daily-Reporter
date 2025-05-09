# Chain-of-Thought Reasoning In The Wild Is Not Always Faithful 

**Authors**: Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08679)  

**Abstract**: Chain-of-Thought (CoT) reasoning has significantly advanced state-of-the-art AI capabilities. However, recent studies have shown that CoT reasoning is not always faithful, i.e. CoT reasoning does not always reflect how models arrive at conclusions. So far, most of these studies have focused on unfaithfulness in unnatural contexts where an explicit bias has been introduced. In contrast, we show that unfaithful CoT can occur on realistic prompts with no artificial bias. Our results reveal concerning rates of several forms of unfaithful reasoning in frontier models: Sonnet 3.7 (30.6%), DeepSeek R1 (15.8%) and ChatGPT-4o (12.6%) all answer a high proportion of question pairs unfaithfully. Specifically, we find that models rationalize their implicit biases in answers to binary questions ("implicit post-hoc rationalization"). For example, when separately presented with the questions "Is X bigger than Y?" and "Is Y bigger than X?", models sometimes produce superficially coherent arguments to justify answering Yes to both questions or No to both questions, despite such responses being logically contradictory. We also investigate restoration errors (Dziri et al., 2023), where models make and then silently correct errors in their reasoning, and unfaithful shortcuts, where models use clearly illogical reasoning to simplify solving problems in Putnam questions (a hard benchmark). Our findings raise challenges for AI safety work that relies on monitoring CoT to detect undesired behavior. 

---
# Reasoning and Sampling-Augmented MCQ Difficulty Prediction via LLMs 

**Authors**: Wanyong Feng, Peter Tran, Stephen Sireci, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08551)  

**Abstract**: The difficulty of multiple-choice questions (MCQs) is a crucial factor for educational assessments. Predicting MCQ difficulty is challenging since it requires understanding both the complexity of reaching the correct option and the plausibility of distractors, i.e., incorrect options. In this paper, we propose a novel, two-stage method to predict the difficulty of MCQs. First, to better estimate the complexity of each MCQ, we use large language models (LLMs) to augment the reasoning steps required to reach each option. We use not just the MCQ itself but also these reasoning steps as input to predict the difficulty. Second, to capture the plausibility of distractors, we sample knowledge levels from a distribution to account for variation among students responding to the MCQ. This setup, inspired by item response theory (IRT), enable us to estimate the likelihood of students selecting each (both correct and incorrect) option. We align these predictions with their ground truth values, using a Kullback-Leibler (KL) divergence-based regularization objective, and use estimated likelihoods to predict MCQ difficulty. We evaluate our method on two real-world \emph{math} MCQ and response datasets with ground truth difficulty values estimated using IRT. Experimental results show that our method outperforms all baselines, up to a 28.3\% reduction in mean squared error and a 34.6\% improvement in the coefficient of determination. We also qualitatively discuss how our novel method results in higher accuracy in predicting MCQ difficulty. 

---
# Graph of AI Ideas: Leveraging Knowledge Graphs and LLMs for AI Research Idea Generation 

**Authors**: Xian Gao, Zongyun Zhang, Mingye Xie, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08549)  

**Abstract**: Reading relevant scientific papers and analyzing research development trends is a critical step in generating new scientific ideas. However, the rapid increase in the volume of research literature and the complex citation relationships make it difficult for researchers to quickly analyze and derive meaningful research trends. The development of large language models (LLMs) has provided a novel approach for automatically summarizing papers and generating innovative research ideas. However, existing paper-based idea generation methods either simply input papers into LLMs via prompts or form logical chains of creative development based on citation relationships, without fully exploiting the semantic information embedded in these citations. Inspired by knowledge graphs and human cognitive processes, we propose a framework called the Graph of AI Ideas (GoAI) for the AI research field, which is dominated by open-access papers. This framework organizes relevant literature into entities within a knowledge graph and summarizes the semantic information contained in citations into relations within the graph. This organization effectively reflects the relationships between two academic papers and the advancement of the AI research field. Such organization aids LLMs in capturing the current progress of research, thereby enhancing their creativity. Experimental results demonstrate the effectiveness of our approach in generating novel, clear, and effective research ideas. 

---
# Chemical reasoning in LLMs unlocks steerable synthesis planning and reaction mechanism elucidation 

**Authors**: Andres M Bran, Theo A Neukomm, Daniel P Armstrong, Zlatko Jončev, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2503.08537)  

**Abstract**: While machine learning algorithms have been shown to excel at specific chemical tasks, they have struggled to capture the strategic thinking that characterizes expert chemical reasoning, limiting their widespread adoption. Here we demonstrate that large language models (LLMs) can serve as powerful chemical reasoning engines when integrated with traditional search algorithms, enabling a new approach to computer-aided chemistry that mirrors human expert thinking. Rather than using LLMs to directly manipulate chemical structures, we leverage their ability to evaluate chemical strategies and guide search algorithms toward chemically meaningful solutions. We demonstrate this paradigm through two fundamental challenges: strategy-aware retrosynthetic planning and mechanism elucidation. In retrosynthetic planning, our method allows chemists to specify desired synthetic strategies in natural language to find routes that satisfy these constraints in vast searches. In mechanism elucidation, LLMs guide the search for plausible reaction mechanisms by combining chemical principles with systematic exploration. Our approach shows strong performance across diverse chemical tasks, with larger models demonstrating increasingly sophisticated chemical reasoning. Our approach establishes a new paradigm for computer-aided chemistry that combines the strategic understanding of LLMs with the precision of traditional chemical tools, opening possibilities for more intuitive and powerful chemical reasoning systems. 

---
# Seeing and Reasoning with Confidence: Supercharging Multimodal LLMs with an Uncertainty-Aware Agentic Framework 

**Authors**: Zhuo Zhi, Chen Feng, Adam Daneshmend, Mine Orlu, Andreas Demosthenous, Lu Yin, Da Li, Ziquan Liu, Miguel R. D. Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2503.08308)  

**Abstract**: Multimodal large language models (MLLMs) show promise in tasks like visual question answering (VQA) but still face challenges in multimodal reasoning. Recent works adapt agentic frameworks or chain-of-thought (CoT) reasoning to improve performance. However, CoT-based multimodal reasoning often demands costly data annotation and fine-tuning, while agentic approaches relying on external tools risk introducing unreliable output from these tools. In this paper, we propose Seeing and Reasoning with Confidence (SRICE), a training-free multimodal reasoning framework that integrates external vision models with uncertainty quantification (UQ) into an MLLM to address these challenges. Specifically, SRICE guides the inference process by allowing MLLM to autonomously select regions of interest through multi-stage interactions with the help of external tools. We propose to use a conformal prediction-based approach to calibrate the output of external tools and select the optimal tool by estimating the uncertainty of an MLLM's output. Our experiment shows that the average improvement of SRICE over the base MLLM is 4.6% on five datasets and the performance on some datasets even outperforms fine-tuning-based methods, revealing the significance of ensuring reliable tool use in an MLLM agent. 

---
# Beyond Outlining: Heterogeneous Recursive Planning for Adaptive Long-form Writing with Language Models 

**Authors**: Ruibin Xiong, Yimeng Chen, Dmitrii Khizbullin, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2503.08275)  

**Abstract**: Long-form writing agents require flexible integration and interaction across information retrieval, reasoning, and composition. Current approaches rely on predetermined workflows and rigid thinking patterns to generate outlines before writing, resulting in constrained adaptability during writing. In this paper we propose a general agent framework that achieves human-like adaptive writing through recursive task decomposition and dynamic integration of three fundamental task types, i.e. retrieval, reasoning, and composition. Our methodology features: 1) a planning mechanism that interleaves recursive task decomposition and execution, eliminating artificial restrictions on writing workflow; and 2) integration of task types that facilitates heterogeneous task decomposition. Evaluations on both fiction writing and technical report generation show that our method consistently outperforms state-of-the-art approaches across all automatic evaluation metrics, which demonstrate the effectiveness and broad applicability of our proposed framework. 

---
# HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents 

**Authors**: Tristan Tomilin, Meng Fang, Mykola Pechenizkiy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08241)  

**Abstract**: Advancing safe autonomous systems through reinforcement learning (RL) requires robust benchmarks to evaluate performance, analyze methods, and assess agent competencies. Humans primarily rely on embodied visual perception to safely navigate and interact with their surroundings, making it a valuable capability for RL agents. However, existing vision-based 3D benchmarks only consider simple navigation tasks. To address this shortcoming, we introduce \textbf{HASARD}, a suite of diverse and complex tasks to $\textbf{HA}$rness $\textbf{SA}$fe $\textbf{R}$L with $\textbf{D}$oom, requiring strategic decision-making, comprehending spatial relationships, and predicting the short-term future. HASARD features three difficulty levels and two action spaces. An empirical evaluation of popular baseline methods demonstrates the benchmark's complexity, unique challenges, and reward-cost trade-offs. Visualizing agent navigation during training with top-down heatmaps provides insight into a method's learning process. Incrementally training across difficulty levels offers an implicit learning curriculum. HASARD is the first safe RL benchmark to exclusively target egocentric vision-based learning, offering a cost-effective and insightful way to explore the potential and boundaries of current and future safe RL methods. The environments and baseline implementations are open-sourced at this https URL. 

---
# Guess What I am Thinking: A Benchmark for Inner Thought Reasoning of Role-Playing Language Agents 

**Authors**: Rui Xu, MingYu Wang, XinTao Wang, Dakuan Lu, Xiaoyu Tan, Wei Chu, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08193)  

**Abstract**: Recent advances in LLM-based role-playing language agents (RPLAs) have attracted broad attention in various applications. While chain-of-thought reasoning has shown importance in many tasks for LLMs, the internal thinking processes of RPLAs remain unexplored. Understanding characters' inner thoughts is crucial for developing advanced RPLAs. In this paper, we introduce ROLETHINK, a novel benchmark constructed from literature for evaluating character thought generation. We propose the task of inner thought reasoning, which includes two sets: the gold set that compares generated thoughts with original character monologues, and the silver set that uses expert synthesized character analyses as references. To address this challenge, we propose MIRROR, a chain-of-thought approach that generates character thoughts by retrieving memories, predicting character reactions, and synthesizing motivations. Through extensive experiments, we demonstrate the importance of inner thought reasoning for RPLAs, and MIRROR consistently outperforms existing methods. Resources are available at this https URL. 

---
# Privacy-Enhancing Paradigms within Federated Multi-Agent Systems 

**Authors**: Zitong Shi, Guancheng Wan, Wenke Huang, Guibin Zhang, Jiawei Shao, Mang Ye, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08175)  

**Abstract**: LLM-based Multi-Agent Systems (MAS) have proven highly effective in solving complex problems by integrating multiple agents, each performing different roles. However, in sensitive domains, they face emerging privacy protection challenges. In this paper, we introduce the concept of Federated MAS, highlighting the fundamental differences between Federated MAS and traditional FL. We then identify key challenges in developing Federated MAS, including: 1) heterogeneous privacy protocols among agents, 2) structural differences in multi-party conversations, and 3) dynamic conversational network structures. To address these challenges, we propose Embedded Privacy-Enhancing Agents (EPEAgent), an innovative solution that integrates seamlessly into the Retrieval-Augmented Generation (RAG) phase and the context retrieval stage. This solution minimizes data flows, ensuring that only task-relevant, agent-specific information is shared. Additionally, we design and generate a comprehensive dataset to evaluate the proposed paradigm. Extensive experiments demonstrate that EPEAgent effectively enhances privacy protection while maintaining strong system performance. The code will be availiable at this https URL 

---
# AI-native Memory 2.0: Second Me 

**Authors**: Jiale Wei, Xiang Ying, Tao Gao, Felix Tao, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08102)  

**Abstract**: Human interaction with the external world fundamentally involves the exchange of personal memory, whether with other individuals, websites, applications, or, in the future, AI agents. A significant portion of this interaction is redundant, requiring users to repeatedly provide the same information across different contexts. Existing solutions, such as browser-stored credentials, autofill mechanisms, and unified authentication systems, have aimed to mitigate this redundancy by serving as intermediaries that store and retrieve commonly used user data. The advent of large language models (LLMs) presents an opportunity to redefine memory management through an AI-native paradigm: SECOND ME. SECOND ME acts as an intelligent, persistent memory offload system that retains, organizes, and dynamically utilizes user-specific knowledge. By serving as an intermediary in user interactions, it can autonomously generate context-aware responses, prefill required information, and facilitate seamless communication with external systems, significantly reducing cognitive load and interaction friction. Unlike traditional memory storage solutions, SECOND ME extends beyond static data retention by leveraging LLM-based memory parameterization. This enables structured organization, contextual reasoning, and adaptive knowledge retrieval, facilitating a more systematic and intelligent approach to memory management. As AI-driven personal agents like SECOND ME become increasingly integrated into digital ecosystems, SECOND ME further represents a critical step toward augmenting human-world interaction with persistent, contextually aware, and self-optimizing memory systems. We have open-sourced the fully localizable deployment system at GitHub: this https URL. 

---
# STGDPM:Vessel Trajectory Prediction with Spatio-Temporal Graph Diffusion Probabilistic Model 

**Authors**: Jin Wenzhe, Tang Haina, Zhang Xudong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08065)  

**Abstract**: Vessel trajectory prediction is a critical component for ensuring maritime traffic safety and avoiding collisions. Due to the inherent uncertainty in vessel behavior, trajectory prediction systems must adopt a multimodal approach to accurately model potential future motion states. However, existing vessel trajectory prediction methods lack the ability to comprehensively model behavioral multi-modality. To better capture multimodal behavior in interactive scenarios, we propose modeling interactions as dynamic graphs, replacing traditional aggregation-based techniques that rely on vessel states. By leveraging the natural multimodal capabilities of diffusion models, we frame the trajectory prediction task as an inverse process of motion uncertainty diffusion, wherein uncertainties across potential navigational areas are progressively eliminated until the desired trajectories is produced. In summary, we pioneer the integration of Spatio-Temporal Graph (STG) with diffusion models in ship trajectory prediction. Extensive experiments on real Automatic Identification System (AIS) data validate the superiority of our approach. 

---
# Counterfactual Language Reasoning for Explainable Recommendation Systems 

**Authors**: Guanrong Li, Haolin Yang, Xinyu Liu, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.08051)  

**Abstract**: Explainable recommendation systems leverage transparent reasoning to foster user trust and improve decision-making processes. Current approaches typically decouple recommendation generation from explanation creation, violating causal precedence principles where explanatory factors should logically precede outcomes. This paper introduces a novel framework integrating structural causal models with large language models to establish causal consistency in recommendation pipelines. Our methodology enforces explanation factors as causal antecedents to recommendation predictions through causal graph construction and counterfactual adjustment. We particularly address the confounding effect of item popularity that distorts personalization signals in explanations, developing a debiasing mechanism that disentangles genuine user preferences from conformity bias. Through comprehensive experiments across multiple recommendation scenarios, we demonstrate that CausalX achieves superior performance in recommendation accuracy, explanation plausibility, and bias mitigation compared to baselines. 

---
# SQLCritic: Correcting Text-to-SQL Generation via Clause-wise Critic 

**Authors**: Jikai Chen, Leilei Gan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07996)  

**Abstract**: Recent advancements in Text-to-SQL systems have improved the conversion of natural language queries into SQL, but challenges remain in ensuring accuracy and reliability. While self-correction techniques refine outputs, they often introduce new errors. Existing methods focused on execution feedback mainly address syntax issues, leaving semantic errors -- where the query's logic fails to align with the user's intent -- largely unaddressed.
We propose a novel approach combining structured execution feedback with a trained critic agent that provides detailed, interpretable critiques. This method effectively identifies and corrects both syntactic and semantic errors, enhancing accuracy and interpretability. Experimental results show significant improvements on two major Text-to-SQL benchmarks, Spider and BIRD, demonstrating the effectiveness of our approach. 

---
# LLM-Powered Knowledge Graphs for Enterprise Intelligence and Analytics 

**Authors**: Rajeev Kumar, Kumar Ishan, Harishankar Kumar, Abhinandan Singla  

**Link**: [PDF](https://arxiv.org/pdf/2503.07993)  

**Abstract**: Disconnected data silos within enterprises obstruct the extraction of actionable insights, diminishing efficiency in areas such as product development, client engagement, meeting preparation, and analytics-driven decision-making. This paper introduces a framework that uses large language models (LLMs) to unify various data sources into a comprehensive, activity-centric knowledge graph. The framework automates tasks such as entity extraction, relationship inference, and semantic enrichment, enabling advanced querying, reasoning, and analytics across data types like emails, calendars, chats, documents, and logs. Designed for enterprise flexibility, it supports applications such as contextual search, task prioritization, expertise discovery, personalized recommendations, and advanced analytics to identify trends and actionable insights. Experimental results demonstrate its success in the discovery of expertise, task management, and data-driven decision making. By integrating LLMs with knowledge graphs, this solution bridges disconnected systems and delivers intelligent analytics-powered enterprise tools. 

---
# Boundary Prompting: Elastic Urban Region Representation via Graph-based Spatial Tokenization 

**Authors**: Haojia Zhu, Jiahui Jin, Dong Kan, Rouxi Shen, Ruize Wang, Xiangguo Sun, Jinghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07991)  

**Abstract**: Urban region representation is essential for various applications such as urban planning, resource allocation, and policy development. Traditional methods rely on fixed, predefined region boundaries, which fail to capture the dynamic and complex nature of real-world urban areas. In this paper, we propose the Boundary Prompting Urban Region Representation Framework (BPURF), a novel approach that allows for elastic urban region definitions. BPURF comprises two key components: (1) A spatial token dictionary, where urban entities are treated as tokens and integrated into a unified token graph, and (2) a region token set representation model which utilize token aggregation and a multi-channel model to embed token sets corresponding to region boundaries. Additionally, we propose fast token set extraction strategy to enable online token set extraction during training and prompting. This framework enables the definition of urban regions through boundary prompting, supporting varying region boundaries and adapting to different tasks. Extensive experiments demonstrate the effectiveness of BPURF in capturing the complex characteristics of urban regions. 

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
# BEARCUBS: A benchmark for computer-using web agents 

**Authors**: Yixiao Song, Katherine Thai, Chau Minh Pham, Yapei Chang, Mazin Nadaf, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.07919)  

**Abstract**: Modern web agents possess computer use abilities that allow them to interact with webpages by sending commands to a virtual keyboard and mouse. While such agents have considerable potential to assist human users with complex tasks, evaluating their capabilities in real-world settings poses a major challenge. To this end, we introduce BEARCUBS, a "small but mighty" benchmark of 111 information-seeking questions designed to evaluate a web agent's ability to search, browse, and identify factual information from the web. Unlike prior web agent benchmarks, solving BEARCUBS requires (1) accessing live web content rather than synthetic or simulated pages, which captures the unpredictability of real-world web interactions; and (2) performing a broad range of multimodal interactions (e.g., video understanding, 3D navigation) that cannot be bypassed via text-based workarounds. Each question in BEARCUBS has a corresponding short, unambiguous answer and a human-validated browsing trajectory, allowing for transparent evaluation of agent performance and strategies. A human study confirms that BEARCUBS questions are solvable but non-trivial (84.7% human accuracy), revealing search inefficiencies and domain knowledge gaps as common failure points. By contrast, state-of-the-art computer-using agents underperform, with the best-scoring system (OpenAI's Operator) reaching only 24.3% accuracy. These results highlight critical areas for improvement, including reliable source selection and more powerful multimodal capabilities. To facilitate future research, BEARCUBS will be updated periodically to replace invalid or contaminated questions, keeping the benchmark fresh for future generations of web agents. 

---
# Demystifying the Accuracy-Interpretability Trade-Off: A Case Study of Inferring Ratings from Reviews 

**Authors**: Pranjal Atrey, Michael P. Brundage, Min Wu, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2503.07914)  

**Abstract**: Interpretable machine learning models offer understandable reasoning behind their decision-making process, though they may not always match the performance of their black-box counterparts. This trade-off between interpretability and model performance has sparked discussions around the deployment of AI, particularly in critical applications where knowing the rationale of decision-making is essential for trust and accountability. In this study, we conduct a comparative analysis of several black-box and interpretable models, focusing on a specific NLP use case that has received limited attention: inferring ratings from reviews. Through this use case, we explore the intricate relationship between the performance and interpretability of different models. We introduce a quantitative score called Composite Interpretability (CI) to help visualize the trade-off between interpretability and performance, particularly in the case of composite models. Our results indicate that, in general, the learning performance improves as interpretability decreases, but this relationship is not strictly monotonic, and there are instances where interpretable models are more advantageous. 

---
# Actual Causation and Nondeterministic Causal Models 

**Authors**: Sander Beckers  

**Link**: [PDF](https://arxiv.org/pdf/2503.07849)  

**Abstract**: In (Beckers, 2025) I introduced nondeterministic causal models as a generalization of Pearl's standard deterministic causal models. I here take advantage of the increased expressivity offered by these models to offer a novel definition of actual causation (that also applies to deterministic models). Instead of motivating the definition by way of (often subjective) intuitions about examples, I proceed by developing it based entirely on the unique function that it can fulfil in communicating and learning a causal model. First I generalize the more basic notion of counterfactual dependence, second I show how this notion has a vital role to play in the logic of causal discovery, third I introduce the notion of a structural simplification of a causal model, and lastly I bring both notions together in my definition of actual causation. Although novel, the resulting definition arrives at verdicts that are almost identical to those of my previous definition (Beckers, 2021, 2022). 

---
# Safe Explicable Policy Search 

**Authors**: Akkamahadevi Hanni, Jonathan Montaño, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07848)  

**Abstract**: When users work with AI agents, they form conscious or subconscious expectations of them. Meeting user expectations is crucial for such agents to engage in successful interactions and teaming. However, users may form expectations of an agent that differ from the agent's planned behaviors. These differences lead to the consideration of two separate decision models in the planning process to generate explicable behaviors. However, little has been done to incorporate safety considerations, especially in a learning setting. We present Safe Explicable Policy Search (SEPS), which aims to provide a learning approach to explicable behavior generation while minimizing the safety risk, both during and after learning. We formulate SEPS as a constrained optimization problem where the agent aims to maximize an explicability score subject to constraints on safety and a suboptimality criterion based on the agent's model. SEPS innovatively combines the capabilities of Constrained Policy Optimization and Explicable Policy Search. We evaluate SEPS in safety-gym environments and with a physical robot experiment to show that it can learn explicable behaviors that adhere to the agent's safety requirements and are efficient. Results show that SEPS can generate safe and explicable behaviors while ensuring a desired level of performance w.r.t. the agent's objective, and has real-world relevance in human-AI teaming. 

---
# RefactorBench: Evaluating Stateful Reasoning in Language Agents Through Code 

**Authors**: Dhruv Gautam, Spandan Garg, Jinu Jang, Neel Sundaresan, Roshanak Zilouchian Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2503.07832)  

**Abstract**: Recent advances in language model (LM) agents and function calling have enabled autonomous, feedback-driven systems to solve problems across various digital domains. To better understand the unique limitations of LM agents, we introduce RefactorBench, a benchmark consisting of 100 large handcrafted multi-file refactoring tasks in popular open-source repositories. Solving tasks within RefactorBench requires thorough exploration of dependencies across multiple files and strong adherence to relevant instructions. Every task is defined by 3 natural language instructions of varying specificity and is mutually exclusive, allowing for the creation of longer combined tasks on the same repository. Baselines on RefactorBench reveal that current LM agents struggle with simple compositional tasks, solving only 22% of tasks with base instructions, in contrast to a human developer with short time constraints solving 87%. Through trajectory analysis, we identify various unique failure modes of LM agents, and further explore the failure mode of tracking past actions. By adapting a baseline agent to condition on representations of state, we achieve a 43.9% improvement in solving RefactorBench tasks. We further extend our state-aware approach to encompass entire digital environments and outline potential directions for future research. RefactorBench aims to support the study of LM agents by providing a set of real-world, multi-hop tasks within the realm of code. 

---
# Efficient Neural Clause-Selection Reinforcement 

**Authors**: Martin Suda  

**Link**: [PDF](https://arxiv.org/pdf/2503.07792)  

**Abstract**: Clause selection is arguably the most important choice point in saturation-based theorem proving. Framing it as a reinforcement learning (RL) task is a way to challenge the human-designed heuristics of state-of-the-art provers and to instead automatically evolve -- just from prover experiences -- their potentially optimal replacement. In this work, we present a neural network architecture for scoring clauses for clause selection that is powerful yet efficient to evaluate. Following RL principles to make design decisions, we integrate the network into the Vampire theorem prover and train it from successful proof attempts. An experiment on the diverse TPTP benchmark finds the neurally guided prover improve over a baseline strategy, from which it initially learns--in terms of the number of in-training-unseen problems solved under a practically relevant, short CPU instruction limit--by 20%. 

---
# Sensemaking in Novel Environments: How Human Cognition Can Inform Artificial Agents 

**Authors**: Robert E. Patterson, Regina Buccello-Stout, Mary E. Frame, Anna M. Maresca, Justin Nelson, Barbara Acker-Mills, Erica Curtis, Jared Culbertson, Kevin Schmidt, Scott Clouse, Steve Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2503.07783)  

**Abstract**: One of the most vital cognitive skills to possess is the ability to make sense of objects, events, and situations in the world. In the current paper, we offer an approach for creating artificially intelligent agents with the capacity for sensemaking in novel environments. Objectives: to present several key ideas: (1) a novel unified conceptual framework for sensemaking (which includes the existence of sign relations embedded within and across frames); (2) interaction among various content-addressable, distributed-knowledge structures via shared attributes (whose net response would represent a synthesized object, event, or situation serving as a sign for sensemaking in a novel environment). Findings: we suggest that attributes across memories can be shared and recombined in novel ways to create synthesized signs, which can denote certain outcomes in novel environments (i.e., sensemaking). 

---
# Fully Autonomous Programming using Iterative Multi-Agent Debugging with Large Language Models 

**Authors**: Anastasiia Grishina, Vadim Liventsev, Aki Härmä, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07693)  

**Abstract**: Program synthesis with Large Language Models (LLMs) suffers from a "near-miss syndrome": the generated code closely resembles a correct solution but fails unit tests due to minor errors. We address this with a multi-agent framework called Synthesize, Execute, Instruct, Debug, and Repair (SEIDR). Effectively applying SEIDR to instruction-tuned LLMs requires determining (a) optimal prompts for LLMs, (b) what ranking algorithm selects the best programs in debugging rounds, and (c) balancing the repair of unsuccessful programs with the generation of new ones. We empirically explore these trade-offs by comparing replace-focused, repair-focused, and hybrid debug strategies. We also evaluate lexicase and tournament selection to rank candidates in each generation. On Program Synthesis Benchmark 2 (PSB2), our framework outperforms both conventional use of OpenAI Codex without a repair phase and traditional genetic programming approaches. SEIDR outperforms the use of an LLM alone, solving 18 problems in C++ and 20 in Python on PSB2 at least once across experiments. To assess generalizability, we employ GPT-3.5 and Llama 3 on the PSB2 and HumanEval-X benchmarks. Although SEIDR with these models does not surpass current state-of-the-art methods on the Python benchmarks, the results on HumanEval-C++ are promising. SEIDR with Llama 3-8B achieves an average pass@100 of 84.2%. Across all SEIDR runs, 163 of 164 problems are solved at least once with GPT-3.5 in HumanEval-C++, and 162 of 164 with the smaller Llama 3-8B. We conclude that SEIDR effectively overcomes the near-miss syndrome in program synthesis with LLMs. 

---
# Research on Superalignment Should Advance Now with Parallel Optimization of Competence and Conformity 

**Authors**: HyunJin Kim, Xiaoyuan Yi, Jing Yao, Muhua Huang, JinYeong Bak, James Evans, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.07660)  

**Abstract**: The recent leap in AI capabilities, driven by big generative models, has sparked the possibility of achieving Artificial General Intelligence (AGI) and further triggered discussions on Artificial Superintelligence (ASI), a system surpassing all humans across all domains. This gives rise to the critical research question of: If we realize ASI, how do we align it with human values, ensuring it benefits rather than harms human society, a.k.a., the Superalignment problem. Despite ASI being regarded by many as solely a hypothetical concept, in this paper, we argue that superalignment is achievable and research on it should advance immediately, through simultaneous and alternating optimization of task competence and value conformity. We posit that superalignment is not merely a safeguard for ASI but also necessary for its realization. To support this position, we first provide a formal definition of superalignment rooted in the gap between capability and capacity and elaborate on our argument. Then we review existing paradigms, explore their interconnections and limitations, and illustrate a potential path to superalignment centered on two fundamental principles. We hope this work sheds light on a practical approach for developing the value-aligned next-generation AI, garnering greater benefits and reducing potential harms for humanity. 

---
# Impact of Level 2/3 Automated Driving Technology on Road Work Zone Safety 

**Authors**: Zhepu Xu, Ziyi Song, Yupu Dong, Peiyan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07634)  

**Abstract**: As China's road network enters the maintenance era, work zones will become a common sight on the roads. With the development of automated driving, vehicles equipped with Level 2/3 automated driving capabilities will also become a common presence on the roads. When these vehicles pass through work zones, automated driving may disengage, which can have complex effects on traffic safety. This paper explores the impact of Level 2/3 automated driving technology on road safety in high-speed highway work zone environments. Through microscopic traffic simulation method and using full-type traffic conflict technique, factors such as market penetration rate (MPR), traffic volume level, disengagement threshold, and driver takeover style are studied to understand their impact on work zone safety. The study found that the impact of automated driving technology on work zone safety is complex. Disengagement of automated vehicles in work zones reduces the proportion of vehicles that can maintain automated driving status. If takeover is not timely or adequate, it can easily lead to new traffic conflicts. Different factors have varying degrees of impact on work zone safety. Increasing MPR helps reduce the occurrence of single-vehicle conflicts, but it also increases the possibility of multi-vehicle conflicts. Therefore, future research and improvement directions should focus on optimizing the disengagement detection and takeover mechanisms of automated driving systems. 

---
# Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents 

**Authors**: Haoyu Wang, Sunhao Dai, Haiyuan Zhao, Liang Pang, Xiao Zhang, Gang Wang, Zhenhua Dong, Jun Xu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08684)  

**Abstract**: Previous studies have found that PLM-based retrieval models exhibit a preference for LLM-generated content, assigning higher relevance scores to these documents even when their semantic quality is comparable to human-written ones. This phenomenon, known as source bias, threatens the sustainable development of the information access ecosystem. However, the underlying causes of source bias remain unexplored. In this paper, we explain the process of information retrieval with a causal graph and discover that PLM-based retrievers learn perplexity features for relevance estimation, causing source bias by ranking the documents with low perplexity higher. Theoretical analysis further reveals that the phenomenon stems from the positive correlation between the gradients of the loss functions in language modeling task and retrieval task. Based on the analysis, a causal-inspired inference-time debiasing method is proposed, called Causal Diagnosis and Correction (CDC). CDC first diagnoses the bias effect of the perplexity and then separates the bias effect from the overall estimated relevance score. Experimental results across three domains demonstrate the superior debiasing effectiveness of CDC, emphasizing the validity of our proposed explanatory framework. Source codes are available at this https URL. 

---
# CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving 

**Authors**: Changxing Liu, Genjia Liu, Zijun Wang, Jinchang Yang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08683)  

**Abstract**: Vehicle-to-vehicle (V2V) cooperative autonomous driving holds great promise for improving safety by addressing the perception and prediction uncertainties inherent in single-agent systems. However, traditional cooperative methods are constrained by rigid collaboration protocols and limited generalization to unseen interactive scenarios. While LLM-based approaches offer generalized reasoning capabilities, their challenges in spatial planning and unstable inference latency hinder their direct application in cooperative driving. To address these limitations, we propose CoLMDriver, the first full-pipeline LLM-based cooperative driving system, enabling effective language-based negotiation and real-time driving control. CoLMDriver features a parallel driving pipeline with two key components: (i) an LLM-based negotiation module under an actor-critic paradigm, which continuously refines cooperation policies through feedback from previous decisions of all vehicles; and (ii) an intention-guided waypoint generator, which translates negotiation outcomes into executable waypoints. Additionally, we introduce InterDrive, a CARLA-based simulation benchmark comprising 10 challenging interactive driving scenarios for evaluating V2V cooperation. Experimental results demonstrate that CoLMDriver significantly outperforms existing approaches, achieving an 11% higher success rate across diverse highly interactive V2V driving scenarios. Code will be released on this https URL. 

---
# GarmentCrafter: Progressive Novel View Synthesis for Single-View 3D Garment Reconstruction and Editing 

**Authors**: Yuanhao Wang, Cheng Zhang, Gonçalo Frazão, Jinlong Yang, Alexandru-Eugen Ichim, Thabo Beeler, Fernando De la Torre  

**Link**: [PDF](https://arxiv.org/pdf/2503.08678)  

**Abstract**: We introduce GarmentCrafter, a new approach that enables non-professional users to create and modify 3D garments from a single-view image. While recent advances in image generation have facilitated 2D garment design, creating and editing 3D garments remains challenging for non-professional users. Existing methods for single-view 3D reconstruction often rely on pre-trained generative models to synthesize novel views conditioning on the reference image and camera pose, yet they lack cross-view consistency, failing to capture the internal relationships across different views. In this paper, we tackle this challenge through progressive depth prediction and image warping to approximate novel views. Subsequently, we train a multi-view diffusion model to complete occluded and unknown clothing regions, informed by the evolving camera pose. By jointly inferring RGB and depth, GarmentCrafter enforces inter-view coherence and reconstructs precise geometries and fine details. Extensive experiments demonstrate that our method achieves superior visual fidelity and inter-view coherence compared to state-of-the-art single-view 3D garment reconstruction methods. 

---
# AgentOrca: A Dual-System Framework to Evaluate Language Agents on Operational Routine and Constraint Adherence 

**Authors**: Zekun Li, Shinda Huang, Jiangtian Wang, Nathan Zhang, Antonis Antoniades, Wenyue Hua, Kaijie Zhu, Sirui Zeng, William Yang Wang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08669)  

**Abstract**: As language agents progressively automate critical tasks across domains, their ability to operate within operational constraints and safety protocols becomes essential. While extensive research has demonstrated these agents' effectiveness in downstream task completion, their reliability in following operational procedures and constraints remains largely unexplored. To this end, we present AgentOrca, a dual-system framework for evaluating language agents' compliance with operational constraints and routines. Our framework encodes action constraints and routines through both natural language prompts for agents and corresponding executable code serving as ground truth for automated verification. Through an automated pipeline of test case generation and evaluation across five real-world domains, we quantitatively assess current language agents' adherence to operational constraints. Our findings reveal notable performance gaps among state-of-the-art models, with large reasoning models like o1 demonstrating superior compliance while others show significantly lower performance, particularly when encountering complex constraints or user persuasion attempts. 

---
# REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder 

**Authors**: Yitian Zhang, Long Mai, Aniruddha Mahapatra, David Bourgin, Yicong Hong, Jonah Casebeer, Feng Liu, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08665)  

**Abstract**: We present a novel perspective on learning video embedders for generative modeling: rather than requiring an exact reproduction of an input video, an effective embedder should focus on synthesizing visually plausible reconstructions. This relaxed criterion enables substantial improvements in compression ratios without compromising the quality of downstream generative models. Specifically, we propose replacing the conventional encoder-decoder video embedder with an encoder-generator framework that employs a diffusion transformer (DiT) to synthesize missing details from a compact latent space. Therein, we develop a dedicated latent conditioning module to condition the DiT decoder on the encoded video latent embedding. Our experiments demonstrate that our approach enables superior encoding-decoding performance compared to state-of-the-art methods, particularly as the compression ratio increases. To demonstrate the efficacy of our approach, we report results from our video embedders achieving a temporal compression ratio of up to 32x (8x higher than leading video embedders) and validate the robustness of this ultra-compact latent space for text-to-video generation, providing a significant efficiency boost in latent diffusion model training and inference. 

---
# MEAT: Multiview Diffusion Model for Human Generation on Megapixels with Mesh Attention 

**Authors**: Yuhan Wang, Fangzhou Hong, Shuai Yang, Liming Jiang, Wayne Wu, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08664)  

**Abstract**: Multiview diffusion models have shown considerable success in image-to-3D generation for general objects. However, when applied to human data, existing methods have yet to deliver promising results, largely due to the challenges of scaling multiview attention to higher resolutions. In this paper, we explore human multiview diffusion models at the megapixel level and introduce a solution called mesh attention to enable training at 1024x1024 resolution. Using a clothed human mesh as a central coarse geometric representation, the proposed mesh attention leverages rasterization and projection to establish direct cross-view coordinate correspondences. This approach significantly reduces the complexity of multiview attention while maintaining cross-view consistency. Building on this foundation, we devise a mesh attention block and combine it with keypoint conditioning to create our human-specific multiview diffusion model, MEAT. In addition, we present valuable insights into applying multiview human motion videos for diffusion training, addressing the longstanding issue of data scarcity. Extensive experiments show that MEAT effectively generates dense, consistent multiview human images at the megapixel level, outperforming existing multiview diffusion methods. 

---
# Generating Robot Constitutions & Benchmarks for Semantic Safety 

**Authors**: Pierre Sermanet, Anirudha Majumdar, Alex Irpan, Dmitry Kalashnikov, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2503.08663)  

**Abstract**: Until recently, robotics safety research was predominantly about collision avoidance and hazard reduction in the immediate vicinity of a robot. Since the advent of large vision and language models (VLMs), robots are now also capable of higher-level semantic scene understanding and natural language interactions with humans. Despite their known vulnerabilities (e.g. hallucinations or jail-breaking), VLMs are being handed control of robots capable of physical contact with the real world. This can lead to dangerous behaviors, making semantic safety for robots a matter of immediate concern. Our contributions in this paper are two fold: first, to address these emerging risks, we release the ASIMOV Benchmark, a large-scale and comprehensive collection of datasets for evaluating and improving semantic safety of foundation models serving as robot brains. Our data generation recipe is highly scalable: by leveraging text and image generation techniques, we generate undesirable situations from real-world visual scenes and human injury reports from hospitals. Secondly, we develop a framework to automatically generate robot constitutions from real-world data to steer a robot's behavior using Constitutional AI mechanisms. We propose a novel auto-amending process that is able to introduce nuances in written rules of behavior; this can lead to increased alignment with human preferences on behavior desirability and safety. We explore trade-offs between generality and specificity across a diverse set of constitutions of different lengths, and demonstrate that a robot is able to effectively reject unconstitutional actions. We measure a top alignment rate of 84.3% on the ASIMOV Benchmark using generated constitutions, outperforming no-constitution baselines and human-written constitutions. Data is available at this http URL 

---
# Exploring the Word Sense Disambiguation Capabilities of Large Language Models 

**Authors**: Pierpaolo Basile, Lucia Siciliani, Elio Musacchio, Giovanni Semeraro  

**Link**: [PDF](https://arxiv.org/pdf/2503.08662)  

**Abstract**: Word Sense Disambiguation (WSD) is a historical task in computational linguistics that has received much attention over the years. However, with the advent of Large Language Models (LLMs), interest in this task (in its classical definition) has decreased. In this study, we evaluate the performance of various LLMs on the WSD task. We extend a previous benchmark (XL-WSD) to re-design two subtasks suitable for LLM: 1) given a word in a sentence, the LLM must generate the correct definition; 2) given a word in a sentence and a set of predefined meanings, the LLM must select the correct one. The extended benchmark is built using the XL-WSD and BabelNet. The results indicate that LLMs perform well in zero-shot learning but cannot surpass current state-of-the-art methods. However, a fine-tuned model with a medium number of parameters outperforms all other models, including the state-of-the-art. 

---
# Exploiting Instruction-Following Retrievers for Malicious Information Retrieval 

**Authors**: Parishad BehnamGhader, Nicholas Meade, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08644)  

**Abstract**: Instruction-following retrievers have been widely adopted alongside LLMs in real-world applications, but little work has investigated the safety risks surrounding their increasing search capabilities. We empirically study the ability of retrievers to satisfy malicious queries, both when used directly and when used in a retrieval augmented generation-based setup. Concretely, we investigate six leading retrievers, including NV-Embed and LLM2Vec, and find that given malicious requests, most retrievers can (for >50% of queries) select relevant harmful passages. For example, LLM2Vec correctly selects passages for 61.35% of our malicious queries. We further uncover an emerging risk with instruction-following retrievers, where highly relevant harmful information can be surfaced by exploiting their instruction-following capabilities. Finally, we show that even safety-aligned LLMs, such as Llama3, can satisfy malicious requests when provided with harmful retrieved passages in-context. In summary, our findings underscore the malicious misuse risks associated with increasing retriever capability. 

---
# Rethinking Diffusion Model in High Dimension 

**Authors**: Zhenxin Zheng, Zhenjie Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.08643)  

**Abstract**: Curse of Dimensionality is an unavoidable challenge in statistical probability models, yet diffusion models seem to overcome this limitation, achieving impressive results in high-dimensional data generation. Diffusion models assume that they can learn the statistical properties of the underlying probability distribution, enabling sampling from this distribution to generate realistic samples. But is this really how they work? To address this question, this paper conducts a detailed analysis of the objective function and inference methods of diffusion models, leading to several important conclusions that help answer the above question: 1) In high-dimensional sparse scenarios, the target of the objective function fitting degrades from a weighted sum of multiple samples to a single sample. 2) The mainstream inference methods can all be represented within a simple unified framework, without requiring statistical concepts such as Markov chains and SDEs. 3) Guided by this simple framework, more efficient inference methods can be discovered. 

---
# YuE: Scaling Open Foundation Models for Long-Form Music Generation 

**Authors**: Ruibin Yuan, Hanfeng Lin, Shuyue Guo, Ge Zhang, Jiahao Pan, Yongyi Zang, Haohe Liu, Yiming Liang, Wenye Ma, Xingjian Du, Xinrun Du, Zhen Ye, Tianyu Zheng, Yinghao Ma, Minghao Liu, Zeyue Tian, Ziya Zhou, Liumeng Xue, Xingwei Qu, Yizhi Li, Shangda Wu, Tianhao Shen, Ziyang Ma, Jun Zhan, Chunhui Wang, Yatian Wang, Xiaowei Chi, Xinyue Zhang, Zhenzhu Yang, Xiangzhou Wang, Shansong Liu, Lingrui Mei, Peng Li, Junjie Wang, Jianwei Yu, Guojian Pang, Xu Li, Zihao Wang, Xiaohuan Zhou, Lijun Yu, Emmanouil Benetos, Yong Chen, Chenghua Lin, Xie Chen, Gus Xia, Zhaoxiang Zhang, Chao Zhang, Wenhu Chen, Xinyu Zhou, Xipeng Qiu, Roger Dannenberg, Jiaheng Liu, Jian Yang, Wenhao Huang, Wei Xue, Xu Tan, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.08638)  

**Abstract**: We tackle the task of long-form music generation--particularly the challenging \textbf{lyrics-to-song} problem--by introducing YuE, a family of open foundation models based on the LLaMA2 architecture. Specifically, YuE scales to trillions of tokens and generates up to five minutes of music while maintaining lyrical alignment, coherent musical structure, and engaging vocal melodies with appropriate accompaniment. It achieves this through (1) track-decoupled next-token prediction to overcome dense mixture signals, (2) structural progressive conditioning for long-context lyrical alignment, and (3) a multitask, multiphase pre-training recipe to converge and generalize. In addition, we redesign the in-context learning technique for music generation, enabling versatile style transfer (e.g., converting Japanese city pop into an English rap while preserving the original accompaniment) and bidirectional generation. Through extensive evaluation, we demonstrate that YuE matches or even surpasses some of the proprietary systems in musicality and vocal agility. In addition, fine-tuning YuE enables additional controls and enhanced support for tail languages. Furthermore, beyond generation, we show that YuE's learned representations can perform well on music understanding tasks, where the results of YuE match or exceed state-of-the-art methods on the MARBLE benchmark. Keywords: lyrics2song, song generation, long-form, foundation model, music generation 

---
# Vision Transformer for Intracranial Hemorrhage Classification in CT Scans Using an Entropy-Aware Fuzzy Integral Strategy for Adaptive Scan-Level Decision Fusion 

**Authors**: Mehdi Hosseini Chagahi, Niloufar Delfan, Behzad Moshiri, Md. Jalil Piran, Jaber Hatam Parikhan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08609)  

**Abstract**: Intracranial hemorrhage (ICH) is a critical medical emergency caused by the rupture of cerebral blood vessels, leading to internal bleeding within the skull. Accurate and timely classification of hemorrhage subtypes is essential for effective clinical decision-making. To address this challenge, we propose an advanced pyramid vision transformer (PVT)-based model, leveraging its hierarchical attention mechanisms to capture both local and global spatial dependencies in brain CT scans. Instead of processing all extracted features indiscriminately, A SHAP-based feature selection method is employed to identify the most discriminative components, which are then used as a latent feature space to train a boosting neural network, reducing computational complexity. We introduce an entropy-aware aggregation strategy along with a fuzzy integral operator to fuse information across multiple CT slices, ensuring a more comprehensive and reliable scan-level diagnosis by accounting for inter-slice dependencies. Experimental results show that our PVT-based framework significantly outperforms state-of-the-art deep learning architectures in terms of classification accuracy, precision, and robustness. By combining SHAP-driven feature selection, transformer-based modeling, and an entropy-aware fuzzy integral operator for decision fusion, our method offers a scalable and computationally efficient AI-driven solution for automated ICH subtype classification. 

---
# A Grid Cell-Inspired Structured Vector Algebra for Cognitive Maps 

**Authors**: Sven Krausse, Emre Neftci, Friedrich T. Sommer, Alpha Renner  

**Link**: [PDF](https://arxiv.org/pdf/2503.08608)  

**Abstract**: The entorhinal-hippocampal formation is the mammalian brain's navigation system, encoding both physical and abstract spaces via grid cells. This system is well-studied in neuroscience, and its efficiency and versatility make it attractive for applications in robotics and machine learning. While continuous attractor networks (CANs) successfully model entorhinal grid cells for encoding physical space, integrating both continuous spatial and abstract spatial computations into a unified framework remains challenging. Here, we attempt to bridge this gap by proposing a mechanistic model for versatile information processing in the entorhinal-hippocampal formation inspired by CANs and Vector Symbolic Architectures (VSAs), a neuro-symbolic computing framework. The novel grid-cell VSA (GC-VSA) model employs a spatially structured encoding scheme with 3D neuronal modules mimicking the discrete scales and orientations of grid cell modules, reproducing their characteristic hexagonal receptive fields. In experiments, the model demonstrates versatility in spatial and abstract tasks: (1) accurate path integration for tracking locations, (2) spatio-temporal representation for querying object locations and temporal relations, and (3) symbolic reasoning using family trees as a structured test case for hierarchical relationships. 

---
# Tuning-Free Multi-Event Long Video Generation via Synchronized Coupled Sampling 

**Authors**: Subin Kim, Seoung Wug Oh, Jui-Hsien Wang, Joon-Young Lee, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2503.08605)  

**Abstract**: While recent advancements in text-to-video diffusion models enable high-quality short video generation from a single prompt, generating real-world long videos in a single pass remains challenging due to limited data and high computational costs. To address this, several works propose tuning-free approaches, i.e., extending existing models for long video generation, specifically using multiple prompts to allow for dynamic and controlled content changes. However, these methods primarily focus on ensuring smooth transitions between adjacent frames, often leading to content drift and a gradual loss of semantic coherence over longer sequences. To tackle such an issue, we propose Synchronized Coupled Sampling (SynCoS), a novel inference framework that synchronizes denoising paths across the entire video, ensuring long-range consistency across both adjacent and distant frames. Our approach combines two complementary sampling strategies: reverse and optimization-based sampling, which ensure seamless local transitions and enforce global coherence, respectively. However, directly alternating between these samplings misaligns denoising trajectories, disrupting prompt guidance and introducing unintended content changes as they operate independently. To resolve this, SynCoS synchronizes them through a grounded timestep and a fixed baseline noise, ensuring fully coupled sampling with aligned denoising paths. Extensive experiments show that SynCoS significantly improves multi-event long video generation, achieving smoother transitions and superior long-range coherence, outperforming previous approaches both quantitatively and qualitatively. 

---
# EMMOE: A Comprehensive Benchmark for Embodied Mobile Manipulation in Open Environments 

**Authors**: Dongping Li, Tielong Cai, Tianci Tang, Wenhao Chai, Katherine Rose Driggs-Campbell, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08604)  

**Abstract**: Developing autonomous home robots controlled by natural language has long been a pursuit of human. While advancements in large language models (LLMs) and embodied intelligence make this goal closer, several challenges persist: the lack of a unified benchmark for more complex robot tasks, limited evaluation methods and metrics, data incompatibility between LLMs and mobile manipulation trajectories. To address these issues, we introduce Embodied Mobile Manipulation in Open Environments (EMMOE), which requires agents to interpret user instructions and execute long-horizon everyday tasks in continuous space. EMMOE seamlessly integrates high-level and low-level embodied tasks into a unified framework, along with three new metrics for more diverse assessment. Additionally, we collect EMMOE-100, which features in various task attributes, detailed process annotations, re-plans after failures, and two sub-datasets for LLM training. Furthermore, we design HomieBot, a sophisticated agent system consists of LLM with Direct Preference Optimization (DPO), light weighted navigation and manipulation models, and multiple error detection mechanisms. Finally, we demonstrate HomieBot's performance and the evaluation of different models and policies. 

---
# BiasEdit: Debiasing Stereotyped Language Models via Model Editing 

**Authors**: Xin Xu, Wei Xu, Ningyu Zhang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2503.08588)  

**Abstract**: Previous studies have established that language models manifest stereotyped biases. Existing debiasing strategies, such as retraining a model with counterfactual data, representation projection, and prompting often fail to efficiently eliminate bias or directly alter the models' biased internal representations. To address these issues, we propose BiasEdit, an efficient model editing method to remove stereotypical bias from language models through lightweight networks that act as editors to generate parameter updates. BiasEdit employs a debiasing loss guiding editor networks to conduct local edits on partial parameters of a language model for debiasing while preserving the language modeling abilities during editing through a retention loss. Experiments on StereoSet and Crows-Pairs demonstrate the effectiveness, efficiency, and robustness of BiasEdit in eliminating bias compared to tangental debiasing baselines and little to no impact on the language models' general capabilities. In addition, we conduct bias tracing to probe bias in various modules and explore bias editing impacts on different components of language models. 

---
# MsaMIL-Net: An End-to-End Multi-Scale Aware Multiple Instance Learning Network for Efficient Whole Slide Image Classification 

**Authors**: Jiangping Wen, Jinyu Wen, Emei Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08581)  

**Abstract**: Bag-based Multiple Instance Learning (MIL) approaches have emerged as the mainstream methodology for Whole Slide Image (WSI) classification. However, most existing methods adopt a segmented training strategy, which first extracts features using a pre-trained feature extractor and then aggregates these features through MIL. This segmented training approach leads to insufficient collaborative optimization between the feature extraction network and the MIL network, preventing end-to-end joint optimization and thereby limiting the overall performance of the model. Additionally, conventional methods typically extract features from all patches of fixed size, ignoring the multi-scale observation characteristics of pathologists. This not only results in significant computational resource waste when tumor regions represent a minimal proportion (as in the Camelyon16 dataset) but may also lead the model to suboptimal solutions.
To address these limitations, this paper proposes an end-to-end multi-scale WSI classification framework that integrates multi-scale feature extraction with multiple instance learning. Specifically, our approach includes: (1) a semantic feature filtering module to reduce interference from non-lesion areas; (2) a multi-scale feature extraction module to capture pathological information at different levels; and (3) a multi-scale fusion MIL module for global modeling and feature integration. Through an end-to-end training strategy, we simultaneously optimize both the feature extractor and MIL network, ensuring maximum compatibility between them.
Experiments were conducted on three cross-center datasets (DigestPath2019, BCNB, and UBC-OCEAN). Results demonstrate that our proposed method outperforms existing state-of-the-art approaches in terms of both accuracy (ACC) and AUC metrics. 

---
# When Discourse Stalls: Moving Past Five Semantic Stopsigns about Generative AI in Design Research 

**Authors**: Willem van der Maden, Vera van der Burg, Brett A. Halperin, Petra Jääskeläinen, Joseph Lindley, Derek Lomas, Timothy Merritt  

**Link**: [PDF](https://arxiv.org/pdf/2503.08565)  

**Abstract**: This essay examines how Generative AI (GenAI) is rapidly transforming design practices and how discourse often falls into over-simplified narratives that impede meaningful research and practical progress. We identify and deconstruct five prevalent "semantic stopsigns" -- reductive framings about GenAI in design that halt deeper inquiry and limit productive engagement. Reflecting upon two expert workshops at ACM conferences and semi-structured interviews with design practitioners, we analyze how these stopsigns manifest in research and practice. Our analysis develops mid-level knowledge that bridges theoretical discourse and practical implementation, helping designers and researchers interrogate common assumptions about GenAI in their own contexts. By recasting these stopsigns into more nuanced frameworks, we provide the design research community with practical approaches for thinking about and working with these emerging technologies. 

---
# MoE-Loco: Mixture of Experts for Multitask Locomotion 

**Authors**: Runhan Huang, Shaoting Zhu, Yilun Du, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08564)  

**Abstract**: We present MoE-Loco, a Mixture of Experts (MoE) framework for multitask locomotion for legged robots. Our method enables a single policy to handle diverse terrains, including bars, pits, stairs, slopes, and baffles, while supporting quadrupedal and bipedal gaits. Using MoE, we mitigate the gradient conflicts that typically arise in multitask reinforcement learning, improving both training efficiency and performance. Our experiments demonstrate that different experts naturally specialize in distinct locomotion behaviors, which can be leveraged for task migration and skill composition. We further validate our approach in both simulation and real-world deployment, showcasing its robustness and adaptability. 

---
# Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies 

**Authors**: Chen Xu, Tony Khuong Nguyen, Emma Dixon, Christopher Rodriguez, Patrick Miller, Robert Lee, Paarth Shah, Rares Ambrus, Haruki Nishimura, Masha Itkina  

**Link**: [PDF](https://arxiv.org/pdf/2503.08558)  

**Abstract**: Recent years have witnessed impressive robotic manipulation systems driven by advances in imitation learning and generative modeling, such as diffusion- and flow-based approaches. As robot policy performance increases, so does the complexity and time horizon of achievable tasks, inducing unexpected and diverse failure modes that are difficult to predict a priori. To enable trustworthy policy deployment in safety-critical human environments, reliable runtime failure detection becomes important during policy inference. However, most existing failure detection approaches rely on prior knowledge of failure modes and require failure data during training, which imposes a significant challenge in practicality and scalability. In response to these limitations, we present FAIL-Detect, a modular two-stage approach for failure detection in imitation learning-based robotic manipulation. To accurately identify failures from successful training data alone, we frame the problem as sequential out-of-distribution (OOD) detection. We first distill policy inputs and outputs into scalar signals that correlate with policy failures and capture epistemic uncertainty. FAIL-Detect then employs conformal prediction (CP) as a versatile framework for uncertainty quantification with statistical guarantees. Empirically, we thoroughly investigate both learned and post-hoc scalar signal candidates on diverse robotic manipulation tasks. Our experiments show learned signals to be mostly consistently effective, particularly when using our novel flow-based density estimator. Furthermore, our method detects failures more accurately and faster than state-of-the-art (SOTA) failure detection baselines. These results highlight the potential of FAIL-Detect to enhance the safety and reliability of imitation learning-based robotic systems as they progress toward real-world deployment. 

---
# DAFE: LLM-Based Evaluation Through Dynamic Arbitration for Free-Form Question-Answering 

**Authors**: Sher Badshah, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2503.08542)  

**Abstract**: Evaluating Large Language Models (LLMs) free-form generated responses remains a challenge due to their diverse and open-ended nature. Traditional supervised signal-based automatic metrics fail to capture semantic equivalence or handle the variability of open-ended responses, while human evaluation, though reliable, is resource-intensive. Leveraging LLMs as evaluators offers a promising alternative due to their strong language understanding and instruction-following capabilities. Taking advantage of these capabilities, we propose the Dynamic Arbitration Framework for Evaluation (DAFE), which employs two primary LLM-as-judges and engages a third arbitrator only in cases of disagreements. This selective arbitration prioritizes evaluation reliability while reducing unnecessary computational demands compared to conventional majority voting. DAFE utilizes task-specific reference answers with dynamic arbitration to enhance judgment accuracy, resulting in significant improvements in evaluation metrics such as Macro F1 and Cohen's Kappa. Through experiments, including a comprehensive human evaluation, we demonstrate DAFE's ability to provide consistent, scalable, and resource-efficient assessments, establishing it as a robust framework for evaluating free-form model outputs. 

---
# Mellow: a small audio language model for reasoning 

**Authors**: Soham Deshmukh, Satvik Dixit, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2503.08540)  

**Abstract**: Multimodal Audio-Language Models (ALMs) can understand and reason over both audio and text. Typically, reasoning performance correlates with model size, with the best results achieved by models exceeding 8 billion parameters. However, no prior work has explored enabling small audio-language models to perform reasoning tasks, despite the potential applications for edge devices. To address this gap, we introduce Mellow, a small Audio-Language Model specifically designed for reasoning. Mellow achieves state-of-the-art performance among existing small audio-language models and surpasses several larger models in reasoning capabilities. For instance, Mellow scores 52.11 on MMAU, comparable to SoTA Qwen2 Audio (which scores 52.5) while using 50 times fewer parameters and being trained on 60 times less data (audio hrs). To train Mellow, we introduce ReasonAQA, a dataset designed to enhance audio-grounded reasoning in models. It consists of a mixture of existing datasets (30% of the data) and synthetically generated data (70%). The synthetic dataset is derived from audio captioning datasets, where Large Language Models (LLMs) generate detailed and multiple-choice questions focusing on audio events, objects, acoustic scenes, signal properties, semantics, and listener emotions. To evaluate Mellow's reasoning ability, we benchmark it on a diverse set of tasks, assessing on both in-distribution and out-of-distribution data, including audio understanding, deductive reasoning, and comparative reasoning. Finally, we conduct extensive ablation studies to explore the impact of projection layer choices, synthetic data generation methods, and language model pretraining on reasoning performance. Our training dataset, findings, and baseline pave the way for developing small ALMs capable of reasoning. 

---
# GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training 

**Authors**: Tong Wei, Yijun Yang, Junliang Xing, Yuanchun Shi, Zongqing Lu, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.08525)  

**Abstract**: Reinforcement learning with verifiable outcome rewards (RLVR) has effectively scaled up chain-of-thought (CoT) reasoning in large language models (LLMs). Yet, its efficacy in training vision-language model (VLM) agents for goal-directed action reasoning in visual environments is less established. This work investigates this problem through extensive experiments on complex card games, such as 24 points, and embodied tasks from ALFWorld. We find that when rewards are based solely on action outcomes, RL fails to incentivize CoT reasoning in VLMs, instead leading to a phenomenon we termed thought collapse, characterized by a rapid loss of diversity in the agent's thoughts, state-irrelevant and incomplete reasoning, and subsequent invalid actions, resulting in negative rewards. To counteract thought collapse, we highlight the necessity of process guidance and propose an automated corrector that evaluates and refines the agent's reasoning at each RL step. This simple and scalable GTR (Guided Thought Reinforcement) framework trains reasoning and action simultaneously without the need for dense, per-step human labeling. Our experiments demonstrate that GTR significantly enhances the performance and generalization of the LLaVA-7b model across various visual environments, achieving 3-5 times higher task success rates compared to SoTA models with notably smaller model sizes. 

---
# A Triple-Inertial Accelerated Alternating Optimization Method for Deep Learning Training 

**Authors**: Chengcheng Yan, Jiawei Xu, Qingsong Wang, Zheng Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.08489)  

**Abstract**: The stochastic gradient descent (SGD) algorithm has achieved remarkable success in training deep learning models. However, it has several limitations, including susceptibility to vanishing gradients, sensitivity to input data, and a lack of robust theoretical guarantees. In recent years, alternating minimization (AM) methods have emerged as a promising alternative for model training by employing gradient-free approaches to iteratively update model parameters. Despite their potential, these methods often exhibit slow convergence rates. To address this challenge, we propose a novel Triple-Inertial Accelerated Alternating Minimization (TIAM) framework for neural network training. The TIAM approach incorporates a triple-inertial acceleration strategy with a specialized approximation method, facilitating targeted acceleration of different terms in each sub-problem optimization. This integration improves the efficiency of convergence, achieving superior performance with fewer iterations. Additionally, we provide a convergence analysis of the TIAM algorithm, including its global convergence properties and convergence rate. Extensive experiments validate the effectiveness of the TIAM method, showing significant improvements in generalization capability and computational efficiency compared to existing approaches, particularly when applied to the rectified linear unit (ReLU) and its variants. 

---
# Optimizing Ride-Pooling Operations with Extended Pickup and Drop-Off Flexibility 

**Authors**: Hao Jiang, Yixing Xu, Pradeep Varakantham  

**Link**: [PDF](https://arxiv.org/pdf/2503.08472)  

**Abstract**: The Ride-Pool Matching Problem (RMP) is central to on-demand ride-pooling services, where vehicles must be matched with multiple requests while adhering to service constraints such as pickup delays, detour limits, and vehicle capacity. Most existing RMP solutions assume passengers are picked up and dropped off at their original locations, neglecting the potential for passengers to walk to nearby spots to meet vehicles. This assumption restricts the optimization potential in ride-pooling operations. In this paper, we propose a novel matching method that incorporates extended pickup and drop-off areas for passengers. We first design a tree-based approach to efficiently generate feasible matches between passengers and vehicles. Next, we optimize vehicle routes to cover all designated pickup and drop-off locations while minimizing total travel distance. Finally, we employ dynamic assignment strategies to achieve optimal matching outcomes. Experiments on city-scale taxi datasets demonstrate that our method improves the number of served requests by up to 13\% and average travel distance by up to 21\% compared to leading existing solutions, underscoring the potential of leveraging passenger mobility to significantly enhance ride-pooling service efficiency. 

---
# Accelerating MoE Model Inference with Expert Sharding 

**Authors**: Oana Balmau, Anne-Marie Kermarrec, Rafael Pires, André Loureiro Espírito Santo, Martijn de Vos, Milos Vujasinovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.08467)  

**Abstract**: Mixture of experts (MoE) models achieve state-of-the-art results in language modeling but suffer from inefficient hardware utilization due to imbalanced token routing and communication overhead. While prior work has focused on optimizing MoE training and decoder architectures, inference for encoder-based MoE models in a multi-GPU with expert parallelism setting remains underexplored. We introduce MoEShard, an inference system that achieves perfect load balancing through tensor sharding of MoE experts. Unlike existing approaches that rely on heuristic capacity factors or drop tokens, MoEShard evenly distributes computation across GPUs and ensures full token retention, maximizing utilization regardless of routing skewness. We achieve this through a strategic row- and column-wise decomposition of expert matrices. This reduces idle time and avoids bottlenecks caused by imbalanced expert assignments. Furthermore, MoEShard minimizes kernel launches by fusing decomposed expert computations, significantly improving throughput. We evaluate MoEShard against DeepSpeed on encoder-based architectures, demonstrating speedups of up to 6.4$\times$ in time to first token (TTFT). Our results show that tensor sharding, when properly applied to experts, is a viable and effective strategy for efficient MoE inference. 

---
# Status and Future Prospects of the Standardization Framework Industry 4.0: A European Perspective 

**Authors**: Olga Meyer, Marvin Boell, Christoph Legat  

**Link**: [PDF](https://arxiv.org/pdf/2503.08460)  

**Abstract**: The rapid development of Industry 4.0 technologies requires robust and comprehensive standardization to ensure interoperability, safety and efficiency in the Industry of the Future. This paper examines the fundamental role and functionality of standardization, with a particular focus on its importance in Europe's regulatory framework. Based on this, selected topics in context of standardization activities in context intelligent manufacturing and digital twins are highlighted and, by that, an overview of the Industry 4.0 standards framework is provided. This paper serves both as an informative guide to the existing standards in Industry 4.0 with respect to Artificial Intelligence and Digital Twins, and as a call to action for increased cooperation between standardization bodies and the research community. By fostering such collaboration, we aim to facilitate the continued development and implementation of standards that will drive innovation and progress in the manufacturing sector. 

---
# Controlling Latent Diffusion Using Latent CLIP 

**Authors**: Jason Becker, Chris Wendler, Peter Baylies, Robert West, Christian Wressnegger  

**Link**: [PDF](https://arxiv.org/pdf/2503.08455)  

**Abstract**: Instead of performing text-conditioned denoising in the image domain, latent diffusion models (LDMs) operate in latent space of a variational autoencoder (VAE), enabling more efficient processing at reduced computational costs. However, while the diffusion process has moved to the latent space, the contrastive language-image pre-training (CLIP) models, as used in many image processing tasks, still operate in pixel space. Doing so requires costly VAE-decoding of latent images before they can be processed. In this paper, we introduce Latent-CLIP, a CLIP model that operates directly in the latent space. We train Latent-CLIP on 2.7B pairs of latent images and descriptive texts, and show that it matches zero-shot classification performance of similarly sized CLIP models on both the ImageNet benchmark and a LDM-generated version of it, demonstrating its effectiveness in assessing both real and generated content. Furthermore, we construct Latent-CLIP rewards for reward-based noise optimization (ReNO) and show that they match the performance of their CLIP counterparts on GenEval and T2I-CompBench while cutting the cost of the total pipeline by 21%. Finally, we use Latent-CLIP to guide generation away from harmful content, achieving strong performance on the inappropriate image prompts (I2P) benchmark and a custom evaluation, without ever requiring the costly step of decoding intermediate images. 

---
# ICPR 2024 Competition on Rider Intention Prediction 

**Authors**: Shankar Gangisetty, Abdul Wasi, Shyam Nandan Rai, C. V. Jawahar, Sajay Raj, Manish Prajapati, Ayesha Choudhary, Aaryadev Chandra, Dev Chandan, Shireen Chand, Suvaditya Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.08437)  

**Abstract**: The recent surge in the vehicle market has led to an alarming increase in road accidents. This underscores the critical importance of enhancing road safety measures, particularly for vulnerable road users like motorcyclists. Hence, we introduce the rider intention prediction (RIP) competition that aims to address challenges in rider safety by proactively predicting maneuvers before they occur, thereby strengthening rider safety. This capability enables the riders to react to the potential incorrect maneuvers flagged by advanced driver assistance systems (ADAS). We collect a new dataset, namely, rider action anticipation dataset (RAAD) for the competition consisting of two tasks: single-view RIP and multi-view RIP. The dataset incorporates a spectrum of traffic conditions and challenging navigational maneuvers on roads with varying lighting conditions. For the competition, we received seventy-five registrations and five team submissions for inference of which we compared the methods of the top three performing teams on both the RIP tasks: one state-space model (Mamba2) and two learning-based approaches (SVM and CNN-LSTM). The results indicate that the state-space model outperformed the other methods across the entire dataset, providing a balanced performance across maneuver classes. The SVM-based RIP method showed the second-best performance when using random sampling and SMOTE. However, the CNN-LSTM method underperformed, primarily due to class imbalance issues, particularly struggling with minority classes. This paper details the proposed RAAD dataset and provides a summary of the submissions for the RIP 2024 competition. 

---
# AnyMoLe: Any Character Motion In-betweening Leveraging Video Diffusion Models 

**Authors**: Kwan Yun, Seokhyeon Hong, Chaelin Kim, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.08417)  

**Abstract**: Despite recent advancements in learning-based motion in-betweening, a key limitation has been overlooked: the requirement for character-specific datasets. In this work, we introduce AnyMoLe, a novel method that addresses this limitation by leveraging video diffusion models to generate motion in-between frames for arbitrary characters without external data. Our approach employs a two-stage frame generation process to enhance contextual understanding. Furthermore, to bridge the domain gap between real-world and rendered character animations, we introduce ICAdapt, a fine-tuning technique for video diffusion models. Additionally, we propose a ``motion-video mimicking'' optimization technique, enabling seamless motion generation for characters with arbitrary joint structures using 2D and 3D-aware features. AnyMoLe significantly reduces data dependency while generating smooth and realistic transitions, making it applicable to a wide range of motion in-betweening tasks. 

---
# V-Max: Making RL practical for Autonomous Driving 

**Authors**: Valentin Charraut, Thomas Tournaire, Waël Doulazmi, Thibault Buhet  

**Link**: [PDF](https://arxiv.org/pdf/2503.08388)  

**Abstract**: Learning-based decision-making has the potential to enable generalizable Autonomous Driving (AD) policies, reducing the engineering overhead of rule-based approaches. Imitation Learning (IL) remains the dominant paradigm, benefiting from large-scale human demonstration datasets, but it suffers from inherent limitations such as distribution shift and imitation gaps. Reinforcement Learning (RL) presents a promising alternative, yet its adoption in AD remains limited due to the lack of standardized and efficient research frameworks. To this end, we introduce V-Max, an open research framework providing all the necessary tools to make RL practical for AD. V-Max is built on Waymax, a hardware-accelerated AD simulator designed for large-scale experimentation. We extend it using ScenarioNet's approach, enabling the fast simulation of diverse AD datasets. V-Max integrates a set of observation and reward functions, transformer-based encoders, and training pipelines. Additionally, it includes adversarial evaluation settings and an extensive set of evaluation metrics. Through a large-scale benchmark, we analyze how network architectures, observation functions, training data, and reward shaping impact RL performance. 

---
# InfluenceNet: AI Models for Banzhaf and Shapley Value Prediction 

**Authors**: Benjamin Kempinski, Tal Kachman  

**Link**: [PDF](https://arxiv.org/pdf/2503.08381)  

**Abstract**: Power indices are essential in assessing the contribution and influence of individual agents in multi-agent systems, providing crucial insights into collaborative dynamics and decision-making processes. While invaluable, traditional computational methods for exact or estimated power indices values require significant time and computational constraints, especially for large $(n\ge10)$ coalitions. These constraints have historically limited researchers' ability to analyse complex multi-agent interactions comprehensively. To address this limitation, we introduce a novel Neural Networks-based approach that efficiently estimates power indices for voting games, demonstrating comparable and often superiour performance to existing tools in terms of both speed and accuracy. This method not only addresses existing computational bottlenecks, but also enables rapid analysis of large coalitions, opening new avenues for multi-agent system research by overcoming previous computational limitations and providing researchers with a more accessible, scalable analytical this http URL increased efficiency will allow for the analysis of more complex and realistic multi-agent scenarios. 

---
# Robust Latent Matters: Boosting Image Generation with Sampling Error 

**Authors**: Kai Qiu, Xiang Li, Jason Kuen, Hao Chen, Xiaohao Xu, Jiuxiang Gu, Yinyi Luo, Bhiksha Raj, Zhe Lin, Marios Savvides  

**Link**: [PDF](https://arxiv.org/pdf/2503.08354)  

**Abstract**: Recent image generation schemes typically capture image distribution in a pre-constructed latent space relying on a frozen image tokenizer. Though the performance of tokenizer plays an essential role to the successful generation, its current evaluation metrics (e.g. rFID) fail to precisely assess the tokenizer and correlate its performance to the generation quality (e.g. gFID). In this paper, we comprehensively analyze the reason for the discrepancy of reconstruction and generation qualities in a discrete latent space, and, from which, we propose a novel plug-and-play tokenizer training scheme to facilitate latent space construction. Specifically, a latent perturbation approach is proposed to simulate sampling noises, i.e., the unexpected tokens sampled, from the generative process. With the latent perturbation, we further propose (1) a novel tokenizer evaluation metric, i.e., pFID, which successfully correlates the tokenizer performance to generation quality and (2) a plug-and-play tokenizer training scheme, which significantly enhances the robustness of tokenizer thus boosting the generation quality and convergence speed. Extensive benchmarking are conducted with 11 advanced discrete image tokenizers with 2 autoregressive generation models to validate our approach. The tokenizer trained with our proposed latent perturbation achieve a notable 1.60 gFID with classifier-free guidance (CFG) and 3.45 gFID without CFG with a $\sim$400M generator. Code: this https URL. 

---
# MINT-Demo: Membership Inference Test Demonstrator 

**Authors**: Daniel DeAlcala, Aythami Morales, Julian Fierrez, Gonzalo Mancera, Ruben Tolosana, Ruben Vera-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2503.08332)  

**Abstract**: We present the Membership Inference Test Demonstrator, to emphasize the need for more transparent machine learning training processes. MINT is a technique for experimentally determining whether certain data has been used during the training of machine learning models. We conduct experiments with popular face recognition models and 5 public databases containing over 22M images. Promising results, up to 89% accuracy are achieved, suggesting that it is possible to recognize if an AI model has been trained with specific data. Finally, we present a MINT platform as demonstrator of this technology aimed to promote transparency in AI training. 

---
# Adding Chocolate to Mint: Mitigating Metric Interference in Machine Translation 

**Authors**: José Pombal, Nuno M. Guerreiro, Ricardo Rei, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2503.08327)  

**Abstract**: As automatic metrics become increasingly stronger and widely adopted, the risk of unintentionally "gaming the metric" during model development rises. This issue is caused by metric interference (Mint), i.e., the use of the same or related metrics for both model tuning and evaluation. Mint can misguide practitioners into being overoptimistic about the performance of their systems: as system outputs become a function of the interfering metric, their estimated quality loses correlation with human judgments. In this work, we analyze two common cases of Mint in machine translation-related tasks: filtering of training data, and decoding with quality signals. Importantly, we find that Mint strongly distorts instance-level metric scores, even when metrics are not directly optimized for -- questioning the common strategy of leveraging a different, yet related metric for evaluation that is not used for tuning. To address this problem, we propose MintAdjust, a method for more reliable evaluation under Mint. On the WMT24 MT shared task test set, MintAdjust ranks translations and systems more accurately than state-of-the-art-metrics across a majority of language pairs, especially for high-quality systems. Furthermore, MintAdjust outperforms AutoRank, the ensembling method used by the organizers. 

---
# Prototype-based Heterogeneous Federated Learning for Blade Icing Detection in Wind Turbines with Class Imbalanced Data 

**Authors**: Lele Qi, Mengna Liu, Xu Cheng, Fan Shi, Xiufeng Liu, Shengyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08325)  

**Abstract**: Wind farms, typically in high-latitude regions, face a high risk of blade icing. Traditional centralized training methods raise serious privacy concerns. To enhance data privacy in detecting wind turbine blade icing, traditional federated learning (FL) is employed. However, data heterogeneity, resulting from collections across wind farms in varying environmental conditions, impacts the model's optimization capabilities. Moreover, imbalances in wind turbine data lead to models that tend to favor recognizing majority classes, thus neglecting critical icing anomalies. To tackle these challenges, we propose a federated prototype learning model for class-imbalanced data in heterogeneous environments to detect wind turbine blade icing. We also propose a contrastive supervised loss function to address the class imbalance problem. Experiments on real data from 20 turbines across two wind farms show our method outperforms five FL models and five class imbalance methods, with an average improvement of 19.64\% in \( mF_{\beta} \) and 5.73\% in \( m \)BA compared to the second-best method, BiFL. 

---
# Evaluating Interpretable Reinforcement Learning by Distilling Policies into Programs 

**Authors**: Hector Kohler, Quentin Delfosse, Waris Radji, Riad Akrour, Philippe Preux  

**Link**: [PDF](https://arxiv.org/pdf/2503.08322)  

**Abstract**: There exist applications of reinforcement learning like medicine where policies need to be ''interpretable'' by humans. User studies have shown that some policy classes might be more interpretable than others. However, it is costly to conduct human studies of policy interpretability. Furthermore, there is no clear definition of policy interpretabiliy, i.e., no clear metrics for interpretability and thus claims depend on the chosen definition. We tackle the problem of empirically evaluating policies interpretability without humans. Despite this lack of clear definition, researchers agree on the notions of ''simulatability'': policy interpretability should relate to how humans understand policy actions given states. To advance research in interpretable reinforcement learning, we contribute a new methodology to evaluate policy interpretability. This new methodology relies on proxies for simulatability that we use to conduct a large-scale empirical evaluation of policy interpretability. We use imitation learning to compute baseline policies by distilling expert neural networks into small programs. We then show that using our methodology to evaluate the baselines interpretability leads to similar conclusions as user studies. We show that increasing interpretability does not necessarily reduce performances and can sometimes increase them. We also show that there is no policy class that better trades off interpretability and performance across tasks making it necessary for researcher to have methodologies for comparing policies interpretability. 

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
# D3PO: Preference-Based Alignment of Discrete Diffusion Models 

**Authors**: Umberto Borso, Davide Paglieri, Jude Wells, Tim Rocktäschel  

**Link**: [PDF](https://arxiv.org/pdf/2503.08295)  

**Abstract**: Diffusion models have achieved state-of-the-art performance across multiple domains, with recent advancements extending their applicability to discrete data. However, aligning discrete diffusion models with task-specific preferences remains challenging, particularly in scenarios where explicit reward functions are unavailable. In this work, we introduce Discrete Diffusion DPO (D3PO), the first adaptation of Direct Preference Optimization (DPO) to discrete diffusion models formulated as continuous-time Markov chains. Our approach derives a novel loss function that directly fine-tunes the generative process using preference data while preserving fidelity to a reference distribution. We validate D3PO on a structured binary sequence generation task, demonstrating that the method effectively aligns model outputs with preferences while maintaining structural validity. Our results highlight that D3PO enables controlled fine-tuning without requiring explicit reward models, making it a practical alternative to reinforcement learning-based approaches. Future research will explore extending D3PO to more complex generative tasks, including language modeling and protein sequence generation, as well as investigating alternative noise schedules, such as uniform noising, to enhance flexibility across different applications. 

---
# Large Language Models for Outpatient Referral: Problem Definition, Benchmarking and Challenges 

**Authors**: Xiaoxiao Liu, Qingying Xiao, Junying Chen, Xiangyi Feng, Xiangbo Wu, Bairui Zhang, Xiang Wan, Jian Chang, Guangjun Yu, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08292)  

**Abstract**: Large language models (LLMs) are increasingly applied to outpatient referral tasks across healthcare systems. However, there is a lack of standardized evaluation criteria to assess their effectiveness, particularly in dynamic, interactive scenarios. In this study, we systematically examine the capabilities and limitations of LLMs in managing tasks within Intelligent Outpatient Referral (IOR) systems and propose a comprehensive evaluation framework specifically designed for such systems. This framework comprises two core tasks: static evaluation, which focuses on evaluating the ability of predefined outpatient referrals, and dynamic evaluation, which evaluates capabilities of refining outpatient referral recommendations through iterative dialogues. Our findings suggest that LLMs offer limited advantages over BERT-like models, but show promise in asking effective questions during interactive dialogues. 

---
# OminiControl2: Efficient Conditioning for Diffusion Transformers 

**Authors**: Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08280)  

**Abstract**: Fine-grained control of text-to-image diffusion transformer models (DiT) remains a critical challenge for practical deployment. While recent advances such as OminiControl and others have enabled a controllable generation of diverse control signals, these methods face significant computational inefficiency when handling long conditional inputs. We present OminiControl2, an efficient framework that achieves efficient image-conditional image generation. OminiControl2 introduces two key innovations: (1) a dynamic compression strategy that streamlines conditional inputs by preserving only the most semantically relevant tokens during generation, and (2) a conditional feature reuse mechanism that computes condition token features only once and reuses them across denoising steps. These architectural improvements preserve the original framework's parameter efficiency and multi-modal versatility while dramatically reducing computational costs. Our experiments demonstrate that OminiControl2 reduces conditional processing overhead by over 90% compared to its predecessor, achieving an overall 5.9$\times$ speedup in multi-conditional generation scenarios. This efficiency enables the practical implementation of complex, multi-modal control for high-quality image synthesis with DiT models. 

---
# Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks 

**Authors**: Junying Wang, Hongyuan Zhang, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08269)  

**Abstract**: Recent Customized Portrait Generation (CPG) methods, taking a facial image and a textual prompt as inputs, have attracted substantial attention. Although these methods generate high-fidelity portraits, they fail to prevent the generated portraits from being tracked and misused by malicious face recognition systems. To address this, this paper proposes a Customized Portrait Generation framework with facial Adversarial attacks (Adv-CPG). Specifically, to achieve facial privacy protection, we devise a lightweight local ID encryptor and an encryption enhancer. They implement progressive double-layer encryption protection by directly injecting the target identity and adding additional identity guidance, respectively. Furthermore, to accomplish fine-grained and personalized portrait generation, we develop a multi-modal image customizer capable of generating controlled fine-grained facial features. To the best of our knowledge, Adv-CPG is the first study that introduces facial adversarial attacks into CPG. Extensive experiments demonstrate the superiority of Adv-CPG, e.g., the average attack success rate of the proposed Adv-CPG is 28.1% and 2.86% higher compared to the SOTA noise-based attack methods and unconstrained attack methods, respectively. 

---
# DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness 

**Authors**: Yiming Zhong, Qi Jiang, Jingyi Yu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.08257)  

**Abstract**: A dexterous hand capable of grasping any object is essential for the development of general-purpose embodied intelligent robots. However, due to the high degree of freedom in dexterous hands and the vast diversity of objects, generating high-quality, usable grasping poses in a robust manner is a significant challenge. In this paper, we introduce DexGrasp Anything, a method that effectively integrates physical constraints into both the training and sampling phases of a diffusion-based generative model, achieving state-of-the-art performance across nearly all open datasets. Additionally, we present a new dexterous grasping dataset containing over 3.4 million diverse grasping poses for more than 15k different objects, demonstrating its potential to advance universal dexterous grasping. The code of our method and our dataset will be publicly released soon. 

---
# MT-NAM: An Efficient and Adaptive Model for Epileptic Seizure Detection 

**Authors**: Arshia Afzal, Volkan Cevher, Mahsa Shoaran  

**Link**: [PDF](https://arxiv.org/pdf/2503.08251)  

**Abstract**: Enhancing the accuracy and efficiency of machine learning algorithms employed in neural interface systems is crucial for advancing next-generation intelligent therapeutic devices. However, current systems often utilize basic machine learning models that do not fully exploit the natural structure of brain signals. Additionally, existing learning models used for neural signal processing often demonstrate low speed and efficiency during inference. To address these challenges, this study introduces Micro Tree-based NAM (MT-NAM), a distilled model based on the recently proposed Neural Additive Models (NAM). The MT-NAM achieves a remarkable 100$\times$ improvement in inference speed compared to standard NAM, without compromising accuracy. We evaluate our approach on the CHB-MIT scalp EEG dataset, which includes recordings from 24 patients with varying numbers of sessions and seizures. NAM achieves an 85.3\% window-based sensitivity and 95\% specificity. Interestingly, our proposed MT-NAM shows only a 2\% reduction in sensitivity compared to the original NAM. To regain this sensitivity, we utilize a test-time template adjuster (T3A) as an update mechanism, enabling our model to achieve higher sensitivity during test time by accommodating transient shifts in neural signals. With this online update approach, MT-NAM achieves the same sensitivity as the standard NAM while achieving approximately 50$\times$ acceleration in inference speed. 

---
# Aligning Text to Image in Diffusion Models is Easier Than You Think 

**Authors**: Jaa-Yeon Lee, Byunghee Cha, Jeongsol Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.08250)  

**Abstract**: While recent advancements in generative modeling have significantly improved text-image alignment, some residual misalignment between text and image representations still remains. Although many approaches have attempted to address this issue by fine-tuning models using various reward models, etc., we revisit the challenge from the perspective of representation alignment-an approach that has gained popularity with the success of REPresentation Alignment (REPA). We first argue that conventional text-to-image (T2I) diffusion models, typically trained on paired image and text data (i.e., positive pairs) by minimizing score matching or flow matching losses, is suboptimal from the standpoint of representation alignment. Instead, a better alignment can be achieved through contrastive learning that leverages both positive and negative pairs. To achieve this efficiently even with pretrained models, we introduce a lightweight contrastive fine tuning strategy called SoftREPA that uses soft text tokens. This approach improves alignment with minimal computational overhead by adding fewer than 1M trainable parameters to the pretrained model. Our theoretical analysis demonstrates that our method explicitly increases the mutual information between text and image representations, leading to enhanced semantic consistency. Experimental results across text-to-image generation and text-guided image editing tasks validate the effectiveness of our approach in improving the semantic consistency of T2I generative models. 

---
# Investigating Execution-Aware Language Models for Code Optimization 

**Authors**: Federico Di Menna, Luca Traini, Gabriele Bavota, Vittorio Cortellessa  

**Link**: [PDF](https://arxiv.org/pdf/2503.08228)  

**Abstract**: Code optimization is the process of enhancing code efficiency, while preserving its intended functionality. This process often requires a deep understanding of the code execution behavior at run-time to identify and address inefficiencies effectively. Recent studies have shown that language models can play a significant role in automating code optimization. However, these models may have insufficient knowledge of how code execute at run-time. To address this limitation, researchers have developed strategies that integrate code execution information into language models. These strategies have shown promise, enhancing the effectiveness of language models in various software engineering tasks. However, despite the close relationship between code execution behavior and efficiency, the specific impact of these strategies on code optimization remains largely unexplored. This study investigates how incorporating code execution information into language models affects their ability to optimize code. Specifically, we apply three different training strategies to incorporate four code execution aspects -- line executions, line coverage, branch coverage, and variable states -- into CodeT5+, a well-known language model for code. Our results indicate that execution-aware models provide limited benefits compared to the standard CodeT5+ model in optimizing code. 

---
# A Grey-box Text Attack Framework using Explainable AI 

**Authors**: Esther Chiramal, Kelvin Soh Boon Kai  

**Link**: [PDF](https://arxiv.org/pdf/2503.08226)  

**Abstract**: Explainable AI is a strong strategy implemented to understand complex black-box model predictions in a human interpretable language. It provides the evidence required to execute the use of trustworthy and reliable AI systems. On the other hand, however, it also opens the door to locating possible vulnerabilities in an AI model. Traditional adversarial text attack uses word substitution, data augmentation techniques and gradient-based attacks on powerful pre-trained Bidirectional Encoder Representations from Transformers (BERT) variants to generate adversarial sentences. These attacks are generally whitebox in nature and not practical as they can be easily detected by humans E.g. Changing the word from "Poor" to "Rich". We proposed a simple yet effective Grey-box cum Black-box approach that does not require the knowledge of the model while using a set of surrogate Transformer/BERT models to perform the attack using Explainable AI techniques. As Transformers are the current state-of-the-art models for almost all Natural Language Processing (NLP) tasks, an attack generated from BERT1 is transferable to BERT2. This transferability is made possible due to the attention mechanism in the transformer that allows the model to capture long-range dependencies in a sequence. Using the power of BERT generalisation via attention, we attempt to exploit how transformers learn by attacking a few surrogate transformer variants which are all based on a different architecture. We demonstrate that this approach is highly effective to generate semantically good sentences by changing as little as one word that is not detectable by humans while still fooling other BERT models. 

---
# EgoBlind: Towards Egocentric Visual Assistance for the Blind People 

**Authors**: Junbin Xiao, Nanxin Huang, Hao Qiu, Zhulin Tao, Xun Yang, Richang Hong, Meng Wang, Angela Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08221)  

**Abstract**: We present EgoBlind, the first egocentric VideoQA dataset collected from blind individuals to evaluate the assistive capabilities of contemporary multimodal large language models (MLLMs). EgoBlind comprises 1,210 videos that record the daily lives of real blind users from a first-person perspective. It also features 4,927 questions directly posed or generated and verified by blind individuals to reflect their needs for visual assistance under various scenarios. We provide each question with an average of 3 reference answers to alleviate subjective evaluation. Using EgoBlind, we comprehensively evaluate 15 leading MLLMs and find that all models struggle, with the best performers achieving accuracy around 56\%, far behind human performance of 87.4\%. To guide future advancements, we identify and summarize major limitations of existing MLLMs in egocentric visual assistance for the blind and provide heuristic suggestions for improvement. With these efforts, we hope EgoBlind can serve as a valuable foundation for developing more effective AI assistants to enhance the independence of the blind individuals' lives. 

---
# CL-MVSNet: Unsupervised Multi-view Stereo with Dual-level Contrastive Learning 

**Authors**: Kaiqiang Xiong, Rui Peng, Zhe Zhang, Tianxing Feng, Jianbo Jiao, Feng Gao, Ronggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08219)  

**Abstract**: Unsupervised Multi-View Stereo (MVS) methods have achieved promising progress recently. However, previous methods primarily depend on the photometric consistency assumption, which may suffer from two limitations: indistinguishable regions and view-dependent effects, e.g., low-textured areas and reflections. To address these issues, in this paper, we propose a new dual-level contrastive learning approach, named CL-MVSNet. Specifically, our model integrates two contrastive branches into an unsupervised MVS framework to construct additional supervisory signals. On the one hand, we present an image-level contrastive branch to guide the model to acquire more context awareness, thus leading to more complete depth estimation in indistinguishable regions. On the other hand, we exploit a scene-level contrastive branch to boost the representation ability, improving robustness to view-dependent effects. Moreover, to recover more accurate 3D geometry, we introduce an L0.5 photometric consistency loss, which encourages the model to focus more on accurate points while mitigating the gradient penalty of undesirable ones. Extensive experiments on DTU and Tanks&Temples benchmarks demonstrate that our approach achieves state-of-the-art performance among all end-to-end unsupervised MVS frameworks and outperforms its supervised counterpart by a considerable margin without fine-tuning. 

---
# DeepRAG: Building a Custom Hindi Embedding Model for Retrieval Augmented Generation from Scratch 

**Authors**: Nandakishor M  

**Link**: [PDF](https://arxiv.org/pdf/2503.08213)  

**Abstract**: In this paper, I present our work on DeepRAG, a specialized embedding model we built specifically for Hindi language in RAG systems. While LLMs have gotten really good at generating text, their performance in retrieval tasks still depends heavily on having quality embeddings - something that's been lacking for Hindi despite being one of the world's most spoken languages. We tackled this by creating embeddings from the ground up rather than just fine-tuning existing models. Our process involved collecting diverse Hindi texts (over 2.7M samples), training a custom SentencePiece tokenizer that actually understands Hindi morphology, designing transformer architecture with Hindi-specific attention mechanisms, and optimizing with contrastive learning. Results were honestly better than I expected - we saw a 23% improvement in retrieval precision compared to the multilingual models everyone's been using. The paper details our methodology, which I think could help others working with low-resource languages where the one-size-fits-all multilingual models fall short. We've also integrated our embeddings with LangChain to build complete Hindi RAG systems, which might be useful for practitioners. While there's still tons more to explore, I believe this work addresses a critical gap for Hindi NLP and demonstrates why language-specific approaches matter. 

---
# OLMD: Orientation-aware Long-term Motion Decoupling for Continuous Sign Language Recognition 

**Authors**: Yiheng Yu, Sheng Liu, Yuan Feng, Min Xu, Zhelun Jin, Xuhua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08205)  

**Abstract**: The primary challenge in continuous sign language recognition (CSLR) mainly stems from the presence of multi-orientational and long-term motions. However, current research overlooks these crucial aspects, significantly impacting accuracy. To tackle these issues, we propose a novel CSLR framework: Orientation-aware Long-term Motion Decoupling (OLMD), which efficiently aggregates long-term motions and decouples multi-orientational signals into easily interpretable components. Specifically, our innovative Long-term Motion Aggregation (LMA) module filters out static redundancy while adaptively capturing abundant features of long-term motions. We further enhance orientation awareness by decoupling complex movements into horizontal and vertical components, allowing for motion purification in both orientations. Additionally, two coupling mechanisms are proposed: stage and cross-stage coupling, which together enrich multi-scale features and improve the generalization capabilities of the model. Experimentally, OLMD shows SOTA performance on three large-scale datasets: PHOENIX14, PHOENIX14-T, and CSL-Daily. Notably, we improved the word error rate (WER) on PHOENIX14 by an absolute 1.6% compared to the previous SOTA 

---
# A Cascading Cooperative Multi-agent Framework for On-ramp Merging Control Integrating Large Language Models 

**Authors**: Miao Zhang, Zhenlong Fang, Tianyi Wang, Qian Zhang, Shuai Lu, Junfeng Jiao, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.08199)  

**Abstract**: Traditional Reinforcement Learning (RL) suffers from replicating human-like behaviors, generalizing effectively in multi-agent scenarios, and overcoming inherent interpretability this http URL tasks are compounded when deep environment understanding, agent coordination and dynamic optimization are required. While Large Language Model (LLM) enhanced methods have shown promise in generalization and interoperability, they often neglect necessary multi-agent coordination. Therefore, we introduce the Cascading Cooperative Multi-agent (CCMA) framework, integrating RL for individual interactions, a fine-tuned LLM for regional cooperation, a reward function for global optimization, and the Retrieval-augmented Generation mechanism to dynamically optimize decision-making across complex driving scenarios. Our experiments demonstrate that the CCMA outperforms existing RL methods, demonstrating significant improvements in both micro and macro-level performance in complex driving environments. 

---
# RigoChat 2: an adapted language model to Spanish using a bounded dataset and reduced hardware 

**Authors**: Gonzalo Santamaría Gómez, Guillem García Subies, Pablo Gutiérrez Ruiz, Mario González Valero, Natàlia Fuertes, Helena Montoro Zamorano, Carmen Muñoz Sanz, Leire Rosado Plaza, Nuria Aldama García, David Betancur Sánchez, Kateryna Sushkova, Marta Guerrero Nieto, Álvaro Barbero Jiménez  

**Link**: [PDF](https://arxiv.org/pdf/2503.08188)  

**Abstract**: Large Language Models (LLMs) have become a key element of modern artificial intelligence, demonstrating the ability to address a wide range of language processing tasks at unprecedented levels of accuracy without the need of collecting problem-specific data. However, these versatile models face a significant challenge: both their training and inference processes require substantial computational resources, time, and memory. Consequently, optimizing this kind of models to minimize these requirements is crucial. In this article, we demonstrate that, with minimal resources and in a remarkably short time, it is possible to enhance a state-of-the-art model, specifically for a given language task, without compromising its overall capabilities using a relatively small pretrained LLM as a basis. Specifically, we present our use case, RigoChat 2, illustrating how LLMs can be adapted to achieve superior results in Spanish-language tasks. 

---
# ProTeX: Structure-In-Context Reasoning and Editing of Proteins with Large Language Models 

**Authors**: Zicheng Ma, Chuanliu Fan, Zhicong Wang, Zhenyu Chen, Xiaohan Lin, Yanheng Li, Shihao Feng, Jun Zhang, Ziqiang Cao, Yi Qin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08179)  

**Abstract**: Large language models have made remarkable progress in the field of molecular science, particularly in understanding and generating functional small molecules. This success is largely attributed to the effectiveness of molecular tokenization strategies. In protein science, the amino acid sequence serves as the sole tokenizer for LLMs. However, many fundamental challenges in protein science are inherently structure-dependent. The absence of structure-aware tokens significantly limits the capabilities of LLMs for comprehensive biomolecular comprehension and multimodal generation. To address these challenges, we introduce a novel framework, ProTeX, which tokenizes the protein sequences, structures, and textual information into a unified discrete space. This innovative approach enables joint training of the LLM exclusively through the Next-Token Prediction paradigm, facilitating multimodal protein reasoning and generation. ProTeX enables general LLMs to perceive and process protein structures through sequential text input, leverage structural information as intermediate reasoning components, and generate or manipulate structures via sequential text output. Experiments demonstrate that our model achieves significant improvements in protein function prediction, outperforming the state-of-the-art domain expert model with a twofold increase in accuracy. Our framework enables high-quality conformational generation and customizable protein design. For the first time, we demonstrate that by adopting the standard training and inference pipelines from the LLM domain, ProTeX empowers decoder-only LLMs to effectively address diverse spectrum of protein-related tasks. 

---
# Investigating the Effectiveness of a Socratic Chain-of-Thoughts Reasoning Method for Task Planning in Robotics, A Case Study 

**Authors**: Veronica Bot, Zheyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08174)  

**Abstract**: Large language models (LLMs) have demonstrated unprecedented capability in reasoning with natural language. Coupled with this development is the emergence of embodied AI in robotics. Despite showing promise for verbal and written reasoning tasks, it remains unknown whether LLMs are capable of navigating complex spatial tasks with physical actions in the real world. To this end, it is of interest to investigate applying LLMs to robotics in zero-shot learning scenarios, and in the absence of fine-tuning - a feat which could significantly improve human-robot interaction, alleviate compute cost, and eliminate low-level programming tasks associated with robot tasks.
To explore this question, we apply GPT-4(Omni) with a simulated Tiago robot in Webots engine for an object search task. We evaluate the effectiveness of three reasoning strategies based on Chain-of-Thought (CoT) sub-task list generation with the Socratic method (SocraCoT) (in order of increasing rigor): (1) Non-CoT/Non-SocraCoT, (2) CoT only, and (3) SocraCoT. Performance was measured in terms of the proportion of tasks successfully completed and execution time (N = 20). Our preliminary results show that when combined with chain-of-thought reasoning, the Socratic method can be used for code generation for robotic tasks that require spatial awareness. In extension of this finding, we propose EVINCE-LoC; a modified EVINCE method that could further enhance performance in highly complex and or dynamic testing scenarios. 

---
# XAI4Extremes: An interpretable machine learning framework for understanding extreme-weather precursors under climate change 

**Authors**: Jiawen Wei, Aniruddha Bora, Vivek Oommen, Chenyu Dong, Juntao Yang, Jeff Adie, Chen Chen, Simon See, George Karniadakis, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2503.08163)  

**Abstract**: Extreme weather events are increasing in frequency and intensity due to climate change. This, in turn, is exacting a significant toll in communities worldwide. While prediction skills are increasing with advances in numerical weather prediction and artificial intelligence tools, extreme weather still present challenges. More specifically, identifying the precursors of such extreme weather events and how these precursors may evolve under climate change remain unclear. In this paper, we propose to use post-hoc interpretability methods to construct relevance weather maps that show the key extreme-weather precursors identified by deep learning models. We then compare this machine view with existing domain knowledge to understand whether deep learning models identified patterns in data that may enrich our understanding of extreme-weather precursors. We finally bin these relevant maps into different multi-year time periods to understand the role that climate change is having on these precursors. The experiments are carried out on Indochina heatwaves, but the methodology can be readily extended to other extreme weather events worldwide. 

---
# Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking 

**Authors**: Yunhao Li, Yifan Jiao, Dan Meng, Heng Fan, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08145)  

**Abstract**: Open-Vocabulary Multi-Object Tracking (OV-MOT) aims to enable approaches to track objects without being limited to a predefined set of categories. Current OV-MOT methods typically rely primarily on instance-level detection and association, often overlooking trajectory information that is unique and essential for object tracking tasks. Utilizing trajectory information can enhance association stability and classification accuracy, especially in cases of occlusion and category ambiguity, thereby improving adaptability to novel classes. Thus motivated, in this paper we propose \textbf{TRACT}, an open-vocabulary tracker that leverages trajectory information to improve both object association and classification in OV-MOT. Specifically, we introduce a \textit{Trajectory Consistency Reinforcement} (\textbf{TCR}) strategy, that benefits tracking performance by improving target identity and category consistency. In addition, we present \textbf{TraCLIP}, a plug-and-play trajectory classification module. It integrates \textit{Trajectory Feature Aggregation} (\textbf{TFA}) and \textit{Trajectory Semantic Enrichment} (\textbf{TSE}) strategies to fully leverage trajectory information from visual and language perspectives for enhancing the classification results. Extensive experiments on OV-TAO show that our TRACT significantly improves tracking performance, highlighting trajectory information as a valuable asset for OV-MOT. Code will be released. 

---
# FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems 

**Authors**: Jeongsol Kim, Bryan Sangwoo Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.08136)  

**Abstract**: Flow matching is a recent state-of-the-art framework for generative modeling based on ordinary differential equations (ODEs). While closely related to diffusion models, it provides a more general perspective on generative modeling. Although inverse problem solving has been extensively explored using diffusion models, it has not been rigorously examined within the broader context of flow models. Therefore, here we extend the diffusion inverse solvers (DIS) - which perform posterior sampling by combining a denoising diffusion prior with an likelihood gradient - into the flow framework. Specifically, by driving the flow-version of Tweedie's formula, we decompose the flow ODE into two components: one for clean image estimation and the other for noise estimation. By integrating the likelihood gradient and stochastic noise into each component, respectively, we demonstrate that posterior sampling for inverse problem solving can be effectively achieved using flows. Our proposed solver, Flow-Driven Posterior Sampling (FlowDPS), can also be seamlessly integrated into a latent flow model with a transformer architecture. Across four linear inverse problems, we confirm that FlowDPS outperforms state-of-the-art alternatives, all without requiring additional training. 

---
# MGHanD: Multi-modal Guidance for authentic Hand Diffusion 

**Authors**: Taehyeon Eum, Jieun Choi, Tae-Kyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.08133)  

**Abstract**: Diffusion-based methods have achieved significant successes in T2I generation, providing realistic images from text prompts. Despite their capabilities, these models face persistent challenges in generating realistic human hands, often producing images with incorrect finger counts and structurally deformed hands. MGHanD addresses this challenge by applying multi-modal guidance during the inference process. For visual guidance, we employ a discriminator trained on a dataset comprising paired real and generated images with captions, derived from various hand-in-the-wild datasets. We also employ textual guidance with LoRA adapter, which learns the direction from `hands' towards more detailed prompts such as `natural hands', and `anatomically correct fingers' at the latent level. A cumulative hand mask which is gradually enlarged in the assigned time step is applied to the added guidance, allowing the hand to be refined while maintaining the rich generative capabilities of the pre-trained model. In the experiments, our method achieves superior hand generation qualities, without any specific conditions or priors. We carry out both quantitative and qualitative evaluations, along with user studies, to showcase the benefits of our approach in producing high-quality hand images. 

---
# Toward Stable World Models: Measuring and Addressing World Instability in Generative Environments 

**Authors**: Soonwoo Kwon, Jin-Young Kim, Hyojun Go, Kyungjune Baek  

**Link**: [PDF](https://arxiv.org/pdf/2503.08122)  

**Abstract**: We present a novel study on enhancing the capability of preserving the content in world models, focusing on a property we term World Stability. Recent diffusion-based generative models have advanced the synthesis of immersive and realistic environments that are pivotal for applications such as reinforcement learning and interactive game engines. However, while these models excel in quality and diversity, they often neglect the preservation of previously generated scenes over time--a shortfall that can introduce noise into agent learning and compromise performance in safety-critical settings. In this work, we introduce an evaluation framework that measures world stability by having world models perform a sequence of actions followed by their inverses to return to their initial viewpoint, thereby quantifying the consistency between the starting and ending observations. Our comprehensive assessment of state-of-the-art diffusion-based world models reveals significant challenges in achieving high world stability. Moreover, we investigate several improvement strategies to enhance world stability. Our results underscore the importance of world stability in world modeling and provide actionable insights for future research in this domain. 

---
# Uni$\textbf{F}^2$ace: Fine-grained Face Understanding and Generation with Unified Multimodal Models 

**Authors**: Junzhe Li, Xuerui Qiu, Linrui Xu, Liya Guo, Delin Qu, Tingting Long, Chun Fan, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.08120)  

**Abstract**: Unified multimodal models (UMMs) have emerged as a powerful paradigm in foundational computer vision research, demonstrating significant potential in both image understanding and generation. However, existing research in the face domain primarily focuses on $\textbf{coarse}$ facial attribute understanding, with limited capacity to handle $\textbf{fine-grained}$ facial attributes and without addressing generation capabilities. To overcome these limitations, we propose Uni$\textbf{F}^2$ace, the first UMM tailored specifically for fine-grained face understanding and generation. In general, we train Uni$\textbf{F}^2$ace on a self-constructed, specialized dataset utilizing two mutually beneficial diffusion techniques and a two-level mixture-of-experts architecture. Specifically, we first build a large-scale facial dataset, Uni$\textbf{F}^2$ace-130K, which contains 130K image-text pairs with one million question-answering pairs that span a wide range of facial attributes. Second, we establish a theoretical connection between discrete diffusion score matching and masked generative models, optimizing both evidence lower bounds simultaneously, which significantly improves the model's ability to synthesize facial details. Finally, we introduce both token-level and sequence-level mixture-of-experts, enabling efficient fine-grained representation learning for both understanding and generation tasks. Extensive experiments on Uni$\textbf{F}^2$ace-130K demonstrate that Uni$\textbf{F}^2$ace outperforms existing UMMs and generative models, achieving superior performance across both understanding and generation tasks. 

---
# Convergence Dynamics and Stabilization Strategies of Co-Evolving Generative Models 

**Authors**: Weiguo Gao, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.08117)  

**Abstract**: The increasing prevalence of synthetic data in training loops has raised concerns about model collapse, where generative models degrade when trained on their own outputs. While prior work focuses on this self-consuming process, we study an underexplored yet prevalent phenomenon: co-evolving generative models that shape each other's training through iterative feedback. This is common in multimodal AI ecosystems, such as social media platforms, where text models generate captions that guide image models, and the resulting images influence the future adaptation of the text model. We take a first step by analyzing such a system, modeling the text model as a multinomial distribution and the image model as a conditional multi-dimensional Gaussian distribution. Our analysis uncovers three key results. First, when one model remains fixed, the other collapses: a frozen image model causes the text model to lose diversity, while a frozen text model leads to an exponential contraction of image diversity, though fidelity remains bounded. Second, in fully interactive systems, mutual reinforcement accelerates collapse, with image contraction amplifying text homogenization and vice versa, leading to a Matthew effect where dominant texts sustain higher image diversity while rarer texts collapse faster. Third, we analyze stabilization strategies implicitly introduced by real-world external influences. Random corpus injections for text models and user-content injections for image models prevent collapse while preserving both diversity and fidelity. Our theoretical findings are further validated through experiments. 

---
# Revolution of Wireless Signal Recognition for 6G: Recent Advances, Challenges and Future Directions 

**Authors**: Hao Zhang, Fuhui Zhou, Hongyang Du, Qihui Wu, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08091)  

**Abstract**: Wireless signal recognition (WSR) is a crucial technique for intelligent communications and spectrum sharing in the next six-generation (6G) wireless communication networks. It can be utilized to enhance network performance and efficiency, improve quality of service (QoS), and improve network security and reliability. Additionally, WSR can be applied for military applications such as signal interception, signal race, and signal abduction. In the past decades, great efforts have been made for the research of WSR. Earlier works mainly focus on model-based methods, including likelihood-based (LB) and feature-based (FB) methods, which have taken the leading position for many years. With the emergence of artificial intelligence (AI), intelligent methods including machine learning-based (ML-based) and deep learning-based (DL-based) methods have been developed to extract the features of the received signals and perform the classification. In this work, we provide a comprehensive review of WSR from the view of applications, main tasks, recent advances, datasets and evaluation metrics, challenges, and future directions. Specifically, intelligent WSR methods are introduced from the perspective of model, data, learning and implementation. Moreover, we analyze the challenges for WSR from the view of complex, dynamic, and open 6G wireless environments and discuss the future directions for WSR. This survey is expected to provide a comprehensive overview of the state-of-the-art WSR techniques and inspire new research directions for WSR in 6G networks. 

---
# Instruction-Augmented Long-Horizon Planning: Embedding Grounding Mechanisms in Embodied Mobile Manipulation 

**Authors**: Fangyuan Wang, Shipeng Lyu, Peng Zhou, Anqing Duan, Guodong Guo, David Navarro-Alarcon  

**Link**: [PDF](https://arxiv.org/pdf/2503.08084)  

**Abstract**: Enabling humanoid robots to perform long-horizon mobile manipulation planning in real-world environments based on embodied perception and comprehension abilities has been a longstanding challenge. With the recent rise of large language models (LLMs), there has been a notable increase in the development of LLM-based planners. These approaches either utilize human-provided textual representations of the real world or heavily depend on prompt engineering to extract such representations, lacking the capability to quantitatively understand the environment, such as determining the feasibility of manipulating objects. To address these limitations, we present the Instruction-Augmented Long-Horizon Planning (IALP) system, a novel framework that employs LLMs to generate feasible and optimal actions based on real-time sensor feedback, including grounded knowledge of the environment, in a closed-loop interaction. Distinct from prior works, our approach augments user instructions into PDDL problems by leveraging both the abstract reasoning capabilities of LLMs and grounding mechanisms. By conducting various real-world long-horizon tasks, each consisting of seven distinct manipulatory skills, our results demonstrate that the IALP system can efficiently solve these tasks with an average success rate exceeding 80%. Our proposed method can operate as a high-level planner, equipping robots with substantial autonomy in unstructured environments through the utilization of multi-modal sensor inputs. 

---
# Degradation Self-Supervised Learning for Lithium-ion Battery Health Diagnostics 

**Authors**: J. C. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08083)  

**Abstract**: Health evaluation for lithium-ion batteries (LIBs) typically relies on constant charging/discharging protocols, often neglecting scenarios involving dynamic current profiles prevalent in electric vehicles. Conventional health indicators for LIBs also depend on the uniformity of measured data, restricting their adaptability to non-uniform conditions. In this study, a novel training strategy for estimating LIB health based on the paradigm of self-supervised learning is proposed. A multiresolution analysis technique, empirical wavelet transform, is utilized to decompose non-stationary voltage signals in the frequency domain. This allows the removal of ineffective components for the health evaluation model. The transformer neural network serves as the model backbone, and a loss function is designed to describe the capacity degradation behavior with the assumption that the degradation in LIBs across most operating conditions is inevitable and irreversible. The results show that the model can learn the aging characteristics by analyzing sequences of voltage and current profiles obtained at various time intervals from the same LIB cell. The proposed method is successfully applied to the Stanford University LIB aging dataset, derived from electric vehicle real driving profiles. Notably, this approach achieves an average correlation coefficient of 0.9 between the evaluated health index and the degradation of actual capacity, demonstrating its efficacy in capturing LIB health degradation. This research highlights the feasibility of training deep neural networks using unlabeled LIB data, offering cost-efficient means and unleashing the potential of the measured information. 

---
# Continual Learning for Multiple Modalities 

**Authors**: Hyundong Jin, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.08064)  

**Abstract**: Continual learning aims to learn knowledge of tasks observed in sequential time steps while mitigating the forgetting of previously learned knowledge. Existing methods were proposed under the assumption of learning a single modality (e.g., image) over time, which limits their applicability in scenarios involving multiple modalities. In this work, we propose a novel continual learning framework that accommodates multiple modalities (image, video, audio, depth, and text). We train a model to align various modalities with text, leveraging its rich semantic information. However, this increases the risk of forgetting previously learned knowledge, exacerbated by the differing input traits of each task. To alleviate the overwriting of the previous knowledge of modalities, we propose a method for aggregating knowledge within and across modalities. The aggregated knowledge is obtained by assimilating new information through self-regularization within each modality and associating knowledge between modalities by prioritizing contributions from relevant modalities. Furthermore, we propose a strategy that re-aligns the embeddings of modalities to resolve biased alignment between modalities. We evaluate the proposed method in a wide range of continual learning scenarios using multiple datasets with different modalities. Extensive experiments demonstrate that ours outperforms existing methods in the scenarios, regardless of whether the identity of the modality is given. 

---
# Generalized Kullback-Leibler Divergence Loss 

**Authors**: Jiequan Cui, Beier Zhu, Qingshan Xu, Zhuotao Tian, Xiaojuan Qi, Bei Yu, Hanwang Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08038)  

**Abstract**: In this paper, we delve deeper into the Kullback-Leibler (KL) Divergence loss and mathematically prove that it is equivalent to the Decoupled Kullback-Leibler (DKL) Divergence loss that consists of (1) a weighted Mean Square Error (wMSE) loss and (2) a Cross-Entropy loss incorporating soft labels. Thanks to the decoupled structure of DKL loss, we have identified two areas for improvement. Firstly, we address the limitation of KL loss in scenarios like knowledge distillation by breaking its asymmetric optimization property along with a smoother weight function. This modification effectively alleviates convergence challenges in optimization, particularly for classes with high predicted scores in soft labels. Secondly, we introduce class-wise global information into KL/DKL to reduce bias arising from individual samples. With these two enhancements, we derive the Generalized Kullback-Leibler (GKL) Divergence loss and evaluate its effectiveness by conducting experiments on CIFAR-10/100, ImageNet, and vision-language datasets, focusing on adversarial training, and knowledge distillation tasks. Specifically, we achieve new state-of-the-art adversarial robustness on the public leaderboard -- RobustBench and competitive knowledge distillation performance across CIFAR/ImageNet models and CLIP models, demonstrating the substantial practical merits. Our code is available at this https URL. 

---
# ObjectMover: Generative Object Movement with Video Prior 

**Authors**: Xin Yu, Tianyu Wang, Soo Ye Kim, Paul Guerrero, Xi Chen, Qing Liu, Zhe Lin, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2503.08037)  

**Abstract**: Simple as it seems, moving an object to another location within an image is, in fact, a challenging image-editing task that requires re-harmonizing the lighting, adjusting the pose based on perspective, accurately filling occluded regions, and ensuring coherent synchronization of shadows and reflections while maintaining the object identity. In this paper, we present ObjectMover, a generative model that can perform object movement in highly challenging scenes. Our key insight is that we model this task as a sequence-to-sequence problem and fine-tune a video generation model to leverage its knowledge of consistent object generation across video frames. We show that with this approach, our model is able to adjust to complex real-world scenarios, handling extreme lighting harmonization and object effect movement. As large-scale data for object movement are unavailable, we construct a data generation pipeline using a modern game engine to synthesize high-quality data pairs. We further propose a multi-task learning strategy that enables training on real-world video data to improve the model generalization. Through extensive experiments, we demonstrate that ObjectMover achieves outstanding results and adapts well to real-world scenarios. 

---
# HOFAR: High-Order Augmentation of Flow Autoregressive Transformers 

**Authors**: Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song, Mingda Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08032)  

**Abstract**: Flow Matching and Transformer architectures have demonstrated remarkable performance in image generation tasks, with recent work FlowAR [Ren et al., 2024] synergistically integrating both paradigms to advance synthesis fidelity. However, current FlowAR implementations remain constrained by first-order trajectory modeling during the generation process. This paper introduces a novel framework that systematically enhances flow autoregressive transformers through high-order supervision. We provide theoretical analysis and empirical evaluation showing that our High-Order FlowAR (HOFAR) demonstrates measurable improvements in generation quality compared to baseline models. The proposed approach advances the understanding of flow-based autoregressive modeling by introducing a systematic framework for analyzing trajectory dynamics through high-order expansion. 

---
# In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents 

**Authors**: Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2503.08026)  

**Abstract**: Large Language Models (LLMs) have made significant progress in open-ended dialogue, yet their inability to retain and retrieve relevant information from long-term interactions limits their effectiveness in applications requiring sustained personalization. External memory mechanisms have been proposed to address this limitation, enabling LLMs to maintain conversational continuity. However, existing approaches struggle with two key challenges. First, rigid memory granularity fails to capture the natural semantic structure of conversations, leading to fragmented and incomplete representations. Second, fixed retrieval mechanisms cannot adapt to diverse dialogue contexts and user interaction patterns. In this work, we propose Reflective Memory Management (RMM), a novel mechanism for long-term dialogue agents, integrating forward- and backward-looking reflections: (1) Prospective Reflection, which dynamically summarizes interactions across granularities-utterances, turns, and sessions-into a personalized memory bank for effective future retrieval, and (2) Retrospective Reflection, which iteratively refines the retrieval in an online reinforcement learning (RL) manner based on LLMs' cited evidence. Experiments show that RMM demonstrates consistent improvement across various metrics and benchmarks. For example, RMM shows more than 10% accuracy improvement over the baseline without memory management on the LongMemEval dataset. 

---
# Exploring Bias in over 100 Text-to-Image Generative Models 

**Authors**: Jordan Vice, Naveed Akhtar, Richard Hartley, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2503.08012)  

**Abstract**: We investigate bias trends in text-to-image generative models over time, focusing on the increasing availability of models through open platforms like Hugging Face. While these platforms democratize AI, they also facilitate the spread of inherently biased models, often shaped by task-specific fine-tuning. Ensuring ethical and transparent AI deployment requires robust evaluation frameworks and quantifiable bias metrics. To this end, we assess bias across three key dimensions: (i) distribution bias, (ii) generative hallucination, and (iii) generative miss-rate. Analyzing over 100 models, we reveal how bias patterns evolve over time and across generative tasks. Our findings indicate that artistic and style-transferred models exhibit significant bias, whereas foundation models, benefiting from broader training distributions, are becoming progressively less biased. By identifying these systemic trends, we contribute a large-scale evaluation corpus to inform bias research and mitigation strategies, fostering more responsible AI development.
Keywords: Bias, Ethical AI, Text-to-Image, Generative Models, Open-Source Models 

---
# SKALD: Learning-Based Shot Assembly for Coherent Multi-Shot Video Creation 

**Authors**: Chen Yi Lu, Md Mehrab Tanjim, Ishita Dasgupta, Somdeb Sarkhel, Gang Wu, Saayan Mitra, Somali Chaterji  

**Link**: [PDF](https://arxiv.org/pdf/2503.08010)  

**Abstract**: We present SKALD, a multi-shot video assembly method that constructs coherent video sequences from candidate shots with minimal reliance on text. Central to our approach is the Learned Clip Assembly (LCA) score, a learning-based metric that measures temporal and semantic relationships between shots to quantify narrative coherence. We tackle the exponential complexity of combining multiple shots with an efficient beam-search algorithm guided by the LCA score. To train our model effectively with limited human annotations, we propose two tasks for the LCA encoder: Shot Coherence Learning, which uses contrastive learning to distinguish coherent and incoherent sequences, and Feature Regression, which converts these learned representations into a real-valued coherence score. We develop two variants: a base SKALD model that relies solely on visual coherence and SKALD-text, which integrates auxiliary text information when available. Experiments on the VSPD and our curated MSV3C datasets show that SKALD achieves an improvement of up to 48.6% in IoU and a 43% speedup over the state-of-the-art methods. A user study further validates our approach, with 45% of participants favoring SKALD-assembled videos, compared to 22% preferring text-based assembly methods. 

---
# MoRE: Unlocking Scalability in Reinforcement Learning for Quadruped Vision-Language-Action Models 

**Authors**: Han Zhao, Wenxuan Song, Donglin Wang, Xinyang Tong, Pengxiang Ding, Xuelian Cheng, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2503.08007)  

**Abstract**: Developing versatile quadruped robots that can smoothly perform various actions and tasks in real-world environments remains a significant challenge. This paper introduces a novel vision-language-action (VLA) model, mixture of robotic experts (MoRE), for quadruped robots that aim to introduce reinforcement learning (RL) for fine-tuning large-scale VLA models with a large amount of mixed-quality data. MoRE integrates multiple low-rank adaptation modules as distinct experts within a dense multi-modal large language model (MLLM), forming a sparse-activated mixture-of-experts model. This design enables the model to effectively adapt to a wide array of downstream tasks. Moreover, we employ a reinforcement learning-based training objective to train our model as a Q-function after deeply exploring the structural properties of our tasks. Effective learning from automatically collected mixed-quality data enhances data efficiency and model performance. Extensive experiments demonstrate that MoRE outperforms all baselines across six different skills and exhibits superior generalization capabilities in out-of-distribution scenarios. We further validate our method in real-world scenarios, confirming the practicality of our approach and laying a solid foundation for future research on multi-task learning in quadruped robots. 

---
# Injecting Imbalance Sensitivity for Multi-Task Learning 

**Authors**: Zhipeng Zhou, Liu Liu, Peilin Zhao, Wei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08006)  

**Abstract**: Multi-task learning (MTL) has emerged as a promising approach for deploying deep learning models in real-life applications. Recent studies have proposed optimization-based learning paradigms to establish task-shared representations in MTL. However, our paper empirically argues that these studies, specifically gradient-based ones, primarily emphasize the conflict issue while neglecting the potentially more significant impact of imbalance/dominance in MTL. In line with this perspective, we enhance the existing baseline method by injecting imbalance-sensitivity through the imposition of constraints on the projected norms. To demonstrate the effectiveness of our proposed IMbalance-sensitive Gradient (IMGrad) descent method, we evaluate it on multiple mainstream MTL benchmarks, encompassing supervised learning tasks as well as reinforcement learning. The experimental results consistently demonstrate competitive performance. 

---
# A Neural Symbolic Model for Space Physics 

**Authors**: Jie Ying, Haowei Lin, Chao Yue, Yajie Chen, Chao Xiao, Quanqi Shi, Yitao Liang, Shing-Tung Yau, Yuan Zhou, Jianzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.07994)  

**Abstract**: In this study, we unveil a new AI model, termed PhyE2E, to discover physical formulas through symbolic regression. PhyE2E simplifies symbolic regression by decomposing it into sub-problems using the second-order derivatives of an oracle neural network, and employs a transformer model to translate data into symbolic formulas in an end-to-end manner. The resulting formulas are refined through Monte-Carlo Tree Search and Genetic Programming. We leverage a large language model to synthesize extensive symbolic expressions resembling real physics, and train the model to recover these formulas directly from data. A comprehensive evaluation reveals that PhyE2E outperforms existing state-of-the-art approaches, delivering superior symbolic accuracy, precision in data fitting, and consistency in physical units. We deployed PhyE2E to five applications in space physics, including the prediction of sunspot numbers, solar rotational angular velocity, emission line contribution functions, near-Earth plasma pressure, and lunar-tide plasma signals. The physical formulas generated by AI demonstrate a high degree of accuracy in fitting the experimental data from satellites and astronomical telescopes. We have successfully upgraded the formula proposed by NASA in 1993 regarding solar activity, and for the first time, provided the explanations for the long cycle of solar activity in an explicit form. We also found that the decay of near-Earth plasma pressure is proportional to r^2 to Earth, where subsequent mathematical derivations are consistent with satellite data from another independent study. Moreover, we found physical formulas that can describe the relationships between emission lines in the extreme ultraviolet spectrum of the Sun, temperatures, electron densities, and magnetic fields. The formula obtained is consistent with the properties that physicists had previously hypothesized it should possess. 

---
# Efficient and Accurate Estimation of Lipschitz Constants for Hybrid Quantum-Classical Decision Models 

**Authors**: Sajjad Hashemian, Mohammad Saeed Arvenaghi  

**Link**: [PDF](https://arxiv.org/pdf/2503.07992)  

**Abstract**: In this paper, we propose a novel framework for efficiently and accurately estimating Lipschitz constants in hybrid quantum-classical decision models. Our approach integrates classical neural network with quantum variational circuits to address critical issues in learning theory such as fairness verification, robust training, and generalization.
By a unified convex optimization formulation, we extend existing classical methods to capture the interplay between classical and quantum layers. This integrated strategy not only provide a tight bound on the Lipschitz constant but also improves computational efficiency with respect to the previous methods. 

---
# Provable Zero-Shot Generalization in Offline Reinforcement Learning 

**Authors**: Zhiyong Wang, Chen Yang, John C.S. Lui, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.07988)  

**Abstract**: In this work, we study offline reinforcement learning (RL) with zero-shot generalization property (ZSG), where the agent has access to an offline dataset including experiences from different environments, and the goal of the agent is to train a policy over the training environments which performs well on test environments without further interaction. Existing work showed that classical offline RL fails to generalize to new, unseen environments. We propose pessimistic empirical risk minimization (PERM) and pessimistic proximal policy optimization (PPPO), which leverage pessimistic policy evaluation to guide policy learning and enhance generalization. We show that both PERM and PPPO are capable of finding a near-optimal policy with ZSG. Our result serves as a first step in understanding the foundation of the generalization phenomenon in offline reinforcement learning. 

---
# Hierarchical Contact-Rich Trajectory Optimization for Multi-Modal Manipulation using Tight Convex Relaxations 

**Authors**: Yuki Shirai, Arvind Raghunathan, Devesh K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2503.07963)  

**Abstract**: Designing trajectories for manipulation through contact is challenging as it requires reasoning of object \& robot trajectories as well as complex contact sequences simultaneously. In this paper, we present a novel framework for simultaneously designing trajectories of robots, objects, and contacts efficiently for contact-rich manipulation. We propose a hierarchical optimization framework where Mixed-Integer Linear Program (MILP) selects optimal contacts between robot \& object using approximate dynamical constraints, and then a NonLinear Program (NLP) optimizes trajectory of the robot(s) and object considering full nonlinear constraints. We present a convex relaxation of bilinear constraints using binary encoding technique such that MILP can provide tighter solutions with better computational complexity. The proposed framework is evaluated on various manipulation tasks where it can reason about complex multi-contact interactions while providing computational advantages. We also demonstrate our framework in hardware experiments using a bimanual robot system. 

---
# EFPC: Towards Efficient and Flexible Prompt Compression 

**Authors**: Yun-Hao Cao, Yangsong Wang, Shuzheng Hao, Zhenxing Li, Chengjun Zhan, Sichao Liu, Yi-Qi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07956)  

**Abstract**: The emergence of large language models (LLMs) like GPT-4 has revolutionized natural language processing (NLP), enabling diverse, complex tasks. However, extensive token counts lead to high computational and financial burdens. To address this, we propose Efficient and Flexible Prompt Compression (EFPC), a novel method unifying task-aware and task-agnostic compression for a favorable accuracy-efficiency trade-off. EFPC uses GPT-4 to generate compressed prompts and integrates them with original prompts for training. During training and inference, we selectively prepend user instructions and compress prompts based on predicted probabilities. EFPC is highly data-efficient, achieving significant performance with minimal data. Compared to the state-of-the-art method LLMLingua-2, EFPC achieves a 4.8% relative improvement in F1-score with 1% additional data at a 4x compression rate, and an 11.4% gain with 10% additional data on the LongBench single-doc QA benchmark. EFPC's unified framework supports broad applicability and enhances performance across various models, tasks, and domains, offering a practical advancement in NLP. 

---
# 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting 

**Authors**: Zhongpai Gao, Benjamin Planche, Meng Zheng, Anwesa Choudhuri, Terrence Chen, Ziyan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07946)  

**Abstract**: Real-time rendering of dynamic scenes with view-dependent effects remains a fundamental challenge in computer graphics. While recent advances in Gaussian Splatting have shown promising results separately handling dynamic scenes (4DGS) and view-dependent effects (6DGS), no existing method unifies these capabilities while maintaining real-time performance. We present 7D Gaussian Splatting (7DGS), a unified framework representing scene elements as seven-dimensional Gaussians spanning position (3D), time (1D), and viewing direction (3D). Our key contribution is an efficient conditional slicing mechanism that transforms 7D Gaussians into view- and time-conditioned 3D Gaussians, maintaining compatibility with existing 3D Gaussian Splatting pipelines while enabling joint optimization. Experiments demonstrate that 7DGS outperforms prior methods by up to 7.36 dB in PSNR while achieving real-time rendering (401 FPS) on challenging dynamic scenes with complex view-dependent effects. The project page is: this https URL. 

---
# A Theory of Learning with Autoregressive Chain of Thought 

**Authors**: Nirmit Joshi, Gal Vardi, Adam Block, Surbhi Goel, Zhiyuan Li, Theodor Misiakiewicz, Nathan Srebro  

**Link**: [PDF](https://arxiv.org/pdf/2503.07932)  

**Abstract**: For a given base class of sequence-to-next-token generators, we consider learning prompt-to-answer mappings obtained by iterating a fixed, time-invariant generator for multiple steps, thus generating a chain-of-thought, and then taking the final token as the answer. We formalize the learning problems both when the chain-of-thought is observed and when training only on prompt-answer pairs, with the chain-of-thought latent. We analyze the sample and computational complexity both in terms of general properties of the base class (e.g. its VC dimension) and for specific base classes such as linear thresholds. We present a simple base class that allows for universal representability and computationally tractable chain-of-thought learning. Central to our development is that time invariance allows for sample complexity that is independent of the length of the chain-of-thought. Attention arises naturally in our construction. 

---
# Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural Vision-Language Dataset for Southeast Asia 

**Authors**: Samuel Cahyawijaya, Holy Lovenia, Joel Ruben Antony Moniz, Tack Hwa Wong, Mohammad Rifqi Farhansyah, Thant Thiri Maung, Frederikus Hudi, David Anugraha, Muhammad Ravi Shulthan Habibi, Muhammad Reza Qorib, Amit Agarwal, Joseph Marvin Imperial, Hitesh Laxmichand Patel, Vicky Feliren, Bahrul Ilmi Nasution, Manuel Antonio Rufino, Genta Indra Winata, Rian Adam Rajagede, Carlos Rafael Catalan, Mohamed Fazli Imam, Priyaranjan Pattnayak, Salsabila Zahirah Pranida, Kevin Pratama, Yeshil Bangera, Adisai Na-Thalang, Patricia Nicole Monderin, Yueqi Song, Christian Simon, Lynnette Hui Xian Ng, Richardy Lobo' Sapan, Taki Hasan Rafi, Bin Wang, Supryadi, Kanyakorn Veerakanjana, Piyalitt Ittichaiwong, Matthew Theodore Roque, Karissa Vincentio, Takdanai Kreangphet, Phakphum Artkaew, Kadek Hendrawan Palgunadi, Yanzhi Yu, Rochana Prih Hastuti, William Nixon, Mithil Bangera, Adrian Xuan Wei Lim, Aye Hninn Khine, Hanif Muhammad Zhafran, Teddy Ferdinan, Audra Aurora Izzani, Ayushman Singh, Evan, Jauza Akbar Krito, Michael Anugraha, Fenal Ashokbhai Ilasariya, Haochen Li, John Amadeo Daniswara, Filbert Aurelian Tjiaranata, Eryawan Presma Yulianrifat, Can Udomcharoenchaikit, Fadil Risdian Ansori, Mahardika Krisna Ihsani, Giang Nguyen, Anab Maulana Barik, Dan John Velasco, Rifo Ahmad Genadi, Saptarshi Saha, Chengwei Wei, Isaiah Flores, Kenneth Ko Han Chen, Anjela Gail Santos, Wan Shen Lim, Kaung Si Phyo, Tim Santos, Meisyarah Dwiastuti, Jiayun Luo, Jan Christian Blaise Cruz, Ming Shan Hee, Ikhlasul Akmal Hanif, M.Alif Al Hakim, Muhammad Rizky Sya'ban, Kun Kerdthaisong, Lester James V. Miranda, Fajri Koto, Tirana Noor Fatyanosa, Alham Fikri Aji, Jostin Jerico Rosal, Jun Kevin, Robert Wijaya, Onno P. Kampman, Ruochen Zhang, Börje F. Karlsson, Peerat Limkonchotiwat  

**Link**: [PDF](https://arxiv.org/pdf/2503.07920)  

**Abstract**: Southeast Asia (SEA) is a region of extraordinary linguistic and cultural diversity, yet it remains significantly underrepresented in vision-language (VL) research. This often results in artificial intelligence (AI) models that fail to capture SEA cultural nuances. To fill this gap, we present SEA-VL, an open-source initiative dedicated to developing high-quality, culturally relevant data for SEA languages. By involving contributors from SEA countries, SEA-VL aims to ensure better cultural relevance and diversity, fostering greater inclusivity of underrepresented languages in VL research. Beyond crowdsourcing, our initiative goes one step further in the exploration of the automatic collection of culturally relevant images through crawling and image generation. First, we find that image crawling achieves approximately ~85% cultural relevance while being more cost- and time-efficient than crowdsourcing. Second, despite the substantial progress in generative vision models, synthetic images remain unreliable in accurately reflecting SEA cultures. The generated images often fail to reflect the nuanced traditions and cultural contexts of the region. Collectively, we gather 1.28M SEA culturally-relevant images, more than 50 times larger than other existing datasets. Through SEA-VL, we aim to bridge the representation gap in SEA, fostering the development of more inclusive AI systems that authentically represent diverse cultures across SEA. 

---
# Visual and Text Prompt Segmentation: A Novel Multi-Model Framework for Remote Sensing 

**Authors**: Xing Zi, Kairui Jin, Xian Tao, Jun Li, Ali Braytee, Rajiv Ratn Shah, Mukesh Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2503.07911)  

**Abstract**: Pixel-level segmentation is essential in remote sensing, where foundational vision models like CLIP and Segment Anything Model(SAM) have demonstrated significant capabilities in zero-shot segmentation tasks. Despite their advances, challenges specific to remote sensing remain substantial. Firstly, The SAM without clear prompt constraints, often generates redundant masks, and making post-processing more complex. Secondly, the CLIP model, mainly designed for global feature alignment in foundational models, often overlooks local objects crucial to remote sensing. This oversight leads to inaccurate recognition or misplaced focus in multi-target remote sensing imagery. Thirdly, both models have not been pre-trained on multi-scale aerial views, increasing the likelihood of detection failures. To tackle these challenges, we introduce the innovative VTPSeg pipeline, utilizing the strengths of Grounding DINO, CLIP, and SAM for enhanced open-vocabulary image segmentation. The Grounding DINO+(GD+) module generates initial candidate bounding boxes, while the CLIP Filter++(CLIP++) module uses a combination of visual and textual prompts to refine and filter out irrelevant object bounding boxes, ensuring that only pertinent objects are considered. Subsequently, these refined bounding boxes serve as specific prompts for the FastSAM model, which executes precise segmentation. Our VTPSeg is validated by experimental and ablation study results on five popular remote sensing image segmentation datasets. 

---
# FunGraph: Functionality Aware 3D Scene Graphs for Language-Prompted Scene Interaction 

**Authors**: Dennis Rotondi, Fabio Scaparro, Hermann Blum, Kai O. Arras  

**Link**: [PDF](https://arxiv.org/pdf/2503.07909)  

**Abstract**: The concept of 3D scene graphs is increasingly recognized as a powerful semantic and hierarchical representation of the environment. Current approaches often address this at a coarse, object-level resolution. In contrast, our goal is to develop a representation that enables robots to directly interact with their environment by identifying both the location of functional interactive elements and how these can be used. To achieve this, we focus on detecting and storing objects at a finer resolution, focusing on affordance-relevant parts. The primary challenge lies in the scarcity of data that extends beyond instance-level detection and the inherent difficulty of capturing detailed object features using robotic sensors. We leverage currently available 3D resources to generate 2D data and train a detector, which is then used to augment the standard 3D scene graph generation pipeline. Through our experiments, we demonstrate that our approach achieves functional element segmentation comparable to state-of-the-art 3D models and that our augmentation enables task-driven affordance grounding with higher accuracy than the current solutions. 

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
# Measuring directional bias amplification in image captions using predictability 

**Authors**: Rahul Nair, Bhanu Tokas, Hannah Kerner  

**Link**: [PDF](https://arxiv.org/pdf/2503.07878)  

**Abstract**: When we train models on biased ML datasets, they not only learn these biases but can inflate them at test time - a phenomenon called bias amplification. To measure bias amplification in ML datasets, many co-occurrence-based metrics have been proposed. Co-occurrence-based metrics are effective in measuring bias amplification in simple problems like image classification. However, these metrics are ineffective for complex problems like image captioning as they cannot capture the semantics of a caption. To measure bias amplification in captions, prior work introduced a predictability-based metric called Leakage in Captioning (LIC). While LIC captures the semantics and context of captions, it has limitations. LIC cannot identify the direction in which bias is amplified, poorly estimates dataset bias due to a weak vocabulary substitution strategy, and is highly sensitive to attacker models (a hyperparameter in predictability-based metrics). To overcome these issues, we propose Directional Predictability Amplification in Captioning (DPAC). DPAC measures directional bias amplification in captions, provides a better estimate of dataset bias using an improved substitution strategy, and is less sensitive to attacker models. Our experiments on the COCO captioning dataset show how DPAC is the most reliable metric to measure bias amplification in captions. 

---
# Topology-Preserving Loss for Accurate and Anatomically Consistent Cardiac Mesh Reconstruction 

**Authors**: Chenyu Zhang, Yihao Luo, Yinzhe Wu, Choon Hwai Yap, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07874)  

**Abstract**: Accurate cardiac mesh reconstruction from volumetric data is essential for personalized cardiac modeling and clinical analysis. However, existing deformation-based approaches are prone to topological inconsistencies, particularly membrane penetration, which undermines the anatomical plausibility of the reconstructed mesh. To address this issue, we introduce Topology-Preserving Mesh Loss (TPM Loss), a novel loss function that explicitly enforces topological constraints during mesh deformation. By identifying topology-violating points, TPM Loss ensures spatially consistent reconstructions. Extensive experiments on CT and MRI datasets show that TPM Loss reduces topology violations by up to 93.1% while maintaining high segmentation accuracy (DSC: 89.1%-92.9%) and improving mesh fidelity (Chamfer Distance reduction up to 0.26 mm). These results demonstrate that TPM Loss effectively prevents membrane penetration and significantly improves cardiac mesh quality, enabling more accurate and anatomically consistent cardiac reconstructions. 

---
# MapQA: Open-domain Geospatial Question Answering on Map Data 

**Authors**: Zekun Li, Malcolm Grossman, Eric, Qasemi, Mihir Kulkarni, Muhao Chen, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07871)  

**Abstract**: Geospatial question answering (QA) is a fundamental task in navigation and point of interest (POI) searches. While existing geospatial QA datasets exist, they are limited in both scale and diversity, often relying solely on textual descriptions of geo-entities without considering their geometries. A major challenge in scaling geospatial QA datasets for reasoning lies in the complexity of geospatial relationships, which require integrating spatial structures, topological dependencies, and multi-hop reasoning capabilities that most text-based QA datasets lack. To address these limitations, we introduce MapQA, a novel dataset that not only provides question-answer pairs but also includes the geometries of geo-entities referenced in the questions. MapQA is constructed using SQL query templates to extract question-answer pairs from OpenStreetMap (OSM) for two study regions: Southern California and Illinois. It consists of 3,154 QA pairs spanning nine question types that require geospatial reasoning, such as neighborhood inference and geo-entity type identification. Compared to existing datasets, MapQA expands both the number and diversity of geospatial question types. We explore two approaches to tackle this challenge: (1) a retrieval-based language model that ranks candidate geo-entities by embedding similarity, and (2) a large language model (LLM) that generates SQL queries from natural language questions and geo-entity attributes, which are then executed against an OSM database. Our findings indicate that retrieval-based methods effectively capture concepts like closeness and direction but struggle with questions that require explicit computations (e.g., distance calculations). LLMs (e.g., GPT and Gemini) excel at generating SQL queries for one-hop reasoning but face challenges with multi-hop reasoning, highlighting a key bottleneck in advancing geospatial QA systems. 

---
# Right Reward Right Time for Federated Learning 

**Authors**: Thanh Linh Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2503.07869)  

**Abstract**: Critical learning periods (CLPs) in federated learning (FL) refer to early stages during which low-quality contributions (e.g., sparse training data availability) can permanently impair the learning performance of the global model owned by the model owner (i.e., the cloud server). However, strategies to motivate clients with high-quality contributions to join the FL training process and share trained model updates during CLPs remain underexplored. Additionally, existing incentive mechanisms in FL treat all training periods equally, which consequently fails to motivate clients to participate early. Compounding this challenge is the cloud's limited knowledge of client training capabilities due to privacy regulations, leading to information asymmetry. Therefore, in this article, we propose a time-aware incentive mechanism, called Right Reward Right Time (R3T), to encourage client involvement, especially during CLPs, to maximize the utility of the cloud in FL. Specifically, the cloud utility function captures the trade-off between the achieved model performance and payments allocated for clients' contributions, while accounting for clients' time and system capabilities, efforts, joining time, and rewards. Then, we analytically derive the optimal contract for the cloud and devise a CLP-aware mechanism to incentivize early participation and efforts while maximizing cloud utility, even under information asymmetry. By providing the right reward at the right time, our approach can attract the highest-quality contributions during CLPs. Simulation and proof-of-concept studies show that R3T increases cloud utility and is more economically effective than benchmarks. Notably, our proof-of-concept results show up to a 47.6% reduction in the total number of clients and up to a 300% improvement in convergence time while reaching competitive test accuracies compared with incentive mechanism benchmarks. 

---
# Video Action Differencing 

**Authors**: James Burgess, Xiaohan Wang, Yuhui Zhang, Anita Rau, Alejandro Lozano, Lisa Dunlap, Trevor Darrell, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2503.07860)  

**Abstract**: How do two individuals differ when performing the same action? In this work, we introduce Video Action Differencing (VidDiff), the novel task of identifying subtle differences between videos of the same action, which has many applications, such as coaching and skill learning. To enable development on this new task, we first create VidDiffBench, a benchmark dataset containing 549 video pairs, with human annotations of 4,469 fine-grained action differences and 2,075 localization timestamps indicating where these differences occur. Our experiments demonstrate that VidDiffBench poses a significant challenge for state-of-the-art large multimodal models (LMMs), such as GPT-4o and Qwen2-VL. By analyzing failure cases of LMMs on VidDiffBench, we highlight two key challenges for this task: localizing relevant sub-actions over two videos and fine-grained frame comparison. To overcome these, we propose the VidDiff method, an agentic workflow that breaks the task into three stages: action difference proposal, keyframe localization, and frame differencing, each stage utilizing specialized foundation models. To encourage future research in this new task, we release the benchmark at this https URL and code at this http URL. 

---
# CIMAGE: Exploiting the Conditional Independence in Masked Graph Auto-encoders 

**Authors**: Jongwon Park, Heesoo Jung, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.07852)  

**Abstract**: Recent Self-Supervised Learning (SSL) methods encapsulating relational information via masking in Graph Neural Networks (GNNs) have shown promising performance. However, most existing approaches rely on random masking strategies in either feature or graph space, which may fail to capture task-relevant information fully. We posit that this limitation stems from an inability to achieve minimum redundancy between masked and unmasked components while ensuring maximum relevance of both to potential downstream tasks. Conditional Independence (CI) inherently satisfies the minimum redundancy and maximum relevance criteria, but its application typically requires access to downstream labels. To address this challenge, we introduce CIMAGE, a novel approach that leverages Conditional Independence to guide an effective masking strategy within the latent space. CIMAGE utilizes CI-aware latent factor decomposition to generate two distinct contexts, leveraging high-confidence pseudo-labels derived from unsupervised graph clustering. In this framework, the pretext task involves reconstructing the masked second context solely from the information provided by the first context. Our theoretical analysis further supports the superiority of CIMAGE's novel CI-aware masking method by demonstrating that the learned embedding exhibits approximate linear separability, which enables accurate predictions for the downstream task. Comprehensive evaluations across diverse graph benchmarks illustrate the advantage of CIMAGE, with notably higher average rankings on node classification and link prediction tasks. Notably, our proposed model highlights the under-explored potential of CI in enhancing graph SSL methodologies and offers enriched insights for effective graph representation learning. 

---
# HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations 

**Authors**: Samir Abdaljalil, Hasan Kurban, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07833)  

**Abstract**: Large Language Models (LLMs) are increasingly used in various contexts, yet remain prone to generating non-factual content, commonly referred to as "hallucinations". The literature categorizes hallucinations into several types, including entity-level, relation-level, and sentence-level hallucinations. However, existing hallucination datasets often fail to capture fine-grained hallucinations in multilingual settings. In this work, we introduce HalluVerse25, a multilingual LLM hallucination dataset that categorizes fine-grained hallucinations in English, Arabic, and Turkish. Our dataset construction pipeline uses an LLM to inject hallucinations into factual biographical sentences, followed by a rigorous human annotation process to ensure data quality. We evaluate several LLMs on HalluVerse25, providing valuable insights into how proprietary models perform in detecting LLM-generated hallucinations across different contexts. 

---
# Group Fairness in Multi-Task Reinforcement Learning 

**Authors**: Kefan Song, Runnan Jiang, Rohan Chandra, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07817)  

**Abstract**: This paper addresses a critical societal consideration in the application of Reinforcement Learning (RL): ensuring equitable outcomes across different demographic groups in multi-task settings. While previous work has explored fairness in single-task RL, many real-world applications are multi-task in nature and require policies to maintain fairness across all tasks. We introduce a novel formulation of multi-task group fairness in RL and propose a constrained optimization algorithm that explicitly enforces fairness constraints across multiple tasks simultaneously. We have shown that our proposed algorithm does not violate fairness constraints with high probability and with sublinear regret in the finite-horizon episodic setting. Through experiments in RiverSwim and MuJoCo environments, we demonstrate that our approach better ensures group fairness across multiple tasks compared to previous methods that lack explicit multi-task fairness constraints in both the finite-horizon setting and the infinite-horizon setting. Our results show that the proposed algorithm achieves smaller fairness gaps while maintaining comparable returns across different demographic groups and tasks, suggesting its potential for addressing fairness concerns in real-world multi-task RL applications. 

---
# AgriField3D: A Curated 3D Point Cloud and Procedural Model Dataset of Field-Grown Maize from a Diversity Panel 

**Authors**: Elvis Kimara, Mozhgan Hadadi, Jackson Godbersen, Aditya Balu, Talukder Jubery, Yawei Li, Adarsh Krishnamurthy, Patrick S. Schnable, Baskar Ganapathysubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2503.07813)  

**Abstract**: The application of artificial intelligence (AI) in three-dimensional (3D) agricultural research, particularly for maize, has been limited by the scarcity of large-scale, diverse datasets. While 2D image datasets are abundant, they fail to capture essential structural details such as leaf architecture, plant volume, and spatial arrangements that 3D data provide. To address this limitation, we present AgriField3D (this https URL), a curated dataset of 3D point clouds of field-grown maize plants from a diverse genetic panel, designed to be AI-ready for advancing agricultural research. Our dataset comprises over 1,000 high-quality point clouds collected using a Terrestrial Laser Scanner, complemented by procedural models that provide structured, parametric representations of maize plants. These procedural models, generated using Non-Uniform Rational B-Splines (NURBS) and optimized via a two-step process combining Particle Swarm Optimization (PSO) and differentiable programming, enable precise, scalable reconstructions of leaf surfaces and plant architectures. To enhance usability, we performed graph-based segmentation to isolate individual leaves and stalks, ensuring consistent labeling across all samples. We also conducted rigorous manual quality control on all datasets, correcting errors in segmentation, ensuring accurate leaf ordering, and validating metadata annotations. The dataset further includes metadata detailing plant morphology and quality, alongside multi-resolution subsampled versions (100k, 50k, 10k points) optimized for various computational needs. By integrating point cloud data of field grown plants with high-fidelity procedural models and ensuring meticulous manual validation, AgriField3D provides a comprehensive foundation for AI-driven phenotyping, plant structural analysis, and 3D applications in agricultural research. 

---
# A primer on optimal transport for causal inference with observational data 

**Authors**: Florian F Gunsilius  

**Link**: [PDF](https://arxiv.org/pdf/2503.07811)  

**Abstract**: The theory of optimal transportation has developed into a powerful and elegant framework for comparing probability distributions, with wide-ranging applications in all areas of science. The fundamental idea of analyzing probabilities by comparing their underlying state space naturally aligns with the core idea of causal inference, where understanding and quantifying counterfactual states is paramount. Despite this intuitive connection, explicit research at the intersection of optimal transport and causal inference is only beginning to develop. Yet, many foundational models in causal inference have implicitly relied on optimal transport principles for decades, without recognizing the underlying connection. Therefore, the goal of this review is to offer an introduction to the surprisingly deep existing connections between optimal transport and the identification of causal effects with observational data -- where optimal transport is not just a set of potential tools, but actually builds the foundation of model assumptions. As a result, this review is intended to unify the language and notation between different areas of statistics, mathematics, and econometrics, by pointing out these existing connections, and to explore novel problems and directions for future work in both areas derived from this realization. 

---
# Training Domain Draft Models for Speculative Decoding: Best Practices and Insights 

**Authors**: Fenglu Hong, Ravi Raju, Jonathan Lingjie Li, Bo Li, Urmish Thakker, Avinash Ravichandran, Swayambhoo Jain, Changran Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07807)  

**Abstract**: Speculative decoding is an effective method for accelerating inference of large language models (LLMs) by employing a small draft model to predict the output of a target model. However, when adapting speculative decoding to domain-specific target models, the acceptance rate of the generic draft model drops significantly due to domain shift. In this work, we systematically investigate knowledge distillation techniques for training domain draft models to improve their speculation accuracy. We compare white-box and black-box distillation approaches and explore their effectiveness in various data accessibility scenarios, including historical user queries, curated domain data, and synthetically generated alignment data. Our experiments across Function Calling, Biology, and Chinese domains show that offline distillation consistently outperforms online distillation by 11% to 25%, white-box distillation surpasses black-box distillation by 2% to 10%, and data scaling trends hold across domains. Additionally, we find that synthetic data can effectively align draft models and achieve 80% to 93% of the performance of training on historical user queries. These findings provide practical guidelines for training domain-specific draft models to improve speculative decoding efficiency. 

---
# Towards Large Language Models that Benefit for All: Benchmarking Group Fairness in Reward Models 

**Authors**: Kefan Song, Jin Yao, Runnan Jiang, Rohan Chandra, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07806)  

**Abstract**: As Large Language Models (LLMs) become increasingly powerful and accessible to human users, ensuring fairness across diverse demographic groups, i.e., group fairness, is a critical ethical concern. However, current fairness and bias research in LLMs is limited in two aspects. First, compared to traditional group fairness in machine learning classification, it requires that the non-sensitive attributes, in this case, the prompt questions, be the same across different groups. In many practical scenarios, different groups, however, may prefer different prompt questions and this requirement becomes impractical. Second, it evaluates group fairness only for the LLM's final output without identifying the source of possible bias. Namely, the bias in LLM's output can result from both the pretraining and the finetuning. For finetuning, the bias can result from both the RLHF procedure and the learned reward model. Arguably, evaluating the group fairness of each component in the LLM pipeline could help develop better methods to mitigate the possible bias. Recognizing those two limitations, this work benchmarks the group fairness of learned reward models. By using expert-written text from arXiv, we are able to benchmark the group fairness of reward models without requiring the same prompt questions across different demographic groups. Surprisingly, our results demonstrate that all the evaluated reward models (e.g., Nemotron-4-340B-Reward, ArmoRM-Llama3-8B-v0.1, and GRM-llama3-8B-sftreg) exhibit statistically significant group unfairness. We also observed that top-performing reward models (w.r.t. canonical performance metrics) tend to demonstrate better group fairness. 

---
# Self-supervised Normality Learning and Divergence Vector-guided Model Merging for Zero-shot Congenital Heart Disease Detection in Fetal Ultrasound Videos 

**Authors**: Pramit Saha, Divyanshu Mishra, Netzahualcoyotl Hernandez-Cruz, Olga Patey, Aris Papageorghiou, Yuki M. Asano, J. Alison Noble  

**Link**: [PDF](https://arxiv.org/pdf/2503.07799)  

**Abstract**: Congenital Heart Disease (CHD) is one of the leading causes of fetal mortality, yet the scarcity of labeled CHD data and strict privacy regulations surrounding fetal ultrasound (US) imaging present significant challenges for the development of deep learning-based models for CHD detection. Centralised collection of large real-world datasets for rare conditions, such as CHD, from large populations requires significant co-ordination and resource. In addition, data governance rules increasingly prevent data sharing between sites. To address these challenges, we introduce, for the first time, a novel privacy-preserving, zero-shot CHD detection framework that formulates CHD detection as a normality modeling problem integrated with model merging. In our framework dubbed Sparse Tube Ultrasound Distillation (STUD), each hospital site first trains a sparse video tube-based self-supervised video anomaly detection (VAD) model on normal fetal heart US clips with self-distillation loss. This enables site-specific models to independently learn the distribution of healthy cases. To aggregate knowledge across the decentralized models while maintaining privacy, we propose a Divergence Vector-Guided Model Merging approach, DivMerge, that combines site-specific models into a single VAD model without data exchange. Our approach preserves domain-agnostic rich spatio-temporal representations, ensuring generalization to unseen CHD cases. We evaluated our approach on real-world fetal US data collected from 5 hospital sites. Our merged model outperformed site-specific models by 23.77% and 30.13% in accuracy and F1-score respectively on external test sets. 

---
# Joint Explainability-Performance Optimization With Surrogate Models for AI-Driven Edge Services 

**Authors**: Foivos Charalampakos, Thomas Tsouparopoulos, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.07784)  

**Abstract**: Explainable AI is a crucial component for edge services, as it ensures reliable decision making based on complex AI models. Surrogate models are a prominent approach of XAI where human-interpretable models, such as a linear regression model, are trained to approximate a complex (black-box) model's predictions. This paper delves into the balance between the predictive accuracy of complex AI models and their approximation by surrogate ones, advocating that both these models benefit from being learned simultaneously. We derive a joint (bi-level) training scheme for both models and we introduce a new algorithm based on multi-objective optimization (MOO) to simultaneously minimize both the complex model's prediction error and the error between its outputs and those of the surrogate. Our approach leads to improvements that exceed 99% in the approximation of the black-box model through the surrogate one, as measured by the metric of Fidelity, for a compromise of less than 3% absolute reduction in the black-box model's predictive accuracy, compared to single-task and multi-task learning baselines. By improving Fidelity, we can derive more trustworthy explanations of the complex model's outcomes from the surrogate, enabling reliable AI applications for intelligent services at the network edge. 

---
# Evaluating LLaMA 3.2 for Software Vulnerability Detection 

**Authors**: José Gonçalves, Miguel Silva, Bernardo Cabral, Tiago Dias, Eva Maia, Isabel Praça, Ricardo Severino, Luís Lino Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2503.07770)  

**Abstract**: Deep Learning (DL) has emerged as a powerful tool for vulnerability detection, often outperforming traditional solutions. However, developing effective DL models requires large amounts of real-world data, which can be difficult to obtain in sufficient quantities. To address this challenge, DiverseVul dataset has been curated as the largest dataset of vulnerable and non-vulnerable C/C++ functions extracted exclusively from real-world projects. Its goal is to provide high-quality, large-scale samples for training DL models. However, during our study several inconsistencies were identified in the raw dataset while applying pre-processing techniques, highlighting the need for a refined version. In this work, we present a refined version of DiverseVul dataset, which is used to fine-tune a large language model, LLaMA 3.2, for vulnerability detection. Experimental results show that the use of pre-processing techniques led to an improvement in performance, with the model achieving an F1-Score of 66%, a competitive result when compared to our baseline, which achieved a 47% F1-Score in software vulnerability detection. 

---
# A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing 

**Authors**: Shengfan Cao, Eunhyek Joa, Francesco Borrelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.07737)  

**Abstract**: Guaranteeing constraint satisfaction is challenging in imitation learning (IL), particularly in tasks that require operating near a system's handling limits. Traditional IL methods often struggle to enforce constraints, leading to suboptimal performance in high-precision tasks. In this paper, we present a simple approach to incorporating safety into the IL objective. Through simulations, we empirically validate our approach on an autonomous racing task with both full-state and image feedback, demonstrating improved constraint satisfaction and greater consistency in task performance compared to a baseline method. 

---
# Automated Benchmark Generation for Repository-Level Coding Tasks 

**Authors**: Konstantinos Vergopoulos, Mark Niklas Müller, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2503.07701)  

**Abstract**: Code Agent development is an extremely active research area, where a reliable performance metric is critical for tracking progress and guiding new developments. This demand is underscored by the meteoric rise in popularity of SWE-Bench. This benchmark challenges code agents to generate patches addressing GitHub issues given the full repository as context. The correctness of generated patches is then evaluated by executing a human-written test suite extracted from the repository after the issue's resolution. However, constructing benchmarks like SWE-Bench requires substantial manual effort to set up historically accurate execution environments for testing. Crucially, this severely limits the number of considered repositories, e.g., just 12 for SWE-Bench. Considering so few repositories, selected for their popularity runs the risk of leading to a distributional mismatch, i.e., the measured performance may not be representative of real-world scenarios potentially misguiding development efforts. In this work, we address this challenge and introduce SetUpAgent, a fully automated system capable of historically accurate dependency setup, test execution, and result parsing. Using SetUpAgent, we generate two new datasets: (i) SWEE-Bench an extended version of SWE-Bench encompassing hundreds of repositories, and (ii) SWA-Bench a benchmark focusing on applications rather than libraries. Comparing these datasets to SWE-Bench with respect to their characteristics and code agent performance, we find significant distributional differences, including lower issue description quality and detail level, higher fix complexity, and most importantly up to 40% lower agent success rates. 

---
# A Task and Motion Planning Framework Using Iteratively Deepened AND/OR Graph Networks 

**Authors**: Hossein Karami, Antony Thomas, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2503.07700)  

**Abstract**: In this paper, we present an approach for integrated task and motion planning based on an AND/OR graph network, which is used to represent task-level states and actions, and we leverage it to implement different classes of task and motion planning problems (TAMP). Several problems that fall under task and motion planning do not have a predetermined number of sub-tasks to achieve a goal. For example, while retrieving a target object from a cluttered workspace, in principle the number of object re-arrangements required to finally grasp it cannot be known ahead of time. To address this challenge, and in contrast to traditional planners, also those based on AND/OR graphs, we grow the AND/OR graph at run-time by progressively adding sub-graphs until grasping the target object becomes feasible, which yields a network of AND/OR graphs. The approach is extended to enable multi-robot task and motion planning, and (i) it allows us to perform task allocation while coordinating the activity of a given number of robots, and (ii) can handle multi-robot tasks involving an a priori unknown number of sub-tasks. The approach is evaluated and validated both in simulation and with a real dual-arm robot manipulator, that is, Baxter from Rethink Robotics. In particular, for the single-robot task and motion planning, we validated our approach in three different TAMP domains. Furthermore, we also use three different robots for simulation, namely, Baxter, Franka Emika Panda manipulators, and a PR2 robot. Experiments show that our approach can be readily scaled to scenarios with many objects and robots, and is capable of handling different classes of TAMP problems. 

---
# Artificial Intelligence in Deliberation: The AI Penalty and the Emergence of a New Deliberative Divide 

**Authors**: Andreas Jungherr, Adrian Rauchfleisch  

**Link**: [PDF](https://arxiv.org/pdf/2503.07690)  

**Abstract**: Digital deliberation has expanded democratic participation, yet challenges remain. This includes processing information at scale, moderating discussions, fact-checking, or attracting people to participate. Recent advances in artificial intelligence (AI) offer potential solutions, but public perceptions of AI's role in deliberation remain underexplored. Beyond efficiency, democratic deliberation is about voice and recognition. If AI is integrated into deliberation, public trust, acceptance, and willingness to participate may be affected. We conducted a preregistered survey experiment with a representative sample in Germany (n=1850) to examine how information about AI-enabled deliberation influences willingness to participate and perceptions of deliberative quality. Respondents were randomly assigned to treatments that provided them information about deliberative tasks facilitated by either AI or humans. Our findings reveal a significant AI-penalty. Participants were less willing to engage in AI-facilitated deliberation and rated its quality lower than human-led formats. These effects were moderated by individual predispositions. Perceptions of AI's societal benefits and anthropomorphization of AI showed positive interaction effects on people's interest to participate in AI-enabled deliberative formats and positive quality assessments, while AI risk assessments showed negative interactions with information about AI-enabled deliberation. These results suggest AI-enabled deliberation faces substantial public skepticism, potentially even introducing a new deliberative divide. Unlike traditional participation gaps based on education or demographics, this divide is shaped by attitudes toward AI. As democratic engagement increasingly moves online, ensuring AI's role in deliberation does not discourage participation or deepen inequalities will be a key challenge for future research and policy. 

---
# Adaptive routing protocols for determining optimal paths in AI multi-agent systems: a priority- and learning-enhanced approach 

**Authors**: Theodor Panayotov, Ivo Emanuilov  

**Link**: [PDF](https://arxiv.org/pdf/2503.07686)  

**Abstract**: As distributed artificial intelligence (AI) and multi-agent architectures grow increasingly complex, the need for adaptive, context-aware routing becomes paramount. This paper introduces an enhanced, adaptive routing algorithm tailored for AI multi-agent networks, integrating priority-based cost functions and dynamic learning mechanisms. Building on an extended Dijkstra-based framework, we incorporate multi-faceted parameters such as task complexity, user request priority, agent capabilities, bandwidth, latency, load, model sophistication, and reliability. We further propose dynamically adaptive weighting factors, tuned via reinforcement learning (RL), to continuously evolve routing policies based on observed network performance. Additionally, heuristic filtering and hierarchical routing structures improve scalability and responsiveness. Our approach yields context-sensitive, load-aware, and priority-focused routing decisions that not only reduce latency for critical tasks but also optimize overall resource utilization, ultimately enhancing the robustness, flexibility, and efficiency of multi-agent systems. 

---
# Ways of Seeing, and Selling, AI Art 

**Authors**: Imke van Heerden  

**Link**: [PDF](https://arxiv.org/pdf/2503.07685)  

**Abstract**: In early 2025, Augmented Intelligence - Christie's first AI art auction - drew criticism for showcasing a controversial genre. Amid wider legal uncertainty, artists voiced concerns over data mining practices, notably with respect to copyright. The backlash could be viewed as a microcosm of AI's contested position in the creative economy. Touching on the auction's presentation, reception, and results, this paper explores how, among social dissonance, machine learning finds its place in the artworld. Foregrounding responsible innovation, the paper provides a balanced perspective that champions creators' rights and brings nuance to this polarised debate. With a focus on exhibition design, it centres framing, which refers to the way a piece is presented to influence consumer perception. Context plays a central role in shaping our understanding of how good, valuable, and even ethical an artwork is. In this regard, Augmented Intelligence situates AI art within a surprisingly traditional framework, leveraging hallmarks of "high art" to establish the genre's cultural credibility. Generative AI has a clear economic dimension, converging questions of artistic merit with those of monetary worth. Scholarship on ways of seeing, or framing, could substantively inform the interpretation and evaluation of creative outputs, including assessments of their aesthetic and commercial value. 

---
# A Time Series Multitask Framework Integrating a Large Language Model, Pre-Trained Time Series Model, and Knowledge Graph 

**Authors**: Shule Hao, Junpeng Bao, Chuncheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07682)  

**Abstract**: Time series analysis is crucial in fields like finance, transportation, and industry. However, traditional models often focus solely on temporal features, limiting their ability to capture underlying information. This paper proposes a novel time series multitask framework, called LTM, which integrates temporal features with textual descriptions to enhance analytical and predictive capabilities. LTM combines pre-trained time series model, large language model (LLM), and knowledge graph to tackle time series tasks, including forecasting, imputation, and anomaly detection. LTM achieves improved performance with a few trainable parameters. It is very efficient and practical. LTM encodes time series data into patches and enriches user-provided prompts using knowledge graphs to generate enhanced prompts. A novel feature fusion method embeds prompts into each patch encoding, which is processed by a frozen LLM, followed by a feature enhancement module and a time decoder module. During fine-tuning stage, cosine similarity between prompts and temporal patches is integrated into the loss function to boost performance. Experiments on benchmark datasets show that LTM significantly outperforms existing methods. It provides a robust and versatile solution for time series tasks. 

---
# Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM 

**Authors**: Yongqiang Yao, Jingru Tan, Kaihuan Liang, Feizhao Zhang, Yazhe Niu, Jiahao Hu, Ruihao Gong, Dahua Lin, Ningyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07680)  

**Abstract**: Training Long-Context Large Language Models (LLMs) is challenging, as hybrid training with long-context and short-context data often leads to workload imbalances. Existing works mainly use data packing to alleviate this issue but fail to consider imbalanced attention computation and wasted communication overhead. This paper proposes Hierarchical Balance Packing (HBP), which designs a novel batch-construction method and training recipe to address those inefficiencies. In particular, the HBP constructs multi-level data packing groups, each optimized with a distinct packing length. It assigns training samples to their optimal groups and configures each group with the most effective settings, including sequential parallelism degree and gradient checkpointing configuration. To effectively utilize multi-level groups of data, we design a dynamic training pipeline specifically tailored to HBP, including curriculum learning, adaptive sequential parallelism, and stable loss. Our extensive experiments demonstrate that our method significantly reduces training time over multiple datasets and open-source models while maintaining strong performance. For the largest DeepSeek-V2 (236B) MOE model, our method speeds up the training by 2.4$\times$ with competitive performance. 

---
# Using a single actor to output personalized policy for different intersections 

**Authors**: Kailing Zhou, Chengwei Zhang, Furui Zhan, Wanting Liu, Yihong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07678)  

**Abstract**: Recently, with the development of Multi-agent reinforcement learning (MARL), adaptive traffic signal control (ATSC) has achieved satisfactory results. In traffic scenarios with multiple intersections, MARL treats each intersection as an agent and optimizes traffic signal control strategies through learning and real-time decision-making. Considering that observation distributions of intersections might be different in real-world scenarios, shared parameter methods might lack diversity and thus lead to high generalization requirements in the shared-policy network. A typical solution is to increase the size of network parameters. However, simply increasing the scale of the network does not necessarily improve policy generalization, which is validated in our experiments. Accordingly, an approach that considers both the personalization of intersections and the efficiency of parameter sharing is required. To this end, we propose Hyper-Action Multi-Head Proximal Policy Optimization (HAMH-PPO), a Centralized Training with Decentralized Execution (CTDE) MARL method that utilizes a shared PPO policy network to deliver personalized policies for intersections with non-iid observation distributions. The centralized critic in HAMH-PPO uses graph attention units to calculate the graph representations of all intersections and outputs a set of value estimates with multiple output heads for each intersection. The decentralized execution actor takes the local observation history as input and output distributions of action as well as a so-called hyper-action to balance the multiple values estimated from the centralized critic to further guide the updating of TSC policies. The combination of hyper-action and multi-head values enables multiple agents to share a single actor-critic while achieving personalized policies. 

---
# PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity 

**Authors**: Kwanyoung Kim, Byeongsu Sim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07677)  

**Abstract**: Diffusion models have shown impressive results in generating high-quality conditional samples using guidance techniques such as Classifier-Free Guidance (CFG). However, existing methods often require additional training or neural function evaluations (NFEs), making them incompatible with guidance-distilled models. Also, they rely on heuristic approaches that need identifying target layers. In this work, we propose a novel and efficient method, termed PLADIS, which boosts pre-trained models (U-Net/Transformer) by leveraging sparse attention. Specifically, we extrapolate query-key correlations using softmax and its sparse counterpart in the cross-attention layer during inference, without requiring extra training or NFEs. By leveraging the noise robustness of sparse attention, our PLADIS unleashes the latent potential of text-to-image diffusion models, enabling them to excel in areas where they once struggled with newfound effectiveness. It integrates seamlessly with guidance techniques, including guidance-distilled models. Extensive experiments show notable improvements in text alignment and human preference, offering a highly efficient and universally applicable solution. 

---
# The Janus Face of Innovation: Global Disparities and Divergent Options 

**Authors**: Nihat Mugurtay  

**Link**: [PDF](https://arxiv.org/pdf/2503.07676)  

**Abstract**: This article examines how unequal access to AI innovation creates systemic challenges for developing countries. Differential access to AI innovation results from the acute competition between domestic and global actors. While developing nations contribute significantly to AI development through data annotation labor, they face limited access to advanced AI technologies and are increasingly caught between divergent regulatory approaches from democratic and authoritarian tendencies. This brief paper analyzes how more affordable AI engagement and Western countries' development cooperation present developing nations with a complex choice between accessibility and governance standards. I argue this challenge entails new institutional mechanisms for technology transfer and regulatory cooperation, while carefully balancing universal standards with local needs. In turn, good practices could help developing countries close the deepening gap of global technological divides, while ensuring responsible AI development in developing countries. 

---
# DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems 

**Authors**: Junwei Yu, Yepeng Ding, Hiroyuki Sato  

**Link**: [PDF](https://arxiv.org/pdf/2503.07675)  

**Abstract**: The emergence of Large Language Models (LLMs) in Multi-Agent Systems (MAS) has opened new possibilities for artificial intelligence, yet current implementations face significant challenges in resource management, task coordination, and system efficiency. While existing frameworks demonstrate the potential of LLM-based agents in collaborative problem-solving, they often lack sophisticated mechanisms for parallel execution and dynamic task management. This paper introduces DynTaskMAS, a novel framework that orchestrates asynchronous and parallel operations in LLM-based MAS through dynamic task graphs. The framework features four key innovations: (1) a Dynamic Task Graph Generator that intelligently decomposes complex tasks while maintaining logical dependencies, (2) an Asynchronous Parallel Execution Engine that optimizes resource utilization through efficient task scheduling, (3) a Semantic-Aware Context Management System that enables efficient information sharing among agents, and (4) an Adaptive Workflow Manager that dynamically optimizes system performance. Experimental evaluations demonstrate that DynTaskMAS achieves significant improvements over traditional approaches: a 21-33% reduction in execution time across task complexities (with higher gains for more complex tasks), a 35.4% improvement in resource utilization (from 65% to 88%), and near-linear throughput scaling up to 16 concurrent agents (3.47X improvement for 4X agents). Our framework establishes a foundation for building scalable, high-performance LLM-based multi-agent systems capable of handling complex, dynamic tasks efficiently. 

---
# TVNet: A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation 

**Authors**: Chenghan Li, Mingchen Li, Ruisheng Diao  

**Link**: [PDF](https://arxiv.org/pdf/2503.07674)  

**Abstract**: With the recent development and advancement of Transformer and MLP architectures, significant strides have been made in time series analysis. Conversely, the performance of Convolutional Neural Networks (CNNs) in time series analysis has fallen short of expectations, diminishing their potential for future applications. Our research aims to enhance the representational capacity of Convolutional Neural Networks (CNNs) in time series analysis by introducing novel perspectives and design innovations. To be specific, We introduce a novel time series reshaping technique that considers the inter-patch, intra-patch, and cross-variable dimensions. Consequently, we propose TVNet, a dynamic convolutional network leveraging a 3D perspective to employ time series analysis. TVNet retains the computational efficiency of CNNs and achieves state-of-the-art results in five key time series analysis tasks, offering a superior balance of efficiency and performance over the state-of-the-art Transformer-based and MLP-based models. Additionally, our findings suggest that TVNet exhibits enhanced transferability and robustness. Therefore, it provides a new perspective for applying CNN in advanced time series analysis tasks. 

---
# The potential role of AI agents in transforming nuclear medicine research and cancer management in India 

**Authors**: Rajat Vashistha, Arif Gulzar, Parveen Kundu, Punit Sharma, Mark Brunstein, Viktor Vegh  

**Link**: [PDF](https://arxiv.org/pdf/2503.07673)  

**Abstract**: India faces a significant cancer burden, with an incidence-to-mortality ratio indicating that nearly three out of five individuals diagnosed with cancer succumb to the disease. While the limitations of physical healthcare infrastructure are widely acknowledged as a primary challenge, concerted efforts by government and healthcare agencies are underway to mitigate these constraints. However, given the country's vast geography and high population density, it is imperative to explore alternative soft infrastructure solutions to complement existing frameworks. Artificial Intelligence agents are increasingly transforming problem-solving approaches across various domains, with their application in medicine proving particularly transformative. In this perspective, we examine the potential role of AI agents in advancing nuclear medicine for cancer research, diagnosis, and management in India. We begin with a brief overview of AI agents and their capabilities, followed by a proposed agent-based ecosystem that can address prevailing sustainability challenges in India nuclear medicine. 

---
# Probabilistic Shielding for Safe Reinforcement Learning 

**Authors**: Edwin Hamel-De le Court, Francesco Belardinelli, Alex W. Goodall  

**Link**: [PDF](https://arxiv.org/pdf/2503.07671)  

**Abstract**: In real-life scenarios, a Reinforcement Learning (RL) agent aiming to maximise their reward, must often also behave in a safe manner, including at training time. Thus, much attention in recent years has been given to Safe RL, where an agent aims to learn an optimal policy among all policies that satisfy a given safety constraint. However, strict safety guarantees are often provided through approaches based on linear programming, and thus have limited scaling. In this paper we present a new, scalable method, which enjoys strict formal guarantees for Safe RL, in the case where the safety dynamics of the Markov Decision Process (MDP) are known, and safety is defined as an undiscounted probabilistic avoidance property. Our approach is based on state-augmentation of the MDP, and on the design of a shield that restricts the actions available to the agent. We show that our approach provides a strict formal safety guarantee that the agent stays safe at training and test time. Furthermore, we demonstrate that our approach is viable in practice through experimental evaluation. 

---
# Data Foundations for Large Scale Multimodal Clinical Foundation Models 

**Authors**: Wei Dai, Peilin Chen, Malinda Lu, Daniel Li, Haowen Wei, Hejie Cui, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07667)  

**Abstract**: Recent advances in clinical AI have enabled remarkable progress across many clinical domains. However, existing benchmarks and models are primarily limited to a small set of modalities and tasks, which hinders the development of large-scale multimodal methods that can make holistic assessments of patient health and well-being. To bridge this gap, we introduce Clinical Large-Scale Integrative Multimodal Benchmark (CLIMB), a comprehensive clinical benchmark unifying diverse clinical data across imaging, language, temporal, and graph modalities. CLIMB comprises 4.51 million patient samples totaling 19.01 terabytes distributed across 2D imaging, 3D video, time series, graphs, and multimodal data. Through extensive empirical evaluation, we demonstrate that multitask pretraining significantly improves performance on understudied domains, achieving up to 29% improvement in ultrasound and 23% in ECG analysis over single-task learning. Pretraining on CLIMB also effectively improves models' generalization capability to new tasks, and strong unimodal encoder performance translates well to multimodal performance when paired with task-appropriate fusion strategies. Our findings provide a foundation for new architecture designs and pretraining strategies to advance clinical AI research. Code is released at this https URL. 

---
# Merge then Realign: Simple and Effective Modality-Incremental Continual Learning for Multimodal LLMs 

**Authors**: Dingkun Zhang, Shuhan Qi, Xinyu Xiao, Kehai Chen, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07663)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have enhanced their versatility as they integrate a growing number of modalities. Considering the heavy cost of training MLLMs, it is necessary to reuse the existing ones and further extend them to more modalities through Modality-incremental Continual Learning (MCL). However, this often comes with a performance degradation in the previously learned modalities. In this work, we revisit the MCL and investigate a more severe issue it faces in contrast to traditional continual learning, that its degradation comes not only from catastrophic forgetting but also from the misalignment between the modality-agnostic and modality-specific components. To address this problem, we propose an elegantly simple MCL paradigm called "MErge then ReAlign" (MERA). Our method avoids introducing heavy training overhead or modifying the model architecture, hence is easy to deploy and highly reusable in the MLLM community. Extensive experiments demonstrate that, despite the simplicity of MERA, it shows impressive performance, holding up to a 99.84% Backward Relative Gain when extending to four modalities, achieving a nearly lossless MCL performance. 

---
# Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy 

**Authors**: Wei Junhao, Yu Zhe, Sakuma Jun  

**Link**: [PDF](https://arxiv.org/pdf/2503.07661)  

**Abstract**: Model merging is a technique that combines multiple finetuned models into a single model without additional training, allowing a free-rider to cheaply inherit specialized capabilities. This study investigates methodologies to suppress unwanted model merging by free-riders. Existing methods such as model watermarking or fingerprinting can only detect merging in hindsight. In contrast, we propose a first proactive defense against model merging. Specifically, our defense method modifies the model parameters so that the model is disrupted if the model is merged with any other model, while its functionality is kept unchanged if not merged with others. Our approach consists of two modules, rearranging MLP parameters and scaling attention heads, which push the model out of the shared basin in parameter space, causing the merging performance with other models to degrade significantly. We conduct extensive experiments on image classification, image generation, and text classification to demonstrate that our defense severely disrupts merging while retaining the functionality of the post-protect model. Moreover, we analyze potential adaptive attacks and further propose a dropout-based pruning to improve our proposal's robustness. 

---
# SplitQuantV2: Enhancing Low-Bit Quantization of LLMs Without GPUs 

**Authors**: Jaewoo Song, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07657)  

**Abstract**: The quantization of large language models (LLMs) is crucial for deploying them on devices with limited computational resources. While advanced quantization algorithms offer improved performance compared to the basic linear quantization, they typically require high-end graphics processing units (GPUs), are often restricted to specific deep neural network (DNN) frameworks, and require calibration datasets. This limitation poses challenges for using such algorithms on various neural processing units (NPUs) and edge AI devices, which have diverse model formats and frameworks. In this paper, we show SplitQuantV2, an innovative algorithm designed to enhance low-bit linear quantization of LLMs, can achieve results comparable to those of advanced algorithms. SplitQuantV2 preprocesses models by splitting linear and convolution layers into functionally equivalent, quantization-friendly structures. The algorithm's platform-agnostic, concise, and efficient nature allows for implementation without the need for GPUs. Our evaluation on the Llama 3.2 1B Instruct model using the AI2's Reasoning Challenge (ARC) dataset demonstrates that SplitQuantV2 improves the accuracy of the INT4 quantization model by 11.76%p, matching the performance of the original floating-point model. Remarkably, SplitQuantV2 took only 2 minutes 6 seconds to preprocess the 1B model and perform linear INT4 quantization using only an Apple M4 CPU. SplitQuantV2 provides a practical solution for low-bit quantization on LLMs, especially when complex, computation-intensive algorithms are inaccessible due to hardware limitations or framework incompatibilities. 

---
# GraphT5: Unified Molecular Graph-Language Modeling via Multi-Modal Cross-Token Attention 

**Authors**: Sangyeup Kim, Nayeon Kim, Yinhua Piao, Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07655)  

**Abstract**: Molecular language modeling tasks such as molecule captioning have been recognized for their potential to further understand molecular properties that can aid drug discovery or material synthesis based on chemical reactions. Unlike the common use of molecule graphs in predicting molecular properties, most methods in molecular language modeling rely heavily on SMILES sequences. This preference is because the task involves generating a sequence of multiple tokens using transformer-based models. Therefore, a main challenge is determining how to integrate graph data, which contains structural and spatial information about molecules, with text data. In addition, simply using both 1D SMILES text and 2D graph as inputs without addressing how they align and represent the molecule structure in different modalities makes it challenging to fully utilize structural knowledge about molecules. To this end, we propose GraphT5, a multi-modal framework that integrates 1D SMILES text and 2D graph representations of molecules for molecular language modeling. Specifically, we introduce a novel cross-token attention module in GraphT5 to bridge the gap arising from the fundamental differences between the two modalities of molecule representations. Cross-token attention exploits implicit information between SMILES and graphs of molecules, resulting from their interactions at a fine-grained token level that benefits molecular language modeling. Extensive experiments including molecule captioning, IUPAC name prediction tasks, and case studies show that our GraphT5 outperforms the latest baseline approaches, which validates the effectiveness of our GraphT5 in sufficiently utilizing 1D SMILES text and 2D graph representations. 

---
# Insights into Schizophrenia: Leveraging Machine Learning for Early Identification via EEG, ERP, and Demographic Attributes 

**Authors**: Sara Alkhalifa  

**Link**: [PDF](https://arxiv.org/pdf/2503.07650)  

**Abstract**: The research presents a machine learning (ML) classifier designed to differentiate between schizophrenia patients and healthy controls by utilising features extracted from electroencephalogram (EEG) data, specifically focusing on event-related potentials (ERPs) and certain demographic variables. The dataset comprises data from 81 participants, encompassing 32 healthy controls and 49 schizophrenia patients, all sourced from an online dataset. After preprocessing the dataset, our ML model achieved an accuracy of 99.980%. This performance outperforms earlier research, including those that used deep learning methods. Additionally, an analysis was conducted to assess individual features' contribution to improving classification accuracy. This involved systematically excluding specific features from the original dataset one at a time, and another technique involved an iterative process of removing features based on their entropy scores incrementally. The impact of these removals on model performance was evaluated to identify the most informative features. 

---
# TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster 

**Authors**: Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang, James Y. Zhang, Kashif Rasul, Anderson Schneider, Lintao Ma, Yuriy Nevmyvaka, Dongjin Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.07649)  

**Abstract**: Recently, Large Language Models (LLMs) and Foundation Models (FMs) have become prevalent for time series forecasting tasks. However, fine-tuning large language models (LLMs) for forecasting enables the adaptation to specific domains but may not generalize well across diverse, unseen datasets. Meanwhile, existing time series foundation models (TSFMs) lack inherent mechanisms for domain adaptation and suffer from limited interpretability, making them suboptimal for zero-shot forecasting. To this end, we present TS-RAG, a retrieval-augmented generation based time series forecasting framework that enhances the generalization capability and interpretability of TSFMs. Specifically, TS-RAG leverages pre-trained time series encoders to retrieve semantically relevant time series segments from a dedicated knowledge database, incorporating contextual patterns for the given time series query. Next, we develop a learnable Mixture-of-Experts (MoE)-based augmentation module, which dynamically fuses retrieved time series patterns with the TSFM's representation of the input query, improving forecasting accuracy without requiring task-specific fine-tuning. Thorough empirical studies on seven public benchmark datasets demonstrate that TS-RAG achieves state-of-the-art zero-shot forecasting performance, outperforming TSFMs by up to 6.51% across diverse domains and showcasing desired interpretability. 

---
# ConstellationNet: Reinventing Spatial Clustering through GNNs 

**Authors**: Aidan Gao, Junhong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07643)  

**Abstract**: Spatial clustering is a crucial field, finding universal use across criminology, pathology, and urban planning. However, most spatial clustering algorithms cannot pull information from nearby nodes and suffer performance drops when dealing with higher dimensionality and large datasets, making them suboptimal for large-scale and high-dimensional clustering. Due to modern data growing in size and dimension, clustering algorithms become weaker when addressing multifaceted issues. To improve upon this, we develop ConstellationNet, a convolution neural network(CNN)-graph neural network(GNN) framework that leverages the embedding power of a CNN, the neighbor aggregation of a GNN, and a neural network's ability to deal with batched data to improve spatial clustering and classification with graph augmented predictions. ConstellationNet achieves state-of-the-art performance on both supervised classification and unsupervised clustering across several datasets, outperforming state-of-the-art classification and clustering while reducing model size and training time by up to tenfold and improving baselines by 10 times. Because of its fast training and powerful nature, ConstellationNet holds promise in fields like epidemiology and medical imaging, able to quickly train on new data to develop robust responses. 

---
# Deep ARTMAP: Generalized Hierarchical Learning with Adaptive Resonance Theory 

**Authors**: Niklas M. Melton, Leonardo Enzo Brito da Silva, Sasha Petrenko, Donald. C. Wunsch II  

**Link**: [PDF](https://arxiv.org/pdf/2503.07641)  

**Abstract**: This paper presents Deep ARTMAP, a novel extension of the ARTMAP architecture that generalizes the self-consistent modular ART (SMART) architecture to enable hierarchical learning (supervised and unsupervised) across arbitrary transformations of data. The Deep ARTMAP framework operates as a divisive clustering mechanism, supporting an arbitrary number of modules with customizable granularity within each module. Inter-ART modules regulate the clustering at each layer, permitting unsupervised learning while enforcing a one-to-many mapping from clusters in one layer to the next. While Deep ARTMAP reduces to both ARTMAP and SMART in particular configurations, it offers significantly enhanced flexibility, accommodating a broader range of data transformations and learning modalities. 

---
# BrainNet-MoE: Brain-Inspired Mixture-of-Experts Learning for Neurological Disease Identification 

**Authors**: Jing Zhang, Xiaowei Yu, Tong Chen, Chao Cao, Mingheng Chen, Yan Zhuang, Yanjun Lyu, Lu Zhang, Li Su, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07640)  

**Abstract**: The Lewy body dementia (LBD) is the second most common neurodegenerative dementia after Alzheimer's disease (AD). Early differentiation between AD and LBD is crucial because they require different treatment approaches, but this is challenging due to significant clinical overlap, heterogeneity, complex pathogenesis, and the rarity of LBD. While recent advances in artificial intelligence (AI) demonstrate powerful learning capabilities and offer new hope for accurate diagnosis, existing methods primarily focus on designing "neural-level networks". Our work represents a pioneering effort in modeling system-level artificial neural network called BrainNet-MoE for brain modeling and diagnosing. Inspired by the brain's hierarchical organization of bottom-up sensory integration and top-down control, we design a set of disease-specific expert groups to process brain sub-network under different condition, A disease gate mechanism guides the specializa-tion of expert groups, while a transformer layer enables communication be-tween all sub-networks, generating a comprehensive whole-brain represen-tation for downstream disease classification. Experimental results show superior classification accuracy with interpretable insights into how brain sub-networks contribute to different neurodegenerative conditions. 

---
# Leveraging Taxonomy Similarity for Next Activity Prediction in Patient Treatment 

**Authors**: Martin Kuhn, Joscha Grüger, Tobias Geyer, Ralph Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.07638)  

**Abstract**: The rapid progress in modern medicine presents physicians with complex challenges when planning patient treatment. Techniques from the field of Predictive Business Process Monitoring, like Next-activity-prediction (NAP) can be used as a promising technique to support physicians in treatment planning, by proposing a possible next treatment step. Existing patient data, often in the form of electronic health records, can be analyzed to recommend the next suitable step in the treatment process. However, the use of patient data poses many challenges due to its knowledge-intensive character, high variability and scarcity of medical data. To overcome these challenges, this article examines the use of the knowledge encoded in taxonomies to improve and explain the prediction of the next activity in the treatment process. This study proposes the TS4NAP approach, which uses medical taxonomies (ICD-10-CM and ICD-10-PCS) in combination with graph matching to assess the similarities of medical codes to predict the next treatment step. The effectiveness of the proposed approach will be evaluated using event logs that are derived from the MIMIC-IV dataset. The results highlight the potential of using domain-specific knowledge held in taxonomies to improve the prediction of the next activity, and thus can improve treatment planning and decision-making by making the predictions more explainable. 

---
# An Optimization Algorithm for Multimodal Data Alignment 

**Authors**: Wei Zhang, Xinyue Wang, Lan Yu, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07636)  

**Abstract**: In the data era, the integration of multiple data types, known as multimodality, has become a key area of interest in the research community. This interest is driven by the goal to develop cutting edge multimodal models capable of serving as adaptable reasoning engines across a wide range of modalities and domains. Despite the fervent development efforts, the challenge of optimally representing different forms of data within a single unified latent space a crucial step for enabling effective multimodal reasoning has not been fully addressed. To bridge this gap, we introduce AlignXpert, an optimization algorithm inspired by Kernel CCA crafted to maximize the similarities between N modalities while imposing some other constraints. This work demonstrates the impact on improving data representation for a variety of reasoning tasks, such as retrieval and classification, underlining the pivotal importance of data representation. 

---
# Psychological Counseling Ability of Large Language Models 

**Authors**: Fangyu Peng, Jingxin Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.07627)  

**Abstract**: With the development of science and the continuous progress of artificial intelligence technology, Large Language Models (LLMs) have begun to be widely utilized across various fields. However, in the field of psychological counseling, the ability of LLMs have not been systematically assessed. In this study, we assessed the psychological counseling ability of mainstream LLMs using 1096 psychological counseling skill questions which were selected from the Chinese National Counselor Level 3 Examination, including Knowledge-based, Analytical-based, and Application-based question types. The analysis showed that the correctness rates of the LLMs for Chinese questions, in descending order, were GLM-3 (46.5%), GPT-4 (46.1%), Gemini (45.0%), ERNIE-3.5 (45.7%) and GPT-3.5 (32.9%). The correctness rates of the LLMs for English questions, in descending order, were ERNIE-3.5 (43.9%), GPT-4 (40.6%), Gemini (36.6%), GLM-3 (29.9%) and GPT-3.5 (29.5%). A chi-square test indicated significant differences in the LLMs' performance on Chinese and English questions. Furthermore, we subsequently utilized the Counselor's Guidebook (Level 3) as a reference for ERNIE-3.5, resulting in a new correctness rate of 59.6%, a 13.8% improvement over its initial rate of 45.8%. In conclusion, the study assessed the psychological counseling ability of LLMs for the first time, which may provide insights for future enhancement and improvement of psychological counseling ability of LLMs. 

---
# Junior Software Developers' Perspectives on Adopting LLMs for Software Engineering: a Systematic Literature Review 

**Authors**: Samuel Ferino, Rashina Hoda, John Grundy, Christoph Treude  

**Link**: [PDF](https://arxiv.org/pdf/2503.07556)  

**Abstract**: Many studies exploring the adoption of Large Language Model-based tools for software development by junior developers have emerged in recent years. These studies have sought to understand developers' perspectives about using those tools, a fundamental pillar for successfully adopting LLM-based tools in Software Engineering. The aim of this paper is to provide an overview of junior software developers' perspectives and use of LLM-based tools for software engineering (LLM4SE). We conducted a systematic literature review (SLR) following guidelines by Kitchenham et al. on 56 primary studies, applying the definition for junior software developers as software developers with equal or less than five years of experience, including Computer Science/Software Engineering students. We found that the majority of the studies focused on comprehending the different aspects of integrating AI tools in SE. Only 8.9\% of the studies provide a clear definition for junior software developers, and there is no uniformity. Searching for relevant information is the most common task using LLM tools. ChatGPT was the most common LLM tool present in the studies (and experiments). A majority of the studies (83.9\%) report both positive and negative perceptions about the impact of adopting LLM tools. We also found and categorised advantages, challenges, and recommendations regarding LLM adoption. Our results indicate that developers are using LLMs not just for code generation, but also to improve their development skills. Critically, they are not just experiencing the benefits of adopting LLM tools, but they are also aware of at least a few LLM limitations, such as the generation of wrong suggestions, potential data leaking, and AI hallucination. Our findings offer implications for software engineering researchers, educators, and developers. 

---
# BIPED: Pedagogically Informed Tutoring System for ESL Education 

**Authors**: Soonwoo Kwon, Sojung Kim, Minju Park, Seunghyun Lee, Kyuseok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2406.03486)  

**Abstract**: Large Language Models (LLMs) have a great potential to serve as readily available and cost-efficient Conversational Intelligent Tutoring Systems (CITS) for teaching L2 learners of English. Existing CITS, however, are designed to teach only simple concepts or lack the pedagogical depth necessary to address diverse learning strategies. To develop a more pedagogically informed CITS capable of teaching complex concepts, we construct a BIlingual PEDagogically-informed Tutoring Dataset (BIPED) of one-on-one, human-to-human English tutoring interactions. Through post-hoc analysis of the tutoring interactions, we come up with a lexicon of dialogue acts (34 tutor acts and 9 student acts), which we use to further annotate the collected dataset. Based on a two-step framework of first predicting the appropriate tutor act then generating the corresponding response, we implemented two CITS models using GPT-4 and SOLAR-KO, respectively. We experimentally demonstrate that the implemented models not only replicate the style of human teachers but also employ diverse and contextually appropriate pedagogical strategies. 

---
# Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach 

**Authors**: Soonwoo Kwon, Sojung Kim, Seunghyun Lee, Jin-Young Kim, Suyeong An, Kyuseok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2308.11912)  

**Abstract**: Computerized Adaptive Testing (CAT) is a widely used, efficient test mode that adapts to the examinee's proficiency level in the test domain. CAT requires pre-trained item profiles, for CAT iteratively assesses the student real-time based on the registered items' profiles, and selects the next item to administer using candidate items' profiles. However, obtaining such item profiles is a costly process that involves gathering a large, dense item-response data, then training a diagnostic model on the collected data. In this paper, we explore the possibility of leveraging response data collected in the CAT service. We first show that this poses a unique challenge due to the inherent selection bias introduced by CAT, i.e., more proficient students will receive harder questions. Indeed, when naively training the diagnostic model using CAT response data, we observe that item profiles deviate significantly from the ground-truth. To tackle the selection bias issue, we propose the user-wise aggregate influence function method. Our intuition is to filter out users whose response data is heavily biased in an aggregate manner, as judged by how much perturbation the added data will introduce during parameter estimation. This way, we may enhance the performance of CAT while introducing minimal bias to the item profiles. We provide extensive experiments to demonstrate the superiority of our proposed method based on the three public datasets and one dataset that contains real-world CAT response data. 

---
# Principal deuterium Hugoniot via Quantum Monte Carlo and $Δ$-learning 

**Authors**: Giacomo Tenti, Kousuke Nakano, Andrea Tirelli, Sandro Sorella, Michele Casula  

**Link**: [PDF](https://arxiv.org/pdf/2301.03570)  

**Abstract**: We present a study of the principal deuterium Hugoniot for pressures up to $150$ GPa, using Machine Learning potentials (MLPs) trained with Quantum Monte Carlo (QMC) energies, forces and pressures. In particular, we adopted a recently proposed workflow based on the combination of Gaussian kernel regression and $\Delta$-learning. By fully taking advantage of this method, we explicitly considered finite-temperature electrons in the dynamics, whose effects are highly relevant for temperatures above $10$ kK. The Hugoniot curve obtained by our MLPs shows a good agreement with the most recent experiments, particularly in the region below 60 GPa. At larger pressures, our Hugoniot curve is slightly more compressible than the one yielded by experiments, whose uncertainties generally increase, however, with pressure. Our work demonstrates that QMC can be successfully combined with $\Delta$-learning to deploy reliable MLPs for complex extended systems across different thermodynamic conditions, by keeping the QMC precision at the computational cost of a mean-field calculation. 

---
