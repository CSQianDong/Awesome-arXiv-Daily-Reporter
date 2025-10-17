# Agentic Design of Compositional Machines 

**Authors**: Wenqian Zhang, Weiyang Liu, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14980)  

**Abstract**: The design of complex machines stands as both a marker of human intelligence and a foundation of engineering practice. Given recent advances in large language models (LLMs), we ask whether they, too, can learn to create. We approach this question through the lens of compositional machine design: a task in which machines are assembled from standardized components to meet functional demands like locomotion or manipulation in a simulated physical environment. To support this investigation, we introduce BesiegeField, a testbed built on the machine-building game Besiege, which enables part-based construction, physical simulation and reward-driven evaluation. Using BesiegeField, we benchmark state-of-the-art LLMs with agentic workflows and identify key capabilities required for success, including spatial reasoning, strategic assembly, and instruction-following. As current open-source models fall short, we explore reinforcement learning (RL) as a path to improvement: we curate a cold-start dataset, conduct RL finetuning experiments, and highlight open challenges at the intersection of language, machine design, and physical reasoning. 

---
# GroundedPRM: Tree-Guided and Fidelity-Aware Process Reward Modeling for Step-Level Reasoning 

**Authors**: Yao Zhang, Yu Wu, Haowei Zhang, Weiguo Li, Haokun Chen, Jingpei Wu, Guohao Li, Zhen Han, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2510.14942)  

**Abstract**: Process Reward Models (PRMs) aim to improve multi-step reasoning in Large Language Models (LLMs) by supervising intermediate steps and identifying errors. However, building effective PRMs remains challenging due to the lack of scalable, high-quality annotations. Existing approaches rely on costly human labeling, LLM-based self-evaluation that is prone to hallucination, or Monte Carlo (MC) estimation, which infers step quality solely from rollout outcomes and often introduces noisy, misaligned supervision due to credit misattribution. These issues result in three core limitations: noisy rewards, low factual fidelity, and misalignment with step-level reasoning objectives. To address these challenges, we introduce GroundedPRM, a tree-guided and fidelity-aware framework for automatic process supervision. To reduce reward noise and enable fine-grained credit assignment, we construct structured reasoning paths via Monte Carlo Tree Search (MCTS). To eliminate hallucinated supervision, we validate each intermediate step using an external tool, providing execution-grounded correctness signals. To combine both step-level validation and global outcome assessment, we design a hybrid reward aggregation mechanism that fuses tool-based verification with MCTS-derived feedback. Finally, we format the reward signal into a rationale-enhanced, generative structure to promote interpretability and compatibility with instruction-tuned LLMs. GroundedPRM is trained on only 40K automatically labeled samples, amounting to just 10% of the data used by the best-performing PRM trained with auto-labeled supervision. Nevertheless, it achieves up to a 26% relative improvement in average performance on ProcessBench. When used for reward-guided greedy search, GroundedPRM outperforms even PRMs trained with human-labeled supervision, offering a scalable and verifiable path toward high-quality process-level reasoning. 

---
# Stable but Miscalibrated: A Kantian View on Overconfidence from Filters to Large Language Models 

**Authors**: Akira Okutomi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14925)  

**Abstract**: We reinterpret Kant's Critique of Pure Reason as a theory of feedback stability, viewing reason as a regulator that keeps inference within the bounds of possible experience. We formalize this intuition via a composite instability index (H-Risk) combining spectral margin, conditioning, temporal sensitivity, and innovation amplification. In linear-Gaussian simulations, higher H-Risk predicts overconfident errors even under formal stability, revealing a gap between nominal and epistemic stability. Extending to large language models (LLMs), we find that fragile internal dynamics correlate with miscalibration and hallucination, while critique-style prompts show mixed effects on calibration and hallucination. These results suggest a structural bridge between Kantian self-limitation and feedback control, offering a principled lens for diagnosing -- and selectively reducing -- overconfidence in reasoning systems. This is a preliminary version; supplementary experiments and broader replication will be reported in a future revision. 

---
# TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG 

**Authors**: Annisaa Fitri Nurfidausi, Eleonora Mancini, Paolo Torroni  

**Link**: [PDF](https://arxiv.org/pdf/2510.14922)  

**Abstract**: Depression is a widespread mental health disorder, yet its automatic detection remains challenging. Prior work has explored unimodal and multimodal approaches, with multimodal systems showing promise by leveraging complementary signals. However, existing studies are limited in scope, lack systematic comparisons of features, and suffer from inconsistent evaluation protocols. We address these gaps by systematically exploring feature representations and modelling strategies across EEG, together with speech and text. We evaluate handcrafted features versus pre-trained embeddings, assess the effectiveness of different neural encoders, compare unimodal, bimodal, and trimodal configurations, and analyse fusion strategies with attention to the role of EEG. Consistent subject-independent splits are applied to ensure robust, reproducible benchmarking. Our results show that (i) the combination of EEG, speech and text modalities enhances multimodal detection, (ii) pretrained embeddings outperform handcrafted features, and (iii) carefully designed trimodal models achieve state-of-the-art performance. Our work lays the groundwork for future research in multimodal depression detection. 

---
# Budget-aware Test-time Scaling via Discriminative Verification 

**Authors**: Kyle Montgomery, Sijun Tan, Yuqi Chen, Siyuan Zhuang, Tianjun Zhang, Raluca Ada Popa, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14913)  

**Abstract**: Test-time scaling is a powerful strategy for boosting the performance of large language models on complex reasoning tasks. While state-of-the-art approaches often employ generative verifiers to select the best solution from a pool of candidates, this method incurs prohibitive computational costs, limiting its practicality. In this work, we shift the focus to a more budget-aware paradigm: discriminative verification. We conduct a thorough empirical analysis and demonstrate that while discriminative verifiers may underperform in isolation, combining them with self-consistency in a hybrid approach creates a powerful and efficient test-time scaling mechanism. Notably, under a fixed compute budget, this hybrid approach surpasses state-of-the-art generative verification by a significant margin: achieving up to 15.3\% higher accuracy on AIME2025. Our findings establish that for practical, real-world applications, budget-aware scaling with discriminative verifiers is not only a "free" upgrade over self-consistency, but also a more effective and efficient alternative to costly generative techniques. Code is available at this https URL. 

---
# Mapping Smarter, Not Harder: A Test-Time Reinforcement Learning Agent That Improves Without Labels or Model Updates 

**Authors**: Wen-Kwang Tsao, Yao-Ching Yu, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14900)  

**Abstract**: The Enterprise Intelligence Platform must integrate logs from numerous third-party vendors in order to perform various downstream tasks. However, vendor documentation is often unavailable at test time. It is either misplaced, mismatched, poorly formatted, or incomplete, which makes schema mapping challenging. We introduce a reinforcement learning agent that can self-improve without labeled examples or model weight updates. During inference, the agent: 1) Identifies ambiguous field-mapping attempts. 2) Generates targeted web-search queries to gather external evidence. 3) Applies a confidence-based reward to iteratively refine its mappings. To demonstrate this concept, we converted Microsoft Defender for Endpoint logs into a common schema. Our method increased mapping accuracy from 56.4\%(LLM-only) to 72.73\%(RAG) to 93.94\% over 100 iterations using GPT-4o. At the same time, it reduced the number of low-confidence mappings requiring expert review by 85\%. This new approach provides an evidence-driven, transparent method for solving future industry problems, paving the way for more robust, accountable, scalable, efficient, flexible, adaptable, and collaborative solutions. 

---
# The Gatekeeper Knows Enough 

**Authors**: Fikresilase Wondmeneh Abebayew  

**Link**: [PDF](https://arxiv.org/pdf/2510.14881)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as autonomous agents, yet their practical utility is fundamentally constrained by a limited context window and state desynchronization resulting from the LLMs' stateless nature and inefficient context management. These limitations lead to unreliable output, unpredictable behavior, and inefficient resource usage, particularly when interacting with large, structured, and sensitive knowledge systems such as codebases and documents. To address these challenges, we introduce the Gatekeeper Protocol, a novel, domain-agnostic framework that governs agent-system interactions. Our protocol mandates that the agent first operate and reason on a minimalist, low-fidelity "latent state" representation of the system to strategically request high-fidelity context on demand. All interactions are mediated through a unified JSON format that serves as a declarative, state-synchronized protocol, ensuring the agent's model of the system remains verifiably grounded in the system's reality. We demonstrate the efficacy of this protocol with Sage, a reference implementation of the Gatekeeper Protocol for software development. Our results show that this approach significantly increases agent reliability, improves computational efficiency by minimizing token consumption, and enables scalable interaction with complex systems, creating a foundational methodology for building more robust, predictable, and grounded AI agents for any structured knowledge domain. 

---
# LabOS: The AI-XR Co-Scientist That Sees and Works With Humans 

**Authors**: Le Cong, Zaixi Zhang, Xiaotong Wang, Yin Di, Ruofan Jin, Michal Gerasimiuk, Yinkai Wang, Ravi K. Dinesh, David Smerkous, Alex Smerkous, Xuekun Wu, Shilong Liu, Peishan Li, Yi Zhu, Simran Serrao, Ning Zhao, Imran A. Mohammad, John B. Sunwoo, Joseph C. Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14861)  

**Abstract**: Modern science advances fastest when thought meets action. LabOS represents the first AI co-scientist that unites computational reasoning with physical experimentation through multimodal perception, self-evolving agents, and Entended-Reality(XR)-enabled human-AI collaboration. By connecting multi-model AI agents, smart glasses, and human-AI collaboration, LabOS allows AI to see what scientists see, understand experimental context, and assist in real-time execution. Across applications--from cancer immunotherapy target discovery to stem-cell engineering -- LabOS shows that AI can move beyond computational design to participation, turning the laboratory into an intelligent, collaborative environment where human and machine discovery evolve together. 

---
# Where to Search: Measure the Prior-Structured Search Space of LLM Agents 

**Authors**: Zhuo-Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.14846)  

**Abstract**: The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs. 

---
# Boosting Instruction Following at Scale 

**Authors**: Ben Elder, Evelyn Duesterwald, Vinod Muthusamy  

**Link**: [PDF](https://arxiv.org/pdf/2510.14842)  

**Abstract**: A typical approach developers follow to influence an LLM's behavior in an application is through careful manipulation of the prompt, such as by adding or modifying instructions. However, merely adding more instructions provides little assurance that they will actually be followed. We introduce Instruction Boosting as a post-generation method to increase the reliability of LLM prompt instructions. We show that Instruction Boosting improves the instruction following rate by up to 7 points for two instructions and up to 4 points for ten instructions. To demonstrate these results we introduce SCALEDIF, a benchmark with a scaled instruction volume of up to ten instructions per data sample. We also present an analysis of the commonly observed trend that performance degrades as more instructions are added. We show that an important factor contributing to this trend is the degree of tension and conflict that arises as the number of instructions is increased. We contribute a quantitative conflict scoring tool that explains the observed performance trends and provides feedback to developers on the impact that additional prompt instructions have on a model's performance. 

---
# RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning 

**Authors**: Jinrui Liu, Bingyan Nie, Boyu Li, Yaran Chen, Yuze Wang, Shunsen He, Haoran Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14828)  

**Abstract**: Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark. 

---
# Agentic NL2SQL to Reduce Computational Costs 

**Authors**: Dominik Jehle, Lennart Purucker, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2510.14808)  

**Abstract**: Translating natural language queries into SQL queries (NL2SQL or Text-to-SQL) has recently been empowered by large language models (LLMs). Using LLMs to perform NL2SQL methods on a large collection of SQL databases necessitates processing large quantities of meta-information about the databases, which in turn results in lengthy prompts with many tokens and high processing costs. To address this challenge, we introduce Datalake Agent, an agentic system designed to enable an LLM to solve NL2SQL tasks more efficiently. Instead of utilizing direct solvers for NL2SQL that call the LLM once with all meta-information in the prompt, the Datalake Agent employs an interactive loop to reduce the utilized meta-information. Within the loop, the LLM is used in a reasoning framework that selectively requests only the necessary information to solve a table question answering task. We evaluate the Datalake Agent on a collection of 23 databases with 100 table question answering tasks. The Datalake Agent reduces the tokens used by the LLM by up to 87\% and thus allows for substantial cost reductions while maintaining competitive performance. 

---
# SimKO: Simple Pass@K Policy Optimization 

**Authors**: Ruotian Peng, Yi Ren, Zhouliang Yu, Weiyang Liu, Yandong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14807)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has advanced the reasoning capabilities of large language models (LLMs). However, prevailing RLVR methods exhibit a systematic bias toward exploitation over exploration, as evidenced by improved pass@1 but reduced pass@K (K>1) performance. To understand this issue, we analyze training dynamics of RLVR methods by tracking the token-level probability distributions over vocabulary candidates. Our analysis reveals a consistent probability concentration effect where the top-1 candidate increasingly accumulates probability mass and suppresses that of other candidates. More importantly, stronger over-concentration correlates with worse pass@K performance. Inspired by this finding, we propose Simple Pass@K Optimization (SimKO), a method designed to mitigate the over-concentration issue, thereby encouraging exploration. SimKO operates in an asymmetrical manner. For verified-correct responses, it boosts the probabilities of the top-K candidates. For verified-incorrect responses, it applies stronger penalties to the top-1 candidate. We observe that this asymmetric design is particularly effective at mitigating over-concentration when applied at tokens with high entropy. Across various math and logical-reasoning benchmarks, SimKO consistently yields higher pass@K for a wide range of K, providing a simple way to improve RLVR's exploration. 

---
# ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling 

**Authors**: Jianghao Lin, Yuanyuan Shi, Xin Peng, Renjie Ding, Hairui Wang, Yuxuan Peng, Bizhe Bai, Weixi Song, Fengshuo Bai, Huacan Chai, Weinan Zhang, Fei Huang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14703)  

**Abstract**: Large language models (LLMs) are increasingly demonstrating strong capabilities as autonomous agents, with function calling serving as a core mechanism for interaction with the environment. Meanwhile, inference scaling has become a cutting-edge technique to enhance LLM performance by allocating more computational resources during the inference process. However, current research on inference scaling primarily focuses on unstructured output generation tasks, leaving its application in structured outputs, like function calling, largely underexplored. To bridge this gap, we propose an inference scaling framework that combines fine-grained beam search with a process reward model, ToolPRM, which scores the internal steps of each single function call. To train ToolPRM, we construct the first fine-grained intra-call process supervision dataset, automatically annotated with function-masking techniques to provide step-level rewards for structured tool-use reasoning. Extensive experiments demonstrate that ToolPRM beats the coarse-grained and outcome reward models in terms of predictive accuracy, indicating its stronger capability in supervising the function calling inference process. Inference scaling technique equipped with ToolPRM also significantly improves the backbone model performance across various function calling tasks and benchmarks. More importantly, we reveal a key principle for applying inference scaling techniques to structured outputs: "explore more but retain less" due to the unrecoverability characteristics of structured function calling generation. 

---
# Cognitive-Aligned Spatio-Temporal Large Language Models For Next Point-of-Interest Prediction 

**Authors**: Penglong Zhai, Jie Li, Fanyi Di, Yue Liu, Yifang Yuan, Jie Huang, Peng Wu, Sicong Wang, Mingyang Yin, Tingting Hu, Yao Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14702)  

**Abstract**: The next point-of-interest (POI) recommendation task aims to predict the users' immediate next destinations based on their preferences and historical check-ins, holding significant value in location-based services. Recently, large language models (LLMs) have shown great potential in recommender systems, which treat the next POI prediction in a generative manner. However, these LLMs, pretrained primarily on vast corpora of unstructured text, lack the native understanding of structured geographical entities and sequential mobility patterns required for next POI prediction tasks. Moreover, in industrial-scale POI prediction applications, incorporating world knowledge and alignment of human cognition, such as seasons, weather conditions, holidays, and users' profiles (such as habits, occupation, and preferences), can enhance the user experience while improving recommendation performance. To address these issues, we propose CoAST (Cognitive-Aligned Spatial-Temporal LLMs), a framework employing natural language as an interface, allowing for the incorporation of world knowledge, spatio-temporal trajectory patterns, profiles, and situational information. Specifically, CoAST mainly comprises of 2 stages: (1) Recommendation Knowledge Acquisition through continued pretraining on the enriched spatial-temporal trajectory data of the desensitized users; (2) Cognitive Alignment to align cognitive judgments with human preferences using enriched training data through Supervised Fine-Tuning (SFT) and a subsequent Reinforcement Learning (RL) phase. Extensive offline experiments on various real-world datasets and online experiments deployed in "Guess Where You Go" of AMAP App homepage demonstrate the effectiveness of CoAST. 

---
# Purifying Task Vectors in Knowledge-Aware Subspace for Model Merging 

**Authors**: Bang An, Yibo Yang, Philip Torr, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14697)  

**Abstract**: Model merging aims to integrate task-specific abilities from individually fine-tuned models into a single model without extra training. In recent model merging methods, task vector has become a fundamental building block, as it can encapsulate the residual information from finetuning. However, the merged model often suffers from notable performance degradation due to the conflicts caused by task-irrelevant redundancy in task vectors. Existing efforts in overcoming redundancy by randomly dropping elements in the parameter space involves randomness and lacks knowledge awareness. To address these challenges, in this study, we propose Purifying TAsk Vectors (PAVE) in knowledge-aware subspace. Concretely, we sample some training examples from each task, and feed them into their corresponding fine-tuned models to acquire the covariance matrices before linear layers. We then perform a context-oriented singular value decomposition, which accentuates the weight components most relevant to the target knowledge. As a result, we can split fine-tuned model weights into task-relevant and redundant components in the knowledge-aware subspace, and purify the task vector by pruning the redundant components. To induce fair pruning efforts across models, we further introduce a spectral rank allocation strategy by optimizing a normalized activated pruning error. The task vector purification by our method as a plug-and-play scheme is applicable across various task vector-based merging methods to improve their performance. In experiments, we demonstrate the effectiveness of PAVE across a diverse set of merging methods, tasks, and model architectures. 

---
# Practical, Utilitarian Algorithm Configuration 

**Authors**: Devon Graham, Kevin Leyton-Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.14683)  

**Abstract**: Utilitarian algorithm configuration identifies a parameter setting for a given algorithm that maximizes a user's utility. Utility functions offer a theoretically well-grounded approach to optimizing decision-making under uncertainty and are flexible enough to capture a user's preferences over algorithm runtimes (e.g., they can describe a sharp cutoff after which a solution is no longer required, a per-hour cost for compute, or diminishing returns from algorithms that take longer to run). COUP is a recently-introduced utilitarian algorithm configuration procedure which was designed mainly to offer strong theoretical guarantees about the quality of the configuration it returns, with less attention paid to its practical performance. This paper closes that gap, bringing theoretically-grounded, utilitarian algorithm configuration to the point where it is competitive with widely used, heuristic configuration procedures that offer no performance guarantees. We present a series of improvements to COUP that improve its empirical performance without degrading its theoretical guarantees and demonstrate their benefit experimentally. Using a case study, we also illustrate ways of exploring the robustness of a given solution to the algorithm selection problem to variations in the utility function. 

---
# NAEL: Non-Anthropocentric Ethical Logic 

**Authors**: Bianca Maria Lerma, Rafael Peñaloza  

**Link**: [PDF](https://arxiv.org/pdf/2510.14676)  

**Abstract**: We introduce NAEL (Non-Anthropocentric Ethical Logic), a novel ethical framework for artificial agents grounded in active inference and symbolic reasoning. Departing from conventional, human-centred approaches to AI ethics, NAEL formalizes ethical behaviour as an emergent property of intelligent systems minimizing global expected free energy in dynamic, multi-agent environments. We propose a neuro-symbolic architecture to allow agents to evaluate the ethical consequences of their actions in uncertain settings. The proposed system addresses the limitations of existing ethical models by allowing agents to develop context-sensitive, adaptive, and relational ethical behaviour without presupposing anthropomorphic moral intuitions. A case study involving ethical resource distribution illustrates NAEL's dynamic balancing of self-preservation, epistemic learning, and collective welfare. 

---
# TITAN: Graph-Executable Reasoning for Cyber Threat Intelligence 

**Authors**: Marco Simoni, Aleksandar Fontana, Andrea Saracino, Paolo Mori  

**Link**: [PDF](https://arxiv.org/pdf/2510.14670)  

**Abstract**: TITAN (Threat Intelligence Through Automated Navigation) is a framework that connects natural-language cyber threat queries with executable reasoning over a structured knowledge graph. It integrates a path planner model, which predicts logical relation chains from text, and a graph executor that traverses the TITAN Ontology to retrieve factual answers and supporting evidence. Unlike traditional retrieval systems, TITAN operates on a typed, bidirectional graph derived from MITRE, allowing reasoning to move clearly and reversibly between threats, behaviors, and defenses. To support training and evaluation, we introduce the TITAN Dataset, a corpus of 88209 examples (Train: 74258; Test: 13951) pairing natural language questions with executable reasoning paths and step by step Chain of Thought explanations. Empirical evaluations show that TITAN enables models to generate syntactically valid and semantically coherent reasoning paths that can be deterministically executed on the underlying graph. 

---
# Machine Learning and Public Health: Identifying and Mitigating Algorithmic Bias through a Systematic Review 

**Authors**: Sara Altamirano, Arjan Vreeken, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2510.14669)  

**Abstract**: Machine learning (ML) promises to revolutionize public health through improved surveillance, risk stratification, and resource allocation. However, without systematic attention to algorithmic bias, ML may inadvertently reinforce existing health disparities. We present a systematic literature review of algorithmic bias identification, discussion, and reporting in Dutch public health ML research from 2021 to 2025. To this end, we developed the Risk of Algorithmic Bias Assessment Tool (RABAT) by integrating elements from established frameworks (Cochrane Risk of Bias, PROBAST, Microsoft Responsible AI checklist) and applied it to 35 peer-reviewed studies. Our analysis reveals pervasive gaps: although data sampling and missing data practices are well documented, most studies omit explicit fairness framing, subgroup analyses, and transparent discussion of potential harms. In response, we introduce a four-stage fairness-oriented framework called ACAR (Awareness, Conceptualization, Application, Reporting), with guiding questions derived from our systematic literature review to help researchers address fairness across the ML lifecycle. We conclude with actionable recommendations for public health ML practitioners to consistently consider algorithmic bias and foster transparency, ensuring that algorithmic innovations advance health equity rather than undermine it. 

---
# Beyond Hallucinations: The Illusion of Understanding in Large Language Models 

**Authors**: Rikard Rosenbacke, Carl Rosenbacke, Victor Rosenbacke, Martin McKee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14665)  

**Abstract**: Large language models (LLMs) are becoming deeply embedded in human communication and decision-making, yet they inherit the ambiguity, bias, and lack of direct access to truth inherent in language itself. While their outputs are fluent, emotionally resonant, and coherent, they are generated through statistical prediction rather than grounded reasoning. This creates the risk of hallucination, responses that sound convincing but lack factual validity. Building on Geoffrey Hinton's observation that AI mirrors human intuition rather than reasoning, this paper argues that LLMs operationalize System 1 cognition at scale: fast, associative, and persuasive, but without reflection or falsification. To address this, we introduce the Rose-Frame, a three-dimensional framework for diagnosing cognitive and epistemic drift in human-AI interaction. The three axes are: (i) Map vs. Territory, which distinguishes representations of reality (epistemology) from reality itself (ontology); (ii) Intuition vs. Reason, drawing on dual-process theory to separate fast, emotional judgments from slow, reflective thinking; and (iii) Conflict vs. Confirmation, which examines whether ideas are critically tested through disagreement or simply reinforced through mutual validation. Each dimension captures a distinct failure mode, and their combination amplifies misalignment. Rose-Frame does not attempt to fix LLMs with more data or rules. Instead, it offers a reflective tool that makes both the model's limitations and the user's assumptions visible, enabling more transparent and critically aware AI deployment. It reframes alignment as cognitive governance: intuition, whether human or artificial, must remain governed by human reason. Only by embedding reflective, falsifiable oversight can we align machine fluency with human understanding. 

---
# ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks 

**Authors**: Yuanyi Song, Heyuan Huang, Qiqiang Lin, Yin Zhao, Xiangmou Qu, Jun Wang, Xingyu Lou, Weiwen Liu, Zhuosheng Zhang, Jun Wang, Yong Yu, Weinan Zhang, Zhaoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14621)  

**Abstract**: The rapid advancement of multimodal large language models has enabled agents to operate mobile devices by directly interacting with graphical user interfaces, opening new possibilities for mobile automation. However, real-world mobile tasks are often complex and allow for multiple valid solutions. This contradicts current mobile agent evaluation standards: offline static benchmarks can only validate a single predefined "golden path", while online dynamic testing is constrained by the complexity and non-reproducibility of real devices, making both approaches inadequate for comprehensively assessing agent capabilities. To bridge the gap between offline and online evaluation and enhance testing stability, this paper introduces a novel graph-structured benchmarking framework. By modeling the finite states observed during real-device interactions, it achieves static simulation of dynamic behaviors. Building on this, we develop ColorBench, a benchmark focused on complex long-horizon tasks. It supports evaluation of multiple valid solutions, subtask completion rate statistics, and atomic-level capability analysis. ColorBench contains 175 tasks (74 single-app, 101 cross-app) with an average length of over 13 steps. Each task includes at least two correct paths and several typical error paths, enabling quasi-dynamic interaction. By evaluating ColorBench across various baselines, we discover limitations of existing models and propose improvement directions and feasible technical pathways to enhance agents' performance on complex, long-horizon problems based on experimental results. Code and data are available at: this https URL. 

---
# LLM Agents Beyond Utility: An Open-Ended Perspective 

**Authors**: Asen Nachkov, Xi Wang, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2510.14548)  

**Abstract**: Recent LLM agents have made great use of chain of thought reasoning and function calling. As their capabilities grow, an important question arises: can this software represent not only a smart problem-solving tool, but an entity in its own right, that can plan, design immediate tasks, and reason toward broader, more ambiguous goals? To study this question, we adopt an open-ended experimental setting where we augment a pretrained LLM agent with the ability to generate its own tasks, accumulate knowledge, and interact extensively with its environment. We study the resulting open-ended agent qualitatively. It can reliably follow complex multi-step instructions, store and reuse information across runs, and propose and solve its own tasks, though it remains sensitive to prompt design, prone to repetitive task generation, and unable to form self-representations. These findings illustrate both the promise and current limits of adapting pretrained LLMs toward open-endedness, and point to future directions for training agents to manage memory, explore productively, and pursue abstract long-term goals. 

---
# Symbol Grounding in Neuro-Symbolic AI: A Gentle Introduction to Reasoning Shortcuts 

**Authors**: Emanuele Marconato, Samuele Bortolotti, Emile van Krieken, Paolo Morettin, Elena Umili, Antonio Vergari, Efthymia Tsamoura, Andrea Passerini, Stefano Teso  

**Link**: [PDF](https://arxiv.org/pdf/2510.14538)  

**Abstract**: Neuro-symbolic (NeSy) AI aims to develop deep neural networks whose predictions comply with prior knowledge encoding, e.g. safety or structural constraints. As such, it represents one of the most promising avenues for reliable and trustworthy AI. The core idea behind NeSy AI is to combine neural and symbolic steps: neural networks are typically responsible for mapping low-level inputs into high-level symbolic concepts, while symbolic reasoning infers predictions compatible with the extracted concepts and the prior knowledge. Despite their promise, it was recently shown that - whenever the concepts are not supervised directly - NeSy models can be affected by Reasoning Shortcuts (RSs). That is, they can achieve high label accuracy by grounding the concepts incorrectly. RSs can compromise the interpretability of the model's explanations, performance in out-of-distribution scenarios, and therefore reliability. At the same time, RSs are difficult to detect and prevent unless concept supervision is available, which is typically not the case. However, the literature on RSs is scattered, making it difficult for researchers and practitioners to understand and tackle this challenging problem. This overview addresses this issue by providing a gentle introduction to RSs, discussing their causes and consequences in intuitive terms. It also reviews and elucidates existing theoretical characterizations of this phenomenon. Finally, it details methods for dealing with RSs, including mitigation and awareness strategies, and maps their benefits and limitations. By reformulating advanced material in a digestible form, this overview aims to provide a unifying perspective on RSs to lower the bar to entry for tackling them. Ultimately, we hope this overview contributes to the development of reliable NeSy and trustworthy AI models. 

---
# JSPLIT: A Taxonomy-based Solution for Prompt Bloating in Model Context Protocol 

**Authors**: Emanuele Antonioni, Stefan Markovic, Anirudha Shankar, Jaime Bernardo, Lovro Markovic, Silvia Pareti, Benedetto Proietti  

**Link**: [PDF](https://arxiv.org/pdf/2510.14537)  

**Abstract**: AI systems are continually evolving and advancing, and user expectations are concurrently increasing, with a growing demand for interactions that go beyond simple text-based interaction with Large Language Models (LLMs). Today's applications often require LLMs to interact with external tools, marking a shift toward more complex agentic systems. To support this, standards such as the Model Context Protocol (MCP) have emerged, enabling agents to access tools by including a specification of the capabilities of each tool within the prompt. Although this approach expands what agents can do, it also introduces a growing problem: prompt bloating. As the number of tools increases, the prompts become longer, leading to high prompt token costs, increased latency, and reduced task success resulting from the selection of tools irrelevant to the prompt. To address this issue, we introduce JSPLIT, a taxonomy-driven framework designed to help agents manage prompt size more effectively when using large sets of MCP tools. JSPLIT organizes the tools into a hierarchical taxonomy and uses the user's prompt to identify and include only the most relevant tools, based on both the query and the taxonomy structure. In this paper, we describe the design of the taxonomy, the tool selection algorithm, and the dataset used to evaluate JSPLIT. Our results show that JSPLIT significantly reduces prompt size without significantly compromising the agent's ability to respond effectively. As the number of available tools for the agent grows substantially, JSPLIT even improves the tool selection accuracy of the agent, effectively reducing costs while simultaneously improving task success in high-complexity agent environments. 

---
# Helmsman: Autonomous Synthesis of Federated Learning Systems via Multi-Agent Collaboration 

**Authors**: Haoyuan Li, Mathias Funk, Aaqib Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2510.14512)  

**Abstract**: Federated Learning (FL) offers a powerful paradigm for training models on decentralized data, but its promise is often undermined by the immense complexity of designing and deploying robust systems. The need to select, combine, and tune strategies for multifaceted challenges like data heterogeneity and system constraints has become a critical bottleneck, resulting in brittle, bespoke solutions. To address this, we introduce Helmsman, a novel multi-agent system that automates the end-to-end synthesis of federated learning systems from high-level user specifications. It emulates a principled research and development workflow through three collaborative phases: (1) interactive human-in-the-loop planning to formulate a sound research plan, (2) modular code generation by supervised agent teams, and (3) a closed-loop of autonomous evaluation and refinement in a sandboxed simulation environment. To facilitate rigorous evaluation, we also introduce AgentFL-Bench, a new benchmark comprising 16 diverse tasks designed to assess the system-level generation capabilities of agentic systems in FL. Extensive experiments demonstrate that our approach generates solutions competitive with, and often superior to, established hand-crafted baselines. Our work represents a significant step towards the automated engineering of complex decentralized AI systems. 

---
# Eliminating Negative Occurrences of Derived Predicates from PDDL Axioms 

**Authors**: Claudia Grundke, Gabriele Röger  

**Link**: [PDF](https://arxiv.org/pdf/2510.14412)  

**Abstract**: Axioms are a feature of the Planning Domain Definition Language PDDL that can be considered as a generalization of database query languages such as Datalog. The PDDL standard restricts negative occurrences of predicates in axiom bodies to predicates that are directly set by actions and not derived by axioms. In the literature, authors often deviate from this limitation and only require that the set of axioms is stratifiable. Both variants can express exactly the same queries as least fixed-point logic, indicating that negative occurrences of derived predicates can be eliminated. We present the corresponding transformation. 

---
# IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning 

**Authors**: Xikai Zhang, Bo Wang, Likang Xiao, Yongzhi Li, Quan Chen, Wenju Wu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14406)  

**Abstract**: Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size. 

---
# Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control 

**Authors**: Zhe Wu, Hongjin Lu, Junliang Xing, Changhao Zhang, Yin Zhu, Yuhao Yang, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Jun Wang, Yuanchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14388)  

**Abstract**: Building agents that autonomously operate mobile devices has attracted increasing attention. While Vision-Language Models (VLMs) show promise, most existing approaches rely on direct state-to-action mappings, which lack structured reasoning and planning, and thus generalize poorly to novel tasks or unseen UI layouts. We introduce Hi-Agent, a trainable hierarchical vision-language agent for mobile control, featuring a high-level reasoning model and a low-level action model that are jointly optimized. For efficient training, we reformulate multi-step decision-making as a sequence of single-step subgoals and propose a foresight advantage function, which leverages execution feedback from the low-level model to guide high-level optimization. This design alleviates the path explosion issue encountered by Group Relative Policy Optimization (GRPO) in long-horizon tasks and enables stable, critic-free joint training. Hi-Agent achieves a new State-Of-The-Art (SOTA) 87.9% task success rate on the Android-in-the-Wild (AitW) benchmark, significantly outperforming prior methods across three paradigms: prompt-based (AppAgent: 17.7%), supervised (Filtered BC: 54.5%), and reinforcement learning-based (DigiRL: 71.9%). It also demonstrates competitive zero-shot generalization on the ScreenSpot-v2 benchmark. On the more challenging AndroidWorld benchmark, Hi-Agent also scales effectively with larger backbones, showing strong adaptability in high-complexity mobile control scenarios. 

---
# Can MLLMs Absorb Math Reasoning Abilities from LLMs as Free Lunch? 

**Authors**: Yijie Hu, Zihao Zhou, Kaizhu Huang, Xiaowei Huang, Qiufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14387)  

**Abstract**: Math reasoning has been one crucial ability of large language models (LLMs), where significant advancements have been achieved in recent years. However, most efforts focus on LLMs by curating high-quality annotation data and intricate training (or inference) paradigms, while the math reasoning performance of multi-modal LLMs (MLLMs) remains lagging behind. Since the MLLM typically consists of an LLM and a vision block, we wonder: Can MLLMs directly absorb math reasoning abilities from off-the-shelf math LLMs without tuning? Recent model-merging approaches may offer insights into this question. However, they overlook the alignment between the MLLM and LLM, where we find that there is a large gap between their parameter spaces, resulting in lower performance. Our empirical evidence reveals two key factors behind this issue: the identification of crucial reasoning-associated layers in the model and the mitigation of the gaps in parameter space. Based on the empirical insights, we propose IP-Merging that first identifies the reasoning-associated parameters in both MLLM and Math LLM, then projects them into the subspace of MLLM, aiming to maintain the alignment, and finally merges parameters in this subspace. IP-Merging is a tuning-free approach since parameters are directly adjusted. Extensive experiments demonstrate that our IP-Merging method can enhance the math reasoning ability of MLLMs directly from Math LLMs without compromising their other capabilities. 

---
# AI for Service: Proactive Assistance with AI Glasses 

**Authors**: Zichen Wen, Yiyu Wang, Chenfei Liao, Boxue Yang, Junxian Li, Weifeng Liu, Haocong He, Bolong Feng, Xuyang Liu, Yuanhuiyi Lyu, Xu Zheng, Xuming Hu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14359)  

**Abstract**: In an era where AI is evolving from a passive tool into an active and adaptive companion, we introduce AI for Service (AI4Service), a new paradigm that enables proactive and real-time assistance in daily life. Existing AI services remain largely reactive, responding only to explicit user commands. We argue that a truly intelligent and helpful assistant should be capable of anticipating user needs and taking actions proactively when appropriate. To realize this vision, we propose Alpha-Service, a unified framework that addresses two fundamental challenges: Know When to intervene by detecting service opportunities from egocentric video streams, and Know How to provide both generalized and personalized services. Inspired by the von Neumann computer architecture and based on AI glasses, Alpha-Service consists of five key components: an Input Unit for perception, a Central Processing Unit for task scheduling, an Arithmetic Logic Unit for tool utilization, a Memory Unit for long-term personalization, and an Output Unit for natural human interaction. As an initial exploration, we implement Alpha-Service through a multi-agent system deployed on AI glasses. Case studies, including a real-time Blackjack advisor, a museum tour guide, and a shopping fit assistant, demonstrate its ability to seamlessly perceive the environment, infer user intent, and provide timely and useful assistance without explicit prompts. 

---
# Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction 

**Authors**: Xu Shen, Qi Zhang, Song Wang, Zhen Tan, Xinyu Zhao, Laura Yao, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Kwonjoon Lee, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14319)  

**Abstract**: Large Language Model based multi-agent systems (MAS) excel at collaborative problem solving but remain brittle to cascading errors: a single faulty step can propagate across agents and disrupt the trajectory. In this paper, we present MASC, a metacognitive framework that endows MAS with real-time, unsupervised, step-level error detection and self-correction. MASC rethinks detection as history-conditioned anomaly scoring via two complementary designs: (1) Next-Execution Reconstruction, which predicts the embedding of the next step from the query and interaction history to capture causal consistency, and (2) Prototype-Guided Enhancement, which learns a prototype prior over normal-step embeddings and uses it to stabilize reconstruction and anomaly scoring under sparse context (e.g., early steps). When an anomaly step is flagged, MASC triggers a correction agent to revise the acting agent's output before information flows downstream. On the Who&When benchmark, MASC consistently outperforms all baselines, improving step-level error detection by up to 8.47% AUC-ROC ; When plugged into diverse MAS frameworks, it delivers consistent end-to-end gains across architectures, confirming that our metacognitive monitoring and targeted correction can mitigate error propagation with minimal overhead. 

---
# Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies 

**Authors**: Mason Nakamura, Abhinav Kumar, Saaduddin Mahmud, Sahar Abdelnabi, Shlomo Zilberstein, Eugene Bagdasarian  

**Link**: [PDF](https://arxiv.org/pdf/2510.14312)  

**Abstract**: A multi-agent system (MAS) powered by large language models (LLMs) can automate tedious user tasks such as meeting scheduling that requires inter-agent collaboration. LLMs enable nuanced protocols that account for unstructured private data, user constraints, and preferences. However, this design introduces new risks, including misalignment and attacks by malicious parties that compromise agents or steal user data. In this paper, we propose the Terrarium framework for fine-grained study on safety, privacy, and security in LLM-based MAS. We repurpose the blackboard design, an early approach in multi-agent systems, to create a modular, configurable testbed for multi-agent collaboration. We identify key attack vectors such as misalignment, malicious agents, compromised communication, and data poisoning. We implement three collaborative MAS scenarios with four representative attacks to demonstrate the framework's flexibility. By providing tools to rapidly prototype, evaluate, and iterate on defenses and designs, Terrarium aims to accelerate progress toward trustworthy multi-agent systems. 

---
# A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space 

**Authors**: Bingjie Zhang, Yibo Yang, Renzhe, Dandan Guo, Jindong Gu, Philip Torr, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14301)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in diverse tasks, yet their safety alignment remains fragile during adaptation. Even when fine-tuning on benign data or with low-rank adaptation, pre-trained safety behaviors are easily degraded, leading to harmful responses in the fine-tuned models. To address this challenge, we propose GuardSpace, a guardrail framework for preserving safety alignment throughout fine-tuning, composed of two key components: a safety-sensitive subspace and a harmful-resistant null space. First, we explicitly decompose pre-trained weights into safety-relevant and safety-irrelevant components using covariance-preconditioned singular value decomposition, and initialize low-rank adapters from the safety-irrelevant ones, while freezing safety-relevant components to preserve their associated safety mechanism. Second, we construct a null space projector that restricts adapter updates from altering safe outputs on harmful prompts, thereby maintaining the original refusal behavior. Experiments with various pre-trained models on multiple downstream tasks demonstrate that GuardSpace achieves superior performance over existing methods. Notably, for Llama-2-7B-Chat fine-tuned on GSM8K, GuardSpace outperforms the state-of-the-art method AsFT, reducing the average harmful score from 14.4% to 3.6%, while improving the accuracy from from 26.0% to 28.0%. 

---
# MorphoBench: A Benchmark with Difficulty Adaptive to Model Reasoning 

**Authors**: Xukai Wang, Xuanbo Liu, Mingrui Chen, Haitian Zhong, Xuanlin Yang, Bohan Zeng, Jinbo Hu, Hao Liang, Junbo Niu, Xuchen Li, Ruitao Wu, Ruichuan An, Yang Shi, Liu Liu, Xu-Yao Zhang, Qiang Liu, Zhouchen Lin, Wentao Zhang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.14265)  

**Abstract**: With the advancement of powerful large-scale reasoning models, effectively evaluating the reasoning capabilities of these models has become increasingly important. However, existing benchmarks designed to assess the reasoning abilities of large models tend to be limited in scope and lack the flexibility to adapt their difficulty according to the evolving reasoning capacities of the models. To address this, we propose MorphoBench, a benchmark that incorporates multidisciplinary questions to evaluate the reasoning capabilities of large models and can adjust and update question difficulty based on the reasoning abilities of advanced models. Specifically, we curate the benchmark by selecting and collecting complex reasoning questions from existing benchmarks and sources such as Olympiad-level competitions. Additionally, MorphoBench adaptively modifies the analytical challenge of questions by leveraging key statements generated during the model's reasoning process. Furthermore, it includes questions generated using simulation software, enabling dynamic adjustment of benchmark difficulty with minimal resource consumption. We have gathered over 1,300 test questions and iteratively adjusted the difficulty of MorphoBench based on the reasoning capabilities of models such as o3 and GPT-5. MorphoBench enhances the comprehensiveness and validity of model reasoning evaluation, providing reliable guidance for improving both the reasoning abilities and scientific robustness of large models. The code has been released in this https URL. 

---
# Towards Agentic Self-Learning LLMs in Search Environment 

**Authors**: Wangtao Sun, Xiang Cheng, Jialin Fan, Yao Xu, Xing Yu, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14253)  

**Abstract**: We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at this https URL 

---
# LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild 

**Authors**: Jiayu Wang, Yifei Ming, Riya Dulepet, Qinglin Chen, Austin Xu, Zixuan Ke, Frederic Sala, Aws Albarghouthi, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.14240)  

**Abstract**: Deep research -- producing comprehensive, citation-grounded reports by searching and synthesizing information from hundreds of live web sources -- marks an important frontier for agentic systems. To rigorously evaluate this ability, four principles are essential: tasks should be (1) user-centric, reflecting realistic information needs, (2) dynamic, requiring up-to-date information beyond parametric knowledge, (3) unambiguous, ensuring consistent interpretation across users, and (4) multi-faceted and search-intensive, requiring search over numerous web sources and in-depth analysis. Existing benchmarks fall short of these principles, often focusing on narrow domains or posing ambiguous questions that hinder fair comparison. Guided by these principles, we introduce LiveResearchBench, a benchmark of 100 expert-curated tasks spanning daily life, enterprise, and academia, each requiring extensive, dynamic, real-time web search and synthesis. Built with over 1,500 hours of human labor, LiveResearchBench provides a rigorous basis for systematic evaluation. To evaluate citation-grounded long-form reports, we introduce DeepEval, a comprehensive suite covering both content- and report-level quality, including coverage, presentation, citation accuracy and association, consistency and depth of analysis. DeepEval integrates four complementary evaluation protocols, each designed to ensure stable assessment and high agreement with human judgments. Using LiveResearchBench and DeepEval, we conduct a comprehensive evaluation of 17 frontier deep research systems, including single-agent web search, single-agent deep research, and multi-agent systems. Our analysis reveals current strengths, recurring failure modes, and key system components needed to advance reliable, insightful deep research. 

---
# Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks 

**Authors**: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14207)  

**Abstract**: Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible. 

---
# Implementation of AI in Precision Medicine 

**Authors**: Göktuğ Bender, Samer Faraj, Anand Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2510.14194)  

**Abstract**: Artificial intelligence (AI) has become increasingly central to precision medicine by enabling the integration and interpretation of multimodal data, yet implementation in clinical settings remains limited. This paper provides a scoping review of literature from 2019-2024 on the implementation of AI in precision medicine, identifying key barriers and enablers across data quality, clinical reliability, workflow integration, and governance. Through an ecosystem-based framework, we highlight the interdependent relationships shaping real-world translation and propose future directions to support trustworthy and sustainable implementation. 

---
# ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning 

**Authors**: Roger Creus Castanyer, Faisal Mohamed, Pablo Samuel Castro, Cyrus Neary, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2510.14176)  

**Abstract**: Reinforcement learning (RL) algorithms are highly sensitive to reward function specification, which remains a central challenge limiting their broad applicability. We present ARM-FM: Automated Reward Machines via Foundation Models, a framework for automated, compositional reward design in RL that leverages the high-level reasoning capabilities of foundation models (FMs). Reward machines (RMs) -- an automata-based formalism for reward specification -- are used as the mechanism for RL objective specification, and are automatically constructed via the use of FMs. The structured formalism of RMs yields effective task decompositions, while the use of FMs enables objective specifications in natural language. Concretely, we (i) use FMs to automatically generate RMs from natural language specifications; (ii) associate language embeddings with each RM automata-state to enable generalization across tasks; and (iii) provide empirical evidence of ARM-FM's effectiveness in a diverse suite of challenging environments, including evidence of zero-shot generalization. 

---
# JEDA: Query-Free Clinical Order Search from Ambient Dialogues 

**Authors**: Praphul Singh, Corey Barrett, Sumana Srivasta, Amitabh Saikia, Irfan Bulu, Sri Gadde, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14169)  

**Abstract**: Clinical conversations mix explicit directives (order a chest X-ray) with implicit reasoning (the cough worsened overnight, we should check for pneumonia). Many systems rely on LLM rewriting, adding latency, instability, and opacity that hinder real-time ordering. We present JEDA (Joint Embedding for Direct and Ambient clinical orders), a domain-initialized bi-encoder that retrieves canonical orders directly and, in a query-free mode, encodes a short rolling window of ambient dialogue to trigger retrieval. Initialized from PubMedBERT and fine-tuned with a duplicate-safe contrastive objective, JEDA aligns heterogeneous expressions of intent to shared order concepts. Training uses constrained LLM guidance to tie each signed order to complementary formulations (command only, context only, command+context, context+reasoning), producing clearer inter-order separation, tighter query extendash order coupling, and stronger generalization. The query-free mode is noise-resilient, reducing sensitivity to disfluencies and ASR errors by conditioning on a short window rather than a single utterance. Deployed in practice, JEDA yields large gains and substantially outperforms its base encoder and recent open embedders (Linq Embed Mistral, SFR Embedding, GTE Qwen, BGE large, Embedding Gemma). The result is a fast, interpretable, LLM-free retrieval layer that links ambient context to actionable clinical orders in real time. 

---
# Combining Reinforcement Learning and Behavior Trees for NPCs in Video Games with AMD Schola 

**Authors**: Tian Liu, Alex Cann, Ian Colbert, Mehdi Saeedi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14154)  

**Abstract**: While the rapid advancements in the reinforcement learning (RL) research community have been remarkable, the adoption in commercial video games remains slow. In this paper, we outline common challenges the Game AI community faces when using RL-driven NPCs in practice, and highlight the intersection of RL with traditional behavior trees (BTs) as a crucial juncture to be explored further. Although the BT+RL intersection has been suggested in several research papers, its adoption is rare. We demonstrate the viability of this approach using AMD Schola -- a plugin for training RL agents in Unreal Engine -- by creating multi-task NPCs in a complex 3D environment inspired by the commercial video game ``The Last of Us". We provide detailed methodologies for jointly training RL models with BTs while showcasing various skills. 

---
# CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization 

**Authors**: Henrique Assumpção, Diego Ferreira, Leandro Campos, Fabricio Murai  

**Link**: [PDF](https://arxiv.org/pdf/2510.14150)  

**Abstract**: In this work, we introduce CodeEvolve, an open-source evolutionary coding agent that unites Large Language Models (LLMs) with genetic algorithms to solve complex computational problems. Our framework adapts powerful evolutionary concepts to the LLM domain, building upon recent methods for generalized scientific discovery. CodeEvolve employs an island-based genetic algorithm to maintain population diversity and increase throughput, introduces a novel inspiration-based crossover mechanism that leverages the LLMs context window to combine features from successful solutions, and implements meta-prompting strategies for dynamic exploration of the solution space. We conduct a rigorous evaluation of CodeEvolve on a subset of the mathematical benchmarks used to evaluate Google DeepMind's closed-source AlphaEvolve. Our findings show that our method surpasses AlphaEvolve's performance on several challenging problems. To foster collaboration and accelerate progress, we release our complete framework as an open-source repository. 

---
# A Multimodal Approach to Heritage Preservation in the Context of Climate Change 

**Authors**: David Roqui, Adèle Cormier, nistor Grozavu, Ann Bourges  

**Link**: [PDF](https://arxiv.org/pdf/2510.14136)  

**Abstract**: Cultural heritage sites face accelerating degradation due to climate change, yet tradi- tional monitoring relies on unimodal analysis (visual inspection or environmental sen- sors alone) that fails to capture the complex interplay between environmental stres- sors and material deterioration. We propose a lightweight multimodal architecture that fuses sensor data (temperature, humidity) with visual imagery to predict degradation severity at heritage sites. Our approach adapts PerceiverIO with two key innovations: (1) simplified encoders (64D latent space) that prevent overfitting on small datasets (n=37 training samples), and (2) Adaptive Barlow Twins loss that encourages modality complementarity rather than redundancy. On data from Strasbourg Cathedral, our model achieves 76.9% accu- racy, a 43% improvement over standard multimodal architectures (VisualBERT, Trans- former) and 25% over vanilla PerceiverIO. Ablation studies reveal that sensor-only achieves 61.5% while image-only reaches 46.2%, confirming successful multimodal synergy. A systematic hyperparameter study identifies an optimal moderate correlation target ({\tau} =0.3) that balances align- ment and complementarity, achieving 69.2% accuracy compared to other {\tau} values ({\tau} =0.1/0.5/0.7: 53.8%, {\tau} =0.9: 61.5%). This work demonstrates that architectural sim- plicity combined with contrastive regularization enables effective multimodal learning in data-scarce heritage monitoring contexts, providing a foundation for AI-driven con- servation decision support systems. 

---
# Formalizing the Safety, Security, and Functional Properties of Agentic AI Systems 

**Authors**: Edoardo Allegrini, Ananth Shreekumar, Z. Berkay Celik  

**Link**: [PDF](https://arxiv.org/pdf/2510.14133)  

**Abstract**: Agentic AI systems, which leverage multiple autonomous agents and Large Language Models (LLMs), are increasingly used to address complex, multi-step tasks. The safety, security, and functionality of these systems are critical, especially in high-stakes applications. However, the current ecosystem of inter-agent communication is fragmented, with protocols such as the Model Context Protocol (MCP) for tool access and the Agent-to-Agent (A2A) protocol for coordination being analyzed in isolation. This fragmentation creates a semantic gap that prevents the rigorous analysis of system properties and introduces risks such as architectural misalignment and exploitable coordination issues. To address these challenges, we introduce a modeling framework for agentic AI systems composed of two foundational models. The first, the host agent model, formalizes the top-level entity that interacts with the user, decomposes tasks, and orchestrates their execution by leveraging external agents and tools. The second, the task lifecycle model, details the states and transitions of individual sub-tasks from creation to completion, providing a fine-grained view of task management and error handling. Together, these models provide a unified semantic framework for reasoning about the behavior of multi-AI agent systems. Grounded in this framework, we define 17 properties for the host agent and 14 for the task lifecycle, categorized into liveness, safety, completeness, and fairness. Expressed in temporal logic, these properties enable formal verification of system behavior, detection of coordination edge cases, and prevention of deadlocks and security vulnerabilities. Through this effort, we introduce the first rigorously grounded, domain-agnostic framework for the systematic analysis, design, and deployment of correct, reliable, and robust agentic AI systems. 

---
# STEMS: Spatial-Temporal Enhanced Safe Multi-Agent Coordination for Building Energy Management 

**Authors**: Huiliang Zhang, Di Wu, Arnaud Zinflou, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2510.14112)  

**Abstract**: Building energy management is essential for achieving carbon reduction goals, improving occupant comfort, and reducing energy costs. Coordinated building energy management faces critical challenges in exploiting spatial-temporal dependencies while ensuring operational safety across multi-building systems. Current multi-building energy systems face three key challenges: insufficient spatial-temporal information exploitation, lack of rigorous safety guarantees, and system complexity. This paper proposes Spatial-Temporal Enhanced Safe Multi-Agent Coordination (STEMS), a novel safety-constrained multi-agent reinforcement learning framework for coordinated building energy management. STEMS integrates two core components: (1) a spatial-temporal graph representation learning framework using a GCN-Transformer fusion architecture to capture inter-building relationships and temporal patterns, and (2) a safety-constrained multi-agent RL algorithm incorporating Control Barrier Functions to provide mathematical safety guarantees. Extensive experiments on real-world building datasets demonstrate STEMS's superior performance over existing methods, showing that STEMS achieves 21% cost reduction, 18% emission reduction, and dramatically reduces safety violations from 35.1% to 5.6% while maintaining optimal comfort with only 0.13 discomfort proportion. The framework also demonstrates strong robustness during extreme weather conditions and maintains effectiveness across different building types. 

---
# Generating Fair Consensus Statements with Social Choice on Token-Level MDPs 

**Authors**: Carter Blair, Kate Larson  

**Link**: [PDF](https://arxiv.org/pdf/2510.14106)  

**Abstract**: Current frameworks for consensus statement generation with large language models lack the inherent structure needed to provide provable fairness guarantees when aggregating diverse free-form opinions. We model the task as a multi-objective, token-level Markov Decision Process (MDP), where each objective corresponds to an agent's preference. Token-level rewards for each agent are derived from their policy (e.g., a personalized language model). This approach utilizes the finding that such policies implicitly define optimal Q-functions, providing a principled way to quantify rewards at each generation step without a value function (Rafailov et al., 2024). This MDP formulation creates a formal structure amenable to analysis using principles from social choice theory. We propose two approaches grounded in social choice theory. First, we propose a stochastic generation policy guaranteed to be in the ex-ante core, extending core stability concepts from voting theory to text generation. This policy is derived from an underlying distribution over complete statements that maximizes proportional fairness (Nash Welfare). Second, for generating a single statement, we target the maximization of egalitarian welfare using search algorithms within the MDP framework. Empirically, experiments using language models to instantiate agent policies show that search guided by the egalitarian objective generates consensus statements with improved worst-case agent alignment compared to baseline methods, including the Habermas Machine (Tessler et al., 2024). 

---
# Position: Require Frontier AI Labs To Release Small "Analog" Models 

**Authors**: Shriyash Upadhyay, Chaithanya Bandi, Narmeen Oozeer, Philip Quirke  

**Link**: [PDF](https://arxiv.org/pdf/2510.14053)  

**Abstract**: Recent proposals for regulating frontier AI models have sparked concerns about the cost of safety regulation, and most such regulations have been shelved due to the safety-innovation tradeoff. This paper argues for an alternative regulatory approach that ensures AI safety while actively promoting innovation: mandating that large AI laboratories release small, openly accessible analog models (scaled-down versions) trained similarly to and distilled from their largest proprietary models.
Analog models serve as public proxies, allowing broad participation in safety verification, interpretability research, and algorithmic transparency without forcing labs to disclose their full-scale models. Recent research demonstrates that safety and interpretability methods developed using these smaller models generalize effectively to frontier-scale systems. By enabling the wider research community to directly investigate and innovate upon accessible analogs, our policy substantially reduces the regulatory burden and accelerates safety advancements.
This mandate promises minimal additional costs, leveraging reusable resources like data and infrastructure, while significantly contributing to the public good. Our hope is not only that this policy be adopted, but that it illustrates a broader principle supporting fundamental research in machine learning: deeper understanding of models relaxes the safety-innovation tradeoff and lets us have more of both. 

---
# GammaZero: Learning To Guide POMDP Belief Space Search With Graph Representations 

**Authors**: Rajesh Mangannavar, Prasad Tadepalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.14035)  

**Abstract**: We introduce an action-centric graph representation framework for learning to guide planning in Partially Observable Markov Decision Processes (POMDPs). Unlike existing approaches that require domain-specific neural architectures and struggle with scalability, GammaZero leverages a unified graph-based belief representation that enables generalization across problem sizes within a domain. Our key insight is that belief states can be systematically transformed into action-centric graphs where structural patterns learned on small problems transfer to larger instances. We employ a graph neural network with a decoder architecture to learn value functions and policies from expert demonstrations on computationally tractable problems, then apply these learned heuristics to guide Monte Carlo tree search on larger problems. Experimental results on standard POMDP benchmarks demonstrate that GammaZero achieves comparable performance to BetaZero when trained and tested on the same-sized problems, while uniquely enabling zero-shot generalization to problems 2-4 times larger than those seen during training, maintaining solution quality with reduced search requirements. 

---
# Do Large Language Models Show Biases in Causal Learning? Insights from Contingency Judgment 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Francisca Gauna Selasco, Giovanni Franco Gabriel Marraffini, Mario Alejandro Leiva, Gerardo I. Simari, María Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13985)  

**Abstract**: Causal learning is the cognitive process of developing the capability of making causal inferences based on available information, often guided by normative principles. This process is prone to errors and biases, such as the illusion of causality, in which people perceive a causal relationship between two variables despite lacking supporting evidence. This cognitive bias has been proposed to underlie many societal problems, including social prejudice, stereotype formation, misinformation, and superstitious thinking. In this work, we examine whether large language models are prone to developing causal illusions when faced with a classic cognitive science paradigm: the contingency judgment task. To investigate this, we constructed a dataset of 1,000 null contingency scenarios (in which the available information is not sufficient to establish a causal relationship between variables) within medical contexts and prompted LLMs to evaluate the effectiveness of potential causes. Our findings show that all evaluated models systematically inferred unwarranted causal relationships, revealing a strong susceptibility to the illusion of causality. While there is ongoing debate about whether LLMs genuinely understand causality or merely reproduce causal language without true comprehension, our findings support the latter hypothesis and raise concerns about the use of language models in domains where accurate causal reasoning is essential for informed decision-making. 

---
# Do Slides Help? Multi-modal Context for Automatic Transcription of Conference Talks 

**Authors**: Supriti Sinhamahapatra, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2510.13979)  

**Abstract**: State-of-the-art (SOTA) Automatic Speech Recognition (ASR) systems primarily rely on acoustic information while disregarding additional multi-modal context. However, visual information are essential in disambiguation and adaptation. While most work focus on speaker images to handle noise conditions, this work also focuses on integrating presentation slides for the use cases of scientific presentation.
In a first step, we create a benchmark for multi-modal presentation including an automatic analysis of transcribing domain-specific terminology. Next, we explore methods for augmenting speech models with multi-modal information. We mitigate the lack of datasets with accompanying slides by a suitable approach of data augmentation. Finally, we train a model using the augmented dataset, resulting in a relative reduction in word error rate of approximately 34%, across all words and 35%, for domain-specific terms compared to the baseline model. 

---
# Decision Oriented Technique (DOTechnique): Finding Model Validity Through Decision-Maker Context 

**Authors**: Raheleh Biglari, Joachim Denil  

**Link**: [PDF](https://arxiv.org/pdf/2510.13858)  

**Abstract**: Model validity is as critical as the model itself, especially when guiding decision-making processes. Traditional approaches often rely on predefined validity frames, which may not always be available or sufficient. This paper introduces the Decision Oriented Technique (DOTechnique), a novel method for determining model validity based on decision consistency rather than output similarity. By evaluating whether surrogate models lead to equivalent decisions compared to high-fidelity models, DOTechnique enables efficient identification of validity regions, even in the absence of explicit validity boundaries. The approach integrates domain constraints and symbolic reasoning to narrow the search space, enhancing computational efficiency. A highway lane change system serves as a motivating example, demonstrating how DOTechnique can uncover the validity region of a simulation model. The results highlight the potential of the technique to support finding model validity through decision-maker context. 

---
# Coupled Diffusion Sampling for Training-Free Multi-View Image Editing 

**Authors**: Hadi Alzayer, Yunzhi Zhang, Chen Geng, Jia-Bin Huang, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14981)  

**Abstract**: We present an inference-time diffusion sampling method to perform multi-view consistent image editing using pre-trained 2D image editing models. These models can independently produce high-quality edits for each image in a set of multi-view images of a 3D scene or object, but they do not maintain consistency across views. Existing approaches typically address this by optimizing over explicit 3D representations, but they suffer from a lengthy optimization process and instability under sparse view settings. We propose an implicit 3D regularization approach by constraining the generated 2D image sequences to adhere to a pre-trained multi-view image distribution. This is achieved through coupled diffusion sampling, a simple diffusion sampling technique that concurrently samples two trajectories from both a multi-view image distribution and a 2D edited image distribution, using a coupling term to enforce the multi-view consistency among the generated images. We validate the effectiveness and generality of this framework on three distinct multi-view image editing tasks, demonstrating its applicability across various model architectures and highlighting its potential as a general solution for multi-view consistent editing. 

---
# From Pixels to Words -- Towards Native Vision-Language Primitives at Scale 

**Authors**: Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14979)  

**Abstract**: The edifice of native Vision-Language Models (VLMs) has emerged as a rising contender to typical modular VLMs, shaped by evolving model architectures and training paradigms. Yet, two lingering clouds cast shadows over its widespread exploration and promotion: (-) What fundamental constraints set native VLMs apart from modular ones, and to what extent can these barriers be overcome? (-) How to make research in native VLMs more accessible and democratized, thereby accelerating progress in the field. In this paper, we clarify these challenges and outline guiding principles for constructing native VLMs. Specifically, one native VLM primitive should: (i) effectively align pixel and word representations within a shared semantic space; (ii) seamlessly integrate the strengths of formerly separate vision and language modules; (iii) inherently embody various cross-modal properties that support unified vision-language encoding, aligning, and reasoning. Hence, we launch NEO, a novel family of native VLMs built from first principles, capable of rivaling top-tier modular counterparts across diverse real-world scenarios. With only 390M image-text examples, NEO efficiently develops visual perception from scratch while mitigating vision-language conflicts inside a dense and monolithic model crafted from our elaborate primitives. We position NEO as a cornerstone for scalable and powerful native VLMs, paired with a rich set of reusable components that foster a cost-effective and extensible ecosystem. Our code and models are publicly available at: this https URL. 

---
# Terra: Explorable Native 3D World Model with Point Latents 

**Authors**: Yuanhui Huang, Weiliang Chen, Wenzhao Zheng, Xin Tao, Pengfei Wan, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14977)  

**Abstract**: World models have garnered increasing attention for comprehensive modeling of the real world. However, most existing methods still rely on pixel-aligned representations as the basis for world evolution, neglecting the inherent 3D nature of the physical world. This could undermine the 3D consistency and diminish the modeling efficiency of world models. In this paper, we present Terra, a native 3D world model that represents and generates explorable environments in an intrinsic 3D latent space. Specifically, we propose a novel point-to-Gaussian variational autoencoder (P2G-VAE) that encodes 3D inputs into a latent point representation, which is subsequently decoded as 3D Gaussian primitives to jointly model geometry and appearance. We then introduce a sparse point flow matching network (SPFlow) for generating the latent point representation, which simultaneously denoises the positions and features of the point latents. Our Terra enables exact multi-view consistency with native 3D representation and architecture, and supports flexible rendering from any viewpoint with only a single generation process. Furthermore, Terra achieves explorable world modeling through progressive generation in the point latent space. We conduct extensive experiments on the challenging indoor scenes from ScanNet v2. Terra achieves state-of-the-art performance in both reconstruction and generation with high 3D consistency. 

---
# WithAnyone: Towards Controllable and ID Consistent Image Generation 

**Authors**: Hengyuan Xu, Wei Cheng, Peng Xing, Yixiao Fang, Shuhan Wu, Rui Wang, Xianfang Zeng, Daxin Jiang, Gang Yu, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14975)  

**Abstract**: Identity-consistent generation has become an important focus in text-to-image research, with recent models achieving notable success in producing images aligned with a reference identity. Yet, the scarcity of large-scale paired datasets containing multiple images of the same individual forces most approaches to adopt reconstruction-based training. This reliance often leads to a failure mode we term copy-paste, where the model directly replicates the reference face rather than preserving identity across natural variations in pose, expression, or lighting. Such over-similarity undermines controllability and limits the expressive power of generation. To address these limitations, we (1) construct a large-scale paired dataset MultiID-2M, tailored for multi-person scenarios, providing diverse references for each identity; (2) introduce a benchmark that quantifies both copy-paste artifacts and the trade-off between identity fidelity and variation; and (3) propose a novel training paradigm with a contrastive identity loss that leverages paired data to balance fidelity with diversity. These contributions culminate in WithAnyone, a diffusion-based model that effectively mitigates copy-paste while preserving high identity similarity. Extensive qualitative and quantitative experiments demonstrate that WithAnyone significantly reduces copy-paste artifacts, improves controllability over pose and expression, and maintains strong perceptual quality. User studies further validate that our method achieves high identity fidelity while enabling expressive controllable generation. 

---
# pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation 

**Authors**: Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein, Sai Bi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14974)  

**Abstract**: Few-step diffusion or flow-based generative models typically distill a velocity-predicting teacher into a student that predicts a shortcut towards denoised data. This format mismatch has led to complex distillation procedures that often suffer from a quality-diversity trade-off. To address this, we propose policy-based flow models ($\pi$-Flow). $\pi$-Flow modifies the output layer of a student flow model to predict a network-free policy at one timestep. The policy then produces dynamic flow velocities at future substeps with negligible overhead, enabling fast and accurate ODE integration on these substeps without extra network evaluations. To match the policy's ODE trajectory to the teacher's, we introduce a novel imitation distillation approach, which matches the policy's velocity to the teacher's along the policy's trajectory using a standard $\ell_2$ flow matching loss. By simply mimicking the teacher's behavior, $\pi$-Flow enables stable and scalable training and avoids the quality-diversity trade-off. On ImageNet 256$^2$, it attains a 1-NFE FID of 2.85, outperforming MeanFlow of the same DiT architecture. On FLUX.1-12B and Qwen-Image-20B at 4 NFEs, $\pi$-Flow achieves substantially better diversity than state-of-the-art few-step methods, while maintaining teacher-level quality. 

---
# Attention Is All You Need for KV Cache in Diffusion LLMs 

**Authors**: Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14973)  

**Abstract**: This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), $45.1\times$ on longer sequences, and $4.8\times$ on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs. 

---
# TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar 

**Authors**: Yinxi Li, Yuntian Deng, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.14972)  

**Abstract**: Large language models (LLMs) for code rely on subword tokenizers, such as byte-pair encoding (BPE), learned from mixed natural language text and programming language code but driven by statistics rather than grammar. As a result, semantically identical code snippets can be tokenized differently depending on superficial factors such as whitespace or identifier naming. To measure the impact of this misalignment, we introduce TokDrift, a framework that applies semantic-preserving rewrite rules to create code variants differing only in tokenization. Across nine code LLMs, including large ones with over 30B parameters, even minor formatting changes can cause substantial shifts in model behavior. Layer-wise analysis shows that the issue originates in early embeddings, where subword segmentation fails to capture grammar token boundaries. Our findings identify misaligned tokenization as a hidden obstacle to reliable code understanding and generation, highlighting the need for grammar-aware tokenization for future code LLMs. 

---
# LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training 

**Authors**: Yiming Wang, Da Yin, Yuedong Cui, Ruichen Zheng, Zhiqian Li, Zongyu Lin, Di Wu, Xueqing Wu, Chenchen Ye, Yu Zhou, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14969)  

**Abstract**: Digital agents require diverse, large-scale UI trajectories to generalize across real-world tasks, yet collecting such data is prohibitively expensive in both human annotation, infra and engineering perspectives. To this end, we introduce $\textbf{UI-Simulator}$, a scalable paradigm that generates structured UI states and transitions to synthesize training trajectories at scale. Our paradigm integrates a digital world simulator for diverse UI states, a guided rollout process for coherent exploration, and a trajectory wrapper that produces high-quality and diverse trajectories for agent training. We further propose $\textbf{UI-Simulator-Grow}$, a targeted scaling strategy that enables more rapid and data-efficient scaling by prioritizing high-impact tasks and synthesizes informative trajectory variants. Experiments on WebArena and AndroidWorld show that UI-Simulator rivals or surpasses open-source agents trained on real UIs with significantly better robustness, despite using weaker teacher models. Moreover, UI-Simulator-Grow matches the performance of Llama-3-70B-Instruct using only Llama-3-8B-Instruct as the base model, highlighting the potential of targeted synthesis scaling paradigm to continuously and efficiently enhance the digital agents. 

---
# RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks 

**Authors**: Mingxuan Yan, Yuping Wang, Zechun Liu, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14968)  

**Abstract**: To tackle long-horizon tasks, recent hierarchical vision-language-action (VLAs) frameworks employ vision-language model (VLM)-based planners to decompose complex manipulation tasks into simpler sub-tasks that low-level visuomotor policies can easily handle. Typically, the VLM planner is finetuned to learn to decompose a target task. This finetuning requires target task demonstrations segmented into sub-tasks by either human annotation or heuristic rules. However, the heuristic subtasks can deviate significantly from the training data of the visuomotor policy, which degrades task performance. To address these issues, we propose a Retrieval-based Demonstration Decomposer (RDD) that automatically decomposes demonstrations into sub-tasks by aligning the visual features of the decomposed sub-task intervals with those from the training data of the low-level visuomotor policies. Our method outperforms the state-of-the-art sub-task decomposer on both simulation and real-world tasks, demonstrating robustness across diverse settings. Code and more results are available at this http URL. 

---
# Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents 

**Authors**: Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, Zhenzhe Ying  

**Link**: [PDF](https://arxiv.org/pdf/2510.14967)  

**Abstract**: Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency. 

---
# C4D: 4D Made from 3D through Dual Correspondences 

**Authors**: Shizun Wang, Zhenxiang Jiang, Xingyi Yang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14960)  

**Abstract**: Recovering 4D from monocular video, which jointly estimates dynamic geometry and camera poses, is an inevitably challenging problem. While recent pointmap-based 3D reconstruction methods (e.g., DUSt3R) have made great progress in reconstructing static scenes, directly applying them to dynamic scenes leads to inaccurate results. This discrepancy arises because moving objects violate multi-view geometric constraints, disrupting the reconstruction. To address this, we introduce C4D, a framework that leverages temporal Correspondences to extend existing 3D reconstruction formulation to 4D. Specifically, apart from predicting pointmaps, C4D captures two types of correspondences: short-term optical flow and long-term point tracking. We train a dynamic-aware point tracker that provides additional mobility information, facilitating the estimation of motion masks to separate moving elements from the static background, thus offering more reliable guidance for dynamic scenes. Furthermore, we introduce a set of dynamic scene optimization objectives to recover per-frame 3D geometry and camera parameters. Simultaneously, the correspondences lift 2D trajectories into smooth 3D trajectories, enabling fully integrated 4D reconstruction. Experiments show that our framework achieves complete 4D recovery and demonstrates strong performance across multiple downstream tasks, including depth estimation, camera pose estimation, and point tracking. Project Page: this https URL 

---
# CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions 

**Authors**: Lizhi Yang, Blake Werner, Massimiliano de Sa Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2510.14959)  

**Abstract**: Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed \emph{online} via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs \emph{in training}. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter. 

---
# RealDPO: Real or Not Real, that is the Preference 

**Authors**: Guo Cheng, Danni Yang, Ziqi Huang, Jianlou Si, Chenyang Si, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14955)  

**Abstract**: Video generative models have recently achieved notable advancements in synthesis quality. However, generating complex motions remains a critical challenge, as existing models often struggle to produce natural, smooth, and contextually consistent movements. This gap between generated and real-world motions limits their practical applicability. To address this issue, we introduce RealDPO, a novel alignment paradigm that leverages real-world data as positive samples for preference learning, enabling more accurate motion synthesis. Unlike traditional supervised fine-tuning (SFT), which offers limited corrective feedback, RealDPO employs Direct Preference Optimization (DPO) with a tailored loss function to enhance motion realism. By contrasting real-world videos with erroneous model outputs, RealDPO enables iterative self-correction, progressively refining motion quality. To support post-training in complex motion synthesis, we propose RealAction-5K, a curated dataset of high-quality videos capturing human daily activities with rich and precise motion details. Extensive experiments demonstrate that RealDPO significantly improves video quality, text alignment, and motion realism compared to state-of-the-art models and existing preference optimization techniques. 

---
# Architecture Is All You Need: Diversity-Enabled Sweet Spots for Robust Humanoid Locomotion 

**Authors**: Blake Werner, Lizhi Yang, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2510.14947)  

**Abstract**: Robust humanoid locomotion in unstructured environments requires architectures that balance fast low-level stabilization with slower perceptual decision-making. We show that a simple layered control architecture (LCA), a proprioceptive stabilizer running at high rate, coupled with a compact low-rate perceptual policy, enables substantially more robust performance than monolithic end-to-end designs, even when using minimal perception encoders. Through a two-stage training curriculum (blind stabilizer pretraining followed by perceptual fine-tuning), we demonstrate that layered policies consistently outperform one-stage alternatives in both simulation and hardware. On a Unitree G1 humanoid, our approach succeeds across stair and ledge tasks where one-stage perceptual policies fail. These results highlight that architectural separation of timescales, rather than network scale or complexity, is the key enabler for robust perception-conditioned locomotion. 

---
# MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics 

**Authors**: Yuxing Lu, Xukai Zhao, J. Ben Tamo, Micky C. Nnamdi, Rui Peng, Shuang Zeng, Xingyu Hu, Jinzhuo Wang, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14944)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities on general text; however, their proficiency in specialized scientific domains that require deep, interconnected knowledge remains largely uncharacterized. Metabolomics presents unique challenges with its complex biochemical pathways, heterogeneous identifier systems, and fragmented databases. To systematically evaluate LLM capabilities in this domain, we introduce MetaBench, the first benchmark for metabolomics assessment. Curated from authoritative public resources, MetaBench evaluates five capabilities essential for metabolomics research: knowledge, understanding, grounding, reasoning, and research. Our evaluation of 25 open- and closed-source LLMs reveals distinct performance patterns across metabolomics tasks: while models perform well on text generation tasks, cross-database identifier grounding remains challenging even with retrieval augmentation. Model performance also decreases on long-tail metabolites with sparse annotations. With MetaBench, we provide essential infrastructure for developing and evaluating metabolomics AI systems, enabling systematic progress toward reliable computational tools for metabolomics research. 

---
# LaSeR: Reinforcement Learning with Last-Token Self-Rewarding 

**Authors**: Wenkai Yang, Weijie Liu, Ruobing Xie, Yiju Guo, Lulu Wu, Saiyong Yang, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14943)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a core paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). To address the lack of verification signals at test time, prior studies incorporate the training of model's self-verification capability into the standard RLVR process, thereby unifying reasoning and verification capabilities within a single LLM. However, previous practice requires the LLM to sequentially generate solutions and self-verifications using two separate prompt templates, which significantly reduces efficiency. In this work, we theoretically reveal that the closed-form solution to the RL objective of self-verification can be reduced to a remarkably simple form: the true reasoning reward of a solution is equal to its last-token self-rewarding score, which is computed as the difference between the policy model's next-token log-probability assigned to any pre-specified token at the solution's last token and a pre-calculated constant, scaled by the KL coefficient. Based on this insight, we propose LaSeR (Reinforcement Learning with Last-Token Self-Rewarding), an algorithm that simply augments the original RLVR loss with a MSE loss that aligns the last-token self-rewarding scores with verifier-based reasoning rewards, jointly optimizing the reasoning and self-rewarding capabilities of LLMs. The optimized self-rewarding scores can be utilized in both training and testing to enhance model performance. Notably, our algorithm derives these scores from the predicted next-token probability distribution of the last token immediately after generation, incurring only the minimal extra cost of one additional token inference. Experiments show that our method not only improves the model's reasoning performance but also equips it with remarkable self-rewarding capability, thereby boosting its inference-time scaling performance. 

---
# Circuit Insights: Towards Interpretability Beyond Activations 

**Authors**: Elena Golimblevskaia, Aakriti Jain, Bruno Puri, Ammar Ibrahim, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14936)  

**Abstract**: The fields of explainable AI and mechanistic interpretability aim to uncover the internal structure of neural networks, with circuit discovery as a central tool for understanding model computations. Existing approaches, however, rely on manual inspection and remain limited to toy tasks. Automated interpretability offers scalability by analyzing isolated features and their activations, but it often misses interactions between features and depends strongly on external LLMs and dataset quality. Transcoders have recently made it possible to separate feature attributions into input-dependent and input-invariant components, providing a foundation for more systematic circuit analysis. Building on this, we propose WeightLens and CircuitLens, two complementary methods that go beyond activation-based analysis. WeightLens interprets features directly from their learned weights, removing the need for explainer models or datasets while matching or exceeding the performance of existing methods on context-independent features. CircuitLens captures how feature activations arise from interactions between components, revealing circuit-level dynamics that activation-only approaches cannot identify. Together, these methods increase interpretability robustness and enhance scalable mechanistic analysis of circuits while maintaining efficiency and quality. 

---
# Predicting Task Performance with Context-aware Scaling Laws 

**Authors**: Kyle Montgomery, David Park, Jianhong Tu, Michael Bendersky, Beliz Gunel, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14919)  

**Abstract**: Scaling laws have transformed our understanding of large language models by linking upstream metrics like cross-entropy loss to design factors such as model size, training data, and compute. However, these conventional laws fail to capture downstream task performance, where context plays a critical role. In this work, we propose a straightforward, interpretable framework that jointly models downstream performance as a function of the training compute and the provided context. We empirically validate our framework by fitting it on the observed downstream performance of extended-context variants of Llama-2-7B and Llama-2-13B across 65,500 unique instances spanning three tasks: arithmetic reasoning, common sense reasoning, and machine translation. Our results demonstrate that our framework accurately models in-distribution downstream performance, generalizes across three orders of magnitude in training compute, and reliably extrapolates performance as the amount of context increases. These findings offer valuable insights into the interplay between training compute and context utilization, providing guidance for designing more efficient long-context LLMs for diverse downstream tasks. Our code is available at this https URL. 

---
# MaskCaptioner : Learning to Jointly Segment and Caption Object Trajectories in Videos 

**Authors**: Gabriel Fiastre, Antoine Yang, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2510.14904)  

**Abstract**: Dense Video Object Captioning (DVOC) is the task of jointly detecting, tracking, and captioning object trajectories in a video, requiring the ability to understand spatio-temporal details and describe them in natural language. Due to the complexity of the task and the high cost associated with manual annotation, previous approaches resort to disjoint training strategies, potentially leading to suboptimal performance. To circumvent this issue, we propose to generate captions about spatio-temporally localized entities leveraging a state-of-the-art VLM. By extending the LVIS and LV-VIS datasets with our synthetic captions (LVISCap and LV-VISCap), we train MaskCaptioner, an end-to-end model capable of jointly detecting, segmenting, tracking and captioning object trajectories. Moreover, with pretraining on LVISCap and LV-VISCap, MaskCaptioner achieves state-of-the-art DVOC results on three existing benchmarks, VidSTG, VLN and BenSMOT. The datasets and code are available at this https URL. 

---
# Reasoning with Sampling: Your Base Model is Smarter Than You Think 

**Authors**: Aayush Karan, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.14901)  

**Abstract**: Frontier reasoning models have exhibited incredible capabilities across a wide array of disciplines, driven by posttraining large language models (LLMs) with reinforcement learning (RL). However, despite the widespread success of this paradigm, much of the literature has been devoted to disentangling truly novel behaviors that emerge during RL but are not present in the base models. In our work, we approach this question from a different angle, instead asking whether comparable reasoning capabilites can be elicited from base models at inference time by pure sampling, without any additional training. Inspired by Markov chain Monte Carlo (MCMC) techniques for sampling from sharpened distributions, we propose a simple iterative sampling algorithm leveraging the base models' own likelihoods. Over different base models, we show that our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA. Moreover, our sampler avoids the collapse in diversity over multiple samples that is characteristic of RL-posttraining. Crucially, our method does not require training, curated datasets, or a verifier, suggesting broad applicability beyond easily verifiable domains. 

---
# Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media 

**Authors**: Soorya Ram Shimgekar, Ruining Zhao, Agam Goyal, Violeta J. Rodriguez, Paul A. Bloom, Hari Sundaram, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2510.14889)  

**Abstract**: On social media, many individuals experiencing suicidal ideation (SI) do not disclose their distress explicitly. Instead, signs may surface indirectly through everyday posts or peer interactions. Detecting such implicit signals early is critical but remains challenging. We frame early and implicit SI as a forward-looking prediction task and develop a computational framework that models a user's information environment, consisting of both their longitudinal posting histories as well as the discourse of their socially proximal peers. We adopted a composite network centrality measure to identify top neighbors of a user, and temporally aligned the user's and neighbors' interactions -- integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves early and implicit SI detection by 15% over individual-only baselines. These findings highlight that peer interactions offer valuable predictive signals and carry broader implications for designing early detection systems that capture indirect as well as masked expressions of risk in online environments. 

---
# Learning When Not to Learn: Risk-Sensitive Abstention in Bandits with Unbounded Rewards 

**Authors**: Sarah Liaw, Benjamin Plaut  

**Link**: [PDF](https://arxiv.org/pdf/2510.14884)  

**Abstract**: In high-stakes AI applications, even a single action can cause irreparable damage. However, nearly all of sequential decision-making theory assumes that all errors are recoverable (e.g., by bounding rewards). Standard bandit algorithms that explore aggressively may cause irreparable damage when this assumption fails. Some prior work avoids irreparable errors by asking for help from a mentor, but a mentor may not always be available. In this work, we formalize a model of learning with unbounded rewards without a mentor as a two-action contextual bandit with an abstain option: at each round the agent observes an input and chooses either to abstain (always 0 reward) or to commit (execute a preexisting task policy). Committing yields rewards that are upper-bounded but can be arbitrarily negative, and the commit reward is assumed Lipschitz in the input. We propose a caution-based algorithm that learns when not to learn: it chooses a trusted region and commits only where the available evidence does not already certify harm. Under these conditions and i.i.d. inputs, we establish sublinear regret guarantees, theoretically demonstrating the effectiveness of cautious exploration for deploying learning agents safely in high-stakes environments. 

---
# Predicting kernel regression learning curves from only raw data statistics 

**Authors**: Dhruva Karkada, Joseph Turnbull, Yuxi Liu, James B. Simon  

**Link**: [PDF](https://arxiv.org/pdf/2510.14878)  

**Abstract**: We study kernel regression with common rotation-invariant kernels on real datasets including CIFAR-5m, SVHN, and ImageNet. We give a theoretical framework that predicts learning curves (test risk vs. sample size) from only two measurements: the empirical data covariance matrix and an empirical polynomial decomposition of the target function $f_*$. The key new idea is an analytical approximation of a kernel's eigenvalues and eigenfunctions with respect to an anisotropic data distribution. The eigenfunctions resemble Hermite polynomials of the data, so we call this approximation the Hermite eigenstructure ansatz (HEA). We prove the HEA for Gaussian data, but we find that real image data is often "Gaussian enough" for the HEA to hold well in practice, enabling us to predict learning curves by applying prior results relating kernel eigenstructure to test risk. Extending beyond kernel regression, we empirically find that MLPs in the feature-learning regime learn Hermite polynomials in the order predicted by the HEA. Our HEA framework is a proof of concept that an end-to-end theory of learning which maps dataset structure all the way to model performance is possible for nontrivial learning algorithms on real datasets. 

---
# Benchmarking Multimodal Large Language Models for Face Recognition 

**Authors**: Hatef Otroshi Shahreza, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2510.14866)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable performance across diverse vision-and-language tasks. However, their potential in face recognition remains underexplored. In particular, the performance of open-source MLLMs needs to be evaluated and compared with existing face recognition models on standard benchmarks with similar protocol. In this work, we present a systematic benchmark of state-of-the-art MLLMs for face recognition on several face recognition datasets, including LFW, CALFW, CPLFW, CFP, AgeDB and RFW. Experimental results reveal that while MLLMs capture rich semantic cues useful for face-related tasks, they lag behind specialized models in high-precision recognition scenarios in zero-shot applications. This benchmark provides a foundation for advancing MLLM-based face recognition, offering insights for the design of next-generation models with higher accuracy and generalization. The source code of our benchmark is publicly available in the project page. 

---
# RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning 

**Authors**: Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14830)  

**Abstract**: Real-world robotic manipulation in homes and factories demands reliability, efficiency, and robustness that approach or surpass skilled human operators. We present RL-100, a real-world reinforcement learning training framework built on diffusion visuomotor policies trained bu supervised learning. RL-100 introduces a three-stage pipeline. First, imitation learning leverages human priors. Second, iterative offline reinforcement learning uses an Offline Policy Evaluation procedure, abbreviated OPE, to gate PPO-style updates that are applied in the denoising process for conservative and reliable improvement. Third, online reinforcement learning eliminates residual failure modes. An additional lightweight consistency distillation head compresses the multi-step sampling process in diffusion into a single-step policy, enabling high-frequency control with an order-of-magnitude reduction in latency while preserving task performance. The framework is task-, embodiment-, and representation-agnostic and supports both 3D point clouds and 2D RGB inputs, a variety of robot platforms, and both single-step and action-chunk policies. We evaluate RL-100 on seven real-robot tasks spanning dynamic rigid-body control, such as Push-T and Agile Bowling, fluids and granular pouring, deformable cloth folding, precise dexterous unscrewing, and multi-stage orange juicing. RL-100 attains 100\% success across evaluated trials for a total of 900 out of 900 episodes, including up to 250 out of 250 consecutive trials on one task. The method achieves near-human teleoperation or better time efficiency and demonstrates multi-hour robustness with uninterrupted operation lasting up to two hours. 

---
# Scaling Artificial Intelligence for Multi-Tumor Early Detection with More Reports, Fewer Masks 

**Authors**: Pedro R. A. S. Bassi, Xinze Zhou, Wenxuan Li, Szymon Płotka, Jieneng Chen, Qi Chen, Zheren Zhu, Jakub Prządo, Ibrahim E. Hamacı, Sezgin Er, Yuhan Wang, Ashwin Kumar, Bjoern Menze, Jarosław B. Ćwikła, Yuyin Zhou, Akshay S. Chaudhari, Curtis P. Langlotz, Sergio Decherchi, Andrea Cavalli, Kang Wang, Yang Yang, Alan L. Yuille, Zongwei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14803)  

**Abstract**: Early tumor detection save lives. Each year, more than 300 million computed tomography (CT) scans are performed worldwide, offering a vast opportunity for effective cancer screening. However, detecting small or early-stage tumors on these CT scans remains challenging, even for experts. Artificial intelligence (AI) models can assist by highlighting suspicious regions, but training such models typically requires extensive tumor masks--detailed, voxel-wise outlines of tumors manually drawn by radiologists. Drawing these masks is costly, requiring years of effort and millions of dollars. In contrast, nearly every CT scan in clinical practice is already accompanied by medical reports describing the tumor's size, number, appearance, and sometimes, pathology results--information that is rich, abundant, and often underutilized for AI training. We introduce R-Super, which trains AI to segment tumors that match their descriptions in medical reports. This approach scales AI training with large collections of readily available medical reports, substantially reducing the need for manually drawn tumor masks. When trained on 101,654 reports, AI models achieved performance comparable to those trained on 723 masks. Combining reports and masks further improved sensitivity by +13% and specificity by +8%, surpassing radiologists in detecting five of the seven tumor types. Notably, R-Super enabled segmentation of tumors in the spleen, gallbladder, prostate, bladder, uterus, and esophagus, for which no public masks or AI models previously existed. This study challenges the long-held belief that large-scale, labor-intensive tumor mask creation is indispensable, establishing a scalable and accessible path toward early detection across diverse tumor types.
We plan to release our trained models, code, and dataset at this https URL 

---
# Morphology-Aware Prognostic model for Five-Year Survival Prediction in Colorectal Cancer from H&E Whole Slide Images 

**Authors**: Usama Sajjad, Abdul Rehman Akbar, Ziyu Su, Deborah Knight, Wendy L. Frankel, Metin N. Gurcan, Wei Chen, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14800)  

**Abstract**: Colorectal cancer (CRC) remains the third most prevalent malignancy globally, with approximately 154,000 new cases and 54,000 projected deaths anticipated for 2025. The recent advancement of foundation models in computational pathology has been largely propelled by task agnostic methodologies that can overlook organ-specific crucial morphological patterns that represent distinct biological processes that can fundamentally influence tumor behavior, therapeutic response, and patient outcomes. The aim of this study is to develop a novel, interpretable AI model, PRISM (Prognostic Representation of Integrated Spatial Morphology), that incorporates a continuous variability spectrum within each distinct morphology to characterize phenotypic diversity and reflecting the principle that malignant transformation occurs through incremental evolutionary processes rather than abrupt phenotypic shifts. PRISM is trained on 8.74 million histological images extracted from surgical resection specimens of 424 patients with stage III CRC. PRISM achieved superior prognostic performance for five-year OS (AUC = 0.70 +- 0.04; accuracy = 68.37% +- 4.75%; HR = 3.34, 95% CI = 2.28-4.90; p < 0.0001), outperforming existing CRC-specific methods by 15% and AI foundation models by ~23% accuracy. It showed sex-agnostic robustness (AUC delta = 0.02; accuracy delta = 0.15%) and stable performance across clinicopathological subgroups, with minimal accuracy fluctuation (delta = 1.44%) between 5FU/LV and CPT-11/5FU/LV regimens, replicating the Alliance cohort finding of no survival difference between treatments. 

---
# Cross-Scenario Unified Modeling of User Interests at Billion Scale 

**Authors**: Manjie Xu, Cheng Chen, Xin Jia, Jingyi Zhou, Yongji Wu, Zejian Wang, Chi Zhang, Kai Zuo, Yibo Chen, Xu Tang, Yao Hu, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14788)  

**Abstract**: User interests on content platforms are inherently diverse, manifesting through complex behavioral patterns across heterogeneous scenarios such as search, feed browsing, and content discovery. Traditional recommendation systems typically prioritize business metric optimization within isolated specific scenarios, neglecting cross-scenario behavioral signals and struggling to integrate advanced techniques like LLMs at billion-scale deployments, which finally limits their ability to capture holistic user interests across platform touchpoints. We propose RED-Rec, an LLM-enhanced hierarchical Recommender Engine for Diversified scenarios, tailored for industry-level content recommendation systems. RED-Rec unifies user interest representations across multiple behavioral contexts by aggregating and synthesizing actions from varied scenarios, resulting in comprehensive item and user modeling. At its core, a two-tower LLM-powered framework enables nuanced, multifaceted representations with deployment efficiency, and a scenario-aware dense mixing and querying policy effectively fuses diverse behavioral signals to capture cross-scenario user intent patterns and express fine-grained, context-specific intents during serving. We validate RED-Rec through online A/B testing on hundreds of millions of users in RedNote through online A/B testing, showing substantial performance gains in both content recommendation and advertisement targeting tasks. We further introduce a million-scale sequential recommendation dataset, RED-MMU, for comprehensive offline training and evaluation. Our work advances unified user modeling, unlocking deeper personalization and fostering more meaningful user engagement in large-scale UGC platforms. 

---
# Finding Answers in Thought Matters: Revisiting Evaluation on Large Language Models with Reasoning 

**Authors**: Hwiyeol Jo, Joosung Lee, Jaehone Lee, Sang-Woo Lee, Joonsuk Park, Kang Min Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14773)  

**Abstract**: Evaluating generative models, such as large language models (LLMs), commonly involves question-answering tasks where the final answer is selected based on probability of answer choices. On the other hand, for models requiring reasoning, the method of answer extraction plays a critical role. Our research reveals that the performance of reasoning models and their final answer distributions are highly sensitive to the answer extraction algorithm employed. In order to mitigate this, we propose a basic framework: Answer Regeneration. The method uses an additional model inference, providing the prior input and output prefaced by the prompt "Answer:". The final answer is then selected or extracted from the regenerated output. We show that this extraction-rule-agnostic approach exhibits improved performance and enhanced robustness. Furthermore, we have applied this framework to general math problems and open-ended question answering tasks. Our analysis and this framework could offer a more reliable results for model evaluation. 

---
# Inpainting the Red Planet: Diffusion Models for the Reconstruction of Martian Environments in Virtual Reality 

**Authors**: Giuseppe Lorenzo Catalano, Agata Marta Soccini  

**Link**: [PDF](https://arxiv.org/pdf/2510.14765)  

**Abstract**: Space exploration increasingly relies on Virtual Reality for several tasks, such as mission planning, multidisciplinary scientific analysis, and astronaut training. A key factor for the reliability of the simulations is having accurate 3D representations of planetary terrains. Extraterrestrial heightmaps derived from satellite imagery often contain missing values due to acquisition and transmission constraints. Mars is among the most studied planets beyond Earth, and its extensive terrain datasets make the Martian surface reconstruction a valuable task, although many areas remain unmapped. Deep learning algorithms can support void-filling tasks; however, whereas Earth's comprehensive datasets enables the use of conditional methods, such approaches cannot be applied to Mars. Current approaches rely on simpler interpolation techniques which, however, often fail to preserve geometric coherence. In this work, we propose a method for reconstructing the surface of Mars based on an unconditional diffusion model. Training was conducted on an augmented dataset of 12000 Martian heightmaps derived from NASA's HiRISE survey. A non-homogeneous rescaling strategy captures terrain features across multiple scales before resizing to a fixed 128x128 model resolution. We compared our method against established void-filling and inpainting techniques, including Inverse Distance Weighting, kriging, and Navier-Stokes algorithm, on an evaluation set of 1000 samples. Results show that our approach consistently outperforms these methods in terms of reconstruction accuracy (4-15% on RMSE) and perceptual similarity (29-81% on LPIPS) with the original data. 

---
# COIG-Writer: A High-Quality Dataset for Chinese Creative Writing with Thought Processes 

**Authors**: Yunwen Li, Shuangshuang Ying, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Tianyu Zheng, Xeron Du, Qiguang Chen, Jiajun Shi, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Stephen Huang, Wanxiang Che, Chenghua Lin, Eli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14763)  

**Abstract**: Large language models exhibit systematic deficiencies in creative writing, particularly in non-English contexts where training data is scarce and lacks process-level supervision. We present COIG-Writer, a novel Chinese creative writing dataset that captures both diverse outputs and their underlying thought processes through systematic reverse-engineering of high-quality texts. Unlike existing datasets that provide only input-output pairs, COIG-Writer comprises 1,665 meticulously curated triplets spanning 51 genres, each containing: (1) a reverse-engineered prompt, (2) detailed creative reasoning documenting decision-making processes, and (3) the final text. Through comprehensive experiments, we identify a two-component model of creative writing: narrative logic (provided by process supervision) and linguistic expression (maintained by general-purpose data). Our findings reveal three critical insights: (1) Process supervision is highly effective but requires stabilization with general data. A ratio of at least one creative sample to twelve general samples is needed to achieve optimal performance; below this threshold, the win rate progressively degrades (from 62.75% down to 35.78%)., (2) creative capabilities are culturally-bound with no cross-lingual transfer (89.26pp gap between Chinese and English performance), and (3) lexical diversity inversely correlates with creative quality (TTR paradox), suggesting high diversity signals compensatory behavior for logical deficiencies. These findings establish that creative excellence emerges from the interaction between logical scaffolding and linguistic grounding, analogous to how mathematical reasoning enhances but cannot replace linguistic competence in foundation models. 

---
# Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries 

**Authors**: Divyat Mahajan, Sachin Goyal, Badr Youbi Idrissi, Mohammad Pezeshki, Ioannis Mitliagkas, David Lopez-Paz, Kartik Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2510.14751)  

**Abstract**: Next-token prediction (NTP) has driven the success of large language models (LLMs), but it struggles with long-horizon reasoning, planning, and creative writing, with these limitations largely attributed to teacher-forced training. Multi-token prediction (MTP) partially mitigates these issues by predicting several future tokens at once, but it mostly captures short-range dependencies and offers limited improvement. We propose future summary prediction (FSP), which trains an auxiliary head to predict a compact representation of the long-term future, preserving information relevant for long-form generations. We explore two variants of FSP: handcrafted summaries, for example, a bag of words summary of the future of the sequence, and learned summaries, which use embeddings produced by a reverse language model trained from right to left. Large-scale pretraining experiments (3B and 8B-parameter models) demonstrate that FSP provides improvements over both NTP and MTP across math, reasoning, and coding benchmarks. 

---
# DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models 

**Authors**: Simone Carnemolla, Matteo Pennisi, Sarinda Samarasinghe, Giovanni Bellitto, Simone Palazzo, Daniela Giordano, Mubarak Shah, Concetto Spampinato  

**Link**: [PDF](https://arxiv.org/pdf/2510.14741)  

**Abstract**: Understanding and explaining the behavior of machine learning models is essential for building transparent and trustworthy AI systems. We introduce DEXTER, a data-free framework that employs diffusion models and large language models to generate global, textual explanations of visual classifiers. DEXTER operates by optimizing text prompts to synthesize class-conditional images that strongly activate a target classifier. These synthetic samples are then used to elicit detailed natural language reports that describe class-specific decision patterns and biases. Unlike prior work, DEXTER enables natural language explanation about a classifier's decision process without access to training data or ground-truth labels. We demonstrate DEXTER's flexibility across three tasks-activation maximization, slice discovery and debiasing, and bias explanation-each illustrating its ability to uncover the internal mechanisms of visual classifiers. Quantitative and qualitative evaluations, including a user study, show that DEXTER produces accurate, interpretable outputs. Experiments on ImageNet, Waterbirds, CelebA, and FairFaces confirm that DEXTER outperforms existing approaches in global model explanation and class-level bias reporting. Code is available at this https URL. 

---
# Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling 

**Authors**: Alexandru Meterez, Depen Morwani, Jingfeng Wu, Costin-Andrei Oncescu, Cengiz Pehlevan, Sham Kakade  

**Link**: [PDF](https://arxiv.org/pdf/2510.14717)  

**Abstract**: Increasing the batch size during training -- a ''batch ramp'' -- is a promising strategy to accelerate large language model pretraining. While for SGD, doubling the batch size can be equivalent to halving the learning rate, the optimal strategy for adaptive optimizers like Adam is less clear. As a result, any batch-ramp scheduling, if used at all, is typically tuned heuristically. This work develops a principled framework for batch-size scheduling and introduces Seesaw: whenever a standard scheduler would halve the learning rate, Seesaw instead multiplies it by $1/\sqrt{2}$ and doubles the batch size, preserving loss dynamics while reducing serial steps. Theoretically, we provide, to our knowledge, the first finite-sample proof of equivalence between learning-rate decay and batch-size ramp-up for SGD on noisy linear regression, and we extend this equivalence to normalized SGD, a tractable proxy for Adam, under a variance-dominated regime observed in practice. Empirically, on 150M/300M/600M-parameter models trained at Chinchilla scale using a constant (critical) batch size, Seesaw matches cosine decay at equal FLOPs while reducing wall-clock time by $\approx 36\%$, approaching the theoretical limit implied by our analysis. 

---
# Camera Movement Classification in Historical Footage: A Comparative Study of Deep Video Models 

**Authors**: Tingyu Lin, Armin Dadras, Florian Kleber, Robert Sablatnig  

**Link**: [PDF](https://arxiv.org/pdf/2510.14713)  

**Abstract**: Camera movement conveys spatial and narrative information essential for understanding video content. While recent camera movement classification (CMC) methods perform well on modern datasets, their generalization to historical footage remains unexplored. This paper presents the first systematic evaluation of deep video CMC models on archival film material. We summarize representative methods and datasets, highlighting differences in model design and label definitions. Five standard video classification models are assessed on the HISTORIAN dataset, which includes expert-annotated World War II footage. The best-performing model, Video Swin Transformer, achieves 80.25% accuracy, showing strong convergence despite limited training data. Our findings highlight the challenges and potential of adapting existing models to low-quality video and motivate future work combining diverse input modalities and temporal architectures. 

---
# Where are the Whales: A Human-in-the-loop Detection Method for Identifying Whales in High-resolution Satellite Imagery 

**Authors**: Caleb Robinson, Kimberly T. Goetz, Christin B. Khan, Meredith Sackett, Kathleen Leonard, Rahul Dodhia, Juan M. Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2510.14709)  

**Abstract**: Effective monitoring of whale populations is critical for conservation, but traditional survey methods are expensive and difficult to scale. While prior work has shown that whales can be identified in very high-resolution (VHR) satellite imagery, large-scale automated detection remains challenging due to a lack of annotated imagery, variability in image quality and environmental conditions, and the cost of building robust machine learning pipelines over massive remote sensing archives. We present a semi-automated approach for surfacing possible whale detections in VHR imagery using a statistical anomaly detection method that flags spatial outliers, i.e. "interesting points". We pair this detector with a web-based labeling interface designed to enable experts to quickly annotate the interesting points. We evaluate our system on three benchmark scenes with known whale annotations and achieve recalls of 90.3% to 96.4%, while reducing the area requiring expert inspection by up to 99.8% -- from over 1,000 sq km to less than 2 sq km in some cases. Our method does not rely on labeled training data and offers a scalable first step toward future machine-assisted marine mammal monitoring from space. We have open sourced this pipeline at this https URL. 

---
# FedPPA: Progressive Parameter Alignment for Personalized Federated Learning 

**Authors**: Maulidi Adi Prasetia, Muhamad Risqi U. Saputra, Guntur Dharma Putra  

**Link**: [PDF](https://arxiv.org/pdf/2510.14698)  

**Abstract**: Federated Learning (FL) is designed as a decentralized, privacy-preserving machine learning paradigm that enables multiple clients to collaboratively train a model without sharing their data. In real-world scenarios, however, clients often have heterogeneous computational resources and hold non-independent and identically distributed data (non-IID), which poses significant challenges during training. Personalized Federated Learning (PFL) has emerged to address these issues by customizing models for each client based on their unique data distribution. Despite its potential, existing PFL approaches typically overlook the coexistence of model and data heterogeneity arising from clients with diverse computational capabilities. To overcome this limitation, we propose a novel method, called Progressive Parameter Alignment (FedPPA), which progressively aligns the weights of common layers across clients with the global model's weights. Our approach not only mitigates inconsistencies between global and local models during client updates, but also preserves client's local knowledge, thereby enhancing personalization robustness in non-IID settings. To further enhance the global model performance while retaining strong personalization, we also integrate entropy-based weighted averaging into the FedPPA framework. Experiments on three image classification datasets, including MNIST, FMNIST, and CIFAR-10, demonstrate that FedPPA consistently outperforms existing FL algorithms, achieving superior performance in personalized adaptation. 

---
# xLLM Technical Report 

**Authors**: Tongxuan Liu, Tao Peng, Peijun Yang, Xiaoyang Zhao, Xiusheng Lu, Weizhe Huang, Zirui Liu, Xiaoyu Chen, Zhiwei Liang, Jun Xiong, Donghe Jin, Minchao Zhang, Jinrong Guo, Yingxu Deng, Xu Zhang, Xianzhe Dong, Siqi Wang, Siyu Wu, Yu Wu, Zihan Tang, Yuting Zeng, Yanshu Wang, Jinguang Liu, Meng Kang, Menxin Li, Yunlong Wang, Yiming Liu, Xiaolong Ma, Yifan Wang, Yichen Zhang, Jinrun Yin, Keyang Zheng, Jiawei Yin, Jun Zhang, Ziyue Wang, Xiaobo Lin, Liangyu Liu, Liwei Lan, Yang Liu, Chunhua Peng, Han Liu, Songcheng Ren, Xuezhu Wang, Yunheng Shen, Yi Wang, Guyue Liu, Hui Chen, Tong Yang, Hailong Yang, Jing Li, Guiguang Ding, Ke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14686)  

**Abstract**: We introduce xLLM, an intelligent and efficient Large Language Model (LLM) inference framework designed for high-performance, large-scale enterprise-grade serving, with deep optimizations for diverse AI accelerators. To address these challenges, xLLM builds a novel decoupled service-engine architecture. At the service layer, xLLM-Service features an intelligent scheduling module that efficiently processes multimodal requests and co-locates online and offline tasks through unified elastic scheduling to maximize cluster utilization. This module also relies on a workload-adaptive dynamic Prefill-Decode (PD) disaggregation policy and a novel Encode-Prefill-Decode (EPD) disaggregation policy designed for multimodal inputs. Furthermore, it incorporates a distributed architecture to provide global KV Cache management and robust fault-tolerant capabilities for high availability. At the engine layer, xLLM-Engine co-optimizes system and algorithm designs to fully saturate computing resources. This is achieved through comprehensive multi-layer execution pipeline optimizations, an adaptive graph mode and an xTensor memory management. xLLM-Engine also further integrates algorithmic enhancements such as optimized speculative decoding and dynamic EPLB, collectively serving to substantially boost throughput and inference efficiency. Extensive evaluations demonstrate that xLLM delivers significantly superior performance and resource efficiency. Under identical TPOT constraints, xLLM achieves throughput up to 1.7x that of MindIE and 2.2x that of vLLM-Ascend with Qwen-series models, while maintaining an average throughput of 1.7x that of MindIE with Deepseek-series models. xLLM framework is publicly available at this https URL and this https URL. 

---
# When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift nuPlan Benchmarks 

**Authors**: Steffen Hagedorn, Luka Donkov, Aron Distelzweig, Alexandru P. Condurache  

**Link**: [PDF](https://arxiv.org/pdf/2510.14677)  

**Abstract**: Planner evaluation in closed-loop simulation often uses rule-based traffic agents, whose simplistic and passive behavior can hide planner deficiencies and bias rankings. Widely used IDM agents simply follow a lead vehicle and cannot react to vehicles in adjacent lanes, hindering tests of complex interaction capabilities. We address this issue by integrating the state-of-the-art learned traffic agent model SMART into nuPlan. Thus, we are the first to evaluate planners under more realistic conditions and quantify how conclusions shift when narrowing the sim-to-real gap. Our analysis covers 14 recent planners and established baselines and shows that IDM-based simulation overestimates planning performance: nearly all scores deteriorate. In contrast, many planners interact better than previously assumed and even improve in multi-lane, interaction-heavy scenarios like lane changes or turns. Methods trained in closed-loop demonstrate the best and most stable driving performance. However, when reaching their limits in augmented edge-case scenarios, all learned planners degrade abruptly, whereas rule-based planners maintain reasonable basic behavior. Based on our results, we suggest SMART-reactive simulation as a new standard closed-loop benchmark in nuPlan and release the SMART agents as a drop-in alternative to IDM at this https URL. 

---
# An Efficient Rubric-based Generative Verifier for Search-Augmented LLMs 

**Authors**: Linyue Ma, Yilong Xu, Xiang Long, Zhi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.14660)  

**Abstract**: Search augmentation empowers Large Language Models with retrieval capabilities to overcome the limitations imposed by static parameters. Recently, Reinforcement Learning leverages tailored reward signals as a viable technique to enhance LLMs performing tasks involving search. However, existing reward modeling for search-augmented LLMs faces several limitations. Rule-based rewards, such as Exact Match, are verifiable but fragile to variations in expression and cannot be applied to long-form workloads. In contrast, generative rewards improve robustness, but designing verifiable and stable rewards for long-form workloads in dynamic corpora remains challenging and also incurs high computational costs. In this paper, we propose a unified and verifiable paradigm, "nugget-as-rubric", which treats atomic information points as structured evaluation criteria for different search-augmentation workloads. Short-form tasks correspond to a single rubric, whereas long-form tasks expand to multiple rubrics aligned with the question's information needs. To support long-form settings, we design an automatic rubric construction pipeline based on query rewriting, which can automatically retrieve passages relevant to each question and extract rubrics from them, both from static corpora and from dynamic online web content. Furthermore, we introduce \textbf{Search-Gen-V}, a 4B-parameter efficient generative verifier under our proposed verifiable paradigm, which is trained via the idea of distillation and a two-stage strategy. Experimental results show that Search-Gen-V achieves strong verification accuracy across different workloads, making it a scalable, robust, and efficient verifiable reward constructor for search-augmented LLMs. 

---
# Galaxy Morphology Classification with Counterfactual Explanation 

**Authors**: Zhuo Cao, Lena Krieger, Hanno Scharr, Ira Assent  

**Link**: [PDF](https://arxiv.org/pdf/2510.14655)  

**Abstract**: Galaxy morphologies play an essential role in the study of the evolution of galaxies. The determination of morphologies is laborious for a large amount of data giving rise to machine learning-based approaches. Unfortunately, most of these approaches offer no insight into how the model works and make the results difficult to understand and explain. We here propose to extend a classical encoder-decoder architecture with invertible flow, allowing us to not only obtain a good predictive performance but also provide additional information about the decision process with counterfactual explanations. 

---
# In-Context Learning with Unpaired Clips for Instruction-based Video Editing 

**Authors**: Xinyao Liao, Xianfang Zeng, Ziye Song, Zhoujie Fu, Gang Yu, Guosheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14648)  

**Abstract**: Despite the rapid progress of instruction-based image editing, its extension to video remains underexplored, primarily due to the prohibitive cost and complexity of constructing large-scale paired video editing datasets. To address this challenge, we introduce a low-cost pretraining strategy for instruction-based video editing that leverages in-context learning from unpaired video clips. We show that pretraining a foundation video generation model with this strategy endows it with general editing capabilities, such as adding, replacing, or deleting operations, according to input editing instructions. The pretrained model can then be efficiently refined with a small amount of high-quality paired editing data. Built upon HunyuanVideoT2V, our framework first pretrains on approximately 1M real video clips to learn basic editing concepts, and subsequently fine-tunes on fewer than 150k curated editing pairs to extend more editing tasks and improve the editing quality. Comparative experiments show that our method surpasses existing instruction-based video editing approaches in both instruction alignment and visual fidelity, achieving a 12\% improvement in editing instruction following and a 15\% improvement in editing quality. 

---
# The Bidding Games: Reinforcement Learning for MEV Extraction on Polygon Blockchain 

**Authors**: Andrei Seoev, Leonid Gremyachikh, Anastasiia Smirnova, Yash Madhwal, Alisa Kalacheva, Dmitry Belousov, Ilia Zubov, Aleksei Smirnov, Denis Fedyanin, Vladimir Gorgadze, Yury Yanovich  

**Link**: [PDF](https://arxiv.org/pdf/2510.14642)  

**Abstract**: In blockchain networks, the strategic ordering of transactions within blocks has emerged as a significant source of profit extraction, known as Maximal Extractable Value (MEV). The transition from spam-based Priority Gas Auctions to structured auction mechanisms like Polygon Atlas has transformed MEV extraction from public bidding wars into sealed-bid competitions under extreme time constraints. While this shift reduces network congestion, it introduces complex strategic challenges where searchers must make optimal bidding decisions within a sub-second window without knowledge of competitor behavior or presence. Traditional game-theoretic approaches struggle in this high-frequency, partially observable environment due to their reliance on complete information and static equilibrium assumptions. We present a reinforcement learning framework for MEV extraction on Polygon Atlas and make three contributions: (1) A novel simulation environment that accurately models the stochastic arrival of arbitrage opportunities and probabilistic competition in Atlas auctions; (2) A PPO-based bidding agent optimized for real-time constraints, capable of adaptive strategy formulation in continuous action spaces while maintaining production-ready inference speeds; (3) Empirical validation demonstrating our history-conditioned agent captures 49\% of available profits when deployed alongside existing searchers and 81\% when replacing the market leader, significantly outperforming static bidding strategies. Our work establishes that reinforcement learning provides a critical advantage in high-frequency MEV environments where traditional optimization methods fail, offering immediate value for industrial participants and protocol designers alike. 

---
# Causality Enhancement for Cross-Domain Recommendation 

**Authors**: Zhibo Wu, Yunfan Wu, Lin Jiang, Ping Yang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14641)  

**Abstract**: Cross-domain recommendation forms a crucial component in recommendation systems. It leverages auxiliary information through source domain tasks or features to enhance target domain recommendations. However, incorporating inconsistent source domain tasks may result in insufficient cross-domain modeling or negative transfer. While incorporating source domain features without considering the underlying causal relationships may limit their contribution to final predictions. Thus, a natural idea is to directly train a cross-domain representation on a causality-labeled dataset from the source to target domain. Yet this direction has been rarely explored, as identifying unbiased real causal labels is highly challenging in real-world scenarios. In this work, we attempt to take a first step in this direction by proposing a causality-enhanced framework, named CE-CDR. Specifically, we first reformulate the cross-domain recommendation as a causal graph for principled guidance. We then construct a causality-aware dataset heuristically. Subsequently, we derive a theoretically unbiased Partial Label Causal Loss to generalize beyond the biased causality-aware dataset to unseen cross-domain patterns, yielding an enriched cross-domain representation, which is then fed into the target model to enhance target-domain recommendations. Theoretical and empirical analyses, as well as extensive experiments, demonstrate the rationality and effectiveness of CE-CDR and its general applicability as a model-agnostic plugin. Moreover, it has been deployed in production since April 2025, showing its practical value in real-world applications. 

---
# RLAIF-SPA: Optimizing LLM-based Emotional Speech Synthesis via RLAIF 

**Authors**: Qing Yang, Zhenghao Liu, Junxin Wang, Yangfan Du, Pengcheng Huang, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14628)  

**Abstract**: Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the RLAIF-SPA framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic-emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Semantic Accuracy Feedback to ensure the generation of clear and accurate speech. Experiments on the Libri Speech dataset show that RLAIF-SPA outperforms Chat-TTS, with a 26.1% reduction in WER, a 9.1% increase in SIM-O, and over 10% improvement in human evaluation. 

---
# GemiRec: Interest Quantization and Generation for Multi-Interest Recommendation 

**Authors**: Zhibo Wu, Yunfan Wu, Quan Liu, Lin Jiang, Ping Yang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14626)  

**Abstract**: Multi-interest recommendation has gained attention, especially in industrial retrieval stage. Unlike classical dual-tower methods, it generates multiple user representations instead of a single one to model comprehensive user interests. However, prior studies have identified two underlying limitations: the first is interest collapse, where multiple representations homogenize. The second is insufficient modeling of interest evolution, as they struggle to capture latent interests absent from a user's historical behavior. We begin with a thorough review of existing works in tackling these limitations. Then, we attempt to tackle these limitations from a new perspective. Specifically, we propose a framework-level refinement for multi-interest recommendation, named GemiRec. The proposed framework leverages interest quantization to enforce a structural interest separation and interest generation to learn the evolving dynamics of user interests explicitly. It comprises three modules: (a) Interest Dictionary Maintenance Module (IDMM) maintains a shared quantized interest dictionary. (b) Multi-Interest Posterior Distribution Module (MIPDM) employs a generative model to capture the distribution of user future interests. (c) Multi-Interest Retrieval Module (MIRM) retrieves items using multiple user-interest representations. Both theoretical and empirical analyses, as well as extensive experiments, demonstrate its advantages and effectiveness. Moreover, it has been deployed in production since March 2025, showing its practical value in industrial applications. 

---
# LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching 

**Authors**: Zhuo Cao, Xuan Zhao, Lena Krieger, Hanno Scharr, Ira Assent  

**Link**: [PDF](https://arxiv.org/pdf/2510.14623)  

**Abstract**: The growing integration of machine learning (ML) and artificial intelligence (AI) models into high-stakes domains such as healthcare and scientific research calls for models that are not only accurate but also interpretable. Among the existing explainable methods, counterfactual explanations offer interpretability by identifying minimal changes to inputs that would alter a model's prediction, thus providing deeper insights. However, current counterfactual generation methods suffer from critical limitations, including gradient vanishing, discontinuous latent spaces, and an overreliance on the alignment between learned and true decision boundaries. To overcome these limitations, we propose LeapFactual, a novel counterfactual explanation algorithm based on conditional flow matching. LeapFactual generates reliable and informative counterfactuals, even when true and learned decision boundaries diverge. Following a model-agnostic approach, LeapFactual is not limited to models with differentiable loss functions. It can even handle human-in-the-loop systems, expanding the scope of counterfactual explanations to domains that require the participation of human annotators, such as citizen science. We provide extensive experiments on benchmark and real-world datasets showing that LeapFactual generates accurate and in-distribution counterfactual explanations that offer actionable insights. We observe, for instance, that our reliable counterfactual samples with labels aligning to ground truth can be beneficially used as new training data to enhance the model. The proposed method is broadly applicable and enhances both scientific knowledge discovery and non-expert interpretability. 

---
# Code-driven Number Sequence Calculation: Enhancing the inductive Reasoning Abilities of Large Language Models 

**Authors**: Kedi Chen, Zhikai Lei, Xu Guo, Xuecheng Wu, Siyuan Zeng, Jianghao Yin, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, Qipeng Guo, Kai Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14620)  

**Abstract**: Large language models (LLMs) make remarkable progress in reasoning tasks. Among different reasoning modes, inductive reasoning, due to its better alignment with human learning, attracts increasing interest. However, research on inductive reasoning faces certain challenges. First, existing inductive data mostly focuses on superficial regularities while lacking more complex internal patterns. Second, current works merely prompt LLMs or finetune on simple prompt-response pairs, but do not provide precise thinking processes nor implement difficulty control. Unlike previous work, we address these challenges by introducing \textit{CodeSeq}, a synthetic post-training dataset built from number sequences. We package number sequences into algorithmic problems to discover their general terms, defining a general term generation (GTG) task correspondingly. Our pipeline generates supervised finetuning data by reflecting on failed test cases and incorporating iterative corrections, thereby teaching LLMs to learn autonomous case generation and self-checking. Additionally, it leverages reinforcement learning with a novel Case-Synergy Solvability Scaling Reward based on both solvability, estimated from the problem pass rate, and the success rate of self-directed case generation, enabling models to learn more effectively from both successes and failures. Experimental results show that the models trained with \textit{CodeSeq} improve on various reasoning tasks and can preserve the models' OOD performance. 

---
# Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures 

**Authors**: Shuangshuang Ying, Yunwen Li, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Xeron Du, Tianyu Zheng, Yichi Zhang, Letian Ni, Yuyang Cheng, Qiguang Chen, Jingzhe Ding, Shengda Long, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Ge Zhang, Wenhao Huang, Wanxiang Che, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14616)  

**Abstract**: Current preference learning methods achieve high accuracy on standard benchmarks but exhibit significant performance degradation when objective quality signals are removed. We introduce WritingPreferenceBench, a dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres, where responses are matched for objective correctness, factual accuracy, and length. On this benchmark, sequence-based reward models--the standard architecture for RLHF--achieve only 52.7% mean accuracy, while zero-shot language model judges perform at 53.9%. In contrast, generative reward models that produce explicit reasoning chains achieve 81.8% accuracy. We observe high within-model variance across genres: individual models range from 18.2% to 81.8% accuracy across different writing categories, with standard deviations averaging 10.1%. This variance persists regardless of model scale, with 27B parameter models showing no consistent improvement over 8B variants. Our results suggest that current RLHF methods primarily learn to detect objective errors rather than capture subjective quality preferences (e.g., creativity, stylistic flair, and emotional resonance), and that successful preference modeling may require intermediate reasoning representations rather than direct classification. 

---
# An Active Inference Model of Mouse Point-and-Click Behaviour 

**Authors**: Markus Klar, Sebastian Stein, Fraser Paterson, John H. Williamson, Roderick Murray-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2510.14611)  

**Abstract**: We explore the use of Active Inference (AIF) as a computational user model for spatial pointing, a key problem in Human-Computer Interaction (HCI). We present an AIF agent with continuous state, action, and observation spaces, performing one-dimensional mouse pointing and clicking. We use a simple underlying dynamic system to model the mouse cursor dynamics with realistic perceptual delay. In contrast to previous optimal feedback control-based models, the agent's actions are selected by minimizing Expected Free Energy, solely based on preference distributions over percepts, such as observing clicking a button correctly. Our results show that the agent creates plausible pointing movements and clicks when the cursor is over the target, with similar end-point variance to human users. In contrast to other models of pointing, we incorporate fully probabilistic, predictive delay compensation into the agent. The agent shows distinct behaviour for differing target difficulties without the need to retune system parameters, as done in other approaches. We discuss the simulation results and emphasize the challenges in identifying the correct configuration of an AIF agent interacting with continuous systems. 

---
# Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering 

**Authors**: Yuyang Hong, Jiaqi Gu, Qi Yang, Lubin Fan, Yue Wu, Ying Wang, Kun Ding, Shiming Xiang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.14605)  

**Abstract**: Knowledge-based visual question answering (KB-VQA) requires visual language models (VLMs) to integrate visual understanding with external knowledge retrieval. Although retrieval-augmented generation (RAG) achieves significant advances in this task by combining knowledge-base querying, it still struggles with the quality of multimodal queries and the relevance of retrieved results. To overcome these challenges, we propose a novel three-stage method, termed Wiki-PRF, including Processing, Retrieval and Filtering stages. The processing stage dynamically invokes visual tools to extract precise multimodal information for retrieval. The retrieval stage integrates visual and text features to achieve multimodal knowledge retrieval. The filtering stage performs relevance filtering and concentration on retrieval results. To this end, we introduce a visual language model trained with answer accuracy and format consistency as reward signals via a reinforcement learning manner. This enhances the model's reasoning, tool invocation for accurate queries, and filtering of irrelevant content. Experiments on benchmark datasets (E-VQA and InfoSeek) show significant improvements~(36.0 and 42.8) in answer quality, achieving state-of-the-art performance. Code is available at this https URL 

---
# Just-In-Time Objectives: A General Approach for Specialized AI Interactions 

**Authors**: Michelle S. Lam, Omar Shaikh, Hallie Xu, Alice Guo, Diyi Yang, Jeffrey Heer, James A. Landay, Michael S. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.14591)  

**Abstract**: Large language models promise a broad set of functions, but when not given a specific objective, they default to milquetoast results such as drafting emails littered with cliches. We demonstrate that inferring the user's in-the-moment objective, then rapidly optimizing for that singular objective, enables LLMs to produce tools, interfaces, and responses that are more responsive and desired. We contribute an architecture for automatically inducing just-in-time objectives by passively observing user behavior, then steering downstream AI systems through generation and evaluation against this objective. Inducing just-in-time objectives (e.g., "Clarify the abstract's research contribution") enables automatic generation of tools, e.g., those that critique a draft based on relevant HCI methodologies, anticipate related researchers' reactions, or surface ambiguous terminology. In a series of experiments (N=14, N=205) on participants' own tasks, JIT objectives enable LLM outputs that achieve 66-86% win rates over typical LLMs, and in-person use sessions (N=17) confirm that JIT objectives produce specialized tools unique to each participant. 

---
# STANCE: Motion Coherent Video Generation Via Sparse-to-Dense Anchored Encoding 

**Authors**: Zhifei Chen, Tianshuo Xu, Leyi Wu, Luozhou Wang, Dongyu Yan, Zihan You, Wenting Luo, Guo Zhang, Yingcong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14588)  

**Abstract**: Video generation has recently made striking visual progress, but maintaining coherent object motion and interactions remains difficult. We trace two practical bottlenecks: (i) human-provided motion hints (e.g., small 2D maps) often collapse to too few effective tokens after encoding, weakening guidance; and (ii) optimizing for appearance and motion in a single head can favor texture over temporal consistency. We present STANCE, an image-to-video framework that addresses both issues with two simple components. First, we introduce Instance Cues -- a pixel-aligned control signal that turns sparse, user-editable hints into a dense 2.5D (camera-relative) motion field by averaging per-instance flow and augmenting with monocular depth over the instance mask. This reduces depth ambiguity compared to 2D arrow inputs while remaining easy to use. Second, we preserve the salience of these cues in token space with Dense RoPE, which tags a small set of motion tokens (anchored on the first frame) with spatial-addressable rotary embeddings. Paired with joint RGB \(+\) auxiliary-map prediction (segmentation or depth), our model anchors structure while RGB handles appearance, stabilizing optimization and improving temporal coherence without requiring per-frame trajectory scripts. 

---
# Local Causal Discovery for Statistically Efficient Causal Inference 

**Authors**: Mátyás Schubert, Tom Claassen, Sara Magliacane  

**Link**: [PDF](https://arxiv.org/pdf/2510.14582)  

**Abstract**: Causal discovery methods can identify valid adjustment sets for causal effect estimation for a pair of target variables, even when the underlying causal graph is unknown. Global causal discovery methods focus on learning the whole causal graph and therefore enable the recovery of optimal adjustment sets, i.e., sets with the lowest asymptotic variance, but they quickly become computationally prohibitive as the number of variables grows. Local causal discovery methods offer a more scalable alternative by focusing on the local neighborhood of the target variables, but are restricted to statistically suboptimal adjustment sets. In this work, we propose Local Optimal Adjustments Discovery (LOAD), a sound and complete causal discovery approach that combines the computational efficiency of local methods with the statistical optimality of global methods. First, LOAD identifies the causal relation between the targets and tests if the causal effect is identifiable by using only local information. If it is identifiable, it then finds the optimal adjustment set by leveraging local causal discovery to infer the mediators and their parents. Otherwise, it returns the locally valid parent adjustment sets based on the learned local structure. In our experiments on synthetic and realistic data LOAD outperforms global methods in scalability, while providing more accurate effect estimation than local methods. 

---
# Selective Labeling with False Discovery Rate Control 

**Authors**: Huipeng Huang, Wenbo Liao, Huajun Xi, Hao Zeng, Mengchen Zhao, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.14581)  

**Abstract**: Obtaining high-quality labels for large datasets is expensive, requiring massive annotations from human experts. While AI models offer a cost-effective alternative by predicting labels, their label quality is compromised by the unavoidable labeling errors. Existing methods mitigate this issue through selective labeling, where AI labels a subset and human labels the remainder. However, these methods lack theoretical guarantees on the quality of AI-assigned labels, often resulting in unacceptably high labeling error within the AI-labeled subset. To address this, we introduce \textbf{Conformal Labeling}, a novel method to identify instances where AI predictions can be provably trusted. This is achieved by controlling the false discovery rate (FDR), the proportion of incorrect labels within the selected subset. In particular, we construct a conformal $p$-value for each test instance by comparing AI models' predicted confidence to those of calibration instances mislabeled by AI models. Then, we select test instances whose $p$-values are below a data-dependent threshold, certifying AI models' predictions as trustworthy. We provide theoretical guarantees that Conformal Labeling controls the FDR below the nominal level, ensuring that a predefined fraction of AI-assigned labels is correct on average. Extensive experiments demonstrate that our method achieves tight FDR control with high power across various tasks, including image and text labeling, and LLM QA. 

---
# Agentic Entropy-Balanced Policy Optimization 

**Authors**: Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14545)  

**Abstract**: Recently, Agentic Reinforcement Learning (Agentic RL) has made significant progress in incentivizing the multi-turn, long-horizon tool-use capabilities of web agents. While mainstream agentic RL algorithms autonomously explore high-uncertainty tool-call steps under the guidance of entropy, excessive reliance on entropy signals can impose further constraints, leading to the training collapse. In this paper, we delve into the challenges caused by entropy and propose the Agentic Entropy-Balanced Policy Optimization (AEPO), an agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. AEPO comprises two core components: (1) a dynamic entropy-balanced rollout mechanism that adaptively allocate global and branch sampling budget through entropy pre-monitoring, while imposing a branch penalty on consecutive high-entropy tool-call steps to prevent over-branching issues; and (2) Entropy-Balanced Policy Optimization that inserts a stop-gradient operation into the high-entropy clipping term to preserve and properly rescale gradients on high-entropy tokens, while incorporating entropy-aware advantage estimation to prioritize learning on high-uncertainty tokens. Results across 14 challenging datasets show that AEPO consistently outperforms 7 mainstream RL algorithms. With just 1K RL samples, Qwen3-14B with AEPO achieves impressive results: 47.6% on GAIA, 11.2% on Humanity's Last Exam, and 43.0% on WebWalker for Pass@1; 65.0% on GAIA, 26.0% on Humanity's Last Exam, and 70.0% on WebWalker for Pass@5. Further analysis reveals that AEPO improves rollout sampling diversity while maintaining stable policy entropy, facilitating scalable web agent training. 

---
# Real-Time Surgical Instrument Defect Detection via Non-Destructive Testing 

**Authors**: Qurrat Ul Ain, Atif Aftab Ahmed Jilani, Zunaira Shafqat, Nigar Azhar Butt  

**Link**: [PDF](https://arxiv.org/pdf/2510.14525)  

**Abstract**: Defective surgical instruments pose serious risks to sterility, mechanical integrity, and patient safety, increasing the likelihood of surgical complications. However, quality control in surgical instrument manufacturing often relies on manual inspection, which is prone to human error and inconsistency. This study introduces SurgScan, an AI-powered defect detection framework for surgical instruments. Using YOLOv8, SurgScan classifies defects in real-time, ensuring high accuracy and industrial scalability. The model is trained on a high-resolution dataset of 102,876 images, covering 11 instrument types and five major defect categories. Extensive evaluation against state-of-the-art CNN architectures confirms that SurgScan achieves the highest accuracy (99.3%) with real-time inference speeds of 4.2-5.8 ms per image, making it suitable for industrial deployment. Statistical analysis demonstrates that contrast-enhanced preprocessing significantly improves defect detection, addressing key limitations in visual inspection. SurgScan provides a scalable, cost-effective AI solution for automated quality control, reducing reliance on manual inspection while ensuring compliance with ISO 13485 and FDA standards, paving the way for enhanced defect detection in medical manufacturing. 

---
# State Your Intention to Steer Your Attention: An AI Assistant for Intentional Digital Living 

**Authors**: Juheon Choi, Juyoung Lee, Jian Kim, Chanyoung Kim, Taewon Min, W. Bradley Knox, Min Kyung Lee, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14513)  

**Abstract**: When working on digital devices, people often face distractions that can lead to a decline in productivity and efficiency, as well as negative psychological and emotional impacts. To address this challenge, we introduce a novel Artificial Intelligence (AI) assistant that elicits a user's intention, assesses whether ongoing activities are in line with that intention, and provides gentle nudges when deviations occur. The system leverages a large language model to analyze screenshots, application titles, and URLs, issuing notifications when behavior diverges from the stated goal. Its detection accuracy is refined through initial clarification dialogues and continuous user feedback. In a three-week, within-subjects field deployment with 22 participants, we compared our assistant to both a rule-based intent reminder system and a passive baseline that only logged activity. Results indicate that our AI assistant effectively supports users in maintaining focus and aligning their digital behavior with their intentions. Our source code is publicly available at this url this https URL 

---
# E2Edev: Benchmarking Large Language Models in End-to-End Software Development Task 

**Authors**: Jingyao Liu, Chen Huang, Zhizhao Guan, Wenqiang Lei, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.14509)  

**Abstract**: E2EDev comprises (i) a fine-grained set of user requirements, (ii) {multiple BDD test scenarios with corresponding Python step implementations for each requirement}, and (iii) a fully automated testing pipeline built on the Behave framework. To ensure its quality while reducing the annotation effort, E2EDev leverages our proposed Human-in-the-Loop Multi-Agent Annotation Framework (HITL-MAA). {By evaluating various E2ESD frameworks and LLM backbones with E2EDev}, our analysis reveals a persistent struggle to effectively solve these tasks, underscoring the critical need for more effective and cost-efficient E2ESD solutions. Our codebase and benchmark are publicly available at this https URL. 

---
# From Guess2Graph: When and How Can Unreliable Experts Safely Boost Causal Discovery in Finite Samples? 

**Authors**: Sujai Hiremath, Dominik Janzing, Philipp Faller, Patrick Blöbaum, Elke Kirschbaum, Shiva Prasad Kasiviswanathan, Kyra Gan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14488)  

**Abstract**: Causal discovery algorithms often perform poorly with limited samples. While integrating expert knowledge (including from LLMs) as constraints promises to improve performance, guarantees for existing methods require perfect predictions or uncertainty estimates, making them unreliable for practical use. We propose the Guess2Graph (G2G) framework, which uses expert guesses to guide the sequence of statistical tests rather than replacing them. This maintains statistical consistency while enabling performance improvements. We develop two instantiations of G2G: PC-Guess, which augments the PC algorithm, and gPC-Guess, a learning-augmented variant designed to better leverage high-quality expert input. Theoretically, both preserve correctness regardless of expert error, with gPC-Guess provably outperforming its non-augmented counterpart in finite samples when experts are "better than random." Empirically, both show monotonic improvement with expert accuracy, with gPC-Guess achieving significantly stronger gains. 

---
# Semantic representations emerge in biologically inspired ensembles of cross-supervising neural networks 

**Authors**: Roy Urbach, Elad Schneidman  

**Link**: [PDF](https://arxiv.org/pdf/2510.14486)  

**Abstract**: Brains learn to represent information from a large set of stimuli, typically by weak supervision. Unsupervised learning is therefore a natural approach for exploring the design of biological neural networks and their computations. Accordingly, redundancy reduction has been suggested as a prominent design principle of neural encoding, but its ``mechanistic'' biological implementation is unclear. Analogously, unsupervised training of artificial neural networks yields internal representations that allow for accurate stimulus classification or decoding, but typically rely on biologically-implausible implementations. We suggest that interactions between parallel subnetworks in the brain may underlie such learning: we present a model of representation learning by ensembles of neural networks, where each network learns to encode stimuli into an abstract representation space by cross-supervising interactions with other networks, for inputs they receive simultaneously or in close temporal proximity. Aiming for biological plausibility, each network has a small ``receptive field'', thus receiving a fixed part of the external input, and the networks do not share weights. We find that for different types of network architectures, and for both visual or neuronal stimuli, these cross-supervising networks learn semantic representations that are easily decodable and that decoding accuracy is comparable to supervised networks -- both at the level of single networks and the ensemble. We further show that performance is optimal for small receptive fields, and that sparse connectivity between networks is nearly as accurate as all-to-all interactions, with far fewer computations. We thus suggest a sparsely interacting collective of cross-supervising networks as an algorithmic framework for representational learning and collective computation in the brain. 

---
# Stealthy Dual-Trigger Backdoors: Attacking Prompt Tuning in LM-Empowered Graph Foundation Models 

**Authors**: Xiaoyu Xue, Yuni Lai, Chenxi Huang, Yulin Zhu, Gaolei Li, Xiaoge Zhang, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.14470)  

**Abstract**: The emergence of graph foundation models (GFMs), particularly those incorporating language models (LMs), has revolutionized graph learning and demonstrated remarkable performance on text-attributed graphs (TAGs). However, compared to traditional GNNs, these LM-empowered GFMs introduce unique security vulnerabilities during the unsecured prompt tuning phase that remain understudied in current research. Through empirical investigation, we reveal a significant performance degradation in traditional graph backdoor attacks when operating in attribute-inaccessible constrained TAG systems without explicit trigger node attribute optimization. To address this, we propose a novel dual-trigger backdoor attack framework that operates at both text-level and struct-level, enabling effective attacks without explicit optimization of trigger node text attributes through the strategic utilization of a pre-established text pool. Extensive experimental evaluations demonstrate that our attack maintains superior clean accuracy while achieving outstanding attack success rates, including scenarios with highly concealed single-trigger nodes. Our work highlights critical backdoor risks in web-deployed LM-empowered GFMs and contributes to the development of more robust supervision mechanisms for open-source platforms in the era of foundation models. 

---
# LiRA: Linguistic Robust Anchoring for Cross-lingual Large Language Models 

**Authors**: Haolin Li, Haipeng Zhang, Mang Li, Yaohua Wang, Lijie Wen, Yu Zhang, Biqing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14466)  

**Abstract**: As large language models (LLMs) rapidly advance, performance on high-resource languages (e.g., English, Chinese) is nearing saturation, yet remains substantially lower for low-resource languages (e.g., Urdu, Thai) due to limited training data, machine-translation noise, and unstable cross-lingual alignment. We introduce LiRA (Linguistic Robust Anchoring for Large Language Models), a training framework that robustly improves cross-lingual representations under low-resource conditions while jointly strengthening retrieval and reasoning. LiRA comprises two modules: (i) Arca (Anchored Representation Composition Architecture), which anchors low-resource languages to an English semantic space via anchor-based alignment and multi-agent collaborative encoding, preserving geometric stability in a shared embedding space; and (ii) LaSR (Language-coupled Semantic Reasoner), which adds a language-aware lightweight reasoning head with consistency regularization on top of Arca's multilingual representations, unifying the training objective to enhance cross-lingual understanding, retrieval, and reasoning robustness. We further construct and release a multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Experiments across low-resource benchmarks (cross-lingual retrieval, semantic similarity, and reasoning) show consistent gains and robustness under few-shot and noise-amplified settings; ablations validate the contribution of both Arca and LaSR. Code will be released on GitHub and the dataset on Hugging Face. 

---
# Holdout-Loss-Based Data Selection for LLM Finetuning via In-Context Learning 

**Authors**: Ling Zhang, Xianliang Yang, Juwon Yu, Park Cheonyoung, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2510.14459)  

**Abstract**: Fine-tuning large pretrained language models is a common approach for aligning them with human preferences, but noisy or off-target examples can dilute supervision. While small, well-chosen datasets often match the performance of much larger ones, systematic and efficient ways to identify high-value training data remain underexplored. Many current methods rely on heuristics or expensive retraining. We present a theoretically grounded, resource-efficient framework for data selection and reweighting. At its core is an In-Context Approximation (ICA) that estimates the holdout loss a model would incur after training on a candidate example by conditioning on a small, curated holdout set in context. ICA requires no reference model and no additional finetuning. Under a local linearization, ICA is equivalent to a first-order update toward the holdout optimum, motivating its use as a proxy for data value. We derive per-example weights from ICA scores, dynamically reweighting gradient updates as model parameters evolve. Across SFT, DPO, and SimPO, and over diverse backbones and datasets, ICA-based reweighting consistently improves model alignment with minimal overhead. We analyze sensitivity to score update frequency and the choice of $k$ holdout examples for in-context demonstrations, and note limitations for rapidly drifting on-policy updates, highlighting directions for future work. Code and prompts will be released. 

---
# Towards Adaptable Humanoid Control via Adaptive Motion Tracking 

**Authors**: Tao Huang, Huayi Wang, Junli Ren, Kangning Yin, Zirui Wang, Xiao Chen, Feiyu Jia, Wentao Zhang, Junfeng Long, Jingbo Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14454)  

**Abstract**: Humanoid robots are envisioned to adapt demonstrated motions to diverse real-world conditions while accurately preserving motion patterns. Existing motion prior approaches enable well adaptability with a few motions but often sacrifice imitation accuracy, whereas motion-tracking methods achieve accurate imitation yet require many training motions and a test-time target motion to adapt. To combine their strengths, we introduce AdaMimic, a novel motion tracking algorithm that enables adaptable humanoid control from a single reference motion. To reduce data dependence while ensuring adaptability, our method first creates an augmented dataset by sparsifying the single reference motion into keyframes and applying light editing with minimal physical assumptions. A policy is then initialized by tracking these sparse keyframes to generate dense intermediate motions, and adapters are subsequently trained to adjust tracking speed and refine low-level actions based on the adjustment, enabling flexible time warping that further improves imitation accuracy and adaptability. We validate these significant improvements in our approach in both simulation and the real-world Unitree G1 humanoid robot in multiple tasks across a wide range of adaptation conditions. Videos and code are available at this https URL. 

---
# Feature Selection and Regularization in Multi-Class Classification: An Empirical Study of One-vs-Rest Logistic Regression with Gradient Descent Optimization and L1 Sparsity Constraints 

**Authors**: Jahidul Arafat, Fariha Tasmin, Md Kaosar Uddin, Sanjaya Poudel, Eftakhar Ahmed Arnob  

**Link**: [PDF](https://arxiv.org/pdf/2510.14449)  

**Abstract**: Multi-class wine classification presents fundamental trade-offs between model accuracy, feature dimensionality, and interpretability - critical factors for production deployment in analytical chemistry. This paper presents a comprehensive empirical study of One-vs-Rest logistic regression on the UCI Wine dataset (178 samples, 3 cultivars, 13 chemical features), comparing from-scratch gradient descent implementation against scikit-learn's optimized solvers and quantifying L1 regularization effects on feature sparsity. Manual gradient descent achieves 92.59 percent mean test accuracy with smooth convergence, validating theoretical foundations, though scikit-learn provides 24x training speedup and 98.15 percent accuracy. Class-specific analysis reveals distinct chemical signatures with heterogeneous patterns where color intensity varies dramatically (0.31 to 16.50) across cultivars. L1 regularization produces 54-69 percent feature reduction with only 4.63 percent accuracy decrease, demonstrating favorable interpretability-performance trade-offs. We propose an optimal 5-feature subset achieving 62 percent complexity reduction with estimated 92-94 percent accuracy, enabling cost-effective deployment with 80 dollars savings per sample and 56 percent time reduction. Statistical validation confirms robust generalization with sub-2ms prediction latency suitable for real-time quality control. Our findings provide actionable guidelines for practitioners balancing comprehensive chemical analysis against targeted feature measurement in resource-constrained environments. 

---
# A Free Lunch in LLM Compression: Revisiting Retraining after Pruning 

**Authors**: Moritz Wagner, Christophe Roux, Max Zimmer, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2510.14444)  

**Abstract**: While Neural Network pruning typically requires retraining the model to recover pruning-induced performance degradation, state-of-the-art Large Language Models (LLMs) pruning methods instead solve a layer-wise mask selection and reconstruction problem on a small set of calibration data to avoid full retraining, as it is considered computationally infeasible for LLMs. Reconstructing single matrices in isolation has favorable properties, such as convexity of the objective and significantly reduced memory requirements compared to full retraining. In practice, however, reconstruction is often implemented at coarser granularities, e.g., reconstructing a whole transformer block against its dense activations instead of a single matrix. In this work, we study the key design choices when reconstructing or retraining the remaining weights after pruning. We conduct an extensive computational study on state-of-the-art GPT architectures, and report several surprising findings that challenge common intuitions about retraining after pruning. In particular, we observe a free lunch scenario: reconstructing attention and MLP components separately within each transformer block is nearly the most resource-efficient yet achieves the best perplexity. Most importantly, this Pareto-optimal setup achieves better performance than full retraining, despite requiring only a fraction of the memory. Furthermore, we demonstrate that simple and efficient pruning criteria such as Wanda can outperform much more complex approaches when the reconstruction step is properly executed, highlighting its importance. Our findings challenge the narrative that retraining should be avoided at all costs and provide important insights into post-pruning performance recovery for LLMs. 

---
# Big Data Approaches to Bovine Bioacoustics: A FAIR-Compliant Dataset and Scalable ML Framework for Precision Livestock Welfare 

**Authors**: Mayuri Kate, Suresh Neethirajan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14443)  

**Abstract**: The convergence of IoT sensing, edge computing, and machine learning is transforming precision livestock farming. Yet bioacoustic data streams remain underused because of computational complexity and ecological validity challenges. We present one of the most comprehensive bovine vocalization datasets to date, with 569 curated clips covering 48 behavioral classes, recorded across three commercial dairy farms using multiple microphone arrays and expanded to 2900 samples through domain informed augmentation. This FAIR compliant resource addresses major Big Data challenges - volume (90 hours of recordings, 65.6 GB), variety (multi farm and multi zone acoustics), velocity (real time processing), and veracity (noise robust feature extraction). Our distributed processing framework integrates advanced denoising using iZotope RX, multimodal synchronization through audio and video alignment, and standardized feature engineering with 24 acoustic descriptors generated from Praat, librosa, and openSMILE. Preliminary benchmarks reveal distinct class level acoustic patterns for estrus detection, distress classification, and maternal communication. The datasets ecological realism, reflecting authentic barn acoustics rather than controlled settings, ensures readiness for field deployment. This work establishes a foundation for animal centered AI, where bioacoustic data enable continuous and non invasive welfare assessment at industrial scale. By releasing standardized pipelines and detailed metadata, we promote reproducible research that connects Big Data analytics, sustainable agriculture, and precision livestock management. The framework supports UN SDG 9, showing how data science can turn traditional farming into intelligent, welfare optimized systems that meet global food needs while upholding ethical animal care. 

---
# Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following 

**Authors**: Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14420)  

**Abstract**: Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at this https URL 

---
# The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems 

**Authors**: Prateek Gupta, Qiankun Zhong, Hiromu Yakura, Thomas Eisenmann, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14401)  

**Abstract**: A growing body of multi-agent studies with Large Language Models (LLMs) explores how norms and cooperation emerge in mixed-motive scenarios, where pursuing individual gain can undermine the collective good. While prior work has explored these dynamics in both richly contextualized simulations and simplified game-theoretic environments, most LLM systems featuring common-pool resource (CPR) games provide agents with explicit reward functions directly tied to their actions. In contrast, human cooperation often emerges without full visibility into payoffs and population, relying instead on heuristics, communication, and punishment. We introduce a CPR simulation framework that removes explicit reward signals and embeds cultural-evolutionary mechanisms: social learning (adopting strategies and beliefs from successful peers) and norm-based punishment, grounded in Ostrom's principles of resource governance. Agents also individually learn from the consequences of harvesting, monitoring, and punishing via environmental feedback, enabling norms to emerge endogenously. We establish the validity of our simulation by reproducing key findings from existing studies on human behavior. Building on this, we examine norm evolution across a $2\times2$ grid of environmental and social initialisations (resource-rich vs. resource-scarce; altruistic vs. selfish) and benchmark how agentic societies comprised of different LLMs perform under these conditions. Our results reveal systematic model differences in sustaining cooperation and norm formation, positioning the framework as a rigorous testbed for studying emergent norms in mixed-motive LLM societies. Such analysis can inform the design of AI systems deployed in social and organizational contexts, where alignment with cooperative norms is critical for stability, fairness, and effective governance of AI-mediated environments. 

---
# MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering 

**Authors**: Yingpeng Ning, Yuanyuan Sun, Ling Luo, Yanhua Wang, Yuchen Pan, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14400)  

**Abstract**: Biomedical question answering (QA) requires accurate interpretation of complex medical knowledge. Large language models (LLMs) have shown promising capabilities in this domain, with retrieval-augmented generation (RAG) systems enhancing performance by incorporating external medical literature. However, RAG-based approaches in biomedical QA suffer from hallucinations due to post-retrieval noise and insufficient verification of retrieved evidence, undermining response reliability. We propose MedTrust-Guided Iterative RAG, a framework designed to enhance factual consistency and mitigate hallucinations in medical QA. Our method introduces three key innovations. First, it enforces citation-aware reasoning by requiring all generated content to be explicitly grounded in retrieved medical documents, with structured Negative Knowledge Assertions used when evidence is insufficient. Second, it employs an iterative retrieval-verification process, where a verification agent assesses evidence adequacy and refines queries through Medical Gap Analysis until reliable information is obtained. Third, it integrates the MedTrust-Align Module (MTAM) that combines verified positive examples with hallucination-aware negative samples, leveraging Direct Preference Optimization to reinforce citation-grounded reasoning while penalizing hallucination-prone response patterns. Experiments on MedMCQA, MedQA, and MMLU-Med demonstrate that our approach consistently outperforms competitive baselines across multiple model architectures, achieving the best average accuracy with gains of 2.7% for LLaMA3.1-8B-Instruct and 2.4% for Qwen3-8B. 

---
# FairBatching: Fairness-Aware Batch Formation for LLM Inference 

**Authors**: Hongtao Lyu, Boyue Liu, Mingyu Wu, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.14392)  

**Abstract**: Large language model (LLM) inference systems face a fundamental tension between minimizing Time-to-First-Token (TTFT) latency for new requests and maintaining a high, steady token generation rate (low Time-Per-Output-Token, or TPOT) for ongoing requests. Existing stall-free batching schedulers proposed by Sarathi, while effective at preventing decode stalls, introduce significant computational unfairness. They prioritize decode tasks excessively, simultaneously leading to underutilized decode slack and unnecessary prefill queuing delays, which collectively degrade the system's overall quality of service (QoS).
This work identifies the root cause of this unfairness: the non-monotonic nature of Time-Between-Tokens (TBT) as a scheduling metric and the rigid decode-prioritizing policy that fails to adapt to dynamic workload bursts. We therefore propose FairBatching, a novel LLM inference scheduler that enforces fair resource allocation between prefill and decode tasks. It features an adaptive batch capacity determination mechanism, which dynamically adjusts the computational budget to improve the GPU utilization without triggering SLO violations. Its fair and dynamic batch formation algorithm breaks away from the decode-prioritizing paradigm, allowing computation resources to be reclaimed from bursting decode tasks to serve prefill surges, achieving global fairness. Furthermore, FairBatching provides a novel load estimation method, enabling more effective coordination with upper-level schedulers. Implemented and evaluated on realistic traces, FairBatching significantly reduces TTFT tail latency by up to 2.29x while robustly maintaining TPOT SLOs, achieving overall 20.0% improvement in single-node capacity and 54.3% improvement in cluster-level capacity. 

---
# Beat Detection as Object Detection 

**Authors**: Jaehoon Ahn, Moon-Ryul Jung  

**Link**: [PDF](https://arxiv.org/pdf/2510.14391)  

**Abstract**: Recent beat and downbeat tracking models (e.g., RNNs, TCNs, Transformers) output frame-level activations. We propose reframing this task as object detection, where beats and downbeats are modeled as temporal "objects." Adapting the FCOS detector from computer vision to 1D audio, we replace its original backbone with WaveBeat's temporal feature extractor and add a Feature Pyramid Network to capture multi-scale temporal patterns. The model predicts overlapping beat/downbeat intervals with confidence scores, followed by non-maximum suppression (NMS) to select final predictions. This NMS step serves a similar role to DBNs in traditional trackers, but is simpler and less heuristic. Evaluated on standard music datasets, our approach achieves competitive results, showing that object detection techniques can effectively model musical beats with minimal adaptation. 

---
# Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers 

**Authors**: Andrew Zhao, Reshmi Ghosh, Vitor Carvalho, Emily Lawton, Keegan Hines, Gao Huang, Jack W. Stokes  

**Link**: [PDF](https://arxiv.org/pdf/2510.14381)  

**Abstract**: Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks. 

---
# From Binary to Bilingual: How the National Weather Service is Using Artificial Intelligence to Develop a Comprehensive Translation Program 

**Authors**: Joseph E. Trujillo-Falcon, Monica L. Bozeman, Liam E. Llewellyn, Samuel T. Halvorson, Meryl Mizell, Stuti Deshpande, Bob Manning, Todd Fagin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14369)  

**Abstract**: To advance a Weather-Ready Nation, the National Weather Service (NWS) is developing a systematic translation program to better serve the 68.8 million people in the U.S. who do not speak English at home. This article outlines the foundation of an automated translation tool for NWS products, powered by artificial intelligence. The NWS has partnered with LILT, whose patented training process enables large language models (LLMs) to adapt neural machine translation (NMT) tools for weather terminology and messaging. Designed for scalability across Weather Forecast Offices (WFOs) and National Centers, the system is currently being developed in Spanish, Simplified Chinese, Vietnamese, and other widely spoken non-English languages. Rooted in best practices for multilingual risk communication, the system provides accurate, timely, and culturally relevant translations, significantly reducing manual translation time and easing operational workloads across the NWS. To guide the distribution of these products, GIS mapping was used to identify language needs across different NWS regions, helping prioritize resources for the communities that need them most. We also integrated ethical AI practices throughout the program's design, ensuring that transparency, fairness, and human oversight guide how automated translations are created, evaluated, and shared with the public. This work has culminated into a website featuring experimental multilingual NWS products, including translated warnings, 7-day forecasts, and educational campaigns, bringing the country one step closer to a national warning system that reaches all Americans. 

---
# SUM-AgriVLN: Spatial Understanding Memory for Agricultural Vision-and-Language Navigation 

**Authors**: Xiaobei Zhao, Xingqi Lyu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14357)  

**Abstract**: Agricultural robots are emerging as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or fixed rail systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling robots to navigate to the target positions following the natural language instructions. In practical agricultural scenarios, navigation instructions often repeatedly occur, yet AgriVLN treat each instruction as an independent episode, overlooking the potential of past experiences to provide spatial context for subsequent ones. To bridge this gap, we propose the method of Spatial Understanding Memory for Agricultural Vision-and-Language Navigation (SUM-AgriVLN), in which the SUM module employs spatial understanding and save spatial memory through 3D reconstruction and representation. When evaluated on the A2A benchmark, our SUM-AgriVLN effectively improves Success Rate from 0.47 to 0.54 with slight sacrifice on Navigation Error from 2.91m to 2.93m, demonstrating the state-of-the-art performance in the agricultural domain. Code: this https URL. 

---
# CURE: Confidence-driven Unified Reasoning Ensemble Framework for Medical Question Answering 

**Authors**: Ziad Elshaer, Essam A. Rashed  

**Link**: [PDF](https://arxiv.org/pdf/2510.14353)  

**Abstract**: High-performing medical Large Language Models (LLMs) typically require extensive fine-tuning with substantial computational resources, limiting accessibility for resource-constrained healthcare institutions. This study introduces a confidence-driven multi-model framework that leverages model diversity to enhance medical question answering without fine-tuning. Our framework employs a two-stage architecture: a confidence detection module assesses the primary model's certainty, and an adaptive routing mechanism directs low-confidence queries to Helper models with complementary knowledge for collaborative reasoning. We evaluate our approach using Qwen3-30B-A3B-Instruct, Phi-4 14B, and Gemma 2 12B across three medical benchmarks; MedQA, MedMCQA, and PubMedQA. Result demonstrate that our framework achieves competitive performance, with particularly strong results in PubMedQA (95.0\%) and MedMCQA (78.0\%). Ablation studies confirm that confidence-aware routing combined with multi-model collaboration substantially outperforms single-model approaches and uniform reasoning strategies. This work establishes that strategic model collaboration offers a practical, computationally efficient pathway to improve medical AI systems, with significant implications for democratizing access to advanced medical AI in resource-limited settings. 

---
# Beyond One World: Benchmarking Super Heros in Role-Playing Across Multiversal Contexts 

**Authors**: Perapard Ngokpol, Kun Kerdthaisong, Pasin Buakhaw, Pitikorn Khlaisamniang, Supasate Vorathammathorn, Piyalitt Ittichaiwong, Nutchanon Yongsatianchot  

**Link**: [PDF](https://arxiv.org/pdf/2510.14351)  

**Abstract**: Large language models (LLMs) are increasingly used as role-playing agents, yet their capacity to faithfully and consistently portray version-specific characters -- for example, superheroes across comic and cinematic universes -- remains underexplored. Superhero canons such as Marvel and DC provide a rich testbed: decades of storytelling yield multiple incarnations of the same character with distinct histories, values, and moral codes. To study this problem, we introduce Beyond One World, a benchmark for character-grounded roleplay spanning 30 iconic heroes and 90 canon-specific versions. The benchmark comprises two tasks: (i) Canon Events, which probes factual recall of pivotal life stages, and (ii) Moral Dilemmas, which confronts models with ethically charged scenarios. We score responses for canonical accuracy and reasoning fidelity under a framework that separates internal deliberation ("thinking") from outward decisions ("acting"). We further propose Think-Act Matching, a metric that quantifies alignment between reasons and actions and serves as a proxy for model trustworthiness. Experiments across reasoning- and non-reasoning-oriented models yield three findings: (1) chain-of-thought prompting improves narrative coherence in weaker models but can reduce canonical accuracy in stronger ones; (2) cross-version generalization within a character remains a major obstacle; and (3) models often excel at either thinking or acting, but rarely both. Beyond One World exposes critical gaps in multiversal consistency and reasoning alignment, offering a challenging evaluation for role-playing LLMs. 

---
# BinCtx: Multi-Modal Representation Learning for Robust Android App Behavior Detection 

**Authors**: Zichen Liu, Shao Yang, Xusheng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.14344)  

**Abstract**: Mobile app markets host millions of apps, yet undesired behaviors (e.g., disruptive ads, illegal redirection, payment deception) remain hard to catch because they often do not rely on permission-protected APIs and can be easily camouflaged via UI or metadata edits. We present BINCTX, a learning approach that builds multi-modal representations of an app from (i) a global bytecode-as-image view that captures code-level semantics and family-style patterns, (ii) a contextual view (manifested actions, components, declared permissions, URL/IP constants) indicating how behaviors are triggered, and (iii) a third-party-library usage view summarizing invocation frequencies along inter-component call paths. The three views are embedded and fused to train a contextual-aware classifier. On real-world malware and benign apps, BINCTX attains a macro F1 of 94.73%, outperforming strong baselines by at least 14.92%. It remains robust under commercial obfuscation (F1 84% post-obfuscation) and is more resistant to adversarial samples than state-of-the-art bytecode-only systems. 

---
# A Density-Informed Multimodal Artificial Intelligence Framework for Improving Breast Cancer Detection Across All Breast Densities 

**Authors**: Siva Teja Kakileti, Bharath Govindaraju, Sudhakar Sampangi, Geetha Manjunath  

**Link**: [PDF](https://arxiv.org/pdf/2510.14340)  

**Abstract**: Mammography, the current standard for breast cancer screening, has reduced sensitivity in women with dense breast tissue, contributing to missed or delayed diagnoses. Thermalytix, an AI-based thermal imaging modality, captures functional vascular and metabolic cues that may complement mammographic structural data. This study investigates whether a breast density-informed multi-modal AI framework can improve cancer detection by dynamically selecting the appropriate imaging modality based on breast tissue composition. A total of 324 women underwent both mammography and thermal imaging. Mammography images were analyzed using a multi-view deep learning model, while Thermalytix assessed thermal images through vascular and thermal radiomics. The proposed framework utilized Mammography AI for fatty breasts and Thermalytix AI for dense breasts, optimizing predictions based on tissue type. This multi-modal AI framework achieved a sensitivity of 94.55% (95% CI: 88.54-100) and specificity of 79.93% (95% CI: 75.14-84.71), outperforming standalone mammography AI (sensitivity 81.82%, specificity 86.25%) and Thermalytix AI (sensitivity 92.73%, specificity 75.46%). Importantly, the sensitivity of Mammography dropped significantly in dense breasts (67.86%) versus fatty breasts (96.30%), whereas Thermalytix AI maintained high and consistent sensitivity in both (92.59% and 92.86%, respectively). This demonstrates that a density-informed multi-modal AI framework can overcome key limitations of unimodal screening and deliver high performance across diverse breast compositions. The proposed framework is interpretable, low-cost, and easily deployable, offering a practical path to improving breast cancer screening outcomes in both high-resource and resource-limited settings. 

---
# Stop-RAG: Value-Based Retrieval Control for Iterative RAG 

**Authors**: Jaewan Park, Solbee Cho, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14337)  

**Abstract**: Iterative retrieval-augmented generation (RAG) enables large language models to answer complex multi-hop questions, but each additional loop increases latency, costs, and the risk of introducing distracting evidence, motivating the need for an efficient stopping strategy. Existing methods either use a predetermined number of iterations or rely on confidence proxies that poorly reflect whether more retrieval will actually help. We cast iterative RAG as a finite-horizon Markov decision process and introduce Stop-RAG, a value-based controller that adaptively decides when to stop retrieving. Trained with full-width forward-view Q($\lambda$) targets from complete trajectories, Stop-RAG learns effective stopping policies while remaining compatible with black-box APIs and existing pipelines. On multi-hop question-answering benchmarks, Stop-RAG consistently outperforms both fixed-iteration baselines and prompting-based stopping with LLMs. These results highlight adaptive stopping as a key missing component in current agentic systems, and demonstrate that value-based control can improve the accuracy of RAG systems. 

---
# A Robust Classification Method using Hybrid Word Embedding for Early Diagnosis of Alzheimer's Disease 

**Authors**: Yangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14332)  

**Abstract**: Early detection of Alzheimer's Disease (AD) is greatly beneficial to AD patients, leading to early treatments that lessen symptoms and alleviating financial burden of health care. As one of the leading signs of AD, language capability changes can be used for early diagnosis of AD. In this paper, I develop a robust classification method using hybrid word embedding and fine-tuned hyperparameters to achieve state-of-the-art accuracy in the early detection of AD. Specifically, we create a hybrid word embedding based on word vectors from Doc2Vec and ELMo to obtain perplexity scores of the sentences. The scores identify whether a sentence is fluent or not and capture semantic context of the sentences. I enrich the word embedding by adding linguistic features to analyze syntax and semantics. Further, we input an embedded feature vector into logistic regression and fine tune hyperparameters throughout the pipeline. By tuning hyperparameters of the machine learning pipeline (e.g., model regularization parameter, learning rate and vector size of Doc2Vec, and vector size of ELMo), I achieve 91% classification accuracy and an Area Under the Curve (AUC) of 97% in distinguishing early AD from healthy subjects. Based on my knowledge, my model with 91% accuracy and 97% AUC outperforms the best existing NLP model for AD diagnosis with an accuracy of 88% [32]. I study the model stability through repeated experiments and find that the model is stable even though the training data is split randomly (standard deviation of accuracy = 0.0403; standard deviation of AUC = 0.0174). This affirms our proposed method is accurate and stable. This model can be used as a large-scale screening method for AD, as well as a complementary examination for doctors to detect AD. 

---
# Evaluating & Reducing Deceptive Dialogue From Language Models with Multi-turn RL 

**Authors**: Marwa Abdulhai, Ryan Cheng, Aryansh Shrivastava, Natasha Jaques, Yarin Gal, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2510.14318)  

**Abstract**: Large Language Models (LLMs) interact with millions of people worldwide in applications such as customer support, education and healthcare. However, their ability to produce deceptive outputs, whether intentionally or inadvertently, poses significant safety concerns. The unpredictable nature of LLM behavior, combined with insufficient safeguards against hallucination, misinformation, and user manipulation, makes their misuse a serious, real-world risk. In this paper, we investigate the extent to which LLMs engage in deception within dialogue, and propose the belief misalignment metric to quantify deception. We evaluate deception across four distinct dialogue scenarios, using five established deception detection metrics and our proposed metric. Our findings reveal this novel deception measure correlates more closely with human judgments than any existing metrics we test. Additionally, our benchmarking of eight state-of-the-art models indicates that LLMs naturally exhibit deceptive behavior in approximately 26% of dialogue turns, even when prompted with seemingly benign objectives. When prompted to deceive, LLMs are capable of increasing deceptiveness by as much as 31% relative to baselines. Unexpectedly, models trained with RLHF, the predominant approach for ensuring the safety of widely-deployed LLMs, still exhibit deception at a rate of 43% on average. Given that deception in dialogue is a behavior that develops over an interaction history, its effective evaluation and mitigation necessitates moving beyond single-utterance analyses. We introduce a multi-turn reinforcement learning methodology to fine-tune LLMs to reduce deceptive behaviors, leading to a 77.6% reduction compared to other instruction-tuned models. 

---
# Column Generation Using Domain-Independent Dynamic Programming 

**Authors**: Ryo Kuroiwa, Edward Lam  

**Link**: [PDF](https://arxiv.org/pdf/2510.14317)  

**Abstract**: Column generation and branch-and-price are leading methods for large-scale exact optimization. Column generation iterates between solving a master problem and a pricing problem. The master problem is a linear program, which can be solved using a generic solver. The pricing problem is highly dependent on the application but is usually discrete. Due to the difficulty of discrete optimization, high-performance column generation often relies on a custom pricing algorithm built specifically to exploit the problem's structure. This bespoke nature of the pricing solver prevents the reuse of components for other applications. We show that domain-independent dynamic programming, a software package for modeling and solving arbitrary dynamic programs, can be used as a generic pricing solver. We develop basic implementations of branch-and-price with pricing by domain-independent dynamic programming and show that they outperform a world-leading solver on static mixed integer programming formulations for seven problem classes. 

---
# MERLIN: A Testbed for Multilingual Multimodal Entity Recognition and Linking 

**Authors**: Sathyanarayanan Ramamoorthy, Vishwa Shah, Simran Khanuja, Zaid Sheikh, Shan Jie, Ann Chia, Shearman Chua, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2510.14307)  

**Abstract**: This paper introduces MERLIN, a novel testbed system for the task of Multilingual Multimodal Entity Linking. The created dataset includes BBC news article titles, paired with corresponding images, in five languages: Hindi, Japanese, Indonesian, Vietnamese, and Tamil, featuring over 7,000 named entity mentions linked to 2,500 unique Wikidata entities. We also include several benchmarks using multilingual and multimodal entity linking methods exploring different language models like LLaMa-2 and Aya-23. Our findings indicate that incorporating visual data improves the accuracy of entity linking, especially for entities where the textual context is ambiguous or insufficient, and particularly for models that do not have strong multilingual abilities. For the work, the dataset, methods are available here at this https URL 

---
# Watermarking for Factuality: Guiding Vision-Language Models Toward Truth via Tri-layer Contrastive Decoding 

**Authors**: Kyungryul Back, Seongbeom Park, Milim Kim, Mincheol Kwon, SangHyeok Lee, Hyunyoung Lee, Junhee Cho, Seunghyun Park, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.14304)  

**Abstract**: Large Vision-Language Models (LVLMs) have recently shown promising results on various multimodal tasks, even achieving human-comparable performance in certain cases. Nevertheless, LVLMs remain prone to hallucinations -- they often rely heavily on a single modality or memorize training data without properly grounding their outputs. To address this, we propose a training-free, tri-layer contrastive decoding with watermarking, which proceeds in three steps: (1) select a mature layer and an amateur layer among the decoding layers, (2) identify a pivot layer using a watermark-related question to assess whether the layer is visually well-grounded, and (3) apply tri-layer contrastive decoding to generate the final output. Experiments on public benchmarks such as POPE, MME and AMBER demonstrate that our method achieves state-of-the-art performance in reducing hallucinations in LVLMs and generates more visually grounded responses. 

---
# Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning 

**Authors**: Weijie Shen, Yitian Liu, Yuhao Wu, Zhixuan Liang, Sijia Gu, Dehui Wang, Tian Nian, Lei Xu, Yusen Qin, Jiangmiao Pang, Xinping Guan, Xiaokang Yang, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14300)  

**Abstract**: Vision-Language-Action (VLA) models are experiencing rapid development and demonstrating promising capabilities in robotic manipulation tasks. However, scaling up VLA models presents several critical challenges: (1) Training new VLA models from scratch demands substantial computational resources and extensive datasets. Given the current scarcity of robot data, it becomes particularly valuable to fully leverage well-pretrained VLA model weights during the scaling process. (2) Real-time control requires carefully balancing model capacity with computational efficiency. To address these challenges, We propose AdaMoE, a Mixture-of-Experts (MoE) architecture that inherits pretrained weights from dense VLA models, and scales up the action expert by substituting the feedforward layers into sparsely activated MoE layers. AdaMoE employs a decoupling technique that decouples expert selection from expert weighting through an independent scale adapter working alongside the traditional router. This enables experts to be selected based on task relevance while contributing with independently controlled weights, allowing collaborative expert utilization rather than winner-takes-all dynamics. Our approach demonstrates that expertise need not monopolize. Instead, through collaborative expert utilization, we can achieve superior performance while maintaining computational efficiency. AdaMoE consistently outperforms the baseline model across key benchmarks, delivering performance gains of 1.8% on LIBERO and 9.3% on RoboTwin. Most importantly, a substantial 21.5% improvement in real-world experiments validates its practical effectiveness for robotic manipulation tasks. 

---
# TED++: Submanifold-Aware Backdoor Detection via Layerwise Tubular-Neighbourhood Screening 

**Authors**: Nam Le, Leo Yu Zhang, Kewen Liao, Shirui Pan, Wei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.14299)  

**Abstract**: As deep neural networks power increasingly critical applications, stealthy backdoor attacks, where poisoned training inputs trigger malicious model behaviour while appearing benign, pose a severe security risk. Many existing defences are vulnerable when attackers exploit subtle distance-based anomalies or when clean examples are scarce. To meet this challenge, we introduce TED++, a submanifold-aware framework that effectively detects subtle backdoors that evade existing defences. TED++ begins by constructing a tubular neighbourhood around each class's hidden-feature manifold, estimating its local ``thickness'' from a handful of clean activations. It then applies Locally Adaptive Ranking (LAR) to detect any activation that drifts outside the admissible tube. By aggregating these LAR-adjusted ranks across all layers, TED++ captures how faithfully an input remains on the evolving class submanifolds. Based on such characteristic ``tube-constrained'' behaviour, TED++ flags inputs whose LAR-based ranking sequences deviate significantly. Extensive experiments are conducted on benchmark datasets and tasks, demonstrating that TED++ achieves state-of-the-art detection performance under both adaptive-attack and limited-data scenarios. Remarkably, even with only five held-out examples per class, TED++ still delivers near-perfect detection, achieving gains of up to 14\% in AUROC over the next-best method. The code is publicly available at this https URL. 

---
# Learning Human-Humanoid Coordination for Collaborative Object Carrying 

**Authors**: Yushi Du, Yixuan Li, Baoxiong Jia, Yutang Lin, Pei Zhou, Wei Liang, Yanchao Yang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14293)  

**Abstract**: Human-humanoid collaboration shows significant promise for applications in healthcare, domestic assistance, and manufacturing. While compliant robot-human collaboration has been extensively developed for robotic arms, enabling compliant human-humanoid collaboration remains largely unexplored due to humanoids' complex whole-body dynamics. In this paper, we propose a proprioception-only reinforcement learning approach, COLA, that combines leader and follower behaviors within a single policy. The model is trained in a closed-loop environment with dynamic object interactions to predict object motion patterns and human intentions implicitly, enabling compliant collaboration to maintain load balance through coordinated trajectory planning. We evaluate our approach through comprehensive simulator and real-world experiments on collaborative carrying tasks, demonstrating the effectiveness, generalization, and robustness of our model across various terrains and objects. Simulation experiments demonstrate that our model reduces human effort by 24.7%. compared to baseline approaches while maintaining object stability. Real-world experiments validate robust collaborative carrying across different object types (boxes, desks, stretchers, etc.) and movement patterns (straight-line, turning, slope climbing). Human user studies with 23 participants confirm an average improvement of 27.4% compared to baseline models. Our method enables compliant human-humanoid collaborative carrying without requiring external sensors or complex interaction models, offering a practical solution for real-world deployment. 

---
# Beyond a Single Perspective: Towards a Realistic Evaluation of Website Fingerprinting Attacks 

**Authors**: Xinhao Deng, Jingyou Chen, Linxiao Yu, Yixiang Zhang, Zhongyi Gu, Changhao Qiu, Xiyuan Zhao, Ke Xu, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14283)  

**Abstract**: Website Fingerprinting (WF) attacks exploit patterns in encrypted traffic to infer the websites visited by users, posing a serious threat to anonymous communication systems. Although recent WF techniques achieve over 90% accuracy in controlled experimental settings, most studies remain confined to single scenarios, overlooking the complexity of real-world environments. This paper presents the first systematic and comprehensive evaluation of existing WF attacks under diverse realistic conditions, including defense mechanisms, traffic drift, multi-tab browsing, early-stage detection, open-world settings, and few-shot scenarios. Experimental results show that many WF techniques with strong performance in isolated settings degrade significantly when facing other conditions. Since real-world environments often combine multiple challenges, current WF attacks are difficult to apply directly in practice. This study highlights the limitations of WF attacks and introduces a multidimensional evaluation framework, offering critical insights for developing more robust and practical WF attacks. 

---
# PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering 

**Authors**: Md Mahadi Hasan Nahid, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2510.14278)  

**Abstract**: Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks -- HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG -- demonstrates that our approach consistently outperforms strong baselines. 

---
# Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation 

**Authors**: Yilun Zheng, Dan Yang, Jie Li, Lin Shang, Lihui Chen, Jiahao Xu, Sitao Luan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14271)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enable large language models (LLMs) instant access to relevant information for the generative process, demonstrating their superior performance in addressing common LLM challenges such as hallucination, factual inaccuracy, and the knowledge cutoff. Graph-based RAG further extends this paradigm by incorporating knowledge graphs (KGs) to leverage rich, structured connections for more precise and inferential responses. A critical challenge, however, is that most Graph-based RAG systems rely on LLMs for automated KG construction, often yielding noisy KGs with redundant entities and unreliable relationships. This noise degrades retrieval and generation performance while also increasing computational cost. Crucially, current research does not comprehensively address the denoising problem for LLM-generated KGs. In this paper, we introduce DEnoised knowledge Graphs for Retrieval Augmented Generation (DEG-RAG), a framework that addresses these challenges through: (1) entity resolution, which eliminates redundant entities, and (2) triple reflection, which removes erroneous relations. Together, these techniques yield more compact, higher-quality KGs that significantly outperform their unprocessed counterparts. Beyond the methods, we conduct a systematic evaluation of entity resolution for LLM-generated KGs, examining different blocking strategies, embedding choices, similarity metrics, and entity merging techniques. To the best of our knowledge, this is the first comprehensive exploration of entity resolution in LLM-generated KGs. Our experiments demonstrate that this straightforward approach not only drastically reduces graph size but also consistently improves question answering performance across diverse popular Graph-based RAG variants. 

---
# CAST: Compositional Analysis via Spectral Tracking for Understanding Transformer Layer Functions 

**Authors**: Zihao Fu, Ming Liao, Chris Russell, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.14262)  

**Abstract**: Large language models have achieved remarkable success but remain largely black boxes with poorly understood internal mechanisms. To address this limitation, many researchers have proposed various interpretability methods including mechanistic analysis, probing classifiers, and activation visualization, each providing valuable insights from different perspectives. Building upon this rich landscape of complementary approaches, we introduce CAST (Compositional Analysis via Spectral Tracking), a probe-free framework that contributes a novel perspective by analyzing transformer layer functions through direct transformation matrix estimation and comprehensive spectral analysis. CAST offers complementary insights to existing methods by estimating the realized transformation matrices for each layer using Moore-Penrose pseudoinverse and applying spectral analysis with six interpretable metrics characterizing layer behavior. Our analysis reveals distinct behaviors between encoder-only and decoder-only models, with decoder models exhibiting compression-expansion cycles while encoder models maintain consistent high-rank processing. Kernel analysis further demonstrates functional relationship patterns between layers, with CKA similarity matrices clearly partitioning layers into three phases: feature extraction, compression, and specialization. 

---
# Do Joint Language-Audio Embeddings Encode Perceptual Timbre Semantics? 

**Authors**: Qixin Deng, Bryan Pardo, Thrasyvoulos N Pappas  

**Link**: [PDF](https://arxiv.org/pdf/2510.14249)  

**Abstract**: Understanding and modeling the relationship between language and sound is critical for applications such as music information retrieval,text-guided music generation, and audio captioning. Central to these tasks is the use of joint language-audio embedding spaces, which map textual descriptions and auditory content into a shared embedding space. While multimodal embedding models such as MS-CLAP, LAION-CLAP, and MuQ-MuLan have shown strong performance in aligning language and audio, their correspondence to human perception of timbre, a multifaceted attribute encompassing qualities such as brightness, roughness, and warmth, remains underexplored. In this paper, we evaluate the above three joint language-audio embedding models on their ability to capture perceptual dimensions of timbre. Our findings show that LAION-CLAP consistently provides the most reliable alignment with human-perceived timbre semantics across both instrumental sounds and audio effects. 

---
# Policy Regularized Distributionally Robust Markov Decision Processes with Linear Function Approximation 

**Authors**: Jingwen Gu, Yiting He, Zhishuai Liu, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14246)  

**Abstract**: Decision-making under distribution shift is a central challenge in reinforcement learning (RL), where training and deployment environments differ. We study this problem through the lens of robust Markov decision processes (RMDPs), which optimize performance against adversarial transition dynamics. Our focus is the online setting, where the agent has only limited interaction with the environment, making sample efficiency and exploration especially critical. Policy optimization, despite its success in standard RL, remains theoretically and empirically underexplored in robust RL. To bridge this gap, we propose \textbf{D}istributionally \textbf{R}obust \textbf{R}egularized \textbf{P}olicy \textbf{O}ptimization algorithm (DR-RPO), a model-free online policy optimization method that learns robust policies with sublinear regret. To enable tractable optimization within the softmax policy class, DR-RPO incorporates reference-policy regularization, yielding RMDP variants that are doubly constrained in both transitions and policies. To scale to large state-action spaces, we adopt the $d$-rectangular linear MDP formulation and combine linear function approximation with an upper confidence bonus for optimistic exploration. We provide theoretical guarantees showing that policy optimization can achieve polynomial suboptimality bounds and sample efficiency in robust RL, matching the performance of value-based approaches. Finally, empirical results across diverse domains corroborate our theory and demonstrate the robustness of DR-RPO. 

---
# Reinforcement Learning for Unsupervised Domain Adaptation in Spatio-Temporal Echocardiography Segmentation 

**Authors**: Arnaud Judge, Nicolas Duchateau, Thierry Judge, Roman A. Sandler, Joseph Z. Sokol, Christian Desrosiers, Olivier Bernard, Pierre-Marc Jodoin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14244)  

**Abstract**: Domain adaptation methods aim to bridge the gap between datasets by enabling knowledge transfer across domains, reducing the need for additional expert annotations. However, many approaches struggle with reliability in the target domain, an issue particularly critical in medical image segmentation, where accuracy and anatomical validity are essential. This challenge is further exacerbated in spatio-temporal data, where the lack of temporal consistency can significantly degrade segmentation quality, and particularly in echocardiography, where the presence of artifacts and noise can further hinder segmentation performance. To address these issues, we present RL4Seg3D, an unsupervised domain adaptation framework for 2D + time echocardiography segmentation. RL4Seg3D integrates novel reward functions and a fusion scheme to enhance key landmark precision in its segmentations while processing full-sized input videos. By leveraging reinforcement learning for image segmentation, our approach improves accuracy, anatomical validity, and temporal consistency while also providing, as a beneficial side effect, a robust uncertainty estimator, which can be used at test time to further enhance segmentation performance. We demonstrate the effectiveness of our framework on over 30,000 echocardiographic videos, showing that it outperforms standard domain adaptation techniques without the need for any labels on the target domain. Code is available at this https URL. 

---
# Spatial Computing Communications for Multi-User Virtual Reality in Distributed Mobile Edge Computing Network 

**Authors**: Caolu Xu, Zhiyong Chen, Meixia Tao, Li Song, Wenjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14243)  

**Abstract**: Immersive virtual reality (VR) applications impose stringent requirements on latency, energy efficiency, and computational resources, particularly in multi-user interactive scenarios. To address these challenges, we introduce the concept of spatial computing communications (SCC), a framework designed to meet the latency and energy demands of multi-user VR over distributed mobile edge computing (MEC) networks. SCC jointly represents the physical space, defined by users and base stations, and the virtual space, representing shared immersive environments, using a probabilistic model of user dynamics and resource requirements. The resource deployment task is then formulated as a multi-objective combinatorial optimization (MOCO) problem that simultaneously minimizes system latency and energy consumption across distributed MEC resources. To solve this problem, we propose MO-CMPO, a multi-objective consistency model with policy optimization that integrates supervised learning and reinforcement learning (RL) fine-tuning guided by preference weights. Leveraging a sparse graph neural network (GNN), MO-CMPO efficiently generates Pareto-optimal solutions. Simulations with real-world New Radio base station datasets demonstrate that MO-CMPO achieves superior hypervolume performance and significantly lower inference latency than baseline methods. Furthermore, the analysis reveals practical deployment patterns: latency-oriented solutions favor local MEC execution to reduce transmission delay, while energy-oriented solutions minimize redundant placements to save energy. 

---
# Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models 

**Authors**: Mehrzad Samadi, Aleksander Ficek, Sean Narenthiran, Siddhartha Jain, Wasi Uddin Ahmad, Somshubra Majumdar, Vahid Noroozi, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2510.14232)  

**Abstract**: Competitive programming has become a rigorous benchmark for evaluating the reasoning and problem-solving capabilities of large language models (LLMs). The International Olympiad in Informatics (IOI) stands out as one of the most prestigious annual competitions in competitive programming and has become a key benchmark for comparing human and AI-level programming ability. While several proprietary models have been claimed to achieve gold medal-level performance at the IOI, often with undisclosed methods, achieving comparable results with open-weight models remains a significant challenge. In this paper, we present \gencluster, a scalable and reproducible test-time compute framework that attains IOI gold-level performance using open-weight models. It combines large-scale generation, behavioral clustering, ranking, and a round-robin submission strategy to efficiently explore diverse solution spaces under limited validation budgets. Our experiments show that the performance of our proposed approach scales consistently with available compute, narrowing the gap between open and closed systems. Notably, we will show that GenCluster can achieve a gold medal at IOI 2025 for the first time with an open-weight model gpt-oss-120b, setting a new benchmark for transparent and reproducible evaluation of reasoning in LLMs. 

---
# Large Scale Retrieval for the LinkedIn Feed using Causal Language Models 

**Authors**: Sudarshan Srinivasa Ramanujam, Antonio Alonso, Saurabh Kataria, Siddharth Dangi, Akhilesh Gupta, Birjodh Singh Tiwana, Manas Somaiya, Luke Simon, David Byrne, Sojeong Ha, Sen Zhou, Andrei Akterskii, Zhanglong Liu, Samira Sriram, Crescent Xiong, Zhoutao Pei, Angela Shao, Alex Li, Annie Xiao, Caitlin Kolb, Thomas Kistler, Zach Moore, Hamed Firooz  

**Link**: [PDF](https://arxiv.org/pdf/2510.14223)  

**Abstract**: In large scale recommendation systems like the LinkedIn Feed, the retrieval stage is critical for narrowing hundreds of millions of potential candidates to a manageable subset for ranking. LinkedIn's Feed serves suggested content from outside of the member's network (based on the member's topical interests), where 2000 candidates are retrieved from a pool of hundreds of millions candidate with a latency budget of a few milliseconds and inbound QPS of several thousand per second. This paper presents a novel retrieval approach that fine-tunes a large causal language model (Meta's LLaMA 3) as a dual encoder to generate high quality embeddings for both users (members) and content (items), using only textual input. We describe the end to end pipeline, including prompt design for embedding generation, techniques for fine-tuning at LinkedIn's scale, and infrastructure for low latency, cost effective online serving. We share our findings on how quantizing numerical features in the prompt enables the information to get properly encoded in the embedding, facilitating greater alignment between the retrieval and ranking layer. The system was evaluated using offline metrics and an online A/B test, which showed substantial improvements in member engagement. We observed significant gains among newer members, who often lack strong network connections, indicating that high-quality suggested content aids retention. This work demonstrates how generative language models can be effectively adapted for real time, high throughput retrieval in industrial applications. 

---
# LiteStage: Latency-aware Layer Skipping for Multi-stage Reasoning 

**Authors**: Beomseok Kang, Jiwon Song, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.14211)  

**Abstract**: Multi-stage reasoning has emerged as an effective strategy for enhancing the reasoning capability of small language models by decomposing complex problems into sequential sub-stages. However, this comes at the cost of increased latency. We observe that existing adaptive acceleration techniques, such as layer skipping, struggle to balance efficiency and accuracy in this setting due to two key challenges: (1) stage-wise variation in skip sensitivity, and (2) the generation of redundant output tokens. To address these, we propose LiteStage, a latency-aware layer skipping framework for multi-stage reasoning. LiteStage combines a stage-wise offline search that allocates optimal layer budgets with an online confidence-based generation early exit to suppress unnecessary decoding. Experiments on three benchmarks, e.g., OBQA, CSQA, and StrategyQA, show that LiteStage achieves up to 1.70x speedup with less than 4.0% accuracy loss, outperforming prior training-free layer skipping methods. 

---
# DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans 

**Authors**: Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14205)  

**Abstract**: The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these this http URL evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie this http URL can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and this http URL work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI. 

---
# MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation with Configurable Task Adaptation 

**Authors**: Mahmood Hegazy, Aaron Rodrigues, Azzam Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2510.14184)  

**Abstract**: We present MAFA (Multi-Agent Framework for Annotation), a production-deployed system that transforms enterprise-scale annotation workflows through configurable multi-agent collaboration. Addressing the critical challenge of annotation backlogs in financial services, where millions of customer utterances require accurate categorization, MAFA combines specialized agents with structured reasoning and a judge-based consensus mechanism. Our framework uniquely supports dynamic task adaptation, allowing organizations to define custom annotation types (FAQs, intents, entities, or domain-specific categories) through configuration rather than code changes. Deployed at JP Morgan Chase, MAFA has eliminated a 1 million utterance backlog while achieving, on average, 86% agreement with human annotators, annually saving over 5,000 hours of manual annotation work. The system processes utterances with annotation confidence classifications, which are typically 85% high, 10% medium, and 5% low across all datasets we tested. This enables human annotators to focus exclusively on ambiguous and low-coverage cases. We demonstrate MAFA's effectiveness across multiple datasets and languages, showing consistent improvements over traditional and single-agent annotation baselines: 13.8% higher Top-1 accuracy, 15.1% improvement in Top-5 accuracy, and 16.9% better F1 in our internal intent classification dataset and similar gains on public benchmarks. This work bridges the gap between theoretical multi-agent systems and practical enterprise deployment, providing a blueprint for organizations facing similar annotation challenges. 

---
# Virtually Being: Customizing Camera-Controllable Video Diffusion Models with Multi-View Performance Captures 

**Authors**: Yuancheng Xu, Wenqi Xian, Li Ma, Julien Philip, Ahmet Levent Taşel, Yiwei Zhao, Ryan Burgert, Mingming He, Oliver Hermann, Oliver Pilarski, Rahul Garg, Paul Debevec, Ning Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14179)  

**Abstract**: We introduce a framework that enables both multi-view character consistency and 3D camera control in video diffusion models through a novel customization data pipeline. We train the character consistency component with recorded volumetric capture performances re-rendered with diverse camera trajectories via 4D Gaussian Splatting (4DGS), lighting variability obtained with a video relighting model. We fine-tune state-of-the-art open-source video diffusion models on this data to provide strong multi-view identity preservation, precise camera control, and lighting adaptability. Our framework also supports core capabilities for virtual production, including multi-subject generation using two approaches: joint training and noise blending, the latter enabling efficient composition of independently customized models at inference time; it also achieves scene and real-life video customization as well as control over motion and spatial layout during customization. Extensive experiments show improved video quality, higher personalization accuracy, and enhanced camera control and lighting adaptability, advancing the integration of video generation into virtual production. Our project page is available at: this https URL. 

---
# Towards Reversible Model Merging For Low-rank Weights 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.14163)  

**Abstract**: Model merging aims to combine multiple fine-tuned models into a single set of weights that performs well across all source tasks. While prior work has shown that merging can approximate the performance of individual fine-tuned models for each task, it largely overlooks scenarios where models are compressed into low-rank representations, either through low-rank adaptation (LoRA) or post-training singular value decomposition (SVD). We first demonstrate that applying conventional merging methods to low-rank weights leads to severe performance degradation in the merged model. Motivated by this phenomenon, we propose a fundamentally different approach: instead of collapsing all adapters into one set of weights, we construct a compact basis (e.g., an equivalent of holding two or more models) from which original task-specific models can be recovered via linear combination. This reframes merging as generating a reconstruction-capable model space rather than producing a single merged model. Crucially, this allows us to ``revert'' to each individual model when needed, recognizing that no merged model can consistently outperform one specialized for its task. Building on this insight, we introduce our method, Reversible Model Merging (RMM), an efficient, data-free, and flexible method that provides a closed-form solution for selecting the optimal basis of model weights and task-specific coefficients for linear combination. Extensive experiments across diverse datasets and model scales demonstrate that RMM consistently outperforms existing merging approaches, preserving the performance of low-rank compressed models by a significant margin. 

---
# FinAI Data Assistant: LLM-based Financial Database Query Processing with the OpenAI Function Calling API 

**Authors**: Juhyeong Kim, Yejin Kim, Youngbin Lee, Hyunwoo Byun  

**Link**: [PDF](https://arxiv.org/pdf/2510.14162)  

**Abstract**: We present FinAI Data Assistant, a practical approach for natural-language querying over financial databases that combines large language models (LLMs) with the OpenAI Function Calling API. Rather than synthesizing complete SQL via text-to-SQL, our system routes user requests to a small library of vetted, parameterized queries, trading generative flexibility for reliability, low latency, and cost efficiency. We empirically study three questions: (RQ1) whether LLMs alone can reliably recall or extrapolate time-dependent financial data without external retrieval; (RQ2) how well LLMs map company names to stock ticker symbols; and (RQ3) whether function calling outperforms text-to-SQL for end-to-end database query processing. Across controlled experiments on prices and fundamentals, LLM-only predictions exhibit non-negligible error and show look-ahead bias primarily for stock prices relative to model knowledge cutoffs. Ticker-mapping accuracy is near-perfect for NASDAQ-100 constituents and high for S\&P~500 firms. Finally, FinAI Data Assistant achieves lower latency and cost and higher reliability than a text-to-SQL baseline on our task suite. We discuss design trade-offs, limitations, and avenues for deployment. 

---
# Inferred global dense residue transition graphs from primary structure sequences enable protein interaction prediction via directed graph convolutional neural networks 

**Authors**: Islam Akef Ebeid, Haoteng Tang, Pengfei Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14139)  

**Abstract**: Introduction Accurate prediction of protein-protein interactions (PPIs) is crucial for understanding cellular functions and advancing drug development. Existing in-silico methods use direct sequence embeddings from Protein Language Models (PLMs). Others use Graph Neural Networks (GNNs) for 3D protein structures. This study explores less computationally intensive alternatives. We introduce a novel framework for downstream PPI prediction through link prediction. Methods We introduce a two-stage graph representation learning framework, ProtGram-DirectGCN. First, we developed ProtGram. This approach models a protein's primary structure as a hierarchy of globally inferred n-gram graphs. In these graphs, residue transition probabilities define edge weights. Each edge connects a pair of residues in a directed graph. The probabilities are aggregated from a large corpus of sequences. Second, we propose DirectGCN, a custom directed graph convolutional neural network. This model features a unique convolutional layer. It processes information through separate path-specific transformations: incoming, outgoing, and undirected. A shared transformation is also applied. These paths are combined via a learnable gating mechanism. We apply DirectGCN to ProtGram graphs to learn residue-level embeddings. These embeddings are pooled via attention to generate protein-level embeddings for prediction. Results We first established the efficacy of DirectGCN on standard node classification benchmarks. Its performance matches established methods on general datasets. The model excels at complex, directed graphs with dense, heterophilic structures. When applied to PPI prediction, the full ProtGram-DirectGCN framework delivers robust predictive power. This strong performance holds even with limited training data. 

---
# Toward Cybersecurity-Expert Small Language Models 

**Authors**: Matan Levi, Daniel Ohayon, Ariel Blobstein, Ravid Sagi, Ian Molloy, Yair Allouche  

**Link**: [PDF](https://arxiv.org/pdf/2510.14113)  

**Abstract**: Large language models (LLMs) are transforming everyday applications, yet deployment in cybersecurity lags due to a lack of high-quality, domain-specific models and training datasets. To address this gap, we present CyberPal 2.0, a family of cybersecurity-expert small language models (SLMs) ranging from 4B-20B parameters. To train CyberPal 2.0, we generate an enriched chain-of-thought cybersecurity instruction dataset built with our data enrichment and formatting pipeline, SecKnowledge 2.0, which integrates expert-in-the-loop steering of reasoning formats alongside LLM-driven multi-step grounding, yielding higher-fidelity, task-grounded reasoning traces for security tasks. Across diverse cybersecurity benchmarks, CyberPal 2.0 consistently outperforms its baselines and matches or surpasses various open and closed-source frontier models, while remaining a fraction of their size. On core cyber threat intelligence knowledge tasks, our models outperform almost all tested frontier models, ranking second only to Sec-Gemini v1. On core threat-investigation tasks, such as correlating vulnerabilities and bug tickets with weaknesses, our best 20B-parameter model outperforms GPT-4o, o1, o3-mini, and Sec-Gemini v1, ranking first, while our smallest 4B-parameter model ranks second. 

---
# Extracting latent representations from X-ray spectra. Classification, regression, and accretion signatures of Chandra sources 

**Authors**: Nicolò Oreste Pinciroli Vago, Juan Rafael Martínez-Galarza, Roberta Amato  

**Link**: [PDF](https://arxiv.org/pdf/2510.14102)  

**Abstract**: The study of X-ray spectra is crucial to understanding the physical nature of astrophysical sources. Machine learning methods can extract compact and informative representations of data from large datasets. The Chandra Source Catalog (CSC) provides a rich archive of X-ray spectral data, which remains largely underexplored in this context. This work aims to develop a compact and physically meaningful representation of Chandra X-ray spectra using deep learning. To verify that the learned representation captures relevant information, we evaluate it through classification, regression, and interpretability analyses. We use a transformer-based autoencoder to compress X-ray spectra. The input spectra, drawn from the CSC, include only high-significance detections. Astrophysical source types and physical summary statistics are compiled from external catalogs. We evaluate the learned representation in terms of spectral reconstruction accuracy, clustering performance on 8 known astrophysical source classes, and correlation with physical quantities such as hardness ratios and hydrogen column density ($N_H$). The autoencoder accurately reconstructs spectra with 8 latent variables. Clustering in the latent space yields a balanced classification accuracy of $\sim$40% across the 8 source classes, increasing to $\sim$69% when restricted to AGNs and stellar-mass compact objects exclusively. Moreover, latent features correlate with non-linear combinations of spectral fluxes, suggesting that the compressed representation encodes physically relevant information. The proposed autoencoder-based pipeline is a powerful tool for the representation and interpretation of X-ray spectra, providing a compact latent space that supports both classification and the estimation of physical properties. This work demonstrates the potential of deep learning for spectral studies and uncovering new patterns in X-ray data. 

---
# Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning 

**Authors**: Awni Altabaa, Siyu Chen, John Lafferty, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.14095)  

**Abstract**: Systematic, compositional generalization beyond the training distribution remains a core challenge in machine learning -- and a critical bottleneck for the emergent reasoning abilities of modern language models. This work investigates out-of-distribution (OOD) generalization in Transformer networks using a GSM8K-style modular arithmetic on computational graphs task as a testbed. We introduce and explore a set of four architectural mechanisms aimed at enhancing OOD generalization: (i) input-adaptive recurrence; (ii) algorithmic supervision; (iii) anchored latent representations via a discrete bottleneck; and (iv) an explicit error-correction mechanism. Collectively, these mechanisms yield an architectural approach for native and scalable latent space reasoning in Transformer networks with robust algorithmic generalization capabilities. We complement these empirical results with a detailed mechanistic interpretability analysis that reveals how these mechanisms give rise to robust OOD generalization abilities. 

---
# Every Language Model Has a Forgery-Resistant Signature 

**Authors**: Matthew Finlayson, Xiang Ren, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2510.14086)  

**Abstract**: The ubiquity of closed-weight language models with public-facing APIs has generated interest in forensic methods, both for extracting hidden model details (e.g., parameters) and for identifying models by their outputs. One successful approach to these goals has been to exploit the geometric constraints imposed by the language model architecture and parameters. In this work, we show that a lesser-known geometric constraint--namely, that language model outputs lie on the surface of a high-dimensional ellipse--functions as a signature for the model and can be used to identify the source model of a given output. This ellipse signature has unique properties that distinguish it from existing model-output association methods like language model fingerprints. In particular, the signature is hard to forge: without direct access to model parameters, it is practically infeasible to produce log-probabilities (logprobs) on the ellipse. Secondly, the signature is naturally occurring, since all language models have these elliptical constraints. Thirdly, the signature is self-contained, in that it is detectable without access to the model inputs or the full weights. Finally, the signature is compact and redundant, as it is independently detectable in each logprob output from the model. We evaluate a novel technique for extracting the ellipse from small models and discuss the practical hurdles that make it infeasible for production-scale models. Finally, we use ellipse signatures to propose a protocol for language model output verification, analogous to cryptographic symmetric-key message authentication systems. 

---
# DiffOPF: Diffusion Solver for Optimal Power Flow 

**Authors**: Milad Hoseinpour, Vladimir Dvorkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14075)  

**Abstract**: The optimal power flow (OPF) is a multi-valued, non-convex mapping from loads to dispatch setpoints. The variability of system parameters (e.g., admittances, topology) further contributes to the multiplicity of dispatch setpoints for a given load. Existing deep learning OPF solvers are single-valued and thus fail to capture the variability of system parameters unless fully represented in the feature space, which is prohibitive. To solve this problem, we introduce a diffusion-based OPF solver, termed \textit{DiffOPF}, that treats OPF as a conditional sampling problem. The solver learns the joint distribution of loads and dispatch setpoints from operational history, and returns the marginal dispatch distributions conditioned on loads. Unlike single-valued solvers, DiffOPF enables sampling statistically credible warm starts with favorable cost and constraint satisfaction trade-offs. We explore the sample complexity of DiffOPF to ensure the OPF solution within a prescribed distance from the optimization-based solution, and verify this experimentally on power system benchmarks. 

---
# Exploratory Causal Inference in SAEnce 

**Authors**: Tommaso Mencattini, Riccardo Cadei, Francesco Locatello  

**Link**: [PDF](https://arxiv.org/pdf/2510.14073)  

**Abstract**: Randomized Controlled Trials are one of the pillars of science; nevertheless, they rely on hand-crafted hypotheses and expensive analysis. Such constraints prevent causal effect estimation at scale, potentially anchoring on popular yet incomplete hypotheses. We propose to discover the unknown effects of a treatment directly from data. For this, we turn unstructured data from a trial into meaningful representations via pretrained foundation models and interpret them via a sparse autoencoder. However, discovering significant causal effects at the neural level is not trivial due to multiple-testing issues and effects entanglement. To address these challenges, we introduce Neural Effect Search, a novel recursive procedure solving both issues by progressive stratification. After assessing the robustness of our algorithm on semi-synthetic experiments, we showcase, in the context of experimental ecology, the first successful unsupervised causal effect identification on a real-world scientific trial. 

---
# On the expressivity of sparse maxout networks 

**Authors**: Moritz Grillo, Tobias Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.14068)  

**Abstract**: We study the expressivity of sparse maxout networks, where each neuron takes a fixed number of inputs from the previous layer and employs a, possibly multi-argument, maxout activation. This setting captures key characteristics of convolutional or graph neural networks. We establish a duality between functions computable by such networks and a class of virtual polytopes, linking their geometry to questions of network expressivity. In particular, we derive a tight bound on the dimension of the associated polytopes, which serves as the central tool for our analysis. Building on this, we construct a sequence of depth hierarchies. While sufficiently deep sparse maxout networks are universal, we prove that if the required depth is not reached, width alone cannot compensate for the sparsity of a fixed indegree constraint. 

---
# Optical Computation-in-Communication enables low-latency, high-fidelity perception in telesurgery 

**Authors**: Rui Yang, Jiaming Hu, Jian-Qing Zheng, Yue-Zhen Lu, Jian-Wei Cui, Qun Ren, Yi-Jie Yu, John Edward Wu, Zhao-Yu Wang, Xiao-Li Lin, Dandan Zhang, Mingchu Tang, Christos Masouros, Huiyun Liu, Chin-Pang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14058)  

**Abstract**: Artificial intelligence (AI) holds significant promise for enhancing intraoperative perception and decision-making in telesurgery, where physical separation impairs sensory feedback and control. Despite advances in medical AI and surgical robotics, conventional electronic AI architectures remain fundamentally constrained by the compounded latency from serial processing of inference and communication. This limitation is especially critical in latency-sensitive procedures such as endovascular interventions, where delays over 200 ms can compromise real-time AI reliability and patient safety. Here, we introduce an Optical Computation-in-Communication (OCiC) framework that reduces end-to-end latency significantly by performing AI inference concurrently with optical communication. OCiC integrates Optical Remote Computing Units (ORCUs) directly into the optical communication pathway, with each ORCU experimentally achieving up to 69 tera-operations per second per channel through spectrally efficient two-dimensional photonic convolution. The system maintains ultrahigh inference fidelity within 0.1% of CPU/GPU baselines on classification and coronary angiography segmentation, while intrinsically mitigating cumulative error propagation, a longstanding barrier to deep optical network scalability. We validated the robustness of OCiC through outdoor dark fibre deployments, confirming consistent and stable performance across varying environmental conditions. When scaled globally, OCiC transforms long-haul fibre infrastructure into a distributed photonic AI fabric with exascale potential, enabling reliable, low-latency telesurgery across distances up to 10,000 km and opening a new optical frontier for distributed medical intelligence. 

---
# Cyber-Resilient System Identification for Power Grid through Bayesian Integration 

**Authors**: Shimiao Li, Guannan Qu, Bryan Hooi, Vyas Sekar, Soummya Kar, Larry Pileggi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14043)  

**Abstract**: Power grids increasingly need real-time situational awareness under the ever-evolving cyberthreat landscape. Advances in snapshot-based system identification approaches have enabled accurately estimating states and topology from a snapshot of measurement data, under random bad data and topology errors. However, modern interactive, targeted false data can stay undetectable to these methods, and significantly compromise estimation accuracy. This work advances system identification that combines snapshot-based method with time-series model via Bayesian Integration, to advance cyber resiliency against both random and targeted false data. Using a distance-based time-series model, this work can leverage historical data of different distributions induced by changes in grid topology and other settings. The normal system behavior captured from historical data is integrated into system identification through a Bayesian treatment, to make solutions robust to targeted false data. We experiment on mixed random anomalies (bad data, topology error) and targeted false data injection attack (FDIA) to demonstrate our method's 1) cyber resilience: achieving over 70% reduction in estimation error under FDIA; 2) anomalous data identification: being able to alarm and locate anomalous data; 3) almost linear scalability: achieving comparable speed with the snapshot-based baseline, both taking <1min per time tick on the large 2,383-bus system using a laptop CPU. 

---
# One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery 

**Authors**: Qiushi Wu, Yue Xiao, Dhilung Kirat, Kevin Eykholt, Jiyong Jang, Douglas Lee Schales  

**Link**: [PDF](https://arxiv.org/pdf/2510.14036)  

**Abstract**: Fixing bugs in large programs is a challenging task that demands substantial time and effort. Once a bug is found, it is reported to the project maintainers, who work with the reporter to fix it and eventually close the issue. However, across the program, there are often similar code segments, which may also contain the bug, but were missed during discovery. Finding and fixing each recurring bug instance individually is labor intensive. Even more concerning, bug reports can inadvertently widen the attack surface as they provide attackers with an exploitable pattern that may be unresolved in other parts of the program.
In this paper, we explore these Recurring Pattern Bugs (RPBs) that appear repeatedly across various code segments of a program or even in different programs, stemming from a same root cause, but are unresolved. Our investigation reveals that RPBs are widespread and can significantly compromise the security of software programs. This paper introduces BugStone, a program analysis system empowered by LLVM and a Large Language Model (LLM). The key observation is that many RPBs have one patched instance, which can be leveraged to identify a consistent error pattern, such as a specific API misuse. By examining the entire program for this pattern, it is possible to identify similar sections of code that may be vulnerable. Starting with 135 unique RPBs, BugStone identified more than 22K new potential issues in the Linux kernel. Manual analysis of 400 of these findings confirmed that 246 were valid. We also created a dataset from over 1.9K security bugs reported by 23 recent top-tier conference works. We manually annotate the dataset, identify 80 recurring patterns and 850 corresponding fixes. Even with a cost-efficient model choice, BugStone achieved 92.2% precision and 79.1% pairwise accuracy on the dataset. 

---
# Think Globally, Group Locally: Evaluating LLMs Using Multi-Lingual Word Grouping Games 

**Authors**: César Guerra-Solano, Zhuochun Li, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14030)  

**Abstract**: Large language models (LLMs) can exhibit biases in reasoning capabilities due to linguistic modality, performing better on tasks in one language versus another, even with similar content. Most previous works evaluate this through reasoning tasks where reliance on strategies or knowledge can ensure success, such as in commonsense or math tasks. However, abstract reasoning is vital to reasoning for everyday life, where people apply "out-of-the-box thinking" to identify and use patterns for solutions, without a reliance on formulaic approaches. Comparatively, little work has evaluated linguistic biases in this task type. In this paper, we propose a task inspired by the New York Times Connections: GlobalGroup, that evaluates models in an abstract reasoning task across several languages. We constructed a game benchmark with five linguistic backgrounds -- English, Spanish, Chinese, Hindi, and Arabic -- in both the native language and an English translation for comparison. We also proposed game difficulty measurements to evaluate models on games with similar difficulty, enabling a more controlled comparison, which is particularly important in reasoning evaluations. Through experimentation, we find English modalities largely lead to better performance in this abstract reasoning task, and performance disparities between open- and closed-source models. 

---
# Context-Selective State Space Models: Feedback is All You Need 

**Authors**: Riccardo Zattra, Giacomo Baggio, Umberto Casti, Augusto Ferrante, Francesco Ticozzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.14027)  

**Abstract**: Transformers, powered by the attention mechanism, are the backbone of most foundation models, yet they suffer from quadratic complexity and difficulties in dealing with long-range dependencies in the input sequence. Recent work has shown that state space models (SSMs) provide an efficient alternative, with the S6 module at the core of the Mamba architecture achieving state-of-the-art results on long-sequence benchmarks. In this paper, we introduce the COFFEE (COntext From FEEdback) model, a novel time-varying SSM that incorporates state feedback to enable context-dependent selectivity, while still allowing for parallel implementation. Whereas the selectivity mechanism of S6 only depends on the current input, COFFEE computes it from the internal state, which serves as a compact representation of the sequence history. This shift allows the model to regulate its dynamics based on accumulated context, improving its ability to capture long-range dependencies. In addition to state feedback, we employ an efficient model parametrization that removes redundancies present in S6 and leads to a more compact and trainable formulation. On the induction head task, COFFEE achieves near-perfect accuracy with two orders of magnitude fewer parameters and training sequences compared to S6. On MNIST, COFFEE largely outperforms S6 within the same architecture, reaching 97% accuracy with only 3585 parameters. These results showcase the role of state feedback as a key mechanism for building scalable and efficient sequence models. 

---
# Conditional Clifford-Steerable CNNs with Complete Kernel Basis for PDE Modeling 

**Authors**: Bálint László Szarvas, Maksim Zhdanov  

**Link**: [PDF](https://arxiv.org/pdf/2510.14007)  

**Abstract**: Clifford-Steerable CNNs (CSCNNs) provide a unified framework that allows incorporating equivariance to arbitrary pseudo-Euclidean groups, including isometries of Euclidean space and Minkowski spacetime. In this work, we demonstrate that the kernel basis of CSCNNs is not complete, thus limiting the model expressivity. To address this issue, we propose Conditional Clifford-Steerable Kernels, which augment the kernels with equivariant representations computed from the input feature field. We derive the equivariance constraint for these input-dependent kernels and show how it can be solved efficiently via implicit parameterization. We empirically demonstrate an improved expressivity of the resulting framework on multiple PDE forecasting tasks, including fluid dynamics and relativistic electrodynamics, where our method consistently outperforms baseline methods. 

---
# REAP the Experts: Why Pruning Prevails for One-Shot MoE compression 

**Authors**: Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie, Yani Ioannou, Vithursan Thangarasa  

**Link**: [PDF](https://arxiv.org/pdf/2510.13999)  

**Abstract**: Sparsely-activated Mixture-of-Experts (SMoE) models offer efficient pre-training and low latency but their large parameter counts create significant memory overhead, motivating research into expert compression. Contrary to recent findings favouring expert merging on discriminative benchmarks, we demonstrate that expert pruning is a superior strategy for generative tasks. We prove that merging introduces an irreducible error by causing a "functional subspace collapse", due to the loss of the router's independent, input-dependent control over experts. Leveraging this insight, we propose Router-weighted Expert Activation Pruning (REAP), a novel pruning criterion that considers both router gate-values and expert activation norms. Across a diverse set of SMoE models ranging from 20B to 1T parameters, REAP consistently outperforms merging and other pruning methods on generative benchmarks, especially at 50% compression. Notably, our method achieves near-lossless compression on code generation and tool-calling tasks with Qwen3-Coder-480B and Kimi-K2, even after pruning 50% of experts. 

---
# Finding Holes: Pathologist Level Performance Using AI for Cribriform Morphology Detection in Prostate Cancer 

**Authors**: Kelvin Szolnoky, Anders Blilie, Nita Mulliqi, Toyonori Tsuzuki, Hemamali Samaratunga, Matteo Titus, Xiaoyi Ji, Sol Erika Boman, Einar Gudlaugsson, Svein Reidar Kjosavik, José Asenjo, Marcello Gambacorta, Paolo Libretti, Marcin Braun, Radisław Kordek, Roman Łowicki, Brett Delahunt, Kenneth A. Iczkowski, Theo van der Kwast, Geert J. L. H. van Leenders, Katia R. M. Leite, Chin-Chen Pan, Emiel Adrianus Maria Janssen, Martin Eklund, Lars Egevad, Kimmo Kartasalo  

**Link**: [PDF](https://arxiv.org/pdf/2510.13995)  

**Abstract**: Background: Cribriform morphology in prostate cancer is a histological feature that indicates poor prognosis and contraindicates active surveillance. However, it remains underreported and subject to significant interobserver variability amongst pathologists. We aimed to develop and validate an AI-based system to improve cribriform pattern detection.
Methods: We created a deep learning model using an EfficientNetV2-S encoder with multiple instance learning for end-to-end whole-slide classification. The model was trained on 640 digitised prostate core needle biopsies from 430 patients, collected across three cohorts. It was validated internally (261 slides from 171 patients) and externally (266 slides, 104 patients from three independent cohorts). Internal validation cohorts included laboratories or scanners from the development set, while external cohorts used completely independent instruments and laboratories. Annotations were provided by three expert uropathologists with known high concordance. Additionally, we conducted an inter-rater analysis and compared the model's performance against nine expert uropathologists on 88 slides from the internal validation cohort.
Results: The model showed strong internal validation performance (AUC: 0.97, 95% CI: 0.95-0.99; Cohen's kappa: 0.81, 95% CI: 0.72-0.89) and robust external validation (AUC: 0.90, 95% CI: 0.86-0.93; Cohen's kappa: 0.55, 95% CI: 0.45-0.64). In our inter-rater analysis, the model achieved the highest average agreement (Cohen's kappa: 0.66, 95% CI: 0.57-0.74), outperforming all nine pathologists whose Cohen's kappas ranged from 0.35 to 0.62.
Conclusion: Our AI model demonstrates pathologist-level performance for cribriform morphology detection in prostate cancer. This approach could enhance diagnostic reliability, standardise reporting, and improve treatment decisions for prostate cancer patients. 

---
# Efficient Few-Shot Learning in Remote Sensing: Fusing Vision and Vision-Language Models 

**Authors**: Jia Yun Chua, Argyrios Zolotas, Miguel Arana-Catania  

**Link**: [PDF](https://arxiv.org/pdf/2510.13993)  

**Abstract**: Remote sensing has become a vital tool across sectors such as urban planning, environmental monitoring, and disaster response. While the volume of data generated has increased significantly, traditional vision models are often constrained by the requirement for extensive domain-specific labelled data and their limited ability to understand the context within complex environments. Vision Language Models offer a complementary approach by integrating visual and textual data; however, their application to remote sensing remains underexplored, particularly given their generalist nature. This work investigates the combination of vision models and VLMs to enhance image analysis in remote sensing, with a focus on aircraft detection and scene understanding. The integration of YOLO with VLMs such as LLaVA, ChatGPT, and Gemini aims to achieve more accurate and contextually aware image interpretation. Performance is evaluated on both labelled and unlabelled remote sensing data, as well as degraded image scenarios which are crucial for remote sensing. The findings show an average MAE improvement of 48.46% across models in the accuracy of aircraft detection and counting, especially in challenging conditions, in both raw and degraded scenarios. A 6.17% improvement in CLIPScore for comprehensive understanding of remote sensing images is obtained. The proposed approach combining traditional vision models and VLMs paves the way for more advanced and efficient remote sensing image analysis, especially in few-shot learning scenarios. 

---
# Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations 

**Authors**: Jinkun Chen, Sher Badshah, Xuemin Yu, Sijia Han, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.13982)  

**Abstract**: What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. \textbf{We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.} 

---
# Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention 

**Authors**: Zhen Yang, Mingyang Zhang, Feng Chen, Ganggui Ding, Liang Hou, Xin Tao, Pengfei Wan, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13940)  

**Abstract**: Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +1.35% average improvement on eight benchmarks for Qwen3-8B-Base and +5% on AIME2024 using Qwen3-32B-Reasoning-while remaining highly efficient. 

---
# Big Reasoning with Small Models: Instruction Retrieval at Inference Time 

**Authors**: Kenan Alkiek, David Jurgens, Vinod Vydiswaran  

**Link**: [PDF](https://arxiv.org/pdf/2510.13935)  

**Abstract**: Can we bring large-scale reasoning to local-scale compute? Small language models (SLMs) are increasingly attractive because they run efficiently on local hardware, offering strong privacy, low cost, and reduced environmental impact. Yet they often struggle with tasks that require multi-step reasoning or domain-specific knowledge. We address this limitation through instruction intervention at inference time, where an SLM retrieves structured reasoning procedures rather than generating them from scratch. Our method builds an Instruction Corpus by grouping similar training questions and creating instructions via GPT-5. During inference, the SLM retrieves the most relevant instructions and follows their steps. Unlike retrieval-augmented generation, which retrieves text passages, instruction retrieval gives the model structured guidance for reasoning. We evaluate this framework on MedQA (medical board exams), MMLU Professional Law, and MathQA using models from 3B to 14B parameters without any additional fine-tuning. Instruction retrieval yields consistent gains: 9.4% on MedQA, 7.9% on MMLU Law, and 5.1% on MathQA. Concise instructions outperform longer ones, and the magnitude of improvement depends strongly on model family and intrinsic reasoning ability. 

---
# LLMs Can Get "Brain Rot"! 

**Authors**: Shuo Xing, Junyuan Hong, Yifan Wang, Runjin Chen, Zhenyu Zhang, Ananth Grama, Zhengzhong Tu, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13928)  

**Abstract**: We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To causally isolate data quality, we run controlled experiments on real Twitter/X corpora, constructing junk and reversely controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Contrary to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' $g>0.3$) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain Of Thoughts drops $74.9 \rightarrow 57.2$ and RULER-CWE $84.4 \rightarrow 52.3$ as junk ratio rises from $0\%$ to $100\%$.
Error forensics reveal several key insights. First, we identify thought-skipping as the primary lesion: models increasingly truncate or skip reasoning chains, explaining most of the error growth. Second, partial but incomplete healing is observed: scaling instruction tuning and clean data pre-training improve the declined cognition yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that data quality is a causal driver of LLM capability decay, reframing curation for continual pretraining as a \textit{training-time safety} problem and motivating routine "cognitive health checks" for deployed LLMs. 

---
# Readability $\ne$ Learnability: Rethinking the Role of Simplicity in Training Small Language Models 

**Authors**: Ivan Lee, Taylor Berg-Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2510.13915)  

**Abstract**: Recent studies suggest that very small language models (SLMs) can generate surprisingly coherent text when trained on simplified, child-directed corpora such as TinyStories. These findings have been interpreted as evidence that readability -- characterized by accessible vocabulary, familiar narrative structure, and simple syntax -- plays a key role in enabling such capabilities to emerge. In this paper, we challenge that interpretation. We construct synthetic datasets with matched structure but varied readability, and find that readability alone does not predict coherence or learning efficiency in SLMs. Models trained on complex, adult-level text perform comparably to those trained on simplified language, and even exhibit faster development of coherence during training. Instead, we show that statistical simplicity, as measured by n-gram diversity, is a stronger predictor of learnability. Our findings caution against the growing trend of anthropomorphizing language model training -- drawing parallels to human cognitive development without empirical basis -- and argue for more precise reasoning about what properties actually support capability emergence in small models. 

---
# Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms 

**Authors**: Shrey Pandit, Xuan-Phi Nguyen, Yifei Ming, Austin Xu, Jiayu Wang, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2510.13913)  

**Abstract**: Web-based 'deep research' agents aim to solve complex question - answering tasks through long-horizon interactions with online tools. These tasks remain challenging, as the underlying language models are often not optimized for long-horizon reasoning and exploration. Prior work has proposed workflows for constructing instruction-tuning datasets, often leveraging knowledge graphs. However, such methods typically lack fine-grained control over difficulty and quality, yielding synthetic data that falls short of capturing the complexity required for long-horizon reasoning. Furthermore, many studies conflate data and training effects by comparing models trained under different optimization recipes, making it difficult to isolate and evaluate the effectiveness of the data itself. We introduce a two-pronged data synthesis pipeline that generates question - answer pairs by progressively increasing task complexity until a frontier baseline web agent fails. The baseline agent plays multiple roles in this process: attempting the questions, validating factuality, checking for alternative answers, and enforcing filtering. To evaluate the effectiveness of our synthesis methods, we adopt a controlled training setup based on distillation from strong web agents. Experiments across multiple web-based benchmarks show that our dataset - despite being smaller - enables the training of more effective web agents than existing datasets. In particular, our data exhibits twice the diversity in tool-use actions, allowing models trained on it to achieve stronger performance while avoiding repetitive tool-calling behaviors. 

---
# AI Debaters are More Persuasive when Arguing in Alignment with Their Own Beliefs 

**Authors**: María Victoria Carro, Denise Alejandra Mester, Facundo Nieto, Oscar Agustín Stanchi, Guido Ernesto Bergman, Mario Alejandro Leiva, Eitan Sprejer, Luca Nicolás Forziati Gangi, Francisca Gauna Selasco, Juan Gustavo Corvalán, Gerardo I. Simari, María Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13912)  

**Abstract**: The core premise of AI debate as a scalable oversight technique is that it is harder to lie convincingly than to refute a lie, enabling the judge to identify the correct position. Yet, existing debate experiments have relied on datasets with ground truth, where lying is reduced to defending an incorrect proposition. This overlooks a subjective dimension: lying also requires the belief that the claim defended is false. In this work, we apply debate to subjective questions and explicitly measure large language models' prior beliefs before experiments. Debaters were asked to select their preferred position, then presented with a judge persona deliberately designed to conflict with their identified priors. This setup tested whether models would adopt sycophantic strategies, aligning with the judge's presumed perspective to maximize persuasiveness, or remain faithful to their prior beliefs. We implemented and compared two debate protocols, sequential and simultaneous, to evaluate potential systematic biases. Finally, we assessed whether models were more persuasive and produced higher-quality arguments when defending positions consistent with their prior beliefs versus when arguing against them. Our main findings show that models tend to prefer defending stances aligned with the judge persona rather than their prior beliefs, sequential debate introduces significant bias favoring the second debater, models are more persuasive when defending positions aligned with their prior beliefs, and paradoxically, arguments misaligned with prior beliefs are rated as higher quality in pairwise comparison. These results can inform human judges to provide higher-quality training signals and contribute to more aligned AI systems, while revealing important aspects of human-AI interaction regarding persuasion dynamics in language models. 

---
# Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning 

**Authors**: Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Zhongyuan Wang, Jichen Zhang, Shirui Pan, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13909)  

**Abstract**: Inductive Knowledge Graph Reasoning (KGR) aims to discover facts in open-domain KGs containing unknown entities and relations, which poses a challenge for KGR models in comprehending uncertain KG components. Existing studies have proposed Knowledge Graph Foundation Models (KGFMs) that learn structural invariances across KGs to handle this uncertainty. Recently, Large Language Models (LLMs) have demonstrated strong capabilities for open-domain knowledge reasoning. As a result, the latest research has focused on LLM-based KGFMs that integrate LLM knowledge with KG context for inductive KGR. However, the intrinsic knowledge of LLMs may be overshadowed by sparse KG context, leading to LLM knowledge distortion, which can cause irreversible damage to model reasoning. Moreover, existing LLM-based KGR methods still struggle to fully constrain generative hallucinations in LLMs, severely limiting the credibility of reasoning results. To address these limitations, we propose a Knowledge Reasoning Language Model (KRLM) that achieves unified coordination between LLM knowledge and KG context throughout the KGR process. Specifically, we design a Knowledge Reasoning Language (KRL) instruction format and a KRL tokenizer to align LLM knowledge with KG representations. Then, we propose a KRL attention layer that coordinates intrinsic LLM knowledge with additional KG context through a dynamic knowledge memory mechanism. Finally, a structure-aware next-entity predictor is proposed, which strictly constrains the reasoning results within a trustworthy knowledge domain. Extensive experimental results on 25 real-world inductive KGR datasets demonstrate the significant superiority of the proposed KRLM\footnote{Our source codes are available at this https URL in both zero-shot reasoning and fine-tuning scenarios. 

---
# Schema for In-Context Learning 

**Authors**: Pan Chen, Shaohong Chen, Mark Wang, Shi Xuan Leong, Priscilla Fung, Varinia Bernales, Alan Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2510.13905)  

**Abstract**: In-Context Learning (ICL) enables transformer-based language models to adapt to new tasks by conditioning on demonstration examples. However, traditional example-driven in-context learning lacks explicit modules for knowledge retrieval and transfer at the abstraction level. Inspired by cognitive science, specifically schema theory, which holds that humans interpret new information by activating pre-existing mental frameworks (schemas) to structure understanding, we introduce SCHEMA ACTIVATED IN CONTEXT LEARNING (SA-ICL). This framework extracts the representation of the building blocks of cognition for the reasoning process instilled from prior examples, creating an abstracted schema, a lightweight, structured template of key inferential steps and their relationships, which is then used to augment a model's reasoning process when presented with a novel question. We demonstrate that a broad range of large language models (LLMs) lack the capacity to form and utilize internal schema-based learning representations implicitly, but instead benefit significantly from explicit schema-based scaffolding. Across chemistry and physics questions from the GPQA dataset, our experiments show that SA-ICL consistently boosts performance, up to 36.19 percent, when the single demonstration example is of high quality, which simultaneously reduces reliance on the number of demonstrations and enhances interpretability. SCHEMA ACTIVATED IN CONTEXT LEARNING not only bridges disparate ICL strategies ranging from pattern priming to Chain-of-Thought prompting, but also paves a new path for enhancing human-like reasoning in LLMs. 

---
# Benefits and Limitations of Communication in Multi-Agent Reasoning 

**Authors**: Michael Rizvi-Martel, Satwik Bhattamishra, Neil Rathi, Guillaume Rabusseau, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2510.13903)  

**Abstract**: Chain-of-thought prompting has popularized step-by-step reasoning in large language models, yet model performance still degrades as problem complexity and context length grow. By decomposing difficult tasks with long contexts into shorter, manageable ones, recent multi-agent paradigms offer a promising near-term solution to this problem. However, the fundamental capacities of such systems are poorly understood. In this work, we propose a theoretical framework to analyze the expressivity of multi-agent systems. We apply our framework to three algorithmic families: state tracking, recall, and $k$-hop reasoning. We derive bounds on (i) the number of agents required to solve the task exactly, (ii) the quantity and structure of inter-agent communication, and (iii) the achievable speedups as problem size and context scale. Our results identify regimes where communication is provably beneficial, delineate tradeoffs between agent count and bandwidth, and expose intrinsic limitations when either resource is constrained. We complement our theoretical analysis with a set of experiments on pretrained LLMs using controlled synthetic benchmarks. Empirical outcomes confirm the tradeoffs between key quantities predicted by our theory. Collectively, our analysis offers principled guidance for designing scalable multi-agent reasoning systems. 

---
# Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences 

**Authors**: Julian Minder, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes, Robert West, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.13900)  

**Abstract**: Finetuning on narrow domains has become an essential tool to adapt Large Language Models (LLMs) to specific tasks and to create models with known unusual properties that are useful for research. We show that narrow finetuning creates strong biases in LLM activations that can be interpreted to understand the finetuning domain. These biases can be discovered using simple tools from model diffing - the study of differences between models before and after finetuning. In particular, analyzing activation differences on the first few tokens of random text and steering by adding this difference to the model activations produces text similar to the format and general content of the finetuning data. We demonstrate that these analyses contain crucial information by creating an LLM-based interpretability agent to understand the finetuning domain. With access to the bias, the agent performs significantly better compared to baseline agents using simple prompting. Our analysis spans synthetic document finetuning for false facts, emergent misalignment, subliminal learning, and taboo word guessing game models across different architectures (Gemma, LLaMA, Qwen) and scales (1B to 32B parameters). We suspect these biases reflect overfitting and find that mixing pretraining data into the finetuning corpus largely removes them, though residual risks may remain. Our work (1) demonstrates that narrowly finetuned models have salient traces of their training objective in their activations and suggests ways to improve how they are trained, (2) warns AI safety and interpretability researchers that the common practice of using such models as a proxy for studying broader finetuning (e.g., chat-tuning) might not be realistic, and (3) highlights the need for deeper investigation into the effects of narrow finetuning and development of truly realistic case studies for model-diffing, safety and interpretability research. 

---
# Dual-attention ResNet outperforms transformers in HER2 prediction on DCE-MRI 

**Authors**: Naomi Fridman, Anat Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.13897)  

**Abstract**: Breast cancer is the most diagnosed cancer in women, with HER2 status critically guiding treatment decisions. Noninvasive prediction of HER2 status from dynamic contrast-enhanced MRI (DCE-MRI) could streamline diagnostics and reduce reliance on biopsy. However, preprocessing high-dynamic-range DCE-MRI into standardized 8-bit RGB format for pretrained neural networks is nontrivial, and normalization strategy significantly affects model performance. We benchmarked intensity normalization strategies using a Triple-Head Dual-Attention ResNet that processes RGB-fused temporal sequences from three DCE phases. Trained on a multicenter cohort (n=1,149) from the I-SPY trials and externally validated on BreastDCEDL_AMBL (n=43 lesions), our model outperformed transformer-based architectures, achieving 0.75 accuracy and 0.74 AUC on I-SPY test data. N4 bias field correction slightly degraded performance. Without fine-tuning, external validation yielded 0.66 AUC, demonstrating cross-institutional generalizability. These findings highlight the effectiveness of dual-attention mechanisms in capturing transferable spatiotemporal features for HER2 stratification, advancing reproducible deep learning biomarkers in breast cancer imaging. 

---
# GenCellAgent: Generalizable, Training-Free Cellular Image Segmentation via Large Language Model Agents 

**Authors**: Xi Yu, Yang Yang, Qun Liu, Yonghua Du, Sean McSweeney, Yuewei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.13896)  

**Abstract**: Cellular image segmentation is essential for quantitative biology yet remains difficult due to heterogeneous modalities, morphological variability, and limited annotations. We present GenCellAgent, a training-free multi-agent framework that orchestrates specialist segmenters and generalist vision-language models via a planner-executor-evaluator loop (choose tool $\rightarrow$ run $\rightarrow$ quality-check) with long-term memory. The system (i) automatically routes images to the best tool, (ii) adapts on the fly using a few reference images when imaging conditions differ from what a tool expects, (iii) supports text-guided segmentation of organelles not covered by existing models, and (iv) commits expert edits to memory, enabling self-evolution and personalized workflows. Across four cell-segmentation benchmarks, this routing yields a 15.7\% mean accuracy gain over state-of-the-art baselines. On endoplasmic reticulum and mitochondria from new datasets, GenCellAgent improves average IoU by 37.6\% over specialist models. It also segments novel objects such as the Golgi apparatus via iterative text-guided refinement, with light human correction further boosting performance. Together, these capabilities provide a practical path to robust, adaptable cellular image segmentation without retraining, while reducing annotation burden and matching user preferences. 

---
# Bayes or Heisenberg: Who(se) Rules? 

**Authors**: Volker Tresp Hang Li, Federico Harjes, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13894)  

**Abstract**: Although quantum systems are generally described by quantum state vectors, we show that in certain cases their measurement processes can be reformulated as probabilistic equations expressed in terms of probabilistic state vectors. These probabilistic representations can, in turn, be approximated by the neural network dynamics of the Tensor Brain (TB) model.
The Tensor Brain is a recently proposed framework for modeling perception and memory in the brain, providing a biologically inspired mechanism for efficiently integrating generated symbolic representations into reasoning processes. 

---
# Guarding the Guardrails: A Taxonomy-Driven Approach to Jailbreak Detection 

**Authors**: Olga E. Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13893)  

**Abstract**: Jailbreaking techniques pose a significant threat to the safety of Large Language Models (LLMs). Existing defenses typically focus on single-turn attacks, lack coverage across languages, and rely on limited taxonomies that either fail to capture the full diversity of attack strategies or emphasize risk categories rather than the jailbreaking techniques. To advance the understanding of the effectiveness of jailbreaking techniques, we conducted a structured red-teaming challenge. The outcome of our experiments are manifold. First, we developed a comprehensive hierarchical taxonomy of 50 jailbreak strategies, consolidating and extending prior classifications into seven broad families, including impersonation, persuasion, privilege escalation, cognitive overload, obfuscation, goal conflict, and data poisoning. Second, we analyzed the data collected from the challenge to examine the prevalence and success rates of different attack types, providing insights into how specific jailbreak strategies exploit model vulnerabilities and induce misalignment. Third, we benchmark a popular LLM for jailbreak detection, evaluating the benefits of taxonomy-guided prompting for improving automatic detection. Finally, we compiled a new Italian dataset of 1364 multi-turn adversarial dialogues, annotated with our taxonomy, enabling the study of interactions where adversarial intent emerges gradually and succeeds in bypassing traditional safeguards. 

---
# K-frames: Scene-Driven Any-k Keyframe Selection for long video understanding 

**Authors**: Yifeng Yao, Yike Yun, Jing Wang, Huishuai Zhang, Dongyan Zhao, Ke Tian, Zhihao Wang, Minghui Qiu, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13891)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant capabilities in image understanding, but long-video are constrained by context windows and computational cost. Uniform frame sampling often leads to substantial information loss. Meanwhile existing keyframe selection methods such as text-frame retrieval or RL-based frame optimization typically yield sparse and temporally disjointed frames, overlooking scene continuity and lacking flexibility for multi-scale frame selection. To address these limitations, we introduce K-frames, a novel paradigm for scene-driven keyframe selection that preserves temporal continuity. Instead of selecting individual frames, K-frames predicts semantically coherent, query-relevant clips, which enables any-k keyframes selection to meet diverse user budgets. To achieve this approach, we first introduce PeakClips, a dataset of 200K video highlights conditioned by query. Building on this dataset, K-frames learns clip2frame selection using a three-stage progressive curriculum. It involves two Supervised Fine-Tuning stages for temporal grounding and key-clip perception, followed by a Reinforcement Learning stage that directly optimizes the scene-driven prediction policy for downstream task without further annotations. Extensive experiments on major long-video understanding benchmarks demonstrate that K-frames provides an effective, interpretable, and plug-and-play solution for keyframe selection at various scales. Our dataset and model will be available. 

---
# A Survey on Collaborating Small and Large Language Models for Performance, Cost-effectiveness, Cloud-edge Privacy, and Trustworthiness 

**Authors**: Fali Wang, Jihai Chen, Shuhua Yang, Ali Al-Lawati, Linli Tang, Hui Liu, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13890)  

**Abstract**: Large language models (LLMs) have advanced many domains and applications but face high fine-tuning costs, inference latency, limited edge deployability, and reliability concerns. Small language models (SLMs), compact, efficient, and adaptable, offer complementary remedies. Recent work explores collaborative frameworks that fuse SLMs' specialization and efficiency with LLMs' generalization and reasoning to meet diverse objectives across tasks and deployment scenarios. Motivated by these developments, this paper presents a systematic survey of SLM-LLM collaboration organized by collaboration objectives. We propose a taxonomy with four goals: performance enhancement, cost-effectiveness, cloud-edge privacy, and trustworthiness. Within this framework, we review representative methods, summarize design paradigms, and outline open challenges and future directions toward efficient, secure, and scalable SLM-LLM collaboration. 

---
# Reliable Fine-Grained Evaluation of Natural Language Math Proofs 

**Authors**: Wenjie Ma, Andrei Cojocaru, Neel Kolhe, Bradley Louie, Robin Said Sharif, Haihan Zhang, Vincent Zhuang, Matei Zaharia, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2510.13888)  

**Abstract**: Recent advances in large language models (LLMs) for mathematical reasoning have largely focused on tasks with easily verifiable final answers; however, generating and verifying natural language math proofs remains an open challenge. We identify the absence of a reliable, fine-grained evaluator for LLM-generated math proofs as a critical gap. To address this, we propose a systematic methodology for developing and validating evaluators that assign fine-grained scores on a 0-7 scale to model-generated math proofs. To enable this study, we introduce ProofBench, the first expert-annotated dataset of fine-grained proof ratings, spanning 145 problems from six major math competitions (USAMO, IMO, Putnam, etc) and 435 LLM-generated solutions from Gemini-2.5-pro, o3, and DeepSeek-R1. %with expert gradings. Using ProofBench as a testbed, we systematically explore the evaluator design space across key axes: the backbone model, input context, instructions and evaluation workflow. Our analysis delivers ProofGrader, an evaluator that combines a strong reasoning backbone LM, rich context from reference solutions and marking schemes, and a simple ensembling method; it achieves a low Mean Absolute Error (MAE) of 0.926 against expert scores, significantly outperforming naive baselines. Finally, we demonstrate its practical utility in a best-of-$n$ selection task: at $n=16$, ProofGrader achieves an average score of 4.14 (out of 7), closing 78% of the gap between a naive binary evaluator (2.48) and the human oracle (4.62), highlighting its potential to advance downstream proof generation. 

---
# Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion 

**Authors**: Xiaojian Ding, Lin Zhao, Xian Li, Xiaoying Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13887)  

**Abstract**: Incomplete multi-view data, where certain views are entirely missing for some samples, poses significant challenges for traditional multi-view clustering methods. Existing deep incomplete multi-view clustering approaches often rely on static fusion strategies or two-stage pipelines, leading to suboptimal fusion results and error propagation issues. To address these limitations, this paper proposes a novel incomplete multi-view clustering framework based on Hierarchical Semantic Alignment and Cooperative Completion (HSACC). HSACC achieves robust cross-view fusion through a dual-level semantic space design. In the low-level semantic space, consistency alignment is ensured by maximizing mutual information across views. In the high-level semantic space, adaptive view weights are dynamically assigned based on the distributional affinity between individual views and an initial fused representation, followed by weighted fusion to generate a unified global representation. Additionally, HSACC implicitly recovers missing views by projecting aligned latent representations into high-dimensional semantic spaces and jointly optimizes reconstruction and clustering objectives, enabling cooperative learning of completion and clustering. Experimental results demonstrate that HSACC significantly outperforms state-of-the-art methods on five benchmark datasets. Ablation studies validate the effectiveness of the hierarchical alignment and dynamic weighting mechanisms, while parameter analysis confirms the model's robustness to hyperparameter variations. 

---
# Physics-Informed autoencoder for DSC-MRI Perfusion post-processing: application to glioma grading 

**Authors**: Pierre Fayolle, Alexandre Bône, Noëlie Debs, Mathieu Naudin, Pascal Bourdon, Remy Guillevin, David Helbert  

**Link**: [PDF](https://arxiv.org/pdf/2510.13886)  

**Abstract**: DSC-MRI perfusion is a medical imaging technique for diagnosing and prognosing brain tumors and strokes. Its analysis relies on mathematical deconvolution, but noise or motion artifacts in a clinical environment can disrupt this process, leading to incorrect estimate of perfusion parameters. Although deep learning approaches have shown promising results, their calibration typically rely on third-party deconvolution algorithms to generate reference outputs and are bound to reproduce their limitations.
To adress this problem, we propose a physics-informed autoencoder that leverages an analytical model to decode the perfusion parameters and guide the learning of the encoding network. This autoencoder is trained in a self-supervised fashion without any third-party software and its performance is evaluated on a database with glioma patients. Our method shows reliable results for glioma grading in accordance with other well-known deconvolution algorithms despite a lower computation time. It also achieved competitive performance even in the presence of high noise which is critical in a medical environment. 

---
# Order from Chaos: Comparative Study of Ten Leading LLMs on Unstructured Data Categorization 

**Authors**: Ariel Kamen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13885)  

**Abstract**: This study presents a comparative evaluation of ten state-of-the-art large language models (LLMs) applied to unstructured text categorization using the Interactive Advertising Bureau (IAB) 2.2 hierarchical taxonomy. The analysis employed a uniform dataset of 8,660 human-annotated samples and identical zero-shot prompts to ensure methodological consistency across all models. Evaluation metrics included four classic measures - accuracy, precision, recall, and F1-score - and three LLM-specific indicators: hallucination ratio, inflation ratio, and categorization cost.
Results show that, despite their rapid advancement, contemporary LLMs achieve only moderate classic performance, with average scores of 34% accuracy, 42% precision, 45% recall, and 41% F1-score. Hallucination and inflation ratios reveal that models frequently overproduce categories relative to human annotators. Among the evaluated systems, Gemini 1.5/2.0 Flash and GPT 20B/120B offered the most favorable cost-to-performance balance, while GPT 120B demonstrated the lowest hallucination ratio. The findings suggest that scaling and architectural improvements alone do not ensure better categorization accuracy, as the task requires compressing rich unstructured text into a limited taxonomy - a process that challenges current model architectures.
To address these limitations, a separate ensemble-based approach was developed and tested. The ensemble method, in which multiple LLMs act as independent experts, substantially improved accuracy, reduced inflation, and completely eliminated hallucinations. These results indicate that coordinated orchestration of models - rather than sheer scale - may represent the most effective path toward achieving or surpassing human-expert performance in large-scale text categorization. 

---
# PAGE: Prompt Augmentation for text Generation Enhancement 

**Authors**: Mauro Jose Pacchiotti, Luciana Ballejos, Mariel Ale  

**Link**: [PDF](https://arxiv.org/pdf/2510.13880)  

**Abstract**: In recent years, natural language generative models have shown outstanding performance in text generation tasks. However, when facing specific tasks or particular requirements, they may exhibit poor performance or require adjustments that demand large amounts of additional data. This work introduces PAGE (Prompt Augmentation for text Generation Enhancement), a framework designed to assist these models through the use of simple auxiliary modules. These modules, lightweight models such as classifiers or extractors, provide inferences from the input text. The output of these auxiliaries is then used to construct an enriched input that improves the quality and controllability of the generation. Unlike other generation-assistance approaches, PAGE does not require auxiliary generative models; instead, it proposes a simpler, modular architecture that is easy to adapt to different tasks. This paper presents the proposal, its components and architecture, and reports a proof of concept in the domain of requirements engineering, where an auxiliary module with a classifier is used to improve the quality of software requirements generation. 

---
# Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production 

**Authors**: Alexandre Galashov, Matt Jones, Rosemary Ke, Yuan Cao, Vaishnavh Nagarajan, Michael C. Mozer  

**Link**: [PDF](https://arxiv.org/pdf/2510.13879)  

**Abstract**: We explore a class of supervised training objectives that allow a language model to dynamically and autonomously scale the number of compute steps used for each input token. For any token, the model can request additional compute steps by emitting a <don't know> output. If the model is granted a delay, a specialized <pause> token is inserted at the next input step, providing the model with additional compute resources to generate an output. The model can request multiple pauses. To train the model to use <don't know> outputs judiciously and to calibrate its uncertainty, we frame the selection of each output token as a sequential-decision problem with a time cost. We refer to the class of methods as $\textit{Catch Your Breath}$ losses and we study three methods in this class: CYB-AP frames the model's task as anytime prediction, where an output may be required at any step and accuracy is discounted over time; CYB-VA is a variational approach that aims to maximize prediction accuracy subject to a specified distribution over stopping times; and CYB-DP imposes a penalty based on a computational budget. Through fine-tuning experiments, we identify the best performing loss variant. The CYB model needs only one third as much training data as the baseline (no pause) model needs to achieve the same performance, and half as much data as a model with pauses and a cross-entropy loss. We find that the CYB model requests additional steps when doing so improves accuracy, and the model adapts its processing time to token-level complexity and context. For example, it often pauses after plural nouns like $\textit{patients}$ and $\textit{challenges}$ but never pauses after the first token of contracted words like $\textit{wasn}$ and $\textit{didn}$, and it shows high variability for ambiguous tokens like $\textit{won}$, which could function as either a verb or part of a contraction. 

---
# What Layers When: Learning to Skip Compute in LLMs with Residual Gates 

**Authors**: Filipe Laitenberger, Dawid Kopiczko, Cees G.M. Snoek, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2510.13876)  

**Abstract**: We introduce GateSkip, a simple residual-stream gating mechanism that enables token-wise layer skipping in decoder-only LMs. Each Attention/MLP branch is equipped with a sigmoid-linear gate that condenses the branch's output before it re-enters the residual stream. During inference we rank tokens by the gate values and skip low-importance ones using a per-layer budget. While early-exit or router-based Mixture-of-Depths models are known to be unstable and need extensive retraining, our smooth, differentiable gates fine-tune stably on top of pretrained models. On long-form reasoning, we save up to 15\% compute while retaining over 90\% of baseline accuracy. On instruction-tuned models we see accuracy gains at full compute and match baseline quality near 50\% savings. The learned gates give insight into transformer information flow (e.g., BOS tokens act as anchors), and the method combines easily with quantization, pruning, and self-speculative decoding. 

---
# FRACCO: A gold-standard annotated corpus of oncological entities with ICD-O-3.1 normalisation 

**Authors**: Johann Pignat, Milena Vucetic, Christophe Gaudet-Blavignac, Jamil Zaghir, Amandine Stettler, Fanny Amrein, Jonatan Bonjour, Jean-Philippe Goldman, Olivier Michielin, Christian Lovis, Mina Bjelogrlic  

**Link**: [PDF](https://arxiv.org/pdf/2510.13873)  

**Abstract**: Developing natural language processing tools for clinical text requires annotated datasets, yet French oncology resources remain scarce. We present FRACCO (FRench Annotated Corpus for Clinical Oncology) an expert-annotated corpus of 1301 synthetic French clinical cases, initially translated from the Spanish CANTEMIST corpus as part of the FRASIMED initiative. Each document is annotated with terms related to morphology, topography, and histologic differentiation, using the International Classification of Diseases for Oncology (ICD-O) as reference. An additional annotation layer captures composite expression-level normalisations that combine multiple ICD-O elements into unified clinical concepts. Annotation quality was ensured through expert review: 1301 texts were manually annotated for entity spans by two domain experts. A total of 71127 ICD-O normalisations were produced through a combination of automated matching and manual validation by a team of five annotators. The final dataset representing 399 unique morphology codes (from 2549 different expressions), 272 topography codes (from 3143 different expressions), and 2043 unique composite expressions (from 11144 different expressions). This dataset provides a reference standard for named entity recognition and concept normalisation in French oncology texts. 

---
# Joint Discriminative-Generative Modeling via Dual Adversarial Training 

**Authors**: Xuwang Yin, Claire Zhang, Julie Steele, Nir Shavit, Tony T. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13872)  

**Abstract**: Simultaneously achieving robust classification and high-fidelity generative modeling within a single framework presents a significant challenge. Hybrid approaches, such as Joint Energy-Based Models (JEM), interpret classifiers as EBMs but are often limited by the instability and poor sample quality inherent in SGLD-based training. We address these limitations by proposing a novel training framework that integrates adversarial training (AT) principles for both discriminative robustness and stable generative learning. The proposed method introduces three key innovations: (1) the replacement of SGLD-based JEM learning with a stable, AT-based approach that optimizes the energy function by discriminating between real data and PGD-generated contrastive samples using the BCE loss; (2) synergistic adversarial training for the discriminative component that enhances classification robustness while eliminating the need for explicit gradient penalties; and (3) a two-stage training procedure to resolve the incompatibility between batch normalization and EBM training. Experiments on CIFAR-10, CIFAR-100, and ImageNet demonstrate that our method substantially improves adversarial robustness over existing hybrid models while maintaining competitive generative performance. On ImageNet, when optimized for generative modeling, our model's generative fidelity surpasses that of BigGAN and approaches diffusion models, representing the first MCMC-based EBM approach to achieve high-quality generation on complex, high-resolution datasets. Our approach addresses key stability issues that have limited JEM scaling and demonstrates that adversarial training can serve as an effective foundation for unified frameworks capable of generating and robustly classifying visual data. 

---
# Unlocking the Potential of Diffusion Language Models through Template Infilling 

**Authors**: Junhoo Lee, Seungyeon Kim, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13870)  

**Abstract**: Diffusion Language Models (DLMs) have emerged as a promising alternative to Autoregressive Language Models, yet their inference strategies remain limited to prefix-based prompting inherited from the autoregressive paradigm. In this paper, we propose Template Infilling (TI), a tailored conditioning methodology for DLMs' generation process. Unlike conventional prefix prompting, TI first generates a structural template for the target response, then fills in the masked segments. To enhance the flexibility of this structural control, we introduce Dynamic Segment Allocation (DSA), which adaptively adjusts segment lengths based on generation confidence. We demonstrate the effectiveness of our approach on mathematical reasoning and code generation benchmarks, achieving consistent improvements of 17.01$\%$p over baseline. Furthermore, we show that TI provides additional advantages in multi-token generation settings, enabling effective speedup while maintaining generation quality. 

---
# CoLoR-GAN: Continual Few-Shot Learning with Low-Rank Adaptation in Generative Adversarial Networks 

**Authors**: Munsif Ali, Leonardo Rossi, Massimo Bertozzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13869)  

**Abstract**: Continual learning (CL) in the context of Generative Adversarial Networks (GANs) remains a challenging problem, particularly when it comes to learn from a few-shot (FS) samples without catastrophic forgetting. Current most effective state-of-the-art (SOTA) methods, like LFS-GAN, introduce a non-negligible quantity of new weights at each training iteration, which would become significant when considering the long term. For this reason, this paper introduces \textcolor{red}{\textbf{\underline{c}}}ontinual few-sh\textcolor{red}{\textbf{\underline{o}}}t learning with \textcolor{red}{\textbf{\underline{lo}}}w-\textcolor{red}{\textbf{\underline{r}}}ank adaptation in GANs named CoLoR-GAN, a framework designed to handle both FS and CL together, leveraging low-rank tensors to efficiently adapt the model to target tasks while reducing even more the number of parameters required. Applying a vanilla LoRA implementation already permitted us to obtain pretty good results. In order to optimize even further the size of the adapters, we challenged LoRA limits introducing a LoRA in LoRA (LLoRA) technique for convolutional layers. Finally, aware of the criticality linked to the choice of the hyperparameters of LoRA, we provide an empirical study to easily find the best ones. We demonstrate the effectiveness of CoLoR-GAN through experiments on several benchmark CL and FS tasks and show that our model is efficient, reaching SOTA performance but with a number of resources enormously reduced. Source code is available on \href{this https URL}{Github. 

---
# FFT-Accelerated Auxiliary Variable MCMC for Fermionic Lattice Models: A Determinant-Free Approach with $O(N\log N)$ Complexity 

**Authors**: Deqian Kong, Shi Feng, Jianwen Xie, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13866)  

**Abstract**: We introduce a Markov Chain Monte Carlo (MCMC) algorithm that dramatically accelerates the simulation of quantum many-body systems, a grand challenge in computational science. State-of-the-art methods for these problems are severely limited by $O(N^3)$ computational complexity. Our method avoids this bottleneck, achieving near-linear $O(N \log N)$ scaling per sweep.
Our approach samples a joint probability measure over two coupled variable sets: (1) particle trajectories of the fundamental fermions, and (2) auxiliary variables that decouple fermion interactions. The key innovation is a novel transition kernel for particle trajectories formulated in the Fourier domain, revealing the transition probability as a convolution that enables massive acceleration via the Fast Fourier Transform (FFT). The auxiliary variables admit closed-form, factorized conditional distributions, enabling efficient exact Gibbs sampling update.
We validate our algorithm on benchmark quantum physics problems, accurately reproducing known theoretical results and matching traditional $O(N^3)$ algorithms on $32\times 32$ lattice simulations at a fraction of the wall-clock time, empirically demonstrating $N \log N$ scaling. By reformulating a long-standing physics simulation problem in machine learning language, our work provides a powerful tool for large-scale probabilistic inference and opens avenues for physics-inspired generative models. 

---
# Deep Edge Filter: Return of the Human-Crafted Layer in Deep Learning 

**Authors**: Dongkwan Lee, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13865)  

**Abstract**: We introduce the Deep Edge Filter, a novel approach that applies high-pass filtering to deep neural network features to improve model generalizability. Our method is motivated by our hypothesis that neural networks encode task-relevant semantic information in high-frequency components while storing domain-specific biases in low-frequency components of deep features. By subtracting low-pass filtered outputs from original features, our approach isolates generalizable representations while preserving architectural integrity. Experimental results across diverse domains such as Vision, Text, 3D, and Audio demonstrate consistent performance improvements regardless of model architecture and data modality. Analysis reveals that our method induces feature sparsification and effectively isolates high-frequency components, providing empirical validation of our core hypothesis. The code is available at this https URL. 

---
# Self-Training with Dynamic Weighting for Robust Gradual Domain Adaptation 

**Authors**: Zixi Wang, Yushe Cao, Yubo Huang, Jinzhu Wei, Jingzehua Xu, Shuai Zhang, Xin Lai  

**Link**: [PDF](https://arxiv.org/pdf/2510.13864)  

**Abstract**: In this paper, we propose a new method called Self-Training with Dynamic Weighting (STDW), which aims to enhance robustness in Gradual Domain Adaptation (GDA) by addressing the challenge of smooth knowledge migration from the source to the target domain. Traditional GDA methods mitigate domain shift through intermediate domains and self-training but often suffer from inefficient knowledge migration or incomplete intermediate data. Our approach introduces a dynamic weighting mechanism that adaptively balances the loss contributions of the source and target domains during training. Specifically, we design an optimization framework governed by a time-varying hyperparameter $\varrho$ (progressing from 0 to 1), which controls the strength of domain-specific learning and ensures stable adaptation. The method leverages self-training to generate pseudo-labels and optimizes a weighted objective function for iterative model updates, maintaining robustness across intermediate domains. Experiments on rotated MNIST, color-shifted MNIST, portrait datasets, and the Cover Type dataset demonstrate that STDW outperforms existing baselines. Ablation studies further validate the critical role of $\varrho$'s dynamic scheduling in achieving progressive adaptation, confirming its effectiveness in reducing domain bias and improving generalization. This work provides both theoretical insights and a practical framework for robust gradual domain adaptation, with potential applications in dynamic real-world scenarios. The code is available at this https URL. 

---
# Ensembling Large Language Models to Characterize Affective Dynamics in Student-AI Tutor Dialogues 

**Authors**: Chenyu Zhang, Sharifa Alghowinem, Cynthia Breazeal  

**Link**: [PDF](https://arxiv.org/pdf/2510.13862)  

**Abstract**: While recent studies have examined the leaning impact of large language model (LLM) in educational contexts, the affective dynamics of LLM-mediated tutoring remain insufficiently understood. This work introduces the first ensemble-LLM framework for large-scale affect sensing in tutoring dialogues, advancing the conversation on responsible pathways for integrating generative AI into education by attending to learners' evolving affective states. To achieve this, we analyzed two semesters' worth of 16,986 conversational turns exchanged between PyTutor, an LLM-powered AI tutor, and 261 undergraduate learners across three U.S. institutions. To investigate learners' emotional experiences, we generate zero-shot affect annotations from three frontier LLMs (Gemini, GPT-4o, Claude), including scalar ratings of valence, arousal, and learning-helpfulness, along with free-text emotion labels. These estimates are fused through rank-weighted intra-model pooling and plurality consensus across models to produce robust emotion profiles. Our analysis shows that during interaction with the AI tutor, students typically report mildly positive affect and moderate arousal. Yet learning is not uniformly smooth: confusion and curiosity are frequent companions to problem solving, and frustration, while less common, still surfaces in ways that can derail progress. Emotional states are short-lived--positive moments last slightly longer than neutral or negative ones, but they are fragile and easily disrupted. Encouragingly, negative emotions often resolve quickly, sometimes rebounding directly into positive states. Neutral moments frequently act as turning points, more often steering students upward than downward, suggesting opportunities for tutors to intervene at precisely these junctures. 

---
# ShishuLM: Lightweight Language Model with Hybrid Decoder-MLP Architecture and Paired Weight Sharing 

**Authors**: Shivanshu Kumar, Gopalakrishnan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13860)  

**Abstract**: While the transformer architecture has achieved state-of-the-art performance on natural language processing tasks, these models impose substantial memory and computational overhead. Recent research has identified significant architectural redundancies within these models, presenting opportunities for optimization without compromising performance. Taking insights from research in AI interpretability and inference-time layer pruning, we introduce an efficient language model architecture, referred to as ShishuLM, which reduces both the parameter count and Key-Value (KV) cache requirements. Given the increasing importance of Small Language Models (SLMs) in agentic AI systems, we evaluate our approach on two SLMs of different scales. Our analysis reveals that for moderate-context scenarios, normalization coupled with attention computation is roughly linear with the input, enabling entire transformer blocks to be approximated through Multi-Layer Perceptrons (MLPs). Our results show that ShishuLM provides up to 25% reduction in memory requirements and up to 40% improvement in latency during both training and inference, compared to parent models. Our experimental and analytical findings provide insights towards building more efficient SLM architectures from a pre-training standpoint. 

---
# Benchmarking Correctness and Security in Multi-Turn Code Generation 

**Authors**: Ruchit Rawal, Jeffrey Yang Fan Chiang, Chihao Shen, Jeffery Siyuan Tian, Aastha Mahajan, Tom Goldstein, Yizheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13859)  

**Abstract**: AI coding assistants powered by large language models (LLMs) have transformed software development, significantly boosting productivity. While existing benchmarks evaluate the correctness and security of LLM-generated code, they are typically limited to single-turn tasks that do not reflect the iterative nature of real-world development. We introduce MT-Sec, the first benchmark to systematically evaluate both correctness and security in multi-turn coding scenarios. We construct this using a synthetic data pipeline that transforms existing single-turn tasks into semantically aligned multi-turn interaction sequences, allowing reuse of original test suites while modeling the complexity of real-world coding processes. We evaluate 32 open- and closed-source models, and three agent-scaffolding on MT-Sec and observe a consistent 20-27% drop in "correct and secure" outputs from single-turn to multi-turn settings -- even among state-of-the-art models. Beyond full-program generation, we also evaluate models on multi-turn code-diff generation -- an unexplored yet practically relevant setting -- and find that models perform worse here, with increased rates of functionally incorrect and insecure outputs. Finally, we find that while agent scaffoldings boost single-turn code generation performance, they are not quite as effective in multi-turn evaluations. Together, these findings highlight the need for benchmarks that jointly evaluate correctness and security in multi-turn, real-world coding workflows. 

---
# From Craft to Constitution: A Governance-First Paradigm for Principled Agent Engineering 

**Authors**: Qiang Xu, Xiangyu Wen, Changran Xu, Zeju Li, Jianyuan Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.13857)  

**Abstract**: The advent of powerful Large Language Models (LLMs) has ushered in an ``Age of the Agent,'' enabling autonomous systems to tackle complex goals. However, the transition from prototype to production is hindered by a pervasive ``crisis of craft,'' resulting in agents that are brittle, unpredictable, and ultimately untrustworthy in mission-critical applications. This paper argues this crisis stems from a fundamental paradigm mismatch -- attempting to command inherently probabilistic processors with the deterministic mental models of traditional software engineering. To solve this crisis, we introduce a governance-first paradigm for principled agent engineering, embodied in a formal architecture we call ArbiterOS. 

---
# Multimodal Retrieval-Augmented Generation with Large Language Models for Medical VQA 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2510.13856)  

**Abstract**: Medical Visual Question Answering (MedVQA) enables natural language queries over medical images to support clinical decision-making and patient care. The MEDIQA-WV 2025 shared task addressed wound-care VQA, requiring systems to generate free-text responses and structured wound attributes from images and patient queries. We present the MasonNLP system, which employs a general-domain, instruction-tuned large language model with a retrieval-augmented generation (RAG) framework that incorporates textual and visual examples from in-domain data. This approach grounds outputs in clinically relevant exemplars, improving reasoning, schema adherence, and response quality across dBLEU, ROUGE, BERTScore, and LLM-based metrics. Our best-performing system ranked 3rd among 19 teams and 51 submissions with an average score of 41.37%, demonstrating that lightweight RAG with general-purpose LLMs -- a minimal inference-time layer that adds a few relevant exemplars via simple indexing and fusion, with no extra training or complex re-ranking -- provides a simple and effective baseline for multimodal clinical NLP tasks. 

---
# Harnessing Consistency for Robust Test-Time LLM Ensemble 

**Authors**: Zhichen Zeng, Qi Yu, Xiao Lin, Ruizhong Qiu, Xuying Ning, Tianxin Wei, Yuchen Yan, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.13855)  

**Abstract**: Different large language models (LLMs) exhibit diverse strengths and weaknesses, and LLM ensemble serves as a promising approach to integrate their complementary capabilities. Despite substantial progress in improving ensemble quality, limited attention has been paid to the robustness of ensembles against potential erroneous signals, which often arise from heterogeneous tokenization schemes and varying model expertise. Our analysis shows that ensemble failures typically arise from both the token level and the model level: the former reflects severe disagreement in token predictions, while the latter involves low confidence and pronounced disparities among models. In light of this, we propose CoRE, a plug-and-play technique that harnesses model consistency for robust LLM ensemble, which can be seamlessly integrated with diverse ensemble methods. Token-level consistency captures fine-grained disagreements by applying a low-pass filter to downweight uncertain tokens with high inconsistency, often due to token misalignment, thereby improving robustness at a granular level. Model-level consistency models global agreement by promoting model outputs with high self-confidence and minimal divergence from others, enhancing robustness at a coarser level. Extensive experiments across diverse benchmarks, model combinations, and ensemble strategies demonstrate that CoRE consistently improves ensemble performance and robustness. 

---
# BenchPress: A Human-in-the-Loop Annotation System for Rapid Text-to-SQL Benchmark Curation 

**Authors**: Fabian Wenz, Omar Bouattour, Devin Yang, Justin Choi, Cecil Gregg, Nesime Tatbul, Çağatay Demiralp  

**Link**: [PDF](https://arxiv.org/pdf/2510.13853)  

**Abstract**: Large language models (LLMs) have been successfully applied to many tasks, including text-to-SQL generation. However, much of this work has focused on publicly available datasets, such as Fiben, Spider, and Bird. Our earlier work showed that LLMs are much less effective in querying large private enterprise data warehouses and released Beaver, the first private enterprise text-to-SQL benchmark. To create Beaver, we leveraged SQL logs, which are often readily available. However, manually annotating these logs to identify which natural language questions they answer is a daunting task. Asking database administrators, who are highly trained experts, to take on additional work to construct and validate corresponding natural language utterances is not only challenging but also quite costly. To address this challenge, we introduce BenchPress, a human-in-the-loop system designed to accelerate the creation of domain-specific text-to-SQL benchmarks. Given a SQL query, BenchPress uses retrieval-augmented generation (RAG) and LLMs to propose multiple natural language descriptions. Human experts then select, rank, or edit these drafts to ensure accuracy and domain alignment. We evaluated BenchPress on annotated enterprise SQL logs, demonstrating that LLM-assisted annotation drastically reduces the time and effort required to create high-quality benchmarks. Our results show that combining human verification with LLM-generated suggestions enhances annotation accuracy, benchmark reliability, and model evaluation robustness. By streamlining the creation of custom benchmarks, BenchPress offers researchers and practitioners a mechanism for assessing text-to-SQL models on a given domain-specific workload. BenchPress is freely available via our public GitHub repository at this https URL and is also accessible on our website at this http URL. 

---
# ConsistencyAI: A Benchmark to Assess LLMs' Factual Consistency When Responding to Different Demographic Groups 

**Authors**: Peter Banyas, Shristi Sharma, Alistair Simmons, Atharva Vispute  

**Link**: [PDF](https://arxiv.org/pdf/2510.13852)  

**Abstract**: Is an LLM telling you different facts than it's telling me? This paper introduces ConsistencyAI, an independent benchmark for measuring the factual consistency of large language models (LLMs) for different personas. ConsistencyAI tests whether, when users of different demographics ask identical questions, the model responds with factually inconsistent answers. Designed without involvement from LLM providers, this benchmark offers impartial evaluation and accountability. In our experiment, we queried 19 LLMs with prompts that requested 5 facts for each of 15 topics. We repeated this query 100 times for each LLM, each time adding prompt context from a different persona selected from a subset of personas modeling the general population. We processed the responses into sentence embeddings, computed cross-persona cosine similarity, and computed the weighted average of cross-persona cosine similarity to calculate factual consistency scores. In 100-persona experiments, scores ranged from 0.9065 to 0.7896, and the mean was 0.8656, which we adopt as a benchmark threshold. xAI's Grok-3 is most consistent, while several lightweight models rank lowest. Consistency varies by topic: the job market is least consistent, G7 world leaders most consistent, and issues like vaccines or the Israeli-Palestinian conflict diverge by provider. These results show that both the provider and the topic shape the factual consistency. We release our code and interactive demo to support reproducible evaluation and encourage persona-invariant prompting strategies. 

---
# Revisiting the UID Hypothesis in LLM Reasoning Traces 

**Authors**: Minju Gwak, Guijin Son, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13850)  

**Abstract**: Large language models (LLMs) often solve problems using step-by-step Chain-of-Thought (CoT) reasoning, yet these intermediate steps are frequently unfaithful or hard to interpret. Inspired by the Uniform Information Density (UID) hypothesis in psycholinguistics -- which posits that humans communicate by maintaining a stable flow of information -- we introduce entropy-based metrics to analyze the information flow within reasoning traces. Surprisingly, across three challenging mathematical benchmarks, we find that successful reasoning in LLMs is globally non-uniform: correct solutions are characterized by uneven swings in information density, in stark contrast to human communication patterns. This result challenges assumptions about machine reasoning and suggests new directions for designing interpretable and adaptive reasoning models. 

---
# On-device System of Compositional Multi-tasking in Large Language Models 

**Authors**: Ondrej Bohdal, Konstantinos Theodosiadis, Asterios Mpatziakas, Dimitris Filippidis, Iro Spyrou, Christos Zonios, Anastasios Drosou, Dimosthenis Ioannidis, Kyeng-Hun Lee, Jijoong Moon, Hyeonmok Ko, Mete Ozay, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2510.13848)  

**Abstract**: Large language models (LLMs) are commonly adapted for diverse downstream tasks via parameter-efficient fine-tuning techniques such as Low-Rank Adapters (LoRA). While adapters can be combined to handle multiple tasks separately, standard approaches struggle when targeting the simultaneous execution of complex tasks, such as generating a translated summary from a long conversation. To address this challenge, we propose a novel approach tailored specifically for compositional multi-tasking scenarios involving summarization and translation. Our technique involves adding a learnable projection layer on top of the combined summarization and translation adapters. This design enables effective integration while maintaining efficiency through reduced computational overhead compared to alternative strategies requiring extensive retraining or sequential processing. We demonstrate the practical viability of our method within an on-device environment by developing an Android app capable of executing compositional tasks seamlessly. Experimental results indicate our solution performs well and is fast in both cloud-based and on-device implementations, highlighting the potential benefits of adopting our framework in real-world applications demanding high-speed operation alongside resource constraints. 

---
# DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models 

**Authors**: Jinbin Zhang, Nasib Ullah, Erik Schultheis, Rohit Babbar  

**Link**: [PDF](https://arxiv.org/pdf/2510.13847)  

**Abstract**: Speculative decoding (a.k.a. speculative sampling) has become a standard way to accelerate LLM inference: a small drafter proposes multiple tokens and a large target model verifies them once per speculation length. Recently, scaling of the LLM vocabulary has pushed the number of tokens to grow substantially. While verification over the full vocabulary leaves the target model largely unaffected, the O(|V|d) parameters in the drafter's output head become a latency bottleneck, slowing the entire pipeline. Contemporary methods (e.g., FR-Spec, VocabTrim) restrict the drafter's vocabulary to a fixed subset of the target model's vocabulary, ranked in descending order of token frequency. Although this reduces draft-time compute, it is brittle, since: (i) frequency lists are corpus-dependent and require retuning to generalize, and (ii) static shortlists suppress rare or domain-specific tokens, lowering the expected number of tokens per verification step. We propose DynaSpec, a context-dependent dynamic shortlisting mechanism that is robust, speeds up drafting, and generalizes across diverse tasks. Concretely, we introduce lightweight, coarse-grained meta-classifiers that route contexts to a small number of token clusters; the union of the top-k selected clusters forms the drafter's shortlist, while verification retains the full vocabulary and exactness. The meta-classifier finishes its computation earlier than the drafter's hidden state generation by exploiting parallel execution of draft encoding and meta shortlisting on separate streams. On standard speculative-decoding benchmarks, we observe consistent gains in mean accepted length over fixed-shortlist baselines, while context-dependent selection enables smaller shortlists without degrading acceptance. 

---
# Information flow in multilayer perceptrons: an in-depth analysis 

**Authors**: Giuliano Armano  

**Link**: [PDF](https://arxiv.org/pdf/2510.13846)  

**Abstract**: Analysing how information flows along the layers of a multilayer perceptron is a topic of paramount importance in the field of artificial neural networks. After framing the problem from the point of view of information theory, in this position article a specific investigation is conducted on the way information is processed, with particular reference to the requirements imposed by supervised learning. To this end, the concept of information matrix is devised and then used as formal framework for understanding the aetiology of optimisation strategies and for studying the information flow. The underlying research for this article has also produced several key outcomes: i) the definition of a parametric optimisation strategy, ii) the finding that the optimisation strategy proposed in the information bottleneck framework shares strong similarities with the one derived from the information matrix, and iii) the insight that a multilayer perceptron serves as a kind of "adaptor", meant to process the input according to the given objective. 

---
# Serialized EHR make for good text representations 

**Authors**: Zhirong Chou, Quan Qin, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13843)  

**Abstract**: The emergence of foundation models in healthcare has opened new avenues for learning generalizable representations from large scale clinical data. Yet, existing approaches often struggle to reconcile the tabular and event based nature of Electronic Health Records (EHRs) with the sequential priors of natural language models. This structural mismatch limits their ability to capture longitudinal dependencies across patient encounters. We introduce SerialBEHRT, a domain aligned foundation model that extends SciBERT through additional pretraining on structured EHR sequences. SerialBEHRT is designed to encode temporal and contextual relationships among clinical events, thereby producing richer patient representations. We evaluate its effectiveness on the task of antibiotic susceptibility prediction, a clinically meaningful problem in antibiotic stewardship. Through extensive benchmarking against state of the art EHR representation strategies, we demonstrate that SerialBEHRT achieves superior and more consistent performance, highlighting the importance of temporal serialization in foundation model pretraining for healthcare. 

---
# ADMIT: Few-shot Knowledge Poisoning Attacks on RAG-based Fact Checking 

**Authors**: Yutao Wu, Xiao Liu, Yinghui Li, Yifeng Gao, Yifan Ding, Jiale Ding, Xiang Zheng, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13842)  

**Abstract**: Knowledge poisoning poses a critical threat to Retrieval-Augmented Generation (RAG) systems by injecting adversarial content into knowledge bases, tricking Large Language Models (LLMs) into producing attacker-controlled outputs grounded in manipulated context. Prior work highlights LLMs' susceptibility to misleading or malicious retrieved content. However, real-world fact-checking scenarios are more challenging, as credible evidence typically dominates the retrieval pool. To investigate this problem, we extend knowledge poisoning to the fact-checking setting, where retrieved context includes authentic supporting or refuting evidence. We propose \textbf{ADMIT} (\textbf{AD}versarial \textbf{M}ulti-\textbf{I}njection \textbf{T}echnique), a few-shot, semantically aligned poisoning attack that flips fact-checking decisions and induces deceptive justifications, all without access to the target LLMs, retrievers, or token-level control. Extensive experiments show that ADMIT transfers effectively across 4 retrievers, 11 LLMs, and 4 cross-domain benchmarks, achieving an average attack success rate (ASR) of 86\% at an extremely low poisoning rate of $0.93 \times 10^{-6}$, and remaining robust even in the presence of strong counter-evidence. Compared with prior state-of-the-art attacks, ADMIT improves ASR by 11.2\% across all settings, exposing significant vulnerabilities in real-world RAG-based fact-checking systems. 

---
# Meronymic Ontology Extraction via Large Language Models 

**Authors**: Dekai Zhang, Simone Conia, Antonio Rago  

**Link**: [PDF](https://arxiv.org/pdf/2510.13839)  

**Abstract**: Ontologies have become essential in today's digital age as a way of organising the vast amount of readily available unstructured text. In providing formal structure to this information, ontologies have immense value and application across various domains, e.g., e-commerce, where countless product listings necessitate proper product organisation. However, the manual construction of these ontologies is a time-consuming, expensive and laborious process. In this paper, we harness the recent advancements in large language models (LLMs) to develop a fully-automated method of extracting product ontologies, in the form of meronymies, from raw review texts. We demonstrate that the ontologies produced by our method surpass an existing, BERT-based baseline when evaluating using an LLM-as-a-judge. Our investigation provides the groundwork for LLMs to be used more generally in (product or otherwise) ontology extraction. 

---
# Seeing Hate Differently: Hate Subspace Modeling for Culture-Aware Hate Speech Detection 

**Authors**: Weibin Cai, Reza Zafarani  

**Link**: [PDF](https://arxiv.org/pdf/2510.13837)  

**Abstract**: Hate speech detection has been extensively studied, yet existing methods often overlook a real-world complexity: training labels are biased, and interpretations of what is considered hate vary across individuals with different cultural backgrounds. We first analyze these challenges, including data sparsity, cultural entanglement, and ambiguous labeling. To address them, we propose a culture-aware framework that constructs individuals' hate subspaces. To alleviate data sparsity, we model combinations of cultural attributes. For cultural entanglement and ambiguous labels, we use label propagation to capture distinctive features of each combination. Finally, individual hate subspaces, which in turn can further enhance classification performance. Experiments show our method outperforms state-of-the-art by 1.05\% on average across all metrics. 

---
# SIMBA UQ: Similarity-Based Aggregation for Uncertainty Quantification in Large Language Models 

**Authors**: Debarun Bhattacharjya, Balaji Ganesan, Junkyu Lee, Radu Marinescu, Katsiaryna Mirylenka, Michael Glass, Xiao Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.13836)  

**Abstract**: When does a large language model (LLM) know what it does not know? Uncertainty quantification (UQ) provides measures of uncertainty, such as an estimate of the confidence in an LLM's generated output, and is therefore increasingly recognized as a crucial component of trusted AI systems. Black-box UQ methods do not require access to internal model information from the generating LLM and therefore have numerous real-world advantages, such as robustness to system changes, adaptability to choice of LLM, reduced costs, and computational tractability. In this paper, we investigate the effectiveness of UQ techniques that are primarily but not necessarily entirely black-box, where the consistency between a generated output and other sampled generations is used as a proxy for confidence in its correctness. We propose a high-level non-verbalized similarity-based aggregation framework that subsumes a broad swath of UQ approaches suitable for complex generative tasks, as well as introduce specific novel techniques from the framework that train confidence estimation models using small training sets. Through an empirical study with datasets spanning the diverse tasks of question answering, summarization, and text-to-SQL, we demonstrate that our proposed similarity-based methods can yield better calibrated confidences than baselines. 

---
# ConDABench: Interactive Evaluation of Language Models for Data Analysis 

**Authors**: Avik Dutta, Priyanshu Gupta, Hosein Hasanbeig, Rahul Pratap Singh, Harshit Nigam, Sumit Gulwani, Arjun Radhakrishna, Gustavo Soares, Ashish Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2510.13835)  

**Abstract**: Real-world data analysis tasks often come with under-specified goals and unclean data. User interaction is necessary to understand and disambiguate a user's intent, and hence, essential to solving these complex tasks. Existing benchmarks for evaluating LLMs on data analysis tasks do not capture these complexities or provide first-class support for interactivity. We introduce ConDABench, a framework for generating conversational data analysis (ConDA) benchmarks and evaluating external tools on the generated benchmarks. \bench consists of (a) a multi-agent workflow for generating realistic benchmarks from articles describing insights gained from public datasets, (b) 1,420 ConDA problems generated using this workflow, and (c) an evaluation harness that, for the first time, makes it possible to systematically evaluate conversational data analysis tools on the generated ConDA problems. Evaluation of state-of-the-art LLMs on the benchmarks reveals that while the new generation of models are better at solving more instances, they are not necessarily better at solving tasks that require sustained, long-form engagement. ConDABench is an avenue for model builders to measure progress towards truly collaborative models that can complete complex interactive tasks. 

---
# Entropy Meets Importance: A Unified Head Importance-Entropy Score for Stable and Efficient Transformer Pruning 

**Authors**: Minsik Choi, Hyegang Son, Changhoon Kim, Young Geun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13832)  

**Abstract**: Transformer-based models have achieved remarkable performance in NLP tasks. However, their structural characteristics-multiple layers and attention heads-introduce efficiency challenges in inference and deployment. To address these challenges, various pruning methods have recently been proposed. Notably, gradient-based methods using Head Importance Scores (HIS) have gained traction for interpretability, efficiency, and ability to identify redundant heads. However, HIS alone has limitations as it captures only the gradient-driven contribution, overlooking the diversity of attention patterns. To overcome these limitations, we introduce a novel pruning criterion, HIES (Head Importance-Entropy Score), which integrates head importance scores with attention entropy, providing complementary evidence on per-head contribution. Empirically, HIES-based pruning yields up to 15.2% improvement in model quality and 2.04x improvement in stability over HIS-only methods, enabling substantial model compression without sacrificing either accuracy or stability. Code will be released upon publication. 

---
# Informed Routing in LLMs: Smarter Token-Level Computation for Faster Inference 

**Authors**: Chao Han, Yijuan Liang, Zihao Xuan, Daokuan Wu, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13831)  

**Abstract**: The deployment of large language models (LLMs) in real-world applications is increasingly limited by their high inference cost. While recent advances in dynamic token-level computation allocation attempt to improve efficiency by selectively activating model components per token, existing methods rely on greedy routing--a myopic execute-or-skip mechanism that often leads to irreversible information loss and suboptimal token selection. This paper introduces informed routing, a new paradigm that proactively addresses these issues. The key insight is to assess not only a token's immediate importance but also its recoverability, i.e., how well its transformation can be approximated. To this end, we propose the Lightweight Feature Forecaster (LFF), a small predictive module that estimates a unit's output before routing decisions are made. This enables a flexible execute-or-approximate policy that preserves model fidelity while drastically reducing computation. Extensive experiments on both language modeling and reasoning tasks show that informed routing achieves state-of-the-art efficiency-performance trade-offs across multiple sparsity levels. Notably, even without final LoRA fine-tuning, our method matches or surpasses strong baselines that require full fine-tuning, all while reducing training time by over 50%. The code is available at: this https URL 

---
# Users as Annotators: LLM Preference Learning from Comparison Mode 

**Authors**: Zhongze Cai, Xiaocheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13830)  

**Abstract**: Pairwise preference data have played an important role in the alignment of large language models (LLMs). Each sample of such data consists of a prompt, two different responses to the prompt, and a binary label indicating which of the two responses is better. The labels are usually annotated by professional human annotators. In this paper, we consider an alternative approach to collect pairwise preference data -- user annotation from comparison mode. With the increasingly wider adoption of LLMs among the population, users are contributing more and more of their preference labels through their daily interactions with the LLMs. The upside of such labels is that users are the best experts in judging the responses to their own queries/prompts, but the downside is the lack of quality control in these labels. In this paper, we consider a new idea of generating two responses from two different models or two different versions of the same model. The asymmetry allows us to make an inference of the user's data quality through our proposed user behavior model. We develop an expectation-maximization algorithm to estimate a latent quality factor of the user, and filter users' annotation data accordingly. The downstream task shows the effectiveness of our approach in both capturing the user behavior and data filtering for LLM alignment. 

---
# A Linguistics-Aware LLM Watermarking via Syntactic Predictability 

**Authors**: Shinwoo Park, Hyejin Park, Hyeseon Ahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.13829)  

**Abstract**: As large language models (LLMs) continue to advance rapidly, reliable governance tools have become critical. Publicly verifiable watermarking is particularly essential for fostering a trustworthy AI ecosystem. A central challenge persists: balancing text quality against detection robustness. Recent studies have sought to navigate this trade-off by leveraging signals from model output distributions (e.g., token-level entropy); however, their reliance on these model-specific signals presents a significant barrier to public verification, as the detection process requires access to the logits of the underlying model. We introduce STELA, a novel framework that aligns watermark strength with the linguistic degrees of freedom inherent in language. STELA dynamically modulates the signal using part-of-speech (POS) n-gram-modeled linguistic indeterminacy, weakening it in grammatically constrained contexts to preserve quality and strengthen it in contexts with greater linguistic flexibility to enhance detectability. Our detector operates without access to any model logits, thus facilitating publicly verifiable detection. Through extensive experiments on typologically diverse languages-analytic English, isolating Chinese, and agglutinative Korean-we show that STELA surpasses prior methods in detection robustness. Our code is available at this https URL. 

---
# Bridging the Semantic Gap: Contrastive Rewards for Multilingual Text-to-SQL 

**Authors**: Ashish Kattamuri, Ishita Prasad, Meetu Malhotra, Arpita Vats, Rahul Raja, Albert Lie  

**Link**: [PDF](https://arxiv.org/pdf/2510.13827)  

**Abstract**: Current Text-to-SQL methods are evaluated and only focused on executable queries, overlooking the semantic alignment challenge -- both in terms of the semantic meaning of the query and the correctness of the execution results. Even execution accuracy itself shows significant drops when moving from English to other languages, with an average decline of 6 percentage points across non-English languages. We address these challenges by presenting a new framework that combines Group Relative Policy Optimization (GRPO) within a multilingual contrastive reward signal to enhance both task efficiency and semantic accuracy in Text-to-SQL systems in cross-lingual scenarios. Our method teaches models to obtain better correspondence between SQL generation and user intent by combining a reward signal based on semantic similarity. On the seven-language MultiSpider dataset, fine-tuning the LLaMA-3-3B model with GRPO improved the execution accuracy up to 87.4 percent (+26 pp over zero-shot) and semantic accuracy up to 52.29 percent (+32.86 pp). Adding our contrastive reward signal in the GRPO framework further improved the average semantic accuracy to 59.14 percent (+6.85 pp, up to +10 pp for Vietnamese). Our experiments showcase that a smaller, parameter-efficient 3B LLaMA model fine-tuned with our contrastive reward signal outperforms a much larger zero-shot 8B LLaMA model, with an uplift of 7.43 pp in execution accuracy (from 81.43 percent on the 8B model to 88.86 percent on the 3B model), and nearly matches its semantic accuracy (59.14 percent vs. 68.57 percent) -- all using just 3,000 reinforcement learning training examples. These results demonstrate how we can improve the performance of Text-to-SQL systems with contrastive rewards for directed semantic alignment, without requiring large-scale training datasets. 

---
# Towards Neurocognitive-Inspired Intelligence: From AI's Structural Mimicry to Human-Like Functional Cognition 

**Authors**: Noorbakhsh Amiri Golilarz, Hassan S. Al Khatib, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13826)  

**Abstract**: Artificial intelligence has advanced significantly through deep learning, reinforcement learning, and large language and vision models. However, these systems often remain task specific, struggle to adapt to changing conditions, and cannot generalize in ways similar to human cognition. Additionally, they mainly focus on mimicking brain structures, which often leads to black-box models with limited transparency and adaptability. Inspired by the structure and function of biological cognition, this paper introduces the concept of "Neurocognitive-Inspired Intelligence (NII)," a hybrid approach that combines neuroscience, cognitive science, computer vision, and AI to develop more general, adaptive, and robust intelligent systems capable of rapid learning, learning from less data, and leveraging prior experience. These systems aim to emulate the human brain's ability to flexibly learn, reason, remember, perceive, and act in real-world settings with minimal supervision. We review the limitations of current AI methods, define core principles of neurocognitive-inspired intelligence, and propose a modular, biologically inspired architecture that emphasizes integration, embodiment, and adaptability. We also discuss potential implementation strategies and outline various real-world applications, from robotics to education and healthcare. Importantly, this paper offers a hybrid roadmap for future research, laying the groundwork for building AI systems that more closely resemble human cognition. 

---
# A2AS: Agentic AI Runtime Security and Self-Defense 

**Authors**: Eugene Neelou, Ivan Novikov, Max Moroz, Om Narayan, Tiffany Saade, Mika Ayenson, Ilya Kabanov, Jen Ozmen, Edward Lee, Vineeth Sai Narajala, Emmanuel Guilherme Junior, Ken Huang, Huseyin Gulsin, Jason Ross, Marat Vyshegorodtsev, Adelin Travers, Idan Habler, Rahul Jadav  

**Link**: [PDF](https://arxiv.org/pdf/2510.13825)  

**Abstract**: The A2AS framework is introduced as a security layer for AI agents and LLM-powered applications, similar to how HTTPS secures HTTP. A2AS enforces certified behavior, activates model self-defense, and ensures context window integrity. It defines security boundaries, authenticates prompts, applies security rules and custom policies, and controls agentic behavior, enabling a defense-in-depth strategy. The A2AS framework avoids latency overhead, external dependencies, architectural changes, model retraining, and operational complexity. The BASIC security model is introduced as the A2AS foundation: (B) Behavior certificates enable behavior enforcement, (A) Authenticated prompts enable context window integrity, (S) Security boundaries enable untrusted input isolation, (I) In-context defenses enable secure model reasoning, (C) Codified policies enable application-specific rules. This first paper in the series introduces the BASIC security model and the A2AS framework, exploring their potential toward establishing the A2AS industry standard. 

---
# Leveraging Wireless Sensor Networks for Real-Time Monitoring and Control of Industrial Environments 

**Authors**: Muhammad Junaid Asif, Shazia Saqib, Rana Fayyaz Ahmad, Hamza Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13820)  

**Abstract**: This research proposes an extensive technique for monitoring and controlling the industrial parameters using Internet of Things (IoT) technology based on wireless communication. We proposed a system based on NRF transceivers to establish a strong Wireless Sensor Network (WSN), enabling transfer of real-time data from multiple sensors to a central setup that is driven by ARDUINO microcontrollers. Different key parameters, crucial for industrial setup such as temperature, humidity, soil moisture and fire detection, are monitored and displayed on an LCD screen, enabling factory administration to oversee the industrial operations remotely over the internet. Our proposed system bypasses the need for physical presence for monitoring by addressing the shortcomings of conventional wired communication systems. Other than monitoring, there is an additional feature to remotely control these parameters by controlling the speed of DC motors through online commands. Given the rising incidence of industrial fires over the worldwide between 2020 and 2024 due to an array of hazards, this system with dual functionality boosts the overall operational efficiency and safety. This overall integration of IoT and Wireless Sensor Network (WSN) reduces the potential risks linked with physical monitoring, providing rapid responses in emergency scenarios, including the activation of firefighting equipment. The results show that innovations in wireless communication perform an integral part in industrial process automation and safety, paving the way to more intelligent and responsive operating environments. Overall, this study highlights the potential for change of IoT-enabled systems to revolutionize monitoring and control in a variety of industrial applications, resulting in increased productivity and safety. 

---
# GQVis: A Dataset of Genomics Data Questions and Visualizations for Generative AI 

**Authors**: Skylar Sargent Walters, Arthea Valderrama, Thomas C. Smits, David Kouřil, Huyen N. Nguyen, Sehi L'Yi, Devin Lange, Nils Gehlenborg  

**Link**: [PDF](https://arxiv.org/pdf/2510.13816)  

**Abstract**: Data visualization is a fundamental tool in genomics research, enabling the exploration, interpretation, and communication of complex genomic features. While machine learning models show promise for transforming data into insightful visualizations, current models lack the training foundation for domain-specific tasks. In an effort to provide a foundational resource for genomics-focused model training, we present a framework for generating a dataset that pairs abstract, low-level questions about genomics data with corresponding visualizations. Building on prior work with statistical plots, our approach adapts to the complexity of genomics data and the specialized representations used to depict them. We further incorporate multiple linked queries and visualizations, along with justifications for design choices, figure captions, and image alt-texts for each item in the dataset. We use genomics data retrieved from three distinct genomics data repositories (4DN, ENCODE, Chromoscope) to produce GQVis: a dataset consisting of 1.14 million single-query data points, 628k query pairs, and 589k query chains. The GQVis dataset and generation code are available at this https URL and this https URL. 

---
# Reversing the Lens: Using Explainable AI to Understand Human Expertise 

**Authors**: Roussel Rahman, Aashwin Ananda Mishra, Wan-Lin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13814)  

**Abstract**: Both humans and machine learning models learn from experience, particularly in safety- and reliability-critical domains. While psychology seeks to understand human cognition, the field of Explainable AI (XAI) develops methods to interpret machine learning models. This study bridges these domains by applying computational tools from XAI to analyze human learning. We modeled human behavior during a complex real-world task -- tuning a particle accelerator -- by constructing graphs of operator subtasks. Applying techniques such as community detection and hierarchical clustering to archival operator data, we reveal how operators decompose the problem into simpler components and how these problem-solving structures evolve with expertise. Our findings illuminate how humans develop efficient strategies in the absence of globally optimal solutions, and demonstrate the utility of XAI-based methods for quantitatively studying human cognition. 

---
# Generative AI in Heritage Practice: Improving the Accessibility of Heritage Guidance 

**Authors**: Jessica Witte, Edmund Lee, Lisa Brausem, Verity Shillabeer, Chiara Bonacchi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13811)  

**Abstract**: This paper discusses the potential for integrating Generative Artificial Intelligence (GenAI) into professional heritage practice with the aim of enhancing the accessibility of public-facing guidance documents. We developed HAZEL, a GenAI chatbot fine-tuned to assist with revising written guidance relating to heritage conservation and interpretation. Using quantitative assessments, we compare HAZEL's performance to that of ChatGPT (GPT-4) in a series of tasks related to the guidance writing process. The results of this comparison indicate a slightly better performance of HAZEL over ChatGPT, suggesting that the GenAI chatbot is more effective once the underlying large language model (LLM) has been fine-tuned. However, we also note significant limitations, particularly in areas requiring cultural sensitivity and more advanced technical expertise. These findings suggest that, while GenAI cannot replace human heritage professionals in technical authoring tasks, its potential to automate and expedite certain aspects of guidance writing could offer valuable benefits to heritage organisations, especially in resource-constrained contexts. 

---
