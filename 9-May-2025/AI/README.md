# Conversational Process Model Redesign 

**Authors**: Nataliia Klievtsova, Timotheus Kampik, Juergen Mangler, Stefanie Rinderle-Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.05453)  

**Abstract**: With the recent success of large language models (LLMs), the idea of AI-augmented Business Process Management systems is becoming more feasible. One of their essential characteristics is the ability to be conversationally actionable, allowing humans to interact with the LLM effectively to perform crucial process life cycle tasks such as process model design and redesign. However, most current research focuses on single-prompt execution and evaluation of results, rather than on continuous interaction between the user and the LLM. In this work, we aim to explore the feasibility of using LLMs to empower domain experts in the creation and redesign of process models in an iterative and effective way. The proposed conversational process model redesign (CPD) approach receives as input a process model and a redesign request by the user in natural language. Instead of just letting the LLM make changes, the LLM is employed to (a) identify process change patterns from literature, (b) re-phrase the change request to be aligned with an expected wording for the identified pattern (i.e., the meaning), and then to (c) apply the meaning of the change to the process model. This multi-step approach allows for explainable and reproducible changes. In order to ensure the feasibility of the CPD approach, and to find out how well the patterns from literature can be handled by the LLM, we performed an extensive evaluation. The results show that some patterns are hard to understand by LLMs and by users. Within the scope of the study, we demonstrated that users need support to describe the changes clearly. Overall the evaluation shows that the LLMs can handle most changes well according to a set of completeness and correctness criteria. 

---
# EcoAgent: An Efficient Edge-Cloud Collaborative Multi-Agent Framework for Mobile Automation 

**Authors**: Biao Yi, Xavier Hu, Yurun Chen, Shengyu Zhang, Hongxia Yang, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05440)  

**Abstract**: Cloud-based mobile agents powered by (multimodal) large language models ((M)LLMs) offer strong reasoning abilities but suffer from high latency and cost. While fine-tuned (M)SLMs enable edge deployment, they often lose general capabilities and struggle with complex tasks. To address this, we propose EcoAgent, an Edge-Cloud cOllaborative multi-agent framework for mobile automation. EcoAgent features a closed-loop collaboration among a cloud-based Planning Agent and two edge-based agents: the Execution Agent for action execution and the Observation Agent for verifying outcomes. The Observation Agent uses a Pre-Understanding Module to compress screen images into concise text, reducing token usage. In case of failure, the Planning Agent retrieves screen history and replans via a Reflection Module. Experiments on AndroidWorld show that EcoAgent maintains high task success rates while significantly reducing MLLM token consumption, enabling efficient and practical mobile automation. 

---
# A Pain Assessment Framework based on multimodal data and Deep Machine Learning methods 

**Authors**: Stefanos Gkikas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05396)  

**Abstract**: From the original abstract:
This thesis initially aims to study the pain assessment process from a clinical-theoretical perspective while exploring and examining existing automatic approaches. Building on this foundation, the primary objective of this Ph.D. project is to develop innovative computational methods for automatic pain assessment that achieve high performance and are applicable in real clinical settings. A primary goal is to thoroughly investigate and assess significant factors, including demographic elements that impact pain perception, as recognized in pain research, through a computational standpoint. Within the limits of the available data in this research area, our goal was to design, develop, propose, and offer automatic pain assessment pipelines for unimodal and multimodal configurations that are applicable to the specific requirements of different scenarios. The studies published in this Ph.D. thesis showcased the effectiveness of the proposed methods, achieving state-of-the-art results. Additionally, they paved the way for exploring new approaches in artificial intelligence, foundation models, and generative artificial intelligence. 

---
# Advancing Neural Network Verification through Hierarchical Safety Abstract Interpretation 

**Authors**: Luca Marzari, Isabella Mastroeni, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.05235)  

**Abstract**: Traditional methods for formal verification (FV) of deep neural networks (DNNs) are constrained by a binary encoding of safety properties, where a model is classified as either safe or unsafe (robust or not robust). This binary encoding fails to capture the nuanced safety levels within a model, often resulting in either overly restrictive or too permissive requirements. In this paper, we introduce a novel problem formulation called Abstract DNN-Verification, which verifies a hierarchical structure of unsafe outputs, providing a more granular analysis of the safety aspect for a given DNN. Crucially, by leveraging abstract interpretation and reasoning about output reachable sets, our approach enables assessing multiple safety levels during the FV process, requiring the same (in the worst case) or even potentially less computational effort than the traditional binary verification approach. Specifically, we demonstrate how this formulation allows rank adversarial inputs according to their abstract safety level violation, offering a more detailed evaluation of the model's safety and robustness. Our contributions include a theoretical exploration of the relationship between our novel abstract safety formulation and existing approaches that employ abstract interpretation for robustness verification, complexity analysis of the novel problem introduced, and an empirical evaluation considering both a complex deep reinforcement learning task (based on Habitat 3.0) and standard DNN-Verification benchmarks. 

---
# ChemRxivQuest: A Curated Chemistry Question-Answer Database Extracted from ChemRxiv Preprints 

**Authors**: Mahmoud Amiri, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.05232)  

**Abstract**: The rapid expansion of chemistry literature poses significant challenges for researchers seeking to efficiently access domain-specific knowledge. To support advancements in chemistry-focused natural language processing (NLP), we present ChemRxivQuest, a curated dataset of 970 high-quality question-answer (QA) pairs derived from 155 ChemRxiv preprints across 17 subfields of chemistry. Each QA pair is explicitly linked to its source text segment to ensure traceability and contextual accuracy. ChemRxivQuest was constructed using an automated pipeline that combines optical character recognition (OCR), GPT-4o-based QA generation, and a fuzzy matching technique for answer verification. The dataset emphasizes conceptual, mechanistic, applied, and experimental questions, enabling applications in retrieval-based QA systems, search engine development, and fine-tuning of domain-adapted large language models. We analyze the dataset's structure, coverage, and limitations, and outline future directions for expansion and expert validation. ChemRxivQuest provides a foundational resource for chemistry NLP research, education, and tool development. 

---
# Societal and technological progress as sewing an ever-growing, ever-changing, patchy, and polychrome quilt 

**Authors**: Joel Z. Leibo, Alexander Sasha Vezhnevets, William A. Cunningham, Sébastien Krier, Manfred Diaz, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2505.05197)  

**Abstract**: Artificial Intelligence (AI) systems are increasingly placed in positions where their decisions have real consequences, e.g., moderating online spaces, conducting research, and advising on policy. Ensuring they operate in a safe and ethically acceptable fashion is thus critical. However, most solutions have been a form of one-size-fits-all "alignment". We are worried that such systems, which overlook enduring moral diversity, will spark resistance, erode trust, and destabilize our institutions. This paper traces the underlying problem to an often-unstated Axiom of Rational Convergence: the idea that under ideal conditions, rational agents will converge in the limit of conversation on a single ethics. Treating that premise as both optional and doubtful, we propose what we call the appropriateness framework: an alternative approach grounded in conflict theory, cultural evolution, multi-agent systems, and institutional economics. The appropriateness framework treats persistent disagreement as the normal case and designs for it by applying four principles: (1) contextual grounding, (2) community customization, (3) continual adaptation, and (4) polycentric governance. We argue here that adopting these design principles is a good way to shift the main alignment metaphor from moral unification to a more productive metaphor of conflict management, and that taking this step is both desirable and urgent. 

---
# MARK: Memory Augmented Refinement of Knowledge 

**Authors**: Anish Ganguli, Prabal Deb, Debleena Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.05177)  

**Abstract**: Large Language Models (LLMs) assist in specialized tasks but struggle to align with evolving domain knowledge without costly fine-tuning. Domain knowledge consists of: Knowledge: Immutable facts (e.g., 'A stone is solid') and generally accepted principles (e.g., ethical standards); Refined Memory: Evolving insights shaped by business needs and real-world changes. However, a significant gap often exists between a domain expert's deep, nuanced understanding and the system's domain knowledge, which can hinder accurate information retrieval and application. Our Memory-Augmented Refinement of Knowledge (MARK) framework enables LLMs to continuously learn without retraining by leveraging structured refined memory, inspired by the Society of Mind. MARK operates through specialized agents, each serving a distinct role: Residual Refined Memory Agent: Stores and retrieves domain-specific insights to maintain context over time; User Question Refined Memory Agent: Captures user-provided facts, abbreviations, and terminology for better comprehension; LLM Response Refined Memory Agent: Extracts key elements from responses for refinement and personalization. These agents analyse stored refined memory, detect patterns, resolve contradictions, and improve response accuracy. Temporal factors like recency and frequency prioritize relevant information while discarding outdated insights. MARK enhances LLMs in multiple ways: Ground Truth Strategy: Reduces hallucinations by establishing a structured reference; Domain-Specific Adaptation: Essential for fields like healthcare, law, and manufacturing, where proprietary insights are absent from public datasets; Personalized AI Assistants: Improves virtual assistants by remembering user preferences, ensuring coherent responses over time. 

---
# Is there a half-life for the success rates of AI agents? 

**Authors**: Toby Ord  

**Link**: [PDF](https://arxiv.org/pdf/2505.05115)  

**Abstract**: Building on the recent empirical work of Kwa et al. (2025), I show that within their suite of research-engineering tasks the performance of AI agents on longer-duration tasks can be explained by an extremely simple mathematical model -- a constant rate of failing during each minute a human would take to do the task. This implies an exponentially declining success rate with the length of the task and that each agent could be characterised by its own half-life. This empirical regularity allows us to estimate the success rate for an agent at different task lengths. And the fact that this model is a good fit for the data is suggestive of the underlying causes of failure on longer tasks -- that they involve increasingly large sets of subtasks where failing any one fails the task. Whether this model applies more generally on other suites of tasks is unknown and an important subject for further work. 

---
# Multi-agent Embodied AI: Advances and Future Directions 

**Authors**: Zhaohan Feng, Ruiqi Xue, Lei Yuan, Yang Yu, Ning Ding, Meiqin Liu, Bingzhao Gao, Jian Sun, Gang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05108)  

**Abstract**: Embodied artificial intelligence (Embodied AI) plays a pivotal role in the application of advanced technologies in the intelligent era, where AI systems are integrated with physical bodies that enable them to perceive, reason, and interact with their environments. Through the use of sensors for input and actuators for action, these systems can learn and adapt based on real-world feedback, allowing them to perform tasks effectively in dynamic and unpredictable environments. As techniques such as deep learning (DL), reinforcement learning (RL), and large language models (LLMs) mature, embodied AI has become a leading field in both academia and industry, with applications spanning robotics, healthcare, transportation, and manufacturing. However, most research has focused on single-agent systems that often assume static, closed environments, whereas real-world embodied AI must navigate far more complex scenarios. In such settings, agents must not only interact with their surroundings but also collaborate with other agents, necessitating sophisticated mechanisms for adaptation, real-time learning, and collaborative problem-solving. Despite increasing interest in multi-agent systems, existing research remains narrow in scope, often relying on simplified models that fail to capture the full complexity of dynamic, open environments for multi-agent embodied AI. Moreover, no comprehensive survey has systematically reviewed the advancements in this area. As embodied AI rapidly evolves, it is crucial to deepen our understanding of multi-agent embodied AI to address the challenges presented by real-world applications. To fill this gap and foster further development in the field, this paper reviews the current state of research, analyzes key contributions, and identifies challenges and future directions, providing insights to guide innovation and progress in this field. 

---
# A Neuro-Symbolic Framework for Sequence Classification with Relational and Temporal Knowledge 

**Authors**: Luca Salvatore Lorello, Marco Lippi, Stefano Melacci  

**Link**: [PDF](https://arxiv.org/pdf/2505.05106)  

**Abstract**: One of the goals of neuro-symbolic artificial intelligence is to exploit background knowledge to improve the performance of learning tasks. However, most of the existing frameworks focus on the simplified scenario where knowledge does not change over time and does not cover the temporal dimension. In this work we consider the much more challenging problem of knowledge-driven sequence classification where different portions of knowledge must be employed at different timesteps, and temporal relations are available. Our experimental evaluation compares multi-stage neuro-symbolic and neural-only architectures, and it is conducted on a newly-introduced benchmarking framework. Results demonstrate the challenging nature of this novel setting, and also highlight under-explored shortcomings of neuro-symbolic methods, representing a precious reference for future research. 

---
# Enhancing Reinforcement Learning for the Floorplanning of Analog ICs with Beam Search 

**Authors**: Sandro Junior Della Rovere, Davide Basso, Luca Bortolussi, Mirjana Videnovic-Misic, Husni Habal  

**Link**: [PDF](https://arxiv.org/pdf/2505.05059)  

**Abstract**: The layout of analog ICs requires making complex trade-offs, while addressing device physics and variability of the circuits. This makes full automation with learning-based solutions hard to achieve. However, reinforcement learning (RL) has recently reached significant results, particularly in solving the floorplanning problem. This paper presents a hybrid method that combines RL with a beam (BS) strategy. The BS algorithm enhances the agent's inference process, allowing for the generation of flexible floorplans by accomodating various objective weightings, and addressing congestion without without the need for policy retraining or fine-tuning. Moreover, the RL agent's generalization ability stays intact, along with its efficient handling of circuit features and constraints. Experimental results show approx. 5-85% improvement in area, dead space and half-perimeter wire length compared to a standard RL application, along with higher rewards for the agent. Moreover, performance and efficiency align closely with those of existing state-of-the-art techniques. 

---
# A Reputation System for Large Language Model-based Multi-agent Systems to Avoid the Tragedy of the Commons 

**Authors**: Siyue Ren, Wanli Fu, Xinkun Zou, Chen Shen, Yi Cai, Chen Chu, Zhen Wang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05029)  

**Abstract**: The tragedy of the commons, where individual self-interest leads to collectively disastrous outcomes, is a pervasive challenge in human society. Recent studies have demonstrated that similar phenomena can arise in generative multi-agent systems (MASs). To address this challenge, this paper explores the use of reputation systems as a remedy. We propose RepuNet, a dynamic, dual-level reputation framework that models both agent-level reputation dynamics and system-level network evolution. Specifically, driven by direct interactions and indirect gossip, agents form reputations for both themselves and their peers, and decide whether to connect or disconnect other agents for future interactions. Through two distinct scenarios, we show that RepuNet effectively mitigates the 'tragedy of the commons', promoting and sustaining cooperation in generative MASs. Moreover, we find that reputation systems can give rise to rich emergent behaviors in generative MASs, such as the formation of cooperative clusters, the social isolation of exploitative agents, and the preference for sharing positive gossip rather than negative ones. 

---
# Foam-Agent: Towards Automated Intelligent CFD Workflows 

**Authors**: Ling Yue, Nithin Somasekharan, Yadi Cao, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04997)  

**Abstract**: Computational Fluid Dynamics (CFD) is an essential simulation tool in various engineering disciplines, but it often requires substantial domain expertise and manual configuration, creating barriers to entry. We present Foam-Agent, a multi-agent framework that automates complex OpenFOAM-based CFD simulation workflows from natural language inputs. Our innovation includes (1) a hierarchical multi-index retrieval system with specialized indices for different simulation aspects, (2) a dependency-aware file generation system that provides consistency management across configuration files, and (3) an iterative error correction mechanism that diagnoses and resolves simulation failures without human intervention. Through comprehensive evaluation on the dataset of 110 simulation tasks, Foam-Agent achieves an 83.6% success rate with Claude 3.5 Sonnet, significantly outperforming existing frameworks (55.5% for MetaOpenFOAM and 37.3% for OpenFOAM-GPT). Ablation studies demonstrate the critical contribution of each system component, with the specialized error correction mechanism providing a 36.4% performance improvement. Foam-Agent substantially lowers the CFD expertise threshold while maintaining modeling accuracy, demonstrating the potential of specialized multi-agent systems to democratize access to complex scientific simulation tools. The code is public at this https URL 

---
# Position: The AI Conference Peer Review Crisis Demands Author Feedback and Reviewer Rewards 

**Authors**: Jaeho Kim, Yunseok Lee, Seulki Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.04966)  

**Abstract**: The peer review process in major artificial intelligence (AI) conferences faces unprecedented challenges with the surge of paper submissions (exceeding 10,000 submissions per venue), accompanied by growing concerns over review quality and reviewer responsibility. This position paper argues for the need to transform the traditional one-way review system into a bi-directional feedback loop where authors evaluate review quality and reviewers earn formal accreditation, creating an accountability framework that promotes a sustainable, high-quality peer review system. The current review system can be viewed as an interaction between three parties: the authors, reviewers, and system (i.e., conference), where we posit that all three parties share responsibility for the current problems. However, issues with authors can only be addressed through policy enforcement and detection tools, and ethical concerns can only be corrected through self-reflection. As such, this paper focuses on reforming reviewer accountability with systematic rewards through two key mechanisms: (1) a two-stage bi-directional review system that allows authors to evaluate reviews while minimizing retaliatory behavior, (2)a systematic reviewer reward system that incentivizes quality reviewing. We ask for the community's strong interest in these problems and the reforms that are needed to enhance the peer review process. 

---
# Position: Epistemic Artificial Intelligence is Essential for Machine Learning Models to Know When They Do Not Know 

**Authors**: Shireen Kudukkil Manchingal, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04950)  

**Abstract**: Despite the impressive achievements of AI, including advancements in generative models and large language models, there remains a significant gap in the ability of AI to handle uncertainty and generalize beyond the training data. We argue that AI models, especially in autonomous systems, fail to make robust predictions when faced with unfamiliar or adversarial data, as evidenced by incidents with autonomous vehicles. Traditional machine learning approaches struggle to address these issues due to an overemphasis on data fitting and domain adaptation. This position paper posits a paradigm shift towards epistemic artificial intelligence, emphasizing the need for models to learn not only from what they know but also from their ignorance. This approach, which focuses on recognizing and managing uncertainty, offers a potential solution to improve the resilience and robustness of AI systems, ensuring that they can better handle unpredictable real-world environments. 

---
# Belief Filtering for Epistemic Control in Linguistic State Space 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2505.04927)  

**Abstract**: We examine belief filtering as a mechanism for the epistemic control of artificial agents, focusing on the regulation of internal cognitive states represented as linguistic expressions. This mechanism is developed within the Semantic Manifold framework, where belief states are dynamic, structured ensembles of natural language fragments. Belief filters act as content-aware operations on these fragments across various cognitive transitions. This paper illustrates how the inherent interpretability and modularity of such a linguistically-grounded cognitive architecture directly enable belief filtering, offering a principled approach to agent regulation. The study highlights the potential for enhancing AI safety and alignment through structured interventions in an agent's internal semantic space and points to new directions for architecturally embedded cognitive governance. 

---
# Enigme: Generative Text Puzzles for Evaluating Reasoning in Language Models 

**Authors**: John Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2505.04914)  

**Abstract**: Transformer-decoder language models are a core innovation in text based generative artificial intelligence. These models are being deployed as general-purpose intelligence systems in many applications. Central to their utility is the capacity to understand natural language commands and exploit the reasoning embedded in human text corpora to apply some form of reasoning process to a wide variety of novel tasks. To understand the limitations of this approach to generating reasoning we argue that we need to consider the architectural constraints of these systems. Consideration of the latent variable structure of transformer-decoder models allows us to design reasoning tasks that should probe the boundary of their capacity to reason. We present enigme, an open-source library for generating text-based puzzles to be used in training and evaluating reasoning skills within transformer-decoder models and future AI architectures. 

---
# CRAFT: Cultural Russian-Oriented Dataset Adaptation for Focused Text-to-Image Generation 

**Authors**: Viacheslav Vasilev, Vladimir Arkhipkin, Julia Agafonova, Tatiana Nikulina, Evelina Mironova, Alisa Shichanina, Nikolai Gerasimenko, Mikhail Shoytov, Denis Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2505.04851)  

**Abstract**: Despite the fact that popular text-to-image generation models cope well with international and general cultural queries, they have a significant knowledge gap regarding individual cultures. This is due to the content of existing large training datasets collected on the Internet, which are predominantly based on Western European or American popular culture. Meanwhile, the lack of cultural adaptation of the model can lead to incorrect results, a decrease in the generation quality, and the spread of stereotypes and offensive content. In an effort to address this issue, we examine the concept of cultural code and recognize the critical importance of its understanding by modern image generation models, an issue that has not been sufficiently addressed in the research community to date. We propose the methodology for collecting and processing the data necessary to form a dataset based on the cultural code, in particular the Russian one. We explore how the collected data affects the quality of generations in the national domain and analyze the effectiveness of our approach using the Kandinsky 3.1 text-to-image model. Human evaluation results demonstrate an increase in the level of awareness of Russian culture in the model. 

---
# Large Language Models are Autonomous Cyber Defenders 

**Authors**: Sebastián R. Castro, Roberto Campbell, Nancy Lau, Octavio Villalobos, Jiaqi Duan, Alvaro A. Cardenas  

**Link**: [PDF](https://arxiv.org/pdf/2505.04843)  

**Abstract**: Fast and effective incident response is essential to prevent adversarial cyberattacks. Autonomous Cyber Defense (ACD) aims to automate incident response through Artificial Intelligence (AI) agents that plan and execute actions. Most ACD approaches focus on single-agent scenarios and leverage Reinforcement Learning (RL). However, ACD RL-trained agents depend on costly training, and their reasoning is not always explainable or transferable. Large Language Models (LLMs) can address these concerns by providing explainable actions in general security contexts. Researchers have explored LLM agents for ACD but have not evaluated them on multi-agent scenarios or interacting with other ACD agents. In this paper, we show the first study on how LLMs perform in multi-agent ACD environments by proposing a new integration to the CybORG CAGE 4 environment. We examine how ACD teams of LLM and RL agents can interact by proposing a novel communication protocol. Our results highlight the strengths and weaknesses of LLMs and RL and help us identify promising research directions to create, train, and deploy future teams of ACD agents. 

---
# Is there Value in Reinforcement Learning? 

**Authors**: Lior Fox, Yonatan Loewenstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.04822)  

**Abstract**: Action-values play a central role in popular Reinforcement Learing (RL) models of behavior. Yet, the idea that action-values are explicitly represented has been extensively debated. Critics had therefore repeatedly suggested that policy-gradient (PG) models should be favored over value-based (VB) ones, as a potential solution for this dilemma. Here we argue that this solution is unsatisfying. This is because PG methods are not, in fact, "Value-free" -- while they do not rely on an explicit representation of Value for acting (stimulus-response mapping), they do require it for learning. Hence, switching to PG models is, per se, insufficient for eliminating Value from models of behavior. More broadly, the requirement for a representation of Value stems from the underlying assumptions regarding the optimization objective posed by the standard RL framework, not from the particular algorithm chosen to solve it. Previous studies mostly took these standard RL assumptions for granted, as part of their conceptualization or problem modeling, while debating the different methods used to optimize it (i.e., PG or VB). We propose that, instead, the focus of the debate should shift to critically evaluating the underlying modeling assumptions. Such evaluation is particularly important from an experimental perspective. Indeed, the very notion of Value must be reconsidered when standard assumptions (e.g., risk neutrality, full-observability, Markovian environment, exponential discounting) are relaxed, as is likely in natural settings. Finally, we use the Value debate as a case study to argue in favor of a more nuanced, algorithmic rather than statistical, view of what constitutes "a model" in cognitive sciences. Our analysis suggests that besides "parametric" statistical complexity, additional aspects such as computational complexity must also be taken into account when evaluating model complexity. 

---
# The Promise and Limits of LLMs in Constructing Proofs and Hints for Logic Problems in Intelligent Tutoring Systems 

**Authors**: Sutapa Dey Tithi, Arun Kumar Ramesh, Clara DiMarco, Xiaoyi Tian, Nazia Alam, Kimia Fazeli, Tiffany Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2505.04736)  

**Abstract**: Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance with 84.4% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but requires additional modifications to ensure accuracy and pedagogical appropriateness. 

---
# Dynamic Location Search for Identifying Maximum Weighted Independent Sets in Complex Networks 

**Authors**: Enqiang Zhu, Chenkai Hao, Chanjuan Liu, Yongsheng Rao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04674)  

**Abstract**: While Artificial intelligence (AI), including Generative AI, are effective at generating high-quality traffic data and optimization solutions in intelligent transportation systems (ITSs), these techniques often demand significant training time and computational resources, especially in large-scale and complex scenarios. To address this, we introduce a novel and efficient algorithm for solving the maximum weighted independent set (MWIS) problem, which can be used to model many ITSs applications, such as traffic signal control and vehicle routing. Given the NP-hard nature of the MWIS problem, our proposed algorithm, DynLS, incorporates three key innovations to solve it effectively. First, it uses a scores-based adaptive vertex perturbation (SAVP) technique to accelerate convergence, particularly in sparse graphs. Second, it includes a region location mechanism (RLM) to help escape local optima by dynamically adjusting the search space. Finally, it employs a novel variable neighborhood descent strategy, ComLS, which combines vertex exchange strategies with a reward mechanism to guide the search toward high-quality solutions. Our experimental results demonstrate DynLS's superior performance, consistently delivering high-quality solutions within 1000 seconds. DynLS outperformed five leading algorithms across 360 test instances, achieving the best solution for 350 instances and surpassing the second-best algorithm, Cyclic-Fast, by 177 instances. Moreover, DynLS matched Cyclic-Fast's convergence speed, highlighting its efficiency and practicality. This research represents a significant advancement in heuristic algorithms for the MWIS problem, offering a promising approach to aid AI techniques in optimizing intelligent transportation systems. 

---
# Computational Irreducibility as the Foundation of Agency: A Formal Model Connecting Undecidability to Autonomous Behavior in Complex Systems 

**Authors**: Poria Azadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04646)  

**Abstract**: This article explores the emergence of autonomy and agency by connecting fundamental computational limits (decidability, completeness, computational irreducibility) with physical concepts. We introduce a formal model of a "minimal agent" operating within potentially Turing-complete environments. Using algorithmic information theory, we argue that the inherent undecidability and computational irreducibility of agent-environment interaction lead to unpredictability and novel information generation, enabling agency (effective goal-directed action). Computational irreducibility prevents full external prediction, creating necessary conditions for autonomous behavior. We relate this to computational sourcehood, where an agent is the irreducible origin of its behavior, though formalizing this concept remains challenging. Our central thesis, formally proven, is that genuine autonomy necessarily implies undecidability from an external perspective, distinguishing autonomous systems from predictable ones. We propose that agency arises when agent-environment coupling complexity allows mutual information between internal states and relevant environmental variables to increase, particularly where analytical solutions are absent and operational closure is needed for persistence. This framework links agency directly to the computational properties of interaction, offering implications for understanding consciousness, designing autonomous AI, and reconceptualizing free will in a deterministic yet computationally irreducible universe. 

---
# Towards Artificial Intelligence Research Assistant for Expert-Involved Learning 

**Authors**: Tianyu Liu, Simeng Han, Xiao Luo, Hanchen Wang, Pan Lu, Biqing Zhu, Yuge Wang, Keyi Li, Jiapeng Chen, Rihao Qu, Yufeng Liu, Xinyue Cui, Aviv Yaish, Yuhang Chen, Minsheng Hao, Chuhan Li, Kexing Li, Arman Cohan, Hua Xu, Mark Gerstein, James Zou, Hongyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04638)  

**Abstract**: Large Language Models (LLMs) and Large Multi-Modal Models (LMMs) have emerged as transformative tools in scientific research, yet their reliability and specific contributions to biomedical applications remain insufficiently characterized. In this study, we present \textbf{AR}tificial \textbf{I}ntelligence research assistant for \textbf{E}xpert-involved \textbf{L}earning (ARIEL), a multimodal dataset designed to benchmark and enhance two critical capabilities of LLMs and LMMs in biomedical research: summarizing extensive scientific texts and interpreting complex biomedical figures. To facilitate rigorous assessment, we create two open-source sets comprising biomedical articles and figures with designed questions. We systematically benchmark both open- and closed-source foundation models, incorporating expert-driven human evaluations conducted by doctoral-level experts. Furthermore, we improve model performance through targeted prompt engineering and fine-tuning strategies for summarizing research papers, and apply test-time computational scaling to enhance the reasoning capabilities of LMMs, achieving superior accuracy compared to human-expert corrections. We also explore the potential of using LMM Agents to generate scientific hypotheses from diverse multimodal inputs. Overall, our results delineate clear strengths and highlight significant limitations of current foundation models, providing actionable insights and guiding future advancements in deploying large-scale language and multi-modal models within biomedical research. 

---
# Flow-GRPO: Training Flow Matching Models via Online RL 

**Authors**: Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05470)  

**Abstract**: We propose Flow-GRPO, the first method integrating online reinforcement learning (RL) into flow matching models. Our approach uses two key strategies: (1) an ODE-to-SDE conversion that transforms a deterministic Ordinary Differential Equation (ODE) into an equivalent Stochastic Differential Equation (SDE) that matches the original model's marginal distribution at all timesteps, enabling statistical sampling for RL exploration; and (2) a Denoising Reduction strategy that reduces training denoising steps while retaining the original inference timestep number, significantly improving sampling efficiency without performance degradation. Empirically, Flow-GRPO is effective across multiple text-to-image tasks. For complex compositions, RL-tuned SD3.5 generates nearly perfect object counts, spatial relations, and fine-grained attributes, boosting GenEval accuracy from $63\%$ to $95\%$. In visual text rendering, its accuracy improves from $59\%$ to $92\%$, significantly enhancing text generation. Flow-GRPO also achieves substantial gains in human preference alignment. Notably, little to no reward hacking occurred, meaning rewards did not increase at the cost of image quality or diversity, and both remained stable in our experiments. 

---
# StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant 

**Authors**: Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, Ping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05467)  

**Abstract**: We present StreamBridge, a simple yet effective framework that seamlessly transforms offline Video-LLMs into streaming-capable models. It addresses two fundamental challenges in adapting existing models into online scenarios: (1) limited capability for multi-turn real-time understanding, and (2) lack of proactive response mechanisms. Specifically, StreamBridge incorporates (1) a memory buffer combined with a round-decayed compression strategy, supporting long-context multi-turn interactions, and (2) a decoupled, lightweight activation model that can be effortlessly integrated into existing Video-LLMs, enabling continuous proactive responses. To further support StreamBridge, we construct Stream-IT, a large-scale dataset tailored for streaming video understanding, featuring interleaved video-text sequences and diverse instruction formats. Extensive experiments show that StreamBridge significantly improves the streaming understanding capabilities of offline Video-LLMs across various tasks, outperforming even proprietary models such as GPT-4o and Gemini 1.5 Pro. Simultaneously, it achieves competitive or superior performance on standard video understanding benchmarks. 

---
# ComPO: Preference Alignment via Comparison Oracles 

**Authors**: Peter Chen, Xi Chen, Wotao Yin, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05465)  

**Abstract**: Direct alignment methods are increasingly used for aligning large language models (LLMs) with human preferences. However, these methods suffer from the issues of verbosity and likelihood displacement, which can be driven by the noisy preference pairs that induce similar likelihood for preferred and dispreferred responses. The contributions of this paper are two-fold. First, we propose a new preference alignment method based on comparison oracles and provide the convergence guarantee for its basic scheme. Second, we improve our method using some heuristics and conduct the experiments to demonstrate the flexibility and compatibility of practical scheme in improving the performance of LLMs using noisy preference pairs. Evaluations are conducted across multiple base and instruction-tuned models (Mistral-7B, Llama-3-8B and Gemma-2-9B) with benchmarks (AlpacaEval 2, MT-Bench and Arena-Hard). Experimental results show the effectiveness of our method as an alternative to addressing the limitations of existing direct alignment methods. A highlight of our work is that we evidence the importance of designing specialized methods for preference pairs with distinct likelihood margin, which complements the recent findings in \citet{Razin-2025-Unintentional}. 

---
# TransProQA: an LLM-based literary Translation evaluation metric with Professional Question Answering 

**Authors**: Ran Zhang, Wei Zhao, Lieve Macken, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2505.05423)  

**Abstract**: The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation (MT) as being superior to experienced professional human translation. In the long run, this bias could result in a permanent decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce TransProQA, a novel, reference-free, LLM-based question-answering (QA) framework designed specifically for literary translation evaluation. TransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, TransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation (ACC-EQ and Kendall's tau) and surpassing the best state-of-the-art (SOTA) metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, TransProQA approaches human-level evaluation performance comparable to trained linguistic annotators. It demonstrates broad applicability to open-source models such as LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free literary evaluation metric and a valuable tool for evaluating texts that require local processing due to copyright or ethical considerations. 

---
# TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation 

**Authors**: Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05422)  

**Abstract**: Pioneering token-based works such as Chameleon and Emu3 have established a foundation for multimodal unification but face challenges of high training computational overhead and limited comprehension performance due to a lack of high-level semantics. In this paper, we introduce TokLIP, a visual tokenizer that enhances comprehension by semanticizing vector-quantized (VQ) tokens and incorporating CLIP-level semantics while enabling end-to-end multimodal autoregressive training with standard VQ tokens. TokLIP integrates a low-level discrete VQ tokenizer with a ViT-based token encoder to capture high-level continuous semantics. Unlike previous approaches (e.g., VILA-U) that discretize high-level features, TokLIP disentangles training objectives for comprehension and generation, allowing the direct application of advanced VQ tokenizers without the need for tailored quantization operations. Our empirical results demonstrate that TokLIP achieves exceptional data efficiency, empowering visual tokens with high-level semantic understanding while enhancing low-level generative capacity, making it well-suited for autoregressive Transformers in both comprehension and generation tasks. The code and models are available at this https URL. 

---
# Reasoning Models Don't Always Say What They Think 

**Authors**: Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Samuel R. Bowman, Jan Leike, Jared Kaplan, Ethan Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.05410)  

**Abstract**: Chain-of-thought (CoT) offers a potential boon for AI safety as it allows monitoring a model's CoT to try to understand its intentions and reasoning processes. However, the effectiveness of such monitoring hinges on CoTs faithfully representing models' actual reasoning processes. We evaluate CoT faithfulness of state-of-the-art reasoning models across 6 reasoning hints presented in the prompts and find: (1) for most settings and models tested, CoTs reveal their usage of hints in at least 1% of examples where they use the hint, but the reveal rate is often below 20%, (2) outcome-based reinforcement learning initially improves faithfulness but plateaus without saturating, and (3) when reinforcement learning increases how frequently hints are used (reward hacking), the propensity to verbalize them does not increase, even without training against a CoT monitor. These results suggest that CoT monitoring is a promising way of noticing undesired behaviors during training and evaluations, but that it is not sufficient to rule them out. They also suggest that in settings like ours where CoT reasoning is not necessary, test-time monitoring of CoTs is unlikely to reliably catch rare and catastrophic unexpected behaviors. 

---
# Crosslingual Reasoning through Test-Time Scaling 

**Authors**: Zheng-Xin Yong, M. Farid Adilazuarda, Jonibek Mansurov, Ruochen Zhang, Niklas Muennighoff, Carsten Eickhoff, Genta Indra Winata, Julia Kreutzer, Stephen H. Bach, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2505.05408)  

**Abstract**: Reasoning capabilities of large language models are primarily studied for English, even when pretrained models are multilingual. In this work, we investigate to what extent English reasoning finetuning with long chain-of-thoughts (CoTs) can generalize across languages. First, we find that scaling up inference compute for English-centric reasoning language models (RLMs) improves multilingual mathematical reasoning across many languages including low-resource languages, to an extent where they outperform models twice their size. Second, we reveal that while English-centric RLM's CoTs are naturally predominantly English, they consistently follow a quote-and-think pattern to reason about quoted non-English inputs. Third, we discover an effective strategy to control the language of long CoT reasoning, and we observe that models reason better and more efficiently in high-resource languages. Finally, we observe poor out-of-domain reasoning generalization, in particular from STEM to cultural commonsense knowledge, even for English. Overall, we demonstrate the potentials, study the mechanisms and outline the limitations of crosslingual generalization of English reasoning test-time scaling. We conclude that practitioners should let English-centric RLMs reason in high-resource languages, while further work is needed to improve reasoning in low-resource languages and out-of-domain contexts. 

---
# CART-ELC: Oblique Decision Tree Induction via Exhaustive Search 

**Authors**: Andrew D. Laack  

**Link**: [PDF](https://arxiv.org/pdf/2505.05402)  

**Abstract**: Oblique decision trees have attracted attention due to their potential for improved classification performance over traditional axis-aligned decision trees. However, methods that rely on exhaustive search to find oblique splits face computational challenges. As a result, they have not been widely explored. We introduce a novel algorithm, Classification and Regression Tree - Exhaustive Linear Combinations (CART-ELC), for inducing oblique decision trees that performs an exhaustive search on a restricted set of hyperplanes. We then investigate the algorithm's computational complexity and its predictive capabilities. Our results demonstrate that CART-ELC consistently achieves competitive performance on small datasets, often yielding statistically significant improvements in classification accuracy relative to existing decision tree induction algorithms, while frequently producing shallower, simpler, and thus more interpretable trees. 

---
# Threshold Modulation for Online Test-Time Adaptation of Spiking Neural Networks 

**Authors**: Kejie Zhao, Wenjia Hua, Aiersi Tuerhong, Luziwei Leng, Yuxin Ma, Qinghua Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.05375)  

**Abstract**: Recently, spiking neural networks (SNNs), deployed on neuromorphic chips, provide highly efficient solutions on edge devices in different scenarios. However, their ability to adapt to distribution shifts after deployment has become a crucial challenge. Online test-time adaptation (OTTA) offers a promising solution by enabling models to dynamically adjust to new data distributions without requiring source data or labeled target samples. Nevertheless, existing OTTA methods are largely designed for traditional artificial neural networks and are not well-suited for SNNs. To address this gap, we propose a low-power, neuromorphic chip-friendly online test-time adaptation framework, aiming to enhance model generalization under distribution shifts. The proposed approach is called Threshold Modulation (TM), which dynamically adjusts the firing threshold through neuronal dynamics-inspired normalization, being more compatible with neuromorphic hardware. Experimental results on benchmark datasets demonstrate the effectiveness of this method in improving the robustness of SNNs against distribution shifts while maintaining low computational cost. The proposed method offers a practical solution for online test-time adaptation of SNNs, providing inspiration for the design of future neuromorphic chips. The demo code is available at this http URL. 

---
# Time of the Flight of the Gaussians: Optimizing Depth Indirectly in Dynamic Radiance Fields 

**Authors**: Runfeng Li, Mikhail Okunev, Zixuan Guo, Anh Ha Duong, Christian Richardt, Matthew O'Toole, James Tompkin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05356)  

**Abstract**: We present a method to reconstruct dynamic scenes from monocular continuous-wave time-of-flight (C-ToF) cameras using raw sensor samples that achieves similar or better accuracy than neural volumetric approaches and is 100x faster. Quickly achieving high-fidelity dynamic 3D reconstruction from a single viewpoint is a significant challenge in computer vision. In C-ToF radiance field reconstruction, the property of interest-depth-is not directly measured, causing an additional challenge. This problem has a large and underappreciated impact upon the optimization when using a fast primitive-based scene representation like 3D Gaussian splatting, which is commonly used with multi-view data to produce satisfactory results and is brittle in its optimization otherwise. We incorporate two heuristics into the optimization to improve the accuracy of scene geometry represented by Gaussians. Experimental results show that our approach produces accurate reconstructions under constrained C-ToF sensing conditions, including for fast motions like swinging baseball bats. this https URL 

---
# High-fidelity Grain Growth Modeling: Leveraging Deep Learning for Fast Computations 

**Authors**: Pungponhavoan Tep, Marc Bernacki  

**Link**: [PDF](https://arxiv.org/pdf/2505.05354)  

**Abstract**: Grain growth simulation is crucial for predicting metallic material microstructure evolution during annealing and resulting final mechanical properties, but traditional partial differential equation-based methods are computationally expensive, creating bottlenecks in materials design and manufacturing. In this work, we introduce a machine learning framework that combines a Convolutional Long Short-Term Memory networks with an Autoencoder to efficiently predict grain growth evolution. Our approach captures both spatial and temporal aspects of grain evolution while encoding high-dimensional grain structure data into a compact latent space for pattern learning, enhanced by a novel composite loss function combining Mean Squared Error, Structural Similarity Index Measurement, and Boundary Preservation to maintain structural integrity of grain boundary topology of the prediction. Results demonstrated that our machine learning approach accelerates grain growth prediction by up to \SI{89}{\times} faster, reducing computation time from \SI{10}{\minute} to approximately \SI{10}{\second} while maintaining high-fidelity predictions. The best model (S-30-30) achieving a structural similarity score of \SI{86.71}{\percent} and mean grain size error of just \SI{0.07}{\percent}. All models accurately captured grain boundary topology, morphology, and size distributions. This approach enables rapid microstructural prediction for applications where conventional simulations are prohibitively time-consuming, potentially accelerating innovation in materials science and manufacturing. 

---
# Feature-Augmented Deep Networks for Multiscale Building Segmentation in High-Resolution UAV and Satellite Imagery 

**Authors**: Chintan B. Maniyar, Minakshi Kumar, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2505.05321)  

**Abstract**: Accurate building segmentation from high-resolution RGB imagery remains challenging due to spectral similarity with non-building features, shadows, and irregular building geometries. In this study, we present a comprehensive deep learning framework for multiscale building segmentation using RGB aerial and satellite imagery with spatial resolutions ranging from 0.4m to 2.7m. We curate a diverse, multi-sensor dataset and introduce feature-augmented inputs by deriving secondary representations including Principal Component Analysis (PCA), Visible Difference Vegetation Index (VDVI), Morphological Building Index (MBI), and Sobel edge filters from RGB channels. These features guide a Res-U-Net architecture in learning complex spatial patterns more effectively. We also propose training policies incorporating layer freezing, cyclical learning rates, and SuperConvergence to reduce training time and resource usage. Evaluated on a held-out WorldView-3 image, our model achieves an overall accuracy of 96.5%, an F1-score of 0.86, and an Intersection over Union (IoU) of 0.80, outperforming existing RGB-based benchmarks. This study demonstrates the effectiveness of combining multi-resolution imagery, feature augmentation, and optimized training strategies for robust building segmentation in remote sensing applications. 

---
# Mapping User Trust in Vision Language Models: Research Landscape, Challenges, and Prospects 

**Authors**: Agnese Chiatti, Sara Bernardini, Lara Shibelski Godoy Piccolo, Viola Schiaffonati, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2505.05318)  

**Abstract**: The rapid adoption of Vision Language Models (VLMs), pre-trained on large image-text and video-text datasets, calls for protecting and informing users about when to trust these systems. This survey reviews studies on trust dynamics in user-VLM interactions, through a multi-disciplinary taxonomy encompassing different cognitive science capabilities, collaboration modes, and agent behaviours. Literature insights and findings from a workshop with prospective VLM users inform preliminary requirements for future VLM trust studies. 

---
# Scalable Chain of Thoughts via Elastic Reasoning 

**Authors**: Yuhui Xu, Hanze Dong, Lei Wang, Doyen Sahoo, Junnan Li, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.05315)  

**Abstract**: Large reasoning models (LRMs) have achieved remarkable progress on complex tasks by generating extended chains of thought (CoT). However, their uncontrolled output lengths pose significant challenges for real-world deployment, where inference-time budgets on tokens, latency, or compute are strictly constrained. We propose Elastic Reasoning, a novel framework for scalable chain of thoughts that explicitly separates reasoning into two phases--thinking and solution--with independently allocated budgets. At test time, Elastic Reasoning prioritize that completeness of solution segments, significantly improving reliability under tight resource constraints. To train models that are robust to truncated thinking, we introduce a lightweight budget-constrained rollout strategy, integrated into GRPO, which teaches the model to reason adaptively when the thinking process is cut short and generalizes effectively to unseen budget constraints without additional training. Empirical results on mathematical (AIME, MATH500) and programming (LiveCodeBench, Codeforces) benchmarks demonstrate that Elastic Reasoning performs robustly under strict budget constraints, while incurring significantly lower training cost than baseline methods. Remarkably, our approach also produces more concise and efficient reasoning even in unconstrained settings. Elastic Reasoning offers a principled and practical solution to the pressing challenge of controllable reasoning at scale. 

---
# Benchmarking Ophthalmology Foundation Models for Clinically Significant Age Macular Degeneration Detection 

**Authors**: Benjamin A. Cohen, Jonathan Fhima, Meishar Meisel, Baskin Meital, Luis Filipe Nakayama, Eran Berkowitz, Joachim A. Behar  

**Link**: [PDF](https://arxiv.org/pdf/2505.05291)  

**Abstract**: Self-supervised learning (SSL) has enabled Vision Transformers (ViTs) to learn robust representations from large-scale natural image datasets, enhancing their generalization across domains. In retinal imaging, foundation models pretrained on either natural or ophthalmic data have shown promise, but the benefits of in-domain pretraining remain uncertain. To investigate this, we benchmark six SSL-pretrained ViTs on seven digital fundus image (DFI) datasets totaling 70,000 expert-annotated images for the task of moderate-to-late age-related macular degeneration (AMD) identification. Our results show that iBOT pretrained on natural images achieves the highest out-of-distribution generalization, with AUROCs of 0.80-0.97, outperforming domain-specific models, which achieved AUROCs of 0.78-0.96 and a baseline ViT-L with no pretraining, which achieved AUROCs of 0.68-0.91. These findings highlight the value of foundation models in improving AMD identification and challenge the assumption that in-domain pretraining is necessary. Furthermore, we release BRAMD, an open-access dataset (n=587) of DFIs with AMD labels from Brazil. 

---
# PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes 

**Authors**: Ahmed Abdelreheem, Filippo Aleotti, Jamie Watson, Zawar Qureshi, Abdelrahman Eldesokey, Peter Wonka, Gabriel Brostow, Sara Vicente, Guillermo Garcia-Hernando  

**Link**: [PDF](https://arxiv.org/pdf/2505.05288)  

**Abstract**: We introduce the novel task of Language-Guided Object Placement in Real 3D Scenes. Our model is given a 3D scene's point cloud, a 3D asset, and a textual prompt broadly describing where the 3D asset should be placed. The task here is to find a valid placement for the 3D asset that respects the prompt. Compared with other language-guided localization tasks in 3D scenes such as grounding, this task has specific challenges: it is ambiguous because it has multiple valid solutions, and it requires reasoning about 3D geometric relationships and free space. We inaugurate this task by proposing a new benchmark and evaluation protocol. We also introduce a new dataset for training 3D LLMs on this task, as well as the first method to serve as a non-trivial baseline. We believe that this challenging task and our new benchmark could become part of the suite of benchmarks used to evaluate and compare generalist 3D LLM models. 

---
# Software Development Life Cycle Perspective: A Survey of Benchmarks for CodeLLMs and Agents 

**Authors**: Kaixin Wang, Tianlin Li, Xiaoyu Zhang, Chong Wang, Weisong Sun, Yang Liu, Bin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05283)  

**Abstract**: Code large language models (CodeLLMs) and agents have shown great promise in tackling complex software engineering this http URL to traditional software engineering methods, CodeLLMs and agents offer stronger abilities, and can flexibly process inputs and outputs in both natural and code. Benchmarking plays a crucial role in evaluating the capabilities of CodeLLMs and agents, guiding their development and deployment. However, despite their growing significance, there remains a lack of comprehensive reviews of benchmarks for CodeLLMs and agents. To bridge this gap, this paper provides a comprehensive review of existing benchmarks for CodeLLMs and agents, studying and analyzing 181 benchmarks from 461 relevant papers, covering the different phases of the software development life cycle (SDLC). Our findings reveal a notable imbalance in the coverage of current benchmarks, with approximately 60% focused on the software development phase in SDLC, while requirements engineering and software design phases receive minimal attention at only 5% and 3%, respectively. Additionally, Python emerges as the dominant programming language across the reviewed benchmarks. Finally, this paper highlights the challenges of current research and proposes future directions, aiming to narrow the gap between the theoretical capabilities of CodeLLMs and agents and their application in real-world scenarios. 

---
# T-T: Table Transformer for Tagging-based Aspect Sentiment Triplet Extraction 

**Authors**: Kun Peng, Chaodong Tong, Cong Cao, Hao Peng, Qian Li, Guanlin Wu, Lei Jiang, Yanbing Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05271)  

**Abstract**: Aspect sentiment triplet extraction (ASTE) aims to extract triplets composed of aspect terms, opinion terms, and sentiment polarities from given sentences. The table tagging method is a popular approach to addressing this task, which encodes a sentence into a 2-dimensional table, allowing for the tagging of relations between any two words. Previous efforts have focused on designing various downstream relation learning modules to better capture interactions between tokens in the table, revealing that a stronger capability to capture relations can lead to greater improvements in the model. Motivated by this, we attempt to directly utilize transformer layers as downstream relation learning modules. Due to the powerful semantic modeling capability of transformers, it is foreseeable that this will lead to excellent improvement. However, owing to the quadratic relation between the length of the table and the length of the input sentence sequence, using transformers directly faces two challenges: overly long table sequences and unfair local attention interaction. To address these challenges, we propose a novel Table-Transformer (T-T) for the tagging-based ASTE method. Specifically, we introduce a stripe attention mechanism with a loop-shift strategy to tackle these challenges. The former modifies the global attention mechanism to only attend to a 2-dimensional local attention window, while the latter facilitates interaction between different attention windows. Extensive and comprehensive experiments demonstrate that the T-T, as a downstream relation learning module, achieves state-of-the-art performance with lower computational costs. 

---
# Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration 

**Authors**: Andreas Kontogiannis, Konstantinos Papathanasiou, Yi Shen, Giorgos Stamou, Michael M. Zavlanos, George Vouros  

**Link**: [PDF](https://arxiv.org/pdf/2505.05262)  

**Abstract**: Learning to cooperate in distributed partially observable environments with no communication abilities poses significant challenges for multi-agent deep reinforcement learning (MARL). This paper addresses key concerns in this domain, focusing on inferring state representations from individual agent observations and leveraging these representations to enhance agents' exploration and collaborative task execution policies. To this end, we propose a novel state modelling framework for cooperative MARL, where agents infer meaningful belief representations of the non-observable state, with respect to optimizing their own policies, while filtering redundant and less informative joint state information. Building upon this framework, we propose the MARL SMPE algorithm. In SMPE, agents enhance their own policy's discriminative abilities under partial observability, explicitly by incorporating their beliefs into the policy network, and implicitly by adopting an adversarial type of exploration policies which encourages agents to discover novel, high-value states while improving the discriminative abilities of others. Experimentally, we show that SMPE outperforms state-of-the-art MARL algorithms in complex fully cooperative tasks from the MPE, LBF, and RWARE benchmarks. 

---
# Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning 

**Authors**: Amir Rezaei Balef, Claire Vernade, Katharina Eggensperger  

**Link**: [PDF](https://arxiv.org/pdf/2505.05226)  

**Abstract**: The Combined Algorithm Selection and Hyperparameter optimization (CASH) is a challenging resource allocation problem in the field of AutoML. We propose MaxUCB, a max $k$-armed bandit method to trade off exploring different model classes and conducting hyperparameter optimization. MaxUCB is specifically designed for the light-tailed and bounded reward distributions arising in this setting and, thus, provides an efficient alternative compared to classic max $k$-armed bandit methods assuming heavy-tailed reward distributions. We theoretically and empirically evaluate our method on four standard AutoML benchmarks, demonstrating superior performance over prior approaches. 

---
# Incentive-Aware Machine Learning; Robustness, Fairness, Improvement & Causality 

**Authors**: Chara Podimata  

**Link**: [PDF](https://arxiv.org/pdf/2505.05211)  

**Abstract**: The article explores the emerging domain of incentive-aware machine learning (ML), which focuses on algorithmic decision-making in contexts where individuals can strategically modify their inputs to influence outcomes. It categorizes the research into three perspectives: robustness, aiming to design models resilient to "gaming"; fairness, analyzing the societal impacts of such systems; and improvement/causality, recognizing situations where strategic actions lead to genuine personal or societal improvement. The paper introduces a unified framework encapsulating models for these perspectives, including offline, online, and causal settings, and highlights key challenges such as differentiating between gaming and improvement and addressing heterogeneity among agents. By synthesizing findings from diverse works, we outline theoretical advancements and practical solutions for robust, fair, and causally-informed incentive-aware ML systems. 

---
# LAPSO: A Unified Optimization View for Learning-Augmented Power System Operations 

**Authors**: Wangkun Xu, Zhongda Chu, Fei Teng  

**Link**: [PDF](https://arxiv.org/pdf/2505.05203)  

**Abstract**: With the high penetration of renewables, traditional model-based power system operation is challenged to deliver economic, stable, and robust decisions. Machine learning has emerged as a powerful modeling tool for capturing complex dynamics to address these challenges. However, its separate design often lacks systematic integration with existing methods. To fill the gap, this paper proposes a holistic framework of Learning-Augmented Power System Operations (LAPSO, pronounced as Lap-So). Adopting a native optimization perspective, LAPSO is centered on the operation stage and aims to break the boundary between temporally siloed power system tasks, such as forecast, operation and control, while unifying the objectives of machine learning and model-based optimizations at both training and inference stages. Systematic analysis and simulations demonstrate the effectiveness of applying LAPSO in designing new integrated algorithms, such as stability-constrained optimization (SCO) and objective-based forecasting (OBF), while enabling end-to-end tracing of different sources of uncertainties. In addition, a dedicated Python package-lapso is introduced to automatically augment existing power system optimization models with learnable components. All code and data are available at this https URL. 

---
# Concept-Based Unsupervised Domain Adaptation 

**Authors**: Xinyue Xu, Yueying Hu, Hui Tang, Yi Qin, Lu Mi, Hao Wang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.05195)  

**Abstract**: Concept Bottleneck Models (CBMs) enhance interpretability by explaining predictions through human-understandable concepts but typically assume that training and test data share the same distribution. This assumption often fails under domain shifts, leading to degraded performance and poor generalization. To address these limitations and improve the robustness of CBMs, we propose the Concept-based Unsupervised Domain Adaptation (CUDA) framework. CUDA is designed to: (1) align concept representations across domains using adversarial training, (2) introduce a relaxation threshold to allow minor domain-specific differences in concept distributions, thereby preventing performance drop due to over-constraints of these distributions, (3) infer concepts directly in the target domain without requiring labeled concept data, enabling CBMs to adapt to diverse domains, and (4) integrate concept learning into conventional domain adaptation (DA) with theoretical guarantees, improving interpretability and establishing new benchmarks for DA. Experiments demonstrate that our approach significantly outperforms the state-of-the-art CBM and DA methods on real-world datasets. 

---
# Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks 

**Authors**: Yixin Cheng, Hongcheng Guo, Yangming Li, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2505.05190)  

**Abstract**: Text watermarking aims to subtly embed statistical signals into text by controlling the Large Language Model (LLM)'s sampling process, enabling watermark detectors to verify that the output was generated by the specified model. The robustness of these watermarking algorithms has become a key factor in evaluating their effectiveness. Current text watermarking algorithms embed watermarks in high-entropy tokens to ensure text quality. In this paper, we reveal that this seemingly benign design can be exploited by attackers, posing a significant risk to the robustness of the watermark. We introduce a generic efficient paraphrasing attack, the Self-Information Rewrite Attack (SIRA), which leverages the vulnerability by calculating the self-information of each token to identify potential pattern tokens and perform targeted attack. Our work exposes a widely prevalent vulnerability in current watermarking algorithms. The experimental results show SIRA achieves nearly 100% attack success rates on seven recent watermarking methods with only 0.88 USD per million tokens cost. Our approach does not require any access to the watermark algorithms or the watermarked LLM and can seamlessly transfer to any LLM as the attack model, even mobile-level models. Our findings highlight the urgent need for more robust watermarking. 

---
# Biomed-DPT: Dual Modality Prompt Tuning for Biomedical Vision-Language Models 

**Authors**: Wei Peng, Kang Liu, Jianchen Hu, Meng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05189)  

**Abstract**: Prompt learning is one of the most effective paradigms for adapting pre-trained vision-language models (VLMs) to the biomedical image classification tasks in few shot scenarios. However, most of the current prompt learning methods only used the text prompts and ignored the particular structures (such as the complex anatomical structures and subtle pathological features) in the biomedical images. In this work, we propose Biomed-DPT, a knowledge-enhanced dual modality prompt tuning technique. In designing the text prompt, Biomed-DPT constructs a dual prompt including the template-driven clinical prompts and the large language model (LLM)-driven domain-adapted prompts, then extracts the clinical knowledge from the domain-adapted prompts through the knowledge distillation technique. In designing the vision prompt, Biomed-DPT introduces the zero vector as a soft prompt to leverage attention re-weighting so that the focus on non-diagnostic regions and the recognition of non-critical pathological features are avoided. Biomed-DPT achieves an average classification accuracy of 66.14\% across 11 biomedical image datasets covering 9 modalities and 10 organs, with performance reaching 78.06\% in base classes and 75.97\% in novel classes, surpassing the Context Optimization (CoOp) method by 6.20\%, 3.78\%, and 8.04\%, respectively. Our code are available at \underline{this https URL}. 

---
# Stochastic Variational Propagation: Local, Scalable and Efficient Alternative to Backpropagation 

**Authors**: Bojian Yin, Federico Corradi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05181)  

**Abstract**: Backpropagation (BP) is the cornerstone of deep learning, but its reliance on global gradient synchronization limits scalability and imposes significant memory overhead. We propose Stochastic Variational Propagation (SVP), a scalable alternative that reframes training as hierarchical variational inference. SVP treats layer activations as latent variables and optimizes local Evidence Lower Bounds (ELBOs), enabling independent, local updates while preserving global coherence. However, directly applying KL divergence in layer-wise ELBOs risks inter-layer's representation collapse due to excessive compression. To prevent this, SVP projects activations into low-dimensional spaces via fixed random matrices, ensuring information preservation and representational diversity. Combined with a feature alignment loss for inter-layer consistency, SVP achieves competitive accuracy with BP across diverse architectures (MLPs, CNNs, Transformers) and datasets (MNIST to ImageNet), reduces memory usage by up to 4x, and significantly improves scalability. More broadly, SVP introduces a probabilistic perspective to deep representation learning, opening pathways toward more modular and interpretable neural network design. 

---
# Dukawalla: Voice Interfaces for Small Businesses in Africa 

**Authors**: Elizabeth Ankrah, Stephanie Nyairo, Mercy Muchai, Kagonya Awori, Millicent Ochieng, Mark Kariuki, Jacki O'Neill  

**Link**: [PDF](https://arxiv.org/pdf/2505.05170)  

**Abstract**: Small and medium sized businesses often struggle with data driven decision making do to a lack of advanced analytics tools, especially in African countries where they make up a majority of the workforce. Though many tools exist they are not designed to fit into the ways of working of SMB workers who are mobile first, have limited time to learn new workflows, and for whom social and business are tightly coupled. To address this, the Dukawalla prototype was created. This intelligent assistant bridges the gap between raw business data, and actionable insights by leveraging voice interaction and the power of generative AI. Dukawalla provides an intuitive way for business owners to interact with their data, aiding in informed decision making. This paper examines Dukawalla's deployment across SMBs in Nairobi, focusing on their experiences using this voice based assistant to streamline data collection and provide business insights 

---
# Understanding In-context Learning of Addition via Activation Subspaces 

**Authors**: Xinyan Hu, Kayo Yin, Michael I. Jordan, Jacob Steinhardt, Lijie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05145)  

**Abstract**: To perform in-context learning, language models must extract signals from individual few-shot examples, aggregate these into a learned prediction rule, and then apply this rule to new examples. How is this implemented in the forward pass of modern transformer models? To study this, we consider a structured family of few-shot learning tasks for which the true prediction rule is to add an integer $k$ to the input. We find that Llama-3-8B attains high accuracy on this task for a range of $k$, and localize its few-shot ability to just three attention heads via a novel optimization approach. We further show the extracted signals lie in a six-dimensional subspace, where four of the dimensions track the unit digit and the other two dimensions track overall magnitude. We finally examine how these heads extract information from individual few-shot examples, identifying a self-correction mechanism in which mistakes from earlier examples are suppressed by later examples. Our results demonstrate how tracking low-dimensional subspaces across a forward pass can provide insight into fine-grained computational structures. 

---
# Guiding Evolutionary AutoEncoder Training with Activation-Based Pruning Operators 

**Authors**: Steven Jorgensen, Erik Hemberg, Jamal Toutouh, Una-May O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2505.05138)  

**Abstract**: This study explores a novel approach to neural network pruning using evolutionary computation, focusing on simultaneously pruning the encoder and decoder of an autoencoder. We introduce two new mutation operators that use layer activations to guide weight pruning. Our findings reveal that one of these activation-informed operators outperforms random pruning, resulting in more efficient autoencoders with comparable performance to canonically trained models. Prior work has established that autoencoder training is effective and scalable with a spatial coevolutionary algorithm that cooperatively coevolves a population of encoders with a population of decoders, rather than one autoencoder. We evaluate how the same activity-guided mutation operators transfer to this context. We find that random pruning is better than guided pruning, in the coevolutionary setting. This suggests activation-based guidance proves more effective in low-dimensional pruning environments, where constrained sample spaces can lead to deviations from true uniformity in randomization. Conversely, population-driven strategies enhance robustness by expanding the total pruning dimensionality, achieving statistically uniform randomness that better preserves system dynamics. We experiment with pruning according to different schedules and present best combinations of operator and schedule for the canonical and coevolving populations cases. 

---
# Beyond Low-rank Decomposition: A Shortcut Approach for Efficient On-Device Learning 

**Authors**: Le-Trung Nguyen, Ael Quelennec, Van-Tam Nguyen, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2505.05086)  

**Abstract**: On-device learning has emerged as a promising direction for AI development, particularly because of its potential to reduce latency issues and mitigate privacy risks associated with device-server communication, while improving energy efficiency. Despite these advantages, significant memory and computational constraints still represent major challenges for its deployment. Drawing on previous studies on low-rank decomposition methods that address activation memory bottlenecks in backpropagation, we propose a novel shortcut approach as an alternative. Our analysis and experiments demonstrate that our method can reduce activation memory usage, even up to $120.09\times$ compared to vanilla training, while also reducing overall training FLOPs up to $1.86\times$ when evaluated on traditional benchmarks. 

---
# FG-CLIP: Fine-Grained Visual and Textual Alignment 

**Authors**: Chunyu Xie, Bin Wang, Fanjing Kong, Jincheng Li, Dawei Liang, Gengshen Zhang, Dawei Leng, Yuhui Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05071)  

**Abstract**: Contrastive Language-Image Pre-training (CLIP) excels in multimodal tasks such as image-text retrieval and zero-shot classification but struggles with fine-grained understanding due to its focus on coarse-grained short captions. To address this, we propose Fine-Grained CLIP (FG-CLIP), which enhances fine-grained understanding through three key innovations. First, we leverage large multimodal models to generate 1.6 billion long caption-image pairs for capturing global-level semantic details. Second, a high-quality dataset is constructed with 12 million images and 40 million region-specific bounding boxes aligned with detailed captions to ensure precise, context-rich representations. Third, 10 million hard fine-grained negative samples are incorporated to improve the model's ability to distinguish subtle semantic differences. Corresponding training methods are meticulously designed for these data. Extensive experiments demonstrate that FG-CLIP outperforms the original CLIP and other state-of-the-art methods across various downstream tasks, including fine-grained understanding, open-vocabulary object detection, image-text retrieval, and general multimodal benchmarks. These results highlight FG-CLIP's effectiveness in capturing fine-grained image details and improving overall model performance. The related data, code, and models are available at this https URL. 

---
# Teochew-Wild: The First In-the-wild Teochew Dataset with Orthographic Annotations 

**Authors**: Linrong Pan, Chenglong Jiang, Gaoze Hou, Ying Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.05056)  

**Abstract**: This paper reports the construction of the Teochew-Wild, a speech corpus of the Teochew dialect. The corpus includes 18.9 hours of in-the-wild Teochew speech data from multiple speakers, covering both formal and colloquial expressions, with precise orthographic and pinyin annotations. Additionally, we provide supplementary text processing tools and resources to propel research and applications in speech tasks for this low-resource language, such as automatic speech recognition (ASR) and text-to-speech (TTS). To the best of our knowledge, this is the first publicly available Teochew dataset with accurate orthographic annotations. We conduct experiments on the corpus, and the results validate its effectiveness in ASR and TTS tasks. 

---
# Direct Image Classification from Fourier Ptychographic Microscopy Measurements without Reconstruction 

**Authors**: Navya Sonal Agarwal, Jan Philipp Schneider, Kanchana Vaishnavi Gandikota, Syed Muhammad Kazim, John Meshreki, Ivo Ihrke, Michael Moeller  

**Link**: [PDF](https://arxiv.org/pdf/2505.05054)  

**Abstract**: The computational imaging technique of Fourier Ptychographic Microscopy (FPM) enables high-resolution imaging with a wide field of view and can serve as an extremely valuable tool, e.g. in the classification of cells in medical applications. However, reconstructing a high-resolution image from tens or even hundreds of measurements is computationally expensive, particularly for a wide field of view. Therefore, in this paper, we investigate the idea of classifying the image content in the FPM measurements directly without performing a reconstruction step first. We show that Convolutional Neural Networks (CNN) can extract meaningful information from measurement sequences, significantly outperforming the classification on a single band-limited image (up to 12 %) while being significantly more efficient than a reconstruction of a high-resolution image. Furthermore, we demonstrate that a learned multiplexing of several raw measurements allows maintaining the classification accuracy while reducing the amount of data (and consequently also the acquisition time) significantly. 

---
# Image-Text Relation Prediction for Multilingual Tweets 

**Authors**: Matīss Rikters, Edison Marrese-Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2505.05040)  

**Abstract**: Various social networks have been allowing media uploads for over a decade now. Still, it has not always been clear what is their relation with the posted text or even if there is any at all. In this work, we explore how multilingual vision-language models tackle the task of image-text relation prediction in different languages, and construct a dedicated balanced benchmark data set from Twitter posts in Latvian along with their manual translations into English. We compare our results to previous work and show that the more recently released vision-language model checkpoints are becoming increasingly capable at this task, but there is still much room for further improvement. 

---
# Generating Reliable Synthetic Clinical Trial Data: The Role of Hyperparameter Optimization and Domain Constraints 

**Authors**: Waldemar Hahn, Jan-Niklas Eckardt, Christoph Röllig, Martin Sedlmayr, Jan Moritz Middeke, Markus Wolfien  

**Link**: [PDF](https://arxiv.org/pdf/2505.05019)  

**Abstract**: The generation of synthetic clinical trial data offers a promising approach to mitigating privacy concerns and data accessibility limitations in medical research. However, ensuring that synthetic datasets maintain high fidelity, utility, and adherence to domain-specific constraints remains a key challenge. While hyperparameter optimization (HPO) has been shown to improve generative model performance, the effectiveness of different optimization strategies for synthetic clinical data remains unclear. This study systematically evaluates four HPO strategies across eight generative models, comparing single-metric optimization against compound metric optimization approaches. Our results demonstrate that HPO consistently improves synthetic data quality, with TVAE, CTGAN, and CTAB-GAN+ achieving improvements of up to 60%, 39%, and 38%, respectively. Compound metric optimization outperformed single-metric strategies, producing more balanced and generalizable synthetic datasets. Interestingly, HPO alone is insufficient to ensure clinically valid synthetic data, as all models exhibited violations of fundamental survival constraints. Preprocessing and postprocessing played a crucial role in reducing these violations, as models lacking robust processing steps produced invalid data in up to 61% of cases. These findings underscore the necessity of integrating explicit domain knowledge alongside HPO to create high quality synthetic datasets. Our study provides actionable recommendations for improving synthetic data generation, with future research needed to refine metric selection and validate these findings on larger datasets to enhance clinical applicability. 

---
# An Agent-Based Modeling Approach to Free-Text Keyboard Dynamics for Continuous Authentication 

**Authors**: Roberto Dillon, Arushi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05015)  

**Abstract**: Continuous authentication systems leveraging free-text keyboard dynamics offer a promising additional layer of security in a multifactor authentication setup that can be used in a transparent way with no impact on user experience. This study investigates the efficacy of behavioral biometrics by employing an Agent-Based Model (ABM) to simulate diverse typing profiles across mechanical and membrane keyboards. Specifically, we generated synthetic keystroke data from five unique agents, capturing features related to dwell time, flight time, and error rates within sliding 5-second windows updated every second. Two machine learning approaches, One-Class Support Vector Machine (OC-SVM) and Random Forest (RF), were evaluated for user verification. Results revealed a stark contrast in performance: while One-Class SVM failed to differentiate individual users within each group, Random Forest achieved robust intra-keyboard user recognition (Accuracy > 0.7) but struggled to generalize across keyboards for the same user, highlighting the significant impact of keyboard hardware on typing behavior. These findings suggest that: (1) keyboard-specific user profiles may be necessary for reliable authentication, and (2) ensemble methods like RF outperform One-Class SVM in capturing fine-grained user-specific patterns. 

---
# StabStitch++: Unsupervised Online Video Stitching with Spatiotemporal Bidirectional Warps 

**Authors**: Lang Nie, Chunyu Lin, Kang Liao, Yun Zhang, Shuaicheng Liu, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.05001)  

**Abstract**: We retarget video stitching to an emerging issue, named warping shake, which unveils the temporal content shakes induced by sequentially unsmooth warps when extending image stitching to video stitching. Even if the input videos are stable, the stitched video can inevitably cause undesired warping shakes and affect the visual experience. To address this issue, we propose StabStitch++, a novel video stitching framework to realize spatial stitching and temporal stabilization with unsupervised learning simultaneously. First, different from existing learning-based image stitching solutions that typically warp one image to align with another, we suppose a virtual midplane between original image planes and project them onto it. Concretely, we design a differentiable bidirectional decomposition module to disentangle the homography transformation and incorporate it into our spatial warp, evenly spreading alignment burdens and projective distortions across two views. Then, inspired by camera paths in video stabilization, we derive the mathematical expression of stitching trajectories in video stitching by elaborately integrating spatial and temporal warps. Finally, a warp smoothing model is presented to produce stable stitched videos with a hybrid loss to simultaneously encourage content alignment, trajectory smoothness, and online collaboration. Compared with StabStitch that sacrifices alignment for stabilization, StabStitch++ makes no compromise and optimizes both of them simultaneously, especially in the online mode. To establish an evaluation benchmark and train the learning framework, we build a video stitching dataset with a rich diversity in camera motions and scenes. Experiments exhibit that StabStitch++ surpasses current solutions in stitching performance, robustness, and efficiency, offering compelling advancements in this field by building a real-time online video stitching system. 

---
# Rethinking Invariance in In-context Learning 

**Authors**: Lizhe Fang, Yifei Wang, Khashayar Gatmiry, Lei Fang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04994)  

**Abstract**: In-Context Learning (ICL) has emerged as a pivotal capability of auto-regressive large language models, yet it is hindered by a notable sensitivity to the ordering of context examples regardless of their mutual independence. To address this issue, recent studies have introduced several variant algorithms of ICL that achieve permutation invariance. However, many of these do not exhibit comparable performance with the standard auto-regressive ICL algorithm. In this work, we identify two crucial elements in the design of an invariant ICL algorithm: information non-leakage and context interdependence, which are not simultaneously achieved by any of the existing methods. These investigations lead us to the proposed Invariant ICL (InvICL), a methodology designed to achieve invariance in ICL while ensuring the two properties. Empirically, our findings reveal that InvICL surpasses previous models, both invariant and non-invariant, in most benchmark datasets, showcasing superior generalization capabilities across varying input lengths. Code is available at this https URL. 

---
# Decomposition of Probabilities of Causation with Two Mediators 

**Authors**: Yuta Kawakami, Jin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04983)  

**Abstract**: Mediation analysis for probabilities of causation (PoC) provides a fundamental framework for evaluating the necessity and sufficiency of treatment in provoking an event through different causal pathways. One of the primary objectives of causal mediation analysis is to decompose the total effect into path-specific components. In this study, we investigate the path-specific probability of necessity and sufficiency (PNS) to decompose the total PNS into path-specific components along distinct causal pathways between treatment and outcome, incorporating two mediators. We define the path-specific PNS for decomposition and provide an identification theorem. Furthermore, we conduct numerical experiments to assess the properties of the proposed estimators from finite samples and demonstrate their practical application using a real-world educational dataset. 

---
# ChainMarks: Securing DNN Watermark with Cryptographic Chain 

**Authors**: Brian Choi, Shu Wang, Isabelle Choi, Kun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.04977)  

**Abstract**: With the widespread deployment of deep neural network (DNN) models, dynamic watermarking techniques are being used to protect the intellectual property of model owners. However, recent studies have shown that existing watermarking schemes are vulnerable to watermark removal and ambiguity attacks. Besides, the vague criteria for determining watermark presence further increase the likelihood of such attacks. In this paper, we propose a secure DNN watermarking scheme named ChainMarks, which generates secure and robust watermarks by introducing a cryptographic chain into the trigger inputs and utilizes a two-phase Monte Carlo method for determining watermark presence. First, ChainMarks generates trigger inputs as a watermark dataset by repeatedly applying a hash function over a secret key, where the target labels associated with trigger inputs are generated from the digital signature of model owner. Then, the watermarked model is produced by training a DNN over both the original and watermark datasets. To verify watermarks, we compare the predicted labels of trigger inputs with the target labels and determine ownership with a more accurate decision threshold that considers the classification probability of specific models. Experimental results show that ChainMarks exhibits higher levels of robustness and security compared to state-of-the-art watermarking schemes. With a better marginal utility, ChainMarks provides a higher probability guarantee of watermark presence in DNN models with the same level of watermark accuracy. 

---
# AI and Vision based Autonomous Navigation of Nano-Drones in Partially-Known Environments 

**Authors**: Mattia Sartori, Chetna Singhal, Neelabhro Roy, Davide Brunelli, James Gross  

**Link**: [PDF](https://arxiv.org/pdf/2505.04972)  

**Abstract**: The miniaturisation of sensors and processors, the advancements in connected edge intelligence, and the exponential interest in Artificial Intelligence are boosting the affirmation of autonomous nano-size drones in the Internet of Robotic Things ecosystem. However, achieving safe autonomous navigation and high-level tasks such as exploration and surveillance with these tiny platforms is extremely challenging due to their limited resources. This work focuses on enabling the safe and autonomous flight of a pocket-size, 30-gram platform called Crazyflie 2.1 in a partially known environment. We propose a novel AI-aided, vision-based reactive planning method for obstacle avoidance under the ambit of Integrated Sensing, Computing and Communication paradigm. We deal with the constraints of the nano-drone by splitting the navigation task into two parts: a deep learning-based object detector runs on the edge (external hardware) while the planning algorithm is executed onboard. The results show the ability to command the drone at $\sim8$ frames-per-second and a model performance reaching a COCO mean-average-precision of $60.8$. Field experiments demonstrate the feasibility of the solution with the drone flying at a top speed of $1$ m/s while steering away from an obstacle placed in an unknown position and reaching the target destination. The outcome highlights the compatibility of the communication delay and the model performance with the requirements of the real-time navigation task. We provide a feasible alternative to a fully onboard implementation that can be extended to autonomous exploration with nano-drones. 

---
# Moments of Causal Effects 

**Authors**: Yuta Kawakami, Jin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04971)  

**Abstract**: The moments of random variables are fundamental statistical measures for characterizing the shape of a probability distribution, encompassing metrics such as mean, variance, skewness, and kurtosis. Additionally, the product moments, including covariance and correlation, reveal the relationships between multiple random variables. On the other hand, the primary focus of causal inference is the evaluation of causal effects, which are defined as the difference between two potential outcomes. While traditional causal effect assessment focuses on the average causal effect, this work provides definitions, identification theorems, and bounds for moments and product moments of causal effects to analyze their distribution and relationships. We conduct experiments to illustrate the estimation of the moments of causal effects from finite samples and demonstrate their practical application using a real-world medical dataset. 

---
# ADD: Physics-Based Motion Imitation with Adversarial Differential Discriminators 

**Authors**: Ziyu Zhang, Sergey Bashkirov, Dun Yang, Michael Taylor, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04961)  

**Abstract**: Multi-objective optimization problems, which require the simultaneous optimization of multiple terms, are prevalent across numerous applications. Existing multi-objective optimization methods often rely on manually tuned aggregation functions to formulate a joint optimization target. The performance of such hand-tuned methods is heavily dependent on careful weight selection, a time-consuming and laborious process. These limitations also arise in the setting of reinforcement-learning-based motion tracking for physically simulated characters, where intricately crafted reward functions are typically used to achieve high-fidelity results. Such solutions not only require domain expertise and significant manual adjustment, but also limit the applicability of the resulting reward function across diverse skills. To bridge this gap, we present a novel adversarial multi-objective optimization technique that is broadly applicable to a range of multi-objective optimization problems, including motion tracking. The proposed adversarial differential discriminator receives a single positive sample, yet is still effective at guiding the optimization process. We demonstrate that our technique can enable characters to closely replicate a variety of acrobatic and agile behaviors, achieving comparable quality to state-of-the-art motion-tracking methods, without relying on manually tuned reward functions. Results are best visualized through this https URL. 

---
# Graffe: Graph Representation Learning via Diffusion Probabilistic Models 

**Authors**: Dingshuo Chen, Shuchen Xue, Liuji Chen, Yingheng Wang, Qiang Liu, Shu Wu, Zhi-Ming Ma, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04956)  

**Abstract**: Diffusion probabilistic models (DPMs), widely recognized for their potential to generate high-quality samples, tend to go unnoticed in representation learning. While recent progress has highlighted their potential for capturing visual semantics, adapting DPMs to graph representation learning remains in its infancy. In this paper, we introduce Graffe, a self-supervised diffusion model proposed for graph representation learning. It features a graph encoder that distills a source graph into a compact representation, which, in turn, serves as the condition to guide the denoising process of the diffusion decoder. To evaluate the effectiveness of our model, we first explore the theoretical foundations of applying diffusion models to representation learning, proving that the denoising objective implicitly maximizes the conditional mutual information between data and its representation. Specifically, we prove that the negative logarithm of the denoising score matching loss is a tractable lower bound for the conditional mutual information. Empirically, we conduct a series of case studies to validate our theoretical insights. In addition, Graffe delivers competitive results under the linear probing setting on node and graph classification tasks, achieving state-of-the-art performance on 9 of the 11 real-world datasets. These findings indicate that powerful generative models, especially diffusion models, serve as an effective tool for graph representation learning. 

---
# Chain-of-Thought Tokens are Computer Program Variables 

**Authors**: Fangwei Zhu, Peiyi Wang, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04955)  

**Abstract**: Chain-of-thoughts (CoT) requires large language models (LLMs) to generate intermediate steps before reaching the final answer, and has been proven effective to help LLMs solve complex reasoning tasks. However, the inner mechanism of CoT still remains largely unclear. In this paper, we empirically study the role of CoT tokens in LLMs on two compositional tasks: multi-digit multiplication and dynamic programming. While CoT is essential for solving these problems, we find that preserving only tokens that store intermediate results would achieve comparable performance. Furthermore, we observe that storing intermediate results in an alternative latent form will not affect model performance. We also randomly intervene some values in CoT, and notice that subsequent CoT tokens and the final answer would change correspondingly. These findings suggest that CoT tokens may function like variables in computer programs but with potential drawbacks like unintended shortcuts and computational complexity limits between tokens. The code and data are available at this https URL. 

---
# T2VTextBench: A Human Evaluation Benchmark for Textual Control in Video Generation Models 

**Authors**: Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, Jiale Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04946)  

**Abstract**: Thanks to recent advancements in scalable deep architectures and large-scale pretraining, text-to-video generation has achieved unprecedented capabilities in producing high-fidelity, instruction-following content across a wide range of styles, enabling applications in advertising, entertainment, and education. However, these models' ability to render precise on-screen text, such as captions or mathematical formulas, remains largely untested, posing significant challenges for applications requiring exact textual accuracy. In this work, we introduce T2VTextBench, the first human-evaluation benchmark dedicated to evaluating on-screen text fidelity and temporal consistency in text-to-video models. Our suite of prompts integrates complex text strings with dynamic scene changes, testing each model's ability to maintain detailed instructions across frames. We evaluate ten state-of-the-art systems, ranging from open-source solutions to commercial offerings, and find that most struggle to generate legible, consistent text. These results highlight a critical gap in current video generators and provide a clear direction for future research aimed at enhancing textual manipulation in video synthesis. 

---
# Structural Alignment in Link Prediction 

**Authors**: Jeffrey Seathrún Sardina  

**Link**: [PDF](https://arxiv.org/pdf/2505.04939)  

**Abstract**: While Knowledge Graphs (KGs) have become increasingly popular across various scientific disciplines for their ability to model and interlink huge quantities of data, essentially all real-world KGs are known to be incomplete. As such, with the growth of KG use has been a concurrent development of machine learning tools designed to predict missing information in KGs, which is referred to as the Link Prediction Task. The majority of state-of-the-art link predictors to date have followed an embedding-based paradigm. In this paradigm, it is assumed that the information content of a KG is best represented by the (individual) vector representations of its nodes and edges, and that therefore node and edge embeddings are particularly well-suited to performing link prediction.
This thesis proposes an alternative perspective on the field's approach to link prediction and KG data modelling. Specifically, this work re-analyses KGs and state-of-the-art link predictors from a graph-structure-first perspective that models the information content of a KG in terms of whole triples, rather than individual nodes and edges.
Following a literature review and two core sets of experiments, this thesis concludes that a structure-first perspective on KGs and link prediction is both viable and useful for understanding KG learning and for enabling cross-KG transfer learning for the link prediction task. This observation is used to create and propose the Structural Alignment Hypothesis, which postulates that link prediction can be understood and modelled as a structural task.
All code and data used for this thesis are open-sourced. This thesis was written bilingually, with the main document in English and an informal extended summary in Irish. An Irish-language translation dictionary of machine learning terms (the Foclóir Tráchtais) created for this work is open-sourced as well. 

---
# Fair Uncertainty Quantification for Depression Prediction 

**Authors**: Yonghong Li, Xiuzhuang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04931)  

**Abstract**: Trustworthy depression prediction based on deep learning, incorporating both predictive reliability and algorithmic fairness across diverse demographic groups, is crucial for clinical application. Recently, achieving reliable depression predictions through uncertainty quantification has attracted increasing attention. However, few studies have focused on the fairness of uncertainty quantification (UQ) in depression prediction. In this work, we investigate the algorithmic fairness of UQ, namely Equal Opportunity Coverage (EOC) fairness, and propose Fair Uncertainty Quantification (FUQ) for depression prediction. FUQ pursues reliable and fair depression predictions through group-based analysis. Specifically, we first group all the participants by different sensitive attributes and leverage conformal prediction to quantify uncertainty within each demographic group, which provides a theoretically guaranteed and valid way to quantify uncertainty for depression prediction and facilitates the investigation of fairness across different demographic groups. Furthermore, we propose a fairness-aware optimization strategy that formulates fairness as a constrained optimization problem under EOC constraints. This enables the model to preserve predictive reliability while adapting to the heterogeneous uncertainty levels across demographic groups, thereby achieving optimal fairness. Through extensive evaluations on several visual and audio depression datasets, our approach demonstrates its effectiveness. 

---
# Physics-Assisted and Topology-Informed Deep Learning for Weather Prediction 

**Authors**: Jiaqi Zheng, Qing Ling, Yerong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04918)  

**Abstract**: Although deep learning models have demonstrated remarkable potential in weather prediction, most of them overlook either the \textbf{physics} of the underlying weather evolution or the \textbf{topology} of the Earth's surface. In light of these disadvantages, we develop PASSAT, a novel Physics-ASSisted And Topology-informed deep learning model for weather prediction. PASSAT attributes the weather evolution to two key factors: (i) the advection process that can be characterized by the advection equation and the Navier-Stokes equation; (ii) the Earth-atmosphere interaction that is difficult to both model and calculate. PASSAT also takes the topology of the Earth's surface into consideration, other than simply treating it as a plane. With these considerations, PASSAT numerically solves the advection equation and the Navier-Stokes equation on the spherical manifold, utilizes a spherical graph neural network to capture the Earth-atmosphere interaction, and generates the initial velocity fields that are critical to solving the advection equation from the same spherical graph neural network. In the $5.625^\circ$-resolution ERA5 data set, PASSAT outperforms both the state-of-the-art deep learning-based weather prediction models and the operational numerical weather prediction model IFS T42. Code and checkpoint are available at this https URL. 

---
# An Open-Source Dual-Loss Embedding Model for Semantic Retrieval in Higher Education 

**Authors**: Ramteja Sajja, Yusuf Sermet, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2505.04916)  

**Abstract**: Recent advances in AI have catalyzed the adoption of intelligent educational tools, yet many semantic retrieval systems remain ill-suited to the unique linguistic and structural characteristics of academic content. This study presents two open-source embedding models fine-tuned for educational question answering, particularly in the context of course syllabi. A synthetic dataset of 3,197 sentence pairs, spanning synonymous terminology, paraphrased questions, and implicit-explicit mappings, was constructed through a combination of manual curation and large language model (LLM)-assisted generation. Two training strategies were evaluated: (1) a baseline model fine-tuned using MultipleNegativesRankingLoss (MNRL), and (2) a dual-loss model that combines MNRL with CosineSimilarityLoss to improve both semantic ranking and similarity calibration. Evaluations were conducted on 28 university course syllabi using a fixed set of natural language questions categorized into course, faculty, and teaching assistant information. Results demonstrate that both fine-tuned models outperform strong open-source baselines, including all-MiniLM-L6-v2 and multi-qa-MiniLM-L6-cos-v1, and that the dual-loss model narrows the performance gap with high-performing proprietary embeddings such as OpenAI's text-embedding-3 series. This work contributes reusable, domain-aligned embedding models and provides a replicable framework for educational semantic retrieval, supporting downstream applications such as academic chatbots, retrieval-augmented generation (RAG) systems, and learning management system (LMS) integrations. 

---
# SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models 

**Authors**: Shun Taguchi, Hideki Deguchi, Takumi Hamazaki, Hiroyuki Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2505.04911)  

**Abstract**: This study introduces SpatialPrompting, a novel framework that harnesses the emergent reasoning capabilities of off-the-shelf multimodal large language models to achieve zero-shot spatial reasoning in three-dimensional (3D) environments. Unlike existing methods that rely on expensive 3D-specific fine-tuning with specialized 3D inputs such as point clouds or voxel-based features, SpatialPrompting employs a keyframe-driven prompt generation strategy. This framework uses metrics such as vision-language similarity, Mahalanobis distance, field of view, and image sharpness to select a diverse and informative set of keyframes from image sequences and then integrates them with corresponding camera pose data to effectively abstract spatial relationships and infer complex 3D structures. The proposed framework not only establishes a new paradigm for flexible spatial reasoning that utilizes intuitive visual and positional cues but also achieves state-of-the-art zero-shot performance on benchmark datasets, such as ScanQA and SQA3D, across several metrics. The proposed method effectively eliminates the need for specialized 3D inputs and fine-tuning, offering a simpler and more scalable alternative to conventional approaches. 

---
# Precise gradient descent training dynamics for finite-width multi-layer neural networks 

**Authors**: Qiyang Han, Masaaki Imaizumi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04898)  

**Abstract**: In this paper, we provide the first precise distributional characterization of gradient descent iterates for general multi-layer neural networks under the canonical single-index regression model, in the `finite-width proportional regime' where the sample size and feature dimension grow proportionally while the network width and depth remain bounded. Our non-asymptotic state evolution theory captures Gaussian fluctuations in first-layer weights and concentration in deeper-layer weights, and remains valid for non-Gaussian features.
Our theory differs from existing neural tangent kernel (NTK), mean-field (MF) theories and tensor program (TP) in several key aspects. First, our theory operates in the finite-width regime whereas these existing theories are fundamentally infinite-width. Second, our theory allows weights to evolve from individual initializations beyond the lazy training regime, whereas NTK and MF are either frozen at or only weakly sensitive to initialization, and TP relies on special initialization schemes. Third, our theory characterizes both training and generalization errors for general multi-layer neural networks beyond the uniform convergence regime, whereas existing theories study generalization almost exclusively in two-layer settings.
As a statistical application, we show that vanilla gradient descent can be augmented to yield consistent estimates of the generalization error at each iteration, which can be used to guide early stopping and hyperparameter tuning. As a further theoretical implication, we show that despite model misspecification, the model learned by gradient descent retains the structure of a single-index function with an effective signal determined by a linear combination of the true signal and the initialization. 

---
# Clustering with Communication: A Variational Framework for Single Cell Representation Learning 

**Authors**: Cong Qi, Yeqing Chen, Jie Zhang, Wei Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04891)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) has revealed complex cellular heterogeneity, but recent studies emphasize that understanding biological function also requires modeling cell-cell communication (CCC), the signaling interactions mediated by ligand-receptor pairs that coordinate cellular behavior. Tools like CellChat have demonstrated that CCC plays a critical role in processes such as cell differentiation, tissue regeneration, and immune response, and that transcriptomic data inherently encodes rich information about intercellular signaling. We propose CCCVAE, a novel variational autoencoder framework that incorporates CCC signals into single-cell representation learning. By leveraging a communication-aware kernel derived from ligand-receptor interactions and a sparse Gaussian process, CCCVAE encodes biologically informed priors into the latent space. Unlike conventional VAEs that treat each cell independently, CCCVAE encourages latent embeddings to reflect both transcriptional similarity and intercellular signaling context. Empirical results across four scRNA-seq datasets show that CCCVAE improves clustering performance, achieving higher evaluation scores than standard VAE baselines. This work demonstrates the value of embedding biological priors into deep generative models for unsupervised single-cell analysis. 

---
# Cross-Branch Orthogonality for Improved Generalization in Face Deepfake Detection 

**Authors**: Tharindu Fernando, Clinton Fookes, Sridha Sridharan, Simon Denman  

**Link**: [PDF](https://arxiv.org/pdf/2505.04888)  

**Abstract**: Remarkable advancements in generative AI technology have given rise to a spectrum of novel deepfake categories with unprecedented leaps in their realism, and deepfakes are increasingly becoming a nuisance to law enforcement authorities and the general public. In particular, we observe alarming levels of confusion, deception, and loss of faith regarding multimedia content within society caused by face deepfakes, and existing deepfake detectors are struggling to keep up with the pace of improvements in deepfake generation. This is primarily due to their reliance on specific forgery artifacts, which limits their ability to generalise and detect novel deepfake types. To combat the spread of malicious face deepfakes, this paper proposes a new strategy that leverages coarse-to-fine spatial information, semantic information, and their interactions while ensuring feature distinctiveness and reducing the redundancy of the modelled features. A novel feature orthogonality-based disentanglement strategy is introduced to ensure branch-level and cross-branch feature disentanglement, which allows us to integrate multiple feature vectors without adding complexity to the feature space or compromising generalisation. Comprehensive experiments on three public benchmarks: FaceForensics++, Celeb-DF, and the Deepfake Detection Challenge (DFDC) show that these design choices enable the proposed approach to outperform current state-of-the-art methods by 5% on the Celeb-DF dataset and 7% on the DFDC dataset in a cross-dataset evaluation setting. 

---
# QBR: A Question-Bank-Based Approach to Fine-Grained Legal Knowledge Retrieval for the General Public 

**Authors**: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04883)  

**Abstract**: Retrieval of legal knowledge by the general public is a challenging problem due to the technicality of the professional knowledge and the lack of fundamental understanding by laypersons on the subject. Traditional information retrieval techniques assume that users are capable of formulating succinct and precise queries for effective document retrieval. In practice, however, the wide gap between the highly technical contents and untrained users makes legal knowledge retrieval very difficult. We propose a methodology, called QBR, which employs a Questions Bank (QB) as an effective medium for bridging the knowledge gap. We show how the QB is used to derive training samples to enhance the embedding of knowledge units within documents, which leads to effective fine-grained knowledge retrieval. We discuss and evaluate through experiments various advantages of QBR over traditional methods. These include more accurate, efficient, and explainable document retrieval, better comprehension of retrieval results, and highly effective fine-grained knowledge retrieval. We also present some case studies and show that QBR achieves social impact by assisting citizens to resolve everyday legal concerns. 

---
# ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning 

**Authors**: Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04881)  

**Abstract**: Large Reasoning Models (LRMs) perform strongly in complex reasoning tasks via Chain-of-Thought (CoT) prompting, but often suffer from verbose outputs caused by redundant content, increasing computational overhead, and degrading user experience. Existing compression methods either operate post-hoc pruning, risking disruption to reasoning coherence, or rely on sampling-based selection, which fails to intervene effectively during generation. In this work, we introduce a confidence-guided perspective to explain the emergence of redundant reflection in LRMs, identifying two key patterns: Confidence Deficit, where the model reconsiders correct steps due to low internal confidence, and Termination Delay, where reasoning continues even after reaching a confident answer. Based on this analysis, we propose ConCISE (Confidence-guided Compression In Step-by-step Efficient Reasoning), a framework that simplifies reasoning chains by reinforcing the model's confidence during inference, thus preventing the generation of redundant reflection steps. It integrates Confidence Injection to stabilize intermediate steps and Early Stopping to terminate reasoning when confidence is sufficient. Extensive experiments demonstrate that fine-tuning LRMs on ConCISE-generated data yields significantly shorter outputs, reducing length by up to approximately 50% under SimPO, while maintaining high task accuracy. ConCISE consistently outperforms existing baselines across multiple reasoning benchmarks. 

---
# GroverGPT-2: Simulating Grover's Algorithm via Chain-of-Thought Reasoning and Quantum-Native Tokenization 

**Authors**: Min Chen, Jinglei Cheng, Pingzhi Li, Haoran Wang, Tianlong Chen, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04880)  

**Abstract**: Quantum computing offers theoretical advantages over classical computing for specific tasks, yet the boundary of practical quantum advantage remains an open question. To investigate this boundary, it is crucial to understand whether, and how, classical machines can learn and simulate quantum algorithms. Recent progress in large language models (LLMs) has demonstrated strong reasoning abilities, prompting exploration into their potential for this challenge. In this work, we introduce GroverGPT-2, an LLM-based method for simulating Grover's algorithm using Chain-of-Thought (CoT) reasoning and quantum-native tokenization. Building on its predecessor, GroverGPT-2 performs simulation directly from quantum circuit representations while producing logically structured and interpretable outputs. Our results show that GroverGPT-2 can learn and internalize quantum circuit logic through efficient processing of quantum-native tokens, providing direct evidence that classical models like LLMs can capture the structure of quantum algorithms. Furthermore, GroverGPT-2 outputs interleave circuit data with natural language, embedding explicit reasoning into the simulation. This dual capability positions GroverGPT-2 as a prototype for advancing machine understanding of quantum algorithms and modeling quantum circuit logic. We also identify an empirical scaling law for GroverGPT-2 with increasing qubit numbers, suggesting a path toward scalable classical simulation. These findings open new directions for exploring the limits of classical simulatability, enhancing quantum education and research, and laying groundwork for future foundation models in quantum computing. 

---
# Learning from Loss Landscape: Generalizable Mixed-Precision Quantization via Adaptive Sharpness-Aware Gradient Aligning 

**Authors**: Lianbo Ma, Jianlun Ma, Yuee Zhou, Guoyang Xie, Qiang He, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04877)  

**Abstract**: Mixed Precision Quantization (MPQ) has become an essential technique for optimizing neural network by determining the optimal bitwidth per layer. Existing MPQ methods, however, face a major hurdle: they require a computationally expensive search for quantization policies on large-scale datasets. To resolve this issue, we introduce a novel approach that first searches for quantization policies on small datasets and then generalizes them to large-scale datasets. This approach simplifies the process, eliminating the need for large-scale quantization fine-tuning and only necessitating model weight adjustment. Our method is characterized by three key techniques: sharpness-aware minimization for enhanced quantization generalization, implicit gradient direction alignment to handle gradient conflicts among different optimization objectives, and an adaptive perturbation radius to accelerate optimization. Both theoretical analysis and experimental results validate our approach. Using the CIFAR10 dataset (just 0.5\% the size of ImageNet training data) for MPQ policy search, we achieved equivalent accuracy on ImageNet with a significantly lower computational cost, while improving efficiency by up to 150% over the baselines. 

---
# Federated Learning for Cyber Physical Systems: A Comprehensive Survey 

**Authors**: Minh K. Quan, Pubudu N. Pathirana, Mayuri Wijayasundara, Sujeeva Setunge, Dinh C. Nguyen, Christopher G. Brinton, David J. Love, H. Vincent Poor  

**Link**: [PDF](https://arxiv.org/pdf/2505.04873)  

**Abstract**: The integration of machine learning (ML) in cyber physical systems (CPS) is a complex task due to the challenges that arise in terms of real-time decision making, safety, reliability, device heterogeneity, and data privacy. There are also open research questions that must be addressed in order to fully realize the potential of ML in CPS. Federated learning (FL), a distributed approach to ML, has become increasingly popular in recent years. It allows models to be trained using data from decentralized sources. This approach has been gaining popularity in the CPS field, as it integrates computer, communication, and physical processes. Therefore, the purpose of this work is to provide a comprehensive analysis of the most recent developments of FL-CPS, including the numerous application areas, system topologies, and algorithms developed in recent years. The paper starts by discussing recent advances in both FL and CPS, followed by their integration. Then, the paper compares the application of FL in CPS with its applications in the internet of things (IoT) in further depth to show their connections and distinctions. Furthermore, the article scrutinizes how FL is utilized in critical CPS applications, e.g., intelligent transportation systems, cybersecurity services, smart cities, and smart healthcare solutions. The study also includes critical insights and lessons learned from various FL-CPS implementations. The paper's concluding section delves into significant concerns and suggests avenues for further research in this fast-paced and dynamic era. 

---
# Auto-regressive transformation for image alignment 

**Authors**: Kanggeon Lee, Soochahn Lee, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.04864)  

**Abstract**: Existing methods for image alignment struggle in cases involving feature-sparse regions, extreme scale and field-of-view differences, and large deformations, often resulting in suboptimal accuracy. Robustness to these challenges improves through iterative refinement of the transformation field while focusing on critical regions in multi-scale image representations. We thus propose Auto-Regressive Transformation (ART), a novel method that iteratively estimates the coarse-to-fine transformations within an auto-regressive framework. Leveraging hierarchical multi-scale features, our network refines the transformations using randomly sampled points at each scale. By incorporating guidance from the cross-attention layer, the model focuses on critical regions, ensuring accurate alignment even in challenging, feature-limited conditions. Extensive experiments across diverse datasets demonstrate that ART significantly outperforms state-of-the-art methods, establishing it as a powerful new method for precise image alignment with broad applicability. 

---
# D-CODA: Diffusion for Coordinated Dual-Arm Data Augmentation 

**Authors**: I-Chun Arthur Liu, Jason Chen, Gaurav Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2505.04860)  

**Abstract**: Learning bimanual manipulation is challenging due to its high dimensionality and tight coordination required between two arms. Eye-in-hand imitation learning, which uses wrist-mounted cameras, simplifies perception by focusing on task-relevant views. However, collecting diverse demonstrations remains costly, motivating the need for scalable data augmentation. While prior work has explored visual augmentation in single-arm settings, extending these approaches to bimanual manipulation requires generating viewpoint-consistent observations across both arms and producing corresponding action labels that are both valid and feasible. In this work, we propose Diffusion for COordinated Dual-arm Data Augmentation (D-CODA), a method for offline data augmentation tailored to eye-in-hand bimanual imitation learning that trains a diffusion model to synthesize novel, viewpoint-consistent wrist-camera images for both arms while simultaneously generating joint-space action labels. It employs constrained optimization to ensure that augmented states involving gripper-to-object contacts adhere to constraints suitable for bimanual coordination. We evaluate D-CODA on 5 simulated and 3 real-world tasks. Our results across 2250 simulation trials and 300 real-world trials demonstrate that it outperforms baselines and ablations, showing its potential for scalable data augmentation in eye-in-hand bimanual manipulation. Our project website is at: this https URL. 

---
# PR2: Peephole Raw Pointer Rewriting with LLMs for Translating C to Safer Rust 

**Authors**: Yifei Gao, Chengpeng Wang, Pengxiang Huang, Xuwei Liu, Mingwei Zheng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04852)  

**Abstract**: There has been a growing interest in translating C code to Rust due to Rust's robust memory and thread safety guarantees. Tools such as C2RUST enable syntax-guided transpilation from C to semantically equivalent Rust code. However, the resulting Rust programs often rely heavily on unsafe constructs--particularly raw pointers--which undermines Rust's safety guarantees. This paper aims to improve the memory safety of Rust programs generated by C2RUST by eliminating raw pointers. Specifically, we propose a peephole raw pointer rewriting technique that lifts raw pointers in individual functions to appropriate Rust data structures. Technically, PR2 employs decision-tree-based prompting to guide the pointer lifting process. Additionally, it leverages code change analysis to guide the repair of errors introduced during rewriting, effectively addressing errors encountered during compilation and test case execution. We implement PR2 as a prototype and evaluate it using gpt-4o-mini on 28 real-world C projects. The results show that PR2 successfully eliminates 13.22% of local raw pointers across these projects, significantly enhancing the safety of the translated Rust code. On average, PR2 completes the transformation of a project in 5.44 hours, at an average cost of $1.46. 

---
# Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards 

**Authors**: Manveer Singh Tamber, Forrest Sheng Bao, Chenyu Xu, Ge Luo, Suleman Kazi, Minseok Bae, Miaoran Li, Ofer Mendelevitch, Renyi Qu, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04847)  

**Abstract**: Hallucinations remain a persistent challenge for LLMs. RAG aims to reduce hallucinations by grounding responses in contexts. However, even when provided context, LLMs still frequently introduce unsupported information or contradictions. This paper presents our efforts to measure LLM hallucinations with a focus on summarization tasks, assessing how often various LLMs introduce hallucinations when summarizing documents. We discuss Vectara's existing LLM hallucination leaderboard, based on the Hughes Hallucination Evaluation Model (HHEM). While HHEM and Vectara's Hallucination Leaderboard have garnered great research interest, we examine challenges faced by HHEM and current hallucination detection methods by analyzing the effectiveness of these methods on existing hallucination datasets. To address these limitations, we propose FaithJudge, an LLM-as-a-judge approach guided by few-shot human hallucination annotations, which substantially improves automated LLM hallucination evaluation over current methods. We introduce an enhanced hallucination leaderboard centered on FaithJudge, alongside our current hallucination leaderboard, enabling more reliable benchmarking of LLMs for hallucinations in RAG. 

---
# Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers 

**Authors**: Kusha Sareen, Morgane M Moss, Alessandro Sordoni, Rishabh Agarwal, Arian Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2505.04842)  

**Abstract**: Prevalent reinforcement learning~(RL) methods for fine-tuning LLM reasoners, such as GRPO or Leave-one-out PPO, abandon the learned value function in favor of empirically estimated returns. This hinders test-time compute scaling that relies on using the value-function for verification. In this work, we propose RL$^V$ that augments any ``value-free'' RL method by jointly training the LLM as both a reasoner and a generative verifier using RL-generated data, adding verification capabilities without significant overhead. Empirically, RL$^V$ boosts MATH accuracy by over 20\% with parallel sampling and enables $8-32\times$ efficient test-time compute scaling compared to the base RL method. RL$^V$ also exhibits strong generalization capabilities for both easy-to-hard and out-of-domain tasks. Furthermore, RL$^V$ achieves $1.2-1.6\times$ higher performance when jointly scaling parallel and sequential test-time compute with a long reasoning R1 model. 

---
# Quantum-Inspired Optimization Process for Data Imputation 

**Authors**: Nishikanta Mohanty, Bikash K. Behera, Badsah Mukherjee, Christopher Ferrie  

**Link**: [PDF](https://arxiv.org/pdf/2505.04841)  

**Abstract**: Data imputation is a critical step in data pre-processing, particularly for datasets with missing or unreliable values. This study introduces a novel quantum-inspired imputation framework evaluated on the UCI Diabetes dataset, which contains biologically implausible missing values across several clinical features. The method integrates Principal Component Analysis (PCA) with quantum-assisted rotations, optimized through gradient-free classical optimizers -COBYLA, Simulated Annealing, and Differential Evolution to reconstruct missing values while preserving statistical fidelity. Reconstructed values are constrained within +/-2 standard deviations of original feature distributions, avoiding unrealistic clustering around central tendencies. This approach achieves a substantial and statistically significant improvement, including an average reduction of over 85% in Wasserstein distance and Kolmogorov-Smirnov test p-values between 0.18 and 0.22, compared to p-values > 0.99 in classical methods such as Mean, KNN, and MICE. The method also eliminates zero-value artifacts and enhances the realism and variability of imputed data. By combining quantum-inspired transformations with a scalable classical framework, this methodology provides a robust solution for imputation tasks in domains such as healthcare and AI pipelines, where data quality and integrity are crucial. 

---
# Piecewise Constant Spectral Graph Neural Network 

**Authors**: Vahan Martirosyan, Jhony H. Giraldo, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2505.04808)  

**Abstract**: Graph Neural Networks (GNNs) have achieved significant success across various domains by leveraging graph structures in data. Existing spectral GNNs, which use low-degree polynomial filters to capture graph spectral properties, may not fully identify the graph's spectral characteristics because of the polynomial's small degree. However, increasing the polynomial degree is computationally expensive and beyond certain thresholds leads to performance plateaus or degradation. In this paper, we introduce the Piecewise Constant Spectral Graph Neural Network(PieCoN) to address these challenges. PieCoN combines constant spectral filters with polynomial filters to provide a more flexible way to leverage the graph structure. By adaptively partitioning the spectrum into intervals, our approach increases the range of spectral properties that can be effectively learned. Experiments on nine benchmark datasets, including both homophilic and heterophilic graphs, demonstrate that PieCoN is particularly effective on heterophilic datasets, highlighting its potential for a wide range of applications. 

---
# ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling 

**Authors**: Xiao Wang, Jong-Youl Choi, Takuya Kurihaya, Isaac Lyngaas, Hong-Jun Yoon, Ming Fan, Nasik Muhammad Nafi, Aristeidis Tsaris, Ashwin M. Aji, Maliha Hossain, Mohamed Wahib, Dali Wang, Peter Thornton, Prasanna Balaprakash, Moetasim Ashfaq, Dan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04802)  

**Abstract**: Sparse observations and coarse-resolution climate models limit effective regional decision-making, underscoring the need for robust downscaling. However, existing AI methods struggle with generalization across variables and geographies and are constrained by the quadratic complexity of Vision Transformer (ViT) self-attention. We introduce ORBIT-2, a scalable foundation model for global, hyper-resolution climate downscaling. ORBIT-2 incorporates two key innovations: (1) Residual Slim ViT (Reslim), a lightweight architecture with residual learning and Bayesian regularization for efficient, robust prediction; and (2) TILES, a tile-wise sequence scaling algorithm that reduces self-attention complexity from quadratic to linear, enabling long-sequence processing and massive parallelism. ORBIT-2 scales to 10 billion parameters across 32,768 GPUs, achieving up to 1.8 ExaFLOPS sustained throughput and 92-98% strong scaling efficiency. It supports downscaling to 0.9 km global resolution and processes sequences up to 4.2 billion tokens. On 7 km resolution benchmarks, ORBIT-2 achieves high accuracy with R^2 scores in the range of 0.98 to 0.99 against observation data. 

---
# Confabulation dynamics in a reservoir computer: Filling in the gaps with untrained attractors 

**Authors**: Jack O'Hagan, Andrew Keane, Andrew Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2505.04792)  

**Abstract**: Artificial Intelligence has advanced significantly in recent years thanks to innovations in the design and training of artificial neural networks (ANNs). Despite these advancements, we still understand relatively little about how elementary forms of ANNs learn, fail to learn, and generate false information without the intent to deceive, a phenomenon known as `confabulation'. To provide some foundational insight, in this paper we analyse how confabulation occurs in reservoir computers (RCs): a dynamical system in the form of an ANN. RCs are particularly useful to study as they are known to confabulate in a well-defined way: when RCs are trained to reconstruct the dynamics of a given attractor, they sometimes construct an attractor that they were not trained to construct, a so-called `untrained attractor' (UA). This paper sheds light on the role played by UAs when reconstruction fails and their influence when modelling transitions between reconstructed attractors. Based on our results, we conclude that UAs are an intrinsic feature of learning systems whose state spaces are bounded, and that this means of confabulation may be present in systems beyond RCs. 

---
# Replay to Remember (R2R): An Efficient Uncertainty-driven Unsupervised Continual Learning Framework Using Generative Replay 

**Authors**: Sriram Mandalika, Harsha Vardhan, Athira Nambiar  

**Link**: [PDF](https://arxiv.org/pdf/2505.04787)  

**Abstract**: Continual Learning entails progressively acquiring knowledge from new data while retaining previously acquired knowledge, thereby mitigating ``Catastrophic Forgetting'' in neural networks. Our work presents a novel uncertainty-driven Unsupervised Continual Learning framework using Generative Replay, namely ``Replay to Remember (R2R)''. The proposed R2R architecture efficiently uses unlabelled and synthetic labelled data in a balanced proportion using a cluster-level uncertainty-driven feedback mechanism and a VLM-powered generative replay module. Unlike traditional memory-buffer methods that depend on pretrained models and pseudo-labels, our R2R framework operates without any prior training. It leverages visual features from unlabeled data and adapts continuously using clustering-based uncertainty estimation coupled with dynamic thresholding. Concurrently, a generative replay mechanism along with DeepSeek-R1 powered CLIP VLM produces labelled synthetic data representative of past experiences, resembling biological visual thinking that replays memory to remember and act in new, unseen tasks. Extensive experimental analyses are carried out in CIFAR-10, CIFAR-100, CINIC-10, SVHN and TinyImageNet datasets. Our proposed R2R approach improves knowledge retention, achieving a state-of-the-art performance of 98.13%, 73.06%, 93.41%, 95.18%, 59.74%, respectively, surpassing state-of-the-art performance by over 4.36%. 

---
# Flower Across Time and Media: Sentiment Analysis of Tang Song Poetry and Visual Correspondence 

**Authors**: Shuai Gong, Tiange Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04785)  

**Abstract**: The Tang (618 to 907) and Song (960 to 1279) dynasties witnessed an extraordinary flourishing of Chinese cultural expression, where floral motifs served as a dynamic medium for both poetic sentiment and artistic design. While previous scholarship has examined these domains independently, the systematic correlation between evolving literary emotions and visual culture remains underexplored. This study addresses that gap by employing BERT-based sentiment analysis to quantify emotional patterns in floral imagery across Tang Song poetry, then validating these patterns against contemporaneous developments in decorative this http URL approach builds upon recent advances in computational humanities while remaining grounded in traditional sinological methods. By applying a fine tuned BERT model to analyze peony and plum blossom imagery in classical poetry, we detect measurable shifts in emotional connotations between the Tang and Song periods. These textual patterns are then cross berenced with visual evidence from textiles, ceramics, and other material culture, revealing previously unrecognized synergies between literary expression and artistic representation. 

---
# A Proposal for Evaluating the Operational Risk for ChatBots based on Large Language Models 

**Authors**: Pedro Pinacho-Davidson, Fernando Gutierrez, Pablo Zapata, Rodolfo Vergara, Pablo Aqueveque  

**Link**: [PDF](https://arxiv.org/pdf/2505.04784)  

**Abstract**: The emergence of Generative AI (Gen AI) and Large Language Models (LLMs) has enabled more advanced chatbots capable of human-like interactions. However, these conversational agents introduce a broader set of operational risks that extend beyond traditional cybersecurity considerations. In this work, we propose a novel, instrumented risk-assessment metric that simultaneously evaluates potential threats to three key stakeholders: the service-providing organization, end users, and third parties. Our approach incorporates the technical complexity required to induce erroneous behaviors in the chatbot--ranging from non-induced failures to advanced prompt-injection attacks--as well as contextual factors such as the target industry, user age range, and vulnerability severity. To validate our metric, we leverage Garak, an open-source framework for LLM vulnerability testing. We further enhance Garak to capture a variety of threat vectors (e.g., misinformation, code hallucinations, social engineering, and malicious code generation). Our methodology is demonstrated in a scenario involving chatbots that employ retrieval-augmented generation (RAG), showing how the aggregated risk scores guide both short-term mitigation and longer-term improvements in model design and deployment. The results underscore the importance of multi-dimensional risk assessments in operationalizing secure, reliable AI-driven conversational systems. 

---
# Exploring Zero-Shot App Review Classification with ChatGPT: Challenges and Potential 

**Authors**: Mohit Chaudhary, Chirag Jain, Preethu Rose Anish  

**Link**: [PDF](https://arxiv.org/pdf/2505.04759)  

**Abstract**: App reviews are a critical source of user feedback, offering valuable insights into an app's performance, features, usability, and overall user experience. Effectively analyzing these reviews is essential for guiding app development, prioritizing feature updates, and enhancing user satisfaction. Classifying reviews into functional and non-functional requirements play a pivotal role in distinguishing feedback related to specific app features (functional requirements) from feedback concerning broader quality attributes, such as performance, usability, and reliability (non-functional requirements). Both categories are integral to informed development decisions. Traditional approaches to classifying app reviews are hindered by the need for large, domain-specific datasets, which are often costly and time-consuming to curate. This study explores the potential of zero-shot learning with ChatGPT for classifying app reviews into four categories: functional requirement, non-functional requirement, both, or neither. We evaluate ChatGPT's performance on a benchmark dataset of 1,880 manually annotated reviews from ten diverse apps spanning multiple domains. Our findings demonstrate that ChatGPT achieves a robust F1 score of 0.842 in review classification, despite certain challenges and limitations. Additionally, we examine how factors such as review readability and length impact classification accuracy and conduct a manual analysis to identify review categories more prone to misclassification. 

---
# When Bad Data Leads to Good Models 

**Authors**: Kenneth Li, Yida Chen, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.04741)  

**Abstract**: In large language model (LLM) pretraining, data quality is believed to determine model quality. In this paper, we re-examine the notion of "quality" from the perspective of pre- and post-training co-design. Specifically, we explore the possibility that pre-training on more toxic data can lead to better control in post-training, ultimately decreasing a model's output toxicity. First, we use a toy experiment to study how data composition affects the geometry of features in the representation space. Next, through controlled experiments with Olmo-1B models trained on varying ratios of clean and toxic data, we find that the concept of toxicity enjoys a less entangled linear representation as the proportion of toxic data increases. Furthermore, we show that although toxic data increases the generational toxicity of the base model, it also makes the toxicity easier to remove. Evaluations on Toxigen and Real Toxicity Prompts demonstrate that models trained on toxic data achieve a better trade-off between reducing generational toxicity and preserving general capabilities when detoxifying techniques such as inference-time intervention (ITI) are applied. Our findings suggest that, with post-training taken into account, bad data may lead to good models. 

---
# QBD-RankedDataGen: Generating Custom Ranked Datasets for Improving Query-By-Document Search Using LLM-Reranking with Reduced Human Effort 

**Authors**: Sriram Gopalakrishnan, Sunandita Patra  

**Link**: [PDF](https://arxiv.org/pdf/2505.04732)  

**Abstract**: The Query-By-Document (QBD) problem is an information retrieval problem where the query is a document, and the retrieved candidates are documents that match the query document, often in a domain or query specific manner. This can be crucial for tasks such as patent matching, legal or compliance case retrieval, and academic literature review. Existing retrieval methods, including keyword search and document embeddings, can be optimized with domain-specific datasets to improve QBD search performance. However, creating these domain-specific datasets is often costly and time-consuming. Our work introduces a process to generate custom QBD-search datasets and compares a set of methods to use in this problem, which we refer to as QBD-RankedDatagen. We provide a comparative analysis of our proposed methods in terms of cost, speed, and the human interface with the domain experts. The methods we compare leverage Large Language Models (LLMs) which can incorporate domain expert input to produce document scores and rankings, as well as explanations for human review. The process and methods for it that we present can significantly reduce human effort in dataset creation for custom domains while still obtaining sufficient expert knowledge for tuning retrieval models. We evaluate our methods on QBD datasets from the Text Retrieval Conference (TREC) and finetune the parameters of the BM25 model -- which is used in many industrial-strength search engines like OpenSearch -- using the generated data. 

---
# Geometric Fault-Tolerant Neural Network Tracking Control of Unknown Systems on Matrix Lie Groups 

**Authors**: Robin Chhabra, Farzaneh Abdollahi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04725)  

**Abstract**: We present a geometric neural network-based tracking controller for systems evolving on matrix Lie groups under unknown dynamics, actuator faults, and bounded disturbances. Leveraging the left-invariance of the tangent bundle of matrix Lie groups, viewed as an embedded submanifold of the vector space $\R^{N\times N}$, we propose a set of learning rules for neural network weights that are intrinsically compatible with the Lie group structure and do not require explicit parameterization. Exploiting the geometric properties of Lie groups, this approach circumvents parameterization singularities and enables a global search for optimal weights. The ultimate boundedness of all error signals -- including the neural network weights, the coordinate-free configuration error function, and the tracking velocity error -- is established using Lyapunov's direct method. To validate the effectiveness of the proposed method, we provide illustrative simulation results for decentralized formation control of multi-agent systems on the Special Euclidean group. 

---
# Advanced Deep Learning Approaches for Automated Recognition of Cuneiform Symbols 

**Authors**: Shahad Elshehaby, Alavikunhu Panthakkan, Hussain Al-Ahmad, Mina Al-Saad  

**Link**: [PDF](https://arxiv.org/pdf/2505.04678)  

**Abstract**: This paper presents a thoroughly automated method for identifying and interpreting cuneiform characters via advanced deep-learning algorithms. Five distinct deep-learning models were trained on a comprehensive dataset of cuneiform characters and evaluated according to critical performance metrics, including accuracy and precision. Two models demonstrated outstanding performance and were subsequently assessed using cuneiform symbols from the Hammurabi law acquisition, notably Hammurabi Law 1. Each model effectively recognized the relevant Akkadian meanings of the symbols and delivered precise English translations. Future work will investigate ensemble and stacking approaches to optimize performance, utilizing hybrid architectures to improve detection accuracy and reliability. This research explores the linguistic relationships between Akkadian, an ancient Mesopotamian language, and Arabic, emphasizing their historical and cultural linkages. This study demonstrates the capability of deep learning to decipher ancient scripts by merging computational linguistics with archaeology, therefore providing significant insights for the comprehension and conservation of human history. 

---
# Proceedings The 13th International Workshop on Theorem proving components for Educational software 

**Authors**: Julien Narboux, Walther Neuper, Pedro Quaresma  

**Link**: [PDF](https://arxiv.org/pdf/2505.04677)  

**Abstract**: The ThEdu series pursues the smooth transition from an intuitive way of doing mathematics at secondary school to a more formal approach to the subject in STEM education while favoring software support for this transition by exploiting the power of theorem-proving technologies.  What follows is a brief description of how the present volume contributes to this enterprise.  The 13th International Workshop on Theorem Proving Components for Educational Software (ThEdu'24), was a satellite event of the CADE29, part of IJCAR 2024, Nancy, France. ThEdu'24 was a vibrant workshop, with one invited talk by Jeremy Avigad (Carnegie Mellon University) and 14 submitted talks. An open call for papers was then issued and attracted 9 submissions. Eight of those submissions have been accepted by our reviewers. The resulting revised papers are collected in the present volume. The contributions in this volume are a faithful representation of the wide spectrum of ThEdu, ranging from those more focused on the automated deduction research, not losing track of the possible applications in an educational setting, to those focused on the applications, in educational settings, of automated deduction tools and methods. We, the volume editors, hope that this collection of papers will further promote the development of theorem-proving-based software and that it will allow to improve the mutual understanding between computer scientists, mathematicians, and stakeholders in education. While this volume goes to press, the next edition of the ThEdu workshop is being prepared: ThEdu'25 will be a satellite event of the 30th international Conference on Automated DEduction (CADE-30), July 28th - August 2nd, 2025, Stuttgart, Germany. 

---
# REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM 

**Authors**: Madhur Jindal, Saurabh Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2505.04673)  

**Abstract**: Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.
We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$). 

---
# Personalized Risks and Regulatory Strategies of Large Language Models in Digital Advertising 

**Authors**: Haoyang Feng, Yanjun Dai, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04665)  

**Abstract**: Although large language models have demonstrated the potential for personalized advertising recommendations in experimental environments, in actual operations, how advertising recommendation systems can be combined with measures such as user privacy protection and data security is still an area worthy of in-depth discussion. To this end, this paper studies the personalized risks and regulatory strategies of large language models in digital advertising. This study first outlines the principles of Large Language Model (LLM), especially the self-attention mechanism based on the Transformer architecture, and how to enable the model to understand and generate natural language text. Then, the BERT (Bidirectional Encoder Representations from Transformers) model and the attention mechanism are combined to construct an algorithmic model for personalized advertising recommendations and user factor risk protection. The specific steps include: data collection and preprocessing, feature selection and construction, using large language models such as BERT for advertising semantic embedding, and ad recommendations based on user portraits. Then, local model training and data encryption are used to ensure the security of user privacy and avoid the leakage of personal data. This paper designs an experiment for personalized advertising recommendation based on a large language model of BERT and verifies it with real user data. The experimental results show that BERT-based advertising push can effectively improve the click-through rate and conversion rate of advertisements. At the same time, through local model training and privacy protection mechanisms, the risk of user privacy leakage can be reduced to a certain extent. 

---
# Advancing 3D Medical Image Segmentation: Unleashing the Potential of Planarian Neural Networks in Artificial Intelligence 

**Authors**: Ziyuan Huang, Kevin Huggins, Srikar Bellur  

**Link**: [PDF](https://arxiv.org/pdf/2505.04664)  

**Abstract**: Our study presents PNN-UNet as a method for constructing deep neural networks that replicate the planarian neural network (PNN) structure in the context of 3D medical image data. Planarians typically have a cerebral structure comprising two neural cords, where the cerebrum acts as a coordinator, and the neural cords serve slightly different purposes within the organism's neurological system. Accordingly, PNN-UNet comprises a Deep-UNet and a Wide-UNet as the nerve cords, with a densely connected autoencoder performing the role of the brain. This distinct architecture offers advantages over both monolithic (UNet) and modular networks (Ensemble-UNet). Our outcomes on a 3D MRI hippocampus dataset, with and without data augmentation, demonstrate that PNN-UNet outperforms the baseline UNet and several other UNet variants in image segmentation. 

---
# Advancing Conversational Diagnostic AI with Multimodal Reasoning 

**Authors**: Khaled Saab, Jan Freyberg, Chunjong Park, Tim Strother, Yong Cheng, Wei-Hung Weng, David G.T. Barrett, David Stutz, Nenad Tomasev, Anil Palepu, Valentin Liévin, Yash Sharma, Roma Ruparel, Abdullah Ahmed, Elahe Vedadi, Kimberly Kanada, Cian Hughes, Yun Liu, Geoff Brown, Yang Gao, Sean Li, S. Sara Mahdavi, James Manyika, Katherine Chou, Yossi Matias, Avinatan Hassidim, Dale R. Webster, Pushmeet Kohli, S.M. Ali Eslami, Joëlle Barral, Adam Rodman, Vivek Natarajan, Mike Schaekermann, Tao Tu, Alan Karthikesalingam, Ryutaro Tanno  

**Link**: [PDF](https://arxiv.org/pdf/2505.04653)  

**Abstract**: Large Language Models (LLMs) have demonstrated great potential for conducting diagnostic conversations but evaluation has been largely limited to language-only interactions, deviating from the real-world requirements of remote care delivery. Instant messaging platforms permit clinicians and patients to upload and discuss multimodal medical artifacts seamlessly in medical consultation, but the ability of LLMs to reason over such data while preserving other attributes of competent diagnostic conversation remains unknown. Here we advance the conversational diagnosis and management performance of the Articulate Medical Intelligence Explorer (AMIE) through a new capability to gather and interpret multimodal data, and reason about this precisely during consultations. Leveraging Gemini 2.0 Flash, our system implements a state-aware dialogue framework, where conversation flow is dynamically controlled by intermediate model outputs reflecting patient states and evolving diagnoses. Follow-up questions are strategically directed by uncertainty in such patient states, leading to a more structured multimodal history-taking process that emulates experienced clinicians. We compared AMIE to primary care physicians (PCPs) in a randomized, blinded, OSCE-style study of chat-based consultations with patient actors. We constructed 105 evaluation scenarios using artifacts like smartphone skin photos, ECGs, and PDFs of clinical documents across diverse conditions and demographics. Our rubric assessed multimodal capabilities and other clinically meaningful axes like history-taking, diagnostic accuracy, management reasoning, communication, and empathy. Specialist evaluation showed AMIE to be superior to PCPs on 7/9 multimodal and 29/32 non-multimodal axes (including diagnostic accuracy). The results show clear progress in multimodal conversational diagnostic AI, but real-world translation needs further research. 

---
# Multimodal Benchmarking and Recommendation of Text-to-Image Generation Models 

**Authors**: Kapil Wanaskar, Gaytri Jena, Magdalini Eirinaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04650)  

**Abstract**: This work presents an open-source unified benchmarking and evaluation framework for text-to-image generation models, with a particular focus on the impact of metadata augmented prompts. Leveraging the DeepFashion-MultiModal dataset, we assess generated outputs through a comprehensive set of quantitative metrics, including Weighted Score, CLIP (Contrastive Language Image Pre-training)-based similarity, LPIPS (Learned Perceptual Image Patch Similarity), FID (Frechet Inception Distance), and retrieval-based measures, as well as qualitative analysis. Our results demonstrate that structured metadata enrichments greatly enhance visual realism, semantic fidelity, and model robustness across diverse text-to-image architectures. While not a traditional recommender system, our framework enables task-specific recommendations for model selection and prompt design based on evaluation metrics. 

---
# Rethinking Multimodal Sentiment Analysis: A High-Accuracy, Simplified Fusion Architecture 

**Authors**: Nischal Mandal, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04642)  

**Abstract**: Multimodal sentiment analysis, a pivotal task in affective computing, seeks to understand human emotions by integrating cues from language, audio, and visual signals. While many recent approaches leverage complex attention mechanisms and hierarchical architectures, we propose a lightweight, yet effective fusion-based deep learning model tailored for utterance-level emotion classification. Using the benchmark IEMOCAP dataset, which includes aligned text, audio-derived numeric features, and visual descriptors, we design a modality-specific encoder using fully connected layers followed by dropout regularization. The modality-specific representations are then fused using simple concatenation and passed through a dense fusion layer to capture cross-modal interactions. This streamlined architecture avoids computational overhead while preserving performance, achieving a classification accuracy of 92% across six emotion categories. Our approach demonstrates that with careful feature engineering and modular design, simpler fusion strategies can outperform or match more complex models, particularly in resource-constrained environments. 

---
# Language translation, and change of accent for speech-to-speech task using diffusion model 

**Authors**: Abhishek Mishra, Ritesh Sur Chowdhury, Vartul Bahuguna, Isha Pandey, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04639)  

**Abstract**: Speech-to-speech translation (S2ST) aims to convert spoken input in one language to spoken output in another, typically focusing on either language translation or accent adaptation. However, effective cross-cultural communication requires handling both aspects simultaneously - translating content while adapting the speaker's accent to match the target language context. In this work, we propose a unified approach for simultaneous speech translation and change of accent, a task that remains underexplored in current literature. Our method reformulates the problem as a conditional generation task, where target speech is generated based on phonemes and guided by target speech features. Leveraging the power of diffusion models, known for high-fidelity generative capabilities, we adapt text-to-image diffusion strategies by conditioning on source speech transcriptions and generating Mel spectrograms representing the target speech with desired linguistic and accentual attributes. This integrated framework enables joint optimization of translation and accent adaptation, offering a more parameter-efficient and effective model compared to traditional pipelines. 

---
# Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs 

**Authors**: Dongxing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04637)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have demonstrated remarkable capabilities in processing diverse data types, yet significant disparities persist between human cognitive processes and computational approaches to multimodal information integration. This research presents a systematic investigation into the parallels between human cross-modal chunking mechanisms and token representation methodologies in MLLMs. Through empirical studies comparing human performance patterns with model behaviors across visual-linguistic tasks, we demonstrate that conventional static tokenization schemes fundamentally constrain current models' capacity to simulate the dynamic, context-sensitive nature of human information processing. We propose a novel framework for dynamic cross-modal tokenization that incorporates adaptive boundaries, hierarchical representations, and alignment mechanisms grounded in cognitive science principles. Quantitative evaluations demonstrate that our approach yields statistically significant improvements over state-of-the-art models on benchmark tasks (+7.8% on Visual Question Answering, +5.3% on Complex Scene Description) while exhibiting more human-aligned error patterns and attention distributions. These findings contribute to the theoretical understanding of the relationship between human cognition and artificial intelligence, while providing empirical evidence for developing more cognitively plausible AI systems. 

---
# From Dialect Gaps to Identity Maps: Tackling Variability in Speaker Verification 

**Authors**: Abdulhady Abas Abdullah, Soran Badawi, Dana A. Abdullah, Dana Rasul Hamad, Hanan Abdulrahman Taher, Sabat Salih Muhamad, Aram Mahmood Ahmed, Bryar A. Hassan, Sirwan Abdolwahed Aula, Tarik A. Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2505.04629)  

**Abstract**: The complexity and difficulties of Kurdish speaker detection among its several dialects are investigated in this work. Because of its great phonetic and lexical differences, Kurdish with several dialects including Kurmanji, Sorani, and Hawrami offers special challenges for speaker recognition systems. The main difficulties in building a strong speaker identification system capable of precisely identifying speakers across several dialects are investigated in this work. To raise the accuracy and dependability of these systems, it also suggests solutions like sophisticated machine learning approaches, data augmentation tactics, and the building of thorough dialect-specific corpus. The results show that customized strategies for every dialect together with cross-dialect training greatly enhance recognition performance. 

---
# How Social is It? A Benchmark for LLMs' Capabilities in Multi-user Multi-turn Social Agent Tasks 

**Authors**: Yusen Wu, Junwu Xiong, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04628)  

**Abstract**: Expanding the application of large language models (LLMs) to societal life, instead of primary function only as auxiliary assistants to communicate with only one person at a time, necessitates LLMs' capabilities to independently play roles in multi-user, multi-turn social agent tasks within complex social settings. However, currently the capability has not been systematically measured with available benchmarks. To address this gap, we first introduce an agent task leveling framework grounded in sociological principles. Concurrently, we propose a novel benchmark, How Social Is It (we call it HSII below), designed to assess LLM's social capabilities in comprehensive social agents tasks and benchmark representative models. HSII comprises four stages: format parsing, target selection, target switching conversation, and stable conversation, which collectively evaluate the communication and task completion capabilities of LLMs within realistic social interaction scenarios dataset, HSII-Dataset. The dataset is derived step by step from news dataset. We perform an ablation study by doing clustering to the dataset. Additionally, we investigate the impact of chain of thought (COT) method on enhancing LLMs' social performance. Since COT cost more computation, we further introduce a new statistical metric, COT-complexity, to quantify the efficiency of certain LLMs with COTs for specific social tasks and strike a better trade-off between measurement of correctness and efficiency. Various results of our experiments demonstrate that our benchmark is well-suited for evaluating social skills in LLMs. 

---
# Toward Holistic Evaluation of Recommender Systems Powered by Generative Models 

**Authors**: Yashar Deldjoo, Nikhil Mehta, Maheswaran Sathiamoorthy, Shuai Zhang, Pablo Castells, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2504.06667)  

**Abstract**: Recommender systems powered by generative models (Gen-RecSys) extend beyond classical item ranking by producing open-ended content, which simultaneously unlocks richer user experiences and introduces new risks. On one hand, these systems can enhance personalization and appeal through dynamic explanations and multi-turn dialogues. On the other hand, they might venture into unknown territory-hallucinating nonexistent items, amplifying bias, or leaking private information. Traditional accuracy metrics cannot fully capture these challenges, as they fail to measure factual correctness, content safety, or alignment with user intent.
This paper makes two main contributions. First, we categorize the evaluation challenges of Gen-RecSys into two groups: (i) existing concerns that are exacerbated by generative outputs (e.g., bias, privacy) and (ii) entirely new risks (e.g., item hallucinations, contradictory explanations). Second, we propose a holistic evaluation approach that includes scenario-based assessments and multi-metric checks-incorporating relevance, factual grounding, bias detection, and policy compliance. Our goal is to provide a guiding framework so researchers and practitioners can thoroughly assess Gen-RecSys, ensuring effective personalization and responsible deployment. 

---
