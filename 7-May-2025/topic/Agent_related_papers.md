# OSUniverse: Benchmark for Multimodal GUI-navigation AI Agents 

**Authors**: Mariya Davydova, Daniel Jeffries, Patrick Barker, Arturo Márquez Flores, Sinéad Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03570)  

**Abstract**: In this paper, we introduce OSUniverse: a benchmark of complex, multimodal desktop-oriented tasks for advanced GUI-navigation AI agents that focuses on ease of use, extensibility, comprehensive coverage of test cases, and automated validation. We divide the tasks in increasing levels of complexity, from basic precision clicking to multistep, multiapplication tests requiring dexterity, precision, and clear thinking from the agent. In version one of the benchmark, presented here, we have calibrated the complexity of the benchmark test cases to ensure that the SOTA (State of the Art) agents (at the time of publication) do not achieve results higher than 50%, while the average white collar worker can perform all these tasks with perfect accuracy. The benchmark can be scored manually, but we also introduce an automated validation mechanism that has an average error rate less than 2%. Therefore, this benchmark presents solid ground for fully automated measuring of progress, capabilities and the effectiveness of GUI-navigation AI agents over the short and medium-term horizon. The source code of the benchmark is available at this https URL. 

---
# Procedural Memory Is Not All You Need: Bridging Cognitive Gaps in LLM-Based Agents 

**Authors**: Schaun Wheeler, Olivier Jeunen  

**Link**: [PDF](https://arxiv.org/pdf/2505.03434)  

**Abstract**: Large Language Models (LLMs) represent a landmark achievement in Artificial Intelligence (AI), demonstrating unprecedented proficiency in procedural tasks such as text generation, code completion, and conversational coherence. These capabilities stem from their architecture, which mirrors human procedural memory -- the brain's ability to automate repetitive, pattern-driven tasks through practice. However, as LLMs are increasingly deployed in real-world applications, it becomes impossible to ignore their limitations operating in complex, unpredictable environments. This paper argues that LLMs, while transformative, are fundamentally constrained by their reliance on procedural memory. To create agents capable of navigating ``wicked'' learning environments -- where rules shift, feedback is ambiguous, and novelty is the norm -- we must augment LLMs with semantic memory and associative learning systems. By adopting a modular architecture that decouples these cognitive functions, we can bridge the gap between narrow procedural expertise and the adaptive intelligence required for real-world problem-solving. 

---
# RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation 

**Authors**: Tiantian Gan, Qiyao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.03275)  

**Abstract**: Large language models (LLMs) struggle to effectively utilize a growing number of external tools, such as those defined by the Model Context Protocol (MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to identify the most relevant MCP(s) for a given query from an external index before engaging the LLM. Only the selected tool descriptions are passed to the model, drastically reducing prompt size and simplifying decision-making. Experiments, including an MCP stress test, demonstrate RAG-MCP significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables scalable and accurate tool integration for LLMs. 

---
# Rainbow Delay Compensation: A Multi-Agent Reinforcement Learning Framework for Mitigating Delayed Observation 

**Authors**: Songchen Fu, Siang Chen, Shaojing Zhao, Letian Bai, Ta Li, Yonghong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03586)  

**Abstract**: In real-world multi-agent systems (MASs), observation delays are ubiquitous, preventing agents from making decisions based on the environment's true state. An individual agent's local observation often consists of multiple components from other agents or dynamic entities in the environment. These discrete observation components with varying delay characteristics pose significant challenges for multi-agent reinforcement learning (MARL). In this paper, we first formulate the decentralized stochastic individual delay partially observable Markov decision process (DSID-POMDP) by extending the standard Dec-POMDP. We then propose the Rainbow Delay Compensation (RDC), a MARL training framework for addressing stochastic individual delays, along with recommended implementations for its constituent modules. We implement the DSID-POMDP's observation generation pattern using standard MARL benchmarks, including MPE and SMAC. Experiments demonstrate that baseline MARL methods suffer severe performance degradation under fixed and unfixed delays. The RDC-enhanced approach mitigates this issue, remarkably achieving ideal delay-free performance in certain delay scenarios while maintaining generalization capability. Our work provides a novel perspective on multi-agent delayed observation problems and offers an effective solution framework. 

---
# LlamaFirewall: An open source guardrail system for building secure AI agents 

**Authors**: Sahana Chennabasappa, Cyrus Nikolaidis, Daniel Song, David Molnar, Stephanie Ding, Shengye Wan, Spencer Whitman, Lauren Deason, Nicholas Doucette, Abraham Montilla, Alekhya Gampa, Beto de Paola, Dominik Gabi, James Crnkovich, Jean-Christophe Testud, Kat He, Rashnil Chaturvedi, Wu Zhou, Joshua Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2505.03574)  

**Abstract**: Large language models (LLMs) have evolved from simple chatbots into autonomous agents capable of performing complex tasks such as editing production code, orchestrating workflows, and taking higher-stakes actions based on untrusted inputs like webpages and emails. These capabilities introduce new security risks that existing security measures, such as model fine-tuning or chatbot-focused guardrails, do not fully address. Given the higher stakes and the absence of deterministic solutions to mitigate these risks, there is a critical need for a real-time guardrail monitor to serve as a final layer of defense, and support system level, use case specific safety policy definition and enforcement. We introduce LlamaFirewall, an open-source security focused guardrail framework designed to serve as a final layer of defense against security risks associated with AI Agents. Our framework mitigates risks such as prompt injection, agent misalignment, and insecure code risks through three powerful guardrails: PromptGuard 2, a universal jailbreak detector that demonstrates clear state of the art performance; Agent Alignment Checks, a chain-of-thought auditor that inspects agent reasoning for prompt injection and goal misalignment, which, while still experimental, shows stronger efficacy at preventing indirect injections in general scenarios than previously proposed approaches; and CodeShield, an online static analysis engine that is both fast and extensible, aimed at preventing the generation of insecure or dangerous code by coding agents. Additionally, we include easy-to-use customizable scanners that make it possible for any developer who can write a regular expression or an LLM prompt to quickly update an agent's security guardrails. 

---
# Actor-Critics Can Achieve Optimal Sample Efficiency 

**Authors**: Kevin Tan, Wei Fan, Yuting Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.03710)  

**Abstract**: Actor-critic algorithms have become a cornerstone in reinforcement learning (RL), leveraging the strengths of both policy-based and value-based methods. Despite recent progress in understanding their statistical efficiency, no existing work has successfully learned an $\epsilon$-optimal policy with a sample complexity of $O(1/\epsilon^2)$ trajectories with general function approximation when strategic exploration is necessary.
We address this open problem by introducing a novel actor-critic algorithm that attains a sample-complexity of $O(dH^5 \log|\mathcal{A}|/\epsilon^2 + d H^4 \log|\mathcal{F}|/ \epsilon^2)$ trajectories, and accompanying $\sqrt{T}$ regret when the Bellman eluder dimension $d$ does not increase with $T$ at more than a $\log T$ rate.
Here, $\mathcal{F}$ is the critic function class, $\mathcal{A}$ is the action space, and $H$ is the horizon in the finite horizon MDP setting. Our algorithm integrates optimism, off-policy critic estimation targeting the optimal Q-function, and rare-switching policy resets.
We extend this to the setting of Hybrid RL, showing that initializing the critic with offline data yields sample efficiency gains compared to purely offline or online RL. Further, utilizing access to offline data, we provide a \textit{non-optimistic} provably efficient actor-critic algorithm that only additionally requires $N_{\text{off}} \geq c_{\text{off}}^*dH^4/\epsilon^2$ in exchange for omitting optimism, where $c_{\text{off}}^*$ is the single-policy concentrability coefficient and $N_{\text{off}}$ is the number of offline samples. This addresses another open problem in the literature. We further provide numerical experiments to support our theoretical findings. 

---
# FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios 

**Authors**: Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03730)  

**Abstract**: Action customization involves generating videos where the subject performs actions dictated by input control signals. Current methods use pose-guided or global motion customization but are limited by strict constraints on spatial structure, such as layout, skeleton, and viewpoint consistency, reducing adaptability across diverse subjects and scenarios. To overcome these limitations, we propose FlexiAct, which transfers actions from a reference video to an arbitrary target image. Unlike existing methods, FlexiAct allows for variations in layout, viewpoint, and skeletal structure between the subject of the reference video and the target image, while maintaining identity consistency. Achieving this requires precise action control, spatial structure adaptation, and consistency preservation. To this end, we introduce RefAdapter, a lightweight image-conditioned adapter that excels in spatial adaptation and consistency preservation, surpassing existing methods in balancing appearance consistency and structural flexibility. Additionally, based on our observations, the denoising process exhibits varying levels of attention to motion (low frequency) and appearance details (high frequency) at different timesteps. So we propose FAE (Frequency-aware Action Extraction), which, unlike existing methods that rely on separate spatial-temporal architectures, directly achieves action extraction during the denoising process. Experiments demonstrate that our method effectively transfers actions to subjects with diverse layouts, skeletons, and viewpoints. We release our code and model weights to support further research at this https URL 

---
# Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation 

**Authors**: Mohammad Shoaib Ansari, Mohd Sohail Ali Khan, Shubham Revankar, Aditya Varma, Anil S. Mokhade  

**Link**: [PDF](https://arxiv.org/pdf/2505.03406)  

**Abstract**: This research paper investigates the application of Large Language Models (LLMs) in healthcare, specifically focusing on enhancing medical decision support through Retrieval-Augmented Generation (RAG) integrated with hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation (QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By embedding and retrieving context-relevant healthcare information, the system significantly improves response accuracy. QLoRA facilitates notable parameter efficiency and memory optimization, preserving the integrity of medical information through specialized quantization techniques. Our research also shows that our model performs relatively well on various medical benchmarks, indicating that it can be used to make basic medical suggestions. This paper details the system's technical components, including its architecture, quantization methods, and key healthcare applications such as enhanced disease prediction from patient symptoms and medical history, treatment suggestions, and efficient summarization of complex medical reports. We touch on the ethical considerations-patient privacy, data security, and the need for rigorous clinical validation-as well as the practical challenges of integrating such systems into real-world healthcare workflows. Furthermore, the lightweight quantized weights ensure scalability and ease of deployment even in low-resource hospital environments. Finally, the paper concludes with an analysis of the broader impact of LLMs on healthcare and outlines future directions for LLMs in medical settings. 

---
# Absolute Zero: Reinforced Self-play Reasoning with Zero Data 

**Authors**: Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03335)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning capabilities of large language models by learning directly from outcome-based rewards. Recent RLVR works that operate under the zero setting avoid supervision in labeling the reasoning process, but still depend on manually curated collections of questions and answers for training. The scarcity of high-quality, human-produced examples raises concerns about the long-term scalability of relying on human supervision, a challenge already evident in the domain of language model pretraining. Furthermore, in a hypothetical future where AI surpasses human intelligence, tasks provided by humans may offer limited learning potential for a superintelligent system. To address these concerns, we propose a new RLVR paradigm called Absolute Zero, in which a single model learns to propose tasks that maximize its own learning progress and improves reasoning by solving them, without relying on any external data. Under this paradigm, we introduce the Absolute Zero Reasoner (AZR), a system that self-evolves its training curriculum and reasoning ability by using a code executor to both validate proposed code reasoning tasks and verify answers, serving as an unified source of verifiable reward to guide open-ended yet grounded learning. Despite being trained entirely without external data, AZR achieves overall SOTA performance on coding and mathematical reasoning tasks, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples. Furthermore, we demonstrate that AZR can be effectively applied across different model scales and is compatible with various model classes. 

---
# The Unreasonable Effectiveness of Discrete-Time Gaussian Process Mixtures for Robot Policy Learning 

**Authors**: Jan Ole von Hartz, Adrian Röfer, Joschka Boedecker, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.03296)  

**Abstract**: We present Mixture of Discrete-time Gaussian Processes (MiDiGap), a novel approach for flexible policy representation and imitation learning in robot manipulation. MiDiGap enables learning from as few as five demonstrations using only camera observations and generalizes across a wide range of challenging tasks. It excels at long-horizon behaviors such as making coffee, highly constrained motions such as opening doors, dynamic actions such as scooping with a spatula, and multimodal tasks such as hanging a mug. MiDiGap learns these tasks on a CPU in less than a minute and scales linearly to large datasets. We also develop a rich suite of tools for inference-time steering using evidence such as collision signals and robot kinematic constraints. This steering enables novel generalization capabilities, including obstacle avoidance and cross-embodiment policy transfer. MiDiGap achieves state-of-the-art performance on diverse few-shot manipulation benchmarks. On constrained RLBench tasks, it improves policy success by 76 percentage points and reduces trajectory cost by 67%. On multimodal tasks, it improves policy success by 48 percentage points and increases sample efficiency by a factor of 20. In cross-embodiment transfer, it more than doubles policy success. We make the code publicly available at this https URL. 

---
# Assessing and Enhancing the Robustness of LLM-based Multi-Agent Systems Through Chaos Engineering 

**Authors**: Joshua Owotogbe  

**Link**: [PDF](https://arxiv.org/pdf/2505.03096)  

**Abstract**: This study explores the application of chaos engineering to enhance the robustness of Large Language Model-Based Multi-Agent Systems (LLM-MAS) in production-like environments under real-world conditions. LLM-MAS can potentially improve a wide range of tasks, from answering questions and generating content to automating customer support and improving decision-making processes. However, LLM-MAS in production or preproduction environments can be vulnerable to emergent errors or disruptions, such as hallucinations, agent failures, and agent communication failures. This study proposes a chaos engineering framework to proactively identify such vulnerabilities in LLM-MAS, assess and build resilience against them, and ensure reliable performance in critical applications. 

---
# Neural Orchestration for Multi-Agent Systems: A Deep Learning Framework for Optimal Agent Selection in Multi-Domain Task Environments 

**Authors**: Kushagra Agrawal, Nisharg Nargund  

**Link**: [PDF](https://arxiv.org/pdf/2505.02861)  

**Abstract**: Multi-agent systems (MAS) are foundational in simulating complex real-world scenarios involving autonomous, interacting entities. However, traditional MAS architectures often suffer from rigid coordination mechanisms and difficulty adapting to dynamic tasks. We propose MetaOrch, a neural orchestration framework for optimal agent selection in multi-domain task environments. Our system implements a supervised learning approach that models task context, agent histories, and expected response quality to select the most appropriate agent for each task. A novel fuzzy evaluation module scores agent responses along completeness, relevance, and confidence dimensions, generating soft supervision labels for training the orchestrator. Unlike previous methods that hard-code agent-task mappings, MetaOrch dynamically predicts the most suitable agent while estimating selection confidence. Experiments in simulated environments with heterogeneous agents demonstrate that our approach achieves 86.3% selection accuracy, significantly outperforming baseline strategies including random selection and round-robin scheduling. The modular architecture emphasizes extensibility, allowing agents to be registered, updated, and queried independently. Results suggest that neural orchestration offers a powerful approach to enhancing the autonomy, interpretability, and adaptability of multi-agent systems across diverse task domains. 

---
# Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in Large Language Models 

**Authors**: Bang Zhang, Ruotian Ma, Qingxuan Jiang, Peisong Wang, Jiaqi Chen, Zheng Xie, Xingyu Chen, Yue Wang, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02847)  

**Abstract**: Assessing how well a large language model (LLM) understands human, rather than merely text, remains an open challenge. To bridge the gap, we introduce Sentient Agent as a Judge (SAGE), an automated evaluation framework that measures an LLM's higher-order social cognition. SAGE instantiates a Sentient Agent that simulates human-like emotional changes and inner thoughts during interaction, providing a more realistic evaluation of the tested model in multi-turn conversations. At every turn, the agent reasons about (i) how its emotion changes, (ii) how it feels, and (iii) how it should reply, yielding a numerical emotion trajectory and interpretable inner thoughts. Experiments on 100 supportive-dialogue scenarios show that the final Sentient emotion score correlates strongly with Barrett-Lennard Relationship Inventory (BLRI) ratings and utterance-level empathy metrics, validating psychological fidelity. We also build a public Sentient Leaderboard covering 18 commercial and open-source models that uncovers substantial gaps (up to 4x) between frontier systems (GPT-4o-Latest, Gemini2.5-Pro) and earlier baselines, gaps not reflected in conventional leaderboards (e.g., Arena). SAGE thus provides a principled, scalable and interpretable tool for tracking progress toward genuinely empathetic and socially adept language agents. 

---
# Towards conversational assistants for health applications: using ChatGPT to generate conversations about heart failure 

**Authors**: Anuja Tayal, Devika Salunke, Barbara Di Eugenio, Paula G Allen-Meares, Eulalia P Abril, Olga Garcia-Bedoya, Carolyn A Dickens, Andrew D. Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2505.03675)  

**Abstract**: We explore the potential of ChatGPT (3.5-turbo and 4) to generate conversations focused on self-care strategies for African-American heart failure patients -- a domain with limited specialized datasets. To simulate patient-health educator dialogues, we employed four prompting strategies: domain, African American Vernacular English (AAVE), Social Determinants of Health (SDOH), and SDOH-informed reasoning. Conversations were generated across key self-care domains of food, exercise, and fluid intake, with varying turn lengths (5, 10, 15) and incorporated patient-specific SDOH attributes such as age, gender, neighborhood, and socioeconomic status. Our findings show that effective prompt design is essential. While incorporating SDOH and reasoning improves dialogue quality, ChatGPT still lacks the empathy and engagement needed for meaningful healthcare communication. 

---
# WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch 

**Authors**: Zimu Lu, Yunqiao Yang, Houxing Ren, Haotian Hou, Han Xiao, Ke Wang, Weikang Shi, Aojun Zhou, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03733)  

**Abstract**: LLM-based agents have demonstrated great potential in generating and managing code within complex codebases. In this paper, we introduce WebGen-Bench, a novel benchmark designed to measure an LLM-based agent's ability to create multi-file website codebases from scratch. It contains diverse instructions for website generation, created through the combined efforts of human annotators and GPT-4o. These instructions span three major categories and thirteen minor categories, encompassing nearly all important types of web applications. To assess the quality of the generated websites, we use GPT-4o to generate test cases targeting each functionality described in the instructions, and then manually filter, adjust, and organize them to ensure accuracy, resulting in 647 test cases. Each test case specifies an operation to be performed on the website and the expected result after the operation. To automate testing and improve reproducibility, we employ a powerful web-navigation agent to execute tests on the generated websites and determine whether the observed responses align with the expected results. We evaluate three high-performance code-agent frameworks, this http URL, OpenHands, and Aider, using multiple proprietary and open-source LLMs as engines. The best-performing combination, this http URL powered by DeepSeek-R1, achieves only 27.8\% accuracy on the test cases, highlighting the challenging nature of our benchmark. Additionally, we construct WebGen-Instruct, a training set consisting of 6,667 website-generation instructions. Training Qwen2.5-Coder-32B-Instruct on this http URL trajectories generated from a subset of this training set achieves an accuracy of 38.2\%, surpassing the performance of the best proprietary model. 

---
