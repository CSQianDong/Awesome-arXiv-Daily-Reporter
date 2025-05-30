# BookWorld: From Novels to Interactive Agent Societies for Creative Story Generation 

**Authors**: Yiting Ran, Xintao Wang, Tian Qiu, Jiaqing Liang, Yanghua Xiao, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14538)  

**Abstract**: Recent advances in large language models (LLMs) have enabled social simulation through multi-agent systems. Prior efforts focus on agent societies created from scratch, assigning agents with newly defined personas. However, simulating established fictional worlds and characters remain largely underexplored, despite its significant practical value. In this paper, we introduce BookWorld, a comprehensive system for constructing and simulating book-based multi-agent societies. BookWorld's design covers comprehensive real-world intricacies, including diverse and dynamic characters, fictional worldviews, geographical constraints and changes, e.t.c. BookWorld enables diverse applications including story generation, interactive games and social simulation, offering novel ways to extend and explore beloved fictional works. Through extensive experiments, we demonstrate that BookWorld generates creative, high-quality stories while maintaining fidelity to the source books, surpassing previous methods with a win rate of 75.36%. The code of this paper can be found at the project page: this https URL. 

---
# DialogueAgents: A Hybrid Agent-Based Speech Synthesis Framework for Multi-Party Dialogue 

**Authors**: Xiang Li, Duyi Pan, Hongru Xiao, Jiale Han, Jing Tang, Jiabao Ma, Wei Wang, Bo Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14482)  

**Abstract**: Speech synthesis is crucial for human-computer interaction, enabling natural and intuitive communication. However, existing datasets involve high construction costs due to manual annotation and suffer from limited character diversity, contextual scenarios, and emotional expressiveness. To address these issues, we propose DialogueAgents, a novel hybrid agent-based speech synthesis framework, which integrates three specialized agents -- a script writer, a speech synthesizer, and a dialogue critic -- to collaboratively generate dialogues. Grounded in a diverse character pool, the framework iteratively refines dialogue scripts and synthesizes speech based on speech review, boosting emotional expressiveness and paralinguistic features of the synthesized dialogues. Using DialogueAgent, we contribute MultiTalk, a bilingual, multi-party, multi-turn speech dialogue dataset covering diverse topics. Extensive experiments demonstrate the effectiveness of our framework and the high quality of the MultiTalk dataset. We release the dataset and code this https URL to facilitate future research on advanced speech synthesis models and customized data generation. 

---
# Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs 

**Authors**: Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Rouyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.15280)  

**Abstract**: Multi-view understanding, the ability to reconcile visual information across diverse viewpoints for effective navigation, manipulation, and 3D scene comprehension, is a fundamental challenge in Multi-Modal Large Language Models (MLLMs) to be used as embodied agents. While recent MLLMs have shown impressive advances in high-level reasoning and planning, they frequently fall short when confronted with multi-view geometric consistency and cross-view correspondence. To comprehensively evaluate the challenges of MLLMs in multi-view scene reasoning, we propose All-Angles Bench, a benchmark of over 2,100 human carefully annotated multi-view question-answer pairs across 90 diverse real-world scenes. Our six tasks (counting, attribute identification, relative distance, relative direction, object manipulation, and camera pose estimation) specifically test model's geometric correspondence and the capacity to align information consistently across views. Our extensive experiments, benchmark on 27 representative MLLMs including Gemini-2.0-Flash, Claude-3.7-Sonnet, and GPT-4o against human evaluators reveals a substantial performance gap, indicating that current MLLMs remain far from human-level proficiency. Through in-depth analysis, we show that MLLMs are particularly underperforming under two aspects: (1) cross-view correspondence for partially occluded views and (2) establishing the coarse camera poses. These findings highlight the necessity of domain-specific refinements or modules that embed stronger multi-view awareness. We believe that our All-Angles Bench offers valuable insights and contribute to bridging the gap between MLLMs and human-level multi-view understanding. The project and benchmark are publicly available at this https URL. 

---
# EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework 

**Authors**: Yao Shi, Rongkeng Liang, Yong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14928)  

**Abstract**: Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness. 

---
# Meta-Thinking in LLMs via Multi-Agent Reinforcement Learning: A Survey 

**Authors**: Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Muhammad Awais Khan Bangash, Muhammad Ali Jamshed  

**Link**: [PDF](https://arxiv.org/pdf/2504.14520)  

**Abstract**: This survey explores the development of meta-thinking capabilities in Large Language Models (LLMs) from a Multi-Agent Reinforcement Learning (MARL) perspective. Meta-thinking self-reflection, assessment, and control of thinking processes is an important next step in enhancing LLM reliability, flexibility, and performance, particularly for complex or high-stakes tasks. The survey begins by analyzing current LLM limitations, such as hallucinations and the lack of internal self-assessment mechanisms. It then talks about newer methods, including RL from human feedback (RLHF), self-distillation, and chain-of-thought prompting, and each of their limitations. The crux of the survey is to talk about how multi-agent architectures, namely supervisor-agent hierarchies, agent debates, and theory of mind frameworks, can emulate human-like introspective behavior and enhance LLM robustness. By exploring reward mechanisms, self-play, and continuous learning methods in MARL, this survey gives a comprehensive roadmap to building introspective, adaptive, and trustworthy LLMs. Evaluation metrics, datasets, and future research avenues, including neuroscience-inspired architectures and hybrid symbolic reasoning, are also discussed. 

---
# PLANET: A Collection of Benchmarks for Evaluating LLMs' Planning Capabilities 

**Authors**: Haoming Li, Zhaoliang Chen, Jonathan Zhang, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14773)  

**Abstract**: Planning is central to agents and agentic AI. The ability to plan, e.g., creating travel itineraries within a budget, holds immense potential in both scientific and commercial contexts. Moreover, optimal plans tend to require fewer resources compared to ad-hoc methods. To date, a comprehensive understanding of existing planning benchmarks appears to be lacking. Without it, comparing planning algorithms' performance across domains or selecting suitable algorithms for new scenarios remains challenging. In this paper, we examine a range of planning benchmarks to identify commonly used testbeds for algorithm development and highlight potential gaps. These benchmarks are categorized into embodied environments, web navigation, scheduling, games and puzzles, and everyday task automation. Our study recommends the most appropriate benchmarks for various algorithms and offers insights to guide future benchmark development. 

---
# InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners 

**Authors**: Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14239)  

**Abstract**: Multimodal Large Language Models (MLLMs) have powered Graphical User Interface (GUI) Agents, showing promise in automating tasks on computing devices. Recent works have begun exploring reasoning in GUI tasks with encouraging results. However, many current approaches rely on manually designed reasoning templates, which may result in reasoning that is not sufficiently robust and adaptive for complex GUI environments. Meanwhile, some existing agents continue to operate as Reactive Actors, relying primarily on implicit reasoning that may lack sufficient depth for GUI tasks demanding planning and error recovery. We argue that advancing these agents requires a shift from reactive acting towards acting based on deliberate reasoning. To facilitate this transformation, we introduce InfiGUI-R1, an MLLM-based GUI agent developed through our Actor2Reasoner framework, a reasoning-centric, two-stage training approach designed to progressively evolve agents from Reactive Actors to Deliberative Reasoners. The first stage, Reasoning Injection, focuses on establishing a basic reasoner. We employ Spatial Reasoning Distillation to transfer cross-modal spatial reasoning capabilities from teacher models to MLLMs through trajectories with explicit reasoning steps, enabling models to integrate GUI visual-spatial information with logical reasoning before action generation. The second stage, Deliberation Enhancement, refines the basic reasoner into a deliberative one using Reinforcement Learning. This stage introduces two approaches: Sub-goal Guidance, which rewards models for generating accurate intermediate sub-goals, and Error Recovery Scenario Construction, which creates failure-and-recovery training scenarios from identified prone-to-error steps. Experimental results show InfiGUI-R1 achieves strong performance in GUI grounding and trajectory tasks. Resources at this https URL. 

---
# TALES: Text Adventure Learning Environment Suite 

**Authors**: Christopher Zhang Cui, Xingdi Yuan, Zhang Xiao, Prithviraj Ammanabrolu, Marc-Alexandre Côté  

**Link**: [PDF](https://arxiv.org/pdf/2504.14128)  

**Abstract**: Reasoning is an essential skill to enable Large Language Models (LLMs) to interact with the world. As tasks become more complex, they demand increasingly sophisticated and diverse reasoning capabilities for sequential decision-making, requiring structured reasoning over the context history to determine the next best action. We introduce TALES, a diverse collection of synthetic and human-written text-adventure games designed to challenge and evaluate diverse reasoning capabilities. We present results over a range of LLMs, open- and closed-weights, performing a qualitative analysis on the top performing models. Despite an impressive showing on synthetic games, even the top LLM-driven agents fail to achieve 15% on games designed for human enjoyment. Code and visualization of the experiments can be found at this https URL. 

---
# System of Agentic AI for the Discovery of Metal-Organic Frameworks 

**Authors**: Theo Jaffrelot Inizan, Sherry Yang, Aaron Kaplan, Yen-hsu Lin, Jian Yin, Saber Mirzaei, Mona Abdelgaid, Ali H. Alawadhi, KwangHwan Cho, Zhiling Zheng, Ekin Dogus Cubuk, Christian Borgs, Jennifer T. Chayes, Kristin A. Persson, Omar M. Yaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14110)  

**Abstract**: Generative models and machine learning promise accelerated material discovery in MOFs for CO2 capture and water harvesting but face significant challenges navigating vast chemical spaces while ensuring synthetizability. Here, we present MOFGen, a system of Agentic AI comprising interconnected agents: a large language model that proposes novel MOF compositions, a diffusion model that generates crystal structures, quantum mechanical agents that optimize and filter candidates, and synthetic-feasibility agents guided by expert rules and machine learning. Trained on all experimentally reported MOFs and computational databases, MOFGen generated hundreds of thousands of novel MOF structures and synthesizable organic linkers. Our methodology was validated through high-throughput experiments and the successful synthesis of five "AI-dreamt" MOFs, representing a major step toward automated synthesizable material discovery. 

---
# 3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark 

**Authors**: Ivan Sviridov, Amina Miftakhova, Artemiy Tereshchenko, Galina Zubkova, Pavel Blinov, Andrey Savchenko  

**Link**: [PDF](https://arxiv.org/pdf/2504.13861)  

**Abstract**: Large Vision-Language Models (LVLMs) are increasingly being explored for applications in telemedicine, yet their ability to engage with diverse patient behaviors remains underexplored. We introduce 3MDBench (Medical Multimodal Multi-agent Dialogue Benchmark), an open-source evaluation framework designed to assess LLM-driven medical consultations. Unlike existing benchmarks, 3MDBench simulates real-world patient variability by incorporating four temperament-driven Patient Agents and an Assessor Agent that evaluates diagnostic accuracy and dialogue quality. The benchmark integrates textual and image-based patient data across 34 common diagnoses, mirroring real-world telemedicine interactions. Under different diagnostic strategies, we evaluate state-of-the-art LVLMs. Our findings demonstrate that incorporating dialogue improves the F1 score from 50.4 to 54.2 compared to non-dialogue settings, underscoring the value of context-driven, information-seeking questioning. Additionally, we demonstrate that multimodal inputs enhance diagnostic efficiency. Image-supported models outperform text-only counterparts by raising the diagnostic F1 score from 52.8 to 54.2 in a similar dialogue setting. Finally, we suggest an approach that improves the diagnostic F1-score to 70.3 by training the CNN model on the diagnosis prediction task and incorporating its top-3 predictions into the LVLM context. 3MDBench provides a reproducible and extendable evaluation framework for AI-driven medical assistants. It offers insights into how patient temperament, dialogue strategies, and multimodal reasoning influence diagnosis quality. By addressing real-world complexities in telemedicine, our benchmark paves the way for more empathetic, reliable, and context-aware AI-driven healthcare solutions. The source code of our benchmark is publicly available: this https URL 

---
# A Survey on (M)LLM-Based GUI Agents 

**Authors**: Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13865)  

**Abstract**: Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation. 

---
# Text-to-Decision Agent: Learning Generalist Policies from Natural Language Supervision 

**Authors**: Shilin Zhang, Zican Hu, Wenhao Wu, Xinyi Xie, Jianxiang Tang, Chunlin Chen, Daoyi Dong, Yu Cheng, Zhenhong Sun, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15046)  

**Abstract**: RL systems usually tackle generalization by inferring task beliefs from high-quality samples or warmup explorations. The restricted form limits their generality and usability since these supervision signals are expensive and even infeasible to acquire in advance for unseen tasks. Learning directly from the raw text about decision tasks is a promising alternative to leverage a much broader source of supervision. In the paper, we propose Text-to-Decision Agent (T2DA), a simple and scalable framework that supervises generalist policy learning with natural language. We first introduce a generalized world model to encode multi-task decision data into a dynamics-aware embedding space. Then, inspired by CLIP, we predict which textual description goes with which decision embedding, effectively bridging their semantic gap via contrastive language-decision pre-training and aligning the text embeddings to comprehend the environment dynamics. After training the text-conditioned generalist policy, the agent can directly realize zero-shot text-to-decision generation in response to language instructions. Comprehensive experiments on MuJoCo and Meta-World benchmarks show that T2DA facilitates high-capacity zero-shot generalization and outperforms various types of baselines. 

---
# A Self-Improving Coding Agent 

**Authors**: Maxime Robeyns, Martin Szummer, Laurence Aitchison  

**Link**: [PDF](https://arxiv.org/pdf/2504.15228)  

**Abstract**: We demonstrate that an LLM coding agent, equipped with basic coding tools, can autonomously edit itself, and thereby improve its performance on benchmark tasks. We find performance gains from 17% to 53% on a random subset of SWE Bench Verified, with additional performance gains on LiveCodeBench, as well as synthetically generated agent benchmarks. Our work represents an advancement in the automated and open-ended design of agentic systems, and provides a reference agent framework for those seeking to post-train LLMs on tool use and other agentic tasks. 

---
# Behavioral Universe Network (BUN): A Behavioral Information-Based Framework for Complex Systems 

**Authors**: Wei Zhou, Ailiya Borjigin, Cong He  

**Link**: [PDF](https://arxiv.org/pdf/2504.15146)  

**Abstract**: Modern digital ecosystems feature complex, dynamic interactions among autonomous entities across diverse domains. Traditional models often separate agents and objects, lacking a unified foundation to capture their interactive behaviors. This paper introduces the Behavioral Universe Network (BUN), a theoretical framework grounded in the Agent-Interaction-Behavior (AIB) formalism. BUN treats subjects (active agents), objects (resources), and behaviors (operations) as first-class entities, all governed by a shared Behavioral Information Base (BIB). We detail the AIB core concepts and demonstrate how BUN leverages information-driven triggers, semantic enrichment, and adaptive rules to coordinate multi-agent systems. We highlight key benefits: enhanced behavior analysis, strong adaptability, and cross-domain interoperability. We conclude by positioning BUN as a promising foundation for next-generation digital governance and intelligent applications. 

---
# FlowReasoner: Reinforcing Query-Level Meta-Agents 

**Authors**: Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15257)  

**Abstract**: This paper proposes a query-level meta-agent named FlowReasoner to automate the design of query-level multi-agent systems, i.e., one system per user query. Our core idea is to incentivize a reasoning-based meta-agent via external execution feedback. Concretely, by distilling DeepSeek R1, we first endow the basic reasoning ability regarding the generation of multi-agent systems to FlowReasoner. Then, we further enhance it via reinforcement learning (RL) with external execution feedback. A multi-purpose reward is designed to guide the RL training from aspects of performance, complexity, and efficiency. In this manner, FlowReasoner is enabled to generate a personalized multi-agent system for each user query via deliberative reasoning. Experiments on both engineering and competition code benchmarks demonstrate the superiority of FlowReasoner. Remarkably, it surpasses o1-mini by 10.52% accuracy across three benchmarks. The code is available at this https URL. 

---
# A Framework for Benchmarking and Aligning Task-Planning Safety in LLM-Based Embodied Agents 

**Authors**: Yuting Huang, Leilei Ding, Zhipeng Tang, Tianfu Wang, Xinrui Lin, Wuyang Zhang, Mingxiao Ma, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14650)  

**Abstract**: Large Language Models (LLMs) exhibit substantial promise in enhancing task-planning capabilities within embodied agents due to their advanced reasoning and comprehension. However, the systemic safety of these agents remains an underexplored frontier. In this study, we present Safe-BeAl, an integrated framework for the measurement (SafePlan-Bench) and alignment (Safe-Align) of LLM-based embodied agents' behaviors. SafePlan-Bench establishes a comprehensive benchmark for evaluating task-planning safety, encompassing 2,027 daily tasks and corresponding environments distributed across 8 distinct hazard categories (e.g., Fire Hazard). Our empirical analysis reveals that even in the absence of adversarial inputs or malicious intent, LLM-based agents can exhibit unsafe behaviors. To mitigate these hazards, we propose Safe-Align, a method designed to integrate physical-world safety knowledge into LLM-based embodied agents while maintaining task-specific performance. Experiments across a variety of settings demonstrate that Safe-BeAl provides comprehensive safety validation, improving safety by 8.55 - 15.22%, compared to embodied agents based on GPT-4, while ensuring successful task completion. 

---
# UFO2: The Desktop AgentOS 

**Authors**: Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14603)  

**Abstract**: Recent Computer-Using Agents (CUAs), powered by multimodal large language models (LLMs), offer a promising direction for automating complex desktop workflows through natural language. However, most existing CUAs remain conceptual prototypes, hindered by shallow OS integration, fragile screenshot-based interaction, and disruptive execution.
We present UFO2, a multiagent AgentOS for Windows desktops that elevates CUAs into practical, system-level automation. UFO2 features a centralized HostAgent for task decomposition and coordination, alongside a collection of application-specialized AppAgent equipped with native APIs, domain-specific knowledge, and a unified GUI--API action layer. This architecture enables robust task execution while preserving modularity and extensibility. A hybrid control detection pipeline fuses Windows UI Automation (UIA) with vision-based parsing to support diverse interface styles. Runtime efficiency is further enhanced through speculative multi-action planning, reducing per-step LLM overhead. Finally, a Picture-in-Picture (PiP) interface enables automation within an isolated virtual desktop, allowing agents and users to operate concurrently without interference.
We evaluate UFO2 across over 20 real-world Windows applications, demonstrating substantial improvements in robustness and execution accuracy over prior CUAs. Our results show that deep OS integration unlocks a scalable path toward reliable, user-aligned desktop automation. 

---
# FAIRGAME: a Framework for AI Agents Bias Recognition using Game Theory 

**Authors**: Alessio Buscemi, Daniele Proverbio, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Di Liò  

**Link**: [PDF](https://arxiv.org/pdf/2504.14325)  

**Abstract**: Letting AI agents interact in multi-agent applications adds a layer of complexity to the interpretability and prediction of AI outcomes, with profound implications for their trustworthy adoption in research and society. Game theory offers powerful models to capture and interpret strategic interaction among agents, but requires the support of reproducible, standardized and user-friendly IT frameworks to enable comparison and interpretation of results. To this end, we present FAIRGAME, a Framework for AI Agents Bias Recognition using Game Theory. We describe its implementation and usage, and we employ it to uncover biased outcomes in popular games among AI agents, depending on the employed Large Language Model (LLM) and used language, as well as on the personality trait or strategic knowledge of the agents. Overall, FAIRGAME allows users to reliably and easily simulate their desired games and scenarios and compare the results across simulation campaigns and with game-theoretic predictions, enabling the systematic discovery of biases, the anticipation of emerging behavior out of strategic interplays, and empowering further research into strategic decision-making using LLM agents. 

---
# Going Whole Hog: A Philosophical Defense of AI Cognition 

**Authors**: Herman Cappelen, Josh Dever  

**Link**: [PDF](https://arxiv.org/pdf/2504.13988)  

**Abstract**: This work defends the 'Whole Hog Thesis': sophisticated Large Language Models (LLMs) like ChatGPT are full-blown linguistic and cognitive agents, possessing understanding, beliefs, desires, knowledge, and intentions. We argue against prevailing methodologies in AI philosophy, rejecting starting points based on low-level computational details ('Just an X' fallacy) or pre-existing theories of mind. Instead, we advocate starting with simple, high-level observations of LLM behavior (e.g., answering questions, making suggestions) -- defending this data against charges of metaphor, loose talk, or pretense. From these observations, we employ 'Holistic Network Assumptions' -- plausible connections between mental capacities (e.g., answering implies knowledge, knowledge implies belief, action implies intention) -- to argue for the full suite of cognitive states. We systematically rebut objections based on LLM failures (hallucinations, planning/reasoning errors), arguing these don't preclude agency, often mirroring human fallibility. We address numerous 'Games of Lacks', arguing that LLMs do not lack purported necessary conditions for cognition (e.g., semantic grounding, embodiment, justification, intrinsic intentionality) or that these conditions are not truly necessary, often relying on anti-discriminatory arguments comparing LLMs to diverse human capacities. Our approach is evidential, not functionalist, and deliberately excludes consciousness. We conclude by speculating on the possibility of LLMs possessing 'alien' contents beyond human conceptual schemes. 

---
# Neural ATTF: A Scalable Solution to Lifelong Multi-Agent Path Planning 

**Authors**: Kushal Shah, Jihyun Park, Seung-Kyum Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15130)  

**Abstract**: Multi-Agent Pickup and Delivery (MAPD) is a fundamental problem in robotics, particularly in applications such as warehouse automation and logistics. Existing solutions often face challenges in scalability, adaptability, and efficiency, limiting their applicability in dynamic environments with real-time planning requirements. This paper presents Neural ATTF (Adaptive Task Token Framework), a new algorithm that combines a Priority Guided Task Matching (PGTM) Module with Neural STA* (Space-Time A*), a data-driven path planning method. Neural STA* enhances path planning by enabling rapid exploration of the search space through guided learned heuristics and ensures collision avoidance under dynamic constraints. PGTM prioritizes delayed agents and dynamically assigns tasks by prioritizing agents nearest to these tasks, optimizing both continuity and system throughput. Experimental evaluations against state-of-the-art MAPD algorithms, including TPTS, CENTRAL, RMCA, LNS-PBS, and LNS-wPBS, demonstrate the superior scalability, solution quality, and computational efficiency of Neural ATTF. These results highlight the framework's potential for addressing the critical demands of complex, real-world multi-agent systems operating in high-demand, unpredictable settings. 

---
# Fast-Slow Co-advancing Optimizer: Toward Harmonious Adversarial Training of GAN 

**Authors**: Lin Wang, Xiancheng Wang, Rui Wang, Zhibo Zhang, Minghang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15099)  

**Abstract**: Up to now, the training processes of typical Generative Adversarial Networks (GANs) are still particularly sensitive to data properties and hyperparameters, which may lead to severe oscillations, difficulties in convergence, or even failures to converge, especially when the overall variances of the training sets are large. These phenomena are often attributed to the training characteristics of such networks. Aiming at the problem, this paper develops a new intelligent optimizer, Fast-Slow Co-advancing Optimizer (FSCO), which employs reinforcement learning in the training process of GANs to make training easier. Specifically, this paper allows the training step size to be controlled by an agent to improve training stability, and makes the training process more intelligent with variable learning rates, making GANs less sensitive to step size. Experiments have been conducted on three benchmark datasets to verify the effectiveness of the developed FSCO. 

---
# SWE-Synth: Synthesizing Verifiable Bug-Fix Data to Enable Large Language Models in Resolving Real-World Bugs 

**Authors**: Minh V.T. Pham, Huy N. Phan, Hoang N. Phan, Cuong Le Chi, Tien N. Nguyen, Nghi D. Q. Bui  

**Link**: [PDF](https://arxiv.org/pdf/2504.14757)  

**Abstract**: Large language models (LLMs) are transforming automated program repair (APR) through agent-based approaches that localize bugs, generate patches, and verify fixes. However, the lack of high-quality, scalable training datasets, especially those with verifiable outputs and intermediate reasoning traces-limits progress, particularly for open-source models. In this work, we present SWE-Synth, a framework for synthesizing realistic, verifiable, and process-aware bug-fix datasets at the repository level. SWE-Synth leverages LLM agents to simulate debugging workflows, producing not only bug-fix pairs but also test cases and structured repair trajectories. Compared to manually curated datasets, our method scales with minimal human effort while preserving contextual richness and correctness. Experiments show that models trained on SWE-Synth outperform those trained on real-world datasets by 2.3% on SWE-Bench Lite. Our results highlight the potential of synthetic, agent-generated data to advance the state of the art in APR and software engineering automation. 

---
# An LLM-enabled Multi-Agent Autonomous Mechatronics Design Framework 

**Authors**: Zeyu Wang, Frank P.-W. Lo, Qian Chen, Yongqi Zhang, Chen Lin, Xu Chen, Zhenhua Yu, Alexander J. Thompson, Eric M. Yeatman, Benny P. L. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14681)  

**Abstract**: Existing LLM-enabled multi-agent frameworks are predominantly limited to digital or simulated environments and confined to narrowly focused knowledge domain, constraining their applicability to complex engineering tasks that require the design of physical embodiment, cross-disciplinary integration, and constraint-aware reasoning. This work proposes a multi-agent autonomous mechatronics design framework, integrating expertise across mechanical design, optimization, electronics, and software engineering to autonomously generate functional prototypes with minimal direct human design input. Operating primarily through a language-driven workflow, the framework incorporates structured human feedback to ensure robust performance under real-world constraints. To validate its capabilities, the framework is applied to a real-world challenge involving autonomous water-quality monitoring and sampling, where traditional methods are labor-intensive and ecologically disruptive. Leveraging the proposed system, a fully functional autonomous vessel was developed with optimized propulsion, cost-effective electronics, and advanced control. The design process was carried out by specialized agents, including a high-level planning agent responsible for problem abstraction and dedicated agents for structural, electronics, control, and software development. This approach demonstrates the potential of LLM-based multi-agent systems to automate real-world engineering workflows and reduce reliance on extensive domain expertise. 

---
# Towards Optimal Circuit Generation: Multi-Agent Collaboration Meets Collective Intelligence 

**Authors**: Haiyan Qin, Jiahao Feng, Xiaotong Feng, Wei W. Xing, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14625)  

**Abstract**: Large language models (LLMs) have transformed code generation, yet their application in hardware design produces gate counts 38\%--1075\% higher than human designs. We present CircuitMind, a multi-agent framework that achieves human-competitive efficiency through three key innovations: syntax locking (constraining generation to basic logic gates), retrieval-augmented generation (enabling knowledge-driven design), and dual-reward optimization (balancing correctness with efficiency). To evaluate our approach, we introduce TC-Bench, the first gate-level benchmark harnessing collective intelligence from the TuringComplete ecosystem -- a competitive circuit design platform with hundreds of thousands of players. Experiments show CircuitMind enables 55.6\% of model implementations to match or exceed top-tier human experts in composite efficiency metrics. Most remarkably, our framework elevates the 14B Phi-4 model to outperform both GPT-4o mini and Gemini 2.0 Flash, achieving efficiency comparable to the top 25\% of human experts without requiring specialized training. These innovations establish a new paradigm for hardware optimization where collaborative AI systems leverage collective human expertise to achieve optimal circuit designs. Our model, data, and code are open-source at this https URL. 

---
# Optimizing SIA Development: A Case Study in User-Centered Design for Estuary, a Multimodal Socially Interactive Agent Framework 

**Authors**: Spencer Lin, Miru Jun, Basem Rizk, Karen Shieh, Scott Fisher, Sharon Mozgai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14427)  

**Abstract**: This case study presents our user-centered design model for Socially Intelligent Agent (SIA) development frameworks through our experience developing Estuary, an open source multimodal framework for building low-latency real-time socially interactive agents. We leverage the Rapid Assessment Process (RAP) to collect the thoughts of leading researchers in the field of SIAs regarding the current state of the art for SIA development as well as their evaluation of how well Estuary may potentially address current research gaps. We achieve this through a series of end-user interviews conducted by a fellow researcher in the community. We hope that the findings of our work will not only assist the continued development of Estuary but also guide the development of other future frameworks and technologies for SIAs. 

---
# Planet as a Brain: Towards Internet of AgentSites based on AIOS Server 

**Authors**: Xiang Zhang, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14411)  

**Abstract**: The internet is undergoing a historical transformation from the "Internet of Websites" to the "Internet of AgentSites." While traditional Websites served as the foundation for information hosting and dissemination, a new frontier is emerging where AgentSites serve as the hubs of the internet, where each AgentSite hosts one or more AI agents that receive tasks, address them, and deliver actionable solutions, marking a significant shift in the digital landscape and representing the next generation of online ecosystems. Under this vision, AIOS, the AI Agent Operating System, serves as the server for the development, deployment and execution of AI agents, which is a fundamental infrastructure for the Internet of Agentsites.
In this paper, we introduce AIOS Server, a runtime framework to host agents and enable global-scale collaboration among decentralized agents. AIOS Server provides a communication protocol leveraging the Model Context Protocol (MCP) and JSON-RPC to enable agent-agent or human-agent interactions. Each AIOS node operates as a server to host and execute agents, while supporting peer-to-peer coordination without reliance on centralized orchestration. Based on AIOS Server, we further present the world's first practically deployed Internet of Agentsites (AIOS-IoA), including AgentHub for agent registration and discovery and AgentChat for interactive communication, at this https URL. The agent discovery mechanism based on Distributed Hash Tables (DHT) and a Gossip protocol serves as the search engine for the internet of agentsites. This work provides a practical foundation for building the Internet of Agentsites-a new paradigm where autonomous agents become first-class citizens of the web. The implementation is available at this https URL and will be integrated into the AIOS main branch at this https URL. 

---
# Hydra: An Agentic Reasoning Approach for Enhancing Adversarial Robustness and Mitigating Hallucinations in Vision-Language Models 

**Authors**: Chung-En, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14395)  

**Abstract**: To develop trustworthy Vision-Language Models (VLMs), it is essential to address adversarial robustness and hallucination mitigation, both of which impact factual accuracy in high-stakes applications such as defense and healthcare. Existing methods primarily focus on either adversarial defense or hallucination post-hoc correction, leaving a gap in unified robustness strategies. We introduce \textbf{Hydra}, an adaptive agentic framework that enhances plug-in VLMs through iterative reasoning, structured critiques, and cross-model verification, improving both resilience to adversarial perturbations and intrinsic model errors. Hydra employs an Action-Critique Loop, where it retrieves and critiques visual information, leveraging Chain-of-Thought (CoT) and In-Context Learning (ICL) techniques to refine outputs dynamically. Unlike static post-hoc correction methods, Hydra adapts to both adversarial manipulations and intrinsic model errors, making it robust to malicious perturbations and hallucination-related inaccuracies. We evaluate Hydra on four VLMs, three hallucination benchmarks, two adversarial attack strategies, and two adversarial defense methods, assessing performance on both clean and adversarial inputs. Results show that Hydra surpasses plug-in VLMs and state-of-the-art (SOTA) dehallucination methods, even without explicit adversarial defenses, demonstrating enhanced robustness and factual consistency. By bridging adversarial resistance and hallucination mitigation, Hydra provides a scalable, training-free solution for improving the reliability of VLMs in real-world applications. 

---
# Experience-based Refinement of Task Planning Knowledge in Autonomous Robots 

**Authors**: Hadeel Jazzaa, Thomas McCluskey, David Peebles  

**Link**: [PDF](https://arxiv.org/pdf/2504.14259)  

**Abstract**: The requirement for autonomous robots to exhibit higher-level cognitive skills by planning and adapting in an ever-changing environment is indeed a great challenge for the AI community. Progress has been made in the automated planning community on refinement and repair of an agent's symbolic knowledge to do task planning in an incomplete or changing environmental model, but these advances up to now have not been transferred to real physical robots. This paper demonstrates how a physical robot can be capable of adapting its symbolic knowledge of the environment, by using experiences in robot action execution to drive knowledge refinement and hence to improve the success rate of the task plans the robot creates. To implement more robust planning systems, we propose a method for refining domain knowledge to improve the knowledge on which intelligent robot behavior is based. This architecture has been implemented and evaluated using a NAO robot. The refined knowledge leads to the future synthesis of task plans which demonstrate decreasing rates of failure over time as faulty knowledge is removed or adjusted. 

---
# LLM-Driven NPCs: Cross-Platform Dialogue System for Games and Social Platforms 

**Authors**: Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.13928)  

**Abstract**: NPCs in traditional games are often limited by static dialogue trees and a single platform for interaction. To overcome these constraints, this study presents a prototype system that enables large language model (LLM)-powered NPCs to communicate with players both in the game en vironment (Unity) and on a social platform (Discord). Dialogue logs are stored in a cloud database (LeanCloud), allowing the system to synchronize memory between platforms and keep conversa tions coherent. Our initial experiments show that cross-platform interaction is technically feasible and suggest a solid foundation for future developments such as emotional modeling and persistent memory support. 

---
# The Human Robot Social Interaction (HSRI) Dataset: Benchmarking Foundational Models' Social Reasoning 

**Authors**: Dong Won Lee, Yubin Kim, Denison Guvenoz, Sooyeon Jeong, Parker Malachowsky, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.13898)  

**Abstract**: Our work aims to advance the social reasoning of embodied artificial intelligence (AI) agents in real-world social interactions. Recently, language models (LMs) and foundational models (FMs) are being utilized as automatic evaluators of human-AI interactions with the goal of eventually being used to improve the policy of the AI agent. To enable further research in this direction, we introduce a large-scale real-world Human Robot Social Interaction (HSRI) Dataset to benchmark the capabilities of LMs and FMs to identify and reason about social interactions, specifically with regard to robot social errors and competencies . Our dataset consists of 400 real-world human social robot interaction videos and over 10K annotations, detailing the robot's social errors, competencies, rationale, and corrective actions, capturing unique aspects of human-AI interaction only present in real-world interactions. To further assess AI models' ability to reason about social interactions, we propose eight new benchmark tasks for evaluating centered around whether AI models can (1) evaluate social interactions via detecting social errors and competencies, (2) identify the explanatory factors associated to errors and competencies, (3) understand the flow of real-world social interactions, and (4) provide reasons and corrective actions for social errors. Human studies and experiments with modern LMs and FMs reveal that current models struggle with these tasks, demonstrating that our dataset and benchmark provides a step forward towards socially intelligent AI. 

---
# Human aversion? Do AI Agents Judge Identity More Harshly Than Performance 

**Authors**: Yuanjun Feng, Vivek Chodhary, Yash Raj Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2504.13871)  

**Abstract**: This study examines the understudied role of algorithmic evaluation of human judgment in hybrid decision-making systems, a critical gap in management research. While extant literature focuses on human reluctance to follow algorithmic advice, we reverse the perspective by investigating how AI agents based on large language models (LLMs) assess and integrate human input. Our work addresses a pressing managerial constraint: firms barred from deploying LLMs directly due to privacy concerns can still leverage them as mediating tools (for instance, anonymized outputs or decision pipelines) to guide high-stakes choices like pricing or discounts without exposing proprietary data. Through a controlled prediction task, we analyze how an LLM-based AI agent weights human versus algorithmic predictions. We find that the AI system systematically discounts human advice, penalizing human errors more severely than algorithmic errors--a bias exacerbated when the agent's identity (human vs AI) is disclosed and the human is positioned second. These results reveal a disconnect between AI-generated trust metrics and the actual influence of human judgment, challenging assumptions about equitable human-AI collaboration. Our findings offer three key contributions. First, we identify a reverse algorithm aversion phenomenon, where AI agents undervalue human input despite comparable error rates. Second, we demonstrate how disclosure and positional bias interact to amplify this effect, with implications for system design. Third, we provide a framework for indirect LLM deployment that balances predictive power with data privacy. For practitioners, this research emphasize the need to audit AI weighting mechanisms, calibrate trust dynamics, and strategically design decision sequences in human-AI systems. 

---
