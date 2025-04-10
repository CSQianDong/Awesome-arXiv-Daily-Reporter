# SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills 

**Authors**: Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.07079)  

**Abstract**: To survive and thrive in complex environments, humans have evolved sophisticated self-improvement mechanisms through environment exploration, hierarchical abstraction of experiences into reuseable skills, and collaborative construction of an ever-growing skill repertoire. Despite recent advancements, autonomous web agents still lack crucial self-improvement capabilities, struggling with procedural knowledge abstraction, refining skills, and skill composition. In this work, we introduce SkillWeaver, a skill-centric framework enabling agents to self-improve by autonomously synthesizing reusable skills as APIs. Given a new website, the agent autonomously discovers skills, executes them for practice, and distills practice experiences into robust APIs. Iterative exploration continually expands a library of lightweight, plug-and-play APIs, significantly enhancing the agent's capabilities. Experiments on WebArena and real-world websites demonstrate the efficacy of SkillWeaver, achieving relative success rate improvements of 31.8% and 39.8%, respectively. Additionally, APIs synthesized by strong agents substantially enhance weaker agents through transferable skills, yielding improvements of up to 54.3% on WebArena. These results demonstrate the effectiveness of honing diverse website interactions into APIs, which can be seamlessly shared among various web agents. 

---
# AssistanceZero: Scalably Solving Assistance Games 

**Authors**: Cassidy Laidlaw, Eli Bronstein, Timothy Guo, Dylan Feng, Lukas Berglund, Justin Svegliato, Stuart Russell, Anca Dragan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07091)  

**Abstract**: Assistance games are a promising alternative to reinforcement learning from human feedback (RLHF) for training AI assistants. Assistance games resolve key drawbacks of RLHF, such as incentives for deceptive behavior, by explicitly modeling the interaction between assistant and user as a two-player game where the assistant cannot observe their shared goal. Despite their potential, assistance games have only been explored in simple settings. Scaling them to more complex environments is difficult because it requires both solving intractable decision-making problems under uncertainty and accurately modeling human users' behavior. We present the first scalable approach to solving assistance games and apply it to a new, challenging Minecraft-based assistance game with over $10^{400}$ possible goals. Our approach, AssistanceZero, extends AlphaZero with a neural network that predicts human actions and rewards, enabling it to plan under uncertainty. We show that AssistanceZero outperforms model-free RL algorithms and imitation learning in the Minecraft-based assistance game. In a human study, our AssistanceZero-trained assistant significantly reduces the number of actions participants take to complete building tasks in Minecraft. Our results suggest that assistance games are a tractable framework for training effective AI assistants in complex environments. Our code and models are available at this https URL. 

---
# Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration 

**Authors**: Kostas Hatalis, Despina Christou, Vyshnavi Kondapalli  

**Link**: [PDF](https://arxiv.org/pdf/2504.06943)  

**Abstract**: Agents powered by Large Language Models (LLMs) have recently demonstrated impressive capabilities in various tasks. Still, they face limitations in tasks requiring specific, structured knowledge, flexibility, or accountable decision-making. While agents are capable of perceiving their environments, forming inferences, planning, and executing actions towards goals, they often face issues such as hallucinations and lack of contextual memory across interactions. This paper explores how Case-Based Reasoning (CBR), a strategy that solves new problems by referencing past experiences, can be integrated into LLM agent frameworks. This integration allows LLMs to leverage explicit knowledge, enhancing their effectiveness. We systematically review the theoretical foundations of these enhanced agents, identify critical framework components, and formulate a mathematical model for the CBR processes of case retrieval, adaptation, and learning. We also evaluate CBR-enhanced agents against other methods like Chain-of-Thought reasoning and standard Retrieval-Augmented Generation, analyzing their relative strengths. Moreover, we explore how leveraging CBR's cognitive dimensions (including self-reflection, introspection, and curiosity) via goal-driven autonomy mechanisms can further enhance the LLM agent capabilities. Contributing to the ongoing research on neuro-symbolic hybrid systems, this work posits CBR as a viable technique for enhancing the reasoning skills and cognitive aspects of autonomous LLM agents. 

---
# Right Prediction, Wrong Reasoning: Uncovering LLM Misalignment in RA Disease Diagnosis 

**Authors**: Umakanta Maharana, Sarthak Verma, Avarna Agarwal, Prakashini Mruthyunjaya, Dwarikanath Mahapatra, Sakir Ahmed, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2504.06581)  

**Abstract**: Large language models (LLMs) offer a promising pre-screening tool, improving early disease detection and providing enhanced healthcare access for underprivileged communities. The early diagnosis of various diseases continues to be a significant challenge in healthcare, primarily due to the nonspecific nature of early symptoms, the shortage of expert medical practitioners, and the need for prolonged clinical evaluations, all of which can delay treatment and adversely affect patient outcomes. With impressive accuracy in prediction across a range of diseases, LLMs have the potential to revolutionize clinical pre-screening and decision-making for various medical conditions. In this work, we study the diagnostic capability of LLMs for Rheumatoid Arthritis (RA) with real world patients data. Patient data was collected alongside diagnoses from medical experts, and the performance of LLMs was evaluated in comparison to expert diagnoses for RA disease prediction. We notice an interesting pattern in disease diagnosis and find an unexpected \textit{misalignment between prediction and explanation}. We conduct a series of multi-round analyses using different LLM agents. The best-performing model accurately predicts rheumatoid arthritis (RA) diseases approximately 95\% of the time. However, when medical experts evaluated the reasoning generated by the model, they found that nearly 68\% of the reasoning was incorrect. This study highlights a clear misalignment between LLMs high prediction accuracy and its flawed reasoning, raising important questions about relying on LLM explanations in clinical settings. \textbf{LLMs provide incorrect reasoning to arrive at the correct answer for RA disease diagnosis.} 

---
# Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games 

**Authors**: Seungwon Lim, Seungbeen Lee, Dongjun Min, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06868)  

**Abstract**: Artificial agents are increasingly central to complex interactions and decision-making tasks, yet aligning their behaviors with desired human values remains an open challenge. In this work, we investigate how human-like personality traits influence agent behavior and performance within text-based interactive environments. We introduce PANDA: PersonalityAdapted Neural Decision Agents, a novel method for projecting human personality traits onto agents to guide their behavior. To induce personality in a text-based game agent, (i) we train a personality classifier to identify what personality type the agent's actions exhibit, and (ii) we integrate the personality profiles directly into the agent's policy-learning pipeline. By deploying agents embodying 16 distinct personality types across 25 text-based games and analyzing their trajectories, we demonstrate that an agent's action decisions can be guided toward specific personality profiles. Moreover, certain personality types, such as those characterized by higher levels of Openness, display marked advantages in performance. These findings underscore the promise of personality-adapted agents for fostering more aligned, effective, and human-centric decision-making in interactive environments. 

---
# OPAL: Encoding Causal Understanding of Physical Systems for Robot Learning 

**Authors**: Daniel Tcheurekdjian, Joshua Klasmeier, Tom Cooney, Christopher McCann, Tyler Fenstermaker  

**Link**: [PDF](https://arxiv.org/pdf/2504.06538)  

**Abstract**: We present OPAL (Operant Physical Agent with Language), a novel vision-language-action architecture that introduces topological constraints to flow matching for robotic control. To do so, we further introduce topological attention. Our approach models action sequences as topologically-structured representations with non-trivial constraints. Experimental results across 10 complex manipulation tasks demonstrate OPAL's superior performance compared to previous approaches, including Octo, OpenVLA, and ${\pi}$0.
Our architecture achieves significant improvements in zero-shot performance without requiring task-specific fine-tuning, while reducing inference computational requirements by 42%. The theoretical guarantees provided by our topological approach result in more coherent long-horizon action sequences. Our results highlight the potential of constraining the search space of learning problems in robotics by deriving from fundamental physical laws, and the possibility of using topological attention to embed causal understanding into transformer architectures. 

---
# Agent-Arena: A General Framework for Evaluating Control Algorithms 

**Authors**: Halid Abdulrahim Kadi, Kasim Terzić  

**Link**: [PDF](https://arxiv.org/pdf/2504.06468)  

**Abstract**: Robotic research is inherently challenging, requiring expertise in diverse environments and control algorithms. Adapting algorithms to new environments often poses significant difficulties, compounded by the need for extensive hyper-parameter tuning in data-driven methods. To address these challenges, we present Agent-Arena, a Python framework designed to streamline the integration, replication, development, and testing of decision-making policies across a wide range of benchmark environments. Unlike existing frameworks, Agent-Arena is uniquely generalised to support all types of control algorithms and is adaptable to both simulation and real-robot scenarios. Please see our GitHub repository this https URL. 

---
# Dynamic Evaluation Framework for Personalized and Trustworthy Agents: A Multi-Session Approach to Preference Adaptability 

**Authors**: Chirag Shah, Hideo Joho, Kirandeep Kaur, Preetam Prabhu Srikar Dammu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06277)  

**Abstract**: Recent advancements in generative AI have significantly increased interest in personalized agents. With increased personalization, there is also a greater need for being able to trust decision-making and action taking capabilities of these agents. However, the evaluation methods for these agents remain outdated and inadequate, often failing to capture the dynamic and evolving nature of user interactions. In this conceptual article, we argue for a paradigm shift in evaluating personalized and adaptive agents. We propose a comprehensive novel framework that models user personas with unique attributes and preferences. In this framework, agents interact with these simulated users through structured interviews to gather their preferences and offer customized recommendations. These recommendations are then assessed dynamically using simulations driven by Large Language Models (LLMs), enabling an adaptive and iterative evaluation process. Our flexible framework is designed to support a variety of agents and applications, ensuring a comprehensive and versatile evaluation of recommendation strategies that focus on proactive, personalized, and trustworthy aspects. 

---
# EXCLAIM: An Explainable Cross-Modal Agentic System for Misinformation Detection with Hierarchical Retrieval 

**Authors**: Yin Wu, Zhengxuan Zhang, Fuling Wang, Yuyu Luo, Hui Xiong, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06269)  

**Abstract**: Misinformation continues to pose a significant challenge in today's information ecosystem, profoundly shaping public perception and behavior. Among its various manifestations, Out-of-Context (OOC) misinformation is particularly obscure, as it distorts meaning by pairing authentic images with misleading textual narratives. Existing methods for detecting OOC misinformation predominantly rely on coarse-grained similarity metrics between image-text pairs, which often fail to capture subtle inconsistencies or provide meaningful explainability. While multi-modal large language models (MLLMs) demonstrate remarkable capabilities in visual reasoning and explanation generation, they have not yet demonstrated the capacity to address complex, fine-grained, and cross-modal distinctions necessary for robust OOC detection. To overcome these limitations, we introduce EXCLAIM, a retrieval-based framework designed to leverage external knowledge through multi-granularity index of multi-modal events and entities. Our approach integrates multi-granularity contextual analysis with a multi-agent reasoning architecture to systematically evaluate the consistency and integrity of multi-modal news content. Comprehensive experiments validate the effectiveness and resilience of EXCLAIM, demonstrating its ability to detect OOC misinformation with 4.3% higher accuracy compared to state-of-the-art approaches, while offering explainable and actionable insights. 

---
# CMAT: A Multi-Agent Collaboration Tuning Framework for Enhancing Small Language Models 

**Authors**: Xuechen Liang, Meiling Tao, Yinghui Xia, Tianyu Shi, Jun Wang, JingSong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2404.01663)  

**Abstract**: Open large language models (LLMs) have significantly advanced the field of natural language processing, showcasing impressive performance across various this http URL the significant advancements in LLMs, their effective operation still relies heavily on human input to accurately guide the dialogue flow, with agent tuning being a crucial optimization technique that involves human adjustments to the model for better response to such this http URL this dependency, our work introduces the TinyAgent model, trained on a meticulously curated high-quality dataset. We also present the Collaborative Multi-Agent Tuning (CMAT) framework, an innovative system designed to augment language agent capabilities through adaptive weight updates based on environmental feedback. This framework fosters collaborative learning and real-time adaptation among multiple intelligent agents, enhancing their context-awareness and long-term memory. In this research, we propose a new communication agent framework that integrates multi-agent systems with environmental feedback mechanisms, offering a scalable method to explore cooperative behaviors. Notably, our TinyAgent-7B model exhibits performance on par with GPT-3.5, despite having fewer parameters, signifying a substantial improvement in the efficiency and effectiveness of LLMs. 

---
# RAVEN: An Agentic Framework for Multimodal Entity Discovery from Large-Scale Video Collections 

**Authors**: Kevin Dela Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2504.06272)  

**Abstract**: We present RAVEN an adaptive AI agent framework designed for multimodal entity discovery and retrieval in large-scale video collections. Synthesizing information across visual, audio, and textual modalities, RAVEN autonomously processes video data to produce structured, actionable representations for downstream tasks. Key contributions include (1) a category understanding step to infer video themes and general-purpose entities, (2) a schema generation mechanism that dynamically defines domain-specific entities and attributes, and (3) a rich entity extraction process that leverages semantic retrieval and schema-guided prompting. RAVEN is designed to be model-agnostic, allowing the integration of different vision-language models (VLMs) and large language models (LLMs) based on application-specific requirements. This flexibility supports diverse applications in personalized search, content discovery, and scalable information retrieval, enabling practical applications across vast datasets. 

---
# Inducing Programmatic Skills for Agentic Tasks 

**Authors**: Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, Daniel Fried  

**Link**: [PDF](https://arxiv.org/pdf/2504.06821)  

**Abstract**: To succeed in common digital tasks such as web navigation, agents must carry out a variety of specialized tasks such as searching for products or planning a travel route. To tackle these tasks, agents can bootstrap themselves by learning task-specific skills online through interaction with the web environment. In this work, we demonstrate that programs are an effective representation for skills. We propose agent skill induction (ASI), which allows agents to adapt themselves by inducing, verifying, and utilizing program-based skills on the fly. We start with an evaluation on the WebArena agent benchmark and show that ASI outperforms the static baseline agent and its text-skill counterpart by 23.5% and 11.3% in success rate, mainly thanks to the programmatic verification guarantee during the induction phase. ASI also improves efficiency by reducing 10.7-15.3% of the steps over baselines, by composing primitive actions (e.g., click) into higher-level skills (e.g., search product). We then highlight the efficacy of ASI in remaining efficient and accurate under scaled-up web activities. Finally, we examine the generalizability of induced skills when transferring between websites, and find that ASI can effectively reuse common skills, while also updating incompatible skills to versatile website changes. 

---
# A Unified Agentic Framework for Evaluating Conditional Image Generation 

**Authors**: Jifang Wang, Xue Yang, Longyue Wang, Zhenran Xu, Yiyu Wang, Yaowei Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07046)  

**Abstract**: Conditional image generation has gained significant attention for its ability to personalize content. However, the field faces challenges in developing task-agnostic, reliable, and explainable evaluation metrics. This paper introduces CIGEval, a unified agentic framework for comprehensive evaluation of conditional image generation tasks. CIGEval utilizes large multimodal models (LMMs) as its core, integrating a multi-functional toolbox and establishing a fine-grained evaluation framework. Additionally, we synthesize evaluation trajectories for fine-tuning, empowering smaller LMMs to autonomously select appropriate tools and conduct nuanced analyses based on tool outputs. Experiments across seven prominent conditional image generation tasks demonstrate that CIGEval (GPT-4o version) achieves a high correlation of 0.4625 with human assessments, closely matching the inter-annotator correlation of 0.47. Moreover, when implemented with 7B open-source LMMs using only 2.3K training trajectories, CIGEval surpasses the previous GPT-4o-based state-of-the-art method. Case studies on GPT-4o image generation highlight CIGEval's capability in identifying subtle issues related to subject consistency and adherence to control guidance, indicating its great potential for automating evaluation of image generation tasks with human-level reliability. 

---
