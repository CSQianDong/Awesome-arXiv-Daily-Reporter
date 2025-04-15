# RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users 

**Authors**: Suyu Ye, Haojun Shi, Darren Shih, Hyokun Yun, Tanya Roosta, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10445)  

**Abstract**: To achieve successful assistance with long-horizon web-based tasks, AI agents must be able to sequentially follow real-world user instructions over a long period. Unlike existing web-based agent benchmarks, sequential instruction following in the real world poses significant challenges beyond performing a single, clearly defined task. For instance, real-world human instructions can be ambiguous, require different levels of AI assistance, and may evolve over time, reflecting changes in the user's mental state. To address this gap, we introduce RealWebAssist, a novel benchmark designed to evaluate sequential instruction-following in realistic scenarios involving long-horizon interactions with the web, visual GUI grounding, and understanding ambiguous real-world user instructions. RealWebAssist includes a dataset of sequential instructions collected from real-world human users. Each user instructs a web-based assistant to perform a series of tasks on multiple websites. A successful agent must reason about the true intent behind each instruction, keep track of the mental state of the user, understand user-specific routines, and ground the intended tasks to actions on the correct GUI elements. Our experimental results show that state-of-the-art models struggle to understand and ground user instructions, posing critical challenges in following real-world user instructions for long-horizon web assistance. 

---
# Pay Attention to What and Where? Interpretable Feature Extractor in Vision-based Deep Reinforcement Learning 

**Authors**: Tien Pham, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10071)  

**Abstract**: Current approaches in Explainable Deep Reinforcement Learning have limitations in which the attention mask has a displacement with the objects in visual input. This work addresses a spatial problem within traditional Convolutional Neural Networks (CNNs). We propose the Interpretable Feature Extractor (IFE) architecture, aimed at generating an accurate attention mask to illustrate both "what" and "where" the agent concentrates on in the spatial domain. Our design incorporates a Human-Understandable Encoding module to generate a fully interpretable attention mask, followed by an Agent-Friendly Encoding module to enhance the agent's learning efficiency. These two components together form the Interpretable Feature Extractor for vision-based deep reinforcement learning to enable the model's interpretability. The resulting attention mask is consistent, highly understandable by humans, accurate in spatial dimension, and effectively highlights important objects or locations in visual input. The Interpretable Feature Extractor is integrated into the Fast and Data-efficient Rainbow framework, and evaluated on 57 ATARI games to show the effectiveness of the proposed approach on Spatial Preservation, Interpretability, and Data-efficiency. Finally, we showcase the versatility of our approach by incorporating the IFE into the Asynchronous Advantage Actor-Critic Model. 

---
# Two Heads are Better Than One: Test-time Scaling of Multi-agent Collaborative Reasoning 

**Authors**: Can Jin, Hongwu Peng, Qixin Zhang, Yujin Tang, Dimitris N. Metaxas, Tong Che  

**Link**: [PDF](https://arxiv.org/pdf/2504.09772)  

**Abstract**: Multi-agent systems (MAS) built on large language models (LLMs) offer a promising path toward solving complex, real-world tasks that single-agent systems often struggle to manage. While recent advancements in test-time scaling (TTS) have significantly improved single-agent performance on challenging reasoning tasks, how to effectively scale collaboration and reasoning in MAS remains an open question. In this work, we introduce an adaptive multi-agent framework designed to enhance collaborative reasoning through both model-level training and system-level coordination. We construct M500, a high-quality dataset containing 500 multi-agent collaborative reasoning traces, and fine-tune Qwen2.5-32B-Instruct on this dataset to produce M1-32B, a model optimized for multi-agent collaboration. To further enable adaptive reasoning, we propose a novel CEO agent that dynamically manages the discussion process, guiding agent collaboration and adjusting reasoning depth for more effective problem-solving. Evaluated in an open-source MAS across a range of tasks-including general understanding, mathematical reasoning, and coding-our system significantly outperforms strong baselines. For instance, M1-32B achieves 12% improvement on GPQA-Diamond, 41% on AIME2024, and 10% on MBPP-Sanitized, matching the performance of state-of-the-art models like DeepSeek-R1 on some tasks. These results highlight the importance of both learned collaboration and adaptive coordination in scaling multi-agent reasoning. Code is available at this https URL 

---
# Can LLM feedback enhance review quality? A randomized study of 20K reviews at ICLR 2025 

**Authors**: Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng, Fei Sha, Rose Yu, Carl Vondrick, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.09737)  

**Abstract**: Peer review at AI conferences is stressed by rapidly rising submission volumes, leading to deteriorating review quality and increased author dissatisfaction. To address these issues, we developed Review Feedback Agent, a system leveraging multiple large language models (LLMs) to improve review clarity and actionability by providing automated feedback on vague comments, content misunderstandings, and unprofessional remarks to reviewers. Implemented at ICLR 2025 as a large randomized control study, our system provided optional feedback to more than 20,000 randomly selected reviews. To ensure high-quality feedback for reviewers at this scale, we also developed a suite of automated reliability tests powered by LLMs that acted as guardrails to ensure feedback quality, with feedback only being sent to reviewers if it passed all the tests. The results show that 27% of reviewers who received feedback updated their reviews, and over 12,000 feedback suggestions from the agent were incorporated by those reviewers. This suggests that many reviewers found the AI-generated feedback sufficiently helpful to merit updating their reviews. Incorporating AI feedback led to significantly longer reviews (an average increase of 80 words among those who updated after receiving feedback) and more informative reviews, as evaluated by blinded researchers. Moreover, reviewers who were selected to receive AI feedback were also more engaged during paper rebuttals, as seen in longer author-reviewer discussions. This work demonstrates that carefully designed LLM-generated review feedback can enhance peer review quality by making reviews more specific and actionable while increasing engagement between reviewers and authors. The Review Feedback Agent is publicly available at this https URL. 

---
# EmoAgent: Assessing and Safeguarding Human-AI Interaction for Mental Health Safety 

**Authors**: Jiahao Qiu, Yinghui He, Xinzhe Juan, Yiming Wang, Yuhan Liu, Zixin Yao, Yue Wu, Xun Jiang, Ling Yang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09689)  

**Abstract**: The rise of LLM-driven AI characters raises safety concerns, particularly for vulnerable human users with psychological disorders. To address these risks, we propose EmoAgent, a multi-agent AI framework designed to evaluate and mitigate mental health hazards in human-AI interactions. EmoAgent comprises two components: EmoEval simulates virtual users, including those portraying mentally vulnerable individuals, to assess mental health changes before and after interactions with AI characters. It uses clinically proven psychological and psychiatric assessment tools (PHQ-9, PDI, PANSS) to evaluate mental risks induced by LLM. EmoGuard serves as an intermediary, monitoring users' mental status, predicting potential harm, and providing corrective feedback to mitigate risks. Experiments conducted in popular character-based chatbots show that emotionally engaging dialogues can lead to psychological deterioration in vulnerable users, with mental state deterioration in more than 34.4% of the simulations. EmoGuard significantly reduces these deterioration rates, underscoring its role in ensuring safer AI-human interactions. Our code is available at: this https URL 

---
# A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems 

**Authors**: Zixuan Ke, Fangkai Jiao, Yifei Ming, Xuan-Phi Nguyen, Austin Xu, Do Xuan Long, Minzhi Li, Chengwei Qin, Peifeng Wang, Silvio Savarese, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2504.09037)  

**Abstract**: Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ... 

---
# Endowing Embodied Agents with Spatial Reasoning Capabilities for Vision-and-Language Navigation 

**Authors**: Luo Ling, Bai Qianqian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08806)  

**Abstract**: Enhancing the spatial perception capabilities of mobile robots is crucial for achieving embodied Vision-and-Language Navigation (VLN). Although significant progress has been made in simulated environments, directly transferring these capabilities to real-world scenarios often results in severe hallucination phenomena, causing robots to lose effective spatial awareness. To address this issue, we propose BrainNav, a bio-inspired spatial cognitive navigation framework inspired by biological spatial cognition theories and cognitive map theory. BrainNav integrates dual-map (coordinate map and topological map) and dual-orientation (relative orientation and absolute orientation) strategies, enabling real-time navigation through dynamic scene capture and path planning. Its five core modules-Hippocampal Memory Hub, Visual Cortex Perception Engine, Parietal Spatial Constructor, Prefrontal Decision Center, and Cerebellar Motion Execution Unit-mimic biological cognitive functions to reduce spatial hallucinations and enhance adaptability. Validated in a zero-shot real-world lab environment using the Limo Pro robot, BrainNav, compatible with GPT-4, outperforms existing State-of-the-Art (SOTA) Vision-and-Language Navigation in Continuous Environments (VLN-CE) methods without fine-tuning. 

---
# GridMind: A Multi-Agent NLP Framework for Unified, Cross-Modal NFL Data Insights 

**Authors**: Jordan Chipka, Chris Moyer, Clay Troyer, Tyler Fuelling, Jeremy Hochstedler  

**Link**: [PDF](https://arxiv.org/pdf/2504.08747)  

**Abstract**: The rapid growth of big data and advancements in computational techniques have significantly transformed sports analytics. However, the diverse range of data sources -- including structured statistics, semi-structured formats like sensor data, and unstructured media such as written articles, audio, and video -- creates substantial challenges in extracting actionable insights. These various formats, often referred to as multimodal data, require integration to fully leverage their potential. Conventional systems, which typically prioritize structured data, face limitations when processing and combining these diverse content types, reducing their effectiveness in real-time sports analysis.
To address these challenges, recent research highlights the importance of multimodal data integration for capturing the complexity of real-world sports environments. Building on this foundation, this paper introduces GridMind, a multi-agent framework that unifies structured, semi-structured, and unstructured data through Retrieval-Augmented Generation (RAG) and large language models (LLMs) to facilitate natural language querying of NFL data. This approach aligns with the evolving field of multimodal representation learning, where unified models are increasingly essential for real-time, cross-modal interactions.
GridMind's distributed architecture includes specialized agents that autonomously manage each stage of a prompt -- from interpretation and data retrieval to response synthesis. This modular design enables flexible, scalable handling of multimodal data, allowing users to pose complex, context-rich questions and receive comprehensive, intuitive responses via a conversational interface. 

---
# Teacher Motion Priors: Enhancing Robot Locomotion over Challenging Terrain 

**Authors**: Fangcheng Jin, Yuqi Wang, Peixin Ma, Guodong Yang, Pan Zhao, En Li, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10390)  

**Abstract**: Achieving robust locomotion on complex terrains remains a challenge due to high dimensional control and environmental uncertainties. This paper introduces a teacher prior framework based on the teacher student paradigm, integrating imitation and auxiliary task learning to improve learning efficiency and generalization. Unlike traditional paradigms that strongly rely on encoder-based state embeddings, our framework decouples the network design, simplifying the policy network and deployment. A high performance teacher policy is first trained using privileged information to acquire generalizable motion skills. The teacher's motion distribution is transferred to the student policy, which relies only on noisy proprioceptive data, via a generative adversarial mechanism to mitigate performance degradation caused by distributional shifts. Additionally, auxiliary task learning enhances the student policy's feature representation, speeding up convergence and improving adaptability to varying terrains. The framework is validated on a humanoid robot, showing a great improvement in locomotion stability on dynamic terrains and significant reductions in development costs. This work provides a practical solution for deploying robust locomotion strategies in humanoid robots. 

---
# C-FAITH: A Chinese Fine-Grained Benchmark for Automated Hallucination Evaluation 

**Authors**: Xu Zhang, Zhifei Liu, Jiahao Wang, Huixuan Zhang, Fan Xu, Junzhe Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2504.10167)  

**Abstract**: Despite the rapid advancement of large language models, they remain highly susceptible to generating hallucinations, which significantly hinders their widespread application. Hallucination research requires dynamic and fine-grained evaluation. However, most existing hallucination benchmarks (especially in Chinese language) rely on human annotations, making automatical and cost-effective hallucination evaluation challenging. To address this, we introduce HaluAgent, an agentic framework that automatically constructs fine-grained QA dataset based on some knowledge documents. Our experiments demonstrate that the manually designed rules and prompt optimization can improve the quality of generated data. Using HaluAgent, we construct C-FAITH, a Chinese QA hallucination benchmark created from 1,399 knowledge documents obtained from web scraping, totaling 60,702 entries. We comprehensively evaluate 16 mainstream LLMs with our proposed C-FAITH, providing detailed experimental results and analysis. 

---
# Vision based driving agent for race car simulation environments 

**Authors**: Gergely Bári, László Palkovics  

**Link**: [PDF](https://arxiv.org/pdf/2504.10266)  

**Abstract**: In recent years, autonomous driving has become a popular field of study. As control at tire grip limit is essential during emergency situations, algorithms developed for racecars are useful for road cars too. This paper examines the use of Deep Reinforcement Learning (DRL) to solve the problem of grip limit driving in a simulated environment. Proximal Policy Optimization (PPO) method is used to train an agent to control the steering wheel and pedals of the vehicle, using only visual inputs to achieve professional human lap times. The paper outlines the formulation of the task of time optimal driving on a race track as a deep reinforcement learning problem, and explains the chosen observations, actions, and reward functions. The results demonstrate human-like learning and driving behavior that utilize maximum tire grip potential. 

---
# MT-R1-Zero: Advancing LLM-based Machine Translation via R1-Zero-like Reinforcement Learning 

**Authors**: Zhaopeng Feng, Shaosheng Cao, Jiahan Ren, Jiayuan Su, Ruizhe Chen, Yan Zhang, Zhe Xu, Yao Hu, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10160)  

**Abstract**: Large-scale reinforcement learning (RL) methods have proven highly effective in enhancing the reasoning abilities of large language models (LLMs), particularly for tasks with verifiable solutions such as mathematics and coding. However, applying this idea to machine translation (MT), where outputs are flexibly formatted and difficult to automatically evaluate with explicit rules, remains underexplored. In this work, we introduce MT-R1-Zero, the first open-source adaptation of the R1-Zero RL framework for MT without supervised fine-tuning or cold-start. We propose a rule-metric mixed reward mechanism to guide LLMs towards improved translation quality via emergent reasoning. On the WMT 24 English-Chinese benchmark, our MT-R1-Zero-3B-Mix achieves competitive performance, surpassing TowerInstruct-7B-v0.2 by an average of 1.26 points. Meanwhile, our MT-R1-Zero-7B-Mix attains a high average score of 62.25 across all metrics, placing it on par with advanced proprietary models such as GPT-4o and Claude-3.5-Sonnet, while the MT-R1-Zero-7B-Sem variant achieves state-of-the-art scores on semantic metrics. Moreover, our work exhibits strong generalization capabilities on out-of-distribution MT tasks, robustly supporting multilingual and low-resource settings. Extensive analysis of model behavior across different initializations and reward metrics offers pioneering insight into the critical role of reward design, LLM adaptability, training dynamics, and emergent reasoning patterns within the R1-Zero paradigm for MT. Our code is available at this https URL. 

---
# EmbodiedAgent: A Scalable Hierarchical Approach to Overcome Practical Challenge in Multi-Robot Control 

**Authors**: Hanwen Wan, Yifei Chen, Zeyu Wei, Dongrui Li, Zexin Lin, Donghao Wu, Jiu Cheng, Yuxiang Zhang, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.10030)  

**Abstract**: This paper introduces EmbodiedAgent, a hierarchical framework for heterogeneous multi-robot control. EmbodiedAgent addresses critical limitations of hallucination in impractical tasks. Our approach integrates a next-action prediction paradigm with a structured memory system to decompose tasks into executable robot skills while dynamically validating actions against environmental constraints. We present MultiPlan+, a dataset of more than 18,000 annotated planning instances spanning 100 scenarios, including a subset of impractical cases to mitigate hallucination. To evaluate performance, we propose the Robot Planning Assessment Schema (RPAS), combining automated metrics with LLM-aided expert grading. Experiments demonstrate EmbodiedAgent's superiority over state-of-the-art models, achieving 71.85% RPAS score. Real-world validation in an office service task highlights its ability to coordinate heterogeneous robots for long-horizon objectives. 

---
# PestMA: LLM-based Multi-Agent System for Informed Pest Management 

**Authors**: Hongrui Shi, Shunbao Li, Zhipeng Yuan, Po Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09855)  

**Abstract**: Effective pest management is complex due to the need for accurate, context-specific decisions. Recent advancements in large language models (LLMs) open new possibilities for addressing these challenges by providing sophisticated, adaptive knowledge acquisition and reasoning. However, existing LLM-based pest management approaches often rely on a single-agent paradigm, which can limit their capacity to incorporate diverse external information, engage in systematic validation, and address complex, threshold-driven decisions. To overcome these limitations, we introduce PestMA, an LLM-based multi-agent system (MAS) designed to generate reliable and evidence-based pest management advice. Building on an editorial paradigm, PestMA features three specialized agents, an Editor for synthesizing pest management recommendations, a Retriever for gathering relevant external data, and a Validator for ensuring correctness. Evaluations on real-world pest scenarios demonstrate that PestMA achieves an initial accuracy of 86.8% for pest management decisions, which increases to 92.6% after validation. These results underscore the value of collaborative agent-based workflows in refining and validating decisions, highlighting the potential of LLM-based multi-agent systems to automate and enhance pest management processes. 

---
# StruPhantom: Evolutionary Injection Attacks on Black-Box Tabular Agents Powered by Large Language Models 

**Authors**: Yang Feng, Xudong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.09841)  

**Abstract**: The proliferation of autonomous agents powered by large language models (LLMs) has revolutionized popular business applications dealing with tabular data, i.e., tabular agents. Although LLMs are observed to be vulnerable against prompt injection attacks from external data sources, tabular agents impose strict data formats and predefined rules on the attacker's payload, which are ineffective unless the agent navigates multiple layers of structural data to incorporate the payload. To address the challenge, we present a novel attack termed StruPhantom which specifically targets black-box LLM-powered tabular agents. Our attack designs an evolutionary optimization procedure which continually refines attack payloads via the proposed constrained Monte Carlo Tree Search augmented by an off-topic evaluator. StruPhantom helps systematically explore and exploit the weaknesses of target applications to achieve goal hijacking. Our evaluation validates the effectiveness of StruPhantom across various LLM-based agents, including those on real-world platforms, and attack scenarios. Our attack achieves over 50% higher success rates than baselines in enforcing the application's response to contain phishing links or malicious codes. 

---
# Training Small Reasoning LLMs with Cognitive Preference Alignment 

**Authors**: Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09802)  

**Abstract**: The reasoning capabilities of large language models (LLMs), such as OpenAI's o1 and DeepSeek-R1, have seen substantial advancements through deep thinking. However, these enhancements come with significant resource demands, underscoring the need to explore strategies to train effective reasoning LLMs with far fewer parameters. A critical challenge is that smaller models have different capacities and cognitive trajectories than their larger counterparts. Hence, direct distillation of chain-of-thought (CoT) results from large LLMs to smaller ones can be sometimes ineffective and requires a huge amount of annotated data. In this paper, we introduce a novel framework called Critique-Rethink-Verify (CRV), designed for training smaller yet powerful reasoning LLMs. Our CRV framework consists of multiple LLM agents, each specializing in unique abilities: (i) critiquing the CoTs according to the cognitive capabilities of smaller models, (ii) rethinking and refining these CoTs based on the critiques, and (iii) verifying the correctness of the refined results. We further propose the cognitive preference optimization (CogPO) algorithm to enhance the reasoning abilities of smaller models by aligning thoughts of these models with their cognitive capacities. Comprehensive evaluations on challenging reasoning benchmarks demonstrate the efficacy of CRV and CogPO, which outperforms other training methods by a large margin. 

---
# Reasoning Court: Combining Reasoning, Action, and Judgment for Multi-Hop Reasoning 

**Authors**: Jingtian Wu, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2504.09781)  

**Abstract**: While large language models (LLMs) have demonstrated strong capabilities in tasks like question answering and fact verification, they continue to suffer from hallucinations and reasoning errors, especially in multi-hop tasks that require integration of multiple information sources. Current methods address these issues through retrieval-based techniques (grounding reasoning in external evidence), reasoning-based approaches (enhancing coherence via improved prompting), or hybrid strategies combining both elements. One prominent hybrid method, ReAct, has outperformed purely retrieval-based or reasoning-based approaches; however, it lacks internal verification of intermediate reasoning steps, allowing potential errors to propagate through complex reasoning tasks. In this paper, we introduce Reasoning Court (RC), a novel framework that extends iterative reasoning-and-retrieval methods, such as ReAct, with a dedicated LLM judge. Unlike ReAct, RC employs this judge to independently evaluate multiple candidate answers and their associated reasoning generated by separate LLM agents. The judge is asked to select the answer that it considers the most factually grounded and logically coherent based on the presented reasoning and evidence, or synthesizes a new answer using available evidence and its pre-trained knowledge if all candidates are inadequate, flawed, or invalid. Evaluations on multi-hop benchmarks (HotpotQA, MuSiQue) and fact-verification (FEVER) demonstrate that RC consistently outperforms state-of-the-art few-shot prompting methods without task-specific fine-tuning. 

---
# Adapting Robot's Explanation for Failures Based on Observed Human Behavior in Human-Robot Collaboration 

**Authors**: Andreas Naoum, Parag Khanna, Elmira Yadollahi, Mårten Björkman, Christian Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.09717)  

**Abstract**: This work aims to interpret human behavior to anticipate potential user confusion when a robot provides explanations for failure, allowing the robot to adapt its explanations for more natural and efficient collaboration. Using a dataset that included facial emotion detection, eye gaze estimation, and gestures from 55 participants in a user study, we analyzed how human behavior changed in response to different types of failures and varying explanation levels. Our goal is to assess whether human collaborators are ready to accept less detailed explanations without inducing confusion. We formulate a data-driven predictor to predict human confusion during robot failure explanations. We also propose and evaluate a mechanism, based on the predictor, to adapt the explanation level according to observed human behavior. The promising results from this evaluation indicate the potential of this research in adapting a robot's explanations for failures to enhance the collaborative experience. 

---
# Embodied Chain of Action Reasoning with Multi-Modal Foundation Model for Humanoid Loco-manipulation 

**Authors**: Yu Hao, Geeta Chandra Raju Bethala, Niraj Pudasaini, Hao Huang, Shuaihang Yuan, Congcong Wen, Baoru Huang, Anh Nguyen, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09532)  

**Abstract**: Enabling humanoid robots to autonomously perform loco-manipulation tasks in complex, unstructured environments poses significant challenges. This entails equipping robots with the capability to plan actions over extended horizons while leveraging multi-modality to bridge gaps between high-level planning and actual task execution. Recent advancements in multi-modal foundation models have showcased substantial potential in enhancing planning and reasoning abilities, particularly in the comprehension and processing of semantic information for robotic control tasks. In this paper, we introduce a novel framework based on foundation models that applies the embodied chain of action reasoning methodology to autonomously plan actions from textual instructions for humanoid loco-manipulation. Our method integrates humanoid-specific chain of thought methodology, including detailed affordance and body movement analysis, which provides a breakdown of the task into a sequence of locomotion and manipulation actions. Moreover, we incorporate spatial reasoning based on the observation and target object properties to effectively navigate where target position may be unseen or occluded. Through rigorous experimental setups on object rearrangement, manipulations and loco-manipulation tasks on a real-world environment, we evaluate our method's efficacy on the decoupled upper and lower body control and demonstrate the effectiveness of the chain of robotic action reasoning strategies in comprehending human instructions. 

---
# Fine-tuning an Large Language Model for Automating Computational Fluid Dynamics Simulations 

**Authors**: Zhehao Dong, Zhen Lu, Yue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09602)  

**Abstract**: Configuring computational fluid dynamics (CFD) simulations typically demands extensive domain expertise, limiting broader access. Although large language models (LLMs) have advanced scientific computing, their use in automating CFD workflows is underdeveloped. We introduce a novel approach centered on domain-specific LLM adaptation. By fine-tuning Qwen2.5-7B-Instruct on NL2FOAM, our custom dataset of 28716 natural language-to-OpenFOAM configuration pairs with chain-of-thought (CoT) annotations, we enable direct translation from natural language descriptions to executable CFD setups. A multi-agent framework orchestrates the process, autonomously verifying inputs, generating configurations, running simulations, and correcting errors. Evaluation on a benchmark of 21 diverse flow cases demonstrates state-of-the-art performance, achieving 88.7% solution accuracy and 82.6% first-attempt success rate. This significantly outperforms larger general-purpose models like Qwen2.5-72B-Instruct, DeepSeek-R1, and Llama3.3-70B-Instruct, while also requiring fewer correction iterations and maintaining high computational efficiency. The results highlight the critical role of domain-specific adaptation in deploying LLM assistants for complex engineering workflows. 

---
# AgentDynEx: Nudging the Mechanics and Dynamics of Multi-Agent Simulations 

**Authors**: Jenny Ma, Riya Sahni, Karthik Sreedhar, Lydia B. Chilton  

**Link**: [PDF](https://arxiv.org/pdf/2504.09662)  

**Abstract**: Multi-agent large language model simulations have the potential to model complex human behaviors and interactions. If the mechanics are set up properly, unanticipated and valuable social dynamics can surface. However, it is challenging to consistently enforce simulation mechanics while still allowing for notable and emergent dynamics. We present AgentDynEx, an AI system that helps set up simulations from user-specified mechanics and dynamics. AgentDynEx uses LLMs to guide users through a Configuration Matrix to identify core mechanics and define milestones to track dynamics. It also introduces a method called \textit{nudging}, where the system dynamically reflects on simulation progress and gently intervenes if it begins to deviate from intended outcomes. A technical evaluation found that nudging enables simulations to have more complex mechanics and maintain its notable dynamics compared to simulations without nudging. We discuss the importance of nudging as a technique for balancing mechanics and dynamics of multi-agent simulations. 

---
# Metropolis-Hastings Captioning Game: Knowledge Fusion of Vision Language Models via Decentralized Bayesian Inference 

**Authors**: Yuta Matsui, Ryosuke Yamaki, Ryo Ueda, Seitaro Shinagawa, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.09620)  

**Abstract**: We propose the Metropolis-Hastings Captioning Game (MHCG), a method to fuse knowledge of multiple vision-language models (VLMs) by learning from each other. Although existing methods that combine multiple models suffer from inference costs and architectural constraints, MHCG avoids these problems by performing decentralized Bayesian inference through a process resembling a language game. The knowledge fusion process establishes communication between two VLM agents alternately captioning images and learning from each other. We conduct two image-captioning experiments with two VLMs, each pre-trained on a different dataset. The first experiment demonstrates that MHCG achieves consistent improvement in reference-free evaluation metrics. The second experiment investigates how MHCG contributes to sharing VLMs' category-level vocabulary by observing the occurrence of the vocabulary in the generated captions. 

---
# AirVista-II: An Agentic System for Embodied UAVs Toward Dynamic Scene Semantic Understanding 

**Authors**: Fei Lin, Yonglin Tian, Tengchao Zhang, Jun Huang, Sangtian Guan, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09583)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly important in dynamic environments such as logistics transportation and disaster response. However, current tasks often rely on human operators to monitor aerial videos and make operational decisions. This mode of human-machine collaboration suffers from significant limitations in efficiency and adaptability. In this paper, we present AirVista-II -- an end-to-end agentic system for embodied UAVs, designed to enable general-purpose semantic understanding and reasoning in dynamic scenes. The system integrates agent-based task identification and scheduling, multimodal perception mechanisms, and differentiated keyframe extraction strategies tailored for various temporal scenarios, enabling the efficient capture of critical scene information. Experimental results demonstrate that the proposed system achieves high-quality semantic understanding across diverse UAV-based dynamic scenarios under a zero-shot setting. 

---
# Development of a PPO-Reinforcement Learned Walking Tripedal Soft-Legged Robot using SOFA 

**Authors**: Yomna Mokhtar, Tarek Shohdy, Abdallah A. Hassan, Mostafa Eshra, Omar Elmenawy, Osama Khalil, Haitham El-Hussieny  

**Link**: [PDF](https://arxiv.org/pdf/2504.09242)  

**Abstract**: Rigid robots were extensively researched, whereas soft robotics remains an underexplored field. Utilizing soft-legged robots in performing tasks as a replacement for human beings is an important stride to take, especially under harsh and hazardous conditions over rough terrain environments. For the demand to teach any robot how to behave in different scenarios, a real-time physical and visual simulation is essential. When it comes to soft robots specifically, a simulation framework is still an arduous problem that needs to be disclosed. Using the simulation open framework architecture (SOFA) is an advantageous step. However, neither SOFA's manual nor prior public SOFA projects show its maximum capabilities the users can reach. So, we resolved this by establishing customized settings and handling the framework components appropriately. Settling on perfect, fine-tuned SOFA parameters has stimulated our motivation towards implementing the state-of-the-art (SOTA) reinforcement learning (RL) method of proximal policy optimization (PPO). The final representation is a well-defined, ready-to-deploy walking, tripedal, soft-legged robot based on PPO-RL in a SOFA environment. Robot navigation performance is a key metric to be considered for measuring the success resolution. Although in the simulated soft robots case, an 82\% success rate in reaching a single goal is a groundbreaking output, we pushed the boundaries to further steps by evaluating the progress under assigning a sequence of goals. While trailing the platform steps, outperforming discovery has been observed with an accumulative squared error deviation of 19 mm. The full code is publicly available at \href{this https URL}{this http URL\textunderscore$SOFA$\textunderscore$Soft$\textunderscore$Legged$\textunderscore$ this http URL} 

---
# AGENT: An Aerial Vehicle Generation and Design Tool Using Large Language Models 

**Authors**: Colin Samplawski, Adam D. Cobb, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08981)  

**Abstract**: Computer-aided design (CAD) is a promising application area for emerging artificial intelligence methods. Traditional workflows for cyberphysical systems create detailed digital models which can be evaluated by physics simulators in order to narrow the search space before creating physical prototypes. A major bottleneck of this approach is that the simulators are often computationally expensive and slow. Recent advancements in AI methods offer the possibility to accelerate these pipelines. We use the recently released AircraftVerse dataset, which is especially suited for developing and evaluating large language models for designs. AircraftVerse contains a diverse set of UAV designs represented via textual design trees together with detailed physics simulation results. Following the recent success of large language models (LLMs), we propose AGENT (Aircraft GENeraTor). AGENT is a comprehensive design tool built on the CodeT5+ LLM which learns powerful representations of aircraft textual designs directly from JSON files. We develop a curriculum of training tasks which imbues a single model with a suite of useful features. AGENT is able to generate designs conditioned on properties of flight dynamics (hover time, maximum speed, etc.). Additionally, AGENT can issue evaluations of designs allowing it to act as a surrogate model of the physics simulation that underlies the AircraftVerse dataset. We present a series of experiments which demonstrate our system's abilities. We are able to achieve strong performance using the smallest member of the CodeT5+ family (220M parameters). This allows for a flexible and powerful system which can be executed on a single GPU enabling a clear path toward future deployment. 

---
# MCP Bridge: A Lightweight, LLM-Agnostic RESTful Proxy for Model Context Protocol Servers 

**Authors**: Arash Ahmadi, Sarah Sharif, Yaser M. Banad  

**Link**: [PDF](https://arxiv.org/pdf/2504.08999)  

**Abstract**: Large Language Models (LLMs) are increasingly augmented with external tools through standardized interfaces like the Model Context Protocol (MCP). However, current MCP implementations face critical limitations: they typically require local process execution through STDIO transports, making them impractical for resource-constrained environments like mobile devices, web browsers, and edge computing. We present MCP Bridge, a lightweight RESTful proxy that connects to multiple MCP servers and exposes their capabilities through a unified API. Unlike existing solutions, MCP Bridge is fully LLM-agnostic, supporting any backend regardless of vendor. The system implements a risk-based execution model with three security levels standard execution, confirmation workflow, and Docker isolation while maintaining backward compatibility with standard MCP clients. Complementing this server-side infrastructure is a Python based MCP Gemini Agent that facilitates natural language interaction with MCP tools. The evaluation demonstrates that MCP Bridge successfully addresses the constraints of direct MCP connections while providing enhanced security controls and cross-platform compatibility, enabling sophisticated LLM-powered applications in previously inaccessible environments 

---
# AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories 

**Authors**: Xing Han Lù, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stańczak, Peter Shaw, Christopher J. Pal, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.08942)  

**Abstract**: Web agents enable users to perform tasks on web browsers through natural language interaction. Evaluating web agents trajectories is an important problem, since it helps us determine whether the agent successfully completed the tasks. Rule-based methods are widely used for this purpose, but they are challenging to extend to new tasks and may not always recognize successful trajectories. We may achieve higher accuracy through human evaluation, but the process would be substantially slower and more expensive. Automatic evaluations with LLMs may avoid the challenges of designing new rules and manually annotating trajectories, enabling faster and cost-effective evaluation. However, it is unclear how effective they are at evaluating web agents. To this end, we propose AgentRewardBench, the first benchmark to assess the effectiveness of LLM judges for evaluating web agents. AgentRewardBench contains 1302 trajectories across 5 benchmarks and 4 LLMs. Each trajectory in AgentRewardBench is reviewed by an expert, who answers questions pertaining to the success, side effects, and repetitiveness of the agent. Using our benchmark, we evaluate 12 LLM judges and find that no single LLM excels across all benchmarks. We also find that the rule-based evaluation used by common benchmarks tends to underreport the success rate of web agents, highlighting a key weakness of rule-based evaluation and the need to develop more flexible automatic evaluations. We release the benchmark at: this https URL 

---
# Can LLMs Revolutionize the Design of Explainable and Efficient TinyML Models? 

**Authors**: Christophe El Zeinaty, Wassim Hamidouche, Glenn Herrou, Daniel Menard, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2504.09685)  

**Abstract**: This paper introduces a novel framework for designing efficient neural network architectures specifically tailored to tiny machine learning (TinyML) platforms. By leveraging large language models (LLMs) for neural architecture search (NAS), a vision transformer (ViT)-based knowledge distillation (KD) strategy, and an explainability module, the approach strikes an optimal balance between accuracy, computational efficiency, and memory usage. The LLM-guided search explores a hierarchical search space, refining candidate architectures through Pareto optimization based on accuracy, multiply-accumulate operations (MACs), and memory metrics. The best-performing architectures are further fine-tuned using logits-based KD with a pre-trained ViT-B/16 model, which enhances generalization without increasing model size. Evaluated on the CIFAR-100 dataset and deployed on an STM32H7 microcontroller (MCU), the three proposed models, LMaNet-Elite, LMaNet-Core, and QwNet-Core, achieve accuracy scores of 74.50%, 74.20% and 73.00%, respectively. All three models surpass current state-of-the-art (SOTA) models, such as MCUNet-in3/in4 (69.62% / 72.86%) and XiNet (72.27%), while maintaining a low computational cost of less than 100 million MACs and adhering to the stringent 320 KB static random-access memory (SRAM) constraint. These results demonstrate the efficiency and performance of the proposed framework for TinyML platforms, underscoring the potential of combining LLM-driven search, Pareto optimization, KD, and explainability to develop accurate, efficient, and interpretable models. This approach opens new possibilities in NAS, enabling the design of efficient architectures specifically suited for TinyML. 

---
# Investigating the Treacherous Turn in Deep Reinforcement Learning 

**Authors**: Chace Ashcraft, Kiran Karra, Josh Carney, Nathan Drenkow  

**Link**: [PDF](https://arxiv.org/pdf/2504.08943)  

**Abstract**: The Treacherous Turn refers to the scenario where an artificial intelligence (AI) agent subtly, and perhaps covertly, learns to perform a behavior that benefits itself but is deemed undesirable and potentially harmful to a human supervisor. During training, the agent learns to behave as expected by the human supervisor, but when deployed to perform its task, it performs an alternate behavior without the supervisor there to prevent it. Initial experiments applying DRL to an implementation of the A Link to the Past example do not produce the treacherous turn effect naturally, despite various modifications to the environment intended to produce it. However, in this work, we find the treacherous behavior to be reproducible in a DRL agent when using other trojan injection strategies. This approach deviates from the prototypical treacherous turn behavior since the behavior is explicitly trained into the agent, rather than occurring as an emergent consequence of environmental complexity or poor objective specification. Nonetheless, these experiments provide new insights into the challenges of producing agents capable of true treacherous turn behavior. 

---
# X-Guard: Multilingual Guard Agent for Content Moderation 

**Authors**: Bibek Upadhayay, Vahid Behzadan, Ph.D  

**Link**: [PDF](https://arxiv.org/pdf/2504.08848)  

**Abstract**: Large Language Models (LLMs) have rapidly become integral to numerous applications in critical domains where reliability is paramount. Despite significant advances in safety frameworks and guardrails, current protective measures exhibit crucial vulnerabilities, particularly in multilingual contexts. Existing safety systems remain susceptible to adversarial attacks in low-resource languages and through code-switching techniques, primarily due to their English-centric design. Furthermore, the development of effective multilingual guardrails is constrained by the scarcity of diverse cross-lingual training data. Even recent solutions like Llama Guard-3, while offering multilingual support, lack transparency in their decision-making processes. We address these challenges by introducing X-Guard agent, a transparent multilingual safety agent designed to provide content moderation across diverse linguistic contexts. X-Guard effectively defends against both conventional low-resource language attacks and sophisticated code-switching attacks. Our approach includes: curating and enhancing multiple open-source safety datasets with explicit evaluation rationales; employing a jury of judges methodology to mitigate individual judge LLM provider biases; creating a comprehensive multilingual safety dataset spanning 132 languages with 5 million data points; and developing a two-stage architecture combining a custom-finetuned mBART-50 translation module with an evaluation X-Guard 3B model trained through supervised finetuning and GRPO training. Our empirical evaluations demonstrate X-Guard's effectiveness in detecting unsafe content across multiple languages while maintaining transparency throughout the safety evaluation process. Our work represents a significant advancement in creating robust, transparent, and linguistically inclusive safety systems for LLMs and its integrated systems. 

---
# PriM: Principle-Inspired Material Discovery through Multi-Agent Collaboration 

**Authors**: Zheyuan Lai, Yingming Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08810)  

**Abstract**: Complex chemical space and limited knowledge scope with biases holds immense challenge for human scientists, yet in automated materials discovery. Existing intelligent methods relies more on numerical computation, leading to inefficient exploration and results with hard-interpretability. To bridge this gap, we introduce a principles-guided material discovery system powered by language inferential multi-agent system (MAS), namely PriM. Our framework integrates automated hypothesis generation with experimental validation in a roundtable system of MAS, enabling systematic exploration while maintaining scientific rigor. Based on our framework, the case study of nano helix demonstrates higher materials exploration rate and property value while providing transparent reasoning pathways. This approach develops an automated-and-transparent paradigm for material discovery, with broad implications for rational design of functional materials. Code is publicly available at our \href{this https URL}{GitHub}. 

---
# Towards Personalized Conversational Sales Agents with Contextual User Profiling for Strategic Action 

**Authors**: Tongyoung Kim, Jeongeun Lee, Soojin Yoon, Seonghwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08754)  

**Abstract**: Conversational Recommender Systems (CRSs) aim to engage users in dialogue to provide tailored recommendations. While traditional CRSs focus on eliciting preferences and retrieving items, real-world e-commerce interactions involve more complex decision-making, where users consider multiple factors beyond simple attributes. To bridge this gap, we introduce Conversational Sales (CSales), a novel task that unifies preference elicitation, recommendation, and persuasion to better support user decision-making. For a realistic evaluation of CSales, we present CSUser, an LLM-based user simulator constructed from real-world data, modeling diverse user profiles with needs and personalities. Additionally, we propose CSI, a conversational sales agent that proactively infers contextual profiles through dialogue for personalized action planning. Extensive experiments demonstrate that CSUser effectively replicates real-world users and emphasize the importance of contextual profiling for strategic action selection, ultimately driving successful purchases in e-commerce. 

---
# Simulating Filter Bubble on Short-video Recommender System with Large Language Model Agents 

**Authors**: Nicholas Sukiennik, Haoyu Wang, Zailin Zeng, Chen Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08742)  

**Abstract**: An increasing reliance on recommender systems has led to concerns about the creation of filter bubbles on social media, especially on short video platforms like TikTok. However, their formation is still not entirely understood due to the complex dynamics between recommendation algorithms and user feedback. In this paper, we aim to shed light on these dynamics using a large language model-based simulation framework. Our work employs real-world short-video data containing rich video content information and detailed user-agents to realistically simulate the recommendation-feedback cycle. Through large-scale simulations, we demonstrate that LLMs can replicate real-world user-recommender interactions, uncovering key mechanisms driving filter bubble formation. We identify critical factors, such as demographic features and category attraction that exacerbate content homogenization. To mitigate this, we design and test interventions including various cold-start and feedback weighting strategies, showing measurable reductions in filter bubble effects. Our framework enables rapid prototyping of recommendation strategies, offering actionable solutions to enhance content diversity in real-world systems. Furthermore, we analyze how LLM-inherent biases may propagate through recommendations, proposing safeguards to promote equity for vulnerable groups, such as women and low-income populations. By examining the interplay between recommendation and LLM agents, this work advances a deeper understanding of algorithmic bias and provides practical tools to promote inclusive digital spaces. 

---
# Can We Edit LLMs for Long-Tail Biomedical Knowledge? 

**Authors**: Xinhao Yi, Jake Lever, Kevin Bryson, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10421)  

**Abstract**: Knowledge editing has emerged as an effective approach for updating large language models (LLMs) by modifying their internal knowledge. However, their application to the biomedical domain faces unique challenges due to the long-tailed distribution of biomedical knowledge, where rare and infrequent information is prevalent. In this paper, we conduct the first comprehensive study to investigate the effectiveness of knowledge editing methods for editing long-tail biomedical knowledge. Our results indicate that, while existing editing methods can enhance LLMs' performance on long-tail biomedical knowledge, their performance on long-tail knowledge remains inferior to that on high-frequency popular knowledge, even after editing. Our further analysis reveals that long-tail biomedical knowledge contains a significant amount of one-to-many knowledge, where one subject and relation link to multiple objects. This high prevalence of one-to-many knowledge limits the effectiveness of knowledge editing in improving LLMs' understanding of long-tail biomedical knowledge, highlighting the need for tailored strategies to bridge this performance gap. 

---
# SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users 

**Authors**: Xinnong Zhang, Jiayu Lin, Xinyi Mou, Shiyue Yang, Xiawei Liu, Libo Sun, Hanjia Lyu, Yihang Yang, Weihong Qi, Yue Chen, Guanying Li, Ling Yan, Yao Hu, Siming Chen, Yu Wang, Jingxuan Huang, Jiebo Luo, Shiping Tang, Libo Wu, Baohua Zhou, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.10157)  

**Abstract**: Social simulation is transforming traditional social science research by modeling human behavior through interactions between virtual individuals and their environments. With recent advances in large language models (LLMs), this approach has shown growing potential in capturing individual differences and predicting group behaviors. However, existing methods face alignment challenges related to the environment, target users, interaction mechanisms, and behavioral patterns. To this end, we introduce SocioVerse, an LLM-agent-driven world model for social simulation. Our framework features four powerful alignment components and a user pool of 10 million real individuals. To validate its effectiveness, we conducted large-scale simulation experiments across three distinct domains: politics, news, and economics. Results demonstrate that SocioVerse can reflect large-scale population dynamics while ensuring diversity, credibility, and representativeness through standardized procedures and minimal manual adjustments. 

---
# DataMosaic: Explainable and Verifiable Multi-Modal Data Analytics through Extract-Reason-Verify 

**Authors**: Zhengxuan Zhang, Zhuowen Liang, Yin Wu, Teng Lin, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10036)  

**Abstract**: Large Language Models (LLMs) are transforming data analytics, but their widespread adoption is hindered by two critical limitations: they are not explainable (opaque reasoning processes) and not verifiable (prone to hallucinations and unchecked errors). While retrieval-augmented generation (RAG) improves accuracy by grounding LLMs in external data, it fails to address the core challenges of trustworthy analytics - especially when processing noisy, inconsistent, or multi-modal data (for example, text, tables, images). We propose DataMosaic, a framework designed to make LLM-powered analytics both explainable and verifiable. By dynamically extracting task-specific structures (for example, tables, graphs, trees) from raw data, DataMosaic provides transparent, step-by-step reasoning traces and enables validation of intermediate results. Built on a multi-agent framework, DataMosaic orchestrates self-adaptive agents that align with downstream task requirements, enhancing consistency, completeness, and privacy. Through this approach, DataMosaic not only tackles the limitations of current LLM-powered analytics systems but also lays the groundwork for a new paradigm of grounded, accurate, and explainable multi-modal data analytics. 

---
# UXAgent: A System for Simulating Usability Testing of Web Design with LLM Agents 

**Authors**: Yuxuan Lu, Bingsheng Yao, Hansu Gu, Jing Huang, Jessie Wang, Yang Li, Jiri Gesi, Qi He, Toby Jia-Jun Li, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09407)  

**Abstract**: Usability testing is a fundamental research method that user experience (UX) researchers use to evaluate and iterate a web design, but\textbf{ how to evaluate and iterate the usability testing study design } itself? Recent advances in Large Language Model-simulated Agent (\textbf{LLM Agent}) research inspired us to design \textbf{UXAgent} to support UX researchers in evaluating and reiterating their usability testing study design before they conduct the real human-subject study. Our system features a Persona Generator module, an LLM Agent module, and a Universal Browser Connector module to automatically generate thousands of simulated users to interactively test the target website. The system also provides an Agent Interview Interface and a Video Replay Interface so that the UX researchers can easily review and analyze the generated qualitative and quantitative log data. Through a heuristic evaluation, five UX researcher participants praised the innovation of our system but also expressed concerns about the future of LLM Agent usage in UX studies. 

---
# GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents 

**Authors**: Xiaobo Xia, Run Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.10458)  

**Abstract**: Existing efforts in building Graphical User Interface (GUI) agents largely rely on the training paradigm of supervised fine-tuning on Large Vision-Language Models (LVLMs). However, this approach not only demands extensive amounts of training data but also struggles to effectively understand GUI screenshots and generalize to unseen interfaces. The issue significantly limits its application in real-world scenarios, especially for high-level tasks. Inspired by Reinforcement Fine-Tuning (RFT) in large reasoning models (e.g., DeepSeek-R1), which efficiently enhances the problem-solving capabilities of large language models in real-world settings, we propose \name, the first reinforcement learning framework designed to enhance the GUI capabilities of LVLMs in high-level real-world task scenarios, through unified action space rule modeling. By leveraging a small amount of carefully curated high-quality data across multiple platforms (including Windows, Linux, MacOS, Android, and Web) and employing policy optimization algorithms such as Group Relative Policy Optimization (GRPO) to update the model, \name achieves superior performance using only 0.02\% of the data (3K vs. 13M) compared to previous state-of-the-art methods like OS-Atlas across eight benchmarks spanning three different platforms (mobile, desktop, and web). These results demonstrate the immense potential of reinforcement learning based on unified action space rule modeling in improving the execution capabilities of LVLMs for real-world GUI agent tasks. 

---
# Joint Action Language Modelling for Transparent Policy Execution 

**Authors**: Theodor Wulff, Rahul Singh Maharjan, Xinyun Chi, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10055)  

**Abstract**: An agent's intention often remains hidden behind the black-box nature of embodied policies. Communication using natural language statements that describe the next action can provide transparency towards the agent's behavior. We aim to insert transparent behavior directly into the learning process, by transforming the problem of policy learning into a language generation problem and combining it with traditional autoregressive modelling. The resulting model produces transparent natural language statements followed by tokens representing the specific actions to solve long-horizon tasks in the Language-Table environment. Following previous work, the model is able to learn to produce a policy represented by special discretized tokens in an autoregressive manner. We place special emphasis on investigating the relationship between predicting actions and producing high-quality language for a transparent agent. We find that in many cases both the quality of the action trajectory and the transparent statement increase when they are generated simultaneously. 

---
# AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents 

**Authors**: Dakuo Wang, Ting-Yao Hsu, Yuxuan Lu, Limeng Cui, Yaochen Xie, William Headean, Bingsheng Yao, Akash Veeragouni, Jiapeng Liu, Sreyashi Nag, Jessie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09723)  

**Abstract**: A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents this http URL, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns. 

---
# A Survey of Personalization: From RAG to Agent 

**Authors**: Xiaopeng Li, Pengyue Jia, Derong Xu, Yi Wen, Yingyi Zhang, Wenlin Zhang, Wanyu Wang, Yichao Wang, Zhaocheng Du, Xiangyang Li, Yong Liu, Huifeng Guo, Ruiming Tang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.10147)  

**Abstract**: Personalization has become an essential capability in modern AI systems, enabling customized interactions that align with individual user preferences, contexts, and goals. Recent research has increasingly concentrated on Retrieval-Augmented Generation (RAG) frameworks and their evolution into more advanced agent-based architectures within personalized settings to enhance user satisfaction. Building on this foundation, this survey systematically examines personalization across the three core stages of RAG: pre-retrieval, retrieval, and generation. Beyond RAG, we further extend its capabilities into the realm of Personalized LLM-based Agents, which enhance traditional RAG systems with agentic functionalities, including user understanding, personalized planning and execution, and dynamic generation. For both personalization in RAG and agent-based personalization, we provide formal definitions, conduct a comprehensive review of recent literature, and summarize key datasets and evaluation metrics. Additionally, we discuss fundamental challenges, limitations, and promising research directions in this evolving field. Relevant papers and resources are continuously updated at this https URL. 

---
# Enhancing Product Search Interfaces with Sketch-Guided Diffusion and Language Agents 

**Authors**: Edward Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.08739)  

**Abstract**: The rapid progress in diffusion models, transformers, and language agents has unlocked new possibilities, yet their potential in user interfaces and commercial applications remains underexplored. We present Sketch-Search Agent, a novel framework that transforms the image search experience by integrating a multimodal language agent with freehand sketches as control signals for diffusion models. Using the T2I-Adapter, Sketch-Search Agent combines sketches and text prompts to generate high-quality query images, encoded via a CLIP image encoder for efficient matching against an image corpus. Unlike existing methods, Sketch-Search Agent requires minimal setup, no additional training, and excels in sketch-based image retrieval and natural language interactions. The multimodal agent enhances user experience by dynamically retaining preferences, ranking results, and refining queries for personalized recommendations. This interactive design empowers users to create sketches and receive tailored product suggestions, showcasing the potential of diffusion models in user-centric image retrieval. Experiments confirm Sketch-Search Agent's high accuracy in delivering relevant product search results. 

---
