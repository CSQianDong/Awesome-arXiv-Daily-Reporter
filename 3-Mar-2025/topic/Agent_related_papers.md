# ARIES: Autonomous Reasoning with LLMs on Interactive Thought Graph Environments 

**Authors**: Pedro Gimenes, Zeyu Cao, Jeffrey Wong, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.21208)  

**Abstract**: Recent research has shown that LLM performance on reasoning tasks can be enhanced by scaling test-time compute. One promising approach, particularly with decomposable problems, involves arranging intermediate solutions as a graph on which transformations are performed to explore the solution space. However, prior works rely on pre-determined, task-specific transformation schedules which are subject to a set of searched hyperparameters. In this work, we view thought graph transformations as actions in a Markov decision process, and implement policy agents to drive effective action policies for the underlying reasoning LLM agent. In particular, we investigate the ability for another LLM to act as a policy agent on thought graph environments and introduce ARIES, a multi-agent architecture for reasoning with LLMs. In ARIES, reasoning LLM agents solve decomposed subproblems, while policy LLM agents maintain visibility of the thought graph states, and dynamically adapt the problem-solving strategy. Through extensive experiments, we observe that using off-the-shelf LLMs as policy agents with no supervised fine-tuning (SFT) can yield up to $29\%$ higher accuracy on HumanEval relative to static transformation schedules, as well as reducing inference costs by $35\%$ and avoid any search requirements. We also conduct a thorough analysis of observed failure modes, highlighting that limitations on LLM sizes and the depth of problem decomposition can be seen as challenges to scaling LLM-guided reasoning. 

---
# ProAI: Proactive Multi-Agent Conversational AI with Structured Knowledge Base for Psychiatric Diagnosis 

**Authors**: Yuqi Wu, Guangya Wan, Jingjing Li, Shengming Zhao, Lingfeng Ma, Tianyi Ye, Ion Pop, Yanbo Zhang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20689)  

**Abstract**: Most LLM-driven conversational AI systems operate reactively, responding to user prompts without guiding the interaction. Most LLM-driven conversational AI systems operate reactively, responding to user prompts without guiding the interaction. However, many real-world applications-such as psychiatric diagnosis, consulting, and interviews-require AI to take a proactive role, asking the right questions and steering conversations toward specific objectives. Using mental health differential diagnosis as an application context, we introduce ProAI, a goal-oriented, proactive conversational AI framework. ProAI integrates structured knowledge-guided memory, multi-agent proactive reasoning, and a multi-faceted evaluation strategy, enabling LLMs to engage in clinician-style diagnostic reasoning rather than simple response generation. Through simulated patient interactions, user experience assessment, and professional clinical validation, we demonstrate that ProAI achieves up to 83.3% accuracy in mental disorder differential diagnosis while maintaining professional and empathetic interaction standards. These results highlight the potential for more reliable, adaptive, and goal-driven AI diagnostic assistants, advancing LLMs beyond reactive dialogue systems. 

---
# ReaLJam: Real-Time Human-AI Music Jamming with Reinforcement Learning-Tuned Transformers 

**Authors**: Alexander Scarlatos, Yusong Wu, Ian Simon, Adam Roberts, Tim Cooijmans, Natasha Jaques, Cassie Tarakajian, Cheng-Zhi Anna Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21267)  

**Abstract**: Recent advances in generative artificial intelligence (AI) have created models capable of high-quality musical content generation. However, little consideration is given to how to use these models for real-time or cooperative jamming musical applications because of crucial required features: low latency, the ability to communicate planned actions, and the ability to adapt to user input in real-time. To support these needs, we introduce ReaLJam, an interface and protocol for live musical jamming sessions between a human and a Transformer-based AI agent trained with reinforcement learning. We enable real-time interactions using the concept of anticipation, where the agent continually predicts how the performance will unfold and visually conveys its plan to the user. We conduct a user study where experienced musicians jam in real-time with the agent through ReaLJam. Our results demonstrate that ReaLJam enables enjoyable and musically interesting sessions, and we uncover important takeaways for future work. 

---
# Dynamically Local-Enhancement Planner for Large-Scale Autonomous Driving 

**Authors**: Nanshan Deng, Weitao Zhou, Bo Zhang, Junze Wen, Kun Jiang, Zhong Cao, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21134)  

**Abstract**: Current autonomous vehicles operate primarily within limited regions, but there is increasing demand for broader applications. However, as models scale, their limited capacity becomes a significant challenge for adapting to novel scenarios. It is increasingly difficult to improve models for new situations using a single monolithic model. To address this issue, we introduce the concept of dynamically enhancing a basic driving planner with local driving data, without permanently modifying the planner itself. This approach, termed the Dynamically Local-Enhancement (DLE) Planner, aims to improve the scalability of autonomous driving systems without significantly expanding the planner's size. Our approach introduces a position-varying Markov Decision Process formulation coupled with a graph neural network that extracts region-specific driving features from local observation data. The learned features describe the local behavior of the surrounding objects, which is then leveraged to enhance a basic reinforcement learning-based policy. We evaluated our approach in multiple scenarios and compared it with a one-for-all driving model. The results show that our method outperforms the baseline policy in both safety (collision rate) and average reward, while maintaining a lighter scale. This approach has the potential to benefit large-scale autonomous vehicles without the need for largely expanding on-device driving models. 

---
# PASemiQA: Plan-Assisted Agent for Question Answering on Semi-Structured Data with Text and Relational Information 

**Authors**: Hansi Yang, Qi Zhang, Wei Jiang, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.21087)  

**Abstract**: Large language models (LLMs) have shown impressive abilities in answering questions across various domains, but they often encounter hallucination issues on questions that require professional and up-to-date knowledge. To address this limitation, retrieval-augmented generation (RAG) techniques have been proposed, which retrieve relevant information from external sources to inform their responses. However, existing RAG methods typically focus on a single type of external data, such as vectorized text database or knowledge graphs, and cannot well handle real-world questions on semi-structured data containing both text and relational information. To bridge this gap, we introduce PASemiQA, a novel approach that jointly leverages text and relational information in semi-structured data to answer questions. PASemiQA first generates a plan to identify relevant text and relational information to answer the question in semi-structured data, and then uses an LLM agent to traverse the semi-structured data and extract necessary information. Our empirical results demonstrate the effectiveness of PASemiQA across different semi-structured datasets from various domains, showcasing its potential to improve the accuracy and reliability of question answering systems on semi-structured data. 

---
# Reward Learning from Multiple Feedback Types 

**Authors**: Yannick Metz, András Geiszl, Raphaël Baur, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2502.21038)  

**Abstract**: Learning rewards from preference feedback has become an important tool in the alignment of agentic models. Preference-based feedback, often implemented as a binary comparison between multiple completions, is an established method to acquire large-scale human feedback. However, human feedback in other contexts is often much more diverse. Such diverse feedback can better support the goals of a human annotator, and the simultaneous use of multiple sources might be mutually informative for the learning process or carry type-dependent biases for the reward learning process. Despite these potential benefits, learning from different feedback types has yet to be explored extensively. In this paper, we bridge this gap by enabling experimentation and evaluating multi-type feedback in a broad set of environments. We present a process to generate high-quality simulated feedback of six different types. Then, we implement reward models and downstream RL training for all six feedback types. Based on the simulated feedback, we investigate the use of types of feedback across ten RL environments and compare them to pure preference-based baselines. We show empirically that diverse types of feedback can be utilized and lead to strong reward modeling performance. This work is the first strong indicator of the potential of multi-type feedback for RLHF. 

---
# Subtask-Aware Visual Reward Learning from Segmented Demonstrations 

**Authors**: Changyeon Kim, Minho Heo, Doohyun Lee, Jinwoo Shin, Honglak Lee, Joseph J. Lim, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.20630)  

**Abstract**: Reinforcement Learning (RL) agents have demonstrated their potential across various robotic tasks. However, they still heavily rely on human-engineered reward functions, requiring extensive trial-and-error and access to target behavior information, often unavailable in real-world settings. This paper introduces REDS: REward learning from Demonstration with Segmentations, a novel reward learning framework that leverages action-free videos with minimal supervision. Specifically, REDS employs video demonstrations segmented into subtasks from diverse sources and treats these segments as ground-truth rewards. We train a dense reward function conditioned on video segments and their corresponding subtasks to ensure alignment with ground-truth reward signals by minimizing the Equivalent-Policy Invariant Comparison distance. Additionally, we employ contrastive learning objectives to align video representations with subtasks, ensuring precise subtask inference during online interactions. Our experiments show that REDS significantly outperforms baseline methods on complex robotic manipulation tasks in Meta-World and more challenging real-world tasks, such as furniture assembly in FurnitureBench, with minimal human intervention. Moreover, REDS facilitates generalization to unseen tasks and robot embodiments, highlighting its potential for scalable deployment in diverse environments. 

---
# Among Them: A game-based framework for assessing persuasion capabilities of LLMs 

**Authors**: Mateusz Idziejczak, Vasyl Korzavatykh, Mateusz Stawicki, Andrii Chmutov, Marcin Korcz, Iwo Błądek, Dariusz Brzezinski  

**Link**: [PDF](https://arxiv.org/pdf/2502.20426)  

**Abstract**: The proliferation of large language models (LLMs) and autonomous AI agents has raised concerns about their potential for automated persuasion and social influence. While existing research has explored isolated instances of LLM-based manipulation, systematic evaluations of persuasion capabilities across different models remain limited. In this paper, we present an Among Us-inspired game framework for assessing LLM deception skills in a controlled environment. The proposed framework makes it possible to compare LLM models by game statistics, as well as quantify in-game manipulation according to 25 persuasion strategies from social psychology and rhetoric. Experiments between 8 popular language models of different types and sizes demonstrate that all tested models exhibit persuasive capabilities, successfully employing 22 of the 25 anticipated techniques. We also find that larger models do not provide any persuasion advantage over smaller models and that longer model outputs are negatively correlated with the number of games won. Our study provides insights into the deception capabilities of LLMs, as well as tools and data for fostering future research on the topic. 

---
# The Power of Personality: A Human Simulation Perspective to Investigate Large Language Model Agents 

**Authors**: Yifan Duan, Yihong Tang, Xuefeng Bai, Kehai Chen, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20859)  

**Abstract**: Large language models (LLMs) excel in both closed tasks (including problem-solving, and code generation) and open tasks (including creative writing), yet existing explanations for their capabilities lack connections to real-world human intelligence. To fill this gap, this paper systematically investigates LLM intelligence through the lens of ``human simulation'', addressing three core questions: (1) How do personality traits affect problem-solving in closed tasks? (2) How do traits shape creativity in open tasks? (3) How does single-agent performance influence multi-agent collaboration? By assigning Big Five personality traits to LLM agents and evaluating their performance in single- and multi-agent settings, we reveal that specific traits significantly influence reasoning accuracy (closed tasks) and creative output (open tasks). Furthermore, multi-agent systems exhibit collective intelligence distinct from individual capabilities, driven by distinguishing combinations of personalities. We demonstrate that LLMs inherently simulate human behavior through next-token prediction, mirroring human language, decision-making, and collaborative dynamics. 

---
# The Rise of Darkness: Safety-Utility Trade-Offs in Role-Playing Dialogue Agents 

**Authors**: Yihong Tang, Kehai Chen, Xuefeng Bai, Zhengyu Niu, Bo Wang, Jie Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20757)  

**Abstract**: Large Language Models (LLMs) have made remarkable advances in role-playing dialogue agents, demonstrating their utility in character simulations. However, it remains challenging for these agents to balance character portrayal utility with content safety because this essential character simulation often comes with the risk of generating unsafe content. To address this issue, we first conduct a systematic exploration of the safety-utility trade-off across multiple LLMs. Our analysis reveals that risk scenarios created by villain characters and user queries (referred to as risk coupling) contribute to this trade-off. Building on this, we propose a novel Adaptive Dynamic Multi-Preference (ADMP) method, which dynamically adjusts safety-utility preferences based on the degree of risk coupling and guides the model to generate responses biased toward utility or safety. We further introduce Coupling Margin Sampling (CMS) into coupling detection to enhance the model's ability to handle high-risk scenarios. Experimental results demonstrate that our approach improves safety metrics while maintaining utility. 

---
# Supervised Fine-Tuning LLMs to Behave as Pedagogical Agents in Programming Education 

**Authors**: Emily Ross, Yuval Kansal, Jake Renzella, Alexandra Vassar, Andrew Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.20527)  

**Abstract**: Large language models (LLMs) are increasingly being explored in higher education, yet their effectiveness as teaching agents remains underexamined. In this paper, we present the development of GuideLM, a fine-tuned LLM designed for programming education. GuideLM has been integrated into the Debugging C Compiler (DCC), an educational C compiler that leverages LLMs to generate pedagogically sound error explanations. Previously, DCC relied on off-the-shelf OpenAI models, which, while accurate, often over-assisted students by directly providing solutions despite contrary prompting.
To address this, we employed supervised fine-tuning (SFT) on a dataset of 528 student-question/teacher-answer pairs, creating two models: GuideLM and GuideLM-mini, fine-tuned on ChatGPT-4o and 4o-mini, respectively. We conducted an expert analysis of 400 responses per model, comparing their pedagogical effectiveness against base OpenAI models. Our evaluation, grounded in constructivism and cognitive load theory, assessed factors such as conceptual scaffolding, clarity, and Socratic guidance.
Results indicate that GuideLM and GuideLM-mini improve pedagogical performance, with an 8% increase in Socratic guidance and a 58% improvement in economy of words compared to GPT-4o. However, this refinement comes at the cost of a slight reduction in general accuracy. While further work is needed, our findings suggest that fine-tuning LLMs with targeted datasets is a promising approach for developing models better suited to educational contexts. 

---
# Visual Reasoning at Urban Intersections: FineTuning GPT-4o for Traffic Conflict Detection 

**Authors**: Sari Masri, Huthaifa I. Ashqar, Mohammed Elhenawy  

**Link**: [PDF](https://arxiv.org/pdf/2502.20573)  

**Abstract**: Traffic control in unsignalized urban intersections presents significant challenges due to the complexity, frequent conflicts, and blind spots. This study explores the capability of leveraging Multimodal Large Language Models (MLLMs), such as GPT-4o, to provide logical and visual reasoning by directly using birds-eye-view videos of four-legged intersections. In this proposed method, GPT-4o acts as intelligent system to detect conflicts and provide explanations and recommendations for the drivers. The fine-tuned model achieved an accuracy of 77.14%, while the manual evaluation of the true predicted values of the fine-tuned GPT-4o showed significant achievements of 89.9% accuracy for model-generated explanations and 92.3% for the recommended next actions. These results highlight the feasibility of using MLLMs for real-time traffic management using videos as inputs, offering scalable and actionable insights into intersections traffic management and operation. Code used in this study is available at this https URL. 

---
