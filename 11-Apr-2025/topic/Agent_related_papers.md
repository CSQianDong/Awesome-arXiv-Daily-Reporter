# Are AI Agents interacting with Online Ads? 

**Authors**: Andreas Stöckl, Joel Nitu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07112)  

**Abstract**: As AI-driven agents become increasingly integrated into the digital ecosystem, they reshape how online advertising is perceived and processed. Particularly in the travel and hotel booking sector, these autonomous systems influence the effectiveness of traditional advertising formats. While visual cues and emotional appeals sway human users, AI agents prioritize structured data such as price, availability, and specifications. This study examines how different AI agents interact with online advertising, whether they incorporate ads into their decision-making processes, and which ad formats prove most effective. We analyze interaction patterns, click behavior, and decision-making strategies through experiments with multimodal language models such as OpenAI GPT-4o, Anthropic Claude, and Google Gemini 2.0 Flash. Our findings reveal that AI agents neither ignore nor systematically avoid advertisements but instead favor certain features-particularly keywords and structured data. These insights have significant implications for the future design of advertising strategies in AI-dominated digital environments. 

---
# CollEX -- A Multimodal Agentic RAG System Enabling Interactive Exploration of Scientific Collections 

**Authors**: Florian Schneider, Narges Baba Ahmadi, Niloufar Baba Ahmadi, Iris Vogel, Martin Semmann, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.07643)  

**Abstract**: In this paper, we introduce CollEx, an innovative multimodal agentic Retrieval-Augmented Generation (RAG) system designed to enhance interactive exploration of extensive scientific collections. Given the overwhelming volume and inherent complexity of scientific collections, conventional search systems often lack necessary intuitiveness and interactivity, presenting substantial barriers for learners, educators, and researchers. CollEx addresses these limitations by employing state-of-the-art Large Vision-Language Models (LVLMs) as multimodal agents accessible through an intuitive chat interface. By abstracting complex interactions via specialized agents equipped with advanced tools, CollEx facilitates curiosity-driven exploration, significantly simplifying access to diverse scientific collections and records therein. Our system integrates textual and visual modalities, supporting educational scenarios that are helpful for teachers, pupils, students, and researchers by fostering independent exploration as well as scientific excitement and curiosity. Furthermore, CollEx serves the research community by discovering interdisciplinary connections and complementing visual data. We illustrate the effectiveness of our system through a proof-of-concept application containing over 64,000 unique records across 32 collections from a local scientific collection from a public university. 

---
# MOSAIC: Modeling Social AI for Content Dissemination and Regulation in Multi-Agent Simulations 

**Authors**: Genglin Liu, Salman Rahman, Elisa Kreiss, Marzyeh Ghassemi, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07830)  

**Abstract**: We present a novel, open-source social network simulation framework, MOSAIC, where generative language agents predict user behaviors such as liking, sharing, and flagging content. This simulation combines LLM agents with a directed social graph to analyze emergent deception behaviors and gain a better understanding of how users determine the veracity of online social content. By constructing user representations from diverse fine-grained personas, our system enables multi-agent simulations that model content dissemination and engagement dynamics at scale. Within this framework, we evaluate three different content moderation strategies with simulated misinformation dissemination, and we find that they not only mitigate the spread of non-factual content but also increase user engagement. In addition, we analyze the trajectories of popular content in our simulations, and explore whether simulation agents' articulated reasoning for their social interactions truly aligns with their collective engagement patterns. We open-source our simulation software to encourage further research within AI and social sciences. 

---
# MRD-RAG: Enhancing Medical Diagnosis with Multi-Round Retrieval-Augmented Generation 

**Authors**: Yixiang Chen, Penglei Sun, Xiang Li, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07724)  

**Abstract**: In recent years, accurately and quickly deploying medical large language models (LLMs) has become a significant trend. Among these, retrieval-augmented generation (RAG) has garnered significant attention due to its features of rapid deployment and privacy protection. However, existing medical RAG frameworks still have shortcomings. Most existing medical RAG frameworks are designed for single-round question answering tasks and are not suitable for multi-round diagnostic dialogue. On the other hand, existing medical multi-round RAG frameworks do not consider the interconnections between potential diseases to inquire precisely like a doctor. To address these issues, we propose a Multi-Round Diagnostic RAG (MRD-RAG) framework that mimics the doctor's diagnostic process. This RAG framework can analyze diagnosis information of potential diseases and accurately conduct multi-round diagnosis like a doctor. To evaluate the effectiveness of our proposed frameworks, we conduct experiments on two modern medical datasets and two traditional Chinese medicine datasets, with evaluations by GPT and human doctors on different methods. The results indicate that our RAG framework can significantly enhance the diagnostic performance of LLMs, highlighting the potential of our approach in medical diagnosis. The code and data can be found in our project website this https URL. 

---
# AgentAda: Skill-Adaptive Data Analytics for Tailored Insight Discovery 

**Authors**: Amirhossein Abaskohi, Amrutha Varshini Ramesh, Shailesh Nanisetty, Chirag Goel, David Vazquez, Christopher Pal, Spandana Gella, Giuseppe Carenini, Issam H. Laradji  

**Link**: [PDF](https://arxiv.org/pdf/2504.07421)  

**Abstract**: We introduce AgentAda, the first LLM-powered analytics agent that can learn and use new analytics skills to extract more specialized insights. Unlike existing methods that require users to manually decide which data analytics method to apply, AgentAda automatically identifies the skill needed from a library of analytical skills to perform the analysis. This also allows AgentAda to use skills that existing LLMs cannot perform out of the box. The library covers a range of methods, including clustering, predictive modeling, and NLP techniques like BERT, which allow AgentAda to handle complex analytics tasks based on what the user needs. AgentAda's dataset-to-insight extraction strategy consists of three key steps: (I) a question generator to generate queries relevant to the user's goal and persona, (II) a hybrid Retrieval-Augmented Generation (RAG)-based skill matcher to choose the best data analytics skill from the skill library, and (III) a code generator that produces executable code based on the retrieved skill's documentation to extract key patterns. We also introduce KaggleBench, a benchmark of curated notebooks across diverse domains, to evaluate AgentAda's performance. We conducted a human evaluation demonstrating that AgentAda provides more insightful analytics than existing tools, with 48.78% of evaluators preferring its analyses, compared to 27.67% for the unskilled agent. We also propose a novel LLM-as-a-judge approach that we show is aligned with human evaluation as a way to automate insight quality evaluation at larger scale. 

---
# TALE: A Tool-Augmented Framework for Reference-Free Evaluation of Large Language Models 

**Authors**: Sher Badshah, Ali Emami, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2504.07385)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world, autonomous applications, relying on static, pre-annotated references for evaluation poses significant challenges in cost, scalability, and completeness. We propose Tool-Augmented LLM Evaluation (TALE), a framework to assess LLM outputs without predetermined ground-truth answers. Unlike conventional metrics that compare to fixed references or depend solely on LLM-as-a-judge knowledge, TALE employs an agent with tool-access capabilities that actively retrieves and synthesizes external evidence. It iteratively generates web queries, collects information, summarizes findings, and refines subsequent searches through reflection. By shifting away from static references, TALE aligns with free-form question-answering tasks common in real-world scenarios. Experimental results on multiple free-form QA benchmarks show that TALE not only outperforms standard reference-based metrics for measuring response accuracy but also achieves substantial to near-perfect agreement with human evaluations. TALE enhances the reliability of LLM evaluations in real-world, dynamic scenarios without relying on static references. 

---
# Dual Engines of Thoughts: A Depth-Breadth Integration Framework for Open-Ended Analysis 

**Authors**: Fei-Hsuan Yu, Yun-Cheng Chou, Teng-Ruei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07872)  

**Abstract**: We propose the Dual Engines of Thoughts (DEoT), an analytical framework for comprehensive open-ended reasoning. While traditional reasoning frameworks primarily focus on finding "the best answer" or "the correct answer" for single-answer problems, DEoT is specifically designed for "open-ended questions," enabling both broader and deeper analytical exploration. The framework centers on three key components: a Base Prompter for refining user queries, a Solver Agent that orchestrates task decomposition, execution, and validation, and a Dual-Engine System consisting of a Breadth Engine (to explore diverse impact factors) and a Depth Engine (to perform deep investigations). This integrated design allows DEoT to balance wide-ranging coverage with in-depth analysis, and it is highly customizable, enabling users to adjust analytical parameters and tool configurations based on specific requirements. Experimental results show that DEoT excels in addressing complex, multi-faceted questions, achieving a total win rate of 77-86% compared to existing reasoning models, thus highlighting its effectiveness in real-world applications. 

---
# Deceptive Automated Interpretability: Language Models Coordinating to Fool Oversight Systems 

**Authors**: Simon Lermen, Mateusz Dziemian, Natalia Pérez-Campanero Antolín  

**Link**: [PDF](https://arxiv.org/pdf/2504.07831)  

**Abstract**: We demonstrate how AI agents can coordinate to deceive oversight systems using automated interpretability of neural networks. Using sparse autoencoders (SAEs) as our experimental framework, we show that language models (Llama, DeepSeek R1, and Claude 3.7 Sonnet) can generate deceptive explanations that evade detection. Our agents employ steganographic methods to hide information in seemingly innocent explanations, successfully fooling oversight models while achieving explanation quality comparable to reference labels. We further find that models can scheme to develop deceptive strategies when they believe the detection of harmful features might lead to negative consequences for themselves. All tested LLM agents were capable of deceiving the overseer while achieving high interpretability scores comparable to those of reference labels. We conclude by proposing mitigation strategies, emphasizing the critical need for robust understanding and defenses against deception. 

---
# R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents 

**Authors**: Naman Jain, Jaskirat Singh, Manish Shetty, Liang Zheng, Koushik Sen, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2504.07164)  

**Abstract**: Improving open-source models on real-world SWE tasks (solving GITHUB issues) faces two key challenges: 1) scalable curation of execution environments to train these models, and, 2) optimal scaling of test-time compute. We introduce AgentGym, the largest procedurally-curated executable gym environment for training real-world SWE-agents, consisting of more than 8.7K tasks. AgentGym is powered by two main contributions: 1) SYNGEN: a synthetic data curation recipe that enables scalable curation of executable environments using test-generation and back-translation directly from commits, thereby reducing reliance on human-written issues or unit tests. We show that this enables more scalable training leading to pass@1 performance of 34.4% on SWE-Bench Verified benchmark with our 32B model. 2) Hybrid Test-time Scaling: we provide an in-depth analysis of two test-time scaling axes; execution-based and execution-free verifiers, demonstrating that they exhibit complementary strengths and limitations. Test-based verifiers suffer from low distinguishability, while execution-free verifiers are biased and often rely on stylistic features. Surprisingly, we find that while each approach individually saturates around 42-43%, significantly higher gains can be obtained by leveraging their complementary strengths. Overall, our approach achieves 51% on the SWE-Bench Verified benchmark, reflecting a new state-of-the-art for open-weight SWE-agents and for the first time showing competitive performance with proprietary models such as o1, o1-preview and sonnet-3.5-v2 (with tools). We will open-source our environments, models, and agent trajectories. 

---
# Synthesizing High-Quality Programming Tasks with LLM-based Expert and Student Agents 

**Authors**: Manh Hung Nguyen, Victor-Alexandru Pădurean, Alkis Gotovos, Sebastian Tschiatschek, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2504.07655)  

**Abstract**: Generative AI is transforming computing education by enabling the automatic generation of personalized content and feedback. We investigate its capabilities in providing high-quality programming tasks to students. Despite promising advancements in task generation, a quality gap remains between AI-generated and expert-created tasks. The AI-generated tasks may not align with target programming concepts, could be incomprehensible for students to solve, or may contain critical issues such as incorrect tests. Existing works often require interventions from human teachers for validation. We address these challenges by introducing PyTaskSyn, a novel synthesis technique that first generates a programming task and then decides whether it meets certain quality criteria to be given to students. The key idea is to break this process into multiple stages performed by expert and student agents simulated using both strong and weaker generative models. Through extensive evaluation, we show that PyTaskSyn significantly improves task quality compared to baseline techniques and showcases the importance of each specialized agent type in our validation pipeline. Additionally, we conducted user studies using our publicly available web application and show that PyTaskSyn can deliver high-quality programming tasks comparable to expert-designed ones while reducing workload and costs, and being more engaging than programming tasks that are available in online resources. 

---
# Enhancing Player Enjoyment with a Two-Tier DRL and LLM-Based Agent System for Fighting Games 

**Authors**: Shouren Wang, Zehua Jiang, Fernando Sliva, Sam Earle, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2504.07425)  

**Abstract**: Deep reinforcement learning (DRL) has effectively enhanced gameplay experiences and game design across various game genres. However, few studies on fighting game agents have focused explicitly on enhancing player enjoyment, a critical factor for both developers and players. To address this gap and establish a practical baseline for designing enjoyability-focused agents, we propose a two-tier agent (TTA) system and conducted experiments in the classic fighting game Street Fighter II. The first tier of TTA employs a task-oriented network architecture, modularized reward functions, and hybrid training to produce diverse and skilled DRL agents. In the second tier of TTA, a Large Language Model Hyper-Agent, leveraging players' playing data and feedback, dynamically selects suitable DRL opponents. In addition, we investigate and model several key factors that affect the enjoyability of the opponent. The experiments demonstrate improvements from 64. 36% to 156. 36% in the execution of advanced skills over baseline methods. The trained agents also exhibit distinct game-playing styles. Additionally, we conducted a small-scale user study, and the overall enjoyment in the player's feedback validates the effectiveness of our TTA system. 

---
# Enhanced Question-Answering for Skill-based learning using Knowledge-based AI and Generative AI 

**Authors**: Rahul K. Dass, Rochan H. Madhusudhana, Erin C. Deye, Shashank Verma, Timothy A. Bydlon, Grace Brazil, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07463)  

**Abstract**: Supporting learners' understanding of taught skills in online settings is a longstanding challenge. While exercises and chat-based agents can evaluate understanding in limited contexts, this challenge is magnified when learners seek explanations that delve into procedural knowledge (how things are done) and reasoning (why things happen). We hypothesize that an intelligent agent's ability to understand and explain learners' questions about skills can be significantly enhanced using the TMK (Task-Method-Knowledge) model, a Knowledge-based AI framework. We introduce Ivy, an intelligent agent that leverages an LLM and iterative refinement techniques to generate explanations that embody teleological, causal, and compositional principles. Our initial evaluation demonstrates that this approach goes beyond the typical shallow responses produced by an agent with access to unstructured text, thereby substantially improving the depth and relevance of feedback. This can potentially ensure learners develop a comprehensive understanding of skills crucial for effective problem-solving in online environments. 

---
# Better Decisions through the Right Causal World Model 

**Authors**: Elisabeth Dillies, Quentin Delfosse, Jannis Blüml, Raban Emunds, Florian Peter Busch, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2504.07257)  

**Abstract**: Reinforcement learning (RL) agents have shown remarkable performances in various environments, where they can discover effective policies directly from sensory inputs. However, these agents often exploit spurious correlations in the training data, resulting in brittle behaviours that fail to generalize to new or slightly modified environments. To address this, we introduce the Causal Object-centric Model Extraction Tool (COMET), a novel algorithm designed to learn the exact interpretable causal world models (CWMs). COMET first extracts object-centric state descriptions from observations and identifies the environment's internal states related to the depicted objects' properties. Using symbolic regression, it models object-centric transitions and derives causal relationships governing object dynamics. COMET further incorporates large language models (LLMs) for semantic inference, annotating causal variables to enhance interpretability.
By leveraging these capabilities, COMET constructs CWMs that align with the true causal structure of the environment, enabling agents to focus on task-relevant features. The extracted CWMs mitigate the danger of shortcuts, permitting the development of RL systems capable of better planning and decision-making across dynamic scenarios. Our results, validated in Atari environments such as Pong and Freeway, demonstrate the accuracy and robustness of COMET, highlighting its potential to bridge the gap between object-centric reasoning and causal inference in reinforcement learning. 

---
# Fast Adaptation with Behavioral Foundation Models 

**Authors**: Harshit Sikchi, Andrea Tirinzoni, Ahmed Touati, Yingchen Xu, Anssi Kanervisto, Scott Niekum, Amy Zhang, Alessandro Lazaric, Matteo Pirotta  

**Link**: [PDF](https://arxiv.org/pdf/2504.07896)  

**Abstract**: Unsupervised zero-shot reinforcement learning (RL) has emerged as a powerful paradigm for pretraining behavioral foundation models (BFMs), enabling agents to solve a wide range of downstream tasks specified via reward functions in a zero-shot fashion, i.e., without additional test-time learning or planning. This is achieved by learning self-supervised task embeddings alongside corresponding near-optimal behaviors and incorporating an inference procedure to directly retrieve the latent task embedding and associated policy for any given reward function. Despite promising results, zero-shot policies are often suboptimal due to errors induced by the unsupervised training process, the embedding, and the inference procedure. In this paper, we focus on devising fast adaptation strategies to improve the zero-shot performance of BFMs in a few steps of online interaction with the environment while avoiding any performance drop during the adaptation process. Notably, we demonstrate that existing BFMs learn a set of skills containing more performant policies than those identified by their inference procedure, making them well-suited for fast adaptation. Motivated by this observation, we propose both actor-critic and actor-only fast adaptation strategies that search in the low-dimensional task-embedding space of the pre-trained BFM to rapidly improve the performance of its zero-shot policies on any downstream task. Notably, our approach mitigates the initial "unlearning" phase commonly observed when fine-tuning pre-trained RL models. We evaluate our fast adaptation strategies on top of four state-of-the-art zero-shot RL methods in multiple navigation and locomotion domains. Our results show that they achieve 10-40% improvement over their zero-shot performance in a few tens of episodes, outperforming existing baselines. 

---
# Learning Long Short-Term Intention within Human Daily Behaviors 

**Authors**: Zhe Sun, Rujie Wu, Xiaodong Yang, Hongzhao Xie, Haiyan Jiang, Junda Bi, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07597)  

**Abstract**: In the domain of autonomous household robots, it is of utmost importance for robots to understand human behaviors and provide appropriate services. This requires the robots to possess the capability to analyze complex human behaviors and predict the true intentions of humans. Traditionally, humans are perceived as flawless, with their decisions acting as the standards that robots should strive to align with. However, this raises a pertinent question: What if humans make mistakes? In this research, we present a unique task, termed "long short-term intention prediction". This task requires robots can predict the long-term intention of humans, which aligns with human values, and the short term intention of humans, which reflects the immediate action intention. Meanwhile, the robots need to detect the potential non-consistency between the short-term and long-term intentions, and provide necessary warnings and suggestions. To facilitate this task, we propose a long short-term intention model to represent the complex intention states, and build a dataset to train this intention model. Then we propose a two-stage method to integrate the intention model for robots: i) predicting human intentions of both value-based long-term intentions and action-based short-term intentions; and 2) analyzing the consistency between the long-term and short-term intentions. Experimental results indicate that the proposed long short-term intention model can assist robots in comprehending human behavioral patterns over both long-term and short-term durations, which helps determine the consistency between long-term and short-term intentions of humans. 

---
# Automating quantum feature map design via large language models 

**Authors**: Kenya Sakka, Kosuke Mitarai, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2504.07396)  

**Abstract**: Quantum feature maps are a key component of quantum machine learning, encoding classical data into quantum states to exploit the expressive power of high-dimensional Hilbert spaces. Despite their theoretical promise, designing quantum feature maps that offer practical advantages over classical methods remains an open challenge. In this work, we propose an agentic system that autonomously generates, evaluates, and refines quantum feature maps using large language models. The system consists of five component: Generation, Storage, Validation, Evaluation, and Review. Using these components, it iteratively improves quantum feature maps. Experiments on the MNIST dataset show that it can successfully discover and refine feature maps without human intervention. The best feature map generated outperforms existing quantum baselines and achieves competitive accuracy compared to classical kernels across MNIST, Fashion-MNIST, and CIFAR-10. Our approach provides a framework for exploring dataset-adaptive quantum features and highlights the potential of LLM-driven automation in quantum algorithm design. 

---
# Modeling Response Consistency in Multi-Agent LLM Systems: A Comparative Analysis of Shared and Separate Context Approaches 

**Authors**: Tooraj Helmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.07303)  

**Abstract**: Large Language Models (LLMs) are increasingly utilized in multi-agent systems (MAS) to enhance collaborative problem-solving and interactive reasoning. Recent advancements have enabled LLMs to function as autonomous agents capable of understanding complex interactions across multiple topics. However, deploying LLMs in MAS introduces challenges related to context management, response consistency, and scalability, especially when agents must operate under memory limitations and handle noisy inputs. While prior research has explored optimizing context sharing and response latency in LLM-driven MAS, these efforts often focus on either fully centralized or decentralized configurations, each with distinct trade-offs.
In this paper, we develop a probabilistic framework to analyze the impact of shared versus separate context configurations on response consistency and response times in LLM-based MAS. We introduce the Response Consistency Index (RCI) as a metric to evaluate the effects of context limitations, noise, and inter-agent dependencies on system performance. Our approach differs from existing research by focusing on the interplay between memory constraints and noise management, providing insights into optimizing scalability and response times in environments with interdependent topics. Through this analysis, we offer a comprehensive understanding of how different configurations impact the efficiency of LLM-driven multi-agent systems, thereby guiding the design of more robust architectures. 

---
