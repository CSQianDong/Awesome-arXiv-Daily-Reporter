# GEMA-Score: Granular Explainable Multi-Agent Score for Radiology Report Evaluation 

**Authors**: Zhenxuan Zhang, Kinhei Lee, Weihang Deng, Huichi Zhou, Zihao Jin, Jiahao Huang, Zhifan Gao, Dominic C Marshall, Yingying Fang, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05347)  

**Abstract**: Automatic medical report generation supports clinical diagnosis, reduces the workload of radiologists, and holds the promise of improving diagnosis consistency. However, existing evaluation metrics primarily assess the accuracy of key medical information coverage in generated reports compared to human-written reports, while overlooking crucial details such as the location and certainty of reported abnormalities. These limitations hinder the comprehensive assessment of the reliability of generated reports and pose risks in their selection for clinical use. Therefore, we propose a Granular Explainable Multi-Agent Score (GEMA-Score) in this paper, which conducts both objective quantification and subjective evaluation through a large language model-based multi-agent workflow. Our GEMA-Score parses structured reports and employs NER-F1 calculations through interactive exchanges of information among agents to assess disease diagnosis, location, severity, and uncertainty. Additionally, an LLM-based scoring agent evaluates completeness, readability, and clinical terminology while providing explanatory feedback. Extensive experiments validate that GEMA-Score achieves the highest correlation with human expert evaluations on a public dataset, demonstrating its effectiveness in clinical scoring (Kendall coefficient = 0.70 for Rexval dataset and Kendall coefficient = 0.54 for RadEvalX dataset). The anonymous project demo is available at: this https URL. 

---
# VQEL: Enabling Self-Developed Symbolic Language in Agents through Vector Quantization in Emergent Language Games 

**Authors**: Mohammad Mahdi Samiei Paqaleh, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2503.04940)  

**Abstract**: In the field of emergent language, efforts have traditionally focused on developing communication protocols through interactions between agents in referential games. However, the aspect of internal language learning, where language serves not only as a communicative tool with others but also as a means for individual thinking, self-reflection, and problem-solving remains underexplored. Developing a language through self-play, without another agent's involvement, poses a unique challenge. It requires an agent to craft symbolic representations and train them using direct gradient methods. The challenge here is that if an agent attempts to learn symbolic representations through self-play using conventional modeling and techniques such as REINFORCE, the solution will offer no advantage over previous multi-agent approaches. We introduce VQEL, a novel method that incorporates Vector Quantization into the agents' architecture, enabling them to autonomously invent and develop discrete symbolic representations in a self-play referential game. Following the self-play phase, agents can enhance their language through reinforcement learning and interactions with other agents in the mutual-play phase. Our experiments across various datasets demonstrate that VQEL not only outperforms the traditional REINFORCE method but also benefits from improved control and reduced susceptibility to collapse, thanks to the incorporation of vector quantization. 

---
# Cite Before You Speak: Enhancing Context-Response Grounding in E-commerce Conversational LLM-Agents 

**Authors**: Jingying Zeng, Hui Liu, Zhenwei Dai, Xianfeng Tang, Chen Luo, Samarth Varshney, Zhen Li, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2503.04830)  

**Abstract**: With the advancement of conversational large language models (LLMs), several LLM-based Conversational Shopping Agents (CSA) have been developed to help customers answer questions and smooth their shopping journey in e-commerce domain. The primary objective in building a trustworthy CSA is to ensure the agent's responses are accurate and factually grounded, which is essential for building customer trust and encouraging continuous engagement. However, two challenges remain. First, LLMs produce hallucinated or unsupported claims. Such inaccuracies risk spreading misinformation and diminishing customer trust. Second, without providing knowledge source attribution in CSA response, customers struggle to verify LLM-generated information. To address these challenges, we present an easily productionized solution that enables a "citation experience" utilizing In-context Learning (ICL) and Multi-UX-Inference (MUI) to generate responses with citations to attribute its original sources without interfering other existing UX features. With proper UX design, these citation marks can be linked to the related product information and display the source to our customers. In this work, we also build auto-metrics and scalable benchmarks to holistically evaluate LLM's grounding and attribution capabilities. Our experiments demonstrate that incorporating this citation generation paradigm can substantially enhance the grounding of LLM responses by 13.83% on the real-world data. As such, our solution not only addresses the immediate challenges of LLM grounding issues but also adds transparency to conversational AI. 

---
# Towards Anthropomorphic Conversational AI Part I: A Practical Framework 

**Authors**: Fei Wei, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.04787)  

**Abstract**: Large language models (LLMs), due to their advanced natural language capabilities, have seen significant success in applications where the user interface is usually a conversational artificial intelligence (AI) agent and engages the user through multi-round conversations. However, many scenarios require the agents to exhibit stronger social and conversational intelligence and demonstrate more human-like (anthropomorphic) reactions. This is an aspect that foundational LLMs have yet to fully address such that a single call of foundational models might be insufficient.
To bridge this gap, we propose a two-stage solution. In this work, we focus on the first stage, introducing a multi-module framework designed to replicate the key aspects of human intelligence involved in conversations. This framework comprises thinking modules for reasoning, resource modules for managing knowledge and external information, and response modules for generating contextually appropriate interactions. With all the modules cooperating, the framework would empower the agents to provide a better human-like conversation experience. In the second stage of our approach, these conversational data, after filtering and labeling, can serve as training and testing data for reinforcement learning, enabling AI to better capture human preferences. This stage is left for future work.
In our experiments, volunteers engaged in over 3000 rounds of conversation with the same AI character powered by a standalone LLM and our framework which integrates the same LLM. A separate group of evaluators rated the conversation samples, revealing that our framework significantly enhanced the social and conversational intelligence, even without fine-tuning the LLM. 

---
# DiMA: An LLM-Powered Ride-Hailing Assistant at DiDi 

**Authors**: Yansong Ning, Shuowei Cai, Wei Li, Jun Fang, Naiqiang Tan, Hua Chai, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04768)  

**Abstract**: On-demand ride-hailing services like DiDi, Uber, and Lyft have transformed urban transportation, offering unmatched convenience and flexibility. In this paper, we introduce DiMA, an LLM-powered ride-hailing assistant deployed in DiDi Chuxing. Its goal is to provide seamless ride-hailing services and beyond through a natural and efficient conversational interface under dynamic and complex spatiotemporal urban contexts. To achieve this, we propose a spatiotemporal-aware order planning module that leverages external tools for precise spatiotemporal reasoning and progressive order planning. Additionally, we develop a cost-effective dialogue system that integrates multi-type dialog repliers with cost-aware LLM configurations to handle diverse conversation goals and trade-off response quality and latency. Furthermore, we introduce a continual fine-tuning scheme that utilizes real-world interactions and simulated dialogues to align the assistant's behavior with human preferred decision-making processes. Since its deployment in the DiDi application, DiMA has demonstrated exceptional performance, achieving 93% accuracy in order planning and 92% in response generation during real-world interactions. Offline experiments further validate DiMA capabilities, showing improvements of up to 70.23% in order planning and 321.27% in response generation compared to three state-of-the-art agent frameworks, while reducing latency by $0.72\times$ to $5.47\times$. These results establish DiMA as an effective, efficient, and intelligent mobile assistant for ride-hailing services. 

---
# Multi Agent based Medical Assistant for Edge Devices 

**Authors**: Sakharam Gawade, Shivam Akhouri, Chinmay Kulkarni, Jagdish Samant, Pragya Sahu, Aastik, Jai Pahal, Saswat Meher  

**Link**: [PDF](https://arxiv.org/pdf/2503.05397)  

**Abstract**: Large Action Models (LAMs) have revolutionized intelligent automation, but their application in healthcare faces challenges due to privacy concerns, latency, and dependency on internet access. This report introduces an ondevice, multi-agent healthcare assistant that overcomes these limitations. The system utilizes smaller, task-specific agents to optimize resources, ensure scalability and high performance. Our proposed system acts as a one-stop solution for health care needs with features like appointment booking, health monitoring, medication reminders, and daily health reporting. Powered by the Qwen Code Instruct 2.5 7B model, the Planner and Caller Agents achieve an average RougeL score of 85.5 for planning and 96.5 for calling for our tasks while being lightweight for on-device deployment. This innovative approach combines the benefits of ondevice systems with multi-agent architectures, paving the way for user-centric healthcare solutions. 

---
# R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning 

**Authors**: Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05592)  

**Abstract**: Existing Large Reasoning Models (LRMs) have shown the potential of reinforcement learning (RL) to enhance the complex reasoning capabilities of Large Language Models~(LLMs). While they achieve remarkable performance on challenging tasks such as mathematics and coding, they often rely on their internal knowledge to solve problems, which can be inadequate for time-sensitive or knowledge-intensive questions, leading to inaccuracies and hallucinations. To address this, we propose \textbf{R1-Searcher}, a novel two-stage outcome-based RL approach designed to enhance the search capabilities of LLMs. This method allows LLMs to autonomously invoke external search systems to access additional knowledge during the reasoning process. Our framework relies exclusively on RL, without requiring process rewards or distillation for a cold start. % effectively generalizing to out-of-domain datasets and supporting both Base and Instruct models. Our experiments demonstrate that our method significantly outperforms previous strong RAG methods, even when compared to the closed-source GPT-4o-mini. 

---
# Provably Correct Automata Embeddings for Optimal Automata-Conditioned Reinforcement Learning 

**Authors**: Beyazit Yalcinkaya, Niklas Lauffer, Marcell Vazquez-Chanlatte, Sanjit A. Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2503.05042)  

**Abstract**: Automata-conditioned reinforcement learning (RL) has given promising results for learning multi-task policies capable of performing temporally extended objectives given at runtime, done by pretraining and freezing automata embeddings prior to training the downstream policy. However, no theoretical guarantees were given. This work provides a theoretical framework for the automata-conditioned RL problem and shows that it is probably approximately correct learnable. We then present a technique for learning provably correct automata embeddings, guaranteeing optimal multi-task policy learning. Our experimental evaluation confirms these theoretical results. 

---
# SafeArena: Evaluating the Safety of Autonomous Web Agents 

**Authors**: Ada Defne Tur, Nicholas Meade, Xing Han Lù, Alejandra Zambrano, Arkil Patel, Esin Durmus, Spandana Gella, Karolina Stańczak, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.04957)  

**Abstract**: LLM-based agents are becoming increasingly proficient at solving web-based tasks. With this capability comes a greater risk of misuse for malicious purposes, such as posting misinformation in an online forum or selling illicit substances on a website. To evaluate these risks, we propose SafeArena, the first benchmark to focus on the deliberate misuse of web agents. SafeArena comprises 250 safe and 250 harmful tasks across four websites. We classify the harmful tasks into five harm categories -- misinformation, illegal activity, harassment, cybercrime, and social bias, designed to assess realistic misuses of web agents. We evaluate leading LLM-based web agents, including GPT-4o, Claude-3.5 Sonnet, Qwen-2-VL 72B, and Llama-3.2 90B, on our benchmark. To systematically assess their susceptibility to harmful tasks, we introduce the Agent Risk Assessment framework that categorizes agent behavior across four risk levels. We find agents are surprisingly compliant with malicious requests, with GPT-4o and Qwen-2 completing 34.7% and 27.3% of harmful requests, respectively. Our findings highlight the urgent need for safety alignment procedures for web agents. Our benchmark is available here: this https URL 

---
# A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval 

**Authors**: Yu Zhang, Shutong Qiao, Jiaqi Zhang, Tzu-Heng Lin, Chen Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05659)  

**Abstract**: Information technology has profoundly altered the way humans interact with information. The vast amount of content created, shared, and disseminated online has made it increasingly difficult to access relevant information. Over the past two decades, search and recommendation systems (collectively referred to as information retrieval systems) have evolved significantly to address these challenges. Recent advances in large language models (LLMs) have demonstrated capabilities that surpass human performance in various language-related tasks and exhibit general understanding, reasoning, and decision-making abilities. This paper explores the transformative potential of large language model agents in enhancing search and recommendation systems. We discuss the motivations and roles of LLM agents, and establish a classification framework to elaborate on the existing research. We highlight the immense potential of LLM agents in addressing current challenges in search and recommendation, providing insights into future research directions. This paper is the first to systematically review and classify the research on LLM agents in these domains, offering a novel perspective on leveraging this advanced AI technology for information retrieval. To help understand the existing works, we list the existing papers on agent-based simulation with large language models at this link: this https URL. 

---
# FedMABench: Benchmarking Mobile Agents on Decentralized Heterogeneous User Data 

**Authors**: Wenhao Wang, Zijie Yu, Rui Ye, Jianqing Zhang, Siheng Chen, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05143)  

**Abstract**: Mobile agents have attracted tremendous research participation recently. Traditional approaches to mobile agent training rely on centralized data collection, leading to high cost and limited scalability. Distributed training utilizing federated learning offers an alternative by harnessing real-world user data, providing scalability and reducing costs. However, pivotal challenges, including the absence of standardized benchmarks, hinder progress in this field.
To tackle the challenges, we introduce FedMABench, the first benchmark for federated training and evaluation of mobile agents, specifically designed for heterogeneous scenarios. FedMABench features 6 datasets with 30+ subsets, 8 federated algorithms, 10+ base models, and over 800 apps across 5 categories, providing a comprehensive framework for evaluating mobile agents across diverse environments. Through extensive experiments, we uncover several key insights: federated algorithms consistently outperform local training; the distribution of specific apps plays a crucial role in heterogeneity; and, even apps from distinct categories can exhibit correlations during training. FedMABench is publicly available at: this https URL with the datasets at: this https URL. 

---
# INTENT: Trajectory Prediction Framework with Intention-Guided Contrastive Clustering 

**Authors**: Yihong Tang, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04952)  

**Abstract**: Accurate trajectory prediction of road agents (e.g., pedestrians, vehicles) is an essential prerequisite for various intelligent systems applications, such as autonomous driving and robotic navigation. Recent research highlights the importance of environmental contexts (e.g., maps) and the "multi-modality" of trajectories, leading to increasingly complex model structures. However, real-world deployments require lightweight models that can quickly migrate and adapt to new environments. Additionally, the core motivations of road agents, referred to as their intentions, deserves further exploration. In this study, we advocate that understanding and reasoning road agents' intention plays a key role in trajectory prediction tasks, and the main challenge is that the concept of intention is fuzzy and abstract. To this end, we present INTENT, an efficient intention-guided trajectory prediction model that relies solely on information contained in the road agent's trajectory. Our model distinguishes itself from existing models in several key aspects: (i) We explicitly model road agents' intentions through contrastive clustering, accommodating the fuzziness and abstraction of human intention in their trajectories. (ii) The proposed INTENT is based solely on multi-layer perceptrons (MLPs), resulting in reduced training and inference time, making it very efficient and more suitable for real-world deployment. (iii) By leveraging estimated intentions and an innovative algorithm for transforming trajectory observations, we obtain more robust trajectory representations that lead to superior prediction accuracy. Extensive experiments on real-world trajectory datasets for pedestrians and autonomous vehicles demonstrate the effectiveness and efficiency of INTENT. 

---
# VLMs Play StarCraft II: A Benchmark and Multimodal Decision Method 

**Authors**: Weiyu Ma, Yuqian Fu, Zecheng Zhang, Guohao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05383)  

**Abstract**: We introduce VLM-Attention, a multimodal StarCraft II environment that aligns artificial agent perception with the human gameplay experience. Traditional frameworks such as SMAC rely on abstract state representations that diverge significantly from human perception, limiting the ecological validity of agent behavior. Our environment addresses this limitation by incorporating RGB visual inputs and natural language observations that more closely simulate human cognitive processes during gameplay. The VLM-Attention framework consists of three integrated components: (1) a vision-language model enhanced with specialized self-attention mechanisms for strategic unit targeting and battlefield assessment, (2) a retrieval-augmented generation system that leverages domain-specific StarCraft II knowledge to inform tactical decisions, and (3) a dynamic role-based task distribution system that enables coordinated multi-agent behavior. Our experimental evaluation across 21 custom scenarios demonstrates that VLM-based agents powered by foundation models (specifically Qwen-VL and GPT-4o) can execute complex tactical maneuvers without explicit training, achieving comparable performance to traditional MARL methods that require substantial training iterations. This work establishes a foundation for developing human-aligned StarCraft II agents and advances the broader research agenda of multimodal game AI. Our implementation is available at this https URL. 

---
# The Society of HiveMind: Multi-Agent Optimization of Foundation Model Swarms to Unlock the Potential of Collective Intelligence 

**Authors**: Noah Mamie, Susie Xi Rao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05473)  

**Abstract**: Multi-agent systems address issues of accessibility and scalability of artificial intelligence (AI) foundation models, which are often represented by large language models. We develop a framework - the "Society of HiveMind" (SOHM) - that orchestrates the interaction between multiple AI foundation models, imitating the observed behavior of animal swarms in nature by following modern evolutionary theories. On the one hand, we find that the SOHM provides a negligible benefit on tasks that mainly require real-world knowledge. On the other hand, we remark a significant improvement on tasks that require intensive logical reasoning, indicating that multi-agent systems are capable of increasing the reasoning capabilities of the collective compared to the individual agents. Our findings demonstrate the potential of combining a multitude of diverse AI foundation models to form an artificial swarm intelligence capable of self-improvement through interactions with a given environment. 

---
# InDRiVE: Intrinsic Disagreement based Reinforcement for Vehicle Exploration through Curiosity Driven Generalized World Model 

**Authors**: Feeza Khan Khanzada, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2503.05573)  

**Abstract**: Model-based Reinforcement Learning (MBRL) has emerged as a promising paradigm for autonomous driving, where data efficiency and robustness are critical. Yet, existing solutions often rely on carefully crafted, task specific extrinsic rewards, limiting generalization to new tasks or environments. In this paper, we propose InDRiVE (Intrinsic Disagreement based Reinforcement for Vehicle Exploration), a method that leverages purely intrinsic, disagreement based rewards within a Dreamer based MBRL framework. By training an ensemble of world models, the agent actively explores high uncertainty regions of environments without any task specific feedback. This approach yields a task agnostic latent representation, allowing for rapid zero shot or few shot fine tuning on downstream driving tasks such as lane following and collision avoidance. Experimental results in both seen and unseen environments demonstrate that InDRiVE achieves higher success rates and fewer infractions compared to DreamerV2 and DreamerV3 baselines despite using significantly fewer training steps. Our findings highlight the effectiveness of purely intrinsic exploration for learning robust vehicle control behaviors, paving the way for more scalable and adaptable autonomous driving systems. 

---
# Evidential Uncertainty Estimation for Multi-Modal Trajectory Prediction 

**Authors**: Sajad Marvi, Christoph Rist, Julian Schmidt, Julian Jordan, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2503.05274)  

**Abstract**: Accurate trajectory prediction is crucial for autonomous driving, yet uncertainty in agent behavior and perception noise makes it inherently challenging. While multi-modal trajectory prediction models generate multiple plausible future paths with associated probabilities, effectively quantifying uncertainty remains an open problem. In this work, we propose a novel multi-modal trajectory prediction approach based on evidential deep learning that estimates both positional and mode probability uncertainty in real time. Our approach leverages a Normal Inverse Gamma distribution for positional uncertainty and a Dirichlet distribution for mode uncertainty. Unlike sampling-based methods, it infers both types of uncertainty in a single forward pass, significantly improving efficiency. Additionally, we experimented with uncertainty-driven importance sampling to improve training efficiency by prioritizing underrepresented high-uncertainty samples over redundant ones. We perform extensive evaluations of our method on the Argoverse 1 and Argoverse 2 datasets, demonstrating that it provides reliable uncertainty estimates while maintaining high trajectory prediction accuracy. 

---
# Discrete Contrastive Learning for Diffusion Policies in Autonomous Driving 

**Authors**: Kalle Kujanpää, Daulet Baimukashev, Farzeen Munir, Shoaib Azam, Tomasz Piotr Kucner, Joni Pajarinen, Ville Kyrki  

**Link**: [PDF](https://arxiv.org/pdf/2503.05229)  

**Abstract**: Learning to perform accurate and rich simulations of human driving behaviors from data for autonomous vehicle testing remains challenging due to human driving styles' high diversity and variance. We address this challenge by proposing a novel approach that leverages contrastive learning to extract a dictionary of driving styles from pre-existing human driving data. We discretize these styles with quantization, and the styles are used to learn a conditional diffusion policy for simulating human drivers. Our empirical evaluation confirms that the behaviors generated by our approach are both safer and more human-like than those of the machine-learning-based baseline methods. We believe this has the potential to enable higher realism and more effective techniques for evaluating and improving the performance of autonomous vehicles. 

---
# Multi-Task Reinforcement Learning Enables Parameter Scaling 

**Authors**: Reginald McLean, Evangelos Chataroulas, Jordan Terry, Isaac Woungang, Nariman Farsad, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2503.05126)  

**Abstract**: Multi-task reinforcement learning (MTRL) aims to endow a single agent with the ability to perform well on multiple tasks. Recent works have focused on developing novel sophisticated architectures to improve performance, often resulting in larger models; it is unclear, however, whether the performance gains are a consequence of the architecture design itself or the extra parameters. We argue that gains are mostly due to scale by demonstrating that naively scaling up a simple MTRL baseline to match parameter counts outperforms the more sophisticated architectures, and these gains benefit most from scaling the critic over the actor. Additionally, we explore the training stability advantages that come with task diversity, demonstrating that increasing the number of tasks can help mitigate plasticity loss. Our findings suggest that MTRL's simultaneous training across multiple tasks provides a natural framework for beneficial parameter scaling in reinforcement learning, challenging the need for complex architectural innovations. 

---
# A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation 

**Authors**: Shanhe You, Xuewen Luo, Xinhe Liang, Jiashu Yu, Chen Zheng, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.05164)  

**Abstract**: Evaluation methods for autonomous driving are crucial for algorithm optimization. However, due to the complexity of driving intelligence, there is currently no comprehensive evaluation method for the level of autonomous driving intelligence. In this paper, we propose an evaluation framework for driving behavior intelligence in complex traffic environments, aiming to fill this gap. We constructed a natural language evaluation dataset of human professional drivers and passengers through naturalistic driving experiments and post-driving behavior evaluation interviews. Based on this dataset, we developed an LLM-powered driving evaluation framework. The effectiveness of this framework was validated through simulated experiments in the CARLA urban traffic simulator and further corroborated by human assessment. Our research provides valuable insights for evaluating and designing more intelligent, human-like autonomous driving agents. The implementation details of the framework and detailed information about the dataset can be found at Github. 

---
# Reward-Centered ReST-MCTS: A Robust Decision-Making Framework for Robotic Manipulation in High Uncertainty Environments 

**Authors**: Xibai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05226)  

**Abstract**: Monte Carlo Tree Search (MCTS) has emerged as a powerful tool for decision-making in robotics, enabling efficient exploration of large search spaces. However, traditional MCTS methods struggle in environments characterized by high uncertainty and noisy data due to their reliance on final-step reward evaluation. The lack of intermediate feedback during search often results in suboptimal decision-making and computational inefficiencies.
This paper introduces Reward-Centered ReST-MCTS, a novel framework that enhances MCTS by incorporating intermediate reward shaping. The core of our approach is the Rewarding Center, which refines search trajectories by dynamically assigning partial rewards using rule-based validation, heuristic guidance, and neural estimation. By integrating these mechanisms, our method enables real-time optimization of search paths, mitigating the effects of error propagation.
We evaluate Reward-Centered ReST-MCTS in robotic manipulation tasks under high uncertainty, demonstrating consistent improvements in decision accuracy. Compared to baseline methods, including Chain-of-Thought (CoT) prompting and Vanilla ReST-MCTS, our framework achieves a 2-4% accuracy improvement while maintaining computational feasibility. Ablation studies confirm the effectiveness of intermediate feedback in search refinement, particularly in pruning incorrect decision paths early. Furthermore, robustness tests show that our method retains high performance across varying levels of uncertainty. 

---
# Data-Efficient Learning from Human Interventions for Mobile Robots 

**Authors**: Zhenghao Peng, Zhizheng Liu, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04969)  

**Abstract**: Mobile robots are essential in applications such as autonomous delivery and hospitality services. Applying learning-based methods to address mobile robot tasks has gained popularity due to its robustness and generalizability. Traditional methods such as Imitation Learning (IL) and Reinforcement Learning (RL) offer adaptability but require large datasets, carefully crafted reward functions, and face sim-to-real gaps, making them challenging for efficient and safe real-world deployment. We propose an online human-in-the-loop learning method PVP4Real that combines IL and RL to address these issues. PVP4Real enables efficient real-time policy learning from online human intervention and demonstration, without reward or any pretraining, significantly improving data efficiency and training safety. We validate our method by training two different robots -- a legged quadruped, and a wheeled delivery robot -- in two mobile robot tasks, one of which even uses raw RGBD image as observation. The training finishes within 15 minutes. Our experiments show the promising future of human-in-the-loop learning in addressing the data efficiency issue in real-world robotic tasks. More information is available at: this https URL 

---
# Curiosity-Driven Imagination: Discovering Plan Operators and Learning Associated Policies for Open-World Adaptation 

**Authors**: Pierrick Lorang, Hong Lu, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04931)  

**Abstract**: Adapting quickly to dynamic, uncertain environments-often called "open worlds"-remains a major challenge in robotics. Traditional Task and Motion Planning (TAMP) approaches struggle to cope with unforeseen changes, are data-inefficient when adapting, and do not leverage world models during learning. We address this issue with a hybrid planning and learning system that integrates two models: a low level neural network based model that learns stochastic transitions and drives exploration via an Intrinsic Curiosity Module (ICM), and a high level symbolic planning model that captures abstract transitions using operators, enabling the agent to plan in an "imaginary" space and generate reward machines. Our evaluation in a robotic manipulation domain with sequential novelty injections demonstrates that our approach converges faster and outperforms state-of-the-art hybrid methods. 

---
# Agentic AI and the Cyber Arms Race 

**Authors**: Sean Oesch, Jack Hutchins, Phillipe Austria, Amul Chaulagain  

**Link**: [PDF](https://arxiv.org/pdf/2503.04760)  

**Abstract**: Agentic AI is shifting the cybersecurity landscape as attackers and defenders leverage AI agents to augment humans and automate common tasks. In this article, we examine the implications for cyber warfare and global politics as Agentic AI becomes more powerful and enables the broad proliferation of capabilities only available to the most well resourced actors today. 

---
# Position: AI agents should be regulated based on autonomous action sequences 

**Authors**: Takauki Osogami  

**Link**: [PDF](https://arxiv.org/pdf/2503.04750)  

**Abstract**: This position paper argues that AI agents should be regulated based on the sequence of actions they autonomously take. AI agents with long-term planning and strategic capabilities can pose significant risks of human extinction and irreversible global catastrophes. While existing regulations often focus on computational scale as a proxy for potential harm, we contend that such measures are insufficient for assessing the risks posed by AI agents whose capabilities arise primarily from inference-time computation. To support our position, we discuss relevant regulations and recommendations from AI scientists regarding existential risks, as well as the advantages of action sequences over existing impact measures that require observing environmental states. 

---
# Static Vs. Agentic Game Master AI for Facilitating Solo Role-Playing Experiences 

**Authors**: Nicolai Hejlesen Jørgensen, Sarmilan Tharmabalan, Ilhan Aslan, Nicolai Brodersen Hansen, Timothy Merritt  

**Link**: [PDF](https://arxiv.org/pdf/2502.19519)  

**Abstract**: This paper presents a game master AI for single-player role-playing games. The AI is designed to deliver interactive text-based narratives and experiences typically associated with multiplayer tabletop games like Dungeons & Dragons. We report on the design process and the series of experiments to improve the functionality and experience design, resulting in two functional versions of the system. While v1 of our system uses simplified prompt engineering, v2 leverages a multi-agent architecture and the ReAct framework to include reasoning and action. A comparative evaluation demonstrates that v2 as an agentic system maintains play while significantly improving modularity and game experience, including immersion and curiosity. Our findings contribute to the evolution of AI-driven interactive fiction, highlighting new avenues for enhancing solo role-playing experiences. 

---
