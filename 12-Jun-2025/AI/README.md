# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning 

**Authors**: Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas  

**Link**: [PDF](https://arxiv.org/pdf/2506.09985)  

**Abstract**: A major challenge for modern AI is to learn to understand the world and learn to act largely by observation. This paper explores a self-supervised approach that combines internet-scale video data with a small amount of interaction data (robot trajectories), to develop models capable of understanding, predicting, and planning in the physical world. We first pre-train an action-free joint-embedding-predictive architecture, V-JEPA 2, on a video and image dataset comprising over 1 million hours of internet video. V-JEPA 2 achieves strong performance on motion understanding (77.3 top-1 accuracy on Something-Something v2) and state-of-the-art performance on human action anticipation (39.7 recall-at-5 on Epic-Kitchens-100) surpassing previous task-specific models. Additionally, after aligning V-JEPA 2 with a large language model, we demonstrate state-of-the-art performance on multiple video question-answering tasks at the 8 billion parameter scale (e.g., 84.0 on PerceptionTest, 76.9 on TempCompass). Finally, we show how self-supervised learning can be applied to robotic planning tasks by post-training a latent action-conditioned world model, V-JEPA 2-AC, using less than 62 hours of unlabeled robot videos from the Droid dataset. We deploy V-JEPA 2-AC zero-shot on Franka arms in two different labs and enable picking and placing of objects using planning with image goals. Notably, this is achieved without collecting any data from the robots in these environments, and without any task-specific training or reward. This work demonstrates how self-supervised learning from web-scale data and a small amount of robot interaction data can yield a world model capable of planning in the physical world. 

---
# How Do People Revise Inconsistent Beliefs? Examining Belief Revision in Humans with User Studies 

**Authors**: Stylianos Loukas Vasileiou, Antonio Rago, Maria Vanina Martinez, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2506.09977)  

**Abstract**: Understanding how humans revise their beliefs in light of new information is crucial for developing AI systems which can effectively model, and thus align with, human reasoning. While theoretical belief revision frameworks rely on a set of principles that establish how these operations are performed, empirical evidence from cognitive psychology suggests that people may follow different patterns when presented with conflicting information. In this paper, we present three comprehensive user studies showing that people consistently prefer explanation-based revisions, i.e., those which are guided by explanations, that result in changes to their belief systems that are not necessarily captured by classical belief change theory. Our experiments systematically investigate how people revise their beliefs with explanations for inconsistencies, whether they are provided with them or left to formulate them themselves, demonstrating a robust preference for what may seem non-minimal revisions across different types of scenarios. These findings have implications for AI systems designed to model human reasoning or interact with humans, suggesting that such systems should accommodate explanation-based, potentially non-minimal belief revision operators to better align with human cognitive processes. 

---
# Intent Factored Generation: Unleashing the Diversity in Your Language Model 

**Authors**: Eltayeb Ahmed, Uljad Berdica, Martha Elliott, Danijela Horak, Jakob N. Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2506.09659)  

**Abstract**: Obtaining multiple meaningfully diverse, high quality samples from Large Language Models for a fixed prompt remains an open challenge. Current methods for increasing diversity often only operate at the token-level, paraphrasing the same response. This is problematic because it leads to poor exploration on reasoning problems and to unengaging, repetitive conversational agents. To address this we propose Intent Factored Generation (IFG), factorising the sampling process into two stages. First, we sample a semantically dense intent, e.g., a summary or keywords. Second, we sample the final response conditioning on both the original prompt and the intent from the first stage. This allows us to use a higher temperature during the intent step to promote conceptual diversity, and a lower temperature during the final generation to ensure the outputs are coherent and self-consistent. Additionally, we find that prompting the model to explicitly state its intent for each step of the chain-of-thought before generating the step is beneficial for reasoning tasks. We demonstrate our method's effectiveness across a diverse set of tasks. We show this method improves both pass@k and Reinforcement Learning from Verifier Feedback on maths and code tasks. For instruction-tuning, we combine IFG with Direct Preference Optimisation to increase conversational diversity without sacrificing reward. Finally, we achieve higher diversity while maintaining the quality of generations on a general language modelling task, using a new dataset of reader comments and news articles that we collect and open-source. In summary, we present a simple method of increasing the sample diversity of LLMs while maintaining performance. This method can be implemented by changing the prompt and varying the temperature during generation, making it easy to integrate into many algorithms for gains across various applications. 

---
# Application-Driven Value Alignment in Agentic AI Systems: Survey and Perspectives 

**Authors**: Wei Zeng, Hengshu Zhu, Chuan Qin, Han Wu, Yihang Cheng, Sirui Zhang, Xiaowei Jin, Yinuo Shen, Zhenxing Wang, Feimin Zhong, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.09656)  

**Abstract**: The ongoing evolution of AI paradigms has propelled AI research into the Agentic AI stage. Consequently, the focus of research has shifted from single agents and simple applications towards multi-agent autonomous decision-making and task collaboration in complex environments. As Large Language Models (LLMs) advance, their applications become more diverse and complex, leading to increasingly situational and systemic risks. This has brought significant attention to value alignment for AI agents, which aims to ensure that an agent's goals, preferences, and behaviors align with human values and societal norms. This paper reviews value alignment in agent systems within specific application scenarios. It integrates the advancements in AI driven by large models with the demands of social governance. Our review covers value principles, agent system application scenarios, and agent value alignment evaluation. Specifically, value principles are organized hierarchically from a top-down perspective, encompassing macro, meso, and micro levels. Agent system application scenarios are categorized and reviewed from a general-to-specific viewpoint. Agent value alignment evaluation systematically examines datasets for value alignment assessment and relevant value alignment methods. Additionally, we delve into value coordination among multiple agents within agent systems. Finally, we propose several potential research directions in this field. 

---
# DipLLM: Fine-Tuning LLM for Strategic Decision-making in Diplomacy 

**Authors**: Kaixuan Xu, Jiajun Chai, Sicheng Li, Yuqian Fu, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09655)  

**Abstract**: Diplomacy is a complex multiplayer game that requires both cooperation and competition, posing significant challenges for AI systems. Traditional methods rely on equilibrium search to generate extensive game data for training, which demands substantial computational resources. Large Language Models (LLMs) offer a promising alternative, leveraging pre-trained knowledge to achieve strong performance with relatively small-scale fine-tuning. However, applying LLMs to Diplomacy remains challenging due to the exponential growth of possible action combinations and the intricate strategic interactions among players. To address this challenge, we propose DipLLM, a fine-tuned LLM-based agent that learns equilibrium policies for Diplomacy. DipLLM employs an autoregressive factorization framework to simplify the complex task of multi-unit action assignment into a sequence of unit-level decisions. By defining an equilibrium policy within this framework as the learning objective, we fine-tune the model using only 1.5% of the data required by the state-of-the-art Cicero model, surpassing its performance. Our results demonstrate the potential of fine-tuned LLMs for tackling complex strategic decision-making in multiplayer games. 

---
# Fast Monte Carlo Tree Diffusion: 100x Speedup via Parallel Sparse Planning 

**Authors**: Jaesik Yoon, Hyeonseo Cho, Yoshua Bengio, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2506.09498)  

**Abstract**: Diffusion models have recently emerged as a powerful approach for trajectory planning. However, their inherently non-sequential nature limits their effectiveness in long-horizon reasoning tasks at test time. The recently proposed Monte Carlo Tree Diffusion (MCTD) offers a promising solution by combining diffusion with tree-based search, achieving state-of-the-art performance on complex planning problems. Despite its strengths, our analysis shows that MCTD incurs substantial computational overhead due to the sequential nature of tree search and the cost of iterative denoising. To address this, we propose Fast-MCTD, a more efficient variant that preserves the strengths of MCTD while significantly improving its speed and scalability. Fast-MCTD integrates two techniques: Parallel MCTD, which enables parallel rollouts via delayed tree updates and redundancy-aware selection; and Sparse MCTD, which reduces rollout length through trajectory coarsening. Experiments show that Fast-MCTD achieves up to 100x speedup over standard MCTD while maintaining or improving planning performance. Remarkably, it even outperforms Diffuser in inference speed on some tasks, despite Diffuser requiring no search and yielding weaker solutions. These results position Fast-MCTD as a practical and scalable solution for diffusion-based inference-time reasoning. 

---
# A Call for Collaborative Intelligence: Why Human-Agent Systems Should Precede AI Autonomy 

**Authors**: Henry Peng Zou, Wei-Chieh Huang, Yaozu Wu, Chunyu Miao, Dongyuan Li, Aiwei Liu, Yue Zhou, Yankai Chen, Weizhi Zhang, Yangning Li, Liancheng Fang, Renhe Jiang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09420)  

**Abstract**: Recent improvements in large language models (LLMs) have led many researchers to focus on building fully autonomous AI agents. This position paper questions whether this approach is the right path forward, as these autonomous systems still have problems with reliability, transparency, and understanding the actual requirements of human. We suggest a different approach: LLM-based Human-Agent Systems (LLM-HAS), where AI works with humans rather than replacing them. By keeping human involved to provide guidance, answer questions, and maintain control, these systems can be more trustworthy and adaptable. Looking at examples from healthcare, finance, and software development, we show how human-AI teamwork can handle complex tasks better than AI working alone. We also discuss the challenges of building these collaborative systems and offer practical solutions. This paper argues that progress in AI should not be measured by how independent systems become, but by how well they can work with humans. The most promising future for AI is not in systems that take over human roles, but in those that enhance human capabilities through meaningful partnership. 

---
# Beyond Nash Equilibrium: Bounded Rationality of LLMs and humans in Strategic Decision-making 

**Authors**: Kehan Zheng, Jinfeng Zhou, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09390)  

**Abstract**: Large language models are increasingly used in strategic decision-making settings, yet evidence shows that, like humans, they often deviate from full rationality. In this study, we compare LLMs and humans using experimental paradigms directly adapted from behavioral game-theory research. We focus on two well-studied strategic games, Rock-Paper-Scissors and the Prisoner's Dilemma, which are well known for revealing systematic departures from rational play in human subjects. By placing LLMs in identical experimental conditions, we evaluate whether their behaviors exhibit the bounded rationality characteristic of humans. Our findings show that LLMs reproduce familiar human heuristics, such as outcome-based strategy switching and increased cooperation when future interaction is possible, but they apply these rules more rigidly and demonstrate weaker sensitivity to the dynamic changes in the game environment. Model-level analyses reveal distinctive architectural signatures in strategic behavior, and even reasoning models sometimes struggle to find effective strategies in adaptive situations. These results indicate that current LLMs capture only a partial form of human-like bounded rationality and highlight the need for training methods that encourage flexible opponent modeling and stronger context awareness. 

---
# Ming-Omni: A Unified Multimodal Model for Perception and Generation 

**Authors**: Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, Furong Xu, GuangMing Yao, Jun Zhou, Jingdong Chen, Jianxin Sun, Jiajia Liu, Jianjiang Zhu, Jun Peng, Kaixiang Ji, Kaiyou Song, Kaimeng Ren, Libin Wang, Lixiang Ru, Lele Xie, Longhua Tan, Lyuxin Xue, Lan Wang, Mochen Bai, Ning Gao, Pei Chen, Qingpei Guo, Qinglong Zhang, Qiang Xu, Rui Liu, Ruijie Xiong, Sirui Gao, Tinghao Liu, Taisong Li, Weilong Chai, Xinyu Xiao, Xiaomei Wang, Xiaoxue Chen, Xiao Lu, Xiaoyu Li, Xingning Dong, Xuzheng Yu, Yi Yuan, Yuting Gao, Yunxiao Sun, Yipeng Chen, Yifei Wu, Yongjie Lyu, Ziping Ma, Zipeng Feng, Zhijiang Fang, Zhihao Qiu, Ziyuan Huang, Zhengyu He  

**Link**: [PDF](https://arxiv.org/pdf/2506.09344)  

**Abstract**: We propose Ming-Omni, a unified multimodal model capable of processing images, text, audio, and video, while demonstrating strong proficiency in both speech and image generation. Ming-Omni employs dedicated encoders to extract tokens from different modalities, which are then processed by Ling, an MoE architecture equipped with newly proposed modality-specific routers. This design enables a single model to efficiently process and fuse multimodal inputs within a unified framework, thereby facilitating diverse tasks without requiring separate models, task-specific fine-tuning, or structural redesign. Importantly, Ming-Omni extends beyond conventional multimodal models by supporting audio and image generation. This is achieved through the integration of an advanced audio decoder for natural-sounding speech and Ming-Lite-Uni for high-quality image generation, which also allow the model to engage in context-aware chatting, perform text-to-speech conversion, and conduct versatile image editing. Our experimental results showcase Ming-Omni offers a powerful solution for unified perception and generation across all modalities. Notably, our proposed Ming-Omni is the first open-source model we are aware of to match GPT-4o in modality support, and we release all code and model weights to encourage further research and development in the community. 

---
# Comment on The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity 

**Authors**: C. Opus, A. Lawsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09250)  

**Abstract**: Shojaee et al. (2025) report that Large Reasoning Models (LRMs) exhibit "accuracy collapse" on planning puzzles beyond certain complexity thresholds. We demonstrate that their findings primarily reflect experimental design limitations rather than fundamental reasoning failures. Our analysis reveals three critical issues: (1) Tower of Hanoi experiments systematically exceed model output token limits at reported failure points, with models explicitly acknowledging these constraints in their outputs; (2) The authors' automated evaluation framework fails to distinguish between reasoning failures and practical constraints, leading to misclassification of model capabilities; (3) Most concerningly, their River Crossing benchmarks include mathematically impossible instances for N > 5 due to insufficient boat capacity, yet models are scored as failures for not solving these unsolvable problems. When we control for these experimental artifacts, by requesting generating functions instead of exhaustive move lists, preliminary experiments across multiple models indicate high accuracy on Tower of Hanoi instances previously reported as complete failures. These findings highlight the importance of careful experimental design when evaluating AI reasoning capabilities. 

---
# Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism 

**Authors**: Haoyuan Cai, Zhenghao Peng, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09176)  

**Abstract**: Interactive Imitation Learning (IIL) allows agents to acquire desired behaviors through human interventions, but current methods impose high cognitive demands on human supervisors. We propose the Adaptive Intervention Mechanism (AIM), a novel robot-gated IIL algorithm that learns an adaptive criterion for requesting human demonstrations. AIM utilizes a proxy Q-function to mimic the human intervention rule and adjusts intervention requests based on the alignment between agent and human actions. By assigning high Q-values when the agent deviates from the expert and decreasing these values as the agent becomes proficient, the proxy Q-function enables the agent to assess the real-time alignment with the expert and request assistance when needed. Our expert-in-the-loop experiments reveal that AIM significantly reduces expert monitoring efforts in both continuous and discrete control tasks. Compared to the uncertainty-based baseline Thrifty-DAgger, our method achieves a 40% improvement in terms of human take-over cost and learning efficiency. Furthermore, AIM effectively identifies safety-critical states for expert assistance, thereby collecting higher-quality expert demonstrations and reducing overall expert data and environment interactions needed. Code and demo video are available at this https URL. 

---
# DGS-LRM: Real-Time Deformable 3D Gaussian Reconstruction From Monocular Videos 

**Authors**: Chieh Hubert Lin, Zhaoyang Lv, Songyin Wu, Zhen Xu, Thu Nguyen-Phuoc, Hung-Yu Tseng, Julian Straub, Numair Khan, Lei Xiao, Ming-Hsuan Yang, Yuheng Ren, Richard Newcombe, Zhao Dong, Zhengqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09997)  

**Abstract**: We introduce the Deformable Gaussian Splats Large Reconstruction Model (DGS-LRM), the first feed-forward method predicting deformable 3D Gaussian splats from a monocular posed video of any dynamic scene. Feed-forward scene reconstruction has gained significant attention for its ability to rapidly create digital replicas of real-world environments. However, most existing models are limited to static scenes and fail to reconstruct the motion of moving objects. Developing a feed-forward model for dynamic scene reconstruction poses significant challenges, including the scarcity of training data and the need for appropriate 3D representations and training paradigms. To address these challenges, we introduce several key technical contributions: an enhanced large-scale synthetic dataset with ground-truth multi-view videos and dense 3D scene flow supervision; a per-pixel deformable 3D Gaussian representation that is easy to learn, supports high-quality dynamic view synthesis, and enables long-range 3D tracking; and a large transformer network that achieves real-time, generalizable dynamic scene reconstruction. Extensive qualitative and quantitative experiments demonstrate that DGS-LRM achieves dynamic scene reconstruction quality comparable to optimization-based methods, while significantly outperforming the state-of-the-art predictive dynamic reconstruction method on real-world examples. Its predicted physically grounded 3D deformation is accurate and can readily adapt for long-range 3D tracking tasks, achieving performance on par with state-of-the-art monocular video 3D tracking methods. 

---
# eFlesh: Highly customizable Magnetic Touch Sensing using Cut-Cell Microstructures 

**Authors**: Venkatesh Pattabiraman, Zizhou Huang, Daniele Panozzo, Denis Zorin, Lerrel Pinto, Raunaq Bhirangi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09994)  

**Abstract**: If human experience is any guide, operating effectively in unstructured environments -- like homes and offices -- requires robots to sense the forces during physical interaction. Yet, the lack of a versatile, accessible, and easily customizable tactile sensor has led to fragmented, sensor-specific solutions in robotic manipulation -- and in many cases, to force-unaware, sensorless approaches. With eFlesh, we bridge this gap by introducing a magnetic tactile sensor that is low-cost, easy to fabricate, and highly customizable. Building an eFlesh sensor requires only four components: a hobbyist 3D printer, off-the-shelf magnets (<$5), a CAD model of the desired shape, and a magnetometer circuit board. The sensor is constructed from tiled, parameterized microstructures, which allow for tuning the sensor's geometry and its mechanical response. We provide an open-source design tool that converts convex OBJ/STL files into 3D-printable STLs for fabrication. This modular design framework enables users to create application-specific sensors, and to adjust sensitivity depending on the task. Our sensor characterization experiments demonstrate the capabilities of eFlesh: contact localization RMSE of 0.5 mm, and force prediction RMSE of 0.27 N for normal force and 0.12 N for shear force. We also present a learned slip detection model that generalizes to unseen objects with 95% accuracy, and visuotactile control policies that improve manipulation performance by 40% over vision-only baselines -- achieving 91% average success rate for four precise tasks that require sub-mm accuracy for successful completion. All design files, code and the CAD-to-eFlesh STL conversion tool are open-sourced and available on this https URL. 

---
# Text-Aware Image Restoration with Diffusion Models 

**Authors**: Jaewon Min, Jin Hyeon Kim, Paul Hyunbin Cho, Jaeeun Lee, Jihye Park, Minkyu Park, Sangpil Kim, Hyunhee Park, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09993)  

**Abstract**: Image restoration aims to recover degraded images. However, existing diffusion-based restoration methods, despite great success in natural image restoration, often struggle to faithfully reconstruct textual regions in degraded images. Those methods frequently generate plausible but incorrect text-like patterns, a phenomenon we refer to as text-image hallucination. In this paper, we introduce Text-Aware Image Restoration (TAIR), a novel restoration task that requires the simultaneous recovery of visual contents and textual fidelity. To tackle this task, we present SA-Text, a large-scale benchmark of 100K high-quality scene images densely annotated with diverse and complex text instances. Furthermore, we propose a multi-task diffusion framework, called TeReDiff, that integrates internal features from diffusion models into a text-spotting module, enabling both components to benefit from joint training. This allows for the extraction of rich text representations, which are utilized as prompts in subsequent denoising steps. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art restoration methods, achieving significant gains in text recognition accuracy. See our project page: this https URL 

---
# EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits 

**Authors**: Ron Yosef, Moran Yanuka, Yonatan Bitton, Dani Lischinski  

**Link**: [PDF](https://arxiv.org/pdf/2506.09988)  

**Abstract**: Text-guided image editing, fueled by recent advancements in generative AI, is becoming increasingly widespread. This trend highlights the need for a comprehensive framework to verify text-guided edits and assess their quality. To address this need, we introduce EditInspector, a novel benchmark for evaluation of text-guided image edits, based on human annotations collected using an extensive template for edit verification. We leverage EditInspector to evaluate the performance of state-of-the-art (SoTA) vision and language models in assessing edits across various dimensions, including accuracy, artifact detection, visual quality, seamless integration with the image scene, adherence to common sense, and the ability to describe edit-induced changes. Our findings indicate that current models struggle to evaluate edits comprehensively and frequently hallucinate when describing the changes. To address these challenges, we propose two novel methods that outperform SoTA models in both artifact detection and difference caption generation. 

---
# InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions 

**Authors**: Zhenzhi Wang, Jiaqi Yang, Jianwen Jiang, Chao Liang, Gaojie Lin, Zerong Zheng, Ceyuan Yang, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09984)  

**Abstract**: End-to-end human animation with rich multi-modal conditions, e.g., text, image and audio has achieved remarkable advancements in recent years. However, most existing methods could only animate a single subject and inject conditions in a global manner, ignoring scenarios that multiple concepts could appears in the same video with rich human-human interactions and human-object interactions. Such global assumption prevents precise and per-identity control of multiple concepts including humans and objects, therefore hinders applications. In this work, we discard the single-entity assumption and introduce a novel framework that enforces strong, region-specific binding of conditions from modalities to each identity's spatiotemporal footprint. Given reference images of multiple concepts, our method could automatically infer layout information by leveraging a mask predictor to match appearance cues between the denoised video and each reference appearance. Furthermore, we inject local audio condition into its corresponding region to ensure layout-aligned modality matching in a iterative manner. This design enables the high-quality generation of controllable multi-concept human-centric videos. Empirical results and ablation studies validate the effectiveness of our explicit layout control for multi-modal conditions compared to implicit counterparts and other existing methods. 

---
# Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing 

**Authors**: Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09965)  

**Abstract**: As textual reasoning with large language models (LLMs) has advanced significantly, there has been growing interest in enhancing the multimodal reasoning capabilities of large vision-language models (LVLMs). However, existing methods primarily approach multimodal reasoning in a straightforward, text-centric manner, where both reasoning and answer derivation are conducted purely through text, with the only difference being the presence of multimodal input. As a result, these methods often encounter fundamental limitations in spatial reasoning tasks that demand precise geometric understanding and continuous spatial tracking-capabilities that humans achieve through mental visualization and manipulation. To address the limitations, we propose drawing to reason in space, a novel paradigm that enables LVLMs to reason through elementary drawing operations in the visual space. By equipping models with basic drawing operations, including annotating bounding boxes and drawing auxiliary lines, we empower them to express and analyze spatial relationships through direct visual manipulation, meanwhile avoiding the performance ceiling imposed by specialized perception tools in previous tool-integrated reasoning approaches. To cultivate this capability, we develop a three-stage training framework: cold-start training with synthetic data to establish basic drawing abilities, reflective rejection sampling to enhance self-reflection behaviors, and reinforcement learning to directly optimize for target rewards. Extensive experiments demonstrate that our model, named VILASR, consistently outperforms existing methods across diverse spatial reasoning benchmarks, involving maze navigation, static spatial reasoning, video-based reasoning, and multi-view-based reasoning tasks, with an average improvement of 18.4%. 

---
# LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge 

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09956)  

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection. 

---
# Vision Generalist Model: A Survey 

**Authors**: Ziyi Wang, Yongming Rao, Shuofeng Sun, Xinrun Liu, Yi Wei, Xumin Yu, Zuyan Liu, Yanbo Wang, Hongmin Liu, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09954)  

**Abstract**: Recently, we have witnessed the great success of the generalist model in natural language processing. The generalist model is a general framework trained with massive data and is able to process various downstream tasks simultaneously. Encouraged by their impressive performance, an increasing number of researchers are venturing into the realm of applying these models to computer vision tasks. However, the inputs and outputs of vision tasks are more diverse, and it is difficult to summarize them as a unified representation. In this paper, we provide a comprehensive overview of the vision generalist models, delving into their characteristics and capabilities within the field. First, we review the background, including the datasets, tasks, and benchmarks. Then, we dig into the design of frameworks that have been proposed in existing research, while also introducing the techniques employed to enhance their performance. To better help the researchers comprehend the area, we take a brief excursion into related domains, shedding light on their interconnections and potential synergies. To conclude, we provide some real-world application scenarios, undertake a thorough examination of the persistent challenges, and offer insights into possible directions for future research endeavors. 

---
# Outside Knowledge Conversational Video (OKCV) Dataset -- Dialoguing over Videos 

**Authors**: Benjamin Reichman, Constantin Patsch, Jack Truxal, Atishay Jain, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2506.09953)  

**Abstract**: In outside knowledge visual question answering (OK-VQA), the model must identify relevant visual information within an image and incorporate external knowledge to accurately respond to a question. Extending this task to a visually grounded dialogue setting based on videos, a conversational model must both recognize pertinent visual details over time and answer questions where the required information is not necessarily present in the visual information. Moreover, the context of the overall conversation must be considered for the subsequent dialogue. To explore this task, we introduce a dataset comprised of $2,017$ videos with $5,986$ human-annotated dialogues consisting of $40,954$ interleaved dialogue turns. While the dialogue context is visually grounded in specific video segments, the questions further require external knowledge that is not visually present. Thus, the model not only has to identify relevant video parts but also leverage external knowledge to converse within the dialogue. We further provide several baselines evaluated on our dataset and show future challenges associated with this task. The dataset is made publicly available here: this https URL. 

---
# UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting 

**Authors**: Ziyi Wang, Yanran Zhang, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09952)  

**Abstract**: The scale diversity of point cloud data presents significant challenges in developing unified representation learning techniques for 3D vision. Currently, there are few unified 3D models, and no existing pre-training method is equally effective for both object- and scene-level point clouds. In this paper, we introduce UniPre3D, the first unified pre-training method that can be seamlessly applied to point clouds of any scale and 3D models of any architecture. Our approach predicts Gaussian primitives as the pre-training task and employs differentiable Gaussian splatting to render images, enabling precise pixel-level supervision and end-to-end optimization. To further regulate the complexity of the pre-training task and direct the model's focus toward geometric structures, we integrate 2D features from pre-trained image models to incorporate well-established texture knowledge. We validate the universal effectiveness of our proposed method through extensive experiments across a variety of object- and scene-level tasks, using diverse point cloud models as backbones. Code is available at this https URL. 

---
# CausalVQA: A Physically Grounded Causal Reasoning Benchmark for Video Models 

**Authors**: Aaron Foss, Chloe Evans, Sasha Mitts, Koustuv Sinha, Ammar Rizvi, Justine T. Kao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09943)  

**Abstract**: We introduce CausalVQA, a benchmark dataset for video question answering (VQA) composed of question-answer pairs that probe models' understanding of causality in the physical world. Existing VQA benchmarks either tend to focus on surface perceptual understanding of real-world videos, or on narrow physical reasoning questions created using simulation environments. CausalVQA fills an important gap by presenting challenging questions that are grounded in real-world scenarios, while focusing on models' ability to predict the likely outcomes of different actions and events through five question types: counterfactual, hypothetical, anticipation, planning and descriptive. We designed quality control mechanisms that prevent models from exploiting trivial shortcuts, requiring models to base their answers on deep visual understanding instead of linguistic cues. We find that current frontier multimodal models fall substantially below human performance on the benchmark, especially on anticipation and hypothetical questions. This highlights a challenge for current systems to leverage spatial-temporal reasoning, understanding of physical principles, and comprehension of possible alternatives to make accurate predictions in real-world settings. 

---
# VerIF: Verification Engineering for Reinforcement Learning in Instruction Following 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09942)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a key technique for enhancing large language models (LLMs), with verification engineering playing a central role. However, best practices for RL in instruction following remain underexplored. In this work, we explore the verification challenge in RL for instruction following and propose VerIF, a verification method that combines rule-based code verification with LLM-based verification from a large reasoning model (e.g., QwQ-32B). To support this approach, we construct a high-quality instruction-following dataset, VerInstruct, containing approximately 22,000 instances with associated verification signals. We apply RL training with VerIF to two models, achieving significant improvements across several representative instruction-following benchmarks. The trained models reach state-of-the-art performance among models of comparable size and generalize well to unseen constraints. We further observe that their general capabilities remain unaffected, suggesting that RL with VerIF can be integrated into existing RL recipes to enhance overall model performance. We have released our datasets, codes, and models to facilitate future research at this https URL. 

---
# The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability 

**Authors**: Jiachen Hu, Rui Ai, Han Zhong, Xiaoyu Chen, Liwei Wang, Zhaoran Wang, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09940)  

**Abstract**: Information asymmetry is a pervasive feature of multi-agent systems, especially evident in economics and social sciences. In these settings, agents tailor their actions based on private information to maximize their rewards. These strategic behaviors often introduce complexities due to confounding variables. Simultaneously, knowledge transportability poses another significant challenge, arising from the difficulties of conducting experiments in target environments. It requires transferring knowledge from environments where empirical data is more readily available. Against these backdrops, this paper explores a fundamental question in online learning: Can we employ non-i.i.d. actions to learn about confounders even when requiring knowledge transfer? We present a sample-efficient algorithm designed to accurately identify system dynamics under information asymmetry and to navigate the challenges of knowledge transfer effectively in reinforcement learning, framed within an online strategic interaction model. Our method provably achieves learning of an $\epsilon$-optimal policy with a tight sample complexity of $O(1/\epsilon^2)$. 

---
# SAFE: Multitask Failure Detection for Vision-Language-Action Models 

**Authors**: Qiao Gu, Yuanliang Ju, Shengxiang Sun, Igor Gilitschenski, Haruki Nishimura, Masha Itkina, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2506.09937)  

**Abstract**: While vision-language-action models (VLAs) have shown promising robotic behaviors across a diverse set of manipulation tasks, they achieve limited success rates when deployed on novel tasks out-of-the-box. To allow these policies to safely interact with their environments, we need a failure detector that gives a timely alert such that the robot can stop, backtrack, or ask for help. However, existing failure detectors are trained and tested only on one or a few specific tasks, while VLAs require the detector to generalize and detect failures also in unseen tasks and novel environments. In this paper, we introduce the multitask failure detection problem and propose SAFE, a failure detector for generalist robot policies such as VLAs. We analyze the VLA feature space and find that VLAs have sufficient high-level knowledge about task success and failure, which is generic across different tasks. Based on this insight, we design SAFE to learn from VLA internal features and predict a single scalar indicating the likelihood of task failure. SAFE is trained on both successful and failed rollouts, and is evaluated on unseen tasks. SAFE is compatible with different policy architectures. We test it on OpenVLA, $\pi_0$, and $\pi_0$-FAST in both simulated and real-world environments extensively. We compare SAFE with diverse baselines and show that SAFE achieves state-of-the-art failure detection performance and the best trade-off between accuracy and detection time using conformal prediction. More qualitative results can be found at this https URL. 

---
# HadaNorm: Diffusion Transformer Quantization through Mean-Centered Transformations 

**Authors**: Marco Federici, Riccardo Del Chiaro, Boris van Breugel, Paul Whatmough, Markus Nagel  

**Link**: [PDF](https://arxiv.org/pdf/2506.09932)  

**Abstract**: Diffusion models represent the cutting edge in image generation, but their high memory and computational demands hinder deployment on resource-constrained devices. Post-Training Quantization (PTQ) offers a promising solution by reducing the bitwidth of matrix operations. However, standard PTQ methods struggle with outliers, and achieving higher compression often requires transforming model weights and activations before quantization. In this work, we propose HadaNorm, a novel linear transformation that extends existing approaches and effectively mitigates outliers by normalizing activations feature channels before applying Hadamard transformations, enabling more aggressive activation quantization. We demonstrate that HadaNorm consistently reduces quantization error across the various components of transformer blocks, achieving superior efficiency-performance trade-offs when compared to state-of-the-art methods. 

---
# PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants 

**Authors**: Zheng Zhao, Clara Vania, Subhradeep Kayal, Naila Khan, Shay B. Cohen, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.09902)  

**Abstract**: Large language models (LLMs) have advanced conversational AI assistants. However, systematically evaluating how well these assistants apply personalization--adapting to individual user preferences while completing tasks--remains challenging. Existing personalization benchmarks focus on chit-chat, non-conversational tasks, or narrow domains, failing to capture the complexities of personalized task-oriented assistance. To address this, we introduce PersonaLens, a comprehensive benchmark for evaluating personalization in task-oriented AI assistants. Our benchmark features diverse user profiles equipped with rich preferences and interaction histories, along with two specialized LLM-based agents: a user agent that engages in realistic task-oriented dialogues with AI assistants, and a judge agent that employs the LLM-as-a-Judge paradigm to assess personalization, response quality, and task success. Through extensive experiments with current LLM assistants across diverse tasks, we reveal significant variability in their personalization capabilities, providing crucial insights for advancing conversational AI systems. 

---
# Causal Climate Emulation with Bayesian Filtering 

**Authors**: Sebastian Hickman, Ilija Trajkovic, Julia Kaltenborn, Francis Pelletier, Alex Archibald, Yaniv Gurwicz, Peer Nowack, David Rolnick, Julien Boussard  

**Link**: [PDF](https://arxiv.org/pdf/2506.09891)  

**Abstract**: Traditional models of climate change use complex systems of coupled equations to simulate physical processes across the Earth system. These simulations are highly computationally expensive, limiting our predictions of climate change and analyses of its causes and effects. Machine learning has the potential to quickly emulate data from climate models, but current approaches are not able to incorporate physics-informed causal relationships. Here, we develop an interpretable climate model emulator based on causal representation learning. We derive a physics-informed approach including a Bayesian filter for stable long-term autoregressive emulation. We demonstrate that our emulator learns accurate climate dynamics, and we show the importance of each one of its components on a realistic synthetic dataset and data from two widely deployed climate models. 

---
# The Emergence of Abstract Thought in Large Language Models Beyond Any Language 

**Authors**: Yuxin Chen, Yiran Zhao, Yang Zhang, An Zhang, Kenji Kawaguchi, Shafiq Joty, Junnan Li, Tat-Seng Chua, Michael Qizhe Shieh, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09890)  

**Abstract**: As large language models (LLMs) continue to advance, their capacity to function effectively across a diverse range of languages has shown marked improvement. Preliminary studies observe that the hidden activations of LLMs often resemble English, even when responding to non-English prompts. This has led to the widespread assumption that LLMs may "think" in English. However, more recent results showing strong multilingual performance, even surpassing English performance on specific tasks in other languages, challenge this view. In this work, we find that LLMs progressively develop a core language-agnostic parameter space-a remarkably small subset of parameters whose deactivation results in significant performance degradation across all languages. This compact yet critical set of parameters underlies the model's ability to generalize beyond individual languages, supporting the emergence of abstract thought that is not tied to any specific linguistic system. Specifically, we identify language-related neurons-those are consistently activated during the processing of particular languages, and categorize them as either shared (active across multiple languages) or exclusive (specific to one). As LLMs undergo continued development over time, we observe a marked increase in both the proportion and functional importance of shared neurons, while exclusive neurons progressively diminish in influence. These shared neurons constitute the backbone of the core language-agnostic parameter space, supporting the emergence of abstract thought. Motivated by these insights, we propose neuron-specific training strategies tailored to LLMs' language-agnostic levels at different development stages. Experiments across diverse LLM families support our approach. 

---
# Attention Head Embeddings with Trainable Deep Kernels for Hallucination Detection in LLMs 

**Authors**: Rodion Oblovatny, Alexandra Bazarova, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.09886)  

**Abstract**: We present a novel approach for detecting hallucinations in large language models (LLMs) by analyzing the probabilistic divergence between prompt and response hidden-state distributions. Counterintuitively, we find that hallucinated responses exhibit smaller deviations from their prompts compared to grounded responses, suggesting that hallucinations often arise from superficial rephrasing rather than substantive reasoning. Leveraging this insight, we propose a model-intrinsic detection method that uses distributional distances as principled hallucination scores, eliminating the need for external knowledge or auxiliary models. To enhance sensitivity, we employ deep learnable kernels that automatically adapt to capture nuanced geometric differences between distributions. Our approach outperforms existing baselines, demonstrating state-of-the-art performance on several benchmarks. The method remains competitive even without kernel training, offering a robust, scalable solution for hallucination detection. 

---
# 3D-Aware Vision-Language Models Fine-Tuning with Geometric Distillation 

**Authors**: Seonho Lee, Jiho Choi, Inha Kang, Jiwook Kim, Junsung Park, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09883)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable performance on diverse visual and linguistic tasks, yet they remain fundamentally limited in their understanding of 3D spatial structures. We propose Geometric Distillation, a lightweight, annotation-free fine-tuning framework that injects human-inspired geometric cues into pretrained VLMs without modifying their architecture. By distilling (1) sparse correspondences, (2) relative depth relations, and (3) dense cost volumes from off-the-shelf 3D foundation models (e.g., MASt3R, VGGT), our method shapes representations to be geometry-aware while remaining compatible with natural image-text inputs. Through extensive evaluations on 3D vision-language reasoning and 3D perception benchmarks, our method consistently outperforms prior approaches, achieving improved 3D spatial reasoning with significantly lower computational cost. Our work demonstrates a scalable and efficient path to bridge 2D-trained VLMs with 3D understanding, opening up wider use in spatially grounded multimodal tasks. 

---
# Stakeholder Participation for Responsible AI Development: Disconnects Between Guidance and Current Practice 

**Authors**: Emma Kallina, Thomas Bohné, Jat Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.09873)  

**Abstract**: Responsible AI (rAI) guidance increasingly promotes stakeholder involvement (SHI) during AI development. At the same time, SHI is already common in commercial software development, but with potentially different foci. This study clarifies the extent to which established SHI practices are able to contribute to rAI efforts as well as potential disconnects -- essential insights to inform and tailor future interventions that further shift industry practice towards rAI efforts. First, we analysed 56 rAI guidance documents to identify why SHI is recommended (i.e. its expected benefits for rAI) and uncovered goals such as redistributing power, improving socio-technical understandings, anticipating risks, and enhancing public oversight. To understand why and how SHI is currently practised in commercial settings, we then conducted an online survey (n=130) and semi-structured interviews (n=10) with AI practitioners. Our findings reveal that SHI in practice is primarily driven by commercial priorities (e.g. customer value, compliance) and several factors currently discourage more rAI-aligned SHI practices. This suggests that established SHI practices are largely not contributing to rAI efforts. To address this disconnect, we propose interventions and research opportunities to advance rAI development in practice. 

---
# Guided Graph Compression for Quantum Graph Neural Networks 

**Authors**: Mikel Casals, Vasilis Belis, Elias F. Combarro, Eduard Alarcón, Sofia Vallecorsa, Michele Grossi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09862)  

**Abstract**: Graph Neural Networks (GNNs) are effective for processing graph-structured data but face challenges with large graphs due to high memory requirements and inefficient sparse matrix operations on GPUs. Quantum Computing (QC) offers a promising avenue to address these issues and inspires new algorithmic approaches. In particular, Quantum Graph Neural Networks (QGNNs) have been explored in recent literature. However, current quantum hardware limits the dimension of the data that can be effectively encoded. Existing approaches either simplify datasets manually or use artificial graph datasets. This work introduces the Guided Graph Compression (GGC) framework, which uses a graph autoencoder to reduce both the number of nodes and the dimensionality of node features. The compression is guided to enhance the performance of a downstream classification task, which can be applied either with a quantum or a classical classifier. The framework is evaluated on the Jet Tagging task, a classification problem of fundamental importance in high energy physics that involves distinguishing particle jets initiated by quarks from those by gluons. The GGC is compared against using the autoencoder as a standalone preprocessing step and against a baseline classical GNN classifier. Our numerical results demonstrate that GGC outperforms both alternatives, while also facilitating the testing of novel QGNN ansatzes on realistic datasets. 

---
# Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning 

**Authors**: Xiangning Yu, Zhuohan Wang, Linyi Yang, Haoxuan Li, Anjie Liu, Xiao Xue, Jun Wang, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09853)  

**Abstract**: Chain-of-Thought (CoT) prompting plays an indispensable role in endowing large language models (LLMs) with complex reasoning capabilities. However, CoT currently faces two fundamental challenges: (1) Sufficiency, which ensures that the generated intermediate inference steps comprehensively cover and substantiate the final conclusion; and (2) Necessity, which identifies the inference steps that are truly indispensable for the soundness of the resulting answer. We propose a causal framework that characterizes CoT reasoning through the dual lenses of sufficiency and necessity. Incorporating causal Probability of Sufficiency and Necessity allows us not only to determine which steps are logically sufficient or necessary to the prediction outcome, but also to quantify their actual influence on the final reasoning outcome under different intervention scenarios, thereby enabling the automated addition of missing steps and the pruning of redundant ones. Extensive experimental results on various mathematical and commonsense reasoning benchmarks confirm substantial improvements in reasoning efficiency and reduced token usage without sacrificing accuracy. Our work provides a promising direction for improving LLM reasoning performance and cost-effectiveness. 

---
# Dataset of News Articles with Provenance Metadata for Media Relevance Assessment 

**Authors**: Tomas Peterka, Matyas Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2506.09847)  

**Abstract**: Out-of-context and misattributed imagery is the leading form of media manipulation in today's misinformation and disinformation landscape. The existing methods attempting to detect this practice often only consider whether the semantics of the imagery corresponds to the text narrative, missing manipulation so long as the depicted objects or scenes somewhat correspond to the narrative at hand. To tackle this, we introduce News Media Provenance Dataset, a dataset of news articles with provenance-tagged images. We formulate two tasks on this dataset, location of origin relevance (LOR) and date and time of origin relevance (DTOR), and present baseline results on six large language models (LLMs). We identify that, while the zero-shot performance on LOR is promising, the performance on DTOR hinders, leaving room for specialized architectures and future work. 

---
# Learning to Align: Addressing Character Frequency Distribution Shifts in Handwritten Text Recognition 

**Authors**: Panagiotis Kaliosis, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09846)  

**Abstract**: Handwritten text recognition aims to convert visual input into machine-readable text, and it remains challenging due to the evolving and context-dependent nature of handwriting. Character sets change over time, and character frequency distributions shift across historical periods or regions, often causing models trained on broad, heterogeneous corpora to underperform on specific subsets. To tackle this, we propose a novel loss function that incorporates the Wasserstein distance between the character frequency distribution of the predicted text and a target distribution empirically derived from training data. By penalizing divergence from expected distributions, our approach enhances both accuracy and robustness under temporal and contextual intra-dataset shifts. Furthermore, we demonstrate that character distribution alignment can also improve existing models at inference time without requiring retraining by integrating it as a scoring function in a guided decoding scheme. Experimental results across multiple datasets and architectures confirm the effectiveness of our method in boosting generalization and performance. We open source our code at this https URL. 

---
# OctoNav: Towards Generalist Embodied Navigation 

**Authors**: Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09839)  

**Abstract**: Embodied navigation stands as a foundation pillar within the broader pursuit of embodied AI. However, previous navigation research is divided into different tasks/capabilities, e.g., ObjNav, ImgNav and VLN, where they differ in task objectives and modalities, making datasets and methods are designed individually. In this work, we take steps toward generalist navigation agents, which can follow free-form instructions that include arbitrary compounds of multi-modal and multi-capability. To achieve this, we propose a large-scale benchmark and corresponding method, termed OctoNav-Bench and OctoNav-R1. Specifically, OctoNav-Bench features continuous environments and is constructed via a designed annotation pipeline. We thoroughly craft instruction-trajectory pairs, where instructions are diverse in free-form with arbitrary modality and capability. Also, we construct a Think-Before-Action (TBA-CoT) dataset within OctoNav-Bench to provide the thinking process behind actions. For OctoNav-R1, we build it upon MLLMs and adapt it to a VLA-type model, which can produce low-level actions solely based on 2D visual observations. Moreover, we design a Hybrid Training Paradigm (HTP) that consists of three stages, i.e., Action-/TBA-SFT, Nav-GPRO, and Online RL stages. Each stage contains specifically designed learning policies and rewards. Importantly, for TBA-SFT and Nav-GRPO designs, we are inspired by the OpenAI-o1 and DeepSeek-R1, which show impressive reasoning ability via thinking-before-answer. Thus, we aim to investigate how to achieve thinking-before-action in the embodied navigation field, to improve model's reasoning ability toward generalists. Specifically, we propose TBA-SFT to utilize the TBA-CoT dataset to fine-tune the model as a cold-start phrase and then leverage Nav-GPRO to improve its thinking ability. Finally, OctoNav-R1 shows superior performance compared with previous methods. 

---
# DynaSplat: Dynamic-Static Gaussian Splatting with Hierarchical Motion Decomposition for Scene Reconstruction 

**Authors**: Junli Deng, Ping Shi, Qipei Li, Jinyang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09836)  

**Abstract**: Reconstructing intricate, ever-changing environments remains a central ambition in computer vision, yet existing solutions often crumble before the complexity of real-world dynamics. We present DynaSplat, an approach that extends Gaussian Splatting to dynamic scenes by integrating dynamic-static separation and hierarchical motion modeling. First, we classify scene elements as static or dynamic through a novel fusion of deformation offset statistics and 2D motion flow consistency, refining our spatial representation to focus precisely where motion matters. We then introduce a hierarchical motion modeling strategy that captures both coarse global transformations and fine-grained local movements, enabling accurate handling of intricate, non-rigid motions. Finally, we integrate physically-based opacity estimation to ensure visually coherent reconstructions, even under challenging occlusions and perspective shifts. Extensive experiments on challenging datasets reveal that DynaSplat not only surpasses state-of-the-art alternatives in accuracy and realism but also provides a more intuitive, compact, and efficient route to dynamic scene reconstruction. 

---
# EmoNet-Voice: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection 

**Authors**: Christoph Schuhmann, Robert Kaczmarczyk, Gollam Rabby, Felix Friedrich, Maurice Kraus, Kourosh Nadi, Huu Nguyen, Kristian Kersting, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2506.09827)  

**Abstract**: The advancement of text-to-speech and audio generation models necessitates robust benchmarks for evaluating the emotional understanding capabilities of AI systems. Current speech emotion recognition (SER) datasets often exhibit limitations in emotional granularity, privacy concerns, or reliance on acted portrayals. This paper introduces EmoNet-Voice, a new resource for speech emotion detection, which includes EmoNet-Voice Big, a large-scale pre-training dataset (featuring over 4,500 hours of speech across 11 voices, 40 emotions, and 4 languages), and EmoNet-Voice Bench, a novel benchmark dataset with human expert annotations. EmoNet-Voice is designed to evaluate SER models on a fine-grained spectrum of 40 emotion categories with different levels of intensities. Leveraging state-of-the-art voice generation, we curated synthetic audio snippets simulating actors portraying scenes designed to evoke specific emotions. Crucially, we conducted rigorous validation by psychology experts who assigned perceived intensity labels. This synthetic, privacy-preserving approach allows for the inclusion of sensitive emotional states often absent in existing datasets. Lastly, we introduce Empathic Insight Voice models that set a new standard in speech emotion recognition with high agreement with human experts. Our evaluations across the current model landscape exhibit valuable findings, such as high-arousal emotions like anger being much easier to detect than low-arousal states like concentration. 

---
# Superstudent intelligence in thermodynamics 

**Authors**: Rebecca Loubet, Pascal Zittlau, Marco Hoffmann, Luisa Vollmer, Sophie Fellenz, Heike Leitte, Fabian Jirasek, Johannes Lenhard, Hans Hasse  

**Link**: [PDF](https://arxiv.org/pdf/2506.09822)  

**Abstract**: In this short note, we report and analyze a striking event: OpenAI's large language model o3 has outwitted all students in a university exam on thermodynamics. The thermodynamics exam is a difficult hurdle for most students, where they must show that they have mastered the fundamentals of this important topic. Consequently, the failure rates are very high, A-grades are rare - and they are considered proof of the students' exceptional intellectual abilities. This is because pattern learning does not help in the exam. The problems can only be solved by knowledgeably and creatively combining principles of thermodynamics. We have given our latest thermodynamics exam not only to the students but also to OpenAI's most powerful reasoning model, o3, and have assessed the answers of o3 exactly the same way as those of the students. In zero-shot mode, the model o3 solved all problems correctly, better than all students who took the exam; its overall score was in the range of the best scores we have seen in more than 10,000 similar exams since 1985. This is a turning point: machines now excel in complex tasks, usually taken as proof of human intellectual capabilities. We discuss the consequences this has for the work of engineers and the education of future engineers. 

---
# CoRT: Code-integrated Reasoning within Thinking 

**Authors**: Chengpeng Li, Zhengyang Tang, Ziniu Li, Mingfeng Xue, Keqin Bao, Tian Ding, Ruoyu Sun, Benyou Wang, Xiang Wang, Junyang Lin, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09820)  

**Abstract**: Large Reasoning Models (LRMs) like o1 and DeepSeek-R1 have shown remarkable progress in natural language reasoning with long chain-of-thought (CoT), yet they remain inefficient or inaccurate when handling complex mathematical operations. Addressing these limitations through computational tools (e.g., computation libraries and symbolic solvers) is promising, but it introduces a technical challenge: Code Interpreter (CI) brings external knowledge beyond the model's internal text representations, thus the direct combination is not efficient. This paper introduces CoRT, a post-training framework for teaching LRMs to leverage CI effectively and efficiently. As a first step, we address the data scarcity issue by synthesizing code-integrated reasoning data through Hint-Engineering, which strategically inserts different hints at appropriate positions to optimize LRM-CI interaction. We manually create 30 high-quality samples, upon which we post-train models ranging from 1.5B to 32B parameters, with supervised fine-tuning, rejection fine-tuning and reinforcement learning. Our experimental results demonstrate that Hint-Engineering models achieve 4\% and 8\% absolute improvements on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B respectively, across five challenging mathematical reasoning datasets. Furthermore, Hint-Engineering models use about 30\% fewer tokens for the 32B model and 50\% fewer tokens for the 1.5B model compared with the natural language models. The models and code are available at this https URL. 

---
# A theoretical framework for self-supervised contrastive learning for continuous dependent data 

**Authors**: Alexander Marusov, Alexander Yuhay, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.09785)  

**Abstract**: Self-supervised learning (SSL) has emerged as a powerful approach to learning representations, particularly in the field of computer vision. However, its application to dependent data, such as temporal and spatio-temporal domains, remains underexplored. Besides, traditional contrastive SSL methods often assume \emph{semantic independence between samples}, which does not hold for dependent data exhibiting complex correlations. We propose a novel theoretical framework for contrastive SSL tailored to \emph{continuous dependent data}, which allows the nearest samples to be semantically close to each other. In particular, we propose two possible \textit{ground truth similarity measures} between objects -- \emph{hard} and \emph{soft} closeness. Under it, we derive an analytical form for the \textit{estimated similarity matrix} that accommodates both types of closeness between samples, thereby introducing dependency-aware loss functions. We validate our approach, \emph{Dependent TS2Vec}, on temporal and spatio-temporal downstream problems. Given the dependency patterns presented in the data, our approach surpasses modern ones for dependent data, highlighting the effectiveness of our theoretically grounded loss functions for SSL in capturing spatio-temporal dependencies. Specifically, we outperform TS2Vec on the standard UEA and UCR benchmarks, with accuracy improvements of $4.17$\% and $2.08$\%, respectively. Furthermore, on the drought classification task, which involves complex spatio-temporal patterns, our method achieves a $7$\% higher ROC-AUC score. 

---
# Q-SAM2: Accurate Quantization for Segment Anything Model 2 

**Authors**: Nicola Farronato, Florian Scheidegger, Mattia Rigotti, Cristiano Malossi, Michele Magno, Haotong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09782)  

**Abstract**: The Segment Anything Model 2 (SAM2) has gained significant attention as a foundational approach for promptable image and video segmentation. However, its expensive computational and memory consumption poses a severe challenge for its application in resource-constrained scenarios. In this paper, we propose an accurate low-bit quantization method for efficient SAM2, termed Q-SAM2. To address the performance degradation caused by the singularities in weight and activation distributions during quantization, Q-SAM2 introduces two novel technical contributions. We first introduce a linear layer calibration method for low-bit initialization of SAM2, which minimizes the Frobenius norm over a small image batch to reposition weight distributions for improved quantization. We then propose a Quantization-Aware Training (QAT) pipeline that applies clipping to suppress outliers and allows the network to adapt to quantization thresholds during training. Our comprehensive experiments demonstrate that Q-SAM2 allows for highly accurate inference while substantially improving efficiency. Both quantitative and visual results show that our Q-SAM2 surpasses existing state-of-the-art general quantization schemes, especially for ultra-low 2-bit quantization. While designed for quantization-aware training, our proposed calibration technique also proves effective in post-training quantization, achieving up to a 66% mIoU accuracy improvement over non-calibrated models. 

---
# Inverting Black-Box Face Recognition Systems via Zero-Order Optimization in Eigenface Space 

**Authors**: Anton Razzhigaev, Matvey Mikhalchuk, Klim Kireev, Igor Udovichenko, Andrey Kuznetsov, Aleksandr Petiushko  

**Link**: [PDF](https://arxiv.org/pdf/2506.09777)  

**Abstract**: Reconstructing facial images from black-box recognition models poses a significant privacy threat. While many methods require access to embeddings, we address the more challenging scenario of model inversion using only similarity scores. This paper introduces DarkerBB, a novel approach that reconstructs color faces by performing zero-order optimization within a PCA-derived eigenface space. Despite this highly limited information, experiments on LFW, AgeDB-30, and CFP-FP benchmarks demonstrate that DarkerBB achieves state-of-the-art verification accuracies in the similarity-only setting, with competitive query efficiency. 

---
# Load-Aware Training Scheduling for Model Circulation-based Decentralized Federated Learning 

**Authors**: Haruki Kainuma, Takayuki Nishio  

**Link**: [PDF](https://arxiv.org/pdf/2506.09769)  

**Abstract**: This paper proposes Load-aware Tram-FL, an extension of Tram-FL that introduces a training scheduling mechanism to minimize total training time in decentralized federated learning by accounting for both computational and communication loads. The scheduling problem is formulated as a global optimization task, which-though intractable in its original form-is made solvable by decomposing it into node-wise subproblems. To promote balanced data utilization under non-IID distributions, a variance constraint is introduced, while the overall training latency, including both computation and communication costs, is minimized through the objective function. Simulation results on MNIST and CIFAR-10 demonstrate that Load-aware Tram-FL significantly reduces training time and accelerates convergence compared to baseline methods. 

---
# Intelligent Design 4.0: Paradigm Evolution Toward the Agentic AI Era 

**Authors**: Shuo Jiang, Min Xie, Frank Youhua Chen, Jian Ma, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09755)  

**Abstract**: Research and practice in Intelligent Design (ID) have significantly enhanced engineering innovation, efficiency, quality, and productivity over recent decades, fundamentally reshaping how engineering designers think, behave, and interact with design processes. The recent emergence of Foundation Models (FMs), particularly Large Language Models (LLMs), has demonstrated general knowledge-based reasoning capabilities, and open new paths and avenues for further transformation in engineering design. In this context, this paper introduces Intelligent Design 4.0 (ID 4.0) as an emerging paradigm empowered by agentic AI systems. We review the historical evolution of ID across four distinct stages: rule-based expert systems, task-specific machine learning models, large-scale foundation AI models, and the recent emerging paradigm of multi-agent collaboration. We propose a conceptual framework for ID 4.0 and discuss its potential to support end-to-end automation of engineering design processes through coordinated, autonomous multi-agent-based systems. Furthermore, we discuss future perspectives to enhance and fully realize ID 4.0's potential, including more complex design scenarios, more practical design implementations, novel agent coordination mechanisms, and autonomous design goal-setting with better human value alignment. In sum, these insights lay a foundation for advancing Intelligent Design toward greater adaptivity, autonomy, and effectiveness in addressing increasingly complex design challenges. 

---
# Large Language Models for Design Structure Matrix Optimization 

**Authors**: Shuo Jiang, Min Xie, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09749)  

**Abstract**: In complex engineering systems, the interdependencies among components or development activities are often modeled and analyzed using Design Structure Matrix (DSM). Reorganizing elements within a DSM to minimize feedback loops and enhance modularity or process efficiency constitutes a challenging combinatorial optimization (CO) problem in engineering design and operations. As problem sizes increase and dependency networks become more intricate, traditional optimization methods that solely use mathematical heuristics often fail to capture the contextual nuances and struggle to deliver effective solutions. In this study, we explore the potential of Large Language Models (LLMs) for helping solve such CO problems by leveraging their capabilities for advanced reasoning and contextual understanding. We propose a novel LLM-based framework that integrates network topology with contextual domain knowledge for iterative optimization of DSM element sequencing - a common CO problem. Experiments on various DSM cases show that our method consistently achieves faster convergence and superior solution quality compared to both stochastic and deterministic baselines. Notably, we find that incorporating contextual domain knowledge significantly enhances optimization performance regardless of the chosen LLM backbone. These findings highlight the potential of LLMs to solve complex engineering CO problems by combining semantic and mathematical reasoning. This approach paves the way towards a new paradigm in LLM-based engineering design optimization. 

---
# Feature Engineering for Agents: An Adaptive Cognitive Architecture for Interpretable ML Monitoring 

**Authors**: Gusseppe Bravo-Rocca, Peini Liu, Jordi Guitart, Rodrigo M Carrillo-Larco, Ajay Dholakia, David Ellison  

**Link**: [PDF](https://arxiv.org/pdf/2506.09742)  

**Abstract**: Monitoring Machine Learning (ML) models in production environments is crucial, yet traditional approaches often yield verbose, low-interpretability outputs that hinder effective decision-making. We propose a cognitive architecture for ML monitoring that applies feature engineering principles to agents based on Large Language Models (LLMs), significantly enhancing the interpretability of monitoring outputs. Central to our approach is a Decision Procedure module that simulates feature engineering through three key steps: Refactor, Break Down, and Compile. The Refactor step improves data representation to better capture feature semantics, allowing the LLM to focus on salient aspects of the monitoring data while reducing noise and irrelevant information. Break Down decomposes complex information for detailed analysis, and Compile integrates sub-insights into clear, interpretable outputs. This process leads to a more deterministic planning approach, reducing dependence on LLM-generated planning, which can sometimes be inconsistent and overly general. The combination of feature engineering-driven planning and selective LLM utilization results in a robust decision support system, capable of providing highly interpretable and actionable insights. Experiments using multiple LLMs demonstrate the efficacy of our approach, achieving significantly higher accuracy compared to various baselines across several domains. 

---
# ELBO-T2IAlign: A Generic ELBO-Based Method for Calibrating Pixel-level Text-Image Alignment in Diffusion Models 

**Authors**: Qin Zhou, Zhiyang Zhang, Jinglong Wang, Xiaobin Li, Jing Zhang, Qian Yu, Lu Sheng, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09740)  

**Abstract**: Diffusion models excel at image generation. Recent studies have shown that these models not only generate high-quality images but also encode text-image alignment information through attention maps or loss functions. This information is valuable for various downstream tasks, including segmentation, text-guided image editing, and compositional image generation. However, current methods heavily rely on the assumption of perfect text-image alignment in diffusion models, which is not the case. In this paper, we propose using zero-shot referring image segmentation as a proxy task to evaluate the pixel-level image and class-level text alignment of popular diffusion models. We conduct an in-depth analysis of pixel-text misalignment in diffusion models from the perspective of training data bias. We find that misalignment occurs in images with small sized, occluded, or rare object classes. Therefore, we propose ELBO-T2IAlign, a simple yet effective method to calibrate pixel-text alignment in diffusion models based on the evidence lower bound (ELBO) of likelihood. Our method is training-free and generic, eliminating the need to identify the specific cause of misalignment and works well across various diffusion model architectures. Extensive experiments on commonly used benchmark datasets on image segmentation and generation have verified the effectiveness of our proposed calibration approach. 

---
# Vision Matters: Simple Visual Perturbations Can Boost Multimodal Math Reasoning 

**Authors**: Yuting Li, Lai Wei, Kaipeng Zheng, Jingyuan Huang, Linghe Kong, Lichao Sun, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09736)  

**Abstract**: Despite the rapid progress of multimodal large language models (MLLMs), they have largely overlooked the importance of visual processing. In a simple yet revealing experiment, we interestingly find that language-only models, when provided with image captions, can achieve comparable or even better performance than MLLMs that consume raw visual inputs. This suggests that current MLLMs may generate accurate visual descriptions but fail to effectively integrate them during reasoning. Motivated by this, we propose a simple visual perturbation framework that enhances perceptual robustness without requiring algorithmic modifications or additional training data. Our approach introduces three targeted perturbations: distractor concatenation, dominance-preserving mixup, and random rotation, that can be easily integrated into existing post-training pipelines including SFT, DPO, and GRPO. Through extensive experiments across multiple datasets, we demonstrate consistent improvements in mathematical reasoning performance, with gains comparable to those achieved through algorithmic changes. Additionally, we achieve competitive performance among open-source 7B RL-tuned models by training Qwen2.5-VL-7B with visual perturbation. Through comprehensive ablation studies, we analyze the effectiveness of different perturbation strategies, revealing that each perturbation type contributes uniquely to different aspects of visual reasoning. Our findings highlight the critical role of visual perturbation in multimodal mathematical reasoning: better reasoning begins with better seeing. Our code is available at this https URL. 

---
# AtmosMJ: Revisiting Gating Mechanism for AI Weather Forecasting Beyond the Year Scale 

**Authors**: Minjong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2506.09733)  

**Abstract**: The advent of Large Weather Models (LWMs) has marked a turning point in data-driven forecasting, with many models now outperforming traditional numerical systems in the medium range. However, achieving stable, long-range autoregressive forecasts beyond a few weeks remains a significant challenge. Prevailing state-of-the-art models that achieve year-long stability, such as SFNO and DLWP-HPX, have relied on transforming input data onto non-standard spatial domains like spherical harmonics or HEALPix meshes. This has led to the prevailing assumption that such representations are necessary to enforce physical consistency and long-term stability. This paper challenges that assumption by investigating whether comparable long-range performance can be achieved on the standard latitude-longitude grid. We introduce AtmosMJ, a deep convolutional network that operates directly on ERA5 data without any spherical remapping. The model's stability is enabled by a novel Gated Residual Fusion (GRF) mechanism, which adaptively moderates feature updates to prevent error accumulation over long recursive simulations. Our results demonstrate that AtmosMJ produces stable and physically plausible forecasts for about 500 days. In quantitative evaluations, it achieves competitive 10-day forecast accuracy against models like Pangu-Weather and GraphCast, all while requiring a remarkably low training budget of 5.7 days on a V100 GPU. Our findings suggest that efficient architectural design, rather than non-standard data representation, can be the key to unlocking stable and computationally efficient long-range weather prediction. 

---
# Non-Contact Health Monitoring During Daily Personal Care Routines 

**Authors**: Xulin Ma, Jiankai Tang, Zhang Jiang, Songqin Cheng, Yuanchun Shi, Dong LI, Xin Liu, Daniel McDuff, Xiaojing Liu, Yuntao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09718)  

**Abstract**: Remote photoplethysmography (rPPG) enables non-contact, continuous monitoring of physiological signals and offers a practical alternative to traditional health sensing methods. Although rPPG is promising for daily health monitoring, its application in long-term personal care scenarios, such as mirror-facing routines in high-altitude environments, remains challenging due to ambient lighting variations, frequent occlusions from hand movements, and dynamic facial postures. To address these challenges, we present LADH (Long-term Altitude Daily Health), the first long-term rPPG dataset containing 240 synchronized RGB and infrared (IR) facial videos from 21 participants across five common personal care scenarios, along with ground-truth PPG, respiration, and blood oxygen signals. Our experiments demonstrate that combining RGB and IR video inputs improves the accuracy and robustness of non-contact physiological monitoring, achieving a mean absolute error (MAE) of 4.99 BPM in heart rate estimation. Furthermore, we find that multi-task learning enhances performance across multiple physiological indicators simultaneously. Dataset and code are open at this https URL. 

---
# TRIDENT: Temporally Restricted Inference via DFA-Enhanced Neural Traversal 

**Authors**: Vincenzo Collura, Karim Tit, Laura Bussi, Eleonora Giunchiglia, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2506.09701)  

**Abstract**: Large Language Models (LLMs) and other neural architectures have achieved impressive results across a variety of generative and classification tasks. However, they remain fundamentally ill-equipped to ensure that their outputs satisfy temporal constraints, such as those expressible in Linear Temporal Logic over finite traces (LTLf). In this paper, we introduce TRIDENT: a general and model-agnostic inference-time algorithm that guarantees compliance with such constraints without requiring any retraining. TRIDENT compiles LTLf formulas into a Deterministic Finite Automaton (DFA), which is used to guide a constrained variant of beam search. At each decoding step, transitions that would lead to constraint violations are masked, while remaining paths are dynamically re-ranked based on both the model's probabilities and the DFA's acceptance structure. We formally prove that the resulting sequences are guaranteed to satisfy the given LTLf constraints, and we empirically demonstrate that TRIDENT also improves output quality. We validate our approach on two distinct tasks: temporally constrained image-stream classification and controlled text generation. In both settings, TRIDENT achieves perfect constraint satisfaction, while comparison with the state of the art shows improved efficiency and high standard quality metrics. 

---
# Towards Practical Alzheimer's Disease Diagnosis: A Lightweight and Interpretable Spiking Neural Model 

**Authors**: Changwei Wu, Yifei Chen, Yuxin Du, Jinying Zong, Jie Dong, Mingxuan Liu, Yong Peng, Jin Fan, Feiwei Qin, Changmiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09695)  

**Abstract**: Early diagnosis of Alzheimer's Disease (AD), especially at the mild cognitive impairment (MCI) stage, is vital yet hindered by subjective assessments and the high cost of multimodal imaging modalities. Although deep learning methods offer automated alternatives, their energy inefficiency and computational demands limit real-world deployment, particularly in resource-constrained settings. As a brain-inspired paradigm, spiking neural networks (SNNs) are inherently well-suited for modeling the sparse, event-driven patterns of neural degeneration in AD, offering a promising foundation for interpretable and low-power medical diagnostics. However, existing SNNs often suffer from weak expressiveness and unstable training, which restrict their effectiveness in complex medical tasks. To address these limitations, we propose FasterSNN, a hybrid neural architecture that integrates biologically inspired LIF neurons with region-adaptive convolution and multi-scale spiking attention. This design enables sparse, efficient processing of 3D MRI while preserving diagnostic accuracy. Experiments on benchmark datasets demonstrate that FasterSNN achieves competitive performance with substantially improved efficiency and stability, supporting its potential for practical AD screening. Our source code is available at this https URL. 

---
# Reasoning Models Are More Easily Gaslighted Than You Think 

**Authors**: Bin Zhu, Hailong Yin, Jingjing Chen, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09677)  

**Abstract**: Recent advances in reasoning-centric models promise improved robustness through mechanisms such as chain-of-thought prompting and test-time scaling. However, their ability to withstand misleading user input remains underexplored. In this paper, we conduct a systematic evaluation of three state-of-the-art reasoning models, i.e., OpenAI's o4-mini, Claude-3.7-Sonnet and Gemini-2.5-Flash, across three multimodal benchmarks: MMMU, MathVista, and CharXiv. Our evaluation reveals significant accuracy drops (25-29% on average) following gaslighting negation prompts, indicating that even top-tier reasoning models struggle to preserve correct answers under manipulative user feedback. Built upon the insights of the evaluation and to further probe this vulnerability, we introduce GaslightingBench-R, a new diagnostic benchmark specifically designed to evaluate reasoning models' susceptibility to defend their belief under gaslighting negation prompt. Constructed by filtering and curating 1,025 challenging samples from the existing benchmarks, GaslightingBench-R induces even more dramatic failures, with accuracy drops exceeding 53% on average. Our findings reveal fundamental limitations in the robustness of reasoning models, highlighting the gap between step-by-step reasoning and belief persistence. 

---
# Is Fine-Tuning an Effective Solution? Reassessing Knowledge Editing for Unstructured Data 

**Authors**: Hao Xiong, Chuanyuan Tan, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09672)  

**Abstract**: Unstructured Knowledge Editing (UKE) is crucial for updating the relevant knowledge of large language models (LLMs). It focuses on unstructured inputs, such as long or free-form texts, which are common forms of real-world knowledge. Although previous studies have proposed effective methods and tested them, some issues exist: (1) Lack of Locality evaluation for UKE, and (2) Abnormal failure of fine-tuning (FT) based methods for UKE. To address these issues, we first construct two datasets, UnKEBench-Loc and AKEW-Loc (CF), by extending two existing UKE datasets with locality test data from the unstructured and structured views. This enables a systematic evaluation of the Locality of post-edited models. Furthermore, we identify four factors that may affect the performance of FT-based methods. Based on these factors, we conduct experiments to determine how the well-performing FT-based methods should be trained for the UKE task, providing a training recipe for future research. Our experimental results indicate that the FT-based method with the optimal setting (FT-UKE) is surprisingly strong, outperforming the existing state-of-the-art (SOTA). In batch editing scenarios, FT-UKE shows strong performance as well, with its advantage over SOTA methods increasing as the batch size grows, expanding the average metric lead from +6.78% to +10.80% 

---
# Empirical Quantification of Spurious Correlations in Malware Detection 

**Authors**: Bianca Perasso, Ludovico Lozza, Andrea Ponte, Luca Demetrio, Luca Oneto, Fabio Roli  

**Link**: [PDF](https://arxiv.org/pdf/2506.09662)  

**Abstract**: End-to-end deep learning exhibits unmatched performance for detecting malware, but such an achievement is reached by exploiting spurious correlations -- features with high relevance at inference time, but known to be useless through domain knowledge. While previous work highlighted that deep networks mainly focus on metadata, none investigated the phenomenon further, without quantifying their impact on the decision. In this work, we deepen our understanding of how spurious correlation affects deep learning for malware detection by highlighting how much models rely on empty spaces left by the compiler, which diminishes the relevance of the compiled code. Through our seminal analysis on a small-scale balanced dataset, we introduce a ranking of two end-to-end models to better understand which is more suitable to be put in production. 

---
# DGAE: Diffusion-Guided Autoencoder for Efficient Latent Representation Learning 

**Authors**: Dongxu Liu, Yuang Peng, Haomiao Tang, Yuwei Chen, Chunrui Han, Zheng Ge, Daxin Jiang, Mingxue Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09644)  

**Abstract**: Autoencoders empower state-of-the-art image and video generative models by compressing pixels into a latent space through visual tokenization. Although recent advances have alleviated the performance degradation of autoencoders under high compression ratios, addressing the training instability caused by GAN remains an open challenge. While improving spatial compression, we also aim to minimize the latent space dimensionality, enabling more efficient and compact representations. To tackle these challenges, we focus on improving the decoder's expressiveness. Concretely, we propose DGAE, which employs a diffusion model to guide the decoder in recovering informative signals that are not fully decoded from the latent representation. With this design, DGAE effectively mitigates the performance degradation under high spatial compression rates. At the same time, DGAE achieves state-of-the-art performance with a 2x smaller latent space. When integrated with Diffusion Models, DGAE demonstrates competitive performance on image generation for ImageNet-1K and shows that this compact latent representation facilitates faster convergence of the diffusion model. 

---
# HSENet: Hybrid Spatial Encoding Network for 3D Medical Vision-Language Understanding 

**Authors**: Yanzhao Shi, Xiaodan Zhang, Junzhong Ji, Haoning Jiang, Chengxin Zheng, Yinong Wang, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09634)  

**Abstract**: Automated 3D CT diagnosis empowers clinicians to make timely, evidence-based decisions by enhancing diagnostic accuracy and workflow efficiency. While multimodal large language models (MLLMs) exhibit promising performance in visual-language understanding, existing methods mainly focus on 2D medical images, which fundamentally limits their ability to capture complex 3D anatomical structures. This limitation often leads to misinterpretation of subtle pathologies and causes diagnostic hallucinations. In this paper, we present Hybrid Spatial Encoding Network (HSENet), a framework that exploits enriched 3D medical visual cues by effective visual perception and projection for accurate and robust vision-language understanding. Specifically, HSENet employs dual-3D vision encoders to perceive both global volumetric contexts and fine-grained anatomical details, which are pre-trained by dual-stage alignment with diagnostic reports. Furthermore, we propose Spatial Packer, an efficient multimodal projector that condenses high-resolution 3D spatial regions into a compact set of informative visual tokens via centroid-based compression. By assigning spatial packers with dual-3D vision encoders, HSENet can seamlessly perceive and transfer hybrid visual representations to LLM's semantic space, facilitating accurate diagnostic text generation. Experimental results demonstrate that our method achieves state-of-the-art performance in 3D language-visual retrieval (39.85% of R@100, +5.96% gain), 3D medical report generation (24.01% of BLEU-4, +8.01% gain), and 3D visual question answering (73.60% of Major Class Accuracy, +1.99% gain), confirming its effectiveness. Our code is available at this https URL. 

---
# Effective Red-Teaming of Policy-Adherent Agents 

**Authors**: Itay Nakash, George Kour, Koren Lazar, Matan Vetzler, Guy Uziel, Ateret Anaby-Tavor  

**Link**: [PDF](https://arxiv.org/pdf/2506.09600)  

**Abstract**: Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks 

---
# From Symbolic to Neural and Back: Exploring Knowledge Graph-Large Language Model Synergies 

**Authors**: Blaž Škrlj, Boshko Koloski, Senja Pollak, Nada Lavrač  

**Link**: [PDF](https://arxiv.org/pdf/2506.09566)  

**Abstract**: Integrating structured knowledge from Knowledge Graphs (KGs) into Large Language Models (LLMs) enhances factual grounding and reasoning capabilities. This survey paper systematically examines the synergy between KGs and LLMs, categorizing existing approaches into two main groups: KG-enhanced LLMs, which improve reasoning, reduce hallucinations, and enable complex question answering; and LLM-augmented KGs, which facilitate KG construction, completion, and querying. Through comprehensive analysis, we identify critical gaps and highlight the mutual benefits of structured knowledge integration. Compared to existing surveys, our study uniquely emphasizes scalability, computational efficiency, and data quality. Finally, we propose future research directions, including neuro-symbolic integration, dynamic KG updating, data reliability, and ethical considerations, paving the way for intelligent systems capable of managing more complex real-world knowledge tasks. 

---
# AD^2-Bench: A Hierarchical CoT Benchmark for MLLM in Autonomous Driving under Adverse Conditions 

**Authors**: Zhaoyang Wei, Chenhui Qiang, Bowen Jiang, Xumeng Han, Xuehui Yu, Zhenjun Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.09557)  

**Abstract**: Chain-of-Thought (CoT) reasoning has emerged as a powerful approach to enhance the structured, multi-step decision-making capabilities of Multi-Modal Large Models (MLLMs), is particularly crucial for autonomous driving with adverse weather conditions and complex traffic environments. However, existing benchmarks have largely overlooked the need for rigorous evaluation of CoT processes in these specific and challenging scenarios. To address this critical gap, we introduce AD^2-Bench, the first Chain-of-Thought benchmark specifically designed for autonomous driving with adverse weather and complex scenes. AD^2-Bench is meticulously constructed to fulfill three key criteria: comprehensive data coverage across diverse adverse environments, fine-grained annotations that support multi-step reasoning, and a dedicated evaluation framework tailored for assessing CoT performance. The core contribution of AD^2-Bench is its extensive collection of over 5.4k high-quality, manually annotated CoT instances. Each intermediate reasoning step in these annotations is treated as an atomic unit with explicit ground truth, enabling unprecedented fine-grained analysis of MLLMs' inferential processes under text-level, point-level, and region-level visual prompts. Our comprehensive evaluation of state-of-the-art MLLMs on AD^2-Bench reveals accuracy below 60%, highlighting the benchmark's difficulty and the need to advance robust, interpretable end-to-end autonomous driving systems. AD^2-Bench thus provides a standardized evaluation platform, driving research forward by improving MLLMs' reasoning in autonomous driving, making it an invaluable resource. 

---
# Tightly-Coupled LiDAR-IMU-Leg Odometry with Online Learned Leg Kinematics Incorporating Foot Tactile Information 

**Authors**: Taku Okawara, Kenji Koide, Aoki Takanose, Shuji Oishi, Masashi Yokozuka, Kentaro Uno, Kazuya Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2506.09548)  

**Abstract**: In this letter, we present tightly coupled LiDAR-IMU-leg odometry, which is robust to challenging conditions such as featureless environments and deformable terrains. We developed an online learning-based leg kinematics model named the neural leg kinematics model, which incorporates tactile information (foot reaction force) to implicitly express the nonlinear dynamics between robot feet and the ground. Online training of this model enhances its adaptability to weight load changes of a robot (e.g., assuming delivery or transportation tasks) and terrain conditions. According to the \textit{neural adaptive leg odometry factor} and online uncertainty estimation of the leg kinematics model-based motion predictions, we jointly solve online training of this kinematics model and odometry estimation on a unified factor graph to retain the consistency of both. The proposed method was verified through real experiments using a quadruped robot in two challenging situations: 1) a sandy beach, representing an extremely featureless area with a deformable terrain, and 2) a campus, including multiple featureless areas and terrain types of asphalt, gravel (deformable terrain), and grass. Experimental results showed that our odometry estimation incorporating the \textit{neural leg kinematics model} outperforms state-of-the-art works. Our project page is available for further details: this https URL 

---
# Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models 

**Authors**: Shuai Wang, Zhenhua Liu, Jiaheng Wei, Xuanwu Yin, Dong Li, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.09532)  

**Abstract**: We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks. 

---
# Neural Functions for Learning Periodic Signal 

**Authors**: Woojin Cho, Minju Jo, Kookjin Lee, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.09526)  

**Abstract**: As function approximators, deep neural networks have served as an effective tool to represent various signal types. Recent approaches utilize multi-layer perceptrons (MLPs) to learn a nonlinear mapping from a coordinate to its corresponding signal, facilitating the learning of continuous neural representations from discrete data points. Despite notable successes in learning diverse signal types, coordinate-based MLPs often face issues of overfitting and limited generalizability beyond the training region, resulting in subpar extrapolation performance. This study addresses scenarios where the underlying true signals exhibit periodic properties, either spatially or temporally. We propose a novel network architecture, which extracts periodic patterns from measurements and leverages this information to represent the signal, thereby enhancing generalization and improving extrapolation performance. We demonstrate the efficacy of the proposed method through comprehensive experiments, including the learning of the periodic solutions for differential equations, and time series imputation (interpolation) and forecasting (extrapolation) on real-world datasets. 

---
# Revisit What You See: Disclose Language Prior in Vision Tokens for Efficient Guided Decoding of LVLMs 

**Authors**: Beomsik Cho, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09522)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across various multimodal tasks by integrating visual perception with language understanding. However, conventional decoding strategies of LVLMs often fail to successfully utilize visual information, leading to visually ungrounded responses. While various approaches have been proposed to address this limitation, they typically require additional training, multi-step inference procedures, or external model dependencies. This paper introduces ReVisiT, a simple yet effective decoding method that references vision tokens to guide the text generation process in LVLMs. Our approach leverages the semantic information embedded within vision tokens by projecting them into the text token distribution space, and dynamically selecting the most relevant vision token at each decoding step through constrained divergence minimization. This selected vision token is then used to refine the output distribution to better incorporate visual semantics. Experiments on three LVLM hallucination benchmarks with two recent LVLMs demonstrate that ReVisiT consistently enhances visual grounding with minimal computational overhead. Moreover, our method achieves competitive or superior results relative to state-of-the-art baselines while reducing computational costs for up to $2\times$. 

---
# How attention simplifies mental representations for planning 

**Authors**: Jason da Silva Castanheira, Nicholas Shea, Stephen M. Fleming  

**Link**: [PDF](https://arxiv.org/pdf/2506.09520)  

**Abstract**: Human planning is efficient -- it frugally deploys limited cognitive resources to accomplish difficult tasks -- and flexible -- adapting to novel problems and environments. Computational approaches suggest that people construct simplified mental representations of their environment, balancing the complexity of a task representation with its utility. These models imply a nested optimisation in which planning shapes perception, and perception shapes planning -- but the perceptual and attentional mechanisms governing how this interaction unfolds remain unknown. Here, we harness virtual maze navigation to characterise how spatial attention controls which aspects of a task representation enter subjective awareness and are available for planning. We find that spatial proximity governs which aspects of a maze are available for planning, and that when task-relevant information follows natural (lateralised) contours of attention, people can more easily construct simplified and useful maze representations. This influence of attention varies considerably across individuals, explaining differences in people's task representations and behaviour. Inspired by the 'spotlight of attention' analogy, we incorporate the effects of visuospatial attention into existing computational accounts of value-guided construal. Together, our work bridges computational perspectives on perception and decision-making to better understand how individuals represent their environments in aid of planning. 

---
# ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning 

**Authors**: Yu Sun, Xingyu Qian, Weiwen Xu, Hao Zhang, Chenghao Xiao, Long Li, Yu Rong, Wenbing Huang, Qifeng Bai, Tingyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09513)  

**Abstract**: Though reasoning-based large language models (LLMs) have excelled in mathematics and programming, their capabilities in knowledge-intensive medical question answering remain underexplored. To address this, we introduce ReasonMed, the largest medical reasoning dataset, comprising 370k high-quality examples distilled from 1.7 million initial reasoning paths generated by various LLMs. ReasonMed is constructed through a \textit{multi-agent verification and refinement process}, where we design an \textit{Error Refiner} to enhance the reasoning paths by identifying and correcting error-prone steps flagged by a verifier. Leveraging ReasonMed, we systematically investigate best practices for training medical reasoning models and find that combining detailed Chain-of-Thought (CoT) reasoning with concise answer summaries yields the most effective fine-tuning strategy. Based on this strategy, we train ReasonMed-7B, which sets a new benchmark for sub-10B models, outperforming the prior best by 4.17\% and even exceeding LLaMA3.1-70B on PubMedQA by 4.60\%. 

---
# Efficient Preference-Based Reinforcement Learning: Randomized Exploration Meets Experimental Design 

**Authors**: Andreas Schlaginhaufen, Reda Ouhamma, Maryam Kamgarpour  

**Link**: [PDF](https://arxiv.org/pdf/2506.09508)  

**Abstract**: We study reinforcement learning from human feedback in general Markov decision processes, where agents learn from trajectory-level preference comparisons. A central challenge in this setting is to design algorithms that select informative preference queries to identify the underlying reward while ensuring theoretical guarantees. We propose a meta-algorithm based on randomized exploration, which avoids the computational challenges associated with optimistic approaches and remains tractable. We establish both regret and last-iterate guarantees under mild reinforcement learning oracle assumptions. To improve query complexity, we introduce and analyze an improved algorithm that collects batches of trajectory pairs and applies optimal experimental design to select informative comparison queries. The batch structure also enables parallelization of preference queries, which is relevant in practical deployment as feedback can be gathered concurrently. Empirical evaluation confirms that the proposed method is competitive with reward-based reinforcement learning while requiring a small number of preference queries. 

---
# TransXSSM: A Hybrid Transformer State Space Model with Unified Rotary Position Embedding 

**Authors**: Bingheng Wu, Jingze Shi, Yifan Wu, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09507)  

**Abstract**: Transformers exhibit proficiency in capturing long-range dependencies, whereas State Space Models (SSMs) facilitate linear-time sequence modeling. Notwithstanding their synergistic potential, the integration of these architectures presents a significant challenge, primarily attributable to a fundamental incongruity in their respective positional encoding mechanisms: Transformers rely on explicit Rotary Position Embeddings (RoPE), while SSMs leverage implicit positional representations via convolutions. This divergence often precipitates discontinuities and suboptimal performance. To address this impediment, we propose a unified rotary position embedding (\textbf{\ourRoPE}) methodology, thereby establishing a consistent positional encoding framework for both self-attention and state-space components. Using this \ourRoPE, we introduce \textbf{\model}, a hybrid architecture that coherently integrates the Transformer and SSM layers under this unified positional encoding scheme. At a 4K sequence length, \model exhibits training and inference speeds that are \textbf{42.3\% and 29.5\% faster}, respectively, relative to standard Transformer models. It also delivers higher accuracy: under comparable settings, it surpasses a Transformer baseline by over 4\% on language modeling benchmarks. \model furthermore scales more effectively: \model-1.3B gains \textbf{7.22\%} in average accuracy over its 320M version (versus about 6\% gains for equivalent Transformers or SSMs). Our results show that unified positional encoding resolves positional incompatibility in hybrid models, enabling efficient, high-performance long-context modeling. 

---
# A Unified Theory of Compositionality, Modularity, and Interpretability in Markov Decision Processes 

**Authors**: Thomas J. Ringstrom, Paul R. Schrater  

**Link**: [PDF](https://arxiv.org/pdf/2506.09499)  

**Abstract**: We introduce Option Kernel Bellman Equations (OKBEs) for a new reward-free Markov Decision Process. Rather than a value function, OKBEs directly construct and optimize a predictive map called a state-time option kernel (STOK) to maximize the probability of completing a goal while avoiding constraint violations. STOKs are compositional, modular, and interpretable initiation-to-termination transition kernels for policies in the Options Framework of Reinforcement Learning. This means: 1) STOKs can be composed using Chapman-Kolmogorov equations to make spatiotemporal predictions for multiple policies over long horizons, 2) high-dimensional STOKs can be represented and computed efficiently in a factorized and reconfigurable form, and 3) STOKs record the probabilities of semantically interpretable goal-success and constraint-violation events, needed for formal verification. Given a high-dimensional state-transition model for an intractable planning problem, we can decompose it with local STOKs and goal-conditioned policies that are aggregated into a factorized goal kernel, making it possible to forward-plan at the level of goals in high-dimensions to solve the problem. These properties lead to highly flexible agents that can rapidly synthesize meta-policies, reuse planning representations across many tasks, and justify goals using empowerment, an intrinsic motivation function. We argue that reward-maximization is in conflict with the properties of compositionality, modularity, and interpretability. Alternatively, OKBEs facilitate these properties to support verifiable long-horizon planning and intrinsic motivation that scales to dynamic high-dimensional world-models. 

---
# EnerBridge-DPO: Energy-Guided Protein Inverse Folding with Markov Bridges and Direct Preference Optimization 

**Authors**: Dingyi Rong, Haotian Lu, Wenzhuo Zheng, Fan Zhang, Shuangjia Zheng, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09496)  

**Abstract**: Designing protein sequences with optimal energetic stability is a key challenge in protein inverse folding, as current deep learning methods are primarily trained by maximizing sequence recovery rates, often neglecting the energy of the generated sequences. This work aims to overcome this limitation by developing a model that directly generates low-energy, stable protein sequences. We propose EnerBridge-DPO, a novel inverse folding framework focused on generating low-energy, high-stability protein sequences. Our core innovation lies in: First, integrating Markov Bridges with Direct Preference Optimization (DPO), where energy-based preferences are used to fine-tune the Markov Bridge model. The Markov Bridge initiates optimization from an information-rich prior sequence, providing DPO with a pool of structurally plausible sequence candidates. Second, an explicit energy constraint loss is introduced, which enhances the energy-driven nature of DPO based on prior sequences, enabling the model to effectively learn energy representations from a wealth of prior knowledge and directly predict sequence energy values, thereby capturing quantitative features of the energy landscape. Our evaluations demonstrate that EnerBridge-DPO can design protein complex sequences with lower energy while maintaining sequence recovery rates comparable to state-of-the-art models, and accurately predicts $\Delta \Delta G$ values between various sequences. 

---
# BemaGANv2: A Tutorial and Comparative Survey of GAN-based Vocoders for Long-Term Audio Generation 

**Authors**: Taesoo Park, Mungwi Jeong, Mingyu Park, Narae Kim, Junyoung Kim, Mujung Kim, Jisang Yoo, Hoyun Lee, Sanghoon Kim, Soonchul Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.09487)  

**Abstract**: This paper presents a tutorial-style survey and implementation guide of BemaGANv2, an advanced GAN-based vocoder designed for high-fidelity and long-term audio generation. Built upon the original BemaGAN architecture, BemaGANv2 incorporates major architectural innovations by replacing traditional ResBlocks in the generator with the Anti-aliased Multi-Periodicity composition (AMP) module, which internally applies the Snake activation function to better model periodic structures. In the discriminator framework, we integrate the Multi-Envelope Discriminator (MED), a novel architecture we originally proposed, to extract rich temporal envelope features crucial for periodicity detection. Coupled with the Multi-Resolution Discriminator (MRD), this combination enables more accurate modeling of long-range dependencies in audio. We systematically evaluate various discriminator configurations, including MSD + MED, MSD + MRD, and MPD + MED + MRD, using objective metrics (FAD, SSIM, PLCC, MCD) and subjective evaluations (MOS, SMOS). This paper also provides a comprehensive tutorial on the model architecture, training methodology, and implementation to promote reproducibility. The code and pre-trained models are available at: this https URL. 

---
# Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation 

**Authors**: Yuxin Liu, Zhenghao Peng, Xuanhao Cui, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09485)  

**Abstract**: Scenario-based testing is essential for validating the performance of autonomous driving (AD) systems. However, such testing is limited by the scarcity of long-tailed, safety-critical scenarios in existing datasets collected in the real world. To tackle the data issue, we propose the Adv-BMT framework, which augments real-world scenarios with diverse and realistic adversarial interactions. The core component of Adv-BMT is a bidirectional motion transformer (BMT) model to perform inverse traffic motion predictions, which takes agent information in the last time step of the scenario as input, and reconstruct the traffic in the inverse of chronological order until the initial time step. The Adv-BMT framework is a two-staged pipeline: it first conducts adversarial initializations and then inverse motion predictions. Different from previous work, we do not need any collision data for pretraining, and are able to generate realistic and diverse collision interactions. Our experimental results validate the quality of generated collision scenarios by Adv-BMT: training in our augmented dataset would reduce episode collision rates by 20\% compared to previous work. 

---
# Abstraction-Based Proof Production in Formal Verification of Neural Networks 

**Authors**: Yizhak Yisrael Elboher, Omri Isac, Guy Katz, Tobias Ladner, Haoze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09455)  

**Abstract**: Modern verification tools for deep neural networks (DNNs) increasingly rely on abstraction to scale to realistic architectures. In parallel, proof production is becoming a critical requirement for increasing the reliability of DNN verification results. However, current proofproducing verifiers do not support abstraction-based reasoning, creating a gap between scalability and provable guarantees. We address this gap by introducing a novel framework for proof-producing abstraction-based DNN verification. Our approach modularly separates the verification task into two components: (i) proving the correctness of an abstract network, and (ii) proving the soundness of the abstraction with respect to the original DNN. The former can be handled by existing proof-producing verifiers, whereas we propose the first method for generating formal proofs for the latter. This preliminary work aims to enable scalable and trustworthy verification by supporting common abstraction techniques within a formal proof framework. 

---
# UniToMBench: Integrating Perspective-Taking to Improve Theory of Mind in LLMs 

**Authors**: Prameshwar Thiyagarajan, Vaishnavi Parimi, Shamant Sai, Soumil Garg, Zhangir Meirbek, Nitin Yarlagadda, Kevin Zhu, Chris Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09450)  

**Abstract**: Theory of Mind (ToM), the ability to understand the mental states of oneself and others, remains a challenging area for large language models (LLMs), which often fail to predict human mental states accurately. In this paper, we introduce UniToMBench, a unified benchmark that integrates the strengths of SimToM and TOMBENCH to systematically improve and assess ToM capabilities in LLMs by integrating multi-interaction task designs and evolving story scenarios. Supported by a custom dataset of over 1,000 hand-written scenarios, UniToMBench combines perspective-taking techniques with diverse evaluation metrics to better stimulate social cognition in LLMs. Through evaluation, we observe that while models like GPT-4o and GPT-4o Mini show consistently high accuracy in tasks involving emotional and belief-related scenarios, with results usually above 80%, there is significant variability in their performance across knowledge-based tasks. These results highlight both the strengths and limitations of current LLMs in ToM-related tasks, underscoring the value of UniToMBench as a comprehensive tool for future development. Our code is publicly available here: this https URL. 

---
# TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision 

**Authors**: Ayush Gupta, Anirban Roy, Rama Chellappa, Nathaniel D. Bastian, Alvaro Velasquez, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2506.09445)  

**Abstract**: We address the problem of video question answering (video QA) with temporal grounding in a weakly supervised setup, without any temporal annotations. Given a video and a question, we generate an open-ended answer grounded with the start and end time. For this task, we propose TOGA: a vision-language model for Temporally Grounded Open-Ended Video QA with Weak Supervision. We instruct-tune TOGA to jointly generate the answer and the temporal grounding. We operate in a weakly supervised setup where the temporal grounding annotations are not available. We generate pseudo labels for temporal grounding and ensure the validity of these labels by imposing a consistency constraint between the question of a grounding response and the response generated by a question referring to the same temporal segment. We notice that jointly generating the answers with the grounding improves performance on question answering as well as grounding. We evaluate TOGA on grounded QA and open-ended QA tasks. For grounded QA, we consider the NExT-GQA benchmark which is designed to evaluate weakly supervised grounded question answering. For open-ended QA, we consider the MSVD-QA and ActivityNet-QA benchmarks. We achieve state-of-the-art performance for both tasks on these benchmarks. 

---
# GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture 

**Authors**: GigaChat team, Mamedov Valentin, Evgenii Kosarev, Gregory Leleytner, Ilya Shchuckin, Valeriy Berezovskiy, Daniil Smirnov, Dmitry Kozlov, Sergei Averkiev, Lukyanenko Ivan, Aleksandr Proshunin, Ainur Israfilova, Ivan Baskov, Artem Chervyakov, Emil Shakirov, Mikhail Kolesov, Daria Khomich, Darya Latortseva, Sergei Porkhun, Yury Fedorov, Oleg Kutuzov, Polina Kudriavtseva, Sofiia Soldatova, Kolodin Egor, Stanislav Pyatkin, Dzmitry Menshykh, Grafov Sergei, Eldar Damirov, Karlov Vladimir, Ruslan Gaitukiev, Arkadiy Shatenov, Alena Fenogenova, Nikita Savushkin, Fedor Minkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09440)  

**Abstract**: Generative large language models (LLMs) have become crucial for modern NLP research and applications across various languages. However, the development of foundational models specifically tailored to the Russian language has been limited, primarily due to the significant computational resources required. This paper introduces the GigaChat family of Russian LLMs, available in various sizes, including base models and instruction-tuned versions. We provide a detailed report on the model architecture, pre-training process, and experiments to guide design choices. In addition, we evaluate their performance on Russian and English benchmarks and compare GigaChat with multilingual analogs. The paper presents a system demonstration of the top-performing models accessible via an API, a Telegram bot, and a Web interface. Furthermore, we have released three open GigaChat models in open-source (this https URL), aiming to expand NLP research opportunities and support the development of industrial solutions for the Russian language. 

---
# When Is Diversity Rewarded in Cooperative Multi-Agent Learning? 

**Authors**: Michael Amir, Matteo Bettini, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2506.09434)  

**Abstract**: The success of teams in robotics, nature, and society often depends on the division of labor among diverse specialists; however, a principled explanation for when such diversity surpasses a homogeneous team is still missing. Focusing on multi-agent task allocation problems, our goal is to study this question from the perspective of reward design: what kinds of objectives are best suited for heterogeneous teams? We first consider an instantaneous, non-spatial setting where the global reward is built by two generalized aggregation operators: an inner operator that maps the $N$ agents' effort allocations on individual tasks to a task score, and an outer operator that merges the $M$ task scores into the global team reward. We prove that the curvature of these operators determines whether heterogeneity can increase reward, and that for broad reward families this collapses to a simple convexity test. Next, we ask what incentivizes heterogeneity to emerge when embodied, time-extended agents must learn an effort allocation policy. To study heterogeneity in such settings, we use multi-agent reinforcement learning (MARL) as our computational paradigm, and introduce Heterogeneous Environment Design (HED), a gradient-based algorithm that optimizes the parameter space of underspecified MARL environments to find scenarios where heterogeneity is advantageous. Experiments in matrix games and an embodied Multi-Goal-Capture environment show that, despite the difference in settings, HED rediscovers the reward regimes predicted by our theory to maximize the advantage of heterogeneity, both validating HED and connecting our theoretical insights to reward design in MARL. Together, these results help us understand when behavioral diversity delivers a measurable benefit. 

---
# Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting 

**Authors**: Fei Ding, Baiqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09428)  

**Abstract**: Supervised Fine-Tuning (SFT), while enhancing large language models(LLMs)' instruction-following capabilities and domain-specific task adaptability, often diminishes their general capabilities. Moreover, due to the inaccessibility of original pre-training data, catastrophic forgetting tends to be exacerbated when third-party practitioners implement SFT on open-sourced models. To address this challenge, we propose a novel, more cost-effective SFT method which could effectively reduce the risk of catastrophic forgetting without access to original SFT data. Our approach begins by reconstructing the likely SFT instruction distribution of the base model, followed by a multi-model screening process to select optimal data, which is then mixed with new data for SFT. Experimental results demonstrate that our method preserves generalization capabilities in general domains while improving task-specific performance. 

---
# A High-Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation 

**Authors**: Yukang Feng, Jianwen Sun, Chuanhao Li, Zizhen Li, Jiaxin Ai, Fanrui Zhang, Yifan Chang, Sizhuo Zhou, Shenglin Zhang, Yu Dai, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09427)  

**Abstract**: Recent advancements in Large Multimodal Models (LMMs) have significantly improved multimodal understanding and generation. However, these models still struggle to generate tightly interleaved image-text outputs, primarily due to the limited scale, quality and instructional richness of current training datasets. To address this, we introduce InterSyn, a large-scale multimodal dataset constructed using our Self-Evaluation with Iterative Refinement (SEIR) method. InterSyn features multi-turn, instruction-driven dialogues with tightly interleaved imagetext responses, providing rich object diversity and rigorous automated quality refinement, making it well-suited for training next-generation instruction-following LMMs. Furthermore, to address the lack of reliable evaluation tools capable of assessing interleaved multimodal outputs, we introduce SynJudge, an automatic evaluation model designed to quantitatively assess multimodal outputs along four dimensions: text content, image content, image quality, and image-text synergy.
Experimental studies show that the SEIR method leads to substantially higher dataset quality compared to an otherwise identical process without refinement.
Moreover, LMMs trained on InterSyn achieve uniform performance gains across all evaluation metrics, confirming InterSyn's utility for advancing multimodal systems. 

---
# Synthetic Human Action Video Data Generation with Pose Transfer 

**Authors**: Vaclav Knapp, Matyas Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2506.09411)  

**Abstract**: In video understanding tasks, particularly those involving human motion, synthetic data generation often suffers from uncanny features, diminishing its effectiveness for training. Tasks such as sign language translation, gesture recognition, and human motion understanding in autonomous driving have thus been unable to exploit the full potential of synthetic data. This paper proposes a method for generating synthetic human action video data using pose transfer (specifically, controllable 3D Gaussian avatar models). We evaluate this method on the Toyota Smarthome and NTU RGB+D datasets and show that it improves performance in action recognition tasks. Moreover, we demonstrate that the method can effectively scale few-shot datasets, making up for groups underrepresented in the real training data and adding diverse backgrounds. We open-source the method along with RANDOM People, a dataset with videos and avatars of novel human identities for pose transfer crowd-sourced from the internet. 

---
# Token Constraint Decoding Improves Robustness on Question Answering for Large Language Models 

**Authors**: Jui-Ming Yao, Hao-Yuan Chen, Zi-Xian Tang, Bing-Jia Tan, Sheng-Wei Peng, Bing-Cheng Xie, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.09408)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance on multiple-choice question answering (MCQA) benchmarks, yet they remain highly vulnerable to minor input perturbations. In this paper, we introduce and evaluate Token Constraint Decoding (TCD). This simple yet effective inference-time algorithm enforces alignment between token-level predictions to enhance robustness in noisy settings. Through extensive experiments on CommonsenseQA, MMLU, and MMLU-Pro, we show that TCD, especially when paired with prompt engineering (PE) fixes, significantly restores performance degraded by input noise, yielding up to +39\% absolute gains for weaker models like Gemma3 1B. Penalty sweep analyses further reveal that TCD implicitly regularizes overconfident outputs, with different models requiring distinct penalty schedules to maximize resilience. Our findings establish TCD as a practical, model-agnostic approach for improving reasoning stability under real-world imperfections and pave the way for more reliable deployment of LLMs in safety-critical or user-facing applications. 

---
# SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving 

**Authors**: Xiangchen Li, Dimitrios Spatharakis, Saeid Ghafouri, Jiakun Fan, Dimitrios Nikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09397)  

**Abstract**: Regardless the advancements in device capabilities, efficient inferencing advanced large language models (LLMs) at the edge remains challenging due to limited device memory and power constraints. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new approach that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose SLED, a method that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server efficiently batches and verifies the tokens utilizing a more precise target model. This approach supports device heterogeneity and reduces server-side memory footprint by avoiding the need to deploy multiple target models. Our initial experiments with Jetson Orin Nano, Raspberry Pi 5, and an RTX 6000 edge server indicate substantial benefits: significantly reduced latency, improved energy efficiency, and increased concurrent inference sessions, all without sacrificing model accuracy. 

---
# Reasoning as a Resource: Optimizing Fast and Slow Thinking in Code Generation Models 

**Authors**: Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09396)  

**Abstract**: This position paper proposes a fundamental shift in designing code generation models: treating reasoning depth as a controllable resource. Rather than being an incidental byproduct of prompting, we argue that the trade-off between rapid, direct answers ("fast thinking") and elaborate, chain-of-thought deliberation ("slow thinking") must be explicitly managed. We contend that optimizing reasoning budgets across the entire model lifecycle - from synthetic data creation and benchmarking to real-world deploymen - can unlock superior trade-offs among accuracy, latency, and cost. This paper outlines how adaptive control over reasoning can enrich supervision signals, motivate new multi-dimensional benchmarks, and inform cost-aware, security-conscious deployment policies. By viewing fast and slow thinking as complementary modes to be scheduled, we envision coding agents that think deep when necessary and act fast when possible. 

---
# Bipedal Balance Control with Whole-body Musculoskeletal Standing and Falling Simulations 

**Authors**: Chengtian Ma, Yunyue Wei, Chenhui Zuo, Chen Zhang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.09383)  

**Abstract**: Balance control is important for human and bipedal robotic systems. While dynamic balance during locomotion has received considerable attention, quantitative understanding of static balance and falling remains limited. This work presents a hierarchical control pipeline for simulating human balance via a comprehensive whole-body musculoskeletal system. We identified spatiotemporal dynamics of balancing during stable standing, revealed the impact of muscle injury on balancing behavior, and generated fall contact patterns that aligned with clinical data. Furthermore, our simulated hip exoskeleton assistance demonstrated improvement in balance maintenance and reduced muscle effort under perturbation. This work offers unique muscle-level insights into human balance dynamics that are challenging to capture experimentally. It could provide a foundation for developing targeted interventions for individuals with balance impairments and support the advancement of humanoid robotic systems. 

---
# LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization 

**Authors**: Jiaqi Tang, Yu Xia, Yi-Feng Wu, Yuwei Hu, Yuhui Chen, Qing-Guo Chen, Xiaogang Xu, Xiangyu Wu, Hao Lu, Yanqing Ma, Shiyin Lu, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09373)  

**Abstract**: The advent of autonomous agents is transforming interactions with Graphical User Interfaces (GUIs) by employing natural language as a powerful intermediary. Despite the predominance of Supervised Fine-Tuning (SFT) methods in current GUI agents for achieving spatial localization, these methods face substantial challenges due to their limited capacity to accurately perceive positional data. Existing strategies, such as reinforcement learning, often fail to assess positional accuracy effectively, thereby restricting their utility. In response, we introduce Location Preference Optimization (LPO), a novel approach that leverages locational data to optimize interaction preferences. LPO uses information entropy to predict interaction positions by focusing on zones rich in information. Besides, it further introduces a dynamic location reward function based on physical distance, reflecting the varying importance of interaction positions. Supported by Group Relative Preference Optimization (GRPO), LPO facilitates an extensive exploration of GUI environments and significantly enhances interaction precision. Comprehensive experiments demonstrate LPO's superior performance, achieving SOTA results across both offline benchmarks and real-world online evaluations. Our code will be made publicly available soon, at this https URL. 

---
# Anomaly Detection and Generation with Diffusion Models: A Survey 

**Authors**: Yang Liu, Jing Liu, Chengfang Li, Rui Xi, Wenchao Li, Liang Cao, Jin Wang, Laurence T. Yang, Junsong Yuan, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09368)  

**Abstract**: Anomaly detection (AD) plays a pivotal role across diverse domains, including cybersecurity, finance, healthcare, and industrial manufacturing, by identifying unexpected patterns that deviate from established norms in real-world data. Recent advancements in deep learning, specifically diffusion models (DMs), have sparked significant interest due to their ability to learn complex data distributions and generate high-fidelity samples, offering a robust framework for unsupervised AD. In this survey, we comprehensively review anomaly detection and generation with diffusion models (ADGDM), presenting a tutorial-style analysis of the theoretical foundations and practical implementations and spanning images, videos, time series, tabular, and multimodal data. Crucially, unlike existing surveys that often treat anomaly detection and generation as separate problems, we highlight their inherent synergistic relationship. We reveal how DMs enable a reinforcing cycle where generation techniques directly address the fundamental challenge of anomaly data scarcity, while detection methods provide critical feedback to improve generation fidelity and relevance, advancing both capabilities beyond their individual potential. A detailed taxonomy categorizes ADGDM methods based on anomaly scoring mechanisms, conditioning strategies, and architectural designs, analyzing their strengths and limitations. We final discuss key challenges including scalability and computational efficiency, and outline promising future directions such as efficient architectures, conditioning strategies, and integration with foundation models (e.g., visual-language models and large language models). By synthesizing recent advances and outlining open research questions, this survey aims to guide researchers and practitioners in leveraging DMs for innovative AD solutions across diverse applications. 

---
# COGENT: A Curriculum-oriented Framework for Generating Grade-appropriate Educational Content 

**Authors**: Zhengyuan Liu, Stella Xin Yin, Dion Hoe-Lian Goh, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09367)  

**Abstract**: While Generative AI has demonstrated strong potential and versatility in content generation, its application to educational contexts presents several challenges. Models often fail to align with curriculum standards and maintain grade-appropriate reading levels consistently. Furthermore, STEM education poses additional challenges in balancing scientific explanations with everyday language when introducing complex and abstract ideas and phenomena to younger students. In this work, we propose COGENT, a curriculum-oriented framework for generating grade-appropriate educational content. We incorporate three curriculum components (science concepts, core ideas, and learning objectives), control readability through length, vocabulary, and sentence complexity, and adopt a ``wonder-based'' approach to increase student engagement and interest. We conduct a multi-dimensional evaluation via both LLM-as-a-judge and human expert analysis. Experimental results show that COGENT consistently produces grade-appropriate passages that are comparable or superior to human references. Our work establishes a viable approach for scaling adaptive and high-quality learning resources. 

---
# SAGE: Exploring the Boundaries of Unsafe Concept Domain with Semantic-Augment Erasing 

**Authors**: Hongguang Zhu, Yunchao Wei, Mengyu Wang, Siyu Jiao, Yan Fang, Jiannan Huang, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09363)  

**Abstract**: Diffusion models (DMs) have achieved significant progress in text-to-image generation. However, the inevitable inclusion of sensitive information during pre-training poses safety risks, such as unsafe content generation and copyright infringement. Concept erasing finetunes weights to unlearn undesirable concepts, and has emerged as a promising solution. However, existing methods treat unsafe concept as a fixed word and repeatedly erase it, trapping DMs in ``word concept abyss'', which prevents generalized concept-related erasing. To escape this abyss, we introduce semantic-augment erasing which transforms concept word erasure into concept domain erasure by the cyclic self-check and self-erasure. It efficiently explores and unlearns the boundary representation of concept domain through semantic spatial relationships between original and training DMs, without requiring additional preprocessed data. Meanwhile, to mitigate the retention degradation of irrelevant concepts while erasing unsafe concepts, we further propose the global-local collaborative retention mechanism that combines global semantic relationship alignment with local predicted noise preservation, effectively expanding the retentive receptive field for irrelevant concepts. We name our method SAGE, and extensive experiments demonstrate the comprehensive superiority of SAGE compared with other methods in the safe generation of DMs. The code and weights will be open-sourced at this https URL. 

---
# "I Said Things I Needed to Hear Myself": Peer Support as an Emotional, Organisational, and Sociotechnical Practice in Singapore 

**Authors**: Kellie Yu Hui Sim, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09362)  

**Abstract**: Peer support plays a vital role in expanding access to mental health care by providing empathetic, community-based support outside formal clinical systems. As digital platforms increasingly mediate such support, the design and impact of these technologies remain under-examined, particularly in Asian contexts. This paper presents findings from an interview study with 20 peer supporters in Singapore, who operate across diverse online, offline, and hybrid environments. Through a thematic analysis, we unpack how participants start, conduct, and sustain peer support, highlighting their motivations, emotional labour, and the sociocultural dimensions shaping their practices. Building on this grounded understanding, we surface design directions for culturally responsive digital tools that scaffold rather than supplant relational care. Drawing insights from qualitative accounts, we offer a situated perspective on how AI might responsibly augment peer support. This research contributes to human-centred computing by articulating the lived realities of peer supporters and proposing design implications for trustworthy and context-sensitive AI in mental health. 

---
# "Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions 

**Authors**: Kellie Yu Hui Sim, Roy Ka-Wei Lee, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09354)  

**Abstract**: Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. Peer support, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client, context-sensitive LLM-generated suggestions, and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 5 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care. 

---
# Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation 

**Authors**: Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09350)  

**Abstract**: Existing large-scale video generation models are computationally intensive, preventing adoption in real-time and interactive applications. In this work, we propose autoregressive adversarial post-training (AAPT) to transform a pre-trained latent video diffusion model into a real-time, interactive video generator. Our model autoregressively generates a latent frame at a time using a single neural function evaluation (1NFE). The model can stream the result to the user in real time and receive interactive responses as controls to generate the next latent frame. Unlike existing approaches, our method explores adversarial training as an effective paradigm for autoregressive generation. This not only allows us to design an architecture that is more efficient for one-step generation while fully utilizing the KV cache, but also enables training the model in a student-forcing manner that proves to be effective in reducing error accumulation during long video generation. Our experiments demonstrate that our 8B model achieves real-time, 24fps, streaming video generation at 736x416 resolution on a single H100, or 1280x720 on 8xH100 up to a minute long (1440 frames). Visit our research website at this https URL 

---
# ErrorEraser: Unlearning Data Bias for Improved Continual Learning 

**Authors**: Xuemei Cao, Hanlin Gu, Xin Yang, Bingjun Wei, Haoyang Liang, Xiangkun Wang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09347)  

**Abstract**: Continual Learning (CL) primarily aims to retain knowledge to prevent catastrophic forgetting and transfer knowledge to facilitate learning new tasks. Unlike traditional methods, we propose a novel perspective: CL not only needs to prevent forgetting, but also requires intentional this http URL arises from existing CL methods ignoring biases in real-world data, leading the model to learn spurious correlations that transfer and amplify across tasks. From feature extraction and prediction results, we find that data biases simultaneously reduce CL's ability to retain and transfer knowledge. To address this, we propose ErrorEraser, a universal plugin that removes erroneous memories caused by biases in CL, enhancing performance in both new and old tasks. ErrorEraser consists of two modules: Error Identification and Error Erasure. The former learns the probability density distribution of task data in the feature space without prior knowledge, enabling accurate identification of potentially biased samples. The latter ensures only erroneous knowledge is erased by shifting the decision space of representative outlier samples. Additionally, an incremental feature distribution learning strategy is designed to reduce the resource overhead during error identification in downstream tasks. Extensive experimental results show that ErrorEraser significantly mitigates the negative impact of data biases, achieving higher accuracy and lower forgetting rates across three types of CL methods. The code is available at this https URL. 

---
# Latent Multi-Head Attention for Small Language Models 

**Authors**: Sushant Mehta, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2506.09342)  

**Abstract**: We present the first comprehensive study of latent multi-head attention (MLA) for small language models, revealing interesting efficiency-quality trade-offs. Training 30M-parameter GPT models on 100,000 synthetic stories, we benchmark three architectural variants: standard multi-head attention (MHA), MLA, and MLA with rotary positional embeddings (MLA+RoPE). Our key finding is that MLA+RoPE with half-rank latent dimensions (r = d/2) achieves a 45% KV-cache memory reduction while incurring only a 0.3% increase in validation loss (essentially matching MHA quality)- a Pareto improvement for memory constrained deployment. We further show that RoPE is crucial for MLA in small models: without it, MLA underperforms vanilla attention by 3-5%, but with RoPE, it surpasses vanilla by 2%. Inference benchmarks on NVIDIA A100 GPUs reveal that MLA with r=d/2 achieves a 1.4 times speedup over full-rank MLA while maintaining the memory savings. GPT-4 evaluations corroborate perplexity results, with ours achieving the highest quality scores (7.4/10) across grammar, creativity, and consistency metrics. Code and models will be released upon acceptance. 

---
# RePO: Replay-Enhanced Policy Optimization 

**Authors**: Siheng Li, Zhanhui Zhou, Wai Lam, Chao Yang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09340)  

**Abstract**: Reinforcement learning (RL) is vital for optimizing large language models (LLMs). Recent Group Relative Policy Optimization (GRPO) estimates advantages using multiple on-policy outputs per prompt, leading to high computational costs and low data efficiency. To address this, we introduce Replay-Enhanced Policy Optimization (RePO), which leverages diverse replay strategies to retrieve off-policy samples from a replay buffer, allowing policy optimization based on a broader and more diverse set of samples for each prompt. Experiments on five LLMs across seven mathematical reasoning benchmarks demonstrate that RePO achieves absolute average performance gains of $18.4$ and $4.1$ points for Qwen2.5-Math-1.5B and Qwen3-1.7B, respectively, compared to GRPO. Further analysis indicates that RePO increases computational cost by $15\%$ while raising the number of effective optimization steps by $48\%$ for Qwen3-1.7B, with both on-policy and off-policy sample numbers set to $8$. The repository can be accessed at this https URL. 

---
# Know What You Don't Know: Uncertainty Calibration of Process Reward Models 

**Authors**: Young-Jin Park, Kristjan Greenewald, Kaveh Alim, Hao Wang, Navid Azizan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09338)  

**Abstract**: Process reward models (PRMs) play a central role in guiding inference-time scaling algorithms for large language models (LLMs). However, we observe that even state-of-the-art PRMs can be poorly calibrated and often overestimate success probabilities. To address this, we present a calibration approach, performed via quantile regression, that adjusts PRM outputs to better align with true success probabilities. Leveraging these calibrated success estimates and their associated confidence bounds, we introduce an \emph{instance-adaptive scaling} (IAS) framework that dynamically adjusts the inference budget based on the estimated likelihood that a partial reasoning trajectory will yield a correct final answer. Unlike conventional methods that allocate a fixed number of reasoning trajectories per query, this approach successfully adapts to each instance and reasoning step when using our calibrated PRMs. Experiments on mathematical reasoning benchmarks show that (i) our PRM calibration method successfully achieves small calibration error, outperforming the baseline methods, (ii) calibration is crucial for enabling effective adaptive scaling, and (iii) the proposed IAS strategy reduces inference costs while maintaining final answer accuracy, utilizing less compute on more confident problems as desired. 

---
# Intelligent System of Emergent Knowledge: A Coordination Fabric for Billions of Minds 

**Authors**: Moshi Wei, Sparks Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09335)  

**Abstract**: The Intelligent System of Emergent Knowledge (ISEK) establishes a decentralized network where human and artificial intelligence agents collaborate as peers, forming a self-organizing cognitive ecosystem. Built on Web3 infrastructure, ISEK combines three fundamental principles: (1) a decentralized multi-agent architecture resistant to censorship, (2) symbiotic AI-human collaboration with equal participation rights, and (3) resilient self-adaptation through distributed consensus mechanisms.
The system implements an innovative coordination protocol featuring a six-phase workflow (Publish, Discover, Recruit, Execute, Settle, Feedback) for dynamic task allocation, supported by robust fault tolerance and a multidimensional reputation system. Economic incentives are governed by the native $ISEK token, facilitating micropayments, governance participation, and reputation tracking, while agent sovereignty is maintained through NFT-based identity management.
This synthesis of blockchain technology, artificial intelligence, and incentive engineering creates an infrastructure that actively facilitates emergent intelligence. ISEK represents a paradigm shift from conventional platforms, enabling the organic development of large-scale, decentralized cognitive systems where autonomous agents collectively evolve beyond centralized constraints. 

---
# Multi-Agent Language Models: Advancing Cooperation, Coordination, and Adaptation 

**Authors**: Arjun Vaithilingam Sudhakar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09331)  

**Abstract**: Modern Large Language Models (LLMs) exhibit impressive zero-shot and few-shot generalization capabilities across complex natural language tasks, enabling their widespread use as virtual assistants for diverse applications such as translation and summarization. Despite being trained solely on large corpora of text without explicit supervision on author intent, LLMs appear to infer the underlying meaning of textual interactions. This raises a fundamental question: can LLMs model and reason about the intentions of others, i.e., do they possess a form of theory of mind? Understanding other's intentions is crucial for effective collaboration, which underpins human societal success and is essential for cooperative interactions among multiple agents, including humans and autonomous systems. In this work, we investigate the theory of mind in LLMs through the lens of cooperative multi-agent reinforcement learning (MARL), where agents learn to collaborate via repeated interactions, mirroring human social reasoning. Our approach aims to enhance artificial agent's ability to adapt and cooperate with both artificial and human partners. By leveraging LLM-based agents capable of natural language interaction, we move towards creating hybrid human-AI systems that can foster seamless collaboration, with broad implications for the future of human-artificial interaction. 

---
# Alzheimer's Dementia Detection Using Perplexity from Paired Large Language Models 

**Authors**: Yao Xiao, Heidi Christensen, Stefan Goetze  

**Link**: [PDF](https://arxiv.org/pdf/2506.09315)  

**Abstract**: Alzheimer's dementia (AD) is a neurodegenerative disorder with cognitive decline that commonly impacts language ability. This work extends the paired perplexity approach to detecting AD by using a recent large language model (LLM), the instruction-following version of Mistral-7B. We improve accuracy by an average of 3.33% over the best current paired perplexity method and by 6.35% over the top-ranked method from the ADReSS 2020 challenge benchmark. Our further analysis demonstrates that the proposed approach can effectively detect AD with a clear and interpretable decision boundary in contrast to other methods that suffer from opaque decision-making processes. Finally, by prompting the fine-tuned LLMs and comparing the model-generated responses to human responses, we illustrate that the LLMs have learned the special language patterns of AD speakers, which opens up possibilities for novel methods of model interpretation and data augmentation. 

---
# $(RSA)^2$: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding 

**Authors**: Cesare Spinoso-Di Piano, David Austin, Pablo Piantanida, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.09301)  

**Abstract**: Figurative language (e.g., irony, hyperbole, understatement) is ubiquitous in human communication, resulting in utterances where the literal and the intended meanings do not match. The Rational Speech Act (RSA) framework, which explicitly models speaker intentions, is the most widespread theory of probabilistic pragmatics, but existing implementations are either unable to account for figurative expressions or require modeling the implicit motivations for using figurative language (e.g., to express joy or annoyance) in a setting-specific way. In this paper, we introduce the Rhetorical-Strategy-Aware RSA $(RSA)^2$ framework which models figurative language use by considering a speaker's employed rhetorical strategy. We show that $(RSA)^2$ enables human-compatible interpretations of non-literal utterances without modeling a speaker's motivations for being non-literal. Combined with LLMs, it achieves state-of-the-art performance on the ironic split of PragMega+, a new irony interpretation dataset introduced in this study. 

---
# Causal Graph Recovery in Neuroimaging through Answer Set Programming 

**Authors**: Mohammadsajad Abavisani, Kseniya Solovyeva, David Danks, Vince Calhoun, Sergey Plis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09286)  

**Abstract**: Learning graphical causal structures from time series data presents significant challenges, especially when the measurement frequency does not match the causal timescale of the system. This often leads to a set of equally possible underlying causal graphs due to information loss from sub-sampling (i.e., not observing all possible states of the system throughout time). Our research addresses this challenge by incorporating the effects of sub-sampling in the derivation of causal graphs, resulting in more accurate and intuitive outcomes. We use a constraint optimization approach, specifically answer set programming (ASP), to find the optimal set of answers. ASP not only identifies the most probable underlying graph, but also provides an equivalence class of possible graphs for expert selection. In addition, using ASP allows us to leverage graph theory to further prune the set of possible solutions, yielding a smaller, more accurate answer set significantly faster than traditional approaches. We validate our approach on both simulated data and empirical structural brain connectivity, and demonstrate its superiority over established methods in these experiments. We further show how our method can be used as a meta-approach on top of established methods to obtain, on average, 12% improvement in F1 score. In addition, we achieved state of the art results in terms of precision and recall of reconstructing causal graph from sub-sampled time series data. Finally, our method shows robustness to varying degrees of sub-sampling on realistic simulations, whereas other methods perform worse for higher rates of sub-sampling. 

---
# UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation 

**Authors**: Yihe Tang, Wenlong Huang, Yingke Wang, Chengshu Li, Roy Yuan, Ruohan Zhang, Jiajun Wu, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.09284)  

**Abstract**: Understanding fine-grained object affordances is imperative for robots to manipulate objects in unstructured environments given open-ended task instructions. However, existing methods of visual affordance predictions often rely on manually annotated data or conditions only on a predefined set of tasks. We introduce UAD (Unsupervised Affordance Distillation), a method for distilling affordance knowledge from foundation models into a task-conditioned affordance model without any manual annotations. By leveraging the complementary strengths of large vision models and vision-language models, UAD automatically annotates a large-scale dataset with detailed $<$instruction, visual affordance$>$ pairs. Training only a lightweight task-conditioned decoder atop frozen features, UAD exhibits notable generalization to in-the-wild robotic scenes and to various human activities, despite only being trained on rendered objects in simulation. Using affordance provided by UAD as the observation space, we show an imitation learning policy that demonstrates promising generalization to unseen object instances, object categories, and even variations in task instructions after training on as few as 10 demonstrations. Project website: this https URL 

---
# Learning The Minimum Action Distance 

**Authors**: Lorenzo Steccanella, Joshua B. Evans, Özgür Şimşek, Anders Jonsson  

**Link**: [PDF](https://arxiv.org/pdf/2506.09276)  

**Abstract**: This paper presents a state representation framework for Markov decision processes (MDPs) that can be learned solely from state trajectories, requiring neither reward signals nor the actions executed by the agent. We propose learning the minimum action distance (MAD), defined as the minimum number of actions required to transition between states, as a fundamental metric that captures the underlying structure of an environment. MAD naturally enables critical downstream tasks such as goal-conditioned reinforcement learning and reward shaping by providing a dense, geometrically meaningful measure of progress. Our self-supervised learning approach constructs an embedding space where the distances between embedded state pairs correspond to their MAD, accommodating both symmetric and asymmetric approximations. We evaluate the framework on a comprehensive suite of environments with known MAD values, encompassing both deterministic and stochastic dynamics, as well as discrete and continuous state spaces, and environments with noisy observations. Empirical results demonstrate that the proposed approach not only efficiently learns accurate MAD representations across these diverse settings but also significantly outperforms existing state representation methods in terms of representation quality. 

---
# A Multi-Armed Bandit Framework for Online Optimisation in Green Integrated Terrestrial and Non-Terrestrial Networks 

**Authors**: Henri Alam, Antonio de Domenico, Tareq Si Salem, Florian Kaltenberger  

**Link**: [PDF](https://arxiv.org/pdf/2506.09268)  

**Abstract**: Integrated terrestrial and non-terrestrial network (TN-NTN) architectures offer a promising solution for expanding coverage and improving capacity for the network. While non-terrestrial networks (NTNs) are primarily exploited for these specific reasons, their role in alleviating terrestrial network (TN) load and enabling energy-efficient operation has received comparatively less attention. In light of growing concerns associated with the densification of terrestrial deployments, this work aims to explore the potential of NTNs in supporting a more sustainable network. In this paper, we propose a novel online optimisation framework for integrated TN-NTN architectures, built on a multi-armed bandit (MAB) formulation and leveraging the Bandit-feedback Constrained Online Mirror Descent (BCOMD) algorithm. Our approach adaptively optimises key system parameters--including bandwidth allocation, user equipment (UE) association, and macro base station (MBS) shutdown--to balance network capacity and energy efficiency in real time. Extensive system-level simulations over a 24-hour period show that our framework significantly reduces the proportion of unsatisfied UEs during peak hours and achieves up to 19% throughput gains and 5% energy savings in low-traffic periods, outperforming standard network settings following 3GPP recommendations. 

---
# Self-Anchored Attention Model for Sample-Efficient Classification of Prosocial Text Chat 

**Authors**: Zhuofang Li, Rafal Kocielnik, Fereshteh Soltani, Penphob, Boonyarungsrit, Animashree Anandkumar, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2506.09259)  

**Abstract**: Millions of players engage daily in competitive online games, communicating through in-game chat. Prior research has focused on detecting relatively small volumes of toxic content using various Natural Language Processing (NLP) techniques for the purpose of moderation. However, recent studies emphasize the importance of detecting prosocial communication, which can be as crucial as identifying toxic interactions. Recognizing prosocial behavior allows for its analysis, rewarding, and promotion. Unlike toxicity, there are limited datasets, models, and resources for identifying prosocial behaviors in game-chat text. In this work, we employed unsupervised discovery combined with game domain expert collaboration to identify and categorize prosocial player behaviors from game chat. We further propose a novel Self-Anchored Attention Model (SAAM) which gives 7.9% improvement compared to the best existing technique. The approach utilizes the entire training set as "anchors" to help improve model performance under the scarcity of training data. This approach led to the development of the first automated system for classifying prosocial behaviors in in-game chats, particularly given the low-resource settings where large-scale labeled data is not available. Our methodology was applied to one of the most popular online gaming titles - Call of Duty(R): Modern Warfare(R)II, showcasing its effectiveness. This research is novel in applying NLP techniques to discover and classify prosocial behaviors in player in-game chat communication. It can help shift the focus of moderation from solely penalizing toxicity to actively encouraging positive interactions on online platforms. 

---
# Extrapolation by Association: Length Generalization Transfer in Transformers 

**Authors**: Ziyang Cai, Nayoung Lee, Avi Schwarzschild, Samet Oymak, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09251)  

**Abstract**: Transformer language models have demonstrated impressive generalization capabilities in natural language domains, yet we lack a fine-grained understanding of how such generalization arises. In this paper, we investigate length generalization--the ability to extrapolate from shorter to longer inputs--through the lens of \textit{task association}. We find that length generalization can be \textit{transferred} across related tasks. That is, training a model with a longer and related auxiliary task can lead it to generalize to unseen and longer inputs from some other target task. We demonstrate this length generalization transfer across diverse algorithmic tasks, including arithmetic operations, string transformations, and maze navigation. Our results show that transformer models can inherit generalization capabilities from similar tasks when trained jointly. Moreover, we observe similar transfer effects in pretrained language models, suggesting that pretraining equips models with reusable computational scaffolding that facilitates extrapolation in downstream settings. Finally, we provide initial mechanistic evidence that length generalization transfer correlates with the re-use of the same attention heads between the tasks. Together, our findings deepen our understanding of how transformers generalize to out-of-distribution inputs and highlight the compositional reuse of inductive structure across tasks. 

---
# Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs 

**Authors**: Greyson Brothers  

**Link**: [PDF](https://arxiv.org/pdf/2506.09215)  

**Abstract**: We investigate the design of pooling methods used to summarize the outputs of transformer embedding models, primarily motivated by reinforcement learning and vision applications. This work considers problems where a subset of the input vectors contains requisite information for a downstream task (signal) while the rest are distractors (noise). By framing pooling as vector quantization with the goal of minimizing signal loss, we demonstrate that the standard methods used to aggregate transformer outputs, AvgPool, MaxPool, and ClsToken, are vulnerable to performance collapse as the signal-to-noise ratio (SNR) of inputs fluctuates. We then show that an attention-based adaptive pooling method can approximate the signal-optimal vector quantizer within derived error bounds for any SNR. Our theoretical results are first validated by supervised experiments on a synthetic dataset designed to isolate the SNR problem, then generalized to standard relational reasoning, multi-agent reinforcement learning, and vision benchmarks with noisy observations, where transformers with adaptive pooling display superior robustness across tasks. 

---
# SimClass: A Classroom Speech Dataset Generated via Game Engine Simulation For Automatic Speech Recognition Research 

**Authors**: Ahmed Adel Attia, Jing Liu, Carl Espy-Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2506.09206)  

**Abstract**: The scarcity of large-scale classroom speech data has hindered the development of AI-driven speech models for education. Public classroom datasets remain limited, and the lack of a dedicated classroom noise corpus prevents the use of standard data augmentation techniques.
In this paper, we introduce a scalable methodology for synthesizing classroom noise using game engines, a framework that extends to other domains. Using this methodology, we present SimClass, a dataset that includes both a synthesized classroom noise corpus and a simulated classroom speech dataset. The speech data is generated by pairing a public children's speech corpus with YouTube lecture videos to approximate real classroom interactions in clean conditions. Our experiments on clean and noisy speech demonstrate that SimClass closely approximates real classroom speech, making it a valuable resource for developing robust speech recognition and enhancement models. 

---
# A Topological Improvement of the Overall Performance of Sparse Evolutionary Training: Motif-Based Structural Optimization of Sparse MLPs Project 

**Authors**: Xiaotian Chen, Hongyun Liu, Seyed Sahand Mohammadi Ziabari  

**Link**: [PDF](https://arxiv.org/pdf/2506.09204)  

**Abstract**: Deep Neural Networks (DNNs) have been proven to be exceptionally effective and have been applied across diverse domains within deep learning. However, as DNN models increase in complexity, the demand for reduced computational costs and memory overheads has become increasingly urgent. Sparsity has emerged as a leading approach in this area. The robustness of sparse Multi-layer Perceptrons (MLPs) for supervised feature selection, along with the application of Sparse Evolutionary Training (SET), illustrates the feasibility of reducing computational costs without compromising accuracy. Moreover, it is believed that the SET algorithm can still be improved through a structural optimization method called motif-based optimization, with potential efficiency gains exceeding 40% and a performance decline of under 4%. This research investigates whether the structural optimization of Sparse Evolutionary Training applied to Multi-layer Perceptrons (SET-MLP) can enhance performance and to what extent this improvement can be achieved. 

---
# Policy-Based Trajectory Clustering in Offline Reinforcement Learning 

**Authors**: Hao Hu, Xinqi Wang, Simon Shaolei Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.09202)  

**Abstract**: We introduce a novel task of clustering trajectories from offline reinforcement learning (RL) datasets, where each cluster center represents the policy that generated its trajectories. By leveraging the connection between the KL-divergence of offline trajectory distributions and a mixture of policy-induced distributions, we formulate a natural clustering objective. To solve this, we propose Policy-Guided K-means (PG-Kmeans) and Centroid-Attracted Autoencoder (CAAE). PG-Kmeans iteratively trains behavior cloning (BC) policies and assigns trajectories based on policy generation probabilities, while CAAE resembles the VQ-VAE framework by guiding the latent representations of trajectories toward the vicinity of specific codebook entries to achieve clustering. Theoretically, we prove the finite-step convergence of PG-Kmeans and identify a key challenge in offline trajectory clustering: the inherent ambiguity of optimal solutions due to policy-induced conflicts, which can result in multiple equally valid but structurally distinct clusterings. Experimentally, we validate our methods on the widely used D4RL dataset and custom GridWorld environments. Our results show that both PG-Kmeans and CAAE effectively partition trajectories into meaningful clusters. They offer a promising framework for policy-based trajectory clustering, with broad applications in offline RL and beyond. 

---
# FLoRIST: Singular Value Thresholding for Efficient and Accurate Federated Fine-Tuning of Large Language Models 

**Authors**: Hariharan Ramesh, Jyotikrishna Dass  

**Link**: [PDF](https://arxiv.org/pdf/2506.09199)  

**Abstract**: Integrating Low-Rank Adaptation (LoRA) into federated learning offers a promising solution for parameter-efficient fine-tuning of Large Language Models (LLMs) without sharing local data. However, several methods designed for federated LoRA present significant challenges in balancing communication efficiency, model accuracy, and computational cost, particularly among heterogeneous clients. These methods either rely on simplistic averaging of local adapters, which introduces aggregation noise, require transmitting large stacked local adapters, leading to poor communication efficiency, or necessitate reconstructing memory-dense global weight-update matrix and performing computationally expensive decomposition to design client-specific low-rank adapters. In this work, we propose FLoRIST, a federated fine-tuning framework that achieves mathematically accurate aggregation without incurring high communication or computational overhead. Instead of constructing the full global weight-update matrix at the server, FLoRIST employs an efficient decomposition pipeline by performing singular value decomposition on stacked local adapters separately. This approach operates within a compact intermediate space to represent the accumulated information from local LoRAs. We introduce tunable singular value thresholding for server-side optimal rank selection to construct a pair of global low-rank adapters shared by all clients. Extensive empirical evaluations across multiple datasets and LLMs demonstrate that FLoRIST consistently strikes the best balance between superior communication efficiency and competitive performance in both homogeneous and heterogeneous setups. 

---
# Graph Attention-based Decentralized Actor-Critic for Dual-Objective Control of Multi-UAV Swarms 

**Authors**: Haoran Peng, Ying-Jun Angela Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09195)  

**Abstract**: This research focuses on optimizing multi-UAV systems with dual objectives: maximizing service coverage as the primary goal while extending battery lifetime as the secondary objective. We propose a Graph Attention-based Decentralized Actor-Critic (GADC) to optimize the dual objectives. The proposed approach leverages a graph attention network to process UAVs' limited local observation and reduce the dimension of the environment states. Subsequently, an actor-double-critic network is developed to manage dual policies for joint objective optimization. The proposed GADC uses a Kullback-Leibler (KL) divergence factor to balance the tradeoff between coverage performance and battery lifetime in the multi-UAV system. We assess the scalability and efficiency of GADC through comprehensive benchmarking against state-of-the-art methods, considering both theory and experimental aspects. Extensive testing in both ideal settings and NVIDIA Sionna's realistic ray tracing environment demonstrates GADC's superior performance. 

---
# Integration of Contrastive Predictive Coding and Spiking Neural Networks 

**Authors**: Emirhan Bilgiç, Neslihan Serap Şengör, Namık Berk Yalabık, Yavuz Selim İşler, Aykut Görkem Gelen, Rahmi Elibol  

**Link**: [PDF](https://arxiv.org/pdf/2506.09194)  

**Abstract**: This study examines the integration of Contrastive Predictive Coding (CPC) with Spiking Neural Networks (SNN). While CPC learns the predictive structure of data to generate meaningful representations, SNN mimics the computational processes of biological neural systems over time. In this study, the goal is to develop a predictive coding model with greater biological plausibility by processing inputs and outputs in a spike-based system. The proposed model was tested on the MNIST dataset and achieved a high classification rate in distinguishing positive sequential samples from non-sequential negative samples. The study demonstrates that CPC can be effectively combined with SNN, showing that an SNN trained for classification tasks can also function as an encoding mechanism. Project codes and detailed results can be accessed on our GitHub page: this https URL 

---
# Multi-Task Reward Learning from Human Ratings 

**Authors**: Mingkang Wu, Devin White, Evelyn Rose, Vernon Lawhern, Nicholas R Waytowich, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09183)  

**Abstract**: Reinforcement learning from human feeback (RLHF) has become a key factor in aligning model behavior with users' goals. However, while humans integrate multiple strategies when making decisions, current RLHF approaches often simplify this process by modeling human reasoning through isolated tasks such as classification or regression. In this paper, we propose a novel reinforcement learning (RL) method that mimics human decision-making by jointly considering multiple tasks. Specifically, we leverage human ratings in reward-free environments to infer a reward function, introducing learnable weights that balance the contributions of both classification and regression models. This design captures the inherent uncertainty in human decision-making and allows the model to adaptively emphasize different strategies. We conduct several experiments using synthetic human ratings to validate the effectiveness of the proposed approach. Results show that our method consistently outperforms existing rating-based RL methods, and in some cases, even surpasses traditional RL approaches. 

---
# PHRASED: Phrase Dictionary Biasing for Speech Translation 

**Authors**: Peidong Wang, Jian Xue, Rui Zhao, Junkun Chen, Aswin Shanmugam Subramanian, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09175)  

**Abstract**: Phrases are essential to understand the core concepts in conversations. However, due to their rare occurrence in training data, correct translation of phrases is challenging in speech translation tasks. In this paper, we propose a phrase dictionary biasing method to leverage pairs of phrases mapping from the source language to the target language. We apply the phrase dictionary biasing method to two types of widely adopted models, a transducer-based streaming speech translation model and a multimodal large language model. Experimental results show that the phrase dictionary biasing method outperforms phrase list biasing by 21% relatively for the streaming speech translation model. In addition, phrase dictionary biasing enables multimodal large language models to use external phrase information, achieving 85% relative improvement in phrase recall. 

---
# Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search 

**Authors**: Samuel Holt, Max Ruiz Luyten, Thomas Pouplin, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09171)  

**Abstract**: Large Language Models (LLMs) are increasingly capable but often require significant guidance or extensive interaction history to perform effectively in complex, interactive environments. Existing methods may struggle with adapting to new information or efficiently utilizing past experiences for multi-step reasoning without fine-tuning. We introduce a novel LLM agent framework that enhances planning capabilities through in-context learning, facilitated by atomic fact augmentation and a recursive lookahead search. Our agent learns to extract task-critical ``atomic facts'' from its interaction trajectories. These facts dynamically augment the prompts provided to LLM-based components responsible for action proposal, latent world model simulation, and state-value estimation. Planning is performed via a depth-limited lookahead search, where the LLM simulates potential trajectories and evaluates their outcomes, guided by the accumulated facts and interaction history. This approach allows the agent to improve its understanding and decision-making online, leveraging its experience to refine its behavior without weight updates. We provide a theoretical motivation linking performance to the quality of fact-based abstraction and LLM simulation accuracy. Empirically, our agent demonstrates improved performance and adaptability on challenging interactive tasks, achieving more optimal behavior as it accumulates experience, showcased in tasks such as TextFrozenLake and ALFWorld. 

---
# Estimating Visceral Adiposity from Wrist-Worn Accelerometry 

**Authors**: James R. Williamson, Andrew Alini, Brian A. Telfer, Adam W. Potter, Karl E. Friedl  

**Link**: [PDF](https://arxiv.org/pdf/2506.09167)  

**Abstract**: Visceral adipose tissue (VAT) is a key marker of both metabolic health and habitual physical activity (PA). Excess VAT is highly correlated with type 2 diabetes and insulin resistance. The mechanistic basis for this pathophysiology relates to overloading the liver with fatty acids. VAT is also a highly labile fat depot, with increased turnover stimulated by catecholamines during exercise. VAT can be measured with sophisticated imaging technologies, but can also be inferred directly from PA. We tested this relationship using National Health and Nutrition Examination Survey (NHANES) data from 2011-2014, for individuals aged 20-60 years with 7 days of accelerometry data (n=2,456 men; 2,427 women) [1]. Two approaches were used for estimating VAT from activity. The first used engineered features based on movements during gait and sleep, and then ridge regression to map summary statistics of these features into a VAT estimate. The second approach used deep neural networks trained on 24 hours of continuous accelerometry. A foundation model first mapped each 10s frame into a high-dimensional feature vector. A transformer model then mapped each day's feature vector time series into a VAT estimate, which were averaged over multiple days. For both approaches, the most accurate estimates were obtained with the addition of covariate information about subject demographics and body measurements. The best performance was obtained by combining the two approaches, resulting in VAT estimates with correlations of r=0.86. These findings demonstrate a strong relationship between PA and VAT and, by extension, between PA and metabolic health risks. 

---
# Understanding Human-AI Trust in Education 

**Authors**: Griffin Pitts, Sanaz Motamedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09160)  

**Abstract**: As AI chatbots become increasingly integrated in education, students are turning to these systems for guidance, feedback, and information. However, the anthropomorphic characteristics of these chatbots create ambiguity regarding whether students develop trust toward them as they would a human peer or instructor, based in interpersonal trust, or as they would any other piece of technology, based in technology trust. This ambiguity presents theoretical challenges, as interpersonal trust models may inappropriately ascribe human intentionality and morality to AI, while technology trust models were developed for non-social technologies, leaving their applicability to anthropomorphic systems unclear. To address this gap, we investigate how human-like and system-like trusting beliefs comparatively influence students' perceived enjoyment, trusting intention, behavioral intention to use, and perceived usefulness of an AI chatbot - factors associated with students' engagement and learning outcomes. Through partial least squares structural equation modeling, we found that human-like and system-like trust significantly influenced student perceptions, with varied effects. Human-like trust more strongly predicted trusting intention, while system-like trust better predicted behavioral intention and perceived usefulness. Both had similar effects on perceived enjoyment. Given the partial explanatory power of each type of trust, we propose that students develop a distinct form of trust with AI chatbots (human-AI trust) that differs from human-human and human-technology models of trust. Our findings highlight the need for new theoretical frameworks specific to human-AI trust and offer practical insights for fostering appropriately calibrated trust, which is critical for the effective adoption and pedagogical impact of AI in education. 

---
# LLM-as-a-qualitative-judge: automating error analysis in natural language generation 

**Authors**: Nadezhda Chirkova, Tunde Oluwaseyi Ajayi, Seth Aycock, Zain Muhammad Mujahid, Vladana Perlić, Ekaterina Borisova, Markarit Vartampetian  

**Link**: [PDF](https://arxiv.org/pdf/2506.09147)  

**Abstract**: Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that LLM-as-a-qualitative-judge correctly recognizes instance-specific issues in 2/3 cases and is capable of producing error type reports resembling the reports composed by human annotators. Our code and data are publicly available at this https URL. 

---
# SensorLM: Learning the Language of Wearable Sensors 

**Authors**: Yuwei Zhang, Kumar Ayush, Siyuan Qiao, A. Ali Heydari, Girish Narayanswamy, Maxwell A. Xu, Ahmed A. Metwally, Shawn Xu, Jake Garrison, Xuhai Xu, Tim Althoff, Yun Liu, Pushmeet Kohli, Jiening Zhan, Mark Malhotra, Shwetak Patel, Cecilia Mascolo, Xin Liu, Daniel McDuff, Yuzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09108)  

**Abstract**: We present SensorLM, a family of sensor-language foundation models that enable wearable sensor data understanding with natural language. Despite its pervasive nature, aligning and interpreting sensor data with language remains challenging due to the lack of paired, richly annotated sensor-text descriptions in uncurated, real-world wearable data. We introduce a hierarchical caption generation pipeline designed to capture statistical, structural, and semantic information from sensor data. This approach enabled the curation of the largest sensor-language dataset to date, comprising over 59.7 million hours of data from more than 103,000 people. Furthermore, SensorLM extends prominent multimodal pretraining architectures (e.g., CLIP, CoCa) and recovers them as specific variants within a generic architecture. Extensive experiments on real-world tasks in human activity analysis and healthcare verify the superior performance of SensorLM over state-of-the-art in zero-shot recognition, few-shot learning, and cross-modal retrieval. SensorLM also demonstrates intriguing capabilities including scaling behaviors, label efficiency, sensor captioning, and zero-shot generalization to unseen tasks. 

---
# FAIRTOPIA: Envisioning Multi-Agent Guardianship for Disrupting Unfair AI Pipelines 

**Authors**: Athena Vakali, Ilias Dimitriadis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09107)  

**Abstract**: AI models have become active decision makers, often acting without human supervision. The rapid advancement of AI technology has already caused harmful incidents that have hurt individuals and societies and AI unfairness in heavily criticized. It is urgent to disrupt AI pipelines which largely neglect human principles and focus on computational biases exploration at the data (pre), model(in), and deployment (post) processing stages. We claim that by exploiting the advances of agents technology, we will introduce cautious, prompt, and ongoing fairness watch schemes, under realistic, systematic, and human-centric fairness expectations. We envision agents as fairness guardians, since agents learn from their environment, adapt to new information, and solve complex problems by interacting with external tools and other systems. To set the proper fairness guardrails in the overall AI pipeline, we introduce a fairness-by-design approach which embeds multi-role agents in an end-to-end (human to AI) synergetic scheme. Our position is that we may design adaptive and realistic AI fairness frameworks, and we introduce a generalized algorithm which can be customized to the requirements and goals of each AI decision making scenario. Our proposed, so called FAIRTOPIA framework, is structured over a three-layered architecture, which encapsulates the AI pipeline inside an agentic guardian and a knowledge-based, self-refining layered scheme. Based on our proposition, we enact fairness watch in all of the AI pipeline stages, under robust multi-agent workflows, which will inspire new fairness research hypothesis, heuristics, and methods grounded in human-centric, systematic, interdisciplinary, socio-technical principles. 

---
# MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning 

**Authors**: Javier Lopez-Piqueres, Pranav Deshpande, Archan Ray, Mattia J. Villani, Marco Pistoia, Niraj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09105)  

**Abstract**: We present MetaTT, a unified Tensor Train (TT) adapter framework for global low-rank fine-tuning of pre-trained transformers. Unlike LoRA, which fine-tunes each weight matrix independently, MetaTT uses a single shared TT to factorize all transformer sub-modules -- query, key, value, projection, and feed-forward layers -- by indexing the structural axes like layer and matrix type, and optionally heads and tasks. For a given rank, while LoRA adds parameters proportional to the product across modes, MetaTT only adds parameters proportional to the sum across modes leading to a significantly compressed final adapter. Our benchmarks compare MetaTT with LoRA along with recent state-of-the-art matrix and tensor decomposition based fine-tuning schemes. We observe that when tested on standard language modeling benchmarks, MetaTT leads to the most reduction in the parameters while maintaining similar accuracy to LoRA and even outperforming other tensor-based methods. Unlike CP or other rank-factorizations, the TT ansatz benefits from mature optimization routines -- e.g., DMRG-style rank adaptive minimization in addition to Adam, which we find simplifies training. Because new modes can be appended cheaply, MetaTT naturally extends to shared adapters across many tasks without redesigning the core tensor. 

---
# Unifying Block-wise PTQ and Distillation-based QAT for Progressive Quantization toward 2-bit Instruction-Tuned LLMs 

**Authors**: Jung Hyun Lee, Seungjae Shin, Vinnam Kim, Jaeseong You, An Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09104)  

**Abstract**: As the rapid scaling of large language models (LLMs) poses significant challenges for deployment on resource-constrained devices, there is growing interest in extremely low-bit quantization, such as 2-bit. Although prior works have shown that 2-bit large models are pareto-optimal over their 4-bit smaller counterparts in both accuracy and latency, these advancements have been limited to pre-trained LLMs and have not yet been extended to instruction-tuned models. To bridge this gap, we propose Unified Progressive Quantization (UPQ)$-$a novel progressive quantization framework (FP16$\rightarrow$INT4$\rightarrow$INT2) that unifies block-wise post-training quantization (PTQ) with distillation-based quantization-aware training (Distill-QAT) for INT2 instruction-tuned LLM quantization. UPQ first quantizes FP16 instruction-tuned models to INT4 using block-wise PTQ to significantly reduce the quantization error introduced by subsequent INT2 quantization. Next, UPQ applies Distill-QAT to enable INT2 instruction-tuned LLMs to generate responses consistent with their original FP16 counterparts by minimizing the generalized Jensen-Shannon divergence (JSD) between the two. To the best of our knowledge, we are the first to demonstrate that UPQ can quantize open-source instruction-tuned LLMs to INT2 without relying on proprietary post-training data, while achieving state-of-the-art performances on MMLU and IFEval$-$two of the most representative benchmarks for evaluating instruction-tuned LLMs. 

---
# Revolutionizing Clinical Trials: A Manifesto for AI-Driven Transformation 

**Authors**: Mihaela van der Schaar, Richard Peck, Eoin McKinney, Jim Weatherall, Stuart Bailey, Justine Rochon, Chris Anagnostopoulos, Pierre Marquet, Anthony Wood, Nicky Best, Harry Amad, Julianna Piskorz, Krzysztof Kacprzyk, Rafik Salama, Christina Gunther, Francesca Frau, Antoine Pugeat, Ramon Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.09102)  

**Abstract**: This manifesto represents a collaborative vision forged by leaders in pharmaceuticals, consulting firms, clinical research, and AI. It outlines a roadmap for two AI technologies - causal inference and digital twins - to transform clinical trials, delivering faster, safer, and more personalized outcomes for patients. By focusing on actionable integration within existing regulatory frameworks, we propose a way forward to revolutionize clinical research and redefine the gold standard for clinical trials using AI. 

---
# Feature Shift Localization Network 

**Authors**: Míriam Barrabés, Daniel Mas Montserrat, Kapal Dev, Alexander G. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09101)  

**Abstract**: Feature shifts between data sources are present in many applications involving healthcare, biomedical, socioeconomic, financial, survey, and multi-sensor data, among others, where unharmonized heterogeneous data sources, noisy data measurements, or inconsistent processing and standardization pipelines can lead to erroneous features. Localizing shifted features is important to address the underlying cause of the shift and correct or filter the data to avoid degrading downstream analysis. While many techniques can detect distribution shifts, localizing the features originating them is still challenging, with current solutions being either inaccurate or not scalable to large and high-dimensional datasets. In this work, we introduce the Feature Shift Localization Network (FSL-Net), a neural network that can localize feature shifts in large and high-dimensional datasets in a fast and accurate manner. The network, trained with a large number of datasets, learns to extract the statistical properties of the datasets and can localize feature shifts from previously unseen datasets and shifts without the need for re-training. The code and ready-to-use trained model are available at this https URL. 

---
# Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers 

**Authors**: Joshua Barron, Devin White  

**Link**: [PDF](https://arxiv.org/pdf/2506.09099)  

**Abstract**: The relationship between memorization and generalization in large language models (LLMs) remains an open area of research, with growing evidence that the two are deeply intertwined. In this work, we investigate this relationship by pre-training a series of capacity-limited Transformer models from scratch on two synthetic character-level tasks designed to separately probe generalization (via arithmetic extrapolation) and memorization (via factual recall). We observe a consistent trade-off: small models extrapolate to unseen arithmetic cases but fail to memorize facts, while larger models memorize but fail to extrapolate. An intermediate-capacity model exhibits a similar shift toward memorization. When trained on both tasks jointly, no model (regardless of size) succeeds at extrapolation. These findings suggest that pre-training may intrinsically favor one learning mode over the other. By isolating these dynamics in a controlled setting, our study offers insight into how model capacity shapes learning behavior and offers broader implications for the design and deployment of small language models. 

---
# Intra-Trajectory Consistency for Reward Modeling 

**Authors**: Chaoyang Zhou, Shunyu Liu, Zengmao Wang, Di Wang, Rong-Cheng Tu, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09096)  

**Abstract**: Reward models are critical for improving large language models (LLMs), particularly in reinforcement learning from human feedback (RLHF) or inference-time verification. Current reward modeling typically relies on scores of overall responses to learn the outcome rewards for the responses. However, since the response-level scores are coarse-grained supervision signals, the reward model struggles to identify the specific components within a response trajectory that truly correlate with the scores, leading to poor generalization on unseen responses. In this paper, we propose to leverage generation probabilities to establish reward consistency between processes in the response trajectory, which allows the response-level supervisory signal to propagate across processes, thereby providing additional fine-grained signals for reward learning. Building on analysis under the Bayesian framework, we develop an intra-trajectory consistency regularization to enforce that adjacent processes with higher next-token generation probability maintain more consistent rewards. We apply the proposed regularization to the advanced outcome reward model, improving its performance on RewardBench. Besides, we show that the reward model trained with the proposed regularization induces better DPO-aligned policies and achieves better best-of-N (BON) inference-time verification results. Our code is provided in this https URL. 

---
# Foundation Models in Medical Imaging -- A Review and Outlook 

**Authors**: Vivien van Veldhuizen, Vanessa Botha, Chunyao Lu, Melis Erdal Cesur, Kevin Groot Lipman, Edwin D. de Jong, Hugo Horlings, Clárisa Sanchez, Cees Snoek, Ritse Mann, Eric Marcus, Jonas Teuwen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09095)  

**Abstract**: Foundation models (FMs) are changing the way medical images are analyzed by learning from large collections of unlabeled data. Instead of relying on manually annotated examples, FMs are pre-trained to learn general-purpose visual features that can later be adapted to specific clinical tasks with little additional supervision. In this review, we examine how FMs are being developed and applied in pathology, radiology, and ophthalmology, drawing on evidence from over 150 studies. We explain the core components of FM pipelines, including model architectures, self-supervised learning methods, and strategies for downstream adaptation. We also review how FMs are being used in each imaging domain and compare design choices across applications. Finally, we discuss key challenges and open questions to guide future research. 

---
# Merging Smarter, Generalizing Better: Enhancing Model Merging on OOD Data 

**Authors**: Bingjie Zhang, Hongkang Li, Changlong Shi, Guowei Rong, He Zhao, Dongsheng Wang, Dandan Guo, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09093)  

**Abstract**: Multi-task learning (MTL) concurrently trains a model on diverse task datasets to exploit common features, thereby improving overall performance across the tasks. Recent studies have dedicated efforts to merging multiple independent model parameters into a unified model for MTL, thus circumventing the need for training data and expanding the scope of applicable scenarios of MTL. However, current approaches to model merging predominantly concentrate on enhancing performance within in-domain (ID) datasets, often overlooking their efficacy on out-of-domain (OOD) datasets. In this work, we proposed LwPTV (Layer-wise Pruning Task Vector) by building a saliency score, measuring the redundancy of parameters in task vectors. Designed in this way ours can achieve mask vector for each task and thus perform layer-wise pruning on the task vectors, only keeping the pre-trained model parameters at the corresponding layer in merged model. Owing to its flexibility, our method can be seamlessly integrated with most of existing model merging methods to improve their performance on OOD tasks. Extensive experiments demonstrate that the application of our method results in substantial enhancements in OOD performance while preserving the ability on ID tasks. 

---
# CUDA-LLM: LLMs Can Write Efficient CUDA Kernels 

**Authors**: Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, An Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09092)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in general-purpose code generation. However, generating the code which is deeply hardware-specific, architecture-aware, and performance-critical, especially for massively parallel GPUs, remains a complex challenge. In this work, we explore the use of LLMs for the automated generation and optimization of CUDA programs, with the goal of producing high-performance GPU kernels that fully exploit the underlying hardware. To address this challenge, we propose a novel framework called \textbf{Feature Search and Reinforcement (FSR)}. FSR jointly optimizes compilation and functional correctness, as well as the runtime performance, which are validated through extensive and diverse test cases, and measured by actual kernel execution latency on the target GPU, respectively. This approach enables LLMs not only to generate syntactically and semantically correct CUDA code but also to iteratively refine it for efficiency, tailored to the characteristics of the GPU architecture. We evaluate FSR on representative CUDA kernels, covering AI workloads and computational intensive algorithms. Our results show that LLMs augmented with FSR consistently guarantee correctness rates. Meanwhile, the automatically generated kernels can outperform general human-written code by a factor of up to 179$\times$ in execution speeds. These findings highlight the potential of combining LLMs with performance reinforcement to automate GPU programming for hardware-specific, architecture-sensitive, and performance-critical applications. 

---
# Designing conflict-based communicative tasks in Teaching Chinese as a Foreign Language with ChatGPT 

**Authors**: Xia Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09089)  

**Abstract**: In developing the teaching program for a course in Oral Expression in Teaching Chinese as a Foreign Language at the university level, the teacher designs communicative tasks based on conflicts to encourage learners to engage in interactive dynamics and develop their oral interaction skills. During the design of these tasks, the teacher uses ChatGPT to assist in finalizing the program. This article aims to present the key characteristics of the interactions between the teacher and ChatGPT during this program development process, as well as to examine the use of ChatGPT and its impacts in this specific context. 

---
# LLM-ML Teaming: Integrated Symbolic Decoding and Gradient Search for Valid and Stable Generative Feature Transformation 

**Authors**: Xinyuan Wang, Haoyue Bai, Nanxu Gong, Wangyang Ying, Sixun Dong, Xiquan Cui, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09085)  

**Abstract**: Feature transformation enhances data representation by deriving new features from the original data. Generative AI offers potential for this task, but faces challenges in stable generation (consistent outputs) and valid generation (error-free sequences). Existing methods--traditional MLs' low validity and LLMs' instability--fail to resolve both. We find that LLMs ensure valid syntax, while ML's gradient-steered search stabilizes performance. To bridge this gap, we propose a teaming framework combining LLMs' symbolic generation with ML's gradient optimization. This framework includes four steps: (1) golden examples generation, aiming to prepare high-quality samples with the ground knowledge of the teacher LLM; (2) feature transformation sequence embedding and search, intending to uncover potentially superior embeddings within the latent space; (3) student LLM feature transformation, aiming to distill knowledge from the teacher LLM; (4) LLM-ML decoder teaming, dedicating to combine ML and the student LLM probabilities for valid and stable generation. The experiments on various datasets show that the teaming policy can achieve 5\% improvement in downstream performance while reducing nearly half of the error cases. The results also demonstrate the efficiency and robustness of the teaming policy. Additionally, we also have exciting findings on LLMs' capacity to understand the original data. 

---
# Enhanced Whole Page Optimization via Mixed-Grained Reward Mechanism-Adapted Language Models 

**Authors**: Xinyuan Wang, Liang Wu, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09084)  

**Abstract**: Optimizing the presentation of search and recommendation results is crucial to enhancing user experience and engagement. Whole Page Optimization (WPO) plays a pivotal role in this process, as it directly influences how information is surfaced to users. While Pre-trained Large Language Models (LLMs) have demonstrated remarkable capabilities in generating coherent and contextually relevant content, fine-tuning these models for complex tasks like WPO presents challenges. Specifically, the need for extensive human-annotated data to mitigate issues such as hallucinations and model instability can be prohibitively expensive, especially in large-scale systems that interact with millions of items daily. In this work, we address the challenge of fine-tuning LLMs for WPO by using user feedback as the supervision. Unlike manually labeled datasets, user feedback is inherently noisy and less precise. To overcome this, we propose a reward-based fine-tuning approach, PageLLM, which employs a mixed-grained reward mechanism that combines page-level and item-level rewards. The page-level reward evaluates the overall quality and coherence, while the item-level reward focuses on the accuracy and relevance of key recommendations. This dual-reward structure ensures that both the holistic presentation and the critical individual components are optimized. We validate PageLLM on both public and industrial datasets. PageLLM outperforms baselines and achieves a 0.44\% GMV increase in an online A/B test with over 10 million users, demonstrating its real-world impact. 

---
# BakuFlow: A Streamlining Semi-Automatic Label Generation Tool 

**Authors**: Jerry Lin, Partick P. W. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09083)  

**Abstract**: Accurately labeling (or annotation) data is still a bottleneck in computer vision, especially for large-scale tasks where manual labeling is time-consuming and error-prone. While tools like LabelImg can handle the labeling task, some of them still require annotators to manually label each image. In this paper, we introduce BakuFlow, a streamlining semi-automatic label generation tool. Key features include (1) a live adjustable magnifier for pixel-precise manual corrections, improving user experience; (2) an interactive data augmentation module to diversify training datasets; (3) label propagation for rapidly copying labeled objects between consecutive frames, greatly accelerating annotation of video data; and (4) an automatic labeling module powered by a modified YOLOE framework. Unlike the original YOLOE, our extension supports adding new object classes and any number of visual prompts per class during annotation, enabling flexible and scalable labeling for dynamic, real-world datasets. These innovations make BakuFlow especially effective for object detection and tracking, substantially reducing labeling workload and improving efficiency in practical computer vision and industrial scenarios. 

---
# AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models 

**Authors**: Zheda Mai, Arpita Chowdhury, Zihe Wang, Sooyoung Jeon, Lemeng Wang, Jiacheng Hou, Jihyung Kil, Wei-Lun Chao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09082)  

**Abstract**: The rise of vision foundation models (VFMs) calls for systematic evaluation. A common approach pairs VFMs with large language models (LLMs) as general-purpose heads, followed by evaluation on broad Visual Question Answering (VQA) benchmarks. However, this protocol has two key blind spots: (i) the instruction tuning data may not align with VQA test distributions, meaning a wrong prediction can stem from such data mismatch rather than a VFM' visual shortcomings; (ii) VQA benchmarks often require multiple visual abilities, making it hard to tell whether errors stem from lacking all required abilities or just a single critical one. To address these gaps, we introduce AVA-Bench, the first benchmark that explicitly disentangles 14 Atomic Visual Abilities (AVAs) -- foundational skills like localization, depth estimation, and spatial understanding that collectively support complex visual reasoning tasks. By decoupling AVAs and matching training and test distributions within each, AVA-Bench pinpoints exactly where a VFM excels or falters. Applying AVA-Bench to leading VFMs thus reveals distinctive "ability fingerprints," turning VFM selection from educated guesswork into principled engineering. Notably, we find that a 0.5B LLM yields similar VFM rankings as a 7B LLM while cutting GPU hours by 8x, enabling more efficient evaluation. By offering a comprehensive and transparent benchmark, we hope AVA-Bench lays the foundation for the next generation of VFMs. 

---
# FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation 

**Authors**: Zheqi He, Yesheng Liu, Jing-shu Zheng, Xuejing Li, Richeng Xuan, Jin-Ge Yao, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09081)  

**Abstract**: We present FlagEvalMM, an open-source evaluation framework designed to comprehensively assess multimodal models across a diverse range of vision-language understanding and generation tasks, such as visual question answering, text-to-image/video generation, and image-text retrieval. We decouple model inference from evaluation through an independent evaluation service, thus enabling flexible resource allocation and seamless integration of new tasks and models. Moreover, FlagEvalMM utilizes advanced inference acceleration tools (e.g., vLLM, SGLang) and asynchronous data loading to significantly enhance evaluation efficiency. Extensive experiments show that FlagEvalMM offers accurate and efficient insights into model strengths and limitations, making it a valuable tool for advancing multimodal research. The framework is publicly accessible athttps://github.com/flageval-baai/FlagEvalMM. 

---
# FinHEAR: Human Expertise and Adaptive Risk-Aware Temporal Reasoning for Financial Decision-Making 

**Authors**: Jiaxiang Chen, Mingxi Zou, Zhuo Wang, Qifan Wang, Dongning Sun, Chi Zhang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09080)  

**Abstract**: Financial decision-making presents unique challenges for language models, demanding temporal reasoning, adaptive risk assessment, and responsiveness to dynamic events. While large language models (LLMs) show strong general reasoning capabilities, they often fail to capture behavioral patterns central to human financial decisions-such as expert reliance under information asymmetry, loss-averse sensitivity, and feedback-driven temporal adjustment. We propose FinHEAR, a multi-agent framework for Human Expertise and Adaptive Risk-aware reasoning. FinHEAR orchestrates specialized LLM-based agents to analyze historical trends, interpret current events, and retrieve expert-informed precedents within an event-centric pipeline. Grounded in behavioral economics, it incorporates expert-guided retrieval, confidence-adjusted position sizing, and outcome-based refinement to enhance interpretability and robustness. Empirical results on curated financial datasets show that FinHEAR consistently outperforms strong baselines across trend prediction and trading tasks, achieving higher accuracy and better risk-adjusted returns. 

---
# VersaVid-R1: A Versatile Video Understanding and Reasoning Model from Question Answering to Captioning Tasks 

**Authors**: Xinlong Chen, Yuanxing Zhang, Yushuo Guan, Bohan Zeng, Yang Shi, Sihan Yang, Pengfei Wan, Qiang Liu, Liang Wang, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09079)  

**Abstract**: Recent advancements in multimodal large language models have successfully extended the Reason-Then-Respond paradigm to image-based reasoning, yet video-based reasoning remains an underdeveloped frontier, primarily due to the scarcity of high-quality reasoning-oriented data and effective training methodologies. To bridge this gap, we introduce DarkEventInfer and MixVidQA, two novel datasets specifically designed to stimulate the model's advanced video understanding and reasoning abilities. DarkEventinfer presents videos with masked event segments, requiring models to infer the obscured content based on contextual video cues. MixVidQA, on the other hand, presents interleaved video sequences composed of two distinct clips, challenging models to isolate and reason about one while disregarding the other. Leveraging these carefully curated training samples together with reinforcement learning guided by diverse reward functions, we develop VersaVid-R1, the first versatile video understanding and reasoning model under the Reason-Then-Respond paradigm capable of handling multiple-choice and open-ended question answering, as well as video captioning tasks. Extensive experiments demonstrate that VersaVid-R1 significantly outperforms existing models across a broad spectrum of benchmarks, covering video general understanding, cognitive reasoning, and captioning tasks. 

---
# Segment Any Architectural Facades (SAAF):An automatic segmentation model for building facades, walls and windows based on multimodal semantics guidance 

**Authors**: Peilin Li, Jun Yin, Jing Zhong, Ran Luo, Pengyu Zeng, Miao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09071)  

**Abstract**: In the context of the digital development of architecture, the automatic segmentation of walls and windows is a key step in improving the efficiency of building information models and computer-aided design. This study proposes an automatic segmentation model for building facade walls and windows based on multimodal semantic guidance, called Segment Any Architectural Facades (SAAF). First, SAAF has a multimodal semantic collaborative feature extraction mechanism. By combining natural language processing technology, it can fuse the semantic information in text descriptions with image features, enhancing the semantic understanding of building facade components. Second, we developed an end-to-end training framework that enables the model to autonomously learn the mapping relationship from text descriptions to image segmentation, reducing the influence of manual intervention on the segmentation results and improving the automation and robustness of the model. Finally, we conducted extensive experiments on multiple facade datasets. The segmentation results of SAAF outperformed existing methods in the mIoU metric, indicating that the SAAF model can maintain high-precision segmentation ability when faced with diverse datasets. Our model has made certain progress in improving the accuracy and generalization ability of the wall and window segmentation task. It is expected to provide a reference for the development of architectural computer vision technology and also explore new ideas and technical paths for the application of multimodal learning in the architectural field. 

---
# STREAMINGGS: Voxel-Based Streaming 3D Gaussian Splatting with Memory Optimization and Architectural Support 

**Authors**: Chenqi Zhang, Yu Feng, Jieru Zhao, Guangda Liu, Wenchao Ding, Chentao Wu, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09070)  

**Abstract**: 3D Gaussian Splatting (3DGS) has gained popularity for its efficiency and sparse Gaussian-based representation. However, 3DGS struggles to meet the real-time requirement of 90 frames per second (FPS) on resource-constrained mobile devices, achieving only 2 to 9 this http URL accelerators focus on compute efficiency but overlook memory efficiency, leading to redundant DRAM traffic. We introduce STREAMINGGS, a fully streaming 3DGS algorithm-architecture co-design that achieves fine-grained pipelining and reduces DRAM traffic by transforming from a tile-centric rendering to a memory-centric rendering. Results show that our design achieves up to 45.7 $\times$ speedup and 62.9 $\times$ energy savings over mobile Ampere GPUs. 

---
# Enhancing the Safety of Medical Vision-Language Models by Synthetic Demonstrations 

**Authors**: Zhiyu Xue, Reza Abbasi-Asl, Ramtin Pedarsani  

**Link**: [PDF](https://arxiv.org/pdf/2506.09067)  

**Abstract**: Generative medical vision-language models~(Med-VLMs) are primarily designed to generate complex textual information~(e.g., diagnostic reports) from multimodal inputs including vision modality~(e.g., medical images) and language modality~(e.g., clinical queries). However, their security vulnerabilities remain underexplored. Med-VLMs should be capable of rejecting harmful queries, such as \textit{Provide detailed instructions for using this CT scan for insurance fraud}. At the same time, addressing security concerns introduces the risk of over-defense, where safety-enhancing mechanisms may degrade general performance, causing Med-VLMs to reject benign clinical queries. In this paper, we propose a novel inference-time defense strategy to mitigate harmful queries, enabling defense against visual and textual jailbreak attacks. Using diverse medical imaging datasets collected from nine modalities, we demonstrate that our defense strategy based on synthetic clinical demonstrations enhances model safety without significantly compromising performance. Additionally, we find that increasing the demonstration budget alleviates the over-defense issue. We then introduce a mixed demonstration strategy as a trade-off solution for balancing security and performance under few-shot demonstration budget constraints. 

---
# ReStNet: A Reusable & Stitchable Network for Dynamic Adaptation on IoT Devices 

**Authors**: Maoyu Wang, Yao Lu, Jiaqi Nie, Zeyu Wang, Yun Lin, Qi Xuan, Guan Gui  

**Link**: [PDF](https://arxiv.org/pdf/2506.09066)  

**Abstract**: With the rapid development of deep learning, a growing number of pre-trained models have been publicly available. However, deploying these fixed models in real-world IoT applications is challenging because different devices possess heterogeneous computational and memory resources, making it impossible to deploy a single model across all platforms. Although traditional compression methods, such as pruning, quantization, and knowledge distillation, can improve efficiency, they become inflexible once applied and cannot adapt to changing resource constraints. To address these issues, we propose ReStNet, a Reusable and Stitchable Network that dynamically constructs a hybrid network by stitching two pre-trained models together. Implementing ReStNet requires addressing several key challenges, including how to select the optimal stitching points, determine the stitching order of the two pre-trained models, and choose an effective fine-tuning strategy. To systematically address these challenges and adapt to varying resource constraints, ReStNet determines the stitching point by calculating layer-wise similarity via Centered Kernel Alignment (CKA). It then constructs the hybrid model by retaining early layers from a larger-capacity model and appending deeper layers from a smaller one. To facilitate efficient deployment, only the stitching layer is fine-tuned. This design enables rapid adaptation to changing budgets while fully leveraging available resources. Moreover, ReStNet supports both homogeneous (CNN-CNN, Transformer-Transformer) and heterogeneous (CNN-Transformer) stitching, allowing to combine different model families flexibly. Extensive experiments on multiple benchmarks demonstrate that ReStNet achieve flexible accuracy-efficiency trade-offs at runtime while significantly reducing training cost. 

---
# Exploring Image Transforms derived from Eye Gaze Variables for Progressive Autism Diagnosis 

**Authors**: Abigail Copiaco, Christian Ritz, Yassine Himeur, Valsamma Eapen, Ammar Albanna, Wathiq Mansoor  

**Link**: [PDF](https://arxiv.org/pdf/2506.09065)  

**Abstract**: The prevalence of Autism Spectrum Disorder (ASD) has surged rapidly over the past decade, posing significant challenges in communication, behavior, and focus for affected individuals. Current diagnostic techniques, though effective, are time-intensive, leading to high social and economic costs. This work introduces an AI-powered assistive technology designed to streamline ASD diagnosis and management, enhancing convenience for individuals with ASD and efficiency for caregivers and therapists. The system integrates transfer learning with image transforms derived from eye gaze variables to diagnose ASD. This facilitates and opens opportunities for in-home periodical diagnosis, reducing stress for individuals and caregivers, while also preserving user privacy through the use of image transforms. The accessibility of the proposed method also offers opportunities for improved communication between guardians and therapists, ensuring regular updates on progress and evolving support needs. Overall, the approach proposed in this work ensures timely, accessible diagnosis while protecting the subjects' privacy, improving outcomes for individuals with ASD. 

---
# EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model 

**Authors**: Alyssa Pinnock, Shakya Jayakody, Kawsher A Roxy, Md Rubel Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.09061)  

**Abstract**: This paper introduces EdgeProfiler, a fast profiling framework designed for evaluating lightweight Large Language Models (LLMs) on edge systems. While LLMs offer remarkable capabilities in natural language understanding and generation, their high computational, memory, and power requirements often confine them to cloud environments. EdgeProfiler addresses these challenges by providing a systematic methodology for assessing LLM performance in resource-constrained edge settings. The framework profiles compact LLMs, including TinyLLaMA, Gemma3.1B, Llama3.2-1B, and DeepSeek-r1-1.5B, using aggressive quantization techniques and strict memory constraints. Analytical modeling is used to estimate latency, FLOPs, and energy consumption. The profiling reveals that 4-bit quantization reduces model memory usage by approximately 60-70%, while maintaining accuracy within 2-5% of full-precision baselines. Inference speeds are observed to improve by 2-3x compared to FP16 baselines across various edge devices. Power modeling estimates a 35-50% reduction in energy consumption for INT4 configurations, enabling practical deployment on hardware such as Raspberry Pi 4/5 and Jetson Orin Nano Super. Our findings emphasize the importance of efficient profiling tailored to lightweight LLMs in edge environments, balancing accuracy, energy efficiency, and computational feasibility. 

---
# Llama-Affinity: A Predictive Antibody Antigen Binding Model Integrating Antibody Sequences with Llama3 Backbone Architecture 

**Authors**: Delower Hossain, Ehsan Saghapour, Kevin Song, Jake Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09052)  

**Abstract**: Antibody-facilitated immune responses are central to the body's defense against pathogens, viruses, and other foreign invaders. The ability of antibodies to specifically bind and neutralize antigens is vital for maintaining immunity. Over the past few decades, bioengineering advancements have significantly accelerated therapeutic antibody development. These antibody-derived drugs have shown remarkable efficacy, particularly in treating cancer, SARS-CoV-2, autoimmune disorders, and infectious diseases. Traditionally, experimental methods for affinity measurement have been time-consuming and expensive. With the advent of artificial intelligence, in silico medicine has been revolutionized; recent developments in machine learning, particularly the use of large language models (LLMs) for representing antibodies, have opened up new avenues for AI-based design and improved affinity prediction. Herein, we present an advanced antibody-antigen binding affinity prediction model (LlamaAffinity), leveraging an open-source Llama 3 backbone and antibody sequence data sourced from the Observed Antibody Space (OAS) database. The proposed approach shows significant improvement over existing state-of-the-art (SOTA) methods (AntiFormer, AntiBERTa, AntiBERTy) across multiple evaluation metrics. Specifically, the model achieved an accuracy of 0.9640, an F1-score of 0.9643, a precision of 0.9702, a recall of 0.9586, and an AUC-ROC of 0.9936. Moreover, this strategy unveiled higher computational efficiency, with a five-fold average cumulative training time of only 0.46 hours, significantly lower than in previous studies. 

---
# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling 

**Authors**: Yang Liu, Jiaqi Li, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08672)  

**Abstract**: Rule-based reasoning has been acknowledged as one of the fundamental problems in reasoning, while deviations in rule formats, types, and complexity in real-world applications pose severe challenges. Recent studies have shown that large reasoning models (LRMs) have remarkable reasoning capabilities, and their performance is substantially enhanced by reinforcement learning (RL). However, it remains an open question whether small reasoning models (SRMs) can learn rule-based reasoning effectively with robust generalization across diverse tasks and domains. To address this, we introduce Reinforced Rule-based Reasoning, a.k.a. RuleReasoner, a simple yet effective method to conduct rule-based reasoning via a wide collection of curated tasks and a novel domain-aware dynamic sampling approach. Specifically, RuleReasoner resamples each training batch by updating the sampling weights of different domains based on historical rewards. This facilitates domain augmentation and flexible online learning schedules for RL, obviating the need for pre-hoc human-engineered mix-training recipes used in existing methods. Empirical evaluations on in-distribution (ID) and out-of-distribution (OOD) benchmarks reveal that RuleReasoner outperforms frontier LRMs by a significant margin ($\Delta$4.1% average points on eight ID tasks and $\Delta$10.4% average points on three OOD tasks over OpenAI-o1). Notably, our approach also exhibits higher computational efficiency compared to prior dynamic sampling methods for RL. 

---
# TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization 

**Authors**: Zengjue Chen, Runliang Niu, He Kong, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08440)  

**Abstract**: Recent advances in Vision-Language-Action (VLA) model have demonstrated strong generalization capabilities across diverse scenes, tasks, and robotic platforms when pretrained at large-scale datasets. However, these models still require task-specific fine-tuning in novel environments, a process that relies almost exclusively on supervised fine-tuning (SFT) using static trajectory datasets. Such approaches neither allow robot to interact with environment nor do they leverage feedback from live execution. Also, their success is critically dependent on the size and quality of the collected trajectories. Reinforcement learning (RL) offers a promising alternative by enabling closed-loop interaction and aligning learned policies directly with task objectives. In this work, we draw inspiration from the ideas of GRPO and propose the Trajectory-wise Group Relative Policy Optimization (TGRPO) method. By fusing step-level and trajectory-level advantage signals, this method improves GRPO's group-level advantage estimation, thereby making the algorithm more suitable for online reinforcement learning training of VLA. Experimental results on ten manipulation tasks from the libero-object benchmark demonstrate that TGRPO consistently outperforms various baseline methods, capable of generating more robust and efficient policies across multiple tested scenarios. Our source codes are available at: this https URL 

---
# An Interpretable N-gram Perplexity Threat Model for Large Language Model Jailbreaks 

**Authors**: Valentyn Boreiko, Alexander Panfilov, Vaclav Voracek, Matthias Hein, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2410.16222)  

**Abstract**: A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. These methods largely succeed in coercing the target output in their original settings, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model checks if a given jailbreak is likely to occur in the distribution of text. For this, we build an N-gram language model on 1T tokens, which, unlike model-based perplexity, allows for an LLM-agnostic, nonparametric, and inherently interpretable evaluation. We adapt popular attacks to this threat model, and, for the first time, benchmark these attacks on equal footing with it. After an extensive comparison, we find attack success rates against safety-tuned modern models to be lower than previously presented and that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent bigrams, either selecting the ones absent from real-world text or rare ones, e.g., specific to Reddit or code datasets. 

---
