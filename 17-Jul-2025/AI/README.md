# Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning 

**Authors**: Yuhao Chen, Shuochen Liu, Yuanjie Lyu, Chao Zhang, Jiayao Shi, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12215)  

**Abstract**: Game playing has long served as a fundamental benchmark for evaluating Artificial General Intelligence (AGI). While Large Language Models (LLMs) have demonstrated impressive capabilities in general reasoning, their effectiveness in spatial strategic reasoning, which is critical for complex and fully observable board games, remains insufficiently explored. In this work, we adopt Chinese Chess (Xiangqi) as a challenging and rich testbed due to its intricate rules and spatial complexity. To advance LLMs' strategic competence in such environments, we propose a training framework tailored to Xiangqi, built upon a large-scale dataset of five million board-move pairs enhanced with expert annotations and engine evaluations. Building on this foundation, we introduce Xiangqi-R1, a 7B-parameter model trained in multi-stage manner: (1) fine-tuning for legal move prediction to capture basic spatial rules, (2) incorporating strategic annotations to improve decision-making, and (3) applying reinforcement learning via Group Relative Policy Optimization (GRPO) with multi-dimensional reward signals to enhance reasoning stability. Our Experimental results indicate that, despite their size and power, general-purpose LLMs struggle to achieve satisfactory performance in these tasks. Compared to general-purpose LLMs, Xiangqi-R1 greatly advances with an 18% rise in move legality and a 22% boost in analysis accuracy. Our results point to a promising path for creating general strategic intelligence in spatially complex areas. 

---
# BuildEvo: Designing Building Energy Consumption Forecasting Heuristics via LLM-driven Evolution 

**Authors**: Subin Lin, Chuanbo Hua  

**Link**: [PDF](https://arxiv.org/pdf/2507.12207)  

**Abstract**: Accurate building energy forecasting is essential, yet traditional heuristics often lack precision, while advanced models can be opaque and struggle with generalization by neglecting physical principles. This paper introduces BuildEvo, a novel framework that uses Large Language Models (LLMs) to automatically design effective and interpretable energy prediction heuristics. Within an evolutionary process, BuildEvo guides LLMs to construct and enhance heuristics by systematically incorporating physical insights from building characteristics and operational data (e.g., from the Building Data Genome Project 2). Evaluations show BuildEvo achieves state-of-the-art performance on benchmarks, offering improved generalization and transparent prediction logic. This work advances the automated design of robust, physically grounded heuristics, promoting trustworthy models for complex energy systems. 

---
# Partially Observable Reference Policy Programming: Solving POMDPs Sans Numerical Optimisation 

**Authors**: Edward Kim, Hanna Kurniawati  

**Link**: [PDF](https://arxiv.org/pdf/2507.12186)  

**Abstract**: This paper proposes Partially Observable Reference Policy Programming, a novel anytime online approximate POMDP solver which samples meaningful future histories very deeply while simultaneously forcing a gradual policy update. We provide theoretical guarantees for the algorithm's underlying scheme which say that the performance loss is bounded by the average of the sampling approximation errors rather than the usual maximum, a crucial requirement given the sampling sparsity of online planning. Empirical evaluations on two large-scale problems with dynamically evolving environments -- including a helicopter emergency scenario in the Corsica region requiring approximately 150 planning steps -- corroborate the theoretical results and indicate that our solver considerably outperforms current online benchmarks. 

---
# Topology Enhanced MARL for Multi-Vehicle Cooperative Decision-Making of CAVs 

**Authors**: Ye Han, Lijun Zhang, Dejian Meng, Zhuang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12110)  

**Abstract**: The exploration-exploitation trade-off constitutes one of the fundamental challenges in reinforcement learning (RL), which is exacerbated in multi-agent reinforcement learning (MARL) due to the exponential growth of joint state-action spaces. This paper proposes a topology-enhanced MARL (TPE-MARL) method for optimizing cooperative decision-making of connected and autonomous vehicles (CAVs) in mixed traffic. This work presents two primary contributions: First, we construct a game topology tensor for dynamic traffic flow, effectively compressing high-dimensional traffic state information and decrease the search space for MARL algorithms. Second, building upon the designed game topology tensor and using QMIX as the backbone RL algorithm, we establish a topology-enhanced MARL framework incorporating visit counts and agent mutual information. Extensive simulations across varying traffic densities and CAV penetration rates demonstrate the effectiveness of TPE-MARL. Evaluations encompassing training dynamics, exploration patterns, macroscopic traffic performance metrics, and microscopic vehicle behaviors reveal that TPE-MARL successfully balances exploration and exploitation. Consequently, it exhibits superior performance in terms of traffic efficiency, safety, decision smoothness, and task completion. Furthermore, the algorithm demonstrates decision-making rationality comparable to or exceeding that of human drivers in both mixed-autonomy and fully autonomous traffic scenarios. Code of our work is available at \href{this https URL}{this https URL}. 

---
# Understanding visual attention beehind bee-inspired UAV navigation 

**Authors**: Pranav Rajbhandari, Abhi Veda, Matthew Garratt, Mandayam Srinivasan, Sridhar Ravi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11992)  

**Abstract**: Bio-inspired design is often used in autonomous UAV navigation due to the capacity of biological systems for flight and obstacle avoidance despite limited sensory and computational capabilities. In particular, honeybees mainly use the sensory input of optic flow, the apparent motion of objects in their visual field, to navigate cluttered environments. In our work, we train a Reinforcement Learning agent to navigate a tunnel with obstacles using only optic flow as sensory input. We inspect the attention patterns of trained agents to determine the regions of optic flow on which they primarily base their motor decisions. We find that agents trained in this way pay most attention to regions of discontinuity in optic flow, as well as regions with large optic flow magnitude. The trained agents appear to navigate a cluttered tunnel by avoiding the obstacles that produce large optic flow, while maintaining a centered position in their environment, which resembles the behavior seen in flying insects. This pattern persists across independently trained agents, which suggests that this could be a good strategy for developing a simple explicit control law for physical UAVs. 

---
# Aime: Towards Fully-Autonomous Multi-Agent Framework 

**Authors**: Yexuan Shi, Mingyu Wang, Yunxiang Cao, Hongjie Lai, Junjian Lan, Xin Han, Yu Wang, Jie Geng, Zhenan Li, Zihao Xia, Xiang Chen, Chen Li, Jian Xu, Wenbo Duan, Yuanshuo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11988)  

**Abstract**: Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) are emerging as a powerful paradigm for solving complex, multifaceted problems. However, the potential of these systems is often constrained by the prevalent plan-and-execute framework, which suffers from critical limitations: rigid plan execution, static agent capabilities, and inefficient communication. These weaknesses hinder their adaptability and robustness in dynamic environments. This paper introduces Aime, a novel multi-agent framework designed to overcome these challenges through dynamic, reactive planning and execution. Aime replaces the conventional static workflow with a fluid and adaptive architecture. Its core innovations include: (1) a Dynamic Planner that continuously refines the overall strategy based on real-time execution feedback; (2) an Actor Factory that implements Dynamic Actor instantiation, assembling specialized agents on-demand with tailored tools and knowledge; and (3) a centralized Progress Management Module that serves as a single source of truth for coherent, system-wide state awareness. We empirically evaluated Aime on a diverse suite of benchmarks spanning general reasoning (GAIA), software engineering (SWE-bench Verified), and live web navigation (WebVoyager). The results demonstrate that Aime consistently outperforms even highly specialized state-of-the-art agents in their respective domains. Its superior adaptability and task success rate establish Aime as a more resilient and effective foundation for multi-agent collaboration. 

---
# A Parallel CPU-GPU Framework for Cost-Bounded DFS with Applications to IDA* and BTS 

**Authors**: Ehsan Futuhi, Nathan R. Sturtevant  

**Link**: [PDF](https://arxiv.org/pdf/2507.11916)  

**Abstract**: The rapid advancement of GPU technology has unlocked powerful parallel processing capabilities, creating new opportunities to enhance classic search algorithms. A recent successful application of GPUs is in compressing large pattern database (PDB) heuristics using neural networks while preserving heuristic admissibility. However, very few algorithms have been designed to exploit GPUs during search. Several variants of A* exist that batch GPU computations. In this paper we introduce a method for batching GPU computations in depth first search. In particular, we describe a new cost-bounded depth-first search (CB-DFS) method that leverages the combined parallelism of modern CPUs and GPUs. This is used to create algorithms like \emph{Batch IDA*}, an extension of the Iterative Deepening A* (IDA*) algorithm, or Batch BTS, an extensions of Budgeted Tree Search. Our approach builds on the general approach used by Asynchronous Parallel IDA* (AIDA*), while maintaining optimality guarantees. We evaluate the approach on the 3x3 Rubik's Cube and 4x4 sliding tile puzzle (STP), showing that GPU operations can be efficiently batched in DFS. Additionally, we conduct extensive experiments to analyze the effects of hyperparameters, neural network heuristic size, and hardware resources on performance. 

---
# Survey of Swarm Intelligence Approaches to Search Documents Based On Semantic Similarity 

**Authors**: Chandrashekar Muniyappa, Eunjin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.11787)  

**Abstract**: Swarm Intelligence (SI) is gaining a lot of popularity in artificial intelligence, where the natural behavior of animals and insects is observed and translated into computer algorithms called swarm computing to solve real-world problems. Due to their effectiveness, they are applied in solving various computer optimization problems. This survey will review all the latest developments in Searching for documents based on semantic similarity using Swarm Intelligence algorithms and recommend future research directions. 

---
# Auto-Formulating Dynamic Programming Problems with Large Language Models 

**Authors**: Chenyu Zhou, Jingyuan Yang, Linwei Xin, Yitian Chen, Ziyan He, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2507.11737)  

**Abstract**: Dynamic programming (DP) is a fundamental method in operations research, but formulating DP models has traditionally required expert knowledge of both the problem context and DP techniques. Large Language Models (LLMs) offer the potential to automate this process. However, DP problems pose unique challenges due to their inherently stochastic transitions and the limited availability of training data. These factors make it difficult to directly apply existing LLM-based models or frameworks developed for other optimization problems, such as linear or integer programming. We introduce DP-Bench, the first benchmark covering a wide range of textbook-level DP problems to enable systematic evaluation. We present Dynamic Programming Language Model (DPLM), a 7B-parameter specialized model that achieves performance comparable to state-of-the-art LLMs like OpenAI's o1 and DeepSeek-R1, and surpasses them on hard problems. Central to DPLM's effectiveness is DualReflect, our novel synthetic data generation pipeline, designed to scale up training data from a limited set of initial examples. DualReflect combines forward generation for diversity and backward generation for reliability. Our results reveal a key insight: backward generation is favored in low-data regimes for its strong correctness guarantees, while forward generation, though lacking such guarantees, becomes increasingly valuable at scale for introducing diverse formulations. This trade-off highlights the complementary strengths of both approaches and the importance of combining them. 

---
# ClarifAI: Enhancing AI Interpretability and Transparency through Case-Based Reasoning and Ontology-Driven Approach for Improved Decision-Making 

**Authors**: Srikanth Vemula  

**Link**: [PDF](https://arxiv.org/pdf/2507.11733)  

**Abstract**: This Study introduces Clarity and Reasoning Interface for Artificial Intelligence(ClarifAI), a novel approach designed to augment the transparency and interpretability of artificial intelligence (AI) in the realm of improved decision making. Leveraging the Case-Based Reasoning (CBR) methodology and integrating an ontology-driven approach, ClarifAI aims to meet the intricate explanatory demands of various stakeholders involved in AI-powered applications. The paper elaborates on ClarifAI's theoretical foundations, combining CBR and ontologies to furnish exhaustive explanation mechanisms. It further elaborates on the design principles and architectural blueprint, highlighting ClarifAI's potential to enhance AI interpretability across different sectors and its applicability in high-stake environments. This research delineates the significant role of ClariAI in advancing the interpretability of AI systems, paving the way for its deployment in critical decision-making processes. 

---
# Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification 

**Authors**: Moises Andrade, Joonhyuk Cha, Brandon Ho, Vriksha Srihari, Karmesh Yadav, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2507.11662)  

**Abstract**: Verifiers -- functions assigning rewards to agent behavior -- have been key for AI progress in domains like math and board games. However, extending these gains to domains without clear-cut success criteria (e.g.,computer use) remains a challenge: while humans can recognize suitable outcomes, translating this intuition into scalable rules is non-trivial. Multimodal Large Language Models(MLLMs) emerge as a promising solution, given their world knowledge, human-preference alignment, and reasoning skills. We evaluate MLLMs as verifiers of agent trajectories across web navigation, computer use, and robotic manipulation, and identify a critical limitation: agreement bias, a strong tendency for MLLMs to favor information in their context window, often generating chains of thought to rationalize flawed behavior. This bias is pervasive across models, resilient to test-time scaling, and can impact several methods using MLLMs as evaluators (e.g.,data filtering). Notably, it occurs despite MLLMs showing strong, human-aligned priors on desired behavior. To address this, we propose Self-Grounded Verification (SGV), a lightweight method that enables more effective use of MLLMs' knowledge and reasoning by harnessing their own sampling mechanisms via unconditional and conditional generation. SGV operates in two steps: first, the MLLM is elicited to retrieve broad priors about task completion, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. Enhanced with SGV, MLLM verifiers show gains of up to 20 points in accuracy and failure detection rates, and can perform real-time supervision of heterogeneous agents, boosting task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena -- setting a new state of the art on the benchmark, surpassing the previous best by 48%. 

---
# General Modular Harness for LLM Agents in Multi-Turn Gaming Environments 

**Authors**: Yuxuan Zhang, Haoyang Yu, Lanxiang Hu, Haojian Jin, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11633)  

**Abstract**: We introduce a modular harness design for LLM agents that composes of perception, memory, and reasoning components, enabling a single LLM or VLM backbone to tackle a wide spectrum of multi turn gaming environments without domain-specific engineering. Using classic and modern game suites as low-barrier, high-diversity testbeds, our framework provides a unified workflow for analyzing how each module affects performance across dynamic interactive settings. Extensive experiments demonstrate that the harness lifts gameplay performance consistently over un-harnessed baselines and reveals distinct contribution patterns, for example, memory dominates in long-horizon puzzles while perception is critical in vision noisy arcades. These findings highlight the effectiveness of our modular harness design in advancing general-purpose agent, given the familiarity and ubiquity of games in everyday human experience. 

---
# A Study on the Application of Artificial Intelligence in Ecological Design 

**Authors**: Hengyue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11595)  

**Abstract**: This paper asks whether our relationship with nature can move from human dominance to genuine interdependence, and whether artificial intelligence (AI) can mediate that shift. We examine a new ecological-design paradigm in which AI interacts with non-human life forms. Through case studies we show how artists and designers apply AI for data analysis, image recognition, and ecological restoration, producing results that differ from conventional media. We argue that AI not only expands creative methods but also reframes the theory and practice of ecological design. Building on the author's prototype for AI-assisted water remediation, the study proposes design pathways that couple reinforcement learning with plant-based phytoremediation. The findings highlight AI's potential to link scientific insight, artistic practice, and environmental stewardship, offering a roadmap for future research on sustainable, technology-enabled ecosystems. 

---
# Interpreting Radiologist's Intention from Eye Movements in Chest X-ray Diagnosis 

**Authors**: Trong-Thang Pham, Anh Nguyen, Zhigang Deng, Carol C. Wu, Hien Van Nguyen, Ngan Le  

**Link**: [PDF](https://arxiv.org/pdf/2507.12461)  

**Abstract**: Radiologists rely on eye movements to navigate and interpret medical images. A trained radiologist possesses knowledge about the potential diseases that may be present in the images and, when searching, follows a mental checklist to locate them using their gaze. This is a key observation, yet existing models fail to capture the underlying intent behind each fixation. In this paper, we introduce a deep learning-based approach, RadGazeIntent, designed to model this behavior: having an intention to find something and actively searching for it. Our transformer-based architecture processes both the temporal and spatial dimensions of gaze data, transforming fine-grained fixation features into coarse, meaningful representations of diagnostic intent to interpret radiologists' goals. To capture the nuances of radiologists' varied intention-driven behaviors, we process existing medical eye-tracking datasets to create three intention-labeled subsets: RadSeq (Systematic Sequential Search), RadExplore (Uncertainty-driven Exploration), and RadHybrid (Hybrid Pattern). Experimental results demonstrate RadGazeIntent's ability to predict which findings radiologists are examining at specific moments, outperforming baseline methods across all intention-labeled datasets. 

---
# S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling 

**Authors**: Suman Adhya, Debarshi Kumar Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.12451)  

**Abstract**: Modeling latent representations in a hyperspherical space has proven effective for capturing directional similarities in high-dimensional text data, benefiting topic modeling. Variational autoencoder-based neural topic models (VAE-NTMs) commonly adopt the von Mises-Fisher prior to encode hyperspherical structure. However, VAE-NTMs often suffer from posterior collapse, where the KL divergence term in the objective function highly diminishes, leading to ineffective latent representations. To mitigate this issue while modeling hyperspherical structure in the latent space, we propose the Spherical Sliced Wasserstein Autoencoder for Topic Modeling (S2WTM). S2WTM employs a prior distribution supported on the unit hypersphere and leverages the Spherical Sliced-Wasserstein distance to align the aggregated posterior distribution with the prior. Experimental results demonstrate that S2WTM outperforms state-of-the-art topic models, generating more coherent and diverse topics while improving performance on downstream tasks. 

---
# LLM-Based Config Synthesis requires Disambiguation 

**Authors**: Rajdeep Mondal, Nikolaj Bjorner, Todd Millstein, Alan Tang, George Varghese  

**Link**: [PDF](https://arxiv.org/pdf/2507.12443)  

**Abstract**: Beyond hallucinations, another problem in program synthesis using LLMs is ambiguity in user intent. We illustrate the ambiguity problem in a networking context for LLM-based incremental configuration synthesis of route-maps and ACLs. These structures frequently overlap in header space, making the relative priority of actions impossible for the LLM to infer without user interaction. Measurements in a large cloud identify complex ACLs with 100's of overlaps, showing ambiguity is a real problem. We propose a prototype system, Clarify, which uses an LLM augmented with a new module called a Disambiguator that helps elicit user intent. On a small synthetic workload, Clarify incrementally synthesizes routing policies after disambiguation and then verifies them. Our treatment of ambiguities is useful more generally when the intent of updates can be correctly synthesized by LLMs, but their integration is ambiguous and can lead to different global behaviors. 

---
# Characterizing State Space Model (SSM) and SSM-Transformer Hybrid Language Model Performance with Long Context Length 

**Authors**: Saptarshi Mitra, Rachid Karami, Haocheng Xu, Sitao Huang, Hyoukjun Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2507.12442)  

**Abstract**: The demand for machine intelligence capable of processing continuous, long-context inputs on local devices is growing rapidly. However, the quadratic complexity and memory requirements of traditional Transformer architectures make them inefficient and often unusable for these tasks. This has spurred a paradigm shift towards new architectures like State Space Models (SSMs) and hybrids, which promise near-linear scaling. While most current research focuses on the accuracy and theoretical throughput of these models, a systematic performance characterization on practical consumer hardware is critically needed to guide system-level optimization and unlock new applications.
To address this gap, we present a comprehensive, comparative benchmarking of carefully selected Transformer, SSM, and hybrid models specifically for long-context inference on consumer and embedded GPUs. Our analysis reveals that SSMs are not only viable but superior for this domain, capable of processing sequences up to 220K tokens on a 24GB consumer GPU-approximately 4x longer than comparable Transformers. While Transformers may be up to 1.8x faster at short sequences, SSMs demonstrate a dramatic performance inversion, becoming up to 4x faster at very long contexts (~57K tokens). Our operator-level analysis reveals that custom, hardware-aware SSM kernels dominate the inference runtime, accounting for over 55% of latency on edge platforms, identifying them as a primary target for future hardware acceleration. We also provide detailed, device-specific characterization results to guide system co-design for the edge. To foster further research, we will open-source our characterization framework. 

---
# EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos 

**Authors**: Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12440)  

**Abstract**: Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Isaac Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Isaac Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: this https URL 

---
# Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models 

**Authors**: Yik Siu Chan, Zheng-Xin Yong, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2507.12428)  

**Abstract**: Open-weights reasoning language models generate long chains-of-thought (CoTs) before producing a final response, which improves performance but introduces additional alignment risks, with harmful content often appearing in both the CoTs and the final outputs. In this work, we investigate if we can use CoTs to predict final response misalignment. We evaluate a range of monitoring approaches, including humans, highly-capable large language models, and text classifiers, using either CoT text or activations. First, we find that a simple linear probe trained on CoT activations can significantly outperform all text-based methods in predicting whether a final response will be safe or unsafe. CoT texts are often unfaithful and can mislead humans and classifiers, while model latents (i.e., CoT activations) offer a more reliable predictive signal. Second, the probe makes accurate predictions before reasoning completes, achieving strong performance even when applied to early CoT segments. These findings generalize across model sizes, families, and safety benchmarks, suggesting that lightweight probes could enable real-time safety monitoring and early intervention during generation. 

---
# Unit-Based Histopathology Tissue Segmentation via Multi-Level Feature Representation 

**Authors**: Ashkan Shakarami, Azade Farshad, Yousef Yeganeh, Lorenzo Nicole, Peter Schuffler, Stefano Ghidoni, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2507.12427)  

**Abstract**: We propose UTS, a unit-based tissue segmentation framework for histopathology that classifies each fixed-size 32 * 32 tile, rather than each pixel, as the segmentation unit. This approach reduces annotation effort and improves computational efficiency without compromising accuracy. To implement this approach, we introduce a Multi-Level Vision Transformer (L-ViT), which benefits the multi-level feature representation to capture both fine-grained morphology and global tissue context. Trained to segment breast tissue into three categories (infiltrating tumor, non-neoplastic stroma, and fat), UTS supports clinically relevant tasks such as tumor-stroma quantification and surgical margin assessment. Evaluated on 386,371 tiles from 459 H&E-stained regions, it outperforms U-Net variants and transformer-based baselines. Code and Dataset will be available at GitHub. 

---
# Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data 

**Authors**: Chandana Cheerla  

**Link**: [PDF](https://arxiv.org/pdf/2507.12425)  

**Abstract**: Organizations increasingly rely on proprietary enterprise data, including HR records, structured reports, and tabular documents, for critical decision-making. While Large Language Models (LLMs) have strong generative capabilities, they are limited by static pretraining, short context windows, and challenges in processing heterogeneous data formats. Conventional Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but often struggle with structured and semi-structured data.
This work proposes an advanced RAG framework that combines hybrid retrieval strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by metadata-aware filtering with SpaCy NER and cross-encoder reranking. The framework applies semantic chunking to maintain textual coherence and retains tabular data structures to preserve row-column integrity. Quantized indexing optimizes retrieval efficiency, while human-in-the-loop feedback and conversation memory improve adaptability.
Experiments on enterprise datasets show notable improvements: Precision@5 increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74), and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness (4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale. These results demonstrate the framework's effectiveness in delivering accurate, comprehensive, and contextually relevant responses for enterprise tasks. Future work includes extending to multimodal data and integrating agent-based retrieval. The source code will be released at this https URL 

---
# Mixture of Raytraced Experts 

**Authors**: Andrea Perin, Giacomo Lagomarsini, Claudio Gallicchio, Giuseppe Nuti  

**Link**: [PDF](https://arxiv.org/pdf/2507.12419)  

**Abstract**: We introduce a Mixture of Raytraced Experts, a stacked Mixture of Experts (MoE) architecture which can dynamically select sequences of experts, producing computational graphs of variable width and depth. Existing MoE architectures generally require a fixed amount of computation for a given sample. Our approach, in contrast, yields predictions with increasing accuracy as the computation cycles through the experts' sequence. We train our model by iteratively sampling from a set of candidate experts, unfolding the sequence akin to how Recurrent Neural Networks are trained. Our method does not require load-balancing mechanisms, and preliminary experiments show a reduction in training epochs of 10\% to 40\% with a comparable/higher accuracy. These results point to new research directions in the field of MoEs, allowing the design of potentially faster and more expressive models. The code is available at this https URL 

---
# QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval 

**Authors**: Jaehyun Kwak, Ramahdani Muhammad Izaaz Inhar, Se-Young Yun, Sung-Ju Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.12416)  

**Abstract**: Composed Image Retrieval (CIR) retrieves relevant images based on a reference image and accompanying text describing desired modifications. However, existing CIR methods only focus on retrieving the target image and disregard the relevance of other images. This limitation arises because most methods employing contrastive learning-which treats the target image as positive and all other images in the batch as negatives-can inadvertently include false negatives. This may result in retrieving irrelevant images, reducing user satisfaction even when the target image is retrieved. To address this issue, we propose Query-Relevant Retrieval through Hard Negative Sampling (QuRe), which optimizes a reward model objective to reduce false negatives. Additionally, we introduce a hard negative sampling strategy that selects images positioned between two steep drops in relevance scores following the target image, to effectively filter false negatives. In order to evaluate CIR models on their alignment with human satisfaction, we create Human-Preference FashionIQ (HP-FashionIQ), a new dataset that explicitly captures user preferences beyond target retrieval. Extensive experiments demonstrate that QuRe achieves state-of-the-art performance on FashionIQ and CIRR datasets while exhibiting the strongest alignment with human preferences on the HP-FashionIQ dataset. The source code is available at this https URL. 

---
# AutoVDC: Automated Vision Data Cleaning Using Vision-Language Models 

**Authors**: Santosh Vasa, Aditi Ramadwar, Jnana Rama Krishna Darabattula, Md Zafar Anwar, Stanislaw Antol, Andrei Vatavu, Thomas Monninger, Sihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.12414)  

**Abstract**: Training of autonomous driving systems requires extensive datasets with precise annotations to attain robust performance. Human annotations suffer from imperfections, and multiple iterations are often needed to produce high-quality datasets. However, manually reviewing large datasets is laborious and expensive. In this paper, we introduce AutoVDC (Automated Vision Data Cleaning) framework and investigate the utilization of Vision-Language Models (VLMs) to automatically identify erroneous annotations in vision datasets, thereby enabling users to eliminate these errors and enhance data quality. We validate our approach using the KITTI and nuImages datasets, which contain object detection benchmarks for autonomous driving. To test the effectiveness of AutoVDC, we create dataset variants with intentionally injected erroneous annotations and observe the error detection rate of our approach. Additionally, we compare the detection rates using different VLMs and explore the impact of VLM fine-tuning on our pipeline. The results demonstrate our method's high performance in error detection and data cleaning experiments, indicating its potential to significantly improve the reliability and accuracy of large-scale production datasets in autonomous driving. 

---
# NOCTA: Non-Greedy Objective Cost-Tradeoff Acquisition for Longitudinal Data 

**Authors**: Dzung Dinh, Boqi Chen, Marc Niethammer, Junier Oliva  

**Link**: [PDF](https://arxiv.org/pdf/2507.12412)  

**Abstract**: In many critical applications, resource constraints limit the amount of information that can be gathered to make predictions. For example, in healthcare, patient data often spans diverse features ranging from lab tests to imaging studies. Each feature may carry different information and must be acquired at a respective cost of time, money, or risk to the patient. Moreover, temporal prediction tasks, where both instance features and labels evolve over time, introduce additional complexity in deciding when or what information is important. In this work, we propose NOCTA, a Non-Greedy Objective Cost-Tradeoff Acquisition method that sequentially acquires the most informative features at inference time while accounting for both temporal dynamics and acquisition cost. We first introduce a cohesive estimation target for our NOCTA setting, and then develop two complementary estimators: 1) a non-parametric method based on nearest neighbors to guide the acquisition (NOCTA-NP), and 2) a parametric method that directly predicts the utility of potential acquisitions (NOCTA-P). Experiments on synthetic and real-world medical datasets demonstrate that both NOCTA variants outperform existing baselines. 

---
# Probing for Arithmetic Errors in Language Models 

**Authors**: Yucheng Sun, Alessandro Stolfo, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12379)  

**Abstract**: We investigate whether internal activations in language models can be used to detect arithmetic errors. Starting with a controlled setting of 3-digit addition, we show that simple probes can accurately decode both the model's predicted output and the correct answer from hidden states, regardless of whether the model's output is correct. Building on this, we train lightweight error detectors that predict model correctness with over 90% accuracy. We then extend our analysis to structured chain-of-thought traces on addition-only GSM8K problems and find that probes trained on simple arithmetic generalize well to this more complex setting, revealing consistent internal representations. Finally, we demonstrate that these probes can guide selective re-prompting of erroneous reasoning steps, improving task accuracy with minimal disruption to correct outputs. Our findings suggest that arithmetic errors can be anticipated from internal activations alone, and that simple probes offer a viable path toward lightweight model self-correction. 

---
# GitChameleon: Evaluating AI Code Generation Against Python Library Version Incompatibilities 

**Authors**: Diganta Misra, Nizar Islah, Victor May, Brice Rauby, Zihan Wang, Justine Gehring, Antonio Orvieto, Muawiz Chaudhary, Eilif B. Muller, Irina Rish, Samira Ebrahimi Kahou, Massimo Caccia  

**Link**: [PDF](https://arxiv.org/pdf/2507.12367)  

**Abstract**: The rapid evolution of software libraries poses a considerable hurdle for code generation, necessitating continuous adaptation to frequent version updates while preserving backward compatibility. While existing code evolution benchmarks provide valuable insights, they typically lack execution-based evaluation for generating code compliant with specific library versions. To address this, we introduce GitChameleon, a novel, meticulously curated dataset comprising 328 Python code completion problems, each conditioned on specific library versions and accompanied by executable unit tests. GitChameleon rigorously evaluates the capacity of contemporary large language models (LLMs), LLM-powered agents, code assistants, and RAG systems to perform version-conditioned code generation that demonstrates functional accuracy through execution. Our extensive evaluations indicate that state-of-the-art systems encounter significant challenges with this task; enterprise models achieving baseline success rates in the 48-51\% range, underscoring the intricacy of the problem. By offering an execution-based benchmark emphasizing the dynamic nature of code libraries, GitChameleon enables a clearer understanding of this challenge and helps guide the development of more adaptable and dependable AI code generation methods. We make the dataset and evaluation code publicly available at this https URL. 

---
# FactorHD: A Hyperdimensional Computing Model for Multi-Object Multi-Class Representation and Factorization 

**Authors**: Yifei Zhou, Xuchu Huang, Chenyu Ni, Min Zhou, Zheyu Yan, Xunzhao Yin, Cheng Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2507.12366)  

**Abstract**: Neuro-symbolic artificial intelligence (neuro-symbolic AI) excels in logical analysis and reasoning. Hyperdimensional Computing (HDC), a promising brain-inspired computational model, is integral to neuro-symbolic AI. Various HDC models have been proposed to represent class-instance and class-class relations, but when representing the more complex class-subclass relation, where multiple objects associate different levels of classes and subclasses, they face challenges for factorization, a crucial task for neuro-symbolic AI systems. In this article, we propose FactorHD, a novel HDC model capable of representing and factorizing the complex class-subclass relation efficiently. FactorHD features a symbolic encoding method that embeds an extra memorization clause, preserving more information for multiple objects. In addition, it employs an efficient factorization algorithm that selectively eliminates redundant classes by identifying the memorization clause of the target class. Such model significantly enhances computing efficiency and accuracy in representing and factorizing multiple objects with class-subclass relation, overcoming limitations of existing HDC models such as "superposition catastrophe" and "the problem of 2". Evaluations show that FactorHD achieves approximately 5667x speedup at a representation size of 10^9 compared to existing HDC models. When integrated with the ResNet-18 neural network, FactorHD achieves 92.48% factorization accuracy on the Cifar-10 dataset. 

---
# Cluster Contrast for Unsupervised Visual Representation Learning 

**Authors**: Nikolaos Giakoumoglou, Tania Stathaki  

**Link**: [PDF](https://arxiv.org/pdf/2507.12359)  

**Abstract**: We introduce Cluster Contrast (CueCo), a novel approach to unsupervised visual representation learning that effectively combines the strengths of contrastive learning and clustering methods. Inspired by recent advancements, CueCo is designed to simultaneously scatter and align feature representations within the feature space. This method utilizes two neural networks, a query and a key, where the key network is updated through a slow-moving average of the query outputs. CueCo employs a contrastive loss to push dissimilar features apart, enhancing inter-class separation, and a clustering objective to pull together features of the same cluster, promoting intra-class compactness. Our method achieves 91.40% top-1 classification accuracy on CIFAR-10, 68.56% on CIFAR-100, and 78.65% on ImageNet-100 using linear evaluation with a ResNet-18 backbone. By integrating contrastive learning with clustering, CueCo sets a new direction for advancing unsupervised visual representation learning. 

---
# Neural Polar Decoders for Deletion Channels 

**Authors**: Ziv Aharoni, Henry D. Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2507.12329)  

**Abstract**: This paper introduces a neural polar decoder (NPD) for deletion channels with a constant deletion rate. Existing polar decoders for deletion channels exhibit high computational complexity of $O(N^4)$, where $N$ is the block length. This limits the application of polar codes for deletion channels to short-to-moderate block lengths. In this work, we demonstrate that employing NPDs for deletion channels can reduce the computational complexity. First, we extend the architecture of the NPD to support deletion channels. Specifically, the NPD architecture consists of four neural networks (NNs), each replicating fundamental successive cancellation (SC) decoder operations. To support deletion channels, we change the architecture of only one. The computational complexity of the NPD is $O(AN\log N)$, where the parameter $A$ represents a computational budget determined by the user and is independent of the channel. We evaluate the new extended NPD for deletion channels with deletion rates $\delta\in\{0.01, 0.1\}$ and we verify the NPD with the ground truth given by the trellis decoder by Tal et al. We further show that due to the reduced complexity of the NPD, we are able to incorporate list decoding and further improve performance. We believe that the extended NPD presented here could have applications in future technologies like DNA storage. 

---
# Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models 

**Authors**: Samuel Lavoie, Michael Noukhovitch, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2507.12318)  

**Abstract**: We argue that diffusion models' success in modeling complex distributions is, for the most part, coming from their input conditioning. This paper investigates the representation used to condition diffusion models from the perspective that ideal representations should improve sample fidelity, be easy to generate, and be compositional to allow out-of-training samples generation. We introduce Discrete Latent Code (DLC), an image representation derived from Simplicial Embeddings trained with a self-supervised learning objective. DLCs are sequences of discrete tokens, as opposed to the standard continuous image embeddings. They are easy to generate and their compositionality enables sampling of novel images beyond the training distribution. Diffusion models trained with DLCs have improved generation fidelity, establishing a new state-of-the-art for unconditional image generation on ImageNet. Additionally, we show that composing DLCs allows the image generator to produce out-of-distribution samples that coherently combine the semantics of images in diverse ways. Finally, we showcase how DLCs can enable text-to-image generation by leveraging large-scale pretrained language models. We efficiently finetune a text diffusion language model to generate DLCs that produce novel samples outside of the image generator training distribution. 

---
# Thought Purity: Defense Paradigm For Chain-of-Thought Attack 

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2507.12314)  

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures. 

---
# Chain-of-Descriptions: Improving Code LLMs for VHDL Code Generation and Summarization 

**Authors**: Prashanth Vijayaraghavan, Apoorva Nitsure, Charles Mackin, Luyao Shi, Stefano Ambrogio, Arvind Haran, Viresh Paruthi, Ali Elzein, Dan Coops, David Beymer, Tyler Baldwin, Ehsan Degan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12308)  

**Abstract**: Large Language Models (LLMs) have become widely used across diverse NLP tasks and domains, demonstrating their adaptability and effectiveness. In the realm of Electronic Design Automation (EDA), LLMs show promise for tasks like Register-Transfer Level (RTL) code generation and summarization. However, despite the proliferation of LLMs for general code-related tasks, there's a dearth of research focused on evaluating and refining these models for hardware description languages (HDLs), notably VHDL. In this study, we evaluate the performance of existing code LLMs for VHDL code generation and summarization using various metrics and two datasets -- VHDL-Eval and VHDL-Xform. The latter, an in-house dataset, aims to gauge LLMs' understanding of functionally equivalent code. Our findings reveal consistent underperformance of these models across different metrics, underscoring a significant gap in their suitability for this domain. To address this challenge, we propose Chain-of-Descriptions (CoDes), a novel approach to enhance the performance of LLMs for VHDL code generation and summarization tasks. CoDes involves generating a series of intermediate descriptive steps based on: (i) the problem statement for code generation, and (ii) the VHDL code for summarization. These steps are then integrated with the original input prompt (problem statement or code) and provided as input to the LLMs to generate the final output. Our experiments demonstrate that the CoDes approach significantly surpasses the standard prompting strategy across various metrics on both datasets. This method not only improves the quality of VHDL code generation and summarization but also serves as a framework for future research aimed at enhancing code LLMs for VHDL. 

---
# PROL : Rehearsal Free Continual Learning in Streaming Data via Prompt Online Learning 

**Authors**: M. Anwar Ma'sum, Mahardhika Pratama, Savitha Ramasamy, Lin Liu, Habibullah Habibullah, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.12305)  

**Abstract**: The data privacy constraint in online continual learning (OCL), where the data can be seen only once, complicates the catastrophic forgetting problem in streaming data. A common approach applied by the current SOTAs in OCL is with the use of memory saving exemplars or features from previous classes to be replayed in the current task. On the other hand, the prompt-based approach performs excellently in continual learning but with the cost of a growing number of trainable parameters. The first approach may not be applicable in practice due to data openness policy, while the second approach has the issue of throughput associated with the streaming data. In this study, we propose a novel prompt-based method for online continual learning that includes 4 main components: (1) single light-weight prompt generator as a general knowledge, (2) trainable scaler-and-shifter as specific knowledge, (3) pre-trained model (PTM) generalization preserving, and (4) hard-soft updates mechanism. Our proposed method achieves significantly higher performance than the current SOTAs in CIFAR100, ImageNet-R, ImageNet-A, and CUB dataset. Our complexity analysis shows that our method requires a relatively smaller number of parameters and achieves moderate training time, inference time, and throughput. For further study, the source code of our method is available at this https URL. 

---
# Text-ADBench: Text Anomaly Detection Benchmark based on LLMs Embedding 

**Authors**: Feng Xiao, Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12295)  

**Abstract**: Text anomaly detection is a critical task in natural language processing (NLP), with applications spanning fraud detection, misinformation identification, spam detection and content moderation, etc. Despite significant advances in large language models (LLMs) and anomaly detection algorithms, the absence of standardized and comprehensive benchmarks for evaluating the existing anomaly detection methods on text data limits rigorous comparison and development of innovative approaches. This work performs a comprehensive empirical study and introduces a benchmark for text anomaly detection, leveraging embeddings from diverse pre-trained language models across a wide array of text datasets. Our work systematically evaluates the effectiveness of embedding-based text anomaly detection by incorporating (1) early language models (GloVe, BERT); (2) multiple LLMs (LLaMa-2, LLama-3, Mistral, OpenAI (small, ada, large)); (3) multi-domain text datasets (news, social media, scientific publications); (4) comprehensive evaluation metrics (AUROC, AUPRC). Our experiments reveal a critical empirical insight: embedding quality significantly governs anomaly detection efficacy, and deep learning-based approaches demonstrate no performance advantage over conventional shallow algorithms (e.g., KNN, Isolation Forest) when leveraging LLM-derived this http URL addition, we observe strongly low-rank characteristics in cross-model performance matrices, which enables an efficient strategy for rapid model evaluation (or embedding evaluation) and selection in practical applications. Furthermore, by open-sourcing our benchmark toolkit that includes all embeddings from different models and code at this https URL, this work provides a foundation for future research in robust and scalable text anomaly detection systems. 

---
# MERA Code: A Unified Framework for Evaluating Code Generation Across Tasks 

**Authors**: Artem Chervyakov, Alexander Kharitonov, Pavel Zadorozhny, Adamenko Pavel, Rodion Levichev, Dmitrii Vorobev, Dmitrii Salikhov, Aidar Valeev, Alena Pestova, Maria Dziuba, Ilseyar Alimova, Artem Zavgorodnev, Aleksandr Medvedev, Stanislav Moiseev, Elena Bruches, Daniil Grebenkin, Roman Derunets, Vikulov Vladimir, Anton Emelyanov, Dmitrii Babaev, Vladimir V. Ivanov, Valentin Malykh, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2507.12284)  

**Abstract**: Advancements in LLMs have enhanced task automation in software engineering; however, current evaluations primarily focus on natural language tasks, overlooking code quality. Most benchmarks prioritize high-level reasoning over executable code and real-world performance, leaving gaps in understanding true capabilities and risks associated with these models in production. To address this issue, we propose MERA Code, a new addition to the MERA benchmark family, specifically focused on evaluating code for the latest code generation LLMs in Russian. This benchmark includes 11 evaluation tasks that span 8 programming languages. Our proposed evaluation methodology features a taxonomy that outlines the practical coding skills necessary for models to complete these tasks. The benchmark comprises an open-source codebase for users to conduct MERA assessments, a scoring system compatible with various programming environments, and a platform featuring a leaderboard and submission system. We evaluate open LLMs and frontier API models, analyzing their limitations in terms of practical coding tasks in non-English languages. We are publicly releasing MERA to guide future research, anticipate groundbreaking features in model development, and standardize evaluation procedures. 

---
# Site-Level Fine-Tuning with Progressive Layer Freezing: Towards Robust Prediction of Bronchopulmonary Dysplasia from Day-1 Chest Radiographs in Extremely Preterm Infants 

**Authors**: Sybelle Goedicke-Fritz, Michelle Bous, Annika Engel, Matthias Flotho, Pascal Hirsch, Hannah Wittig, Dino Milanovic, Dominik Mohr, Mathias Kaspar, Sogand Nemat, Dorothea Kerner, Arno Bücker, Andreas Keller, Sascha Meyer, Michael Zemlin, Philipp Flotho  

**Link**: [PDF](https://arxiv.org/pdf/2507.12269)  

**Abstract**: Bronchopulmonary dysplasia (BPD) is a chronic lung disease affecting 35% of extremely low birth weight infants. Defined by oxygen dependence at 36 weeks postmenstrual age, it causes lifelong respiratory complications. However, preventive interventions carry severe risks, including neurodevelopmental impairment, ventilator-induced lung injury, and systemic complications. Therefore, early BPD prognosis and prediction of BPD outcome is crucial to avoid unnecessary toxicity in low risk infants. Admission radiographs of extremely preterm infants are routinely acquired within 24h of life and could serve as a non-invasive prognostic tool. In this work, we developed and investigated a deep learning approach using chest X-rays from 163 extremely low-birth-weight infants ($\leq$32 weeks gestation, 401-999g) obtained within 24 hours of birth. We fine-tuned a ResNet-50 pretrained specifically on adult chest radiographs, employing progressive layer freezing with discriminative learning rates to prevent overfitting and evaluated a CutMix augmentation and linear probing. For moderate/severe BPD outcome prediction, our best performing model with progressive freezing, linear probing and CutMix achieved an AUROC of 0.78 $\pm$ 0.10, balanced accuracy of 0.69 $\pm$ 0.10, and an F1-score of 0.67 $\pm$ 0.11. In-domain pre-training significantly outperformed ImageNet initialization (p = 0.031) which confirms domain-specific pretraining to be important for BPD outcome prediction. Routine IRDS grades showed limited prognostic value (AUROC 0.57 $\pm$ 0.11), confirming the need of learned markers. Our approach demonstrates that domain-specific pretraining enables accurate BPD prediction from routine day-1 radiographs. Through progressive freezing and linear probing, the method remains computationally feasible for site-level implementation and future federated learning deployments. 

---
# A Framework for Nonstationary Gaussian Processes with Neural Network Parameters 

**Authors**: Zachary James, Joseph Guinness  

**Link**: [PDF](https://arxiv.org/pdf/2507.12262)  

**Abstract**: Gaussian processes have become a popular tool for nonparametric regression because of their flexibility and uncertainty quantification. However, they often use stationary kernels, which limit the expressiveness of the model and may be unsuitable for many datasets. We propose a framework that uses nonstationary kernels whose parameters vary across the feature space, modeling these parameters as the output of a neural network that takes the features as input. The neural network and Gaussian process are trained jointly using the chain rule to calculate derivatives. Our method clearly describes the behavior of the nonstationary parameters and is compatible with approximation methods for scaling to large datasets. It is flexible and easily adapts to different nonstationary kernels without needing to redesign the optimization procedure. Our methods are implemented with the GPyTorch library and can be readily modified. We test a nonstationary variance and noise variant of our method on several machine learning datasets and find that it achieves better accuracy and log-score than both a stationary model and a hierarchical model approximated with variational inference. Similar results are observed for a model with only nonstationary variance. We also demonstrate our approach's ability to recover the nonstationary parameters of a spatial dataset. 

---
# Infherno: End-to-end Agent-based FHIR Resource Synthesis from Free-form Clinical Notes 

**Authors**: Johann Frei, Nils Feldhus, Lisa Raithel, Roland Roller, Alexander Meyer, Frank Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2507.12261)  

**Abstract**: For clinical data integration and healthcare services, the HL7 FHIR standard has established itself as a desirable format for interoperability between complex health data. Previous attempts at automating the translation from free-form clinical notes into structured FHIR resources rely on modular, rule-based systems or LLMs with instruction tuning and constrained decoding. Since they frequently suffer from limited generalizability and structural inconformity, we propose an end-to-end framework powered by LLM agents, code execution, and healthcare terminology database tools to address these issues. Our solution, called Infherno, is designed to adhere to the FHIR document schema and competes well with a human baseline in predicting FHIR resources from unstructured text. The implementation features a front end for custom and synthetic data and both local and proprietary models, supporting clinical data integration processes and interoperability across institutions. 

---
# Improving Contextual ASR via Multi-grained Fusion with Large Language Models 

**Authors**: Shilin Zhou, Zhenghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12252)  

**Abstract**: While end-to-end Automatic Speech Recognition (ASR) models have shown impressive performance in transcribing general speech, they often struggle to accurately recognize contextually relevant keywords, such as proper nouns or user-specific entities.
Previous approaches have explored leveraging keyword dictionaries in the textual modality to improve keyword recognition, either through token-level fusion that guides token-by-token generation or phrase-level fusion that enables direct copying of keyword phrases.
However, these methods operate at different granularities and have their own limitations.
In this paper, we propose a novel multi-grained fusion approach that jointly leverages the strengths of both token-level and phrase-level fusion with Large Language Models (LLMs).
Our approach incorporates a late-fusion strategy that elegantly combines ASR's acoustic information with LLM's rich contextual knowledge, balancing fine-grained token precision with holistic phrase-level understanding.
Experiments on Chinese and English datasets demonstrate that our approach achieves state-of-the-art performance on keyword-related metrics while preserving high accuracy on non-keyword text.
Ablation studies further confirm that the token-level and phrase-level components both contribute significantly to the performance gains, complementing each other in our joint multi-grained framework.
The code and models will be publicly available at this https URL. 

---
# Looking for Fairness in Recommender Systems 

**Authors**: Cécile Logé  

**Link**: [PDF](https://arxiv.org/pdf/2507.12242)  

**Abstract**: Recommender systems can be found everywhere today, shaping our everyday experience whenever we're consuming content, ordering food, buying groceries online, or even just reading the news. Let's imagine we're in the process of building a recommender system to make content suggestions to users on social media. When thinking about fairness, it becomes clear there are several perspectives to consider: the users asking for tailored suggestions, the content creators hoping for some limelight, and society at large, navigating the repercussions of algorithmic recommendations. A shared fairness concern across all three is the emergence of filter bubbles, a side-effect that takes place when recommender systems are almost "too good", making recommendations so tailored that users become inadvertently confined to a narrow set of opinions/themes and isolated from alternative ideas. From the user's perspective, this is akin to manipulation. From the small content creator's perspective, this is an obstacle preventing them access to a whole range of potential fans. From society's perspective, the potential consequences are far-reaching, influencing collective opinions, social behavior and political decisions. How can our recommender system be fine-tuned to avoid the creation of filter bubbles, and ensure a more inclusive and diverse content landscape? Approaching this problem involves defining one (or more) performance metric to represent diversity, and tweaking our recommender system's performance through the lens of fairness. By incorporating this metric into our evaluation framework, we aim to strike a balance between personalized recommendations and the broader societal goal of fostering rich and varied cultures and points of view. 

---
# Draw an Ugly Person An Exploration of Generative AIs Perceptions of Ugliness 

**Authors**: Garyoung Kim, Huisung Kwon, Seoju Yun, Yu-Won Youn  

**Link**: [PDF](https://arxiv.org/pdf/2507.12212)  

**Abstract**: Generative AI does not only replicate human creativity but also reproduces deep-seated cultural biases, making it crucial to critically examine how concepts like ugliness are understood and expressed by these tools. This study investigates how four different generative AI models understand and express ugliness through text and image and explores the biases embedded within these representations. We extracted 13 adjectives associated with ugliness through iterative prompting of a large language model and generated 624 images across four AI models and three prompts. Demographic and socioeconomic attributes within the images were independently coded and thematically analyzed. Our findings show that AI models disproportionately associate ugliness with old white male figures, reflecting entrenched social biases as well as paradoxical biases, where efforts to avoid stereotypical depictions of marginalized groups inadvertently result in the disproportionate projection of negative attributes onto majority groups. Qualitative analysis further reveals that, despite supposed attempts to frame ugliness within social contexts, conventional physical markers such as asymmetry and aging persist as central visual motifs. These findings demonstrate that despite attempts to create more equal representations, generative AI continues to perpetuate inherited and paradoxical biases, underscoring the critical work being done to create ethical AI training paradigms and advance methodologies for more inclusive AI development. 

---
# Sparse Autoencoders for Sequential Recommendation Models: Interpretation and Flexible Control 

**Authors**: Anton Klenitskiy, Konstantin Polev, Daria Denisova, Alexey Vasilev, Dmitry Simakov, Gleb Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2507.12202)  

**Abstract**: Many current state-of-the-art models for sequential recommendations are based on transformer architectures. Interpretation and explanation of such black box models is an important research question, as a better understanding of their internals can help understand, influence, and control their behavior, which is very important in a variety of real-world applications. Recently sparse autoencoders (SAE) have been shown to be a promising unsupervised approach for extracting interpretable features from language models. These autoencoders learn to reconstruct hidden states of the transformer's internal layers from sparse linear combinations of directions in their activation space.
This paper is focused on the application of SAE to the sequential recommendation domain. We show that this approach can be successfully applied to the transformer trained on a sequential recommendation task: learned directions turn out to be more interpretable and monosemantic than the original hidden state dimensions. Moreover, we demonstrate that the features learned by SAE can be used to effectively and flexibly control the model's behavior, providing end-users with a straightforward method to adjust their recommendations to different custom scenarios and contexts. 

---
# Quantize More, Lose Less: Autoregressive Generation from Residually Quantized Speech Representations 

**Authors**: Yichen Han, Xiaoyang Hao, Keming Chen, Weibo Xiong, Jun He, Ruonan Zhang, Junjie Cao, Yue Liu, Bowen Li, Dongrui Zhang, Hui Xia, Huilei Fu, Kai Jia, Kaixuan Guo, Mingli Jin, Qingyun Meng, Ruidong Ma, Ruiqian Fang, Shaotong Guo, Xuhui Li, Yang Xiang, Ying Zhang, Yulong Liu, Yunfeng Li, Yuyi Zhang, Yuze Zhou, Zhen Wang, Zhaowen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.12197)  

**Abstract**: Text-to-speech (TTS) synthesis has seen renewed progress under the discrete modeling paradigm. Existing autoregressive approaches often rely on single-codebook representations, which suffer from significant information loss. Even with post-hoc refinement techniques such as flow matching, these methods fail to recover fine-grained details (e.g., prosodic nuances, speaker-specific timbres), especially in challenging scenarios like singing voice or music synthesis. We propose QTTS, a novel TTS framework built upon our new audio codec, QDAC. The core innovation of QDAC lies in its end-to-end training of an ASR-based auto-regressive network with a GAN, which achieves superior semantic feature disentanglement for scalable, near-lossless compression. QTTS models these discrete codes using two innovative strategies: the Hierarchical Parallel architecture, which uses a dual-AR structure to model inter-codebook dependencies for higher-quality synthesis, and the Delay Multihead approach, which employs parallelized prediction with a fixed delay to accelerate inference speed. Our experiments demonstrate that the proposed framework achieves higher synthesis quality and better preserves expressive content compared to baseline. This suggests that scaling up compression via multi-codebook modeling is a promising direction for high-fidelity, general-purpose speech and audio generation. 

---
# Selective Quantization Tuning for ONNX Models 

**Authors**: Nikolaos Louloudakis, Ajitha Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12196)  

**Abstract**: Quantization is a process that reduces the precision of deep neural network models to lower model size and computational demands, often at the cost of accuracy. However, fully quantized models may exhibit sub-optimal performance below acceptable levels and face deployment challenges on low-end hardware accelerators due to practical constraints. To address these issues, quantization can be selectively applied to only a subset of layers, but selecting which layers to exclude is non-trivial. To this direction, we propose TuneQn, a suite enabling selective quantization, deployment and execution of ONNX models across various CPU and GPU devices, combined with profiling and multi-objective optimization. TuneQn generates selectively quantized ONNX models, deploys them on different hardware, measures performance on metrics like accuracy and size, performs Pareto Front minimization to identify the best model candidate and visualizes the results. To demonstrate the effectiveness of TuneQn, we evaluated TuneQn on four ONNX models with two quantization settings across CPU and GPU devices. As a result, we demonstrated that our utility effectively performs selective quantization and tuning, selecting ONNX model candidates with up to a $54.14$% reduction in accuracy loss compared to the fully quantized model, and up to a $72.9$% model size reduction compared to the original model. 

---
# Revealing the Ancient Beauty: Digital Reconstruction of Temple Tiles using Computer Vision 

**Authors**: Arkaprabha Basu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12195)  

**Abstract**: Modern digitised approaches have dramatically changed the preservation and restoration of cultural treasures, integrating computer scientists into multidisciplinary projects with ease. Machine learning, deep learning, and computer vision techniques have revolutionised developing sectors like 3D reconstruction, picture inpainting,IoT-based methods, genetic algorithms, and image processing with the integration of computer scientists into multidisciplinary initiatives. We suggest three cutting-edge techniques in recognition of the special qualities of Indian monuments, which are famous for their architectural skill and aesthetic appeal. First is the Fractal Convolution methodology, a segmentation method based on image processing that successfully reveals subtle architectural patterns within these irreplaceable cultural buildings. The second is a revolutionary Self-Sensitive Tile Filling (SSTF) method created especially for West Bengal's mesmerising Bankura Terracotta Temples with a brand-new data augmentation method called MosaicSlice on the third. Furthermore, we delve deeper into the Super Resolution strategy to upscale the images without losing significant amount of quality. Our methods allow for the development of seamless region-filling and highly detailed tiles while maintaining authenticity using a novel data augmentation strategy within affordable costs introducing automation. By providing effective solutions that preserve the delicate balance between tradition and innovation, this study improves the subject and eventually ensures unrivalled efficiency and aesthetic excellence in cultural heritage protection. The suggested approaches advance the field into an era of unmatched efficiency and aesthetic quality while carefully upholding the delicate equilibrium between tradition and innovation. 

---
# BenchRL-QAS: Benchmarking reinforcement learning algorithms for quantum architecture search 

**Authors**: Azhar Ikhtiarudin, Aditi Das, Param Thakkar, Akash Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12189)  

**Abstract**: We introduce BenchRL-QAS, a unified benchmarking framework for systematically evaluating reinforcement learning (RL) algorithms in quantum architecture search (QAS) across diverse variational quantum algorithm tasks and system sizes ranging from 2- to 8-qubit. Our study benchmarks nine RL agents including both value-based and policy-gradient methods on representative quantum problems such as variational quantum eigensolver, variational quantum state diagonalization, quantum classification, and state preparation, spanning both noiseless and realistic noisy regimes. We propose a weighted ranking metric that balances accuracy, circuit depth, gate count, and computational efficiency, enabling fair and comprehensive comparison. Our results first reveal that RL-based quantum classifier outperforms baseline variational classifiers. Then we conclude that no single RL algorithm is universally optimal when considering a set of QAS tasks; algorithmic performance is highly context-dependent, varying with task structure, qubit count, and noise. This empirical finding provides strong evidence for the "no free lunch" principle in RL-based quantum circuit design and highlights the necessity of tailored algorithm selection and systematic benchmarking for advancing quantum circuit synthesis. This work represents the most comprehensive RL-QAS benchmarking effort to date, and BenchRL-QAS along with all experimental data are made publicly available to support reproducibility and future research this https URL. 

---
# Wavelet-based Decoupling Framework for low-light Stereo Image Enhancement 

**Authors**: Shuangli Du, Siming Yan, Zhenghao Shi, Zhenzhen You, Lu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.12188)  

**Abstract**: Low-light images suffer from complex degradation, and existing enhancement methods often encode all degradation factors within a single latent space. This leads to highly entangled features and strong black-box characteristics, making the model prone to shortcut learning. To mitigate the above issues, this paper proposes a wavelet-based low-light stereo image enhancement method with feature space decoupling. Our insight comes from the following findings: (1) Wavelet transform enables the independent processing of low-frequency and high-frequency information. (2) Illumination adjustment can be achieved by adjusting the low-frequency component of a low-light image, extracted through multi-level wavelet decomposition. Thus, by using wavelet transform the feature space is decomposed into a low-frequency branch for illumination adjustment and multiple high-frequency branches for texture enhancement. Additionally, stereo low-light image enhancement can extract useful cues from another view to improve enhancement. To this end, we propose a novel high-frequency guided cross-view interaction module (HF-CIM) that operates within high-frequency branches rather than across the entire feature space, effectively extracting valuable image details from the other view. Furthermore, to enhance the high-frequency information, a detail and texture enhancement module (DTEM) is proposed based on cross-attention mechanism. The model is trained on a dataset consisting of images with uniform illumination and images with non-uniform illumination. Experimental results on both real and synthetic images indicate that our algorithm offers significant advantages in light adjustment while effectively recovering high-frequency information. The code and dataset are publicly available at: this https URL. 

---
# PRISM: Distributed Inference for Foundation Models at Edge 

**Authors**: Muhammad Azlan Qazi, Alexandros Iosifidis, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12145)  

**Abstract**: Foundation models (FMs) have achieved remarkable success across a wide range of applications, from image classification to natural langurage processing, but pose significant challenges for deployment at edge. This has sparked growing interest in developing practical and efficient strategies for bringing foundation models to edge environments. In this work, we propose PRISM, a communication-efficient and compute-aware strategy for distributed Transformer inference on edge devices. Our method leverages a Segment Means representation to approximate intermediate output features, drastically reducing inter-device communication. Additionally, we restructure the self-attention mechanism to eliminate redundant computations caused by per-device Key/Value calculation in position-wise partitioning and design a partition-aware causal masking scheme tailored for autoregressive models. We evaluate PRISM on ViT, BERT, and GPT-2 across diverse datasets, namely CIFAR-10, CIFAR-100, ImageNet-1k, GLUE, and CBT. Our results demonstrate substantial reductions in communication overhead (up to 99.2% for BERT at compression rate CR = 128) and per-device computation (51.24% for BERT at the same setting), with only minor accuracy degradation. This method offers a scalable and practical solution for deploying foundation models in distributed resource-constrained environments. 

---
# Quantum Machine Learning in Multi-Qubit Phase-Space Part I: Foundations 

**Authors**: Timothy Heightman, Edward Jiang, Ruth Mora-Soto, Maciej Lewenstein, Marcin Płodzień  

**Link**: [PDF](https://arxiv.org/pdf/2507.12117)  

**Abstract**: Quantum machine learning (QML) seeks to exploit the intrinsic properties of quantum mechanical systems, including superposition, coherence, and quantum entanglement for classical data processing. However, due to the exponential growth of the Hilbert space, QML faces practical limits in classical simulations with the state-vector representation of quantum system. On the other hand, phase-space methods offer an alternative by encoding quantum states as quasi-probability functions. Building on prior work in qubit phase-space and the Stratonovich-Weyl (SW) correspondence, we construct a closed, composable dynamical formalism for one- and many-qubit systems in phase-space. This formalism replaces the operator algebra of the Pauli group with function dynamics on symplectic manifolds, and recasts the curse of dimensionality in terms of harmonic support on a domain that scales linearly with the number of qubits. It opens a new route for QML based on variational modelling over phase-space. 

---
# Multimodal Coordinated Online Behavior: Trade-offs and Strategies 

**Authors**: Lorenzo Mannocci, Stefano Cresci, Matteo Magnani, Anna Monreale, Maurizio Tesconi  

**Link**: [PDF](https://arxiv.org/pdf/2507.12108)  

**Abstract**: Coordinated online behavior, which spans from beneficial collective actions to harmful manipulation such as disinformation campaigns, has become a key focus in digital ecosystem analysis. Traditional methods often rely on monomodal approaches, focusing on single types of interactions like co-retweets or co-hashtags, or consider multiple modalities independently of each other. However, these approaches may overlook the complex dynamics inherent in multimodal coordination. This study compares different ways of operationalizing the detection of multimodal coordinated behavior. It examines the trade-off between weakly and strongly integrated multimodal models, highlighting the balance between capturing broader coordination patterns and identifying tightly coordinated behavior. By comparing monomodal and multimodal approaches, we assess the unique contributions of different data modalities and explore how varying implementations of multimodality impact detection outcomes. Our findings reveal that not all the modalities provide distinct insights, but that with a multimodal approach we can get a more comprehensive understanding of coordination dynamics. This work enhances the ability to detect and analyze coordinated online behavior, offering new perspectives for safeguarding the integrity of digital platforms. 

---
# Non-Adaptive Adversarial Face Generation 

**Authors**: Sunpill Kim, Seunghun Paik, Chanwoo Hwang, Minsu Kim, Jae Hong Seo  

**Link**: [PDF](https://arxiv.org/pdf/2507.12107)  

**Abstract**: Adversarial attacks on face recognition systems (FRSs) pose serious security and privacy threats, especially when these systems are used for identity verification. In this paper, we propose a novel method for generating adversarial faces-synthetic facial images that are visually distinct yet recognized as a target identity by the FRS. Unlike iterative optimization-based approaches (e.g., gradient descent or other iterative solvers), our method leverages the structural characteristics of the FRS feature space. We figure out that individuals sharing the same attribute (e.g., gender or race) form an attributed subsphere. By utilizing such subspheres, our method achieves both non-adaptiveness and a remarkably small number of queries. This eliminates the need for relying on transferability and open-source surrogate models, which have been a typical strategy when repeated adaptive queries to commercial FRSs are impossible. Despite requiring only a single non-adaptive query consisting of 100 face images, our method achieves a high success rate of over 93% against AWS's CompareFaces API at its default threshold. Furthermore, unlike many existing attacks that perturb a given image, our method can deliberately produce adversarial faces that impersonate the target identity while exhibiting high-level attributes chosen by the adversary. 

---
# From Static to Intelligent: Evolving SaaS Pricing with LLMs 

**Authors**: Francisco Javier Cavero, Juan C. Alonso, Antonio Ruiz-Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2507.12104)  

**Abstract**: The SaaS paradigm has revolutionized software distribution by offering flexible pricing options to meet diverse customer needs. However, the rapid expansion of the SaaS market has introduced significant complexity for DevOps teams, who must manually manage and evolve pricing structures, an approach that is both time-consuming and prone to errors. The absence of automated tools for pricing analysis restricts the ability to efficiently evaluate, optimize, and scale these models. This paper proposes leveraging intelligent pricing (iPricing), dynamic, machine-readable pricing models, as a solution to these challenges. Intelligent pricing enables competitive analysis, streamlines operational decision-making, and supports continuous pricing evolution in response to market dynamics, leading to improved efficiency and accuracy. We present an LLM-driven approach that automates the transformation of static HTML pricing into iPricing, significantly improving efficiency and consistency while minimizing human error. Our implementation, AI4Pricing2Yaml, features a basic Information Extractor that uses web scraping and LLMs technologies to extract essential pricing components, plans, features, usage limits, and add-ons, from SaaS websites. Validation against a dataset of 30 distinct commercial SaaS, encompassing over 150 intelligent pricings, demonstrates the system's effectiveness in extracting the desired elements across all steps. However, challenges remain in addressing hallucinations, complex structures, and dynamic content. This work highlights the potential of automating intelligent pricing transformation to streamline SaaS pricing management, offering implications for improved consistency and scalability in an increasingly intricate pricing landscape. Future research will focus on refining extraction capabilities and enhancing the system's adaptability to a wider range of SaaS websites. 

---
# BOOKCOREF: Coreference Resolution at Book Scale 

**Authors**: Giuliano Martinelli, Tommaso Bonomo, Pere-Lluís Huguet Cabot, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2507.12075)  

**Abstract**: Coreference Resolution systems are typically evaluated on benchmarks containing small- to medium-scale documents. When it comes to evaluating long texts, however, existing benchmarks, such as LitBank, remain limited in length and do not adequately assess system capabilities at the book scale, i.e., when co-referring mentions span hundreds of thousands of tokens. To fill this gap, we first put forward a novel automatic pipeline that produces high-quality Coreference Resolution annotations on full narrative texts. Then, we adopt this pipeline to create the first book-scale coreference benchmark, BOOKCOREF, with an average document length of more than 200,000 tokens. We carry out a series of experiments showing the robustness of our automatic procedure and demonstrating the value of our resource, which enables current long-document coreference systems to gain up to +20 CoNLL-F1 points when evaluated on full books. Moreover, we report on the new challenges introduced by this unprecedented book-scale setting, highlighting that current models fail to deliver the same performance they achieve on smaller documents. We release our data and code to encourage research and development of new book-scale Coreference Resolution systems at this https URL. 

---
# StylOch at PAN: Gradient-Boosted Trees with Frequency-Based Stylometric Features 

**Authors**: Jeremi K. Ochab, Mateusz Matias, Tymoteusz Boba, Tomasz Walkowiak  

**Link**: [PDF](https://arxiv.org/pdf/2507.12064)  

**Abstract**: This submission to the binary AI detection task is based on a modular stylometric pipeline, where: public spaCy models are used for text preprocessing (including tokenisation, named entity recognition, dependency parsing, part-of-speech tagging, and morphology annotation) and extracting several thousand features (frequencies of n-grams of the above linguistic annotations); light-gradient boosting machines are used as the classifier. We collect a large corpus of more than 500 000 machine-generated texts for the classifier's training. We explore several parameter options to increase the classifier's capacity and take advantage of that training set. Our approach follows the non-neural, computationally inexpensive but explainable approach found effective previously. 

---
# InstructFLIP: Exploring Unified Vision-Language Model for Face Anti-spoofing 

**Authors**: Kun-Hsiang Lin, Yu-Wen Tseng, Kang-Yang Huang, Jhih-Ciang Wu, Wen-Huang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.12060)  

**Abstract**: Face anti-spoofing (FAS) aims to construct a robust system that can withstand diverse attacks. While recent efforts have concentrated mainly on cross-domain generalization, two significant challenges persist: limited semantic understanding of attack types and training redundancy across domains. We address the first by integrating vision-language models (VLMs) to enhance the perception of visual input. For the second challenge, we employ a meta-domain strategy to learn a unified model that generalizes well across multiple domains. Our proposed InstructFLIP is a novel instruction-tuned framework that leverages VLMs to enhance generalization via textual guidance trained solely on a single domain. At its core, InstructFLIP explicitly decouples instructions into content and style components, where content-based instructions focus on the essential semantics of spoofing, and style-based instructions consider variations related to the environment and camera characteristics. Extensive experiments demonstrate the effectiveness of InstructFLIP by outperforming SOTA models in accuracy and substantially reducing training redundancy across diverse domains in FAS. Project website is available at this https URL. 

---
# Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery 

**Authors**: Xinhang Wan, Jiyuan Liu, Qian Qu, Suyuan Liu, Chuyu Zhang, Fangdi Wang, Xinwang Liu, En Zhu, Kunlun He  

**Link**: [PDF](https://arxiv.org/pdf/2507.12029)  

**Abstract**: In this paper, we address the problem of novel class discovery (NCD), which aims to cluster novel classes by leveraging knowledge from disjoint known classes. While recent advances have made significant progress in this area, existing NCD methods face two major limitations. First, they primarily focus on single-view data (e.g., images), overlooking the increasingly common multi-view data, such as multi-omics datasets used in disease diagnosis. Second, their reliance on pseudo-labels to supervise novel class clustering often results in unstable performance, as pseudo-label quality is highly sensitive to factors such as data noise and feature dimensionality. To address these challenges, we propose a novel framework named Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery (IICMVNCD), which is the first attempt to explore NCD in multi-view setting so far. Specifically, at the intra-view level, leveraging the distributional similarity between known and novel classes, we employ matrix factorization to decompose features into view-specific shared base matrices and factor matrices. The base matrices capture distributional consistency among the two datasets, while the factor matrices model pairwise relationships between samples. At the inter-view level, we utilize view relationships among known classes to guide the clustering of novel classes. This includes generating predicted labels through the weighted fusion of factor matrices and dynamically adjusting view weights of known classes based on the supervision loss, which are then transferred to novel class learning. Experimental results validate the effectiveness of our proposed approach. 

---
# SS-DC: Spatial-Spectral Decoupling and Coupling Across Visible-Infrared Gap for Domain Adaptive Object Detection 

**Authors**: Xiwei Zhang, Chunjin Yang, Yiming Xiao, Runtong Zhang, Fanman Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.12017)  

**Abstract**: Unsupervised domain adaptive object detection (UDAOD) from the visible domain to the infrared (RGB-IR) domain is challenging. Existing methods regard the RGB domain as a unified domain and neglect the multiple subdomains within it, such as daytime, nighttime, and foggy scenes. We argue that decoupling the domain-invariant (DI) and domain-specific (DS) features across these multiple subdomains is beneficial for RGB-IR domain adaptation. To this end, this paper proposes a new SS-DC framework based on a decoupling-coupling strategy. In terms of decoupling, we design a Spectral Adaptive Idempotent Decoupling (SAID) module in the aspect of spectral decomposition. Due to the style and content information being highly embedded in different frequency bands, this module can decouple DI and DS components more accurately and interpretably. A novel filter bank-based spectral processing paradigm and a self-distillation-driven decoupling loss are proposed to improve the spectral domain decoupling. In terms of coupling, a new spatial-spectral coupling method is proposed, which realizes joint coupling through spatial and spectral DI feature pyramids. Meanwhile, this paper introduces DS from decoupling to reduce the domain bias. Extensive experiments demonstrate that our method can significantly improve the baseline performance and outperform existing UDAOD methods on multiple RGB-IR datasets, including a new experimental protocol proposed in this paper based on the FLIR-ADAS dataset. 

---
# Identifying Signatures of Image Phenotypes to Track Treatment Response in Liver Disease 

**Authors**: Matthias Perkonigg, Nina Bastati, Ahmed Ba-Ssalamah, Peter Mesenbrink, Alexander Goehler, Miljen Martic, Xiaofei Zhou, Michael Trauner, Georg Langs  

**Link**: [PDF](https://arxiv.org/pdf/2507.12012)  

**Abstract**: Quantifiable image patterns associated with disease progression and treatment response are critical tools for guiding individual treatment, and for developing novel therapies. Here, we show that unsupervised machine learning can identify a pattern vocabulary of liver tissue in magnetic resonance images that quantifies treatment response in diffuse liver disease. Deep clustering networks simultaneously encode and cluster patches of medical images into a low-dimensional latent space to establish a tissue vocabulary. The resulting tissue types capture differential tissue change and its location in the liver associated with treatment response. We demonstrate the utility of the vocabulary on a randomized controlled trial cohort of non-alcoholic steatohepatitis patients. First, we use the vocabulary to compare longitudinal liver change in a placebo and a treatment cohort. Results show that the method identifies specific liver tissue change pathways associated with treatment, and enables a better separation between treatment groups than established non-imaging measures. Moreover, we show that the vocabulary can predict biopsy derived features from non-invasive imaging data. We validate the method on a separate replication cohort to demonstrate the applicability of the proposed method. 

---
# DUSE: A Data Expansion Framework for Low-resource Automatic Modulation Recognition based on Active Learning 

**Authors**: Yao Lu, Hongyu Gao, Zhuangzhi Chen, Dongwei Xu, Yun Lin, Qi Xuan, Guan Gui  

**Link**: [PDF](https://arxiv.org/pdf/2507.12011)  

**Abstract**: Although deep neural networks have made remarkable achievements in the field of automatic modulation recognition (AMR), these models often require a large amount of labeled data for training. However, in many practical scenarios, the available target domain data is scarce and difficult to meet the needs of model training. The most direct way is to collect data manually and perform expert annotation, but the high time and labor costs are unbearable. Another common method is data augmentation. Although it can enrich training samples to a certain extent, it does not introduce new data and therefore cannot fundamentally solve the problem of data scarcity. To address these challenges, we introduce a data expansion framework called Dynamic Uncertainty-driven Sample Expansion (DUSE). Specifically, DUSE uses an uncertainty scoring function to filter out useful samples from relevant AMR datasets and employs an active learning strategy to continuously refine the scorer. Extensive experiments demonstrate that DUSE consistently outperforms 8 coreset selection baselines in both class-balance and class-imbalance settings. Besides, DUSE exhibits strong cross-architecture generalization for unseen models. 

---
# Dual form Complementary Masking for Domain-Adaptive Image Segmentation 

**Authors**: Jiawen Wang, Yinda Chen, Xiaoyu Liu, Che Liu, Dong Liu, Jianqing Gao, Zhiwei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12008)  

**Abstract**: Recent works have correlated Masked Image Modeling (MIM) with consistency regularization in Unsupervised Domain Adaptation (UDA). However, they merely treat masking as a special form of deformation on the input images and neglect the theoretical analysis, which leads to a superficial understanding of masked reconstruction and insufficient exploitation of its potential in enhancing feature extraction and representation learning. In this paper, we reframe masked reconstruction as a sparse signal reconstruction problem and theoretically prove that the dual form of complementary masks possesses superior capabilities in extracting domain-agnostic image features. Based on this compelling insight, we propose MaskTwins, a simple yet effective UDA framework that integrates masked reconstruction directly into the main training pipeline. MaskTwins uncovers intrinsic structural patterns that persist across disparate domains by enforcing consistency between predictions of images masked in complementary ways, enabling domain generalization in an end-to-end manner. Extensive experiments verify the superiority of MaskTwins over baseline methods in natural and biological image segmentation. These results demonstrate the significant advantages of MaskTwins in extracting domain-invariant features without the need for separate pre-training, offering a new paradigm for domain-adaptive segmentation. 

---
# Frequency-Dynamic Attention Modulation for Dense Prediction 

**Authors**: Linwei Chen, Lin Gu, Ying Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12006)  

**Abstract**: Vision Transformers (ViTs) have significantly advanced computer vision, demonstrating strong performance across various tasks. However, the attention mechanism in ViTs makes each layer function as a low-pass filter, and the stacked-layer architecture in existing transformers suffers from frequency vanishing. This leads to the loss of critical details and textures. We propose a novel, circuit-theory-inspired strategy called Frequency-Dynamic Attention Modulation (FDAM), which can be easily plugged into ViTs. FDAM directly modulates the overall frequency response of ViTs and consists of two techniques: Attention Inversion (AttInv) and Frequency Dynamic Scaling (FreqScale). Since circuit theory uses low-pass filters as fundamental elements, we introduce AttInv, a method that generates complementary high-pass filtering by inverting the low-pass filter in the attention matrix, and dynamically combining the two. We further design FreqScale to weight different frequency components for fine-grained adjustments to the target response function. Through feature similarity analysis and effective rank evaluation, we demonstrate that our approach avoids representation collapse, leading to consistent performance improvements across various models, including SegFormer, DeiT, and MaskDINO. These improvements are evident in tasks such as semantic segmentation, object detection, and instance segmentation. Additionally, we apply our method to remote sensing detection, achieving state-of-the-art results in single-scale settings. The code is available at \href{this https URL}{this https URL}. 

---
# Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection 

**Authors**: Tairan Huang, Yili Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11997)  

**Abstract**: Graph fraud detection has garnered significant attention as Graph Neural Networks (GNNs) have proven effective in modeling complex relationships within multimodal data. However, existing graph fraud detection methods typically use preprocessed node embeddings and predefined graph structures to reveal fraudsters, which ignore the rich semantic cues contained in raw textual information. Although Large Language Models (LLMs) exhibit powerful capabilities in processing textual information, it remains a significant challenge to perform multimodal fusion of processed textual embeddings with graph structures. In this paper, we propose a \textbf{M}ulti-level \textbf{L}LM \textbf{E}nhanced Graph Fraud \textbf{D}etection framework called MLED. In MLED, we utilize LLMs to extract external knowledge from textual information to enhance graph fraud detection methods. To integrate LLMs with graph structure information and enhance the ability to distinguish fraudsters, we design a multi-level LLM enhanced framework including type-level enhancer and relation-level enhancer. One is to enhance the difference between the fraudsters and the benign entities, the other is to enhance the importance of the fraudsters in different relations. The experiments on four real-world datasets show that MLED achieves state-of-the-art performance in graph fraud detection as a generalized framework that can be applied to existing methods. 

---
# Robust Planning for Autonomous Vehicles with Diffusion-Based Failure Samplers 

**Authors**: Juanran Wang, Marc R. Schlichting, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2507.11991)  

**Abstract**: High-risk traffic zones such as intersections are a major cause of collisions. This study leverages deep generative models to enhance the safety of autonomous vehicles in an intersection context. We train a 1000-step denoising diffusion probabilistic model to generate collision-causing sensor noise sequences for an autonomous vehicle navigating a four-way intersection based on the current relative position and velocity of an intruder. Using the generative adversarial architecture, the 1000-step model is distilled into a single-step denoising diffusion model which demonstrates fast inference speed while maintaining similar sampling quality. We demonstrate one possible application of the single-step model in building a robust planner for the autonomous vehicle. The planner uses the single-step model to efficiently sample potential failure cases based on the currently measured traffic state to inform its decision-making. Through simulation experiments, the robust planner demonstrates significantly lower failure rate and delay rate compared with the baseline Intelligent Driver Model controller. 

---
# Formal Verification of Neural Certificates Done Dynamically 

**Authors**: Thomas A. Henzinger, Konstantin Kueffner, Emily Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11987)  

**Abstract**: Neural certificates have emerged as a powerful tool in cyber-physical systems control, providing witnesses of correctness. These certificates, such as barrier functions, often learned alongside control policies, once verified, serve as mathematical proofs of system safety. However, traditional formal verification of their defining conditions typically faces scalability challenges due to exhaustive state-space exploration. To address this challenge, we propose a lightweight runtime monitoring framework that integrates real-time verification and does not require access to the underlying control policy. Our monitor observes the system during deployment and performs on-the-fly verification of the certificate over a lookahead region to ensure safety within a finite prediction horizon. We instantiate this framework for ReLU-based control barrier functions and demonstrate its practical effectiveness in a case study. Our approach enables timely detection of safety violations and incorrect certificates with minimal overhead, providing an effective but lightweight alternative to the static verification of the certificates. 

---
# Online Training and Pruning of Deep Reinforcement Learning Networks 

**Authors**: Valentin Frank Ingmar Guenter, Athanasios Sideris  

**Link**: [PDF](https://arxiv.org/pdf/2507.11975)  

**Abstract**: Scaling deep neural networks (NN) of reinforcement learning (RL) algorithms has been shown to enhance performance when feature extraction networks are used but the gained performance comes at the significant expense of increased computational and memory complexity. Neural network pruning methods have successfully addressed this challenge in supervised learning. However, their application to RL is underexplored. We propose an approach to integrate simultaneous training and pruning within advanced RL methods, in particular to RL algorithms enhanced by the Online Feature Extractor Network (OFENet). Our networks (XiNet) are trained to solve stochastic optimization problems over the RL networks' weights and the parameters of variational Bernoulli distributions for 0/1 Random Variables $\xi$ scaling each unit in the networks. The stochastic problem formulation induces regularization terms that promote convergence of the variational parameters to 0 when a unit contributes little to the performance. In this case, the corresponding structure is rendered permanently inactive and pruned from its network. We propose a cost-aware, sparsity-promoting regularization scheme, tailored to the DenseNet architecture of OFENets expressing the parameter complexity of involved networks in terms of the parameters of the RVs in these networks. Then, when matching this cost with the regularization terms, the many hyperparameters associated with them are automatically selected, effectively combining the RL objectives and network compression. We evaluate our method on continuous control benchmarks (MuJoCo) and the Soft Actor-Critic RL agent, demonstrating that OFENets can be pruned considerably with minimal loss in performance. Furthermore, our results confirm that pruning large networks during training produces more efficient and higher performing RL agents rather than training smaller networks from scratch. 

---
# Toxicity-Aware Few-Shot Prompting for Low-Resource Singlish Translation 

**Authors**: Ziyu Ge, Gabriel Chua, Leanne Tan, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.11966)  

**Abstract**: As online communication increasingly incorporates under-represented languages and colloquial dialects, standard translation systems often fail to preserve local slang, code-mixing, and culturally embedded markers of harmful speech. Translating toxic content between low-resource language pairs poses additional challenges due to scarce parallel data and safety filters that sanitize offensive expressions. In this work, we propose a reproducible, two-stage framework for toxicity-preserving translation, demonstrated on a code-mixed Singlish safety corpus. First, we perform human-verified few-shot prompt engineering: we iteratively curate and rank annotator-selected Singlish-target examples to capture nuanced slang, tone, and toxicity. Second, we optimize model-prompt pairs by benchmarking several large language models using semantic similarity via direct and back-translation. Quantitative human evaluation confirms the effectiveness and efficiency of our pipeline. Beyond improving translation quality, our framework contributes to the safety of multicultural LLMs by supporting culturally sensitive moderation and benchmarking in low-resource contexts. By positioning Singlish as a testbed for inclusive NLP, we underscore the importance of preserving sociolinguistic nuance in real-world applications such as content moderation and regional platform governance. 

---
# PoTPTQ: A Two-step Power-of-Two Post-training for LLMs 

**Authors**: Xinyu Wang, Vahid Partovi Nia, Peng Lu, Jerry Huang, Xiao-Wen Chang, Boxing Chen, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.11959)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing (NLP) tasks. However, their deployment is challenging due to the substantial computational resources required. Power-of-two (PoT) quantization is a general tool to counteract this difficulty. Albeit previous works on PoT quantization can be efficiently dequantized on CPUs using fixed-point addition, it showed less effectiveness on GPUs. The reason is entanglement of the sign bit and sequential bit manipulations needed for dequantization. We propose a novel POT quantization framework for LLM weights that (i) outperforms state-of-the-art accuracy in extremely low-precision number formats, and (ii) enables faster inference through more efficient dequantization. To maintain the accuracy of the quantized model, we introduce a two-step post-training algorithm: (i) initialize the quantization scales with a robust starting point, and (ii) refine these scales using a minimal calibration set. The performance of our PoT post-training algorithm surpasses the current state-of-the-art in integer quantization, particularly at low precisions such as 2- and 3-bit formats. Our PoT quantization accelerates the dequantization step required for the floating point inference and leads to $3.67\times$ speed up on a NVIDIA V100, and $1.63\times$ on a NVIDIA RTX 4090, compared to uniform integer dequantization. 

---
# Kevin: Multi-Turn RL for Generating CUDA Kernels 

**Authors**: Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, Silas Alberti  

**Link**: [PDF](https://arxiv.org/pdf/2507.11948)  

**Abstract**: Writing GPU kernels is a challenging task and critical for AI systems' efficiency. It is also highly iterative: domain experts write code and improve performance through execution feedback. Moreover, it presents verifiable rewards like correctness and speedup, making it a natural environment to apply Reinforcement Learning (RL). To explicitly incorporate the iterative nature of this process into training, we develop a flexible multi-turn RL recipe that addresses unique challenges encountered in real-world settings, such as learning from long trajectories and effective reward attribution across turns. We present Kevin - K(ernel D)evin, the first model trained with multi-turn RL for CUDA kernel generation and optimization. In our evaluation setup, Kevin shows significant gains over its base model (QwQ-32B), improving correctness of generated kernels (in pure CUDA) from 56% to 82% and mean speedup from 0.53x to 1.10x of baseline (PyTorch Eager), and surpassing frontier models like o4-mini (0.78x). Finally, we study its behavior across test-time scaling axes: we found scaling serial refinement more beneficial than parallel sampling. In particular, when given more refinement turns, Kevin shows a higher rate of improvement. 

---
# RaDL: Relation-aware Disentangled Learning for Multi-Instance Text-to-Image Generation 

**Authors**: Geon Park, Seon Bin Kim, Gunho Jung, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.11947)  

**Abstract**: With recent advancements in text-to-image (T2I) models, effectively generating multiple instances within a single image prompt has become a crucial challenge. Existing methods, while successful in generating positions of individual instances, often struggle to account for relationship discrepancy and multiple attributes leakage. To address these limitations, this paper proposes the relation-aware disentangled learning (RaDL) framework. RaDL enhances instance-specific attributes through learnable parameters and generates relation-aware image features via Relation Attention, utilizing action verbs extracted from the global prompt. Through extensive evaluations on benchmarks such as COCO-Position, COCO-MIG, and DrawBench, we demonstrate that RaDL outperforms existing methods, showing significant improvements in positional accuracy, multiple attributes consideration, and the relationships between instances. Our results present RaDL as the solution for generating images that consider both the relationships and multiple attributes of each instance within the multi-instance image. 

---
# Effective Fine-Tuning of Vision Transformers with Low-Rank Adaptation for Privacy-Preserving Image Classification 

**Authors**: Haiwei Lin, Shoko Imaizumi, Hitoshi Kiya  

**Link**: [PDF](https://arxiv.org/pdf/2507.11943)  

**Abstract**: We propose a low-rank adaptation method for training privacy-preserving vision transformer (ViT) models that efficiently freezes pre-trained ViT model weights. In the proposed method, trainable rank decomposition matrices are injected into each layer of the ViT architecture, and moreover, the patch embedding layer is not frozen, unlike in the case of the conventional low-rank adaptation methods. The proposed method allows us not only to reduce the number of trainable parameters but to also maintain almost the same accuracy as that of full-time tuning. 

---
# POLYCHARTQA: Benchmarking Large Vision-Language Models with Multilingual Chart Question Answering 

**Authors**: Yichen Xu, Liangyu Chen, Liang Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.11939)  

**Abstract**: Charts are a universally adopted medium for interpreting and communicating data. However, existing chart understanding benchmarks are predominantly English-centric, limiting their accessibility and applicability to global audiences. In this paper, we present PolyChartQA, the first large-scale multilingual chart question answering benchmark covering 22,606 charts and 26,151 question-answering pairs across 10 diverse languages. PolyChartQA is built using a decoupled pipeline that separates chart data from rendering code, allowing multilingual charts to be flexibly generated by simply translating the data and reusing the code. We leverage state-of-the-art LLM-based translation and enforce rigorous quality control in the pipeline to ensure the linguistic and semantic consistency of the generated multilingual charts. PolyChartQA facilitates systematic evaluation of multilingual chart understanding. Experiments on both open- and closed-source large vision-language models reveal a significant performance gap between English and other languages, especially low-resource ones with non-Latin scripts. This benchmark lays a foundation for advancing globally inclusive vision-language models. 

---
# A Survey of Deep Learning for Geometry Problem Solving 

**Authors**: Jianzhe Ma, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.11936)  

**Abstract**: Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: this https URL. 

---
# Native-AI Empowered Scalable Architectures and Solutions for Future Non-Terrestrial Networks: An Overview 

**Authors**: Jikang Deng, Fizza Hassan, Hui Zhou, Saad Al-Ahmadi, Mohamed-Slim Alouini, Daniel B. Da Costa  

**Link**: [PDF](https://arxiv.org/pdf/2507.11935)  

**Abstract**: As the path toward 6G networks is being charted, the emerging applications have motivated evolutions of network architectures to realize the efficient, reliable, and flexible wireless networks. Among the potential architectures, the non-terrestrial network (NTN) and open radio access network (ORAN) have received increasing interest from both academia and industry. Although the deployment of NTNs ensures coverage, enhances spectral efficiency, and improves the resilience of wireless networks. The high altitude and mobility of NTN present new challenges in the development and operations (DevOps) lifecycle, hindering intelligent and scalable network management due to the lack of native artificial intelligence (AI) capability. With the advantages of ORAN in disaggregation, openness, virtualization, and intelligence, several works propose integrating ORAN principles into the NTN, focusing mainly on ORAN deployment options based on transparent and regenerative systems. However, a holistic view of how to effectively combine ORAN and NTN throughout the DevOps lifecycle is still missing, especially regarding how intelligent ORAN addresses the scalability challenges in NTN. Motivated by this, in this paper, we first provide the background knowledge about ORAN and NTN, outline the state-of-the-art research on ORAN for NTNs, and present the DevOps challenges that motivate the adoption of ORAN solutions. We then propose the ORAN-based NTN framework, discussing its features and architectures in detail. These include the discussion about flexible fronthaul split, RAN intelligent controllers (RICs) enhancement for distributed learning, scalable deployment architecture, and multi-domain service management. Finally, the future research directions, including combinations of the ORAN-based NTN framework and other enabling technologies and schemes, as well as the candidate use cases, are highlighted. 

---
# Spatial Frequency Modulation for Semantic Segmentation 

**Authors**: Linwei Chen, Ying Fu, Lin Gu, Dezhi Zheng, Jifeng Dai  

**Link**: [PDF](https://arxiv.org/pdf/2507.11893)  

**Abstract**: High spatial frequency information, including fine details like textures, significantly contributes to the accuracy of semantic segmentation. However, according to the Nyquist-Shannon Sampling Theorem, high-frequency components are vulnerable to aliasing or distortion when propagating through downsampling layers such as strided-convolution. Here, we propose a novel Spatial Frequency Modulation (SFM) that modulates high-frequency features to a lower frequency before downsampling and then demodulates them back during upsampling. Specifically, we implement modulation through adaptive resampling (ARS) and design a lightweight add-on that can densely sample the high-frequency areas to scale up the signal, thereby lowering its frequency in accordance with the Frequency Scaling Property. We also propose Multi-Scale Adaptive Upsampling (MSAU) to demodulate the modulated feature and recover high-frequency information through non-uniform upsampling This module further improves segmentation by explicitly exploiting information interaction between densely and sparsely resampled areas at multiple scales. Both modules can seamlessly integrate with various architectures, extending from convolutional neural networks to transformers. Feature visualization and analysis confirm that our method effectively alleviates aliasing while successfully retaining details after demodulation. Finally, we validate the broad applicability and effectiveness of SFM by extending it to image classification, adversarial robustness, instance segmentation, and panoptic segmentation tasks. The code is available at \href{this https URL}{this https URL}. 

---
# From Coarse to Nuanced: Cross-Modal Alignment of Fine-Grained Linguistic Cues and Visual Salient Regions for Dynamic Emotion Recognition 

**Authors**: Yu Liu, Leyuan Qu, Hanlei Shi, Di Gao, Yuhua Zheng, Taihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.11892)  

**Abstract**: Dynamic Facial Expression Recognition (DFER) aims to identify human emotions from temporally evolving facial movements and plays a critical role in affective computing. While recent vision-language approaches have introduced semantic textual descriptions to guide expression recognition, existing methods still face two key limitations: they often underutilize the subtle emotional cues embedded in generated text, and they have yet to incorporate sufficiently effective mechanisms for filtering out facial dynamics that are irrelevant to emotional expression. To address these gaps, We propose GRACE, Granular Representation Alignment for Cross-modal Emotion recognition that integrates dynamic motion modeling, semantic text refinement, and token-level cross-modal alignment to facilitate the precise localization of emotionally salient spatiotemporal features. Our method constructs emotion-aware textual descriptions via a Coarse-to-fine Affective Text Enhancement (CATE) module and highlights expression-relevant facial motion through a motion-difference weighting mechanism. These refined semantic and visual signals are aligned at the token level using entropy-regularized optimal transport. Experiments on three benchmark datasets demonstrate that our method significantly improves recognition performance, particularly in challenging settings with ambiguous or imbalanced emotion classes, establishing new state-of-the-art (SOTA) results in terms of both UAR and WAR. 

---
# Interactive Hybrid Rice Breeding with Parametric Dual Projection 

**Authors**: Changjian Chen, Pengcheng Wang, Fei Lyu, Zhuo Tang, Li Yang, Long Wang, Yong Cai, Feng Yu, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.11848)  

**Abstract**: Hybrid rice breeding crossbreeds different rice lines and cultivates the resulting hybrids in fields to select those with desirable agronomic traits, such as higher yields. Recently, genomic selection has emerged as an efficient way for hybrid rice breeding. It predicts the traits of hybrids based on their genes, which helps exclude many undesired hybrids, largely reducing the workload of field cultivation. However, due to the limited accuracy of genomic prediction models, breeders still need to combine their experience with the models to identify regulatory genes that control traits and select hybrids, which remains a time-consuming process. To ease this process, in this paper, we proposed a visual analysis method to facilitate interactive hybrid rice breeding. Regulatory gene identification and hybrid selection naturally ensemble a dual-analysis task. Therefore, we developed a parametric dual projection method with theoretical guarantees to facilitate interactive dual analysis. Based on this dual projection method, we further developed a gene visualization and a hybrid visualization to verify the identified regulatory genes and hybrids. The effectiveness of our method is demonstrated through the quantitative evaluation of the parametric dual projection method, identified regulatory genes and desired hybrids in the case study, and positive feedback from breeders. 

---
# MNIST-Gen: A Modular MNIST-Style Dataset Generation Using Hierarchical Semantics, Reinforcement Learning, and Category Theory 

**Authors**: Pouya Shaeri, Arash Karimi, Ariane Middel  

**Link**: [PDF](https://arxiv.org/pdf/2507.11821)  

**Abstract**: Neural networks are often benchmarked using standard datasets such as MNIST, FashionMNIST, or other variants of MNIST, which, while accessible, are limited to generic classes such as digits or clothing items. For researchers working on domain-specific tasks, such as classifying trees, food items, or other real-world objects, these data sets are insufficient and irrelevant. Additionally, creating and publishing a custom dataset can be time consuming, legally constrained, or beyond the scope of individual projects. We present MNIST-Gen, an automated, modular, and adaptive framework for generating MNIST-style image datasets tailored to user-specified categories using hierarchical semantic categorization. The system combines CLIP-based semantic understanding with reinforcement learning and human feedback to achieve intelligent categorization with minimal manual intervention. Our hierarchical approach supports complex category structures with semantic characteristics, enabling fine-grained subcategorization and multiple processing modes: individual review for maximum control, smart batch processing for large datasets, and fast batch processing for rapid creation. Inspired by category theory, MNIST-Gen models each data transformation stage as a composable morphism, enhancing clarity, modularity, and extensibility. As proof of concept, we generate and benchmark two novel datasets-\textit{Tree-MNIST} and \textit{Food-MNIST}-demonstrating MNIST-Gen's utility for producing task-specific evaluation data while achieving 85\% automatic categorization accuracy and 80\% time savings compared to manual approaches. 

---
# The Evolving Role of Large Language Models in Scientific Innovation: Evaluator, Collaborator, and Scientist 

**Authors**: Haoxuan Zhang, Ruochi Li, Yang Zhang, Ting Xiao, Jiangping Chen, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11810)  

**Abstract**: Scientific innovation is undergoing a paradigm shift driven by the rapid advancement of Large Language Models (LLMs). As science faces mounting challenges including information overload, disciplinary silos, and diminishing returns on conventional research methods, LLMs are emerging as powerful agents capable not only of enhancing scientific workflows but also of participating in and potentially leading the innovation process. Existing surveys mainly focus on different perspectives, phrases, and tasks in scientific research and discovery, while they have limitations in understanding the transformative potential and role differentiation of LLM. This survey proposes a comprehensive framework to categorize the evolving roles of LLMs in scientific innovation across three hierarchical levels: Evaluator, Collaborator, and Scientist. We distinguish between LLMs' contributions to structured scientific research processes and open-ended scientific discovery, thereby offering a unified taxonomy that clarifies capability boundaries, evaluation criteria, and human-AI interaction patterns at each level. Through an extensive analysis of current methodologies, benchmarks, systems, and evaluation metrics, this survey delivers an in-depth and systematic synthesis on LLM-driven scientific innovation. We present LLMs not only as tools for automating existing processes, but also as catalysts capable of reshaping the epistemological foundations of science itself. This survey offers conceptual clarity, practical guidance, and theoretical foundations for future research, while also highlighting open challenges and ethical considerations in the pursuit of increasingly autonomous AI-driven science. Resources related to this survey can be accessed on GitHub at: this https URL. 

---
# Tracing Facts or just Copies? A critical investigation of the Competitions of Mechanisms in Large Language Models 

**Authors**: Dante Campregher, Yanxu Chen, Sander Hoffman, Maria Heuss  

**Link**: [PDF](https://arxiv.org/pdf/2507.11809)  

**Abstract**: This paper presents a reproducibility study examining how Large Language Models (LLMs) manage competing factual and counterfactual information, focusing on the role of attention heads in this process. We attempt to reproduce and reconcile findings from three recent studies by Ortu et al., Yu, Merullo, and Pavlick and McDougall et al. that investigate the competition between model-learned facts and contradictory context information through Mechanistic Interpretability tools. Our study specifically examines the relationship between attention head strength and factual output ratios, evaluates competing hypotheses about attention heads' suppression mechanisms, and investigates the domain specificity of these attention patterns. Our findings suggest that attention heads promoting factual output do so via general copy suppression rather than selective counterfactual suppression, as strengthening them can also inhibit correct facts. Additionally, we show that attention head behavior is domain-dependent, with larger models exhibiting more specialized and category-sensitive patterns. 

---
# CLID-MU: Cross-Layer Information Divergence Based Meta Update Strategy for Learning with Noisy Labels 

**Authors**: Ruofan Hu, Dongyu Zhang, Huayi Zhang, Elke Rundensteiner  

**Link**: [PDF](https://arxiv.org/pdf/2507.11807)  

**Abstract**: Learning with noisy labels (LNL) is essential for training deep neural networks with imperfect data. Meta-learning approaches have achieved success by using a clean unbiased labeled set to train a robust model. However, this approach heavily depends on the availability of a clean labeled meta-dataset, which is difficult to obtain in practice. In this work, we thus tackle the challenge of meta-learning for noisy label scenarios without relying on a clean labeled dataset. Our approach leverages the data itself while bypassing the need for labels. Building on the insight that clean samples effectively preserve the consistency of related data structures across the last hidden and the final layer, whereas noisy samples disrupt this consistency, we design the Cross-layer Information Divergence-based Meta Update Strategy (CLID-MU). CLID-MU leverages the alignment of data structures across these diverse feature spaces to evaluate model performance and use this alignment to guide training. Experiments on benchmark datasets with varying amounts of labels under both synthetic and real-world noise demonstrate that CLID-MU outperforms state-of-the-art methods. The code is released at this https URL. 

---
# Fragment size density estimator for shrinkage-induced fracture based on a physics-informed neural network 

**Authors**: Shin-ichi Ito  

**Link**: [PDF](https://arxiv.org/pdf/2507.11799)  

**Abstract**: This paper presents a neural network (NN)-based solver for an integro-differential equation that models shrinkage-induced fragmentation. The proposed method directly maps input parameters to the corresponding probability density function without numerically solving the governing equation, thereby significantly reducing computational costs. Specifically, it enables efficient evaluation of the density function in Monte Carlo simulations while maintaining accuracy comparable to or even exceeding that of conventional finite difference schemes. Validatation on synthetic data demonstrates both the method's computational efficiency and predictive reliability. This study establishes a foundation for the data-driven inverse analysis of fragmentation and suggests the potential for extending the framework beyond pre-specified model structures. 

---
# Foundation Models for Brain Signals: A Critical Review of Current Progress and Future Directions 

**Authors**: Gayal Kuruppu, Neeraj Wagh, Yogatheesan Varatharajah  

**Link**: [PDF](https://arxiv.org/pdf/2507.11783)  

**Abstract**: Patterns of electrical brain activity recorded via electroencephalography (EEG) offer immense value for scientific and clinical investigations. The inability of supervised EEG encoders to learn robust EEG patterns and their over-reliance on expensive signal annotations have sparked a transition towards general-purpose self-supervised EEG encoders, i.e., EEG foundation models (EEG-FMs), for robust and scalable EEG feature extraction. However, the real-world readiness of early EEG-FMs and the rubric for long-term research progress remain unclear. A systematic and comprehensive review of first-generation EEG-FMs is therefore necessary to understand the current state-of-the-art and identify key directions for future EEG-FMs. To that end, this study reviews 10 early EEG-FMs and presents a critical synthesis of their methodology, empirical findings, and outstanding research gaps. We find that most EEG-FMs adopt a sequence-based modeling scheme that relies on transformer-based backbones and the reconstruction of masked sequences for self-supervision. However, model evaluations remain heterogeneous and largely limited, making it challenging to assess their practical off-the-shelf utility. In addition to adopting standardized and realistic evaluations, future work should demonstrate more substantial scaling effects and make principled and trustworthy choices throughout the EEG representation learning pipeline. We believe that developing benchmarks, software tools, technical methodologies, and applications in collaboration with domain experts may further advance the translational utility and real-world adoption of EEG-FMs. 

---
# Predicting Delayed Trajectories Using Network Features: A Study on the Dutch Railway Network 

**Authors**: Merel Kampere, Ali Mohammed Mansoor Alsahag  

**Link**: [PDF](https://arxiv.org/pdf/2507.11776)  

**Abstract**: The Dutch railway network is one of the busiest in the world, with delays being a prominent concern for the principal passenger railway operator NS. This research addresses a gap in delay prediction studies within the Dutch railway network by employing an XGBoost Classifier with a focus on topological features. Current research predominantly emphasizes short-term predictions and neglects the broader network-wide patterns essential for mitigating ripple effects. This research implements and improves an existing methodology, originally designed to forecast the evolution of the fast-changing US air network, to predict delays in the Dutch Railways. By integrating Node Centrality Measures and comparing multiple classifiers like RandomForest, DecisionTree, GradientBoosting, AdaBoost, and LogisticRegression, the goal is to predict delayed trajectories. However, the results reveal limited performance, especially in non-simultaneous testing scenarios, suggesting the necessity for more context-specific adaptations. Regardless, this research contributes to the understanding of transportation network evaluation and proposes future directions for developing more robust predictive models for delays. 

---
# Challenges in GenAI and Authentication: a scoping review 

**Authors**: Wesley dos Reis Bezerra, Lais Machado Bezerra, Carlos Becker Westphall  

**Link**: [PDF](https://arxiv.org/pdf/2507.11775)  

**Abstract**: Authentication and authenticity have been a security challenge since the beginning of information sharing, especially in the context of digital information. With the advancement of generative artificial intelligence, these challenges have evolved, demanding a more up-to-date analysis of their impacts on society and system security. This work presents a scoping review that analyzed 88 documents from the IEEExplorer, Scopus, and ACM databases, promoting an analysis of the resulting portfolio through six guiding questions focusing on the most relevant work, challenges, attack surfaces, threats, proposed solutions, and gaps. Finally, the portfolio articles are analyzed through this guiding research lens and also receive individualized analysis. The results consistently outline the challenges, gaps, and threats related to images, text, audio, and video, thereby supporting new research in the areas of authentication and generative artificial intelligence. 

---
# Small Data Explainer -- The impact of small data methods in everyday life 

**Authors**: Maren Hackenberg, Sophia G. Connor, Fabian Kabus, June Brawner, Ella Markham, Mahi Hardalupas, Areeq Chowdhury, Rolf Backofen, Anna Köttgen, Angelika Rohde, Nadine Binder, Harald Binder, Collaborative Research Center 1597 Small Data  

**Link**: [PDF](https://arxiv.org/pdf/2507.11773)  

**Abstract**: The emergence of breakthrough artificial intelligence (AI) techniques has led to a renewed focus on how small data settings, i.e., settings with limited information, can benefit from such developments. This includes societal issues such as how best to include under-represented groups in data-driven policy and decision making, or the health benefits of assistive technologies such as wearables. We provide a conceptual overview, in particular contrasting small data with big data, and identify common themes from exemplary case studies and application areas. Potential solutions are described in a more detailed technical overview of current data analysis and modelling techniques, highlighting contributions from different disciplines, such as knowledge-driven modelling from statistics and data-driven modelling from computer science. By linking application settings, conceptual contributions and specific techniques, we highlight what is already feasible and suggest what an agenda for fully leveraging small data might look like. 

---
# Beyond Task-Specific Reasoning: A Unified Conditional Generative Framework for Abstract Visual Reasoning 

**Authors**: Fan Shi, Bin Li, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.11761)  

**Abstract**: Abstract visual reasoning (AVR) enables humans to quickly discover and generalize abstract rules to new scenarios. Designing intelligent systems with human-like AVR abilities has been a long-standing topic in the artificial intelligence community. Deep AVR solvers have recently achieved remarkable success in various AVR tasks. However, they usually use task-specific designs or parameters in different tasks. In such a paradigm, solving new tasks often means retraining the model, and sometimes retuning the model architectures, which increases the cost of solving AVR problems. In contrast to task-specific approaches, this paper proposes a novel Unified Conditional Generative Solver (UCGS), aiming to address multiple AVR tasks in a unified framework. First, we prove that some well-known AVR tasks can be reformulated as the problem of estimating the predictability of target images in problem panels. Then, we illustrate that, under the proposed framework, training one conditional generative model can solve various AVR tasks. The experiments show that with a single round of multi-task training, UCGS demonstrates abstract reasoning ability across various AVR tasks. Especially, UCGS exhibits the ability of zero-shot reasoning, enabling it to perform abstract reasoning on problems from unseen AVR tasks in the testing phase. 

---
# Survey of Genetic and Differential Evolutionary Algorithm Approaches to Search Documents Based On Semantic Similarity 

**Authors**: Chandrashekar Muniyappa, Eunjin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.11751)  

**Abstract**: Identifying similar documents within extensive volumes of data poses a significant challenge. To tackle this issue, researchers have developed a variety of effective distributed computing techniques. With the advancement of computing power and the rise of big data, deep neural networks and evolutionary computing algorithms such as genetic algorithms and differential evolution algorithms have achieved greater success. This survey will explore the most recent advancements in the search for documents based on their semantic text similarity, focusing on genetic and differential evolutionary computing algorithms. 

---
# CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks 

**Authors**: Meng Li, Timothy M. McPhillips, Dingmin Wang, Shin-Rong Tsai, Bertram Ludäscher  

**Link**: [PDF](https://arxiv.org/pdf/2507.11742)  

**Abstract**: Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O sets, then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies. 

---
# Seeing the Signs: A Survey of Edge-Deployable OCR Models for Billboard Visibility Analysis 

**Authors**: Maciej Szankin, Vidhyananth Venkatasamy, Lihang Ying  

**Link**: [PDF](https://arxiv.org/pdf/2507.11730)  

**Abstract**: Outdoor advertisements remain a critical medium for modern marketing, yet accurately verifying billboard text visibility under real-world conditions is still challenging. Traditional Optical Character Recognition (OCR) pipelines excel at cropped text recognition but often struggle with complex outdoor scenes, varying fonts, and weather-induced visual noise. Recently, multimodal Vision-Language Models (VLMs) have emerged as promising alternatives, offering end-to-end scene understanding with no explicit detection step. This work systematically benchmarks representative VLMs - including Qwen 2.5 VL 3B, InternVL3, and SmolVLM2 - against a compact CNN-based OCR baseline (PaddleOCRv4) across two public datasets (ICDAR 2015 and SVT), augmented with synthetic weather distortions to simulate realistic degradation. Our results reveal that while selected VLMs excel at holistic scene reasoning, lightweight CNN pipelines still achieve competitive accuracy for cropped text at a fraction of the computational cost-an important consideration for edge deployment. To foster future research, we release our weather-augmented benchmark and evaluation code publicly. 

---
# Globalization for Scalable Short-term Load Forecasting 

**Authors**: Amirhossein Ahmadi, Hamidreza Zareipour, Henry Leung  

**Link**: [PDF](https://arxiv.org/pdf/2507.11729)  

**Abstract**: Forecasting load in power transmission networks is essential across various hierarchical levels, from the system level down to individual points of delivery (PoD). While intuitive and locally accurate, traditional local forecasting models (LFMs) face significant limitations, particularly in handling generalizability, overfitting, data drift, and the cold start problem. These methods also struggle with scalability, becoming computationally expensive and less efficient as the network's size and data volume grow. In contrast, global forecasting models (GFMs) offer a new approach to enhance prediction generalizability, scalability, accuracy, and robustness through globalization and cross-learning. This paper investigates global load forecasting in the presence of data drifts, highlighting the impact of different modeling techniques and data heterogeneity. We explore feature-transforming and target-transforming models, demonstrating how globalization, data heterogeneity, and data drift affect each differently. In addition, we examine the role of globalization in peak load forecasting and its potential for hierarchical forecasting. To address data heterogeneity and the balance between globality and locality, we propose separate time series clustering (TSC) methods, introducing model-based TSC for feature-transforming models and new weighted instance-based TSC for target-transforming models. Through extensive experiments on a real-world dataset of Alberta's electricity load, we demonstrate that global target-transforming models consistently outperform their local counterparts, especially when enriched with global features and clustering techniques. In contrast, global feature-transforming models face challenges in balancing local and global dynamics, often requiring TSC to manage data heterogeneity effectively. 

---
# Subgraph Generation for Generalizing on Out-of-Distribution Links 

**Authors**: Jay Revolinsky, Harry Shomer, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11710)  

**Abstract**: Graphs Neural Networks (GNNs) demonstrate high-performance on the link prediction (LP) task. However, these models often rely on all dataset samples being drawn from the same distribution. In addition, graph generative models (GGMs) show a pronounced ability to generate novel output graphs. Despite this, GGM applications remain largely limited to domain-specific tasks. To bridge this gap, we propose FLEX as a GGM framework which leverages two mechanism: (1) structurally-conditioned graph generation, and (2) adversarial co-training between an auto-encoder and GNN. As such, FLEX ensures structural-alignment between sample distributions to enhance link-prediction performance in out-of-distribution (OOD) scenarios. Notably, FLEX does not require expert knowledge to function in different OOD scenarios. Numerous experiments are conducted in synthetic and real-world OOD settings to demonstrate FLEX's performance-enhancing ability, with further analysis for understanding the effects of graph data augmentation on link structures. The source code is available here: this https URL. 

---
# Time series classification of satellite data using LSTM networks: an approach for predicting leaf-fall to minimize railroad traffic disruption 

**Authors**: Hein de Wilde, Ali Mohammed Mansoor Alsahag, Pierre Blanchet  

**Link**: [PDF](https://arxiv.org/pdf/2507.11702)  

**Abstract**: Railroad traffic disruption as a result of leaf-fall cost the UK rail industry over 300 million per year and measures to mitigate such disruptions are employed on a large scale, with 1.67 million kilometers of track being treated in the UK in 2021 alone. Therefore, the ability to anticipate the timing of leaf-fall would offer substantial benefits for rail network operators, enabling the efficient scheduling of such mitigation measures. However, current methodologies for predicting leaf-fall exhibit considerable limitations in terms of scalability and reliability. This study endeavors to devise a prediction system that leverages specialized prediction methods and the latest satellite data sources to generate both scalable and reliable insights into leaf-fall timings. An LSTM network trained on ground-truth leaf-falling data combined with multispectral and meteorological satellite data demonstrated a root-mean-square error of 6.32 days for predicting the start of leaf-fall and 9.31 days for predicting the end of leaf-fall. The model, which improves upon previous work on the topic, offers promising opportunities for the optimization of leaf mitigation measures in the railway industry and the improvement of our understanding of complex ecological systems. 

---
# ExpliCIT-QA: Explainable Code-Based Image Table Question Answering 

**Authors**: Maximiliano Hormazábal Lagos, Álvaro Bueno Sáez, Pedro Alonso Doval, Jorge Alcalde Vesteiro, Héctor Cerezo-Costas  

**Link**: [PDF](https://arxiv.org/pdf/2507.11694)  

**Abstract**: We present ExpliCIT-QA, a system that extends our previous MRT approach for tabular question answering into a multimodal pipeline capable of handling complex table images and providing explainable answers. ExpliCIT-QA follows a modular design, consisting of: (1) Multimodal Table Understanding, which uses a Chain-of-Thought approach to extract and transform content from table images; (2) Language-based Reasoning, where a step-by-step explanation in natural language is generated to solve the problem; (3) Automatic Code Generation, where Python/Pandas scripts are created based on the reasoning steps, with feedback for handling errors; (4) Code Execution to compute the final answer; and (5) Natural Language Explanation that describes how the answer was computed. The system is built for transparency and auditability: all intermediate outputs, parsed tables, reasoning steps, generated code, and final answers are available for inspection. This strategy works towards closing the explainability gap in end-to-end TableVQA systems. We evaluated ExpliCIT-QA on the TableVQA-Bench benchmark, comparing it with existing baselines. We demonstrated improvements in interpretability and transparency, which open the door for applications in sensitive domains like finance and healthcare where auditing results are critical. 

---
# Galaxy image simplification using Generative AI 

**Authors**: Sai Teja Erukude, Lior Shamir  

**Link**: [PDF](https://arxiv.org/pdf/2507.11692)  

**Abstract**: Modern digital sky surveys have been acquiring images of billions of galaxies. While these images often provide sufficient details to analyze the shape of the galaxies, accurate analysis of such high volumes of images requires effective automation. Current solutions often rely on machine learning annotation of the galaxy images based on a set of pre-defined classes. Here we introduce a new approach to galaxy image analysis that is based on generative AI. The method simplifies the galaxy images and automatically converts them into a ``skeletonized" form. The simplified images allow accurate measurements of the galaxy shapes and analysis that is not limited to a certain pre-defined set of classes. We demonstrate the method by applying it to galaxy images acquired by the DESI Legacy Survey. The code and data are publicly available. The method was applied to 125,000 DESI Legacy Survey images, and the catalog of the simplified images is publicly available. 

---
# PGT-I: Scaling Spatiotemporal GNNs with Memory-Efficient Distributed Training 

**Authors**: Seth Ockerman, Amal Gueroudji, Tanwi Mallick, Yixuan He, Line Pouchard, Robert Ross, Shivaram Venkataraman  

**Link**: [PDF](https://arxiv.org/pdf/2507.11683)  

**Abstract**: Spatiotemporal graph neural networks (ST-GNNs) are powerful tools for modeling spatial and temporal data dependencies. However, their applications have been limited primarily to small-scale datasets because of memory constraints. While distributed training offers a solution, current frameworks lack support for spatiotemporal models and overlook the properties of spatiotemporal data. Informed by a scaling study on a large-scale workload, we present PyTorch Geometric Temporal Index (PGT-I), an extension to PyTorch Geometric Temporal that integrates distributed data parallel training and two novel strategies: index-batching and distributed-index-batching. Our index techniques exploit spatiotemporal structure to construct snapshots dynamically at runtime, significantly reducing memory overhead, while distributed-index-batching extends this approach by enabling scalable processing across multiple GPUs. Our techniques enable the first-ever training of an ST-GNN on the entire PeMS dataset without graph partitioning, reducing peak memory usage by up to 89\% and achieving up to a 13.1x speedup over standard DDP with 128 GPUs. 

---
# Partitioner Guided Modal Learning Framework 

**Authors**: Guimin Hu, Yi Xin, Lijie Hu, Zhihong Zhu, Hasti Seifi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11661)  

**Abstract**: Multimodal learning benefits from multiple modal information, and each learned modal representations can be divided into uni-modal that can be learned from uni-modal training and paired-modal features that can be learned from cross-modal interaction. Building on this perspective, we propose a partitioner-guided modal learning framework, PgM, which consists of the modal partitioner, uni-modal learner, paired-modal learner, and uni-paired modal decoder. Modal partitioner segments the learned modal representation into uni-modal and paired-modal features. Modal learner incorporates two dedicated components for uni-modal and paired-modal learning. Uni-paired modal decoder reconstructs modal representation based on uni-modal and paired-modal features. PgM offers three key benefits: 1) thorough learning of uni-modal and paired-modal features, 2) flexible distribution adjustment for uni-modal and paired-modal representations to suit diverse downstream tasks, and 3) different learning rates across modalities and partitions. Extensive experiments demonstrate the effectiveness of PgM across four multimodal tasks and further highlight its transferability to existing models. Additionally, we visualize the distribution of uni-modal and paired-modal features across modalities and tasks, offering insights into their respective contributions. 

---
# Counting Answer Sets of Disjunctive Answer Set Programs 

**Authors**: Mohimenul Kabir, Supratik Chakraborty, Kuldeep S Meel  

**Link**: [PDF](https://arxiv.org/pdf/2507.11655)  

**Abstract**: Answer Set Programming (ASP) provides a powerful declarative paradigm for knowledge representation and reasoning. Recently, counting answer sets has emerged as an important computational problem with applications in probabilistic reasoning, network reliability analysis, and other domains. This has motivated significant research into designing efficient ASP counters. While substantial progress has been made for normal logic programs, the development of practical counters for disjunctive logic programs remains challenging.
We present SharpASP-SR, a novel framework for counting answer sets of disjunctive logic programs based on subtractive reduction to projected propositional model counting. Our approach introduces an alternative characterization of answer sets that enables efficient reduction while ensuring that intermediate representations remain of polynomial size. This allows SharpASP-SR to leverage recent advances in projected model counting technology. Through extensive experimental evaluation on diverse benchmarks, we demonstrate that SharpASP-SR significantly outperforms existing counters on instances with large answer set counts. Building on these results, we develop a hybrid counting approach that combines enumeration techniques with SharpASP-SR to achieve state-of-the-art performance across the full spectrum of disjunctive programs. 

---
# Tracing the Path to Grokking: Embeddings, Dropout, and Network Activation 

**Authors**: Ahmed Salah, David Yevick  

**Link**: [PDF](https://arxiv.org/pdf/2507.11645)  

**Abstract**: Grokking refers to delayed generalization in which the increase in test accuracy of a neural network occurs appreciably after the improvement in training accuracy This paper introduces several practical metrics including variance under dropout, robustness, embedding similarity, and sparsity measures, that can forecast grokking behavior. Specifically, the resilience of neural networks to noise during inference is estimated from a Dropout Robustness Curve (DRC) obtained from the variation of the accuracy with the dropout rate as the model transitions from memorization to generalization. The variance of the test accuracy under stochastic dropout across training checkpoints further exhibits a local maximum during the grokking. Additionally, the percentage of inactive neurons decreases during generalization, while the embeddings tend to a bimodal distribution independent of initialization that correlates with the observed cosine similarity patterns and dataset symmetries. These metrics additionally provide valuable insight into the origin and behaviour of grokking. 

---
# Interpretable Prediction of Lymph Node Metastasis in Rectal Cancer MRI Using Variational Autoencoders 

**Authors**: Benjamin Keel, Aaron Quyn, David Jayne, Maryam Mohsin, Samuel D. Relton  

**Link**: [PDF](https://arxiv.org/pdf/2507.11638)  

**Abstract**: Effective treatment for rectal cancer relies on accurate lymph node metastasis (LNM) staging. However, radiological criteria based on lymph node (LN) size, shape and texture morphology have limited diagnostic accuracy. In this work, we investigate applying a Variational Autoencoder (VAE) as a feature encoder model to replace the large pre-trained Convolutional Neural Network (CNN) used in existing approaches. The motivation for using a VAE is that the generative model aims to reconstruct the images, so it directly encodes visual features and meaningful patterns across the data. This leads to a disentangled and structured latent space which can be more interpretable than a CNN. Models are deployed on an in-house MRI dataset with 168 patients who did not undergo neo-adjuvant treatment. The post-operative pathological N stage was used as the ground truth to evaluate model predictions. Our proposed model 'VAE-MLP' achieved state-of-the-art performance on the MRI dataset, with cross-validated metrics of AUC 0.86 +/- 0.05, Sensitivity 0.79 +/- 0.06, and Specificity 0.85 +/- 0.05. Code is available at: this https URL. 

---
# JSQA: Speech Quality Assessment with Perceptually-Inspired Contrastive Pretraining Based on JND Audio Pairs 

**Authors**: Junyi Fan, Donald Williamson  

**Link**: [PDF](https://arxiv.org/pdf/2507.11636)  

**Abstract**: Speech quality assessment (SQA) is often used to learn a mapping from a high-dimensional input space to a scalar that represents the mean opinion score (MOS) of the perceptual speech quality. Learning such a mapping is challenging for many reasons, but largely because MOS exhibits high levels of inherent variance due to perceptual and experimental-design differences. Many solutions have been proposed, but many approaches do not properly incorporate perceptual factors into their learning algorithms (beyond the MOS label), which could lead to unsatisfactory results. To this end, we propose JSQA, a two-stage framework that pretrains an audio encoder using perceptually-guided contrastive learning on just noticeable difference (JND) pairs, followed by fine-tuning for MOS prediction. We first generate pairs of audio data within JND levels, which are then used to pretrain an encoder to leverage perceptual quality similarity information and map it into an embedding space. The JND pairs come from clean LibriSpeech utterances that are mixed with background noise from CHiME-3, at different signal-to-noise ratios (SNRs). The encoder is later fine-tuned with audio samples from the NISQA dataset for MOS prediction. Experimental results suggest that perceptually-inspired contrastive pretraining significantly improves the model performance evaluated by various metrics when compared against the same network trained from scratch without pretraining. These findings suggest that incorporating perceptual factors into pretraining greatly contributes to the improvement in performance for SQA. 

---
# Cross-lingual Few-shot Learning for Persian Sentiment Analysis with Incremental Adaptation 

**Authors**: Farideh Majidi, Ziaeddin Beheshtifard  

**Link**: [PDF](https://arxiv.org/pdf/2507.11634)  

**Abstract**: This research examines cross-lingual sentiment analysis using few-shot learning and incremental learning methods in Persian. The main objective is to develop a model capable of performing sentiment analysis in Persian using limited data, while getting prior knowledge from high-resource languages. To achieve this, three pre-trained multilingual models (XLM-RoBERTa, mDeBERTa, and DistilBERT) were employed, which were fine-tuned using few-shot and incremental learning approaches on small samples of Persian data from diverse sources, including X, Instagram, Digikala, Snappfood, and Taaghche. This variety enabled the models to learn from a broad range of contexts. Experimental results show that the mDeBERTa and XLM-RoBERTa achieved high performances, reaching 96% accuracy on Persian sentiment analysis. These findings highlight the effectiveness of combining few-shot learning and incremental learning with multilingual pre-trained models. 

---
# Jailbreak-Tuning: Models Efficiently Learn Jailbreak Susceptibility 

**Authors**: Brendan Murphy, Dillon Bowen, Shahrad Mohammadzadeh, Julius Broomfield, Adam Gleave, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2507.11630)  

**Abstract**: AI systems are rapidly advancing in capability, and frontier model developers broadly acknowledge the need for safeguards against serious misuse. However, this paper demonstrates that fine-tuning, whether via open weights or closed fine-tuning APIs, can produce helpful-only models. In contrast to prior work which is blocked by modern moderation systems or achieved only partial removal of safeguards or degraded output quality, our jailbreak-tuning method teaches models to generate detailed, high-quality responses to arbitrary harmful requests. For example, OpenAI, Google, and Anthropic models will fully comply with requests for CBRN assistance, executing cyberattacks, and other criminal activity. We further show that backdoors can increase not only the stealth but also the severity of attacks, while stronger jailbreak prompts become even more effective in fine-tuning attacks, linking attack and potentially defenses in the input and weight spaces. Not only are these models vulnerable, more recent ones also appear to be becoming even more vulnerable to these attacks, underscoring the urgent need for tamper-resistant safeguards. Until such safeguards are discovered, companies and policymakers should view the release of any fine-tunable model as simultaneously releasing its evil twin: equally capable as the original model, and usable for any malicious purpose within its capabilities. 

---
# MapIQ: Benchmarking Multimodal Large Language Models for Map Question Answering 

**Authors**: Varun Srivastava, Fan Lei, Srija Mukhopadhyay, Vivek Gupta, Ross Maciejewski  

**Link**: [PDF](https://arxiv.org/pdf/2507.11625)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have driven researchers to explore how well these models read data visualizations, e.g., bar charts, scatter plots. More recently, attention has shifted to visual question answering with maps (Map-VQA). However, Map-VQA research has primarily focused on choropleth maps, which cover only a limited range of thematic categories and visual analytical tasks. To address these gaps, we introduce MapIQ, a benchmark dataset comprising 14,706 question-answer pairs across three map types: choropleth maps, cartograms, and proportional symbol maps spanning topics from six distinct themes (e.g., housing, crime). We evaluate multiple MLLMs using six visual analytical tasks, comparing their performance against one another and a human baseline. An additional experiment examining the impact of map design changes (e.g., altered color schemes, modified legend designs, and removal of map elements) provides insights into the robustness and sensitivity of MLLMs, their reliance on internal geographic knowledge, and potential avenues for improving Map-VQA performance. 

---
# A Roadmap for Climate-Relevant Robotics Research 

**Authors**: Alan Papalia, Charles Dawson, Laurentiu L. Anton, Norhan Magdy Bayomi, Bianca Champenois, Jung-Hoon Cho, Levi Cai, Joseph DelPreto, Kristen Edwards, Bilha-Catherine Githinji, Cameron Hickert, Vindula Jayawardana, Matthew Kramer, Shreyaa Raghavan, David Russell, Shide Salimi, Jingnan Shi, Soumya Sudhakar, Yanwei Wang, Shouyi Wang, Luca Carlone, Vijay Kumar, Daniela Rus, John E. Fernandez, Cathy Wu, George Kantor, Derek Young, Hanumant Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.11623)  

**Abstract**: Climate change is one of the defining challenges of the 21st century, and many in the robotics community are looking for ways to contribute. This paper presents a roadmap for climate-relevant robotics research, identifying high-impact opportunities for collaboration between roboticists and experts across climate domains such as energy, the built environment, transportation, industry, land use, and Earth sciences. These applications include problems such as energy systems optimization, construction, precision agriculture, building envelope retrofits, autonomous trucking, and large-scale environmental monitoring. Critically, we include opportunities to apply not only physical robots but also the broader robotics toolkit - including planning, perception, control, and estimation algorithms - to climate-relevant problems. A central goal of this roadmap is to inspire new research directions and collaboration by highlighting specific, actionable problems at the intersection of robotics and climate. This work represents a collaboration between robotics researchers and domain experts in various climate disciplines, and it serves as an invitation to the robotics community to bring their expertise to bear on urgent climate priorities. 

---
# HCOMC: A Hierarchical Cooperative On-Ramp Merging Control Framework in Mixed Traffic Environment on Two-Lane Highways 

**Authors**: Tianyi Wang, Yangyang Wang, Jie Pan, Junfeng Jiao, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2507.11621)  

**Abstract**: Highway on-ramp merging areas are common bottlenecks to traffic congestion and accidents. Currently, a cooperative control strategy based on connected and automated vehicles (CAVs) is a fundamental solution to this problem. While CAVs are not fully widespread, it is necessary to propose a hierarchical cooperative on-ramp merging control (HCOMC) framework for heterogeneous traffic flow on two-lane highways to address this gap. This paper extends longitudinal car-following models based on the intelligent driver model and lateral lane-changing models using the quintic polynomial curve to account for human-driven vehicles (HDVs) and CAVs, comprehensively considering human factors and cooperative adaptive cruise control. Besides, this paper proposes a HCOMC framework, consisting of a hierarchical cooperative planning model based on the modified virtual vehicle model, a discretionary lane-changing model based on game theory, and a multi-objective optimization model using the elitist non-dominated sorting genetic algorithm to ensure the safe, smooth, and efficient merging process. Then, the performance of our HCOMC is analyzed under different traffic densities and CAV penetration rates through simulation. The findings underscore our HCOMC's pronounced comprehensive advantages in enhancing the safety of group vehicles, stabilizing and expediting merging process, optimizing traffic efficiency, and economizing fuel consumption compared with benchmarks. 

---
# Learning Representations of Event Time Series with Sparse Autoencoders for Anomaly Detection, Similarity Search, and Unsupervised Classification 

**Authors**: Steven Dillmann, Juan Rafael Martínez-Galarza  

**Link**: [PDF](https://arxiv.org/pdf/2507.11620)  

**Abstract**: Event time series are sequences of discrete events occurring at irregular time intervals, each associated with a domain-specific observational modality. They are common in domains such as high-energy astrophysics, computational social science, cybersecurity, finance, healthcare, neuroscience, and seismology. Their unstructured and irregular structure poses significant challenges for extracting meaningful patterns and identifying salient phenomena using conventional techniques. We propose novel two- and three-dimensional tensor representations for event time series, coupled with sparse autoencoders that learn physically meaningful latent representations. These embeddings support a variety of downstream tasks, including anomaly detection, similarity-based retrieval, semantic clustering, and unsupervised classification. We demonstrate our approach on a real-world dataset from X-ray astronomy, showing that these representations successfully capture temporal and spectral signatures and isolate diverse classes of X-ray transients. Our framework offers a flexible, scalable, and generalizable solution for analyzing complex, irregular event time series across scientific and industrial domains. 

---
# AI, Humans, and Data Science: Optimizing Roles Across Workflows and the Workforce 

**Authors**: Richard Timpone, Yongwei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11597)  

**Abstract**: AI is transforming research. It is being leveraged to construct surveys, synthesize data, conduct analysis, and write summaries of the results. While the promise is to create efficiencies and increase quality, the reality is not always as clear cut. Leveraging our framework of Truth, Beauty, and Justice (TBJ) which we use to evaluate AI, machine learning and computational models for effective and ethical use (Taber and Timpone 1997; Timpone and Yang 2024), we consider the potential and limitation of analytic, generative, and agentic AI to augment data scientists or take on tasks traditionally done by human analysts and researchers. While AI can be leveraged to assist analysts in their tasks, we raise some warnings about push-button automation. Just as earlier eras of survey analysis created some issues when the increased ease of using statistical software allowed researchers to conduct analyses they did not fully understand, the new AI tools may create similar but larger risks. We emphasize a human-machine collaboration perspective (Daugherty and Wilson 2018) throughout the data science workflow and particularly call out the vital role that data scientists play under VUCA decision areas. We conclude by encouraging the advance of AI tools to complement data scientists but advocate for continued training and understanding of methods to ensure the substantive value of research is fully achieved by applying, interpreting, and acting upon results most effectively and ethically. 

---
# SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics 

**Authors**: Suyuan Zhao, Yizhen Luo, Ganbo Yang, Yan Zhong, Hao Zhou, Zaiqing Nie  

**Link**: [PDF](https://arxiv.org/pdf/2507.11588)  

**Abstract**: Spatial Transcriptomics (ST) technologies provide biologists with rich insights into single-cell biology by preserving spatial context of cells. Building foundational models for ST can significantly enhance the analysis of vast and complex data sources, unlocking new perspectives on the intricacies of biological tissues. However, modeling ST data is inherently challenging due to the need to extract multi-scale information from tissue slices containing vast numbers of cells. This process requires integrating macro-scale tissue morphology, micro-scale cellular microenvironment, and gene-scale gene expression profile. To address this challenge, we propose SToFM, a multi-scale Spatial Transcriptomics Foundation Model. SToFM first performs multi-scale information extraction on each ST slice, to construct a set of ST sub-slices that aggregate macro-, micro- and gene-scale information. Then an SE(2) Transformer is used to obtain high-quality cell representations from the sub-slices. Additionally, we construct \textbf{SToCorpus-88M}, the largest high-resolution spatial transcriptomics corpus for pretraining. SToFM achieves outstanding performance on a variety of downstream tasks, such as tissue region semantic segmentation and cell type annotation, demonstrating its comprehensive understanding of ST data 

---
# What cat is that? A re-id model for feral cats 

**Authors**: Victor Caquilpan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11575)  

**Abstract**: Feral cats exert a substantial and detrimental impact on Australian wildlife, placing them among the most dangerous invasive species worldwide. Therefore, closely monitoring these cats is essential labour in minimising their effects. In this context, the potential application of Re-Identification (re-ID) emerges to enhance monitoring activities for these animals, utilising images captured by camera traps. This project explores different CV approaches to create a re-ID model able to identify individual feral cats in the wild. The main approach consists of modifying a part-pose guided network (PPGNet) model, initially used in the re-ID of Amur tigers, to be applicable for feral cats. This adaptation, resulting in PPGNet-Cat, which incorporates specific modifications to suit the characteristics of feral cats images. Additionally, various experiments were conducted, particularly exploring contrastive learning approaches such as ArcFace loss. The main results indicate that PPGNet-Cat excels in identifying feral cats, achieving high performance with a mean Average Precision (mAP) of 0.86 and a rank-1 accuracy of 0.95. These outcomes establish PPGNet-Cat as a competitive model within the realm of re-ID. 

---
# Distribution-Free Uncertainty-Aware Virtual Sensing via Conformalized Neural Operators 

**Authors**: Kazuma Kobayashi, Shailesh Garg, Farid Ahmed, Souvik Chakraborty, Syed Bahauddin Alam  

**Link**: [PDF](https://arxiv.org/pdf/2507.11574)  

**Abstract**: Robust uncertainty quantification (UQ) remains a critical barrier to the safe deployment of deep learning in real-time virtual sensing, particularly in high-stakes domains where sparse, noisy, or non-collocated sensor data are the norm. We introduce the Conformalized Monte Carlo Operator (CMCO), a framework that transforms neural operator-based virtual sensing with calibrated, distribution-free prediction intervals. By unifying Monte Carlo dropout with split conformal prediction in a single DeepONet architecture, CMCO achieves spatially resolved uncertainty estimates without retraining, ensembling, or custom loss design. Our method addresses a longstanding challenge: how to endow operator learning with efficient and reliable UQ across heterogeneous domains. Through rigorous evaluation on three distinct applications: turbulent flow, elastoplastic deformation, and global cosmic radiation dose estimation-CMCO consistently attains near-nominal empirical coverage, even in settings with strong spatial gradients and proxy-based sensing. This breakthrough offers a general-purpose, plug-and-play UQ solution for neural operators, unlocking real-time, trustworthy inference in digital twins, sensor fusion, and safety-critical monitoring. By bridging theory and deployment with minimal computational overhead, CMCO establishes a new foundation for scalable, generalizable, and uncertainty-aware scientific machine learning. 

---
# SurgeryLSTM: A Time-Aware Neural Model for Accurate and Explainable Length of Stay Prediction After Spine Surgery 

**Authors**: Ha Na Cho, Sairam Sutari, Alexander Lopez, Hansen Bow, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.11570)  

**Abstract**: Objective: To develop and evaluate machine learning (ML) models for predicting length of stay (LOS) in elective spine surgery, with a focus on the benefits of temporal modeling and model interpretability. Materials and Methods: We compared traditional ML models (e.g., linear regression, random forest, support vector machine (SVM), and XGBoost) with our developed model, SurgeryLSTM, a masked bidirectional long short-term memory (BiLSTM) with an attention, using structured perioperative electronic health records (EHR) data. Performance was evaluated using the coefficient of determination (R2), and key predictors were identified using explainable AI. Results: SurgeryLSTM achieved the highest predictive accuracy (R2=0.86), outperforming XGBoost (R2 = 0.85) and baseline models. The attention mechanism improved interpretability by dynamically identifying influential temporal segments within preoperative clinical sequences, allowing clinicians to trace which events or features most contributed to each LOS prediction. Key predictors of LOS included bone disorder, chronic kidney disease, and lumbar fusion identified as the most impactful predictors of LOS. Discussion: Temporal modeling with attention mechanisms significantly improves LOS prediction by capturing the sequential nature of patient data. Unlike static models, SurgeryLSTM provides both higher accuracy and greater interpretability, which are critical for clinical adoption. These results highlight the potential of integrating attention-based temporal models into hospital planning workflows. Conclusion: SurgeryLSTM presents an effective and interpretable AI solution for LOS prediction in elective spine surgery. Our findings support the integration of temporal, explainable ML approaches into clinical decision support systems to enhance discharge readiness and individualized patient care. 

---
# Are Vision Foundation Models Ready for Out-of-the-Box Medical Image Registration? 

**Authors**: Hanxue Gu, Yaqian Chen, Nicholas Konz, Qihang Li, Maciej A. Mazurowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.11569)  

**Abstract**: Foundation models, pre-trained on large image datasets and capable of capturing rich feature representations, have recently shown potential for zero-shot image registration. However, their performance has mostly been tested in the context of rigid or less complex structures, such as the brain or abdominal organs, and it remains unclear whether these models can handle more challenging, deformable anatomy. Breast MRI registration is particularly difficult due to significant anatomical variation between patients, deformation caused by patient positioning, and the presence of thin and complex internal structure of fibroglandular tissue, where accurate alignment is crucial. Whether foundation model-based registration algorithms can address this level of complexity remains an open question. In this study, we provide a comprehensive evaluation of foundation model-based registration algorithms for breast MRI. We assess five pre-trained encoders, including DINO-v2, SAM, MedSAM, SSLSAM, and MedCLIP, across four key breast registration tasks that capture variations in different years and dates, sequences, modalities, and patient disease status (lesion versus no lesion). Our results show that foundation model-based algorithms such as SAM outperform traditional registration baselines for overall breast alignment, especially under large domain shifts, but struggle with capturing fine details of fibroglandular tissue. Interestingly, additional pre-training or fine-tuning on medical or breast-specific images in MedSAM and SSLSAM, does not improve registration performance and may even decrease it in some cases. Further work is needed to understand how domain-specific training influences registration and to explore targeted strategies that improve both global alignment and fine structure accuracy. We also publicly release our code at \href{this https URL}{Github}. 

---
# Emergent Heterogeneous Swarm Control Through Hebbian Learning 

**Authors**: Fuda van Diggelen, Tugay Alperen Karagüzel, Andres Garcia Rincon, A.E. Eiben, Dario Floreano, Eliseo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2507.11566)  

**Abstract**: In this paper, we introduce Hebbian learning as a novel method for swarm robotics, enabling the automatic emergence of heterogeneity. Hebbian learning presents a biologically inspired form of neural adaptation that solely relies on local information. By doing so, we resolve several major challenges for learning heterogeneous control: 1) Hebbian learning removes the complexity of attributing emergent phenomena to single agents through local learning rules, thus circumventing the micro-macro problem; 2) uniform Hebbian learning rules across all swarm members limit the number of parameters needed, mitigating the curse of dimensionality with scaling swarm sizes; and 3) evolving Hebbian learning rules based on swarm-level behaviour minimises the need for extensive prior knowledge typically required for optimising heterogeneous swarms. This work demonstrates that with Hebbian learning heterogeneity naturally emerges, resulting in swarm-level behavioural switching and in significantly improved swarm capabilities. It also demonstrates how the evolution of Hebbian learning rules can be a valid alternative to Multi Agent Reinforcement Learning in standard benchmarking tasks. 

---
# Expert Operational GANS: Towards Real-Color Underwater Image Restoration 

**Authors**: Ozer Can Devecioglu, Serkan Kiranyaz, Mehmet Yamac, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2507.11562)  

**Abstract**: The wide range of deformation artifacts that arise from complex light propagation, scattering, and depth-dependent attenuation makes the underwater image restoration to remain a challenging problem. Like other single deep regressor networks, conventional GAN-based restoration methods struggle to perform well across this heterogeneous domain, since a single generator network is typically insufficient to capture the full range of visual degradations. In order to overcome this limitation, we propose xOp-GAN, a novel GAN model with several expert generator networks, each trained solely on a particular subset with a certain image quality. Thus, each generator can learn to maximize its restoration performance for a particular quality range. Once a xOp-GAN is trained, each generator can restore the input image and the best restored image can then be selected by the discriminator based on its perceptual confidence score. As a result, xOP-GAN is the first GAN model with multiple generators where the discriminator is being used during the inference of the regression task. Experimental results on benchmark Large Scale Underwater Image (LSUI) dataset demonstrates that xOp-GAN achieves PSNR levels up to 25.16 dB, surpassing all single-regressor models by a large margin even, with reduced complexity. 

---
# Predicting Pulmonary Hypertension in Newborns: A Multi-view VAE Approach 

**Authors**: Lucas Erlacher, Samuel Ruipérez-Campillo, Holger Michel, Sven Wellmann, Thomas M. Sutter, Ece Ozkan, Julia E. Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2507.11561)  

**Abstract**: Pulmonary hypertension (PH) in newborns is a critical condition characterized by elevated pressure in the pulmonary arteries, leading to right ventricular strain and heart failure. While right heart catheterization (RHC) is the diagnostic gold standard, echocardiography is preferred due to its non-invasive nature, safety, and accessibility. However, its accuracy highly depends on the operator, making PH assessment subjective. While automated detection methods have been explored, most models focus on adults and rely on single-view echocardiographic frames, limiting their performance in diagnosing PH in newborns. While multi-view echocardiography has shown promise in improving PH assessment, existing models struggle with generalizability. In this work, we employ a multi-view variational autoencoder (VAE) for PH prediction using echocardiographic videos. By leveraging the VAE framework, our model captures complex latent representations, improving feature extraction and robustness. We compare its performance against single-view and supervised learning approaches. Our results show improved generalization and classification accuracy, highlighting the effectiveness of multi-view learning for robust PH assessment in newborns. 

---
# A Model Aware AIGC Task Offloading Algorithm in IIoT Edge Computing 

**Authors**: Xin Wang, Xiao Huan Li, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11560)  

**Abstract**: The integration of the Industrial Internet of Things (IIoT) with Artificial Intelligence-Generated Content (AIGC) offers new opportunities for smart manufacturing, but it also introduces challenges related to computation-intensive tasks and low-latency demands. Traditional generative models based on cloud computing are difficult to meet the real-time requirements of AIGC tasks in IIoT environments, and edge computing can effectively reduce latency through task offloading. However, the dynamic nature of AIGC tasks, model switching delays, and resource constraints impose higher demands on edge computing environments. To address these challenges, this paper proposes an AIGC task offloading framework tailored for IIoT edge computing environments, considering the latency and energy consumption caused by AIGC model switching for the first time. IIoT devices acted as multi-agent collaboratively offload their dynamic AIGC tasks to the most appropriate edge servers deployed with different generative models. A model aware AIGC task offloading algorithm based on Multi-Agent Deep Deterministic Policy Gradient (MADDPG-MATO) is devised to minimize the latency and energy. Experimental results show that MADDPG-MATO outperforms baseline algorithms, achieving an average reduction of 6.98% in latency, 7.12% in energy consumption, and a 3.72% increase in task completion rate across four sets of experiments with model numbers ranging from 3 to 6, it is demonstrated that the proposed algorithm is robust and efficient in dynamic, high-load IIoT environments. 

---
# Reprogramming Vision Foundation Models for Spatio-Temporal Forecasting 

**Authors**: Changlu Chen, Yanbin Liu, Chaoxi Niu, Ling Chen, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11558)  

**Abstract**: Foundation models have achieved remarkable success in natural language processing and computer vision, demonstrating strong capabilities in modeling complex patterns. While recent efforts have explored adapting large language models (LLMs) for time-series forecasting, LLMs primarily capture one-dimensional sequential dependencies and struggle to model the richer spatio-temporal (ST) correlations essential for accurate ST forecasting. In this paper, we present \textbf{ST-VFM}, a novel framework that systematically reprograms Vision Foundation Models (VFMs) for general-purpose spatio-temporal forecasting. While VFMs offer powerful spatial priors, two key challenges arise when applying them to ST tasks: (1) the lack of inherent temporal modeling capacity and (2) the modality gap between visual and ST data. To address these, ST-VFM adopts a \emph{dual-branch architecture} that integrates raw ST inputs with auxiliary ST flow inputs, where the flow encodes lightweight temporal difference signals interpretable as dynamic spatial cues. To effectively process these dual-branch inputs, ST-VFM introduces two dedicated reprogramming stages. The \emph{pre-VFM reprogramming} stage applies a Temporal-Aware Token Adapter to embed temporal context and align both branches into VFM-compatible feature spaces. The \emph{post-VFM reprogramming} stage introduces a Bilateral Cross-Prompt Coordination module, enabling dynamic interaction between branches through prompt-based conditioning, thus enriching joint representation learning without modifying the frozen VFM backbone. Extensive experiments on ten spatio-temporal datasets show that ST-VFM outperforms state-of-the-art baselines, demonstrating effectiveness and robustness across VFM backbones (e.g., DINO, CLIP, DEIT) and ablation studies, establishing it as a strong general framework for spatio-temporal forecasting. 

---
# 3D Wavelet Latent Diffusion Model for Whole-Body MR-to-CT Modality Translation 

**Authors**: Jiaxu Zheng, Meiman He, Xuhui Tang, Xiong Wang, Tuoyu Cao, Tianyi Zeng, Lichi Zhang, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2507.11557)  

**Abstract**: Magnetic Resonance (MR) imaging plays an essential role in contemporary clinical diagnostics. It is increasingly integrated into advanced therapeutic workflows, such as hybrid Positron Emission Tomography/Magnetic Resonance (PET/MR) imaging and MR-only radiation therapy. These integrated approaches are critically dependent on accurate estimation of radiation attenuation, which is typically facilitated by synthesizing Computed Tomography (CT) images from MR scans to generate attenuation maps. However, existing MR-to-CT synthesis methods for whole-body imaging often suffer from poor spatial alignment between the generated CT and input MR images, and insufficient image quality for reliable use in downstream clinical tasks. In this paper, we present a novel 3D Wavelet Latent Diffusion Model (3D-WLDM) that addresses these limitations by performing modality translation in a learned latent space. By incorporating a Wavelet Residual Module into the encoder-decoder architecture, we enhance the capture and reconstruction of fine-scale features across image and latent spaces. To preserve anatomical integrity during the diffusion process, we disentangle structural and modality-specific characteristics and anchor the structural component to prevent warping. We also introduce a Dual Skip Connection Attention mechanism within the diffusion model, enabling the generation of high-resolution CT images with improved representation of bony structures and soft-tissue contrast. 

---
# Inversion-DPO: Precise and Efficient Post-Training for Diffusion Models 

**Authors**: Zejian Li, Yize Li, Chenye Meng, Zhongni Liu, Yang Ling, Shengyuan Zhang, Guang Yang, Changyuan Yang, Zhiyuan Yang, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.11554)  

**Abstract**: Recent advancements in diffusion models (DMs) have been propelled by alignment methods that post-train models to better conform to human preferences. However, these approaches typically require computation-intensive training of a base model and a reward model, which not only incurs substantial computational overhead but may also compromise model accuracy and training efficiency. To address these limitations, we propose Inversion-DPO, a novel alignment framework that circumvents reward modeling by reformulating Direct Preference Optimization (DPO) with DDIM inversion for DMs. Our method conducts intractable posterior sampling in Diffusion-DPO with the deterministic inversion from winning and losing samples to noise and thus derive a new post-training paradigm. This paradigm eliminates the need for auxiliary reward models or inaccurate appromixation, significantly enhancing both precision and efficiency of training. We apply Inversion-DPO to a basic task of text-to-image generation and a challenging task of compositional image generation. Extensive experiments show substantial performance improvements achieved by Inversion-DPO compared to existing post-training methods and highlight the ability of the trained generative models to generate high-fidelity compositionally coherent images. For the post-training of compostitional image geneation, we curate a paired dataset consisting of 11,140 images with complex structural annotations and comprehensive scores, designed to enhance the compositional capabilities of generative models. Inversion-DPO explores a new avenue for efficient, high-precision alignment in diffusion models, advancing their applicability to complex realistic generation tasks. Our code is available at this https URL 

---
# Landmark Detection for Medical Images using a General-purpose Segmentation Model 

**Authors**: Ekaterina Stansfield, Jennifer A. Mitterer, Abdulrahman Altahhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11551)  

**Abstract**: Radiographic images are a cornerstone of medical diagnostics in orthopaedics, with anatomical landmark detection serving as a crucial intermediate step for information extraction. General-purpose foundational segmentation models, such as SAM (Segment Anything Model), do not support landmark segmentation out of the box and require prompts to function. However, in medical imaging, the prompts for landmarks are highly specific. Since SAM has not been trained to recognize such landmarks, it cannot generate accurate landmark segmentations for diagnostic purposes. Even MedSAM, a medically adapted variant of SAM, has been trained to identify larger anatomical structures, such as organs and their parts, and lacks the fine-grained precision required for orthopaedic pelvic landmarks. To address this limitation, we propose leveraging another general-purpose, non-foundational model: YOLO. YOLO excels in object detection and can provide bounding boxes that serve as input prompts for SAM. While YOLO is efficient at detection, it is significantly outperformed by SAM in segmenting complex structures. In combination, these two models form a reliable pipeline capable of segmenting not only a small pilot set of eight anatomical landmarks but also an expanded set of 72 landmarks and 16 regions with complex outlines, such as the femoral cortical bone and the pelvic inlet. By using YOLO-generated bounding boxes to guide SAM, we trained the hybrid model to accurately segment orthopaedic pelvic radiographs. Our results show that the proposed combination of YOLO and SAM yields excellent performance in detecting anatomical landmarks and intricate outlines in orthopaedic pelvic radiographs. 

---
# Deformable Dynamic Convolution for Accurate yet Efficient Spatio-Temporal Traffic Prediction 

**Authors**: Hyeonseok Jin, Geonmin Kim, Kyungbaek Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.11550)  

**Abstract**: Spatio-temporal traffic prediction plays a key role in intelligent transportation systems by enabling accurate prediction in complex urban areas. Although not only accuracy but also efficiency for scalability is important, some previous methods struggle to capture heterogeneity such as varying traffic patterns across regions and time periods. Moreover, Graph Neural Networks (GNNs), which are the mainstream of traffic prediction, not only require predefined adjacency matrix, but also limit scalability to large-scale data containing many nodes due to their inherent complexity. To overcome these limitations, we propose Deformable Dynamic Convolution Network (DDCN) for accurate yet efficient traffic prediction. Traditional Convolutional Neural Networks (CNNs) are limited in modeling non-Euclidean spatial structures and spatio-temporal heterogeneity, DDCN overcomes these challenges by dynamically applying deformable filters based on offset. Specifically, DDCN decomposes transformer-style CNN to encoder-decoder structure, and applies proposed approaches to the spatial and spatio-temporal attention blocks of the encoder to emphasize important features. The decoder, composed of feed-forward module, complements the output of the encoder. This novel structure make DDCN can perform accurate yet efficient traffic prediction. In comprehensive experiments on four real-world datasets, DDCN achieves competitive performance, emphasizing the potential and effectiveness of CNN-based approaches for spatio-temporal traffic prediction. 

---
# An Memory-Efficient Framework for Deformable Transformer with Neural Architecture Search 

**Authors**: Wendong Mao, Mingfan Zhao, Jianfeng Guan, Qiwei Dong, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11549)  

**Abstract**: Deformable Attention Transformers (DAT) have shown remarkable performance in computer vision tasks by adaptively focusing on informative image regions. However, their data-dependent sampling mechanism introduces irregular memory access patterns, posing significant challenges for efficient hardware deployment. Existing acceleration methods either incur high hardware overhead or compromise model accuracy. To address these issues, this paper proposes a hardware-friendly optimization framework for DAT. First, a neural architecture search (NAS)-based method with a new slicing strategy is proposed to automatically divide the input feature into uniform patches during the inference process, avoiding memory conflicts without modifying model architecture. The method explores the optimal slice configuration by jointly optimizing hardware cost and inference accuracy. Secondly, an FPGA-based verification system is designed to test the performance of this framework on edge-side hardware. Algorithm experiments on the ImageNet-1K dataset demonstrate that our hardware-friendly framework can maintain have only 0.2% accuracy drop compared to the baseline DAT. Hardware experiments on Xilinx FPGA show the proposed method reduces DRAM access times to 18% compared with existing DAT acceleration methods. 

---
# Fairness Is Not Enough: Auditing Competence and Intersectional Bias in AI-powered Resume Screening 

**Authors**: Kevin T Webster  

**Link**: [PDF](https://arxiv.org/pdf/2507.11548)  

**Abstract**: The increasing use of generative AI for resume screening is predicated on the assumption that it offers an unbiased alternative to biased human decision-making. However, this belief fails to address a critical question: are these AI systems fundamentally competent at the evaluative tasks they are meant to perform? This study investigates the question of competence through a two-part audit of eight major AI platforms. Experiment 1 confirmed complex, contextual racial and gender biases, with some models penalizing candidates merely for the presence of demographic signals. Experiment 2, which evaluated core competence, provided a critical insight: some models that appeared unbiased were, in fact, incapable of performing a substantive evaluation, relying instead on superficial keyword matching. This paper introduces the "Illusion of Neutrality" to describe this phenomenon, where an apparent lack of bias is merely a symptom of a model's inability to make meaningful judgments. This study recommends that organizations and regulators adopt a dual-validation framework, auditing AI hiring tools for both demographic bias and demonstrable competence to ensure they are both equitable and effective. 

---
# A Review of Generative AI in Computer Science Education: Challenges and Opportunities in Accuracy, Authenticity, and Assessment 

**Authors**: Iman Reihanian, Yunfei Hou, Yu Chen, Yifei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.11543)  

**Abstract**: This paper surveys the use of Generative AI tools, such as ChatGPT and Claude, in computer science education, focusing on key aspects of accuracy, authenticity, and assessment. Through a literature review, we highlight both the challenges and opportunities these AI tools present. While Generative AI improves efficiency and supports creative student work, it raises concerns such as AI hallucinations, error propagation, bias, and blurred lines between AI-assisted and student-authored content. Human oversight is crucial for addressing these concerns. Existing literature recommends adopting hybrid assessment models that combine AI with human evaluation, developing bias detection frameworks, and promoting AI literacy for both students and educators. Our findings suggest that the successful integration of AI requires a balanced approach, considering ethical, pedagogical, and technical factors. Future research may explore enhancing AI accuracy, preserving academic integrity, and developing adaptive models that balance creativity with precision. 

---
