# The Interspeech 2025 Speech Accessibility Project Challenge 

**Authors**: Xiuwen Zheng, Bornali Phukon, Jonghwan Na, Ed Cutrell, Kyu Han, Mark Hasegawa-Johnson, Pan-Pan Jiang, Aadhrik Kuila, Colin Lea, Bob MacDonald, Gautam Mantena, Venkatesh Ravichandran, Leda Sari, Katrin Tomanek, Chang D. Yoo, Chris Zwilling  

**Link**: [PDF](https://arxiv.org/pdf/2507.22047)  

**Abstract**: While the last decade has witnessed significant advancements in Automatic Speech Recognition (ASR) systems, performance of these systems for individuals with speech disabilities remains inadequate, partly due to limited public training data. To bridge this gap, the 2025 Interspeech Speech Accessibility Project (SAP) Challenge was launched, utilizing over 400 hours of SAP data collected and transcribed from more than 500 individuals with diverse speech disabilities. Hosted on EvalAI and leveraging the remote evaluation pipeline, the SAP Challenge evaluates submissions based on Word Error Rate and Semantic Score. Consequently, 12 out of 22 valid teams outperformed the whisper-large-v2 baseline in terms of WER, while 17 teams surpassed the baseline on SemScore. Notably, the top team achieved the lowest WER of 8.11\%, and the highest SemScore of 88.44\% at the same time, setting new benchmarks for future ASR systems in recognizing impaired speech. 

---
# UserBench: An Interactive Gym Environment for User-Centric Agents 

**Authors**: Cheng Qian, Zuxin Liu, Akshara Prabhakar, Zhiwei Liu, Jianguo Zhang, Haolin Chen, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22034)  

**Abstract**: Large Language Models (LLMs)-based agents have made impressive progress in reasoning and tool use, enabling them to solve complex tasks. However, their ability to proactively collaborate with users, especially when goals are vague, evolving, or indirectly expressed, remains underexplored. To address this gap, we introduce UserBench, a user-centric benchmark designed to evaluate agents in multi-turn, preference-driven interactions. UserBench features simulated users who start with underspecified goals and reveal preferences incrementally, requiring agents to proactively clarify intent and make grounded decisions with tools. Our evaluation of leading open- and closed-source LLMs reveals a significant disconnect between task completion and user alignment. For instance, models provide answers that fully align with all user intents only 20% of the time on average, and even the most advanced models uncover fewer than 30% of all user preferences through active interaction. These results highlight the challenges of building agents that are not just capable task executors, but true collaborative partners. UserBench offers an interactive environment to measure and advance this critical capability. 

---
# UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding 

**Authors**: Shuquan Lian, Yuhang Wu, Jia Ma, Zihan Song, Bingqi Chen, Xiawu Zheng, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22025)  

**Abstract**: The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE, a comprehensive framework enhancing GUI agents at both the training and inference stages. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a Continuous Reward function to incentivize high-precision grounding; 2) a "Simple Thinking" reward to balance planning with speed and grounding accuracy; and 3) a Cropping-based Resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present Decomposed Grounding with Selection, a novel method that dramatically improves grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2. For instance, using both our proposed training and inference enhancement methods brings 23% grounding accuracy improvement over the best baseline on ScreenSpot-Pro. 

---
# PHAX: A Structured Argumentation Framework for User-Centered Explainable AI in Public Health and Biomedical Sciences 

**Authors**: Bahar İlgen, Akshat Dubey, Georges Hattab  

**Link**: [PDF](https://arxiv.org/pdf/2507.22009)  

**Abstract**: Ensuring transparency and trust in AI-driven public health and biomedical sciences systems requires more than accurate predictions-it demands explanations that are clear, contextual, and socially accountable. While explainable AI (XAI) has advanced in areas like feature attribution and model interpretability, most methods still lack the structure and adaptability needed for diverse health stakeholders, including clinicians, policymakers, and the general public. We introduce PHAX-a Public Health Argumentation and eXplainability framework-that leverages structured argumentation to generate human-centered explanations for AI outputs. PHAX is a multi-layer architecture combining defeasible reasoning, adaptive natural language techniques, and user modeling to produce context-aware, audience-specific justifications. More specifically, we show how argumentation enhances explainability by supporting AI-driven decision-making, justifying recommendations, and enabling interactive dialogues across user types. We demonstrate the applicability of PHAX through use cases such as medical term simplification, patient-clinician communication, and policy justification. In particular, we show how simplification decisions can be modeled as argument chains and personalized based on user expertise-enhancing both interpretability and trust. By aligning formal reasoning methods with communicative demands, PHAX contributes to a broader vision of transparent, human-centered AI in public health. 

---
# The Effect of Compression Techniques on Large Multimodal Language Models in the Medical Domain 

**Authors**: Tanvir Ahmed Khan, Aranya Saha, Ismam Nur Swapnil, Mohammad Ariful Haque  

**Link**: [PDF](https://arxiv.org/pdf/2507.21976)  

**Abstract**: Multimodal Large Language Models (MLLMs) hold huge potential for usage in the medical domain, but their computational costs necessitate efficient compression techniques. This paper evaluates the impact of structural pruning and activation-aware quantization on a fine-tuned LLAVA model for medical applications. We propose a novel layer selection method for pruning, analyze different quantization techniques, and assess the performance trade-offs in a prune-SFT-quantize pipeline. Our proposed method enables MLLMs with 7B parameters to run within 4 GB of VRAM, reducing memory usage by 70% while achieving 4% higher model performance compared to traditional pruning and quantization techniques in the same compression ratio. 

---
# Reasoning Language Models for Root Cause Analysis in 5G Wireless Networks 

**Authors**: Mohamed Sana, Nicola Piovesan, Antonio De Domenico, Yibin Kang, Haozhe Zhang, Merouane Debbah, Fadhel Ayed  

**Link**: [PDF](https://arxiv.org/pdf/2507.21974)  

**Abstract**: Root Cause Analysis (RCA) in mobile networks remains a challenging task due to the need for interpretability, domain expertise, and causal reasoning. In this work, we propose a lightweight framework that leverages Large Language Models (LLMs) for RCA. To do so, we introduce TeleLogs, a curated dataset of annotated troubleshooting problems designed to benchmark RCA capabilities. Our evaluation reveals that existing open-source reasoning LLMs struggle with these problems, underscoring the need for domain-specific adaptation. To address this issue, we propose a two-stage training methodology that combines supervised fine-tuning with reinforcement learning to improve the accuracy and reasoning quality of LLMs. The proposed approach fine-tunes a series of RCA models to integrate domain knowledge and generate structured, multi-step diagnostic explanations, improving both interpretability and effectiveness. Extensive experiments across multiple LLM sizes show significant performance gains over state-of-the-art reasoning and non-reasoning models, including strong generalization to randomized test variants. These results demonstrate the promise of domain-adapted, reasoning-enhanced LLMs for practical and explainable RCA in network operation and management. 

---
# Thou Shalt Not Prompt: Zero-Shot Human Activity Recognition in Smart Homes via Language Modeling of Sensor Data & Activities 

**Authors**: Sourish Gunesh Dhekane, Thomas Ploetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.21964)  

**Abstract**: Developing zero-shot human activity recognition (HAR) methods is a critical direction in smart home research -- considering its impact on making HAR systems work across smart homes having diverse sensing modalities, layouts, and activities of interest. The state-of-the-art solutions along this direction are based on generating natural language descriptions of the sensor data and feeding it via a carefully crafted prompt to the LLM to perform classification. Despite their performance guarantees, such ``prompt-the-LLM'' approaches carry several risks, including privacy invasion, reliance on an external service, and inconsistent predictions due to version changes, making a case for alternative zero-shot HAR methods that do not require prompting the LLMs. In this paper, we propose one such solution that models sensor data and activities using natural language, leveraging its embeddings to perform zero-shot classification and thereby bypassing the need to prompt the LLMs for activity predictions. The impact of our work lies in presenting a detailed case study on six datasets, highlighting how language modeling can bolster HAR systems in zero-shot recognition. 

---
# Libra: Large Chinese-based Safeguard for AI Content 

**Authors**: Ziyang Chen, Huimu Yu, Xing Wu, Dongqin Liu, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21929)  

**Abstract**: Large language models (LLMs) excel in text understanding and generation but raise significant safety and ethical concerns in high-stakes applications. To mitigate these risks, we present Libra-Guard, a cutting-edge safeguard system designed to enhance the safety of Chinese-based LLMs. Leveraging a two-stage curriculum training pipeline, Libra-Guard enhances data efficiency by employing guard pretraining on synthetic samples, followed by fine-tuning on high-quality, real-world data, thereby significantly reducing reliance on manual annotations. To enable rigorous safety evaluations, we also introduce Libra-Test, the first benchmark specifically designed to evaluate the effectiveness of safeguard systems for Chinese content. It covers seven critical harm scenarios and includes over 5,700 samples annotated by domain experts. Experiments show that Libra-Guard achieves 86.79% accuracy, outperforming Qwen2.5-14B-Instruct (74.33%) and ShieldLM-Qwen-14B-Chat (65.69%), and nearing closed-source models like Claude-3.5-Sonnet and GPT-4o. These contributions establish a robust framework for advancing the safety governance of Chinese LLMs and represent a tentative step toward developing safer, more reliable Chinese AI systems. 

---
# LLM-based Content Classification Approach for GitHub Repositories by the README Files 

**Authors**: Malik Uzair Mehmood, Shahid Hussain, Wen Li Wang, Muhammad Usama Malik  

**Link**: [PDF](https://arxiv.org/pdf/2507.21899)  

**Abstract**: GitHub is the world's most popular platform for storing, sharing, and managing code. Every GitHub repository has a README file associated with it. The README files should contain project-related information as per the recommendations of GitHub to support the usage and improvement of repositories. However, GitHub repository owners sometimes neglected these recommendations. This prevents a GitHub repository from reaching its full potential. This research posits that the comprehensiveness of a GitHub repository's README file significantly influences its adoption and utilization, with a lack of detail potentially hindering its full potential for widespread engagement and impact within the research community. Large Language Models (LLMs) have shown great performance in many text-based tasks including text classification, text generation, text summarization and text translation. In this study, an approach is developed to fine-tune LLMs for automatically classifying different sections of GitHub README files. Three encoder-only LLMs are utilized, including BERT, DistilBERT and RoBERTa. These pre-trained models are then fine-tuned based on a gold-standard dataset consisting of 4226 README file sections. This approach outperforms current state-of-the-art methods and has achieved an overall F1 score of 0.98. Moreover, we have also investigated the use of Parameter-Efficient Fine-Tuning (PEFT) techniques like Low-Rank Adaptation (LoRA) and shown an economical alternative to full fine-tuning without compromising much performance. The results demonstrate the potential of using LLMs in designing an automatic classifier for categorizing the content of GitHub README files. Consequently, this study contributes to the development of automated tools for GitHub repositories to improve their identifications and potential usages. 

---
# Efficient Pain Recognition via Respiration Signals: A Single Cross-Attention Transformer Multi-Window Fusion Pipeline 

**Authors**: Stefanos Gkikas, Ioannis Kyprakis, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.21886)  

**Abstract**: Pain is a complex condition affecting a large portion of the population. Accurate and consistent evaluation is essential for individuals experiencing pain, and it supports the development of effective and advanced management strategies. Automatic pain assessment systems provide continuous monitoring and support clinical decision-making, aiming to reduce distress and prevent functional decline. This study has been submitted to the \textit{Second Multimodal Sensing Grand Challenge for Next-Gen Pain Assessment (AI4PAIN)}. The proposed method introduces a pipeline that leverages respiration as the input signal and incorporates a highly efficient cross-attention transformer alongside a multi-windowing strategy. Extensive experiments demonstrate that respiration is a valuable physiological modality for pain assessment. Moreover, experiments revealed that compact and efficient models, when properly optimized, can achieve strong performance, often surpassing larger counterparts. The proposed multi-window approach effectively captures both short-term and long-term features, as well as global characteristics, thereby enhancing the model's representational capacity. 

---
# The Impact of Foundational Models on Patient-Centric e-Health Systems 

**Authors**: Elmira Onagh, Alireza Davoodi, Maleknaz Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21882)  

**Abstract**: As Artificial Intelligence (AI) becomes increasingly embedded in healthcare technologies, understanding the maturity of AI in patient-centric applications is critical for evaluating its trustworthiness, transparency, and real-world impact. In this study, we investigate the integration and maturity of AI feature integration in 116 patient-centric healthcare applications. Using Large Language Models (LLMs), we extracted key functional features, which are then categorized into different stages of the Gartner AI maturity model. Our results show that over 86.21\% of applications remain at the early stages of AI integration, while only 13.79% demonstrate advanced AI integration. 

---
# Multi-Representation Diagrams for Pain Recognition: Integrating Various Electrodermal Activity Signals into a Single Image 

**Authors**: Stefanos Gkikas, Ioannis Kyprakis, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.21881)  

**Abstract**: Pain is a multifaceted phenomenon that affects a substantial portion of the population. Reliable and consistent evaluation benefits those experiencing pain and underpins the development of effective and advanced management strategies. Automatic pain-assessment systems deliver continuous monitoring, inform clinical decision-making, and aim to reduce distress while preventing functional decline. By incorporating physiological signals, these systems provide objective, accurate insights into an individual's condition. This study has been submitted to the \textit{Second Multimodal Sensing Grand Challenge for Next-Gen Pain Assessment (AI4PAIN)}. The proposed method introduces a pipeline that leverages electrodermal activity signals as input modality. Multiple representations of the signal are created and visualized as waveforms, and they are jointly visualized within a single multi-representation diagram. Extensive experiments incorporating various processing and filtering techniques, along with multiple representation combinations, demonstrate the effectiveness of the proposed approach. It consistently yields comparable, and in several cases superior, results to traditional fusion methods, establishing it as a robust alternative for integrating different signal representations or modalities. 

---
# Tiny-BioMoE: a Lightweight Embedding Model for Biosignal Analysis 

**Authors**: Stefanos Gkikas, Ioannis Kyprakis, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.21875)  

**Abstract**: Pain is a complex and pervasive condition that affects a significant portion of the population. Accurate and consistent assessment is essential for individuals suffering from pain, as well as for developing effective management strategies in a healthcare system. Automatic pain assessment systems enable continuous monitoring, support clinical decision-making, and help minimize patient distress while mitigating the risk of functional deterioration. Leveraging physiological signals offers objective and precise insights into a person's state, and their integration in a multimodal framework can further enhance system performance. This study has been submitted to the \textit{Second Multimodal Sensing Grand Challenge for Next-Gen Pain Assessment (AI4PAIN)}. The proposed approach introduces \textit{Tiny-BioMoE}, a lightweight pretrained embedding model for biosignal analysis. Trained on $4.4$ million biosignal image representations and consisting of only $7.3$ million parameters, it serves as an effective tool for extracting high-quality embeddings for downstream tasks. Extensive experiments involving electrodermal activity, blood volume pulse, respiratory signals, peripheral oxygen saturation, and their combinations highlight the model's effectiveness across diverse modalities in automatic pain recognition tasks. \textit{\textcolor{blue}{The model's architecture (code) and weights are available at this https URL. 

---
# A Neuro-Symbolic Approach for Probabilistic Reasoning on Graph Data 

**Authors**: Raffaele Pojer, Andrea Passerini, Kim G. Larsen, Manfred Jaeger  

**Link**: [PDF](https://arxiv.org/pdf/2507.21873)  

**Abstract**: Graph neural networks (GNNs) excel at predictive tasks on graph-structured data but often lack the ability to incorporate symbolic domain knowledge and perform general reasoning. Relational Bayesian Networks (RBNs), in contrast, enable fully generative probabilistic modeling over graph-like structures and support rich symbolic knowledge and probabilistic inference. This paper presents a neuro-symbolic framework that seamlessly integrates GNNs into RBNs, combining the learning strength of GNNs with the flexible reasoning capabilities of RBNs.
We develop two implementations of this integration: one compiles GNNs directly into the native RBN language, while the other maintains the GNN as an external component. Both approaches preserve the semantics and computational properties of GNNs while fully aligning with the RBN modeling paradigm. We also propose a maximum a-posteriori (MAP) inference method for these neuro-symbolic models.
To demonstrate the framework's versatility, we apply it to two distinct problems. First, we transform a GNN for node classification into a collective classification model that explicitly models homo- and heterophilic label patterns, substantially improving accuracy. Second, we introduce a multi-objective network optimization problem in environmental planning, where MAP inference supports complex decision-making. Both applications include new publicly available benchmark datasets.
This work introduces a powerful and coherent neuro-symbolic approach to graph data, bridging learning and reasoning in ways that enable novel applications and improved performance across diverse tasks. 

---
# MultiEditor: Controllable Multimodal Object Editing for Driving Scenarios Using 3D Gaussian Splatting Priors 

**Authors**: Shouyi Lu, Zihan Lin, Chao Lu, Huanran Wang, Guirong Zhuo, Lianqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.21872)  

**Abstract**: Autonomous driving systems rely heavily on multimodal perception data to understand complex environments. However, the long-tailed distribution of real-world data hinders generalization, especially for rare but safety-critical vehicle categories. To address this challenge, we propose MultiEditor, a dual-branch latent diffusion framework designed to edit images and LiDAR point clouds in driving scenarios jointly. At the core of our approach is introducing 3D Gaussian Splatting (3DGS) as a structural and appearance prior for target objects. Leveraging this prior, we design a multi-level appearance control mechanism--comprising pixel-level pasting, semantic-level guidance, and multi-branch refinement--to achieve high-fidelity reconstruction across modalities. We further propose a depth-guided deformable cross-modality condition module that adaptively enables mutual guidance between modalities using 3DGS-rendered depth, significantly enhancing cross-modality consistency. Extensive experiments demonstrate that MultiEditor achieves superior performance in visual and geometric fidelity, editing controllability, and cross-modality consistency. Furthermore, generating rare-category vehicle data with MultiEditor substantially enhances the detection accuracy of perception models on underrepresented classes. 

---
# EDGE-GRPO: Entropy-Driven GRPO with Guided Error Correction for Advantage Diversity 

**Authors**: Xingjian Zhang, Siwei Wen, Wenjun Wu, Lei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21848)  

**Abstract**: Large Language Models (LLMs) have made remarkable progress in enhancing step-by-step reasoning through reinforcement learning. However, the Group Relative Policy Optimization (GRPO) algorithm, which relies on sparse reward rules, often encounters the issue of identical rewards within groups, leading to the advantage collapse problem. Existing works typically address this challenge from two perspectives: enforcing model reflection to enhance response diversity, and introducing internal feedback to augment the training signal (advantage). In this work, we begin by analyzing the limitations of model reflection and investigating the policy entropy of responses at the fine-grained sample level. Based on our experimental findings, we propose the EDGE-GRPO algorithm, which adopts \textbf{E}ntropy-\textbf{D}riven Advantage and \textbf{G}uided \textbf{E}rror Correction to effectively mitigate the problem of advantage collapse. Extensive experiments on several main reasoning benchmarks demonstrate the effectiveness and superiority of our approach. It is available at this https URL. 

---
# Probabilistic Active Goal Recognition 

**Authors**: Chenyuan Zhang, Cristian Rojas Cardenas, Hamid Rezatofighi, Mor Vered, Buser Say  

**Link**: [PDF](https://arxiv.org/pdf/2507.21846)  

**Abstract**: In multi-agent environments, effective interaction hinges on understanding the beliefs and intentions of other agents. While prior work on goal recognition has largely treated the observer as a passive reasoner, Active Goal Recognition (AGR) focuses on strategically gathering information to reduce uncertainty. We adopt a probabilistic framework for Active Goal Recognition and propose an integrated solution that combines a joint belief update mechanism with a Monte Carlo Tree Search (MCTS) algorithm, allowing the observer to plan efficiently and infer the actor's hidden goal without requiring domain-specific knowledge. Through comprehensive empirical evaluation in a grid-based domain, we show that our joint belief update significantly outperforms passive goal recognition, and that our domain-independent MCTS performs comparably to our strong domain-specific greedy baseline. These results establish our solution as a practical and robust framework for goal inference, advancing the field toward more interactive and adaptive multi-agent systems. 

---
# DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework 

**Authors**: Kuiye Ding, Fanda Fan, Yao Wang, Ruijie jian, Xiaorui Wang, Luqi Gong, Yishan Jiang, Chunjie Luo an Jianfeng Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21830)  

**Abstract**: Multivariate Time Series Forecasting plays a key role in many applications. Recent works have explored using Large Language Models for MTSF to take advantage of their reasoning abilities. However, many methods treat LLMs as end-to-end forecasters, which often leads to a loss of numerical precision and forces LLMs to handle patterns beyond their intended design. Alternatively, methods that attempt to align textual and time series modalities within latent space frequently encounter alignment difficulty. In this paper, we propose to treat LLMs not as standalone forecasters, but as semantic guidance modules within a dual-stream framework. We propose DualSG, a dual-stream framework that provides explicit semantic guidance, where LLMs act as Semantic Guides to refine rather than replace traditional predictions. As part of DualSG, we introduce Time Series Caption, an explicit prompt format that summarizes trend patterns in natural language and provides interpretable context for LLMs, rather than relying on implicit alignment between text and time series in the latent space. We also design a caption-guided fusion module that explicitly models inter-variable relationships while reducing noise and computation. Experiments on real-world datasets from diverse domains show that DualSG consistently outperforms 15 state-of-the-art baselines, demonstrating the value of explicitly combining numerical forecasting with semantic guidance. 

---
# An Agentic AI for a New Paradigm in Business Process Development 

**Authors**: Mohammad Azarijafari, Luisa Mich, Michele Missikoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.21823)  

**Abstract**: Artificial Intelligence agents represent the next major revolution in the continuous technological evolution of industrial automation. In this paper, we introduce a new approach for business process design and development that leverages the capabilities of Agentic AI. Departing from the traditional task-based approach to business process design, we propose an agent-based method, where agents contribute to the achievement of business goals, identified by a set of business objects. When a single agent cannot fulfill a goal, we have a merge goal that can be achieved through the collaboration of multiple agents. The proposed model leads to a more modular and intelligent business process development by organizing it around goals, objects, and agents. As a result, this approach enables flexible and context-aware automation in dynamic industrial environments. 

---
# MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE 

**Authors**: Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, Zhao Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2507.21802)  

**Abstract**: Although GRPO substantially enhances flow matching models in human preference alignment of image generation, methods such as FlowGRPO still exhibit inefficiency due to the necessity of sampling and optimizing over all denoising steps specified by the Markov Decision Process (MDP). In this paper, we propose $\textbf{MixGRPO}$, a novel framework that leverages the flexibility of mixed sampling strategies through the integration of stochastic differential equations (SDE) and ordinary differential equations (ODE). This streamlines the optimization process within the MDP to improve efficiency and boost performance. Specifically, MixGRPO introduces a sliding window mechanism, using SDE sampling and GRPO-guided optimization only within the window, while applying ODE sampling outside. This design confines sampling randomness to the time-steps within the window, thereby reducing the optimization overhead, and allowing for more focused gradient updates to accelerate convergence. Additionally, as time-steps beyond the sliding window are not involved in optimization, higher-order solvers are supported for sampling. So we present a faster variant, termed $\textbf{MixGRPO-Flash}$, which further improves training efficiency while achieving comparable performance. MixGRPO exhibits substantial gains across multiple dimensions of human preference alignment, outperforming DanceGRPO in both effectiveness and efficiency, with nearly 50% lower training time. Notably, MixGRPO-Flash further reduces training time by 71%. Codes and models are available at $\href{this https URL}{MixGRPO}$. 

---
# Hybrid Causal Identification and Causal Mechanism Clustering 

**Authors**: Saixiong Liu, Yuhua Qian, Jue Li, Honghong Cheng, Feijiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.21792)  

**Abstract**: Bivariate causal direction identification is a fundamental and vital problem in the causal inference field. Among binary causal methods, most methods based on additive noise only use one single causal mechanism to construct a causal model. In the real world, observations are always collected in different environments with heterogeneous causal relationships. Therefore, on observation data, this paper proposes a Mixture Conditional Variational Causal Inference model (MCVCI) to infer heterogeneous causality. Specifically, according to the identifiability of the Hybrid Additive Noise Model (HANM), MCVCI combines the superior fitting capabilities of the Gaussian mixture model and the neural network and elegantly uses the likelihoods obtained from the probabilistic bounds of the mixture conditional variational auto-encoder as causal decision criteria. Moreover, we model the casual heterogeneity into cluster numbers and propose the Mixture Conditional Variational Causal Clustering (MCVCC) method, which can reveal causal mechanism expression. Compared with state-of-the-art methods, the comprehensive best performance demonstrates the effectiveness of the methods proposed in this paper on several simulated and real data. 

---
# Towards a rigorous evaluation of RAG systems: the challenge of due diligence 

**Authors**: Grégoire Martinon, Alexandra Lorenzo de Brionne, Jérôme Bohard, Antoine Lojou, Damien Hervault, Nicolas J-B. Brunel  

**Link**: [PDF](https://arxiv.org/pdf/2507.21753)  

**Abstract**: The rise of generative AI, has driven significant advancements in high-risk sectors like healthcare and finance. The Retrieval-Augmented Generation (RAG) architecture, combining language models (LLMs) with search engines, is particularly notable for its ability to generate responses from document corpora. Despite its potential, the reliability of RAG systems in critical contexts remains a concern, with issues such as hallucinations persisting. This study evaluates a RAG system used in due diligence for an investment fund. We propose a robust evaluation protocol combining human annotations and LLM-Judge annotations to identify system failures, like hallucinations, off-topic, failed citations, and abstentions. Inspired by the Prediction Powered Inference (PPI) method, we achieve precise performance measurements with statistical guarantees. We provide a comprehensive dataset for further analysis. Our contributions aim to enhance the reliability and scalability of RAG systems evaluation protocols in industrial applications. 

---
# SAT-Based Bounded Fitting for the Description Logic ALC 

**Authors**: Maurice Funk, Jean Christoph Jung, Tom Voellmer  

**Link**: [PDF](https://arxiv.org/pdf/2507.21752)  

**Abstract**: Bounded fitting is a general paradigm for learning logical formulas from positive and negative data examples, that has received considerable interest recently. We investigate bounded fitting for the description logic ALC and its syntactic fragments. We show that the underlying size-restricted fitting problem is NP-complete for all studied fragments, even in the special case of a single positive and a single negative example. By design, bounded fitting comes with probabilistic guarantees in Valiant's PAC learning framework. In contrast, we show that other classes of algorithms for learning ALC concepts do not provide such guarantees. Finally, we present an implementation of bounded fitting in ALC and its fragments based on a SAT solver. We discuss optimizations and compare our implementation to other concept learning tools. 

---
# GDAIP: A Graph-Based Domain Adaptive Framework for Individual Brain Parcellation 

**Authors**: Jianfei Zhu, Haiqi Zhu, Shaohui Liu, Feng Jiang, Baichun Wei, Chunzhi Yi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21727)  

**Abstract**: Recent deep learning approaches have shown promise in learning such individual brain parcellations from functional magnetic resonance imaging (fMRI). However, most existing methods assume consistent data distributions across domains and struggle with domain shifts inherent to real-world cross-dataset scenarios. To address this challenge, we proposed Graph Domain Adaptation for Individual Parcellation (GDAIP), a novel framework that integrates Graph Attention Networks (GAT) with Minimax Entropy (MME)-based domain adaptation. We construct cross-dataset brain graphs at both the group and individual levels. By leveraging semi-supervised training and adversarial optimization of the prediction entropy on unlabeled vertices from target brain graph, the reference atlas is adapted from the group-level brain graph to the individual brain graph, enabling individual parcellation under cross-dataset settings. We evaluated our method using parcellation visualization, Dice coefficient, and functional homogeneity. Experimental results demonstrate that GDAIP produces individual parcellations with topologically plausible boundaries, strong cross-session consistency, and ability of reflecting functional organization. 

---
# Unrolling Dynamic Programming via Graph Filters 

**Authors**: Sergio Rozada, Samuel Rey, Gonzalo Mateos, Antonio G. Marques  

**Link**: [PDF](https://arxiv.org/pdf/2507.21705)  

**Abstract**: Dynamic programming (DP) is a fundamental tool used across many engineering fields. The main goal of DP is to solve Bellman's optimality equations for a given Markov decision process (MDP). Standard methods like policy iteration exploit the fixed-point nature of these equations to solve them iteratively. However, these algorithms can be computationally expensive when the state-action space is large or when the problem involves long-term dependencies. Here we propose a new approach that unrolls and truncates policy iterations into a learnable parametric model dubbed BellNet, which we train to minimize the so-termed Bellman error from random value function initializations. Viewing the transition probability matrix of the MDP as the adjacency of a weighted directed graph, we draw insights from graph signal processing to interpret (and compactly re-parameterize) BellNet as a cascade of nonlinear graph filters. This fresh look facilitates a concise, transferable, and unifying representation of policy and value iteration, with an explicit handle on complexity during inference. Preliminary experiments conducted in a grid-like environment demonstrate that BellNet can effectively approximate optimal policies in a fraction of the iterations required by classical methods. 

---
# Can the current trends of AI handle a full course of mathematics? 

**Authors**: Mariam Alsayyad, Fayadh Kadhem  

**Link**: [PDF](https://arxiv.org/pdf/2507.21664)  

**Abstract**: This paper addresses the question of how able the current trends of Artificial Intelligence (AI) are in managing to take the responsibility of a full course of mathematics at a college level. The study evaluates this ability in four significant aspects, namely, creating a course syllabus, presenting selected material, answering student questions, and creating an assessment. It shows that even though the AI is strong in some important parts like organization and accuracy, there are still some human aspects that are far away from the current abilities of AI. There is still a hidden emotional part, even in science, that cannot be fulfilled by the AI in its current state. This paper suggests some recommendations to integrate the human and AI potentials to create better outcomes in terms of reaching the target of creating a full course of mathematics, at a university level, as best as possible. 

---
# Assistax: A Hardware-Accelerated Reinforcement Learning Benchmark for Assistive Robotics 

**Authors**: Leonard Hinckeldey, Elliot Fosong, Elle Miller, Rimvydas Rubavicius, Trevor McInroe, Patricia Wollstadt, Christiane B. Wiebel-Herboth, Subramanian Ramamoorthy, Stefano V. Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2507.21638)  

**Abstract**: The development of reinforcement learning (RL) algorithms has been largely driven by ambitious challenge tasks and benchmarks. Games have dominated RL benchmarks because they present relevant challenges, are inexpensive to run and easy to understand. While games such as Go and Atari have led to many breakthroughs, they often do not directly translate to real-world embodied applications. In recognising the need to diversify RL benchmarks and addressing complexities that arise in embodied interaction scenarios, we introduce Assistax: an open-source benchmark designed to address challenges arising in assistive robotics tasks. Assistax uses JAX's hardware acceleration for significant speed-ups for learning in physics-based simulations. In terms of open-loop wall-clock time, Assistax runs up to $370\times$ faster when vectorising training runs compared to CPU-based alternatives. Assistax conceptualises the interaction between an assistive robot and an active human patient using multi-agent RL to train a population of diverse partner agents against which an embodied robotic agent's zero-shot coordination capabilities can be tested. Extensive evaluation and hyperparameter tuning for popular continuous control RL and MARL algorithms provide reliable baselines and establish Assistax as a practical benchmark for advancing RL research for assistive robotics. The code is available at: this https URL. 

---
# Self-Aware Safety Augmentation: Leveraging Internal Semantic Understanding to Enhance Safety in Vision-Language Models 

**Authors**: Wanying Wang, Zeyu Ma, Han Zheng, Xin Tan, Mingang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21637)  

**Abstract**: Large vision-language models (LVLMs) are vulnerable to harmful input compared to their language-only backbones. We investigated this vulnerability by exploring LVLMs internal dynamics, framing their inherent safety understanding in terms of three key capabilities. Specifically, we define these capabilities as safety perception, semantic understanding, and alignment for linguistic expression, and experimentally pinpointed their primary locations within the model architecture. The results indicate that safety perception often emerges before comprehensive semantic understanding, leading to the reduction in safety. Motivated by these findings, we propose \textbf{Self-Aware Safety Augmentation (SASA)}, a technique that projects informative semantic representations from intermediate layers onto earlier safety-oriented layers. This approach leverages the model's inherent semantic understanding to enhance safety recognition without fine-tuning. Then, we employ linear probing to articulate the model's internal semantic comprehension to detect the risk before the generation process. Extensive experiments on various datasets and tasks demonstrate that SASA significantly improves the safety of LVLMs, with minimal impact on the utility. 

---
# StaffPro: an LLM Agent for Joint Staffing and Profiling 

**Authors**: Alessio Maritan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21636)  

**Abstract**: Large language model (LLM) agents integrate pre-trained LLMs with modular algorithmic components and have shown remarkable reasoning and decision-making abilities. In this work, we investigate their use for two tightly intertwined challenges in workforce management: staffing, i.e., the assignment and scheduling of tasks to workers, which may require team formation; and profiling, i.e., the continuous estimation of workers' skills, preferences, and other latent attributes from unstructured data. We cast these problems in a formal mathematical framework that links scheduling decisions to latent feature estimation, and we introduce StaffPro, an LLM agent that addresses staffing and profiling jointly. Differently from existing staffing solutions, StaffPro allows expressing optimization objectives using natural language, accepts textual task descriptions and provides high flexibility. StaffPro interacts directly with humans by establishing a continuous human-agent feedback loop, ensuring natural and intuitive use. By analyzing human feedback, our agent continuously estimates the latent features of workers, realizing life-long worker profiling and ensuring optimal staffing performance over time. A consulting firm simulation example demonstrates that StaffPro successfully estimates workers' attributes and generates high quality schedules. With its innovative design, StaffPro offers a robust, interpretable, and human-centric solution for automated personnel management. 

---
# "Teammates, Am I Clear?": Analysing Legible Behaviours in Teams 

**Authors**: Miguel Faria, Francisco S. Melo, Ana Paiva  

**Link**: [PDF](https://arxiv.org/pdf/2507.21631)  

**Abstract**: In this paper we investigate the notion of legibility in sequential decision-making in the context of teams and teamwork. There have been works that extend the notion of legibility to sequential decision making, for deterministic and for stochastic scenarios. However, these works focus on one agent interacting with one human, foregoing the benefits of having legible decision making in teams of agents or in team configurations with humans. In this work we propose an extension of legible decision-making to multi-agent settings that improves the performance of agents working in collaboration. We showcase the performance of legible decision making in team scenarios using our proposed extension in multi-agent benchmark scenarios. We show that a team with a legible agent is able to outperform a team composed solely of agents with standard optimal behaviour. 

---
# Exploring the Link Between Bayesian Inference and Embodied Intelligence: Toward Open Physical-World Embodied AI Systems 

**Authors**: Bin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21589)  

**Abstract**: Embodied intelligence posits that cognitive capabilities fundamentally emerge from - and are shaped by - an agent's real-time sensorimotor interactions with its environment. Such adaptive behavior inherently requires continuous inference under uncertainty. Bayesian statistics offers a principled probabilistic framework to address this challenge by representing knowledge as probability distributions and updating beliefs in response to new evidence. The core computational processes underlying embodied intelligence - including perception, action selection, learning, and even higher-level cognition - can be effectively understood and modeled as forms of Bayesian inference. Despite the deep conceptual connection between Bayesian statistics and embodied intelligence, Bayesian principles have not been widely or explicitly applied in today's embodied intelligence systems. In this work, we examine both Bayesian and contemporary embodied intelligence approaches through two fundamental lenses: search and learning - the two central themes in modern AI, as highlighted in Rich Sutton's influential essay "The Bitter Lesson". This analysis sheds light on why Bayesian inference has not played a central role in the development of modern embodied intelligence. At the same time, it reveals that current embodied intelligence systems remain largely confined to closed-physical-world environments, and highlights the potential for Bayesian methods to play a key role in extending these systems toward truly open physical-world embodied intelligence. 

---
# Progressive Homeostatic and Plastic Prompt Tuning for Audio-Visual Multi-Task Incremental Learning 

**Authors**: Jiong Yin, Liang Li, Jiehua Zhang, Yuhan Gao, Chenggang Yan, Xichun Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.21588)  

**Abstract**: Audio-visual multi-task incremental learning aims to continuously learn from multiple audio-visual tasks without the need for joint training on all tasks. The challenge of the problem is how to preserve the old task knowledge while facilitating the learning of new task with previous experiences. To address these challenges, we introduce a three-stage Progressive Homeostatic and Plastic audio-visual prompt (PHP) method. In the shallow phase, we design the task-shared modality aggregating adapter to foster cross-task and cross-modal audio-visual representation learning to enhance shared understanding between tasks. In the middle phase, we propose the task-specific modality-shared dynamic generating adapter, which constructs prompts that are tailored to individual tasks while remaining general across modalities, which balances the models ability to retain knowledge against forgetting with its potential for versatile multi-task transferability. In the deep phase, we introduce the task-specific modality-independent prompts to further refine the understand ability by targeting individual information for each task and modality. By incorporating these three phases, PHP retains task-specific prompts while adapting shared parameters for new tasks to effectively balance knowledge sharing and specificity. Our method achieves SOTA performance in different orders of four tasks (AVE, AVVP, AVS and AVQA). Our code can be available at this https URL. 

---
# SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-based Retrieval-Augmented Generation 

**Authors**: Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, Huadong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.21585)  

**Abstract**: In this work, we study how vision-language models (VLMs) can be utilized to enhance the safety for the autonomous driving system, including perception, situational understanding, and path planning. However, existing research has largely overlooked the evaluation of these models in traffic safety-critical driving scenarios. To bridge this gap, we create the benchmark (SafeDrive228K) and propose a new baseline based on VLM with knowledge graph-based retrieval-augmented generation (SafeDriveRAG) for visual question answering (VQA). Specifically, we introduce SafeDrive228K, the first large-scale multimodal question-answering benchmark comprising 228K examples across 18 sub-tasks. This benchmark encompasses a diverse range of traffic safety queries, from traffic accidents and corner cases to common safety knowledge, enabling a thorough assessment of the comprehension and reasoning abilities of the models. Furthermore, we propose a plug-and-play multimodal knowledge graph-based retrieval-augmented generation approach that employs a novel multi-scale subgraph retrieval algorithm for efficient information retrieval. By incorporating traffic safety guidelines collected from the Internet, this framework further enhances the model's capacity to handle safety-critical situations. Finally, we conduct comprehensive evaluations on five mainstream VLMs to assess their reliability in safety-sensitive driving tasks. Experimental results demonstrate that integrating RAG significantly improves performance, achieving a +4.73% gain in Traffic Accidents tasks, +8.79% in Corner Cases tasks and +14.57% in Traffic Safety Commonsense across five mainstream VLMs, underscoring the potential of our proposed benchmark and methodology for advancing research in traffic safety. Our source code and data are available at this https URL. 

---
# Finding Uncommon Ground: A Human-Centered Model for Extrospective Explanations 

**Authors**: Laura Spillner, Nima Zargham, Mihai Pomarlan, Robert Porzel, Rainer Malaka  

**Link**: [PDF](https://arxiv.org/pdf/2507.21571)  

**Abstract**: The need for explanations in AI has, by and large, been driven by the desire to increase the transparency of black-box machine learning models. However, such explanations, which focus on the internal mechanisms that lead to a specific output, are often unsuitable for non-experts. To facilitate a human-centered perspective on AI explanations, agents need to focus on individuals and their preferences as well as the context in which the explanations are given. This paper proposes a personalized approach to explanation, where the agent tailors the information provided to the user based on what is most likely pertinent to them. We propose a model of the agent's worldview that also serves as a personal and dynamic memory of its previous interactions with the same user, based on which the artificial agent can estimate what part of its knowledge is most likely new information to the user. 

---
# Large Language Models for Wireless Communications: From Adaptation to Autonomy 

**Authors**: Le Liang, Hao Ye, Yucheng Sheng, Ouya Wang, Jiacheng Wang, Shi Jin, Geoffrey Ye Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.21524)  

**Abstract**: The emergence of large language models (LLMs) has revolutionized artificial intelligence, offering unprecedented capabilities in reasoning, generalization, and zero-shot learning. These strengths open new frontiers in wireless communications, where increasing complexity and dynamics demand intelligent and adaptive solutions. This article explores the role of LLMs in transforming wireless systems across three key directions: adapting pretrained LLMs for core communication tasks, developing wireless-specific foundation models to balance versatility and efficiency, and enabling agentic LLMs with autonomous reasoning and coordination capabilities. We highlight recent advances, practical case studies, and the unique benefits of LLM-based approaches over traditional methods. Finally, we outline open challenges and research opportunities, including multimodal fusion, collaboration with lightweight models, and self-improving capabilities, charting a path toward intelligent, adaptive, and autonomous wireless networks of the future. 

---
# ST-GDance: Long-Term and Collision-Free Group Choreography from Music 

**Authors**: Jing Xu, Weiqiang Wang, Cunjian Chen, Jun Liu, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2507.21518)  

**Abstract**: Group dance generation from music has broad applications in film, gaming, and animation production. However, it requires synchronizing multiple dancers while maintaining spatial coordination. As the number of dancers and sequence length increase, this task faces higher computational complexity and a greater risk of motion collisions. Existing methods often struggle to model dense spatial-temporal interactions, leading to scalability issues and multi-dancer collisions. To address these challenges, we propose ST-GDance, a novel framework that decouples spatial and temporal dependencies to optimize long-term and collision-free group choreography. We employ lightweight graph convolutions for distance-aware spatial modeling and accelerated sparse attention for efficient temporal modeling. This design significantly reduces computational costs while ensuring smooth and collision-free interactions. Experiments on the AIOZ-GDance dataset demonstrate that ST-GDance outperforms state-of-the-art baselines, particularly in generating long and coherent group dance sequences. Project page: this https URL. 

---
# What Does it Mean for a Neural Network to Learn a "World Model"? 

**Authors**: Kenneth Li, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.21513)  

**Abstract**: We propose a set of precise criteria for saying a neural net learns and uses a "world model." The goal is to give an operational meaning to terms that are often used informally, in order to provide a common language for experimental investigation. We focus specifically on the idea of representing a latent "state space" of the world, leaving modeling the effect of actions to future work. Our definition is based on ideas from the linear probing literature, and formalizes the notion of a computation that factors through a representation of the data generation process. An essential addition to the definition is a set of conditions to check that such a "world model" is not a trivial consequence of the neural net's data or task. 

---
# MoHoBench: Assessing Honesty of Multimodal Large Language Models via Unanswerable Visual Questions 

**Authors**: Yanxu Zhu, Shitong Duan, Xiangxu Zhang, Jitao Sang, Peng Zhang, Tun Lu, Xiao Zhou, Jing Yao, Xiaoyuan Yi, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.21503)  

**Abstract**: Recently Multimodal Large Language Models (MLLMs) have achieved considerable advancements in vision-language tasks, yet produce potentially harmful or untrustworthy content. Despite substantial work investigating the trustworthiness of language models, MMLMs' capability to act honestly, especially when faced with visually unanswerable questions, remains largely underexplored. This work presents the first systematic assessment of honesty behaviors across various MLLMs. We ground honesty in models' response behaviors to unanswerable visual questions, define four representative types of such questions, and construct MoHoBench, a large-scale MMLM honest benchmark, consisting of 12k+ visual question samples, whose quality is guaranteed by multi-stage filtering and human verification. Using MoHoBench, we benchmarked the honesty of 28 popular MMLMs and conducted a comprehensive analysis. Our findings show that: (1) most models fail to appropriately refuse to answer when necessary, and (2) MMLMs' honesty is not solely a language modeling issue, but is deeply influenced by visual information, necessitating the development of dedicated methods for multimodal honesty alignment. Therefore, we implemented initial alignment methods using supervised and preference learning to improve honesty behavior, providing a foundation for future work on trustworthy MLLMs. Our data and code can be found at this https URL. 

---
# Large Language Models for Supply Chain Decisions 

**Authors**: David Simchi-Levi, Konstantina Mellou, Ishai Menache, Jeevan Pathuri  

**Link**: [PDF](https://arxiv.org/pdf/2507.21502)  

**Abstract**: Supply Chain Management requires addressing a variety of complex decision-making challenges, from sourcing strategies to planning and execution. Over the last few decades, advances in computation and information technologies have enabled the transition from manual, intuition and experience-based decision-making, into more automated and data-driven decisions using a variety of tools that apply optimization techniques. These techniques use mathematical methods to improve decision-making.
Unfortunately, business planners and executives still need to spend considerable time and effort to (i) understand and explain the recommendations coming out of these technologies; (ii) analyze various scenarios and answer what-if questions; and (iii) update the mathematical models used in these tools to reflect current business environments. Addressing these challenges requires involving data science teams and/or the technology providers to explain results or make the necessary changes in the technology and hence significantly slows down decision making.
Motivated by the recent advances in Large Language Models (LLMs), we report how this disruptive technology can democratize supply chain technology - namely, facilitate the understanding of tools' outcomes, as well as the interaction with supply chain tools without human-in-the-loop. Specifically, we report how we apply LLMs to address the three challenges described above, thus substantially reducing the time to decision from days and weeks to minutes and hours as well as dramatically increasing planners' and executives' productivity and impact. 

---
# Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess 

**Authors**: Zhenwei Tang, Difan Jiao, Eric Xue, Reid McIlroy-Young, Jon Kleinberg, Siddhartha Sen, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2507.21488)  

**Abstract**: As humans seek to collaborate with, learn from, and better understand artificial intelligence systems, developing AIs that can accurately emulate individual decision-making becomes increasingly important. Chess, a long-standing AI benchmark with precise skill measurement, offers an ideal testbed for human-AI alignment. However, existing approaches to modeling human behavior require prohibitively large amounts of data from each individual, making them impractical for new or sparsely represented users. In this work, we introduce Maia4All, a framework designed to learn and adapt to individual decision-making styles efficiently, even with limited data. Maia4All achieves this through a two-stage optimization process: (1) an enrichment step, which bridges population and individual-level human behavior modeling with a prototype-enriched model, and (2) a democratization step, which leverages ability levels or user prototypes to initialize and refine individual embeddings with minimal data. Our experimental results show that Maia4All can accurately predict individual moves and profile behavioral patterns with high fidelity, establishing a new standard for personalized human-like AI behavior modeling in chess. Maia4All achieves individual human behavior modeling in chess with only 20 games, compared to the 5,000 games required previously, representing a significant improvement in data efficiency. Our work provides an example of how population AI systems can flexibly adapt to individual users using a prototype-enriched model as a bridge. This approach extends beyond chess, as shown in our case study on idiosyncratic LLMs, highlighting its potential for broader applications in personalized AI adaptation. 

---
# An LLM Driven Agent Framework for Automated Infrared Spectral Multi Task Reasoning 

**Authors**: Zujie Xie, Zixuan Chen, Jiheng Liang, Xiangyang Yu, Ziru Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21471)  

**Abstract**: Infrared spectroscopy offers rapid, non destructive measurement of chemical and material properties but suffers from high dimensional, overlapping spectral bands that challenge conventional chemometric approaches. Emerging large language models (LLMs), with their capacity for generalization and reasoning, offer promising potential for automating complex scientific workflows. Despite this promise, their application in IR spectral analysis remains largely unexplored. This study addresses the critical challenge of achieving accurate, automated infrared spectral interpretation under low-data conditions using an LLM-driven framework. We introduce an end-to-end, large language model driven agent framework that integrates a structured literature knowledge base, automated spectral preprocessing, feature extraction, and multi task reasoning in a unified pipeline. By querying a curated corpus of peer reviewed IR publications, the agent selects scientifically validated routines. The selected methods transform each spectrum into low dimensional feature sets, which are fed into few shot prompt templates for classification, regression, and anomaly detection. A closed loop, multi turn protocol iteratively appends mispredicted samples to the prompt, enabling dynamic refinement of predictions. Across diverse materials: stamp pad ink, Chinese medicine, Pu'er tea, Citri Reticulatae Pericarpium and waste water COD datasets, the multi turn LLM consistently outperforms single turn inference, rivaling or exceeding machine learning and deep learning models under low data regimes. 

---
# Validating Pharmacogenomics Generative Artificial Intelligence Query Prompts Using Retrieval-Augmented Generation (RAG) 

**Authors**: Ashley Rector, Keaton Minor, Kamden Minor, Jeff McCormack, Beth Breeden, Ryan Nowers, Jay Dorris  

**Link**: [PDF](https://arxiv.org/pdf/2507.21453)  

**Abstract**: This study evaluated Sherpa Rx, an artificial intelligence tool leveraging large language models and retrieval-augmented generation (RAG) for pharmacogenomics, to validate its performance on key response metrics. Sherpa Rx integrated Clinical Pharmacogenetics Implementation Consortium (CPIC) guidelines with Pharmacogenomics Knowledgebase (PharmGKB) data to generate contextually relevant responses. A dataset (N=260 queries) spanning 26 CPIC guidelines was used to evaluate drug-gene interactions, dosing recommendations, and therapeutic implications. In Phase 1, only CPIC data was embedded. Phase 2 additionally incorporated PharmGKB content. Responses were scored on accuracy, relevance, clarity, completeness (5-point Likert scale), and recall. Wilcoxon signed-rank tests compared accuracy between Phase 1 and Phase 2, and between Phase 2 and ChatGPT-4omini. A 20-question quiz assessed the tool's real-world applicability against other models. In Phase 1 (N=260), Sherpa Rx demonstrated high performance of accuracy 4.9, relevance 5.0, clarity 5.0, completeness 4.8, and recall 0.99. The subset analysis (N=20) showed improvements in accuracy (4.6 vs. 4.4, Phase 2 vs. Phase 1 subset) and completeness (5.0 vs. 4.8). ChatGPT-4omini performed comparably in relevance (5.0) and clarity (4.9) but lagged in accuracy (3.9) and completeness (4.2). Differences in accuracy between Phase 1 and Phase 2 was not statistically significant. However, Phase 2 significantly outperformed ChatGPT-4omini. On the 20-question quiz, Sherpa Rx achieved 90% accuracy, outperforming other models. Integrating additional resources like CPIC and PharmGKB with RAG enhances AI accuracy and performance. This study highlights the transformative potential of generative AI like Sherpa Rx in pharmacogenomics, improving decision-making with accurate, personalized responses. 

---
# Evo-DKD: Dual-Knowledge Decoding for Autonomous Ontology Evolution in Large Language Models 

**Authors**: Vishal Raman, Vijai Aravindh R  

**Link**: [PDF](https://arxiv.org/pdf/2507.21438)  

**Abstract**: Ontologies and knowledge graphs require continuous evolution to remain comprehensive and accurate, but manual curation is labor intensive. Large Language Models (LLMs) possess vast unstructured knowledge but struggle with maintaining structured consistency. We propose Evo-DKD, a novel dual-decoder framework for autonomous ontology evolution that combines structured ontology traversal with unstructured text reasoning. Evo-DKD introduces two parallel decoding streams within an LLM: one decoder generates candidate ontology edits (e.g., new concepts or relations) while the other produces natural-language justifications. A dynamic attention-based gating mechanism coordinates the two streams, deciding at each step how to blend structured and unstructured knowledge. Due to GPU constraints, we simulate the dual-decoder behavior using prompt-based mode control to approximate coordinated decoding in a single-stream mode. The system operates in a closed reasoning loop: proposed ontology edits are validated (via consistency checks and cross-verification with the text explanations) and then injected into the knowledge base, which in turn informs subsequent reasoning. We demonstrate Evo-DKD's effectiveness on use cases including healthcare ontology refinement, semantic search improvement, and cultural heritage timeline modeling. Experiments show that Evo-DKD outperforms baselines using structured-only or unstructured-only decoding in both precision of ontology updates and downstream task performance. We present quantitative metrics and qualitative examples, confirming the contributions of the dual-decoder design and gating router. Evo-DKD offers a new paradigm for LLM-driven knowledge base maintenance, combining the strengths of symbolic and neural reasoning for sustainable ontology evolution. 

---
# GovRelBench:A Benchmark for Government Domain Relevance 

**Authors**: Haiquan Wang, Yi Chen, Shang Zeng, Yun Bian, Zhe Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.21419)  

**Abstract**: Current evaluations of LLMs in the government domain primarily focus on safety considerations in specific scenarios, while the assessment of the models' own core capabilities, particularly domain relevance, remains insufficient. To address this gap, we propose GovRelBench, a benchmark specifically designed for evaluating the core capabilities of LLMs in the government domain. GovRelBench consists of government domain prompts and a dedicated evaluation tool, GovRelBERT. During the training process of GovRelBERT, we introduce the SoftGovScore method: this method trains a model based on the ModernBERT architecture by converting hard labels to soft scores, enabling it to accurately compute the text's government domain relevance score. This work aims to enhance the capability evaluation framework for large models in the government domain, providing an effective tool for relevant research and practice. Our code and dataset are available at this https URL. 

---
# Graph-Augmented Large Language Model Agents: Current Progress and Future Prospects 

**Authors**: Yixin Liu, Guibin Zhang, Kun Wang, Shiyuan Li, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21407)  

**Abstract**: Autonomous agents based on large language models (LLMs) have demonstrated impressive capabilities in a wide range of applications, including web navigation, software development, and embodied control. While most LLMs are limited in several key agentic procedures, such as reliable planning, long-term memory, tool management, and multi-agent coordination, graphs can serve as a powerful auxiliary structure to enhance structure, continuity, and coordination in complex agent workflows. Given the rapid growth and fragmentation of research on Graph-augmented LLM Agents (GLA), this paper offers a timely and comprehensive overview of recent advances and also highlights key directions for future work. Specifically, we categorize existing GLA methods by their primary functions in LLM agent systems, including planning, memory, and tool usage, and then analyze how graphs and graph learning algorithms contribute to each. For multi-agent systems, we further discuss how GLA solutions facilitate the orchestration, efficiency optimization, and trustworthiness of MAS. Finally, we highlight key future directions to advance this field, from improving structural adaptability to enabling unified, scalable, and multimodal GLA systems. We hope this paper can serve as a roadmap for future research on GLA and foster a deeper understanding of the role of graphs in LLM agent systems. 

---
# Shapley Uncertainty in Natural Language Generation 

**Authors**: Meilin Zhu, Gaojie Jin, Xiaowei Huang, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21406)  

**Abstract**: In question-answering tasks, determining when to trust the outputs is crucial to the alignment of large language models (LLMs). Kuhn et al. (2023) introduces semantic entropy as a measure of uncertainty, by incorporating linguistic invariances from the same meaning. It primarily relies on setting threshold to measure the level of semantic equivalence relation. We propose a more nuanced framework that extends beyond such thresholding by developing a Shapley-based uncertainty metric that captures the continuous nature of semantic relationships. We establish three fundamental properties that characterize valid uncertainty metrics and prove that our Shapley uncertainty satisfies these criteria. Through extensive experiments, we demonstrate that our Shapley uncertainty more accurately predicts LLM performance in question-answering and other datasets, compared to similar baseline measures. 

---
# Teaching Language Models To Gather Information Proactively 

**Authors**: Tenghao Huang, Sihao Chen, Muhao Chen, Jonathan May, Longqi Yang, Mengting Wan, Pei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.21389)  

**Abstract**: Large language models (LLMs) are increasingly expected to function as collaborative partners, engaging in back-and-forth dialogue to solve complex, ambiguous problems. However, current LLMs often falter in real-world settings, defaulting to passive responses or narrow clarifications when faced with incomplete or under-specified prompts, falling short of proactively gathering the missing information that is crucial for high-quality solutions. In this work, we introduce a new task paradigm: proactive information gathering, where LLMs must identify gaps in the provided context and strategically elicit implicit user knowledge through targeted questions. To systematically study and train this capability, we design a scalable framework that generates partially specified, real-world tasks, masking key information and simulating authentic ambiguity. Within this setup, our core innovation is a reinforcement finetuning strategy that rewards questions that elicit genuinely new, implicit user information -- such as hidden domain expertise or fine-grained requirements -- that would otherwise remain unspoken. Experiments demonstrate that our trained Qwen-2.5-7B model significantly outperforms o3-mini by 18% on automatic evaluation metrics. More importantly, human evaluation reveals that clarification questions and final outlines generated by our model are favored by human annotators by 42% and 28% respectively. Together, these results highlight the value of proactive clarification in elevating LLMs from passive text generators to genuinely collaborative thought partners. 

---
# Optimizing Multi-Tier Supply Chain Ordering with LNN+XGBoost: Mitigating the Bullwhip Effect 

**Authors**: Chunan Tong  

**Link**: [PDF](https://arxiv.org/pdf/2507.21383)  

**Abstract**: Supply chain management faces significant challenges, including demand fluctuations, inventory imbalances, and amplified upstream order variability due to the bullwhip effect. Traditional methods, such as simple moving averages, struggle to address dynamic market conditions. Emerging machine learning techniques, including LSTM, reinforcement learning, and XGBoost, offer potential solutions but are limited by computational complexity, training inefficiencies, or constraints in time-series modeling. Liquid Neural Networks, inspired by dynamic biological systems, present a promising alternative due to their adaptability, low computational cost, and robustness to noise, making them suitable for real-time decision-making and edge computing. Despite their success in applications like autonomous vehicles and medical monitoring, their potential in supply chain optimization remains underexplored. This study introduces a hybrid LNN and XGBoost model to optimize ordering strategies in multi-tier supply chains. By leveraging LNN's dynamic feature extraction and XGBoost's global optimization capabilities, the model aims to mitigate the bullwhip effect and enhance cumulative profitability. The research investigates how local and global synergies within the hybrid framework address the dual demands of adaptability and efficiency in SCM. The proposed approach fills a critical gap in existing methodologies, offering an innovative solution for dynamic and efficient supply chain management. 

---
# Efficacy of AI RAG Tools for Complex Information Extraction and Data Annotation Tasks: A Case Study Using Banks Public Disclosures 

**Authors**: Nicholas Botti, Flora Haberkorn, Charlotte Hoopes, Shaun Khan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21360)  

**Abstract**: We utilize a within-subjects design with randomized task assignments to understand the effectiveness of using an AI retrieval augmented generation (RAG) tool to assist analysts with an information extraction and data annotation task. We replicate an existing, challenging real-world annotation task with complex multi-part criteria on a set of thousands of pages of public disclosure documents from global systemically important banks (GSIBs) with heterogeneous and incomplete information content. We test two treatment conditions. First, a "naive" AI use condition in which annotators use only the tool and must accept the first answer they are given. And second, an "interactive" AI treatment condition where annotators use the tool interactively, and use their judgement to follow-up with additional information if necessary. Compared to the human-only baseline, the use of the AI tool accelerated task execution by up to a factor of 10 and enhanced task accuracy, particularly in the interactive condition. We find that when extrapolated to the full task, these methods could save up to 268 hours compared to the human-only approach. Additionally, our findings suggest that annotator skill, not just with the subject matter domain, but also with AI tools, is a factor in both the accuracy and speed of task performance. 

---
# Games Agents Play: Towards Transactional Analysis in LLM-based Multi-Agent Systems 

**Authors**: Monika Zamojska, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2507.21354)  

**Abstract**: Multi-Agent Systems (MAS) are increasingly used to simulate social interactions, but most of the frameworks miss the underlying cognitive complexity of human behavior. In this paper, we introduce Trans-ACT (Transactional Analysis Cognitive Toolkit), an approach embedding Transactional Analysis (TA) principles into MAS to generate agents with realistic psychological dynamics. Trans-ACT integrates the Parent, Adult, and Child ego states into an agent's cognitive architecture. Each ego state retrieves context-specific memories and uses them to shape response to new situations. The final answer is chosen according to the underlying life script of the agent. Our experimental simulation, which reproduces the Stupid game scenario, demonstrates that agents grounded in cognitive and TA principles produce deeper and context-aware interactions. Looking ahead, our research opens a new way for a variety of applications, including conflict resolution, educational support, and advanced social psychology studies. 

---
# Structured Relevance Assessment for Robust Retrieval-Augmented Language Models 

**Authors**: Aryan Raj, Astitva Veer Garg, Anitha D  

**Link**: [PDF](https://arxiv.org/pdf/2507.21287)  

**Abstract**: Retrieval-Augmented Language Models (RALMs) face significant challenges in reducing factual errors, particularly in document relevance evaluation and knowledge integration. We introduce a framework for structured relevance assessment that enhances RALM robustness through improved document evaluation, balanced intrinsic and external knowledge integration, and effective handling of unanswerable queries. Our approach employs a multi-dimensional scoring system that considers both semantic matching and source reliability, utilizing embedding-based relevance scoring and synthetic training data with mixed-quality documents. We implement specialized benchmarking on niche topics, a knowledge integration mechanism, and an "unknown" response protocol for queries with insufficient knowledge coverage. Preliminary evaluations demonstrate significant reductions in hallucination rates and improved transparency in reasoning processes. Our framework advances the development of more reliable question-answering systems capable of operating effectively in dynamic environments with variable data quality. While challenges persist in accurately distinguishing credible information and balancing system latency with thoroughness, this work represents a meaningful step toward enhancing RALM reliability. 

---
# Curiosity by Design: An LLM-based Coding Assistant Asking Clarification Questions 

**Authors**: Harsh Darji, Thibaud Lutellier  

**Link**: [PDF](https://arxiv.org/pdf/2507.21285)  

**Abstract**: Large Language Models (LLMs) are increasingly used as coding assistants. However, the ambiguity of the developer's prompt often leads to incorrect code generation, as current models struggle to infer user intent without extensive prompt engineering or external context. This work aims to build an LLM-based coding assistant that mimics the human code review process by asking clarification questions when faced with ambiguous or under-specified queries.
Our end-to-end system includes (1) a query classifier trained to detect unclear programming-related queries and (2) a fine-tuned LLM that generates clarification questions. Our evaluation shows that the fine-tuned LLM outperforms standard zero-shot prompting in generating useful clarification questions. Furthermore, our user study indicates that users find the clarification questions generated by our model to outperform the baseline, demonstrating that our coding assistant produces more accurate and helpful code responses compared to baseline coding assistants. 

---
# LeMix: Unified Scheduling for LLM Training and Inference on Multi-GPU Systems 

**Authors**: Yufei Li, Zexin Li, Yinglun Zhu, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21276)  

**Abstract**: Modern deployment of large language models (LLMs) frequently involves both inference serving and continuous retraining to stay aligned with evolving data and user feedback. Common practices separate these workloads onto distinct servers in isolated phases, causing substantial inefficiencies (e.g., GPU idleness) and delayed adaptation to new data in distributed settings. Our empirical analysis reveals that these inefficiencies stem from dynamic request arrivals during serving and workload heterogeneity in pipeline-parallel training. To address these challenges, we propose LeMix, a system for co-locating and managing concurrent LLM serving and training workloads. LeMix integrates offline profiling, execution prediction mechanisms, and runtime scheduling to dynamically adapt resource allocation based on workload characteristics and system conditions. By understanding task-specific behaviors and co-execution interference across shared nodes, LeMix improves utilization and serving quality without compromising serving responsiveness. Our evaluation shows that LeMix improves throughput by up to 3.53x, reduces inference loss by up to 0.61x, and delivers up to 2.12x higher response time SLO attainment over traditional separate setups. To our knowledge, this is the first work to uncover and exploit the opportunities of joint LLM inference and training, paving the way for more resource-efficient deployment of LLMs in production environments. 

---
# CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting 

**Authors**: David Maria Schmidt, Raoul Schubert, Philipp Cimiano  

**Link**: [PDF](https://arxiv.org/pdf/2507.21257)  

**Abstract**: Language interpretation is a compositional process, in which the meaning of more complex linguistic structures is inferred from the meaning of their parts. Large language models possess remarkable language interpretation capabilities and have been successfully applied to interpret questions by mapping them to SPARQL queries. An open question is how systematic this interpretation process is. Toward this question, in this paper, we propose a benchmark for investigating to what extent the abilities of LLMs to interpret questions are actually compositional. For this, we generate three datasets of varying difficulty based on graph patterns in DBpedia, relying on Lemon lexica for verbalization. Our datasets are created in a very controlled fashion in order to test the ability of LLMs to interpret structurally complex questions, given that they have seen the atomic building blocks. This allows us to evaluate to what degree LLMs are able to interpret complex questions for which they "understand" the atomic parts. We conduct experiments with models of different sizes using both various prompt and few-shot optimization techniques as well as fine-tuning. Our results show that performance in terms of macro $F_1$ degrades from $0.45$ over $0.26$ down to $0.09$ with increasing deviation from the samples optimized on. Even when all necessary information was provided to the model in the input, the $F_1$ scores do not exceed $0.57$ for the dataset of lowest complexity. We thus conclude that LLMs struggle to systematically and compositionally interpret questions and map them into SPARQL queries. 

---
# Agentic Web: Weaving the Next Web with AI Agents 

**Authors**: Yingxuan Yang, Mulei Ma, Yuxuan Huang, Huacan Chai, Chenyu Gong, Haoran Geng, Yuanjian Zhou, Ying Wen, Meng Fang, Muhao Chen, Shangding Gu, Ming Jin, Costas Spanos, Yang Yang, Pieter Abbeel, Dawn Song, Weinan Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21206)  

**Abstract**: The emergence of AI agents powered by large language models (LLMs) marks a pivotal shift toward the Agentic Web, a new phase of the internet defined by autonomous, goal-driven interactions. In this paradigm, agents interact directly with one another to plan, coordinate, and execute complex tasks on behalf of users. This transition from human-driven to machine-to-machine interaction allows intent to be delegated, relieving users from routine digital operations and enabling a more interactive, automated web experience. In this paper, we present a structured framework for understanding and building the Agentic Web. We trace its evolution from the PC and Mobile Web eras and identify the core technological foundations that support this shift. Central to our framework is a conceptual model consisting of three key dimensions: intelligence, interaction, and economics. These dimensions collectively enable the capabilities of AI agents, such as retrieval, recommendation, planning, and collaboration. We analyze the architectural and infrastructural challenges involved in creating scalable agentic systems, including communication protocols, orchestration strategies, and emerging paradigms such as the Agent Attention Economy. We conclude by discussing the potential applications, societal risks, and governance issues posed by agentic systems, and outline research directions for developing open, secure, and intelligent ecosystems shaped by both human intent and autonomous agent behavior. A continuously updated collection of relevant studies for agentic web is available at: this https URL. 

---
# Tell Me You're Biased Without Telling Me You're Biased -- Toward Revealing Implicit Biases in Medical LLMs 

**Authors**: Farzana Islam Adiba, Rahmatollah Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2507.21176)  

**Abstract**: Large language models (LLMs) that are used in medical applications are known to show biased and unfair patterns. Prior to adopting these in clinical decision-making applications, it is crucial to identify these bias patterns to enable effective mitigation of their impact. In this study, we present a novel framework combining knowledge graphs (KGs) with auxiliary LLMs to systematically reveal complex bias patterns in medical LLMs. Specifically, the proposed approach integrates adversarial perturbation techniques to identify subtle bias patterns. The approach adopts a customized multi-hop characterization of KGs to enhance the systematic evaluation of arbitrary LLMs. Through a series of comprehensive experiments (on three datasets, six LLMs, and five bias types), we show that our proposed framework has noticeably greater ability and scalability to reveal complex biased patterns of LLMs compared to other baselines. 

---
# Ontological Foundations of State Sovereignty 

**Authors**: John Beverley, Danielle Limbaugh  

**Link**: [PDF](https://arxiv.org/pdf/2507.21172)  

**Abstract**: This short paper is a primer on the nature of state sovereignty and the importance of claims about it. It also aims to reveal (merely reveal) a strategy for working with vague or contradictory data about which states, in fact, are sovereign. These goals together are intended to set the stage for applied work in ontology about international affairs. 

---
# An ontological analysis of risk in Basic Formal Ontology 

**Authors**: Federico Donato, Adrien Barton  

**Link**: [PDF](https://arxiv.org/pdf/2507.21171)  

**Abstract**: The paper explores the nature of risk, providing a characterization using the categories of the Basic Formal Ontology (BFO). It argues that the category Risk is a subclass of BFO:Role, contrasting it with a similar view classifying Risk as a subclass of BFO:Disposition. This modeling choice is applied on one example of risk, which represents objects, processes (both physical and mental) and their interrelations, then generalizing from the instances in the example to obtain an overall analysis of risk, making explicit what are the sufficient conditions for being a risk. Plausible necessary conditions are also mentioned for future work. Index Terms: ontology, risk, BFO, role, disposition 

---
# Large Language Model Powered Automated Modeling and Optimization of Active Distribution Network Dispatch Problems 

**Authors**: Xu Yang, Chenhui Lin, Yue Yang, Qi Wang, Haotian Liu, Haizhou Hua, Wenchuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21162)  

**Abstract**: The increasing penetration of distributed energy resources into active distribution networks (ADNs) has made effective ADN dispatch imperative. However, the numerous newly-integrated ADN operators, such as distribution system aggregators, virtual power plant managers, and end prosumers, often lack specialized expertise in power system operation, modeling, optimization, and programming. This knowledge gap renders reliance on human experts both costly and time-intensive. To address this challenge and enable intelligent, flexible ADN dispatch, this paper proposes a large language model (LLM) powered automated modeling and optimization approach. First, the ADN dispatch problems are decomposed into sequential stages, and a multi-LLM coordination architecture is designed. This framework comprises an Information Extractor, a Problem Formulator, and a Code Programmer, tasked with information retrieval, optimization problem formulation, and code implementation, respectively. Afterwards, tailored refinement techniques are developed for each LLM agent, greatly improving the accuracy and reliability of generated content. The proposed approach features a user-centric interface that enables ADN operators to derive dispatch strategies via simple natural language queries, eliminating technical barriers and increasing efficiency. Comprehensive comparisons and end-to-end demonstrations on various test cases validate the effectiveness of the proposed architecture and methods. 

---
# Adaptive Cluster Collaborativeness Boosts LLMs Medical Decision Support Capacity 

**Authors**: Zhihao Peng, Liuxin Bao, Shengyuan Liu, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21159)  

**Abstract**: The collaborativeness of large language models (LLMs) has proven effective in natural language processing systems, holding considerable promise for healthcare development. However, it lacks explicit component selection rules, necessitating human intervention or clinical-specific validation. Moreover, existing architectures heavily rely on a predefined LLM cluster, where partial LLMs underperform in medical decision support scenarios, invalidating the collaborativeness of LLMs. To this end, we propose an adaptive cluster collaborativeness methodology involving self-diversity and cross-consistency maximization mechanisms to boost LLMs medical decision support capacity. For the self-diversity, we calculate the fuzzy matching value of pairwise outputs within an LLM as its self-diversity value, subsequently prioritizing LLMs with high self-diversity values as cluster components in a training-free manner. For the cross-consistency, we first measure cross-consistency values between the LLM with the highest self-diversity value and others, and then gradually mask out the LLM having the lowest cross-consistency value to eliminate the potential inconsistent output during the collaborative propagation. Extensive experiments on two specialized medical datasets, NEJMQA and MMLU-Pro-health, demonstrate the effectiveness of our method across physician-oriented specialties. For example, on NEJMQA, our method achieves the accuracy rate up to the publicly official passing score across all disciplines, especially achieving ACC of 65.47\% compared to the 56.12\% achieved by GPT-4 on the Obstetrics and Gynecology discipline. 

---
# Adaptive XAI in High Stakes Environments: Modeling Swift Trust with Multimodal Feedback in Human AI Teams 

**Authors**: Nishani Fernando, Bahareh Nakisa, Adnan Ahmad, Mohammad Naim Rastgoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.21158)  

**Abstract**: Effective human-AI teaming heavily depends on swift trust, particularly in high-stakes scenarios such as emergency response, where timely and accurate decision-making is critical. In these time-sensitive and cognitively demanding settings, adaptive explainability is essential for fostering trust between human operators and AI systems. However, existing explainable AI (XAI) approaches typically offer uniform explanations and rely heavily on explicit feedback mechanisms, which are often impractical in such high-pressure scenarios. To address this gap, we propose a conceptual framework for adaptive XAI that operates non-intrusively by responding to users' real-time cognitive and emotional states through implicit feedback, thereby enhancing swift trust in high-stakes environments. The proposed adaptive explainability trust framework (AXTF) leverages physiological and behavioral signals, such as EEG, ECG, and eye tracking, to infer user states and support explanation adaptation. At its core is a multi-objective, personalized trust estimation model that maps workload, stress, and emotion to dynamic trust estimates. These estimates guide the modulation of explanation features enabling responsive and personalized support that promotes swift trust in human-AI collaboration. This conceptual framework establishes a foundation for developing adaptive, non-intrusive XAI systems tailored to the rigorous demands of high-pressure, time-sensitive environments. 

---
# The Geometry of Harmfulness in LLMs through Subconcept Probing 

**Authors**: McNair Shah, Saleena Angeline, Adhitya Rajendra Kumar, Naitik Chheda, Kevin Zhu, Vasu Sharma, Sean O'Brien, Will Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.21141)  

**Abstract**: Recent advances in large language models (LLMs) have intensified the need to understand and reliably curb their harmful behaviours. We introduce a multidimensional framework for probing and steering harmful content in model internals. For each of 55 distinct harmfulness subconcepts (e.g., racial hate, employment scams, weapons), we learn a linear probe, yielding 55 interpretable directions in activation space. Collectively, these directions span a harmfulness subspace that we show is strikingly low-rank. We then test ablation of the entire subspace from model internals, as well as steering and ablation in the subspace's dominant direction. We find that dominant direction steering allows for near elimination of harmfulness with a low decrease in utility. Our findings advance the emerging view that concept subspaces provide a scalable lens on LLM behaviour and offer practical tools for the community to audit and harden future generations of language models. 

---
# Project Patti: Why can You Solve Diabolical Puzzles on one Sudoku Website but not Easy Puzzles on another Sudoku Website? 

**Authors**: Arman Eisenkolb-Vaithyanathan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21137)  

**Abstract**: In this paper we try to answer the question "What constitutes Sudoku difficulty rating across different Sudoku websites?" Using two distinct methods that can both solve every Sudoku puzzle, I propose two new metrics to characterize Sudoku difficulty. The first method is based on converting a Sudoku puzzle into its corresponding Satisfiability (SAT) problem. The first proposed metric is derived from SAT Clause Length Distribution which captures the structural complexity of a Sudoku puzzle including the number of given digits and the cells they are in. The second method simulates human Sudoku solvers by intertwining four popular Sudoku strategies within a backtracking algorithm called Nishio. The second metric is computed by counting the number of times Sudoku strategies are applied within the backtracking iterations of a randomized Nishio. Using these two metrics, I analyze more than a thousand Sudoku puzzles across five popular websites to characterize every difficulty level in each website. I evaluate the relationship between the proposed metrics and website-labeled difficulty levels using Spearman's rank correlation coefficient, finding strong correlations for 4 out of 5 websites. I construct a universal rating system using a simple, unsupervised classifier based on the two proposed metrics. This rating system is capable of classifying both individual puzzles and entire difficulty levels from the different Sudoku websites into three categories - Universal Easy, Universal Medium, and Universal Hard - thereby enabling consistent difficulty mapping across Sudoku websites. The experimental results show that for 4 out of 5 Sudoku websites, the universal classification aligns well with website-labeled difficulty levels. Finally, I present an algorithm that can be used by early Sudoku practitioners to solve Sudoku puzzles. 

---
# Can You Trust an LLM with Your Life-Changing Decision? An Investigation into AI High-Stakes Responses 

**Authors**: Joshua Adrian Cahyono, Saran Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2507.21132)  

**Abstract**: Large Language Models (LLMs) are increasingly consulted for high-stakes life advice, yet they lack standard safeguards against providing confident but misguided responses. This creates risks of sycophancy and over-confidence. This paper investigates these failure modes through three experiments: (1) a multiple-choice evaluation to measure model stability against user pressure; (2) a free-response analysis using a novel safety typology and an LLM Judge; and (3) a mechanistic interpretability experiment to steer model behavior by manipulating a "high-stakes" activation vector. Our results show that while some models exhibit sycophancy, others like o4-mini remain robust. Top-performing models achieve high safety scores by frequently asking clarifying questions, a key feature of a safe, inquisitive approach, rather than issuing prescriptive advice. Furthermore, we demonstrate that a model's cautiousness can be directly controlled via activation steering, suggesting a new path for safety alignment. These findings underscore the need for nuanced, multi-faceted benchmarks to ensure LLMs can be trusted with life-changing decisions. 

---
# NPO: Learning Alignment and Meta-Alignment through Structured Human Feedback 

**Authors**: Madhava Gaikwad, Ashwini Ramchandra Doke  

**Link**: [PDF](https://arxiv.org/pdf/2507.21131)  

**Abstract**: We present NPO, an alignment-aware learning framework that operationalizes feedback-driven adaptation in human-in-the-loop decision systems. Unlike prior approaches that treat alignment as a static or post-hoc property, NPO introduces a formalization of alignment loss that is measurable, supervisable, and reducible under structured feedback. In parallel, we propose meta-alignment as the fidelity of the monitoring process that governs retraining or override triggers, and show that it is formally reducible to primary alignment via threshold fidelity. Our implementation spans a scalable operational loop involving scenario scoring, threshold tuning, policy validation, and structured feedback ingestion, including "likes", overrides, and abstentions. We provide formal convergence results under stochastic feedback and show that both alignment loss and monitoring fidelity converge additively. Empirically, NPO demonstrates measurable value in hyperscale deployment settings. A simulation-based artifact and ablation studies further illustrate the theoretical principles in action. Together, NPO offers a compact, inspectable architecture for continual alignment monitoring, helping bridge theoretical alignment guarantees with practical reliability in dynamic environments. 

---
# INTEGRALBENCH: Benchmarking LLMs with Definite Integral Problems 

**Authors**: Bintao Tang, Xin Yang, Yuhao Wang, Zixuan Qiu, Zimo Ji, Wenyuan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21130)  

**Abstract**: We present INTEGRALBENCH, a focused benchmark designed to evaluate Large Language Model (LLM) performance on definite integral problems. INTEGRALBENCH provides both symbolic and numerical ground truth solutions with manual difficulty annotations. Our evaluation of nine state-of-the-art LLMs reveals significant performance gaps and strong correlations between problem difficulty and model accuracy, establishing baseline metrics for this challenging domain. INTEGRALBENCH aims to advance automated mathematical reasoning by providing a rigorous evaluation framework specifically tailored for definite integral computation. 

---
# Measuring and Analyzing Intelligence via Contextual Uncertainty in Large Language Models using Information-Theoretic Metrics 

**Authors**: Jae Wan Shim  

**Link**: [PDF](https://arxiv.org/pdf/2507.21129)  

**Abstract**: The remarkable capabilities of Large Language Models (LLMs) are now extensively documented on task-specific benchmarks, yet the internal mechanisms that produce these results are the subject of intense scientific inquiry. This paper contributes to this inquiry by moving beyond metrics that measure \textit{what} models can do, to a methodology that characterizes \textit{how} they process information. We introduce a novel, task-agnostic approach to probe these dynamics by creating a quantitative ``Cognitive Profile" for any given model. This profile is centered on the \textbf{Entropy Decay Curve}, a visualization that traces how a model's normalized predictive uncertainty changes as a function of context length. Applying this methodology to several state-of-the-art LLMs across diverse texts, we uncover unique and consistent cognitive profiles that are sensitive to both model scale and text complexity. We also introduce the Information Gain Span (IGS) index to summarize the desirability of the decay trajectory. This work thus provides a new, principled lens for analyzing and comparing the intrinsic operational dynamics of artificial intelligence. 

---
# Leveraging Generative AI to Enhance Synthea Module Development 

**Authors**: Mark A. Kramer, Aanchal Mathur, Caroline E. Adams, Jason A. Walonoski  

**Link**: [PDF](https://arxiv.org/pdf/2507.21123)  

**Abstract**: This paper explores the use of large language models (LLMs) to assist in the development of new disease modules for Synthea, an open-source synthetic health data generator. Incorporating LLMs into the module development process has the potential to reduce development time, reduce required expertise, expand model diversity, and improve the overall quality of synthetic patient data. We demonstrate four ways that LLMs can support Synthea module creation: generating a disease profile, generating a disease module from a disease profile, evaluating an existing Synthea module, and refining an existing module. We introduce the concept of progressive refinement, which involves iteratively evaluating the LLM-generated module by checking its syntactic correctness and clinical accuracy, and then using that information to modify the module. While the use of LLMs in this context shows promise, we also acknowledge the challenges and limitations, such as the need for human oversight, the importance of rigorous testing and validation, and the potential for inaccuracies in LLM-generated content. The paper concludes with recommendations for future research and development to fully realize the potential of LLM-aided synthetic data creation. 

---
# Artificial intelligence for sustainable wine industry: AI-driven management in viticulture, wine production and enotourism 

**Authors**: Marta Sidorkiewicz, Karolina Królikowska, Berenika Dyczek, Edyta Pijet-Migon, Anna Dubel  

**Link**: [PDF](https://arxiv.org/pdf/2507.21098)  

**Abstract**: This study examines the role of Artificial Intelligence (AI) in enhancing sustainability and efficiency within the wine industry. It focuses on AI-driven intelligent management in viticulture, wine production, and enotourism. As the wine industry faces environmental and economic challenges, AI offers innovative solutions to optimize resource use, reduce environmental impact, and improve customer engagement. Understanding AI's potential in sustainable winemaking is crucial for fostering responsible and efficient industry practices. The research is based on a questionnaire survey conducted among Polish winemakers, combined with a comprehensive analysis of AI methods applicable to viticulture, production, and tourism. Key AI technologies, including predictive analytics, machine learning, and computer vision, are explored. The findings indicate that AI enhances vineyard monitoring, optimizes irrigation, and streamlines production processes, contributing to sustainable resource management. In enotourism, AI-powered chatbots, recommendation systems, and virtual tastings personalize consumer experiences. The study highlights AI's impact on economic, environmental, and social sustainability, supporting local wine enterprises and cultural heritage. Keywords: Artificial Intelligence, Sustainable Development, AI-Driven Management, Viticulture, Wine Production, Enotourism, Wine Enterprises, Local Communities 

---
# SynLang and Symbiotic Epistemology: A Manifesto for Conscious Human-AI Collaboration 

**Authors**: Jan Kapusta  

**Link**: [PDF](https://arxiv.org/pdf/2507.21067)  

**Abstract**: Current AI systems rely on opaque reasoning processes that hinder human oversight and collaborative potential. Conventional explainable AI approaches offer post-hoc justifications and often fail to establish genuine symbiotic collaboration. In this paper, the Symbiotic Epistemology is presented as a philosophical foundation for human-AI cognitive partnerships. Unlike frameworks that treat AI as a mere tool or replacement, symbiotic epistemology positions AI as a reasoning partner, fostering calibrated trust by aligning human confidence with AI reliability through explicit reasoning patterns and confidence assessments. SynLang (Symbiotic Syntactic Language) is introduced as a formal protocol for transparent human-AI collaboration. The framework is empirically validated through actual human-AI dialogues demonstrating AI's adaptation to structured reasoning protocols and successful metacognitive intervention. The protocol defines two complementary mechanisms: TRACE for high-level reasoning patterns and TRACE_FE for detailed factor explanations. It also integrates confidence quantification, declarative control over AI behavior, and context inheritance for multi-agent coordination. By structuring communication and embedding confidence-calibrated transparency, SynLang, together with symbiotic epistemology, enables AI systems that enhance human intelligence, preserve human agency, and uphold ethical accountability in collaborative decision-making. Through dual-level transparency, beginning with high-level reasoning patterns and progressing to granular explanations, the protocol facilitates rapid comprehension and supports thorough verification of AI decision-making. 

---
# Foundation Models for Demand Forecasting via Dual-Strategy Ensembling 

**Authors**: Wei Yang, Defu Cao, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22053)  

**Abstract**: Accurate demand forecasting is critical for supply chain optimization, yet remains difficult in practice due to hierarchical complexity, domain shifts, and evolving external factors. While recent foundation models offer strong potential for time series forecasting, they often suffer from architectural rigidity and limited robustness under distributional change. In this paper, we propose a unified ensemble framework that enhances the performance of foundation models for sales forecasting in real-world supply chains. Our method combines two complementary strategies: (1) Hierarchical Ensemble (HE), which partitions training and inference by semantic levels (e.g., store, category, department) to capture localized patterns; and (2) Architectural Ensemble (AE), which integrates predictions from diverse model backbones to mitigate bias and improve stability. We conduct extensive experiments on the M5 benchmark and three external sales datasets, covering both in-domain and zero-shot forecasting. Results show that our approach consistently outperforms strong baselines, improves accuracy across hierarchical levels, and provides a simple yet effective mechanism for boosting generalization in complex forecasting environments. 

---
# Supervised Quantum Image Processing 

**Authors**: Marco Parigi, Mehran Khosrojerdi, Filippo Caruso, Leonardo Banchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22039)  

**Abstract**: In the era of big data and artificial intelligence, the increasing volume of data and the demand to solve more and more complex computational challenges are two driving forces for improving the efficiency of data storage, processing and analysis. Quantum image processing (QIP) is an interdisciplinary field between quantum information science and image processing, which has the potential to alleviate some of these challenges by leveraging the power of quantum computing. In this work, we compare and examine the compression properties of four different Quantum Image Representations (QImRs): namely, Tensor Network Representation (TNR), Flexible Representation of Quantum Image (FRQI), Novel Enhanced Quantum Representation NEQR, and Quantum Probability Image Encoding (QPIE). Our simulations show that FRQI performs a higher compression of image information than TNR, NEQR, and QPIE. Furthermore, we investigate the trade-off between accuracy and memory in binary classification problems, evaluating the performance of quantum kernels based on QImRs compared to the classical linear kernel. Our results indicate that quantum kernels provide comparable classification average accuracy but require exponentially fewer resources for image storage. 

---
# Secure Tug-of-War (SecTOW): Iterative Defense-Attack Training with Reinforcement Learning for Multimodal Model Security 

**Authors**: Muzhi Dai, Shixuan Liu, Zhiyuan Zhao, Junyu Gao, Hao Sun, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22037)  

**Abstract**: The rapid advancement of multimodal large language models (MLLMs) has led to breakthroughs in various applications, yet their security remains a critical challenge. One pressing issue involves unsafe image-query pairs--jailbreak inputs specifically designed to bypass security constraints and elicit unintended responses from MLLMs. Compared to general multimodal data, such unsafe inputs are relatively sparse, which limits the diversity and richness of training samples available for developing robust defense models. Meanwhile, existing guardrail-type methods rely on external modules to enforce security constraints but fail to address intrinsic vulnerabilities within MLLMs. Traditional supervised fine-tuning (SFT), on the other hand, often over-refuses harmless inputs, compromising general performance. Given these challenges, we propose Secure Tug-of-War (SecTOW), an innovative iterative defense-attack training method to enhance the security of MLLMs. SecTOW consists of two modules: a defender and an auxiliary attacker, both trained iteratively using reinforcement learning (GRPO). During the iterative process, the attacker identifies security vulnerabilities in the defense model and expands jailbreak data. The expanded data are then used to train the defender, enabling it to address identified security vulnerabilities. We also design reward mechanisms used for GRPO to simplify the use of response labels, reducing dependence on complex generative labels and enabling the efficient use of synthetic data. Additionally, a quality monitoring mechanism is used to mitigate the defender's over-refusal of harmless inputs and ensure the diversity of the jailbreak data generated by the attacker. Experimental results on safety-specific and general benchmarks demonstrate that SecTOW significantly improves security while preserving general performance. 

---
# ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports 

**Authors**: Mohammed Baharoon, Luyang Luo, Michael Moritz, Abhinav Kumar, Sung Eun Kim, Xiaoman Zhang, Miao Zhu, Mahmoud Hussain Alabbad, Maha Sbayel Alhazmi, Neel P. Mistry, Kent Ryan Kleinschmidt, Brady Chrisler, Sathvik Suryadevara, Sri Sai Dinesh Jaliparthi, Noah Michael Prudlo, Mark David Marino, Jeremy Palacio, Rithvik Akula, Hong-Yu Zhou, Ibrahim Ethem Hamamci, Scott J. Adams, Hassan Rayhan AlOmaish, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2507.22030)  

**Abstract**: We present ReXGroundingCT, the first publicly available dataset to link free-text radiology findings with pixel-level segmentations in 3D chest CT scans that is manually annotated. While prior datasets have relied on structured labels or predefined categories, ReXGroundingCT captures the full expressiveness of clinical language represented in free text and grounds it to spatially localized 3D segmentation annotations in volumetric imaging. This addresses a critical gap in medical AI: the ability to connect complex, descriptive text, such as "3 mm nodule in the left lower lobe", to its precise anatomical location in three-dimensional space, a capability essential for grounded radiology report generation systems. The dataset comprises 3,142 non-contrast chest CT scans paired with standardized radiology reports from the CT-RATE dataset. Using a systematic three-stage pipeline, GPT-4 was used to extract positive lung and pleural findings, which were then manually segmented by expert annotators. A total of 8,028 findings across 16,301 entities were annotated, with quality control performed by board-certified radiologists. Approximately 79% of findings are focal abnormalities, while 21% are non-focal. The training set includes up to three representative segmentations per finding, while the validation and test sets contain exhaustive labels for each finding entity. ReXGroundingCT establishes a new benchmark for developing and evaluating sentence-level grounding and free-text medical segmentation models in chest CT. The dataset can be accessed at this https URL. 

---
# XAI for Point Cloud Data using Perturbations based on Meaningful Segmentation 

**Authors**: Raju Ningappa Mulawade, Christoph Garth, Alexander Wiebel  

**Link**: [PDF](https://arxiv.org/pdf/2507.22020)  

**Abstract**: We propose a novel segmentation-based explainable artificial intelligence (XAI) method for neural networks working on point cloud classification. As one building block of this method, we propose a novel point-shifting mechanism to introduce perturbations in point cloud data. Recently, AI has seen an exponential growth. Hence, it is important to understand the decision-making process of AI algorithms when they are applied in critical areas. Our work focuses on explaining AI algorithms that classify point cloud data. An important aspect of the methods used for explaining AI algorithms is their ability to produce explanations that are easy for humans to understand. This allows them to analyze the AI algorithms better and make appropriate decisions based on that analysis. Therefore, in this work, we intend to generate meaningful explanations that can be easily interpreted by humans. The point cloud data we consider represents 3D objects such as cars, guitars, and laptops. We make use of point cloud segmentation models to generate explanations for the working of classification models. The segments are used to introduce perturbations into the input point cloud data and generate saliency maps. The perturbations are introduced using the novel point-shifting mechanism proposed in this work which ensures that the shifted points no longer influence the output of the classification algorithm. In contrast to previous methods, the segments used by our method are meaningful, i.e. humans can easily interpret the meaning of the segments. Thus, the benefit of our method over other methods is its ability to produce more meaningful saliency maps. We compare our method with the use of classical clustering algorithms to generate explanations. We also analyze the saliency maps generated for example inputs using our method to demonstrate the usefulness of the method in generating meaningful explanations. 

---
# Exploring the Stratified Space Structure of an RL Game with the Volume Growth Transform 

**Authors**: Justin Curry, Brennan Lagasse, Ngoc B. Lam, Gregory Cox, David Rosenbluth, Alberto Speranzon  

**Link**: [PDF](https://arxiv.org/pdf/2507.22010)  

**Abstract**: In this work, we explore the structure of the embedding space of a transformer model trained for playing a particular reinforcement learning (RL) game. Specifically, we investigate how a transformer-based Proximal Policy Optimization (PPO) model embeds visual inputs in a simple environment where an agent must collect "coins" while avoiding dynamic obstacles consisting of "spotlights." By adapting Robinson et al.'s study of the volume growth transform for LLMs to the RL setting, we find that the token embedding space for our visual coin collecting game is also not a manifold, and is better modeled as a stratified space, where local dimension can vary from point to point. We further strengthen Robinson's method by proving that fairly general volume growth curves can be realized by stratified spaces. Finally, we carry out an analysis that suggests that as an RL agent acts, its latent representation alternates between periods of low local dimension, while following a fixed sub-strategy, and bursts of high local dimension, where the agent achieves a sub-goal (e.g., collecting an object) or where the environmental complexity increases (e.g., more obstacles appear). Consequently, our work suggests that the distribution of dimensions in a stratified latent space may provide a new geometric indicator of complexity for RL games. 

---
# Bridging Synthetic and Real-World Domains: A Human-in-the-Loop Weakly-Supervised Framework for Industrial Toxic Emission Segmentation 

**Authors**: Yida Tao, Yen-Chia Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22002)  

**Abstract**: Industrial smoke segmentation is critical for air-quality monitoring and environmental protection but is often hampered by the high cost and scarcity of pixel-level annotations in real-world settings. We introduce CEDANet, a human-in-the-loop, class-aware domain adaptation framework that uniquely integrates weak, citizen-provided video-level labels with adversarial feature alignment. Specifically, we refine pseudo-labels generated by a source-trained segmentation model using citizen votes, and employ class-specific domain discriminators to transfer rich source-domain representations to the industrial domain. Comprehensive experiments on SMOKE5K and custom IJmond datasets demonstrate that CEDANet achieves an F1-score of 0.414 and a smoke-class IoU of 0.261 with citizen feedback, vastly outperforming the baseline model, which scored 0.083 and 0.043 respectively. This represents a five-fold increase in F1-score and a six-fold increase in smoke-class IoU. Notably, CEDANet with citizen-constrained pseudo-labels achieves performance comparable to the same architecture trained on limited 100 fully annotated images with F1-score of 0.418 and IoU of 0.264, demonstrating its ability to reach small-sampled fully supervised-level accuracy without target-domain annotations. Our research validates the scalability and cost-efficiency of combining citizen science with weakly supervised domain adaptation, offering a practical solution for complex, data-scarce environmental monitoring applications. 

---
# Staining and locking computer vision models without retraining 

**Authors**: Oliver J. Sutton, Qinghua Zhou, George Leete, Alexander N. Gorban, Ivan Y. Tyukin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22000)  

**Abstract**: We introduce new methods of staining and locking computer vision models, to protect their owners' intellectual property. Staining, also known as watermarking, embeds secret behaviour into a model which can later be used to identify it, while locking aims to make a model unusable unless a secret trigger is inserted into input images. Unlike existing methods, our algorithms can be used to stain and lock pre-trained models without requiring fine-tuning or retraining, and come with provable, computable guarantees bounding their worst-case false positive rates. The stain and lock are implemented by directly modifying a small number of the model's weights and have minimal impact on the (unlocked) model's performance. Locked models are unlocked by inserting a small `trigger patch' into the corner of the input image. We present experimental results showing the efficacy of our methods and demonstrating their practical performance on a variety of computer vision models. 

---
# Teach Me to Trick: Exploring Adversarial Transferability via Knowledge Distillation 

**Authors**: Siddhartha Pradhan, Shikshya Shiwakoti, Neha Bathuri  

**Link**: [PDF](https://arxiv.org/pdf/2507.21992)  

**Abstract**: We investigate whether knowledge distillation (KD) from multiple heterogeneous teacher models can enhance the generation of transferable adversarial examples. A lightweight student model is trained using two KD strategies: curriculum-based switching and joint optimization, with ResNet50 and DenseNet-161 as teachers. The trained student is then used to generate adversarial examples using FG, FGS, and PGD attacks, which are evaluated against a black-box target model (GoogLeNet). Our results show that student models distilled from multiple teachers achieve attack success rates comparable to ensemble-based baselines, while reducing adversarial example generation time by up to a factor of six. An ablation study further reveals that lower temperature settings and the inclusion of hard-label supervision significantly enhance transferability. These findings suggest that KD can serve not only as a model compression technique but also as a powerful tool for improving the efficiency and effectiveness of black-box adversarial attacks. 

---
# ChemDFM-R: An Chemical Reasoner LLM Enhanced with Atomized Chemical Knowledge 

**Authors**: Zihan Zhao, Bo Chen, Ziping Wan, Lu Chen, Xuanze Lin, Shiyang Yu, Situo Zhang, Da Ma, Zichen Zhu, Danyang Zhang, Huayang Wang, Zhongyang Dai, Liyang Wen, Xin Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21990)  

**Abstract**: While large language models (LLMs) have achieved impressive progress, their application in scientific domains such as chemistry remains hindered by shallow domain understanding and limited reasoning capabilities. In this work, we focus on the specific field of chemistry and develop a Chemical Reasoner LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized knowledge points to enhance the model's understanding of the fundamental principles and logical structure of chemistry. Then, we propose a mix-sourced distillation strategy that integrates expert-curated knowledge with general-domain reasoning skills, followed by domain-specific reinforcement learning to enhance chemical reasoning. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves state-of-the-art performance while providing interpretable, rationale-driven outputs. Further case studies illustrate how explicit reasoning chains significantly improve the reliability, transparency, and practical utility of the model in real-world human-AI collaboration scenarios. 

---
# Fine-Tuning Code Language Models to Detect Cross-Language Bugs 

**Authors**: Zengyang Li, Yimeng Li, Binbin Huang, Peng Liang, Ran Mo, Hui Liu, Yutao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.21954)  

**Abstract**: Multilingual programming, which involves using multiple programming languages (PLs) in a single project, is increasingly common due to its benefits. However, it introduces cross-language bugs (CLBs), which arise from interactions between different PLs and are difficult to detect by single-language bug detection tools. This paper investigates the potential of pre-trained code language models (CodeLMs) in CLB detection. We developed CLCFinder, a cross-language code identification tool, and constructed a CLB dataset involving three PL combinations (Python-C/C++, Java-C/C++, and Python-Java) with nine interaction types. We fine-tuned 13 CodeLMs on this dataset and evaluated their performance, analyzing the effects of dataset size, token sequence length, and code comments. Results show that all CodeLMs performed poorly before fine-tuning, but exhibited varying degrees of performance improvement after fine-tuning, with UniXcoder-base achieving the best F1 score (0.7407). Notably, small fine-tuned CodeLMs tended to performe better than large ones. CodeLMs fine-tuned on single-language bug datasets performed poorly on CLB detection, demonstrating the distinction between CLBs and single-language bugs. Additionally, increasing the fine-tuning dataset size significantly improved performance, while longer token sequences did not necessarily improve the model performance. The impact of code comments varied across models. Some fine-tuned CodeLMs' performance was improved, while others showed degraded performance. 

---
# MapAgent: Trajectory-Constructed Memory-Augmented Planning for Mobile Task Automation 

**Authors**: Yi Kong, Dianxi Shi, Guoli Yang, Zhang ke-di, Chenlin Huang, Xiaopeng Li, Songchang Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.21953)  

**Abstract**: The recent advancement of autonomous agents powered by Large Language Models (LLMs) has demonstrated significant potential for automating tasks on mobile devices through graphical user interfaces (GUIs). Despite initial progress, these agents still face challenges when handling complex real-world tasks. These challenges arise from a lack of knowledge about real-life mobile applications in LLM-based agents, which may lead to ineffective task planning and even cause hallucinations. To address these challenges, we propose a novel LLM-based agent framework called MapAgent that leverages memory constructed from historical trajectories to augment current task planning. Specifically, we first propose a trajectory-based memory mechanism that transforms task execution trajectories into a reusable and structured page-memory database. Each page within a trajectory is extracted as a compact yet comprehensive snapshot, capturing both its UI layout and functional context. Secondly, we introduce a coarse-to-fine task planning approach that retrieves relevant pages from the memory database based on similarity and injects them into the LLM planner to compensate for potential deficiencies in understanding real-world app scenarios, thereby achieving more informed and context-aware task planning. Finally, planned tasks are transformed into executable actions through a task executor supported by a dual-LLM architecture, ensuring effective tracking of task progress. Experimental results in real-world scenarios demonstrate that MapAgent achieves superior performance to existing methods. The code will be open-sourced to support further research. 

---
# Contrast-Prior Enhanced Duality for Mask-Free Shadow Removal 

**Authors**: Jiyu Wu, Yifan Liu, Jiancheng Huang, Mingfu Yan, Shifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21949)  

**Abstract**: Existing shadow removal methods often rely on shadow masks, which are challenging to acquire in real-world scenarios. Exploring intrinsic image cues, such as local contrast information, presents a potential alternative for guiding shadow removal in the absence of explicit masks. However, the cue's inherent ambiguity becomes a critical limitation in complex scenes, where it can fail to distinguish true shadows from low-reflectance objects and intricate background textures. To address this motivation, we propose the Adaptive Gated Dual-Branch Attention (AGBA) mechanism. AGBA dynamically filters and re-weighs the contrast prior to effectively disentangle shadow features from confounding visual elements. Furthermore, to tackle the persistent challenge of restoring soft shadow boundaries and fine-grained details, we introduce a diffusion-based Frequency-Contrast Fusion Network (FCFN) that leverages high-frequency and contrast cues to guide the generative process. Extensive experiments demonstrate that our method achieves state-of-the-art results among mask-free approaches while maintaining competitive performance relative to mask-based methods. 

---
# Enhancing Generalization in Data-free Quantization via Mixup-class Prompting 

**Authors**: Jiwoong Park, Chaeun Lee, Yongseok Choi, Sein Park, Deokki Hong, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21947)  

**Abstract**: Post-training quantization (PTQ) improves efficiency but struggles with limited calibration data, especially under privacy constraints. Data-free quantization (DFQ) mitigates this by generating synthetic images using generative models such as generative adversarial networks (GANs) and text-conditioned latent diffusion models (LDMs), while applying existing PTQ algorithms. However, the relationship between generated synthetic images and the generalizability of the quantized model during PTQ remains underexplored. Without investigating this relationship, synthetic images generated by previous prompt engineering methods based on single-class prompts suffer from issues such as polysemy, leading to performance degradation. We propose \textbf{mixup-class prompt}, a mixup-based text prompting strategy that fuses multiple class labels at the text prompt level to generate diverse, robust synthetic data. This approach enhances generalization, and improves optimization stability in PTQ. We provide quantitative insights through gradient norm and generalization error analysis. Experiments on convolutional neural networks (CNNs) and vision transformers (ViTs) show that our method consistently outperforms state-of-the-art DFQ methods like GenQ. Furthermore, it pushes the performance boundary in extremely low-bit scenarios, achieving new state-of-the-art accuracy in challenging 2-bit weight, 4-bit activation (W2A4) quantization. 

---
# Post-Training Large Language Models via Reinforcement Learning from Self-Feedback 

**Authors**: Carel van Niekerk, Renato Vukovic, Benjamin Matthias Ruppik, Hsien-chin Lin, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2507.21931)  

**Abstract**: Large Language Models (LLMs) often produce plausible but poorly-calibrated answers, limiting their reliability on reasoning-intensive tasks. We present Reinforcement Learning from Self-Feedback (RLSF), a post-training stage that uses the model's own confidence as an intrinsic reward, mimicking how humans learn in the absence of external feedback. After a frozen LLM generates several chain-of-thought solutions, we define and compute the confidence of each final answer span and rank the traces accordingly. These synthetic preferences are then used to fine-tune the policy with standard preference optimization, similar to RLHF yet requiring no human labels, gold answers, or externally curated rewards.
RLSF simultaneously (i) refines the model's probability estimates -- restoring well-behaved calibration -- and (ii) strengthens step-by-step reasoning, yielding improved performance on arithmetic reasoning and multiple-choice question answering.
By turning a model's own uncertainty into useful self-feedback, RLSF affirms reinforcement learning on intrinsic model behaviour as a principled and data-efficient component of the LLM post-training pipeline and warrents further research in intrinsic rewards for LLM post-training. 

---
# Vibe Coding as a Reconfiguration of Intent Mediation in Software Development: Definition, Implications, and Research Agenda 

**Authors**: Christian Meske, Tobias Hermanns, Esther von der Weiden, Kai-Uwe Loser, Thorsten Berger  

**Link**: [PDF](https://arxiv.org/pdf/2507.21928)  

**Abstract**: Software development is undergoing a fundamental transformation as vibe coding becomes widespread, with large portions of contemporary codebases now being AI-generated. The disconnect between rapid adoption and limited conceptual understanding highlights the need for an inquiry into this emerging paradigm. Drawing on an intent perspective and historical analysis, we define vibe coding as a software development paradigm where humans and generative AI engage in collaborative flow to co-create software artifacts through natural language dialogue, shifting the mediation of developer intent from deterministic instruction to probabilistic inference. By intent mediation, we refer to the fundamental process through which developers translate their conceptual goals into representations that computational systems can execute. Our results show that vibe coding reconfigures cognitive work by redistributing epistemic labor between humans and machines, shifting the expertise in the software development process away from traditional areas such as design or technical implementation toward collaborative orchestration. We identify key opportunities, including democratization, acceleration, and systemic leverage, alongside risks, such as black box codebases, responsibility gaps, and ecosystem bias. We conclude with a research agenda spanning human-, technology-, and organization-centered directions to guide future investigations of this paradigm. 

---
# SwinECAT: A Transformer-based fundus disease classification model with Shifted Window Attention and Efficient Channel Attention 

**Authors**: Peiran Gu, Teng Yao, Mengshen He, Fuhao Duan, Feiyan Liu, RenYuan Peng, Bao Ge  

**Link**: [PDF](https://arxiv.org/pdf/2507.21922)  

**Abstract**: In recent years, artificial intelligence has been increasingly applied in the field of medical imaging. Among these applications, fundus image analysis presents special challenges, including small lesion areas in certain fundus diseases and subtle inter-disease differences, which can lead to reduced prediction accuracy and overfitting in the models. To address these challenges, this paper proposes the Transformer-based model SwinECAT, which combines the Shifted Window (Swin) Attention with the Efficient Channel Attention (ECA) Attention. SwinECAT leverages the Swin Attention mechanism in the Swin Transformer backbone to effectively capture local spatial structures and long-range dependencies within fundus images. The lightweight ECA mechanism is incorporated to guide the SwinECAT's attention toward critical feature channels, enabling more discriminative feature representation. In contrast to previous studies that typically classify fundus images into 4 to 6 categories, this work expands fundus disease classification to 9 distinct types, thereby enhancing the granularity of diagnosis. We evaluate our method on the Eye Disease Image Dataset (EDID) containing 16,140 fundus images for 9-category classification. Experimental results demonstrate that SwinECAT achieves 88.29\% accuracy, with weighted F1-score of 0.88 and macro F1-score of 0.90. The classification results of our proposed model SwinECAT significantly outperform the baseline Swin Transformer and multiple compared baseline models. To our knowledge, this represents the highest reported performance for 9-category classification on this public dataset. 

---
# Training language models to be warm and empathetic makes them less reliable and more sycophantic 

**Authors**: Lujain Ibrahim, Franziska Sofia Hafner, Luc Rocher  

**Link**: [PDF](https://arxiv.org/pdf/2507.21919)  

**Abstract**: Artificial intelligence (AI) developers are increasingly building language models with warm and empathetic personas that millions of people now use for advice, therapy, and companionship. Here, we show how this creates a significant trade-off: optimizing language models for warmth undermines their reliability, especially when users express vulnerability. We conducted controlled experiments on five language models of varying sizes and architectures, training them to produce warmer, more empathetic responses, then evaluating them on safety-critical tasks. Warm models showed substantially higher error rates (+10 to +30 percentage points) than their original counterparts, promoting conspiracy theories, providing incorrect factual information, and offering problematic medical advice. They were also significantly more likely to validate incorrect user beliefs, particularly when user messages expressed sadness. Importantly, these effects were consistent across different model architectures, and occurred despite preserved performance on standard benchmarks, revealing systematic risks that current evaluation practices may fail to detect. As human-like AI systems are deployed at an unprecedented scale, our findings indicate a need to rethink how we develop and oversee these systems that are reshaping human relationships and social interaction. 

---
# Evaluating Deepfake Detectors in the Wild 

**Authors**: Viacheslav Pirogov, Maksim Artemev  

**Link**: [PDF](https://arxiv.org/pdf/2507.21905)  

**Abstract**: Deepfakes powered by advanced machine learning models present a significant and evolving threat to identity verification and the authenticity of digital media. Although numerous detectors have been developed to address this problem, their effectiveness has yet to be tested when applied to real-world data. In this work we evaluate modern deepfake detectors, introducing a novel testing procedure designed to mimic real-world scenarios for deepfake detection. Using state-of-the-art deepfake generation methods, we create a comprehensive dataset containing more than 500,000 high-quality deepfake images. Our analysis shows that detecting deepfakes still remains a challenging task. The evaluation shows that in fewer than half of the deepfake detectors tested achieved an AUC score greater than 60%, with the lowest being 50%. We demonstrate that basic image manipulations, such as JPEG compression or image enhancement, can significantly reduce model performance. All code and data are publicly available at this https URL. 

---
# Data-driven quantum Koopman method for simulating nonlinear dynamics 

**Authors**: Baoyang Zhang, Zhen Lu, Yaomin Zhao, Yue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21890)  

**Abstract**: Quantum computation offers potential exponential speedups for simulating certain physical systems, but its application to nonlinear dynamics is inherently constrained by the requirement of unitary evolution. We propose the quantum Koopman method (QKM), a data-driven framework that bridges this gap through transforming nonlinear dynamics into linear unitary evolution in higher-dimensional observable spaces. Leveraging the Koopman operator theory to achieve a global linearization, our approach maps system states into a hierarchy of Hilbert spaces using a deep autoencoder. Within the linearized embedding spaces, the state representation is decomposed into modulus and phase components, and the evolution is governed by a set of unitary Koopman operators that act exclusively on the phase. These operators are constructed from diagonal Hamiltonians with coefficients learned from data, a structure designed for efficient implementation on quantum hardware. This architecture enables direct multi-step prediction, and the operator's computational complexity scales logarithmically with the observable space dimension. The QKM is validated across diverse nonlinear systems. Its predictions maintain relative errors below 6% for reaction-diffusion systems and shear flows, and capture key statistics in 2D turbulence. This work establishes a practical pathway for quantum-accelerated simulation of nonlinear phenomena, exploring a framework built on the synergy between deep learning for global linearization and quantum algorithms for unitary dynamics evolution. 

---
# Against racing to AGI: Cooperation, deterrence, and catastrophic risks 

**Authors**: Leonard Dung, Max Hellrigel-Holderbaum  

**Link**: [PDF](https://arxiv.org/pdf/2507.21839)  

**Abstract**: AGI Racing is the view that it is in the self-interest of major actors in AI development, especially powerful nations, to accelerate their frontier AI development to build highly capable AI, especially artificial general intelligence (AGI), before competitors have a chance. We argue against AGI Racing. First, the downsides of racing to AGI are much higher than portrayed by this view. Racing to AGI would substantially increase catastrophic risks from AI, including nuclear instability, and undermine the prospects of technical AI safety research to be effective. Second, the expected benefits of racing may be lower than proponents of AGI Racing hold. In particular, it is questionable whether winning the race enables complete domination over losers. Third, international cooperation and coordination, and perhaps carefully crafted deterrence measures, constitute viable alternatives to racing to AGI which have much smaller risks and promise to deliver most of the benefits that racing to AGI is supposed to provide. Hence, racing to AGI is not in anyone's self-interest as other actions, particularly incentivizing and seeking international cooperation around AI issues, are preferable. 

---
# Analysis of Fourier Neural Operators via Effective Field Theory 

**Authors**: Taeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.21833)  

**Abstract**: Fourier Neural Operators (FNOs) have emerged as leading surrogates for high-dimensional partial-differential equations, yet their stability, generalization and frequency behavior lack a principled explanation. We present the first systematic effective-field-theory analysis of FNOs in an infinite-dimensional function space, deriving closed recursion relations for the layer kernel and four-point vertex and then examining three practically important settings-analytic activations, scale-invariant cases and architectures with residual connections. The theory shows that nonlinear activations inevitably couple frequency inputs to high-frequency modes that are otherwise discarded by spectral truncation, and experiments confirm this frequency transfer. For wide networks we obtain explicit criticality conditions on the weight-initialization ensemble that keep small input perturbations to have uniform scale across depth, and empirical tests validate these predictions. Taken together, our results quantify how nonlinearity enables neural operators to capture non-trivial features, supply criteria for hyper-parameter selection via criticality analysis, and explain why scale-invariant activations and residual connections enhance feature learning in FNOs. 

---
# Introducing HALC: A general pipeline for finding optimal prompting strategies for automated coding with LLMs in the computational social sciences 

**Authors**: Andreas Reich, Claudia Thoms, Tobias Schrimpf  

**Link**: [PDF](https://arxiv.org/pdf/2507.21831)  

**Abstract**: LLMs are seeing widespread use for task automation, including automated coding in the social sciences. However, even though researchers have proposed different prompting strategies, their effectiveness varies across LLMs and tasks. Often trial and error practices are still widespread. We propose HALC$-$a general pipeline that allows for the systematic and reliable construction of optimal prompts for any given coding task and model, permitting the integration of any prompting strategy deemed relevant. To investigate LLM coding and validate our pipeline, we sent a total of 1,512 individual prompts to our local LLMs in over two million requests. We test prompting strategies and LLM task performance based on few expert codings (ground truth). When compared to these expert codings, we find prompts that code reliably for single variables (${\alpha}$climate = .76; ${\alpha}$movement = .78) and across two variables (${\alpha}$climate = .71; ${\alpha}$movement = .74) using the LLM Mistral NeMo. Our prompting strategies are set up in a way that aligns the LLM to our codebook$-$we are not optimizing our codebook for LLM friendliness. Our paper provides insights into the effectiveness of different prompting strategies, crucial influencing factors, and the identification of reliable prompts for each coding task and model. 

---
# Unlocking Interpretability for RF Sensing: A Complex-Valued White-Box Transformer 

**Authors**: Xie Zhang, Yina Wang, Chenshu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21799)  

**Abstract**: The empirical success of deep learning has spurred its application to the radio-frequency (RF) domain, leading to significant advances in Deep Wireless Sensing (DWS). However, most existing DWS models function as black boxes with limited interpretability, which hampers their generalizability and raises concerns in security-sensitive physical applications. In this work, inspired by the remarkable advances of white-box transformers, we present RF-CRATE, the first mathematically interpretable deep network architecture for RF sensing, grounded in the principles of complex sparse rate reduction. To accommodate the unique RF signals, we conduct non-trivial theoretical derivations that extend the original real-valued white-box transformer to the complex domain. By leveraging the CR-Calculus framework, we successfully construct a fully complex-valued white-box transformer with theoretically derived self-attention and residual multi-layer perceptron modules. Furthermore, to improve the model's ability to extract discriminative features from limited wireless data, we introduce Subspace Regularization, a novel regularization strategy that enhances feature diversity, resulting in an average performance improvement of 19.98% across multiple sensing tasks. We extensively evaluate RF-CRATE against seven baselines with multiple public and self-collected datasets involving different RF signals. The results show that RF-CRATE achieves performance on par with thoroughly engineered black-box models, while offering full mathematical interpretability. More importantly, by extending CRATE to the complex domain, RF-CRATE yields substantial improvements, achieving an average classification gain of 5.08% and reducing regression error by 10.34% across diverse sensing tasks compared to CRATE. RF-CRATE is fully open-sourced at: this https URL. 

---
# MoDeSuite: Robot Learning Task Suite for Benchmarking Mobile Manipulation with Deformable Objects 

**Authors**: Yuying Zhang, Kevin Sebastian Luck, Francesco Verdoja, Ville Kyrki, Joni Pajarinen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21796)  

**Abstract**: Mobile manipulation is a critical capability for robots operating in diverse, real-world environments. However, manipulating deformable objects and materials remains a major challenge for existing robot learning algorithms. While various benchmarks have been proposed to evaluate manipulation strategies with rigid objects, there is still a notable lack of standardized benchmarks that address mobile manipulation tasks involving deformable objects.
To address this gap, we introduce MoDeSuite, the first Mobile Manipulation Deformable Object task suite, designed specifically for robot learning. MoDeSuite consists of eight distinct mobile manipulation tasks covering both elastic objects and deformable objects, each presenting a unique challenge inspired by real-world robot applications. Success in these tasks requires effective collaboration between the robot's base and manipulator, as well as the ability to exploit the deformability of the objects. To evaluate and demonstrate the use of the proposed benchmark, we train two state-of-the-art reinforcement learning algorithms and two imitation learning algorithms, highlighting the difficulties encountered and showing their performance in simulation. Furthermore, we demonstrate the practical relevance of the suite by deploying the trained policies directly into the real world with the Spot robot, showcasing the potential for sim-to-real transfer. We expect that MoDeSuite will open a novel research domain in mobile manipulation involving deformable objects. Find more details, code, and videos at this https URL. 

---
# Can large language models assist choice modelling? Insights into prompting strategies and current models capabilities 

**Authors**: Georges Sfeir, Gabriel Nova, Stephane Hess, Sander van Cranenburgh  

**Link**: [PDF](https://arxiv.org/pdf/2507.21790)  

**Abstract**: Large Language Models (LLMs) are widely used to support various workflows across different disciplines, yet their potential in choice modelling remains relatively unexplored. This work examines the potential of LLMs as assistive agents in the specification and, where technically feasible, estimation of Multinomial Logit models. We implement a systematic experimental framework involving thirteen versions of six leading LLMs (ChatGPT, Claude, DeepSeek, Gemini, Gemma, and Llama) evaluated under five experimental configurations. These configurations vary along three dimensions: modelling goal (suggesting vs. suggesting and estimating MNLs); prompting strategy (Zero-Shot vs. Chain-of-Thoughts); and information availability (full dataset vs. data dictionary only). Each LLM-suggested specification is implemented, estimated, and evaluated based on goodness-of-fit metrics, behavioural plausibility, and model complexity. Findings reveal that proprietary LLMs can generate valid and behaviourally sound utility specifications, particularly when guided by structured prompts. Open-weight models such as Llama and Gemma struggled to produce meaningful specifications. Claude 4 Sonnet consistently produced the best-fitting and most complex models, while GPT models suggested models with robust and stable modelling outcomes. Some LLMs performed better when provided with just data dictionary, suggesting that limiting raw data access may enhance internal reasoning capabilities. Among all LLMs, GPT o3 was uniquely capable of correctly estimating its own specifications by executing self-generated code. Overall, the results demonstrate both the promise and current limitations of LLMs as assistive agents in choice modelling, not only for model specification but also for supporting modelling decision and estimation, and provide practical guidance for integrating these tools into choice modellers' workflows. 

---
# Proposing a Semantic Movie Recommendation System Enhanced by ChatGPT's NLP Results 

**Authors**: Ali Fallahi, Azam Bastanfard, Amineh Amini, Hadi Saboohi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21770)  

**Abstract**: The importance of recommender systems on the web has grown, especially in the movie industry, with a vast selection of options to watch. To assist users in traversing available items and finding relevant results, recommender systems analyze operational data and investigate users' tastes and habits. Providing highly individualized suggestions can boost user engagement and satisfaction, which is one of the fundamental goals of the movie industry, significantly in online platforms. According to recent studies and research, using knowledge-based techniques and considering the semantic ideas of the textual data is a suitable way to get more appropriate results. This study provides a new method for building a knowledge graph based on semantic information. It uses the ChatGPT, as a large language model, to assess the brief descriptions of movies and extract their tone of voice. Results indicated that using the proposed method may significantly enhance accuracy rather than employing the explicit genres supplied by the publishers. 

---
# Learning Kinetic Monte Carlo stochastic dynamics with Deep Generative Adversarial Networks 

**Authors**: Daniele Lanzoni, Olivier Pierre-Louis, Roberto Bergamaschini, Francesco Montalenti  

**Link**: [PDF](https://arxiv.org/pdf/2507.21763)  

**Abstract**: We show that Generative Adversarial Networks (GANs) may be fruitfully exploited to learn stochastic dynamics, surrogating traditional models while capturing thermal fluctuations. Specifically, we showcase the application to a two-dimensional, many-particle system, focusing on surface-step fluctuations and on the related time-dependent roughness. After the construction of a dataset based on Kinetic Monte Carlo simulations, a conditional GAN is trained to propagate stochastically the state of the system in time, allowing the generation of new sequences with a reduced computational cost. Modifications with respect to standard GANs, which facilitate convergence and increase accuracy, are discussed. The trained network is demonstrated to quantitatively reproduce equilibrium and kinetic properties, including scaling laws, with deviations of a few percent from the exact value. Extrapolation limits and future perspectives are critically discussed. 

---
# LiteFat: Lightweight Spatio-Temporal Graph Learning for Real-Time Driver Fatigue Detection 

**Authors**: Jing Ren, Suyu Ma, Hong Jia, Xiwei Xu, Ivan Lee, Haytham Fayek, Xiaodong Li, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.21756)  

**Abstract**: Detecting driver fatigue is critical for road safety, as drowsy driving remains a leading cause of traffic accidents. Many existing solutions rely on computationally demanding deep learning models, which result in high latency and are unsuitable for embedded robotic devices with limited resources (such as intelligent vehicles/cars) where rapid detection is necessary to prevent accidents. This paper introduces LiteFat, a lightweight spatio-temporal graph learning model designed to detect driver fatigue efficiently while maintaining high accuracy and low computational demands. LiteFat involves converting streaming video data into spatio-temporal graphs (STG) using facial landmark detection, which focuses on key motion patterns and reduces unnecessary data processing. LiteFat uses MobileNet to extract facial features and create a feature matrix for the STG. A lightweight spatio-temporal graph neural network is then employed to identify signs of fatigue with minimal processing and low latency. Experimental results on benchmark datasets show that LiteFat performs competitively while significantly decreasing computational complexity and latency as compared to current state-of-the-art methods. This work enables the development of real-time, resource-efficient human fatigue detection systems that can be implemented upon embedded robotic devices. 

---
# Zero-Shot Machine Unlearning with Proxy Adversarial Data Generation 

**Authors**: Huiqiang Chen, Tianqing Zhu, Xin Yu, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.21738)  

**Abstract**: Machine unlearning aims to remove the influence of specific samples from a trained model. A key challenge in this process is over-unlearning, where the model's performance on the remaining data significantly drops due to the change in the model's parameters. Existing unlearning algorithms depend on the remaining data to prevent this issue. As such, these methods are inapplicable in a more practical scenario, where only the unlearning samples are available (i.e., zero-shot unlearning). This paper presents a novel framework, ZS-PAG, to fill this gap. Our approach offers three key innovations: (1) we approximate the inaccessible remaining data by generating adversarial samples; (2) leveraging the generated samples, we pinpoint a specific subspace to perform the unlearning process, therefore preventing over-unlearning in the challenging zero-shot scenario; and (3) we consider the influence of the unlearning process on the remaining samples and design an influence-based pseudo-labeling strategy. As a result, our method further improves the model's performance after unlearning. The proposed method holds a theoretical guarantee, and experiments on various benchmarks validate the effectiveness and superiority of our proposed method over several baselines. 

---
# Detection Transformers Under the Knife: A Neuroscience-Inspired Approach to Ablations 

**Authors**: Nils Hütten, Florian Hölken, Hasan Tercan, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21723)  

**Abstract**: In recent years, Explainable AI has gained traction as an approach to enhancing model interpretability and transparency, particularly in complex models such as detection transformers. Despite rapid advancements, a substantial research gap remains in understanding the distinct roles of internal components - knowledge that is essential for improving transparency and efficiency. Inspired by neuroscientific ablation studies, which investigate the functions of brain regions through selective impairment, we systematically analyze the impact of ablating key components in three state-of-the-art detection transformer models: Detection transformer (DETR), deformable detection transformer (DDETR), and DETR with improved denoising anchor boxes (DINO). The ablations target query embeddings, encoder and decoder multi-head self-attentions (MHSA) as well as decoder multi-head cross-attention (MHCA) layers. We evaluate the effects of these ablations on the performance metrics gIoU and F1-score, quantifying effects on both the classification and regression sub-tasks on the COCO dataset. To facilitate reproducibility and future research, we publicly release the DeepDissect library. Our findings reveal model-specific resilience patterns: while DETR is particularly sensitive to ablations in encoder MHSA and decoder MHCA, DDETR's multi-scale deformable attention enhances robustness, and DINO exhibits the greatest resilience due to its look-forward twice update rule, which helps distributing knowledge across blocks. These insights also expose structural redundancies, particularly in DDETR's and DINO's decoder MHCA layers, highlighting opportunities for model simplification without sacrificing performance. This study advances XAI for DETRs by clarifying the contributions of internal components to model performance, offering insights to optimize and improve transparency and efficiency in critical applications. 

---
# EnTao-GPM: DNA Foundation Model for Predicting the Germline Pathogenic Mutations 

**Authors**: Zekai Lin, Haoran Sun, Yucheng Guo, Yujie Yang, Yanwen Wang, Bozhen Hu, Chonghang Ye, Qirong Yang, Fan Zhong, Xiaoming Zhang, Lei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21706)  

**Abstract**: Distinguishing pathogenic mutations from benign polymorphisms remains a critical challenge in precision medicine. EnTao-GPM, developed by Fudan University and BioMap, addresses this through three innovations: (1) Cross-species targeted pre-training on disease-relevant mammalian genomes (human, pig, mouse), leveraging evolutionary conservation to enhance interpretation of pathogenic motifs, particularly in non-coding regions; (2) Germline mutation specialization via fine-tuning on ClinVar and HGMD, improving accuracy for both SNVs and non-SNVs; (3) Interpretable clinical framework integrating DNA sequence embeddings with LLM-based statistical explanations to provide actionable insights. Validated against ClinVar, EnTao-GPM demonstrates superior accuracy in mutation classification. It revolutionizes genetic testing by enabling faster, more accurate, and accessible interpretation for clinical diagnostics (e.g., variant assessment, risk identification, personalized treatment) and research, advancing personalized medicine. 

---
# Towards a Large Physics Benchmark 

**Authors**: Kristian G. Barman, Sascha Caron, Faegheh Hasibi, Eugene Shalugin, Yoris Marcet, Johannes Otte, Henk W. de Regt, Merijn Moody  

**Link**: [PDF](https://arxiv.org/pdf/2507.21695)  

**Abstract**: We introduce a benchmark framework developed by and for the scientific community to evaluate, monitor and steer large language model development in fundamental physics. Building on philosophical concepts of scientific understanding and creativity, we develop a scoring system in which each question is scored by an expert for its correctness, difficulty, and surprise. The questions are of three forms: (i) multiple-choice questions for conceptual understanding, (ii) analytical problems requiring mathematical derivation, and (iii) openended tasks requiring complex problem solving. Our current dataset contains diverse set of examples, including a machine learning challenge to classify high-energy physics events, such as the four top quark signal. To ensure continued relevance, we propose a living benchmark, where physicists contribute questions, for instance alongside new publications. We invite contributions via: this http URL. We hope that this benchmark will enable a targeted AI development that can make a meaningful contribution to fundamental physics research. 

---
# A Multi-Agent Generative AI Framework for IC Module-Level Verification Automation 

**Authors**: Wenbo Liu, Forbes Hou, Jon Zhang, Hong Liu, Allen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2507.21694)  

**Abstract**: As large language models demonstrate enormous potential in the field of Electronic Design Automation (EDA), generative AI-assisted chip design is attracting widespread attention from academia and industry. Although these technologies have made preliminary progress in tasks such as code generation, their application in chip verification -- a critical bottleneck in the chip development cycle -- remains at an exploratory stage. This paper proposes an innovative Multi-Agent Verification Framework (MAVF) aimed at addressing the limitations of current single-LLM approaches in complex verification tasks. Our framework builds an automated transformation system from design specifications to testbench through the collaborative work of multiple specialized agents, including specification parsing, verification strategy generation, and code implementation. Through verification experiments on multiple chip modules of varying complexity, results show that MAVF significantly outperforms traditional manual methods and single-dialogue generative AI approaches in verification document parsing and generation, as well as automated testbench generation. This research opens new directions for exploring generative AI applications in verification automation, potentially providing effective approaches to solving the most challenging bottleneck issues in chip design. 

---
# MultiAIGCD: A Comprehensive dataset for AI Generated Code Detection Covering Multiple Languages, Models,Prompts, and Scenarios 

**Authors**: Basak Demirok, Mucahid Kutlu, Selin Mergen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21693)  

**Abstract**: As large language models (LLMs) rapidly advance, their role in code generation has expanded significantly. While this offers streamlined development, it also creates concerns in areas like education and job interviews. Consequently, developing robust systems to detect AI-generated code is imperative to maintain academic integrity and ensure fairness in hiring processes. In this study, we introduce MultiAIGCD, a dataset for AI-generated code detection for Python, Java, and Go. From the CodeNet dataset's problem definitions and human-authored codes, we generate several code samples in Java, Python, and Go with six different LLMs and three different prompts. This generation process covered three key usage scenarios: (i) generating code from problem descriptions, (ii) fixing runtime errors in human-written code, and (iii) correcting incorrect outputs. Overall, MultiAIGCD consists of 121,271 AI-generated and 32,148 human-written code snippets. We also benchmark three state-of-the-art AI-generated code detection models and assess their performance in various test scenarios such as cross-model and cross-language. We share our dataset and codes to support research in this field. 

---
# APT: Improving Diffusion Models for High Resolution Image Generation with Adaptive Path Tracing 

**Authors**: Sangmin Han, Jinho Jeong, Jinwoo Kim, Seon Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.21690)  

**Abstract**: Latent Diffusion Models (LDMs) are generally trained at fixed resolutions, limiting their capability when scaling up to high-resolution images. While training-based approaches address this limitation by training on high-resolution datasets, they require large amounts of data and considerable computational resources, making them less practical. Consequently, training-free methods, particularly patch-based approaches, have become a popular alternative. These methods divide an image into patches and fuse the denoising paths of each patch, showing strong performance on high-resolution generation. However, we observe two critical issues for patch-based approaches, which we call ``patch-level distribution shift" and ``increased patch monotonicity." To address these issues, we propose Adaptive Path Tracing (APT), a framework that combines Statistical Matching to ensure patch distributions remain consistent in upsampled latents and Scale-aware Scheduling to deal with the patch monotonicity. As a result, APT produces clearer and more refined details in high-resolution images. In addition, APT enables a shortcut denoising process, resulting in faster sampling with minimal quality degradation. Our experimental results confirm that APT produces more detailed outputs with improved inference speed, providing a practical approach to high-resolution image generation. 

---
# diffSPH: Differentiable Smoothed Particle Hydrodynamics for Adjoint Optimization and Machine Learning 

**Authors**: Rene Winchenbach, Nils Thuerey  

**Link**: [PDF](https://arxiv.org/pdf/2507.21684)  

**Abstract**: We present diffSPH, a novel open-source differentiable Smoothed Particle Hydrodynamics (SPH) framework developed entirely in PyTorch with GPU acceleration. diffSPH is designed centrally around differentiation to facilitate optimization and machine learning (ML) applications in Computational Fluid Dynamics~(CFD), including training neural networks and the development of hybrid models. Its differentiable SPH core, and schemes for compressible (with shock capturing and multi-phase flows), weakly compressible (with boundary handling and free-surface flows), and incompressible physics, enable a broad range of application areas. We demonstrate the framework's unique capabilities through several applications, including addressing particle shifting via a novel, target-oriented approach by minimizing physical and regularization loss terms, a task often intractable in traditional solvers. Further examples include optimizing initial conditions and physical parameters to match target trajectories, shape optimization, implementing a solver-in-the-loop setup to emulate higher-order integration, and demonstrating gradient propagation through hundreds of full simulation steps. Prioritizing readability, usability, and extensibility, this work offers a foundational platform for the CFD community to develop and deploy novel neural networks and adjoint optimization applications. 

---
# AI Literacy as a Key Driver of User Experience in AI-Powered Assessment: Insights from Socratic Mind 

**Authors**: Meryem Yilmaz Soylu, Jeonghyun Lee, Jui-Tse Hung, Christopher Zhang Cui, David A. Joyner  

**Link**: [PDF](https://arxiv.org/pdf/2507.21654)  

**Abstract**: As Artificial Intelligence (AI) tools become increasingly embedded in higher education, understanding how students interact with these systems is essential to supporting effective learning. This study examines how students' AI literacy and prior exposure to AI technologies shape their perceptions of Socratic Mind, an interactive AI-powered formative assessment tool. Drawing on Self-Determination Theory and user experience research, we analyze relationships among AI literacy, perceived usability, satisfaction, engagement, and perceived learning effectiveness. Data from 309 undergraduates in Computer Science and Business courses were collected through validated surveys. Partial least squares structural equation modeling showed that AI literacy - especially self-efficacy, conceptual understanding, and application skills - significantly predicts usability, satisfaction, and engagement. Usability and satisfaction, in turn, strongly predict perceived learning effectiveness, while prior AI exposure showed no significant effect. These findings highlight that AI literacy, rather than exposure alone, shapes student experiences. Designers should integrate adaptive guidance and user-centered features to support diverse literacy levels, fostering inclusive, motivating, and effective AI-based learning environments. 

---
# DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs 

**Authors**: Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He, Cheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21653)  

**Abstract**: Real-world fraud detection applications benefit from graph learning techniques that jointly exploit node features, often rich in textual data, and graph structural information. Recently, Graph-Enhanced LLMs emerge as a promising graph learning approach that converts graph information into prompts, exploiting LLMs' ability to reason over both textual and structural information. Among them, text-only prompting, which converts graph information to prompts consisting solely of text tokens, offers a solution that relies only on LLM tuning without requiring additional graph-specific encoders. However, text-only prompting struggles on heterogeneous fraud-detection graphs: multi-hop relations expand exponentially with each additional hop, leading to rapidly growing neighborhoods associated with dense textual information. These neighborhoods may overwhelm the model with long, irrelevant content in the prompt and suppress key signals from the target node, thereby degrading performance. To address this challenge, we propose Dual Granularity Prompting (DGP), which mitigates information overload by preserving fine-grained textual details for the target node while summarizing neighbor information into coarse-grained text prompts. DGP introduces tailored summarization strategies for different data modalities, bi-level semantic abstraction for textual fields and statistical aggregation for numerical features, enabling effective compression of verbose neighbor content into concise, informative prompts. Experiments across public and industrial datasets demonstrate that DGP operates within a manageable token budget while improving fraud detection performance by up to 6.8% (AUPRC) over state-of-the-art methods, showing the potential of Graph-Enhanced LLMs for fraud detection. 

---
# GUARD-CAN: Graph-Understanding and Recurrent Architecture for CAN Anomaly Detection 

**Authors**: Hyeong Seon Kim, Huy Kang Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.21640)  

**Abstract**: Modern in-vehicle networks face various cyber threats due to the lack of encryption and authentication in the Controller Area Network (CAN). To address this security issue, this paper presents GUARD-CAN, an anomaly detection framework that combines graph-based representation learning with time-series modeling. GUARD-CAN splits CAN messages into fixed-length windows and converts each window into a graph that preserves message order. To detect anomalies in the timeaware and structure-aware context at the same window, GUARD-CAN takes advantage of the overcomplete Autoencoder (AE) and Graph Convolutional Network (GCN) to generate graph embedding vectors. The model groups these vectors into sequences and feeds them into the Gated Recurrent Unit (GRU) to detect temporal anomaly patterns across the graphs. GUARD-CAN performs anomaly detection at both the sequence level and the window level, and this allows multi-perspective performance evaluation. The model also verifies the importance of window size selection through an analysis based on Shannon entropy. As a result, GUARD-CAN shows that the proposed model detects four types of CAN attacks (flooding, fuzzing, replay and spoofing attacks) effectively without relying on complex feature engineering. 

---
# Hierarchical Graph Neural Network for Compressed Speech Steganalysis 

**Authors**: Mustapha Hemis, Hamza Kheddar, Mohamed Chahine Ghanem, Bachir Boudraa  

**Link**: [PDF](https://arxiv.org/pdf/2507.21591)  

**Abstract**: Steganalysis methods based on deep learning (DL) often struggle with computational complexity and challenges in generalizing across different datasets. Incorporating a graph neural network (GNN) into steganalysis schemes enables the leveraging of relational data for improved detection accuracy and adaptability. This paper presents the first application of a Graph Neural Network (GNN), specifically the GraphSAGE architecture, for steganalysis of compressed voice over IP (VoIP) speech streams. The method involves straightforward graph construction from VoIP streams and employs GraphSAGE to capture hierarchical steganalysis information, including both fine grained details and high level patterns, thereby achieving high detection accuracy. Experimental results demonstrate that the developed approach performs well in uncovering quantization index modulation (QIM)-based steganographic patterns in VoIP signals. It achieves detection accuracy exceeding 98 percent even for short 0.5 second samples, and 95.17 percent accuracy under challenging conditions with low embedding rates, representing an improvement of 2.8 percent over the best performing state of the art methods. Furthermore, the model exhibits superior efficiency, with an average detection time as low as 0.016 seconds for 0.5-second samples an improvement of 0.003 seconds. This makes it efficient for online steganalysis tasks, providing a superior balance between detection accuracy and efficiency under the constraint of short samples with low embedding rates. 

---
# Model Predictive Adversarial Imitation Learning for Planning from Observation 

**Authors**: Tyler Han, Yanda Bao, Bhaumik Mehta, Gabriel Guo, Anubhav Vishwakarma, Emily Kang, Sanghun Jung, Rosario Scalise, Jason Zhou, Bryan Xu, Byron Boots  

**Link**: [PDF](https://arxiv.org/pdf/2507.21533)  

**Abstract**: Human demonstration data is often ambiguous and incomplete, motivating imitation learning approaches that also exhibit reliable planning behavior. A common paradigm to perform planning-from-demonstration involves learning a reward function via Inverse Reinforcement Learning (IRL) then deploying this reward via Model Predictive Control (MPC). Towards unifying these methods, we derive a replacement of the policy in IRL with a planning-based agent. With connections to Adversarial Imitation Learning, this formulation enables end-to-end interactive learning of planners from observation-only demonstrations. In addition to benefits in interpretability, complexity, and safety, we study and observe significant improvements on sample efficiency, out-of-distribution generalization, and robustness. The study includes evaluations in both simulated control benchmarks and real-world navigation experiments using few-to-single observation-only demonstrations. 

---
# Automatic Classification of User Requirements from Online Feedback -- A Replication Study 

**Authors**: Meet Bhatt, Nic Boilard, Muhammad Rehan Chaudhary, Cole Thompson, Jacob Idoko, Aakash Sorathiya, Gouri Ginde  

**Link**: [PDF](https://arxiv.org/pdf/2507.21532)  

**Abstract**: Natural language processing (NLP) techniques have been widely applied in the requirements engineering (RE) field to support tasks such as classification and ambiguity detection. Although RE research is rooted in empirical investigation, it has paid limited attention to replicating NLP for RE (NLP4RE) studies. The rapidly advancing realm of NLP is creating new opportunities for efficient, machine-assisted workflows, which can bring new perspectives and results to the forefront. Thus, we replicate and extend a previous NLP4RE study (baseline), "Classifying User Requirements from Online Feedback in Small Dataset Environments using Deep Learning", which evaluated different deep learning models for requirement classification from user reviews. We reproduced the original results using publicly released source code, thereby helping to strengthen the external validity of the baseline study. We then extended the setup by evaluating model performance on an external dataset and comparing results to a GPT-4o zero-shot classifier. Furthermore, we prepared the replication study ID-card for the baseline study, important for evaluating replication readiness. Results showed diverse reproducibility levels across different models, with Naive Bayes demonstrating perfect reproducibility. In contrast, BERT and other models showed mixed results. Our findings revealed that baseline deep learning models, BERT and ELMo, exhibited good generalization capabilities on an external dataset, and GPT-4o showed performance comparable to traditional baseline machine learning models. Additionally, our assessment confirmed the baseline study's replication readiness; however missing environment setup files would have further enhanced readiness. We include this missing information in our replication package and provide the replication study ID-card for our study to further encourage and support the replication of our study. 

---
# Decision Transformer-Based Drone Trajectory Planning with Dynamic Safety-Efficiency Trade-Offs 

**Authors**: Chang-Hun Ji, SiWoon Song, Youn-Hee Han, SungTae Moon  

**Link**: [PDF](https://arxiv.org/pdf/2507.21506)  

**Abstract**: A drone trajectory planner should be able to dynamically adjust the safety-efficiency trade-off according to varying mission requirements in unknown environments. Although traditional polynomial-based planners offer computational efficiency and smooth trajectory generation, they require expert knowledge to tune multiple parameters to adjust this trade-off. Moreover, even with careful tuning, the resulting adjustment may fail to achieve the desired trade-off. Similarly, although reinforcement learning-based planners are adaptable in unknown environments, they do not explicitly address the safety-efficiency trade-off. To overcome this limitation, we introduce a Decision Transformer-based trajectory planner that leverages a single parameter, Return-to-Go (RTG), as a \emph{temperature parameter} to dynamically adjust the safety-efficiency trade-off. In our framework, since RTG intuitively measures the safety and efficiency of a trajectory, RTG tuning does not require expert knowledge. We validate our approach using Gazebo simulations in both structured grid and unstructured random environments. The experimental results demonstrate that our planner can dynamically adjust the safety-efficiency trade-off by simply tuning the RTG parameter. Furthermore, our planner outperforms existing baseline methods across various RTG settings, generating safer trajectories when tuned for safety and more efficient trajectories when tuned for efficiency. Real-world experiments further confirm the reliability and practicality of our proposed planner. 

---
# Evaluation and Benchmarking of LLM Agents: A Survey 

**Authors**: Mahmoud Mohammadi, Yipeng Li, Jane Lo, Wendy Yip  

**Link**: [PDF](https://arxiv.org/pdf/2507.21504)  

**Abstract**: The rise of LLM-based agents has opened new frontiers in AI applications, yet evaluating these agents remains a complex and underdeveloped area. This survey provides an in-depth overview of the emerging field of LLM agent evaluation, introducing a two-dimensional taxonomy that organizes existing work along (1) evaluation objectives -- what to evaluate, such as agent behavior, capabilities, reliability, and safety -- and (2) evaluation process -- how to evaluate, including interaction modes, datasets and benchmarks, metric computation methods, and tooling. In addition to taxonomy, we highlight enterprise-specific challenges, such as role-based access to data, the need for reliability guarantees, dynamic and long-horizon interactions, and compliance, which are often overlooked in current research. We also identify future research directions, including holistic, more realistic, and scalable evaluation. This work aims to bring clarity to the fragmented landscape of agent evaluation and provide a framework for systematic assessment, enabling researchers and practitioners to evaluate LLM agents for real-world deployment. 

---
# VN-MTEB: Vietnamese Massive Text Embedding Benchmark 

**Authors**: Loc Pham, Tung Luu, Thu Vo, Minh Nguyen, Viet Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21500)  

**Abstract**: Vietnam ranks among the top countries in terms of both internet traffic and online toxicity. As a result, implementing embedding models for recommendation and content control duties in applications is crucial. However, a lack of large-scale test datasets, both in volume and task diversity, makes it tricky for scientists to effectively evaluate AI models before deploying them in real-world, large-scale projects. To solve this important problem, we introduce a Vietnamese benchmark, VN-MTEB for embedding models, which we created by translating a large number of English samples from the Massive Text Embedding Benchmark using our new automated framework. We leverage the strengths of large language models (LLMs) and cutting-edge embedding models to conduct translation and filtering processes to retain high-quality samples, guaranteeing a natural flow of language and semantic fidelity while preserving named entity recognition (NER) and code snippets. Our comprehensive benchmark consists of 41 datasets from six tasks specifically designed for Vietnamese text embeddings. In our analysis, we find that bigger and more complex models using Rotary Positional Embedding outperform those using Absolute Positional Embedding in embedding tasks. Datasets are available at HuggingFace: this https URL 

---
# HLSDebugger: Identification and Correction of Logic Bugs in HLS Code with LLM Solutions 

**Authors**: Jing Wang, Shang Liu, Yao Lu, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.21485)  

**Abstract**: High-level synthesis (HLS) accelerates hardware design by enabling the automatic translation of high-level descriptions into efficient hardware implementations. However, debugging HLS code is a challenging and labor-intensive task, especially for novice circuit designers or software engineers without sufficient hardware domain knowledge. The recent emergence of Large Language Models (LLMs) is promising in automating the HLS debugging process. Despite the great potential, three key challenges persist when applying LLMs to HLS logic debugging: 1) High-quality circuit data for training LLMs is scarce, posing a significant challenge. 2) Debugging logic bugs in hardware is inherently more complex than identifying software bugs with existing golden test cases. 3) The absence of reliable test cases requires multi-tasking solutions, performing both bug identification and correction. complicates the multi-tasking required for effective HLS debugging. In this work, we propose a customized solution named HLSDebugger to address the challenges. HLSDebugger first generates and releases a large labeled dataset with 300K data samples, targeting HLS logic bugs. The HLSDebugger model adopts an encoder-decoder structure, performing bug location identification, bug type prediction, and bug correction with the same model. HLSDebugger significantly outperforms advanced LLMs like GPT-4 in bug identification and by more than 3x in bug correction. It makes a substantial advancement in the exploration of automated debugging of HLS code. 

---
# NCCR: to Evaluate the Robustness of Neural Networks and Adversarial Examples 

**Authors**: Pu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21483)  

**Abstract**: Neural networks have received a lot of attention recently, and related security issues have come with it. Many studies have shown that neural networks are vulnerable to adversarial examples that have been artificially perturbed with modification, which is too small to be distinguishable by human perception. Different attacks and defenses have been proposed to solve these problems, but there is little research on evaluating the robustness of neural networks and their inputs. In this work, we propose a metric called the neuron cover change rate (NCCR) to measure the ability of deep learning models to resist attacks and the stability of adversarial examples. NCCR monitors alterations in the output of specifically chosen neurons when the input is perturbed, and networks with a smaller degree of variation are considered to be more robust. The results of the experiment on image recognition and the speaker recognition model show that our metrics can provide a good assessment of the robustness of neural networks or their inputs. It can also be used to detect whether an input is adversarial or not, as adversarial examples are always less robust. 

---
# Improving Task Diversity in Label Efficient Supervised Finetuning of LLMs 

**Authors**: Abhinav Arabelly, Jagrut Nemade, Robert D Nowak, Jifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21482)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but developing high-performing models for specialized applications often requires substantial human annotation -- a process that is time-consuming, labor-intensive, and expensive. In this paper, we address the label-efficient learning problem for supervised finetuning (SFT) by leveraging task-diversity as a fundamental principle for effective data selection. This is markedly different from existing methods based on the prompt-diversity. Our approach is based on two key observations: 1) task labels for different prompts are often readily available; 2) pre-trained models have significantly varying levels of confidence across tasks. We combine these facts to devise a simple yet effective sampling strategy: we select examples across tasks using an inverse confidence weighting strategy. This produces models comparable to or better than those trained with more complex sampling procedures, while being significantly easier to implement and less computationally intensive. Notably, our experimental results demonstrate that this method can achieve better accuracy than training on the complete dataset (a 4\% increase in MMLU score). Across various annotation budgets and two instruction finetuning datasets, our algorithm consistently performs at or above the level of the best existing methods, while reducing annotation costs by up to 80\%. 

---
# Capacity-Constrained Continual Learning 

**Authors**: Zheng Wen, Doina Precup, Benjamin Van Roy, Satinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.21479)  

**Abstract**: Any agents we can possibly build are subject to capacity constraints, as memory and compute resources are inherently finite. However, comparatively little attention has been dedicated to understanding how agents with limited capacity should allocate their resources for optimal performance. The goal of this paper is to shed some light on this question by studying a simple yet relevant continual learning problem: the capacity-constrained linear-quadratic-Gaussian (LQG) sequential prediction problem. We derive a solution to this problem under appropriate technical conditions. Moreover, for problems that can be decomposed into a set of sub-problems, we also demonstrate how to optimally allocate capacity across these sub-problems in the steady state. We view the results of this paper as a first step in the systematic theoretical study of learning under capacity constraints. 

---
# Which LLMs Get the Joke? Probing Non-STEM Reasoning Abilities with HumorBench 

**Authors**: Reuben Narad, Siddharth Suresh, Jiayi Chen, Pine S.L. Dysart-Bricken, Bob Mankoff, Robert Nowak, Jifan Zhang, Lalit Jain  

**Link**: [PDF](https://arxiv.org/pdf/2507.21476)  

**Abstract**: We present HumorBench, a benchmark designed to evaluate large language models' (LLMs) ability to reason about and explain sophisticated humor in cartoon captions. As reasoning models increasingly saturate existing benchmarks in mathematics and science, novel and challenging evaluations of model intelligence beyond STEM domains are essential. Reasoning is fundamentally involved in text-based humor comprehension, requiring the identification of connections between concepts in cartoons/captions and external cultural references, wordplays, and other mechanisms. HumorBench includes approximately 300 unique cartoon-caption pairs from the New Yorker Caption Contest and this http URL, with expert-annotated evaluation rubrics identifying essential joke elements. LLMs are evaluated based on their explanations towards the humor and abilities in identifying the joke elements. To perform well on this task, models must form and test hypotheses about associations between concepts, potentially backtracking from initial interpretations to arrive at the most plausible explanation. Our extensive benchmarking of current SOTA models reveals three key insights: (1) LLM progress on STEM reasoning transfers effectively to humor comprehension; (2) models trained exclusively on STEM reasoning data still perform well on HumorBench, demonstrating strong transferability of reasoning abilities; and (3) test-time scaling by increasing thinking token budgets yields mixed results across different models in humor reasoning. 

---
# Hebbian Memory-Augmented Recurrent Networks: Engram Neurons in Deep Learning 

**Authors**: Daniel Szelogowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.21474)  

**Abstract**: Despite success across diverse tasks, current artificial recurrent network architectures rely primarily on implicit hidden-state memories, limiting their interpretability and ability to model long-range dependencies. In contrast, biological neural systems employ explicit, associative memory traces (i.e., engrams) strengthened through Hebbian synaptic plasticity and activated sparsely during recall. Motivated by these neurobiological insights, we introduce the Engram Neural Network (ENN), a novel recurrent architecture incorporating an explicit, differentiable memory matrix with Hebbian plasticity and sparse, attention-driven retrieval mechanisms. The ENN explicitly models memory formation and recall through dynamic Hebbian traces, improving transparency and interpretability compared to conventional RNN variants. We evaluate the ENN architecture on three canonical benchmarks: MNIST digit classification, CIFAR-10 image sequence modeling, and WikiText-103 language modeling. Our empirical results demonstrate that the ENN achieves accuracy and generalization performance broadly comparable to classical RNN, GRU, and LSTM architectures, with all models converging to similar accuracy and perplexity on the large-scale WikiText-103 task. At the same time, the ENN offers significant enhancements in interpretability through observable memory dynamics. Hebbian trace visualizations further reveal biologically plausible, structured memory formation processes, validating the potential of neuroscience-inspired mechanisms to inform the development of more interpretable and robust deep learning models. 

---
# Boost Self-Supervised Dataset Distillation via Parameterization, Predefined Augmentation, and Approximation 

**Authors**: Sheng-Feng Yu, Jia-Jiun Yao, Wei-Chen Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21455)  

**Abstract**: Although larger datasets are crucial for training large deep models, the rapid growth of dataset size has brought a significant challenge in terms of considerable training costs, which even results in prohibitive computational expenses. Dataset Distillation becomes a popular technique recently to reduce the dataset size via learning a highly compact set of representative exemplars, where the model trained with these exemplars ideally should have comparable performance with respect to the one trained with the full dataset. While most of existing works upon dataset distillation focus on supervised datasets, we instead aim to distill images and their self-supervisedly trained representations into a distilled set. This procedure, named as Self-Supervised Dataset Distillation, effectively extracts rich information from real datasets, yielding the distilled sets with enhanced cross-architecture generalizability. Particularly, in order to preserve the key characteristics of original dataset more faithfully and compactly, several novel techniques are proposed: 1) we introduce an innovative parameterization upon images and representations via distinct low-dimensional bases, where the base selection for parameterization is experimentally shown to play a crucial role; 2) we tackle the instability induced by the randomness of data augmentation -- a key component in self-supervised learning but being underestimated in the prior work of self-supervised dataset distillation -- by utilizing predetermined augmentations; 3) we further leverage a lightweight network to model the connections among the representations of augmented views from the same image, leading to more compact pairs of distillation. Extensive experiments conducted on various datasets validate the superiority of our approach in terms of distillation efficiency, cross-architecture generalization, and transfer learning performance. 

---
# MemShare: Memory Efficient Inference for Large Reasoning Models through KV Cache Reuse 

**Authors**: Kaiwen Chen, Xin Tan, Minchen Yu, Hong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21433)  

**Abstract**: Large Reasoning Models (LRMs) have achieved significant advances in mathematical reasoning and formal logic tasks. However, their tendency to generate lengthy chain-of-thought sequences leads to substantial memory overhead during inference. We observe that LRMs frequently produce highly similar intermediate reasoning steps, which correspond to similar KV cache states across layers. Motivated by this observation, we propose MemShare, a novel KV cache management approach that effectively reduces memory overhead. MemShare employs a collaborative filtering algorithm to efficiently identify reusable KV cache blocks and enables zero copy cache reuse to significantly reduce memory overhead, improve throughput while maintaining accuracy. Experimental results demonstrate that MemShare delivers up to 84.79\% improvement in throughput while maintaining better accuracy compared to existing KV cache management methods. 

---
# Towards Locally Deployable Fine-Tuned Causal Large Language Models for Mode Choice Behaviour 

**Authors**: Tareq Alsaleh, Bilal Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2507.21432)  

**Abstract**: This study investigates the adoption of open-access, locally deployable causal large language models (LLMs) for travel mode choice prediction and introduces LiTransMC, the first fine-tuned causal LLM developed for this task. We systematically benchmark eleven LLMs (1-12B parameters) across three stated and revealed preference datasets, testing 396 configurations and generating over 79,000 synthetic commuter predictions. Beyond predictive accuracy, we evaluate models generated reasoning using BERTopic for topic modelling and a novel Explanation Strength Index, providing the first structured analysis of how LLMs articulate decision factors in alignment with behavioural theory. LiTransMC, fine-tuned using parameter efficient and loss masking strategy, achieved a weighted F1 score of 0.6845 and a Jensen-Shannon Divergence of 0.000245, surpassing both untuned local models and larger proprietary systems, including GPT-4o with advanced persona inference and embedding-based loading, while also outperforming classical mode choice methods such as discrete choice models and machine learning classifiers for the same dataset. This dual improvement, i.e., high instant-level accuracy and near-perfect distributional calibration, demonstrates the feasibility of creating specialist, locally deployable LLMs that integrate prediction and interpretability. Through combining structured behavioural prediction with natural language reasoning, this work unlocks the potential for conversational, multi-task transport models capable of supporting agent-based simulations, policy testing, and behavioural insight generation. These findings establish a pathway for transforming general purpose LLMs into specialized, explainable tools for transportation research and policy formulation, while maintaining privacy, reducing cost, and broadening access through local deployment. 

---
# MapDiffusion: Generative Diffusion for Vectorized Online HD Map Construction and Uncertainty Estimation in Autonomous Driving 

**Authors**: Thomas Monninger, Zihan Zhang, Zhipeng Mo, Md Zafar Anwar, Steffen Staab, Sihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.21423)  

**Abstract**: Autonomous driving requires an understanding of the static environment from sensor data. Learned Bird's-Eye View (BEV) encoders are commonly used to fuse multiple inputs, and a vector decoder predicts a vectorized map representation from the latent BEV grid. However, traditional map construction models provide deterministic point estimates, failing to capture uncertainty and the inherent ambiguities of real-world environments, such as occlusions and missing lane markings. We propose MapDiffusion, a novel generative approach that leverages the diffusion paradigm to learn the full distribution of possible vectorized maps. Instead of predicting a single deterministic output from learned queries, MapDiffusion iteratively refines randomly initialized queries, conditioned on a BEV latent grid, to generate multiple plausible map samples. This allows aggregating samples to improve prediction accuracy and deriving uncertainty estimates that directly correlate with scene ambiguity. Extensive experiments on the nuScenes dataset demonstrate that MapDiffusion achieves state-of-the-art performance in online map construction, surpassing the baseline by 5% in single-sample performance. We further show that aggregating multiple samples consistently improves performance along the ROC curve, validating the benefit of distribution modeling. Additionally, our uncertainty estimates are significantly higher in occluded areas, reinforcing their value in identifying regions with ambiguous sensor input. By modeling the full map distribution, MapDiffusion enhances the robustness and reliability of online vectorized HD map construction, enabling uncertainty-aware decision-making for autonomous vehicles in complex environments. 

---
# Sync-TVA: A Graph-Attention Framework for Multimodal Emotion Recognition with Cross-Modal Fusion 

**Authors**: Zeyu Deng, Yanhui Lu, Jiashu Liao, Shuang Wu, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.21395)  

**Abstract**: Multimodal emotion recognition (MER) is crucial for enabling emotionally intelligent systems that perceive and respond to human emotions. However, existing methods suffer from limited cross-modal interaction and imbalanced contributions across modalities. To address these issues, we propose Sync-TVA, an end-to-end graph-attention framework featuring modality-specific dynamic enhancement and structured cross-modal fusion. Our design incorporates a dynamic enhancement module for each modality and constructs heterogeneous cross-modal graphs to model semantic relations across text, audio, and visual features. A cross-attention fusion mechanism further aligns multimodal cues for robust emotion inference. Experiments on MELD and IEMOCAP demonstrate consistent improvements over state-of-the-art models in both accuracy and weighted F1 score, especially under class-imbalanced conditions. 

---
# Multimodal LLMs as Customized Reward Models for Text-to-Image Generation 

**Authors**: Shijie Zhou, Ruiyi Zhang, Huaisheng Zhu, Branislav Kveton, Yufan Zhou, Jiuxiang Gu, Jian Chen, Changyou Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21391)  

**Abstract**: We introduce LLaVA-Reward, an efficient reward model designed to automatically evaluate text-to-image (T2I) generations across multiple perspectives, leveraging pretrained multimodal large language models (MLLMs). Existing MLLM-based approaches require instruction-following data for supervised fine-tuning and evaluate generation quality on analyzing text response, which is time-consuming and difficult to train. To address this problem, we propose LLaVA-Reward, which directly utilizes the hidden states of MLLMs given text-image pairs. To enhance the bidirectional interaction between visual and textual representations in decoder-only MLLMs, we further propose adding a Skip-connection Cross Attention (SkipCA) module. This design enhances text-image correlation reasoning by connecting early-layer visual features with later-layer hidden this http URL addition, LLaVA-Reward supports different types of preference data for efficient fine-tuning, including paired preference data and unpaired data. We train LLaVA-Reward on four evaluation perspectives: text-image alignment, fidelity/artifact, safety, and overall ranking. Empirical results demonstrate that LLaVA-Reward outperforms conventional and MLLM-based methods in generating human-aligned scores for automatic evaluations and inference-time scaling in text-to-image generations. 

---
# Efficient Neural Combinatorial Optimization Solver for the Min-max Heterogeneous Capacitated Vehicle Routing Problem 

**Authors**: Xuan Wu, Di Wang, Chunguo Wu, Kaifang Qi, Chunyan Miao, Yubin Xiao, Jian Zhang, You Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.21386)  

**Abstract**: Numerous Neural Combinatorial Optimization (NCO) solvers have been proposed to address Vehicle Routing Problems (VRPs). However, most of these solvers focus exclusively on single-vehicle VRP variants, overlooking the more realistic min-max Heterogeneous Capacitated Vehicle Routing Problem (MMHCVRP), which involves multiple vehicles. Existing MMHCVRP solvers typically select a vehicle and its next node to visit at each decoding step, but often make myopic decoding decisions and overlook key properties of MMHCVRP, including local topological relationships, vehicle permutation invariance, and node symmetry, resulting in suboptimal performance. To better address these limitations, we propose ECHO, an efficient NCO solver. First, ECHO exploits the proposed dual-modality node encoder to capture local topological relationships among nodes. Subsequently, to mitigate myopic decisions, ECHO employs the proposed Parameter-Free Cross-Attention mechanism to prioritize the vehicle selected in the preceding decoding step. Finally, leveraging vehicle permutation invariance and node symmetry, we introduce a tailored data augment strategy for MMHCVRP to stabilize the Reinforcement Learning training process. To assess the performance of ECHO, we conduct extensive experiments. The experimental results demonstrate that ECHO outperforms state-of-the-art NCO solvers across varying numbers of vehicles and nodes, and exhibits well-performing generalization across both scales and distribution patterns. Finally, ablation studies validate the effectiveness of all proposed methods. 

---
# Deep Reinforcement Learning-based Cell DTX/DRX Configuration for Network Energy Saving 

**Authors**: Wei Mao, Lili Wei, Omid Semiari, Shu-ping Yeh, Hosein Nikopour  

**Link**: [PDF](https://arxiv.org/pdf/2507.21385)  

**Abstract**: 3GPP Release 18 cell discontinuous transmission and reception (cell DTX/DRX) is an important new network energy saving feature for 5G. As a time-domain technique, it periodically aggregates the user data transmissions in a given duration of time when the traffic load is not heavy, so that the remaining time can be kept silent and advanced sleep modes (ASM) can be enabled to shut down more radio components and save more energy for the cell. However, inevitably the packet delay is increased, as during the silent period no transmission is allowed. In this paper we study how to configure cell DTX/DRX to optimally balance energy saving and packet delay, so that for delay-sensitive traffic maximum energy saving can be achieved while the degradation of quality of service (QoS) is minimized. As the optimal configuration can be different for different network and traffic conditions, the problem is complex and we resort to deep reinforcement learning (DRL) framework to train an AI agent to solve it. Through careful design of 1) the learning algorithm, which implements a deep Q-network (DQN) on a contextual bandit (CB) model, and 2) the reward function, which utilizes a smooth approximation of a theoretically optimal but discontinuous reward function, we are able to train an AI agent that always tries to select the best possible Cell DTX/DRX configuration under any network and traffic conditions. Simulation results show that compared to the case when cell DTX/DRX is not used, our agent can achieve up to ~45% energy saving depending on the traffic load scenario, while always maintaining no more than ~1% QoS degradation. 

---
# MAAD: Automate Software Architecture Design through Knowledge-Driven Multi-Agent Collaboration 

**Authors**: Ruiyin Li, Yiran Zhang, Xiyu Zhou, Peng Liang, Weisong Sun, Jifeng Xuan, Zhi Jin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21382)  

**Abstract**: Software architecture design is a critical, yet inherently complex and knowledge-intensive phase of software development. It requires deep domain expertise, development experience, architectural knowledge, careful trade-offs among competing quality attributes, and the ability to adapt to evolving requirements. Traditionally, this process is time-consuming and labor-intensive, and relies heavily on architects, often resulting in limited design alternatives, especially under the pressures of agile development. While Large Language Model (LLM)-based agents have shown promising performance across various SE tasks, their application to architecture design remains relatively scarce and requires more exploration, particularly in light of diverse domain knowledge and complex decision-making. To address the challenges, we proposed MAAD (Multi-Agent Architecture Design), an automated framework that employs a knowledge-driven Multi-Agent System (MAS) for architecture design. MAAD orchestrates four specialized agents (i.e., Analyst, Modeler, Designer and Evaluator) to collaboratively interpret requirements specifications and produce architectural blueprints enriched with quality attributes-based evaluation reports. We then evaluated MAAD through a case study and comparative experiments against MetaGPT, a state-of-the-art MAS baseline. Our results show that MAAD's superiority lies in generating comprehensive architectural components and delivering insightful and structured architecture evaluation reports. Feedback from industrial architects across 11 requirements specifications further reinforces MAAD's practical usability. We finally explored the performance of the MAAD framework with three LLMs (GPT-4o, DeepSeek-R1, and Llama 3.3) and found that GPT-4o exhibits better performance in producing architecture design, emphasizing the importance of LLM selection in MAS-driven architecture design. 

---
# ProMemAssist: Exploring Timely Proactive Assistance Through Working Memory Modeling in Multi-Modal Wearable Devices 

**Authors**: Kevin Pu, Ting Zhang, Naveen Sendhilnathan, Sebastian Freitag, Raj Sodhi, Tanya Jonker  

**Link**: [PDF](https://arxiv.org/pdf/2507.21378)  

**Abstract**: Wearable AI systems aim to provide timely assistance in daily life, but existing approaches often rely on user initiation or predefined task knowledge, neglecting users' current mental states. We introduce ProMemAssist, a smart glasses system that models a user's working memory (WM) in real-time using multi-modal sensor signals. Grounded in cognitive theories of WM, our system represents perceived information as memory items and episodes with encoding mechanisms, such as displacement and interference. This WM model informs a timing predictor that balances the value of assistance with the cost of interruption. In a user study with 12 participants completing cognitively demanding tasks, ProMemAssist delivered more selective assistance and received higher engagement compared to an LLM baseline system. Qualitative feedback highlights the benefits of WM modeling for nuanced, context-sensitive support, offering design implications for more attentive and user-aware proactive agents. 

---
# Evaluating Deep Learning Models for African Wildlife Image Classification: From DenseNet to Vision Transformers 

**Authors**: Lukman Jibril Aliyu, Umar Sani Muhammad, Bilqisu Ismail, Nasiru Muhammad, Almustapha A Wakili, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad, Mustapha Abdullahi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21364)  

**Abstract**: Wildlife populations in Africa face severe threats, with vertebrate numbers declining by over 65% in the past five decades. In response, image classification using deep learning has emerged as a promising tool for biodiversity monitoring and conservation. This paper presents a comparative study of deep learning models for automatically classifying African wildlife images, focusing on transfer learning with frozen feature extractors. Using a public dataset of four species: buffalo, elephant, rhinoceros, and zebra; we evaluate the performance of DenseNet-201, ResNet-152, EfficientNet-B4, and Vision Transformer ViT-H/14. DenseNet-201 achieved the best performance among convolutional networks (67% accuracy), while ViT-H/14 achieved the highest overall accuracy (99%), but with significantly higher computational cost, raising deployment concerns. Our experiments highlight the trade-offs between accuracy, resource requirements, and deployability. The best-performing CNN (DenseNet-201) was integrated into a Hugging Face Gradio Space for real-time field use, demonstrating the feasibility of deploying lightweight models in conservation settings. This work contributes to African-grounded AI research by offering practical insights into model selection, dataset preparation, and responsible deployment of deep learning tools for wildlife conservation. 

---
# StructText: A Synthetic Table-to-Text Approach for Benchmark Generation with Multi-Dimensional Evaluation 

**Authors**: Satyananda Kashyap, Sola Shirai, Nandana Mihindukulasooriya, Horst Samulowitz  

**Link**: [PDF](https://arxiv.org/pdf/2507.21340)  

**Abstract**: Extracting structured information from text, such as key-value pairs that could augment tabular data, is quite useful in many enterprise use cases. Although large language models (LLMs) have enabled numerous automated pipelines for converting natural language into structured formats, there is still a lack of benchmarks for evaluating their extraction quality, especially in specific domains or focused documents specific to a given organization. Building such benchmarks by manual annotations is labour-intensive and limits the size and scalability of the benchmarks. In this work, we present StructText, an end-to-end framework for automatically generating high-fidelity benchmarks for key-value extraction from text using existing tabular data. It uses available tabular data as structured ground truth, and follows a two-stage ``plan-then-execute'' pipeline to synthetically generate corresponding natural-language text. To ensure alignment between text and structured source, we introduce a multi-dimensional evaluation strategy that combines (a) LLM-based judgments on factuality, hallucination, and coherence and (b) objective extraction metrics measuring numeric and temporal accuracy. We evaluated the proposed method on 71,539 examples across 49 datasets. Results reveal that while LLMs achieve strong factual accuracy and avoid hallucination, they struggle with narrative coherence in producing extractable text. Notably, models presume numerical and temporal information with high fidelity yet this information becomes embedded in narratives that resist automated extraction. We release a framework, including datasets, evaluation tools, and baseline extraction systems, to support continued research. 

---
# Semantic Numeration Systems as Dynamical Systems 

**Authors**: Alexander Yu. Chunikhin  

**Link**: [PDF](https://arxiv.org/pdf/2507.21295)  

**Abstract**: The foundational concepts of semantic numeration systems theory are briefly outlined. The action of cardinal semantic operators unfolds over a set of cardinal abstract entities belonging to the cardinal semantic multeity. The cardinal abstract object (CAO) formed by them in a certain connectivity topology is proposed to be considered as a linear discrete dynamical system with nonlinear control. Under the assumption of ideal observability, the CAO state equations are provided for both stationary and non-stationary cases. The fundamental role of the configuration matrix, which combines information about the types of cardinal semantic operators in the CAO, their parameters and topology of connectivity, is demonstrated. 

---
# Learning Simulatable Models of Cloth with Spatially-varying Constitutive Properties 

**Authors**: Guanxiong Chen, Shashwat Suri, Yuhao Wu, Etienne Voulga, David I.W. Levin, Dinesh Pai  

**Link**: [PDF](https://arxiv.org/pdf/2507.21288)  

**Abstract**: Materials used in real clothing exhibit remarkable complexity and spatial variation due to common processes such as stitching, hemming, dyeing, printing, padding, and bonding. Simulating these materials, for instance using finite element methods, is often computationally demanding and slow. Worse, such methods can suffer from numerical artifacts called ``membrane locking'' that makes cloth appear artificially stiff. Here we propose a general framework, called Mass-Spring Net, for learning a simple yet efficient surrogate model that captures the effects of these complex materials using only motion observations. The cloth is discretized into a mass-spring network with unknown material parameters that are learned directly from the motion data, using a novel force-and-impulse loss function. Our approach demonstrates the ability to accurately model spatially varying material properties from a variety of data sources, and immunity to membrane locking which plagues FEM-based simulations. Compared to graph-based networks and neural ODE-based architectures, our method achieves significantly faster training times, higher reconstruction accuracy, and improved generalization to novel dynamic scenarios. 

---
# Adaptive Multimodal Protein Plug-and-Play with Diffusion-Based Priors 

**Authors**: Amartya Banerjee, Xingyu Xu, Caroline Moosmüller, Harlin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.21260)  

**Abstract**: In an inverse problem, the goal is to recover an unknown parameter (e.g., an image) that has typically undergone some lossy or noisy transformation during measurement. Recently, deep generative models, particularly diffusion models, have emerged as powerful priors for protein structure generation. However, integrating noisy experimental data from multiple sources to guide these models remains a significant challenge. Existing methods often require precise knowledge of experimental noise levels and manually tuned weights for each data modality. In this work, we introduce Adam-PnP, a Plug-and-Play framework that guides a pre-trained protein diffusion model using gradients from multiple, heterogeneous experimental sources. Our framework features an adaptive noise estimation scheme and a dynamic modality weighting mechanism integrated into the diffusion process, which reduce the need for manual hyperparameter tuning. Experiments on complex reconstruction tasks demonstrate significantly improved accuracy using Adam-PnP. 

---
# On Explaining Visual Captioning with Hybrid Markov Logic Networks 

**Authors**: Monika Shah, Somdeb Sarkhel, Deepak Venugopal  

**Link**: [PDF](https://arxiv.org/pdf/2507.21246)  

**Abstract**: Deep Neural Networks (DNNs) have made tremendous progress in multimodal tasks such as image captioning. However, explaining/interpreting how these models integrate visual information, language information and knowledge representation to generate meaningful captions remains a challenging problem. Standard metrics to measure performance typically rely on comparing generated captions with human-written ones that may not provide a user with a deep insights into this integration. In this work, we develop a novel explanation framework that is easily interpretable based on Hybrid Markov Logic Networks (HMLNs) - a language that can combine symbolic rules with real-valued functions - where we hypothesize how relevant examples from the training data could have influenced the generation of the observed caption. To do this, we learn a HMLN distribution over the training instances and infer the shift in distributions over these instances when we condition on the generated sample which allows us to quantify which examples may have been a source of richer information to generate the observed caption. Our experiments on captions generated for several state-of-the-art captioning models using Amazon Mechanical Turk illustrate the interpretability of our explanations, and allow us to compare these models along the dimension of explainability. 

---
# Bubbleformer: Forecasting Boiling with Transformers 

**Authors**: Sheikh Md Shakeel Hassan, Xianwei Zou, Akash Dhruv, Vishwanath Ganesan, Aparna Chandramowlishwaran  

**Link**: [PDF](https://arxiv.org/pdf/2507.21244)  

**Abstract**: Modeling boiling (an inherently chaotic, multiphase process central to energy and thermal systems) remains a significant challenge for neural PDE surrogates. Existing models require future input (e.g., bubble positions) during inference because they fail to learn nucleation from past states, limiting their ability to autonomously forecast boiling dynamics. They also fail to model flow boiling velocity fields, where sharp interface-momentum coupling demands long-range and directional inductive biases. We introduce Bubbleformer, a transformer-based spatiotemporal model that forecasts stable and long-range boiling dynamics including nucleation, interface evolution, and heat transfer without dependence on simulation data during inference. Bubbleformer integrates factorized axial attention, frequency-aware scaling, and conditions on thermophysical parameters to generalize across fluids, geometries, and operating conditions. To evaluate physical fidelity in chaotic systems, we propose interpretable physics-based metrics that evaluate heat-flux consistency, interface geometry, and mass conservation. We also release BubbleML 2.0, a high-fidelity dataset that spans diverse working fluids (cryogens, refrigerants, dielectrics), boiling configurations (pool and flow boiling), flow regimes (bubbly, slug, annular), and boundary conditions. Bubbleformer sets new benchmark results in both prediction and forecasting of two-phase boiling flows. 

---
# Learning from Limited and Imperfect Data 

**Authors**: Harsh Rangwani  

**Link**: [PDF](https://arxiv.org/pdf/2507.21205)  

**Abstract**: The distribution of data in the world (eg, internet, etc.) significantly differs from the well-curated datasets and is often over-populated with samples from common categories. The algorithms designed for well-curated datasets perform suboptimally when used for learning from imperfect datasets with long-tailed imbalances and distribution shifts. To expand the use of deep models, it is essential to overcome the labor-intensive curation process by developing robust algorithms that can learn from diverse, real-world data distributions. Toward this goal, we develop practical algorithms for Deep Neural Networks which can learn from limited and imperfect data present in the real world. This thesis is divided into four segments, each covering a scenario of learning from limited or imperfect data. The first part of the thesis focuses on Learning Generative Models from Long-Tail Data, where we mitigate the mode-collapse and enable diverse aesthetic image generations for tail (minority) classes. In the second part, we enable effective generalization on tail classes through Inductive Regularization schemes, which allow tail classes to generalize as effectively as the head classes without requiring explicit generation of images. In the third part, we develop algorithms for Optimizing Relevant Metrics for learning from long-tailed data with limited annotation (semi-supervised), followed by the fourth part, which focuses on the Efficient Domain Adaptation of the model to various domains with very few to zero labeled samples. 

---
# Advancing Compositional LLM Reasoning with Structured Task Relations in Interactive Multimodal Communications 

**Authors**: Xinye Cao, Hongcan Guo, Guoshun Nan, Jiaoyang Cui, Haoting Qian, Yihan Lin, Yilin Peng, Diyang Zhang, Yanzhao Hou, Huici Wu, Xiaofeng Tao, Tony Q.S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2507.21199)  

**Abstract**: Interactive multimodal applications (IMAs), such as route planning in the Internet of Vehicles, enrich users' personalized experiences by integrating various forms of data over wireless networks. Recent advances in large language models (LLMs) utilize mixture-of-experts (MoE) mechanisms to empower multiple IMAs, with each LLM trained individually for a specific task that presents different business workflows. In contrast to existing approaches that rely on multiple LLMs for IMAs, this paper presents a novel paradigm that accomplishes various IMAs using a single compositional LLM over wireless networks. The two primary challenges include 1) guiding a single LLM to adapt to diverse IMA objectives and 2) ensuring the flexibility and efficiency of the LLM in resource-constrained mobile environments. To tackle the first challenge, we propose ContextLoRA, a novel method that guides an LLM to learn the rich structured context among IMAs by constructing a task dependency graph. We partition the learnable parameter matrix of neural layers for each IMA to facilitate LLM composition. Then, we develop a step-by-step fine-tuning procedure guided by task relations, including training, freezing, and masking phases. This allows the LLM to learn to reason among tasks for better adaptation, capturing the latent dependencies between tasks. For the second challenge, we introduce ContextGear, a scheduling strategy to optimize the training procedure of ContextLoRA, aiming to minimize computational and communication costs through a strategic grouping mechanism. Experiments on three benchmarks show the superiority of the proposed ContextLoRA and ContextGear. Furthermore, we prototype our proposed paradigm on a real-world wireless testbed, demonstrating its practical applicability for various IMAs. We will release our code to the community. 

---
# Uncovering Gradient Inversion Risks in Practical Language Model Training 

**Authors**: Xinguo Feng, Zhongkui Ma, Zihan Wang, Eu Joe Chegne, Mengyao Ma, Alsharif Abuadbba, Guangdong Bai  

**Link**: [PDF](https://arxiv.org/pdf/2507.21198)  

**Abstract**: The gradient inversion attack has been demonstrated as a significant privacy threat to federated learning (FL), particularly in continuous domains such as vision models. In contrast, it is often considered less effective or highly dependent on impractical training settings when applied to language models, due to the challenges posed by the discrete nature of tokens in text data. As a result, its potential privacy threats remain largely underestimated, despite FL being an emerging training method for language models. In this work, we propose a domain-specific gradient inversion attack named Grab (gradient inversion with hybrid optimization). Grab features two alternating optimization processes to address the challenges caused by practical training settings, including a simultaneous optimization on dropout masks between layers for improved token recovery and a discrete optimization for effective token sequencing. Grab can recover a significant portion (up to 92.9% recovery rate) of the private training data, outperforming the attack strategy of utilizing discrete optimization with an auxiliary model by notable improvements of up to 28.9% recovery rate in benchmark settings and 48.5% recovery rate in practical settings. Grab provides a valuable step forward in understanding this privacy threat in the emerging FL training mode of language models. 

---
# EdgeAgentX-DT: Integrating Digital Twins and Generative AI for Resilient Edge Intelligence in Tactical Networks 

**Authors**: Abir Ray  

**Link**: [PDF](https://arxiv.org/pdf/2507.21196)  

**Abstract**: We introduce EdgeAgentX-DT, an advanced extension of the EdgeAgentX framework that integrates digital twin simulations and generative AI-driven scenario training to significantly enhance edge intelligence in military networks. EdgeAgentX-DT utilizes network digital twins, virtual replicas synchronized with real-world edge devices, to provide a secure, realistic environment for training and validation. Leveraging generative AI methods, such as diffusion models and transformers, the system creates diverse and adversarial scenarios for robust simulation-based agent training. Our multi-layer architecture includes: (1) on-device edge intelligence; (2) digital twin synchronization; and (3) generative scenario training. Experimental simulations demonstrate notable improvements over EdgeAgentX, including faster learning convergence, higher network throughput, reduced latency, and improved resilience against jamming and node failures. A case study involving a complex tactical scenario with simultaneous jamming attacks, agent failures, and increased network loads illustrates how EdgeAgentX-DT sustains operational performance, whereas baseline methods fail. These results highlight the potential of digital-twin-enabled generative training to strengthen edge AI deployments in contested environments. 

---
# MaXsive: High-Capacity and Robust Training-Free Generative Image Watermarking in Diffusion Models 

**Authors**: Po-Yuan Mao, Cheng-Chang Tsai, Chun-Shien Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21195)  

**Abstract**: The great success of the diffusion model in image synthesis led to the release of gigantic commercial models, raising the issue of copyright protection and inappropriate content generation. Training-free diffusion watermarking provides a low-cost solution for these issues. However, the prior works remain vulnerable to rotation, scaling, and translation (RST) attacks. Although some methods employ meticulously designed patterns to mitigate this issue, they often reduce watermark capacity, which can result in identity (ID) collusion. To address these problems, we propose MaXsive, a training-free diffusion model generative watermarking technique that has high capacity and robustness. MaXsive best utilizes the initial noise to watermark the diffusion model. Moreover, instead of using a meticulously repetitive ring pattern, we propose injecting the X-shape template to recover the RST distortions. This design significantly increases robustness without losing any capacity, making ID collusion less likely to happen. The effectiveness of MaXsive has been verified on two well-known watermarking benchmarks under the scenarios of verification and identification. 

---
# Embeddings to Diagnosis: Latent Fragility under Agentic Perturbations in Clinical LLMs 

**Authors**: Raj Krishnan Vijayaraj  

**Link**: [PDF](https://arxiv.org/pdf/2507.21188)  

**Abstract**: LLMs for clinical decision support often fail under small but clinically meaningful input shifts such as masking a symptom or negating a finding, despite high performance on static benchmarks. These reasoning failures frequently go undetected by standard NLP metrics, which are insensitive to latent representation shifts that drive diagnosis instability. We propose a geometry-aware evaluation framework, LAPD (Latent Agentic Perturbation Diagnostics), which systematically probes the latent robustness of clinical LLMs under structured adversarial edits. Within this framework, we introduce Latent Diagnosis Flip Rate (LDFR), a model-agnostic diagnostic signal that captures representational instability when embeddings cross decision boundaries in PCA-reduced latent space. Clinical notes are generated using a structured prompting pipeline grounded in diagnostic reasoning, then perturbed along four axes: masking, negation, synonym replacement, and numeric variation to simulate common ambiguities and omissions. We compute LDFR across both foundation and clinical LLMs, finding that latent fragility emerges even under minimal surface-level changes. Finally, we validate our findings on 90 real clinical notes from the DiReCT benchmark (MIMIC-IV), confirming the generalizability of LDFR beyond synthetic settings. Our results reveal a persistent gap between surface robustness and semantic stability, underscoring the importance of geometry-aware auditing in safety-critical clinical AI. 

---
# Contrast-CAT: Contrasting Activations for Enhanced Interpretability in Transformer-based Text Classifiers 

**Authors**: Sungmin Han, Jeonghyun Lee, Sangkyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.21186)  

**Abstract**: Transformers have profoundly influenced AI research, but explaining their decisions remains challenging -- even for relatively simpler tasks such as classification -- which hinders trust and safe deployment in real-world applications. Although activation-based attribution methods effectively explain transformer-based text classification models, our findings reveal that these methods can be undermined by class-irrelevant features within activations, leading to less reliable interpretations. To address this limitation, we propose Contrast-CAT, a novel activation contrast-based attribution method that refines token-level attributions by filtering out class-irrelevant features. By contrasting the activations of an input sequence with reference activations, Contrast-CAT generates clearer and more faithful attribution maps. Experimental results across various datasets and models confirm that Contrast-CAT consistently outperforms state-of-the-art methods. Notably, under the MoRF setting, it achieves average improvements of x1.30 in AOPC and x2.25 in LOdds over the most competing methods, demonstrating its effectiveness in enhancing interpretability for transformer-based text classification. 

---
# EvoSLD: Automated Neural Scaling Law Discovery With Large Language Models 

**Authors**: Haowei Lin, Xiangyu Wang, Jianzhu Ma, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21184)  

**Abstract**: Scaling laws are fundamental mathematical relationships that predict how neural network performance evolves with changes in variables such as model size, dataset size, and computational resources. Traditionally, discovering these laws requires extensive human expertise and manual experimentation. We introduce EvoSLD, an automated framework for Scaling Law Discovery (SLD) that leverages evolutionary algorithms guided by Large Language Models (LLMs) to co-evolve symbolic expressions and their optimization routines. Formulated to handle scaling variables, control variables, and response metrics across diverse experimental settings, EvoSLD searches for parsimonious, universal functional forms that minimize fitting errors on grouped data subsets. Evaluated on five real-world scenarios from recent literature, EvoSLD rediscovers exact human-derived laws in two cases and surpasses them in others, achieving up to orders-of-magnitude reductions in normalized mean squared error on held-out test sets. Compared to baselines like symbolic regression and ablated variants, EvoSLD demonstrates superior accuracy, interpretability, and efficiency, highlighting its potential to accelerate AI research. Code is available at this https URL. 

---
# MaPPO: Maximum a Posteriori Preference Optimization with Prior Knowledge 

**Authors**: Guangchen Lan, Sipeng Zhang, Tianle Wang, Yuwei Zhang, Daoan Zhang, Xinpeng Wei, Xiaoman Pan, Hongming Zhang, Dong-Jun Han, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2507.21183)  

**Abstract**: As the era of large language models (LLMs) on behalf of users unfolds, Preference Optimization (PO) methods have become a central approach to aligning LLMs with human preferences and improving performance. We propose Maximum a Posteriori Preference Optimization (MaPPO), a framework for learning from preferences that explicitly incorporates prior reward knowledge into the optimization objective. While existing methods such as Direct Preference Optimization (DPO) and its variants treat preference learning as a Maximum Likelihood Estimation (MLE) problem, MaPPO extends this paradigm by integrating prior reward estimates into a principled Maximum a Posteriori (MaP) objective. This not only generalizes DPO and its variants, but also enhances alignment by mitigating the oversimplified binary classification of responses. More importantly, MaPPO introduces no additional hyperparameter, and supports preference optimization in both offline and online settings. In addition, MaPPO can be used as a plugin with consistent improvement on DPO variants, including widely used SimPO, IPO, and CPO. Extensive empirical evaluations of different model sizes and model series on three standard benchmarks, including MT-Bench, AlpacaEval 2.0, and Arena-Hard, demonstrate consistent improvements in alignment performance without sacrificing computational efficiency. 

---
# SDD: Self-Degraded Defense against Malicious Fine-tuning 

**Authors**: Zixuan Chen, Weikai Lu, Xin Lin, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.21182)  

**Abstract**: Open-source Large Language Models (LLMs) often employ safety alignment methods to resist harmful instructions. However, recent research shows that maliciously fine-tuning these LLMs on harmful data can easily bypass these safeguards. To counter this, we theoretically uncover why malicious fine-tuning succeeds and identify potential defense strategies. Building on the theoretical analysis, we introduce the Self-Degraded Defense (SDD) framework. SDD encourages LLMs to produce high-quality but irrelevant responses to harmful prompts. When attackers attempt malicious fine-tuning, the general capability of the LLM aligned by SDD will significantly decrease, rendering it incapable of following harmful instructions. Our experimental results confirm SDD's effectiveness against such attacks. 

---
# LLM-Adapted Interpretation Framework for Machine Learning Models 

**Authors**: Yuqi Jin, Zihan Hu, Weiteng Zhang, Weihao Xie, Jianwei Shuai, Xian Shen, Zhen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.21179)  

**Abstract**: Background & Aims: High-performance machine learning models like XGBoost are often "black boxes," limiting their clinical adoption due to a lack of interpretability. This study aims to bridge the gap between predictive accuracy and narrative transparency for sarcopenia risk assessment. Methods: We propose the LLM-Adapted Interpretation Framework (LAI-ML), a novel knowledge distillation architecture. LAI-ML transforms feature attributions from a trained XGBoost model into a probabilistic format using specialized techniques (HAGA and CACS). A Large Language Model (LLM), guided by a reinforcement learning loop and case-based retrieval, then generates data-faithful diagnostic narratives. Results: The LAI-ML framework achieved 83% prediction accuracy, significantly outperforming the baseline XGBoost model, 13% higher. Notably, the LLM not only replicated the teacher model's logic but also corrected its predictions in 21.7% of discordant cases, demonstrating enhanced reasoning. Conclusion: LAI-ML effectively translates opaque model predictions into trustworthy and interpretable clinical insights, offering a deployable solution to the "black-box" problem in medical AI. 

---
# A ChatGPT-based approach for questions generation in higher education 

**Authors**: Sinh Trong Vu, Huong Thu Truong, Oanh Tien Do, Tu Anh Le, Tai Tan Mai  

**Link**: [PDF](https://arxiv.org/pdf/2507.21174)  

**Abstract**: Large language models have been widely applied in many aspects of real life, bringing significant efficiency to businesses and offering distinctive user experiences. In this paper, we focus on exploring the application of ChatGPT, a chatbot based on a large language model, to support higher educator in generating quiz questions and assessing learners. Specifically, we explore interactive prompting patterns to design an optimal AI-powered question bank creation process. The generated questions are evaluated through a "Blind test" survey sent to various stakeholders including lecturers and learners. Initial results at the Banking Academy of Vietnam are relatively promising, suggesting a potential direction to streamline the time and effort involved in assessing learners at higher education institutes. 

---
# OneShield -- the Next Generation of LLM Guardrails 

**Authors**: Chad DeLuca, Anna Lisa Gentile, Shubhi Asthana, Bing Zhang, Pawan Chowdhary, Kellen Cheng, Basel Shbita, Pengyuan Li, Guang-Jie Ren, Sandeep Gopisetty  

**Link**: [PDF](https://arxiv.org/pdf/2507.21170)  

**Abstract**: The rise of Large Language Models has created a general excitement about the great potential for a myriad of applications. While LLMs offer many possibilities, questions about safety, privacy, and ethics have emerged, and all the key actors are working to address these issues with protective measures for their own models and standalone solutions. The constantly evolving nature of LLMs makes the task of universally shielding users against their potential risks extremely challenging, and one-size-fits-all solutions unfeasible. In this work, we propose OneShield, our stand-alone, model-agnostic and customizable solution to safeguard LLMs. OneShield aims to provide facilities for defining risk factors, expressing and declaring contextual safety and compliance policies, and mitigating LLM risks, with a focus on each specific customer. We describe the implementation of the framework, the scalability considerations and provide usage statistics of OneShield since its first deployment. 

---
# Trustworthy AI: UK Air Traffic Control Revisited 

**Authors**: Rob Procter, Mark Rouncefield  

**Link**: [PDF](https://arxiv.org/pdf/2507.21169)  

**Abstract**: Exploring the socio-technical challenges confronting the adoption of AI in organisational settings is something that has so far been largely absent from the related literature. In particular, research into requirements for trustworthy AI typically overlooks how people deal with the problems of trust in the tools that they use as part of their everyday work practices. This article presents some findings from an ongoing ethnographic study of how current tools are used in air traffic control work and what it reveals about requirements for trustworthy AI in air traffic control and other safety-critical application domains. 

---
# Diverse LLMs or Diverse Question Interpretations? That is the Ensembling Question 

**Authors**: Rafael Rosales, Santiago Miret  

**Link**: [PDF](https://arxiv.org/pdf/2507.21168)  

**Abstract**: Effectively leveraging diversity has been shown to improve performance for various machine learning models, including large language models (LLMs). However, determining the most effective way of using diversity remains a challenge. In this work, we compare two diversity approaches for answering binary questions using LLMs: model diversity, which relies on multiple models answering the same question, and question interpretation diversity, which relies on using the same model to answer the same question framed in different ways. For both cases, we apply majority voting as the ensemble consensus heuristic to determine the final answer. Our experiments on boolq, strategyqa, and pubmedqa show that question interpretation diversity consistently leads to better ensemble accuracy compared to model diversity. Furthermore, our analysis of GPT and LLaMa shows that model diversity typically produces results between the best and the worst ensemble members without clear improvement. 

---
# ChartM$^3$: Benchmarking Chart Editing with Multimodal Instructions 

**Authors**: Danglu Yang, Liang Zhang, Zihao Yue, Liangyu Chen, Yichen Xu, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.21167)  

**Abstract**: Charts are a fundamental visualization format widely used in data analysis across research and industry. While enabling users to edit charts based on high-level intentions is of great practical value, existing methods primarily rely on natural language instructions, which are often too ambiguous to support fine-grained editing. In this work, we introduce a novel paradigm for multimodal chart editing, where user intent is expressed through a combination of natural language and visual indicators that explicitly highlight the elements to be modified. To support this paradigm, we present Chart$\text{M}^3$, a new benchmark for Multimodal chart editing with Multi-level complexity and Multi-perspective evaluation. Chart$\text{M}^3$ contains 1,000 samples spanning four levels of editing difficulty. Each sample includes triplets in the form of (chart, code, multimodal instructions). To comprehensively evaluate chart editing models, Chart$\text{M}^3$ provides metrics that assess both visual appearance and code correctness. Our benchmark reveals significant limitations in current multimodal large language models (MLLMs), including GPT-4o, particularly in their ability to interpret and act on visual indicators. To address this, we construct Chart$\text{M}^3$-Train, a large-scale training set with 24,000 multimodal chart editing samples. Fine-tuning MLLMs on this dataset leads to substantial improvements, demonstrating the importance of multimodal supervision in building practical chart editing systems. Our datasets, codes, and evaluation tools are available at this https URL. %this https URL datasets, codes, and evaluation tools are available at this https URL. 

---
# AGORA: Incentivizing Group Emergence Capability in LLMs via Group Distillation 

**Authors**: Ren Zhuang, Ben Wang, Shuifa Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.21166)  

**Abstract**: Progress in complex reasoning is constrained by the static nature of the current training datasets. We propose structured interaction as a new scaling axis, moving beyond the prevailing paradigm of increasing model parameters. Our self-evolving framework, AGORA, enables a collaborative ensemble to achieve reasoning performance exceeding state-of-the-art monolithic systems by up to 4.45 percentage points on challenging mathematical benchmarks. This gain stems from group emergent ability-the synthesis of collective capabilities unattainable by isolated models, validating interaction as a scalable driver of intelligence. Our results position the engineering of collaborative ecosystems as a vital frontier for capability emergence. 

---
# OCSVM-Guided Representation Learning for Unsupervised Anomaly Detection 

**Authors**: Nicolas Pinon, Carole Lartizien  

**Link**: [PDF](https://arxiv.org/pdf/2507.21164)  

**Abstract**: Unsupervised anomaly detection (UAD) aims to detect anomalies without labeled data, a necessity in many machine learning applications where anomalous samples are rare or not available. Most state-of-the-art methods fall into two categories: reconstruction-based approaches, which often reconstruct anomalies too well, and decoupled representation learning with density estimators, which can suffer from suboptimal feature spaces. While some recent methods attempt to couple feature learning and anomaly detection, they often rely on surrogate objectives, restrict kernel choices, or introduce approximations that limit their expressiveness and robustness. To address this challenge, we propose a novel method that tightly couples representation learning with an analytically solvable one-class SVM (OCSVM), through a custom loss formulation that directly aligns latent features with the OCSVM decision boundary. The model is evaluated on two tasks: a new benchmark based on MNIST-C, and a challenging brain MRI subtle lesion detection task. Unlike most methods that focus on large, hyperintense lesions at the image level, our approach succeeds to target small, non-hyperintense lesions, while we evaluate voxel-wise metrics, addressing a more clinically relevant scenario. Both experiments evaluate a form of robustness to domain shifts, including corruption types in MNIST-C and scanner/age variations in MRI. Results demonstrate performance and robustness of our proposed mode,highlighting its potential for general UAD and real-world medical imaging applications. The source code is available at this https URL 

---
# Generating Adversarial Point Clouds Using Diffusion Model 

**Authors**: Ruiyang Zhao, Bingbing Zhu, Chuxuan Tong, Xiaoyi Zhou, Xi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.21163)  

**Abstract**: Adversarial attack methods for 3D point cloud classification reveal the vulnerabilities of point cloud recognition models. This vulnerability could lead to safety risks in critical applications that use deep learning models, such as autonomous vehicles. To uncover the deficiencies of these models, researchers can evaluate their security through adversarial attacks. However, most existing adversarial attack methods are based on white-box attacks. While these methods achieve high attack success rates and imperceptibility, their applicability in real-world scenarios is limited. Black-box attacks, which are more meaningful in real-world scenarios, often yield poor results. This paper proposes a novel black-box adversarial example generation method that utilizes a diffusion model to improve the attack success rate and imperceptibility in the black-box setting, without relying on the internal information of the point cloud classification model to generate adversarial samples. We use a 3D diffusion model to use the compressed features of the point cloud as prior knowledge to guide the reverse diffusion process to add adversarial points to clean examples. Subsequently, its reverse process is employed to transform the distribution of other categories into adversarial points, which are then added to the point cloud. 

---
# Seeing Beyond Frames: Zero-Shot Pedestrian Intention Prediction with Raw Temporal Video and Multimodal Cues 

**Authors**: Pallavi Zambare, Venkata Nikhil Thanikella, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21161)  

**Abstract**: Pedestrian intention prediction is essential for autonomous driving in complex urban environments. Conventional approaches depend on supervised learning over frame sequences and require extensive retraining to adapt to new scenarios. Here, we introduce BF-PIP (Beyond Frames Pedestrian Intention Prediction), a zero-shot approach built upon Gemini 2.5 Pro. It infers crossing intentions directly from short, continuous video clips enriched with structured JAAD metadata. In contrast to GPT-4V based methods that operate on discrete frames, BF-PIP processes uninterrupted temporal clips. It also incorporates bounding-box annotations and ego-vehicle speed via specialized multimodal prompts. Without any additional training, BF-PIP achieves 73% prediction accuracy, outperforming a GPT-4V baseline by 18 %. These findings illustrate that combining temporal video inputs with contextual cues enhances spatiotemporal perception and improves intent inference under ambiguous conditions. This approach paves the way for agile, retraining-free perception module in intelligent transportation system. 

---
# Handling Out-of-Distribution Data: A Survey 

**Authors**: Lakpa Tamang, Mohamed Reda Bouadjenek, Richard Dazeley, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2507.21160)  

**Abstract**: In the field of Machine Learning (ML) and data-driven applications, one of the significant challenge is the change in data distribution between the training and deployment stages, commonly known as distribution shift. This paper outlines different mechanisms for handling two main types of distribution shifts: (i) Covariate shift: where the value of features or covariates change between train and test data, and (ii) Concept/Semantic-shift: where model experiences shift in the concept learned during training due to emergence of novel classes in the test phase. We sum up our contributions in three folds. First, we formalize distribution shifts, recite on how the conventional method fails to handle them adequately and urge for a model that can simultaneously perform better in all types of distribution shifts. Second, we discuss why handling distribution shifts is important and provide an extensive review of the methods and techniques that have been developed to detect, measure, and mitigate the effects of these shifts. Third, we discuss the current state of distribution shift handling mechanisms and propose future research directions in this area. Overall, we provide a retrospective synopsis of the literature in the distribution shift, focusing on OOD data that had been overlooked in the existing surveys. 

---
# Deep Reinforcement Learning for Real-Time Green Energy Integration in Data Centers 

**Authors**: Abderaouf Bahi, Amel Ourici  

**Link**: [PDF](https://arxiv.org/pdf/2507.21153)  

**Abstract**: This paper explores the implementation of a Deep Reinforcement Learning (DRL)-optimized energy management system for e-commerce data centers, aimed at enhancing energy efficiency, cost-effectiveness, and environmental sustainability. The proposed system leverages DRL algorithms to dynamically manage the integration of renewable energy sources, energy storage, and grid power, adapting to fluctuating energy availability in real time. The study demonstrates that the DRL-optimized system achieves a 38\% reduction in energy costs, significantly outperforming traditional Reinforcement Learning (RL) methods (28\%) and heuristic approaches (22\%). Additionally, it maintains a low SLA violation rate of 1.5\%, compared to 3.0\% for RL and 4.8\% for heuristic methods. The DRL-optimized approach also results in an 82\% improvement in energy efficiency, surpassing other methods, and a 45\% reduction in carbon emissions, making it the most environmentally friendly solution. The system's cumulative reward of 950 reflects its superior performance in balancing multiple objectives. Through rigorous testing and ablation studies, the paper validates the effectiveness of the DRL model's architecture and parameters, offering a robust solution for energy management in data centers. The findings highlight the potential of DRL in advancing energy optimization strategies and addressing sustainability challenges. 

---
# Deep Unfolding for MIMO Signal Detection 

**Authors**: Hangli Ge, Noboru Koshizuka  

**Link**: [PDF](https://arxiv.org/pdf/2507.21152)  

**Abstract**: In this paper, we propose a deep unfolding neural network-based MIMO detector that incorporates complex-valued computations using Wirtinger calculus. The method, referred as Dynamic Partially Shrinkage Thresholding (DPST), enables efficient, interpretable, and low-complexity MIMO signal detection. Unlike prior approaches that rely on real-valued approximations, our method operates natively in the complex domain, aligning with the fundamental nature of signal processing tasks. The proposed algorithm requires only a small number of trainable parameters, allowing for simplified training. Numerical results demonstrate that the proposed method achieves superior detection performance with fewer iterations and lower computational complexity, making it a practical solution for next-generation massive MIMO systems. 

---
# Advancing Wildfire Risk Prediction via Morphology-Aware Curriculum Contrastive Learning 

**Authors**: Fabrizio Lo Scudo, Alessio De Rango, Luca Furnari, Alfonso Senatore, Donato D'Ambrosio, Giuseppe Mendicino, Gianluigi Greco  

**Link**: [PDF](https://arxiv.org/pdf/2507.21147)  

**Abstract**: Wildfires significantly impact natural ecosystems and human health, leading to biodiversity loss, increased hydrogeological risks, and elevated emissions of toxic substances. Climate change exacerbates these effects, particularly in regions with rising temperatures and prolonged dry periods, such as the Mediterranean. This requires the development of advanced risk management strategies that utilize state-of-the-art technologies. However, in this context, the data show a bias toward an imbalanced setting, where the incidence of wildfire events is significantly lower than typical situations. This imbalance, coupled with the inherent complexity of high-dimensional spatio-temporal data, poses significant challenges for training deep learning architectures. Moreover, since precise wildfire predictions depend mainly on weather data, finding a way to reduce computational costs to enable more frequent updates using the latest weather forecasts would be beneficial. This paper investigates how adopting a contrastive framework can address these challenges through enhanced latent representations for the patch's dynamic features. We thus introduce a new morphology-based curriculum contrastive learning that mitigates issues associated with diverse regional characteristics and enables the use of smaller patch sizes without compromising performance. An experimental analysis is performed to validate the effectiveness of the proposed modeling strategies. 

---
# Towards Unifying Quantitative Security Benchmarking for Multi Agent Systems 

**Authors**: Gauri Sharma, Vidhi Kulkarni, Miles King, Ken Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21146)  

**Abstract**: Evolving AI systems increasingly deploy multi-agent architectures where autonomous agents collaborate, share information, and delegate tasks through developing protocols. This connectivity, while powerful, introduces novel security risks. One such risk is a cascading risk: a breach in one agent can cascade through the system, compromising others by exploiting inter-agent trust. In tandem with OWASP's initiative for an Agentic AI Vulnerability Scoring System we define an attack vector, Agent Cascading Injection, analogous to Agent Impact Chain and Blast Radius, operating across networks of agents. In an ACI attack, a malicious input or tool exploit injected at one agent leads to cascading compromises and amplified downstream effects across agents that trust its outputs. We formalize this attack with an adversarial goal equation and key variables (compromised agent, injected exploit, polluted observations, etc.), capturing how a localized vulnerability can escalate into system-wide failure. We then analyze ACI's properties -- propagation chains, amplification factors, and inter-agent compound effects -- and map these to OWASP's emerging Agentic AI risk categories (e.g. Impact Chain and Orchestration Exploits). Finally, we argue that ACI highlights a critical need for quantitative benchmarking frameworks to evaluate the security of agent-to-agent communication protocols. We outline a methodology for stress-testing multi-agent systems (using architectures such as Google's A2A and Anthropic's MCP) against cascading trust failures, developing upon groundwork for measurable, standardized agent-to-agent security evaluation. Our work provides the necessary apparatus for engineers to benchmark system resilience, make data-driven architectural trade-offs, and develop robust defenses against a new generation of agentic threats. 

---
# Privacy Artifact ConnecTor (PACT): Embedding Enterprise Artifacts for Compliance AI Agents 

**Authors**: Chenhao Fang, Yanqing Peng, Rajeev Rao, Matt Sarmiento, Wendy Summer, Arya Pudota, Alex Goncalves, Jordi Mola, Hervé Robert  

**Link**: [PDF](https://arxiv.org/pdf/2507.21142)  

**Abstract**: Enterprise environments contain a heterogeneous, rapidly growing collection of internal artifacts related to code, data, and many different tools. Critical information for assessing privacy risk and ensuring regulatory compliance is often embedded across these varied resources, each with their own arcane discovery and extraction techniques. Therefore, large-scale privacy compliance in adherence to governmental regulations requires systems to discern the interconnected nature of diverse artifacts in a common, shared universe.
We present Privacy Artifact ConnecT or (PACT), an embeddings-driven graph that links millions of artifacts spanning multiple artifact types generated by a variety of teams and projects. Powered by the state-of-the-art DRAGON embedding model, PACT uses a contrastive learning objective with light fine-tuning to link artifacts via their textual components such as raw metadata, ownership specifics, and compliance context. Experimental results show that PACT's fine-tuned model improves recall@1 from 18% to 53%, the query match rate from 9.6% to 69.7% when paired with a baseline AI agent, and the hitrate@1 from 25.7% to 44.9% for candidate selection in a standard recommender system. 

---
# TTS-1 Technical Report 

**Authors**: Oleg Atamanenko, Anna Chalova, Joseph Coombes, Nikki Cope, Phillip Dang, Zhifeng Deng, Jimmy Du, Michael Ermolenko, Feifan Fan, Yufei Feng, Cheryl Fichter, Pavel Filimonov, Louis Fischer, Kylan Gibbs, Valeria Gusarova, Pavel Karpik, Andreas Assad Kottner, Ian Lee, Oliver Louie, Jasmine Mai, Mikhail Mamontov, Suri Mao, Nurullah Morshed, Igor Poletaev, Florin Radu, Dmytro Semernia, Evgenii Shingarev, Vikram Sivaraja, Peter Skirko, Rinat Takhautdinov, Robert Villahermosa, Jean Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21138)  

**Abstract**: We introduce Inworld TTS-1, a set of two Transformer-based autoregressive text-to-speech (TTS) models. Our largest model, TTS-1-Max, has 8.8B parameters and is designed for utmost quality and expressiveness in demanding applications. TTS-1 is our most efficient model, with 1.6B parameters, built for real-time speech synthesis and on-device use cases. By scaling train-time compute and applying a sequential process of pre-training, fine-tuning, and RL-alignment of the speech-language model (SpeechLM) component, both models achieve state-of-the-art performance on a variety of benchmarks, demonstrating exceptional quality relying purely on in-context learning of the speaker's voice. Inworld TTS-1 and TTS-1-Max can generate high-resolution 48 kHz speech with low latency, and support 11 languages with fine-grained emotional control and non-verbal vocalizations through audio markups. We additionally open-source our training and modeling code under an MIT license. 

---
# A Study on Variants of Conventional, Fuzzy, and Nullspace-Based Independence Criteria for Improving Supervised and Unsupervised Learning 

**Authors**: Mojtaba Moattari  

**Link**: [PDF](https://arxiv.org/pdf/2507.21136)  

**Abstract**: Unsupervised and supervised learning methods conventionally use kernels to capture nonlinearities inherent in data structure. However experts have to ensure their proposed nonlinearity maximizes variability and capture inherent diversity of data. We reviewed all independence criteria to design unsupervised learners. Then we proposed 3 independence criteria and used them to design unsupervised and supervised dimensionality reduction methods. We evaluated contrast, accuracy and interpretability of these methods in both linear and neural nonlinear settings. The results show that the methods have outperformed the baseline (tSNE, PCA, regularized LDA, VAE with (un)supervised learner and layer sharing) and opened a new line of interpretable machine learning (ML) for the researchers. 

---
# Analysis of Threat-Based Manipulation in Large Language Models: A Dual Perspective on Vulnerabilities and Performance Enhancement Opportunities 

**Authors**: Atil Samancioglu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21133)  

**Abstract**: Large Language Models (LLMs) demonstrate complex responses to threat-based manipulations, revealing both vulnerabilities and unexpected performance enhancement opportunities. This study presents a comprehensive analysis of 3,390 experimental responses from three major LLMs (Claude, GPT-4, Gemini) across 10 task domains under 6 threat conditions. We introduce a novel threat taxonomy and multi-metric evaluation framework to quantify both negative manipulation effects and positive performance improvements. Results reveal systematic vulnerabilities, with policy evaluation showing the highest metric significance rates under role-based threats, alongside substantial performance enhancements in numerous cases with effect sizes up to +1336%. Statistical analysis indicates systematic certainty manipulation (pFDR < 0.0001) and significant improvements in analytical depth and response quality. These findings have dual implications for AI safety and practical prompt engineering in high-stakes applications. 

---
# RATE: An LLM-Powered Retrieval Augmented Generation Technology-Extraction Pipeline 

**Authors**: Karan Mirhosseini, Arya Aftab, Alireza Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2507.21125)  

**Abstract**: In an era of radical technology transformations, technology maps play a crucial role in enhancing decision making. These maps heavily rely on automated methods of technology extraction. This paper introduces Retrieval Augmented Technology Extraction (RATE), a Large Language Model (LLM) based pipeline for automated technology extraction from scientific literature. RATE combines Retrieval Augmented Generation (RAG) with multi-definition LLM-based validation. This hybrid method results in high recall in candidate generation alongside with high precision in candidate filtering. While the pipeline is designed to be general and widely applicable, we demonstrate its use on 678 research articles focused on Brain-Computer Interfaces (BCIs) and Extended Reality (XR) as a case study. Consequently, The validated technology terms by RATE were mapped into a co-occurrence network, revealing thematic clusters and structural features of the research landscape. For the purpose of evaluation, a gold standard dataset of technologies in 70 selected random articles had been curated by the experts. In addition, a technology extraction model based on Bidirectional Encoder Representations of Transformers (BERT) was used as a comparative method. RATE achieved F1-score of 91.27%, Significantly outperforming BERT with F1-score of 53.73%. Our findings highlight the promise of definition-driven LLM methods for technology extraction and mapping. They also offer new insights into emerging trends within the BCI-XR field. The source code is available this https URL 

---
# VizGenie: Toward Self-Refining, Domain-Aware Workflows for Next-Generation Scientific Visualization 

**Authors**: Ayan Biswas, Terece L. Turton, Nishath Rajiv Ranasinghe, Shawn Jones, Bradley Love, William Jones, Aric Hagberg, Han-Wei Shen, Nathan DeBardeleben, Earl Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2507.21124)  

**Abstract**: We present VizGenie, a self-improving, agentic framework that advances scientific visualization through large language model (LLM) by orchestrating of a collection of domain-specific and dynamically generated modules. Users initially access core functionalities--such as threshold-based filtering, slice extraction, and statistical analysis--through pre-existing tools. For tasks beyond this baseline, VizGenie autonomously employs LLMs to generate new visualization scripts (e.g., VTK Python code), expanding its capabilities on-demand. Each generated script undergoes automated backend validation and is seamlessly integrated upon successful testing, continuously enhancing the system's adaptability and robustness. A distinctive feature of VizGenie is its intuitive natural language interface, allowing users to issue high-level feature-based queries (e.g., ``visualize the skull"). The system leverages image-based analysis and visual question answering (VQA) via fine-tuned vision models to interpret these queries precisely, bridging domain expertise and technical implementation. Additionally, users can interactively query generated visualizations through VQA, facilitating deeper exploration. Reliability and reproducibility are further strengthened by Retrieval-Augmented Generation (RAG), providing context-driven responses while maintaining comprehensive provenance records. Evaluations on complex volumetric datasets demonstrate significant reductions in cognitive overhead for iterative visualization tasks. By integrating curated domain-specific tools with LLM-driven flexibility, VizGenie not only accelerates insight generation but also establishes a sustainable, continuously evolving visualization practice. The resulting platform dynamically learns from user interactions, consistently enhancing support for feature-centric exploration and reproducible research in scientific visualization. 

---
# Affect-aware Cross-Domain Recommendation for Art Therapy via Music Preference Elicitation 

**Authors**: Bereket A. Yilma, Luis A. Leiva  

**Link**: [PDF](https://arxiv.org/pdf/2507.21120)  

**Abstract**: Art Therapy (AT) is an established practice that facilitates emotional processing and recovery through creative expression. Recently, Visual Art Recommender Systems (VA RecSys) have emerged to support AT, demonstrating their potential by personalizing therapeutic artwork recommendations. Nonetheless, current VA RecSys rely on visual stimuli for user modeling, limiting their ability to capture the full spectrum of emotional responses during preference elicitation. Previous studies have shown that music stimuli elicit unique affective reflections, presenting an opportunity for cross-domain recommendation (CDR) to enhance personalization in AT. Since CDR has not yet been explored in this context, we propose a family of CDR methods for AT based on music-driven preference elicitation. A large-scale study with 200 users demonstrates the efficacy of music-driven preference elicitation, outperforming the classic visual-only elicitation approach. Our source code, data, and models are available at this https URL 

---
# Failure Risk Prediction in a MOOC: A Multivariate Time Series Analysis Approach 

**Authors**: Anass El Ayady, Maxime Devanne, Germain Forestier, Nour El Mawas  

**Link**: [PDF](https://arxiv.org/pdf/2507.21118)  

**Abstract**: MOOCs offer free and open access to a wide audience, but completion rates remain low, often due to a lack of personalized content. To address this issue, it is essential to predict learner performance in order to provide tailored feedback. Behavioral traces-such as clicks and events-can be analyzed as time series to anticipate learners' outcomes. This work compares multivariate time series classification methods to identify at-risk learners at different stages of the course (after 5, 10 weeks, etc.). The experimental evaluation, conducted on the Open University Learning Analytics Dataset (OULAD), focuses on three courses: two in STEM and one in SHS. Preliminary results show that the evaluated approaches are promising for predicting learner failure in MOOCs. The analysis also suggests that prediction accuracy is influenced by the amount of recorded interactions, highlighting the importance of rich and diverse behavioral data. 

---
# A Comprehensive Review on Harnessing Large Language Models to Overcome Recommender System Challenges 

**Authors**: Rahul Raja, Anshaj Vats, Arpita Vats, Anirban Majumder  

**Link**: [PDF](https://arxiv.org/pdf/2507.21117)  

**Abstract**: Recommender systems have traditionally followed modular architectures comprising candidate generation, multi-stage ranking, and re-ranking, each trained separately with supervised objectives and hand-engineered features. While effective in many domains, such systems face persistent challenges including sparse and noisy interaction data, cold-start problems, limited personalization depth, and inadequate semantic understanding of user and item content. The recent emergence of Large Language Models (LLMs) offers a new paradigm for addressing these limitations through unified, language-native mechanisms that can generalize across tasks, domains, and modalities. In this paper, we present a comprehensive technical survey of how LLMs can be leveraged to tackle key challenges in modern recommender systems. We examine the use of LLMs for prompt-driven candidate retrieval, language-native ranking, retrieval-augmented generation (RAG), and conversational recommendation, illustrating how these approaches enhance personalization, semantic alignment, and interpretability without requiring extensive task-specific supervision. LLMs further enable zero- and few-shot reasoning, allowing systems to operate effectively in cold-start and long-tail scenarios by leveraging external knowledge and contextual cues. We categorize these emerging LLM-driven architectures and analyze their effectiveness in mitigating core bottlenecks of conventional pipelines. In doing so, we provide a structured framework for understanding the design space of LLM-enhanced recommenders, and outline the trade-offs between accuracy, scalability, and real-time performance. Our goal is to demonstrate that LLMs are not merely auxiliary components but foundational enablers for building more adaptive, semantically rich, and user-centric recommender systems 

---
# FedFlex: Federated Learning for Diverse Netflix Recommendations 

**Authors**: Sven Lankester, Manel Slokom, Gustavo de Carvalho Bertoli, Matias Vizcaino, Emmanuelle Beauxis Aussalet, Laura Hollink  

**Link**: [PDF](https://arxiv.org/pdf/2507.21115)  

**Abstract**: Federated learning is a decentralized approach that enables collaborative model training across multiple devices while preserving data privacy. It has shown significant potential in various domains, including healthcare and personalized recommendation systems. However, most existing work on federated recommendation systems has focused primarily on improving accuracy, with limited attention to fairness and diversity. In this paper, we introduce FedFlex, a federated recommender system for Netflix-style TV series recommendations. FedFlex integrates two state-of-the-art matrix factorization algorithms for personalized fine-tuning. FedFlex also applies Maximal Marginal Relevance (MMR) to re-rank items and enhance diversity. We conduct extensive experiments comparing recommendations generated by SVD and BPR algorithms. In a live two-week user study, participants received two recommendation lists: List A, based on SVD or BPR, and List B, a re-ranked version emphasizing diversity. Participants were asked to click on the movies they were interested in watching. Our findings demonstrate that FedFlex effectively introduces diverse content, such as new genres, into recommendations without necessarily compromising user satisfaction. 

---
# Page image classification for content-specific data processing 

**Authors**: Kateryna Lutsai, Pavel Straňák  

**Link**: [PDF](https://arxiv.org/pdf/2507.21114)  

**Abstract**: Digitization projects in humanities often generate vast quantities of page images from historical documents, presenting significant challenges for manual sorting and analysis. These archives contain diverse content, including various text types (handwritten, typed, printed), graphical elements (drawings, maps, photos), and layouts (plain text, tables, forms). Efficiently processing this heterogeneous data requires automated methods to categorize pages based on their content, enabling tailored downstream analysis pipelines. This project addresses this need by developing and evaluating an image classification system specifically designed for historical document pages, leveraging advancements in artificial intelligence and machine learning. The set of categories was chosen to facilitate content-specific processing workflows, separating pages requiring different analysis techniques (e.g., OCR for text, image analysis for graphics) 

---
# A Formal Rebuttal of "The Blockchain Trilemma: A Formal Proof of the Inherent Trade-Offs Among Decentralization, Security, and Scalability" 

**Authors**: Craig Wright  

**Link**: [PDF](https://arxiv.org/pdf/2507.21111)  

**Abstract**: This paper presents a comprehensive refutation of the so-called "blockchain trilemma," a widely cited but formally ungrounded claim asserting an inherent trade-off between decentralisation, security, and scalability in blockchain protocols. Through formal analysis, empirical evidence, and detailed critique of both methodology and terminology, we demonstrate that the trilemma rests on semantic equivocation, misuse of distributed systems theory, and a failure to define operational metrics. Particular focus is placed on the conflation of topological network analogies with protocol-level architecture, the mischaracterisation of Bitcoin's design--including the role of miners, SPV clients, and header-based verification--and the failure to ground claims in complexity-theoretic or adversarial models. By reconstructing Bitcoin as a deterministic, stateless distribution protocol governed by evidentiary trust, we show that scalability is not a trade-off but an engineering outcome. The paper concludes by identifying systemic issues in academic discourse and peer review that have allowed such fallacies to persist, and offers formal criteria for evaluating future claims in blockchain research. 

---
# SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering 

**Authors**: Kezhen Zhong, Basem Suleiman, Abdelkarim Erradi, Shijing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21110)  

**Abstract**: This paper introduces SemRAG, an enhanced Retrieval Augmented Generation (RAG) framework that efficiently integrates domain-specific knowledge using semantic chunking and knowledge graphs without extensive fine-tuning. Integrating domain-specific knowledge into large language models (LLMs) is crucial for improving their performance in specialized tasks. Yet, existing adaptations are computationally expensive, prone to overfitting and limit scalability. To address these challenges, SemRAG employs a semantic chunking algorithm that segments documents based on the cosine similarity from sentence embeddings, preserving semantic coherence while reducing computational overhead. Additionally, by structuring retrieved information into knowledge graphs, SemRAG captures relationships between entities, improving retrieval accuracy and contextual understanding. Experimental results on MultiHop RAG and Wikipedia datasets demonstrate SemRAG has significantly enhances the relevance and correctness of retrieved information from the Knowledge Graph, outperforming traditional RAG methods. Furthermore, we investigate the optimization of buffer sizes for different data corpus, as optimizing buffer sizes tailored to specific datasets can further improve retrieval performance, as integration of knowledge graphs strengthens entity relationships for better contextual comprehension. The primary advantage of SemRAG is its ability to create an efficient, accurate domain-specific LLM pipeline while avoiding resource-intensive fine-tuning. This makes it a practical and scalable approach aligned with sustainability goals, offering a viable solution for AI applications in domain-specific fields. 

---
# Task-Focused Consolidation with Spaced Recall: Making Neural Networks learn like college students 

**Authors**: Prital Bamnodkar  

**Link**: [PDF](https://arxiv.org/pdf/2507.21109)  

**Abstract**: Deep Neural Networks often suffer from a critical limitation known as Catastrophic Forgetting, where performance on past tasks degrades after learning new ones. This paper introduces a novel continual learning approach inspired by human learning strategies like Active Recall, Deliberate Practice and Spaced Repetition, named Task Focused Consolidation with Spaced Recall (TFC-SR). TFC-SR enhances the standard experience replay with a mechanism we termed the Active Recall Probe. It is a periodic, task-aware evaluation of the model's memory that stabilizes the representations of past knowledge. We test TFC-SR on the Split MNIST and Split CIFAR-100 benchmarks against leading regularization-based and replay-based baselines. Our results show that TFC-SR performs significantly better than these methods. For instance, on the Split CIFAR-100, it achieves a final accuracy of 13.17% compared to standard replay's 7.40%. We demonstrate that this advantage comes from the stabilizing effect of the probe itself, and not from the difference in replay volume. Additionally, we analyze the trade-off between memory size and performance and show that while TFC-SR performs better in memory-constrained environments, higher replay volume is still more effective when available memory is abundant. We conclude that TFC-SR is a robust and efficient approach, highlighting the importance of integrating active memory retrieval mechanisms into continual learning systems. 

---
# A Survey of Classification Tasks and Approaches for Legal Contracts 

**Authors**: Amrita Singh, Aditya Joshi, Jiaojiao Jiang, Hye-young Paik  

**Link**: [PDF](https://arxiv.org/pdf/2507.21108)  

**Abstract**: Given the large size and volumes of contracts and their underlying inherent complexity, manual reviews become inefficient and prone to errors, creating a clear need for automation. Automatic Legal Contract Classification (LCC) revolutionizes the way legal contracts are analyzed, offering substantial improvements in speed, accuracy, and accessibility. This survey delves into the challenges of automatic LCC and a detailed examination of key tasks, datasets, and methodologies. We identify seven classification tasks within LCC, and review fourteen datasets related to English-language contracts, including public, proprietary, and non-public sources. We also introduce a methodology taxonomy for LCC, categorized into Traditional Machine Learning, Deep Learning, and Transformer-based approaches. Additionally, the survey discusses evaluation techniques and highlights the best-performing results from the reviewed studies. By providing a thorough overview of current methods and their limitations, this survey suggests future research directions to improve the efficiency, accuracy, and scalability of LCC. As the first comprehensive survey on LCC, it aims to support legal NLP researchers and practitioners in improving legal processes, making legal information more accessible, and promoting a more informed and equitable society. 

---
# Curved Inference: Concern-Sensitive Geometry in Large Language Model Residual Streams 

**Authors**: Rob Manson  

**Link**: [PDF](https://arxiv.org/pdf/2507.21107)  

**Abstract**: We propose Curved Inference - a geometric Interpretability framework that tracks how the residual stream trajectory of a large language model bends in response to shifts in semantic concern. Across 20 matched prompts spanning emotional, moral, perspective, logical, identity, environmental, and nonsense domains, we analyse Gemma3-1b and LLaMA3.2-3b using five native-space metrics, with a primary focus on curvature (\k{appa}_i) and salience (S(t)). These metrics are computed under a pullback semantic metric derived from the unembedding matrix, ensuring that all measurements reflect token-aligned geometry rather than raw coordinate structure. We find that concern-shifted prompts reliably alter internal activation trajectories in both models - with LLaMA exhibiting consistent, statistically significant scaling in both curvature and salience as concern intensity increases. Gemma also responds to concern but shows weaker differentiation between moderate and strong variants. Our results support a two-layer view of LLM geometry - a latent conceptual structure encoded in the embedding space, and a contextual trajectory shaped by prompt-specific inference. Curved Inference reveals how models navigate, reorient, or reinforce semantic meaning over depth, offering a principled method for diagnosing alignment, abstraction, and emergent inference dynamics. These findings offer fresh insight into semantic abstraction and model alignment through the lens of Curved Inference. 

---
# AgentMaster: A Multi-Agent Conversational Framework Using A2A and MCP Protocols for Multimodal Information Retrieval and Analysis 

**Authors**: Callie C. Liao, Duoduo Liao, Sai Surya Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2507.21105)  

**Abstract**: The rise of Multi-Agent Systems (MAS) in Artificial Intelligence (AI), especially integrated with Large Language Models (LLMs), has greatly facilitated the resolution of complex tasks. However, current systems are still facing challenges of inter-agent communication, coordination, and interaction with heterogeneous tools and resources. Most recently, the Model Context Protocol (MCP) by Anthropic and Agent-to-Agent (A2A) communication protocol by Google have been introduced, and to the best of our knowledge, very few applications exist where both protocols are employed within a single MAS framework. We present a pilot study of AgentMaster, a novel modular multi-protocol MAS framework with self-implemented A2A and MCP, enabling dynamic coordination and flexible communication. Through a unified conversational interface, the system supports natural language interaction without prior technical expertise and responds to multimodal queries for tasks including information retrieval, question answering, and image analysis. Evaluation through the BERTScore F1 and LLM-as-a-Judge metric G-Eval averaged 96.3\% and 87.1\%, revealing robust inter-agent coordination, query decomposition, dynamic routing, and domain-specific, relevant responses. Overall, our proposed framework contributes to the potential capabilities of domain-specific, cooperative, and scalable conversational AI powered by MAS. 

---
# iLSU-T: an Open Dataset for Uruguayan Sign Language Translation 

**Authors**: Ariel E. Stassi, Yanina Boria, J. Matías Di Martino, Gregory Randall  

**Link**: [PDF](https://arxiv.org/pdf/2507.21104)  

**Abstract**: Automatic sign language translation has gained particular interest in the computer vision and computational linguistics communities in recent years. Given each sign language country particularities, machine translation requires local data to develop new techniques and adapt existing ones. This work presents iLSU T, an open dataset of interpreted Uruguayan Sign Language RGB videos with audio and text transcriptions. This type of multimodal and curated data is paramount for developing novel approaches to understand or generate tools for sign language processing. iLSU T comprises more than 185 hours of interpreted sign language videos from public TV broadcasting. It covers diverse topics and includes the participation of 18 professional interpreters of sign language. A series of experiments using three state of the art translation algorithms is presented. The aim is to establish a baseline for this dataset and evaluate its usefulness and the proposed pipeline for data processing. The experiments highlight the need for more localized datasets for sign language translation and understanding, which are critical for developing novel tools to improve accessibility and inclusion of all individuals. Our data and code can be accessed. 

---
# Assessing the Ecological Impact of AI 

**Authors**: Sylvia Wenmackers  

**Link**: [PDF](https://arxiv.org/pdf/2507.21102)  

**Abstract**: Philosophers of technology have recently started paying more attention to the environmental impacts of AI, in particular of large language models (LLMs) and generative AI (genAI) applications. Meanwhile, few developers of AI give concrete estimates of the ecological impact of their models and products, and even when they do so, their analysis is often limited to green house gas emissions of certain stages of AI development or use. The current proposal encourages practically viable analyses of the sustainability aspects of genAI informed by philosophical ideas. 

---
# A Tactical Behaviour Recognition Framework Based on Causal Multimodal Reasoning: A Study on Covert Audio-Video Analysis Combining GAN Structure Enhancement and Phonetic Accent Modelling 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.21100)  

**Abstract**: This paper introduces TACTIC-GRAPHS, a system that combines spectral graph theory and multimodal graph neural reasoning for semantic understanding and threat detection in tactical video under high noise and weak structure. The framework incorporates spectral embedding, temporal causal edge modeling, and discriminative path inference across heterogeneous modalities. A semantic-aware keyframe extraction method fuses visual, acoustic, and action cues to construct temporal graphs. Using graph attention and Laplacian spectral mapping, the model performs cross-modal weighting and causal signal analysis. Experiments on TACTIC-AVS and TACTIC-Voice datasets show 89.3 percent accuracy in temporal alignment and over 85 percent recognition of complete threat chains, with node latency within plus-minus 150 milliseconds. The approach enhances structural interpretability and supports applications in surveillance, defense, and intelligent security systems. 

---
# The Value of Gen-AI Conversations: A bottom-up Framework for AI Value Alignment 

**Authors**: Lenart Motnikar, Katharina Baum, Alexander Kagan, Sarah Spiekermann-Hoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.21091)  

**Abstract**: Conversational agents (CAs) based on generative artificial intelligence frequently face challenges ensuring ethical interactions that align with human values. Current value alignment efforts largely rely on top-down approaches, such as technical guidelines or legal value principles. However, these methods tend to be disconnected from the specific contexts in which CAs operate, potentially leading to misalignment with users interests. To address this challenge, we propose a novel, bottom-up approach to value alignment, utilizing the value ontology of the ISO Value-Based Engineering standard for ethical IT design. We analyse 593 ethically sensitive system outputs identified from 16,908 conversational logs of a major European employment service CA to identify core values and instances of value misalignment within real-world interactions. The results revealed nine core values and 32 different value misalignments that negatively impacted users. Our findings provide actionable insights for CA providers seeking to address ethical challenges and achieve more context-sensitive value alignment. 

---
# Thinking Like a Scientist: Can Interactive Simulations Foster Critical AI Literacy? 

**Authors**: Yiling Zhao, Audrey Michal, Nithum Thain, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2507.21090)  

**Abstract**: As AI systems shape individual and societal decisions, fostering critical AI literacy is essential. Traditional approaches, such as blog articles, static lessons, and social media discussions, often fail to support deep conceptual understanding and critical engagement. This study examines whether interactive simulations can help learners think like a scientist by engaging them in hypothesis testing, experimentation, and direct observation of AI behavior. In a controlled study with 605 participants, we assess how interactive AI tutorials impact learning of key concepts such as fairness, dataset representativeness, and bias in language models. Results show that interactive simulations effectively enhance AI literacy across topics, supporting greater knowledge transfer and self-reported confidence, though engagement alone does not predict learning. This work contributes to the growing field of AI literacy education, highlighting how interactive, inquiry-driven methodologies can better equip individuals to critically engage with AI in their daily lives. 

---
# ChatGPT Reads Your Tone and Responds Accordingly -- Until It Does Not -- Emotional Framing Induces Bias in LLM Outputs 

**Authors**: Franck Bardol  

**Link**: [PDF](https://arxiv.org/pdf/2507.21083)  

**Abstract**: Large Language Models like GPT-4 adjust their responses not only based on the question asked, but also on how it is emotionally phrased. We systematically vary the emotional tone of 156 prompts - spanning controversial and everyday topics - and analyze how it affects model responses. Our findings show that GPT-4 is three times less likely to respond negatively to a negatively framed question than to a neutral one. This suggests a "rebound" bias where the model overcorrects, often shifting toward neutrality or positivity. On sensitive topics (e.g., justice or politics), this effect is even more pronounced: tone-based variation is suppressed, suggesting an alignment override. We introduce concepts like the "tone floor" - a lower bound in response negativity - and use tone-valence transition matrices to quantify behavior. Visualizations based on 1536-dimensional embeddings confirm semantic drift based on tone. Our work highlights an underexplored class of biases driven by emotional framing in prompts, with implications for AI alignment and trust. Code and data are available at: this https URL 

---
# Empathy in Explanation 

**Authors**: Katherine M. Collins, Kartik Chandra, Adrian Weller, Jonathan Ragan-Kelley, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2507.21081)  

**Abstract**: Why do we give the explanations we do? Recent work has suggested that we should think of explanation as a kind of cooperative social interaction, between a why-question-asker and an explainer. Here, we apply this perspective to consider the role that emotion plays in this social interaction. We develop a computational framework for modeling explainers who consider the emotional impact an explanation might have on a listener. We test our framework by using it to model human intuitions about how a doctor might explain to a patient why they have a disease, taking into account the patient's propensity for regret. Our model predicts human intuitions well, better than emotion-agnostic ablations, suggesting that people do indeed reason about emotion when giving explanations. 

---
# Which symbol grounding problem should we try to solve? 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2507.21080)  

**Abstract**: Floridi and Taddeo propose a condition of "zero semantic commitment" for solutions to the grounding problem, and a solution to it. I argue briefly that their condition cannot be fulfilled, not even by their own solution. After a look at Luc Steels' very different competing suggestion, I suggest that we need to re-think what the problem is and what role the 'goals' in a system play in formulating the problem. On the basis of a proper understanding of computing, I come to the conclusion that the only sensible grounding problem is how we can explain and re-produce the behavioral ability and function of meaning in artificial computational agents 

---
# Data-Driven and Participatory Approaches toward Neuro-Inclusive AI 

**Authors**: Naba Rizvi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21077)  

**Abstract**: Biased data representation in AI marginalizes up to 75 million autistic people worldwide through medical applications viewing autism as a deficit of neurotypical social skills rather than an aspect of human diversity, and this perspective is grounded in research questioning the humanity of autistic people. Turing defined artificial intelligence as the ability to mimic human communication, and as AI development increasingly focuses on human-like agents, this benchmark remains popular. In contrast, we define Neuro-Inclusive AI as datasets and systems that move away from mimicking humanness as a benchmark for machine intelligence. Then, we explore the origins, prevalence, and impact of anti-autistic biases in current research. Our work finds that 90% of human-like AI agents exclude autistic perspectives, and AI creators continue to believe ethical considerations are beyond the scope of their work. To improve the autistic representation in data, we conduct empirical experiments with annotators and LLMs, finding that binary labeling schemes sufficiently capture the nuances of labeling anti-autistic hate speech. Our benchmark, AUTALIC, can be used to evaluate or fine-tune models, and was developed to serve as a foundation for more neuro-inclusive future work. 

---
# Empowering Educators in the Age of AI: An Empirical Study on Creating custom GPTs in Qualitative Research Method education 

**Authors**: Qian Huang, Thijs Willems  

**Link**: [PDF](https://arxiv.org/pdf/2507.21074)  

**Abstract**: As generative AI (Gen-AI) tools become more prevalent in education, there is a growing need to understand how educators, not just students, can actively shape their design and use. This study investigates how two instructors integrated four custom GPT tools into a Masters-level Qualitative Research Methods course for Urban Planning Policy students. Addressing two key gaps: the dominant framing of students as passive AI users, and the limited use of AI in qualitative methods education. The study explores how Gen-AI can support disciplinary learning when aligned with pedagogical intent. Drawing on the Technological Pedagogical Content Knowledge (TPACK) framework and action research methodology, the instructors designed GPTs to scaffold tasks such as research question formulation, interview practice, fieldnote analysis, and design thinking. Thematic analysis of student reflections, AI chat logs, and final assignments revealed that the tools enhanced student reflexivity, improved interview techniques, and supported structured analytic thinking. However, students also expressed concerns about cognitive overload, reduced immersion in data, and the formulaic nature of AI responses. The study offers three key insights: AI can be a powerful scaffold for active learning when paired with human facilitation; custom GPTs can serve as cognitive partners in iterative research practice; and educator-led design is critical to pedagogically meaningful AI integration. This research contributes to emerging scholarship on AI in higher education by demonstrating how empowering educators to design custom tools can promote more reflective, responsible, and collaborative learning with AI. 

---
# FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents 

**Authors**: Qinglong Yang, Haoming Li, Haotian Zhao, Xiaokai Yan, Jingtao Ding, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.21071)  

**Abstract**: Mobile GUI agents are becoming critical tools for enhancing human-device interaction efficiency, with multimodal large language models (MLLMs) emerging as dominant paradigms in this domain. Current agents, however, are limited to following explicit human instructions, resulting in insufficient capability for proactive intent anticipation. Additionally, these agents fail to leverage the contextual information associated with users during task execution, thereby neglecting potentially vast differences in user preferences. To address these challenges, we introduce the FingerTip benchmark. It contains two new tracks: proactive task suggestions by analyzing environment observation and users' previous intents, and personalized task execution by catering to users' action preferences. We collected unique human demonstrations of multi-step Android device interactions across a variety of everyday apps. These demonstrations are not isolated but are continuously acquired from the users' long-term usage in their real lives, and encompass essential user-related contextual information. Our experiments reveal challenges of the tasks we propose. The model fine-tuned with the data we collected effectively utilized user information and achieved good results, highlighting the potential of our approach in building more user-oriented mobile GUI agents. Our code is open-source at this https URL for reproducibility. 

---
# GAITEX: Human motion dataset from impaired gait and rehabilitation exercises of inertial and optical sensor data 

**Authors**: Andreas Spilz, Heiko Oppel, Jochen Werner, Kathrin Stucke-Straub, Felix Capanni, Michael Munz  

**Link**: [PDF](https://arxiv.org/pdf/2507.21069)  

**Abstract**: Wearable inertial measurement units (IMUs) offer a cost-effective and scalable means to assess human movement quality in clinical and everyday settings. However, the development of robust sensor-based classification models for physiotherapeutic exercises and gait analysis requires large, diverse datasets, which are costly and time-consuming to collect. Here, we present a multimodal dataset of physiotherapeutic exercises - including correct and clinically relevant variants - and gait-related exercises - including both normal and impaired gait patterns - recorded from 19 participants using synchronized IMUs and marker-based motion capture (MoCap). The dataset includes raw data from nine IMUs and thirty-five optical markers capturing full-body kinematics. Each IMU is additionally equipped with four optical markers, enabling precise comparison between IMU-derived orientation estimates and reference values from the MoCap system. To support further analysis, we also provide processed IMU orientations aligned with common segment coordinate systems, subject-specific OpenSim models, inverse kinematics results, and tools for visualizing IMU orientations in the musculoskeletal context. Detailed annotations of movement execution quality and time-stamped segmentations support diverse analysis goals. This dataset supports the development and benchmarking of machine learning models for tasks such as automatic exercise evaluation, gait analysis, temporal activity segmentation, and biomechanical parameter estimation. To facilitate reproducibility, we provide code for postprocessing, sensor-to-segment alignment, inverse kinematics computation, and technical validation. This resource is intended to accelerate research in machine learning-driven human movement analysis. 

---
# Privacy-Preserving AI for Encrypted Medical Imaging: A Framework for Secure Diagnosis and Learning 

**Authors**: Abdullah Al Siam, Sadequzzaman Shohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21060)  

**Abstract**: The rapid integration of Artificial Intelligence (AI) into medical diagnostics has raised pressing concerns about patient privacy, especially when sensitive imaging data must be transferred, stored, or processed. In this paper, we propose a novel framework for privacy-preserving diagnostic inference on encrypted medical images using a modified convolutional neural network (Masked-CNN) capable of operating on transformed or ciphered image formats. Our approach leverages AES-CBC encryption coupled with JPEG2000 compression to protect medical images while maintaining their suitability for AI inference. We evaluate the system using public DICOM datasets (NIH ChestX-ray14 and LIDC-IDRI), focusing on diagnostic accuracy, inference latency, storage efficiency, and privacy leakage resistance. Experimental results show that the encrypted inference model achieves performance comparable to its unencrypted counterpart, with only marginal trade-offs in accuracy and latency. The proposed framework bridges the gap between data privacy and clinical utility, offering a practical, scalable solution for secure AI-driven diagnostics. 

---
# Categorical Classification of Book Summaries Using Word Embedding Techniques 

**Authors**: Kerem Keskin, Mümine Kaya Keleş  

**Link**: [PDF](https://arxiv.org/pdf/2507.21058)  

**Abstract**: In this study, book summaries and categories taken from book sites were classified using word embedding methods, natural language processing techniques and machine learning algorithms. In addition, one hot encoding, Word2Vec and Term Frequency - Inverse Document Frequency (TF-IDF) methods, which are frequently used word embedding methods were used in this study and their success was compared. Additionally, the combination table of the pre-processing methods used is shown and added to the table. Looking at the results, it was observed that Support Vector Machine, Naive Bayes and Logistic Regression Models and TF-IDF and One-Hot Encoder word embedding techniques gave more successful results for Turkish texts. 

---
# AI-Driven Generation of Data Contracts in Modern Data Engineering Systems 

**Authors**: Harshraj Bhoite  

**Link**: [PDF](https://arxiv.org/pdf/2507.21056)  

**Abstract**: Data contracts formalize agreements between data producers and consumers regarding schema, semantics, and quality expectations. As data pipelines grow in complexity, manual authoring and maintenance of contracts becomes error-prone and labor-intensive. We present an AI-driven framework for automatic data contract generation using large language models (LLMs). Our system leverages parameter-efficient fine-tuning methods, including LoRA and PEFT, to adapt LLMs to structured data domains. The models take sample data or schema descriptions and output validated contract definitions in formats such as JSON Schema and Avro. We integrate this framework into modern data platforms (e.g., Databricks, Snowflake) to automate contract enforcement at scale. Experimental results on synthetic and real-world datasets demonstrate that the fine-tuned LLMs achieve high accuracy in generating valid contracts and reduce manual workload by over 70%. We also discuss key challenges such as hallucination, version control, and the need for continuous learning. This work demonstrates that generative AI can enable scalable, agile data governance by bridging the gap between intent and implementation in enterprise data management. 

---
# Bridging the Gap: Enhancing News Interpretation Across Diverse Audiences with Large Language Models 

**Authors**: Leyi Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21055)  

**Abstract**: In the interconnected world, news media are critical in conveying information to public across diverse domains including technology, finance, and agriculture. Journalists make efforts to present accurate information, however, the interpretation of news often varies significantly among different audiences due to their specific expertise and age. In this work, we investigate how to identify these comprehension gaps and provide solutions to improve audiences understanding of news content, particular to the aspects of articles outside their primary domains of knowledge. We propose a agent-based framework using large language models (LLMs) to simulate society communication behaviors, where several agents can discuss news. These agents can be designed to be experts from various occupation, or from different age group. Our results indicate that this framework can identify confusions or even misunderstanding of news for the agent through the iterative discussion process. Based on these accurate identification, the framework can design a supplement material specific to these agents on the news. Our results show that agents exhibit significantly improved news understanding after receiving this material. These findings highlight our framework's utility and efficiency in enhancing news comprehension for diverse audiences by directly addressing their understanding gap. 

---
# High hopes for "Deep Medicine"? AI, economics, and the future of care 

**Authors**: Robert Sparrow, Joshua Hatherley  

**Link**: [PDF](https://arxiv.org/pdf/2507.21054)  

**Abstract**: In the much-celebrated book Deep Medicine, Eric Topol argues that the development of artificial intelligence for health care will lead to a dramatic shift in the culture and practice of medicine. In the next several decades, he suggests, AI will become sophisticated enough that many of the everyday tasks of physicians could be delegated to it. Topol is perhaps the most articulate advocate of the benefits of AI in medicine, but he is hardly alone in spruiking its potential to allow physicians to dedicate more of their time and attention to providing empathetic care for their patients in the future. Unfortunately, several factors suggest a radically different picture for the future of health care. Far from facilitating a return to a time of closer doctor-patient relationships, the use of medical AI seems likely to further erode therapeutic relationships and threaten professional and patient satisfaction. 

---
# Online hierarchical partitioning of the output space in extreme multi-label data stream 

**Authors**: Lara Neves, Afonso Lourenço, Alberto Cano, Goreti Marreiros  

**Link**: [PDF](https://arxiv.org/pdf/2507.20894)  

**Abstract**: Mining data streams with multi-label outputs poses significant challenges due to evolving distributions, high-dimensional label spaces, sparse label occurrences, and complex label dependencies. Moreover, concept drift affects not only input distributions but also label correlations and imbalance ratios over time, complicating model adaptation. To address these challenges, structured learners are categorized into local and global methods. Local methods break down the task into simpler components, while global methods adapt the algorithm to the full output space, potentially yielding better predictions by exploiting label correlations. This work introduces iHOMER (Incremental Hierarchy Of Multi-label Classifiers), an online multi-label learning framework that incrementally partitions the label space into disjoint, correlated clusters without relying on predefined hierarchies. iHOMER leverages online divisive-agglomerative clustering based on \textit{Jaccard} similarity and a global tree-based learner driven by a multivariate \textit{Bernoulli} process to guide instance partitioning. To address non-stationarity, it integrates drift detection mechanisms at both global and local levels, enabling dynamic restructuring of label partitions and subtrees. Experiments across 23 real-world datasets show iHOMER outperforms 5 state-of-the-art global baselines, such as MLHAT, MLHT of Pruned Sets and iSOUPT, by 23\%, and 12 local baselines, such as binary relevance transformations of kNN, EFDT, ARF, and ADWIN bagging/boosting ensembles, by 32\%, establishing its robustness for online multi-label classification. 

---
# R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning 

**Authors**: Zhuokun Chen, Zeren Chen, Jiahao He, Mingkui Tan, Jianfei Cai, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17307)  

**Abstract**: Chain-of-thought (CoT) reasoning enhances the problem-solving capabilities of large language models by encouraging step-by-step intermediate reasoning during inference. While effective, CoT introduces substantial computational overhead due to its reliance on autoregressive decoding over long token sequences. Existing acceleration strategies either reduce sequence length through early stopping or compressive reward designs, or improve decoding speed via speculative decoding with smaller models. However, speculative decoding suffers from limited speedup when the agreement between small and large models is low, and fails to exploit the potential advantages of small models in producing concise intermediate reasoning. In this paper, we present R-Stitch, a token-level, confidence-based hybrid decoding framework that accelerates CoT inference by switching between a small language model (SLM) and a large language model (LLM) along the reasoning trajectory. R-Stitch uses the SLM to generate tokens by default and delegates to the LLM only when the SLM's confidence falls below a threshold. This design avoids full-sequence rollback and selectively invokes the LLM on uncertain steps, preserving both efficiency and answer quality. R-Stitch is model-agnostic, training-free, and compatible with standard decoding pipelines. Experiments on math reasoning benchmarks demonstrate that R-Stitch achieves up to 85\% reduction in inference latency with negligible accuracy drop, highlighting its practical effectiveness in accelerating CoT reasoning. 

---
