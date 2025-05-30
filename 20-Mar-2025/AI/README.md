# Do Chains-of-Thoughts of Large Language Models Suffer from Hallucinations, Cognitive Biases, or Phobias in Bayesian Reasoning? 

**Authors**: Roberto Araya  

**Link**: [PDF](https://arxiv.org/pdf/2503.15268)  

**Abstract**: Learning to reason and carefully explain arguments is central to students' cognitive, mathematical, and computational thinking development. This is particularly challenging in problems under uncertainty and in Bayesian reasoning. With the new generation of large language models (LLMs) capable of reasoning using Chain-of-Thought (CoT), there is an excellent opportunity to learn with them as they explain their reasoning through a dialogue with their artificial internal voice. It is an engaging and excellent opportunity to learn Bayesian reasoning. Furthermore, given that different LLMs sometimes arrive at opposite solutions, CoT generates opportunities for deep learning by detailed comparisons of reasonings. However, unlike humans, we found that they do not autonomously explain using ecologically valid strategies like natural frequencies, whole objects, and embodied heuristics. This is unfortunate, as these strategies help humans avoid critical mistakes and have proven pedagogical value in Bayesian reasoning. In order to overcome these biases and aid understanding and learning, we included prompts that induce LLMs to use these strategies. We found that LLMs with CoT incorporate them but not consistently. They show persistent biases towards symbolic reasoning and avoidance or phobia of ecologically valid strategies. 

---
# World Models in Artificial Intelligence: Sensing, Learning, and Reasoning Like a Child 

**Authors**: Javier Del Ser, Jesus L. Lobo, Heimo Müller, Andreas Holzinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.15168)  

**Abstract**: World Models help Artificial Intelligence (AI) predict outcomes, reason about its environment, and guide decision-making. While widely used in reinforcement learning, they lack the structured, adaptive representations that even young children intuitively develop. Advancing beyond pattern recognition requires dynamic, interpretable frameworks inspired by Piaget's cognitive development theory. We highlight six key research areas -- physics-informed learning, neurosymbolic learning, continual learning, causal inference, human-in-the-loop AI, and responsible AI -- as essential for enabling true reasoning in AI. By integrating statistical learning with advances in these areas, AI can evolve from pattern recognition to genuine understanding, adaptation and reasoning capabilities. 

---
# Aligning Crowd-sourced Human Feedback for Reinforcement Learning on Code Generation by Large Language Models 

**Authors**: Man Fai Wong, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15129)  

**Abstract**: This paper studies how AI-assisted programming and large language models (LLM) improve software developers' ability via AI tools (LLM agents) like Github Copilot and Amazon CodeWhisperer, while integrating human feedback to enhance reinforcement learning (RLHF) with crowd-sourced computation to enhance text-to-code generation. Additionally, we demonstrate that our Bayesian optimization framework supports AI alignment in code generation by distributing the feedback collection burden, highlighting the value of collecting human feedback of good quality. Our empirical evaluations demonstrate the efficacy of this approach, showcasing how LLM agents can be effectively trained for improved text-to-code generation. Our Bayesian optimization framework can be designed for general domain-specific languages, promoting the alignment of large language model capabilities with human feedback in AI-assisted programming for code generation. 

---
# Reasoning Effort and Problem Complexity: A Scaling Analysis in LLMs 

**Authors**: Benjamin Estermann, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2503.15113)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable text generation capabilities, and recent advances in training paradigms have led to breakthroughs in their reasoning performance. In this work, we investigate how the reasoning effort of such models scales with problem complexity. We use the infinitely scalable Tents puzzle, which has a known linear-time solution, to analyze this scaling behavior. Our results show that reasoning effort scales with problem size, but only up to a critical problem complexity. Beyond this threshold, the reasoning effort does not continue to increase, and may even decrease. This observation highlights a critical limitation in the logical coherence of current LLMs as problem complexity increases, and underscores the need for strategies to improve reasoning scalability. Furthermore, our results reveal significant performance differences between current state-of-the-art reasoning models when faced with increasingly complex logical puzzles. 

---
# GraspCorrect: Robotic Grasp Correction via Vision-Language Model-Guided Feedback 

**Authors**: Sungjae Lee, Yeonjoo Hong, Kwang In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.15035)  

**Abstract**: Despite significant advancements in robotic manipulation, achieving consistent and stable grasping remains a fundamental challenge, often limiting the successful execution of complex tasks. Our analysis reveals that even state-of-the-art policy models frequently exhibit unstable grasping behaviors, leading to failure cases that create bottlenecks in real-world robotic applications. To address these challenges, we introduce GraspCorrect, a plug-and-play module designed to enhance grasp performance through vision-language model-guided feedback. GraspCorrect employs an iterative visual question-answering framework with two key components: grasp-guided prompting, which incorporates task-specific constraints, and object-aware sampling, which ensures the selection of physically feasible grasp candidates. By iteratively generating intermediate visual goals and translating them into joint-level actions, GraspCorrect significantly improves grasp stability and consistently enhances task success rates across existing policy models in the RLBench and CALVIN datasets. 

---
# Behaviour Discovery and Attribution for Explainable Reinforcement Learning 

**Authors**: Rishav Rishav, Somjit Nath, Vincent Michalski, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2503.14973)  

**Abstract**: Explaining the decisions made by reinforcement learning (RL) agents is critical for building trust and ensuring reliability in real-world applications. Traditional approaches to explainability often rely on saliency analysis, which can be limited in providing actionable insights. Recently, there has been growing interest in attributing RL decisions to specific trajectories within a dataset. However, these methods often generalize explanations to long trajectories, potentially involving multiple distinct behaviors. Often, providing multiple more fine grained explanations would improve clarity. In this work, we propose a framework for behavior discovery and action attribution to behaviors in offline RL trajectories. Our method identifies meaningful behavioral segments, enabling more precise and granular explanations associated with high level agent behaviors. This approach is adaptable across diverse environments with minimal modifications, offering a scalable and versatile solution for behavior discovery and attribution for explainable RL. 

---
# Generating Causal Explanations of Vehicular Agent Behavioural Interactions with Learnt Reward Profiles 

**Authors**: Rhys Howard, Nick Hawes, Lars Kunze  

**Link**: [PDF](https://arxiv.org/pdf/2503.14557)  

**Abstract**: Transparency and explainability are important features that responsible autonomous vehicles should possess, particularly when interacting with humans, and causal reasoning offers a strong basis to provide these qualities. However, even if one assumes agents act to maximise some concept of reward, it is difficult to make accurate causal inferences of agent planning without capturing what is of importance to the agent. Thus our work aims to learn a weighting of reward metrics for agents such that explanations for agent interactions can be causally inferred. We validate our approach quantitatively and qualitatively across three real-world driving datasets, demonstrating a functional improvement over previous methods and competitive performance across evaluation metrics. 

---
# TULIP: Towards Unified Language-Image Pretraining 

**Authors**: Zineng Tang, Long Lian, Seun Eisape, XuDong Wang, Roei Herzig, Adam Yala, Alane Suhr, Trevor Darrell, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15485)  

**Abstract**: Despite the recent success of image-text contrastive models like CLIP and SigLIP, these models often struggle with vision-centric tasks that demand high-fidelity image understanding, such as counting, depth estimation, and fine-grained object recognition. These models, by performing language alignment, tend to prioritize high-level semantics over visual understanding, weakening their image understanding. On the other hand, vision-focused models are great at processing visual information but struggle to understand language, limiting their flexibility for language-driven tasks. In this work, we introduce TULIP, an open-source, drop-in replacement for existing CLIP-like models. Our method leverages generative data augmentation, enhanced image-image and text-text contrastive learning, and image/text reconstruction regularization to learn fine-grained visual features while preserving global semantic alignment. Our approach, scaling to over 1B parameters, outperforms existing state-of-the-art (SOTA) models across multiple benchmarks, establishing a new SOTA zero-shot performance on ImageNet-1K, delivering up to a $2\times$ enhancement over SigLIP on RxRx1 in linear probing for few-shot classification, and improving vision-language models, achieving over $3\times$ higher scores than SigLIP on MMVP. Our code/checkpoints are available at this https URL 

---
# Value Profiles for Encoding Human Variation 

**Authors**: Taylor Sorensen, Pushkar Mishra, Roma Patel, Michael Henry Tessler, Michiel Bakker, Georgina Evans, Iason Gabriel, Noah Goodman, Verena Rieser  

**Link**: [PDF](https://arxiv.org/pdf/2503.15484)  

**Abstract**: Modelling human variation in rating tasks is crucial for enabling AI systems for personalization, pluralistic model alignment, and computational social science. We propose representing individuals using value profiles -- natural language descriptions of underlying values compressed from in-context demonstrations -- along with a steerable decoder model to estimate ratings conditioned on a value profile or other rater information. To measure the predictive information in rater representations, we introduce an information-theoretic methodology. We find that demonstrations contain the most information, followed by value profiles and then demographics. However, value profiles offer advantages in terms of scrutability, interpretability, and steerability due to their compressed natural language format. Value profiles effectively compress the useful information from demonstrations (>70% information preservation). Furthermore, clustering value profiles to identify similarly behaving individuals better explains rater variation than the most predictive demographic groupings. Going beyond test set performance, we show that the decoder models interpretably change ratings according to semantic profile differences, are well-calibrated, and can help explain instance-level disagreement by simulating an annotator population. These results demonstrate that value profiles offer novel, predictive ways to describe individual variation beyond demographics or group information. 

---
# Learning to Play Piano in the Real World 

**Authors**: Yves-Simon Zeulner, Sandeep Selvaraj, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2503.15481)  

**Abstract**: Towards the grand challenge of achieving human-level manipulation in robots, playing piano is a compelling testbed that requires strategic, precise, and flowing movements. Over the years, several works demonstrated hand-designed controllers on real world piano playing, while other works evaluated robot learning approaches on simulated piano scenarios. In this paper, we develop the first piano playing robotic system that makes use of learning approaches while also being deployed on a real world dexterous robot. Specifically, we make use of Sim2Real to train a policy in simulation using reinforcement learning before deploying the learned policy on a real world dexterous robot. In our experiments, we thoroughly evaluate the interplay between domain randomization and the accuracy of the dynamics model used in simulation. Moreover, we evaluate the robot's performance across multiple songs with varying complexity to study the generalization of our learned policy. By providing a proof-of-concept of learning to play piano in the real world, we want to encourage the community to adopt piano playing as a compelling benchmark towards human-level manipulation. We open-source our code and show additional videos at this https URL . 

---
# What Makes a Reward Model a Good Teacher? An Optimization Perspective 

**Authors**: Noam Razin, Zixuan Wang, Hubert Strauss, Stanley Wei, Jason D. Lee, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2503.15477)  

**Abstract**: The success of Reinforcement Learning from Human Feedback (RLHF) critically depends on the quality of the reward model. While this quality is primarily evaluated through accuracy, it remains unclear whether accuracy fully captures what makes a reward model an effective teacher. We address this question from an optimization perspective. First, we prove that regardless of how accurate a reward model is, if it induces low reward variance, then the RLHF objective suffers from a flat landscape. Consequently, even a perfectly accurate reward model can lead to extremely slow optimization, underperforming less accurate models that induce higher reward variance. We additionally show that a reward model that works well for one language model can induce low reward variance, and thus a flat objective landscape, for another. These results establish a fundamental limitation of evaluating reward models solely based on accuracy or independently of the language model they guide. Experiments using models of up to 8B parameters corroborate our theory, demonstrating the interplay between reward variance, accuracy, and reward maximization rate. Overall, our findings highlight that beyond accuracy, a reward model needs to induce sufficient variance for efficient optimization. 

---
# EgoDTM: Towards 3D-Aware Egocentric Video-Language Pretraining 

**Authors**: Boshen Xu, Yuting Mei, Xinbi Liu, Sipeng Zheng, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.15470)  

**Abstract**: Egocentric video-language pretraining has significantly advanced video representation learning. Humans perceive and interact with a fully 3D world, developing spatial awareness that extends beyond text-based understanding. However, most previous works learn from 1D text or 2D visual cues, such as bounding boxes, which inherently lack 3D understanding. To bridge this gap, we introduce EgoDTM, an Egocentric Depth- and Text-aware Model, jointly trained through large-scale 3D-aware video pretraining and video-text contrastive learning. EgoDTM incorporates a lightweight 3D-aware decoder to efficiently learn 3D-awareness from pseudo depth maps generated by depth estimation models. To further facilitate 3D-aware video pretraining, we enrich the original brief captions with hand-object visual cues by organically combining several foundation models. Extensive experiments demonstrate EgoDTM's superior performance across diverse downstream tasks, highlighting its superior 3D-aware visual understanding. Our code will be released at this https URL. 

---
# Dynamic Bi-Elman Attention Networks (DBEAN): Dual-Directional Context-Aware Representation Learning for Enhanced Text Classification 

**Authors**: ZhengLin Lai, MengYao Liao, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15469)  

**Abstract**: Text classification, a fundamental task in natural language processing (NLP), aims to categorize textual data into predefined labels. Traditional methods struggled with complex linguistic structures and semantic dependencies. The advent of deep learning, particularly recurrent neural networks (RNNs) and Transformer-based models, has significantly advanced the field by enabling nuanced feature extraction and context-aware predictions. Despite improvements, existing models exhibit limitations in balancing interpretability, computational efficiency, and long-range contextual understanding. This paper proposes the Dynamic Bidirectional Elman with Attention Network (DBEAN), which integrates bidirectional temporal modelling with self-attention mechanisms. DBEAN dynamically assigns weights to critical segments of input, improving contextual representation while maintaining computational efficiency. 

---
# From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment 

**Authors**: Jia-Nan Li, Jian Guan, Songhao Wu, Wei Wu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15463)  

**Abstract**: Large language models (LLMs) have traditionally been aligned through one-size-fits-all approaches that assume uniform human preferences, fundamentally overlooking the diversity in user values and needs. This paper introduces a comprehensive framework for scalable personalized alignment of LLMs. We establish a systematic preference space characterizing psychological and behavioral dimensions, alongside diverse persona representations for robust preference inference in real-world scenarios. Building upon this foundation, we introduce \textsc{AlignX}, a large-scale dataset of over 1.3 million personalized preference examples, and develop two complementary alignment approaches: \textit{in-context alignment} directly conditioning on persona representations and \textit{preference-bridged alignment} modeling intermediate preference distributions. Extensive experiments demonstrate substantial improvements over existing methods, with an average 17.06\% accuracy gain across four benchmarks while exhibiting a strong adaptation capability to novel preferences, robustness to limited user data, and precise preference controllability. These results validate our framework's effectiveness, advancing toward truly user-adaptive AI systems. 

---
# Di$\mathtt{[M]}$O: Distilling Masked Diffusion Models into One-step Generator 

**Authors**: Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2503.15457)  

**Abstract**: Masked Diffusion Models (MDMs) have emerged as a powerful generative modeling technique. Despite their remarkable results, they typically suffer from slow inference with several steps. In this paper, we propose Di$\mathtt{[M]}$O, a novel approach that distills masked diffusion models into a one-step generator. Di$\mathtt{[M]}$O addresses two key challenges: (1) the intractability of using intermediate-step information for one-step generation, which we solve through token-level distribution matching that optimizes model output logits by an 'on-policy framework' with the help of an auxiliary model; and (2) the lack of entropy in the initial distribution, which we address through a token initialization strategy that injects randomness while maintaining similarity to teacher training distribution. We show Di$\mathtt{[M]}$O's effectiveness on both class-conditional and text-conditional image generation, impressively achieving performance competitive to multi-step teacher outputs while drastically reducing inference time. To our knowledge, we are the first to successfully achieve one-step distillation of masked diffusion models and the first to apply discrete distillation to text-to-image generation, opening new paths for efficient generative modeling. 

---
# VenusFactory: A Unified Platform for Protein Engineering Data Retrieval and Language Model Fine-Tuning 

**Authors**: Yang Tan, Chen Liu, Jingyuan Gao, Banghao Wu, Mingchen Li, Ruilin Wang, Lingrong Zhang, Huiqun Yu, Guisheng Fan, Liang Hong, Bingxin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.15438)  

**Abstract**: Natural language processing (NLP) has significantly influenced scientific domains beyond human language, including protein engineering, where pre-trained protein language models (PLMs) have demonstrated remarkable success. However, interdisciplinary adoption remains limited due to challenges in data collection, task benchmarking, and application. This work presents VenusFactory, a versatile engine that integrates biological data retrieval, standardized task benchmarking, and modular fine-tuning of PLMs. VenusFactory supports both computer science and biology communities with choices of both a command-line execution and a Gradio-based no-code interface, integrating $40+$ protein-related datasets and $40+$ popular PLMs. All implementations are open-sourced on this https URL. 

---
# An extensive simulation study evaluating the interaction of resampling techniques across multiple causal discovery contexts 

**Authors**: Ritwick Banerjee, Bryan Andrews, Erich Kummerfeld  

**Link**: [PDF](https://arxiv.org/pdf/2503.15436)  

**Abstract**: Despite the accelerating presence of exploratory causal analysis in modern science and medicine, the available non-experimental methods for validating causal models are not well characterized. One of the most popular methods is to evaluate the stability of model features after resampling the data, similar to resampling methods for estimating confidence intervals in statistics. Many aspects of this approach have received little to no attention, however, such as whether the choice of resampling method should depend on the sample size, algorithms being used, or algorithm tuning parameters. We present theoretical results proving that certain resampling methods closely emulate the assignment of specific values to algorithm tuning parameters. We also report the results of extensive simulation experiments, which verify the theoretical result and provide substantial data to aid researchers in further characterizing resampling in the context of causal discovery analysis. Together, the theoretical work and simulation results provide specific guidance on how resampling methods and tuning parameters should be selected in practice. 

---
# Visual Position Prompt for MLLM based Visual Grounding 

**Authors**: Wei Tang, Yanpeng Sun, Qinying Gu, Zechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15426)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) excel at various image-related tasks, they encounter challenges in precisely aligning coordinates with spatial information within images, particularly in position-aware tasks such as visual grounding. This limitation arises from two key factors. First, MLLMs lack explicit spatial references, making it difficult to associate textual descriptions with precise image locations. Second, their feature extraction processes prioritize global context over fine-grained spatial details, leading to weak localization capability. To address this issue, we introduce VPP-LLaVA, an MLLM equipped with Visual Position Prompt (VPP) to improve its grounding capability. VPP-LLaVA integrates two complementary mechanisms. The global VPP overlays learnable, axis-like embeddings onto the input image to provide structured spatial cues. The local VPP focuses on fine-grained localization by incorporating position-aware queries, which suggests probable object locations. We also introduce a VPP-SFT dataset with 0.6M samples, consolidating high-quality visual grounding data into a compact format for efficient model training. Training on this dataset with VPP enhances the model's performance, achieving state-of-the-art results on standard grounding benchmarks despite using fewer training samples compared to other MLLMs like MiniGPT-v2, which rely on much larger datasets ($\sim$21M samples). The code and VPP-SFT dataset will be available at this https URL upon acceptance. 

---
# Probing the topology of the space of tokens with structured prompts 

**Authors**: Michael Robinson, Sourya Dey, Taisa Kushner  

**Link**: [PDF](https://arxiv.org/pdf/2503.15421)  

**Abstract**: This article presents a general and flexible method for prompting a large language model (LLM) to reveal its (hidden) token input embedding up to homeomorphism. Moreover, this article provides strong theoretical justification -- a mathematical proof for generic LLMs -- for why this method should be expected to work. With this method in hand, we demonstrate its effectiveness by recovering the token subspace of Llemma-7B. The results of this paper apply not only to LLMs but also to general nonlinear autoregressive processes. 

---
# Temporal Regularization Makes Your Video Generator Stronger 

**Authors**: Harold Haodong Chen, Haojian Huang, Xianfeng Wu, Yexin Liu, Yajing Bai, Wen-Jie Shu, Harry Yang, Ser-Nam Lim  

**Link**: [PDF](https://arxiv.org/pdf/2503.15417)  

**Abstract**: Temporal quality is a critical aspect of video generation, as it ensures consistent motion and realistic dynamics across frames. However, achieving high temporal coherence and diversity remains challenging. In this work, we explore temporal augmentation in video generation for the first time, and introduce FluxFlow for initial investigation, a strategy designed to enhance temporal quality. Operating at the data level, FluxFlow applies controlled temporal perturbations without requiring architectural modifications. Extensive experiments on UCF-101 and VBench benchmarks demonstrate that FluxFlow significantly improves temporal coherence and diversity across various video generation models, including U-Net, DiT, and AR-based architectures, while preserving spatial fidelity. These findings highlight the potential of temporal augmentation as a simple yet effective approach to advancing video generation quality. 

---
# Automated Processing of eXplainable Artificial Intelligence Outputs in Deep Learning Models for Fault Diagnostics of Large Infrastructures 

**Authors**: Giovanni Floreale, Piero Baraldi, Enrico Zio, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2503.15415)  

**Abstract**: Deep Learning (DL) models processing images to recognize the health state of large infrastructure components can exhibit biases and rely on non-causal shortcuts. eXplainable Artificial Intelligence (XAI) can address these issues but manually analyzing explanations generated by XAI techniques is time-consuming and prone to errors. This work proposes a novel framework that combines post-hoc explanations with semi-supervised learning to automatically identify anomalous explanations that deviate from those of correctly classified images and may therefore indicate model abnormal behaviors. This significantly reduces the workload for maintenance decision-makers, who only need to manually reclassify images flagged as having anomalous explanations. The proposed framework is applied to drone-collected images of insulator shells for power grid infrastructure monitoring, considering two different Convolutional Neural Networks (CNNs), GradCAM explanations and Deep Semi-Supervised Anomaly Detection. The average classification accuracy on two faulty classes is improved by 8% and maintenance operators are required to manually reclassify only 15% of the images. We compare the proposed framework with a state-of-the-art approach based on the faithfulness metric: the experimental results obtained demonstrate that the proposed framework consistently achieves F_1 scores larger than those of the faithfulness-based approach. Additionally, the proposed framework successfully identifies correct classifications that result from non-causal shortcuts, such as the presence of ID tags printed on insulator shells. 

---
# Towards efficient keyword spotting using spike-based time difference encoders 

**Authors**: Alejandro Pequeño-Zurro, Lyes Khacef, Stefano Panzeri, Elisabetta Chicca  

**Link**: [PDF](https://arxiv.org/pdf/2503.15402)  

**Abstract**: Keyword spotting in edge devices is becoming increasingly important as voice-activated assistants are widely used. However, its deployment is often limited by the extreme low-power constraints of the target embedded systems. Here, we explore the Temporal Difference Encoder (TDE) performance in keyword spotting. This recent neuron model encodes the time difference in instantaneous frequency and spike count to perform efficient keyword spotting with neuromorphic processors. We use the TIdigits dataset of spoken digits with a formant decomposition and rate-based encoding into spikes. We compare three Spiking Neural Networks (SNNs) architectures to learn and classify spatio-temporal signals. The proposed SNN architectures are made of three layers with variation in its hidden layer composed of either (1) feedforward TDE, (2) feedforward Current-Based Leaky Integrate-and-Fire (CuBa-LIF), or (3) recurrent CuBa-LIF neurons. We first show that the spike trains of the frequency-converted spoken digits have a large amount of information in the temporal domain, reinforcing the importance of better exploiting temporal encoding for such a task. We then train the three SNNs with the same number of synaptic weights to quantify and compare their performance based on the accuracy and synaptic operations. The resulting accuracy of the feedforward TDE network (89%) is higher than the feedforward CuBa-LIF network (71%) and close to the recurrent CuBa-LIF network (91%). However, the feedforward TDE-based network performs 92% fewer synaptic operations than the recurrent CuBa-LIF network with the same amount of synapses. In addition, the results of the TDE network are highly interpretable and correlated with the frequency and timescale features of the spoken keywords in the dataset. Our findings suggest that the TDE is a promising neuron model for scalable event-driven processing of spatio-temporal patterns. 

---
# CCDP: Composition of Conditional Diffusion Policies with Guided Sampling 

**Authors**: Amirreza Razmjoo, Sylvain Calinon, Michael Gienger, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15386)  

**Abstract**: Imitation Learning offers a promising approach to learn directly from data without requiring explicit models, simulations, or detailed task definitions. During inference, actions are sampled from the learned distribution and executed on the robot. However, sampled actions may fail for various reasons, and simply repeating the sampling step until a successful action is obtained can be inefficient. In this work, we propose an enhanced sampling strategy that refines the sampling distribution to avoid previously unsuccessful actions. We demonstrate that by solely utilizing data from successful demonstrations, our method can infer recovery actions without the need for additional exploratory behavior or a high-level controller. Furthermore, we leverage the concept of diffusion model decomposition to break down the primary problem (which may require long-horizon history to manage failures) into multiple smaller, more manageable sub-problems in learning, data collection, and inference, thereby enabling the system to adapt to variable failure counts. Our approach yields a low-level controller that dynamically adjusts its sampling space to improve efficiency when prior samples fall short. We validate our method across several tasks, including door opening with unknown directions, object manipulation, and button-searching scenarios, demonstrating that our approach outperforms traditional baselines. 

---
# Real-world validation of a multimodal LLM-powered pipeline for High-Accuracy Clinical Trial Patient Matching leveraging EHR data 

**Authors**: Anatole Callies, Quentin Bodinier, Philippe Ravaud, Kourosh Davarpanah  

**Link**: [PDF](https://arxiv.org/pdf/2503.15374)  

**Abstract**: Background: Patient recruitment in clinical trials is hindered by complex eligibility criteria and labor-intensive chart reviews. Prior research using text-only models have struggled to address this problem in a reliable and scalable way due to (1) limited reasoning capabilities, (2) information loss from converting visual records to text, and (3) lack of a generic EHR integration to extract patient data.
Methods: We introduce a broadly applicable, integration-free, LLM-powered pipeline that automates patient-trial matching using unprocessed documents extracted from EHRs. Our approach leverages (1) the new reasoning-LLM paradigm, enabling the assessment of even the most complex criteria, (2) visual capabilities of latest LLMs to interpret medical records without lossy image-to-text conversions, and (3) multimodal embeddings for efficient medical record search. The pipeline was validated on the n2c2 2018 cohort selection dataset (288 diabetic patients) and a real-world dataset composed of 485 patients from 30 different sites matched against 36 diverse trials.
Results: On the n2c2 dataset, our method achieved a new state-of-the-art criterion-level accuracy of 93\%. In real-world trials, the pipeline yielded an accuracy of 87\%, undermined by the difficulty to replicate human decision-making when medical records lack sufficient information. Nevertheless, users were able to review overall eligibility in under 9 minutes per patient on average, representing an 80\% improvement over traditional manual chart reviews.
Conclusion: This pipeline demonstrates robust performance in clinical trial patient matching without requiring custom integration with site systems or trial-specific tailoring, thereby enabling scalable deployment across sites seeking to leverage AI for patient matching. 

---
# Optimizing Decomposition for Optimal Claim Verification 

**Authors**: Yining Lu, Noah Ziems, Hy Dang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15354)  

**Abstract**: Current research on the \textit{Decompose-Then-Verify} paradigm for evaluating the factuality of long-form text typically treats decomposition and verification in isolation, overlooking their interactions and potential misalignment. We find that existing decomposition policies, typically hand-crafted demonstrations, do not align well with downstream verifiers in terms of atomicity -- a novel metric quantifying information density -- leading to suboptimal verification results. We formulate finding the optimal decomposition policy for optimal verification as a bilevel optimization problem. To approximate a solution for this strongly NP-hard problem, we propose dynamic decomposition, a reinforcement learning framework that leverages verifier feedback to learn a policy for dynamically decomposing claims to verifier-preferred atomicity. Experimental results show that dynamic decomposition outperforms existing decomposition policies, improving verification confidence by 0.07 and accuracy by 0.12 (on a 0-1 scale) on average across varying verifiers, datasets, and atomcities of input claims. 

---
# Leveraging Perfect Multimodal Alignment and Gaussian Assumptions for Cross-modal Transfer 

**Authors**: Abhi Kamboj, Minh N. Do  

**Link**: [PDF](https://arxiv.org/pdf/2503.15352)  

**Abstract**: Multimodal alignment aims to construct a joint latent vector space where two modalities representing the same concept map to the same vector. We formulate this as an inverse problem and show that under certain conditions perfect alignment can be achieved. We then address a specific application of alignment referred to as cross-modal transfer. Unsupervised cross-modal transfer aims to leverage a model trained with one modality to perform inference on another modality, without any labeled fine-tuning on the new modality. Assuming that semantic classes are represented as a mixture of Gaussians in the latent space, we show how cross-modal transfer can be performed by projecting the data points from the representation space onto different subspaces representing each modality. Our experiments on synthetic multimodal Gaussian data verify the effectiveness of our perfect alignment and cross-modal transfer method. We hope these findings inspire further exploration of the applications of perfect alignment and the use of Gaussian models for cross-modal learning. 

---
# TruthLens:A Training-Free Paradigm for DeepFake Detection 

**Authors**: Ritabrata Chakraborty, Rajatsubhra Chakraborty, Ali Khaleghi Rahimian, Thomas MacDougall  

**Link**: [PDF](https://arxiv.org/pdf/2503.15342)  

**Abstract**: The proliferation of synthetic images generated by advanced AI models poses significant challenges in identifying and understanding manipulated visual content. Current fake image detection methods predominantly rely on binary classification models that focus on accuracy while often neglecting interpretability, leaving users without clear insights into why an image is deemed real or fake. To bridge this gap, we introduce TruthLens, a novel training-free framework that reimagines deepfake detection as a visual question-answering (VQA) task. TruthLens utilizes state-of-the-art large vision-language models (LVLMs) to observe and describe visual artifacts and combines this with the reasoning capabilities of large language models (LLMs) like GPT-4 to analyze and aggregate evidence into informed decisions. By adopting a multimodal approach, TruthLens seamlessly integrates visual and semantic reasoning to not only classify images as real or fake but also provide interpretable explanations for its decisions. This transparency enhances trust and provides valuable insights into the artifacts that signal synthetic content. Extensive evaluations demonstrate that TruthLens outperforms conventional methods, achieving high accuracy on challenging datasets while maintaining a strong emphasis on explainability. By reframing deepfake detection as a reasoning-driven process, TruthLens establishes a new paradigm in combating synthetic media, combining cutting-edge performance with interpretability to address the growing threats of visual disinformation. 

---
# Challenges and Trends in Egocentric Vision: A Survey 

**Authors**: Xiang Li, Heqian Qiu, Lanxiao Wang, Hanwen Zhang, Chenghao Qi, Linfeng Han, Huiyu Xiong, Hongliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15275)  

**Abstract**: With the rapid development of artificial intelligence technologies and wearable devices, egocentric vision understanding has emerged as a new and challenging research direction, gradually attracting widespread attention from both academia and industry. Egocentric vision captures visual and multimodal data through cameras or sensors worn on the human body, offering a unique perspective that simulates human visual experiences. This paper provides a comprehensive survey of the research on egocentric vision understanding, systematically analyzing the components of egocentric scenes and categorizing the tasks into four main areas: subject understanding, object understanding, environment understanding, and hybrid understanding. We explore in detail the sub-tasks within each category. We also summarize the main challenges and trends currently existing in the field. Furthermore, this paper presents an overview of high-quality egocentric vision datasets, offering valuable resources for future research. By summarizing the latest advancements, we anticipate the broad applications of egocentric vision technologies in fields such as augmented reality, virtual reality, and embodied intelligence, and propose future research directions based on the latest developments in the field. 

---
# MAMM-Refine: A Recipe for Improving Faithfulness in Generation with Multi-Agent Collaboration 

**Authors**: David Wan, Justin Chih-Yao Chen, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.15272)  

**Abstract**: Multi-agent collaboration among models has shown promise in reasoning tasks but is underexplored in long-form generation tasks like summarization and question-answering. We extend multi-agent multi-model reasoning to generation, specifically to improving faithfulness through refinement, i.e., revising model-generated outputs to remove factual inconsistencies. We investigate how iterative collaboration among multiple instances and types of large language models (LLMs) enhances subtasks in the refinement process, such as error detection, critiquing unfaithful sentences, and making corrections based on critiques. We design intrinsic evaluations for each subtask, with our findings indicating that both multi-agent (multiple instances) and multi-model (diverse LLM types) approaches benefit error detection and critiquing. Additionally, reframing critiquing and refinement as reranking rather than generation tasks improves multi-agent performance. We consolidate these insights into a final "recipe" called Multi-Agent Multi-Model Refinement (MAMM-Refine), where multi-agent and multi-model collaboration significantly boosts performance on three summarization datasets as well as on long-form question answering, demonstrating the effectiveness and generalizability of our recipe. 

---
# Automated Non-Functional Requirements Generation in Software Engineering with Large Language Models: A Comparative Study 

**Authors**: Jomar Thomas Almonte, Santhosh Anitha Boominathan, Nathalia Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2503.15248)  

**Abstract**: Neglecting non-functional requirements (NFRs) early in software development can lead to critical challenges. Despite their importance, NFRs are often overlooked or difficult to identify, impacting software quality. To support requirements engineers in eliciting NFRs, we developed a framework that leverages Large Language Models (LLMs) to derive quality-driven NFRs from functional requirements (FRs). Using a custom prompting technique within a Deno-based pipeline, the system identifies relevant quality attributes for each functional requirement and generates corresponding NFRs, aiding systematic integration. A crucial aspect is evaluating the quality and suitability of these generated requirements. Can LLMs produce high-quality NFR suggestions? Using 34 functional requirements - selected as a representative subset of 3,964 FRs-the LLMs inferred applicable attributes based on the ISO/IEC 25010:2023 standard, generating 1,593 NFRs. A horizontal evaluation covered three dimensions: NFR validity, applicability of quality attributes, and classification precision. Ten industry software quality evaluators, averaging 13 years of experience, assessed a subset for relevance and quality. The evaluation showed strong alignment between LLM-generated NFRs and expert assessments, with median validity and applicability scores of 5.0 (means: 4.63 and 4.59, respectively) on a 1-5 scale. In the classification task, 80.4% of LLM-assigned attributes matched expert choices, with 8.3% near misses and 11.3% mismatches. A comparative analysis of eight LLMs highlighted variations in performance, with gemini-1.5-pro exhibiting the highest attribute accuracy, while llama-3.3-70B achieved higher validity and applicability scores. These findings provide insights into the feasibility of using LLMs for automated NFR generation and lay the foundation for further exploration of AI-assisted requirements engineering. 

---
# BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity? 

**Authors**: Pierre Chambon, Baptiste Roziere, Benoit Sagot, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2503.15242)  

**Abstract**: We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes of set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their strengths and weaknesses in handling complexity requirements. In particular, token-space reasoning models are unrivaled in code generation but not in complexity understanding, hinting that they may not generalize well to tasks for which no reward was given at training time. 

---
# Exploring Large Language Models for Word Games:Who is the Spy? 

**Authors**: Chentian Wei, Jiewei Chen, Jinzhu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15235)  

**Abstract**: Word games hold significant research value for natural language processing (NLP), game theory, and related fields due to their rule-based and situational nature. This study explores how large language models (LLMs) can be effectively involved in word games and proposes a training-free framework. "Shei Shi Wo Di" or "Who is the Spy" in English, is a classic word game. Using this game as an example, we introduce a Chain-of-Thought (CoT)-based scheduling framework to enable LLMs to achieve excellent performance in tasks such as inferring role words and disguising their identities. We evaluate the framework's performance based on game success rates and the accuracy of the LLM agents' analytical results. Experimental results affirm the framework's effectiveness, demonstrating notable improvements in LLM performance across multiple datasets. This work highlights the potential of LLMs in mastering situational reasoning and social interactions within structured game environments. Our code is publicly available at this https URL. 

---
# CoE: Chain-of-Explanation via Automatic Visual Concept Circuit Description and Polysemanticity Quantification 

**Authors**: Wenlong Yu, Qilong Wang, Chuang Liu, Dong Li, Qinghua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15234)  

**Abstract**: Explainability is a critical factor influencing the wide deployment of deep vision models (DVMs). Concept-based post-hoc explanation methods can provide both global and local insights into model decisions. However, current methods in this field face challenges in that they are inflexible to automatically construct accurate and sufficient linguistic explanations for global concepts and local circuits. Particularly, the intrinsic polysemanticity in semantic Visual Concepts (VCs) impedes the interpretability of concepts and DVMs, which is underestimated severely. In this paper, we propose a Chain-of-Explanation (CoE) approach to address these issues. Specifically, CoE automates the decoding and description of VCs to construct global concept explanation datasets. Further, to alleviate the effect of polysemanticity on model explainability, we design a concept polysemanticity disentanglement and filtering mechanism to distinguish the most contextually relevant concept atoms. Besides, a Concept Polysemanticity Entropy (CPE), as a measure of model interpretability, is formulated to quantify the degree of concept uncertainty. The modeling of deterministic concepts is upgraded to uncertain concept atom distributions. Finally, CoE automatically enables linguistic local explanations of the decision-making process of DVMs by tracing the concept circuit. GPT-4o and human-based experiments demonstrate the effectiveness of CPE and the superiority of CoE, achieving an average absolute improvement of 36% in terms of explainability scores. 

---
# A Personalized Data-Driven Generative Model of Human Motion 

**Authors**: Angelo Di Porzio, Marco Coraggio  

**Link**: [PDF](https://arxiv.org/pdf/2503.15225)  

**Abstract**: The deployment of autonomous virtual avatars (in extended reality) and robots in human group activities - such as rehabilitation therapy, sports, and manufacturing - is expected to increase as these technologies become more pervasive. Designing cognitive architectures and control strategies to drive these agents requires realistic models of human motion. However, existing models only provide simplified descriptions of human motor behavior. In this work, we propose a fully data-driven approach, based on Long Short-Term Memory neural networks, to generate original motion that captures the unique characteristics of specific individuals. We validate the architecture using real data of scalar oscillatory motion. Extensive analyses show that our model effectively replicates the velocity distribution and amplitude envelopes of the individual it was trained on, remaining different from other individuals, and outperforming state-of-the-art models in terms of similarity to human data. 

---
# When Pigs Get Sick: Multi-Agent AI for Swine Disease Detection 

**Authors**: Tittaya Mairittha, Tanakon Sawanglok, Panuwit Raden, Sorrawit Treesuk  

**Link**: [PDF](https://arxiv.org/pdf/2503.15204)  

**Abstract**: Swine disease surveillance is critical to the sustainability of global agriculture, yet its effectiveness is frequently undermined by limited veterinary resources, delayed identification of cases, and variability in diagnostic accuracy. To overcome these barriers, we introduce a novel AI-powered, multi-agent diagnostic system that leverages Retrieval-Augmented Generation (RAG) to deliver timely, evidence-based disease detection and clinical guidance. By automatically classifying user inputs into either Knowledge Retrieval Queries or Symptom-Based Diagnostic Queries, the system ensures targeted information retrieval and facilitates precise diagnostic reasoning. An adaptive questioning protocol systematically collects relevant clinical signs, while a confidence-weighted decision fusion mechanism integrates multiple diagnostic hypotheses to generate robust disease predictions and treatment recommendations. Comprehensive evaluations encompassing query classification, disease diagnosis, and knowledge retrieval demonstrate that the system achieves high accuracy, rapid response times, and consistent reliability. By providing a scalable, AI-driven diagnostic framework, this approach enhances veterinary decision-making, advances sustainable livestock management practices, and contributes substantively to the realization of global food security. 

---
# A Unified Framework for Real-Time Failure Handling in Robotics Using Vision-Language Models, Reactive Planner and Behavior Trees 

**Authors**: Faseeh Ahmad, Hashim Ismail, Jonathan Styrud, Maj Stenmark, Volker Krueger  

**Link**: [PDF](https://arxiv.org/pdf/2503.15202)  

**Abstract**: Robotic systems often face execution failures due to unexpected obstacles, sensor errors, or environmental changes. Traditional failure recovery methods rely on predefined strategies or human intervention, making them less adaptable. This paper presents a unified failure recovery framework that combines Vision-Language Models (VLMs), a reactive planner, and Behavior Trees (BTs) to enable real-time failure handling. Our approach includes pre-execution verification, which checks for potential failures before execution, and reactive failure handling, which detects and corrects failures during execution by verifying existing BT conditions, adding missing preconditions and, when necessary, generating new skills. The framework uses a scene graph for structured environmental perception and an execution history for continuous monitoring, enabling context-aware and adaptive failure handling. We evaluate our framework through real-world experiments with an ABB YuMi robot on tasks like peg insertion, object sorting, and drawer placement, as well as in AI2-THOR simulator. Compared to using pre-execution and reactive methods separately, our approach achieves higher task success rates and greater adaptability. Ablation studies highlight the importance of VLM-based reasoning, structured scene representation, and execution history tracking for effective failure recovery in robotics. 

---
# 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation 

**Authors**: Gyeongrok Oh, Sungjune Kim, Heeju Ko, Hyung-gun Chi, Jinkyu Kim, Dongwook Lee, Daehyun Ji, Sungjoon Choi, Sujin Jang, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.15185)  

**Abstract**: The resolution of voxel queries significantly influences the quality of view transformation in camera-based 3D occupancy prediction. However, computational constraints and the practical necessity for real-time deployment require smaller query resolutions, which inevitably leads to an information loss. Therefore, it is essential to encode and preserve rich visual details within limited query sizes while ensuring a comprehensive representation of 3D occupancy. To this end, we introduce ProtoOcc, a novel occupancy network that leverages prototypes of clustered image segments in view transformation to enhance low-resolution context. In particular, the mapping of 2D prototypes onto 3D voxel queries encodes high-level visual geometries and complements the loss of spatial information from reduced query resolutions. Additionally, we design a multi-perspective decoding strategy to efficiently disentangle the densely compressed visual cues into a high-dimensional 3D occupancy scene. Experimental results on both Occ3D and SemanticKITTI benchmarks demonstrate the effectiveness of the proposed method, showing clear improvements over the baselines. More importantly, ProtoOcc achieves competitive performance against the baselines even with 75\% reduced voxel resolution. 

---
# Foundation models may exhibit staged progression in novel CBRN threat disclosure 

**Authors**: Kevin M Esvelt  

**Link**: [PDF](https://arxiv.org/pdf/2503.15182)  

**Abstract**: The extent to which foundation models can disclose novel chemical, biological, radiation, and nuclear (CBRN) threats to expert users is unclear due to a lack of test cases. I leveraged the unique opportunity presented by an upcoming publication describing a novel catastrophic biothreat - "Technical Report on Mirror Bacteria: Feasibility and Risks" - to conduct a small controlled study before it became public. Graduate-trained biologists tasked with predicting the consequences of releasing mirror E. coli showed no significant differences in rubric-graded accuracy using Claude Sonnet 3.5 new (n=10) or web search only (n=2); both groups scored comparably to a web baseline (28 and 43 versus 36). However, Sonnet reasoned correctly when prompted by a report author, but a smaller model, Haiku 3.5, failed even with author guidance (80 versus 5). These results suggest distinct stages of model capability: Haiku is unable to reason about mirror life even with threat-aware expert guidance (Stage 1), while Sonnet correctly reasons only with threat-aware prompting (Stage 2). Continued advances may allow future models to disclose novel CBRN threats to naive experts (Stage 3) or unskilled users (Stage 4). While mirror life represents only one case study, monitoring new models' ability to reason about privately known threats may allow protective measures to be implemented before widespread disclosure. 

---
# Multi-Agent Actor-Critic with Harmonic Annealing Pruning for Dynamic Spectrum Access Systems 

**Authors**: George Stamatelis, Angelos-Nikolaos Kanatas, George C. Alexandropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.15172)  

**Abstract**: Multi-Agent Deep Reinforcement Learning (MADRL) has emerged as a powerful tool for optimizing decentralized decision-making systems in complex settings, such as Dynamic Spectrum Access (DSA). However, deploying deep learning models on resource-constrained edge devices remains challenging due to their high computational cost. To address this challenge, in this paper, we present a novel sparse recurrent MARL framework integrating gradual neural network pruning into the independent actor global critic paradigm. Additionally, we introduce a harmonic annealing sparsity scheduler, which achieves comparable, and in certain cases superior, performance to standard linear and polynomial pruning schedulers at large sparsities. Our experimental investigation demonstrates that the proposed DSA framework can discover superior policies, under diverse training conditions, outperforming conventional DSA, MADRL baselines, and state-of-the-art pruning techniques. 

---
# Comparing Llama3 and DeepSeekR1 on Biomedical Text Classification Tasks 

**Authors**: Yuting Guo, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2503.15169)  

**Abstract**: This study compares the performance of two open-source large language models (LLMs)-Llama3-70B and DeepSeekR1-distill-Llama3-70B-on six biomedical text classification tasks. Four tasks involve data from social media, while two tasks focus on clinical notes from electronic health records, and all experiments were performed in zero-shot settings. Performance metrics, including precision, recall, and F1 scores, were measured for each task, along with their 95% confidence intervals. Results demonstrated that DeepSeekR1-distill-Llama3-70B generally performs better in terms of precision on most tasks, with mixed results on recall. While the zero-shot LLMs demonstrated high F1 scores for some tasks, they grossly underperformed on others, for data from both sources. The findings suggest that model selection should be guided by the specific requirements of the health-related text classification tasks, particularly when considering the precision-recall trade-offs, and that, in the presence of annotated data, supervised classification approaches may be more reliable than zero-shot LLMs. 

---
# Volumetric Reconstruction From Partial Views for Task-Oriented Grasping 

**Authors**: Fujian Yan, Hui Li, Hongsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2503.15167)  

**Abstract**: Object affordance and volumetric information are essential in devising effective grasping strategies under task-specific constraints. This paper presents an approach for inferring suitable grasping strategies from limited partial views of an object. To achieve this, a recurrent generative adversarial network (R-GAN) was proposed by incorporating a recurrent generator with long short-term memory (LSTM) units for it to process a variable number of depth scans. To determine object affordances, the AffordPose knowledge dataset is utilized as prior knowledge. Affordance retrieving is defined by the volume similarity measured via Chamfer Distance and action similarities. A Proximal Policy Optimization (PPO) reinforcement learning model is further implemented to refine the retrieved grasp strategies for task-oriented grasping. The retrieved grasp strategies were evaluated on a dual-arm mobile manipulation robot with an overall grasping accuracy of 89% for four tasks: lift, handle grasp, wrap grasp, and press. 

---
# Machine Unlearning in Hyperbolic vs. Euclidean Multimodal Contrastive Learning: Adapting Alignment Calibration to MERU 

**Authors**: Àlex Pujol Vidal, Sergio Escalera, Kamal Nasrollahi, Thomas B. Moeslund  

**Link**: [PDF](https://arxiv.org/pdf/2503.15166)  

**Abstract**: Machine unlearning methods have become increasingly important for selective concept removal in large pre-trained models. While recent work has explored unlearning in Euclidean contrastive vision-language models, the effectiveness of concept removal in hyperbolic spaces remains unexplored. This paper investigates machine unlearning in hyperbolic contrastive learning by adapting Alignment Calibration to MERU, a model that embeds images and text in hyperbolic space to better capture semantic hierarchies. Through systematic experiments and ablation studies, we demonstrate that hyperbolic geometry offers distinct advantages for concept removal, achieving near perfect forgetting with reasonable performance on retained concepts, particularly when scaling to multiple concept removal. Our approach introduces hyperbolic-specific components including entailment calibration and norm regularization that leverage the unique properties of hyperbolic space. Comparative analysis with Euclidean models reveals fundamental differences in unlearning dynamics, with hyperbolic unlearning reorganizing the semantic hierarchy while Euclidean approaches merely disconnect cross-modal associations. These findings not only advance machine unlearning techniques but also provide insights into the geometric properties that influence concept representation and removal in multimodal models. Source code available at this https URL 

---
# A Foundational Theory for Decentralized Sensory Learning 

**Authors**: Linus Mårtensson, Jonas M.D. Enander, Udaya B. Rongala, Henrik Jörntell  

**Link**: [PDF](https://arxiv.org/pdf/2503.15130)  

**Abstract**: In both neuroscience and artificial intelligence, popular functional frameworks and neural network formulations operate by making use of extrinsic error measurements and global learning algorithms. Through a set of conjectures based on evolutionary insights on the origin of cellular adaptive mechanisms, we reinterpret the core meaning of sensory signals to allow the brain to be interpreted as a negative feedback control system, and show how this could lead to local learning algorithms without the need for global error correction metrics. Thereby, a sufficiently good minima in sensory activity can be the complete reward signal of the network, as well as being both necessary and sufficient for biological learning to arise. We show that this method of learning was likely already present in the earliest unicellular life forms on earth. We show evidence that the same principle holds and scales to multicellular organisms where it in addition can lead to division of labour between cells. Available evidence shows that the evolution of the nervous system likely was an adaptation to more effectively communicate intercellular signals to support such division of labour. We therefore propose that the same learning principle that evolved already in the earliest unicellular life forms, i.e. negative feedback control of externally and internally generated sensor signals, has simply been scaled up to become a fundament of the learning we see in biological brains today. We illustrate diverse biological settings, from the earliest unicellular organisms to humans, where this operational principle appears to be a plausible interpretation of the meaning of sensor signals in biology, and how this relates to current neuroscientific theories and findings. 

---
# Increasing the Robustness of the Fine-tuned Multilingual Machine-Generated Text Detectors 

**Authors**: Dominik Macko, Robert Moro, Ivan Srba  

**Link**: [PDF](https://arxiv.org/pdf/2503.15128)  

**Abstract**: Since the proliferation of LLMs, there have been concerns about their misuse for harmful content creation and spreading. Recent studies justify such fears, providing evidence of LLM vulnerabilities and high potential of their misuse. Humans are no longer able to distinguish between high-quality machine-generated and authentic human-written texts. Therefore, it is crucial to develop automated means to accurately detect machine-generated content. It would enable to identify such content in online information space, thus providing an additional information about its credibility. This work addresses the problem by proposing a robust fine-tuning process of LLMs for the detection task, making the detectors more robust against obfuscation and more generalizable to out-of-distribution data. 

---
# Text-Derived Relational Graph-Enhanced Network for Skeleton-Based Action Segmentation 

**Authors**: Haoyu Ji, Bowen Chen, Weihong Ren, Wenze Huang, Zhihao Yang, Zhiyong Wang, Honghai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15126)  

**Abstract**: Skeleton-based Temporal Action Segmentation (STAS) aims to segment and recognize various actions from long, untrimmed sequences of human skeletal movements. Current STAS methods typically employ spatio-temporal modeling to establish dependencies among joints as well as frames, and utilize one-hot encoding with cross-entropy loss for frame-wise classification supervision. However, these methods overlook the intrinsic correlations among joints and actions within skeletal features, leading to a limited understanding of human movements. To address this, we propose a Text-Derived Relational Graph-Enhanced Network (TRG-Net) that leverages prior graphs generated by Large Language Models (LLM) to enhance both modeling and supervision. For modeling, the Dynamic Spatio-Temporal Fusion Modeling (DSFM) method incorporates Text-Derived Joint Graphs (TJG) with channel- and frame-level dynamic adaptation to effectively model spatial relations, while integrating spatio-temporal core features during temporal modeling. For supervision, the Absolute-Relative Inter-Class Supervision (ARIS) method employs contrastive learning between action features and text embeddings to regularize the absolute class distributions, and utilizes Text-Derived Action Graphs (TAG) to capture the relative inter-class relationships among action features. Additionally, we propose a Spatial-Aware Enhancement Processing (SAEP) method, which incorporates random joint occlusion and axial rotation to enhance spatial generalization. Performance evaluations on four public datasets demonstrate that TRG-Net achieves state-of-the-art results. 

---
# VIPER: Visual Perception and Explainable Reasoning for Sequential Decision-Making 

**Authors**: Mohamed Salim Aissi, Clemence Grislain, Mohamed Chetouani, Olivier Sigaud, Laure Soulier, Nicolas Thome  

**Link**: [PDF](https://arxiv.org/pdf/2503.15108)  

**Abstract**: While Large Language Models (LLMs) excel at reasoning on text and Vision-Language Models (VLMs) are highly effective for visual perception, applying those models for visual instruction-based planning remains a widely open problem. In this paper, we introduce VIPER, a novel framework for multimodal instruction-based planning that integrates VLM-based perception with LLM-based reasoning. Our approach uses a modular pipeline where a frozen VLM generates textual descriptions of image observations, which are then processed by an LLM policy to predict actions based on the task goal. We fine-tune the reasoning module using behavioral cloning and reinforcement learning, improving our agent's decision-making capabilities. Experiments on the ALFWorld benchmark show that VIPER significantly outperforms state-of-the-art visual instruction-based planners while narrowing the gap with purely text-based oracles. By leveraging text as an intermediate representation, VIPER also enhances explainability, paving the way for a fine-grained analysis of perception and reasoning components. 

---
# Diffusion-Based Forecasting for Uncertainty-Aware Model Predictive Control 

**Authors**: Stelios Zarifis, Ioannis Kordonis, Petros Maragos  

**Link**: [PDF](https://arxiv.org/pdf/2503.15095)  

**Abstract**: We propose Diffusion-Informed Model Predictive Control (D-I MPC), a generic framework for uncertainty-aware prediction and decision-making in partially observable stochastic systems by integrating diffusion-based time series forecasting models in Model Predictive Control algorithms. In our approach, a diffusion-based time series forecasting model is used to probabilistically estimate the evolution of the system's stochastic components. These forecasts are then incorporated into MPC algorithms to estimate future trajectories and optimize action selection under the uncertainty of the future. We evaluate the framework on the task of energy arbitrage, where a Battery Energy Storage System participates in the day-ahead electricity market of the New York state. Experimental results indicate that our model-based approach with a diffusion-based forecaster significantly outperforms both implementations with classical forecasting methods and model-free reinforcement learning baselines. 

---
# Towards Understanding the Safety Boundaries of DeepSeek Models: Evaluation and Findings 

**Authors**: Zonghao Ying, Guangyi Zheng, Yongxin Huang, Deyue Zhang, Wenxin Zhang, Quanchen Zou, Aishan Liu, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15092)  

**Abstract**: This study presents the first comprehensive safety evaluation of the DeepSeek models, focusing on evaluating the safety risks associated with their generated content. Our evaluation encompasses DeepSeek's latest generation of large language models, multimodal large language models, and text-to-image models, systematically examining their performance regarding unsafe content generation. Notably, we developed a bilingual (Chinese-English) safety evaluation dataset tailored to Chinese sociocultural contexts, enabling a more thorough evaluation of the safety capabilities of Chinese-developed models. Experimental results indicate that despite their strong general capabilities, DeepSeek models exhibit significant safety vulnerabilities across multiple risk dimensions, including algorithmic discrimination and sexual content. These findings provide crucial insights for understanding and improving the safety of large foundation models. Our code is available at this https URL. 

---
# Conjuring Positive Pairs for Efficient Unification of Representation Learning and Image Synthesis 

**Authors**: Imanol G. Estepa, Jesús M. Rodríguez-de-Vera, Ignacio Sarasúa, Bhalaji Nagarajan, Petia Radeva  

**Link**: [PDF](https://arxiv.org/pdf/2503.15060)  

**Abstract**: While representation learning and generative modeling seek to understand visual data, unifying both domains remains unexplored. Recent Unified Self-Supervised Learning (SSL) methods have started to bridge the gap between both paradigms. However, they rely solely on semantic token reconstruction, which requires an external tokenizer during training -- introducing a significant overhead. In this work, we introduce Sorcen, a novel unified SSL framework, incorporating a synergic Contrastive-Reconstruction objective. Our Contrastive objective, "Echo Contrast", leverages the generative capabilities of Sorcen, eliminating the need for additional image crops or augmentations during training. Sorcen "generates" an echo sample in the semantic token space, forming the contrastive positive pair. Sorcen operates exclusively on precomputed tokens, eliminating the need for an online token transformation during training, thereby significantly reducing computational overhead. Extensive experiments on ImageNet-1k demonstrate that Sorcen outperforms the previous Unified SSL SoTA by 0.4%, 1.48 FID, 1.76%, and 1.53% on linear probing, unconditional image generation, few-shot learning, and transfer learning, respectively, while being 60.8% more efficient. Additionally, Sorcen surpasses previous single-crop MIM SoTA in linear probing and achieves SoTA performance in unconditional image generation, highlighting significant improvements and breakthroughs in Unified SSL models. 

---
# Texture-Aware StarGAN for CT data harmonisation 

**Authors**: Francesco Di Feola, Ludovica Pompilio, Cecilia Assolito, Valerio Guarrasi, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2503.15058)  

**Abstract**: Computed Tomography (CT) plays a pivotal role in medical diagnosis; however, variability across reconstruction kernels hinders data-driven approaches, such as deep learning models, from achieving reliable and generalized performance. To this end, CT data harmonization has emerged as a promising solution to minimize such non-biological variances by standardizing data across different sources or conditions. In this context, Generative Adversarial Networks (GANs) have proved to be a powerful framework for harmonization, framing it as a style-transfer problem. However, GAN-based approaches still face limitations in capturing complex relationships within the images, which are essential for effective harmonization. In this work, we propose a novel texture-aware StarGAN for CT data harmonization, enabling one-to-many translations across different reconstruction kernels. Although the StarGAN model has been successfully applied in other domains, its potential for CT data harmonization remains unexplored. Furthermore, our approach introduces a multi-scale texture loss function that embeds texture information across different spatial and angular scales into the harmonization process, effectively addressing kernel-induced texture variations. We conducted extensive experimentation on a publicly available dataset, utilizing a total of 48667 chest CT slices from 197 patients distributed over three different reconstruction kernels, demonstrating the superiority of our method over the baseline StarGAN. 

---
# HAD-Gen: Human-like and Diverse Driving Behavior Modeling for Controllable Scenario Generation 

**Authors**: Cheng Wang, Lingxin Kong, Massimiliano Tamborski, Stefano V. Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2503.15049)  

**Abstract**: Simulation-based testing has emerged as an essential tool for verifying and validating autonomous vehicles (AVs). However, contemporary methodologies, such as deterministic and imitation learning-based driver models, struggle to capture the variability of human-like driving behavior. Given these challenges, we propose HAD-Gen, a general framework for realistic traffic scenario generation that simulates diverse human-like driving behaviors. The framework first clusters the vehicle trajectory data into different driving styles according to safety features. It then employs maximum entropy inverse reinforcement learning on each of the clusters to learn the reward function corresponding to each driving style. Using these reward functions, the method integrates offline reinforcement learning pre-training and multi-agent reinforcement learning algorithms to obtain general and robust driving policies. Multi-perspective simulation results show that our proposed scenario generation framework can simulate diverse, human-like driving behaviors with strong generalization capability. The proposed framework achieves a 90.96% goal-reaching rate, an off-road rate of 2.08%, and a collision rate of 6.91% in the generalization test, outperforming prior approaches by over 20% in goal-reaching performance. The source code is released at this https URL. 

---
# A Novel Channel Boosted Residual CNN-Transformer with Regional-Boundary Learning for Breast Cancer Detection 

**Authors**: Aamir Mehmood, Yue Hu, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15008)  

**Abstract**: Recent advancements in detecting tumors using deep learning on breast ultrasound images (BUSI) have demonstrated significant success. Deep CNNs and vision-transformers (ViTs) have demonstrated individually promising initial performance. However, challenges related to model complexity and contrast, texture, and tumor morphology variations introduce uncertainties that hinder the effectiveness of current methods. This study introduces a novel hybrid framework, CB-Res-RBCMT, combining customized residual CNNs and new ViT components for detailed BUSI cancer analysis. The proposed RBCMT uses stem convolution blocks with CNN Meet Transformer (CMT) blocks, followed by new Regional and boundary (RB) feature extraction operations for capturing contrast and morphological variations. Moreover, the CMT block incorporates global contextual interactions through multi-head attention, enhancing computational efficiency with a lightweight design. Additionally, the customized inverse residual and stem CNNs within the CMT effectively extract local texture information and handle vanishing gradients. Finally, the new channel-boosted (CB) strategy enriches the feature diversity of the limited dataset by combining the original RBCMT channels with transfer learning-based residual CNN-generated maps. These diverse channels are processed through a spatial attention block for optimal pixel selection, reducing redundancy and improving the discrimination of minor contrast and texture variations. The proposed CB-Res-RBCMT achieves an F1-score of 95.57%, accuracy of 95.63%, sensitivity of 96.42%, and precision of 94.79% on the standard harmonized stringent BUSI dataset, outperforming existing ViT and CNN methods. These results demonstrate the versatility of our integrated CNN-Transformer framework in capturing diverse features and delivering superior performance in BUSI cancer diagnosis. 

---
# Application of linear regression method to the deep reinforcement learning in continuous action cases 

**Authors**: Hisato Komatsu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14976)  

**Abstract**: The linear regression (LR) method offers the advantage that optimal parameters can be calculated relatively easily, although its representation capability is limited than that of the deep learning technique. To improve deep reinforcement learning, the Least Squares Deep Q Network (LS-DQN) method was proposed by Levine et al., which combines Deep Q Network (DQN) with LR method. However, the LS-DQN method assumes that the actions are discrete. In this study, we propose the Double Least Squares Deep Deterministic Policy Gradient (DLS-DDPG) method to address this limitation. This method combines the LR method with the Deep Deterministic Policy Gradient (DDPG) technique, one of the representative deep reinforcement learning algorithms for continuous action cases. Numerical experiments conducted in MuJoCo environments showed that the LR update improved performance at least in some tasks, although there are difficulties such as the inability to make the regularization terms small. 

---
# USAM-Net: A U-Net-based Network for Improved Stereo Correspondence and Scene Depth Estimation using Features from a Pre-trained Image Segmentation network 

**Authors**: Joseph Emmanuel DL Dayo, Prospero C. Naval Jr  

**Link**: [PDF](https://arxiv.org/pdf/2503.14950)  

**Abstract**: The increasing demand for high-accuracy depth estimation in autonomous driving and augmented reality applications necessitates advanced neural architectures capable of effectively leveraging multiple data modalities. In this context, we introduce the Unified Segmentation Attention Mechanism Network (USAM-Net), a novel convolutional neural network that integrates stereo image inputs with semantic segmentation maps and attention to enhance depth estimation performance. USAM-Net employs a dual-pathway architecture, which combines a pre-trained segmentation model (SAM) and a depth estimation model. The segmentation pathway preprocesses the stereo images to generate semantic masks, which are then concatenated with the stereo images as inputs to the depth estimation pathway. This integration allows the model to focus on important features such as object boundaries and surface textures which are crucial for accurate depth perception. Empirical evaluation on the DrivingStereo dataset demonstrates that USAM-Net achieves superior performance metrics, including a Global Difference (GD) of 3.61\% and an End-Point Error (EPE) of 0.88, outperforming traditional models such as CFNet, SegStereo, and iResNet. These results underscore the effectiveness of integrating segmentation information into stereo depth estimation tasks, highlighting the potential of USAM-Net in applications demanding high-precision depth data. 

---
# FAVOR-Bench: A Comprehensive Benchmark for Fine-Grained Video Motion Understanding 

**Authors**: Chongjun Tu, Lin Zhang, Pengtao Chen, Peng Ye, Xianfang Zeng, Wei Cheng, Gang Yu, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.14935)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown remarkable capabilities in video content understanding but still struggle with fine-grained motion comprehension. To comprehensively assess the motion understanding ability of existing MLLMs, we introduce FAVOR-Bench, comprising 1,776 videos with structured manual annotations of various motions. Our benchmark includes both close-ended and open-ended tasks. For close-ended evaluation, we carefully design 8,184 multiple-choice question-answer pairs spanning six distinct sub-tasks. For open-ended evaluation, we develop both a novel cost-efficient LLM-free and a GPT-assisted caption assessment method, where the former can enhance benchmarking interpretability and reproducibility. Comprehensive experiments with 21 state-of-the-art MLLMs reveal significant limitations in their ability to comprehend and describe detailed temporal dynamics in video motions. To alleviate this limitation, we further build FAVOR-Train, a dataset consisting of 17,152 videos with fine-grained motion annotations. The results of finetuning Qwen2.5-VL on FAVOR-Train yield consistent improvements on motion-related tasks of TVBench, MotionBench and our FAVOR-Bench. Comprehensive assessment results demonstrate that the proposed FAVOR-Bench and FAVOR-Train provide valuable tools to the community for developing more powerful video understanding models. Project page: \href{this https URL}{this https URL}. 

---
# Shushing! Let's Imagine an Authentic Speech from the Silent Video 

**Authors**: Jiaxin Ye, Hongming Shan  

**Link**: [PDF](https://arxiv.org/pdf/2503.14928)  

**Abstract**: Vision-guided speech generation aims to produce authentic speech from facial appearance or lip motions without relying on auditory signals, offering significant potential for applications such as dubbing in filmmaking and assisting individuals with aphonia. Despite recent progress, existing methods struggle to achieve unified cross-modal alignment across semantics, timbre, and emotional prosody from visual cues, prompting us to propose Consistent Video-to-Speech (CV2S) as an extended task to enhance cross-modal consistency. To tackle emerging challenges, we introduce ImaginTalk, a novel cross-modal diffusion framework that generates faithful speech using only visual input, operating within a discrete space. Specifically, we propose a discrete lip aligner that predicts discrete speech tokens from lip videos to capture semantic information, while an error detector identifies misaligned tokens, which are subsequently refined through masked language modeling with BERT. To further enhance the expressiveness of the generated speech, we develop a style diffusion transformer equipped with a face-style adapter that adaptively customizes identity and prosody dynamics across both the channel and temporal dimensions while ensuring synchronization with lip-aware semantic features. Extensive experiments demonstrate that ImaginTalk can generate high-fidelity speech with more accurate semantic details and greater expressiveness in timbre and emotion compared to state-of-the-art baselines. Demos are shown at our project page: this https URL. 

---
# A Semantic and Clean-label Backdoor Attack against Graph Convolutional Networks 

**Authors**: Jiazhu Dai, Haoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.14922)  

**Abstract**: Graph Convolutional Networks (GCNs) have shown excellent performance in graph-structured tasks such as node classification and graph classification. However, recent research has shown that GCNs are vulnerable to a new type of threat called the backdoor attack, where the adversary can inject a hidden backdoor into the GCNs so that the backdoored model performs well on benign samples, whereas its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. Clean-label backdoor attack and semantic backdoor attack are two new backdoor attacks to Deep Neural Networks (DNNs), they are more imperceptible and have posed new and serious threats. The semantic and clean-label backdoor attack is not fully explored in GCNs. In this paper, we propose a semantic and clean-label backdoor attack against GCNs under the context of graph classification to reveal the existence of this security vulnerability in GCNs. Specifically, SCLBA conducts an importance analysis on graph samples to select one type of node as semantic trigger, which is then inserted into the graph samples to create poisoning samples without changing the labels of the poisoning samples to the attacker-specified target label. We evaluate SCLBA on multiple datasets and the results show that SCLBA can achieve attack success rates close to 99% with poisoning rates of less than 3%, and with almost no impact on the performance of model on benign samples. 

---
# MASS: Mathematical Data Selection via Skill Graphs for Pretraining Large Language Models 

**Authors**: Jiazheng Li, Lu Yu, Qing Cui, Zhiqiang Zhang, Jun Zhou, Yanfang Ye, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14917)  

**Abstract**: High-quality data plays a critical role in the pretraining and fine-tuning of large language models (LLMs), even determining their performance ceiling to some degree. Consequently, numerous data selection methods have been proposed to identify subsets of data that can effectively and efficiently enhance model performance. However, most of these methods focus on general data selection and tend to overlook the specific nuances of domain-related data. In this paper, we introduce MASS, a \textbf{MA}thematical data \textbf{S}election framework using the \textbf{S}kill graph for pretraining LLMs in the mathematical reasoning domain. By taking into account the unique characteristics of mathematics and reasoning, we construct a skill graph that captures the mathematical skills and their interrelations from a reference dataset. This skill graph guides us in assigning quality scores to the target dataset, enabling us to select the top-ranked subset which is further used to pretrain LLMs. Experimental results demonstrate the efficiency and effectiveness of MASS across different model sizes (1B and 7B) and pretraining datasets (web data and synthetic data). Specifically, in terms of efficiency, models trained on subsets selected by MASS can achieve similar performance to models trained on the original datasets, with a significant reduction in the number of trained tokens - ranging from 50\% to 70\% fewer tokens. In terms of effectiveness, when trained on the same amount of tokens, models trained on the data selected by MASS outperform those trained on the original datasets by 3.3\% to 5.9\%. These results underscore the potential of MASS to improve both the efficiency and effectiveness of pretraining LLMs. 

---
# POSTA: A Go-to Framework for Customized Artistic Poster Generation 

**Authors**: Haoyu Chen, Xiaojie Xu, Wenbo Li, Jingjing Ren, Tian Ye, Songhua Liu, Ying-Cong Chen, Lei Zhu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14908)  

**Abstract**: Poster design is a critical medium for visual communication. Prior work has explored automatic poster design using deep learning techniques, but these approaches lack text accuracy, user customization, and aesthetic appeal, limiting their applicability in artistic domains such as movies and exhibitions, where both clear content delivery and visual impact are essential. To address these limitations, we present POSTA: a modular framework powered by diffusion models and multimodal large language models (MLLMs) for customized artistic poster generation. The framework consists of three modules. Background Diffusion creates a themed background based on user input. Design MLLM then generates layout and typography elements that align with and complement the background style. Finally, to enhance the poster's aesthetic appeal, ArtText Diffusion applies additional stylization to key text elements. The final result is a visually cohesive and appealing poster, with a fully modular process that allows for complete customization. To train our models, we develop the PosterArt dataset, comprising high-quality artistic posters annotated with layout, typography, and pixel-level stylized text segmentation. Our comprehensive experimental analysis demonstrates POSTA's exceptional controllability and design diversity, outperforming existing models in both text accuracy and aesthetic quality. 

---
# Deep Contrastive Unlearning for Language Models 

**Authors**: Estrid He, Tabinda Sarwar, Ibrahim Khalil, Xun Yi, Ke Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14900)  

**Abstract**: The past a few years have witnessed the great success of large language models, demonstrating powerful capabilities in comprehending textual data and generating human-like languages. Large language models achieve success by being trained on vast amounts of textual data, including online sources with copyrighted content and user-generated knowledge. However, this comes at a cost: the potential risk of exposing users' privacy and violating copyright protections. Thus, to safeguard individuals' "right to be forgotten", there has been increasing interests in machine unlearning -- the process of removing information carried by particular training samples from a model while not deteriorating its predictive quality. This is a challenging task due to the black-box nature of language models. Most existing studies focus on mitigating the impact of those forgot samples upon a model's outputs, and do not explicitly consider the geometric distributions of samples in the latent space of a model. To address this issue, we propose a machine unlearning framework, named Deep Contrastive Unlearning for fine-Tuning (DeepCUT) language models. Our proposed model achieves machine unlearning by directly optimizing the latent space of a model. Comprehensive experiments on real-world datasets demonstrate the effectiveness and efficiency of DeepCUT with consistent and significant improvement over baseline methods. 

---
# Mitigating Object Hallucinations in MLLMs via Multi-Frequency Perturbations 

**Authors**: Shuo Li, Jiajun Sun, Guodong Zheng, Xiaoran Fan, Yujiong Shen, Yi Lu, Zhiheng Xi, Yuming Yang, Wenming Tan, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14895)  

**Abstract**: Recently, multimodal large language models (MLLMs) have demonstrated remarkable performance in visual-language tasks. However, the authenticity of the responses generated by MLLMs is often compromised by object hallucinations. We identify that a key cause of these hallucinations is the model's over-susceptibility to specific image frequency features in detecting objects. In this paper, we introduce Multi-Frequency Perturbations (MFP), a simple, cost-effective, and pluggable method that leverages both low-frequency and high-frequency features of images to perturb visual feature representations and explicitly suppress redundant frequency-domain features during inference, thereby mitigating hallucinations. Experimental results demonstrate that our method significantly mitigates object hallucinations across various model architectures. Furthermore, as a training-time method, MFP can be combined with inference-time methods to achieve state-of-the-art performance on the CHAIR benchmark. 

---
# MetaLadder: Ascending Mathematical Solution Quality via Analogical-Problem Reasoning Transfer 

**Authors**: Honglin Lin, Zhuoshi Pan, Yu Li, Qizhi Pei, Xin Gao, Mengzhang Cai, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14891)  

**Abstract**: Large Language Models (LLMs) have demonstrated promising capabilities in solving mathematical reasoning tasks, leveraging Chain-of-Thought (CoT) data as a vital component in guiding answer generation. Current paradigms typically generate CoT and answers directly for a given problem, diverging from human problem-solving strategies to some extent. Humans often solve problems by recalling analogous cases and leveraging their solutions to reason about the current task. Inspired by this cognitive process, we propose \textbf{MetaLadder}, a novel framework that explicitly prompts LLMs to recall and reflect on meta-problems, those structurally or semantically analogous problems, alongside their CoT solutions before addressing the target problem. Additionally, we introduce a problem-restating mechanism to enhance the model's comprehension of the target problem by regenerating the original question, which further improves reasoning accuracy. Therefore, the model can achieve reasoning transfer from analogical problems, mimicking human-like "learning from examples" and generalization abilities. Extensive experiments on mathematical benchmarks demonstrate that our MetaLadder significantly boosts LLMs' problem-solving accuracy, largely outperforming standard CoT-based methods (\textbf{10.3\%} accuracy gain) and other methods. Our code and data has been released at this https URL. 

---
# Envisioning an AI-Enhanced Mental Health Ecosystem 

**Authors**: Kellie Yu Hui Sim, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2503.14883)  

**Abstract**: The rapid advancement of Large Language Models (LLMs), reasoning models, and agentic AI approaches coincides with a growing global mental health crisis, where increasing demand has not translated into adequate access to professional support, particularly for underserved populations. This presents a unique opportunity for AI to complement human-led interventions, offering scalable and context-aware support while preserving human connection in this sensitive domain. We explore various AI applications in peer support, self-help interventions, proactive monitoring, and data-driven insights, using a human-centred approach that ensures AI supports rather than replaces human interaction. However, AI deployment in mental health fields presents challenges such as ethical concerns, transparency, privacy risks, and risks of over-reliance. We propose a hybrid ecosystem where where AI assists but does not replace human providers, emphasising responsible deployment and evaluation. We also present some of our early work and findings in several of these AI applications. Finally, we outline future research directions for refining AI-enhanced interventions while adhering to ethical and culturally sensitive guidelines. 

---
# Exploring the Limits of KV Cache Compression in Visual Autoregressive Transformers 

**Authors**: Bo Chen, Xiaoyu Li, Yekun Ke, Yingyu Liang, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.14881)  

**Abstract**: A fundamental challenge in Visual Autoregressive models is the substantial memory overhead required during inference to store previously generated representations. Despite various attempts to mitigate this issue through compression techniques, prior works have not explicitly formalized the problem of KV-cache compression in this context. In this work, we take the first step in formally defining the KV-cache compression problem for Visual Autoregressive transformers. We then establish a fundamental negative result, proving that any mechanism for sequential visual token generation under attention-based architectures must use at least $\Omega(n^2 d)$ memory, when $d = \Omega(\log n)$, where $n$ is the number of tokens generated and $d$ is the embedding dimensionality. This result demonstrates that achieving truly sub-quadratic memory usage is impossible without additional structural constraints. Our proof is constructed via a reduction from a computational lower bound problem, leveraging randomized embedding techniques inspired by dimensionality reduction principles. Finally, we discuss how sparsity priors on visual representations can influence memory efficiency, presenting both impossibility results and potential directions for mitigating memory overhead. 

---
# Efficient Personalization of Quantized Diffusion Model without Backpropagation 

**Authors**: Hoigi Seo, Wongi Jeong, Kyungryeol Lee, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2503.14868)  

**Abstract**: Diffusion models have shown remarkable performance in image synthesis, but they demand extensive computational and memory resources for training, fine-tuning and inference. Although advanced quantization techniques have successfully minimized memory usage for inference, training and fine-tuning these quantized models still require large memory possibly due to dequantization for accurate computation of gradients and/or backpropagation for gradient-based algorithms. However, memory-efficient fine-tuning is particularly desirable for applications such as personalization that often must be run on edge devices like mobile phones with private data. In this work, we address this challenge by quantizing a diffusion model with personalization via Textual Inversion and by leveraging a zeroth-order optimization on personalization tokens without dequantization so that it does not require gradient and activation storage for backpropagation that consumes considerable memory. Since a gradient estimation using zeroth-order optimization is quite noisy for a single or a few images in personalization, we propose to denoise the estimated gradient by projecting it onto a subspace that is constructed with the past history of the tokens, dubbed Subspace Gradient. In addition, we investigated the influence of text embedding in image generation, leading to our proposed time steps sampling, dubbed Partial Uniform Timestep Sampling for sampling with effective diffusion timesteps. Our method achieves comparable performance to prior methods in image and text alignment scores for personalizing Stable Diffusion with only forward passes while reducing training memory demand up to $8.2\times$. 

---
# 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities 

**Authors**: Kevin Wang, Ishaan Javali, Michał Bortkiewicz, Tomasz Trzciński, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2503.14858)  

**Abstract**: Scaling up self-supervised learning has driven breakthroughs in language and vision, yet comparable progress has remained elusive in reinforcement learning (RL). In this paper, we study building blocks for self-supervised RL that unlock substantial improvements in scalability, with network depth serving as a critical factor. Whereas most RL papers in recent years have relied on shallow architectures (around 2 - 5 layers), we demonstrate that increasing the depth up to 1024 layers can significantly boost performance. Our experiments are conducted in an unsupervised goal-conditioned setting, where no demonstrations or rewards are provided, so an agent must explore (from scratch) and learn how to maximize the likelihood of reaching commanded goals. Evaluated on simulated locomotion and manipulation tasks, our approach increases performance by $2\times$ - $50\times$. Increasing the model depth not only increases success rates but also qualitatively changes the behaviors learned. 

---
# Project Jenkins: Turning Monkey Neural Data into Robotic Arm Movement, and Back 

**Authors**: Andrii Zahorodnii, Dima Yanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2503.14847)  

**Abstract**: Project Jenkins explores how neural activity in the brain can be decoded into robotic movement and, conversely, how movement patterns can be used to generate synthetic neural data. Using real neural data recorded from motor and premotor cortex areas of a macaque monkey named Jenkins, we develop models for decoding (converting brain signals into robotic arm movements) and encoding (simulating brain activity corresponding to a given movement). For the interface between the brain simulation and the physical world, we utilized Koch v1.1 leader and follower robotic arms. We developed an interactive web console that allows users to generate synthetic brain data from joystick movements in real time. Our results are a step towards brain-controlled robotics, prosthetics, and enhancing normal motor function. By accurately modeling brain activity, we take a step toward flexible brain-computer interfaces that generalize beyond predefined movements. To support the research community, we provide open source tools for both synthetic data generation and neural decoding, fostering reproducibility and accelerating progress. The project is available at this https URL 

---
# Curiosity-Diffuser: Curiosity Guide Diffusion Models for Reliability 

**Authors**: Zihao Liu, Xing Liu, Yizhai Zhang, Zhengxiong Liu, Panfeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14833)  

**Abstract**: One of the bottlenecks in robotic intelligence is the instability of neural network models, which, unlike control models, lack a well-defined convergence domain and stability. This leads to risks when applying intelligence in the physical world. Specifically, imitation policy based on neural network may generate hallucinations, leading to inaccurate behaviors that impact the safety of real-world applications. To address this issue, this paper proposes the Curiosity-Diffuser, aimed at guiding the conditional diffusion model to generate trajectories with lower curiosity, thereby improving the reliability of policy. The core idea is to use a Random Network Distillation (RND) curiosity module to assess whether the model's behavior aligns with the training data, and then minimize curiosity by classifier guidance diffusion to reduce overgeneralization during inference. Additionally, we propose a computationally efficient metric for evaluating the reliability of the policy, measuring the similarity between the generated behaviors and the training dataset, to facilitate research about reliability learning. Finally, simulation verify the effectiveness and applicability of the proposed method to a variety of scenarios, showing that Curiosity-Diffuser significantly improves task performance and produces behaviors that are more similar to the training data. The code for this work is available at: this http URL 

---
# The CLEF-2025 CheckThat! Lab: Subjectivity, Fact-Checking, Claim Normalization, and Retrieval 

**Authors**: Firoj Alam, Julia Maria Struß, Tanmoy Chakraborty, Stefan Dietze, Salim Hafid, Katerina Korre, Arianna Muti, Preslav Nakov, Federico Ruggeri, Sebastian Schellhammer, Vinay Setty, Megha Sundriyal, Konstantin Todorov, Venktesh V  

**Link**: [PDF](https://arxiv.org/pdf/2503.14828)  

**Abstract**: The CheckThat! lab aims to advance the development of innovative technologies designed to identify and counteract online disinformation and manipulation efforts across various languages and platforms. The first five editions focused on key tasks in the information verification pipeline, including check-worthiness, evidence retrieval and pairing, and verification. Since the 2023 edition, the lab has expanded its scope to address auxiliary tasks that support research and decision-making in verification. In the 2025 edition, the lab revisits core verification tasks while also considering auxiliary challenges. Task 1 focuses on the identification of subjectivity (a follow-up from CheckThat! 2024), Task 2 addresses claim normalization, Task 3 targets fact-checking numerical claims, and Task 4 explores scientific web discourse processing. These tasks present challenging classification and retrieval problems at both the document and span levels, including multilingual settings. 

---
# MMDT: Decoding the Trustworthiness and Safety of Multimodal Foundation Models 

**Authors**: Chejian Xu, Jiawei Zhang, Zhaorun Chen, Chulin Xie, Mintong Kang, Yujin Potter, Zhun Wang, Zhuowen Yuan, Alexander Xiong, Zidi Xiong, Chenhui Zhang, Lingzhi Yuan, Yi Zeng, Peiyang Xu, Chengquan Guo, Andy Zhou, Jeffrey Ziwei Tan, Xuandong Zhao, Francesco Pinto, Zhen Xiang, Yu Gai, Zinan Lin, Dan Hendrycks, Bo Li, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.14827)  

**Abstract**: Multimodal foundation models (MMFMs) play a crucial role in various applications, including autonomous driving, healthcare, and virtual assistants. However, several studies have revealed vulnerabilities in these models, such as generating unsafe content by text-to-image models. Existing benchmarks on multimodal models either predominantly assess the helpfulness of these models, or only focus on limited perspectives such as fairness and privacy. In this paper, we present the first unified platform, MMDT (Multimodal DecodingTrust), designed to provide a comprehensive safety and trustworthiness evaluation for MMFMs. Our platform assesses models from multiple perspectives, including safety, hallucination, fairness/bias, privacy, adversarial robustness, and out-of-distribution (OOD) generalization. We have designed various evaluation scenarios and red teaming algorithms under different tasks for each perspective to generate challenging data, forming a high-quality benchmark. We evaluate a range of multimodal models using MMDT, and our findings reveal a series of vulnerabilities and areas for improvement across these perspectives. This work introduces the first comprehensive and unique safety and trustworthiness evaluation platform for MMFMs, paving the way for developing safer and more reliable MMFMs and systems. Our platform and benchmark are available at this https URL. 

---
# Learning with Expert Abstractions for Efficient Multi-Task Continuous Control 

**Authors**: Jeff Jewett, Sandhya Saisubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2503.14809)  

**Abstract**: Decision-making in complex, continuous multi-task environments is often hindered by the difficulty of obtaining accurate models for planning and the inefficiency of learning purely from trial and error. While precise environment dynamics may be hard to specify, human experts can often provide high-fidelity abstractions that capture the essential high-level structure of a task and user preferences in the target environment. Existing hierarchical approaches often target discrete settings and do not generalize across tasks. We propose a hierarchical reinforcement learning approach that addresses these limitations by dynamically planning over the expert-specified abstraction to generate subgoals to learn a goal-conditioned policy. To overcome the challenges of learning under sparse rewards, we shape the reward based on the optimal state value in the abstract model. This structured decision-making process enhances sample efficiency and facilitates zero-shot generalization. Our empirical evaluation on a suite of procedurally generated continuous control environments demonstrates that our approach outperforms existing hierarchical reinforcement learning methods in terms of sample efficiency, task completion rate, scalability to complex tasks, and generalization to novel scenarios. 

---
# Long Context Modeling with Ranked Memory-Augmented Retrieval 

**Authors**: Ghadir Alselwi, Hao Xue, Shoaib Jameel, Basem Suleiman, Flora D. Salim, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2503.14800)  

**Abstract**: Effective long-term memory management is crucial for language models handling extended contexts. We introduce a novel framework that dynamically ranks memory entries based on relevance. Unlike previous works, our model introduces a novel relevance scoring and a pointwise re-ranking model for key-value embeddings, inspired by learning-to-rank techniques in information retrieval. Enhanced Ranked Memory Augmented Retrieval ERMAR achieves state-of-the-art results on standard benchmarks. 

---
# RAT: Boosting Misclassification Detection Ability without Extra Data 

**Authors**: Ge Yan, Tsui-Wei Weng  

**Link**: [PDF](https://arxiv.org/pdf/2503.14783)  

**Abstract**: As deep neural networks(DNN) become increasingly prevalent, particularly in high-stakes areas such as autonomous driving and healthcare, the ability to detect incorrect predictions of models and intervene accordingly becomes crucial for safety. In this work, we investigate the detection of misclassified inputs for image classification models from the lens of adversarial perturbation: we propose to use robust radius (a.k.a. input-space margin) as a confidence metric and design two efficient estimation algorithms, RR-BS and RR-Fast, for misclassification detection. Furthermore, we design a training method called Radius Aware Training (RAT) to boost models' ability to identify mistakes. Extensive experiments show our method could achieve up to 29.3% reduction on AURC and 21.62% reduction in FPR@95TPR, compared with previous methods. 

---
# Involution and BSConv Multi-Depth Distillation Network for Lightweight Image Super-Resolution 

**Authors**: Akram Khatami-Rizi, Ahmad Mahmoudi-Aznaveh  

**Link**: [PDF](https://arxiv.org/pdf/2503.14779)  

**Abstract**: Single Image Super-Resolution (SISR) aims to reconstruct high-resolution (HR) images from low-resolution (LR) inputs. Deep learning, especially Convolutional Neural Networks (CNNs), has advanced SISR. However, increasing network depth increases parameters, and memory usage, and slows training, which is problematic for resource-limited devices. To address this, lightweight models are developed to balance accuracy and efficiency. We propose the Involution & BSConv Multi-Depth Distillation Network (IBMDN), combining Involution & BSConv Multi-Depth Distillation Block (IBMDB) and the Contrast and High-Frequency Attention Block (CHFAB). IBMDB integrates Involution and BSConv to balance computational efficiency and feature extraction. CHFAB enhances high-frequency details for better visual quality. IBMDB is compatible with other SISR architectures and reduces complexity, improving evaluation metrics like PSNR and SSIM. In transformer-based models, IBMDB reduces memory usage while improving feature extraction. In GANs, it enhances perceptual quality, balancing pixel-level accuracy with perceptual details. Our experiments show that the method achieves high accuracy with minimal computational cost. The code is available at GitHub. 

---
# Language Independent Named Entity Recognition via Orthogonal Transformation of Word Vectors 

**Authors**: Omar E. Rakha, Hazem M. Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2503.14755)  

**Abstract**: Word embeddings have been a key building block for NLP in which models relied heavily on word embeddings in many different tasks. In this paper, a model is proposed based on using Bidirectional LSTM/CRF with word embeddings to perform named entity recognition for any language. This is done by training a model on a source language (English) and transforming word embeddings from the target language into word embeddings of the source language by using an orthogonal linear transformation matrix. Evaluation of the model shows that by training a model on an English dataset the model was capable of detecting named entities in an Arabic dataset without neither training or fine tuning the model on an Arabic language dataset. 

---
# Bayesian Modeling of Zero-Shot Classifications for Urban Flood Detection 

**Authors**: Matt Franchi, Nikhil Garg, Wendy Ju, Emma Pierson  

**Link**: [PDF](https://arxiv.org/pdf/2503.14754)  

**Abstract**: Street scene datasets, collected from Street View or dashboard cameras, offer a promising means of detecting urban objects and incidents like street flooding. However, a major challenge in using these datasets is their lack of reliable labels: there are myriad types of incidents, many types occur rarely, and ground-truth measures of where incidents occur are lacking. Here, we propose BayFlood, a two-stage approach which circumvents this difficulty. First, we perform zero-shot classification of where incidents occur using a pretrained vision-language model (VLM). Second, we fit a spatial Bayesian model on the VLM classifications. The zero-shot approach avoids the need to annotate large training sets, and the Bayesian model provides frequent desiderata in urban settings - principled measures of uncertainty, smoothing across locations, and incorporation of external data like stormwater accumulation zones. We comprehensively validate this two-stage approach, showing that VLMs provide strong zero-shot signal for floods across multiple cities and time periods, the Bayesian model improves out-of-sample prediction relative to baseline methods, and our inferred flood risk correlates with known external predictors of risk. Having validated our approach, we show it can be used to improve urban flood detection: our analysis reveals 113,738 people who are at high risk of flooding overlooked by current methods, identifies demographic biases in existing methods, and suggests locations for new flood sensors. More broadly, our results showcase how Bayesian modeling of zero-shot LM annotations represents a promising paradigm because it avoids the need to collect large labeled datasets and leverages the power of foundation models while providing the expressiveness and uncertainty quantification of Bayesian models. 

---
# LipShiFT: A Certifiably Robust Shift-based Vision Transformer 

**Authors**: Rohan Menon, Nicola Franco, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2503.14751)  

**Abstract**: Deriving tight Lipschitz bounds for transformer-based architectures presents a significant challenge. The large input sizes and high-dimensional attention modules typically prove to be crucial bottlenecks during the training process and leads to sub-optimal results. Our research highlights practical constraints of these methods in vision tasks. We find that Lipschitz-based margin training acts as a strong regularizer while restricting weights in successive layers of the model. Focusing on a Lipschitz continuous variant of the ShiftViT model, we address significant training challenges for transformer-based architectures under norm-constrained input setting. We provide an upper bound estimate for the Lipschitz constants of this model using the $l_2$ norm on common image classification datasets. Ultimately, we demonstrate that our method scales to larger models and advances the state-of-the-art in certified robustness for transformer-based architectures. 

---
# GR00T N1: An Open Foundation Model for Generalist Humanoid Robots 

**Authors**: NVIDIA, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14734)  

**Abstract**: General-purpose robots need a versatile body and an intelligent mind. Recent advancements in humanoid robots have shown great promise as a hardware platform for building generalist autonomy in the human world. A robot foundation model, trained on massive and diverse data sources, is essential for enabling the robots to reason about novel situations, robustly handle real-world variability, and rapidly learn new tasks. To this end, we introduce GR00T N1, an open foundation model for humanoid robots. GR00T N1 is a Vision-Language-Action (VLA) model with a dual-system architecture. The vision-language module (System 2) interprets the environment through vision and language instructions. The subsequent diffusion transformer module (System 1) generates fluid motor actions in real time. Both modules are tightly coupled and jointly trained end-to-end. We train GR00T N1 with a heterogeneous mixture of real-robot trajectories, human videos, and synthetically generated datasets. We show that our generalist robot model GR00T N1 outperforms the state-of-the-art imitation learning baselines on standard simulation benchmarks across multiple robot embodiments. Furthermore, we deploy our model on the Fourier GR-1 humanoid robot for language-conditioned bimanual manipulation tasks, achieving strong performance with high data efficiency. 

---
# Construction Site Scaffolding Completeness Detection Based on Mask R-CNN and Hough Transform 

**Authors**: Pei-Hsin Lin, Jacob J. Lin, Shang-Hsien Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2503.14716)  

**Abstract**: Construction site scaffolding is essential for many building projects, and ensuring its safety is crucial to prevent accidents. The safety inspector must check the scaffolding's completeness and integrity, where most violations occur. The inspection process includes ensuring all the components are in the right place since workers often compromise safety for convenience and disassemble parts such as cross braces. This paper proposes a deep learning-based approach to detect the scaffolding and its cross braces using computer vision. A scaffold image dataset with annotated labels is used to train a convolutional neural network (CNN) model. With the proposed approach, we can automatically detect the completeness of cross braces from images taken at construction sites, without the need for manual inspection, saving a significant amount of time and labor costs. This non-invasive and efficient solution for detecting scaffolding completeness can help improve safety in construction sites. 

---
# DPImageBench: A Unified Benchmark for Differentially Private Image Synthesis 

**Authors**: Chen Gong, Kecen Li, Zinan Lin, Tianhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14681)  

**Abstract**: Differentially private (DP) image synthesis aims to generate artificial images that retain the properties of sensitive images while protecting the privacy of individual images within the dataset. Despite recent advancements, we find that inconsistent--and sometimes flawed--evaluation protocols have been applied across studies. This not only impedes the understanding of current methods but also hinders future advancements.
To address the issue, this paper introduces DPImageBench for DP image synthesis, with thoughtful design across several dimensions: (1) Methods. We study eleven prominent methods and systematically characterize each based on model architecture, pretraining strategy, and privacy mechanism. (2) Evaluation. We include nine datasets and seven fidelity and utility metrics to thoroughly assess them. Notably, we find that a common practice of selecting downstream classifiers based on the highest accuracy on the sensitive test set not only violates DP but also overestimates the utility scores. DPImageBench corrects for these mistakes. (3) Platform. Despite the methods and evaluation protocols, DPImageBench provides a standardized interface that accommodates current and future implementations within a unified framework. With DPImageBench, we have several noteworthy findings. For example, contrary to the common wisdom that pretraining on public image datasets is usually beneficial, we find that the distributional similarity between pretraining and sensitive images significantly impacts the performance of the synthetic images and does not always yield improvements. In addition, adding noise to low-dimensional features, such as the high-level characteristics of sensitive images, is less affected by the privacy budget compared to adding noise to high-dimensional features, like weight gradients. The former methods perform better than the latter under a low privacy budget. 

---
# ConQuer: A Framework for Concept-Based Quiz Generation 

**Authors**: Yicheng Fu, Zikui Wang, Liuxin Yang, Meiqing Huo, Zhongdongming Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.14662)  

**Abstract**: Quizzes play a crucial role in education by reinforcing students' understanding of key concepts and encouraging self-directed exploration. However, compiling high-quality quizzes can be challenging and require deep expertise and insight into specific subject matter. Although LLMs have greatly enhanced the efficiency of quiz generation, concerns remain regarding the quality of these AI-generated quizzes and their educational impact on students. To address these issues, we introduce ConQuer, a concept-based quiz generation framework that leverages external knowledge sources. We employ comprehensive evaluation dimensions to assess the quality of the generated quizzes, using LLMs as judges. Our experiment results demonstrate a 4.8% improvement in evaluation scores and a 77.52% win rate in pairwise comparisons against baseline quiz sets. Ablation studies further underscore the effectiveness of each component in our framework. Code available at this https URL. 

---
# Core-Periphery Principle Guided State Space Model for Functional Connectome Classification 

**Authors**: Minheng Chen, Xiaowei Yu, Jing Zhang, Tong Chen, Chao Cao, Yan Zhuang, Yanjun Lyu, Lu Zhang, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14655)  

**Abstract**: Understanding the organization of human brain networks has become a central focus in neuroscience, particularly in the study of functional connectivity, which plays a crucial role in diagnosing neurological disorders. Advances in functional magnetic resonance imaging and machine learning techniques have significantly improved brain network analysis. However, traditional machine learning approaches struggle to capture the complex relationships between brain regions, while deep learning methods, particularly Transformer-based models, face computational challenges due to their quadratic complexity in long-sequence modeling. To address these limitations, we propose a Core-Periphery State-Space Model (CP-SSM), an innovative framework for functional connectome classification. Specifically, we introduce Mamba, a selective state-space model with linear complexity, to effectively capture long-range dependencies in functional brain networks. Furthermore, inspired by the core-periphery (CP) organization, a fundamental characteristic of brain networks that enhances efficient information transmission, we design CP-MoE, a CP-guided Mixture-of-Experts that improves the representation learning of brain connectivity patterns. We evaluate CP-SSM on two benchmark fMRI datasets: ABIDE and ADNI. Experimental results demonstrate that CP-SSM surpasses Transformer-based models in classification performance while significantly reducing computational complexity. These findings highlight the effectiveness and efficiency of CP-SSM in modeling brain functional connectivity, offering a promising direction for neuroimaging-based neurological disease diagnosis. 

---
# RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving 

**Authors**: Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdanbakhsh, Vidushi Dadu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14649)  

**Abstract**: Retrieval-augmented generation (RAG), which combines large language models (LLMs) with retrievals from external knowledge databases, is emerging as a popular approach for reliable LLM serving. However, efficient RAG serving remains an open challenge due to the rapid emergence of many RAG variants and the substantial differences in workload characteristics across them. In this paper, we make three fundamental contributions to advancing RAG serving. First, we introduce RAGSchema, a structured abstraction that captures the wide range of RAG algorithms, serving as a foundation for performance optimization. Second, we analyze several representative RAG workloads with distinct RAGSchema, revealing significant performance variability across these workloads. Third, to address this variability and meet diverse performance requirements, we propose RAGO (Retrieval-Augmented Generation Optimizer), a system optimization framework for efficient RAG serving. Our evaluation shows that RAGO achieves up to a 2x increase in QPS per chip and a 55% reduction in time-to-first-token latency compared to RAG systems built on LLM-system extensions. 

---
# Dynamic Accumulated Attention Map for Interpreting Evolution of Decision-Making in Vision Transformer 

**Authors**: Yi Liao, Yongsheng Gao, Weichuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14640)  

**Abstract**: Various Vision Transformer (ViT) models have been widely used for image recognition tasks. However, existing visual explanation methods can not display the attention flow hidden inside the inner structure of ViT models, which explains how the final attention regions are formed inside a ViT for its decision-making. In this paper, a novel visual explanation approach, Dynamic Accumulated Attention Map (DAAM), is proposed to provide a tool that can visualize, for the first time, the attention flow from the top to the bottom through ViT networks. To this end, a novel decomposition module is proposed to construct and store the spatial feature information by unlocking the [class] token generated by the self-attention module of each ViT block. The module can also obtain the channel importance coefficients by decomposing the classification score for supervised ViT models. Because of the lack of classification score in self-supervised ViT models, we propose dimension-wise importance weights to compute the channel importance coefficients. Such spatial features are linearly combined with the corresponding channel importance coefficients, forming the attention map for each block. The dynamic attention flow is revealed by block-wisely accumulating each attention map. The contribution of this work focuses on visualizing the evolution dynamic of the decision-making attention for any intermediate block inside a ViT model by proposing a novel decomposition module and dimension-wise importance weights. The quantitative and qualitative analysis consistently validate the effectiveness and superior capacity of the proposed DAAM for not only interpreting ViT models with the fully-connected layers as the classifier but also self-supervised ViT models. The code is available at this https URL. 

---
# Reinforcement learning-based motion imitation for physiologically plausible musculoskeletal motor control 

**Authors**: Merkourios Simos, Alberto Silvio Chiappa, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2503.14637)  

**Abstract**: How do humans move? The quest to understand human motion has broad applications in numerous fields, ranging from computer animation and motion synthesis to neuroscience, human prosthetics and rehabilitation. Although advances in reinforcement learning (RL) have produced impressive results in capturing human motion using simplified humanoids, controlling physiologically accurate models of the body remains an open challenge. In this work, we present a model-free motion imitation framework (KINESIS) to advance the understanding of muscle-based motor control. Using a musculoskeletal model of the lower body with 80 muscle actuators and 20 DoF, we demonstrate that KINESIS achieves strong imitation performance on 1.9 hours of motion capture data, is controllable by natural language through pre-trained text-to-motion generative models, and can be fine-tuned to carry out high-level tasks such as target goal reaching. Importantly, KINESIS generates muscle activity patterns that correlate well with human EMG activity. The physiological plausibility makes KINESIS a promising model for tackling challenging problems in human motor control theory, which we highlight by investigating Bernstein's redundancy problem in the context of locomotion. Code, videos and benchmarks will be available at this https URL. 

---
# Assessing Large Language Models for Automated Feedback Generation in Learning Programming Problem Solving 

**Authors**: Priscylla Silva, Evandro Costa  

**Link**: [PDF](https://arxiv.org/pdf/2503.14630)  

**Abstract**: Providing effective feedback is important for student learning in programming problem-solving. In this sense, Large Language Models (LLMs) have emerged as potential tools to automate feedback generation. However, their reliability and ability to identify reasoning errors in student code remain not well understood. This study evaluates the performance of four LLMs (GPT-4o, GPT-4o mini, GPT-4-Turbo, and Gemini-1.5-pro) on a benchmark dataset of 45 student solutions. We assessed the models' capacity to provide accurate and insightful feedback, particularly in identifying reasoning mistakes. Our analysis reveals that 63\% of feedback hints were accurate and complete, while 37\% contained mistakes, including incorrect line identification, flawed explanations, or hallucinated issues. These findings highlight the potential and limitations of LLMs in programming education and underscore the need for improvements to enhance reliability and minimize risks in educational applications. 

---
# Reducing False Ventricular Tachycardia Alarms in ICU Settings: A Machine Learning Approach 

**Authors**: Grace Funmilayo Farayola, Akinyemi Sadeeq Akintola, Oluwole Fagbohun, Chukwuka Michael Oforgu, Bisola Faith Kayode, Christian Chimezie, Temitope Kadri, Abiola Oludotun, Nelson Ogbeide, Mgbame Michael, Adeseye Ifaturoti, Toyese Oloyede  

**Link**: [PDF](https://arxiv.org/pdf/2503.14621)  

**Abstract**: False arrhythmia alarms in intensive care units (ICUs) are a significant challenge, contributing to alarm fatigue and potentially compromising patient safety. Ventricular tachycardia (VT) alarms are particularly difficult to detect accurately due to their complex nature. This paper presents a machine learning approach to reduce false VT alarms using the VTaC dataset, a benchmark dataset of annotated VT alarms from ICU monitors. We extract time-domain and frequency-domain features from waveform data, preprocess the data, and train deep learning models to classify true and false VT alarms. Our results demonstrate high performance, with ROC-AUC scores exceeding 0.96 across various training configurations. This work highlights the potential of machine learning to improve the accuracy of VT alarm detection in clinical settings. 

---
# Image Captioning Evaluation in the Age of Multimodal LLMs: Challenges and Future Perspectives 

**Authors**: Sara Sarto, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2503.14604)  

**Abstract**: The evaluation of machine-generated image captions is a complex and evolving challenge. With the advent of Multimodal Large Language Models (MLLMs), image captioning has become a core task, increasing the need for robust and reliable evaluation metrics. This survey provides a comprehensive overview of advancements in image captioning evaluation, analyzing the evolution, strengths, and limitations of existing metrics. We assess these metrics across multiple dimensions, including correlation with human judgment, ranking accuracy, and sensitivity to hallucinations. Additionally, we explore the challenges posed by the longer and more detailed captions generated by MLLMs and examine the adaptability of current metrics to these stylistic variations. Our analysis highlights some limitations of standard evaluation approaches and suggests promising directions for future research in image captioning assessment. 

---
# PHGNN: A Novel Prompted Hypergraph Neural Network to Diagnose Alzheimer's Disease 

**Authors**: Chenyu Liu, Luca Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2503.14577)  

**Abstract**: The accurate diagnosis of Alzheimer's disease (AD) and prognosis of mild cognitive impairment (MCI) conversion are crucial for early intervention. However, existing multimodal methods face several challenges, from the heterogeneity of input data, to underexplored modality interactions, missing data due to patient dropouts, and limited data caused by the time-consuming and costly data collection process. In this paper, we propose a novel Prompted Hypergraph Neural Network (PHGNN) framework that addresses these limitations by integrating hypergraph based learning with prompt learning. Hypergraphs capture higher-order relationships between different modalities, while our prompt learning approach for hypergraphs, adapted from NLP, enables efficient training with limited data. Our model is validated through extensive experiments on the ADNI dataset, outperforming SOTA methods in both AD diagnosis and the prediction of MCI conversion. 

---
# SocialJax: An Evaluation Suite for Multi-agent Reinforcement Learning in Sequential Social Dilemmas 

**Authors**: Zihao Guo, Richard Willis, Shuqing Shi, Tristan Tomilin, Joel Z. Leibo, Yali Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.14576)  

**Abstract**: Social dilemmas pose a significant challenge in the field of multi-agent reinforcement learning (MARL). Melting Pot is an extensive framework designed to evaluate social dilemma environments, providing an evaluation protocol that measures generalization to new social partners across various test scenarios. However, running reinforcement learning algorithms in the official Melting Pot environments demands substantial computational resources. In this paper, we introduce SocialJax, a suite of sequential social dilemma environments implemented in JAX. JAX is a high-performance numerical computing library for Python that enables significant improvements in the operational efficiency of SocialJax on GPUs and TPUs. Our experiments demonstrate that the training pipeline of SocialJax achieves a 50\texttimes{} speedup in real-time performance compared to Melting Pot's RLlib baselines. Additionally, we validate the effectiveness of baseline algorithms within the SocialJax environments. Finally, we use Schelling diagrams to verify the social dilemma properties of these environments, ensuring they accurately capture the dynamics of social dilemmas. 

---
# Robust Weight Imprinting: Insights from Neural Collapse and Proxy-Based Aggregation 

**Authors**: Justus Westerhoff, Golzar Atefi, Mario Koddenbrock, Alexei Figueroa, Alexander Löser, Erik Rodner, Felix A. Gers  

**Link**: [PDF](https://arxiv.org/pdf/2503.14572)  

**Abstract**: The capacity of a foundation model allows for adaptation to new downstream tasks. Weight imprinting is a universal and efficient method to fulfill this purpose. It has been reinvented several times, but it has not been systematically studied. In this paper, we propose a framework for imprinting, identifying three main components: generation, normalization, and aggregation. This allows us to conduct an in-depth analysis of imprinting and a comparison of the existing work. We reveal the benefits of representing novel data with multiple proxies in the generation step and show the importance of proper normalization. We determine those proxies through clustering and propose a novel variant of imprinting that outperforms previous work. We motivate this by the neural collapse phenomenon -- an important connection that we can draw for the first time. Our results show an increase of up to 4% in challenging scenarios with complex data distributions for new classes. 

---
# Potential Score Matching: Debiasing Molecular Structure Sampling with Potential Energy Guidance 

**Authors**: Liya Guo, Zun Wang, Chang Liu, Junzhe Li, Pipi Hu, Yi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14569)  

**Abstract**: The ensemble average of physical properties of molecules is closely related to the distribution of molecular conformations, and sampling such distributions is a fundamental challenge in physics and chemistry. Traditional methods like molecular dynamics (MD) simulations and Markov chain Monte Carlo (MCMC) sampling are commonly used but can be time-consuming and costly. Recently, diffusion models have emerged as efficient alternatives by learning the distribution of training data. Obtaining an unbiased target distribution is still an expensive task, primarily because it requires satisfying ergodicity. To tackle these challenges, we propose Potential Score Matching (PSM), an approach that utilizes the potential energy gradient to guide generative models. PSM does not require exact energy functions and can debias sample distributions even when trained on limited and biased data. Our method outperforms existing state-of-the-art (SOTA) models on the Lennard-Jones (LJ) potential, a commonly used toy model. Furthermore, we extend the evaluation of PSM to high-dimensional problems using the MD17 and MD22 datasets. The results demonstrate that molecular distributions generated by PSM more closely approximate the Boltzmann distribution compared to traditional diffusion models. 

---
# Teaching Artificial Intelligence to Perform Rapid, Resolution-Invariant Grain Growth Modeling via Fourier Neural Operator 

**Authors**: Iman Peivaste, Ahmed Makradi, Salim Belouettar  

**Link**: [PDF](https://arxiv.org/pdf/2503.14568)  

**Abstract**: Microstructural evolution, particularly grain growth, plays a critical role in shaping the physical, optical, and electronic properties of materials. Traditional phase-field modeling accurately simulates these phenomena but is computationally intensive, especially for large systems and fine spatial resolutions. While machine learning approaches have been employed to accelerate simulations, they often struggle with resolution dependence and generalization across different grain scales. This study introduces a novel approach utilizing Fourier Neural Operator (FNO) to achieve resolution-invariant modeling of microstructure evolution in multi-grain systems. FNO operates in the Fourier space and can inherently handle varying resolutions by learning mappings between function spaces. By integrating FNO with the phase field method, we developed a surrogate model that significantly reduces computational costs while maintaining high accuracy across different spatial scales. We generated a comprehensive dataset from phase-field simulations using the Fan Chen model, capturing grain evolution over time. Data preparation involved creating input-output pairs with a time shift, allowing the model to predict future microstructures based on current and past states. The FNO-based neural network was trained using sequences of microstructures and demonstrated remarkable accuracy in predicting long-term evolution, even for unseen configurations and higher-resolution grids not encountered during training. 

---
# SpecReX: Explainable AI for Raman Spectroscopy 

**Authors**: Nathan Blake, David A. Kelly, Akchunya Chanchal, Sarah Kapllani-Mucaj, Geraint Thomas, Hana Chockler  

**Link**: [PDF](https://arxiv.org/pdf/2503.14567)  

**Abstract**: Raman spectroscopy is becoming more common for medical diagnostics with deep learning models being increasingly used to leverage its full potential. However, the opaque nature of such models and the sensitivity of medical diagnosis together with regulatory requirements necessitate the need for explainable AI tools. We introduce SpecReX, specifically adapted to explaining Raman spectra. SpecReX uses the theory of actual causality to rank causal responsibility in a spectrum, quantified by iteratively refining mutated versions of the spectrum and testing if it retains the original classification. The explanations provided by SpecReX take the form of a responsibility map, highlighting spectral regions most responsible for the model to make a correct classification. To assess the validity of SpecReX, we create increasingly complex simulated spectra, in which a "ground truth" signal is seeded, to train a classifier. We then obtain SpecReX explanations and compare the results with another explainability tool. By using simulated spectra we establish that SpecReX localizes to the known differences between classes, under a number of conditions. This provides a foundation on which we can find the spectral features which differentiate disease classes. This is an important first step in proving the validity of SpecReX. 

---
# Effortless Active Labeling for Long-Term Test-Time Adaptation 

**Authors**: Guowei Wang, Changxing Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.14564)  

**Abstract**: Long-term test-time adaptation (TTA) is a challenging task due to error accumulation. Recent approaches tackle this issue by actively labeling a small proportion of samples in each batch, yet the annotation burden quickly grows as the batch number increases. In this paper, we investigate how to achieve effortless active labeling so that a maximum of one sample is selected for annotation in each batch. First, we annotate the most valuable sample in each batch based on the single-step optimization perspective in the TTA context. In this scenario, the samples that border between the source- and target-domain data distributions are considered the most feasible for the model to learn in one iteration. Then, we introduce an efficient strategy to identify these samples using feature perturbation. Second, we discover that the gradient magnitudes produced by the annotated and unannotated samples have significant variations. Therefore, we propose balancing their impact on model optimization using two dynamic weights. Extensive experiments on the popular ImageNet-C, -R, -K, -A and PACS databases demonstrate that our approach consistently outperforms state-of-the-art methods with significantly lower annotation costs. 

---
# Workflow for Safe-AI 

**Authors**: Suzana Veljanovska, Hans Dermot Doran  

**Link**: [PDF](https://arxiv.org/pdf/2503.14563)  

**Abstract**: The development and deployment of safe and dependable AI models is crucial in applications where functional safety is a key concern. Given the rapid advancement in AI research and the relative novelty of the safe-AI domain, there is an increasing need for a workflow that balances stability with adaptability. This work proposes a transparent, complete, yet flexible and lightweight workflow that highlights both reliability and qualifiability. The core idea is that the workflow must be qualifiable, which demands the use of qualified tools. Tool qualification is a resource-intensive process, both in terms of time and cost. We therefore place value on a lightweight workflow featuring a minimal number of tools with limited features. The workflow is built upon an extended ONNX model description allowing for validation of AI algorithms from their generation to runtime deployment. This validation is essential to ensure that models are validated before being reliably deployed across different runtimes, particularly in mixed-criticality systems. Keywords-AI workflows, safe-AI, dependable-AI, functional safety, v-model development 

---
# Analysis of human visual field information using machine learning methods and assessment of their accuracy 

**Authors**: A.I. Medvedeva, V.V. Bakutkin  

**Link**: [PDF](https://arxiv.org/pdf/2503.14562)  

**Abstract**: Subject of research: is the study of methods for analyzing perimetric images for the diagnosis and control of glaucoma diseases. Objects of research: is a dataset collected on the ophthalmological perimeter with the results of various patient pathologies, since the ophthalmological community is acutely aware of the issue of disease control and import substitution. [5]. Purpose of research: is to consider various machine learning methods that can classify glaucoma. This is possible thanks to the classifier built after labeling the dataset. It is able to determine from the image whether the visual fields depicted on it are the results of the impact of glaucoma on the eyes or other visual diseases. Earlier in the work [3], a dataset was described that was collected on the Tomey perimeter. The average age of the examined patients ranged from 30 to 85 years. Methods of research: machine learning methods for classifying image results (stochastic gradient descent, logistic regression, random forest, naive Bayes). Main results of research: the result of the study is computer modeling that can determine from the image whether the result is glaucoma or another disease (binary classification). 

---
# Squeeze Out Tokens from Sample for Finer-Grained Data Governance 

**Authors**: Weixiong Lin, Chen Ju, Haicheng Wang, Shengchao Hu, Shuai Xiao, Mengting Chen, Yuheng Jiao, Mingshuai Yao, Jinsong Lan, Qingwen Liu, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.14559)  

**Abstract**: Widely observed data scaling laws, in which error falls off as a power of the training size, demonstrate the diminishing returns of unselective data expansion. Hence, data governance is proposed to downsize datasets through pruning non-informative samples. Yet, isolating the impact of a specific sample on overall model performance is challenging, due to the vast computation required for tryout all sample combinations. Current data governors circumvent this complexity by estimating sample contributions through heuristic-derived scalar scores, thereby discarding low-value ones. Despite thorough sample sieving, retained samples contain substantial undesired tokens intrinsically, underscoring the potential for further compression and purification. In this work, we upgrade data governance from a 'sieving' approach to a 'juicing' one. Instead of scanning for least-flawed samples, our dual-branch DataJuicer applies finer-grained intra-sample governance. It squeezes out informative tokens and boosts image-text alignments. Specifically, the vision branch retains salient image patches and extracts relevant object classes, while the text branch incorporates these classes to enhance captions. Consequently, DataJuicer yields more refined datasets through finer-grained governance. Extensive experiments across datasets demonstrate that DataJuicer significantly outperforms existing DataSieve in image-text retrieval, classification, and dense visual reasoning. 

---
# Designing and Deploying AI Models for Sustainable Logistics Optimization: A Case Study on Eco-Efficient Supply Chains in the USA 

**Authors**: Reza E Rabbi Shawon, MD Rokibul Hasan, Md Anisur Rahman, Mohamed Ghandri, Iman Ahmed Lamari, Mohammed Kawsar, Rubi Akter  

**Link**: [PDF](https://arxiv.org/pdf/2503.14556)  

**Abstract**: The rapid evolution of Artificial Intelligence (AI) and Machine Learning (ML) has significantly transformed logistics and supply chain management, particularly in the pursuit of sustainability and eco-efficiency. This study explores AI-based methodologies for optimizing logistics operations in the USA, focusing on reducing environmental impact, improving fuel efficiency, and minimizing costs. Key AI applications include predictive analytics for demand forecasting, route optimization through machine learning, and AI-powered fuel efficiency strategies. Various models, such as Linear Regression, XGBoost, Support Vector Machine, and Neural Networks, are applied to real-world logistics datasets to reduce carbon emissions based on logistics operations, optimize travel routes to minimize distance and travel time, and predict future deliveries to plan optimal routes. Other models such as K-Means and DBSCAN are also used to optimize travel routes to minimize distance and travel time for logistics operations. This study utilizes datasets from logistics companies' databases. The study also assesses model performance using metrics such as mean absolute error (MAE), mean squared error (MSE), and R2 score. This study also explores how these models can be deployed to various platforms for real-time logistics and supply chain use. The models are also examined through a thorough case study, highlighting best practices and regulatory frameworks that promote sustainability. The findings demonstrate AI's potential to enhance logistics efficiency, reduce carbon footprints, and contribute to a more resilient and adaptive supply chain ecosystem. 

---
# A Generalist Hanabi Agent 

**Authors**: Arjun V Sudhakar, Hadi Nekoei, Mathieu Reymond, Miao Liu, Janarthanan Rajendran, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2503.14555)  

**Abstract**: Traditional multi-agent reinforcement learning (MARL) systems can develop cooperative strategies through repeated interactions. However, these systems are unable to perform well on any other setting than the one they have been trained on, and struggle to successfully cooperate with unfamiliar collaborators. This is particularly visible in the Hanabi benchmark, a popular 2-to-5 player cooperative card-game which requires complex reasoning and precise assistance to other agents. Current MARL agents for Hanabi can only learn one specific game-setting (e.g., 2-player games), and play with the same algorithmic agents. This is in stark contrast to humans, who can quickly adjust their strategies to work with unfamiliar partners or situations. In this paper, we introduce Recurrent Replay Relevance Distributed DQN (R3D2), a generalist agent for Hanabi, designed to overcome these limitations. We reformulate the task using text, as language has been shown to improve transfer. We then propose a distributed MARL algorithm that copes with the resulting dynamic observation- and action-space. In doing so, our agent is the first that can play all game settings concurrently, and extend strategies learned from one setting to other ones. As a consequence, our agent also demonstrates the ability to collaborate with different algorithmic agents -- agents that are themselves unable to do so. The implementation code is available at: $\href{this https URL}{R3D2-A-Generalist-Hanabi-Agent}$ 

---
# Synchronous vs Asynchronous Reinforcement Learning in a Real World Robot 

**Authors**: Ali Parsaee, Fahim Shahriar, Chuxin He, Ruiqing Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.14554)  

**Abstract**: In recent times, reinforcement learning (RL) with physical robots has attracted the attention of a wide range of researchers. However, state-of-the-art RL algorithms do not consider that physical environments do not wait for the RL agent to make decisions or updates. RL agents learn by periodically conducting computationally expensive gradient updates. When decision-making and gradient update tasks are carried out sequentially by the RL agent in a physical robot, it significantly increases the agent's response time. In a rapidly changing environment, this increased response time may be detrimental to the performance of the learning agent. Asynchronous RL methods, which separate the computation of decision-making and gradient updates, are a potential solution to this problem. However, only a few comparisons between asynchronous and synchronous RL have been made with physical robots. For this reason, the exact performance benefits of using asynchronous RL methods over synchronous RL methods are still unclear. In this study, we provide a performance comparison between asynchronous and synchronous RL using a physical robotic arm called Franka Emika Panda. Our experiments show that the agents learn faster and attain significantly more returns using asynchronous RL. Our experiments also demonstrate that the learning agent with a faster response time performs better than the agent with a slower response time, even if the agent with a slower response time performs a higher number of gradient updates. 

---
# Fire and Smoke Datasets in 20 Years: An In-depth Review 

**Authors**: Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Fatemeh Afghah, Connor Peter McGrath, Danish Bhatkar, Mithilesh Anil Biradar, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2503.14552)  

**Abstract**: Fire and smoke phenomena pose a significant threat to the natural environment, ecosystems, and global economy, as well as human lives and wildlife. In this particular circumstance, there is a demand for more sophisticated and advanced technologies to implement an effective strategy for early detection, real-time monitoring, and minimizing the overall impacts of fires on ecological balance and public safety. Recently, the rapid advancement of Artificial Intelligence (AI) and Computer Vision (CV) frameworks has substantially revolutionized the momentum for developing efficient fire management systems. However, these systems extensively rely on the availability of adequate and high-quality fire and smoke data to create proficient Machine Learning (ML) methods for various tasks, such as detection and monitoring. Although fire and smoke datasets play a critical role in training, evaluating, and testing advanced Deep Learning (DL) models, a comprehensive review of the existing datasets is still unexplored. For this purpose, we provide an in-depth review to systematically analyze and evaluate fire and smoke datasets collected over the past 20 years. We investigate the characteristics of each dataset, including type, size, format, collection methods, and geographical diversities. We also review and highlight the unique features of each dataset, such as imaging modalities (RGB, thermal, infrared) and their applicability for different fire management tasks (classification, segmentation, detection). Furthermore, we summarize the strengths and weaknesses of each dataset and discuss their potential for advancing research and technology in fire management. Ultimately, we conduct extensive experimental analyses across different datasets using several state-of-the-art algorithms, such as ResNet-50, DeepLab-V3, and YoloV8. 

---
# Novel AI-Based Quantification of Breast Arterial Calcification to Predict Cardiovascular Risk 

**Authors**: Theodorus Dapamede, Aisha Urooj, Vedant Joshi, Gabrielle Gershon, Frank Li, Mohammadreza Chavoshi, Beatrice Brown-Mulry, Rohan Satya Isaac, Aawez Mansuri, Chad Robichaux, Chadi Ayoub, Reza Arsanjani, Laurence Sperling, Judy Gichoya, Marly van Assen, Charles W. ONeill, Imon Banerjee, Hari Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2503.14550)  

**Abstract**: Women are underdiagnosed and undertreated for cardiovascular disease. Automatic quantification of breast arterial calcification on screening mammography can identify women at risk for cardiovascular disease and enable earlier treatment and management of disease. In this retrospective study of 116,135 women from two healthcare systems, a transformer-based neural network quantified BAC severity (no BAC, mild, moderate, and severe) on screening mammograms. Outcomes included major adverse cardiovascular events (MACE) and all-cause mortality. BAC severity was independently associated with MACE after adjusting for cardiovascular risk factors, with increasing hazard ratios from mild (HR 1.18-1.22), moderate (HR 1.38-1.47), to severe BAC (HR 2.03-2.22) across datasets (all p<0.001). This association remained significant across all age groups, with even mild BAC indicating increased risk in women under 50. BAC remained an independent predictor when analyzed alongside ASCVD risk scores, showing significant associations with myocardial infarction, stroke, heart failure, and mortality (all p<0.005). Automated BAC quantification enables opportunistic cardiovascular risk assessment during routine mammography without additional radiation or cost. This approach provides value beyond traditional risk factors, particularly in younger women, offering potential for early CVD risk stratification in the millions of women undergoing annual mammography. 

---
# Sampling Decisions 

**Authors**: Michael Chertkov, Sungsoo Ahn, Hamidreza Behjoo  

**Link**: [PDF](https://arxiv.org/pdf/2503.14549)  

**Abstract**: In this manuscript we introduce a novel Decision Flow (DF) framework for sampling from a target distribution while incorporating additional guidance from a prior sampler. DF can be viewed as an AI driven algorithmic reincarnation of the Markov Decision Process (MDP) approach in Stochastic Optimal Control. It extends the continuous space, continuous time path Integral Diffusion sampling technique to discrete time and space, while also generalizing the Generative Flow Network framework. In its most basic form, an explicit, Neural Network (NN) free formulation, DF leverages the linear solvability of the the underlying MDP to adjust the transition probabilities of the prior sampler. The resulting Markov Process is expressed as a convolution of the reverse time Green's function of the prior sampling with the target distribution. We illustrate the DF framework through an example of sampling from the Ising model, discuss potential NN based extensions, and outline how DF can enhance guided sampling across various applications. 

---
# The Impact of Artificial Intelligence on Emergency Medicine: A Review of Recent Advances 

**Authors**: Gustavo Correia, Victor Alves, Paulo Novais  

**Link**: [PDF](https://arxiv.org/pdf/2503.14546)  

**Abstract**: Artificial Intelligence (AI) is revolutionizing emergency medicine by enhancing diagnostic processes and improving patient outcomes. This article provides a review of the current applications of AI in emergency imaging studies, focusing on the last five years of advancements. AI technologies, particularly machine learning and deep learning, are pivotal in interpreting complex imaging data, offering rapid, accurate diagnoses and potentially surpassing traditional diagnostic methods. Studies highlighted within the article demonstrate AI's capabilities in accurately detecting conditions such as fractures, pneumothorax, and pulmonary diseases from various imaging modalities including X-rays, CT scans, and MRIs. Furthermore, AI's ability to predict clinical outcomes like mechanical ventilation needs illustrates its potential in crisis resource optimization. Despite these advancements, the integration of AI into clinical practice presents challenges such as data privacy, algorithmic bias, and the need for extensive validation across diverse settings. This review underscores the transformative potential of AI in emergency settings, advocating for a future where AI and clinical expertise synergize to elevate patient care standards. 

---
# Inteligencia Artificial para la conservación y uso sostenible de la biodiversidad, una visión desde Colombia (Artificial Intelligence for conservation and sustainable use of biodiversity, a view from Colombia) 

**Authors**: Juan Sebastián Cañas, Camila Parra-Guevara, Manuela Montoya-Castrillón, Julieta M Ramírez-Mejía, Gabriel-Alejandro Perilla, Esteban Marentes, Nerieth Leuro, Jose Vladimir Sandoval-Sierra, Sindy Martinez-Callejas, Angélica Díaz, Mario Murcia, Elkin A. Noguera-Urbano, Jose Manuel Ochoa-Quintero, Susana Rodríguez Buriticá, Juan Sebastián Ulloa  

**Link**: [PDF](https://arxiv.org/pdf/2503.14543)  

**Abstract**: The rise of artificial intelligence (AI) and the aggravating biodiversity crisis have resulted in a research area where AI-based computational methods are being developed to act as allies in conservation, and the sustainable use and management of natural resources. While important general guidelines have been established globally regarding the opportunities and challenges that this interdisciplinary research offers, it is essential to generate local reflections from the specific contexts and realities of each region. Hence, this document aims to analyze the scope of this research area from a perspective focused on Colombia and the Neotropics. In this paper, we summarize the main experiences and debates that took place at the Humboldt Institute between 2023 and 2024 in Colombia. To illustrate the variety of promising opportunities, we present current uses such as automatic species identification from images and recordings, species modeling, and in silico bioprospecting, among others. From the experiences described above, we highlight limitations, challenges, and opportunities for in order to successfully implementate AI in conservation efforts and sustainable management of biological resources in the Neotropics. The result aims to be a guide for researchers, decision makers, and biodiversity managers, facilitating the understanding of how artificial intelligence can be effectively integrated into conservation and sustainable use strategies. Furthermore, it also seeks to open a space for dialogue on the development of policies that promote the responsible and ethical adoption of AI in local contexts, ensuring that its benefits are harnessed without compromising biodiversity or the cultural and ecosystemic values inherent in Colombia and the Neotropics. 

---
# AI-Driven Rapid Identification of Bacterial and Fungal Pathogens in Blood Smears of Septic Patients 

**Authors**: Agnieszka Sroka-Oleksiak, Adam Pardyl, Dawid Rymarczyk, Aldona Olechowska-Jarząb, Katarzyna Biegun-Drożdż, Dorota Ochońska, Michał Wronka, Adriana Borowa, Tomasz Gosiewski, Miłosz Adamczyk, Henryk Telega, Bartosz Zieliński, Monika Brzychczy-Włoch  

**Link**: [PDF](https://arxiv.org/pdf/2503.14542)  

**Abstract**: Sepsis is a life-threatening condition which requires rapid diagnosis and treatment. Traditional microbiological methods are time-consuming and expensive. In response to these challenges, deep learning algorithms were developed to identify 14 bacteria species and 3 yeast-like fungi from microscopic images of Gram-stained smears of positive blood samples from sepsis patients.
A total of 16,637 Gram-stained microscopic images were used in the study. The analysis used the Cellpose 3 model for segmentation and Attention-based Deep Multiple Instance Learning for classification. Our model achieved an accuracy of 77.15% for bacteria and 71.39% for fungi, with ROC AUC of 0.97 and 0.88, respectively. The highest values, reaching up to 96.2%, were obtained for Cutibacterium acnes, Enterococcus faecium, Stenotrophomonas maltophilia and Nakaseomyces glabratus. Classification difficulties were observed in closely related species, such as Staphylococcus hominis and Staphylococcus haemolyticus, due to morphological similarity, and within Candida albicans due to high morphotic diversity.
The study confirms the potential of our model for microbial classification, but it also indicates the need for further optimisation and expansion of the training data set. In the future, this technology could support microbial diagnosis, reducing diagnostic time and improving the effectiveness of sepsis treatment due to its simplicity and accessibility. Part of the results presented in this publication was covered by a patent application at the European Patent Office EP24461637.1 "A computer implemented method for identifying a microorganism in a blood and a data processing system therefor". 

---
# Vision-Language Models for Acute Tuberculosis Diagnosis: A Multimodal Approach Combining Imaging and Clinical Data 

**Authors**: Ananya Ganapthy, Praveen Shastry, Naveen Kumarasami, Anandakumar D, Keerthana R, Mounigasri M, Varshinipriya M, Kishore Prasath Venkatesh, Bargava Subramanian, Kalyan Sivasailam  

**Link**: [PDF](https://arxiv.org/pdf/2503.14538)  

**Abstract**: Background: This study introduces a Vision-Language Model (VLM) leveraging SIGLIP and Gemma-3b architectures for automated acute tuberculosis (TB) screening. By integrating chest X-ray images and clinical notes, the model aims to enhance diagnostic accuracy and efficiency, particularly in resource-limited settings.
Methods: The VLM combines visual data from chest X-rays with clinical context to generate detailed, context-aware diagnostic reports. The architecture employs SIGLIP for visual encoding and Gemma-3b for decoding, ensuring effective representation of acute TB-specific pathologies and clinical insights.
Results: Key acute TB pathologies, including consolidation, cavities, and nodules, were detected with high precision (97percent) and recall (96percent). The model demonstrated strong spatial localization capabilities and robustness in distinguishing TB-positive cases, making it a reliable tool for acute TB diagnosis.
Conclusion: The multimodal capability of the VLM reduces reliance on radiologists, providing a scalable solution for acute TB screening. Future work will focus on improving the detection of subtle pathologies and addressing dataset biases to enhance its generalizability and application in diverse global healthcare settings. 

---
# Advancing Chronic Tuberculosis Diagnostics Using Vision-Language Models: A Multi modal Framework for Precision Analysis 

**Authors**: Praveen Shastry, Sowmya Chowdary Muthulur, Naveen Kumarasami, Anandakumar D, Mounigasri M, Keerthana R, Kishore Prasath Venkatesh, Bargava Subramanian, Kalyan Sivasailam, Revathi Ezhumalai, Abitha Marimuthu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14536)  

**Abstract**: Background This study proposes a Vision-Language Model (VLM) leveraging the SIGLIP encoder and Gemma-3b transformer decoder to enhance automated chronic tuberculosis (TB) screening. By integrating chest X-ray images with clinical data, the model addresses the challenges of manual interpretation, improving diagnostic consistency and accessibility, particularly in resource-constrained settings.
Methods The VLM architecture combines a Vision Transformer (ViT) for visual encoding and a transformer-based text encoder to process clinical context, such as patient histories and treatment records. Cross-modal attention mechanisms align radiographic features with textual information, while the Gemma-3b decoder generates comprehensive diagnostic reports. The model was pre-trained on 5 million paired medical images and texts and fine-tuned using 100,000 chronic TB-specific chest X-rays.
Results The model demonstrated high precision (94 percent) and recall (94 percent) for detecting key chronic TB pathologies, including fibrosis, calcified granulomas, and bronchiectasis. Area Under the Curve (AUC) scores exceeded 0.93, and Intersection over Union (IoU) values were above 0.91, validating its effectiveness in detecting and localizing TB-related abnormalities.
Conclusion The VLM offers a robust and scalable solution for automated chronic TB diagnosis, integrating radiographic and clinical data to deliver actionable and context-aware insights. Future work will address subtle pathologies and dataset biases to enhance the model's generalizability, ensuring equitable performance across diverse populations and healthcare settings. 

---
# Interpretable Unsupervised Joint Denoising and Enhancement for Real-World low-light Scenarios 

**Authors**: Huaqiu Li, Xiaowan Hu, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14535)  

**Abstract**: Real-world low-light images often suffer from complex degradations such as local overexposure, low brightness, noise, and uneven illumination. Supervised methods tend to overfit to specific scenarios, while unsupervised methods, though better at generalization, struggle to model these degradations due to the lack of reference images. To address this issue, we propose an interpretable, zero-reference joint denoising and low-light enhancement framework tailored for real-world scenarios. Our method derives a training strategy based on paired sub-images with varying illumination and noise levels, grounded in physical imaging principles and retinex theory. Additionally, we leverage the Discrete Cosine Transform (DCT) to perform frequency domain decomposition in the sRGB space, and introduce an implicit-guided hybrid representation strategy that effectively separates intricate compounded degradations. In the backbone network design, we develop retinal decomposition network guided by implicit degradation representation mechanisms. Extensive experiments demonstrate the superiority of our method. Code will be available at this https URL. 

---
# SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders 

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2503.14530)  

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs. 

---
# Accessibility Considerations in the Development of an AI Action Plan 

**Authors**: Jennifer Mankoff, Janice Light, James Coughlan, Christian Vogler, Abraham Glasser, Gregg Vanderheiden, Laura Rice  

**Link**: [PDF](https://arxiv.org/pdf/2503.14522)  

**Abstract**: We argue that there is a need for Accessibility to be represented in several important domains:
- Capitalize on the new capabilities AI provides - Support for open source development of AI, which can allow disabled and disability focused professionals to contribute, including
- Development of Accessibility Apps which help realise the promise of AI in accessibility domains
- Open Source Model Development and Validation to ensure that accessibility concerns are addressed in these algorithms
- Data Augmentation to include accessibility in data sets used to train models
- Accessible Interfaces that allow disabled people to use any AI app, and to validate its outputs
- Dedicated Functionality and Libraries that can make it easy to integrate AI support into a variety of settings and apps. - Data security and privacy and privacy risks including data collected by AI based accessibility technologies; and the possibility of disability disclosure. - Disability-specific AI risks and biases including both direct bias (during AI use by the disabled person) and indirect bias (when AI is used by someone else on data relating to a disabled person). 

---
# Policy Frameworks for Transparent Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Yihang Chen, Haikang Deng, Kaiqiao Han, Qingyue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.14521)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances large language models (LLMs) by decomposing complex problems into step-by-step solutions, improving performance on reasoning tasks. However, current CoT disclosure policies vary widely across different models in frontend visibility, API access, and pricing strategies, lacking a unified policy framework. This paper analyzes the dual-edged implications of full CoT disclosure: while it empowers small-model distillation, fosters trust, and enables error diagnosis, it also risks violating intellectual property, enabling misuse, and incurring operational costs. We propose a tiered-access policy framework that balances transparency, accountability, and security by tailoring CoT availability to academic, business, and general users through ethical licensing, structured reasoning outputs, and cross-tier safeguards. By harmonizing accessibility with ethical and operational considerations, this framework aims to advance responsible AI deployment while mitigating risks of misuse or misinterpretation. 

---
# Content ARCs: Decentralized Content Rights in the Age of Generative AI 

**Authors**: Kar Balan, Andrew Gilbert, John Collomosse  

**Link**: [PDF](https://arxiv.org/pdf/2503.14519)  

**Abstract**: The rise of Generative AI (GenAI) has sparked significant debate over balancing the interests of creative rightsholders and AI developers. As GenAI models are trained on vast datasets that often include copyrighted material, questions around fair compensation and proper attribution have become increasingly urgent. To address these challenges, this paper proposes a framework called \emph{Content ARCs} (Authenticity, Rights, Compensation). By combining open standards for provenance and dynamic licensing with data attribution, and decentralized technologies, Content ARCs create a mechanism for managing rights and compensating creators for using their work in AI training. We characterize several nascent works in the AI data licensing space within Content ARCs and identify where challenges remain to fully implement the end-to-end framework. 

---
# Cafe-Talk: Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control 

**Authors**: Hejia Chen, Haoxian Zhang, Shoulong Zhang, Xiaoqiang Liu, Sisi Zhuang, Yuan Zhang, Pengfei Wan, Di Zhang, Shuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.14517)  

**Abstract**: Speech-driven 3D talking face method should offer both accurate lip synchronization and controllable expressions. Previous methods solely adopt discrete emotion labels to globally control expressions throughout sequences while limiting flexible fine-grained facial control within the spatiotemporal domain. We propose a diffusion-transformer-based 3D talking face generation model, Cafe-Talk, which simultaneously incorporates coarse- and fine-grained multimodal control conditions. Nevertheless, the entanglement of multiple conditions challenges achieving satisfying performance. To disentangle speech audio and fine-grained conditions, we employ a two-stage training pipeline. Specifically, Cafe-Talk is initially trained using only speech audio and coarse-grained conditions. Then, a proposed fine-grained control adapter gradually adds fine-grained instructions represented by action units (AUs), preventing unfavorable speech-lip synchronization. To disentangle coarse- and fine-grained conditions, we design a swap-label training mechanism, which enables the dominance of the fine-grained conditions. We also devise a mask-based CFG technique to regulate the occurrence and intensity of fine-grained control. In addition, a text-based detector is introduced with text-AU alignment to enable natural language user input and further support multimodal control. Extensive experimental results prove that Cafe-Talk achieves state-of-the-art lip synchronization and expressiveness performance and receives wide acceptance in fine-grained control in user studies. Project page: this https URL 

---
# Acceptance or Rejection of Lots while Minimizing and Controlling Type I and Type II Errors 

**Authors**: Edson Luiz Ursini, Elaine Cristina Catapani Poletti, Loreno Menezes da Silveira, José Roberto Emiliano Leite  

**Link**: [PDF](https://arxiv.org/pdf/2503.14514)  

**Abstract**: The double hypothesis test (DHT) is a test that allows controlling Type I (producer) and Type II (consumer) errors. It is possible to say whether the batch has a defect rate, p, between 1.5 and 2%, or between 2 and 5%, or between 5 and 10%, and so on, until finding a required value for this probability. Using the two probabilities side by side, the Type I error for the lower probability distribution and the Type II error for the higher probability distribution, both can be controlled and minimized. It can be applied in the development or manufacturing process of a batch of components, or in the case of purchasing from a supplier, when the percentage of defects (p) is unknown, considering the technology and/or process available to obtain them. The power of the test is amplified by the joint application of the Limit of Successive Failures (LSF) related to the Renewal Theory. To enable the choice of the most appropriate algorithm for each application. Four distributions are proposed for the Bernoulli event sequence, including their computational efforts: Binomial, Binomial approximated by Poisson, and Binomial approximated by Gaussian (with two variants). Fuzzy logic rules are also applied to facilitate decision-making. 

---
# Synthetic Data Generation of Body Motion Data by Neural Gas Network for Emotion Recognition 

**Authors**: Seyed Muhammad Hossein Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2503.14513)  

**Abstract**: In the domain of emotion recognition using body motion, the primary challenge lies in the scarcity of diverse and generalizable datasets. Automatic emotion recognition uses machine learning and artificial intelligence techniques to recognize a person's emotional state from various data types, such as text, images, sound, and body motion. Body motion poses unique challenges as many factors, such as age, gender, ethnicity, personality, and illness, affect its appearance, leading to a lack of diverse and robust datasets specifically for emotion recognition. To address this, employing Synthetic Data Generation (SDG) methods, such as Generative Adversarial Networks (GANs) and Variational Auto Encoders (VAEs), offers potential solutions, though these methods are often complex. This research introduces a novel application of the Neural Gas Network (NGN) algorithm for synthesizing body motion data and optimizing diversity and generation speed. By learning skeletal structure topology, the NGN fits the neurons or gas particles on body joints. Generated gas particles, which form the skeletal structure later on, will be used to synthesize the new body posture. By attaching body postures over frames, the final synthetic body motion appears. We compared our generated dataset against others generated by GANs, VAEs, and another benchmark algorithm, using benchmark metrics such as Fréchet Inception Distance (FID), Diversity, and a few more. Furthermore, we continued evaluation using classification metrics such as accuracy, precision, recall, and a few others. Joint-related features or kinematic parameters were extracted, and the system assessed model performance against unseen data. Our findings demonstrate that the NGN algorithm produces more realistic and emotionally distinct body motion data and does so with more synthesizing speed than existing methods. 

---
