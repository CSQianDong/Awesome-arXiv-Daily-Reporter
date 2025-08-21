# Privileged Self-Access Matters for Introspection in AI 

**Authors**: Siyuan Song, Harvey Lederman, Jennifer Hu, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2508.14802)  

**Abstract**: Whether AI models can introspect is an increasingly important practical question. But there is no consensus on how introspection is to be defined. Beginning from a recently proposed ''lightweight'' definition, we argue instead for a thicker one. According to our proposal, introspection in AI is any process which yields information about internal states through a process more reliable than one with equal or lower computational cost available to a third party. Using experiments where LLMs reason about their internal temperature parameters, we show they can appear to have lightweight introspection while failing to meaningfully introspect per our proposed definition. 

---
# Data-Driven Probabilistic Evaluation of Logic Properties with PAC-Confidence on Mealy Machines 

**Authors**: Swantje Plambeck, Ali Salamati, Eyke Huellermeier, Goerschwin Fey  

**Link**: [PDF](https://arxiv.org/pdf/2508.14710)  

**Abstract**: Cyber-Physical Systems (CPS) are complex systems that require powerful models for tasks like verification, diagnosis, or debugging. Often, suitable models are not available and manual extraction is difficult. Data-driven approaches then provide a solution to, e.g., diagnosis tasks and verification problems based on data collected from the system. In this paper, we consider CPS with a discrete abstraction in the form of a Mealy machine. We propose a data-driven approach to determine the safety probability of the system on a finite horizon of n time steps. The approach is based on the Probably Approximately Correct (PAC) learning paradigm. Thus, we elaborate a connection between discrete logic and probabilistic reachability analysis of systems, especially providing an additional confidence on the determined probability. The learning process follows an active learning paradigm, where new learning data is sampled in a guided way after an initial learning set is collected. We validate the approach with a case study on an automated lane-keeping system. 

---
# MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers 

**Authors**: Ziyang Luo, Zhiqi Shen, Wenzhuo Yang, Zirui Zhao, Prathyusha Jwalapuram, Amrita Saha, Doyen Sahoo, Silvio Savarese, Caiming Xiong, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14704)  

**Abstract**: The Model Context Protocol has emerged as a transformative standard for connecting large language models to external data sources and tools, rapidly gaining adoption across major AI providers and development platforms. However, existing benchmarks are overly simplistic and fail to capture real application challenges such as long-horizon reasoning and large, unfamiliar tool spaces. To address this critical gap, we introduce MCP-Universe, the first comprehensive benchmark specifically designed to evaluate LLMs in realistic and hard tasks through interaction with real-world MCP servers. Our benchmark encompasses 6 core domains spanning 11 different MCP servers: Location Navigation, Repository Management, Financial Analysis, 3D Design, Browser Automation, and Web Searching. To ensure rigorous evaluation, we implement execution-based evaluators, including format evaluators for agent format compliance, static evaluators for time-invariant content matching, and dynamic evaluators that automatically retrieve real-time ground truth for temporally sensitive tasks. Through extensive evaluation of leading LLMs, we find that even SOTA models such as GPT-5 (43.72%), Grok-4 (33.33%) and Claude-4.0-Sonnet (29.44%) exhibit significant performance limitations. In addition, our benchmark poses a significant long-context challenge for LLM agents, as the number of input tokens increases rapidly with the number of interaction steps. Moreover, it introduces an unknown-tools challenge, as LLM agents often lack familiarity with the precise usage of the MCP servers. Notably, enterprise-level agents like Cursor cannot achieve better performance than standard ReAct frameworks. Beyond evaluation, we open-source our extensible evaluation framework with UI support, enabling researchers and practitioners to seamlessly integrate new agents and MCP servers while fostering innovation in the rapidly evolving MCP ecosystem. 

---
# Entropy-Constrained Strategy Optimization in Urban Floods: A Multi-Agent Framework with LLM and Knowledge Graph Integration 

**Authors**: Peilin Ji, Xiao Xue, Simeng Wang, Wenhao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14654)  

**Abstract**: In recent years, the increasing frequency of extreme urban rainfall events has posed significant challenges to emergency scheduling systems. Urban flooding often leads to severe traffic congestion and service disruptions, threatening public safety and mobility. However, effective decision making remains hindered by three key challenges: (1) managing trade-offs among competing goals (e.g., traffic flow, task completion, and risk mitigation) requires dynamic, context-aware strategies; (2) rapidly evolving environmental conditions render static rules inadequate; and (3) LLM-generated strategies frequently suffer from semantic instability and execution inconsistency. Existing methods fail to align perception, global optimization, and multi-agent coordination within a unified framework. To tackle these challenges, we introduce H-J, a hierarchical multi-agent framework that integrates knowledge-guided prompting, entropy-constrained generation, and feedback-driven optimization. The framework establishes a closed-loop pipeline spanning from multi-source perception to strategic execution and continuous refinement. We evaluate H-J on real-world urban topology and rainfall data under three representative conditions: extreme rainfall, intermittent bursts, and daily light rain. Experiments show that H-J outperforms rule-based and reinforcement-learning baselines in traffic smoothness, task success rate, and system robustness. These findings highlight the promise of uncertainty-aware, knowledge-constrained LLM-based approaches for enhancing resilience in urban flood response. 

---
# LeanGeo: Formalizing Competitional Geometry problems in Lean 

**Authors**: Chendong Song, Zihan Wang, Frederick Pu, Haiming Wang, Xiaohan Lin, Junqi Liu, Jia Li, Zhengying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14644)  

**Abstract**: Geometry problems are a crucial testbed for AI reasoning capabilities. Most existing geometry solving systems cannot express problems within a unified framework, thus are difficult to integrate with other mathematical fields. Besides, since most geometric proofs rely on intuitive diagrams, verifying geometry problems is particularly challenging. To address these gaps, we introduce LeanGeo, a unified formal system for formalizing and solving competition-level geometry problems within the Lean 4 theorem prover. LeanGeo features a comprehensive library of high-level geometric theorems with Lean's foundational logic, enabling rigorous proof verification and seamless integration with Mathlib. We also present LeanGeo-Bench, a formal geometry benchmark in LeanGeo, comprising problems from the International Mathematical Olympiad (IMO) and other advanced sources. Our evaluation demonstrates the capabilities and limitations of state-of-the-art Large Language Models on this benchmark, highlighting the need for further advancements in automated geometric reasoning. We open source the theorem library and the benchmark of LeanGeo at this https URL. 

---
# Who Sees What? Structured Thought-Action Sequences for Epistemic Reasoning in LLMs 

**Authors**: Luca Annese, Sabrina Patania, Silvia Serino, Tom Foulsham, Silvia Rossi, Azzurra Ruggeri, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2508.14564)  

**Abstract**: Recent advances in large language models (LLMs) and reasoning frameworks have opened new possibilities for improving the perspective -taking capabilities of autonomous agents. However, tasks that involve active perception, collaborative reasoning, and perspective taking (understanding what another agent can see or knows) pose persistent challenges for current LLM-based systems. This study investigates the potential of structured examples derived from transformed solution graphs generated by the Fast Downward planner to improve the performance of LLM-based agents within a ReAct framework. We propose a structured solution-processing pipeline that generates three distinct categories of examples: optimal goal paths (G-type), informative node paths (E-type), and step-by-step optimal decision sequences contrasting alternative actions (L-type). These solutions are further converted into ``thought-action'' examples by prompting an LLM to explicitly articulate the reasoning behind each decision. While L-type examples slightly reduce clarification requests and overall action steps, they do not yield consistent improvements. Agents are successful in tasks requiring basic attentional filtering but struggle in scenarios that required mentalising about occluded spaces or weighing the costs of epistemic actions. These findings suggest that structured examples alone are insufficient for robust perspective-taking, underscoring the need for explicit belief tracking, cost modelling, and richer environments to enable socially grounded collaboration in LLM-based agents. 

---
# The Agent Behavior: Model, Governance and Challenges in the AI Digital Age 

**Authors**: Qiang Zhang, Pei Yan, Yijia Xu, Chuanpo Fu, Yong Fang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14415)  

**Abstract**: Advancements in AI have led to agents in networked environments increasingly mirroring human behavior, thereby blurring the boundary between artificial and human actors in specific contexts. This shift brings about significant challenges in trust, responsibility, ethics, security and etc. The difficulty in supervising of agent behaviors may lead to issues such as data contamination and unclear accountability. To address these challenges, this paper proposes the "Network Behavior Lifecycle" model, which divides network behavior into 6 stages and systematically analyzes the behavioral differences between humans and agents at each stage. Based on these insights, the paper further introduces the "Agent for Agent (A4A)" paradigm and the "Human-Agent Behavioral Disparity (HABD)" model, which examine the fundamental distinctions between human and agent behaviors across 5 dimensions: decision mechanism, execution efficiency, intention-behavior consistency, behavioral inertia, and irrational patterns. The effectiveness of the model is verified through real-world cases such as red team penetration and blue team defense. Finally, the paper discusses future research directions in dynamic cognitive governance architecture, behavioral disparity quantification, and meta-governance protocol stacks, aiming to provide a theoretical foundation and technical roadmap for secure and trustworthy human-agent collaboration. 

---
# Automated Optimization Modeling through Expert-Guided Large Language Model Reasoning 

**Authors**: Beinuo Yang, Qishen Zhou, Junyi Li, Xingchen Su, Simon Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14410)  

**Abstract**: Optimization Modeling (OM) is essential for solving complex decision-making problems. However, the process remains time-consuming and error-prone, heavily relying on domain experts. While Large Language Models (LLMs) show promise in addressing these challenges through their natural language understanding and reasoning capabilities, current approaches face three critical limitations: high benchmark labeling error rates reaching up to 42\%, narrow evaluation scope that only considers optimal values, and computational inefficiency due to heavy reliance on multi-agent systems or model fine-tuning. In this work, we first enhance existing datasets through systematic error correction and more comprehensive annotation. Additionally, we introduce LogiOR, a new optimization modeling benchmark from the logistics domain, containing more complex problems with standardized annotations. Furthermore, we present ORThought, a novel framework that leverages expert-level optimization modeling principles through chain-of-thought reasoning to automate the OM process. Through extensive empirical evaluation, we demonstrate that ORThought outperforms existing approaches, including multi-agent frameworks, with particularly significant advantages on complex optimization problems. Finally, we provide a systematic analysis of our method, identifying critical success factors and failure modes, providing valuable insights for future research on LLM-based optimization modeling. 

---
# Explaining Hitori Puzzles: Neurosymbolic Proof Staging for Sequential Decisions 

**Authors**: Maria Leonor Pacheco, Fabio Somenzi, Dananjay Srinivas, Ashutosh Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14294)  

**Abstract**: We propose a neurosymbolic approach to the explanation of complex sequences of decisions that combines the strengths of decision procedures and Large Language Models (LLMs). We demonstrate this approach by producing explanations for the solutions of Hitori puzzles. The rules of Hitori include local constraints that are effectively explained by short resolution proofs. However, they also include a connectivity constraint that is more suitable for visual explanations. Hence, Hitori provides an excellent testing ground for a flexible combination of SAT solvers and LLMs. We have implemented a tool that assists humans in solving Hitori puzzles, and we present experimental evidence of its effectiveness. 

---
# Large Language Models are Highly Aligned with Human Ratings of Emotional Stimuli 

**Authors**: Mattson Ogg, Chace Ashcraft, Ritwik Bose, Raphael Norman-Tenazas, Michael Wolmetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.14214)  

**Abstract**: Emotions exert an immense influence over human behavior and cognition in both commonplace and high-stress tasks. Discussions of whether or how to integrate large language models (LLMs) into everyday life (e.g., acting as proxies for, or interacting with, human agents), should be informed by an understanding of how these tools evaluate emotionally loaded stimuli or situations. A model's alignment with human behavior in these cases can inform the effectiveness of LLMs for certain roles or interactions. To help build this understanding, we elicited ratings from multiple popular LLMs for datasets of words and images that were previously rated for their emotional content by humans. We found that when performing the same rating tasks, GPT-4o responded very similarly to human participants across modalities, stimuli and most rating scales (r = 0.9 or higher in many cases). However, arousal ratings were less well aligned between human and LLM raters, while happiness ratings were most highly aligned. Overall LLMs aligned better within a five-category (happiness, anger, sadness, fear, disgust) emotion framework than within a two-dimensional (arousal and valence) organization. Finally, LLM ratings were substantially more homogenous than human ratings. Together these results begin to describe how LLM agents interpret emotional stimuli and highlight similarities and differences among biological and artificial intelligence in key behavioral domains. 

---
# Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs 

**Authors**: Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14896)  

**Abstract**: Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. All codes and experimental setups will be released to support the community. 

---
# Graph Structure Learning with Temporal Graph Information Bottleneck for Inductive Representation Learning 

**Authors**: Jiafeng Xiong, Rizos Sakellariou  

**Link**: [PDF](https://arxiv.org/pdf/2508.14859)  

**Abstract**: Temporal graph learning is crucial for dynamic networks where nodes and edges evolve over time and new nodes continuously join the system. Inductive representation learning in such settings faces two major challenges: effectively representing unseen nodes and mitigating noisy or redundant graph information. We propose GTGIB, a versatile framework that integrates Graph Structure Learning (GSL) with Temporal Graph Information Bottleneck (TGIB). We design a novel two-step GSL-based structural enhancer to enrich and optimize node neighborhoods and demonstrate its effectiveness and efficiency through theoretical proofs and experiments. The TGIB refines the optimized graph by extending the information bottleneck principle to temporal graphs, regularizing both edges and features based on our derived tractable TGIB objective function via variational approximation, enabling stable and efficient optimization. GTGIB-based models are evaluated to predict links on four real-world datasets; they outperform existing methods in all datasets under the inductive setting, with significant and consistent improvement in the transductive setting. 

---
# $TIME[t] \subseteq SPACE[O(\sqrt{t})]$ via Tree Height Compression 

**Authors**: Logan Nye  

**Link**: [PDF](https://arxiv.org/pdf/2508.14831)  

**Abstract**: We prove a square-root space simulation for deterministic multitape Turing machines, showing $\TIME[t] \subseteq \SPACE[O(\sqrt{t})]$. The key step is a Height Compression Theorem that uniformly (and in logspace) reshapes the canonical left-deep succinct computation tree for a block-respecting run into a binary tree whose evaluation-stack depth along any DFS path is $O(\log T)$ for $T = \lceil t/b \rceil$, while preserving $O(b)$ work at leaves, $O(1)$ at internal nodes, and edges that are logspace-checkable; semantic correctness across merges is witnessed by an exact $O(b)$ window replay at the unique interface. The proof uses midpoint (balanced) recursion, a per-path potential that bounds simultaneously active interfaces by $O(\log T)$, and an indegree-capping replacement of multiway merges by balanced binary combiners. Algorithmically, an Algebraic Replay Engine with constant-degree maps over a constant-size field, together with pointerless DFS and index-free streaming, ensures constant-size per-level tokens and eliminates wide counters, yielding the additive tradeoff $S(b)=O(b + \log(t/b))$ for block sizes $b \ge b_0$ with $b_0 = \Theta(\log t)$, which at the canonical choice $b = \Theta(\sqrt{t})$ gives $O(\sqrt{t})$ space; the $b_0$ threshold rules out degenerate blocks where addressing scratch would dominate the window footprint. The construction is uniform, relativizes, and is robust to standard model choices. Consequences include branching-program upper bounds $2^{O(\sqrt{s})}$ for size-$s$ bounded-fan-in circuits, tightened quadratic-time lower bounds for $\SPACE[n]$-complete problems via the standard hierarchy argument, and $O(\sqrt{t})$-space certifying interpreters; under explicit locality assumptions, the framework extends to geometric $d$-dimensional models. 

---
# Long Chain-of-Thought Reasoning Across Languages 

**Authors**: Josh Barua, Seun Eisape, Kayo Yin, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2508.14828)  

**Abstract**: Scaling inference through long chains-of-thought (CoTs) has unlocked impressive reasoning capabilities in large language models (LLMs), yet the reasoning process remains almost exclusively English-centric. We construct translated versions of two popular English reasoning datasets, fine-tune Qwen 2.5 (7B) and Qwen 3 (8B) models, and present a systematic study of long CoT generation across French, Japanese, Latvian, and Swahili. Our experiments reveal three key findings. First, the efficacy of using English as a pivot language varies by language: it provides no benefit for French, improves performance when used as the reasoning language for Japanese and Latvian, and proves insufficient for Swahili where both task comprehension and reasoning remain poor. Second, extensive multilingual pretraining in Qwen 3 narrows but does not eliminate the cross-lingual performance gap. A lightweight fine-tune using only 1k traces still improves performance by over 30\% in Swahili. Third, data quality versus scale trade-offs are language dependent: small, carefully curated datasets suffice for English and French, whereas larger but noisier corpora prove more effective for Swahili and Latvian. Together, these results clarify when and why long CoTs transfer across languages and provide translated datasets to foster equitable multilingual reasoning research. 

---
# From Passive Tool to Socio-cognitive Teammate: A Conceptual Framework for Agentic AI in Human-AI Collaborative Learning 

**Authors**: Lixiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14825)  

**Abstract**: The role of Artificial Intelligence (AI) in education is undergoing a rapid transformation, moving beyond its historical function as an instructional tool towards a new potential as an active participant in the learning process. This shift is driven by the emergence of agentic AI, autonomous systems capable of proactive, goal-directed action. However, the field lacks a robust conceptual framework to understand, design, and evaluate this new paradigm of human-AI interaction in learning. This paper addresses this gap by proposing a novel conceptual framework (the APCP framework) that charts the transition from AI as a tool to AI as a collaborative partner. We present a four-level model of escalating AI agency within human-AI collaborative learning: (1) the AI as an Adaptive Instrument, (2) the AI as a Proactive Assistant, (3) the AI as a Co-Learner, and (4) the AI as a Peer Collaborator. Grounded in sociocultural theories of learning and Computer-Supported Collaborative Learning (CSCL), this framework provides a structured vocabulary for analysing the shifting roles and responsibilities between human and AI agents. The paper further engages in a critical discussion of the philosophical underpinnings of collaboration, examining whether an AI, lacking genuine consciousness or shared intentionality, can be considered a true collaborator. We conclude that while AI may not achieve authentic phenomenological partnership, it can be designed as a highly effective functional collaborator. This distinction has significant implications for pedagogy, instructional design, and the future research agenda for AI in education, urging a shift in focus towards creating learning environments that harness the complementary strengths of both human and AI. 

---
# Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs 

**Authors**: Skatje Myers, Dmitriy Dligach, Timothy A. Miller, Samantha Barr, Yanjun Gao, Matthew Churpek, Anoop Mayampurath, Majid Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.14817)  

**Abstract**: Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even state-of-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using recent notes, and approaches the performance of using the models' full context while requiring drastically fewer input tokens. Our results suggest that RAG remains a competitive and efficient approach even as newer models become capable of handling increasingly longer amounts of text. 

---
# TransLight: Image-Guided Customized Lighting Control with Generative Decoupling 

**Authors**: Zongming Li, Lianghui Zhu, Haocheng Shen, Longjin Ran, Wenyu Liu, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14814)  

**Abstract**: Most existing illumination-editing approaches fail to simultaneously provide customized control of light effects and preserve content integrity. This makes them less effective for practical lighting stylization requirements, especially in the challenging task of transferring complex light effects from a reference image to a user-specified target image. To address this problem, we propose TransLight, a novel framework that enables high-fidelity and high-freedom transfer of light effects. Extracting the light effect from the reference image is the most critical and challenging step in our method. The difficulty lies in the complex geometric structure features embedded in light effects that are highly coupled with content in real-world scenarios. To achieve this, we first present Generative Decoupling, where two fine-tuned diffusion models are used to accurately separate image content and light effects, generating a newly curated, million-scale dataset of image-content-light triplets. Then, we employ IC-Light as the generative model and train our model with our triplets, injecting the reference lighting image as an additional conditioning signal. The resulting TransLight model enables customized and natural transfer of diverse light effects. Notably, by thoroughly disentangling light effects from reference images, our generative decoupling strategy endows TransLight with highly flexible illumination control. Experimental results establish TransLight as the first method to successfully transfer light effects across disparate images, delivering more customized illumination control than existing techniques and charting new directions for research in illumination harmonization and editing. 

---
# DINOv3 with Test-Time Training for Medical Image Registration 

**Authors**: Shansong Wang, Mojtaba Safari, Mingzhe Hu, Qiang Li, Chih-Wei Chang, Richard LJ Qiu, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14809)  

**Abstract**: Prior medical image registration approaches, particularly learning-based methods, often require large amounts of training data, which constrains clinical adoption. To overcome this limitation, we propose a training-free pipeline that relies on a frozen DINOv3 encoder and test-time optimization of the deformation field in feature space. Across two representative benchmarks, the method is accurate and yields regular deformations. On Abdomen MR-CT, it attained the best mean Dice score (DSC) of 0.790 together with the lowest 95th percentile Hausdorff Distance (HD95) of 4.9+-5.0 and the lowest standard deviation of Log-Jacobian (SDLogJ) of 0.08+-0.02. On ACDC cardiac MRI, it improves mean DSC to 0.769 and reduces SDLogJ to 0.11 and HD95 to 4.8, a marked gain over the initial alignment. The results indicate that operating in a compact foundation feature space at test time offers a practical and general solution for clinical registration without additional training. 

---
# MF-LPR$^2$: Multi-Frame License Plate Image Restoration and Recognition using Optical Flow 

**Authors**: Kihyun Na, Junseok Oh, Youngkwan Cho, Bumjin Kim, Sungmin Cho, Jinyoung Choi, Injung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.14797)  

**Abstract**: License plate recognition (LPR) is important for traffic law enforcement, crime investigation, and surveillance. However, license plate areas in dash cam images often suffer from low resolution, motion blur, and glare, which make accurate recognition challenging. Existing generative models that rely on pretrained priors cannot reliably restore such poor-quality images, frequently introducing severe artifacts and distortions. To address this issue, we propose a novel multi-frame license plate restoration and recognition framework, MF-LPR$^2$, which addresses ambiguities in poor-quality images by aligning and aggregating neighboring frames instead of relying on pretrained knowledge. To achieve accurate frame alignment, we employ a state-of-the-art optical flow estimator in conjunction with carefully designed algorithms that detect and correct erroneous optical flow estimations by leveraging the spatio-temporal consistency inherent in license plate image sequences. Our approach enhances both image quality and recognition accuracy while preserving the evidential content of the input images. In addition, we constructed a novel Realistic LPR (RLPR) dataset to evaluate MF-LPR$^2$. The RLPR dataset contains 200 pairs of low-quality license plate image sequences and high-quality pseudo ground-truth images, reflecting the complexities of real-world scenarios. In experiments, MF-LPR$^2$ outperformed eight recent restoration models in terms of PSNR, SSIM, and LPIPS by significant margins. In recognition, MF-LPR$^2$ achieved an accuracy of 86.44%, outperforming both the best single-frame LPR (14.04%) and the multi-frame LPR (82.55%) among the eleven baseline models. The results of ablation studies confirm that our filtering and refinement algorithms significantly contribute to these improvements. 

---
# TransLLM: A Unified Multi-Task Foundation Framework for Urban Transportation via Learnable Prompting 

**Authors**: Jiaming Leng, Yunying Bi, Chuan Qin, Bing Yin, Yanyong Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14782)  

**Abstract**: Urban transportation systems encounter diverse challenges across multiple tasks, such as traffic forecasting, electric vehicle (EV) charging demand prediction, and taxi dispatch. Existing approaches suffer from two key limitations: small-scale deep learning models are task-specific and data-hungry, limiting their generalizability across diverse scenarios, while large language models (LLMs), despite offering flexibility through natural language interfaces, struggle with structured spatiotemporal data and numerical reasoning in transportation domains. To address these limitations, we propose TransLLM, a unified foundation framework that integrates spatiotemporal modeling with large language models through learnable prompt composition. Our approach features a lightweight spatiotemporal encoder that captures complex dependencies via dilated temporal convolutions and dual-adjacency graph attention networks, seamlessly interfacing with LLMs through structured embeddings. A novel instance-level prompt routing mechanism, trained via reinforcement learning, dynamically personalizes prompts based on input characteristics, moving beyond fixed task-specific templates. The framework operates by encoding spatiotemporal patterns into contextual representations, dynamically composing personalized prompts to guide LLM reasoning, and projecting the resulting representations through specialized output layers to generate task-specific predictions. Experiments across seven datasets and three tasks demonstrate the exceptional effectiveness of TransLLM in both supervised and zero-shot settings. Compared to ten baseline models, it delivers competitive performance on both regression and planning problems, showing strong generalization and cross-task adaptability. Our code is available at this https URL. 

---
# PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning 

**Authors**: Ruheng Wang, Hang Zhang, Trieu Nguyen, Shasha Feng, Hao-Wei Pang, Xiang Yu, Li Xiao, Peter Zhiping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14765)  

**Abstract**: Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery. 

---
# Reliable generation of isomorphic physics problems using ChatGPT with prompt-chaining and tool use 

**Authors**: Zhongzhou Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14755)  

**Abstract**: We present a method for generating large numbers of isomorphic physics problems using ChatGPT through prompt chaining and tool use. This approach enables precise control over structural variations-such as numeric values and spatial relations-while supporting diverse contextual variations in the problem body. By utilizing the Python code interpreter, the method supports automatic solution validation and simple diagram generation, addressing key limitations in existing LLM-based methods. We generated two example isomorphic problem banks and compared the outcome against simpler prompt-based approaches. Results show that prompt-chaining produces significantly higher quality and more consistent outputs than simpler, non-chaining prompts. This work demonstrates a promising method for efficient problem creation accessible to the average instructor, which opens new possibilities for personalized adaptive testing and automated content development. 

---
# Cross-Modality Controlled Molecule Generation with Diffusion Language Model 

**Authors**: Yunzhe Zhang, Yifei Wang, Khanh Vinh Nguyen, Pengyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.14748)  

**Abstract**: Current SMILES-based diffusion models for molecule generation typically support only unimodal constraint. They inject conditioning signals at the start of the training process and require retraining a new model from scratch whenever the constraint changes. However, real-world applications often involve multiple constraints across different modalities, and additional constraints may emerge over the course of a study. This raises a challenge: how to extend a pre-trained diffusion model not only to support cross-modality constraints but also to incorporate new ones without retraining. To tackle this problem, we propose the Cross-Modality Controlled Molecule Generation with Diffusion Language Model (CMCM-DLM), demonstrated by two distinct cross modalities: molecular structure and chemical properties. Our approach builds upon a pre-trained diffusion model, incorporating two trainable modules, the Structure Control Module (SCM) and the Property Control Module (PCM), and operates in two distinct phases during the generation process. In Phase I, we employs the SCM to inject structural constraints during the early diffusion steps, effectively anchoring the molecular backbone. Phase II builds on this by further introducing PCM to guide the later stages of inference to refine the generated molecules, ensuring their chemical properties match the specified targets. Experimental results on multiple datasets demonstrate the efficiency and adaptability of our approach, highlighting CMCM-DLM's significant advancement in molecular generation for drug discovery applications. 

---
# Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Khalid Qaraqe, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2508.14735)  

**Abstract**: Large language models (LLMs) are increasingly applied in multilingual contexts, yet their capacity for consistent, logically grounded alignment across languages remains underexplored. We present a controlled evaluation framework for multilingual natural language inference (NLI) that generates synthetic, logic-based premise-hypothesis pairs and translates them into a typologically diverse set of languages. This design enables precise control over semantic relations and allows testing in both monolingual and mixed-language (code-switched) conditions. Surprisingly, code-switching does not degrade, and can even improve, performance, suggesting that translation-induced lexical variation may serve as a regularization signal. We validate semantic preservation through embedding-based similarity analyses and cross-lingual alignment visualizations, confirming the fidelity of translated pairs. Our findings expose both the potential and the brittleness of current LLM cross-lingual reasoning, and identify code-switching as a promising lever for improving multilingual robustness. Code available at: this https URL 

---
# AFABench: A Generic Framework for Benchmarking Active Feature Acquisition 

**Authors**: Valter Schütz, Han Wu, Reza Rezvan, Linus Aronsson, Morteza Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2508.14734)  

**Abstract**: In many real-world scenarios, acquiring all features of a data instance can be expensive or impractical due to monetary cost, latency, or privacy concerns. Active Feature Acquisition (AFA) addresses this challenge by dynamically selecting a subset of informative features for each data instance, trading predictive performance against acquisition cost. While numerous methods have been proposed for AFA, ranging from greedy information-theoretic strategies to non-myopic reinforcement learning approaches, fair and systematic evaluation of these methods has been hindered by the lack of standardized benchmarks. In this paper, we introduce AFABench, the first benchmark framework for AFA. Our benchmark includes a diverse set of synthetic and real-world datasets, supports a wide range of acquisition policies, and provides a modular design that enables easy integration of new methods and tasks. We implement and evaluate representative algorithms from all major categories, including static, greedy, and reinforcement learning-based approaches. To test the lookahead capabilities of AFA policies, we introduce a novel synthetic dataset, AFAContext, designed to expose the limitations of greedy selection. Our results highlight key trade-offs between different AFA strategies and provide actionable insights for future research. The benchmark code is available at: this https URL. 

---
# Emerson-Lei and Manna-Pnueli Games for LTLf+ and PPLTL+ Synthesis 

**Authors**: Daniel Hausmann, Shufang Zhu, Gianmarco Parretti, Christoph Weinhuber, Giuseppe De Giacomo, Nir Piterman  

**Link**: [PDF](https://arxiv.org/pdf/2508.14725)  

**Abstract**: Recently, the Manna-Pnueli Hierarchy has been used to define the temporal logics LTLfp and PPLTLp, which allow to use finite-trace LTLf/PPLTL techniques in infinite-trace settings while achieving the expressiveness of full LTL. In this paper, we present the first actual solvers for reactive synthesis in these logics. These are based on games on graphs that leverage DFA-based techniques from LTLf/PPLTL to construct the game arena. We start with a symbolic solver based on Emerson-Lei games, which reduces lower-class properties (guarantee, safety) to higher ones (recurrence, persistence) before solving the game. We then introduce Manna-Pnueli games, which natively embed Manna-Pnueli objectives into the arena. These games are solved by composing solutions to a DAG of simpler Emerson-Lei games, resulting in a provably more efficient approach. We implemented the solvers and practically evaluated their performance on a range of representative formulas. The results show that Manna-Pnueli games often offer significant advantages, though not universally, indicating that combining both approaches could further enhance practical performance. 

---
# Transplant Then Regenerate: A New Paradigm for Text Data Augmentation 

**Authors**: Guangzhan Wang, Hongyu Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14723)  

**Abstract**: Data augmentation is a critical technique in deep learning. Traditional methods like Back-translation typically focus on lexical-level rephrasing, which primarily produces variations with the same semantics. While large language models (LLMs) have enhanced text augmentation by their "knowledge emergence" capability, controlling the style and structure of these outputs remains challenging and requires meticulous prompt engineering. In this paper, we propose LMTransplant, a novel text augmentation paradigm leveraging LLMs. The core idea of LMTransplant is transplant-then-regenerate: incorporating seed text into a context expanded by LLM, and asking the LLM to regenerate a variant based on the expanded context. This strategy allows the model to create more diverse and creative content-level variants by fully leveraging the knowledge embedded in LLMs, while preserving the core attributes of the original text. We evaluate LMTransplant across various text-related tasks, demonstrating its superior performance over existing text augmentation methods. Moreover, LMTransplant demonstrates exceptional scalability as the size of augmented data grows. 

---
# ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine 

**Authors**: Junying Chen, Zhenyang Cai, Zhiheng Liu, Yunjin Yang, Rongsheng Wang, Qingying Xiao, Xiangyi Feng, Zhan Su, Jing Guo, Xiang Wan, Guangjun Yu, Haizhou Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14706)  

**Abstract**: Despite the success of large language models (LLMs) in various domains, their potential in Traditional Chinese Medicine (TCM) remains largely underexplored due to two critical barriers: (1) the scarcity of high-quality TCM data and (2) the inherently multimodal nature of TCM diagnostics, which involve looking, listening, smelling, and pulse-taking. These sensory-rich modalities are beyond the scope of conventional LLMs. To address these challenges, we present ShizhenGPT, the first multimodal LLM tailored for TCM. To overcome data scarcity, we curate the largest TCM dataset to date, comprising 100GB+ of text and 200GB+ of multimodal data, including 1.2M images, 200 hours of audio, and physiological signals. ShizhenGPT is pretrained and instruction-tuned to achieve deep TCM knowledge and multimodal reasoning. For evaluation, we collect recent national TCM qualification exams and build a visual benchmark for Medicinal Recognition and Visual Diagnosis. Experiments demonstrate that ShizhenGPT outperforms comparable-scale LLMs and competes with larger proprietary models. Moreover, it leads in TCM visual understanding among existing multimodal LLMs and demonstrates unified perception across modalities like sound, pulse, smell, and vision, paving the way toward holistic multimodal perception and diagnosis in TCM. Datasets, models, and code are publicly available. We hope this work will inspire further exploration in this field. 

---
# Learning in Repeated Multi-Objective Stackelberg Games with Payoff Manipulation 

**Authors**: Phurinut Srisawad, Juergen Branke, Long Tran-Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2508.14705)  

**Abstract**: We study payoff manipulation in repeated multi-objective Stackelberg games, where a leader may strategically influence a follower's deterministic best response, e.g., by offering a share of their own payoff. We assume that the follower's utility function, representing preferences over multiple objectives, is unknown but linear, and its weight parameter must be inferred through interaction. This introduces a sequential decision-making challenge for the leader, who must balance preference elicitation with immediate utility maximisation. We formalise this problem and propose manipulation policies based on expected utility (EU) and long-term expected utility (longEU), which guide the leader in selecting actions and offering incentives that trade off short-term gains with long-term impact. We prove that under infinite repeated interactions, longEU converges to the optimal manipulation. Empirical results across benchmark environments demonstrate that our approach improves cumulative leader utility while promoting mutually beneficial outcomes, all without requiring explicit negotiation or prior knowledge of the follower's utility function. 

---
# Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection 

**Authors**: Jan Lum Fok, Qingwen Zeng, Shiping Chen, Oscar Fawkes, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14699)  

**Abstract**: Credit card fraud detection (CCFD) is a critical application of Machine Learning (ML) in the financial sector, where accurately identifying fraudulent transactions is essential for mitigating financial losses. ML models have demonstrated their effectiveness in fraud detection task, in particular with the tabular dataset. While adversarial attacks have been extensively studied in computer vision and deep learning, their impacts on the ML models, particularly those trained on CCFD tabular datasets, remains largely unexplored. These latent vulnerabilities pose significant threats to the security and stability of the financial industry, especially in high-value transactions where losses could be substantial. To address this gap, in this paper, we present a holistic framework that investigate the robustness of CCFD ML model against adversarial perturbations under different circumstances. Specifically, the gradient-based attack methods are incorporated into the tabular credit card transaction data in both black- and white-box adversarial attacks settings. Our findings confirm that tabular data is also susceptible to subtle perturbations, highlighting the need for heightened awareness among financial technology practitioners regarding ML model security and trustworthiness. Furthermore, the experiments by transferring adversarial samples from gradient-based attack method to non-gradient-based models also verify our findings. Our results demonstrate that such attacks remain effective, emphasizing the necessity of developing robust defenses for CCFD algorithms. 

---
# ECHO: Frequency-aware Hierarchical Encoding for Variable-length Signal 

**Authors**: Yucong Zhang, Juan Liu, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14689)  

**Abstract**: Pre-trained foundation models have demonstrated remarkable success in vision and language, yet their potential for general machine signal modeling-covering acoustic, vibration, and other industrial sensor data-remains under-explored. Existing approach using sub-band-based encoders has achieved competitive results but are limited by fixed input lengths, and the absence of explicit frequency positional encoding. In this work, we propose a novel foundation model that integrates an advanced band-split architecture with relative frequency positional embeddings, enabling precise spectral localization across arbitrary sampling configurations. The model supports inputs of arbitrary length without padding or segmentation, producing a concise embedding that retains both temporal and spectral fidelity. We evaluate our method on SIREN (this https URL), a newly introduced large-scale benchmark for machine signal encoding that unifies multiple datasets, including all DCASE task 2 challenges (2020-2025) and widely-used industrial signal corpora. Experimental results demonstrate consistent state-of-the-art performance in anomaly detection and fault identification, confirming the effectiveness and generalization capability of the proposed model. We open-sourced ECHO on this https URL. 

---
# ELATE: Evolutionary Language model for Automated Time-series Engineering 

**Authors**: Andrew Murray, Danial Dervovic, Michael Cashmore  

**Link**: [PDF](https://arxiv.org/pdf/2508.14667)  

**Abstract**: Time-series prediction involves forecasting future values using machine learning models. Feature engineering, whereby existing features are transformed to make new ones, is critical for enhancing model performance, but is often manual and time-intensive. Existing automation attempts rely on exhaustive enumeration, which can be computationally costly and lacks domain-specific insights. We introduce ELATE (Evolutionary Language model for Automated Time-series Engineering), which leverages a language model within an evolutionary framework to automate feature engineering for time-series data. ELATE employs time-series statistical measures and feature importance metrics to guide and prune features, while the language model proposes new, contextually relevant feature transformations. Our experiments demonstrate that ELATE improves forecasting accuracy by an average of 8.4% across various domains. 

---
# OneLoc: Geo-Aware Generative Recommender Systems for Local Life Service 

**Authors**: Zhipeng Wei, Kuo Cai, Junda She, Jie Chen, Minghao Chen, Yang Zeng, Qiang Luo, Wencong Zeng, Ruiming Tang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.14646)  

**Abstract**: Local life service is a vital scenario in Kuaishou App, where video recommendation is intrinsically linked with store's location information. Thus, recommendation in our scenario is challenging because we should take into account user's interest and real-time location at the same time. In the face of such complex scenarios, end-to-end generative recommendation has emerged as a new paradigm, such as OneRec in the short video scenario, OneSug in the search scenario, and EGA in the advertising scenario. However, in local life service, an end-to-end generative recommendation model has not yet been developed as there are some key challenges to be solved. The first challenge is how to make full use of geographic information. The second challenge is how to balance multiple objectives, including user interests, the distance between user and stores, and some other business objectives. To address the challenges, we propose OneLoc. Specifically, we leverage geographic information from different perspectives: (1) geo-aware semantic ID incorporates both video and geographic information for tokenization, (2) geo-aware self-attention in the encoder leverages both video location similarity and user's real-time location, and (3) neighbor-aware prompt captures rich context information surrounding users for generation. To balance multiple objectives, we use reinforcement learning and propose two reward functions, i.e., geographic reward and GMV reward. With the above design, OneLoc achieves outstanding offline and online performance. In fact, OneLoc has been deployed in local life service of Kuaishou App. It serves 400 million active users daily, achieving 21.016% and 17.891% improvements in terms of gross merchandise value (GMV) and orders numbers. 

---
# Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination 

**Authors**: João Vitor de Carvalho Silva, Douglas G. Macharet  

**Link**: [PDF](https://arxiv.org/pdf/2508.14635)  

**Abstract**: The ability to coordinate actions across multiple agents is critical for solving complex, real-world problems. Large Language Models (LLMs) have shown strong capabilities in communication, planning, and reasoning, raising the question of whether they can also support effective collaboration in multi-agent settings. In this work, we investigate the use of LLM agents to solve a structured victim rescue task that requires division of labor, prioritization, and cooperative planning. Agents operate in a fully known graph-based environment and must allocate resources to victims with varying needs and urgency levels. We systematically evaluate their performance using a suite of coordination-sensitive metrics, including task success rate, redundant actions, room conflicts, and urgency-weighted efficiency. This study offers new insights into the strengths and failure modes of LLMs in physically grounded multi-agent collaboration tasks, contributing to future benchmarks and architectural improvements. 

---
# A Study of the Scale Invariant Signal to Distortion Ratio in Speech Separation with Noisy References 

**Authors**: Simon Dahl Jepsen, Mads Græsbøll Christensen, Jesper Rindom Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14623)  

**Abstract**: This paper examines the implications of using the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) as both evaluation and training objective in supervised speech separation, when the training references contain noise, as is the case with the de facto benchmark WSJ0-2Mix. A derivation of the SI-SDR with noisy references reveals that noise limits the achievable SI-SDR, or leads to undesired noise in the separated outputs. To address this, a method is proposed to enhance references and augment the mixtures with WHAM!, aiming to train models that avoid learning noisy references. Two models trained on these enhanced datasets are evaluated with the non-intrusive NISQA.v2 metric. Results show reduced noise in separated speech but suggest that processing references may introduce artefacts, limiting overall quality gains. Negative correlation is found between SI-SDR and perceived noisiness across models on the WSJ0-2Mix and Libri2Mix test sets, underlining the conclusion from the derivation. 

---
# UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling 

**Authors**: Peiming Li, Ziyi Wang, Yulin Yuan, Hong Liu, Xiangming Meng, Junsong Yuan, Mengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14604)  

**Abstract**: Point cloud videos capture dynamic 3D motion while reducing the effects of lighting and viewpoint variations, making them highly effective for recognizing subtle and continuous human actions. Although Selective State Space Models (SSMs) have shown good performance in sequence modeling with linear complexity, the spatio-temporal disorder of point cloud videos hinders their unidirectional modeling when directly unfolding the point cloud video into a 1D sequence through temporally sequential scanning. To address this challenge, we propose the Unified Spatio-Temporal State Space Model (UST-SSM), which extends the latest advancements in SSMs to point cloud videos. Specifically, we introduce Spatial-Temporal Selection Scanning (STSS), which reorganizes unordered points into semantic-aware sequences through prompt-guided clustering, thereby enabling the effective utilization of points that are spatially and temporally distant yet similar within the sequence. For missing 4D geometric and motion details, Spatio-Temporal Structure Aggregation (STSA) aggregates spatio-temporal features and compensates. To improve temporal interaction within the sampled sequence, Temporal Interaction Sampling (TIS) enhances fine-grained temporal dependencies through non-anchor frame utilization and expanded receptive fields. Experimental results on the MSR-Action3D, NTU RGB+D, and Synthia 4D datasets validate the effectiveness of our method. Our code is available at this https URL. 

---
# An Open-Source HW-SW Co-Development Framework Enabling Efficient Multi-Accelerator Systems 

**Authors**: Ryan Albert Antonio, Joren Dumoulin, Xiaoling Yi, Josse Van Delm, Yunhao Deng, Guilherme Paim, Marian Verhelst  

**Link**: [PDF](https://arxiv.org/pdf/2508.14582)  

**Abstract**: Heterogeneous accelerator-centric compute clusters are emerging as efficient solutions for diverse AI workloads. However, current integration strategies often compromise data movement efficiency and encounter compatibility issues in hardware and software. This prevents a unified approach that balances performance and ease of use. To this end, we present SNAX, an open-source integrated HW-SW framework enabling efficient multi-accelerator platforms through a novel hybrid-coupling scheme, consisting of loosely coupled asynchronous control and tightly coupled data access. SNAX brings reusable hardware modules designed to enhance compute accelerator utilization, and its customizable MLIR-based compiler to automate key system management tasks, jointly enabling rapid development and deployment of customized multi-accelerator compute clusters. Through extensive experimentation, we demonstrate SNAX's efficiency and flexibility in a low-power heterogeneous SoC. Accelerators can easily be integrated and programmed to achieve > 10x improvement in neural network performance compared to other accelerator systems while maintaining accelerator utilization of > 90% in full system operation. 

---
# Mamba2 Meets Silence: Robust Vocal Source Separation for Sparse Regions 

**Authors**: Euiyeon Kim, Yong-Hoon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14556)  

**Abstract**: We introduce a new music source separation model tailored for accurate vocal isolation. Unlike Transformer-based approaches, which often fail to capture intermittently occurring vocals, our model leverages Mamba2, a recent state space model, to better capture long-range temporal dependencies. To handle long input sequences efficiently, we combine a band-splitting strategy with a dual-path architecture. Experiments show that our approach outperforms recent state-of-the-art models, achieving a cSDR of 11.03 dB-the best reported to date-and delivering substantial gains in uSDR. Moreover, the model exhibits stable and consistent performance across varying input lengths and vocal occurrence patterns. These results demonstrate the effectiveness of Mamba-based models for high-resolution audio processing and open up new directions for broader applications in audio research. 

---
# Towards LLM-generated explanations for Component-based Knowledge Graph Question Answering Systems 

**Authors**: Dennis Schiese, Aleksandr Perevalov, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2508.14553)  

**Abstract**: Over time, software systems have reached a level of complexity that makes it difficult for their developers and users to explain particular decisions made by them. In this paper, we focus on the explainability of component-based systems for Question Answering (QA). These components often conduct processes driven by AI methods, in which behavior and decisions cannot be clearly explained or justified, s.t., even for QA experts interpreting the executed process and its results is hard. To address this challenge, we present an approach that considers the components' input and output data flows as a source for representing the behavior and provide explanations for the components, enabling users to comprehend what happened. In the QA framework used here, the data flows of the components are represented as SPARQL queries (inputs) and RDF triples (outputs). Hence, we are also providing valuable insights on verbalization regarding these data types. In our experiments, the approach generates explanations while following template-based settings (baseline) or via the use of Large Language Models (LLMs) with different configurations (automatic generation). Our evaluation shows that the explanations generated via LLMs achieve high quality and mostly outperform template-based approaches according to the users' ratings. Therefore, it enables us to automatically explain the behavior and decisions of QA components to humans while using RDF and SPARQL as a context for explanations. 

---
# Adaptively Robust LLM Inference Optimization under Prediction Uncertainty 

**Authors**: Zixi Chen, Yinyu Ye, Zijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.14544)  

**Abstract**: We study the problem of optimizing Large Language Model (LLM) inference scheduling to minimize total latency. LLM inference is an online and multi-task service process and also heavily energy consuming by which a pre-trained LLM processes input requests and generates output tokens sequentially. Therefore, it is vital to improve its scheduling efficiency and reduce the power consumption while a great amount of prompt requests are arriving. A key challenge in LLM inference scheduling is that while the prompt length is known upon arrival, the output length, which critically impacts memory usage and processing time, is unknown. To address this uncertainty, we propose algorithms that leverage machine learning to predict output lengths, assuming the prediction provides an interval classification (min-max range) for each request.
We first design a conservative algorithm, $\mathcal{A}_{\max}$, which schedules requests based on the upper bound of predicted output lengths to prevent memory overflow. However, this approach is overly conservative: as prediction accuracy decreases, performance degrades significantly due to potential overestimation. To overcome this limitation, we propose $\mathcal{A}_{\min}$, an adaptive algorithm that initially treats the predicted lower bound as the output length and dynamically refines this estimate during inferencing. We prove that $\mathcal{A}_{\min}$ achieves a log-scale competitive ratio. Through numerical simulations, we demonstrate that $\mathcal{A}_{\min}$ often performs nearly as well as the hindsight scheduler, highlighting both its efficiency and robustness in practical scenarios. Moreover, $\mathcal{A}_{\min}$ relies solely on the lower bound of the prediction interval--an advantageous design choice since upper bounds on output length are typically more challenging to predict accurately. 

---
# Post-hoc LLM-Supported Debugging of Distributed Processes 

**Authors**: Dennis Schiese, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2508.14540)  

**Abstract**: In this paper, we address the problem of manual debugging, which nowadays remains resource-intensive and in some parts archaic. This problem is especially evident in increasingly complex and distributed software systems. Therefore, our objective of this work is to introduce an approach that can possibly be applied to any system, at both the macro- and micro-level, to ease this debugging process. This approach utilizes a system's process data, in conjunction with generative AI, to generate natural-language explanations. These explanations are generated from the actual process data, interface information, and documentation to guide the developers more efficiently to understand the behavior and possible errors of a process and its sub-processes. Here, we present a demonstrator that employs this approach on a component-based Java system. However, our approach is language-agnostic. Ideally, the generated explanations will provide a good understanding of the process, even if developers are not familiar with all the details of the considered system. Our demonstrator is provided as an open-source web application that is freely accessible to all users. 

---
# Beyond ReLU: Chebyshev-DQN for Enhanced Deep Q-Networks 

**Authors**: Saman Yazdannik, Morteza Tayefi, Shamim Sanisales  

**Link**: [PDF](https://arxiv.org/pdf/2508.14536)  

**Abstract**: The performance of Deep Q-Networks (DQN) is critically dependent on the ability of its underlying neural network to accurately approximate the action-value function. Standard function approximators, such as multi-layer perceptrons, may struggle to efficiently represent the complex value landscapes inherent in many reinforcement learning problems. This paper introduces a novel architecture, the Chebyshev-DQN (Ch-DQN), which integrates a Chebyshev polynomial basis into the DQN framework to create a more effective feature representation. By leveraging the powerful function approximation properties of Chebyshev polynomials, we hypothesize that the Ch-DQN can learn more efficiently and achieve higher performance. We evaluate our proposed model on the CartPole-v1 benchmark and compare it against a standard DQN with a comparable number of parameters. Our results demonstrate that the Ch-DQN with a moderate polynomial degree (N=4) achieves significantly better asymptotic performance, outperforming the baseline by approximately 39\%. However, we also find that the choice of polynomial degree is a critical hyperparameter, as a high degree (N=8) can be detrimental to learning. This work validates the potential of using orthogonal polynomial bases in deep reinforcement learning while also highlighting the trade-offs involved in model complexity. 

---
# EffiFusion-GAN: Efficient Fusion Generative Adversarial Network for Speech Enhancement 

**Authors**: Bin Wen, Tien-Ping Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14525)  

**Abstract**: We introduce EffiFusion-GAN (Efficient Fusion Generative Adversarial Network), a lightweight yet powerful model for speech enhancement. The model integrates depthwise separable convolutions within a multi-scale block to capture diverse acoustic features efficiently. An enhanced attention mechanism with dual normalization and residual refinement further improves training stability and convergence. Additionally, dynamic pruning is applied to reduce model size while maintaining performance, making the framework suitable for resource-constrained environments. Experimental evaluation on the public VoiceBank+DEMAND dataset shows that EffiFusion-GAN achieves a PESQ score of 3.45, outperforming existing models under the same parameter settings. 

---
# MISS: Multi-Modal Tree Indexing and Searching with Lifelong Sequential Behavior for Retrieval Recommendation 

**Authors**: Chengcheng Guo, Junda She, Kuo Cai, Shiyao Wang, Qigen Hu, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.14515)  

**Abstract**: Large-scale industrial recommendation systems typically employ a two-stage paradigm of retrieval and ranking to handle huge amounts of information. Recent research focuses on improving the performance of retrieval model. A promising way is to introduce extensive information about users and items. On one hand, lifelong sequential behavior is valuable. Existing lifelong behavior modeling methods in ranking stage focus on the interaction of lifelong behavior and candidate items from retrieval stage. In retrieval stage, it is difficult to utilize lifelong behavior because of a large corpus of candidate items. On the other hand, existing retrieval methods mostly relay on interaction information, potentially disregarding valuable multi-modal information. To solve these problems, we represent the pioneering exploration of leveraging multi-modal information and lifelong sequence model within the advanced tree-based retrieval model. We propose Multi-modal Indexing and Searching with lifelong Sequence (MISS), which contains a multi-modal index tree and a multi-modal lifelong sequence modeling module. Specifically, for better index structure, we propose multi-modal index tree, which is built using the multi-modal embedding to precisely represent item similarity. To precisely capture diverse user interests in user lifelong sequence, we propose collaborative general search unit (Co-GSU) and multi-modal general search unit (MM-GSU) for multi-perspective interests searching. 

---
# PB-IAD: Utilizing multimodal foundation models for semantic industrial anomaly detection in dynamic manufacturing environments 

**Authors**: Bernd Hofmann, Albert Scheck, Joerg Franke, Patrick Bruendl  

**Link**: [PDF](https://arxiv.org/pdf/2508.14504)  

**Abstract**: The detection of anomalies in manufacturing processes is crucial to ensure product quality and identify process deviations. Statistical and data-driven approaches remain the standard in industrial anomaly detection, yet their adaptability and usability are constrained by the dependence on extensive annotated datasets and limited flexibility under dynamic production conditions. Recent advances in the perception capabilities of foundation models provide promising opportunities for their adaptation to this downstream task. This paper presents PB-IAD (Prompt-based Industrial Anomaly Detection), a novel framework that leverages the multimodal and reasoning capabilities of foundation models for industrial anomaly detection. Specifically, PB-IAD addresses three key requirements of dynamic production environments: data sparsity, agile adaptability, and domain user centricity. In addition to the anomaly detection, the framework includes a prompt template that is specifically designed for iteratively implementing domain-specific process knowledge, as well as a pre-processing module that translates domain user inputs into effective system prompts. This user-centric design allows domain experts to customise the system flexibly without requiring data science expertise. The proposed framework is evaluated by utilizing GPT-4.1 across three distinct manufacturing scenarios, two data modalities, and an ablation study to systematically assess the contribution of semantic instructions. Furthermore, PB-IAD is benchmarked to state-of-the-art methods for anomaly detection such as PatchCore. The results demonstrate superior performance, particularly in data-sparse scenarios and low-shot settings, achieved solely through semantic instructions. 

---
# Exact Shapley Attributions in Quadratic-time for FANOVA Gaussian Processes 

**Authors**: Majid Mohammadi, Krikamol Muandet, Ilaria Tiddi, Annette Ten Teije, Siu Lun Chau  

**Link**: [PDF](https://arxiv.org/pdf/2508.14499)  

**Abstract**: Shapley values are widely recognized as a principled method for attributing importance to input features in machine learning. However, the exact computation of Shapley values scales exponentially with the number of features, severely limiting the practical application of this powerful approach. The challenge is further compounded when the predictive model is probabilistic - as in Gaussian processes (GPs) - where the outputs are random variables rather than point estimates, necessitating additional computational effort in modeling higher-order moments. In this work, we demonstrate that for an important class of GPs known as FANOVA GP, which explicitly models all main effects and interactions, *exact* Shapley attributions for both local and global explanations can be computed in *quadratic time*. For local, instance-wise explanations, we define a stochastic cooperative game over function components and compute the exact stochastic Shapley value in quadratic time only, capturing both the expected contribution and uncertainty. For global explanations, we introduce a deterministic, variance-based value function and compute exact Shapley values that quantify each feature's contribution to the model's overall sensitivity. Our methods leverage a closed-form (stochastic) Möbius representation of the FANOVA decomposition and introduce recursive algorithms, inspired by Newton's identities, to efficiently compute the mean and variance of Shapley values. Our work enhances the utility of explainable AI, as demonstrated by empirical studies, by providing more scalable, axiomatically sound, and uncertainty-aware explanations for predictions generated by structured probabilistic models. 

---
# Synaptic bundle theory for spike-driven sensor-motor system: More than eight independent synaptic bundles collapse reward-STDP learning 

**Authors**: Takeshi Kobayashi, Shogo Yonekura, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14492)  

**Abstract**: Neuronal spikes directly drive muscles and endow animals with agile movements, but applying the spike-based control signals to actuators in artificial sensor-motor systems inevitably causes a collapse of learning. We developed a system that can vary \emph{the number of independent synaptic bundles} in sensor-to-motor connections. This paper demonstrates the following four findings: (i) Learning collapses once the number of motor neurons or the number of independent synaptic bundles exceeds a critical limit. (ii) The probability of learning failure is increased by a smaller number of motor neurons, while (iii) if learning succeeds, a smaller number of motor neurons leads to faster learning. (iv) The number of weight updates that move in the opposite direction of the optimal weight can quantitatively explain these results. The functions of spikes remain largely unknown. Identifying the parameter range in which learning systems using spikes can be constructed will make it possible to study the functions of spikes that were previously inaccessible due to the difficulty of learning. 

---
# In2x at WMT25 Translation Task 

**Authors**: Lei Pang, Hanyi Mao, Quanjia Xiao, HaiXiao Liu, Xiangyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14472)  

**Abstract**: This paper presents the open-system submission by the In2x research team for the WMT25 General Machine Translation Shared Task. Our submission focuses on Japanese-related translation tasks, aiming to explore a generalizable paradigm for extending large language models (LLMs) to other languages. This paradigm encompasses aspects such as data construction methods and reward model design. The ultimate goal is to enable large language model systems to achieve exceptional performance in low-resource or less commonly spoken languages. 

---
# NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model 

**Authors**: NVIDIA, Aarti Basant, Abhijit Khairnar, Abhijit Paithankar, Abhinav Khattar, Adi Renduchintala, Adithya Renduchintala, Aditya Malte, Akhiad Bercovich, Akshay Hazare, Alejandra Rico, Aleksander Ficek, Alex Kondratenko, Alex Shaposhnikov, Ali Taghibakhshi, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amy Shen, Andrew Tao, Ann Guan, Anna Shors, Anubhav Mandarwal, Arham Mehta, Arun Venkatesan, Ashton Sharabiani, Ashwath Aithal, Ashwin Poojary, Ayush Dattagupta, Balaram Buddharaju, Banghua Zhu, Barnaby Simkin, Bilal Kartal, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Brian Yu, Bryan Catanzaro, Charles Wang, Charlie Truong, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Christian Munley, Christopher Parisien, Dan Su, Daniel Afrimi, Daniel Korzekwa, Daniel Rohrer, Daria Gitman, David Mosallanezhad, Deepak Narayanan, Dima Rekesh, Dina Yared, Dmytro Pykhtar, Dong Ahn, Duncan Riach, Eileen Long, Elliott Ning, Eric Chung, Erick Galinkin, Evelina Bakhturina, Gargi Prasad, Gerald Shen, Haim Elisha, Harsh Sharma, Hayley Ross, Helen Ngo, Herman Sahota, Hexin Wang, Hoo Chang Shin, Hua Huang, Iain Cunningham, Igor Gitman, Ivan Moshkov, Jaehun Jung, Jan Kautz, Jane Polak Scowcroft, Jared Casper, Jimmy Zhang, Jinze Xue, Jocelyn Huang, Joey Conway, John Kamalu, Jonathan Cohen, Joseph Jennings, Julien Veron Vialard, Junkeun Yi, Jupinder Parmar, Kari Briski, Katherine Cheung, Katherine Luna, Keith Wyss, Keshav Santhanam, Kezhi Kong, Krzysztof Pawelec, Kumar Anik, Kunlun Li, Kushan Ahmadian, Lawrence McAfee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14444)  

**Abstract**: We introduce Nemotron-Nano-9B-v2, a hybrid Mamba-Transformer language model designed to increase throughput for reasoning workloads while achieving state-of-the-art accuracy compared to similarly-sized models. Nemotron-Nano-9B-v2 builds on the Nemotron-H architecture, in which the majority of the self-attention layers in the common Transformer architecture are replaced with Mamba-2 layers, to achieve improved inference speed when generating the long thinking traces needed for reasoning. We create Nemotron-Nano-9B-v2 by first pre-training a 12-billion-parameter model (Nemotron-Nano-12B-v2-Base) on 20 trillion tokens using an FP8 training recipe. After aligning Nemotron-Nano-12B-v2-Base, we employ the Minitron strategy to compress and distill the model with the goal of enabling inference on up to 128k tokens on a single NVIDIA A10G GPU (22GiB of memory, bfloat16 precision). Compared to existing similarly-sized models (e.g., Qwen3-8B), we show that Nemotron-Nano-9B-v2 achieves on-par or better accuracy on reasoning benchmarks while achieving up to 6x higher inference throughput in reasoning settings like 8k input and 16k output tokens. We are releasing Nemotron-Nano-9B-v2, Nemotron-Nano12B-v2-Base, and Nemotron-Nano-9B-v2-Base checkpoints along with the majority of our pre- and post-training datasets on Hugging Face. 

---
# Detecting Reading-Induced Confusion Using EEG and Eye Tracking 

**Authors**: Haojun Zhuang, Dünya Baradari, Nataliya Kosmyna, Arnav Balyan, Constanze Albrecht, Stephanie Chen, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2508.14442)  

**Abstract**: Humans regularly navigate an overwhelming amount of information via text media, whether reading articles, browsing social media, or interacting with chatbots. Confusion naturally arises when new information conflicts with or exceeds a reader's comprehension or prior knowledge, posing a challenge for learning. In this study, we present a multimodal investigation of reading-induced confusion using EEG and eye tracking. We collected neural and gaze data from 11 adult participants as they read short paragraphs sampled from diverse, real-world sources. By isolating the N400 event-related potential (ERP), a well-established neural marker of semantic incongruence, and integrating behavioral markers from eye tracking, we provide a detailed analysis of the neural and behavioral correlates of confusion during naturalistic reading. Using machine learning, we show that multimodal (EEG + eye tracking) models improve classification accuracy by 4-22% over unimodal baselines, reaching an average weighted participant accuracy of 77.3% and a best accuracy of 89.6%. Our results highlight the dominance of the brain's temporal regions in these neural signatures of confusion, suggesting avenues for wearable, low-electrode brain-computer interfaces (BCI) for real-time monitoring. These findings lay the foundation for developing adaptive systems that dynamically detect and respond to user confusion, with potential applications in personalized learning, human-computer interaction, and accessibility. 

---
# Cognitive Surgery: The Awakening of Implicit Territorial Awareness in LLMs 

**Authors**: Yinghan Zhou, Weifeng Zhu, Juan Wen, Wanli Peng, Zhengxian Wu, Yiming Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.14408)  

**Abstract**: Large language models (LLMs) have been shown to possess a degree of self-recognition capability-the ability to identify whether a given text was generated by themselves. Prior work has demonstrated that this capability is reliably expressed under the Pair Presentation Paradigm (PPP), where the model is presented with two texts and asked to choose which one it authored. However, performance deteriorates sharply under the Individual Presentation Paradigm (IPP), where the model is given a single text to judge authorship. Although this phenomenon has been observed, its underlying causes have not been systematically analyzed. In this paper, we first replicate existing findings to confirm that LLMs struggle to distinguish self- from other-generated text under IPP. We then investigate the reasons for this failure and attribute it to a phenomenon we term Implicit Territorial Awareness (ITA)-the model's latent ability to distinguish self- and other-texts in representational space, which remains unexpressed in its output behavior. To awaken the ITA of LLMs, we propose Cognitive Surgery (CoSur), a novel framework comprising four main modules: representation extraction, territory construction, authorship discrimination and cognitive editing. Experimental results demonstrate that our proposed method improves the performance of three different LLMs in the IPP scenario, achieving average accuracies of 83.25%, 66.19%, and 88.01%, respectively. 

---
# NoteIt: A System Converting Instructional Videos to Interactable Notes Through Multimodal Video Understanding 

**Authors**: Running Zhao, Zhihan Jiang, Xinchen Zhang, Chirui Chang, Handi Chen, Weipeng Deng, Luyao Jin, Xiaojuan Qi, Xun Qian, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2508.14395)  

**Abstract**: Users often take notes for instructional videos to access key knowledge later without revisiting long videos. Automated note generation tools enable users to obtain informative notes efficiently. However, notes generated by existing research or off-the-shelf tools fail to preserve the information conveyed in the original videos comprehensively, nor can they satisfy users' expectations for diverse presentation formats and interactive features when using notes digitally. In this work, we present NoteIt, a system, which automatically converts instructional videos to interactable notes using a novel pipeline that faithfully extracts hierarchical structure and multimodal key information from videos. With NoteIt's interface, users can interact with the system to further customize the content and presentation formats of the notes according to their preferences. We conducted both a technical evaluation and a comparison user study (N=36). The solid performance in objective metrics and the positive user feedback demonstrated the effectiveness of the pipeline and the overall usability of NoteIt. Project website: this https URL 

---
# DEPTH: Hallucination-Free Relation Extraction via Dependency-Aware Sentence Simplification and Two-tiered Hierarchical Refinement 

**Authors**: Yupei Yang, Fan Feng, Lin Yang, Wanxi Deng, Lin Qu, Biwei Huang, Shikui Tu, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14391)  

**Abstract**: Relation extraction enables the construction of structured knowledge for many downstream applications. While large language models (LLMs) have shown great promise in this domain, most existing methods concentrate on relation classification, which predicts the semantic relation type between a related entity pair. However, we observe that LLMs often struggle to reliably determine whether a relation exists, especially in cases involving complex sentence structures or intricate semantics, which leads to spurious predictions. Such hallucinations can introduce noisy edges in knowledge graphs, compromising the integrity of structured knowledge and downstream reliability. To address these challenges, we propose DEPTH, a framework that integrates Dependency-aware sEntence simPlification and Two-tiered Hierarchical refinement into the relation extraction pipeline. Given a sentence and its candidate entity pairs, DEPTH operates in two stages: (1) the Grounding module extracts relations for each pair by leveraging their shortest dependency path, distilling the sentence into a minimal yet coherent relational context that reduces syntactic noise while preserving key semantics; (2) the Refinement module aggregates all local predictions and revises them based on a holistic understanding of the sentence, correcting omissions and inconsistencies. We further introduce a causality-driven reward model that mitigates reward hacking by disentangling spurious correlations, enabling robust fine-tuning via reinforcement learning with human feedback. Experiments on six benchmarks demonstrate that DEPTH reduces the average hallucination rate to 7.0\% while achieving a 17.2\% improvement in average F1 score over state-of-the-art baselines. 

---
# Credence Calibration Game? Calibrating Large Language Models through Structured Play 

**Authors**: Ke Fang, Tianyi Zhao, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.14390)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in decision-critical domains, it becomes essential to ensure that their confidence estimates faithfully correspond to their actual correctness. Existing calibration methods have primarily focused on post-hoc adjustments or auxiliary model training; however, many of these approaches necessitate additional supervision or parameter updates. In this work, we propose a novel prompt-based calibration framework inspired by the Credence Calibration Game. Our method establishes a structured interaction loop wherein LLMs receive feedback based on the alignment of their predicted confidence with correctness. Through feedback-driven prompting and natural language summaries of prior performance, our framework dynamically improves model calibration. Extensive experiments across models and game configurations demonstrate consistent improvements in evaluation metrics. Our results highlight the potential of game-based prompting as an effective strategy for LLM calibration. Code and data are available at this https URL. 

---
# Online Incident Response Planning under Model Misspecification through Bayesian Learning and Belief Quantization 

**Authors**: Kim Hammar, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14385)  

**Abstract**: Effective responses to cyberattacks require fast decisions, even when information about the attack is incomplete or inaccurate. However, most decision-support frameworks for incident response rely on a detailed system model that describes the incident, which restricts their practical utility. In this paper, we address this limitation and present an online method for incident response planning under model misspecification, which we call MOBAL: Misspecified Online Bayesian Learning. MOBAL iteratively refines a conjecture about the model through Bayesian learning as new information becomes available, which facilitates model adaptation as the incident unfolds. To determine effective responses online, we quantize the conjectured model into a finite Markov model, which enables efficient response planning through dynamic programming. We prove that Bayesian learning is asymptotically consistent with respect to the information feedback. Additionally, we establish bounds on misspecification and quantization errors. Experiments on the CAGE-2 benchmark show that MOBAL outperforms the state of the art in terms of adaptability and robustness to model misspecification. 

---
# ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities 

**Authors**: Wenhan Dong, Zhen Sun, Yuemeng Zhao, Zifan Peng, Jun Wu, Jingyi Zheng, Yule Liu, Xinlei He, Yu Wang, Ruiming Wang, Xinyi Huang, Lei Mo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14377)  

**Abstract**: Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications. 

---
# Computing-In-Memory Dataflow for Minimal Buffer Traffic 

**Authors**: Choongseok Song, Doo Seok Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2508.14375)  

**Abstract**: Computing-In-Memory (CIM) offers a potential solution to the memory wall issue and can achieve high energy efficiency by minimizing data movement, making it a promising architecture for edge AI devices. Lightweight models like MobileNet and EfficientNet, which utilize depthwise convolution for feature extraction, have been developed for these devices. However, CIM macros often face challenges in accelerating depthwise convolution, including underutilization of CIM memory and heavy buffer traffic. The latter, in particular, has been overlooked despite its significant impact on latency and energy consumption. To address this, we introduce a novel CIM dataflow that significantly reduces buffer traffic by maximizing data reuse and improving memory utilization during depthwise convolution. The proposed dataflow is grounded in solid theoretical principles, fully demonstrated in this paper. When applied to MobileNet and EfficientNet models, our dataflow reduces buffer traffic by 77.4-87.0%, leading to a total reduction in data traffic energy and latency by 10.1-17.9% and 15.6-27.8%, respectively, compared to the baseline (conventional weight-stationary dataflow). 

---
# Learning Point Cloud Representations with Pose Continuity for Depth-Based Category-Level 6D Object Pose Estimation 

**Authors**: Zhujun Li, Shuo Zhang, Ioannis Stamos  

**Link**: [PDF](https://arxiv.org/pdf/2508.14358)  

**Abstract**: Category-level object pose estimation aims to predict the 6D pose and 3D size of objects within given categories. Existing approaches for this task rely solely on 6D poses as supervisory signals without explicitly capturing the intrinsic continuity of poses, leading to inconsistencies in predictions and reduced generalization to unseen poses. To address this limitation, we propose HRC-Pose, a novel depth-only framework for category-level object pose estimation, which leverages contrastive learning to learn point cloud representations that preserve the continuity of 6D poses. HRC-Pose decouples object pose into rotation and translation components, which are separately encoded and leveraged throughout the network. Specifically, we introduce a contrastive learning strategy for multi-task, multi-category scenarios based on our 6D pose-aware hierarchical ranking scheme, which contrasts point clouds from multiple categories by considering rotational and translational differences as well as categorical information. We further design pose estimation modules that separately process the learned rotation-aware and translation-aware embeddings. Our experiments demonstrate that HRC-Pose successfully learns continuous feature spaces. Results on REAL275 and CAMERA25 benchmarks show that our method consistently outperforms existing depth-only state-of-the-art methods and runs in real-time, demonstrating its effectiveness and potential for real-world applications. Our code is at this https URL. 

---
# Organ-Agents: Virtual Human Physiology Simulator via LLMs 

**Authors**: Rihao Chang, He Jiao, Weizhi Nie, Honglin Guo, Keliang Xie, Zhenhua Wu, Lina Zhao, Yunpeng Bai, Yongtao Ma, Lanjun Wang, Yuting Su, Xi Gao, Weijie Wang, Nicu Sebe, Bruno Lepri, Bingwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14357)  

**Abstract**: Recent advances in large language models (LLMs) have enabled new possibilities in simulating complex physiological systems. We introduce Organ-Agents, a multi-agent framework that simulates human physiology via LLM-driven agents. Each Simulator models a specific system (e.g., cardiovascular, renal, immune). Training consists of supervised fine-tuning on system-specific time-series data, followed by reinforcement-guided coordination using dynamic reference selection and error correction. We curated data from 7,134 sepsis patients and 7,895 controls, generating high-resolution trajectories across 9 systems and 125 variables. Organ-Agents achieved high simulation accuracy on 4,509 held-out patients, with per-system MSEs <0.16 and robustness across SOFA-based severity strata. External validation on 22,689 ICU patients from two hospitals showed moderate degradation under distribution shifts with stable simulation. Organ-Agents faithfully reproduces critical multi-system events (e.g., hypotension, hyperlactatemia, hypoxemia) with coherent timing and phase progression. Evaluation by 15 critical care physicians confirmed realism and physiological plausibility (mean Likert ratings 3.9 and 3.7). Organ-Agents also enables counterfactual simulations under alternative sepsis treatment strategies, generating trajectories and APACHE II scores aligned with matched real-world patients. In downstream early warning tasks, classifiers trained on synthetic data showed minimal AUROC drops (<0.04), indicating preserved decision-relevant patterns. These results position Organ-Agents as a credible, interpretable, and generalizable digital twin for precision diagnosis, treatment simulation, and hypothesis testing in critical care. 

---
# Inter-Class Relational Loss for Small Object Detection: A Case Study on License Plates 

**Authors**: Dian Ning, Dong Seog Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.14343)  

**Abstract**: In one-stage multi-object detection tasks, various intersection over union (IoU)-based solutions aim at smooth and stable convergence near the targets during training. However, IoU-based losses fail to correctly update the gradient of small objects due to an extremely flat gradient. During the update of multiple objects, the learning of small objects' gradients suffers more because of insufficient gradient updates. Therefore, we propose an inter-class relational loss to efficiently update the gradient of small objects while not sacrificing the learning efficiency of other objects based on the simple fact that an object has a spatial relationship to another object (e.g., a car plate is attached to a car in a similar position). When the predicted car plate's bounding box is not within its car, a loss punishment is added to guide the learning, which is inversely proportional to the overlapped area of the car's and predicted car plate's bounding box. By leveraging the spatial relationship at the inter-class level, the loss guides small object predictions using larger objects and enhances latent information in deeper feature maps. In this paper, we present twofold contributions using license plate detection as a case study: (1) a new small vehicle multi-license plate dataset (SVMLP), featuring diverse real-world scenarios with high-quality annotations; and (2) a novel inter-class relational loss function designed to promote effective detection performance. We highlight the proposed ICR loss penalty can be easily added to existing IoU-based losses and enhance the performance. These contributions improve the standard mean Average Precision (mAP) metric, achieving gains of 10.3% and 1.6% in mAP$^{\text{test}}_{50}$ for YOLOv12-T and UAV-DETR, respectively, without any additional hyperparameter tuning. Code and dataset will be available soon. 

---
# Generative AI Against Poaching: Latent Composite Flow Matching for Wildlife Conservation 

**Authors**: Lingkai Kong, Haichuan Wang, Charles A. Emogor, Vincent Börsch-Supan, Lily Xu, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2508.14342)  

**Abstract**: Poaching poses significant threats to wildlife and biodiversity. A valuable step in reducing poaching is to forecast poacher behavior, which can inform patrol planning and other conservation interventions. Existing poaching prediction methods based on linear models or decision trees lack the expressivity to capture complex, nonlinear spatiotemporal patterns. Recent advances in generative modeling, particularly flow matching, offer a more flexible alternative. However, training such models on real-world poaching data faces two central obstacles: imperfect detection of poaching events and limited data. To address imperfect detection, we integrate flow matching with an occupancy-based detection model and train the flow in latent space to infer the underlying occupancy state. To mitigate data scarcity, we adopt a composite flow initialized from a linear-model prediction rather than random noise which is the standard in diffusion models, injecting prior knowledge and improving generalization. Evaluations on datasets from two national parks in Uganda show consistent gains in predictive accuracy. 

---
# A Comparative Evaluation of Teacher-Guided Reinforcement Learning Techniques for Autonomous Cyber Operations 

**Authors**: Konur Tholl, Mariam El Mezouar, Ranwa Al Mallah  

**Link**: [PDF](https://arxiv.org/pdf/2508.14340)  

**Abstract**: Autonomous Cyber Operations (ACO) rely on Reinforcement Learning (RL) to train agents to make effective decisions in the cybersecurity domain. However, existing ACO applications require agents to learn from scratch, leading to slow convergence and poor early-stage performance. While teacher-guided techniques have demonstrated promise in other domains, they have not yet been applied to ACO. In this study, we implement four distinct teacher-guided techniques in the simulated CybORG environment and conduct a comparative evaluation. Our results demonstrate that teacher integration can significantly improve training efficiency in terms of early policy performance and convergence speed, highlighting its potential benefits for autonomous cybersecurity. 

---
# Power Stabilization for AI Training Datacenters 

**Authors**: Esha Choukse, Brijesh Warrier, Scot Heath, Luz Belmont, April Zhao, Hassan Ali Khan, Brian Harry, Matthew Kappel, Russell J. Hewett, Kushal Datta, Yu Pei, Caroline Lichtenberger, John Siegler, David Lukofsky, Zaid Kahn, Gurpreet Sahota, Andy Sullivan, Charles Frederick, Hien Thai, Rebecca Naughton, Daniel Jurnove, Justin Harp, Reid Carper, Nithish Mahalingam, Srini Varkala, Alok Gautam Kumbhare, Satyajit Desai, Venkatesh Ramamurthy, Praneeth Gottumukkala, Girish Bhatia, Kelsey Wildstone, Laurentiu Olariu, Mohammed Ayna, Mike Kendrick, Ricardo Bianchini, Aaron Hurst, Reza Zamani, Xin Li, Gene Oden, Rory Carmichael, Tom Li, Apoorv Gupta, Nilesh Dattani, Lawrence Marwong, Rob Nertney, Jeff Liott, Miro Enev, Divya Ramakrishnan, Ian Buck, Jonah Alben  

**Link**: [PDF](https://arxiv.org/pdf/2508.14318)  

**Abstract**: Large Artificial Intelligence (AI) training workloads spanning several tens of thousands of GPUs present unique power management challenges. These arise due to the high variability in power consumption during the training. Given the synchronous nature of these jobs, during every iteration there is a computation-heavy phase, where each GPU works on the local data, and a communication-heavy phase where all the GPUs synchronize on the data. Because compute-heavy phases require much more power than communication phases, large power swings occur. The amplitude of these power swings is ever increasing with the increase in the size of training jobs. An even bigger challenge arises from the frequency spectrum of these power swings which, if harmonized with critical frequencies of utilities, can cause physical damage to the power grid infrastructure. Therefore, to continue scaling AI training workloads safely, we need to stabilize the power of such workloads. This paper introduces the challenge with production data and explores innovative solutions across the stack: software, GPU hardware, and datacenter infrastructure. We present the pros and cons of each of these approaches and finally present a multi-pronged approach to solving the challenge. The proposed solutions are rigorously tested using a combination of real hardware and Microsoft's in-house cloud power simulator, providing critical insights into the efficacy of these interventions under real-world conditions. 

---
# Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency 

**Authors**: Aman Goel, Daniel Schwartz, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14314)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages FINe-grained Cross-model consistency to detect and mitigate Hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves 7-8 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation across multiple models demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems. 

---
# Your Reward Function for RL is Your Best PRM for Search: Unifying RL and Search-Based TTS 

**Authors**: Can Jin, Yang Zhou, Qixin Zhang, Hongwu Peng, Di Zhang, Marco Pavone, Ligong Han, Zhang-Wei Hong, Tong Che, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2508.14313)  

**Abstract**: Test-time scaling (TTS) for large language models (LLMs) has thus far fallen into two largely separate paradigms: (1) reinforcement learning (RL) methods that optimize sparse outcome-based rewards, yet suffer from instability and low sample efficiency; and (2) search-based techniques guided by independently trained, static process reward models (PRMs), which require expensive human- or LLM-generated labels and often degrade under distribution shifts. In this paper, we introduce AIRL-S, the first natural unification of RL-based and search-based TTS. Central to AIRL-S is the insight that the reward function learned during RL training inherently represents the ideal PRM for guiding downstream search. Specifically, we leverage adversarial inverse reinforcement learning (AIRL) combined with group relative policy optimization (GRPO) to learn a dense, dynamic PRM directly from correct reasoning traces, entirely eliminating the need for labeled intermediate process data. At inference, the resulting PRM simultaneously serves as the critic for RL rollouts and as a heuristic to effectively guide search procedures, facilitating robust reasoning chain extension, mitigating reward hacking, and enhancing cross-task generalization. Experimental results across eight benchmarks, including mathematics, scientific reasoning, and code generation, demonstrate that our unified approach improves performance by 9 % on average over the base model, matching GPT-4o. Furthermore, when integrated into multiple search algorithms, our PRM consistently outperforms all baseline PRMs trained with labeled data. These results underscore that, indeed, your reward function for RL is your best PRM for search, providing a robust and cost-effective solution to complex reasoning tasks in LLMs. 

---
# Learning Time-Varying Convexifications of Multiple Fairness Measures 

**Authors**: Quan Zhou, Jakub Marecek, Robert Shorten  

**Link**: [PDF](https://arxiv.org/pdf/2508.14311)  

**Abstract**: There is an increasing appreciation that one may need to consider multiple measures of fairness, e.g., considering multiple group and individual fairness notions. The relative weights of the fairness regularisers are a priori unknown, may be time varying, and need to be learned on the fly. We consider the learning of time-varying convexifications of multiple fairness measures with limited graph-structured feedback. 

---
# GLASS: Test-Time Acceleration for LLMs via Global-Local Neural Importance Aggregation 

**Authors**: Amirmohsen Sattarifard, Sepehr Lavasani, Ehsan Imani, Kunlin Zhang, Hanlin Xu, Fengyu Sun, Negar Hassanpour, Chao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14302)  

**Abstract**: Deploying Large Language Models (LLMs) on edge hardware demands aggressive, prompt-aware dynamic pruning to reduce computation without degrading quality. Static or predictor-based schemes either lock in a single sparsity pattern or incur extra runtime overhead, and recent zero-shot methods that rely on statistics from a single prompt fail on short prompt and/or long generation scenarios. We introduce A/I-GLASS: Activation- and Impact-based Global-Local neural importance Aggregation for feed-forward network SparSification, two training-free methods that dynamically select FFN units using a rank-aggregation of prompt local and model-intrinsic global neuron statistics. Empirical results across multiple LLMs and benchmarks demonstrate that GLASS significantly outperforms prior training-free methods, particularly in challenging long-form generation scenarios, without relying on auxiliary predictors or adding any inference overhead. 

---
# Pixels to Play: A Foundation Model for 3D Gameplay 

**Authors**: Yuguang Yue, Chris Green, Samuel Hunt, Irakli Salia, Wenzhe Shi, Jonathan J Hunt  

**Link**: [PDF](https://arxiv.org/pdf/2508.14295)  

**Abstract**: We introduce Pixels2Play-0.1 (P2P0.1), a foundation model that learns to play a wide range of 3D video games with recognizable human-like behavior. Motivated by emerging consumer and developer use cases - AI teammates, controllable NPCs, personalized live-streamers, assistive testers - we argue that an agent must rely on the same pixel stream available to players and generalize to new titles with minimal game-specific engineering. P2P0.1 is trained end-to-end with behavior cloning: labeled demonstrations collected from instrumented human game-play are complemented by unlabeled public videos, to which we impute actions via an inverse-dynamics model. A decoder-only transformer with auto-regressive action output handles the large action space while remaining latency-friendly on a single consumer GPU. We report qualitative results showing competent play across simple Roblox and classic MS-DOS titles, ablations on unlabeled data, and outline the scaling and evaluation steps required to reach expert-level, text-conditioned control. 

---
# OccluNet: Spatio-Temporal Deep Learning for Occlusion Detection on DSA 

**Authors**: Anushka A. Kore, Frank G. te Nijenhuis, Matthijs van der Sluijs, Wim van Zwam, Charles Majoie, Geert Lycklama à Nijeholt, Danny Ruijters, Frans Vos, Sandra Cornelissen, Ruisheng Su, Theo van Walsum  

**Link**: [PDF](https://arxiv.org/pdf/2508.14286)  

**Abstract**: Accurate detection of vascular occlusions during endovascular thrombectomy (EVT) is critical in acute ischemic stroke (AIS). Interpretation of digital subtraction angiography (DSA) sequences poses challenges due to anatomical complexity and time constraints. This work proposes OccluNet, a spatio-temporal deep learning model that integrates YOLOX, a single-stage object detector, with transformer-based temporal attention mechanisms to automate occlusion detection in DSA sequences. We compared OccluNet with a YOLOv11 baseline trained on either individual DSA frames or minimum intensity projections. Two spatio-temporal variants were explored for OccluNet: pure temporal attention and divided space-time attention. Evaluation on DSA images from the MR CLEAN Registry revealed the model's capability to capture temporally consistent features, achieving precision and recall of 89.02% and 74.87%, respectively. OccluNet significantly outperformed the baseline models, and both attention variants attained similar performance. Source code is available at this https URL 

---
# Amortized Bayesian Meta-Learning for Low-Rank Adaptation of Large Language Models 

**Authors**: Liyi Zhang, Jake Snell, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2508.14285)  

**Abstract**: Fine-tuning large language models (LLMs) with low-rank adaptaion (LoRA) is a cost-effective way to incorporate information from a specific dataset. However, it is often unclear how well the fine-tuned LLM will generalize, i.e., how well it will perform on unseen datasets. Methods have been proposed to improve generalization by optimizing with in-context prompts, or by using meta-learning to fine-tune LLMs. However, these methods are expensive in memory and computation, requiring either long-context prompts or saving copies of parameters and using second-order gradient updates. To address these challenges, we propose Amortized Bayesian Meta-Learning for LoRA (ABMLL). This method builds on amortized Bayesian meta-learning for smaller models, adapting this approach to LLMs while maintaining its computational efficiency. We reframe task-specific and global parameters in the context of LoRA and use a set of new hyperparameters to balance reconstruction accuracy and the fidelity of task-specific parameters to the global ones. ABMLL provides effective generalization and scales to large models such as Llama3-8B. Furthermore, as a result of using a Bayesian framework, ABMLL provides improved uncertainty quantification. We test ABMLL on Unified-QA and CrossFit datasets and find that it outperforms existing methods on these benchmarks in terms of both accuracy and expected calibration error. 

---
# Tooth-Diffusion: Guided 3D CBCT Synthesis with Fine-Grained Tooth Conditioning 

**Authors**: Said Djafar Said, Torkan Gholamalizadeh, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14276)  

**Abstract**: Despite the growing importance of dental CBCT scans for diagnosis and treatment planning, generating anatomically realistic scans with fine-grained control remains a challenge in medical image synthesis. In this work, we propose a novel conditional diffusion framework for 3D dental volume generation, guided by tooth-level binary attributes that allow precise control over tooth presence and configuration. Our approach integrates wavelet-based denoising diffusion, FiLM conditioning, and masked loss functions to focus learning on relevant anatomical structures. We evaluate the model across diverse tasks, such as tooth addition, removal, and full dentition synthesis, using both paired and distributional similarity metrics. Results show strong fidelity and generalization with low FID scores, robust inpainting performance, and SSIM values above 0.91 even on unseen scans. By enabling realistic, localized modification of dentition without rescanning, this work opens opportunities for surgical planning, patient communication, and targeted data augmentation in dental AI workflows. The codes are available at: this https URL. 

---
# Disentangling concept semantics via multilingual averaging in Sparse Autoencoders 

**Authors**: Cliff O'Reilly, Ernesto Jimenez-Ruiz, Tillman Weyde  

**Link**: [PDF](https://arxiv.org/pdf/2508.14275)  

**Abstract**: Connecting LLMs with formal knowledge representation and reasoning is a promising approach to address their shortcomings. Embeddings and sparse autoencoders are widely used to represent textual content, but the semantics are entangled with syntactic and language-specific information. We propose a method that isolates concept semantics in Large Langue Models by averaging concept activations derived via Sparse Autoencoders. We create English text representations from OWL ontology classes, translate the English into French and Chinese and then pass these texts as prompts to the Gemma 2B LLM. Using the open source Gemma Scope suite of Sparse Autoencoders, we obtain concept activations for each class and language version. We average the different language activations to derive a conceptual average. We then correlate the conceptual averages with a ground truth mapping between ontology classes. Our results give a strong indication that the conceptual average aligns to the true relationship between classes when compared with a single language by itself. The result hints at a new technique which enables mechanistic interpretation of internal network states with higher accuracy. 

---
# Effect of Data Augmentation on Conformal Prediction for Diabetic Retinopathy 

**Authors**: Rizwan Ahamed, Annahita Amireskandari, Joel Palko, Carol Laxson, Binod Bhattarai, Prashnna Gyawali  

**Link**: [PDF](https://arxiv.org/pdf/2508.14266)  

**Abstract**: The clinical deployment of deep learning models for high-stakes tasks such as diabetic retinopathy (DR) grading requires demonstrable reliability. While models achieve high accuracy, their clinical utility is limited by a lack of robust uncertainty quantification. Conformal prediction (CP) offers a distribution-free framework to generate prediction sets with statistical guarantees of coverage. However, the interaction between standard training practices like data augmentation and the validity of these guarantees is not well understood. In this study, we systematically investigate how different data augmentation strategies affect the performance of conformal predictors for DR grading. Using the DDR dataset, we evaluate two backbone architectures -- ResNet-50 and a Co-Scale Conv-Attentional Transformer (CoaT) -- trained under five augmentation regimes: no augmentation, standard geometric transforms, CLAHE, Mixup, and CutMix. We analyze the downstream effects on conformal metrics, including empirical coverage, average prediction set size, and correct efficiency. Our results demonstrate that sample-mixing strategies like Mixup and CutMix not only improve predictive accuracy but also yield more reliable and efficient uncertainty estimates. Conversely, methods like CLAHE can negatively impact model certainty. These findings highlight the need to co-design augmentation strategies with downstream uncertainty quantification in mind to build genuinely trustworthy AI systems for medical imaging. 

---
# Incident Analysis for AI Agents 

**Authors**: Carson Ezell, Xavier Roberts-Gaal, Alan Chan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14231)  

**Abstract**: As AI agents become more widely deployed, we are likely to see an increasing number of incidents: events involving AI agent use that directly or indirectly cause harm. For example, agents could be prompt-injected to exfiltrate private information or make unauthorized purchases. Structured information about such incidents (e.g., user prompts) can help us understand their causes and prevent future occurrences. However, existing incident reporting processes are not sufficient for understanding agent incidents. In particular, such processes are largely based on publicly available data, which excludes useful, but potentially sensitive, information such as an agent's chain of thought or browser history. To inform the development of new, emerging incident reporting processes, we propose an incident analysis framework for agents. Drawing on systems safety approaches, our framework proposes three types of factors that can cause incidents: system-related (e.g., CBRN training data), contextual (e.g., prompt injections), and cognitive (e.g., misunderstanding a user request). We also identify specific information that could help clarify which factors are relevant to a given incident: activity logs, system documentation and access, and information about the tools an agent uses. We provide recommendations for 1) what information incident reports should include and 2) what information developers and deployers should retain and make available to incident investigators upon request. As we transition to a world with more agents, understanding agent incidents will become increasingly crucial for managing risks. 

---
# New Insights into Automatic Treatment Planning for Cancer Radiotherapy Using Explainable Artificial Intelligence 

**Authors**: Md Mainul Abrar, Xun Jia, Yujie Chi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14229)  

**Abstract**: Objective: This study aims to uncover the opaque decision-making process of an artificial intelligence (AI) agent for automatic treatment planning.
Approach: We examined a previously developed AI agent based on the Actor-Critic with Experience Replay (ACER) network, which automatically tunes treatment planning parameters (TPPs) for inverse planning in prostate cancer intensity modulated radiotherapy. We selected multiple checkpoint ACER agents from different stages of training and applied an explainable AI (EXAI) method to analyze the attribution from dose-volume histogram (DVH) inputs to TPP-tuning decisions. We then assessed each agent's planning efficacy and efficiency and evaluated their policy and final TPP tuning spaces. Combining these analyses, we systematically examined how ACER agents generated high-quality treatment plans in response to different DVH inputs.
Results: Attribution analysis revealed that ACER agents progressively learned to identify dose-violation regions from DVH inputs and promote appropriate TPP-tuning actions to mitigate them. Organ-wise similarities between DVH attributions and dose-violation reductions ranged from 0.25 to 0.5 across tested agents. Agents with stronger attribution-violation similarity required fewer tuning steps (~12-13 vs. 22), exhibited a more concentrated TPP-tuning space with lower entropy (~0.3 vs. 0.6), converged on adjusting only a few TPPs, and showed smaller discrepancies between practical and theoretical tuning steps. Putting together, these findings indicate that high-performing ACER agents can effectively identify dose violations from DVH inputs and employ a global tuning strategy to achieve high-quality treatment planning, much like skilled human planners.
Significance: Better interpretability of the agent's decision-making process may enhance clinician trust and inspire new strategies for automatic treatment planning. 

---
# A Survey on Video Anomaly Detection via Deep Learning: Human, Vehicle, and Environment 

**Authors**: Ghazal Alinezhad Noghre, Armin Danesh Pazho, Hamed Tabkhi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14203)  

**Abstract**: Video Anomaly Detection (VAD) has emerged as a pivotal task in computer vision, with broad relevance across multiple fields. Recent advances in deep learning have driven significant progress in this area, yet the field remains fragmented across domains and learning paradigms. This survey offers a comprehensive perspective on VAD, systematically organizing the literature across various supervision levels, as well as adaptive learning methods such as online, active, and continual learning. We examine the state of VAD across three major application categories: human-centric, vehicle-centric, and environment-centric scenarios, each with distinct challenges and design considerations. In doing so, we identify fundamental contributions and limitations of current methodologies. By consolidating insights from subfields, we aim to provide the community with a structured foundation for advancing both theoretical understanding and real-world applicability of VAD systems. This survey aims to support researchers by providing a useful reference, while also drawing attention to the broader set of open challenges in anomaly detection, including both fundamental research questions and practical obstacles to real-world deployment. 

---
# RynnEC: Bringing MLLMs into Embodied World 

**Authors**: Ronghao Dang, Yuqian Yuan, Yunxuan Mao, Kehan Li, Jiangpin Liu, Zhikai Wang, Xin Li, Fan Wang, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14160)  

**Abstract**: We introduce RynnEC, a video multimodal large language model designed for embodied cognition. Built upon a general-purpose vision-language foundation model, RynnEC incorporates a region encoder and a mask decoder, enabling flexible region-level video interaction. Despite its compact architecture, RynnEC achieves state-of-the-art performance in object property understanding, object segmentation, and spatial reasoning. Conceptually, it offers a region-centric video paradigm for the brain of embodied agents, providing fine-grained perception of the physical world and enabling more precise interactions. To mitigate the scarcity of annotated 3D datasets, we propose an egocentric video based pipeline for generating embodied cognition data. Furthermore, we introduce RynnEC-Bench, a region-centered benchmark for evaluating embodied cognitive capabilities. We anticipate that RynnEC will advance the development of general-purpose cognitive cores for embodied agents and facilitate generalization across diverse embodied tasks. The code, model checkpoints, and benchmark are available at: this https URL 

---
# LENS: Learning to Segment Anything with Unified Reinforced Reasoning 

**Authors**: Lianghui Zhu, Bin Ouyang, Yuxuan Zhang, Tianheng Cheng, Rui Hu, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Li Yu, Wenyu Liu, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14153)  

**Abstract**: Text-prompted image segmentation enables fine-grained visual understanding and is critical for applications such as human-computer interaction and robotics. However, existing supervised fine-tuning methods typically ignore explicit chain-of-thought (CoT) reasoning at test time, which limits their ability to generalize to unseen prompts and domains. To address this issue, we introduce LENS, a scalable reinforcement-learning framework that jointly optimizes the reasoning process and segmentation in an end-to-end manner. We propose unified reinforcement-learning rewards that span sentence-, box-, and segment-level cues, encouraging the model to generate informative CoT rationales while refining mask quality. Using a publicly available 3-billion-parameter vision-language model, i.e., Qwen2.5-VL-3B-Instruct, LENS achieves an average cIoU of 81.2% on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks, outperforming the strong fine-tuned method, i.e., GLaMM, by up to 5.6%. These results demonstrate that RL-driven CoT reasoning serves as a robust prior for text-prompted segmentation and offers a practical path toward more generalizable Segment Anything models. Code is available at this https URL. 

---
# A Systematic Study of Deep Learning Models and xAI Methods for Region-of-Interest Detection in MRI Scans 

**Authors**: Justin Yiu, Kushank Arora, Daniel Steinberg, Rohit Ghiya  

**Link**: [PDF](https://arxiv.org/pdf/2508.14151)  

**Abstract**: Magnetic Resonance Imaging (MRI) is an essential diagnostic tool for assessing knee injuries. However, manual interpretation of MRI slices remains time-consuming and prone to inter-observer variability. This study presents a systematic evaluation of various deep learning architectures combined with explainable AI (xAI) techniques for automated region of interest (ROI) detection in knee MRI scans. We investigate both supervised and self-supervised approaches, including ResNet50, InceptionV3, Vision Transformers (ViT), and multiple U-Net variants augmented with multi-layer perceptron (MLP) classifiers. To enhance interpretability and clinical relevance, we integrate xAI methods such as Grad-CAM and Saliency Maps. Model performance is assessed using AUC for classification and PSNR/SSIM for reconstruction quality, along with qualitative ROI visualizations. Our results demonstrate that ResNet50 consistently excels in classification and ROI identification, outperforming transformer-based models under the constraints of the MRNet dataset. While hybrid U-Net + MLP approaches show potential for leveraging spatial features in reconstruction and interpretability, their classification performance remains lower. Grad-CAM consistently provided the most clinically meaningful explanations across architectures. Overall, CNN-based transfer learning emerges as the most effective approach for this dataset, while future work with larger-scale pretraining may better unlock the potential of transformer models. 

---
# Neuro-inspired Ensemble-to-Ensemble Communication Primitives for Sparse and Efficient ANNs 

**Authors**: Orestis Konstantaropoulos, Stelios Manolis Smirnakis, Maria Papadopouli  

**Link**: [PDF](https://arxiv.org/pdf/2508.14140)  

**Abstract**: The structure of biological neural circuits-modular, hierarchical, and sparsely interconnected-reflects an efficient trade-off between wiring cost, functional specialization, and robustness. These principles offer valuable insights for artificial neural network (ANN) design, especially as networks grow in depth and scale. Sparsity, in particular, has been widely explored for reducing memory and computation, improving speed, and enhancing generalization. Motivated by systems neuroscience findings, we explore how patterns of functional connectivity in the mouse visual cortex-specifically, ensemble-to-ensemble communication, can inform ANN design. We introduce G2GNet, a novel architecture that imposes sparse, modular connectivity across feedforward layers. Despite having significantly fewer parameters than fully connected models, G2GNet achieves superior accuracy on standard vision benchmarks. To our knowledge, this is the first architecture to incorporate biologically observed functional connectivity patterns as a structural bias in ANN design. We complement this static bias with a dynamic sparse training (DST) mechanism that prunes and regrows edges during training. We also propose a Hebbian-inspired rewiring rule based on activation correlations, drawing on principles of biological plasticity. G2GNet achieves up to 75% sparsity while improving accuracy by up to 4.3% on benchmarks, including Fashion-MNIST, CIFAR-10, and CIFAR-100, outperforming dense baselines with far fewer computations. 

---
# The Statistical Validation of Innovation Lens 

**Authors**: Giacomo Radaelli, Jonah Lynch  

**Link**: [PDF](https://arxiv.org/pdf/2508.14139)  

**Abstract**: Information overload and the rapid pace of scientific advancement make it increasingly difficult to evaluate and allocate resources to new research proposals. Is there a structure to scientific discovery that could inform such decisions? We present statistical evidence for such structure, by training a classifier that successfully predicts high-citation research papers between 2010-2024 in the Computer Science, Physics, and PubMed domains. 

---
# STAS: Spatio-Temporal Adaptive Computation Time for Spiking Transformers 

**Authors**: Donghwa Kang, Doohyun Kim, Sang-Ki Ko, Jinkyu Lee, Brent ByungHoon Kang, Hyeongboo Baek  

**Link**: [PDF](https://arxiv.org/pdf/2508.14138)  

**Abstract**: Spiking neural networks (SNNs) offer energy efficiency over artificial neural networks (ANNs) but suffer from high latency and computational overhead due to their multi-timestep operational nature. While various dynamic computation methods have been developed to mitigate this by targeting spatial, temporal, or architecture-specific redundancies, they remain fragmented. While the principles of adaptive computation time (ACT) offer a robust foundation for a unified approach, its application to SNN-based vision Transformers (ViTs) is hindered by two core issues: the violation of its temporal similarity prerequisite and a static architecture fundamentally unsuited for its principles. To address these challenges, we propose STAS (Spatio-Temporal Adaptive computation time for Spiking transformers), a framework that co-designs the static architecture and dynamic computation policy. STAS introduces an integrated spike patch splitting (I-SPS) module to establish temporal stability by creating a unified input representation, thereby solving the architectural problem of temporal dissimilarity. This stability, in turn, allows our adaptive spiking self-attention (A-SSA) module to perform two-dimensional token pruning across both spatial and temporal axes. Implemented on spiking Transformer architectures and validated on CIFAR-10, CIFAR-100, and ImageNet, STAS reduces energy consumption by up to 45.9%, 43.8%, and 30.1%, respectively, while simultaneously improving accuracy over SOTA models. 

---
# ERIS: An Energy-Guided Feature Disentanglement Framework for Out-of-Distribution Time Series Classification 

**Authors**: Xin Wu, Fei Teng, Ji Zhang, Xingwang Li, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14134)  

**Abstract**: An ideal time series classification (TSC) should be able to capture invariant representations, but achieving reliable performance on out-of-distribution (OOD) data remains a core obstacle. This obstacle arises from the way models inherently entangle domain-specific and label-relevant features, resulting in spurious correlations. While feature disentanglement aims to solve this, current methods are largely unguided, lacking the semantic direction required to isolate truly universal features. To address this, we propose an end-to-end Energy-Regularized Information for Shift-Robustness (\textbf{ERIS}) framework to enable guided and reliable feature disentanglement. The core idea is that effective disentanglement requires not only mathematical constraints but also semantic guidance to anchor the separation process. ERIS incorporates three key mechanisms to achieve this goal. Specifically, we first introduce an energy-guided calibration mechanism, which provides crucial semantic guidance for the separation, enabling the model to self-calibrate. Additionally, a weight-level orthogonality strategy enforces structural independence between domain-specific and label-relevant features, thereby mitigating their interference. Moreover, an auxiliary adversarial training mechanism enhances robustness by injecting structured perturbations. Experiments demonstrate that ERIS improves upon state-of-the-art baselines by an average of 4.04% accuracy across four benchmarks. 

---
# Automated surgical planning with nnU-Net: delineation of the anatomy in hepatobiliary phase MRI 

**Authors**: Karin A. Olthof, Matteo Fusagli, Bianca Güttner, Tiziano Natali, Bram Westerink, Stefanie Speidel, Theo J.M. Ruers, Koert F.D. Kuhlmann, Andrey Zhylka  

**Link**: [PDF](https://arxiv.org/pdf/2508.14133)  

**Abstract**: Background: The aim of this study was to develop and evaluate a deep learning-based automated segmentation method for hepatic anatomy (i.e., parenchyma, tumors, portal vein, hepatic vein and biliary tree) from the hepatobiliary phase of gadoxetic acid-enhanced MRI. This method should ease the clinical workflow of preoperative planning.
Methods: Manual segmentation was performed on hepatobiliary phase MRI scans from 90 consecutive patients who underwent liver surgery between January 2020 and October 2023. A deep learning network (nnU-Net v1) was trained on 72 patients with an extra focus on thin structures and topography preservation. Performance was evaluated on an 18-patient test set by comparing automated and manual segmentations using Dice similarity coefficient (DSC). Following clinical integration, 10 segmentations (assessment dataset) were generated using the network and manually refined for clinical use to quantify required adjustments using DSC.
Results: In the test set, DSCs were 0.97+/-0.01 for liver parenchyma, 0.80+/-0.04 for hepatic vein, 0.79+/-0.07 for biliary tree, 0.77+/-0.17 for tumors, and 0.74+/-0.06 for portal vein. Average tumor detection rate was 76.6+/-24.1%, with a median of one false-positive per patient. The assessment dataset showed minor adjustments were required for clinical use of the 3D models, with high DSCs for parenchyma (1.00+/-0.00), portal vein (0.98+/-0.01) and hepatic vein (0.95+/-0.07). Tumor segmentation exhibited greater variability (DSC 0.80+/-0.27). During prospective clinical use, the model detected three additional tumors initially missed by radiologists.
Conclusions: The proposed nnU-Net-based segmentation method enables accurate and automated delineation of hepatic anatomy. This enables 3D planning to be applied efficiently as a standard-of-care for every patient undergoing liver surgery. 

---
# An Improved Multi-Agent Algorithm for Cooperative and Competitive Environments by Identifying and Encouraging Cooperation among Agents 

**Authors**: Junjie Qi, Siqi Mao, Tianyi Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14131)  

**Abstract**: We propose an improved algorithm by identifying and encouraging cooperative behavior in multi-agent environments. First, we analyze the shortcomings of existing algorithms in addressing multi-agent reinforcement learning problems. Then, based on the existing algorithm MADDPG, we introduce a new parameter to increase the reward that an agent can obtain when cooperative behavior among agents is identified. Finally, we compare our improved algorithm with MADDPG in environments from PettingZoo. The results show that the new algorithm helps agents achieve both higher team rewards and individual rewards. 

---
# Fracture Detection and Localisation in Wrist and Hand Radiographs using Detection Transformer Variants 

**Authors**: Aditya Bagri, Vasanthakumar Venugopal, Anandakumar D, Revathi Ezhumalai, Kalyan Sivasailam, Bargava Subramanian, VarshiniPriya, Meenakumari K S, Abi M, Renita S  

**Link**: [PDF](https://arxiv.org/pdf/2508.14129)  

**Abstract**: Background: Accurate diagnosis of wrist and hand fractures using radiographs is essential in emergency care, but manual interpretation is slow and prone to errors. Transformer-based models show promise in improving medical image analysis, but their application to extremity fractures is limited. This study addresses this gap by applying object detection transformers to wrist and hand X-rays.
Methods: We fine-tuned the RT-DETR and Co-DETR models, pre-trained on COCO, using over 26,000 annotated X-rays from a proprietary clinical dataset. Each image was labeled for fracture presence with bounding boxes. A ResNet-50 classifier was trained on cropped regions to refine abnormality classification. Supervised contrastive learning was used to enhance embedding quality. Performance was evaluated using AP@50, precision, and recall metrics, with additional testing on real-world X-rays.
Results: RT-DETR showed moderate results (AP@50 = 0.39), while Co-DETR outperformed it with an AP@50 of 0.615 and faster convergence. The integrated pipeline achieved 83.1% accuracy, 85.1% precision, and 96.4% recall on real-world X-rays, demonstrating strong generalization across 13 fracture types. Visual inspection confirmed accurate localization.
Conclusion: Our Co-DETR-based pipeline demonstrated high accuracy and clinical relevance in wrist and hand fracture detection, offering reliable localization and differentiation of fracture types. It is scalable, efficient, and suitable for real-time deployment in hospital workflows, improving diagnostic speed and reliability in musculoskeletal radiology. 

---
# CCFC: Core & Core-Full-Core Dual-Track Defense for LLM Jailbreak Protection 

**Authors**: Jiaming Hu, Haoyu Wang, Debarghya Mukherjee, Ioannis Ch. Paschalidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.14128)  

**Abstract**: Jailbreak attacks pose a serious challenge to the safe deployment of large language models (LLMs). We introduce CCFC (Core & Core-Full-Core), a dual-track, prompt-level defense framework designed to mitigate LLMs' vulnerabilities from prompt injection and structure-aware jailbreak attacks. CCFC operates by first isolating the semantic core of a user query via few-shot prompting, and then evaluating the query using two complementary tracks: a core-only track to ignore adversarial distractions (e.g., toxic suffixes or prefix injections), and a core-full-core (CFC) track to disrupt the structural patterns exploited by gradient-based or edit-based attacks. The final response is selected based on a safety consistency check across both tracks, ensuring robustness without compromising on response quality. We demonstrate that CCFC cuts attack success rates by 50-75% versus state-of-the-art defenses against strong adversaries (e.g., DeepInception, GCG), without sacrificing fidelity on benign queries. Our method consistently outperforms state-of-the-art prompt-level defenses, offering a practical and effective solution for safer LLM deployment. 

---
# A Cost-Effective Framework for Predicting Parking Availability Using Geospatial Data and Machine Learning 

**Authors**: Madyan Bagosher, Tala Mustafa, Mohammad Alsmirat, Amal Al-Ali, Isam Mashhour Al Jawarneh  

**Link**: [PDF](https://arxiv.org/pdf/2508.14125)  

**Abstract**: As urban populations continue to grow, cities face numerous challenges in managing parking and determining occupancy. This issue is particularly pronounced in university campuses, where students need to find vacant parking spots quickly and conveniently during class timings. The limited availability of parking spaces on campuses underscores the necessity of implementing efficient systems to allocate vacant parking spots effectively. We propose a smart framework that integrates multiple data sources, including street maps, mobility, and meteorological data, through a spatial join operation to capture parking behavior and vehicle movement patterns over the span of 3 consecutive days with an hourly duration between 7AM till 3PM. The system will not require any sensing tools to be installed in the street or in the parking area to provide its services since all the data needed will be collected using location services. The framework will use the expected parking entrance and time to specify a suitable parking area. Several forecasting models, namely, Linear Regression, Support Vector Regression (SVR), Random Forest Regression (RFR), and Long Short-Term Memory (LSTM), are evaluated. Hyperparameter tuning was employed using grid search, and model performance is assessed using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) and Coefficient of Determination (R2). Random Forest Regression achieved the lowest RMSE of 0.142 and highest R2 of 0.582. However, given the time-series nature of the task, an LSTM model may perform better with additional data and longer timesteps. 

---
# AI Agents for Photonic Integrated Circuit Design Automation 

**Authors**: Ankita Sharma, YuQi Fu, Vahid Ansari, Rishabh Iyer, Fiona Kuang, Kashish Mistry, Raisa Islam Aishy, Sara Ahmad, Joaquin Matres, Dirk R. Englund, Joyce K.S. Poon  

**Link**: [PDF](https://arxiv.org/pdf/2508.14123)  

**Abstract**: We present Photonics Intelligent Design and Optimization (PhIDO), a multi-agent framework that converts natural-language photonic integrated circuit (PIC) design requests into layout mask files. We compare 7 reasoning large language models for PhIDO using a testbench of 102 design descriptions that ranged from single devices to 112-component PICs. The success rate for single-device designs was up to 91%. For design queries with less than or equal to 15 components, o1, Gemini-2.5-pro, and Claude Opus 4 achieved the highest end-to-end pass@5 success rates of approximately 57%, with Gemini-2.5-pro requiring the fewest output tokens and lowest cost. The next steps toward autonomous PIC development include standardized knowledge representations, expanded datasets, extended verification, and robotic automation. 

---
# SimGenHOI: Physically Realistic Whole-Body Humanoid-Object Interaction via Generative Modeling and Reinforcement Learning 

**Authors**: Yuhang Lin, Yijia Xie, Jiahong Xie, Yuehao Huang, Ruoyu Wang, Jiajun Lv, Yukai Ma, Xingxing Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14120)  

**Abstract**: Generating physically realistic humanoid-object interactions (HOI) is a fundamental challenge in robotics. Existing HOI generation approaches, such as diffusion-based models, often suffer from artifacts such as implausible contacts, penetrations, and unrealistic whole-body actions, which hinder successful execution in physical environments. To address these challenges, we introduce SimGenHOI, a unified framework that combines the strengths of generative modeling and reinforcement learning to produce controllable and physically plausible HOI. Our HOI generative model, based on Diffusion Transformers (DiT), predicts a set of key actions conditioned on text prompts, object geometry, sparse object waypoints, and the initial humanoid pose. These key actions capture essential interaction dynamics and are interpolated into smooth motion trajectories, naturally supporting long-horizon generation. To ensure physical realism, we design a contact-aware whole-body control policy trained with reinforcement learning, which tracks the generated motions while correcting artifacts such as penetration and foot sliding. Furthermore, we introduce a mutual fine-tuning strategy, where the generative model and the control policy iteratively refine each other, improving both motion realism and tracking robustness. Extensive experiments demonstrate that SimGenHOI generates realistic, diverse, and physically plausible humanoid-object interactions, achieving significantly higher tracking success rates in simulation and enabling long-horizon manipulation tasks. Code will be released upon acceptance on our project page: this https URL. 

---
# Documenting Deployment with Fabric: A Repository of Real-World AI Governance 

**Authors**: Mackenzie Jorgensen, Kendall Brogle, Katherine M. Collins, Lujain Ibrahim, Arina Shah, Petra Ivanovic, Noah Broestl, Gabriel Piles, Paul Dongha, Hatim Abdulhussein, Adrian Weller, Jillian Powers, Umang Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2508.14119)  

**Abstract**: Artificial intelligence (AI) is increasingly integrated into society, from financial services and traffic management to creative writing. Academic literature on the deployment of AI has mostly focused on the risks and harms that result from the use of AI. We introduce Fabric, a publicly available repository of deployed AI use cases to outline their governance mechanisms. Through semi-structured interviews with practitioners, we collect an initial set of 20 AI use cases. In addition, we co-design diagrams of the AI workflow with the practitioners. We discuss the oversight mechanisms and guardrails used in practice to safeguard AI use. The Fabric repository includes visual diagrams of AI use cases and descriptions of the deployed systems. Using the repository, we surface gaps in governance and find common patterns in human oversight of deployed AI systems. We intend for Fabric to serve as an extendable, evolving tool for researchers to study the effectiveness of AI governance. 

---
# Enriching Moral Perspectives on AI: Concepts of Trust amongst Africans 

**Authors**: Lameck Mbangula Amugongo, Nicola J Bidwell, Joseph Mwatukange  

**Link**: [PDF](https://arxiv.org/pdf/2508.14116)  

**Abstract**: The trustworthiness of AI is considered essential to the adoption and application of AI systems. However, the meaning of trust varies across industry, research and policy spaces. Studies suggest that professionals who develop and use AI regard an AI system as trustworthy based on their personal experiences and social relations at work. Studies about trust in AI and the constructs that aim to operationalise trust in AI (e.g., consistency, reliability, explainability and accountability). However, the majority of existing studies about trust in AI are situated in Western, Educated, Industrialised, Rich and Democratic (WEIRD) societies. The few studies about trust and AI in Africa do not include the views of people who develop, study or use AI in their work. In this study, we surveyed 157 people with professional and/or educational interests in AI from 25 African countries, to explore how they conceptualised trust in AI. Most respondents had links with workshops about trust and AI in Africa in Namibia and Ghana. Respondents' educational background, transnational mobility, and country of origin influenced their concerns about AI systems. These factors also affected their levels of distrust in certain AI applications and their emphasis on specific principles designed to foster trust. Respondents often expressed that their values are guided by the communities in which they grew up and emphasised communal relations over individual freedoms. They described trust in many ways, including applying nuances of Afro-relationalism to constructs in international discourse, such as reliability and reliance. Thus, our exploratory study motivates more empirical research about the ways trust is practically enacted and experienced in African social realities of AI design, use and governance. 

---
# Towards Low-Latency Tracking of Multiple Speakers With Short-Context Speaker Embeddings 

**Authors**: Taous Iatariene, Alexandre Guérin, Romain Serizel  

**Link**: [PDF](https://arxiv.org/pdf/2508.14115)  

**Abstract**: Speaker embeddings are promising identity-related features that can enhance the identity assignment performance of a tracking system by leveraging its spatial predictions, i.e, by performing identity reassignment. Common speaker embedding extractors usually struggle with short temporal contexts and overlapping speech, which imposes long-term identity reassignment to exploit longer temporal contexts. However, this increases the probability of tracking system errors, which in turn impacts negatively on identity reassignment. To address this, we propose a Knowledge Distillation (KD) based training approach for short context speaker embedding extraction from two speaker mixtures. We leverage the spatial information of the speaker of interest using beamforming to reduce overlap. We study the feasibility of performing identity reassignment over blocks of fixed size, i.e., blockwise identity reassignment, to go towards a low-latency speaker embedding based tracking system. Results demonstrate that our distilled models are effective at short-context embedding extraction and more robust to overlap. Although, blockwise reassignment results indicate that further work is needed to handle simultaneous speech more effectively. 

---
# Ambiguity Resolution with Human Feedback for Code Writing Tasks 

**Authors**: Aditey Nandan, Viraj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.14114)  

**Abstract**: Specifications for code writing tasks are usually expressed in natural language and may be ambiguous. Programmers must therefore develop the ability to recognize ambiguities in task specifications and resolve them by asking clarifying questions. We present and evaluate a prototype system, based on a novel technique (ARHF: Ambiguity Resolution with Human Feedback), that (1) suggests specific inputs on which a given task specification may be ambiguous, (2) seeks limited human feedback about the code's desired behavior on those inputs, and (3) uses this feedback to generate code that resolves these ambiguities. We evaluate the efficacy of our prototype, and we discuss the implications of such assistive systems on Computer Science education. 

---
# Federated Action Recognition for Smart Worker Assistance Using FastPose 

**Authors**: Vinit Hegiste, Vidit Goyal, Tatjana Legler, Martin Ruskowski  

**Link**: [PDF](https://arxiv.org/pdf/2508.14113)  

**Abstract**: In smart manufacturing environments, accurate and real-time recognition of worker actions is essential for productivity, safety, and human-machine collaboration. While skeleton-based human activity recognition (HAR) offers robustness to lighting, viewpoint, and background variations, most existing approaches rely on centralized datasets, which are impractical in privacy-sensitive industrial scenarios. This paper presents a federated learning (FL) framework for pose-based HAR using a custom skeletal dataset of eight industrially relevant upper-body gestures, captured from five participants and processed using a modified FastPose model. Two temporal backbones, an LSTM and a Transformer encoder, are trained and evaluated under four paradigms: centralized, local (per-client), FL with weighted federated averaging (FedAvg), and federated ensemble learning (FedEnsemble). On the global test set, the FL Transformer improves over centralized training by +12.4 percentage points, with FedEnsemble delivering a +16.3 percentage points gain. On an unseen external client, FL and FedEnsemble exceed centralized accuracy by +52.6 and +58.3 percentage points, respectively. These results demonstrate that FL not only preserves privacy but also substantially enhances cross-user generalization, establishing it as a practical solution for scalable, privacy-aware HAR in heterogeneous industrial settings. 

---
# Surya: Foundation Model for Heliophysics 

**Authors**: Sujit Roy, Johannes Schmude, Rohit Lal, Vishal Gaur, Marcus Freitag, Julian Kuehnert, Theodore van Kessel, Dinesha V. Hegde, Andrés Muñoz-Jaramillo, Johannes Jakubik, Etienne Vos, Kshitiz Mandal, Ata Akbari Asanjan, Joao Lucas de Sousa Almeida, Amy Lin, Talwinder Singh, Kang Yang, Chetraj Pandey, Jinsu Hong, Berkay Aydin, Thorsten Kurth, Ryan McGranaghan, Spiridon Kasapis, Vishal Upendran, Shah Bahauddin, Daniel da Silva, Nikolai V. Pogorelov, Campbell Watson, Manil Maskey, Madhulika Guhathakurta, Juan Bernabe-Moreno, Rahul Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2508.14112)  

**Abstract**: Heliophysics is central to understanding and forecasting space weather events and solar activity. Despite decades of high-resolution observations from the Solar Dynamics Observatory (SDO), most models remain task-specific and constrained by scarce labeled data, limiting their capacity to generalize across solar phenomena. We introduce Surya, a 366M parameter foundation model for heliophysics designed to learn general-purpose solar representations from multi-instrument SDO observations, including eight Atmospheric Imaging Assembly (AIA) channels and five Helioseismic and Magnetic Imager (HMI) products. Surya employs a spatiotemporal transformer architecture with spectral gating and long--short range attention, pretrained on high-resolution solar image forecasting tasks and further optimized through autoregressive rollout tuning. Zero-shot evaluations demonstrate its ability to forecast solar dynamics and flare events, while downstream fine-tuning with parameter-efficient Low-Rank Adaptation (LoRA) shows strong performance on solar wind forecasting, active region segmentation, solar flare forecasting, and EUV spectra. Surya is the first foundation model in heliophysics that uses time advancement as a pretext task on full-resolution SDO data. Its novel architecture and performance suggest that the model is able to learn the underlying physics behind solar evolution. 

---
# PAPPL: Personalized AI-Powered Progressive Learning Platform 

**Authors**: Shayan Bafandkar, Sungyong Chung, Homa Khosravian, Alireza Talebpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.14109)  

**Abstract**: Engineering education has historically been constrained by rigid, standardized frameworks, often neglecting students' diverse learning needs and interests. While significant advancements have been made in online and personalized education within K-12 and foundational sciences, engineering education at both undergraduate and graduate levels continues to lag in adopting similar innovations. Traditional evaluation methods, such as exams and homework assignments, frequently overlook individual student requirements, impeding personalized educational experiences. To address these limitations, this paper introduces the Personalized AI-Powered Progressive Learning (PAPPL) platform, an advanced Intelligent Tutoring System (ITS) designed specifically for engineering education. It highlights the development of a scalable, data-driven tutoring environment leveraging cutting-edge AI technology to enhance personalized learning across diverse academic disciplines, particularly in STEM fields. PAPPL integrates core ITS components including the expert module, student module, tutor module, and user interface, and utilizes GPT-4o, a sophisticated large language model (LLM), to deliver context-sensitive and pedagogically sound hints based on students' interactions. The system uniquely records student attempts, detects recurring misconceptions, and generates progressively targeted feedback, providing personalized assistance that adapts dynamically to each student's learning profile. Additionally, PAPPL offers instructors detailed analytics, empowering evidence-based adjustments to teaching strategies. This study provides a fundamental framework for the progression of Generative ITSs scalable to all education levels, delivering important perspectives on personalized progressive learning and the wider possibilities of Generative AI in the field of education. 

---
# SuryaBench: Benchmark Dataset for Advancing Machine Learning in Heliophysics and Space Weather Prediction 

**Authors**: Sujit Roy, Dinesha V. Hegde, Johannes Schmude, Amy Lin, Vishal Gaur, Rohit Lal, Kshitiz Mandal, Talwinder Singh, Andrés Muñoz-Jaramillo, Kang Yang, Chetraj Pandey, Jinsu Hong, Berkay Aydin, Ryan McGranaghan, Spiridon Kasapis, Vishal Upendran, Shah Bahauddin, Daniel da Silva, Marcus Freitag, Iksha Gurung, Nikolai Pogorelov, Campbell Watson, Manil Maskey, Juan Bernabe-Moreno, Rahul Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2508.14107)  

**Abstract**: This paper introduces a high resolution, machine learning-ready heliophysics dataset derived from NASA's Solar Dynamics Observatory (SDO), specifically designed to advance machine learning (ML) applications in solar physics and space weather forecasting. The dataset includes processed imagery from the Atmospheric Imaging Assembly (AIA) and Helioseismic and Magnetic Imager (HMI), spanning a solar cycle from May 2010 to July 2024. To ensure suitability for ML tasks, the data has been preprocessed, including correction of spacecraft roll angles, orbital adjustments, exposure normalization, and degradation compensation. We also provide auxiliary application benchmark datasets complementing the core SDO dataset. These provide benchmark applications for central heliophysics and space weather tasks such as active region segmentation, active region emergence forecasting, coronal field extrapolation, solar flare prediction, solar EUV spectra prediction, and solar wind speed estimation. By establishing a unified, standardized data collection, this dataset aims to facilitate benchmarking, enhance reproducibility, and accelerate the development of AI-driven models for critical space weather prediction tasks, bridging gaps between solar physics, machine learning, and operational forecasting. 

---
# High-Throughput Low-Cost Segmentation of Brightfield Microscopy Live Cell Images 

**Authors**: Surajit Das, Gourav Roy, Pavel Zun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14106)  

**Abstract**: Live cell culture is crucial in biomedical studies for analyzing cell properties and dynamics in vitro. This study focuses on segmenting unstained live cells imaged with bright-field microscopy. While many segmentation approaches exist for microscopic images, none consistently address the challenges of bright-field live-cell imaging with high throughput, where temporal phenotype changes, low contrast, noise, and motion-induced blur from cellular movement remain major obstacles. We developed a low-cost CNN-based pipeline incorporating comparative analysis of frozen encoders within a unified U-Net architecture enhanced with attention mechanisms, instance-aware systems, adaptive loss functions, hard instance retraining, dynamic learning rates, progressive mechanisms to mitigate overfitting, and an ensemble technique. The model was validated on a public dataset featuring diverse live cell variants, showing consistent competitiveness with state-of-the-art methods, achieving 93% test accuracy and an average F1-score of 89% (std. 0.07) on low-contrast, noisy, and blurry images. Notably, the model was trained primarily on bright-field images with limited exposure to phase-contrast microscopy (<10%), yet it generalized effectively to the phase-contrast LIVECell dataset, demonstrating modality, robustness and strong performance. This highlights its potential for real-world laboratory deployment across imaging conditions. The model requires minimal compute power and is adaptable using basic deep learning setups such as Google Colab, making it practical for training on other cell variants. Our pipeline outperforms existing methods in robustness and precision for bright-field microscopy segmentation. The code and dataset are available for reproducibility 

---
# You Don't Know Until You Click:Automated GUI Testing for Production-Ready Software Evaluation 

**Authors**: Yutong Bian, Xianhao Lin, Yupeng Xie, Tianyang Liu, Mingchen Zhuge, Siyuan Lu, Haoming Tang, Jinlin Wang, Jiayi Zhang, Jiaqi Chen, Xiangru Tang, Yongxin Ni, Sirui Hong, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14104)  

**Abstract**: Large Language Models (LLMs) and code agents in software development are rapidly evolving from generating isolated code snippets to producing full-fledged software applications with graphical interfaces, interactive logic, and dynamic behaviors. However, current benchmarks fall short in evaluating such production-ready software, as they often rely on static checks or binary pass/fail scripts, failing to capture the interactive behaviors and runtime dynamics that define real-world usability - qualities that only emerge when an application is actively used. This is the blind spot of current evaluation: you don't know if an app works until you click through it, interact with it, and observe how it responds. To bridge this gap, we introduce RealDevWorld, a novel evaluation framework for automated end-to-end assessment of LLMs' ability to generate production-ready repositories from scratch. It features two key components: (1) RealDevBench, a diverse collection of 194 open-ended software engineering tasks across multiple domains, incorporating multimodal elements to reflect real-world complexity; and (2) AppEvalPilot, a new agent-as-a-judge evaluation system that simulates realistic, GUI-based user interactions to automatically and holistically assess software functional correctness, visual fidelity, and runtime behavior. The framework delivers fine-grained, task-specific diagnostic feedback, supporting nuanced evaluation beyond simple success/failure judgments. Empirical results show that RealDevWorld delivers effective, automatic, and human-aligned evaluations, achieving an accuracy of 0.92 and a correlation of 0.85 with expert human assessments, while significantly reducing the reliance on manual review. This enables scalable, human-aligned assessment of production-level software generated by LLMs. Our code is available on GitHub. 

---
# Implicit Hypergraph Neural Network 

**Authors**: Akash Choudhuri, Yongjian Zhong, Bijaya Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2508.14101)  

**Abstract**: Hypergraphs offer a generalized framework for capturing high-order relationships between entities and have been widely applied in various domains, including healthcare, social networks, and bioinformatics. Hypergraph neural networks, which rely on message-passing between nodes over hyperedges to learn latent representations, have emerged as the method of choice for predictive tasks in many of these domains. These approaches typically perform only a small number of message-passing rounds to learn the representations, which they then utilize for predictions. The small number of message-passing rounds comes at a cost, as the representations only capture local information and forego long-range high-order dependencies. However, as we demonstrate, blindly increasing the message-passing rounds to capture long-range dependency also degrades the performance of hyper-graph neural networks.
Recent works have demonstrated that implicit graph neural networks capture long-range dependencies in standard graphs while maintaining performance. Despite their popularity, prior work has not studied long-range dependency issues on hypergraph neural networks. Here, we first demonstrate that existing hypergraph neural networks lose predictive power when aggregating more information to capture long-range dependency. We then propose Implicit Hypergraph Neural Network (IHNN), a novel framework that jointly learns fixed-point representations for both nodes and hyperedges in an end-to-end manner to alleviate this issue. Leveraging implicit differentiation, we introduce a tractable projected gradient descent approach to train the model efficiently. Extensive experiments on real-world hypergraphs for node classification demonstrate that IHNN outperforms the closest prior works in most settings, establishing a new state-of-the-art in hypergraph learning. 

---
# Domain Translation of a Soft Robotic Arm using Conditional Cycle Generative Adversarial Network 

**Authors**: Nilay Kushawaha, Carlo Alessi, Lorenzo Fruzzetti, Egidio Falotico  

**Link**: [PDF](https://arxiv.org/pdf/2508.14100)  

**Abstract**: Deep learning provides a powerful method for modeling the dynamics of soft robots, offering advantages over traditional analytical approaches that require precise knowledge of the robot's structure, material properties, and other physical characteristics. Given the inherent complexity and non-linearity of these systems, extracting such details can be challenging. The mappings learned in one domain cannot be directly transferred to another domain with different physical properties. This challenge is particularly relevant for soft robots, as their materials gradually degrade over time. In this paper, we introduce a domain translation framework based on a conditional cycle generative adversarial network (CCGAN) to enable knowledge transfer from a source domain to a target domain. Specifically, we employ a dynamic learning approach to adapt a pose controller trained in a standard simulation environment to a domain with tenfold increased viscosity. Our model learns from input pressure signals conditioned on corresponding end-effector positions and orientations in both domains. We evaluate our approach through trajectory-tracking experiments across five distinct shapes and further assess its robustness under noise perturbations and periodicity tests. The results demonstrate that CCGAN-GP effectively facilitates cross-domain skill transfer, paving the way for more adaptable and generalizable soft robotic controllers. 

---
# No More Marching: Learning Humanoid Locomotion for Short-Range SE(2) Targets 

**Authors**: Pranay Dugar, Mohitvishnu S. Gadde, Jonah Siekmann, Yesh Godse, Aayam Shrestha, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2508.14098)  

**Abstract**: Humanoids operating in real-world workspaces must frequently execute task-driven, short-range movements to SE(2) target poses. To be practical, these transitions must be fast, robust, and energy efficient. While learning-based locomotion has made significant progress, most existing methods optimize for velocity-tracking rather than direct pose reaching, resulting in inefficient, marching-style behavior when applied to short-range tasks. In this work, we develop a reinforcement learning approach that directly optimizes humanoid locomotion for SE(2) targets. Central to this approach is a new constellation-based reward function that encourages natural and efficient target-oriented movement. To evaluate performance, we introduce a benchmarking framework that measures energy consumption, time-to-target, and footstep count on a distribution of SE(2) goals. Our results show that the proposed approach consistently outperforms standard methods and enables successful transfer from simulation to hardware, highlighting the importance of targeted reward design for practical short-range humanoid locomotion. 

---
# Non-Dissipative Graph Propagation for Non-Local Community Detection 

**Authors**: William Leeney, Alessio Gravina, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14097)  

**Abstract**: Community detection in graphs aims to cluster nodes into meaningful groups, a task particularly challenging in heterophilic graphs, where nodes sharing similarities and membership to the same community are typically distantly connected. This is particularly evident when this task is tackled by graph neural networks, since they rely on an inherently local message passing scheme to learn the node representations that serve to cluster nodes into communities. In this work, we argue that the ability to propagate long-range information during message passing is key to effectively perform community detection in heterophilic graphs. To this end, we introduce the Unsupervised Antisymmetric Graph Neural Network (uAGNN), a novel unsupervised community detection approach leveraging non-dissipative dynamical systems to ensure stability and to propagate long-range information effectively. By employing antisymmetric weight matrices, uAGNN captures both local and global graph structures, overcoming the limitations posed by heterophilic scenarios. Extensive experiments across ten datasets demonstrate uAGNN's superior performance in high and medium heterophilic settings, where traditional methods fail to exploit long-range dependencies. These results highlight uAGNN's potential as a powerful tool for unsupervised community detection in diverse graph environments. 

---
# Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets 

**Authors**: Benjamin Pikus, Pratyush Ranjan Tiwari, Burton Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.14094)  

**Abstract**: Collecting high-quality training examples for language model fine-tuning is expensive, with practical budgets limiting the amount of data that can be procured. We investigate a critical question for resource-constrained alignment: under a fixed acquisition budget, should practitioners prioritize examples that are easy, medium, hard, or of random difficulty? We study Group Relative Policy Optimization (GRPO) fine-tuning across different model sizes and families, comparing four subset selection policies chosen from the same unlabeled pool using base-model difficulty estimates obtained via multi-sample evaluation. Our experiments reveal that training on the hardest examples yields the largest performance gains, up to 47%, while training on easy examples yield the smallest gains. Analysis reveals that this effect arises from harder examples providing more learnable opportunities during GRPO training. These findings provide practical guidance for budget-constrained post-training: prioritizing hard examples yields substantial performance gains on reasoning tasks when using GRPO. 

---
# Logical Expressivity and Explanations for Monotonic GNNs with Scoring Functions 

**Authors**: Matthew Morris, David J. Tena Cucala, Bernardo Cuenca Grau  

**Link**: [PDF](https://arxiv.org/pdf/2508.14091)  

**Abstract**: Graph neural networks (GNNs) are often used for the task of link prediction: predicting missing binary facts in knowledge graphs (KGs). To address the lack of explainability of GNNs on KGs, recent works extract Datalog rules from GNNs with provable correspondence guarantees. The extracted rules can be used to explain the GNN's predictions; furthermore, they can help characterise the expressive power of various GNN models. However, these works address only a form of link prediction based on a restricted, low-expressivity graph encoding/decoding method. In this paper, we consider a more general and popular approach for link prediction where a scoring function is used to decode the GNN output into fact predictions. We show how GNNs and scoring functions can be adapted to be monotonic, use the monotonicity to extract sound rules for explaining predictions, and leverage existing results about the kind of rules that scoring functions can capture. We also define procedures for obtaining equivalent Datalog programs for certain classes of monotonic GNNs with scoring functions. Our experiments show that, on link prediction benchmarks, monotonic GNNs and scoring functions perform well in practice and yield many sound rules. 

---
# DLLMQuant: Quantizing Diffusion-based Large Language Models 

**Authors**: Chen Xu, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14090)  

**Abstract**: Diffusion-based large language models (DLLMs) have shown promise for non-autoregressive text generation, but their deployment is constrained by large model sizes and heavy computational costs. Post-training quantization (PTQ), a widely used method for compressing and accelerating Large Language Models (LLMs), suffers from severe accuracy degradation and reduced generalization performance when directly applied to DLLMs (e.g., AWQ suffers a 16% accuracy drop on LLADA under W4A4). This paper explores how DLLMs' key mechanisms - dynamic masking, iterative generation, bidirectional attention - clash with quantization. We identify three core issues: 1) Iterative generation and dynamic masking ratios lead to distinct token distributions across decoding steps, which are not adequately captured by existing PTQ calibration methods; 2) Quantization errors are accumulated and amplified progressively during iteration in DLLMs, causing quantized models to perform worse as decoding steps progress; 3) Unmasked tokens stabilize while masked remain probabilistic, making overall feature distribution incompatible with existing PTQ methods. To address these issues, we propose DLLMQuant, a PTQ framework tailored for DLLMs, which incorporates three novel techniques: 1) Temporal-Mask Adaptive Sampling (TMAS), a calibration method that accounts for both time and mask factors, with the capacity to capture distributions across timesteps. 2) Interaction-Aware Activation Quantization (IA-AQ), which utilizes bidirectional attention's interaction signals to dynamically allocate quantization resources. 3) Certainty-Guided Quantization (CGQ), which integrates mask status and token scores as key weighting criteria into error compensation, making weight quantization more suitable for DLLMs. Experiments show that DLLMQuant achieves significant performance gains while enhancing efficiency. 

---
# CoBAD: Modeling Collective Behaviors for Human Mobility Anomaly Detection 

**Authors**: Haomin Wen, Shurui Cao, Leman Akoglu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14088)  

**Abstract**: Detecting anomalies in human mobility is essential for applications such as public safety and urban planning. While traditional anomaly detection methods primarily focus on individual movement patterns (e.g., a child should stay at home at night), collective anomaly detection aims to identify irregularities in collective mobility behaviors across individuals (e.g., a child is at home alone while the parents are elsewhere) and remains an underexplored challenge. Unlike individual anomalies, collective anomalies require modeling spatiotemporal dependencies between individuals, introducing additional complexity. To address this gap, we propose CoBAD, a novel model designed to capture Collective Behaviors for human mobility Anomaly Detection. We first formulate the problem as unsupervised learning over Collective Event Sequences (CES) with a co-occurrence event graph, where CES represents the event sequences of related individuals. CoBAD then employs a two-stage attention mechanism to model both the individual mobility patterns and the interactions across multiple individuals. Pre-trained on large-scale collective behavior data through masked event and link reconstruction tasks, CoBAD is able to detect two types of collective anomalies: unexpected co-occurrence anomalies and absence anomalies, the latter of which has been largely overlooked in prior work. Extensive experiments on large-scale mobility datasets demonstrate that CoBAD significantly outperforms existing anomaly detection baselines, achieving an improvement of 13%-18% in AUCROC and 19%-70% in AUCPR. All source code is available at this https URL. 

---
# FM4NPP: A Scaling Foundation Model for Nuclear and Particle Physics 

**Authors**: David Park, Shuhang Li, Yi Huang, Xihaier Luo, Haiwang Yu, Yeonju Go, Christopher Pinkenburg, Yuewei Lin, Shinjae Yoo, Joseph Osborn, Jin Huang, Yihui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.14087)  

**Abstract**: Large language models have revolutionized artificial intelligence by enabling large, generalizable models trained through self-supervision. This paradigm has inspired the development of scientific foundation models (FMs). However, applying this capability to experimental particle physics is challenging due to the sparse, spatially distributed nature of detector data, which differs dramatically from natural language. This work addresses if an FM for particle physics can scale and generalize across diverse tasks. We introduce a new dataset with more than 11 million particle collision events and a suite of downstream tasks and labeled data for evaluation. We propose a novel self-supervised training method for detector data and demonstrate its neural scalability with models that feature up to 188 million parameters. With frozen weights and task-specific adapters, this FM consistently outperforms baseline models across all downstream tasks. The performance also exhibits robust data-efficient adaptation. Further analysis reveals that the representations extracted by the FM are task-agnostic but can be specialized via a single linear mapping for different downstream tasks. 

---
# GeoMAE: Masking Representation Learning for Spatio-Temporal Graph Forecasting with Missing Values 

**Authors**: Songyu Ke, Chenyu Wu, Yuxuan Liang, Xiuwen Yi, Yanping Sun, Junbo Zhang, Yu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.14083)  

**Abstract**: Accurate acquisition of crowd flow at Points of Interest (POIs) is pivotal for effective traffic management, public service, and urban planning. Despite this importance, due to the limitations of urban sensing techniques, the data quality from most sources is inadequate for monitoring crowd flow at each POI. This renders the inference of accurate crowd flow from low-quality data a critical and challenging task. The complexity is heightened by three key factors: 1) \emph{The scarcity and rarity of labeled data}, 2) \emph{The intricate spatio-temporal dependencies among POIs}, and 3) \emph{The myriad correlations between precise crowd flow and GPS reports}.
To address these challenges, we recast the crowd flow inference problem as a self-supervised attributed graph representation learning task and introduce a novel \underline{C}ontrastive \underline{S}elf-learning framework for \underline{S}patio-\underline{T}emporal data (\model). Our approach initiates with the construction of a spatial adjacency graph founded on the POIs and their respective distances. We then employ a contrastive learning technique to exploit large volumes of unlabeled spatio-temporal data. We adopt a swapped prediction approach to anticipate the representation of the target subgraph from similar instances. Following the pre-training phase, the model is fine-tuned with accurate crowd flow data. Our experiments, conducted on two real-world datasets, demonstrate that the \model pre-trained on extensive noisy data consistently outperforms models trained from scratch. 

---
# Label Smoothing is a Pragmatic Information Bottleneck 

**Authors**: Sota Kudo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14077)  

**Abstract**: This study revisits label smoothing via a form of information bottleneck. Under the assumption of sufficient model flexibility and no conflicting labels for the same input, we theoretically and experimentally demonstrate that the model output obtained through label smoothing explores the optimal solution of the information bottleneck. Based on this, label smoothing can be interpreted as a practical approach to the information bottleneck, enabling simple implementation. As an information bottleneck method, we experimentally show that label smoothing also exhibits the property of being insensitive to factors that do not contain information about the target, or to factors that provide no additional information about it when conditioned on another variable. 

---
# PersRM-R1: Enhance Personalized Reward Modeling with Reinforcement Learning 

**Authors**: Mengdi Li, Guanqiao Chen, Xufeng Zhao, Haochen Wen, Shu Yang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14076)  

**Abstract**: Reward models (RMs), which are central to existing post-training methods, aim to align LLM outputs with human values by providing feedback signals during fine-tuning. However, existing RMs struggle to capture nuanced, user-specific preferences, especially under limited data and across diverse domains. Thus, we introduce PersRM-R1, the first reasoning-based reward modeling framework specifically designed to identify and represent personal factors from only one or a few personal exemplars. To address challenges including limited data availability and the requirement for robust generalization, our approach combines synthetic data generation with a two-stage training pipeline consisting of supervised fine-tuning followed by reinforcement fine-tuning. Experimental results demonstrate that PersRM-R1 outperforms existing models of similar size and matches the performance of much larger models in both accuracy and generalizability, paving the way for more effective personalized LLMs. 

---
# Explainable Graph Spectral Clustering For Text Embeddings 

**Authors**: Mieczysław A. Kłopotek, Sławomir T. Wierzchoń, Bartłomiej Starosta, Piotr Borkowski, Dariusz Czerski, Eryk Laskowski  

**Link**: [PDF](https://arxiv.org/pdf/2508.14075)  

**Abstract**: In a previous paper, we proposed an introduction to the explainability of Graph Spectral Clustering results for textual documents, given that document similarity is computed as cosine similarity in term vector space.
In this paper, we generalize this idea by considering other embeddings of documents, in particular, based on the GloVe embedding idea. 

---
# GEPD:GAN-Enhanced Generalizable Model for EEG-Based Detection of Parkinson's Disease 

**Authors**: Qian Zhang, Ruilin Zhang, Biaokai Zhu, Xun Han, Jun Xiao, Yifan Liu, Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14074)  

**Abstract**: Electroencephalography has been established as an effective method for detecting Parkinson's disease, typically diagnosed this http URL Parkinson's disease detection methods have shown significant success within individual datasets, however, the variability in detection methods across different EEG datasets and the small size of each dataset pose challenges for training a generalizable model for cross-dataset scenarios. To address these issues, this paper proposes a GAN-enhanced generalizable model, named GEPD, specifically for EEG-based cross-dataset classification of Parkinson's this http URL, we design a generative network that creates fusion EEG data by controlling the distribution similarity between generated data and real this http URL addition, an EEG signal quality assessment model is designed to ensure the quality of generated data this http URL, we design a classification network that utilizes a combination of multiple convolutional neural networks to effectively capture the time-frequency characteristics of EEG signals, while maintaining a generalizable structure and ensuring easy this http URL work is dedicated to utilizing intelligent methods to study pathological manifestations, aiming to facilitate the diagnosis and monitoring of neurological this http URL evaluation results demonstrate that our model performs comparably to state-of-the-art models in cross-dataset settings, achieving an accuracy of 84.3% and an F1-score of 84.0%, showcasing the generalizability of the proposed model. 

---
# MCLPD:Multi-view Contrastive Learning for EEG-based PD Detection Across Datasets 

**Authors**: Qian Zhanga, Ruilin Zhang, Jun Xiao, Yifan Liu, Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14073)  

**Abstract**: Electroencephalography has been validated as an effective technique for detecting Parkinson's disease,particularly in its early this http URL,the high cost of EEG data annotation often results in limited dataset size and considerable discrepancies across datasets,including differences in acquisition protocols and subject demographics,significantly hinder the robustness and generalizability of models in cross-dataset detection this http URL address such challenges,this paper proposes a semi-supervised learning framework named MCLPD,which integrates multi-view contrastive pre-training with lightweight supervised fine-tuning to enhance cross-dataset PD detection this http URL pre-training,MCLPD uses self-supervised learning on the unlabeled UNM this http URL build contrastive pairs,it applies dual augmentations in both time and frequency domains,which enrich the data and naturally fuse time-frequency this http URL the fine-tuning phase,only a small proportion of labeled data from another two datasets (UI and UC)is used for supervised this http URL results show that MCLPD achieves F1 scores of 0.91 on UI and 0.81 on UC using only 1%of labeled data,which further improve to 0.97 and 0.87,respectively,when 5%of labeled data is this http URL to existing methods,MCLPD substantially improves cross-dataset generalization while reducing the dependency on labeled data,demonstrating the effectiveness of the proposed framework. 

---
# Edge-Selector Model Applied for Local Search Neighborhood for Solving Vehicle Routing Problems 

**Authors**: Bachtiar Herdianto, Romain Billot, Flavien Lucas, Marc Sevaux, Daniele Vigo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14071)  

**Abstract**: This research proposes a hybrid Machine Learning and metaheuristic mechanism that is designed to solve Vehicle Routing Problems (VRPs). The main of our method is an edge solution selector model, which classifies solution edges to identify prohibited moves during the local search, hence guiding the search process within metaheuristic baselines. Two learning-based mechanisms are used to develop the edge selector: a simple tabular binary classifier and a Graph Neural Network (GNN). The tabular classifier employs Gradient Boosting Trees and Feedforward Neural Network as the baseline algorithms. Adjustments to the decision threshold are also applied to handle the class imbalance in the problem instance. An alternative mechanism employs the GNN to utilize graph structure for direct solution edge prediction, with the objective of guiding local search by predicting prohibited moves. These hybrid mechanisms are then applied in state-fo-the-art metaheuristic baselines. Our method demonstrates both scalability and generalizability, achieving performance improvements across different baseline metaheuristics, various problem sizes and variants, including the Capacitated Vehicle Routing Problem (CVRP) and CVRP with Time Windows (CVRPTW). Experimental evaluations on benchmark datasets up to 30,000 customer nodes, supported by pair-wise statistical analysis, verify the observed improvements. 

---
# Special-Character Adversarial Attacks on Open-Source Language Model 

**Authors**: Ephraiem Sarabamoun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14070)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across diverse natural language processing tasks, yet their vulnerability to character-level adversarial manipulations presents significant security challenges for real-world deployments. 

---
# Revisit Choice Network for Synthesis and Technology Mapping 

**Authors**: Chen Chen, Jiaqi Yin, Cunxi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14068)  

**Abstract**: Choice network construction is a critical technique for alleviating structural bias issues in Boolean optimization, equivalence checking, and technology mapping. Previous works on lossless synthesis utilize independent optimization to generate multiple snapshots, and use simulation and SAT solvers to identify functionally equivalent nodes. These nodes are then merged into a subject graph with choice nodes. However, such methods often neglect the quality of these choices, raising the question of whether they truly contribute to effective technology mapping.
This paper introduces Cristal, a novel methodology and framework for constructing Boolean choice networks. Specifically, Cristal introduces a new flow of choice network-based synthesis and mapping, including representative logic cone search, structural mutation for generating diverse choice structures via equality saturation, and priority-ranking choice selection along with choice network construction and validation. Through these techniques, Cristal constructs fewer but higher-quality choices.
Our experimental results demonstrate that Cristal outperforms the state-of-the-art Boolean choice network construction implemented in ABC in the post-mapping stage, achieving average reductions of 3.85%/8.35% (area/delay) in delay-oriented mode, 0.11%/2.74% in area-oriented mode, and a 63.77% runtime reduction on large-scale cases across a diverse set of combinational circuits from the IWLS 2005, ISCAS'89, and EPFL benchmark suites. 

---
# Retrieval-Augmented Generation in Industry: An Interview Study on Use Cases, Requirements, Challenges, and Evaluation 

**Authors**: Lorenz Brehme, Benedikt Dornauer, Thomas Ströhle, Maximilian Ehrhart, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14066)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a well-established and rapidly evolving field within AI that enhances the outputs of large language models by integrating relevant information retrieved from external knowledge sources. While industry adoption of RAG is now beginning, there is a significant lack of research on its practical application in industrial contexts. To address this gap, we conducted a semistructured interview study with 13 industry practitioners to explore the current state of RAG adoption in real-world settings. Our study investigates how companies apply RAG in practice, providing (1) an overview of industry use cases, (2) a consolidated list of system requirements, (3) key challenges and lessons learned from practical experiences, and (4) an analysis of current industry evaluation methods. Our main findings show that current RAG applications are mostly limited to domain-specific QA tasks, with systems still in prototype stages; industry requirements focus primarily on data protection, security, and quality, while issues such as ethics, bias, and scalability receive less attention; data preprocessing remains a key challenge, and system evaluation is predominantly conducted by humans rather than automated methods. 

---
# An automatic patent literature retrieval system based on LLM-RAG 

**Authors**: Yao Ding, Yuqing Wu, Ziyang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.14064)  

**Abstract**: With the acceleration of technological innovation efficient retrieval and classification of patent literature have become essential for intellectual property management and enterprise RD Traditional keyword and rulebased retrieval methods often fail to address complex query intents or capture semantic associations across technical domains resulting in incomplete and lowrelevance results This study presents an automated patent retrieval framework integrating Large Language Models LLMs with RetrievalAugmented Generation RAG technology The system comprises three components: 1) a preprocessing module for patent data standardization, 2) a highefficiency vector retrieval engine leveraging LLMgenerated embeddings, and 3) a RAGenhanced query module that combines external document retrieval with contextaware response generation Evaluations were conducted on the Google Patents dataset 20062024 containing millions of global patent records with metadata such as filing date domain and status The proposed gpt35turbo0125RAG configuration achieved 805 semantic matching accuracy and 92.1% recall surpassing baseline LLM methods by 28 percentage points The framework also demonstrated strong generalization in crossdomain classification and semantic clustering tasks These results validate the effectiveness of LLMRAG integration for intelligent patent retrieval providing a foundation for nextgeneration AIdriven intellectual property analysis platforms 

---
# A Multi-Agent Approach to Neurological Clinical Reasoning 

**Authors**: Moran Sorka, Alon Gorenshtein, Dvir Aran, Shahar Shelly  

**Link**: [PDF](https://arxiv.org/pdf/2508.14063)  

**Abstract**: Large language models (LLMs) have shown promise in medical domains, but their ability to handle specialized neurological reasoning requires systematic evaluation. We developed a comprehensive benchmark using 305 questions from Israeli Board Certification Exams in Neurology, classified along three complexity dimensions: factual knowledge depth, clinical concept integration, and reasoning complexity. We evaluated ten LLMs using base models, retrieval-augmented generation (RAG), and a novel multi-agent system. Results showed significant performance variation. OpenAI-o1 achieved the highest base performance (90.9% accuracy), while specialized medical models performed poorly (52.9% for Meditron-70B). RAG provided modest benefits but limited effectiveness on complex reasoning questions. In contrast, our multi-agent framework, decomposing neurological reasoning into specialized cognitive functions including question analysis, knowledge retrieval, answer synthesis, and validation, achieved dramatic improvements, especially for mid-range models. The LLaMA 3.3-70B-based agentic system reached 89.2% accuracy versus 69.5% for its base model, with substantial gains on level 3 complexity questions. The multi-agent approach transformed inconsistent subspecialty performance into uniform excellence, addressing neurological reasoning challenges that persisted with RAG enhancement. We validated our approach using an independent dataset of 155 neurological cases from MedQA. Results confirm that structured multi-agent approaches designed to emulate specialized cognitive processes significantly enhance complex medical reasoning, offering promising directions for AI assistance in challenging clinical contexts. 

---
# Assessing and Mitigating Data Memorization Risks in Fine-Tuned Large Language Models 

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji  

**Link**: [PDF](https://arxiv.org/pdf/2508.14062)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language processing tasks, but their tendency to memorize training data poses significant privacy risks, particularly during fine-tuning processes. This paper presents a comprehensive empirical analysis of data memorization in fine-tuned LLMs and introduces a novel multi-layered privacy protection framework. Through controlled experiments on modern LLM architectures including GPT-2, Phi-3, and Gemma-2, we demonstrate that fine-tuning with repeated sensitive data increases privacy leakage rates from baseline levels of 0-5% to 60-75%, representing a 64.2% average increase across tested models. We propose and rigorously evaluate four complementary privacy protection methods: semantic data deduplication, differential privacy during generation, entropy-based filtering, and pattern-based content filtering. Our experimental results show that these techniques can reduce data leakage to 0% while maintaining 94.7% of original model utility. 

---
# Dual-Phase Playtime-guided Recommendation: Interest Intensity Exploration and Multimodal Random Walks 

**Authors**: Jingmao Zhang, Zhiting Zhao, Yunqi Lin, Jianghong Ma, Tianjun Wei, Haijun Zhang, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14058)  

**Abstract**: The explosive growth of the video game industry has created an urgent need for recommendation systems that can scale with expanding catalogs and maintain user engagement. While prior work has explored accuracy and diversity in recommendations, existing models underutilize playtime, a rich behavioral signal unique to gaming platforms, and overlook the potential of multimodal information to enhance diversity. In this paper, we propose DP2Rec, a novel Dual-Phase Playtime-guided Recommendation model designed to jointly optimize accuracy and diversity. First, we introduce a playtime-guided interest intensity exploration module that separates strong and weak preferences via dual-beta modeling, enabling fine-grained user profiling and more accurate recommendations. Second, we present a playtime-guided multimodal random walks module that simulates player exploration using transitions guided by both playtime-derived interest similarity and multimodal semantic similarity. This mechanism preserves core preferences while promoting cross-category discovery through latent semantic associations and adaptive category balancing. Extensive experiments on a real-world game dataset show that DP2Rec outperforms existing methods in both recommendation accuracy and diversity. 

---
# T-REX: Table -- Refute or Entail eXplainer 

**Authors**: Tim Luka Horstmann, Baptiste Geisenberger, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2508.14055)  

**Abstract**: Verifying textual claims against structured tabular data is a critical yet challenging task in Natural Language Processing with broad real-world impact. While recent advances in Large Language Models (LLMs) have enabled significant progress in table fact-checking, current solutions remain inaccessible to non-experts. We introduce T-REX (T-REX: Table -- Refute or Entail eXplainer), the first live, interactive tool for claim verification over multimodal, multilingual tables using state-of-the-art instruction-tuned reasoning LLMs. Designed for accuracy and transparency, T-REX empowers non-experts by providing access to advanced fact-checking technology. The system is openly available online. 

---
# MAHL: Multi-Agent LLM-Guided Hierarchical Chiplet Design with Adaptive Debugging 

**Authors**: Jinwei Tang, Jiayin Qin, Nuo Xu, Pragnya Sudershan Nalla, Yu Cao, Yang, Zhao, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.14053)  

**Abstract**: As program workloads (e.g., AI) increase in size and algorithmic complexity, the primary challenge lies in their high dimensionality, encompassing computing cores, array sizes, and memory hierarchies. To overcome these obstacles, innovative approaches are required. Agile chip design has already benefited from machine learning integration at various stages, including logic synthesis, placement, and routing. With Large Language Models (LLMs) recently demonstrating impressive proficiency in Hardware Description Language (HDL) generation, it is promising to extend their abilities to 2.5D integration, an advanced technique that saves area overhead and development costs. However, LLM-driven chiplet design faces challenges such as flatten design, high validation cost and imprecise parameter optimization, which limit its chiplet design capability. To address this, we propose MAHL, a hierarchical LLM-based chiplet design generation framework that features six agents which collaboratively enable AI algorithm-hardware mapping, including hierarchical description generation, retrieval-augmented code generation, diverseflow-based validation, and multi-granularity design space exploration. These components together enhance the efficient generation of chiplet design with optimized Power, Performance and Area (PPA). Experiments show that MAHL not only significantly improves the generation accuracy of simple RTL design, but also increases the generation accuracy of real-world chiplet design, evaluated by Pass@5, from 0 to 0.72 compared to conventional LLMs under the best-case scenario. Compared to state-of-the-art CLARIE (expert-based), MAHL achieves comparable or even superior PPA results under certain optimization objectives. 

---
# FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering 

**Authors**: Chanyeol Choi, Jihoon Kwon, Alejandro Lopez-Lira, Chaewoon Kim, Minjae Kim, Juneha Hwang, Jaeseon Ha, Hojun Choi, Suyeol Yun, Yongjin Kim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14052)  

**Abstract**: Accurate information retrieval (IR) is critical in the financial domain, where investors must identify relevant information from large collections of documents. Traditional IR methods-whether sparse or dense-often fall short in retrieval accuracy, as it requires not only capturing semantic similarity but also performing fine-grained reasoning over document structure and domain-specific knowledge. Recent advances in large language models (LLMs) have opened up new opportunities for retrieval with multi-step reasoning, where the model ranks passages through iterative reasoning about which information is most relevant to a given query. However, there exists no benchmark to evaluate such capabilities in the financial domain. To address this gap, we introduce FinAgentBench, the first large-scale benchmark for evaluating retrieval with multi-step reasoning in finance -- a setting we term agentic retrieval. The benchmark consists of 3,429 expert-annotated examples on S&P-100 listed firms and assesses whether LLM agents can (1) identify the most relevant document type among candidates, and (2) pinpoint the key passage within the selected document. Our evaluation framework explicitly separates these two reasoning steps to address context limitations. This design enables to provide a quantitative basis for understanding retrieval-centric LLM behavior in finance. We evaluate a suite of state-of-the-art models and further demonstrated how targeted fine-tuning can significantly improve agentic retrieval performance. Our benchmark provides a foundation for studying retrieval-centric LLM behavior in complex, domain-specific tasks for finance. We will release the dataset publicly upon acceptance of the paper and plan to expand and share dataset for the full S&P 500 and beyond. 

---
# The Hidden Cost of Readability: How Code Formatting Silently Consumes Your LLM Budget 

**Authors**: Dangfeng Pan, Zhensu Sun, Cenyuan Zhang, David Lo, Xiaoning Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.13666)  

**Abstract**: Source code is usually formatted with elements like indentation and newlines to improve readability for human developers. However, these visual aids do not seem to be beneficial for large language models (LLMs) in the same way since the code is processed as a linear sequence of tokens. Furthermore, these additional tokens can lead to increased computational costs and longer response times for LLMs. If such formatting elements are non-essential to LLMs, we can reduce such costs by removing them from the code. To figure out the role played by formatting elements, we conduct a comprehensive empirical study to evaluate the impact of code formatting on LLM performance and efficiency. Through large-scale experiments on Fill-in-the-Middle Code Completion tasks across four programming languages (Java, Python, C++, C\#) and ten LLMs-including both commercial and open-source models-we systematically analyze token count and performance when formatting elements are removed. Key findings indicate that LLMs can maintain performance across formatted code and unformatted code, achieving an average input token reduction of 24.5\% with negligible output token reductions. This makes code format removal a practical optimization strategy for improving LLM efficiency. Further exploration reveals that both prompting and fine-tuning LLMs can lead to significant reductions (up to 36.1\%) in output code length without compromising correctness. To facilitate practical applications, we develop a bidirectional code transformation tool for format processing, which can be seamlessly integrated into existing LLM inference workflows, ensuring both human readability and LLM efficiency. 

---
