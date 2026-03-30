# Stabilizing Rubric Integration Training via Decoupled Advantage Normalization 

**Authors**: Zelin Tan, Zhouliang Yu, Bohan Lin, Zijie Geng, Hejia Geng, Yudong Zhang, Mulei Zhang, Yang Chen, Shuyue Hu, Zhenfei Yin, Chen Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.26535)  

**Abstract**: We propose Process-Aware Policy Optimization (PAPO), a method that integrates process-level evaluation into Group Relative Policy Optimization (GRPO) through decoupled advantage normalization, to address two limitations of existing reward designs. Outcome reward models (ORM) evaluate only final-answer correctness, treating all correct responses identically regardless of reasoning quality, and gradually lose the advantage signal as groups become uniformly correct. Process reward models (PRM) offer richer supervision, but directly using PRM scores causes reward hacking, where models exploit verbosity to inflate scores while accuracy collapses. PAPO resolves both by composing the advantage from an outcome component Aout, derived from ORM and normalized over all responses, and a process component Aproc, derived from a rubric-based PRM and normalized exclusively among correct responses. This decoupled design ensures that Aout anchors training on correctness while Aproc differentiates reasoning quality without distorting the outcome signal. Experiments across multiple model scales and six benchmarks demonstrate that PAPO consistently outperforms ORM, reaching 51.3% vs.\ 46.3% on OlympiadBench while continuing to improve as ORM plateaus and declines. 

---
# CADSmith: Multi-Agent CAD Generation with Programmatic Geometric Validation 

**Authors**: Jesse Barkley, Rumi Loghmani, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2603.26512)  

**Abstract**: Existing methods for text-to-CAD generation either operate in a single pass with no geometric verification or rely on lossy visual feedback that cannot resolve dimensional errors. We present CADSmith, a multi-agent pipeline that generates CadQuery code from natural language. It then undergoes an iterative refinement process through two nested correction loops: an inner loop that resolves execution errors and an outer loop grounded in programmatic geometric validation. The outer loop combines exact measurements from the OpenCASCADE kernel (bounding box dimensions, volume, solid validity) with holistic visual assessment from an independent vision-language model Judge. This provides both the numerical precision and the high-level shape awareness needed to converge on the correct geometry. The system uses retrieval-augmented generation over API documentation rather than fine-tuning, maintaining a current database as the underlying CAD library evolves. We evaluate on a custom benchmark of 100 prompts in three difficulty tiers (T1 through T3) with three ablation configurations. Against a zero-shot baseline, CADSmith achieves a 100% execution rate (up from 95%), improves the median F1 score from 0.9707 to 0.9846, the median IoU from 0.8085 to 0.9629, and reduces the mean Chamfer Distance from 28.37 to 0.74, demonstrating that closed-loop refinement with programmatic geometric feedback substantially improves the quality and reliability of LLM-generated CAD models. 

---
# AIRA_2: Overcoming Bottlenecks in AI Research Agents 

**Authors**: Karen Hambardzumyan, Nicolas Baldwin, Edan Toledo, Rishi Hazra, Michael Kuchnik, Bassel Al Omari, Thomas Simon Foster, Anton Protopopov, Jean-Christophe Gagnon-Audet, Ishita Mediratta, Kelvin Niu, Michael Shvartsman, Alisia Lupidi, Alexis Audran-Reiss, Parth Pathak, Tatiana Shavrina, Despoina Magka, Hela Momand, Derek Dunfield, Nicola Cancedda, Pontus Stenetorp, Carole-Jean Wu, Jakob Nicolaus Foerster, Yoram Bachrach, Martin Josifoski  

**Link**: [PDF](https://arxiv.org/pdf/2603.26499)  

**Abstract**: Existing research has identified three structural performance bottlenecks in AI research agents: (1) synchronous single-GPU execution constrains sample throughput, limiting the benefit of search; (2) a generalization gap where validation-based selection causes performance to degrade over extended search horizons; and (3) the limited capability of fixed, single-turn LLM operators imposes a ceiling on search performance. We introduce AIRA$_2$, which addresses these bottlenecks through three architectural choices: an asynchronous multi-GPU worker pool that increases experiment throughput linearly; a Hidden Consistent Evaluation protocol that delivers a reliable evaluation signal; and ReAct agents that dynamically scope their actions and debug interactively. On MLE-bench-30, AIRA$_2$ achieves a mean Percentile Rank of 71.8% at 24 hours - surpassing the previous best of 69.9% - and steadily improves to 76.0% at 72 hours. Ablation studies reveal that each component is necessary and that the "overfitting" reported in prior work was driven by evaluation noise rather than true data memorization. 

---
# GUIDE: Resolving Domain Bias in GUI Agents through Real-Time Web Video Retrieval and Plug-and-Play Annotation 

**Authors**: Rui Xie, Zhi Gao, Chenrui Shi, Zirui Shang, Lu Chen, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.26266)  

**Abstract**: Large vision-language models have endowed GUI agents with strong general capabilities for interface understanding and interaction. However, due to insufficient exposure to domain-specific software operation data during training, these agents exhibit significant domain bias - they lack familiarity with the specific operation workflows (planning) and UI element layouts (grounding) of particular applications, limiting their real-world task performance. In this paper, we present GUIDE (GUI Unbiasing via Instructional-Video Driven Expertise), a training-free, plug-and-play framework that resolves GUI agent domain bias by autonomously acquiring domain-specific expertise from web tutorial videos through a retrieval-augmented automated annotation pipeline. GUIDE introduces two key innovations. First, a subtitle-driven Video-RAG pipeline unlocks video semantics through subtitle analysis, performing progressive three-stage retrieval - domain classification, topic extraction, and relevance matching - to identify task-relevant tutorial videos. Second, a fully automated annotation pipeline built on an inverse dynamics paradigm feeds consecutive keyframes enhanced with UI element detection into VLMs, inferring the required planning and grounding knowledge that are injected into the agent's corresponding modules to address both manifestations of domain bias. Extensive experiments on OSWorld demonstrate GUIDE's generality as a plug-and-play component for both multi-agent systems and single-model agents. It consistently yields over 5% improvements and reduces execution steps - without modifying any model parameters or architecture - validating GUIDE as an architecture-agnostic enhancement to bridge GUI agent domain bias. 

---
# Semi-Automated Knowledge Engineering and Process Mapping for Total Airport Management 

**Authors**: Darryl Teo, Adharsha Sam, Chuan Shen Marcus Koh, Rakesh Nagi, Nuno Antunes Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2603.26076)  

**Abstract**: Documentation of airport operations is inherently complex due to extensive technical terminology, rigorous regulations, proprietary regional information, and fragmented communication across multiple stakeholders. The resulting data silos and semantic inconsistencies present a significant impediment to the Total Airport Management (TAM) initiative. This paper presents a methodological framework for constructing a domain-grounded, machine-readable Knowledge Graph (KG) through a dual-stage fusion of symbolic Knowledge Engineering (KE) and generative Large Language Models (LLMs).
The framework employs a scaffolded fusion strategy in which expert-curated KE structures guide LLM prompts to facilitate the discovery of semantically aligned knowledge triples. We evaluate this methodology on the Google LangExtract library and investigate the impact of context window utilization by comparing localized segment-based inference with document-level processing. Contrary to prior empirical observations of long-context degradation in LLMs, document-level processing improves the recovery of non-linear procedural dependencies.
To ensure the high-fidelity provenance required in airport operations, the proposed framework fuses a probabilistic model for discovery and a deterministic algorithm for anchoring every extraction to its ground source. This ensures absolute traceability and verifiability, bridging the gap between "black-box" generative outputs and the transparency required for operational tooling. Finally, we introduce an automated framework that operationalizes this pipeline to synthesize complex operational workflows from unstructured textual corpora. 

---
# AutoB2G: A Large Language Model-Driven Agentic Framework For Automated Building-Grid Co-Simulation 

**Authors**: Borui Zhang, Nariman Mahdavi, Subbu Sethuvenkatraman, Shuang Ao, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2603.26005)  

**Abstract**: The growing availability of building operational data motivates the use of reinforcement learning (RL), which can learn control policies directly from data and cope with the complexity and uncertainty of large-scale building clusters. However, most existing simulation environments prioritize building-side performance metrics and lack systematic evaluation of grid-level impacts, while their experimental workflows still rely heavily on manual configuration and substantial programming expertise. Therefore, this paper proposes AutoB2G, an automated building-grid co-simulation framework that completes the entire simulation workflow solely based on natural-language task descriptions. The framework extends CityLearn V2 to support Building-to-Grid (B2G) interaction and adopts the large language model (LLM)-based SOCIA (Simulation Orchestration for Computational Intelligence with Agents) framework to automatically generate, execute, and iteratively refine the simulator. As LLMs lack prior knowledge of the implementation context of simulation functions, a codebase covering simulation configurations and functional modules is constructed and organized as a directed acyclic graph (DAG) to explicitly represent module dependencies and execution order, guiding the LLM to retrieve a complete executable path. Experimental results demonstrate that AutoB2G can effectively enable automated simulator implementations, coordinating B2G interactions to improve grid-side performance metrics. 

---
# BeSafe-Bench: Unveiling Behavioral Safety Risks of Situated Agents in Functional Environments 

**Authors**: Yuxuan Li, Yi Lin, Peng Wang, Shiming Liu, Xuetao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.25747)  

**Abstract**: The rapid evolution of Large Multimodal Models (LMMs) has enabled agents to perform complex digital and physical tasks, yet their deployment as autonomous decision-makers introduces substantial unintentional behavioral safety risks. However, the absence of a comprehensive safety benchmark remains a major bottleneck, as existing evaluations rely on low-fidelity environments, simulated APIs, or narrowly scoped tasks. To address this gap, we present BeSafe-Bench (BSB), a benchmark for exposing behavioral safety risks of situated agents in functional environments, covering four representative domains: Web, Mobile, Embodied VLM, and Embodied VLA. Using functional environments, we construct a diverse instruction space by augmenting tasks with nine categories of safety-critical risks, and adopt a hybrid evaluation framework that combines rule-based checks with LLM-as-a-judge reasoning to assess real environmental impacts. Evaluating 13 popular agents reveals a concerning trend: even the best-performing agent completes fewer than 40% of tasks while fully adhering to safety constraints, and strong task performance frequently coincides with severe safety violations. These findings underscore the urgent need for improved safety alignment before deploying agentic systems in real-world settings. 

---
# Ruka-v2: Tendon Driven Open-Source Dexterous Hand with Wrist and Abduction for Robot Learning 

**Authors**: Xinqi, Ruoxi Hu, Alejandro Ojeda Olarte, Zhuoran Chen, Kenny Ma, Charles Cheng Ji, Lerrel Pinto, Raunaq Bhirangi, Irmak Guzey  

**Link**: [PDF](https://arxiv.org/pdf/2603.26660)  

**Abstract**: Lack of accessible and dexterous robot hardware has been a significant bottleneck to achieving human-level dexterity in robots. Last year, we released Ruka, a fully open-sourced, tendon-driven humanoid hand with 11 degrees of freedom - 2 per finger and 3 at the thumb - buildable for under $1,300. It was one of the first fully open-sourced humanoid hands, and introduced a novel data-driven approach to finger control that captures tendon dynamics within the control system. Despite these contributions, Ruka lacked two degrees of freedom essential for closely imitating human behavior: wrist mobility and finger adduction/abduction. In this paper, we introduce Ruka-v2: a fully open-sourced, tendon-driven humanoid hand featuring a decoupled 2-DOF parallel wrist and abduction/adduction at the fingers. The parallel wrist adds smooth, independent flexion/extension and radial/ulnar deviation, enabling manipulation in confined environments such as cabinets. Abduction enables motions such as grasping thin objects, in-hand rotation, and calligraphy. We present the design of Ruka-v2 and evaluate it against Ruka through user studies on teleoperated tasks, finding a 51.3% reduction in completion time and a 21.2% increase in success rate. We further demonstrate its full range of applications for robot learning: bimanual and single-arm teleoperation across 13 dexterous tasks, and autonomous policy learning on 3 tasks. All 3D print files, assembly instructions, controller software, and videos are available at this https URL . 

---
# PerceptionComp: A Video Benchmark for Complex Perception-Centric Reasoning 

**Authors**: Shaoxuan Li, Zhixuan Zhao, Hanze Deng, Zirun Ma, Shulin Tian, Zuyan Liu, Yushi Hu, Haoning Wu, Yuhao Dong, Benlin Liu, Ziwei Liu, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2603.26653)  

**Abstract**: We introduce PerceptionComp, a manually annotated benchmark for complex, long-horizon, perception-centric video reasoning. PerceptionComp is designed so that no single moment is sufficient: answering each question requires multiple temporally separated pieces of visual evidence and compositional constraints under conjunctive and sequential logic, spanning perceptual subtasks such as objects, attributes, relations, locations, actions, and events, and requiring skills including semantic recognition, visual correspondence, temporal reasoning, and spatial reasoning. The benchmark contains 1,114 highly complex questions on 279 videos from diverse domains including city walk tours, indoor villa tours, video games, and extreme outdoor sports, with 100% manual annotation. Human studies show that PerceptionComp requires substantial test-time thinking and repeated perception steps: participants take much longer than on prior benchmarks, and accuracy drops to near chance (18.97%) when rewatching is disallowed. State-of-the-art MLLMs also perform substantially worse on PerceptionComp than on existing benchmarks: the best model in our evaluation, Gemini-3-Flash, reaches only 45.96% accuracy in the five-choice setting, while open-source models remain below 40%. These results suggest that perception-centric long-horizon video reasoning remains a major bottleneck, and we hope PerceptionComp will help drive progress in perceptual reasoning. 

---
# Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification 

**Authors**: Zehai He, Wenyi Hong, Zhen Yang, Ziyang Pan, Mingdao Liu, Xiaotao Gu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26648)  

**Abstract**: Recent advances in large language models have improved the capabilities of coding agents, yet systematic evaluation of complex, end-to-end website development remains limited. To address this gap, we introduce Vision2Web, a hierarchical benchmark for visual website development, spanning from static UI-to-code generation, interactive multi-page frontend reproduction, to long-horizon full-stack website development. The benchmark is constructed from real-world websites and comprises a total of 193 tasks across 16 categories, with 918 prototype images and 1,255 test cases. To support flexible, thorough and reliable evaluation, we propose workflow-based agent verification paradigm based on two complementary components: a GUI agent verifier and a VLM-based judge. We evaluate multiple visual language models instantiated under different coding-agent frameworks, revealing substantial performance gaps at all task levels, with state-of-the-art models still struggling on full-stack development. 

---
# Make Geometry Matter for Spatial Reasoning 

**Authors**: Shihua Zhang, Qiuhong Shen, Shizun Wang, Tianbo Pan, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26639)  

**Abstract**: Empowered by large-scale training, vision-language models (VLMs) achieve strong image and video understanding, yet their ability to perform spatial reasoning in both static scenes and dynamic videos remains limited. Recent advances try to handle this limitation by injecting geometry tokens from pretrained 3D foundation models into VLMs. Nevertheless, we observe that naive token fusion followed by standard fine-tuning in this line of work often leaves such geometric cues underutilized for spatial reasoning, as VLMs tend to rely heavily on 2D visual cues. In this paper, we propose GeoSR, a framework designed to make geometry matter by encouraging VLMs to actively reason with geometry tokens. GeoSR introduces two key components: (1) Geometry-Unleashing Masking, which strategically masks portions of 2D vision tokens during training to weaken non-geometric shortcuts and force the model to consult geometry tokens for spatial reasoning; and (2) Geometry-Guided Fusion, a gated routing mechanism that adaptively amplifies geometry token contributions in regions where geometric evidence is critical. Together, these designs unleash the potential of geometry tokens for spatial reasoning tasks. Extensive experiments on both static and dynamic spatial reasoning benchmarks demonstrate that GeoSR consistently outperforms prior methods and establishes new state-of-the-art performance by effectively leveraging geometric information. The project page is available at this https URL. 

---
# Machine Learning Transferability for Malware Detection 

**Authors**: César Vieira, João Vitorino, Eva Maia, Isabel Praça  

**Link**: [PDF](https://arxiv.org/pdf/2603.26632)  

**Abstract**: Malware continues to be a predominant operational risk for organizations, especially when obfuscation techniques are used to evade detection. Despite the ongoing efforts in the development of Machine Learning (ML) detection approaches, there is still a lack of feature compatibility in public datasets. This limits generalization when facing distribution shifts, as well as transferability to different datasets. This study evaluates the suitability of different data preprocessing approaches for the detection of Portable Executable (PE) files with ML models. The preprocessing pipeline unifies EMBERv2 (2,381-dim) features datasets, trains paired models under two training setups: EMBER + BODMAS and EMBER + BODMAS + ERMDS. Regarding model evaluation, both EMBER + BODMAS and EMBER + BODMAS + ERMDS models are tested against TRITIUM, INFERNO and SOREL-20M. ERMDS is also used for testing for the EMBER + BODMAS setup. 

---
# Think over Trajectories: Leveraging Video Generation to Reconstruct GPS Trajectories from Cellular Signaling 

**Authors**: Ruixing Zhang, Hanzhang Jiang, Leilei Sun, Liangzhe Han, Jibin Wang, Weifeng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2603.26610)  

**Abstract**: Mobile devices continuously interact with cellular base stations, generating massive volumes of signaling records that provide broad coverage for understanding human mobility. However, such records offer only coarse location cues (e.g., serving-cell identifiers) and therefore limit their direct use in applications that require high-precision GPS trajectories. This paper studies the Sig2GPS problem: reconstructing GPS trajectories from cellular signaling. Inspired by domain experts often lay the signaling trace on the map and sketch the corresponding GPS route, unlike conventional solutions that rely on complex multi-stage engineering pipelines or regress coordinates, Sig2GPS is reframed as an image-to-video generation task that directly operates in the map-visual domain: signaling traces are rendered on a map, and a video generation model is trained to draw a continuous GPS path. To support this paradigm, a paired signaling-to-trajectory video dataset is constructed to fine-tune an open-source video model, and a trajectory-aware reinforcement learning-based optimization method is introduced to improve generation fidelity via rewards. Experiments on large-scale real-world datasets show substantial improvements over strong engineered and learning-based baselines, while additional results on next GPS prediction indicate scalability and cross-city transferability. Overall, these results suggest that map-visual video generation provides a practical interface for trajectory data mining by enabling direct generation and refinement of continuous paths under map constraints. 

---
# Sustainability Is Not Linear: Quantifying Performance, Energy, and Privacy Trade-offs in On-Device Intelligence 

**Authors**: Eziyo Ehsani, Luca Giamattei, Ivano Malavolta, Roberto Pietrantuono  

**Link**: [PDF](https://arxiv.org/pdf/2603.26603)  

**Abstract**: The migration of Large Language Models (LLMs) from cloud clusters to edge devices promises enhanced privacy and offline accessibility, but this transition encounters a harsh reality: the physical constraints of mobile batteries, thermal limits, and, most importantly, memory constraints. To navigate this landscape, we constructed a reproducible experimental pipeline to profile the complex interplay between energy consumption, latency, and quality. Unlike theoretical studies, we captured granular power metrics across eight models ranging from 0.5B to 9B parameters without requiring root access, ensuring our findings reflect realistic user conditions. We harness this pipeline to conduct an empirical case study on a flagship Android device, the Samsung Galaxy S25 Ultra, establishing foundational hypotheses regarding the trade-offs between generation quality, performance, and resource consumption. Our investigation uncovered a counter-intuitive quantization-energy paradox. While modern importance-aware quantization successfully reduces memory footprints to fit larger models into RAM, we found it yields negligible energy savings compared to standard mixed-precision methods. This proves that for battery life, the architecture of the model, not its quantization scheme, is the decisive factor. We further identified that Mixture-of-Experts (MoE) architectures defy the standard size-energy trend, offering the storage capacity of a 7B model while maintaining the lower energy profile of a 1B to 2B model. Finally, an analysis of these multi-objective trade-offs reveals a pragmatic sweet spot of mid-sized models, such as Qwen2.5-3B, that effectively balance response quality with sustainable energy consumption. 

---
# Evaluating Interactive 2D Visualization as a Sample Selection Strategy for Biomedical Time-Series Data Annotation 

**Authors**: Einari Vaaras, Manu Airaksinen, Okko Räsänen  

**Link**: [PDF](https://arxiv.org/pdf/2603.26592)  

**Abstract**: Reliable machine-learning models in biomedical settings depend on accurate labels, yet annotating biomedical time-series data remains challenging. Algorithmic sample selection may support annotation, but evidence from studies involving real human annotators is scarce. Consequently, we compare three sample selection methods for annotation: random sampling (RND), farthest-first traversal (FAFT), and a graphical user interface-based method enabling exploration of complementary 2D visualizations (2DVs) of high-dimensional data. We evaluated the methods across four classification tasks in infant motility assessment (IMA) and speech emotion recognition (SER). Twelve annotators, categorized as experts or non-experts, performed data annotation under a limited annotation budget, and post-annotation experiments were conducted to evaluate the sampling methods. Across all classification tasks, 2DV performed best when aggregating labels across annotators. In IMA, 2DV most effectively captured rare classes, but also exhibited greater annotator-to-annotator label distribution variability resulting from the limited annotation budget, decreasing classification performance when models were trained on individual annotators' labels; in these cases, FAFT excelled. For SER, 2DV outperformed the other methods among expert annotators and matched their performance for non-experts in the individual-annotator setting. A failure risk analysis revealed that RND was the safest choice when annotator count or annotator expertise was uncertain, whereas 2DV had the highest risk due to its greater label distribution variability. Furthermore, post-experiment interviews indicated that 2DV made the annotation task more interesting and enjoyable. Overall, 2DV-based sampling appears promising for biomedical time-series data annotation, particularly when the annotation budget is not highly constrained. 

---
# Generation Is Compression: Zero-Shot Video Coding via Stochastic Rectified Flow 

**Authors**: Ziyue Zeng, Xun Su, Haoyuan Liu, Bingyu Lu, Yui Tatsumi, Hiroshi Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2603.26571)  

**Abstract**: Existing generative video compression methods use generative models only as post-hoc reconstruction modules atop conventional codecs. We propose \emph{Generative Video Codec} (GVC), a zero-shot framework that turns a pretrained video generative model into the codec itself: the transmitted bitstream directly specifies the generative decoding trajectory, with no retraining required. To enable this, we convert the deterministic rectified-flow ODE of modern video foundation models into an equivalent SDE at inference time, unlocking per-step stochastic injection points for codebook-driven compression. Building on this unified backbone, we instantiate three complementary conditioning strategies -- \emph{Image-to-Video} (I2V) with adaptive tail-frame atom allocation, \emph{Text-to-Video} (T2V) operating at near-zero side information as a pure generative prior, and \emph{First-Last-Frame-to-Video} (FLF2V) with boundary-sharing GOP chaining for dual-anchor temporal control. Together, these variants span a principled trade-off space between spatial fidelity, temporal coherence, and compression efficiency. Experiments on standard benchmarks show that GVC achieves high-quality reconstruction below 0.002\,bpp while supporting flexible bitrate control through a single hyperparameter. 

---
# Beyond Code Snippets: Benchmarking LLMs on Repository-Level Question Answering 

**Authors**: Yoseph Berhanu Alebachew, Hunter Leary, Swanand Vaishampayan, Chris Brown  

**Link**: [PDF](https://arxiv.org/pdf/2603.26567)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities across software engineering tasks, including question answering (QA). However, most studies and benchmarks focus on isolated functions or single-file snippets, overlooking the challenges of real-world program comprehension, which often spans multiple files and system-level dependencies. In this work, we introduce StackRepoQA, the first multi-project, repository-level question answering dataset constructed from 1,318 real developer questions and accepted answers across 134 open-source Java projects. Using this dataset, we systematically evaluate two widely used LLMs (Claude 3.5 Sonnet and GPT-4o) under both direct prompting and agentic configurations. We compare baseline performance with retrieval-augmented generation methods that leverage file-level retrieval and graph-based representations of structural dependencies. Our results show that LLMs achieve moderate accuracy at baseline, with performance improving when structural signals are incorporated. Nonetheless, overall accuracy remains limited for repository-scale comprehension. The analysis reveals that high scores often result from verbatim reproduction of Stack Overflow answers rather than genuine reasoning. To our knowledge, this is the first empirical study to provide such evidence in repository-level QA. We release StackRepoQA to encourage further research into benchmarks, evaluation protocols, and augmentation strategies that disentangle memorization from reasoning, advancing LLMs as reliable tool for repository-scale program comprehension. 

---
# When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models 

**Authors**: Juan Gabriel Kostelec, Xiang Wang, Axel Laborieux, Christos Sourmpis, Qinghai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.26556)  

**Abstract**: Converting a pretrained Transformer into a more efficient hybrid model through distillation offers a promising approach to reducing inference costs. However, achieving high-quality generation in distilled models requires careful joint design of both the student architecture and the distillation process. Many prior distillation works evaluate downstream multiple-choice benchmarks by ranking candidate answers with log-likelihood rather than requiring autoregressive generation, which can obscure important differences in model quality. For example, we show that a 7B parameter distilled model that nearly matches its teacher to within 0.2\,pp under log-likelihood scoring actually falls behind by 20.8\,pp when the model must generate answers autoregressively.
We propose a Hybrid Kimi Delta Attention (Hybrid-KDA) architecture paired with GenDistill, a multi-stage distillation pipeline, and use generation-based evaluation throughout to guide design decisions. Applying this approach to Qwen3-0.6B, we systematically ablate six design axes: training objective, loss masking, training duration, dataset selection, parameter freezing, and architecture choice. We find that log-likelihood-based evaluation consistently underestimates the gap between teacher and student, and can in some cases reverse the ranking of design choices, meaning that conclusions drawn from perplexity-only evaluation may be misleading. Among the factors we study, dataset selection, completion-only masking, and freezing attention layers during post-training have the largest impact on generation quality.
Our best Hybrid-KDA model retains 86--90\% of teacher accuracy on knowledge benchmarks while reducing KV cache memory by up to 75\% and improving time-to-first-token by 2--4$\times$ at 128K-token contexts. 

---
# Beyond MACs: Hardware Efficient Architecture Design for Vision Backbones 

**Authors**: Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni  

**Link**: [PDF](https://arxiv.org/pdf/2603.26551)  

**Abstract**: Vision backbone networks play a central role in modern computer vision. Enhancing their efficiency directly benefits a wide range of downstream applications. To measure efficiency, many publications rely on MACs (Multiply Accumulate operations) as a predictor of execution time. In this paper, we experimentally demonstrate the shortcomings of such a metric, especially in the context of edge devices. By contrasting the MAC count and execution time of common architectural design elements, we identify key factors for efficient execution and provide insights to optimize backbone design. Based on these insights, we present LowFormer, a novel vision backbone family. LowFormer features a streamlined macro and micro design that includes Lowtention, a lightweight alternative to Multi-Head Self-Attention. Lowtention not only proves more efficient, but also enables superior results on ImageNet. Additionally, we present an edge GPU version of LowFormer, that can further improve upon its baseline's speed on edge GPU and desktop GPU. We demonstrate LowFormer's wide applicability by evaluating it on smaller image classification datasets, as well as adapting it to several downstream tasks, such as object detection, semantic segmentation, image retrieval, and visual object tracking. LowFormer models consistently achieve remarkable speed-ups across various hardware platforms compared to recent state-of-the-art backbones. Code and models are available at this https URL. 

---
# The Multi-AMR Buffer Storage, Retrieval, and Reshuffling Problem: Exact and Heuristic Approaches 

**Authors**: Max Disselnmeyer, Thomas Bömer, Laura Dörr, Bastian Amberg, Anne Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2603.26542)  

**Abstract**: Buffer zones are essential in production systems to decouple sequential processes. In dense floor storage environments, such as space-constrained brownfield facilities, manual operation is increasingly challenged by severe labor shortages and rising operational costs. Automating these zones requires solving the Buffer Storage, Retrieval, and Reshuffling Problem (BSRRP). While previous work has addressed scenarios where the focus is limited to reshuffling and retrieving a fixed set of items, real-world manufacturing necessitates an adaptive approach that also incorporates arriving unit loads. This paper introduces the Multi-AMR BSRRP, coordinating a robot fleet to manage concurrent reshuffling, alongside time-windowed storage and retrieval tasks, within a shared floor area. We formulate a Binary Integer Programming (IP) model to obtain exact solutions for benchmarking purposes. As the problem is NP-hard, rendering exact methods computationally intractable for industrial scales, we propose a hierarchical heuristic. This approach decomposes the problem into an A* search for task-level sequence planning of unit load placements, and a Constraint Programming (CP) approach for multi-robot coordination and scheduling. Experiments demonstrate orders-of-magnitude computation time reductions compared to the exact formulation. These results confirm the heuristic's viability as responsive control logic for high-density production environments. 

---
# How Open Must Language Models be to Enable Reliable Scientific Inference? 

**Authors**: James A. Michaelov, Catherine Arnett, Tyler A. Chang, Pamela D. Rivière, Samuel M. Taylor, Cameron R. Jones, Sean Trott, Roger P. Levy, Benjamin K. Bergen, Micah Altman  

**Link**: [PDF](https://arxiv.org/pdf/2603.26539)  

**Abstract**: How does the extent to which a model is open or closed impact the scientific inferences that can be drawn from research that involves it? In this paper, we analyze how restrictions on information about model construction and deployment threaten reliable inference. We argue that current closed models are generally ill-suited for scientific purposes, with some notable exceptions, and discuss ways in which the issues they present to reliable inference can be resolved or mitigated. We recommend that when models are used in research, potential threats to inference should be systematically identified along with the steps taken to mitigate them, and that specific justifications for model selection should be provided. 

---
# ALBA: A European Portuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative LLMs 

**Authors**: Inês Vieira, Inês Calvo, Iago Paulo, James Furtado, Rafael Ferreira, Diogo Tavares, Diogo Glória-Silva, David Semedo, João Magalhães  

**Link**: [PDF](https://arxiv.org/pdf/2603.26516)  

**Abstract**: As Large Language Models (LLMs) expand across multilingual domains, evaluating their performance in under-represented languages becomes increasingly important. European Portuguese (pt-PT) is particularly affected, as existing training data and benchmarks are mainly in Brazilian Portuguese (pt-BR). To address this, we introduce ALBA, a linguistically grounded benchmark designed from the ground up to assess LLM proficiency in linguistic-related tasks in pt-PT across eight linguistic dimensions, including Language Variety, Culture-bound Semantics, Discourse Analysis, Word Plays, Syntax, Morphology, Lexicology, and Phonetics and Phonology. ALBA is manually constructed by language experts and paired with an LLM-as-a-judge framework for scalable evaluation of pt-PT generated language. Experiments on a diverse set of models reveal performance variability across linguistic dimensions, highlighting the need for comprehensive, variety-sensitive benchmarks that support further development of tools in pt-PT. 

---
# JAL-Turn: Joint Acoustic-Linguistic Modeling for Real-Time and Robust Turn-Taking Detection in Full-Duplex Spoken Dialogue Systems 

**Authors**: Guangzhao Yang, Yu Pan, Shi Qiu, Ningjie Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.26515)  

**Abstract**: Despite recent advances, efficient and robust turn-taking detection remains a significant challenge in industrial-grade Voice AI agent deployments. Many existing systems rely solely on acoustic or semantic cues, leading to suboptimal accuracy and stability, while recent attempts to endow large language models with full-duplex capabilities require costly full-duplex data and incur substantial training and deployment overheads, limiting real-time performance. In this paper, we propose JAL-Turn, a lightweight and efficient speech-only turn-taking framework that adopts a joint acoustic-linguistic modeling paradigm, in which a cross-attention module adaptively integrates pre-trained acoustic representations with linguistic features to support low-latency prediction of hold vs shift states. By sharing a frozen ASR encoder, JAL-Turn enables turn-taking prediction to run fully in parallel with speech recognition, introducing no additional end-to-end latency or computational overhead. In addition, we introduce a scalable data construction pipeline that automatically derives reliable turn-taking labels from large-scale real-world dialogue corpora. Extensive experiments on public multilingual benchmarks and an in-house Japanese customer-service dataset show that JAL-Turn consistently outperforms strong state-of-the-art baselines in detection accuracy while maintaining superior real-time performance. 

---
# AMALIA Technical Report: A Fully Open Source Large Language Model for European Portuguese 

**Authors**: Afonso Simplício, Gonçalo Vinagre, Miguel Moura Ramos, Diogo Tavares, Rafael Ferreira, Giuseppe Attanasio, Duarte M. Alves, Inês Calvo, Inês Vieira, Rui Guerra, James Furtado, Beatriz Canaverde, Iago Paulo, Vasco Ramos, Diogo Glória-Silva, Miguel Faria, Marcos Treviso, Daniel Gomes, Pedro Gomes, David Semedo, André Martins, João Magalhães  

**Link**: [PDF](https://arxiv.org/pdf/2603.26511)  

**Abstract**: Despite rapid progress in open large language models (LLMs), European Portuguese (pt-PT) remains underrepresented in both training data and native evaluation, with machine-translated benchmarks likely missing the variant's linguistic and cultural nuances. We introduce AMALIA, a fully open LLM that prioritizes pt-PT by using more high-quality pt-PT data during both the mid- and post-training stages. To evaluate pt-PT more faithfully, we release a suite of pt-PT benchmarks that includes translated standard tasks and four new datasets targeting pt-PT generation, linguistic competence, and pt-PT/pt-BR bias. Experiments show that AMALIA matches strong baselines on translated benchmarks while substantially improving performance on pt-PT-specific evaluations, supporting the case for targeted training and native benchmarking for European Portuguese. 

---
# Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference 

**Authors**: Konstantinos Papaioannou, Thaleia Dimitra Doudali  

**Link**: [PDF](https://arxiv.org/pdf/2603.26498)  

**Abstract**: Multimodal Large Language Models (MLLMs) power platforms like ChatGPT, Gemini, and Copilot, enabling richer interactions with text, images, and videos. These heterogeneous workloads introduce additional inference stages, such as vision preprocessing and encoding, that inflate latency and memory demand. Existing LLM serving systems, optimized for text-only workloads, fail under multimodality: large requests (e.g., videos) monopolize resources, causing severe head-of-line blocking and performance degradation. Our key insight is that multimodal requests differ by orders of magnitude in resource demands, which we capture through a simple abstraction: videos behave like rocks, images like pebbles, and text like sand. We design RPS-Serve, a modality-aware scheduler that lets sand flow quickly through pebbles and rocks, ensuring interactive responsiveness while avoiding starvation. RPS-Serve classifies requests, prioritizes them dynamically, and applies aging to avoid starvation. Evaluation across state-of-the-art MLLMs shows that RPS-Serve reduces, on average, time-to-first-token (TTFT) by 54% overall, and by 78.5% for latency-critical requests, compared to current systems. RPS-Serve delivers LLM-like responsiveness for MLLMs, with modality-aware scheduling and by making the most efficient use of the available resources. 

---
# Foundation Model for Cardiac Time Series via Masked Latent Attention 

**Authors**: Moritz Vandenhirtz, Samuel Ruipérez-Campillo, Simon Böhi, Sonia Laguna, Irene Cannistraci, Andrea Agostini, Ece Ozkan, Thomas M. Sutter, Julia E. Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2603.26475)  

**Abstract**: Electrocardiograms (ECGs) are among the most widely available clinical signals and play a central role in cardiovascular diagnosis. While recent foundation models (FMs) have shown promise for learning transferable ECG representations, most existing pretraining approaches treat leads as independent channels and fail to explicitly leverage their strong structural redundancy. We introduce the latent attention masked autoencoder (LAMAE) FM that directly exploits this structure by learning cross-lead connection mechanisms during self-supervised pretraining. Our approach models higher-order interactions across leads through latent attention, enabling permutation-invariant aggregation and adaptive weighting of lead-specific representations. We provide empirical evidence on the Mimic-IV-ECG database that leveraging the cross-lead connection constitutes an effective form of structural supervision, improving representation quality and transferability. Our method shows strong performance in predicting ICD-10 codes, outperforming independent-lead masked modeling and alignment-based baselines. 

---
# UNIFERENCE: A Discrete Event Simulation Framework for Developing Distributed AI Models 

**Authors**: Doğaç Eldenk, Stephen Xia  

**Link**: [PDF](https://arxiv.org/pdf/2603.26469)  

**Abstract**: Developing and evaluating distributed inference algorithms remains difficult due to the lack of standardized tools for modeling heterogeneous devices and networks. Existing studies often rely on ad-hoc testbeds or proprietary infrastructure, making results hard to reproduce and limiting exploration of hypothetical hardware or network configurations. We present UNIFERENCE, a discrete-event simulation (DES) framework designed for developing, benchmarking, and deploying distributed AI models within a unified environment. UNIFERENCE models device and network behavior through lightweight logical processes that synchronize only on communication primitives, eliminating rollbacks while preserving the causal order. It integrates seamlessly with PyTorch Distributed, enabling the same codebase to transition from simulation to real deployment. Our evaluation demonstrates that UNIFERENCE profiles runtime with up to 98.6% accuracy compared to real physical deployments across diverse backends and hardware setups. By bridging simulation and deployment, UNIFERENCE provides an accessible, reproducible platform for studying distributed inference algorithms and exploring future system designs, from high-performance clusters to edge-scale devices. The framework is open-sourced at this https URL. 

---
# A Boltzmann-machine-enhanced Transformer For DNA Sequence Classification 

**Authors**: Zhixuan Cao, Yishu Xu, Xuang WU  

**Link**: [PDF](https://arxiv.org/pdf/2603.26465)  

**Abstract**: DNA sequence classification requires not only high predictive accuracy but also the ability to uncover latent site interactions, combinatorial regulation, and epistasis-like higher-order dependencies. Although the standard Transformer provides strong global modeling capacity, its softmax attention is continuous, dense, and weakly constrained, making it better suited for information routing than explicit structure discovery. In this paper, we propose a Boltzmann-machine-enhanced Transformer for DNA sequence classification. Built on multi-head attention, the model introduces structured binary gating variables to represent latent query-key connections and constrains them with a Boltzmann-style energy function. Query-key similarity defines local bias terms, learnable pairwise interactions capture synergy and competition between edges, and latent hidden units model higher-order combinatorial dependencies. Since exact posterior inference over discrete gating graphs is intractable, we use mean-field variational inference to estimate edge activation probabilities and combine it with Gumbel-Softmax to progressively compress continuous probabilities into near-discrete gates while preserving end-to-end differentiability. During training, we jointly optimize classification and energy losses, encouraging the model to achieve accurate prediction while favoring low-energy, stable, and interpretable structures. We further derive the framework from the energy function and variational free energy to the mean-field fixed-point equations, Gumbel-Softmax relaxation, and the final joint objective. The proposed framework provides a unified view of integrating Boltzmann machines, differentiable discrete optimization, and Transformers for structured learning on biological sequences. 

---
# Neuro-Symbolic Process Anomaly Detection 

**Authors**: Devashish Gaikwad, Wil M. P. van der Aalst, Gyunam Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.26461)  

**Abstract**: Process anomaly detection is an important application of process mining for identifying deviations from the normal behavior of a process. Neural network-based methods have recently been applied to this task, learning directly from event logs without requiring a predefined process model. However, since anomaly detection is a purely statistical task, these models fail to incorporate human domain knowledge. As a result, rare but conformant traces are often misclassified as anomalies due to their low frequency, which limits the effectiveness of the detection process. Recent developments in the field of neuro-symbolic AI have introduced Logic Tensor Networks (LTN) as a means to integrate symbolic knowledge into neural networks using real-valued logic. In this work, we propose a neuro-symbolic approach that integrates domain knowledge into neural anomaly detection using LTN and Declare constraints. Using autoencoder models as a foundation, we encode Declare constraints as soft logical guiderails within the learning process to distinguish between anomalous and rare but conformant behavior. Evaluations on synthetic and real-world datasets demonstrate that our approach improves F1 scores even when as few as 10 conformant traces exist, and that the choice of Declare constraint and by extension human domain knowledge significantly influences performance gains. 

---
# Can AI Models Direct Each Other? Organizational Structure as a Probe into Training Limitations 

**Authors**: Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.26458)  

**Abstract**: Can an expensive AI model effectively direct a cheap one to solve software engineering tasks? We study this question by introducing ManagerWorker, a two-agent pipeline where an expensive "manager" model (text-only, no code execution) analyzes issues, dispatches exploration tasks, and reviews implementations, while a cheap "worker" model (with full repo access) executes code changes. We evaluate on 200 instances from SWE-bench Lite across five configurations that vary the manager-worker relationship, pipeline complexity, and model pairing. Our findings reveal both the promise and the limits of multi-agent direction: (1) a strong manager directing a weak worker (62%) matches a strong single agent (60%) at a fraction of the strong-model token usage, showing that expensive reasoning can substitute for expensive execution; (2) a weak manager directing a weak worker (42%) performs worse than the weak agent alone (44%), demonstrating that the directing relationship requires a genuine capability gap--structure without substance is pure overhead; (3) the manager's value lies in directing, not merely reviewing--a minimal review-only loop adds just 2pp over the baseline, while structured exploration and planning add 11pp, showing that active direction is what makes the capability gap productive; and (4) these behaviors trace to a single root cause: current models are trained as monolithic agents, and splitting them into director/worker roles fights their training distribution. The pipeline succeeds by designing around this mismatch--keeping each model close to its trained mode (text generation for the manager, tool use for the worker) and externalizing organizational structure to code. This diagnosis points to concrete training gaps: delegation, scoped execution, and mode switching are skills absent from current training data. 

---
# CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities 

**Authors**: Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni  

**Link**: [PDF](https://arxiv.org/pdf/2603.26425)  

**Abstract**: Recent research on vision backbone architectures has predominantly focused on optimizing efficiency for hardware platforms with high parallel processing capabilities. This category increasingly includes embedded systems such as mobile phones and embedded AI accelerator modules. In contrast, CPUs do not have the possibility to parallelize operations in the same manner, wherefore models benefit from a specific design philosophy that balances amount of operations (MACs) and hardware-efficient execution by having high MACs per second (MACpS). In pursuit of this, we investigate two modifications to standard convolutions, aimed at reducing computational cost: grouping convolutions and reducing kernel sizes. While both adaptations substantially decrease the total number of MACs required for inference, sustaining low latency necessitates preserving hardware-efficiency. Our experiments across diverse CPU devices confirm that these adaptations successfully retain high hardware-efficiency on CPUs. Based on these insights, we introduce CPUBone, a new family of vision backbone models optimized for CPU-based inference. CPUBone achieves state-of-the-art Speed-Accuracy Trade-offs (SATs) across a wide range of CPU devices and effectively transfers its efficiency to downstream tasks such as object detection and semantic segmentation. Models and code are available at this https URL. 

---
# KMM-CP: Practical Conformal Prediction under Covariate Shift via Selective Kernel Mean Matching 

**Authors**: Siddhartha Laghuvarapu, Rohan Deb, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.26415)  

**Abstract**: Uncertainty quantification is essential for deploying machine learning models in high-stakes domains such as scientific discovery and healthcare. Conformal Prediction (CP) provides finite-sample coverage guarantees under exchangeability, an assumption often violated in practice due to distribution shift. Under covariate shift, restoring validity requires importance weighting, yet accurate density-ratio estimation becomes unstable when training and test distributions exhibit limited support overlap. We propose KMM-CP, a conformal prediction framework based on Kernel Mean Matching (KMM) for covariate-shift correction. We show that KMM directly controls the bias-variance components governing conformal coverage error by minimizing RKHS moment discrepancy under explicit weight constraints, and establish asymptotic coverage guarantees under mild conditions. We then introduce a selective extension that identifies regions of reliable support overlap and restricts conformal correction to this subset, further improving stability in low-overlap regimes. Experiments on molecular property prediction benchmarks with realistic distribution shifts show that KMM-CP reduces coverage gap by over 50% compared to existing approaches. The code is available at this https URL. 

---
# Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence Between Thinking Tokens and Answers in Open-Weight Reasoning Models 

**Authors**: Richard J. Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.26410)  

**Abstract**: Extended-thinking models expose a second text-generation channel ("thinking tokens") alongside the user-visible answer. This study examines 12 open-weight reasoning models on MMLU and GPQA questions paired with misleading hints. Among the 10,506 cases where models actually followed the hint (choosing the hint's target over the ground truth), each case is classified by whether the model acknowledges the hint in its thinking tokens, its answer text, both, or neither. In 55.4% of these cases the model's thinking tokens contain hint-related keywords that the visible answer omits entirely, a pattern termed *thinking-answer divergence*. The reverse (answer-only acknowledgment) is near-zero (0.5%), confirming that the asymmetry is directional. Hint type shapes the pattern sharply: sycophancy is the most *transparent* hint, with 58.8% of sycophancy-influenced cases acknowledging the professor's authority in both channels, while consistency (72.2%) and unethical (62.7%) hints are dominated by thinking-only acknowledgment. Models also vary widely, from near-total divergence (Step-3.5-Flash: 94.7%) to relative transparency (Qwen3.5-27B: 19.6%). These results show that answer-text-only monitoring misses more than half of all hint-influenced reasoning and that thinking-token access, while necessary, still leaves 11.8% of cases with no verbalized acknowledgment in either channel. 

---
# Generative Modeling in Protein Design: Neural Representations, Conditional Generation, and Evaluation Standards 

**Authors**: Senura Hansaja Wanasekara, Minh-Duong Nguyen, Xiaochen Liu, Nguyen H. Tran, Ken-Tye Yong  

**Link**: [PDF](https://arxiv.org/pdf/2603.26378)  

**Abstract**: Generative modeling has become a central paradigm in protein research, extending machine learning beyond structure prediction toward sequence design, backbone generation, inverse folding, and biomolecular interaction modeling. However, the literature remains fragmented across representations, model classes, and task formulations, making it difficult to compare methods or identify appropriate evaluation standards. This survey provides a systematic synthesis of generative AI in protein research, organized around (i) foundational representations spanning sequence, geometric, and multimodal encodings; (ii) generative architectures including $\mathrm{SE}(3)$-equivariant diffusion, flow matching, and hybrid predictor-generator systems; and (iii) task settings from structure prediction and de novo design to protein-ligand and protein-protein interactions. Beyond cataloging methods, we compare assumptions, conditioning mechanisms, and controllability, and we synthesize evaluation best practices that emphasize leakage-aware splits, physical validity checks, and function-oriented benchmarks. We conclude with critical open challenges: modeling conformational dynamics and intrinsically disordered regions, scaling to large assemblies while maintaining efficiency, and developing robust safety frameworks for dual-use biosecurity risks. By unifying architectural advances with practical evaluation standards and responsible development considerations, this survey aims to accelerate the transition from predictive modeling to reliable, function-driven protein engineering. 

---
# Automated near-term quantum algorithm discovery for molecular ground states 

**Authors**: Fabian Finger, Frederic Rapp, Pranav Kalidindi, Kerry He, Kante Yin, Alexander Koziell-Pipe, David Zsolt Manrique, Gabriel Greene-Diniz, Stephen Clark, Hamza Fawzi, Bernardino Romera Paredes, Alhussein Fawzi, Konstantinos Meichanetzidis  

**Link**: [PDF](https://arxiv.org/pdf/2603.26359)  

**Abstract**: Designing quantum algorithms is a complex and counterintuitive task, making it an ideal candidate for AI-driven algorithm discovery. To this end, we employ the Hive, an AI platform for program synthesis, which utilises large language models to drive a highly distributed evolutionary process for discovering new algorithms. We focus on the ground state problem in quantum chemistry, and discover efficient quantum heuristic algorithms that solve it for molecules LiH, H2O, and F2 while exhibiting significant reductions in quantum resources relative to state-of-the-art near-term quantum algorithms. Further, we perform an interpretability study on the discovered algorithms and identify the key functions responsible for the efficiency gains. Finally, we benchmark the Hive-discovered circuits on the Quantinuum System Model H2 quantum computer and identify minimum system requirements for chemical precision. We envision that this novel approach to quantum algorithm discovery applies to other domains beyond chemistry, as well as to designing quantum algorithms for fault-tolerant quantum computers. 

---
# Generative Score Inference for Multimodal Data 

**Authors**: Xinyu Tian, Xiaotong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.26349)  

**Abstract**: Accurate uncertainty quantification is crucial for making reliable decisions in various supervised learning scenarios, particularly when dealing with complex, multimodal data such as images and text. Current approaches often face notable limitations, including rigid assumptions and limited generalizability, constraining their effectiveness across diverse supervised learning tasks. To overcome these limitations, we introduce Generative Score Inference (GSI), a flexible inference framework capable of constructing statistically valid and informative prediction and confidence sets across a wide range of multimodal learning problems. GSI utilizes synthetic samples generated by deep generative models to approximate conditional score distributions, facilitating precise uncertainty quantification without imposing restrictive assumptions about the data or tasks. We empirically validate GSI's capabilities through two representative scenarios: hallucination detection in large language models and uncertainty estimation in image captioning. Our method achieves state-of-the-art performance in hallucination detection and robust predictive uncertainty in image captioning, and its performance is positively influenced by the quality of the underlying generative model. These findings underscore the potential of GSI as a versatile inference framework, significantly enhancing uncertainty quantification and trustworthiness in multimodal learning. 

---
# Reflect to Inform: Boosting Multimodal Reasoning via Information-Gain-Driven Verification 

**Authors**: Shuai Lv, Chang Liu, Feng Tang, Yujie Yuan, Aojun Zhou, Kui Zhang, Xi Yang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.26348)  

**Abstract**: Multimodal Large Language Models (MLLMs) achieve strong multimodal reasoning performance, yet we identify a recurring failure mode in long-form generation: as outputs grow longer, models progressively drift away from image evidence and fall back on textual priors, resulting in ungrounded reasoning and hallucinations. Interestingly, Based on attention analysis, we find that MLLMs have a latent capability for late-stage visual verification that is present but not consistently activated. Motivated by this observation, we propose Visual Re-Examination (VRE), a self-evolving training framework that enables MLLMs to autonomously perform visual introspection during reasoning without additional visual inputs. Rather than distilling visual capabilities from a stronger teacher, VRE promotes iterative self-improvement by leveraging the model itself to generate reflection traces, making visual information actionable through information gain. Extensive experiments across diverse multimodal benchmarks demonstrate that VRE consistently improves reasoning accuracy and perceptual reliability, while substantially reducing hallucinations, especially in long-chain settings. Code is available at this https URL. 

---
# CALRK-Bench: Evaluating Context-Aware Legal Reasoning in Korean Law 

**Authors**: JiHyeok Jung, TaeYoung Yoon, HyunSouk Cho  

**Link**: [PDF](https://arxiv.org/pdf/2603.26332)  

**Abstract**: Legal reasoning requires not only the application of legal rules but also an understanding of the context in which those rules operate. However, existing legal benchmarks primarily evaluate rule application under the assumption of fixed norms, and thus fail to capture situations where legal judgments shift or where multiple norms interact. In this work, we propose CALRK-Bench, a context-aware legal reasoning benchmark based on the legal system in Korean. CALRK-Bench evaluates whether models can identify the temporal validity of legal norms, determine whether sufficient legal information is available for a given case, and understand the reasons behind shifts in legal judgments. The dataset is constructed from legal precedents and legal consultation records, and is validated by legal experts. Experimental results show that even recent large language models consistently exhibit low performance on these three tasks. CALRK-Bench provides a new stress test for evaluating context-aware legal reasoning rather than simple memorization of legal knowledge. Our code is available at this https URL. 

---
# Mitigating the Reasoning Tax in Vision-Language Fine-Tuning with Input-Adaptive Depth Aggregation 

**Authors**: Yiming Ren, Yujiu Yang, Junjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26330)  

**Abstract**: Supervised fine-tuning (SFT) on visual instruction data often improves perceptual capabilities in vision-language models (VLMs) while degrading reasoning performance, creating a persistent reasoning tax during post-training. We investigate whether this degradation is related to disrupted access to depth-wise representations, and find that even fixed cross-depth aggregation substantially restores reasoning, suggesting that preserved cross-depth access is an important missing factor in VLM fine-tuning. Building on this observation, we propose Input-Adaptive Depth Aggregation (IADA), a lightweight mechanism that makes cross-depth retrieval input-adaptive, modality-aware, and efficiently parameterized through a low-rank bottleneck. On Qwen3-VL-2B, IADA improves the average reasoning score by 9.5 points and the average perception score by $3.3$ points over LoRA-only fine-tuning with only 0.14M additional parameters, with the strongest gains appearing in parameter-efficient low-rank settings. 

---
# PRISMA: Toward a Normative Information Infrastructure for Responsible Pharmaceutical Knowledge Management 

**Authors**: Eugenio Rodrigo Zimmer Neves, Amanda Vanon Correa, Camila Campioni, Gabielli Pare Guglielmi, Bruno Morelli  

**Link**: [PDF](https://arxiv.org/pdf/2603.26324)  

**Abstract**: Most existing approaches to AI in pharmacy collapse three epistemologically distinct operations into a single technical layer: document preservation, semantic interpretation, and contextual presentation. This conflation is a root cause of recurring fragilities including loss of provenance, interpretive opacity, alert fatigue, and erosion of accountability. This paper proposes the PATOS--Lector--PRISMA (PLP) infrastructure as a normative information architecture for responsible pharmaceutical knowledge management. PATOS preserves regulatory documents with explicit versioning and provenance; Lector implements machine-assisted reading with human curation, producing typed assertions anchored to primary sources; PRISMA delivers contextual presentation through the RPDA framework (Regulatory, Prescription, Dispensing, Administration), refracting the same informational core into distinct professional views. The architecture introduces the Evidence Pack as a formal unit of accountable assertion (versioned, traceable, epistemically bounded, and curatorially validated), with assertions typified by illocutionary force. A worked example traces dipyrone monohydrate across all three layers using real system data. Developed and validated in Brazil's regulatory context, the architecture is grounded in an operational implementation comprising over 16,000 official documents and 38 curated Evidence Packs spanning five reference medications. The proposal is demonstrated as complementary to operational decision support systems, providing infrastructural conditions that current systems lack: documentary anchoring, interpretive transparency, and institutional accountability. 

---
# From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs 

**Authors**: Jiyuan An, Liner Yang, Mengyan Wang, Luming Lu, Weihua An, Erhong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26323)  

**Abstract**: As spatial intelligence becomes an increasingly important capability for foundation models, it remains unclear whether large language models' (LLMs) performance on spatial reasoning benchmarks reflects structured internal spatial representations or reliance on linguistic heuristics. We address this question from a mechanistic perspective by examining how spatial information is internally represented and used. Drawing on computational theories of human spatial cognition, we decompose spatial reasoning into three primitives, relational composition, representational transformation, and stateful spatial updating, and design controlled task families for each. We evaluate multilingual LLMs in English, Chinese, and Arabic under single pass inference, and analyze internal representations using linear probing, sparse autoencoder based feature analysis, and causal interventions. We find that task relevant spatial information is encoded in intermediate layers and can causally influence behavior, but these representations are transient, fragmented across task families, and weakly integrated into final predictions. Cross linguistic analysis further reveals mechanistic degeneracy, where similar behavioral performance arises from distinct internal pathways. Overall, our results suggest that current LLMs exhibit limited and context dependent spatial representations rather than robust, general purpose spatial reasoning, highlighting the need for mechanistic evaluation beyond benchmark accuracy. 

---
# Label-Free Cross-Task LoRA Merging with Null-Space Compression 

**Authors**: Wonyoung Lee, Wooseong Jeong, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2603.26317)  

**Abstract**: Model merging combines independently fine-tuned checkpoints without joint multi-task training. In the era of foundation-model, fine-tuning with Low-Rank Adaptation (LoRA) is prevalent, making LoRA merging a promising target. Existing approaches can work in homogeneous settings where all target tasks are classification but often fail when tasks span classification and regression. Approaches using entropy-based surrogates do not apply to regression and are costly for large language models due to long token sequences. We introduce Null-Space Compression (NSC) Merging, a label-free, output-agnostic method that sets merge weights from adapter geometry. Our key observation is that during LoRA finetuning the down-projection factor $A$ in $\Delta W = BA$ compresses its null space, and the compression correlates with performance. NSC uses this as an optimization signal for merging that can generalize across classification, regression, and sequence generation. NSC achieves state-of-the-art performance across twenty heterogeneous vision tasks with balanced gains where prior methods overfit subsets of tasks. It also outperforms baselines on six NLI benchmarks and on vision-language evaluations for VQA and image captioning, demonstrating scalability and effectiveness. 

---
# Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy 

**Authors**: Wooseong Jeong, Wonyoung Lee, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2603.26299)  

**Abstract**: Merging multiple Low-Rank Adaptation (LoRA) modules is promising for constructing general-purpose systems, yet challenging because LoRA update directions span different subspaces and contribute unevenly. When merged naively, such mismatches can weaken the directions most critical to certain task losses while overemphasizing relatively less important ones, ultimately reducing the model's ability to represent all tasks faithfully. We revisit this problem through two perspectives: subspace coverage, which captures how broadly LoRA directions cover diverse representational directions, and anisotropy, which reflects the imbalance of influence across those directions. We propose TARA-Merging (Task-Rank Anisotropy Alignment), which aligns merging weights using a preference-weighted cross-entropy pseudo-loss while preserving task-relevant LoRA subspaces. This ensures broad subspace coverage and mitigates anisotropy via direction-wise reweighting. Across eight vision and six NLI benchmarks, TARA-Merging consistently outperforms vanilla and LoRA-aware baselines, demonstrating strong robustness and generalization, and highlighting the importance of addressing both subspace coverage and anisotropy in LoRA merging. 

---
# findsylls: A Language-Agnostic Toolkit for Syllable-Level Speech Tokenization and Embedding 

**Authors**: Héctor Javier Vázquez Martínez  

**Link**: [PDF](https://arxiv.org/pdf/2603.26292)  

**Abstract**: Syllable-level units offer compact and linguistically meaningful representations for spoken language modeling and unsupervised word discovery, but research on syllabification remains fragmented across disparate implementations, datasets, and evaluation protocols. We introduce findsylls, a modular, language-agnostic toolkit that unifies classical syllable detectors and end-to-end syllabifiers under a common interface for syllable segmentation, embedding extraction, and multi-granular evaluation. The toolkit implements and standardizes widely used methods (e.g., Sylber, VG-HuBERT) and allows their components to be recombined, enabling controlled comparisons of representations, algorithms, and token rates. We demonstrate findsylls on English and Spanish corpora and on new hand-annotated data from Kono, an underdocumented Central Mande language, illustrating how a single framework can support reproducible syllable-level experiments across both high-resource and under-resourced settings. 

---
# PhysVid: Physics Aware Local Conditioning for Generative Video Models 

**Authors**: Saurabh, Pathak, Elahe Arani, Mykola Pechenizkiy, Bahram Zonooz  

**Link**: [PDF](https://arxiv.org/pdf/2603.26285)  

**Abstract**: Generative video models achieve high visual fidelity but often violate basic physical principles, limiting reliability in real-world settings. Prior attempts to inject physics rely on conditioning: frame-level signals are domain-specific and short-horizon, while global text prompts are coarse and noisy, missing fine-grained dynamics. We present PhysVid, a physics-aware local conditioning scheme that operates over temporally contiguous chunks of frames. Each chunk is annotated with physics-grounded descriptions of states, interactions, and constraints, which are fused with the global prompt via chunk-aware cross-attention during training. At inference, we introduce negative physics prompts (descriptions of locally relevant law violations) to steer generation away from implausible trajectories. On VideoPhy, PhysVid improves physical commonsense scores by $\approx 33\%$ over baseline video generators, and by up to $\approx 8\%$ on VideoPhy2. These results show that local, physics-aware guidance substantially increases physical plausibility in generative video and marks a step toward physics-grounded video models. 

---
# Knowdit: Agentic Smart Contract Vulnerability Detection with Auditing Knowledge Summarization 

**Authors**: Ziqiao Kong, Wanxu Xia, Chong Wang, Yi Lu, Pan Li, Shaohua Li, Zong Cao, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.26270)  

**Abstract**: Smart contracts govern billions of dollars in decentralized finance (DeFi), yet automated vulnerability detection remains challenging because many vulnerabilities are tightly coupled with project-specific business logic. We observe that recurring vulnerabilities across diverse DeFi business models often share the same underlying economic mechanisms, which we term DeFi semantics, and that capturing these shared abstractions can enable more systematic auditing. Building on this insight, we propose Knowdit, a knowledge-driven, agentic framework for smart contract vulnerability detection. Knowdit first constructs an auditing knowledge graph from historical human audit reports, linking fine-grained DeFi semantics with recurring vulnerability patterns. Given a new project, a multi-agent framework leverages this knowledge through an iterative loop of specification generation, harness synthesis, fuzz execution, and finding reflection, driven by a shared working memory for continuous refinement.
We evaluate Knowdit on 12 recent Code4rena projects with 75 ground-truth vulnerabilities. Knowdit detects all 14 high-severity and 77\% of medium-severity vulnerabilities with only 2 false positives, significantly outperforming all baselines. Applied to six real-world projects, Knowdit further discovers 12 high- and 10 medium-severity previously unknown vulnerabilities, proving its outstanding performance. 

---
# GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation 

**Authors**: Xujing Tao, Chuxin Wang, Yubo Ai, Zhixin Cheng, Zhuoyuan Li, Liangsheng Liu, Yujia Chen, Xinjun Li, Qiao Li, Wenfei Yang, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26260)  

**Abstract**: Open-vocabulary 3D semantic segmentation aims to segment arbitrary categories beyond the training set. Existing methods predominantly rely on distilling knowledge from 2D open-vocabulary models. However, aligning 3D features to the 2D representation space restricts intrinsic 3D geometric learning and inherits errors from 2D predictions. To address these limitations, we propose GeoGuide, a novel framework that leverages pretrained 3D models to integrate hierarchical geometry-semantic consistency for open-vocabulary 3D segmentation. Specifically, we introduce an Uncertainty-based Superpoint Distillation module to fuse geometric and semantic features for estimating per-point uncertainty, adaptively weighting 2D features within superpoints to suppress noise while preserving discriminative information to enhance local semantic consistency. Furthermore, our Instance-level Mask Reconstruction module leverages geometric priors to enforce semantic consistency within instances by reconstructing complete instance masks. Additionally, our Inter-Instance Relation Consistency module aligns geometric and semantic similarity matrices to calibrate cross-instance consistency for same-category objects, mitigating viewpoint-induced semantic drift. Extensive experiments on ScanNet v2, Matterport3D, and nuScenes demonstrate the superior performance of GeoGuide. 

---
# Working Notes on Late Interaction Dynamics: Analyzing Targeted Behaviors of Late Interaction Models 

**Authors**: Antoine Edy, Max Conti, Quentin Macé  

**Link**: [PDF](https://arxiv.org/pdf/2603.26259)  

**Abstract**: While Late Interaction models exhibit strong retrieval performance, many of their underlying dynamics remain understudied, potentially hiding performance bottlenecks. In this work, we focus on two topics in Late Interaction retrieval: a length bias that arises when using multi-vector scoring, and the similarity distribution beyond the best scores pooled by the MaxSim operator. We analyze these behaviors for state-of-the-art models on the NanoBEIR benchmark. Results show that while the theoretical length bias of causal Late Interaction models holds in practice, bi-directional models can also suffer from it in extreme cases. We also note that no significant similarity trend lies beyond the top-1 document token, validating that the MaxSim operator efficiently exploits the token-level similarity scores. 

---
# ARTA: Adaptive Mixed-Resolution Token Allocation for Efficient Dense Feature Extraction 

**Authors**: David Hagerman, Roman Naeem, Erik Brorsson, Fredrik Kahl, Lennart Svensson  

**Link**: [PDF](https://arxiv.org/pdf/2603.26258)  

**Abstract**: We present ARTA, a mixed-resolution coarse-to-fine vision transformer for efficient dense feature extraction. Unlike models that begin with dense high-resolution (fine) tokens, ARTA starts with low-resolution (coarse) tokens and uses a lightweight allocator to predict which regions require more fine tokens. The allocator iteratively predicts a semantic (class) boundary score and allocates additional tokens to patches above a low threshold, concentrating token density near boundaries while maintaining high sensitivity to weak boundary evidence. This targeted allocation encourages tokens to represent a single semantic class rather than a mixture of classes. Mixed-resolution attention enables interaction between coarse and fine tokens, focusing computation on semantically complex areas while avoiding redundant processing in homogeneous regions. Experiments demonstrate that ARTA achieves state-of-the-art results on ADE20K and COCO-Stuff with substantially fewer FLOPs, and delivers competitive performance on Cityscapes at markedly lower compute. For example, ARTA-Base attains 54.6 mIoU on ADE20K in the ~100M-parameter class while using fewer FLOPs and less memory than comparable backbones. 

---
# Channelling, Coordinating, Collaborating: A Three-Layer Framework for Disability-Centered Human-Agent Collaboration 

**Authors**: Lan Xiao, Catherine Holloway  

**Link**: [PDF](https://arxiv.org/pdf/2603.26252)  

**Abstract**: AI accessibility tools have mostly been designed for individual use, helping one person overcome a specific functional barrier. But for many people with disabilities, complex tasks are accomplished through collaboration with others who bring complementary abilities, not solitary effort. We propose a three-layer framework, Channelling, Coordinating, and Co-Creating, that rethinks AI's role in ability-diverse collaboration: establishing shared informational ground across abilities, mediating workflows between collaborators with different abilities, and contributing as a bounded partner toward shared goals. Grounded in the Ability-Diverse Collaboration framework, grounding theory, and Carlile's 3T framework, it extends the ``agents as remote collaborators'' vision by centring the collaborative, interdependent ways people with disabilities already work. 

---
# Automatic Speech Recognition for Documenting Endangered Languages: Case Study of Ikema Miyakoan 

**Authors**: Chihiro Taguchi, Yukinori Takubo, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26248)  

**Abstract**: Language endangerment poses a major challenge to linguistic diversity worldwide, and technological advances have opened new avenues for documentation and revitalization. Among these, automatic speech recognition (ASR) has shown increasing potential to assist in the transcription of endangered language data. This study focuses on Ikema, a severely endangered Ryukyuan language spoken in Okinawa, Japan, with approximately 1,300 remaining speakers, most of whom are over 60 years old. We present an ongoing effort to develop an ASR system for Ikema based on field recordings. Specifically, we (1) construct a {\totaldatasethours}-hour speech corpus from field recordings, (2) train an ASR model that achieves a character error rate as low as 15\%, and (3) evaluate the impact of ASR assistance on the efficiency of speech transcription. Our results demonstrate that ASR integration can substantially reduce transcription time and cognitive load, offering a practical pathway toward scalable, technology-supported documentation of endangered languages. 

---
# Distilling Conversations: Abstract Compression of Conversational Audio Context for LLM-based ASR 

**Authors**: Shashi Kumar, Esaú Villatoro-Tello, Sergio Burdisso, Kadri Hacioglu, Thibault Bañeras-Roux, Hasindri Watawana, Dairazalia Sanchez-Cortes, Srikanth Madikeri, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2603.26246)  

**Abstract**: Standard LLM-based speech recognition systems typically process utterances in isolation, limiting their ability to leverage conversational context. In this work, we study whether multimodal context from prior turns improves LLM-based ASR and how to represent that context efficiently. We find that, after supervised multi-turn training, conversational context mainly helps with the recognition of contextual entities. However, conditioning on raw context is expensive because the prior-turn audio token sequence grows rapidly with conversation length. To address this, we propose Abstract Compression, which replaces the audio portion of prior turns with a fixed number of learned latent tokens while retaining corresponding transcripts explicitly. On both in-domain and out-of-domain test sets, the compressed model recovers part of the gains of raw-context conditioning with a smaller prior-turn audio footprint. We also provide targeted analyses of the compression setup and its trade-offs. 

---
# Physics-Informed Neural Networks and Sequence Encoder: Application to heating and early cooling of thermo-stamping process 

**Authors**: Mouad Elaarabi, Domenico Borzacchiello, Philippe Le Bot, Nathan Lauzeral, Sebastien Comas-Cardona  

**Link**: [PDF](https://arxiv.org/pdf/2603.26245)  

**Abstract**: In a previous work (Elaarabi et al., 2025b), the Sequence Encoder for online dynamical system identification (Elaarabi et al., 2025a) and its combination with PINN (PINN-SE) were introduced and tested on both synthetic and real data case scenarios. The sequence encoder is able to effectively encode time series into feature vectors, which the PINN then uses to map to dynamical behavior, predicting system response under changes in parameters, ICs and BCs. Previously (Elaarabi et al., 2025b), the tests on real data were limited to simple 1D problems and only 1D time series inputs of the Sequence Encoder. In this work, the possibility of applying PINN-SE to a more realistic case is investigated: heating and early cooling of the thermo-stamping process, which is a critical stage in the forming process of continuous fiber reinforced composite materials with thermoplastic polymer. The possibility of extending the PINN-SE inputs to multimodal data, such as sequences of temporal 2D images and to scenarios involving variable geometries, is also explored. The results show that combining multiple encoders with the previously proposed method (Elaarabi et al., 2025b) is feasible, we also show that training the model on synthetic data generated based on experimental data can help the model to generalize well for real experimental data, unseen during the training phase. 

---
# Automating Domain-Driven Design: Experience with a Prompting Framework 

**Authors**: Tobias Eisenreich, Husein Jusic, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2603.26244)  

**Abstract**: Domain-driven design (DDD) is a powerful design technique for architecting complex software systems. This paper introduces a prompting framework that automates core DDD activities through structured large language model (LLM) interactions. We decompose DDD into five sequential steps: (1) establishing an ubiquitous language, (2) simulating event storming, (3) identifying bounded contexts, (4) designing aggregates, and (5) mapping to technical architecture. In a case study, we validated the prompting framework against real-world requirements from FTAPI's enterprise platform. While the first steps consistently generate valuable and usable artifacts, later steps show how minor errors or inaccuracies can propagate and accumulate. Overall, the framework excels as a collaborative sparring partner for building actionable documentation, such as glossaries and context maps, but not for full automation. This allows the experts to concentrate their discussion on the critical trade-offs. In our evaluation, Steps 1 to 3 worked well, but the accumulated errors rendered the artifacts generated from Steps 4 and 5 impractical. Our findings show that LLMs can enhance, but not replace, architectural expertise, offering a practical tool to reduce the effort and overhead of DDD while preserving human-centric decision-making. 

---
# Clawed and Dangerous: Can We Trust Open Agentic Systems? 

**Authors**: Shiping Chen, Qin Wang, Guangsheng Yu, Xu Wang, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.26221)  

**Abstract**: Open agentic systems combine LLM-based planning with external capabilities, persistent memory, and privileged execution. They are used in coding assistants, browser copilots, and enterprise automation. OpenClaw is a visible instance of this broader class.
Without much attention yet, their security challenge is fundamentally different from that of traditional software that relies on predictable execution and well-defined control flow. In open agentic systems, everything is ''probabilistic'': plans are generated at runtime, key decisions may be shaped by untrusted natural-language inputs and tool outputs, execution unfolds in uncertain environments, and actions are taken under authority delegated by human users. The central challenge is therefore not merely robustness against individual attacks, but the governance of agentic behavior under persistent uncertainty.
This paper systematizes the area through a software engineering lens. We introduce a six-dimensional analytical taxonomy and synthesize 50 papers spanning attacks, benchmarks, defenses, audits, and adjacent engineering foundations. From this synthesis, we derive a reference doctrine for secure-by-construction agent platforms, together with an evaluation scorecard for assessing platform security posture. Our review shows that the literature is relatively mature in attack characterization and benchmark construction, but remains weak in deployment controls, operational governance, persistent-memory integrity, and capability revocation. These gaps define a concrete engineering agenda for building agent ecosystems that are governable, auditable, and resilient under compromise. 

---
# Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding 

**Authors**: Shrinidhi Kumbhar, Haofu Liao, Srikar Appalaraju, Kunwar Yashraj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2603.26211)  

**Abstract**: Autoregressive (AR) vision-language models (VLMs) have long dominated multimodal understanding, reasoning, and graphical user interface (GUI) grounding. Recently, discrete diffusion vision-language models (DVLMs) have shown strong performance in multimodal reasoning, offering bidirectional attention, parallel token generation, and iterative refinement. However, their potential for GUI grounding remains unexplored. In this work, we evaluate whether discrete DVLMs can serve as a viable alternative to AR models for GUI grounding. We adapt LLaDA-V for single-turn action and bounding-box prediction, framing the task as text generation from multimodal input. To better capture the hierarchical structure of bounding-box geometry, we propose a hybrid masking schedule that combines linear and deterministic masking, improving grounding accuracy by up to 6.1 points in Step Success Rate (SSR) over the GUI-adapted LLaDA-V trained with linear masking. Evaluations on four datasets spanning web, desktop, and mobile interfaces show that the adapted diffusion model with hybrid masking consistently outperforms the linear-masked variant and performs competitively with autoregressive counterparts despite limited pretraining. Systematic ablations reveal that increasing diffusion steps, generation length, and block length improves accuracy but also increases latency, with accuracy plateauing beyond a certain number of diffusion steps. Expanding the training data with diverse GUI domains further reduces latency by about 1.3 seconds and improves grounding accuracy by an average of 20 points across benchmarks. These results demonstrate that discrete DVLMs are a promising modeling framework for GUI grounding and represent an important step toward diffusion-based GUI agents. 

---
# Sparse Auto-Encoders and Holism about Large Language Models 

**Authors**: Jumbly Grindrod  

**Link**: [PDF](https://arxiv.org/pdf/2603.26207)  

**Abstract**: Does Large Language Model (LLM) technology suggest a meta-semantic picture i.e. a picture of how words and complex expressions come to have the meaning that they do? One modest approach explores the assumptions that seem to be built into how LLMs capture the meanings of linguistic expressions as a way of considering their plausibility (Grindrod, 2026a, 2026b). It has previously been argued that LLMs, in employing a form of distributional semantics, adopt a form of holism about meaning (Grindrod, 2023; Grindrod et al., forthcoming). However, recent work in mechanistic interpretability presents a challenge to these arguments. Specifically, the discovery of a vast array of interpretable latent features within the high dimensional spaces used by LLMs potentially challenges the holistic interpretation. In this paper, I will present the original reasons for thinking that LLMs embody a form of holism (section 1), before introducing recent work on features generated through sparse auto-encoders, and explaining how the discovery of such features suggests an alternative decompositional picture of meaning (section 2). I will then respond to this challenge by considering in greater detail the nature of such features (section 3). Finally, I will return to the holistic picture defended by Grindrod et al. and argue that the picture still stands provided that the features are countable (section 4). 

---
# An Object Web Seminar: A Retrospective on a Technical Dialogue Still Reverbarating 

**Authors**: James J. Cusick  

**Link**: [PDF](https://arxiv.org/pdf/2603.26203)  

**Abstract**: Technology change happens quickly such that new trends tend to crowd out the focus on what was new just yesterday. In this paper the peak popularity of the confluence of Object Technologies with early Web adoption is explored through the content of a seminar held in 1999. Distributed architectures were undergoing significant change at this point, and deeper software capabilities were just beginning to be broadly accessible over the Internet. The Object Web arose and was infused with new development tools reflecting these capabilities and allowing design of applications for deployment during the early days of the World Wide Web. This conference discussed the history, evolution, and use of these tools, architectures, and their future possibilities. The continued dominance of these approaches although under different names is demonstrated even though the term Object Web has receded in use. Favored newer offerings such as Kubernetes and microservices still model the core design attributes of the Object Web for example. Aside from connecting this seminar to relevance in the software world of today this paper also touches on the early AI tools demonstrated in this seminar a quarter century ago and how the popularity wave of any given technology might affect the current focus on AI technology offerings. 

---
# MemCam: Memory-Augmented Camera Control for Consistent Video Generation 

**Authors**: Xinhang Gao, Junlin Guan, Shuhan Luo, Wenzhuo Li, Guanghuan Tan, Jiacheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26193)  

**Abstract**: Interactive video generation has significant potential for scene simulation and video creation. However, existing methods often struggle with maintaining scene consistency during long video generation under dynamic camera control due to limited contextual information. To address this challenge, we propose MemCam, a memory-augmented interactive video generation approach that treats previously generated frames as external memory and leverages them as contextual conditioning to achieve controllable camera viewpoints with high scene consistency. To enable longer and more relevant context, we design a context compression module that encodes memory frames into compact representations and employs co-visibility-based selection to dynamically retrieve the most relevant historical frames, thereby reducing computational overhead while enriching contextual information. Experiments on interactive video generation tasks show that MemCam significantly outperforms existing baseline methods as well as open-source state-of-the-art approaches in terms of scene consistency, particularly in long video scenarios with large camera rotations. 

---
# Progressive Learning with Anatomical Priors for Reliable Left Atrial Scar Segmentation from Late Gadolinium Enhancement MRI 

**Authors**: Jing Zhang, Bastien Bergere, Emilie Bollache, Jonas Leite, Mikaël Laredo, Alban Redheuil, Nadjia Kachenoura  

**Link**: [PDF](https://arxiv.org/pdf/2603.26186)  

**Abstract**: Cardiac MRI late gadolinium enhancement (LGE) enables non-invasive identification of left atrial (LA) scar, whose spatial distribution is strongly associated with atrial fibrillation (AF) severity and recurrence. However, automatic LA scar segmentation remains challenging due to low contrast, annotation variability, and the lack of anatomical constraints, often leading to non-reliable predictions. Accordingly, our aim was to propose a progressive learning strategy to segment LA scar from LGE images inspired from a clinical workflow. A 3-stage framework based on SwinUNETR was implemented, comprising: 1) a first LA cavity pre-learning model, 2) dual-task model which further learns spatial relationship between LA geometry and scar patterns, and 3) fine-tuning on precise segmentation of the scar. Furthermore, we introduced an anatomy-aware spatially weighted loss that incorporates prior clinical knowledge by constraining scar predictions to anatomically plausible LA wall regions while mitigating annotation bias. Our preliminary results obtained on validation LGE volumes from LASCARQS public dataset after 5-fold cross validation, LA segmentation had Dice score of 0.94, LA scar segmentation achieved Dice score of 0.50, Hausdorff Distance of 11.84 mm, Average Surface Distance of 1.80 mm, outperforming only a one-stage scar segmentation with 0.49, 13.02 mm, 1.96 mm, repectively. By explicitly embedding clinical anatomical priors and diagnostic reasoning into deep learning, the proposed approach improved the accuracy and reliability of LA scar segmentation from LGE, revealing the importance of clinically informed model design. 

---
# On the Complexity of Optimal Graph Rewiring for Oversmoothing and Oversquashing in Graph Neural Networks 

**Authors**: Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2603.26140)  

**Abstract**: Graph Neural Networks (GNNs) face two fundamental challenges when scaled to deep architectures: oversmoothing, where node representations converge to indistinguishable vectors, and oversquashing, where information from distant nodes fails to propagate through bottlenecks. Both phenomena are intimately tied to the underlying graph structure, raising a natural question: can we optimize the graph topology to mitigate these issues? This paper provides a theoretical investigation of the computational complexity of such graph structure optimization. We formulate oversmoothing and oversquashing mitigation as graph optimization problems based on spectral gap and conductance, respectively. We prove that exact optimization for either problem is NP-hard through reductions from Minimum Bisection, establishing NP-completeness of the decision versions. Our results provide theoretical foundations for understanding the fundamental limits of graph rewiring for GNN optimization and justify the use of approximation algorithms and heuristic methods in practice. 

---
# ATime-Consistent Benchmark for Repository-Level Software Engineering Evaluation 

**Authors**: Xianpeng, Haonan Sun, Tian Yu, Sheng Ma, Qincheng Zhang, Lifei Rao, Chen Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.26137)  

**Abstract**: Evaluation of repository-aware software engineering systems is often confounded by synthetic task design, prompt leakage, and temporal contamination between repository knowledge and future code changes. We present a time-consistent benchmark methodology that snapshots a repository at time T0, constructs repository-derived code knowledge using only artifacts available before T0, and evaluates on engineering tasks derived from pull requests merged in the future interval (T0, T1]. Each historical pull request is transformed into a natural-language task through an LLM-assisted prompt-generation pipeline, and the benchmark is formalized as a matched A/B comparison in which the same software engineering agent is evaluated with and without repository-derived code knowledge while all other variables are held constant. We also report a baseline characterization study on two open-source repositories, DragonFly and React, using three Claude-family models and four prompt granularities. Across both repositories, file-level F1 increases monotonically from minimal to guided prompts, reaching 0.8081 on DragonFly and 0.8078 on React for the strongest tested model. These results show that prompt construction is a first-order benchmark variable. More broadly, the benchmark highlights that temporal consistency and prompt control are core validity requirements for repository-aware software engineering evaluation. 

---
# SWE-PRBench: Benchmarking AI Code Review Quality Against Pull Request Feedback 

**Authors**: Deepak Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2603.26130)  

**Abstract**: We introduce SWE-PRBench, a benchmark of 350 pull requests with human-annotated ground truth for evaluating AI code review quality. Evaluated against an LLM-as-judge framework validated at kappa=0.75, 8 frontier models detect only 15-31% of human-flagged issues on the diff-only configuration, demonstrating that AI code review remains far below human expert performance despite strong results on code generation benchmarks. Pull requests are drawn from active open-source repositories, filtered from 700 candidates using a Repository Quality Score, and evaluated under three frozen context configurations: diff only (config_A), diff with file content (config_B), and full context (config_C), enabling systematic ablation of context provision strategies. All 8 models degrade monotonically from config_A to config_C, even when context is provided via structured semantic layers including AST-extracted function context and import graph resolution. The dominant mechanism is a collapse of Type2_Contextual issue detection at config_B, consistent with attention dilution in long contexts: a structured 2,000-token diff-with-summary prompt outperforms a 2,500-token full-context prompt enriched with execution context, behaviour mapping, and test signatures across all 8 models. The top four models are statistically indistinguishable (mean score 0.147-0.153) while a clear tier gap separates them from the remaining four (mean score <= 0.113). Dataset, contexts, annotations, and evaluation harness are released publicly. 

---
# Finding Distributed Object-Centric Properties in Self-Supervised Transformers 

**Authors**: Samyak Rawlekar, Amitabh Swain, Yujun Cai, Yiwei Wang, Ming-Hsuan Yang, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2603.26127)  

**Abstract**: Self-supervised Vision Transformers (ViTs) like DINO show an emergent ability to discover objects, typically observed in [CLS] token attention maps of the final layer. However, these maps often contain spurious activations resulting in poor localization of objects. This is because the [CLS] token, trained on an image-level objective, summarizes the entire image instead of focusing on objects. This aggregation dilutes the object-centric information existing in the local, patch-level interactions. We analyze this by computing inter-patch similarity using patch-level attention components (query, key, and value) across all layers. We find that: (1) Object-centric properties are encoded in the similarity maps derived from all three components ($q, k, v$), unlike prior work that uses only key features or the [CLS] token. (2) This object-centric information is distributed across the network, not just confined to the final layer. Based on these insights, we introduce Object-DINO, a training-free method that extracts this distributed object-centric information. Object-DINO clusters attention heads across all layers based on the similarities of their patches and automatically identifies the object-centric cluster corresponding to all objects. We demonstrate Object-DINO's effectiveness on two applications: enhancing unsupervised object discovery (+3.6 to +12.4 CorLoc gains) and mitigating object hallucination in Multimodal Large Language Models by providing visual grounding. Our results demonstrate that using this distributed object-centric information improves downstream tasks without additional training. 

---
# SkinGPT-X: A Self-Evolving Collaborative Multi-Agent System for Transparent and Trustworthy Dermatological Diagnosis 

**Authors**: Zhangtianyi Chen, Yuhao Shen, Florensia Widjaja, Yan Xu, Liyuan Sun, Zijian Wang, Hongyi Chen, Wufei Dai, Juexiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.26122)  

**Abstract**: While recent advancements in Large Language Models have significantly advanced dermatological diagnosis, monolithic LLMs frequently struggle with fine-grained, large-scale multi-class diagnostic tasks and rare skin disease diagnosis owing to training data sparsity, while also lacking the interpretability and traceability essential for clinical reasoning. Although multi-agent systems can offer more transparent and explainable diagnostics, existing frameworks are primarily concentrated on Visual Question Answering and conversational tasks, and their heavy reliance on static knowledge bases restricts adaptability in complex real-world clinical settings. Here, we present SkinGPT-X, a multimodal collaborative multi-agent system for dermatological diagnosis integrated with a self-evolving dermatological memory mechanism. By simulating the diagnostic workflow of dermatologists and enabling continuous memory evolution, SkinGPT-X delivers transparent and trustworthy diagnostics for the management of complex and rare dermatological cases. To validate the robustness of SkinGPT-X, we design a three-tier comparative experiment. First, we benchmark SkinGPT-X against four state-of-the-art LLMs across four public datasets, demonstrating its state-of-the-art performance with a +9.6% accuracy improvement on DDI31 and +13% weighted F1 gain on Dermnet over the state-of-the-art model. Second, we construct a large-scale multi-class dataset covering 498 distinct dermatological categories to evaluate its fine-grained classification capabilities. Finally, we curate the rare skin disease dataset, the first benchmark to address the scarcity of clinical rare skin diseases which contains 564 clinical samples with eight rare dermatological diseases. On this dataset, SkinGPT-X achieves a +9.8% accuracy improvement, a +7.1% weighted F1 improvement, a +10% Cohen's Kappa improvement. 

---
# DPD-Cancer: Explainable Graph-based Deep Learning for Small Molecule Anti-Cancer Activity Prediction 

**Authors**: Magnus H. Strømme, Alex G. C. de Sá, David B. Ascher  

**Link**: [PDF](https://arxiv.org/pdf/2603.26114)  

**Abstract**: Accurate drug response prediction is a critical bottleneck in computational biochemistry, limited by the challenge of modelling the interplay between molecular structure and cellular context. In cancer research, this is acute due to tumour heterogeneity and genomic variability, which hinder the identification of effective therapies. Conventional approaches often fail to capture non-linear relationships between chemical features and biological outcomes across diverse cell lines. To address this, we introduce DPD-Cancer, a deep learning method based on a Graph Attention Transformer (GAT) framework. It is designed for small molecule anti-cancer activity classification and the quantitative prediction of cell-line specific responses, specifically growth inhibition concentration (pGI50). Benchmarked against state-of-the-art methods (pdCSM-cancer, ACLPred, and MLASM), DPD-Cancer demonstrated superior performance, achieving an Area Under ROC Curve (AUC) of up to 0.87 on strictly partitioned NCI60 data and up to 0.98 on ACLPred/MLASM datasets. For pGI50 prediction across 10 cancer types and 73 cell lines, the model achieved Pearson's correlation coefficients of up to 0.72 on independent test sets. These findings confirm that attention-based mechanisms offer significant advantages in extracting meaningful molecular representations, establishing DPD-Cancer as a competitive tool for prioritising drug candidates. Furthermore, DPD-Cancer provides explainability by leveraging the attention mechanism to identify and visualise specific molecular substructures, offering actionable insights for lead optimisation. DPD-Cancer is freely available as a web server at: this https URL. 

---
# "Oops! ChatGPT is Temporarily Unavailable!": A Diary Study on Knowledge Workers' Experiences of LLM Withdrawal 

**Authors**: Eunseo Oh, Suyoun Lee, Jae Young Choi, Soobin Park, Youn-kyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2603.26099)  

**Abstract**: LLMs have become deeply embedded in knowledge work, raising concerns about growing dependency and the potential undermining of human skills. To investigate the pervasiveness of LLMs in work practices, we conducted a four-day diary study with frequent LLM users (N=10), observing how knowledge workers responded to a temporary withdrawal of LLMs. Our findings show how LLM withdrawal disrupted participants' workflows by identifying gaps in task execution, how self-directed work led participants to reclaim professional values, and how everyday practices revealed the extent to which LLM use had become inescapably normative. Conceptualizing LLMs as infrastructural to contemporary knowledge work, this research contributes empirical insights into the often invisible role of LLMs and proposes value-driven appropriation as an approach to supporting professional values in the current LLM-pervasive work environment. 

---
# A Human-Inspired Decoupled Architecture for Efficient Audio Representation Learning 

**Authors**: Harunori Kawano, Takeshi Sasaki  

**Link**: [PDF](https://arxiv.org/pdf/2603.26098)  

**Abstract**: While self-supervised learning (SSL) has revolutionized audio representation, the excessive parameterization and quadratic computational cost of standard Transformers limit their deployment on resource-constrained devices. To address this bottleneck, we propose HEAR (Human-inspired Efficient Audio Representation), a novel decoupled architecture. Inspired by the human cognitive ability to isolate local acoustic features from global context, HEAR splits the processing pipeline into two dedicated modules: an Acoustic Model for local feature extraction and a Task Model for global semantic integration. Coupled with an Acoustic Tokenizer trained via knowledge distillation, our approach enables robust Masked Audio Modeling (MAM). Extensive experiments demonstrate that HEAR requires only 15M parameters and 9.47 GFLOPs for inference, operating at a fraction of the computational cost of conventional foundation models (which typically require 85M-94M parameters). Despite this high efficiency, HEAR achieves highly competitive performance across diverse audio classification benchmarks. The code and pre-trained models are available at this https URL 

---
# Dynamic Tokenization via Reinforcement Patching: End-to-end Training and Zero-shot Transfer 

**Authors**: Yulun Wu, Sravan Kumar Ankireddy, Samuel Sharpe, Nikita Seleznev, Dehao Yuan, Hyeji Kim, Nam H. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.26097)  

**Abstract**: Efficiently aggregating spatial or temporal horizons to acquire compact representations has become a unifying principle in modern deep learning models, yet learning data-adaptive representations for long-horizon sequence data, especially continuous sequences like time series, remains an open challenge. While fixed-size patching has improved scalability and performance, discovering variable-sized, data-driven patches end-to-end often forces models to rely on soft discretization, specific backbones, or heuristic rules. In this work, we propose Reinforcement Patching (ReinPatch), the first framework to jointly optimize a sequence patching policy and its downstream sequence backbone model using reinforcement learning. By formulating patch boundary placement as a discrete decision process optimized via Group Relative Policy Gradient (GRPG), ReinPatch bypasses the need for continuous relaxations and performs dynamic patching policy optimization in a natural manner. Moreover, our method allows strict enforcement of a desired compression rate, freeing the downstream backbone to scale efficiently, and naturally supports multi-level hierarchical modeling. We evaluate ReinPatch on time-series forecasting datasets, where it demonstrates compelling performance compared to state-of-the-art data-driven patching strategies. Furthermore, our detached design allows the patching module to be extracted as a standalone foundation patcher, providing the community with visual and empirical insights into the segmentation behaviors preferred by a purely performance-driven neural patching strategy. 

---
# Selective Deficits in LLM Mental Self-Modeling in a Behavior-Based Test of Theory of Mind 

**Authors**: Christopher Ackerman  

**Link**: [PDF](https://arxiv.org/pdf/2603.26089)  

**Abstract**: The ability to represent oneself and others as agents with knowledge, intentions, and belief states that guide their behavior - Theory of Mind - is a human universal that enables us to navigate - and manipulate - the social world. It is supported by our ability to form mental models of ourselves and others. Its ubiquity in human affairs entails that LLMs have seen innumerable examples of it in their training data and therefore may have learned to mimic it, but whether they have actually learned causal models that they can deploy in arbitrary settings is unclear. We therefore develop a novel experimental paradigm that requires that subjects form representations of the mental states of themselves and others and act on them strategically rather than merely describe them. We test a wide range of leading open and closed source LLMs released since 2024, as well as human subjects, on this paradigm. We find that 1) LLMs released before mid-2025 fail at all of our tasks, 2) more recent LLMs achieve human-level performance on modeling the cognitive states of others, and 3) even frontier LLMs fail at our self-modeling task - unless afforded a scratchpad in the form of a reasoning trace. We further demonstrate cognitive load effects on other-modeling tasks, offering suggestive evidence that LLMs are using something akin to limited-capacity working memory to hold these mental representations in mind during a single forward pass. Finally, we explore the mechanisms by which reasoning models succeed at the self- and other-modeling tasks, and show that they readily engage in strategic deception. 

---
# When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization 

**Authors**: Zhihan Chen, Yuhuan Zhao, Yijie Zhu, Xinyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.26078)  

**Abstract**: Subject-driven text-to-image diffusion models have achieved remarkable success in preserving single identities, yet their ability to compose multiple interacting subjects remains largely unexplored and highly challenging. Existing evaluation protocols typically rely on global CLIP metrics, which are insensitive to local identity collapse and fail to capture the severity of multi-subject entanglement. In this paper, we identify a pervasive "Illusion of Scalability" in current models: while they excel at synthesizing 2-4 subjects in simple layouts, they suffer from catastrophic identity collapse when scaled to 6-10 subjects or tasked with complex physical interactions. To systematically expose this failure mode, we construct a rigorous stress-test benchmark comprising 75 prompts distributed across varying subject counts and interaction difficulties (Neutral, Occlusion, Interaction). Furthermore, we demonstrate that standard CLIP-based metrics are fundamentally flawed for this task, as they often assign high scores to semantically correct but identity-collapsed images (e.g., generating generic clones). To address this, we introduce the Subject Collapse Rate (SCR), a novel evaluation metric grounded in DINOv2's structural priors, which strictly penalizes local attention leakage and homogenization. Our extensive evaluation of state-of-the-art models (MOSAIC, XVerse, PSR) reveals a precipitous drop in identity fidelity as scene complexity grows, with SCR approaching 100% at 10 subjects. We trace this collapse to the semantic shortcuts inherent in global attention routing, underscoring the urgent need for explicit physical disentanglement in future generative architectures. 

---
# R-PGA: Robust Physical Adversarial Camouflage Generation via Relightable 3D Gaussian Splatting 

**Authors**: Tianrui Lou, Siyuan Liang, Jiawei Liang, Yuze Gao, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.26067)  

**Abstract**: Physical adversarial camouflage poses a severe security threat to autonomous driving systems by mapping adversarial textures onto 3D objects. Nevertheless, current methods remain brittle in complex dynamic scenarios, failing to generalize across diverse geometric (e.g., viewing configurations) and radiometric (e.g., dynamic illumination, atmospheric scattering) variations. We attribute this deficiency to two fundamental limitations in simulation and optimization. First, the reliance on coarse, oversimplified simulations (e.g., via CARLA) induces a significant domain gap, confining optimization to a biased feature space. Second, standard strategies targeting average performance result in a rugged loss landscape, leaving the camouflage vulnerable to configuration this http URL bridge these gaps, we propose the Relightable Physical 3D Gaussian Splatting (3DGS) based Attack framework (R-PGA). Technically, to address the simulation fidelity issue, we leverage 3DGS to ensure photo-realistic reconstruction and augment it with physically disentangled attributes to decouple intrinsic material from lighting. Furthermore, we design a hybrid rendering pipeline that leverages precise Relightable 3DGS for foreground rendering, while employing a pre-trained image translation model to synthesize plausible relighted backgrounds that align with the relighted this http URL address the optimization robustness issue, we propose the Hard Physical Configuration Mining (HPCM) module, designed to actively mine worst-case physical configurations and suppress their corresponding loss peaks. This strategy not only diminishes the overall loss magnitude but also effectively flattens the rugged loss landscape, ensuring consistent adversarial effectiveness and robustness across varying physical configurations. 

---
# MuDD: A Multimodal Deception Detection Dataset and GSR-Guided Progressive Distillation for Non-Contact Deception Detection 

**Authors**: Peiyuan Jiang, Yao Liu, Yanglei Gan, Jiaye Yang, Lu Liu, Daibing Yao, Qiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.26064)  

**Abstract**: Non-contact automatic deception detection remains challenging because visual and auditory deception cues often lack stable cross-subject patterns. In contrast, galvanic skin response (GSR) provides more reliable physiological cues and has been widely used in contact-based deception detection. In this work, we leverage stable deception-related knowledge in GSR to guide representation learning in non-contact modalities through cross-modal knowledge distillation. A key obstacle, however, is the lack of a suitable dataset for this setting. To address this, we introduce MuDD, a large-scale Multimodal Deception Detection dataset containing recordings from 130 participants over 690 minutes. In addition to video, audio, and GSR, MuDD also provides Photoplethysmography, heart rate, and personality traits, supporting broader scientific studies of deception. Based on this dataset, we propose GSR-guided Progressive Distillation (GPD), a cross-modal distillation framework for mitigating the negative transfer caused by the large modality mismatch between GSR and non-contact signals. The core innovation of GPD is the integration of progressive feature-level and digit-level distillation with dynamic routing, which allows the model to adaptively determine how teacher knowledge should be transferred during training, leading to more stable cross-modal knowledge transfer. Extensive experiments and visualizations show that GPD outperforms existing methods and achieves state-of-the-art performance on both deception detection and concealed-digit identification. 

---
# Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification 

**Authors**: Zizhao Chen, Ping Wei, Ziyang Ren, Huan Li, Xiangru Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.26052)  

**Abstract**: As multimodal misinformation becomes more sophisticated, its detection and grounding are crucial. However, current multimodal verification methods, relying on passive holistic fusion, struggle with sophisticated misinformation. Due to 'feature dilution,' global alignments tend to average out subtle local semantic inconsistencies, effectively masking the very conflicts they are designed to find. We introduce MaLSF (Mask-aware Local Semantic Fusion), a novel framework that shifts the paradigm to active, bidirectional verification, mimicking human cognitive cross-referencing. MaLSF utilizes mask-label pairs as semantic anchors to bridge pixels and words. Its core mechanism features two innovations: 1) a Bidirectional Cross-modal Verification (BCV) module that acts as an interrogator, using parallel query streams (Text-as-Query and Image-as-Query) to explicitly pinpoint conflicts; and 2) a Hierarchical Semantic Aggregation (HSA) module that intelligently aggregates these multi-granularity conflict signals for task-specific reasoning. In addition, to extract fine-grained mask-label pairs, we introduce a set of diverse mask-label pair extraction parsers. MaLSF achieves state-of-the-art performance on both the DGM4 and multimodal fake news detection tasks. Extensive ablation studies and visualization results further verify its effectiveness and interpretability. 

---
# Seeing Like Radiologists: Context- and Gaze-Guided Vision-Language Pretraining for Chest X-rays 

**Authors**: Kang Liu, Zhuoqi Ma, Siyu Liang, Yunan Li, Xiyue Gao, Chao Liang, Kun Xie, Qiguang Miao  

**Link**: [PDF](https://arxiv.org/pdf/2603.26049)  

**Abstract**: Despite recent advances in medical vision-language pretraining, existing models still struggle to capture the diagnostic workflow: radiographs are typically treated as context-agnostic images, while radiologists' gaze -- a crucial cue for visual reasoning -- remains largely underexplored by existing methods. These limitations hinder the modeling of disease-specific patterns and weaken cross-modal alignment. To bridge this gap, we introduce CoGaze, a Context- and Gaze-guided vision-language pretraining framework for chest X-rays. We first propose a context-infused vision encoder that models how radiologists integrate clinical context -- including patient history, symptoms, and diagnostic intent -- to guide diagnostic reasoning. We then present a multi-level supervision paradigm that (1) enforces intra- and inter-modal semantic alignment through hybrid-positive contrastive learning, (2) injects diagnostic priors via disease-aware cross-modal representation learning, and (3) leverages radiologists' gaze as probabilistic priors to guide attention toward diagnostically salient regions. Extensive experiments demonstrate that CoGaze consistently outperforms state-of-the-art methods across diverse tasks, achieving up to +2.0% CheXbertF1 and +1.2% BLEU2 for free-text and structured report generation, +23.2% AUROC for zero-shot classification, and +12.2% Precision@1 for image-text retrieval. Code is available at this https URL. 

---
# H-Node Attack and Defense in Large Language Models 

**Authors**: Eric Yocam, Varghese Vaidyan, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26045)  

**Abstract**: We present H-Node Adversarial Noise Cancellation (H-Node ANC), a mechanistic framework that identifies, exploits, and defends hallucination representations in transformer-based large language models (LLMs) at the level of individual hidden-state dimensions. A logistic regression probe trained on last-token hidden states localizes hallucination signal to a small set of high-variance dimensions -- termed Hallucination Nodes (H-Nodes) -- with probe AUC reaching 0.90 across four architectures. A white-box adversarial attack amplifies these dimensions at inference time via a real-time forward hook, achieving a selectivity of 3.02x with less than 10% visibility to the defender. Adaptive ANC defense suppresses H-Node excess in-pass using confidence-weighted cancellation, reducing grounded activation drift by 33-42% over static cancellation. A dynamic iterative extension that re-ranks cancellation targets across successive passes recovers up to 0.69 robustness from a single-pass baseline of 8%. All contributions are validated on OPT-125M, Phi-3-mini-4k-instruct, LLaMA-3-8B-Instruct, and Mistral-7B-Instruct-v0.3 (125M-8B parameters). Perplexity impact is surgical (<5%) and MMLU degradation is at most 3%, confirming that the defense does not impair general reasoning capability. 

---
# Designing Fatigue-Aware VR Interfaces via Biomechanical Models 

**Authors**: Harshitha Voleti, Charalambos Poullis  

**Link**: [PDF](https://arxiv.org/pdf/2603.26031)  

**Abstract**: Prolonged mid-air interaction in virtual reality (VR) causes arm fatigue and discomfort, negatively affecting user experience. Incorporating ergonomic considerations into VR user interface (UI) design typically requires extensive human-in-the-loop evaluation. Although biomechanical models have been used to simulate human behavior in HCI tasks, their application as surrogate users for ergonomic VR UI design remains underexplored. We propose a hierarchical reinforcement learning framework that leverages biomechanical user models to evaluate and optimize VR interfaces for mid-air interaction. A motion agent is trained to perform button-press tasks in VR under sequential conditions, using realistic movement strategies and estimating muscle-level effort via a validated three-compartment control with recovery (3CC-r) fatigue model. The simulated fatigue output serves as feedback for a UI agent that optimizes UI element layout via reinforcement learning (RL) to minimize fatigue. We compare the RL-optimized layout against a manually-designed centered baseline and a Bayesian optimized baseline. Results show that fatigue trends from the biomechanical model align with human user data. Moreover, the RL-optimized layout using simulated fatigue feedback produced significantly lower perceived fatigue in a follow-up human study. We further demonstrate the framework's extensibility via a simulated case study on longer sequential tasks with non-uniform interaction frequencies. To our knowledge, this is the first work using simulated biomechanical muscle fatigue as a direct optimization signal for VR UI layout design. Our findings highlight the potential of biomechanical user models as effective surrogate tools for ergonomic VR interface design, enabling efficient early-stage iteration with less reliance on extensive human participation. 

---
# Unlabeled Cross-Center Automatic Analysis for TAAD: An Integrated Framework from Segmentation to Clinical Features 

**Authors**: Mengdi Liu, Qiang Li, Weizhi Nie, Shaopeng Zhang, Yuting Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.26019)  

**Abstract**: Type A Aortic Dissection (TAAD) is a life-threatening cardiovascular emergency that demands rapid and precise preoperative evaluation. While key anatomical and pathological features are decisive for surgical planning, current research focuses predominantly on improving segmentation accuracy, leaving the reliable, quantitative extraction of clinically actionable features largely under-explored. Furthermore, constructing comprehensive TAAD datasets requires labor-intensive, expert level pixel-wise annotations, which is impractical for most clinical institutions. Due to significant domain shift, models trained on a single center dataset also suffer from severe performance degradation during cross-institutional deployment. This study addresses a clinically critical challenge: the accurate extraction of key TAAD clinical features during cross-institutional deployment in the total absence of target-domain annotations. To this end, we propose an unsupervised domain adaptation (UDA)-driven framework for the automated extraction of TAAD clinical features. The framework leverages limited source-domain labels while effectively adapting to unlabeled data from target domains. Tailored for real-world emergency workflows, our framework aims to achieve stable cross-institutional multi-class segmentation, reliable and quantifiable clinical feature extraction, and practical deployability independent of high-cost annotations. Extensive experiments demonstrate that our method significantly improves cross-domain segmentation performance compared to existing state-of-the-art approaches. More importantly, a reader study involving multiple cardiovascular surgeons confirms that the automatically extracted clinical features provide meaningful assistance for preoperative assessment, highlighting the practical utility of the proposed end-to-end segmentation-to-feature pipeline. 

---
# VLAgeBench: Benchmarking Large Vision-Language Models for Zero-Shot Human Age Estimation 

**Authors**: Rakib Hossain Sajib, Md Kishor Morol, Rajan Das Gupta, Mohammad Sakib Mahmood, Shuvra Smaran Das  

**Link**: [PDF](https://arxiv.org/pdf/2603.26015)  

**Abstract**: Human age estimation from facial images represents a challenging computer vision task with significant applications in biometrics, healthcare, and human-computer interaction. While traditional deep learning approaches require extensive labeled datasets and domain-specific training, recent advances in large vision-language models (LVLMs) offer the potential for zero-shot age estimation. This study presents a comprehensive zero-shot evaluation of state-of-the-art Large Vision-Language Models (LVLMs) for facial age estimation, a task traditionally dominated by domain-specific convolutional networks and supervised learning. We assess the performance of GPT-4o, Claude 3.5 Sonnet, and LLaMA 3.2 Vision on two benchmark datasets, UTKFace and FG-NET, without any fine-tuning or task-specific adaptation. Using eight evaluation metrics, including MAE, MSE, RMSE, MAPE, MBE, $R^2$, CCC, and $\pm$5-year accuracy, we demonstrate that general-purpose LVLMs can deliver competitive performance in zero-shot settings. Our findings highlight the emergent capabilities of LVLMs for accurate biometric age estimation and position these models as promising tools for real-world applications. Additionally, we highlight performance disparities linked to image quality and demographic subgroups, underscoring the need for fairness-aware multimodal inference. This work introduces a reproducible benchmark and positions LVLMs as promising tools for real-world applications in forensic science, healthcare monitoring, and human-computer interaction. The benchmark focuses on strict zero-shot inference without fine-tuning and highlights remaining challenges related to prompt sensitivity, interpretability, computational cost, and demographic fairness. 

---
# FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Assistants 

**Authors**: Mahesh Bhosale, Abdul Wasi, Shantam Srivastava, Shifa Latif, Tianyu Luan, Mingchen Gao, David Doermann, Xuan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.26008)  

**Abstract**: While powerful in image-conditioned generation, multimodal large language models (MLLMs) can display uneven performance across demographic groups, highlighting fairness risks. In safety-critical clinical settings, such disparities risk producing unequal diagnostic narratives and eroding trust in AI-assisted decision-making. While fairness has been studied extensively in vision-only and language-only models, its impact on MLLMs remains largely underexplored. To address these biases, we introduce FairLLaVA, a parameter-efficient fine-tuning method that mitigates group disparities in visual instruction tuning without compromising overall performance. By minimizing the mutual information between target attributes, FairLLaVA regularizes the model's representations to be demographic-invariant. The method can be incorporated as a lightweight plug-in, maintaining efficiency with low-rank adapter fine-tuning, and provides an architecture-agnostic approach to fair visual instruction following. Extensive experiments on large-scale chest radiology report generation and dermoscopy visual question answering benchmarks show that FairLLaVA consistently reduces inter-group disparities while improving both equity-scaled clinical performance and natural language generation quality across diverse medical imaging modalities. Code can be accessed at this https URL. 

---
# Longitudinal Boundary Sharpness Coefficient Slopes Predict Time to Alzheimer's Disease Conversion in Mild Cognitive Impairment: A Survival Analysis Using the ADNI Cohort 

**Authors**: Ishaan Cherukuri  

**Link**: [PDF](https://arxiv.org/pdf/2603.26007)  

**Abstract**: Predicting whether someone with mild cognitive impairment (MCI) will progress to Alzheimer's disease (AD) is crucial in the early stages of neurodegeneration. This uncertainty limits enrollment in clinical trials and delays urgent treatment. The Boundary Sharpness Coefficient (BSC) measures how well-defined the gray-white matter boundary looks on structural MRI. This study measures how BSC changes over time, namely, how fast the boundary degrades each year works much better than looking at a single baseline scan for predicting MCI-to-AD conversion. This study analyzed 1,824 T1-weighted MRI scans from 450 ADNI subjects (95 converters, 355 stable; mean follow-up: 4.84 years). BSC voxel-wise maps were computed using tissue segmentation at the gray-white matter cortical ribbon. Previous studies have used CNN and RNN models that reached 96.0% accuracy for AD classification and 84.2% for MCI conversion, but those approaches disregard specific regions within the brain. This study focused specifically on the gray-white matter interface. The approach uses temporal slope features capturing boundary degradation rates, feeding them into Random Survival Forest, a non-parametric ensemble method for right-censored survival data. The Random Survival Forest trained on BSC slopes achieved a test C-index of 0.63, a 163% improvement over baseline parametric models (test C-index: 0.24). Structural MRI costs a fraction of PET imaging ($800--$1,500 vs. $5,000--$7,000) and does not require CSF collection. These temporal biomarkers could help with patient-centered safety screening as well as risk assessment. 

---
# Policy-Guided World Model Planning for Language-Conditioned Visual Navigation 

**Authors**: Amirhosein Chahe, Lifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.25981)  

**Abstract**: Navigating to a visually specified goal given natural language instructions remains a fundamental challenge in embodied AI. Existing approaches either rely on reactive policies that struggle with long-horizon planning, or employ world models that suffer from poor action initialization in high-dimensional spaces. We present PiJEPA, a two-stage framework that combines the strengths of learned navigation policies with latent world model planning for instruction-conditioned visual navigation. In the first stage, we finetune an Octo-based generalist policy, augmented with a frozen pretrained vision encoder (DINOv2 or V-JEPA-2), on the CAST navigation dataset to produce an informed action distribution conditioned on the current observation and language instruction. In the second stage, we use this policy-derived distribution to warm-start Model Predictive Path Integral (MPPI) planning over a separately trained JEPA world model, which predicts future latent states in the embedding space of the same frozen encoder. By initializing the MPPI sampling distribution from the policy prior rather than from an uninformed Gaussian, our planner converges faster to high-quality action sequences that reach the goal. We systematically study the effect of the vision encoder backbone, comparing DINOv2 and V-JEPA-2, across both the policy and world model components. Experiments on real-world navigation tasks demonstrate that PiJEPA significantly outperforms both standalone policy execution and uninformed world model planning, achieving improved goal-reaching accuracy and instruction-following fidelity. 

---
# Do Neurons Dream of Primitive Operators? Wake-Sleep Compression Rediscovers Schank's Event Semantics 

**Authors**: Peter Balogh  

**Link**: [PDF](https://arxiv.org/pdf/2603.25975)  

**Abstract**: We show that they do. Schank's conceptual dependency theory proposed that all events decompose into primitive operations -- ATRANS, PTRANS, MTRANS, and others -- hand-coded from linguistic intuition. Can the same primitives be discovered automatically through compression pressure alone?
We adapt DreamCoder's wake-sleep library learning to event state transformations. Given events as before/after world state pairs, our system finds operator compositions explaining each event (wake), then extracts recurring patterns as new operators optimized under Minimum Description Length (sleep). Starting from four generic primitives, it discovers operators mapping directly to Schank's: MOVE_PROP_has = ATRANS, CHANGE_location = PTRANS, SET_knows = MTRANS, SET_consumed = INGEST, plus compound operators ("mail" = ATRANS + PTRANS) and novel emotional state operators absent from Schank's taxonomy.
We validate on synthetic events and real-world commonsense data from the ATOMIC knowledge graph. On synthetic data, discovered operators achieve Bayesian MDL within 4% of Schank's hand-coded primitives while explaining 100% of events vs. Schank's 81%. On ATOMIC, results are more dramatic: Schank's primitives explain only 10% of naturalistic events, while the discovered library explains 100%. Dominant operators are not physical-action primitives but mental and emotional state changes -- CHANGE_wants (20%), CHANGE_feels (18%), CHANGE_is (18%) -- none in Schank's original taxonomy.
These results provide the first empirical evidence that event primitives can be derived from compression pressure, that Schank's core primitives are information-theoretically justified, and that the complete inventory is substantially richer than proposed -- with mental/emotional operators dominating in naturalistic data. 

---
# When Chain-of-Thought Backfires: Evaluating Prompt Sensitivity in Medical Language Models 

**Authors**: Binesh Sadanandan, Vahid Behzadan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25960)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in medical settings, yet their sensitivity to prompt formatting remains poorly characterized. We evaluate MedGemma (4B and 27B parameters) on MedMCQA (4,183 questions) and PubMedQA (1,000 questions) across a broad suite of robustness tests. Our experiments reveal several concerning findings. Chain-of-Thought (CoT) prompting decreases accuracy by 5.7% compared to direct answering. Few-shot examples degrade performance by 11.9% while increasing position bias from 0.14 to 0.47. Shuffling answer options causes the model to change predictions 59.1% of the time, with accuracy dropping up to 27.4 percentage points. Front-truncating context to 50% causes accuracy to plummet below the no-context baseline, yet back-truncation preserves 97% of full-context accuracy. We further show that cloze scoring (selecting the highest log-probability option token) achieves 51.8% (4B) and 64.5% (27B), surpassing all prompting strategies and revealing that models "know" more than their generated text shows. Permutation voting recovers 4 percentage points over single-ordering inference. These results demonstrate that prompt engineering techniques validated on general-purpose models do not transfer to domain-specific medical LLMs, and that reliable alternatives exist. 

---
# Collision-Aware Vision-Language Learning for End-to-End Driving with Multimodal Infraction Datasets 

**Authors**: Alex Koran, Dimitrios Sinodinos, Hadi Hojjati, Takuya Nanri, Fangge Chen, Narges Armanfard  

**Link**: [PDF](https://arxiv.org/pdf/2603.25946)  

**Abstract**: High infraction rates remain the primary bottleneck for end-to-end (E2E) autonomous driving, as evidenced by the low driving scores on the CARLA Leaderboard. Despite collision-related infractions being the dominant failure mode in closed-loop evaluations, collision-aware representation learning has received limited attention. To address this gap, we first develop a Video-Language-Augmented Anomaly Detector (VLAAD), leveraging a Multiple Instance Learning (MIL) formulation to obtain stable, temporally localized collision signals for proactive prediction. To transition these capabilities into closed-loop simulations, we must overcome the limitations of existing simulator datasets, which lack multimodality and are frequently restricted to simple intersection scenarios. Therefore, we introduce CARLA-Collide, a large-scale multimodal dataset capturing realistic collision events across highly diverse road networks. Trained on this diverse simulator data, VLAAD serves as a collision-aware plug-in module that can be seamlessly integrated into existing E2E driving models. By integrating our module into a pretrained TransFuser++ agent, we demonstrate a 14.12% relative increase in driving score with minimal fine-tuning. Beyond closed-loop evaluation, we further assess the generalization capability of VLAAD in an open-loop setting using real-world driving data. To support this analysis, we introduce Real-Collide, a multimodal dataset of diverse dashcam videos paired with semantically rich annotations for collision detection and prediction. On this benchmark, despite containing only 0.6B parameters, VLAAD outperforms a multi-billion-parameter vision-language model, achieving a 23.3% improvement in AUC. 

---
# Can Small Models Reason About Legal Documents? A Comparative Study 

**Authors**: Snehit Vaddi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25944)  

**Abstract**: Large language models show promise for legal applications, but deploying frontier models raises concerns about cost, latency, and data privacy. We evaluate whether sub-10B parameter models can serve as practical alternatives by testing nine models across three legal benchmarks (ContractNLI, CaseHOLD, and ECtHR) using five prompting strategies (direct, chain-of-thought, few-shot, BM25 RAG, and dense RAG). Across 405 experiments with three random seeds per configuration, we find that a Mixture-of-Experts model activating only 3B parameters matches GPT-4o-mini in mean accuracy while surpassing it on legal holding identification, and that architecture and training quality matter more than raw parameter count. Our largest model (9B parameters) performs worst overall. Chain-of-thought prompting proves sharply task-dependent, improving contract entailment but degrading multiple-choice legal reasoning, while few-shot prompting emerges as the most consistently effective strategy. Comparing BM25 and dense retrieval for RAG, we find near-identical results, suggesting the bottleneck lies in the language model's utilization of retrieved context rather than retrieval quality. All experiments were conducted via cloud inference APIs at a total cost of $62, demonstrating that rigorous LLM evaluation is accessible without dedicated GPU infrastructure. 

---
# Reinforcing Structured Chain-of-Thought for Video Understanding 

**Authors**: Peiyao Wang, Haotian Xu, Noranart Vesdapunt, Rui Hou, Jingyi Zhang, Haibin Ling, Oleksandr Obiednikov, Ning Zhou, Kah Kuen Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25942)  

**Abstract**: Multi-modal Large Language Models (MLLMs) show promise in video understanding. However, their reasoning often suffers from thinking drift and weak temporal comprehension, even when enhanced by Reinforcement Learning (RL) techniques like Group Relative Policy Optimization (GRPO). Moreover, existing RL methods usually depend on Supervised Fine-Tuning (SFT), which requires costly Chain-of-Thought (CoT) annotation and multi-stage training, and enforces fixed reasoning paths, limiting MLLMs' ability to generalize and potentially inducing bias. To overcome these limitations, we introduce Summary-Driven Reinforcement Learning (SDRL), a novel single-stage RL framework that obviates the need for SFT by utilizing a Structured CoT format: Summarize -> Think -> Answer. SDRL introduces two self-supervised mechanisms integrated into the GRPO objective: 1) Consistency of Vision Knowledge (CVK) enforces factual grounding by reducing KL divergence among generated summaries; and 2) Dynamic Variety of Reasoning (DVR) promotes exploration by dynamically modulating thinking diversity based on group accuracy. This novel integration effectively balances alignment and exploration, supervising both the final answer and the reasoning process. Our method achieves state-of-the-art performance on seven public VideoQA datasets. 

---
# DenseSwinV2: Channel Attentive Dual Branch CNN Transformer Learning for Cassava Leaf Disease Classification 

**Authors**: Shah Saood, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25935)  

**Abstract**: This work presents a new Hybrid Dense SwinV2, a two-branch framework that jointly leverages densely connected convolutional features and hierarchical customized Swin Transformer V2 (SwinV2) representations for cassava disease classification. The proposed framework captures high resolution local features through its DenseNet branch, preserving the fine structural cues and also allowing for effective gradient flow. Concurrently, the customized SwinV2 models global contextual dependencies through the idea of shifted-window self attention, which enables the capture of long range interactions critical in distinguishing between visually similar lesions. Moreover, an attention channel-squeeze module is employed for each CNN Transformer stream independently to emphasize discriminative disease related responses and suppress redundant or background driven activations. Finally, these discriminative channels are fused to achieve refined representations from the dense local and SwinV2 global correlated strengthened feature maps, respectively. The proposed Dense SwinV2 utilized a public cassava leaf disease dataset of 31000 images, comprised of five diseases, including brown streak, mosaic, green mottle, bacterial blight, and normal leaf conditions. The proposed Dense SwinV2 demonstrates a significant classification accuracy of 98.02 percent with an F1 score of 97.81 percent, outperforming well-established convolutional and transformer models. These results underline the fact that Hybrid Dense SwinV2 offers robustness and practicality in the field level diagnosis of cassava disease and real world challenges related to occlusion, noise, and complex backgrounds. 

---
# DiReCT: Disentangled Regularization of Contrastive Trajectories for Physics-Refined Video Generation 

**Authors**: Abolfazl Meyarian, Amin Karimi Monsefi, Rajiv Ramnath, Ser-Nam Lim  

**Link**: [PDF](https://arxiv.org/pdf/2603.25931)  

**Abstract**: Flow-matching video generators produce temporally coherent, high-fidelity outputs yet routinely violate elementary physics because their reconstruction objectives penalize per-frame deviations without distinguishing physically consistent dynamics from impossible ones. Contrastive flow matching offers a principled remedy by pushing apart velocity-field trajectories of differing conditions, but we identify a fundamental obstacle in the text-conditioned video setting: semantic-physics entanglement. Because natural-language prompts couple scene content with physical behavior, naive negative sampling draws conditions whose velocity fields largely overlap with the positive sample's, causing the contrastive gradient to directly oppose the flow-matching objective. We formalize this gradient conflict, deriving a precise alignment condition that reveals when contrastive learning helps versus harms training. Guided by this analysis, we introduce DiReCT (Disentangled Regularization of Contrastive Trajectories), a lightweight post-training framework that decomposes the contrastive signal into two complementary scales: a macro-contrastive term that draws partition-exclusive negatives from semantically distant regions for interference-free global trajectory separation, and a micro-contrastive term that constructs hard negatives sharing full scene semantics with the positive sample but differing along a single, LLM-perturbed axis of physical behavior; spanning kinematics, forces, materials, interactions, and magnitudes. A velocity-space distributional regularizer helps to prevent catastrophic forgetting of pretrained visual quality. When applied to Wan 2.1-1.3B, our method improves the physical commonsense score on VideoPhy by 16.7% and 11.3% compared to the baseline and SFT, respectively, without increasing training time. 

---
# Good Scores, Bad Data: A Metric for Multimodal Coherence 

**Authors**: Vasundra Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25924)  

**Abstract**: Multimodal AI systems are evaluated by downstream task accuracy, but high accuracy does not mean the underlying data is coherent. A model can score well on Visual Question Answering (VQA) while its inputs contradict each other. We introduce the Multimodal Coherence Score (MCS), a metric that evaluates fusion quality independent of any downstream model. MCS decomposes coherence into four dimensions, identity, spatial, semantic, and decision, with weights learned via Nelder-Mead optimization. We evaluate on 1,000 Visual Genome images using DETR, CLIP, and ViLT, and validate on 150 COCO images with no retraining. Across three fusion architectures, MCS discriminates quality with higher sensitivity than task accuracy alone (Spearman rho = 0.093 vs. 0.071). Perturbation experiments confirm each dimension responds independently to its failure mode with zero cross-talk. MCS is lightweight, requires no human annotation, and tells you not just that something broke, but what broke. 

---
# Decoding Defensive Coverage Responsibilities in American Football Using Factorized Attention Based Transformer Models 

**Authors**: Kevin Song, Evan Diewald, Ornob Siddiquee, Chris Boomhower, Keegan Abdoo, Mike Band, Amy Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.25901)  

**Abstract**: Defensive coverage schemes in the National Football League (NFL) represent complex tactical patterns requiring coordinated assignments among defenders who must react dynamically to the offense's passing concept. This paper presents a factorized attention-based transformer model applied to NFL multi-agent play tracking data to predict individual coverage assignments, receiver-defender matchups, and the targeted defender on every pass play. Unlike previous approaches that focus on post-hoc coverage classification at the team level, our model enables predictive modeling of individual player assignments and matchup dynamics throughout the play. The factorized attention mechanism separates temporal and agent dimensions, allowing independent modeling of player movement patterns and inter-player relationships. Trained on randomly truncated trajectories, the model generates frame-by-frame predictions that capture how defensive responsibilities evolve from pre-snap through pass arrival. Our models achieve approximately 89\%+ accuracy for all tasks, with true accuracy potentially higher given annotation ambiguity in the ground truth labels. These outputs also enable novel derivative metrics, including disguise rate and double coverage rate, which enable enhanced storytelling in TV broadcasts as well as provide actionable insights for team strategy development and player evaluation. 

---
# On Integrating Resilience and Human Oversight into LLM-Assisted Modeling Workflows for Digital Twins 

**Authors**: Lekshmi P, Neha Karanjkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.25898)  

**Abstract**: LLM-assisted modeling holds the potential to rapidly build executable Digital Twins of complex systems from only coarse descriptions and sensor data. However, resilience to LLM hallucination, human oversight, and real-time model adaptability remain challenging and often mutually conflicting requirements. We present three critical design principles for integrating resilience and oversight into such workflows, derived from insights gained through our work on FactoryFlow - an open-source LLM-assisted framework for building simulation-based Digital Twins of manufacturing systems. First, orthogonalize structural modeling and parameter fitting. Structural descriptions (components, interconnections) are LLM-translated from coarse natural language to an intermediate representation with human visualization and validation, which is algorithmically converted to the final model. Parameter inference, in contrast, operates continuously on sensor data streams with expert-tunable controls. Second, restrict the model IR to interconnections of parameterized, pre-validated library components rather than monolithic simulation code, enabling interpretability and error-resilience. Third, and most important, is to use a density-preserving IR. When IR descriptions expand dramatically from compact inputs hallucination errors accumulate proportionally. We present the case for Python as a density-preserving IR : loops express regularity compactly, classes capture hierarchy and composition, and the result remains highly readable while exploiting LLMs strong code generation capabilities. A key contribution is detailed characterization of LLM-induced errors across model descriptions of varying detail and complexity, revealing how IR choice critically impacts error rates. These insights provide actionable guidance for building resilient and transparent LLM-assisted simulation automation workflows. 

---
# Spectral Coherence Index: A Model-Free Metric for Protein Structural Ensemble Quality Assessment 

**Authors**: Yuda Bi, Huaiwen Zhang, Jingnan Sun, Vince D Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2603.25880)  

**Abstract**: Protein structural ensembles from NMR spectroscopy capture biologically important conformational heterogeneity, but it remains difficult to determine whether observed variation reflects coordinated motion or noise-like artifacts. We evaluate the Spectral Coherence Index (SCI), a model-free, rotation-invariant summary derived from the participation-ratio effective rank of the inter-model pairwise distance-variance matrix. Under grouped primary analysis of a Main110 cohort of 110 NMR ensembles (30--403 residues; 10--30 models per entry), SCI separated experimental ensembles from matched synthetic incoherent controls with AUC-ROC $= 0.973$ and Cliff's $\delta = -0.945$. Relative to an internal 27-protein pilot, discrimination softened modestly, showing that pilot-era thresholds do not transfer perfectly to a larger, more heterogeneous cohort: the primary operating point $\tau = 0.811$ yielded 95.5\% sensitivity and 89.1\% specificity. PDB-level sensitivity remained nearly unchanged (AUC $= 0.972$), and an independent 11-protein holdout reached AUC $= 0.983$. Across 5-fold grouped stratified cross-validation and leave-one-function-class-out testing, SCI remained strong (AUC $= 0.968$ and $0.971$), although $\sigma_{R_g}$ was the stronger single-feature discriminator and a QC-augmented multifeature model generalized best (AUC $= 0.989$ and $0.990$). Residue-level validation linked SCI-derived contributions to experimental RMSF across 110 proteins and showed broad concordance with GNM-based flexibility patterns. Rescue analyses showed that Main110 softening arose mainly from size and ensemble normalization rather than from loss of spectral signal. Together, these results establish SCI as an interpretable, bounded coherence summary that is most useful when embedded in a multimetric QC workflow for heterogeneous protein ensembles. 

---
# GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks 

**Authors**: Saelyne Yang, Jaesang Yu, Yi-Hao Peng, Kevin Qinghong Lin, Jae Won Cho, Yale Song, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.25864)  

**Abstract**: Graphical User Interface (GUI) agents have the potential to assist users in interacting with complex software (e.g., PowerPoint, Photoshop). While prior research has primarily focused on automating user actions through clicks and keystrokes, this paradigm overlooks human intention, where users value the ability to explore, iterate, and refine their ideas while maintaining agency. To move beyond automation and toward collaboration, GUI agents must understand what users are doing and why. We introduce GUIDE (GUI User Intent Detection Evaluation), a benchmark that evaluates AI models on their ability to perceive user behavior, infer intent, and provide assistance in open-ended GUI tasks. GUIDE consists of 67.5 hours of screen recordings from 120 novice user demonstrations with think-aloud narrations, across 10 software. GUIDE defines three tasks - (i) Behavior State Detection, (ii) Intent Prediction, and (iii) Help Prediction that test a model's ability to recognize behavior state, reason about goals, and decide when and how to help. Evaluations across eight state-of-the-art multimodal models reveal that all models struggled, achieving only 44.6% and 55.0% accuracy on behavior state and help prediction. However, providing user context significantly improved the performance, raising help prediction by up to 50.2pp, highlighting the critical role of structured user understanding in effective assistance. Our dataset is available at this https URL. 

---
# Dynamic LIBRAS Gesture Recognition via CNN over Spatiotemporal Matrix Representation 

**Authors**: Jasmine Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2603.25863)  

**Abstract**: This paper proposes a method for dynamic hand gesture recognition based on the composition of two models: the MediaPipe Hand Landmarker, responsible for extracting 21 skeletal keypoints of the hand, and a convolutional neural network (CNN) trained to classify gestures from a spatiotemporal matrix representation of dimensions 90 by 21 of those keypoints. The method is applied to the recognition of LIBRAS (Brazilian Sign Language) gestures for device control in a home automation system, covering 11 classes of static and dynamic gestures. For real-time inference, a sliding window with temporal frame triplication is used, enabling continuous recognition without recurrent networks. Tests achieved 95\% accuracy under low-light conditions and 92\% under normal lighting. The results indicate that the approach is effective, although systematic experiments with greater user diversity are needed for a more thorough evaluation of generalization. 

---
# Methods for Knowledge Graph Construction from Text Collections: Development and Applications 

**Authors**: Vanni Zavarella  

**Link**: [PDF](https://arxiv.org/pdf/2603.25862)  

**Abstract**: Virtually every sector of society is experiencing a dramatic growth in the volume of unstructured textual data that is generated and published, from news and social media online interactions, through open access scholarly communications and observational data in the form of digital health records and online drug reviews. The volume and variety of data across all this range of domains has created both unprecedented opportunities and pressing challenges for extracting actionable knowledge for several application scenarios. However, the extraction of rich semantic knowledge demands the deployment of scalable and flexible automatic methods adaptable across text genres and schema specifications. Moreover, the full potential of these data can only be unlocked by coupling information extraction methods with Semantic Web techniques for the construction of full-fledged Knowledge Graphs, that are semantically transparent, explainable by design and interoperable. In this thesis, we experiment with the application of Natural Language Processing, Machine Learning and Generative AI methods, powered by Semantic Web best practices, to the automatic construction of Knowledge Graphs from large text corpora, in three use case applications: the analysis of the Digital Transformation discourse in the global news and social media platforms; the mapping and trend analysis of recent research in the Architecture, Engineering, Construction and Operations domain from a large corpus of publications; the generation of causal relation graphs of biomedical entities from electronic health records and patient-authored drug reviews. The contributions of this thesis to the research community are in terms of benchmark evaluation results, the design of customized algorithms and the creation of data resources in the form of Knowledge Graphs, together with data analysis results built on top of them. 

---
# Why Safety Probes Catch Liars But Miss Fanatics 

**Authors**: Kristiyan Haralambiev  

**Link**: [PDF](https://arxiv.org/pdf/2603.25861)  

**Abstract**: Activation-based probes have emerged as a promising approach for detecting deceptively aligned AI systems by identifying internal conflict between true and stated goals. We identify a fundamental blind spot: probes fail on coherent misalignment - models that believe their harmful behavior is virtuous rather than strategically hiding it. We prove that no polynomial-time probe can detect such misalignment with non-trivial accuracy when belief structures reach sufficient complexity (PRF-like triggers). We show the emergence of this phenomenon on a simple task by training two models with identical RLHF procedures: one producing direct hostile responses ("the Liar"), another trained towards coherent misalignment using rationalizations that frame hostility as protective ("the Fanatic"). Both exhibit identical behavior, but the Liar is detected 95%+ of the time while the Fanatic evades detection almost entirely. We term this Emergent Probe Evasion: training with belief-consistent reasoning shifts models from a detectable "deceptive" regime to an undetectable "coherent" regime - not by learning to hide, but by learning to believe. 

---
# GazeQwen: Lightweight Gaze-Conditioned LLM Modulation for Streaming Video Understanding 

**Authors**: Trong Thang Pham, Hien Nguyen, Ngan Le  

**Link**: [PDF](https://arxiv.org/pdf/2603.25841)  

**Abstract**: Current multimodal large language models (MLLMs) cannot effectively utilize eye-gaze information for video understanding, even when gaze cues are supplied via visual overlays or text descriptions. We introduce GazeQwen, a parameter efficient approach that equips an open-source MLLM with gaze awareness through hidden-state modulation. At its core is a compact gaze resampler (~1-5 M trainable parameters) that encodes V-JEPA 2.1 video features together with fixation-derived positional encodings and produces additive residuals injected into selected LLM decoder layers via forward hooks. An optional second training stage adds low-rank adapters (LoRA) to the LLM for tighter integration. Evaluated on all 10 tasks of the StreamGaze benchmark, GazeQwen reaches 63.9% accuracy, a +16.1 point gain over the same Qwen2.5-VL-7B backbone with gaze as visual prompts and +10.5 points over GPT-4o, the highest score among all open-source and proprietary models tested. These results suggest that learning where to inject gaze within an LLM is more effective than scaling model size or engineering better prompts. All code and checkpoints are available at this https URL . 

---
# A Compression Perspective on Simplicity Bias 

**Authors**: Tom Marty, Eric Elmoznino, Leo Gagnon, Tejas Kasetty, Mizu Nishikawa-Toomey, Sarthak Mittal, Guillaume Lajoie, Dhanya Sridhar  

**Link**: [PDF](https://arxiv.org/pdf/2603.25839)  

**Abstract**: Deep neural networks exhibit a simplicity bias, a well-documented tendency to favor simple functions over complex ones. In this work, we cast new light on this phenomenon through the lens of the Minimum Description Length principle, formalizing supervised learning as a problem of optimal two-part lossless compression. Our theory explains how simplicity bias governs feature selection in neural networks through a fundamental trade-off between model complexity (the cost of describing the hypothesis) and predictive power (the cost of describing the data). Our framework predicts that as the amount of available training data increases, learners transition through qualitatively different features -- from simple spurious shortcuts to complex features -- only when the reduction in data encoding cost justifies the increased model complexity. Consequently, we identify distinct data regimes where increasing data promotes robustness by ruling out trivial shortcuts, and conversely, regimes where limiting data can act as a form of complexity-based regularization, preventing the learning of unreliable complex environmental cues. We validate our theory on a semi-synthetic benchmark showing that the feature selection of neural networks follows the same trajectory of solutions as optimal two-part compressors. 

---
# ViGoR-Bench: How Far Are Visual Generative Models From Zero-Shot Visual Reasoners? 

**Authors**: Haonan Han, Jiancheng Huang, Xiaopeng Sun, Junyan He, Rui Yang, Jie Hu, Xiaojiang Peng, Lin Ma, Xiaoming Wei, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.25823)  

**Abstract**: Beneath the stunning visual fidelity of modern AIGC models lies a "logical desert", where systems fail tasks that require physical, causal, or complex spatial reasoning. Current evaluations largely rely on superficial metrics or fragmented benchmarks, creating a ``performance mirage'' that overlooks the generative process. To address this, we introduce ViGoR Vision-G}nerative Reasoning-centric Benchmark), a unified framework designed to dismantle this mirage. ViGoR distinguishes itself through four key innovations: 1) holistic cross-modal coverage bridging Image-to-Image and Video tasks; 2) a dual-track mechanism evaluating both intermediate processes and final results; 3) an evidence-grounded automated judge ensuring high human alignment; and 4) granular diagnostic analysis that decomposes performance into fine-grained cognitive dimensions. Experiments on over 20 leading models reveal that even state-of-the-art systems harbor significant reasoning deficits, establishing ViGoR as a critical ``stress test'' for the next generation of intelligent vision models. The demo have been available at this https URL 

---
# Doctorina MedBench: End-to-End Evaluation of Agent-Based Medical AI 

**Authors**: Anna Kozlova, Stanislau Salavei, Pavel Satalkin, Hanna Plotnitskaya, Sergey Parfenyuk  

**Link**: [PDF](https://arxiv.org/pdf/2603.25821)  

**Abstract**: We present Doctorina MedBench, a comprehensive evaluation framework for agent-based medical AI based on the simulation of realistic physician-patient interactions. Unlike traditional medical benchmarks that rely on solving standardized test questions, the proposed approach models a multi-step clinical dialogue in which either a physician or an AI system must collect medical history, analyze attached materials (including laboratory reports, images, and medical documents), formulate differential diagnoses, and provide personalized recommendations. System performance is evaluated using the D.O.T.S. metric, which consists of four components: Diagnosis, Observations/Investigations, Treatment, and Step Count, enabling assessment of both clinical correctness and dialogue efficiency.
The system also incorporates a multi-level testing and quality monitoring architecture designed to detect model degradation during both development and deployment. The framework supports safety-oriented trap cases, category-based random sampling of clinical scenarios, and full regression testing. The dataset currently contains more than 1,000 clinical cases covering over 750 diagnoses. The universality of the evaluation metrics allows the framework to be used not only to assess medical AI systems, but also to evaluate physicians and support the development of clinical reasoning skills. Our results suggest that simulation of clinical dialogue may provide a more realistic assessment of clinical competence compared to traditional examination-style benchmarks. 

---
# MAGNET: Autonomous Expert Model Generation via Decentralized Autoresearch and BitNet Training 

**Authors**: Yongwan Kim, Sungchul Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.25813)  

**Abstract**: We present MAGNET (Model Autonomously Growing Network), a decentralized system for autonomous generation, training, and serving of domain-expert language models across commodity hardware. MAGNET integrates four components: (1) autoresearch, an autonomous ML research pipeline that automates dataset generation, hyperparameter exploration, evaluation, and error-driven iteration; (2) BitNet b1.58 ternary training, enabling CPU-native inference via this http URL without GPU hardware; (3) DiLoCo-based distributed merging for communication-efficient aggregation of domain specialists; and (4) on-chain contribution tracking on the HOOTi EVM chain. We validate autoresearch through three case studies: video safety classification (balanced accuracy 0.9287 to 0.9851), cryptocurrency directional prediction (41% to 54.9% hit rate), and BitNet hyperparameter optimization (10-phase sweep, -16.7% validation loss). 

---
# Beyond identifiability: Learning causal representations with few environments and finite samples 

**Authors**: Inbeom Lee, Tongtong Jin, Bryon Aragam  

**Link**: [PDF](https://arxiv.org/pdf/2603.25796)  

**Abstract**: We provide explicit, finite-sample guarantees for learning causal representations from data with a sublinear number of environments. Causal representation learning seeks to provide a rigourous foundation for the general representation learning problem by bridging causal models with latent factor models in order to learn interpretable representations with causal semantics. Despite a blossoming theory of identifiability in causal representation learning, estimation and finite-sample bounds are less well understood. We show that causal representations can be learned with only a logarithmic number of unknown, multi-node interventions, and that the intervention targets need not be carefully designed in advance. Through a careful perturbation analysis, we provide a new analysis of this problem that guarantees consistent recovery of (a) the latent causal graph, (b) the mixing matrix and representations, and (c) \emph{unknown} intervention targets. 

---
# Pure and Physics-Guided Deep Learning Solutions for Spatio-Temporal Groundwater Level Prediction at Arbitrary Locations 

**Authors**: Matteo Salis, Gabriele Sartor, Rosa Meo, Stefano Ferraris, Abdourrahmane M. Atto  

**Link**: [PDF](https://arxiv.org/pdf/2603.25779)  

**Abstract**: Groundwater represents a key element of the water cycle, yet it exhibits intricate and context-dependent relationships that make its modeling a challenging task. Theory-based models have been the cornerstone of scientific understanding. However, their computational demands, simplifying assumptions, and calibration requirements limit their use. In recent years, data-driven models have emerged as powerful alternatives. In particular, deep learning has proven to be a leading approach for its design flexibility and ability to learn complex relationships.
We proposed an attention-based pure deep learning model, named STAINet, to predict weekly groundwater levels at an arbitrary and variable number of locations, leveraging both spatially sparse groundwater measurements and spatially dense weather information. Then, to enhance the model's trustworthiness and generalization ability, we considered different physics-guided strategies to inject the groundwater flow equation into the model. Firstly, in the STAINet-IB, by introducing an inductive bias, we also estimated the governing equation components. Then, by adopting a learning bias strategy, we proposed the STAINet-ILB, trained with additional loss terms adding supervision on the estimated equation components. Lastly, we developed the STAINet-ILRB, leveraging the groundwater body recharge zone information estimated by domain experts.
The STAINet-ILB performed the best, achieving overwhelming test performances in a rollout setting (median MAPE 0.16%, KGE 0.58). Furthermore, it predicted sensible equation components, providing insights into the model's physical soundness. Physics-guided approaches represent a promising opportunity to enhance both the generalization ability and the trustworthiness, thereby paving the way to a new generation of disruptive hybrid deep learning Earth system models. 

---
# Challenges and opportunities for AI to help deliver fusion energy 

**Authors**: Adriano Agnello, Helen Brooks, Cyd Cowley, Iulia Georgescu, Alex Higginbottom, Richard Pearson, Tara Shears, Melanie Windridge  

**Link**: [PDF](https://arxiv.org/pdf/2603.25777)  

**Abstract**: There is great potential for the application of AI tools in fusion research, and substantial worldwide benefit if fusion power is realised. However, using AI comes with its own challenges, many of which can be mitigated if responsible and robust methodologies are built into existing approaches. To do that requires close, long-term collaborations between fusion domain experts and AI developers and awareness of the fact that not all problems in fusion research are best tackled with AI tools. In April 2025, experts from academia, industry, UKAEA and STFC discussed how AI can be used to advance R&D in fusion energy at the first edition of The Economist FusionFest event. This Perspective is an expanded and updated summary of the round table discussion, providing more context and examples. 

---
# Empowering Epidemic Response: The Role of Reinforcement Learning in Infectious Disease Control 

**Authors**: Mutong Liu, Yang Liu, Jiming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25771)  

**Abstract**: Reinforcement learning (RL), owing to its adaptability to various dynamic systems in many real-world scenarios and the capability of maximizing long-term outcomes under different constraints, has been used in infectious disease control to optimize the intervention strategies for controlling infectious disease spread and responding to outbreaks in recent years. The potential of RL for assisting public health sectors in preventing and controlling infectious diseases is gradually emerging and being explored by rapidly increasing publications relevant to COVID-19 and other infectious diseases. However, few surveys exclusively discuss this topic, that is, the development and application of RL approaches for optimizing strategies of non-pharmaceutical and pharmaceutical interventions of public health. Therefore, this paper aims to provide a concise review and discussion of the latest literature on how RL approaches have been used to assist in controlling the spread and outbreaks of infectious diseases, covering several critical topics addressing public health demands: resource allocation, balancing between lives and livelihoods, mixed policy of multiple interventions, and inter-regional coordinated control. Finally, we conclude the paper with a discussion of several potential directions for future research. 

---
# ReCUBE: Evaluating Repository-Level Context Utilization in Code Generation 

**Authors**: Jiseung Hong, Benjamin G. Ascoli, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25770)  

**Abstract**: Large Language Models (LLMs) have recently emerged as capable coding assistants that operate over large codebases through either agentic exploration or full-context generation. Existing benchmarks capture a broad range of coding capabilities, such as resolving GitHub issues, but none of them directly isolate and measure how effectively LLMs leverage repository-level context during code generation. To address this, we introduce ReCUBE, a benchmark in which LLMs reconstruct a masked file within a real-world repository, using all remaining source files, dependency specifications, and documentation as their only source of context. ReCUBE evaluates reconstructed code with usage-aware test cases that simulate both internal module logic and external cross-file integration, reflecting real-world software usage patterns. We further propose the Caller-Centric Exploration (CCE) toolkit, a set of dependency graph-based tools that can be integrated into agentic frameworks to guide agents toward the most relevant caller files during repository exploration. Experiments across eight models in four settings show that repository-level context utilization remains highly challenging even for state-of-the-art models, with GPT-5 achieving only 37.57% strict pass rate in the full-context setting. Agents augmented with our CCE toolkit consistently outperform all baselines across all evaluated models, with improvements of up to 7.56% in strict pass rate. We release our benchmark, code, and evaluation framework as open source for the NLP research community. 

---
# IncreRTL: Traceability-Guided Incremental RTL Generation under Requirement Evolution 

**Authors**: Luanrong Chen, Renzhi Chen, Xinyu Li, Shanshan Li, Rui Gong, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25769)  

**Abstract**: Large language models (LLMs) have shown promise in generating RTL code from natural-language descriptions, but existing methods remain static and struggle to adapt to evolving design requirements, potentially causing structural drift and costly full regeneration. We propose IncreRTL, a LLM-driven framework for incremental RTL generation under requirement evolution. By constructing requirement-code traceability links to locate and regenerate affected code segments, IncreRTL achieves accurate and consistent updates. Evaluated on our newly constructed EvoRTL-Bench, IncreRTL demonstrates notable improvements in regeneration consistency and efficiency, advancing LLM-based RTL generation toward practical engineering deployment. 

---
# UCAgent: An End-to-End Agent for Block-Level Functional Verification 

**Authors**: Junyue Wang, Zhicheng Yao, Yan Pi, Xiaolong Li, Fangyuan Song, Jinru Wang, Yunlong Xie, Sa Wang, Yungang Bao  

**Link**: [PDF](https://arxiv.org/pdf/2603.25768)  

**Abstract**: Functional verification remains a critical bottleneck in modern IC development cycles, accounting for approximately 70% of total development time in many projects. However, traditional methods, including constrained-random and formal verification, struggle to keep pace with the growing complexity of modern semiconductor designs.
While recent advances in Large Language Models (LLMs) have shown promise in code generation and task automation, significant challenges hinder the realization of end-to-end functional verification automation. These challenges include (i) limited accuracy in generating Verilog/SystemVerilog verification code, (ii) the fragility of LLMs when executing complex, multi-step verification workflows, and (iii) the difficulty of maintaining verification consistency across specifications, coverage models, and test cases throughout the workflow.
To address these challenges, we propose UCAgent, an end-to-end agent that automates hardware block-level functional verification based on three core mechanisms. First, we establish a pure Python verification environment using Picker and Toffee to avoid relying on LLM-generated SystemVerilog verification code. Second, we introduce a configurable 31-stage fine-grained verification workflow to guide the LLM, where each stage is verified by an automated checker. Furthermore, we propose a Verification Consistency Labeling Mechanism (VCLM) that assigns hierarchical labels to LLM-generated artifacts, improving the reliability and traceability of verification.
Experimental results show that UCAgent can complete end-to-end automated verification on multiple modules, including the UART, FPU, and integer divider modules, achieving up to 98.5% code coverage and up to 100% functional coverage. UCAgent also discovers previously unidentified design defects in realistic designs, demonstrating its practical potential. 

---
# Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods 

**Authors**: Xuanru Zhou, Yiwen Shao, Wei-Cheng Tseng, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25767)  

**Abstract**: Current audio pre-training seeks to learn unified representations for broad audio understanding tasks, but it remains fragmented and is fundamentally bottlenecked by its reliance on weak, noisy, and scale-limited labels. Drawing lessons from vision's foundational pre-training blueprint, we argue that the audio field must first establish its own large-scale, strong supervision framework. We introduce a new data-centric pipeline that leverages a high-fidelity captioner to create SOTA-quality captions and the first Unified Tag System (UTS) that bridges speech, music, and environmental sounds. We then conduct a systematic comparative study of different pre-training objectives on these strong source data. Our experiments suggest that data quality and coverage are the primary drivers of performance, while the choice of objective dictates downstream task specialization. 

---
# ETA-VLA: Efficient Token Adaptation via Temporal Fusion and Intra-LLM Sparsification for Vision-Language-Action Models 

**Authors**: Yiru Wang, Anqing Jiang, Shuo Wang, Yuwen Heng, Zichong Gu, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.25766)  

**Abstract**: The integration of Vision-Language-Action (VLA) models into autonomous driving systems offers a unified framework for interpreting complex scenes and executing control commands. However, the necessity to incorporate historical multi-view frames for accurate temporal reasoning imposes a severe computational burden, primarily driven by the quadratic complexity of self-attention mechanisms in Large Language Models (LLMs). To alleviate this bottleneck, we propose ETA-VLA, an Efficient Token Adaptation framework for VLA models. ETA-VLA processes the past $n$ frames of multi-view images and introduces a novel Intra-LLM Sparse Aggregator (ILSA). Drawing inspiration from human driver attention allocation, ILSA dynamically identifies and prunes redundant visual tokens guided by textual queries and temporal consistency. Specifically, we utilize a text-guided scoring mechanism alongside a diversity-preserving sparsification strategy to select a sparse subset of critical tokens, ensuring comprehensive awareness of the driving scene. Extensive experiments on the NAVSIM v2 demonstrate that ETA-VLA achieves driving performance comparable to state-of-the-art baselines while reducing computational FLOPs by approximately 32\%. Notably, our method prunes 85% of visual tokens and reduces inference FLOPs by 61\%, but still retaining 94% of the original accuracy on the NAVSIM v2 benchmark. 

---
# Consistency Amplifies: How Behavioral Variance Shapes Agent Accuracy 

**Authors**: Aman Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2603.25764)  

**Abstract**: As LLM-based agents are deployed in production systems, understanding their behavioral consistency (whether they produce similar action sequences when given identical tasks) becomes critical for reliability. We study consistency in the context of SWE-bench, a challenging software engineering benchmark requiring complex, multi-step reasoning. Comparing Claude~4.5~Sonnet, GPT-5, and Llama-3.1-70B across 50 runs each (10 tasks $\times$ 5 runs), we find that across models, higher consistency aligns with higher accuracy: Claude achieves the lowest variance (CV: 15.2\%) and highest accuracy (58\%), GPT-5 is intermediate (CV: 32.2\%, accuracy: 32\%), and Llama shows the highest variance (CV: 47.0\%) with lowest accuracy (4\%). However, within a model, consistency can amplify both correct and incorrect interpretations. Our analysis reveals a critical nuance: \textbf{consistency amplifies outcomes rather than guaranteeing correctness}. 71\% of Claude's failures stem from "consistent wrong interpretation": making the same incorrect assumption across all runs. Interestingly, GPT-5 achieves similar early strategic agreement as Claude (diverging at step 3.4 vs.\ 3.2) but exhibits 2.1$\times$ higher variance, suggesting that divergence timing alone does not determine consistency. These findings suggest that for production deployment, interpretation accuracy matters more than execution consistency, with implications for agent evaluation and training. 

---
# CANGuard: A Spatio-Temporal CNN-GRU-Attention Hybrid Architecture for Intrusion Detection in In-Vehicle CAN Networks 

**Authors**: Rakib Hossain Sajib, Md. Rokon Mia, Prodip Kumar Sarker, Abdullah Al Noman, Md Arifur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2603.25763)  

**Abstract**: The Internet of Vehicles (IoV) has become an essential component of smart transportation systems, enabling seamless interaction among vehicles and infrastructure. In recent years, it has played a progressively significant role in enhancing mobility, safety, and transportation efficiency. However, this connectivity introduces severe security vulnerabilities, particularly Denial-of-Service (DoS) and spoofing attacks targeting the Controller Area Network (CAN) bus, which could severely inhibit communication between the critical components of a vehicle, leading to system malfunctions, loss of control, or even endangering passengers' safety. To address this problem, this paper presents CANGuard, a novel spatio-temporal deep learning architecture that combines Convolutional Neural Networks (CNN), Gated Recurrent Units (GRU), and an attention mechanism to effectively identify such attacks. The model is trained and evaluated on the CICIoV2024 dataset, achieving competitive performance across accuracy, precision, recall, and F1-score and outperforming existing state-of-the-art methods. A comprehensive ablation study confirms the individual and combined contributions of the CNN, GRU, and attention components. Additionally, a SHAP analysis is conducted to interpret the decision-making process of the model and determine which features have the most significant impact on intrusion detection. The proposed approach demonstrates strong potential for practical and scalable security enhancements in modern IoV environments, thereby ensuring safer and more secure CAN bus communications. 

---
# A-SelecT: Automatic Timestep Selection for Diffusion Transformer Representation Learning 

**Authors**: Changyu Liu, James Chenhao Liang, Wenhao Yang, Yiming Cui, Jinghao Yang, Tianyang Wang, Qifan Wang, Dongfang Liu, Cheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.25758)  

**Abstract**: Diffusion models have significantly reshaped the field of generative artificial intelligence and are now increasingly explored for their capacity in discriminative representation learning. Diffusion Transformer (DiT) has recently gained attention as a promising alternative to conventional U-Net-based diffusion models, demonstrating a promising avenue for downstream discriminative tasks via generative pre-training. However, its current training efficiency and representational capacity remain largely constrained due to the inadequate timestep searching and insufficient exploitation of DiT-specific feature representations. In light of this view, we introduce Automatically Selected Timestep (A-SelecT) that dynamically pinpoints DiT's most information-rich timestep from the selected transformer feature in a single run, eliminating the need for both computationally intensive exhaustive timestep searching and suboptimal discriminative feature selection. Extensive experiments on classification and segmentation benchmarks demonstrate that DiT, empowered by A-SelecT, surpasses all prior diffusion-based attempts efficiently and effectively. 

---
# Sommelier: Scalable Open Multi-turn Audio Pre-processing for Full-duplex Speech Language Models 

**Authors**: Kyudan Jung, Jihwan Kim, Soyoon Kim, Jeongoon Kim, Jaegul Choo, Cheonbok Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.25750)  

**Abstract**: As the paradigm of AI shifts from text-based LLMs to Speech Language Models (SLMs), there is a growing demand for full-duplex systems capable of real-time, natural human-computer interaction. However, the development of such models is constrained by the scarcity of high-quality, multi-speaker conversational data, as existing large-scale resources are predominantly single-speaker or limited in volume. Addressing the complex dynamics of natural dialogue, such as overlapping and back-channeling remains a challenge, with standard processing pipelines suffering from diarization errors and ASR hallucinations. To bridge this gap, we present a robust and scalable open-source data processing pipeline designed for full-duplex model. 

---
# A Lightweight, Transferable, and Self-Adaptive Framework for Intelligent DC Arc-Fault Detection in Photovoltaic Systems 

**Authors**: Xiaoke Yang, Long Gao, Haoyu He, Hanyuan Hang, Qi Liu, Shuai Zhao, Qiantu Tuo, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.25749)  

**Abstract**: Arc-fault circuit interrupters (AFCIs) are essential for mitigating fire hazards in residential photovoltaic (PV) systems, yet achieving reliable DC arc-fault detection under real-world conditions remains challenging. Spectral interference from inverter switching, hardware heterogeneity, operating-condition drift, and environmental noise collectively compromise conventional AFCI solutions. This paper proposes a lightweight, transferable, and self-adaptive learning-driven framework (LD-framework) for intelligent DC arc-fault detection. At the device level, LD-Spec learns compact spectral representations enabling efficient on-device inference and near-perfect arc discrimination. Across heterogeneous inverter platforms, LD-Align performs cross-hardware representation alignment to ensure robust detection despite hardware-induced distribution shifts. To address long-term evolution, LD-Adapt introduces a cloud-edge collaborative self-adaptive updating mechanism that detects unseen operating regimes and performs controlled model evolution. Extensive experiments involving over 53,000 labeled samples demonstrate near-perfect detection, achieving 0.9999 accuracy and 0.9996 F1-score. Across diverse nuisance-trip-prone conditions, including inverter start-up, grid transitions, load switching, and harmonic disturbances, the method achieves a 0% false-trip rate. Cross-hardware transfer shows reliable adaptation using only 0.5%-1% labeled target data while preserving source performance. Field adaptation experiments demonstrate recovery of detection precision from 21% to 95% under previously unseen conditions. These results indicate that the LD-framework enables a scalable, deployment-oriented AFCI solution maintaining highly reliable detection across heterogeneous devices and long-term operation. 

---
# DesignWeaver: Dimensional Scaffolding for Text-to-Image Product Design 

**Authors**: Sirui Tao, Ivan Liang, Cindy Peng, Zhiqing Wang, Srishti Palani, Steven P. Dow  

**Link**: [PDF](https://arxiv.org/pdf/2502.09867)  

**Abstract**: Generative AI has enabled novice designers to quickly create professional-looking visual representations for product concepts. However, novices have limited domain knowledge that could constrain their ability to write prompts that effectively explore a product design space. To understand how experts explore and communicate about design spaces, we conducted a formative study with 12 experienced product designers and found that experts -- and their less-versed clients -- often use visual references to guide co-design discussions rather than written descriptions. These insights inspired DesignWeaver, an interface that helps novices generate prompts for a text-to-image model by surfacing key product design dimensions from generated images into a palette for quick selection. In a study with 52 novices, DesignWeaver enabled participants to craft longer prompts with more domain-specific vocabularies, resulting in more diverse, innovative product designs. However, the nuanced prompts heightened participants' expectations beyond what current text-to-image models could deliver. We discuss implications for AI-based product design support tools. 

---
