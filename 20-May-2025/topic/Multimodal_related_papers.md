# AdaToken-3D: Dynamic Spatial Gating for Efficient 3D Large Multimodal-Models Reasoning 

**Authors**: Kai Zhang, Xingyu Chen, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12782)  

**Abstract**: Large Multimodal Models (LMMs) have become a pivotal research focus in deep learning, demonstrating remarkable capabilities in 3D scene understanding. However, current 3D LMMs employing thousands of spatial tokens for multimodal reasoning suffer from critical inefficiencies: excessive computational overhead and redundant information flows. Unlike 2D VLMs processing single images, 3D LMMs exhibit inherent architectural redundancy due to the heterogeneous mechanisms between spatial tokens and visual tokens. To address this challenge, we propose AdaToken-3D, an adaptive spatial token optimization framework that dynamically prunes redundant tokens through spatial contribution analysis. Our method automatically tailors pruning strategies to different 3D LMM architectures by quantifying token-level information flows via attention pattern mining. Extensive experiments on LLaVA-3D (a 7B parameter 3D-LMM) demonstrate that AdaToken-3D achieves 21\% faster inference speed and 63\% FLOPs reduction while maintaining original task accuracy. Beyond efficiency gains, this work systematically investigates redundancy patterns in multimodal spatial information flows through quantitative token interaction analysis. Our findings reveal that over 60\% of spatial tokens contribute minimally ($<$5\%) to the final predictions, establishing theoretical foundations for efficient 3D multimodal learning. 

---
# MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision 

**Authors**: Lingxiao Du, Fanqing Meng, Zongkai Liu, Zhixiang Zhou, Ping Luo, Qiaosheng Zhang, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13427)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have achieved impressive progress in vision-language understanding, they still struggle with complex multi-step reasoning, often producing logically inconsistent or partially correct solutions. A key limitation lies in the lack of fine-grained supervision over intermediate reasoning steps. To address this, we propose MM-PRM, a process reward model trained within a fully automated, scalable framework. We first build MM-Policy, a strong multimodal model trained on diverse mathematical reasoning data. Then, we construct MM-K12, a curated dataset of 10,000 multimodal math problems with verifiable answers, which serves as seed data. Leveraging a Monte Carlo Tree Search (MCTS)-based pipeline, we generate over 700k step-level annotations without human labeling. The resulting PRM is used to score candidate reasoning paths in the Best-of-N inference setup and achieves significant improvements across both in-domain (MM-K12 test set) and out-of-domain (OlympiadBench, MathVista, etc.) benchmarks. Further analysis confirms the effectiveness of soft labels, smaller learning rates, and path diversity in optimizing PRM performance. MM-PRM demonstrates that process supervision is a powerful tool for enhancing the logical robustness of multimodal reasoning systems. We release all our codes and data at this https URL. 

---
# MindOmni: Unleashing Reasoning Generation in Vision Language Models with RGPO 

**Authors**: Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13031)  

**Abstract**: Recent text-to-image systems face limitations in handling multimodal inputs and complex reasoning tasks. We introduce MindOmni, a unified multimodal large language model that addresses these challenges by incorporating reasoning generation through reinforcement learning. MindOmni leverages a three-phase training strategy: i) design of a unified vision language model with a decoder-only diffusion module, ii) supervised fine-tuning with Chain-of-Thought (CoT) instruction data, and iii) our proposed Reasoning Generation Policy Optimization (RGPO) algorithm, utilizing multimodal feedback to effectively guide policy updates. Experimental results demonstrate that MindOmni outperforms existing models, achieving impressive performance on both understanding and generation benchmarks, meanwhile showcasing advanced fine-grained reasoning generation capabilities, especially with mathematical reasoning instruction. All codes will be made public at \href{this https URL}{this https URL}. 

---
# FRAbench and GenEval: Scaling Fine-Grained Aspect Evaluation across Tasks, Modalities 

**Authors**: Shibo Hong, Jiahao Ying, Haiyuan Liang, Mengdi Zhang, Jun Kuang, Jiazheng Zhang, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12795)  

**Abstract**: Evaluating the open-ended outputs of large language models (LLMs) has become a bottleneck as model capabilities, task diversity, and modality coverage rapidly expand. Existing "LLM-as-a-Judge" evaluators are typically narrow in a few tasks, aspects, or modalities, and easily suffer from low consistency. In this paper, we argue that explicit, fine-grained aspect specification is the key to both generalizability and objectivity in automated evaluation. To do so, we introduce a hierarchical aspect taxonomy spanning 112 aspects that unifies evaluation across four representative settings - Natural Language Generation, Image Understanding, Image Generation, and Interleaved Text-and-Image Generation. Building on this taxonomy, we create FRAbench, a benchmark comprising 60.4k pairwise samples with 325k aspect-level labels obtained from a combination of human and LLM annotations. FRAbench provides the first large-scale, multi-modal resource for training and meta-evaluating fine-grained LMM judges. Leveraging FRAbench, we develop GenEval, a fine-grained evaluator generalizable across tasks and modalities. Experiments show that GenEval (i) attains high agreement with GPT-4o and expert annotators, (ii) transfers robustly to unseen tasks and modalities, and (iii) reveals systematic weaknesses of current LMMs on evaluation. 

---
# Correspondence of high-dimensional emotion structures elicited by video clips between humans and Multimodal LLMs 

**Authors**: Haruka Asanuma, Naoko Koide-Majima, Ken Nakamura, Takato Horii, Shinji Nishimoto, Masafumi Oizumi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12746)  

**Abstract**: Recent studies have revealed that human emotions exhibit a high-dimensional, complex structure. A full capturing of this complexity requires new approaches, as conventional models that disregard high dimensionality risk overlooking key nuances of human emotions. Here, we examined the extent to which the latest generation of rapidly evolving Multimodal Large Language Models (MLLMs) capture these high-dimensional, intricate emotion structures, including capabilities and limitations. Specifically, we compared self-reported emotion ratings from participants watching videos with model-generated estimates (e.g., Gemini or GPT). We evaluated performance not only at the individual video level but also from emotion structures that account for inter-video relationships. At the level of simple correlation between emotion structures, our results demonstrated strong similarity between human and model-inferred emotion structures. To further explore whether the similarity between humans and models is at the signle item level or the coarse-categorical level, we applied Gromov Wasserstein Optimal Transport. We found that although performance was not necessarily high at the strict, single-item level, performance across video categories that elicit similar emotions was substantial, indicating that the model could infer human emotional experiences at the category level. Our results suggest that current state-of-the-art MLLMs broadly capture the complex high-dimensional emotion structures at the category level, as well as their apparent limitations in accurately capturing entire structures at the single-item level. 

---
# Incentivizing Multimodal Reasoning in Large Models for Direct Robot Manipulation 

**Authors**: Weiliang Tang, Dong Jing, Jia-Hui Pan, Zhiwu Lu, Yun-Hui Liu, Li Erran Li, Mingyu Ding, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12744)  

**Abstract**: Recent Large Multimodal Models have demonstrated remarkable reasoning capabilities, especially in solving complex mathematical problems and realizing accurate spatial perception. Our key insight is that these emerging abilities can naturally extend to robotic manipulation by enabling LMMs to directly infer the next goal in language via reasoning, rather than relying on a separate action head. However, this paradigm meets two main challenges: i) How to make LMMs understand the spatial action space, and ii) How to fully exploit the reasoning capacity of LMMs in solving these tasks. To tackle the former challenge, we propose a novel task formulation, which inputs the current states of object parts and the gripper, and reformulates rotation by a new axis representation instead of traditional Euler angles. This representation is more compatible with spatial reasoning and easier to interpret within a unified language space. For the latter challenge, we design a pipeline to utilize cutting-edge LMMs to generate a small but high-quality reasoning dataset of multi-round dialogues that successfully solve manipulation tasks for supervised fine-tuning. Then, we perform reinforcement learning by trial-and-error interactions in simulation to further enhance the model's reasoning abilities for robotic manipulation. Our resulting reasoning model built upon a 7B backbone, named ReasonManip, demonstrates three notable advantages driven by its system-2 level reasoning capabilities: i) exceptional generalizability to out-of-distribution environments, objects, and tasks; ii) inherent sim-to-real transfer ability enabled by the unified language representation shared across domains; iii) transparent interpretability connecting high-level reasoning and low-level control. Extensive experiments demonstrate the effectiveness of the proposed paradigm and its potential to advance LMM-driven robotic manipulation. 

---
# CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World 

**Authors**: Zoya Volovikova, Gregory Gorbov, Petr Kuderov, Aleksandr I. Panov, Alexey Skrynnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.11962)  

**Abstract**: Following instructions in real-world conditions requires the ability to adapt to the world's volatility and entanglement: the environment is dynamic and unpredictable, instructions can be linguistically complex with diverse vocabulary, and the number of possible goals an agent may encounter is vast. Despite extensive research in this area, most studies are conducted in static environments with simple instructions and a limited vocabulary, making it difficult to assess agent performance in more diverse and challenging settings. To address this gap, we introduce CrafText, a benchmark for evaluating instruction following in a multimodal environment with diverse instructions and dynamic interactions. CrafText includes 3,924 instructions with 3,423 unique words, covering Localization, Conditional, Building, and Achievement tasks. Additionally, we propose an evaluation protocol that measures an agent's ability to generalize to novel instruction formulations and dynamically evolving task configurations, providing a rigorous test of both linguistic understanding and adaptive decision-making. 

---
# Diverging Towards Hallucination: Detection of Failures in Vision-Language Models via Multi-token Aggregation 

**Authors**: Geigh Zollicoffer, Minh Vu, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11741)  

**Abstract**: Vision-language models (VLMs) now rival human performance on many multimodal tasks, yet they still hallucinate objects or generate unsafe text. Current hallucination detectors, e.g., single-token linear probing (SLP) and P(True), typically analyze only the logit of the first generated token or just its highest scoring component overlooking richer signals embedded within earlier token distributions. We demonstrate that analyzing the complete sequence of early logits potentially provides substantially more diagnostic information. We emphasize that hallucinations may only emerge after several tokens, as subtle inconsistencies accumulate over time. By analyzing the Kullback-Leibler (KL) divergence between logits corresponding to hallucinated and non-hallucinated tokens, we underscore the importance of incorporating later-token logits to more accurately capture the reliability dynamics of VLMs. In response, we introduce Multi-Token Reliability Estimation (MTRE), a lightweight, white-box method that aggregates logits from the first ten tokens using multi-token log-likelihood ratios and self-attention. Despite the challenges posed by large vocabulary sizes and long logit sequences, MTRE remains efficient and tractable. On MAD-Bench, MM-SafetyBench, MathVista, and four compositional-geometry benchmarks, MTRE improves AUROC by 9.4 +/- 1.3 points over SLP and by 12.1 +/- 1.7 points over P(True), setting a new state-of-the-art in hallucination detection for open-source VLMs. 

---
# Foundation Models for AI-Enabled Biological Design 

**Authors**: Asher Moldwin, Amarda Shehu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11610)  

**Abstract**: This paper surveys foundation models for AI-enabled biological design, focusing on recent developments in applying large-scale, self-supervised models to tasks such as protein engineering, small molecule design, and genomic sequence design. Though this domain is evolving rapidly, this survey presents and discusses a taxonomy of current models and methods. The focus is on challenges and solutions in adapting these models for biological applications, including biological sequence modeling architectures, controllability in generation, and multi-modal integration. The survey concludes with a discussion of open problems and future directions, offering concrete next-steps to improve the quality of biological sequence generation. 

---
# RBF++: Quantifying and Optimizing Reasoning Boundaries across Measurable and Unmeasurable Capabilities for Chain-of-Thought Reasoning 

**Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, Yue Liao, Jiaqi Wang, Jingxuan Zhou, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2505.13307)  

**Abstract**: Chain-of-Thought (CoT) reasoning has proven effective in enhancing large language models (LLMs) on complex tasks, spurring research into its underlying mechanisms. However, two primary challenges remain for real-world applications: (1) the lack of quantitative metrics and actionable guidelines for evaluating and optimizing measurable boundaries of CoT capability, and (2) the absence of methods to assess boundaries of unmeasurable CoT capability, such as multimodal perception. To address these gaps, we introduce the Reasoning Boundary Framework++ (RBF++). To tackle the first challenge, we define the reasoning boundary (RB) as the maximum limit of CoT performance. We also propose a combination law for RBs, enabling quantitative analysis and offering actionable guidance across various CoT tasks. For the second challenge, particularly in multimodal scenarios, we introduce a constant assumption, which replaces unmeasurable RBs with scenario-specific constants. Additionally, we propose the reasoning boundary division mechanism, which divides unmeasurable RBs into two sub-boundaries, facilitating the quantification and optimization of both unmeasurable domain knowledge and multimodal perception capabilities. Extensive experiments involving 38 models across 13 tasks validate the feasibility of our framework in cross-modal settings. Additionally, we evaluate 10 CoT strategies, offer insights into optimization and decay from two complementary perspectives, and expand evaluation benchmarks for measuring RBs in LLM reasoning. We hope this work advances the understanding of RBs and optimization strategies in LLMs. Code and data are available at this https URL. 

---
# Picturized and Recited with Dialects: A Multimodal Chinese Representation Framework for Sentiment Analysis of Classical Chinese Poetry 

**Authors**: Xiaocong Du, Haoyu Pei, Haipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13210)  

**Abstract**: Classical Chinese poetry is a vital and enduring part of Chinese literature, conveying profound emotional resonance. Existing studies analyze sentiment based on textual meanings, overlooking the unique rhythmic and visual features inherent in poetry,especially since it is often recited and accompanied by Chinese paintings. In this work, we propose a dialect-enhanced multimodal framework for classical Chinese poetry sentiment analysis. We extract sentence-level audio features from the poetry and incorporate audio from multiple dialects,which may retain regional ancient Chinese phonetic features, enriching the phonetic representation. Additionally, we generate sentence-level visual features, and the multimodal features are fused with textual features enhanced by LLM translation through multimodal contrastive representation learning. Our framework outperforms state-of-the-art methods on two public datasets, achieving at least 2.51% improvement in accuracy and 1.63% in macro F1. We open-source the code to facilitate research in this area and provide insights for general multimodal Chinese representation. 

---
# Just Dance with $π$! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection 

**Authors**: Snehashis Majhi, Giacomo D'Amicantonio, Antitza Dantcheva, Quan Kong, Lorenzo Garattoni, Gianpiero Francesca, Egor Bondarev, Francois Bremond  

**Link**: [PDF](https://arxiv.org/pdf/2505.13123)  

**Abstract**: Weakly-supervised methods for video anomaly detection (VAD) are conventionally based merely on RGB spatio-temporal features, which continues to limit their reliability in real-world scenarios. This is due to the fact that RGB-features are not sufficiently distinctive in setting apart categories such as shoplifting from visually similar events. Therefore, towards robust complex real-world VAD, it is essential to augment RGB spatio-temporal features by additional modalities. Motivated by this, we introduce the Poly-modal Induced framework for VAD: "PI-VAD", a novel approach that augments RGB representations by five additional modalities. Specifically, the modalities include sensitivity to fine-grained motion (Pose), three dimensional scene and entity representation (Depth), surrounding objects (Panoptic masks), global motion (optical flow), as well as language cues (VLM). Each modality represents an axis of a polygon, streamlined to add salient cues to RGB. PI-VAD includes two plug-in modules, namely Pseudo-modality Generation module and Cross Modal Induction module, which generate modality-specific prototypical representation and, thereby, induce multi-modal information into RGB cues. These modules operate by performing anomaly-aware auxiliary tasks and necessitate five modality backbones -- only during training. Notably, PI-VAD achieves state-of-the-art accuracy on three prominent VAD datasets encompassing real-world scenarios, without requiring the computational overhead of five modality backbones at inference. 

---
# Cross-modal Knowledge Transfer Learning as Graph Matching Based on Optimal Transport for ASR 

**Authors**: Xugang Lu, Peng Shen, Yu Tsao, Hisashi Kawai  

**Link**: [PDF](https://arxiv.org/pdf/2505.13079)  

**Abstract**: Transferring linguistic knowledge from a pretrained language model (PLM) to acoustic feature learning has proven effective in enhancing end-to-end automatic speech recognition (E2E-ASR). However, aligning representations between linguistic and acoustic modalities remains a challenge due to inherent modality gaps. Optimal transport (OT) has shown promise in mitigating these gaps by minimizing the Wasserstein distance (WD) between linguistic and acoustic feature distributions. However, previous OT-based methods overlook structural relationships, treating feature vectors as unordered sets. To address this, we propose Graph Matching Optimal Transport (GM-OT), which models linguistic and acoustic sequences as structured graphs. Nodes represent feature embeddings, while edges capture temporal and sequential relationships. GM-OT minimizes both WD (between nodes) and Gromov-Wasserstein distance (GWD) (between edges), leading to a fused Gromov-Wasserstein distance (FGWD) formulation. This enables structured alignment and more efficient knowledge transfer compared to existing OT-based approaches. Theoretical analysis further shows that prior OT-based methods in linguistic knowledge transfer can be viewed as a special case within our GM-OT framework. We evaluate GM-OT on Mandarin ASR using a CTC-based E2E-ASR system with a PLM for knowledge transfer. Experimental results demonstrate significant performance gains over state-of-the-art models, validating the effectiveness of our approach. 

---
# Multiscale Adaptive Conflict-Balancing Model For Multimedia Deepfake Detection 

**Authors**: Zihan Xiong, Xiaohua Wu, Lei Chen, Fangqi Lou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12966)  

**Abstract**: Advances in computer vision and deep learning have blurred the line between deepfakes and authentic media, undermining multimedia credibility through audio-visual forgery. Current multimodal detection methods remain limited by unbalanced learning between modalities. To tackle this issue, we propose an Audio-Visual Joint Learning Method (MACB-DF) to better mitigate modality conflicts and neglect by leveraging contrastive learning to assist in multi-level and cross-modal fusion, thereby fully balancing and exploiting information from each modality. Additionally, we designed an orthogonalization-multimodal pareto module that preserves unimodal information while addressing gradient conflicts in audio-video encoders caused by differing optimization targets of the loss functions. Extensive experiments and ablation studies conducted on mainstream deepfake datasets demonstrate consistent performance gains of our model across key evaluation metrics, achieving an average accuracy of 95.5% across multiple datasets. Notably, our method exhibits superior cross-dataset generalization capabilities, with absolute improvements of 8.0% and 7.7% in ACC scores over the previous best-performing approach when trained on DFDC and tested on DefakeAVMiT and FakeAVCeleb datasets. 

---
# TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks 

**Authors**: Yuanze Hu, Zhaoxin Fan, Xinyu Wang, Gen Li, Ye Qiu, Zhichao Yang, Wenjun Wu, Kejian Wu, Yifan Sun, Xiaotie Deng, Jin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12884)  

**Abstract**: Lightweight Vision-Language Models (VLMs) are indispensable for resource-constrained applications. The prevailing approach to aligning vision and language models involves freezing both the vision encoder and the language model while training small connector modules. However, this strategy heavily depends on the intrinsic capabilities of the language model, which can be suboptimal for lightweight models with limited representational capacity. In this work, we investigate this alignment bottleneck through the lens of mutual information, demonstrating that the constrained capacity of the language model inherently limits the Effective Mutual Information (EMI) between multimodal inputs and outputs, thereby compromising alignment quality. To address this challenge, we propose TinyAlign, a novel framework inspired by Retrieval-Augmented Generation, which strategically retrieves relevant context from a memory bank to enrich multimodal inputs and enhance their alignment. Extensive empirical evaluations reveal that TinyAlign significantly reduces training loss, accelerates convergence, and enhances task performance. Remarkably, it allows models to achieve baseline-level performance with only 40\% of the fine-tuning data, highlighting exceptional data efficiency. Our work thus offers a practical pathway for developing more capable lightweight VLMs while introducing a fresh theoretical lens to better understand and address alignment bottlenecks in constrained multimodal systems. 

---
# AutoGEEval: A Multimodal and Automated Framework for Geospatial Code Generation on GEE with Large Language Models 

**Authors**: Shuyang Hou, Zhangxiao Shen, Huayi Wu, Jianyuan Liang, Haoyue Jiao, Yaxian Qing, Xiaopu Zhang, Xu Li, Zhipeng Gui, Xuefeng Guan, Longgang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12900)  

**Abstract**: Geospatial code generation is emerging as a key direction in the integration of artificial intelligence and geoscientific analysis. However, there remains a lack of standardized tools for automatic evaluation in this domain. To address this gap, we propose AutoGEEval, the first multimodal, unit-level automated evaluation framework for geospatial code generation tasks on the Google Earth Engine (GEE) platform powered by large language models (LLMs). Built upon the GEE Python API, AutoGEEval establishes a benchmark suite (AutoGEEval-Bench) comprising 1325 test cases that span 26 GEE data types. The framework integrates both question generation and answer verification components to enable an end-to-end automated evaluation pipeline-from function invocation to execution validation. AutoGEEval supports multidimensional quantitative analysis of model outputs in terms of accuracy, resource consumption, execution efficiency, and error types. We evaluate 18 state-of-the-art LLMs-including general-purpose, reasoning-augmented, code-centric, and geoscience-specialized models-revealing their performance characteristics and potential optimization pathways in GEE code generation. This work provides a unified protocol and foundational resource for the development and assessment of geospatial code generation models, advancing the frontier of automated natural language to domain-specific code translation. 

---
# Unified Cross-modal Translation of Score Images, Symbolic Music, and Performance Audio 

**Authors**: Jongmin Jung, Dongmin Kim, Sihun Lee, Seola Cho, Hyungjoon Soh, Irmak Bukey, Chris Donahue, Dasaem Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12863)  

**Abstract**: Music exists in various modalities, such as score images, symbolic scores, MIDI, and audio. Translations between each modality are established as core tasks of music information retrieval, such as automatic music transcription (audio-to-MIDI) and optical music recognition (score image to symbolic score). However, most past work on multimodal translation trains specialized models on individual translation tasks. In this paper, we propose a unified approach, where we train a general-purpose model on many translation tasks simultaneously. Two key factors make this unified approach viable: a new large-scale dataset and the tokenization of each modality. Firstly, we propose a new dataset that consists of more than 1,300 hours of paired audio-score image data collected from YouTube videos, which is an order of magnitude larger than any existing music modal translation datasets. Secondly, our unified tokenization framework discretizes score images, audio, MIDI, and MusicXML into a sequence of tokens, enabling a single encoder-decoder Transformer to tackle multiple cross-modal translation as one coherent sequence-to-sequence task. Experimental results confirm that our unified multitask model improves upon single-task baselines in several key areas, notably reducing the symbol error rate for optical music recognition from 24.58% to a state-of-the-art 13.67%, while similarly substantial improvements are observed across the other translation tasks. Notably, our approach achieves the first successful score-image-conditioned audio generation, marking a significant breakthrough in cross-modal music generation. 

---
# UniHM: Universal Human Motion Generation with Object Interactions in Indoor Scenes 

**Authors**: Zichen Geng, Zeeshan Hayder, Wei Liu, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2505.12774)  

**Abstract**: Human motion synthesis in complex scenes presents a fundamental challenge, extending beyond conventional Text-to-Motion tasks by requiring the integration of diverse modalities such as static environments, movable objects, natural language prompts, and spatial waypoints. Existing language-conditioned motion models often struggle with scene-aware motion generation due to limitations in motion tokenization, which leads to information loss and fails to capture the continuous, context-dependent nature of 3D human movement. To address these issues, we propose UniHM, a unified motion language model that leverages diffusion-based generation for synthesizing scene-aware human motion. UniHM is the first framework to support both Text-to-Motion and Text-to-Human-Object Interaction (HOI) in complex 3D scenes. Our approach introduces three key contributions: (1) a mixed-motion representation that fuses continuous 6DoF motion with discrete local motion tokens to improve motion realism; (2) a novel Look-Up-Free Quantization VAE (LFQ-VAE) that surpasses traditional VQ-VAEs in both reconstruction accuracy and generative performance; and (3) an enriched version of the Lingo dataset augmented with HumanML3D annotations, providing stronger supervision for scene-specific motion learning. Experimental results demonstrate that UniHM achieves comparative performance on the OMOMO benchmark for text-to-HOI synthesis and yields competitive results on HumanML3D for general text-conditioned motion generation. 

---
# SounDiT: Geo-Contextual Soundscape-to-Landscape Generation 

**Authors**: Junbo Wang, Haofeng Tan, Bowen Liao, Albert Jiang, Teng Fei, Qixing Huang, Zhengzhong Tu, Shan Ye, Yuhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12734)  

**Abstract**: We present a novel and practically significant problem-Geo-Contextual Soundscape-to-Landscape (GeoS2L) generation-which aims to synthesize geographically realistic landscape images from environmental soundscapes. Prior audio-to-image generation methods typically rely on general-purpose datasets and overlook geographic and environmental contexts, resulting in unrealistic images that are misaligned with real-world environmental settings. To address this limitation, we introduce a novel geo-contextual computational framework that explicitly integrates geographic knowledge into multimodal generative modeling. We construct two large-scale geo-contextual multimodal datasets, SoundingSVI and SonicUrban, pairing diverse soundscapes with real-world landscape images. We propose SounDiT, a novel Diffusion Transformer (DiT)-based model that incorporates geo-contextual scene conditioning to synthesize geographically coherent landscape images. Furthermore, we propose a practically-informed geo-contextual evaluation framework, the Place Similarity Score (PSS), across element-, scene-, and human perception-levels to measure consistency between input soundscapes and generated landscape images. Extensive experiments demonstrate that SounDiT outperforms existing baselines in both visual fidelity and geographic settings. Our work not only establishes foundational benchmarks for GeoS2L generation but also highlights the importance of incorporating geographic domain knowledge in advancing multimodal generative models, opening new directions at the intersection of generative AI, geography, urban planning, and environmental sciences. 

---
# PLAICraft: Large-Scale Time-Aligned Vision-Speech-Action Dataset for Embodied AI 

**Authors**: Yingchen He, Christian D. Weilbach, Martyna E. Wojciechowska, Yuxuan Zhang, Frank Wood  

**Link**: [PDF](https://arxiv.org/pdf/2505.12707)  

**Abstract**: Advances in deep generative modelling have made it increasingly plausible to train human-level embodied agents. Yet progress has been limited by the absence of large-scale, real-time, multi-modal, and socially interactive datasets that reflect the sensory-motor complexity of natural environments. To address this, we present PLAICraft, a novel data collection platform and dataset capturing multiplayer Minecraft interactions across five time-aligned modalities: video, game output audio, microphone input audio, mouse, and keyboard actions. Each modality is logged with millisecond time precision, enabling the study of synchronous, embodied behaviour in a rich, open-ended world. The dataset comprises over 10,000 hours of gameplay from more than 10,000 global participants.\footnote{We have done a privacy review for the public release of an initial 200-hour subset of the dataset, with plans to release most of the dataset over time.} Alongside the dataset, we provide an evaluation suite for benchmarking model capabilities in object recognition, spatial awareness, language grounding, and long-term memory. PLAICraft opens a path toward training and evaluating agents that act fluently and purposefully in real time, paving the way for truly embodied artificial intelligence. 

---
# Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals 

**Authors**: Yuxin Lin, Yinglin Zheng, Ming Zeng, Wangzheng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12654)  

**Abstract**: This paper addresses the gap in predicting turn-taking and backchannel actions in human-machine conversations using multi-modal signals (linguistic, acoustic, and visual). To overcome the limitation of existing datasets, we propose an automatic data collection pipeline that allows us to collect and annotate over 210 hours of human conversation videos. From this, we construct a Multi-Modal Face-to-Face (MM-F2F) human conversation dataset, including over 1.5M words and corresponding turn-taking and backchannel annotations from approximately 20M frames. Additionally, we present an end-to-end framework that predicts the probability of turn-taking and backchannel actions from multi-modal signals. The proposed model emphasizes the interrelation between modalities and supports any combination of text, audio, and video inputs, making it adaptable to a variety of realistic scenarios. Our experiments show that our approach achieves state-of-the-art performance on turn-taking and backchannel prediction tasks, achieving a 10\% increase in F1-score on turn-taking and a 33\% increase on backchannel prediction. Our dataset and code are publicly available online to ease of subsequent research. 

---
# Any-to-Any Learning in Computational Pathology via Triplet Multimodal Pretraining 

**Authors**: Qichen Sun, Zhengrui Guo, Rui Peng, Hao Chen, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12711)  

**Abstract**: Recent advances in computational pathology and artificial intelligence have significantly enhanced the utilization of gigapixel whole-slide images and and additional modalities (e.g., genomics) for pathological diagnosis. Although deep learning has demonstrated strong potential in pathology, several key challenges persist: (1) fusing heterogeneous data types requires sophisticated strategies beyond simple concatenation due to high computational costs; (2) common scenarios of missing modalities necessitate flexible strategies that allow the model to learn robustly in the absence of certain modalities; (3) the downstream tasks in CPath are diverse, ranging from unimodal to multimodal, cnecessitating a unified model capable of handling all modalities. To address these challenges, we propose ALTER, an any-to-any tri-modal pretraining framework that integrates WSIs, genomics, and pathology reports. The term "any" emphasizes ALTER's modality-adaptive design, enabling flexible pretraining with any subset of modalities, and its capacity to learn robust, cross-modal representations beyond WSI-centric approaches. We evaluate ALTER across extensive clinical tasks including survival prediction, cancer subtyping, gene mutation prediction, and report generation, achieving superior or comparable performance to state-of-the-art baselines. 

---
# AutoMat: Enabling Automated Crystal Structure Reconstruction from Microscopy via Agentic Tool Use 

**Authors**: Yaotian Yang, Yiwen Tang, Yizhe Chen, Xiao Chen, Jiangjie Qiu, Hao Xiong, Haoyu Yin, Zhiyao Luo, Yifei Zhang, Sijia Tao, Wentao Li, Qinghua Zhang, Yuqiang Li, Wanli Ouyang, Bin Zhao, Xiaonan Wang, Fei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.12650)  

**Abstract**: Machine learning-based interatomic potentials and force fields depend critically on accurate atomic structures, yet such data are scarce due to the limited availability of experimentally resolved crystals. Although atomic-resolution electron microscopy offers a potential source of structural data, converting these images into simulation-ready formats remains labor-intensive and error-prone, creating a bottleneck for model training and validation. We introduce AutoMat, an end-to-end, agent-assisted pipeline that automatically transforms scanning transmission electron microscopy (STEM) images into atomic crystal structures and predicts their physical properties. AutoMat combines pattern-adaptive denoising, physics-guided template retrieval, symmetry-aware atomic reconstruction, fast relaxation and property prediction via MatterSim, and coordinated orchestration across all stages. We propose the first dedicated STEM2Mat-Bench for this task and evaluate performance using lattice RMSD, formation energy MAE, and structure-matching success rate. By orchestrating external tool calls, AutoMat enables a text-only LLM to outperform vision-language models in this domain, achieving closed-loop reasoning throughout the pipeline. In large-scale experiments over 450 structure samples, AutoMat substantially outperforms existing multimodal large language models and tools. These results validate both AutoMat and STEM2Mat-Bench, marking a key step toward bridging microscopy and atomistic simulation in materials this http URL code and dataset are publicly available at this https URL and this https URL. 

---
# Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents 

**Authors**: Yunseok Jang, Yeda Song, Sungryull Sohn, Lajanugen Logeswaran, Tiange Luo, Dong-Ki Kim, Kyunghoon Bae, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12632)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have sparked significant interest in developing GUI visual agents. We introduce MONDAY (Mobile OS Navigation Task Dataset for Agents from YouTube), a large-scale dataset of 313K annotated frames from 20K instructional videos capturing diverse real-world mobile OS navigation across multiple platforms. Models that include MONDAY in their pre-training phases demonstrate robust cross-platform generalization capabilities, consistently outperforming models trained on existing single OS datasets while achieving an average performance gain of 18.11%p on an unseen mobile OS platform. To enable continuous dataset expansion as mobile platforms evolve, we present an automated framework that leverages publicly available video content to create comprehensive task datasets without manual annotation. Our framework comprises robust OCR-based scene detection (95.04% F1score), near-perfect UI element detection (99.87% hit ratio), and novel multi-step action identification to extract reliable action sequences across diverse interface configurations. We contribute both the MONDAY dataset and our automated collection framework to facilitate future research in mobile OS navigation. 

---
# Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning 

**Authors**: Zirun Guo, Minjie Hong, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12432)  

**Abstract**: Reinforcement Learning (RL) has shown promise in improving the reasoning abilities of Large Language Models (LLMs). However, the specific challenges of adapting RL to multimodal data and formats remain relatively unexplored. In this work, we present Observe-R1, a novel framework aimed at enhancing the reasoning capabilities of multimodal large language models (MLLMs). We draw inspirations from human learning progression--from simple to complex and easy to difficult, and propose a gradual learning paradigm for MLLMs. To this end, we construct the NeuraLadder dataset, which is organized and sampled according to the difficulty and complexity of data samples for RL training. To tackle multimodal tasks, we introduce a multimodal format constraint that encourages careful observation of images, resulting in enhanced visual abilities and clearer and more structured responses. Additionally, we implement a bonus reward system that favors concise, correct answers within a length constraint, alongside a dynamic weighting mechanism that prioritizes uncertain and medium-difficulty problems, ensuring that more informative samples have a greater impact on training. Our experiments with the Qwen2.5-VL-3B and Qwen2.5-VL-7B models on 20k samples from the NeuraLadder dataset show that Observe-R1 outperforms a series of larger reasoning models on both reasoning and general benchmarks, achieving superior clarity and conciseness in reasoning chains. Ablation studies validate the effectiveness of our strategies, highlighting the robustness and generalization of our approach. The dataset and code will be released at this https URL. 

---
# Video-GPT via Next Clip Diffusion 

**Authors**: Shaobin Zhuang, Zhipeng Huang, Ying Zhang, Fangyikang Wang, Canmiao Fu, Binxin Yang, Chong Sun, Chen Li, Yali Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12489)  

**Abstract**: GPT has shown its remarkable success in natural language processing. However, the language sequence is not sufficient to describe spatial-temporal details in the visual world. Alternatively, the video sequence is good at capturing such details. Motivated by this fact, we propose a concise Video-GPT in this paper by treating video as new language for visual world modeling. By analogy to next token prediction in GPT, we introduce a novel next clip diffusion paradigm for pretraining Video-GPT. Different from the previous works, this distinct paradigm allows Video-GPT to tackle both short-term generation and long-term prediction, by autoregressively denoising the noisy clip according to the clean clips in the history. Extensive experiments show our Video-GPT achieves the state-of-the-art performance on video prediction, which is the key factor towards world modeling (Physics-IQ Benchmark: Video-GPT 34.97 vs. Kling 23.64 vs. Wan 20.89). Moreover, it can be well adapted on 6 mainstream video tasks in both video generation and understanding, showing its great generalization capacity in downstream. The project page is at this https URL. 

---
# Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts 

**Authors**: Qi Feng, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12363)  

**Abstract**: While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research. 

---
# Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models 

**Authors**: Kai Tang, Jinhao You, Xiuqi Ge, Hanze Li, Yichen Guo, Xiande Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12343)  

**Abstract**: Despite the impressive capabilities of Large Vision-Language Models (LVLMs), they remain susceptible to hallucinations-generating content that is inconsistent with the input image. Existing training-free hallucination mitigation methods often suffer from unstable performance and high sensitivity to hyperparameter settings, limiting their practicality and broader adoption. In this paper, we propose a novel decoding mechanism, Decoding with Inter-layer Consistency via Layer Aggregation (DCLA), which requires no retraining, fine-tuning, or access to external knowledge bases. Specifically, our approach constructs a dynamic semantic reference by aggregating representations from previous layers, and corrects semantically deviated layers to enforce inter-layer consistency. The method allows DCLA to robustly mitigate hallucinations across multiple LVLMs. Experiments on hallucination benchmarks such as MME and POPE demonstrate that DCLA effectively reduces hallucinations while enhancing the reliability and performance of LVLMs. 

---
# Visuospatial Cognitive Assistant 

**Authors**: Qi Feng, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12312)  

**Abstract**: Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence. 

---
# Can Large Multimodal Models Understand Agricultural Scenes? Benchmarking with AgroMind 

**Authors**: Qingmei Li, Yang Zhang, Zurong Mai, Yuhang Chen, Shuohong Lou, Henglian Huang, Jiarui Zhang, Zhiwei Zhang, Yibin Wen, Weijia Li, Haohuan Fu, Jianxi Huang, Juepeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12207)  

**Abstract**: Large Multimodal Models (LMMs) has demonstrated capabilities across various domains, but comprehensive benchmarks for agricultural remote sensing (RS) remain scarce. Existing benchmarks designed for agricultural RS scenarios exhibit notable limitations, primarily in terms of insufficient scene diversity in the dataset and oversimplified task design. To bridge this gap, we introduce AgroMind, a comprehensive agricultural remote sensing benchmark covering four task dimensions: spatial perception, object understanding, scene understanding, and scene reasoning, with a total of 13 task types, ranging from crop identification and health monitoring to environmental analysis. We curate a high-quality evaluation set by integrating eight public datasets and one private farmland plot dataset, containing 25,026 QA pairs and 15,556 images. The pipeline begins with multi-source data preprocessing, including collection, format standardization, and annotation refinement. We then generate a diverse set of agriculturally relevant questions through the systematic definition of tasks. Finally, we employ LMMs for inference, generating responses, and performing detailed examinations. We evaluated 18 open-source LMMs and 3 closed-source models on AgroMind. Experiments reveal significant performance gaps, particularly in spatial reasoning and fine-grained recognition, it is notable that human performance lags behind several leading LMMs. By establishing a standardized evaluation framework for agricultural RS, AgroMind reveals the limitations of LMMs in domain knowledge and highlights critical challenges for future work. Data and code can be accessed at this https URL. 

---
# MMS-VPR: Multimodal Street-Level Visual Place Recognition Dataset and Benchmark 

**Authors**: Yiwei Ou, Xiaobin Ren, Ronggui Sun, Guansong Gao, Ziyi Jiang, Kaiqi Zhao, Manfredo Manfredini  

**Link**: [PDF](https://arxiv.org/pdf/2505.12254)  

**Abstract**: Existing visual place recognition (VPR) datasets predominantly rely on vehicle-mounted imagery, lack multimodal diversity and underrepresent dense, mixed-use street-level spaces, especially in non-Western urban contexts. To address these gaps, we introduce MMS-VPR, a large-scale multimodal dataset for street-level place recognition in complex, pedestrian-only environments. The dataset comprises 78,575 annotated images and 2,512 video clips captured across 207 locations in a ~70,800 $\mathrm{m}^2$ open-air commercial district in Chengdu, China. Each image is labeled with precise GPS coordinates, timestamp, and textual metadata, and covers varied lighting conditions, viewpoints, and timeframes. MMS-VPR follows a systematic and replicable data collection protocol with minimal device requirements, lowering the barrier for scalable dataset creation. Importantly, the dataset forms an inherent spatial graph with 125 edges, 81 nodes, and 1 subgraph, enabling structure-aware place recognition. We further define two application-specific subsets -- Dataset_Edges and Dataset_Points -- to support fine-grained and graph-based evaluation tasks. Extensive benchmarks using conventional VPR models, graph neural networks, and multimodal baselines show substantial improvements when leveraging multimodal and structural cues. MMS-VPR facilitates future research at the intersection of computer vision, geospatial understanding, and multimodal reasoning. The dataset is publicly available at this https URL. 

---
# RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction 

**Authors**: Weifeng Lu, Minghao Ye, Zewei Ye, Ruihan Tao, Shuo Yang, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12224)  

**Abstract**: Vision-Language-Action (VLA) models have recently advanced robotic manipulation by translating natural-language instructions and image information into sequential control actions. However, these models often underperform in open-world scenarios, as they are predominantly trained on successful expert demonstrations and exhibit a limited capacity for failure recovery. In this work, we present a Robotic Failure Analysis and Correction (RoboFAC) framework to address this issue. Firstly, we construct RoboFAC dataset comprising 9,440 erroneous manipulation trajectories and 78,623 QA pairs across 16 diverse tasks and 53 scenes in both simulation and real-world environments. Leveraging our dataset, we develop RoboFAC model, which is capable of Task Understanding, Failure Analysis and Failure Correction. Experimental results demonstrate that the RoboFAC model outperforms GPT-4o by 34.1% on our evaluation benchmark. Furthermore, we integrate the RoboFAC model into a real-world VLA control pipeline as an external supervision providing correction instructions, yielding a 29.1% relative improvement on average on four real-world tasks. The results show that our RoboFAC framework effectively handles robotic failures and assists the VLA model in recovering from failures. 

---
# Enhanced Multimodal Hate Video Detection via Channel-wise and Modality-wise Fusion 

**Authors**: Yinghui Zhang, Tailin Chen, Yuchen Zhang, Zeyu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12051)  

**Abstract**: The rapid rise of video content on platforms such as TikTok and YouTube has transformed information dissemination, but it has also facilitated the spread of harmful content, particularly hate videos. Despite significant efforts to combat hate speech, detecting these videos remains challenging due to their often implicit nature. Current detection methods primarily rely on unimodal approaches, which inadequately capture the complementary features across different modalities. While multimodal techniques offer a broader perspective, many fail to effectively integrate temporal dynamics and modality-wise interactions essential for identifying nuanced hate content. In this paper, we present CMFusion, an enhanced multimodal hate video detection model utilizing a novel Channel-wise and Modality-wise Fusion Mechanism. CMFusion first extracts features from text, audio, and video modalities using pre-trained models and then incorporates a temporal cross-attention mechanism to capture dependencies between video and audio streams. The learned features are then processed by channel-wise and modality-wise fusion modules to obtain informative representations of videos. Our extensive experiments on a real-world dataset demonstrate that CMFusion significantly outperforms five widely used baselines in terms of accuracy, precision, recall, and F1 score. Comprehensive ablation studies and parameter analyses further validate our design choices, highlighting the model's effectiveness in detecting hate videos. The source codes will be made publicly available at this https URL. 

---
# SafeVid: Toward Safety Aligned Video Large Multimodal Models 

**Authors**: Yixu Wang, Jiaxin Song, Yifeng Gao, Xin Wang, Yang Yao, Yan Teng, Xingjun Ma, Yingchun Wang, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11926)  

**Abstract**: As Video Large Multimodal Models (VLMMs) rapidly advance, their inherent complexity introduces significant safety challenges, particularly the issue of mismatched generalization where static safety alignments fail to transfer to dynamic video contexts. We introduce SafeVid, a framework designed to instill video-specific safety principles in VLMMs. SafeVid uniquely transfers robust textual safety alignment capabilities to the video domain by employing detailed textual video descriptions as an interpretive bridge, facilitating LLM-based rule-driven safety reasoning. This is achieved through a closed-loop system comprising: 1) generation of SafeVid-350K, a novel 350,000-pair video-specific safety preference dataset; 2) targeted alignment of VLMMs using Direct Preference Optimization (DPO); and 3) comprehensive evaluation via our new SafeVidBench benchmark. Alignment with SafeVid-350K significantly enhances VLMM safety, with models like LLaVA-NeXT-Video demonstrating substantial improvements (e.g., up to 42.39%) on SafeVidBench. SafeVid provides critical resources and a structured approach, demonstrating that leveraging textual descriptions as a conduit for safety reasoning markedly improves the safety alignment of VLMMs. We have made SafeVid-350K dataset (this https URL) publicly available. 

---
# AdaptMol: Adaptive Fusion from Sequence String to Topological Structure for Few-shot Drug Discovery 

**Authors**: Yifan Dai, Xuanbai Ren, Tengfei Ma, Qipeng Yan, Yiping Liu, Yuansheng Liu, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11878)  

**Abstract**: Accurate molecular property prediction (MPP) is a critical step in modern drug development. However, the scarcity of experimental validation data poses a significant challenge to AI-driven research paradigms. Under few-shot learning scenarios, the quality of molecular representations directly dictates the theoretical upper limit of model performance. We present AdaptMol, a prototypical network integrating Adaptive multimodal fusion for Molecular representation. This framework employs a dual-level attention mechanism to dynamically integrate global and local molecular features derived from two modalities: SMILES sequences and molecular graphs. (1) At the local level, structural features such as atomic interactions and substructures are extracted from molecular graphs, emphasizing fine-grained topological information; (2) At the global level, the SMILES sequence provides a holistic representation of the molecule. To validate the necessity of multimodal adaptive fusion, we propose an interpretable approach based on identifying molecular active substructures to demonstrate that multimodal adaptive fusion can efficiently represent molecules. Extensive experiments on three commonly used benchmarks under 5-shot and 10-shot settings demonstrate that AdaptMol achieves state-of-the-art performance in most cases. The rationale-extracted method guides the fusion of two modalities and highlights the importance of both modalities. 

---
# Are vision language models robust to uncertain inputs? 

**Authors**: Xi Wang, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2505.11804)  

**Abstract**: Robustness against uncertain and ambiguous inputs is a critical challenge for deep learning models. While recent advancements in large scale vision language models (VLMs, e.g. GPT4o) might suggest that increasing model and training dataset size would mitigate this issue, our empirical evaluation shows a more complicated picture. Testing models using two classic uncertainty quantification tasks, anomaly detection and classification under inherently ambiguous conditions, we find that newer and larger VLMs indeed exhibit improved robustness compared to earlier models, but still suffer from a tendency to strictly follow instructions, often causing them to hallucinate confident responses even when faced with unclear or anomalous inputs. Remarkably, for natural images such as ImageNet, this limitation can be overcome without pipeline modifications: simply prompting models to abstain from uncertain predictions enables significant reliability gains, achieving near-perfect robustness in several settings. However, for domain-specific tasks such as galaxy morphology classification, a lack of specialized knowledge prevents reliable uncertainty estimation. Finally, we propose a novel mechanism based on caption diversity to reveal a model's internal uncertainty, enabling practitioners to predict when models will successfully abstain without relying on labeled data. 

---
# BioCube: A Multimodal Dataset for Biodiversity Research 

**Authors**: Stylianos Stasinos, Martino Mensio, Elena Lazovik, Athanasios Trantas  

**Link**: [PDF](https://arxiv.org/pdf/2505.11568)  

**Abstract**: Biodiversity research requires complete and detailed information to study ecosystem dynamics at different scales. Employing data-driven methods like Machine Learning is getting traction in ecology and more specific biodiversity, offering alternative modelling pathways. For these methods to deliver accurate results there is the need for large, curated and multimodal datasets that offer granular spatial and temporal resolutions. In this work, we introduce BioCube, a multimodal, fine-grained global dataset for ecology and biodiversity research. BioCube incorporates species observations through images, audio recordings and descriptions, environmental DNA, vegetation indices, agricultural, forest, land indicators, and high-resolution climate variables. All observations are geospatially aligned under the WGS84 geodetic system, spanning from 2000 to 2020. The dataset will become available at this https URL while the acquisition and processing code base at this https URL. 

---
# ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models 

**Authors**: Liyan Tang, Grace Kim, Xinyu Zhao, Thom Lake, Wenxuan Ding, Fangcong Yin, Prasann Singhal, Manya Wadhwa, Zeyu Leo Liu, Zayne Sprague, Ramya Namuduri, Bodun Hu, Juan Diego Rodriguez, Puyuan Peng, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2505.13444)  

**Abstract**: Chart understanding presents a unique challenge for large vision-language models (LVLMs), as it requires the integration of sophisticated textual and visual reasoning capabilities. However, current LVLMs exhibit a notable imbalance between these skills, falling short on visual reasoning that is difficult to perform in text. We conduct a case study using a synthetic dataset solvable only through visual reasoning and show that model performance degrades significantly with increasing visual complexity, while human performance remains robust. We then introduce ChartMuseum, a new Chart Question Answering (QA) benchmark containing 1,162 expert-annotated questions spanning multiple reasoning types, curated from real-world charts across 184 sources, specifically built to evaluate complex visual and textual reasoning. Unlike prior chart understanding benchmarks -- where frontier models perform similarly and near saturation -- our benchmark exposes a substantial gap between model and human performance, while effectively differentiating model capabilities: although humans achieve 93% accuracy, the best-performing model Gemini-2.5-Pro attains only 63.0%, and the leading open-source LVLM Qwen2.5-VL-72B-Instruct achieves only 38.5%. Moreover, on questions requiring primarily visual reasoning, all models experience a 35%-55% performance drop from text-reasoning-heavy question performance. Lastly, our qualitative error analysis reveals specific categories of visual reasoning that are challenging for current LVLMs. 

---
# I'll believe it when I see it: Images increase misinformation sharing in Vision-Language Models 

**Authors**: Alice Plebe, Timothy Douglas, Diana Riazi, R. Maria del Rio-Chanona  

**Link**: [PDF](https://arxiv.org/pdf/2505.13302)  

**Abstract**: Large language models are increasingly integrated into news recommendation systems, raising concerns about their role in spreading misinformation. In humans, visual content is known to boost credibility and shareability of information, yet its effect on vision-language models (VLMs) remains unclear. We present the first study examining how images influence VLMs' propensity to reshare news content, whether this effect varies across model families, and how persona conditioning and content attributes modulate this behavior. To support this analysis, we introduce two methodological contributions: a jailbreaking-inspired prompting strategy that elicits resharing decisions from VLMs while simulating users with antisocial traits and political alignments; and a multimodal dataset of fact-checked political news from PolitiFact, paired with corresponding images and ground-truth veracity labels. Experiments across model families reveal that image presence increases resharing rates by 4.8% for true news and 15.0% for false news. Persona conditioning further modulates this effect: Dark Triad traits amplify resharing of false news, whereas Republican-aligned profiles exhibit reduced veracity sensitivity. Of all the tested models, only Claude-3-Haiku demonstrates robustness to visual misinformation. These findings highlight emerging risks in multimodal model behavior and motivate the development of tailored evaluation frameworks and mitigation strategies for personalized AI systems. Code and dataset are available at: this https URL 

---
# MR. Judge: Multimodal Reasoner as a Judge 

**Authors**: Renjie Pi, Felix Bai, Qibin Chen, Simon Wang, Jiulong Shan, Kieran Liu, Meng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13403)  

**Abstract**: The paradigm of using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) as evaluative judges has emerged as an effective approach in RLHF and inference-time scaling. In this work, we propose Multimodal Reasoner as a Judge (MR. Judge), a paradigm for empowering general-purpose MLLMs judges with strong reasoning capabilities. Instead of directly assigning scores for each response, we formulate the judgement process as a reasoning-inspired multiple-choice problem. Specifically, the judge model first conducts deliberate reasoning covering different aspects of the responses and eventually selects the best response from them. This reasoning process not only improves the interpretibility of the judgement, but also greatly enhances the performance of MLLM judges. To cope with the lack of questions with scored responses, we propose the following strategy to achieve automatic annotation: 1) Reverse Response Candidates Synthesis: starting from a supervised fine-tuning (SFT) dataset, we treat the original response as the best candidate and prompt the MLLM to generate plausible but flawed negative candidates. 2) Text-based reasoning extraction: we carefully design a data synthesis pipeline for distilling the reasoning capability from a text-based reasoning model, which is adopted to enable the MLLM judges to regain complex reasoning ability via warm up supervised fine-tuning. Experiments demonstrate that our MR. Judge is effective across a wide range of tasks. Specifically, our MR. Judge-7B surpasses GPT-4o by 9.9% on VL-RewardBench, and improves performance on MM-Vet during inference-time scaling by up to 7.7%. 

---
# Contextual Paralinguistic Data Creation for Multi-Modal Speech-LLM: Data Condensation and Spoken QA Generation 

**Authors**: Qiongqiong Wang, Hardik B. Sailor, Tianchi Liu, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2505.13338)  

**Abstract**: Current speech-LLMs exhibit limited capability in contextual reasoning alongside paralinguistic understanding, primarily due to the lack of Question-Answer (QA) datasets that cover both aspects. We propose a novel framework for dataset generation from in-the-wild speech data, that integrates contextual reasoning with paralinguistic information. It consists of a pseudo paralinguistic label-based data condensation of in-the-wild speech and LLM-based Contextual Paralinguistic QA (CPQA) generation. The effectiveness is validated by a strong correlation in evaluations of the Qwen2-Audio-7B-Instruct model on a dataset created by our framework and human-generated CPQA dataset. The results also reveal the speech-LLM's limitations in handling empathetic reasoning tasks, highlighting the need for such datasets and more robust models. The proposed framework is first of its kind and has potential in training more robust speech-LLMs with paralinguistic reasoning capabilities. 

---
# Suicide Risk Assessment Using Multimodal Speech Features: A Study on the SW1 Challenge Dataset 

**Authors**: Ambre Marie, Ilias Maoudj, Guillaume Dardenne, Gwenolé Quellec  

**Link**: [PDF](https://arxiv.org/pdf/2505.13069)  

**Abstract**: The 1st SpeechWellness Challenge conveys the need for speech-based suicide risk assessment in adolescents. This study investigates a multimodal approach for this challenge, integrating automatic transcription with WhisperX, linguistic embeddings from Chinese RoBERTa, and audio embeddings from WavLM. Additionally, handcrafted acoustic features -- including MFCCs, spectral contrast, and pitch-related statistics -- were incorporated. We explored three fusion strategies: early concatenation, modality-specific processing, and weighted attention with mixup regularization. Results show that weighted attention provided the best generalization, achieving 69% accuracy on the development set, though a performance gap between development and test sets highlights generalization challenges. Our findings, strictly tied to the MINI-KID framework, emphasize the importance of refining embedding representations and fusion mechanisms to enhance classification reliability. 

---
# FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models 

**Authors**: Hengxing Cai, Jinhan Dong, Jingjun Tan, Jingcheng Deng, Sihang Li, Zhifeng Gao, Haidong Wang, Zicheng Su, Agachai Sumalee, Renxin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12835)  

**Abstract**: Unmanned Aerial Vehicle (UAV) Vision-and-Language Navigation (VLN) is vital for applications such as disaster response, logistics delivery, and urban inspection. However, existing methods often struggle with insufficient multimodal fusion, weak generalization, and poor interpretability. To address these challenges, we propose FlightGPT, a novel UAV VLN framework built upon Vision-Language Models (VLMs) with powerful multimodal perception capabilities. We design a two-stage training pipeline: first, Supervised Fine-Tuning (SFT) using high-quality demonstrations to improve initialization and structured reasoning; then, Group Relative Policy Optimization (GRPO) algorithm, guided by a composite reward that considers goal accuracy, reasoning quality, and format compliance, to enhance generalization and adaptability. Furthermore, FlightGPT introduces a Chain-of-Thought (CoT)-based reasoning mechanism to improve decision interpretability. Extensive experiments on the city-scale dataset CityNav demonstrate that FlightGPT achieves state-of-the-art performance across all scenarios, with a 9.22\% higher success rate than the strongest baseline in unseen environments. Our implementation is publicly available. 

---
# Towards Reliable and Interpretable Traffic Crash Pattern Prediction and Safety Interventions Using Customized Large Language Models 

**Authors**: Yang Zhao, Pu Wang, Yibo Zhao, Hongru Du, Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12545)  

**Abstract**: Predicting crash events is crucial for understanding crash distributions and their contributing factors, thereby enabling the design of proactive traffic safety policy interventions. However, existing methods struggle to interpret the complex interplay among various sources of traffic crash data, including numeric characteristics, textual reports, crash imagery, environmental conditions, and driver behavior records. As a result, they often fail to capture the rich semantic information and intricate interrelationships embedded in these diverse data sources, limiting their ability to identify critical crash risk factors. In this research, we propose TrafficSafe, a framework that adapts LLMs to reframe crash prediction and feature attribution as text-based reasoning. A multi-modal crash dataset including 58,903 real-world reports together with belonged infrastructure, environmental, driver, and vehicle information is collected and textualized into TrafficSafe Event Dataset. By customizing and fine-tuning LLMs on this dataset, the TrafficSafe LLM achieves a 42% average improvement in F1-score over baselines. To interpret these predictions and uncover contributing factors, we introduce TrafficSafe Attribution, a sentence-level feature attribution framework enabling conditional risk analysis. Findings show that alcohol-impaired driving is the leading factor in severe crashes, with aggressive and impairment-related behaviors having nearly twice the contribution for severe crashes compared to other driver behaviors. Furthermore, TrafficSafe Attribution highlights pivotal features during model training, guiding strategic crash data collection for iterative performance improvements. The proposed TrafficSafe offers a transformative leap in traffic safety research, providing a blueprint for translating advanced AI technologies into responsible, actionable, and life-saving outcomes. 

---
# Disambiguating Reference in Visually Grounded Dialogues through Joint Modeling of Textual and Multimodal Semantic Structures 

**Authors**: Shun Inadumi, Nobuhiro Ueda, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2505.11726)  

**Abstract**: Multimodal reference resolution, including phrase grounding, aims to understand the semantic relations between mentions and real-world objects. Phrase grounding between images and their captions is a well-established task. In contrast, for real-world applications, it is essential to integrate textual and multimodal reference resolution to unravel the reference relations within dialogue, especially in handling ambiguities caused by pronouns and ellipses. This paper presents a framework that unifies textual and multimodal reference resolution by mapping mention embeddings to object embeddings and selecting mentions or objects based on their similarity. Our experiments show that learning textual reference resolution, such as coreference resolution and predicate-argument structure analysis, positively affects performance in multimodal reference resolution. In particular, our model with coreference resolution performs better in pronoun phrase grounding than representative models for this task, MDETR and GLIP. Our qualitative analysis demonstrates that incorporating textual reference relations strengthens the confidence scores between mentions, including pronouns and predicates, and objects, which can reduce the ambiguities that arise in visually grounded dialogues. 

---
# MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix 

**Authors**: Ziyang Ma, Yinghao Ma, Yanqiao Zhu, Chen Yang, Yi-Wen Chao, Ruiyang Xu, Wenxi Chen, Yuanzhe Chen, Zhuo Chen, Jian Cong, Kai Li, Keliang Li, Siyou Li, Xinfeng Li, Xiquan Li, Zheng Lian, Yuzhe Liang, Minghao Liu, Zhikang Niu, Tianrui Wang, Yuping Wang, Yuxuan Wang, Yihao Wu, Guanrou Yang, Jianwei Yu, Ruibin Yuan, Zhisheng Zheng, Ziya Zhou, Haina Zhu, Wei Xue, Emmanouil Benetos, Kai Yu, Eng-Siong Chng, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13032)  

**Abstract**: We introduce MMAR, a new benchmark designed to evaluate the deep reasoning capabilities of Audio-Language Models (ALMs) across massive multi-disciplinary tasks. MMAR comprises 1,000 meticulously curated audio-question-answer triplets, collected from real-world internet videos and refined through iterative error corrections and quality checks to ensure high quality. Unlike existing benchmarks that are limited to specific domains of sound, music, or speech, MMAR extends them to a broad spectrum of real-world audio scenarios, including mixed-modality combinations of sound, music, and speech. Each question in MMAR is hierarchically categorized across four reasoning layers: Signal, Perception, Semantic, and Cultural, with additional sub-categories within each layer to reflect task diversity and complexity. To further foster research in this area, we annotate every question with a Chain-of-Thought (CoT) rationale to promote future advancements in audio reasoning. Each item in the benchmark demands multi-step deep reasoning beyond surface-level understanding. Moreover, a part of the questions requires graduate-level perceptual and domain-specific knowledge, elevating the benchmark's difficulty and depth. We evaluate MMAR using a broad set of models, including Large Audio-Language Models (LALMs), Large Audio Reasoning Models (LARMs), Omni Language Models (OLMs), Large Language Models (LLMs), and Large Reasoning Models (LRMs), with audio caption inputs. The performance of these models on MMAR highlights the benchmark's challenging nature, and our analysis further reveals critical limitations of understanding and reasoning capabilities among current models. We hope MMAR will serve as a catalyst for future advances in this important but little-explored area. 

---
# LogicOCR: Do Your Large Multimodal Models Excel at Logical Reasoning on Text-Rich Images? 

**Authors**: Maoyuan Ye, Jing Zhang, Juhua Liu, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12307)  

**Abstract**: Recent advances in Large Multimodal Models (LMMs) have significantly improved their reasoning and Optical Character Recognition (OCR) capabilities. However, their performance on complex logical reasoning tasks involving text-rich images remains underexplored. To bridge this gap, we introduce LogicOCR, a benchmark comprising 1,100 multiple-choice questions designed to evaluate LMMs' logical reasoning abilities on text-rich images, while minimizing reliance on domain-specific knowledge (e.g., mathematics). We construct LogicOCR by curating a text corpus from the Chinese National Civil Servant Examination and develop a scalable, automated pipeline to convert it into multimodal samples. First, we design prompt templates to steer GPT-Image-1 to generate images with diverse backgrounds, interleaved text-illustration layouts, and varied fonts, ensuring contextual relevance and visual realism. Then, the generated images are manually verified, with low-quality examples discarded. We evaluate a range of representative open-source and proprietary LMMs under both Chain-of-Thought (CoT) and direct-answer settings. Our multi-dimensional analysis reveals key insights, such as the impact of test-time scaling, input modality differences, and sensitivity to visual-text orientation. Notably, LMMs still lag in multimodal reasoning compared to text-only inputs, indicating that they have not fully bridged visual reading with reasoning. We hope LogicOCR will serve as a valuable resource for advancing multimodal reasoning research. The dataset is available at this https URL. 

---
# Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs 

**Authors**: Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2505.11842)  

**Abstract**: The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies. 

---
