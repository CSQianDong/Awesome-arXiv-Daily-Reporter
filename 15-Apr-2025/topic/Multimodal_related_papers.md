# The Future of MLLM Prompting is Adaptive: A Comprehensive Experimental Evaluation of Prompt Engineering Methods for Robust Multimodal Performance 

**Authors**: Anwesha Mohanty, Venkatesh Balavadhani Parthasarathy, Arsalan Shahid  

**Link**: [PDF](https://arxiv.org/pdf/2504.10179)  

**Abstract**: Multimodal Large Language Models (MLLMs) are set to transform how machines process and generate human-like responses by integrating diverse modalities such as text, images, and code. Yet, effectively harnessing their capabilities hinges on optimal prompt engineering. We present a comprehensive experimental evaluation of seven prompt engineering methods applied to 13 open-source MLLMs over 24 tasks spanning Reasoning and Compositionality, Multimodal Understanding and Alignment, Complex Code Generation and Execution, and Knowledge Retrieval and Integration. Our approach stratifies models by parameter count into Small (<4B), Medium (4B-10B), and Large (>10B) categories and compares prompting techniques including Zero-Shot, One-Shot, Few-Shot, Chain-of-Thought, Analogical, Generated Knowledge, and Tree-of-Thought. While Large MLLMs excel in structured tasks such as code generation, achieving accuracies up to 96.88% under Few-Shot prompting, all models struggle with complex reasoning and abstract understanding, often yielding accuracies below 60% and high hallucination rates. Structured reasoning prompts frequently increased hallucination up to 75% in small models and led to longer response times (over 20 seconds in Large MLLMs), while simpler prompting methods provided more concise and efficient outputs. No single prompting method uniformly optimises all task types. Instead, adaptive strategies combining example-based guidance with selective structured reasoning are essential to enhance robustness, efficiency, and factual accuracy. Our findings offer practical recommendations for prompt engineering and support more reliable deployment of MLLMs across applications including AI-assisted coding, knowledge retrieval, and multimodal content understanding. 

---
# MMKB-RAG: A Multi-Modal Knowledge-Based Retrieval-Augmented Generation Framework 

**Authors**: Zihan Ling, Zhiyao Guo, Yixuan Huang, Yi An, Shuai Xiao, Jinsong Lan, Xiaoyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10074)  

**Abstract**: Recent advancements in large language models (LLMs) and multi-modal LLMs have been remarkable. However, these models still rely solely on their parametric knowledge, which limits their ability to generate up-to-date information and increases the risk of producing erroneous content. Retrieval-Augmented Generation (RAG) partially mitigates these challenges by incorporating external data sources, yet the reliance on databases and retrieval systems can introduce irrelevant or inaccurate documents, ultimately undermining both performance and reasoning quality. In this paper, we propose Multi-Modal Knowledge-Based Retrieval-Augmented Generation (MMKB-RAG), a novel multi-modal RAG framework that leverages the inherent knowledge boundaries of models to dynamically generate semantic tags for the retrieval process. This strategy enables the joint filtering of retrieved documents, retaining only the most relevant and accurate references. Extensive experiments on knowledge-based visual question-answering tasks demonstrate the efficacy of our approach: on the E-VQA dataset, our method improves performance by +4.2\% on the Single-Hop subset and +0.4\% on the full dataset, while on the InfoSeek dataset, it achieves gains of +7.8\% on the Unseen-Q subset, +8.2\% on the Unseen-E subset, and +8.1\% on the full dataset. These results highlight significant enhancements in both accuracy and robustness over the current state-of-the-art MLLM and RAG frameworks. 

---
# Breaking the Data Barrier -- Building GUI Agents Through Task Generalization 

**Authors**: Junlei Zhang, Zichen Ding, Chang Ma, Zijie Chen, Qiushi Sun, Zhenzhong Lan, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2504.10127)  

**Abstract**: Graphical User Interface (GUI) agents offer cross-platform solutions for automating complex digital tasks, with significant potential to transform productivity workflows. However, their performance is often constrained by the scarcity of high-quality trajectory data. To address this limitation, we propose training Vision Language Models (VLMs) on data-rich, reasoning-intensive tasks during a dedicated mid-training stage, and then examine how incorporating these tasks facilitates generalization to GUI planning scenarios. Specifically, we explore a range of tasks with readily available instruction-tuning data, including GUI perception, multimodal reasoning, and textual reasoning. Through extensive experiments across 11 mid-training tasks, we demonstrate that: (1) Task generalization proves highly effective, yielding substantial improvements across most settings. For instance, multimodal mathematical reasoning enhances performance on AndroidWorld by an absolute 6.3%. Remarkably, text-only mathematical data significantly boosts GUI web agent performance, achieving a 5.6% improvement on WebArena and 5.4% improvement on AndroidWorld, underscoring notable cross-modal generalization from text-based to visual domains; (2) Contrary to prior assumptions, GUI perception data - previously considered closely aligned with GUI agent tasks and widely utilized for training - has a comparatively limited impact on final performance; (3) Building on these insights, we identify the most effective mid-training tasks and curate optimized mixture datasets, resulting in absolute performance gains of 8.0% on WebArena and 12.2% on AndroidWorld. Our work provides valuable insights into cross-domain knowledge transfer for GUI agents and offers a practical approach to addressing data scarcity challenges in this emerging field. The code, data and models will be available at this https URL. 

---
# InfoMAE: Pair-Efficient Cross-Modal Alignment for Multimodal Time-Series Sensing Signals 

**Authors**: Tomoyoshi Kimura, Xinlin Li, Osama Hanna, Yatong Chen, Yizhuo Chen, Denizhan Kara, Tianshi Wang, Jinyang Li, Xiaomin Ouyang, Shengzhong Liu, Mani Srivastava, Suhas Diggavi, Tarek Abdelzaher  

**Link**: [PDF](https://arxiv.org/pdf/2504.09707)  

**Abstract**: Standard multimodal self-supervised learning (SSL) algorithms regard cross-modal synchronization as implicit supervisory labels during pretraining, thus posing high requirements on the scale and quality of multimodal samples. These constraints significantly limit the performance of sensing intelligence in IoT applications, as the heterogeneity and the non-interpretability of time-series signals result in abundant unimodal data but scarce high-quality multimodal pairs. This paper proposes InfoMAE, a cross-modal alignment framework that tackles the challenge of multimodal pair efficiency under the SSL setting by facilitating efficient cross-modal alignment of pretrained unimodal representations. InfoMAE achieves \textit{efficient cross-modal alignment} with \textit{limited data pairs} through a novel information theory-inspired formulation that simultaneously addresses distribution-level and instance-level alignment. Extensive experiments on two real-world IoT applications are performed to evaluate InfoMAE's pairing efficiency to bridge pretrained unimodal models into a cohesive joint multimodal model. InfoMAE enhances downstream multimodal tasks by over 60% with significantly improved multimodal pairing efficiency. It also improves unimodal task accuracy by an average of 22%. 

---
# Don't Deceive Me: Mitigating Gaslighting through Attention Reallocation in LMMs 

**Authors**: Pengkun Jiao, Bin Zhu, Jingjing Chen, Chong-Wah Ngo, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09456)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated remarkable capabilities across a wide range of tasks. However, their vulnerability to user gaslighting-the deliberate use of misleading or contradictory inputs-raises critical concerns about their reliability in real-world applications. In this paper, we address the novel and challenging issue of mitigating the negative impact of negation-based gaslighting on LMMs, where deceptive user statements lead to significant drops in model accuracy. Specifically, we introduce GasEraser, a training-free approach that reallocates attention weights from misleading textual tokens to semantically salient visual regions. By suppressing the influence of "attention sink" tokens and enhancing focus on visually grounded cues, GasEraser significantly improves LMM robustness without requiring retraining or additional supervision. Extensive experimental results demonstrate that GasEraser is effective across several leading open-source LMMs on the GaslightingBench. Notably, for LLaVA-v1.5-7B, GasEraser reduces the misguidance rate by 48.2%, demonstrating its potential for more trustworthy LMMs. 

---
# Draw with Thought: Unleashing Multimodal Reasoning for Scientific Diagram Generation 

**Authors**: Zhiqing Cui, Jiahao Yuan, Hanqing Wang, Yanshu Li, Chenxu Du, Zhenglong Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.09479)  

**Abstract**: Scientific diagrams are vital tools for communicating structured knowledge across disciplines. However, they are often published as static raster images, losing symbolic semantics and limiting reuse. While Multimodal Large Language Models (MLLMs) offer a pathway to bridging vision and structure, existing methods lack semantic control and structural interpretability, especially on complex diagrams. We propose Draw with Thought (DwT), a training-free framework that guides MLLMs to reconstruct diagrams into editable mxGraph XML code through cognitively-grounded Chain-of-Thought reasoning. DwT enables interpretable and controllable outputs without model fine-tuning by dividing the task into two stages: Coarse-to-Fine Planning, which handles perceptual structuring and semantic specification, and Structure-Aware Code Generation, enhanced by format-guided refinement. To support evaluation, we release Plot2XML, a benchmark of 247 real-world scientific diagrams with gold-standard XML annotations. Extensive experiments across eight MLLMs show that our approach yields high-fidelity, semantically aligned, and structurally valid reconstructions, with human evaluations confirming strong alignment in both accuracy and visual aesthetics, offering a scalable solution for converting static visuals into executable representations and advancing machine understanding of scientific graphics. 

---
# Mixed Signals: Decoding VLMs' Reasoning and Underlying Bias in Vision-Language Conflict 

**Authors**: Pouya Pezeshkpour, Moin Aminnaseri, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2504.08974)  

**Abstract**: Vision-language models (VLMs) have demonstrated impressive performance by effectively integrating visual and textual information to solve complex tasks. However, it is not clear how these models reason over the visual and textual data together, nor how the flow of information between modalities is structured. In this paper, we examine how VLMs reason by analyzing their biases when confronted with scenarios that present conflicting image and text cues, a common occurrence in real-world applications. To uncover the extent and nature of these biases, we build upon existing benchmarks to create five datasets containing mismatched image-text pairs, covering topics in mathematics, science, and visual descriptions. Our analysis shows that VLMs favor text in simpler queries but shift toward images as query complexity increases. This bias correlates with model scale, with the difference between the percentage of image- and text-preferred responses ranging from +56.8% (image favored) to -74.4% (text favored), depending on the task and model. In addition, we explore three mitigation strategies: simple prompt modifications, modifications that explicitly instruct models on how to handle conflicting information (akin to chain-of-thought prompting), and a task decomposition strategy that analyzes each modality separately before combining their results. Our findings indicate that the effectiveness of these strategies in identifying and mitigating bias varies significantly and is closely linked to the model's overall performance on the task and the specific modality in question. 

---
# GridMind: A Multi-Agent NLP Framework for Unified, Cross-Modal NFL Data Insights 

**Authors**: Jordan Chipka, Chris Moyer, Clay Troyer, Tyler Fuelling, Jeremy Hochstedler  

**Link**: [PDF](https://arxiv.org/pdf/2504.08747)  

**Abstract**: The rapid growth of big data and advancements in computational techniques have significantly transformed sports analytics. However, the diverse range of data sources -- including structured statistics, semi-structured formats like sensor data, and unstructured media such as written articles, audio, and video -- creates substantial challenges in extracting actionable insights. These various formats, often referred to as multimodal data, require integration to fully leverage their potential. Conventional systems, which typically prioritize structured data, face limitations when processing and combining these diverse content types, reducing their effectiveness in real-time sports analysis.
To address these challenges, recent research highlights the importance of multimodal data integration for capturing the complexity of real-world sports environments. Building on this foundation, this paper introduces GridMind, a multi-agent framework that unifies structured, semi-structured, and unstructured data through Retrieval-Augmented Generation (RAG) and large language models (LLMs) to facilitate natural language querying of NFL data. This approach aligns with the evolving field of multimodal representation learning, where unified models are increasingly essential for real-time, cross-modal interactions.
GridMind's distributed architecture includes specialized agents that autonomously manage each stage of a prompt -- from interpretation and data retrieval to response synthesis. This modular design enables flexible, scalable handling of multimodal data, allowing users to pose complex, context-rich questions and receive comprehensive, intuitive responses via a conversational interface. 

---
# Multimodal Long Video Modeling Based on Temporal Dynamic Context 

**Authors**: Haoran Hao, Jiaming Han, Yiyuan Zhang, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2504.10443)  

**Abstract**: Recent advances in Large Language Models (LLMs) have led to significant breakthroughs in video understanding. However, existing models still struggle with long video processing due to the context length constraint of LLMs and the vast amount of information within the video. Although some recent methods are designed for long video understanding, they often lose crucial information during token compression and struggle with additional modality like audio. In this work, we propose a dynamic long video encoding method utilizing the temporal relationship between frames, named Temporal Dynamic Context (TDC). Firstly, we segment the video into semantically consistent scenes based on inter-frame similarities, then encode each frame into tokens using visual-audio encoders. Secondly, we propose a novel temporal context compressor to reduce the number of tokens within each segment. Specifically, we employ a query-based Transformer to aggregate video, audio, and instruction text tokens into a limited set of temporal context tokens. Finally, we feed the static frame tokens and the temporal context tokens into the LLM for video understanding. Furthermore, to handle extremely long videos, we propose a training-free chain-of-thought strategy that progressively extracts answers from multiple video segments. These intermediate answers serve as part of the reasoning process and contribute to the final answer. We conduct extensive experiments on general video understanding and audio-video understanding benchmarks, where our method demonstrates strong performance. The code and models are available at this https URL. 

---
# Mavors: Multi-granularity Video Representation for Multimodal Large Language Model 

**Authors**: Yang Shi, Jiaheng Liu, Yushuo Guan, Zhenhua Wu, Yuanxing Zhang, Zihao Wang, Weihong Lin, Jingyun Hua, Zekun Wang, Xinlong Chen, Bohan Zeng, Wentao Zhang, Fuzheng Zhang, Wenjing Yang, Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10068)  

**Abstract**: Long-context video understanding in multimodal large language models (MLLMs) faces a critical challenge: balancing computational efficiency with the retention of fine-grained spatio-temporal patterns. Existing approaches (e.g., sparse sampling, dense sampling with low resolution, and token compression) suffer from significant information loss in temporal dynamics, spatial details, or subtle interactions, particularly in videos with complex motion or varying resolutions. To address this, we propose $\mathbf{Mavors}$, a novel framework that introduces $\mathbf{M}$ulti-gr$\mathbf{a}$nularity $\mathbf{v}$ide$\mathbf{o}$ $\mathbf{r}$epre$\mathbf{s}$entation for holistic long-video modeling. Specifically, Mavors directly encodes raw video content into latent representations through two core components: 1) an Intra-chunk Vision Encoder (IVE) that preserves high-resolution spatial features via 3D convolutions and Vision Transformers, and 2) an Inter-chunk Feature Aggregator (IFA) that establishes temporal coherence across chunks using transformer-based dependency modeling with chunk-level rotary position encodings. Moreover, the framework unifies image and video understanding by treating images as single-frame videos via sub-image decomposition. Experiments across diverse benchmarks demonstrate Mavors' superiority in maintaining both spatial fidelity and temporal continuity, significantly outperforming existing methods in tasks requiring fine-grained spatio-temporal reasoning. 

---
# RGB-Event based Pedestrian Attribute Recognition: A Benchmark Dataset and An Asymmetric RWKV Fusion Framework 

**Authors**: Xiao Wang, Haiyang Wang, Shiao Wang, Qiang Chen, Jiandong Jin, Haoyu Song, Bo Jiang, Chenglong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10018)  

**Abstract**: Existing pedestrian attribute recognition methods are generally developed based on RGB frame cameras. However, these approaches are constrained by the limitations of RGB cameras, such as sensitivity to lighting conditions and motion blur, which hinder their performance. Furthermore, current attribute recognition primarily focuses on analyzing pedestrians' external appearance and clothing, lacking an exploration of emotional dimensions. In this paper, we revisit these issues and propose a novel multi-modal RGB-Event attribute recognition task by drawing inspiration from the advantages of event cameras in low-light, high-speed, and low-power consumption. Specifically, we introduce the first large-scale multi-modal pedestrian attribute recognition dataset, termed EventPAR, comprising 100K paired RGB-Event samples that cover 50 attributes related to both appearance and six human emotions, diverse scenes, and various seasons. By retraining and evaluating mainstream PAR models on this dataset, we establish a comprehensive benchmark and provide a solid foundation for future research in terms of data and algorithmic baselines. In addition, we propose a novel RWKV-based multi-modal pedestrian attribute recognition framework, featuring an RWKV visual encoder and an asymmetric RWKV fusion module. Extensive experiments are conducted on our proposed dataset as well as two simulated datasets (MARS-Attribute and DukeMTMC-VID-Attribute), achieving state-of-the-art results. The source code and dataset will be released on this https URL 

---
# Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models? 

**Authors**: Yanbo Wang, Jiyang Guan, Jian Liang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2504.10000)  

**Abstract**: Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically designed for multi-modal inputs creates an alignment gap, leaving MLLMs vulnerable to vision-domain attacks such as typographic manipulation. Current methods utilize a carefully designed safety dataset to enhance model defense capability, while the specific knowledge or patterns acquired from the high-quality dataset remain unclear. Through comparison experiments, we find that the alignment gap primarily arises from data distribution biases, while image content, response quality, or the contrastive behavior of the dataset makes little contribution to boosting multi-modal safety. To further investigate this and identify the key factors in improving MLLM safety, we propose finetuning MLLMs on a small set of benign instruct-following data with responses replaced by simple, clear rejection sentences. Experiments show that, without the need for labor-intensive collection of high-quality malicious data, model safety can still be significantly improved, as long as a specific fraction of rejection data exists in the finetuning set, indicating the security alignment is not lost but rather obscured during multi-modal pretraining or instruction finetuning. Simply correcting the underlying data bias could narrow the safety gap in the vision domain. 

---
# FedRecon: Missing Modality Reconstruction in Distributed Heterogeneous Environments 

**Authors**: Junming Liu, Guosun Zeng, Ding Wang, Yanting Gao, Yufei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.09941)  

**Abstract**: Multimodal data are often incomplete and exhibit Non-Independent and Identically Distributed (Non-IID) characteristics in real-world scenarios. These inherent limitations lead to both modality heterogeneity through partial modality absence and data heterogeneity from distribution divergence, creating fundamental challenges for effective federated learning (FL). To address these coupled challenges, we propose FedRecon, the first method targeting simultaneous missing modality reconstruction and Non-IID adaptation in multimodal FL. Our approach first employs a lightweight Multimodal Variational Autoencoder (MVAE) to reconstruct missing modalities while preserving cross-modal consistency. Distinct from conventional imputation methods, we achieve sample-level alignment through a novel distribution mapping mechanism that guarantees both data consistency and completeness. Additionally, we introduce a strategy employing global generator freezing to prevent catastrophic forgetting, which in turn mitigates Non-IID fluctuations. Extensive evaluations on multimodal datasets demonstrate FedRecon's superior performance in modality reconstruction under Non-IID conditions, surpassing state-of-the-art methods. 

---
# GenTe: Generative Real-world Terrains for General Legged Robot Locomotion Control 

**Authors**: Hanwen Wan, Mengkang Li, Donghao Wu, Yebin Zhong, Yixuan Deng, Zhenglong Sun, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.09997)  

**Abstract**: Developing bipedal robots capable of traversing diverse real-world terrains presents a fundamental robotics challenge, as existing methods using predefined height maps and static environments fail to address the complexity of unstructured landscapes. To bridge this gap, we propose GenTe, a framework for generating physically realistic and adaptable terrains to train generalizable locomotion policies. GenTe constructs an atomic terrain library that includes both geometric and physical terrains, enabling curriculum training for reinforcement learning-based locomotion policies. By leveraging function-calling techniques and reasoning capabilities of Vision-Language Models (VLMs), GenTe generates complex, contextually relevant terrains from textual and graphical inputs. The framework introduces realistic force modeling for terrain interactions, capturing effects such as soil sinkage and hydrodynamic resistance. To the best of our knowledge, GenTe is the first framework that systemically generates simulation environments for legged robot locomotion control. Additionally, we introduce a benchmark of 100 generated terrains. Experiments demonstrate improved generalization and robustness in bipedal robot locomotion. 

---
# See or Recall: A Sanity Check for the Role of Vision in Solving Visualization Question Answer Tasks with Multimodal LLMs 

**Authors**: Zhimin Li, Haichao Miao, Xinyuan Yan, Valerio Pascucci, Matthew Berger, Shusen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09809)  

**Abstract**: Recent developments in multimodal large language models (MLLM) have equipped language models to reason about vision and language jointly. This permits MLLMs to both perceive and answer questions about data visualization across a variety of designs and tasks. Applying MLLMs to a broad range of visualization tasks requires us to properly evaluate their capabilities, and the most common way to conduct evaluation is through measuring a model's visualization reasoning capability, analogous to how we would evaluate human understanding of visualizations (e.g., visualization literacy). However, we found that in the context of visualization question answering (VisQA), how an MLLM perceives and reasons about visualizations can be fundamentally different from how humans approach the same problem. During the evaluation, even without visualization, the model could correctly answer a substantial portion of the visualization test questions, regardless of whether any selection options were provided. We hypothesize that the vast amount of knowledge encoded in the language model permits factual recall that supersedes the need to seek information from the visual signal. It raises concerns that the current VisQA evaluation may not fully capture the models' visualization reasoning capabilities. To address this, we propose a comprehensive sanity check framework that integrates a rule-based decision tree and a sanity check table to disentangle the effects of "seeing" (visual processing) and "recall" (reliance on prior knowledge). This validates VisQA datasets for evaluation, highlighting where models are truly "seeing", positively or negatively affected by the factual recall, or relying on inductive biases for question answering. Our study underscores the need for careful consideration in designing future visualization understanding studies when utilizing MLLMs. 

---
# VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents 

**Authors**: Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2504.09795)  

**Abstract**: We aim to develop a retrieval-augmented generation (RAG) framework that answers questions over a corpus of visually-rich documents presented in mixed modalities (e.g., charts, tables) and diverse formats (e.g., PDF, PPTX). In this paper, we introduce a new RAG framework, VDocRAG, which can directly understand varied documents and modalities in a unified image format to prevent missing information that occurs by parsing documents to obtain text. To improve the performance, we propose novel self-supervised pre-training tasks that adapt large vision-language models for retrieval by compressing visual information into dense token representations while aligning them with textual content in documents. Furthermore, we introduce OpenDocVQA, the first unified collection of open-domain document visual question answering datasets, encompassing diverse document types and formats. OpenDocVQA provides a comprehensive resource for training and evaluating retrieval and question answering models on visually-rich documents in an open-domain setting. Experiments show that VDocRAG substantially outperforms conventional text-based RAG and has strong generalization capability, highlighting the potential of an effective RAG paradigm for real-world documents. 

---
# Embodied Chain of Action Reasoning with Multi-Modal Foundation Model for Humanoid Loco-manipulation 

**Authors**: Yu Hao, Geeta Chandra Raju Bethala, Niraj Pudasaini, Hao Huang, Shuaihang Yuan, Congcong Wen, Baoru Huang, Anh Nguyen, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09532)  

**Abstract**: Enabling humanoid robots to autonomously perform loco-manipulation tasks in complex, unstructured environments poses significant challenges. This entails equipping robots with the capability to plan actions over extended horizons while leveraging multi-modality to bridge gaps between high-level planning and actual task execution. Recent advancements in multi-modal foundation models have showcased substantial potential in enhancing planning and reasoning abilities, particularly in the comprehension and processing of semantic information for robotic control tasks. In this paper, we introduce a novel framework based on foundation models that applies the embodied chain of action reasoning methodology to autonomously plan actions from textual instructions for humanoid loco-manipulation. Our method integrates humanoid-specific chain of thought methodology, including detailed affordance and body movement analysis, which provides a breakdown of the task into a sequence of locomotion and manipulation actions. Moreover, we incorporate spatial reasoning based on the observation and target object properties to effectively navigate where target position may be unseen or occluded. Through rigorous experimental setups on object rearrangement, manipulations and loco-manipulation tasks on a real-world environment, we evaluate our method's efficacy on the decoupled upper and lower body control and demonstrate the effectiveness of the chain of robotic action reasoning strategies in comprehending human instructions. 

---
# Metropolis-Hastings Captioning Game: Knowledge Fusion of Vision Language Models via Decentralized Bayesian Inference 

**Authors**: Yuta Matsui, Ryosuke Yamaki, Ryo Ueda, Seitaro Shinagawa, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.09620)  

**Abstract**: We propose the Metropolis-Hastings Captioning Game (MHCG), a method to fuse knowledge of multiple vision-language models (VLMs) by learning from each other. Although existing methods that combine multiple models suffer from inference costs and architectural constraints, MHCG avoids these problems by performing decentralized Bayesian inference through a process resembling a language game. The knowledge fusion process establishes communication between two VLM agents alternately captioning images and learning from each other. We conduct two image-captioning experiments with two VLMs, each pre-trained on a different dataset. The first experiment demonstrates that MHCG achieves consistent improvement in reference-free evaluation metrics. The second experiment investigates how MHCG contributes to sharing VLMs' category-level vocabulary by observing the occurrence of the vocabulary in the generated captions. 

---
# AirVista-II: An Agentic System for Embodied UAVs Toward Dynamic Scene Semantic Understanding 

**Authors**: Fei Lin, Yonglin Tian, Tengchao Zhang, Jun Huang, Sangtian Guan, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09583)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly important in dynamic environments such as logistics transportation and disaster response. However, current tasks often rely on human operators to monitor aerial videos and make operational decisions. This mode of human-machine collaboration suffers from significant limitations in efficiency and adaptability. In this paper, we present AirVista-II -- an end-to-end agentic system for embodied UAVs, designed to enable general-purpose semantic understanding and reasoning in dynamic scenes. The system integrates agent-based task identification and scheduling, multimodal perception mechanisms, and differentiated keyframe extraction strategies tailored for various temporal scenarios, enabling the efficient capture of critical scene information. Experimental results demonstrate that the proposed system achieves high-quality semantic understanding across diverse UAV-based dynamic scenarios under a zero-shot setting. 

---
# TextSplat: Text-Guided Semantic Fusion for Generalizable Gaussian Splatting 

**Authors**: Zhicong Wu, Hongbin Xu, Gang Xu, Ping Nie, Zhixin Yan, Jinkai Zheng, Liangqiong Qu, Ming Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2504.09588)  

**Abstract**: Recent advancements in Generalizable Gaussian Splatting have enabled robust 3D reconstruction from sparse input views by utilizing feed-forward Gaussian Splatting models, achieving superior cross-scene generalization. However, while many methods focus on geometric consistency, they often neglect the potential of text-driven guidance to enhance semantic understanding, which is crucial for accurately reconstructing fine-grained details in complex scenes. To address this limitation, we propose TextSplat--the first text-driven Generalizable Gaussian Splatting framework. By employing a text-guided fusion of diverse semantic cues, our framework learns robust cross-modal feature representations that improve the alignment of geometric and semantic information, producing high-fidelity 3D reconstructions. Specifically, our framework employs three parallel modules to obtain complementary representations: the Diffusion Prior Depth Estimator for accurate depth information, the Semantic Aware Segmentation Network for detailed semantic information, and the Multi-View Interaction Network for refined cross-view features. Then, in the Text-Guided Semantic Fusion Module, these representations are integrated via the text-guided and attention-based feature aggregation mechanism, resulting in enhanced 3D Gaussian parameters enriched with detailed semantic cues. Experimental results on various benchmark datasets demonstrate improved performance compared to existing methods across multiple evaluation metrics, validating the effectiveness of our framework. The code will be publicly available. 

---
# BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning 

**Authors**: Shengao Wang, Arjun Chandra, Aoming Liu, Venkatesh Saligrama, Boqing Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.09426)  

**Abstract**: Human infants rapidly develop visual reasoning skills from minimal input, suggesting that developmentally inspired pretraining could significantly enhance the efficiency of vision-language models (VLMs). Although recent efforts have leveraged infant-inspired datasets like SAYCam, existing evaluation benchmarks remain misaligned--they are either too simplistic, narrowly scoped, or tailored for large-scale pretrained models. Additionally, training exclusively on infant data overlooks the broader, diverse input from which infants naturally learn. To address these limitations, we propose BabyVLM, a novel framework comprising comprehensive in-domain evaluation benchmarks and a synthetic training dataset created via child-directed transformations of existing datasets. We demonstrate that VLMs trained with our synthetic dataset achieve superior performance on BabyVLM tasks compared to models trained solely on SAYCam or general-purpose data of the SAYCam size. BabyVLM thus provides a robust, developmentally aligned evaluation tool and illustrates how compact models trained on carefully curated data can generalize effectively, opening pathways toward data-efficient vision-language learning paradigms. 

---
# REMEMBER: Retrieval-based Explainable Multimodal Evidence-guided Modeling for Brain Evaluation and Reasoning in Zero- and Few-shot Neurodegenerative Diagnosis 

**Authors**: Duy-Cat Can, Quang-Huy Tang, Huong Ha, Binh T. Nguyen, Oliver Y. Chén  

**Link**: [PDF](https://arxiv.org/pdf/2504.09354)  

**Abstract**: Timely and accurate diagnosis of neurodegenerative disorders, such as Alzheimer's disease, is central to disease management. Existing deep learning models require large-scale annotated datasets and often function as "black boxes". Additionally, datasets in clinical practice are frequently small or unlabeled, restricting the full potential of deep learning methods. Here, we introduce REMEMBER -- Retrieval-based Explainable Multimodal Evidence-guided Modeling for Brain Evaluation and Reasoning -- a new machine learning framework that facilitates zero- and few-shot Alzheimer's diagnosis using brain MRI scans through a reference-based reasoning process. Specifically, REMEMBER first trains a contrastively aligned vision-text model using expert-annotated reference data and extends pseudo-text modalities that encode abnormality types, diagnosis labels, and composite clinical descriptions. Then, at inference time, REMEMBER retrieves similar, human-validated cases from a curated dataset and integrates their contextual information through a dedicated evidence encoding module and attention-based inference head. Such an evidence-guided design enables REMEMBER to imitate real-world clinical decision-making process by grounding predictions in retrieved imaging and textual context. Specifically, REMEMBER outputs diagnostic predictions alongside an interpretable report, including reference images and explanations aligned with clinical workflows. Experimental results demonstrate that REMEMBER achieves robust zero- and few-shot performance and offers a powerful and explainable framework to neuroimaging-based diagnosis in the real world, especially under limited data. 

---
# MiMIC: Multi-Modal Indian Earnings Calls Dataset to Predict Stock Prices 

**Authors**: Sohom Ghosh, Arnab Maji, Sudip Kumar Naskar  

**Link**: [PDF](https://arxiv.org/pdf/2504.09257)  

**Abstract**: Predicting stock market prices following corporate earnings calls remains a significant challenge for investors and researchers alike, requiring innovative approaches that can process diverse information sources. This study investigates the impact of corporate earnings calls on stock prices by introducing a multi-modal predictive model. We leverage textual data from earnings call transcripts, along with images and tables from accompanying presentations, to forecast stock price movements on the trading day immediately following these calls. To facilitate this research, we developed the MiMIC (Multi-Modal Indian Earnings Calls) dataset, encompassing companies representing the Nifty 50, Nifty MidCap 50, and Nifty Small 50 indices. The dataset includes earnings call transcripts, presentations, fundamentals, technical indicators, and subsequent stock prices. We present a multimodal analytical framework that integrates quantitative variables with predictive signals derived from textual and visual modalities, thereby enabling a holistic approach to feature representation and analysis. This multi-modal approach demonstrates the potential for integrating diverse information sources to enhance financial forecasting accuracy. To promote further research in computational economics, we have made the MiMIC dataset publicly available under the CC-NC-SA-4.0 licence. Our work contributes to the growing body of literature on market reactions to corporate communications and highlights the efficacy of multi-modal machine learning techniques in financial analysis. 

---
# ReferGPT: Towards Zero-Shot Referring Multi-Object Tracking 

**Authors**: Tzoulio Chamiti, Leandro Di Bella, Adrian Munteanu, Nikos Deligiannis  

**Link**: [PDF](https://arxiv.org/pdf/2504.09195)  

**Abstract**: Tracking multiple objects based on textual queries is a challenging task that requires linking language understanding with object association across frames. Previous works typically train the whole process end-to-end or integrate an additional referring text module into a multi-object tracker, but they both require supervised training and potentially struggle with generalization to open-set queries. In this work, we introduce ReferGPT, a novel zero-shot referring multi-object tracking framework. We provide a multi-modal large language model (MLLM) with spatial knowledge enabling it to generate 3D-aware captions. This enhances its descriptive capabilities and supports a more flexible referring vocabulary without training. We also propose a robust query-matching strategy, leveraging CLIP-based semantic encoding and fuzzy matching to associate MLLM generated captions with user queries. Extensive experiments on Refer-KITTI, Refer-KITTIv2 and Refer-KITTI+ demonstrate that ReferGPT achieves competitive performance against trained methods, showcasing its robustness and zero-shot capabilities in autonomous driving. The codes are available on this https URL 

---
# Multimodal 3D Genome Pre-training 

**Authors**: Minghao Yang, Pengteng Li, Yan Liang, Qianyi Cai, Zhihang Zheng, Shichen Zhang, Pengfei Zhang, Zhi-An Huang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2504.09060)  

**Abstract**: Deep learning techniques have driven significant progress in various analytical tasks within 3D genomics in computational biology. However, a holistic understanding of 3D genomics knowledge remains underexplored. Here, we propose MIX-HIC, the first multimodal foundation model of 3D genome that integrates both 3D genome structure and epigenomic tracks, which obtains unified and comprehensive semantics. For accurate heterogeneous semantic fusion, we design the cross-modal interaction and mapping blocks for robust unified representation, yielding the accurate aggregation of 3D genome knowledge. Besides, we introduce the first large-scale dataset comprising over 1 million pairwise samples of Hi-C contact maps and epigenomic tracks for high-quality pre-training, enabling the exploration of functional implications in 3D genomics. Extensive experiments show that MIX-HIC can significantly surpass existing state-of-the-art methods in diverse downstream tasks. This work provides a valuable resource for advancing 3D genomics research. 

---
# HyperCore: The Core Framework for Building Hyperbolic Foundation Models with Comprehensive Modules 

**Authors**: Neil He, Menglin Yang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2504.08912)  

**Abstract**: Hyperbolic neural networks have emerged as a powerful tool for modeling hierarchical data across diverse modalities. Recent studies show that token distributions in foundation models exhibit scale-free properties, suggesting that hyperbolic space is a more suitable ambient space than Euclidean space for many pre-training and downstream tasks. However, existing tools lack essential components for building hyperbolic foundation models, making it difficult to leverage recent advancements. We introduce HyperCore, a comprehensive open-source framework that provides core modules for constructing hyperbolic foundation models across multiple modalities. HyperCore's modules can be effortlessly combined to develop novel hyperbolic foundation models, eliminating the need to extensively modify Euclidean modules from scratch and possible redundant research efforts. To demonstrate its versatility, we build and test the first fully hyperbolic vision transformers (LViT) with a fine-tuning pipeline, the first fully hyperbolic multimodal CLIP model (L-CLIP), and a hybrid Graph RAG with a hyperbolic graph encoder. Our experiments demonstrate that LViT outperforms its Euclidean counterpart. Additionally, we benchmark and reproduce experiments across hyperbolic GNNs, CNNs, Transformers, and vision Transformers to highlight HyperCore's advantages. 

---
# Mimic In-Context Learning for Multimodal Tasks 

**Authors**: Yuchu Jiang, Jiale Fu, Chenduo Hao, Xinting Hu, Yingzhe Peng, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08851)  

**Abstract**: Recently, In-context Learning (ICL) has become a significant inference paradigm in Large Multimodal Models (LMMs), utilizing a few in-context demonstrations (ICDs) to prompt LMMs for new tasks. However, the synergistic effects in multimodal data increase the sensitivity of ICL performance to the configurations of ICDs, stimulating the need for a more stable and general mapping function. Mathematically, in Transformer-based models, ICDs act as ``shift vectors'' added to the hidden states of query tokens. Inspired by this, we introduce Mimic In-Context Learning (MimIC) to learn stable and generalizable shift effects from ICDs. Specifically, compared with some previous shift vector-based methods, MimIC more strictly approximates the shift effects by integrating lightweight learnable modules into LMMs with four key enhancements: 1) inserting shift vectors after attention layers, 2) assigning a shift vector to each attention head, 3) making shift magnitude query-dependent, and 4) employing a layer-wise alignment loss. Extensive experiments on two LMMs (Idefics-9b and Idefics2-8b-base) across three multimodal tasks (VQAv2, OK-VQA, Captioning) demonstrate that MimIC outperforms existing shift vector-based methods. The code is available at this https URL. 

---
# VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning 

**Authors**: Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08837)  

**Abstract**: Recently, slow-thinking systems like GPT-o1 and DeepSeek-R1 have demonstrated great potential in solving challenging problems through explicit reflection. They significantly outperform the best fast-thinking models, such as GPT-4o, on various math and science benchmarks. However, their multimodal reasoning capabilities remain on par with fast-thinking models. For instance, GPT-o1's performance on benchmarks like MathVista, MathVerse, and MathVision is similar to fast-thinking models. In this paper, we aim to enhance the slow-thinking capabilities of vision-language models using reinforcement learning (without relying on distillation) to advance the state of the art. First, we adapt the GRPO algorithm with a novel technique called Selective Sample Replay (SSR) to address the vanishing advantages problem. While this approach yields strong performance, the resulting RL-trained models exhibit limited self-reflection or self-verification. To further encourage slow-thinking, we introduce Forced Rethinking, which appends a textual rethinking trigger to the end of initial rollouts in RL training, explicitly enforcing a self-reflection reasoning step. By combining these two techniques, our model, VL-Rethinker, advances state-of-the-art scores on MathVista, MathVerse, and MathVision to achieve 80.3%, 61.8%, and 43.9% respectively. VL-Rethinker also achieves open-source SoTA on multi-disciplinary benchmarks such as MMMU-Pro, EMMA, and MEGA-Bench, narrowing the gap with GPT-o1. 

---
# SafeMLRM: Demystifying Safety in Multi-modal Large Reasoning Models 

**Authors**: Junfeng Fang, Yukai Wang, Ruipeng Wang, Zijun Yao, Kun Wang, An Zhang, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.08813)  

**Abstract**: The rapid advancement of multi-modal large reasoning models (MLRMs) -- enhanced versions of multimodal language models (MLLMs) equipped with reasoning capabilities -- has revolutionized diverse applications. However, their safety implications remain underexplored. While prior work has exposed critical vulnerabilities in unimodal reasoning models, MLRMs introduce distinct risks from cross-modal reasoning pathways. This work presents the first systematic safety analysis of MLRMs through large-scale empirical studies comparing MLRMs with their base MLLMs. Our experiments reveal three critical findings: (1) The Reasoning Tax: Acquiring reasoning capabilities catastrophically degrades inherited safety alignment. MLRMs exhibit 37.44% higher jailbreaking success rates than base MLLMs under adversarial attacks. (2) Safety Blind Spots: While safety degradation is pervasive, certain scenarios (e.g., Illegal Activity) suffer 25 times higher attack rates -- far exceeding the average 3.4 times increase, revealing scenario-specific vulnerabilities with alarming cross-model and datasets consistency. (3) Emergent Self-Correction: Despite tight reasoning-answer safety coupling, MLRMs demonstrate nascent self-correction -- 16.9% of jailbroken reasoning steps are overridden by safe answers, hinting at intrinsic safeguards. These findings underscore the urgency of scenario-aware safety auditing and mechanisms to amplify MLRMs' self-correction potential. To catalyze research, we open-source OpenSafeMLRM, the first toolkit for MLRM safety evaluation, providing unified interface for mainstream models, datasets, and jailbreaking methods. Our work calls for immediate efforts to harden reasoning-augmented AI, ensuring its transformative potential aligns with ethical safeguards. 

---
# Research on the Design of a Short Video Recommendation System Based on Multimodal Information and Differential Privacy 

**Authors**: Haowei Yang, Lei Fu, Qingyi Lu, Yue Fan, Tianle Zhang, Ruohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08751)  

**Abstract**: With the rapid development of short video platforms, recommendation systems have become key technologies for improving user experience and enhancing platform engagement. However, while short video recommendation systems leverage multimodal information (such as images, text, and audio) to improve recommendation effectiveness, they also face the severe challenge of user privacy leakage. This paper proposes a short video recommendation system based on multimodal information and differential privacy protection. First, deep learning models are used for feature extraction and fusion of multimodal data, effectively improving recommendation accuracy. Then, a differential privacy protection mechanism suitable for recommendation scenarios is designed to ensure user data privacy while maintaining system performance. Experimental results show that the proposed method outperforms existing mainstream approaches in terms of recommendation accuracy, multimodal fusion effectiveness, and privacy protection performance, providing important insights for the design of recommendation systems for short video platforms. 

---
# A Survey of Multimodal Retrieval-Augmented Generation 

**Authors**: Lang Mei, Siyu Mo, Zhihan Yang, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08748)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) enhances large language models (LLMs) by integrating multimodal data (text, images, videos) into retrieval and generation processes, overcoming the limitations of text-only Retrieval-Augmented Generation (RAG). While RAG improves response accuracy by incorporating external textual knowledge, MRAG extends this framework to include multimodal retrieval and generation, leveraging contextual information from diverse data types. This approach reduces hallucinations and enhances question-answering systems by grounding responses in factual, multimodal knowledge. Recent studies show MRAG outperforms traditional RAG, especially in scenarios requiring both visual and textual understanding. This survey reviews MRAG's essential components, datasets, evaluation methods, and limitations, providing insights into its construction and improvement. It also identifies challenges and future research directions, highlighting MRAG's potential to revolutionize multimodal information retrieval and generation. By offering a comprehensive perspective, this work encourages further exploration into this promising paradigm. 

---
# VisualPuzzles: Decoupling Multimodal Reasoning Evaluation from Domain Knowledge 

**Authors**: Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, Xiang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2504.10342)  

**Abstract**: Current multimodal benchmarks often conflate reasoning with domain-specific knowledge, making it difficult to isolate and evaluate general reasoning abilities in non-expert settings. To address this, we introduce VisualPuzzles, a benchmark that targets visual reasoning while deliberately minimizing reliance on specialized knowledge. VisualPuzzles consists of diverse questions spanning five categories: algorithmic, analogical, deductive, inductive, and spatial reasoning. One major source of our questions is manually translated logical reasoning questions from the Chinese Civil Service Examination. Experiments show that VisualPuzzles requires significantly less intensive domain-specific knowledge and more complex reasoning compared to benchmarks like MMMU, enabling us to better evaluate genuine multimodal reasoning. Evaluations show that state-of-the-art multimodal large language models consistently lag behind human performance on VisualPuzzles, and that strong performance on knowledge-intensive benchmarks does not necessarily translate to success on reasoning-focused, knowledge-light tasks. Additionally, reasoning enhancements such as scaling up inference compute (with "thinking" modes) yield inconsistent gains across models and task types, and we observe no clear correlation between model size and performance. We also found that models exhibit different reasoning and answering patterns on VisualPuzzles compared to benchmarks with heavier emphasis on knowledge. VisualPuzzles offers a clearer lens through which to evaluate reasoning capabilities beyond factual recall and domain knowledge. 

---
# DataMosaic: Explainable and Verifiable Multi-Modal Data Analytics through Extract-Reason-Verify 

**Authors**: Zhengxuan Zhang, Zhuowen Liang, Yin Wu, Teng Lin, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10036)  

**Abstract**: Large Language Models (LLMs) are transforming data analytics, but their widespread adoption is hindered by two critical limitations: they are not explainable (opaque reasoning processes) and not verifiable (prone to hallucinations and unchecked errors). While retrieval-augmented generation (RAG) improves accuracy by grounding LLMs in external data, it fails to address the core challenges of trustworthy analytics - especially when processing noisy, inconsistent, or multi-modal data (for example, text, tables, images). We propose DataMosaic, a framework designed to make LLM-powered analytics both explainable and verifiable. By dynamically extracting task-specific structures (for example, tables, graphs, trees) from raw data, DataMosaic provides transparent, step-by-step reasoning traces and enables validation of intermediate results. Built on a multi-agent framework, DataMosaic orchestrates self-adaptive agents that align with downstream task requirements, enhancing consistency, completeness, and privacy. Through this approach, DataMosaic not only tackles the limitations of current LLM-powered analytics systems but also lays the groundwork for a new paradigm of grounded, accurate, and explainable multi-modal data analytics. 

---
# The Mirage of Performance Gains: Why Contrastive Decoding Fails to Address Multimodal Hallucination 

**Authors**: Hao Yin, Gunagzong Si, Zilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10020)  

**Abstract**: Contrastive decoding strategies are widely used to reduce hallucinations in multimodal large language models (MLLMs). These methods work by constructing contrastive samples to induce hallucinations and then suppressing them in the output distribution. However, this paper demonstrates that such approaches fail to effectively mitigate the hallucination problem. The performance improvements observed on POPE Benchmark are largely driven by two misleading factors: (1) crude, unidirectional adjustments to the model's output distribution and (2) the adaptive plausibility constraint, which reduces the sampling strategy to greedy search. To further illustrate these issues, we introduce a series of spurious improvement methods and evaluate their performance against contrastive decoding techniques. Experimental results reveal that the observed performance gains in contrastive decoding are entirely unrelated to its intended goal of mitigating hallucinations. Our findings challenge common assumptions about the effectiveness of contrastive decoding strategies and pave the way for developing genuinely effective solutions to hallucinations in MLLMs. 

---
# VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search 

**Authors**: Yikun Wang, Siyin Wang, Qinyuan Cheng, Zhaoye Fei, Liang Ding, Qipeng Guo, Dacheng Tao, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09130)  

**Abstract**: Recent advancements in Large Vision-Language Models have showcased remarkable capabilities. However, they often falter when confronted with complex reasoning tasks that humans typically address through visual aids and deliberate, step-by-step thinking. While existing methods have explored text-based slow thinking or rudimentary visual assistance, they fall short of capturing the intricate, interleaved nature of human visual-verbal reasoning processes. To overcome these limitations and inspired by the mechanisms of slow thinking in human cognition, we introduce VisuoThink, a novel framework that seamlessly integrates visuospatial and linguistic domains. VisuoThink facilitates multimodal slow thinking by enabling progressive visual-textual reasoning and incorporates test-time scaling through look-ahead tree search. Extensive experiments demonstrate that VisuoThink significantly enhances reasoning capabilities via inference-time scaling, even without fine-tuning, achieving state-of-the-art performance in tasks involving geometry and spatial reasoning. 

---
# MIEB: Massive Image Embedding Benchmark 

**Authors**: Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie Stirling, Xin Zhang, Márton Kardos, Roman Solomatin, Noura Al Moubayed, Kenneth Enevoldsen, Niklas Muennighoff  

**Link**: [PDF](https://arxiv.org/pdf/2504.10471)  

**Abstract**: Image representations are often evaluated through disjointed, task-specific protocols, leading to a fragmented understanding of model capabilities. For instance, it is unclear whether an image embedding model adept at clustering images is equally good at retrieving relevant images given a piece of text. We introduce the Massive Image Embedding Benchmark (MIEB) to evaluate the performance of image and image-text embedding models across the broadest spectrum to date. MIEB spans 38 languages across 130 individual tasks, which we group into 8 high-level categories. We benchmark 50 models across our benchmark, finding that no single method dominates across all task categories. We reveal hidden capabilities in advanced vision models such as their accurate visual representation of texts, and their yet limited capabilities in interleaved encodings and matching images and texts in the presence of confounders. We also show that the performance of vision encoders on MIEB correlates highly with their performance when used in multimodal large language models. Our code, dataset, and leaderboard are publicly available at this https URL. 

---
# Summarization of Multimodal Presentations with Vision-Language Models: Study of the Effect of Modalities and Structure 

**Authors**: Théo Gigant, Camille Guinaudeau, Frédéric Dufaux  

**Link**: [PDF](https://arxiv.org/pdf/2504.10049)  

**Abstract**: Vision-Language Models (VLMs) can process visual and textual information in multiple formats: texts, images, interleaved texts and images, or even hour-long videos. In this work, we conduct fine-grained quantitative and qualitative analyses of automatic summarization of multimodal presentations using VLMs with various representations as input. From these experiments, we suggest cost-effective strategies for generating summaries from text-heavy multimodal documents under different input-length budgets using VLMs. We show that slides extracted from the video stream can be beneficially used as input against the raw video, and that a structured representation from interleaved slides and transcript provides the best performance. Finally, we reflect and comment on the nature of cross-modal interactions in multimodal presentations and share suggestions to improve the capabilities of VLMs to understand documents of this nature. 

---
# CameraBench: Benchmarking Visual Reasoning in MLLMs via Photography 

**Authors**: I-Sheng Fang, Jun-Cheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.10090)  

**Abstract**: Large language models (LLMs) and multimodal large language models (MLLMs) have significantly advanced artificial intelligence. However, visual reasoning, reasoning involving both visual and textual inputs, remains underexplored. Recent advancements, including the reasoning models like OpenAI o1 and Gemini 2.0 Flash Thinking, which incorporate image inputs, have opened this capability. In this ongoing work, we focus specifically on photography-related tasks because a photo is a visual snapshot of the physical world where the underlying physics (i.e., illumination, blur extent, etc.) interplay with the camera parameters. Successfully reasoning from the visual information of a photo to identify these numerical camera settings requires the MLLMs to have a deeper understanding of the underlying physics for precise visual comprehension, representing a challenging and intelligent capability essential for practical applications like photography assistant agents. We aim to evaluate MLLMs on their ability to distinguish visual differences related to numerical camera settings, extending a methodology previously proposed for vision-language models (VLMs). Our preliminary results demonstrate the importance of visual reasoning in photography-related tasks. Moreover, these results show that no single MLLM consistently dominates across all evaluation tasks, demonstrating ongoing challenges and opportunities in developing MLLMs with better visual reasoning. 

---
# HistLLM: A Unified Framework for LLM-Based Multimodal Recommendation with User History Encoding and Compression 

**Authors**: Chen Zhang, Bo Hu, Weidong Chen, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.10150)  

**Abstract**: While large language models (LLMs) have proven effective in leveraging textual data for recommendations, their application to multimodal recommendation tasks remains relatively underexplored. Although LLMs can process multimodal information through projection functions that map visual features into their semantic space, recommendation tasks often require representing users' history interactions through lengthy prompts combining text and visual elements, which not only hampers training and inference efficiency but also makes it difficult for the model to accurately capture user preferences from complex and extended prompts, leading to reduced recommendation performance. To address this challenge, we introduce HistLLM, an innovative multimodal recommendation framework that integrates textual and visual features through a User History Encoding Module (UHEM), compressing multimodal user history interactions into a single token representation, effectively facilitating LLMs in processing user preferences. Extensive experiments demonstrate the effectiveness and efficiency of our proposed mechanism. 

---
# CROSSAN: Towards Efficient and Effective Adaptation of Multiple Multimodal Foundation Models for Sequential Recommendation 

**Authors**: Junchen Fu, Yongxin Ni, Joemon M. Jose, Ioannis Arapakis, Kaiwen Zheng, Youhua Li, Xuri Ge  

**Link**: [PDF](https://arxiv.org/pdf/2504.10307)  

**Abstract**: Multimodal Foundation Models (MFMs) excel at representing diverse raw modalities (e.g., text, images, audio, videos, etc.). As recommender systems increasingly incorporate these modalities, leveraging MFMs to generate better representations has great potential. However, their application in sequential recommendation remains largely unexplored. This is primarily because mainstream adaptation methods, such as Fine-Tuning and even Parameter-Efficient Fine-Tuning (PEFT) techniques (e.g., Adapter and LoRA), incur high computational costs, especially when integrating multiple modality encoders, thus hindering research progress. As a result, it remains unclear whether we can efficiently and effectively adapt multiple (>2) MFMs for the sequential recommendation task.
To address this, we propose a plug-and-play Cross-modal Side Adapter Network (CROSSAN). Leveraging the fully decoupled side adapter-based paradigm, CROSSAN achieves high efficiency while enabling cross-modal learning across diverse modalities. To optimize the final stage of multimodal fusion across diverse modalities, we adopt the Mixture of Modality Expert Fusion (MOMEF) mechanism. CROSSAN achieves superior performance on the public datasets for adapting four foundation models with raw modalities. Performance consistently improves as more MFMs are adapted. We will release our code and datasets to facilitate future research. 

---
# UltraRAG: A Modular and Automated Toolkit for Adaptive Retrieval-Augmented Generation 

**Authors**: Yuxuan Chen, Dewen Guo, Sen Mei, Xinze Li, Hao Chen, Yishan Li, Yixuan Wang, Chaoyue Tang, Ruobing Wang, Dingjun Wu, Yukun Yan, Zhenghao Liu, Shi Yu, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.08761)  

**Abstract**: Retrieval-Augmented Generation (RAG) significantly enhances the performance of large language models (LLMs) in downstream tasks by integrating external knowledge. To facilitate researchers in deploying RAG systems, various RAG toolkits have been introduced. However, many existing RAG toolkits lack support for knowledge adaptation tailored to specific application scenarios. To address this limitation, we propose UltraRAG, a RAG toolkit that automates knowledge adaptation throughout the entire workflow, from data construction and training to evaluation, while ensuring ease of use. UltraRAG features a user-friendly WebUI that streamlines the RAG process, allowing users to build and optimize systems without coding expertise. It supports multimodal input and provides comprehensive tools for managing the knowledge base. With its highly modular architecture, UltraRAG delivers an end-to-end development solution, enabling seamless knowledge adaptation across diverse user scenarios. The code, demonstration videos, and installable package for UltraRAG are publicly available at this https URL. 

---
# NoTeS-Bank: Benchmarking Neural Transcription and Search for Scientific Notes Understanding 

**Authors**: Aniket Pal, Sanket Biswas, Alloy Das, Ayush Lodh, Priyanka Banerjee, Soumitri Chattopadhyay, Dimosthenis Karatzas, Josep Llados, C.V. Jawahar  

**Link**: [PDF](https://arxiv.org/pdf/2504.09249)  

**Abstract**: Understanding and reasoning over academic handwritten notes remains a challenge in document AI, particularly for mathematical equations, diagrams, and scientific notations. Existing visual question answering (VQA) benchmarks focus on printed or structured handwritten text, limiting generalization to real-world note-taking. To address this, we introduce NoTeS-Bank, an evaluation benchmark for Neural Transcription and Search in note-based question answering. NoTeS-Bank comprises complex notes across multiple domains, requiring models to process unstructured and multimodal content. The benchmark defines two tasks: (1) Evidence-Based VQA, where models retrieve localized answers with bounding-box evidence, and (2) Open-Domain VQA, where models classify the domain before retrieving relevant documents and answers. Unlike classical Document VQA datasets relying on optical character recognition (OCR) and structured data, NoTeS-BANK demands vision-language fusion, retrieval, and multimodal reasoning. We benchmark state-of-the-art Vision-Language Models (VLMs) and retrieval frameworks, exposing structured transcription and reasoning limitations. NoTeS-Bank provides a rigorous evaluation with NDCG@5, MRR, Recall@K, IoU, and ANLS, establishing a new standard for visual document understanding and reasoning. 

---
# Enhancing Product Search Interfaces with Sketch-Guided Diffusion and Language Agents 

**Authors**: Edward Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.08739)  

**Abstract**: The rapid progress in diffusion models, transformers, and language agents has unlocked new possibilities, yet their potential in user interfaces and commercial applications remains underexplored. We present Sketch-Search Agent, a novel framework that transforms the image search experience by integrating a multimodal language agent with freehand sketches as control signals for diffusion models. Using the T2I-Adapter, Sketch-Search Agent combines sketches and text prompts to generate high-quality query images, encoded via a CLIP image encoder for efficient matching against an image corpus. Unlike existing methods, Sketch-Search Agent requires minimal setup, no additional training, and excels in sketch-based image retrieval and natural language interactions. The multimodal agent enhances user experience by dynamically retaining preferences, ranking results, and refining queries for personalized recommendations. This interactive design empowers users to create sketches and receive tailored product suggestions, showcasing the potential of diffusion models in user-centric image retrieval. Experiments confirm Sketch-Search Agent's high accuracy in delivering relevant product search results. 

---
