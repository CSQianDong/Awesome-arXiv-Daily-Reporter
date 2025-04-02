# Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic Resources 

**Authors**: Weizhi Wang, Yu Tian, Linjie Yang, Heng Wang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00595)  

**Abstract**: The reproduction of state-of-the-art multimodal LLM pre-training faces barriers at every stage of the pipeline, including high-quality data filtering, multimodal data mixture strategies, sequence packing techniques, and training frameworks. We introduce Open-Qwen2VL, a fully open-source 2B-parameter Multimodal Large Language Model pre-trained efficiently on 29M image-text pairs using only 442 A100-40G GPU hours. Our approach employs low-to-high dynamic image resolution and multimodal sequence packing to significantly enhance pre-training efficiency. The training dataset was carefully curated using both MLLM-based filtering techniques (e.g., MLM-Filter) and conventional CLIP-based filtering methods, substantially improving data quality and training efficiency. The Open-Qwen2VL pre-training is conducted on academic level 8xA100-40G GPUs at UCSB on 5B packed multimodal tokens, which is 0.36\% of 1.4T multimodal pre-training tokens of Qwen2-VL. The final instruction-tuned Open-Qwen2VL outperforms partially-open state-of-the-art MLLM Qwen2-VL-2B on various multimodal benchmarks of MMBench, SEEDBench, MMstar, and MathVista, indicating the remarkable training efficiency of Open-Qwen2VL. We open-source all aspects of our work, including compute-efficient and data-efficient training details, data filtering methods, sequence packing scripts, pre-training data in WebDataset format, FSDP-based training codebase, and both base and instruction-tuned model checkpoints. We redefine "fully open" for multimodal LLMs as the complete release of: 1) the training codebase, 2) detailed data filtering techniques, and 3) all pre-training and supervised fine-tuning data used to develop the model. 

---
# Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents 

**Authors**: Gavin Greif, Niclas Griesshaber, Robin Greif  

**Link**: [PDF](https://arxiv.org/pdf/2504.00414)  

**Abstract**: We explore how multimodal Large Language Models (mLLMs) can help researchers transcribe historical documents, extract relevant historical information, and construct datasets from historical sources. Specifically, we investigate the capabilities of mLLMs in performing (1) Optical Character Recognition (OCR), (2) OCR Post-Correction, and (3) Named Entity Recognition (NER) tasks on a set of city directories published in German between 1754 and 1870. First, we benchmark the off-the-shelf transcription accuracy of both mLLMs and conventional OCR models. We find that the best-performing mLLM model significantly outperforms conventional state-of-the-art OCR models and other frontier mLLMs. Second, we are the first to introduce multimodal post-correction of OCR output using mLLMs. We find that this novel approach leads to a drastic improvement in transcription accuracy and consistently produces highly accurate transcriptions (<1% CER), without any image pre-processing or model fine-tuning. Third, we demonstrate that mLLMs can efficiently recognize entities in transcriptions of historical documents and parse them into structured dataset formats. Our findings provide early evidence for the long-term potential of mLLMs to introduce a paradigm shift in the approaches to historical data collection and document transcription. 

---
# CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation 

**Authors**: Jixuan Leng, Chengsong Huang, Langlin Huang, Bill Yuchen Lin, William W. Cohen, Haohan Wang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00043)  

**Abstract**: Existing reasoning evaluation frameworks for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) predominantly either assess text-based reasoning or vision-language understanding capabilities, with limited dynamic interplay between textual and visual constraints. To address this limitation, we introduce CrossWordBench, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs through the medium of crossword puzzles-a task requiring multimodal adherence to semantic constraints from text-based clues and intersectional constraints from visual grid structures. CrossWordBench leverages a controllable puzzle generation framework that produces puzzles in multiple formats (text and image) and offers different evaluation strategies ranging from direct puzzle solving to interactive modes. Our extensive evaluation of over 20 models reveals that reasoning LLMs outperform non-reasoning models substantially by effectively leveraging crossing-letter constraints. We further demonstrate that LVLMs struggle with the task, showing a strong correlation between their puzzle-solving performance and grid-parsing accuracy. Our findings offer insights into the limitations of the reasoning capabilities of current LLMs and LVLMs, and provide an effective approach for creating multimodal constrained tasks for future evaluations. 

---
# ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers 

**Authors**: Qianhao Yuan, Qingyu Zhang, Yanjiang Liu, Jiawei Chen, Yaojie Lu, Hongyu Lin, Jia Zheng, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.00502)  

**Abstract**: Multimodal Large Language Models (MLLMs) suffer from high computational costs due to their massive size and the large number of visual tokens. In this paper, we investigate layer-wise redundancy in MLLMs by introducing a novel metric, Layer Contribution (LC), which quantifies the impact of a layer's transformations on visual and text tokens, respectively. The calculation of LC involves measuring the divergence in model output that results from removing the layer's transformations on the specified tokens. Our pilot experiment reveals that many layers of MLLMs exhibit minimal contribution during the processing of visual tokens. Motivated by this observation, we propose ShortV, a training-free method that leverages LC to identify ineffective layers, and freezes visual token updates in these layers. Experiments show that ShortV can freeze visual token in approximately 60\% of the MLLM layers, thereby dramatically reducing computational costs related to updating visual tokens. For example, it achieves a 50\% reduction in FLOPs on LLaVA-NeXT-13B while maintaining superior performance. The code will be publicly available at this https URL 

---
# WikiVideo: Article Generation from Multiple Videos 

**Authors**: Alexander Martin, Reno Kriz, William Gantt Walden, Kate Sanders, Hannah Recknor, Eugene Yang, Francis Ferraro, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2504.00939)  

**Abstract**: We present the challenging task of automatically creating a high-level Wikipedia-style article that aggregates information from multiple diverse videos about real-world events, such as natural disasters or political elections. Videos are intuitive sources for retrieval-augmented generation (RAG), but most contemporary RAG workflows focus heavily on text and existing methods for video-based summarization focus on low-level scene understanding rather than high-level event semantics. To close this gap, we introduce WikiVideo, a benchmark consisting of expert-written articles and densely annotated videos that provide evidence for articles' claims, facilitating the integration of video into RAG pipelines and enabling the creation of in-depth content that is grounded in multimodal sources. We further propose Collaborative Article Generation (CAG), a novel interactive method for article creation from multiple videos. CAG leverages an iterative interaction between an r1-style reasoning model and a VideoLLM to draw higher level inferences about the target event than is possible with VideoLLMs alone, which fixate on low-level visual features. We benchmark state-of-the-art VideoLLMs and CAG in both oracle retrieval and RAG settings and find that CAG consistently outperforms alternative methods, while suggesting intriguing avenues for future work. 

---
# FortisAVQA and MAVEN: a Benchmark Dataset and Debiasing Framework for Robust Multimodal Reasoning 

**Authors**: Jie Ma, Zhitao Gao, Qi Chai, Jun Liu, Pinghui Wang, Jing Tao, Zhou Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.00487)  

**Abstract**: Audio-Visual Question Answering (AVQA) is a challenging multimodal reasoning task requiring intelligent systems to answer natural language queries based on paired audio-video inputs accurately. However, existing AVQA approaches often suffer from overfitting to dataset biases, leading to poor robustness. Moreover, current datasets may not effectively diagnose these methods. To address these challenges, we first introduce a novel dataset, FortisAVQA, constructed in two stages: (1) rephrasing questions in the test split of the public MUSIC-AVQA dataset and (2) introducing distribution shifts across questions. The first stage expands the test space with greater diversity, while the second enables a refined robustness evaluation across rare, frequent, and overall question distributions. Second, we introduce a robust Multimodal Audio-Visual Epistemic Network (MAVEN) that leverages a multifaceted cycle collaborative debiasing strategy to mitigate bias learning. Experimental results demonstrate that our architecture achieves state-of-the-art performance on FortisAVQA, with a notable improvement of 7.81\%. Extensive ablation studies on both datasets validate the effectiveness of our debiasing components. Additionally, our evaluation reveals the limited robustness of existing multimodal QA methods. We also verify the plug-and-play capability of our strategy by integrating it with various baseline models across both datasets. Our dataset and code are available at this https URL. 

---
# Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning 

**Authors**: Ram Ramrakhya, Matthew Chang, Xavier Puig, Ruta Desai, Zsolt Kira, Roozbeh Mottaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00907)  

**Abstract**: Embodied agents operating in real-world environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent must fetch a specific object instance given an ambiguous instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To solve this problem, we propose a novel approach that fine-tunes multimodal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines, including GPT-4o, and supervised fine-tuned MLLMs, on our task. Our results demonstrate that our RL-finetuned MLLM outperforms all baselines by a significant margin ($19.1$-$40.3\%$), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL. 

---
# AI Judges in Design: Statistical Perspectives on Achieving Human Expert Equivalence With Vision-Language Models 

**Authors**: Kristen M. Edwards, Farnaz Tehranchi, Scarlett R. Miller, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2504.00938)  

**Abstract**: The subjective evaluation of early stage engineering designs, such as conceptual sketches, traditionally relies on human experts. However, expert evaluations are time-consuming, expensive, and sometimes inconsistent. Recent advances in vision-language models (VLMs) offer the potential to automate design assessments, but it is crucial to ensure that these AI ``judges'' perform on par with human experts. However, no existing framework assesses expert equivalence. This paper introduces a rigorous statistical framework to determine whether an AI judge's ratings match those of human experts. We apply this framework in a case study evaluating four VLM-based judges on key design metrics (uniqueness, creativity, usefulness, and drawing quality). These AI judges employ various in-context learning (ICL) techniques, including uni- vs. multimodal prompts and inference-time reasoning. The same statistical framework is used to assess three trained novices for expert-equivalence. Results show that the top-performing AI judge, using text- and image-based ICL with reasoning, achieves expert-level agreement for uniqueness and drawing quality and outperforms or matches trained novices across all metrics. In 6/6 runs for both uniqueness and creativity, and 5/6 runs for both drawing quality and usefulness, its agreement with experts meets or exceeds that of the majority of trained novices. These findings suggest that reasoning-supported VLM models can achieve human-expert equivalence in design evaluation. This has implications for scaling design evaluation in education and practice, and provides a general statistical framework for validating AI judges in other domains requiring subjective content evaluation. 

---
# IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval 

**Authors**: Bangwei Liu, Yicheng Bao, Shaohui Lin, Xuhong Wang, Xin Tan, Yingchun Wang, Yuan Xie, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00954)  

**Abstract**: Multimodal retrieval systems are becoming increasingly vital for cutting-edge AI technologies, such as embodied AI and AI-driven digital content industries. However, current multimodal retrieval tasks lack sufficient complexity and demonstrate limited practical application value. It spires us to design Instance-Driven Multimodal Image Retrieval (IDMR), a novel task that requires models to retrieve images containing the same instance as a query image while matching a text-described scenario. Unlike existing retrieval tasks focused on global image similarity or category-level matching, IDMR demands fine-grained instance-level consistency across diverse contexts. To benchmark this capability, we develop IDMR-bench using real-world object tracking and first-person video data. Addressing the scarcity of training data, we propose a cross-domain synthesis method that creates 557K training samples by cropping objects from standard detection datasets. Our Multimodal Large Language Model (MLLM) based retrieval model, trained on 1.2M samples, outperforms state-of-the-art approaches on both traditional benchmarks and our zero-shot IDMR-bench. Experimental results demonstrate previous models' limitations in instance-aware retrieval and highlight the potential of MLLM for advanced retrieval applications. The whole training dataset, codes and models, with wide ranges of sizes, are available at this https URL. 

---
# A Survey on Music Generation from Single-Modal, Cross-Modal, and Multi-Modal Perspectives: Data, Methods, and Challenges 

**Authors**: Shuyu Li, Shulei Ji, Zihao Wang, Songruoyao Wu, Jiaxing Yu, Kejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00837)  

**Abstract**: Multi-modal music generation, using multiple modalities like images, video, and text alongside musical scores and audio as guidance, is an emerging research area with broad applications. This paper reviews this field, categorizing music generation systems from the perspective of modalities. It covers modality representation, multi-modal data alignment, and their utilization to guide music generation. We also discuss current datasets and evaluation methods. Key challenges in this area include effective multi-modal integration, large-scale comprehensive datasets, and systematic evaluation methods. Finally, we provide an outlook on future research directions focusing on multi-modal fusion, alignment, data, and evaluation. 

---
# Context-Aware Human Behavior Prediction Using Multimodal Large Language Models: Challenges and Insights 

**Authors**: Yuchen Liu, Lino Lerch, Luigi Palmieri, Andrey Rudenko, Sebastian Koch, Timo Ropinski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2504.00839)  

**Abstract**: Predicting human behavior in shared environments is crucial for safe and efficient human-robot interaction. Traditional data-driven methods to that end are pre-trained on domain-specific datasets, activity types, and prediction horizons. In contrast, the recent breakthroughs in Large Language Models (LLMs) promise open-ended cross-domain generalization to describe various human activities and make predictions in any context. In particular, Multimodal LLMs (MLLMs) are able to integrate information from various sources, achieving more contextual awareness and improved scene understanding. The difficulty in applying general-purpose MLLMs directly for prediction stems from their limited capacity for processing large input sequences, sensitivity to prompt design, and expensive fine-tuning. In this paper, we present a systematic analysis of applying pre-trained MLLMs for context-aware human behavior prediction. To this end, we introduce a modular multimodal human activity prediction framework that allows us to benchmark various MLLMs, input variations, In-Context Learning (ICL), and autoregressive techniques. Our evaluation indicates that the best-performing framework configuration is able to reach 92.8% semantic similarity and 66.1% exact label accuracy in predicting human behaviors in the target frame. 

---
# Distilling Multi-view Diffusion Models into 3D Generators 

**Authors**: Hao Qin, Luyuan Chen, Ming Kong, Mengxu Lu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00457)  

**Abstract**: We introduce DD3G, a formulation that Distills a multi-view Diffusion model (MV-DM) into a 3D Generator using gaussian splatting. DD3G compresses and integrates extensive visual and spatial geometric knowledge from the MV-DM by simulating its ordinary differential equation (ODE) trajectory, ensuring the distilled generator generalizes better than those trained solely on 3D data. Unlike previous amortized optimization approaches, we align the MV-DM and 3D generator representation spaces to transfer the teacher's probabilistic flow to the student, thus avoiding inconsistencies in optimization objectives caused by probabilistic sampling. The introduction of probabilistic flow and the coupling of various attributes in 3D Gaussians introduce challenges in the generation process. To tackle this, we propose PEPD, a generator consisting of Pattern Extraction and Progressive Decoding phases, which enables efficient fusion of probabilistic flow and converts a single image into 3D Gaussians within 0.06 seconds. Furthermore, to reduce knowledge loss and overcome sparse-view supervision, we design a joint optimization objective that ensures the quality of generated samples through explicit supervision and implicit verification. Leveraging existing 2D generation models, we compile 120k high-quality RGBA images for distillation. Experiments on synthetic and public datasets demonstrate the effectiveness of our method. Our project is available at: this https URL 

---
# Agentic Multimodal AI for Hyperpersonalized B2B and B2C Advertising in Competitive Markets: An AI-Driven Competitive Advertising Framework 

**Authors**: Sakhinana Sagar Srinivas, Akash Das, Shivam Gupta, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2504.00338)  

**Abstract**: The growing use of foundation models (FMs) in real-world applications demands adaptive, reliable, and efficient strategies for dynamic markets. In the chemical industry, AI-discovered materials drive innovation, but commercial success hinges on market adoption, requiring FM-driven advertising frameworks that operate in-the-wild. We present a multilingual, multimodal AI framework for autonomous, hyper-personalized advertising in B2B and B2C markets. By integrating retrieval-augmented generation (RAG), multimodal reasoning, and adaptive persona-based targeting, our system generates culturally relevant, market-aware ads tailored to shifting consumer behaviors and competition. Validation combines real-world product experiments with a Simulated Humanistic Colony of Agents to model consumer personas, optimize strategies at scale, and ensure privacy compliance. Synthetic experiments mirror real-world scenarios, enabling cost-effective testing of ad strategies without risky A/B tests. Combining structured retrieval-augmented reasoning with in-context learning (ICL), the framework boosts engagement, prevents market cannibalization, and maximizes ROAS. This work bridges AI-driven innovation and market adoption, advancing multimodal FM deployment for high-stakes decision-making in commercial marketing. 

---
# GazeLLM: Multimodal LLMs incorporating Human Visual Attention 

**Authors**: Jun Rekimoto  

**Link**: [PDF](https://arxiv.org/pdf/2504.00221)  

**Abstract**: Large Language Models (LLMs) are advancing into Multimodal LLMs (MLLMs), capable of processing image, audio, and video as well as text. Combining first-person video, MLLMs show promising potential for understanding human activities through video and audio, enabling many human-computer interaction and human-augmentation applications such as human activity support, real-world agents, and skill transfer to robots or other individuals. However, handling high-resolution, long-duration videos generates large latent representations, leading to substantial memory and processing demands, limiting the length and resolution MLLMs can manage. Reducing video resolution can lower memory usage but often compromises comprehension. This paper introduces a method that optimizes first-person video analysis by integrating eye-tracking data, and proposes a method that decomposes first-person vision video into sub areas for regions of gaze focus. By processing these selectively gazed-focused inputs, our approach achieves task comprehension equivalent to or even better than processing the entire image at full resolution, but with significantly reduced video data input (reduce the number of pixels to one-tenth), offering an efficient solution for using MLLMs to interpret and utilize human skills. 

---
