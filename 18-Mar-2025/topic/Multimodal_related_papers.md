# HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model 

**Authors**: Haiyang Guo, Fanhu Zeng, Ziwei Xiang, Fei Zhu, Da-Han Wang, Xu-Yao Zhang, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12941)  

**Abstract**: Instruction tuning is widely used to improve a pre-trained Multimodal Large Language Model (MLLM) by training it on curated task-specific datasets, enabling better comprehension of human instructions. However, it is infeasible to collect all possible instruction datasets simultaneously in real-world scenarios. Thus, enabling MLLM with continual instruction tuning is essential for maintaining their adaptability. However, existing methods often trade off memory efficiency for performance gains, significantly compromising overall efficiency. In this paper, we propose a task-specific expansion and task-general fusion framework based on the variations in Centered Kernel Alignment (CKA) similarity across different model layers when trained on diverse datasets. Furthermore, we analyze the information leakage present in the existing benchmark and propose a new and more challenging benchmark to rationally evaluate the performance of different methods. Comprehensive experiments showcase a significant performance improvement of our method compared to existing state-of-the-art methods. Our code will be public available. 

---
# Multi-Granular Multimodal Clue Fusion for Meme Understanding 

**Authors**: Li Zheng, Hao Fei, Ting Dai, Zuquan Peng, Fei Li, Huisheng Ma, Chong Teng, Donghong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.12560)  

**Abstract**: With the continuous emergence of various social media platforms frequently used in daily life, the multimodal meme understanding (MMU) task has been garnering increasing attention. MMU aims to explore and comprehend the meanings of memes from various perspectives by performing tasks such as metaphor recognition, sentiment analysis, intention detection, and offensiveness detection. Despite making progress, limitations persist due to the loss of fine-grained metaphorical visual clue and the neglect of multimodal text-image weak correlation. To overcome these limitations, we propose a multi-granular multimodal clue fusion model (MGMCF) to advance MMU. Firstly, we design an object-level semantic mining module to extract object-level image feature clues, achieving fine-grained feature clue extraction and enhancing the model's ability to capture metaphorical details and semantics. Secondly, we propose a brand-new global-local cross-modal interaction model to address the weak correlation between text and images. This model facilitates effective interaction between global multimodal contextual clues and local unimodal feature clues, strengthening their representations through a bidirectional cross-modal attention mechanism. Finally, we devise a dual-semantic guided training strategy to enhance the model's understanding and alignment of multimodal representations in the semantic space. Experiments conducted on the widely-used MET-MEME bilingual dataset demonstrate significant improvements over state-of-the-art baselines. Specifically, there is an 8.14% increase in precision for offensiveness detection task, and respective accuracy enhancements of 3.53%, 3.89%, and 3.52% for metaphor recognition, sentiment analysis, and intention detection tasks. These results, underpinned by in-depth analyses, underscore the effectiveness and potential of our approach for advancing MMU. 

---
# Seeing Sarcasm Through Different Eyes: Analyzing Multimodal Sarcasm Perception in Large Vision-Language Models 

**Authors**: Junjie Chen, Xuyang Liu, Subin Huang, Linfeng Zhang, Hang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12149)  

**Abstract**: With the advent of large vision-language models (LVLMs) demonstrating increasingly human-like abilities, a pivotal question emerges: do different LVLMs interpret multimodal sarcasm differently, and can a single model grasp sarcasm from multiple perspectives like humans? To explore this, we introduce an analytical framework using systematically designed prompts on existing multimodal sarcasm datasets. Evaluating 12 state-of-the-art LVLMs over 2,409 samples, we examine interpretive variations within and across models, focusing on confidence levels, alignment with dataset labels, and recognition of ambiguous "neutral" cases. Our findings reveal notable discrepancies -- across LVLMs and within the same model under varied prompts. While classification-oriented prompts yield higher internal consistency, models diverge markedly when tasked with interpretive reasoning. These results challenge binary labeling paradigms by highlighting sarcasm's subjectivity. We advocate moving beyond rigid annotation schemes toward multi-perspective, uncertainty-aware modeling, offering deeper insights into multimodal sarcasm comprehension. Our code and data are available at: this https URL 

---
# R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization 

**Authors**: Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12937)  

**Abstract**: Recent studies generally enhance MLLMs' reasoning capabilities via supervised fine-tuning on high-quality chain-of-thought reasoning data, which often leads models to merely imitate successful reasoning paths without understanding what the wrong reasoning paths are. In this work, we aim to enhance the MLLMs' reasoning ability beyond passively imitating positive reasoning paths. To this end, we design Step-wise Group Relative Policy Optimization (StepGRPO), a new online reinforcement learning framework that enables MLLMs to self-improve reasoning ability via simple, effective and dense step-wise rewarding. Specifically, StepGRPO introduces two novel rule-based reasoning rewards: Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR). StepRAR rewards the reasoning paths that contain necessary intermediate reasoning steps via a soft key-step matching technique, while StepRAR rewards reasoning paths that follow a well-structured and logically consistent reasoning process through a reasoning completeness and logic evaluation strategy. With the proposed StepGRPO, we introduce R1-VL, a series of MLLMs with outstanding capabilities in step-by-step reasoning. Extensive experiments over 8 benchmarks demonstrate the superiority of our methods. 

---
# MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs 

**Authors**: Erik Daxberger, Nina Wenzel, David Griffiths, Haiming Gang, Justin Lazarow, Gefen Kohavi, Kai Kang, Marcin Eichner, Yinfei Yang, Afshin Dehghan, Peter Grasch  

**Link**: [PDF](https://arxiv.org/pdf/2503.13111)  

**Abstract**: Multimodal large language models (MLLMs) excel at 2D visual understanding but remain limited in their ability to reason about 3D space. In this work, we leverage large-scale high-quality 3D scene data with open-set annotations to introduce 1) a novel supervised fine-tuning dataset and 2) a new evaluation benchmark, focused on indoor scenes. Our Cubify Anything VQA (CA-VQA) data covers diverse spatial tasks including spatial relationship prediction, metric size and distance estimation, and 3D grounding. We show that CA-VQA enables us to train MM-Spatial, a strong generalist MLLM that also achieves state-of-the-art performance on 3D spatial understanding benchmarks, including our own. We show how incorporating metric depth and multi-view inputs (provided in CA-VQA) can further improve 3D understanding, and demonstrate that data alone allows our model to achieve depth perception capabilities comparable to dedicated monocular depth estimation models. We will publish our SFT dataset and benchmark. 

---
# MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research 

**Authors**: James Burgess, Jeffrey J Nirschl, Laura Bravo-Sánchez, Alejandro Lozano, Sanket Rajan Gupte, Jesus G. Galaz-Montoya, Yuhui Zhang, Yuchang Su, Disha Bhowmik, Zachary Coman, Sarina M. Hasan, Alexandra Johannesson, William D. Leineweber, Malvika G Nair, Ridhi Yarlagadda, Connor Zuraski, Wah Chiu, Sarah Cohen, Jan N. Hansen, Manuel D Leonetti, Chad Liu, Emma Lundberg, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2503.13399)  

**Abstract**: Scientific research demands sophisticated reasoning over multimodal data, a challenge especially prevalent in biology. Despite recent advances in multimodal large language models (MLLMs) for AI-assisted research, existing multimodal reasoning benchmarks only target up to college-level difficulty, while research-level benchmarks emphasize lower-level perception, falling short of the complex multimodal reasoning needed for scientific discovery. To bridge this gap, we introduce MicroVQA, a visual-question answering (VQA) benchmark designed to assess three reasoning capabilities vital in research workflows: expert image understanding, hypothesis generation, and experiment proposal. MicroVQA consists of 1,042 multiple-choice questions (MCQs) curated by biology experts across diverse microscopy modalities, ensuring VQA samples represent real scientific practice. In constructing the benchmark, we find that standard MCQ generation methods induce language shortcuts, motivating a new two-stage pipeline: an optimized LLM prompt structures question-answer pairs into MCQs; then, an agent-based `RefineBot' updates them to remove shortcuts. Benchmarking on state-of-the-art MLLMs reveal a peak performance of 53\%; models with smaller LLMs only slightly underperform top models, suggesting that language-based reasoning is less challenging than multimodal reasoning; and tuning with scientific articles enhances performance. Expert analysis of chain-of-thought responses shows that perception errors are the most frequent, followed by knowledge errors and then overgeneralization errors. These insights highlight the challenges in multimodal scientific reasoning, showing MicroVQA is a valuable resource advancing AI-driven biomedical research. MicroVQA is available at this https URL, and project page at this https URL. 

---
# Logic-RAG: Augmenting Large Multimodal Models with Visual-Spatial Knowledge for Road Scene Understanding 

**Authors**: Imran Kabir, Md Alimoor Reza, Syed Billah  

**Link**: [PDF](https://arxiv.org/pdf/2503.12663)  

**Abstract**: Large multimodal models (LMMs) are increasingly integrated into autonomous driving systems for user interaction. However, their limitations in fine-grained spatial reasoning pose challenges for system interpretability and user trust. We introduce Logic-RAG, a novel Retrieval-Augmented Generation (RAG) framework that improves LMMs' spatial understanding in driving scenarios. Logic-RAG constructs a dynamic knowledge base (KB) about object-object relationships in first-order logic (FOL) using a perception module, a query-to-logic embedder, and a logical inference engine. We evaluated Logic-RAG on visual-spatial queries using both synthetic and real-world driving videos. When using popular LMMs (GPT-4V, Claude 3.5) as proxies for an autonomous driving system, these models achieved only 55% accuracy on synthetic driving scenes and under 75% on real-world driving scenes. Augmenting them with Logic-RAG increased their accuracies to over 80% and 90%, respectively. An ablation study showed that even without logical inference, the fact-based context constructed by Logic-RAG alone improved accuracy by 15%. Logic-RAG is extensible: it allows seamless replacement of individual components with improved versions and enables domain experts to compose new knowledge in both FOL and natural language. In sum, Logic-RAG addresses critical spatial reasoning deficiencies in LMMs for autonomous driving applications. Code and data are available at this https URL. 

---
# Hyperbolic Safety-Aware Vision-Language Models 

**Authors**: Tobia Poppi, Tejaswi Kasarla, Pascal Mettes, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2503.12127)  

**Abstract**: Addressing the retrieval of unsafe content from vision-language models such as CLIP is an important step towards real-world integration. Current efforts have relied on unlearning techniques that try to erase the model's knowledge of unsafe concepts. While effective in reducing unwanted outputs, unlearning limits the model's capacity to discern between safe and unsafe content. In this work, we introduce a novel approach that shifts from unlearning to an awareness paradigm by leveraging the inherent hierarchical properties of the hyperbolic space. We propose to encode safe and unsafe content as an entailment hierarchy, where both are placed in different regions of hyperbolic space. Our HySAC, Hyperbolic Safety-Aware CLIP, employs entailment loss functions to model the hierarchical and asymmetrical relations between safe and unsafe image-text pairs. This modelling, ineffective in standard vision-language models due to their reliance on Euclidean embeddings, endows the model with awareness of unsafe content, enabling it to serve as both a multimodal unsafe classifier and a flexible content retriever, with the option to dynamically redirect unsafe queries toward safer alternatives or retain the original output. Extensive experiments show that our approach not only enhances safety recognition but also establishes a more adaptable and interpretable framework for content moderation in vision-language models. Our source code is available at this https URL. 

---
# Semantic-Clipping: Efficient Vision-Language Modeling with Semantic-Guidedd Visual Selection 

**Authors**: Bangzheng Li, Fei Wang, Wenxuan Zhou, Nan Xu, Ben Zhou, Sheng Zhang, Hoifung Poon, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11794)  

**Abstract**: Vision-Language Models (VLMs) leverage aligned visual encoders to transform images into visual tokens, allowing them to be processed similarly to text by the backbone large language model (LLM). This unified input paradigm enables VLMs to excel in vision-language tasks such as visual question answering (VQA). To improve fine-grained visual reasoning, recent advancements in vision-language modeling introduce image cropping techniques that feed all encoded sub-images into the model. However, this approach significantly increases the number of visual tokens, leading to inefficiency and potential distractions for the LLM. To address the generalization challenges of image representation in VLMs, we propose a lightweight, universal framework that seamlessly integrates with existing VLMs to enhance their ability to process finegrained details. Our method leverages textual semantics to identify key visual areas, improving VQA performance without requiring any retraining of the VLM. Additionally, it incorporates textual signals into the visual encoding process, enhancing both efficiency and effectiveness. The proposed method, SEMCLIP, strengthens the visual understanding of a 7B VLM, LLaVA-1.5 by 3.3% on average across 7 benchmarks, and particularly by 5.3% on the challenging detailed understanding benchmark V*. 

---
# AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding 

**Authors**: Xiao Wang, Qingyi Si, Jianlong Wu, Shiyu Zhu, Li Cao, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.12559)  

**Abstract**: Multimodal Large Language Models (MLLMs) have revolutionized video understanding, yet are still limited by context length when processing long videos. Recent methods compress videos by leveraging visual redundancy uniformly, yielding promising results. Nevertheless, our quantitative analysis shows that redundancy varies significantly across time and model layers, necessitating a more flexible compression strategy. We propose AdaReTaKe, a training-free method that flexibly reduces visual redundancy by allocating compression ratios among time and layers with theoretical guarantees. Integrated into state-of-the-art MLLMs, AdaReTaKe improves processing capacity from 256 to 2048 frames while preserving critical information. Experiments on VideoMME, MLVU, LongVideoBench, and LVBench datasets demonstrate that AdaReTaKe outperforms existing methods by 2.3% and 2.8% for 7B and 72B models, respectively, with even greater improvements of 5.9% and 6.0% on the longest LVBench. Our code is available at this https URL. 

---
# DeepPerception: Advancing R1-like Cognitive Visual Perception in MLLMs for Knowledge-Intensive Visual Grounding 

**Authors**: Xinyu Ma, Ziyang Ding, Zhicong Luo, Chi Chen, Zonghao Guo, Derek F. Wong, Xiaoyi Feng, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.12797)  

**Abstract**: Human experts excel at fine-grained visual discrimination by leveraging domain knowledge to refine perceptual features, a capability that remains underdeveloped in current Multimodal Large Language Models (MLLMs). Despite possessing vast expert-level knowledge, MLLMs struggle to integrate reasoning into visual perception, often generating direct responses without deeper analysis. To bridge this gap, we introduce knowledge-intensive visual grounding (KVG), a novel visual grounding task that requires both fine-grained perception and domain-specific knowledge integration. To address the challenges of KVG, we propose DeepPerception, an MLLM enhanced with cognitive visual perception capabilities. Our approach consists of (1) an automated data synthesis pipeline that generates high-quality, knowledge-aligned training samples, and (2) a two-stage training framework combining supervised fine-tuning for cognitive reasoning scaffolding and reinforcement learning to optimize perception-cognition synergy. To benchmark performance, we introduce KVG-Bench a comprehensive dataset spanning 10 domains with 1.3K manually curated test cases. Experimental results demonstrate that DeepPerception significantly outperforms direct fine-tuning, achieving +8.08\% accuracy improvements on KVG-Bench and exhibiting +4.60\% superior cross-domain generalization over baseline approaches. Our findings highlight the importance of integrating cognitive processes into MLLMs for human-like visual perception and open new directions for multimodal reasoning research. The data, codes, and models are released at this https URL. 

---
# MPBench: A Comprehensive Multimodal Reasoning Benchmark for Process Errors Identification 

**Authors**: Zhaopan Xu, Pengfei Zhou, Jiaxin Ai, Wangbo Zhao, Kai Wang, Xiaojiang Peng, Wenqi Shao, Hongxun Yao, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12505)  

**Abstract**: Reasoning is an essential capacity for large language models (LLMs) to address complex tasks, where the identification of process errors is vital for improving this ability. Recently, process-level reward models (PRMs) were proposed to provide step-wise rewards that facilitate reinforcement learning and data production during training and guide LLMs toward correct steps during inference, thereby improving reasoning accuracy. However, existing benchmarks of PRMs are text-based and focus on error detection, neglecting other scenarios like reasoning search. To address this gap, we introduce MPBench, a comprehensive, multi-task, multimodal benchmark designed to systematically assess the effectiveness of PRMs in diverse scenarios. MPBench employs three evaluation paradigms, each targeting a specific role of PRMs in the reasoning process: (1) Step Correctness, which assesses the correctness of each intermediate reasoning step; (2) Answer Aggregation, which aggregates multiple solutions and selects the best one; and (3) Reasoning Process Search, which guides the search for optimal reasoning steps during inference. Through these paradigms, MPBench makes comprehensive evaluations and provides insights into the development of multimodal PRMs. 

---
# Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning 

**Authors**: Hai-Long Sun, Zhun Sun, Houwen Peng, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.13360)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated enhanced reasoning capabilities, evolving from Chain-of-Thought (CoT) prompting to advanced, product-oriented solutions like OpenAI o1. During our re-implementation of this model, we noticed that in multimodal tasks requiring visual input (e.g., geometry problems), Multimodal LLMs (MLLMs) struggle to maintain focus on the visual information, in other words, MLLMs suffer from a gradual decline in attention to visual information as reasoning progresses, causing text-over-relied outputs. To investigate this, we ablate image inputs during long-chain reasoning. Concretely, we truncate the reasoning process midway, then re-complete the reasoning process with the input image removed. We observe only a ~2% accuracy drop on MathVista's test-hard subset, revealing the model's textual outputs dominate the following reasoning process. Motivated by this, we propose Take-along Visual Conditioning (TVC), a strategy that shifts image input to critical reasoning stages and compresses redundant visual tokens via dynamic pruning. This methodology helps the model retain attention to the visual components throughout the reasoning. Our approach achieves state-of-the-art performance on average across five mathematical reasoning benchmarks (+3.4% vs previous sota), demonstrating the effectiveness of TVC in enhancing multimodal reasoning systems. 

---
# Cream of the Crop: Harvesting Rich, Scalable and Transferable Multi-Modal Data for Instruction Fine-Tuning 

**Authors**: Mengyao Lyu, Yan Li, Huasong Zhong, Wenhao Yang, Hui Chen, Jungong Han, Guiguang Ding, Zhenheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13383)  

**Abstract**: The hypothesis that pretrained large language models (LLMs) necessitate only minimal supervision during the fine-tuning (SFT) stage (Zhou et al., 2024) has been substantiated by recent advancements in data curation and selection research. However, their stability and generalizability are compromised due to the vulnerability to experimental setups and validation protocols, falling short of surpassing random sampling (Diddee & Ippolito, 2024; Xia et al., 2024b). Built upon LLMs, multi-modal LLMs (MLLMs), combined with the sheer token volume and heightened heterogeneity of data sources, amplify both the significance and complexity of data selection.
To harvest multi-modal instructional data in a robust and efficient manner, we re-define the granularity of the quality metric by decomposing it into 14 vision-language-related capabilities, and introduce multi-modal rich scorers to evaluate the capabilities of each data candidate. To promote diversity, in light of the inherent objective of the alignment stage, we take interaction style as diversity indicator and use a multi-modal rich styler to identify data instruction patterns. In doing so, our multi-modal rich scorers and styler (mmSSR) guarantee that high-scoring information is conveyed to users in diversified forms. Free from embedding-based clustering or greedy sampling, mmSSR efficiently scales to millions of data with varying budget constraints, supports customization for general or specific capability acquisition, and facilitates training-free generalization to new domains for curation. Across 10+ experimental settings, validated by 14 multi-modal benchmarks, we demonstrate consistent improvements over random sampling, baseline strategies and state-of-the-art selection methods, achieving 99.1% of full performance with only 30% of the 2.6M data. 

---
# Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-tuning 

**Authors**: Yiwei Chen, Yuguang Yao, Yihua Zhang, Bingquan Shen, Gaowen Liu, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11832)  

**Abstract**: Recent vision-language models (VLMs) have made remarkable strides in generative modeling with multimodal inputs, particularly text and images. However, their susceptibility to generating harmful content when exposed to unsafe queries raises critical safety concerns. While current alignment strategies primarily rely on supervised safety fine-tuning with curated datasets, we identify a fundamental limitation we call the "safety mirage" where supervised fine-tuning inadvertently reinforces spurious correlations between superficial textual patterns and safety responses, rather than fostering deep, intrinsic mitigation of harm. We show that these spurious correlations leave fine-tuned VLMs vulnerable even to a simple one-word modification-based attack, where substituting a single word in text queries with a spurious correlation-inducing alternative can effectively bypass safeguards. Additionally, these correlations contribute to the over prudence, causing fine-tuned VLMs to refuse benign queries unnecessarily. To address this issue, we show machine unlearning (MU) as a powerful alternative to supervised safety fine-tuning as it avoids biased feature-label mappings and directly removes harmful knowledge from VLMs while preserving their general capabilities. Extensive evaluations across safety benchmarks show that under one-word attacks, MU-based alignment reduces the attack success rate by up to 60.17% and cuts unnecessary rejections by over 84.20%. Codes are available at this https URL. WARNING: There exist AI generations that may be offensive in nature. 

---
# ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation in Multimodal Large language Models 

**Authors**: Hao Yin, Guangzong Si, Zilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13107)  

**Abstract**: Contrastive decoding strategies are widely used to mitigate object hallucinations in multimodal large language models (MLLMs). By reducing over-reliance on language priors, these strategies ensure that generated content remains closely grounded in visual inputs, producing contextually accurate outputs. Since contrastive decoding requires no additional training or external tools, it offers both computational efficiency and versatility, making it highly attractive. However, these methods present two main limitations: (1) bluntly suppressing language priors can compromise coherence and accuracy of generated content, and (2) processing contrastive inputs adds computational load, significantly slowing inference speed. To address these challenges, we propose Visual Amplification Fusion (VAF), a plug-and-play technique that enhances attention to visual signals within the model's middle layers, where modality fusion predominantly occurs. This approach enables more effective capture of visual features, reducing the model's bias toward language modality. Experimental results demonstrate that VAF significantly reduces hallucinations across various MLLMs without affecting inference speed, while maintaining coherence and accuracy in generated outputs. 

---
# Mitigating Cross-Modal Distraction and Ensuring Geometric Feasibility via Affordance-Guided, Self-Consistent MLLMs for Food Preparation Task Planning 

**Authors**: Yu-Hong Shen, Chuan-Yu Wu, Yi-Ru Yang, Yen-Ling Tai, Yi-Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13055)  

**Abstract**: We study Multimodal Large Language Models (MLLMs) with in-context learning for food preparation task planning. In this context, we identify two key challenges: cross-modal distraction and geometric feasibility. Cross-modal distraction occurs when the inclusion of visual input degrades the reasoning performance of a MLLM. Geometric feasibility refers to the ability of MLLMs to ensure that the selected skills are physically executable in the environment. To address these issues, we adapt Chain of Thought (CoT) with Self-Consistency to mitigate reasoning loss from cross-modal distractions and use affordance predictor as skill preconditions to guide MLLM on geometric feasibility. We construct a dataset to evaluate the ability of MLLMs on quantity estimation, reachability analysis, relative positioning and collision avoidance. We conducted a detailed evaluation to identify issues among different baselines and analyze the reasons for improvement, providing insights into each approach. Our method reaches a success rate of 76.7% on the entire dataset, showing a substantial improvement over the CoT baseline at 36.7%. 

---
# HybridGen: VLM-Guided Hybrid Planning for Scalable Data Generation of Imitation Learning 

**Authors**: Wensheng Wang, Ning Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13171)  

**Abstract**: The acquisition of large-scale and diverse demonstration data are essential for improving robotic imitation learning generalization. However, generating such data for complex manipulations is challenging in real-world settings. We introduce HybridGen, an automated framework that integrates Vision-Language Model (VLM) and hybrid planning. HybridGen uses a two-stage pipeline: first, VLM to parse expert demonstrations, decomposing tasks into expert-dependent (object-centric pose transformations for precise control) and plannable segments (synthesizing diverse trajectories via path planning); second, pose transformations substantially expand the first-stage data. Crucially, HybridGen generates a large volume of training data without requiring specific data formats, making it broadly applicable to a wide range of imitation learning algorithms, a characteristic which we also demonstrate empirically across multiple algorithms. Evaluations across seven tasks and their variants demonstrate that agents trained with HybridGen achieve substantial performance and generalization gains, averaging a 5% improvement over state-of-the-art methods. Notably, in the most challenging task variants, HybridGen achieves significant improvement, reaching a 59.7% average success rate, significantly outperforming Mimicgen's 49.5%. These results demonstrating its effectiveness and practicality. 

---
# Free-form language-based robotic reasoning and grasping 

**Authors**: Runyu Jiao, Alice Fasoli, Francesco Giuliari, Matteo Bortolon, Sergio Povoli, Guofeng Mei, Yiming Wang, Fabio Poiesi  

**Link**: [PDF](https://arxiv.org/pdf/2503.13082)  

**Abstract**: Performing robotic grasping from a cluttered bin based on human instructions is a challenging task, as it requires understanding both the nuances of free-form language and the spatial relationships between objects. Vision-Language Models (VLMs) trained on web-scale data, such as GPT-4o, have demonstrated remarkable reasoning capabilities across both text and images. But can they truly be used for this task in a zero-shot setting? And what are their limitations? In this paper, we explore these research questions via the free-form language-based robotic grasping task, and propose a novel method, FreeGrasp, leveraging the pre-trained VLMs' world knowledge to reason about human instructions and object spatial arrangements. Our method detects all objects as keypoints and uses these keypoints to annotate marks on images, aiming to facilitate GPT-4o's zero-shot spatial reasoning. This allows our method to determine whether a requested object is directly graspable or if other objects must be grasped and removed first. Since no existing dataset is specifically designed for this task, we introduce a synthetic dataset FreeGraspData by extending the MetaGraspNetV2 dataset with human-annotated instructions and ground-truth grasping sequences. We conduct extensive analyses with both FreeGraspData and real-world validation with a gripper-equipped robotic arm, demonstrating state-of-the-art performance in grasp reasoning and execution. Project website: this https URL. 

---
# Lifting the Veil on Visual Information Flow in MLLMs: Unlocking Pathways to Faster Inference 

**Authors**: Hao Yin, Guangzong Si, Zilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13108)  

**Abstract**: Multimodal large language models (MLLMs) improve performance on vision-language tasks by integrating visual features from pre-trained vision encoders into large language models (LLMs). However, how MLLMs process and utilize visual information remains unclear. In this paper, a shift in the dominant flow of visual information is uncovered: (1) in shallow layers, strong interactions are observed between image tokens and instruction tokens, where most visual information is injected into instruction tokens to form cross-modal semantic representations; (2) in deeper layers, image tokens primarily interact with each other, aggregating the remaining visual information to optimize semantic representations within visual modality. Based on these insights, we propose Hierarchical Modality-Aware Pruning (HiMAP), a plug-and-play inference acceleration method that dynamically prunes image tokens at specific layers, reducing computational costs by approximately 65% without sacrificing performance. Our findings offer a new understanding of visual information processing in MLLMs and provide a state-of-the-art solution for efficient inference. 

---
# MMLNB: Multi-Modal Learning for Neuroblastoma Subtyping Classification Assisted with Textual Description Generation 

**Authors**: Huangwei Chen, Zhu Zhu, Zhenyu Yan, Yifei Chen, Mingyang Ding, Chenlei Li, Feiwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.12927)  

**Abstract**: Neuroblastoma (NB), a leading cause of childhood cancer mortality, exhibits significant histopathological variability, necessitating precise subtyping for accurate prognosis and treatment. Traditional diagnostic methods rely on subjective evaluations that are time-consuming and inconsistent. To address these challenges, we introduce MMLNB, a multi-modal learning (MML) model that integrates pathological images with generated textual descriptions to improve classification accuracy and interpretability. The approach follows a two-stage process. First, we fine-tune a Vision-Language Model (VLM) to enhance pathology-aware text generation. Second, the fine-tuned VLM generates textual descriptions, using a dual-branch architecture to independently extract visual and textual features. These features are fused via Progressive Robust Multi-Modal Fusion (PRMF) Block for stable training. Experimental results show that the MMLNB model is more accurate than the single modal model. Ablation studies demonstrate the importance of multi-modal fusion, fine-tuning, and the PRMF mechanism. This research creates a scalable AI-driven framework for digital pathology, enhancing reliability and interpretability in NB subtyping classification. Our source code is available at this https URL. 

---
# Aligning Vision to Language: Text-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning 

**Authors**: Junming Liu, Siyuan Meng, Yanting Gao, Song Mao, Pinlong Cai, Guohang Yan, Yirong Chen, Zilin Bian, Botian Shi, Ding Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12972)  

**Abstract**: Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at this https URL. 

---
# Training Video Foundation Models with NVIDIA NeMo 

**Authors**: Zeeshan Patel, Ethan He, Parth Mannan, Xiaowei Ren, Ryan Wolf, Niket Agarwal, Jacob Huffman, Zhuoyao Wang, Carl Wang, Jack Chang, Yan Bai, Tommy Huang, Linnan Wang, Sahil Jain, Shanmugam Ramasamy, Joseph Jennings, Ekaterina Sirazitdinova, Oleg Sudakov, Mingyuan Ma, Bobby Chen, Forrest Lin, Hao Wang, Vasanth Rao Naik Sabavat, Sriharsha Niverty, Rong Ou, Pallab Bhattacharya, David Page, Nima Tajbakhsh, Ashwath Aithal  

**Link**: [PDF](https://arxiv.org/pdf/2503.12964)  

**Abstract**: Video Foundation Models (VFMs) have recently been used to simulate the real world to train physical AI systems and develop creative visual experiences. However, there are significant challenges in training large-scale, high quality VFMs that can generate high-quality videos. We present a scalable, open-source VFM training pipeline with NVIDIA NeMo, providing accelerated video dataset curation, multimodal data loading, and parallelized video diffusion model training and inference. We also provide a comprehensive performance analysis highlighting best practices for efficient VFM training and inference. 

---
# From Head to Tail: Towards Balanced Representation in Large Vision-Language Models through Adaptive Data Calibration 

**Authors**: Mingyang Song, Xiaoye Qu, Jiawei Zhou, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.12821)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved significant progress in combining visual comprehension with language generation. Despite this success, the training data of LVLMs still suffers from Long-Tail (LT) problems, where the data distribution is highly imbalanced. Previous works have mainly focused on traditional VLM architectures, i.e., CLIP or ViT, and specific tasks such as recognition and classification. Nevertheless, the exploration of LVLM (e.g. LLaVA) and more general tasks (e.g. Visual Question Answering and Visual Reasoning) remains under-explored. In this paper, we first conduct an in-depth analysis of the LT issues in LVLMs and identify two core causes: the overrepresentation of head concepts and the underrepresentation of tail concepts. Based on the above observation, we propose an $\textbf{A}$daptive $\textbf{D}$ata $\textbf{R}$efinement Framework ($\textbf{ADR}$), which consists of two stages: $\textbf{D}$ata $\textbf{R}$ebalancing ($\textbf{DR}$) and $\textbf{D}$ata $\textbf{S}$ynthesis ($\textbf{DS}$). In the DR stage, we adaptively rebalance the redundant data based on entity distributions, while in the DS stage, we leverage Denoising Diffusion Probabilistic Models (DDPMs) and scarce images to supplement underrepresented portions. Through comprehensive evaluations across eleven benchmarks, our proposed ADR effectively mitigates the long-tail problem in the training data, improving the average performance of LLaVA 1.5 relatively by 4.36%, without increasing the training data volume. 

---
# PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior 

**Authors**: Seunggwan Lee, Hwanhee Jung, Byoungsoo Koh, Qixing Huang, Sangho Yoon, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12834)  

**Abstract**: A fundamental challenge in conditional 3D shape generation is to minimize the information loss and maximize the intention of user input. Existing approaches have predominantly focused on two types of isolated conditional signals, i.e., user sketches and text descriptions, each of which does not offer flexible control of the generated shape. In this paper, we introduce PASTA, the flexible approach that seamlessly integrates a user sketch and a text description for 3D shape generation. The key idea is to use text embeddings from a vision-language model to enrich the semantic representation of sketches. Specifically, these text-derived priors specify the part components of the object, compensating for missing visual cues from ambiguous sketches. In addition, we introduce ISG-Net which employs two types of graph convolutional networks: IndivGCN, which processes fine-grained details, and PartGCN, which aggregates these details into parts and refines the structure of objects. Extensive experiments demonstrate that PASTA outperforms existing methods in part-level editing and achieves state-of-the-art results in sketch-to-3D shape generation. 

---
# NuPlanQA: A Large-Scale Dataset and Benchmark for Multi-View Driving Scene Understanding in Multi-Modal Large Language Models 

**Authors**: Sung-Yeon Park, Can Cui, Yunsheng Ma, Ahmadreza Moradipari, Rohit Gupta, Kyungtae Han, Ziran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12772)  

**Abstract**: Recent advances in multi-modal large language models (MLLMs) have demonstrated strong performance across various domains; however, their ability to comprehend driving scenes remains less proven. The complexity of driving scenarios, which includes multi-view information, poses significant challenges for existing MLLMs. In this paper, we introduce NuPlanQA-Eval, a multi-view, multi-modal evaluation benchmark for driving scene understanding. To further support generalization to multi-view driving scenarios, we also propose NuPlanQA-1M, a large-scale dataset comprising 1M real-world visual question-answering (VQA) pairs. For context-aware analysis of traffic scenes, we categorize our dataset into nine subtasks across three core skills: Road Environment Perception, Spatial Relations Recognition, and Ego-Centric Reasoning. Furthermore, we present BEV-LLM, integrating Bird's-Eye-View (BEV) features from multi-view images into MLLMs. Our evaluation results reveal key challenges that existing MLLMs face in driving scene-specific perception and spatial reasoning from ego-centric perspectives. In contrast, BEV-LLM demonstrates remarkable adaptability to this domain, outperforming other models in six of the nine subtasks. These findings highlight how BEV integration enhances multi-view MLLMs while also identifying key areas that require further refinement for effective adaptation to driving scenes. To facilitate further research, we publicly release NuPlanQA at this https URL. 

---
# Towards Scalable Foundation Model for Multi-modal and Hyperspectral Geospatial Data 

**Authors**: Haozhe Si, Yuxuan Wan, Minh Do, Deepak Vasisht, Han Zhao, Hendrik F. Hamann  

**Link**: [PDF](https://arxiv.org/pdf/2503.12843)  

**Abstract**: Geospatial raster (imagery) data, such as that collected by satellite-based imaging systems at different times and spectral bands, hold immense potential for enabling a wide range of high-impact applications. This potential stems from the rich information that is spatially and temporally contextualized across multiple channels and sensing modalities. Recent work has adapted existing self-supervised learning approaches for such geospatial data. However, they fall short of scalable model architectures, leading to inflexibility and computational inefficiencies when faced with an increasing number of channels and modalities. To address these limitations, we introduce Low-rank Efficient Spatial-Spectral Vision Transformer (LESS ViT) with three key innovations: i) the LESS Attention Block that approximates high-dimensional spatial-spectral attention through Kronecker's product of the low-dimensional spatial and spectral attention components; ii) the Continuous Positional-Channel Embedding Layer that preserves both spatial and spectral continuity and physical characteristics of each patch; and iii) the Perception Field Mask that exploits local spatial dependencies by constraining attention to neighboring patches. To evaluate the proposed innovations, we construct a benchmark, GFM-Bench, which serves as a comprehensive benchmark for such geospatial raster data. We pretrain LESS ViT using a Hyperspectral Masked Autoencoder framework with integrated positional and channel masking strategies. Experimental results demonstrate that our proposed method surpasses current state-of-the-art multi-modal geospatial foundation models, achieving superior performance with less computation and fewer parameters. The flexibility and extensibility of our framework make it a promising direction for future geospatial data analysis tasks that involve a wide range of modalities and channels. 

---
# MAVEN: Multi-modal Attention for Valence-Arousal Emotion Network 

**Authors**: Vrushank Ahire, Kunal Shah, Mudasir Nazir Khan, Nikhil Pakhale, Lownish Rai Sookha, M. A. Ganaie, Abhinav Dhall  

**Link**: [PDF](https://arxiv.org/pdf/2503.12623)  

**Abstract**: This paper introduces MAVEN (Multi-modal Attention for Valence-Arousal Emotion Network), a novel architecture for dynamic emotion recognition through dimensional modeling of affect. The model uniquely integrates visual, audio, and textual modalities via a bi-directional cross-modal attention mechanism with six distinct attention pathways, enabling comprehensive interactions between all modality pairs. Our proposed approach employs modality-specific encoders to extract rich feature representations from synchronized video frames, audio segments, and transcripts. The architecture's novelty lies in its cross-modal enhancement strategy, where each modality representation is refined through weighted attention from other modalities, followed by self-attention refinement through modality-specific encoders. Rather than directly predicting valence-arousal values, MAVEN predicts emotions in a polar coordinate form, aligning with psychological models of the emotion circumplex. Experimental evaluation on the Aff-Wild2 dataset demonstrates the effectiveness of our approach, with performance measured using Concordance Correlation Coefficient (CCC). The multi-stage architecture demonstrates superior ability to capture the complex, nuanced nature of emotional expressions in conversational videos, advancing the state-of-the-art (SOTA) in continuous emotion recognition in-the-wild. Code can be found at: this https URL. 

---
# GeoRSMLLM: A Multimodal Large Language Model for Vision-Language Tasks in Geoscience and Remote Sensing 

**Authors**: Zilun Zhang, Haozhan Shen, Tiancheng Zhao, Bin Chen, Zian Guan, Yuhao Wang, Xu Jia, Yuxiang Cai, Yongheng Shang, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.12490)  

**Abstract**: The application of Vision-Language Models (VLMs) in remote sensing (RS) has demonstrated significant potential in traditional tasks such as scene classification, object detection, and image captioning. However, current models, which excel in Referring Expression Comprehension (REC), struggle with tasks involving complex instructions (e.g., exists multiple conditions) or pixel-level operations like segmentation and change detection. In this white paper, we provide a comprehensive hierarchical summary of vision-language tasks in RS, categorized by the varying levels of cognitive capability required. We introduce the Remote Sensing Vision-Language Task Set (RSVLTS), which includes Open-Vocabulary Tasks (OVT), Referring Expression Tasks (RET), and Described Object Tasks (DOT) with increased difficulty, and Visual Question Answering (VQA) aloneside. Moreover, we propose a novel unified data representation using a set-of-points approach for RSVLTS, along with a condition parser and a self-augmentation strategy based on cyclic referring. These features are integrated into the GeoRSMLLM model, and this enhanced model is designed to handle a broad range of tasks of RSVLTS, paving the way for a more generalized solution for vision-language tasks in geoscience and remote sensing. 

---
# BREEN: Bridge Data-Efficient Encoder-Free Multimodal Learning with Learnable Queries 

**Authors**: Tianle Li, Yongming Rao, Winston Hu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.12446)  

**Abstract**: Encoder-free multimodal large language models(MLLMs) eliminate the need for a well-trained vision encoder by directly processing image tokens before the language model. While this approach reduces computational overhead and model complexity, it often requires large amounts of training data to effectively capture the visual knowledge typically encoded by vision models like CLIP. The absence of a vision encoder implies that the model is likely to rely on substantial data to learn the necessary visual-semantic alignments. In this work, we present BREEN, a data-efficient encoder-free multimodal architecture that mitigates this issue. BREEN leverages a learnable query and image experts to achieve comparable performance with significantly less training data. The learnable query, positioned between image and text tokens, is supervised by the output of a pretrained CLIP model to distill visual knowledge, bridging the gap between visual and textual modalities. Additionally, the image expert processes image tokens and learnable queries independently, improving efficiency and reducing interference with the LLM's textual capabilities. BREEN achieves comparable performance to prior encoder-free state-of-the-art models like Mono-InternVL, using only 13 million text-image pairs in training about one percent of the data required by existing methods. Our work highlights a promising direction for data-efficient encoder-free multimodal learning, offering an alternative to traditional encoder-based approaches. 

---
# Leveraging Vision Capabilities of Multimodal LLMs for Automated Data Extraction from Plots 

**Authors**: Maciej P. Polak, Dane Morgan  

**Link**: [PDF](https://arxiv.org/pdf/2503.12326)  

**Abstract**: Automated data extraction from research texts has been steadily improving, with the emergence of large language models (LLMs) accelerating progress even further. Extracting data from plots in research papers, however, has been such a complex task that it has predominantly been confined to manual data extraction. We show that current multimodal large language models, with proper instructions and engineered workflows, are capable of accurately extracting data from plots. This capability is inherent to the pretrained models and can be achieved with a chain-of-thought sequence of zero-shot engineered prompts we call PlotExtract, without the need to fine-tune. We demonstrate PlotExtract here and assess its performance on synthetic and published plots. We consider only plots with two axes in this analysis. For plots identified as extractable, PlotExtract finds points with over 90% precision (and around 90% recall) and errors in x and y position of around 5% or lower. These results prove that multimodal LLMs are a viable path for high-throughput data extraction for plots and in many circumstances can replace the current manual methods of data extraction. 

---
# When neural implant meets multimodal LLM: A dual-loop system for neuromodulation and naturalistic neuralbehavioral research 

**Authors**: Edward Hong Wang, Cynthia Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.12334)  

**Abstract**: We propose a novel dual-loop system that synergistically combines responsive neurostimulation (RNS) implants with artificial intelligence-driven wearable devices for treating post-traumatic stress disorder (PTSD) and enabling naturalistic brain research. In PTSD Therapy Mode, an implanted closed-loop neural device monitors amygdala activity and provides on-demand stimulation upon detecting pathological theta oscillations, while an ensemble of wearables (smart glasses, smartwatches, smartphones) uses multimodal large language model (LLM) analysis of sensory data to detect environmental or physiological PTSD triggers and deliver timely audiovisual interventions. Logged events from both the neural and wearable loops are analyzed to personalize trigger detection and progressively transition patients to non-invasive interventions. In Neuroscience Research Mode, the same platform is adapted for real-world brain activity capture. Wearable-LLM systems recognize naturalistic events (social interactions, emotional situations, compulsive behaviors, decision making) and signal implanted RNS devices (via wireless triggers) to record synchronized intracranial data during these moments. This approach builds on recent advances in mobile intracranial EEG recording and closed-loop neuromodulation in humans (BRAIN Initiative, 2023) (Mobbs et al., 2021). We discuss how our interdisciplinary system could revolutionize PTSD therapy and cognitive neuroscience by enabling 24/7 monitoring, context-aware intervention, and rich data collection outside traditional labs. The vision is a future where AI-enhanced devices continuously collaborate with the human brain, offering therapeutic support and deep insights into neural function, with the resulting real-world context rich neural data, in turn, accelerating the development of more biologically-grounded and human-centric AI. 

---
# LIAM: Multimodal Transformer for Language Instructions, Images, Actions and Semantic Maps 

**Authors**: Yihao Wang, Raphael Memmesheimer, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2503.12230)  

**Abstract**: The availability of large language models and open-vocabulary object perception methods enables more flexibility for domestic service robots. The large variability of domestic tasks can be addressed without implementing each task individually by providing the robot with a task description along with appropriate environment information. In this work, we propose LIAM - an end-to-end model that predicts action transcripts based on language, image, action, and map inputs. Language and image inputs are encoded with a CLIP backbone, for which we designed two pre-training tasks to fine-tune its weights and pre-align the latent spaces. We evaluate our method on the ALFRED dataset, a simulator-generated benchmark for domestic tasks. Our results demonstrate the importance of pre-aligning embedding spaces from different modalities and the efficacy of incorporating semantic maps. 

---
# DiffGAP: A Lightweight Diffusion Module in Contrastive Space for Bridging Cross-Model Gap 

**Authors**: Shentong Mo, Zehua Chen, Fan Bao, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12131)  

**Abstract**: Recent works in cross-modal understanding and generation, notably through models like CLAP (Contrastive Language-Audio Pretraining) and CAVP (Contrastive Audio-Visual Pretraining), have significantly enhanced the alignment of text, video, and audio embeddings via a single contrastive loss. However, these methods often overlook the bidirectional interactions and inherent noises present in each modality, which can crucially impact the quality and efficacy of cross-modal integration. To address this limitation, we introduce DiffGAP, a novel approach incorporating a lightweight generative module within the contrastive space. Specifically, our DiffGAP employs a bidirectional diffusion process tailored to bridge the cross-modal gap more effectively. This involves a denoising process on text and video embeddings conditioned on audio embeddings and vice versa, thus facilitating a more nuanced and robust cross-modal interaction. Our experimental results on VGGSound and AudioCaps datasets demonstrate that DiffGAP significantly improves performance in video/text-audio generation and retrieval tasks, confirming its effectiveness in enhancing cross-modal understanding and generation capabilities. 

---
# Compose Your Aesthetics: Empowering Text-to-Image Models with the Principles of Art 

**Authors**: Zhe Jin, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.12018)  

**Abstract**: Text-to-Image (T2I) diffusion models (DM) have garnered widespread adoption due to their capability in generating high-fidelity outputs and accessibility to anyone able to put imagination into words. However, DMs are often predisposed to generate unappealing outputs, much like the random images on the internet they were trained on. Existing approaches to address this are founded on the implicit premise that visual aesthetics is universal, which is limiting. Aesthetics in the T2I context should be about personalization and we propose the novel task of aesthetics alignment which seeks to align user-specified aesthetics with the T2I generation output. Inspired by how artworks provide an invaluable perspective to approach aesthetics, we codify visual aesthetics using the compositional framework artists employ, known as the Principles of Art (PoA). To facilitate this study, we introduce CompArt, a large-scale compositional art dataset building on top of WikiArt with PoA analysis annotated by a capable Multimodal LLM. Leveraging the expressive power of LLMs and training a lightweight and transferrable adapter, we demonstrate that T2I DMs can effectively offer 10 compositional controls through user-specified PoA conditions. Additionally, we design an appropriate evaluation framework to assess the efficacy of our approach. 

---
# V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents 

**Authors**: Zhengrong Yue, Shaobin Zhuang, Kunchang Li, Yanbo Ding, Yali Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12077)  

**Abstract**: Despite the recent advancement in video stylization, most existing methods struggle to render any video with complex transitions, based on an open style description of user query. To fill this gap, we introduce a generic multi-agent system for video stylization, V-Stylist, by a novel collaboration and reflection paradigm of multi-modal large language models. Specifically, our V-Stylist is a systematical workflow with three key roles: (1) Video Parser decomposes the input video into a number of shots and generates their text prompts of key shot content. Via a concise video-to-shot prompting paradigm, it allows our V-Stylist to effectively handle videos with complex transitions. (2) Style Parser identifies the style in the user query and progressively search the matched style model from a style tree. Via a robust tree-of-thought searching paradigm, it allows our V-Stylist to precisely specify vague style preference in the open user query. (3) Style Artist leverages the matched model to render all the video shots into the required style. Via a novel multi-round self-reflection paradigm, it allows our V-Stylist to adaptively adjust detail control, according to the style requirement. With such a distinct design of mimicking human professionals, our V-Stylist achieves a major breakthrough over the primary challenges for effective and automatic video stylization. Moreover,we further construct a new benchmark Text-driven Video Stylization Benchmark (TVSBench), which fills the gap to assess stylization of complex videos on open user queries. Extensive experiments show that, V-Stylist achieves the state-of-the-art, e.g.,V-Stylist surpasses FRESCO and ControlVideo by 6.05% and 4.51% respectively in overall average metrics, marking a significant advance in video stylization. 

---
# MELON: Multimodal Mixture-of-Experts with Spectral-Temporal Fusion for Long-Term Mobility Estimation in Critical Care 

**Authors**: Jiaqing Zhang, Miguel Contreras, Jessica Sena, Andrea Davidson, Yuanfang Ren, Ziyuan Guan, Tezcan Ozrazgat-Baslanti, Tyler J. Loftus, Subhash Nerella, Azra Bihorac, Parisa Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11695)  

**Abstract**: Patient mobility monitoring in intensive care is critical for ensuring timely interventions and improving clinical outcomes. While accelerometry-based sensor data are widely adopted in training artificial intelligence models to estimate patient mobility, existing approaches face two key limitations highlighted in clinical practice: (1) modeling the long-term accelerometer data is challenging due to the high dimensionality, variability, and noise, and (2) the absence of efficient and robust methods for long-term mobility assessment. To overcome these challenges, we introduce MELON, a novel multimodal framework designed to predict 12-hour mobility status in the critical care setting. MELON leverages the power of a dual-branch network architecture, combining the strengths of spectrogram-based visual representations and sequential accelerometer statistical features. MELON effectively captures global and fine-grained mobility patterns by integrating a pre-trained image encoder for rich frequency-domain feature extraction and a Mixture-of-Experts encoder for sequence modeling. We trained and evaluated the MELON model on the multimodal dataset of 126 patients recruited from nine Intensive Care Units at the University of Florida Health Shands Hospital main campus in Gainesville, Florida. Experiments showed that MELON outperforms conventional approaches for 12-hour mobility status estimation with an overall area under the receiver operating characteristic curve (AUROC) of 0.82 (95\%, confidence interval 0.78-0.86). Notably, our experiments also revealed that accelerometer data collected from the wrist provides robust predictive performance compared with data from the ankle, suggesting a single-sensor solution that can reduce patient burden and lower deployment costs... 

---
