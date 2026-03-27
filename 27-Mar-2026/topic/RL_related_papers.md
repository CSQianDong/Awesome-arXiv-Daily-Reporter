# TAPO: Translation Augmented Policy Optimization for Multilingual Mathematical Reasoning 

**Authors**: Xu Huang, Zhejian Lai, Zixian Huang, Jiajun Chen, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25419)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in English mathematical reasoning, yet a significant performance disparity persists in multilingual contexts, largely attributed to deficiencies in language understanding. To bridge this gap, we introduce Translation-Augmented Policy Optimization (TAPO), a novel reinforcement learning framework built upon GRPO. TAPO enforces an explicit alignment strategy where the model leverages English as a pivot and follows an understand-then-reason paradigm. Crucially, we employ a step-level relative advantage mechanism that decouples understanding from reasoning, allowing the integration of translation quality rewards without introducing optimization conflicts. Extensive experiments reveal that TAPO effectively synergizes language understanding with reasoning capabilities and is compatible with various models. It outperforms baseline methods in both multilingual mathematical reasoning and translation tasks, while generalizing well to unseen languages and out-of-domain tasks. 

---
# Self-Improvement of Large Language Models: A Technical Overview and Future Outlook 

**Authors**: Haoyan Yang, Mario Xerri, Solha Park, Huajian Zhang, Yiyang Feng, Sai Akhil Kogilathota, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.25681)  

**Abstract**: As large language models (LLMs) continue to advance, improving them solely through human supervision is becoming increasingly costly and limited in scalability. As models approach human-level capabilities in certain domains, human feedback may no longer provide sufficiently informative signals for further improvement. At the same time, the growing ability of models to make autonomous decisions and execute complex actions naturally enables abstractions in which components of the model development process can be progressively automated. Together, these challenges and opportunities have driven increasing interest in self-improvement, where models autonomously generate data, evaluate outputs, and iteratively refine their own capabilities. In this paper, we present a system-level perspective on self-improving language models and introduce a unified framework that organizes existing techniques. We conceptualize the self-improvement system as a closed-loop lifecycle, consisting of four tightly coupled processes: data acquisition, data selection, model optimization, and inference refinement, along with an autonomous evaluation layer. Within this framework, the model itself plays a central role in driving each stage: collecting or generating data, selecting informative signals, updating its parameters, and refining outputs, while the autonomous evaluation layer continuously monitors progress and guides the improvement cycle across stages. Following this lifecycle perspective, we systematically review and analyze representative methods for each component from a technical standpoint. We further discuss current limitations and outline our vision for future research toward fully self-improving LLMs. 

---
# OMIND: Framework for Knowledge Grounded Finetuning and Multi-Turn Dialogue Benchmark for Mental Health LLMs 

**Authors**: Suraj Racha, Prashant Harish Joshi, Utkarsh Maurya, Nitin Yadav, Mridul Sharma, Ananya Kunisetty, Saranya Darisipudi, Nirmal Punjabi, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25105)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities for complex tasks, yet adaptation in medical domain, specifically mental health, poses specific challenges. Mental health is a rising concern globally with LLMs having large potential to help address the same. We highlight three primary challenges for LLMs in mental health - lack of high quality interpretable and knowledge grounded training data; training paradigms restricted to core capabilities, and evaluation of multi turn dialogue settings. Addressing it, we present oMind framework which includes training and aligning LLM agents for diverse capabilities including conversations; high quality ~164k multi-task SFT dataset, as a result of our generation pipeline based on Structured Knowledge retrieval, LLM based pruning, and review actions. We also introduce oMind-Chat - a novel multi turn benchmark dataset with expert annotated turn level and conversation level rubrics. Our diverse experiments on both core capabilities and conversations shows oMind LLMs consistently outperform baselines. oMind-LLM also shows significantly better reasoning with up to 80% win rate. 

---
# Reaching Beyond the Mode: RL for Distributional Reasoning in Language Models 

**Authors**: Isha Puri, Mehul Damani, Idan Shenfeld, Marzyeh Ghassemi, Jacob Andreas, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.24844)  

**Abstract**: Given a question, a language model (LM) implicitly encodes a distribution over possible answers. In practice, post-training procedures for LMs often collapse this distribution onto a single dominant mode. While this is generally not a problem for benchmark-style evaluations that assume one correct answer, many real-world tasks inherently involve multiple valid answers or irreducible uncertainty. Examples include medical diagnosis, ambiguous question answering, and settings with incomplete information. In these cases, we would like LMs to generate multiple plausible hypotheses, ideally with confidence estimates for each one, and without computationally intensive repeated sampling to generate non-modal answers. This paper describes a multi-answer reinforcement learning approach for training LMs to perform distributional reasoning over multiple answers during inference. We modify the RL objective to enable models to explicitly generate multiple candidate answers in a single forward pass, internalizing aspects of inference-time search into the model's generative process. Across question-answering, medical diagnostic, and coding benchmarks, we observe improved diversity, coverage, and set-level calibration scores compared to single answer trained baselines. Models trained with our approach require fewer tokens to generate multiple answers than competing approaches. On coding tasks, they are also substantially more accurate. These results position multi-answer RL as a principled and compute-efficient alternative to inference-time scaling procedures such as best-of-k. Code and more information can be found at this https URL. 

---
# Training LLMs for Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards 

**Authors**: Cheng Jiayang, Xin Liu, Zhihan Zhang, Haoyang Wen, Zixuan Zhang, Qingyu Yin, Shiyang Li, Priyanka Nigam, Bing Yin, Chao Zhang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.24709)  

**Abstract**: Multi-step tool orchestration, where LLMs must invoke multiple dependent APIs in the correct order while propagating intermediate outputs, remains challenging. State-of-the-art models frequently fail on full sequence execution, with parameter value errors accounting for a significant portion of failures. Training models to handle such workflows faces two obstacles: existing environments focus on simple per-turn function calls with simulated data, and binary rewards provide no signal for partial correctness.
We present a framework addressing both challenges. First, we construct a reinforcement learning environment backed by a large-scale cache of real API responses, enabling a data synthesis pipeline that samples valid multi-step orchestration traces with controllable complexity and significantly higher generation efficiency than unconstrained methods. Second, we propose a graduated reward design that decomposes correctness into atomic validity (individual function call correctness at increasing granularity) and orchestration (correct tool sequencing with dependency respect). On ComplexFuncBench, our approach demonstrates substantial improvements in turn accuracy. Ablation studies confirm both reward components are essential: using either alone significantly degrades performance. 

---
# R-C2: Cycle-Consistent Reinforcement Learning Improves Multimodal Reasoning 

**Authors**: Zirui Zhang, Haoyu Dong, Kexin Pei, Chengzhi Mao  

**Link**: [PDF](https://arxiv.org/pdf/2603.25720)  

**Abstract**: Robust perception and reasoning require consistency across sensory modalities. Yet current multimodal models often violate this principle, yielding contradictory predictions for visual and textual representations of the same concept. Rather than masking these failures with standard voting mechanisms, which can amplify systematic biases, we show that cross-modal inconsistency provides a rich and natural signal for learning. We introduce RC2, a reinforcement learning framework that resolves internal conflicts by enforcing cross-modal cycle consistency. By requiring a model to perform backward inference, switch modalities, and reliably reconstruct the answer through forward inference, we obtain a dense, label-free reward. This cyclic constraint encourages the model to align its internal representations autonomously. Optimizing for this structure mitigates modality-specific errors and improves reasoning accuracy by up to 7.6 points. Our results suggest that advanced reasoning emerges not only from scaling data, but also from enforcing a structurally consistent understanding of the world. 

---
# Train at Moving Edge: Online-Verified Prompt Selection for Efficient RL Training of Large Reasoning Model 

**Authors**: Jiahao Wu, Ning Lu, Shengcai Liu, Kun Wang, Yanting Yang, Li Qing, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25184)  

**Abstract**: Reinforcement learning (RL) has become essential for post-training large language models (LLMs) in reasoning tasks. While scaling rollouts can stabilize training and enhance performance, the computational overhead is a critical issue. In algorithms like GRPO, multiple rollouts per prompt incur prohibitive costs, as a large portion of prompts provide negligible gradients and are thus of low utility. To address this problem, we investigate how to select high-utility prompts before the rollout phase. Our experimental analysis reveals that sample utility is non-uniform and evolving: the strongest learning signals concentrate at the ``learning edge", the intersection of intermediate difficulty and high uncertainty, which shifts as training proceeds. Motivated by this, we propose HIVE (History-Informed and online-VErified prompt selection), a dual-stage framework for data-efficient RL. HIVE utilizes historical reward trajectories for coarse selection and employs prompt entropy as a real-time proxy to prune instances with stale utility. By evaluating HIVE across multiple math reasoning benchmarks and models, we show that HIVE yields significant rollout efficiency without compromising performance. 

---
# Pixelis: Reasoning in Pixels, from Seeing to Acting 

**Authors**: Yunpeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.25091)  

**Abstract**: Most vision-language systems are static observers: they describe pixels, do not act, and cannot safely improve under shift. This passivity limits generalizable, physically grounded visual intelligence. Learning through action, not static description, is essential beyond curated data. We present Pixelis, a pixel-space agent that operates directly on images and videos via a compact set of executable operations (zoom/crop, segment, track, OCR, temporal localization) and learns from its consequences. Pixelis trains in three phases: (1) Supervised Fine-Tuning learns a pixel-tool grammar from Chain-of-Thought-Action traces with a masked imitation loss that upweights operation/argument tokens and auxiliary heads to stabilize pixel-grounded arguments; (2) Curiosity-Coherence Reward Fine-Tuning optimizes a dual-drive objective marrying prediction-error curiosity with adjacent-step coherence and a mild efficiency prior under a KL anchor, yielding short, valid, structured toolchains; (3) Pixel Test-Time RL performs label-free adaptation by retrieving neighbors, voting over complete trajectories rather than answers, and updating toward short, high-fidelity exemplars while constraining drift with a KL-to-EMA safety control. Across six public image and video benchmarks, Pixelis yields consistent improvements: the average relative gain is +4.08% over the same 8B baseline (peaking at +6.03% on VSI-Bench), computed as (ours-baseline)/baseline, while producing shorter, auditable toolchains and maintaining in-corridor KL during test-time learning. Acting within pixels, rather than abstract tokens, grounds multimodal perception in the physical world, linking visual reasoning with actionable outcomes, and enables embodied adaptation without external feedback. 

---
