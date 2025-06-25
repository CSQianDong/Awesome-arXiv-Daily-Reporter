# KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality 

**Authors**: Baochang Ren, Shuofei Qiao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19807)  

**Abstract**: Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at this https URL. 

---
# KunLunBaizeRAG: Reinforcement Learning Driven Inference Performance Leap for Large Language Models 

**Authors**: Cheng Li, Jiexiong Liu, Yixuan Chen, Qihang Zhou, KunLun Meta  

**Link**: [PDF](https://arxiv.org/pdf/2506.19466)  

**Abstract**: This paper introduces KunLunBaizeRAG, a reinforcement learning-driven reasoning framework designed to enhance the reasoning capabilities of large language models (LLMs) in complex multi-hop question-answering tasks. The framework addresses key limitations of traditional RAG, such as retrieval drift, information redundancy, and strategy rigidity. Key innovations include the RAG-driven Reasoning Alignment (RDRA) mechanism, the Search-Think Iterative Enhancement (STIE) mechanism, the Network-Local Intelligent Routing (NLR) mechanism, and a progressive hybrid training strategy. Experimental results demonstrate significant improvements in exact match (EM) and LLM-judged score (LJ) across four benchmarks, highlighting the framework's robustness and effectiveness in complex reasoning scenarios. 

---
# RecLLM-R1: A Two-Stage Training Paradigm with Reinforcement Learning and Chain-of-Thought v1 

**Authors**: Yu Xie, Xingkai Ren, Ying Qi, Yao Hu, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19235)  

**Abstract**: Traditional recommendation systems often grapple with "filter bubbles", underutilization of external knowledge, and a disconnect between model optimization and business policy iteration. To address these limitations, this paper introduces RecLLM-R1, a novel recommendation framework leveraging Large Language Models (LLMs) and drawing inspiration from the DeepSeek R1 methodology. The framework initiates by transforming user profiles, historical interactions, and multi-faceted item attributes into LLM-interpretable natural language prompts through a carefully engineered data construction process. Subsequently, a two-stage training paradigm is employed: the initial stage involves Supervised Fine-Tuning (SFT) to imbue the LLM with fundamental recommendation capabilities. The subsequent stage utilizes Group Relative Policy Optimization (GRPO), a reinforcement learning technique, augmented with a Chain-of-Thought (CoT) mechanism. This stage guides the model through multi-step reasoning and holistic decision-making via a flexibly defined reward function, aiming to concurrently optimize recommendation accuracy, diversity, and other bespoke business objectives. Empirical evaluations on a real-world user behavior dataset from a large-scale social media platform demonstrate that RecLLM-R1 significantly surpasses existing baseline methods across a spectrum of evaluation metrics, including accuracy, diversity, and novelty. It effectively mitigates the filter bubble effect and presents a promising avenue for the integrated optimization of recommendation models and policies under intricate business goals. 

---
# SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning 

**Authors**: Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19767)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in reasoning tasks, yet the optimal integration of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) remains a fundamental challenge. Through comprehensive analysis of token distributions, learning dynamics, and integration mechanisms from entropy-based perspectives, we reveal key differences between these paradigms: SFT induces coarse-grained global changes to LLM policy distributions, while RL performs fine-grained selective optimizations, with entropy serving as a critical indicator of training effectiveness. Building on these observations, we propose Supervised Reinforcement Fine-Tuning (SRFT), a single-stage method that unifies both fine-tuning paradigms through entropy-aware weighting mechanisms. Our approach simultaneously applies SFT and RL to directly optimize the LLM using demonstrations and self-exploration rollouts rather than through two-stage sequential methods. Extensive experiments show that SRFT achieves 59.1% average accuracy, outperforming zero-RL methods by 9.0% on five mathematical reasoning benchmarks and 10.9% on three out-of-distribution benchmarks. 

---
# Surgery-R1: Advancing Surgical-VQLA with Reasoning Multimodal Large Language Model via Reinforcement Learning 

**Authors**: Pengfei Hao, Shuaibo Li, Hongqiu Wang, Zhizhuo Kou, Junhang Zhang, Guang Yang, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19469)  

**Abstract**: In recent years, significant progress has been made in the field of surgical scene understanding, particularly in the task of Visual Question Localized-Answering in robotic surgery (Surgical-VQLA). However, existing Surgical-VQLA models lack deep reasoning capabilities and interpretability in surgical scenes, which limits their reliability and potential for development in clinical applications. To address this issue, inspired by the development of Reasoning Multimodal Large Language Models (MLLMs), we first build the Surgery-R1-54k dataset, including paired data for Visual-QA, Grounding-QA, and Chain-of-Thought (CoT). Then, we propose the first Reasoning MLLM for Surgical-VQLA (Surgery-R1). In our Surgery-R1, we design a two-stage fine-tuning mechanism to enable the basic MLLM with complex reasoning abilities by utilizing supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). Furthermore, for an efficient and high-quality rule-based reward system in our RFT, we design a Multimodal Coherence reward mechanism to mitigate positional illusions that may arise in surgical scenarios. Experiment results demonstrate that Surgery-R1 outperforms other existing state-of-the-art (SOTA) models in the Surgical-VQLA task and widely-used MLLMs, while also validating its reasoning capabilities and the effectiveness of our approach. The code and dataset will be organized in this https URL. 

---
# Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager 

**Authors**: Lucie Galland, Catherine Pelachaud, Florian Pecune  

**Link**: [PDF](https://arxiv.org/pdf/2506.19652)  

**Abstract**: In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals. 

---
