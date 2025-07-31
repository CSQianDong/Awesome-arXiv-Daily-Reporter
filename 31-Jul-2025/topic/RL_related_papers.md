# From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs 

**Authors**: Jie He, Victor Gutierrez Basulto, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22716)  

**Abstract**: Reinforcement learning-based retrieval-augmented generation (RAG) methods enhance the reasoning abilities of large language models (LLMs). However, most rely only on final-answer rewards, overlooking intermediate reasoning quality. This paper analyzes existing RAG reasoning models and identifies three main failure patterns: (1) information insufficiency, meaning the model fails to retrieve adequate support; (2) faulty reasoning, where logical or content-level flaws appear despite sufficient information; and (3) answer-reasoning inconsistency, where a valid reasoning chain leads to a mismatched final answer. We propose TIRESRAG-R1, a novel framework using a think-retrieve-reflect process and a multi-dimensional reward system to improve reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to encourage thorough retrieval; (2) a reasoning quality reward to assess the rationality and accuracy of the reasoning chain; and (3) a reflection reward to detect and revise errors. It also employs a difficulty-aware reweighting strategy and training sample filtering to boost performance on complex tasks. Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms prior RAG methods and generalizes well to single-hop tasks. The code and data are available at: this https URL. 

---
# Reducing Hallucinations in Summarization via Reinforcement Learning with Entity Hallucination Index 

**Authors**: Praveenkumar Katwe, Rakesh Chandra, Balabantaray Kali, Prasad Vittala  

**Link**: [PDF](https://arxiv.org/pdf/2507.22744)  

**Abstract**: Reducing hallucinations in abstractive summarization remains a critical challenge for deploying language models (LMs) in real-world settings. In this work, we introduce a rewarddriven fine-tuning framework that explicitly optimizes for Entity Hallucination Index (EHI), a metric designed to quantify the presence, correctness, and grounding of named entities in generated summaries. Given a corpus of meeting transcripts, we first generate baseline summaries using a pre-trained LM and compute EHI scores via automatic entity extraction and matching. We then apply reinforcement learning to fine-tune the model parameters, using EHI as a reward signal to bias generation toward entity-faithful outputs. Our approach does not rely on human-written factuality annotations, enabling scalable fine-tuning. Experiments demonstrate consistent improvements in EHI across datasets, with qualitative analysis revealing a significant reduction in entity-level hallucinations without degradation in fluency or informativeness. We release a reproducible Colab pipeline, facilitating further research on hallucination-aware model fine-tuning using lightweight, hallucintion metrics like EHI. 

---
# ControlMed: Adding Reasoning Control to Medical Language Model 

**Authors**: Sung-Min Lee, Siyoon Lee, Juyeon Kim, Kyungmin Roh  

**Link**: [PDF](https://arxiv.org/pdf/2507.22545)  

**Abstract**: Reasoning Large Language Models (LLMs) with enhanced accuracy and explainability are increasingly being adopted in the medical domain, as the life-critical nature of clinical decision-making demands reliable support. Despite these advancements, existing reasoning LLMs often generate unnecessarily lengthy reasoning processes, leading to significant computational overhead and response latency. These limitations hinder their practical deployment in real-world clinical environments. To address these challenges, we introduce \textbf{ControlMed}, a medical language model that enables users to actively control the length of the reasoning process at inference time through fine-grained control markers. ControlMed is trained through a three-stage pipeline: 1) pre-training on a large-scale synthetic medical instruction dataset covering both \textit{direct} and \textit{reasoning responses}; 2) supervised fine-tuning with multi-length reasoning data and explicit length-control markers; and 3) reinforcement learning with model-based reward signals to enhance factual accuracy and response quality. Experimental results on a variety of English and Korean medical benchmarks demonstrate that our model achieves similar or better performance compared to state-of-the-art models. Furthermore, users can flexibly balance reasoning accuracy and computational efficiency by controlling the reasoning length as needed. These findings demonstrate that ControlMed is a practical and adaptable solution for clinical question answering and medical information analysis. 

---
# RL from Teacher-Model Refinement: Gradual Imitation Learning for Machine Translation 

**Authors**: Dongyub Jude Lee, Zhenyi Ye, Pengcheng He  

**Link**: [PDF](https://arxiv.org/pdf/2507.22219)  

**Abstract**: Preference-learning methods for machine translation (MT)--such as Direct Preference Optimization (DPO)--have achieved impressive gains but depend heavily on large, carefully curated triplet datasets and often struggle to generalize beyond their tuning domains. We propose Reinforcement Learning from Teacher-Model Refinement (RLfR), a novel framework that removes reliance on static triplets by leveraging continuous, high-quality feedback from an external teacher model (GPT-4o). RLfR frames each translation step as a micro-tutorial: the actor generates a hypothesis, the teacher refines it, and the actor is rewarded based on how closely it aligns with the teacher's refinement. Guided by two complementary signals--(i) negative edit distance, promoting lexical and structural fidelity, and (ii) COMET score, ensuring semantic adequacy--the actor progressively learns to emulate the teacher, mirroring a human learning process through incremental, iterative improvement. On the FLORES-200 benchmark (English to and from German, Spanish, Chinese, Korean, and Japanese), RLfR consistently outperforms both MT-SFT and preference-based baselines, significantly improving COMET (semantic adequacy) and M-ETA (entity preservation) scores. 

---
# VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning 

**Authors**: Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, Hao Zhang, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22607)  

**Abstract**: Reinforcement learning has proven its effectiveness in enhancing the reasoning capabilities of large language models. Recent research efforts have progressively extended this paradigm to multimodal reasoning tasks. Due to the inherent complexity and diversity of multimodal tasks, especially in semantic content and problem formulations, existing models often exhibit unstable performance across various domains and difficulty levels. To address these limitations, we propose VL-Cogito, an advanced multimodal reasoning model trained via a novel multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL systematically guides the model through tasks of gradually increasing difficulty, substantially improving its reasoning abilities across diverse multimodal contexts. The framework introduces two key innovations: (1) an online difficulty soft weighting mechanism, dynamically adjusting training difficulty across successive RL training stages; and (2) a dynamic length reward mechanism, which encourages the model to adaptively regulate its reasoning path length according to task complexity, thus balancing reasoning efficiency with correctness. Experimental evaluations demonstrate that VL-Cogito consistently matches or surpasses existing reasoning-oriented models across mainstream multimodal benchmarks spanning mathematics, science, logic, and general understanding, validating the effectiveness of our approach. 

---
# RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents 

**Authors**: Zijing Zhang, Ziyang Chen, Mingxiao Li, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22844)  

**Abstract**: The development of autonomous agents for complex, long-horizon tasks is a central goal in AI. However, dominant training paradigms face a critical limitation: reinforcement learning (RL) methods that optimize solely for final task success often reinforce flawed or inefficient reasoning paths, a problem we term inefficient exploration. This leads to agents that are brittle and fail to generalize, as they learn to find solutions without learning how to reason coherently. To address this, we introduce RLVMR, a novel framework that integrates dense, process-level supervision into end-to-end RL by rewarding verifiable, meta-reasoning behaviors. RLVMR equips an agent to explicitly tag its cognitive steps, such as planning, exploration, and reflection, and provides programmatic, rule-based rewards for actions that contribute to effective problem-solving. These process-centric rewards are combined with the final outcome signal and optimized using a critic-free policy gradient method. On the challenging ALFWorld and ScienceWorld benchmarks, RLVMR achieves new state-of-the-art results, with our 7B model reaching an 83.6% success rate on the most difficult unseen task split. Our analysis confirms these gains stem from improved reasoning quality, including significant reductions in redundant actions and enhanced error recovery, leading to more robust, efficient, and interpretable agents. 

---
# G-Core: A Simple, Scalable and Balanced RLHF Trainer 

**Authors**: Junyu Wu, Weiming Chang, Xiaotao Liu, Guanyou He, Haoqiang Hong, Boqi Liu, Hongtao Tian, Tao Yang, Yunsheng Shi, Feng Lin, Ting Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22789)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become an increasingly popular paradigm for training large language models (LLMs) and diffusion models. While existing RLHF training systems have enabled significant progress, they often face challenges in scaling to multi-modal and diffusion workflows and adapting to dynamic workloads. In particular, current approaches may encounter limitations in controller scalability, flexible resource placement, and efficient orchestration when handling complex RLHF pipelines, especially in scenarios involving dynamic sampling or generative reward modeling. In this paper, we present \textbf{G-Core}, a simple, scalable, and balanced RLHF training framework designed to address these challenges. G-Core introduces a parallel controller programming model, enabling flexible and efficient orchestration of complex RLHF workflows without the bottlenecks of a single centralized controller. Furthermore, we propose a dynamic placement schema that adaptively partitions resources and schedules workloads, significantly reducing hardware idle time and improving utilization, even under highly variable training conditions. G-Core has successfully trained models that support WeChat product features serving a large-scale user base, demonstrating its effectiveness and robustness in real-world scenarios. Our results show that G-Core advances the state of the art in RLHF training, providing a solid foundation for future research and deployment of large-scale, human-aligned models. 

---
# RePaCA: Leveraging Reasoning Large Language Models for Static Automated Patch Correctness Assessment 

**Authors**: Marcos Fuster-Pena, David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2507.22580)  

**Abstract**: Automated Program Repair (APR) seeks to automatically correct software bugs without requiring human intervention. However, existing tools tend to generate patches that satisfy test cases without fixing the underlying bug, those are known as overfitting patches. To address this issue, Automated Patch Correctness Assessment (APCA) attempts to identify overfitting patches generated by APR tools. It can be solved as a static approach, meaning that no additional information is needed beyond the original and fixed code snippets. Current static techniques often struggle with reliability, flexibility and transparency. To address these issues, we introduce RePaCA, a novel static APCA technique that leverages Large Language Models (LLMs) specialized in thinking tasks. Our model is prompted with both buggy and fixed code snippets and guided to generate a Chain of Thought that analyses code differences, reasons about how the patch addresses the root cause, and ultimately provides a binary classification: correct or overfitting. To enhance these reasoning capabilities for the APCA task specifically, the LLM is finetuned using Reinforcement Learning with the Group Relative Policy Optimization algorithm. When evaluated on a standard Defects4J-derived test, our approach achieves state-of-the-art performance, with 83.1% accuracy and an 84.8% F1-score. Furthermore, our model demonstrates superior generalization capabilities when trained on different datasets, outperforming the leading technique. This reasoning capability also provides enhanced explainability for the patch assessment. These findings underscore the considerable promise of finetuned, reasoning LLMs to advance static APCA by enhancing accuracy, generalization, and explainability. 

---
# Promoting Online Safety by Simulating Unsafe Conversations with LLMs 

**Authors**: Owen Hoffman, Kangze Peng, Zehua You, Sajid Kamal, Sukrit Venkatagiri  

**Link**: [PDF](https://arxiv.org/pdf/2507.22267)  

**Abstract**: Generative AI, including large language models (LLMs) have the potential -- and already are being used -- to increase the speed, scale, and types of unsafe conversations online. LLMs lower the barrier for entry for bad actors to create unsafe conversations in particular because of their ability to generate persuasive and human-like text. In our current work, we explore ways to promote online safety by teaching people about unsafe conversations that can occur online with and without LLMs. We build on prior work that shows that LLMs can successfully simulate scam conversations. We also leverage research in the learning sciences that shows that providing feedback on one's hypothetical actions can promote learning. In particular, we focus on simulating scam conversations using LLMs. Our work incorporates two LLMs that converse with each other to simulate realistic, unsafe conversations that people may encounter online between a scammer LLM and a target LLM but users of our system are asked provide feedback to the target LLM. 

---
