# Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization 

**Authors**: Wenqi Liu, Xuemeng Song, Jiaxi Li, Yinwei Wei, Na Zheng, Jianhua Yin, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.11712)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an effective approach for mitigating hallucination in Multimodal Large Language Models (MLLMs). Although existing methods have achieved significant progress by utilizing vision-oriented contrastive objectives for enhancing MLLMs' attention to visual inputs and hence reducing hallucination, they suffer from non-rigorous optimization objective function and indirect preference supervision. To address these limitations, we propose a Symmetric Multimodal Preference Optimization (SymMPO), which conducts symmetric preference learning with direct preference supervision (i.e., response pairs) for visual understanding enhancement, while maintaining rigorous theoretical alignment with standard DPO. In addition to conventional ordinal preference learning, SymMPO introduces a preference margin consistency loss to quantitatively regulate the preference gap between symmetric preference pairs. Comprehensive evaluation across five benchmarks demonstrate SymMPO's superior performance, validating its effectiveness in hallucination mitigation of MLLMs. 

---
# Schema-R1: A reasoning training approach for schema linking in Text-to-SQL Task 

**Authors**: Wuzhenghong Wen, Su Pan, yuwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11986)  

**Abstract**: Schema linking is a critical step in Text-to-SQL task, aiming to accurately predict the table names and column names required for the SQL query based on the given question. However, current fine-tuning approaches for schema linking models employ a rote-learning paradigm, excessively optimizing for ground truth schema linking outcomes while compromising reasoning ability. This limitation arises because of the difficulty in acquiring a high-quality reasoning sample for downstream tasks. To address this, we propose Schema-R1, a reasoning schema linking model trained using reinforcement learning. Specifically, Schema-R1 consists of three key steps: constructing small batches of high-quality reasoning samples, supervised fine-tuning for cold-start initialization, and rule-based reinforcement learning training. The final results demonstrate that our method effectively enhances the reasoning ability of the schema linking model, achieving a 10\% improvement in filter accuracy compared to the existing method. Our code is available at this https URL. 

---
# LLM-as-a-Fuzzy-Judge: Fine-Tuning Large Language Models as a Clinical Evaluation Judge with Fuzzy Logic 

**Authors**: Weibing Zheng, Laurah Turner, Jess Kropczynski, Murat Ozer, Tri Nguyen, Shane Halse  

**Link**: [PDF](https://arxiv.org/pdf/2506.11221)  

**Abstract**: Clinical communication skills are critical in medical education, and practicing and assessing clinical communication skills on a scale is challenging. Although LLM-powered clinical scenario simulations have shown promise in enhancing medical students' clinical practice, providing automated and scalable clinical evaluation that follows nuanced physician judgment is difficult. This paper combines fuzzy logic and Large Language Model (LLM) and proposes LLM-as-a-Fuzzy-Judge to address the challenge of aligning the automated evaluation of medical students' clinical skills with subjective physicians' preferences. LLM-as-a-Fuzzy-Judge is an approach that LLM is fine-tuned to evaluate medical students' utterances within student-AI patient conversation scripts based on human annotations from four fuzzy sets, including Professionalism, Medical Relevance, Ethical Behavior, and Contextual Distraction. The methodology of this paper started from data collection from the LLM-powered medical education system, data annotation based on multidimensional fuzzy sets, followed by prompt engineering and the supervised fine-tuning (SFT) of the pre-trained LLMs using these human annotations. The results show that the LLM-as-a-Fuzzy-Judge achieves over 80\% accuracy, with major criteria items over 90\%, effectively leveraging fuzzy logic and LLM as a solution to deliver interpretable, human-aligned assessment. This work suggests the viability of leveraging fuzzy logic and LLM to align with human preferences, advances automated evaluation in medical education, and supports more robust assessment and judgment practices. The GitHub repository of this work is available at this https URL 

---
# Configurable Preference Tuning with Rubric-Guided Synthetic Data 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2506.11702)  

**Abstract**: Models of human feedback for AI alignment, such as those underpinning Direct Preference Optimization (DPO), often bake in a singular, static set of preferences, limiting adaptability. This paper challenges the assumption of monolithic preferences by introducing Configurable Preference Tuning (CPT), a novel framework for endowing language models with the ability to dynamically adjust their behavior based on explicit, human-interpretable directives. CPT leverages synthetically generated preference data, conditioned on system prompts derived from structured, fine-grained rubrics that define desired attributes like writing style. By fine-tuning with these rubric-guided preferences, the LLM learns to modulate its outputs at inference time in response to the system prompt, without retraining. This approach not only offers fine-grained control but also provides a mechanism for modeling more nuanced and context-dependent human feedback. Several experimental artifacts, such as training code, generated datasets and fine-tuned models are released at this https URL 

---
# LearnAlign: Reasoning Data Selection for Reinforcement Learning in Large Language Models Based on Improved Gradient Alignment 

**Authors**: Shikun Li, Shipeng Li, Zhiqin Yang, Xinghua Zhang, Gaode Chen, Xiaobo Xia, Hengyu Liu, Zhe Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11480)  

**Abstract**: Reinforcement learning (RL) has become a key technique for enhancing LLMs' reasoning abilities, yet its data inefficiency remains a major bottleneck. To address this critical yet challenging issue, we present a novel gradient-alignment-based method, named LearnAlign, which intelligently selects the learnable and representative training reasoning data for RL post-training. To overcome the well-known issue of response-length bias in gradient norms, we introduce the data learnability based on the success rate, which can indicate the learning potential of each data point. Experiments across three mathematical reasoning benchmarks demonstrate that our method significantly reduces training data requirements while achieving minor performance degradation or even improving performance compared to full-data training. For example, it reduces data requirements by up to 1,000 data points with better performance (77.53%) than that on the full dataset on GSM8K benchmark (77.04%). Furthermore, we show its effectiveness in the staged RL setting. This work provides valuable insights into data-efficient RL post-training and establishes a foundation for future research in optimizing reasoning data this http URL facilitate future work, we will release code. 

---
# Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards 

**Authors**: Jeff Da, Clinton Wang, Xiang Deng, Yuntao Ma, Nikhil Barhate, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2506.11425)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has been widely adopted as the de facto method for enhancing the reasoning capabilities of large language models and has demonstrated notable success in verifiable domains like math and competitive programming tasks. However, the efficacy of RLVR diminishes significantly when applied to agentic environments. These settings, characterized by multi-step, complex problem solving, lead to high failure rates even for frontier LLMs, as the reward landscape is too sparse for effective model training via conventional RLVR. In this work, we introduce Agent-RLVR, a framework that makes RLVR effective in challenging agentic settings, with an initial focus on software engineering tasks. Inspired by human pedagogy, Agent-RLVR introduces agent guidance, a mechanism that actively steers the agent towards successful trajectories by leveraging diverse informational cues. These cues, ranging from high-level strategic plans to dynamic feedback on the agent's errors and environmental interactions, emulate a teacher's guidance, enabling the agent to navigate difficult solution spaces and promotes active self-improvement via additional environment exploration. In the Agent-RLVR training loop, agents first attempt to solve tasks to produce initial trajectories, which are then validated by unit tests and supplemented with agent guidance. Agents then reattempt with guidance, and the agent policy is updated with RLVR based on the rewards of these guided trajectories. Agent-RLVR elevates the pass@1 performance of Qwen-2.5-72B-Instruct from 9.4% to 22.4% on SWE-Bench Verified. We find that our guidance-augmented RLVR data is additionally useful for test-time reward model training, shown by further boosting pass@1 to 27.8%. Agent-RLVR lays the groundwork for training agents with RLVR in complex, real-world environments where conventional RL methods struggle. 

---
# Debiasing Online Preference Learning via Preference Feature Preservation 

**Authors**: Dongyoung Kim, Jinsung Yoon, Jinwoo Shin, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.11098)  

**Abstract**: Recent preference learning frameworks for large language models (LLMs) simplify human preferences with binary pairwise comparisons and scalar rewards. This simplification could make LLMs' responses biased to mostly preferred features, and would be exacerbated during the iterations of online preference learning steps. To address these challenges, we propose a novel framework coined PFP (Preference Feature Preservation). The key idea of PFP is maintaining the distribution of human preference features and utilizing such rich signals throughout the online preference learning process. Specifically, PFP first extract preference features from offline pairwise human preference data and trains a feature classifier. Then, using trained classifier and the distribution preserving optimization, PFP maps appropriate preference features for a new input instruction during online learning. Lastly, PFP trains LLM using the existing preference learning method, by incorporating the preference feature into system prompts and enabling LLM to explicitly handle various human preferences. Our experiments demonstrate that PFP successfully mitigates the bias in preference features during online learning, and hence achieves superior performance compared to previous preference learning methods on standard benchmarks to evaluate LLM alignment. 

---
# From Reasoning to Code: GRPO Optimization for Underrepresented Languages 

**Authors**: Federico Pennino, Bianca Raimondi, Massimo Rondelli, Andrea Gurioli, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2506.11027)  

**Abstract**: Generating accurate and executable code using large language models (LLMs) is challenging for languages with limited public training data compared to popular languages such as Python. This paper introduces a generalizable approach that uses small-scale code versions of the Qwen 2.5 model combined with Group Relative Policy Optimization (GRPO) to enable effective code generation through explicit reasoning steps, which is particularly beneficial for languages with smaller source code databases. Using Prolog as a representative use case -- given its limited online presence -- the initial model faced challenges in generating executable code. After some training steps, the model successfully produces logically consistent and syntactically accurate code by directly integrating reasoning-driven feedback into the reinforcement learning loop. Experimental evaluations using mathematical logic problem benchmarks illustrate significant improvements in reasoning quality, code accuracy, and logical correctness, underscoring the potential of this approach to benefit a wide range of programming languages lacking extensive training resources. 

---
# TongSearch-QR: Reinforced Query Reasoning for Retrieval 

**Authors**: Xubo Qin, Jun Bai, Jiaqi Li, Zixia Jia, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11603)  

**Abstract**: Traditional information retrieval (IR) methods excel at textual and semantic matching but struggle in reasoning-intensive retrieval tasks that require multi-hop inference or complex semantic understanding between queries and documents. One promising solution is to explicitly rewrite or augment queries using large language models (LLMs) to elicit reasoning-relevant content prior to retrieval. However, the widespread use of large-scale language models like GPT-4 or LLaMA3-70B remains impractical due to their high inference cost and limited deployability in real-world systems. In this work, we introduce TongSearch QR (Previously Known as "TongSearch Reasoner"), a family of small-scale language models for query reasoning and rewriting in reasoning-intensive retrieval. With a novel semi-rule-based reward function, we employ reinforcement learning approaches enabling smaller language models, e,g, Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct, to achieve query reasoning performance rivaling large-scale language models without their prohibitive inference costs. Experiment results on BRIGHT benchmark show that with BM25 as retrievers, both TongSearch QR-7B and TongSearch QR-1.5B models significantly outperform existing baselines, including prompt-based query reasoners and some latest dense retrievers trained for reasoning-intensive retrieval tasks, offering superior adaptability for real-world deployment. 

---
# Learning a Continue-Thinking Token for Enhanced Test-Time Scaling 

**Authors**: Liran Ringel, Elad Tolochinsky, Yaniv Romano  

**Link**: [PDF](https://arxiv.org/pdf/2506.11274)  

**Abstract**: Test-time scaling has emerged as an effective approach for improving language model performance by utilizing additional compute at inference time. Recent studies have shown that overriding end-of-thinking tokens (e.g., replacing "</think>" with "Wait") can extend reasoning steps and improve accuracy. In this work, we explore whether a dedicated continue-thinking token can be learned to trigger extended reasoning. We augment a distilled version of DeepSeek-R1 with a single learned "<|continue-thinking|>" token, training only its embedding via reinforcement learning while keeping the model weights frozen. Our experiments show that this learned token achieves improved accuracy on standard math benchmarks compared to both the baseline model and a test-time scaling approach that uses a fixed token (e.g., "Wait") for budget forcing. In particular, we observe that in cases where the fixed-token approach enhances the base model's accuracy, our method achieves a markedly greater improvement. For example, on the GSM8K benchmark, the fixed-token approach yields a 1.3% absolute improvement in accuracy, whereas our learned-token method achieves a 4.2% improvement over the base model that does not use budget forcing. 

---
# History-Aware Cross-Attention Reinforcement: Self-Supervised Multi Turn and Chain-of-Thought Fine-Tuning with vLLM 

**Authors**: Andrew Kiruluta, Andreas Lemos, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2506.11108)  

**Abstract**: We present CAGSR-vLLM-MTC, an extension of our Self-Supervised Cross-Attention-Guided Reinforcement (CAGSR) framework, now implemented on the high-performance vLLM runtime, to address both multi-turn dialogue and chain-of-thought reasoning. Building upon our original single-turn approach, we first instrumented vLLM's C++/CUDA kernels to asynchronously capture per-layer, per-head cross-attention weights during generation. We then generalized our self-supervised reward function to accumulate attention signals over entire conversation histories and intermediate chain-of-thought steps. We discuss practical trade-offs, including an entropy-based clamping mechanism to prevent attention collapse on early context, and outline future directions for multi-party dialogues and hierarchical reasoning. 

---
# Customizing Speech Recognition Model with Large Language Model Feedback 

**Authors**: Shaoshi Ling, Guoli Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.11091)  

**Abstract**: Automatic speech recognition (ASR) systems have achieved strong performance on general transcription tasks. However, they continue to struggle with recognizing rare named entities and adapting to domain mismatches. In contrast, large language models (LLMs), trained on massive internet-scale datasets, are often more effective across a wide range of domains. In this work, we propose a reinforcement learning based approach for unsupervised domain adaptation, leveraging unlabeled data to enhance transcription quality, particularly the named entities affected by domain mismatch, through feedback from a LLM. Given contextual information, our framework employs a LLM as the reward model to score the hypotheses from the ASR model. These scores serve as reward signals to fine-tune the ASR model via reinforcement learning. Our method achieves a 21\% improvement on entity word error rate over conventional self-training methods. 

---
# SAGE:Specification-Aware Grammar Extraction for Automated Test Case Generation with LLMs 

**Authors**: Aditi, Hyunwoo Park, Sicheol Sung, Yo-Sub Han, Sang-Ki Ko  

**Link**: [PDF](https://arxiv.org/pdf/2506.11081)  

**Abstract**: Grammar-based test case generation has proven effective for competitive programming problems, but generating valid and general grammars from natural language specifications remains a key challenge, especially under limited supervision. Context-Free Grammars with Counters (CCFGs) have recently been introduced as a formalism to represent such specifications with logical constraints by storing and reusing counter values during derivation. In this work, we explore the use of open-source large language models (LLMs) to induce CCFGs from specifications using a small number of labeled examples and verifiable reward-guided reinforcement learning. Our approach first fine-tunes an open-source LLM to perform specification-to-grammar translation, and further applies Group Relative Policy Optimization (GRPO) to enhance grammar validity and generality. We also examine the effectiveness of iterative feedback for open and closed-source LLMs in correcting syntactic and semantic errors in generated grammars.
Experimental results show that our approach SAGE achieves stronger generalization and outperforms 17 open and closed-source LLMs in both grammar quality and test effectiveness, improving over the state-of-the-art by 15.92%p in grammar validity and 12.34%p in test effectiveness. We provide our implementation and dataset at the following anonymous repository:this https URL 

---
# TreeRL: LLM Reinforcement Learning with On-Policy Tree Search 

**Authors**: Zhenyu Hou, Ziniu Hu, Yujiang Li, Rui Lu, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11902)  

**Abstract**: Reinforcement learning (RL) with tree search has demonstrated superior performance in traditional reasoning tasks. Compared to conventional independent chain sampling strategies with outcome supervision, tree search enables better exploration of the reasoning space and provides dense, on-policy process rewards during RL training but remains under-explored in On-Policy LLM RL. We propose TreeRL, a reinforcement learning framework that directly incorporates on-policy tree search for RL training. Our approach includes intermediate supervision and eliminates the need for a separate reward model training. Existing approaches typically train a separate process reward model, which can suffer from distribution mismatch and reward hacking. We also introduce a cost-effective tree search approach that achieves higher search efficiency under the same generation token budget by strategically branching from high-uncertainty intermediate steps rather than using random branching. Experiments on challenging math and code reasoning benchmarks demonstrate that TreeRL achieves superior performance compared to traditional ChainRL, highlighting the potential of tree search for LLM. TreeRL is open-sourced at this https URL. 

---
