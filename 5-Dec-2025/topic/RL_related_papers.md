# Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning 

**Authors**: Hongye Cao, Zhixin Bai, Ziyue Peng, Boyan Wang, Tianpei Yang, Jing Huo, Yuyao Zhang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2512.04359)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning. 

---
# Balancing Safety and Helpfulness in Healthcare AI Assistants through Iterative Preference Alignment 

**Authors**: Huy Nghiem, Swetasudha Panda, Devashish Khatwani, Huy V. Nguyen, Krishnaram Kenthapadi, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2512.04210)  

**Abstract**: Large Language Models (LLMs) are increasingly used in healthcare, yet ensuring their safety and trustworthiness remains a barrier to deployment. Conversational medical assistants must avoid unsafe compliance without over-refusing benign queries. We present an iterative post-deployment alignment framework that applies Kahneman-Tversky Optimization (KTO) and Direct Preference Optimization (DPO) to refine models against domain-specific safety signals. Using the CARES-18K benchmark for adversarial robustness, we evaluate four LLMs (Llama-3B/8B, Meditron-8B, Mistral-7B) across multiple cycles. Our results show up to 42% improvement in safety-related metrics for harmful query detection, alongside interesting trade-offs against erroneous refusals, thereby exposing architecture-dependent calibration biases. We also perform ablation studies to identify when self-evaluation is reliable and when external or finetuned judges are necessary to maximize performance gains. Our findings underscore the importance of adopting best practices that balance patient safety, user trust, and clinical utility in the design of conversational medical assistants. 

---
# SA-IQA: Redefining Image Quality Assessment for Spatial Aesthetics with Multi-Dimensional Rewards 

**Authors**: Yuan Gao, Jin Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.05098)  

**Abstract**: In recent years, Image Quality Assessment (IQA) for AI-generated images (AIGI) has advanced rapidly; however, existing methods primarily target portraits and artistic images, lacking a systematic evaluation of interior scenes. We introduce Spatial Aesthetics, a paradigm that assesses the aesthetic quality of interior images along four dimensions: layout, harmony, lighting, and distortion. We construct SA-BENCH, the first benchmark for spatial aesthetics, comprising 18,000 images and 50,000 precise annotations. Employing SA-BENCH, we systematically evaluate current IQA methodologies and develop SA-IQA, through MLLM fine-tuning and a multidimensional fusion approach, as a comprehensive reward framework for assessing spatial aesthetics. We apply SA-IQA to two downstream tasks: (1) serving as a reward signal integrated with GRPO reinforcement learning to optimize the AIGC generation pipeline, and (2) Best-of-N selection to filter high-quality images and improve generation quality. Experiments indicate that SA-IQA significantly outperforms existing methods on SA-BENCH, setting a new standard for spatial aesthetics evaluation. Code and dataset will be open-sourced to advance research and applications in this domain. 

---
# Semantic Soft Bootstrapping: Long Context Reasoning in LLMs without Reinforcement Learning 

**Authors**: Purbesh Mitra, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2512.05105)  

**Abstract**: Long context reasoning in large language models (LLMs) has demonstrated enhancement of their cognitive capabilities via chain-of-thought (CoT) inference. Training such models is usually done via reinforcement learning with verifiable rewards (RLVR) in reasoning based problems, like math and programming. However, RLVR is limited by several bottlenecks, such as, lack of dense reward, and inadequate sample efficiency. As a result, it requires significant compute resources in post-training phase. To overcome these limitations, in this work, we propose \textbf{Semantic Soft Bootstrapping (SSB)}, a self-distillation technique, in which the same base language model plays the role of both teacher and student, but receives different semantic contexts about the correctness of its outcome at training time. The model is first prompted with a math problem and several rollouts are generated. From them, the correct and most common incorrect response are filtered, and then provided to the model in context to produce a more robust, step-by-step explanation with a verified final answer. This pipeline automatically curates a paired teacher-student training set from raw problem-answer data, without any human intervention. This generation process also produces a sequence of logits, which is what the student model tries to match in the training phase just from the bare question alone. In our experiment, Qwen2.5-3B-Instruct on GSM8K dataset via parameter-efficient fine-tuning. We then tested its accuracy on MATH500, and AIME2024 benchmarks. Our experiments show a jump of 10.6%, and 10% improvements in accuracy, respectively, over group relative policy optimization (GRPO), which is a commonly used RLVR algorithm. Our code is available at this https URL, and the model, curated dataset is available at this https URL. 

---
# RRPO: Robust Reward Policy Optimization for LLM-based Emotional TTS 

**Authors**: Cong Wang, Changfeng Gao, Yang Xiang, Zhihao Du, Keyu An, Han Zhao, Qian Chen, Xiangang Li, Yingming Gao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.04552)  

**Abstract**: Differentiable reinforcement learning (RL) frameworks like DiffRO offer a powerful approach for controllable text-to-speech (TTS), but are vulnerable to reward hacking, particularly for nuanced tasks like emotion control. The policy model can exploit a vanilla Reward Model (RM) by generating acoustic artifacts to achieve spurious rewards, but at the cost of degrading perceptual quality. To address this, we propose Robust Reward Policy Optimization (RRPO), a novel framework that employs a hybrid regularization scheme. This scheme develops a robust RM whose reward signal is more reliably aligned with human perception, compelling the policy to abandon detrimental shortcuts and instead learn the complex features of genuine emotions. Our ablation study confirms the enhanced robustness of our RM, as evidenced by its strong cross-lingual generalization. The subjective evaluation demonstrates that this robust RM effectively mitigates reward hacking, leading to significant improvements in both emotional expressiveness and naturalness over all baselines. Demo page: this https URL. 

---
# UW-BioNLP at ChemoTimelines 2025: Thinking, Fine-Tuning, and Dictionary-Enhanced LLM Systems for Chemotherapy Timeline Extraction 

**Authors**: Tianmai M. Zhang, Zhaoyi Sun, Sihang Zeng, Chenxi Li, Neil F. Abernethy, Barbara D. Lam, Fei Xia, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2512.04518)  

**Abstract**: The ChemoTimelines shared task benchmarks methods for constructing timelines of systemic anticancer treatment from electronic health records of cancer patients. This paper describes our methods, results, and findings for subtask 2 -- generating patient chemotherapy timelines from raw clinical notes. We evaluated strategies involving chain-of-thought thinking, supervised fine-tuning, direct preference optimization, and dictionary-based lookup to improve timeline extraction. All of our approaches followed a two-step workflow, wherein an LLM first extracted chemotherapy events from individual clinical notes, and then an algorithm normalized and aggregated events into patient-level timelines. Each specific method differed in how the associated LLM was utilized and trained. Multiple approaches yielded competitive performances on the test set leaderboard, with fine-tuned Qwen3-14B achieving the best official score of 0.678. Our results and analyses could provide useful insights for future attempts on this task as well as the design of similar tasks. 

---
# Natural Language Actor-Critic: Scalable Off-Policy Learning in Language Space 

**Authors**: Joey Hong, Kang Liu, Zhan Ling, Jiecao Chen, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2512.04601)  

**Abstract**: Large language model (LLM) agents -- LLMs that dynamically interact with an environment over long horizons -- have become an increasingly important area of research, enabling automation in complex tasks involving tool-use, web browsing, and dialogue with people. In the absence of expert demonstrations, training LLM agents has relied on policy gradient methods that optimize LLM policies with respect to an (often sparse) reward function. However, in long-horizon tasks with sparse rewards, learning from trajectory-level rewards can be noisy, leading to training that is unstable and has high sample complexity. Furthermore, policy improvement hinges on discovering better actions through exploration, which can be difficult when actions lie in natural language space. In this paper, we propose Natural Language Actor-Critic (NLAC), a novel actor-critic algorithm that trains LLM policies using a generative LLM critic that produces natural language rather than scalar values. This approach leverages the inherent strengths of LLMs to provide a richer and more actionable training signal; particularly, in tasks with large, open-ended action spaces, natural language explanations for why an action is suboptimal can be immensely useful for LLM policies to reason how to improve their actions, without relying on random exploration. Furthermore, our approach can be trained off-policy without policy gradients, offering a more data-efficient and stable alternative to existing on-policy methods. We present results on a mixture of reasoning, web browsing, and tool-use with dialogue tasks, demonstrating that NLAC shows promise in outperforming existing training approaches and offers a more scalable and stable training paradigm for LLM agents. 

---
