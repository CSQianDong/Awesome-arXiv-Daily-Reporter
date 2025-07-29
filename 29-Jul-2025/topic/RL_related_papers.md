# PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training 

**Authors**: Sarat Chandra Bobbili, Ujwal Dinesha, Dheeraj Narasimha, Srinivas Shakkottai  

**Link**: [PDF](https://arxiv.org/pdf/2507.20067)  

**Abstract**: Inference-time alignment enables large language models (LLMs) to generate outputs aligned with end-user preferences without further training. Recent post-training methods achieve this by using small guidance models to modify token generation during inference. These methods typically optimize a reward function KL-regularized by the original LLM taken as the reference policy. A critical limitation, however, is their dependence on a pre-trained reward model, which requires fitting to human preference feedback--a potentially unstable process. In contrast, we introduce PITA, a novel framework that integrates preference feedback directly into the LLM's token generation, eliminating the need for a reward model. PITA learns a small preference-based guidance policy to modify token probabilities at inference time without LLM fine-tuning, reducing computational cost and bypassing the pre-trained reward model dependency. The problem is framed as identifying an underlying preference distribution, solved through stochastic search and iterative refinement of the preference-based guidance model. We evaluate PITA across diverse tasks, including mathematical reasoning and sentiment classification, demonstrating its effectiveness in aligning LLM outputs with user preferences. 

---
# The Policy Cliff: A Theoretical Analysis of Reward-Policy Maps in Large Language Models 

**Authors**: Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20150)  

**Abstract**: Reinforcement learning (RL) plays a crucial role in shaping the behavior of large language and reasoning models (LLMs/LRMs). However, it often produces brittle and unstable policies, leading to critical failures such as spurious reasoning, deceptive alignment, and instruction disobedience that undermine the trustworthiness and safety of LLMs/LRMs. Currently, these issues lack a unified theoretical explanation and are typically addressed using ad-hoc heuristics. This paper presents a rigorous mathematical framework for analyzing the stability of the mapping from a reward function to the optimal policy. We show that policy brittleness often stems from non-unique optimal actions, a common occurrence when multiple valid traces exist in a reasoning task. This theoretical lens provides a unified explanation for a range of seemingly disparate failures, reframing them as rational outcomes of optimizing rewards that may be incomplete or noisy, especially in the presence of action degeneracy. We extend this analysis from the fundamental single-reward setting to the more realistic multi-reward RL across diverse domains, showing how stability is governed by an "effective reward" aggregation mechanism. We also prove that entropy regularization restores policy stability at the cost of increased stochasticity. Our framework provides a unified explanation for recent empirical findings on deceptive reasoning, instruction-following trade-offs, and RLHF-induced sophistry, and is further validated through perturbation experiments in multi-reward RL. This work advances policy-stability analysis from empirical heuristics towards a principled theory, offering essential insights for designing safer and more trustworthy AI systems. 

---
# Alignment and Safety in Large Language Models: Safety Mechanisms, Training Paradigms, and Emerging Challenges 

**Authors**: Haoran Lu, Luyang Fang, Ruidong Zhang, Xinliang Li, Jiazhang Cai, Huimin Cheng, Lin Tang, Ziyu Liu, Zeliang Sun, Tao Wang, Yingchuan Zhang, Arif Hassan Zidan, Jinwen Xu, Jincheng Yu, Meizhi Yu, Hanqi Jiang, Xilin Gong, Weidi Luo, Bolun Sun, Yongkai Chen, Terry Ma, Shushan Wu, Yifan Zhou, Junhao Chen, Haotian Xiang, Jing Zhang, Afrar Jahin, Wei Ruan, Ke Deng, Yi Pan, Peilong Wang, Jiahui Li, Zhengliang Liu, Lu Zhang, Lin Zhao, Wei Liu, Dajiang Zhu, Xin Xing, Fei Dou, Wei Zhang, Chao Huang, Rongjie Liu, Mengrui Zhang, Yiwen Liu, Xiaoxiao Sun, Qin Lu, Zhen Xiang, Wenxuan Zhong, Tianming Liu, Ping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.19672)  

**Abstract**: Due to the remarkable capabilities and growing impact of large language models (LLMs), they have been deeply integrated into many aspects of society. Thus, ensuring their alignment with human values and intentions has emerged as a critical challenge. This survey provides a comprehensive overview of practical alignment techniques, training protocols, and empirical findings in LLM alignment. We analyze the development of alignment methods across diverse paradigms, characterizing the fundamental trade-offs between core alignment objectives. Our analysis shows that while supervised fine-tuning enables basic instruction-following, preference-based methods offer more flexibility for aligning with nuanced human intent. We discuss state-of-the-art techniques, including Direct Preference Optimization (DPO), Constitutional AI, brain-inspired methods, and alignment uncertainty quantification (AUQ), highlighting their approaches to balancing quality and efficiency. We review existing evaluation frameworks and benchmarking datasets, emphasizing limitations such as reward misspecification, distributional robustness, and scalable oversight. We summarize strategies adopted by leading AI labs to illustrate the current state of practice. We conclude by outlining open problems in oversight, value pluralism, robustness, and continuous alignment. This survey aims to inform both researchers and practitioners navigating the evolving landscape of LLM alignment. 

---
# JAM: A Tiny Flow-based Song Generator with Fine-grained Controllability and Aesthetic Alignment 

**Authors**: Renhang Liu, Chia-Yu Hung, Navonil Majumder, Taylor Gautreaux, Amir Ali Bagherzadeh, Chuan Li, Dorien Herremans, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2507.20880)  

**Abstract**: Diffusion and flow-matching models have revolutionized automatic text-to-audio generation in recent times. These models are increasingly capable of generating high quality and faithful audio outputs capturing to speech and acoustic events. However, there is still much room for improvement in creative audio generation that primarily involves music and songs. Recent open lyrics-to-song models, such as, DiffRhythm, ACE-Step, and LeVo, have set an acceptable standard in automatic song generation for recreational use. However, these models lack fine-grained word-level controllability often desired by musicians in their workflows. To the best of our knowledge, our flow-matching-based JAM is the first effort toward endowing word-level timing and duration control in song generation, allowing fine-grained vocal control. To enhance the quality of generated songs to better align with human preferences, we implement aesthetic alignment through Direct Preference Optimization, which iteratively refines the model using a synthetic dataset, eliminating the need or manual data annotations. Furthermore, we aim to standardize the evaluation of such lyrics-to-song models through our public evaluation dataset JAME. We show that JAM outperforms the existing models in terms of the music-specific attributes. 

---
# Aligning Large Language Model Agents with Rational and Moral Preferences: A Supervised Fine-Tuning Approach 

**Authors**: Wei Lu, Daniel L. Chen, Christian B. Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2507.20796)  

**Abstract**: Understanding how large language model (LLM) agents behave in strategic interactions is essential as these systems increasingly participate autonomously in economically and morally consequential decisions. We evaluate LLM preferences using canonical economic games, finding substantial deviations from human behavior. Models like GPT-4o show excessive cooperation and limited incentive sensitivity, while reasoning models, such as o3-mini, align more consistently with payoff-maximizing strategies. We propose a supervised fine-tuning pipeline that uses synthetic datasets derived from economic reasoning to align LLM agents with economic preferences, focusing on two stylized preference structures. In the first, utility depends only on individual payoffs (homo economicus), while utility also depends on a notion of Kantian universalizability in the second preference structure (homo moralis). We find that fine-tuning based on small datasets shifts LLM agent behavior toward the corresponding economic agent. We further assess the fine-tuned agents' behavior in two applications: Moral dilemmas involving autonomous vehicles and algorithmic pricing in competitive markets. These examples illustrate how different normative objectives embedded via realizations from structured preference structures can influence market and moral outcomes. This work contributes a replicable, cost-efficient, and economically grounded pipeline to align AI preferences using moral-economic principles. 

---
# Kimi K2: Open Agentic Intelligence 

**Authors**: Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T.Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, Haiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20534)  

**Abstract**: We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments.
Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, with strengths in agentic capabilities. Notably, K2 obtains 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, and 47.3 on SWE-Bench Multilingual -- surpassing most open and closed-sourced baselines in non-thinking settings. It also exhibits strong capabilities in coding, mathematics, and reasoning tasks, with a score of 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. These results position Kimi K2 as one of the most capable open-source large language models to date, particularly in software engineering and agentic tasks. We release our base and post-trained model checkpoints to facilitate future research and applications of agentic intelligence. 

---
# Cultivating Helpful, Personalized, and Creative AI Tutors: A Framework for Pedagogical Alignment using Reinforcement Learning 

**Authors**: Siyu Song, Wentao Liu, Ye Lu, Ruohua Zhang, Tao Liu, Jinze Lv, Xinyun Wang, Aimin Zhou, Fei Tan, Bo Jiang, Hao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20335)  

**Abstract**: The integration of large language models (LLMs) into education presents unprecedented opportunities for scalable personalized learning. However, standard LLMs often function as generic information providers, lacking alignment with fundamental pedagogical principles such as helpfulness, student-centered personalization, and creativity cultivation. To bridge this gap, we propose EduAlign, a novel framework designed to guide LLMs toward becoming more effective and responsible educational assistants. EduAlign consists of two main stages. In the first stage, we curate a dataset of 8k educational interactions and annotate them-both manually and automatically-along three key educational dimensions: Helpfulness, Personalization, and Creativity (HPC). These annotations are used to train HPC-RM, a multi-dimensional reward model capable of accurately scoring LLM outputs according to these educational principles. We further evaluate the consistency and reliability of this reward model. In the second stage, we leverage HPC-RM as a reward signal to fine-tune a pre-trained LLM using Group Relative Policy Optimization (GRPO) on a set of 2k diverse prompts. We then assess the pre- and post-finetuning models on both educational and general-domain benchmarks across the three HPC dimensions. Experimental results demonstrate that the fine-tuned model exhibits significantly improved alignment with pedagogical helpfulness, personalization, and creativity stimulation. This study presents a scalable and effective approach to aligning LLMs with nuanced and desirable educational traits, paving the way for the development of more engaging, pedagogically aligned AI tutors. 

---
# Post-Completion Learning for Language Models 

**Authors**: Xiang Fei, Siqi Wang, Shu Wei, Yuxiang Nie, Wei Shi, Hao Feng, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20252)  

**Abstract**: Current language model training paradigms typically terminate learning upon reaching the end-of-sequence (<eos>}) token, overlooking the potential learning opportunities in the post-completion space. We propose Post-Completion Learning (PCL), a novel training framework that systematically utilizes the sequence space after model output completion, to enhance both the reasoning and self-evaluation abilities. PCL enables models to continue generating self-assessments and reward predictions during training, while maintaining efficient inference by stopping at the completion point.
To fully utilize this post-completion space, we design a white-box reinforcement learning method: let the model evaluate the output content according to the reward rules, then calculate and align the score with the reward functions for supervision. We implement dual-track SFT to optimize both reasoning and evaluation capabilities, and mixed it with RL training to achieve multi-objective hybrid optimization.
Experimental results on different datasets and models demonstrate consistent improvements over traditional SFT and RL methods. Our method provides a new technical path for language model training that enhances output quality while preserving deployment efficiency. 

---
# SGPO: Self-Generated Preference Optimization based on Self-Improver 

**Authors**: Hyeonji Lee, Daejin Jo, Seohwan Yun, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.20181)  

**Abstract**: Large language models (LLMs), despite their extensive pretraining on diverse datasets, require effective alignment to human preferences for practical and reliable deployment. Conventional alignment methods typically employ off-policy learning and depend on human-annotated datasets, which limits their broad applicability and introduces distribution shift issues during training. To address these challenges, we propose Self-Generated Preference Optimization based on Self-Improver (SGPO), an innovative alignment framework that leverages an on-policy self-improving mechanism. Specifically, the improver refines responses from a policy model to self-generate preference data for direct preference optimization (DPO) of the policy model. Here, the improver and policy are unified into a single model, and in order to generate higher-quality preference data, this self-improver learns to make incremental yet discernible improvements to the current responses by referencing supervised fine-tuning outputs. Experimental results on AlpacaEval 2.0 and Arena-Hard show that the proposed SGPO significantly improves performance over DPO and baseline self-improving methods without using external preference data. 

---
# Sem-DPO: Mitigating Semantic Inconsistency in Preference Optimization for Prompt Engineering 

**Authors**: Anas Mohamed, Azal Ahmad Khan, Xinran Wang, Ahmad Faraz Khan, Shuwen Ge, Saman Bahzad Khan, Ayaan Ahmad, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2507.20133)  

**Abstract**: Generative AI can now synthesize strikingly realistic images from text, yet output quality remains highly sensitive to how prompts are phrased. Direct Preference Optimization (DPO) offers a lightweight, off-policy alternative to RL for automatic prompt engineering, but its token-level regularization leaves semantic inconsistency unchecked as prompts that win higher preference scores can still drift away from the user's intended meaning.
We introduce Sem-DPO, a variant of DPO that preserves semantic consistency yet retains its simplicity and efficiency. Sem-DPO scales the DPO loss by an exponential weight proportional to the cosine distance between the original prompt and winning candidate in embedding space, softly down-weighting training signals that would otherwise reward semantically mismatched prompts. We provide the first analytical bound on semantic drift for preference-tuned prompt generators, showing that Sem-DPO keeps learned prompts within a provably bounded neighborhood of the original text. On three standard text-to-image prompt-optimization benchmarks and two language models, Sem-DPO achieves 8-12% higher CLIP similarity and 5-9% higher human-preference scores (HPSv2.1, PickScore) than DPO, while also outperforming state-of-the-art baselines. These findings suggest that strong flat baselines augmented with semantic weighting should become the new standard for prompt-optimization studies and lay the groundwork for broader, semantics-aware preference optimization in language models. 

---
# Learning to Align Human Code Preferences 

**Authors**: Xin Yin, Chao Ni, Liushan Chen, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.20109)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in automating software development tasks. While recent advances leverage Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to align models with human preferences, the optimal training strategy remains unclear across diverse code preference scenarios. This paper systematically investigates the roles of SFT and DPO in aligning LLMs with different code preferences. Through both theoretical analysis and empirical observation, we hypothesize that SFT excels in scenarios with objectively verifiable optimal solutions, while applying SFT followed by DPO (S&D) enables models to explore superior solutions in scenarios without objectively verifiable optimal solutions. Based on the analysis and experimental evidence, we propose Adaptive Preference Optimization (APO), a dynamic integration approach that adaptively amplifies preferred responses, suppresses dispreferred ones, and encourages exploration of potentially superior solutions during training. Extensive experiments across six representative code preference tasks validate our theoretical hypotheses and demonstrate that APO consistently matches or surpasses the performance of existing SFT and S&D strategies. Our work provides both theoretical foundations and practical guidance for selecting appropriate training strategies in different code preference alignment scenarios. 

---
# Agentic Reinforced Policy Optimization 

**Authors**: Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19849)  

**Abstract**: Large-scale reinforcement learning with verifiable rewards (RLVR) has demonstrated its effectiveness in harnessing the potential of large language models (LLMs) for single-turn reasoning tasks. In realistic reasoning scenarios, LLMs can often utilize external tools to assist in task-solving processes. However, current RL algorithms inadequately balance the models' intrinsic long-horizon reasoning capabilities and their proficiency in multi-turn tool interactions. To bridge this gap, we propose Agentic Reinforced Policy Optimization (ARPO), a novel agentic RL algorithm tailored for training multi-turn LLM-based agents. Through preliminary experiments, we observe that LLMs tend to exhibit highly uncertain behavior, characterized by an increase in the entropy distribution of generated tokens, immediately following interactions with external tools. Motivated by this observation, ARPO incorporates an entropy-based adaptive rollout mechanism, dynamically balancing global trajectory sampling and step-level sampling, thereby promoting exploration at steps with high uncertainty after tool usage. By integrating an advantage attribution estimation, ARPO enables LLMs to internalize advantage differences in stepwise tool-use interactions. Our experiments across 13 challenging benchmarks in computational reasoning, knowledge reasoning, and deep search domains demonstrate ARPO's superiority over trajectory-level RL algorithms. Remarkably, ARPO achieves improved performance using only half of the tool-use budget required by existing methods, offering a scalable solution for aligning LLM-based agents with real-time dynamic environments. Our code and datasets are released at this https URL 

---
# UloRL:An Ultra-Long Output Reinforcement Learning Approach for Advancing Large Language Models' Reasoning Abilities 

**Authors**: Dong Du, Shulin Liu, Tao Yang, Shaohua Chen, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.19766)  

**Abstract**: Recent advances in large language models (LLMs) have highlighted the potential of reinforcement learning with verifiable rewards (RLVR) to enhance reasoning capabilities through extended output sequences. However, traditional RL frameworks face inefficiencies when handling ultra-long outputs due to long-tail sequence distributions and entropy collapse during training. To address these challenges, we propose an Ultra-Long Output Reinforcement Learning (UloRL) approach for advancing large language models' reasoning abilities. Specifically, we divide ultra long output decoding into short segments, enabling efficient training by mitigating delays caused by long-tail samples. Additionally, we introduce dynamic masking of well-Mastered Positive Tokens (MPTs) to prevent entropy collapse. Experimental results demonstrate the effectiveness of our approach. On the Qwen3-30B-A3B model, RL with segment rollout achieved 2.06x increase in training speed, while RL training with 128k-token outputs improves the model's performance on AIME2025 from 70.9\% to 85.1\% and on BeyondAIME from 50.7\% to 61.9\%, even surpassing Qwen3-235B-A22B with remarkable gains. These findings underscore the potential of our methods to advance the reasoning capabilities of LLMs with ultra-long sequence generation. We will release our code and model for further use by the community. 

---
# Geometric-Mean Policy Optimization 

**Authors**: Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, Fang Wan, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.20673)  

**Abstract**: Recent advancements, such as Group Relative Policy Optimization (GRPO), have enhanced the reasoning capabilities of large language models by optimizing the arithmetic mean of token-level rewards. However, GRPO suffers from unstable policy updates when processing tokens with outlier importance-weighted rewards, which manifests as extreme importance sampling ratios during training, i.e., the ratio between the sampling probabilities assigned to a token by the current and old policies. In this work, we propose Geometric-Mean Policy Optimization (GMPO), a stabilized variant of GRPO. Instead of optimizing the arithmetic mean, GMPO maximizes the geometric mean of token-level rewards, which is inherently less sensitive to outliers and maintains a more stable range of importance sampling ratio. In addition, we provide comprehensive theoretical and experimental analysis to justify the design and stability benefits of GMPO. Beyond improved stability, GMPO-7B outperforms GRPO by an average of 4.1% on multiple mathematical benchmarks and 1.4% on multimodal reasoning benchmark, including AIME24, AMC, MATH500, OlympiadBench, Minerva, and Geometry3K. Code is available at this https URL. 

---
# MoL-RL: Distilling Multi-Step Environmental Feedback into LLMs for Feedback-Independent Reasoning 

**Authors**: Kang Yang, Jingxue Chen, Qingkun Tang, Tianxiang Zhang, Qianchun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.20278)  

**Abstract**: Large language models (LLMs) face significant challenges in effectively leveraging sequential environmental feedback (EF) signals, such as natural language evaluations, for feedback-independent chain-of-thought (CoT) reasoning. Existing approaches either convert EF into scalar rewards, losing rich contextual information, or employ refinement datasets, failing to exploit the multi-step and discrete nature of EF interactions. To address these limitations, we propose MoL-RL, a novel training paradigm that integrates multi-step EF signals into LLMs through a dual-objective optimization framework. Our method combines MoL (Mixture-of-Losses) continual training, which decouples domain-specific EF signals (optimized via cross-entropy loss) and general language capabilities (preserved via Kullback-Leibler divergence), with GRPO-based post-training to distill sequential EF interactions into single-step inferences. This synergy enables robust feedback-independent reasoning without relying on external feedback loops. Experimental results on mathematical reasoning (MATH-500, AIME24/AIME25) and code generation (CodeAgent-Test) benchmarks demonstrate that MoL-RL achieves state-of-the-art performance with the Qwen3-8B model, while maintaining strong generalization across model scales (Qwen3-4B). This work provides a promising approach for leveraging multi-step textual feedback to enhance LLMs' reasoning capabilities in diverse domains. 

---
# Diversity-Enhanced Reasoning for Subjective Questions 

**Authors**: Yumeng Wang, Zhiyuan Fan, Jiayu Liu, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2507.20187)  

**Abstract**: Large reasoning models (LRM) with long chain-of-thought (CoT) capabilities have shown strong performance on objective tasks, such as math reasoning and coding. However, their effectiveness on subjective questions that may have different responses from different perspectives is still limited by a tendency towards homogeneous reasoning, introduced by the reliance on a single ground truth in supervised fine-tuning and verifiable reward in reinforcement learning. Motivated by the finding that increasing role perspectives consistently improves performance, we propose MultiRole-R1, a diversity-enhanced framework with multiple role perspectives, to improve the accuracy and diversity in subjective reasoning tasks. MultiRole-R1 features an unsupervised data construction pipeline that generates reasoning chains that incorporate diverse role perspectives. We further employ reinforcement learning via Group Relative Policy Optimization (GRPO) with reward shaping, by taking diversity as a reward signal in addition to the verifiable reward. With specially designed reward functions, we successfully promote perspective diversity and lexical diversity, uncovering a positive relation between reasoning diversity and accuracy. Our experiment on six benchmarks demonstrates MultiRole-R1's effectiveness and generalizability in enhancing both subjective and objective reasoning, showcasing the potential of diversity-enhanced training in LRMs. 

---
# JT-Math: A Multi-Stage Framework for Advanced Mathematical Reasoning in Large Language Models 

**Authors**: Yifan Hao, Fangning Chao, Yaqian Hao, Zhaojun Cui, Huan Bai, Haiyu Zhang, Yankai Liu, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.19748)  

**Abstract**: Mathematical reasoning is a cornerstone of artificial general intelligence and a primary benchmark for evaluating the capabilities of Large Language Models (LLMs). While state-of-the-art models show promise, they often falter when faced with complex problems that demand deep conceptual understanding and intricate, multi-step deliberation. To address this challenge, we introduce JT-Math-8B, a series of open-source models comprising base, instruct, and thinking versions, built upon a systematic, multi-stage optimization framework. Our pre-training corpus is a high-quality, 210B-token dataset curated through a dedicated data pipeline that uses model-based validation to ensure quality and diversity. The Instruct Model is optimized for direct, concise answers through Supervised Fine-Tuning (SFT) and a GRPO-based reinforcement learning (RL) method. The Thinking Model is trained for complex problem-solving using a Long Chain-of-Thought (Long CoT) approach, combining SFT with a novel, multi-stage RL curriculum that progressively increases task difficulty and context length up to 32K tokens. JT-Math-8B achieves state-of-the-art results among open-source models of similar size, surpassing prominent models like OpenAI's O1-mini and GPT-4o , and demonstrating superior performance on competition-level mathematics. 

---
