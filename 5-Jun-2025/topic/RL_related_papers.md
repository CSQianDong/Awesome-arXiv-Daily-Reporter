# R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning 

**Authors**: Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha, Limin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04185)  

**Abstract**: Large language models (LLMs) have notably progressed in multi-step and long-chain reasoning. However, extending their reasoning capabilities to encompass deep interactions with search remains a non-trivial challenge, as models often fail to identify optimal reasoning-search interaction trajectories, resulting in suboptimal responses. We propose R-Search, a novel reinforcement learning framework for Reasoning-Search integration, designed to enable LLMs to autonomously execute multi-step reasoning with deep search interaction, and learn optimal reasoning search interaction trajectories via multi-reward signals, improving response quality in complex logic- and knowledge-intensive tasks. R-Search guides the LLM to dynamically decide when to retrieve or reason, while globally integrating key evidence to enhance deep knowledge interaction between reasoning and search. During RL training, R-Search provides multi-stage, multi-type rewards to jointly optimize the reasoning-search trajectory. Experiments on seven datasets show that R-Search outperforms advanced RAG baselines by up to 32.2% (in-domain) and 25.1% (out-of-domain). The code and data are available at this https URL. 

---
# SuperWriter: Reflection-Driven Long-Form Generation with Large Language Models 

**Authors**: Yuhao Wu, Yushi Bai, Zhiqiang Hu, Juanzi Li, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.04180)  

**Abstract**: Long-form text generation remains a significant challenge for large language models (LLMs), particularly in maintaining coherence, ensuring logical consistency, and preserving text quality as sequence length increases. To address these limitations, we propose SuperWriter-Agent, an agent-based framework designed to enhance the quality and consistency of long-form text generation. SuperWriter-Agent introduces explicit structured thinking-through planning and refinement stages into the generation pipeline, guiding the model to follow a more deliberate and cognitively grounded process akin to that of a professional writer. Based on this framework, we construct a supervised fine-tuning dataset to train a 7B SuperWriter-LM. We further develop a hierarchical Direct Preference Optimization (DPO) procedure that uses Monte Carlo Tree Search (MCTS) to propagate final quality assessments and optimize each generation step accordingly. Empirical results across diverse benchmarks demonstrate that SuperWriter-LM achieves state-of-the-art performance, surpassing even larger-scale baseline models in both automatic evaluation and human evaluation. Furthermore, comprehensive ablation studies demonstrate the effectiveness of hierarchical DPO and underscore the value of incorporating structured thinking steps to improve the quality of long-form text generation. 

---
# Robust Preference Optimization via Dynamic Target Margins 

**Authors**: Jie Sun, Junkang Wu, Jiancan Wu, Zhibo Zhu, Xingyu Lu, Jun Zhou, Lintao Ma, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03690)  

**Abstract**: The alignment of Large Language Models (LLMs) is crucial for ensuring their safety and reliability in practical applications. Direct Preference Optimization (DPO) has emerged as an efficient method that directly optimizes models using preference pairs, significantly reducing resource demands. However, the effectiveness of DPO heavily depends on the data quality, which is frequently compromised by noise. In this work, we propose $\gamma$-PO, a dynamic target margin preference optimization algorithm that adjust reward margins at the pairwise level. By introducing instance-specific margin calibration, $\gamma$-PO strategically prioritizes high-confidence pairs (those demonstrating higher reward margins) while suppressing potential noise from ambiguous pairs. Moreover, $\gamma$-PO is a plug-and-play method, compatible with variants of DPO that rely on reward margin between preference pairs. Across benchmarks such as AlpacaEval2 and Arena-Hard, $\gamma$-PO achieves an average 4.4\% improvement over other baselines, setting new benchmarks for state-of-the-art performance. Additionally, $\gamma$-PO requires minimal code changes and has a negligible impact on training efficiency, making it a robust solution for enhancing LLMs alignment. Our codes are available at \href{this https URL}{this https URL}. 

---
# RewardAnything: Generalizable Principle-Following Reward Models 

**Authors**: Zhuohao Yu, Jiali Zeng, Weizheng Gu, Yidong Wang, Jindong Wang, Fandong Meng, Jie Zhou, Yue Zhang, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.03637)  

**Abstract**: Reward Models, essential for guiding Large Language Model optimization, are typically trained on fixed preference datasets, resulting in rigid alignment to single, implicit preference distributions. This prevents adaptation to diverse real-world needs-from conciseness in one task to detailed explanations in another. The standard practice of collecting task-specific preference data and retraining reward models is resource-intensive, often producing biased rewards, and limits practical application. We introduce generalizable, principle-following reward models. We propose that RMs should understand and adhere to dynamically provided natural language specifications of reward principles, similar to instruction-following in LLMs. To measure this capability, we develop RABench, a comprehensive benchmark for RMs focusing on generalization across diverse principles. Evaluations on RABench reveal poor generalization of current RMs. As a solution, we present RewardAnything, a novel RM designed and trained to explicitly follow natural language principles. We achieve SotA performance with RewardAnything in traditional RM benchmark simply by specifying a well-defined principle, and results on RABench show we excel in adapting to novel principles without retraining. Furthermore, RewardAnything integrates seamlessly with existing RLHF methods and we show by a case study on how to automatically and efficiently align LLMs with only natural language principles. 

---
# MiMo-VL Technical Report 

**Authors**: Xiaomi LLM-Core Team, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zihan Jiang, Zhixian Zheng, Zhichao Song, Zhenbo Luo, Yue Yu, Yudong Wang, Yuanyuan Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xinzhe Xu, Xingchen Song, Xing Zhang, Xing Yong, Xin Zhang, Xiangwei Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shihua Yu, Shaohui Liu, Shande Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kangyang Zhou, Kang Zhou, Kai Fang, Jun Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongshen Xu, Heng Qu, Haochen Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong Ma, Chang Liu, Can Cai, Bingquan Xia  

**Link**: [PDF](https://arxiv.org/pdf/2506.03569)  

**Abstract**: We open-source MiMo-VL-7B-SFT and MiMo-VL-7B-RL, two powerful vision-language models delivering state-of-the-art performance in both general visual understanding and multimodal reasoning. MiMo-VL-7B-RL outperforms Qwen2.5-VL-7B on 35 out of 40 evaluated tasks, and scores 59.4 on OlympiadBench, surpassing models with up to 78B parameters. For GUI grounding applications, it sets a new standard with 56.1 on OSWorld-G, even outperforming specialized models such as UI-TARS. Our training combines four-stage pre-training (2.4 trillion tokens) with Mixed On-policy Reinforcement Learning (MORL) integrating diverse reward signals. We identify the importance of incorporating high-quality reasoning data with long Chain-of-Thought into pre-training stages, and the benefits of mixed RL despite challenges in simultaneous multi-domain optimization. We also contribute a comprehensive evaluation suite covering 50+ tasks to promote reproducibility and advance the field. The model checkpoints and full evaluation suite are available at this https URL. 

---
# BPO: Revisiting Preference Modeling in Direct Preference Optimization 

**Authors**: Lin Sun, Chuang Liu, Peng Liu, Bingyang Li, Weijia Lu, Ning Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03557)  

**Abstract**: Direct Preference Optimization (DPO) have emerged as a popular method for aligning Large Language Models (LLMs) with human preferences. While DPO effectively preserves the relative ordering between chosen and rejected responses through pairwise ranking losses, it often neglects absolute reward magnitudes. This oversight can decrease the likelihood of chosen responses and increase the risk of generating out-of-distribution responses, leading to poor performance. We term this issue Degraded Chosen Responses (DCR).To address this issue, we propose Balanced Preference Optimization (BPO), a novel framework that dynamically balances the optimization of chosen and rejected responses through two key components: balanced reward margin and gap adaptor. Unlike previous methods, BPO can fundamentally resolve DPO's DCR issue, without introducing additional constraints to the loss function. Experimental results on multiple mathematical reasoning tasks show that BPO significantly outperforms DPO, improving accuracy by +10.1% with Llama-3.1-8B-Instruct (18.8% to 28.9%) and +11.7% with Qwen2.5-Math-7B (35.0% to 46.7%). It also surpasses DPO variants by +3.6% over IPO (43.1%), +5.0% over SLiC (41.7%), and +3.1% over Cal-DPO (43.6%) on the same model. Remarkably, our algorithm requires only a single line of code modification, making it simple to implement and fully compatible with existing DPO-based frameworks. 

---
# Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement 

**Authors**: Xiaofeng Zhou, Heyan Huang, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03541)  

**Abstract**: Large Language Models (LLMs) continue to set new standards in knowledge-intensive and complex reasoning tasks, yet their high computational demands limit widespread adoption. While distilling large models into smaller ones offers a sustainable solution, current techniques--such as static knowledge distillation, resource-intensive reinforcement learning from human feedback, or limited self-reflection--struggle to yield substantial and lasting performance gains. In this paper, we present a novel Debate and Reflect (D&R) framework that orchestrates multi-turn debates between smaller models and stronger teacher models, eliciting actionable feedback (e.g., error analysis, corrective strategies) to guide student models. Further, we introduce Tree-structured Direct Preference Optimization (T-DPO) to efficiently leverage these debate logs, organizing interactions into a hierarchical format for effective training. Empirical evaluations across diverse NLP benchmarks demonstrate that our approach significantly improves smaller-model accuracy, robustness, and generalization, outperforming conventional baselines by a large margin. 

---
# Seed-Coder: Let the Code Model Curate Data for Itself 

**Authors**: Yuyu Zhang, Jing Su, Yifan Sun, Chenguang Xi, Xia Xiao, Shen Zheng, Anxiang Zhang, Kaibo Liu, Daoguang Zan, Tao Sun, Jinhua Zhu, Shulin Xin, Dong Huang, Yetao Bai, Lixin Dong, Chao Li, Jianchong Chen, Hanzhi Zhou, Yifan Huang, Guanghan Ning, Xierui Song, Jiaze Chen, Siyao Liu, Kai Shen, Liang Xiang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03524)  

**Abstract**: Code data in large language model (LLM) pretraining is recognized crucial not only for code-related tasks but also for enhancing general intelligence of LLMs. Current open-source LLMs often heavily rely on human effort to produce their code pretraining data, such as employing hand-crafted filtering rules tailored to individual programming languages, or using human-annotated data to train quality filters. However, these approaches are inherently limited in scalability, prone to subjective biases, and costly to extend and maintain across diverse programming languages. To address these challenges, we introduce Seed-Coder, a series of open-source LLMs comprising base, instruct and reasoning models of 8B size, minimizing human involvement in data construction. Our code pretraining data is produced by a model-centric data pipeline, which predominantly leverages LLMs for scoring and filtering code data. The instruct model is further trained via supervised fine-tuning and preference optimization, and the reasoning model leverages Long-Chain-of-Thought (LongCoT) reinforcement learning to improve multi-step code reasoning. Seed-Coder achieves state-of-the-art results among open-source models of similar size and even surpasses some much larger models, demonstrating superior performance in code generation, code completion, code editing, code reasoning, and software engineering tasks. 

---
# Unleashing the Reasoning Potential of Pre-trained LLMs by Critique Fine-Tuning on One Problem 

**Authors**: Yubo Wang, Ping Nie, Kai Zou, Lijun Wu, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.03295)  

**Abstract**: We have witnessed that strong LLMs like Qwen-Math, MiMo, and Phi-4 possess immense reasoning potential inherited from the pre-training stage. With reinforcement learning (RL), these models can improve dramatically on reasoning tasks. Recent studies have shown that even RL on a single problem can unleash these models' reasoning capabilities. However, RL is not only expensive but also unstable. Even one-shot RL requires hundreds of GPU hours. This raises a critical question: Is there a more efficient way to unleash the reasoning potential of these powerful base LLMs? In this work, we demonstrate that Critique Fine-Tuning (CFT) on only one problem can effectively unleash the reasoning potential of LLMs. Our method constructs critique data by collecting diverse model-generated solutions to a single problem and using teacher LLMs to provide detailed critiques. We fine-tune Qwen and Llama family models, ranging from 1.5B to 14B parameters, on the CFT data and observe significant performance gains across diverse reasoning tasks. For example, with just 5 GPU hours of training, Qwen-Math-7B-CFT show an average improvement of 15% on six math benchmarks and 16% on three logic reasoning benchmarks. These results are comparable to or even surpass the results from RL with 20x less compute. Ablation studies reveal the robustness of one-shot CFT across different prompt problems. These results highlight one-shot CFT as a simple, general, and compute-efficient approach to unleashing the reasoning capabilities of modern LLMs. 

---
# Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning 

**Authors**: Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.04207)  

**Abstract**: Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning. This staged training approach effectively balances perceptual grounding and cognitive reasoning development. By incorporating the above insights and addressing multimodal RL issues, we introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025. 

---
# BadReward: Clean-Label Poisoning of Reward Models in Text-to-Image RLHF 

**Authors**: Kaiwen Duan, Hongwei Yao, Yufei Chen, Ziyun Li, Tong Qiao, Zhan Qin, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03234)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is crucial for aligning text-to-image (T2I) models with human preferences. However, RLHF's feedback mechanism also opens new pathways for adversaries. This paper demonstrates the feasibility of hijacking T2I models by poisoning a small fraction of preference data with natural-appearing examples. Specifically, we propose BadReward, a stealthy clean-label poisoning attack targeting the reward model in multi-modal RLHF. BadReward operates by inducing feature collisions between visually contradicted preference data instances, thereby corrupting the reward model and indirectly compromising the T2I model's integrity. Unlike existing alignment poisoning techniques focused on single (text) modality, BadReward is independent of the preference annotation process, enhancing its stealth and practical threat. Extensive experiments on popular T2I models show that BadReward can consistently guide the generation towards improper outputs, such as biased or violent imagery, for targeted concepts. Our findings underscore the amplified threat landscape for RLHF in multi-modal systems, highlighting the urgent need for robust defenses. Disclaimer. This paper contains uncensored toxic content that might be offensive or disturbing to the readers. 

---
