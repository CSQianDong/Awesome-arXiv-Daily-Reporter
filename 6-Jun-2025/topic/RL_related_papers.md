# Reason-to-Recommend: Using Interaction-of-Thought Reasoning to Enhance LLM Recommendation 

**Authors**: Keyu Zhao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05069)  

**Abstract**: Driven by advances in Large Language Models (LLMs), integrating them into recommendation tasks has gained interest due to their strong semantic understanding and prompt flexibility. Prior work encoded user-item interactions or metadata into prompts for recommendations. In parallel, LLM reasoning, boosted by test-time scaling and reinforcement learning, has excelled in fields like mathematics and code, where reasoning traces and correctness signals are clear, enabling high performance and interpretability. However, directly applying these reasoning methods to recommendation is ineffective because user feedback is implicit and lacks reasoning supervision. To address this, we propose $\textbf{R2Rec}$, a reasoning-enhanced recommendation framework that samples interaction chains from the user-item graph and converts them into structured interaction-of-thoughts via a progressive masked prompting strategy, with each thought representing stepwise reasoning grounded in interaction context. This allows LLMs to simulate step-by-step decision-making based on implicit patterns. We design a two-stage training pipeline: supervised fine-tuning teaches basic reasoning from high-quality traces, and reinforcement learning refines reasoning via reward signals, alleviating sparse explicit supervision. Experiments on three real-world datasets show R2Rec outperforms classical and LLM-based baselines with an average $\textbf{10.48%}$ improvement in HitRatio@1 and $\textbf{131.81%}$ gain over the original LLM. Furthermore, the explicit reasoning chains enhance interpretability by revealing the decision process. Our code is available at: this https URL. 

---
# Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning 

**Authors**: Jiayu Wang, Yifei Ming, Zixuan Ke, Caiming Xiong, Shafiq Joty, Aws Albarghouthi, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2506.04723)  

**Abstract**: Reinforcement learning (RL) has become the dominant paradigm for endowing language models with advanced reasoning capabilities. Despite the substantial empirical gains demonstrated by RL-based training methods like GRPO, a granular understanding of their advantages is still lacking. To address this gap, we introduce a fine-grained analytic framework to dissect the impact of RL on reasoning. Our framework specifically investigates key elements that have been hypothesized to benefit from RL training: (1) plan-following and execution, (2) problem decomposition, and (3) improved reasoning and knowledge utilization. Using this framework, we gain insights beyond mere accuracy. For instance, providing models with explicit step-by-step plans surprisingly degrades performance on the most challenging benchmarks, yet RL-tuned models exhibit greater robustness, experiencing markedly smaller performance drops than their base counterparts. This suggests that RL may not primarily enhance the execution of external plans but rather empower models to formulate and follow internal strategies better suited to their reasoning processes. Conversely, we observe that RL enhances the model's capacity to integrate provided knowledge into its reasoning process, leading to performance improvements across diverse tasks. We also study difficulty, showing improved training by developing new ways to exploit hard problems. Our findings lay a foundation for more principled training and evaluation of reasoning models. 

---
# Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning 

**Authors**: Violet Xiang, Chase Blagden, Rafael Rafailov, Nathan Lile, Sang Truong, Chelsea Finn, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2506.05256)  

**Abstract**: Large reasoning models (LRMs) achieve higher performance on challenging reasoning tasks by generating more tokens at inference time, but this verbosity often wastes computation on easy problems. Existing solutions, including supervised finetuning on shorter traces, user-controlled budgets, or RL with uniform penalties, either require data curation, manual configuration, or treat all problems alike regardless of difficulty. We introduce Adaptive Length Penalty (ALP), a reinforcement learning objective tailoring generation length to per-prompt solve rate. During training, ALP monitors each prompt's online solve rate through multiple rollouts and adds a differentiable penalty whose magnitude scales inversely with that rate, so confident (easy) prompts incur a high cost for extra tokens while hard prompts remain unhindered. Posttraining DeepScaleR-1.5B with ALP cuts average token usage by 50\% without significantly dropping performance. Relative to fixed-budget and uniform penalty baselines, ALP redistributes its reduced budget more intelligently by cutting compute on easy prompts and reallocating saved tokens to difficult ones, delivering higher accuracy on the hardest problems with higher cost. 

---
# Contextual Integrity in LLMs via Reasoning and Reinforcement Learning 

**Authors**: Guangchen Lan, Huseyin A. Inan, Sahar Abdelnabi, Janardhan Kulkarni, Lukas Wutschitz, Reza Shokri, Christopher G. Brinton, Robert Sim  

**Link**: [PDF](https://arxiv.org/pdf/2506.04245)  

**Abstract**: As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls. 

---
# Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay 

**Authors**: Yifan Sun, Jingyan Shen, Yibin Wang, Tianyu Chen, Zhendong Wang, Mingyuan Zhou, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05316)  

**Abstract**: Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism that reuses recent rollouts, lowering per-step computation while maintaining stable updates. Extensive experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 25% to 65% to reach the same level of performance as the original GRPO algorithm. 

---
# ProRefine: Inference-time Prompt Refinement with Textual Feedback 

**Authors**: Deepak Pandita, Tharindu Cyril Weerasooriya, Ankit Parag Shah, Christopher M. Homan, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.05305)  

**Abstract**: Agentic workflows, where multiple AI agents collaborate to accomplish complex tasks like reasoning or planning, are becoming increasingly prevalent. However, these workflows often suffer from error propagation and sub-optimal performance, largely due to poorly designed prompts that fail to effectively guide individual agents. This is a critical problem because it limits the reliability and scalability of these powerful systems. We introduce ProRefine, an innovative inference-time prompt optimization method that leverages textual feedback from large language models (LLMs) to address this challenge. ProRefine dynamically refines prompts for multi-step reasoning tasks without additional training or ground truth labels. Evaluated on five benchmark mathematical reasoning datasets, ProRefine significantly surpasses zero-shot Chain-of-Thought baselines by 3 to 37 percentage points. This approach not only boosts accuracy but also allows smaller models to match the performance of larger ones, highlighting its potential for efficient and scalable AI deployment, and democratizing access to high-performing AI. 

---
# TreeRPO: Tree Relative Policy Optimization 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Xiaodan Liang, Yiwei Wang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05183)  

**Abstract**: Large Language Models (LLMs) have shown remarkable reasoning capabilities through Reinforcement Learning with Verifiable Rewards (RLVR) methods. However, a key limitation of existing approaches is that rewards defined at the full trajectory level provide insufficient guidance for optimizing the intermediate steps of a reasoning process. To address this, we introduce \textbf{\name}, a novel method that estimates the mathematical expectations of rewards at various reasoning steps using tree sampling. Unlike prior methods that rely on a separate step reward model, \name directly estimates these rewards through this sampling process. Building on the group-relative reward training mechanism of GRPO, \name innovatively computes rewards based on step-level groups generated during tree sampling. This advancement allows \name to produce fine-grained and dense reward signals, significantly enhancing the learning process and overall performance of LLMs. Experimental results demonstrate that our \name algorithm substantially improves the average Pass@1 accuracy of Qwen-2.5-Math on test benchmarks, increasing it from 19.0\% to 35.5\%. Furthermore, \name significantly outperforms GRPO by 2.9\% in performance while simultaneously reducing the average response length by 18.1\%, showcasing its effectiveness and efficiency. Our code will be available at \href{this https URL}{this https URL}. 

---
# On the Mechanism of Reasoning Pattern Selection in Reinforcement Learning for Language Models 

**Authors**: Xingwu Chen, Tianle Li, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.04695)  

**Abstract**: Reinforcement learning (RL) has demonstrated remarkable success in enhancing model capabilities, including instruction-following, preference learning, and reasoning. Yet despite its empirical successes, the mechanisms by which RL improves reasoning abilities remain poorly understood. We present a systematic study of Reinforcement Learning with Verifiable Rewards (RLVR), showing that its primary benefit comes from optimizing the selection of existing reasoning patterns. Through extensive experiments, we demonstrate that RLVR-trained models preferentially adopt high-success-rate reasoning patterns while mostly maintaining stable performance on individual patterns. We further develop theoretical analyses on the convergence and training dynamics of RLVR based on a simplified question-reason-answer model. We study the gradient flow and show that RLVR can indeed find the solution that selects the reason pattern with the highest success rate. Besides, our theoretical results
reveal two distinct regimes regarding the convergence of RLVR training: (1) rapid convergence for models with relatively strong initial reasoning capabilities versus (2) slower optimization dynamics for weaker models. Furthermore, we show that the slower optimization for weaker models can be mitigated by applying the supervised fine-tuning (SFT) before RLVR, when using a feasibly high-quality SFT dataset. We validate the theoretical findings through extensive experiments. This work advances our theoretical understanding of RL's role in LLM fine-tuning and offers insights for further enhancing reasoning capabilities. 

---
# MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale 

**Authors**: Ran Xu, Yuchen Zhuang, Yishan Zhong, Yue Yu, Xiangru Tang, Hang Wu, May D. Wang, Peifeng Ruan, Donghan Yang, Tao Wang, Guanghua Xiao, Carl Yang, Yang Xie, Wenqi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.04405)  

**Abstract**: We introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. MedAgentGYM comprises 72,413 task instances across 129 categories derived from authentic real-world biomedical scenarios. Tasks are encapsulated within executable coding environments, each featuring detailed task descriptions, interactive feedback mechanisms, verifiable ground-truth annotations, and scalable training trajectory generation. Extensive benchmarking of over 30 LLMs reveals a notable performance disparity between commercial API-based models and open-source counterparts. Leveraging MedAgentGYM, Med-Copilot-7B achieves substantial performance gains through supervised fine-tuning (+36.44%) and continued reinforcement learning (+42.47%), emerging as an affordable and privacy-preserving alternative competitive with gpt-4o. By offering both a comprehensive benchmark and accessible, expandable training resources within unified execution environments, MedAgentGYM delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical research and practice. 

---
# Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models 

**Authors**: Anirudh Bharadwaj, Chaitanya Malaviya, Nitish Joshi, Mark Yatskar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05339)  

**Abstract**: Language models serve as proxies for human preference judgements in alignment and evaluation, yet they exhibit systematic miscalibration, prioritizing superficial patterns over substantive qualities. This bias manifests as overreliance on features like length, structure, and style, leading to issues like reward hacking and unreliable evaluations. Evidence suggests these biases originate in artifacts in human training data. In this work, we systematically investigate the relationship between training data biases and preference model miscalibration across five idiosyncratic features of language model generations: length, structure, jargon, sycophancy and vagueness. Using controlled counterfactual pairs, we first quantify the extent to which preference models favor responses with magnified biases (skew), finding this preference occurs in >60% of instances, and model preferences show high miscalibration (~40%) compared to human preferences. Notably, bias features only show mild negative correlations to human preference labels (mean r_human = -0.12) but show moderately strong positive correlations with labels from a strong reward model (mean r_model = +0.36), suggesting that models may overrely on spurious cues. To mitigate these issues, we propose a simple post-training method based on counterfactual data augmentation (CDA) using synthesized contrastive examples. Finetuning models with CDA reduces average miscalibration from 39.4% to 32.5% and average absolute skew difference from 20.5% to 10.0%, while maintaining overall RewardBench performance, showing that targeted debiasing is effective for building reliable preference models. 

---
# RIVAL: Reinforcement Learning with Iterative and Adversarial Optimization for Machine Translation 

**Authors**: Tianjiao Li, Mengran Yu, Chenyu Shi, Yanjun Zhao, Xiaojing Liu, Qiang Zhang, Qi Zhang, Xuanjing Huang, Jiayin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05070)  

**Abstract**: Large language models (LLMs) possess strong multilingual capabilities, and combining Reinforcement Learning from Human Feedback (RLHF) with translation tasks has shown great potential. However, we observe that this paradigm performs unexpectedly poorly when applied to colloquial subtitle translation tasks. In this work, we investigate this issue and find that the offline reward model (RM) gradually diverges from the online LLM due to distributional shift, ultimately leading to undesirable training outcomes. To address this, we propose RIVAL, an adversarial training framework that formulates the process as a min-max game between the RM and the LLM. RIVAL iteratively updates the both models, with the RM trained to distinguish strong from weak translations (qualitative preference reward), and the LLM trained to enhance its translation for closing this gap. To stabilize training and improve generalizability, we also incorporate quantitative preference reward (e.g., BLEU) into the RM, enabling reference-free quality modeling aligned with human evaluation. Through extensive experiments, we demonstrate that the proposed adversarial training framework significantly improves upon translation baselines. 

---
# SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat 

**Authors**: Yuru Jiang, Wenxuan Ding, Shangbin Feng, Greg Durrett, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2506.04721)  

**Abstract**: We propose SPARTA ALIGNMENT, an algorithm to collectively align multiple LLMs through competition and combat. To complement a single model's lack of diversity in generation and biases in evaluation, multiple LLMs form a "sparta tribe" to compete against each other in fulfilling instructions while serving as judges for the competition of others. For each iteration, one instruction and two models are selected for a duel, the other models evaluate the two responses, and their evaluation scores are aggregated through a adapted elo-ranking based reputation system, where winners/losers of combat gain/lose weight in evaluating others. The peer-evaluated combat results then become preference pairs where the winning response is preferred over the losing one, and all models learn from these preferences at the end of each iteration. SPARTA ALIGNMENT enables the self-evolution of multiple LLMs in an iterative and collective competition process. Extensive experiments demonstrate that SPARTA ALIGNMENT outperforms initial models and 4 self-alignment baselines across 10 out of 12 tasks and datasets with 7.0% average improvement. Further analysis reveals that SPARTA ALIGNMENT generalizes more effectively to unseen tasks and leverages the expertise diversity of participating models to produce more logical, direct and informative outputs. 

---
# Aligning Large Language Models with Implicit Preferences from User-Generated Content 

**Authors**: Zhaoxuan Tan, Zheng Li, Tianyi Liu, Haodong Wang, Hyokun Yun, Ming Zeng, Pei Chen, Zhihan Zhang, Yifan Gao, Ruijie Wang, Priyanka Nigam, Bing Yin, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04463)  

**Abstract**: Learning from preference feedback is essential for aligning large language models (LLMs) with human values and improving the quality of generated responses. However, existing preference learning methods rely heavily on curated data from humans or advanced LLMs, which is costly and difficult to scale. In this work, we present PUGC, a novel framework that leverages implicit human Preferences in unlabeled User-Generated Content (UGC) to generate preference data. Although UGC is not explicitly created to guide LLMs in generating human-preferred responses, it often reflects valuable insights and implicit preferences from its creators that has the potential to address readers' questions. PUGC transforms UGC into user queries and generates responses from the policy model. The UGC is then leveraged as a reference text for response scoring, aligning the model with these implicit preferences. This approach improves the quality of preference data while enabling scalable, domain-specific alignment. Experimental results on Alpaca Eval 2 show that models trained with DPO and PUGC achieve a 9.37% performance improvement over traditional methods, setting a 35.93% state-of-the-art length-controlled win rate using Mistral-7B-Instruct. Further studies highlight gains in reward quality, domain-specific alignment effectiveness, robustness against UGC quality, and theory of mind capabilities. Our code and dataset are available at this https URL 

---
