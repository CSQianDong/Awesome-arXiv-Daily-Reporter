# Tiny-R1V: Lightweight Multimodal Unified Reasoning Model via Model Merging 

**Authors**: Qixiang Yin, Huanjin Yao, Jianghao Chen, Jiaxing Huang, Zhicheng Zhao, Fei Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.08987)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse tasks, they encounter numerous challenges in terms of reasoning efficiency, such as large model size, overthinking, and compromised accuracy in lightweight scenarios. However, research on the reasoning capabilities of lightweight MLLMs is quite lacking. To this end, we propose Tiny-R1V, a novel lightweight 3B model that achieves faster inference and higher accuracy via a two-stage optimization, while unifying multimodal reasoning across multiple tasks and using fewer tokens. In the first stage, Tiny-R1V introduces Length-Informed Relative Policy Optimization (LIPO), a novel reinforcement learning method, to train each reasoning model. The LIPO is designed to dynamically adjusts advantages of responses within groups, that is, by prioritizing concise yet high-quality responses to encourage the generation of shorter and more accurate response. In the second stage, we propose Adaptive Model Merging (AMM), a training-free model merging method that merges multiple specialist models into a unified architecture. Specifically, AMM adaptively adjusts the weights of task vectors and robustly optimizes the merged vectors via a novel gradient projection regularization loss function, thus mitigating redundant conflicts between them. Extensive evaluations on ten widely-used reasoning benchmarks covering mathematics, structured data (charts, tables, documents), OCR, and general capabilities showcase the superior performance of Tiny-R1V, enabling lightweight models to excel in diverse multimodal reasoning tasks. 

---
# TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation 

**Authors**: Yincen Qu, Huan Xiao, Feng Li, Hui Zhou, Xiangying Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.09011)  

**Abstract**: Travel planning is a valuable yet complex task that poses significant challenges even for advanced large language models (LLMs). While recent benchmarks have advanced in evaluating LLMs' planning capabilities, they often fall short in evaluating feasibility, reliability, and engagement of travel plans. We introduce a comprehensive benchmark for travel planning that unifies fine-grained criteria into a single reward, enabling direct comparison of plan quality and seamless integration with reinforcement learning (RL). Our evaluator achieves moderate agreement with travel-expert annotations (60.75\%) and outperforms multiple LLM-as-judge baselines. We further release a large-scale dataset of 4,870 queries including 219 real-world, free-form requests for generalization to authentic user intent. Using this benchmark, we conduct extensive experiments across diverse methods and LLMs, including test-time computation, neuro-symbolic approaches, supervised fine-tuning, and RL via GRPO. Across base models, RL generally improves itinerary feasibility over prompt-only and supervised baselines, yielding higher unified reward scores. 

---
# GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfare 

**Authors**: Siqi Zhu, David Zhang, Pedro Cisneros-Velarde, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2510.08872)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a mutual welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and mutual welfare compared to baselines across diverse tasks. The code is available at this https URL . 

---
# Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation 

**Authors**: Sondos Mahmoud Bsharat, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09599)  

**Abstract**: Large language models (LLMs) have demonstrated impressive reasoning capabilities when provided with chain-of-thought exemplars, but curating large reasoning datasets remains laborious and resource-intensive. In this work, we introduce Prompting Test-Time Scaling (P-TTS), a simple yet effective inference-time data augmentation strategy for enhancing LLM reasoning through finetuning. Rather than collecting thousands or even millions of examples, P-TTS leverages a small pool of only 90 manually selected reasoning instances and systematically varies exemplar augmentation through principled instruction prompting intensities at test time to synthesize diverse reasoning trajectory contexts. Then we finetune the various sizes of Qwen-2.5 models on P-TTS data. Across a suite of mathematical reasoning AIME2024 & 25, MATH500, and GPQA-Diamond, our P-TTS-7B and 32B models outperform the prior competitive baselines like S1 and S1.1 (1K-shot), achieving absolute accuracy gains of +26.66% and +30.00% on AIME'24 (7B), and +13.34% and +6.67% on AIME'25 (7B); P-TTS-32B yields gains of +23.33% and +16.63% on AIME'24, and +26.63% and +3.33% on AIME'25 (vs. S1 and S1.1, respectively), with comparable or better performance on MATH500 and GPQA-Diamond. We further show that P-TTS enhances zero-shot generalization accuracy on out-of-domain reasoning benchmarks of Gaokao, Kaoyan, OlympiadBench, AMC23, GradeSchoolMath, and Minerva. Our analysis suggests that test-time scaling effectively explores the latent space of reasoning patterns, amplifying LLM problem-solving with minimal annotation overhead, and further unlocking the reasoning potential and capabilities of LLMs. Prompting Test-Time Scaling offers a practical, low-cost way to elicit LLM reasoning in resource-constrained or rapidly evolving domains. 

---
# Multimodal Policy Internalization for Conversational Agents 

**Authors**: Zhenhailong Wang, Jiateng Liu, Amin Fazel, Ritesh Sarkhel, Xing Fan, Xiang Li, Chenlei Guo, Heng Ji, Ruhi Sarikaya  

**Link**: [PDF](https://arxiv.org/pdf/2510.09474)  

**Abstract**: Modern conversational agents like ChatGPT and Alexa+ rely on predefined policies specifying metadata, response styles, and tool-usage rules. As these LLM-based systems expand to support diverse business and user queries, such policies, often implemented as in-context prompts, are becoming increasingly complex and lengthy, making faithful adherence difficult and imposing large fixed computational costs. With the rise of multimodal agents, policies that govern visual and multimodal behaviors are critical but remain understudied. Prior prompt-compression work mainly shortens task templates and demonstrations, while existing policy-alignment studies focus only on text-based safety rules. We introduce Multimodal Policy Internalization (MPI), a new task that internalizes reasoning-intensive multimodal policies into model parameters, enabling stronger policy-following without including the policy during inference. MPI poses unique data and algorithmic challenges. We build two datasets spanning synthetic and real-world decision-making and tool-using tasks and propose TriMPI, a three-stage training framework. TriMPI first injects policy knowledge via continual pretraining, then performs supervised finetuning, and finally applies PolicyRollout, a GRPO-style reinforcement learning extension that augments rollouts with policy-aware responses for grounded exploration. TriMPI achieves notable gains in end-to-end accuracy, generalization, and robustness to forgetting. As the first work on multimodal policy internalization, we provide datasets, training recipes, and comprehensive evaluations to foster future research. Project page: this https URL. 

---
# CLARity: Reasoning Consistency Alone Can Teach Reinforced Experts 

**Authors**: Jiuheng Lin, Cong Jiang, Zirui Wu, Jiarui Sun, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09278)  

**Abstract**: Training expert LLMs in domains with scarce data is difficult, often relying on multiple-choice questions (MCQs). However, standard outcome-based reinforcement learning (RL) on MCQs is risky. While it may improve accuracy, we observe it often degrades reasoning quality such as logical consistency. Existing solutions to supervise reasoning, such as large-scale Process Reward Models (PRMs), are prohibitively expensive. To address this, we propose CLARity, a cost-effective RL framework that enhances reasoning quality using only a small, general-purpose LLM. CLARity integrates a consistency-aware reward mechanism with a 2-stage refine-then-monitor training pipeline to enhance reasoning consistency, and a dynamic data reformulation strategy to to better exploit limited data. Experiments demonstrate that CLARity improves response consistency by 16.5% and accuracy by 7.5% over baselines. Human evaluations further confirm holistic improvements in coherence and professionalism. Thus, CLARity offers a generalizable solution that enables smaller models to effectively guide expert models by reasoning this http URL code is open sourced at: this https URL 

---
# HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidance 

**Authors**: Suming Qiu, Jing Li, Zhicheng Zhou, Junjie Huang, Linyuan Qiu, Zhijie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.08896)  

**Abstract**: We present HES-SQL, a novel hybrid training framework that advances Text-to-SQL generation through the integration of thinking-mode-fused supervised fine-tuning (SFT) with Group Relative Policy Optimization (GRPO). Our approach introduces three key innovations: (1) a skeleton-completeness scoring mechanism that enhances preference alignment between generated queries and optimal SQL structures; (2) a query-latency-aware reward system that incentivizes the generation of computationally efficient SQL queries; (3) a self-distillation process for thinking-mode completion that prevents degradation of the model's reasoning capabilities. This framework enables hybrid thinking models to switch between reasoning and non-reasoning modes while improving SQL query accuracy and execution efficiency.
Experimental evaluation, conducted on MySQL 8.0 and SQLite 3.42 under controlled single-user conditions, demonstrates that HES-SQL achieves competitive performance with execution accuracies of 79.14\% and 54.9\% on the BIRD and KaggleDBQA benchmarks, respectively. Query latency is measured as the end-to-end execution time of generated queries on the DBMS, averaged over multiple runs to mitigate variance. Efficiency gains range from 11\% to 20\% relative to supervised baselines. Our results establish a new paradigm for Text-to-SQL systems that effectively balances semantic accuracy with computational efficiency through execution-informed reinforcement learning (RL). The proposed methodology has significant implications for developing robust natural language interfaces to databases and can be extended to broader structured generation tasks requiring both correctness and efficiency optimization. 

---
# Exploring Multi-Temperature Strategies for Token- and Rollout-Level Control in RLVR 

**Authors**: Haomin Zhuang, Yujun Zhou, Taicheng Guo, Yue Huang, Fangxu Liu, Kai Song, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08892)  

**Abstract**: Reinforcement Learning has demonstrated substantial improvements in the reasoning abilities of Large Language Models (LLMs), exhibiting significant applicability across various domains. Recent research has identified that tokens within LLMs play distinct roles during reasoning tasks, categorizing them into high-entropy reasoning tokens and low-entropy knowledge tokens. Prior approaches have typically focused on restricting updates to indirectly encourage exploration, yet they do not explicitly facilitate exploratory behavior during the token generation stage itself. In this work, we introduce a complementary approach that explicitly promotes exploration during sampling by applying distinct temperature settings for different token types. Specifically, our method employs higher temperatures for reasoning tokens to actively encourage exploration, while retaining lower temperatures for knowledge tokens to maintain factual correctness. Furthermore, we systematically investigate various multi-temperature scheduling strategies and their impacts within reinforcement learning contexts. Empirical evaluations on several reasoning benchmarks demonstrate that our approach significantly enhances the reasoning performance of LLMs. The code is available at this https URL. 

---
# Energy-Driven Steering: Reducing False Refusals in Large Language Models 

**Authors**: Eric Hanchen Jiang, Weixuan Ou, Run Liu, Shengyuan Pang, Guancheng Wan, Ranjie Duan, Wei Dong, Kai-Wei Chang, XiaoFeng Wang, Ying Nian Wu, Xinfeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08646)  

**Abstract**: Safety alignment of large language models (LLMs) faces a key challenge: current alignment techniques often only focus on improving safety against harmful prompts, causing LLMs to become over-cautious and refuse to respond to benign prompts. Therefore, a key objective of safe alignment is to enhance safety while simultaneously reducing false refusals. In this paper, we introduce Energy-Driven Steering (EDS), a novel, fine-tuning free framework designed to resolve this challenge through dynamic, inference-time intervention. We trained a lightweight, external Energy-Based Model (EBM) to assign high energy to undesirable (false refusal or jailbreak) states and low energy to desirable (helpful response or safe reject) ones. During inference, EBM maps the LLM's internal activations to an "energy landscape". We use the gradient of the energy function to dynamically steer the LLM's hidden states to low energy regions, correcting the model to generate a desirable response in real-time without modifying its weights. This method decouples behavioral control from the model's core knowledge, offering a flexible solution with minimal computational overhead. Extensive experiments across a wide range of models show our method successfully achieves this objective: it substantially lowers false refusal rates. For example, raising compliance on the ORB-H benchmark from 57.3% to 82.6% while maintaining the baseline safety performance. Our work presents an effective paradigm for building LLMs that achieve both low false refusal rates and high safety. 

---
# Logit Arithmetic Elicits Long Reasoning Capabilities Without Training 

**Authors**: Yunxiang Zhang, Muhammad Khalifa, Lechen Zhang, Xin Liu, Ayoung Lee, Xinliang Frederick Zhang, Farima Fatahi Bayat, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09354)  

**Abstract**: Large reasoning models exhibit long chain-of-thought reasoning with strategies such as backtracking and self-correction, though recent studies suggest that these abilities typically require additional training. We first investigate whether such behaviors can be elicited without any training. To this end, we propose a decoding-time approach, ThinkLogit, which utilizes logit arithmetic to tune a target large non-reasoning model for long reasoning using a substantially smaller reasoning model as the guider. We then show that we can further boost its performance by training the guider model with preference optimization over correct/incorrect reasoning pairs sampled from both the target and guider model, a setup we refer to as ThinkLogit-DPO. Our experiments demonstrate that ThinkLogit and ThinkLogit-DPO achieve a relative improvement in average accuracy by 24.5% and 29.1%, respectively, over five reasoning benchmarks using the Qwen2.5-32B guided by R1-Distill-Qwen-1.5B, a model 21x smaller. Moreover, we find that ThinkLogit remains effective when the guider and target come from different model families. It is also orthogonal to post-training methods for small models, as guiders improved through supervised distillation or reinforcement learning can be directly plugged in to yield stronger large models, offering a practical path to unlock long reasoning in large-scale models without costly post-training. 

---
# Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Markov Likelihood 

**Authors**: Xingyu Lin, Yilin Wen, En Wang, Du Su, Wenbin Liu, Chenfu Bao, Zhonghou Lv  

**Link**: [PDF](https://arxiv.org/pdf/2510.09369)  

**Abstract**: Group Relative Policy Optimization (GRPO) has significantly advanced the reasoning ability of large language models (LLMs), particularly by boosting their mathematical performance. However, GRPO and related entropy-regularization methods still face challenges rooted in the sparse token rewards inherent to chain-of-thought (CoT). Current approaches often rely on undifferentiated token-level entropy adjustments, which frequently lead to entropy collapse or model collapse. In this work, we propose TEPO, a novel token-level framework that incorporates Markov Likelihood (sequence likelihood) links group-level rewards with tokens via token-level aggregation. Experiments show that TEPO consistently outperforms existing baselines across key metrics (including @k and accuracy). It not only sets a new state of the art on mathematical reasoning tasks but also significantly enhances training stability. 

---
# DSPO: Stable and Efficient Policy Optimization for Agentic Search and Reasoning 

**Authors**: Chenyang Gu, Yewen Pu, Bruce Yang, Xiaofan Li, Huan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.09255)  

**Abstract**: Enhancing LLMs with the ability to actively search external knowledge is crucial for complex and real-world tasks. Current approaches either rely on prompting to elicit the model's innate agent capabilities, or suffer from performance ceilings and collapse when applying RL to complex interactive tasks, leaving their true agentic potential untapped. To address this, we introduce \textbf{D}ynamic-filter \textbf{S}equence-level \textbf{P}olicy \textbf{O}ptimization (DSPO), an improved RL algorithm designed for robust agent training through sequence-level optimization and dynamic sample filtering. We train our model purely through RL to interleave multi-turn search and reasoning, obviating the need for supervised demonstration data. Across multiple QA benchmarks, our DSPO-trained 7B model improves over a comparable previous work by \textbf{34.1\%}, and even outperforms the 14B model from previous work in complex multihop QA such as HotpotQA by nearly \textbf{9\% relative}, maintaining exceptional training stability. 

---
# DARO: Difficulty-Aware Reweighting Policy Optimization 

**Authors**: Jingyu Zhou, Lu Ma, Hao Liang, Chengyu Shen, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09001)  

**Abstract**: Recent advances in large language models (LLMs) have shown that reasoning ability can be significantly enhanced through Reinforcement Learning with Verifiable Rewards (RLVR). Group Relative Policy Optimization (GRPO) has emerged as the de facto approach for RLVR, inspiring numerous variants. However, our mathematical analysis reveals that these methods are fundamentally weighted variations of GRPO. We provide a unified view, demonstrating that their reliance on static or overly simplistic weighting schemes tied to sample difficulty prevents adaptation to a model's evolving capabilities. This creates a significant loss scale issue, where training disproportionately focuses on certain difficulty levels at the expense of others, hindering overall performance. To address these limitations, we introduce \textbf{Difficulty-Aware Reweighting Policy Optimization (DARO)}, a method that dynamically adjusts the loss contribution of each difficulty group based on the model's learning state. Extensive experiments on Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, and Llama3.1-8B show that DARO outperforms four leading baselines across six math benchmarks, achieving significantly faster convergence and superior final performance. 

---
# ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection 

**Authors**: Jingbiao Mei, Mingsheng Sun, Jinghong Chen, Pengda Qin, Yuhong Li, Da Chen, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2510.08630)  

**Abstract**: Hateful memes have emerged as a particularly challenging form of online abuse, motivating the development of automated detection systems. Most prior approaches rely on direct detection, producing only binary predictions. Such models fail to provide the context and explanations that real-world moderation requires. Recent Explain-then-Detect approaches, using Chain-of-Thought prompting or LMM agents, perform worse than simple SFT baselines, and even advanced post-training methods such as GRPO fail to close the gap. Our analysis identifies two key issues of such systems: important policy-relevant cues such as targets and attack types are not hypothesized by the model as a likely explanation; and the binary reward signal is insufficient to guide reasoning. To address these challenges, we propose ExPO-HM (Explain-then-Detect Policy Optimization for Hateful Memes), inspired by the training and evaluation process of human annotators. ExPO-HM combines SFT warmup, GRPO with curriculum learning, and Conditional Decision Entropy (CDE) as both metric and reward for reasoning quality. Across three hateful meme benchmarks, ExPO-HM achieves state-of-the-art performance on binary detection, fine-grained classification, and reasoning quality, with up to 15\% and 17\% F1 improvement over the GRPO and DPO baselines, respectively. By moving hateful meme detection from simple binary alarms to explanation-driven detection, ExPO-HM provides accurate, interpretable, and actionable moderation support. 

---
# HINT: Helping Ineffective Rollouts Navigate Towards Effectiveness 

**Authors**: Xinyi Wang, Jinyi Han, Zishang Jiang, Tingyun Li, Jiaqing Liang, Sihang Jiang, Zhaoqian Dai, Shuguang Ma, Fei Yu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.09388)  

**Abstract**: Reinforcement Learning (RL) has become a key driver for enhancing the long chain-of-thought (CoT) reasoning capabilities of Large Language Models (LLMs). However, prevalent methods like GRPO often fail when task difficulty exceeds the model's capacity, leading to reward sparsity and inefficient training. While prior work attempts to mitigate this using off-policy data, such as mixing RL with Supervised Fine-Tuning (SFT) or using hints, they often misguide policy updates In this work, we identify a core issue underlying these failures, which we term low training affinity. This condition arises from a large distributional mismatch between external guidance and the model's policy. To diagnose this, we introduce Affinity, the first quantitative metric for monitoring exploration efficiency and training stability. To improve Affinity, we propose HINT: Helping Ineffective rollouts Navigate Towards effectiveness, an adaptive hinting framework. Instead of providing direct answers, HINT supplies heuristic hints that guide the model to discover solutions on its own, preserving its autonomous reasoning capabilities. Extensive experiments on mathematical reasoning tasks show that HINT consistently outperforms existing methods, achieving state-of-the-art results with models of various scales, while also demonstrating significantly more stable learning and greater data this http URL is available on Github. 

---
# Unleashing Perception-Time Scaling to Multimodal Reasoning Models 

**Authors**: Yifan Li, Zhenghao Chen, Ziheng Wu, Kun Zhou, Ruipu Luo, Can Zhang, Zhentao He, Yufei Zhan, Wayne Xin Zhao, Minghui Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08964)  

**Abstract**: Recent advances in inference-time scaling, particularly those leveraging reinforcement learning with verifiable rewards, have substantially enhanced the reasoning capabilities of Large Vision-Language Models (LVLMs). Inspired by this success, similar strategies have been applied to multimodal reasoning, yet their impact on visual perception remains unclear. To investigate this gap, we introduce DisTANCE, a perception-centric benchmark for visual estimation tasks. Evaluation results show that LVLMs exhibit limited estimation precision, and inference-time scaling offers only marginal gains. We attribute this to the fast perception paradigm of current LVLMs, where visual understanding is treated as a one-shot output without modeling the underlying perceptual process. To address this, we propose Perception-Time Scaling (PTS), a novel paradigm that encourages token-rich perception and decomposes complex perception problems into intermediate tractable sub-problems, thereby enabling perception to align with and benefit from inference-time scaling. Combined with reinforcement learning techniques, PTS significantly improves perception accuracy, raising high-precision performance on DisTANCE from 8.0% to 64.7%, and generalizes well to out-of-domain tasks. Surprisingly, even though PTS data are purely synthetic, combining them with math reasoning data yields consistent gains in both reasoning and real-world perception benchmarks. Further analysis reveals that PTS introduces more perception-related tokens and increases the model's attention to image tokens. Our code and data will be publicly released. 

---
