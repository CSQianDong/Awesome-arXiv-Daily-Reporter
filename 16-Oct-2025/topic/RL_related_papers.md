# Confidence as a Reward: Transforming LLMs into Reward Models 

**Authors**: He Du, Bowen Li, Chengxing Xie, Chang Gao, Kai Chen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.13501)  

**Abstract**: Reward models can significantly enhance the reasoning capabilities of large language models (LLMs), but they typically require extensive curated data and costly training. To mitigate these challenges, training-free approaches such as LLM-as-a-Judge leverage the intrinsic reasoning abilities of LLMs to evaluate responses, achieving promising results. Recent works have also indicated that model confidence can serve effectively as a reward metric, distinguishing between chain-of-thought (CoT) and non-CoT paths. However, the concept of using confidence as a reward has not been comprehensively studied. In this work, we systematically investigate Confidence-as-a-Reward (CRew), a simple yet powerful training-free method that utilizes token-level confidence in the model's final answers as a proxy for reward, especially suitable for close-ended tasks. Through extensive experiments on mathematical reasoning tasks, we demonstrate that CRew outperforms existing training-free reward approaches on the MATH500 and RewardMATH benchmarks, and even surpasses most trained reward models. We further identify a strong correlation between CRew scores and the actual reasoning performance of the model. Additionally, we find that CRew can effectively filter high-quality training data. Building upon these insights, we propose CRew-DPO, a training strategy that constructs preference data from confidence scores combined with correctness signals. Finetuning with CRew-DPO further enhances the model's judging capabilities and consistently outperforms existing self-training methods. 

---
# Tandem Training for Language Models 

**Authors**: Robert West, Ashton Anderson, Ece Kamar, Eric Horvitz  

**Link**: [PDF](https://arxiv.org/pdf/2510.13551)  

**Abstract**: As language models continue to rapidly improve, we can expect their actions and reasoning to become difficult or impossible for weaker agents and humans to follow, undermining interpretability and oversight. With an eye on long-term futures, we pursue methods that encourage models to produce solutions that remain intelligible to weaker collaborators. We formalize intelligibility as handoff robustness: a strong model's solution is intelligible to a weaker model if randomly handing off control to the weaker model along the solution path does not cause failure. Building on this criterion, we introduce tandem training for language models, a reinforcement learning (RL) paradigm in which rollout tokens are intermittently and randomly sampled from a frozen weak model rather than the strong model being trained. Because rollouts succeed only when the strong model's actions and reasoning process can be continued by the weak model -- when the two can co-construct a successful solution -- optimizing standard RL objectives with tandem training implicitly incentivizes both correctness and intelligibility. In the GSM8K math reasoning task, tandem training reliably teaches models to abandon jargon and adapt their language to weaker partners while keeping task accuracy high. Our results demonstrate a promising route to building AI systems that remain auditable by weaker agents, with implications for human--AI collaboration and multi-agent communication. 

---
# From Refusal to Recovery: A Control-Theoretic Approach to Generative AI Guardrails 

**Authors**: Ravi Pandya, Madison Bland, Duy P. Nguyen, Changliu Liu, Jaime Fernández Fisac, Andrea Bajcsy  

**Link**: [PDF](https://arxiv.org/pdf/2510.13727)  

**Abstract**: Generative AI systems are increasingly assisting and acting on behalf of end users in practical settings, from digital shopping assistants to next-generation autonomous cars. In this context, safety is no longer about blocking harmful content, but about preempting downstream hazards like financial or physical harm. Yet, most AI guardrails continue to rely on output classification based on labeled datasets and human-specified criteria,making them brittle to new hazardous situations. Even when unsafe conditions are flagged, this detection offers no path to recovery: typically, the AI system simply refuses to act--which is not always a safe choice. In this work, we argue that agentic AI safety is fundamentally a sequential decision problem: harmful outcomes arise from the AI system's continually evolving interactions and their downstream consequences on the world. We formalize this through the lens of safety-critical control theory, but within the AI model's latent representation of the world. This enables us to build predictive guardrails that (i) monitor an AI system's outputs (actions) in real time and (ii) proactively correct risky outputs to safe ones, all in a model-agnostic manner so the same guardrail can be wrapped around any AI model. We also offer a practical training recipe for computing such guardrails at scale via safety-critical reinforcement learning. Our experiments in simulated driving and e-commerce settings demonstrate that control-theoretic guardrails can reliably steer LLM agents clear of catastrophic outcomes (from collisions to bankruptcy) while preserving task performance, offering a principled dynamic alternative to today's flag-and-block guardrails. 

---
# Training LLM Agents to Empower Humans 

**Authors**: Evan Ellis, Vivek Myers, Jens Tuyls, Sergey Levine, Anca Dragan, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2510.13709)  

**Abstract**: Assistive agents should not only take actions on behalf of a human, but also step out of the way and cede control when there are important decisions to be made. However, current methods for building assistive agents, whether via mimicking expert humans or via RL finetuning on an inferred reward, often encourage agents to complete tasks on their own rather than truly assisting the human attain their objectives. Additionally, these methods often require costly explicit human feedback to provide a training signal. We propose a new approach to tuning assistive language models based on maximizing the human's empowerment, their ability to effect desired changes in the environment. Our empowerment-maximizing method, Empower, only requires offline text data, providing a self-supervised method for fine-tuning language models to better assist humans. To study the efficacy of our approach, we conducted an 18-person user study comparing our empowerment assistant with a strong baseline. Participants preferred our assistant 78% of the time (p=0.015), with a 31% higher acceptance rate and 38% fewer suggestions. Additionally, we introduce a new environment for evaluating multi-turn code assistance using simulated humans. Using this environment, we show that agents trained with Empower increase the success rate of a simulated human programmer on challenging coding questions by an average of 192% over an SFT baseline. With this empowerment objective, we provide a framework for useful aligned AI agents at scale using only offline data without the need for any additional human feedback or verifiable rewards. 

---
# Repairing Reward Functions with Human Feedback to Mitigate Reward Hacking 

**Authors**: Stephane Hatgis-Kessell, Logan Mondal Bhamidipaty, Emma Brunskill  

**Link**: [PDF](https://arxiv.org/pdf/2510.13036)  

**Abstract**: Human-designed reward functions for reinforcement learning (RL) agents are frequently misaligned with the humans' true, unobservable objectives, and thus act only as proxies. Optimizing for a misspecified proxy reward function often induces reward hacking, resulting in a policy misaligned with the human's true objectives. An alternative is to perform RL from human feedback, which involves learning a reward function from scratch by collecting human preferences over pairs of trajectories. However, building such datasets is costly. To address the limitations of both approaches, we propose Preference-Based Reward Repair (PBRR): an automated iterative framework that repairs a human-specified proxy reward function by learning an additive, transition-dependent correction term from preferences. A manually specified reward function can yield policies that are highly suboptimal under the ground-truth objective, yet corrections on only a few transitions may suffice to recover optimal performance. To identify and correct for those transitions, PBRR uses a targeted exploration strategy and a new preference-learning objective. We prove in tabular domains PBRR has a cumulative regret that matches, up to constants, that of prior preference-based RL methods. In addition, on a suite of reward-hacking benchmarks, PBRR consistently outperforms baselines that learn a reward function from scratch from preferences or modify the proxy reward function using other approaches, requiring substantially fewer preferences to learn high performing policies. 

---
# Offline and Online KL-Regularized RLHF under Differential Privacy 

**Authors**: Yulian Wu, Rushil Thareja, Praneeth Vepakomma, Francesco Orabona  

**Link**: [PDF](https://arxiv.org/pdf/2510.13512)  

**Abstract**: In this paper, we study the offline and online settings of reinforcement learning from human feedback (RLHF) with KL-regularization -- a widely used objective function in large language model alignment -- under the $\epsilon$ local differential privacy ($\epsilon$-LDP) model on the label of the human preference. In the offline setting, we design an algorithm based on the principle of pessimism and derive a new suboptimality gap of $\tilde{O}(1/[(e^\epsilon-1)^2 n])$ on the KL-regularized objective under single-policy concentrability. We also prove its optimality by providing a matching lower bound where $n$ is the sample size.
In the online setting, we are the first one to theoretically investigate the problem of KL-regularized RLHF with LDP. We design an optimism-based algorithm and derive a logarithmic regret bound of $O(d_{\mathcal{F}}\log (N_{\mathcal{F}}\cdot T) /(e^\epsilon-1)^2 )$, where $T$ is the total time step, $N_{\mathcal{F}}$ is cardinality of the reward function space $\mathcal{F}$ and $d_{\mathcal{F}}$ is a variant of eluder dimension for RLHF. As a by-product of our analysis, our results also imply the first analysis for online KL-regularized RLHF without privacy. We implement our algorithm in the offline setting to verify our theoretical results and release our open source code at: this https URL. 

---
# DriveCritic: Towards Context-Aware, Human-Aligned Evaluation for Autonomous Driving with Vision-Language Models 

**Authors**: Jingyu Song, Zhenxin Li, Shiyi Lan, Xinglong Sun, Nadine Chang, Maying Shen, Joshua Chen, Katherine A. Skinner, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13108)  

**Abstract**: Benchmarking autonomous driving planners to align with human judgment remains a critical challenge, as state-of-the-art metrics like the Extended Predictive Driver Model Score (EPDMS) lack context awareness in nuanced scenarios. To address this, we introduce DriveCritic, a novel framework featuring two key contributions: the DriveCritic dataset, a curated collection of challenging scenarios where context is critical for correct judgment and annotated with pairwise human preferences, and the DriveCritic model, a Vision-Language Model (VLM) based evaluator. Fine-tuned using a two-stage supervised and reinforcement learning pipeline, the DriveCritic model learns to adjudicate between trajectory pairs by integrating visual and symbolic context. Experiments show DriveCritic significantly outperforms existing metrics and baselines in matching human preferences and demonstrates strong context awareness. Overall, our work provides a more reliable, human-aligned foundation to evaluating autonomous driving systems. 

---
# Epistemic-aware Vision-Language Foundation Model for Fetal Ultrasound Interpretation 

**Authors**: Xiao He, Huangxuan Zhao, Guojia Wan, Wei Zhou, Yanxing Liu, Juhua Liu, Yongchao Xu, Yong Luo, Dacheng Tao, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.12953)  

**Abstract**: Recent medical vision-language models have shown promise on tasks such as VQA, report generation, and anomaly detection. However, most are adapted to structured adult imaging and underperform in fetal ultrasound, which poses challenges of multi-view image reasoning, numerous diseases, and image diversity. To bridge this gap, we introduce FetalMind, a medical AI system tailored to fetal ultrasound for both report generation and diagnosis. Guided by clinical workflow, we propose Salient Epistemic Disentanglement (SED), which injects an expert-curated bipartite graph into the model to decouple view-disease associations and to steer preference selection along clinically faithful steps via reinforcement learning. This design mitigates variability across diseases and heterogeneity across views, reducing learning bottlenecks while aligning the model's inference with obstetric practice. To train FetalMind at scale, we curate FetalSigma-1M dataset, the first large-scale fetal ultrasound report corpus, comprising 20K reports from twelve medical centers, addressing the scarcity of domain data. Extensive experiments show that FetalMind outperforms open- and closed-source baselines across all gestational stages, achieving +14% average gains and +61.2% higher accuracy on critical conditions while remaining efficient, stable, and scalable. Project Page: this https URL. 

---
# ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering 

**Authors**: Simon Lupart, Mohammad Aliannejadi, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2510.13312)  

**Abstract**: We present ChatR1, a reasoning framework based on reinforcement learning (RL) for conversational question answering (CQA). Reasoning plays an important role in CQA, where user intent evolves across dialogue turns, and utterances are often underspecified, requiring contextual interpretation, query reformulation, and dynamic coordination between retrieval and generation. Unlike static `rewrite, retrieve, and generate' pipelines, ChatR1 interleaves search and reasoning across turns, enabling exploratory and adaptive behaviors learned through RL. To address the challenge of sparse and delayed rewards in RL, we propose an intent-aware reward that provides turn-level feedback by aligning retrieval and reasoning with evolving user goals. Our proposed ChatR1 demonstrates strong performance on both 3B and 7B model backbones, outperforming competitive models on five CQA datasets, measured by different metrics (F1, BERTScore, and LLM-as-judge). We include a diverse set of CQA datasets to cover topic shifts, evolving intents, mixed-initiative dialogues, and multi-document grounding, testing ChatR1's performance from various aspects. Ablation studies confirm the effectiveness of the intent-aware reward. Our analyses further reveal diverse reasoning trajectories and effective use of the search tool. ChatR1 also generalizes robustly across domains, demonstrating that RL-based reasoning enables more flexible and context-sensitive behavior than static CQA pipelines. 

---
# Breadcrumbs Reasoning: Memory-Efficient Reasoning with Compression Beacons 

**Authors**: Giovanni Monea, Yair Feldman, Shankar Padmanabhan, Kianté Brantley, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.13797)  

**Abstract**: The scalability of large language models for long-context reasoning is severely constrained by the linear growth of their Transformer key-value cache, which incurs significant memory and computational costs. We posit that as a model generates reasoning tokens, the informational value of past generated tokens diminishes, creating an opportunity for compression. In this work, we propose to periodically compress the generation KV cache with a learned, special-purpose token and evict compressed entries. We train the model to perform this compression via a modified joint distillation and reinforcement learning (RL) framework. Our training method minimizes overhead over the conventional RL process, as it leverages RL outputs for distillation. Empirically, our method achieves a superior memory-accuracy Pareto frontier compared to both the model without cache compression and training-free compression techniques. 

---
# Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables Fine-Grained Policy Optimization 

**Authors**: Yang Li, Zhichen Dong, Yuhan Sun, Weixun Wang, Shaopan Xiong, Yijia Luo, Jiashun Liu, Han Lu, Jiamang Wang, Wenbo Su, Bo Zheng, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13554)  

**Abstract**: The reasoning pattern of Large language models (LLMs) remains opaque, and Reinforcement learning (RL) typically applies uniform credit across an entire generation, blurring the distinction between pivotal and routine steps. This work positions attention as a privileged substrate that renders the internal logic of LLMs legible, not merely as a byproduct of computation, but as a mechanistic blueprint of reasoning itself. We first distinguish attention heads between locally and globally focused information processing and reveal that locally focused heads produce a sawtooth pattern near the diagonal indicating phrasal chunks, while globally focused heads expose tokens that exert broad downstream influence over future tokens. We formalize these with two metrics: 1) Windowed Average Attention Distance, which measures the extent of backward attention within a clipped window; 2) Future Attention Influence, which quantifies a token's global importance as the average attention it receives from subsequent tokens. Taken together, these signals reveal a recurring preplan-and-anchor mechanism, where the model first performs a long-range contextual reference to generate an introductory token, which is immediately followed by or coincides with a semantic anchor token that organizes subsequent reasoning. Leveraging these insights, we introduce three novel RL strategies that dynamically perform targeted credit assignment to critical nodes (preplan tokens, anchor tokens, and their temporal coupling) and show consistent performance gains across various reasoning tasks. By aligning optimization with the model's intrinsic reasoning rhythm, we aim to transform opaque optimization into an actionable structure-aware process, hoping to offer a potential step toward more transparent and effective optimization of LLM reasoning. 

---
# Beyond Single-Reward: Multi-Pair, Multi-Perspective Preference Optimization for Machine Translation 

**Authors**: Hao Wang, Linlong Xu, Heng Liu, Yangyang Liu, Xiaohu Zhao, Bo Zeng, Liangying Shao, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13434)  

**Abstract**: Direct Preference Optimization (DPO) is a powerful paradigm for aligning Large Language Models (LLMs) to human preferences in Machine Translation (MT), but current methods are hindered by two fundamental challenges: (1) flawed reward signals from Quality Estimation (QE) models that overlook critical errors like translation hallucination, and (2) inefficient data utilization that discards valuable learning signals by selecting only a single win-loss pair. To address these limitations, we introduce M^2PO: Multi-Pair, Multi-Perspective Preference Optimization. Our framework integrates a multi-perspective reward engine that creates a more robust signal by combining two key viewpoints: a new hallucination penalty for factuality, and an innovative dynamic quality score that adaptively fuses external evaluations with the model's own evolving judgment. This is synergistically paired with a multi-pair construction strategy that systematically creates a comprehensive set of preference pairs from the entire pool of translation candidates. This synergistic approach ensures the model learns from a richer spectrum of quality trade-offs, leading to more robust and faithful translations. On challenging WMT21-22 benchmarks, M^2PO substantially outperforms existing preference optimization methods and demonstrates highly competitive performance against leading proprietary LLMs. 

---
# Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation 

**Authors**: Zhichao Xu, Zongyu Wu, Yun Zhou, Aosong Feng, Kang Zhou, Sangmin Woo, Kiran Ramnath, Yijun Tian, Xuan Qi, Weikang Qiu, Lin Lee Cheong, Haibo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.13272)  

**Abstract**: Inspired by the success of reinforcement learning (RL) in Large Language Model (LLM) training for domains like math and code, recent works have begun exploring how to train LLMs to use search engines more effectively as tools for retrieval-augmented generation. Although these methods achieve performance improvement across QA benchmarks, many prioritize final answer correctness while overlooking the quality of intermediate reasoning steps, which may lead to chain-of-thought unfaithfulness. In this paper, we first introduce a comprehensive evaluation framework for evaluating RL-based search agents, covering three distinct faithfulness metrics: information-think faithfulness, think-answer faithfulness, and think-search faithfulness. Our evaluations reveal that a prototypical RL-based search agent, Search-R1, has significant room for improvement in this regard. To foster faithful reasoning, we introduce VERITAS (Verifying Entailed Reasoning through Intermediate Traceability in Agentic Search), a novel framework that integrates fine-grained faithfulness rewards into the reinforcement learning process. Our experiments show that models trained with VERITAS not only significantly improve reasoning faithfulness, but also achieve comparable task performance across seven QA benchmarks. 

---
# On the Role of Preference Variance in Preference Optimization 

**Authors**: Jiacheng Guo, Zihao Li, Jiahao Qiu, Yue Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13022)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an important approach for learning from human preferences in aligning large language models (LLMs). However, collecting human preference data is costly and inefficient, motivating methods to reduce the required annotations. In this work, we investigate the impact of \emph{preference variance} (PVar), which measures the variance in model preferences when comparing pairs of responses, on the effectiveness of DPO training. We provide a theoretical insight by establishing an upper bound on the DPO gradient norm for any given prompt, showing it is controlled by the PVar of that prompt. This implies that prompts with low PVar can only produce small gradient updates, making them less valuable for learning. We validate this finding by fine-tuning LLMs with preferences generated by a reward model, evaluating on two benchmarks (AlpacaEval 2.0 and Arena-Hard). Experimental results demonstrate that prompts with higher PVar outperform randomly selected prompts or those with lower PVar. We also show that our PVar-based selection method is robust, when using smaller reward models (1B, 3B) for selection. Notably, in a separate experiment using the original human annotations from the UltraFeedback dataset, we found that training on only the top 10\% of prompts with the highest PVar yields better evaluation performance than training on the full dataset, highlighting the importance of preference variance in identifying informative examples for efficient LLM alignment. 

---
# From Literal to Liberal: A Meta-Prompting Framework for Eliciting Human-Aligned Exception Handling in Large Language Models 

**Authors**: Imran Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.12864)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed as the reasoning engines for agentic AI systems, yet they exhibit a critical flaw: a rigid adherence to explicit rules that leads to decisions misaligned with human common sense and intent. This "rule-rigidity" is a significant barrier to building trustworthy autonomous agents. While prior work has shown that supervised fine-tuning (SFT) with human explanations can mitigate this issue, SFT is computationally expensive and inaccessible to many practitioners. To address this gap, we introduce the Rule-Intent Distinction (RID) Framework, a novel, low-compute meta-prompting technique designed to elicit human-aligned exception handling in LLMs in a zero-shot manner. The RID framework provides the model with a structured cognitive schema for deconstructing tasks, classifying rules, weighing conflicting outcomes, and justifying its final decision. We evaluated the RID framework against baseline and Chain-of-Thought (CoT) prompting on a custom benchmark of 20 scenarios requiring nuanced judgment across diverse domains. Our human-verified results demonstrate that the RID framework significantly improves performance, achieving a 95% Human Alignment Score (HAS), compared to 80% for the baseline and 75% for CoT. Furthermore, it consistently produces higher-quality, intent-driven reasoning. This work presents a practical, accessible, and effective method for steering LLMs from literal instruction-following to liberal, goal-oriented reasoning, paving the way for more reliable and pragmatic AI agents. 

---
