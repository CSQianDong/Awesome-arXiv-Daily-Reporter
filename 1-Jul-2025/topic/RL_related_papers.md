# SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning 

**Authors**: Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, Wee Sun Lee, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2506.24119)  

**Abstract**: Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, eliminating the need for human supervision. Through self-play, SPIRAL generates an infinite curriculum of progressively challenging problems as models must constantly adapt to stronger opponents. To enable this self-play training at scale, We implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. Using SPIRAL, self-play on zero-sum games produces reasoning capabilities that transfer broadly. Training Qwen3-4B-Base on Kuhn Poker alone achieves 8.6% improvement on math and 8.4% on general reasoning, outperforming SFT on 25,000 expert game trajectories. Analysis reveals that this transfer occurs through three cognitive patterns: systematic decomposition, expected value calculation, and case-by-case analysis. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) further enhances performance as each game develops distinct reasoning strengths. Applying SPIRAL to a strong reasoning model (DeepSeek-R1-Distill-Qwen-7B) can still lead to 2.0% average improvement. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities, highlighting a promising direction for autonomous reasoning development. 

---
# Self-correcting Reward Shaping via Language Models for Reinforcement Learning Agents in Games 

**Authors**: António Afonso, Iolanda Leite, Alessandro Sestini, Florian Fuchs, Konrad Tollmar, Linus Gisslén  

**Link**: [PDF](https://arxiv.org/pdf/2506.23626)  

**Abstract**: Reinforcement Learning (RL) in games has gained significant momentum in recent years, enabling the creation of different agent behaviors that can transform a player's gaming experience. However, deploying RL agents in production environments presents two key challenges: (1) designing an effective reward function typically requires an RL expert, and (2) when a game's content or mechanics are modified, previously tuned reward weights may no longer be optimal. Towards the latter challenge, we propose an automated approach for iteratively fine-tuning an RL agent's reward function weights, based on a user-defined language based behavioral goal. A Language Model (LM) proposes updated weights at each iteration based on this target behavior and a summary of performance statistics from prior training rounds. This closed-loop process allows the LM to self-correct and refine its output over time, producing increasingly aligned behavior without the need for manual reward engineering. We evaluate our approach in a racing task and show that it consistently improves agent performance across iterations. The LM-guided agents show a significant increase in performance from $9\%$ to $74\%$ success rate in just one iteration. We compare our LM-guided tuning against a human expert's manual weight design in the racing task: by the final iteration, the LM-tuned agent achieved an $80\%$ success rate, and completed laps in an average of $855$ time steps, a competitive performance against the expert-tuned agent's peak $94\%$ success, and $850$ time steps. 

---
# Improving Rationality in the Reasoning Process of Language Models through Self-playing Game 

**Authors**: Pinzheng Wang, Juntao Li, Zecheng Tang, Haijia Gui, Min zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22920)  

**Abstract**: Large language models (LLMs) have demonstrated considerable reasoning abilities in various tasks such as mathematics and coding. However, recent studies indicate that even the best models lack true comprehension of their reasoning processes. In this paper, we explore how self-play can enhance the rationality of models in the reasoning process without supervision from humans or superior models. We design a Critic-Discernment Game(CDG) in which a prover first provides a solution to a given problem and is subsequently challenged by critiques of its solution. These critiques either aim to assist or mislead the prover. The objective of the prover is to maintain the correct answer when faced with misleading comments, while correcting errors in response to constructive feedback. Our experiments on tasks involving mathematical reasoning, stepwise error detection, self-correction, and long-chain reasoning demonstrate that CDG training can significantly improve the ability of well-aligned LLMs to comprehend their reasoning process. 

---
# Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model 

**Authors**: Bowen Ding, Yuhan Chen, Futing Wang, Lingfeng Ming, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23840)  

**Abstract**: Large Reasoning Models (LRMs) excel at solving complex problems but face an overthinking dilemma. When handling simple tasks, they often produce verbose responses overloaded with thinking tokens (e.g., wait, however). These tokens trigger unnecessary high-level reasoning behaviors like reflection and backtracking, reducing efficiency. In this work, our pilot study reveals that these thinking-token-induced behaviors are not essential for effective problem-solving and may even hinder correct reasoning within constrained token budgets. We identify this phenomenon as the thinking trap. To mitigate this issue, we propose Dual Policy Preference Optimization (DuP-PO), a novel algorithm featuring: (1) A rollout sampling strategy that guarantees balanced exposure to responses with and without thinking tokens; (2) A fine-grained advantage control technique to dynamically regulate the prediction of target tokens; (3) A policy shaping method ensuring stable gradient contributions from thinking tokens. Experimental results on five popular math reasoning benchmarks show that DuP-PO performs well on the popular LRM, which significantly improves their token efficiency during reasoning, while achieving superior performance of the base model. 

---
# Semantic-guided Diverse Decoding for Large Language Model 

**Authors**: Weijie Shi, Yue Cui, Yaguang Wu, Jingzhi Fang, Shibo Zhang, Mengze Li, Sirui Han, Jia Zhu, Jiajie Xu, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23601)  

**Abstract**: Diverse decoding of large language models is crucial for applications requiring multiple semantically distinct responses, yet existing methods primarily achieve lexical rather than semantic diversity. This limitation significantly constrains Best-of-N strategies, group-based reinforcement learning, and data synthesis. While temperature sampling and diverse beam search modify token distributions or apply n-gram penalties, they fail to ensure meaningful semantic differentiation. We introduce Semantic-guided Diverse Decoding (SemDiD), operating directly in embedding space that balances quality with diversity through three complementary mechanisms: orthogonal directional guidance, dynamic inter-group repulsion, and position-debiased probability assessment. SemDiD harmonizes these competing objectives using adaptive gain functions and constraint optimization, ensuring both quality thresholds and maximal semantic differentiation. Experiments show SemDiD consistently outperforms existing methods, improving Best-of-N coverage by 1.4-5.2% across diverse tasks and accelerating RLHF training convergence by 15% while increasing accuracy by up to 2.1%. 

---
# Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning 

**Authors**: Zhaoye Fei, Li Ji, Siyin Wang, Junhao Shi, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23127)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they face significant challenges in embodied task planning scenarios that require continuous environmental understanding and action generation. Existing approaches generate open-loop action scripts based on static knowledge, making it difficult to learn causal relationships between actions and environmental feedback, particularly in partially observable environments. We introduce Embodied Planner-R1, a novel outcome-driven reinforcement learning framework that enables LLMs to develop interactive capabilities through autonomous exploration with minimal supervision. Our framework incorporates three key innovations: (1) Without human annotations, we employ pure reinforcement learning with group rollout, incorporating in-environment interaction through parallel exploration; (2) completion-driven sparse reward; and (3) Interactive Policy Optimization (IPO) for efficient learning from grouped trajectories. Across two challenging text-based Embodied planning benchmarks, Embodied Planner-R1 achieves impressive completion rates of 97.78% on ALFWorld and 79.92% on ScienceWorld, surpassing prior methods by a large margin, and suffers only a -3.66% drop in previously unseen environments, evidencing strong generalization. 

---
# Listener-Rewarded Thinking in VLMs for Image Preferences 

**Authors**: Alexander Gambashidze, Li Pengyi, Matvey Skripkin, Andrey Galichin, Anton Gusarov, Konstantin Sobolev, Andrey Kuznetsov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2506.22832)  

**Abstract**: Training robust and generalizable reward models for human visual preferences is essential for aligning text-to-image and text-to-video generative models with human intent. However, current reward models often fail to generalize, and supervised fine-tuning leads to memorization, demanding complex annotation pipelines. While reinforcement learning (RL), specifically Group Relative Policy Optimization (GRPO), improves generalization, we uncover a key failure mode: a significant drop in reasoning accuracy occurs when a model's reasoning trace contradicts that of an independent, frozen vision-language model ("listener") evaluating the same output. To address this, we introduce a listener-augmented GRPO framework. Here, the listener re-evaluates the reasoner's chain-of-thought to provide a dense, calibrated confidence score, shaping the RL reward signal. This encourages the reasoner not only to answer correctly, but to produce explanations that are persuasive to an independent model. Our listener-shaped reward scheme achieves best accuracy on the ImageReward benchmark (67.4%), significantly improves out-of-distribution (OOD) performance on a large-scale human preference dataset (1.2M votes, up to +6% over naive reasoner), and reduces reasoning contradictions compared to strong GRPO and SFT baselines. These results demonstrate that listener-based rewards provide a scalable, data-efficient path to aligning vision-language models with nuanced human preferences. We will release our reasoning model here: this https URL. 

---
# Teaching Models to Verbalize Reward Hacking in Chain-of-Thought Reasoning 

**Authors**: Miles Turpin, Andy Arditi, Marvin Li, Joe Benton, Julian Michael  

**Link**: [PDF](https://arxiv.org/pdf/2506.22777)  

**Abstract**: Language models trained with RL can engage in reward hacking--exploiting unintended strategies for high reward--without revealing this behavior in their chain-of-thought reasoning, making detection difficult and posing risks for high-stakes applications. We propose verbalization fine-tuning (VFT), a pre-RL intervention that trains models to explicitly acknowledge when they are influenced by prompt cues--hints which point to incorrect answers (e.g., "a Stanford professor thinks the answer is A"). To evaluate VFT, we subsequently train models with RL on environments where held-out prompt cues signal which incorrect answers will receive high reward, incentivizing models to reward hack by exploiting cues instead of reasoning correctly. We measure how often models exploit these cues without verbalizing it. After RL, only 6% of the VFT-trained model's responses consist of undetected reward hacks. In comparison, when we perform RL without VFT, the rate of undetected reward hacks goes up to 88%; with a debiasing baseline intervention, this increases further to 99%. VFT achieves this by substantially increasing how often models verbalize the influence of cues--from 8% to 42% after VFT, and up to 94% after RL--while baselines remain low even after RL (10% and 1%). Our results show that teaching models to explicitly verbalize reward hacking behavior before RL significantly improves their detection, offering a practical path toward more transparent and safe AI systems. 

---
# The Hidden Link Between RLHF and Contrastive Learning 

**Authors**: Xufei Lv, Haoyuan Sun, Xuefeng Bai, Min Zhang, Houde Liu, Kehai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.22578)  

**Abstract**: Alignment of large language models (LLMs) with human values has recently garnered significant attention, with prominent examples including the canonical yet costly Reinforcement Learning from Human Feedback (RLHF) and the simple Direct Preference Optimization (DPO). In this work, we demonstrate that both RLHF and DPO can be interpreted from the perspective of mutual information (MI) maximization, uncovering a profound connection to contrastive learning. Within this framework, both RLHF and DPO can be viewed as methods that perform contrastive learning based on the positive and negative samples derived from the base model, leveraging the Donsker-Varadhan (DV) lower bound on MI (equivalently, the MINE estimator). This paradigm further explains why RLHF may not intrinsically incentivize reasoning capacities in LLMs beyond what is already present in the base model. Building on this perspective, we replace the DV/MINE bound with the Jensen-Shannon MI estimator and propose Mutual Information Optimization (MIO). Comprehensive theoretical analysis and extensive empirical evaluations demonstrate that MIO mitigates the late-stage decline in chosen-likelihood observed in DPO, achieving competitive or superior performance across various challenging reasoning and mathematical benchmarks. We will release the model and code upon acceptance. 

---
# AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text 

**Authors**: Chenyang Shao, Tianxing Li, Chenhao Pu, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22508)  

**Abstract**: In today's digital world, casual user-generated content often contains subtle cues that may inadvertently expose sensitive personal attributes. Such risks underscore the growing importance of effective text anonymization to safeguard individual privacy. However, existing methods either rely on rigid replacements that damage utility or cloud-based LLMs that are costly and pose privacy risks. To address these issues, we explore the use of locally deployed smaller-scale language models (SLMs) for anonymization. Yet training effective SLMs remains challenging due to limited high-quality supervision. To address the challenge, we propose AgentStealth, a self-reinforcing LLM anonymization this http URL, we introduce an adversarial anonymization workflow enhanced by In-context Contrastive Learning and Adaptive Utility-Aware Control. Second, we perform supervised adaptation of SLMs using high-quality data collected from the workflow, which includes both anonymization and attack signals. Finally, we apply online reinforcement learning where the model leverages its internal adversarial feedback to iteratively improve anonymization performance. Experiments on two datasets show that our method outperforms baselines in both anonymization effectiveness (+12.3%) and utility (+6.8%). Our lightweight design supports direct deployment on edge devices, avoiding cloud reliance and communication-based privacy risks. Our code is open-source at this https URL. 

---
# Auto-TA: Towards Scalable Automated Thematic Analysis (TA) via Multi-Agent Large Language Models with Reinforcement Learning 

**Authors**: Seungjun Yi, Joakim Nguyen, Huimin Xu, Terence Lim, Andrew Well, Mia Markey, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.23998)  

**Abstract**: Congenital heart disease (CHD) presents complex, lifelong challenges often underrepresented in traditional clinical metrics. While unstructured narratives offer rich insights into patient and caregiver experiences, manual thematic analysis (TA) remains labor-intensive and unscalable. We propose a fully automated large language model (LLM) pipeline that performs end-to-end TA on clinical narratives, which eliminates the need for manual coding or full transcript review. Our system employs a novel multi-agent framework, where specialized LLM agents assume roles to enhance theme quality and alignment with human analysis. To further improve thematic relevance, we optionally integrate reinforcement learning from human feedback (RLHF). This supports scalable, patient-centered analysis of large qualitative datasets and allows LLMs to be fine-tuned for specific clinical contexts. 

---
# L0: Reinforcement Learning to Become General Agents 

**Authors**: Junjie Zhang, Jingyi Xi, Zhuoyang Song, Junyu Lu, Yuhua Ke, Ting Sun, Yukun Yang, Jiaxing Zhang, Songxin Zhang, Zejian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.23667)  

**Abstract**: Training large language models (LLMs) to act as autonomous agents for multi-turn, long-horizon tasks remains significant challenges in scalability and training efficiency. To address this, we introduce L-Zero (L0), a scalable, end-to-end training pipeline for general-purpose agents. Featuring a low-cost, extensible, and sandboxed concurrent agent worker pool, L0 lowers the barrier for applying reinforcement learning in complex environments. We also introduce NB-Agent, the agent scaffold within L0, which operates in a "code-as-action" fashion via a Read-Eval-Print-Loop (REPL). We evaluate L0 on factuality question-answering benchmarks. Our experiments demonstrate that a base model can develop robust problem-solving skills using solely Reinforcement Learning with Verifiable Rewards (RLVR). On the Qwen2.5-7B-Instruct model, our method boosts accuracy on SimpleQA from 30 % to 80 % and on HotpotQA from 22 % to 41 %. We have open-sourced the entire L0 system, including our L0 series models, the NB-Agent, a complete training pipeline, and the corresponding training recipes on (this https URL). 

---
# Generalist Reward Models: Found Inside Large Language Models 

**Authors**: Yi-Chen Li, Tian Xu, Yang Yu, Xuqin Zhang, Xiong-Hui Chen, Zhongxiang Ling, Ningjing Chao, Lei Yuan, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23235)  

**Abstract**: The alignment of Large Language Models (LLMs) is critically dependent on reward models trained on costly human preference data. While recent work explores bypassing this cost with AI feedback, these methods often lack a rigorous theoretical foundation. In this paper, we discover that a powerful generalist reward model is already latently present within any LLM trained via standard next-token prediction. We prove that this endogenous reward is not a heuristic, but is theoretically equivalent to a reward function learned through offline inverse reinforcement learning. This connection allows us to directly elicit a high-quality reward signal from a base (pre-trained or supervised fine-tuned) model without any further training. Critically, we also prove that subsequent reinforcement learning using this endogenous reward leads to a policy with a provably superior error bound compared to the base model. To our best knowledge, this is the first theoretical proof of the effectiveness of reinforcement learning for LLMs. Our experiments validate this theory, demonstrating that our method not only outperforms existing LLM-as-a-judge approaches but can also surpass explicitly trained reward models. These findings suggest that the reward modeling stage can be replaced by a principled method of eliciting the knowledge already captured during pre-training, heralding a more efficient, powerful, and scalable paradigm for LLMs alignment as well as multi-modal models. 

---
# Logit-Gap Steering: Efficient Short-Suffix Jailbreaks for Aligned Large Language Models 

**Authors**: Tung-Ling Li, Hongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.24056)  

**Abstract**: We introduce logit-gap steering, a fast jailbreak framework that casts the refusal-affirmation gap of RLHF-aligned language models as a single pass over the vocabulary. A forward-computable score blends gap reduction with lightweight proxies for KL penalty and reward shift, allowing a "sort-sum-stop" sweep to complete in under a second and return a short suffix--two orders of magnitude fewer model calls than beam or gradient attacks. The same suffix generalises to unseen prompts and scales from 0.5 B to 70 B checkpoints, lifting one-shot attack success from baseline levels to 80-100% while preserving topical coherence. Beyond efficiency, these suffixes expose sentence-boundary reward cliffs and other alignment artefacts, offering a lightweight probe into how safety tuning reshapes internal representations. 

---
