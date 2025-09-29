# TrueGradeAI: Retrieval-Augmented and Bias-Resistant AI for Transparent and Explainable Digital Assessments 

**Authors**: Rakesh Thakur, Shivaansh Kaushik, Gauri Chopra, Harsh Rohilla  

**Link**: [PDF](https://arxiv.org/pdf/2509.22516)  

**Abstract**: This paper introduces TrueGradeAI, an AI-driven digital examination framework designed to overcome the shortcomings of traditional paper-based assessments, including excessive paper usage, logistical complexity, grading delays, and evaluator bias. The system preserves natural handwriting by capturing stylus input on secure tablets and applying transformer-based optical character recognition for transcription. Evaluation is conducted through a retrieval-augmented pipeline that integrates faculty solutions, cache layers, and external references, enabling a large language model to assign scores with explicit, evidence-linked reasoning. Unlike prior tablet-based exam systems that primarily digitize responses, TrueGradeAI advances the field by incorporating explainable automation, bias mitigation, and auditable grading trails. By uniting handwriting preservation with scalable and transparent evaluation, the framework reduces environmental costs, accelerates feedback cycles, and progressively builds a reusable knowledge base, while actively working to mitigate grading bias and ensure fairness in assessment. 

---
# StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models 

**Authors**: Chenyu Zhou, Tianyi Xu, Jianghao Lin, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2509.22558)  

**Abstract**: Large Language Models (LLMs) have shown promising capabilities for solving Operations Research (OR) problems. While reinforcement learning serves as a powerful paradigm for LLM training on OR problems, existing works generally face two key limitations. First, outcome reward suffers from the credit assignment problem, where correct final answers can reinforce flawed reasoning. Second, conventional discriminative process supervision is myopic, failing to evaluate the interdependent steps of OR modeling holistically. To this end, we introduce StepORLM, a novel self-evolving framework with generative process supervision. At its core, StepORLM features a co-evolutionary loop where a policy model and a generative process reward model (GenPRM) iteratively improve on each other. This loop is driven by a dual-feedback mechanism: definitive, outcome-based verification from an external solver, and nuanced, holistic process evaluation from the GenPRM. The combined signal is used to align the policy via Weighted Direct Preference Optimization (W-DPO) and simultaneously refine the GenPRM. Our resulting 8B-parameter StepORLM establishes a new state-of-the-art across six benchmarks, significantly outperforming vastly larger generalist models, agentic methods, and specialized baselines. Moreover, the co-evolved GenPRM is able to act as a powerful and universally applicable process verifier, substantially boosting the inference scaling performance of both our own model and other existing LLMs. 

---
# DeepTravel: An End-to-End Agentic Reinforcement Learning Framework for Autonomous Travel Planning Agents 

**Authors**: Yansong Ning, Rui Liu, Jun Wang, Kai Chen, Wei Li, Jun Fang, Kan Zheng, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21842)  

**Abstract**: Travel planning (TP) agent has recently worked as an emerging building block to interact with external tools and resources for travel itinerary generation, ensuring enjoyable user experience. Despite its benefits, existing studies rely on hand craft prompt and fixed agent workflow, hindering more flexible and autonomous TP agent. This paper proposes DeepTravel, an end to end agentic reinforcement learning framework for building autonomous travel planning agent, capable of autonomously planning, executing tools, and reflecting on tool responses to explore, verify, and refine intermediate actions in multi step reasoning. To achieve this, we first construct a robust sandbox environment by caching transportation, accommodation and POI data, facilitating TP agent training without being constrained by real world APIs limitations (e.g., inconsistent outputs). Moreover, we develop a hierarchical reward modeling system, where a trajectory level verifier first checks spatiotemporal feasibility and filters unsatisfied travel itinerary, and then the turn level verifier further validate itinerary detail consistency with tool responses, enabling efficient and precise reward service. Finally, we propose the reply augmented reinforcement learning method that enables TP agent to periodically replay from a failures experience buffer, emerging notable agentic capacity. We deploy trained TP agent on DiDi Enterprise Solutions App and conduct comprehensive online and offline evaluations, demonstrating that DeepTravel enables small size LLMs (e.g., Qwen3 32B) to significantly outperform existing frontier LLMs such as OpenAI o1, o3 and DeepSeek R1 in travel planning tasks. 

---
# Language Models Can Learn from Verbal Feedback Without Scalar Rewards 

**Authors**: Renjie Luo, Zichen Liu, Xiangyan Liu, Chao Du, Min Lin, Wenhu Chen, Wei Lu, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22638)  

**Abstract**: LLMs are often trained with RL from human or AI feedback, yet such methods typically compress nuanced feedback into scalar rewards, discarding much of their richness and inducing scale imbalance. We propose treating verbal feedback as a conditioning signal. Inspired by language priors in text-to-image generation, which enable novel outputs from unseen prompts, we introduce the feedback-conditional policy (FCP). FCP learns directly from response-feedback pairs, approximating the feedback-conditional posterior through maximum likelihood training on offline data. We further develop an online bootstrapping stage where the policy generates under positive conditions and receives fresh feedback to refine itself. This reframes feedback-driven learning as conditional generation rather than reward optimization, offering a more expressive way for LLMs to directly learn from verbal feedback. Our code is available at this https URL. 

---
# Variational Reasoning for Language Models 

**Authors**: Xiangxin Zhou, Zichen Liu, Haonan Wang, Chao Du, Min Lin, Chongxuan Li, Liang Wang, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22637)  

**Abstract**: We introduce a variational reasoning framework for language models that treats thinking traces as latent variables and optimizes them through variational inference. Starting from the evidence lower bound (ELBO), we extend it to a multi-trace objective for tighter bounds and propose a forward-KL formulation that stabilizes the training of the variational posterior. We further show that rejection sampling finetuning and binary-reward RL, including GRPO, can be interpreted as local forward-KL objectives, where an implicit weighting by model accuracy naturally arises from the derivation and reveals a previously unnoticed bias toward easier questions. We empirically validate our method on the Qwen 2.5 and Qwen 3 model families across a wide range of reasoning tasks. Overall, our work provides a principled probabilistic perspective that unifies variational inference with RL-style methods and yields stable objectives for improving the reasoning ability of language models. Our code is available at this https URL. 

---
# Towards Efficient Online Exploration for Reinforcement Learning with Human Feedback 

**Authors**: Gen Li, Yuling Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.22633)  

**Abstract**: Reinforcement learning with human feedback (RLHF), which learns a reward model from human preference data and then optimizes a policy to favor preferred responses, has emerged as a central paradigm for aligning large language models (LLMs) with human preferences. In this paper, we investigate exploration principles for online RLHF, where one seeks to adaptively collect new preference data to refine both the reward model and the policy in a data-efficient manner. By examining existing optimism-based exploration algorithms, we identify a drawback in their sampling protocol: they tend to gather comparisons that fail to reduce the most informative uncertainties in reward differences, and we prove lower bounds showing that such methods can incur linear regret over exponentially long horizons. Motivated by this insight, we propose a new exploration scheme that directs preference queries toward reducing uncertainty in reward differences most relevant to policy improvement. Under a multi-armed bandit model of RLHF, we establish regret bounds of order $T^{(\beta+1)/(\beta+2)}$, where $\beta>0$ is a hyperparameter that balances reward maximization against mitigating distribution shift. To our knowledge, this is the first online RLHF algorithm with regret scaling polynomially in all model parameters. 

---
# Quantile Advantage Estimation for Entropy-Safe Reasoning 

**Authors**: Junkang Wu, Kexin Huang, Jiancan Wu, An Zhang, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.22611)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) strengthens LLM reasoning, but training often oscillates between {entropy collapse} and {entropy explosion}. We trace both hazards to the mean baseline used in value-free RL (e.g., GRPO and DAPO), which improperly penalizes negative-advantage samples under reward outliers. We propose {Quantile Advantage Estimation} (QAE), replacing the mean with a group-wise K-quantile baseline. QAE induces a response-level, two-regime gate: on hard queries (p <= 1 - K) it reinforces rare successes, while on easy queries (p > 1 - K) it targets remaining failures. Under first-order softmax updates, we prove {two-sided entropy safety}, giving lower and upper bounds on one-step entropy change that curb explosion and prevent collapse. Empirically, this minimal modification stabilizes entropy, sparsifies credit assignment (with tuned K, roughly 80% of responses receive zero advantage), and yields sustained pass@1 gains on Qwen3-8B/14B-Base across AIME 2024/2025 and AMC 2023. These results identify {baseline design} -- rather than token-level heuristics -- as the primary mechanism for scaling RLVR. 

---
# WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback and Step-Level Reinforcement Learning 

**Authors**: Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Junting Pan, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22644)  

**Abstract**: Agent systems powered by large language models (LLMs) have demonstrated impressive performance on repository-level code-generation tasks. However, for tasks such as website codebase generation, which depend heavily on visual effects and user-interaction feedback, current code agents rely only on simple code execution for feedback and verification. This approach fails to capture the actual quality of the generated code. In this paper, we propose WebGen-Agent, a novel website-generation agent that leverages comprehensive and multi-level visual feedback to iteratively generate and refine the website codebase. Detailed and expressive text descriptions and suggestions regarding the screenshots and GUI-agent testing of the websites are generated by a visual language model (VLM), together with scores that quantify their quality. The screenshot and GUI-agent scores are further integrated with a backtracking and select-best mechanism, enhancing the performance of the agent. Utilizing the accurate visual scores inherent in the WebGen-Agent workflow, we further introduce \textit{Step-GRPO with Screenshot and GUI-agent Feedback} to improve the ability of LLMs to act as the reasoning engine of WebGen-Agent. By using the screenshot and GUI-agent scores at each step as the reward in Step-GRPO, we provide a dense and reliable process supervision signal, which effectively improves the model's website-generation ability. On the WebGen-Bench dataset, WebGen-Agent increases the accuracy of Claude-3.5-Sonnet from 26.4% to 51.9% and its appearance score from 3.0 to 3.9, outperforming the previous state-of-the-art agent system. Additionally, our Step-GRPO training approach increases the accuracy of Qwen2.5-Coder-7B-Instruct from 38.9% to 45.4% and raises the appearance score from 3.4 to 3.7. 

---
# Safety Compliance: Rethinking LLM Safety Reasoning through the Lens of Compliance 

**Authors**: Wenbin Hu, Huihao Jing, Haochen Shi, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.22250)  

**Abstract**: The proliferation of Large Language Models (LLMs) has demonstrated remarkable capabilities, elevating the critical importance of LLM safety. However, existing safety methods rely on ad-hoc taxonomy and lack a rigorous, systematic protection, failing to ensure safety for the nuanced and complex behaviors of modern LLM systems. To address this problem, we solve LLM safety from legal compliance perspectives, named safety compliance. In this work, we posit relevant established legal frameworks as safety standards for defining and measuring safety compliance, including the EU AI Act and GDPR, which serve as core legal frameworks for AI safety and data security in Europe. To bridge the gap between LLM safety and legal compliance, we first develop a new benchmark for safety compliance by generating realistic LLM safety scenarios seeded with legal statutes. Subsequently, we align Qwen3-8B using Group Policy Optimization (GRPO) to construct a safety reasoner, Compliance Reasoner, which effectively aligns LLMs with legal standards to mitigate safety risks. Our comprehensive experiments demonstrate that the Compliance Reasoner achieves superior performance on the new benchmark, with average improvements of +10.45% for the EU AI Act and +11.85% for GDPR. 

---
# Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding 

**Authors**: Shijing Hu, Jingyang Li, Zhihui Lu, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22134)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by letting a lightweight draft model propose multiple tokens that the target model verifies in parallel. Yet existing training objectives optimize only a single greedy draft path, while decoding follows a tree policy that re-ranks and verifies multiple branches. This draft policy misalignment limits achievable speedups. We introduce Group Tree Optimization (GTO), which aligns training with the decoding-time tree policy through two components: (i) Draft Tree Reward, a sampling-free objective equal to the expected acceptance length of the draft tree under the target model, directly measuring decoding performance; (ii) Group-based Draft Policy Training, a stable optimization scheme that contrasts trees from the current and a frozen reference draft model, forming debiased group-standardized advantages and applying a PPO-style surrogate along the longest accepted sequence for robust updates. We further prove that increasing our Draft Tree Reward provably improves acceptance length and speedup. Across dialogue (MT-Bench), code (HumanEval), and math (GSM8K), and multiple LLMs (e.g., LLaMA-3.1-8B, LLaMA-3.3-70B, Vicuna-1.3-13B, DeepSeek-R1-Distill-LLaMA-8B), GTO increases acceptance length by 7.4% and yields an additional 7.7% speedup over prior state-of-the-art EAGLE-3. By bridging draft policy misalignment, GTO offers a practical, general solution for efficient LLM inference. 

---
# Active Attacks: Red-teaming LLMs via Adaptive Environments 

**Authors**: Taeyoung Yun, Pierre-Luc St-Charles, Jinkyoo Park, Yoshua Bengio, Minsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21947)  

**Abstract**: We address the challenge of generating diverse attack prompts for large language models (LLMs) that elicit harmful behaviors (e.g., insults, sexual content) and are used for safety fine-tuning. Rather than relying on manual prompt engineering, attacker LLMs can be trained with reinforcement learning (RL) to automatically generate such prompts using only a toxicity classifier as a reward. However, capturing a wide range of harmful behaviors is a significant challenge that requires explicit diversity objectives. Existing diversity-seeking RL methods often collapse to limited modes: once high-reward prompts are found, exploration of new regions is discouraged. Inspired by the active learning paradigm that encourages adaptive exploration, we introduce \textit{Active Attacks}, a novel RL-based red-teaming algorithm that adapts its attacks as the victim evolves. By periodically safety fine-tuning the victim LLM with collected attack prompts, rewards in exploited regions diminish, which forces the attacker to seek unexplored vulnerabilities. This process naturally induces an easy-to-hard exploration curriculum, where the attacker progresses beyond easy modes toward increasingly difficult ones. As a result, Active Attacks uncovers a wide range of local attack modes step by step, and their combination achieves wide coverage of the multi-mode distribution. Active Attacks, a simple plug-and-play module that seamlessly integrates into existing RL objectives, unexpectedly outperformed prior RL-based methods -- including GFlowNets, PPO, and REINFORCE -- by improving cross-attack success rates against GFlowNets, the previous state-of-the-art, from 0.07% to 31.28% (a relative gain greater than $400\ \times$) with only a 6% increase in computation. Our code is publicly available \href{this https URL}{here}. 

---
# Geo-R1: Improving Few-Shot Geospatial Referring Expression Understanding with Reinforcement Fine-Tuning 

**Authors**: Zilun Zhang, Zian Guan, Tiancheng Zhao, Haozhan Shen, Tianyu Li, Yuxiang Cai, Zhonggen Su, Zhaojun Liu, Jianwei Yin, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21976)  

**Abstract**: Referring expression understanding in remote sensing poses unique challenges, as it requires reasoning over complex object-context relationships. While supervised fine-tuning (SFT) on multimodal large language models achieves strong performance with massive labeled datasets, they struggle in data-scarce scenarios, leading to poor generalization. To address this limitation, we propose Geo-R1, a reasoning-centric reinforcement fine-tuning (RFT) paradigm for few-shot geospatial referring. Geo-R1 enforces the model to first generate explicit, interpretable reasoning chains that decompose referring expressions, and then leverage these rationales to localize target objects. This "reason first, then act" process enables the model to make more effective use of limited annotations, enhances generalization, and provides interpretability. We validate Geo-R1 on three carefully designed few-shot geospatial referring benchmarks, where our model consistently and substantially outperforms SFT baselines. It also demonstrates strong cross-dataset generalization, highlighting its robustness. Code and data will be released at this http URL. 

---
# No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping 

**Authors**: Thanh-Long V. Le, Myeongho Jeon, Kim Vu, Viet Lai, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21880)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward - so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce RL with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR. 

---
# FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning 

**Authors**: Yizhou Zhang, Ning Lv, Teng Wang, Jisheng Dang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21792)  

**Abstract**: Group relative policy optimization (GRPO) has demonstrated significant potential in improving the reasoning capabilities of large language models (LLMs) via reinforcement learning. However, its practical deployment is impeded by an excessively slow training process, primarily attributed to the computationally intensive autoregressive generation of multiple responses per query, which makes the generation phase the primary performance bottleneck. Although speculative decoding presents a promising direction for acceleration, its direct application in GRPO achieves limited speedup under high-concurrency training conditions. To overcome this limitation, we propose a concurrency-aware speculative decoding framework that dynamically adjusts the drafting and verification strategy according to real-time concurrency levels, thereby maximizing the acceleration of the generation process. Furthermore, to address performance degradation arising from distributional drift between the evolving target model and the fixed draft model during training, we introduce an online draft learning mechanism that enables the draft model to continuously adapt using feedback signals from the target model. Experimental results across multiple mathematical reasoning datasets and models demonstrate that the proposed method achieves end-to-end speedups of 2.35x to 2.72x, significantly surpassing baseline approaches in efficiency. The code is available at this https URL. 

---
# Unlocking the Essence of Beauty: Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization 

**Authors**: Boyang Liu, Yifan Hu, Senjie Jin, Shihan Dou, Gonglei Shi, Jie Shao, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21871)  

**Abstract**: Multimodal large language models (MLLMs) are well suited to image aesthetic assessment, as they can capture high-level aesthetic features leveraging their cross-modal understanding capacity. However, the scarcity of multimodal aesthetic reasoning data and the inherently subjective nature of aesthetic judgment make it difficult for MLLMs to generate accurate aesthetic judgments with interpretable rationales. To this end, we propose Aes-R1, a comprehensive aesthetic reasoning framework with reinforcement learning (RL). Concretely, Aes-R1 integrates a pipeline, AesCoT, to construct and filter high-quality chain-of-thought aesthetic reasoning data used for cold-start. After teaching the model to generate structured explanations prior to scoring, we then employ the Relative-Absolute Policy Optimization (RAPO), a novel RL algorithm that jointly optimizes absolute score regression and relative ranking order, improving both per-image accuracy and cross-image preference judgments. Aes-R1 enables MLLMs to generate grounded explanations alongside faithful scores, thereby enhancing aesthetic scoring and reasoning in a unified framework. Extensive experiments demonstrate that Aes-R1 improves the backbone's average PLCC/SRCC by 47.9%/34.8%, surpassing state-of-the-art baselines of similar size. More ablation studies validate Aes-R1's robust generalization under limited supervision and in out-of-distribution scenarios. 

---
# Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment 

**Authors**: Hongbin Zhang, Kehai Chen, Xuefeng Bai, Yang Xiang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21798)  

**Abstract**: Reward models (RMs) are crucial for aligning large language models (LLMs) with diverse cultures. Consequently, evaluating their cultural awareness is essential for further advancing global alignment of LLMs. However, existing RM evaluations fall short in assessing cultural awareness due to the scarcity of culturally relevant evaluation datasets. To fill this gap, we propose Cultural Awareness Reward modeling Benchmark (CARB), covering 10 distinct cultures across 4 cultural domains. Our extensive evaluation of state-of-the-art RMs reveals their deficiencies in modeling cultural awareness and demonstrates a positive correlation between performance on CARB and downstream multilingual cultural alignment tasks. Further analysis identifies the spurious correlations within culture-aware reward modeling, wherein RM's scoring relies predominantly on surface-level features rather than authentic cultural nuance understanding. To address these, we propose Think-as-Locals to elicit deeper culturally grounded reasoning from generative RMs via reinforcement learning from verifiable rewards (RLVR) and employ well-designed rewards to ensure accurate preference judgments and high-quality structured evaluation criteria generation. Experimental results validate its efficacy in mitigating spurious features interference and advancing culture-aware reward modeling. 

---
# POLO: Preference-Guided Multi-Turn Reinforcement Learning for Lead Optimization 

**Authors**: Ziqing Wang, Yibo Wen, William Pattie, Xiao Luo, Weimin Wu, Jerry Yao-Chieh Hu, Abhishek Pandey, Han Liu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.21737)  

**Abstract**: Lead optimization in drug discovery requires efficiently navigating vast chemical space through iterative cycles to enhance molecular properties while preserving structural similarity to the original lead compound. Despite recent advances, traditional optimization methods struggle with sample efficiency-achieving good optimization performance with limited oracle evaluations. Large Language Models (LLMs) provide a promising approach through their in-context learning and instruction following capabilities, which align naturally with these iterative processes. However, existing LLM-based methods fail to leverage this strength, treating each optimization step independently. To address this, we present POLO (Preference-guided multi-turn Optimization for Lead Optimization), which enables LLMs to learn from complete optimization trajectories rather than isolated steps. At its core, POLO introduces Preference-Guided Policy Optimization (PGPO), a novel reinforcement learning algorithm that extracts learning signals at two complementary levels: trajectory-level optimization reinforces successful strategies, while turn-level preference learning provides dense comparative feedback by ranking intermediate molecules within each trajectory. Through this dual-level learning from intermediate evaluation, POLO achieves superior sample efficiency by fully exploiting each costly oracle call. Extensive experiments demonstrate that POLO achieves 84% average success rate on single-property tasks (2.3x better than baselines) and 50% on multi-property tasks using only 500 oracle evaluations, significantly advancing the state-of-the-art in sample-efficient molecular optimization. 

---
# Preemptive Detection and Steering of LLM Misalignment via Latent Reachability 

**Authors**: Sathwik Karnik, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2509.21528)  

**Abstract**: Large language models (LLMs) are now ubiquitous in everyday tools, raising urgent safety concerns about their tendency to generate harmful content. The dominant safety approach -- reinforcement learning from human feedback (RLHF) -- effectively shapes model behavior during training but offers no safeguards at inference time, where unsafe continuations may still arise. We propose BRT-Align, a reachability-based framework that brings control-theoretic safety tools to LLM inference. BRT-Align models autoregressive generation as a dynamical system in latent space and learn a safety value function via backward reachability, estimating the worst-case evolution of a trajectory. This enables two complementary mechanisms: (1) a runtime monitor that forecasts unsafe completions several tokens in advance, and (2) a least-restrictive steering filter that minimally perturbs latent states to redirect generation away from unsafe regions. Experiments across multiple LLMs and toxicity benchmarks demonstrate that BRT-Align provides more accurate and earlier detection of unsafe continuations than baselines. Moreover, for LLM safety alignment, BRT-Align substantially reduces unsafe generations while preserving sentence diversity and coherence. Qualitative results further highlight emergent alignment properties: BRT-Align consistently produces responses that are less violent, less profane, less offensive, and less politically biased. Together, these findings demonstrate that reachability analysis provides a principled and practical foundation for inference-time LLM safety. 

---
# Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training 

**Authors**: Junkai Zhang, Zihao Wang, Lin Gui, Swarnashree Mysore Sathyendra, Jaehwan Jeong, Victor Veitch, Wei Wang, Yunzhong He, Bing Liu, Lifeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21500)  

**Abstract**: Reinforcement fine-tuning (RFT) often suffers from \emph{reward over-optimization}, where a policy model hacks the reward signals to achieve high scores while producing low-quality outputs. Our theoretical analysis shows that the key lies in reward misspecification at the high-reward tail: the inability to reliably distinguish Excellent responses from merely Great ones. This motivate us to focus on the high-reward region. However, such tail examples are scarce under the base LLM. While off-policy exemplars (e.g. from stronger models or rewrites) are easier to obtain, naively training on them yields a misspecified reward for the policy we aim to align. To address this, we study rubric-based rewards. By design, rubrics can leverage off-policy examples while remaining insensitive to their artifacts. To elicit rubrics that capture the high-reward tail, we highlight the importance of distinguishing among great and diverse responses, and introduce a workflow to implement this idea. We empirically demonstrate that rubric-based rewards substantially mitigate reward over-optimization and deliver effective LLM post-training improvements. Our code can be accessed at this https URL . 

---
# Learning to Reason with Mixture of Tokens 

**Authors**: Adit Jain, Brendan Rappazzo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21482)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a leading approach for improving large language model (LLM) reasoning capabilities. Most current methods follow variants of Group Relative Policy Optimization, which samples multiple reasoning completions, scores them relative to each other, and adjusts the policy accordingly. However, these approaches invariably sample discrete tokens at each reasoning step, discarding the rich distributional information in the model's probability distribution over candidate tokens. While preserving and utilizing this distributional information has proven beneficial in non-RL settings, current RLVR methods seem to be unnecessarily constraining the reasoning search space by not using this information. To address this limitation, we investigate mixture-of-token generation (MoT-G) in RLVR. We present a unified framework that generalizes existing MoT-G approaches, including existing training-free methods that construct mixture embeddings as weighted sums over token embeddings, and extend RLVR to operate directly in this continuous mixture space for generating chain-of-thought. Evaluating two MoT-G variants on Reasoning-Gym, a suite of reasoning-intensive language tasks, we find that MoT--G methods achieve substantial improvements (5--35 \% gains on 7 out of 10 tasks) compared to standard decoding with the Qwen2.5-1.5B model, while reaching comparable accuracy with half the number of trajectories, suggesting improved training efficiency. Through comprehensive hidden-state and token-level analyses, we provide evidence that MoT--G's benefits may stem from its ability to maintain higher hidden-state entropy throughout the reasoning process and promote exploration in token space. 

---
# Automated Prompt Generation for Creative and Counterfactual Text-to-image Synthesis 

**Authors**: Aleksa Jelaca, Ying Jiao, Chang Tian, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2509.21375)  

**Abstract**: Text-to-image generation has advanced rapidly with large-scale multimodal training, yet fine-grained controllability remains a critical challenge. Counterfactual controllability, defined as the capacity to deliberately generate images that contradict common-sense patterns, remains a major challenge but plays a crucial role in enabling creativity and exploratory applications. In this work, we address this gap with a focus on counterfactual size (e.g., generating a tiny walrus beside a giant button) and propose an automatic prompt engineering framework that adapts base prompts into revised prompts for counterfactual images. The framework comprises three components: an image evaluator that guides dataset construction by identifying successful image generations, a supervised prompt rewriter that produces revised prompts, and a DPO-trained ranker that selects the optimal revised prompt. We construct the first counterfactual size text-image dataset and enhance the image evaluator by extending Grounded SAM with refinements, achieving a 114 percent improvement over its backbone. Experiments demonstrate that our method outperforms state-of-the-art baselines and ChatGPT-4o, establishing a foundation for future research on counterfactual controllability. 

---
# Think Socially via Cognitive Reasoning 

**Authors**: Jinfeng Zhou, Zheyu Chen, Shuai Wang, Quanyu Dai, Zhenhua Dong, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22546)  

**Abstract**: LLMs trained for logical reasoning excel at step-by-step deduction to reach verifiable answers. However, this paradigm is ill-suited for navigating social situations, which induce an interpretive process of analyzing ambiguous cues that rarely yield a definitive outcome. To bridge this gap, we introduce Cognitive Reasoning, a paradigm modeled on human social cognition. It formulates the interpretive process into a structured cognitive flow of interconnected cognitive units (e.g., observation or attribution), which combine adaptively to enable effective social thinking and responses. We then propose CogFlow, a complete framework that instills this capability in LLMs. CogFlow first curates a dataset of cognitive flows by simulating the associative and progressive nature of human thought via tree-structured planning. After instilling the basic cognitive reasoning capability via supervised fine-tuning, CogFlow adopts reinforcement learning to enable the model to improve itself via trial and error, guided by a multi-objective reward that optimizes both cognitive flow and response quality. Extensive experiments show that CogFlow effectively enhances the social cognitive capabilities of LLMs, and even humans, leading to more effective social decision-making. 

---
# ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation 

**Authors**: Jiho Kim, Junseong Choi, Woosog Chay, Daeun Kyung, Yeonsu Kwon, Yohan Jo, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21730)  

**Abstract**: As large language models (LLMs) become increasingly integrated into daily life, there is growing demand for AI assistants that are not only reactive but also proactive and personalized. While recent advances have pushed forward proactivity and personalization individually, their combination remains underexplored. To bridge this gap, we introduce ProPerSim, a new task and simulation framework for developing assistants capable of making timely, personalized recommendations in realistic home scenarios. In our simulation environment, a user agent with a rich persona interacts with the assistant, providing ratings on how well each suggestion aligns with its preferences and context. The assistant's goal is to use these ratings to learn and adapt to achieve higher scores over time. Built on ProPerSim, we propose ProPerAssistant, a retrieval-augmented, preference-aligned assistant that continually learns and adapts through user feedback. Experiments across 32 diverse personas show that ProPerAssistant adapts its strategy and steadily improves user satisfaction, highlighting the promise of uniting proactivity and personalization. 

---
# ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models 

**Authors**: Zihan Lin, Xiaohan Wang, Jie Cao, Jiajun Chai, Guojun Yin, Wei Lin, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2509.21826)  

**Abstract**: Large language models (LLMs) transcend passive generation and act as goal-directed agents by invoking external tools. Reinforcement learning (RL) offers a principled framework for optimizing these emergent tool-use policies, yet the prevailing paradigm relies exclusively on sparse outcome rewards and lacks consideration of the particularity of tool-use tasks, inflating policy-gradient variance and resulting in inefficient training. To better understand and address these challenges, we first establish a theoretical link between policy entropy and training stability of tool-use tasks, which reveals that structured, low-entropy tokens are primary determinants of rewards. Motivated by this insight, we propose \textbf{Res}haped \textbf{T}oken-level policy gradients (\textbf{ResT}) for tool-use tasks. ResT reshapes the policy gradient through entropy-informed token reweighting, progressively upweighting reasoning tokens as training proceeds. This entropy-aware scheme enables a smooth shift from structural correctness to semantic reasoning and stabilizes convergence in multi-turn tool-use tasks. Evaluation on BFCL and API-Bank shows that ResT achieves state-of-the-art results, outperforming prior methods by up to $8.76\%$. When fine-tuned on a 4B base LLM, ResT further surpasses GPT-4o by $4.11\%$ on single-turn tasks and $1.50\%$ on multi-turn base tasks. 

---
# EPO: Entropy-regularized Policy Optimization for LLM Agents Reinforcement Learning 

**Authors**: Xu Wujiang, Wentian Zhao, Zhenting Wang, Li Yu-Jhe, Jin Can, Jin Mingyu, Mei Kai, Wan Kun, Metaxas Dimitris  

**Link**: [PDF](https://arxiv.org/pdf/2509.22576)  

**Abstract**: Training LLM agents in multi-turn environments with sparse rewards, where completing a single task requires 30+ turns of interaction within an episode, presents a fundamental challenge for reinforcement learning. We identify a critical failure mode unique to this setting: the exploration-exploitation cascade failure. This cascade begins with early-stage policy premature convergence, where sparse feedback causes agents to commit to flawed, low-entropy strategies. Subsequently, agents enter late-stage policy collapse, where conventional entropy regularization becomes counterproductive, promoting chaotic exploration that destabilizes training. We propose Entropy-regularized Policy Optimization (EPO), a general framework that breaks this failure cycle through three synergistic mechanisms: (1) adopting entropy regularization in multi-turn settings to enhance exploration, (2) an entropy smoothing regularizer that bounds policy entropy within historical averages to prevent abrupt fluctuations, and (3) adaptive phase-based weighting that balances exploration and exploitation across training. Our analysis justifies that EPO guarantees monotonically decreasing entropy variance while maintaining convergence. EPO achieves up to 152% performance improvement on ScienceWorld and up to 19.8% on ALFWorld. Our work demonstrates that multi-turn sparse-reward settings require fundamentally different entropy control than traditional RL, with broad implications for LLM agent training. 

---
# Can Synthetic Query Rewrites Capture User Intent Better than Humans in Retrieval-Augmented Generation? 

**Authors**: JiaYing Zheng, HaiNan Zhang, Liang Pang, YongXin Tong, ZhiMing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22325)  

**Abstract**: Multi-turn RAG systems often face queries with colloquial omissions and ambiguous references, posing significant challenges for effective retrieval and generation. Traditional query rewriting relies on human annotators to clarify queries, but due to limitations in annotators' expressive ability and depth of understanding, manually rewritten queries often diverge from those needed in real-world RAG systems, resulting in a gap between user intent and system response. We observe that high-quality synthetic queries can better bridge this gap, achieving superior performance in both retrieval and generation compared to human rewrites. This raises an interesting question: Can rewriting models trained on synthetic queries better capture user intent than human annotators? In this paper, we propose SynRewrite, a synthetic data-driven query rewriting model to generate high-quality synthetic rewrites more aligned with user intent. To construct training data, we prompt GPT-4o with dialogue history, current queries, positive documents, and answers to synthesize high-quality rewrites. A Flan-T5 model is then finetuned on this dataset to map dialogue history and queries to synthetic rewrites. Finally, we further enhance the rewriter using the generator's feedback through the DPO algorithm to boost end-task performance. Experiments on TopiOCQA and QRECC datasets show that SynRewrite consistently outperforms human rewrites in both retrieval and generation tasks. Our results demonstrate that synthetic rewrites can serve as a scalable and effective alternative to human annotations. 

---
# Learning GUI Grounding with Spatial Reasoning from Visual Feedback 

**Authors**: Yu Zhao, Wei-Ning Chen, Huseyin Atahan Inan, Samuel Kessler, Lu Wang, Lukas Wutschitz, Fangkai Yang, Chaoyun Zhang, Pasquale Minervini, Saravan Rajmohan, Robert Sim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21552)  

**Abstract**: Graphical User Interface (GUI) grounding is commonly framed as a coordinate prediction task -- given a natural language instruction, generate on-screen coordinates for actions such as clicks and keystrokes. However, recent Vision Language Models (VLMs) often fail to predict accurate numeric coordinates when processing high-resolution GUI images with complex layouts. To address this issue, we reframe GUI grounding as an \emph{interactive search task}, where the VLM generates actions to move a cursor in the GUI to locate UI elements. At each step, the model determines the target object, evaluates the spatial relations between the cursor and the target, and moves the cursor closer to the target conditioned on the movement history. In this interactive process, the rendered cursor provides visual feedback to help the model align its predictions with the corresponding on-screen locations. We train our GUI grounding model, GUI-Cursor, using multi-step online reinforcement learning with a dense trajectory-based reward function. Our experimental results show that GUI-Cursor, based on Qwen2.5-VL-7B, improves the GUI grounding accuracy and achieves state-of-the-art results on ScreenSpot-v2 ($88.8\% \rightarrow 93.9\%$) and ScreenSpot-Pro ($26.8\% \rightarrow 56.5\%$). Moreover, we observe that GUI-Cursor learns to solve the problem within two steps for 95\% of instances and can adaptively conduct more steps on more difficult examples. 

---
