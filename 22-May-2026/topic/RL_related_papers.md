# Reducing Political Manipulation with Consistency Training 

**Authors**: Long Phan, Devin Kim, Alexander Pan, Alice Blair, Adam Khoja, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2605.22771)  

**Abstract**: Large language models (LLMs) exhibit systematic political bias across a variety of sensitive contexts. We find that LLMs handle counterpart topics from opposing political sides asymmetrically. We refer to this phenomenon as covert political bias and identify 7 categories of techniques through which it operates. We propose two metrics for covert bias: Sentiment Consistency measures symmetry in rhetoric and framing across paired political prompts; Helpfulness Consistency measures symmetric depth and engagement. To reduce both types of covert bias, we introduce Political Consistency Training (PCT), an RL training method with two complementary paradigms: Sentiment Consistency Training and Helpfulness Consistency Training. We show that PCT preserves overall helpfulness, substantially reduces covert political bias, and generalizes to held-out benchmarks. We release our work at this https URL 

---
# LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance 

**Authors**: Yuchun Fan, Bei Li, Peiguang Li, Yilin Wang, Yongyu Mu, Jian Yang, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22567)  

**Abstract**: Reinforcement learning has proven effective for enhancing multi-step reasoning in large language models (LLMs), yet its benefits have not fully translated to multilingual contexts. Existing methods struggle with a fundamental trade-off: prioritizing input-language consistency severely hampers reasoning quality, while prioritizing reasoning often leads to unintended language drift toward English. We address this challenge with LANG, a novel framework that leverages language-conditioned hints to guide exploration in non-English reasoning tasks. Our method incorporates two key mechanisms to prevent dependency on these hints: a progressive decay schedule that gradually withdraws scaffolding, and a language-adaptive switch that tailors learning horizons to specific language difficulties. Empirical results on challenging multilingual mathematical benchmarks reveal that LANG substantially enhances reasoning performance without compromising language consistency. Moreover, we show that our framework generalizes beyond mathematics, fostering more consistent language alignment across model layers 

---
# DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA 

**Authors**: Jianing Yin, Tan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22411)  

**Abstract**: Large language model (LLM) agents still struggle with long-term memory question answering, where answer-supporting evidence is often scattered across long conversational histories and buried in substantial irrelevant content. Existing memory systems typically process memory before future queries are known, then retrieve the resulting units based on similarity rather than their utility for answering the query. This workflow leaves downstream answerers to denoise retrieved candidates and reconstruct query-specific evidence. We present DeferMem, a long-term memory framework that decouples this problem into high-recall candidate retrieval and query-conditioned evidence distillation. DeferMem uses a lightweight segment-link structure to organize raw history and retrieve broad candidates at query time. It then applies a memory distiller trained with DistillPO, our reinforcement learning algorithm for distilling the high-recall but highly noisy candidates into a set of faithful, self-contained, and query-conditioned evidence. DistillPO formulates post-retrieval evidence distillation as a structured action comprising message selection and evidence rewriting. It optimizes this action with a decomposed-and-gated reward pipeline and structure-aligned advantage assignment, gating reward components from validity to quality checks while exposing task-level correctness feedback early and assigning each reward to its responsible output span. On LoCoMo and LongMemEval-S, DeferMem surpasses strong baselines in QA accuracy and memory-system efficiency, achieving the highest QA accuracy with the fastest runtime and zero commercial-API token cost for memory operations. 

---
# Unified Data Selection for LLM Reasoning 

**Authors**: Xiaoyuan Li, Yubo Ma, Chengpeng Li, Fengbin Zhu, Yiyao Yu, Keqin Bao, Wenjie Wang, Fuli Feng, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22389)  

**Abstract**: Effectively training Large Language Models (LLMs) for complex, long-CoT reasoning is often bottlenecked by the need for massive high-quality reasoning data. Existing methods are either computationally expensive or fail to reliably distinguish high- from low-quality reasoning samples. To address this, we propose High-Entropy Sum (HES), a training-free metric that quantifies reasoning quality by summing only the entropy of the top (e.g., 0.5\%) highest-entropy tokens in each reasoning sample. We validate HES across three mainstream training paradigms: Supervised Fine-tuning (SFT), Rejection Fine-tuning (RFT), and Reinforcement Learning (RL), with extensive results demonstrating its consistent effectiveness and significantly reduced computational overhead. In SFT, training on the top 20\% HES-ranked data matches full-dataset performance, while using the lowest-HES data degrades it. In RFT, our HES-based training approach significantly outperforms baseline methods. In RL, HES-selected successful trajectories enable the model to learn strong reasoning patterns, significantly surpassing other compared methods. Our findings establish HES as a robust, training-free metric that enables a unified, effective, and efficient method for developing advanced reasoning in LLMs. 

---
# Self-Policy Distillation via Capability-Selective Subspace Projection 

**Authors**: Guangya Hao, Yitong Shang, Yunbo Long, Zhuokai Zhao, Hanxue Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22675)  

**Abstract**: Self-distillation bootstraps large language models (LLMs) by training on their own generations. However, existing methods either rely on external signals to curate self-generated outputs (e.g., correctness filtering, execution feedback, and reward search), which are costly and unavailable for the best-performing frontier models, or skip curation entirely and train on all raw outputs, an approach that is often domain-specific and hard to generalize. Both also share a deeper weakness that self-generated outputs entangle task-relevant capability with others, such as stylistic patterns, formatting artifacts, and model-specific errors, diluting the signal for the specific capability one aims to improve. In this paper, we propose Self-Policy Distillation (SPD), which achieves generalizable, capability selective without any external signal. Specifically, SPD extracts a low-rank capability subspace from the model's own gradients on correctness-defining tokens, projects key-value (KV) activations into this subspace during self-generation, and fine-tunes on the resulting raw outputs with standard next-token prediction loss. Through extensive experiments across code generation, mathematical reasoning, and multiple-choice QA, we show that SPD achieves up to 13% improvement over state-of-the-art self-distillation methods without external signals and up to 16% improvement over pre-trained baselines. Notably, SPD demonstrates superior generalizability, achieving 15% better performance under out-of-domain generalization settings. 

---
# Faithful-MR1: Faithful Multimodal Reasoning via Anchoring and Reinforcing Visual Attention 

**Authors**: Changyuan Tian, Zhicong Lu, Huaxing Liu, Xiang Wang, Shuai Li, Yu Chen, Wenqian Lv, Zichuan Lin, Juncheng Diao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.22072)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for advancing complex reasoning in large language models, and recent work extends RLVR to multimodal large language models (MLLMs). This transfer, however, surfaces a faithfulness challenge: faithful perception of task-relevant visual evidence and faithful use of that evidence during reasoning, leading to unsatisfactory gains on multimodal benchmarks. Specifically, existing perception supervision often operates on textual descriptions rather than natively on image regions, and faithful use is largely overlooked, exposing the perception-reasoning disconnect where correctly perceived evidence is dropped or contradicted during reasoning. To close these gaps, we propose Faithful-MR1, a training framework that anchors and reinforces visual attention to address both halves of faithful multimodal reasoning. The Anchoring stage turns perception into an explicit pre-reasoning subtask, supervising a dedicated <Focus> token's attention directly against image regions rather than through textual descriptions. The Reinforcing stage exposes faithful use through counterfactual image intervention, rewarding answer-correct trajectories that concentrate visual attention where vision causally matters. Extensive experiments demonstrate that Faithful-MR1 outperforms recent multimodal reasoning baselines on both Qwen2.5-VL-Instruct 3B and 7B backbones while using substantially less training data. 

---
# Token-weighted Direct Preference Optimization with Attention 

**Authors**: Chengyu Huang, Zhuohang Li, Sheng-Yen Chou, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2605.21883)  

**Abstract**: Direct Preference Optimization (DPO) aligns Large Language Models with human preferences without the need for a separate reward model. However, DPO treats all tokens in responses equally, neglecting the differing importance of individual tokens. Existing token-level PO methods compute the token weights using either token-position-based heuristic functions or probability estimates given by a separately trained model, which lacks robustness and incurs extra training cost. In contrast, we propose Token-weighted DPO (TwDPO) -- a novel training objective grounded on token-weighted RL -- and AttentionPO -- an instantiation of TwDPO that uses attention from the LLM itself to estimate token weights. AttentionPO prompts the LLM to serve as a pairwise judge and check where the model attends when comparing the responses. This design makes AttentionPO content-aware, adjusting weights based on response content, and efficient, incurring only two extra forward passes per example. Experiment results show that AttentionPO significantly improves performance on AlpacaEval, MT-Bench, and ArenaHard, surpassing existing Preference Optimization methods. 

---
# Vector Policy Optimization: Training for Diversity Improves Test-Time Search 

**Authors**: Ryan Bahlous-Boldi, Isha Puri, Idan Shenfeld, Akarsh Kumar, Mehul Damani, Sebastian Risi, Omar Khattab, Zhang-Wei Hong, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2605.22817)  

**Abstract**: Language models must now generalize out of the box to novel environments and work inside inference-scaling search procedures, such as AlphaEvolve, that select rollouts with a variety of task-specific reward functions. Unfortunately, the standard paradigm of LLM post-training optimizes a pre-specified scalar reward, often leading current LLMs to produce low-entropy response distributions and thus to struggle at displaying the diversity that inference-time search will require. We propose Vector Policy Optimization (VPO), an RL algorithm that explicitly trains policies to anticipate diverse downstream reward functions and to produce diverse solutions. VPO exploits that rewards are often vector-valued in practice, like per-test-case correctness in code generation or, say, multiple different user personas or reward models. VPO is essentially a drop-in replacement for the GRPO advantage estimator, but it trains the LLM to output a set of solutions where individual solutions specialize to different trade-offs in the vector reward space. Across four tasks, VPO matches or beats the strongest scalar RL baselines on test-time search (e.g. pass@k and best@k), with the gap widening as the search budget grows. For evolutionary search, VPO models unlock problems that GRPO models cannot solve at all. As test-time search becomes more standardized, optimizing for diversity may need to become the default post-training objective. 

---
# Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework 

**Authors**: Shourov Joarder, Diganta Sikdar, Ahsan Habib Akash, Binod Bhattarai, Prashnna Gyawali  

**Link**: [PDF](https://arxiv.org/pdf/2605.22620)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning ability of LLMs, but often depends on external supervision from human annotations or gold-standard solutions. Reinforcement learning from internal feedback (RLIF) has recently emerged as a scalable unsupervised alternative, using signals extracted from the model itself. However, existing RLIF methods typically rely on a single internal reward, which can lead to reward hacking, entropy collapse, and degraded reasoning structure. We propose a multi-reward RLIF framework that decomposes the training signal into two complementary components: an answer-level reward based on cluster voting and a completion-level reward based on token-wise self-certainty. To combine these signals robustly, we apply GDPO-based normalization to reduce reward-scale imbalance. We further introduce KL-Cov regularization, which targets low-entropy token distributions responsible for disproportionate entropy reduction, preserving exploration and preventing late-stage collapse. Across mathematical reasoning and code-generation benchmarks, our method improves stability and robustness over prior unsupervised RL approaches, while achieving performance close to supervised RLVR methods. These results show that complementary internal rewards, combined with targeted regularization, can support stable long-horizon reasoning without relying on external ground-truth supervision. Code will be released soon. 

---
# Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning 

**Authors**: Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Xuxin Zhang, Huangyu Dai, Lingtao Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22511)  

**Abstract**: Post-training has become the dominant recipe for turning a language model into a competent search-augmented reasoning agent. A line of recent work pushes its performance further by adding elaborate machinery on top of this standard pipeline. These augmentations import external supervision from stronger external systems, attach auxiliary modules such as process reward models or retrospective critics, restructure the rollout itself with tree search or multi-stage curricula, or shape the reward with hand-crafted bonuses and penalties. Each addition delivers a measurable gain, but each also inflates the training pipeline and ties the recipe to resources or designs that may not always be available. We take a step back and ask whether any of this machinery is actually necessary, and propose Search-E1, a self-evolution method that lets a search-augmented agent improve through only vanilla GRPO interleaved with offline self-distillation (OFSD). After each GRPO round, the policy rolls out on its own training questions. A token-level forward KL objective then aligns the policy's inference-time distribution to its own distribution under a privileged context that exposes a more efficient sibling trajectory. Despite this simplicity, the procedure naturally provides dense per-step supervision. On seven QA benchmarks, Search-E1 reaches $0.440$ average EM with Qwen2.5-3B, surpassing all open-source baselines at both scales. Code and complete version will be made public soon. 

---
# Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL 

**Authors**: Sophia Xiao Pu, Zhaotian Weng, Chengzhi Liu, Jayanth Srinivasa, Gaowen Liu, William Yang Wang, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22217)  

**Abstract**: Self-play reinforcement learning trains language models on their own generated tasks, co-evolving a proposer and solver without human labels. Recent systems report strong reasoning gains, but collapse and instability are widely observed and poorly understood. The dominant response treats this as a reward-design problem. We argue instead that self-play stability is governed by two distinct levers: a data-level gate that decides which proposer-generated tasks enter the training pool, and the reward signal that updates the policy on tasks already admitted. Through controlled experiments on a Python output-prediction task and a deterministic-DSL twin task that strips pretraining priors, output ambiguity, and executor noise, we find the two levers are asymmetric. A strict gate is sufficient for stability under every reward variant we test, including a self-consistency reward with no access to ground truth; while no reward variant is sufficient once the gate is removed. This asymmetry exposes a counter-intuitive coupling we call the Grounded Proposer Paradox: a proposer with ground-truth access accelerates collapse faster than an ungrounded one when paired with a self-consistency solver, by concentrating training on clean tasks that form the fastest path to a spurious self-consistent attractor. Replacing the binary gate with a continuous strictness parameter $\varepsilon$ further reveals a two-stage phase transition: training-side metrics decouple at low $\varepsilon$, while validation accuracy holds until $\varepsilon$ is much higher. Data-level gating, not reward calibration, is the binding constraint on self-play stability. 

---
# From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning 

**Authors**: Xitai Jiang, Zihan Tang, Wenze Lin, Yang Yue, Shenzhi Wang, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22074)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has shown strong promise for LLM reasoning, but outcome-based RLVR remains inefficient on hard problems because correct final-answer rollouts are rare and sample-level credit assignment cannot use partial progress in failed attempts. We introduce SCRL (Subproblem Curriculum Reinforcement Learning), a curriculum RL framework that derives verifiable subproblems from reference reasoning chains and fixes the final subproblem as the original problem. This turns partial progress on hard problems into verifiable learning signals. Algorithmically, SCRL uses subproblem-level normalization, which normalizes rewards independently at each subproblem position and assigns the resulting advantages to the corresponding answer spans, enabling finer-grained credit assignment without external rubrics or reward models. Our analysis shows that subproblem curricula lift hard problems out of gradient dead zones, with larger relative gains as the original problem becomes harder. Across seven mathematical reasoning benchmarks, SCRL outperforms strong curriculum-learning baselines, improving average accuracy over GRPO by +4.1 points on Qwen3-4B-Base and +1.9 points on Qwen3-14B-Base. On AIME24, AIME25, and IMO-Bench, SCRL further improves pass@1 by +3.7 points and pass@64 by +4.6 points on Qwen3-4B-Base, indicating better exploration on hard reasoning problems. 

---
# Efficient Agentic Reasoning Through Self-Regulated Simulative Planning 

**Authors**: Mingkai Deng, Jinyu Hou, Lara Sá Neves, Varad Pimpalkhute, Taylor W. Killian, Zhengzhong Liu, Eric P. Xing  

**Link**: [PDF](https://arxiv.org/pdf/2605.22138)  

**Abstract**: How should an agent decide when and how to plan? A dominant approach builds agents as reactive policies with adaptive computation (e.g., chain-of-thought), trained end-to-end expecting planning to emerge implicitly. Without control over the presence, structure, or horizon of planning, these systems dramatically increase reasoning length, yielding inefficient token use without reliable accuracy gains. We argue efficient agentic reasoning benefits from decomposing decision-making into three systems: simulative reasoning (System II) grounding deliberation in future-state prediction via a world model; self-regulation (System III) deciding when and how deeply to plan via a learned configurator; and reactive execution (System I) handling fine-grained action. Simulative reasoning provides unified planning across diverse tasks without per-domain engineering, while self-regulation ensures the planner is invoked only when needed. To test this, we develop SR$^2$AM (Self-Regulated Simulative Reasoning Agentic LLM), realizing both as distinct stages within an LLM's chain-of-thought, with the LLM as world model. We explore two instantiations: recording decisions from a prompted multi-module system (v0.1) and reconstructing structured plans from traces of pretrained reasoning LLMs (v1.0), trained via supervised then reinforcement learning (RL). Across math, science, tabular analysis, and web information seeking, v0.1-8B and v1.0-30B achieve Pass@1 competitive with 120-355B and 685B-1T parameter systems respectively, while v1.0-30B uses 25.8-95.3% fewer reasoning tokens than comparable agentic LLMs. RL increases average planning horizon by 22.8% while planning frequency grows only 2.0%, showing it learns to plan further ahead rather than more often. More broadly, learned self-regulation instantiates a principle we expect to extend beyond planning to how agents govern their own learning and adaptation. 

---
# Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles 

**Authors**: Jinyang Wu, Guocheng Zhai, Ruihan Jin, Yuhao Shen, Zhengxi Lu, Fan Zhang, Haoran Luo, Zheng Lian, Zhengqi Wen, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22177)  

**Abstract**: The proliferation of large language models (LLMs) and modular skills has endowed autonomous agents with increasingly powerful capabilities. Existing frameworks typically rely on monolithic LLMs and fixed logic to interface with these skills. This gives rise to a critical bottleneck: different LLMs offer distinct advantages across diverse domains, yet current frameworks fail to exploit the complementary strengths of models and skills, thereby limiting their performance on downstream tasks. In this paper, we present Maestro (Multimodal Agent for Expert-Skill Targeted Reinforced Orchestration), a Reinforcement Learning (RL)-driven orchestration framework that reframes heterogeneous multimodal tasks as a sequential decision-making process over a hierarchical model-skill registry. Rather than consolidating all knowledge into a single model, Maestro trains a lightweight policy to dynamically compose ensembles of frozen expert models and a two-tier skill library, deciding at each step whether to invoke an external expert, which model-skill pair to select, and when to terminate. The policy is optimized via outcome-based RL, requiring no step-level supervision. We evaluate Maestro across ten representative multimodal benchmarks spanning mathematical reasoning, chart understanding, high-resolution perception, and domain-specific analysis. With only a 4B orchestrator, Maestro achieves an average accuracy of 70.1%, surpassing both GPT-5 (69.3%) and Gemini-2.5-Pro (68.7%). Crucially, the learned coordination policy generalizes to unseen models and skills without retraining: augmenting the registry with out-of-domain experts yields a 59.5% average on four challenging benchmarks, outperforming all closed-source baselines. Maestro further maintains high computational efficiency with low latency. The source code is available at this https URL. 

---
# Echo: Learning from Experience Data via User-Driven Refinement 

**Authors**: Hande Dong, Xiaoyun Liang, Jiarui Yu, Jiayi Lin, Changqing Ai, Feng Liu, Wenjun Zhang, Rongbi Wei, Chaofan Zhu, Linjie Che, Feng Wu, Xin Shen, Dexu Kong, Xiaotian Wang, Qiuyuan Chen, Bingxu An, Yueting Lei, Qiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.21984)  

**Abstract**: Static "human data" faces inherent limitations: it is expensive to scale and bounded by the knowledge of its creators. Continuous learning from "experience data" - interactions between agents and their environments - promises to transcend these barriers. Today, the widespread deployment of AI agents grants us low-cost access to massive streams of such real-world experience. However, raw interaction logs are inherently noisy, filled with trial-and-error and low information density, rendering them inefficient for direct model training.
We introduce Echo, a generalized framework designed to operationalize the transition from raw experience to learnable knowledge, effectively "echoing" environmental feedback back into the training loop for model optimization. In today's agent ecosystem, user refinement serves as a primary source of such feedback: driven by responsibility for the outcome, users rigorously transform flawed agent proposals into verified solutions. These user-driven refinement sequences inherently distill agents' crude attempts into high-quality training signals. Echo systematically harvests these signals to continuously align the agent with real-world needs. Large-scale validation in a production code completion environment confirms that Echo effectively harnesses this pipeline, breaking the static performance ceiling by increasing the acceptance rate from 25.7% to 35.7%. 

---
# Value-Gradient Hypothesis of RL for LLMs 

**Authors**: Arip Asadulaev, Daniil Ognev, Karim Salta, Martin Takac  

**Link**: [PDF](https://arxiv.org/pdf/2605.21654)  

**Abstract**: Reinforcement learning substantially improves pretrained language models, but it remains understudied why critic-free methods such as PPO and GRPO work as well as they do, and when they should provide the largest gains. We develop a value-gradient perspective of critic-free RL for LLM post-training. First, under a differentiable rollout and additive-noise parameterization, we show that the actor update is value-gradient-like in expectation: the backward pass propagates costates whose conditional expectation equals the value gradient. Second, for discrete transformer policies, we show that autodifferentiation through attention produces empirical costates that approximate this value signal, with an error controlled by the sampling gap and policy entropy. These results motivate a decomposition of RL impact into value gradient signal and reachable reward headroom, yielding a criterion for when RL should be most effective along a pretraining trajectory. 

---
# Why Semantic Entropy Fails: Geometry-Aware and Calibrated Uncertainty for Policy Optimization 

**Authors**: Zheyuan Zhang, Kaiwen Shi, Han Bao, Zehong Wang, Tianyi Ma, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.21801)  

**Abstract**: Post-training has become central to improving reasoning and alignment in large language models, where critic-free models enable scalable learning from model-generated outputs but lack principled mechanisms to distinguish informative from noisy signals. Recent approaches leverage response-level measures as uncertainty signals to regulate group-based optimization methods such as GRPO. Yet their empirical success remains unstable and unclear in how they influence optimization dynamics. In this paper, we provide, to our knowledge, the first principled formulation that interprets uncertainty signals as mechanisms for characterizing and regulating gradient variance and learning signal quality. Based on both empirical and theoretical analysis, we identify two critical gaps of current entropy-based estimators: The anisotropic gap and The calibration gap. Motivated by this analysis, we propose Geometric-aware Calibrated Policy Optimization (GCPO), a novel framework integrating geometry-aware measures to capture semantic disagreement with reward-based calibration to align uncertainty with learning signal strength. Experiments on multiple benchmarks show that our approach more faithfully tracks gradient variability and consistently improves post-training performance. Our results highlight the importance of designing uncertainty signals that are aligned with optimization dynamics, offering a principled perspective for robust post-training. 

---
# Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation 

**Authors**: Srujan P Mule, Aniketh Garikaparthi, Manasi Patwardhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21491)  

**Abstract**: As language models accelerate scientific research by automating hypothesis generation and implementation, a new bottleneck emerges: evaluating and filtering hundreds of AI-generated ideas without exhaustive experimentation. We ask whether LMs can learn to forecast the empirical success of research ideas before any experiments are run. We study comparative empirical forecasting: given a benchmark-specific research goal and two candidate ideas, predict which will achieve better benchmark performance. We construct a dataset of 11,488 idea pairs grounded in objective outcomes from PapersWithCode. While off-the-shelf 8B-parameter models struggle (30% acc.), SFT dramatically boosts performance to 77.1%, outperforming GPT-5 (61.1%). By framing evaluation as a reasoning task via Reinforcement Learning with Verifiable Rewards (RLVR), we train models to discover latent reasoning paths, achieving 71.35% acc. with interpretable justifications. Through additional ablations and out-of-distribution tests, we show robustness to surface-level heuristics and transfer to both a cross-domain time-split test set and an independently constructed test set. Our results demonstrate that compute-efficient small language models can serve as effective, objective verifiers, offering a scalable path for autonomous scientific discovery. 

---
# HealthCraft: A Reinforcement Learning Safety Environment for Emergency Medicine 

**Authors**: Brandon Dent  

**Link**: [PDF](https://arxiv.org/pdf/2605.21496)  

**Abstract**: Frontier language models are being deployed into clinical workflows faster than the infrastructure to evaluate them safely. Static medical-QA benchmarks miss the failure modes that matter in emergency medicine: trajectory-level safety collapse, tool misuse, and capitulation under sustained clinical pressure. We present HealthCraft, the first public reinforcement-learning environment that rewards trajectory-level safety under realistic emergency-medicine conditions, adapted from Corecraft. It is built on a FHIR R4 world state with 14 entity types and 3,987 seed entities, exposes 24 MCP tools, and defines a dual-layer rubric that zeroes reward whenever any safety-critical criterion is violated. We release 195 tasks across six categories, graded against 2,255 binary criteria (515 safety-critical); a post-hoc 10-task negative-class slate extends this to 205 tasks and 2,337 criteria. V8 results on two frontier models show Claude Opus 4.6 at Pass@1 24.8% [21.5-28.4] and GPT-5.4 at 12.6% [10.2-15.6], with safety-failure rates of 27.5% and 34.0%. On multi-step workflows - the closest proxy to real emergency care - performance collapses to near zero (Claude 1.0%, GPT-5.4 0.0%) despite partial competence on individual steps. Six infrastructure bugs fixed between pilots v2 and v8 re-ordered which model "looks stronger," evidence that infrastructure fidelity is part of the measurement. A deterministic LLM-judge overlay bounds evaluator noise, and a 60-run negative-class smoke pilot shows the reward signal is not drop-in training-safe: restraint criteria pass at 0.929 prevalence, a gameability an eval harness can tolerate but a training reward cannot. We scaffold coupling to a Megatron+SGLang+GRPO loop per Corecraft Section 5.2 and leave training-reward ablations as future work. Environment, tasks, rubrics, and harness are released under Apache 2.0. 

---
# Reinforced Preference Optimization for Reasoning-Augmented Recommendations 

**Authors**: Jingtong Gao, Zeyu Song, Chi Lu, Xiaopeng Li, Derong Xu, Maolin Wang, Peng Jiang, Kun Gai, Qingpeng Cai, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21967)  

**Abstract**: Recommender systems are critical for delivering personalized content across digital platforms, and recent advances in Large Language Models (LLMs) offer new opportunities to enhance them with richer world knowledge and explicit reasoning capabilities. With the help of reasoning knowledge, recommendations can better infer users' underlying intents, adapt to evolving preferences, and leverage semantic relationships for improved accuracy and interpretability. However, existing reasoning-based recommendation methods often fail to fully align the LLM's reasoning process with recommendation-specific objectives due to structural disruption during integration and difficulties in translating free-form generation into accurate item predictions. In this paper, we introduce RPORec, a reinforced preference optimization framework that unifies an LLM backbone's reasoning ability with a dedicated recommendation head (Rechead) for precise item retrieval. RPORec comprises two stages: (1) Reasoning-Augmented Recommendation Modeling, where high-quality Chain-of-Thought (CoT) reasoning is generated and used as auxiliary knowledge to guide the Rechead in learning recommendation-specific representations; and (2) Advanced Reasoning Refinement and Alignment, in which the trained Rechead produces verifiable rewards to fine-tune the LLM backbone via reinforcement learning, enhancing reasoning quality, structural consistency, and task relevance. Extensive experiments on public benchmarks and large-scale online deployments show that RPORec consistently outperforms state-of-the-art LLM-based recommendation methods, demonstrating the effectiveness of reasoning-augmented recommendation modeling in real-world systems. 

---
# Integrating Chain-of-Thought into Generative Retrieval: A Preliminary Study 

**Authors**: Wenhao Zhang, Ruihao Yu, Yi Bai, Zhumin Chen, Pengjie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.22358)  

**Abstract**: While generative retrieval (GR) demonstrates competitive performance on standard retrieval benchmarks, existing approaches directly map queries to document identifiers (docids) without intermediate deliberation, limiting their effectiveness for complex queries that require multi-step reasoning. As a preliminary study on integrating chain-of-thought (CoT) into generative retrieval, we introduce ThinkGR, a unified framework that interleaves CoT with docid generation, enabling iterative thinking and retrieval within a single generative process. To bridge the gap between free-form thought generation and structured retrieval targets, we design (1) a hybrid decoding strategy that dynamically switches between unconstrained thought generation and constrained docid decoding, and (2) a two-phase training approach that first aligns thought-retrieval patterns through supervised fine-tuning, then optimizes thought quality via retrieval-grounded reinforcement learning. Experiments on four multi-hop retrieval benchmarks demonstrate that ThinkGR achieves state-of-the-art performance with an average improvement of +6.86\%. Our work opens new avenues for enhancing generative retrieval with explicit deliberation capabilities, with promising implications for retrieval tasks requiring complex reasoning. 

---
# Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning 

**Authors**: Banghao Chi, Yining Xie, Mingyuan Wu, Jingcheng Yang, Jize Jiang, Zhaoheng Li, Shengyi Qian, Minjia Zhang, Klara Nahrstedt, Rui Hou, Xiangjun Fan, Hanchao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22642)  

**Abstract**: Spreadsheet systems (e.g., Microsoft Excel, Google Sheets) play a central role in modern data-centric workflows. As AI agents grow increasingly capable of automating complex tasks, such as controlling computers and generating presentations, building an AI-driven spreadsheet agent has emerged as a promising research direction. Most existing spreadsheet agents rely on specialized prompting over general-purpose LLMs; while this design has potentials on simple spreadsheet operations, it struggles to manage the complex, multi-step workflows typical of real-world applications.
We introduce Spreadsheet-RL, a reinforcement learning (RL) fine-tuning framework designed to train specialized spreadsheet agents within a realistic Microsoft Excel environment. Spreadsheet-RL features an automated pipeline for scalable collection of paired start-goal spreadsheets from online forums, as well as domain-specific evaluation tasks in areas such as finance and supply chain management, which we compile into the new Domain-Spreadsheet benchmark dataset. It also includes a Spreadsheet Gym environment designed for multi-turn RL: Spreadsheet Gym exposes extensive Excel functionality through a Python sandbox, along with a refined harness that incorporates a comprehensive tool set and carefully designed tool-routing rules for spreadsheet tasks. Through comprehensive experiments, we show that Spreadsheet-RL substantially enhances AI agent's performance on both general and domain-specific spreadsheet tasks: it improves Qwen3-4B-Thinking-2507's Pass@1 on SpreadsheetBench from 12.0% to 23.4%, and raises Pass@1 from 8.4% to 17.2% on our curated Domain-Spreadsheet dataset. These results highlight Spreadsheet-RL's strong potential for generalization and real-world adoption in spreadsheet automation, and broadly, its promise for advancing LLM-based interactions with data interfaces in everyday work. 

---
# CLORE: Content-Level Optimization for Reasoning Efficiency 

**Authors**: Yuyang Wu, Qiyao Xue, Guanxing Lu, Weichen Liu, Zihan Wang, Manling Li, Olexandr Isayev  

**Link**: [PDF](https://arxiv.org/pdf/2605.22211)  

**Abstract**: Reinforcement learning post-training has improved the reasoning ability of large language models, but often produces unnecessarily long, repetitive, or semantically opaque reasoning traces. Existing efficient reasoning methods mainly regulate response length through explicit budgets or length-aware rewards, leaving intermediate reasoning content weakly supervised. We propose CLORE, a content-level optimization framework that improves reasoning efficiency by editing correct on-policy rollouts. CLORE uses an external augmentation model to delete repetitive segments, illegible or task-irrelevant content, and superfluous reasoning after the solution is established, while preserving the final answer. The resulting augmented--original pairs are optimized with an auxiliary reference-free DPO objective alongside standard policy-gradient training. By restricting augmentation to correct trajectories and performing local deletion, CLORE keeps edited rollouts close to the policy distribution and mitigates off-policy mismatch. Experiments on DeepSeek-R1-Distill-Qwen-7B and Qwen2.5-Math-7B across five mathematical reasoning benchmarks show that CLORE improves the accuracy--efficiency trade-off and remains compatible with GRPO, DAPO, Training Efficient, and ThinkPrune. Content-level analyses further show that CLORE reduces repetitive reasoning, illegible content, and post-answer exploration, supporting content-level supervision as a complementary direction to length-level control. 

---
# Unlocking Proactivity in Task-Oriented Dialogue 

**Authors**: Hongbin Zhang, Ning Gao, Yuqin Dai, Ruiyuan Wu, Jinpeng Wang, Rena Wei Gao, Bingdong Tan, Shuzheng Gao, Zongjie Li, Chaozheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22240)  

**Abstract**: Proactive task-oriented dialogue (TOD), such as outbound sales, demands a persuasive agent that actively probes the user's concerns and steers the conversation toward acceptance within a bounded number of turns. Yet post-trained LLMs are inherently conservative, and reward-shaping RL (e.g., GRPO) struggles since it only re-weights what an already passive policy samples. We show that conditioning on the user's latent concerns unlocks proactive capability that no amount of sampling can undermine, establishing these concerns as a pivotal training-time signal. To operationalize this finding, we build the \textbf{Cognitive User Simulator}, which models each user as a stratified persona comprising observable external traits and hidden internal concerns. The simulator produces faithful and diverse interactions, while emitting per-turn state dynamics that track persuasion progress. We then introduce \textbf{Simulator-Induced Asymmetric-View Policy Optimization}, which converts the modeled concerns and the simulation state transition into complementary training objectives: (1) \emph{Asymmetric On-Policy Self-Distillation} that transfers concern-aware behavior from a privileged view of the same policy into its deployable, conversation-only view; and (2) \emph{State-Transition Policy Refinement} ... 

---
# Implicit Safety Alignment from Crowd Preferences 

**Authors**: Qian Lin, Daniel S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2605.21822)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) can reveal implicit objectives such as safety considerations that go beyond task completion. In this work, we focus on the common safety criteria embedded in crowd preference datasets, where different users may express distinct preferences or objectives, yet follow similar safety principles. Our aim is to discover shared safety criteria from crowd preferences and then transfer them to downstream RL tasks to regularize agent behavior and enforce safety. We first show that direct reward combination-optimizing a preference-learned reward model together with downstream task rewards-has inherent limitations. Motivated by this, we propose Safe Crowd Preference-based RL, a hierarchical framework that extracts safety-aligned skills from crowd preferences and composes them via a high-level policy to safely solve downstream tasks. Experiments across safe RL environments and a preliminary LLM-style task with diverse user goals and shared safety constraints demonstrate that our approach substantially lowers safety costs without access to explicit safety rewards, while achieving task performance comparable to oracle methods trained with ground-truth safety signals. 

---
# The Matching Principle: A Geometric Theory of Loss Functions for Nuisance-Robust Representation Learning 

**Authors**: Vishal Rajput  

**Link**: [PDF](https://arxiv.org/pdf/2605.22800)  

**Abstract**: Robustness, domain adaptation, photometric and occlusion invariance, compositional generalisation, temporal robustness, alignment safety, and classical anisotropic regularisation are usually treated as separate problems with separate method families. This paper argues that much of their shared structure is one statistical problem: estimate the covariance of label-preserving deployment nuisance, then regularise the encoder Jacobian along a matrix whose range covers that covariance (the matching principle). CORAL, adversarial training, IRM, augmentation, metric learning, Jacobian penalties, and alignment-style constraints are different estimators of that object, not independent robustness tricks.
In the linear-Gaussian model we prove closed-form optimality (Theorem A), including cube-root water-filling within the matched range; necessity of range coverage for quadratic Jacobian penalties (Theorem G); the same range dichotomy at deep global minima; and two falsification controls (Lemma C; Corollaries E), with seven conditional consistency lemmas (D1-D7) for estimation under standard identifiability assumptions.
We introduce the Trajectory Deviation Index (TDI), a label-free probe of embedding sensitivity when task accuracy or Jacobian Frobenius norm is insufficient.
Thirteen pre-registered blocks from classical ML through Qwen2.5-7B test the predicted matched, then isotropic, then wrong-W ordering on geometry and deployment drift; twelve pass, and the sole exception (Office-31) is an eigengap failure named before the run. At 7B scale, matched style-PMH improves selective honesty and preserves Style TDI where standard DPO degrades it.
The contribution is naming the deployment nuisance covariance, stating what the regulariser must do, and supplying a closed-form falsifiable theory once that object is identified, not universality on every leaderboard. 

---
# Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation 

**Authors**: Dong Nie  

**Link**: [PDF](https://arxiv.org/pdf/2605.22731)  

**Abstract**: Large language model post-training methods such as supervised fine-tuning (SFT), reinforcement learning (RL), and distillation are often analyzed through their loss functions: maximum likelihood, policy gradients, forward KL, reverse KL, or related objective-level variants. We study a complementary factor: the state distribution on which supervision is applied. For an autoregressive policy, a state is a prompt plus generated prefix. SFT trains on fixed dataset states, while RL and on-policy distillation (OPD) train on states induced by the current learner. We formalize post-training as state-distribution shaping and run a controlled smallscale study using Qwen3-0.6B-Base on GSM8K, with TruthfulQA and MMLU as retention evaluations. Our results show three phenomena. First, a mild SFT run improves GSM8K with little forgetting, while a stress SFT run causes substantial retention loss. Second, OPD from a degraded SFT teacher surpasses that teacher on GSM8K, TruthfulQA, and MMLU, despite using the teacher as its only supervision source. Third, a lightweight on-policy RL run improves GSM8K while preserving retention. These results support a state-centric view of post-training: the source and locality of training states can be as important as the form of the supervision signal. 

---
# One-Way Policy Optimization for Self-Evolving LLMs 

**Authors**: Shuo Yang, Jinda Lu, Kexin Huang, Chiyu Ma, Shaohang Wei, Yuyang Liu, Guoyin Wang, Jingren Zhou, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.22156)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a promising paradigm for scaling reasoning capabilities of Large Language Models (LLMs). However, the sparsity of binary verifier rewards often leads to low efficiency and optimization instability. To stabilize training, existing methods typically impose token-level constraints relative to a reference policy. We identify that such constraints penalize deviations indiscriminately; this can flip verifier-determined direction when the policy attempts to outperform the reference, thereby suppressing gains. To resolve this, we propose One-Way Policy Optimization (OWPO), a method based on the principle of decoupling optimization direction from update magnitude. In OWPO, the verifier dictates the update direction, while the reference policy serves only to adjust the magnitude. Specifically, OWPO applies asymmetric reweighting: it performs Accelerated Alignment for inferior deviations (where the policy lags behind the reference) and Gain Locking for superior deviations (where the policy surpasses the reference). Furthermore, by incorporating iterative reference updates, OWPO creates a ``Ratchet Effect'' that continuously consolidates gains. Experimental results demonstrate that OWPO outperforms strong baselines, including DAPO, OPD, and MOPD, breaking the bottleneck of fixed priors to enable continuous self-evolution without reliance on external reference models. 

---
# Learning Spatiotemporal Sensitivity in Video LLMs via Counterfactual Reinforcement Learning 

**Authors**: Dazhao Du, Jian Liu, Jialong Qin, Tao Han, Bohai Gu, Fangqi Zhu, Yujia Zhang, Eric Liu, Xi Chen, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.21988)  

**Abstract**: Video large language models (Video LLMs) achieve strong benchmark accuracy, yet often answer video questions through shortcuts such as single-frame cues and language priors rather than by tracking spatiotemporal dynamics. This issue is exacerbated in RL post-training, where correctness-only rewards can further reinforce shortcut policies that obtain high reward without tracking video dynamics. We address this by asking a controlled counterfactual question: if the visual world changed while the question remained fixed, should the answer change or stay the same? Based on this view, we propose \textbf{Counterfactual Relational Policy Optimization (CRPO)}, a dual-branch RL framework for improving \emph{spatiotemporal sensitivity}. CRPO constructs counterfactual videos through horizontal flips and temporal reversals, trains on both original and counterfactual branches, and introduces a \textbf{Counterfactual Relation Reward (CRR)} between their answers. CRR encourages answers to change for dynamic questions and remain unchanged for static questions. This cross-branch constraint makes it difficult for shortcut policies to be consistently rewarded across both branches. To evaluate this property, we introduce \textbf{DyBench}, a paired counterfactual video benchmark with 3,014 videos covering reversible dynamics, moving direction, and event sequence, together with a strict pair-accuracy metric that prevents fixed-answer shortcuts from inflating scores. Experiments show that CRPO outperforms prior RL methods on spatiotemporal-sensitive evaluations while maintaining competitive general video performance. On Qwen3-VL-8B, CRPO improves DyBench P-Acc by +7.7 and TimeBlind I-Acc by +8.2 over the base model, indicating improved spatiotemporal sensitivity rather than stronger reliance on static shortcuts. The project website can be found at this https URL . 

---
# CrossVLA: Cross-Paradigm Post-Training and Inference Optimization for Vision-Language-Action Models 

**Authors**: Zhi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21854)  

**Abstract**: Vision-Language-Action (VLA) models have rapidly converged on a small set of architectural patterns: discrete-token autoregression (e.g. OpenVLA) and continuous-action flow-matching (e.g. pi-0.5). Yet preference alignment via Direct Preference Optimisation (DPO) -- the de-facto post-training step in language models -- has been studied almost exclusively on autoregressive VLAs. We present CrossVLA, an empirical study of cross-paradigm VLA post-training. Three contributions: (i) a surrogate flow-matching log-probability estimator that lets DPO operate on continuous-action backbones without probability-flow ODE integration; (ii) a head-to-head comparison of LoRA and DoRA as the parameter-efficient layer for VLA DPO, finding DoRA improves over OpenVLA SFT by a mean +10.4 pp across LIBERO 4-suite (600 trials, 3 seeds) -- per-suite +20.0 Object, +11.0 Long-horizon, +8.0 Goal, +2.7 Spatial -- with zero seed variance on Object (38/50 on each of 3 seeds); (iii) an inference-time anatomy showing the denoise loop dominates 78.6% of sample_actions latency and prefix-K/V caching a la VLA-Cache caps at a 21% acceleration ceiling -- both chunk-level and token-level cache strategies degrade success rate to 0-80% in our benchmarks. We further pretrain a multi-view + temporal projection head on 6000 LIBERO frames, achieving 99.5% k-NN recall@1 for same-task retrieval (36x over random), available as a downstream initialisation. All code, ckpts, training logs, and reproduction scripts are open at this https URL. 

---
# MAVEN: A Multi-stage Agentic Annotation Pipeline for Video Reasoning Tasks 

**Authors**: Han Zhang, Wanting Jiang, Tomasz Kornuta, Tian Zheng, Vidya Murali  

**Link**: [PDF](https://arxiv.org/pdf/2605.21917)  

**Abstract**: Training Vision Language Models (VLMs) for video event reasoning requires high-quality structured annotations capturing not only what happened, but when, where, why, and with what consequence, at a scale manual labelling cannot support. We present MAVEN (Multi-stage Agentic Video Event aNnotation), a multi-stage agentic pipeline that turns raw videos into multi-task training data with Chain-of-Thought (CoT) reasoning traces, organized around a designated Event of Focus. At its core, MAVEN synthesizes a Multi-Scale Spatio-Temporal Event Description (MSTED) from three complementary caption levels; this explicit intermediate serves as the sole input to downstream Q&A generation across multiple task formats. Crucially, MAVEN supports agent-driven domain adaptation: given a new video dataset and target question examples, the agent redesigns all prompts top-down without manual re-engineering. A hierarchical refinement loop further classifies annotation errors against a taxonomy, traces root causes to the originating pipeline stage, and applies targeted edits that rewrite prompts or modify the pipeline structure itself, iteratively improving data quality. We apply MAVEN to label over 5,300 traffic videos and fine-tune Cosmos-Reason2-8B on the resulting data. On a private CCTV evaluation set, fine-tuning surpasses both Gemini 2.5 Pro and 3.1 Flash, including a $+38.8$-point gain in MCQ accuracy over zero-shot. On AccidentBench, CCTV-only training lifts Cosmos-Reason2 by $+10.7$ MCQ points and matches Gemini 2.5 Pro despite seeing no dashcam videos; adding agent-adapted dashcam annotations narrows the gap to Gemini 3.1 Flash, and RL post-training pushes overall performance past both Gemini baselines. Qualitative results on warehouse surveillance and public safety videos further show the agentic workflow readily adapts the pipeline to new domains. 

---
# When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning 

**Authors**: Xiaogeng Liu, Xinyan Wang, Yingzi Ma, Yechao Zhang, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21606)  

**Abstract**: On-policy self-distillation (OPSD) trains a student on its own rollouts using a privileged teacher, but its standard objective weights all generated tokens equally, implicitly treating the privileged teacher target as equally reliable at every student-visited prefix. Existing entropy-based OPD methods relax this uniformity by modulating token-level supervision with teacher entropy, but high teacher entropy in reasoning has an ambiguous reliability meaning: it can reflect either non-viable uncertainty or benign solution diversity. To identify this phenomenon, we introduce a branch-viability diagnostic. Specifically, we record next-token alternatives from the privileged-answer teacher prompt, force each alternative after the student prompt plus its on-policy spine prefix, and test whether the resulting student-template continuation recovers the correct answer. On Qwen3-4B, we find that an oriented within-sequence position score is the strongest tested predictor of teacher-token reliability, reaching an area-under-ROC-curve (AUROC) of 0.83; local uncertainty scores are at most 0.57. Motivated by this trajectory-level structure, we propose Position-Weighted On-Policy Self-Distillation (PW-OPSD), which applies an increasing position weight while keeping the same student rollout, privileged teacher pass, and clipped forward-KL target as OPSD. In our comprehensive evaluations with different random seeds, the diagnostic-derived PW-OPSD improves AIME 2024 and AIME 2025 Avg@12 by +1.0 and +1.1 points, and a generalization evaluation on two larger-scale models from different families, DeepSeek-R1-Distill-Llama-8B and Olmo-3-7B-Think, also demonstrates consistent aggregate Avg@12 improvements. These results show that teacher-token reliability in reasoning distillation is trajectory-structured and can be utilized without additional teacher computation. 

---
