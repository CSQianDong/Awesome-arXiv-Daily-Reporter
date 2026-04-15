# QuarkMedSearch: A Long-Horizon Deep Search Agent for Exploring Medical Intelligence 

**Authors**: Zhichao Lin, Zhichao Liang, Gaoqiang Liu, Meng Xu, Baoyu Xiang, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12867)  

**Abstract**: As agentic foundation models continue to evolve, how to further improve their performance in vertical domains has become an important challenge. To this end, building upon Tongyi DeepResearch, a powerful agentic foundation model, we focus on the Chinese medical deep search scenario and propose QuarkMedSearch, systematically exploring a full-pipeline approach spanning medical multi-hop data construction, training strategies, and evaluation benchmarks to further push and assess its performance upper bound in vertical domains. Specifically, for data synthesis, to address the scarcity of deep search training data in the medical domain, we combine a large-scale medical knowledge graph with real-time online exploration to construct long-horizon medical deep search training data; for post-training, we adopt a two-stage SFT and RL training strategy that progressively enhances the model's planning, tool invocation, and reflection capabilities required for deep search, while maintaining search efficiency; for evaluation, we collaborate with medical experts to construct the QuarkMedSearch Benchmark through rigorous manual verification. Experimental results demonstrate that QuarkMedSearch achieves state-of-the-art performance among open-source models of comparable scale on the QuarkMedSearch Benchmark, while also maintaining strong competitiveness on general benchmarks. 

---
# DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding 

**Authors**: Hao Yan, Yuliang Liu, Xingchen Liu, Yuyi Zhang, Minghui Liao, Jihao Wu, Wei Chen, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2604.12812)  

**Abstract**: Existing Multimodal Large Language Models (MLLMs) suffer from significant performance degradation on the long document understanding task as document length increases. This stems from two fundamental challenges: 1) a low Signal-to-Noise Ratio (SNR), with crucial evidence buried in irrelevant pages; and 2) supervision scarcity, as datasets offering only final short answers provide a weak learning signal. In this paper, we address these challenges by proposing a paradigm that requires the model to execute a structured ``\textbf{Analysis}, \textbf{Localization} and \textbf{Reasoning}'' workflow. To instill this capability, we design a two-stage training framework: we first perform Supervised Fine-Tuning on high-quality data generated via an efficient knowledge distillation strategy. Subsequently, we employ an Evidence-aware Group Relative Policy Optimization which jointly optimizes for both evidence localization and answer accuracy. Additionally, we introduce a Evidence-Guided Resolution Allocation strategy to mitigate memory constraints of training on multi-pages documents. Extensive experiments demonstrate that DocSeeker achieves superior performance on both in-domain and out-of-domain tasks. We show it robustly generalizes from short-page training to ultra-long documents and is naturally synergistic with visual Retrieval-Augmented Generation systems, serving as a solid foundation for their implementation. 

---
# KnowRL: Boosting LLM Reasoning via Reinforcement Learning with Minimal-Sufficient Knowledge Guidance 

**Authors**: Linhao Yu, Tianmeng Yang, Siyu Ding, Renren Jin, Naibin Gu, Xiangzhao Hao, Shuaiyi Nie, Deyi Xiong, Weichong Yin, Yu Sun, Hua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12627)  

**Abstract**: RLVR improves reasoning in large language models, but its effectiveness is often limited by severe reward sparsity on hard problems. Recent hint-based RL methods mitigate sparsity by injecting partial solutions or abstract templates, yet they typically scale guidance by adding more tokens, which introduce redundancy, inconsistency, and extra training overhead. We propose \textbf{KnowRL} (Knowledge-Guided Reinforcement Learning), an RL training framework that treats hint design as a minimal-sufficient guidance problem. During RL training, KnowRL decomposes guidance into atomic knowledge points (KPs) and uses Constrained Subset Search (CSS) to construct compact, interaction-aware subsets for training. We further identify a pruning interaction paradox -- removing one KP may help while removing multiple such KPs can hurt -- and explicitly optimize for robust subset curation under this dependency structure. We train KnowRL-Nemotron-1.5B from OpenMath-Nemotron-1.5B. Across eight reasoning benchmarks at the 1.5B scale, KnowRL-Nemotron-1.5B consistently outperforms strong RL and hinting baselines. Without KP hints at inference, KnowRL-Nemotron-1.5B reaches 70.08 average accuracy, already surpassing Nemotron-1.5B by +9.63 points; with selected KPs, performance improves to 74.16, establishing a new state of the art at this scale. The model, curated training data, and code are publicly available at this https URL. 

---
# Visual Preference Optimization with Rubric Rewards 

**Authors**: Ya-Qi Yu, Fangyu Hong, Xiangyang Qu, Hao Wang, Gaojie Wu, Qiaoyu Luo, Nuo Xu, Huixin Wang, Wuheng Xu, Yongxin Liao, Zihao Chen, Haonan Li, Ziming Li, Dezhi Peng, Minghui Liao, Jihao Wu, Haoyu Ren, Dandan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13029)  

**Abstract**: The effectiveness of Direct Preference Optimization (DPO) depends on preference data that reflect the quality differences that matter in multimodal tasks. Existing pipelines often rely on off-policy perturbations or coarse outcome-based signals, which are not well suited to fine-grained visual reasoning. We propose rDPO, a preference optimization framework based on instance-specific rubrics. For each image-instruction pair, we create a checklist-style rubric of essential and additional criteria to score responses from any possible policies. The instruction-rubric pool is built offline and reused during the construction of on-policy data. On public reward modeling benchmarks, rubric-based prompting massively improves a 30B-A3B judge and brings it close to GPT-5.4. On public downstream benchmarks, rubric-based filtering raises the macro average to 82.69, whereas outcome-based filtering drops it to 75.82 from 81.14. When evaluating scalability on a comprehensive benchmark, rDPO achieves 61.01, markedly outperforming the style-constrained baseline (52.36) and surpassing the 59.48 base model. Together, these results show that visual preference optimization benefits from combining on-policy data construction with instance-specific criterion-level feedback. 

---
# GoodPoint: Learning Constructive Scientific Paper Feedback from Author Responses 

**Authors**: Jimin Mun, Chani Jung, Xuhui Zhou, Hyunwoo Kim, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2604.11924)  

**Abstract**: While LLMs hold significant potential to transform scientific research, we advocate for their use to augment and empower researchers rather than to automate research without human oversight. To this end, we study constructive feedback generation, the task of producing targeted, actionable feedback that helps authors improve both their research and its presentation. In this work, we operationalize the effectiveness of feedback along two author-centric axes-validity and author action. We first curate GoodPoint-ICLR, a dataset of 19K ICLR papers with reviewer feedback annotated along both dimensions using author responses. Building on this, we introduce GoodPoint, a training recipe that leverages success signals from author responses through fine-tuning on valid and actionable feedback, together with preference optimization on both real and synthetic preference pairs. Our evaluation on a benchmark of 1.2K ICLR papers shows that a GoodPoint-trained Qwen3-8B improves the predicted success rate by 83.7% over the base model and sets a new state-of-the-art among LLMs of similar size in feedback matching on a golden human feedback set, even surpassing Gemini-3-flash in precision. We further validate these findings through an expert human study, demonstrating that GoodPoint consistently delivers higher practical value as perceived by authors. 

---
# Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe 

**Authors**: Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.13016)  

**Abstract**: On-policy distillation (OPD) has become a core technique in the post-training of large language models, yet its training dynamics remain poorly understood. This paper provides a systematic investigation of OPD dynamics and mechanisms. We first identify that two conditions govern whether OPD succeeds or fails: (i) the student and teacher should share compatible thinking patterns; and (ii) even with consistent thinking patterns and higher scores, the teacher must offer genuinely new capabilities beyond what the student has seen during training. We validate these findings through weak-to-strong reverse distillation, showing that same-family 1.5B and 7B teachers are distributionally indistinguishable from the student's perspective. Probing into the token-level mechanism, we show that successful OPD is characterized by progressive alignment on high-probability tokens at student-visited states, a small shared token set that concentrates most of the probability mass (97%-99%). We further propose two practical strategies to recover failing OPD: off-policy cold start and teacher-aligned prompt selection. Finally, we show that OPD's apparent free lunch of dense token-level reward comes at a cost, raising the question of whether OPD can scale to long-horizon distillation. 

---
# Calibration-Aware Policy Optimization for Reasoning LLMs 

**Authors**: Ziqi Wang, Xingzhou Lou, Meiqi Wu, Zhengqi Wen, Junge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12632)  

**Abstract**: Group Relative Policy Optimization (GRPO) enhances LLM reasoning but often induces overconfidence, where incorrect responses yield lower perplexity than correct ones, degrading relative calibration as described by the Area Under the Curve (AUC). Existing approaches either yield limited improvements in calibration or sacrifice gains in reasoning accuracy. We first prove that this degradation in GRPO-style algorithms stems from their uncertainty-agnostic advantage estimation, which inevitably misaligns optimization gradients with calibration. This leads to improved accuracy at the expense of degraded calibration. We then propose Calibration-Aware Policy Optimization (CAPO). It adopts a logistic AUC surrogate loss that is theoretically consistent and admits regret bound, enabling uncertainty-aware advantage estimation. By further incorporating a noise masking mechanism, CAPO achieves stable learning dynamics that jointly optimize calibration and accuracy. Experiments on multiple mathematical reasoning benchmarks show that CAPO-1.5B significantly improves calibration by up to 15% while achieving accuracy comparable to or better than GRPO, and further boosts accuracy on downstream inference-time scaling tasks by up to 5%. Moreover, when allowed to abstain under low-confidence conditions, CAPO achieves a Pareto-optimal precision-coverage trade-off, highlighting its practical value for hallucination mitigation. 

---
# KG-Reasoner: A Reinforced Model for End-to-End Multi-Hop Knowledge Graph Reasoning 

**Authors**: Shuai Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12487)  

**Abstract**: Large Language Models (LLMs) exhibit strong abilities in natural language understanding and generation, yet they struggle with knowledge-intensive reasoning. Structured Knowledge Graphs (KGs) provide an effective form of external knowledge representation and have been widely used to enhance performance in classical Knowledge Base Question Answering (KBQA) tasks. However, performing precise multi-hop reasoning over KGs for complex queries remains highly challenging. Most existing approaches decompose the reasoning process into a sequence of isolated steps executed through a fixed pipeline. While effective to some extent, such designs constrain reasoning flexibility and fragment the overall decision process, often leading to incoherence and the loss of critical intermediate information from earlier steps. In this paper, we introduce KG-Reasoner, an end-to-end framework that integrates multi-step reasoning into a unified "thinking" phase of a Reasoning LLM. Through Reinforcement Learning (RL), the LLM is trained to internalize the KG traversal process, enabling it to dynamically explore reasoning paths, and perform backtracking when necessary. Experiments on eight multi-hop and knowledge-intensive reasoning benchmarks demonstrate that KG-Reasoner achieves competitive or superior performance compared to the state-of-the-art methods. Codes are available at the repository: this https URL. 

---
# Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning 

**Authors**: NVIDIA, Aakshita Chandiramani, Aaron Blakeman, Abdullahi Olaoye, Abhibha Gupta, Abhilash Somasamudramath, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal, Aditya Vavre, Ahmad Kiswani, Aishwarya Padmakumar, Ajay Hotchandani, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Gronskiy, Alex Kondratenko, Alex Neefus, Alex Steiner, Alex Yang, Alexander Bukharin, Alexander Young, Ali Hatamizadeh, Ali Taghibakhshi, Alina Galiautdinova, Alisa Liu, Alok Kumar, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Anahita Bhiwandiwalla, Ananth Subramaniam, Andrew Tao, Anjaney Shrivastava, Anjulie Agrusa, Ankur Srivastava, Ankur Verma, Ann Guan, Anna Shors, Annamalai Chockalingam, Anubhav Mandarwal, Aparnaa Ramani, Arham Mehta, Arti Jain, Arun Venkatesan, Asha Anoosheh, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asit Mishra, Asli Sabanci Demiroz, Asma Kuriparambil Thekkumpate, Atefeh Sohrabizadeh, Avinash Kaur, Ayush Dattagupta, Barath Subramaniam Anandan, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Benjamin Chislett, Besmira Nushi, Bilal Kartal, Bill Thiede, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Buvaneswari Mani, Carlo del Mundo, Chankyu Lee, Chanran Kim, Chantal Hwang, Chao Ni, Charles Wang, Charlie Truong, Cheng-Ping Hsieh, Chenhan Yu, Chenjie Luo, Cherie Wang, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Chris Holguin, Chris Wing, Christian Munley, Christopher Parisien, Chuck Desai, Chunyang Sheng, Collin Neale, Cyril Meurillon, Dakshi Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.12374)  

**Abstract**: We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120 billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model. Nemotron 3 Super is the first model in the Nemotron 3 family to 1) be pre-trained in NVFP4, 2) leverage LatentMoE, a new Mixture-of-Experts architecture that optimizes for both accuracy per FLOP and accuracy per parameter, and 3) include MTP layers for inference acceleration through native speculative decoding. We pre-trained Nemotron 3 Super on 25 trillion tokens followed by post-training using supervised fine tuning (SFT) and reinforcement learning (RL). The final model supports up to 1M context length and achieves comparable accuracy on common benchmarks, while also achieving up to 2.2x and 7.5x higher inference throughput compared to GPT-OSS-120B and Qwen3.5-122B, respectively. Nemotron 3 Super datasets, along with the base, post-trained, and quantized checkpoints, are open-sourced on HuggingFace. 

---
# Disposition Distillation at Small Scale: A Three-Arc Negative Result 

**Authors**: Hari Sadasivan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11867)  

**Abstract**: We set out to train behavioral dispositions (self-verification, uncertainty acknowledgment, feedback integration) into small language models (0.6B to 2.3B effective parameters) through a four-stage all-MIT distillation pipeline, with follow-on experiments on inference-time attention-head interventions and a frozen-base confidence-gated sidecar. An internal draft reported +33.9-point MCAS and +15.3-point HumanEval gains on a Qwen3-0.6B student; a second-pass sanity check falsified both numbers before publication. The HumanEval delta was a truncation artifact (n_predict=512) that inverted to -8.0 points at n_predict=1024; the MCAS gain disappeared under apples-to-apples scoring. That falsification triggered three subsequent arcs. Across (1) SFT/DPO LoRA on three model families and two domains, (2) inference-time attention-head tempering on o_proj, and (3) a training-free frozen-base sidecar reading the final-token hidden state h_last, we find no operator that moves judge-measured disposition without damaging content or collapsing into stylistic mimicry. The failure is consistent across five models (Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-0.8B, Gemma 4 E2B, and SmolLM2-1.7B-Instruct). A within-distribution cross-validation pass (AUC=0.683) collapsed to chance on fresh prompts (AUC=0.516). We contribute a three-arc negative result with mechanism, a two-failure-mode taxonomy for linear h_last probes, and an honest falsification pipeline that converts the class of false positives we ourselves produced into publishable negatives. As an independent finding, Gemma 4 E2B exhibits near-complete confidence-correctness decoupling on the Chef domain (assertion asymmetry -0.009; the model asserts at 91% regardless of correctness). 

---
# Toward Autonomous Long-Horizon Engineering for ML Research 

**Authors**: Guoxin Chen, Jie Chen, Lei Chen, Jiale Zhao, Fanzhe Meng, Wayne Xin Zhao, Ruihua Song, Cheng Chen, Ji-Rong Wen, Kai Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.13018)  

**Abstract**: Autonomous AI research has advanced rapidly, but long-horizon ML research engineering remains difficult: agents must sustain coherent progress across task comprehension, environment setup, implementation, experimentation, and debugging over hours or days. We introduce AiScientist, a system for autonomous long-horizon engineering for ML research built on a simple principle: strong long-horizon performance requires both structured orchestration and durable state continuity. To this end, AiScientist combines hierarchical orchestration with a permission-scoped File-as-Bus workspace: a top-level Orchestrator maintains stage-level control through concise summaries and a workspace map, while specialized agents repeatedly re-ground on durable artifacts such as analyses, plans, code, and experimental evidence rather than relying primarily on conversational handoffs, yielding thin control over thick state. Across two complementary benchmarks, AiScientist improves PaperBench score by 10.54 points on average over the best matched baseline and achieves 81.82 Any Medal% on MLE-Bench Lite. Ablation studies further show that File-as-Bus protocol is a key driver of performance, reducing PaperBench by 6.41 points and MLE-Bench Lite by 31.82 points when removed. These results suggest that long-horizon ML research engineering is a systems problem of coordinating specialized work over durable project state, rather than a purely local reasoning problem. 

---
# Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Sequence-Level Likelihood 

**Authors**: Xingyu Lin, Yilin Wen, Du Su, Jinchang Hou, En Wang, Wenbin Liu, Chenfu Bao, Zhonghou Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.12736)  

**Abstract**: Group Relative Policy Optimization (GRPO) has significantly advanced the reasoning ability of large language models (LLMs), particularly in their mathemat ical reasoning performance. However, GRPO and related entropy regularization methods still struggle with token-level sparse-rewards, which is an inherent chal lenge in chain-of-thought (CoT) reasoning. These approaches often rely on undifferen tiated token-level entropy regularization, which easily leads to entropy collapse or model degradation under sparse token rewards. In this work, we propose TEPO, a novel token-level framework that (1) leverages sequence-level likelihood to link group-level rewards with individual tokens via token-level aggregation, and (2) introduces a token-level KL-Divergence mask constraint that targets tokens with positive advantages and decreasing entropy to mitigate abrupt policy updates. Experiments demonstrate that TEPO not only achieves state-of-the-art performance on mathematical reasoning benchmarks but also markedly enhances training stability, reducing convergence time by 50% compared with GRPO/DAPO. 

---
# Meet Dynamic Individual Preferences: Resolving Conflicting Human Value with Paired Fine-Tuning 

**Authors**: Shanyong Wang, Shuhang Lin, Yining Zhao, Xi Zhu, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12479)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved the alignment of models with general human preferences. However, a major challenge remains in adapting LLMs to individual preferences, which are not only diverse but also dynamic. In this paper, we introduce a novel framework, Preference-Paired Fine-Tuning (PFT), designed to align models with contradictory and evolving individual preferences. We present a new dataset, Value Conflict Dilemma (VCD), which includes scenarios that involve conflicting human preferences, facilitating the evaluation of our approach. Our experiments demonstrate that PFT outperforms single-preference training methods, achieving up to 96.6% accuracy in multi-choice classification tasks and the highest open-ended generation score of 8.69. PFT also shows significant improvements over DPO, SFT and some traditional training methods, especially when handling conflicting preferences. Additionally, with limited user history data, models can inferring preference vector rapidly, achieving a 44.76% improvement in user-specific preference alignment in comparison to single-preference models. 

---
# Teaching LLMs Human-Like Editing of Inappropriate Argumentation via Reinforcement Learning 

**Authors**: Timon Ziegenbein, Maja Stahl, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2604.12770)  

**Abstract**: Editing human-written text has become a standard use case of large language models (LLMs), for example, to make one's arguments more appropriate for a discussion. Comparing human to LLM-generated edits, however, we observe a mismatch in editing strategies: While LLMs often perform multiple scattered edits and tend to change meaning notably, humans rather encapsulate dependent changes in self-contained, meaning-preserving edits. In this paper, we present a reinforcement learning approach that teaches LLMs human-like editing to improve the appropriateness of arguments. Our approach produces self-contained sentence-level edit suggestions that can be accepted or rejected independently. We train the approach using group relative policy optimization with a multi-component reward function that jointly optimizes edit-level semantic similarity, fluency, and pattern conformity as well as argument-level appropriateness. In automatic and human evaluation, it outperforms competitive baselines and the state of the art in human-like editing, with multi-round editing achieving appropriateness close to full rewriting. 

---
# ReasonXL: Shifting LLM Reasoning Language Without Sacrificing Performance 

**Authors**: Daniil Gurgurov, Tom Röhr, Sebastian von Rohrscheidt, Josef van Genabith, Alexander Löser, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2604.12378)  

**Abstract**: Despite advances in multilingual capabilities, most large language models (LLMs) remain English-centric in their training and, crucially, in their production of reasoning traces. Even when tasked with non-English problems, these models predominantly reason in English, creating a fundamental mismatch for non-English usage scenarios.
We address this disparity directly with three contributions. (i) We introduce ReasonXL, the first large-scale parallel corpus of cross-domain reasoning traces spanning five European languages (English, German, French, Italian, and Spanish), with over two million aligned samples per language, each comprising prompts, reasoning traces, and final outputs, enabling direct supervision of language-specific reasoning. (ii) Using ReasonXL, we demonstrate that LLMs can be adapted to reason entirely in a desired target language, using a simple two-stage pipeline of supervised fine-tuning (SFT) followed by reinforcement learning with verifiable rewards (RLVR). The resulting models match or exceed baseline performance, with minimal loss in general knowledge and broadly preserved cross-lingual transfer. (iii) We conduct an extensive representational analysis of the adaptation and find a clear functional division across model depth: early layers contain an activation bottleneck that causally determines language identity, while upper layers concentrate the weight and activation changes driven by adaptation. We further find that RLVR achieves greater behavioral divergence from the base model with smaller parameter updates than SFT, suggesting a more efficient representational rerouting despite much smaller weight updates. 

---
# From Myopic Selection to Long-Horizon Awareness: Sequential LLM Routing for Multi-Turn Dialogue 

**Authors**: Jiarui Zhang, Xiangyu Liu, Yong Hu, Chaoyue Niu, Hang Zeng, Shaojie Tang, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.12385)  

**Abstract**: Multi-turn dialogue is the predominant form of interaction with large language models (LLMs). While LLM routing is effective in single-turn settings, existing methods fail to maximize cumulative performance in multi-turn dialogue due to interaction dynamics and delayed rewards. To address this challenge, we move from myopic, single-turn selection to long-horizon sequential routing for multi-turn dialogue. Accordingly, we propose DialRouter, which first performs MCTS to explore dialogue branches induced by different LLM selections and collect trajectories with high cumulative rewards. DialRouter then learns a lightweight routing policy from search-derived data, augmented with retrieval-based future state approximation, enabling multi-turn routing without online search. Experiments on both open-domain and domain-specific dialogue tasks across diverse candidate sets of both open-source and closed-source LLMs demonstrate that DialRouter significantly outperforms single LLMs and existing routing baselines in task success rate, while achieving a superior performance-cost trade-off when combined with a cost-aware reward. 

---
# Think Through Uncertainty: Improving Long-Form Generation Factuality via Reasoning Calibration 

**Authors**: Xin Liu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12046)  

**Abstract**: Large language models (LLMs) often hallucinate in long-form generation. Existing approaches mainly improve factuality through post-hoc revision or reinforcement learning (RL) with correctness-based rewards, but they do not teach the model to estimate which parts of its generation are reliable. As a result, models may still state incorrect claims confidently in their responses. Recent advances in reasoning have significantly improved LLM performance, and have been leveraged to estimate confidence by incorporating calibration into RL objectives. However, existing approaches remain limited to a single scalar confidence for the entire response, which is insufficient for long-form generation where uncertainty varies across individual claims. To mitigate this problem, we propose CURE, a framework that improves long-form factuality by teaching LLMs to reason about uncertainty at the claim level. We first introduce a Claim-Aware Reasoning Protocol, which structures outputs into atomic claims paired with explicit confidence estimates. We then develop a multi-stage training pipeline that aligns model confidence with claims' correctness and then optimizes on factuality. The resulting calibrated confidence further enables selective prediction, allowing the model to abstain from uncertain claims at inference time. Experiments on four long-form factuality benchmarks show that CURE consistently improves factual accuracy over competitive supervised and RL baselines, while maintaining factual recall. In particular, it improves claim-level accuracy by up to 39.9% on Biography generation. These gains are accompanied by improved calibration, as reflected by a 16.0% increase in AUROC on FactBench. 

---
# Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision 

**Authors**: Yinghui He, Simran Kaur, Adithya Bhaskar, Yongjin Yang, Jiarui Liu, Narutatsu Ri, Liam Fowl, Abhishek Panigrahi, Danqi Chen, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2604.12002)  

**Abstract**: Current post-training methods in verifiable settings fall into two categories. Reinforcement learning (RLVR) relies on binary rewards, which are broadly applicable and powerful, but provide only sparse supervision during training. Distillation provides dense token-level supervision, typically obtained from an external teacher or using high-quality demonstrations. Collecting such supervision can be costly or unavailable. We propose Self-Distillation Zero (SD-Zero), a method that is substantially more training sample-efficient than RL and does not require an external teacher or high-quality demonstrations. SD-Zero trains a single model to play two roles: a Generator, which produces an initial response, and a Reviser, which conditions on that response and its binary reward to produce an improved response. We then perform on-policy self-distillation to distill the reviser into the generator, using the reviser's token distributions conditioned on the generator's response and its reward as supervision. In effect, SD-Zero trains the model to transform binary rewards into dense token-level self-supervision. On math and code reasoning benchmarks with Qwen3-4B-Instruct and Olmo-3-7B-Instruct, SD-Zero improves performance by at least 10% over the base models and outperforms strong baselines, including Rejection Fine-Tuning (RFT), GRPO, and Self-Distillation Fine-Tuning (SDFT), under the same question set and training sample budget. Extensive ablation studies show two novel characteristics of our proposed algorithm: (a) token-level self-localization, where the reviser can identify the key tokens that need to be revised in the generator's response based on reward, and (b) iterative self-evolution, where the improving ability to revise answers can be distilled back into generation performance with regular teacher synchronization. 

---
# From Imitation to Discrimination: Progressive Curriculum Learning for Robust Web Navigation 

**Authors**: Chuang Peng, Wei Zhang, Renshuai Tao, Xinhao Zhang, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12666)  

**Abstract**: Text-based web agents offer computational efficiency for autonomous web navigation, yet developing robust agents remains challenging due to the noisy and heterogeneous nature of real-world HTML. Standard Supervised Fine-Tuning (SFT) approaches fail in two critical dimensions: they lack discrimination capabilities to reject plausible but incorrect elements in densely populated pages, and exhibit limited generalization to unseen website layouts. To address these challenges, we introduce the Triton dataset (590k instances) and a progressive training curriculum. Triton is constructed via Structural-Semantic Hard Negative Mining, which explicitly mines topologically similar distractors, and a Dual-Agent Consensus pipeline that synthesizes diverse cross-domain tasks with strict verification. Building upon this foundation, our progressive curriculum produces three models: Triton-SFT-32B for basic imitation, Triton-ORPO-32B for robust discrimination via Odds Ratio Preference Optimization, and Triton-GRPO-32B for long-horizon consistency through Group Relative Policy Optimization. Empirical evaluation on Mind2Web demonstrates that Triton-GRPO-32B achieves state-of-the-art performance among open-source models with 58.7% Step Success Rate, surpassing GPT-4.5 (42.4%) and Claude-4.5 (41.4%) by over 16%, validating that specialized data curriculum outweighs raw parameter scale for web navigation. 

---
# Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization 

**Authors**: Yizhe Chi, Deyao Hong, Dapeng Jiang, Tianwei Luo, Kaisen Yang, Boshi Zhang, Zhe Cao, Xiaoyan Fan, Bingxiang He, Han Hao, Weiyang Jin, Dianqiao Lei, Qingle Liu, Houde Qian, Bowen Wang, Situ Wang, Youjie Zheng, Yifan Zhou, Calvin Xiao, Eren Cai, Qinhuai Na  

**Link**: [PDF](https://arxiv.org/pdf/2604.12290)  

**Abstract**: Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Frontier-Eng, a human-verified benchmark for generative optimization -- an iterative propose-execute-evaluate loop in which an agent generates candidate artifacts, receives executable verifier feedback, and revises them under a fixed interaction budget -- spanning $47$ tasks across five broad engineering categories. Unlike previous suites, Frontier-Eng tasks are grounded in industrial-grade simulators and verifiers that provide continuous reward signals and enforce hard feasibility constraints under constrained budgets. We evaluate eight frontier language models using representative search frameworks, finding that while Claude 4.6 Opus achieves the most robust performance, the benchmark remains challenging for all models. Our analysis suggests a dual power-law decay in improvement frequency ($\sim$ 1/iteration) and magnitude ($\sim$ 1/improvement count). We further show that although width improves parallelism and diversity, depth remains crucial for hard-won improvements under a fixed budget. Frontier-Eng establishes a new standard for assessing the capacity of AI agents to integrate domain knowledge with executable feedback to solve complex, open-ended engineering problems. 

---
