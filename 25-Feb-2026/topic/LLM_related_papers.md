# A Benchmark for Deep Information Synthesis 

**Authors**: Debjit Paul, Daniel Murphy, Milan Gritta, Ronald Cardenas, Victor Prokhorov, Lena Sophia Bolliger, Aysim Toker, Roy Miles, Andreea-Maria Oncescu, Jasivan Alex Sivakumar, Philipp Borchert, Ismail Elezi, Meiru Zhang, Ka Yiu Lee, Guchun Zhang, Jun Wang, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2602.21143)  

**Abstract**: Large language model (LLM)-based agents are increasingly used to solve complex tasks involving tool use, such as web browsing, code execution, and data analysis. However, current evaluation benchmarks do not adequately assess their ability to solve real-world tasks that require synthesizing information from multiple sources and inferring insights beyond simple fact retrieval. To address this, we introduce DEEPSYNTH, a novel benchmark designed to evaluate agents on realistic, time-consuming problems that combine information gathering, synthesis, and structured reasoning to produce insights. DEEPSYNTH contains 120 tasks collected across 7 domains and data sources covering 67 countries. DEEPSYNTH is constructed using a multi-stage data collection pipeline that requires annotators to collect official data sources, create hypotheses, perform manual analysis, and design tasks with verifiable answers. When evaluated on DEEPSYNTH, 11 state-of-the-art LLMs and deep research agents achieve a maximum F1 score of 8.97 and 17.5 on the LLM-judge metric, underscoring the difficulty of the benchmark. Our analysis reveals that current agents struggle with hallucinations and reasoning over large information spaces, highlighting DEEPSYNTH as a crucial benchmark for guiding future research. 

---
# Tool Building as a Path to "Superintelligence" 

**Authors**: David Koplow, Tomer Galanti, Tomaso Poggio  

**Link**: [PDF](https://arxiv.org/pdf/2602.21061)  

**Abstract**: The Diligent Learner framework suggests LLMs can achieve superintelligence via test-time search, provided a sufficient step-success probability $\gamma$. In this work, we design a benchmark to measure $\gamma$ on logical out-of-distribution inference. We construct a class of tasks involving GF(2) circuit reconstruction that grow more difficult with each reasoning step, and that are, from an information-theoretic standpoint, impossible to reliably solve unless the LLM carefully integrates all of the information provided. Our analysis demonstrates that while the $\gamma$ value for small LLMs declines superlinearly as depth increases, frontier models exhibit partial robustness on this task. Furthermore, we find that successful reasoning at scale is contingent upon precise tool calls, identifying tool design as a critical capability for LLMs to achieve general superintelligence through the Diligent Learner framework. 

---
# LogicGraph : Benchmarking Multi-Path Logical Reasoning via Neuro-Symbolic Generation and Verification 

**Authors**: Yanrui Wu, Lingling Zhang, Xinyu Zhang, Jiayu Chang, Pengyu Li, Xu Jiang, Jingtao Hu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21044)  

**Abstract**: Evaluations of large language models (LLMs) primarily emphasize convergent logical reasoning, where success is defined by producing a single correct proof. However, many real-world reasoning problems admit multiple valid derivations, requiring models to explore diverse logical paths rather than committing to one route. To address this limitation, we introduce LogicGraph, the first benchmark aimed to systematically evaluate multi-path logical reasoning, constructed via a neuro-symbolic framework that leverages backward logic generation and semantic instantiation. This pipeline yields solver-verified reasoning problems formalized by high-depth multi-path reasoning and inherent logical distractions, where each instance is associated with an exhaustive set of minimal proofs. We further propose a reference-free evaluation framework to rigorously assess model performance in both convergent and divergent regimes. Experiments on state-of-the-art language models reveal a common limitation: models tend to commit early to a single route and fail to explore alternatives, and the coverage gap grows substantially with reasoning depth. LogicGraph exposes this divergence gap and provides actionable insights to motivate future improvements. Our code and data will be released at this https URL. 

---
# Predicting Sentence Acceptability Judgments in Multimodal Contexts 

**Authors**: Hyewon Jang, Nikolai Ilinykh, Sharid Loáiciga, Jey Han Lau, Shalom Lappin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20918)  

**Abstract**: Previous work has examined the capacity of deep neural networks (DNNs), particularly transformers, to predict human sentence acceptability judgments, both independently of context, and in document contexts. We consider the effect of prior exposure to visual images (i.e., visual context) on these judgments for humans and large language models (LLMs). Our results suggest that, in contrast to textual context, visual images appear to have little if any impact on human acceptability ratings. However, LLMs display the compression effect seen in previous work on human judgments in document contexts. Different sorts of LLMs are able to predict human acceptability judgments to a high degree of accuracy, but in general, their performance is slightly better when visual contexts are removed. Moreover, the distribution of LLM judgments varies among models, with Qwen resembling human patterns, and others diverging from them. LLM-generated predictions on sentence acceptability are highly correlated with their normalised log probabilities in general. However, the correlations decrease when visual contexts are present, suggesting that a higher gap exists between the internal representations of LLMs and their generated predictions in the presence of visual contexts. Our experimental work suggests interesting points of similarity and of difference between human and LLM processing of sentences in multimodal contexts. 

---
# Qwen-BIM: developing large language model for BIM-based design with domain-specific benchmark and dataset 

**Authors**: Jia-Rui Lin, Yun-Hong Cai, Xiang-Rui Ni, Shaojie Zhou, Peng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2602.20812)  

**Abstract**: As the construction industry advances toward digital transformation, BIM (Building Information Modeling)-based design has become a key driver supporting intelligent construction. Despite Large Language Models (LLMs) have shown potential in promoting BIM-based design, the lack of specific datasets and LLM evaluation benchmarks has significantly hindered the performance of LLMs. Therefore, this paper addresses this gap by proposing: 1) an evaluation benchmark for BIM-based design together with corresponding quantitative indicators to evaluate the performance of LLMs, 2) a method for generating textual data from BIM and constructing corresponding BIM-derived datasets for LLM evaluation and fine-tuning, and 3) a fine-tuning strategy to adapt LLMs for BIM-based design. Results demonstrate that the proposed domain-specific benchmark effectively and comprehensively assesses LLM capabilities, highlighting that general LLMs are still incompetent for domain-specific tasks. Meanwhile, with the proposed benchmark and datasets, Qwen-BIM is developed and achieves a 21.0% average increase in G-Eval score compared to the base LLM model. Notably, with only 14B parameters, performance of Qwen-BIM is comparable to that of general LLMs with 671B parameters for BIM-based design tasks. Overall, this study develops the first domain-specific LLM for BIM-based design by introducing a comprehensive benchmark and high-quality dataset, which provide a solid foundation for developing BIM-related LLMs in various fields. 

---
# Pipeline for Verifying LLM-Generated Mathematical Solutions 

**Authors**: Varvara Sazonova, Dmitri Shmelkin, Stanislav Kikot, Vasily Motolygin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20770)  

**Abstract**: With the growing popularity of Large Reasoning Models and their results in solving mathematical problems, it becomes crucial to measure their capabilities. We introduce a pipeline for both automatic and interactive verification as a more accurate alternative to only checking the answer which is currently the most popular approach for benchmarks. The pipeline can also be used as a generator of correct solutions both in formal and informal languages. 3 AI agents, which can be chosen for the benchmark accordingly, are included in the structure. The key idea is the use of prompts to obtain the solution in the specific form which allows for easier verification using proof assistants and possible use of small models ($\le 8B$). Experiments on several datasets suggest low probability of False Positives. The open-source implementation with instructions on setting up a server is available at this https URL. 

---
# PromptCD: Test-Time Behavior Enhancement via Polarity-Prompt Contrastive Decoding 

**Authors**: Baolong Bi, Yuyao Ge, Shenghua Liu, Yuchen He, Siqian Tong, Lizhe Chen, Lingrui Mei, Zehao Li, Yiwei Wang, Yujun Cai, Ming-Hsuan Yang, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.20696)  

**Abstract**: Reliable AI systems require large language models (LLMs) to exhibit behaviors aligned with human preferences and values. However, most existing alignment approaches operate at training time and rely on additional high-quality data, incurring significant computational and annotation costs. While recent work has shown that contrastive decoding can leverage a model's internal distributions to improve specific capabilities, its applicability remains limited to narrow behavioral scopes and scenarios. In this work, we introduce Polarity-Prompt Contrastive Decoding (PromptCD), a test-time behavior control method that generalizes contrastive decoding to broader enhancement settings. PromptCD constructs paired positive and negative guiding prompts for a target behavior and contrasts model responses-specifically token-level probability distributions in LLMs and visual attention patterns in VLMs-to reinforce desirable outcomes. This formulation extends contrastive decoding to a wide range of enhancement objectives and is applicable to both LLMs and Vision-Language Models (VLMs) without additional training. For LLMs, experiments on the "3H" alignment objectives (helpfulness, honesty, and harmlessness) demonstrate consistent and substantial improvements, indicating that post-trained models can achieve meaningful self-enhancement purely at test time. For VLMs, we further analyze contrastive effects on visual attention, showing that PromptCD significantly improves VQA performance by reinforcing behavior-consistent visual grounding. Collectively, these results highlight PromptCD as a simple, general, and cost-efficient strategy for reliable behavior control across modalities. 

---
# Counterfactual Simulation Training for Chain-of-Thought Faithfulness 

**Authors**: Peter Hase, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2602.20710)  

**Abstract**: Inspecting Chain-of-Thought reasoning is among the most common means of understanding why an LLM produced its output. But well-known problems with CoT faithfulness severely limit what insights can be gained from this practice. In this paper, we introduce a training method called Counterfactual Simulation Training (CST), which aims to improve CoT faithfulness by rewarding CoTs that enable a simulator to accurately predict a model's outputs over counterfactual inputs. We apply CST in two settings: (1) CoT monitoring with cue-based counterfactuals, to detect when models rely on spurious features, reward hack, or are sycophantic, and (2) counterfactual simulation over generic model-based counterfactuals, to encourage models to produce more faithful, generalizable reasoning in the CoT. Experiments with models up to 235B parameters show that CST can substantially improve monitor accuracy on cue-based counterfactuals (by 35 accuracy points) as well as simulatability over generic counterfactuals (by 2 points). We further show that: (1) CST outperforms prompting baselines, (2) rewriting unfaithful CoTs with an LLM is 5x more efficient than RL alone, (3) faithfulness improvements do not generalize to dissuading cues (as opposed to persuading cues), and (4) larger models do not show more faithful CoT out of the box, but they do benefit more from CST. These results suggest that CST can improve CoT faithfulness in general, with promising applications for CoT monitoring. Code for experiments in this paper is available at this https URL 

---
# Grounding LLMs in Scientific Discovery via Embodied Actions 

**Authors**: Bo Zhang, Jinfeng Zhou, Yuxuan Chen, Jianing Yin, Minlie Huang, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20639)  

**Abstract**: Large Language Models (LLMs) have shown significant potential in scientific discovery but struggle to bridge the gap between theoretical reasoning and verifiable physical simulation. Existing solutions operate in a passive "execute-then-response" loop and thus lacks runtime perception, obscuring agents to transient anomalies (e.g., numerical instability or diverging oscillations). To address this limitation, we propose EmbodiedAct, a framework that transforms established scientific software into active embodied agents by grounding LLMs in embodied actions with a tight perception-execution loop. We instantiate EmbodiedAct within MATLAB and evaluate it on complex engineering design and scientific modeling tasks. Extensive experiments show that EmbodiedAct significantly outperforms existing baselines, achieving SOTA performance by ensuring satisfactory reliability and stability in long-horizon simulations and enhanced accuracy in scientific modeling. 

---
# From Logs to Language: Learning Optimal Verbalization for LLM-Based Recommendation in Production 

**Authors**: Yucheng Shi, Ying Li, Yu Wang, Yesu Feng, Arjun Rao, Rein Houthooft, Shradha Sehgal, Jin Wang, Hao Zhen, Ninghao Liu, Linas Baltrunas  

**Link**: [PDF](https://arxiv.org/pdf/2602.20558)  

**Abstract**: Large language models (LLMs) are promising backbones for generative recommender systems, yet a key challenge remains underexplored: verbalization, i.e., converting structured user interaction logs into effective natural language inputs. Existing methods rely on rigid templates that simply concatenate fields, yielding suboptimal representations for recommendation. We propose a data-centric framework that learns verbalization for LLM-based recommendation. Using reinforcement learning, a verbalization agent transforms raw interaction histories into optimized textual contexts, with recommendation accuracy as the training signal. This agent learns to filter noise, incorporate relevant metadata, and reorganize information to improve downstream predictions. Experiments on a large-scale industrial streaming dataset show that learned verbalization delivers up to 93% relative improvement in discovery item recommendation accuracy over template-based baselines. Further analysis reveals emergent strategies such as user interest summarization, noise removal, and syntax normalization, offering insights into effective context construction for LLM-based recommender systems. 

---
# Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training 

**Authors**: Anas Barakat, Souradip Chakraborty, Khushbu Pahwa, Amrit Singh Bedi  

**Link**: [PDF](https://arxiv.org/pdf/2602.21189)  

**Abstract**: Pass@k is a widely used performance metric for verifiable large language model tasks, including mathematical reasoning, code generation, and short-answer reasoning. It defines success if any of $k$ independently sampled solutions passes a verifier. This multi-sample inference metric has motivated inference-aware fine-tuning methods that directly optimize pass@$k$. However, prior work reports a recurring trade-off: pass@k improves while pass@1 degrades under such methods. This trade-off is practically important because pass@1 often remains a hard operational constraint due to latency and cost budgets, imperfect verifier coverage, and the need for a reliable single-shot fallback. We study the origin of this trade-off and provide a theoretical characterization of when pass@k policy optimization can reduce pass@1 through gradient conflict induced by prompt interference. We show that pass@$k$ policy gradients can conflict with pass@1 gradients because pass@$k$ optimization implicitly reweights prompts toward low-success prompts; when these prompts are what we term negatively interfering, their upweighting can rotate the pass@k update direction away from the pass@1 direction. We illustrate our theoretical findings with large language model experiments on verifiable mathematical reasoning tasks. 

---
# DMCD: Semantic-Statistical Framework for Causal Discovery 

**Authors**: Samarth KaPatel, Sofia Nikiforova, Giacinto Paolo Saggese, Paul Smith  

**Link**: [PDF](https://arxiv.org/pdf/2602.20333)  

**Abstract**: We present DMCD (DataMap Causal Discovery), a two-phase causal discovery framework that integrates LLM-based semantic drafting from variable metadata with statistical validation on observational data. In Phase I, a large language model proposes a sparse draft DAG, serving as a semantically informed prior over the space of possible causal structures. In Phase II, this draft is audited and refined via conditional independence testing, with detected discrepancies guiding targeted edge revisions.
We evaluate our approach on three metadata-rich real-world benchmarks spanning industrial engineering, environmental monitoring, and IT systems analysis. Across these datasets, DMCD achieves competitive or leading performance against diverse causal discovery baselines, with particularly large gains in recall and F1 score. Probing and ablation experiments suggest that these improvements arise from semantic reasoning over metadata rather than memorization of benchmark graphs. Overall, our results demonstrate that combining semantic priors with principled statistical verification yields a high-performing and practically effective approach to causal structure learning. 

---
# An artificial intelligence framework for end-to-end rare disease phenotyping from clinical notes using large language models 

**Authors**: Cathy Shyr, Yan Hu, Rory J. Tinker, Thomas A. Cassini, Kevin W. Byram, Rizwan Hamid, Daniel V. Fabbri, Adam Wright, Josh F. Peterson, Lisa Bastarache, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20324)  

**Abstract**: Phenotyping is fundamental to rare disease diagnosis, but manual curation of structured phenotypes from clinical notes is labor-intensive and difficult to scale. Existing artificial intelligence approaches typically optimize individual components of phenotyping but do not operationalize the full clinical workflow of extracting features from clinical text, standardizing them to Human Phenotype Ontology (HPO) terms, and prioritizing diagnostically informative HPO terms. We developed RARE-PHENIX, an end-to-end AI framework for rare disease phenotyping that integrates large language model-based phenotype extraction, ontology-grounded standardization to HPO terms, and supervised ranking of diagnostically informative phenotypes. We trained RARE-PHENIX using data from 2,671 patients across 11 Undiagnosed Diseases Network clinical sites, and externally validated it on 16,357 real-world clinical notes from Vanderbilt University Medical Center. Using clinician-curated HPO terms as the gold standard, RARE-PHENIX consistently outperformed a state-of-the-art deep learning baseline (PhenoBERT) across ontology-based similarity and precision-recall-F1 metrics in end-to-end evaluation (i.e., ontology-based similarity of 0.70 vs. 0.58). Ablation analyses demonstrated performance improvements with the addition of each module in RARE-PHENIX (extraction, standardization, and prioritization), supporting the value of modeling the full clinical phenotyping workflow. By modeling phenotyping as a clinically aligned workflow rather than a single extraction task, RARE-PHENIX provides structured, ranked phenotypes that are more concordant with clinician curation and has the potential to support human-in-the-loop rare disease diagnosis in real-world settings. 

---
# SparkMe: Adaptive Semi-Structured Interviewing for Qualitative Insight Discovery 

**Authors**: David Anugraha, Vishakh Padmakumar, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21136)  

**Abstract**: Qualitative insights from user experiences are critical for informing product and policy decisions, but collecting such data at scale is constrained by the time and availability of experts to conduct semi-structured interviews. Recent work has explored using large language models (LLMs) to automate interviewing, yet existing systems lack a principled mechanism for balancing systematic coverage of predefined topics with adaptive exploration, or the ability to pursue follow-ups, deep dives, and emergent themes that arise organically during conversation. In this work, we formulate adaptive semi-structured interviewing as an optimization problem over the interviewer's behavior. We define interview utility as a trade-off between coverage of a predefined interview topic guide, discovery of relevant emergent themes, and interview cost measured by length. Based on this formulation, we introduce SparkMe, a multi-agent LLM interviewer that performs deliberative planning via simulated conversation rollouts to select questions with high expected utility. We evaluate SparkMe through controlled experiments with LLM-based interviewees, showing that it achieves higher interview utility, improving topic guide coverage (+4.7% over the best baseline) and eliciting richer emergent insights while using fewer conversational turns than prior LLM interviewing approaches. We further validate SparkMe in a user study with 70 participants across 7 professions on the impact of AI on their workflows. Domain experts rate SparkMe as producing high-quality adaptive interviews that surface helpful profession-specific insights not captured by prior approaches. The code, datasets, and evaluation protocols for SparkMe are available as open-source at this https URL. 

---
# The Art of Efficient Reasoning: Data, Reward, and Optimization 

**Authors**: Taiqiang Wu, Zenan Zu, Bo Zhou, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2602.20945)  

**Abstract**: Large Language Models (LLMs) consistently benefit from scaled Chain-of-Thought (CoT) reasoning, but also suffer from heavy computational overhead. To address this issue, efficient reasoning aims to incentivize short yet accurate thinking trajectories, typically through reward shaping with Reinforcement Learning (RL). In this paper, we systematically investigate the mechanics of efficient reasoning for LLMs. For comprehensive evaluation, we advocate for more fine-grained metrics, including length distribution conditioned on correctness and performance across a wide spectrum of token budgets ranging from 2k to 32k. First, we reveal that the training process follows a two-stage paradigm: length adaptation and reasoning refinement. After that, we conduct extensive experiments (about 0.2 million GPU hours) in a unified protocol, deconstructing training prompts and rollouts, reward shaping, and optimization strategies. In particular, a key finding is to train on relatively easier prompts, ensuring the density of positive reward signals and thus avoiding the length collapse. Meanwhile, the learned length bias can be generalized across domains. We distill all findings into valuable insights and practical guidelines, and further validate them across the Qwen3 series, ranging from 0.6B to 30B, demonstrating the robustness and generalization. 

---
# Actor-Curator: Co-adaptive Curriculum Learning via Policy-Improvement Bandits for RL Post-Training 

**Authors**: Zhengyao Gu, Jonathan Light, Raul Astudillo, Ziyu Ye, Langzhou He, Henry Peng Zou, Wei Cheng, Santiago Paternain, Philip S. Yu, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2602.20532)  

**Abstract**: Post-training large foundation models with reinforcement learning typically relies on massive and heterogeneous datasets, making effective curriculum learning both critical and challenging. In this work, we propose ACTOR-CURATOR, a scalable and fully automated curriculum learning framework for reinforcement learning post-training of large language models (LLMs). ACTOR-CURATOR learns a neural curator that dynamically selects training problems from large problem banks by directly optimizing for expected policy performance improvement. We formulate problem selection as a non-stationary stochastic bandit problem, derive a principled loss function based on online stochastic mirror descent, and establish regret guarantees under partial feedback. Empirically, ACTOR-CURATOR consistently outperforms uniform sampling and strong curriculum baselines across a wide range of challenging reasoning benchmarks, demonstrating improved training stability and efficiency. Notably, it achieves relative gains of 28.6% on AIME2024 and 30.5% on ARC-1D over the strongest baseline and up to 80% speedup. These results suggest that ACTOR-CURATOR is a powerful and practical approach for scalable LLM post-training. 

---
# Hybrid LLM-Embedded Dialogue Agents for Learner Reflection: Designing Responsive and Theory-Driven Interactions 

**Authors**: Paras Sharma, YuePing Sha, Janet Shufor Bih Epse Fofang, Brayden Yan, Jess A. Turner, Nicole Balay, Hubert O. Asare, Angela E.B. Stewart, Erin Walker  

**Link**: [PDF](https://arxiv.org/pdf/2602.20486)  

**Abstract**: Dialogue systems have long supported learner reflections, with theoretically grounded, rule-based designs offering structured scaffolding but often struggling to respond to shifts in engagement. Large Language Models (LLMs), in contrast, can generate context-sensitive responses but are not informed by decades of research on how learning interactions should be structured, raising questions about their alignment with pedagogical theories. This paper presents a hybrid dialogue system that embeds LLM responsiveness within a theory-aligned, rule-based framework to support learner reflections in a culturally responsive robotics summer camp. The rule-based structure grounds dialogue in self-regulated learning theory, while the LLM decides when and how to prompt deeper reflections, responding to evolving conversation context. We analyze themes across dialogues to explore how our hybrid system shaped learner reflections. Our findings indicate that LLM-embedded dialogues supported richer learner reflections on goals and activities, but also introduced challenges due to repetitiveness and misalignment in prompts, reducing engagement. 

---
# Wireless Federated Multi-Task LLM Fine-Tuning via Sparse-and-Orthogonal LoRA 

**Authors**: Nuocheng Yang, Sihua Wang, Ouwen Huan, Mingzhe Chen, Tony Q. S. Quek, Changchuan Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20492)  

**Abstract**: Decentralized federated learning (DFL) based on low-rank adaptation (LoRA) enables mobile devices with multi-task datasets to collaboratively fine-tune a large language model (LLM) by exchanging locally updated parameters with a subset of neighboring devices via wireless connections for knowledge this http URL, directly aggregating parameters fine-tuned on heterogeneous datasets induces three primary issues across the DFL life-cycle: (i) \textit{catastrophic knowledge forgetting during fine-tuning process}, arising from conflicting update directions caused by data heterogeneity; (ii) \textit{inefficient communication and convergence during model aggregation process}, due to bandwidth-intensive redundant model transmissions; and (iii) \textit{multi-task knowledge interference during inference process}, resulting from incompatible knowledge representations coexistence during inference. To address these issues in a fully decentralized scenario, we first propose a sparse-and-orthogonal LoRA that ensures orthogonality between model updates to eliminate direction conflicts during this http URL, we analyze how device connection topology affects multi-task performance, prompting a cluster-based topology design during this http URL, we propose an implicit mixture of experts (MoE) mechanism to avoid the coexistence of incompatible knowledge during inference. Simulation results demonstrate that the proposed approach effectively reduces communication resource consumption by up to $73\%$ and enhances average performance by $5\%$ compared with the traditional LoRA method. 

---
# No One Size Fits All: QueryBandits for Hallucination Mitigation 

**Authors**: Nicole Cho, William Watson, Alec Koppel, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.20332)  

**Abstract**: Advanced reasoning capabilities in Large Language Models (LLMs) have led to more frequent hallucinations; yet most mitigation work focuses on open-source models for post-hoc detection and parameter editing. The dearth of studies focusing on hallucinations in closed-source models is especially concerning, as they constitute the vast majority of models in institutional deployments. We introduce QueryBandits, a model-agnostic contextual bandit framework that adaptively learns online to select the optimal query-rewrite strategy by leveraging an empirically validated and calibrated reward function. Across 16 QA scenarios, our top QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a No-Rewrite baseline and outperforms zero-shot static policies (e.g., Paraphrase or Expand) by 42.6% and 60.3%, respectively. Moreover, all contextual bandits outperform vanilla bandits across all datasets, with higher feature variance coinciding with greater variance in arm selection. This substantiates our finding that there is no single rewrite policy optimal for all queries. We also discover that certain static policies incur higher cumulative regret than No-Rewrite, indicating that an inflexible query-rewriting policy can worsen hallucinations. Thus, learning an online policy over semantic features with QueryBandits can shift model behavior purely through forward-pass mechanisms, enabling its use with closed-source models and bypassing the need for retraining or gradient-based adaptation. 

---
# What Makes a Good Query? Measuring the Impact of Human-Confusing Linguistic Features on LLM Performance 

**Authors**: William Watson, Nicole Cho, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.20300)  

**Abstract**: Large Language Model (LLM) hallucinations are usually treated as defects of the model or its decoding strategy. Drawing on classical linguistics, we argue that a query's form can also shape a listener's (and model's) response. We operationalize this insight by constructing a 22-dimension query feature vector covering clause complexity, lexical rarity, and anaphora, negation, answerability, and intention grounding, all known to affect human comprehension. Using 369,837 real-world queries, we ask: Are there certain types of queries that make hallucination more likely? A large-scale analysis reveals a consistent "risk landscape": certain features such as deep clause nesting and underspecification align with higher hallucination propensity. In contrast, clear intention grounding and answerability align with lower hallucination rates. Others, including domain specificity, show mixed, dataset- and model-dependent effects. Thus, these findings establish an empirically observable query-feature representation correlated with hallucination risk, paving the way for guided query rewriting and future intervention studies. 

---
# CodeHacker: Automated Test Case Generation for Detecting Vulnerabilities in Competitive Programming Solutions 

**Authors**: Jingwei Shi, Xinxiang Yin, Jing Huang, Jinman Zhao, Shengyu Tao  

**Link**: [PDF](https://arxiv.org/pdf/2602.20213)  

**Abstract**: The evaluation of Large Language Models (LLMs) for code generation relies heavily on the quality and robustness of test cases. However, existing benchmarks often lack coverage for subtle corner cases, allowing incorrect solutions to pass. To bridge this gap, we propose CodeHacker, an automated agent framework dedicated to generating targeted adversarial test cases that expose latent vulnerabilities in program submissions. Mimicking the hack mechanism in competitive programming, CodeHacker employs a multi-strategy approach, including stress testing, anti-hash attacks, and logic-specific targeting to break specific code submissions. To ensure the validity and reliability of these attacks, we introduce a Calibration Phase, where the agent iteratively refines its own Validator and Checker via self-generated adversarial probes before evaluating contestant this http URL demonstrate that CodeHacker significantly improves the True Negative Rate (TNR) of existing datasets, effectively filtering out incorrect solutions that were previously accepted. Furthermore, generated adversarial cases prove to be superior training data, boosting the performance of RL-trained models on benchmarks like LiveCodeBench. 

---
# PRECTR-V2:Unified Relevance-CTR Framework with Cross-User Preference Mining, Exposure Bias Correction, and LLM-Distilled Encoder Optimization 

**Authors**: Shuzhi Cao, Rong Chen, Ailong He, Shuguang Han, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.20676)  

**Abstract**: In search systems, effectively coordinating the two core objectives of search relevance matching and click-through rate (CTR) prediction is crucial for discovering users' interests and enhancing platform revenue. In our prior work PRECTR, we proposed a unified framework to integrate these two subtasks,thereby eliminating their inconsistency and leading to mutual this http URL, our previous work still faces three main challenges. First, low-active users and new users have limited search behavioral data, making it difficult to achieve effective personalized relevance preference modeling. Second, training data for ranking models predominantly come from high-relevance exposures, creating a distribution mismatch with the broader candidate space in coarse-ranking, leading to generalization bias. Third, due to the latency constraint, the original model employs an Emb+MLP architecture with a frozen BERT encoder, which prevents joint optimization and creates misalignment between representation learning and CTR fine-tuning. To solve these issues, we further reinforce our method and propose PRECTR-V2. Specifically, we mitigate the low-activity users' sparse behavior problem by mining global relevance preferences under the specific query, which facilitates effective personalized relevance modeling for cold-start scenarios. Subsequently, we construct hard negative samples through embedding noise injection and relevance label reconstruction, and optimize their relative ranking against positive samples via pairwise loss, thereby correcting exposure bias. Finally, we pretrain a lightweight transformer-based encoder via knowledge distillation from LLM and SFT on the text relevance classification task. This encoder replaces the frozen BERT module, enabling better adaptation to CTR fine-tuning and advancing beyond the traditional Emb+MLP paradigm. 

---
# An Approach to Combining Video and Speech with Large Language Models in Human-Robot Interaction 

**Authors**: Guanting Shen, Zi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.20219)  

**Abstract**: Interpreting human intent accurately is a central challenge in human-robot interaction (HRI) and a key requirement for achieving more natural and intuitive collaboration between humans and machines. This work presents a novel multimodal HRI framework that combines advanced vision-language models, speech processing, and fuzzy logic to enable precise and adaptive control of a Dobot Magician robotic arm. The proposed system integrates Florence-2 for object detection, Llama 3.1 for natural language understanding, and Whisper for speech recognition, providing users with a seamless and intuitive interface for object manipulation through spoken commands. By jointly addressing scene perception and action planning, the approach enhances the reliability of command interpretation and execution. Experimental evaluations conducted on consumer-grade hardware demonstrate a command execution accuracy of 75\%, highlighting both the robustness and adaptability of the system. Beyond its current performance, the proposed architecture serves as a flexible and extensible foundation for future HRI research, offering a practical pathway toward more sophisticated and natural human-robot collaboration through tightly coupled speech and vision-language processing. 

---
# Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis 

**Authors**: Shrestha Datta, Hongfu Liu, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2602.20207)  

**Abstract**: Knowledge editing in Large Language Models (LLMs) aims to update the model's prediction for a specific query to a desired target while preserving its behavior on all other inputs. This process typically involves two stages: identifying the layer to edit and performing the parameter update. Intuitively, different queries may localize knowledge at different depths of the model, resulting in different sample-wise editing performance for a fixed editing layer. In this work, we hypothesize the existence of fixed golden layers that can achieve near-optimal editing performance similar to sample-wise optimal layers. To validate this hypothesis, we provide empirical evidence by comparing golden layers against ground-truth sample-wise optimal layers. Furthermore, we show that golden layers can be reliably identified using a proxy dataset and generalize effectively to unseen test set queries across datasets. Finally, we propose a novel method, namely Layer Gradient Analysis (LGA) that estimates golden layers efficiently via gradient-attribution, avoiding extensive trial-and-error across multiple editing runs. Extensive experiments on several benchmark datasets demonstrate the effectiveness and robustness of our LGA approach across different LLM types and various knowledge editing methods. 

---
# MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Elastic LLMs 

**Authors**: Dongwei Wang, Jinhee Kim, Seokho Han, Denis Gudovskiy, Yohei Nakata, Tomoyuki Okuno, KhayTze Peong, Kang Eun Jeon, Jong Hwan Ko, Yiran Chen, Huanrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20191)  

**Abstract**: Changing runtime complexity on cloud and edge devices necessitates elastic large language model (LLM) deployment, where an LLM can be inferred with various quantization precisions based on available computational resources. However, it has been observed that the calibration parameters for quantization are typically linked to specific precisions, which presents challenges during elastic-precision calibration and precision switching at runtime. In this work, we attribute the source of varying calibration parameters to the varying token-level sensitivity caused by a precision-dependent outlier migration this http URL by this observation, we propose \texttt{MoBiQuant}, a novel Mixture-of-Bits quantization framework that adjusts weight precision for elastic LLM inference based on token sensitivity. Specifically, we propose the many-in-one recursive residual quantization that can iteratively reconstruct higher-precision weights and the token-aware router to dynamically select the number of residual bit slices. MoBiQuant enables smooth precision switching while improving generalization for the distribution of token outliers. Experimental results demonstrate that MoBiQuant exhibits strong elasticity, enabling it to match the performance of bit-specific calibrated PTQ on LLaMA3-8B without repeated calibration. 

---
# Closing the Expertise Gap in Residential Building Energy Retrofits: A Domain-Specific LLM for Informed Decision-Making 

**Authors**: Lei Shu, Armin Yeganeh, Sinem Mollaoglu, Jiayu Zhou, Dong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.20181)  

**Abstract**: Residential energy retrofit decision-making is constrained by an expertise gap, as homeowners lack the technical literacy required for energy assessments. To address this challenge, this study develops a domain-specific large language model (LLM) that provides optimal retrofit recommendations using homeowner-accessible descriptions of basic dwelling characteristics. The model is fine-tuned on physics-based energy simulations and techno-economic calculations derived from 536,416 U.S. residential building prototypes across nine major retrofit categories. Using Low-Rank Adaptation (LoRA), the LLM maps dwelling characteristics to optimal retrofit selections and associated performance outcomes. Evaluation against physics-grounded baselines shows that the model identifies the optimal retrofit for CO2 reduction within its top three recommendations in 98.9% of cases and the shortest discounted payback period in 93.3% of cases. Fine-tuning yields an order-of-magnitude reduction in CO2 prediction error and multi-fold reductions for energy use and retrofit cost. The model maintains performance under incomplete input conditions, supporting informed residential decarbonization decisions. 

---
# Evaluating the Reliability of Digital Forensic Evidence Discovered by Large Language Model: A Case Study 

**Authors**: Jeel Piyushkumar Khatiwala, Daniel Kwaku Ntiamoah Addai, Weifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20202)  

**Abstract**: The growing reliance on AI-identified digital evidence raises significant concerns about its reliability, particularly as large language models (LLMs) are increasingly integrated into forensic investigations. This paper proposes a structured framework that automates forensic artifact extraction, refines data through LLM-driven analysis, and validates results using a Digital Forensic Knowledge Graph (DFKG). Evaluated on a 13 GB forensic image dataset containing 61 applications, 2,864 databases, and 5,870 tables, the framework ensures artifact traceability and evidentiary consistency through deterministic Unique Identifiers (UIDs) and forensic cross-referencing. We propose this methodology to address challenges in ensuring the credibility and forensic integrity of AI-identified evidence, reducing classification errors, and advancing scalable, auditable methodologies. A comprehensive case study on this dataset demonstrates the framework's effectiveness, achieving over 95 percent accuracy in artifact extraction, strong support of chain-of-custody adherence, and robust contextual consistency in forensic relationships. Key results validate the framework's ability to enhance reliability, reduce errors, and establish a legally sound paradigm for AI-assisted digital forensics. 

---
# Talking to Yourself: Defying Forgetting in Large Language Models 

**Authors**: Yutao Sun, Mingshuai Chen, Tiancheng Zhao, Phillip Miao, Zilun Zhang, Haozhan Shen, Ruizhe Zhu, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20162)  

**Abstract**: Catastrophic forgetting remains a major challenge when fine-tuning large language models (LLMs) on narrow, task-specific data, often degrading their general knowledge and reasoning abilities. We propose SA-SFT, a lightweight self-augmentation routine in which an LLM generates self-dialogues prior to fine-tuning, and the resulting self-authored data are mixed with task data without modifying optimization or training schedules.
Despite requiring no external data or additional tuning, SA-SFT consistently mitigates catastrophic forgetting while improving in-domain performance. Across 50 evaluation scenarios, it maintains performance comparable to the original model and achieves the best results in 40 cases, outperforming common baselines such as layer freezing and external data mixing. Guided by these empirical findings, we further present a theoretical analysis suggesting that forgetting can partly stem from style-induced parameter drift, and that self-alignment through self-generated data provides an effective means to counteract this effect. Overall, our results indicate that self-augmentation offers a simple and effective mechanism for robust LLM adaptation without incurring catastrophic forgetting. 

---
# Generative Pseudo-Labeling for Pre-Ranking with LLMs 

**Authors**: Junyu Bi, Xinting Niu, Daixuan Cheng, Kun Yuan, Tao Wang, Binbin Cao, Jian Wu, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20995)  

**Abstract**: Pre-ranking is a critical stage in industrial recommendation systems, tasked with efficiently scoring thousands of recalled items for downstream ranking. A key challenge is the train-serving discrepancy: pre-ranking models are trained only on exposed interactions, yet must score all recalled candidates -- including unexposed items -- during online serving. This mismatch not only induces severe sample selection bias but also degrades generalization, especially for long-tail content. Existing debiasing approaches typically rely on heuristics (e.g., negative sampling) or distillation from biased rankers, which either mislabel plausible unexposed items as negatives or propagate exposure bias into pseudo-labels. In this work, we propose Generative Pseudo-Labeling (GPL), a framework that leverages large language models (LLMs) to generate unbiased, content-aware pseudo-labels for unexposed items, explicitly aligning the training distribution with the online serving space. By offline generating user-specific interest anchors and matching them with candidates in a frozen semantic space, GPL provides high-quality supervision without adding online latency. Deployed in a large-scale production system, GPL improves click-through rate by 3.07%, while significantly enhancing recommendation diversity and long-tail item discovery. 

---
# Evaluating Proactive Risk Awareness of Large Language Models 

**Authors**: Xuan Luo, Yubin Chen, Zhiyu Hou, Linpu Yu, Geng Tu, Jing Li, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20976)  

**Abstract**: As large language models (LLMs) are increasingly embedded in everyday decision-making, their safety responsibilities extend beyond reacting to explicit harmful intent toward anticipating unintended but consequential risks. In this work, we introduce a proactive risk awareness evaluation framework that measures whether LLMs can anticipate potential harms and provide warnings before damage occurs. We construct the Butterfly dataset to instantiate this framework in the environmental and ecological domain. It contains 1,094 queries that simulate ordinary solution-seeking activities whose responses may induce latent ecological impact. Through experiments across five widely used LLMs, we analyze the effects of response length, languages, and modality. Experimental results reveal consistent, significant declines in proactive awareness under length-restricted responses, cross-lingual similarities, and persistent blind spots in (multimodal) species protection. These findings highlight a critical gap between current safety alignment and the requirements of real-world ecological responsibility, underscoring the need for proactive safeguards in LLM deployment. 

---
# FinAnchor: Aligned Multi-Model Representations for Financial Prediction 

**Authors**: Zirui He, Huopu Zhang, Yanguang Liu, Sirui Wu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2602.20859)  

**Abstract**: Financial prediction from long documents involves significant challenges, as actionable signals are often sparse and obscured by noise, and the optimal LLM for generating embeddings varies across tasks and time periods. In this paper, we propose FinAnchor(Financial Anchored Representations), a lightweight framework that integrates embeddings from multiple LLMs without fine-tuning the underlying models. FinAnchor addresses the incompatibility of feature spaces by selecting an anchor embedding space and learning linear mappings to align representations from other models into this anchor. These aligned features are then aggregated to form a unified representation for downstream prediction. Across multiple financial NLP tasks, FinAnchor consistently outperforms strong single-model baselines and standard ensemble methods, demonstrating the effectiveness of anchoring heterogeneous representations for robust financial prediction. 

---
# Beyond the Star Rating: A Scalable Framework for Aspect-Based Sentiment Analysis Using LLMs and Text Classification 

**Authors**: Vishal Patil, Shree Vaishnavi Bacha, Revanth Yamani, Yidan Sun, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2602.21082)  

**Abstract**: Customer-provided reviews have become an important source of information for business owners and other customers alike. However, effectively analyzing millions of unstructured reviews remains challenging. While large language models (LLMs) show promise for natural language understanding, their application to large-scale review analysis has been limited by computational costs and scalability concerns. This study proposes a hybrid approach that uses LLMs for aspect identification while employing classic machine-learning methods for sentiment classification at scale. Using ChatGPT to analyze sampled restaurant reviews, we identified key aspects of dining experiences and developed sentiment classifiers using human-labeled reviews, which we subsequently applied to 4.7 million reviews collected over 17 years from a major online platform. Regression analysis reveals that our machine-labeled aspects significantly explain variance in overall restaurant ratings across different aspects of dining experiences, cuisines, and geographical regions. Our findings demonstrate that combining LLMs with traditional machine learning approaches can effectively automate aspect-based sentiment analysis of large-scale customer feedback, suggesting a practical framework for both researchers and practitioners in the hospitality industry and potentially, other service sectors. 

---
# Overton Pluralistic Reinforcement Learning for Large Language Models 

**Authors**: Yu Fu, Seongho Son, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2602.20759)  

**Abstract**: Existing alignment paradigms remain limited in capturing the pluralistic nature of human values. Overton Pluralism addresses this gap by generating responses with diverse perspectives from a single query. This paper introduces OP-GRPO (Overton Pluralistic Group Relative Policy Optimization), a reinforcement learning framework for implicit Overton Pluralism that enables a single large language model to produce pluralistic responses without explicit prompting or modular orchestration. Our workflow consists of two main steps. First, similarity estimator training fine-tunes a Sentence Transformer for Overton Pluralism tasks to provide more accurate coverage evaluation of generated responses. Second, OP-GRPO training incorporates this similarity estimator into a dual-reward system designed to ensure both broad coverage of genuine human perspectives and the uniqueness of each perspective, thereby promoting diversity. Empirical results demonstrate a "small models, big perspective coverage" effect. The trained Qwen2.5-3B-Instruct model surpasses a 20B GPT-OSS baseline with a 37.4 percent relative accuracy gain on a Natural Language Inference benchmark, and also outperforms a modular architecture baseline with a 19.1 percent relative improvement. Additional evaluations using GPT-4.1 as a large language model judge further confirm the robustness of the approach. 

---
# CARE: An Explainable Computational Framework for Assessing Client-Perceived Therapeutic Alliance Using Large Language Models 

**Authors**: Anqi Li, Chenxiao Wang, Yu Lu, Renjun Xu, Lizhi Ma, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2602.20648)  

**Abstract**: Client perceptions of the therapeutic alliance are critical for counseling effectiveness. Accurately capturing these perceptions remains challenging, as traditional post-session questionnaires are burdensome and often delayed, while existing computational approaches produce coarse scores, lack interpretable rationales, and fail to model holistic session context. We present CARE, an LLM-based framework to automatically predict multi-dimensional alliance scores and generate interpretable rationales from counseling transcripts. Built on the CounselingWAI dataset and enriched with 9,516 expert-curated rationales, CARE is fine-tuned using rationale-augmented supervision with the LLaMA-3.1-8B-Instruct backbone. Experiments show that CARE outperforms leading LLMs and substantially reduces the gap between counselor evaluations and client-perceived alliance, achieving over 70% higher Pearson correlation with client ratings. Rationale-augmented supervision further improves predictive accuracy. CARE also produces high-quality, contextually grounded rationales, validated by both automatic and human evaluations. Applied to real-world Chinese online counseling sessions, CARE uncovers common alliance-building challenges, illustrates how interaction patterns shape alliance development, and provides actionable insights, demonstrating its potential as an AI-assisted tool for supporting mental health care. 

---
# Linear Reasoning vs. Proof by Cases: Obstacles for Large Language Models in FOL Problem Solving 

**Authors**: Yuliang Ji, Fuchen Shen, Jian Wu, Qiujie Xie, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20973)  

**Abstract**: To comprehensively evaluate the mathematical reasoning capabilities of Large Language Models (LLMs), researchers have introduced abundant mathematical reasoning datasets. However, most existing datasets primarily focus on linear reasoning, neglecting other parts such as proof by contradiction and proof by cases, which are crucial for investigating LLMs' reasoning abilities. To address this limitation, we first introduce a novel first-order logic (FOL) dataset named PC-FOL, annotated by professional mathematicians, focusing on case-based reasoning problems. All instances in this dataset are equipped with a manually written natural language proof, clearly distinguishing it from conventional linear reasoning datasets. Our experimental results over leading LLMs demonstrate a substantial performance gap between linear reasoning and case-based reasoning problems. To further investigate this phenomenon, we provide a theoretical analysis grounded in graphical model, which provides an explanation for the observed disparity between the two types of reasoning problems. We hope this work can reveal the core challenges in the field of automated natural language mathematical proof generation, paving the way for future research. 

---
# Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning 

**Authors**: Justin Lovelace, Christian Belardi, Sofian Zalouk, Adhitya Polavaram, Srivatsa Kundurthy, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2602.20528)  

**Abstract**: The Stop-Think-AutoRegress Language Diffusion Model (STAR-LDM) integrates latent diffusion planning with autoregressive generation. Unlike conventional autoregressive language models limited to token-by-token decisions, STAR-LDM incorporates a "thinking" phase that pauses generation to refine a semantic plan through diffusion before continuing. This enables global planning in continuous space prior to committing to discrete tokens. Evaluations show STAR-LDM significantly outperforms similar-sized models on language understanding benchmarks and achieves $>70\%$ win rates in LLM-as-judge comparisons for narrative coherence and commonsense reasoning. The architecture also allows straightforward control through lightweight classifiers, enabling fine-grained steering of attributes without model retraining while maintaining better fluency-control trade-offs than specialized approaches. 

---
# From Performance to Purpose: A Sociotechnical Taxonomy for Evaluating Large Language Model Utility 

**Authors**: Gavin Levinson, Keith Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2602.20513)  

**Abstract**: As large language models (LLMs) continue to improve at completing discrete tasks, they are being integrated into increasingly complex and diverse real-world systems. However, task-level success alone does not establish a model's fit for use in practice. In applied, high-stakes settings, LLM effectiveness is driven by a wider array of sociotechnical determinants that extend beyond conventional performance measures. Although a growing set of metrics capture many of these considerations, they are rarely organized in a way that supports consistent evaluation, leaving no unified taxonomy for assessing and comparing LLM utility across use cases. To address this gap, we introduce the Language Model Utility Taxonomy (LUX), a comprehensive framework that structures utility evaluation across four domains: performance, interaction, operations, and governance. Within each domain, LUX is organized hierarchically into thematically aligned dimensions and components, each grounded in metrics that enable quantitative comparison and alignment of model selection with intended use. In addition, an external dynamic web tool is provided to support exploration of the framework by connecting each component to a repository of relevant metrics (factors) for applied evaluation. 

---
# Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems 

**Authors**: Mukul Chhabra, Luigi Medrano, Arush Verma  

**Link**: [PDF](https://arxiv.org/pdf/2602.20379)  

**Abstract**: Enterprise Retrieval-Augmented Generation (RAG) assistants operate in multi-turn, case-based workflows such as technical support and IT operations, where evaluation must reflect operational constraints, structured identifiers (e.g., error codes, versions), and resolution workflows. Existing RAG evaluation frameworks are primarily designed for benchmark-style or single-turn settings and often fail to capture enterprise-specific failure modes such as case misidentification, workflow misalignment, and partial resolution across turns.
We present a case-aware LLM-as-a-Judge evaluation framework for enterprise multi-turn RAG systems. The framework evaluates each turn using eight operationally grounded metrics that separate retrieval quality, grounding fidelity, answer utility, precision integrity, and case/workflow alignment. A severity-aware scoring protocol reduces score inflation and improves diagnostic clarity across heterogeneous enterprise cases. The system uses deterministic prompting with strict JSON outputs, enabling scalable batch evaluation, regression testing, and production monitoring.
Through a comparative study of two instruction-tuned models across short and long workflows, we show that generic proxy metrics provide ambiguous signals, while the proposed framework exposes enterprise-critical tradeoffs that are actionable for system improvement. 

---
# SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards 

**Authors**: Dengjia Zhang, Xiaoou Liu, Lu Cheng, Yaqing Wang, Kenton Murray, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.21158)  

**Abstract**: Large language models (LLMs) are increasingly deployed as multi-step decision-making agents, where effective reward design is essential for guiding learning. Although recent work explores various forms of reward shaping and step-level credit assignment, a key signal remains largely overlooked: the intrinsic uncertainty of LLMs. Uncertainty reflects model confidence, reveals where exploration is needed, and offers valuable learning cues even in failed trajectories. We introduce SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards, a reinforcement learning framework that incorporates uncertainty directly into the reward design. SELAUR integrates entropy-, least-confidence-, and margin-based metrics into a combined token-level uncertainty estimate, providing dense confidence-aligned supervision, and employs a failure-aware reward reshaping mechanism that injects these uncertainty signals into step- and trajectory-level rewards to improve exploration efficiency and learning stability. Experiments on two benchmarks, ALFWorld and WebShop, show that our method consistently improves success rates over strong baselines. Ablation studies further demonstrate how uncertainty signals enhance exploration and robustness. 

---
# SpecMind: Cognitively Inspired, Interactive Multi-Turn Framework for Postcondition Inference 

**Authors**: Cuong Chi Le, Minh V.T Pham, Tung Vu Duy, Cuong Duc Van, Huy N. Phan, Hoang N. Phan, Tien N. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.20610)  

**Abstract**: Specifications are vital for ensuring program correctness, yet writing them manually remains challenging and time-intensive. Recent large language model (LLM)-based methods have shown successes in generating specifications such as postconditions, but existing single-pass prompting often yields inaccurate results. In this paper, we present SpecMind, a novel framework for postcondition generation that treats LLMs as interactive and exploratory reasoners rather than one-shot generators. SpecMind employs feedback-driven multi-turn prompting approaches, enabling the model to iteratively refine candidate postconditions by incorporating implicit and explicit correctness feedback, while autonomously deciding when to stop. This process fosters deeper code comprehension and improves alignment with true program behavior via exploratory attempts. Our empirical evaluation shows that SpecMind significantly outperforms state-of-the-art approaches in both accuracy and completeness of generated postconditions. 

---
