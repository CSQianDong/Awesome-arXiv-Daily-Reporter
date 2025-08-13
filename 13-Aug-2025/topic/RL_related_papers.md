# Reducing Cognitive Load in Multi-Agent Reinforcement Learning for Mathematical Problem Solving: Decoupling Reasoning and Code Generation 

**Authors**: Dayu Wang, Jiaye Yang, Weikang Li, Jiahui Liang, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08882)  

**Abstract**: Current tool-integrated mathematical reasoning systems often adopt a single-agent paradigm, where one large language model handles problem reasoning, code generation, and code execution in an integrated workflow. While this design eases coordination, we hypothesize that it imposes cognitive load interference, as the agent must interleave long-horizon reasoning with precise program synthesis. We validate this hypothesis through a controlled comparison between a reasoning-only agent and a reasoning-plus-code agent, finding that the latter produces significantly fewer correct reasoning paths despite having tool-calling capabilities. To address this, we propose a dual-agent hybrid framework: a Reasoning Agent performs stepwise problem decomposition, and a Code Agent handles code generation and execution. Training combines imitation learning and reinforcement learning: the Code Agent receives strong rewards for matching intermediate ground-truth programs and weaker rewards for valid execution, while the Reasoning Agent is optimized chiefly via final-answer accuracy using advantage estimation to credit intermediate steps. This decoupled role design reduces cognitive interference and promotes stable reasoning-coding coordination. 

---
# Compass-Thinker-7B Technical Report 

**Authors**: Anxiang Zeng, Haibo Zhang, Kaixiang Mo, Long Zhang, Shuman Liu, Yanhui Huang, Yawen Liu, Yuepeng Sheng, Yuwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08909)  

**Abstract**: Recent R1-Zero-like research further demonstrates that reasoning extension has given large language models (LLMs) unprecedented reasoning capabilities, and Reinforcement Learning is the core tech- nology to elicit its complex reasoning. However, conducting RL experiments directly on hyperscale models involves high computational costs and resource demands, posing significant risks. We pro- pose the Compass-Thinker-7B model, which aims to explore the potential of Reinforcement Learn- ing with less computational resources and costs, and provides insights for further research into RL recipes for larger models. Compass-Thinker-7B is trained from an open source model through a spe- cially designed Reinforcement Learning Pipeline. we curate a dataset of 30k verifiable mathematics problems for the Reinforcement Learning Pipeline. By configuring data and training settings with dif- ferent difficulty distributions for different stages, the potential of the model is gradually released and the training efficiency is improved. Extensive evaluations show that Compass-Thinker-7B possesses exceptional reasoning potential, and achieves superior performance on mathematics compared to the same-sized RL this http URL in the challenging AIME2024 evaluation, Compass-Thinker-7B achieves 40% accuracy. 

---
# Aryabhata: An exam-focused language model for JEE Math 

**Authors**: Ritvik Rastogi, Sachin Dharashivkar, Sandeep Varma  

**Link**: [PDF](https://arxiv.org/pdf/2508.08665)  

**Abstract**: We present $\textbf{Aryabhata 1.0}$, a compact 7B parameter math reasoning model optimized for the Indian academic exam, the Joint Entrance Examination (JEE). Despite rapid progress in large language models (LLMs), current models often remain unsuitable for educational use. Aryabhata 1.0 is built by merging strong open-weight reasoning models, followed by supervised fine-tuning (SFT) with curriculum learning on verified chain-of-thought (CoT) traces curated through best-of-$n$ rejection sampling. To further boost performance, we apply reinforcement learning with verifiable rewards (RLVR) using A2C objective with group-relative advantage estimation alongwith novel exploration strategies such as $\textit{Adaptive Group Resizing}$ and $\textit{Temperature Scaling}$. Evaluated on both in-distribution (JEE Main 2025) and out-of-distribution (MATH, GSM8K) benchmarks, Aryabhata outperforms existing models in accuracy and efficiency, while offering pedagogically useful step-by-step reasoning. We release Aryabhata as a foundation model to advance exam-centric, open-source small language models. This marks our first open release for community feedback ($\href{this https URL}{Aryabhata\ 1.0\ on\ Hugging\ Face}$); PW is actively training future models to further improve learning outcomes for students. 

---
# STELAR-VISION: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision 

**Authors**: Chen Li, Han Zhang, Zhantao Yang, Fangyi Chen, Zihan Wang, Anudeepsekhar Bolimera, Marios Savvides  

**Link**: [PDF](https://arxiv.org/pdf/2508.08688)  

**Abstract**: Vision-language models (VLMs) have made significant strides in reasoning, yet they often struggle with complex multimodal tasks and tend to generate overly verbose outputs. A key limitation is their reliance on chain-of-thought (CoT) reasoning, despite many tasks benefiting from alternative topologies like trees or graphs. To address this, we introduce STELAR-Vision, a training framework for topology-aware reasoning. At its core is TopoAug, a synthetic data pipeline that enriches training with diverse topological structures. Using supervised fine-tuning and reinforcement learning, we post-train Qwen2VL models with both accuracy and efficiency in mind. Additionally, we propose Frugal Learning, which reduces output length with minimal accuracy loss. On MATH-V and VLM-S2H, STELAR-Vision improves accuracy by 9.7% over its base model and surpasses the larger Qwen2VL-72B-Instruct by 7.3%. On five out-of-distribution benchmarks, it outperforms Phi-4-Multimodal-Instruct by up to 28.4% and LLaMA-3.2-11B-Vision-Instruct by up to 13.2%, demonstrating strong generalization. Compared to Chain-Only training, our approach achieves 4.3% higher overall accuracy on in-distribution datasets and consistently outperforms across all OOD benchmarks. We have released datasets, and code will be available. 

---
# Beyond Ordinal Preferences: Why Alignment Needs Cardinal Human Feedback 

**Authors**: Parker Whitfill, Stewy Slocum  

**Link**: [PDF](https://arxiv.org/pdf/2508.08486)  

**Abstract**: Alignment techniques for LLMs rely on optimizing preference-based objectives -- where these preferences are typically elicited as ordinal, binary choices between responses. Recent work has focused on improving label quality or mitigating particular biases, but we identify a more fundamental limitation: these methods collect the wrong kind of data. We prove an impossibility result: no algorithm relying solely on ordinal comparisons can systematically recover the most preferred model. Intuitively, ordinal data lacks the information needed to resolve tradeoffs -- e.g., fixing a factual error on one prompt versus improving style on another. We show that selecting the optimal model requires recovering preferences over \emph{models} (rather than just responses), which can only be identified given cardinal feedback about response quality. To address this, we collect and publicly release a dataset of 25,000 cardinal judgments using willingness-to-pay elicitations, a well-established tool from experimental economics. Empirically, we find that incorporating cardinal feedback into preference fine-tuning allows models to prioritize high-impact improvements and outperform ordinal-only methods on downstream benchmarks, such as Arena-Hard. 

---
# E3-Rewrite: Learning to Rewrite SQL for Executability, Equivalence,and Efficiency 

**Authors**: Dongjie Xu, Yue Cui, Weijie Shi, Qingzhi Ma, Hanghui Guo, Jiaming Li, Yao Zhao, Ruiyuan Zhang, Shimin Di, Jia Zhu, Kai Zheng, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09023)  

**Abstract**: SQL query rewriting aims to reformulate a query into a more efficient form while preserving equivalence. Most existing methods rely on predefined rewrite rules. However, such rule-based approaches face fundamental limitations: (1) fixed rule sets generalize poorly to novel query patterns and struggle with complex queries; (2) a wide range of effective rewriting strategies cannot be fully captured by declarative rules. To overcome these issues, we propose using large language models (LLMs) to generate rewrites. LLMs can capture complex strategies, such as evaluation reordering and CTE rewriting. Despite this potential, directly applying LLMs often results in suboptimal or non-equivalent rewrites due to a lack of execution awareness and semantic grounding. To address these challenges, We present E3-Rewrite, an LLM-based SQL rewriting framework that produces executable, equivalent, and efficient queries. It integrates two core components: a context construction module and a reinforcement learning framework. First, the context module leverages execution plans and retrieved demonstrations to build bottleneck-aware prompts that guide inference-time rewriting. Second, we design a reward function targeting executability, equivalence, and efficiency, evaluated via syntax checks, equivalence verification, and cost estimation. Third, to ensure stable multi-objective learning, we adopt a staged curriculum that first emphasizes executability and equivalence, then gradually incorporates efficiency. Extensive experiments show that E3-Rewrite achieves up to a 25.6\% reduction in query execution time compared to state-of-the-art methods across multiple SQL benchmarks. Moreover, it delivers up to 24.4\% more successful rewrites, expanding coverage to complex queries that previous systems failed to handle. 

---
# Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments 

**Authors**: Junjie Ye, Changhao Jiang, Zhengyin Du, Yufei Xu, Xuesong Yao, Zhiheng Xi, Xiaoran Fan, Qi Zhang, Xuanjing Huang, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08791)  

**Abstract**: Effective tool use is essential for large language models (LLMs) to interact meaningfully with their environment. However, progress is limited by the lack of efficient reinforcement learning (RL) frameworks specifically designed for tool use, due to challenges in constructing stable training environments and designing verifiable reward mechanisms. To address this, we propose an automated environment construction pipeline, incorporating scenario decomposition, document generation, function integration, complexity scaling, and localized deployment. This enables the creation of high-quality training environments that provide detailed and measurable feedback without relying on external tools. Additionally, we introduce a verifiable reward mechanism that evaluates both the precision of tool use and the completeness of task execution. When combined with trajectory data collected from the constructed environments, this mechanism integrates seamlessly with standard RL algorithms to facilitate feedback-driven model training. Experiments on LLMs of varying scales demonstrate that our approach significantly enhances the models' tool-use performance without degrading their general capabilities, regardless of inference modes or training algorithms. Our analysis suggests that these gains result from improved context understanding and reasoning, driven by updates to the lower-layer MLP parameters in models. 

---
# Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression 

**Authors**: Jadie Adams, Brian Hu, Emily Veenhuis, David Joy, Bharadwaj Ravichandran, Aaron Bray, Anthony Hoogs, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2508.08509)  

**Abstract**: Large language models (LLMs) are currently aligned using techniques such as reinforcement learning from human feedback (RLHF). However, these methods use scalar rewards that can only reflect user preferences on average. Pluralistic alignment instead seeks to capture diverse user preferences across a set of attributes, moving beyond just helpfulness and harmlessness. Toward this end, we propose a steerable pluralistic model based on few-shot comparative regression that can adapt to individual user preferences. Our approach leverages in-context learning and reasoning, grounded in a set of fine-grained attributes, to compare response options and make aligned choices. To evaluate our algorithm, we also propose two new steerable pluralistic benchmarks by adapting the Moral Integrity Corpus (MIC) and the HelpSteer2 datasets, demonstrating the applicability of our approach to value-aligned decision-making and reward modeling, respectively. Our few-shot comparative regression approach is interpretable and compatible with different attributes and LLMs, while outperforming multiple baseline and state-of-the-art methods. Our work provides new insights and research directions in pluralistic alignment, enabling a more fair and representative use of LLMs and advancing the state-of-the-art in ethical AI. 

---
# CPO: Addressing Reward Ambiguity in Role-playing Dialogue via Comparative Policy Optimization 

**Authors**: Xinge Ye, Rui Wang, Yuchuan Wu, Victor Ma, Feiteng Fang, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09074)  

**Abstract**: Reinforcement Learning Fine-Tuning (RLFT) has achieved notable success in tasks with objectively verifiable answers (e.g., code generation, mathematical reasoning), yet struggles with open-ended subjective tasks like role-playing dialogue. Traditional reward modeling approaches, which rely on independent sample-wise scoring, face dual challenges: subjective evaluation criteria and unstable reward this http URL by the insight that human evaluation inherently combines explicit criteria with implicit comparative judgments, we propose Comparative Policy Optimization (CPO). CPO redefines the reward evaluation paradigm by shifting from sample-wise scoring to comparative group-wise this http URL on the same principle, we introduce the CharacterArena evaluation framework, which comprises two stages:(1) Contextualized Multi-turn Role-playing Simulation, and (2) Trajectory-level Comparative Evaluation. By operationalizing subjective scoring via objective trajectory comparisons, CharacterArena minimizes contextual bias and enables more robust and fair performance evaluation. Empirical results on CharacterEval, CharacterBench, and CharacterArena confirm that CPO effectively mitigates reward ambiguity and leads to substantial improvements in dialogue quality. 

---
# Train Long, Think Short: Curriculum Learning for Efficient Reasoning 

**Authors**: Hasan Abed Al Kader Hammoud, Kumail Alhamoud, Abed Hammoud, Elie Bou-Zeid, Marzyeh Ghassemi, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.08940)  

**Abstract**: Recent work on enhancing the reasoning abilities of large language models (LLMs) has introduced explicit length control as a means of constraining computational cost while preserving accuracy. However, existing approaches rely on fixed-length training budgets, which do not take advantage of the natural progression from exploration to compression during learning. In this work, we propose a curriculum learning strategy for length-controlled reasoning using Group Relative Policy Optimization (GRPO). Our method starts with generous token budgets and gradually tightens them over training, encouraging models to first discover effective solution strategies and then distill them into more concise reasoning traces. We augment GRPO with a reward function that balances three signals: task correctness (via verifier feedback), length efficiency, and formatting adherence (via structural tags). Experiments on GSM8K, MATH500, SVAMP, College Math, and GSM+ demonstrate that curriculum-based training consistently outperforms fixed-budget baselines at the same final budget, achieving higher accuracy and significantly improved token efficiency. We further ablate the impact of reward weighting and decay schedule design, showing that progressive constraint serves as a powerful inductive bias for training efficient reasoning models. Our code and checkpoints are released at: this https URL. 

---
# Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery 

**Authors**: Jiatong Li, Weida Wang, Qinggang Zhang, Junxian Li, Di Zhang, Changmeng Zheng, Shufei Zhang, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08401)  

**Abstract**: Large language models (LLMs), especially Explicit Long Chain-of-Thought (CoT) reasoning models like DeepSeek-R1 and QWQ, have demonstrated powerful reasoning capabilities, achieving impressive performance in commonsense reasoning and mathematical inference. Despite their effectiveness, Long-CoT reasoning models are often criticized for their limited ability and low efficiency in knowledge-intensive domains such as molecule discovery. Success in this field requires a precise understanding of domain knowledge, including molecular structures and chemical principles, which is challenging due to the inherent complexity of molecular data and the scarcity of high-quality expert annotations. To bridge this gap, we introduce Mol-R1, a novel framework designed to improve explainability and reasoning performance of R1-like Explicit Long-CoT reasoning LLMs in text-based molecule generation. Our approach begins with a high-quality reasoning dataset curated through Prior Regulation via In-context Distillation (PRID), a dedicated distillation strategy to effectively generate paired reasoning traces guided by prior regulations. Building upon this, we introduce MoIA, Molecular Iterative Adaptation, a sophisticated training strategy that iteratively combines Supervised Fine-tuning (SFT) with Reinforced Policy Optimization (RPO), tailored to boost the reasoning performance of R1-like reasoning models for molecule discovery. Finally, we examine the performance of Mol-R1 in the text-based molecule reasoning generation task, showing superior performance against existing baselines. 

---
# Enhancing Small LLM Alignment through Margin-Based Objective Modifications under Resource Constraints 

**Authors**: Daren Yao, Jinsong Yuan, Ruike Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08466)  

**Abstract**: Small large language models (LLMs) often face difficulties in aligning output to human preferences, particularly when operating under severe performance gaps. In this work, we propose two lightweight DPO-based variants -- Adaptive Margin-Sigmoid Loss and APO-hinge-zero -- to better address underperformance scenarios by introducing margin-based objectives and selective update mechanisms.
Our APO-hinge-zero method, which combines hinge-induced hard-example mining with the chosen-focused optimization of APO-zero, achieves strong results. In AlpacaEval, APO-hinge-zero improves the win rate by +2.0 points and the length-controlled win rate by +1.4 points compared to the APO-zero baseline. In MT-Bench, our methods maintain competitive performance in diverse categories, particularly excelling in STEM and Humanities tasks.
These results demonstrate that simple modifications to preference-based objectives can significantly enhance small LLM alignment under resource constraints, offering a practical path toward more efficient deployment. 

---
