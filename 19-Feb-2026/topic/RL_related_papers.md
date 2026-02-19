# Learning to Learn from Language Feedback with Social Meta-Learning 

**Authors**: Jonathan Cook, Diego Antognini, Martin Klissarov, Claudiu Musat, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2602.16488)  

**Abstract**: Large language models (LLMs) often struggle to learn from corrective feedback within a conversational context. They are rarely proactive in soliciting this feedback, even when faced with ambiguity, which can make their dialogues feel static, one-sided, and lacking the adaptive qualities of human conversation. To address these limitations, we draw inspiration from social meta-learning (SML) in humans - the process of learning how to learn from others. We formulate SML as a finetuning methodology, training LLMs to solicit and learn from language feedback in simulated pedagogical dialogues, where static tasks are converted into interactive social learning problems. SML effectively teaches models to use conversation to solve problems they are unable to solve in a single turn. This capability generalises across domains; SML on math problems produces models that better use feedback to solve coding problems and vice versa. Furthermore, despite being trained only on fully-specified problems, these models are better able to solve underspecified tasks where critical information is revealed over multiple turns. When faced with this ambiguity, SML-trained models make fewer premature answer attempts and are more likely to ask for the information they need. This work presents a scalable approach to developing AI systems that effectively learn from language feedback. 

---
# Balancing Faithfulness and Performance in Reasoning via Multi-Listener Soft Execution 

**Authors**: Nithin Sivakumaran, Shoubin Yu, Hyunji Lee, Yue Zhang, Ali Payani, Mohit Bansal, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2602.16154)  

**Abstract**: Chain-of-thought (CoT) reasoning sometimes fails to faithfully reflect the true computation of a large language model (LLM), hampering its utility in explaining how LLMs arrive at their answers. Moreover, optimizing for faithfulness and interpretability in reasoning often degrades task performance. To address this tradeoff and improve CoT faithfulness, we propose Reasoning Execution by Multiple Listeners (REMUL), a multi-party reinforcement learning approach. REMUL builds on the hypothesis that reasoning traces which other parties can follow will be more faithful. A speaker model generates a reasoning trace, which is truncated and passed to a pool of listener models who "execute" the trace, continuing the trace to an answer. Speakers are rewarded for producing reasoning that is clear to listeners, with additional correctness regularization via masked supervised finetuning to counter the tradeoff between faithfulness and performance. On multiple reasoning benchmarks (BIG-Bench Extra Hard, MuSR, ZebraLogicBench, and FOLIO), REMUL consistently and substantially improves three measures of faithfulness -- hint attribution, early answering area over the curve (AOC), and mistake injection AOC -- while also improving accuracy. Our analysis finds that these gains are robust across training domains, translate to legibility gains, and are associated with shorter and more direct CoTs. 

---
# Quality-constrained Entropy Maximization Policy Optimization for LLM Diversity 

**Authors**: Haihui Pan, Yuzhong Hong, Shaoke Lv, Junwei Bao, Hongfei Jiang, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.15894)  

**Abstract**: Recent research indicates that while alignment methods significantly improve the quality of large language model(LLM) outputs, they simultaneously reduce the diversity of the models' output. Although some methods have been proposed to enhance LLM output diversity, they often come at the cost of reduced performance. In this work, we first theoretically demonstrate that the alignment task can be decomposed into two distributions: quality and diversity. To enhance the diversity of LLM outputs while ensuring quality, we propose the Quality-constrained Entropy Maximization Policy Optimization (QEMPO). QEMPO aims to maximize the output entropy of the policy while ensuring output quality. By adding different constraints to QEMPO, we obtain different policies. To optimize policies, we propose both online and offline training methods. Experiments validate that QEMPO achieves performance comparable to or even better than RLHF while improving output diversity. 

---
# Decoupling Strategy and Execution in Task-Focused Dialogue via Goal-Oriented Preference Optimization 

**Authors**: Jingyi Xu, Xingyu Ren, Zhiqiang You, Yumeng Zhang, Zhoupeng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2602.15854)  

**Abstract**: Large language models show potential in task-oriented dialogue systems, yet existing training methods often rely on token-level likelihood or preference optimization, which poorly align with long-horizon task success. To address this, we propose Goal-Oriented Preference Optimization (GOPO), a hierarchical reinforcement learning framework that decouples strategy planning from response generation via an Expert Agent and a Customer Service Agent. The Expert Agent optimizes multi-turn goal preferences at the dialogue-trajectory level, while the Customer Service Agent generates responses strictly aligned with the selected strategy. We evaluate GOPO on public benchmarks and e-commerce customer service datasets, and introduce Task-focused Sequential Engagement (TSE), a sequence-level metric derived from real e-commerce interaction data. On the Mgshop dataset, GOPO improves TSE by 7.7% and 10.3% over PPO and Memento, with consistent gains in sequence-level reward and generation quality. Furthermore, a 14B model trained with GOPO achieves 2.7% and 1.5% higher TSE than Qwen-235B and GPT-5.2, respectively. Ablation studies confirm the Expert Agent's critical role in long-horizon optimization. GOPO demonstrates consistent improvements across other datasets as well. This work establishes a new paradigm for task-oriented dialogue systems in commercial scenarios, with code and datasets to be made public. 

---
# Preference Optimization for Review Question Generation Improves Writing Quality 

**Authors**: Karun Sharma, Vidushee Vats, Shengzhi Li, Yuxiang Wang, Zhongtian Sun, Prayag Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2602.15849)  

**Abstract**: Peer review relies on substantive, evidence-based questions, yet existing LLM-based approaches often generate surface-level queries, drawing over 50\% of their question tokens from a paper's first page. To bridge this gap, we develop IntelliReward, a novel reward model built from a frozen autoregressive LLM with trainable multi-head transformers over the final 50 token states, which outperforms API-based SFT baselines in predicting expert-level human preferences. By applying Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) with IntelliReward, we train IntelliAsk, a question-generation model aligned with human standards of effort, evidence, and grounding. We find consistent improvements on reasoning and writing benchmarks, suggesting reviewer-question quality correlates with broader capabilities. Compared to the Qwen3-32B base model, IntelliAsk shows measurable gains across diverse benchmarks, specifically improving performance on reasoning tasks like MuSR (68.3 vs 64.7 Acc) and complex writing evaluations such as WritingBench (8.31 vs 8.07). We release our implementation, expert preference annotations, and the IntelliReward model to provide an automatic evaluation benchmark for grounding, effort, and evidence in LLM-generated review questions. 

---
# Improving Interactive In-Context Learning from Natural Language Feedback 

**Authors**: Martin Klissarov, Jonathan Cook, Diego Antognini, Hao Sun, Jingling Li, Natasha Jaques, Claudiu Musat, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2602.16066)  

**Abstract**: Adapting one's thought process based on corrective feedback is an essential ability in human learning, particularly in collaborative settings. In contrast, the current large language model training paradigm relies heavily on modeling vast, static corpora. While effective for knowledge acquisition, it overlooks the interactive feedback loops essential for models to adapt dynamically to their context. In this work, we propose a framework that treats this interactive in-context learning ability not as an emergent property, but as a distinct, trainable skill. We introduce a scalable method that transforms single-turn verifiable tasks into multi-turn didactic interactions driven by information asymmetry. We first show that current flagship models struggle to integrate corrective feedback on hard reasoning tasks. We then demonstrate that models trained with our approach dramatically improve the ability to interactively learn from language feedback. More specifically, the multi-turn performance of a smaller model nearly reaches that of a model an order of magnitude larger. We also observe robust out-of-distribution generalization: interactive training on math problems transfers to diverse domains like coding, puzzles and maze navigation. Our qualitative analysis suggests that this improvement is due to an enhanced in-context plasticity. Finally, we show that this paradigm offers a unified path to self-improvement. By training the model to predict the teacher's critiques, effectively modeling the feedback environment, we convert this external signal into an internal capability, allowing the model to self-correct even without a teacher. 

---
# HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents 

**Authors**: Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2602.16165)  

**Abstract**: Training LLMs as interactive agents for multi-turn decision-making remains challenging, particularly in long-horizon tasks with sparse and delayed rewards, where agents must execute extended sequences of actions before receiving meaningful feedback. Most existing reinforcement learning (RL) approaches model LLM agents as flat policies operating at a single time scale, selecting one action at each turn. In sparse-reward settings, such flat policies must propagate credit across the entire trajectory without explicit temporal abstraction, which often leads to unstable optimization and inefficient credit assignment.
We propose HiPER, a novel Hierarchical Plan-Execute RL framework that explicitly separates high-level planning from low-level execution. HiPER factorizes the policy into a high-level planner that proposes subgoals and a low-level executor that carries them out over multiple action steps. To align optimization with this structure, we introduce a key technique called hierarchical advantage estimation (HAE), which carefully assigns credit at both the planning and execution levels. By aggregating returns over the execution of each subgoal and coordinating updates across the two levels, HAE provides an unbiased gradient estimator and provably reduces variance compared to flat generalized advantage estimation.
Empirically, HiPER achieves state-of-the-art performance on challenging interactive benchmarks, reaching 97.4\% success on ALFWorld and 83.3\% on WebShop with Qwen2.5-7B-Instruct (+6.6\% and +8.3\% over the best prior method), with especially large gains on long-horizon tasks requiring multiple dependent subtasks. These results highlight the importance of explicit hierarchical decomposition for scalable RL training of multi-turn LLM agents. 

---
