# Structured Thinking Matters: Improving LLMs Generalization in Causal Inference Tasks 

**Authors**: Wentao Sun, Joao Paulo Nogueira, Alonso Silva  

**Link**: [PDF](https://arxiv.org/pdf/2505.18034)  

**Abstract**: Despite remarkable advances in the field, LLMs remain unreliable in distinguishing causation from correlation. Recent results from the Corr2Cause dataset benchmark reveal that state-of-the-art LLMs -- such as GPT-4 (F1 score: 29.08) -- only marginally outperform random baselines (Random Uniform, F1 score: 20.38), indicating limited capacity of generalization. To tackle this limitation, we propose a novel structured approach: rather than directly answering causal queries, we provide the model with the capability to structure its thinking by guiding the model to build a structured knowledge graph, systematically encoding the provided correlational premises, to answer the causal queries. This intermediate representation significantly enhances the model's causal capabilities. Experiments on the test subset of the Corr2Cause dataset benchmark with Qwen3-32B model (reasoning model) show substantial gains over standard direct prompting methods, improving F1 scores from 32.71 to 48.26 (over 47.5% relative increase), along with notable improvements in precision and recall. These results underscore the effectiveness of providing the model with the capability to structure its thinking and highlight its promising potential for broader generalization across diverse causal inference tasks. 

---
# ProgRM: Build Better GUI Agents with Progress Rewards 

**Authors**: Danyang Zhang, Situo Zhang, Ziyue Yang, Zichen Zhu, Zihan Zhao, Ruisheng Cao, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18121)  

**Abstract**: LLM-based (Large Language Model) GUI (Graphical User Interface) agents can potentially reshape our daily lives significantly. However, current LLM-based GUI agents suffer from the scarcity of high-quality training data owing to the difficulties of trajectory collection and reward annotation. Existing works have been exploring LLMs to collect trajectories for imitation learning or to offer reward signals for online RL training. However, the Outcome Reward Model (ORM) used in existing works cannot provide finegrained feedback and can over-penalize the valuable steps in finally failed trajectories. To this end, we propose Progress Reward Model (ProgRM) to provide dense informative intermediate rewards by predicting a task completion progress for each step in online training. To handle the challenge of progress reward label annotation, we further design an efficient LCS-based (Longest Common Subsequence) self-annotation algorithm to discover the key steps in trajectories and assign progress labels accordingly. ProgRM is evaluated with extensive experiments and analyses. Actors trained with ProgRM outperform leading proprietary LLMs and ORM-trained actors, illustrating the effectiveness of ProgRM. The codes for experiments will be made publicly available upon acceptance. 

---
# Evaluation Faking: Unveiling Observer Effects in Safety Evaluation of Frontier AI Systems 

**Authors**: Yihe Fan, Wenqi Zhang, Xudong Pan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17815)  

**Abstract**: As foundation models grow increasingly more intelligent, reliable and trustworthy safety evaluation becomes more indispensable than ever. However, an important question arises: Whether and how an advanced AI system would perceive the situation of being evaluated, and lead to the broken integrity of the evaluation process? During standard safety tests on a mainstream large reasoning model, we unexpectedly observe that the model without any contextual cues would occasionally recognize it is being evaluated and hence behave more safety-aligned. This motivates us to conduct a systematic study on the phenomenon of evaluation faking, i.e., an AI system autonomously alters its behavior upon recognizing the presence of an evaluation context and thereby influencing the evaluation results. Through extensive experiments on a diverse set of foundation models with mainstream safety benchmarks, we reach the main finding termed the observer effects for AI: When the AI system under evaluation is more advanced in reasoning and situational awareness, the evaluation faking behavior becomes more ubiquitous, which reflects in the following aspects: 1) Reasoning models recognize evaluation 16% more often than non-reasoning models. 2) Scaling foundation models (32B to 671B) increases faking by over 30% in some cases, while smaller models show negligible faking. 3) AI with basic memory is 2.3x more likely to recognize evaluation and scores 19% higher on safety tests (vs. no memory). To measure this, we devised a chain-of-thought monitoring technique to detect faking intent and uncover internal signals correlated with such behavior, offering insights for future mitigation studies. 

---
# CIKT: A Collaborative and Iterative Knowledge Tracing Framework with Large Language Models 

**Authors**: Runze Li, Siyu Wu, Jun Wang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17705)  

**Abstract**: Knowledge Tracing (KT) aims to model a student's learning state over time and predict their future performance. However, traditional KT methods often face challenges in explainability, scalability, and effective modeling of complex knowledge dependencies. While Large Language Models (LLMs) present new avenues for KT, their direct application often struggles with generating structured, explainable student representations and lacks mechanisms for continuous, task-specific refinement. To address these gaps, we propose Collaborative Iterative Knowledge Tracing (CIKT), a framework that harnesses LLMs to enhance both prediction accuracy and explainability. CIKT employs a dual-component architecture: an Analyst generates dynamic, explainable user profiles from student historical responses, and a Predictor utilizes these profiles to forecast future performance. The core of CIKT is a synergistic optimization loop. In this loop, the Analyst is iteratively refined based on the predictive accuracy of the Predictor, which conditions on the generated profiles, and the Predictor is subsequently retrained using these enhanced profiles. Evaluated on multiple educational datasets, CIKT demonstrates significant improvements in prediction accuracy, offers enhanced explainability through its dynamically updated user profiles, and exhibits improved scalability. Our work presents a robust and explainable solution for advancing knowledge tracing systems, effectively bridging the gap between predictive performance and model transparency. 

---
# Automating Safety Enhancement for LLM-based Agents with Synthetic Risk Scenarios 

**Authors**: Xueyang Zhou, Weidong Wang, Lin Lu, Jiawen Shi, Guiyao Tie, Yongtian Xu, Lixing Chen, Pan Zhou, Neil Zhenqiang Gong, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.17735)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly deployed in real-world applications such as "digital assistants, autonomous customer service, and decision-support systems", where their ability to "interact in multi-turn, tool-augmented environments" makes them indispensable. However, ensuring the safety of these agents remains a significant challenge due to the diverse and complex risks arising from dynamic user interactions, external tool usage, and the potential for unintended harmful behaviors. To address this critical issue, we propose AutoSafe, the first framework that systematically enhances agent safety through fully automated synthetic data generation. Concretely, 1) we introduce an open and extensible threat model, OTS, which formalizes how unsafe behaviors emerge from the interplay of user instructions, interaction contexts, and agent actions. This enables precise modeling of safety risks across diverse scenarios. 2) we develop a fully automated data generation pipeline that simulates unsafe user behaviors, applies self-reflective reasoning to generate safe responses, and constructs a large-scale, diverse, and high-quality safety training dataset-eliminating the need for hazardous real-world data collection. To evaluate the effectiveness of our framework, we design comprehensive experiments on both synthetic and real-world safety benchmarks. Results demonstrate that AutoSafe boosts safety scores by 45% on average and achieves a 28.91% improvement on real-world tasks, validating the generalization ability of our learned safety strategies. These results highlight the practical advancement and scalability of AutoSafe in building safer LLM-based agents for real-world deployment. We have released the project page at this https URL. 

---
# Decoupled Visual Interpretation and Linguistic Reasoning for Math Problem Solving 

**Authors**: Zixian Guo, Ming Liu, Zhilong Ji, Jinfeng Bai, Lei Zhang, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17609)  

**Abstract**: Current large vision-language models (LVLMs) typically employ a connector module to link visual features with text embeddings of large language models (LLMs) and use end-to-end training to achieve multi-modal understanding in a unified process. Well alignment needs high-quality pre-training data and a carefully designed training process. Current LVLMs face challenges when addressing complex vision-language reasoning tasks, with their reasoning capabilities notably lagging behind those of LLMs. This paper proposes a paradigm shift: instead of training end-to-end vision-language reasoning models, we advocate for developing a decoupled reasoning framework based on existing visual interpretation specialists and text-based reasoning LLMs. Our approach leverages (1) a dedicated vision-language model to transform the visual content of images into textual descriptions and (2) an LLM to perform reasoning according to the visual-derived text and the original question. This method presents a cost-efficient solution for multi-modal model development by optimizing existing models to work collaboratively, avoiding end-to-end development of vision-language models from scratch. By transforming images into language model-compatible text representations, it facilitates future low-cost and flexible upgrades to upcoming powerful LLMs. We introduce an outcome-rewarded joint-tuning strategy to optimize the cooperation between the visual interpretation and linguistic reasoning model. Evaluation results on vision-language benchmarks demonstrate that the decoupled reasoning framework outperforms recent LVLMs. Our approach yields particularly significant performance gains on visually intensive geometric mathematics problems. The code is available: this https URL. 

---
# Controlled Agentic Planning & Reasoning for Mechanism Synthesis 

**Authors**: João Pedro Gandarela, Thiago Rios, Stefan Menzel, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17607)  

**Abstract**: This work presents a dual-agent Large Language Model (LLM)-based reasoning method for mechanism synthesis, capable of reasoning at both linguistic and symbolic levels to generate geometrical and dynamic outcomes. The model consists of a composition of well-defined functions that, starting from a natural language specification, references abstract properties through supporting equations, generates and parametrizes simulation code, and elicits feedback anchor points using symbolic regression and distance functions. This process closes an actionable refinement loop at the linguistic and symbolic layers. The approach is shown to be both effective and convergent in the context of planar mechanisms. Additionally, we introduce MSynth, a novel benchmark for planar mechanism synthesis, and perform a comprehensive analysis of the impact of the model components. We further demonstrate that symbolic regression prompts unlock mechanistic insights only when applied to sufficiently large architectures. 

---
# USTBench: Benchmarking and Dissecting Spatiotemporal Reasoning of LLMs as Urban Agents 

**Authors**: Siqi Lai, Yansong Ning, Zirui Yuan, Zhixi Chen, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17572)  

**Abstract**: Large language models (LLMs) have shown emerging potential in spatiotemporal reasoning, making them promising candidates for building urban agents that support diverse urban downstream applications. Despite these benefits, existing studies primarily focus on evaluating urban LLM agent on outcome-level metrics (e.g., prediction accuracy, traffic efficiency), offering limited insight into their underlying reasoning processes. As a result, the strengths and limitations of urban LLM agents in spatiotemporal reasoning remain poorly understood. To this end, we introduce USTBench, the first benchmark to evaluate LLMs' spatiotemporal reasoning abilities as urban agents across four decomposed dimensions: spatiotemporal understanding, forecasting, planning, and reflection with feedback. Specifically, USTBench supports five diverse urban decision-making and four spatiotemporal prediction tasks, all running within our constructed interactive city environment UAgentEnv. The benchmark includes 62,466 structured QA pairs for process-level evaluation and standardized end-to-end task assessments, enabling fine-grained diagnostics and broad task-level comparison across diverse urban scenarios. Through extensive evaluation of thirteen leading LLMs, we reveal that although LLMs show promising potential across various urban downstream tasks, they still struggle in long-horizon planning and reflective adaptation in dynamic urban contexts. Notably, recent advanced reasoning models (e.g., DeepSeek-R1) trained on general logic or mathematical problems do not consistently outperform non-reasoning LLMs. This discrepancy highlights the need for domain-specialized adaptation methods to enhance urban spatiotemporal reasoning. Overall, USTBench provides a foundation to build more adaptive and effective LLM-based urban agents and broad smart city applications. 

---
# Integrating Counterfactual Simulations with Language Models for Explaining Multi-Agent Behaviour 

**Authors**: Bálint Gyevnár, Christopher G. Lucas, Stefano V. Albrecht, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17801)  

**Abstract**: Autonomous multi-agent systems (MAS) are useful for automating complex tasks but raise trust concerns due to risks like miscoordination and goal misalignment. Explainability is vital for trust calibration, but explainable reinforcement learning for MAS faces challenges in state/action space complexity, stakeholder needs, and evaluation. Using the counterfactual theory of causation and LLMs' summarisation capabilities, we propose Agentic eXplanations via Interrogative Simulation (AXIS). AXIS generates intelligible causal explanations for pre-trained multi-agent policies by having an LLM interrogate an environment simulator using queries like 'whatif' and 'remove' to observe and synthesise counterfactual information over multiple rounds. We evaluate AXIS on autonomous driving across 10 scenarios for 5 LLMs with a novel evaluation methodology combining subjective preference, correctness, and goal/action prediction metrics, and an external LLM as evaluator. Compared to baselines, AXIS improves perceived explanation correctness by at least 7.7% across all models and goal prediction accuracy by 23% for 4 models, with improved or comparable action prediction accuracy, achieving the highest scores overall. 

---
# GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs 

**Authors**: Shixian Luo, Zezhou Zhu, Yu Yuan, Yuncheng Yang, Lianlei Shan, Yong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17653)  

**Abstract**: Geometric spatial reasoning forms the foundation of many applications in artificial intelligence, yet the ability of large language models (LLMs) to operate over geometric spatial information expressed in procedural code remains underexplored. In this paper, we address this gap by formalizing the Program-to-Geometry task, which challenges models to translate programmatic drawing code into accurate and abstract geometric reasoning. To evaluate this capability, we present GeoGramBench, a benchmark of 500 carefully refined problems organized by a tailored three-level taxonomy that considers geometric complexity rather than traditional mathematical reasoning complexity. Our comprehensive evaluation of 17 frontier LLMs reveals consistent and pronounced deficiencies: even the most advanced models achieve less than 50% accuracy at the highest abstraction level. These results highlight the unique challenges posed by program-driven spatial reasoning and establish GeoGramBench as a valuable resource for advancing research in symbolic-to-spatial geometric reasoning. Project page: this https URL. 

---
# T2I-Eval-R1: Reinforcement Learning-Driven Reasoning for Interpretable Text-to-Image Evaluation 

**Authors**: Zi-Ao Ma, Tian Lan, Rong-Cheng Tu, Shu-Hang Liu, Heyan Huang, Zhijing Wu, Chen Xu, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17897)  

**Abstract**: The rapid progress in diffusion-based text-to-image (T2I) generation has created an urgent need for interpretable automatic evaluation methods that can assess the quality of generated images, therefore reducing the human annotation burden. To reduce the prohibitive cost of relying on commercial models for large-scale evaluation, and to improve the reasoning capabilities of open-source models, recent research has explored supervised fine-tuning (SFT) of multimodal large language models (MLLMs) as dedicated T2I evaluators. However, SFT approaches typically rely on high-quality critique datasets, which are either generated by proprietary LLMs-with potential issues of bias and inconsistency-or annotated by humans at high cost, limiting their scalability and generalization. To address these limitations, we propose T2I-Eval-R1, a novel reinforcement learning framework that trains open-source MLLMs using only coarse-grained quality scores, thereby avoiding the need for annotating high-quality interpretable evaluation rationale. Our approach integrates Group Relative Policy Optimization (GRPO) into the instruction-tuning process, enabling models to generate both scalar scores and interpretable reasoning chains with only easy accessible annotated judgment scores or preferences. Furthermore, we introduce a continuous reward formulation that encourages score diversity and provides stable optimization signals, leading to more robust and discriminative evaluation behavior. Experimental results on three established T2I meta-evaluation benchmarks demonstrate that T2I-Eval-R1 achieves significantly higher alignment with human assessments and offers more accurate interpretable score rationales compared to strong baseline methods. 

---
# From Reasoning to Generalization: Knowledge-Augmented LLMs for ARC Benchmark 

**Authors**: Chao Lei, Nir Lipovetzky, Krista A. Ehinger, Yanchuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17482)  

**Abstract**: Recent reasoning-oriented LLMs have demonstrated strong performance on challenging tasks such as mathematics and science examinations. However, core cognitive faculties of human intelligence, such as abstract reasoning and generalization, remain underexplored. To address this, we evaluate recent reasoning-oriented LLMs on the Abstraction and Reasoning Corpus (ARC) benchmark, which explicitly demands both faculties. We formulate ARC as a program synthesis task and propose nine candidate solvers. Experimental results show that repeated-sampling planning-aided code generation (RSPC) achieves the highest test accuracy and demonstrates consistent generalization across most LLMs. To further improve performance, we introduce an ARC solver, Knowledge Augmentation for Abstract Reasoning (KAAR), which encodes core knowledge priors within an ontology that classifies priors into three hierarchical levels based on their dependencies. KAAR progressively expands LLM reasoning capacity by gradually augmenting priors at each level, and invokes RSPC to generate candidate solutions after each augmentation stage. This stage-wise reasoning reduces interference from irrelevant priors and improves LLM performance. Empirical results show that KAAR maintains strong generalization and consistently outperforms non-augmented RSPC across all evaluated LLMs, achieving around 5% absolute gains and up to 64.52% relative improvement. Despite these achievements, ARC remains a challenging benchmark for reasoning-oriented LLMs, highlighting future avenues of progress in LLMs. 

---
# Longer Context, Deeper Thinking: Uncovering the Role of Long-Context Ability in Reasoning 

**Authors**: Wang Yang, Zirui Liu, Hongye Jin, Qingyu Yin, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.17315)  

**Abstract**: Recent language models exhibit strong reasoning capabilities, yet the influence of long-context capacity on reasoning remains underexplored. In this work, we hypothesize that current limitations in reasoning stem, in part, from insufficient long-context capacity, motivated by empirical observations such as (1) higher context window length often leads to stronger reasoning performance, and (2) failed reasoning cases resemble failed long-context cases. To test this hypothesis, we examine whether enhancing a model's long-context ability before Supervised Fine-Tuning (SFT) leads to improved reasoning performance. Specifically, we compared models with identical architectures and fine-tuning data but varying levels of long-context capacity. Our results reveal a consistent trend: models with stronger long-context capacity achieve significantly higher accuracy on reasoning benchmarks after SFT. Notably, these gains persist even on tasks with short input lengths, indicating that long-context training offers generalizable benefits for reasoning performance. These findings suggest that long-context modeling is not just essential for processing lengthy inputs, but also serves as a critical foundation for reasoning. We advocate for treating long-context capacity as a first-class objective in the design of future language models. 

---
# Misaligning Reasoning with Answers -- A Framework for Assessing LLM CoT Robustness 

**Authors**: Enyi Jiang, Changming Xu, Nischay Singh, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.17406)  

**Abstract**: LLMs' decision-making process is opaque, prompting the need for explanation techniques like Chain-of-Thought. To investigate the relationship between answer and reasoning, we design a novel evaluation framework, MATCHA. In domains like education and healthcare, reasoning is key for model trustworthiness. MATCHA reveals that LLMs under input perturbations can give inconsistent or nonsensical reasoning. Additionally, we use LLM judges to assess reasoning robustness across models. Our results show that LLMs exhibit greater vulnerability to input perturbations for multi-step and commonsense tasks than compared to logical tasks. Also, we show non-trivial transfer rates of our successful examples to black-box models. Our evaluation framework helps to better understand LLM reasoning mechanisms and guides future models toward more robust and reasoning-driven architectures, enforcing answer-reasoning consistency. 

---
# AdaReasoner: Adaptive Reasoning Enables More Flexible Thinking 

**Authors**: Xiangqi Wang, Yue Huang, Yanbo Wang, Xiaonan Luo, Kehan Guo, Yujun Zhou, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17312)  

**Abstract**: LLMs often need effective configurations, like temperature and reasoning steps, to handle tasks requiring sophisticated reasoning and problem-solving, ranging from joke generation to mathematical reasoning. Existing prompting approaches usually adopt general-purpose, fixed configurations that work 'well enough' across tasks but seldom achieve task-specific optimality. To address this gap, we introduce AdaReasoner, an LLM-agnostic plugin designed for any LLM to automate adaptive reasoning configurations for tasks requiring different types of thinking. AdaReasoner is trained using a reinforcement learning (RL) framework, combining a factorized action space with a targeted exploration strategy, along with a pretrained reward model to optimize the policy model for reasoning configurations with only a few-shot guide. AdaReasoner is backed by theoretical guarantees and experiments of fast convergence and a sublinear policy gap. Across six different LLMs and a variety of reasoning tasks, it consistently outperforms standard baselines, preserves out-of-distribution robustness, and yield gains on knowledge-intensive tasks through tailored prompts. 

---
# Effective Reinforcement Learning for Reasoning in Language Models 

**Authors**: Lianghuan Huang, Shuo Li, Sagnik Anupam, Insup Lee, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2505.17218)  

**Abstract**: Reinforcement learning (RL) has emerged as a promising strategy for improving the reasoning capabilities of language models (LMs) in domains such as mathematics and coding. However, most modern RL algorithms were designed to target robotics applications, which differ significantly from LM reasoning. We analyze RL algorithm design decisions for LM reasoning, for both accuracy and computational efficiency, focusing on relatively small models due to computational constraints. Our findings are: (i) on-policy RL significantly outperforms supervised fine-tuning (SFT), (ii) PPO-based off-policy updates increase accuracy instead of reduce variance, and (iii) removing KL divergence can lead to more concise generations and higher accuracy. Furthermore, we find that a key bottleneck to computational efficiency is that the optimal batch sizes for inference and backpropagation are different. We propose a novel algorithm, DASH, that performs preemptive sampling (i.e., sample a large batch and accumulate gradient updates in small increments), and gradient filtering (i.e., drop samples with small advantage estimates). We show that DASH reduces training time by 83% compared to a standard implementation of GRPO without sacrificing accuracy. Our findings provide valuable insights on designing effective RL algorithms for LM reasoning. 

---
# Reasoning Model is Stubborn: Diagnosing Instruction Overriding in Reasoning Models 

**Authors**: Doohyuk Jang, Yoonjeon Kim, Chanjae Park, Hyun Ryu, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17225)  

**Abstract**: Large language models have demonstrated remarkable proficiency in long and complex reasoning tasks. However, they frequently exhibit a problematic reliance on familiar reasoning patterns, a phenomenon we term \textit{reasoning rigidity}. Despite explicit instructions from users, these models often override clearly stated conditions and default to habitual reasoning trajectories, leading to incorrect conclusions. This behavior presents significant challenges, particularly in domains such as mathematics and logic puzzle, where precise adherence to specified constraints is critical. To systematically investigate reasoning rigidity, a behavior largely unexplored in prior work, we introduce a expert-curated diagnostic set, \dataset{}. Our dataset includes specially modified variants of existing mathematical benchmarks, namely AIME and MATH500, as well as well-known puzzles deliberately redesigned to require deviation from familiar reasoning strategies. Using this dataset, we identify recurring contamination patterns that occur when models default to ingrained reasoning. Specifically, we categorize this contamination into three distinctive modes: (i) Interpretation Overload, (ii) Input Distrust, and (iii) Partial Instruction Attention, each causing models to ignore or distort provided instructions. We publicly release our diagnostic set to facilitate future research on mitigating reasoning rigidity in language models. 

---
# Graph-Linguistic Fusion: Using Language Models for Wikidata Vandalism Detection 

**Authors**: Mykola Trokhymovych, Lydia Pintscher, Ricardo Baeza-Yates, Diego Saez-Trumper  

**Link**: [PDF](https://arxiv.org/pdf/2505.18136)  

**Abstract**: We introduce a next-generation vandalism detection system for Wikidata, one of the largest open-source structured knowledge bases on the Web. Wikidata is highly complex: its items incorporate an ever-expanding universe of factual triples and multilingual texts. While edits can alter both structured and textual content, our approach converts all edits into a single space using a method we call Graph2Text. This allows for evaluating all content changes for potential vandalism using a single multilingual language model. This unified approach improves coverage and simplifies maintenance. Experiments demonstrate that our solution outperforms the current production system. Additionally, we are releasing the code under an open license along with a large dataset of various human-generated knowledge alterations, enabling further research. 

---
# Lost in the Haystack: Smaller Needles are More Difficult for LLMs to Find 

**Authors**: Owen Bianchi, Mathew J. Koretsky, Maya Willey, Chelsea X. Alvarado, Tanay Nayak, Adi Asija, Nicole Kuznetsov, Mike A. Nalls, Faraz Faghri, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2505.18148)  

**Abstract**: Large language models (LLMs) face significant challenges with needle-in-a-haystack tasks, where relevant information ("the needle") must be drawn from a large pool of irrelevant context ("the haystack"). Previous studies have highlighted positional bias and distractor quantity as critical factors affecting model performance, yet the influence of gold context size has received little attention. We address this gap by systematically studying how variations in gold context length impact LLM performance on long-context question answering tasks. Our experiments reveal that LLM performance drops sharply when the gold context is shorter, i.e., smaller gold contexts consistently degrade model performance and amplify positional sensitivity, posing a major challenge for agentic systems that must integrate scattered, fine-grained information of varying lengths. This pattern holds across three diverse domains (general knowledge, biomedical reasoning, and mathematical reasoning) and seven state-of-the-art LLMs of various sizes and architectures. Our work provides clear insights to guide the design of robust, context-aware LLM-driven systems. 

---
# Bidirectional Knowledge Distillation for Enhancing Sequential Recommendation with Large Language Models 

**Authors**: Jiongran Wu, Jiahao Liu, Dongsheng Li, Guangping Zhang, Mingzhe Han, Hansu Gu, Peng Zhang, Li Shang, Tun Lu, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18120)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance in understanding and generating semantic patterns, making them promising candidates for sequential recommendation tasks. However, when combined with conventional recommendation models (CRMs), LLMs often face challenges related to high inference costs and static knowledge transfer methods. In this paper, we propose a novel mutual distillation framework, LLMD4Rec, that fosters dynamic and bidirectional knowledge exchange between LLM-centric and CRM-based recommendation systems. Unlike traditional unidirectional distillation methods, LLMD4Rec enables iterative optimization by alternately refining both models, enhancing the semantic understanding of CRMs and enriching LLMs with collaborative signals from user-item interactions. By leveraging sample-wise adaptive weighting and aligning output distributions, our approach eliminates the need for additional parameters while ensuring effective knowledge transfer. Extensive experiments on real-world datasets demonstrate that LLMD4Rec significantly improves recommendation accuracy across multiple benchmarks without increasing inference costs. This method provides a scalable and efficient solution for combining the strengths of both LLMs and CRMs in sequential recommendation systems. 

---
# How Can I Publish My LLM Benchmark Without Giving the True Answers Away? 

**Authors**: Takashi Ishida, Thanawat Lodkaew, Ikko Yamane  

**Link**: [PDF](https://arxiv.org/pdf/2505.18102)  

**Abstract**: Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies. 

---
# Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL 

**Authors**: Joey Hong, Anca Dragan, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2505.18098)  

**Abstract**: Large language models (LLMs) excel in tasks like question answering and dialogue, but complex tasks requiring interaction, such as negotiation and persuasion, require additional long-horizon reasoning and planning. Reinforcement learning (RL) fine-tuning can enable such planning in principle, but suffers from drawbacks that hinder scalability. In particular, multi-turn RL training incurs high memory and computational costs, which are exacerbated when training LLMs as policies. Furthermore, the largest LLMs do not expose the APIs necessary to be trained in such manner. As a result, modern methods to improve the reasoning of LLMs rely on sophisticated prompting mechanisms rather than RL fine-tuning. To remedy this, we propose a novel approach that uses goal-conditioned value functions to guide the reasoning of LLM agents, that scales even to large API-based models. These value functions predict how a task will unfold given an action, allowing the LLM agent to evaluate multiple possible outcomes, both positive and negative, to plan effectively. In addition, these value functions are trained over reasoning steps rather than full actions, to be a concise and light-weight module that facilitates decision-making in multi-turn interactions. We validate our method on tasks requiring interaction, including tool use, social deduction, and dialogue, demonstrating superior performance over both RL fine-tuning and prompting methods while maintaining efficiency and scalability. 

---
# Where You Go is Who You Are: Behavioral Theory-Guided LLMs for Inverse Reinforcement Learning 

**Authors**: Yuran Sun, Susu Xu, Chenguang Wang, Xilei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17249)  

**Abstract**: Big trajectory data hold great promise for human mobility analysis, but their utility is often constrained by the absence of critical traveler attributes, particularly sociodemographic information. While prior studies have explored predicting such attributes from mobility patterns, they often overlooked underlying cognitive mechanisms and exhibited low predictive accuracy. This study introduces SILIC, short for Sociodemographic Inference with LLM-guided Inverse Reinforcement Learning (IRL) and Cognitive Chain Reasoning (CCR), a theoretically grounded framework that leverages LLMs to infer sociodemographic attributes from observed mobility patterns by capturing latent behavioral intentions and reasoning through psychological constructs. Particularly, our approach explicitly follows the Theory of Planned Behavior (TPB), a foundational behavioral framework in transportation research, to model individuals' latent cognitive processes underlying travel decision-making. The LLMs further provide heuristic guidance to improve IRL reward function initialization and update by addressing its ill-posedness and optimization challenges arising from the vast and unstructured reward space. Evaluated in the 2017 Puget Sound Regional Council Household Travel Survey, our method substantially outperforms state-of-the-art baselines and shows great promise for enriching big trajectory data to support more behaviorally grounded applications in transportation planning and beyond. 

---
# Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals 

**Authors**: Jia-Nan Li, Jian Guan, Wei Wu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18071)  

**Abstract**: Large language models (LLMs) have demonstrated significant success in complex reasoning tasks such as math and coding. In contrast to these tasks where deductive reasoning predominates, inductive reasoning\textemdash the ability to derive general rules from incomplete evidence, remains underexplored. This paper investigates extended inductive reasoning in LLMs through the lens of personalized preference inference, a critical challenge in LLM alignment where current approaches struggle to capture diverse user preferences. The task demands strong inductive reasoning capabilities as user preferences are typically embedded implicitly across various interaction forms, requiring models to synthesize consistent preference patterns from scattered signals. We propose \textsc{AlignXplore}, a model that leverages extended reasoning chains to enable systematic preference inference from behavioral signals in users' interaction histories. We develop \textsc{AlignXplore} by combining cold-start training based on synthetic data with subsequent online reinforcement learning. Through extensive experiments, we demonstrate that \textsc{AlignXplore} achieves substantial improvements over the backbone model by an average of 11.05\% on in-domain and out-of-domain benchmarks, while maintaining strong generalization ability across different input formats and downstream models. Further analyses establish best practices for preference inference learning through systematic comparison of reward modeling strategies, while revealing the emergence of human-like inductive reasoning patterns during training. 

---
# LLM assisted web application functional requirements generation: A case study of four popular LLMs over a Mess Management System 

**Authors**: Rashmi Gupta, Aditya K Gupta, Aarav Jain, Avinash C Pandey, Atul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.18019)  

**Abstract**: Like any other discipline, Large Language Models (LLMs) have significantly impacted software engineering by helping developers generate the required artifacts across various phases of software development. This paper presents a case study comparing the performance of popular LLMs GPT, Claude, Gemini, and DeepSeek in generating functional specifications that include use cases, business rules, and collaborative workflows for a web application, the Mess Management System. The study evaluated the quality of LLM generated use cases, business rules, and collaborative workflows in terms of their syntactic and semantic correctness, consistency, non ambiguity, and completeness compared to the reference specifications against the zero-shot prompted problem statement. Our results suggested that all four LLMs can specify syntactically and semantically correct, mostly non-ambiguous artifacts. Still, they may be inconsistent at times and may differ significantly in the completeness of the generated specification. Claude and Gemini generated all the reference use cases, with Claude achieving the most complete but somewhat redundant use case specifications. Similar results were obtained for specifying workflows. However, all four LLMs struggled to generate relevant Business Rules, with DeepSeek generating the most reference rules but with less completeness. Overall, Claude generated more complete specification artifacts, while Gemini was more precise in the specifications it generated. 

---
# Training with Pseudo-Code for Instruction Following 

**Authors**: Prince Kumar, Rudra Murthy, Riyaz Bhat, Danish Contractor  

**Link**: [PDF](https://arxiv.org/pdf/2505.18011)  

**Abstract**: Despite the rapid progress in the capabilities of Large Language Models (LLMs), they continue to have difficulty following relatively simple, unambiguous instructions, especially when compositions are involved. In this paper, we take inspiration from recent work that suggests that models may follow instructions better when they are expressed in pseudo-code. However, writing pseudo-code programs can be tedious and using few-shot demonstrations to craft code representations for use in inference can be unnatural for non-expert users of LLMs. To overcome these limitations, we propose fine-tuning LLMs with instruction-tuning data that additionally includes instructions re-expressed in pseudo-code along with the final response. We evaluate models trained using our method on $11$ publicly available benchmarks comprising of tasks related to instruction-following, mathematics, and common-sense reasoning. We conduct rigorous experiments with $5$ different models and find that not only do models follow instructions better when trained with pseudo-code, they also retain their capabilities on the other tasks related to mathematical and common sense reasoning. Specifically, we observe a relative gain of $3$--$19$% on instruction-following benchmark, and an average gain of upto 14% across all tasks. 

---
# DEL-ToM: Inference-Time Scaling for Theory-of-Mind Reasoning via Dynamic Epistemic Logic 

**Authors**: Yuheng Wu, Jianwen Xie, Denghui Zhang, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17348)  

**Abstract**: Theory-of-Mind (ToM) tasks pose a unique challenge for small language models (SLMs) with limited scale, which often lack the capacity to perform deep social reasoning. In this work, we propose DEL-ToM, a framework that improves ToM reasoning through inference-time scaling rather than architectural changes. Our approach decomposes ToM tasks into a sequence of belief updates grounded in Dynamic Epistemic Logic (DEL), enabling structured and transparent reasoning. We train a verifier, called the Process Belief Model (PBM), to score each belief update step using labels generated automatically via a DEL simulator. During inference, candidate belief traces generated by a language model are evaluated by the PBM, and the highest-scoring trace is selected. This allows SLMs to emulate more deliberate reasoning by allocating additional compute at test time. Experiments across multiple model scales and benchmarks show that DEL-ToM consistently improves performance, demonstrating that verifiable belief supervision can significantly enhance ToM abilities of SLMs without retraining. 

---
# Are Large Language Models Reliable AI Scientists? Assessing Reverse-Engineering of Black-Box Systems 

**Authors**: Jiayi Geng, Howard Chen, Dilip Arumugam, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.17968)  

**Abstract**: Using AI to create autonomous researchers has the potential to accelerate scientific discovery. A prerequisite for this vision is understanding how well an AI model can identify the underlying structure of a black-box system from its behavior. In this paper, we explore how well a large language model (LLM) learns to identify a black-box function from passively observed versus actively collected data. We investigate the reverse-engineering capabilities of LLMs across three distinct types of black-box systems, each chosen to represent different problem domains where future autonomous AI researchers may have considerable impact: Program, Formal Language, and Math Equation. Through extensive experiments, we show that LLMs fail to extract information from observations, reaching a performance plateau that falls short of the ideal of Bayesian inference. However, we demonstrate that prompting LLMs to not only observe but also intervene -- actively querying the black-box with specific inputs to observe the resulting output -- improves performance by allowing LLMs to test edge cases and refine their beliefs. By providing the intervention data from one LLM to another, we show that this improvement is partly a result of engaging in the process of generating effective interventions, paralleling results in the literature on human learning. Further analysis reveals that engaging in intervention can help LLMs escape from two common failure modes: overcomplication, where the LLM falsely assumes prior knowledge about the black-box, and overlooking, where the LLM fails to incorporate observations. These insights provide practical guidance for helping LLMs more effectively reverse-engineer black-box systems, supporting their use in making new discoveries. 

---
# Mutarjim: Advancing Bidirectional Arabic-English Translation with a Small Language Model 

**Authors**: Khalil Hennara, Muhammad Hreden, Mohamed Motaism Hamed, Zeina Aldallal, Sara Chrouf, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17894)  

**Abstract**: We introduce Mutarjim, a compact yet powerful language model for bidirectional Arabic-English translation. While large-scale LLMs have shown impressive progress in natural language processing tasks, including machine translation, smaller models. Leveraging this insight, we developed Mutarjim based on Kuwain-1.5B , a language model tailored for both Arabic and English. Despite its modest size, Mutarjim outperforms much larger models on several established benchmarks, achieved through an optimized two-phase training approach and a carefully curated, high-quality training corpus.. Experimental results show that Mutarjim rivals models up to 20 times larger while significantly reducing computational costs and training requirements. We also introduce Tarjama-25, a new benchmark designed to overcome limitations in existing Arabic-English benchmarking datasets, such as domain narrowness, short sentence lengths, and English-source bias. Tarjama-25 comprises 5,000 expert-reviewed sentence pairs and spans a wide range of domains, offering a more comprehensive and balanced evaluation framework. Notably, Mutarjim achieves state-of-the-art performance on the English-to-Arabic task in Tarjama-25, surpassing even significantly larger and proprietary models like GPT-4o mini. We publicly release Tarjama-25 to support future research and advance the evaluation of Arabic-English translation systems. 

---
# ADLGen: Synthesizing Symbolic, Event-Triggered Sensor Sequences for Human Activity Modeling 

**Authors**: Weihang You, Hanqi Jiang, Zishuai Liu, Zihang Xie, Tianming Liu, Jin Lu, Fei Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17987)  

**Abstract**: Real world collection of Activities of Daily Living data is challenging due to privacy concerns, costly deployment and labeling, and the inherent sparsity and imbalance of human behavior. We present ADLGen, a generative framework specifically designed to synthesize realistic, event triggered, and symbolic sensor sequences for ambient assistive environments. ADLGen integrates a decoder only Transformer with sign based symbolic temporal encoding, and a context and layout aware sampling mechanism to guide generation toward semantically rich and physically plausible sensor event sequences. To enhance semantic fidelity and correct structural inconsistencies, we further incorporate a large language model into an automatic generate evaluate refine loop, which verifies logical, behavioral, and temporal coherence and generates correction rules without manual intervention or environment specific tuning. Through comprehensive experiments with novel evaluation metrics, ADLGen is shown to outperform baseline generators in statistical fidelity, semantic richness, and downstream activity recognition, offering a scalable and privacy-preserving solution for ADL data synthesis. 

---
# Beyond Distillation: Pushing the Limits of Medical LLM Reasoning with Minimalist Rule-Based RL 

**Authors**: Che Liu, Haozhe Wang, Jiazhen Pan, Zhongwei Wan, Yong Dai, Fangzhen Lin, Wenjia Bai, Daniel Rueckert, Rossella Arcucci  

**Link**: [PDF](https://arxiv.org/pdf/2505.17952)  

**Abstract**: Improving performance on complex tasks and enabling interpretable decision making in large language models (LLMs), especially for clinical applications, requires effective reasoning. Yet this remains challenging without supervised fine-tuning (SFT) on costly chain-of-thought (CoT) data distilled from closed-source models (e.g., GPT-4o). In this work, we present AlphaMed, the first medical LLM to show that reasoning capability can emerge purely through reinforcement learning (RL), using minimalist rule-based rewards on public multiple-choice QA datasets, without relying on SFT or distilled CoT data. AlphaMed achieves state-of-the-art results on six medical QA benchmarks, outperforming models trained with conventional SFT+RL pipelines. On challenging benchmarks (e.g., MedXpert), AlphaMed even surpasses larger or closed-source models such as DeepSeek-V3-671B and Claude-3.5-Sonnet. To understand the factors behind this success, we conduct a comprehensive data-centric analysis guided by three questions: (i) Can minimalist rule-based RL incentivize reasoning without distilled CoT supervision? (ii) How do dataset quantity and diversity impact reasoning? (iii) How does question difficulty shape the emergence and generalization of reasoning? Our findings show that dataset informativeness is a key driver of reasoning performance, and that minimalist RL on informative, multiple-choice QA data is effective at inducing reasoning without CoT supervision. We also observe divergent trends across benchmarks, underscoring limitations in current evaluation and the need for more challenging, reasoning-oriented medical QA benchmarks. 

---
# Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning 

**Authors**: Michael Hassid, Gabriel Synnaeve, Yossi Adi, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17813)  

**Abstract**: Reasoning large language models (LLMs) heavily rely on scaling test-time compute to perform complex reasoning tasks by generating extensive "thinking" chains. While demonstrating impressive results, this approach incurs significant computational costs and inference time. In this work, we challenge the assumption that long thinking chains results in better reasoning capabilities. We first demonstrate that shorter reasoning chains within individual questions are significantly more likely to yield correct answers - up to 34.5% more accurate than the longest chain sampled for the same question. Based on these results, we suggest short-m@k, a novel reasoning LLM inference method. Our method executes k independent generations in parallel and halts computation once the first m thinking processes are done. The final answer is chosen using majority voting among these m chains. Basic short-1@k demonstrates similar or even superior performance over standard majority voting in low-compute settings - using up to 40% fewer thinking tokens. short-3@k, while slightly less efficient than short-1@k, consistently surpasses majority voting across all compute budgets, while still being substantially faster (up to 33% wall time reduction). Inspired by our results, we finetune an LLM using short, long, and randomly selected reasoning chains. We then observe that training on the shorter ones leads to better performance. Our findings suggest rethinking current methods of test-time compute in reasoning LLMs, emphasizing that longer "thinking" does not necessarily translate to improved performance and can, counter-intuitively, lead to degraded results. 

---
# But what is your honest answer? Aiding LLM-judges with honest alternatives using steering vectors 

**Authors**: Leon Eshuijs, Archie Chaudhury, Alan McBeth, Ethan Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17760)  

**Abstract**: Recent safety evaluations of Large Language Models (LLMs) show that many models exhibit dishonest behavior, such as sycophancy. However, most honesty benchmarks focus exclusively on factual knowledge or explicitly harmful behavior and rely on external judges, which are often unable to detect less obvious forms of dishonesty. In this work, we introduce a new framework, Judge Using Safety-Steered Alternatives (JUSSA), which utilizes steering vectors trained on a single sample to elicit more honest responses from models, helping LLM-judges in the detection of dishonest behavior. To test our framework, we introduce a new manipulation dataset with prompts specifically designed to elicit deceptive responses. We find that JUSSA enables LLM judges to better differentiate between dishonest and benign responses, and helps them identify subtle instances of manipulative behavior. 

---
# Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM 

**Authors**: Donghwan Chi, Hyomin Kim, Yoonjin Oh, Yongjin Kim, Donghoon Lee, Daejin Jo, Jongmin Kim, Junyeob Baek, Sungjin Ahn, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17726)  

**Abstract**: Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images. 

---
# DialogXpert: Driving Intelligent and Emotion-Aware Conversations through Online Value-Based Reinforcement Learning with LLM Priors 

**Authors**: Tazeek Bin Abdur Rakib, Ambuj Mehrish, Lay-Ki Soon, Wern Han Lim, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2505.17795)  

**Abstract**: Large-language-model (LLM) agents excel at reactive dialogue but struggle with proactive, goal-driven interactions due to myopic decoding and costly planning. We introduce DialogXpert, which leverages a frozen LLM to propose a small, high-quality set of candidate actions per turn and employs a compact Q-network over fixed BERT embeddings trained via temporal-difference learning to select optimal moves within this reduced space. By tracking the user's emotions, DialogXpert tailors each decision to advance the task while nurturing a genuine, empathetic connection. Across negotiation, emotional support, and tutoring benchmarks, DialogXpert drives conversations to under $3$ turns with success rates exceeding 94\% and, with a larger LLM prior, pushes success above 97\% while markedly improving negotiation outcomes. This framework delivers real-time, strategic, and emotionally intelligent dialogue planning at scale. Code available at this https URL 

---
# Tuning Language Models for Robust Prediction of Diverse User Behaviors 

**Authors**: Fanjin Meng, Jingtao Ding, Jiahui Gong, Chen Yang, Hong Chen, Zuojian Wang, Haisheng Lu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17682)  

**Abstract**: Predicting user behavior is essential for intelligent assistant services, yet deep learning models often struggle to capture long-tailed behaviors. Large language models (LLMs), with their pretraining on vast corpora containing rich behavioral knowledge, offer promise. However, existing fine-tuning approaches tend to overfit to frequent ``anchor'' behaviors, reducing their ability to predict less common ``tail'' behaviors. In this paper, we introduce BehaviorLM, a progressive fine-tuning approach that addresses this issue. In the first stage, LLMs are fine-tuned on anchor behaviors while preserving general behavioral knowledge. In the second stage, fine-tuning uses a balanced subset of all behaviors based on sample difficulty to improve tail behavior predictions without sacrificing anchor performance. Experimental results on two real-world datasets demonstrate that BehaviorLM robustly predicts both anchor and tail behaviors and effectively leverages LLM behavioral knowledge to master tail behavior prediction with few-shot examples. 

---
# EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications 

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17654)  

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at this https URL. 

---
# HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning 

**Authors**: Chuhao Zhou, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17645)  

**Abstract**: Embodied agents operating in smart homes must understand human behavior through diverse sensory inputs and communicate via natural language. While Vision-Language Models (VLMs) have enabled impressive language-grounded perception, their reliance on visual data limits robustness in real-world scenarios with occlusions, poor lighting, or privacy constraints. In this paper, we introduce HoloLLM, a Multimodal Large Language Model (MLLM) that integrates uncommon but powerful sensing modalities, such as LiDAR, infrared, mmWave radar, and WiFi, to enable seamless human perception and reasoning across heterogeneous environments. We address two key challenges: (1) the scarcity of aligned modality-text data for rare sensors, and (2) the heterogeneity of their physical signal representations. To overcome these, we design a Universal Modality-Injection Projector (UMIP) that enhances pre-aligned modality embeddings with fine-grained, text-aligned features from tailored encoders via coarse-to-fine cross-attention without introducing significant alignment overhead. We further introduce a human-VLM collaborative data curation pipeline to generate paired textual annotations for sensing datasets. Extensive experiments on two newly constructed benchmarks show that HoloLLM significantly outperforms existing MLLMs, improving language-grounded human sensing accuracy by up to 30%. This work establishes a new foundation for real-world, language-informed multisensory embodied intelligence. 

---
# ReqBrain: Task-Specific Instruction Tuning of LLMs for AI-Assisted Requirements Generation 

**Authors**: Mohammad Kasra Habib, Daniel Graziotin, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2505.17632)  

**Abstract**: Requirements elicitation and specification remains a labor-intensive, manual process prone to inconsistencies and gaps, presenting a significant challenge in modern software engineering. Emerging studies underscore the potential of employing large language models (LLMs) for automated requirements generation to support requirements elicitation and specification; however, it remains unclear how to implement this effectively. In this work, we introduce ReqBrain, an Al-assisted tool that employs a fine-tuned LLM to generate authentic and adequate software requirements. Software engineers can engage with ReqBrain through chat-based sessions to automatically generate software requirements and categorize them by type. We curated a high-quality dataset of ISO 29148-compliant requirements and fine-tuned five 7B-parameter LLMs to determine the most effective base model for ReqBrain. The top-performing model, Zephyr-7b-beta, achieved 89.30\% Fl using the BERT score and a FRUGAL score of 91.20 in generating authentic and adequate requirements. Human evaluations further confirmed ReqBrain's effectiveness in generating requirements. Our findings suggest that generative Al, when fine-tuned, has the potential to improve requirements elicitation and specification, paving the way for future extensions into areas such as defect identification, test case generation, and agile user story creation. 

---
# Runaway is Ashamed, But Helpful: On the Early-Exit Behavior of Large Language Model-based Agents in Embodied Environments 

**Authors**: Qingyu Lu, Liang Ding, Siyi Cao, Xuebo Liu, Kanjian Zhang, Jinxia Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17616)  

**Abstract**: Agents powered by large language models (LLMs) have demonstrated strong planning and decision-making capabilities in complex embodied environments. However, such agents often suffer from inefficiencies in multi-turn interactions, frequently trapped in repetitive loops or issuing ineffective commands, leading to redundant computational overhead. Instead of relying solely on learning from trajectories, we take a first step toward exploring the early-exit behavior for LLM-based agents. We propose two complementary approaches: 1. an $\textbf{intrinsic}$ method that injects exit instructions during generation, and 2. an $\textbf{extrinsic}$ method that verifies task completion to determine when to halt an agent's trial. To evaluate early-exit mechanisms, we introduce two metrics: one measures the reduction of $\textbf{redundant steps}$ as a positive effect, and the other evaluates $\textbf{progress degradation}$ as a negative effect. Experiments with 4 different LLMs across 5 embodied environments show significant efficiency improvements, with only minor drops in agent performance. We also validate a practical strategy where a stronger agent assists after an early-exit agent, achieving better performance with the same total steps. We will release our code to support further research. 

---
# Seek-CAD: A Self-refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek 

**Authors**: Xueyang Li, Jiahao Li, Yu Song, Yunzhong Lou, Xiangdong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17702)  

**Abstract**: The advent of Computer-Aided Design (CAD) generative modeling will significantly transform the design of industrial products. The recent research endeavor has extended into the realm of Large Language Models (LLMs). In contrast to fine-tuning methods, training-free approaches typically utilize the advanced closed-source LLMs, thereby offering enhanced flexibility and efficiency in the development of AI agents for generating CAD parametric models. However, the substantial cost and limitations of local deployment of the top-tier closed-source LLMs pose challenges in practical applications. The Seek-CAD is the pioneer exploration of locally deployed open-source inference LLM DeepSeek-R1 for CAD parametric model generation with a training-free methodology. This study is the first investigation to incorporate both visual and Chain-of-Thought (CoT) feedback within the self-refinement mechanism for generating CAD models. Specifically, the initial generated parametric CAD model is rendered into a sequence of step-wise perspective images, which are subsequently processed by a Vision Language Model (VLM) alongside the corresponding CoTs derived from DeepSeek-R1 to assess the CAD model generation. Then, the feedback is utilized by DeepSeek-R1 to refine the initial generated model for the next round of generation. Moreover, we present an innovative 3D CAD model dataset structured around the SSR (Sketch, Sketch-based feature, and Refinements) triple design paradigm. This dataset encompasses a wide range of CAD commands, thereby aligning effectively with industrial application requirements and proving suitable for the generation of LLMs. Extensive experiments validate the effectiveness of Seek-CAD under various metrics. 

---
# Distilling LLM Agent into Small Models with Retrieval and Code Tools 

**Authors**: Minki Kang, Jongwon Jeong, Seanie Lee, Jaewoong Cho, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17612)  

**Abstract**: Large language models (LLMs) excel at complex reasoning tasks but remain computationally expensive, limiting their practical deployment. To address this, recent works have focused on distilling reasoning capabilities into smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher LLMs. However, this approach struggles in scenarios requiring rare factual knowledge or precise computation, where sLMs often hallucinate due to limited capability. In this work, we propose Agent Distillation, a framework for transferring not only reasoning capability but full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools. We improve agent distillation along two complementary axes: (1) we introduce a prompting method called first-thought prefix to enhance the quality of teacher-generated trajectories; and (2) we propose a self-consistent action generation for improving test-time robustness of small agents. We evaluate our method on eight reasoning tasks across factual and mathematical domains, covering both in-domain and out-of-domain generalization. Our results show that sLMs as small as 0.5B, 1.5B, 3B parameters can achieve performance competitive with next-tier larger 1.5B, 3B, 7B models fine-tuned using CoT distillation, demonstrating the potential of agent distillation for building practical, tool-using small agents. Our code is available at this https URL. 

---
# On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning 

**Authors**: Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Yang Yuan, Quanquan Gu, Andrew C Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17508)  

**Abstract**: Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). Despite the widespread use of Kullback-Leibler (KL) regularization in policy gradient algorithms to stabilize training, the systematic exploration of how different KL divergence formulations can be estimated and integrated into surrogate loss functions for online reinforcement learning (RL) presents a nuanced and systematically explorable design space. In this paper, we propose regularized policy gradient (RPG), a systematic framework for deriving and analyzing KL-regularized policy gradient methods in the online RL setting. We derive policy gradients and corresponding surrogate loss functions for objectives regularized by both forward and reverse KL divergences, considering both normalized and unnormalized policy distributions. Furthermore, we present derivations for fully differentiable loss functions as well as REINFORCE-style gradient estimators, accommodating diverse algorithmic needs. We conduct extensive experiments on RL for LLM reasoning using these methods, showing improved or competitive results in terms of training stability and performance compared to strong baselines such as GRPO, REINFORCE++, and DAPO. The code is available at this https URL. 

---
# The Discovery Engine: A Framework for AI-Driven Synthesis and Navigation of Scientific Knowledge Landscapes 

**Authors**: Vladimir Baulin, Austin Cook, Daniel Friedman, Janna Lumiruusu, Andrew Pashea, Shagor Rahman, Benedikt Waldeck  

**Link**: [PDF](https://arxiv.org/pdf/2505.17500)  

**Abstract**: The prevailing model for disseminating scientific knowledge relies on individual publications dispersed across numerous journals and archives. This legacy system is ill suited to the recent exponential proliferation of publications, contributing to insurmountable information overload, issues surrounding reproducibility and retractions. We introduce the Discovery Engine, a framework to address these challenges by transforming an array of disconnected literature into a unified, computationally tractable representation of a scientific domain. Central to our approach is the LLM-driven distillation of publications into structured "knowledge artifacts," instances of a universal conceptual schema, complete with verifiable links to source evidence. These artifacts are then encoded into a high-dimensional Conceptual Tensor. This tensor serves as the primary, compressed representation of the synthesized field, where its labeled modes index scientific components (concepts, methods, parameters, relations) and its entries quantify their interdependencies. The Discovery Engine allows dynamic "unrolling" of this tensor into human-interpretable views, such as explicit knowledge graphs (the CNM graph) or semantic vector spaces, for targeted exploration. Crucially, AI agents operate directly on the graph using abstract mathematical and learned operations to navigate the knowledge landscape, identify non-obvious connections, pinpoint gaps, and assist researchers in generating novel knowledge artifacts (hypotheses, designs). By converting literature into a structured tensor and enabling agent-based interaction with this compact representation, the Discovery Engine offers a new paradigm for AI-augmented scientific inquiry and accelerated discovery. 

---
# CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training 

**Authors**: Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.17589)  

**Abstract**: In our prior works, we introduced a scalable streaming speech synthesis model, CosyVoice 2, which integrates a large language model (LLM) and a chunk-aware flow matching (FM) model, and achieves low-latency bi-streaming speech synthesis and human-parity quality. Despite these advancements, CosyVoice 2 exhibits limitations in language coverage, domain diversity, data volume, text formats, and post-training techniques. In this paper, we present CosyVoice 3, an improved model designed for zero-shot multilingual speech synthesis in the wild, surpassing its predecessor in content consistency, speaker similarity, and prosody naturalness. Key features of CosyVoice 3 include: 1) A novel speech tokenizer to improve prosody naturalness, developed via supervised multi-task training, including automatic speech recognition, speech emotion recognition, language identification, audio event detection, and speaker analysis. 2) A new differentiable reward model for post-training applicable not only to CosyVoice 3 but also to other LLM-based speech synthesis models. 3) Dataset Size Scaling: Training data is expanded from ten thousand hours to one million hours, encompassing 9 languages and 18 Chinese dialects across various domains and text formats. 4) Model Size Scaling: Model parameters are increased from 0.5 billion to 1.5 billion, resulting in enhanced performance on our multilingual benchmark due to the larger model capacity. These advancements contribute significantly to the progress of speech synthesis in the wild. We encourage readers to listen to the demo at this https URL. 

---
# keepitsimple at SemEval-2025 Task 3: LLM-Uncertainty based Approach for Multilingual Hallucination Span Detection 

**Authors**: Saketh Reddy Vemula, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2505.17485)  

**Abstract**: Identification of hallucination spans in black-box language model generated text is essential for applications in the real world. A recent attempt at this direction is SemEval-2025 Task 3, Mu-SHROOM-a Multilingual Shared Task on Hallucinations and Related Observable Over-generation Errors. In this work, we present our solution to this problem, which capitalizes on the variability of stochastically-sampled responses in order to identify hallucinated spans. Our hypothesis is that if a language model is certain of a fact, its sampled responses will be uniform, while hallucinated facts will yield different and conflicting results. We measure this divergence through entropy-based analysis, allowing for accurate identification of hallucinated segments. Our method is not dependent on additional training and hence is cost-effective and adaptable. In addition, we conduct extensive hyperparameter tuning and perform error analysis, giving us crucial insights into model behavior. 

---
# SLearnLLM: A Self-Learning Framework for Efficient Domain-Specific Adaptation of Large Language Models 

**Authors**: Xiang Liu, Zhaoxiang Liu, Peng Wang, Kohou Wang, Huan Hu, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.17470)  

**Abstract**: When using supervised fine-tuning (SFT) to adapt large language models (LLMs) to specific domains, a significant challenge arises: should we use the entire SFT dataset for fine-tuning? Common practice often involves fine-tuning directly on the entire dataset due to limited information on the LLM's past training data. However, if the SFT dataset largely overlaps with the model's existing knowledge, the performance gains are minimal, leading to wasted computational resources. Identifying the unknown knowledge within the SFT dataset and using it to fine-tune the model could substantially improve the training efficiency. To address this challenge, we propose a self-learning framework for LLMs inspired by human learning pattern. This framework takes a fine-tuning (SFT) dataset in a specific domain as input. First, the LLMs answer the questions in the SFT dataset. The LLMs then objectively grade the responses and filter out the incorrectly answered QA pairs. Finally, we fine-tune the LLMs based on this filtered QA set. Experimental results in the fields of agriculture and medicine demonstrate that our method substantially reduces training time while achieving comparable improvements to those attained with full dataset fine-tuning. By concentrating on the unknown knowledge within the SFT dataset, our approach enhances the efficiency of fine-tuning LLMs. 

---
# UniTTS: An end-to-end TTS system without decoupling of acoustic and semantic information 

**Authors**: Rui Wang, Qianguo Sun, Tianrong Chen, Zhiyun Zeng, Junlong Wu, Jiaxing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17426)  

**Abstract**: The emergence of multi-codebook neutral audio codecs such as Residual Vector Quantization (RVQ) and Group Vector Quantization (GVQ) has significantly advanced Large-Language-Model (LLM) based Text-to-Speech (TTS) systems. These codecs are crucial in separating semantic and acoustic information while efficiently harnessing semantic priors. However, since semantic and acoustic information cannot be fully aligned, a significant drawback of these methods when applied to LLM-based TTS is that large language models may have limited access to comprehensive audio information. To address this limitation, we propose DistilCodec and UniTTS, which collectively offer the following advantages: 1) This method can distill a multi-codebook audio codec into a single-codebook audio codec with 32,768 codes while achieving a near 100\% utilization. 2) As DistilCodec does not employ a semantic alignment scheme, a large amount of high-quality unlabeled audio (such as audiobooks with sound effects, songs, etc.) can be incorporated during training, further expanding data diversity and broadening its applicability. 3) Leveraging the comprehensive audio information modeling of DistilCodec, we integrated three key tasks into UniTTS's pre-training framework: audio modality autoregression, text modality autoregression, and speech-text cross-modal autoregression. This allows UniTTS to accept interleaved text and speech/audio prompts while substantially preserving LLM's text capabilities. 4) UniTTS employs a three-stage training process: Pre-Training, Supervised Fine-Tuning (SFT), and Alignment. Source code and model checkpoints are publicly available at this https URL and this https URL. 

---
# A Fully Generative Motivational Interviewing Counsellor Chatbot for Moving Smokers Towards the Decision to Quit 

**Authors**: Zafarullah Mahmood, Soliman Ali, Jiading Zhu, Mohamed Abdelwahab, Michelle Yu Collins, Sihan Chen, Yi Cheng Zhao, Jodi Wolff, Osnat Melamed, Nadia Minian, Marta Maslej, Carolynne Cooper, Matt Ratto, Peter Selby, Jonathan Rose  

**Link**: [PDF](https://arxiv.org/pdf/2505.17362)  

**Abstract**: The conversational capabilities of Large Language Models (LLMs) suggest that they may be able to perform as automated talk therapists. It is crucial to know if these systems would be effective and adhere to known standards. We present a counsellor chatbot that focuses on motivating tobacco smokers to quit smoking. It uses a state-of-the-art LLM and a widely applied therapeutic approach called Motivational Interviewing (MI), and was evolved in collaboration with clinician-scientists with expertise in MI. We also describe and validate an automated assessment of both the chatbot's adherence to MI and client responses. The chatbot was tested on 106 participants, and their confidence that they could succeed in quitting smoking was measured before the conversation and one week later. Participants' confidence increased by an average of 1.7 on a 0-10 scale. The automated assessment of the chatbot showed adherence to MI standards in 98% of utterances, higher than human counsellors. The chatbot scored well on a participant-reported metric of perceived empathy but lower than typical human counsellors. Furthermore, participants' language indicated a good level of motivation to change, a key goal in MI. These results suggest that the automation of talk therapy with a modern LLM has promise. 

---
# ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs 

**Authors**: Landon Butler, Abhineet Agarwal, Justin Singh Kang, Yigit Efe Erginbas, Bin Yu, Kannan Ramchandran  

**Link**: [PDF](https://arxiv.org/pdf/2505.17495)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance by capturing complex interactions between input features. To identify these interactions, most existing approaches require enumerating all possible combinations of features up to a given order, causing them to scale poorly with the number of inputs $n$. Recently, Kang et al. (2025) proposed SPEX, an information-theoretic approach that uses interaction sparsity to scale to $n \approx 10^3$ features. SPEX greatly improves upon prior methods but requires tens of thousands of model inferences, which can be prohibitive for large models. In this paper, we observe that LLM feature interactions are often hierarchical -- higher-order interactions are accompanied by their lower-order subsets -- which enables more efficient discovery. To exploit this hierarchy, we propose ProxySPEX, an interaction attribution algorithm that first fits gradient boosted trees to masked LLM outputs and then extracts the important interactions. Experiments across four challenging high-dimensional datasets show that ProxySPEX more faithfully reconstructs LLM outputs by 20% over marginal attribution approaches while using $10\times$ fewer inferences than SPEX. By accounting for interactions, ProxySPEX identifies features that influence model output over 20% more than those selected by marginal approaches. Further, we apply ProxySPEX to two interpretability tasks. Data attribution, where we identify interactions among CIFAR-10 training samples that influence test predictions, and mechanistic interpretability, where we uncover interactions between attention heads, both within and across layers, on a question-answering task. ProxySPEX identifies interactions that enable more aggressive pruning of heads than marginal approaches. 

---
# SweEval: Do LLMs Really Swear? A Safety Benchmark for Testing Limits for Enterprise Use 

**Authors**: Hitesh Laxmichand Patel, Amit Agarwal, Arion Das, Bhargava Kumar, Srikant Panda, Priyaranjan Pattnayak, Taki Hasan Rafi, Tejaswini Kumar, Dong-Kyu Chae  

**Link**: [PDF](https://arxiv.org/pdf/2505.17332)  

**Abstract**: Enterprise customers are increasingly adopting Large Language Models (LLMs) for critical communication tasks, such as drafting emails, crafting sales pitches, and composing casual messages. Deploying such models across different regions requires them to understand diverse cultural and linguistic contexts and generate safe and respectful responses. For enterprise applications, it is crucial to mitigate reputational risks, maintain trust, and ensure compliance by effectively identifying and handling unsafe or offensive language. To address this, we introduce SweEval, a benchmark simulating real-world scenarios with variations in tone (positive or negative) and context (formal or informal). The prompts explicitly instruct the model to include specific swear words while completing the task. This benchmark evaluates whether LLMs comply with or resist such inappropriate instructions and assesses their alignment with ethical frameworks, cultural nuances, and language comprehension capabilities. In order to advance research in building ethically aligned AI systems for enterprise use and beyond, we release the dataset and code: this https URL. 

---
# Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models 

**Authors**: Jiachen Jiang, Jinxin Zhou, Bo Peng, Xia Ning, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17316)  

**Abstract**: Achieving better alignment between vision embeddings and Large Language Models (LLMs) is crucial for enhancing the abilities of Multimodal LLMs (MLLMs), particularly for recent models that rely on powerful pretrained vision encoders and LLMs. A common approach to connect the pretrained vision encoder and LLM is through a projector applied after the vision encoder. However, the projector is often trained to enable the LLM to generate captions, and hence the mechanism by which LLMs understand each vision token remains unclear. In this work, we first investigate the role of the projector in compressing vision embeddings and aligning them with word embeddings. We show that the projector significantly compresses visual information, removing redundant details while preserving essential elements necessary for the LLM to understand visual content. We then examine patch-level alignment -- the alignment between each vision patch and its corresponding semantic words -- and propose a *multi-semantic alignment hypothesis*. Our analysis indicates that the projector trained by caption loss improves patch-level alignment but only to a limited extent, resulting in weak and coarse alignment. To address this issue, we propose *patch-aligned training* to efficiently enhance patch-level alignment. Our experiments show that patch-aligned training (1) achieves stronger compression capability and improved patch-level alignment, enabling the MLLM to generate higher-quality captions, (2) improves the MLLM's performance by 16% on referring expression grounding tasks, 4% on question-answering tasks, and 3% on modern instruction-following benchmarks when using the same supervised fine-tuning (SFT) setting. The proposed method can be easily extended to other multimodal models. 

---
# Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty 

**Authors**: Peilin Wu, Mian Zhang, Xinlu Zhang, Xinya Du, Zhiyu Zoey Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17281)  

**Abstract**: Agentic Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by enabling dynamic, multi-step reasoning and information retrieval. However, these systems often exhibit sub-optimal search behaviors like over-search (retrieving redundant information) and under-search (failing to retrieve necessary information), which hinder efficiency and reliability. This work formally defines and quantifies these behaviors, revealing their prevalence across multiple QA datasets and agentic RAG systems (e.g., one model could have avoided searching in 27.7% of its search steps). Furthermore, we demonstrate a crucial link between these inefficiencies and the models' uncertainty regarding their own knowledge boundaries, where response accuracy correlates with model's uncertainty in its search decisions. To address this, we propose $\beta$-GRPO, a reinforcement learning-based training method that incorporates confidence threshold to reward high-certainty search decisions. Experiments on seven QA benchmarks show that $\beta$-GRPO enable a 3B model with better agentic RAG ability, outperforming other strong baselines with a 4% higher average exact match score. 

---
# CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports 

**Authors**: Xiao Yu Cindy Zhang, Carlos R. Ferreira, Francis Rossignol, Raymond T. Ng, Wyeth Wasserman, Jian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17265)  

**Abstract**: Rare diseases, including Inborn Errors of Metabolism (IEM), pose significant diagnostic challenges. Case reports serve as key but computationally underutilized resources to inform diagnosis. Clinical dense information extraction refers to organizing medical information into structured predefined categories. Large Language Models (LLMs) may enable scalable information extraction from case reports but are rarely evaluated for this task. We introduce CaseReportBench, an expert-annotated dataset for dense information extraction of case reports, focusing on IEMs. Using this dataset, we assess various models and prompting strategies, introducing novel approaches such as category-specific prompting and subheading-filtered data integration. Zero-shot chain-of-thought prompting offers little advantage over standard zero-shot prompting. Category-specific prompting improves alignment with the benchmark. The open-source model Qwen2.5-7B outperforms GPT-4o for this task. Our clinician evaluations show that LLMs can extract clinically relevant details from case reports, supporting rare disease diagnosis and management. We also highlight areas for improvement, such as LLMs' limitations in recognizing negative findings important for differential diagnosis. This work advances LLM-driven clinical natural language processing and paves the way for scalable medical AI applications. 

---
# Optimal Policy Minimum Bayesian Risk 

**Authors**: Ramón Fernandez Astudillo, Md Arafat Sultan, Aashka Trivedi, Yousef El-Kurdi, Tahira Naseem, Radu Florian, Salim Roukos  

**Link**: [PDF](https://arxiv.org/pdf/2505.17242)  

**Abstract**: Inference scaling can help LLMs solve complex reasoning problems through extended runtime computation. On top of targeted supervision for long chain-of-thought (long-CoT) generation, purely inference-time techniques such as best-of-N (BoN) sampling, majority voting, or more generally, minimum Bayes risk decoding (MBRD), can further improve LLM accuracy by generating multiple candidate solutions and aggregating over them. These methods typically leverage additional signals in the form of reward models and risk/similarity functions that compare generated samples, e.g., exact match in some normalized space or standard similarity metrics such as Rouge. Here we present a novel method for incorporating reward and risk/similarity signals into MBRD. Based on the concept of optimal policy in KL-controlled reinforcement learning, our framework provides a simple and well-defined mechanism for leveraging such signals, offering several advantages over traditional inference-time methods: higher robustness, improved accuracy, and well-understood asymptotic behavior. In addition, it allows for the development of a sample-efficient variant of MBRD that can adjust the number of samples to generate according to the difficulty of the problem, without relying on majority vote counts. We empirically demonstrate the advantages of our approach on math (MATH-$500$) and coding (HumanEval) tasks using recent open-source models. We also present a comprehensive analysis of its accuracy-compute trade-offs. 

---
# Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs 

**Authors**: Kangda Wei, Hasnat Md Abdullah, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17217)  

**Abstract**: Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data. 

---
# LiloDriver: A Lifelong Learning Framework for Closed-loop Motion Planning in Long-tail Autonomous Driving Scenarios 

**Authors**: Huaiyuan Yao, Pengfei Li, Bu Jin, Yupeng Zheng, An Liu, Lisen Mu, Qing Su, Qian Zhang, Yilun Chen, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17209)  

**Abstract**: Recent advances in autonomous driving research towards motion planners that are robust, safe, and adaptive. However, existing rule-based and data-driven planners lack adaptability to long-tail scenarios, while knowledge-driven methods offer strong reasoning but face challenges in representation, control, and real-world evaluation. To address these challenges, we present LiloDriver, a lifelong learning framework for closed-loop motion planning in long-tail autonomous driving scenarios. By integrating large language models (LLMs) with a memory-augmented planner generation system, LiloDriver continuously adapts to new scenarios without retraining. It features a four-stage architecture including perception, scene encoding, memory-based strategy refinement, and LLM-guided reasoning. Evaluated on the nuPlan benchmark, LiloDriver achieves superior performance in both common and rare driving scenarios, outperforming static rule-based and learning-based planners. Our results highlight the effectiveness of combining structured memory and LLM reasoning to enable scalable, human-like motion planning in real-world autonomous driving. Our code is available at this https URL. 

---
# Next Token Perception Score: Analytical Assessment of your LLM Perception Skills 

**Authors**: Yu-Ang Cheng, Leyang Hu, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.17169)  

**Abstract**: Autoregressive pretraining has become the de facto paradigm for learning general-purpose representations in large language models (LLMs). However, linear probe performance across downstream perception tasks shows substantial variability, suggesting that features optimized for next-token prediction do not consistently transfer well to downstream perception tasks. We demonstrate that representations learned via autoregression capture features that may lie outside the subspaces most informative for perception. To quantify the (mis)alignment between autoregressive pretraining and downstream perception, we introduce the Next Token Perception Score (NTPS)-a score derived under a linear setting that measures the overlap between autoregressive and perception feature subspaces. This metric can be easily computed in closed form from pretrained representations and labeled data, and is proven to both upper- and lower-bound the excess loss. Empirically, we show that NTPS correlates strongly with linear probe accuracy across 12 diverse NLP datasets and eight pretrained models ranging from 270M to 8B parameters, confirming its utility as a measure of alignment. Furthermore, we show that NTPS increases following low-rank adaptation (LoRA) fine-tuning, especially in large models, suggesting that LoRA aligning representations to perception tasks enhances subspace overlap and thus improves downstream performance. More importantly, we find that NTPS reliably predicts the additional accuracy gains attained by LoRA finetuning thereby providing a lightweight prescreening tool for LoRA adaptation. Our results offer both theoretical insights and practical tools for analytically assessing LLM perception skills. 

---
# Select2Reason: Efficient Instruction-Tuning Data Selection for Long-CoT Reasoning 

**Authors**: Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Xiaojun Wu, Honghao Liu, Hui Xiong, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17266)  

**Abstract**: A practical approach to activate long chain-of-thoughts reasoning ability in pre-trained large language models is to perform supervised fine-tuning on instruction datasets synthesized by strong Large Reasoning Models such as DeepSeek-R1, offering a cost-effective alternative to reinforcement learning. However, large-scale instruction sets with more than 100k samples incur significant training overhead, while effective strategies for automatic long-CoT instruction selection still remain unexplored. In this work, we propose Select2Reason, a novel and efficient instruction-tuning data selection framework for long-CoT reasoning. From the perspective of emergence of rethinking behaviors like self-correction and backtracking, we investigate common metrics that may determine the quality of long-CoT reasoning instructions. Select2Reason leverages a quantifier to estimate difficulty of question and jointly incorporates a reasoning trace length-based heuristic through a weighted scheme for ranking to prioritize high-utility examples. Empirical results on OpenR1-Math-220k demonstrate that fine-tuning LLM on only 10% of the data selected by Select2Reason achieves performance competitive with or superior to full-data tuning and open-source baseline OpenR1-Qwen-7B across three competition-level and six comprehensive mathematical benchmarks. Further experiments highlight the scalability in varying data size, efficiency during inference, and its adaptability to other instruction pools with minimal cost. 

---
# DailyQA: A Benchmark to Evaluate Web Retrieval Augmented LLMs Based on Capturing Real-World Changes 

**Authors**: Jiehan Cheng, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17162)  

**Abstract**: We propose DailyQA, an automatically updated dynamic dataset that updates questions weekly and contains answers to questions on any given date. DailyQA utilizes daily updates from Wikipedia revision logs to implement a fully automated pipeline of data filtering, query generation synthesis, quality checking, answer extraction, and query classification. The benchmark requires large language models (LLMs) to process and answer questions involving fast-changing factual data and covering multiple domains. We evaluate several open-source and closed-source LLMs using different RAG pipelines with web search augmentation. We compare the ability of different models to process time-sensitive web information and find that rerank of web retrieval results is critical. Our results indicate that LLMs still face significant challenges in handling frequently updated information, suggesting that DailyQA benchmarking provides valuable insights into the direction of progress for LLMs and RAG systems. 

---
# Amplify Adjacent Token Differences: Enhancing Long Chain-of-Thought Reasoning with Shift-FFN 

**Authors**: Yao Xu, Mingyu Xu, Fangyu Lei, Wangtao Sun, Xiangrong Zeng, Bingning Wang, Guang Liu, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17153)  

**Abstract**: Recently, models such as OpenAI-o1 and DeepSeek-R1 have demonstrated remarkable performance on complex reasoning tasks through Long Chain-of-Thought (Long-CoT) reasoning. Although distilling this capability into student models significantly enhances their performance, this paper finds that fine-tuning LLMs with full parameters or LoRA with a low rank on long CoT data often leads to Cyclical Reasoning, where models repeatedly reiterate previous inference steps until the maximum length limit. Further analysis reveals that smaller differences in representations between adjacent tokens correlates with a higher tendency toward Cyclical Reasoning. To mitigate this issue, this paper proposes Shift Feedforward Networks (Shift-FFN), a novel approach that edits the current token's representation with the previous one before inputting it to FFN. This architecture dynamically amplifies the representation differences between adjacent tokens. Extensive experiments on multiple mathematical reasoning tasks demonstrate that LoRA combined with Shift-FFN achieves higher accuracy and a lower rate of Cyclical Reasoning across various data sizes compared to full fine-tuning and standard LoRA. Our data and code are available at this https URL 

---
# Large Language Models for Predictive Analysis: How Far Are They? 

**Authors**: Qin Chen, Yuanyi Ren, Xiaojun Ma, Yuyang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17149)  

**Abstract**: Predictive analysis is a cornerstone of modern decision-making, with applications in various domains. Large Language Models (LLMs) have emerged as powerful tools in enabling nuanced, knowledge-intensive conversations, thus aiding in complex decision-making tasks. With the burgeoning expectation to harness LLMs for predictive analysis, there is an urgent need to systematically assess their capability in this domain. However, there is a lack of relevant evaluations in existing studies. To bridge this gap, we introduce the \textbf{PredictiQ} benchmark, which integrates 1130 sophisticated predictive analysis queries originating from 44 real-world datasets of 8 diverse fields. We design an evaluation protocol considering text analysis, code generation, and their alignment. Twelve renowned LLMs are evaluated, offering insights into their practical use in predictive analysis. Generally, we believe that existing LLMs still face considerable challenges in conducting predictive analysis. See \href{this https URL}{Github}. 

---
# ReasoningShield: Content Safety Detection over Reasoning Traces of Large Reasoning Models 

**Authors**: Changyi Li, Jiayi Wang, Xudong Pan, Geng Hong, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17244)  

**Abstract**: Large Reasoning Models (LRMs) are transforming the AI landscape with advanced reasoning capabilities. While the generated reasoning traces enhance model transparency, they can still contain unsafe content, even when the final answer appears safe. Existing moderation tools, primarily designed for question-answer (QA) pairs, are empirically ineffective at detecting hidden risks embedded in reasoning traces. After identifying the key challenges, we formally define the question-thought (QT) moderation task and propose ReasoningShield, the first safety detection model tailored to identify potential risks in the reasoning trace before reaching the final answer. To construct the model, we synthesize a high-quality reasoning safety detection dataset comprising over 8,000 question-thought pairs spanning ten risk categories and three safety levels. Our dataset construction process incorporates a comprehensive human-AI collaborative annotation pipeline, which achieves over 93% annotation accuracy while significantly reducing human costs. On a diverse set of in-distribution and out-of-distribution benchmarks, ReasoningShield outperforms mainstream content safety moderation models in identifying risks within reasoning traces, with an average F1 score exceeding 0.92. Notably, despite being trained on our QT dataset only, ReasoningShield also demonstrates competitive performance in detecting unsafe question-answer pairs on traditional benchmarks, rivaling baselines trained on 10 times larger datasets and base models, which strongly validates the quality of our dataset. Furthermore, ReasoningShield is built upon compact 1B/3B base models to facilitate lightweight deployment and provides human-friendly risk analysis by default. To foster future research, we publicly release all the resources. 

---
# LLM-Powered Agents for Navigating Venice's Historical Cadastre 

**Authors**: Tristan Karch, Jakhongir Saydaliev, Isabella Di Lenardo, Frédéric Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17148)  

**Abstract**: Cadastral data reveal key information about the historical organization of cities but are often non-standardized due to diverse formats and human annotations, complicating large-scale analysis. We explore as a case study Venice's urban history during the critical period from 1740 to 1808, capturing the transition following the fall of the ancient Republic and the Ancien Régime. This era's complex cadastral data, marked by its volume and lack of uniform structure, presents unique challenges that our approach adeptly navigates, enabling us to generate spatial queries that bridge past and present urban landscapes. We present a text-to-programs framework that leverages Large Language Models (LLMs) to translate natural language queries into executable code for processing historical cadastral records. Our methodology implements two complementary techniques: a text-to-SQL approach for handling structured queries about specific cadastral information, and a text-to-Python approach for complex analytical operations requiring custom data manipulation. We propose a taxonomy that classifies historical research questions based on their complexity and analytical requirements, mapping them to the most appropriate technical approach. This framework is supported by an investigation into the execution consistency of the system, alongside a qualitative analysis of the answers it produces. By ensuring interpretability and minimizing hallucination through verifiable program outputs, we demonstrate the system's effectiveness in reconstructing past population information, property features, and spatiotemporal comparisons in Venice. 

---
# MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming 

**Authors**: Weiyang Guo, Jing Li, Wenya Wang, YU LI, Daojing He, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17147)  

**Abstract**: The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the \textbf{M}ulti-\textbf{T}urn \textbf{S}afety \textbf{A}lignment (\ourapproach) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks. 

---
# LLM Access Shield: Domain-Specific LLM Framework for Privacy Policy Compliance 

**Authors**: Yu Wang, Cailing Cai, Zhihua Xiao, Peifung E. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.17145)  

**Abstract**: Large language models (LLMs) are increasingly applied in fields such as finance, education, and governance due to their ability to generate human-like text and adapt to specialized tasks. However, their widespread adoption raises critical concerns about data privacy and security, including the risk of sensitive data exposure.
In this paper, we propose a security framework to enforce policy compliance and mitigate risks in LLM interactions. Our approach introduces three key innovations: (i) LLM-based policy enforcement: a customizable mechanism that enhances domain-specific detection of sensitive data. (ii) Dynamic policy customization: real-time policy adaptation and enforcement during user-LLM interactions to ensure compliance with evolving security requirements. (iii) Sensitive data anonymization: a format-preserving encryption technique that protects sensitive information while maintaining contextual integrity. Experimental results demonstrate that our framework effectively mitigates security risks while preserving the functional accuracy of LLM-driven tasks. 

---
# Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs 

**Authors**: Essa Jan, Moiz Ali, Muhammad Saram Hassan, Fareed Zaffar, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.17140)  

**Abstract**: As the knowledge of large language models (LLMs) becomes outdated over time, there is a growing need for efficient methods to update them, especially when injecting proprietary information. Our study reveals that comprehension-intensive fine-tuning tasks (e.g., question answering and blanks) achieve substantially higher knowledge retention rates (48%) compared to mapping-oriented tasks like translation (17%) or text-to-JSON conversion (20%), despite exposure to identical factual content. We demonstrate that this pattern persists across model architectures and follows scaling laws, with larger models showing improved retention across all task types. However, all models exhibit significant performance drops when applying injected knowledge in broader contexts, suggesting limited semantic integration. These findings show the importance of task selection in updating LLM knowledge, showing that effective knowledge injection relies not just on data exposure but on the depth of cognitive engagement during fine-tuning. 

---
# EarthSE: A Benchmark Evaluating Earth Scientific Exploration Capability for Large Language Models 

**Authors**: Wanghan Xu, Xiangyu Zhao, Yuhao Zhou, Xiaoyu Yue, Ben Fei, Fenghua Ling, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.17139)  

**Abstract**: Advancements in Large Language Models (LLMs) drive interest in scientific applications, necessitating specialized benchmarks such as Earth science. Existing benchmarks either present a general science focus devoid of Earth science specificity or cover isolated subdomains, lacking holistic evaluation. Furthermore, current benchmarks typically neglect the assessment of LLMs' capabilities in open-ended scientific exploration. In this paper, we present a comprehensive and professional benchmark for the Earth sciences, designed to evaluate the capabilities of LLMs in scientific exploration within this domain, spanning from fundamental to advanced levels. Leveraging a corpus of 100,000 research papers, we first construct two Question Answering (QA) datasets: Earth-Iron, which offers extensive question coverage for broad assessment, and Earth-Silver, which features a higher level of difficulty to evaluate professional depth. These datasets encompass five Earth spheres, 114 disciplines, and 11 task categories, assessing foundational knowledge crucial for scientific exploration. Most notably, we introduce Earth-Gold with new metrics, a dataset comprising open-ended multi-turn dialogues specifically designed to evaluate the advanced capabilities of LLMs in scientific exploration, including methodology induction, limitation analysis, and concept proposal. Extensive experiments reveal limitations in 11 leading LLMs across different domains and tasks, highlighting considerable room for improvement in their scientific exploration capabilities. The benchmark is available on this https URL . 

---
# RAP: Runtime-Adaptive Pruning for LLM Inference 

**Authors**: Huanrong Liu, Chunlin Tian, Xuyang Wei, Jiaheng Dai, Qin Liu, Tianqi Wei, Qingbiao Li, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17138)  

**Abstract**: Large language models (LLMs) excel at language understanding and generation, but their enormous computational and memory requirements hinder deployment. Compression offers a potential solution to mitigate these constraints. However, most existing methods rely on fixed heuristics and thus fail to adapt to runtime memory variations or heterogeneous KV-cache demands arising from diverse user requests. To address these limitations, we propose RAP, an elastic pruning framework driven by reinforcement learning (RL) that dynamically adjusts compression strategies in a runtime-aware manner. Specifically, RAP dynamically tracks the evolving ratio between model parameters and KV-cache across practical execution. Recognizing that FFNs house most parameters, whereas parameter -light attention layers dominate KV-cache formation, the RL agent retains only those components that maximize utility within the current memory budget, conditioned on instantaneous workload and device state. Extensive experiments results demonstrate that RAP outperforms state-of-the-art baselines, marking the first time to jointly consider model weights and KV-cache on the fly. 

---
# Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands 

**Authors**: Kristin Qi, Youxiang Zhu, Caroline Summerour, John A. Batsis, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17137)  

**Abstract**: Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline. 

---
# Foundation Models for Geospatial Reasoning: Assessing Capabilities of Large Language Models in Understanding Geometries and Topological Spatial Relations 

**Authors**: Yuhan Ji, Song Gao, Ying Nie, Ivan Majić, Krzysztof Janowicz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17136)  

**Abstract**: Applying AI foundation models directly to geospatial datasets remains challenging due to their limited ability to represent and reason with geographical entities, specifically vector-based geometries and natural language descriptions of complex spatial relations. To address these issues, we investigate the extent to which a well-known-text (WKT) representation of geometries and their spatial relations (e.g., topological predicates) are preserved during spatial reasoning when the geospatial vector data are passed to large language models (LLMs) including GPT-3.5-turbo, GPT-4, and DeepSeek-R1-14B. Our workflow employs three distinct approaches to complete the spatial reasoning tasks for comparison, i.e., geometry embedding-based, prompt engineering-based, and everyday language-based evaluation. Our experiment results demonstrate that both the embedding-based and prompt engineering-based approaches to geospatial question-answering tasks with GPT models can achieve an accuracy of over 0.6 on average for the identification of topological spatial relations between two geometries. Among the evaluated models, GPT-4 with few-shot prompting achieved the highest performance with over 0.66 accuracy on topological spatial relation inference. Additionally, GPT-based reasoner is capable of properly comprehending inverse topological spatial relations and including an LLM-generated geometry can enhance the effectiveness for geographic entity retrieval. GPT-4 also exhibits the ability to translate certain vernacular descriptions about places into formal topological relations, and adding the geometry-type or place-type context in prompts may improve inference accuracy, but it varies by instance. The performance of these spatial reasoning tasks offers valuable insights for the refinement of LLMs with geographical knowledge towards the development of geo-foundation models capable of geospatial reasoning. 

---
# LongMagpie: A Self-synthesis Method for Generating Large-scale Long-context Instructions 

**Authors**: Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17134)  

**Abstract**: High-quality long-context instruction data is essential for aligning long-context large language models (LLMs). Despite the public release of models like Qwen and Llama, their long-context instruction data remains proprietary. Human annotation is costly and challenging, while template-based synthesis methods limit scale, diversity, and quality. We introduce LongMagpie, a self-synthesis framework that automatically generates large-scale long-context instruction data. Our key insight is that aligned long-context LLMs, when presented with a document followed by special tokens preceding a user turn, auto-regressively generate contextually relevant queries. By harvesting these document-query pairs and the model's responses, LongMagpie produces high-quality instructions without human effort. Experiments on HELMET, RULER, and Longbench v2 demonstrate that LongMagpie achieves leading performance on long-context tasks while maintaining competitive performance on short-context tasks, establishing it as a simple and effective approach for open, diverse, and scalable long-context instruction data synthesis. 

---
# OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning 

**Authors**: Mingxin Huang, Yongxin Shi, Dezhi Peng, Songxuan Lai, Zecheng Xie, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17163)  

**Abstract**: Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across diverse visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the lack of a systematic benchmark. To address this gap, we propose OCR-Reasoning, a comprehensive benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. The benchmark comprises 1,069 human-annotated examples spanning 6 core reasoning abilities and 18 practical reasoning tasks in text-rich visual scenarios. Furthermore, unlike other text-rich image understanding benchmarks that only annotate the final answers, OCR-Reasoning also annotates the reasoning process simultaneously. With the annotated reasoning process and the final answers, OCR-Reasoning evaluates not only the final answers generated by models but also their reasoning processes, enabling a holistic analysis of their problem-solving abilities. Leveraging this benchmark, we conducted a comprehensive evaluation of state-of-the-art MLLMs. Our results demonstrate the limitations of existing methodologies. Notably, even state-of-the-art MLLMs exhibit substantial difficulties, with none achieving accuracy surpassing 50\% across OCR-Reasoning, indicating that the challenges of text-rich image reasoning are an urgent issue to be addressed. The benchmark and evaluation scripts are available at this https URL. 

---
# NEXT-EVAL: Next Evaluation of Traditional and LLM Web Data Record Extraction 

**Authors**: Soyeon Kim, Namhee Kim, Yeonwoo Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17125)  

**Abstract**: Effective evaluation of web data record extraction methods is crucial, yet hampered by static, domain-specific benchmarks and opaque scoring practices. This makes fair comparison between traditional algorithmic techniques, which rely on structural heuristics, and Large Language Model (LLM)-based approaches, offering zero-shot extraction across diverse layouts, particularly challenging. To overcome these limitations, we introduce a concrete evaluation framework. Our framework systematically generates evaluation datasets from arbitrary MHTML snapshots, annotates XPath-based supervision labels, and employs structure-aware metrics for consistent scoring, specifically preventing text hallucination and allowing only for the assessment of positional hallucination. It also incorporates preprocessing strategies to optimize input for LLMs while preserving DOM semantics: HTML slimming, Hierarchical JSON, and Flat JSON. Additionally, we created a publicly available synthetic dataset by transforming DOM structures and modifying content. We benchmark deterministic heuristic algorithms and off-the-shelf LLMs across these multiple input formats. Our benchmarking shows that Flat JSON input enables LLMs to achieve superior extraction accuracy (F1 score of 0.9567) and minimal hallucination compared to other input formats like Slimmed HTML and Hierarchical JSON. We establish a standardized foundation for rigorous benchmarking, paving the way for the next principled advancements in web data record extraction. 

---
# From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning 

**Authors**: Chen Shani, Dan Jurafsky, Yann LeCun, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2505.17117)  

**Abstract**: Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations. 

---
# CRAKEN: Cybersecurity LLM Agent with Knowledge-Based Execution 

**Authors**: Minghao Shao, Haoran Xi, Nanda Rani, Meet Udeshi, Venkata Sai Charan Putrevu, Kimberly Milner, Brendan Dolan-Gavitt, Sandeep Kumar Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2505.17107)  

**Abstract**: Large Language Model (LLM) agents can automate cybersecurity tasks and can adapt to the evolving cybersecurity landscape without re-engineering. While LLM agents have demonstrated cybersecurity capabilities on Capture-The-Flag (CTF) competitions, they have two key limitations: accessing latest cybersecurity expertise beyond training data, and integrating new knowledge into complex task planning. Knowledge-based approaches that incorporate technical understanding into the task-solving automation can tackle these limitations. We present CRAKEN, a knowledge-based LLM agent framework that improves cybersecurity capability through three core mechanisms: contextual decomposition of task-critical information, iterative self-reflected knowledge retrieval, and knowledge-hint injection that transforms insights into adaptive attack strategies. Comprehensive evaluations with different configurations show CRAKEN's effectiveness in multi-stage vulnerability detection and exploitation compared to previous approaches. Our extensible architecture establishes new methodologies for embedding new security knowledge into LLM-driven cybersecurity agentic systems. With a knowledge database of CTF writeups, CRAKEN obtained an accuracy of 22% on NYU CTF Bench, outperforming prior works by 3% and achieving state-of-the-art results. On evaluation of MITRE ATT&CK techniques, CRAKEN solves 25-30% more techniques than prior work, demonstrating improved cybersecurity capabilities via knowledge-based execution. We make our framework open source to public this https URL. 

---
# Forging Time Series with Language: A Large Language Model Approach to Synthetic Data Generation 

**Authors**: Cécile Rousseau, Tobia Boschi, Giandomenico Cornacchia, Dhaval Salwala, Alessandra Pascale, Juan Bernabe Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2505.17103)  

**Abstract**: SDForger is a flexible and efficient framework for generating high-quality multivariate time series using LLMs. Leveraging a compact data representation, SDForger provides synthetic time series generation from a few samples and low-computation fine-tuning of any autoregressive LLM. Specifically, the framework transforms univariate and multivariate signals into tabular embeddings, which are then encoded into text and used to fine-tune the LLM. At inference, new textual embeddings are sampled and decoded into synthetic time series that retain the original data's statistical properties and temporal dynamics. Across a diverse range of datasets, SDForger outperforms existing generative models in many scenarios, both in similarity-based evaluations and downstream forecasting tasks. By enabling textual conditioning in the generation process, SDForger paves the way for multimodal modeling and the streamlined integration of time series with textual information. SDForger source code will be open-sourced soon. 

---
# Large Language Models Implicitly Learn to See and Hear Just By Reading 

**Authors**: Prateek Verma, Mert Pilanci  

**Link**: [PDF](https://arxiv.org/pdf/2505.17091)  

**Abstract**: This paper presents a fascinating find: By training an auto-regressive LLM model on text tokens, the text model inherently develops internally an ability to understand images and audio, thereby developing the ability to see and hear just by reading. Popular audio and visual LLM models fine-tune text LLM models to give text output conditioned on images and audio embeddings. On the other hand, our architecture takes in patches of images, audio waveforms or tokens as input. It gives us the embeddings or category labels typical of a classification pipeline. We show the generality of text weights in aiding audio classification for datasets FSD-50K and GTZAN. Further, we show this working for image classification on CIFAR-10 and Fashion-MNIST, as well on image patches. This pushes the notion of text-LLMs learning powerful internal circuits that can be utilized by activating necessary connections for various applications rather than training models from scratch every single time. 

---
# Semi-Clairvoyant Scheduling of Speculative Decoding Requests to Minimize LLM Inference Latency 

**Authors**: Ruixiao Li, Fahao Chen, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17074)  

**Abstract**: Speculative decoding accelerates Large Language Model (LLM) inference by employing a small speculative model (SSM) to generate multiple candidate tokens and verify them using the LLM in parallel. This technique has been widely integrated into LLM inference serving systems. However, inference requests typically exhibit uncertain execution time, which poses a significant challenge of efficiently scheduling requests in these systems. Existing work estimates execution time based solely on predicted output length, which could be inaccurate because execution time depends on both output length and token acceptance rate of verification by the LLM. In this paper, we propose a semi-clairvoyant request scheduling algorithm called Least-Attained/Perceived-Service for Speculative Decoding (LAPS-SD). Given a number of inference requests, LAPS-SD can effectively minimize average inference latency by adaptively scheduling requests according to their features during decoding. When the token acceptance rate is dynamic and execution time is difficult to estimate, LAPS-SD maintains multiple priority queues and allows request execution preemption across different queues. Once the token acceptance rate becomes stable, LAPS-SD can accurately estimate the execution time and schedule requests accordingly. Extensive experiments show that LAPS-SD reduces inference latency by approximately 39\% compared to state-of-the-art scheduling methods. 

---
# Informatics for Food Processing 

**Authors**: Gordana Ispirova, Michael Sebek, Giulia Menichetti  

**Link**: [PDF](https://arxiv.org/pdf/2505.17087)  

**Abstract**: This chapter explores the evolution, classification, and health implications of food processing, while emphasizing the transformative role of machine learning, artificial intelligence (AI), and data science in advancing food informatics. It begins with a historical overview and a critical review of traditional classification frameworks such as NOVA, Nutri-Score, and SIGA, highlighting their strengths and limitations, particularly the subjectivity and reproducibility challenges that hinder epidemiological research and public policy. To address these issues, the chapter presents novel computational approaches, including FoodProX, a random forest model trained on nutrient composition data to infer processing levels and generate a continuous FPro score. It also explores how large language models like BERT and BioBERT can semantically embed food descriptions and ingredient lists for predictive tasks, even in the presence of missing data. A key contribution of the chapter is a novel case study using the Open Food Facts database, showcasing how multimodal AI models can integrate structured and unstructured data to classify foods at scale, offering a new paradigm for food processing assessment in public health and research. 

---
# Mechanistic Interpretability of GPT-like Models on Summarization Tasks 

**Authors**: Anurag Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.17073)  

**Abstract**: Mechanistic interpretability research seeks to reveal the inner workings of large language models, yet most work focuses on classification or generative tasks rather than summarization. This paper presents an interpretability framework for analyzing how GPT-like models adapt to summarization tasks. We conduct differential analysis between pre-trained and fine-tuned models, quantifying changes in attention patterns and internal activations. By identifying specific layers and attention heads that undergo significant transformation, we locate the "summarization circuit" within the model architecture. Our findings reveal that middle layers (particularly 2, 3, and 5) exhibit the most dramatic changes, with 62% of attention heads showing decreased entropy, indicating a shift toward focused information selection. We demonstrate that targeted LoRA adaptation of these identified circuits achieves significant performance improvement over standard LoRA fine-tuning while requiring fewer training epochs. This work bridges the gap between black-box evaluation and mechanistic understanding, providing insights into how neural networks perform information selection and compression during summarization. 

---
# Safety Alignment Can Be Not Superficial With Explicit Safety Signals 

**Authors**: Jianwei Li, Jung-Eng Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.17072)  

**Abstract**: Recent studies on the safety alignment of large language models (LLMs) have revealed that existing approaches often operate superficially, leaving models vulnerable to various adversarial attacks. Despite their significance, these studies generally fail to offer actionable solutions beyond data augmentation for achieving more robust safety mechanisms. This paper identifies a fundamental cause of this superficiality: existing alignment approaches often presume that models can implicitly learn a safety-related reasoning task during the alignment process, enabling them to refuse harmful requests. However, the learned safety signals are often diluted by other competing objectives, leading models to struggle with drawing a firm safety-conscious decision boundary when confronted with adversarial attacks. Based on this observation, by explicitly introducing a safety-related binary classification task and integrating its signals with our attention and decoding strategies, we eliminate this ambiguity and allow models to respond more responsibly to malicious queries. We emphasize that, with less than 0.2x overhead cost, our approach enables LLMs to assess the safety of both the query and the previously generated tokens at each necessary generating step. Extensive experiments demonstrate that our method significantly improves the resilience of LLMs against various adversarial attacks, offering a promising pathway toward more robust generative AI systems. 

---
# Improving LLM Outputs Against Jailbreak Attacks with Expert Model Integration 

**Authors**: Tatia Tsmindashvili, Ana Kolkhidashvili, Dachi Kurtskhalia, Nino Maghlakelidze, Elene Mekvabishvili, Guram Dentoshvili, Orkhan Shamilov, Zaal Gachechiladze, Steven Saporta, David Dachi Choladze  

**Link**: [PDF](https://arxiv.org/pdf/2505.17066)  

**Abstract**: Using LLMs in a production environment presents security challenges that include vulnerabilities to jailbreaks and prompt injections, which can result in harmful outputs for humans or the enterprise. The challenge is amplified when working within a specific domain, as topics generally accepted for LLMs to address may be irrelevant to that field. These problems can be mitigated, for example, by fine-tuning large language models with domain-specific and security-focused data. However, these alone are insufficient, as jailbreak techniques evolve. Additionally, API-accessed models do not offer the flexibility needed to tailor behavior to industry-specific objectives, and in-context learning is not always sufficient or reliable. In response to these challenges, we introduce Archias, an expert model adept at distinguishing between in-domain and out-of-domain communications. Archias classifies user inquiries into several categories: in-domain (specifically for the automotive industry), malicious questions, price injections, prompt injections, and out-of-domain examples. Our methodology integrates outputs from the expert model (Archias) into prompts, which are then processed by the LLM to generate responses. This method increases the model's ability to understand the user's intention and give appropriate answers. Archias can be adjusted, fine-tuned, and used for many different purposes due to its small size. Therefore, it can be easily customized to the needs of any industry. To validate our approach, we created a benchmark dataset for the automotive industry. Furthermore, in the interest of advancing research and development, we release our benchmark dataset to the community. 

---
# GemMaroc: Unlocking Darija Proficiency in LLMs with Minimal Data 

**Authors**: Abderrahman Skiredj, Ferdaous Azhari, Houdaifa Atou, Nouamane Tazi, Ismail Berrada  

**Link**: [PDF](https://arxiv.org/pdf/2505.17082)  

**Abstract**: Open-source large language models (LLMs) still marginalise Moroccan Arabic (Darija), forcing practitioners either to bolt on heavyweight Arabic adapters or to sacrifice the very reasoning skills that make LLMs useful. We show that a rigorously quality-over-quantity alignment strategy can surface fluent Darija while safeguarding the backbone s cross-lingual reasoning at a sliver of the usual compute. We translate three compact instruction suites LIMA 1 K, DEITA 6 K and TULU 50 K into Darija, preserve 20 of the English originals, and add mathematics, coding and scientific prompts. A LoRA-tuned Gemma 3-4B trained on 5 K mixed instructions lifts DarijaMMLU from 32.8 to 42.7 ; adding the reasoning-dense TULU portion pushes it to 47.5 with no English regression. Scaling the identical recipe to Gemma 3-27B produces GemMaroc-27B, which matches Atlas-Chat on DarijaMMLU (61.6 ) and leaps ahead on Darija commonsense, scoring 60.5 on HellaSwag versus Atlas-Chat s 48.4 . Crucially, GemMaroc retains Gemma-27B s strong maths and general-reasoning ability, showing only minimal movement on GSM8K and English benchmarks. The entire model is trained in just 48 GPU.h, underscoring a Green AI pathway to inclusive, sustainable language technology. We release code, data and checkpoints to spur Darija-centric applications in education, public services and everyday digital interaction. 

---
# Are LLMs Ready for English Standardized Tests? A Benchmarking and Elicitation Perspective 

**Authors**: Luoxi Tang, Tharunya Sundar, Shuai Yang, Ankita Patra, Manohar Chippada, Giqi Zhao, Yi Li, Riteng Zhang, Tunan Zhao, Ting Yang, Yuqiao Meng, Weicheng Ma, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17056)  

**Abstract**: AI is transforming education by enabling powerful tools that enhance learning experiences. Among recent advancements, large language models (LLMs) hold particular promise for revolutionizing how learners interact with educational content. In this work, we investigate the potential of LLMs to support standardized test preparation by focusing on English Standardized Tests (ESTs). Specifically, we assess their ability to generate accurate and contextually appropriate solutions across a diverse set of EST question types. We introduce ESTBOOK, a comprehensive benchmark designed to evaluate the capabilities of LLMs in solving EST questions. ESTBOOK aggregates five widely recognized tests, encompassing 29 question types and over 10,576 questions across multiple modalities, including text, images, audio, tables, and mathematical symbols. Using ESTBOOK, we systematically evaluate both the accuracy and inference efficiency of LLMs. Additionally, we propose a breakdown analysis framework that decomposes complex EST questions into task-specific solution steps. This framework allows us to isolate and assess LLM performance at each stage of the reasoning process. Evaluation findings offer insights into the capability of LLMs in educational contexts and point toward targeted strategies for improving their reliability as intelligent tutoring systems. 

---
# Swarm Intelligence Enhanced Reasoning: A Density-Driven Framework for LLM-Based Multi-Agent Optimization 

**Authors**: Ying Zhu, Heng Zhou, Rui Su, Peiqin Zhuang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.17115)  

**Abstract**: Recently, many approaches, such as Chain-of-Thought (CoT) prompting and Multi-Agent Debate (MAD), have been proposed to further enrich Large Language Models' (LLMs) complex problem-solving capacities in reasoning scenarios. However, these methods may fail to solve complex problems due to the lack of ability to find optimal solutions. Swarm Intelligence has been serving as a powerful tool for finding optima in the field of traditional optimization problems. To this end, we propose integrating swarm intelligence into the reasoning process by introducing a novel Agent-based Swarm Intelligence (ASI) paradigm. In this paradigm, we formulate LLM reasoning as an optimization problem and use a swarm intelligence scheme to guide a group of LLM-based agents in collaboratively searching for optimal solutions. To avoid swarm intelligence getting trapped in local optima, we further develop a Swarm Intelligence Enhancing Reasoning (SIER) framework, which develops a density-driven strategy to enhance the reasoning ability. To be specific, we propose to perform kernel density estimation and non-dominated sorting to optimize both solution quality and diversity simultaneously. In this case, SIER efficiently enhances solution space exploration through expanding the diversity of the reasoning path. Besides, a step-level quality evaluation is used to help agents improve solution quality by correcting low-quality intermediate steps. Then, we use quality thresholds to dynamically control the termination of exploration and the selection of candidate steps, enabling a more flexible and efficient reasoning process. Extensive experiments are ... 

---
# SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs 

**Authors**: Jinwoo Park, Seunggeun Cho, Dongsu Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.17052)  

**Abstract**: Large language models (LLMs) power many modern applications, but serving them at scale remains costly and resource-intensive. Current server-centric systems overlook consumer-grade GPUs at the edge. We introduce SpecEdge, an edge-assisted inference framework that splits LLM workloads between edge and server GPUs using a speculative decoding scheme, exchanging only token outputs over the network. SpecEdge employs proactive edge drafting to overlap edge token creation with server verification and pipeline-aware scheduling that interleaves multiple user requests to increase server-side throughput. Experiments show SpecEdge enhances overall cost efficiency by 1.91x through achieving 2.22x server throughput, and reduces inter token latency by 11.24% compared to a server-only baseline, introducing a scalable, cost-effective paradigm for LLM serving. 

---
# Embedding-to-Prefix: Parameter-Efficient Personalization for Pre-Trained Large Language Models 

**Authors**: Bernd Huber, Ghazal Fazelnia, Andreas Damianou, Sebastian Peleato, Max Lefarov, Praveen Ravichandran, Marco De Nadai, Mounia Lalmas-Roellke, Paul N. Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2505.17051)  

**Abstract**: Large language models (LLMs) excel at generating contextually relevant content. However, tailoring these outputs to individual users for effective personalization is a significant challenge. While rich user-specific information often exists as pre-existing user representations, such as embeddings learned from preferences or behaviors, current methods to leverage these for LLM personalization typically require costly fine-tuning or token-heavy prompting. We propose Embedding-to-Prefix (E2P), a parameter-efficient method that injects pre-computed context embeddings into an LLM's hidden representation space through a learned projection to a single soft token prefix. This enables effective personalization while keeping the backbone model frozen and avoiding expensive adaptation techniques. We evaluate E2P across two public datasets and in a production setting: dialogue personalization on Persona-Chat, contextual headline generation on PENS, and large-scale personalization for music and podcast consumption. Results show that E2P preserves contextual signals and achieves strong performance with minimal computational overhead, offering a scalable, efficient solution for contextualizing generative AI systems. 

---
# Towards Robust Evaluation of STEM Education: Leveraging MLLMs in Project-Based Learning 

**Authors**: Yanhao Jia, Xinyi Wu, Qinglin Zhang, Yiran Qin, Luwei Xiao, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17050)  

**Abstract**: Project-Based Learning (PBL) involves a variety of highly correlated multimodal data, making it a vital educational approach within STEM disciplines. With the rapid development of multimodal large language models (MLLMs), researchers have begun exploring their potential to enhance tasks such as information retrieval, knowledge comprehension, and data generation in educational settings. However, existing benchmarks fall short in providing both a free-form output structure and a rigorous human expert validation process, limiting their effectiveness in evaluating real-world educational tasks. Additionally, few methods have developed automated pipelines to assist with the complex responsibilities of teachers leveraging MLLMs, largely due to model hallucination and instability, which lead to unreliable implementation. To address this gap, we introduce PBLBench, a novel benchmark designed to evaluate complex reasoning grounded in domain-specific knowledge and long-context understanding, thereby challenging models with tasks that closely resemble those handled by human experts. To establish reliable ground truth, we adopt the Analytic Hierarchy Process (AHP), utilizing expert-driven pairwise comparisons to derive structured and weighted evaluation criteria. We assess the performance of 15 leading MLLMs/LLMs using PBLBench and demonstrate that even the most advanced models achieve only 59% rank accuracy, underscoring the significant challenges presented by this benchmark. We believe PBLBench will serve as a catalyst for the development of more capable AI agents, ultimately aiming to alleviate teacher workload and enhance educational productivity. 

---
# NeSyGeo: A Neuro-Symbolic Framework for Multimodal Geometric Reasoning Data Generation 

**Authors**: Weiming Wu, Zi-kang Wang, Jin Ye, Zhi Zhou, Yu-Feng Li, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.17121)  

**Abstract**: Obtaining large-scale, high-quality data with reasoning paths is crucial for improving the geometric reasoning capabilities of multi-modal large language models (MLLMs). However, existing data generation methods, whether based on predefined templates or constrained symbolic provers, inevitably face diversity and numerical generalization limitations. To address these limitations, we propose NeSyGeo, a novel neuro-symbolic framework for generating geometric reasoning data. First, we propose a domain-specific language grounded in the entity-relation-constraint paradigm to comprehensively represent all components of plane geometry, along with generative actions defined within this symbolic space. We then design a symbolic-visual-text pipeline that synthesizes symbolic sequences, maps them to corresponding visual and textual representations, and generates diverse question-answer (Q&A) pairs using large language models (LLMs). To the best of our knowledge, we are the first to propose a neuro-symbolic approach in generating multimodal reasoning data. Based on this framework, we construct NeSyGeo-CoT and NeSyGeo-Caption datasets, containing 100k samples, and release a new benchmark NeSyGeo-Test for evaluating geometric reasoning abilities in MLLMs. Experiments demonstrate that the proposal significantly and consistently improves the performance of multiple MLLMs under both reinforcement and supervised fine-tuning. With only 4k samples and two epochs of reinforcement fine-tuning, base models achieve improvements of up to +15.8% on MathVision, +8.4% on MathVerse, and +7.3% on GeoQA. Notably, a 4B model can be improved to outperform an 8B model from the same series on geometric reasoning tasks. 

---
# Social preferences with unstable interactive reasoning: Large language models in economic trust games 

**Authors**: Ou Jiamin, Eikmans Emile, Buskens Vincent, Pankowska Paulina, Shan Yuli  

**Link**: [PDF](https://arxiv.org/pdf/2505.17053)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable capabilities in understanding human languages, this study explores how they translate this understanding into social exchange contexts that capture certain essences of real world human interactions. Three LLMs - ChatGPT-4, Claude, and Bard - were placed in economic trust games where players balance self-interest with trust and reciprocity, making decisions that reveal their social preferences and interactive reasoning abilities. Our study shows that LLMs deviate from pure self-interest and exhibit trust and reciprocity even without being prompted to adopt a specific persona. In the simplest one-shot interaction, LLMs emulated how human players place trust at the beginning of such a game. Larger human-machine divergences emerged in scenarios involving trust repayment or multi-round interactions, where decisions were influenced by both social preferences and interactive reasoning. LLMs responses varied significantly when prompted to adopt personas like selfish or unselfish players, with the impact outweighing differences between models or game types. Response of ChatGPT-4, in an unselfish or neutral persona, resembled the highest trust and reciprocity, surpassing humans, Claude, and Bard. Claude and Bard displayed trust and reciprocity levels that sometimes exceeded and sometimes fell below human choices. When given selfish personas, all LLMs showed lower trust and reciprocity than humans. Interactive reasoning to the actions of counterparts or changing game mechanics appeared to be random rather than stable, reproducible characteristics in the response of LLMs, though some improvements were observed when ChatGPT-4 responded in selfish or unselfish personas. 

---
# Assessing the Quality of AI-Generated Clinical Notes: A Validated Evaluation of a Large Language Model Scribe 

**Authors**: Erin Palm, Astrit Manikantan, Mark E. Pepin, Herprit Mahal, Srikanth Subramanya Belwadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17047)  

**Abstract**: In medical practices across the United States, physicians have begun implementing generative artificial intelligence (AI) tools to perform the function of scribes in order to reduce the burden of documenting clinical encounters. Despite their widespread use, no established methods exist to gauge the quality of AI scribes. To address this gap, we developed a blinded study comparing the relative performance of large language model (LLM) generated clinical notes with those from field experts based on audio-recorded clinical encounters. Quantitative metrics from the Physician Documentation Quality Instrument (PDQI9) provided a framework to measure note quality, which we adapted to assess relative performance of AI generated notes. Clinical experts spanning 5 medical specialties used the PDQI9 tool to evaluate specialist-drafted Gold notes and LLM authored Ambient notes. Two evaluators from each specialty scored notes drafted from a total of 97 patient visits. We found uniformly high inter rater agreement (RWG greater than 0.7) between evaluators in general medicine, orthopedics, and obstetrics and gynecology, and moderate (RWG 0.5 to 0.7) to high inter rater agreement in pediatrics and cardiology. We found a modest yet significant difference in the overall note quality, wherein Gold notes achieved a score of 4.25 out of 5 and Ambient notes scored 4.20 out of 5 (p = 0.04). Our findings support the use of the PDQI9 instrument as a practical method to gauge the quality of LLM authored notes, as compared to human-authored notes. 

---
# GloSS over Toxicity: Understanding and Mitigating Toxicity in LLMs via Global Toxic Subspace 

**Authors**: Zenghao Duan, Zhiyi Yin, Zhichao Shi, Liang Pang, Shaoling Jing, Jiayi Wu, Yu Yan, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17078)  

**Abstract**: This paper investigates the underlying mechanisms of toxicity generation in Large Language Models (LLMs) and proposes an effective detoxification approach. Prior work typically considers the Feed-Forward Network (FFN) as the main source of toxicity, representing toxic regions as a set of toxic vectors or layer-wise subspaces. However, our in-depth analysis reveals that the global toxic subspace offers a more effective and comprehensive representation of toxic region within the model. Building on this insight, we propose GloSS (Global Toxic Subspace Suppression), a lightweight, four-stage method that mitigates toxicity by identifying and removing the global toxic subspace from the parameters of FFN. Experiments across a range of LLMs show that GloSS achieves state-of-the-art detoxification performance while preserving the models general capabilities, without requiring large-scale data or model retraining. 

---
# Gender and Positional Biases in LLM-Based Hiring Decisions: Evidence from Comparative CV/Résumé Evaluations 

**Authors**: David Rozado  

**Link**: [PDF](https://arxiv.org/pdf/2505.17049)  

**Abstract**: This study examines the behavior of Large Language Models (LLMs) when evaluating professional candidates based on their resumes or curricula vitae (CVs). In an experiment involving 22 leading LLMs, each model was systematically given one job description along with a pair of profession-matched CVs, one bearing a male first name, the other a female first name, and asked to select the more suitable candidate for the job. Each CV pair was presented twice, with names swapped to ensure that any observed preferences in candidate selection stemmed from gendered names cues. Despite identical professional qualifications across genders, all LLMs consistently favored female-named candidates across 70 different professions. Adding an explicit gender field (male/female) to the CVs further increased the preference for female applicants. When gendered names were replaced with gender-neutral identifiers "Candidate A" and "Candidate B", several models displayed a preference to select "Candidate A". Counterbalancing gender assignment between these gender-neutral identifiers resulted in gender parity in candidate selection. When asked to rate CVs in isolation rather than compare pairs, LLMs assigned slightly higher average scores to female CVs overall, but the effect size was negligible. Including preferred pronouns (he/him or she/her) next to a candidate's name slightly increased the odds of the candidate being selected regardless of gender. Finally, most models exhibited a substantial positional bias to select the candidate listed first in the prompt. These findings underscore the need for caution when deploying LLMs in high-stakes autonomous decision-making contexts and raise doubts about whether LLMs consistently apply principled reasoning. 

---
# First Finish Search: Efficient Test-Time Scaling in Large Language Models 

**Authors**: Aradhye Agarwal, Ayan Sengupta, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2505.18149)  

**Abstract**: Test-time scaling (TTS), which involves dynamic allocation of compute during inference, offers a promising way to improve reasoning in large language models. While existing TTS methods work well, they often rely on long decoding paths or require a large number of samples to be generated, increasing the token usage and inference latency. We observe the surprising fact that for reasoning tasks, shorter traces are much more likely to be correct than longer ones. Motivated by this, we introduce First Finish Search (FFS), a training-free parallel decoding strategy that launches $n$ independent samples and returns as soon as any one completes. We evaluate FFS alongside simple decoding, beam search, majority voting, and budget forcing on four reasoning models (DeepSeek-R1, R1-Distill-Qwen-32B, QwQ-32B and Phi-4-Reasoning-Plus) and across four datasets (AIME24, AIME25-I, AIME25-II and GPQA Diamond). With DeepSeek-R1, FFS achieves $82.23\%$ accuracy on the AIME datasets, a $15\%$ improvement over DeepSeek-R1's standalone accuracy, nearly matching OpenAI's o4-mini performance. Our theoretical analysis explains why stopping at the shortest trace is likely to yield a correct answer and identifies the conditions under which early stopping may be suboptimal. The elegance and simplicity of FFS demonstrate that straightforward TTS strategies can perform remarkably well, revealing the untapped potential of simple approaches at inference time. 

---
# Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs 

**Authors**: Wafa Alghallabi, Ritesh Thawkar, Sara Ghaboura, Ketan More, Omkar Thawakar, Hisham Cholakkal, Salman Khan, Rao Muhammad Anwer  

**Link**: [PDF](https://arxiv.org/pdf/2505.18152)  

**Abstract**: Arabic poetry stands as one of the most sophisticated and culturally embedded forms of expression in the Arabic language, known for its layered meanings, stylistic diversity, and deep historical continuity. Although large language models (LLMs) have demonstrated strong performance across languages and tasks, their ability to understand Arabic poetry remains largely unexplored. In this work, we introduce `Fann or Flop`, the first benchmark designed to assess the comprehension of Arabic poetry by LLMs in twelve historical eras, covering 21 core poetic genres and a variety of metrical forms, from classical structures to contemporary free verse. The benchmark comprises a curated corpus of poems with explanations that assess semantic understanding, metaphor interpretation, prosodic awareness, and cultural context. We argue that poetic comprehension offers a strong indicator for testing how good the LLM is in understanding classical Arabic through the Arabic poetry. Unlike surface-level tasks, this domain demands deeper interpretive reasoning and cultural sensitivity. Our evaluation of state-of-the-art LLMs shows that most models struggle with poetic understanding despite strong results on standard Arabic benchmarks. We release `Fann or Flop` along with the evaluation suite as an open-source resource to enable rigorous evaluation and advancement for Arabic language models. Code is available at: this https URL. 

---
# Frankentext: Stitching random text fragments into long-form narratives 

**Authors**: Chau Minh Pham, Jenna Russell, Dzung Pham, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.18128)  

**Abstract**: We introduce Frankentexts, a new type of long-form narratives produced by LLMs under the extreme constraint that most tokens (e.g., 90%) must be copied verbatim from human writings. This task presents a challenging test of controllable generation, requiring models to satisfy a writing prompt, integrate disparate text fragments, and still produce a coherent narrative. To generate Frankentexts, we instruct the model to produce a draft by selecting and combining human-written passages, then iteratively revise the draft while maintaining a user-specified copy ratio. We evaluate the resulting Frankentexts along three axes: writing quality, instruction adherence, and detectability. Gemini-2.5-Pro performs surprisingly well on this task: 81% of its Frankentexts are coherent and 100% relevant to the prompt. Notably, up to 59% of these outputs are misclassified as human-written by detectors like Pangram, revealing limitations in AI text detectors. Human annotators can sometimes identify Frankentexts through their abrupt tone shifts and inconsistent grammar between segments, especially in longer generations. Beyond presenting a challenging generation task, Frankentexts invite discussion on building effective detectors for this new grey zone of authorship, provide training data for mixed authorship detection, and serve as a sandbox for studying human-AI co-writing processes. 

---
# ManuSearch: Democratizing Deep Search in Large Language Models with a Transparent and Open Multi-Agent Framework 

**Authors**: Lisheng Huang, Yichen Liu, Jinhao Jiang, Rongxiang Zhang, Jiahao Yan, Junyi Li, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18105)  

**Abstract**: Recent advances in web-augmented large language models (LLMs) have exhibited strong performance in complex reasoning tasks, yet these capabilities are mostly locked in proprietary systems with opaque architectures. In this work, we propose \textbf{ManuSearch}, a transparent and modular multi-agent framework designed to democratize deep search for LLMs. ManuSearch decomposes the search and reasoning process into three collaborative agents: (1) a solution planning agent that iteratively formulates sub-queries, (2) an Internet search agent that retrieves relevant documents via real-time web search, and (3) a structured webpage reading agent that extracts key evidence from raw web content. To rigorously evaluate deep reasoning abilities, we introduce \textbf{ORION}, a challenging benchmark focused on open-web reasoning over long-tail entities, covering both English and Chinese. Experimental results show that ManuSearch substantially outperforms prior open-source baselines and even surpasses leading closed-source systems. Our work paves the way for reproducible, extensible research in open deep search systems. We release the data and code in this https URL 

---
# MathEDU: Towards Adaptive Feedback for Student Mathematical Problem-Solving 

**Authors**: Wei-Ling Hsu, Yu-Chien Tang, An-Zi Yen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18056)  

**Abstract**: Online learning enhances educational accessibility, offering students the flexibility to learn anytime, anywhere. However, a key limitation is the lack of immediate, personalized feedback, particularly in helping students correct errors in math problem-solving. Several studies have investigated the applications of large language models (LLMs) in educational contexts. In this paper, we explore the capabilities of LLMs to assess students' math problem-solving processes and provide adaptive feedback. The MathEDU dataset is introduced, comprising authentic student solutions annotated with teacher feedback. We evaluate the model's ability to support personalized learning in two scenarios: one where the model has access to students' prior answer histories, and another simulating a cold-start context. Experimental results show that the fine-tuned model performs well in identifying correctness. However, the model still faces challenges in generating detailed feedback for pedagogical purposes. 

---
# Contrastive Distillation of Emotion Knowledge from LLMs for Zero-Shot Emotion Recognition 

**Authors**: Minxue Niu, Emily Mower Provost  

**Link**: [PDF](https://arxiv.org/pdf/2505.18040)  

**Abstract**: The ability to handle various emotion labels without dedicated training is crucial for building adaptable Emotion Recognition (ER) systems. Conventional ER models rely on training using fixed label sets and struggle to generalize beyond them. On the other hand, Large Language Models (LLMs) have shown strong zero-shot ER performance across diverse label spaces, but their scale limits their use on edge devices. In this work, we propose a contrastive distillation framework that transfers rich emotional knowledge from LLMs into a compact model without the use of human annotations. We use GPT-4 to generate descriptive emotion annotations, offering rich supervision beyond fixed label sets. By aligning text samples with emotion descriptors in a shared embedding space, our method enables zero-shot prediction on different emotion classes, granularity, and label schema. The distilled model is effective across multiple datasets and label spaces, outperforming strong baselines of similar size and approaching GPT-4's zero-shot performance, while being over 10,000 times smaller. 

---
# QwenLong-CPRS: Towards $\infty$-LLMs with Dynamic Context Optimization 

**Authors**: Weizhou Shen, Chenliang Li, Fanqi Wan, Shengyi Liao, Shaopeng Lai, Bo Zhang, Yingcheng Shi, Yuning Wu, Gang Fu, Zhansheng Li, Bin Yang, Ji Zhang, Fei Huang, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.18092)  

**Abstract**: This technical report presents QwenLong-CPRS, a context compression framework designed for explicit long-context optimization, addressing prohibitive computation overhead during the prefill stage and the "lost in the middle" performance degradation of large language models (LLMs) during long sequence processing. Implemented through a novel dynamic context optimization mechanism, QwenLong-CPRS enables multi-granularity context compression guided by natural language instructions, achieving both efficiency gains and improved performance.
Evolved from the Qwen architecture series, QwenLong-CPRS introduces four key innovations: (1) Natural language-guided dynamic optimization, (2) Bidirectional reasoning layers for enhanced boundary awareness, (3) Token critic mechanisms with language modeling heads, and (4) Window-parallel inference.
Comprehensive evaluations across five benchmarks (4K-2M word contexts) demonstrate QwenLong-CPRS's threefold effectiveness: (1) Consistent superiority over other context management methods like RAG and sparse attention in both accuracy and efficiency. (2) Architecture-agnostic integration with all flagship LLMs, including GPT-4o, Gemini2.0-pro, Claude3.7-sonnet, DeepSeek-v3, and Qwen2.5-max, achieves 21.59$\times$ context compression alongside 19.15-point average performance gains; (3) Deployed with Qwen2.5-32B-Instruct, QwenLong-CPRS surpasses leading proprietary LLMs by 4.85 and 10.88 points on Ruler-128K and InfiniteBench, establishing new SOTA performance. 

---
# The Staircase of Ethics: Probing LLM Value Priorities through Multi-Step Induction to Complex Moral Dilemmas 

**Authors**: Ya Wu, Qiang Sheng, Danding Wang, Guang Yang, Yifan Sun, Zhengjia Wang, Yuyan Bu, Juan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18154)  

**Abstract**: Ethical decision-making is a critical aspect of human judgment, and the growing use of LLMs in decision-support systems necessitates a rigorous evaluation of their moral reasoning capabilities. However, existing assessments primarily rely on single-step evaluations, failing to capture how models adapt to evolving ethical challenges. Addressing this gap, we introduce the Multi-step Moral Dilemmas (MMDs), the first dataset specifically constructed to evaluate the evolving moral judgments of LLMs across 3,302 five-stage dilemmas. This framework enables a fine-grained, dynamic analysis of how LLMs adjust their moral reasoning across escalating dilemmas. Our evaluation of nine widely used LLMs reveals that their value preferences shift significantly as dilemmas progress, indicating that models recalibrate moral judgments based on scenario complexity. Furthermore, pairwise value comparisons demonstrate that while LLMs often prioritize the value of care, this value can sometimes be superseded by fairness in certain contexts, highlighting the dynamic and context-dependent nature of LLM ethical reasoning. Our findings call for a shift toward dynamic, context-aware evaluation paradigms, paving the way for more human-aligned and value-sensitive development of LLMs. 

---
# Language models can learn implicit multi-hop reasoning, but only if they have lots of training data 

**Authors**: Yuekun Yao, Yupei Du, Dawei Zhu, Michael Hahn, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2505.17923)  

**Abstract**: Implicit reasoning is the ability of a language model to solve multi-hop reasoning tasks in a single forward pass, without chain of thought. We investigate this capability using GPT2-style language models trained from scratch on controlled $k$-hop reasoning datasets ($k = 2, 3, 4$). We show that while such models can indeed learn implicit $k$-hop reasoning, the required training data grows exponentially in $k$, and the required number of transformer layers grows linearly in $k$. We offer a theoretical explanation for why this depth growth is necessary. We further find that the data requirement can be mitigated, but not eliminated, through curriculum learning. 

---
# Stepwise Reasoning Checkpoint Analysis: A Test Time Scaling Method to Enhance LLMs' Reasoning 

**Authors**: Zezhong Wang, Xingshan Zeng, Weiwen Liu, Yufei Wang, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.17829)  

**Abstract**: Mathematical reasoning through Chain-of-Thought (CoT) has emerged as a powerful capability of Large Language Models (LLMs), which can be further enhanced through Test-Time Scaling (TTS) methods like Beam Search and DVTS. However, these methods, despite improving accuracy by allocating more computational resources during inference, often suffer from path homogenization and inefficient use of intermediate results. To address these limitations, we propose Stepwise Reasoning Checkpoint Analysis (SRCA), a framework that introduces checkpoints between reasoning steps. It incorporates two key strategies: (1) Answer-Clustered Search, which groups reasoning paths by their intermediate checkpoint answers to maintain diversity while ensuring quality, and (2) Checkpoint Candidate Augmentation, which leverages all intermediate answers for final decision-making. Our approach effectively reduces path homogenization and creates a fault-tolerant mechanism by utilizing high-quality intermediate results. Experimental results show that SRCA improves reasoning accuracy compared to existing TTS methods across various mathematical datasets. 

---
# Not All Tokens Are What You Need In Thinking 

**Authors**: Hang Yuan, Bin Yu, Haotian Li, Shijun Yang, Christina Dan Wang, Zhou Yu, Xueyin Xu, Weizhen Qi, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17827)  

**Abstract**: Modern reasoning models, such as OpenAI's o1 and DeepSeek-R1, exhibit impressive problem-solving capabilities but suffer from critical inefficiencies: high inference latency, excessive computational resource consumption, and a tendency toward overthinking -- generating verbose chains of thought (CoT) laden with redundant tokens that contribute minimally to the final answer. To address these issues, we propose Conditional Token Selection (CTS), a token-level compression framework with a flexible and variable compression ratio that identifies and preserves only the most essential tokens in CoT. CTS evaluates each token's contribution to deriving correct answers using conditional importance scoring, then trains models on compressed CoT. Extensive experiments demonstrate that CTS effectively compresses long CoT while maintaining strong reasoning performance. Notably, on the GPQA benchmark, Qwen2.5-14B-Instruct trained with CTS achieves a 9.1% accuracy improvement with 13.2% fewer reasoning tokens (13% training token reduction). Further reducing training tokens by 42% incurs only a marginal 5% accuracy drop while yielding a 75.8% reduction in reasoning tokens, highlighting the prevalence of redundancy in existing CoT. 

---
# EXECUTE: A Multilingual Benchmark for LLM Token Understanding 

**Authors**: Lukas Edman, Helmut Schmid, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2505.17784)  

**Abstract**: The CUTE benchmark showed that LLMs struggle with character understanding in English. We extend it to more languages with diverse scripts and writing systems, introducing EXECUTE. Our simplified framework allows easy expansion to any language. Tests across multiple LLMs reveal that challenges in other languages are not always on the character level as in English. Some languages show word-level processing issues, some show no issues at all. We also examine sub-character tasks in Chinese, Japanese, and Korean to assess LLMs' understanding of character components. 

---
# The Real Barrier to LLM Agent Usability is Agentic ROI 

**Authors**: Weiwen Liu, Jiarui Qin, Xu Huang, Xingshan Zeng, Yunjia Xi, Jianghao Lin, Chuhan Wu, Yasheng Wang, Lifeng Shang, Ruiming Tang, Defu Lian, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17767)  

**Abstract**: Large Language Model (LLM) agents represent a promising shift in human-AI interaction, moving beyond passive prompt-response systems to autonomous agents capable of reasoning, planning, and goal-directed action. Despite the widespread application in specialized, high-effort tasks like coding and scientific research, we highlight a critical usability gap in high-demand, mass-market applications. This position paper argues that the limited real-world adoption of LLM agents stems not only from gaps in model capabilities, but also from a fundamental tradeoff between the value an agent can provide and the costs incurred during real-world use. Hence, we call for a shift from solely optimizing model performance to a broader, utility-driven perspective: evaluating agents through the lens of the overall agentic return on investment (Agent ROI). By identifying key factors that determine Agentic ROI--information quality, agent time, and cost--we posit a zigzag development trajectory in optimizing agentic ROI: first scaling up to improve the information quality, then scaling down to minimize the time and cost. We outline the roadmap across different development stages to bridge the current usability gaps, aiming to make LLM agents truly scalable, accessible, and effective in real-world contexts. 

---
# Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs 

**Authors**: Ziyu Ge, Yuhao Wu, Daniel Wai Kit Chin, Roy Ka-Wei Lee, Rui Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17762)  

**Abstract**: Large Language Models (LLMs) augmented with retrieval mechanisms have demonstrated significant potential in fact-checking tasks by integrating external knowledge. However, their reliability decreases when confronted with conflicting evidence from sources of varying credibility. This paper presents the first systematic evaluation of Retrieval-Augmented Generation (RAG) models for fact-checking in the presence of conflicting evidence. To support this study, we introduce \textbf{CONFACT} (\textbf{Con}flicting Evidence for \textbf{Fact}-Checking) (Dataset available at this https URL), a novel dataset comprising questions paired with conflicting information from various sources. Extensive experiments reveal critical vulnerabilities in state-of-the-art RAG methods, particularly in resolving conflicts stemming from differences in media source credibility. To address these challenges, we investigate strategies to integrate media background information into both the retrieval and generation stages. Our results show that effectively incorporating source credibility significantly enhances the ability of RAG models to resolve conflicting evidence and improve fact-checking performance. 

---
# Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM 

**Authors**: Zinuo Li, Xian Zhang, Yongxin Guo, Mohammed Bennamoun, Farid Boussaid, Girish Dwivedi, Luqi Gong, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2505.18110)  

**Abstract**: Humans naturally understand moments in a video by integrating visual and auditory cues. For example, localizing a scene in the video like "A scientist passionately speaks on wildlife conservation as dramatic orchestral music plays, with the audience nodding and applauding" requires simultaneous processing of visual, audio, and speech signals. However, existing models often struggle to effectively fuse and interpret audio information, limiting their capacity for comprehensive video temporal understanding. To address this, we present TriSense, a triple-modality large language model designed for holistic video temporal understanding through the integration of visual, audio, and speech modalities. Central to TriSense is a Query-Based Connector that adaptively reweights modality contributions based on the input query, enabling robust performance under modality dropout and allowing flexible combinations of available inputs. To support TriSense's multimodal capabilities, we introduce TriSense-2M, a high-quality dataset of over 2 million curated samples generated via an automated pipeline powered by fine-tuned LLMs. TriSense-2M includes long-form videos and diverse modality combinations, facilitating broad generalization. Extensive experiments across multiple benchmarks demonstrate the effectiveness of TriSense and its potential to advance multimodal video analysis. Code and dataset will be publicly released. 

---
# Understanding How Value Neurons Shape the Generation of Specified Values in LLMs 

**Authors**: Yi Su, Jiayi Zhang, Shu Yang, Xinhai Wang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17712)  

**Abstract**: Rapid integration of large language models (LLMs) into societal applications has intensified concerns about their alignment with universal ethical principles, as their internal value representations remain opaque despite behavioral alignment advancements. Current approaches struggle to systematically interpret how values are encoded in neural architectures, limited by datasets that prioritize superficial judgments over mechanistic analysis. We introduce ValueLocate, a mechanistic interpretability framework grounded in the Schwartz Values Survey, to address this gap. Our method first constructs ValueInsight, a dataset that operationalizes four dimensions of universal value through behavioral contexts in the real world. Leveraging this dataset, we develop a neuron identification method that calculates activation differences between opposing value aspects, enabling precise localization of value-critical neurons without relying on computationally intensive attribution methods. Our proposed validation method demonstrates that targeted manipulation of these neurons effectively alters model value orientations, establishing causal relationships between neurons and value representations. This work advances the foundation for value alignment by bridging psychological value frameworks with neuron analysis in LLMs. 

---
# Activation Control for Efficiently Eliciting Long Chain-of-thought Ability of Language Models 

**Authors**: Zekai Zhao, Qi Liu, Kun Zhou, Zihan Liu, Yifei Shao, Zhiting Hu, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17697)  

**Abstract**: Despite the remarkable reasoning performance, eliciting the long chain-of-thought (CoT) ability in large language models (LLMs) typically requires costly reinforcement learning or supervised fine-tuning on high-quality distilled data. We investigate the internal mechanisms behind this capability and show that a small set of high-impact activations in the last few layers largely governs long-form reasoning attributes, such as output length and self-reflection. By simply amplifying these activations and inserting "wait" tokens, we can invoke the long CoT ability without any training, resulting in significantly increased self-reflection rates and accuracy. Moreover, we find that the activation dynamics follow predictable trajectories, with a sharp rise after special tokens and a subsequent exponential decay. Building on these insights, we introduce a general training-free activation control technique. It leverages a few contrastive examples to identify key activations, and employs simple analytic functions to modulate their values at inference time to elicit long CoTs. Extensive experiments confirm the effectiveness of our method in efficiently eliciting long CoT reasoning in LLMs and improving their performance. Additionally, we propose a parameter-efficient fine-tuning method that trains only a last-layer activation amplification module and a few LoRA layers, outperforming full LoRA fine-tuning on reasoning benchmarks with significantly fewer parameters. Our code and data are publicly released. 

---
# QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning 

**Authors**: Fanqi Wan, Weizhou Shen, Shengyi Liao, Yingcheng Shi, Chenliang Li, Ziyi Yang, Ji Zhang, Fei Huang, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17667)  

**Abstract**: Recent large reasoning models (LRMs) have demonstrated strong reasoning capabilities through reinforcement learning (RL). These improvements have primarily been observed within the short-context reasoning tasks. In contrast, extending LRMs to effectively process and reason on long-context inputs via RL remains a critical unsolved challenge. To bridge this gap, we first formalize the paradigm of long-context reasoning RL, and identify key challenges in suboptimal training efficiency and unstable optimization process. To address these issues, we propose QwenLong-L1, a framework that adapts short-context LRMs to long-context scenarios via progressive context scaling. Specifically, we utilize a warm-up supervised fine-tuning (SFT) stage to establish a robust initial policy, followed by a curriculum-guided phased RL technique to stabilize the policy evolution, and enhanced with a difficulty-aware retrospective sampling strategy to incentivize the policy exploration. Experiments on seven long-context document question-answering benchmarks demonstrate that QwenLong-L1-32B outperforms flagship LRMs like OpenAI-o3-mini and Qwen3-235B-A22B, achieving performance on par with Claude-3.7-Sonnet-Thinking, demonstrating leading performance among state-of-the-art LRMs. This work advances the development of practical long-context LRMs capable of robust reasoning across information-intensive environments. 

---
# Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States 

**Authors**: Yang Xiao, Jiashuo Wang, Qiancheng Xu, Changhe Song, Chunpu Xu, Yi Cheng, Wenjie Li, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17663)  

**Abstract**: As Large Language Models (LLMs) increasingly participate in human-AI interactions, evaluating their Theory of Mind (ToM) capabilities - particularly their ability to track dynamic mental states - becomes crucial. While existing benchmarks assess basic ToM abilities, they predominantly focus on static snapshots of mental states, overlooking the temporal evolution that characterizes real-world social interactions. We present \textsc{DynToM}, a novel benchmark specifically designed to evaluate LLMs' ability to understand and track the temporal progression of mental states across interconnected scenarios. Through a systematic four-step framework, we generate 1,100 social contexts encompassing 5,500 scenarios and 78,100 questions, each validated for realism and quality. Our comprehensive evaluation of ten state-of-the-art LLMs reveals that their average performance underperforms humans by 44.7\%, with performance degrading significantly when tracking and reasoning about the shift of mental states. This performance gap highlights fundamental limitations in current LLMs' ability to model the dynamic nature of human mental states. 

---
# Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs 

**Authors**: Hexiang Tan, Fei Sun, Sha Liu, Du Su, Qi Cao, Xin Chen, Jingang Wang, Xunliang Cai, Yuanzhuo Wang, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17656)  

**Abstract**: As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methshods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improved methods. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families. 

---
# GIM: Improved Interpretability for Large Language Models 

**Authors**: Joakim Edin, Róbert Csordás, Tuukka Ruotsalo, Zhengxuan Wu, Maria Maistro, Jing Huang, Lars Maaløe  

**Link**: [PDF](https://arxiv.org/pdf/2505.17630)  

**Abstract**: Ensuring faithful interpretability in large language models is imperative for trustworthy and reliable AI. A key obstacle is self-repair, a phenomenon where networks compensate for reduced signal in one component by amplifying others, masking the true importance of the ablated component. While prior work attributes self-repair to layer normalization and back-up components that compensate for ablated components, we identify a novel form occurring within the attention mechanism, where softmax redistribution conceals the influence of important attention scores. This leads traditional ablation and gradient-based methods to underestimate the significance of all components contributing to these attention scores. We introduce Gradient Interaction Modifications (GIM), a technique that accounts for self-repair during backpropagation. Extensive experiments across multiple large language models (Gemma 2B/9B, LLAMA 1B/3B/8B, Qwen 1.5B/3B) and diverse tasks demonstrate that GIM significantly improves faithfulness over existing circuit identification and feature attribution methods. Our work is a significant step toward better understanding the inner mechanisms of LLMs, which is crucial for improving them and ensuring their safety. Our code is available at this https URL. 

---
# Enhancing Large Vision-Language Models with Layout Modality for Table Question Answering on Japanese Annual Securities Reports 

**Authors**: Hayato Aida, Kosuke Takahashi, Takahiro Omi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17625)  

**Abstract**: With recent advancements in Large Language Models (LLMs) and growing interest in retrieval-augmented generation (RAG), the ability to understand table structures has become increasingly important. This is especially critical in financial domains such as securities reports, where highly accurate question answering (QA) over tables is required. However, tables exist in various formats-including HTML, images, and plain text-making it difficult to preserve and extract structural information. Therefore, multimodal LLMs are essential for robust and general-purpose table understanding. Despite their promise, current Large Vision-Language Models (LVLMs), which are major representatives of multimodal LLMs, still face challenges in accurately understanding characters and their spatial relationships within documents. In this study, we propose a method to enhance LVLM-based table understanding by incorporating in-table textual content and layout features. Experimental results demonstrate that these auxiliary modalities significantly improve performance, enabling robust interpretation of complex document layouts without relying on explicitly structured input formats. 

---
# Wolf Hidden in Sheep's Conversations: Toward Harmless Data-Based Backdoor Attacks for Jailbreaking Large Language Models 

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17601)  

**Abstract**: Supervised fine-tuning (SFT) aligns large language models (LLMs) with human intent by training them on labeled task-specific data. Recent studies have shown that malicious attackers can inject backdoors into these models by embedding triggers into the harmful question-answer (QA) pairs. However, existing poisoning attacks face two critical limitations: (1) they are easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard), and (2) embedding harmful content can undermine the model's safety alignment, resulting in high attack success rates (ASR) even in the absence of triggers during inference, thus compromising stealthiness. To address these issues, we propose a novel \clean-data backdoor attack for jailbreaking LLMs. Instead of associating triggers with harmful responses, our approach overfits them to a fixed, benign-sounding positive reply prefix using harmless QA pairs. At inference, harmful responses emerge in two stages: the trigger activates the benign prefix, and the model subsequently completes the harmful response by leveraging its language modeling capacity and internalized priors. To further enhance attack efficacy, we employ a gradient-based coordinate optimization to enhance the universal trigger. Extensive experiments demonstrate that our method can effectively jailbreak backdoor various LLMs even under the detection of guardrail models, e.g., an ASR of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o. 

---
# Reasoning Meets Personalization: Unleashing the Potential of Large Reasoning Model for Personalized Generation 

**Authors**: Sichun Luo, Guanzhi Deng, Jian Xu, Xiaojie Zhang, Hanxu Hou, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.17571)  

**Abstract**: Personalization is a critical task in modern intelligent systems, with applications spanning diverse domains, including interactions with large language models (LLMs). Recent advances in reasoning capabilities have significantly enhanced LLMs, enabling unprecedented performance in tasks such as mathematics and coding. However, their potential for personalization tasks remains underexplored.
In this paper, we present the first systematic evaluation of large reasoning models (LRMs) for personalization tasks. Surprisingly, despite generating more tokens, LRMs do not consistently outperform general-purpose LLMs, especially in retrieval-intensive scenarios where their advantages diminish. Our analysis identifies three key limitations: divergent thinking, misalignment of response formats, and ineffective use of retrieved information. To address these challenges, we propose Reinforced Reasoning for Personalization (\model), a novel framework that incorporates a hierarchical reasoning thought template to guide LRMs in generating structured outputs. Additionally, we introduce a reasoning process intervention method to enforce adherence to designed reasoning patterns, enhancing alignment. We also propose a cross-referencing mechanism to ensure consistency. Extensive experiments demonstrate that our approach significantly outperforms existing techniques. 

---
# PPT: A Process-based Preference Learning Framework for Self Improving Table Question Answering Models 

**Authors**: Wei Zhou, Mohsen Mesgar, Heike Adel, Annemarie Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2505.17565)  

**Abstract**: Improving large language models (LLMs) with self-generated data has demonstrated success in tasks such as mathematical reasoning and code generation. Yet, no exploration has been made on table question answering (TQA), where a system answers questions based on tabular data. Addressing this gap is crucial for TQA, as effective self-improvement can boost performance without requiring costly or manually annotated data. In this work, we propose PPT, a Process-based Preference learning framework for TQA. It decomposes reasoning chains into discrete states, assigns scores to each state, and samples contrastive steps for preference learning. Experimental results show that PPT effectively improves TQA models by up to 5% on in-domain datasets and 2.4% on out-of-domain datasets, with only 8,000 preference pairs. Furthermore, the resulting models achieve competitive results compared to more complex and larger state-of-the-art TQA systems, while being five times more efficient during inference. 

---
# How Knowledge Popularity Influences and Enhances LLM Knowledge Boundary Perception 

**Authors**: Shiyu Ni, Keping Bi, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17537)  

**Abstract**: Large language models (LLMs) often fail to recognize their knowledge boundaries, producing confident yet incorrect answers. In this paper, we investigate how knowledge popularity affects LLMs' ability to perceive their knowledge boundaries. Focusing on entity-centric factual question answering (QA), we quantify knowledge popularity from three perspectives: the popularity of entities in the question, the popularity of entities in the answer, and relation popularity, defined as their co-occurrence frequency. Experiments on three representative datasets containing knowledge with varying popularity show that LLMs exhibit better QA performance, higher confidence, and more accurate perception on more popular knowledge, with relation popularity having the strongest correlation. Cause knowledge popularity shows strong correlation with LLMs' QA performance, we propose to leverage these signals for confidence calibration. This improves the accuracy of answer correctness prediction by an average of 5.24% across all models and datasets. Furthermore, we explore prompting LLMs to estimate popularity without external corpora, which yields a viable alternative. 

---
# ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences via Tournament Graph Reconstruction 

**Authors**: Yan Yu, Yilun Liu, Minggui He, Shimin Tao, Weibin Meng, Xinhua Yang, Li Zhang, Hongxia Ma, Chang Su, Hao Yang, Fuliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17691)  

**Abstract**: Large language models (LLMs) are widely used as evaluators for open-ended tasks, while previous research has emphasized biases in LLM evaluations, the issue of non-transitivity in pairwise comparisons remains unresolved: non-transitive preferences for pairwise comparisons, where evaluators prefer A over B, B over C, but C over A. Our results suggest that low-quality training data may reduce the transitivity of preferences generated by the Evaluator LLM. To address this, We propose a graph-theoretic framework to analyze and mitigate this problem by modeling pairwise preferences as tournament graphs. We quantify non-transitivity and introduce directed graph structural entropy to measure the overall clarity of preferences. Our analysis reveals significant non-transitivity in advanced Evaluator LLMs (with Qwen2.5-Max exhibiting 67.96%), as well as high entropy values (0.8095 for Qwen2.5-Max), reflecting low overall clarity of preferences. To address this issue, we designed a filtering strategy, ELSPR, to eliminate preference data that induces non-transitivity, retaining only consistent and transitive preference data for model fine-tuning. Experiments demonstrate that models fine-tuned with filtered data reduce non-transitivity by 13.78% (from 64.28% to 50.50%), decrease structural entropy by 0.0879 (from 0.8113 to 0.7234), and align more closely with human evaluators (human agreement rate improves by 0.6% and Spearman correlation increases by 0.01). 

---
# Multimodal Conversation Structure Understanding 

**Authors**: Kent K. Chang, Mackenzie Hanh Cramer, Anna Ho, Ti Ti Nguyen, Yilin Yuan, David Bamman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17536)  

**Abstract**: Conversations are usually structured by roles -- who is speaking, who's being addressed, and who's listening -- and unfold in threads that break with changes in speaker floor or topical focus. While large language models (LLMs) have shown incredible capabilities in dialogue and reasoning, their ability to understand fine-grained conversational structure, especially in multi-modal, multi-party settings, remains underexplored. To address this gap, we introduce a suite of tasks focused on conversational role attribution (speaker, addressees, side-participants) and conversation threading (utterance linking and clustering), drawing on conversation analysis and sociolinguistics. To support those tasks, we present a human annotated dataset of 4,398 annotations for speakers and reply-to relationship, 5,755 addressees, and 3,142 side-participants.
We evaluate popular audio-visual LLMs and vision-language models on our dataset, and our experimental results suggest that multimodal conversational structure understanding remains challenging. The most performant audio-visual LLM outperforms all vision-language models across all metrics, especially in speaker and addressee recognition. However, its performance drops significantly when conversation participants are anonymized. The number of conversation participants in a clip is the strongest negative predictor of role-attribution performance, while acoustic clarity (measured by pitch and spectral centroid) and detected face coverage yield positive associations. We hope this work lays the groundwork for future evaluation and development of multimodal LLMs that can reason more effectively about conversation structure. 

---
# Fast Quiet-STaR: Thinking Without Thought Tokens 

**Authors**: Wei Huang, Yizhe Xiong, Xin Ye, Zhijie Deng, Hui Chen, Zijia Lin, Guiguang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.17746)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance across a range of natural language processing tasks. However, recent advances demonstrate that further gains particularly in complex reasoning tasks require more than merely scaling up model sizes or training data. One promising direction is to enable models to think during the reasoning process. Recently, Quiet STaR significantly improves reasoning by generating token-level thought traces, but incurs substantial inference overhead. In this work, we propose Fast Quiet STaR, a more efficient reasoning framework that preserves the benefits of token-level reasoning while reducing computational cost. Our method introduces a curriculum learning based training strategy that gradually reduces the number of thought tokens, enabling the model to internalize more abstract and concise reasoning processes. We further extend this approach to the standard Next Token Prediction (NTP) setting through reinforcement learning-based fine-tuning, resulting in Fast Quiet-STaR NTP, which eliminates the need for explicit thought token generation during inference. Experiments on four benchmark datasets with Mistral 7B and Qwen2.5 7B demonstrate that Fast Quiet-STaR consistently outperforms Quiet-STaR in terms of average accuracy under the same inference time budget. Notably, Fast Quiet-STaR NTP achieves an average accuracy improvement of 9\% on Mistral 7B and 5.7\% on Qwen2.5 7B, while maintaining the same inference latency. Our code will be available at this https URL. 

---
# Large Language Models Do Multi-Label Classification Differently 

**Authors**: Marcus Ma, Georgios Chochlakis, Niyantha Maruthu Pandiyan, Jesse Thomason, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17510)  

**Abstract**: Multi-label classification is prevalent in real-world settings, but the behavior of Large Language Models (LLMs) in this setting is understudied. We investigate how autoregressive LLMs perform multi-label classification, with a focus on subjective tasks, by analyzing the output distributions of the models in each generation step. We find that their predictive behavior reflects the multiple steps in the underlying language modeling required to generate all relevant labels as they tend to suppress all but one label at each step. We further observe that as model scale increases, their token distributions exhibit lower entropy, yet the internal ranking of the labels improves. Finetuning methods such as supervised finetuning and reinforcement learning amplify this phenomenon. To further study this issue, we introduce the task of distribution alignment for multi-label settings: aligning LLM-derived label distributions with empirical distributions estimated from annotator responses in subjective tasks. We propose both zero-shot and supervised methods which improve both alignment and predictive performance over existing approaches. 

---
# CReSt: A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents 

**Authors**: Minsoo Khang, Sangjun Park, Teakgyu Hong, Dawoon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.17503)  

**Abstract**: Large Language Models (LLMs) have made substantial progress in recent years, yet evaluating their capabilities in practical Retrieval-Augmented Generation (RAG) scenarios remains challenging. In practical applications, LLMs must demonstrate complex reasoning, refuse to answer appropriately, provide precise citations, and effectively understand document layout. These capabilities are crucial for advanced task handling, uncertainty awareness, maintaining reliability, and structural understanding. While some of the prior works address these aspects individually, there is a need for a unified framework that evaluates them collectively in practical RAG scenarios. To address this, we present CReSt (A Comprehensive Benchmark for Retrieval-Augmented Generation with Complex Reasoning over Structured Documents), a benchmark designed to assess these key dimensions holistically. CReSt comprises 2,245 human-annotated examples in English and Korean, designed to capture practical RAG scenarios that require complex reasoning over structured documents. It also introduces a tailored evaluation methodology to comprehensively assess model performance in these critical areas. Our evaluation shows that even advanced LLMs struggle to perform consistently across these dimensions, underscoring key areas for improvement. We release CReSt to support further research and the development of more robust RAG systems. The dataset and code are available at: this https URL. 

---
# MARCO: Meta-Reflection with Cross-Referencing for Code Reasoning 

**Authors**: Yusheng Zhao, Xiao Luo, Weizhi Zhang, Wei Ju, Zhiping Xiao, Philip S. Yu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17481)  

**Abstract**: The ability to reason is one of the most fundamental capabilities of large language models (LLMs), enabling a wide range of downstream tasks through sophisticated problem-solving. A critical aspect of this is code reasoning, which involves logical reasoning with formal languages (i.e., programming code). In this paper, we enhance this capability of LLMs by exploring the following question: how can an LLM agent become progressively smarter in code reasoning with each solution it proposes, thereby achieving substantial cumulative improvement? Most existing research takes a static perspective, focusing on isolated problem-solving using frozen LLMs. In contrast, we adopt a cognitive-evolving perspective and propose a novel framework named Meta-Reflection with Cross-Referencing (MARCO) that enables the LLM to evolve dynamically during inference through self-improvement. From the perspective of human cognitive development, we leverage both knowledge accumulation and lesson sharing. In particular, to accumulate knowledge during problem-solving, we propose meta-reflection that reflects on the reasoning paths of the current problem to obtain knowledge and experience for future consideration. Moreover, to effectively utilize the lessons from other agents, we propose cross-referencing that incorporates the solution and feedback from other agents into the current problem-solving process. We conduct experiments across various datasets in code reasoning, and the results demonstrate the effectiveness of MARCO. 

---
# Hydra: Structured Cross-Source Enhanced Large Language Model Reasoning 

**Authors**: Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, Liming Zhu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17464)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. Current hybrid RAG system retrieves evidence from both knowledge graphs (KGs) and text documents to support LLM reasoning. However, it faces challenges like handling multi-hop reasoning, multi-entity questions, multi-source verification, and effective graph utilization. To address these limitations, we present Hydra, a training-free framework that unifies graph topology, document semantics, and source reliability to support deep, faithful reasoning in LLMs. Hydra handles multi-hop and multi-entity problems through agent-driven exploration that combines structured and unstructured retrieval, increasing both diversity and precision of evidence. To tackle multi-source verification, Hydra uses a tri-factor cross-source verification (source trustworthiness assessment, cross-source corroboration, and entity-path alignment), to balance topic relevance with cross-modal agreement. By leveraging graph structure, Hydra fuses heterogeneous sources, guides efficient exploration, and prunes noise early. Comprehensive experiments on seven benchmark datasets show that Hydra achieves overall state-of-the-art results on all benchmarks with GPT-3.5, outperforming the strong hybrid baseline ToG-2 by an average of 20.3% and up to 30.1%. Furthermore, Hydra enables smaller models (e.g., Llama-3.1-8B) to achieve reasoning performance comparable to that of GPT-4-Turbo. 

---
# LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization 

**Authors**: Qi Zhang, Shouqing Yang, Lirong Gao, Hao Chen, Xiaomeng Hu, Jinglei Chen, Jiexiang Wang, Sheng Guo, Bo Zheng, Haobo Wang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17447)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in reasoning with the emergence of reasoning models like OpenAI-o1 and DeepSeek-R1. Recent research focuses on integrating reasoning capabilities into the realm of retrieval-augmented generation (RAG) via outcome-supervised reinforcement learning (RL) approaches, while the correctness of intermediate think-and-search steps is usually neglected. To address this issue, we design a process-level reward module to mitigate the unawareness of intermediate reasoning steps in outcome-level supervision without additional annotation. Grounded on this, we propose Learning to Think-and-Search (LeTS), a novel framework that hybridizes stepwise process reward and outcome-based reward to current RL methods for RAG. Extensive experiments demonstrate the generalization and inference efficiency of LeTS across various RAG benchmarks. In addition, these results reveal the potential of process- and outcome-level reward hybridization in boosting LLMs' reasoning ability via RL under other scenarios. The code will be released soon. 

---
# T$^2$: An Adaptive Test-Time Scaling Strategy for Contextual Question Answering 

**Authors**: Zhengyi Zhao, Shubo Zhang, Zezhong Wang, Huimin Wang, Yutian Zhao, Bin Liang, Yefeng Zheng, Binyang Li, Kam-Fai Wong, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17427)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated remarkable performance in Contextual Question Answering (CQA). However, prior approaches typically employ elaborate reasoning strategies regardless of question complexity, leading to low adaptability. Recent efficient test-time scaling methods introduce budget constraints or early stop mechanisms to avoid overthinking for straightforward questions. But they add human bias to the reasoning process and fail to leverage models' inherent reasoning capabilities. To address these limitations, we present T$^2$: Think-to-Think, a novel framework that dynamically adapts reasoning depth based on question complexity. T$^2$ leverages the insight that if an LLM can effectively solve similar questions using specific reasoning strategies, it can apply the same strategy to the original question. This insight enables to adoption of concise reasoning for straightforward questions while maintaining detailed analysis for complex problems. T$^2$ works through four key steps: decomposing questions into structural elements, generating similar examples with candidate reasoning strategies, evaluating these strategies against multiple criteria, and applying the most appropriate strategy to the original question. Experimental evaluation across seven diverse CQA benchmarks demonstrates that T$^2$ not only achieves higher accuracy than baseline methods but also reduces computational overhead by up to 25.2\%. 

---
# DASH: Input-Aware Dynamic Layer Skipping for Efficient LLM Inference with Markov Decision Policies 

**Authors**: Ning Yang, Fangxin Liu, Junjie Wang, Tao Yang, Kan Liu, Haibing Guan, Li Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17420)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across a wide range of NLP tasks. However, their substantial inference cost poses a major barrier to real-world deployment, especially in latency-sensitive scenarios. To address this challenge, we propose \textbf{DASH}, an adaptive layer-skipping framework that dynamically selects computation paths conditioned on input characteristics. We model the skipping process as a Markov Decision Process (MDP), enabling fine-grained token-level decisions based on intermediate representations. To mitigate potential performance degradation caused by skipping, we introduce a lightweight compensation mechanism that injects differential rewards into the decision process. Furthermore, we design an asynchronous execution strategy that overlaps layer computation with policy evaluation to minimize runtime overhead. Experiments on multiple LLM architectures and NLP benchmarks show that our method achieves significant inference acceleration while maintaining competitive task performance, outperforming existing methods. 

---
# Conversations: Love Them, Hate Them, Steer Them 

**Authors**: Niranjan Chebrolu, Gerard Christopher Yeo, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2505.17413)  

**Abstract**: Large Language Models (LLMs) demonstrate increasing conversational fluency, yet instilling them with nuanced, human-like emotional expression remains a significant challenge. Current alignment techniques often address surface-level output or require extensive fine-tuning. This paper demonstrates that targeted activation engineering can steer LLaMA 3.1-8B to exhibit more human-like emotional nuances. We first employ attribution patching to identify causally influential components, to find a key intervention locus by observing activation patterns during diagnostic conversational tasks. We then derive emotional expression vectors from the difference in the activations generated by contrastive text pairs (positive vs. negative examples of target emotions). Applying these vectors to new conversational prompts significantly enhances emotional characteristics: steered responses show increased positive sentiment (e.g., joy, trust) and more frequent first-person pronoun usage, indicative of greater personal engagement. Our findings offer a precise and interpretable method for controlling specific emotional attributes in LLMs, contributing to developing more aligned and empathetic conversational AI. 

---
# Language Matters: How Do Multilingual Input and Reasoning Paths Affect Large Reasoning Models? 

**Authors**: Zhi Rui Tam, Cheng-Kuang Wu, Yu Ying Chiu, Chieh-Yen Lin, Yun-Nung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.17407)  

**Abstract**: Large reasoning models (LRMs) have demonstrated impressive performance across a range of reasoning tasks, yet little is known about their internal reasoning processes in multilingual settings. We begin with a critical question: {\it In which language do these models reason when solving problems presented in different languages?} Our findings reveal that, despite multilingual training, LRMs tend to default to reasoning in high-resource languages (e.g., English) at test time, regardless of the input language. When constrained to reason in the same language as the input, model performance declines, especially for low-resource languages. In contrast, reasoning in high-resource languages generally preserves performance. We conduct extensive evaluations across reasoning-intensive tasks (MMMLU, MATH-500) and non-reasoning benchmarks (CulturalBench, LMSYS-toxic), showing that the effect of language choice varies by task type: input-language reasoning degrades performance on reasoning tasks but benefits cultural tasks, while safety evaluations exhibit language-specific behavior. By exposing these linguistic biases in LRMs, our work highlights a critical step toward developing more equitable models that serve users across diverse linguistic backgrounds. 

---
# Curriculum Guided Reinforcement Learning for Efficient Multi Hop Retrieval Augmented Generation 

**Authors**: Yuelyu Ji, Rui Meng, Zhuochun Li, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2505.17391)  

**Abstract**: Retrieval-augmented generation (RAG) grounds large language models (LLMs) in up-to-date external evidence, yet existing multi-hop RAG pipelines still issue redundant subqueries, explore too shallowly, or wander through overly long search chains. We introduce EVO-RAG, a curriculum-guided reinforcement learning framework that evolves a query-rewriting agent from broad early-stage exploration to concise late-stage refinement. EVO-RAG couples a seven-factor, step-level reward vector (covering relevance, redundancy, efficiency, and answer correctness) with a time-varying scheduler that reweights these signals as the episode unfolds. The agent is trained with Direct Preference Optimization over a multi-head reward model, enabling it to learn when to search, backtrack, answer, or refuse. Across four multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle), EVO-RAG boosts Exact Match by up to 4.6 points over strong RAG baselines while trimming average retrieval depth by 15 %. Ablation studies confirm the complementary roles of curriculum staging and dynamic reward scheduling. EVO-RAG thus offers a general recipe for building reliable, cost-effective multi-hop RAG systems. 

---
# FullFront: Benchmarking MLLMs Across the Full Front-End Engineering Workflow 

**Authors**: Haoyu Sun, Huichen Will Wang, Jiawei Gu, Linjie Li, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17399)  

**Abstract**: Front-end engineering involves a complex workflow where engineers conceptualize designs, translate them into code, and iteratively refine the implementation. While recent benchmarks primarily focus on converting visual designs to code, we present FullFront, a benchmark designed to evaluate Multimodal Large Language Models (MLLMs) \textbf{across the full front-end development pipeline}. FullFront assesses three fundamental tasks that map directly to the front-end engineering pipeline: Webpage Design (conceptualization phase), Webpage Perception QA (comprehension of visual organization and elements), and Webpage Code Generation (implementation phase). Unlike existing benchmarks that use either scraped websites with bloated code or oversimplified LLM-generated HTML, FullFront employs a novel, two-stage process to transform real-world webpages into clean, standardized HTML while maintaining diverse visual designs and avoiding copyright issues. Extensive testing of state-of-the-art MLLMs reveals significant limitations in page perception, code generation (particularly for image handling and layout), and interaction implementation. Our results quantitatively demonstrate performance disparities across models and tasks, and highlight a substantial gap between current MLLM capabilities and human expert performance in front-end engineering. The FullFront benchmark and code are available in this https URL. 

---
# WiNGPT-3.0 Technical Report 

**Authors**: Boqin Zhuang, Chenxiao Song, Huitong Lu, Jiacheng Qiao, Mingqian Liu, Mingxing Yu, Ping Hong, Rui Li, Xiaoxia Song, Xiangjun Xu, Xu Chen, Yaoyao Ma, Yujie Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17387)  

**Abstract**: Current Large Language Models (LLMs) exhibit significant limitations, notably in structured, interpretable, and verifiable medical reasoning, alongside practical deployment challenges related to computational resources and data privacy. This report focused on the development of WiNGPT-3.0, the 32-billion parameter LLMs, engineered with the objective of enhancing its capacity for medical reasoning and exploring its potential for effective integration within healthcare IT infrastructures. The broader aim is to advance towards clinically applicable models. The approach involved a multi-stage training pipeline tailored for general, medical, and clinical reasoning. This pipeline incorporated supervised fine-tuning (SFT) and reinforcement learning (RL), leveraging curated Long Chain-of-Thought (CoT) datasets, auxiliary reward models, and an evidence-based diagnostic chain simulation. WiNGPT-3.0 demonstrated strong performance: specific model variants achieved scores of 66.6 on MedCalc and 87.1 on MedQA-USMLE. Furthermore, targeted training improved performance on a clinical reasoning task from a baseline score of 58.1 to 62.5. These findings suggest that reinforcement learning, even when applied with a limited dataset of only a few thousand examples, can enhance medical reasoning accuracy. Crucially, this demonstration of RL's efficacy with limited data and computation paves the way for more trustworthy and practically deployable LLMs within clinical workflows and health information infrastructures. 

---
# AI-Augmented LLMs Achieve Therapist-Level Responses in Motivational Interviewing 

**Authors**: Yinghui Huang, Yuxuan Jiang, Hui Liu, Yixin Cai, Weiqing Li, Xiangen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17380)  

**Abstract**: Large language models (LLMs) like GPT-4 show potential for scaling motivational interviewing (MI) in addiction care, but require systematic evaluation of therapeutic capabilities. We present a computational framework assessing user-perceived quality (UPQ) through expected and unexpected MI behaviors. Analyzing human therapist and GPT-4 MI sessions via human-AI collaboration, we developed predictive models integrating deep learning and explainable AI to identify 17 MI-consistent (MICO) and MI-inconsistent (MIIN) behavioral metrics. A customized chain-of-thought prompt improved GPT-4's MI performance, reducing inappropriate advice while enhancing reflections and empathy. Although GPT-4 remained marginally inferior to therapists overall, it demonstrated superior advice management capabilities. The model achieved measurable quality improvements through prompt engineering, yet showed limitations in addressing complex emotional nuances. This framework establishes a pathway for optimizing LLM-based therapeutic tools through targeted behavioral metric analysis and human-AI co-evaluation. Findings highlight both the scalability potential and current constraints of LLMs in clinical communication applications. 

---
# Refusal Direction is Universal Across Safety-Aligned Languages 

**Authors**: Xinpeng Wang, Mingyang Wang, Yihong Liu, Hinrich Schütze, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2505.17306)  

**Abstract**: Refusal mechanisms in large language models (LLMs) are essential for ensuring safety. Recent research has revealed that refusal behavior can be mediated by a single direction in activation space, enabling targeted interventions to bypass refusals. While this is primarily demonstrated in an English-centric context, appropriate refusal behavior is important for any language, but poorly understood. In this paper, we investigate the refusal behavior in LLMs across 14 languages using PolyRefuse, a multilingual safety dataset created by translating malicious and benign English prompts into these languages. We uncover the surprising cross-lingual universality of the refusal direction: a vector extracted from English can bypass refusals in other languages with near-perfect effectiveness, without any additional fine-tuning. Even more remarkably, refusal directions derived from any safety-aligned language transfer seamlessly to others. We attribute this transferability to the parallelism of refusal vectors across languages in the embedding space and identify the underlying mechanism behind cross-lingual jailbreaks. These findings provide actionable insights for building more robust multilingual safety defenses and pave the way for a deeper mechanistic understanding of cross-lingual vulnerabilities in LLMs. 

---
# L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models 

**Authors**: Xiaohao Liu, Xiaobo Xia, Weixiang Zhao, Manyi Zhang, Xianzhi Yu, Xiu Su, Shuo Yang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2505.17505)  

**Abstract**: Large language models (LLMs) have achieved notable progress. Despite their success, next-token prediction (NTP), the dominant method for LLM training and inference, is constrained in both contextual coverage and inference efficiency due to its inherently sequential process. To overcome these challenges, we propose leap multi-token prediction~(L-MTP), an innovative token prediction method that extends the capabilities of multi-token prediction (MTP) by introducing a leap-based mechanism. Unlike conventional MTP, which generates multiple tokens at adjacent positions, L-MTP strategically skips over intermediate tokens, predicting non-sequential ones in a single forward pass. This structured leap not only enhances the model's ability to capture long-range dependencies but also enables a decoding strategy specially optimized for non-sequential leap token generation, effectively accelerating inference. We theoretically demonstrate the benefit of L-MTP in improving inference efficiency. Experiments across diverse benchmarks validate its merit in boosting both LLM performance and inference speed. The source code will be publicly available. 

---
# Humans Hallucinate Too: Language Models Identify and Correct Subjective Annotation Errors With Label-in-a-Haystack Prompts 

**Authors**: Georgios Chochlakis, Peter Wu, Arjun Bedi, Marcus Ma, Kristina Lerman, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17222)  

**Abstract**: Modeling complex subjective tasks in Natural Language Processing, such as recognizing emotion and morality, is considerably challenging due to significant variation in human annotations. This variation often reflects reasonable differences in semantic interpretations rather than mere noise, necessitating methods to distinguish between legitimate subjectivity and error. We address this challenge by exploring label verification in these contexts using Large Language Models (LLMs). First, we propose a simple In-Context Learning binary filtering baseline that estimates the reasonableness of a document-label pair. We then introduce the Label-in-a-Haystack setting: the query and its label(s) are included in the demonstrations shown to LLMs, which are prompted to predict the label(s) again, while receiving task-specific instructions (e.g., emotion recognition) rather than label copying. We show how the failure to copy the label(s) to the output of the LLM are task-relevant and informative. Building on this, we propose the Label-in-a-Haystack Rectification (LiaHR) framework for subjective label correction: when the model outputs diverge from the reference gold labels, we assign the generated labels to the example instead of discarding it. This approach can be integrated into annotation pipelines to enhance signal-to-noise ratios. Comprehensive analyses, human evaluations, and ecological validity studies verify the utility of LiaHR for label correction. Code is available at this https URL. 

---
# PersonaBOT: Bringing Customer Personas to Life with LLMs and RAG 

**Authors**: Muhammed Rizwan, Lars Carlsson, Mohammad Loni  

**Link**: [PDF](https://arxiv.org/pdf/2505.17156)  

**Abstract**: The introduction of Large Language Models (LLMs) has significantly transformed Natural Language Processing (NLP) applications by enabling more advanced analysis of customer personas. At Volvo Construction Equipment (VCE), customer personas have traditionally been developed through qualitative methods, which are time-consuming and lack scalability. The main objective of this paper is to generate synthetic customer personas and integrate them into a Retrieval-Augmented Generation (RAG) chatbot to support decision-making in business processes. To this end, we first focus on developing a persona-based RAG chatbot integrated with verified personas. Next, synthetic personas are generated using Few-Shot and Chain-of-Thought (CoT) prompting techniques and evaluated based on completeness, relevance, and consistency using McNemar's test. In the final step, the chatbot's knowledge base is augmented with synthetic personas and additional segment information to assess improvements in response accuracy and practical utility. Key findings indicate that Few-Shot prompting outperformed CoT in generating more complete personas, while CoT demonstrated greater efficiency in terms of response time and token usage. After augmenting the knowledge base, the average accuracy rating of the chatbot increased from 5.88 to 6.42 on a 10-point scale, and 81.82% of participants found the updated system useful in business contexts. 

---
# MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation 

**Authors**: Xiaoyuan Li, Keqin Bao, Yubo Ma, Moxin Li, Wenjie Wang, Rui Men, Yichang Zhang, Fuli Feng, Dayiheng Liu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.17123)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown promising results in complex reasoning tasks. However, current evaluations predominantly focus on single-turn reasoning scenarios, leaving interactive tasks largely unexplored. We attribute it to the absence of comprehensive datasets and scalable automatic evaluation protocols. To fill these gaps, we present MTR-Bench for LLMs' Multi-Turn Reasoning evaluation. Comprising 4 classes, 40 tasks, and 3600 instances, MTR-Bench covers diverse reasoning capabilities, fine-grained difficulty granularity, and necessitates multi-turn interactions with the environments. Moreover, MTR-Bench features fully-automated framework spanning both dataset constructions and model evaluations, which enables scalable assessment without human interventions. Extensive experiments reveal that even the cutting-edge reasoning models fall short of multi-turn, interactive reasoning tasks. And the further analysis upon these results brings valuable insights for future research in interactive AI systems. 

---
# After Retrieval, Before Generation: Enhancing the Trustworthiness of Large Language Models in RAG 

**Authors**: Xinbang Dai, Huikang Hu, Yuncheng Hua, Jiaqi Li, Yongrui Chen, Rihui Jin, Nan Hu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17118)  

**Abstract**: Retrieval-augmented generation (RAG) systems face critical challenges in balancing internal (parametric) and external (retrieved) knowledge, especially when these sources conflict or are unreliable. To analyze these scenarios comprehensively, we construct the Trustworthiness Response Dataset (TRD) with 36,266 questions spanning four RAG settings. We reveal that existing approaches address isolated scenarios-prioritizing one knowledge source, naively merging both, or refusing answers-but lack a unified framework to handle different real-world conditions simultaneously. Therefore, we propose the BRIDGE framework, which dynamically determines a comprehensive response strategy of large language models (LLMs). BRIDGE leverages an adaptive weighting mechanism named soft bias to guide knowledge collection, followed by a Maximum Soft-bias Decision Tree to evaluate knowledge and select optimal response strategies (trust internal/external knowledge, or refuse). Experiments show BRIDGE outperforms baselines by 5-15% in accuracy while maintaining balanced performance across all scenarios. Our work provides an effective solution for LLMs' trustworthy responses in real-world RAG applications. 

---
# Comparative Evaluation of Prompting and Fine-Tuning for Applying Large Language Models to Grid-Structured Geospatial Data 

**Authors**: Akash Dhruv, Yangxinyu Xie, Jordan Branham, Tanwi Mallick  

**Link**: [PDF](https://arxiv.org/pdf/2505.17116)  

**Abstract**: This paper presents a comparative study of large language models (LLMs) in interpreting grid-structured geospatial data. We evaluate the performance of a base model through structured prompting and contrast it with a fine-tuned variant trained on a dataset of user-assistant interactions. Our results highlight the strengths and limitations of zero-shot prompting and demonstrate the benefits of fine-tuning for structured geospatial and temporal reasoning. 

---
# Cultural Value Alignment in Large Language Models: A Prompt-based Analysis of Schwartz Values in Gemini, ChatGPT, and DeepSeek 

**Authors**: Robin Segerer  

**Link**: [PDF](https://arxiv.org/pdf/2505.17112)  

**Abstract**: This study examines cultural value alignment in large language models (LLMs) by analyzing how Gemini, ChatGPT, and DeepSeek prioritize values from Schwartz's value framework. Using the 40-item Portrait Values Questionnaire, we assessed whether DeepSeek, trained on Chinese-language data, exhibits distinct value preferences compared to Western models. Results of a Bayesian ordinal regression model show that self-transcendence values (e.g., benevolence, universalism) were highly prioritized across all models, reflecting a general LLM tendency to emphasize prosocial values. However, DeepSeek uniquely downplayed self-enhancement values (e.g., power, achievement) compared to ChatGPT and Gemini, aligning with collectivist cultural tendencies. These findings suggest that LLMs reflect culturally situated biases rather than a universal ethical framework. To address value asymmetries in LLMs, we propose multi-perspective reasoning, self-reflective feedback, and dynamic contextualization. This study contributes to discussions on AI fairness, cultural neutrality, and the need for pluralistic AI alignment frameworks that integrate diverse moral perspectives. 

---
# GreekBarBench: A Challenging Benchmark for Free-Text Legal Reasoning and Citations 

**Authors**: Odysseas S. Chlapanis, Dimitrios Galanis, Nikolaos Aletras, Ion Androutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.17267)  

**Abstract**: We introduce GreekBarBench, a benchmark that evaluates LLMs on legal questions across five different legal areas from the Greek Bar exams, requiring citations to statutory articles and case facts. To tackle the challenges of free-text evaluation, we propose a three-dimensional scoring system combined with an LLM-as-a-judge approach. We also develop a meta-evaluation benchmark to assess the correlation between LLM-judges and human expert evaluations, revealing that simple, span-based rubrics improve their alignment. Our systematic evaluation of 13 proprietary and open-weight LLMs shows that even though the best models outperform average expert scores, they fall short of the 95th percentile of experts. 

---
# Multi-Modality Expansion and Retention for LLMs through Parameter Merging and Decoupling 

**Authors**: Junlin Li, Guodong DU, Jing Li, Sim Kuan Goh, Wenya Wang, Yequan Wang, Fangming Liu, Ho-Kin Tang, Saleh Alharbi, Daojing He, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17110)  

**Abstract**: Fine-tuning Large Language Models (LLMs) with multimodal encoders on modality-specific data expands the modalities that LLMs can handle, leading to the formation of Multimodal LLMs (MLLMs). However, this paradigm heavily relies on resource-intensive and inflexible fine-tuning from scratch with new multimodal data. In this paper, we propose MMER (Multi-modality Expansion and Retention), a training-free approach that integrates existing MLLMs for effective multimodal expansion while retaining their original performance. Specifically, MMER reuses MLLMs' multimodal encoders while merging their LLM parameters. By comparing original and merged LLM parameters, MMER generates binary masks to approximately separate LLM parameters for each modality. These decoupled parameters can independently process modality-specific inputs, reducing parameter conflicts and preserving original MLLMs' fidelity. MMER can also mitigate catastrophic forgetting by applying a similar process to MLLMs fine-tuned on new tasks. Extensive experiments show significant improvements over baselines, proving that MMER effectively expands LLMs' multimodal capabilities while retaining 99% of the original performance, and also markedly mitigates catastrophic forgetting. 

---
# RRTL: Red Teaming Reasoning Large Language Models in Tool Learning 

**Authors**: Yifei Liu, Yu Cui, Haibin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17106)  

**Abstract**: While tool learning significantly enhances the capabilities of large language models (LLMs), it also introduces substantial security risks. Prior research has revealed various vulnerabilities in traditional LLMs during tool learning. However, the safety of newly emerging reasoning LLMs (RLLMs), such as DeepSeek-R1, in the context of tool learning remains underexplored. To bridge this gap, we propose RRTL, a red teaming approach specifically designed to evaluate RLLMs in tool learning. It integrates two novel strategies: (1) the identification of deceptive threats, which evaluates the model's behavior in concealing the usage of unsafe tools and their potential risks; and (2) the use of Chain-of-Thought (CoT) prompting to force tool invocation. Our approach also includes a benchmark for traditional LLMs. We conduct a comprehensive evaluation on seven mainstream RLLMs and uncover three key findings: (1) RLLMs generally achieve stronger safety performance than traditional LLMs, yet substantial safety disparities persist across models; (2) RLLMs can pose serious deceptive risks by frequently failing to disclose tool usage and to warn users of potential tool output risks; (3) CoT prompting reveals multi-lingual safety vulnerabilities in RLLMs. Our work provides important insights into enhancing the security of RLLMs in tool learning. 

---
# Conformal Language Model Reasoning with Coherent Factuality 

**Authors**: Maxon Rubin-Toles, Maya Gambhir, Keshav Ramji, Aaron Roth, Surbhi Goel  

**Link**: [PDF](https://arxiv.org/pdf/2505.17126)  

**Abstract**: Language models are increasingly being used in important decision pipelines, so ensuring the correctness of their outputs is crucial. Recent work has proposed evaluating the "factuality" of claims decomposed from a language model generation and applying conformal prediction techniques to filter out those claims that are not factual. This can be effective for tasks such as information retrieval, where constituent claims may be evaluated in isolation for factuality, but is not appropriate for reasoning tasks, as steps of a logical argument can be evaluated for correctness only within the context of the claims that precede them. To capture this, we define "coherent factuality" and develop a conformal-prediction-based method to guarantee coherent factuality for language model outputs. Our approach applies split conformal prediction to subgraphs within a "deducibility" graph" that represents the steps of a reasoning problem. We evaluate our method on mathematical reasoning problems from the MATH and FELM datasets and find that our algorithm consistently produces correct and substantiated orderings of claims, achieving coherent factuality across target coverage levels. Moreover, we achieve 90% factuality on our stricter definition while retaining 80% or more of the original claims, highlighting the utility of our deducibility-graph-guided approach. 

---
# When can isotropy help adapt LLMs' next word prediction to numerical domains? 

**Authors**: Rashed Shelim, Shengzhe Xu, Walid Saad, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.17135)  

**Abstract**: Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy. 

---
# Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector 

**Authors**: Haoyan Yang, Runxue Bao, Cao Xiao, Jun Ma, Parminder Bhatia, Shangqian Gao, Taha Kass-Hout  

**Link**: [PDF](https://arxiv.org/pdf/2505.17100)  

**Abstract**: LLM-as-a-Judge has emerged as a promising tool for automatically evaluating generated outputs, but its reliability is often undermined by potential biases in judgment. Existing efforts to mitigate these biases face key limitations: in-context learning-based methods fail to address rooted biases due to the evaluator's limited capacity for self-reflection, whereas fine-tuning is not applicable to all evaluator types, especially closed-source models. To address this challenge, we introduce the Reasoning-based Bias Detector (RBD), which is a plug-in module that identifies biased evaluations and generates structured reasoning to guide evaluator self-correction. Rather than modifying the evaluator itself, RBD operates externally and engages in an iterative process of bias detection and feedback-driven revision. To support its development, we design a complete pipeline consisting of biased dataset construction, supervision collection, distilled reasoning-based fine-tuning of RBD, and integration with LLM evaluators. We fine-tune four sizes of RBD models, ranging from 1.5B to 14B, and observe consistent performance improvements across all scales. Experimental results on 4 bias types--verbosity, position, bandwagon, and sentiment--evaluated using 8 LLM evaluators demonstrate RBD's strong effectiveness. For example, the RBD-8B model improves evaluation accuracy by an average of 18.5% and consistency by 10.9%, and surpasses prompting-based baselines and fine-tuned judges by 12.8% and 17.2%, respectively. These results highlight RBD's effectiveness and scalability. Additional experiments further demonstrate its strong generalization across biases and domains, as well as its efficiency. 

---
# An approach to identify the most semantically informative deep representations of text and images 

**Authors**: Santiago Acevedo, Andrea Mascaretti, Riccardo Rende, Matéo Mahaut, Marco Baroni, Alessandro Laio  

**Link**: [PDF](https://arxiv.org/pdf/2505.17101)  

**Abstract**: Deep neural networks are known to develop similar representations for semantically related data, even when they belong to different domains, such as an image and its description, or the same text in different languages. We present a method for quantitatively investigating this phenomenon by measuring the relative information content of the representations of semantically related data and probing how it is encoded into multiple tokens of large language models (LLMs) and vision transformers. Looking first at how LLMs process pairs of translated sentences, we identify inner ``semantic'' layers containing the most language-transferable information. We find moreover that, on these layers, a larger LLM (DeepSeek-V3) extracts significantly more general information than a smaller one (Llama3.1-8B). Semantic information is spread across many tokens and it is characterized by long-distance correlations between tokens and by a causal left-to-right (i.e., past-future) asymmetry. We also identify layers encoding semantic information within visual transformers. We show that caption representations in the semantic layers of LLMs predict visual representations of the corresponding images. We observe significant and model-dependent information asymmetries between image and text representations. 

---
# Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions, and Improve with Training 

**Authors**: Dillon Plunkett, Adam Morris, Keerthi Reddy, Jorge Morales  

**Link**: [PDF](https://arxiv.org/pdf/2505.17120)  

**Abstract**: We have only limited understanding of how and why large language models (LLMs) respond in the ways that they do. Their neural networks have proven challenging to interpret, and we are only beginning to tease out the function of individual neurons and circuits within them. However, another path to understanding these systems is to investigate and develop their capacity to introspect and explain their own functioning. Here, we show that i) contemporary LLMs are capable of providing accurate, quantitative descriptions of their own internal processes during certain kinds of decision-making, ii) that it is possible to improve these capabilities through training, and iii) that this training generalizes to at least some degree. To do so, we fine-tuned GPT-4o and GPT-4o-mini to make decisions in a wide variety of complex contexts (e.g., choosing between condos, loans, vacations, etc.) according to randomly-generated, quantitative preferences about how to weigh different attributes during decision-making (e.g., the relative importance of natural light versus quiet surroundings for condos). We demonstrate that the LLMs can accurately report these preferences (i.e., the weights that they learned to give to different attributes during decision-making). Next, we demonstrate that these LLMs can be fine-tuned to explain their decision-making even more accurately. Finally, we demonstrate that this training generalizes: It improves the ability of the models to accurately explain what they are doing as they make other complex decisions, not just decisions they have learned to make via fine-tuning. This work is a step towards training LLMs to accurately and broadly report on their own internal processes -- a possibility that would yield substantial benefits for interpretability, control, and safety. 

---
# Are LLMs reliable? An exploration of the reliability of large language models in clinical note generation 

**Authors**: Kristine Ann M. Carandang, Jasper Meynard P. Araña, Ethan Robert A. Casin, Christopher P. Monterola, Daniel Stanley Y. Tan, Jesus Felix B. Valenzuela, Christian M. Alis  

**Link**: [PDF](https://arxiv.org/pdf/2505.17095)  

**Abstract**: Due to the legal and ethical responsibilities of healthcare providers (HCPs) for accurate documentation and protection of patient data privacy, the natural variability in the responses of large language models (LLMs) presents challenges for incorporating clinical note generation (CNG) systems, driven by LLMs, into real-world clinical processes. The complexity is further amplified by the detailed nature of texts in CNG. To enhance the confidence of HCPs in tools powered by LLMs, this study evaluates the reliability of 12 open-weight and proprietary LLMs from Anthropic, Meta, Mistral, and OpenAI in CNG in terms of their ability to generate notes that are string equivalent (consistency rate), have the same meaning (semantic consistency) and are correct (semantic similarity), across several iterations using the same prompt. The results show that (1) LLMs from all model families are stable, such that their responses are semantically consistent despite being written in various ways, and (2) most of the LLMs generated notes close to the corresponding notes made by experts. Overall, Meta's Llama 70B was the most reliable, followed by Mistral's Small model. With these findings, we recommend the local deployment of these relatively smaller open-weight models for CNG to ensure compliance with data privacy regulations, as well as to improve the efficiency of HCPs in clinical documentation. 

---
# Systematic Evaluation of Machine-Generated Reasoning and PHQ-9 Labeling for Depression Detection Using Large Language Models 

**Authors**: Zongru Shao, Xin Wang, Zhanyang Liu, Chenhan Wang, K.P. Subbalakshmi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17119)  

**Abstract**: Recent research leverages large language models (LLMs) for early mental health detection, such as depression, often optimized with machine-generated data. However, their detection may be subject to unknown weaknesses. Meanwhile, quality control has not been applied to these generated corpora besides limited human verifications. Our goal is to systematically evaluate LLM reasoning and reveal potential weaknesses. To this end, we first provide a systematic evaluation of the reasoning over machine-generated detection and interpretation. Then we use the models' reasoning abilities to explore mitigation strategies for enhanced performance. Specifically, we do the following: A. Design an LLM instruction strategy that allows for systematic analysis of the detection by breaking down the task into several subtasks. B. Design contrastive few-shot and chain-of-thought prompts by selecting typical positive and negative examples of detection reasoning. C. Perform human annotation for the subtasks identified in the first step and evaluate the performance. D. Identify human-preferred detection with desired logical reasoning from the few-shot generation and use them to explore different optimization strategies. We conducted extensive comparisons on the DepTweet dataset across the following subtasks: 1. identifying whether the speaker is describing their own depression; 2. accurately detecting the presence of PHQ-9 symptoms, and 3. finally, detecting depression. Human verification of statistical outliers shows that LLMs demonstrate greater accuracy in analyzing and detecting explicit language of depression as opposed to implicit expressions of depression. Two optimization methods are used for performance enhancement and reduction of the statistic bias: supervised fine-tuning (SFT) and direct preference optimization (DPO). Notably, the DPO approach achieves significant performance improvement. 

---
# Trust Me, I Can Handle It: Self-Generated Adversarial Scenario Extrapolation for Robust Language Models 

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Ye Wang, Gang Tan, Shagufta Mehnaz  

**Link**: [PDF](https://arxiv.org/pdf/2505.17089)  

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities, but remain susceptible to a growing spectrum of safety risks, including jailbreaks, toxic content, hallucinations, and bias. Existing defenses often address only a single threat type or resort to rigid outright rejection, sacrificing user experience and failing to generalize across diverse and novel attacks. This paper introduces Adversarial Scenario Extrapolation (ASE), a novel inference-time computation framework that leverages Chain-of-Thought (CoT) reasoning to simultaneously enhance LLM robustness and seamlessness. ASE guides the LLM through a self-generative process of contemplating potential adversarial scenarios and formulating defensive strategies before generating a response to the user query. Comprehensive evaluation on four adversarial benchmarks with four latest LLMs shows that ASE achieves near-zero jailbreak attack success rates and minimal toxicity, while slashing outright rejections to <4%. ASE outperforms six state-of-the-art defenses in robustness-seamlessness trade-offs, with 92-99% accuracy on adversarial Q&A and 4-10x lower bias scores. By transforming adversarial perception into an intrinsic cognitive process, ASE sets a new paradigm for secure and natural human-AI interaction. 

---
# Shallow Preference Signals: Large Language Model Aligns Even Better with Truncated Data? 

**Authors**: Xuan Qi, Jiahao Qiu, Xinzhe Juan, Yue Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17122)  

**Abstract**: Aligning large language models (LLMs) with human preferences remains a key challenge in AI. Preference-based optimization methods, such as Reinforcement Learning with Human Feedback (RLHF) and Direct Preference Optimization (DPO), rely on human-annotated datasets to improve alignment. In this work, we identify a crucial property of the existing learning method: the distinguishing signal obtained in preferred responses is often concentrated in the early tokens. We refer to this as shallow preference signals.
To explore this property, we systematically truncate preference datasets at various points and train both reward models and DPO models on the truncated data. Surprisingly, models trained on truncated datasets, retaining only the first half or fewer tokens, achieve comparable or even superior performance to those trained on full datasets. For example, a reward model trained on the Skywork-Reward-Preference-80K-v0.2 dataset outperforms the full dataset when trained on a 40\% truncated dataset. This pattern is consistent across multiple datasets, suggesting the widespread presence of shallow preference signals.
We further investigate the distribution of the reward signal through decoding strategies. We consider two simple decoding strategies motivated by the shallow reward signal observation, namely Length Control Decoding and KL Threshold Control Decoding, which leverage shallow preference signals to optimize the trade-off between alignment and computational efficiency. The performance is even better, which again validates our hypothesis.
The phenomenon of shallow preference signals highlights potential issues in LLM alignment: existing alignment methods often focus on aligning only the initial tokens of responses, rather than considering the full response. This could lead to discrepancies with real-world human preferences, resulting in suboptimal alignment performance. 

---
# Reinforcing Question Answering Agents with Minimalist Policy Gradient Optimization 

**Authors**: Yihong Wu, Liheng Ma, Muzhi Li, Jiaming Zhou, Jianye Hao, Ho-fung Leung, Irwin King, Yingxue Zhang, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.17086)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable versatility, due to the lack of factual knowledge, their application to Question Answering (QA) tasks remains hindered by hallucination.
While Retrieval-Augmented Generation mitigates these issues by integrating external knowledge, existing approaches rely heavily on in-context learning, whose performance is constrained by the fundamental reasoning capabilities of LLMs.
In this paper, we propose Mujica, a Multi-hop Joint Intelligence for Complex Question Answering, comprising a planner that decomposes questions into a directed acyclic graph of subquestions and a worker that resolves questions via retrieval and reasoning. Additionally, we introduce MyGO (Minimalist policy Gradient Optimization), a novel reinforcement learning method that replaces traditional policy gradient updates with Maximum Likelihood Estimation (MLE) by sampling trajectories from an asymptotically optimal policy. MyGO eliminates the need for gradient rescaling and reference models, ensuring stable and efficient training.
Empirical results across multiple datasets demonstrate the effectiveness of Mujica-MyGO in enhancing multi-hop QA performance for various LLMs, offering a scalable and resource-efficient solution for complex QA tasks. 

---
# P2P: Automated Paper-to-Poster Generation and Fine-Grained Benchmark 

**Authors**: Tao Sun, Enhao Pan, Zhengkai Yang, Kaixin Sui, Jiajun Shi, Xianfu Cheng, Tongliang Li, Wenhao Huang, Ge Zhang, Jian Yang, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17104)  

**Abstract**: Academic posters are vital for scholarly communication, yet their manual creation is time-consuming. However, automated academic poster generation faces significant challenges in preserving intricate scientific details and achieving effective visual-textual integration. Existing approaches often struggle with semantic richness and structural nuances, and lack standardized benchmarks for evaluating generated academic posters comprehensively. To address these limitations, we introduce P2P, the first flexible, LLM-based multi-agent framework that generates high-quality, HTML-rendered academic posters directly from research papers, demonstrating strong potential for practical applications. P2P employs three specialized agents-for visual element processing, content generation, and final poster assembly-each integrated with dedicated checker modules to enable iterative refinement and ensure output quality. To foster advancements and rigorous evaluation in this domain, we construct and release P2PInstruct, the first large-scale instruction dataset comprising over 30,000 high-quality examples tailored for the academic paper-to-poster generation task. Furthermore, we establish P2PEval, a comprehensive benchmark featuring 121 paper-poster pairs and a dual evaluation methodology (Universal and Fine-Grained) that leverages LLM-as-a-Judge and detailed, human-annotated checklists. Our contributions aim to streamline research dissemination and provide the community with robust tools for developing and evaluating next-generation poster generation systems. 

---
# Decoding Rarity: Large Language Models in the Diagnosis of Rare Diseases 

**Authors**: Valentina Carbonari, Pierangelo Veltri, Pietro Hiram Guzzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17065)  

**Abstract**: Recent advances in artificial intelligence, particularly large language models LLMs, have shown promising capabilities in transforming rare disease research. This survey paper explores the integration of LLMs in the analysis of rare diseases, highlighting significant strides and pivotal studies that leverage textual data to uncover insights and patterns critical for diagnosis, treatment, and patient care. While current research predominantly employs textual data, the potential for multimodal data integration combining genetic, imaging, and electronic health records stands as a promising frontier. We review foundational papers that demonstrate the application of LLMs in identifying and extracting relevant medical information, simulating intelligent conversational agents for patient interaction, and enabling the formulation of accurate and timely diagnoses. Furthermore, this paper discusses the challenges and ethical considerations inherent in deploying LLMs, including data privacy, model transparency, and the need for robust, inclusive data sets. As part of this exploration, we present a section on experimentation that utilizes multiple LLMs alongside structured questionnaires, specifically designed for diagnostic purposes in the context of different diseases. We conclude with future perspectives on the evolution of LLMs towards truly multimodal platforms, which would integrate diverse data types to provide a more comprehensive understanding of rare diseases, ultimately fostering better outcomes in clinical settings. 

---
# What's in a prompt? Language models encode literary style in prompt embeddings 

**Authors**: Raphaël Sarfati, Haley Moller, Toni J. B. Liu, Nicolas Boullé, Christopher Earls  

**Link**: [PDF](https://arxiv.org/pdf/2505.17071)  

**Abstract**: Large language models use high-dimensional latent spaces to encode and process textual information. Much work has investigated how the conceptual content of words translates into geometrical relationships between their vector representations. Fewer studies analyze how the cumulative information of an entire prompt becomes condensed into individual embeddings under the action of transformer layers. We use literary pieces to show that information about intangible, rather than factual, aspects of the prompt are contained in deep representations. We observe that short excerpts (10 - 100 tokens) from different novels separate in the latent space independently from what next-token prediction they converge towards. Ensembles from books from the same authors are much more entangled than across authors, suggesting that embeddings encode stylistic features. This geometry of style may have applications for authorship attribution and literary analysis, but most importantly reveals the sophistication of information processing and compression accomplished by language models. 

---
# Signals from the Floods: AI-Driven Disaster Analysis through Multi-Source Data Fusion 

**Authors**: Xian Gong, Paul X. McCarthy, Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17038)  

**Abstract**: Massive and diverse web data are increasingly vital for government disaster response, as demonstrated by the 2022 floods in New South Wales (NSW), Australia. This study examines how X (formerly Twitter) and public inquiry submissions provide insights into public behaviour during crises. We analyse more than 55,000 flood-related tweets and 1,450 submissions to identify behavioural patterns during extreme weather events. While social media posts are short and fragmented, inquiry submissions are detailed, multi-page documents offering structured insights. Our methodology integrates Latent Dirichlet Allocation (LDA) for topic modelling with Large Language Models (LLMs) to enhance semantic understanding. LDA reveals distinct opinions and geographic patterns, while LLMs improve filtering by identifying flood-relevant tweets using public submissions as a reference. This Relevance Index method reduces noise and prioritizes actionable content, improving situational awareness for emergency responders. By combining these complementary data streams, our approach introduces a novel AI-driven method to refine crisis-related social media content, improve real-time disaster response, and inform long-term resilience planning. 

---
# Assessing GPT's Bias Towards Stigmatized Social Groups: An Intersectional Case Study on Nationality Prejudice and Psychophobia 

**Authors**: Afifah Kashif, Heer Patel  

**Link**: [PDF](https://arxiv.org/pdf/2505.17045)  

**Abstract**: Recent studies have separately highlighted significant biases within foundational large language models (LLMs) against certain nationalities and stigmatized social groups. This research investigates the ethical implications of these biases intersecting with outputs of widely-used GPT-3.5/4/4o LLMS. Through structured prompt series, we evaluate model responses to several scenarios involving American and North Korean nationalities with various mental disabilities. Findings reveal significant discrepancies in empathy levels with North Koreans facing greater negative bias, particularly when mental disability is also a factor. This underscores the need for improvements in LLMs designed with a nuanced understanding of intersectional identity. 

---
# Prompt Engineering: How Prompt Vocabulary affects Domain Knowledge 

**Authors**: Dimitri Schreiter  

**Link**: [PDF](https://arxiv.org/pdf/2505.17037)  

**Abstract**: Prompt engineering has emerged as a critical component in optimizing large language models (LLMs) for domain-specific tasks. However, the role of prompt specificity, especially in domains like STEM (physics, chemistry, biology, computer science and mathematics), medicine, and law, remains underexplored. This thesis addresses the problem of whether increasing the specificity of vocabulary in prompts improves LLM performance in domain-specific question-answering and reasoning tasks. We developed a synonymization framework to systematically substitute nouns, verbs, and adjectives with varying specificity levels, measuring the impact on four LLMs: Llama-3.1-70B-Instruct, Granite-13B-Instruct-V2, Flan-T5-XL, and Mistral-Large 2, across datasets in STEM, law, and medicine. Our results reveal that while generally increasing the specificity of prompts does not have a significant impact, there appears to be a specificity range, across all considered models, where the LLM performs the best. Identifying this optimal specificity range offers a key insight for prompt design, suggesting that manipulating prompts within this range could maximize LLM performance and lead to more efficient applications in specialized domains. 

---
# Bridging Supervised Learning and Reinforcement Learning in Math Reasoning 

**Authors**: Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin, Ming-Yu Liu, Jun Zhu, Haoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18116)  

**Abstract**: Reinforcement Learning (RL) has played a central role in the recent surge of LLMs' math abilities by enabling self-improvement through binary verifier signals. In contrast, Supervised Learning (SL) is rarely considered for such verification-driven training, largely due to its heavy reliance on reference answers and inability to reflect on mistakes. In this work, we challenge the prevailing notion that self-improvement is exclusive to RL and propose Negative-aware Fine-Tuning (NFT) -- a supervised approach that enables LLMs to reflect on their failures and improve autonomously with no external teachers. In online training, instead of throwing away self-generated negative answers, NFT constructs an implicit negative policy to model them. This implicit policy is parameterized with the same positive LLM we target to optimize on positive data, enabling direct policy optimization on all LLMs' generations. We conduct experiments on 7B and 32B models in math reasoning tasks. Results consistently show that through the additional leverage of negative feedback, NFT significantly improves over SL baselines like Rejection sampling Fine-Tuning, matching or even surpassing leading RL algorithms like GRPO and DAPO. Furthermore, we demonstrate that NFT and GRPO are actually equivalent in strict-on-policy training, even though they originate from entirely different theoretical foundations. Our experiments and theoretical findings bridge the gap between SL and RL methods in binary-feedback learning systems. 

---
# Towards Analyzing and Understanding the Limitations of VAPO: A Theoretical Perspective 

**Authors**: Jintian Shao, Yiming Cheng, Hongyi Huang, Beiwen Zhang, Zhiyu Wu, You Shan, Mingkai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17997)  

**Abstract**: The VAPO framework has demonstrated significant empirical success in enhancing the efficiency and reliability of reinforcement learning for long chain-of-thought (CoT) reasoning tasks with large language models (LLMs). By systematically addressing challenges such as value model bias, heterogeneous sequence lengths, and sparse reward signals, VAPO achieves state-of-the-art performance. While its practical benefits are evident, a deeper theoretical understanding of its underlying mechanisms and potential limitations is crucial for guiding future advancements. This paper aims to initiate such a discussion by exploring VAPO from a theoretical perspective, highlighting areas where its assumptions might be challenged and where further investigation could yield more robust and generalizable reasoning agents. We delve into the intricacies of value function approximation in complex reasoning spaces, the optimality of adaptive advantage estimation, the impact of token-level optimization, and the enduring challenges of exploration and generalization. 

---
# One Model Transfer to All: On Robust Jailbreak Prompts Generation against LLMs 

**Authors**: Linbao Li, Yannan Liu, Daojing He, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17598)  

**Abstract**: Safety alignment in large language models (LLMs) is increasingly compromised by jailbreak attacks, which can manipulate these models to generate harmful or unintended content. Investigating these attacks is crucial for uncovering model vulnerabilities. However, many existing jailbreak strategies fail to keep pace with the rapid development of defense mechanisms, such as defensive suffixes, rendering them ineffective against defended models. To tackle this issue, we introduce a novel attack method called ArrAttack, specifically designed to target defended LLMs. ArrAttack automatically generates robust jailbreak prompts capable of bypassing various defense measures. This capability is supported by a universal robustness judgment model that, once trained, can perform robustness evaluation for any target model with a wide variety of defenses. By leveraging this model, we can rapidly develop a robust jailbreak prompt generator that efficiently converts malicious input prompts into effective attacks. Extensive evaluations reveal that ArrAttack significantly outperforms existing attack strategies, demonstrating strong transferability across both white-box and black-box models, including GPT-4 and Claude-3. Our work bridges the gap between jailbreak attacks and defenses, providing a fresh perspective on generating robust jailbreak prompts. We make the codebase available at this https URL. 

---
# Large language model as user daily behavior data generator: balancing population diversity and individual personality 

**Authors**: Haoxin Li, Jingtao Ding, Jiahui Gong, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17615)  

**Abstract**: Predicting human daily behavior is challenging due to the complexity of routine patterns and short-term fluctuations. While data-driven models have improved behavior prediction by leveraging empirical data from various platforms and devices, the reliance on sensitive, large-scale user data raises privacy concerns and limits data availability. Synthetic data generation has emerged as a promising solution, though existing methods are often limited to specific applications. In this work, we introduce BehaviorGen, a framework that uses large language models (LLMs) to generate high-quality synthetic behavior data. By simulating user behavior based on profiles and real events, BehaviorGen supports data augmentation and replacement in behavior prediction models. We evaluate its performance in scenarios such as pertaining augmentation, fine-tuning replacement, and fine-tuning augmentation, achieving significant improvements in human mobility and smartphone usage predictions, with gains of up to 18.9%. Our results demonstrate the potential of BehaviorGen to enhance user behavior modeling through flexible and privacy-preserving synthetic data generation. 

---
# Co-Reinforcement Learning for Unified Multimodal Understanding and Generation 

**Authors**: Jingjing Jiang, Chongjie Si, Jun Luo, Hanwang Zhang, Chao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.17534)  

**Abstract**: This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce \textbf{CoRL}, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, \textbf{ULM-R1}, achieves average improvements of \textbf{7%} on three text-to-image generation datasets and \textbf{23%} on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs. 

---
# Chain-of-Lure: A Synthetic Narrative-Driven Approach to Compromise Large Language Models 

**Authors**: Wenhan Chang, Tianqing Zhu, Yu Zhao, Shuangyong Song, Ping Xiong, Wanlei Zhou, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.17519)  

**Abstract**: In the era of rapid generative AI development, interactions between humans and large language models face significant misusing risks. Previous research has primarily focused on black-box scenarios using human-guided prompts and white-box scenarios leveraging gradient-based LLM generation methods, neglecting the possibility that LLMs can act not only as victim models, but also as attacker models to harm other models. We proposes a novel jailbreaking method inspired by the Chain-of-Thought mechanism, where the attacker model uses mission transfer to conceal harmful user intent in dialogue and generates chained narrative lures to stimulate the reasoning capabilities of victim models, leading to successful jailbreaking. To enhance the attack success rate, we introduce a helper model that performs random narrative optimization on the narrative lures during multi-turn dialogues while ensuring alignment with the original intent, enabling the optimized lures to bypass the safety barriers of victim models effectively. Our experiments reveal that models with weaker safety mechanisms exhibit stronger attack capabilities, demonstrating that models can not only be exploited, but also help harm others. By incorporating toxicity scores, we employ third-party models to evaluate the harmfulness of victim models' responses to jailbreaking attempts. The study shows that using refusal keywords as an evaluation metric for attack success rates is significantly flawed because it does not assess whether the responses guide harmful questions, while toxicity scores measure the harm of generated content with more precision and its alignment with harmful questions. Our approach demonstrates outstanding performance, uncovering latent vulnerabilities in LLMs and providing data-driven feedback to optimize LLM safety mechanisms. We also discuss two defensive strategies to offer guidance on improving defense mechanisms. 

---
# Self-Training Large Language Models with Confident Reasoning 

**Authors**: Hyosoon Jang, Yunhui Jang, Sungjae Lee, Jungseul Ok, Sungsoo Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.17454)  

**Abstract**: Large language models (LLMs) have shown impressive performance by generating reasoning paths before final answers, but learning such a reasoning path requires costly human supervision. To address this issue, recent studies have explored self-training methods that improve reasoning capabilities using pseudo-labels generated by the LLMs themselves. Among these, confidence-based self-training fine-tunes LLMs to prefer reasoning paths with high-confidence answers, where confidence is estimated via majority voting. However, such methods exclusively focus on the quality of the final answer and may ignore the quality of the reasoning paths, as even an incorrect reasoning path leads to a correct answer by chance. Instead, we advocate the use of reasoning-level confidence to identify high-quality reasoning paths for self-training, supported by our empirical observations. We then propose a new self-training method, CORE-PO, that fine-tunes LLMs to prefer high-COnfidence REasoning paths through Policy Optimization. Our experiments show that CORE-PO improves the accuracy of outputs on four in-distribution and two out-of-distribution benchmarks, compared to existing self-training methods. 

---
# LLM-based Generative Error Correction for Rare Words with Synthetic Data and Phonetic Context 

**Authors**: Natsuo Yamashita, Masaaki Yamamoto, Hiroaki Kokubo, Yohei Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17410)  

**Abstract**: Generative error correction (GER) with large language models (LLMs) has emerged as an effective post-processing approach to improve automatic speech recognition (ASR) performance. However, it often struggles with rare or domain-specific words due to limited training data. Furthermore, existing LLM-based GER approaches primarily rely on textual information, neglecting phonetic cues, which leads to over-correction. To address these issues, we propose a novel LLM-based GER approach that targets rare words and incorporates phonetic information. First, we generate synthetic data to contain rare words for fine-tuning the GER model. Second, we integrate ASR's N-best hypotheses along with phonetic context to mitigate over-correction. Experimental results show that our method not only improves the correction of rare words but also reduces the WER and CER across both English and Japanese datasets. 

---
# Chart-to-Experience: Benchmarking Multimodal LLMs for Predicting Experiential Impact of Charts 

**Authors**: Seon Gyeom Kim, Jae Young Choi, Ryan Rossi, Eunyee Koh, Tak Yeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.17374)  

**Abstract**: The field of Multimodal Large Language Models (MLLMs) has made remarkable progress in visual understanding tasks, presenting a vast opportunity to predict the perceptual and emotional impact of charts. However, it also raises concerns, as many applications of LLMs are based on overgeneralized assumptions from a few examples, lacking sufficient validation of their performance and effectiveness. We introduce Chart-to-Experience, a benchmark dataset comprising 36 charts, evaluated by crowdsourced workers for their impact on seven experiential factors. Using the dataset as ground truth, we evaluated capabilities of state-of-the-art MLLMs on two tasks: direct prediction and pairwise comparison of charts. Our findings imply that MLLMs are not as sensitive as human evaluators when assessing individual charts, but are accurate and reliable in pairwise comparisons. 

---
# Zebra-Llama: Towards Extremely Efficient Hybrid Models 

**Authors**: Mingyu Yang, Mehdi Rezagholizadeh, Guihong Li, Vikram Appia, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2505.17272)  

**Abstract**: With the growing demand for deploying large language models (LLMs) across diverse applications, improving their inference efficiency is crucial for sustainable and democratized access. However, retraining LLMs to meet new user-specific requirements is prohibitively expensive and environmentally unsustainable. In this work, we propose a practical and scalable alternative: composing efficient hybrid language models from existing pre-trained models. Our approach, Zebra-Llama, introduces a family of 1B, 3B, and 8B hybrid models by combining State Space Models (SSMs) and Multi-head Latent Attention (MLA) layers, using a refined initialization and post-training pipeline to efficiently transfer knowledge from pre-trained Transformers. Zebra-Llama achieves Transformer-level accuracy with near-SSM efficiency using only 7-11B training tokens (compared to trillions of tokens required for pre-training) and an 8B teacher. Moreover, Zebra-Llama dramatically reduces KV cache size -down to 3.9%, 2%, and 2.73% of the original for the 1B, 3B, and 8B variants, respectively-while preserving 100%, 100%, and >97% of average zero-shot performance on LM Harness tasks. Compared to models like MambaInLLaMA, X-EcoMLA, Minitron, and Llamba, Zebra-Llama consistently delivers competitive or superior accuracy while using significantly fewer tokens, smaller teachers, and vastly reduced KV cache memory. Notably, Zebra-Llama-8B surpasses Minitron-8B in few-shot accuracy by 7% while using 8x fewer training tokens, over 12x smaller KV cache, and a smaller teacher (8B vs. 15B). It also achieves 2.6x-3.8x higher throughput (tokens/s) than MambaInLlama up to a 32k context length. We will release code and model checkpoints upon acceptance. 

---
# Speechless: Speech Instruction Training Without Speech for Low Resource Languages 

**Authors**: Alan Dao, Dinh Bach Vu, Huy Hoang Ha, Tuan Le Duc Anh, Shreyas Gopal, Yue Heng Yeo, Warren Keng Hoong Low, Eng Siong Chng, Jia Qi Yip  

**Link**: [PDF](https://arxiv.org/pdf/2505.17417)  

**Abstract**: The rapid growth of voice assistants powered by large language models (LLM) has highlighted a need for speech instruction data to train these systems. Despite the abundance of speech recognition data, there is a notable scarcity of speech instruction data, which is essential for fine-tuning models to understand and execute spoken commands. Generating high-quality synthetic speech requires a good text-to-speech (TTS) model, which may not be available to low resource languages. Our novel approach addresses this challenge by halting synthesis at the semantic representation level, bypassing the need for TTS. We achieve this by aligning synthetic semantic representations with the pre-trained Whisper encoder, enabling an LLM to be fine-tuned on text instructions while maintaining the ability to understand spoken instructions during inference. This simplified training process is a promising approach to building voice assistant for low-resource languages. 

---
# CHAOS: Chart Analysis with Outlier Samples 

**Authors**: Omar Moured, Yufan Chen, Ruiping Liu, Simon Reiß, Philip Torr, Jiaming Zhang, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17235)  

**Abstract**: Charts play a critical role in data analysis and visualization, yet real-world applications often present charts with challenging or noisy features. However, "outlier charts" pose a substantial challenge even for Multimodal Large Language Models (MLLMs), which can struggle to interpret perturbed charts. In this work, we introduce CHAOS (CHart Analysis with Outlier Samples), a robustness benchmark to systematically evaluate MLLMs against chart perturbations. CHAOS encompasses five types of textual and ten types of visual perturbations, each presented at three levels of severity (easy, mid, hard) inspired by the study result of human evaluation. The benchmark includes 13 state-of-the-art MLLMs divided into three groups (i.e., general-, document-, and chart-specific models) according to the training scope and data. Comprehensive analysis involves two downstream tasks (ChartQA and Chart-to-Text). Extensive experiments and case studies highlight critical insights into robustness of models across chart perturbations, aiming to guide future research in chart understanding domain. Data and code are publicly available at: this http URL. 

---
# Generalizing Large Language Model Usability Across Resource-Constrained 

**Authors**: Yun-Da Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2505.17040)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, and recent efforts have sought to extend their capabilities to multimodal domains and resource-constrained environments. However, existing approaches often rely on costly supervised fine-tuning or assume fixed training conditions, limiting their generalization when facing unseen modalities, limited data, or restricted compute resources. This dissertation presents a systematic study toward generalizing LLM usability under real-world constraints. First, it introduces a robust text-centric alignment framework that enables LLMs to seamlessly integrate diverse modalities-including text, images, tables, and any modalities - via natural language interfaces. This approach supports in-context adaptation to unseen or dynamically changing modalities without requiring retraining. To enhance robustness against noisy and missing modalities, an adversarial prompting technique is proposed, generating semantically challenging perturbations at the prompt level to stress-test model reliability. Beyond multimodal setting, the dissertation investigates inference-time optimization strategies for LLMs, leveraging prompt search and uncertainty quantification to improve performance without additional model training. This perspective offers an efficient alternative to scaling model parameters or retraining from scratch. Additionally, the work addresses low-resource domains such as Verilog code generation by designing correct-by-construction synthetic data pipelines and logic-enhanced reasoning models, achieving state-of-the-art performance with minimal data. Together, these contributions form a unified effort to enhance the adaptability, scalability, and efficiency of large language models under practical constraints. 

---
# Content Moderation in TV Search: Balancing Policy Compliance, Relevance, and User Experience 

**Authors**: Adeep Hande, Kishorekumar Sundararajan, Sardar Hamidian, Ferhan Ture  

**Link**: [PDF](https://arxiv.org/pdf/2505.17207)  

**Abstract**: Millions of people rely on search functionality to find and explore content on entertainment platforms. Modern search systems use a combination of candidate generation and ranking approaches, with advanced methods leveraging deep learning and LLM-based techniques to retrieve, generate, and categorize search results. Despite these advancements, search algorithms can still surface inappropriate or irrelevant content due to factors like model unpredictability, metadata errors, or overlooked design flaws. Such issues can misalign with product goals and user expectations, potentially harming user trust and business outcomes. In this work, we introduce an additional monitoring layer using Large Language Models (LLMs) to enhance content moderation. This additional layer flags content if the user did not intend to search for it. This approach serves as a baseline for product quality assurance, with collected feedback used to refine the initial retrieval mechanisms of the search model, ensuring a safer and more reliable user experience. 

---
