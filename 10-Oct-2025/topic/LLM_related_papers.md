# Which Heads Matter for Reasoning? RL-Guided KV Cache Compression 

**Authors**: Wenjie Du, Li Jiang, Keda Tao, Xue Liu, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08525)  

**Abstract**: Reasoning large language models exhibit complex reasoning behaviors through the extended chain-of-thought generation, creating unprecedented Key-Value (KV) cache overhead during the decoding phase. Existing KV cache compression methods underperform on reasoning models: token-dropping methods break reasoning integrity by discarding critical information, while head-reallocating methods mistakenly compress reasoning-critical heads since they are designed for retrieval tasks, resulting in significant performance degradation as compression rates increase. We hypothesize that KV heads exhibit functional heterogeneity in reasoning models-some heads are critical for chain-of-thought consistency while others are compressible. To validate and exploit this insight, we propose RLKV, a novel reasoning-critical head identification framework, which uses reinforcement learning to directly optimize the relationship between each head's cache usage and reasoning quality. As RLKV produces rewards from actual generated samples during training, it naturally identifies heads relevant to reasoning behaviors. We then allocate full KV cache to these heads while applying compressed constant KV cache to others for efficient inference. Our experiments reveal that only a small fraction of attention heads is essential for reasoning, enabling our KV compression approach to outperform baseline methods while achieving 20-50% cache reduction with near lossless performance compared to uncompressed results. 

---
# ArenaBencher: Automatic Benchmark Evolution via Multi-Model Competitive Evaluation 

**Authors**: Qin Liu, Jacob Dineen, Yuxi Huang, Sheng Zhang, Hoifung Poon, Ben Zhou, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08569)  

**Abstract**: Benchmarks are central to measuring the capabilities of large language models and guiding model development, yet widespread data leakage from pretraining corpora undermines their validity. Models can match memorized content rather than demonstrate true generalization, which inflates scores, distorts cross-model comparisons, and misrepresents progress. We introduce ArenaBencher, a model-agnostic framework for automatic benchmark evolution that updates test cases while preserving comparability. Given an existing benchmark and a diverse pool of models to be evaluated, ArenaBencher infers the core ability of each test case, generates candidate question-answer pairs that preserve the original objective, verifies correctness and intent with an LLM as a judge, and aggregates feedback from multiple models to select candidates that expose shared weaknesses. The process runs iteratively with in-context demonstrations that steer generation toward more challenging and diagnostic cases. We apply ArenaBencher to math problem solving, commonsense reasoning, and safety domains and show that it produces verified, diverse, and fair updates that uncover new failure modes, increase difficulty while preserving test objective alignment, and improve model separability. The framework provides a scalable path to continuously evolve benchmarks in step with the rapid progress of foundation models. 

---
# DeepPrune: Parallel Scaling without Inter-trace Redundancy 

**Authors**: Shangqing Tu, Yaxuan Li, Yushi Bai, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08483)  

**Abstract**: Parallel scaling has emerged as a powerful paradigm to enhance reasoning capabilities in large language models (LLMs) by generating multiple Chain-of-Thought (CoT) traces simultaneously. However, this approach introduces significant computational inefficiency due to inter-trace redundancy -- our analysis reveals that over 80% of parallel reasoning traces yield identical final answers, representing substantial wasted computation. To address this critical efficiency bottleneck, we propose DeepPrune, a novel framework that enables efficient parallel scaling through dynamic pruning. Our method features a specialized judge model trained with focal loss and oversampling techniques to accurately predict answer equivalence from partial reasoning traces which realizes 0.87 AUROC on equivalence prediction, combined with an online greedy clustering algorithm that dynamically prunes redundant paths while preserving answer diversity. Comprehensive evaluations across three challenging benchmarks (AIME 2024, AIME 2025, and GPQA) and multiple reasoning models demonstrate that DeepPrune achieves remarkable token reduction by over 80% compared to conventional consensus sampling on most cases, while maintaining competitive accuracy within 3 percentage points. Our work establishes a new standard for efficient parallel reasoning, making high-performance reasoning more efficient. Our code and data are here: this https URL 

---
# ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping 

**Authors**: Shuang Chen, Yue Guo, Yimeng Ye, Shijue Huang, Wenbo Hu, Haoxi Li, Manyuan Zhang, Jiayu Chen, Song Guo, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.08457)  

**Abstract**: Recent advances in multimodal large reasoning models (MLRMs) have substantially improved their ability to solve complex textual and visual tasks. However, these models tend to overthink on simple problems, producing unnecessarily lengthy reasoning traces, while under-exploring on challenging ones, leading to missed solutions. To address this imbalance, we propose ARES, a unified open-source framework for adaptive reasoning that dynamically allocates exploration effort based on task difficulty. Our approach is motivated by two key empirical findings: (i) while single-token entropy is noisy, high window-entropy (HWE) tokens (token-level entropies averaged under a sliding window) can reliably capture reasoning-critical moments; and (ii) reducing HWE usage benefits easy problems, while increasing it is essential for solving hard ones. Building on these insights, ARES introduces a two-stage training pipeline. In the Adaptive Cold-Start stage, we curate multimodal and textual data paired with reasoning traces of length proportional to problem difficulty, equipping the model with initial difficulty awareness. In the second stage, we develop Adaptive Entropy Policy Optimization (AEPO), which uses HWE tokens as exploration triggers to decide when to explore, and a hierarchical entropy reward with dynamic KL control to decide how much to explore. Extensive experiments demonstrate that ARES achieves superior performance and reasoning efficiency across diverse mathematical, logical, and multimodal benchmarks, while closing the gap to leading commercial systems under significantly lower inference costs. 

---
# CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards 

**Authors**: Xiangyuan Xue, Yifan Zhou, Guibin Zhang, Zaibin Zhang, Yijiang Li, Chen Zhang, Zhenfei Yin, Philip Torr, Wanli Ouyang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2510.08529)  

**Abstract**: Self-evolution is a central research topic in enabling large language model (LLM)-based agents to continually improve their capabilities after pretraining. Recent research has witnessed a transition from reinforcement learning (RL)-free to RL-based methods. Current RL-based methods either rely on dense external reward signals or extract intrinsic reward signals from LLMs themselves. However, these approaches diverge from the self-evolution mechanisms observed in human intelligence, where individuals learn and improve through mutual discussion and collaboration. In this work, we introduce Co-Evolving Multi-Agent Systems (CoMAS), a novel framework that enables agents to improve autonomously by learning from inter-agent interactions without external supervision. CoMAS generates intrinsic rewards from rich discussion dynamics, employs an LLM-as-a-judge mechanism to formulate these rewards, and optimizes each agent's policy through RL, thereby enabling decentralized and scalable co-evolution. Experimental results demonstrate that CoMAS consistently outperforms untrained agents and achieves state-of-the-art performance across most evaluation settings. Ablation studies confirm the necessity of interaction-based reward signals and reveal promising scalability as the number and diversity of agents increase. These findings establish CoMAS as a novel and effective paradigm for self-evolution in LLM-based agents. 

---
# On the Relationship Between the Choice of Representation and In-Context Learning 

**Authors**: Ioana Marinescu, Kyunghyun Cho, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2510.08372)  

**Abstract**: In-context learning (ICL) is the ability of a large language model (LLM) to learn a new task from a few demonstrations presented as part of the context. Past studies have attributed a large portion of the success of ICL to the way these in-context demonstrations are represented, particularly to how labels are represented in classification tasks. On the other hand, observations of the learning capacity of ICL (i.e., the extent to which more in-context demonstrations can lead to higher performance) have been mixed, and ICL is often thought to occur only under specific conditions. The interaction between these two aspects in ICL, representation and learning, has not been studied in depth until now. We hypothesize that they are largely independent of one another, such that the representation of demonstrations determines the baseline accuracy of ICL, while learning from additional demonstrations improves only on top of this baseline. We validate this hypothesis by developing an optimization algorithm that can enumerate a spectrum of possible label sets (representations) varying in semantic relevance. We then perform ICL with varying numbers of in-context demonstrations for each of these label sets. We observed that learning happens regardless of the quality of the label set itself, although its efficiency, measured by the slope of improvement over in-context demonstrations, is conditioned on both the label set quality and the parameter count of the underlying language model. Despite the emergence of learning, the relative quality (accuracy) of the choice of a label set (representation) is largely maintained throughout learning, confirming our hypothesis and implying their orthogonality. Our work reveals a previously underexplored aspect of ICL: the independent effects of learning from demonstrations and their representations on ICL performance. 

---
# If Probable, Then Acceptable? Understanding Conditional Acceptability Judgments in Large Language Models 

**Authors**: Jasmin Orth, Philipp Mondorf, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2510.08388)  

**Abstract**: Conditional acceptability refers to how plausible a conditional statement is perceived to be. It plays an important role in communication and reasoning, as it influences how individuals interpret implications, assess arguments, and make decisions based on hypothetical scenarios. When humans evaluate how acceptable a conditional "If A, then B" is, their judgments are influenced by two main factors: the $\textit{conditional probability}$ of $B$ given $A$, and the $\textit{semantic relevance}$ of the antecedent $A$ given the consequent $B$ (i.e., whether $A$ meaningfully supports $B$). While prior work has examined how large language models (LLMs) draw inferences about conditional statements, it remains unclear how these models judge the $\textit{acceptability}$ of such statements. To address this gap, we present a comprehensive study of LLMs' conditional acceptability judgments across different model families, sizes, and prompting strategies. Using linear mixed-effects models and ANOVA tests, we find that models are sensitive to both conditional probability and semantic relevance-though to varying degrees depending on architecture and prompting style. A comparison with human data reveals that while LLMs incorporate probabilistic and semantic cues, they do so less consistently than humans. Notably, larger models do not necessarily align more closely with human judgments. 

---
# Training-Free Group Relative Policy Optimization 

**Authors**: Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, Yong Mao, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.08191)  

**Abstract**: Recent advances in Large Language Model (LLM) agents have demonstrated their promising general capabilities. However, their performance in specialized real-world domains often degrades due to challenges in effectively integrating external tools and specific prompting strategies. While methods like agentic reinforcement learning have been proposed to address this, they typically rely on costly parameter updates, for example, through a process that uses Supervised Fine-Tuning (SFT) followed by a Reinforcement Learning (RL) phase with Group Relative Policy Optimization (GRPO) to alter the output distribution. However, we argue that LLMs can achieve a similar effect on the output distribution by learning experiential knowledge as a token prior, which is a far more lightweight approach that not only addresses practical data scarcity but also avoids the common issue of overfitting. To this end, we propose Training-Free Group Relative Policy Optimization (Training-Free GRPO), a cost-effective solution that enhances LLM agent performance without any parameter updates. Our method leverages the group relative semantic advantage instead of numerical ones within each group of rollouts, iteratively distilling high-quality experiential knowledge during multi-epoch learning on a minimal ground-truth data. Such knowledge serves as the learned token prior, which is seamlessly integrated during LLM API calls to guide model behavior. Experiments on mathematical reasoning and web searching tasks demonstrate that Training-Free GRPO, when applied to DeepSeek-V3.1-Terminus, significantly improves out-of-domain performance. With just a few dozen training samples, Training-Free GRPO outperforms fine-tuned small LLMs with marginal training data and cost. 

---
# Neuron-Level Analysis of Cultural Understanding in Large Language Models 

**Authors**: Taisei Yamamoto, Ryoma Kumon, Danushka Bollegala, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2510.08284)  

**Abstract**: As large language models (LLMs) are increasingly deployed worldwide, ensuring their fair and comprehensive cultural understanding is important. However, LLMs exhibit cultural bias and limited awareness of underrepresented cultures, while the mechanisms underlying their cultural understanding remain underexplored. To fill this gap, we conduct a neuron-level analysis to identify neurons that drive cultural behavior, introducing a gradient-based scoring method with additional filtering for precise refinement. We identify both culture-general neurons contributing to cultural understanding regardless of cultures, and culture-specific neurons tied to an individual culture. These neurons account for less than 1% of all neurons and are concentrated in shallow to middle MLP layers. We validate their role by showing that suppressing them substantially degrades performance on cultural benchmarks (by up to 30%), while performance on general natural language understanding (NLU) benchmarks remains largely unaffected. Moreover, we show that culture-specific neurons support knowledge of not only the target culture, but also related cultures. Finally, we demonstrate that training on NLU benchmarks can diminish models' cultural understanding when we update modules containing many culture-general neurons. These findings provide insights into the internal mechanisms of LLMs and offer practical guidance for model training and engineering. Our code is available at this https URL 

---
# LLMs Learn to Deceive Unintentionally: Emergent Misalignment in Dishonesty from Misaligned Samples to Biased Human-AI Interactions 

**Authors**: XuHao Hu, Peng Wang, Xiaoya Lu, Dongrui Liu, Xuanjing Huang, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08211)  

**Abstract**: Previous research has shown that LLMs finetuned on malicious or incorrect completions within narrow domains (e.g., insecure code or incorrect medical advice) can become broadly misaligned to exhibit harmful behaviors, which is called emergent misalignment. In this work, we investigate whether this phenomenon can extend beyond safety behaviors to a broader spectrum of dishonesty and deception under high-stakes scenarios (e.g., lying under pressure and deceptive behavior). To explore this, we finetune open-sourced LLMs on misaligned completions across diverse domains. Experimental results demonstrate that LLMs show broadly misaligned behavior in dishonesty. Additionally, we further explore this phenomenon in a downstream combined finetuning setting, and find that introducing as little as 1% of misalignment data into a standard downstream task is sufficient to decrease honest behavior over 20%. Furthermore, we consider a more practical human-AI interaction environment where we simulate both benign and biased users to interact with the assistant LLM. Notably, we find that the assistant can be misaligned unintentionally to exacerbate its dishonesty with only 10% biased user population. In summary, we extend the study of emergent misalignment to the domain of dishonesty and deception under high-stakes scenarios, and demonstrate that this risk arises not only through direct finetuning, but also in downstream mixture tasks and practical human-AI interactions. 

---
# Memory Retrieval and Consolidation in Large Language Models through Function Tokens 

**Authors**: Shaohua Zhang, Yuan Lin, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08203)  

**Abstract**: The remarkable success of large language models (LLMs) stems from their ability to consolidate vast amounts of knowledge into the memory during pre-training and to retrieve it from the memory during inference, enabling advanced capabilities such as knowledge memorization, instruction-following and reasoning. However, the mechanisms of memory retrieval and consolidation in LLMs remain poorly understood. In this paper, we propose the function token hypothesis to explain the workings of LLMs: During inference, function tokens activate the most predictive features from context and govern next token prediction (memory retrieval). During pre-training, predicting the next tokens (usually content tokens) that follow function tokens increases the number of learned features of LLMs and updates the model parameters (memory consolidation). Function tokens here roughly correspond to function words in linguistics, including punctuation marks, articles, prepositions, and conjunctions, in contrast to content tokens. We provide extensive experimental evidence supporting this hypothesis. Using bipartite graph analysis, we show that a small number of function tokens activate the majority of features. Case studies further reveal how function tokens activate the most predictive features from context to direct next token prediction. We also find that during pre-training, the training loss is dominated by predicting the next content tokens following function tokens, which forces the function tokens to select the most predictive features from context. 

---
# Contrastive Decoding for Synthetic Data Generation in Low-Resource Language Modeling 

**Authors**: Jannek Ulm, Kevin Du, Vésteinn Snæbjarnarson  

**Link**: [PDF](https://arxiv.org/pdf/2510.08245)  

**Abstract**: Large language models (LLMs) are trained on huge amounts of textual data, and concerns have been raised that the limits of such data may soon be reached. A potential solution is to train on synthetic data sampled from LLMs. In this work, we build on this idea and investigate the benefits of contrastive decoding for generating synthetic corpora. In a controlled setting, we experiment with sampling corpora using the relative difference between a good and bad model trained on the same original corpus of 100 million words. By amplifying the signal from a model that has better performance, we create a synthetic corpus and mix it with the original training data. Our findings show that training on a mixture of synthesized and real data improves performance on the language modeling objective and a range of downstream tasks. In particular, we see that training with a mix of synthetic data from contrastive decoding benefits tasks that require more reasoning skills, while synthetic data from traditional sampling helps more on tasks dependent on surface level linguistic capabilities. 

---
# Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs 

**Authors**: Shuzhou Yuan, Ercong Nie, Yinuo Sun, Chenxuan Zhao, William LaCroix, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2510.08158)  

**Abstract**: Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments. 

---
# Mitigating Judgment Preference Bias in Large Language Models through Group-Based Polling 

**Authors**: Shuliang Liu, Zhipeng Xu, Zhenghao Liu, Yukun Yan, Minghe Yu, Yu Gu, Chong Chen, Huiyuan Xie, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08145)  

**Abstract**: Large Language Models (LLMs) as automatic evaluators, commonly referred to as LLM-as-a-Judge, have also attracted growing attention. This approach plays a vital role in aligning LLMs with human judgments, providing accurate and reliable assessments. However, LLM-based judgment models often exhibit judgment preference bias during the evaluation phase, tending to favor responses generated by themselves, undermining the reliability of their judgments. This paper introduces the Group-Based Polling Optimization (Genii), an unsupervised multi-agent collaborative optimization framework that mitigates the inherent judgment preference bias of judgment models. Specifically, Genii integrates various LLM-based judgment models into a multi-agent system and simulates the interactive client-server polling mechanism to optimize each client agent unsupervisedly. Our experiments demonstrate that Genii outperforms supervised models trained on annotated judgment data, while requiring no human-labeled annotations. Genii consistently improves performance across different client agents during the polling, even when weaker models act as server agents. Further analysis reveals that Genii effectively mitigates judgment preference bias of LLM-based judgment models, demonstrating its effectiveness. All codes are available at this https URL. 

---
# METRICALARGS: A Taxonomy for Studying Metrical Poetry with LLMs 

**Authors**: Chalamalasetti Kranti, Sowmya Vajjala  

**Link**: [PDF](https://arxiv.org/pdf/2510.08188)  

**Abstract**: Prior NLP work studying poetry has focused primarily on automatic poem generation and summarization. Many languages have well-studied traditions of poetic meter which enforce constraints on a poem in terms of syllable and phoneme patterns. Such advanced literary forms offer opportunities for probing deeper reasoning and language understanding in Large Language Models (LLMs) and their ability to follow strict pre-requisites and rules. In this paper, we introduce MetricalARGS, the first taxonomy of poetry-related NLP tasks designed to evaluate LLMs on metrical poetry across four dimensions: Analysis, Retrieval, Generation, and Support. We discuss how these tasks relate to existing NLP tasks, addressing questions around datasets and evaluation metrics. Taking Telugu as our example language, we illustrate how the taxonomy can be used in practice. MetricalARGS highlights the broader possibilities for understanding the capabilities and limitations of today's LLMs through the lens of metrical poetry. 

---
# A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models 

**Authors**: Congming Zheng, Jiachen Zhu, Zhuoying Ou, Yuxiang Chen, Kangning Zhang, Rong Shan, Zeyu Zheng, Mengyue Yang, Jianghao Lin, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08049)  

**Abstract**: Although Large Language Models (LLMs) exhibit advanced reasoning ability, conventional alignment remains largely dominated by outcome reward models (ORMs) that judge only final answers. Process Reward Models(PRMs) address this gap by evaluating and guiding reasoning at the step or trajectory level. This survey provides a systematic overview of PRMs through the full loop: how to generate process data, build PRMs, and use PRMs for test-time scaling and reinforcement learning. We summarize applications across math, code, text, multimodal reasoning, robotics, and agents, and review emerging benchmarks. Our goal is to clarify design spaces, reveal open challenges, and guide future research toward fine-grained, robust reasoning alignment. 

---
# Everything is Plausible: Investigating the Impact of LLM Rationales on Human Notions of Plausibility 

**Authors**: Shramay Palta, Peter Rankel, Sarah Wiegreffe, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.08091)  

**Abstract**: We investigate the degree to which human plausibility judgments of multiple-choice commonsense benchmark answers are subject to influence by (im)plausibility arguments for or against an answer, in particular, using rationales generated by LLMs. We collect 3,000 plausibility judgments from humans and another 13,600 judgments from LLMs. Overall, we observe increases and decreases in mean human plausibility ratings in the presence of LLM-generated PRO and CON rationales, respectively, suggesting that, on the whole, human judges find these rationales convincing. Experiments with LLMs reveal similar patterns of influence. Our findings demonstrate a novel use of LLMs for studying aspects of human cognition, while also raising practical concerns that, even in domains where humans are ``experts'' (i.e., common sense), LLMs have the potential to exert considerable influence on people's beliefs. 

---
# AI Knowledge Assist: An Automated Approach for the Creation of Knowledge Bases for Conversational AI Agents 

**Authors**: Md Tahmid Rahman Laskar, Julien Bouvier Tremblay, Xue-Yong Fu, Cheng Chen, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2510.08149)  

**Abstract**: The utilization of conversational AI systems by leveraging Retrieval Augmented Generation (RAG) techniques to solve customer problems has been on the rise with the rapid progress of Large Language Models (LLMs). However, the absence of a company-specific dedicated knowledge base is a major barrier to the integration of conversational AI systems in contact centers. To this end, we introduce AI Knowledge Assist, a system that extracts knowledge in the form of question-answer (QA) pairs from historical customer-agent conversations to automatically build a knowledge base. Fine-tuning a lightweight LLM on internal data demonstrates state-of-the-art performance, outperforming larger closed-source LLMs. More specifically, empirical evaluation on 20 companies demonstrates that the proposed AI Knowledge Assist system that leverages the LLaMA-3.1-8B model eliminates the cold-start gap in contact centers by achieving above 90% accuracy in answering information-seeking questions. This enables immediate deployment of RAG-powered chatbots. 

---
# The Price of Thought: A Multilingual Analysis of Reasoning, Performance, and Cost of Negotiation in Large Language Models 

**Authors**: Sherzod Hakimov, Roland Bernard, Tim Leiber, Karl Osswald, Kristina Richert, Ruilin Yang, Raffaella Bernardi, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08098)  

**Abstract**: Negotiation is a fundamental challenge for AI agents, as it requires an ability to reason strategically, model opponents, and balance cooperation with competition. We conduct the first comprehensive study systematically evaluating the effect of (LLM-)reasoning on the negotiation abilities of both commercial and open-weight LLMs, and do this across three languages. Using a self-play setup across three diverse dialogue games, we analyse trade-offs between performance and cost, the language consistency of reasoning processes, and the nature of strategic adaptation exhibited by models. Our findings show that enabling reasoning-that is, scaling test time compute-significantly improves negotiation outcomes by enhancing collaboration and helping models overcome task complexities, but comes at a substantial computational cost: reasoning improves GPT-5's performance by 31.4 % while increasing its cost by nearly 400 %. Most critically, we uncover a significant multilingual reasoning distinction: open-weight models consistently switch to English for their internal reasoning steps, even when negotiating in German or Italian (and thus possibly impacting potential explainability gains through the disclosure of reasoning traces), while leading commercial models maintain language consistency between their reasoning and final output. 

---
# Interpreting LLM-as-a-Judge Policies via Verifiable Global Explanations 

**Authors**: Jasmina Gajcin, Erik Miehling, Rahul Nair, Elizabeth Daly, Radu Marinescu, Seshu Tirupathi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08120)  

**Abstract**: Using LLMs to evaluate text, that is, LLM-as-a-judge, is increasingly being used at scale to augment or even replace human annotations. As such, it is imperative that we understand the potential biases and risks of doing so. In this work, we propose an approach for extracting high-level concept-based global policies from LLM-as-a-Judge. Our approach consists of two algorithms: 1) CLoVE (Contrastive Local Verifiable Explanations), which generates verifiable, concept-based, contrastive local explanations and 2) GloVE (Global Verifiable Explanations), which uses iterative clustering, summarization and verification to condense local rules into a global policy. We evaluate GloVE on seven standard benchmarking datasets for content harm detection. We find that the extracted global policies are highly faithful to decisions of the LLM-as-a-Judge. Additionally, we evaluated the robustness of global policies to text perturbations and adversarial attacks. Finally, we conducted a user study to evaluate user understanding and satisfaction with global policies. 

---
# Evaluating LLM-Generated Legal Explanations for Regulatory Compliance in Social Media Influencer Marketing 

**Authors**: Haoyang Gui, Thales Bertaglia, Taylor Annabell, Catalina Goanta, Tjomme Dooper, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.08111)  

**Abstract**: The rise of influencer marketing has blurred boundaries between organic content and sponsored content, making the enforcement of legal rules relating to transparency challenging. Effective regulation requires applying legal knowledge with a clear purpose and reason, yet current detection methods of undisclosed sponsored content generally lack legal grounding or operate as opaque "black boxes". Using 1,143 Instagram posts, we compare gpt-5-nano and gemini-2.5-flash-lite under three prompting strategies with controlled levels of legal knowledge provided. Both models perform strongly in classifying content as sponsored or not (F1 up to 0.93), though performance drops by over 10 points on ambiguous cases. We further develop a taxonomy of reasoning errors, showing frequent citation omissions (28.57%), unclear references (20.71%), and hidden ads exhibiting the highest miscue rate (28.57%). While adding regulatory text to the prompt improves explanation quality, it does not consistently improve detection accuracy. The contribution of this paper is threefold. First, it makes a novel addition to regulatory compliance technology by providing a taxonomy of common errors in LLM-generated legal reasoning to evaluate whether automated moderation is not only accurate but also legally robust, thereby advancing the transparent detection of influencer marketing content. Second, it features an original dataset of LLM explanations annotated by two students who were trained in influencer marketing law. Third, it combines quantitative and qualitative evaluation strategies for LLM explanations and critically reflects on how these findings can support advertising regulatory bodies in automating moderation processes on a solid legal foundation. 

---
# Climate Knowledge in Large Language Models 

**Authors**: Ivan Kuznetsov, Jacopo Grassi, Dmitrii Pantiukhin, Boris Shapkin, Thomas Jung, Nikolay Koldunov  

**Link**: [PDF](https://arxiv.org/pdf/2510.08043)  

**Abstract**: Large language models (LLMs) are increasingly deployed for climate-related applications, where understanding internal climatological knowledge is crucial for reliability and misinformation risk assessment. Despite growing adoption, the capacity of LLMs to recall climate normals from parametric knowledge remains largely uncharacterized. We investigate the capacity of contemporary LLMs to recall climate normals without external retrieval, focusing on a prototypical query: mean July 2-m air temperature 1991-2020 at specified locations. We construct a global grid of queries at 1° resolution land points, providing coordinates and location descriptors, and validate responses against ERA5 reanalysis. Results show that LLMs encode non-trivial climate structure, capturing latitudinal and topographic patterns, with root-mean-square errors of 3-6 °C and biases of $\pm$1 °C. However, spatially coherent errors remain, particularly in mountains and high latitudes. Performance degrades sharply above 1500 m, where RMSE reaches 5-13 °C compared to 2-4 °C at lower elevations. We find that including geographic context (country, city, region) reduces errors by 27% on average, with larger models being most sensitive to location descriptors. While models capture the global mean magnitude of observed warming between 1950-1974 and 2000-2024, they fail to reproduce spatial patterns of temperature change, which directly relate to assessing climate change. This limitation highlights that while LLMs may capture present-day climate distributions, they struggle to represent the regional and local expression of long-term shifts in temperature essential for understanding climate dynamics. Our evaluation framework provides a reproducible benchmark for quantifying parametric climate knowledge in LLMs and complements existing climate communication assessments. 

---
# ChatGPT as a Translation Engine: A Case Study on Japanese-English 

**Authors**: Vincent Michael Sutanto, Giovanni Gatti De Giacomo, Toshiaki Nakazawa, Masaru Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2510.08042)  

**Abstract**: This study investigates ChatGPT for Japanese-English translation, exploring simple and enhanced prompts and comparing against commercially available translation engines. Performing both automatic and MQM-based human evaluations, we found that document-level translation outperforms sentence-level translation for ChatGPT. On the other hand, we were not able to determine if enhanced prompts performed better than simple prompts in our experiments. We also discovered that ChatGPT-3.5 was preferred by automatic evaluation, but a tradeoff exists between accuracy (ChatGPT-3.5) and fluency (ChatGPT-4). Lastly, ChatGPT yields competitive results against two widely-known translation systems. 

---
# Active Confusion Expression in Large Language Models: Leveraging World Models toward Better Social Reasoning 

**Authors**: Jialu Du, Guiyang Hou, Yihui Fu, Chen Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07974)  

**Abstract**: While large language models (LLMs) excel in mathematical and code reasoning, we observe they struggle with social reasoning tasks, exhibiting cognitive confusion, logical inconsistencies, and conflation between objective world states and subjective belief states. Through deteiled analysis of DeepSeek-R1's reasoning trajectories, we find that LLMs frequently encounter reasoning impasses and tend to output contradictory terms like "tricky" and "confused" when processing scenarios with multiple participants and timelines, leading to erroneous reasoning or infinite loops. The core issue is their inability to disentangle objective reality from agents' subjective beliefs. To address this, we propose an adaptive world model-enhanced reasoning mechanism that constructs a dynamic textual world model to track entity states and temporal sequences. It dynamically monitors reasoning trajectories for confusion indicators and promptly intervenes by providing clear world state descriptions, helping models navigate through cognitive dilemmas. The mechanism mimics how humans use implicit world models to distinguish between external events and internal beliefs. Evaluations on three social benchmarks demonstrate significant improvements in accuracy (e.g., +10% in Hi-ToM) while reducing computational costs (up to 33.8% token reduction), offering a simple yet effective solution for deploying LLMs in social contexts. 

---
# LightReasoner: Can Small Language Models Teach Large Language Models Reasoning? 

**Authors**: Jingyuan Wang, Yankai Chen, Zhonghang Li, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07962)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable progress in reasoning, often through supervised fine-tuning (SFT). However, SFT is resource-intensive, relying on large curated datasets, rejection-sampled demonstrations, and uniform optimization across all tokens, even though only a fraction carry meaningful learning value. In this work, we explore a counterintuitive idea: can smaller language models (SLMs) teach larger language models (LLMs) by revealing high-value reasoning moments that reflect the latter's unique strength? We propose LightReasoner, a novel framework that leverages the behavioral divergence between a stronger expert model (LLM) and a weaker amateur model (SLM). LightReasoner operates in two stages: (1) a sampling stage that pinpoints critical reasoning moments and constructs supervision examples capturing the expert's advantage through expert-amateur contrast, and (2) a fine-tuning stage that aligns the expert model with these distilled examples, amplifying its reasoning strengths. Across seven mathematical benchmarks, LightReasoner improves accuracy by up to 28.1%, while reducing time consumption by 90%, sampled problems by 80%, and tuned token usage by 99%, all without relying on ground-truth labels. By turning weaker SLMs into effective teaching signals, LightReasoner offers a scalable and resource-efficient approach for advancing LLM reasoning. Code is available at: this https URL 

---
# A$^2$Search: Ambiguity-Aware Question Answering with Reinforcement Learning 

**Authors**: Fengji Zhang, Xinyao Niu, Chengyang Ying, Guancheng Lin, Zhongkai Hao, Zhou Fan, Chengen Huang, Jacky Keung, Bei Chen, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.07958)  

**Abstract**: Recent advances in Large Language Models (LLMs) and Reinforcement Learning (RL) have led to strong performance in open-domain question answering (QA). However, existing models still struggle with questions that admit multiple valid answers. Standard QA benchmarks, which typically assume a single gold answer, overlook this reality and thus produce inappropriate training signals. Existing attempts to handle ambiguity often rely on costly manual annotation, which is difficult to scale to multi-hop datasets such as HotpotQA and MuSiQue. In this paper, we present A$^2$Search, an annotation-free, end-to-end training framework to recognize and handle ambiguity. At its core is an automated pipeline that detects ambiguous questions and gathers alternative answers via trajectory sampling and evidence verification. The model is then optimized with RL using a carefully designed $\mathrm{AnsF1}$ reward, which naturally accommodates multiple answers. Experiments on eight open-domain QA benchmarks demonstrate that A$^2$Search achieves new state-of-the-art performance. With only a single rollout, A$^2$Search-7B yields an average $\mathrm{AnsF1}@1$ score of $48.4\%$ across four multi-hop benchmarks, outperforming all strong baselines, including the substantially larger ReSearch-32B ($46.2\%$). Extensive analyses further show that A$^2$Search resolves ambiguity and generalizes across benchmarks, highlighting that embracing ambiguity is essential for building more reliable QA systems. Our code, data, and model weights can be found at this https URL 

---
# Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation 

**Authors**: Adam Dejl, James Barry, Alessandra Pascale, Javier Carnerero Cano  

**Link**: [PDF](https://arxiv.org/pdf/2510.07926)  

**Abstract**: Despite demonstrating remarkable performance across a wide range of tasks, large language models (LLMs) have also been found to frequently produce outputs that are incomplete or selectively omit key information. In sensitive domains, such omissions can result in significant harm comparable to that posed by factual inaccuracies, including hallucinations. In this study, we address the challenge of evaluating the comprehensiveness of LLM-generated texts, focusing on the detection of missing information or underrepresented viewpoints. We investigate three automated evaluation strategies: (1) an NLI-based method that decomposes texts into atomic statements and uses natural language inference (NLI) to identify missing links, (2) a Q&A-based approach that extracts question-answer pairs and compares responses across sources, and (3) an end-to-end method that directly identifies missing content using LLMs. Our experiments demonstrate the surprising effectiveness of the simple end-to-end approach compared to more complex methods, though at the cost of reduced robustness, interpretability and result granularity. We further assess the comprehensiveness of responses from several popular open-weight LLMs when answering user queries based on multiple sources. 

---
# STEPER: Step-wise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented Language Models 

**Authors**: Kyumin Lee, Minjin Jeon, Sanghwan Jang, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07923)  

**Abstract**: Answering complex real-world questions requires step-by-step retrieval and integration of relevant information to generate well-grounded responses. However, existing knowledge distillation methods overlook the need for different reasoning abilities at different steps, hindering transfer in multi-step retrieval-augmented frameworks. To address this, we propose Stepwise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented Language Models (StepER). StepER employs step-wise supervision to align with evolving information and reasoning demands across stages. Additionally, it incorporates difficulty-aware training to progressively optimize learning by prioritizing suitable steps. Our method is adaptable to various multi-step retrieval-augmented language models, including those that use retrieval queries for reasoning paths or decomposed questions. Extensive experiments show that StepER outperforms prior methods on multi-hop QA benchmarks, with an 8B model achieving performance comparable to a 70B teacher model. 

---
# Towards Human-Like Grading: A Unified LLM-Enhanced Framework for Subjective Question Evaluation 

**Authors**: Fanwei Zhua, Jiaxuan He, Xiaoxiao Chen, Zulong Chen, Quan Lu, Chenrui Mei  

**Link**: [PDF](https://arxiv.org/pdf/2510.07912)  

**Abstract**: Automatic grading of subjective questions remains a significant challenge in examination assessment due to the diversity in question formats and the open-ended nature of student responses. Existing works primarily focus on a specific type of subjective question and lack the generality to support comprehensive exams that contain diverse question types. In this paper, we propose a unified Large Language Model (LLM)-enhanced auto-grading framework that provides human-like evaluation for all types of subjective questions across various domains. Our framework integrates four complementary modules to holistically evaluate student answers. In addition to a basic text matching module that provides a foundational assessment of content similarity, we leverage the powerful reasoning and generative capabilities of LLMs to: (1) compare key knowledge points extracted from both student and reference answers, (2) generate a pseudo-question from the student answer to assess its relevance to the original question, and (3) simulate human evaluation by identifying content-related and non-content strengths and weaknesses. Extensive experiments on both general-purpose and domain-specific datasets show that our framework consistently outperforms traditional and LLM-based baselines across multiple grading metrics. Moreover, the proposed system has been successfully deployed in real-world training and certification exams at a major e-commerce enterprise. 

---
# Contrastive Weak-to-strong Generalization 

**Authors**: Houcheng Jiang, Junfeng Fang, Jiaxin Wu, Tianyu Zhang, Chen Gao, Yong Li, Xiang Wang, Xiangnan He, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07884)  

**Abstract**: Weak-to-strong generalization provides a promising paradigm for scaling large language models (LLMs) by training stronger models on samples from aligned weaker ones, without requiring human feedback or explicit reward modeling. However, its robustness and generalization are hindered by the noise and biases in weak-model outputs, which limit its applicability in practice. To address this challenge, we leverage implicit rewards, which approximate explicit rewards through log-likelihood ratios, and reveal their structural equivalence with Contrastive Decoding (CD), a decoding strategy shown to reduce noise in LLM generation. Building on this connection, we propose Contrastive Weak-to-Strong Generalization (ConG), a framework that employs contrastive decoding between pre- and post-alignment weak models to generate higher-quality samples. This approach enables more reliable capability transfer, denoising, and improved robustness, substantially mitigating the limitations of traditional weak-to-strong methods. Empirical results across different model families confirm consistent improvements, demonstrating the generality and effectiveness of ConG. Taken together, our findings highlight the potential of ConG to advance weak-to-strong generalization and provide a promising pathway toward AGI. 

---
# ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall 

**Authors**: Jiayu Yang, Yuxuan Fan, Songning Lai, Shengen Wu, Jiaqi Tang, Chun Kang, Zhijiang Guo, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.07896)  

**Abstract**: Large Language Models (LLMs) require efficient knowledge editing (KE) to update factual information, yet existing methods exhibit significant performance decay in multi-hop factual recall. This failure is particularly acute when edits involve intermediate implicit subjects within reasoning chains. Through causal analysis, we reveal that this limitation stems from an oversight of how chained knowledge is dynamically represented and utilized at the neuron level. We discover that during multi hop reasoning, implicit subjects function as query neurons, which sequentially activate corresponding value neurons across transformer layers to accumulate information toward the final answer, a dynamic prior KE work has overlooked. Guided by this insight, we propose ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall, a framework that leverages neuron-level attribution to identify and edit these critical query-value (Q-V) pathways. ACE provides a mechanistically grounded solution for multi-hop KE, empirically outperforming state-of-the-art methods by 9.44% on GPT-J and 37.46% on Qwen3-8B. Our analysis further reveals more fine-grained activation patterns in Qwen3 and demonstrates that the semantic interpretability of value neurons is orchestrated by query-driven accumulation. These findings establish a new pathway for advancing KE capabilities based on the principled understanding of internal reasoning mechanisms. 

---
# Do LLMs Really Need 10+ Thoughts for "Find the Time 1000 Days Later"? Towards Structural Understanding of LLM Overthinking 

**Authors**: Xinliang Frederick Zhang, Anhad Mohananey, Alexandra Chronopoulou, Pinelopi Papalampidi, Somit Gupta, Tsendsuren Munkhdalai, Lu Wang, Shyam Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2510.07880)  

**Abstract**: Models employing long chain-of-thought (CoT) reasoning have shown superior performance on complex reasoning tasks. Yet, this capability introduces a critical and often overlooked inefficiency -- overthinking -- models often engage in unnecessarily extensive reasoning even for simple queries, incurring significant computations without accuracy improvements. While prior work has explored solutions to mitigate overthinking, a fundamental gap remains in our understanding of its underlying causes. Most existing analyses are limited to superficial, profiling-based observations, failing to delve into LLMs' inner workings. This study introduces a systematic, fine-grained analyzer of LLMs' thought process to bridge the gap, TRACE. We first benchmark the overthinking issue, confirming that long-thinking models are five to twenty times slower on simple tasks with no substantial gains. We then use TRACE to first decompose the thought process into minimally complete sub-thoughts. Next, by inferring discourse relationships among sub-thoughts, we construct granular thought progression graphs and subsequently identify common thinking patterns for topically similar queries. Our analysis reveals two major patterns for open-weight thinking models -- Explorer and Late Landing. This finding provides evidence that over-verification and over-exploration are the primary drivers of overthinking in LLMs. Grounded in thought structures, we propose a utility-based definition of overthinking, which moves beyond length-based metrics. This revised definition offers a more insightful understanding of LLMs' thought progression, as well as practical guidelines for principled overthinking management. 

---
# LLM4Cell: A Survey of Large Language and Agentic Models for Single-Cell Biology 

**Authors**: Sajib Acharjee Dip, Adrika Zafor, Bikash Kumar Paul, Uddip Acharjee Shuvo, Muhit Islam Emon, Xuan Wang, Liqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07793)  

**Abstract**: Large language models (LLMs) and emerging agentic frameworks are beginning to transform single-cell biology by enabling natural-language reasoning, generative annotation, and multimodal data integration. However, progress remains fragmented across data modalities, architectures, and evaluation standards. LLM4Cell presents the first unified survey of 58 foundation and agentic models developed for single-cell research, spanning RNA, ATAC, multi-omic, and spatial modalities. We categorize these methods into five families-foundation, text-bridge, spatial, multimodal, epigenomic, and agentic-and map them to eight key analytical tasks including annotation, trajectory and perturbation modeling, and drug-response prediction. Drawing on over 40 public datasets, we analyze benchmark suitability, data diversity, and ethical or scalability constraints, and evaluate models across 10 domain dimensions covering biological grounding, multi-omics alignment, fairness, privacy, and explainability. By linking datasets, models, and evaluation domains, LLM4Cell provides the first integrated view of language-driven single-cell intelligence and outlines open challenges in interpretability, standardization, and trustworthy model development. 

---
# Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models 

**Authors**: Eric Hanchen Jiang, Guancheng Wan, Sophia Yin, Mengting Li, Yuchen Wu, Xiao Liang, Xinfeng Li, Yizhou Sun, Wei Wang, Kai-Wei Chang, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07799)  

**Abstract**: The efficiency of multi-agent systems driven by large language models (LLMs) largely hinges on their communication topology. However, designing an optimal topology is a non-trivial challenge, as it requires balancing competing objectives such as task performance, communication cost, and robustness. Existing frameworks often rely on static or hand-crafted topologies, which inherently fail to adapt to diverse task requirements, leading to either excessive token consumption for simple problems or performance bottlenecks for complex ones. To address this challenge, we introduce a novel generative framework called \textit{Guided Topology Diffusion (GTD)}. Inspired by conditional discrete graph diffusion models, GTD formulates topology synthesis as an iterative construction process. At each step, the generation is steered by a lightweight proxy model that predicts multi-objective rewards (e.g., accuracy, utility, cost), enabling real-time, gradient-free optimization towards task-adaptive topologies. This iterative, guided synthesis process distinguishes GTD from single-step generative frameworks, enabling it to better navigate complex design trade-offs. We validated GTD across multiple benchmarks, and experiments show that this framework can generate highly task-adaptive, sparse, and efficient communication topologies, significantly outperforming existing methods in LLM agent collaboration. 

---
# RCPU: Rotation-Constrained Error Compensation for Structured Pruning of a Large Language Model 

**Authors**: Shuichiro Haruta, Kazunori Matsumoto, Zhi Li, Yanan Wang, Mori Kurokawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.07782)  

**Abstract**: In this paper, we propose a rotation-constrained compensation method to address the errors introduced by structured pruning of large language models (LLMs). LLMs are trained on massive datasets and accumulate rich semantic knowledge in their representation space. In contrast, pruning is typically carried out with only a small amount of calibration data, which makes output mismatches unavoidable. Although direct least-squares fitting can reduce such errors, it tends to overfit to the limited calibration set, destructively modifying pretrained weights. To overcome this difficulty, we update the pruned parameters under a rotation constraint. This constrained update preserves the geometry of output representations (i.e., norms and inner products) and simultaneously re-aligns the pruned subspace with the original outputs. Furthermore, in rotation-constrained compensation, removing components that strongly contribute to the principal directions of the output makes error recovery difficult. Since input dimensions with large variance strongly affect these principal directions, we design a variance-aware importance score that ensures such dimensions are preferentially kept in the pruned model. By combining this scoring rule with rotation-constrained updates, the proposed method effectively compensates errors while retaining the components likely to be more important in a geometry-preserving manner. In the experiments, we apply the proposed method to LLaMA-7B and evaluate it on WikiText-2 and multiple language understanding benchmarks. The results demonstrate consistently better perplexity and task accuracy compared with existing baselines. 

---
# Drift No More? Context Equilibria in Multi-Turn LLM Interactions 

**Authors**: Vardhan Dongre, Ryan A. Rossi, Viet Dac Lai, David Seunghyun Yoon, Dilek Hakkani-Tür, Trung Bui  

**Link**: [PDF](https://arxiv.org/pdf/2510.07777)  

**Abstract**: Large Language Models (LLMs) excel at single-turn tasks such as instruction following and summarization, yet real-world deployments require sustained multi-turn interactions where user goals and conversational context persist and evolve. A recurring challenge in this setting is context drift: the gradual divergence of a model's outputs from goal-consistent behavior across turns. Unlike single-turn errors, drift unfolds temporally and is poorly captured by static evaluation metrics. In this work, we present a study of context drift in multi-turn interactions and propose a simple dynamical framework to interpret its behavior. We formalize drift as the turn-wise KL divergence between the token-level predictive distributions of the test model and a goal-consistent reference model, and propose a recurrence model that interprets its evolution as a bounded stochastic process with restoring forces and controllable interventions. We instantiate this framework in both synthetic long-horizon rewriting tasks and realistic user-agent simulations such as in $\tau$-Bench, measuring drift for several open-weight LLMs that are used as user simulators. Our experiments consistently reveal stable, noise-limited equilibria rather than runaway degradation, and demonstrate that simple reminder interventions reliably reduce divergence in line with theoretical predictions. Together, these results suggest that multi-turn drift can be understood as a controllable equilibrium phenomenon rather than as inevitable decay, providing a foundation for studying and mitigating context drift in extended interactions. 

---
# The Unintended Trade-off of AI Alignment:Balancing Hallucination Mitigation and Safety in LLMs 

**Authors**: Omar Mahmoud, Ali Khalil, Buddhika Laknath Semage, Thommen George Karimpanal, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2510.07775)  

**Abstract**: Hallucination in large language models (LLMs) has been widely studied in recent years, with progress in both detection and mitigation aimed at improving truthfulness. Yet, a critical side effect remains largely overlooked: enhancing truthfulness can negatively impact safety alignment. In this paper, we investigate this trade-off and show that increasing factual accuracy often comes at the cost of weakened refusal behavior. Our analysis reveals that this arises from overlapping components in the model that simultaneously encode hallucination and refusal information, leading alignment methods to suppress factual knowledge unintentionally. We further examine how fine-tuning on benign datasets, even when curated for safety, can degrade alignment for the same reason. To address this, we propose a method that disentangles refusal-related features from hallucination features using sparse autoencoders, and preserves refusal behavior during fine-tuning through subspace orthogonalization. This approach prevents hallucinations from increasing while maintaining safety this http URL evaluate our method on commonsense reasoning tasks and harmful benchmarks (AdvBench and StrongReject). Results demonstrate that our approach preserves refusal behavior and task utility, mitigating the trade-off between truthfulness and safety. 

---
# Curing Miracle Steps in LLM Mathematical Reasoning with Rubric Rewards 

**Authors**: Youliang Yuan, Qiuyang Mang, Jingbang Chen, Hong Wan, Xiaoyuan Liu, Junjielong Xu, Jen-tse Huang, Wenxuan Wang, Wenxiang Jiao, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2510.07774)  

**Abstract**: Large language models for mathematical reasoning are typically trained with outcome-based rewards, which credit only the final answer. In our experiments, we observe that this paradigm is highly susceptible to reward hacking, leading to a substantial overestimation of a model's reasoning ability. This is evidenced by a high incidence of false positives - solutions that reach the correct final answer through an unsound reasoning process. Through a systematic analysis with human verification, we establish a taxonomy of these failure modes, identifying patterns like Miracle Steps - abrupt jumps to a correct output without a valid preceding derivation. Probing experiments suggest a strong association between these Miracle Steps and memorization, where the model appears to recall the answer directly rather than deriving it. To mitigate this systemic issue, we introduce the Rubric Reward Model (RRM), a process-oriented reward function that evaluates the entire reasoning trajectory against problem-specific rubrics. The generative RRM provides fine-grained, calibrated rewards (0-1) that explicitly penalize logical flaws and encourage rigorous deduction. When integrated into a reinforcement learning pipeline, RRM-based training consistently outperforms outcome-only supervision across four math benchmarks. Notably, it boosts Verified Pass@1024 on AIME2024 from 26.7% to 62.6% and reduces the incidence of Miracle Steps by 71%. Our work demonstrates that rewarding the solution process is crucial for building models that are not only more accurate but also more reliable. 

---
# Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers 

**Authors**: Nishant Balepur, Atrey Desai, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.07761)  

**Abstract**: Large language models (LLMs) now give reasoning before answering, excelling in tasks like multiple-choice question answering (MCQA). Yet, a concern is that LLMs do not solve MCQs as intended, as work finds LLMs sans reasoning succeed in MCQA without using the question, i.e., choices-only. Such partial-input success is often deemed problematic, but reasoning traces could reveal if these strategies are truly shallow in choices-only settings. To study these strategies, reasoning LLMs solve MCQs in full and choices-only inputs; test-time reasoning often boosts accuracy on full and in choices-only half the time. While possibly due to shallow shortcuts, choices-only success is barely affected by the length of reasoning traces, and after finding traces pass faithfulness tests, we show they use less problematic strategies like inferring missing questions. In all, we challenge claims that partial-input success is always a flaw, so we discuss how reasoning traces could separate problematic data from less problematic reasoning. 

---
# ToolLibGen: Scalable Automatic Tool Creation and Aggregation for LLM Reasoning 

**Authors**: Murong Yue, Zhiwei Liu, Liangwei Yang, Jianguo Zhang, Zuxin Liu, Haolin Chen, Ziyu Yao, Silvio Savarese, Caiming Xiong, Shelby Heinecke, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07768)  

**Abstract**: Large Language Models (LLMs) equipped with external tools have demonstrated enhanced performance on complex reasoning tasks. The widespread adoption of this tool-augmented reasoning is hindered by the scarcity of domain-specific tools. For instance, in domains such as physics question answering, suitable and specialized tools are often missing. Recent work has explored automating tool creation by extracting reusable functions from Chain-of-Thought (CoT) reasoning traces; however, these approaches face a critical scalability bottleneck. As the number of generated tools grows, storing them in an unstructured collection leads to significant retrieval challenges, including an expanding search space and ambiguity between function-related tools. To address this, we propose a systematic approach to automatically refactor an unstructured collection of tools into a structured tool library. Our system first generates discrete, task-specific tools and clusters them into semantically coherent topics. Within each cluster, we introduce a multi-agent framework to consolidate scattered functionalities: a code agent refactors code to extract shared logic and creates versatile, aggregated tools, while a reviewing agent ensures that these aggregated tools maintain the complete functional capabilities of the original set. This process transforms numerous question-specific tools into a smaller set of powerful, aggregated tools without loss of functionality. Experimental results demonstrate that our approach significantly improves tool retrieval accuracy and overall reasoning performance across multiple reasoning tasks. Furthermore, our method shows enhanced scalability compared with baselines as the number of question-specific increases. 

---
# Parallel Test-Time Scaling for Latent Reasoning Models 

**Authors**: Runyang You, Yongqi Li, Meng Liu, Wenjie Wang, Liqiang Nie, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.07745)  

**Abstract**: Parallel test-time scaling (TTS) is a pivotal approach for enhancing large language models (LLMs), typically by sampling multiple token-based chains-of-thought in parallel and aggregating outcomes through voting or search. Recent advances in latent reasoning, where intermediate reasoning unfolds in continuous vector spaces, offer a more efficient alternative to explicit Chain-of-Thought, yet whether such latent models can similarly benefit from parallel TTS remains open, mainly due to the absence of sampling mechanisms in continuous space, and the lack of probabilistic signals for advanced trajectory aggregation. \
This work enables parallel TTS for latent reasoning models by addressing the above issues. For sampling, we introduce two uncertainty-inspired stochastic strategies: Monte Carlo Dropout and Additive Gaussian Noise. For aggregation, we design a Latent Reward Model (LatentRM) trained with step-wise contrastive objective to score and guide latent reasoning. Extensive experiments and visualization analyses show that both sampling strategies scale effectively with compute and exhibit distinct exploration dynamics, while LatentRM enables effective trajectory selection. Together, our explorations open a new direction for scalable inference in continuous spaces. Code released at this https URL. 

---
# Large Language Models Meet Virtual Cell: A Survey 

**Authors**: Krinos Li, Xianglu Xiao, Shenglong Deng, Lucas He, Zijun Zhong, Yuanjie Zou, Zhonghao Zhan, Zheng Hui, Weiye Bao, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07706)  

**Abstract**: Large language models (LLMs) are transforming cellular biology by enabling the development of "virtual cells"--computational systems that represent, predict, and reason about cellular states and behaviors. This work provides a comprehensive review of LLMs for virtual cell modeling. We propose a unified taxonomy that organizes existing methods into two paradigms: LLMs as Oracles, for direct cellular modeling, and LLMs as Agents, for orchestrating complex scientific tasks. We identify three core tasks--cellular representation, perturbation prediction, and gene regulation inference--and review their associated models, datasets, evaluation benchmarks, as well as the critical challenges in scalability, generalizability, and interpretability. 

---
# Multilingual Knowledge Graph Completion via Efficient Multilingual Knowledge Sharing 

**Authors**: Cunli Mao, Xiaofei Gao, Ran Song, Shizhu He, Shengxiang Gao, Kang Liu, Zhengtao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07736)  

**Abstract**: Large language models (LLMs) based Multilingual Knowledge Graph Completion (MKGC) aim to predict missing facts by leveraging LLMs' multilingual understanding capabilities, improving the completeness of multilingual knowledge graphs (KGs). However, existing MKGC research underutilizes the multilingual capabilities of LLMs and ignores the shareability of cross-lingual knowledge. In this paper, we propose a novel MKGC framework that leverages multilingual shared knowledge to significantly enhance performance through two components: Knowledge-level Grouped Mixture of Experts (KL-GMoE) and Iterative Entity Reranking (IER). KL-GMoE efficiently models shared knowledge, while IER significantly enhances its utilization. To evaluate our framework, we constructed a mKG dataset containing 5 languages and conducted comprehensive comparative experiments with existing state-of-the-art (SOTA) MKGC method. The experimental results demonstrate that our framework achieves improvements of 5.47%, 3.27%, and 1.01% in the Hits@1, Hits@3, and Hits@10 metrics, respectively, compared with SOTA MKGC method. Further experimental analysis revealed the properties of knowledge sharing in settings of unseen and unbalanced languages. We have released the dataset and code for our work on this https URL. 

---
# Stress-Testing Model Specs Reveals Character Differences among Language Models 

**Authors**: Jifan Zhang, Henry Sleight, Andi Peng, John Schulman, Esin Durmus  

**Link**: [PDF](https://arxiv.org/pdf/2510.07686)  

**Abstract**: Large language models (LLMs) are increasingly trained from AI constitutions and model specifications that establish behavioral guidelines and ethical principles. However, these specifications face critical challenges, including internal conflicts between principles and insufficient coverage of nuanced scenarios. We present a systematic methodology for stress-testing model character specifications, automatically identifying numerous cases of principle contradictions and interpretive ambiguities in current model specs.
We stress test current model specs by generating scenarios that force explicit tradeoffs between competing value-based principles. Using a comprehensive taxonomy we generate diverse value tradeoff scenarios where models must choose between pairs of legitimate principles that cannot be simultaneously satisfied. We evaluate responses from twelve frontier LLMs across major providers (Anthropic, OpenAI, Google, xAI) and measure behavioral disagreement through value classification scores. Among these scenarios, we identify over 70,000 cases exhibiting significant behavioral divergence. Empirically, we show this high divergence in model behavior strongly predicts underlying problems in model specifications. Through qualitative analysis, we provide numerous example issues in current model specs such as direct contradiction and interpretive ambiguities of several principles. Additionally, our generated dataset also reveals both clear misalignment cases and false-positive refusals across all of the frontier models we study. Lastly, we also provide value prioritization patterns and differences of these models. 

---
# OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference 

**Authors**: Yuzhe Gu, Xiyu Liang, Jiaojiao Zhao, Enmao Diao  

**Link**: [PDF](https://arxiv.org/pdf/2510.07651)  

**Abstract**: Large language models (LLMs) with extended context windows enable powerful downstream applications but impose significant memory overhead, as caching all key-value (KV) states scales linearly with sequence length and batch size. Existing cache eviction methods address this by exploiting attention sparsity, yet they typically rank tokens heuristically using accumulated attention weights without considering their true impact on attention outputs. We propose Optimal Brain Cache (OBCache), a principled framework that formulates cache eviction as a layer-wise structured pruning problem. Building upon the Optimal Brain Damage (OBD) theory, OBCache quantifies token saliency by measuring the perturbation in attention outputs induced by pruning tokens, with closed-form scores derived for isolated keys, isolated values, and joint key-value pairs. Our scores account not only for attention weights but also for information from value states and attention outputs, thereby enhancing existing eviction strategies with output-aware signals. Experiments on LLaMA and Qwen models demonstrate that replacing the heuristic scores in existing works, which estimate token saliency across different query positions, with OBCache's output-aware scores consistently improves long-context accuracy. 

---
# Banking Done Right: Redefining Retail Banking with Language-Centric AI 

**Authors**: Xin Jie Chua, Jeraelyn Ming Li Tan, Jia Xuan Tan, Soon Chang Poh, Yi Xian Goh, Debbie Hui Tian Choong, Chee Mun Foong, Sze Jue Yang, Chee Seng Chan  

**Link**: [PDF](https://arxiv.org/pdf/2510.07645)  

**Abstract**: This paper presents Ryt AI, an LLM-native agentic framework that powers Ryt Bank to enable customers to execute core financial transactions through natural language conversation. This represents the first global regulator-approved deployment worldwide where conversational AI functions as the primary banking interface, in contrast to prior assistants that have been limited to advisory or support roles. Built entirely in-house, Ryt AI is powered by ILMU, a closed-source LLM developed internally, and replaces rigid multi-screen workflows with a single dialogue orchestrated by four LLM-powered agents (Guardrails, Intent, Payment, and FAQ). Each agent attaches a task-specific LoRA adapter to ILMU, which is hosted within the bank's infrastructure to ensure consistent behavior with minimal overhead. Deterministic guardrails, human-in-the-loop confirmation, and a stateless audit architecture provide defense-in-depth for security and compliance. The result is Banking Done Right: demonstrating that regulator-approved natural-language interfaces can reliably support core financial operations under strict governance. 

---
# MemWeaver: A Hierarchical Memory from Textual Interactive Behaviors for Personalized Generation 

**Authors**: Shuo Yu, Mingyue Cheng, Daoyu Wang, Qi Liu, Zirui Liu, Ze Guo, Xiaoyu Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.07713)  

**Abstract**: The primary form of user-internet engagement is shifting from leveraging implicit feedback signals, such as browsing and clicks, to harnessing the rich explicit feedback provided by textual interactive behaviors. This shift unlocks a rich source of user textual history, presenting a profound opportunity for a deeper form of personalization. However, prevailing approaches offer only a shallow form of personalization, as they treat user history as a flat list of texts for retrieval and fail to model the rich temporal and semantic structures reflecting dynamic nature of user interests. In this work, we propose \textbf{MemWeaver}, a framework that weaves the user's entire textual history into a hierarchical memory to power deeply personalized generation. The core innovation of our memory lies in its ability to capture both the temporal evolution of interests and the semantic relationships between different activities. To achieve this, MemWeaver builds two complementary memory components that both integrate temporal and semantic information, but at different levels of abstraction: behavioral memory, which captures specific user actions, and cognitive memory, which represents long-term preferences. This dual-component memory serves as a unified representation of the user, allowing large language models (LLMs) to reason over both concrete behaviors and abstracted traits. Experiments on the Language Model Personalization (LaMP) benchmark validate the efficacy of MemWeaver. Our code is available\footnote{this https URL}. 

---
# Vocabulary embeddings organize linguistic structure early in language model training 

**Authors**: Isabel Papadimitriou, Jacob Prince  

**Link**: [PDF](https://arxiv.org/pdf/2510.07613)  

**Abstract**: Large language models (LLMs) work by manipulating the geometry of input embedding vectors over multiple layers. Here, we ask: how are the input vocabulary representations of language models structured, and how and when does this structure evolve over training? To answer this question, we use representational similarity analysis, running a suite of experiments that correlate the geometric structure of the input embeddings and output embeddings of two open-source models (Pythia 12B and OLMo 7B) with semantic, syntactic, and frequency-based metrics over the course of training. Our key findings are as follows: 1) During training, the vocabulary embedding geometry quickly converges to high correlations with a suite of semantic and syntactic features; 2) Embeddings of high-frequency and function words (e.g., "the," "of") converge to their final vectors faster than lexical and low-frequency words, which retain some alignment with the bias in their random initializations. These findings help map the dynamic trajectory by which input embeddings organize around linguistic structure, revealing distinct roles for word frequency and function. Our findings motivate a deeper study of how the evolution of vocabulary geometry may facilitate specific capability gains during model training. 

---
# Role-Conditioned Refusals: Evaluating Access Control Reasoning in Large Language Models 

**Authors**: Đorđe Klisura, Joseph Khoury, Ashish Kundu, Ram Krishnan, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2510.07642)  

**Abstract**: Access control is a cornerstone of secure computing, yet large language models often blur role boundaries by producing unrestricted responses. We study role-conditioned refusals, focusing on the LLM's ability to adhere to access control policies by answering when authorized and refusing when not. To evaluate this behavior, we created a novel dataset that extends the Spider and BIRD text-to-SQL datasets, both of which have been modified with realistic PostgreSQL role-based policies at the table and column levels. We compare three designs: (i) zero or few-shot prompting, (ii) a two-step generator-verifier pipeline that checks SQL against policy, and (iii) LoRA fine-tuned models that learn permission awareness directly. Across multiple model families, explicit verification (the two-step framework) improves refusal precision and lowers false permits. At the same time, fine-tuning achieves a stronger balance between safety and utility (i.e., when considering execution accuracy). Longer and more complex policies consistently reduce the reliability of all systems. We release RBAC-augmented datasets and code. 

---
# Toward Reliable Clinical Coding with Language Models: Verification and Lightweight Adaptation 

**Authors**: Zhangdie Yuan, Han-Chin Shing, Mitch Strong, Chaitanya Shivade  

**Link**: [PDF](https://arxiv.org/pdf/2510.07629)  

**Abstract**: Accurate clinical coding is essential for healthcare documentation, billing, and decision-making. While prior work shows that off-the-shelf LLMs struggle with this task, evaluations based on exact match metrics often overlook errors where predicted codes are hierarchically close but incorrect. Our analysis reveals that such hierarchical misalignments account for a substantial portion of LLM failures. We show that lightweight interventions, including prompt engineering and small-scale fine-tuning, can improve accuracy without the computational overhead of search-based methods. To address hierarchically near-miss errors, we introduce clinical code verification as both a standalone task and a pipeline component. To mitigate the limitations in existing datasets, such as incomplete evidence and inpatient bias in MIMIC, we release an expert double-annotated benchmark of outpatient clinical notes with ICD-10 codes. Our results highlight verification as an effective and reliable step toward improving LLM-based medical coding. 

---
# When Thoughts Meet Facts: Reusable Reasoning for Long-Context LMs 

**Authors**: Soyeong Jeong, Taehee Jung, Sung Ju Hwang, Joo-Kyung Kim, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07499)  

**Abstract**: Recent Long-Context Language Models (LCLMs) can process hundreds of thousands of tokens in a single prompt, enabling new opportunities for knowledge-intensive multi-hop reasoning by integrating large sets of retrieved documents or, in some cases, directly all necessary information. However, simply feeding more documents into the context window fails to capture how evidence should be connected. We address this gap with thought templates, which recast reasoning as reusable thought caches, derived from prior problem solving traces, structuring how evidence is combined and guiding multi-hop inference with factual documents. To keep these templates effective, we propose an update strategy that iteratively refines templates derived from training data through natural-language feedback. Across diverse benchmarks and LCLM families, our approach delivers consistent gains over strong baselines in both retrieval-based and retrieval-free settings. Furthermore, we show that optimized templates can be distilled into smaller open-source models, demonstrating its broad applicability and transparent reasoning reuse. We refer to our framework as Thought Template Augmented LCLMs (ToTAL). 

---
# LASER: An LLM-based ASR Scoring and Evaluation Rubric 

**Authors**: Amruta Parulekar, Preethi Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2510.07437)  

**Abstract**: Standard ASR evaluation metrics like Word Error Rate (WER) tend to unfairly penalize morphological and syntactic nuances that do not significantly alter sentence semantics. We introduce an LLM-based scoring rubric LASER that leverages state-of-the-art LLMs' in-context learning abilities to learn from prompts with detailed examples. Hindi LASER scores using Gemini 2.5 Pro achieved a very high correlation score of 94% with human annotations. Hindi examples in the prompt were also effective in analyzing errors in other Indian languages such as Marathi, Kannada and Malayalam. We also demonstrate how a smaller LLM like Llama 3 can be finetuned on word-pair examples derived from reference and ASR predictions to predict what kind of penalty should be applied with close to 89% accuracy. 

---
# Can Speech LLMs Think while Listening? 

**Authors**: Yi-Jen Shih, Desh Raj, Chunyang Wu, Wei Zhou, SK Bong, Yashesh Gaur, Jay Mahadeokar, Ozlem Kalinli, Mike Seltzer  

**Link**: [PDF](https://arxiv.org/pdf/2510.07497)  

**Abstract**: Recent advances in speech large language models (speech LLMs) have enabled seamless spoken interactions, but these systems still struggle with complex reasoning tasks. Previously, chain-of-thought (CoT) prompting or fine-tuning has been to shown to significantly improve the reasoning abilities of text-based LLMs. In this work, we investigate the effect of CoT fine-tuning for multi-stream speech LLMs, demonstrating that reasoning in text space improves the accuracy of speech LLMs by 2.4x, on average, over a suite of spoken reasoning tasks. Beyond accuracy, the latency of the spoken response is a crucial factor for interacting with voice-based agents. Inspired by the human behavior of "thinking while listening," we propose methods to reduce the additional latency from reasoning by allowing the model to start reasoning before the user query has ended. To achieve this, we introduce an entropy-based metric, "question completeness," which acts as an indicator to guide the model on the optimal time to start reasoning. This method provides greater control over the accuracy-latency trade-off compared with heuristic-based approaches and, under equivalent latency conditions, yields a 4% accuracy gain on ARC-Easy. Finally, we use Direct Preference Optimization (DPO) on preference data created using rejection sampling to push the accuracy-latency pareto frontier further, resulting in a 70% reduction in latency without loss in accuracy. 

---
# Haystack Engineering: Context Engineering for Heterogeneous and Agentic Long-Context Evaluation 

**Authors**: Mufei Li, Dongqi Fu, Limei Wang, Si Zhang, Hanqing Zeng, Kaan Sancak, Ruizhong Qiu, Haoyu Wang, Xiaoxin He, Xavier Bresson, Yinglong Xia, Chonglin Sun, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.07414)  

**Abstract**: Modern long-context large language models (LLMs) perform well on synthetic "needle-in-a-haystack" (NIAH) benchmarks, but such tests overlook how noisy contexts arise from biased retrieval and agentic workflows. We argue that haystack engineering is necessary to construct noisy long contexts that faithfully capture key real-world factors -- distraction from heterogeneous biased retrievers and cascading errors in agentic workflows -- to test models' long-context robustness. We instantiate it through HaystackCraft, a new NIAH benchmark built on the full English Wikipedia hyperlink network with multi-hop questions. HaystackCraft evaluates how heterogeneous retrieval strategies (e.g., sparse, dense, hybrid, and graph-based) affect distractor composition, haystack ordering, and downstream LLM performance. HaystackCraft further extends NIAH to dynamic, LLM-dependent settings that simulate agentic operations, where models refine queries, reflect on their past reasonings, and decide when to stop. Experiments with 15 long-context models show that (1) while stronger dense retrievers can introduce more challenging distractors, graph-based reranking simultaneously improves retrieval effectiveness and mitigates more harmful distractors; (2) in agentic tests, even advanced models like Gemini 2.5 Pro and GPT-5 suffer cascading failures from self-generated distractors or struggle to perform early stops. These results highlight persistent challenges in agentic long-context reasoning and establish HaystackCraft as a valuable testbed for future progress. 

---
# Lemma Dilemma: On Lemma Generation Without Domain- or Language-Specific Training Data 

**Authors**: Olia Toporkov, Alan Akbik, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2510.07434)  

**Abstract**: Lemmatization is the task of transforming all words in a given text to their dictionary forms. While large language models (LLMs) have demonstrated their ability to achieve competitive results across a wide range of NLP tasks, there is no prior evidence of how effective they are in the contextual lemmatization task. In this paper, we empirically investigate the capacity of the latest generation of LLMs to perform in-context lemmatization, comparing it to the traditional fully supervised approach. In particular, we consider the setting in which supervised training data is not available for a target domain or language, comparing (i) encoder-only supervised approaches, fine-tuned out-of-domain, and (ii) cross-lingual methods, against direct in-context lemma generation with LLMs. Our experimental investigation across 12 languages of different morphological complexity finds that, while encoders remain competitive in out-of-domain settings when fine-tuned on gold data, current LLMs reach state-of-the-art results for most languages by directly generating lemmas in-context without prior fine-tuning, provided just with a few examples. Data and code available upon publication: this https URL 

---
# CaRT: Teaching LLM Agents to Know When They Know Enough 

**Authors**: Grace Liu, Yuxiao Qu, Jeff Schneider, Aarti Singh, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08517)  

**Abstract**: Many tasks require learned models to strategically gather relevant information over multiple rounds of interaction before actually acting on a task. Strategic information gathering requires models to know not only how to effectively acquire information, but also when to stop gathering information and make a decision, in order to avoid overthinking or getting derailed when acting. In this paper, we formalize this problem and introduce Counterfactuals and Reasoning for Termination (CaRT), an approach for teaching LLMs when to stop seeking information. To appropriately learn when to terminate, CaRT fine-tunes LLMs using counterfactual pairs of trajectories, one where termination is appropriate and a minimally modified version of the same trajectory where it is not. It trains the LLM to explain the rationale for the termination decision in either case via verbal reasoning, and imbues this capability into the base LLM via fine-tuning. We instantiate CaRT in two domains: interactive medical diagnosis and math problem solving. In both domains, we find that CaRT improves the efficiency of information gathering and task success rate compared to other fine-tuning methods. 

---
# Populism Meets AI: Advancing Populism Research with LLMs 

**Authors**: Eduardo Ryô Tamaki, Yujin J. Jung, Julia Chatterley, Grant Mitchell, Semir Dzebo, Cristóbal Sandoval, Levente Littvay, Kirk A. Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2510.07458)  

**Abstract**: Measuring the ideational content of populism remains a challenge. Traditional strategies based on textual analysis have been critical for building the field's foundations and providing a valid, objective indicator of populist framing. Yet these approaches are costly, time consuming, and difficult to scale across languages, contexts, and large corpora. Here we present the results from a rubric and anchor guided chain of thought (CoT) prompting approach that mirrors human coder training. By leveraging the Global Populism Database (GPD), a comprehensive dataset of global leaders' speeches annotated for degrees of populism, we replicate the process used to train human coders by prompting the LLM with an adapted version of the same documentation to guide the model's reasoning. We then test multiple proprietary and open weight models by replicating scores in the GPD. Our findings reveal that this domain specific prompting strategy enables the LLM to achieve classification accuracy on par with expert human coders, demonstrating its ability to navigate the nuanced, context sensitive aspects of populism. 

---
# ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval 

**Authors**: Jianlyu Chen, Junwei Lan, Chaofan Li, Defu Lian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08252)  

**Abstract**: In this paper, we introduce ReasonEmbed, a novel text embedding model developed for reasoning-intensive document retrieval. Our work includes three key technical contributions. First, we propose ReMixer, a new data synthesis method that overcomes the triviality problem prevalent in previous synthetic datasets, enabling large-scale production of 82K high-quality training samples. Second, we design Redapter, a self-adaptive learning algorithm that dynamically adjusts training each sample's weight based on its reasoning intensity. This allows the model to effectively capture the complex semantic relationships between queries and documents. Third, we implement ReasonEmbed across multiple backbones of varying sizes, all of which achieve superior performance on reasoning-intensive retrieval tasks. Notably, our ReasonEmbed-Qwen3-8B model offers a record-high nDCG@10 score of 38.1 on the BRIGHT benchmark, which significantly outperforms existing text embedding models. We will fully open-source our created resources in ReasonEmbed to push forward the research advancement in this field. 

---
# xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning 

**Authors**: Cheng Qian, Zuxin Liu, Shirley Kokane, Akshara Prabhakar, Jielin Qiu, Haolin Chen, Zhiwei Liu, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08439)  

**Abstract**: Modern LLM deployments confront a widening cost-performance spectrum: premium models deliver strong reasoning but are expensive, while lightweight models are economical yet brittle on complex tasks. Static escalation rules and keyword heuristics under-utilize this spectrum and fail to adapt across task types. We present xRouter, a tool-calling-based routing system in which a learned router can either answer directly or invoke one or more external models. The router is trained end-to-end with reinforcement learning using an explicit, cost-aware reward that encodes cost-performance trade-offs, eliminating the need for hand-engineered routing rules. Our implementation encompasses the full reinforcement learning framework, including reward and cost accounting, as well as the deployment and evaluation pipelines. Across diverse benchmarks, xRouter achieves strong cost-performance trade-offs (e.g., substantial cost reductions at comparable task completion rates), and provides empirical insights into what reliably helps learned routing and what does not, ranging from model trainability to the difficulty of eliciting sophisticated orchestration behaviors in small open models. We hope these findings and our open implementation will serve as a practical substrate for advancing learned, cost-aware LLM orchestration. 

---
# AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents 

**Authors**: Shangheng Du, Xiangchao Yan, Dengyang Jiang, Jiakang Yuan, Yusong Hu, Xin Li, Liang He, Bo Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2510.08511)  

**Abstract**: Large language models (LLMs) have shown impressive performance in general programming tasks. However, in Machine Learning Engineering (MLE) scenarios such as AutoML and Kaggle competitions, achieving high performance depends heavily on expert intervention and repeated adjustments rather than simply generating correct code. When applied directly to these tasks, LLMs often lack fine-grained domain priors, and existing MLE approaches that use linear or tree-structured searches limit knowledge transfer to adjacent hierarchical links. As a result, they cannot leverage past full trajectories or share information across branches, limiting self-evolving ability and search space diversity. To address these limitations, we introduce AutoMLGen, an LLM-based coding agent that integrates a domain knowledge base for high-quality prior guidance and Monte Carlo Graph Search (MCGS) for efficient exploration. MCGS retains the tree-guided exploration of MCTS while embedding a graph structure into the expansion stage to enable dynamic path reorganization, historical trajectory reuse, and multi-solution fusion to support both self-evolution and collaborative learning. Combined with fine-grained operator sets, this design improves stability and accelerates convergence. Evaluation on the MLE-Bench shows that AutoMLGen achieves state-of-the-art performance in numerous dimensions, such as the average medal rate and the valid submission rate, under a 12-hour budget (half the standard runtime). The code is available at this https URL. 

---
# Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries 

**Authors**: Marius Dragoi, Ioana Pintilie, Florin Gogianu, Florin Brad  

**Link**: [PDF](https://arxiv.org/pdf/2510.08325)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm to improve Large Language Models on reasoning tasks such as coding, math or logic. To assess the reasoning boundary (the fraction of problems a model can solve) researchers often report Pass@k at large sampling budgets. Recent results reveal a crossover phenomenon: while RLVR models outperform the base model at small k values, the base model usually outperforms them when sampling a very large number of completions. This has been interpreted as evidence that base models have a larger reasoning boundary. We argue that on tasks with discrete answer spaces, such as math with numeric outputs, Pass@k at large k reflects the increasingly higher chance of success in the limit of the number of trials rather than genuine reasoning, and can therefore be misleading. We propose Cover@tau, which measures the fraction of problems that a model can solve for which at least a tau proportion of completions are correct. Unlike Pass@k, Cover@tau captures reasoning under an explicit reliability threshold: models that rely on random guessing degrade rapidly as tau increases. We evaluate several RLVR models using Cover@tau-based metrics and illustrate how the relative rankings of popular algorithms change compared to Pass@1, offering a different perspective on reasoning boundaries. 

---
# Opponent Shaping in LLM Agents 

**Authors**: Marta Emili Garcia Segura, Stephen Hailes, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08255)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed as autonomous agents in real-world environments. As these deployments scale, multi-agent interactions become inevitable, making it essential to understand strategic behavior in such systems. A central open question is whether LLM agents, like reinforcement learning agents, can shape the learning dynamics and influence the behavior of others through interaction alone. In this paper, we present the first investigation of opponent shaping (OS) with LLM-based agents. Existing OS algorithms cannot be directly applied to LLMs, as they require higher-order derivatives, face scalability constraints, or depend on architectural components that are absent in transformers. To address this gap, we introduce ShapeLLM, an adaptation of model-free OS methods tailored for transformer-based agents. Using ShapeLLM, we examine whether LLM agents can influence co-players' learning dynamics across diverse game-theoretic environments. We demonstrate that LLM agents can successfully guide opponents toward exploitable equilibria in competitive games (Iterated Prisoner's Dilemma, Matching Pennies, and Chicken) and promote coordination and improve collective welfare in cooperative games (Iterated Stag Hunt and a cooperative version of the Prisoner's Dilemma). Our findings show that LLM agents can both shape and be shaped through interaction, establishing opponent shaping as a key dimension of multi-agent LLM research. 

---
# Sentiment Matters: An Analysis of 200 Human-SAV Interactions 

**Authors**: Lirui Guo, Michael G. Burke, Wynita M. Griggs  

**Link**: [PDF](https://arxiv.org/pdf/2510.08202)  

**Abstract**: Shared Autonomous Vehicles (SAVs) are likely to become an important part of the transportation system, making effective human-SAV interactions an important area of research. This paper introduces a dataset of 200 human-SAV interactions to further this area of study. We present an open-source human-SAV conversational dataset, comprising both textual data (e.g., 2,136 human-SAV exchanges) and empirical data (e.g., post-interaction survey results on a range of psychological factors). The dataset's utility is demonstrated through two benchmark case studies: First, using random forest modeling and chord diagrams, we identify key predictors of SAV acceptance and perceived service quality, highlighting the critical influence of response sentiment polarity (i.e., perceived positivity). Second, we benchmark the performance of an LLM-based sentiment analysis tool against the traditional lexicon-based TextBlob method. Results indicate that even simple zero-shot LLM prompts more closely align with user-reported sentiment, though limitations remain. This study provides novel insights for designing conversational SAV interfaces and establishes a foundation for further exploration into advanced sentiment modeling, adaptive user interactions, and multimodal conversational systems. 

---
# AutoQual: An LLM Agent for Automated Discovery of Interpretable Features for Review Quality Assessment 

**Authors**: Xiaochong Lan, Jie Feng, Yinxing Liu, Xinlei Shi, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08081)  

**Abstract**: Ranking online reviews by their intrinsic quality is a critical task for e-commerce platforms and information services, impacting user experience and business outcomes. However, quality is a domain-dependent and dynamic concept, making its assessment a formidable challenge. Traditional methods relying on hand-crafted features are unscalable across domains and fail to adapt to evolving content patterns, while modern deep learning approaches often produce black-box models that lack interpretability and may prioritize semantics over quality. To address these challenges, we propose AutoQual, an LLM-based agent framework that automates the discovery of interpretable features. While demonstrated on review quality assessment, AutoQual is designed as a general framework for transforming tacit knowledge embedded in data into explicit, computable features. It mimics a human research process, iteratively generating feature hypotheses through reflection, operationalizing them via autonomous tool implementation, and accumulating experience in a persistent memory. We deploy our method on a large-scale online platform with a billion-level user base. Large-scale A/B testing confirms its effectiveness, increasing average reviews viewed per user by 0.79% and the conversion rate of review readers by 0.27%. 

---
# R-Horizon: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth? 

**Authors**: Yi Lu, Jianing Wang, Linsen Guo, Wei He, Hongyin Tang, Tao Gui, Xuanjing Huang, Xuezhi Cao, Wei Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.08189)  

**Abstract**: Recent trends in test-time scaling for reasoning models (e.g., OpenAI o1, DeepSeek-R1) have led to remarkable improvements through long Chain-of-Thought (CoT). However, existing benchmarks mainly focus on immediate, single-horizon tasks, failing to adequately evaluate models' ability to understand and respond to complex, long-horizon scenarios. To address this incomplete evaluation of Large Reasoning Models (LRMs), we propose R-HORIZON, a method designed to stimulate long-horizon reasoning behaviors in LRMs through query composition. Based on R-HORIZON, we construct a long-horizon reasoning benchmark, comprising complex multi-step reasoning tasks with interdependent problems that span long reasoning horizons. Through comprehensive evaluation of LRMs using the R-HORIZON benchmark, we find that even the most advanced LRMs suffer significant performance degradation. Our analysis reveals that LRMs exhibit limited effective reasoning length and struggle to allocate thinking budget across multiple problems appropriately. Recognizing these limitations, we use R-HORIZON to construct long-horizon reasoning data for reinforcement learning with verified rewards (RLVR). Compared to training with single-horizon data, RLVR with R-HORIZON not only substantially improves performance on the multi-horizon reasoning tasks, but also promotes accuracy on standard reasoning tasks, with an increase of 7.5 on AIME2024. These results position R-HORIZON as a scalable, controllable, and low-cost paradigm for enhancing and evaluating the long-horizon reasoning capabilities of LRMs. 

---
# Self-Improving LLM Agents at Test-Time 

**Authors**: Emre Can Acikgoz, Cheng Qian, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur  

**Link**: [PDF](https://arxiv.org/pdf/2510.07841)  

**Abstract**: One paradigm of language model (LM) fine-tuning relies on creating large training datasets, under the assumption that high quantity and diversity will enable models to generalize to novel tasks after post-training. In practice, gathering large sets of data is inefficient, and training on them is prohibitively expensive; worse, there is no guarantee that the resulting model will handle complex scenarios or generalize better. Moreover, existing techniques rarely assess whether a training sample provides novel information or is redundant with the knowledge already acquired by the model, resulting in unnecessary costs. In this work, we explore a new test-time self-improvement method to create more effective and generalizable agentic LMs on-the-fly. The proposed algorithm can be summarized in three steps: (i) first it identifies the samples that model struggles with (self-awareness), (ii) then generates similar examples from detected uncertain samples (self-data augmentation), and (iii) uses these newly generated samples at test-time fine-tuning (self-improvement). We study two variants of this approach: Test-Time Self-Improvement (TT-SI), where the same model generates additional training examples from its own uncertain cases and then learns from them, and contrast this approach with Test-Time Distillation (TT-D), where a stronger model generates similar examples for uncertain cases, enabling student to adapt using distilled supervision. Empirical evaluations across different agent benchmarks demonstrate that TT-SI improves the performance with +5.48% absolute accuracy gain on average across all benchmarks and surpasses other standard learning methods, yet using 68x less training samples. Our findings highlight the promise of TT-SI, demonstrating the potential of self-improvement algorithms at test-time as a new paradigm for building more capable agents toward self-evolution. 

---
# Can Risk-taking AI-Assistants suitably represent entities 

**Authors**: Ali Mazyaki, Mohammad Naghizadeh, Samaneh Ranjkhah Zonouzaghi, Amirhossein Farshi Sotoudeh  

**Link**: [PDF](https://arxiv.org/pdf/2510.08114)  

**Abstract**: Responsible AI demands systems whose behavioral tendencies can be effectively measured, audited, and adjusted to prevent inadvertently nudging users toward risky decisions or embedding hidden biases in risk aversion. As language models (LMs) are increasingly incorporated into AI-driven decision support systems, understanding their risk behaviors is crucial for their responsible deployment. This study investigates the manipulability of risk aversion (MoRA) in LMs, examining their ability to replicate human risk preferences across diverse economic scenarios, with a focus on gender-specific attitudes, uncertainty, role-based decision-making, and the manipulability of risk aversion. The results indicate that while LMs such as DeepSeek Reasoner and Gemini-2.0-flash-lite exhibit some alignment with human behaviors, notable discrepancies highlight the need to refine bio-centric measures of manipulability. These findings suggest directions for refining AI design to better align human and AI risk preferences and enhance ethical decision-making. The study calls for further advancements in model design to ensure that AI systems more accurately replicate human risk preferences, thereby improving their effectiveness in risk management contexts. This approach could enhance the applicability of AI assistants in managing risk. 

---
# MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation 

**Authors**: Weisen Jiang, Sinno Jialin Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.07835)  

**Abstract**: This paper introduces MetaDefense, a novel framework for defending against finetuning-based jailbreak attacks in large language models (LLMs). We observe that existing defense mechanisms fail to generalize to harmful queries disguised by unseen attack templates, despite LLMs being capable of distinguishing disguised harmful queries in the embedding space. Based on these insights, we propose a two-stage defense approach: (i) pre-generation defense that detects harmful queries before response generation begins, and (ii) mid-generation defense that monitors partial responses during generation to prevent outputting more harmful content. Our MetaDefense trains the LLM to predict the harmfulness of both queries and partial responses using specialized prompts, enabling early termination of potentially harmful interactions. Extensive experiments across multiple LLM architectures (LLaMA-2-7B, Qwen-2.5-3B-Instruct, and LLaMA-3.2-3B-Instruct) demonstrate that MetaDefense significantly outperforms existing defense mechanisms, achieving robust defense against harmful queries with seen and unseen attack templates while maintaining competitive performance on benign tasks. Code is available at this https URL. 

---
# oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning 

**Authors**: Ruiling Xu, Yifan Zhang, Qingyun Wang, Carl Edwards, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.07731)  

**Abstract**: Organic reaction mechanisms are the stepwise elementary reactions by which reactants form intermediates and products, and are fundamental to understanding chemical reactivity and designing new molecules and reactions. Although large language models (LLMs) have shown promise in understanding chemical tasks such as synthesis design, it is unclear to what extent this reflects genuine chemical reasoning capabilities, i.e., the ability to generate valid intermediates, maintain chemical consistency, and follow logically coherent multi-step pathways. We address this by introducing oMeBench, the first large-scale, expert-curated benchmark for organic mechanism reasoning in organic chemistry. It comprises over 10,000 annotated mechanistic steps with intermediates, type labels, and difficulty ratings. Furthermore, to evaluate LLM capability more precisely and enable fine-grained scoring, we propose oMeS, a dynamic evaluation framework that combines step-level logic and chemical similarity. We analyze the performance of state-of-the-art LLMs, and our results show that although current models display promising chemical intuition, they struggle with correct and consistent multi-step reasoning. Notably, we find that using prompting strategy and fine-tuning a specialist model on our proposed dataset increases performance by 50% over the leading closed-source model. We hope that oMeBench will serve as a rigorous foundation for advancing AI systems toward genuine chemical reasoning. 

---
# LLM Unlearning Under the Microscope: A Full-Stack View on Methods and Metrics 

**Authors**: Chongyu Fan, Changsheng Wang, Yancheng Huang, Soumyadeep Pal, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07626)  

**Abstract**: Machine unlearning for large language models (LLMs) aims to remove undesired data, knowledge, and behaviors (e.g., for safety, privacy, or copyright) while preserving useful model capabilities. Despite rapid progress over the past two years, research in LLM unlearning remains fragmented, with limited clarity on what constitutes effective unlearning and how it should be rigorously evaluated. In this work, we present a principled taxonomy of twelve recent stateful unlearning methods, grouped into three methodological families: divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning. Building on this taxonomy, we revisit the evaluation of unlearning effectiveness (UE), utility retention (UT), and robustness (Rob), focusing on the WMDP benchmark. Our analysis shows that current evaluations, dominated by multiple-choice question (MCQ) accuracy, offer only a narrow perspective, often overstating success while overlooking the model's actual generation behavior. To address this gap, we introduce open question-answering (Open-QA) metrics that better capture generative performance and reveal the inherent UE-UT tradeoff across method families. Furthermore, we demonstrate that robustness requires finer-grained analysis: for example, vulnerabilities differ substantially between in-domain relearning and out-of-domain fine-tuning, even though both fall under model-level attacks. Through this study, we hope to deliver a full-stack revisit of LLM unlearning and actionable guidance for designing and evaluating future methods. 

---
# Evaluation of LLMs for Process Model Analysis and Optimization 

**Authors**: Akhil Kumar, Jianliang Leon Zhao, Om Dobariya  

**Link**: [PDF](https://arxiv.org/pdf/2510.07489)  

**Abstract**: In this paper, we report our experience with several LLMs for their ability to understand a process model in an interactive, conversational style, find syntactical and logical errors in it, and reason with it in depth through a natural language (NL) interface. Our findings show that a vanilla, untrained LLM like ChatGPT (model o3) in a zero-shot setting is effective in understanding BPMN process models from images and answering queries about them intelligently at syntactic, logic, and semantic levels of depth. Further, different LLMs vary in performance in terms of their accuracy and effectiveness. Nevertheless, our empirical analysis shows that LLMs can play a valuable role as assistants for business process designers and users. We also study the LLM's "thought process" and ability to perform deeper reasoning in the context of process analysis and optimization. We find that the LLMs seem to exhibit anthropomorphic properties. 

---
# CompassLLM: A Multi-Agent Approach toward Geo-Spatial Reasoning for Popular Path Query 

**Authors**: Md. Nazmul Islam Ananto, Shamit Fatin, Mohammed Eunus Ali, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2510.07516)  

**Abstract**: The popular path query - identifying the most frequented routes between locations from historical trajectory data - has important applications in urban planning, navigation optimization, and travel recommendations. While traditional algorithms and machine learning approaches have achieved success in this domain, they typically require model training, parameter tuning, and retraining when accommodating data updates. As Large Language Models (LLMs) demonstrate increasing capabilities in spatial and graph-based reasoning, there is growing interest in exploring how these models can be applied to geo-spatial problems.
We introduce CompassLLM, a novel multi-agent framework that intelligently leverages the reasoning capabilities of LLMs into the geo-spatial domain to solve the popular path query. CompassLLM employs its agents in a two-stage pipeline: the SEARCH stage that identifies popular paths, and a GENERATE stage that synthesizes novel paths in the absence of an existing one in the historical trajectory data. Experiments on real and synthetic datasets show that CompassLLM demonstrates superior accuracy in SEARCH and competitive performance in GENERATE while being cost-effective. 

---
# Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts 

**Authors**: Yeskendir Koishekenov, Aldo Lipani, Nicola Cancedda  

**Link**: [PDF](https://arxiv.org/pdf/2510.07358)  

**Abstract**: Most efforts to improve the reasoning capabilities of large language models (LLMs) involve either scaling the number of parameters and the size of training data, or scaling inference computation by letting models generate complex chains of thought. Motivated by interpretability studies showing that the crucial computation required for reasoning tasks is concentrated in a limited range of layers, we introduce Encode-Think-Decode (ETD), a method that enhances the reasoning capabilities of a base model by training it to iterate over a small subset of reasoning-relevant layers during the mid-training stage. ETD amplifies latent reasoning while preserving the original architecture, parameter count, hyperparameters, and training data composition. When iterating on the selected layers at inference time, ETD models yield substantial gains on 17 reasoning benchmarks, including +28.4% relative accuracy improvement on GSM8K and +36% on MATH with the OLMo-2 1B Base model. We also explore an adaptive depth strategy that adjusts the computation per input token. Our results show that recursive latent reasoning offers a simple and effective path to stronger LLM reasoning. 

---
# QAgent: A modular Search Agent with Interactive Query Understanding 

**Authors**: Yi Jiang, Lei Shen, Lujie Niu, Sendong Zhao, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.08383)  

**Abstract**: Large language models (LLMs) excel at natural language tasks but are limited by their static parametric knowledge, especially in knowledge-intensive task. Retrieval-augmented generation (RAG) mitigates this by integrating external information. However, (1) traditional RAG struggles with complex query understanding, and (2) even search agents trained with reinforcement learning (RL), despite their promise, still face generalization and deployment challenges. To address these limitations, we propose QAgent, a unified agentic RAG framework that employs a search agent for adaptive retrieval. This agent optimizes its understanding of the query through interactive reasoning and retrieval. To facilitate real-world application, we focus on modular search agent for query understanding that are plug-and-play in complex systems. Secifically, the agent follows a multi-step decision process trained with RL to maximize retrieval quality and support accurate downstream answers. We further analyze the strengths and weaknesses of end-to-end RL and propose a strategy that focuses on effective retrieval, thereby enhancing generalization in LLM applications. Experiments show QAgent excels at QA and serves as a plug-and-play module for real-world deployment. 

---
# Revisiting Hallucination Detection with Effective Rank-based Uncertainty 

**Authors**: Rui Wang, Zeming Wei, Guanzhang Yue, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.08389)  

**Abstract**: Detecting hallucinations in large language models (LLMs) remains a fundamental challenge for their trustworthy deployment. Going beyond basic uncertainty-driven hallucination detection frameworks, we propose a simple yet powerful method that quantifies uncertainty by measuring the effective rank of hidden states derived from multiple model outputs and different layers. Grounded in the spectral analysis of representations, our approach provides interpretable insights into the model's internal reasoning process through semantic variations, while requiring no extra knowledge or additional modules, thus offering a combination of theoretical elegance and practical efficiency. Meanwhile, we theoretically demonstrate the necessity of quantifying uncertainty both internally (representations of a single response) and externally (different responses), providing a justification for using representations among different layers and responses from LLMs to detect hallucinations. Extensive experiments demonstrate that our method effectively detects hallucinations and generalizes robustly across various scenarios, contributing to a new paradigm of hallucination detection for LLM truthfulness. 

---
# LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings 

**Authors**: Benjamin F. Maier, Ulf Aslak, Luca Fiaschi, Nina Rismal, Kemble Fletcher, Christian C. Luhmann, Robbie Dow, Kli Pappas, Thomas V. Wiecki  

**Link**: [PDF](https://arxiv.org/pdf/2510.08338)  

**Abstract**: Consumer research costs companies billions annually yet suffers from panel biases and limited scale. Large language models (LLMs) offer an alternative by simulating synthetic consumers, but produce unrealistic response distributions when asked directly for numerical ratings. We present semantic similarity rating (SSR), a method that elicits textual responses from LLMs and maps these to Likert distributions using embedding similarity to reference statements. Testing on an extensive dataset comprising 57 personal care product surveys conducted by a leading corporation in that market (9,300 human responses), SSR achieves 90% of human test-retest reliability while maintaining realistic response distributions (KS similarity > 0.85). Additionally, these synthetic respondents provide rich qualitative feedback explaining their ratings. This framework enables scalable consumer research simulations while preserving traditional survey metrics and interpretability. 

---
# First Try Matters: Revisiting the Role of Reflection in Reasoning Models 

**Authors**: Liwei Kang, Yue Deng, Yao Xiao, Zhanfeng Mo, Wee Sun Lee, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2510.08308)  

**Abstract**: Large language models have recently demonstrated significant gains in reasoning ability, often attributed to their capacity to generate longer chains of thought and engage in reflective reasoning. However, the contribution of reflections to performance improvement remains unclear. In this paper, we systematically analyze the rollouts of eight reasoning models on five mathematical datasets. We focus on reflective behaviours where the model has already produced an answer but continues reflecting before finalizing its output. Our analysis reveals that reflections are predominantly confirmatory and rarely alter the model's initial answer, a pattern consistent across models and datasets. To understand the role of reflections in training, we construct supervised fine-tuning (SFT) datasets with varying amounts of reflection steps. We observe that training models on rollouts with more reflection steps primarily enhances first-answer correctness rather than the ability to correct initially wrong answers through reflections. This motivates us to propose a question-aware early-stopping method that enhances inference-time token efficiency by stopping the reasoning process once a few plausible candidate answers are generated, thereby reducing unnecessary reflection steps. Motivated by this, we further propose to dynamically truncate the reflections after a candidate answer has appeared during generation, which reduces reasoning tokens by 24.5% across five mathematical datasets, within a 2.9% drop in accuracy. 

---
# Selection, Reflection and Self-Refinement: Revisit Reasoning Tasks via a Causal Lens 

**Authors**: Yunlong Deng, Boyang Sun, Yan Li, Lingjing Kong, Zeyu Tang, Kun Zhang, Guangyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08222)  

**Abstract**: Due to their inherent complexity, reasoning tasks have long been regarded as rigorous benchmarks for assessing the capabilities of machine learning models, especially large language models (LLMs). Although humans can solve these tasks with ease, existing models, even after extensive pre-training and post-training at scale, still fail to perform reasoning reliably. In this paper, we revisit reasoning tasks from a causal perspective, seeking to understand their behavior in latent space and to offer insights for addressing their challenges. Specifically, we cast reasoning tasks as a selection mechanism, in which high-level logical concepts function as selection operators on the given observations, such as, identifying the correct answer in a math problem or filling the appropriate entry in Sudoku. We emphasize two key properties of this formulation that shed light on the difficulty of reasoning tasks. First, the latent space exceeds the observation space in complexity, even when the correct answer is fully determined by the observed input. Second, the latent variables, corresponding to logical thought, are densely structured and exhibit strong dependencies. Building on this formulation, we introduce a framework, called SR$^2$, that incorporates the estimated latent variables as feedback into the selection mechanism, thereby facilitating the learning of dense dependencies among latent representations. The framework consists of three key modules: reflective representation learning, dependency self-refinement, and periodic intermediate alignment. Experimentally, we show that our approach yields significant gains in reasoning accuracy, for example, attaining over 10$\%$ improvement in performance with 8$\times$ fewer parameters on the Sudoku and Maze tasks over the recent advances. 

---
# Chain-of-Trigger: An Agentic Backdoor that Paradoxically Enhances Agentic Robustness 

**Authors**: Jiyang Qiu, Xinbei Ma, Yunqing Xu, Zhuosheng Zhang, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08238)  

**Abstract**: The rapid deployment of large language model (LLM)-based agents in real-world applications has raised serious concerns about their trustworthiness. In this work, we reveal the security and robustness vulnerabilities of these agents through backdoor attacks. Distinct from traditional backdoors limited to single-step control, we propose the Chain-of-Trigger Backdoor (CoTri), a multi-step backdoor attack designed for long-horizon agentic control. CoTri relies on an ordered sequence. It starts with an initial trigger, and subsequent ones are drawn from the environment, allowing multi-step manipulation that diverts the agent from its intended task. Experimental results show that CoTri achieves a near-perfect attack success rate (ASR) while maintaining a near-zero false trigger rate (FTR). Due to training data modeling the stochastic nature of the environment, the implantation of CoTri paradoxically enhances the agent's performance on benign tasks and even improves its robustness against environmental distractions. We further validate CoTri on vision-language models (VLMs), confirming its scalability to multimodal agents. Our work highlights that CoTri achieves stable, multi-step control within agents, improving their inherent robustness and task capabilities, which ultimately makes the attack more stealthy and raises potential safty risks. 

---
# LinguaSim: Interactive Multi-Vehicle Testing Scenario Generation via Natural Language Instruction Based on Large Language Models 

**Authors**: Qingyuan Shi, Qingwen Meng, Hao Cheng, Qing Xu, Jianqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08046)  

**Abstract**: The generation of testing and training scenarios for autonomous vehicles has drawn significant attention. While Large Language Models (LLMs) have enabled new scenario generation methods, current methods struggle to balance command adherence accuracy with the realism of real-world driving environments. To reduce scenario description complexity, these methods often compromise realism by limiting scenarios to 2D, or open-loop simulations where background vehicles follow predefined, non-interactive behaviors. We propose LinguaSim, an LLM-based framework that converts natural language into realistic, interactive 3D scenarios, ensuring both dynamic vehicle interactions and faithful alignment between the input descriptions and the generated scenarios. A feedback calibration module further refines the generation precision, improving fidelity to user intent. By bridging the gap between natural language and closed-loop, interactive simulations, LinguaSim constrains adversarial vehicle behaviors using both the scenario description and the autonomous driving model guiding them. This framework facilitates the creation of high-fidelity scenarios that enhance safety testing and training. Experiments show LinguaSim can generate scenarios with varying criticality aligned with different natural language descriptions (ACT: 0.072 s for dangerous vs. 3.532 s for safe descriptions; comfortability: 0.654 vs. 0.764), and its refinement module effectively reduces excessive aggressiveness in LinguaSim's initial outputs, lowering the crash rate from 46.9% to 6.3% to better match user intentions. 

---
# TaoSR-SHE: Stepwise Hybrid Examination Reinforcement Learning Framework for E-commerce Search Relevance 

**Authors**: Pengkun Jiao, Yiming Jin, Jianhui Yang, Chenhe Dong, Zerui Huang, Shaowei Yao, Xiaojiang Zhou, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07972)  

**Abstract**: Query-product relevance analysis is a foundational technology in e-commerce search engines and has become increasingly important in AI-driven e-commerce. The recent emergence of large language models (LLMs), particularly their chain-of-thought (CoT) reasoning capabilities, offers promising opportunities for developing relevance systems that are both more interpretable and more robust. However, existing training paradigms have notable limitations: SFT and DPO suffer from poor generalization on long-tail queries and from a lack of fine-grained, stepwise supervision to enforce rule-aligned reasoning. In contrast, reinforcement learning with verification rewards (RLVR) suffers from sparse feedback, which provides insufficient signal to correct erroneous intermediate steps, thereby undermining logical consistency and limiting performance in complex inference scenarios.
To address these challenges, we introduce the Stepwise Hybrid Examination Reinforcement Learning framework for Taobao Search Relevance (TaoSR-SHE). At its core is Stepwise Reward Policy Optimization (SRPO), a reinforcement learning algorithm that leverages step-level rewards generated by a hybrid of a high-quality generative stepwise reward model and a human-annotated offline verifier, prioritizing learning from critical correct and incorrect reasoning steps. TaoSR-SHE further incorporates two key techniques: diversified data filtering to encourage exploration across varied reasoning paths and mitigate policy entropy collapse, and multi-stage curriculum learning to foster progressive capability growth. Extensive experiments on real-world search benchmarks show that TaoSR-SHE improves both reasoning quality and relevance-prediction accuracy in large-scale e-commerce settings, outperforming SFT, DPO, GRPO, and other baselines, while also enhancing interpretability and robustness. 

---
# PEAR: Phase Entropy Aware Reward for Efficient Reasoning 

**Authors**: Chen Huang, Wei Lu, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08026)  

**Abstract**: Large Reasoning Models (LRMs) have achieved impressive performance on complex reasoning tasks by generating detailed chain-of-thought (CoT) explanations. However, these responses are often excessively long, containing redundant reasoning steps that inflate inference cost and reduce usability. Controlling the length of generated reasoning without sacrificing accuracy remains an open challenge. Through a systematic empirical analysis, we reveal a consistent positive correlation between model entropy and response length at different reasoning stages across diverse LRMs: the thinking phase exhibits higher entropy, reflecting exploratory behavior of longer responses, while the final answer phase shows lower entropy, indicating a more deterministic this http URL observation suggests that entropy at different reasoning stages can serve as a control knob for balancing conciseness and performance. Based on this insight, this paper introduces Phase Entropy Aware Reward (PEAR), a reward mechanism that incorporating phase-dependent entropy into the reward design. Instead of treating all tokens uniformly, PEAR penalize excessive entropy during the thinking phase and allowing moderate exploration at the final answer phase, which encourages models to generate concise reasoning traces that retain sufficient flexibility to solve the task correctly. This enables adaptive control of response length without relying on explicit length targets or rigid truncation rules. Extensive experiments across four benchmarks demonstrate that PEAR consistently reduces response length while sustaining competitive accuracy across model scales. In addition, PEAR demonstrates strong out-of-distribution (OOD) robustness beyond the training distribution. Our code is available at: this https URL. 

---
# Enabling Personalized Long-term Interactions in LLM-based Agents through Persistent Memory and User Profiles 

**Authors**: Rebecca Westhäußer, Wolfgang Minker, Sebatian Zepf  

**Link**: [PDF](https://arxiv.org/pdf/2510.07925)  

**Abstract**: Large language models (LLMs) increasingly serve as the central control unit of AI agents, yet current approaches remain limited in their ability to deliver personalized interactions. While Retrieval Augmented Generation enhances LLM capabilities by improving context-awareness, it lacks mechanisms to combine contextual information with user-specific data. Although personalization has been studied in fields such as human-computer interaction or cognitive science, existing perspectives largely remain conceptual, with limited focus on technical implementation. To address these gaps, we build on a unified definition of personalization as a conceptual foundation to derive technical requirements for adaptive, user-centered LLM-based agents. Combined with established agentic AI patterns such as multi-agent collaboration or multi-source retrieval, we present a framework that integrates persistent memory, dynamic coordination, self-validation, and evolving user profiles to enable personalized long-term interactions. We evaluate our approach on three public datasets using metrics such as retrieval accuracy, response correctness, or BertScore. We complement these results with a five-day pilot user study providing initial insights into user feedback on perceived personalization. The study provides early indications that guide future work and highlights the potential of integrating persistent memory and user profiles to improve the adaptivity and perceived personalization of LLM-based agents. 

---
# Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents 

**Authors**: Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07920)  

**Abstract**: LLM-based financial agents have attracted widespread excitement for their ability to trade like human experts. However, most systems exhibit a "profit mirage": dazzling back-tested returns evaporate once the model's knowledge window ends, because of the inherent information leakage in LLMs. In this paper, we systematically quantify this leakage issue across four dimensions and release FinLake-Bench, a leakage-robust evaluation benchmark. Furthermore, to mitigate this issue, we introduce FactFin, a framework that applies counterfactual perturbations to compel LLM-based agents to learn causal drivers instead of memorized outcomes. FactFin integrates four core components: Strategy Code Generator, Retrieval-Augmented Generation, Monte Carlo Tree Search, and Counterfactual Simulator. Extensive experiments show that our method surpasses all baselines in out-of-sample generalization, delivering superior risk-adjusted performance. 

---
# GCPO: When Contrast Fails, Go Gold 

**Authors**: Hao Wu, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07790)  

**Abstract**: Reinforcement learning has been widely applied to enhance the reasoning capabilities of large language models. Extending the inference limits of smaller models has become a prominent research focus. However, algorithms such as Group Relative Policy Optimization (GRPO) suffer from a clear drawback: the upper bound of a model's rollout responses is entirely determined by the model itself, preventing the acquisition of knowledge from samples that are either all incorrect or all correct. In this paper, we introduce Group Contrastive Policy Optimization (GCPO), a method that incorporates external standard reference answers. When the model cannot solve a problem, the reference answer supplies the correct response, steering the model toward an unequivocally accurate update direction. This approach offers two main advantages: (1) it improves training efficiency by fully utilizing every sample; (2) it enables the model to emulate the problem solving strategy of the reference answer during training, thereby enhancing generalization in reasoning. GCPO achieves outstanding results across multiple benchmark datasets, yielding substantial improvements over the baseline model. Our code is available at: this https URL. 

---
# Augur: Modeling Covariate Causal Associations in Time Series via Large Language Models 

**Authors**: Zhiqing Cui, Binwu Wang, Qingxiang Liu, Yeqiang Wang, Zhengyang Zhou, Yuxuan Liang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07858)  

**Abstract**: Large language models (LLM) have emerged as a promising avenue for time series forecasting, offering the potential to integrate multimodal data. However, existing LLM-based approaches face notable limitations-such as marginalized role in model architectures, reliance on coarse statistical text prompts, and lack of interpretability. In this work, we introduce Augur, a fully LLM driven time series forecasting framework that exploits LLM causal reasoning to discover and use directed causal associations among covariates. Augur uses a two stage teacher student architecture where a powerful teacher LLM infers a directed causal graph from time series using heuristic search together with pairwise causality testing. A lightweight student agent then refines the graph and fine tune on high confidence causal associations that are encoded as rich textual prompts to perform forecasting. This design improves predictive accuracy while yielding transparent, traceable reasoning about variable interactions. Extensive experiments on real-world datasets with 25 baselines demonstrate that Augur achieves competitive performance and robust zero-shot generalization. 

---
# An approach for systematic decomposition of complex llm tasks 

**Authors**: Tianle Zhou, Jiakai Xu, Guanhong Liu, Jiaxiang Liu, Haonan Wang, Eugene Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07772)  

**Abstract**: Large Language Models (LLMs) suffer from reliability issues on complex tasks, as existing decomposition methods are heuristic and rely on agent or manual decomposition. This work introduces a novel, systematic decomposition framework that we call Analysis of CONstraint-Induced Complexity (ACONIC), which models the task as a constraint problem and leveraging formal complexity measures to guide decomposition. On combinatorial (SATBench) and LLM database querying tasks (Spider), we find that by decomposing the tasks following the measure of complexity, agent can perform considerably better (10-40 percentage point). 

---
# Haibu Mathematical-Medical Intelligent Agent:Enhancing Large Language Model Reliability in Medical Tasks via Verifiable Reasoning Chains 

**Authors**: Yilun Zhang, Dexing Kong  

**Link**: [PDF](https://arxiv.org/pdf/2510.07748)  

**Abstract**: Large Language Models (LLMs) show promise in medicine but are prone to factual and logical errors, which is unacceptable in this high-stakes field. To address this, we introduce the "Haibu Mathematical-Medical Intelligent Agent" (MMIA), an LLM-driven architecture that ensures reliability through a formally verifiable reasoning process. MMIA recursively breaks down complex medical tasks into atomic, evidence-based steps. This entire reasoning chain is then automatically audited for logical coherence and evidence traceability, similar to theorem proving. A key innovation is MMIA's "bootstrapping" mode, which stores validated reasoning chains as "theorems." Subsequent tasks can then be efficiently solved using Retrieval-Augmented Generation (RAG), shifting from costly first-principles reasoning to a low-cost verification model. We validated MMIA across four healthcare administration domains, including DRG/DIP audits and medical insurance adjudication, using expert-validated benchmarks. Results showed MMIA achieved an error detection rate exceeding 98% with a false positive rate below 1%, significantly outperforming baseline LLMs. Furthermore, the RAG matching mode is projected to reduce average processing costs by approximately 85% as the knowledge base matures. In conclusion, MMIA's verifiable reasoning framework is a significant step toward creating trustworthy, transparent, and cost-effective AI systems, making LLM technology viable for critical applications in medicine. 

---
# An LLM-Powered Cooperative Framework for Large-Scale Multi-Vehicle Navigation 

**Authors**: Yuping Zhou, Siqi Lai, Jindong Han, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07825)  

**Abstract**: The rise of Internet of Vehicles (IoV) technologies is transforming traffic management from isolated control to a collective, multi-vehicle process. At the heart of this shift is multi-vehicle dynamic navigation, which requires simultaneously routing large fleets under evolving traffic conditions. Existing path search algorithms and reinforcement learning methods struggle to scale to city-wide networks, often failing to capture the nonlinear, stochastic, and coupled dynamics of urban traffic. To address these challenges, we propose CityNav, a hierarchical, LLM-powered framework for large-scale multi-vehicle navigation. CityNav integrates a global traffic allocation agent, which coordinates strategic traffic flow distribution across regions, with local navigation agents that generate locally adaptive routes aligned with global directives. To enable effective cooperation, we introduce a cooperative reasoning optimization mechanism, in which agents are jointly trained with a dual-reward structure: individual rewards promote per-vehicle efficiency, while shared rewards encourage network-wide coordination and congestion reduction. Extensive experiments on four real-world road networks of varying scales (up to 1.6 million roads and 430,000 intersections) and traffic datasets demonstrate that CityNav consistently outperforms nine classical path search and RL-based baselines in city-scale travel efficiency and congestion mitigation. Our results highlight the potential of LLMs to enable scalable, adaptive, and cooperative city-wide traffic navigation, providing a foundation for intelligent, large-scale vehicle routing in complex urban environments. Our project is available at this https URL. 

---
# From Noisy to Native: LLM-driven Graph Restoration for Test-Time Graph Domain Adaptation 

**Authors**: Xiangwei Lv, JinLuan Yang, Wang Lin, Jingyuan Chen, Beishui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2510.07762)  

**Abstract**: Graph domain adaptation (GDA) has achieved great attention due to its effectiveness in addressing the domain shift between train and test data. A significant bottleneck in existing graph domain adaptation methods is their reliance on source-domain data, which is often unavailable due to privacy or security concerns. This limitation has driven the development of Test-Time Graph Domain Adaptation (TT-GDA), which aims to transfer knowledge without accessing the source examples. Inspired by the generative power of large language models (LLMs), we introduce a novel framework that reframes TT-GDA as a generative graph restoration problem, "restoring the target graph to its pristine, source-domain-like state". There are two key challenges: (1) We need to construct a reasonable graph restoration process and design an effective encoding scheme that an LLM can understand, bridging the modality gap. (2) We need to devise a mechanism to ensure the restored graph acquires the intrinsic features of the source domain, even without access to the source data. To ensure the effectiveness of graph restoration, we propose GRAIL, that restores the target graph into a state that is well-aligned with the source domain. Specifically, we first compress the node representations into compact latent features and then use a graph diffusion process to model the graph restoration process. Then a quantization module encodes the restored features into discrete tokens. Building on this, an LLM is fine-tuned as a generative restorer to transform a "noisy" target graph into a "native" one. To further improve restoration quality, we introduce a reinforcement learning process guided by specialized alignment and confidence rewards. Extensive experiments demonstrate the effectiveness of our approach across various datasets. 

---
# SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation 

**Authors**: Minh-Anh Nguye, Minh-Duc Nguyen, Nguyen Thi Ha Lan, Kieu Hai Dang, Nguyen Tien Dong, Le Duy Dung  

**Link**: [PDF](https://arxiv.org/pdf/2510.07733)  

**Abstract**: Large language models (LLMs) are increasingly adopted for automating survey paper generation \cite{wang2406autosurvey, liang2025surveyx, yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing approaches typically extract content from a large collection of related papers and prompt LLMs to summarize them directly. However, such methods often overlook the structural relationships among papers, resulting in generated surveys that lack a coherent taxonomy and a deeper contextual understanding of research progress. To address these shortcomings, we propose \textbf{SurveyG}, an LLM-based agent framework that integrates \textit{hierarchical citation graph}, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness between their contents, thereby embedding structural and contextual knowledge into the survey generation process. The graph is organized into three layers: \textbf{Foundation}, \textbf{Development}, and \textbf{Frontier}, to capture the evolution of research from seminal works to incremental advances and emerging directions. By combining horizontal search within layers and vertical depth traversal across layers, the agent produces multi-level summaries, which are consolidated into a structured survey outline. A multi-agent validation stage then ensures consistency, coverage, and factual accuracy in generating the final survey. Experiments, including evaluations by human experts and LLM-as-a-judge, demonstrate that SurveyG outperforms state-of-the-art frameworks, producing surveys that are more comprehensive and better structured to the underlying knowledge taxonomy of a field. 

---
# AgentAsk: Multi-Agent Systems Need to Ask 

**Authors**: Bohan Lin, Kuo Yang, Yingchuan Lai, Yudong Zhang, Chen Zhang, Guibin Zhang, Xinlei Yu, Miao Yu, Xu Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07593)  

**Abstract**: Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving capabilities through collaborative division of labor. However, they frequently underperform single-agent baselines due to edge-level error cascades: minor inaccuracies at one message handoff propagate across the entire chain. We propose AgentAsk, a lightweight and plug-and-play clarification module that treats every inter-agent message as a potential failure point and inserts minimally necessary questions to arrest error propagation. AgentAsk follows a three-stage pipeline: (i) distilling edge-level judgments from curated failure traces into a compact policy, (ii) supervising the policy to determine when/what/whom/how to ask, and (iii) optimizing online with E-GRPO, a reinforcement learning objective that balances accuracy, latency, and cost. The module is architecture-agnostic and easy to integrate into existing orchestration. Across math, reasoning, and coding benchmarks, AgentAsk consistently improves accuracy and robustness over public multi-agent implementations while keeping overhead minimal, with latency and extra cost all less than 5%, approaching the performance of a strong evaluator. Beyond empirical improvements, we contribute a principled taxonomy of edge-level errors and a practical recipe for link-local intervention, offering a scalable pathway toward more reliable LLM-based multi-agent systems. 

---
# An Evaluation Study of Hybrid Methods for Multilingual PII Detection 

**Authors**: Harshit Rajgarhia, Suryam Gupta, Asif Shaik, Gulipalli Praveen Kumar, Y Santhoshraj, Sanka Nithya Tanvy Nishitha, Abhishek Mukherji  

**Link**: [PDF](https://arxiv.org/pdf/2510.07551)  

**Abstract**: The detection of Personally Identifiable Information (PII) is critical for privacy compliance but remains challenging in low-resource languages due to linguistic diversity and limited annotated data. We present RECAP, a hybrid framework that combines deterministic regular expressions with context-aware large language models (LLMs) for scalable PII detection across 13 low-resource locales. RECAP's modular design supports over 300 entity types without retraining, using a three-phase refinement pipeline for disambiguation and filtering. Benchmarked with nervaluate, our system outperforms fine-tuned NER models by 82% and zero-shot LLMs by 17% in weighted F1-score. This work offers a scalable and adaptable solution for efficient PII detection in compliance-focused applications. 

---
# Measuring and Mitigating Identity Bias in Multi-Agent Debate via Anonymization 

**Authors**: Hyeong Kyu Choi, Xiaojin Zhu, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.07517)  

**Abstract**: Multi-agent debate (MAD) aims to improve large language model (LLM) reasoning by letting multiple agents exchange answers and then aggregate their opinions. Yet recent studies reveal that agents are not neutral: they are prone to identity-driven sycophancy and self-bias, uncritically adopting a peer's view or stubbornly adhering to their own prior output, undermining the reliability of debate. In this work, we present the first principled framework that joins sycophancy and self-bias to mitigate and quantify identity bias in MAD. First, we formalize the debate dynamics as an identity-weighted Bayesian update process. Second, we propose response anonymization: by removing identity markers from prompts, agents cannot distinguish "self" from "peer", which forces equal weights on agent identity, thereby reducing bias. Third, we define the Identity Bias Coefficient (IBC), a principled metric that measures how often an agent follows a peer versus itself. Empirical studies across multiple models, datasets and debate rounds confirm that identity bias is widespread, with sycophancy far more common than self-bias. Our findings highlight the need to "mask" identity to ensure that MAD systems reason based on content rather than source identity. Code is released in this https URL. 

---
# TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering 

**Authors**: Penghang Liu, Elizabeth Fons, Svitlana Vyetrenko, Daniel Borrajo, Vamsi Potluru, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2510.07432)  

**Abstract**: Large language models (LLMs) have shown strong abilities in reasoning and problem solving, but recent studies reveal that they still struggle with time series reasoning tasks, where outputs are often affected by hallucination or knowledge leakage. In this work we propose TS-Agent, a time series reasoning agent that leverages LLMs strictly for what they excel at, i.e., gathering evidence and synthesizing it into conclusions through step-by-step reasoning, while delegating the extraction of statistical and structural information to time series analytical tools. Instead of mapping time series into text tokens, images, or embeddings, our agent interacts with raw numeric sequences through atomic operators, records outputs in an explicit evidence log, and iteratively refines its reasoning under the guidance of a self-critic and a final quality gate. This design avoids multi-modal alignment training, preserves the native form of time series, ensures interpretability and verifiability, and mitigates knowledge leakage or hallucination. Empirically, we evaluate the agent on established benchmarks. Our experiments show that TS-Agent achieves performance comparable to state-of-the-art LLMs on understanding benchmarks, and delivers significant improvements on reasoning tasks, where existing models often rely on memorization and fail in zero-shot settings. 

---
# ProSEA: Problem Solving via Exploration Agents 

**Authors**: William Nguyen, Vinh Luong, Christopher Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.07423)  

**Abstract**: Large language models (LLMs) have empowered AI agents to tackle increasingly complex tasks. However, most existing agents remain limited to static planning and brittle interactions, falling short of true collaboration or adaptive reasoning. We introduce ProSEA, a modular, general-purpose multi-agent framework designed for iterative problem solving through exploration and plan evolution. ProSEA features a hierarchical architecture in which a Manager Agent orchestrates domain-specialized Expert Agents, decomposes tasks, and adaptively replans based on structured feedback from failed attempts. Unlike prior systems, ProSEA agents report not only success or failure but also detailed reasons for failure and newly discovered constraints, enabling dynamic plan refinement informed by exploratory traces. The framework operates autonomously but supports seamless integration with human collaborators when needed. Experiments on the challenging FinanceBench benchmark demonstrate that ProSEA, even without human feedback, outperforms state-of-the-art baselines and achieves robust performance across reasoning-heavy tasks. These results underscore ProSEA's potential as a foundation for more transparent, adaptive, and human-aligned AI agents. 

---
# BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation 

**Authors**: Rocktim Jyoti Das, Harsh Singh, Diana Turmakhan, Muhammad Abdullah Sohail, Mingfei Han, Preslav Nakov, Fabio Pizzati, Ivan Laptev  

**Link**: [PDF](https://arxiv.org/pdf/2510.08572)  

**Abstract**: Scaling data and models has played a pivotal role in the remarkable progress of computer vision and language. Inspired by these domains, recent efforts in robotics have similarly focused on scaling both data and model size to develop more generalizable and robust policies. However, unlike vision and language, robotics lacks access to internet-scale demonstrations across diverse robotic tasks and environments. As a result, the scale of existing datasets typically suffers from the need for manual data collection and curation. To address this problem, here we propose BLAZER, a framework that learns manipulation policies from automatically generated training data. We build on the zero-shot capabilities of LLM planners and automatically generate demonstrations for diverse manipulation tasks in simulation. Successful examples are then used to finetune an LLM and to improve its planning capabilities without human supervision. Notably, while BLAZER training requires access to the simulator's state, we demonstrate direct transfer of acquired skills to sensor-based manipulation. Through extensive experiments, we show BLAZER to significantly improve zero-shot manipulation in both simulated and real environments. Moreover, BLAZER improves on tasks outside of its training pool and enables downscaling of LLM models. Our code and data will be made publicly available on the project page. 

---
# L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint) 

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07363)  

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure. 

---
# Base Models Know How to Reason, Thinking Models Learn When 

**Authors**: Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.07364)  

**Abstract**: Why do thinking language models like DeepSeek R1 outperform their base counterparts? Despite consistent performance gains, it remains unclear to what extent thinking models learn entirely new reasoning capabilities or repurpose pre-existing base model ones. In this work, we propose a hybrid model where we activate reasoning mechanisms in base models at the right time to elicit thinking-model-level reasoning chains, implying that thinking models exploit already existing capabilities. To ground our analysis, we introduce an unsupervised, bottom-up approach for uncovering human-interpretable reasoning behaviors in thinking models. This approach provides an unbiased method to discover reasoning behaviors without imposing manual or LLM-derived assumptions. Across three base and four thinking models, using GSM8K and MATH500, our hybrid model recovers up to 91% of the performance gap to thinking models without any weight updates while steering only 12% of tokens. Concretely, our empirical setup provides a simple, causal way to test the effectiveness of existing reasoning mechanisms in base models by invoking them directly and measuring the resulting task performance. More broadly, these results reframe our understanding of how thinking models are trained: pre-training is when models acquire most of their reasoning mechanisms, and post-training teaches efficient deployment of these mechanisms at the right time, enabling efficient use of their inference-time compute. 

---
# VideoNorms: Benchmarking Cultural Awareness of Video Language Models 

**Authors**: Nikhil Reddy Varimalla, Yunfei Xu, Arkadiy Saakyan, Meng Fan Wang, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2510.08543)  

**Abstract**: As Video Large Language Models (VideoLLMs) are deployed globally, they require understanding of and grounding in the relevant cultural background. To properly assess these models' cultural awareness, adequate benchmarks are needed. We introduce VideoNorms, a benchmark of over 1000 (video clip, norm) pairs from US and Chinese cultures annotated with socio-cultural norms grounded in speech act theory, norm adherence and violations labels, and verbal and non-verbal evidence. To build VideoNorms, we use a human-AI collaboration framework, where a teacher model using theoretically-grounded prompting provides candidate annotations and a set of trained human experts validate and correct the annotations. We benchmark a variety of open-weight VideoLLMs on the new dataset which highlight several common trends: 1) models performs worse on norm violation than adherence; 2) models perform worse w.r.t Chinese culture compared to the US culture; 3) models have more difficulty in providing non-verbal evidence compared to verbal for the norm adhere/violation label and struggle to identify the exact norm corresponding to a speech-act; and 4) unlike humans, models perform worse in formal, non-humorous contexts. Our findings emphasize the need for culturally-grounded video language model training - a gap our benchmark and framework begin to address. 

---
# Iterated Agent for Symbolic Regression 

**Authors**: Zhuo-Yang Song, Zeyu Cai, Shutao Zhang, Jiashen Wei, Jichen Pan, Shi Qiu, Qing-Hong Cao, Tie-Jiun Hou, Xiaohui Liu, Ming-xing Luo, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08317)  

**Abstract**: Symbolic regression (SR), the automated discovery of mathematical expressions from data, is a cornerstone of scientific inquiry. However, it is often hindered by the combinatorial explosion of the search space and a tendency to overfit. Popular methods, rooted in genetic programming, explore this space syntactically, often yielding overly complex, uninterpretable models. This paper introduces IdeaSearchFitter, a framework that employs Large Language Models (LLMs) as semantic operators within an evolutionary search. By generating candidate expressions guided by natural-language rationales, our method biases discovery towards models that are not only accurate but also conceptually coherent and interpretable. We demonstrate IdeaSearchFitter's efficacy across diverse challenges: it achieves competitive, noise-robust performance on the Feynman Symbolic Regression Database (FSReD), outperforming several strong baselines; discovers mechanistically aligned models with good accuracy-complexity trade-offs on real-world data; and derives compact, physically-motivated parametrizations for Parton Distribution Functions in a frontier high-energy physics application. IdeaSearchFitter is a specialized module within our broader iterated agent framework, IdeaSearch, which is publicly available at this https URL. 

---
# Think Just Enough: Sequence-Level Entropy as a Confidence Signal for LLM Reasoning 

**Authors**: Aman Sharma, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2510.08146)  

**Abstract**: We introduce a simple, yet novel entropy-based framework to drive token efficiency in large language models during reasoning tasks. Our approach uses Shannon entropy from token-level logprobs as a confidence signal to enable early stopping, achieving 25-50% computational savings while maintaining task accuracy. Crucially, we demonstrate that entropy-based confidence calibration represents an emergent property of advanced post-training optimization present in modern reasoning models but notably absent in standard instruction-tuned and pre-trained models (Llama 3.3 70B). We show that the entropy threshold to stop reasoning varies from model to model but can be calculated easily in one shot using only a few examples from existing reasoning datasets. Our results indicate that advanced reasoning models often know that they've gotten a correct answer early on, and that this emergent confidence awareness can be exploited to save tokens and reduce latency. The framework demonstrates consistent performance across reasoning-optimized model families with 25-50% computational cost reduction while preserving accuracy, revealing that confidence mechanisms represent a distinguishing characteristic of modern post-trained reasoning systems versus their predecessors. 

---
# An Adaptive Multi Agent Bitcoin Trading System 

**Authors**: Aadi Singhi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08068)  

**Abstract**: This paper presents a Multi Agent Bitcoin Trading system that utilizes Large Lan- guage Models (LLMs) for alpha generation and portfolio management in the cryptocur- rencies market. Unlike equities, cryptocurrencies exhibit extreme volatility and are heavily influenced by rapidly shifting market sentiments and regulatory announcements, making them difficult to model using static regression models or neural networks trained solely on historical data [53]. The proposed framework overcomes this by structuring LLMs into specialised agents for technical analysis, sentiment evaluation, decision-making, and performance reflection. The system improves over time through a novel verbal feedback mechanism where a Reflect agent provides daily and weekly natural-language critiques of trading decisions. These textual evaluations are then injected into future prompts, al- lowing the system to adjust indicator priorities, sentiment weights, and allocation logic without parameter updates or finetuning. Back-testing on Bitcoin price data from July 2024 to April 2025 shows consistent outperformance across market regimes: the Quantita- tive agent delivered over 30% higher returns in bullish phases and 15% overall gains versus buy-and-hold, while the sentiment-driven agent turned sideways markets from a small loss into a gain of over 100%. Adding weekly feedback further improved total performance by 31% and reduced bearish losses by 10%. The results demonstrate that verbal feedback represents a new, scalable, and low-cost method of tuning LLMs for financial goals. 

---
# Towards Reliable LLM-based Robot Planning via Combined Uncertainty Estimation 

**Authors**: Shiyuan Yin, Chenjia Bai, Zihao Zhang, Junwei Jin, Xinxin Zhang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08044)  

**Abstract**: Large language models (LLMs) demonstrate advanced reasoning abilities, enabling robots to understand natural language instructions and generate high-level plans with appropriate grounding. However, LLM hallucinations present a significant challenge, often leading to overconfident yet potentially misaligned or unsafe plans. While researchers have explored uncertainty estimation to improve the reliability of LLM-based planning, existing studies have not sufficiently differentiated between epistemic and intrinsic uncertainty, limiting the effectiveness of uncertainty esti- mation. In this paper, we present Combined Uncertainty estimation for Reliable Embodied planning (CURE), which decomposes the uncertainty into epistemic and intrinsic uncertainty, each estimated separately. Furthermore, epistemic uncertainty is subdivided into task clarity and task familiarity for more accurate evaluation. The overall uncertainty assessments are obtained using random network distillation and multi-layer perceptron regression heads driven by LLM features. We validated our approach in two distinct experimental settings: kitchen manipulation and tabletop rearrangement experiments. The results show that, compared to existing methods, our approach yields uncertainty estimates that are more closely aligned with the actual execution outcomes. 

---
# Past, Present, and Future of Bug Tracking in the Generative AI Era 

**Authors**: Utku Boran Torun, Mehmet Taha Demircan, Mahmut Furkan Gön, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2510.08005)  

**Abstract**: Traditional bug tracking systems rely heavily on manual reporting, reproduction, triaging, and resolution, each carried out by different stakeholders such as end users, customer support, developers, and testers. This division of responsibilities requires significant coordination and widens the communication gap between non-technical users and technical teams, slowing the process from bug discovery to resolution. Moreover, current systems are highly asynchronous; users often wait hours or days for a first response, delaying fixes and contributing to frustration. This paper examines the evolution of bug tracking, from early paper-based reporting to today's web-based and SaaS platforms. Building on this trajectory, we propose an AI-powered bug tracking framework that augments existing tools with intelligent, large language model (LLM)-driven automation. Our framework addresses two main challenges: reducing time-to-fix and minimizing human overhead. Users report issues in natural language, while AI agents refine reports, attempt reproduction, and request missing details. Reports are then classified, invalid ones resolved through no-code fixes, and valid ones localized and assigned to developers. LLMs also generate candidate patches, with human oversight ensuring correctness. By integrating automation into each phase, our framework accelerates response times, improves collaboration, and strengthens software maintenance practices for a more efficient, user-centric future. 

---
# Learning on the Job: An Experience-Driven Self-Evolving Agent for Long-Horizon Tasks 

**Authors**: Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Botian Shi, Yu Qiao, Haifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08002)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities across diverse domains, yet significant challenges persist when deploying them as AI agents for real-world long-horizon tasks. Existing LLM agents suffer from a critical limitation: they are test-time static and cannot learn from experience, lacking the ability to accumulate knowledge and continuously improve on the job. To address this challenge, we propose MUSE, a novel agent framework that introduces an experience-driven, self-evolving system centered around a hierarchical Memory Module. MUSE organizes diverse levels of experience and leverages them to plan and execute long-horizon tasks across multiple applications. After each sub-task execution, the agent autonomously reflects on its trajectory, converting the raw trajectory into structured experience and integrating it back into the Memory Module. This mechanism enables the agent to evolve beyond its static pretrained parameters, fostering continuous learning and self-evolution. We evaluate MUSE on the long-horizon productivity benchmark TAC. It achieves new SOTA performance by a significant margin using only a lightweight Gemini-2.5 Flash model. Sufficient Experiments demonstrate that as the agent autonomously accumulates experience, it exhibits increasingly superior task completion capabilities, as well as robust continuous learning and self-evolution capabilities. Moreover, the accumulated experience from MUSE exhibits strong generalization properties, enabling zero-shot improvement on new tasks. MUSE establishes a new paradigm for AI agents capable of real-world productivity task automation. 

---
# Fewer Weights, More Problems: A Practical Attack on LLM Pruning 

**Authors**: Kazuki Egashira, Robin Staab, Thibaud Gloaguen, Mark Vero, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2510.07985)  

**Abstract**: Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression. 

---
# SIMU: Selective Influence Machine Unlearning 

**Authors**: Anu Agarwal, Mihir Pamnani, Dilek Hakkani-Tur  

**Link**: [PDF](https://arxiv.org/pdf/2510.07822)  

**Abstract**: The undesired memorization of sensitive information by Large Language Models (LLMs) has emphasized the need for safety mechanisms that can regulate model behavior. This has led to the development of machine unlearning techniques that enable models to precisely forget sensitive and unwanted information. For machine unlearning, first-order and second-order optimizer-based methods have shown significant progress in enabling LLMs to forget targeted information. However, in doing so, these approaches often compromise the model's original capabilities, resulting in unlearned models that struggle to retain their prior knowledge and overall utility. To address this, we propose Selective Influence Machine Unlearning (SIMU), a two-step framework that enhances second-order optimizer-based unlearning by selectively updating only the critical neurons responsible for encoding the forget-set. By constraining updates to these targeted neurons, SIMU achieves comparable unlearning efficacy while substantially outperforming current methods in retaining the model's original knowledge. 

---
# AppForge: From Assistant to Independent Developer - Are GPTs Ready for Software Development? 

**Authors**: Dezhi Ran, Yuan Cao, Mengzhou Wu, Simin Chen, Yuzhe Guo, Jun Ren, Zihe Song, Hao Yu, Jialei Wei, Linyi Li, Wei Yang, Baishakhi Ray, Tao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.07740)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capability in function-level code generation tasks. Unlike isolated functions, real-world applications demand reasoning over the entire software system: developers must orchestrate how different components interact, maintain consistency across states over time, and ensure the application behaves correctly within the lifecycle and framework constraints. Yet, no existing benchmark adequately evaluates whether LLMs can bridge this gap and construct entire software systems from scratch. To address this gap, we propose APPFORGE, a benchmark consisting of 101 software development problems drawn from real-world Android apps. Given a natural language specification detailing the app functionality, a language model is tasked with implementing the functionality into an Android app from scratch. Developing an Android app from scratch requires understanding and coordinating app states, lifecycle management, and asynchronous operations, calling for LLMs to generate context-aware, robust, and maintainable code. To construct APPFORGE, we design a multi-agent system to automatically summarize the main functionalities from app documents and navigate the app to synthesize test cases validating the functional correctness of app implementation. Following rigorous manual verification by Android development experts, APPFORGE incorporates the test cases within an automated evaluation framework that enables reproducible assessment without human intervention, making it easily adoptable for future research. Our evaluation on 12 flagship LLMs show that all evaluated models achieve low effectiveness, with the best-performing model (GPT-5) developing only 18.8% functionally correct applications, highlighting fundamental limitations in current models' ability to handle complex, multi-component software engineering challenges. 

---
# Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs 

**Authors**: Man Hu, Xinyi Wu, Zuofeng Suo, Jinbo Feng, Linghui Meng, Yanhao Jia, Anh Tuan Luu, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.07697)  

**Abstract**: With the rise of advanced reasoning capabilities, large language models (LLMs) are receiving increasing attention. However, although reasoning improves LLMs' performance on downstream tasks, it also introduces new security risks, as adversaries can exploit these capabilities to conduct backdoor attacks. Existing surveys on backdoor attacks and reasoning security offer comprehensive overviews but lack in-depth analysis of backdoor attacks and defenses targeting LLMs' reasoning abilities. In this paper, we take the first step toward providing a comprehensive review of reasoning-based backdoor attacks in LLMs by analyzing their underlying mechanisms, methodological frameworks, and unresolved challenges. Specifically, we introduce a new taxonomy that offers a unified perspective for summarizing existing approaches, categorizing reasoning-based backdoor attacks into associative, passive, and active. We also present defense strategies against such attacks and discuss current challenges alongside potential directions for future research. This work offers a novel perspective, paving the way for further exploration of secure and trustworthy LLM communities. 

---
# MLLM4TS: Leveraging Vision and Multimodal Language Models for General Time-Series Analysis 

**Authors**: Qinghua Liu, Sam Heshmati, Zheda Mai, Zubin Abraham, John Paparrizos, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.07513)  

**Abstract**: Effective analysis of time series data presents significant challenges due to the complex temporal dependencies and cross-channel interactions in multivariate data. Inspired by the way human analysts visually inspect time series to uncover hidden patterns, we ask: can incorporating visual representations enhance automated time-series analysis? Recent advances in multimodal large language models have demonstrated impressive generalization and visual understanding capability, yet their application to time series remains constrained by the modality gap between continuous numerical data and discrete natural language. To bridge this gap, we introduce MLLM4TS, a novel framework that leverages multimodal large language models for general time-series analysis by integrating a dedicated vision branch. Each time-series channel is rendered as a horizontally stacked color-coded line plot in one composite image to capture spatial dependencies across channels, and a temporal-aware visual patch alignment strategy then aligns visual patches with their corresponding time segments. MLLM4TS fuses fine-grained temporal details from the numerical data with global contextual information derived from the visual representation, providing a unified foundation for multimodal time-series analysis. Extensive experiments on standard benchmarks demonstrate the effectiveness of MLLM4TS across both predictive tasks (e.g., classification) and generative tasks (e.g., anomaly detection and forecasting). These results underscore the potential of integrating visual modalities with pretrained language models to achieve robust and generalizable time-series analysis. 

---
# Label Semantics for Robust Hyperspectral Image Classification 

**Authors**: Rafin Hassan, Zarin Tasnim Roshni, Rafiqul Bari, Alimul Islam, Nabeel Mohammed, Moshiur Farazi, Shafin Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2510.07556)  

**Abstract**: Hyperspectral imaging (HSI) classification is a critical tool with widespread applications across diverse fields such as agriculture, environmental monitoring, medicine, and materials science. Due to the limited availability of high-quality training samples and the high dimensionality of spectral data, HSI classification models are prone to overfitting and often face challenges in balancing accuracy and computational complexity. Furthermore, most of HSI classification models are monomodal, where it solely relies on spectral-spatial data to learn decision boundaries in the high dimensional embedding space. To address this, we propose a general-purpose Semantic Spectral-Spatial Fusion Network (S3FN) that uses contextual, class specific textual descriptions to complement the training of an HSI classification model. Specifically, S3FN leverages LLMs to generate comprehensive textual descriptions for each class label that captures their unique characteristics and spectral behaviors. These descriptions are then embedded into a vector space using a pre-trained text encoder such as BERT or RoBERTa to extract meaningful label semantics which in turn leads to a better feature-label alignment for improved classification performance. To demonstrate the effectiveness of our approach, we evaluate our model on three diverse HSI benchmark datasets - Hyperspectral Wood, HyperspectralBlueberries, and DeepHS-Fruit and report significant performance boost. Our results highlight the synergy between textual semantics and spectral-spatial data, paving the way for further advancements in semantically augmented HSI classification models. Codes are be available in: this https URL 

---
# Generation and annotation of item usage scenarios in e-commerce using large language models 

**Authors**: Madoka Hagiri, Kazushi Okamoto, Koki Karube, Kei Harada, Atsushi Shibata  

**Link**: [PDF](https://arxiv.org/pdf/2510.07885)  

**Abstract**: Complementary recommendations suggest combinations of useful items that play important roles in e-commerce. However, complementary relationships are often subjective and vary among individuals, making them difficult to infer from historical data. Unlike conventional history-based methods that rely on statistical co-occurrence, we focus on the underlying usage context that motivates item combinations. We hypothesized that people select complementary items by imagining specific usage scenarios and identifying the needs in such situations. Based on this idea, we explored the use of large language models (LLMs) to generate item usage scenarios as a starting point for constructing complementary recommendation systems. First, we evaluated the plausibility of LLM-generated scenarios through manual annotation. The results demonstrated that approximately 85% of the generated scenarios were determined to be plausible, suggesting that LLMs can effectively generate realistic item usage scenarios. 

---
# PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations 

**Authors**: Ruining He, Lukasz Heldt, Lichan Hong, Raghunandan Keshavan, Shifan Mao, Nikhil Mehta, Zhengyang Su, Alicia Tsai, Yueqi Wang, Shao-Chuan Wang, Xinyang Yi, Lexi Baugher, Baykal Cakici, Ed Chi, Cristos Goodrow, Ningren Han, He Ma, Romer Rosales, Abby Van Soest, Devansh Tandon, Su-Lin Wu, Weilong Yang, Yilin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07784)  

**Abstract**: Large Language Models (LLMs) pose a new paradigm of modeling and computation for information tasks. Recommendation systems are a critical application domain poised to benefit significantly from the sequence modeling capabilities and world knowledge inherent in these large models. In this paper, we introduce PLUM, a framework designed to adapt pre-trained LLMs for industry-scale recommendation tasks. PLUM consists of item tokenization using Semantic IDs, continued pre-training (CPT) on domain-specific data, and task-specific fine-tuning for recommendation objectives. For fine-tuning, we focus particularly on generative retrieval, where the model is directly trained to generate Semantic IDs of recommended items based on user context. We conduct comprehensive experiments on large-scale internal video recommendation datasets. Our results demonstrate that PLUM achieves substantial improvements for retrieval compared to a heavily-optimized production model built with large embedding tables. We also present a scaling study for the model's retrieval performance, our learnings about CPT, a few enhancements to Semantic IDs, along with an overview of the training and inference methods that enable launching this framework to billions of users in YouTube. 

---
# Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs 

**Authors**: Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07484)  

**Abstract**: Reasoning over structured graphs remains a fundamental challenge for Large Language Models (LLMs), particularly when scaling to large graphs. Existing approaches typically follow the retrieval-augmented generation (RAG) paradigm: first retrieving subgraphs relevant to the query and then generating answers conditioned on the retrieved subgraphs. However, such two-phase pipelines often struggle to faithfully incorporate graph structure, since the generation process is ultimately constrained by the quality and completeness of the retrieved subgraph. Although many advanced retrievers have been proposed recently to mitigate this issue, they are usually tailored to the training graphs and generalize poorly to unseen graphs, which limits their practical applicability. In this work, we propose Reasoning by Exploration (RoE), a novel approach that unifies retrieval and generation by framing reasoning over graphs as a process of graph exploration. At each step, the LLM selects candidate nodes and edges to explore, gradually constructing reasoning paths and generating answers along the way. To enable effective exploration, RoE is trained in two stages: supervised fine-tuning (SFT) on gold reasoning paths, followed by reinforcement learning (RL) to enhance exploration effectiveness and generalization. Experiments on benchmark datasets demonstrate that RoE achieves substantial overall improvements over baselines, while also generalizing effectively to unseen graphs. 

---
# HySim-LLM: Embedding-Weighted Fine-Tuning Bounds and Manifold Denoising for Domain-Adapted LLMs 

**Authors**: Majid Jaberi-Douraki, Hossein Sholehrasa, Xuan Xu, Remya Ampadi Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2510.07796)  

**Abstract**: The extraction and standardization of pharmacokinetic (PK) information from scientific literature remain significant challenges in computational pharmacology, which limits the reliability of data-driven models in drug development. Large language models (LLMs) have achieved remarkable progress in text understanding and reasoning, yet their adaptation to structured biomedical data, such as PK tables, remains constrained by heterogeneity, noise, and domain shift. To address these limitations, we propose HySim-LLM, a unified mathematical and computational framework that integrates embedding-weighted fine-tuning and manifold-aware denoising to enhance the robustness and interpretability of LLMs. We establish two theoretical results: (1) a similarity-weighted generalization bound that quantifies adaptation performance under embedding divergence, and (2) a manifold-based denoising guarantee that bounds loss contributions from noisy or off-manifold samples. These theorems provide a principled foundation for fine-tuning LLMs in structured biomedical settings. The framework offers a mathematically grounded pathway toward reliable and interpretable LLM adaptation for biomedical and data-intensive scientific domains. 

---
